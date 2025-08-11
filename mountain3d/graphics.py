import typing as tp
from typing import List, Optional, Tuple

import geopy.distance
import numpy as np
import plotly.graph_objects as go
from stl.mesh import Mesh

from mountain3d.map_generator import CachedMapGenerator
from mountain3d.math import create_dome_slice, dome2layers, layers2inner_outer

from .const import ALARM_HEIGHT_COLOR_MAPPING, FIRE_HEIGHT_COLOR_MAPPING


def stl2mesh3d_params(stl_mesh):
    """
    stl_mesh is read by nympy-stl from a stl file;
    it is an array of faces/triangles (i.e. three 3d points);
    this function extracts the unique vertices and the lists I, J, K
    to define a Plotly mesh3d
    """
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain
    # multiple copies of the same vertex;
    # so extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(
        stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
    )
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    x, y, z = vertices.T
    return x, y, z, i, j, k


def load_stl_image(fname, fhandle, center_pos, center_height, scale_coefs):
    stl_mesh = Mesh.from_file(fname, fh=fhandle)
    x, y, z, i, j, k = stl2mesh3d_params(stl_mesh)

    x /= scale_coefs[0] * 100
    y /= scale_coefs[1] * 100
    z /= scale_coefs[2] * 100

    x += center_pos[0]
    y += center_pos[1]
    z += center_height
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="black")


def load_heightmaps(
    heightmap_paths: str,
    lat: float,
    long: float,
    km_side_of_square: int,
    out_shape: Tuple[int, int],
    cache_path: Optional[str] = None,
):
    map_generator = CachedMapGenerator(heightmap_paths, cache_path)
    raster, latitudes, longitudes, req_bounds_np = map_generator.get_map(
        lat=lat, long=long, km_side_of_square=km_side_of_square, out_shape=out_shape
    )

    size_x = geopy.distance.geodesic(
        req_bounds_np[0, :], req_bounds_np[((1, 0), (0, 1))]
    ).meters  # tl-tr
    size_y = geopy.distance.geodesic(
        req_bounds_np[0, :], req_bounds_np[((0, 1), (0, 1))]
    ).meters  # tl-bl

    return raster, (size_x, size_y), (latitudes, longitudes)


def draw_landscape(
    heightmap: np.ndarray,
    fig: go.Figure,
    size: Tuple[float, float],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> go.Figure:
    xs, ys = np.meshgrid(
        np.linspace(0, size[0], heightmap.shape[0]),
        np.linspace(0, size[1], heightmap.shape[1]),
    )

    text = np.array(
        [
            [
                f"\
широта: {latitudes[i,j]:.3f}\
<br>долгота: {longitudes[i,j]:.3f}\
<br>высота:  {heightmap[i,j]}"
                for i in range(heightmap.shape[1])
            ]
            for j in range(heightmap.shape[0])
        ]
    )

    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=heightmap,
            colorscale=[
                (0, "blue"),
                (0.002, "rgb(0,70,0)"),
                (0.33, "rgb(138,130,81)"),
                (0.67, "rgb(102,51,0)"),
                (1.0, "white"),
            ],
            cmin=-10,
            cmax=4500,
            hoverinfo="text",
            text=text,
        )
    )

    return fig


def draw_detection_dome_slices(
    layers_dots: tp.List[np.ndarray],
    dome_heights: np.ndarray,
    fig: go.Figure,
    arma_type: str,
    slice_height_2d: int,
) -> go.Figure:
    layers = dome2layers(layers_dots, dome_heights)
    inner_detection_surface_layers, outer_detection_surface_layers = layers2inner_outer(
        layers
    )
    color_mapper = (
        ALARM_HEIGHT_COLOR_MAPPING
        if arma_type == "alarm"
        else FIRE_HEIGHT_COLOR_MAPPING
    )
    line_color = color_mapper(slice_height_2d)

    for dome_layers in [inner_detection_surface_layers, outer_detection_surface_layers]:
        dome_layers_heights = dome_layers[:, 2, 0]
        if slice_height_2d > max(dome_layers_heights) or slice_height_2d < min(
            dome_layers_heights
        ):
            continue

        xs, ys, zs = create_dome_slice(slice_height_2d, dome_layers)

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                line=dict(color=line_color, width=6),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def draw_detection_dome(
    layers_dots: tp.List[np.ndarray],
    heights: np.ndarray,
    fig: go.Figure,
    color: str,
) -> go.Figure:
    dots_per_layer = layers_dots[0].shape[1]
    total_dots = dots_per_layer * len(heights)

    zs = np.repeat(np.fromiter(heights, float), dots_per_layer)
    xs, ys = np.concatenate(layers_dots, axis=-1)

    i = np.empty((total_dots - dots_per_layer) * 2)
    i[0::2] = np.arange(total_dots - dots_per_layer)
    i[1::2] = np.arange(dots_per_layer, total_dots)

    j = np.roll(i, 1)
    k = np.roll(i, 2)

    fig.add_mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i.tolist(),
        j=j.tolist(),
        k=k.tolist(),
        color=color,
        opacity=0.2,
        hoverinfo="skip",
    )

    return fig


def create_figure(
    heightmap_paths: str,
    lat: float,
    long: float,
    size_km: int,
    map_size_px: Tuple[int, int],
    ratio: Tuple[float, float, float],
    height: Optional[int] = None,
    width: Optional[int] = None,
    mode_2d: bool = False,
    cahce_path: Optional[str] = None,
):
    fig = go.Figure()
    heightmap, map_size_m, (latitudes, longitudes) = load_heightmaps(
        heightmap_paths, lat, long, size_km, map_size_px, cache_path=cahce_path
    )

    map_height_m = heightmap.max() - heightmap.min()

    scale_coefs = np.array(ratio) / np.array([*map_size_m, map_height_m])

    spacing: np.ndarray = np.array(map_size_m) / np.array(heightmap.shape)

    graph_size = {"height": height, "width": width}

    fig = draw_landscape(heightmap.T, fig, map_size_m, latitudes, longitudes)

    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode="pan" if mode_2d else None,
        ),
        scene_camera=dict(eye=dict(x=0.0, y=0.0, z=2.5)),
        **{k: v for k, v in graph_size.items() if v},
    )

    return fig, heightmap, scale_coefs, spacing


def draw_alarm_fire_stl_model(
    fig: go.Figure,
    image_file,
    center: List,
    scale_coefs: List,
) -> go.Figure:
    def stl2mesh3d(stl_mesh):
        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        vertices, ixr = np.unique(
            stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
        )
        i = np.take(ixr, [3 * k for k in range(p)])
        j = np.take(ixr, [3 * k + 1 for k in range(p)])
        k = np.take(ixr, [3 * k + 2 for k in range(p)])
        return vertices, i, j, k

    model = Mesh.from_file(image_file.name, fh=image_file)
    vertices, i, j, k = stl2mesh3d(model)
    x, y, z = vertices.T

    x /= scale_coefs[0] * 3000
    y /= scale_coefs[1] * 3000
    z /= scale_coefs[2] * 3000
    x += center[0]
    y += center[1]
    z += center[2]
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="rgb(0, 0, 0)",
        flatshading=True,
        showscale=False,
        hoverinfo="skip",
    )
    fig.add_traces([mesh])

    return fig
