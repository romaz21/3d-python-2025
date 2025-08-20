import typing as tp
from typing import List, Optional, Tuple

import geopy.distance
import numpy as np
import plotly.graph_objects as go
from stl.mesh import Mesh

from shapely.geometry import Polygon

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
    lod_threshold: int = 1_000_000
) -> go.Figure:
    # if heightmap.size > lod_threshold:
    #     heightmap = heightmap[::2, ::2]
    #     latitudes = latitudes[::2, ::2]
    #     longitudes = longitudes[::2, ::2]

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
    print("Drawing stl mode3...")
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
            # lighting=dict(
            #     ambient=0.8,
            #     diffuse=0.6,
            #     fresnel=0.1,
            #     specular=0.1
            # )
        )
    )

    return fig


def draw_detection_dome_slices(
    layers_dots: tp.List[np.ndarray],
    dome_heights: np.ndarray,
    fig: go.Figure,
    arma_type: str,
    slice_height_2d: int,
    all_polygons: list  # Добавляем новый параметр для сбора полигонов
) -> tp.Optional[Polygon]:  # Возвращаем полигон вместо фигуры
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

    polygons = []
    for dome_layers in [inner_detection_surface_layers, outer_detection_surface_layers]:
        dome_layers_heights = dome_layers[:, 2, 0]
        if slice_height_2d > max(dome_layers_heights) or slice_height_2d < min(
            dome_layers_heights
        ):
            continue

        xs, ys, zs = create_dome_slice(slice_height_2d, dome_layers)
        
        # Создаем полигон и добавляем в список
        if len(xs) > 2:  # Нужно минимум 3 точки для полигона
            polygon = Polygon(zip(xs, ys))
            polygons.append(polygon)
    
    # Объединяем внутренний и внешний полигоны в один (если есть оба)
    zone_polygon = None
    if polygons:
        if len(polygons) == 2:
            # Внешний полигон минус внутренний (создаем кольцо)
            zone_polygon = polygons[1].difference(polygons[0])
        else:
            zone_polygon = polygons[0]
    
    # Добавляем полигон в общий список
    if zone_polygon:
        all_polygons.append(zone_polygon)
    
    return None  # Больше не добавляем следы напрямую


def draw_detection_dome(
    layers_dots: tp.List[np.ndarray],
    heights: np.ndarray,
    fig: go.Figure,
    color: str,
) -> go.Figure:
    """
    # Преобразуем данные в формат для построения поверхности
    dots_per_layer = layers_dots[0].shape[1]
    num_layers = len(heights)
    
    # Создаем массивы координат для поверхности
    x_surface = np.zeros((num_layers, dots_per_layer))
    y_surface = np.zeros((num_layers, dots_per_layer))
    z_surface = np.zeros((num_layers, dots_per_layer))
    
    # Заполняем координаты
    for i in range(num_layers):
        x_surface[i, :] = layers_dots[i][0]
        y_surface[i, :] = layers_dots[i][1]
        z_surface[i, :] = heights[i]
    
    # Добавляем первую точку в конец для замыкания контура
    x_surface = np.concatenate([x_surface, x_surface[:, :1]], axis=1)
    y_surface = np.concatenate([y_surface, y_surface[:, :1]], axis=1)
    z_surface = np.concatenate([z_surface, z_surface[:, :1]], axis=1)
    
    # Создаем гладкую поверхность
    fig.add_trace(go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        colorscale=[[0, color], [1, color]],
        opacity=0.3,
        showscale=False,
        hoverinfo='skip',
        lighting=dict(
            ambient=0.8,
            diffuse=0.2,
            fresnel=0.1,
            specular=0.1,
            roughness=0.5
        ),
        lightposition=dict(x=100, y=100, z=1000)
    ))
    
    # Добавляем контур для лучшей видимости
    fig.add_trace(go.Scatter3d(
        x=x_surface.flatten(),
        y=y_surface.flatten(),
        z=z_surface.flatten(),
        mode='lines',
        line=dict(color=color, width=1),
        opacity=0.6,
        hoverinfo='skip',
        showlegend=False
    ))
    
    return fig
    """

    
    dots_per_layer = layers_dots[0].shape[1]
    num_layers = len(heights)
    
    # Подготовка вершин
    vertices = []
    for i in range(num_layers):
        for j in range(dots_per_layer):
            vertices.append([layers_dots[i][0][j], layers_dots[i][1][j], heights[i]])
    
    # Подготовка треугольников
    faces = []
    for i in range(num_layers - 1):
        for j in range(dots_per_layer - 1):
            idx1 = i * dots_per_layer + j
            idx2 = i * dots_per_layer + j + 1
            idx3 = (i + 1) * dots_per_layer + j
            idx4 = (i + 1) * dots_per_layer + j + 1
            
            faces.extend([[idx1, idx2, idx3], [idx2, idx4, idx3]])
    
    # Замыкаем контур
    for i in range(num_layers - 1):
        j = dots_per_layer - 1
        idx1 = i * dots_per_layer + j
        idx2 = i * dots_per_layer
        idx3 = (i + 1) * dots_per_layer + j
        idx4 = (i + 1) * dots_per_layer
        
        faces.extend([[idx1, idx2, idx3], [idx2, idx4, idx3]])
    
    # Извлекаем координаты
    vertices = np.array(vertices)
    x, y, z = vertices.T
    
    # Создаем индексы для треугольников
    i, j, k = [], [], []
    for face in faces:
        i.append(face[0])
        j.append(face[1])
        k.append(face[2])
    
    # Добавляем гладкую mesh
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=0.3,
        flatshading=False,
        lighting=dict(
            ambient=0.7,
            diffuse=0.3,
            roughness=0.2,
            specular=0.1
        ),
        lightposition=dict(x=100, y=100, z=1000),
        hoverinfo='skip'
    ))
    
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

def stl2mesh3d(stl_mesh):
        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        vertices, ixr = np.unique(
            stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
        )
        i = np.take(ixr, [3 * k for k in range(p)])
        j = np.take(ixr, [3 * k + 1 for k in range(p)])
        k = np.take(ixr, [3 * k + 2 for k in range(p)])
        return vertices, i, j, k

def draw_alarm_fire_stl_model(
    fig: go.Figure,
    image_file,
    center: List,
    scale_coefs: List,
) -> go.Figure:
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

def add_aircraft(
    fig: go.Figure,
    center: List,
    scale: float = 1.0,
    rotation: List = [0, 0, 0],
    color: str = "red",
    scale_coefs: List[float] = None
) -> go.Figure:
    model = Mesh.from_file('./3d-python-media/models/planes/30.stl')
    
    vertices, i, j, k = stl2mesh3d(model)
    x, y, z = vertices.T

    if scale_coefs is None:
        scale_coefs = [1.0, 1.0, 1.0]
    
    base_rotation = np.radians([90, 0, 0])
    pitch_base, yaw_base, roll_base = base_rotation
    
    Rx_base = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_base), -np.sin(pitch_base)],
        [0, np.sin(pitch_base), np.cos(pitch_base)]
    ])
    
    rotated_base = np.dot(np.vstack([x, y, z]).T, Rx_base.T)
    x, y, z = rotated_base.T
    
    if rotation is not None:
        pitch, yaw, roll = np.radians(rotation)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        R = Rx @ Ry @ Rz
        
        rotated_vertices = np.dot(np.vstack([x, y, z]).T, R.T)
        x, y, z = rotated_vertices.T
    
    x /= scale * scale_coefs[0] * 6000
    y /= scale * scale_coefs[1] * 6000
    z /= scale * scale_coefs[2] * 6000
    
    model_center_x = (np.max(x) + np.min(x)) / 2
    model_center_y = (np.max(y) + np.min(y)) / 2
    
    x = x - model_center_x + center[0]
    y = y - model_center_y + center[1]
    z += center[2] + 500
    
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="rgb(255, 0, 0)",
        flatshading=True,
        showscale=False,
        hoverinfo="skip",
        lighting=dict(
            ambient=0.5,  # Увеличить окружающее освещение
            diffuse=1.0,  # Увеличить рассеянное освещение
            fresnel=0.1,
            specular=1.0,  # Увеличить зеркальное отражение
            roughness=0.05  # Уменьшить шероховатость для большего блеска
        ),
        lightposition=dict(x=200, y=200, z=1000)
    )
    
    fig.add_trace(mesh)
    return fig
