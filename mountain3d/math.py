import typing as tp
from math import floor

import numpy as np

from .utils import get_D


def find_azimut(coords, center=(0, 0)):
    """
    coords: array 2 x n
    """
    coords = coords - np.array(center)[None].T
    angles = np.arctan2(coords[0], coords[1])
    angles[coords[0] < 0] = 2 * np.pi + angles[coords[0] < 0]
    return angles


def get_dots_in_sector_pattern_determined(
    azimut_from, azimut_to, radius_in_pixels, dots_amount=None
):
    """
    azimut_from: float, 0 <= azimut_to < 2pi
    azimut_to: float, 0 <= azimut_to - azimut_from < 2pi
    radius_in_pixels: int
    dots_amount: None
    """
    xs = [
        0,
        np.sin(azimut_from) * radius_in_pixels,
        np.sin(azimut_to) * radius_in_pixels,
    ]
    ys = [
        0,
        np.cos(azimut_from) * radius_in_pixels,
        np.cos(azimut_to) * radius_in_pixels,
    ]

    if azimut_from <= 0 <= azimut_to:
        xs.append(0)
        ys.append(radius_in_pixels)
    if azimut_from <= np.pi / 2 <= azimut_to:
        xs.append(radius_in_pixels)
        ys.append(0)
    if azimut_from <= np.pi <= azimut_to:
        xs.append(0)
        ys.append(-radius_in_pixels)
    if azimut_from <= 3 * np.pi / 2 <= azimut_to:
        xs.append(-radius_in_pixels)
        ys.append(0)

    down = np.min(ys)
    up = np.max(ys)
    left = np.min(xs)
    right = np.max(xs)

    coords = np.array(
        np.meshgrid(np.arange(left, right + 1), np.arange(down, up + 1)), dtype=int
    ).reshape(2, -1)
    angles = find_azimut(coords)
    mask = (
        (azimut_from <= angles)
        & (angles <= azimut_to)
        & ((coords[0] ** 2 + coords[1] ** 2) ** 0.5 <= radius_in_pixels)
    )

    return coords[:, mask]


def get_dots_on_grid_determined(sector_pattern, center, grid_shape):
    """
    sector_pattern: array 2 x n in pixels
    center: tuple in pixels
    grid_shape: tuple or list in pixels
    """
    dots = sector_pattern + np.array(center)[:, None]
    mask = (
        (dots[0] >= 0)
        & (dots[1] >= 0)
        & (dots[0] < grid_shape[0])
        & (dots[1] < grid_shape[1])
    )
    return dots[:, mask]


def split_angles_to_bins(angles, bin_size):
    min_angle, max_angle = np.min(angles), np.max(angles)
    sector_amount = (max_angle - min_angle) / bin_size
    rounded_down_sector_amount = floor(sector_amount)

    sector_angles = np.arange(
        min_angle, min_angle + bin_size * (rounded_down_sector_amount + 1), bin_size
    )

    if abs(sector_amount - rounded_down_sector_amount) > 1e-5:
        sector_angles = np.concatenate([sector_angles, [max_angle]])

    idxs = np.arange(angles.shape[0])
    bins = [
        idxs[(angles >= from_) & (angles <= to_)]
        for from_, to_ in zip(sector_angles[:-1], sector_angles[1:])
    ]
    return bins


def obscurance_angles(
    angles_bins,
    coords,
    building_height,
    center_coords,
    heightmap,
    length_coefficient_x,
    length_coefficient_y,
):
    center_height = heightmap[center_coords[0], center_coords[1]]
    max_sectorwise_angles = []
    for idxs in angles_bins:
        xs, ys = coords[:, idxs]
        heights = np.clip(
            heightmap[xs, ys] - center_height,
            a_max=building_height - center_height,
            a_min=0,
        )
        dists = np.sqrt(
            ((xs - center_coords[0]) * length_coefficient_x) ** 2
            + ((ys - center_coords[1]) * length_coefficient_y) ** 2
        )
        tans = np.zeros_like(dists)
        tans[dists != 0] = heights[dists != 0] / dists[dists != 0]
        max_sectorwise_angles.append(np.arctan(tans).max() if len(tans) else 0)
    return np.array(max_sectorwise_angles)


def intersection_with_dome_for_alarm_zone(building_height, config):
    return get_D(building_height, config)


def intersection_with_dome_for_fire_zone(building_height, rocket_config):
    d_min = rocket_config["D_min"]
    d_max = rocket_config["D_max"]
    dead_angle = rocket_config["Dead_angle"] / 180 * np.pi

    r_max = (d_max**2 - building_height**2) ** 0.5
    if building_height < (d_min * np.cos(dead_angle)):
        r_min = (d_min**2 - building_height**2) ** 0.5
    else:
        r_min = building_height * np.tan(dead_angle)
    return r_min, r_max


def create_dome_layer(
    building_height,
    center_height,
    max_sectorwise_angles,
    intersection_with_dome_func,
    config,
):
    relative_height = np.maximum(building_height - center_height, 1e-10)
    r_min, r_max = intersection_with_dome_func(relative_height, config)
    deadlock = relative_height / (np.tan(max_sectorwise_angles) + 1e-20)

    mask = deadlock > r_min
    ans = np.zeros(shape=(2, max_sectorwise_angles.shape[0]))
    if np.sum(mask):
        ans[0][mask] = np.ones(mask.sum()) * r_min
        ans[1][mask] = np.minimum(r_max * np.ones(mask.sum()), deadlock[mask])

    return ans


def build_layers_max_min_distance_for_each_angle_and_height(
    coords,
    center,
    bin_size,
    heights,
    heightmap,
    length_coefficient_x,
    length_coefficient_y,
    intersection_with_dome_func,
    config,
):
    angles = find_azimut(coords, center)
    bins = split_angles_to_bins(angles, bin_size)
    layers_min, layers_max = [], []
    for building_height in heights:
        max_sectorwise_angles = obscurance_angles(
            bins,
            coords,
            building_height,
            center,
            heightmap,
            length_coefficient_x,
            length_coefficient_y,
        )
        mins, maxes = create_dome_layer(
            building_height,
            heightmap[center[0], center[1]],
            max_sectorwise_angles,
            intersection_with_dome_func,
            config,
        )
        layers_min.append(mins)
        layers_max.append(maxes)

    return angles, layers_min, layers_max


def get_layers_dots(angles, bin_size, layers_min, layers_max, center):
    min_angle, max_angle = np.min(angles), np.max(angles)
    sector_amount = (max_angle - min_angle) / bin_size
    rounded_down_sector_amount = floor(sector_amount)

    sector_angles = (
        np.linspace(
            min_angle,
            min_angle + bin_size * (rounded_down_sector_amount - 1),
            rounded_down_sector_amount,
        )
        + bin_size / 2
    )

    if abs(sector_amount - rounded_down_sector_amount) > 1e-5:
        sector_angles = np.concatenate(
            [
                sector_angles,
                np.array(
                    [
                        min_angle
                        + bin_size * (rounded_down_sector_amount + sector_amount) / 2
                    ]
                ),
            ]
        )

    layers_dots_mins = []
    layers_dots_maxes = []
    for layer_min, layer_max in zip(layers_min, layers_max):
        layers_dots_mins.append(
            np.array(
                [
                    (layer_min * np.sin(sector_angles) + center[0]),
                    (layer_min * np.cos(sector_angles) + center[1]),
                ]
            )
        )
        layers_dots_maxes.append(
            np.array(
                [
                    (layer_max * np.sin(sector_angles) + center[0]),
                    (layer_max * np.cos(sector_angles) + center[1]),
                ]
            )
        )
    return layers_dots_mins, layers_dots_maxes


def get_zone(
    zone_type,
    azimut_from,
    azimut_to,
    center,
    bin_size,
    height_amount,
    heightmap,
    config,
    length_coefficient_x,
    length_coefficient_y,
    dots_amount=None,
):
    center_in_pixels = (
        int(center[0] / length_coefficient_x),
        int(center[1] / length_coefficient_y),
    )
    pattern = get_dots_in_sector_pattern_determined(
        azimut_from, azimut_to, radius_in_pixels=2000, dots_amount=dots_amount
    )
    coords = get_dots_on_grid_determined(
        pattern, center=center_in_pixels, grid_shape=heightmap.shape
    )

    dead_angle = config["Dead_angle"] / 180 * np.pi
    antenna_height = config["H_min"] if zone_type == "fire" else config["h_antenna"]
    intersection_height = (
        np.cos(dead_angle) * config["D_max"] if zone_type == "fire" else config["H_max"]
    )
    intersection_with_dome_func = (
        intersection_with_dome_for_fire_zone
        if zone_type == "fire"
        else intersection_with_dome_for_alarm_zone
    )

    min_center_height = heightmap[center_in_pixels] + antenna_height
    max_center_height = heightmap[center_in_pixels] + intersection_height

    low_loyers = np.linspace(
        min_center_height,
        min_center_height + (max_center_height - min_center_height) / 100,
        height_amount,
    )
    high_layers = np.linspace(
        min_center_height + (max_center_height - min_center_height) / 100,
        max_center_height,
        height_amount,
    )
    heights = np.concatenate([low_loyers, high_layers])

    (
        angles,
        layers_min,
        layers_max,
    ) = build_layers_max_min_distance_for_each_angle_and_height(
        coords,
        center_in_pixels,
        bin_size,
        heights,
        heightmap,
        length_coefficient_x,
        length_coefficient_y,
        intersection_with_dome_func,
        config,
    )
    layers_dots_mins, layers_dots_maxes = get_layers_dots(
        angles, bin_size, layers_min, layers_max, center
    )

    return heights, layers_dots_mins, layers_dots_maxes


def dome2layers(
    layers_dots: tp.List[np.ndarray],
    heights: np.ndarray,
):
    layers = []
    dots_per_layer = layers_dots[0].shape[1]

    for layer in range(len(heights) - 1, 0, -1):
        xs = layers_dots[layer][0]
        ys = layers_dots[layer][1]
        zs = [heights[layer]] * dots_per_layer

        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])
        layers.append([xs, ys, zs])
        # layers.sort(key=lambda x: x[3])

    return np.array(layers)


def layers2inner_outer(layers: np.array):
    """Split detection dome layers into innner and outer parts"""
    layer_heights = layers[:, 2, 0]
    max_height_layer_id = np.argmax(layer_heights)

    inner_detection_surface_layers = layers[max_height_layer_id:][::-1]
    outer_detection_surface_layers = layers[:max_height_layer_id]

    return inner_detection_surface_layers, outer_detection_surface_layers


def interpolate(layer1, layer2, p: int) -> tp.List[np.ndarray]:
    x = p * layer1[0, :] + (1 - p) * layer2[0, :]
    y = p * layer1[1, :] + (1 - p) * layer2[1, :]
    z = p * layer1[2, :] + (1 - p) * layer2[2, :]
    return [x, y, z]


def create_dome_slice(height: int, layers: np.ndarray) -> np.ndarray:
    """Peforms linear interpolation of 2 layers closest to the given height"""

    layer_heights = layers[:, 2, 0]
    assert sorted(layer_heights.tolist()) == layer_heights.tolist()

    # Searching for the closest layers
    i = np.argmin(np.abs(layer_heights - height))
    j = (
        max(0, i - 1)
        if layer_heights[i] > height
        else min(len(layer_heights) - 1, i + 1)
    )
    i, j = sorted((i, j))

    closes_lower_layer_height, closest_upper_layer_height = (
        layer_heights[i],
        layer_heights[j],
    )
    p = (
        0
        if closes_lower_layer_height == closest_upper_layer_height
        else (height - closes_lower_layer_height)
        / (closest_upper_layer_height - closes_lower_layer_height)
    )
    interpolated_layer = interpolate(layers[i], layers[j], p)
    return interpolated_layer
