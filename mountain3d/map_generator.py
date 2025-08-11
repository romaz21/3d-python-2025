import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

import geopy
import geopy.distance
import numpy as np
import rasterio as rs
from loguru import logger
from pydantic import BaseModel, field_validator
from rasterio.crs import CRS
from rasterio.merge import merge

FILENAME_TEMPLATE = "map_{:02d}_{:02d}.tif"

logger.remove()
logger.add(sys.stderr, level="INFO")

# Maybe 255.0
NODATA_VALUE = -32768.0


def lat_long_checker_and_corrector(
    lat: float, long: float, EPS: float = 1e-2
) -> Tuple[float, float]:
    assert -90 <= lat <= 90
    assert -180 <= long <= 180

    if np.isclose(np.abs(lat), 90):
        lat -= np.sign(lat) * EPS

    if np.isclose(np.abs(long), 180):
        long -= np.sign(long) * EPS

    return lat, long


class Cell(BaseModel):
    """
    Cells according to: https://srtm.csi.cgiar.org/srtmdata/
    """

    lat_ind: int
    long_ind: int

    @field_validator("lat_ind")
    @classmethod
    def lat_restriction(cls, value: int) -> int:
        if 1 <= value <= 36:
            return value
        raise ValueError(f"lat_ind must be in [1, 36], but get {value=}")

    @field_validator("long_ind")
    @classmethod
    def long_restriction(cls, value: int) -> int:
        if 1 <= value <= 72:
            return value
        raise ValueError(f"lat_ind must be in [1, 72], but get {value=}")

    @staticmethod
    def construct_from_lat_long(lat: float, long: float) -> "Cell":
        """
        @lat: float, latitude btw in [-90, 90]
        @long: float, longtitude [-180, +180] + for E, - for W.
        """
        lat, long = lat_long_checker_and_corrector(lat, long)
        lat_ind = (90 - lat) // 5 + 1
        long_ind = (long + 180) // 5 + 1
        return Cell(lat_ind=lat_ind, long_ind=long_ind)


def get_min_or_max(value_1: int, value_2: int, min_condition: bool) -> int:
    return min(value_1, value_2) if min_condition else max(value_1, value_2)


def get_max_with_threshold(cur_value: int, new_value: int, threshold: int = 20) -> int:
    if new_value < threshold:
        return max(cur_value, new_value)
    return cur_value


def get_min_with_threshold(cur_value: int, new_value: int, threshold: int = 52) -> int:
    if new_value > threshold:
        return min(cur_value, new_value)
    return cur_value


def get_map_geotiff(  # noqa
    heightmap_paths: Path,
    lat: float,
    long: float,
    km_side_of_square: int,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Construct and (optionally) write to file square map from center
    @lat: float, latitude btw in [-90, 90]
    @long: float, longtitude [-180, +180] + for E, - for W.
    @km_side_of_square: int, size of square in kilometers
    @return: Tuple[np.ndarray, np.ndarray, np.ndarray] heights, lats and longs
    """
    lat, long = lat_long_checker_and_corrector(lat, long)

    center = geopy.Point(lat, long)
    center_cell = Cell.construct_from_lat_long(center.latitude, center.longitude)

    map_angles: list[Cell] = []
    coord_angles: list[geopy.Point] = []
    dist = geopy.distance.distance(kilometers=np.sqrt(2) / 2 * km_side_of_square)

    rotations = [45, 135, 225, 315]
    for rot in rotations:
        angle_point = dist.destination(point=center, bearing=rot)
        coord_angles.append(angle_point)
        indeces = Cell.construct_from_lat_long(
            angle_point.latitude, angle_point.longitude
        )
        map_angles.append(indeces)

    low_lat = max(map_angles[1].lat_ind, map_angles[2].lat_ind)
    high_lat = min(map_angles[0].lat_ind, map_angles[3].lat_ind)

    # TODO [low-priority] work properly near
    # poles need to fix flip across the poles
    # low_lat = get_min_or_max(
    #     map_angles[1].lat_ind,
    #     map_angles[2].lat_ind,
    #     (center_cell.lat_ind < map_angles[1].lat_ind
    #     or center_cell.lat_ind < map_angles[2].lat_ind)
    # )

    # TODO [low-priority] work properly near
    # poles need to fix flip across the poles
    # high_lat = get_min_or_max(
    #     map_angles[0].lat_ind,
    #     map_angles[3].lat_ind,
    #     not (center_cell.lat_ind > map_angles[0].lat_ind or
    #          center_cell.lat_ind > map_angles[3].lat_ind)
    # )

    right_long = max(map_angles[0].long_ind, map_angles[1].long_ind)
    if (
        center_cell.long_ind > map_angles[0].long_ind
        or center_cell.long_ind > map_angles[1].long_ind
    ):
        right_long = get_max_with_threshold(1, map_angles[0].long_ind)
        right_long = get_max_with_threshold(right_long, map_angles[1].long_ind)

    left_long = min(map_angles[2].long_ind, map_angles[3].long_ind)
    if (
        center_cell.long_ind < map_angles[2].long_ind
        or center_cell.long_ind < map_angles[3].long_ind
    ):
        left_long = get_min_with_threshold(72, map_angles[2].long_ind)
        left_long = get_min_with_threshold(left_long, map_angles[3].long_ind)

    lat_inds = list(range(high_lat, low_lat + 1))

    long_inds = [left_long]
    cur_long = left_long
    while cur_long != right_long:
        cur_long = cur_long % 72 + 1
        long_inds.append(cur_long)

    # Read GeoTiff
    maps = []
    for i in lat_inds:
        for j in long_inds:
            fname = os.path.join(heightmap_paths, FILENAME_TEMPLATE.format(j, i))
            if os.path.isfile(fname):
                file = rs.open(fname)
                maps.append(file)
                continue
            logger.warning(f"No file {fname=}")

    if not maps:
        logger.error("No map for that area! Closing app...")
        logger.info("'lat' should be in [-60, 60], 'long' in [-180, 180]")
        exit()

    bounds = [
        int(coord_angles[2].longitude),
        int(coord_angles[2].latitude),
        int(coord_angles[0].longitude),
        int(coord_angles[0].latitude),
    ]
    req_bounds_np = np.array(bounds).reshape(2, 2)
    deg_size = req_bounds_np[1] - req_bounds_np[0]
    resolution = deg_size / np.array(out_shape)
    heightmap = merge(maps, bounds, resolution.tolist())
    heightmap, heightmap_transform = heightmap[0][0], heightmap[1]

    # TODO [low-priority] make clever infilling for no-data cells
    heightmap[heightmap == NODATA_VALUE] = 0

    latitudes = np.zeros_like(heightmap, dtype=float)
    longitudes = np.zeros_like(heightmap, dtype=float)
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            lat, long = heightmap_transform * (i, j)
            latitudes[i, j] = lat
            longitudes[i, j] = long

    logger.debug(f"{coord_angles=}")
    logger.debug(f"{map_angles=}")
    logger.debug(f"{low_lat=}, {high_lat=}, {left_long=}, {right_long=}")
    logger.debug("Maps indeces:")
    for i in lat_inds:
        for j in long_inds:
            logger.debug(f"{j, i} ", end="")
        logger.debug("\n")
    logger.debug(f"{req_bounds_np=}")

    return heightmap, heightmap_transform, latitudes, longitudes, req_bounds_np


class CachedMapGenerator:
    def __init__(
        self, heightmap_paths: str, cache_path: Optional[str], cache_size: int = 10
    ):
        self.heightmap_paths: Path = Path(heightmap_paths)
        self.cache_size = cache_size
        self.cache: Optional[OrderedDict[str, None]] = None
        if cache_path is None or cache_size == 0:
            self.cache_size = 0
            return
        self.cache_path: Path = Path(str(cache_path))

        # Create folder for cache if not exisits
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # If already exists, upload cache to RAM
        cached_files = [
            path
            for path in os.listdir(self.cache_path)
            if re.match(r"cached_map_\d+_\d+_\d+_\d+_\d+", path)
        ]

        cached_files_with_meta = []
        for file in cached_files:
            with open(self.cache_path / file / "meta.txt") as f:
                last_access_tstmp = json.load(f)["last_access_tstmp"]
            cached_files_with_meta.append((file, last_access_tstmp))

        # Sort by time of access
        cached_files_with_meta.sort(key=lambda x: x[1])

        to_upload = cached_files_with_meta[-cache_size:]
        to_delete = cached_files_with_meta[:-cache_size]

        # Delete if files more that cache_size
        for del_file in to_delete:
            shutil.rmtree(self.cache_path / del_file[0])

        self.cache = OrderedDict(
            zip((f[0] for f in to_upload), [None] * len(to_upload))
        )

    @staticmethod
    def args_to_cache_name(*args):
        string_params = ["cached_map"]
        for arg in args:
            if isinstance(arg, float):
                string_params.append(str(round(arg * 100)))
            else:
                string_params.append(str(arg))
        return "_".join(string_params)

    def load_map(
        self, cache_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        folder = self.cache_path / cache_name
        with rs.open(folder / f"{cache_name}.tif") as dataset:
            raster = dataset.read(1)
        with open(folder / f"{cache_name}.bytes", "rb") as f:
            (latitudes, longitudes, req_bounds_np) = pickle.load(f)
        with open(folder / "meta.txt", "w") as f:
            json.dump({"last_access_tstmp": int(time.time())}, f)
        return raster, latitudes, longitudes, req_bounds_np

    def dump_map(
        self,
        cache_name: str,
        raster: np.ndarray,
        raster_transforms: Any,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        req_bounds_np: np.ndarray,
    ) -> None:
        folder = self.cache_path / cache_name
        os.makedirs(folder)
        with rs.open(
            folder / f"{cache_name}.tif",
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=CRS.from_string("EPSG:4326"),
            transform=raster_transforms,
        ) as dst:
            dst.write(raster, 1)
        with open(folder / f"{cache_name}.bytes", "wb") as f:
            pickle.dump((latitudes, longitudes, req_bounds_np), f)
        with open(folder / "meta.txt", "w") as f:
            json.dump({"last_access_tstmp": int(time.time())}, f)

    def get_map(
        self,
        lat: float,
        long: float,
        km_side_of_square: int,
        out_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_name = self.args_to_cache_name(lat, long, km_side_of_square, *out_shape)
        if self.cache is not None and cache_name in self.cache:
            logger.info(
                f"Read cached map for {lat=}, {long=}, "
                + f"{km_side_of_square=}, {out_shape=}. "
                + f"Cache name is {cache_name}"
            )
            self.cache.move_to_end(cache_name)
            return self.load_map(cache_name)

        logger.info(f"Compute map {lat=}, {long=}, {km_side_of_square=}, {out_shape=}")

        return_values = get_map_geotiff(
            heightmap_paths=self.heightmap_paths,
            lat=lat,
            long=long,
            km_side_of_square=km_side_of_square,
            out_shape=out_shape,
        )

        if self.cache is not None:
            if self.cache_size == len(self.cache):
                path_to_delete = self.cache.popitem(last=False)[0]
                shutil.rmtree(self.cache_path / path_to_delete)
            self.cache[cache_name] = None
            self.dump_map(cache_name, *return_values)

        heightmap, _, latitudes, longitudes, req_bounds_np = return_values
        return heightmap, latitudes, longitudes, req_bounds_np
