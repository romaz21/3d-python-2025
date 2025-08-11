import os
import re
import zipfile

import rasterio as rs
from rasterio.crs import CRS
from rasterio.enums import Resampling

RESAMPLE_FACTOR = 0.2

DOWNLOAD_FOLDER = "download"
PROCESSED_MAP_FOLDER = "preprocessed"

FILENAME_ZIP_TEMPLATE = "map_{:02d}_{:02d}.zip"
PROCESSED_FILENAME_TEMPLATE = "map_{:02d}_{:02d}.tif"

FILENAME_REGEXP = r"map_\d\d_\d\d.zip"


for filename in os.listdir(DOWNLOAD_FOLDER):
    if not re.match(FILENAME_REGEXP, filename):
        continue

    # Unarchive
    x, y = re.findall(r"\d\d", filename)
    file, _ = os.path.splitext(filename)
    with zipfile.ZipFile(os.path.join(DOWNLOAD_FOLDER, filename), "r") as zip_ref:
        # Shift cells indexing for making latitude start from 1 at North pole
        new_filename = PROCESSED_FILENAME_TEMPLATE.format(int(x), int(y) + 6)
        new_path = os.path.join(PROCESSED_MAP_FOLDER, new_filename)
        zip_ref.getinfo(f"srtm_{x}_{y}.tif").filename = new_path
        zip_ref.extract(f"srtm_{x}_{y}.tif")

    # Compress images by resample_factor
    with rs.open(new_path) as dataset:
        imgdata = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * RESAMPLE_FACTOR),
                int(dataset.width * RESAMPLE_FACTOR),
            ),
            resampling=Resampling.bilinear,
        ).squeeze()
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / imgdata.shape[-1]), (dataset.height / imgdata.shape[-2])
        )

    with rs.open(
        new_path,
        "w",
        driver="GTiff",
        height=imgdata.shape[0],
        width=imgdata.shape[1],
        count=1,
        dtype=imgdata.dtype,
        crs=CRS.from_string("EPSG:4326"),
        transform=transform,
    ) as dst:
        dst.write(imgdata, 1)
