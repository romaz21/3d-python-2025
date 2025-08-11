import os
import time

import requests  # type: ignore
from tqdm import tqdm

BASE_URL = """https://srtm.csi.cgiar.org/wp-content/uploads/files/
srtm_5x5/TIFF/srtm_{:02d}_{:02d}.zip"""

print(BASE_URL)

DOWNLOAD_FOLDER = "download"
FILENAME_ZIP_TEMPLATE = "map_{:02d}_{:02d}.zip"

LATITUDE_PARTS = 72  # 360 / 5
LONGTITUDE_PARTS = 24  # 120 degrees out of 180 are covered


if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)


for lat in tqdm(range(LATITUDE_PARTS)):
    for long in tqdm(range(LONGTITUDE_PARTS)):
        url = BASE_URL.format(lat + 1, long + 1)

        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                continue
            path = os.path.join(
                DOWNLOAD_FOLDER, FILENAME_ZIP_TEMPLATE.format(lat + 1, long + 1)
            )
            with open(path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    if chunk:
                        f.write(chunk)
        time.sleep(0.5)
