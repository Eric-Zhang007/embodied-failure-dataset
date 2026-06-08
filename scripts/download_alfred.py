"""
下载 ALFRED json_2.1.0.7z（纯 JSON，不含图像/特征，~200MB）。
提取到 data/json_2.1.0/ 目录。

运行（WSL2 中）：
    cd ~/embodied-failure-dataset
    uv run python scripts/download_alfred.py
"""

import os
import subprocess
import sys

DATA_DIR = "data"
URL = "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
ARCHIVE = "json_2.1.0.7z"
EXTRACT_DIR = "json_2.1.0"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)

    # 下载
    if os.path.exists(EXTRACT_DIR):
        print(f"Already extracted: {DATA_DIR}/{EXTRACT_DIR}/")
        return

    if not os.path.exists(ARCHIVE):
        print(f"Downloading {URL} ...")
        subprocess.check_call(["wget", URL, "--show-progress"])
    else:
        print(f"Archive already downloaded: {ARCHIVE}")

    # 解压
    print(f"Extracting {ARCHIVE} ...")
    subprocess.check_call(["7z", "x", ARCHIVE, "-y"])
    os.remove(ARCHIVE)
    print(f"Done. Data at {DATA_DIR}/{EXTRACT_DIR}/")


if __name__ == "__main__":
    main()
