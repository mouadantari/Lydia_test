import argparse
import concurrent.futures
import csv
import time
from pathlib import Path

import requests


def download_image(url):
    """Download image from url and save it to filename"""
    filename = url.split("/")[-1]
    file = Path("./images").joinpath(filename)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as handle:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="CSV file with image URLs", default="data.csv")
    args = parser.parse_args()

    with open(args.csv_file) as handle:
        reader = csv.reader(handle)
        urls = [r[0] for i, r in enumerate(reader) if i > 0]

    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_image, urls)

    print(f"Downloaded {len(urls)} images in {time.perf_counter() - t:.2f} seconds")
