"""
Script that downloads all available Noto Sans fonts & Emoji fonts to a specified output directory.
This output directory should be passed to the PangoCairoTextRenderer via `--fallback_fonts_dir` when working
with multilingual text to ensure that as many glyphs as possible can be rendered.

Usage: python download_fallback_fonts.py <output_dir>
"""

import json
import logging
import os
import sys
import urllib.request

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

NOTO_JSON_URL = "https://notofonts.github.io/noto.json"
CJK_FONT_URLS = [
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf",
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChineseHK/NotoSansCJKhk-Regular.otf",
]
EMOJI_FONT_URLS = ["https://github.com/samuelngs/apple-emoji-linux/releases/download/ios-15.4/AppleColorEmoji.ttf"]


def download_file(url: str, path: str):
    """
    Downloads file at specified URL to specified path on disk
    """
    logger.info(f"Downloading {url}")

    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        logger.warning(f"Exception when trying to download {url}. Response {req.status_code}")
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                file_binary.write(chunk)

    os.rename(download_filepath, path)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # Create output directory
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading fonts to {output_dir}")

    # Find Noto fonts on the official Noto Fonts GitHub page
    with urllib.request.urlopen(NOTO_JSON_URL) as url:
        data = json.load(url)
    base_urls = [v["gh_url"] for k, v in data.items()]

    logger.info(f"Resolved {len(base_urls)} Noto fonts")

    # Iterate over html pages to extract and download font files
    for base_url in tqdm(base_urls):
        with urllib.request.urlopen(base_url) as url:
            data = url.read()
        soup = BeautifulSoup(data, "html.parser")
        for link in soup.findAll("a"):
            url = link.get("href")
            if "Regular" in url and "/hinted/" in url and "Serif" not in url:
                remote_file_url = os.path.join(base_url, url)
                download_file(remote_file_url, os.path.join(output_dir, os.path.basename(remote_file_url)))

    # Download CJK fonts from GitHub
    for remote_file_url in tqdm(CJK_FONT_URLS):
        download_file(remote_file_url, os.path.join(output_dir, os.path.basename(remote_file_url)))

    # Download Apple color emoji font from GitHub
    for remote_file_url in tqdm(EMOJI_FONT_URLS):
        download_file(remote_file_url, os.path.join(output_dir, os.path.basename(remote_file_url)))


if __name__ == "__main__":
    main()
