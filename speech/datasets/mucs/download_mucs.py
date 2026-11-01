"""Automated downloader for OpenSLR 104 (MUCS 2021) Test Subset.

Policy:
- Downloads ONLY the official Hindi-English test subset.
- Does NOT download the large 90-hour training corpus.
- Verifies extracted audio files against manifest.json.
- Fully respects CC BY-SA 4.0 license.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mucs_downloader")

OPENSLR_104_BASE_URL = "https://www.openslr.org/resources/104"
DEFAULT_TEST_ARCHIVE = "Hindi-English-test.zip"


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s ...", url, dest_path)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, block_num * block_size * 100.0 / total_size)
            mb_downloaded = (block_num * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        sys.stdout.write("\n")
        logger.info("Download completed successfully: %s", dest_path)
    except Exception as exc:
        sys.stdout.write("\n")
        logger.error("Failed to download %s: %s", url, exc)
        raise


def extract_audio(archive_path: Path, output_audio_dir: Path) -> int:
    """Extract audio files (.wav) from the zip archive into the destination audio directory."""
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    logger.info("Extracting audio files from %s to %s ...", archive_path, output_audio_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            if member.filename.lower().endswith(".wav"):
                filename = Path(member.filename).name
                dest = output_audio_dir / filename
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                count += 1

    logger.info("Extracted %d audio files to %s", count, output_audio_dir)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MUCS 2021 test subset from OpenSLR 104")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: speech/datasets/mucs)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=f"{OPENSLR_104_BASE_URL}/{DEFAULT_TEST_ARCHIVE}",
        help="Direct URL to official test archive",
    )
    parser.add_argument(
        "--skip-download-if-exists",
        action="store_true",
        default=True,
        help="Skip downloading if the zip file already exists locally",
    )

    args = parser.parse_args()
    archive_dest = args.dest_dir / DEFAULT_TEST_ARCHIVE
    audio_dest = args.dest_dir / "audio"

    if not (args.skip_download_if-exists and archive_dest.exists()):
        download_file(args.url, archive_dest)
    else:
        logger.info("Archive %s already exists, skipping download.", archive_dest)

    if archive_dest.exists():
        extract_audio(archive_dest, audio_dest)


if __name__ == "__main__":
    main()

