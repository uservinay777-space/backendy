"""Selective downloader for OpenSLR 17 (MUSAN).

Downloads specific categories (noise, music, speech/babble) from OpenSLR 17
without pulling down the full 11 GB uncompressed archive.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("musan_downloader")

OPENSLR_17_BASE_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def download_musan_tar(dest_path: Path) -> None:
    """Download the musan.tar.gz archive with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s ...", OPENSLR_17_BASE_URL, dest_path)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            mb_cur = (block_num * block_size) / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:5.1f}% ({mb_cur:6.1f} / {mb_tot:6.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(OPENSLR_17_BASE_URL, dest_path, reporthook=_progress)
        sys.stdout.write("\n")
        logger.info("MUSAN archive downloaded successfully: %s", dest_path)
    except Exception as exc:
        sys.stdout.write("\n")
        logger.error("Failed downloading MUSAN archive: %s", exc)
        raise


def extract_category_subset(
    archive_path: Path,
    target_raw_dir: Path,
    category: str,
    max_files: int = 10,
) -> int:
    """Selectively extract up to max_files WAV recordings for a given category."""
    cat_dir = target_raw_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    prefix = f"musan/{category}/" if category != "babble" else "musan/speech/"
    logger.info("Extracting up to %d files for '%s' from %s ...", max_files, category, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.name.startswith(prefix) and member.name.lower().endswith(".wav"):
                filename = Path(member.name).name
                dest = cat_dir / filename
                f = tar.extractfile(member)
                if f:
                    with open(dest, "wb") as out_f:
                        out_f.write(f.read())
                    count += 1
                if count >= max_files:
                    break

    logger.info("Extracted %d audio files into %s", count, cat_dir)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract MUSAN subsets from OpenSLR 17")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: speech/datasets/musan)",
    )
    parser.add_argument(
        "--category",
        choices=["noise", "music", "babble", "all"],
        default="all",
        help="Acoustic category to extract",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum files to extract per category",
    )

    args = parser.parse_args()
    archive_path = args.dest_dir / "musan.tar.gz"
    raw_dir = args.dest_dir / "raw"

    if not archive_path.is_file():
        download_musan_tar(archive_path)

    cats = ["noise", "music", "babble"] if args.category == "all" else [args.category]
    for c in cats:
        extract_category_subset(archive_path, raw_dir, c, max_files=args.max_files)


if __name__ == "__main__":
    main()

