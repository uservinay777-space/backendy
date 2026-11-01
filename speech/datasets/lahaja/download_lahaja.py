"""Automated downloader for AI4Bharat Lahaja (Hindi Accent Dataset).

Manages gated Hugging Face authentication, disk space verification,
and selective downloading / streaming of audio and metadata.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lahaja_downloader")

HF_DATASET_ID = "ai4bharat/Lahaja"
ESTIMATED_DATASET_SIZE_MB = 1420  # ~1.42 GB


def check_disk_space(target_path: Path, required_mb: int = ESTIMATED_DATASET_SIZE_MB) -> bool:
    """Verify that the target partition has sufficient free disk space."""
    total, used, free = shutil.disk_usage(target_path.anchor or "/")
    free_mb = free / (1024 * 1024)
    logger.info("Available disk space: %.1f MB (Required: %d MB)", free_mb, required_mb)
    return free_mb >= required_mb


def verify_hf_token() -> Optional[str]:
    """Retrieve and validate the Hugging Face access token."""
    token = os.getenv("HF_TOKEN")
    if not token or token.strip().startswith("your_") or not token.strip():
        return None
    return token.strip()


def print_gated_access_guidance():
    """Display clear, user-friendly instructions for accessing the gated Lahaja dataset."""
    print("\n" + "=" * 78)
    print("🔒 GATED DATASET AUTHENTICATION REQUIRED: AI4Bharat/Lahaja")
    print("=" * 78)
    print("AI4Bharat/Lahaja is a gated dataset on Hugging Face.")
    print("To download the complete ~1.42 GB audio archive:")
    print("  1. Log in to your Hugging Face account at https://huggingface.co")
    print(f"  2. Visit the dataset card: https://huggingface.co/datasets/{HF_DATASET_ID}")
    print("  3. Read and accept the access terms on the dataset card.")
    print("  4. Generate a Read access token at https://huggingface.co/settings/tokens")
    print("  5. Export the token in your shell or .env file:")
    print("     export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'")
    print("=" * 78 + "\n")


def download_lahaja_dataset(
    output_dir: Path,
    split: str = "test",
    token: Optional[str] = None,
) -> bool:
    """Download audio recordings and metadata from Hugging Face using huggingface_hub / datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not check_disk_space(output_dir):
        logger.error(
            "Insufficient disk space to download Lahaja (~1.42 GB required). Aborting."
        )
        return False

    hf_token = token or verify_hf_token()
    if not hf_token:
        print_gated_access_guidance()
        logger.error("HF_TOKEN is not configured. Cannot download gated dataset.")
        return False

    logger.info("Initiating download for %s (split='%s') with Hugging Face token...", HF_DATASET_ID, split)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed in the current environment. "
            "Install with: pip install huggingface_hub"
        )
        return False

    try:
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            token=hf_token,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.csv", "*.json", f"*{split}*"],
        )
        logger.info("Lahaja dataset successfully downloaded to: %s", output_dir)
        return True
    except Exception as exc:
        err_msg = str(exc)
        if "403" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            logger.error("Hugging Face access denied (HTTP 403). Access terms have not been accepted yet.")
            print_gated_access_guidance()
        else:
            logger.error("Failed downloading Lahaja dataset: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AI4Bharat Lahaja Hindi Accent Dataset")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: speech/datasets/lahaja)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split to retrieve",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify HF_TOKEN and disk space without downloading",
    )

    args = parser.parse_args()

    token = verify_hf_token()
    has_token = token is not None

    print(f"HF_TOKEN Configured: {'[YES]' if has_token else '[NO (Missing)]'}")
    disk_ok = check_disk_space(args.dest_dir)
    print(f"Disk Space Check:   {'[OK]' if disk_ok else '[FAILED]'}")

    if args.check_only:
        if not has_token:
            print_gated_access_guidance()
        return

    if not has_token:
        print_gated_access_guidance()
        sys.exit(1)

    success = download_lahaja_dataset(args.dest_dir, split=args.split, token=token)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

