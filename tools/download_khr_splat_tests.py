#!/usr/bin/env python3
"""Download KHR_gaussian_splatting conformance test assets.

Downloads the official glTF Gaussian splatting example files from the
Khronos conformance archive and extracts them into
tests/assets/khr_splat_conformance/.

Usage:
    python -m tools.download_khr_splat_tests

If the automatic download fails (the URL is a GitHub release attachment
that may require browser authentication), download the zip manually:

    URL: https://github.com/user-attachments/files/25351745/gltf-splat-examples-2026-02-17.zip

Place it as: tests/assets/khr_splat_conformance/gltf-splat-examples-2026-02-17.zip
Then re-run this script to extract.
"""

import os
import sys
import zipfile
from pathlib import Path

DOWNLOAD_URL = (
    "https://github.com/user-attachments/files/25351745/"
    "gltf-splat-examples-2026-02-17.zip"
)
ZIP_NAME = "gltf-splat-examples-2026-02-17.zip"
CONFORMANCE_DIR = Path("tests/assets/khr_splat_conformance")


def download_zip(dest: Path) -> bool:
    """Download the conformance zip. Returns True on success."""
    import urllib.request
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DOWNLOAD_URL}")
    print(f"  -> {dest}")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, str(dest))
        print(f"  Downloaded {dest.stat().st_size:,} bytes")
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP error {e.code}: {e.reason}")
        print()
        print("  The URL may require manual download via a browser.")
        print(f"  Download from: {DOWNLOAD_URL}")
        print(f"  Save to:       {dest}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL error: {e.reason}")
        print()
        print("  The URL may require manual download via a browser.")
        print(f"  Download from: {DOWNLOAD_URL}")
        print(f"  Save to:       {dest}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> list[str]:
    """Extract zip to dest_dir. Returns list of extracted file names."""
    print(f"Extracting {zip_path.name} to {dest_dir}/")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        zf.extractall(str(dest_dir))
    print(f"  Extracted {len(names)} entries")
    return names


def list_assets(directory: Path) -> None:
    """Print a summary of conformance assets found."""
    glb_files = sorted(directory.rglob("*.glb"))
    gltf_files = sorted(directory.rglob("*.gltf"))
    bin_files = sorted(directory.rglob("*.bin"))

    print()
    print(f"=== KHR_gaussian_splatting conformance assets ({directory}) ===")
    print(f"  .glb  files: {len(glb_files)}")
    print(f"  .gltf files: {len(gltf_files)}")
    print(f"  .bin  files: {len(bin_files)}")

    all_assets = glb_files + gltf_files
    if all_assets:
        print()
        for p in all_assets:
            rel = p.relative_to(directory)
            size_kb = p.stat().st_size / 1024
            print(f"  {rel}  ({size_kb:.1f} KB)")
    else:
        print()
        print("  No glTF/GLB assets found.")


def main() -> None:
    CONFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = CONFORMANCE_DIR / ZIP_NAME

    # Check if assets already extracted (look for any .glb or .gltf)
    existing = list(CONFORMANCE_DIR.rglob("*.glb")) + list(CONFORMANCE_DIR.rglob("*.gltf"))
    if existing:
        print(f"Conformance assets already present ({len(existing)} glTF files).")
        list_assets(CONFORMANCE_DIR)
        return

    # Try to extract from existing zip
    if zip_path.exists():
        print(f"Found existing zip: {zip_path}")
        extract_zip(zip_path, CONFORMANCE_DIR)
        list_assets(CONFORMANCE_DIR)
        return

    # Download
    if not download_zip(zip_path):
        print()
        print("=== Manual download instructions ===")
        print(f"1. Download from: {DOWNLOAD_URL}")
        print(f"2. Save to: {zip_path}")
        print(f"3. Run this script again to extract.")
        sys.exit(1)

    # Verify it is a valid zip
    if not zipfile.is_zipfile(str(zip_path)):
        print(f"ERROR: Downloaded file is not a valid zip: {zip_path}")
        print("  The URL may have returned an HTML page instead of the zip.")
        print()
        print("  Please download manually:")
        print(f"    URL:  {DOWNLOAD_URL}")
        print(f"    Save: {zip_path}")
        zip_path.unlink(missing_ok=True)
        sys.exit(1)

    extract_zip(zip_path, CONFORMANCE_DIR)
    list_assets(CONFORMANCE_DIR)
    print()
    print("Done! Run conformance tests with:")
    print("  pytest tests/test_khr_splat_conformance.py -v")


if __name__ == "__main__":
    main()
