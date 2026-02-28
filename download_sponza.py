#!/usr/bin/env python3
"""Download Khronos Sponza glTF model and convert to a single GLB file.

The C++ and Rust loaders only support GLB-embedded textures, so we download
the separate .gltf/.bin/texture files and re-export as assets/Sponza.glb.
"""

import json
import mimetypes
import os
import struct
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Sponza/glTF/"
DOWNLOAD_DIR = Path("assets/sponza_gltf")
OUTPUT_GLB = Path("assets/Sponza.glb")


def download_file(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, str(dest))


def discover_textures(gltf_path: Path) -> list[str]:
    """Parse the glTF JSON and return all referenced image URIs."""
    with open(gltf_path, "r") as f:
        gltf = json.load(f)
    uris = []
    for image in gltf.get("images", []):
        uri = image.get("uri")
        if uri:
            uris.append(uri)
    return uris


def download_sponza() -> None:
    """Download the Sponza glTF + bin + all textures."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Download the main glTF file first
    gltf_path = DOWNLOAD_DIR / "Sponza.gltf"
    download_file(BASE_URL + "Sponza.gltf", gltf_path)

    # Parse to discover textures
    textures = discover_textures(gltf_path)
    print(f"Found {len(textures)} textures in Sponza.gltf")

    # Download .bin file
    download_file(BASE_URL + "Sponza.bin", DOWNLOAD_DIR / "Sponza.bin")

    # Download all textures
    for tex_uri in textures:
        download_file(BASE_URL + tex_uri, DOWNLOAD_DIR / tex_uri)

    print("All files downloaded.")


def _pad_to_4(data: bytes, pad_byte: bytes = b'\x00') -> bytes:
    """Pad data to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder:
        data += pad_byte * (4 - remainder)
    return data


def _mime_for_ext(ext: str) -> str:
    """Return MIME type for image file extension."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext.lower(), "application/octet-stream")


def convert_to_glb_manual() -> bool:
    """Convert glTF + separate files to a single GLB by manually packing the binary.

    GLB layout:
      12-byte header: magic(0x46546C67) + version(2) + totalLength
      JSON chunk:     length + type(0x4E4F534A) + JSON padded to 4 bytes with spaces
      BIN  chunk:     length + type(0x004E4942) + binary data padded to 4 bytes with 0x00
    """
    gltf_path = DOWNLOAD_DIR / "Sponza.gltf"
    bin_path = DOWNLOAD_DIR / "Sponza.bin"

    with open(gltf_path, "r") as f:
        gltf = json.load(f)

    # Read the original .bin buffer
    bin_data = bytearray(bin_path.read_bytes())

    # Strip node transforms (Sponza has a 0.008 scale that shrinks the scene;
    # our light positions are designed for the original coordinate space)
    for node in gltf.get("nodes", []):
        node.pop("scale", None)
        node.pop("rotation", None)
        node.pop("translation", None)
        node.pop("matrix", None)

    # Embed each image that has a URI into the binary buffer as a bufferView
    buffer_views = gltf.get("bufferViews", [])
    images = gltf.get("images", [])
    embedded = 0

    for image in images:
        uri = image.get("uri")
        if not uri:
            continue  # already embedded

        img_path = DOWNLOAD_DIR / uri
        if not img_path.exists():
            print(f"  WARNING: texture not found: {uri}")
            continue

        img_bytes = img_path.read_bytes()
        ext = img_path.suffix

        # Pad current buffer to 4-byte alignment before appending
        while len(bin_data) % 4:
            bin_data.append(0)

        offset = len(bin_data)
        bin_data.extend(img_bytes)

        # Create a new bufferView for this image
        bv_index = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(img_bytes),
        })

        # Update image: remove uri, add bufferView + mimeType
        del image["uri"]
        image["bufferView"] = bv_index
        image["mimeType"] = _mime_for_ext(ext)
        embedded += 1

    print(f"  Embedded {embedded} textures into binary buffer")

    # Update the buffer's total byte length
    gltf["bufferViews"] = buffer_views
    if gltf.get("buffers"):
        gltf["buffers"][0]["byteLength"] = len(bin_data)
        # Remove the URI from the buffer (GLB uses the BIN chunk directly)
        gltf["buffers"][0].pop("uri", None)

    # Encode JSON chunk
    json_bytes = _pad_to_4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b' ')
    # Pad binary chunk
    bin_bytes = _pad_to_4(bytes(bin_data))

    # Build GLB
    total_length = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))
    OUTPUT_GLB.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_GLB, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN chunk
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        f.write(bin_bytes)

    size_mb = OUTPUT_GLB.stat().st_size / 1024 / 1024
    print(f"Saved {OUTPUT_GLB} ({size_mb:.1f} MB)")
    return True


def main() -> None:
    if OUTPUT_GLB.exists():
        size_mb = OUTPUT_GLB.stat().st_size / 1024 / 1024
        print(f"{OUTPUT_GLB} already exists ({size_mb:.1f} MB). Delete it to re-download.")
        return

    print("=== Downloading Khronos Sponza ===")
    download_sponza()

    print("\n=== Converting to GLB ===")
    if not convert_to_glb_manual():
        print("\nERROR: Could not convert to GLB.")
        sys.exit(1)

    print("\nDone! Run the Sponza demo with:")
    print("  run_sponza_cpp.bat   (C++ engine)")
    print("  run_sponza_rust.bat  (Rust engine)")


if __name__ == "__main__":
    main()
