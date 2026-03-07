"""
Generate placeholder AURA extension icons (purple gradient).
Run once: python extension/create_icons.py
Requires: pip install Pillow
"""

import os
import struct
import zlib

SIZES = [16, 48, 128]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "icons")


def _make_png(size: int) -> bytes:
    """Create a minimal valid PNG with a purple gradient — no Pillow required."""

    def _chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    # IHDR: width, height, bit depth 8, color type 2 (RGB), compression 0, filter 0, interlace 0
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)

    # Build raw pixel rows: purple gradient (#5b21b6 → #a78bfa corner to corner)
    raw_rows = bytearray()
    for y in range(size):
        raw_rows.append(0)  # filter byte = None
        for x in range(size):
            t = (x + y) / (2 * (size - 1)) if size > 1 else 0
            r = int(0x5b + t * (0xa7 - 0x5b))
            g = int(0x21 + t * (0x8b - 0x21))
            b = int(0xb6 + t * (0xfa - 0xb6))
            raw_rows += bytes([r, g, b])

    idat_data = zlib.compress(bytes(raw_rows))

    signature = b"\x89PNG\r\n\x1a\n"
    return (
        signature
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat_data)
        + _chunk(b"IEND", b"")
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for size in SIZES:
        path = os.path.join(OUTPUT_DIR, f"icon{size}.png")
        png_bytes = _make_png(size)
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"Created {path} ({len(png_bytes)} bytes)")

    print("Done. Load the extension in Chrome -> chrome://extensions -> Load unpacked -> extension/")


if __name__ == "__main__":
    main()
