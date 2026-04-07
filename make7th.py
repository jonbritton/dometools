#!/usr/bin/env python3
from __future__ import annotations
"""
make7th.py  –  Convert PNG frame sequences to .7TH-422 files

Usage:
    ./make7th.py /frames/            /output/dir/
    ./make7th.py '/frames/*.png'     /output/dir/
        Encode every PNG/EXR in the directory (or quoted glob) into as many
        .7th files as needed. Output names are derived from the input sequence:
            show.1080p.00001.png … show.1080p.00029.png
          →  show.1080p.00000.7th  show.1080p.00010.7th  show.1080p.00020.7th

    ./make7th.py -r 100-1000 /frames/ /output/dir/
        As above, but only encode frames whose sequence number is in [100, 1000].

The 7th-422 format:
    Header : 100 bytes (magic, width, height, padding)
    Frames : 10 × width × height × 2 bytes  (YUYV 4:2:2, 8-bit per sample)
    Markers: frames 0-8 carry a 8-byte sync watermark in the bottom-right corner

File size = 100 + 10 × W × H × 2
  e.g. 4096×4096: 100 + 10×33,554,432 = 335,544,420 bytes exactly
"""

import argparse
import json
import os
import re
import struct
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Optional EXR support – requires:  pip install openexr imath
try:
    import OpenEXR
    import Imath
    _EXR_FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

# constants

MAGIC       = b"7th \n\n\x00\x00"   # 8-byte file signature
NUM_FRAMES  = 10
HEADER_SIZE = 100

# BT.709 luma coefficients (ITU-R BT.709 / sRGB primaries)
# Y  =  Kr·R + (1-Kr-Kb)·G + Kb·B
# Cb = (B - Y) / (2·(1-Kb)) + 0.5
# Cr = (R - Y) / (2·(1-Kr)) + 0.5
KR, KG, KB = 0.2126, 0.7152, 0.0722

# Fill pattern for missing/empty frames (black, near-neutral chroma)
# YUYV: [Y0=0x00, Cb=0x7f, Y1=0x00, Cr=0x7f] repeating
FILL_YUYV = bytes([0x00, 0x7f, 0x00, 0x7f])

# Sync watermark embedded into frames 0-8
# Observed pattern: at row H-1, byte-offset (H-1)·W·2 + (start_col//2)·4
# start_col = 4060 + frame_index·4   (frame 9 would be col 4096 = out-of-bounds, so no marker)
MARKER_BYTES   = bytes([0x00, 0x10, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00])
MARKER_COL_BASE = 4060    # pixel column of the first marker (frame 0)
MARKER_COL_STEP = 4       # columns per frame


# Color conversion 

def rgb_float_to_yuyv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an H×W×3 float32 RGB array (values 0.0–1.0) to a
    flat H×(W*2) uint8 YUYV-packed buffer.

    YUYV macro-pixel layout for pixels (x, x+1):
        [Y0, Cb, Y1, Cr]  – 4 bytes for 2 pixels
    Chroma is horizontally subsampled (4:2:2) by averaging adjacent pairs.
    """
    H, W, _ = rgb.shape
    if W % 2 != 0:
        raise ValueError(f"Width must be even for 4:2:2, got {W}")

    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # BT.709 RGB → YCbCr (full range, floating point, output 0-1)
    Y  =  KR * R + KG * G + KB * B
    Cb = (B - Y) / (2.0 * (1.0 - KB)) + 0.5
    Cr = (R - Y) / (2.0 * (1.0 - KR)) + 0.5

    # Quantise to uint8.
    # Y uses round-to-nearest; chroma uses truncation (floor) to match DFM output.
    Y_u8  = np.clip(Y  * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Cb_u8 = np.clip(Cb * 255.0,       0, 255).astype(np.uint8)
    Cr_u8 = np.clip(Cr * 255.0,       0, 255).astype(np.uint8)

    # Horizontal chroma subsampling: average pairs
    Cb_sub = ((Cb_u8[:, 0::2].astype(np.uint16) + Cb_u8[:, 1::2].astype(np.uint16) + 1) >> 1).astype(np.uint8)
    Cr_sub = ((Cr_u8[:, 0::2].astype(np.uint16) + Cr_u8[:, 1::2].astype(np.uint16) + 1) >> 1).astype(np.uint8)

    # Pack into YUYV: shape (H, W*2)
    yuyv = np.empty((H, W * 2), dtype=np.uint8)
    yuyv[:, 0::4] = Y_u8[:, 0::2]   # Y0
    yuyv[:, 1::4] = Cb_sub           # Cb  (shared for the pair)
    yuyv[:, 2::4] = Y_u8[:, 1::2]   # Y1
    yuyv[:, 3::4] = Cr_sub           # Cr  (shared for the pair)

    return yuyv


# I/O helpers 
def load_png(path):
    """
    Load a PNG (8- or 16-bit, RGB or RGBA) and return a float32 H×W×3
    array with values in [0.0, 1.0].
    """
    img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)

    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        return arr.astype(np.float32)


def load_exr(path):
    """
    Load an EXR file and return a float32 H×W×3 array.

    Channel priority:
      1. R, G, B          – plain RGB (most common)
      2. <layer>.R/G/B    – first layer found (e.g. beauty.R)
      3. Y                – luminance-only, replicated to all three channels

    HDR values above 1.0 are preserved; clipping occurs during uint8
    quantisation in rgb_float_to_yuyv().  For scene-linear ACES content
    apply a viewing LUT / exposure adjustment before passing to this tool.
    """
    if not HAS_OPENEXR:
        raise RuntimeError(
            "OpenEXR is not installed.\n"
            "Install with:  pip install openexr imath\n"
            "Cannot load: " + str(path)
        )

    f     = OpenEXR.InputFile(str(path))
    hdr   = f.header()
    dw    = hdr["dataWindow"]
    W     = dw.max.x - dw.min.x + 1
    H     = dw.max.y - dw.min.y + 1
    chans = list(hdr["channels"].keys())

    def read_chan(name):
        raw = f.channel(name, _EXR_FLOAT)
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W)

    if "R" in chans and "G" in chans and "B" in chans:
        rgb = np.stack([read_chan("R"), read_chan("G"), read_chan("B")], axis=-1)

    else:
        # Try layer-prefixed channels: take whichever layer has all three
        r = next((c for c in chans if c.endswith(".R")), None)
        g = next((c for c in chans if c.endswith(".G")), None)
        b = next((c for c in chans if c.endswith(".B")), None)

        if r and g and b:
            layer = r[:-2]  # strip ".R" for the log message
            print(f"    EXR: using layer '{layer}' channels")
            rgb = np.stack([read_chan(r), read_chan(g), read_chan(b)], axis=-1)

        elif "Y" in chans:
            print(f"    EXR: only luminance channel found, encoding as grey")
            y   = read_chan("Y")
            rgb = np.stack([y, y, y], axis=-1)

        else:
            raise ValueError(
                "No RGB or Y channels found in {}.\nAvailable: {}".format(
                    path, chans)
            )

    return rgb


def load_frame(path):
    """Dispatch to load_png or load_exr based on file extension."""
    if path.suffix.lower() == ".exr":
        return load_exr(path)
    return load_png(path)


def make_fill_frame(W: int, H: int) -> bytearray:
    """Return a full-sized YUYV frame filled with the default fill pattern."""
    tile   = FILL_YUYV * (W // 2)      # one row
    row    = bytes(tile)
    frame  = bytearray(row * H)
    return frame


def stamp_marker(frame: bytearray, W: int, H: int, frame_index: int) -> None:
    """
    Write the 8-byte sync watermark into the frame buffer in-place.

    The watermark occupies 4 consecutive YUYV bytes starting at the
    macro-pixel for pixel column (4060 + frame_index*4) in the last row.
    Frames 0-8 are marked; frame 9 would be out-of-bounds and is skipped.
    """
    start_col = MARKER_COL_BASE + frame_index * MARKER_COL_STEP
    if start_col + 3 >= W:
        return                          # out of bounds – frame 9

    # In a YUYV buffer each row is W*2 bytes.
    # Macro-pixel index for an even column c: c // 2
    # Its byte offset within the row: (c // 2) * 4
    macro_byte = (start_col // 2) * 4
    row_offset = (H - 1) * W * 2
    off = row_offset + macro_byte

    frame[off : off + 8] = MARKER_BYTES


def make_header(W: int, H: int) -> bytes:
    """
    Build the 100-byte .7TH file header.

    Observed layout:
      0- 7  magic  "7th \\n\\n\\0\\0"
      8-11  uint32 LE  8   (source bit-depth; always 8 for 7th-422 output samples)
     12-15  uint32 LE  W
     16-19  uint32 LE  H
     20-21  uint16 LE  W
     22-23  uint16 LE  1
     24-25  uint16 LE  W
     26-27  uint16 LE  0
     28-99  alternating 0x00 0x7f  (72 bytes padding)
    """
    hdr = bytearray(HEADER_SIZE)
    hdr[0:8] = MAGIC
    struct.pack_into("<I", hdr,  8, 8)
    struct.pack_into("<I", hdr, 12, W)
    struct.pack_into("<I", hdr, 16, H)
    struct.pack_into("<H", hdr, 20, W)
    struct.pack_into("<H", hdr, 22, 1)
    struct.pack_into("<H", hdr, 24, W)
    struct.pack_into("<H", hdr, 26, 0)
    for i in range(28, HEADER_SIZE, 2):
        hdr[i]     = 0x00
        hdr[i + 1] = 0x7f
    return bytes(hdr)


# Incremental build manifest - Just snag the new files and convert them

MANIFEST_FILE = ".make7th_state.json"


class Manifest:
    """
    Tracks mtime + size of every source frame that has been encoded.
    Stored as JSON in the output directory.

    Keys are bare filenames (no directory component) so the manifest
    remains valid when the NFS mount point differs between machines.
    """

    def __init__(self, output_dir):
        self._path   = Path(output_dir) / MANIFEST_FILE
        self._data   = {}
        self._dirty  = False
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (ValueError, OSError):
                self._data = {}   # corrupt or unreadable – start fresh

    def _key(self, path):
        return Path(path).name

    def is_changed(self, path):
        """Return True if the file is new or differs from the recorded mtime/size."""
        key = self._key(path)
        if key not in self._data:
            return True
        try:
            st = os.stat(path)
        except OSError:
            return True     # file disappeared – treat as changed
        stored = self._data[key]
        return st.st_mtime != stored["mtime"] or st.st_size != stored["size"]

    def record(self, path):
        """Store the current mtime + size for a source file."""
        try:
            st = os.stat(path)
            self._data[self._key(path)] = {
                "mtime": st.st_mtime,
                "size":  st.st_size,
            }
            self._dirty = True
        except OSError:
            pass

    def save(self):
        if self._dirty:
            self._path.write_text(json.dumps(self._data, indent=2))
            self._dirty = False


# Frame discovery and grouping 

# Matches the trailing frame number in names like "show.1080p.00042.png"
_FRAME_RE = re.compile(r'^(.*\D)(\d+)$')


def parse_frame_number(path):
    """
    Return (frame_number, digit_width, prefix) parsed from the stem of path,
    or (None, None, None) if the stem has no trailing digits.

    e.g. "show.1080p.00042" -> (42, 5, "show.1080p.")
    """
    m = _FRAME_RE.match(path.stem)
    if not m:
        return None, None, None
    return int(m.group(2)), len(m.group(2)), m.group(1)


def output_path(prefix, digit_width, chunk_start, output_dir):
    """Build the output .7th path for a chunk."""
    name = "{}{:0{}d}.7th".format(prefix, chunk_start, digit_width)
    return Path(output_dir) / name


def collect_frames(input_paths, range_start, range_end):
    """
    Filter input_paths to PNGs with parseable frame numbers, optionally
    within [range_start, range_end].  Returns a sorted list of
    (frame_num, digit_width, prefix, path) tuples.
    """
    frames = []
    for p in input_paths:
        if p.suffix.lower() not in (".png", ".exr"):
            continue
        num, width, prefix = parse_frame_number(p)
        if num is None:
            continue
        if range_start is not None and not (range_start <= num <= range_end):
            continue
        frames.append((num, width, prefix, p))
    frames.sort(key=lambda t: t[0])
    return frames


def group_into_chunks(frames):
    """
    Group frames into 10-frame chunks based on their sequence number.

    Chunk key  = floor(frame_num / 10) * 10
    Position within chunk = frame_num % 10  (0-9)

    Returns dict: chunk_start -> {position: (frame_num, digit_width, prefix, path)}
    """
    chunks = {}
    for entry in frames:
        num = entry[0]
        cs  = (num // NUM_FRAMES) * NUM_FRAMES
        pos = num % NUM_FRAMES
        if cs not in chunks:
            chunks[cs] = {}
        chunks[cs][pos] = entry
    return chunks


def detect_size(path):
    """Return (width, height) by reading the image header."""
    if path.suffix.lower() == ".exr":
        if not HAS_OPENEXR:
            raise RuntimeError("OpenEXR not installed; cannot read EXR dimensions.")
        f  = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        return dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    with Image.open(path) as img:
        return img.size   # (width, height)


# MAIN 

def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG frame sequences to .7TH-422 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Input frames: a directory, a quoted glob pattern, or a single file. "
             "e.g.  /frames/  or  '/frames/*.png'",
    )
    parser.add_argument(
        "output",
        help="Output directory, or directory/basename for custom output filenames.",
    )
    parser.add_argument(
        "-r", "--range", dest="frame_range", default=None,
        metavar="START-END",
        help="Only encode frames with sequence numbers in this range (e.g. 100-1000)",
    )
    parser.add_argument(
        "--renum", default="0", metavar="N|false",
        help="Starting output frame number (default: 0). "
             "Pass 'false' to keep original input frame numbers in output filenames.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-encode all chunks even if source frames are unchanged.",
    )
    args = parser.parse_args()

    # Resolve output: directory or directory/basename 
    raw_out = args.output
    out_arg = Path(raw_out)
    if raw_out.endswith(("/", "\\")) or out_arg.is_dir():
        output_dir   = out_arg
        out_basename = None
    else:
        output_dir   = out_arg.parent if out_arg.parent != Path(".") else Path(".")
        out_basename = out_arg.name

    # ── Expand input to a list of Paths (all in Python, no shell limits) ──────
    inp = args.input
    inp_path = Path(inp)
    if inp_path.is_dir():
        # Directory: collect all PNGs and EXRs inside it
        input_paths = sorted(
            p for p in inp_path.iterdir()
            if p.suffix.lower() in (".png", ".exr")
        )
    elif any(c in inp for c in ("*", "?", "[")):
        # Quoted glob pattern: expand with Python
        import glob as _glob
        input_paths = [Path(p) for p in sorted(_glob.iglob(inp))]
    else:
        # Single file
        input_paths = [inp_path]

    if not input_paths:
        print("Error: no input files found.", file=sys.stderr)
        sys.exit(1)

    # Parse --renum  (decides if output 7th files shoudl e numbered starting at 0 or the input number)
    if args.renum.lower() == "false":
        renum_start = None           # keep original frame numbers
    else:
        try:
            renum_start = int(args.renum)
        except ValueError:
            print(f"Error: --renum must be a number or 'false', got '{args.renum}'",
                  file=sys.stderr)
            sys.exit(1)

    # Parse optional frame range 
    range_start = range_end = None
    if args.frame_range:
        try:
            parts = args.frame_range.split("-")
            range_start, range_end = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print(f"Error: --range must be START-END (e.g. 100-1000), got '{args.frame_range}'",
                  file=sys.stderr)
            sys.exit(1)

    # Discover and group frames 
    frames = collect_frames(input_paths, range_start, range_end)
    if not frames:
        print("Error: no matching PNG frames found.", file=sys.stderr)
        sys.exit(1)

    chunks = group_into_chunks(frames)

    # Auto-detect frame dimensions from the first frame.
    W, H = detect_size(frames[0][3])
    expected_size = HEADER_SIZE + NUM_FRAMES * W * H * 2

    # Output naming: custom basename overrides input-derived prefix.
    _, digit_width, in_prefix, _ = frames[0]
    if out_basename is not None:
        prefix = out_basename + "."
    else:
        prefix = in_prefix

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(output_dir)

    print(f"Frames : {len(frames)} PNGs -> {len(chunks)} .7th files  ({W}x{H}, {expected_size:,} bytes each)")
    if range_start is not None:
        print(f"Range  : {range_start}-{range_end}")
    if renum_start is not None:
        print(f"Renum  : output starts at {renum_start:05d}")
    if args.force:
        print(f"Mode   : forced (re-encoding all chunks)")
    print(f"Output : {output_dir}/  (prefix: {prefix})")
    print()

    total         = len(chunks)
    encoded_count = 0
    skipped_count = 0

    for idx, (cs, chunk) in enumerate(sorted(chunks.items()), 1):
        out_num  = renum_start + (idx - 1) * NUM_FRAMES if renum_start is not None else cs
        out_path = output_path(prefix, digit_width, out_num, output_dir)

        # A chunk needs encoding if forced or any source frame in the chunk
        # has changed (or is absent) since it was last recorded in the manifest.
        source_paths = [entry[3] for entry in chunk.values()]

        if not args.force and not any(manifest.is_changed(p) for p in source_paths):
            print(f"  [{idx}/{total}]  {out_path.name}  (unchanged, skipped)")
            skipped_count += 1
            continue
        # do encoding
        filled = NUM_FRAMES - len(chunk)
        status = f"({len(chunk)}/10 frames" + (f", {filled} filled)" if filled else ")")
        print(f"  [{idx}/{total}]  {out_path.name}  {status}")

        with out_path.open("wb") as out:
            out.write(make_header(W, H))

            for i in range(NUM_FRAMES):
                if i in chunk:
                    _, _, _, img_path = chunk[i]
                    rgb   = load_frame(img_path)
                    yuyv  = rgb_float_to_yuyv(rgb)
                    frame = bytearray(yuyv.tobytes())
                else:
                    frame = make_fill_frame(W, H)

                stamp_marker(frame, W, H, i)
                out.write(frame)

        actual = out_path.stat().st_size
        if actual != expected_size:
            print(f"    WARNING: size mismatch - got {actual}, expected {expected_size}",
                  file=sys.stderr)

        # Record all source frames for this chunk in the manifest.
        for p in source_paths:
            manifest.record(p)
        manifest.save()
        encoded_count += 1

    print(f"\nDone.  Encoded: {encoded_count}  Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
