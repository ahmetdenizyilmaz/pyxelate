#!/usr/bin/env python3

import argparse
import sys
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import Pyx, Pal, Vid, list_styles
from .mcp_server import run_stdio_server


def _imread(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _imsave(path: str, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def get_model(args: argparse.Namespace):
    return Pyx(
        height=args.height,
        width=args.width,
        factor=args.factor,
        upscale=args.upscale,
        depth=args.depth,
        palette=args.palette,
        dither=args.dither,
        sobel=args.sobel,
        alpha=args.alpha,
        svd=not args.nosvd,
        style=args.style,
    )


def convert(args: argparse.Namespace):
    pyx = get_model(args)
    image = _imread(args.INFILE)
    pyx.fit(image)
    new_image = pyx.transform(image)
    _imsave(args.OUTFILE, new_image)


def convert_sequence(args: argparse.Namespace):
    p = Path(args.INFILE)
    files = str(p.name)
    assert "%d" in files, "Input filename for sequences must contain %d to denote ordering!"
    candidates = [str(c.name) for c in list(p.parent.resolve().glob(f"*{p.suffix}"))]
    images, names, i = [], [], 0
    while True:
        check = [
            bool(re.search(r"\b" + files.replace("%d", f"[0]*?{i}") + r"\b", c))
            for c in candidates
        ]
        if np.any(check):
            name = p.parent.resolve() / candidates[np.argmax(check)]
            names.append(name)
            images.append(_imread(str(name)))
        elif i > 1 or images:
            break
        i += 1
    total = len(images)
    assert total, f"No images found in {p.parent} that satisfied '{files}'"
    five_percent = max(1, total // 20)

    p = Path(args.OUTFILE)
    files = str(p.name)
    assert "%d" in files, "Output filename for sequences must contain %d to denote ordering!"
    for i, (image, key) in enumerate(
        Vid(
            images,
            pad=args.pad,
            sobel=args.sobel,
            keyframe=args.keyframe,
            sensitivity=args.sensitivity,
        )
    ):
        if i == 0 or (key and args.refit):
            pyx = get_model(args)
            pyx.fit(image)
        out = pyx.transform(image, frame_index=i)
        file = str(p.parent / files.replace("%d", str(i)))
        _imsave(file, out)
        if not args.quiet and i % five_percent == 0:
            print(f"Finished {i+1} out of {total} ({round((i + 1) / total * 100)}%)")


def main():
    if "--mcp" in sys.argv:
        run_stdio_server()
        return

    style_ids = [s["id"] for s in list_styles()]
    parser = argparse.ArgumentParser(
        description="Pyxelate: Numba-accelerated pixel art & 12-style Reel visual engine + MCP server."
    )
    parser.add_argument("INFILE", type=str, nargs="?", help="Input image filename.")
    parser.add_argument("OUTFILE", type=str, nargs="?", help="Output image filename.")
    parser.add_argument("--mcp", action="store_true", help="Run as an MCP stdio server.")
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        choices=style_ids,
        help=f"Optional Reel visual style ({', '.join(style_ids)}).",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--palette", type=str, default="8")
    parser.add_argument(
        "--dither",
        type=str,
        default="none",
        choices=["none", "naive", "bayer", "floyd", "atkinson"],
    )
    parser.add_argument("--sobel", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--nosvd", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--keyframe", type=float, default=0.3)
    parser.add_argument("--sensitivity", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mcp:
        run_stdio_server()
        return
    if not args.INFILE or not args.OUTFILE:
        parser.print_usage()
        sys.exit(1)

    try:
        palette = int(args.palette)
    except ValueError:
        palette = Pal[args.palette]
    args.palette = palette

    if args.sequence:
        convert_sequence(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
