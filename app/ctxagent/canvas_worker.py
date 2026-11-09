"""Run a canvas_script in its own process: python -m ctxagent.canvas_worker IN.png SCRIPT.py OUT.png

The script sees `img` (the current canvas, a Pillow RGBA image), `draw` (ImageDraw on
it), `W`, `H`, `font(size)`, the Pillow modules and `np` when numpy is installed. Whatever
`img` holds at the end becomes the next canvas version; print() output goes back to the
model. A separate process so that a hanging or crashing script cannot take the app down.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: canvas_worker IN.png SCRIPT.py OUT.png", file=sys.stderr)
        return 2
    src, script, dst = argv[1:]
    from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps
    from ctxagent.canvas import font
    img = Image.open(src).convert("RGBA")
    ns = {"img": img, "draw": ImageDraw.Draw(img), "W": img.width, "H": img.height, "font": font,
          "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont, "ImageFilter": ImageFilter,
          "ImageOps": ImageOps, "ImageChops": ImageChops, "ImageColor": ImageColor, "__name__": "canvas_script"}
    try:
        import numpy
        ns["np"] = numpy
    except ImportError:
        pass
    code = Path(script).read_text(encoding="utf-8")
    exec(compile(code, "canvas_script", "exec"), ns)
    out = ns.get("img")
    if not isinstance(out, Image.Image):
        print("canvas_script: `img` must hold a Pillow image at the end", file=sys.stderr)
        return 3
    out.convert("RGBA").save(dst, format="PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
