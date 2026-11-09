"""Draw the CTXAgent icon with Pillow: a context window on a rounded gradient tile.

    python packaging/make_icon.py            -> packaging/ctxagent.ico (+ .png)

The mark: angle brackets (the context window) around three lines of content. The
top line is short and amber, the summary a compaction leaves of the older history;
the two white lines below are the recent work kept verbatim. web/index.html draws
the same mark as the SVG symbol `i-ctx` for the UI.

Everything is computed from a few strokes on a 1024 px canvas (no fonts, no
downloaded artwork) and downsampled per icon size, so the result is crisp from
16 px (taskbar) to 256 px (installer, Explorer large icons).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
TOP, BOTTOM = (44, 62, 140), (20, 150, 160)   # indigo -> teal
GLYPH = (255, 255, 255)
SUMMARY = (255, 200, 87)                      # amber


def _line(p0, p1, steps=600):
    for i in range(steps + 1):
        t = i / steps
        yield (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def _stroke(draw, points, width, fill):
    """A round-capped thick path: stamp discs densely along it."""
    r = width / 2
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _polyline(draw, points, width, fill):
    for a, b in zip(points, points[1:]):
        _stroke(draw, _line(a, b), width, fill)


def tile() -> Image.Image:
    gradient = Image.new("RGB", (S, S))
    px = gradient.load()
    for y in range(S):
        for x in range(S):
            t = min(1.0, max(0.0, (0.65 * y + 0.35 * x) / S))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((28, 28, S - 28, S - 28), radius=230, fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(gradient, (0, 0), mask)
    return out


def glyph(bracket: int = 104, bar: int = 82, summary: bool = True, small: bool = False) -> Image.Image:
    """The brackets and the verbatim lines in white, the summary line in amber (white
    too for the shadow pass). `small`: for 16-24 px, two thicker lines with more space
    between them, since three run together at that size."""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if small:
        bracket, bar = 132, 128
    _polyline(d, [(352, 282), (206, 512), (352, 742)], bracket, GLYPH)
    _polyline(d, [(672, 282), (818, 512), (672, 742)], bracket, GLYPH)
    amber = SUMMARY if summary else GLYPH
    if small:
        _stroke(d, _line((440, 404), (540, 404)), bar, amber)
        _stroke(d, _line((440, 624), (592, 624)), bar, GLYPH)
        return layer
    _stroke(d, _line((432, 512), (604, 512)), bar, GLYPH)
    _stroke(d, _line((432, 667), (604, 667)), bar, GLYPH)
    _stroke(d, _line((432, 357), (522, 357)), bar, amber)
    return layer


def master(small: bool = False) -> Image.Image:
    base = tile()
    shadow = glyph(120, 98, summary=False, small=small).split()[3].filter(ImageFilter.GaussianBlur(18))
    dark = Image.new("RGBA", (S, S), (10, 20, 50, 0))
    dark.putalpha(shadow.point(lambda a: a * 0.45))
    base.alpha_composite(dark, (10, 16))
    base.alpha_composite(glyph(small=small))
    return base


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    big, small = master(), master(small=True)
    frames = [(small if n <= 24 else big).resize((n, n), Image.LANCZOS) for n in SIZES]
    ico = out_dir / "ctxagent.ico"
    frames[-1].save(ico, format="ICO", sizes=[(n, n) for n in SIZES],
                    append_images=frames[:-1])
    big.resize((512, 512), Image.LANCZOS).save(out_dir / "ctxagent.png")
    print(f"wrote {ico} ({ico.stat().st_size} bytes) and ctxagent.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
