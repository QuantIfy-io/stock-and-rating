"""16:9 LinkedIn cover — type only. An upgrade to Buy, two tickers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "upgrade-rise-cover.png"

W, H = 1920, 1080
SCALE = 2
SW, SH = W * SCALE, H * SCALE

BG = (9, 10, 13)
TEXT = (246, 247, 249)
MUTED = (150, 156, 166)
AMBER = (255, 146, 64)
BLUE = (110, 166, 255)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def _hn(size: int, *, index: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(HN, size * SCALE, index=index)
    except OSError:
        return ImageFont.load_default()


def _width(text: str, font) -> int:
    box = font.getbbox(text)
    return box[2] - box[0]


def _height(text: str, font) -> int:
    box = font.getbbox(text)
    return box[3] - box[1]


def _tracked(draw, x, y, text, font, fill, gap) -> None:
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += _width(char, font) + gap


def _grain() -> Image.Image:
    noise = np.random.default_rng(2).integers(0, 10, (SH, SW), dtype=np.uint8)
    arr = np.dstack([noise, noise, noise, np.full((SH, SW), 12, dtype=np.uint8)])
    return Image.fromarray(arr, "RGBA")


def build_cover() -> Image.Image:
    base = Image.new("RGB", (SW, SH), BG).convert("RGBA")
    base.alpha_composite(_grain())
    draw = ImageDraw.Draw(base)

    thin = _hn(54, index=12)
    huge = _hn(108, index=1)
    move = _hn(64, index=1)
    small = _hn(16, index=10)

    x = int(SW * 0.08)

    _tracked(draw, x, int(SH * 0.12), "2 NOV", small, MUTED, 5 * SCALE)

    y = int(SH * 0.22)
    draw.text((x, y), "Upgrade", font=thin, fill=TEXT)
    y += _height("Upgrade", thin) + 2 * SCALE
    draw.text((x, y), "to Buy", font=huge, fill=TEXT)
    y += _height("to Buy", huge) + int(SH * 0.04)
    draw.rectangle([x, y, x + int(SW * 0.07), y + 3 * SCALE], fill=AMBER)

    y = int(SH * 0.64)
    gap = int(SH * 0.11)
    for change, color in (("+8%", AMBER), ("−3.5%", BLUE)):
        draw.text((x, y), change, font=move, fill=color)
        y += gap

    _tracked(
        draw,
        x,
        int(SH * 0.88),
        "THE SESSION MOVES. THE NEXT WEEK DOES NOT.",
        small,
        MUTED,
        3 * SCALE,
    )
    return base.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)


def main() -> None:
    cover = build_cover()
    cover.save(OUT, quality=96, subsampling=0)
    print(f"Wrote {OUT} ({cover.size[0]}x{cover.size[1]})")


if __name__ == "__main__":
    main()
