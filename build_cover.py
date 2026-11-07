"""16:9 LinkedIn cover — an upgrade mark, then two tapes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "upgrade-rise-cover.png"

W, H = 1920, 1080
SCALE = 2
SW, SH = W * SCALE, H * SCALE

BG = (10, 14, 18)
TEXT = (244, 246, 248)
AMBER = (224, 122, 61)
TEAL = (56, 168, 164)
BLUE = (76, 139, 245)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def _hn(size: int, *, index: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(HN, size * SCALE, index=index)
    except OSError:
        return ImageFont.load_default()


def _glow(size, cx, cy, radius, color, alpha) -> Image.Image:
    w, h = size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    a = np.clip(1.0 - d, 0, 1) ** 2.2
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = (a * alpha).astype(np.uint8)
    return Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=int(80 * SCALE / 2)))


def _grain(size, amount: int = 12) -> Image.Image:
    w, h = size
    noise = np.random.default_rng(7).integers(0, amount, (h, w), dtype=np.uint8)
    arr = np.dstack([noise, noise, noise, np.full((h, w), 22, dtype=np.uint8)])
    return Image.fromarray(arr, "RGBA")


def _line_width(text: str, font, word_gap: int) -> int:
    words = text.split()
    widths = [font.getbbox(word)[2] - font.getbbox(word)[0] for word in words]
    return sum(widths) + word_gap * max(0, len(words) - 1)


def _draw_words(draw, x, y, text, font, fill, word_gap) -> None:
    for word in text.split():
        draw.text((x, y), word, font=font, fill=fill)
        box = font.getbbox(word)
        x += (box[2] - box[0]) + word_gap


def _center_words(draw, y, text, font, fill, word_gap) -> None:
    tw = _line_width(text, font, word_gap)
    _draw_words(draw, (SW - tw) // 2, y, text, font, fill, word_gap)


def _path(draw, xs, ys, color, width: int) -> None:
    pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    draw.line(pts, fill=(*color, 230), width=width, joint="curve")


def build_cover() -> Image.Image:
    base = Image.new("RGB", (SW, SH), BG).convert("RGBA")
    base = Image.alpha_composite(base, _glow((SW, SH), int(SW * 0.72), int(SH * 0.38), int(SW * 0.36), AMBER, 48))
    base = Image.alpha_composite(base, _glow((SW, SH), int(SW * 0.28), int(SH * 0.72), int(SW * 0.26), TEAL, 26))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n = 70
    t = np.linspace(0, 1, n)
    xs = np.linspace(int(SW * 0.08), int(SW * 0.92), n)
    # Shared path into the upgrade, then a pop that goes flat, and a path that falls.
    split = 0.58
    before = t <= split
    amber = np.empty(n)
    blue = np.empty(n)
    amber[before] = 0.62 - 0.06 * (t[before] / split)
    blue[before] = amber[before]
    after = ~before
    u = np.clip((t - split) / (1 - split), 0, None)
    amber[after] = 0.56 - 0.10 * (1 - np.exp(-6 * u[after]))
    blue[after] = 0.56 + 0.16 * (1 - np.exp(-4 * u[after]))
    fork = np.where(t >= split)[0]
    fork = np.r_[fork[0] - 1, fork]
    _path(draw, xs, SH * amber, AMBER, 7 * SCALE)
    _path(draw, xs[fork], SH * blue[fork], BLUE, 7 * SCALE)

    notch_x = int(np.interp(split, t, xs))
    draw.line([(notch_x, int(SH * 0.42)), (notch_x, int(SH * 0.78))], fill=(*TEAL, 200), width=3 * SCALE)
    base.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(radius=max(1, SCALE // 2))))
    base = Image.alpha_composite(base, _grain((SW, SH)))

    draw = ImageDraw.Draw(base)
    thin = _hn(64, index=12)
    bold = _hn(78, index=1)
    gap = 8 * SCALE
    word_gap = 22 * SCALE
    h1 = thin.getbbox("Upgrade")[3] - thin.getbbox("Upgrade")[1]
    h2 = bold.getbbox("Rise")[3] - bold.getbbox("Rise")[1]
    block = h1 + gap + h2
    y0 = int(SH * 0.16)
    _center_words(draw, y0, "Upgrade", thin, TEXT, word_gap)
    _center_words(draw, y0 + h1 + gap, "Rise", bold, TEXT, word_gap)
    # y0 used; block kept so the type sits clear of the fork.
    _ = block

    cap = _hn(16, index=10)
    _center_words(
        draw,
        int(SH * 0.88),
        "Rivian +8%   ·   CrowdStrike −3.5%   ·   2 Nov",
        cap,
        (*AMBER, 220),
        14 * SCALE,
    )
    return base.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)


def main() -> None:
    cover = build_cover()
    cover.save(OUT, quality=96, subsampling=0)
    print(f"Wrote {OUT} ({cover.size[0]}x{cover.size[1]})")


if __name__ == "__main__":
    main()
