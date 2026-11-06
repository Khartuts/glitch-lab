"""Generate the application icon (PNG + multi-size ICO) with Pillow."""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "glitchlab", "resources")


def make(size: int = 256) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = size // 6
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=(20, 21, 25, 255))
    try:
        font = ImageFont.load_default(size=int(size * 0.62))
    except TypeError:
        font = ImageFont.load_default()
    for dx, col in ((-size // 28, (47, 224, 255, 255)), (size // 28, (255, 61, 127, 255)), (0, (240, 240, 245, 255))):
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((size // 2 + dx, size // 2), "G", font=font, fill=col, anchor="mm")
        im.alpha_composite(layer)
    # torn scanlines
    px = im.load()
    for y0, shift in ((int(size * 0.38), size // 10), (int(size * 0.62), -size // 12)):
        for y in range(y0, y0 + max(2, size // 22)):
            row = [px[x, y] for x in range(size)]
            for x in range(size):
                px[x, y] = row[(x - shift) % size] if row[(x - shift) % size][3] else px[x, y]
    return im


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    big = make(256)
    big.save(os.path.join(OUT, "glitchlab.png"))
    big.save(os.path.join(OUT, "glitchlab.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon written to", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
