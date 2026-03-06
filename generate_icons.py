"""
AURA Icon Generator
Generates icon16.png, icon48.png, icon128.png with the AURA geometric "A" mark.
Usage: python generate_icons.py
"""

from PIL import Image, ImageDraw, ImageFilter
import math, os

def draw_a_mark(draw, size, color=(255, 255, 255), line_width=None):
    """Draw the AURA geometric 'A' lettermark."""
    pad = size * 0.13
    lw = line_width or max(2, round(size / 8))

    top  = (size / 2,        pad)
    bl   = (pad,             size - pad)
    br   = (size - pad,      size - pad)

    # Crossbar at 62% down the legs
    t = 0.62
    cx_l = top[0] + (bl[0] - top[0]) * t
    cx_r = top[0] + (br[0] - top[0]) * t
    cy   = top[1] + (bl[1] - top[1]) * t

    # Legs
    draw.line([top, bl], fill=color, width=lw)
    draw.line([top, br], fill=color, width=lw)
    # Crossbar
    draw.line([(cx_l, cy), (cx_r, cy)], fill=color, width=lw)


def create_icon(size, output_path):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded-rect background — vivid purple gradient approximated as solid
    radius = max(3, size // 5)
    # Use a rich purple (#4c1d95) visible at all sizes
    draw.rounded_rectangle([0, 0, size - 1, size - 1],
                           radius=radius,
                           fill=(76, 29, 149, 255))

    if size >= 32:
        # Purple glow layer behind the A
        glow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        gd   = ImageDraw.Draw(glow)
        lw_g = max(3, size // 5)
        draw_a_mark(gd, size, color=(255, 255, 255, 80), line_width=lw_g)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=size // 9))
        img  = Image.alpha_composite(img, glow)
        draw = ImageDraw.Draw(img)

    # Primary mark — pure white, bold
    lw = max(2, round(size / 8))
    draw_a_mark(draw, size, color=(255, 255, 255, 255), line_width=lw)

    if size >= 128:
        # Extra outer ring glow for large icon
        ring = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        rd   = ImageDraw.Draw(ring)
        c    = size // 2
        r    = int(size * 0.44)
        rd.ellipse([c - r, c - r, c + r, c + r],
                   outline=(124, 58, 237, 40), width=2)
        ring = ring.filter(ImageFilter.GaussianBlur(radius=6))
        img  = Image.alpha_composite(img, ring)

    # Save as RGB PNG (no alpha needed for extension icons)
    final = Image.new('RGB', (size, size), (0, 0, 0))
    final.paste(img, mask=img.split()[3])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.save(output_path, 'PNG')
    print(f"  icon{size}.png  -> {output_path}")


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(__file__), 'extension', 'icons')
    print("Generating AURA icons...")
    create_icon(16,  os.path.join(base, 'icon16.png'))
    create_icon(48,  os.path.join(base, 'icon48.png'))
    create_icon(128, os.path.join(base, 'icon128.png'))
    print("Done.")
