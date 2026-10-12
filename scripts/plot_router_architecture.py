"""Build a restrained, publication-style architecture schematic."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "figures"
    / "siamese_router_architecture.png"
)
INK = "#202020"
MID = "#626262"
LINE = "#4A4A4A"
LANE = "#F3F3F3"
XAI = "#E9E9E9"


def font(size, bold=False, italic=False):
    if bold:
        name = "timesbd.ttf"
    elif italic:
        name = "timesi.ttf"
    else:
        name = "times.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def text_center(draw, xy, value, size=22, bold=False, fill=INK):
    draw.text(xy, value, font=font(size, bold=bold), fill=fill, anchor="mm")


def box(draw, xy, title, lines=(), fill="white", width=2):
    draw.rectangle(xy, fill=fill, outline=INK, width=width)
    x1, y1, x2, _ = xy
    text_center(draw, ((x1 + x2) / 2, y1 + 31), title, 24, True)
    for row, line in enumerate(lines):
        text_center(draw, ((x1 + x2) / 2, y1 + 69 + row * 29), line, 19)


def arrow(draw, start, end, label=None):
    draw.line((start, end), fill=LINE, width=3)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        draw.polygon([(ex, ey), (ex - sign * 14, ey - 8), (ex - sign * 14, ey + 8)], fill=LINE)
    else:
        sign = 1 if ey > sy else -1
        draw.polygon([(ex, ey), (ex - 8, ey - sign * 14), (ex + 8, ey - sign * 14)], fill=LINE)
    if label:
        draw.text(((sx + ex) / 2, (sy + ey) / 2 - 11), label,
                  font=font(17, italic=True), fill=MID, anchor="ms",
                  stroke_width=5, stroke_fill="white")


def main():
    image = Image.new("RGB", (1900, 865), "white")
    draw = ImageDraw.Draw(image)

    draw.text((45, 28), "Shared image representation", font=font(24, True), fill=INK)
    draw.line((45, 63, 900, 63), fill=INK, width=2)
    box(draw, (45, 105, 260, 245), "Input", ("RGB", "224 × 224"))
    box(draw, (325, 105, 555, 245), "Patchify", ("14 × 14 patches", "256 tokens"))
    box(draw, (620, 105, 900, 245), "DINOv2-small", ("self-attention", "384-d tokens"), fill=LANE)
    arrow(draw, (260, 175), (325, 175))
    arrow(draw, (555, 175), (620, 175))

    draw.rectangle((980, 70, 1395, 360), fill=LANE, outline="#B5B5B5", width=2)
    draw.text((1000, 84), "Historical path", font=font(23, True), fill=INK)
    box(draw, (1015, 135, 1360, 300), "Historical XAI ViT",
        ("frozen transformer", "frozen saliency gate", "old logits + patch map"))

    draw.rectangle((980, 410, 1395, 700), fill="#F5F5F5", outline="#A5A5A5", width=2)
    draw.text((1000, 424), "Adapted path", font=font(23, True), fill=INK)
    box(draw, (1015, 475, 1360, 640), "Updated XAI ViT",
        ("last 3 blocks adapted", "learned saliency gate", "new logits + patch map"), fill=XAI)
    arrow(draw, (900, 155), (1015, 205), "clone + freeze")
    arrow(draw, (900, 215), (1015, 545), "temporal adaptation")

    box(draw, (1465, 210, 1845, 435), "Observable comparison",
        ("saliency JS divergence", "confidence gain", "entropy reduction", "class-evidence gain"))
    arrow(draw, (1360, 220), (1465, 280), "old evidence")
    arrow(draw, (1360, 555), (1465, 365), "new evidence")

    box(draw, (1465, 535, 1845, 740), "Transparent route",
        ("standardized logistic score", "model disagreement", "saliency floor", "probability threshold"), fill=XAI)
    arrow(draw, (1655, 435), (1655, 535))

    box(draw, (620, 700, 1100, 840), "Recorded output",
        ("historical or updated prediction", "features, thresholds, probability, and maps"))
    arrow(draw, (1465, 650), (1100, 775), "selected branch")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
