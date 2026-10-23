"""Draw the asymmetric historical-versus-updated routing architecture."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "siamese_router_architecture.png"
INK = "#1F2933"
LINE = "#52606D"
BLUE = "#DCEEFF"
GREEN = "#DDF3E4"
AMBER = "#FFF0CC"
VIOLET = "#ECE5F7"
GREY = "#F3F5F7"


def font(size, bold=False, italic=False):
    name = "timesbd.ttf" if bold else "timesi.ttf" if italic else "times.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def centred(draw, xy, value, size=30, bold=False, italic=False, fill=INK):
    draw.text(xy, value, font=font(size, bold, italic), fill=fill, anchor="mm")


def box(draw, xy, title, lines=(), fill="white"):
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=INK, width=3)
    x1, y1, x2, _ = xy
    centred(draw, ((x1 + x2) / 2, y1 + 43), title, 31, bold=True)
    for row, line in enumerate(lines):
        centred(draw, ((x1 + x2) / 2, y1 + 92 + row * 37), line, 25)


def arrow(draw, start, end, label=None):
    draw.line((start, end), fill=LINE, width=5)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        tip = [(ex, ey), (ex - sign * 18, ey - 10), (ex - sign * 18, ey + 10)]
    else:
        sign = 1 if ey > sy else -1
        tip = [(ex, ey), (ex - 10, ey - sign * 18), (ex + 10, ey - sign * 18)]
    draw.polygon(tip, fill=LINE)
    if label:
        draw.text(
            ((sx + ex) / 2, (sy + ey) / 2 - 14),
            label,
            font=font(23, italic=True),
            fill=LINE,
            anchor="ms",
            stroke_width=6,
            stroke_fill="white",
        )


def main():
    image = Image.new("RGB", (1800, 930), "white")
    draw = ImageDraw.Draw(image)

    box(draw, (45, 325, 290, 520), "Input image", ("shared by", "both branches"), fill=GREY)
    box(draw, (380, 95, 830, 345), "Historical saliency ViT", ("frozen checkpoint", "old logits + patch map"), fill=BLUE)
    box(draw, (380, 585, 830, 835), "Updated saliency ViT", ("2010 adaptation", "new logits + patch map"), fill=GREEN)
    arrow(draw, (290, 400), (380, 220), "same image")
    arrow(draw, (290, 450), (380, 710), "same image")

    box(draw, (925, 275, 1325, 655), "Observable comparison", ("saliency JS divergence", "confidence gain", "entropy reduction", "class-evidence gain"), fill=AMBER)
    arrow(draw, (830, 220), (925, 365), "old evidence")
    arrow(draw, (830, 710), (925, 565), "new evidence")

    box(draw, (1415, 175, 1755, 460), "Transparent policy", ("logistic score", "saliency floor", "route threshold"), fill=VIOLET)
    arrow(draw, (1325, 430), (1415, 320))
    box(draw, (1415, 600, 1755, 830), "Recorded decision", ("selected prediction", "features + probability", "maps + thresholds"), fill=GREY)
    arrow(draw, (1585, 460), (1585, 600), "select branch")
    centred(draw, (900, 895), "The policy compares named signals; it does not learn an opaque pair embedding.", 27, italic=True, fill=LINE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
