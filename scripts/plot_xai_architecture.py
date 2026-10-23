"""Draw the saliency-guided ViT as a legible publication figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "xai_vit_architecture.png"
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
    centred(draw, ((x1 + x2) / 2, y1 + 42), title, 32, bold=True)
    for row, line in enumerate(lines):
        centred(draw, ((x1 + x2) / 2, y1 + 91 + row * 38), line, 26)


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
            ((sx + ex) / 2, (sy + ey) / 2 - 15),
            label,
            font=font(23, italic=True),
            fill=LINE,
            anchor="ms",
            stroke_width=6,
            stroke_fill="white",
        )


def main():
    image = Image.new("RGB", (1800, 920), "white")
    draw = ImageDraw.Draw(image)

    centred(draw, (900, 42), "Prediction path", 34, bold=True)
    box(draw, (45, 100, 285, 265), "Image", ("RGB, 224 × 224",), fill=GREY)
    box(draw, (350, 85, 700, 280), "DINOv2-small", ("one class token", "256 patch tokens"), fill=BLUE)
    box(draw, (765, 85, 1115, 280), "Saliency gate", ("one score per patch", "normalised weights"), fill=AMBER)
    box(draw, (1180, 70, 1540, 295), "Evidence pooling", ("selected patch sum", "complement patch sum"), fill=GREEN)
    box(draw, (1600, 100, 1760, 265), "Head", ("class", "logits"), fill=VIOLET)

    arrow(draw, (285, 182), (350, 182))
    arrow(draw, (700, 182), (765, 182), "patch tokens")
    arrow(draw, (1115, 182), (1180, 182), "weights")
    arrow(draw, (1540, 182), (1600, 182))

    draw.rounded_rectangle((95, 400, 1705, 850), radius=18, fill="#FAFBFC", outline="#A7B0BA", width=3)
    centred(draw, (900, 444), "Training signals", 34, bold=True)
    box(draw, (150, 520, 595, 720), "Classification", ("cross-entropy", "on selected evidence"), fill=BLUE)
    box(draw, (680, 520, 1125, 720), "Faithfulness", ("selected evidence should", "beat its complement"), fill=AMBER)
    box(draw, (1210, 520, 1655, 720), "Temporal map pair", ("stable when old is right", "free to move when wrong"), fill=GREEN)
    centred(draw, (900, 790), "The same patch weights drive prediction, adaptation, routing, and audit.", 28, italic=True, fill=LINE)
    arrow(draw, (1680, 265), (1500, 400), "predictions and paired maps")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
