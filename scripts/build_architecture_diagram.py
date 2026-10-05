"""Build the paper's ViT-only architecture overview as a high-resolution PNG."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "transparent_router" / "architecture_overview.png"
NAVY = "#17324D"
BLUE = "#DDEBF7"
PALE = "#EEF4F8"
GREEN = "#DDEEDB"
GOLD = "#F7E8BF"
GRAY = "#5D6770"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def box(draw, xy, title, lines, fill):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=NAVY, width=3)
    x1, y1, x2, _ = xy
    draw.text(((x1 + x2) / 2, y1 + 18), title, font=font(28, True), fill=NAVY, anchor="ma")
    for offset, line in enumerate(lines):
        draw.text(((x1 + x2) / 2, y1 + 64 + offset * 34), line, font=font(22), fill="#17212B", anchor="ma")


def arrow(draw, start, end, label=None):
    draw.line((start, end), fill=NAVY, width=5)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        draw.polygon([(ex, ey), (ex - sign * 18, ey - 10), (ex - sign * 18, ey + 10)], fill=NAVY)
    else:
        sign = 1 if ey > sy else -1
        draw.polygon([(ex, ey), (ex - 10, ey - sign * 18), (ex + 10, ey - sign * 18)], fill=NAVY)
    if label:
        draw.text(((sx + ex) / 2, (sy + ey) / 2 - 16), label, font=font(19, True), fill=GRAY, anchor="ms")


def main():
    image = Image.new("RGB", (1900, 1040), "white")
    draw = ImageDraw.Draw(image)
    draw.text((950, 32), "ViT-only temporal update and transparent routing", font=font(38, True), fill=NAVY, anchor="ma")
    draw.text((950, 82), "The saliency gate changes the representation used for classification; the router compares two frozen outputs.", font=font(24), fill=GRAY, anchor="ma")

    box(draw, (40, 170, 300, 370), "Input", ["RGB image", "resize 224 × 224", "normalize channels"], PALE)
    box(draw, (370, 170, 670, 370), "Patch tokens", ["14 × 14 pixel patches", "16 × 16 token grid", "256 spatial tokens"], BLUE)
    box(draw, (740, 170, 1080, 370), "DINOv2-small", ["12 transformer blocks", "global self-attention", "384 dimensions"], BLUE)
    arrow(draw, (300, 270), (370, 270))
    arrow(draw, (670, 270), (740, 270))

    box(draw, (1160, 125, 1505, 315), "Historical branch", ["frozen reference ViT", "class token + patches", "old logits and map"], PALE)
    box(draw, (1160, 350, 1505, 555), "Updated XAI branch", ["adapted later-window ViT", "learned patch gates", "new logits and map"], GREEN)
    arrow(draw, (1080, 245), (1160, 220), "copy")
    arrow(draw, (1080, 295), (1160, 445), "adapt")

    box(draw, (370, 500, 705, 735), "Embedded XAI gate", ["gᵢ = sigmoid(linear(hᵢ))", "sᵢ = gᵢ / Σⱼgⱼ", "v = Σᵢsᵢhᵢ", "classify [CLS ; v]"], GREEN)
    arrow(draw, (910, 370), (705, 500), "patch tokens")
    arrow(draw, (705, 615), (1160, 455), "weighted evidence")

    box(draw, (1580, 205, 1860, 500), "Transparent router", ["saliency JS change", "confidence gain", "entropy reduction", "class-evidence gain", "logistic probability", "+ explicit thresholds"], GOLD)
    arrow(draw, (1505, 220), (1580, 300))
    arrow(draw, (1505, 455), (1580, 405))

    box(draw, (1430, 720, 1870, 945), "Logged decision", ["retain historical prediction", "or select updated prediction", "with features, probability,", "thresholds, and maps"], GOLD)
    arrow(draw, (1720, 500), (1650, 720), "eligible route")

    draw.text((42, 990), "No CNN branch is used. Both predictive branches use the same DINOv2 Vision Transformer family.", font=font(22, True), fill=NAVY)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
