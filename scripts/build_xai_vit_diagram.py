"""Build a publication-style schematic of the embedded saliency-guided ViT."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "transparent_router" / "xai_vit_forward.png"
INK = "#202020"
MID = "#606060"
LINE = "#4A4A4A"
PALE = "#E9E9E9"
LIGHT = "#F3F3F3"


def font(size, bold=False, italic=False):
    name = "timesbd.ttf" if bold else "timesi.ttf" if italic else "times.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def center(draw, xy, value, size=21, bold=False, italic=False, fill=INK):
    draw.text(xy, value, font=font(size, bold, italic), fill=fill, anchor="mm")


def box(draw, xy, title, lines=(), fill="white"):
    draw.rectangle(xy, fill=fill, outline=INK, width=2)
    x1, y1, x2, _ = xy
    center(draw, ((x1 + x2) / 2, y1 + 31), title, 23, bold=True)
    for row, line in enumerate(lines):
        center(draw, ((x1 + x2) / 2, y1 + 68 + row * 27), line, 18)


def arrow(draw, start, end, label=None):
    draw.line((start, end), fill=LINE, width=3)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        draw.polygon([(ex, ey), (ex - sign * 13, ey - 7), (ex - sign * 13, ey + 7)], fill=LINE)
    else:
        sign = 1 if ey > sy else -1
        draw.polygon([(ex, ey), (ex - 7, ey - sign * 13), (ex + 7, ey - sign * 13)], fill=LINE)
    if label:
        draw.text(((sx + ex) / 2, (sy + ey) / 2 - 10), label,
                  font=font(16, italic=True), fill=MID, anchor="ms",
                  stroke_width=5, stroke_fill="white")


def main():
    canvas = Image.new("RGB", (1900, 930), "white")
    draw = ImageDraw.Draw(canvas)
    box(draw, (45, 120, 225, 250), "Image", ("RGB, 224 × 224",))
    box(draw, (275, 120, 515, 250), "Patch tokens", ("16 × 16 grid", "256 × 384"))
    box(draw, (565, 120, 855, 250), "DINOv2-small", ("class token h_cls", "patch tokens h_i"), fill=LIGHT)
    arrow(draw, (225, 185), (275, 185))
    arrow(draw, (515, 185), (565, 185))

    draw.rectangle((910, 85, 1250, 445), fill="#FAFAFA", outline="#B0B0B0", width=2)
    draw.text((930, 98), "Evidence gate", font=font(22, bold=True), fill=INK)
    box(draw, (940, 150, 1220, 270), "Per-patch score",
        ("gᵢ = sigmoid(W LN(hᵢ))",), fill=PALE)
    box(draw, (940, 315, 1220, 405), "Normalise", ("sᵢ = gᵢ / Σⱼ gⱼ",), fill=PALE)
    arrow(draw, (855, 185), (940, 205), "patch tokens")
    arrow(draw, (1080, 270), (1080, 315))

    box(draw, (1310, 95, 1585, 235), "Selected evidence",
        ("v_sel = Σᵢ sᵢhᵢ", "16 × 16 displayed map"), fill=PALE)
    box(draw, (1310, 300, 1585, 440), "Complement evidence",
        ("v_comp = Σᵢ(1 − sᵢ)hᵢ", "normalised separately"), fill=LIGHT)
    arrow(draw, (1220, 350), (1310, 165), "sᵢ")
    arrow(draw, (1220, 370), (1310, 370), "1 − sᵢ")

    box(draw, (1650, 150, 1860, 385), "Shared head",
        ("[h_cls ; v]", "linear logits", "y_sel and y_comp"))
    arrow(draw, (1585, 165), (1650, 215), "selected")
    arrow(draw, (1585, 370), (1650, 325), "complement")
    draw.line((710, 250, 710, 485, 1755, 485, 1755, 385), fill=LINE, width=3)
    draw.polygon([(1755, 385), (1748, 398), (1762, 398)], fill=LINE)
    draw.text((1080, 478), "class token", font=font(16, italic=True), fill=MID,
              anchor="ms", stroke_width=5, stroke_fill="white")

    draw.rectangle((215, 550, 1685, 875), fill="#FAFAFA", outline="#B0B0B0", width=2)
    draw.text((240, 565), "Optimisation objectives", font=font(23, bold=True), fill=INK)
    box(draw, (260, 625, 620, 755), "Classification", ("cross entropy on y_sel",))
    box(draw, (770, 625, 1130, 755), "Faithfulness", ("true-class evidence:", "selected > complement"), fill=PALE)
    box(draw, (1280, 625, 1640, 755), "Temporal map pair", ("stability if old correct", "movement if old wrong"), fill=PALE)
    center(draw, (950, 808), "L = L_cls + λ_faith L_faith + λ_pair L_pair", 22, bold=True)
    center(draw, (950, 846), "The no-auxiliary control sets λ_faith = λ_pair = 0 while preserving the forward architecture.", 18, italic=True, fill=MID)
    arrow(draw, (1755, 385), (950, 550), "predictions, labels, and paired historical map")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.crop((0, 70, 1900, 930)).save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
