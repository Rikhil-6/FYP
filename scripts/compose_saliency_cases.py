"""Compose helpful-route audit panels from distinct drift sources."""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts") / ("timesbd.ttf" if bold else "times.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") /
        ("DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def select_helpful(audit_dir):
    with (audit_dir / "selected_cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["outcome"] == "helpful_routed" and row.get("panel")]
    if not rows:
        raise ValueError(f"No helpful routed panel in {audit_dir}")
    return max(rows, key=lambda row: (float(row["new_faithfulness_margin"]),
                                      float(row["route_probability"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", required=True,
                        help="Label=directory; repeat for each drift source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for value in args.audit:
        label, raw_path = value.split("=", 1)
        audit_dir = Path(raw_path)
        entries.append((label, audit_dir, select_helpful(audit_dir)))

    scale = 2
    width, row_height = 920 * scale, 292 * scale
    canvas = Image.new(
        "RGB", (width, 44 * scale + len(entries) * row_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    headers = ("Clean reference", "Drifted input", "Historical saliency",
               "Updated saliency", "Saliency change")
    for index, header in enumerate(headers):
        draw.text(
            ((14 + index * 180) * scale, 12 * scale),
            header,
            font=font(12 * scale, True),
            fill="#202020",
        )

    for row_index, (label, audit_dir, row) in enumerate(entries):
        top = 44 * scale + row_index * row_height
        source = Image.open(audit_dir / row["panel"]).convert("RGB")
        for panel_index in range(5):
            panel = source.crop((panel_index * 224, 30, (panel_index + 1) * 224, 254))
            panel = panel.resize((176 * scale, 176 * scale), Image.Resampling.LANCZOS)
            canvas.paste(panel, ((12 + panel_index * 180) * scale, top + 28 * scale))
        draw.text(
            (12 * scale, top + 3 * scale),
            label,
            font=font(17 * scale, True),
            fill="#202020",
        )
        summary = (f"true {row['true_class']}   historical {row['old_class']}   updated {row['new_class']}   "
                   f"route p={float(row['route_probability']):.3f}   JS={float(row['saliency_js']):.4f}   "
                   f"deletion margin={float(row['new_faithfulness_margin']):+.3f}")
        draw.text(
            (12 * scale, top + 214 * scale),
            summary,
            font=font(13 * scale),
            fill="#303030",
        )
        if row_index + 1 < len(entries):
            draw.line(
                (
                    12 * scale,
                    top + row_height - 4 * scale,
                    908 * scale,
                    top + row_height - 4 * scale,
                ),
                fill="#B0B0B0",
                width=scale,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True, dpi=(300, 300))
    print(args.output)


if __name__ == "__main__":
    main()
