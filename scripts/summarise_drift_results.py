"""Validate matched drift runs and build compact publication artefacts."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DISPLAY = {
    "clean": "Clean",
    "illumination": "Illumination",
    "white_balance": "White balance",
    "sensor_noise": "Sensor noise",
    "defocus_blur": "Defocus blur",
    "motion_blur": "Motion blur",
    "jpeg_compression": "JPEG compression",
    "resolution_loss": "Resolution loss",
    "viewpoint_shift": "Viewpoint shift",
    "occlusion": "Occlusion",
    "realistic_mix": "Realistic mixture",
}


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


def load_run(path):
    path = Path(path)
    return json.loads((path / "metrics" / "future_drift.json").read_text(encoding="utf-8"))


def metric(report, scenario, year, branch, name="macro_f1"):
    return report["scenarios"][scenario][str(year)]["results"]["all"][branch][name]


def rows_for(saliency, noaux, year):
    saliency_names = list(saliency["scenarios"])
    if set(saliency_names) != set(noaux["scenarios"]):
        raise ValueError("Matched runs contain different drift scenarios")
    rows = []
    for scenario in saliency_names:
        old = metric(saliency, scenario, year, "old")
        noaux_old = metric(noaux, scenario, year, "old")
        if abs(old - noaux_old) > 1e-9:
            raise ValueError(f"Historical score mismatch for {scenario}: {old} vs {noaux_old}")
        item = {"scenario": scenario, "year": year, "old_macro_f1": old}
        for profile, report in (("saliency", saliency), ("noaux", noaux)):
            result = report["scenarios"][scenario][str(year)]["results"]["all"]
            item[f"{profile}_new_macro_f1"] = result["new"]["macro_f1"]
            item[f"{profile}_routed_macro_f1"] = result["routed"]["macro_f1"]
            item[f"{profile}_new_delta_pp"] = 100 * (result["new"]["macro_f1"] - old)
            item[f"{profile}_routed_delta_pp"] = 100 * (result["routed"]["macro_f1"] - old)
            item[f"{profile}_helpful_routes"] = result["helpful_routes"]
            item[f"{profile}_harmful_routes"] = result["harmful_routes"]
        item["auxiliary_new_effect_pp"] = 100 * (
            item["saliency_new_macro_f1"] - item["noaux_new_macro_f1"])
        item["auxiliary_routed_effect_pp"] = 100 * (
            item["saliency_routed_macro_f1"] - item["noaux_routed_macro_f1"])
        rows.append(item)
    return rows


def circle(draw, x, y, color, radius=6):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black")


def diamond(draw, x, y, color, radius=7):
    draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
                 fill=color, outline="black")


def make_figure(rows, output):
    width = 1800
    top, row_height, bottom = 150, 58, 125
    height = top + row_height * len(rows) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Matched 2014 drift evaluation", font=font(30, True), fill="#202020")
    draw.text((70, 78), "Macro F1 change relative to the same frozen historical ViT (percentage points)",
              font=font(22), fill="#4E4E4E")

    label_right, plot_left, plot_right = 300, 340, 1690
    values = [row[key] for row in rows for key in (
        "saliency_new_delta_pp", "saliency_routed_delta_pp",
        "noaux_new_delta_pp", "noaux_routed_delta_pp")]
    extent = max(0.5, max(abs(min(values)), abs(max(values))) * 1.15)
    low, high = -extent, extent

    def x(value):
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    zero = x(0)
    draw.line((zero, top - 25, zero, height - bottom + 15), fill="#707070", width=2)
    for tick_index in range(9):
        value = low + tick_index * (high - low) / 8
        xpos = x(value)
        draw.line((xpos, height - bottom + 18, xpos, height - bottom + 26), fill="#303030", width=2)
        label = "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
        draw.text((xpos, height - bottom + 34), label, font=font(17),
                  fill="#303030", anchor="ma")

    colors = {"saliency": "#0072B2", "noaux": "#999999"}
    for index, row in enumerate(rows):
        y = top + index * row_height
        if index % 2:
            draw.rectangle((55, y - 22, plot_right + 10, y + 28), fill="#F7F7F7")
        draw.text((label_right, y + 2), DISPLAY.get(row["scenario"], row["scenario"]),
                  font=font(19, bold=row["scenario"] == "realistic_mix"), fill="#202020", anchor="rm")
        draw.line((x(row["saliency_new_delta_pp"]), y - 7,
                   x(row["saliency_routed_delta_pp"]), y - 7), fill=colors["saliency"], width=3)
        circle(draw, x(row["saliency_new_delta_pp"]), y - 7, colors["saliency"])
        diamond(draw, x(row["saliency_routed_delta_pp"]), y - 7, colors["saliency"])
        draw.line((x(row["noaux_new_delta_pp"]), y + 13,
                   x(row["noaux_routed_delta_pp"]), y + 13), fill=colors["noaux"], width=3)
        circle(draw, x(row["noaux_new_delta_pp"]), y + 13, colors["noaux"], 5)
        diamond(draw, x(row["noaux_routed_delta_pp"]), y + 13, colors["noaux"], 6)

    legend_y = height - 45
    draw.text((70, legend_y), "Blue: saliency-supervised   Gray: matched no-auxiliary control",
              font=font(18), fill="#202020")
    circle(draw, 740, legend_y + 8, "#0072B2")
    draw.text((755, legend_y), "always-new", font=font(18), fill="#202020")
    diamond(draw, 920, legend_y + 8, "#0072B2")
    draw.text((936, legend_y), "routed", font=font(18), fill="#202020")
    draw.text((1220, legend_y), "Symmetric axis; exact values are retained in the CSV.",
              font=font(17), fill="#555555")
    image.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliency-run", type=Path, required=True)
    parser.add_argument("--noaux-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2014)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saliency, noaux = load_run(args.saliency_run), load_run(args.noaux_run)
    rows = rows_for(saliency, noaux, args.year)
    with (args.output / "metrics_2014.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "year": args.year,
        "matched_historical_scores": True,
        "scenarios": len(rows),
        "largest_saliency_routed_gain": max(rows, key=lambda row: row["saliency_routed_delta_pp"]),
        "smallest_saliency_routed_gain": min(rows, key=lambda row: row["saliency_routed_delta_pp"]),
        "realistic_mix": next(row for row in rows if row["scenario"] == "realistic_mix"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(rows, args.output / "comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
