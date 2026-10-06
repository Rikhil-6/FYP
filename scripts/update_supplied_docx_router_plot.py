"""Replace only Figure 3 and its interpretation in the user-supplied DOCX."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


OLD_CAPTION = "Figure 3. Matched 2014 change by drift source"
NEW_CAPTION = (
    "Figure 3. Selective-routing gain over the historical ViT and the added effect of saliency losses"
)

OLD_TEXT = (
    "The saliency and no-auxiliary runs use the same seed, historical checkpoint, temporal split, "
    "batch size, and scenario definitions. Under motion blur, their routed gains are 3.45 and 2.55 "
    "points, so the auxiliary losses add 0.90 points. Under the realistic mixture, they add 0.84 "
    "points. The always-new motion-blur models are almost tied (+3.96 and +3.94), suggesting that "
    "the larger difference appears in the routing behaviour rather than in the updated classifier "
    "alone. This effect is not consistent across all conditions: on clean data the saliency-supervised "
    "router is 0.18 points below the matched control, and several mild shifts differ by less than 0.10 points."
)

NEW_TEXT = (
    "Figure 3 evaluates routing against its intended deployment baseline, the frozen historical ViT. Every "
    "routed result is positive, led by motion blur (+3.45 points) and the realistic mixture (+2.03 points). "
    "Relative to the matched no-auxiliary router, saliency losses add +0.90 and +0.84 points in those "
    "conditions. Routing also prevents 16 and 20 harmful replacements, although it misses 47 and 30 "
    "beneficial updates. The updated ViT remains a performance ceiling: routing is 0.06–0.51 points lower "
    "overall, yet retains 87.1% and 91.9% of its gain in the two hardest conditions while making model "
    "switches explicit and reviewable."
)

TARGET_MEDIA = "word/media/image4.png"
TARGET_SALIENCY_MEDIA = "word/media/image6.png"

OLD_SALIENCY_TEXT = (
    "Each audit panel shows the clean image, the drifted input, and the historical and updated evidence "
    "maps. The label line records the true class, both predictions, the routed prediction, route probability, "
    "saliency divergence, and deletion margin. The overlays are kept faint enough for the image to remain visible."
)

NEW_SALIENCY_TEXT = (
    "Each audit row now separates the clean reference, drifted input, historical saliency, updated saliency, "
    "and signed saliency change. The two overlays use the same robust 5th–95th percentile normalisation and "
    "a stronger heatmap blend so spatial evidence remains visible. In the change map, red marks patches that "
    "gain weight in the updated ViT and blue marks patches that lose weight. Because each row is scaled for "
    "display, colour shows location and direction rather than absolute effect size; the reported Jensen–Shannon "
    "divergence and deletion margin quantify magnitude and behavioural relevance."
)

OLD_FIGURE4_CAPTION = (
    "Figure 4. Helpful routed corrections under motion blur, viewpoint shift, and a realistic mixture"
)

NEW_FIGURE4_CAPTION = (
    "Figure 4. Helpful routed corrections with historical, updated, and signed saliency-change maps"
)


def replace_once(payload: bytes, old: str, new: str, label: str) -> bytes:
    old_bytes = old.encode("utf-8")
    if payload.count(old_bytes) != 1:
        raise ValueError(f"Expected one {label} match, found {payload.count(old_bytes)}")
    return payload.replace(old_bytes, new.encode("utf-8"), 1)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--saliency-figure", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with Image.open(args.plot) as image:
        if image.size != (1800, 913):
            raise ValueError(f"Replacement plot must be 1800x913, found {image.size}")

    plot_bytes = args.plot.read_bytes()
    saliency_bytes = args.saliency_figure.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.input, "r") as source:
        names = source.namelist()
        if TARGET_MEDIA not in names:
            raise ValueError(f"Target image is missing: {TARGET_MEDIA}")
        old_media = {
            name: digest(source.read(name))
            for name in names
            if name.startswith("word/media/") and name not in {TARGET_MEDIA, TARGET_SALIENCY_MEDIA}
        }
        document_xml = source.read("word/document.xml")
        document_xml = replace_once(document_xml, OLD_CAPTION, NEW_CAPTION, "caption")
        document_xml = replace_once(document_xml, OLD_TEXT, NEW_TEXT, "interpretation paragraph")
        document_xml = replace_once(document_xml, OLD_SALIENCY_TEXT, NEW_SALIENCY_TEXT,
                                    "saliency explanation paragraph")
        document_xml = replace_once(document_xml, OLD_FIGURE4_CAPTION, NEW_FIGURE4_CAPTION,
                                    "Figure 4 caption")

        with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as target:
            for info in source.infolist():
                if info.filename == "word/document.xml":
                    data = document_xml
                elif info.filename == TARGET_MEDIA:
                    data = plot_bytes
                elif info.filename == TARGET_SALIENCY_MEDIA:
                    data = saliency_bytes
                else:
                    data = source.read(info.filename)
                target.writestr(info, data)

    with ZipFile(args.output, "r") as revised:
        for name, expected in old_media.items():
            actual = digest(revised.read(name))
            if actual != expected:
                raise RuntimeError(f"Unexpected change to {name}")
        if digest(revised.read(TARGET_MEDIA)) != digest(plot_bytes):
            raise RuntimeError("Replacement plot was not written correctly")
        if digest(revised.read(TARGET_SALIENCY_MEDIA)) != digest(saliency_bytes):
            raise RuntimeError("Replacement saliency figure was not written correctly")

    print(args.output)


if __name__ == "__main__":
    main()
