"""Logged server experiment: GPU preflight, temporal folds, and final models."""
import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data import open_dataset


def run_command(uv, script, args, log):
    command = [str(uv), "run", "--no-sync", script, *map(str, args)]
    print("$ " + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = process.wait()
        handle.flush()
    if code:
        raise RuntimeError(f"{script} failed with exit code {code}; see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Number of latest chronological years in the official train split to validate")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if args.cv_folds < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("cv-folds must be nonnegative; epochs and batch-size must be positive")
    if args.cv_only and args.cv_folds == 0:
        parser.error("--cv-only needs --cv-folds")
    uv = Path("tools/uv").resolve()
    if not uv.is_file():
        parser.error("Run bash deploy/server.sh setup first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = Path("runs/experiments") / stamp
    experiment.mkdir(parents=True, exist_ok=False)
    log = experiment / "console.log"
    (experiment / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    results = []
    try:
        run_command(uv, "hardware.py", ["--require-gpu"], log)
        try:
            dataset = open_dataset("data")
        except FileNotFoundError as error:
            raise RuntimeError("FMoW data is missing. Run 'bash deploy/server.sh ingest' before the experiment.") from error
        if not args.skip_smoke:
            run_command(uv, "smoke.py", [], log)
        year_column = dataset.metadata_fields.index("year")
        years = sorted({2002 + int(year) for year in
                        dataset.get_subset("train").metadata_array[:, year_column].tolist() if year >= 0})
        eligible = years[2:]
        if args.cv_folds > len(eligible):
            parser.error(f"Only {len(eligible)} chronological folds available: {eligible}")
        common = ["--epochs", args.epochs, "--batch-size", args.batch_size, "--seed", args.seed]
        if args.limit_train:
            common += ["--limit-train", args.limit_train]
        if args.limit_eval:
            common += ["--limit-eval", args.limit_eval]
        for year in eligible[-args.cv_folds:] if args.cv_folds else []:
            for method in ("baseline", "xai"):
                run_id = f"{stamp}-cv-{year}"
                run_command(uv, "train.py", ["--method", method, "--run-id", run_id,
                                             "--cv-val-year", year, *common], log)
                run_dir = Path("runs") / method / run_id
                metrics = json.loads((run_dir / "metrics/final.json").read_text())
                results.append({"method": method, "validation_year": year, "run": str(run_dir),
                                "macro_f1": metrics["id_val"]["all"]["macro_f1"],
                                "macro_iou": metrics["id_val"]["all"]["macro_iou"]})
        if not args.cv_only:
            for method in ("baseline", "xai"):
                run_id = f"{stamp}-final"
                run_command(uv, "train.py", ["--method", method, "--run-id", run_id, *common], log)
                results.append({"method": method, "run": str(Path("runs") / method / run_id)})
            checkpoint = Path("runs/xai") / f"{stamp}-final" / "models/best.pt"
            twin_args = ["--base-checkpoint", checkpoint, "--run-id", f"{stamp}-final",
                         "--epochs", args.epochs, "--batch-size", args.batch_size, "--seed", args.seed]
            if args.limit_train:
                twin_args += ["--limit-adapt", args.limit_train]
            if args.limit_eval:
                twin_args += ["--limit-calibrate", args.limit_eval, "--limit-test", args.limit_eval]
            run_command(uv, "train_siamese.py", twin_args, log)
            results.append({"method": "siamese", "run": str(Path("runs/siamese") / f"{stamp}-final")})
        cv_summary = {}
        for method in ("baseline", "xai"):
            folds = [item for item in results if item["method"] == method and "validation_year" in item]
            if folds:
                cv_summary[method] = {"folds": len(folds),
                                      "macro_f1_mean": statistics.mean(item["macro_f1"] for item in folds),
                                      "macro_f1_std": statistics.pstdev(item["macro_f1"] for item in folds),
                                      "macro_iou_mean": statistics.mean(item["macro_iou"] for item in folds)}
        (experiment / "summary.json").write_text(json.dumps({"status": "completed", "runs": results,
                                                                 "cv_summary": cv_summary}, indent=2), encoding="utf-8")
        print(f"Completed. Summary: {experiment / 'summary.json'}")
    except Exception as error:
        (experiment / "summary.json").write_text(json.dumps({"status": "failed", "error": str(error),
                                                                 "completed_runs": results}, indent=2), encoding="utf-8")
        print(f"Experiment stopped: {error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
