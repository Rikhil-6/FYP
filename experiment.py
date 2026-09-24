"""Logged server experiment: GPU preflight, temporal folds, and final models."""
import argparse
import json
import os
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
                              bufsize=0, env=os.environ | {"PYTHONUNBUFFERED": "1"}) as process:
            while chunk := os.read(process.stdout.fileno(), 4096):
                output = chunk.decode("utf-8", errors="replace")
                print(output, end="", flush=True)
                handle.write(output)
                handle.flush()
            code = process.wait()
        handle.flush()
    if code:
        raise RuntimeError(f"{script} failed with exit code {code}; see {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-repeats", "--cv-folds", dest="temporal_repeats", type=int, default=0,
                        help="Repeated fixed-window temporal backtests (the old --cv-folds name is retained as an alias)")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true", help="Check CUDA and the model without dataset ingestion")
    args = parser.parse_args()
    if args.temporal_repeats < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("temporal-repeats must be nonnegative; epochs and batch-size must be positive")
    if args.cv_only and args.temporal_repeats == 0:
        parser.error("--cv-only needs --temporal-repeats")
    if args.smoke_only and (args.cv_only or args.temporal_repeats or args.skip_smoke):
        parser.error("--smoke-only cannot be combined with CV or --skip-smoke")
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
        if args.smoke_only:
            run_command(uv, "smoke.py", [], log)
            (experiment / "summary.json").write_text(json.dumps({"status": "completed", "smoke_only": True}, indent=2), encoding="utf-8")
            print(f"Completed. Summary: {experiment / 'summary.json'}")
            return
        try:
            dataset = open_dataset("data")
        except FileNotFoundError as error:
            raise RuntimeError("CLEAR-100 data is missing. Run 'bash deploy/server.sh ingest' before the experiment.") from error
        if not args.skip_smoke:
            run_command(uv, "smoke.py", [], log)
        common = ["--epochs", args.epochs, "--batch-size", args.batch_size, "--seed", args.seed]
        if args.limit_train:
            common += ["--limit-train", args.limit_train]
        if args.limit_eval:
            common += ["--limit-eval", args.limit_eval]
        # Every backtest uses the same historical fit window. Later years are
        # evaluation windows only, so future samples never migrate into training.
        for repeat in range(1, args.temporal_repeats + 1):
            for method in ("baseline", "xai"):
                run_id = f"{stamp}-backtest-r{repeat}"
                repeat_common = [value for value in common]
                repeat_common[repeat_common.index("--seed") + 1] = args.seed + repeat - 1
                run_command(uv, "train.py", ["--method", method, "--run-id", run_id,
                                             *repeat_common], log)
                run_dir = Path("runs") / method / run_id
                metrics = json.loads((run_dir / "metrics/final.json").read_text())
                for year in (2012, 2013):
                    yearly = metrics["val"][str(year)]
                    results.append({"method": method, "repeat": repeat,
                                    "training_years": "2004-2010", "selection_year": 2011,
                                    "validation_year": year, "run": str(run_dir),
                                    "macro_f1": yearly["macro_f1"],
                                    "macro_iou": yearly["macro_iou"]})
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
                cv_summary[method] = {"protocol": "fixed 2004-2010 fit; 2011 selection; 2012-2013 temporal backtest",
                                      "repeats": args.temporal_repeats,
                                      "yearly_scores": len(folds),
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
