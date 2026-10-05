"""Isolated run directories and explicit checkpoint metadata."""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


class RunStore:
    def __init__(self, root, method, run_id=None):
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("run_id must contain only letters, numbers, underscore, or dash")
        self.path = Path(root) / method / run_id
        if self.path.exists():
            raise FileExistsError(f"Run already exists: {self.path}")
        for folder in ("logs", "metrics", "models", "manual"):
            (self.path / folder).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing(cls, path):
        path = Path(path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Existing run is missing config.json: {path}")
        store = cls.__new__(cls)
        store.path = path
        return store

    def log(self, event):
        with (self.path / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def save_model(self, model, kind, epoch, metric_name, metric_value):
        if kind not in ("best", "last"):
            raise ValueError(kind)
        destination = self.path / "models" / f"{kind}.pt"
        torch.save(model.state_dict(), destination)
        self.write_json(f"models/{kind}.json", {"epoch": epoch,
                                                  "selection_metric": metric_name,
                                                  "metric_value": metric_value,
                                                  "saved_utc": datetime.now(timezone.utc).isoformat()})
        return destination

    def save_training_checkpoint(self, model, optimizer, epoch, best_score):
        """Atomically save enough state to resume after the latest epoch."""
        destination = self.path / "models" / "resume.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_score": best_score}, temporary)
        temporary.replace(destination)
        self.write_json("models/resume.json", {
            "epoch": epoch, "best_score": best_score,
            "saved_utc": datetime.now(timezone.utc).isoformat()})
        return destination

    def write_json(self, relative, value):
        path = self.path / relative
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def manual_snapshot(run_dir, name, which="best"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("name must contain only letters, numbers, underscore, or dash")
    run_dir = Path(run_dir)
    source = run_dir / "models" / f"{which}.pt"
    metadata = run_dir / "models" / f"{which}.json"
    if which not in ("best", "last") or not source.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"No {which} checkpoint and metadata in {run_dir}")
    destination = run_dir / "manual" / f"{name}.pt"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    details = json.loads(metadata.read_text(encoding="utf-8"))
    details.update({"source": str(source), "manual_name": name,
                    "saved_utc": datetime.now(timezone.utc).isoformat()})
    (run_dir / "manual" / f"{name}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return destination
