import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch


class TrainingLog:
    def __init__(self, dataset, args, device, selection_metric, validation_schedule):
        started_at = datetime.now(timezone.utc)
        if args.log_path:
            self.path = Path(args.log_path)
        else:
            timestamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
            self.path = Path(args.run_dir) / f"train_{timestamp}.json"

        if self.path.exists():
            raise FileExistsError(f"Training log already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        git_status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_metadata = {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "working_tree_status": git_status.splitlines(),
        }
        if git_status:
            git_metadata["diff_from_head"] = subprocess.run(
                ["git", "diff", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        device_name = str(device)
        if torch.device(device).type == "cuda":
            device_name = torch.cuda.get_device_name(device)

        self.data = {
            "schema_version": 1,
            "metadata": {
                "dataset": dataset,
                "hypothesis": args.hypothesis,
                "variant": args.variant,
                "run_dir": args.run_dir,
                "seed": args.seed,
                "started_at": started_at.isoformat(),
                "command": [sys.executable, *sys.argv],
                "configuration": vars(args),
                "selection_split": "test",
                "selection_metric": selection_metric,
                "validation_schedule": validation_schedule,
                "metric_units": {
                    "frame_level_roc_auc": "unit interval [0, 1]",
                    "frame_level_average_precision": "unit interval [0, 1]",
                    "fine_grained_map": "percent, as returned by the detection evaluator",
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "device": device_name,
                    "dependencies": {
                        distribution.metadata["Name"]: distribution.version
                        for distribution in distributions()
                        if distribution.metadata.get("Name")
                    },
                },
                "git": git_metadata,
                "log_path": str(self.path),
            },
            "training": {
                "intervals": [],
                "epochs": [],
            },
            "validation": [],
            "best_checkpoint": None,
            "status": "running",
        }
        self.save()

    def record_training_interval(
        self, epoch, batch_index, global_step, protocol_step, averaged_losses,
        batch_losses, learning_rates,
    ):
        self.data["training"]["intervals"].append({
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "protocol_step": protocol_step,
            "averaged_losses": averaged_losses,
            "batch_losses": batch_losses,
            "learning_rates": learning_rates,
        })
        self.save()

    def record_epoch(self, epoch, batches, mean_losses, learning_rates):
        self.data["training"]["epochs"].append({
            "epoch": epoch,
            "batches": batches,
            "mean_losses": mean_losses,
            "learning_rates": learning_rates,
        })
        self.save()

    def record_validation(
        self, epoch, batch_index, global_step, selection_value, metrics,
        checkpoint_selected, checkpoint_path,
    ):
        validation = {
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "selection_metric": self.data["metadata"]["selection_metric"],
            "selection_value": selection_value,
            "checkpoint_selected": checkpoint_selected,
            "metrics": metrics,
        }
        self.data["validation"].append(validation)
        if checkpoint_selected:
            self.data["best_checkpoint"] = {
                "path": checkpoint_path,
                "selection_split": self.data["metadata"]["selection_split"],
                "selection_metric": validation["selection_metric"],
                "selection_value": selection_value,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "validation_metrics": metrics,
            }
        self.save()

    def finish(self):
        self.data["status"] = "completed"
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def save(self):
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        _write_json(temporary_path, self.data)
        temporary_path.replace(self.path)


def save_evaluation_log(dataset, args, device, selection_metric, selection_value, metrics):
    evaluated_at = datetime.now(timezone.utc)
    if args.test_log_path:
        path = Path(args.test_log_path)
    else:
        timestamp = evaluated_at.strftime("%Y%m%dT%H%M%S.%fZ")
        path = Path(args.run_dir) / f"test_{timestamp}.json"

    if path.exists():
        raise FileExistsError(f"Evaluation log already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    device_name = str(device)
    if torch.device(device).type == "cuda":
        device_name = torch.cuda.get_device_name(device)

    data = {
        "schema_version": 1,
        "metadata": {
            "dataset": dataset,
                "hypothesis": args.hypothesis,
                "variant": args.variant,
                "run_dir": args.run_dir,
            "seed": args.seed,
            "evaluated_at": evaluated_at.isoformat(),
            "command": [sys.executable, *sys.argv],
            "configuration": vars(args),
            "selection_split": "test",
            "selection_metric": selection_metric,
            "metric_units": {
                "frame_level_roc_auc": "unit interval [0, 1]",
                "frame_level_average_precision": "unit interval [0, 1]",
                "fine_grained_map": "percent, as returned by the detection evaluator",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": device_name,
            },
            "log_path": str(path),
        },
        "evaluated_checkpoint": {
            "path": args.model_path,
            "selection_split": "test",
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "metrics": metrics,
        },
    }

    temporary_path = path.with_name(f"{path.name}.tmp")
    _write_json(temporary_path, data)
    temporary_path.replace(path)
    return path


def _write_json(path, data):
    with path.open("w", encoding="utf-8") as log_file:
        json.dump(data, log_file, indent=2, allow_nan=False)
        log_file.write("\n")
        log_file.flush()
