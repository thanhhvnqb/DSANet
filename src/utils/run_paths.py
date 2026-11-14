"""Shared output layout for training and standalone evaluation."""
from pathlib import Path


def add_run_arguments(parser):
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--hypothesis", default="baseline")
    parser.add_argument("--variant", default="DSANet")


def prepare_run(args, dataset, training=False):
    for name in ("hypothesis", "variant"):
        value = getattr(args, name)
        if value in ("", ".", "..") or Path(value).name != value or "\\" in value:
            raise ValueError(f"{name} must be a single directory name")
    run_dir = Path(args.output_root) / args.hypothesis / args.variant / dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    for name in ("log_path", "test_log_path"):
        value = getattr(args, name)
        if value and Path(value).resolve().parent != run_dir.resolve():
            raise ValueError(f"--{name.replace('_', '-')} must be inside {run_dir}")
    for name, filename in (("model_path", "best_model.pth"), ("checkpoint_path", "best_checkpoint.pth")):
        value = getattr(args, name)
        if value is None:
            setattr(args, name, str(run_dir / filename))
        elif training and Path(value).resolve().parent != run_dir.resolve():
            raise ValueError(f"--{name.replace('_', '-')} must be inside {run_dir}")
    if training and Path(args.model_path).resolve() == Path(args.checkpoint_path).resolve():
        raise ValueError("Model weights and training checkpoint must use different files")
    if training and not args.use_checkpoint:
        if any(Path(getattr(args, name)).exists() for name in ("model_path", "checkpoint_path")):
            raise FileExistsError("Run already has model artifacts; choose a new hypothesis/variant or resume explicitly")
    return run_dir
