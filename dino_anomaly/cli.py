from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import MVTecDataset
from .evaluation import evaluate, save_report
from .model import DinoAnomalyDetector


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

MACRO_METRICS = (
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "image_f1",
    "pixel_f1",
)


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("layers must contain non-negative integers")
    return layers


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data_loader(args, dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )


def make_train_validation_loaders(args, category: str) -> tuple[DataLoader, DataLoader]:
    dataset = MVTecDataset(args.data_root, category, "train", args.image_size)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = max(1, round(len(dataset) * args.validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    validation_indices = indices[:validation_size]
    fit_indices = indices[validation_size:]
    return (
        make_data_loader(args, Subset(dataset, fit_indices)),
        make_data_loader(args, Subset(dataset, validation_indices)),
    )


def make_test_loader(args, category: str) -> DataLoader:
    dataset = MVTecDataset(args.data_root, category, "test", args.image_size)
    return make_data_loader(args, dataset)


def create_detector(args) -> DinoAnomalyDetector:
    return DinoAnomalyDetector(
        model_name=args.model,
        device=torch.device(args.device),
        image_size=args.image_size,
        layers=args.layers,
        query_chunk_size=args.query_chunk_size,
    )


def category_paths(args, category: str) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint or f"artifacts/{category}_detector.pt")
    output_dir = Path(args.output_dir or f"artifacts/{category}_evaluation")
    return checkpoint, output_dir


def fit_category(
    args,
    category: str,
    checkpoint: Path,
    detector: DinoAnomalyDetector | None = None,
) -> tuple[DinoAnomalyDetector, dict[str, float | int | str]]:
    detector = detector or create_detector(args)
    fit_loader, validation_loader = make_train_validation_loaders(args, category)
    fit_stats = detector.fit(
        fit_loader,
        args.max_memory_patches,
        args.seed,
        args.sampling,
        args.coreset_projection_dim,
        args.coreset_max_candidates,
    )
    calibration = detector.calibrate(
        validation_loader,
        args.target_image_fpr,
        args.target_pixel_fpr,
    )
    detector.save(checkpoint, category)
    stats = {**fit_stats, **calibration}
    print(
        f"[{category}] saved {stats['memory_patches']:,} {args.sampling} patches "
        f"and calibrated thresholds to {checkpoint}"
    )
    return detector, stats


def evaluate_category(
    args,
    category: str,
    checkpoint: Path,
    output_dir: Path,
    detector: DinoAnomalyDetector | None = None,
) -> dict[str, float | int]:
    if detector is None:
        detector = create_detector(args)
        state = detector.load(checkpoint)
        if state.category != category:
            raise ValueError(
                f"Checkpoint was fitted on '{state.category}', not '{category}'"
            )
    metrics, examples = evaluate(detector, make_test_loader(args, category))
    save_report(metrics, examples, output_dir, args.max_visualizations)
    print(f"[{category}] image AUROC={metrics['image_auroc']:.4f}, pixel AUROC={metrics['pixel_auroc']:.4f}")
    return metrics


def requested_categories(args) -> list[str]:
    if args.categories.strip().lower() == "all":
        categories = list(MVTEC_CATEGORIES)
    else:
        categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(MVTEC_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown MVTec categories: {', '.join(unknown)}")
    missing = [
        category for category in categories if not (Path(args.data_root) / category).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing category directories: {', '.join(missing)}")
    return categories


def run_benchmark(args) -> None:
    categories = requested_categories(args)
    root = Path(args.experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    detector = create_detector(args)
    rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    for index, category in enumerate(categories, start=1):
        print(f"\n[{index}/{len(categories)}] Running {category}")
        category_root = root / category
        try:
            detector, fit_stats = fit_category(
                args, category, category_root / "detector.pt", detector
            )
            metrics = evaluate_category(
                args,
                category,
                category_root / "detector.pt",
                category_root / "evaluation",
                detector,
            )
            rows.append({"category": category, **fit_stats, **metrics})
        except Exception as error:
            failures[category] = f"{type(error).__name__}: {error}"
            print(f"[{category}] FAILED: {failures[category]}")

    if not rows:
        raise RuntimeError("Every benchmark category failed; no summary was created")

    macro_average = {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in MACRO_METRICS
    }
    summary = {
        "configuration": {
            "model": args.model,
            "layers": list(args.layers),
            "image_size": args.image_size,
            "sampling": args.sampling,
            "max_memory_patches": args.max_memory_patches,
            "validation_fraction": args.validation_fraction,
            "target_image_fpr": args.target_image_fpr,
            "target_pixel_fpr": args.target_pixel_fpr,
            "seed": args.seed,
        },
        "macro_average": macro_average,
        "categories": rows,
        "failures": failures,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nMacro averages:")
    print(json.dumps(macro_average, indent=2))
    print(f"Saved benchmark summary to {root}")
    if failures:
        print(f"Warning: {len(failures)} categories failed; inspect summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DINOv2 patch-feature anomaly detection on MVTec AD"
    )
    parser.add_argument("command", choices=("fit", "evaluate", "run", "benchmark"))
    parser.add_argument("--data-root", default="dataset/archive")
    parser.add_argument("--category", default="grid")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--layers", type=parse_layers, default=(8, 11))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--target-image-fpr", type=float, default=0.05)
    parser.add_argument("--target-pixel-fpr", type=float, default=0.01)
    parser.add_argument("--sampling", choices=("coreset", "random"), default="coreset")
    parser.add_argument("--max-memory-patches", type=int, default=2_048)
    parser.add_argument("--coreset-projection-dim", type=int, default=32)
    parser.add_argument("--coreset-max-candidates", type=int, default=25_000)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--experiment-dir", default="artifacts/all_categories")
    parser.add_argument("--max-visualizations", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = str(resolve_device(args.device))
    seed_everything(args.seed)
    print(f"Using device: {args.device}")

    if args.command == "benchmark":
        run_benchmark(args)
        return

    checkpoint, output_dir = category_paths(args, args.category)
    if args.command == "fit":
        fit_category(args, args.category, checkpoint)
    elif args.command == "evaluate":
        metrics = evaluate_category(
            args, args.category, checkpoint, output_dir
        )
        print(json.dumps(metrics, indent=2))
    else:
        detector, _ = fit_category(args, args.category, checkpoint)
        metrics = evaluate_category(
            args, args.category, checkpoint, output_dir, detector
        )
        print(json.dumps(metrics, indent=2))
