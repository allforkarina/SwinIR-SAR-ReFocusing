"""Safely inventory or prune checkpoint files from explicitly named runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


@dataclass(frozen=True)
class PruneAction:
    run_name: str
    mode: str
    path: Path
    size_bytes: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory checkpoint usage or prune checkpoint files from explicitly "
            "selected experiment runs. The default is a non-destructive preview."
        )
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--delete-all",
        action="append",
        default=[],
        metavar="RUN",
        help="Delete every checkpoint file under RUN/checkpoints (repeatable).",
    )
    parser.add_argument(
        "--keep-best-only",
        action="append",
        default=[],
        metavar="RUN",
        help="Keep best.pt and delete other checkpoint files (repeatable).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the previewed deletions. Without this flag nothing is removed.",
    )
    return parser.parse_args(argv)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def validate_run_name(run_name: str) -> None:
    candidate = Path(run_name)
    if (
        not run_name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or run_name in {".", ".."}
    ):
        raise ValueError(f"run must be one immediate directory name, got {run_name!r}")


def checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise ValueError(f"unsafe checkpoint directory: {checkpoint_dir}")

    files: list[Path] = []
    for path in checkpoint_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing checkpoint symlink: {path}")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            files.append(path)
    return sorted(files)


def inventory(runs_dir: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        files = checkpoint_files(run_dir / "checkpoints")
        if files:
            rows.append((run_dir.name, len(files), sum(path.stat().st_size for path in files)))
    return rows


def plan_actions(
    runs_dir: Path,
    delete_all: Iterable[str],
    keep_best_only: Iterable[str],
) -> list[PruneAction]:
    delete_names = list(delete_all)
    best_names = list(keep_best_only)
    overlap = set(delete_names) & set(best_names)
    if overlap:
        raise ValueError(f"runs cannot use both policies: {sorted(overlap)}")

    actions: list[PruneAction] = []
    for mode, names in (("delete_all", delete_names), ("keep_best_only", best_names)):
        for run_name in names:
            validate_run_name(run_name)
            run_dir = runs_dir / run_name
            if not run_dir.is_dir() or run_dir.is_symlink():
                raise FileNotFoundError(f"run directory not found or unsafe: {run_dir}")
            for path in checkpoint_files(run_dir / "checkpoints"):
                if mode == "keep_best_only" and path.name == "best.pt":
                    continue
                actions.append(
                    PruneAction(
                        run_name=run_name,
                        mode=mode,
                        path=path,
                        size_bytes=path.stat().st_size,
                    )
                )
    return actions


def run(args: argparse.Namespace) -> int:
    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")

    if not args.delete_all and not args.keep_best_only:
        rows = inventory(runs_dir)
        total = sum(size for _, _, size in rows)
        for run_name, count, size in rows:
            print(f"{run_name}: files={count} size={format_bytes(size)}")
        print(f"inventory_total: runs={len(rows)} size={format_bytes(total)}")
        return 0

    actions = plan_actions(runs_dir, args.delete_all, args.keep_best_only)
    total = sum(action.size_bytes for action in actions)
    mode = "APPLY" if args.apply else "DRY-RUN"
    for action in actions:
        relative_path = action.path.relative_to(runs_dir)
        print(
            f"{mode} {action.mode} {relative_path} "
            f"size={format_bytes(action.size_bytes)}"
        )
    print(f"{mode} total_files={len(actions)} reclaimable={format_bytes(total)}")

    if args.apply:
        for action in actions:
            action.path.unlink()
        print(f"deleted_files={len(actions)} reclaimed={format_bytes(total)}")
    else:
        print("No files deleted. Re-run the same command with --apply after review.")
    return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
