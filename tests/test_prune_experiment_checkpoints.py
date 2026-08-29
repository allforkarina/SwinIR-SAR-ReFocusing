from pathlib import Path

import pytest

from scripts.prune_experiment_checkpoints import parse_args, run


def make_run(runs_dir: Path, name: str) -> Path:
    checkpoint_dir = runs_dir / name / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (runs_dir / name / "report.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "best.pt").write_bytes(b"best")
    (checkpoint_dir / "latest.pt").write_bytes(b"latest")
    (checkpoint_dir / "notes.txt").write_text("keep", encoding="utf-8")
    return checkpoint_dir


def test_inventory_and_dry_run_do_not_delete(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    checkpoint_dir = make_run(runs_dir, "failed_run")

    assert run(parse_args(["--runs-dir", str(runs_dir)])) == 0
    assert "failed_run: files=2" in capsys.readouterr().out

    assert (
        run(
            parse_args(
                ["--runs-dir", str(runs_dir), "--delete-all", "failed_run"]
            )
        )
        == 0
    )
    assert "No files deleted" in capsys.readouterr().out
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "latest.pt").exists()


def test_apply_delete_all_preserves_non_checkpoint_artifacts(tmp_path):
    runs_dir = tmp_path / "runs"
    checkpoint_dir = make_run(runs_dir, "failed_run")

    assert (
        run(
            parse_args(
                [
                    "--runs-dir",
                    str(runs_dir),
                    "--delete-all",
                    "failed_run",
                    "--apply",
                ]
            )
        )
        == 0
    )

    assert not (checkpoint_dir / "best.pt").exists()
    assert not (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "notes.txt").exists()
    assert (runs_dir / "failed_run" / "report.json").exists()


def test_keep_best_only_preserves_best_checkpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    checkpoint_dir = make_run(runs_dir, "active_run")

    assert (
        run(
            parse_args(
                [
                    "--runs-dir",
                    str(runs_dir),
                    "--keep-best-only",
                    "active_run",
                    "--apply",
                ]
            )
        )
        == 0
    )

    assert (checkpoint_dir / "best.pt").read_bytes() == b"best"
    assert not (checkpoint_dir / "latest.pt").exists()


@pytest.mark.parametrize("unsafe_name", ["../outside", "nested/run", ".", ""])
def test_rejects_unsafe_run_names(tmp_path, unsafe_name):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    args = parse_args(["--runs-dir", str(runs_dir), "--delete-all", unsafe_name])

    with pytest.raises(ValueError):
        run(args)
