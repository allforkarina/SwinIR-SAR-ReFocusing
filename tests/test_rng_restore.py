from __future__ import annotations

import pytest
import torch

from swinir.training import capture_rng_state, restore_rng_state


def test_restore_rng_state_normalizes_torch_state_to_cpu_byte_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture_rng_state()
    state.pop("torch_cuda", None)
    state["torch"] = state["torch"].to(dtype=torch.int64)
    restored: list[torch.Tensor] = []

    monkeypatch.setattr(torch, "set_rng_state", lambda value: restored.append(value))

    restore_rng_state(state)

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_restore_rng_state_normalizes_cuda_states_to_cpu_byte_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture_rng_state()
    state["torch_cuda"] = [state["torch"].to(dtype=torch.int64)]
    restored: list[list[torch.Tensor]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: restored.append(values),
    )

    restore_rng_state(state)

    assert len(restored) == 1
    assert len(restored[0]) == 1
    assert restored[0][0].device.type == "cpu"
    assert restored[0][0].dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_restore_rng_state_accepts_cuda_mapped_checkpoint_tensors() -> None:
    state = capture_rng_state()
    state["torch"] = state["torch"].to("cuda")
    state["torch_cuda"] = [rng_state.to("cuda") for rng_state in state["torch_cuda"]]

    restore_rng_state(state)
