import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wall_x.data.backends.lerobot.worker_rng import (
    FLOW_TIMESTEP_TRAIN_STREAM,
    FLOW_TIMESTEP_VALIDATION_STREAM,
    WorkerNumpyRNG,
)


class _TwoItemDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index


class _WorkerDrawCollator:
    def __init__(self):
        self.rng = WorkerNumpyRNG(rank=3, stream=FLOW_TIMESTEP_TRAIN_STREAM)

    def __call__(self, batch):
        worker_info = torch.utils.data.get_worker_info()
        return {
            "batch": list(batch),
            "worker_id": worker_info.id,
            "worker_seed": worker_info.seed,
            "draw": self.rng.generator().integers(0, 2**31, size=8).tolist(),
        }


def _draw_beta(monkeypatch, *, worker_seed, rank=0, stream=0, size=16):
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(seed=worker_seed),
    )
    return (
        WorkerNumpyRNG(rank=rank, stream=stream).generator().beta(1.5, 1.0, size=size)
    )


def _run_spawn_dataloader():
    loader = torch.utils.data.DataLoader(
        _TwoItemDataset(),
        batch_size=1,
        num_workers=2,
        collate_fn=_WorkerDrawCollator(),
        generator=torch.Generator().manual_seed(20260902),
        multiprocessing_context="spawn",
    )
    return list(loader)


def test_worker_rng_is_reproducible_for_same_namespace(monkeypatch):
    first = _draw_beta(monkeypatch, worker_seed=123456789, rank=2, stream=5)
    second = _draw_beta(monkeypatch, worker_seed=123456789, rank=2, stream=5)

    np.testing.assert_array_equal(first, second)


def test_worker_rng_separates_workers_ranks_and_streams(monkeypatch):
    baseline = _draw_beta(monkeypatch, worker_seed=123456789, rank=0, stream=0)
    worker_one = _draw_beta(monkeypatch, worker_seed=123456790, rank=0, stream=0)
    rank_one = _draw_beta(monkeypatch, worker_seed=123456789, rank=1, stream=0)
    stream_one = _draw_beta(monkeypatch, worker_seed=123456789, rank=0, stream=1)

    assert not np.array_equal(baseline, worker_one)
    assert not np.array_equal(baseline, rank_one)
    assert not np.array_equal(baseline, stream_one)


def test_worker_rng_preserves_high_worker_seed_and_rank_bits(monkeypatch):
    baseline = _draw_beta(monkeypatch, worker_seed=17, rank=9, stream=4)
    high_worker_seed = _draw_beta(monkeypatch, worker_seed=17 + 2**32, rank=9, stream=4)
    high_rank = _draw_beta(monkeypatch, worker_seed=17, rank=9 + 2**32, stream=4)

    assert not np.array_equal(baseline, high_worker_seed)
    assert not np.array_equal(baseline, high_rank)


def test_worker_rng_stream_advances_continuously(monkeypatch):
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(seed=555),
    )
    rng = WorkerNumpyRNG(rank=1, stream=7)

    first = rng.generator().integers(0, 2**31, size=8)
    second = rng.generator().integers(0, 2**31, size=8)
    reference = WorkerNumpyRNG(rank=1, stream=7).generator().integers(0, 2**31, size=16)

    np.testing.assert_array_equal(np.concatenate([first, second]), reference)


def test_worker_rng_reseeds_if_worker_context_changes(monkeypatch):
    worker_info = SimpleNamespace(seed=7)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker_info)
    rng = WorkerNumpyRNG(rank=0, stream=0)

    first = rng.generator().integers(0, 2**31, size=8)
    worker_info.seed = 8
    second = rng.generator().integers(0, 2**31, size=8)
    expected = WorkerNumpyRNG(rank=0, stream=0).generator().integers(0, 2**31, size=8)

    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(second, expected)


def test_worker_rng_pickle_drops_parent_generator_state(monkeypatch):
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(seed=101),
    )
    parent = WorkerNumpyRNG(rank=2, stream=3)
    parent.generator().random(8)

    copied = pickle.loads(pickle.dumps(parent))
    copied_draw = copied.generator().random(8)
    fresh_draw = WorkerNumpyRNG(rank=2, stream=3).generator().random(8)

    np.testing.assert_array_equal(copied_draw, fresh_draw)


def test_spawn_dataloader_uses_reproducible_distinct_worker_streams(monkeypatch):
    first = _run_spawn_dataloader()
    second = _run_spawn_dataloader()

    assert first == second
    assert {result["worker_id"] for result in first} == {0, 1}
    assert first[0]["draw"] != first[1]["draw"]

    for result in first:
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda result=result: SimpleNamespace(seed=result["worker_seed"]),
        )
        expected = (
            WorkerNumpyRNG(rank=3, stream=FLOW_TIMESTEP_TRAIN_STREAM)
            .generator()
            .integers(0, 2**31, size=8)
            .tolist()
        )
        assert result["draw"] == expected


def test_main_process_train_and_validation_streams_are_distinct(monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    monkeypatch.setattr(torch, "initial_seed", lambda: 987654321)

    train = (
        WorkerNumpyRNG(rank=0, stream=FLOW_TIMESTEP_TRAIN_STREAM).generator().random(8)
    )
    validation = (
        WorkerNumpyRNG(rank=0, stream=FLOW_TIMESTEP_VALIDATION_STREAM)
        .generator()
        .random(8)
    )
    train_repeat = (
        WorkerNumpyRNG(rank=0, stream=FLOW_TIMESTEP_TRAIN_STREAM).generator().random(8)
    )

    assert not np.array_equal(train, validation)
    np.testing.assert_array_equal(train, train_repeat)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [("rank", {"rank": -1}), ("stream", {"stream": -1})],
)
def test_worker_rng_rejects_negative_namespace_values(field, kwargs):
    defaults = {"rank": 0, "stream": 0}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=field):
        WorkerNumpyRNG(**defaults)
