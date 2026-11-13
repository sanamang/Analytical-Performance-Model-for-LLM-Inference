"""The explorer page must show the same numbers as the Python model."""

import json

import pytest

from perfmodel.export import apply_knobs, build, step_record
from perfmodel.ops import forward_ops
from perfmodel.predict import predict_step
from perfmodel.specs import load_chip, load_model


@pytest.mark.parametrize("chip", ["a100", "sram-accel"])
@pytest.mark.parametrize("batch, n_past, n_new", [(1, 1024, 1), (256, 512, 1), (4, 0, 2048)])
@pytest.mark.parametrize("knobs", [(1.0, 1.0, 0.0), (0.6, 0.8, 20e-6)])
def test_page_formula_matches_predict_step(chip, batch, n_past, n_new, knobs):
    ce, be, oh = knobs
    m, c = load_model("llama3.2-1b").with_(weight_bits=4), load_chip(chip)
    rec = step_record(predict_step(m, c, batch, n_past, n_new))
    counts = [o.count for o in forward_ops(m, batch, n_past, n_new)]
    page = sum(apply_knobs(rec, counts, ce, be, oh))
    truth = predict_step(
        m,
        c,
        batch,
        n_past,
        n_new,
        compute_efficiency=ce,
        bandwidth_efficiency=be,
        overhead_per_op_s=oh,
    ).time_s
    assert page == pytest.approx(truth, rel=2e-3)  # 4-significant-digit rounding


def test_build_writes_every_pair(tmp_path):
    paths = build(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert len(paths) == len(meta["models"]) * len(meta["chips"]) + 1
    data = json.loads((tmp_path / "llama3.2-1b__a100.json").read_text())
    rec = data["points"]["decode|16|1|1024"]
    assert len(rec["tc"]) == len(data["ops"]) == len(data["counts"])
    tok_s = 1 / sum(apply_knobs(rec, data["counts"]))
    assert tok_s == pytest.approx(812.8, rel=0.01)  # same as `perfmodel predict`
