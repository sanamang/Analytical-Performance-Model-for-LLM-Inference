import pandas as pd
import pytest
from typer.testing import CliRunner

from perfmodel.cli import app
from perfmodel.ops import decode_ops, total_bytes, total_flops
from perfmodel.predict import predict, predict_step, sweep
from perfmodel.specs import load_chip, load_model


def test_decode_b1_is_weight_bandwidth_bound():
    m, c = load_model("llama3-8b"), load_chip("a100")
    r = predict_step(m, c, batch=1, n_past=128, n_new=1)
    expected = total_bytes(decode_ops(m, 1, 128)) / c.dram_bw
    assert r.time_s == pytest.approx(expected, rel=0.01)  # bytes / BW
    assert r.compute_bound_fraction < 0.01
    assert r.flops == pytest.approx(total_flops(decode_ops(m, 1, 128)))


def test_large_batch_decode_matmuls_become_compute_bound():
    m, c = load_model("llama3-8b"), load_chip("a100")
    ridge = c.peak_flops("bf16") / c.dram_bw  # ~153 FLOP/B
    # A weight matmul's intensity is ~ 2*B / w_bytes = B at 16-bit: flips near B ~ ridge.
    lo = {o.name: o for o in predict_step(m, c, int(ridge * 0.5), 16, 1).ops}
    hi = {o.name: o for o in predict_step(m, c, int(ridge * 2), 16, 1).ops}
    assert lo["mlp_up"].bound == "memory"
    assert hi["mlp_up"].bound == "compute"
    assert hi["attn_scores"].bound == "memory"  # KV reads never amortize across batch


def test_prefill_mostly_compute_bound():
    r = predict(load_model("llama3-8b"), load_chip("h100"), 1, 2048)["prefill"]
    assert r.compute_bound_fraction > 0.8


def test_quantization_speeds_up_memory_bound_decode():
    m, c = load_model("llama3.2-1b"), load_chip("a100")
    t16 = predict_step(m, c, 1, 256, 1).time_s
    t4 = predict_step(m.with_(weight_bits=4), c, 1, 256, 1).time_s
    assert 2.5 < t16 / t4 < 4.0  # not quite 4x: KV, activations, norms stay 16-bit


def test_sram_chip_beats_dram_when_model_fits():
    m = load_model("llama3.2-1b").with_(weight_bits=4)
    chip = load_chip("sram-accel")
    resident = predict_step(m, chip, 1, 512, 1)
    streamed = predict_step(m, chip, 1, 512, 1, placement_policy="none")
    assert resident.dram_bytes < 0.01 * streamed.dram_bytes
    assert resident.time_s < streamed.time_s / 10
    assert resident.energy_j < streamed.energy_j


def test_overhead_and_efficiency_knobs():
    m, c = load_model("qwen2.5-1.5b"), load_chip("a100")
    base = predict_step(m, c, 1, 64, 1)
    slow = predict_step(m, c, 1, 64, 1, bandwidth_efficiency=0.5)
    assert slow.time_s == pytest.approx(2 * base.time_s, rel=0.02)
    oh = predict_step(m, c, 1, 64, 1, overhead_per_op_s=10e-6)
    assert oh.time_s == pytest.approx(base.time_s + base.n_kernels * 10e-6)
    with pytest.raises(ValueError):
        predict_step(m, c, 1, 64, 1, compute_efficiency=0)


def test_static_power_adds_energy():
    m = load_model("qwen2.5-1.5b")
    c = load_chip("a100")
    r0 = predict_step(m, c, 1, 64, 1)
    r1 = predict_step(m, c.with_(static_power_w=100), 1, 64, 1)
    assert r1.energy_j - r0.energy_j == pytest.approx(100 * r0.time_s)


def test_sweep_shape():
    df = sweep(load_model("llama3.2-1b"), load_chip("h100"), [1, 8], [128, 512], [16, 4])
    assert len(df) == 2 * 2 * 2 * 2
    assert set(df.phase) == {"prefill", "decode"}
    assert (df.tokens_per_s > 0).all()


def test_cli_predict_and_sweep(tmp_path):
    runner = CliRunner()
    r = runner.invoke(app, ["predict", "-m", "llama3.2-1b", "-c", "a100", "--batch", "4"])
    assert r.exit_code == 0, r.output
    assert "mlp_up" in r.output and "tok/s" in r.output
    out = tmp_path / "s.csv"
    r = runner.invoke(
        app,
        [
            "sweep",
            "-m",
            "llama3.2-1b",
            "-c",
            "a100",
            "--batches",
            "1,2",
            "--contexts",
            "128",
            "--weight-bits",
            "16",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert len(pd.read_csv(out)) == 4
    assert runner.invoke(app, ["list"]).exit_code == 0
