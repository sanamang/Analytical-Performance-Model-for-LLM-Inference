import pytest

from perfmodel.energy import op_energy, pj_per_flop
from perfmodel.memory import place, tensor_traffic
from perfmodel.ops import EMBED, KV_CACHE, MLP_UP, decode_ops, tensor_footprints
from perfmodel.specs import ChipSpec, load_chip, load_model


def sram_chip(mib: float) -> ChipSpec:
    return ChipSpec(
        name=f"sram{mib}",
        peak_tflops={"fp16": 100, "bf16": 100, "fp32": 10},
        dram_gbps=1000,
        sram_mib=mib,
        sram_persistent=True,
        sram_gbps=100_000,
        energy_pj_per_flop={"fp16": 1, "bf16": 1, "fp32": 2},
        sram_pj_per_byte=1,
        dram_pj_per_byte=10,
    )


def setup(model="llama3.2-1b", weight_bits=4, batch=1, context=512, chip=None, policy=None):
    m = load_model(model).with_(weight_bits=weight_bits)
    ops = decode_ops(m, batch, context)
    fp = tensor_footprints(m, batch, context + 1)
    return m, ops, fp, place(ops, fp, chip, policy)


def test_everything_fits():
    _, ops, fp, p = setup(chip=sram_chip(4096))
    assert all(f == 1.0 for f in p.sram_fraction.values())
    assert p.activations_in_sram
    for o in ops:
        sram, dram = p.split(o)
        assert dram == pytest.approx(0)
        assert sram == pytest.approx(o.bytes)


def test_nothing_fits_tiny_sram():
    _, ops, fp, p = setup(chip=sram_chip(0.001))  # ~1 KB: not even the activation buffer
    assert not p.activations_in_sram  # activations spill, the 1 KB goes to weights
    assert p.sram_used <= p.capacity
    placed = sum(f * fp[t] for t, f in p.sram_fraction.items())
    assert placed == pytest.approx(p.capacity)
    total_traffic = sum(o.bytes * o.count for o in ops)
    sram_traffic = sum(p.split(o)[0] * o.count for o in ops)
    assert sram_traffic / total_traffic < 1e-5


def test_partial_fit_respects_capacity():
    m, ops, fp, p = setup(chip=sram_chip(256))  # 1B @ 4-bit is ~618 MB: does not fit
    assert p.sram_used == pytest.approx(p.capacity, rel=1e-9)
    partial = [t for t, f in p.sram_fraction.items() if 0 < f < 1]
    assert len(partial) == 1  # fractional knapsack: exactly one split tensor


def test_greedy_puts_low_density_embedding_last():
    # Untied 8B: the embedding table is only gathered (1 row/token), so it is placed last.
    # 8B @ 4-bit is ~3829 MiB; 3700 MiB leaves the embedding partly in DRAM.
    _, ops, fp, p = setup("llama3-8b", weight_bits=4, chip=sram_chip(3700))
    order = list(p.sram_fraction)
    assert order[-1] == EMBED
    assert p.sram_fraction[EMBED] < 1.0


def test_policy_none_and_kv_first():
    chip = sram_chip(64)
    _, _, _, p = setup(chip=chip, policy="none")
    assert p.sram_used == 0 and not p.activations_in_sram
    # KV for 1B at 2048 ctx is ~64 MiB, so use a 128 MiB chip.
    _, _, fp, p = setup(chip=sram_chip(128), policy="kv_first", context=2048)
    assert p.sram_fraction[KV_CACHE] == 1.0
    assert p.sram_fraction[MLP_UP] < 1.0


def test_gpu_rejects_sram_policy_and_defaults_to_none():
    gpu = load_chip("a100")
    _, ops, _, p = setup(chip=gpu)
    assert p.policy == "none"
    with pytest.raises(ValueError, match="sram_persistent"):
        setup(chip=gpu, policy="greedy")


def test_unknown_policy():
    with pytest.raises(ValueError):
        setup(chip=sram_chip(10), policy="lru")


def test_tied_embedding_traffic_counts_gather_and_lm_head():
    m = load_model("llama3.2-1b")
    traffic = tensor_traffic(decode_ops(m, 1, 0))
    assert traffic[EMBED] == pytest.approx((1 + m.vocab_size) * m.d_model * 2)


def test_energy_hand_computed():
    e = op_energy(
        flops=1e12, pj_flop=0.5, dram_bytes=1e9, dram_pj_byte=30, sram_bytes=2e9, sram_pj_byte=2
    )
    assert e.compute_j == pytest.approx(0.5)
    assert e.dram_j == pytest.approx(0.03)
    assert e.sram_j == pytest.approx(0.004)
    assert e.total_j == pytest.approx(0.534)
    with pytest.raises(ValueError):
        op_energy(1, 1, 1, 1, sram_bytes=1, sram_pj_byte=None)


def test_horowitz_defaults_match_chip_yaml():
    assert pj_per_flop("fp16") == pytest.approx(0.75)
    assert pj_per_flop("fp32") == pytest.approx(2.3)
    assert pj_per_flop("int8") == pytest.approx(0.115)
    assert load_chip("a100").pj_per_flop("fp16") == pytest.approx(pj_per_flop("fp16"))
