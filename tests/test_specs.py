import pytest

from perfmodel.specs import ChipSpec, ModelSpec, available_configs, load_chip, load_model


@pytest.mark.parametrize(
    "name, published",
    [
        ("llama3-8b", 8.03e9),  # Meta: 8.03B
        ("llama3.2-1b", 1.236e9),  # Meta: 1.23B (1.24B in HF card)
        ("qwen2.5-1.5b", 1.54e9),  # Qwen2.5 report: 1.54B
    ],
)
def test_param_counts_match_published(name, published):
    m = load_model(name)
    assert m.param_count() == pytest.approx(published, rel=0.005)


def test_qwen_non_embedding_params():
    m = load_model("qwen2.5-1.5b")
    non_embed = m.param_count() - m.vocab_size * m.d_model  # tied, so subtract once
    assert non_embed == pytest.approx(1.31e9, rel=0.005)  # Qwen2.5 report: 1.31B


def test_all_configs_load():
    assert set(available_configs("models")) >= {"llama3-8b", "llama3.2-1b", "qwen2.5-1.5b"}
    assert set(available_configs("chips")) >= {"a100", "h100", "jetson-orin", "sram-accel"}
    for c in available_configs("chips"):
        chip = load_chip(c)
        assert chip.dram_bw > 0
        assert chip.peak_flops(chip.vector_dtype) > 0


def test_head_dim_default_and_gqa_validation():
    m = ModelSpec("t", n_layers=1, d_model=64, n_heads=8, n_kv_heads=2, d_ff=128, vocab_size=10)
    assert m.head_dim == 8 and m.kv_dim == 16
    with pytest.raises(ValueError):
        ModelSpec("t", n_layers=1, d_model=64, n_heads=8, n_kv_heads=3, d_ff=128, vocab_size=10)


def test_with_changes_precision():
    m = load_model("llama3.2-1b").with_(weight_bits=4)
    assert m.w_bytes == 0.5


def test_persistent_sram_requires_bandwidth():
    with pytest.raises(ValueError):
        ChipSpec(
            name="bad",
            peak_tflops={"fp16": 1},
            dram_gbps=1,
            sram_mib=1,
            sram_persistent=True,
            energy_pj_per_flop={"fp16": 1},
            dram_pj_per_byte=1,
        )


def test_unknown_yaml_key_rejected(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "name: x\nn_layers: 1\nd_model: 8\nn_heads: 2\nn_kv_heads: 1\n"
        "d_ff: 16\nvocab_size: 4\nhidden_sizee: 3\n"
    )
    with pytest.raises(ValueError, match="hidden_sizee"):
        load_model(p)
