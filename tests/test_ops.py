import pytest

from perfmodel.ops import (
    KV_CACHE,
    causal_pairs,
    decode_ops,
    forward_ops,
    prefill_ops,
    tensor_footprints,
    total_bytes,
    total_flops,
)
from perfmodel.specs import ModelSpec, load_model

# Tiny model, small enough to check every number by hand.
# d=8, H=2, Hkv=1, hd=4 -> q=8, kv=4; F=16 gated; V=10; 16-bit everything.
TINY = ModelSpec("tiny", n_layers=1, d_model=8, n_heads=2, n_kv_heads=1, d_ff=16, vocab_size=10)


def by_name(ops):
    return {o.name: o for o in ops}


def test_tiny_decode_hand_computed():
    # B=1, P=3 cached, T=1 new -> M=1, C=4, S=3+1=4
    ops = by_name(decode_ops(TINY, batch=1, context=3))

    qkv = ops["qkv_proj"]
    assert qkv.flops == 2 * 1 * 8 * 16 + 3 * 1 * (8 + 4)  # 256 matmul + 36 RoPE
    assert qkv.weight_bytes == 8 * 16 * 2  # 256
    assert qkv.kv_bytes == 1 * 2 * 4 * 2  # write K and V: 16
    assert qkv.act_bytes == (8 + 8) * 2

    assert ops["attn_scores"].flops == 2 * 1 * 2 * 4 * 4  # 2*B*H*S*hd = 64
    assert ops["attn_scores"].kv_bytes == 1 * 4 * 4 * 2  # B*C*kv*kvb = 32
    assert ops["attn_output"].flops == 64
    assert ops["attn_softmax"].flops == 5 * 2 * 4
    assert ops["attn_softmax"].bytes == 0  # fused

    assert ops["o_proj"].flops == 2 * 8 * 8 and ops["o_proj"].weight_bytes == 128
    assert ops["mlp_up"].flops == 2 * 8 * 16 * 2 and ops["mlp_up"].weight_bytes == 512
    assert ops["mlp_down"].flops == 2 * 16 * 8 and ops["mlp_down"].weight_bytes == 256
    assert ops["lm_head"].flops == 2 * 8 * 10 and ops["lm_head"].weight_bytes == 160
    assert ops["embedding"].flops == 0 and ops["embedding"].weight_bytes == 8 * 2


def test_causal_pairs():
    assert causal_pairs(0, 1) == 1
    assert causal_pairs(5, 1) == 6  # decode attends to P + 1 positions
    assert causal_pairs(0, 4) == 1 + 2 + 3 + 4
    assert causal_pairs(10, 3) == 11 + 12 + 13


def test_prefill_lm_head_last_token_only():
    ops = by_name(prefill_ops(TINY, batch=2, prompt_len=5))
    assert ops["lm_head"].flops == 2 * 2 * 8 * 10  # B=2 rows, not B*T=10
    assert ops["qkv_proj"].flops == 2 * 10 * 8 * 16 + 3 * 10 * 12
    assert ops["attn_scores"].flops == 2 * 2 * 2 * causal_pairs(0, 5) * 4


def test_speculative_verify_needs_all_logits():
    ops = by_name(forward_ops(TINY, batch=1, n_past=10, n_new=4))
    assert ops["lm_head"].flops == 2 * 4 * 8 * 10


@pytest.mark.parametrize("name", ["llama3-8b", "llama3.2-1b", "qwen2.5-1.5b"])
def test_decode_flops_about_2x_params(name):
    """Classic rule of thumb: decode FLOPs/token ~ 2 * params (short context).

    The embedding table is gathered rather than multiplied, so for untied models
    the matmul params are total - V*d. For tied models the table is used once,
    as the LM head.
    """
    m = load_model(name)
    matmul_params = m.param_count() - (0 if m.tie_embeddings else m.vocab_size * m.d_model)
    flops = total_flops(decode_ops(m, batch=1, context=1))
    assert flops == pytest.approx(2 * matmul_params, rel=0.01)


def test_decode_bytes_about_weight_bytes_at_short_context():
    m = load_model("llama3-8b")
    b = total_bytes(decode_ops(m, batch=1, context=1))
    # Every weight read once except the untied embedding (only 1 row gathered).
    expected = (m.param_count() - m.vocab_size * m.d_model) * m.w_bytes
    assert b == pytest.approx(expected, rel=0.01)


def test_attention_grows_linearly_with_context():
    m = load_model("llama3-8b")
    a = by_name(decode_ops(m, 1, 1000))["attn_scores"]
    b = by_name(decode_ops(m, 1, 2000))["attn_scores"]
    assert b.flops / a.flops == pytest.approx(2001 / 1001)
    assert b.kv_bytes / a.kv_bytes == pytest.approx(2001 / 1001)


def test_gqa_reduces_kv_bytes_not_flops():
    mha = TINY.with_(n_kv_heads=2)
    gqa = TINY  # 1 kv head
    a, g = by_name(decode_ops(mha, 1, 100)), by_name(decode_ops(gqa, 1, 100))
    assert a["attn_scores"].flops == g["attn_scores"].flops
    assert a["attn_scores"].kv_bytes == 2 * g["attn_scores"].kv_bytes


def test_ungated_mlp():
    m = TINY.with_(gated_mlp=False)
    ops = by_name(decode_ops(m, 1, 0))
    assert ops["mlp_up"].flops == 2 * 8 * 16
    assert m.layer_params() == TINY.layer_params() - 8 * 16


def test_unfused_attention_moves_scores():
    fused = by_name(decode_ops(TINY, 1, 7))
    naive = by_name(decode_ops(TINY, 1, 7, fused_attention=False))
    S = causal_pairs(7, 1)
    assert naive["attn_softmax"].act_bytes == 2 * 2 * S * 2  # read+write B*H*S at 2 B
    assert naive["attn_scores"].flops == fused["attn_scores"].flops


def test_weights_independent_of_batch():
    a, b = by_name(decode_ops(TINY, 1, 4)), by_name(decode_ops(TINY, 64, 4))
    assert a["mlp_up"].weight_bytes == b["mlp_up"].weight_bytes
    assert b["mlp_up"].flops == 64 * a["mlp_up"].flops


def test_footprints_sum_to_params_and_kv():
    for name in ["llama3-8b", "qwen2.5-1.5b"]:
        m = load_model(name)
        fp = tensor_footprints(m, batch=4, context=128)
        kv = fp.pop(KV_CACHE)
        assert kv == m.n_layers * 4 * 128 * 2 * m.kv_dim * 2
        assert sum(fp.values()) == pytest.approx(m.param_count() * 2)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        forward_ops(TINY, 0, 0, 1)
    with pytest.raises(ValueError):
        forward_ops(TINY, 1, 0, 0)
