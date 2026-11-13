"""FLOP and byte counts for every operation in one forward step.

One "step" processes ``n_new`` new tokens for each of ``batch`` sequences that
already have ``n_past`` tokens in the KV cache:

  * decode:              n_past = context, n_new = 1
  * prefill:             n_past = 0,       n_new = prompt length
  * speculative verify:  n_past = context, n_new = k (draft tokens + 1)

Notation (all formulas below use it):
  B  = batch                 P = n_past             T = n_new
  M  = B * T  (new tokens in the step)               C = P + T (positions after step)
  d  = d_model               F = d_ff               V = vocab_size
  H  = n_heads               Hkv = n_kv_heads       hd = head_dim
  q  = H * hd                kv = Hkv * hd
  wb, kvb, ab = bytes per weight / KV element / activation element
  S  = query-key pairs per sequence per head under a causal mask:
       new token i (0-based) attends to P + i + 1 positions, so
       S = sum_{i=0}^{T-1} (P + i + 1) = T*P + T*(T+1)/2

Conventions:
  * A matmul of (M x K) by (K x N) costs 2*M*K*N FLOPs (one multiply + one add).
  * Weights are read from memory once per step and reused across the batch
    (this is what makes decode memory-bound at small batch).
  * Attention is FUSED by default (FlashAttention-style): scores and
    probabilities stay on chip, so only Q, K, V and the output move through
    memory. Pass ``fused_attention=False`` to model a naive kernel that
    writes the B*H*S score matrix out and reads it back.
  * K/V are read once per sequence per KV head; GQA query heads in the same
    group share the read. GQA does NOT reduce attention FLOPs, only KV bytes.
  * Elementwise FLOP costs (norm, softmax, SiLU, RoPE) are approximate
    conventions, stated per formula. They are tiny next to the matmuls.
  * Norm weights and biases are stored at activation precision (they are
    normally left unquantized).
"""

from __future__ import annotations

from dataclasses import dataclass

from perfmodel.specs import ModelSpec

# Tensor keys. Placement (memory.py) decides, per tensor, how much lives in SRAM.
EMBED, LM_HEAD, QKV, O_PROJ, MLP_UP, MLP_DOWN, NORMS, KV_CACHE = (
    "embed",
    "lm_head",
    "qkv",
    "o_proj",
    "mlp_up",
    "mlp_down",
    "norms",
    "kv_cache",
)


@dataclass(frozen=True)
class Op:
    """One kernel. Counts are per instance; ``count`` instances run per step."""

    name: str
    kind: str  # "matmul" (tensor cores) or "vector" (CUDA cores / vector units)
    flops: float
    weight_bytes: float = 0.0  # parameter bytes read
    weight_tensor: str | None = None  # which tensor those bytes belong to
    kv_bytes: float = 0.0  # KV-cache reads + writes
    act_bytes: float = 0.0  # activation reads + writes
    count: int = 1  # instances per step (n_layers for per-layer ops)

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes + self.act_bytes


def causal_pairs(n_past: int, n_new: int) -> float:
    """S = T*P + T*(T+1)/2, query-key pairs per sequence per head."""
    return n_new * n_past + n_new * (n_new + 1) / 2


def forward_ops(
    m: ModelSpec, batch: int, n_past: int, n_new: int, *, fused_attention: bool = True
) -> list[Op]:
    if batch < 1 or n_new < 1 or n_past < 0:
        raise ValueError("need batch >= 1, n_new >= 1, n_past >= 0")

    B, P, T = batch, n_past, n_new
    M = B * T
    C = P + T
    d, F, V = m.d_model, m.d_ff, m.vocab_size
    H, hd = m.n_heads, m.head_dim
    q, kv = m.q_dim, m.kv_dim
    wb, kvb, ab = m.w_bytes, m.kv_bytes, m.a_bytes
    S = causal_pairs(P, T)
    L = m.n_layers
    up_mats = 2 if m.gated_mlp else 1  # gate+up fused into one wide matmul

    ops: list[Op] = []

    # Embedding lookup: gather M rows of the (V x d) table. No FLOPs.
    #   weight bytes = M*d*wb (only gathered rows are read); act write = M*d*ab
    ops.append(Op("embedding", "vector", 0.0, M * d * wb, EMBED, act_bytes=M * d * ab))

    # ---- per-layer ops (count = L) ------------------------------------------
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma
    #   FLOPs ~ 4 per element (square, accumulate, scale, gamma) = 4*M*d
    #   bytes: gamma d*ab; read x + write y = 2*M*d*ab
    norm = dict(flops=4 * M * d, weight_bytes=d * ab, act_bytes=2 * M * d * ab)
    ops.append(Op("attn_norm", "vector", weight_tensor=NORMS, count=L, **norm))

    # QKV projection (+ RoPE + KV-cache append):
    #   FLOPs = 2*M*d*(q + 2kv)            matmul
    #         + M*(q + 2kv) if qkv_bias    bias add
    #         + 3*M*(q + kv)               RoPE on Q and K (2 mul + 1 add / element)
    #   weight bytes = d*(q + 2kv)*wb (+ (q + 2kv)*ab bias)
    #   act bytes = read x M*d*ab + write Q M*q*ab
    #   KV bytes  = write new K and V into the cache: M*2*kv*kvb
    qkv_out = q + 2 * kv
    qkv_flops = 2 * M * d * qkv_out + 3 * M * (q + kv)
    qkv_w = d * qkv_out * wb
    if m.qkv_bias:
        qkv_flops += M * qkv_out
        qkv_w += qkv_out * ab
    ops.append(
        Op(
            "qkv_proj",
            "matmul",
            qkv_flops,
            qkv_w,
            QKV,
            kv_bytes=M * 2 * kv * kvb,
            act_bytes=M * (d + q) * ab,
            count=L,
        )
    )

    # Attention scores Q K^T: every query head dots hd-vectors with each key it can see.
    #   FLOPs = 2*B*H*S*hd
    #   KV bytes = read K for all C positions once per sequence: B*C*kv*kvb
    #   act bytes = read Q M*q*ab (+ write scores B*H*S*ab if unfused)
    score_elems = B * H * S
    unfused = 0.0 if fused_attention else score_elems * ab
    ops.append(
        Op(
            "attn_scores",
            "matmul",
            2 * B * H * S * hd,
            kv_bytes=B * C * kv * kvb,
            act_bytes=M * q * ab + unfused,
            count=L,
        )
    )

    # Softmax over scores (scale, max, subtract, exp, sum, divide):
    #   FLOPs ~ 5 per score element = 5*B*H*S (exp counted as 1 FLOP)
    #   bytes = 0 when fused (on-chip); else read + write scores 2*B*H*S*ab
    ops.append(Op("attn_softmax", "vector", 5 * score_elems, act_bytes=2 * unfused, count=L))

    # Attention output P V:
    #   FLOPs = 2*B*H*S*hd
    #   KV bytes = read V for all C positions: B*C*kv*kvb
    #   act bytes = write context M*q*ab (+ read probabilities B*H*S*ab if unfused)
    ops.append(
        Op(
            "attn_output",
            "matmul",
            2 * B * H * S * hd,
            kv_bytes=B * C * kv * kvb,
            act_bytes=M * q * ab + unfused,
            count=L,
        )
    )

    # Output projection: (M x q) @ (q x d)
    #   FLOPs = 2*M*q*d; weights q*d*wb; act read M*q*ab + write M*d*ab
    ops.append(
        Op(
            "o_proj",
            "matmul",
            2 * M * q * d,
            q * d * wb,
            O_PROJ,
            act_bytes=M * (q + d) * ab,
            count=L,
        )
    )

    # Two residual adds per layer (after attention, after MLP):
    #   FLOPs = 2 * M*d; bytes = 2 * (read 2 inputs + write 1) = 6*M*d*ab
    ops.append(Op("residual_add", "vector", 2 * M * d, act_bytes=6 * M * d * ab, count=L))

    ops.append(Op("mlp_norm", "vector", weight_tensor=NORMS, count=L, **norm))

    # MLP up (gated: gate and up projections, 2 matrices; else 1):
    #   FLOPs = 2*M*d*F*up_mats; weights d*F*up_mats*wb
    #   act read M*d*ab + write M*F*up_mats*ab
    ops.append(
        Op(
            "mlp_up",
            "matmul",
            2 * M * d * F * up_mats,
            d * F * up_mats * wb,
            MLP_UP,
            act_bytes=M * (d + F * up_mats) * ab,
            count=L,
        )
    )

    # Activation: SiLU(g) * u for gated (SiLU ~ 4 FLOPs + 1 mul = 5 / element),
    # GELU ~ 8 FLOPs / element (tanh approximation) for ungated.
    #   act bytes = read M*F*up_mats*ab + write M*F*ab
    act_flops = (5 if m.gated_mlp else 8) * M * F
    ops.append(Op("mlp_act", "vector", act_flops, act_bytes=M * F * (up_mats + 1) * ab, count=L))

    # MLP down: (M x F) @ (F x d)
    #   FLOPs = 2*M*F*d; weights F*d*wb; act read M*F*ab + write M*d*ab
    ops.append(
        Op(
            "mlp_down",
            "matmul",
            2 * M * F * d,
            F * d * wb,
            MLP_DOWN,
            act_bytes=M * (F + d) * ab,
            count=L,
        )
    )

    # ---- head ------------------------------------------------------------------
    # Logits are needed for every new token, except in prefill (P == 0, T > 1),
    # where only the last position of each sequence is projected (as vLLM and
    # HF generate() do).
    M_lm = B if (P == 0 and T > 1) else M
    ops.append(Op("final_norm", "vector", 4 * M_lm * d, d * ab, NORMS, act_bytes=2 * M_lm * d * ab))

    # LM head: (M_lm x d) @ (d x V)
    #   FLOPs = 2*M_lm*d*V; weights d*V*wb (the embedding table if tied)
    #   act read M_lm*d*ab + write logits M_lm*V*ab
    ops.append(
        Op(
            "lm_head",
            "matmul",
            2 * M_lm * d * V,
            d * V * wb,
            EMBED if m.tie_embeddings else LM_HEAD,
            act_bytes=M_lm * (d + V) * ab,
        )
    )
    return ops


def decode_ops(m: ModelSpec, batch: int, context: int, **kw) -> list[Op]:
    """One decode step: each sequence has ``context`` tokens cached, generates 1."""
    return forward_ops(m, batch, n_past=context, n_new=1, **kw)


def prefill_ops(m: ModelSpec, batch: int, prompt_len: int, **kw) -> list[Op]:
    """Process a prompt of ``prompt_len`` tokens per sequence from an empty cache."""
    return forward_ops(m, batch, n_past=0, n_new=prompt_len, **kw)


def tensor_footprints(m: ModelSpec, batch: int, context: int) -> dict[str, float]:
    """Bytes each tensor occupies in memory (for placement), all layers combined.

    ``context`` is the number of KV positions held per sequence (P + T after the step).
    A tied LM head shares the ``embed`` footprint, so it is not listed separately.
    """
    d, L, ab, wb = m.d_model, m.n_layers, m.a_bytes, m.w_bytes
    qkv_out = m.q_dim + 2 * m.kv_dim
    fp = {
        EMBED: m.vocab_size * d * wb,
        QKV: L * (d * qkv_out * wb + (qkv_out * ab if m.qkv_bias else 0)),
        O_PROJ: L * m.q_dim * d * wb,
        MLP_UP: L * d * m.d_ff * (m.n_mlp_mats - 1) * wb,
        MLP_DOWN: L * m.d_ff * d * wb,
        NORMS: (2 * L + 1) * d * ab,
        KV_CACHE: m.kv_cache_bytes(batch, context),
    }
    if not m.tie_embeddings:
        fp[LM_HEAD] = m.vocab_size * d * wb
    return fp


def total_flops(ops: list[Op]) -> float:
    return sum(o.flops * o.count for o in ops)


def total_bytes(ops: list[Op]) -> float:
    return sum(o.bytes * o.count for o in ops)
