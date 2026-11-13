"""Model and chip descriptions, loaded from YAML.

Units used throughout the package (SI unless noted):
  * time in seconds, energy in joules, FLOPs as raw counts, bytes as raw counts
  * YAML uses human units: TFLOP/s (1e12), GB/s (1e9, vendor convention),
    MiB (2**20) for on-chip SRAM, pJ (1e-12) for energy.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(
    os.environ.get("PERFMODEL_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
)

# Keys allowed in YAML that are documentation only (not model inputs).
_METADATA_KEYS = {"sources", "notes", "estimated"}


@dataclass(frozen=True)
class ModelSpec:
    """Decoder-only transformer (Llama/Qwen style: pre-norm, RMSNorm, RoPE, GQA)."""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    head_dim: int | None = None  # defaults to d_model // n_heads
    gated_mlp: bool = True  # SwiGLU: gate & up projections (3 matrices) vs 2
    tie_embeddings: bool = False  # LM head shares the embedding matrix
    qkv_bias: bool = False  # Qwen2 has biases on Q/K/V projections
    weight_bits: float = 16  # bits per matmul/embedding weight (may be fractional,
    #                          e.g. 4.25 to include group-wise scale overhead)
    kv_bits: float = 16  # bits per KV-cache element
    act_bits: float = 16  # bits per activation element moved through memory
    compute_dtype: str = "fp16"  # precision the matmuls execute in (key into peak_tflops)
    sources: list[str] = field(default_factory=list, compare=False)
    notes: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.d_model % self.n_heads:
                raise ValueError("d_model must be divisible by n_heads when head_dim is unset")
            object.__setattr__(self, "head_dim", self.d_model // self.n_heads)
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads (GQA groups)")
        for name in ("weight_bits", "kv_bits", "act_bits"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    # --- derived sizes -------------------------------------------------------
    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Width of K (or V) per token: n_kv_heads * head_dim."""
        return self.n_kv_heads * self.head_dim

    @property
    def w_bytes(self) -> float:
        return self.weight_bits / 8

    @property
    def kv_bytes(self) -> float:
        return self.kv_bits / 8

    @property
    def a_bytes(self) -> float:
        return self.act_bits / 8

    @property
    def n_mlp_mats(self) -> int:
        return 3 if self.gated_mlp else 2

    def layer_params(self) -> int:
        """Parameters in one transformer block (matmuls + biases + 2 norm vectors)."""
        d = self.d_model
        qkv = d * (self.q_dim + 2 * self.kv_dim)
        bias = (self.q_dim + 2 * self.kv_dim) if self.qkv_bias else 0
        o = self.q_dim * d
        mlp = self.n_mlp_mats * d * self.d_ff
        norms = 2 * d
        return qkv + bias + o + mlp + norms

    def param_count(self) -> int:
        """Total parameters, counting a tied embedding/LM-head matrix once."""
        embed = self.vocab_size * self.d_model
        lm_head = 0 if self.tie_embeddings else embed
        return embed + self.n_layers * self.layer_params() + self.d_model + lm_head

    def kv_cache_bytes(self, batch: int, context: int) -> float:
        """K and V for every layer, sequence and position."""
        return self.n_layers * batch * context * 2 * self.kv_dim * self.kv_bytes

    def with_(self, **changes: Any) -> ModelSpec:
        """Copy with fields changed, e.g. ``m.with_(weight_bits=4)``."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class ChipSpec:
    """Accelerator description for a two-level (SRAM + DRAM) roofline."""

    name: str
    peak_tflops: dict[str, float]  # dense (no sparsity) peak per precision
    dram_gbps: float  # DRAM bandwidth, GB/s (1e9)
    sram_mib: float  # on-chip SRAM usable for placement, MiB
    sram_persistent: bool  # True: software-managed scratchpad that can keep weights/KV
    #                        resident across steps. False: hardware cache (GPU L2) that
    #                        streaming weights evict, so the model treats it as absent.
    energy_pj_per_flop: dict[str, float]  # per precision
    dram_pj_per_byte: float
    sram_gbps: float | None = None  # required only if sram_persistent
    sram_pj_per_byte: float | None = None  # required only if sram_persistent
    dram_gib: float | None = None  # DRAM capacity (informational / OOM checks)
    vector_dtype: str = "fp32"  # precision for non-matmul ops (norms, softmax, ...)
    static_power_w: float = 0.0  # leakage/idle power; not published, set from measurement
    sources: list[str] = field(default_factory=list, compare=False)
    notes: str = field(default="", compare=False)
    estimated: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        if self.sram_persistent and (self.sram_gbps is None or self.sram_pj_per_byte is None):
            raise ValueError(f"{self.name}: persistent SRAM needs sram_gbps and sram_pj_per_byte")

    def peak_flops(self, dtype: str) -> float:
        try:
            return self.peak_tflops[dtype] * 1e12
        except KeyError:
            raise KeyError(
                f"{self.name} has no peak for {dtype!r}; known: {sorted(self.peak_tflops)}"
            ) from None

    def pj_per_flop(self, dtype: str) -> float:
        try:
            return self.energy_pj_per_flop[dtype]
        except KeyError:
            raise KeyError(
                f"{self.name} has no energy/FLOP for {dtype!r}; "
                f"known: {sorted(self.energy_pj_per_flop)}"
            ) from None

    @property
    def dram_bw(self) -> float:
        return self.dram_gbps * 1e9

    @property
    def sram_bw(self) -> float | None:
        return None if self.sram_gbps is None else self.sram_gbps * 1e9

    @property
    def sram_bytes(self) -> float:
        return self.sram_mib * 2**20

    def with_(self, **changes: Any) -> ChipSpec:
        return dataclasses.replace(self, **changes)


# --- YAML loading -----------------------------------------------------------


def _resolve(name_or_path: str | Path, kind: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = CONFIG_DIR / kind / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    available = ", ".join(available_configs(kind)) or "(none)"
    raise FileNotFoundError(f"No {kind[:-1]} config {name_or_path!r}. Available: {available}")


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known - _METADATA_KEYS
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_model(name_or_path: str | Path) -> ModelSpec:
    with open(_resolve(name_or_path, "models")) as f:
        return _from_dict(ModelSpec, yaml.safe_load(f))


def load_chip(name_or_path: str | Path) -> ChipSpec:
    with open(_resolve(name_or_path, "chips")) as f:
        return _from_dict(ChipSpec, yaml.safe_load(f))


def available_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / kind).glob("*.yaml"))
