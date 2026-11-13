"""Glue: ops -> placement -> roofline time + energy, per op and per step.

The pure model is ``predict_step(...)`` with default arguments: perfect
roofline, no kernel-launch overhead. Two optional knobs exist so that
validation can separate "the physics is wrong" from "the software stack has
overhead". They are never applied silently:

  * ``compute_efficiency`` / ``bandwidth_efficiency``: fraction of peak achieved.
  * ``overhead_per_op_s``: fixed cost per kernel instance (launch + framework).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from perfmodel.energy import op_energy
from perfmodel.memory import Placement, place
from perfmodel.ops import forward_ops, tensor_footprints
from perfmodel.roofline import arithmetic_intensity, op_time
from perfmodel.specs import ChipSpec, ModelSpec


@dataclass(frozen=True)
class OpResult:
    """One op row. FLOPs, bytes, time and energy are totals over ``count`` instances."""

    name: str
    kind: str
    count: int
    flops: float
    sram_bytes: float
    dram_bytes: float
    t_compute: float
    t_memory: float
    time: float
    bound: str
    energy_j: float

    @property
    def intensity(self) -> float:
        return arithmetic_intensity(self.flops, self.sram_bytes + self.dram_bytes)


@dataclass
class StepResult:
    phase: str  # "prefill", "decode" or "verify"
    model: str
    chip: str
    batch: int
    n_past: int
    n_new: int
    ops: list[OpResult]
    placement: Placement
    static_energy_j: float

    @property
    def time_s(self) -> float:
        return sum(o.time for o in self.ops)

    @property
    def tokens(self) -> int:
        """Tokens processed in this step (B * T)."""
        return self.batch * self.n_new

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.time_s

    @property
    def energy_j(self) -> float:
        return sum(o.energy_j for o in self.ops) + self.static_energy_j

    @property
    def energy_per_token_j(self) -> float:
        return self.energy_j / self.tokens

    @property
    def flops(self) -> float:
        return sum(o.flops for o in self.ops)

    @property
    def bytes(self) -> float:
        return sum(o.sram_bytes + o.dram_bytes for o in self.ops)

    @property
    def dram_bytes(self) -> float:
        return sum(o.dram_bytes for o in self.ops)

    @property
    def compute_bound_fraction(self) -> float:
        """Share of step time spent in compute-bound ops."""
        t = self.time_s
        return sum(o.time for o in self.ops if o.bound == "compute") / t if t else 0.0

    @property
    def n_kernels(self) -> int:
        return sum(o.count for o in self.ops)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "op": o.name,
                "kind": o.kind,
                "count": o.count,
                "GFLOP": o.flops / 1e9,
                "SRAM_MB": o.sram_bytes / 1e6,
                "DRAM_MB": o.dram_bytes / 1e6,
                "intensity": o.intensity,
                "time_us": o.time * 1e6,
                "time_%": 100 * o.time / self.time_s,
                "bound": o.bound,
                "energy_mJ": o.energy_j * 1e3,
            }
            for o in self.ops
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        return {
            "phase": self.phase,
            "model": self.model,
            "chip": self.chip,
            "batch": self.batch,
            "n_past": self.n_past,
            "n_new": self.n_new,
            "time_s": self.time_s,
            "tokens_per_s": self.tokens_per_s,
            "energy_j": self.energy_j,
            "energy_per_token_j": self.energy_per_token_j,
            "gflop": self.flops / 1e9,
            "gbytes": self.bytes / 1e9,
            "dram_gbytes": self.dram_bytes / 1e9,
            "intensity": arithmetic_intensity(self.flops, self.bytes),
            "compute_bound_fraction": self.compute_bound_fraction,
            "sram_used_mib": self.placement.sram_used / 2**20,
            "n_kernels": self.n_kernels,
        }


def predict_step(
    model: ModelSpec,
    chip: ChipSpec,
    batch: int,
    n_past: int,
    n_new: int,
    *,
    placement_policy: str | None = None,
    fused_attention: bool = True,
    compute_efficiency: float = 1.0,
    bandwidth_efficiency: float = 1.0,
    overhead_per_op_s: float = 0.0,
) -> StepResult:
    """Predict one forward step: ``n_new`` tokens per sequence on top of ``n_past`` cached."""
    if not (0 < compute_efficiency <= 1 and 0 < bandwidth_efficiency <= 1):
        raise ValueError("efficiencies must be in (0, 1]")

    ops = forward_ops(model, batch, n_past, n_new, fused_attention=fused_attention)
    placement = place(ops, tensor_footprints(model, batch, n_past + n_new), chip, placement_policy)

    sram_bw = chip.sram_bw * bandwidth_efficiency if chip.sram_bw else None
    dram_bw = chip.dram_bw * bandwidth_efficiency

    results = []
    for o in ops:
        dtype = model.compute_dtype if o.kind == "matmul" else chip.vector_dtype
        peak = chip.peak_flops(dtype) * compute_efficiency
        sram_b, dram_b = placement.split(o)
        t = op_time(o.flops, peak, dram_b, dram_bw, sram_b, sram_bw)
        e = op_energy(
            o.flops,
            chip.pj_per_flop(dtype),
            dram_b,
            chip.dram_pj_per_byte,
            sram_b,
            chip.sram_pj_per_byte,
        )
        n = o.count
        results.append(
            OpResult(
                name=o.name,
                kind=o.kind,
                count=n,
                flops=o.flops * n,
                sram_bytes=sram_b * n,
                dram_bytes=dram_b * n,
                t_compute=t.t_compute * n,
                t_memory=t.t_memory * n,
                time=(t.time + overhead_per_op_s) * n,
                bound=t.bound,
                energy_j=e.total_j * n,
            )
        )

    phase = "prefill" if n_past == 0 and n_new > 1 else ("decode" if n_new == 1 else "verify")
    total_time = sum(r.time for r in results)
    return StepResult(
        phase=phase,
        model=model.name,
        chip=chip.name,
        batch=batch,
        n_past=n_past,
        n_new=n_new,
        ops=results,
        placement=placement,
        static_energy_j=chip.static_power_w * total_time,
    )


def predict(model: ModelSpec, chip: ChipSpec, batch: int, context: int, **kw) -> dict:
    """Prefill of a ``context``-token prompt, and one decode step at ``context`` cached tokens."""
    return {
        "prefill": predict_step(model, chip, batch, 0, context, **kw),
        "decode": predict_step(model, chip, batch, context, 1, **kw),
    }


def sweep(
    model: ModelSpec,
    chip: ChipSpec,
    batches: list[int],
    contexts: list[int],
    weight_bits: list[float] | None = None,
    phases: tuple[str, ...] = ("prefill", "decode"),
    **kw,
) -> pd.DataFrame:
    """Grid over batch x context x weight precision. One row per (point, phase)."""
    rows = []
    for wbits, b, c in itertools.product(weight_bits or [model.weight_bits], batches, contexts):
        m = model.with_(weight_bits=wbits)
        res = predict(m, chip, b, c, **kw)
        for ph in phases:
            rows.append({"weight_bits": wbits, "context": c, **res[ph].summary()})
    return pd.DataFrame(rows)
