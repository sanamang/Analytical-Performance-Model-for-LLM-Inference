"""Roofline timing: an op takes as long as its slowest resource.

Single level (Williams, Waterman & Patterson, CACM 2009):
    t = max(flops / peak_flops, bytes / bandwidth)

Two memory levels (hierarchical roofline, Yang et al., 2020): SRAM and DRAM
traffic are separate resources that run concurrently, so
    t_memory = max(sram_bytes / sram_bw, dram_bytes / dram_bw)
    t        = max(t_compute, t_memory)

This is a lower bound on time: perfect overlap, perfect utilization, no launch
overhead. predict.py adds optional efficiency factors and per-kernel overhead
on top; this module stays pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """FLOPs per byte moved. inf for pure compute, 0 for pure data movement."""
    if nbytes == 0:
        return math.inf if flops > 0 else 0.0
    return flops / nbytes


def ridge_point(peak_flops: float, bandwidth: float) -> float:
    """Intensity (FLOP/byte) at which an op moves from memory- to compute-bound."""
    return peak_flops / bandwidth


def attainable_flops(intensity: float, peak_flops: float, bandwidth: float) -> float:
    """The roofline itself: min(peak, intensity * bandwidth)."""
    return min(peak_flops, intensity * bandwidth)


@dataclass(frozen=True)
class OpTiming:
    t_compute: float
    t_sram: float
    t_dram: float

    @property
    def t_memory(self) -> float:
        return max(self.t_sram, self.t_dram)

    @property
    def time(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound(self) -> str:
        """'compute', 'memory', 'balanced' (exactly at the ridge) or 'none' (empty op)."""
        if self.t_compute == 0 and self.t_memory == 0:
            return "none"
        if self.t_compute > self.t_memory:
            return "compute"
        if self.t_memory > self.t_compute:
            return "memory"
        return "balanced"

    @property
    def memory_level(self) -> str | None:
        """Which memory is the bottleneck, if memory-bound."""
        if self.bound != "memory":
            return None
        return "dram" if self.t_dram >= self.t_sram else "sram"


def op_time(
    flops: float,
    peak_flops: float,
    dram_bytes: float,
    dram_bw: float,
    sram_bytes: float = 0.0,
    sram_bw: float | None = None,
) -> OpTiming:
    if peak_flops <= 0 or dram_bw <= 0:
        raise ValueError("peak_flops and dram_bw must be positive")
    if min(flops, dram_bytes, sram_bytes) < 0:
        raise ValueError("flops and bytes must be non-negative")
    if sram_bytes > 0 and not sram_bw:
        raise ValueError("SRAM traffic given but chip has no SRAM bandwidth")
    t_sram = sram_bytes / sram_bw if sram_bytes else 0.0
    return OpTiming(flops / peak_flops, t_sram, dram_bytes / dram_bw)
