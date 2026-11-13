"""Energy = FLOPs * e_flop + SRAM bytes * e_sram + DRAM bytes * e_dram (+ static power * time).

Default per-operation energies are APPROXIMATE. They come from

    M. Horowitz, "Computing's Energy Problem (and what we can do about it)",
    ISSCC 2014, keynote. 45 nm process, ~0.9 V.

and are copied into the chip YAMLs. A modern 4-7 nm accelerator spends several
times less energy per operation, so absolute predictions will be off. The
useful output is the ratio between compute and data movement, which is what
drives design decisions. Do not treat a predicted joule figure as a measurement.

Horowitz 2014 table (45 nm):
    8-bit int:  add 0.03 pJ,  mult 0.2 pJ
    32-bit int: add 0.1 pJ,   mult 3.1 pJ
    16-bit FP:  add 0.4 pJ,   mult 1.1 pJ
    32-bit FP:  add 0.9 pJ,   mult 3.7 pJ
    SRAM read (64 bit): 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ
    DRAM (64 bit access): 1.3 - 2.6 nJ
HBM2 (O'Connor et al., "Fine-Grained DRAM", MICRO 2017): ~3.9 pJ/bit.

A FLOP here is half of a multiply-accumulate, so pJ/FLOP = (mult + add) / 2.
"""

from __future__ import annotations

from dataclasses import dataclass

HOROWITZ_2014_PJ = {
    "int8_add": 0.03,
    "int8_mult": 0.2,
    "int32_add": 0.1,
    "int32_mult": 3.1,
    "fp16_add": 0.4,
    "fp16_mult": 1.1,
    "fp32_add": 0.9,
    "fp32_mult": 3.7,
    "sram_8kb_64b": 10.0,
    "sram_32kb_64b": 20.0,
    "sram_1mb_64b": 100.0,
    "dram_64b_low": 1300.0,
    "dram_64b_high": 2600.0,
}
HBM2_PJ_PER_BIT = 3.9  # O'Connor et al. MICRO 2017 (approximate)


def pj_per_flop(dtype: str) -> float:
    """Default pJ per FLOP derived from the Horowitz table (one MAC = 2 FLOPs)."""
    base = {"fp16": "fp16", "bf16": "fp16", "fp32": "fp32", "int8": "int8"}[dtype]
    return (HOROWITZ_2014_PJ[f"{base}_mult"] + HOROWITZ_2014_PJ[f"{base}_add"]) / 2


@dataclass(frozen=True)
class Energy:
    compute_j: float
    sram_j: float
    dram_j: float

    @property
    def total_j(self) -> float:
        return self.compute_j + self.sram_j + self.dram_j


def op_energy(
    flops: float,
    pj_flop: float,
    dram_bytes: float,
    dram_pj_byte: float,
    sram_bytes: float = 0.0,
    sram_pj_byte: float | None = None,
) -> Energy:
    if sram_bytes > 0 and sram_pj_byte is None:
        raise ValueError("SRAM traffic given but chip has no SRAM energy figure")
    return Energy(
        compute_j=flops * pj_flop * 1e-12,
        sram_j=sram_bytes * (sram_pj_byte or 0.0) * 1e-12,
        dram_j=dram_bytes * dram_pj_byte * 1e-12,
    )
