"""Where each tensor lives: on-chip SRAM or off-chip DRAM.

The placement is fractional per tensor group: ``sram_fraction["mlp_up"] = 0.5``
means half of the MLP-up weights (about half the layers) are resident in SRAM.
Traffic to that fraction is served at SRAM bandwidth and energy, and the
rest at DRAM bandwidth and energy.

Policies (pass ``policy=`` to :func:`place`):

* ``none``          Nothing resident. Every weight and KV byte comes from DRAM.
                    This is the right model for GPUs, whose L2 is a hardware
                    cache that per-step weight streaming evicts.
* ``greedy``        Fill SRAM in order of traffic density (bytes read per step
                    / bytes of footprint). This is the fractional-knapsack
                    optimum for minimising DRAM traffic in one step. Ties go to
                    weights, which are static across requests while KV churns.
* ``weights_first`` All weights (densest first), then the KV cache.
* ``kv_first``      KV cache first, then weights. Useful at long context.

Activations: on a chip with persistent SRAM (and policy != none), the model
reserves a scratch buffer as large as the biggest single op's activation
traffic, before placing anything else. If that buffer doesn't fit,
activations go to DRAM. On other chips, activation traffic counts as DRAM
traffic.

Add a new policy by adding a branch to ``_order``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from perfmodel.ops import KV_CACHE, Op
from perfmodel.specs import ChipSpec

POLICIES = ("none", "greedy", "weights_first", "kv_first")


@dataclass
class Placement:
    policy: str
    capacity: float  # SRAM bytes
    activation_reserve: float  # bytes set aside for activations
    activations_in_sram: bool
    footprint: dict[str, float]
    traffic: dict[str, float]  # bytes read/written per step, per tensor
    sram_fraction: dict[str, float] = field(default_factory=dict)

    def frac(self, tensor: str | None) -> float:
        return self.sram_fraction.get(tensor, 0.0) if tensor else 0.0

    @property
    def sram_used(self) -> float:
        placed = sum(self.footprint[t] * f for t, f in self.sram_fraction.items())
        return placed + self.activation_reserve

    def split(self, op: Op) -> tuple[float, float]:
        """(sram_bytes, dram_bytes) for one instance of ``op``."""
        sram = op.weight_bytes * self.frac(op.weight_tensor) + op.kv_bytes * self.frac(KV_CACHE)
        if self.activations_in_sram:
            sram += op.act_bytes
        return sram, op.bytes - sram


def tensor_traffic(ops: list[Op]) -> dict[str, float]:
    traffic: dict[str, float] = {}
    for o in ops:
        if o.weight_tensor and o.weight_bytes:
            traffic[o.weight_tensor] = traffic.get(o.weight_tensor, 0.0) + o.weight_bytes * o.count
        if o.kv_bytes:
            traffic[KV_CACHE] = traffic.get(KV_CACHE, 0.0) + o.kv_bytes * o.count
    return traffic


def default_policy(chip: ChipSpec) -> str:
    return "greedy" if chip.sram_persistent else "none"


def _order(policy: str, footprint: dict[str, float], traffic: dict[str, float]) -> list[str]:
    def density(t: str) -> float:
        return traffic.get(t, 0.0) / footprint[t] if footprint[t] else 0.0

    weights = sorted((t for t in footprint if t != KV_CACHE), key=density, reverse=True)
    if policy == "none":
        return []
    if policy == "weights_first":
        return [*weights, KV_CACHE]
    if policy == "kv_first":
        return [KV_CACHE, *weights]
    if policy == "greedy":
        # Stable sort on -density with weights listed first breaks ties toward weights.
        return sorted([*weights, KV_CACHE], key=lambda t: -density(t))
    raise ValueError(f"unknown placement policy {policy!r}; choose from {POLICIES}")


def place(
    ops: list[Op], footprint: dict[str, float], chip: ChipSpec, policy: str | None = None
) -> Placement:
    policy = policy or default_policy(chip)
    if policy not in POLICIES:
        raise ValueError(f"unknown placement policy {policy!r}; choose from {POLICIES}")
    if policy != "none" and not chip.sram_persistent:
        raise ValueError(
            f"{chip.name} has no software-managed SRAM (sram_persistent: false); "
            "use policy 'none' or set sram_persistent/sram_gbps in the chip config"
        )

    traffic = tensor_traffic(ops)
    capacity = chip.sram_bytes if policy != "none" else 0.0

    reserve = max((o.act_bytes for o in ops), default=0.0) if policy != "none" else 0.0
    acts_in_sram = policy != "none" and reserve <= capacity
    if not acts_in_sram:
        reserve = 0.0

    remaining = capacity - reserve
    fractions: dict[str, float] = {}
    for t in _order(policy, footprint, traffic):
        size = footprint.get(t, 0.0)
        if size <= 0 or remaining <= 0:
            fractions[t] = 0.0 if size > 0 else 1.0
            continue
        f = min(1.0, remaining / size)
        fractions[t] = f
        remaining -= f * size

    return Placement(policy, capacity, reserve, acts_in_sram, footprint, traffic, fractions)
