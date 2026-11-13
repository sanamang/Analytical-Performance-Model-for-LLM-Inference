"""Export model predictions over a grid as JSON for the static explorer page (docs/).

The page does no modelling of its own. It reads these files and applies the
same two optional knobs as ``predict_step``, which are exact because the
roofline is linear in them:

    time_op = max(t_compute / compute_eff, t_memory / bandwidth_eff) + count * overhead
    bound   = compute if t_compute / compute_eff > t_memory / bandwidth_eff else memory

``tests/test_export.py`` checks that this formula reproduces ``predict_step``.
"""

from __future__ import annotations

import json
from pathlib import Path

from perfmodel.ops import forward_ops
from perfmodel.predict import StepResult, predict_step
from perfmodel.specs import available_configs, load_chip, load_model

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CONTEXTS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
WEIGHT_BITS = [16, 8, 4]
PREFILL_BATCHES = [1, 2, 4, 8, 16]


def _r(x: float) -> float:
    """4 significant digits keeps the JSON small without visible error."""
    return float(f"{x:.4g}")


def step_record(r: StepResult) -> dict:
    """Per-op arrays (totals over each op's instances) plus SRAM placement."""
    p = r.placement
    resident = {t: p.footprint[t] * p.frac(t) for t in p.footprint}
    return {
        "tc": [_r(o.t_compute) for o in r.ops],
        "tm": [_r(o.t_memory) for o in r.ops],
        "f": [_r(o.flops) for o in r.ops],
        "sb": [_r(o.sram_bytes) for o in r.ops],
        "db": [_r(o.dram_bytes) for o in r.ops],
        "e": [_r(o.energy_j) for o in r.ops],
        "res": {t: _r(v / 2**20) for t, v in resident.items()},
        "spill": {t: _r((p.footprint[t] - resident[t]) / 2**20) for t in p.footprint},
        "act": _r(p.activation_reserve / 2**20),
    }


def apply_knobs(
    rec: dict, counts: list[int], compute_eff=1.0, bandwidth_eff=1.0, overhead_s=0.0
) -> list[float]:
    """The explorer page's per-op time formula (mirrored in docs/index.html)."""
    return [
        max(tc / compute_eff, tm / bandwidth_eff) + n * overhead_s
        for tc, tm, n in zip(rec["tc"], rec["tm"], counts, strict=True)
    ]


def build(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    meta = {
        "models": {},
        "chips": {},
        "grid": {
            "batches": BATCHES,
            "contexts": CONTEXTS,
            "weight_bits": WEIGHT_BITS,
            "prefill_batches": PREFILL_BATCHES,
        },
    }

    for mk in available_configs("models"):
        m = load_model(mk)
        meta["models"][mk] = {
            "name": m.name,
            "params_b": round(m.param_count() / 1e9, 3),
            "layers": m.n_layers,
            "d_model": m.d_model,
            "heads": m.n_heads,
            "kv_heads": m.n_kv_heads,
            "d_ff": m.d_ff,
            "vocab": m.vocab_size,
            "compute_dtype": m.compute_dtype,
            "tied": m.tie_embeddings,
        }
    for ck in available_configs("chips"):
        c = load_chip(ck)
        meta["chips"][ck] = {
            "name": c.name,
            "peak_tflops": c.peak_tflops,
            "vector_dtype": c.vector_dtype,
            "dram_gbps": c.dram_gbps,
            "sram_gbps": c.sram_gbps,
            "sram_mib": c.sram_mib,
            "sram_persistent": c.sram_persistent,
            "hypothetical": "hypothetical" in c.name.lower(),
        }

    for mk in meta["models"]:
        base = load_model(mk)
        ops = forward_ops(base, 1, 1, 1)
        for ck in meta["chips"]:
            chip = load_chip(ck)
            points = {}
            for wb in WEIGHT_BITS:
                m = base.with_(weight_bits=wb)
                for c in CONTEXTS:
                    for b in BATCHES:
                        points[f"decode|{wb}|{b}|{c}"] = step_record(predict_step(m, chip, b, c, 1))
                    for b in PREFILL_BATCHES:
                        points[f"prefill|{wb}|{b}|{c}"] = step_record(
                            predict_step(m, chip, b, 0, c)
                        )
            data = {
                "model": mk,
                "chip": ck,
                "ops": [o.name for o in ops],
                "kinds": [o.kind for o in ops],
                "counts": [o.count for o in ops],
                "points": points,
            }
            path = out_dir / f"{mk}__{ck}.json"
            path.write_text(json.dumps(data, separators=(",", ":")))
            written.append(path)

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    written.append(meta_path)
    return written
