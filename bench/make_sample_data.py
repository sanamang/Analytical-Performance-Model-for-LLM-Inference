#!/usr/bin/env python
"""Write a SYNTHETIC sample CSV in the same format as bench_decode.py output.

THESE ARE NOT MEASUREMENTS. The file exists so the notebooks and plots can be
built and tested before real GPU data exists. Every row has ``synthetic = 1``,
and the notebooks print a warning when they see it.

How rows are generated: the perfmodel roofline prediction is distorted with
the planted parameters below, then multiplied by log-normal noise. The
planted values are placeholders chosen to look like Hugging Face eager-mode
behaviour. They are not hardware numbers. Because the validation notebook fits
the same functional form, a good fit on this file is CIRCULAR and proves
nothing about the model. Only real measurements can validate it.

    uv run python bench/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from perfmodel import load_chip, load_model, predict_step

PLANTED = {
    "bandwidth_efficiency": 0.80,  # fraction of DRAM peak reached by kernels
    "compute_efficiency": 0.60,  # fraction of tensor-core peak
    "overhead_per_op_s": 20e-6,  # per-kernel CPU/launch overhead (HF eager)
    "idle_power_w": 60.0,  # static power
    "dynamic_energy_scale": 2.5,  # real energy / Horowitz-45nm dynamic energy
    "noise_sigma": 0.04,  # log-normal noise on time
}


def main(out: Path = Path("bench/results/sample_synthetic.csv"), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    m, chip = load_model("qwen2.5-1.5b"), load_chip("a100")
    p = PLANTED
    rows = []
    grid = [("decode", b, c) for b in [1, 2, 4, 8, 16, 32, 64, 128]
            for c in [128, 512, 1024, 2048, 4096]]
    grid += [("prefill", b, c) for b in [1, 4] for c in [128, 512, 1024, 2048, 4096]]
    for phase, b, c in grid:
        n_past, n_new = (c, 1) if phase == "decode" else (0, c)
        r = predict_step(m, chip, b, n_past, n_new,
                         bandwidth_efficiency=p["bandwidth_efficiency"],
                         compute_efficiency=p["compute_efficiency"],
                         overhead_per_op_s=p["overhead_per_op_s"])
        t = r.time_s * rng.lognormal(0, p["noise_sigma"])
        tokens = b * n_new
        e = p["dynamic_energy_scale"] * r.energy_j + p["idle_power_w"] * t
        rows.append({
            "timestamp": "synthetic",
            "gpu_name": "SYNTHETIC (A100-SXM4-80GB-like)",
            "gpu_mem_gib": 80,
            "model_id": "Qwen/Qwen2.5-1.5B",
            "perfmodel_model": "qwen2.5-1.5b",
            "perfmodel_chip": "a100",
            "dtype": "bfloat16",
            "attn_impl": "sdpa",
            "synthetic": 1,
            "idle_power_w": p["idle_power_w"],
            "phase": phase,
            "batch": b,
            "context": c,
            "n_steps": 100,
            "step_ms_median": t * 1e3,
            "tokens_per_s": tokens / t,
            "avg_power_w": e / t,
            "energy_per_token_j": e / tokens,
            "energy_method": "synthetic",
            "energy_reliable": True,
            "fixed_context": True,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} SYNTHETIC rows -> {out}")


if __name__ == "__main__":
    main()
