"""Generate the analysis notebooks from source, so the cells are reviewable in diffs.

    uv run python notebooks/build_notebooks.py          # write .ipynb files
    uv run python notebooks/build_notebooks.py --run    # ... and execute them in place

Point the validation notebook at real data with
    PERFMODEL_BENCH_CSV=bench/results/<your file>.csv uv run python notebooks/build_notebooks.py --run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """\
import sys, os
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "notebooks"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotstyle as ps
from perfmodel import load_model, load_chip, predict_step
ps.apply()
pd.set_option("display.precision", 3)"""


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


# =============================================================================
# (a) Validation
# =============================================================================
VALIDATION = [
    md("""
# (a) Validation: predicted vs measured throughput and energy

This notebook compares `perfmodel` predictions with measurements from
`bench/bench_decode.py`. It reads the CSV named by the `PERFMODEL_BENCH_CSV`
environment variable and falls back to `bench/results/sample_synthetic.csv`.

> **About the sample file.** `sample_synthetic.csv` is **not** a measurement.
> It is the model's own prediction distorted with planted parameters
> (bandwidth and compute efficiency, per-kernel overhead, static power, noise;
> see `bench/make_sample_data.py`). On that file, the calibration below
> recovers the planted values *by construction*, so the fit quality is
> circular and validates nothing. It is here so the plots and statistics work
> before real GPU data exists.
"""),
    code(SETUP),
    code("""
csv = Path(os.environ.get("PERFMODEL_BENCH_CSV", ROOT / "bench/results/sample_synthetic.csv"))
if not csv.is_absolute():
    csv = ROOT / csv
meas = pd.read_csv(csv)
SYNTHETIC = bool(meas.get("synthetic", pd.Series([0])).astype(int).any())
print(f"{csv.name}: {len(meas)} rows, GPUs: {meas.gpu_name.unique().tolist()}")
if SYNTHETIC:
    print("WARNING: SYNTHETIC data. Error statistics below are NOT a validation result.")
meas.head()
"""),
    md("""
## Predictions for every measured point

Each measured row maps to one model step:
* decode at context `C` = one new token on top of `C` cached tokens (the bench
  crops the cache so the context stays fixed);
* prefill of `C` tokens = `n_past = 0`, `n_new = C`.

The **pure** prediction is the roofline with 100% efficiency and no overhead.
"""),
    code("""
def predict_row(r, **kw):
    m, c = load_model(r.perfmodel_model), load_chip(r.perfmodel_chip)
    n_past, n_new = (int(round(r.context)), 1) if r.phase == "decode" else (0, int(r.context))
    return predict_step(m, c, int(r.batch), n_past, n_new, **kw)

preds = [predict_row(r) for r in meas.itertuples()]
df = meas.copy()
df["t_meas"] = df.step_ms_median / 1e3
df["t_pred"] = [p.time_s for p in preds]
df["tps_pred"] = [p.tokens_per_s for p in preds]
df["n_kernels"] = [p.n_kernels for p in preds]
df["e_dyn_pred"] = [p.energy_j for p in preds]                  # J per step, dynamic only
df["e_meas"] = df.energy_per_token_j * df.batch * np.where(df.phase == "decode", 1, df.context)
df["ratio"] = df.tokens_per_s / df.tps_pred                      # < 1: slower than predicted

def err_stats(pred, true):
    ape = np.abs(pred - true) / true
    return pd.Series({"MAPE_%": 100 * ape.mean(), "median_APE_%": 100 * np.median(ape),
                      "max_APE_%": 100 * ape.max(), "within_20%": (ape <= 0.2).mean()})

pure = df.groupby("phase").apply(lambda g: err_stats(g.tps_pred, g.tokens_per_s),
                                 include_groups=False)
pure
"""),
    md("""
## Calibrated model: two physically meaningful parameters

A pure roofline is a lower bound on time. Real kernels reach only part of peak
bandwidth, and every kernel launch pays a fixed cost. For Hugging Face eager
mode that cost is mostly Python and CPU dispatch. Model it as

$$ t_\\text{meas} \\approx \\alpha \\, t_\\text{roofline} + \\beta \\, n_\\text{kernels} $$

where $1/\\alpha$ is the achieved fraction of the roofline and $\\beta$ is the
overhead per kernel. Fit on **decode rows with context ≤ 1024** and test on
the **longer contexts**. The held-out rows check whether the calibration
extrapolates or just memorizes.
"""),
    code("""
dec = df[df.phase == "decode"].copy()
train = dec[dec.context <= 1024]
test = dec[dec.context > 1024]
X = np.c_[train.t_pred, train.n_kernels]
(alpha, beta), *_ = np.linalg.lstsq(X, train.t_meas, rcond=None)
print(f"alpha = {alpha:.3f}  (-> {100/alpha:.0f}% of roofline)   beta = {beta*1e6:.1f} us/kernel")

for d in (df,):
    d["t_cal"] = alpha * d.t_pred + beta * d.n_kernels
    d["tps_cal"] = d.tps_pred * d.t_pred / d.t_cal
dec = df[df.phase == "decode"]
pd.DataFrame({
    "pure, all decode": err_stats(dec.tps_pred, dec.tokens_per_s),
    "calibrated, train (C<=1024)": err_stats(dec[dec.context <= 1024].tps_cal,
                                             dec[dec.context <= 1024].tokens_per_s),
    "calibrated, HELD-OUT (C>1024)": err_stats(dec[dec.context > 1024].tps_cal,
                                               dec[dec.context > 1024].tokens_per_s),
}).T
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
lo = min(df.tokens_per_s.min(), df.tps_pred.min(), df.tps_cal.min()) * 0.6
hi = max(df.tokens_per_s.max(), df.tps_pred.max(), df.tps_cal.max()) * 1.6
for ax, col, title in [(axes[0], "tps_pred", "Pure roofline"),
                       (axes[1], "tps_cal", "Calibrated (alpha, beta fit on C <= 1024)")]:
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs / 1.2, xs * 1.2, color=ps.GRID, alpha=0.6, lw=0, label="±20%")
    ax.plot(xs, xs, color=ps.NEUTRAL, lw=1, ls="--", label="y = x")
    for i, ph in enumerate(["decode", "prefill"]):
        g = df[df.phase == ph]
        ax.scatter(g[col], g.tokens_per_s, s=34, color=ps.SERIES[i], label=ph,
                   edgecolor=ps.SURFACE, linewidth=1.2, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), title=title,
           xlabel="predicted tokens/s")
axes[0].set_ylabel("measured tokens/s")
axes[1].legend(loc="upper left")
tag = "  [SYNTHETIC SAMPLE DATA, not a measurement]" if SYNTHETIC else ""
fig.suptitle(f"{df.perfmodel_model.iloc[0]} on {df.gpu_name.iloc[0]}{tag}", fontsize=10,
             color=ps.INK_2)
ps.save(fig, "validation_tokens_per_s");
"""),
    md("""
### Where the error lives: measured / predicted vs batch

A ratio of 1 means the pure roofline is exact. Values far below 1 at small
batch mean fixed overhead dominates, not memory or compute.
"""),
    code("""
fig, ax = plt.subplots(figsize=(6.4, 4))
for i, (c, g) in enumerate(dec.groupby("context")):
    g = g.sort_values("batch")
    ax.plot(g.batch, g.ratio, marker="o", color=ps.SERIES[i], label=f"context {int(c)}")
ax.axhline(1, color=ps.NEUTRAL, lw=1, ls="--")
ax.set(xscale="log", xlabel="batch size", ylabel="measured / predicted tokens/s",
       title="Decode: fraction of roofline throughput achieved", ylim=(0, 1.15))
ax.set_xticks(sorted(dec.batch.unique()), [str(int(b)) for b in sorted(dec.batch.unique())])
ax.legend(title=None, ncols=2)
ps.save(fig, "validation_ratio_vs_batch");
"""),
    md("""
## Energy per token

The energy model predicts only **dynamic** energy, using Horowitz 45 nm
constants. Measured board energy also includes static and idle power and
everything the model leaves out (control, caches, interconnect, VRMs), and
the GPU is built in a much newer process. Fit

$$ E_\\text{meas} \\approx k \\, E_\\text{dyn} + P_\\text{static} \\, t_\\text{meas} $$

on decode rows, then compare $P_\\text{static}$ with the idle power the bench
measured.
"""),
    code("""
ok = dec[dec.energy_reliable.astype(bool) & dec.e_meas.notna()]
(k, p_static), *_ = np.linalg.lstsq(np.c_[ok.e_dyn_pred, ok.t_meas], ok.e_meas, rcond=None)
print(f"k = {k:.2f}x Horowitz dynamic energy,  P_static = {p_static:.1f} W  "
      f"(bench idle power: {ok.idle_power_w.iloc[0]:.1f} W)")
df["ept_dyn"] = df.e_dyn_pred / (df.batch * np.where(df.phase == "decode", 1, df.context))
df["ept_cal"] = (k * df.e_dyn_pred + p_static * df.t_meas) / (
    df.batch * np.where(df.phase == "decode", 1, df.context))
d = df[df.phase == "decode"]
pd.DataFrame({"dynamic only": err_stats(d.ept_dyn, d.energy_per_token_j),
              "k*E_dyn + P_static*t": err_stats(d.ept_cal, d.energy_per_token_j)}).T
"""),
    code("""
fig, ax = plt.subplots(figsize=(5.6, 4.4))
cols = ["ept_dyn", "ept_cal", "energy_per_token_j"]
lo, hi = d[cols].min().min() * 1e3 * 0.5, d[cols].max().max() * 1e3 * 2  # mJ
ax.plot([lo, hi], [lo, hi], color=ps.NEUTRAL, ls="--", lw=1, label="y = x")
ax.scatter(d.ept_dyn * 1e3, d.energy_per_token_j * 1e3, color=ps.SERIES[0], s=30,
           label="dynamic energy only", edgecolor=ps.SURFACE, lw=1.2, zorder=3)
ax.scatter(d.ept_cal * 1e3, d.energy_per_token_j * 1e3, color=ps.SERIES[1], s=30,
           label="+ static power, scaled", edgecolor=ps.SURFACE, lw=1.2, zorder=3)
ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="predicted mJ / token", ylabel="measured mJ / token",
       title="Decode energy per token")
ax.legend(loc="upper left")
ps.save(fig, "validation_energy");
"""),
    md("""
## Where the model is wrong (and why)

These are the known failure modes, roughly ordered by how much they matter
for **Hugging Face eager decode of a 1-2B model**. Re-read them against the
real numbers above once measurements replace the sample file.

1. **Fixed per-kernel overhead dominates small-batch decode.** The 1.5B model
   moves about 3 GB per step, which takes about 1.5 ms at 2 TB/s. HF eager
   launches ~300+ kernels per token, each with Python dispatch cost, so
   measured tokens/s at batch 1 can be several times below the roofline.
   This is a software property, not a hardware one. CUDA graphs,
   `torch.compile` or a serving engine (vLLM, TensorRT-LLM) remove most of it.
   The roofline is the right target *for those stacks*.
2. **Peak bandwidth isn't reachable.** Streaming kernels typically reach
   80-90% of HBM peak, and small GEMVs (narrow matrices) reach less. A single
   global $\\alpha$ hides the fact that efficiency varies per op.
3. **HF's `DynamicCache` copies the KV cache on every step** (`torch.cat`),
   adding about 2x KV traffic that the model does not count. This error grows
   with context × batch.
4. **Attention kernels.** SDPA decode with one query token is a poorly
   utilized kernel, and the unfused path materializes scores. Try
   `--unfused-attention` in the CLI to bracket this.
5. **Compute efficiency in prefill.** Prefill is compute-bound, and real
   GEMMs reach 60-80% of tensor-core peak, so the pure model is optimistic.
6. **Caches help the model's pessimistic side.** On GPUs the model treats L2
   as absent. Small KV caches and activations do hit L2, so some memory-bound
   ops beat the DRAM roofline.
7. **Logits in fp32, sampling, and host syncs** (the `argmax` each step) are
   small costs that are not modeled.
8. **Energy.** Horowitz numbers are for 45 nm. A 7 nm A100 spends less per
   operation, yet board power includes static power, clocks, memory PHY and
   VRM losses that the model leaves out entirely. Expect large absolute errors
   (the fitted $k$ and $P_\\text{static}$ quantify them). The *relative*
   ordering (DRAM bytes dominate decode energy) is the useful output.
   NVML power is averaged over ≥100 ms windows, so short runs are unreliable.
   The bench enforces ≥2 s timed regions and flags shorter ones.

**What should hold even on real data:** the *shape*. Throughput scales almost
linearly with batch until the weight matmuls cross the ridge point,
throughput falls with context once KV reads rival weight reads, and decode
energy per token falls with batch because weight reads are shared.
"""),
]

# =============================================================================
# (b) Roofline
# =============================================================================
ROOFLINE = [
    md("""
# (b) Roofline: where each operation sits

Every op from one forward step is placed at (arithmetic intensity, attained
FLOP/s). Ceilings: tensor-core peak (matmuls), vector peak (norms, softmax,
activations), and memory bandwidth. Because these are *predicted* points,
each one lies on its own ceiling by construction. What the plot shows is
**which ceiling** each op hits, and how far it is from the ridge.
"""),
    code(SETUP),
    code("""
def op_points(r):
    rows = []
    for o in r.ops:
        b = o.sram_bytes + o.dram_bytes
        if o.flops == 0 or b == 0 or o.time == 0:
            continue  # embedding (no FLOPs) and fused softmax (no bytes) can't be placed
        rows.append({"op": o.name, "kind": o.kind, "intensity": o.flops / b,
                     "tflops": o.flops / (o.time) / 1e12, "time_share": o.time / r.time_s,
                     "bound": o.bound})
    return pd.DataFrame(rows)

def draw_ceilings(ax, chip, dtype, x, mem_levels):
    peak = chip.peak_flops(dtype) / 1e12
    vec = chip.peak_flops(chip.vector_dtype) / 1e12
    for name, bw in mem_levels:
        y = np.minimum(peak, x * bw / 1e12)
        ax.plot(x, y, color=ps.NEUTRAL, lw=1.4)
        i = np.searchsorted(x, peak * 1e12 / bw / 5)  # on the slope, left of this ridge
        ax.annotate(f"{name} {bw/1e12:.3g} TB/s", (x[i], x[i] * bw / 1e12), color=ps.INK_2,
                    fontsize=8, xytext=(-8, 6), textcoords="offset points", ha="right")
    ax.axhline(vec, color=ps.NEUTRAL, lw=1, ls=":")
    ax.annotate(f"{chip.vector_dtype} vector peak {vec:g} TFLOP/s", (x[0], vec), ha="left",
                xytext=(0, 5), textcoords="offset points", fontsize=8, color=ps.INK_2)
    ax.annotate(f"{dtype} peak {peak:g} TFLOP/s", (x[-1], peak), ha="right",
                xytext=(0, -12), textcoords="offset points", fontsize=8, color=ps.INK_2)
"""),
    md("## Llama 3.2 1B (bf16) on A100: decode at batch 1 and 256, prefill of 2048 tokens"),
    code("""
m, a100 = load_model("llama3.2-1b"), load_chip("a100")
scen = {
    "decode, batch 1, ctx 1024": predict_step(m, a100, 1, 1024, 1),
    "decode, batch 256, ctx 1024": predict_step(m, a100, 256, 1024, 1),
    "prefill, 2048 tokens": predict_step(m, a100, 1, 0, 2048),
}
x = np.logspace(-1, 4.3, 400)
ridge = a100.peak_flops(m.compute_dtype) / a100.dram_bw
LABEL = {"mlp_up": ("weight matmuls", -8, 8, "right"),
         "attn_scores": ("attention (QK, PV)", 8, 6, "left"),
         "residual_add": ("elementwise ops", 8, -4, "left")}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
for i, (ax, (label, r)) in enumerate(zip(axes, scen.items())):
    draw_ceilings(ax, a100, m.compute_dtype, x, [("HBM", a100.dram_bw)])
    ax.axvline(ridge, color=ps.GRID, lw=1)
    p = op_points(r)
    ax.scatter(p.intensity, p.tflops, s=50, color=ps.SERIES[i], edgecolor=ps.SURFACE, lw=1.4,
               zorder=3)
    for op, (text, dx, dy, ha) in LABEL.items():
        row = p[p.op == op]
        if len(row):
            ax.annotate(text, (row.intensity.item(), row.tflops.item()), fontsize=8,
                        color=ps.INK, xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.set(xscale="log", yscale="log", title=f"{label}\\n{r.tokens_per_s:,.0f} tok/s",
           xlabel="FLOP / byte", ylim=(0.05, 900))
axes[0].set_ylabel("attained TFLOP/s")
axes[0].annotate(f"ridge {ridge:.0f}", (ridge, 0.07), fontsize=8, color=ps.INK_2,
                 xytext=(4, 0), textcoords="offset points")
fig.suptitle("Roofline: Llama 3.2 1B on A100 (bf16), predicted op placement", fontsize=12,
             color=ps.INK, fontweight="bold", y=1.06)
ps.save(fig, "roofline_a100");
"""),
    code("""
pd.concat({k: op_points(r).set_index("op")[["intensity", "tflops", "time_share", "bound"]]
           for k, r in scen.items()}, axis=1).round(3)
"""),
    md("""
**Reading it.** At batch 1, every weight matmul sits at intensity ≈ 1 FLOP/B
(2 FLOPs per 2-byte weight), about 150x left of the ridge. Attention sits at
≈ 4 FLOP/B because of GQA: the 4 query heads in each group share one K/V
read. At batch 256 the weight matmuls slide right to ≈ 200-230 FLOP/B and become
compute-bound. Attention does not move, because every sequence has its own
KV cache, so batching never amortizes it. In prefill, the matmuls are far
right of the ridge and the elementwise ops become the memory-bound tail.
"""),
    md("""
## Where the time goes

The same model, five operating points. Ops are grouped by what limits them.
At batch 1 the step is almost all weight streaming. Long context or large
batch hands the step to attention (KV-cache reads). In prefill the weight
matmuls are back on top, now compute-bound.
"""),
    code("""
GROUPS = {
    "weight matmuls": ["qkv_proj", "o_proj", "mlp_up", "mlp_down"],
    "LM head": ["lm_head"],
    "attention (KV reads)": ["attn_scores", "attn_softmax", "attn_output"],
    "elementwise + embedding": ["embedding", "attn_norm", "mlp_norm", "residual_add",
                                "mlp_act", "final_norm"],
}
cases = {
    "decode  B=1    ctx 1k": predict_step(m, a100, 1, 1024, 1),
    "decode  B=1    ctx 32k": predict_step(m, a100, 1, 32768, 1),
    "decode  B=64   ctx 4k": predict_step(m, a100, 64, 4096, 1),
    "decode  B=256  ctx 1k": predict_step(m, a100, 256, 1024, 1),
    "prefill 2k tokens": predict_step(m, a100, 1, 0, 2048),
}
share = pd.DataFrame({
    case: {g: sum(o.time for o in r.ops if o.name in ops) / r.time_s for g, ops in GROUPS.items()}
    for case, r in cases.items()
}).T

fig, ax = plt.subplots(figsize=(9, 3.8))
left = np.zeros(len(share))
ys = np.arange(len(share))[::-1]
for i, g in enumerate(share.columns):
    v = share[g].values
    ax.barh(ys, v, left=left, color=ps.SERIES[i], height=0.62, label=g,
            edgecolor=ps.SURFACE, linewidth=2)
    for y, x0, w in zip(ys, left, v):
        if w >= 0.1:
            ax.text(x0 + w / 2, y, f"{100 * w:.0f}%", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
    left += v
for y, r in zip(ys, cases.values()):
    ax.text(1.01, y, f"{r.tokens_per_s:,.0f} tok/s", va="center", fontsize=8, color=ps.INK_2)
ax.set_yticks(ys, share.index, fontfamily="monospace", fontsize=9)
ax.set(xlim=(0, 1), xlabel="share of step time")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
ax.grid(axis="y", visible=False)
ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
ax.set_title("Where the time goes: Llama 3.2 1B (bf16) on A100", pad=28)
ps.save(fig, "time_breakdown");
share.round(3)
"""),
    md("## SRAM-heavy design point: Llama 3.2 1B at 4-bit, decode batch 1, weights resident vs streamed"),
    code("""
m4, sx = load_model("llama3.2-1b").with_(weight_bits=4), load_chip("sram-accel")
res = {"weights resident in SRAM (greedy)": predict_step(m4, sx, 1, 1024, 1),
       "same chip, everything from DRAM": predict_step(m4, sx, 1, 1024, 1, placement_policy="none")}
fig, ax = plt.subplots(figsize=(8.4, 5.2))
draw_ceilings(ax, sx, m4.compute_dtype, x, [("SRAM", sx.sram_bw), ("DRAM", sx.dram_bw)])
for i, (label, r) in enumerate(res.items()):
    p = op_points(r)
    ax.scatter(p.intensity, p.tflops, s=50, color=ps.SERIES[i], edgecolor=ps.SURFACE, lw=1.4,
               zorder=3, label=f"{label}: {r.tokens_per_s:,.0f} tok/s")
ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP / byte)",
       ylabel="attained TFLOP/s", ylim=(0.01, 700),
       title="Hypothetical SRAM-X: same ops, two memory ceilings")
ax.legend(loc="lower right")
ps.save(fig, "roofline_sram_accel");
"""),
    md("""
The ops have the same intensities in both cases. (At 4-bit a weight matmul does
≈ 4 FLOP/B, which is exactly SRAM-X's SRAM ridge (400 TFLOP/s ÷ 100 TB/s).
**With resident weights, batch-1 decode is balanced:** compute and memory
time are within 0.3% of each other, so a faster SRAM or faster math alone
buys almost nothing. Both have to scale together.) Placement only changes which
ceiling they sit under. Residency moves them up the 100 TB/s SRAM line, about
120x above the 0.8 TB/s DRAM line. That gap is the whole argument for
SRAM-heavy inference silicon, and it only holds while the model fits (see
the design study).
"""),
]

# =============================================================================
# (c) Design study
# =============================================================================
DESIGN = [
    md("""
# (c) Design study

Four questions an inference-chip architect would ask, answered with the
model. All numbers are **model predictions** (pure roofline, no overhead), not
measurements. They are upper bounds on throughput and lower bounds on energy.
"""),
    code(SETUP),
    md("""
## 1. How much SRAM does it take to hold a 1B model at 4-bit?

Weights only, plus the KV cache, which also has to live somewhere. 4.25 bits
models group-wise quantization with an fp16 scale per 128 weights (+0.125
bit/weight). Llama 3.2 1B ties its embedding and LM head, so the 128k x 2048
table is read in full every step and has to be resident too.
"""),
    code("""
from perfmodel.ops import tensor_footprints
rows = []
for name in ["llama3.2-1b", "qwen2.5-1.5b"]:
    for wb in [16, 8, 4, 4.25]:
        m = load_model(name).with_(weight_bits=wb)
        fp = tensor_footprints(m, 1, 1)
        fp.pop("kv_cache")
        rows.append({"model": m.name, "weight_bits": wb,
                     "weights_MiB": sum(fp.values()) / 2**20,
                     "of_which_embed_MiB": fp["embed"] / 2**20,
                     "KV_MiB_b1_ctx2k": m.kv_cache_bytes(1, 2048) / 2**20,
                     "KV_MiB_b16_ctx8k": m.kv_cache_bytes(16, 8192) / 2**20})
sram = pd.DataFrame(rows)
sram.round(1)
"""),
    md("""
**Answer.** Llama 3.2 1B at 4-bit needs **≈ 589 MiB** for weights (≈ 626 MiB
with group scales). The tied 128k-vocab embedding alone is 125 MiB of that.
The KV cache adds 64 MiB per 2k-token sequence at 16-bit, so a 1 GiB SRAM
holds the model plus a handful of 2k sequences. A 16-sequence 8k-token batch
(4 GiB of KV) has to spill to DRAM or use KV quantization. Under 512 MiB the
model doesn't fit, and decode falls back to DRAM speed.
"""),
    md("""
### What actually lands in SRAM

SRAM-X (1 GiB) running Llama 3.2 1B at batch 1 and context 2048, by weight
precision. The greedy policy fills SRAM in order of bytes read per byte
stored. Whatever doesn't fit is read from DRAM every step.
"""),
    code("""
m1, sx = load_model("llama3.2-1b"), load_chip("sram-accel")
rows = []
for wb in [16, 8, 4]:
    r = predict_step(m1.with_(weight_bits=wb), sx, 1, 2048, 1)
    p = r.placement
    resident = sum(p.footprint[t] * f for t, f in p.sram_fraction.items())
    total = sum(p.footprint.values())
    rows.append({"weights": f"{wb}-bit", "resident_MiB": resident / 2**20,
                 "spilled_MiB": (total - resident) / 2**20,
                 "act_buffer_MiB": p.activation_reserve / 2**20, "tok_s": r.tokens_per_s})
place = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 3))
ys = np.arange(len(place))[::-1]
ax.barh(ys, place.resident_MiB, color=ps.SERIES[0], height=0.6, label="resident in SRAM",
        edgecolor=ps.SURFACE, linewidth=2)
ax.barh(ys, place.spilled_MiB, left=place.resident_MiB, color=ps.SERIES[1], height=0.6,
        label="spilled: streamed from DRAM each step", edgecolor=ps.SURFACE, linewidth=2)
cap = sx.sram_bytes / 2**20
xmax = place.eval("resident_MiB + spilled_MiB").max()
ax.axvline(cap, color=ps.INK, lw=1.2, ls="--")
ax.annotate(f"SRAM capacity {cap:.0f} MiB", (cap, ys[-1] - 0.55), xytext=(6, 0),
            textcoords="offset points", fontsize=8, color=ps.INK, va="center")
for y, row in zip(ys, place.itertuples()):
    ax.text(xmax * 1.04, y, f"{row.tok_s:,.0f} tok/s", va="center", fontsize=9, color=ps.INK)
ax.set_yticks(ys, place.weights)
ax.set(xlabel="weights + KV cache (MiB)", xlim=(0, xmax * 1.2), ylim=(ys[-1] - 0.85, ys[0] + 0.5))
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=8)
ax.set_title("SRAM-X placement: Llama 3.2 1B, batch 1, ctx 2048", pad=24)
ps.save(fig, "sram_placement");
place.round(1)
"""),
    code("""
# The SRAM cliff: tokens/s on SRAM-X as SRAM capacity grows (1B, 4-bit, batch 1, ctx 2048)
m4, sx = load_model("llama3.2-1b").with_(weight_bits=4), load_chip("sram-accel")
caps = np.arange(64, 1601, 16)
tps = [predict_step(m4, sx.with_(sram_mib=float(c)), 1, 2048, 1).tokens_per_s for c in caps]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(caps, tps, color=ps.SERIES[0])
need = (sram.query("model == 'Llama-3.2-1B' and weight_bits == 4").weights_MiB.item()
        + m4.kv_cache_bytes(1, 2049) / 2**20)
ax.axvline(need, color=ps.NEUTRAL, ls="--", lw=1)
ax.annotate(f"weights + KV = {need:.0f} MiB", (need, max(tps) * 0.5), xytext=(-8, 0),
            textcoords="offset points", ha="right", fontsize=8, color=ps.INK_2)
ax.set(yscale="log", xlabel="on-chip SRAM (MiB)", ylabel="decode tokens/s (batch 1)",
       title="SRAM-X: throughput cliff when the model stops fitting")
ps.save(fig, "design_sram_cliff");
"""),
    md("""
Two regimes. Below the fit point, step time is set by whatever still
streams from DRAM, since SRAM and DRAM run in parallel and the slower level
wins. Each extra MiB of SRAM removes DRAM bytes, so throughput rises
hyperbolically. When the last byte fits, the DRAM path drops out and the
step jumps to SRAM speed: a ~10x cliff over the final few MiB. Sizing SRAM
"just under" the model forfeits most of the benefit.
"""),
    md("""
## 2. At what batch size does decode flip to compute-bound?

For one weight matmul the intensity is $2BdK / (dK \\cdot w_b) = 2B/w_b$
FLOP/byte, so it crosses the ridge at $B^* = \\text{ridge} \\cdot w_b / 2$. At
bf16 that is $B^* \\approx$ ridge itself. The **whole step** is a different
question, because attention's KV reads scale with $B \\times C$ and are never
amortized.
"""),
    code("""
m = load_model("llama3.2-1b")
def first_batch(chip, ctx, test, max_b=2**15):
    b = 1
    while b <= max_b:
        if test(predict_step(m, chip, b, ctx, 1)):
            return b
        b *= 2
    return None
rows = []
for cn in ["a100", "h100", "jetson-orin", "sram-accel"]:
    c = load_chip(cn)
    ridge = c.peak_flops(m.compute_dtype) / c.dram_bw
    for ctx in [128, 512, 4096]:
        mm = first_batch(c, ctx, lambda r: {o.name: o for o in r.ops}["mlp_up"].bound == "compute")
        st = first_batch(c, ctx, lambda r: r.compute_bound_fraction > 0.5)
        rows.append({"chip": cn, "context": ctx, "ridge_FLOP/B": round(ridge),
                     "B* analytic (bf16)": round(ridge),
                     "mlp_up flips at B (pow2)": mm,
                     "whole step >50% compute-bound at B": st or "never (<= 32768)"})
flip = pd.DataFrame(rows)
flip
"""),
    code("""
a100 = load_chip("a100")
bs = 2 ** np.arange(0, 13)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for i, ctx in enumerate([128, 512, 4096]):
    rs = [predict_step(m, a100, int(b), ctx, 1) for b in bs]
    axes[0].plot(bs, [r.tokens_per_s for r in rs], marker="o", color=ps.SERIES[i],
                 label=f"context {ctx}")
    axes[1].plot(bs, [r.compute_bound_fraction for r in rs], marker="o", color=ps.SERIES[i],
                 label=f"context {ctx}")
ridge = a100.peak_flops("bf16") / a100.dram_bw
for ax in axes:
    ax.axvline(ridge, color=ps.NEUTRAL, ls="--", lw=1)
    ax.set(xscale="log", xlabel="batch size")
    ax.set_xticks(bs[::2], [str(b) for b in bs[::2]])
axes[0].set(yscale="log", ylabel="decode tokens/s", title="Throughput vs batch (A100, 1B bf16)")
axes[1].set(ylabel="share of step time compute-bound", ylim=(-0.03, 1.03),
            title="Does the step flip?")
axes[1].annotate(f"ridge ≈ {ridge:.0f}", (ridge, 0.9), xytext=(4, 0), textcoords="offset points",
                 fontsize=8, color=ps.INK_2)
axes[0].legend()
ps.save(fig, "design_batch_flip");
"""),
    md("""
**Answer.** The weight matmuls flip at **B ≈ 153 on A100** and ≈ 295 on H100
(bf16), matching $B^* =$ ridge. The full decode step of Llama 3.2 1B
**never flips at contexts ≥ 512** on any of these chips (searched to
B = 32768). It flips only at short context (128 tokens: B ≈ 256 on A100), and
even there it saturates at about 75% compute-bound time. Per sequence, KV traffic per step is
$L \\cdot 2 \\cdot C \\cdot d_{kv} \\cdot 2$ bytes (32 KiB per cached token here). It
grows with batch exactly as fast as the matmul FLOPs, so by the time the
weights are amortized, attention has become the memory-bound bottleneck.
Large-batch serving is therefore limited by KV bandwidth (and capacity: 4k
sequences at 512 tokens is 64 GiB of KV). That is why KV quantization, GQA/MLA
and paged or offloaded KV matter more than raw TFLOPs.
"""),
    md("## 3. What does weight quantization buy (tokens/s and energy)?"),
    code("""
rows = []
for cn in ["a100", "sram-accel"]:
    c = load_chip(cn)
    for name in ["llama3.2-1b", "llama3-8b"]:
        for wb in [16, 8, 4]:
            for b in [1, 64]:
                r = predict_step(load_model(name).with_(weight_bits=wb), c, b, 1024, 1)
                rows.append({"chip": cn, "model": name, "weight_bits": wb, "batch": b,
                             "tokens_per_s": r.tokens_per_s,
                             "mJ_per_token": r.energy_per_token_j * 1e3,
                             "dram_GB_per_step": r.dram_bytes / 1e9,
                             "compute_bound_frac": r.compute_bound_fraction})
quant = pd.DataFrame(rows)
base = quant[quant.weight_bits == 16].set_index(["chip", "model", "batch"])
quant["speedup_vs_16bit"] = quant.apply(
    lambda r: r.tokens_per_s / base.loc[(r.chip, r.model, r.batch)].tokens_per_s, axis=1)
quant["energy_vs_16bit"] = quant.apply(
    lambda r: r.mJ_per_token / base.loc[(r.chip, r.model, r.batch)].mJ_per_token, axis=1)
quant.round(3)
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
q = quant[quant.batch == 1]
i = 0
for cn in ["a100", "sram-accel"]:
    for name in ["llama3.2-1b", "llama3-8b"]:
        g = q[(q.chip == cn) & (q.model == name)].sort_values("weight_bits", ascending=False)
        lbl = f"{name} on {cn}"
        xs = [str(w) for w in g.weight_bits]
        axes[0].plot(xs, g.tokens_per_s, marker="o", color=ps.SERIES[i], label=lbl)
        axes[1].plot(xs, g.mJ_per_token, marker="o", color=ps.SERIES[i], label=lbl)
        i += 1
axes[0].set(yscale="log", xlabel="weight bits", ylabel="decode tokens/s (batch 1, ctx 1024)",
            title="Throughput")
axes[1].set(yscale="log", xlabel="weight bits", ylabel="mJ / token", title="Energy per token")
axes[0].legend(fontsize=8)
ps.save(fig, "design_quantization");
"""),
    md("""
**Answer.** On a GPU at batch 1, decode is weight-bandwidth-bound, so
16 → 8 → 4 bit gives **≈ 2x and ≈ 3.8x** tokens/s and cuts energy per token by
about the same factor, because DRAM bytes dominate decode energy. It's less
than 4x because KV, activations and norms stay 16-bit. At batch 64 the gain
shrinks: KV traffic is unchanged and the matmuls drift toward compute-bound.
(Weight-only quantization doesn't reduce FLOPs here, since the model computes
in bf16.)

On the SRAM chip, quantization is a **capacity** decision more than a
bandwidth one. 1B at 16-bit (2.3 GiB) spills more than half its bytes to the
0.8 TB/s DRAM and is *slower than an A100* (569 vs 813 tok/s). 8-bit
(1.15 GiB) spills only a little and reaches ~4,000 tok/s. At 4-bit the model
fits and throughput jumps another **≈ 37x**. 8B never fits, so the SRAM chip
is a worse 8B decoder than an A100 at every precision. Batching can even
*hurt* there: at batch 64 the KV cache (≈ 2 GiB at ctx 1024) overflows SRAM,
and 4-bit 1B drops from ~150k to ~30k tok/s. That is a strong argument for KV
quantization or KV in a separate memory on such designs.
"""),
    md("""
## 4. Speculative decoding: several tokens verified per weight read

A verify step processes $k+1$ tokens per sequence (`n_new = k+1`) with the
same weight read as one decode step. If each draft token is accepted
independently with probability $\\alpha$, the expected number of tokens per
step is $(1-\\alpha^{k+1})/(1-\\alpha)$ (Leviathan et al., ICML 2023).

* **with draft:** draft = Llama 3.2 1B (k decode steps), target = Llama 3 8B, A100, ctx 1024.
* **ideal:** draft cost 0 (an upper bound; close to self-speculative / Medusa-style heads).
"""),
    code("""
t8, d1, a100 = load_model("llama3-8b"), load_model("llama3.2-1b"), load_chip("a100")
def spec(batch, k, alpha, draft=True, ctx=1024):
    base = predict_step(t8, a100, batch, ctx, 1).time_s / 1           # s per token (per seq)
    verify = predict_step(t8, a100, batch, ctx, k + 1).time_s
    dcost = k * predict_step(d1, a100, batch, ctx, 1).time_s if draft else 0.0
    tokens = (1 - alpha ** (k + 1)) / (1 - alpha)
    return base / ((verify + dcost) / tokens)

ks = np.arange(1, 11)
alphas = [0.6, 0.7, 0.8, 0.9]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
for j, draft in enumerate([True, False]):
    for i, a in enumerate(alphas):
        axes[j].plot(ks, [spec(1, int(k), a, draft) for k in ks], marker="o",
                     color=ps.SERIES[i], label=f"acceptance {a}")
    axes[j].axhline(1, color=ps.NEUTRAL, ls="--", lw=1)
    axes[j].set(xlabel="draft tokens per step (k)",
                title=["With 1B draft model", "Ideal: free drafts"][j])
axes[0].set_ylabel("predicted speedup vs plain decode")
axes[1].legend()
ps.save(fig, "design_speculative");

table = pd.DataFrame([{"batch": b, "alpha": a, "k": k,
                       "speedup_with_draft": spec(b, k, a), "speedup_ideal": spec(b, k, a, False)}
                      for b in [1, 16, 128] for a in [0.7, 0.9] for k in [2, 4, 8]])
table.pivot_table(index=["batch", "alpha"], columns="k",
                  values=["speedup_with_draft", "speedup_ideal"]).round(2)
"""),
    md("""
**Answer.** At batch 1 the verify step is nearly free. Checking 5 tokens costs
about the same as generating 1, because the step is weight-bandwidth-bound.
So the gain is roughly the expected accepted length, minus draft cost:
**≈ 1.6x at α = 0.7 and ≈ 2.5x at α = 0.9 with a real 1B draft** (the draft's
own weight reads cap it, so the optimum is k ≈ 3-7), and **≈ 3x / 6x with free
drafts** at k = 8. The benefit drops
quickly with batch. At batch 128 the target's matmuls are already near the
ridge, so verifying k+1 tokens costs nearly k+1 times as much, and the speedup
collapses toward (or below) 1. Speculative decoding is a small-batch,
memory-bound optimization. That is also the regime where an SRAM-heavy chip
benefits least, since its weight reads are already cheap.
"""),
]


def write(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = HERE / name
    nbf.write(nb, path)
    return path


def main() -> None:
    paths = [
        write("a_validation.ipynb", VALIDATION),
        write("b_roofline.ipynb", ROOFLINE),
        write("c_design_study.ipynb", DESIGN),
    ]
    if "--run" in sys.argv:
        for p in paths:
            subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--inplace", str(p)],
                check=True,
                cwd=HERE,
            )
    for p in paths:
        print(p.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
