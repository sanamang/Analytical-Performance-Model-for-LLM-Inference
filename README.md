# perfmodel: an analytical performance and energy model for LLM inference

Given a transformer and a chip description, `perfmodel` predicts prefill and
decode latency, tokens/s, energy per token, and whether each operation is
compute- or memory-bound. It uses a two-level roofline that also models
on-chip SRAM: weights and KV cache that fit in SRAM are served at SRAM
bandwidth and energy, and the rest streams from DRAM.

**[Interactive explorer](https://sanamang.github.io/Analytical-Performance-Model-for-LLM-Inference/)**

> **Validation status:** no real GPU measurements are in this repo yet. The
> validation notebook runs on a synthetic sample file
> (`bench/results/sample_synthetic.csv`, every row flagged `synthetic=1`).
> Run `bench/bench_decode.py` on a GPU to replace it.

## How it works

Running an LLM costs time, hardware and energy, and it is hard to know in
advance how much. What happens if you quantize a model or batch more
requests? Would it run fast enough? Instead of answering through trial and
error, this project answers these questions on paper. It counts the math and
data movement a model needs for each token, then combines that with a chip's
peak compute and memory speed to estimate the speed and energy it can reach.

For each operation in a transformer layer, the model computes FLOPs and bytes
moved from the layer shapes. This is checked by 65 tests:

- Parameter counts match the published model cards
- Per-op FLOPs and bytes match hand-computed values on a toy model, for decode and prefill
- Decode FLOPs per token is about 2 × parameters, within 1% at short context
- GQA cuts KV-cache bytes, not FLOPs

The model assumes an operation takes as long as its slowest resource, with
math and data movement overlapping perfectly:

$$
t = \max\left(\frac{F}{P_{peak}},\ \frac{D_{SRAM}}{BW_{SRAM}},\ \frac{D_{DRAM}}{BW_{DRAM}}\right)
$$

This is a strict lower bound on time, so every result is a best case.

1. **Count** ([`ops.py`](src/perfmodel/ops.py)): FLOPs and bytes for every op
   in a forward step (QKV, attention with GQA and causal masking, MLP, norms,
   LM head). It covers prefill, decode and speculative verification.
2. **Place** ([`memory.py`](src/perfmodel/memory.py)): decide which weights
   and KV cache stay in SRAM (`none | greedy | weights_first | kv_first`).
   GPUs default to `none`.
3. **Roofline** ([`roofline.py`](src/perfmodel/roofline.py)): per op,
   `t = max(FLOPs/peak, SRAM bytes/SRAM BW, DRAM bytes/DRAM BW)`.
4. **Energy** ([`energy.py`](src/perfmodel/energy.py)):
   `E = FLOPs·e_flop + bytes·e_byte`, with constants from Horowitz (ISSCC
   2014) and O'Connor (MICRO 2017). These constants are approximate, so trust
   ratios more than absolute joules.

**Configs:**
- Models: Llama 3 8B, Llama 3.2 1B, Qwen2.5 1.5B.
- Chips: A100, H100, Jetson AGX Orin, and SRAM-X, a hypothetical SRAM-heavy
  accelerator.

Every number cites its source, and estimates are flagged.

## Usage

```bash
uv sync
uv run pytest
uv run perfmodel predict --model llama3.2-1b --chip a100 --batch 1 --context 1024 --phase decode
uv run perfmodel sweep --model llama3.2-1b --chip sram-accel --batches 1,8,64 --out sweep.csv
uv run perfmodel export-explorer   # regenerate docs/data for the explorer
```

Useful flags: `--weight-bits 4`, `--kv-bits 8`, `--placement kv_first`,
`--unfused-attention`, `--overhead-us 10`.

## Key findings (model predictions, upper bounds)

- **Batch-1 decode is bandwidth-bound.** Llama 3.2 1B on an A100 reads about
  2.5 GB per token, which predicts about 810 tok/s.
- **Weight matmuls turn compute-bound at batch ≈ 153** on A100 (≈ 295 on
  H100). The full decode step never does, because KV-cache reads grow with
  batch × context.
- **SRAM capacity is a cliff.** On SRAM-X the 4-bit 1B model fits and decodes
  at about 150k tok/s. The 16-bit model spills to DRAM and is slower than an
  A100.
- **4-bit weights give ≈ 3.8× tokens/s and ≈ 0.28× energy/token** at batch 1
  on a GPU.
- **Speculative decoding gives ≈ 1.6–2.5×** at batch 1 but little at large
  batch.

## Limitations

- The model gives an upper bound unless you set the efficiency or overhead
  knobs.
- Ops don't overlap, and GPU caches aren't modeled.
- It covers a single device only.
- Quantization is weight-only.
- Energy constants are 45 nm.

## Validating on a GPU

```bash
uv sync --extra bench
uv run python bench/bench_decode.py --model-id Qwen/Qwen2.5-1.5B --out bench/results/a100.csv
PERFMODEL_BENCH_CSV=bench/results/a100.csv uv run python notebooks/build_notebooks.py --run
```

## References

Williams et al., *Roofline* (CACM 2009) · Yang et al., *Hierarchical Roofline*
(CCPE 2020) · Horowitz, *Computing's Energy Problem* (ISSCC 2014) · O'Connor
et al., *Fine-Grained DRAM* (MICRO 2017) · Leviathan et al., *Speculative
Decoding* (ICML 2023) · Dao et al., *FlashAttention* (NeurIPS 2022) · NVIDIA
datasheets.
