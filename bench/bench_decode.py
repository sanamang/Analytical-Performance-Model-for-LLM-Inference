#!/usr/bin/env python
"""Measure Hugging Face decode (and prefill) throughput and power on an NVIDIA GPU.

Run on the GPU machine:

    uv sync --extra bench
    uv run python bench/bench_decode.py --model-id Qwen/Qwen2.5-1.5B \
        --batches 1,2,4,8,16,32,64,128 --contexts 128,512,1024,2048,4096 \
        --out bench/results/a100_qwen2.5-1.5b.csv

Method
------
* Decode is measured at a FIXED context length: after each step the KV cache
  is cropped back to ``context`` tokens (``DynamicCache.crop``). Every timed
  step then attends over exactly ``context + 1`` positions, which is what
  ``perfmodel`` predicts for ``n_past = context``. Without ``crop`` (very old
  transformers) the context grows and the CSV records the mean context.
* Warmup: one throwaway configuration at the start (CUDA context, cuBLAS
  handles, allocator), then ``--warmup-steps`` untimed decode steps per
  configuration.
* Timing: ``torch.cuda.Event`` pairs around each step. GPU time between
  events includes any CPU launch gaps, which is what a user experiences.
* Steps per configuration are chosen so the timed region lasts about
  ``--min-seconds``, because NVML power readings are averaged over windows of
  100 ms or more.
* Energy: NVML's total-energy counter (Volta and later) when available,
  otherwise the integral of power samples. Idle power is measured once at the
  start so static power can be separated later.
* Prefill uses ``logits_to_keep=1`` where supported, so only the last
  position's logits are computed (as the model assumes).
* Each row is appended to the CSV as soon as it is measured, so a crash or
  OOM keeps earlier results.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
import math
import statistics
import threading
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

try:
    import pynvml  # from the nvidia-ml-py package
except ImportError:  # pragma: no cover
    pynvml = None

# HF model id -> perfmodel config name, so the notebook can match rows to predictions.
MODEL_MAP = {
    "Qwen/Qwen2.5-1.5B": "qwen2.5-1.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b",
    "meta-llama/Llama-3.2-1B": "llama3.2-1b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2-1b",
    "meta-llama/Meta-Llama-3-8B": "llama3-8b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",
}


def guess_chip(gpu_name: str) -> str:
    n = gpu_name.upper()
    if "H100" in n:
        return "h100"
    if "A100" in n:
        return "a100"
    if "ORIN" in n:
        return "jetson-orin"
    return ""


# --------------------------------------------------------------------------- power


class PowerMonitor:
    """Samples NVML power in a background thread; also reads the energy counter."""

    def __init__(self, nvml_index: int = 0, interval_s: float = 0.01):
        self.available = pynvml is not None
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self.has_counter = False
        if not self.available:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] NVML unavailable: {e}")
            self.available = False
            return
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_counter = True
        except Exception:  # noqa: BLE001
            self.has_counter = False

    def driver(self) -> str:
        if not self.available:
            return ""
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else v

    def power_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.power_w()))
            time.sleep(self.interval_s)

    def __enter__(self) -> PowerMonitor:
        self.samples = []
        if self.available:
            self._e0 = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                if self.has_counter
                else None
            )
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_s = time.perf_counter() - self._t0
        if not self.available:
            return
        self._stop.set()
        self._thread.join()
        self._e1 = (
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.has_counter else None
        )

    def result(self) -> tuple[float, float, str]:
        """(avg_power_w, energy_j, method) for the last ``with`` block."""
        if not self.available:
            return math.nan, math.nan, "none"
        avg = statistics.fmean(p for _, p in self.samples) if self.samples else math.nan
        if self.has_counter:
            return avg, (self._e1 - self._e0) / 1000.0, "nvml_energy_counter"  # mJ -> J
        return avg, avg * self.wall_s, "power_samples"

    def idle_power(self, seconds: float = 3.0) -> float:
        if not self.available:
            return math.nan
        torch.cuda.synchronize()
        time.sleep(0.5)
        with self:
            time.sleep(seconds)
        return self.result()[0]


# --------------------------------------------------------------------------- model calls


def _forward_kwargs(model) -> dict:
    """Ask for last-position logits only in prefill, if this transformers version supports it."""
    params = inspect.signature(model.forward).parameters
    if "logits_to_keep" in params:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in params:
        return {"num_logits_to_keep": 1}
    return {}


@torch.inference_mode()
def run_prefill(model, ids, kw):
    return model(input_ids=ids, use_cache=True, **kw)


@torch.inference_mode()
def decode_step(model, tok, cache, crop_to: int | None):
    out = model(input_ids=tok, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    if crop_to is not None:
        cache.crop(crop_to)
    return out.logits[:, -1:].argmax(-1), cache


def timed_steps(fn, n: int) -> list[float]:
    """Run ``fn`` n times, returning per-call GPU milliseconds (CUDA events)."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


def steps_for(seconds: float, step_ms: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, math.ceil(seconds * 1000 / max(step_ms, 1e-3)))))


# --------------------------------------------------------------------------- measurement


def measure_decode(model, batch, context, args, mon, vocab):
    ids = torch.randint(0, vocab, (batch, context), device="cuda")
    out = run_prefill(model, ids, _forward_kwargs(model))
    cache = out.past_key_values
    tok = out.logits[:, -1:].argmax(-1)
    del out
    can_crop = hasattr(cache, "crop")
    crop_to = context if can_crop else None
    state = {"tok": tok, "cache": cache}

    def step():
        state["tok"], state["cache"] = decode_step(model, state["tok"], state["cache"], crop_to)

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    probe = statistics.median(timed_steps(step, 5))
    n = steps_for(args.min_seconds, probe, args.min_steps, args.max_steps)

    ctx_start = state["cache"].get_seq_length() if hasattr(state["cache"], "get_seq_length") \
        else context
    with mon:
        ms = timed_steps(step, n)
    avg_w, energy_j, method = mon.result()

    total_s = sum(ms) / 1000
    ctx_mean = context if can_crop else ctx_start + (n - 1) / 2
    tokens = batch * n
    return {
        "phase": "decode",
        "batch": batch,
        "context": ctx_mean,
        "n_steps": n,
        "time_s_total": total_s,
        "step_ms_mean": statistics.fmean(ms),
        "step_ms_median": statistics.median(ms),
        "step_ms_p10": statistics.quantiles(ms, n=10)[0] if n >= 10 else min(ms),
        "step_ms_p90": statistics.quantiles(ms, n=10)[-1] if n >= 10 else max(ms),
        "tokens_per_s": tokens / total_s,
        "avg_power_w": avg_w,
        "energy_j": energy_j,
        "energy_per_token_j": energy_j / tokens,
        "energy_method": method,
        "energy_reliable": mon.wall_s >= 1.0,
        "fixed_context": can_crop,
    }


def measure_prefill(model, batch, context, args, mon, vocab):
    ids = torch.randint(0, vocab, (batch, context), device="cuda")
    kw = _forward_kwargs(model)

    def step():
        run_prefill(model, ids, kw)

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    probe = statistics.median(timed_steps(step, 3))
    n = steps_for(args.min_seconds, probe, 3, args.max_steps)
    with mon:
        ms = timed_steps(step, n)
    avg_w, energy_j, method = mon.result()
    total_s = sum(ms) / 1000
    tokens = batch * context * n
    return {
        "phase": "prefill",
        "batch": batch,
        "context": context,
        "n_steps": n,
        "time_s_total": total_s,
        "step_ms_mean": statistics.fmean(ms),
        "step_ms_median": statistics.median(ms),
        "step_ms_p10": min(ms),
        "step_ms_p90": max(ms),
        "tokens_per_s": tokens / total_s,
        "avg_power_w": avg_w,
        "energy_j": energy_j,
        "energy_per_token_j": energy_j / tokens,
        "energy_method": method,
        "energy_reliable": mon.wall_s >= 1.0,
        "fixed_context": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--perfmodel-model", default=None, help="perfmodel config name")
    ap.add_argument("--perfmodel-chip", default=None, help="perfmodel chip config name")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn-impl", default="sdpa", help="sdpa | eager | flash_attention_2")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--contexts", default="128,512,1024,2048,4096")
    ap.add_argument("--prefill-batches", default="1,4")
    ap.add_argument("--phases", default="decode,prefill")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--nvml-index", type=int, default=0,
                    help="NVML device index (may differ from CUDA index under "
                         "CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--out", type=Path, default=Path("bench/results/measured.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, attn_implementation=args.attn_impl
    ).to("cuda").eval()
    vocab = model.config.vocab_size

    mon = PowerMonitor(args.nvml_index)
    gpu = torch.cuda.get_device_name()
    props = torch.cuda.get_device_properties(0)
    meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "gpu_name": gpu,
        "gpu_mem_gib": round(props.total_memory / 2**30, 1),
        "driver": mon.driver(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "model_id": args.model_id,
        "perfmodel_model": args.perfmodel_model or MODEL_MAP.get(args.model_id, ""),
        "perfmodel_chip": args.perfmodel_chip or guess_chip(gpu),
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "synthetic": 0,
    }

    print(f"GPU: {gpu} ({meta['gpu_mem_gib']} GiB), model {args.model_id}")
    print("global warmup...")
    measure_decode(model, 1, 64, argparse.Namespace(**{**vars(args), "min_seconds": 0.5}),
                   mon, vocab)
    idle_w = mon.idle_power()
    meta["idle_power_w"] = idle_w
    print(f"idle power: {idle_w:.1f} W, energy method: "
          f"{'counter' if mon.has_counter else 'samples' if mon.available else 'none'}")

    plan = []
    phases = args.phases.split(",")
    contexts = [int(c) for c in args.contexts.split(",")]
    if "decode" in phases:
        plan += [("decode", b, c) for b in map(int, args.batches.split(",")) for c in contexts]
    if "prefill" in phases:
        plan += [("prefill", b, c) for b in map(int, args.prefill_batches.split(","))
                 for c in contexts]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        writer = None
        for phase, b, c in plan:
            fn = measure_decode if phase == "decode" else measure_prefill
            try:
                row = fn(model, b, c, args, mon, vocab)
            except torch.cuda.OutOfMemoryError:
                print(f"{phase:8s} B={b:<4d} C={c:<6d} OOM, skipped")
                torch.cuda.empty_cache()
                continue
            row = {**meta, **row, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"{phase:8s} B={b:<4d} C={c:<6d} {row['tokens_per_s']:10.1f} tok/s  "
                  f"{row['step_ms_median']:8.2f} ms/step  {row['avg_power_w']:6.1f} W  "
                  f"{1e3 * row['energy_per_token_j']:8.3f} mJ/tok")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
