"""Command line: ``perfmodel predict``, ``perfmodel sweep``, ``perfmodel list``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from perfmodel.memory import POLICIES
from perfmodel.predict import StepResult, predict, sweep
from perfmodel.specs import available_configs, load_chip, load_model

app = typer.Typer(add_completion=False, help="Analytical LLM inference performance model.")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _fmt_time(s: float) -> str:
    return f"{s * 1e3:.3f} ms" if s >= 1e-3 else f"{s * 1e6:.1f} us"


def _print_step(r: StepResult, show_ops: bool) -> None:
    title = f"{r.phase.upper()}  batch={r.batch}  " + (
        f"prompt={r.n_new}" if r.phase == "prefill" else f"context={r.n_past}"
    )
    typer.echo(f"\n=== {title} ===")
    if show_ops:
        df = r.to_frame()
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            typer.echo(
                df.to_string(
                    index=False,
                    formatters={
                        "GFLOP": "{:.3f}".format,
                        "SRAM_MB": "{:.2f}".format,
                        "DRAM_MB": "{:.2f}".format,
                        "intensity": "{:.1f}".format,
                        "time_us": "{:.2f}".format,
                        "time_%": "{:.1f}".format,
                        "energy_mJ": "{:.3f}".format,
                    },
                )
            )
    s = r.summary()
    label = "latency (TTFT)" if r.phase == "prefill" else "per-token latency"
    typer.echo(
        f"{label}: {_fmt_time(r.time_s)}   throughput: {r.tokens_per_s:,.1f} tok/s   "
        f"energy: {r.energy_per_token_j * 1e3:.3f} mJ/token\n"
        f"total: {s['gflop']:.2f} GFLOP, {s['gbytes']:.3f} GB moved "
        f"({s['dram_gbytes']:.3f} GB DRAM), intensity {s['intensity']:.1f} FLOP/B, "
        f"{100 * s['compute_bound_fraction']:.0f}% of time compute-bound, "
        f"SRAM used {s['sram_used_mib']:.1f} MiB"
    )


@app.command("predict")
def predict_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model config name or YAML path."),
    chip: str = typer.Option(..., "--chip", "-c", help="Chip config name or YAML path."),
    batch: int = typer.Option(1, "--batch", "-b"),
    context: int = typer.Option(1024, "--context", help="Prompt length / cached tokens."),
    phase: str = typer.Option("both", help="prefill, decode or both."),
    weight_bits: float | None = typer.Option(None, help="Override model weight bits."),
    kv_bits: float | None = typer.Option(None, help="Override KV-cache bits."),
    placement: str | None = typer.Option(
        None,
        help=f"SRAM placement policy {POLICIES}. Default: greedy if chip has "
        "persistent SRAM, else none.",
    ),
    unfused_attention: bool = typer.Option(False, help="Model a non-fused attention kernel."),
    overhead_us: float = typer.Option(0.0, help="Fixed overhead per kernel, microseconds."),
    ops: bool = typer.Option(True, help="Print the per-op table."),
) -> None:
    """Per-op table and totals for prefill and/or one decode step."""
    m = load_model(model)
    if weight_bits is not None:
        m = m.with_(weight_bits=weight_bits)
    if kv_bits is not None:
        m = m.with_(kv_bits=kv_bits)
    c = load_chip(chip)
    typer.echo(
        f"{m.name} ({m.param_count() / 1e9:.2f}B params, W{m.weight_bits:g}/KV{m.kv_bits:g}/"
        f"A{m.act_bits:g}, {m.compute_dtype}) on {c.name}"
    )
    res = predict(
        m,
        c,
        batch,
        context,
        placement_policy=placement,
        fused_attention=not unfused_attention,
        overhead_per_op_s=overhead_us * 1e-6,
    )
    for ph in ("prefill", "decode"):
        if phase in (ph, "both"):
            _print_step(res[ph], ops)


@app.command("sweep")
def sweep_cmd(
    model: str = typer.Option(..., "--model", "-m"),
    chip: str = typer.Option(..., "--chip", "-c"),
    batches: str = typer.Option("1,2,4,8,16,32,64,128,256"),
    contexts: str = typer.Option("128,512,2048,8192"),
    weight_bits: str = typer.Option("16,8,4"),
    placement: str | None = typer.Option(None),
    out: Path = typer.Option(Path("sweep.csv"), "--out", "-o"),
) -> None:
    """Write a CSV over batch sizes x context lengths x weight precisions."""
    df = sweep(
        load_model(model),
        load_chip(chip),
        _ints(batches),
        _ints(contexts),
        _floats(weight_bits),
        placement_policy=placement,
    )
    df.to_csv(out, index=False)
    typer.echo(f"wrote {len(df)} rows to {out}")


@app.command("list")
def list_cmd() -> None:
    """Show available model and chip configs."""
    typer.echo("models: " + ", ".join(available_configs("models")))
    typer.echo("chips:  " + ", ".join(available_configs("chips")))


if __name__ == "__main__":
    app()
