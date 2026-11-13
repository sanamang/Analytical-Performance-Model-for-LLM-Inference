"""Command line: ``perfmodel predict``, ``perfmodel sweep``, ``perfmodel list``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

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


BOUND_STYLE = {
    "compute": "bold blue",
    "memory": "dark_orange",
    "balanced": "magenta",
    "none": "dim",
}


def _console() -> Console:
    c = Console()
    # Piped / captured output: keep a wide, uncolored table instead of squeezing to 80 columns.
    return c if c.is_terminal else Console(width=140, no_color=True)


def _bar(frac: float, width: int = 12) -> str:
    """Share of step time as a block bar, e.g. 0.43 -> '█████▏'."""
    eighths = round(max(0.0, min(1.0, frac)) * width * 8)
    full, rem = divmod(eighths, 8)
    return "█" * full + ("" if rem == 0 else " ▏▎▍▌▋▊▉"[rem])


def _print_step(r: StepResult, show_ops: bool) -> None:
    con = _console()
    title = f"{r.phase.upper()}  batch={r.batch}  " + (
        f"prompt={r.n_new}" if r.phase == "prefill" else f"context={r.n_past}"
    )
    if show_ops:
        t = Table(
            title=title,
            title_justify="left",
            title_style="bold",
            header_style="bold",
            box=box.SIMPLE_HEAD,
            pad_edge=False,
        )
        for col, kw in [
            ("op", {}),
            ("kind", {"style": "dim"}),
            ("count", {"justify": "right"}),
            ("GFLOP", {"justify": "right"}),
            ("SRAM_MB", {"justify": "right"}),
            ("DRAM_MB", {"justify": "right"}),
            ("intensity", {"justify": "right"}),
            ("time_us", {"justify": "right"}),
            ("time_%", {"justify": "right"}),
            ("share of step", {}),
            ("bound", {}),
            ("energy_mJ", {"justify": "right"}),
        ]:
            t.add_column(col, **kw)
        for row in r.to_frame().to_dict("records"):
            style = BOUND_STYLE.get(row["bound"], "")
            share = row["time_%"]
            t.add_row(
                row["op"],
                row["kind"],
                str(row["count"]),
                f"{row['GFLOP']:.3f}",
                f"{row['SRAM_MB']:.2f}",
                f"{row['DRAM_MB']:.2f}",
                f"{row['intensity']:.1f}",
                f"{row['time_us']:.2f}",
                f"{share:.1f}",
                Text(_bar(share / 100), style=style),
                Text(row["bound"], style=style),
                f"{row['energy_mJ']:.3f}",
            )
        con.print(t)
    else:
        con.print(f"\n[bold]{title}[/bold]")
    s = r.summary()
    label = "latency (TTFT)" if r.phase == "prefill" else "per-token latency"
    cb = s["compute_bound_fraction"]
    verdict = "[bold blue]compute-bound[/]" if cb > 0.5 else "[dark_orange]memory-bound[/]"
    con.print(
        f"{label}: [bold]{_fmt_time(r.time_s)}[/]   throughput: [bold]{r.tokens_per_s:,.1f} "
        f"tok/s[/]   energy: [bold]{r.energy_per_token_j * 1e3:.3f} mJ/token[/]\n"
        f"total: {s['gflop']:.2f} GFLOP, {s['gbytes']:.3f} GB moved "
        f"({s['dram_gbytes']:.3f} GB DRAM), intensity {s['intensity']:.1f} FLOP/B, "
        f"{100 * cb:.0f}% of time compute-bound -> step is {verdict}, "
        f"SRAM used {s['sram_used_mib']:.1f} MiB",
        highlight=False,
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
