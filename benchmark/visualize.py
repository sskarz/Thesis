"""Generate thesis-quality charts from benchmark results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from . import config, db

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def _setup_style():
    sns.set_theme(style="whitegrid", font="serif", palette="colorblind")
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _save(fig, name: str):
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: BFCL accuracy by category (horizontal bar)
# ---------------------------------------------------------------------------

MULTI_TURN_CATS = {"multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"}


def _tokens_per_second(token_count, ms):
    if ms <= 0:
        return 0
    return token_count / (ms / 1000.0)


def _normalize_multi_turn_timings(timings: pd.DataFrame) -> pd.DataFrame:
    if timings.empty:
        return timings

    normalized = pd.DataFrame({
        "id": timings["id"],
        "category": timings["category"],
        "prompt_n": timings["total_prompt_n"],
        # Multi-turn timings track cumulative prompt-processing time, not TTFT.
        "prompt_ms": 0,
        "predicted_n": timings["total_predicted_n"],
        "predicted_ms": timings["total_predicted_ms"],
    })
    normalized["prompt_per_second"] = [
        _tokens_per_second(token_count, ms)
        for token_count, ms in zip(
            timings["total_prompt_n"], timings["total_prompt_ms"]
        )
    ]
    normalized["predicted_per_second"] = [
        _tokens_per_second(token_count, ms)
        for token_count, ms in zip(
            timings["total_predicted_n"], timings["total_predicted_ms"]
        )
    ]
    return normalized


def _load_timings(
    include_single: bool = True,
    include_multi: bool = True,
) -> pd.DataFrame:
    run_id = config.RUN_ID
    frames: list[pd.DataFrame] = []

    if run_id:
        # Read from DB
        if include_single:
            rows = db.get_single_turn_timings(run_id)
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"entry_id": "id"})
                frames.append(df[["id", "category", "prompt_n", "prompt_ms",
                                  "prompt_per_second", "predicted_n",
                                  "predicted_ms", "predicted_per_second"]])
        if include_multi:
            rows = db.get_multi_turn_timings(run_id)
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"entry_id": "id"})
                frames.append(_normalize_multi_turn_timings(df))
    else:
        # Fallback to flat files
        if include_single:
            single_path = config.RESPONSES_DIR / "all_timings.csv"
            if single_path.exists():
                frames.append(pd.read_csv(single_path))
        if include_multi:
            multi_path = config.RESPONSES_DIR / "multi_turn_timings.csv"
            if multi_path.exists():
                frames.append(
                    _normalize_multi_turn_timings(pd.read_csv(multi_path))
                )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def chart_accuracy(scores: pd.DataFrame) -> plt.Figure:
    n_cats = len(scores)
    height = max(4, 0.55 * n_cats)
    fig, ax = plt.subplots(figsize=(8, height))
    scores_sorted = scores.sort_values("accuracy")
    # Color multi-turn bars differently from single-turn
    colors = [
        sns.color_palette("Set2")[1] if cat in MULTI_TURN_CATS
        else sns.color_palette("colorblind")[i % 10]
        for i, cat in enumerate(scores_sorted["category"])
    ]
    bars = ax.barh(
        scores_sorted["category"],
        scores_sorted["accuracy"] * 100,
        color=colors,
        edgecolor="white",
    )
    for bar, val in zip(bars, scores_sorted["accuracy"]):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"BFCL Accuracy by Category — {config.MODEL_DISPLAY}")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Throughput distribution (grouped box plot)
# ---------------------------------------------------------------------------

def chart_throughput(timings: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    # Reshape for seaborn
    prompt = timings[["prompt_per_second"]].rename(columns={"prompt_per_second": "tokens/sec"})
    prompt["stage"] = "Prompt processing"
    gen = timings[["predicted_per_second"]].rename(columns={"predicted_per_second": "tokens/sec"})
    gen["stage"] = "Generation"
    combined = pd.concat([prompt, gen], ignore_index=True)
    combined = combined[combined["tokens/sec"] > 0]

    sns.boxplot(data=combined, x="stage", y="tokens/sec", hue="stage", ax=ax, palette="colorblind", width=0.5, legend=False)
    ax.set_title("Throughput Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Tokens / sec")
    return fig


# ---------------------------------------------------------------------------
# Chart 3: TTFT distribution (histogram + KDE)
# ---------------------------------------------------------------------------

def chart_ttft(timings: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ttft = timings["prompt_ms"][timings["prompt_ms"] > 0]
    if ttft.empty:
        ax.text(0.5, 0.5, "No TTFT data", transform=ax.transAxes, ha="center")
        return fig

    sns.histplot(ttft, kde=True, ax=ax, color=sns.color_palette("colorblind")[0], bins=30)
    median = ttft.median()
    ax.axvline(median, color="red", linestyle="--", linewidth=1.2, label=f"Median: {median:.0f} ms")

    # Annotation box
    stats_text = (
        f"Mean: {ttft.mean():.0f} ms\n"
        f"P95:  {ttft.quantile(0.95):.0f} ms\n"
        f"P99:  {ttft.quantile(0.99):.0f} ms"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment="top", horizontalalignment="right",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )
    ax.set_xlabel("Time to First Token (ms)")
    ax.set_ylabel("Count")
    ax.set_title("TTFT Distribution")
    ax.legend(fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Generation speed vs prompt length (scatter)
# ---------------------------------------------------------------------------

def chart_speed_vs_prompt(timings: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = timings[(timings["prompt_n"] > 0) & (timings["predicted_per_second"] > 0)].copy()
    if subset.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    categories = subset["category"].unique()
    palette = dict(zip(categories, sns.color_palette("colorblind", len(categories))))
    for cat in categories:
        d = subset[subset["category"] == cat]
        ax.scatter(d["prompt_n"], d["predicted_per_second"], label=cat,
                   alpha=0.6, s=18, color=palette[cat])

    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Generation speed (tokens/sec)")
    ax.set_title("Generation Speed vs. Prompt Length")
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    return fig


# ---------------------------------------------------------------------------
# Chart 5: Memory usage (annotated bar)
# ---------------------------------------------------------------------------

def chart_memory(metrics: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    mem = metrics.get("memory", {})
    baseline = mem.get("baseline_rss_mb", 0)
    peak = mem.get("peak_rss_mb", 0)

    if baseline == 0 and peak == 0:
        ax.text(0.5, 0.5, "No memory data", transform=ax.transAxes, ha="center")
        return fig

    labels = ["Baseline RSS", "Peak RSS"]
    values = [baseline, peak]
    colors = sns.color_palette("colorblind", 2)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f} MB", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("llama-server Memory Usage")
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)
    return fig


# ---------------------------------------------------------------------------
# Chart 6: Summary dashboard (2x3 grid)
# ---------------------------------------------------------------------------

def chart_dashboard(scores, timings, metrics) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # 1 — Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    scores_sorted = scores.sort_values("accuracy")
    ax1.barh(scores_sorted["category"], scores_sorted["accuracy"] * 100,
             color=sns.color_palette("colorblind", len(scores_sorted)))
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("BFCL Accuracy", fontsize=11)
    ax1.set_xlim(0, 105)

    # 2 — Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    prompt_speeds = timings["prompt_per_second"][timings["prompt_per_second"] > 0]
    gen_speeds = timings["predicted_per_second"][timings["predicted_per_second"] > 0]
    bp = ax2.boxplot(
        [prompt_speeds, gen_speeds],
        labels=["Prompt", "Generation"],
        patch_artist=True,
    )
    colors = sns.color_palette("colorblind", 2)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("Tokens / sec")
    ax2.set_title("Throughput", fontsize=11)

    # 3 — TTFT
    ax3 = fig.add_subplot(gs[0, 2])
    ttft = timings["prompt_ms"][timings["prompt_ms"] > 0]
    if not ttft.empty:
        ax3.hist(ttft, bins=30, color=sns.color_palette("colorblind")[0], alpha=0.8)
        ax3.axvline(ttft.median(), color="red", linestyle="--", linewidth=1)
    ax3.set_xlabel("TTFT (ms)")
    ax3.set_title("TTFT Distribution", fontsize=11)

    # 4 — Scatter
    ax4 = fig.add_subplot(gs[1, 0])
    subset = timings[(timings["prompt_n"] > 0) & (timings["predicted_per_second"] > 0)]
    if not subset.empty:
        cats = subset["category"].unique()
        palette = dict(zip(cats, sns.color_palette("colorblind", len(cats))))
        for cat in cats:
            d = subset[subset["category"] == cat]
            ax4.scatter(d["prompt_n"], d["predicted_per_second"], label=cat,
                        alpha=0.5, s=12, color=palette[cat])
        ax4.legend(fontsize=6, loc="best")
    ax4.set_xlabel("Prompt tokens")
    ax4.set_ylabel("Gen tokens/sec")
    ax4.set_title("Speed vs. Prompt Length", fontsize=11)

    # 5 — Memory
    ax5 = fig.add_subplot(gs[1, 1])
    mem = metrics.get("memory", {})
    baseline = mem.get("baseline_rss_mb", 0)
    peak = mem.get("peak_rss_mb", 0)
    if baseline > 0 or peak > 0:
        bars = ax5.bar(["Baseline", "Peak"], [baseline, peak],
                       color=sns.color_palette("colorblind", 2), width=0.5)
        for bar, val in zip(bars, [baseline, peak]):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f} MB", ha="center", fontsize=9)
        ax5.set_ylim(0, max(baseline, peak) * 1.2)
    ax5.set_ylabel("MB")
    ax5.set_title("Memory Usage", fontsize=11)

    # 6 — Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    overall_acc = scores["accuracy"].mean() * 100
    m = metrics
    lines = [
        f"Model: {config.MODEL_DISPLAY}",
        f"Overall accuracy: {overall_acc:.1f}%",
        f"Prompt throughput: {m.get('prompt_tokens_per_sec', {}).get('mean', 0):.0f} tok/s (mean)",
        f"Gen throughput: {m.get('generation_tokens_per_sec', {}).get('mean', 0):.0f} tok/s (mean)",
        f"TTFT: {m.get('ttft_ms', {}).get('median', 0):.0f} ms (median)",
        f"Memory: {mem.get('peak_rss_mb', 0):.0f} MB peak",
    ]
    if m.get("power_watts") is not None:
        lines.append(f"Power: {m['power_watts']:.1f} W")
    ax6.text(
        0.1, 0.9, "\n".join(lines),
        transform=ax6.transAxes, fontsize=10, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )
    ax6.set_title("Summary", fontsize=11)

    fig.suptitle(
        f"BFCL Benchmark — {config.MODEL_DISPLAY} (Metal)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(include_single: bool = True, include_multi: bool = True):
    """Generate all charts. Reads from DB or results dirs."""
    _setup_style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_id = config.RUN_ID

    # Load data
    if run_id:
        score_rows = db.get_scores(run_id)
        scores = pd.DataFrame(score_rows) if score_rows else pd.DataFrame()
        metrics = db.get_inference_metrics(run_id)
    else:
        scores_path = config.SCORES_DIR / "accuracy.csv"
        metrics_path = config.METRICS_DIR / "inference_metrics.json"
        scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    timings = _load_timings(include_single=include_single, include_multi=include_multi)

    charts_saved = 0

    if not scores.empty:
        fig = chart_accuracy(scores)
        _save(fig, "accuracy_by_category")
        charts_saved += 1
        print("[visualize] Saved accuracy_by_category")

    if not timings.empty:
        fig = chart_throughput(timings)
        _save(fig, "throughput_distribution")
        charts_saved += 1
        print("[visualize] Saved throughput_distribution")

        fig = chart_ttft(timings)
        _save(fig, "ttft_distribution")
        charts_saved += 1
        print("[visualize] Saved ttft_distribution")

        fig = chart_speed_vs_prompt(timings)
        _save(fig, "generation_vs_prompt_length")
        charts_saved += 1
        print("[visualize] Saved generation_vs_prompt_length")

    if metrics:
        fig = chart_memory(metrics)
        _save(fig, "memory_usage")
        charts_saved += 1
        print("[visualize] Saved memory_usage")

    if not scores.empty and not timings.empty:
        fig = chart_dashboard(scores, timings, metrics)
        _save(fig, "summary_dashboard")
        charts_saved += 1
        print("[visualize] Saved summary_dashboard")

    print(f"[visualize] {charts_saved} charts saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
