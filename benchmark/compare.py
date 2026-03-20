"""Generate comparison charts across multiple model benchmark runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from . import config, db

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
COMPARE_DIR = config.RESULTS_DIR / "comparison"


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
    for ext in ("png", "pdf"):
        fig.savefig(COMPARE_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_models() -> list[dict]:
    """Load accuracy + metrics for every model that has results."""
    db_results = db.get_all_model_results()
    if db_results:
        models = []
        for r in db_results:
            scores_df = pd.DataFrame(r["scores"])
            models.append({
                "key": r["key"],
                "display": r["display"],
                "slug": r["slug"],
                "scores": scores_df,
                "metrics": r["metrics"],
            })
        return models

    # Fallback to flat files
    models = []
    for key, info in config.MODELS.items():
        slug = info["slug"]
        model_dir = config.RESULTS_DIR / slug
        scores_path = model_dir / "scores" / "accuracy.csv"
        metrics_path = model_dir / "metrics" / "inference_metrics.json"
        if not scores_path.exists():
            continue
        scores = pd.read_csv(scores_path)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        models.append({
            "key": key,
            "display": info["display"],
            "slug": slug,
            "scores": scores,
            "metrics": metrics,
        })
    return models


# ---------------------------------------------------------------------------
# Chart 1: Grouped accuracy bar chart
# ---------------------------------------------------------------------------

def chart_accuracy_comparison(models: list[dict]) -> plt.Figure:
    categories = models[0]["scores"]["category"].tolist()
    n_cats = len(categories)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_cats)
    width = 0.7 / n_models
    palette = sns.color_palette("colorblind", n_models)

    for i, m in enumerate(models):
        acc = m["scores"].set_index("category").loc[categories, "accuracy"] * 100
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, acc.values, width, label=m["display"],
                      color=palette[i], edgecolor="white")
        for bar, val in zip(bars, acc.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("BFCL Accuracy by Category — Model Comparison")
    ax.legend(fontsize=9, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Performance metrics comparison (grouped bar)
# ---------------------------------------------------------------------------

def chart_performance_comparison(models: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    palette = sns.color_palette("colorblind", len(models))
    labels = [m["display"] for m in models]

    # Prompt throughput
    ax = axes[0]
    vals = [m["metrics"].get("prompt_tokens_per_sec", {}).get("mean", 0) for m in models]
    bars = ax.bar(labels, vals, color=palette, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Prompt Throughput (mean)")
    ax.tick_params(axis="x", rotation=15)

    # Generation throughput
    ax = axes[1]
    vals = [m["metrics"].get("generation_tokens_per_sec", {}).get("mean", 0) for m in models]
    bars = ax.bar(labels, vals, color=palette, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Generation Throughput (mean)")
    ax.tick_params(axis="x", rotation=15)

    # TTFT
    ax = axes[2]
    vals = [m["metrics"].get("ttft_ms", {}).get("median", 0) for m in models]
    bars = ax.bar(labels, vals, color=palette, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f} ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_title("TTFT (median)")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Inference Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Memory comparison
# ---------------------------------------------------------------------------

def chart_memory_comparison(models: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.3
    palette = sns.color_palette("colorblind", 2)

    baselines = [m["metrics"].get("memory", {}).get("baseline_rss_mb", 0) for m in models]
    peaks = [m["metrics"].get("memory", {}).get("peak_rss_mb", 0) for m in models]
    labels = [m["display"] for m in models]

    bars1 = ax.bar(x - width / 2, baselines, width, label="Baseline RSS",
                   color=palette[0], edgecolor="white")
    bars2 = ax.bar(x + width / 2, peaks, width, label="Peak RSS",
                   color=palette[1], edgecolor="white")

    for bar, val in zip(bars1, baselines):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:.0f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, peaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:.0f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage Comparison")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(peaks) * 1.2 if peaks else 100)
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Summary comparison dashboard
# ---------------------------------------------------------------------------

def chart_summary_dashboard(models: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    palette = sns.color_palette("colorblind", len(models))

    categories = models[0]["scores"]["category"].tolist()
    n_cats = len(categories)
    n_models = len(models)

    # --- Panel 1: Accuracy grouped bars ---
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(n_cats)
    width = 0.7 / n_models
    for i, m in enumerate(models):
        acc = m["scores"].set_index("category").loc[categories, "accuracy"] * 100
        offset = (i - (n_models - 1) / 2) * width
        bars = ax1.bar(x + offset, acc.values, width, label=m["display"],
                       color=palette[i], edgecolor="white")
        for bar, val in zip(bars, acc.values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_title("BFCL Accuracy by Category")
    ax1.legend(fontsize=9)

    # --- Panel 2: Throughput comparison ---
    ax2 = fig.add_subplot(gs[1, 0])
    metrics_labels = ["Prompt\n(mean)", "Generation\n(mean)"]
    x2 = np.arange(len(metrics_labels))
    width2 = 0.7 / n_models
    for i, m in enumerate(models):
        vals = [
            m["metrics"].get("prompt_tokens_per_sec", {}).get("mean", 0),
            m["metrics"].get("generation_tokens_per_sec", {}).get("mean", 0),
        ]
        offset = (i - (n_models - 1) / 2) * width2
        bars = ax2.bar(x2 + offset, vals, width2, label=m["display"],
                       color=palette[i], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f}", ha="center", fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylabel("Tokens / sec")
    ax2.set_title("Throughput Comparison")
    ax2.legend(fontsize=8)

    # --- Panel 3: Summary table ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    rows = []
    for m in models:
        mem = m["metrics"].get("memory", {})
        overall = m["scores"]["accuracy"].mean() * 100
        rows.append([
            m["display"],
            f"{overall:.1f}%",
            f"{m['metrics'].get('prompt_tokens_per_sec', {}).get('mean', 0):.0f}",
            f"{m['metrics'].get('generation_tokens_per_sec', {}).get('mean', 0):.0f}",
            f"{m['metrics'].get('ttft_ms', {}).get('median', 0):.0f}",
            f"{mem.get('peak_rss_mb', 0):.0f}",
        ])

    col_labels = ["Model", "Avg Acc", "Prompt\ntok/s", "Gen\ntok/s", "TTFT\n(ms)", "Peak RSS\n(MB)"]
    table = ax3.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax3.set_title("Summary", fontsize=11)

    fig.suptitle(
        "BFCL Benchmark — Model Comparison (Apple M4 Pro, Metal)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run():
    """Generate comparison charts for all models with results."""
    _setup_style()
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    models = _load_models()
    if len(models) < 2:
        print(f"[compare] Need at least 2 models with results, found {len(models)}. Skipping.")
        return

    print(f"[compare] Comparing {len(models)} models: {', '.join(m['display'] for m in models)}")

    fig = chart_accuracy_comparison(models)
    _save(fig, "accuracy_comparison")
    print("[compare] Saved accuracy_comparison")

    fig = chart_performance_comparison(models)
    _save(fig, "performance_comparison")
    print("[compare] Saved performance_comparison")

    fig = chart_memory_comparison(models)
    _save(fig, "memory_comparison")
    print("[compare] Saved memory_comparison")

    fig = chart_summary_dashboard(models)
    _save(fig, "comparison_dashboard")
    print("[compare] Saved comparison_dashboard")

    print(f"[compare] 4 charts saved to {COMPARE_DIR}")


if __name__ == "__main__":
    run()
