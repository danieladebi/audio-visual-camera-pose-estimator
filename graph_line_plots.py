#!/usr/bin/env python3
"""
Line plot comparison of a baseline model vs our model.

Edit the DATA dict below directly and run:
python graph_line_plots.py
"""

from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# EDIT THESE VALUES DIRECTLY
GaussianBlur = {
    "metric": "Accuracy",
    "labels": [0, 1, 2, 4, 8],
    "baseline": [7.33,  6.29, 4.819, 4.278, 4.078],
    "model":    [8.82, 7.62, 5.92, 5.55, 5.52],
    "title": "AUC@20 vs Gaussian Blur Sigma",
    "xlabel": "Gaussian Blur Sigma",
    "ylabel": "AUC@20",
    "save": "blur_plot.png",      # set to None to just show
    "show_diff": False,       # set False to hide difference line
}
# ------------------------------------------------------------------

# EDIT THESE VALUES DIRECTLY
GaussianNoise = {
    "metric": "Accuracy",
    "labels": [0, 0.05, 0.1, 0.2, 0.3],
    "baseline": [7.33,  5.817, 5.07, 3.517, 2.632],
    "model":    [8.82, 7.609, 6.212, 4.406, 3.47],
    "title": "AUC@20 vs Gaussian Noise",
    "xlabel": "Gaussian Noise Sigma",
    "ylabel": "AUC@20",
    "save": "noise_plot.png",      # set to None to just show
    "show_diff": False,       # set False to hide difference line
}

def plot_comparison(baseline, model, labels, metric, title=None,
                    save_path=None, show_diff=True, xlabel=None, ylabel=None):
    if len(baseline) != len(model) or len(labels) != len(baseline):
        raise ValueError("Lengths of baseline, model, and labels must match.")
    plt.figure(figsize=(8, 5))
    plt.plot(labels, baseline, marker='o', label='Baseline', linewidth=2)
    plt.plot(labels, model, marker='s', label='Our Model', linewidth=2)
    if show_diff:
        diff = [m - b for b, m in zip(baseline, model)]
        plt.plot(labels, diff, marker='^', linestyle='--',
                 label='Difference (Model - Baseline)', alpha=0.6)
    plt.xlabel(xlabel or "Step / Epoch")
    plt.ylabel(ylabel or metric)
    plt.title(title or f"{metric} Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    else:
        plt.show()
    plt.close()

def main():
    d = GaussianNoise
    plot_comparison(
        baseline=d["baseline"],
        model=d["model"],
        labels=d["labels"],
        metric=d["metric"],
        title=d.get("title"),
        save_path=d.get("save"),
        show_diff=d.get("show_diff", True),
        xlabel=d.get("xlabel"),
        ylabel=d.get("ylabel"),
    )

if __name__ == "__main__":
    main()
