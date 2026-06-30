import math
from pathlib import Path
from typing import List, Sequence, Optional, Union
import numpy as np

#!/usr/bin/env python3
"""
plot_ambient_audio_results.py

Utility to create spider (radar) plots for ambient audio pattern comparison.
"""

import matplotlib.pyplot as plt


def create_spider_plot(
    categories: Sequence[str],
    data_series: Sequence[Sequence[Union[int, float]]],
    series_labels: Optional[Sequence[str]] = None,
    title: str = "Ambient Audio Pattern Radar",
    normalize: bool = False,
    fill_alpha: float = 0.15,
    figsize=(8, 8),
    palette: Optional[Sequence[str]] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Create a spider (radar) plot.

    categories: names of axes (e.g., audio types)
    data_series: iterable of numeric sequences, one per series
    series_labels: labels for each data series
    normalize: if True, scales each axis to [0,1] across all series
    palette: optional list of colors (fallback to matplotlib cycle)
    """
    if not categories:
        raise ValueError("categories is empty")
    n_axes = len(categories)

    # Validate data
    cleaned = []
    for i, s in enumerate(data_series):
        if len(s) != n_axes:
            raise ValueError(f"Series {i} length {len(s)} != number of categories {n_axes}")
        cleaned.append(np.array(s, dtype=float))
    data_series = cleaned

    if normalize:
        stacked = np.vstack(data_series)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        span = np.where(maxs - mins == 0, 1, maxs - mins)
        data_series = [(ds - mins) / span for ds in data_series]

    # Angles
    angles = np.linspace(0, 2 * math.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close

    # Prepare figure
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, pad=20)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Radial limits
    all_values = np.concatenate([ds for ds in data_series])
    r_min = 0 if normalize or all(v >= 0 for v in all_values) else float(all_values.min())
    r_max = float(all_values.max())
    if r_min == r_max:
        r_max = r_min + 1.0
    ax.set_rlim(r_min, r_max)

    # Optional grid formatting
    ax.grid(color="#888888", linestyle=":", linewidth=0.6)

    # Colors
    if palette is None:
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if series_labels is None:
        series_labels = [f"S{i+1}" for i in range(len(data_series))]

    # Plot each series
    for idx, (values, label) in enumerate(zip(data_series, series_labels)):
        vals = values.tolist()
        vals += vals[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, vals, label=label, linewidth=2, color=color)
        ax.fill(angles, vals, color=color, alpha=fill_alpha)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def example():
    audio_categories = ["Far-field", "Near-field/Egocentric", "Dominant Single Source", "Low Audio Signal", "Frequent Sound Changes"]
    # Three hypothetical environments / time windows
    series = [
        [3, 12, 25, 18, 9],

    ]
    labels = ["How Often is Our Method Better?"]
    create_spider_plot(
        categories,
        series,
        series_labels=labels,
        title="Ambient Audio Feature Intensity",
        normalize=False,
        save_path="radar_audio.png",
    )
    # Normalized version
    create_spider_plot(
        categories,
        series,
        series_labels=labels,
        title="Ambient Audio Feature Intensity (Normalized)",
        normalize=True,
        show=True,
    )


if __name__ == "__main__":
    example()