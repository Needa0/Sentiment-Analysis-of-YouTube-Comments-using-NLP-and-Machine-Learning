"""Basic, theme-aware visualisations for language statistics.

Provides bar and pie charts of the language distribution. All functions return a
Matplotlib ``Figure`` so charts can be embedded in the dashboard or saved to
disk. A non-interactive backend is used so charts render headless.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from .config import LANGUAGE_COLORS  # noqa: E402

# Theme palettes for dark / light mode.
_THEMES = {
    "light": {"bg": "#FFFFFF", "fg": "#1F2933", "grid": "#E4E7EB"},
    "dark": {"bg": "#0E1117", "fg": "#FAFAFA", "grid": "#2A2E37"},
}


def _palette(labels, fallback="#4C72B0"):
    return [LANGUAGE_COLORS.get(label, fallback) for label in labels]


def _apply_theme(fig, ax, theme: str):
    colors = _THEMES.get(theme, _THEMES["light"])
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.tick_params(colors=colors["fg"])
    for spine in ax.spines.values():
        spine.set_color(colors["grid"])
    ax.xaxis.label.set_color(colors["fg"])
    ax.yaxis.label.set_color(colors["fg"])
    ax.title.set_color(colors["fg"])
    return colors


def bar_chart(
    counts: Dict[str, int],
    title: str = "Language Distribution",
    xlabel: str = "Language",
    ylabel: str = "Comments",
    theme: str = "light",
):
    """Create a bar chart from a label -> count mapping."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = _apply_theme(fig, ax, theme)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    bars = ax.bar(labels, values, color=_palette(labels))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=colors["grid"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)
    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            str(value),
            ha="center",
            va="bottom",
            color=colors["fg"],
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def pie_chart(
    counts: Dict[str, int],
    title: str = "Language Share",
    theme: str = "light",
):
    """Create a donut-style pie chart from a label -> count mapping.

    Slice labels are shown in a legend outside the donut (not on the wedges) so
    they never overlap, even when one language dominates. Percentages are drawn
    inside the ring and hidden for very small slices to avoid clutter.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = _apply_theme(fig, ax, theme)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    total = sum(values) or 1

    def _autopct(pct):
        # Hide percentage text for slices too small to render legibly.
        return f"{pct:.1f}%" if pct >= 4 else ""

    wedges, _texts, autotexts = ax.pie(
        values,
        autopct=_autopct,
        startangle=90,
        colors=_palette(labels),
        pctdistance=0.78,
        wedgeprops={"width": 0.45, "edgecolor": colors["bg"], "linewidth": 1.2},
        textprops={"color": "#FFFFFF", "fontsize": 9, "fontweight": "bold"},
    )
    ax.axis("equal")
    ax.set_title(title, fontweight="bold", color=colors["fg"])

    legend_labels = [
        f"{lab}  ·  {val} ({100.0 * val / total:.1f}%)"
        for lab, val in zip(labels, values)
    ]
    legend = ax.legend(
        wedges,
        legend_labels,
        title="Language",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9,
        labelcolor=colors["fg"],
    )
    legend.get_title().set_color(colors["fg"])
    fig.tight_layout()
    return fig


def save_figure(fig, path: str) -> str:
    """Save a Matplotlib figure to disk, creating parent directories."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def language_palette(theme: Optional[str] = None) -> Dict[str, str]:
    """Expose the language colour palette for use in the dashboard."""
    return dict(LANGUAGE_COLORS)
