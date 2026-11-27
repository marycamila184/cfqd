"""Small plotting helpers for the notebooks (budget curves).

Color follows the METHOD, never its rank: every figure uses the same fixed
slot per method (validated categorical palette, light mode).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHOD_COLORS = {          # fixed slots of the reference categorical palette
    "map_elites": "#2a78d6",
    "dice_genetic": "#eb6834",
    "wachter": "#1baf7a",
    "face_knn": "#eda100",
    "dice_random": "#e87ba4",
    "novelty": "#008300",
    "nslc": "#6250d6",
    "moc": "#e34948",
}
MODEL_NAMES = {"lr": "Logistic regression", "rf": "Random forest", "nn": "Neural network"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def budget_curves(summary: pd.DataFrame, metric: str, title: str, ylabel: str,
                  facet: str = "model", ylim=None):
    """One panel per model; one 2px line per method, metric averaged over instances/seeds."""
    g = summary.groupby([facet, "method", "budget"])[metric].mean().reset_index()
    facets = [f for f in ["lr", "rf", "nn"] if f in set(g[facet])] or sorted(set(g[facet]))
    fig, axes = plt.subplots(1, len(facets), figsize=(4.2 * len(facets), 3.4), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, f in zip(axes[0], facets):
        _style(ax)
        sub = g[g[facet] == f]
        for m in [k for k in METHOD_COLORS if k in set(sub["method"])]:
            s = sub[sub["method"] == m].sort_values("budget")
            ax.plot(s["budget"], s[metric], color=METHOD_COLORS[m], linewidth=2,
                    marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=m)
        ax.set_xscale("log")
        ax.set_title(MODEL_NAMES.get(f, f), color=INK, fontsize=10, loc="left")
        ax.set_xlabel("budget (model queries)", color=INK_2, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel, color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.01, y=1.10, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    return fig
