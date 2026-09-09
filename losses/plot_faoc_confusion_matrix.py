"""
Renders the "confusion-matrix corners" scenario from validate_faoc_loss.py
as a pair of annotated heatmaps: plain BCE vs. FAOC-Loss, same predictions,
same true labels, airport fixed at ORD. Saves faoc_confusion_matrix.png
alongside this script.

Color choice: a single-hue, perceptually-uniform sequential colormap
(viridis) with a log color scale - each panel's values span 2-3 orders of
magnitude (e.g. FAOC-Loss: 29.8 to 68,372.8), so a linear scale would make
three of the four cells look identical. Every cell is also annotated with
its exact value, so the reader never has to rely on color alone to read a
number - only to spot the pattern at a glance.

Run from the repo root:
    python -m losses.plot_faoc_confusion_matrix
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from losses.faoc_loss import FAOCLoss

OUTPUT_PATH = Path(__file__).resolve().parent / "faoc_confusion_matrix.png"

AIRPORT = "ORD"
CONFIDENT_LOW = 0.02
CONFIDENT_HIGH = 0.98

# Grid layout: rows = actual label, cols = predicted class.
# [[actual=No,  pred=No ], [actual=No,  pred=Yes]]     ->  [[TN, FP],
#  [actual=Yes, pred=No ], [actual=Yes, pred=Yes]]           [FN, TP]]
SCENARIOS = [
    ("TN", CONFIDENT_LOW, 0),
    ("FP", CONFIDENT_HIGH, 0),
    ("FN", CONFIDENT_LOW, 1),
    ("TP", CONFIDENT_HIGH, 1),
]


def build_matrices():
    criterion = FAOCLoss()
    plain_bce = torch.nn.BCELoss(reduction="none")

    bce_grid = np.zeros((2, 2))
    faoc_grid = np.zeros((2, 2))
    labels_grid = np.empty((2, 2), dtype=object)

    positions = {"TN": (0, 0), "FP": (0, 1), "FN": (1, 0), "TP": (1, 1)}
    for name, p, y in SCENARIOS:
        row, col = positions[name]
        p_t = torch.tensor([p])
        y_t = torch.tensor([float(y)])
        bce_grid[row, col] = plain_bce(p_t.clamp(1e-7, 1 - 1e-7), y_t).item()
        faoc_grid[row, col] = criterion(p_t, y_t, [AIRPORT]).item()
        labels_grid[row, col] = name

    return bce_grid, faoc_grid, labels_grid


def draw_panel(ax, grid, labels_grid, title, value_fmt, cbar_label):
    norm = mcolors.LogNorm(vmin=grid.min(), vmax=grid.max())
    im = ax.imshow(grid, cmap="viridis", norm=norm)

    ax.set_xticks([0, 1], labels=["No disruption", "Disruption"])
    ax.set_yticks([0, 1], labels=["No disruption", "Disruption"])
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Text color follows the cell's normalized brightness, not a fixed
    # choice - viridis runs dark-purple to light-yellow, so a fixed color
    # would go illegible on one end of the range.
    for row in range(2):
        for col in range(2):
            value = grid[row, col]
            brightness = norm(value)
            text_color = "white" if brightness < 0.6 else "black"
            ax.text(
                col, row,
                f"{labels_grid[row, col]}\n{value_fmt(value)}",
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold",
            )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)


def main():
    bce_grid, faoc_grid, labels_grid = build_matrices()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    draw_panel(
        axes[0], bce_grid, labels_grid,
        title="Plain BCE (unweighted)",
        value_fmt=lambda v: f"{v:.3f}",
        cbar_label="loss (nats, log scale)",
    )
    draw_panel(
        axes[1], faoc_grid, labels_grid,
        title=f"FAOC-Loss (airport = {AIRPORT})",
        value_fmt=lambda v: f"{v:,.1f}",
        cbar_label="loss (cost-weighted units, log scale)",
    )

    fig.suptitle(
        f"Same predictions (p={CONFIDENT_LOW}/{CONFIDENT_HIGH}), same labels - "
        "plain BCE vs. FAOC-Loss",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Note: plain BCE's diagonal (TN, TP) is symmetric - it only sees confidence, not the label.\n"
        "FAOC-Loss's diagonal is not (29.8 vs 353.1) - a true positive on a real disruption still\n"
        "carries the higher class weight, even though it costs far less than missing one.",
        ha="center", fontsize=9, style="italic",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
