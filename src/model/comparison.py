import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

models = [
    "Linear\nRegression",
    "KNN\n(k=3)",
    "XGBoost",
    "Random\nForest",
    "Lasso\n(L1 reg)",
    "Ridge\n(L2 reg)",
    "Neural\nNetwork"
]

mae     = [5.003, 3.077, 3.596, 2.741, 5.348, 5.306, 3.020]
rmse    = [7.045, 4.994, 5.331, 4.193, 7.452, 7.472, 4.331]
max_err = [80.173, 86.300, 79.557, 56.209, 84.292, 84.355, 79.951]

baseline_mae    = 4.344
baseline_rmse   = 6.050
baseline_maxerr = 76.529

bg_colors = [
    "#FFE8E8",
    "#E8F0FF",
    "#E8FFE8",
    "#FFFBE8",
    "#F0E8FF",
    "#FFE8F5",
    "#DFFAF5",
]

bar_colors = {
    "MAE (L1)": "#185FA5",
    "RMSE (L2)": "#F28C28",
    "Max Error (L∞)": "#6A0DAD",
}

metrics = [
    ("MAE (L1)", mae, baseline_mae),
    ("RMSE (L2)", rmse, baseline_rmse),
    ("Max Error (L∞)", max_err, baseline_maxerr),
]

n = len(models)
x = np.arange(n)
width = 0.22
offsets = [-width, 0, width]

fig, ax = plt.subplots(figsize=(14, 7))

# Background panels
for i, bg in enumerate(bg_colors):
    ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.25, zorder=0)

# Bars
for (label, data, baseline), offset in zip(metrics, offsets):
    color = bar_colors[label]

    bars = ax.bar(
        x + offset,
        data,
        width * 0.92,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    for bar, value in zip(bars, data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            zorder=4,
        )

    ax.axhline(
        baseline,
        color=color,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Baseline {label} ({baseline:.3f})",
        zorder=2,
    )

ax.set_yscale("log")
ax.set_ylabel("Error (passengers, log scale)", fontsize=12)
ax.set_title(
    "Model comparison: test error under L1, L2, and L∞ norms\n"
    "(predicting average bus load)",
    fontsize=14,
)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)

ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=1, which="both")
ax.spines[["top", "right"]].set_visible(False)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    fontsize=9,
    framealpha=0.9,
)

plt.tight_layout()

out_dir = project_root / "diagram" / "model"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved: {out_path}")