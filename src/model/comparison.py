import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# Absolute test metrics per model (predicting average_load on 2024 test set)
# max_err = L-inf = single worst-case absolute error across the test set
models = [
    "Linear\nRegression",
    "KNN\n(k=3)",
    "XGBoost",
    "Random\nForest",
    "Lasso\n(L1 reg)",
    "Ridge\n(L2 reg)",
]
mae     = [5.003, 3.077, 3.596, 2.741, 5.348, 5.306]
rmse    = [7.045, 4.994, 5.331, 4.193, 7.452, 7.472]
max_err = [80.173, 86.300, 79.557, 56.209, 84.292, 84.355]

# Baseline: mean-lookup (route × time × day_type from training set)
baseline_mae    = 4.344
baseline_rmse   = 6.050
baseline_maxerr = 76.529

# Style — one background colour per model, three bar colours for the three metrics
bg_colors  = ["#FFE8E8", "#E8F0FF", "#E8FFE8", "#FFFBE8", "#F0E8FF", "#FFE8F5"]
bar_colors = {"MAE (L1)": "#185FA5", "RMSE (L2)": "#F28C28", "Max Error (L∞)": "#6A0DAD"}

metrics = [
    ("MAE (L1)",       mae,     baseline_mae),
    ("RMSE (L2)",      rmse,    baseline_rmse),
    ("Max Error (L∞)", max_err, baseline_maxerr),
]

n      = len(models)
x      = np.arange(n)
width  = 0.22
offsets = [-width, 0, width]

fig, ax = plt.subplots(figsize=(13, 7))

# Coloured background panel per model group
for i, bg in enumerate(bg_colors[:n]):
    ax.axvspan(i - 0.5, i + 0.5, color=bg, alpha=0.55, zorder=0)

# Grouped bars: three bars per model (MAE / RMSE / Max Error)
for (label, data, baseline), offset in zip(metrics, offsets):
    color = bar_colors[label]
    vals  = [v if v is not None else 0.0 for v in data]
    bars  = ax.bar(x + offset, vals, width * 0.92, label=label, color=color, zorder=3)

    for bar, orig in zip(bars, data):
        if orig is None or orig == 0.0:
            continue
        # on log scale, offset the label multiplicatively so it sits just above the bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.06,
                f"{orig:.2f}", ha="center", va="bottom", fontsize=8.5, zorder=4)

    if baseline is not None:
        ax.axhline(baseline, color=color, linestyle="--", linewidth=1.1,
                   alpha=0.75, label=f"Baseline {label} ({baseline})", zorder=2)

ax.set_yscale("log")
ax.set_ylabel("Error (passengers, log scale)", fontsize=12)
ax.set_title("Model comparison: test error under L1, L2, and L∞ norms\n(predicting average bus load)",
             fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=1, which="both")

plt.tight_layout()
out_path = project_root / "diagram" / "model" / "model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
