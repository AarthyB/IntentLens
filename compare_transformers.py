import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

LABEL_NAMES = ["Platonic", "Romantic", "Ambiguous"]
OUT_DIR     = Path("results/transformer_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


#  helpers 
def styled_fig_ax(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d14")
    ax.set_facecolor("#111119")
    ax.spines[:].set_color("#2a2a38")
    ax.tick_params(colors="#9898b4")
    return fig, ax


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d14")
    plt.close(fig)
    print(f"  Saved → {path}")


#  load dataset 
print("\n" + "="*62)
print("  Custom Transformer  vs  DistilBERT")
print("="*62)

from data.dataset_builder import build_dataset_v2
if not Path("data/dataset.csv").exists():
    build_dataset_v2("data/dataset.csv")

df = pd.read_csv("data/dataset.csv")
train_df, temp  = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df,   test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

tr_t, tr_l = train_df["text"].tolist(), train_df["label"].tolist()
va_t, va_l = val_df["text"].tolist(),   val_df["label"].tolist()
te_t, te_l = test_df["text"].tolist(),  test_df["label"].tolist()

print(f"\nDataset: {len(df)} total | Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")


#  Model 1: Custom Transformer 
print("\n[1/2] Custom Transformer (trained from scratch)…")
from ml.deep_model import DeepIntentTrainer

custom = DeepIntentTrainer(
    vocab_size=10000, embed_dim=128, num_heads=4,
    num_layers=3, ffn_dim=256, max_len=64,
    batch_size=128, epochs=10, lr=5e-4, warmup_steps=150
)
if custom.model_exists:
    print("  Loading saved weights…")
    custom.load()
else:
    print("  Training…")
    custom.fit(tr_t, tr_l, va_t, va_l)
    custom.save()

custom_preds  = custom.predict(te_t)
custom_probs  = custom.predict_proba(te_t)
custom_hist   = custom.history


#  Model 2: DistilBERT 
print("\n[2/2] DistilBERT (pretrained + fine-tuned)…")
distilbert_available = False
distilbert_preds = None
distilbert_probs = None
distilbert_hist  = None

try:
    from ml.distilbert_model import DistilBERTTrainer
    dbert = DistilBERTTrainer(epochs=5, batch_size=16, lr=2e-5)

    if dbert.model_exists:
        print("  Loading saved weights…")
        dbert.load()
    else:
        print("  Training DistilBERT (this takes ~10-20 min on CPU, faster on GPU)…")
        dbert.fit(tr_t, tr_l, va_t, va_l)
        dbert.save()

    distilbert_preds  = dbert.predict(te_t)
    distilbert_probs  = dbert.predict_proba(te_t)
    distilbert_hist   = getattr(dbert, "history", None)
    distilbert_available = True
    print("  DistilBERT ready ✓")

except ImportError:
    print("  ⚠  transformers not installed.")
    print("     Run: pip install transformers")
    print("     Then re-run this script.")
except Exception as e:
    print(f"  ⚠  DistilBERT error: {e}")


#  Metrics 
def get_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {p:.4f}   Recall: {r:.4f}   F1: {f1:.4f}")
    print(f"\n  Per-class F1:")
    for i, n in enumerate(LABEL_NAMES):
        print(f"    {n:12s}: {f1_c[i]:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}")
    return {"name": name, "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "per_class_f1": f1_c, "cm": cm}

custom_metrics = get_metrics(te_l, custom_preds, "Custom Transformer (from scratch)")
models = [custom_metrics]

if distilbert_available:
    dbert_metrics = get_metrics(te_l, distilbert_preds, "DistilBERT (pretrained + fine-tuned)")
    models.append(dbert_metrics)
else:
    print("\n  DistilBERT not available — showing Custom Transformer results only.")


#  Plot 1: Side-by-side metrics bar chart 
print("\n  Generating plots…")
metrics_keys   = ["accuracy", "precision", "recall", "f1"]
metrics_labels = ["Accuracy", "Precision", "Recall", "F1"]
colors = ["#6366f1", "#f472b6"]
x = np.arange(len(metrics_keys))
width = 0.35 if len(models) == 2 else 0.5

fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d0d14")
ax.set_facecolor("#111119")
ax.spines[:].set_color("#2a2a38")

for i, m in enumerate(models):
    vals   = [m[k] for k in metrics_keys]
    offset = (i - len(models)/2 + 0.5) * width
    bars   = ax.bar(x + offset, vals, width * 0.9,
                    label=m["name"], color=colors[i], alpha=0.85,
                    edgecolor="#111119")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, color="#eeeef5", fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(metrics_labels, fontsize=12, color="#eeeef5")
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score", color="#9898b4", fontsize=11)
ax.set_title("Custom Transformer vs DistilBERT — Overall Metrics",
             fontsize=13, fontweight="bold", color="#eeeef5", pad=12)
ax.tick_params(colors="#9898b4")
ax.grid(axis="y", color="#2a2a38", alpha=0.6)
ax.legend(facecolor="#1f1f2a", labelcolor="#eeeef5",
          edgecolor="#2a2a38", fontsize=10)
plt.tight_layout()
save(fig, "01_overall_metrics.png")


#  Plot 2: Per-class F1 comparison 
cls_colors = ["#60a5fa", "#f472b6", "#fbbf24"]
x2 = np.arange(len(LABEL_NAMES))

fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d0d14")
ax.set_facecolor("#111119")
ax.spines[:].set_color("#2a2a38")

for i, m in enumerate(models):
    offset = (i - len(models)/2 + 0.5) * width
    for j, (f1_val, cls_color) in enumerate(zip(m["per_class_f1"], cls_colors)):
        bar = ax.bar(x2[j] + offset, f1_val, width * 0.9,
                     color=colors[i], alpha=0.85, edgecolor="#111119",
                     label=m["name"] if j == 0 else "")
        ax.text(x2[j] + offset, f1_val + 0.004,
                f"{f1_val:.3f}", ha="center", va="bottom",
                fontsize=9, color="#eeeef5")

ax.set_xticks(x2); ax.set_xticklabels(LABEL_NAMES, fontsize=12, color="#eeeef5")
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score", color="#9898b4", fontsize=11)
ax.set_title("Per-Class F1: Custom Transformer vs DistilBERT",
             fontsize=13, fontweight="bold", color="#eeeef5", pad=12)
ax.tick_params(colors="#9898b4")
ax.grid(axis="y", color="#2a2a38", alpha=0.6)
ax.legend(facecolor="#1f1f2a", labelcolor="#eeeef5",
          edgecolor="#2a2a38", fontsize=10)
plt.tight_layout()
save(fig, "02_per_class_f1.png")


#  Plot 3: Confusion matrices 
n_plots = len(models)
fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6), facecolor="#0d0d14")
if n_plots == 1: axes = [axes]

for ax, m in zip(axes, models):
    cm_norm = m["cm"].astype(float) / m["cm"].sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                linewidths=0.5, ax=ax, cbar=False,
                annot_kws={"size": 11, "color": "white"})
    ax.set_facecolor("#111119")
    ax.set_title(m["name"], fontsize=10, fontweight="bold",
                 color="#eeeef5", pad=10)
    ax.set_xlabel("Predicted", fontsize=10, color="#9898b4")
    ax.set_ylabel("True",      fontsize=10, color="#9898b4")
    ax.tick_params(colors="#9898b4")

plt.suptitle("Confusion Matrices (row-normalised)",
             fontsize=13, fontweight="bold", color="#eeeef5", y=1.02)
plt.tight_layout()
save(fig, "03_confusion_matrices.png")


#  Plot 4: Training curves (if both available) 
histories = []
if custom_hist and "train_loss" in custom_hist:
    histories.append(("Custom Transformer", custom_hist, "#6366f1"))
if distilbert_available and distilbert_hist and "train_loss" in distilbert_hist:
    histories.append(("DistilBERT", distilbert_hist, "#f472b6"))

if histories:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d14")
    for name, hist, color in histories:
        ep = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], marker="o", ms=4,
                     color=color, label=f"{name} train", linewidth=1.8)
        axes[0].plot(ep, hist["val_loss"],   marker="s", ms=4,
                     color=color, label=f"{name} val",
                     linestyle="--", linewidth=1.8, alpha=0.7)
        axes[1].plot(ep, hist["train_acc"],  marker="o", ms=4,
                     color=color, label=f"{name} train", linewidth=1.8)
        axes[1].plot(ep, hist["val_acc"],    marker="s", ms=4,
                     color=color, label=f"{name} val",
                     linestyle="--", linewidth=1.8, alpha=0.7)

    for ax, title, ylabel in zip(axes,
                                  ["Loss Curves", "Accuracy Curves"],
                                  ["Loss", "Accuracy"]):
        ax.set_facecolor("#111119")
        ax.spines[:].set_color("#2a2a38")
        ax.tick_params(colors="#9898b4")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#eeeef5")
        ax.set_xlabel("Epoch", color="#9898b4")
        ax.set_ylabel(ylabel, color="#9898b4")
        ax.legend(facecolor="#1f1f2a", labelcolor="#eeeef5",
                  edgecolor="#2a2a38", fontsize=9)
        ax.grid(alpha=0.3, color="#2a2a38")

    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    save(fig, "04_training_curves.png")


#  Plot 5: Radar chart 
dims   = ["Accuracy", "Precision", "Recall", "F1",
          "Platonic\nF1", "Romantic\nF1", "Ambiguous\nF1"]
N      = len(dims)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True},
                        facecolor="#0d0d14")
ax.set_facecolor("#111119")
ax.spines["polar"].set_color("#2a2a38")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, color="#eeeef5", size=10)
ax.set_ylim(0, 1)
ax.yaxis.set_tick_params(labelleft=False)
for g in ax.yaxis.get_gridlines():
    g.set_color("#2a2a38"); g.set_alpha(0.5)

for m, color in zip(models, colors):
    vals = [m["accuracy"], m["precision"], m["recall"], m["f1"],
            m["per_class_f1"][0], m["per_class_f1"][1], m["per_class_f1"][2]]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2.5, color=color, label=m["name"])
    ax.fill(angles, vals, alpha=0.12, color=color)

ax.set_title("Transformer Comparison — Radar Chart",
             fontsize=13, fontweight="bold", color="#eeeef5", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1),
          facecolor="#1f1f2a", labelcolor="#eeeef5", edgecolor="#2a2a38")
plt.tight_layout()
save(fig, "05_radar_chart.png")


#  Summary 
print(f"\n{'='*62}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*62}")
print(f"\n  {'Metric':<20} {'Custom Transformer':>22} {'DistilBERT':>14}")
print(f"  {'─'*56}")
for k, label in zip(metrics_keys, metrics_labels):
    custom_val = f"{custom_metrics[k]:.4f}"
    dbert_val  = f"{dbert_metrics[k]:.4f}" if distilbert_available else "N/A"
    winner     = ""
    if distilbert_available:
        if custom_metrics[k] > dbert_metrics[k]: winner = " ← better"
        elif dbert_metrics[k] > custom_metrics[k]: winner = "  DistilBERT better →"
    print(f"  {label:<20} {custom_val:>22} {dbert_val:>14}{winner}")

print(f"\n  Per-class F1:")
for i, name in enumerate(LABEL_NAMES):
    custom_val = f"{custom_metrics['per_class_f1'][i]:.4f}"
    dbert_val  = f"{dbert_metrics['per_class_f1'][i]:.4f}" if distilbert_available else "N/A"
    print(f"    {name:<16} Custom={custom_val}   DistilBERT={dbert_val}")

# Save summary CSV
rows = []
for m in models:
    rows.append({
        "Model":        m["name"],
        "Accuracy":     round(m["accuracy"],  4),
        "Precision":    round(m["precision"], 4),
        "Recall":       round(m["recall"],    4),
        "Macro F1":     round(m["f1"],        4),
        "Platonic F1":  round(m["per_class_f1"][0], 4),
        "Romantic F1":  round(m["per_class_f1"][1], 4),
        "Ambiguous F1": round(m["per_class_f1"][2], 4),
    })
csv_path = OUT_DIR / "06_comparison_table.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\n  Saved → {csv_path}")

print(f"\n  ✅ All outputs saved to {OUT_DIR}/")
print(f"{'='*62}\n")
