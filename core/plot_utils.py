"""plot_utils.py — Plotting helpers for thyroid-qml experiment results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, path, title="", normalize=True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax)

    labels = ["No recurrence", "Recurrence"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

    fmt = ".2f" if normalize else "d"
    thresh = 0.5 if normalize else cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_scores, path, title="", auc_val=None):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(4, 4))
    label = f"AUC = {auc_val:.3f}" if auc_val is not None and not np.isnan(auc_val) else None
    ax.plot(fpr, tpr, lw=1.5, color="steelblue", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    if label:
        ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history, path, title=""):
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(loss_history, lw=1.5, color="coral")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Loss (cross-entropy)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_boxplot(df, metric: str, group_col: str, path,
                          title="", hue_col: str | None = None):
    """Boxplot of metric across seeds grouped by group_col, optionally split by hue_col."""
    import pandas as pd

    groups = sorted(df[group_col].unique())

    if hue_col and hue_col in df.columns:
        hues = sorted(df[hue_col].unique())
        n_groups = len(groups)
        n_hues = len(hues)
        width = 0.8 / n_hues
        colors = plt.cm.Set2(np.linspace(0, 0.8, n_hues))

        fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.5 * n_hues), 4))
        for hi, hue in enumerate(hues):
            sub = df[df[hue_col] == hue]
            data = [sub[sub[group_col] == g][metric].dropna().values for g in groups]
            positions = [i + (hi - n_hues / 2 + 0.5) * width for i in range(n_groups)]
            bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[hi])
                patch.set_alpha(0.75)
            ax.plot([], [], color=colors[hi], label=str(hue), linewidth=4, alpha=0.75)

        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.legend(title=hue_col, fontsize=8)
    else:
        data = [df[df[group_col] == g][metric].dropna().values for g in groups]
        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.4), 4))
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)

    ax.set_xlabel(group_col, fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, row_col: str, col_col: str, metric: str, path, title="", fmt=".3f"):
    """Mean metric heatmap: rows=row_col, cols=col_col."""
    pivot = df.groupby([row_col, col_col])[metric].mean().unstack(col_col)

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.2),
                                    max(3, len(pivot) * 0.8)))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel(col_col, fontsize=9)
    ax.set_ylabel(row_col, fontsize=9)

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7,
                        color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")

    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
