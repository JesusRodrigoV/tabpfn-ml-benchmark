#!/usr/bin/env python3
"""
comparativa.py — Lee resultados de los notebooks y genera gráficas comparativas.

Uso:
    1. Ejecutar notebook1_baseline.ipynb  (entrena RF + GB)
     2. Ejecutar notebook2_tabPFN.ipynb    (entrena TabPFN)
    3. python comparativa.py              (genera gráficas)

Requiere: numpy, pandas, matplotlib
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from math import pi

# ── Config ──
OUTPUT_DIR = "figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_COLORS = {
    "Random Forest": "#3498db",
    "Gradient Boosting": "#e67e22",
    "TabPFN": "#27ae60",
}
MODEL_MARKERS = ["o", "s", "D"]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.25,
})


# ── Carga de resultados ──
RESULTADOS_DIR = "resultados"

def cargar_metricas():
    paths = [
        os.path.join(RESULTADOS_DIR, "metrics_notebook1.json"),
        os.path.join(RESULTADOS_DIR, "metrics_notebook2.json"),
    ]
    combined = {}
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            for ds, modelos in data.items():
                if ds not in combined:
                    combined[ds] = {}
                combined[ds].update(modelos)
    if not combined:
        print("  No se encontraron métricas. Ejecutá los notebooks primero.")
        sys.exit(1)
    return combined

def cargar_predicciones():
    paths = [
        os.path.join(RESULTADOS_DIR, "preds_notebook1.npz"),
        os.path.join(RESULTADOS_DIR, "preds_notebook2.npz"),
    ]
    combined = {}
    for p in paths:
        if os.path.exists(p):
            combined.update(dict(np.load(p, allow_pickle=True)))
    if not combined:
        print("  No se encontraron predicciones. Ejecutá los notebooks primero.")
        sys.exit(1)

    # Normalizar etiquetas de Crop: notebook 1 usa strings, notebook 2 usa enteros.
    classes_c = combined.get("classes_c")
    if classes_c is not None:
        for k in ["y_test_c", "rf_preds_c", "gb_preds_c", "tabpfn_preds_c"]:
            arr = combined.get(k)
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) > 0:
                if isinstance(arr[0], (int, np.integer)):
                    combined[k] = np.array([classes_c[i] for i in arr], dtype=object)

    return combined


# ── Plot helpers ──
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  → {path}")
    plt.close(fig)


def _extraer_metricas(metrics):
    datasets = list(metrics.keys())
    modelos = ["Random Forest", "Gradient Boosting", "TabPFN"]
    metricas = ["accuracy", "f1", "auc"]
    metricas_etiqueta = ["Accuracy", "F1-Score", "ROC-AUC"]
    n_models = len(modelos)
    n_metrics = len(metricas)

    data = {}
    for ds in datasets:
        data[ds] = {}
        for m in modelos:
            if m in metrics.get(ds, {}):
                data[ds][m] = {met: metrics[ds][m].get(met, 0) for met in metricas}
                data[ds][m]["fit_time"] = metrics[ds][m].get("fit_time", 0)
                data[ds][m]["pred_time"] = metrics[ds][m].get("pred_time", 0)

    return datasets, modelos, metricas, metricas_etiqueta, data


# ── 1. ROC Curves ──
def plot_roc_german(preds, save_name="01_roc_german.png"):
    from sklearn.metrics import roc_curve, roc_auc_score

    modelos_roc = [
        ("Random Forest", "rf_probs_g", "#3498db", "-"),
        ("Gradient Boosting", "gb_probs_g", "#e67e22", "--"),
        ("TabPFN", "tabpfn_probs_g", "#27ae60", "-."),
    ]
    y_test = preds["y_test_g"]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, key, color, ls in modelos_roc:
        if key in preds:
            probs = preds[key]
            if probs.ndim == 2 and probs.shape[1] >= 2:
                probs = probs[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, color=color, lw=2.2, ls=ls,
                    label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("German Credit — Curvas ROC", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#ddd")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    save(fig, save_name)


# ── 2. Metrics comparison bar chart ──
def plot_metrics_comparison(metrics, save_name="02_metrics_comparison.png"):
    datasets, modelos, _, metricas_etiqueta, data = _extraer_metricas(metrics)
    n_models = 3
    x = np.arange(len(metricas_etiqueta))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for i, model in enumerate(modelos):
            if model not in data.get(dataset, {}):
                continue
            offset = (i - 1) * width
            vals = [data[dataset][model][m] for m in ["accuracy", "f1", "auc"]]
            bars = ax.bar(x + offset, vals, width, label=model if idx == 1 else "",
                          color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(metricas_etiqueta, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(dataset, fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.25)
        if idx == 1:
            ax.legend(framealpha=0.9, edgecolor="#ddd")

    fig.suptitle("Comparación de Métricas Predictivas", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


# ── 3. Timing comparison ──
def plot_timing_comparison(metrics, save_name="03_timing_comparison.png"):
    datasets, modelos, _, _, data = _extraer_metricas(metrics)
    phases = ["fit_time", "pred_time"]
    phase_labels = ["Fit", "Predict"]
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for pidx, (phase, plabel) in enumerate(zip(phases, phase_labels)):
        ax = axes[pidx]
        width = 0.22
        for i, model in enumerate(modelos):
            offset = (i - 1) * width
            vals = [data[d].get(model, {}).get(phase, 0) for d in datasets]
            bars = ax.bar(x + offset, vals, width, label=model if pidx == 1 else "",
                          color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v >= 1.0:
                    label = f"{v:.2f}s"
                elif v >= 0.001:
                    label = f"{v*1000:.0f}ms"
                else:
                    label = f"{v*1e6:.0f}µs"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                        label, ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel("Tiempo (s)")
        ax.set_title(f"{plabel} — Tiempo de Ejecución", fontweight="bold", pad=10)
        ax.set_yscale("log")
        # floor en el eje Y para evitar -inf con valores 0
        all_vals = [data[d].get(model, {}).get(phase, 0) for d in datasets for model in modelos if model in data.get(d, {})]
        non_zero = [v for v in all_vals if v > 0]
        if non_zero:
            ax.set_ylim(bottom=min(non_zero) / 10)
        ax.grid(True, axis="y", alpha=0.25)
        if pidx == 1:
            ax.legend(framealpha=0.9, edgecolor="#ddd")

    fig.suptitle("Comparación de Tiempos de Ejecución", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


# ── 4. Confusion matrices for German ──
def plot_confusion_german(preds, save_name="04_confusion_german.png"):
    modelos_cm = [
        ("Random Forest", "rf_preds_g", "Blues"),
        ("Gradient Boosting", "gb_preds_g", "Oranges"),
        ("TabPFN", "tabpfn_preds_g", "Greens"),
    ]
    y_test = preds["y_test_g"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, key, cmap) in zip(axes, modelos_cm):
        if key not in preds:
            ax.set_title(f"{name} — sin datos", fontweight="bold", pad=10)
            continue
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, preds[key])
        disp = ConfusionMatrixDisplay(cm, display_labels=["good", "bad"])
        disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    fig.suptitle("German Credit — Matrices de Confusión", fontweight="bold", fontsize=14, y=1.04)
    plt.tight_layout()
    save(fig, save_name)


# ── 5. Top-K for Crop ──
def plot_topk_crop(metrics, preds, save_name="05_topk_crop.png"):
    modelos = ["Random Forest", "Gradient Boosting", "TabPFN"]
    probs_keys = ["rf_probs_c", "gb_probs_c", "tabpfn_probs_c"]
    preds_keys = ["rf_preds_c", "gb_preds_c", "tabpfn_preds_c"]
    y_test = preds["y_test_c"]
    classes_c = preds.get("classes_c", None)

    k_vals = [1, 2, 3]
    n_test = len(y_test)
    x = np.arange(len(k_vals))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]

    for i, (name, pk) in enumerate(zip(modelos, probs_keys)):
        if pk not in preds:
            continue
        probs = preds[pk]
        topk = []
        for k in k_vals:
            correct = 0
            for j in range(n_test):
                if y_test[j] in np.argsort(probs[j])[-k:]:
                    correct += 1
            topk.append(correct / n_test)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, topk, width, label=name, color=MODEL_COLORS[name],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, topk):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in k_vals])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Crop — Top-k Accuracy", fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, edgecolor="#ddd")
    ax.grid(True, axis="y", alpha=0.25)

    # Table
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    if "Crop" in metrics:
        for name in modelos:
            if name in metrics["Crop"]:
                r = metrics["Crop"][name]
                rows.append([
                    name,
                    f"{r['accuracy']:.4f}",
                    f"{r['f1']:.4f}",
                    f"{r['auc']:.4f}",
                    f"{r['fit_time']:.2f}s",
                    f"{r['pred_time']:.3f}s",
                ])
    col_labels = ["Modelo", "Accuracy", "F1 (Macro)", "ROC-AUC", "Fit (s)", "Predict (s)"]
    table = ax2.table(cellText=rows, colLabels=col_labels, loc="center",
                      cellLoc="center", colWidths=[0.18, 0.14, 0.14, 0.14, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i, name in enumerate(modelos):
        if i < len(rows):
            table[(i + 1, 0)].set_facecolor(MODEL_COLORS[name])
            table[(i + 1, 0)].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax2.set_title("Crop — Resumen de Métricas", fontweight="bold", pad=10, fontsize=12)

    fig.suptitle("Crop Recommendation — Análisis Comparativo", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


# ── 6. Summary table ──
def plot_summary_table(metrics, save_name="06_summary_table.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = []
    for dataset in metrics:
        for name in ["Random Forest", "Gradient Boosting", "TabPFN"]:
            if name in metrics[dataset]:
                r = metrics[dataset][name]
                rows.append([
                    dataset, name,
                    f"{r['accuracy']:.4f}",
                    f"{r['f1']:.4f}",
                    f"{r['auc']:.4f}",
                    f"{r['fit_time']:.2f}s",
                    f"{r['pred_time']:.3f}s",
                ])

    col_labels = ["Dataset", "Modelo", "Accuracy", "F1 (Macro)", "ROC-AUC",
                  "Fit (s)", "Predict (s)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.18, 0.18, 0.12, 0.12, 0.12, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        idx = i + 1
        model = row[1]
        if model in MODEL_COLORS:
            table[(idx, 1)].set_text_props(color=MODEL_COLORS[model], fontweight="bold")
    fig.suptitle("Comparativa Global — Todos los Modelos y Datasets",
                 fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    save(fig, save_name)


# ── 7. Radar chart ──
def plot_radar(metrics, save_name="07_radar.png"):
    datasets, modelos, metricas, metricas_etiqueta, data = _extraer_metricas(metrics)
    N = len(metricas)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for model in modelos:
            if model not in data.get(dataset, {}):
                continue
            vals = [data[dataset][model][m] for m in metricas]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=2, label=model,
                    color=MODEL_COLORS[model])
            ax.fill(angles, vals, alpha=0.1, color=MODEL_COLORS[model])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metricas_etiqueta, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.set_title(dataset, fontweight="bold", pad=20, fontsize=13)
        ax.grid(True)

    fig.suptitle("Gráfico Radial — Métricas por Modelo", fontweight="bold", fontsize=14, y=1.02)
    fig.legend(modelos, loc="lower center", ncol=3, framealpha=0.9, edgecolor="#ddd")
    plt.tight_layout()
    save(fig, save_name)


# ── 8. Per-class AUC for Crop ──
def plot_auc_per_class_crop(metrics, preds, save_name="08_auc_per_class_crop.png"):
    classes_c = preds.get("classes_c", None)
    if classes_c is None:
        print("  (skip auc_per_class_crop: no classes_c)")
        return

    modelos = ["Random Forest", "Gradient Boosting", "TabPFN"]
    probs_keys = ["rf_probs_c", "gb_probs_c", "tabpfn_probs_c"]
    y_test = preds["y_test_c"]

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_bin = le.fit_transform(y_test)
    if len(le.classes_) != len(classes_c):
        classes_c = le.classes_

    n_classes = len(classes_c)
    auc_data = {}
    for name, pk in zip(modelos, probs_keys):
        if pk not in preds:
            continue
        probs = preds[pk]
        if probs.ndim != 2 or probs.shape[1] != n_classes:
            continue
        auc_scores = []
        for cls_idx in range(n_classes):
            y_true = (y_bin == cls_idx).astype(int)
            if len(np.unique(y_true)) < 2:
                auc_scores.append(0.5)
            else:
                auc_scores.append(roc_auc_score(y_true, probs[:, cls_idx]))
        auc_data[name] = auc_scores

    if not auc_data:
        print("  (skip auc_per_class_crop: no data)")
        return

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (name, scores) in enumerate(auc_data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores, width, label=name,
                      color=MODEL_COLORS[name], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes_c, rotation=90, fontsize=8)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Crop — AUC por Clase (One-vs-Rest)", fontweight="bold", pad=12)
    ax.legend(framealpha=0.9, edgecolor="#ddd")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    save(fig, save_name)


# ── 9. Probability distribution for German ──
def plot_prob_distribution_german(preds, save_name="09_prob_distribution_german.png"):
    modelos_prob = [
        ("Random Forest", "rf_probs_g", "#3498db"),
        ("Gradient Boosting", "gb_probs_g", "#e67e22"),
        ("TabPFN", "tabpfn_probs_g", "#27ae60"),
    ]
    y_test = preds["y_test_g"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, key, color) in zip(axes, modelos_prob):
        if key not in preds:
            ax.set_title(f"{name} — sin datos")
            continue
        probs = preds[key]
        if probs.ndim == 2:
            probs = probs[:, 1]

        ax.hist(probs[y_test == 1], bins=20, alpha=0.6, color=color,
                label="Clase 1 (good)", density=True)
        ax.hist(probs[y_test == 0], bins=20, alpha=0.4, color="gray",
                label="Clase 0 (bad)", density=True)
        ax.axvline(0.5, color="black", ls="--", alpha=0.4, label="Umbral 0.5")
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Densidad")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("German Credit — Distribución de Probabilidades", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


# ── 10. Model ranking ──
def plot_model_ranking(metrics, save_name="10_model_ranking.png"):
    datasets, modelos, metricas, metricas_etiqueta, data = _extraer_metricas(metrics)
    bar_width = 0.7

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        modelos_presentes = [m for m in modelos if m in data.get(dataset, {})]
        if not modelos_presentes:
            continue

        scores = {}
        for met, met_lbl in zip(metricas, metricas_etiqueta):
            vals = {m: data[dataset][m][met] for m in modelos_presentes}
            ranked = sorted(vals, key=vals.get)
            for rank, m in enumerate(ranked):
                if m not in scores:
                    scores[m] = []
                scores[m].append(rank + 1)

        x = np.arange(len(metricas_etiqueta))
        for i, m in enumerate(modelos_presentes):
            offset = (i - 1) * bar_width / 3
            ax.bar(x + offset, scores[m], bar_width / 3,
                   color=MODEL_COLORS.get(m, "#888"), edgecolor="white", linewidth=0.5,
                   label=m)

        ax.set_xticks(x)
        ax.set_xticklabels(metricas_etiqueta, fontsize=10)
        ax.set_ylabel("Ranking (1 = mejor)")
        ax.set_title(f"{dataset} — Ranking por Métrica", fontweight="bold", pad=10)
        ax.set_ylim(0, max(len(modelos_presentes), 3) + 0.5)
        ax.invert_yaxis()
        ax.legend(framealpha=0.9, edgecolor="#ddd")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Ranking de Modelos por Métrica", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


# ── 11. Confusion matrices for Crop ──
def plot_crop_confusion_all(preds, save_name="11_crop_confusion_all.png"):
    modelos = [
        ("Random Forest", "rf_preds_c", "Blues"),
        ("Gradient Boosting", "gb_preds_c", "Oranges"),
        ("TabPFN", "tabpfn_preds_c", "Greens"),
    ]
    y_test = preds["y_test_c"]
    classes_c = preds.get("classes_c", None)
    if classes_c is None:
        classes_c = sorted(set(y_test))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    n_models = sum(1 for _, key, _ in modelos if key in preds)
    if n_models == 0:
        print("  (skip crop_confusion_all: no data)")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 6, 5.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, key, cmap) in zip(axes, modelos):
        if key not in preds:
            continue
        cm = confusion_matrix(y_test, preds[key], labels=classes_c)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes_c)
        disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticklabels(classes_c, rotation=90, fontsize=6)
        ax.set_yticklabels(classes_c, fontsize=6)

    fig.suptitle("Crop Recommendation — Matrices de Confusión (22 clases)",
                 fontweight="bold", fontsize=14, y=1.04)
    plt.tight_layout()
    save(fig, save_name)


# ── 12. German metrics detail (multi-panel) ──
def plot_german_detail(metrics, preds, save_name="12_german_detail.png"):
    if "German Credit" not in metrics:
        print("  (skip german_detail: no data)")
        return

    modelos = ["Random Forest", "Gradient Boosting", "TabPFN"]
    from sklearn.metrics import confusion_matrix

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Metrics bars
    ax_metrics = fig.add_subplot(gs[0, :])
    met_names = ["Accuracy", "F1-Score", "ROC-AUC"]
    x = np.arange(len(met_names))
    width = 0.22

    for i, model in enumerate(modelos):
        if model not in metrics["German Credit"]:
            continue
        offset = (i - 1) * width
        vals = [metrics["German Credit"][model][m] for m in ["accuracy", "f1", "auc"]]
        bars = ax_metrics.bar(x + offset, vals, width, label=model,
                              color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax_metrics.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(met_names)
    ax_metrics.set_ylim(0, 1.12)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Métricas Principales", fontweight="bold")
    ax_metrics.legend(framealpha=0.9, edgecolor="#ddd", fontsize=9)
    ax_metrics.grid(True, axis="y", alpha=0.25)

    # Row 2: Confusion matrices
    cm_configs = [
        ("Random Forest", "rf_preds_g", "Blues", gs[1, 0]),
        ("Gradient Boosting", "gb_preds_g", "Oranges", gs[1, 1]),
        ("TabPFN", "tabpfn_preds_g", "Greens", gs[1, 2]),
    ]
    y_test = preds["y_test_g"]
    for name, key, cmap, pos in cm_configs:
        ax = fig.add_subplot(pos)
        if key not in preds:
            ax.set_title(f"{name} — sin datos")
            continue
        cm = confusion_matrix(y_test, preds[key])
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(cm, display_labels=["good", "bad"])
        disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    fig.suptitle("German Credit — Detalle Completo", fontweight="bold", fontsize=15, y=1.01)
    save(fig, save_name)


# ── 13. Crop metrics detail (multi-panel) ──
def plot_crop_detail(metrics, preds, save_name="13_crop_detail.png"):
    if "Crop" not in metrics:
        print("  (skip crop_detail: no data)")
        return

    modelos = ["Random Forest", "Gradient Boosting", "TabPFN"]
    classes_c = preds.get("classes_c", None)
    if classes_c is None:
        classes_c = sorted(set(preds["y_test_c"]))

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Metrics bars
    ax_metrics = fig.add_subplot(gs[0, :])
    met_names = ["Accuracy", "F1 (Macro)", "ROC-AUC"]
    x = np.arange(len(met_names))
    width = 0.22

    for i, model in enumerate(modelos):
        if model not in metrics["Crop"]:
            continue
        offset = (i - 1) * width
        vals = [metrics["Crop"][model][m] for m in ["accuracy", "f1", "auc"]]
        bars = ax_metrics.bar(x + offset, vals, width, label=model,
                              color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax_metrics.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(met_names)
    ax_metrics.set_ylim(0, 1.12)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Métricas Principales", fontweight="bold")
    ax_metrics.legend(framealpha=0.9, edgecolor="#ddd", fontsize=9)
    ax_metrics.grid(True, axis="y", alpha=0.25)

    # Row 2: Top-k bars
    probs_keys = ["rf_probs_c", "gb_probs_c", "tabpfn_probs_c"]
    y_test = preds["y_test_c"]
    k_vals = [1, 2, 3]
    n_test = len(y_test)

    for idx, (model, pk) in enumerate(zip(modelos, probs_keys)):
        ax = fig.add_subplot(gs[1, idx])
        if pk not in preds:
            ax.set_title(f"{model} — sin datos")
            continue
        probs = preds[pk]
        topk = []
        for k in k_vals:
            correct = 0
            for j in range(n_test):
                if y_test[j] in np.argsort(probs[j])[-k:]:
                    correct += 1
            topk.append(correct / n_test)
        bars = ax.bar(k_vals, topk, color=MODEL_COLORS[model], width=0.5,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, topk):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"Top-{k}" for k in k_vals])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model} — Top-k", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Crop Recommendation — Detalle Completo", fontweight="bold", fontsize=15, y=1.01)
    save(fig, save_name)


# ── 14. Scatter: Accuracy vs Time ──
def plot_accuracy_vs_time(metrics, save_name="14_accuracy_vs_time.png"):
    datasets, modelos, _, _, data = _extraer_metricas(metrics)

    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in datasets:
        for model in modelos:
            if model not in data.get(ds, {}):
                continue
            acc = data[ds][model]["accuracy"]
            total_time = data[ds][model]["fit_time"] + data[ds][model]["pred_time"]
            marker = "o" if ds == datasets[0] else "s"
            ax.scatter(total_time, acc, s=120, marker=marker,
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=0.8,
                       zorder=5)
            ax.annotate(f"{ds[:5]} {model[:2]}", (total_time, acc),
                        xytext=(5, 5), textcoords="offset points", fontsize=7,
                        alpha=0.8)

    ax.set_xlabel("Tiempo Total (Fit + Predict) [s]")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Tiempo de Ejecución", fontweight="bold", pad=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555', label='German Credit'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555', label='Crop'),
    ]
    for model in modelos:
        legend_elements.append(Patch(facecolor=MODEL_COLORS[model], label=model))
    ax.legend(handles=legend_elements, framealpha=0.9, edgecolor="#ddd", fontsize=9)
    plt.tight_layout()
    save(fig, save_name)


# ── Main ──
def main():
    print("=" * 55)
    print("  COMPARATIVA: Random Forest vs Gradient Boosting vs TabPFN")
    print("=" * 55)

    # ── Cargar resultados ──
    print("\n── Cargando resultados...")
    metrics = cargar_metricas()
    preds = cargar_predicciones()
    print(f"  Datasets: {list(metrics.keys())}")
    for ds in metrics:
        print(f"    {ds}: {list(metrics[ds].keys())}")

    # ── Generar gráficas ──
    print(f"\n── Generando gráficas → {OUTPUT_DIR}/ ──")

    print("\n  1/14 — Curvas ROC (German Credit)")
    plot_roc_german(preds)

    print("  2/14 — Comparación de métricas")
    plot_metrics_comparison(metrics)

    print("  3/14 — Comparación de tiempos")
    plot_timing_comparison(metrics)

    print("  4/14 — Matrices de confusión (German Credit)")
    plot_confusion_german(preds)

    print("  5/14 — Top-k Accuracy (Crop)")
    plot_topk_crop(metrics, preds)

    print("  6/14 — Tabla resumen global")
    plot_summary_table(metrics)

    print("  7/14 — Gráfico radial")
    plot_radar(metrics)

    print("  8/14 — AUC por clase (Crop)")
    plot_auc_per_class_crop(metrics, preds)

    print("  9/14 — Distribución de probabilidades (German)")
    plot_prob_distribution_german(preds)

    print("  10/14 — Ranking de modelos")
    plot_model_ranking(metrics)

    print("  11/14 — Matrices de confusión (Crop)")
    plot_crop_confusion_all(preds)

    print("  12/14 — Detalle German Credit")
    plot_german_detail(metrics, preds)

    print("  13/14 — Detalle Crop Recommendation")
    plot_crop_detail(metrics, preds)

    print("  14/14 — Accuracy vs Tiempo")
    plot_accuracy_vs_time(metrics)

    # ── Exportar CSV ──
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "resultados_metricas.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Modelo", "Accuracy", "F1", "ROC-AUC", "Fit (s)", "Predict (s)"])
        for ds in metrics:
            for model in ["Random Forest", "Gradient Boosting", "TabPFN"]:
                if model in metrics[ds]:
                    r = metrics[ds][model]
                    w.writerow([ds, model, f"{r['accuracy']:.4f}", f"{r['f1']:.4f}",
                                f"{r['auc']:.4f}", f"{r['fit_time']:.3f}", f"{r['pred_time']:.5f}"])
    print(f"\n  → {csv_path}")

    n = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
    print(f"\n✅ Listo. {n} gráficas generadas en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
