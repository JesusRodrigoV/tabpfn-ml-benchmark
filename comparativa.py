#!/usr/bin/env python3
"""
comparativa.py — Genera gráficas comparativas RF vs GB vs TabPFN para el informe.

Uso:
    python comparativa.py

Requiere: pip install kagglehub tabpfn
"""

import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from tabpfn import TabPFNClassifier

# ── Config ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_iPZbyLLnlUZKB1kKSHFg2Qu62IHXdl_NdvX8bGbL6V8"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

MODEL_COLORS = {
    "Random Forest": "#3498db",
    "Gradient Boosting": "#e67e22",
    "TabPFN": "#27ae60",
}
MODEL_MARKERS = ["o", "s", "D"]
OUTPUT_DIR = "figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ── Carga de datos ──────────────────────────────────────────────────────
def load_dataset(slug, filename, index_col=None, local_fallback=None):
    try:
        path = kagglehub.dataset_download(slug)
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath, index_col=index_col)
        print(f"  ✓ {slug}")
    except Exception as e:
        print(f"  kagglehub falló ({e}), usando local...")
        if local_fallback and os.path.exists(local_fallback):
            df = pd.read_csv(local_fallback, index_col=index_col)
        else:
            raise FileNotFoundError(f"No se encontró {local_fallback}")
    return df


def load_german_credit():
    """Carga German Credit con Risk target."""
    os.makedirs("datasets_respaldo", exist_ok=True)
    ruta = "datasets_respaldo/german_credit_with_risk.csv"
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "kabure/german-credit-data-with-risk",
            "",
        )
        df.to_csv(ruta, index=False)
        print("  ✓ kabure/german-credit-data-with-risk")
        return df
    except Exception as e:
        print(f"  kagglehub falló ({e}), usando local...")
        if os.path.exists(ruta):
            return pd.read_csv(ruta)
        raise


# ── Preprocesamiento ────────────────────────────────────────────────────
def preprocess_german(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    X = df.drop(columns=["Risk"])
    y = df["Risk"].map({"good": 1, "bad": 0})
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.select_dtypes(include=["object", "category"]).columns.tolist()]
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_t = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([("num", num_t, num), ("cat", cat_t, cat)], verbose_feature_names_out=False)
    return X, y, preprocessor


def preprocess_crop(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    X = df.drop(columns=["label"])
    y = df["label"]
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    preprocessor = ColumnTransformer([("num", num_t, num)], verbose_feature_names_out=False)
    return X, y, preprocessor


# ── Modelos ─────────────────────────────────────────────────────────────
def train_evaluate(X_train, y_train, X_test, y_test, preprocessor, multiclass=False):
    results = {}
    scoring = "roc_auc_ovr" if multiclass else "roc_auc"

    # --- Random Forest ---
    pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t_fit = time.time() - t0
    t0 = time.time()
    probs = pipe.predict_proba(X_test)
    preds = pipe.predict(X_test)
    t_pred = time.time() - t0
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, probs, multi_class="ovr" if multiclass else "raise")
    if not multiclass:
        auc = roc_auc_score(y_test, probs[:, 1])
    results["Random Forest"] = {"model": pipe, "probs": probs, "preds": preds,
                                "acc": acc, "f1": f1, "auc": auc,
                                "fit_time": t_fit, "pred_time": t_pred}

    # --- Gradient Boosting ---
    pipe = Pipeline([("prep", preprocessor), ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t_fit = time.time() - t0
    t0 = time.time()
    probs = pipe.predict_proba(X_test)
    preds = pipe.predict(X_test)
    t_pred = time.time() - t0
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    if not multiclass:
        auc = roc_auc_score(y_test, probs[:, 1])
    else:
        auc = roc_auc_score(y_test, probs, multi_class="ovr")
    results["Gradient Boosting"] = {"model": pipe, "probs": probs, "preds": preds,
                                    "acc": acc, "f1": f1, "auc": auc,
                                    "fit_time": t_fit, "pred_time": t_pred}

    # --- TabPFN ---
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    # Preprocess to numeric for TabPFN
    X_train_num = preprocessor.fit_transform(X_train)
    X_test_num = preprocessor.transform(X_test)
    # Ensure all numeric (TabPFN doesn't handle object dtypes)
    X_train_num = np.array(X_train_num, dtype=np.float64)
    X_test_num = np.array(X_test_num, dtype=np.float64)

    t0 = time.time()
    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_train_num, y_train_enc)
    t_fit = time.time() - t0
    t0 = time.time()
    probs = clf.predict_proba(X_test_num)
    preds = clf.predict(X_test_num)
    t_pred = time.time() - t0
    preds_labels = le.inverse_transform(preds)
    acc = accuracy_score(y_test_enc, preds)
    f1 = f1_score(y_test_enc, preds, average="macro")
    if not multiclass:
        auc = roc_auc_score(y_test_enc, probs[:, 1])
    else:
        auc = roc_auc_score(y_test_enc, probs, multi_class="ovr")

    results["TabPFN"] = {"model": clf, "probs": probs, "preds": preds,
                         "acc": acc, "f1": f1, "auc": auc,
                         "fit_time": t_fit, "pred_time": t_pred}

    return results, le


# ── Plot helpers ────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  → {path}")
    plt.close(fig)


def plot_roc_german(results, y_test, save_name="01_roc_german.png"):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, r in results.items():
        if r["probs"].shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_test, r["probs"][:, 1])
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2.2,
                    label=f"{name} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Clasificador aleatorio")
    ax.fill_between([0], [0], [1], alpha=0)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("German Credit — Curvas ROC", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#ddd")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    save(fig, save_name)


def plot_metrics_comparison(all_metrics, save_name="02_metrics_comparison.png"):
    datasets = list(all_metrics.keys())
    metrics_names = ["Accuracy", "F1-Score", "ROC-AUC"]
    n_metrics = len(metrics_names)
    n_models = 3
    x = np.arange(n_metrics)
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for i, (model, color) in enumerate(MODEL_COLORS.items()):
            offset = (i - 1) * width
            vals = [all_metrics[dataset][model][m] for m in metrics_names]
            bars = ax.bar(x + offset, vals, width, label=model if idx == 1 else "",
                          color=color, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(dataset, fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.25)
        if idx == 1:
            ax.legend(framealpha=0.9, edgecolor="#ddd")

    fig.suptitle("Comparación de Métricas Predictivas", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


def plot_timing_comparison(all_metrics, save_name="03_timing_comparison.png"):
    datasets = list(all_metrics.keys())
    phases = ["fit_time", "pred_time"]
    phase_labels = ["Fit", "Predict"]
    n_models = 3
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for pidx, (phase, plabel) in enumerate(zip(phases, phase_labels)):
        ax = axes[pidx]
        width = 0.22
        for i, (model, color) in enumerate(MODEL_COLORS.items()):
            offset = (i - 1) * width
            vals = [all_metrics[d][model][phase] for d in datasets]
            bars = ax.bar(x + offset, vals, width, label=model if pidx == 1 else "",
                          color=color, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                label = f"{v:.2f}s" if v >= 0.1 else f"{v*1000:.0f}ms"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        label, ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel("Tiempo (s)")
        ax.set_title(f"{plabel} — Tiempo de Ejecución", fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.25)
        if pidx == 1:
            ax.legend(framealpha=0.9, edgecolor="#ddd")

    fig.suptitle("Comparación de Tiempos de Ejecución", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


def plot_confusion_german(results, y_test, le, save_name="04_confusion_german.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["good", "bad"])
        disp.plot(ax=ax, cmap="Blues" if "Forest" in name else
                  ("Oranges" if "Boosting" in name else "Greens"),
                  values_format="d", colorbar=False)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    fig.suptitle("German Credit — Matrices de Confusión", fontweight="bold", fontsize=14, y=1.04)
    plt.tight_layout()
    save(fig, save_name)


def plot_topk_crop(all_metrics, y_test, results_tabpfn, le_crop, save_name="05_topk_crop.png"):
    """Top-k accuracy para Crop + timing + tabla resumen."""
    # Compute top-k for all models from results
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]

    k_vals = [1, 2, 3]
    n_test = len(y_test)
    x = np.arange(len(k_vals))
    width = 0.22

    for i, (name, color) in enumerate(MODEL_COLORS.items()):
        probs = all_metrics["Crop"][name]["probs"]
        topk = []
        for k in k_vals:
            correct = 0
            for j in range(n_test):
                if y_test[j] in np.argsort(probs[j])[-k:]:
                    correct += 1
            topk.append(correct / n_test)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, topk, width, label=name, color=color,
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

    # Table with all metrics for Crop
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for name in MODEL_COLORS:
        r = all_metrics["Crop"][name]
        rows.append([name, f"{r['acc']:.4f}", f"{r['f1']:.4f}",
                     f"{r['auc']:.4f}",
                     f"{r['fit_time']:.2f}s", f"{r['pred_time']:.3f}s"])
    col_labels = ["Modelo", "Accuracy", "F1 (Macro)", "ROC-AUC", "Fit (s)", "Predict (s)"]
    table = ax2.table(cellText=rows, colLabels=col_labels, loc="center",
                      cellLoc="center", colWidths=[0.18, 0.14, 0.14, 0.14, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i, name in enumerate(MODEL_COLORS):
        table[(i + 1, 0)].set_facecolor(MODEL_COLORS[name])
        table[(i + 1, 0)].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax2.set_title("Crop — Resumen de Métricas", fontweight="bold", pad=10, fontsize=12)

    fig.suptitle("Crop Recommendation — Análisis Comparativo", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, save_name)


def plot_summary_table(all_metrics, save_name="06_summary_table.png"):
    """Tabla resumen global con formato de publicación."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = []
    for dataset in all_metrics:
        for name in MODEL_COLORS:
            r = all_metrics[dataset][name]
            rows.append([dataset, name, f"{r['acc']:.4f}", f"{r['f1']:.4f}",
                         f"{r['auc']:.4f}",
                         f"{r['fit_time']:.2f}s", f"{r['pred_time']:.3f}s"])

    col_labels = ["Dataset", "Modelo", "Accuracy", "F1 (Macro)", "ROC-AUC",
                  "Fit (s)", "Predict (s)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.18, 0.18, 0.12, 0.12, 0.12, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    # Color model names
    for i, row in enumerate(rows):
        idx = i + 1
        model = row[1]
        if model in MODEL_COLORS:
            table[(idx, 1)].set_text_props(color=MODEL_COLORS[model], fontweight="bold")
    fig.suptitle("Comparativa Global — Todos los Modelos y Datasets",
                 fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    save(fig, save_name)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  COMPARATIVA: Random Forest vs Gradient Boosting vs TabPFN")
    print("=" * 55)

    # ── 1. German Credit ────────────────────────────────────────────────
    print("\n── German Credit ──")
    print("Cargando datos...")
    df_g = load_german_credit()
    X_g, y_g, preproc_g = preprocess_german(df_g)
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X_g, y_g, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_g
    )
    print(f"  Train: {X_train_g.shape[0]} | Test: {X_test_g.shape[0]}")
    print("Entrenando modelos...")
    results_g, le_g = train_evaluate(X_train_g, y_train_g, X_test_g, y_test_g, preproc_g, multiclass=False)
    for name, r in results_g.items():
        print(f"  {name:.<20} ACC={r['acc']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}  "
              f"Fit={r['fit_time']:.2f}s  Pred={r['pred_time']:.3f}s")

    # ── 2. Crop Recommendation ─────────────────────────────────────────
    print("\n── Crop Recommendation ──")
    print("Cargando datos...")
    df_c = load_dataset("atharvaingle/crop-recommendation-dataset",
                        "Crop_recommendation.csv",
                        local_fallback="Crop_recommendation.csv")
    X_c, y_c, preproc_c = preprocess_crop(df_c)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_c
    )
    print(f"  Train: {X_train_c.shape[0]} | Test: {X_test_c.shape[0]}")
    print("Entrenando modelos...")
    le_c = LabelEncoder()
    y_train_c_enc = le_c.fit_transform(y_train_c)
    y_test_c_enc = le_c.transform(y_test_c)
    results_c, _ = train_evaluate(X_train_c, y_train_c_enc, X_test_c, y_test_c_enc, preproc_c, multiclass=True)
    for name, r in results_c.items():
        print(f"  {name:.<20} ACC={r['acc']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}  "
              f"Fit={r['fit_time']:.2f}s  Pred={r['pred_time']:.3f}s")

    all_metrics = {
        "German Credit": results_g,
        "Crop": results_c,
    }

    # ── 3. Generate plots ──────────────────────────────────────────────
    print(f"\n── Generando gráficas → {OUTPUT_DIR}/ ──")
    plot_roc_german(results_g, y_test_g)
    plot_metrics_comparison(all_metrics)
    plot_timing_comparison(all_metrics)
    plot_confusion_german(results_g, y_test_g, le_g)
    plot_topk_crop(all_metrics, y_test_c_enc, results_c.get("TabPFN"), le_c)
    plot_summary_table(all_metrics)

    print(f"\n✅ Listo. {len(os.listdir(OUTPUT_DIR))} gráficas en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
