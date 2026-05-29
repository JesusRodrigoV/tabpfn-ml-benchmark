#!/usr/bin/env python3
"""Modifica ambos notebooks para eliminar kagglehub, alinear test_size y exportar resultados."""
import json, re, copy

# ── Config ──────────────────────────────────────────────────────────────
EXPORT_CELL_NB1 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "exportar_resultados",
    "metadata": {},
    "outputs": [],
    "source": [
        'import json, os\n',
        'import numpy as np\n',
        'from sklearn.metrics import accuracy_score, f1_score\n',
        'os.makedirs("resultados", exist_ok=True)\n',
        '\n',
        '# ── German Credit: predicciones para todos los modelos ──\n',
        'rf_preds_g = rf_search_g.predict(X_test_g)\n',
        'gb_preds_g = gb_search_g.predict(X_test_g)\n',
        'rf_acc_g = accuracy_score(y_test_g, rf_preds_g)\n',
        'gb_acc_g = accuracy_score(y_test_g, gb_preds_g)\n',
        'rf_f1_g = f1_score(y_test_g, rf_preds_g)\n',
        'gb_f1_g = f1_score(y_test_g, gb_preds_g)\n',
        '\n',
        '# ── Crop: predicciones para todos los modelos ──\n',
        'rf_preds_c = rf_search_c.predict(X_test_c)\n',
        'gb_preds_c = gb_search_c.predict(X_test_c)\n',
        'rf_acc_c = accuracy_score(y_test_c, rf_preds_c)\n',
        'gb_acc_c = accuracy_score(y_test_c, gb_preds_c)\n',
        'rf_f1_c = f1_score(y_test_c, rf_preds_c, average="macro")\n',
        'gb_f1_c = f1_score(y_test_c, gb_preds_c, average="macro")\n',
        'rf_auc_c_val = roc_auc_score(y_test_c, rf_probs_c, multi_class="ovr", average="macro")\n',
        'gb_auc_c_val = roc_auc_score(y_test_c, gb_probs_c, multi_class="ovr", average="macro")\n',
        '\n',
        '# ── Métricas escalares ──\n',
        'metrics = {\n',
        '    "German Credit": {\n',
        '        "Random Forest": {\n',
        '            "accuracy": float(rf_acc_g), "f1": float(rf_f1_g), "auc": float(rf_auc_g),\n',
        '            "fit_time": time_rf_fit_g, "pred_time": time_rf_pred_g\n',
        '        },\n',
        '        "Gradient Boosting": {\n',
        '            "accuracy": float(gb_acc_g), "f1": float(gb_f1_g), "auc": float(gb_auc_g),\n',
        '            "fit_time": time_gb_fit_g, "pred_time": time_gb_pred_g\n',
        '        },\n',
        '    },\n',
        '    "Crop": {\n',
        '        "Random Forest": {\n',
        '            "accuracy": float(rf_acc_c), "f1": float(rf_f1_c), "auc": float(rf_auc_c_val),\n',
        '            "fit_time": time_rf_fit_c, "pred_time": time_rf_pred_c\n',
        '        },\n',
        '        "Gradient Boosting": {\n',
        '            "accuracy": float(gb_acc_c), "f1": float(gb_f1_c), "auc": float(gb_auc_c_val),\n',
        '            "fit_time": time_gb_fit_c, "pred_time": time_gb_pred_c\n',
        '        },\n',
        '    },\n',
        '}\n',
        'with open("resultados/metrics_notebook1.json", "w") as f:\n',
        '    json.dump(metrics, f, indent=2)\n',
        '\n',
        '# ── Predicciones para gráficas ──\n',
        'np.savez_compressed("resultados/preds_notebook1.npz",\n',
        '    y_test_g=y_test_g.values.astype(int),\n',
        '    rf_probs_g=np.array(rf_probs_g, dtype=np.float64),\n',
        '    rf_preds_g=np.array(rf_preds_g, dtype=int),\n',
        '    gb_probs_g=np.array(gb_probs_g, dtype=np.float64),\n',
        '    gb_preds_g=np.array(gb_preds_g, dtype=int),\n',
        '    y_test_c=np.array(y_test_c, dtype=object),\n',
        '    rf_probs_c=np.array(rf_probs_c, dtype=np.float64),\n',
        '    rf_preds_c=np.array(rf_preds_c, dtype=object),\n',
        '    gb_probs_c=np.array(gb_probs_c, dtype=np.float64),\n',
        '    gb_preds_c=np.array(gb_preds_c, dtype=object),\n',
        ')\n',
        'print("✅ Resultados notebook1 exportados a resultados/")\n',
    ]
}

EXPORT_CELL_NB2 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "exportar_resultados",
    "metadata": {},
    "outputs": [],
    "source": [
        'import json, os\n',
        'import numpy as np\n',
        'os.makedirs("resultados", exist_ok=True)\n',
        '\n',
        '# ── Métricas ──\n',
        'acc_g = accuracy_score(y_test, y_pred_german)\n',
        'f1_g = f1_score(y_test, y_pred_german)\n',
        'auc_g = roc_auc_score(y_test, y_prob_german[:, 1])\n',
        '\n',
        'acc_c = accuracy_score(y_test_c, y_pred_crop)\n',
        'f1_c = f1_score(y_test_c, y_pred_crop, average="macro")\n',
        'auc_c = roc_auc_score(y_test_c, y_prob_crop, multi_class="ovr", average="macro")\n',
        '\n',
        'metrics = {\n',
        '    "German Credit": {\n',
        '        "TabPFN": {\n',
        '            "accuracy": float(acc_g), "f1": float(f1_g), "auc": float(auc_g),\n',
        '            "fit_time": t_fit_german, "pred_time": t_pred_german\n',
        '        },\n',
        '    },\n',
        '    "Crop": {\n',
        '        "TabPFN": {\n',
        '            "accuracy": float(acc_c), "f1": float(f1_c), "auc": float(auc_c),\n',
        '            "fit_time": t_fit_crop, "pred_time": t_pred_crop\n',
        '        },\n',
        '    },\n',
        '}\n',
        '\n',
        '# Combinar con notebook1 si existe\n',
        'if os.path.exists("resultados/metrics_notebook1.json"):\n',
        '    with open("resultados/metrics_notebook1.json") as f:\n',
        '        nb1 = json.load(f)\n',
        '    for ds in nb1:\n',
        '        if ds not in metrics:\n',
        '            metrics[ds] = {}\n',
        '        for model, m in nb1[ds].items():\n',
        '            metrics[ds][model] = m\n',
        '\n',
        'with open("resultados/metrics.json", "w") as f:\n',
        '    json.dump(metrics, f, indent=2)\n',
        '\n',
        '# ── Predicciones ──\n',
        '# Cargar predicciones de notebook1\n',
        'preds_nb1 = {}\n',
        'if os.path.exists("resultados/preds_notebook1.npz"):\n',
        '    preds_nb1 = dict(np.load("resultados/preds_notebook1.npz"))\n',
        '\n',
        'save_dict = dict(preds_nb1)\n',
        'save_dict.update({\n',
        '    "tabpfn_probs_g": np.array(y_prob_german, dtype=np.float64),\n',
        '    "tabpfn_preds_g": np.array(y_pred_german, dtype=int),\n',
        '    "tabpfn_probs_c": np.array(y_prob_crop, dtype=np.float64),\n',
        '    "tabpfn_preds_c": np.array(y_pred_crop, dtype=object),\n',
        '    "classes_c": le_crop.classes_,\n',
        '})\n',
        '# Si notebook1 no puso y_test, ponerlos\n',
        'if "y_test_g" not in save_dict:\n',
        '    save_dict["y_test_g"] = np.array(y_test, dtype=int)\n',
        'if "y_test_c" not in save_dict:\n',
        '    save_dict["y_test_c"] = np.array(y_test_c, dtype=object)\n',
        '\n',
        'np.savez_compressed("resultados/preds.npz", **save_dict)\n',
        'print("✅ Resultados combinados exportados a resultados/")\n',
        '\n',
        '# Limpiar archivos temporales de notebook1\n',
        'for f in ["metrics_notebook1.json", "preds_notebook1.npz"]:\n',
        '    p = os.path.join("resultados", f)\n',
        '    if os.path.exists(p):\n',
        '        os.remove(p)\n',
    ]
}

# ── Helper ──────────────────────────────────────────────────────────────
NB_LOAD_DATASET_LOCAL = [
    'def load_dataset(slug, filename, index_col=None, local_fallback=None):\n',
    '    path = local_fallback or filename\n',
    '    if os.path.exists(path):\n',
    '        df = pd.read_csv(path, index_col=index_col)\n',
    '        print(f"Cargado desde local: {path}")\n',
    '    else:\n',
    '        raise FileNotFoundError(f"No se encontró {path}")\n',
    '    print(f"Dimensiones: {df.shape}")\n',
    '    return df\n',
]

NB2_LOAD_LOCAL = [
    'def descargar_y_guardar_dataset(kaggle_id, nombre_archivo_local):\n',
    '    """Carga dataset desde CSV local con respaldo a datasets_respaldo."""\n',
    '    os.makedirs("datasets_respaldo", exist_ok=True)\n',
    '    ruta_respaldo = os.path.join("datasets_respaldo", nombre_archivo_local)\n',
    '    if os.path.exists(nombre_archivo_local):\n',
    '        print(f"Cargando desde local: {nombre_archivo_local}")\n',
    '        return pd.read_csv(nombre_archivo_local)\n',
    '    if os.path.exists(ruta_respaldo):\n',
    '        print(f"Cargando desde respaldo: {ruta_respaldo}")\n',
    '        return pd.read_csv(ruta_respaldo)\n',
    '    raise FileNotFoundError(\n',
    '        f"No se encontró {nombre_archivo_local} ni {ruta_respaldo}. "\n',
    '        "Descargalo de Kaggle manualmente."\n',
    '    )\n',
    '\n',
    'def cargar_german_credit():\n',
    '    """Carga German Credit desde CSV local."""\n',
    '    os.makedirs("datasets_respaldo", exist_ok=True)\n',
    '    ruta_respaldo = os.path.join("datasets_respaldo", "german_credit_with_risk.csv")\n',
    '    rutas = ["german_credit_data.csv", ruta_respaldo]\n',
    '    for r in rutas:\n',
    '        if os.path.exists(r):\n',
    '            print(f"Cargando German Credit desde: {r}")\n',
    '            return pd.read_csv(r)\n',
    '    raise FileNotFoundError(\n',
    '        "No se encontró german_credit_data.csv. Descargalo de Kaggle manualmente."\n',
    '    )\n',
]


def find_code_cell(cells, marker):
    """Find first code cell containing marker text."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if marker in src:
                return i, cell
    return None, None


def modify_notebook1():
    with open("notebook1_baseline.ipynb") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # 1. Remove kagglehub import
    for cell in cells:
        if cell["cell_type"] == "code":
            src_lines = cell["source"]
            new_lines = [l for l in src_lines if "import kagglehub" not in l]
            if len(new_lines) < len(src_lines):
                cell["source"] = new_lines
                print("  ✓ Removed kagglehub import")

    # 2. Change TEST_SIZE = 0.2 -> 0.3
    for cell in cells:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "TEST_SIZE" in src and "0.2" in src:
                cell["source"] = [s.replace("0.2", "0.3") for s in cell["source"]]
                print("  ✓ Changed TEST_SIZE to 0.3")
                break

    # 3. Replace load_dataset function
    idx, _ = find_code_cell(cells, "def load_dataset")
    if idx is not None:
        cells[idx]["source"] = NB_LOAD_DATASET_LOCAL
        print("  ✓ Replaced load_dataset with local version")

    # 4. Add export cell at the end (before the last markdown)
    cells.append(copy.deepcopy(EXPORT_CELL_NB1))
    print("  ✓ Added export cell")

    with open("notebook1_baseline.ipynb", "w") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("✅ notebook1_baseline.ipynb modified successfully")


def modify_notebook2():
    with open("notebook2mio.ipynb") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # 1. Remove kagglehub imports
    for cell in cells:
        if cell["cell_type"] == "code":
            src_lines = cell["source"]
            new_lines = [
                l for l in src_lines
                if "import kagglehub" not in l
                and "from kagglehub" not in l
            ]
            if len(new_lines) < len(src_lines):
                cell["source"] = new_lines
                print("  ✓ Removed kagglehub imports")

    # 2. Replace descargar_y_guardar_dataset
    idx, _ = find_code_cell(cells, "def descargar_y_guardar_dataset")
    if idx is not None:
        cells[idx]["source"] = NB2_LOAD_LOCAL
        print("  ✓ Replaced descargar_y_guardar_dataset")

    # 3. Add export cell at the end
    cells.append(copy.deepcopy(EXPORT_CELL_NB2))
    print("  ✓ Added export cell")

    with open("notebook2mio.ipynb", "w") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("✅ notebook2mio.ipynb modified successfully")


if __name__ == "__main__":
    print("── Modificando notebooks ──")
    modify_notebook1()
    modify_notebook2()
    print("\n✅ Ambos notebooks listos. Ahora ejecutalos en orden:")
    print("  1. notebook1_baseline.ipynb  (RF + GB)")
    print("  2. notebook2mio.ipynb        (TabPFN)")
    print("  3. python comparativa.py     (gráficas)")
