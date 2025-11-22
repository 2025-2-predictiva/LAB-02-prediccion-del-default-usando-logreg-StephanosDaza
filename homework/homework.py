# flake8: noqa: E501

import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



def get_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return project_root


def _leer_y_limpiar(ruta_relativa: str) -> pd.DataFrame:
    project_root = get_paths()
    ruta_completa = os.path.join(project_root, ruta_relativa.replace("/", os.sep))
    
    df = pd.read_csv(ruta_completa, compression="zip").copy()

    df.rename(columns={"default payment next month": "default"}, inplace=True)
    
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Limpieza de Datos
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v > 4 else v)

    return df.dropna()


def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["default"]), df["default"]


def _pipeline_y_busqueda(n_cols_raw: int) -> GridSearchCV:
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Definición inicial del preprocesador (se actualiza en main)
    prepro = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), [])  
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)

    pipe = Pipeline(
        steps=[
            ("prep", prepro),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("clf", clf),
        ]
    )

    # GridSearch Amplio
    grid = {
        "kbest__k": list(range(1, n_cols_raw + 1)),
        "clf__C": [0.1, 1, 10],
        "clf__solver": ["liblinear", "lbfgs"],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )
    return gs


def _metricas(nombre: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def _matriz_conf(nombre: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def main() -> None:
    
    df_train = _leer_y_limpiar("files/input/train_data.csv.zip")
    df_test = _leer_y_limpiar("files/input/test_data.csv.zip")

    X_tr, y_tr = _xy(df_train)
    X_te, y_te = _xy(df_test)

    buscador = _pipeline_y_busqueda(n_cols_raw=X_tr.shape[1])

    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    # Actualizamos el preprocesador con las columnas numéricas correctas
    buscador.estimator.named_steps["prep"] = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    buscador.fit(X_tr, y_tr)

    # Guardar modelo
    project_root = get_paths()
    models_dir = os.path.join(project_root, "files", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    with gzip.open(os.path.join(models_dir, "model.pkl.gz"), "wb") as f:
        pickle.dump(buscador, f)
    
    print(f"Modelo guardado en: {os.path.join(models_dir, 'model.pkl.gz')}")

    # Guardar métricas
    pred_tr = buscador.predict(X_tr)
    pred_te = buscador.predict(X_te)

    resultados: List[dict] = [
        _metricas("train", y_tr, pred_tr),
        _metricas("test", y_te, pred_te),
        _matriz_conf("train", y_tr, pred_tr),
        _matriz_conf("test", y_te, pred_te),
    ]

    output_dir = os.path.join(project_root, "files", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        for r in resultados:
            fh.write(json.dumps(r) + "\n")
            
    print("metrics.json generado correctamente.")


if __name__ == "__main__":
    main()