from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Copper" / "Data" / "Silver" / "serie_a_ready.parquet"
OUTPUT_DIR = BASE_DIR / "Copper"

TARGETS = {
    "yellow": "CrdY",
    "red": "CrdR",
}


def build_feature_list(df: pd.DataFrame) -> List[str]:
    excluded_exact = {"Rk", "Player", "Nation", "Comp"}
    excluded_fragments = ["CrdY", "CrdR", "2CrdY"]

    features: List[str] = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if any(fragment in col for fragment in excluded_fragments):
            continue
        features.append(col)
    return features


def split_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", clf)])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_pred = model.predict(x_test)
    metrics: Dict[str, object] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "support": int(len(y_test)),
    }

    if len(np.unique(y_test)) > 1:
        y_prob = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_values = np.zeros_like(precision)
        valid_denominator = (precision + recall) > 0
        f1_values[valid_denominator] = (2 * precision[valid_denominator] * recall[valid_denominator]) / (
            precision[valid_denominator] + recall[valid_denominator]
        )

        best_idx = int(np.argmax(f1_values))
        best_threshold = float(thresholds[max(0, min(best_idx - 1, len(thresholds) - 1))]) if len(thresholds) else 0.5

        y_pred_best = (y_prob >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
        metrics["recommended_threshold"] = round(best_threshold, 4)
        metrics["f1_at_recommended_threshold"] = round(float(f1_score(y_test, y_pred_best, zero_division=0)), 4)
        metrics["confusion_matrix_at_recommended_threshold"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def train_single_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    target_name: str,
) -> Dict[str, object]:
    x = df[feature_cols].copy()
    y = (df[target_col].fillna(0) > 0).astype(int)

    class_counts = y.value_counts(dropna=False).to_dict()

    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x, y)
        metrics = {
            "accuracy": 1.0,
            "f1": 0.0,
            "support": int(len(y)),
            "note": "Target com apenas uma classe. Modelo Dummy treinado.",
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        }
        return {
            "model": model,
            "metrics": metrics,
        }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Oversampling da classe minoritaria para reduzir o impacto de desbalanceamento extremo.
    train_df = x_train.copy()
    train_df["__target__"] = y_train.values

    class_sizes = train_df["__target__"].value_counts()
    majority_class = int(class_sizes.idxmax())
    minority_class = int(class_sizes.idxmin())

    majority_df = train_df[train_df["__target__"] == majority_class]
    minority_df = train_df[train_df["__target__"] == minority_class]

    if len(minority_df) > 0 and len(minority_df) < len(majority_df):
        minority_upsampled = resample(
            minority_df,
            replace=True,
            n_samples=len(majority_df),
            random_state=42,
        )
        balanced_df = pd.concat([majority_df, minority_upsampled], axis=0).sample(
            frac=1.0,
            random_state=42,
        )
        x_train = balanced_df.drop(columns=["__target__"])
        y_train = balanced_df["__target__"]

    numeric_cols, categorical_cols = split_types(df, feature_cols)
    model = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    metrics["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}

    return {
        "model": model,
        "metrics": metrics,
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset nao encontrado em: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)

    feature_cols = build_feature_list(df)
    summary: Dict[str, object] = {
        "dataset_path": str(DATASET_PATH),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "feature_count": len(feature_cols),
        "targets": {},
    }

    for target_name, target_col in TARGETS.items():
        result = train_single_target(df, feature_cols, target_col, target_name)

        model_path = OUTPUT_DIR / f"cards_{target_name}_model.joblib"
        joblib.dump(result["model"], model_path)

        summary["targets"][target_name] = {
            "target_column": target_col,
            "model_path": str(model_path),
            "metrics": result["metrics"],
        }

    summary_path = OUTPUT_DIR / "cards_model_metrics.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=True, indent=2)

    print(f"Treino concluido. Metricas salvas em: {summary_path}")


if __name__ == "__main__":
    main()
