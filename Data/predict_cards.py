from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
COPPER_DIR = BASE_DIR / "Copper"
DATASET_PATH = COPPER_DIR / "Data" / "Silver" / "serie_a_ready.parquet"
METRICS_PATH = COPPER_DIR / "cards_model_metrics.json"

YELLOW_MODEL_PATH = COPPER_DIR / "cards_yellow_model.joblib"
RED_MODEL_PATH = COPPER_DIR / "cards_red_model.joblib"

OUTPUT_PATH = COPPER_DIR / "cards_predictions.parquet"


def build_feature_list(df: pd.DataFrame) -> list[str]:
    excluded_exact = {"Rk", "Player", "Nation", "Comp"}
    excluded_fragments = ["CrdY", "CrdR", "2CrdY"]

    features: list[str] = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if any(fragment in col for fragment in excluded_fragments):
            continue
        features.append(col)
    return features


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    features = build_feature_list(df)

    yellow_model = joblib.load(YELLOW_MODEL_PATH)
    red_model = joblib.load(RED_MODEL_PATH)

    with METRICS_PATH.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)

    yellow_threshold = metrics["targets"]["yellow"]["metrics"].get("recommended_threshold", 0.5)
    red_threshold = metrics["targets"]["red"]["metrics"].get("recommended_threshold", 0.5)

    x = df[features].copy()

    df_out = df[["Player", "Squad", "Pos", "Age", "MP", "Min", "CrdY", "CrdR"]].copy()
    df_out["prob_yellow"] = yellow_model.predict_proba(x)[:, 1]
    df_out["prob_red"] = red_model.predict_proba(x)[:, 1]

    df_out["pred_yellow"] = (df_out["prob_yellow"] >= yellow_threshold).astype(int)
    df_out["pred_red"] = (df_out["prob_red"] >= red_threshold).astype(int)

    df_out.to_parquet(OUTPUT_PATH, index=False)

    print(f"Predicoes salvas em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
