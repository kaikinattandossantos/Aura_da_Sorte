import os
import joblib
import pandas as pd
import polars as pl
import xgboost as xgb
import numpy as np
import json

# Como estamos em Data/app/core/ml_manager.py, subimos 3 níveis para chegar na pasta raiz "Data"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COPPER_DIR = os.path.join(BASE_DIR, "Copper")

parquet_path = os.path.join(COPPER_DIR, "shots_preparados.parquet")
model_path = os.path.join(COPPER_DIR, "model.json")

cards_dataset_path = os.path.join(COPPER_DIR, "Data", "Silver", "serie_a_ready.parquet")
cards_metrics_path = os.path.join(COPPER_DIR, "cards_model_metrics.json")
cards_yellow_model_path = os.path.join(COPPER_DIR, "cards_yellow_model.joblib")
cards_red_model_path = os.path.join(COPPER_DIR, "cards_red_model.joblib")

oraculo_model_path = os.path.join(BASE_DIR, "Copper", "oraculo_iminencia_HIBRIDO_v1.json")
if not os.path.exists(oraculo_model_path):
    oraculo_model_path = os.path.join(BASE_DIR, "Copper", "oraculo_iminencia_HIBRIDO.json")

oraculo_model = None
df_shots = pd.DataFrame()
cards_df = pd.DataFrame()
cards_yellow_model = None
cards_red_model = None
yellow_threshold = 0.5
red_threshold = 0.5

def build_card_feature_list(df: pd.DataFrame) -> list[str]:
    excluded_exact = {"Rk", "Player", "Nation", "Comp"}
    excluded_fragments = ["CrdY", "CrdR", "2CrdY"]
    features = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if any(fragment in col for fragment in excluded_fragments):
            continue
        features.append(col)
    return features


def init_models():
    global oraculo_model, df_shots, cards_df, cards_yellow_model, cards_red_model, yellow_threshold, red_threshold

    print(f"🔄 Buscando arquivos em: {COPPER_DIR}")

    # Oráculo — carrega como Booster para evitar problema de num_feature=0
    try:
        oraculo_model = xgb.Booster()
        oraculo_model.load_model(oraculo_model_path)
        print("🧠 Oráculo Híbrido carregado com sucesso.")
    except Exception as e:
        print(f"⚠️ Aviso: Oráculo falhou: {e}")

    # Shots
    try:
        df_raw = pl.read_parquet(
            parquet_path,
            columns=["match_id", "player_name", "minute", "location", "under_pressure", "shot_statsbomb_xg", "is_goal", "sob_pressao"]
        )

        print("📏 Calculando geometria para o modelo...")
        
        df_raw = df_raw.with_columns([
            pl.col("location").list.get(0).fill_null(120.0).alias("x"),
            pl.col("location").list.get(1).fill_null(40.0).alias("y")
        ])

        x, y = df_raw["x"].to_numpy(), df_raw["y"].to_numpy()
        
        dist = np.sqrt((120 - x)**2 + (40 - y)**2)
        a = np.sqrt((120 - x)**2 + (36 - y)**2)
        b = np.sqrt((120 - x)**2 + (44 - y)**2)
        cos_theta = np.clip((a**2 + b**2 - 8**2) / (2 * a * b), -1.0, 1.0)
        angulo = np.degrees(np.arccos(cos_theta))

        df_shots = df_raw.with_columns([
            pl.Series("distancia", dist),
            pl.Series("angulo_visao", angulo)
        ]).to_pandas()

        print(f"✅ Pronto! {len(df_shots):,} chutes processados com geometria.")

    except Exception as e:
        print(f"❌ Erro ao processar dados de chutes: {e}")

    # Cards
    try:
        cards_df = pd.read_parquet(cards_dataset_path)
        cards_feature_cols = build_card_feature_list(cards_df)

        cards_yellow_model = joblib.load(cards_yellow_model_path)
        cards_red_model = joblib.load(cards_red_model_path)

        if os.path.exists(cards_metrics_path):
            with open(cards_metrics_path, "r", encoding="utf-8") as fp:
                metrics = json.load(fp)
                yellow_threshold = metrics.get("targets", {}).get("yellow", {}).get("metrics", {}).get("recommended_threshold", 0.5)
                red_threshold = metrics.get("targets", {}).get("red", {}).get("metrics", {}).get("recommended_threshold", 0.5)

        x_cards = cards_df[cards_feature_cols].copy()
        cards_df = cards_df.copy()
        cards_df["prob_yellow"] = cards_yellow_model.predict_proba(x_cards)[:, 1]
        cards_df["prob_red"] = cards_red_model.predict_proba(x_cards)[:, 1]
        cards_df["pred_yellow"] = (cards_df["prob_yellow"] >= yellow_threshold).astype(int)
        cards_df["pred_red"] = (cards_df["prob_red"] >= red_threshold).astype(int)

        print(f"🟨🟥 Modelos de cartões carregados com sucesso. Jogadores: {len(cards_df)}")
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível carregar os modelos de cartões: {e}")
