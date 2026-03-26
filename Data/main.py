from fastapi import FastAPI, HTTPException
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import joblib
from typing import Dict, Any

app = FastAPI(title="Analytics API - Performance Pro")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COPPER_DIR = os.path.join(BASE_DIR, "Copper")

parquet_path = os.path.join(COPPER_DIR, "shots_preparados.parquet")
model_path = os.path.join(COPPER_DIR, "model.json")

cards_dataset_path = os.path.join(COPPER_DIR, "Data", "Silver", "serie_a_ready.parquet")
cards_metrics_path = os.path.join(COPPER_DIR, "cards_model_metrics.json")
cards_yellow_model_path = os.path.join(COPPER_DIR, "cards_yellow_model.joblib")
cards_red_model_path = os.path.join(COPPER_DIR, "cards_red_model.joblib")

print(f"🔄 Buscando arquivos em: {COPPER_DIR}")

df_shots = pd.DataFrame()
model = None

try:
    df_raw = pl.read_parquet(
        parquet_path,
        columns=["match_id", "player_name", "minute", "location", "under_pressure", "shot_statsbomb_xg", "is_goal", "sob_pressao"]
    )

    print("📏 Calculando geometria para o modelo...")
    
    # Extraímos X e Y da lista 'location'
    df_raw = df_raw.with_columns([
        pl.col("location").list.get(0).fill_null(120.0).alias("x"),
        pl.col("location").list.get(1).fill_null(40.0).alias("y")
    ])

    x, y = df_raw["x"].to_numpy(), df_raw["y"].to_numpy()
    
    # Cálculos Geométricos
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
    print(f"❌ Erro ao processar dados: {e}")

try:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("🧠 Modelo carregado com sucesso.")
except Exception as e:
    print(f"⚠️ Aviso: Não foi possível carregar o modelo em {model_path}: {e}")


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


cards_df = pd.DataFrame()
cards_feature_cols: list[str] = []
cards_yellow_model = None
cards_red_model = None
yellow_threshold = 0.5
red_threshold = 0.5

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


@app.get("/")
async def root():
    return {"status": "online", "api": "Analytics Performance"}

@app.post("/matches/analyze/{match_id}")
async def analyze_match(match_id: int):
    try:
        if model is None or df_shots.empty:
            raise HTTPException(status_code=503, detail="Modelo de finalizações indisponível no momento.")

        partida = df_shots[df_shots["match_id"].astype(str) == str(match_id)]
        
        if partida.empty:
            raise HTTPException(status_code=404, detail=f"Partida {match_id} não encontrada no banco local.")

        X = partida[['distancia', 'angulo_visao', 'sob_pressao']].fillna(0)
        
        probs = model.predict_proba(X)[:, 1]
        partida = partida.copy()
        partida['prob_ia'] = probs
        
        partida['eficiencia_clinica'] = probs - partida['shot_statsbomb_xg']

        chutes_response = []
        for _, row in partida.iterrows():
            chutes_response.append({
                "minuto": int(row["minute"]),
                "jogador": row["player_name"],
                "xg_original": round(float(row["shot_statsbomb_xg"]), 4),
                "probabilidade_ia": round(float(row["prob_ia"]), 4),
                "diferencial": round(float(row["eficiencia_clinica"]), 4),
                "gol_confirmado": bool(row["is_goal"])
            })

        return {
            "match_id": match_id,
            "status": "success",
            "estatisticas": {
                "total_finalizacoes": len(partida),
                "expectativa_ia_total": round(float(partida["prob_ia"].sum()), 2),
                "eficiencia_media": round(float(partida["eficiencia_clinica"].mean()), 4)
            },
            "lances": chutes_response,
            "ai_summary": f"A IA analisou {len(partida)} finalizações. O desempenho clínico médio foi de {partida['eficiencia_clinica'].mean():.3f}."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")


@app.get("/cards/player/{player_name}")
async def analyze_player_cards(player_name: str):
    try:
        if cards_df.empty:
            raise HTTPException(status_code=503, detail="Modelos de cartões indisponíveis no momento.")

        player_rows = cards_df[cards_df["Player"].astype(str).str.lower() == player_name.lower()]
        if player_rows.empty:
            raise HTTPException(status_code=404, detail=f"Jogador {player_name} não encontrado.")

        row = player_rows.iloc[0]

        return {
            "status": "success",
            "jogador": row.get("Player"),
            "time": row.get("Squad"),
            "posicao": row.get("Pos"),
            "temporada": {
                "minutos": int(row.get("Min", 0)) if pd.notna(row.get("Min")) else 0,
                "partidas": int(row.get("MP", 0)) if pd.notna(row.get("MP")) else 0,
                "cartoes_amarelos_reais": int(row.get("CrdY", 0)) if pd.notna(row.get("CrdY")) else 0,
                "cartoes_vermelhos_reais": int(row.get("CrdR", 0)) if pd.notna(row.get("CrdR")) else 0,
            },
            "predicao": {
                "probabilidade_amarelo": round(float(row["prob_yellow"]), 4),
                "probabilidade_vermelho": round(float(row["prob_red"]), 4),
                "tendencia_amarelo": bool(int(row["pred_yellow"])),
                "tendencia_vermelho": bool(int(row["pred_red"])),
                "thresholds": {
                    "amarelo": yellow_threshold,
                    "vermelho": red_threshold,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento de cartões: {str(e)}")


@app.get("/cards/team/{team_name}")
async def analyze_team_cards(team_name: str, top_n: int = 10):
    try:
        if cards_df.empty:
            raise HTTPException(status_code=503, detail="Modelos de cartões indisponíveis no momento.")

        team_rows = cards_df[cards_df["Squad"].astype(str).str.lower() == team_name.lower()].copy()
        if team_rows.empty:
            raise HTTPException(status_code=404, detail=f"Time {team_name} não encontrado.")

        top_n = max(1, min(top_n, len(team_rows)))
        top_yellow = team_rows.sort_values("prob_yellow", ascending=False).head(top_n)
        top_red = team_rows.sort_values("prob_red", ascending=False).head(top_n)

        return {
            "status": "success",
            "time": team_name,
            "jogadores_no_dataset": int(len(team_rows)),
            "medias_time": {
                "prob_amarelo_media": round(float(team_rows["prob_yellow"].mean()), 4),
                "prob_vermelho_media": round(float(team_rows["prob_red"].mean()), 4),
            },
            "top_risco_amarelo": [
                {
                    "jogador": r["Player"],
                    "posicao": r.get("Pos"),
                    "prob_amarelo": round(float(r["prob_yellow"]), 4),
                    "cartoes_amarelos_reais": int(r.get("CrdY", 0)) if pd.notna(r.get("CrdY")) else 0,
                }
                for _, r in top_yellow.iterrows()
            ],
            "top_risco_vermelho": [
                {
                    "jogador": r["Player"],
                    "posicao": r.get("Pos"),
                    "prob_vermelho": round(float(r["prob_red"]), 4),
                    "cartoes_vermelhos_reais": int(r.get("CrdR", 0)) if pd.notna(r.get("CrdR")) else 0,
                }
                for _, r in top_red.iterrows()
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento de cartões do time: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)