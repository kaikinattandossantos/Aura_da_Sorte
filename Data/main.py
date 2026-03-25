from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from typing import Dict, Any

app = FastAPI(title="Analytics API - Performance Pro")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COPPER_DIR = os.path.join(BASE_DIR, "Copper")

parquet_path = os.path.join(COPPER_DIR, "shots_preparados.parquet")
model_path = os.path.join(COPPER_DIR, "model.json")

print(f"🔄 Buscando arquivos em: {COPPER_DIR}")

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


@app.get("/")
async def root():
    return {"status": "online", "api": "Analytics Performance"}

@app.post("/matches/analyze/{match_id}")
async def analyze_match(match_id: int):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)