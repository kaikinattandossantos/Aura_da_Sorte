from fastapi import FastAPI, HTTPException
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import joblib
from typing import Dict, Any
import requests
from pydantic import BaseModel, ConfigDict, Field
app = FastAPI(title="Analytics API - Performance Pro")


class MatchAnalyzeRequest(BaseModel):
    ataques_perigosos: Dict[str, Any] = Field(default_factory=dict)
    escanteios: Dict[str, Any] = Field(default_factory=dict)
    x: float | str | None = None
    y: float | str | None = None
    location: list[Any] | None = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    results: list[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COPPER_DIR = os.path.join(BASE_DIR, "Copper")

parquet_path = os.path.join(COPPER_DIR, "shots_preparados.parquet")
model_path = os.path.join(COPPER_DIR, "model.json")

cards_dataset_path = os.path.join(COPPER_DIR, "Data", "Silver", "serie_a_ready.parquet")
cards_metrics_path = os.path.join(COPPER_DIR, "cards_model_metrics.json")
cards_yellow_model_path = os.path.join(COPPER_DIR, "cards_yellow_model.joblib")
cards_red_model_path = os.path.join(COPPER_DIR, "cards_red_model.joblib")

HF_TOKEN = os.getenv("HF_TOKEN") 
oraculo_model_path = os.path.join(BASE_DIR, "Copper", "oraculo_iminencia_HIBRIDO_v1.json")
if not os.path.exists(oraculo_model_path):
    # Fallback para compatibilidade com nome legado.
    oraculo_model_path = os.path.join(BASE_DIR, "Copper", "oraculo_iminencia_HIBRIDO.json")
oraculo_model = xgb.XGBClassifier()
oraculo_model.load_model(oraculo_model_path)

live_buffers: Dict[str, list] = {}



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

async def gerar_narrativa_oraculo(contexto, probabilidade):
    if not HF_TOKEN:
        return {
            "summary": "sem token Hugging Face configurado. Configure a variável de ambiente HF_TOKEN",
            "source": "fallback"
        }
        
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    prompt = f"<s>[INST] Você é um analista de jogo de futebol. Gere um alerta de 1 frase curta e impactante para um dashboard de apostas baseado nisso: {contexto}. Probabilidade de gol: {probabilidade:.1%}. [/INST]"
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=5)
        payload = response.json()
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            text = payload[0]["generated_text"].split("[/INST]")[-1].strip()
            if text:
                return {
                    "summary": text,
                    "source": "llm"
                }
        raise ValueError("Resposta inesperada da API de inferência")
    except Exception:
        return {
            "summary": f"🚨 PERIGO! O Oráculo detectou {probabilidade:.1%} de chance de gol agora!",
            "source": "fallback"
        }


async def gerar_narrativa_cartao(contexto, probabilidade, tipo_cartao):
    if not HF_TOKEN:
        return {
            "summary": f"Sem HF_TOKEN. Risco de cartao {tipo_cartao}: {probabilidade:.1%}.",
            "source": "fallback"
        }

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    prompt = (
        f"<s>[INST] Voce e um analista de futebol. Gere 1 frase curta e objetiva sobre risco de cartao {tipo_cartao}. "
        f"Contexto: {contexto}. Probabilidade: {probabilidade:.1%}. [/INST]"
    )

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=5)
        payload = response.json()
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            text = payload[0]["generated_text"].split("[/INST]")[-1].strip()
            if text:
                return {
                    "summary": text,
                    "source": "llm"
                }
        raise ValueError("Resposta inesperada da API de inferencia")
    except Exception:
        return {
            "summary": f"Risco de cartao {tipo_cartao} em {probabilidade:.1%}.",
            "source": "fallback"
        }


def _to_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            return float(value.replace(",", "."))
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_stats_payload(dados_live: Dict[str, Any]) -> Dict[str, Any]:
    stats = dados_live.get("stats")
    if isinstance(stats, dict):
        return stats

    results = dados_live.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        nested_stats = results[0].get("stats")
        if isinstance(nested_stats, dict):
            return nested_stats

    return {}

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


@app.get("/contracts/matches-analyze")
async def get_matches_analyze_contract():
    return {
        "endpoint": "/matches/analyze/{match_id}",
        "method": "POST",
        "request": {
            "description": "Aceita payload bruto da B365 (v1/event/view, v3/event/view) e/ou payload normalizado.",
            "required": [],
            "optional": [
                "ataques_perigosos.casa",
                "escanteios.total",
                "x",
                "y",
                "location[0]",
                "location[1]",
                "stats.dangerous_attacks",
                "stats.corners",
                "results[0].stats.dangerous_attacks",
                "results[0].stats.corners"
            ]
        },
        "response_success_example": {
            "status": "success",
            "match_id": 12345,
            "analysis": {
                "probabilidade": 0.7342,
                "nivel": "ALTO",
                "cor_alerta": "#FFA500"
            },
            "insights": {
                "summary": "Pressão ofensiva crescente no time da casa.",
                "source": "llm"
            },
            "signals": {
                "x": 105.0,
                "y": 40.0,
                "ataques_perigosos_casa": 37.0,
                "escanteios_total": 8.0
            },
            "timestamp": "2026-03-27T12:00:00"
        },
        "response_error_example": {
            "status": "error",
            "code": "ANALYSIS_PROCESSING_ERROR",
            "message": "Erro ao processar análise da partida.",
            "match_id": 12345
        }
    }



# 1. Função Tradutora (Mapeia o JSON da APIBet para as 9 Features)
def preparar_input_hibrido(dados_live):
    # Extração dos dados Macro com suporte a múltiplos formatos.
    ataques_casa = _to_float(dados_live.get("ataques_perigosos", {}).get("casa", 0), 0)
    escanteios_total = _to_float(dados_live.get("escanteios", {}).get("total", 0), 0)

    stats = _extract_stats_payload(dados_live)
    dangerous_attacks = stats.get("dangerous_attacks", [0, 0])
    corners = stats.get("corners", [0, 0])

    if isinstance(dangerous_attacks, list) and dangerous_attacks:
        ataques_casa = _to_float(dangerous_attacks[0], ataques_casa)
    elif dangerous_attacks is not None:
        ataques_casa = _to_float(dangerous_attacks, ataques_casa)

    if isinstance(corners, list) and corners:
        canto_casa = _to_float(corners[0], 0)
        canto_fora = _to_float(corners[1], 0) if len(corners) > 1 else 0.0
        escanteios_total = canto_casa + canto_fora
    elif corners is not None:
        escanteios_total = _to_float(corners, escanteios_total)
    
    # Dados Micro (Geometria)
    # Se a API não mandar X e Y exatos, usamos a média da zona de ataque (105, 40)
    x = _to_float(dados_live.get("x", 105.0), 105.0)
    y = _to_float(dados_live.get("y", 40.0), 40.0)

    location = dados_live.get("location")
    if isinstance(location, list) and len(location) >= 2:
        x = _to_float(location[0], x)
        y = _to_float(location[1], y)

    # Cálculos Geométricos (O Oráculo exige isso)
    dist = np.sqrt((120 - x)**2 + (40 - y)**2)
    a_dist = np.sqrt((120 - x)**2 + (36 - y)**2)
    b_dist = np.sqrt((120 - x)**2 + (44 - y)**2)
    cos_theta = np.clip((a_dist**2 + b_dist**2 - 8**2) / (2 * a_dist * b_dist), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_theta))

    # Retorna o array exato de 9 colunas que o seu modelo espera
    features = [
        x, ang, dist, x,  # Ajuste a ordem conforme o SEU model.get_booster().feature_names
        0.65, 4.2, 1.8,   # Placeholders de pressão e aceleração
        escanteios_total, 
        ataques_casa
    ]
    return {
        "features": features,
        "signals": {
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "ataques_perigosos_casa": round(float(ataques_casa), 2),
            "escanteios_total": round(float(escanteios_total), 2)
        }
    }

@app.post("/matches/analyze/{match_id}")
async def analyze_match(match_id: int, live_data: MatchAnalyzeRequest):
    try:
        # 1. Transforma o JSON da APIBet no formato da IA
        parsed_input = preparar_input_hibrido(live_data.model_dump())
        features = parsed_input["features"]
        
        # 2. O Oráculo Híbrido (0.78 AUC) entra em ação
        prob = float(oraculo_model.predict_proba([features])[:, 1][0])

        # 3. Chama o Hugging Face para gerar a "Voz do Oráculo"
        contexto_ia = f"Time com {features[8]} ataques e {features[7]} escanteios. Pressão total!"
        narrativa = await gerar_narrativa_oraculo(contexto_ia, prob)

        nivel = "CRÍTICO" if prob > 0.80 else "ALTO" if prob > 0.60 else "NORMAL"
        cor_alerta = "#FF0000" if prob > 0.80 else "#FFA500" if prob > 0.60 else "#00FF00"

        # 4. Contrato padronizado para consumo do backend e frontend.
        return {
            "status": "success",
            "match_id": match_id,
            "analysis": {
                "probabilidade": round(prob, 4),
                "nivel": nivel,
                "cor_alerta": cor_alerta
            },
            "insights": narrativa,
            "signals": parsed_input["signals"],
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Erro na Integração Live: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "code": "ANALYSIS_PROCESSING_ERROR",
                "message": "Erro ao processar análise da partida.",
                "match_id": match_id
            }
        )




@app.get("/cards/player/{player_name}")
async def analyze_player_cards(player_name: str):
    try:
        if cards_df.empty:
            raise HTTPException(status_code=503, detail="Modelos de cartões indisponíveis no momento.")

        player_rows = cards_df[cards_df["Player"].astype(str).str.lower() == player_name.lower()]
        if player_rows.empty:
            raise HTTPException(status_code=404, detail=f"Jogador {player_name} não encontrado.")

        row = player_rows.iloc[0]
        prob_yellow = float(row["prob_yellow"])
        prob_red = float(row["prob_red"])

        contexto_amarelo = (
            f"Jogador {row.get('Player')} do time {row.get('Squad')} com {int(row.get('CrdY', 0)) if pd.notna(row.get('CrdY')) else 0} "
            "amarelos na temporada"
        )
        contexto_vermelho = (
            f"Jogador {row.get('Player')} do time {row.get('Squad')} com {int(row.get('CrdR', 0)) if pd.notna(row.get('CrdR')) else 0} "
            "vermelhos na temporada"
        )

        insight_amarelo = await gerar_narrativa_cartao(contexto_amarelo, prob_yellow, "amarelo")
        insight_vermelho = await gerar_narrativa_cartao(contexto_vermelho, prob_red, "vermelho")

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
                "probabilidade_amarelo": round(prob_yellow, 4),
                "probabilidade_vermelho": round(prob_red, 4),
                "tendencia_amarelo": bool(int(row["pred_yellow"])),
                "tendencia_vermelho": bool(int(row["pred_red"])),
                "thresholds": {
                    "amarelo": yellow_threshold,
                    "vermelho": red_threshold,
                },
            },
            "insights": {
                "amarelo": insight_amarelo,
                "vermelho": insight_vermelho,
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

        prob_amarelo_media = float(team_rows["prob_yellow"].mean())
        prob_vermelho_media = float(team_rows["prob_red"].mean())

        contexto_amarelo_time = f"Time {team_name} com media de risco de amarelo em {prob_amarelo_media:.1%}"
        contexto_vermelho_time = f"Time {team_name} com media de risco de vermelho em {prob_vermelho_media:.1%}"

        insight_amarelo_time = await gerar_narrativa_cartao(contexto_amarelo_time, prob_amarelo_media, "amarelo")
        insight_vermelho_time = await gerar_narrativa_cartao(contexto_vermelho_time, prob_vermelho_media, "vermelho")

        return {
            "status": "success",
            "time": team_name,
            "jogadores_no_dataset": int(len(team_rows)),
            "medias_time": {
                "prob_amarelo_media": round(prob_amarelo_media, 4),
                "prob_vermelho_media": round(prob_vermelho_media, 4),
            },
            "insights": {
                "amarelo": insight_amarelo_time,
                "vermelho": insight_vermelho_time,
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