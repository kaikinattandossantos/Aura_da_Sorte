from fastapi import FastAPI, HTTPException
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import joblib
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


# ---------- Request models for /matches/analyze ----------

class MatchTimer(BaseModel):
    tm: int
    ts: int
    tt: str
    ta: int
    md: int

class TeamInfo(BaseModel):
    id: str
    name: str
    image_id: Optional[str] = None
    cc: Optional[str] = None

class LeagueInfo(BaseModel):
    id: str
    name: str
    cc: Optional[str] = None

class HalfScore(BaseModel):
    home: str
    away: str

class MatchEvent(BaseModel):
    id: str
    text: str

class MatchStats(BaseModel):
    attacks: Optional[List[str]] = None
    ball_safe: Optional[List[str]] = None
    corners: Optional[List[str]] = None
    corner_h: Optional[List[str]] = None
    dangerous_attacks: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    off_target: Optional[List[str]] = None
    on_target: Optional[List[str]] = None
    penalties: Optional[List[str]] = None
    possession_rt: Optional[List[str]] = None
    redcards: Optional[List[str]] = None
    substitutions: Optional[List[str]] = None
    yellowcards: Optional[List[str]] = None
    yellowred_cards: Optional[List[str]] = None

class MatchExtra(BaseModel):
    length: Optional[int] = None
    home_pos: Optional[str] = None
    away_pos: Optional[str] = None
    numberofperiods: Optional[str] = None
    periodlength: Optional[str] = None
    round: Optional[str] = None

class MatchData(BaseModel):
    id: str
    sport_id: Optional[str] = None
    time: Optional[str] = None
    time_status: Optional[str] = None
    league: Optional[LeagueInfo] = None
    home: Optional[TeamInfo] = None
    away: Optional[TeamInfo] = None
    ss: Optional[str] = None
    timer: Optional[MatchTimer] = None
    scores: Optional[Dict[str, HalfScore]] = None
    stats: Optional[MatchStats] = None
    extra: Optional[MatchExtra] = None
    events: Optional[List[MatchEvent]] = None

class AnalyzeMatchBody(BaseModel):
    success: Optional[int] = None
    results: List[MatchData]

# ---------------------------------------------------------

app = FastAPI(title="Analytics API - Performance Pro")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COPPER_DIR = os.path.join(BASE_DIR, "Copper")

parquet_path = os.path.join(COPPER_DIR, "shots_preparados.parquet")
model_path = os.path.join(COPPER_DIR, "model.json")

cards_dataset_path = os.path.join(COPPER_DIR, "Data", "Silver", "serie_a_ready.parquet")
cards_metrics_path = os.path.join(COPPER_DIR, "cards_model_metrics.json")
cards_yellow_model_path = os.path.join(COPPER_DIR, "cards_yellow_model.joblib")
cards_red_model_path = os.path.join(COPPER_DIR, "cards_red_model.joblib")

HF_TOKEN = os.getenv("HF_TOKEN") 
oraculo_model_path = os.path.join(BASE_DIR, "oraculo_iminencia.json")
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

def _stat(lst: Optional[List[str]], idx: int) -> int:
    try:
        return int(lst[idx]) if lst and len(lst) > idx else 0
    except (ValueError, TypeError):
        return 0


@app.post("/matches/analyze/{match_id}")
async def analyze_match(match_id: str, body: AnalyzeMatchBody):
    try:
        match = next((r for r in body.results if r.id == str(match_id)), None)
        if match is None:
            if body.results:
                match = body.results[0]
            else:
                raise HTTPException(status_code=404, detail="Nenhuma partida encontrada no corpo da requisição.")

        s = match.stats or MatchStats()
        timer = match.timer

        # Raw stats
        home_attacks       = _stat(s.attacks, 0)
        away_attacks       = _stat(s.attacks, 1)
        home_dangerous     = _stat(s.dangerous_attacks, 0)
        away_dangerous     = _stat(s.dangerous_attacks, 1)
        home_possession    = _stat(s.possession_rt, 0)
        away_possession    = _stat(s.possession_rt, 1)
        home_on_target     = _stat(s.on_target, 0)
        away_on_target     = _stat(s.on_target, 1)
        home_off_target    = _stat(s.off_target, 0)
        away_off_target    = _stat(s.off_target, 1)
        home_corners       = _stat(s.corners, 0)
        away_corners       = _stat(s.corners, 1)
        home_corners_ht    = _stat(s.corner_h, 0)
        away_corners_ht    = _stat(s.corner_h, 1)
        home_yellow        = _stat(s.yellowcards, 0)
        away_yellow        = _stat(s.yellowcards, 1)
        home_yellowred     = _stat(s.yellowred_cards, 0)
        away_yellowred     = _stat(s.yellowred_cards, 1)
        home_red           = _stat(s.redcards, 0)
        away_red           = _stat(s.redcards, 1)
        home_goals         = _stat(s.goals, 0)
        away_goals         = _stat(s.goals, 1)
        home_ball_safe     = _stat(s.ball_safe, 0)
        away_ball_safe     = _stat(s.ball_safe, 1)
        home_subs          = _stat(s.substitutions, 0)
        away_subs          = _stat(s.substitutions, 1)
        home_penalties     = _stat(s.penalties, 0)
        away_penalties     = _stat(s.penalties, 1)

        # Derived: finishing
        home_shots_total = home_on_target + home_off_target
        away_shots_total = away_on_target + away_off_target
        home_accuracy    = round(home_on_target / home_shots_total * 100, 1) if home_shots_total else 0.0
        away_accuracy    = round(away_on_target / away_shots_total * 100, 1) if away_shots_total else 0.0
        home_conversion  = round(home_goals / home_on_target * 100, 1) if home_on_target else 0.0
        away_conversion  = round(away_goals / away_on_target * 100, 1) if away_on_target else 0.0

        # Derived: attack pressure
        total_attacks = home_attacks + away_attacks + home_dangerous + away_dangerous
        home_atk_share = round((home_attacks + home_dangerous) / total_attacks * 100, 1) if total_attacks else 50.0
        home_danger_pct = round(home_dangerous / (home_attacks + home_dangerous) * 100, 1) if (home_attacks + home_dangerous) else 0.0
        away_danger_pct = round(away_dangerous / (away_attacks + away_dangerous) * 100, 1) if (away_attacks + away_dangerous) else 0.0

        # Derived: discipline pressure (yellow=1, yellow-red=2, red=3)
        home_card_pressure = home_yellow + home_yellowred * 2 + home_red * 3
        away_card_pressure = away_yellow + away_yellowred * 2 + away_red * 3

        # Score parsing
        score_parts = (match.ss or "0-0").split("-")
        current_home = int(score_parts[0]) if len(score_parts) > 0 else 0
        current_away = int(score_parts[1]) if len(score_parts) > 1 else 0

        ht_score = match.scores.get("1") if match.scores else None
        ht_home = int(ht_score.home) if ht_score else 0
        ht_away = int(ht_score.away) if ht_score else 0

        # Events timeline
        goals_tl, cards_tl, corners_tl = [], [], []
        if match.events:
            for ev in match.events:
                t = ev.text
                if "Goal" in t and "Race" not in t:
                    goals_tl.append(t)
                elif "Yellow Card" in t or "Red Card" in t:
                    cards_tl.append(t)
                elif "Corner" in t and "Race" not in t:
                    corners_tl.append(t)

        # Match state
        minute = timer.tm if timer else 0
        added_time = timer.ta if timer else 0
        time_status_map = {"1": "em_jogo", "2": "encerrado", "3": "nao_iniciado", "4": "adiado"}
        status_label = time_status_map.get(match.time_status or "", match.time_status)

        # AI narrative
        leader = match.home.name if match.home and current_home > current_away else (match.away.name if match.away else "Visitante") if current_away > current_home else None
        vantagem = abs(current_home - current_away)
        dominance_team = match.home.name if home_atk_share >= 50 and match.home else (match.away.name if match.away else "Visitante")
        summary_parts = [
            f"Minuto {minute}+{added_time}: placar {match.ss}.",
            f"{dominance_team} domina os ataques com {round(home_atk_share if home_atk_share >= 50 else 100 - home_atk_share, 1)}% do volume ofensivo.",
        ]
        if leader:
            summary_parts.append(f"{leader} vence por {vantagem} gol(s).")
        if home_card_pressure > 2 or away_card_pressure > 2:
            pressured = match.home.name if home_card_pressure > away_card_pressure and match.home else (match.away.name if match.away else "Visitante")
            summary_parts.append(f"{pressured} acumula pressão disciplinar alta (score {max(home_card_pressure, away_card_pressure)}).")
        if home_conversion > 0 or away_conversion > 0:
            best = match.home.name if home_conversion >= away_conversion and match.home else (match.away.name if match.away else "Visitante")
            summary_parts.append(f"{best} é mais clínico: {max(home_conversion, away_conversion):.0f}% de conversão nas finalizações no gol.")

        return {
            "match_id": match.id,
            "status": "success",
            "partida": {
                "liga": match.league.name if match.league else None,
                "rodada": match.extra.round if match.extra else None,
                "casa": match.home.name if match.home else None,
                "fora": match.away.name if match.away else None,
                "placar_atual": match.ss,
                "placar_primeiro_tempo": f"{ht_home}-{ht_away}",
                "gols_segundo_tempo": {"casa": current_home - ht_home, "fora": current_away - ht_away},
                "minuto": minute,
                "acrescimo": added_time,
                "status_jogo": status_label,
            },
            "dominio": {
                "posse_bola_pct": {"casa": home_possession, "fora": away_possession},
                "ataques": {"casa": home_attacks, "fora": away_attacks},
                "ataques_perigosos": {"casa": home_dangerous, "fora": away_dangerous},
                "participacao_ofensiva_pct": {
                    "casa": round(home_atk_share, 1),
                    "fora": round(100 - home_atk_share, 1),
                },
                "ratio_perigo_pct": {"casa": home_danger_pct, "fora": away_danger_pct},
                "bola_segura": {"casa": home_ball_safe, "fora": away_ball_safe},
            },
            "finalizacoes": {
                "casa": {
                    "total": home_shots_total,
                    "no_gol": home_on_target,
                    "fora_gol": home_off_target,
                    "precisao_pct": home_accuracy,
                    "conversao_pct": home_conversion,
                },
                "fora": {
                    "total": away_shots_total,
                    "no_gol": away_on_target,
                    "fora_gol": away_off_target,
                    "precisao_pct": away_accuracy,
                    "conversao_pct": away_conversion,
                },
            },
            "escanteios": {
                "casa": {"total": home_corners, "primeiro_tempo": home_corners_ht, "segundo_tempo": home_corners - home_corners_ht},
                "fora": {"total": away_corners, "primeiro_tempo": away_corners_ht, "segundo_tempo": away_corners - away_corners_ht},
                "total": home_corners + away_corners,
            },
            "disciplina": {
                "casa": {
                    "amarelos": home_yellow,
                    "amarelo_vermelho": home_yellowred,
                    "vermelhos": home_red,
                    "penaltis": home_penalties,
                    "substituicoes": home_subs,
                    "pressao_disciplinar": home_card_pressure,
                },
                "fora": {
                    "amarelos": away_yellow,
                    "amarelo_vermelho": away_yellowred,
                    "vermelhos": away_red,
                    "penaltis": away_penalties,
                    "substituicoes": away_subs,
                    "pressao_disciplinar": away_card_pressure,
                },
            },
            "linha_do_tempo": {
                "gols": goals_tl,
                "cartoes": cards_tl,
                "escanteios": corners_tl,
            },
            "resumo_ia": " ".join(summary_parts),
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