from fastapi import APIRouter, HTTPException
import pandas as pd
from app.services.llm import gerar_narrativa_cartao
from app.core import ml_manager

router = APIRouter(prefix="/cards", tags=["Cards"])

@router.get("/team/{team_name}")
async def analyze_team_cards(team_name: str, top_n: int = 10):
    try:
        if ml_manager.cards_df is None or ml_manager.cards_df.empty:
            raise HTTPException(status_code=503, detail="Modelos de cartões indisponíveis no momento.")

        team_rows = ml_manager.cards_df[ml_manager.cards_df["Squad"].astype(str).str.lower() == team_name.lower()].copy()
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


@router.get("/player/{player_name}")
async def analyze_player_cards(player_name: str):
    try:
        if ml_manager.cards_df is None or ml_manager.cards_df.empty:
            raise HTTPException(status_code=503, detail="Modelos de cartões indisponíveis no momento.")

        player_rows = ml_manager.cards_df[ml_manager.cards_df["Player"].astype(str).str.lower() == player_name.lower()]
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
                    "amarelo": ml_manager.yellow_threshold,
                    "vermelho": ml_manager.red_threshold,
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
