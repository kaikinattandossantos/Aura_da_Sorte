from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import xgboost as xgb
from app.models.schemas import MatchAnalyzeRequest, B365MatchEvent
from app.services.heuristics import (
    preparar_input_hibrido,
    _extract_stats_payload,
    calcular_estatisticas_ao_vivo,
    calcular_probabilidades_heuristicas,
    calcular_previsoes_jogo,
    cc_to_flag,
    extract_timer,
    parse_events_to_alerts,
    _to_float,
)
from app.services.llm import gerar_narrativa_oraculo
from app.core import ml_manager

router = APIRouter(prefix="/matches", tags=["Matches"])

@router.get("/analyze/contract")
async def get_matches_analyze_contract():
    return {
        "endpoint": "/matches/analyze/{match_id}",
        "method": "POST",
        "request": {
            "description": "Aceita payload bruto da B365 (v1/event/view, v3/event/view) e/ou payload normalizado.",
            "required": [],
            "optional": []
        },
        "response_success_example": {
            "status": "success",
            "match_id": "233630",
            "momentum": {
                "equipe_dominante": "Casa",
                "texto": "Casa dominando (73%)",
                "valor": 73
            },
            "estatisticas_ao_vivo": {
                "posse_casa": 73,
                "posse_fora": 17,
                "gols_casa": 7,
                "gols_fora": 0,
                "ataque_casa": 73,
                "ataque_fora": 17
            },
            "proximos_minutos": {
                "gol_prox_10_min": 63.0,
                "escanteio_prox_5_min": 55.0,
                "cartao_amarelo_prox_10_min": 41.0,
                "substituicao_prox_5_min": 70.0,
                "falta_perigosa_prox_5_min": 34.0,
                "chute_gol_prox_3_min": 78.0
            },
            "resultado_final": {
                "vencedor_casa_prob": 92.0,
                "vencedor_fora_prob": 2.0,
                "empate_prob": 6.0,
                "ambos_marcam_prob": 8.0,
                "mais_gols_linha": 7.5,
                "mais_gols_prob": 48.0
            },
            "narrativa_llm": "Alta chance de mais gols — Time pressionando",
            "signals": {
                "x": 105.0,
                "y": 40.0,
                "ataques_perigosos_casa": 37.0,
                "escanteios_total": 8.0
            }
        },
        "response_error_example": {
            "status": "error",
            "code": "ANALYSIS_PROCESSING_ERROR",
            "message": "Erro ao processar análise da partida.",
            "match_id": 12345
        }
    }


@router.post("/analyze/{match_id}")
async def analyze_match(match_id: int, live_data: MatchAnalyzeRequest):
    try:
        dados_dict = live_data.model_dump()
        
        # 1. Transforma o JSON da APIBet no formato da IA principal (Oráculo de Gol)
        parsed_input = preparar_input_hibrido(dados_dict)
        features = parsed_input["features"]
        
        if ml_manager.oraculo_model is None:
             raise Exception("O modelo Oráculo não foi carregado corretamente.")

        # 2. O Oráculo Híbrido (XGBoost) prevê a iminência de gol
        dmatrix = xgb.DMatrix(np.array([features]))
        prob = float(ml_manager.oraculo_model.predict(dmatrix)[0])
        
        # Extrai os metadados do payload da BetsAPI para os nossos insights completos
        stats_api = _extract_stats_payload(dados_dict)
        
        # Extrai infos adicionais, como placar e tempo, do topo do payload ou dentro de "results"
        score = "0-0"
        time_m = 45
        if "results" in dados_dict and dados_dict["results"] and isinstance(dados_dict["results"][0], dict):
            score = dados_dict["results"][0].get("ss", "0-0")
            time_m = int(dados_dict["results"][0].get("time_status", 45))  # Aproximação simples
            if time_m == 1: time_m = 45 # inplay half
            elif time_m == 3: time_m = 90 # finished
            
        # 3. Modelos Heurísticos - A Magia para o Java (probabilidades mastigadas)
        stats_e_momentum = calcular_estatisticas_ao_vivo(stats_api, score)
        insights = calcular_probabilidades_heuristicas(stats_api, prob, time_m)
        previsoes = calcular_previsoes_jogo(stats_api, score)

        # 4. Chama o Hugging Face para gerar a "Voz do Oráculo" baseada em metadados
        contexto_ia = f"Placar {score}. Time da casa sofreu {stats_api.get('dangerous_attacks', [0,0])[1] if stats_api.get('dangerous_attacks') and len(stats_api['dangerous_attacks'])>1 else 0} ataques perigosos."
        narrativa = await gerar_narrativa_oraculo(contexto_ia, prob)

        # 5. Entrega mastigada para o Backend Java e Front
        return {
            "status": "success",
            "match_id": str(match_id),
            "momentum": stats_e_momentum["momentum"],
            "estatisticas_ao_vivo": stats_e_momentum["estatisticas_ao_vivo"],
            "proximos_minutos": insights,
            "resultado_final": previsoes,
            "narrativa_llm": narrativa.get("summary", ""),
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
                "message": f"Erro ao processar análise da partida. {str(e)}",
                "match_id": match_id
            }
        )


@router.post("/dashboard/{match_id}")
async def match_dashboard(match_id: int, event: B365MatchEvent):
    if not event.results:
        raise HTTPException(status_code=422, detail="Campo 'results' vazio no payload.")

    match = event.results[0]

    try:
        analyze_input = {
            "ataques_perigosos": {},
            "escanteios": {},
            "x": 105.0,
            "y": 40.0,
            "location": None,
            "stats": {
                "dangerous_attacks": [int(_to_float(v)) for v in match.stats.dangerous_attacks],
                "corners": [int(_to_float(v)) for v in match.stats.corners],
            },
            "results": [],
        }

        parsed_input = preparar_input_hibrido(analyze_input)
        features = parsed_input["features"]

        if ml_manager.oraculo_model is None:
            raise Exception("O modelo Oráculo não foi carregado corretamente.")

        dmatrix = xgb.DMatrix(np.array([features]))
        prob = float(ml_manager.oraculo_model.predict(dmatrix)[0])

        contexto_ia = (
            f"{match.home.name} vs {match.away.name}: {match.ss}. "
            f"{features[8]:.0f} ataques perigosos e {features[7]:.0f} escanteios."
        )
        narrativa = await gerar_narrativa_oraculo(contexto_ia, prob)

        # --- monta resposta no formato app.json ---
        nivel = "CRÍTICO" if prob > 0.80 else "ALTO" if prob > 0.60 else "NORMAL"
        timer = extract_timer(match)
        stats = match.stats

        try:
            goals_home, goals_away = (int(p) for p in match.ss.split("-", 1))
        except Exception:
            goals_home = int(_to_float(stats.goals[0]))
            goals_away = int(_to_float(stats.goals[1])) if len(stats.goals) > 1 else 0

        poss_home = int(_to_float(stats.possession_rt[0], 50))
        poss_away = int(_to_float(stats.possession_rt[1], 50)) if len(stats.possession_rt) > 1 else 100 - poss_home
        attack_home = int(_to_float(stats.dangerous_attacks[0]))
        attack_away = int(_to_float(stats.dangerous_attacks[1])) if len(stats.dangerous_attacks) > 1 else 0

        total_attacks = max(1, attack_home + attack_away)
        momentum_home = min(95, round(attack_home / total_attacks * 100))
        momentum_away = 100 - momentum_home

        home_prob = round(prob, 2)
        away_prob = round((1 - prob) * 0.35, 2)
        draw_prob = round(max(0.0, 1.0 - home_prob - away_prob), 2)

        alerts = parse_events_to_alerts(match.events, match.home.name, match.away.name)
        if narrativa.get("summary"):
            alerts.insert(0, {"type": "info", "message": narrativa["summary"], "minute": timer})

        stats_api = {
            "dangerous_attacks": [attack_home, attack_away],
            "corners": [int(_to_float(v)) for v in stats.corners],
            "on_target": [int(_to_float(v)) for v in stats.on_target],
            "attacks": [int(_to_float(v)) for v in stats.attacks],
            "yellowcards": [int(_to_float(v)) for v in stats.yellowcards],
            "substitutions": [int(_to_float(v)) for v in stats.substitutions],
            "possession_rt": [poss_home, poss_away],
        }
        proximos = calcular_probabilidades_heuristicas(stats_api, prob, timer)
        previsoes = calcular_previsoes_jogo(stats_api, match.ss)

        total_goals = goals_home + goals_away

        return {
            "match_id": match.id,
            "league": match.league.name,
            "timer": timer,
            "score": match.ss,
            "teams": {
                "home": {"name": match.home.name, "flag": cc_to_flag(match.home.cc)},
                "away": {"name": match.away.name, "flag": cc_to_flag(match.away.cc)},
            },
            "predictions": {
                "home_prob": home_prob,
                "draw_prob": draw_prob,
                "away_prob": away_prob,
                "alert_level": nivel,
            },
            "alerts": alerts,
            "momentum": {
                "home": momentum_home,
                "away": momentum_away,
                "note": f"{match.home.name} dominando" if momentum_home > 60 else f"{match.away.name} dominando" if momentum_away > 60 else "Equilíbrio",
            },
            "live_stats": {
                "possession_home": poss_home,
                "possession_away": poss_away,
                "goals_home": goals_home,
                "goals_away": goals_away,
                "attack_home": attack_home,
                "attack_away": attack_away,
            },
            "next_minutes": [
                {"event": "goal",           "label": "Gol nos próx. 10 min",  "window": f"{timer}' → {timer + 10}'", "prob": round(proximos["gol_prox_10_min"] / 100, 2),            "color": "green" if prob > 0.6 else "yellow"},
                {"event": "corner",         "label": "Escanteio",              "window": "Próximos 5 min",            "prob": round(proximos["escanteio_prox_5_min"] / 100, 2),        "color": "yellow"},
                {"event": "yellow_card",    "label": "Cartão amarelo",         "window": "Próximos 10 min",           "prob": round(proximos["cartao_amarelo_prox_10_min"] / 100, 2),  "color": "yellow"},
                {"event": "substitution",   "label": "Substituição",           "window": "Próximos 5 min",            "prob": round(proximos["substituicao_prox_5_min"] / 100, 2),     "color": "blue"},
                {"event": "dangerous_foul", "label": "Falta perigosa",         "window": "Próximos 5 min",            "prob": round(proximos["falta_perigosa_prox_5_min"] / 100, 2),  "color": "red"},
                {"event": "shot_on_target", "label": "Chute a gol",            "window": "Próximos 3 min",            "prob": round(proximos["chute_gol_prox_3_min"] / 100, 2),        "color": "green" if prob > 0.7 else "yellow"},
            ],
            "final_result": [
                {"market": "home_win",                    "label": f"{match.home.name} vence",          "description": "Vitória do mandante", "prob": round(previsoes["vencedor_casa_prob"] / 100, 2), "color": "green" if previsoes["vencedor_casa_prob"] > 60 else "yellow" if previsoes["vencedor_casa_prob"] > 40 else "red"},
                {"market": "btts",                        "label": "Ambos marcam",                      "description": "Ambas marcam",        "prob": round(previsoes["ambos_marcam_prob"] / 100, 2), "color": "green" if previsoes["ambos_marcam_prob"] > 60 else "yellow" if previsoes["ambos_marcam_prob"] > 40 else "red"},
                {"market": f"over_{total_goals + 1}_5",  "label": f"Mais de {total_goals + 1},5 gols", "description": "Total da partida",    "prob": round(previsoes["mais_gols_prob"] / 100, 2),    "color": "green" if previsoes["mais_gols_prob"] > 60 else "yellow" if previsoes["mais_gols_prob"] > 40 else "red"},
            ],
        }

    except Exception as e:
        print(f"❌ Erro no Dashboard: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "code": "DASHBOARD_PROCESSING_ERROR",
                "message": f"Erro ao processar dashboard da partida. {str(e)}",
                "match_id": match_id,
            },
        )
