from fastapi import APIRouter, HTTPException
import pandas as pd
from app.models.schemas import MatchAnalyzeRequest
from app.services.heuristics import (
    preparar_input_hibrido, 
    _extract_stats_payload, 
    calcular_estatisticas_ao_vivo, 
    calcular_probabilidades_heuristicas, 
    calcular_previsoes_jogo
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
        prob = float(ml_manager.oraculo_model.predict_proba([features])[:, 1][0])
        
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
