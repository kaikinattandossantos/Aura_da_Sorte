import numpy as np
from typing import Dict, Any

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


def calcular_estatisticas_ao_vivo(stats: dict, score: str):
    home_score, away_score = 0, 0
    if score and isinstance(score, str) and "-" in score:
        try:
            p = score.split("-")
            home_score, away_score = int(p[0]), int(p[1])
        except:
            pass

    posse = stats.get("possession_rt", [50, 50]) if stats.get("possession_rt") else [50, 50]
    ataques = stats.get("attacks", [0, 0]) if stats.get("attacks") else [0, 0]
    
    posse_casa = posse[0]
    posse_fora = posse[1] if len(posse) > 1 else (100 - posse_casa)
    
    ataque_casa = ataques[0]
    ataque_fora = ataques[1] if len(ataques) > 1 else 0

    if posse_casa > posse_fora:
        momentum_eq = "Casa"
        momentum_val = posse_casa
        momentum_txt = f"Casa dominando ({posse_casa}%)"
    elif posse_fora > posse_casa:
        momentum_eq = "Fora"
        momentum_val = int(posse_fora)
        momentum_txt = f"Fora dominando ({int(posse_fora)}%)"
    else:
        momentum_eq = "Equilíbrio"
        momentum_val = 50
        momentum_txt = "Partida equilibrada (50%)"

    return {
        "momentum": {
            "equipe_dominante": momentum_eq,
            "texto": momentum_txt,
            "valor": momentum_val
        },
        "estatisticas_ao_vivo": {
            "posse_casa": posse_casa,
            "posse_fora": int(posse_fora),
            "gols_casa": home_score,
            "gols_fora": away_score,
            "ataque_casa": ataque_casa,
            "ataque_fora": ataque_fora
        }
    }

def calcular_probabilidades_heuristicas(stats: dict, prob_gol_oraculo: float, minuto_jogo: int = 45):
    # Extrair stats do payload real do BetsAPI
    dan_attacks_home = stats.get("dangerous_attacks", [0, 0])[0] if stats.get("dangerous_attacks") else 0
    dan_attacks_away = stats.get("dangerous_attacks", [0, 0])[1] if stats.get("dangerous_attacks") and len(stats.get("dangerous_attacks")) > 1 else 0
    dan_attacks = dan_attacks_home + dan_attacks_away
    
    attacks = sum(stats.get("attacks", [0, 0])) or 1
    on_target = sum(stats.get("on_target", [0, 0]))
    yellow_cards = sum(stats.get("yellowcards", [0, 0]))
    
    # Intensidade do jogo (0 a 1)
    intensidade_ofensiva = min(1.0, dan_attacks / (attacks + 0.1))
    
    # 1. Escanteio (+peso de chute ao alvo e ataques perigosos)
    prob_escanteio = (intensidade_ofensiva * 40) + (on_target * 4) + (dan_attacks * 0.5)
    prob_escanteio = min(92.5, max(10.0, prob_escanteio))
    
    # 2. Cartão (+peso do numero de cartoes que ja sairam e pressao defensiva)
    prob_cartao = min(95.0, (dan_attacks * 0.4) + (yellow_cards * 10) + 15.0)
    
    # 3. Falta Perigosa
    prob_falta_perigosa = min(85.0, (intensidade_ofensiva * 60) + (dan_attacks * 0.3))
    
    # 4. Chute a gol
    prob_chute = min(95.0, (intensidade_ofensiva * 70) + (on_target * 5) + 20.0)
    
    # 5. Substituição tem pico perto dos 60-75 min e > 85 min
    if 60 <= minuto_jogo <= 75: prob_substituicao = 85.0
    elif minuto_jogo >= 85: prob_substituicao = 70.0
    else: prob_substituicao = 35.0
    
    return {
        "gol_prox_10_min": round(prob_gol_oraculo * 100, 1),
        "escanteio_prox_5_min": round(prob_escanteio, 1),
        "cartao_amarelo_prox_10_min": round(prob_cartao, 1),
        "substituicao_prox_5_min": round(prob_substituicao, 1),
        "falta_perigosa_prox_5_min": round(prob_falta_perigosa, 1),
        "chute_gol_prox_3_min": round(prob_chute, 1)
    }

def calcular_previsoes_jogo(stats: dict, score: str):
    home_score, away_score = 0, 0
    if score and isinstance(score, str) and "-" in score:
        try:
            p = score.split("-")
            home_score, away_score = int(p[0]), int(p[1])
        except:
            pass
            
    da_home = stats.get("dangerous_attacks", [0, 0])[0] if stats.get("dangerous_attacks") else 0
    da_away = stats.get("dangerous_attacks", [0, 0])[1] if stats.get("dangerous_attacks") and len(stats.get("dangerous_attacks")) > 1 else 0
    total_da = (da_home + da_away) or 1
    
    # 6. Chance de Vitoria 
    base_home = 33.3 + (da_home / total_da * 25.0) + (home_score * 20.0) - (away_score * 10.0)
    base_away = 33.3 + (da_away / total_da * 25.0) + (away_score * 20.0) - (home_score * 10.0)
    draw_chance = max(5.0, 33.3 - (abs(home_score - away_score) * 15))
    
    if base_home < 5: base_home = 5.0
    if base_away < 5: base_away = 5.0
    
    total = base_home + base_away + draw_chance
    home_prob_pct = (base_home / total) * 100
    away_prob_pct = (base_away / total) * 100
    draw_prob_pct = (draw_chance / total) * 100
    
    # 7. Ambas Marcam (BTTS)
    if home_score > 0 and away_score > 0:
        btts_prob = 100.0
    else:
        btts_prob = min(85.0, max(8.0, ((da_home * da_away) / (total_da * total_da)) * 250))

    # 8. Mais de X gols da partida (Linha dinamica)
    total_gols = home_score + away_score
    linha_gols = total_gols + 0.5
    volatilidade = (da_home + da_away) * 0.5
    prob_mais_gols = min(85.0, 15.0 + volatilidade + (total_gols * 5))
        
    return {
        "vencedor_casa_prob": round(home_prob_pct, 1),
        "vencedor_fora_prob": round(away_prob_pct, 1),
        "empate_prob": round(draw_prob_pct, 1),
        "ambos_marcam_prob": round(btts_prob, 1),
        "mais_gols_linha": linha_gols,
        "mais_gols_prob": round(prob_mais_gols, 1)
    }
