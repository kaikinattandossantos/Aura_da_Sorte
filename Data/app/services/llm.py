import os
import requests
from typing import Any

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"


def _build_fallback(ctx: dict) -> str:
    """Gera uma narrativa dinâmica sem LLM baseada nos dados reais da partida."""
    prob      = ctx.get("prob", 0.0)
    nivel     = ctx.get("nivel", "NORMAL")
    score     = ctx.get("score", "0-0")
    timer     = ctx.get("timer", 45)
    da_home   = ctx.get("da_home", 0)
    da_away   = ctx.get("da_away", 0)
    poss_home = ctx.get("poss_home", 50)
    poss_away = ctx.get("poss_away", 50)
    ot_home   = ctx.get("ot_home", 0)
    ot_away   = ctx.get("ot_away", 0)
    yc_total  = ctx.get("yc_home", 0) + ctx.get("yc_away", 0)

    dominant  = "casa" if da_home >= da_away else "visitante"
    dominant_da = max(da_home, da_away)
    dominant_poss = poss_home if da_home >= da_away else poss_away

    if nivel == "CRÍTICO":
        base = f"⚡ IMINÊNCIA MÁXIMA ({prob:.0%}) — time da {dominant} com {dominant_da} ataques perigosos e {dominant_poss}% de posse no {timer}'"
    elif nivel == "ALTO":
        base = f"🔥 Pressão elevada ({prob:.0%}) — {dominant_da} ataques do time da {dominant}, placar {score} no {timer}'"
    else:
        base = f"📊 Partida controlada ({prob:.0%} de gol) — {dominant_da} ataques do time da {dominant}, {dominant_poss}% de posse no {timer}'"

    extras = []
    if ot_home + ot_away > 4:
        extras.append(f"{ot_home + ot_away} chutes ao gol na partida")
    if yc_total >= 3:
        extras.append(f"{yc_total} cartões amarelos acumulados")
    if abs(poss_home - poss_away) > 20:
        extras.append(f"posse desequilibrada ({poss_home}% x {poss_away}%)")

    if extras:
        base += " | " + ", ".join(extras)

    return base


def _build_prompt(ctx: dict) -> str:
    prob      = ctx.get("prob", 0.0)
    nivel     = ctx.get("nivel", "NORMAL")
    score     = ctx.get("score", "0-0")
    timer     = ctx.get("timer", 45)
    da_home   = ctx.get("da_home", 0)
    da_away   = ctx.get("da_away", 0)
    poss_home = ctx.get("poss_home", 50)
    poss_away = ctx.get("poss_away", 50)
    corners_h = ctx.get("corners_home", 0)
    corners_a = ctx.get("corners_away", 0)
    ot_home   = ctx.get("ot_home", 0)
    ot_away   = ctx.get("ot_away", 0)
    yc_home   = ctx.get("yc_home", 0)
    yc_away   = ctx.get("yc_away", 0)
    rc_home   = ctx.get("rc_home", 0)
    rc_away   = ctx.get("rc_away", 0)

    return (
        f"<s>[INST] Você é um analista de futebol ao vivo especializado em apostas esportivas. "
        f"Analise os dados abaixo e gere UMA frase direta, específica e impactante sobre o momento da partida "
        f"e sua principal previsão. Use os números reais. Responda SOMENTE a frase, sem introdução.\n\n"
        f"DADOS DA PARTIDA ({timer}' | Placar: {score} | Nível de risco: {nivel}):\n"
        f"- Probabilidade de gol (Oráculo IA): {prob:.1%}\n"
        f"- Ataques perigosos: Casa {da_home} × Visitante {da_away}\n"
        f"- Posse de bola: Casa {poss_home}% × Visitante {poss_away}%\n"
        f"- Chutes ao gol: Casa {ot_home} × Visitante {ot_away}\n"
        f"- Escanteios: Casa {corners_h} × Visitante {corners_a}\n"
        f"- Cartões amarelos: Casa {yc_home} × Visitante {yc_away}\n"
        f"- Cartões vermelhos: Casa {rc_home} × Visitante {rc_away}\n"
        f" [/INST]"
    )


async def gerar_narrativa_oraculo(ctx: dict) -> dict:
    """
    ctx deve conter: prob, nivel, score, timer, da_home, da_away,
    poss_home, poss_away, corners_home, corners_away, ot_home, ot_away,
    yc_home, yc_away, rc_home, rc_away.
    """
    if not HF_TOKEN:
        return {"summary": _build_fallback(ctx), "source": "fallback"}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = _build_prompt(ctx)

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=8)
        payload = response.json()
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            text = payload[0]["generated_text"].split("[/INST]")[-1].strip()
            if text:
                return {"summary": text, "source": "llm"}
        raise ValueError("Resposta inesperada")
    except Exception:
        return {"summary": _build_fallback(ctx), "source": "fallback"}


async def gerar_narrativa_cartao(contexto: str, probabilidade: float, tipo_cartao: str) -> dict:
    if not HF_TOKEN:
        return {"summary": f"Risco de cartão {tipo_cartao}: {probabilidade:.1%}. {contexto}.", "source": "fallback"}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = (
        f"<s>[INST] Você é um analista de futebol. Gere 1 frase curta e objetiva sobre risco de cartão {tipo_cartao}. "
        f"Contexto: {contexto}. Probabilidade: {probabilidade:.1%}. Responda SOMENTE a frase. [/INST]"
    )

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=8)
        payload = response.json()
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            text = payload[0]["generated_text"].split("[/INST]")[-1].strip()
            if text:
                return {"summary": text, "source": "llm"}
        raise ValueError("Resposta inesperada")
    except Exception:
        return {"summary": f"Risco de cartão {tipo_cartao} em {probabilidade:.1%}. {contexto}.", "source": "fallback"}
