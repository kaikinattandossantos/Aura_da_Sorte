import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

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
