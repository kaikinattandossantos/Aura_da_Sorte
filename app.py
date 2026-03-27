from fastapi import FastAPI, HTTPException
import xgboost as xgb
import numpy as np
import requests
import os
from pathlib import Path
from typing import Dict, Any

app = FastAPI(title="Oráculo Live Lab")

from pathlib import Path

import os
from pathlib import Path
import xgboost as xgb
from dotenv import load_dotenv
load_dotenv() # Isso lê o arquivo .env automaticamente
HF_TOKEN = os.getenv("HF_TOKEN")
# 1. Localiza a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent

# 2. LISTA DE NOMES POSSÍVEIS (Adicionei o _v1 que vi no seu print)
nomes_possiveis = [
    "oraculo_iminencia_HIBRIDO_v1.json",
    "oraculo_iminencia_HIBRIDO.json",
    "oraculo_iminencia.json"
]

MODEL_PATH = None

# 3. BUSCA AUTOMÁTICA (Varre a pasta atual e a pasta Data/Copper)
pastas_para_buscar = [
    BASE_DIR,
    BASE_DIR / "Data" / "Copper",
    BASE_DIR / "Copper",
    BASE_DIR.parent / "Data" / "Copper"
]

print("🔎 Iniciando busca pelo modelo...")
for pasta in pastas_para_buscar:
    for nome in nomes_possiveis:
        tentativa = pasta / nome
        if tentativa.exists():
            MODEL_PATH = tentativa
            break
    if MODEL_PATH: break

# 4. CARREGAMENTO COM TRAVA DE SEGURANÇA
oraculo_model = None
if MODEL_PATH:
    print(f"✅ MODELO ENCONTRADO: {MODEL_PATH}")
    try:
        oraculo_model = xgb.XGBClassifier()
        oraculo_model.load_model(str(MODEL_PATH))
        print("🧠 Inteligência Híbrida carregada com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo: {e}")
else:
    print("❌ ERRO CRÍTICO: Não achei o arquivo .json em LUGAR NENHUM!")
    print(f"Procurei em: {[str(p) for p in pastas_para_buscar]}")


async def gerar_narrativa(contexto, prob):
    if not HF_TOKEN: return f"Iminência de gol detectada: {prob:.1%}"
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = f"<s>[INST] Você é o Oráculo, um narrador místico. Comente este lance: {contexto}. Probabilidade: {prob:.1%}. [/INST]"
    try:
        res = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=5)
        return res.json()[0]['generated_text'].split("[/INST]")[-1].strip()
    except:
        return f"🚨 PERIGO IMINENTE! {prob:.1%} de chance de gol!"

@app.post("/test-live")
async def test_live(data: Dict[str, Any]):
    if oraculo_model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    try:
        # 1. Extração dos Dados do JSON da APIBet
        x = float(data.get("x", 105.0))
        y = float(data.get("y", 40.0))
        ataques = float(data.get("dominio", {}).get("ataques_perigosos", {}).get("casa", 0))
        escanteios = float(data.get("escanteios", {}).get("total", 0))

        # 2. Cálculos Geométricos
        dist = np.sqrt((120 - x)**2 + (40 - y)**2)
        # Lei dos Cossenos para o Ângulo de Visão
        a = np.sqrt((120 - x)**2 + (36 - y)**2)
        b = np.sqrt((120 - x)**2 + (44 - y)**2)
        cos_theta = np.clip((a**2 + b**2 - 8**2) / (2 * a * b), -1.0, 1.0)
        angulo = np.degrees(np.arccos(cos_theta))

        # 3. Input Híbrido (As 9 Features do seu treino)
        # Ordem: [x, y, dist, angulo, pressao, acel, progresso, cantos, ataques]
        X_input = [x, y, dist, angulo, 0.6, 4.0, 1.5, escanteios, ataques]

        # 4. Predição
        prob = float(oraculo_model.predict_proba([X_input])[:, 1][0])

        # 5. Narrativa
        texto = await gerar_narrativa(f"{ataques} ataques e {escanteios} cantos", prob)

        return {
            "status": "Oráculo Online",
            "probabilidade": round(prob, 4),
            "alerta": "CRÍTICO" if prob > 0.8 else "ALTO" if prob > 0.6 else "NORMAL",
            "narrativa": texto
        }
    except Exception as e:
        return {"erro": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Porta 8001 para não conflitar