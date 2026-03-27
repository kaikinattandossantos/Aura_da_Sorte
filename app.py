import xgboost as xgb
import numpy as np
import requests
import os
import json
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import time

# ====================== 1. CONFIGURAÇÕES ======================
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

B365_TOKEN = os.getenv("B365_TOKEN")
GEMINI_KEY = os.getenv("HF_TOKEN")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    llm_model = genai.GenerativeModel('gemini-2.5-flash')

MODEL_PATH = BASE_DIR / "Data" / "Copper" / "oraculo_iminencia_HIBRIDO_v1.json"
model_ia = xgb.XGBClassifier()
model_ia.load_model(str(MODEL_PATH))
print(f"✅ Sistema Híbrido Ativo: {MODEL_PATH.name}")

# ====================== 2. GEMINI ======================
def gerar_comentario_aura(home, away, p_gol, da, minuto):
    if not GEMINI_KEY:
        return "IA Offline."

    prompt = (
        f"Aja como o 'Aura da Sorte', analista técnico da Esporte da Sorte. "
        f"Confronto: {home} vs {away} aos {minuto} min. "
        f"DADOS: Iminência de Gol {p_gol:.1%} | Ataques Perigosos: {da}. "
        f"Dê uma recomendação curta (máximo 15 palavras). Seja direto e use emojis. 🎯⚽"
    )

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip().replace('"', '')
    except Exception as e:
        print(f"❌ Erro na IA ({home}): {e}")
        return "Aura processando tendências... 📈"

# ====================== 3. FEATURES (HONESTAS) ======================
def processar_metricas(match, time_index):
    stats = match.get('stats', {})
    timer = int(match.get("timer", {}).get("tm", 45))

    da      = int(stats.get('dangerous_attacks', ["0", "0"])[time_index])
    corners = int(stats.get('corners',           ["0", "0"])[time_index])
    on_t    = int(stats.get('on_target',         ["0", "0"])[time_index])
    off_t   = int(stats.get('off_target',        ["0", "0"])[time_index])
    total_chutes = on_t + off_t

    # X dinâmico: quanto mais pressão, mais perto do gol simulado
    x_din = 100 + min((da / 5) + (total_chutes * 0.8), 18)
    y_sim = 40.0
    dist  = np.sqrt((120 - x_din)**2 + (40 - y_sim)**2)

    features = [
        x_din,
        y_sim,
        dist,
        45.0,                      # ângulo fixo (limitação conhecida)
        0.5 + (timer / 180),       # pressão temporal
        da / max(timer, 1),        # ritmo de ataque
        1.8,                       # placeholder (limitação conhecida)
        corners,
        da
    ]

    # Probabilidade real do modelo
    prob_gol = float(model_ia.predict_proba(np.array([features]))[0][1])

    return prob_gol, da, total_chutes

# ====================== 4. FILTRO ======================
def jogo_valido(match):
    if match.get("time_status") != "1":
        return False
    if 'stats' not in match or not match['stats']:
        return False
    timer = int(match.get("timer", {}).get("tm", 0))
    return 1 <= timer <= 100

# ====================== 5. EXECUÇÃO ======================
if __name__ == "__main__":
    print("📡 Capturando dados ao vivo...")

    try:
        r = requests.get(
            "https://api.b365api.com/v3/events/inplay",
            params={"sport_id": 1, "token": B365_TOKEN},
            timeout=10
        )
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Erro na API: {e}")
        exit(1)

    matches = [m for m in r.json().get("results", []) if jogo_valido(m)][:10]

    if not matches:
        print("⚠️ Nenhum jogo válido encontrado.")
        exit(0)

    print(f"📊 Processando {len(matches)} jogos...\n")
    final_report = []

    for match in matches:
        home = match["home"]["name"]
        away = match["away"]["name"]
        tm   = match["timer"]["tm"]

        pg_h, da_h, chutes_h = processar_metricas(match, 0)
        pg_a, da_a, chutes_a = processar_metricas(match, 1)

        # Time com maior iminência de gol é o foco
        if pg_h >= pg_a:
            p_gol, t_foco, da_foco, chutes_foco = pg_h, home, da_h, chutes_h
        else:
            p_gol, t_foco, da_foco, chutes_foco = pg_a, away, da_a, chutes_a

        comentario = "Volume ofensivo baixo, aguardando pressão."
        if p_gol > 0.65:
            comentario = gerar_comentario_aura(home, away, p_gol, da_foco, tm)
            time.sleep(2)

        item = {
            "partida": f"{home} vs {away}",
            "minuto": tm,
            "analise": {
                "prob_gol_iminente":  round(p_gol, 4),
                "time_pressionando":  t_foco,
                "ataques_perigosos":  da_foco,
                "chutes_totais":      chutes_foco
            },
            "veredito_ia": comentario
        }
        final_report.append(item)

        icon = "🚨" if p_gol > 0.8 else "📈" if p_gol > 0.65 else "🔘"
        print(f"{icon} {home} vs {away} ({tm}')")
        print(f"   ⚽ Iminência de Gol: {p_gol:.1%} | Time: {t_foco}")
        print(f"   💡 {comentario}\n" + "-"*60)

    with open("analise_aura_da_sorte.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print("✅ Relatório salvo em analise_aura_da_sorte.json")