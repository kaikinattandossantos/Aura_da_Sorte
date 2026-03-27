
import xgboost as xgb
import numpy as np
import requests
import os
import json
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# ====================== 1. CONFIGURAÇÕES ======================
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

B365_TOKEN = os.getenv("B365_TOKEN")
# Você mencionou que manteve o nome HF_TOKEN para a chave do Gemini
GEMINI_KEY = os.getenv("HF_TOKEN") 

# Configura o Gemini
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # Usando o 1.5 Flash por ser mais rápido para tempo real
    llm_model = genai.GenerativeModel('gemini-2.5-flash')

# Carregamento do Modelo XGBoost
MODEL_PATH = BASE_DIR / "Data" / "Copper" / "oraculo_iminencia_HIBRIDO_v1.json"
model_ia = xgb.XGBClassifier()
model_ia.load_model(str(MODEL_PATH))
print(f"✅ Sistema de Iminência (Aura da Sorte) Carregado: {MODEL_PATH.name}")

def gerar_comentario_tecnico(home, away, prob, da, minuto):
    if not GEMINI_KEY:
        return "Chave API não configurada."

    prompt = (
        f"Aja como o 'Aura da Sorte', analista técnico. "
        f"Jogo: {home} vs {away} aos {minuto} min. "
        f"Pressão de {da} ataques perigosos e {prob:.1%} de iminência de gol. "
        f"Dê uma recomendação técnica curta (máximo 12 palavras). 📈⚽"
    )

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # ISSO AQUI vai te dizer o erro real no terminal agora:
        print(f"❌ Erro no Gemini ({home}): {e}")
        return "Análise técnica indisponível (Erro na API)."
# ====================== 3. TRADUTOR DE DADOS ======================
def preparar_input(match, time_index):
    stats = match.get('stats', {})
    timer = int(match.get("timer", {}).get("tm", 45))
    da = int(stats.get('dangerous_attacks', ["0", "0"])[time_index])
    corners = int(stats.get('corners', ["0", "0"])[time_index])

    # Engenharia de Features Dinâmica
    x_din = 100 + min(da / 5, 18)
    y_sim = 40.0
    dist = np.sqrt((120 - x_din)**2 + (40 - y_sim)**2)
    
    features = [
        x_din, y_sim, dist, 45.0, 
        (0.5 + (timer / 180)), # Pressão
        (da / max(timer, 1)),  # Aceleração
        1.8, corners, da
    ]
    return np.array([features])

# ====================== 4. EXECUÇÃO (LIMITADA A 10 JOGOS) ======================
if __name__ == "__main__":
    print(f"📡 Buscando dados ao vivo da BetsApi...")
    r = requests.get("https://api.b365api.com/v3/events/inplay", params={"sport_id": 1, "token": B365_TOKEN})
    all_matches = r.json().get("results", [])

    # Pega apenas os 10 primeiros com estatísticas
    matches_to_process = [m for m in all_matches if 'stats' in m][:10]

    final_data = []

    print(f"📊 Analisando os {len(matches_to_process)} principais confrontos...\n")

    for match in matches_to_process:
        home = match["home"]["name"]
        away = match["away"]["name"]
        timer = match.get("timer", {}).get("tm", 0)
        da_list = match['stats'].get('dangerous_attacks', ['0', '0'])

        # Predição Real com XGBoost
        prob_h = float(model_ia.predict_proba(preparar_input(match, 0))[0][1])
        prob_a = float(model_ia.predict_proba(preparar_input(match, 1))[0][1])
        
        p_max = max(prob_h, prob_a)
        time_pressao = home if prob_h > prob_a else away
        da_p = da_list[0] if prob_h > prob_a else da_list[1]

        # Comentário com Gemini (Apenas se houver volume ofensivo real)
        comentario = "Ritmo de jogo lento, sem oportunidades claras."
        if p_max > 0.65:
            comentario = gerar_comentario_tecnico(home, away, p_max, da_p, timer)

        item = {
            "id": match.get("id"),
            "partida": f"{home} vs {away}",
            "iminencia": round(p_max, 4),
            "comentario": comentario,
            "risco": "ALTO" if p_max > 0.8 else "MEDIO" if p_max > 0.6 else "BAIXO"
        }
        final_data.append(item)
        
        icon = "🚨" if p_max > 0.8 else "📈" if p_max > 0.6 else "🔘"
        print(f"{icon} {item['partida']} | {p_max:.1%} | {comentario}")

    # Exportação Final
    with open("analise_aura_da_sorte.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Relatório 'Aura da Sorte' atualizado em analise_aura_da_sorte.json")