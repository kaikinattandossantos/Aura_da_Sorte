import xgboost as xgb
import numpy as np
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# ====================== 1. CARREGA TOKEN + MODELO ======================
load_dotenv()  # ← carrega .env automaticamente

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Data" / "Copper" / "oraculo_iminencia_HIBRIDO_v1.json"

oraculo = xgb.XGBClassifier()
oraculo.load_model(str(MODEL_PATH))
print(f"✅ Modelo carregado: {MODEL_PATH.name}\n")

B365_TOKEN = os.getenv("B365_TOKEN")
if not B365_TOKEN:
    print("❌ ERRO: B365_TOKEN não encontrado no .env")
    print("   Crie um arquivo .env na raiz do projeto com a linha:")
    print("   B365_TOKEN=SEU_TOKEN_AQUI")
    exit()

# ====================== 2. FUNÇÃO TRADUTORA ======================
def preparar_input_real(jogo_api: dict, time_index: int = 0):
    stats = jogo_api.get('stats', {})
    dangerous_attacks = int(stats.get('dangerous_attacks', ["0", "0"])[time_index])
    corners = int(stats.get('corners', ["0", "0"])[time_index])

    x_simulado = 108.0
    y_simulado = 40.0
    dist = np.sqrt((120 - x_simulado)**2 + (40 - y_simulado)**2)
    a = np.sqrt((120 - x_simulado)**2 + (36 - y_simulado)**2)
    b = np.sqrt((120 - x_simulado)**2 + (44 - y_simulado)**2)
    cos_theta = np.clip((a**2 + b**2 - 64) / (2 * a * b), -1.0, 1.0)
    angulo = np.degrees(np.arccos(cos_theta))

    features = [
        x_simulado,      # x
        y_simulado,      # y
        dist,            # distancia
        angulo,          # angulo_visao
        0.75,            # pressao_10_lances
        3.5,             # aceleracao_ataque
        1.8,             # progresso_ultimo_lance
        corners,         # cantos
        dangerous_attacks # atq_perigosos
    ]
    return np.array([features])

# ====================== 3. BUSCA JOGOS AO VIVO (REAL) ======================
def get_live_matches():
    url = "https://api.b365api.com/v3/events/inplay"
    params = {
        "sport_id": 1,
        "token": B365_TOKEN
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("success") != 1:
            print("❌ API retornou erro:", data.get("error", "desconhecido"))
            return []
        results = data.get("results", [])
        print(f"📡 {len(results)} jogos ao vivo encontrados na BetsApi!")
        return results
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ 401 Unauthorized → Token inválido ou expirado")
        else:
            print(f"❌ Erro HTTP: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return []

# ====================== 4. EXECUÇÃO AO VIVO ======================
# ====================== 4. EXECUÇÃO AO VIVO (CORRIGIDO) ======================
if __name__ == "__main__":
    live_matches = get_live_matches()

    if not live_matches:
        print("\nNenhum jogo ao vivo no momento ou erro na API.")
        exit()

    print("\n" + "="*90)
    print("🔥 ORÁCULO AO VIVO - Probabilidade de Gol Iminente (BetsApi)")
    print("="*90)

    predictions = []

    for match in live_matches:
        # --- TRAVA DE SEGURANÇA: Só processa se tiver 'stats' ---
        if 'stats' not in match or not match['stats']:
            # Opcional: print(f"⚠️ Pulando {match['home']['name']} (sem estatísticas ao vivo)")
            continue

        home_name = match["home"]["name"]
        away_name = match["away"]["name"]
        league = match["league"]["name"]
        timer = match.get("timer", {}).get("tm", "?")
        ss = match.get("ss", "?-?")
        
        # Pegamos as stats com segurança para os prints
        stats = match.get('stats', {})
        da_list = stats.get('dangerous_attacks', ['0', '0'])

        # CASA
        input_home = preparar_input_real(match, 0)
        prob_home = oraculo.predict_proba(input_home)[0][1]

        # FORA
        input_away = preparar_input_real(match, 1)
        prob_away = oraculo.predict_proba(input_away)[0][1]

        print(f"🏟️ {league}")
        print(f"   ⏱️ {timer}' | {home_name} {ss} {away_name}")
        print(f"   🔥 CASA → {prob_home:.1%}  (DA: {da_list[0]})")
        print(f"   🔥 FORA → {prob_away:.1%}  (DA: {da_list[1]})")
        print("-" * 80)

        predictions.append({
            "match": f"{home_name} vs {away_name}",
            "prob_max": max(prob_home, prob_away)
        })

    # ... resto do código (TOP 5)

    # TOP 5
    print("\n🏆 TOP 5 MAIS IMINENTES (maior probabilidade)")
    sorted_preds = sorted(predictions, key=lambda x: x["prob_max"], reverse=True)
    for i, p in enumerate(sorted_preds[:5], 1):
        print(f"{i}. {p['match']} → {p['prob_max']:.1%}")

    print("\n✅ Oráculo AO VIVO funcionando!")
    print("   Rode o script novamente para atualizar os jogos ao vivo.")