import kagglehub
import os
import json
import pandas as pd

# ==================== DOWNLOAD / PATHS ====================
path_fb = kagglehub.dataset_download("hubertsidorowicz/football-players-stats-2025-2026")
path_sb = kagglehub.dataset_download("saurabhshahane/statsbomb-football-data")

print("✅ FBref carregado:", os.listdir(path_fb))
print("✅ StatsBomb root:", os.listdir(path_sb))

# ==================== EXPLORA ESTRUTURA StatsBomb ====================
print("\n=== ESTRUTURA StatsBomb ===")
for root, dirs, files in os.walk(path_sb):
    level = root.replace(path_sb, "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    if files and level <= 2:  # mostra só os primeiros níveis
        for f in files[:5]:
            print(f"{indent}    {f}")

# ==================== CARREGA FBref (já funcionando) ====================
df_players = pd.read_csv(os.path.join(path_fb, "players_data-2025_2026.csv"))
print(f"\nFBref: {len(df_players)} jogadores | Colunas: {list(df_players.columns[:15])}...")

# ==================== EXEMPLO: carrega UMA partida histórica (StatsBomb) ====================
# Escolha um match_id famoso (ex: final Champions ou jogo recente)
match_id = "3916"  # exemplo: você pode trocar depois de ver a pasta matches/

events_path = os.path.join(path_sb, "events", f"{match_id}.json")
if os.path.exists(events_path):
    with open(events_path, "r") as f:
        events = json.load(f)
    df_events = pd.DataFrame(events)
    print(f"\n✅ Partida {match_id} carregada: {len(df_events)} eventos")
    print("Tipos de eventos:", df_events["type"].value_counts().head())
    print(df_events.head(3)[["type", "player", "team", "minute"]])
else:
    print(f"Match {match_id} não encontrado. Rode o os.walk e me manda o output!")