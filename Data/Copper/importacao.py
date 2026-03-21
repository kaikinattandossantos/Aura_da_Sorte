import kagglehub
from kagglehub import KaggleDatasetAdapter
import kagglehub
import os

# Forçar o download para o diretório padrão e listar arquivos
try:
    print("Baixando Player Stats...")
    path_stats = kagglehub.dataset_download("hubertsidorowicz/football-players-stats-2025-2026")
    print(f"Arquivos em Stats: {os.listdir(path_stats)}")

    print("\nBaixando StatsBomb...")
    path_sb = kagglehub.dataset_download("saurabhshahane/statsbomb-football-data")
    print(f"Arquivos em StatsBomb: {os.listdir(path_sb)}")
except Exception as e:
    print(f"Erro no download: {e}")

# 2. Carregando Dados do StatsBomb
# Geralmente o StatsBomb no Kaggle tem arquivos .csv ou .json
file_path_sb = "events.csv" 

try:
    df_sb = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "saurabhshahane/statsbomb-football-data",
        file_path_sb
    )
    print("\nStatsBomb - Primeiros 5 registros:\n", df_sb.head())
except Exception as e:
    print(f"Erro ao carregar StatsBomb: {e}")