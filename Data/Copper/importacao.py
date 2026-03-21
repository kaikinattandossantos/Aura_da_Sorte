import kagglehub
from kagglehub import KaggleDatasetAdapter

import os
import glob
import json
import pandas as pd

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
# O dataset do StatsBomb nesta fonte vem principalmente em JSON (pasta data/events)
try:
    # Tenta o caminho CSV (caso exista em alguma versão alternativa)
    file_path_sb = "events.csv"
    df_sb = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "saurabhshahane/statsbomb-football-data",
        file_path_sb
    )
    print("\nStatsBomb (CSV) - Primeiros 5 registros:\n", df_sb.head())
except Exception:
    try:
        print("\n'events.csv' não encontrado. Procurando JSONs de eventos...")
        event_files = glob.glob(os.path.join(path_sb, "**", "events", "*.json"), recursive=True)
        print(f"Arquivos de eventos encontrados: {len(event_files)}")

        if not event_files:
            raise FileNotFoundError("Nenhum JSON de eventos encontrado em data/events.")

        # Carrega uma amostra inicial para inspeção rápida
        max_files = 10
        records = []
        for fp in event_files[:max_files]:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)

        if not records:
            raise ValueError("JSONs encontrados, mas sem registros de eventos válidos.")

        df_sb = pd.json_normalize(records)
        print(f"\nStatsBomb (JSON) carregado com sucesso -> linhas: {len(df_sb)}, colunas: {df_sb.shape[1]}")
        print("Primeiras colunas:", df_sb.columns[:12].tolist())
        print("\nPrimeiros 5 registros:\n", df_sb.head())
    except Exception as e:
        print(f"Erro ao carregar StatsBomb: {e}")