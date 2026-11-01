"""Chargement du jeu de données (fichiers locaux, téléchargement Kaggle ou import)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AMT, CARD, DATA_DIR, KAGGLE_HANDLE, RANDOM_STATE, TARGET, TIME_COL

KEEP = [
    TIME_COL, CARD, "merchant", "category", AMT, "gender", "city", "state", "city_pop",
    "dob", "lat", "long", "merch_lat", "merch_long", TARGET, "trans_num",
]
REQUIRED = [TIME_COL, CARD, "category", AMT, "gender", "city_pop", "dob", "lat", "long",
            "merch_lat", "merch_long", TARGET]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie les colonnes, convertit les types et trie par ordre chronologique."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {missing}")
    df = df[[c for c in KEEP if c in df.columns]].copy()
    try:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET] = df[TARGET].astype(int)
    df[CARD] = df[CARD].astype("int64")
    df[AMT] = df[AMT].astype("float64")
    return df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)


def read_csv_any(path_or_buffer) -> pd.DataFrame:
    """Lit un CSV au format Sparkov (chemin ou fichier importé)."""
    df = pd.read_csv(path_or_buffer, usecols=lambda c: c in KEEP)
    return clean(df)


def download_from_kaggle(data_dir: Path = DATA_DIR) -> list[Path]:
    """Télécharge le jeu depuis Kaggle (nécessite `pip install kagglehub` + identifiants Kaggle)."""
    import kagglehub  # import tardif : seulement si on télécharge

    data_dir.mkdir(parents=True, exist_ok=True)
    src = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    out = []
    for f in src.rglob("*.csv"):
        dst = data_dir / f.name
        shutil.copy(f, dst)
        out.append(dst)
    if not out:
        raise FileNotFoundError("Aucun CSV trouvé dans le jeu téléchargé.")
    return out


def local_files(data_dir: Path = DATA_DIR) -> list[Path]:
    main = [data_dir / "fraudTrain.csv", data_dir / "fraudTest.csv"]
    main = [p for p in main if p.exists()]
    if main:
        return main
    return sorted(p for p in data_dir.glob("*.csv") if not p.name.startswith("."))


def load_dataset(data_dir: Path = DATA_DIR, download: bool = True, max_cards: int | None = None) -> pd.DataFrame:
    """Charge fraudTrain.csv + fraudTest.csv (ou tout CSV de data/), télécharge si absent."""
    files = local_files(data_dir)
    if not files and download:
        files = download_from_kaggle(data_dir)
        files = local_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier CSV dans {data_dir}. Télécharge fraudTrain.csv et fraudTest.csv depuis "
            f"https://www.kaggle.com/datasets/{KAGGLE_HANDLE} et place-les dans le dossier data/."
        )
    df = pd.concat([read_csv_any(f) for f in files], ignore_index=True)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    if max_cards:
        cards = df[CARD].drop_duplicates().to_numpy()
        if max_cards < len(cards):
            rng = np.random.default_rng(RANDOM_STATE)
            keep = rng.choice(cards, size=max_cards, replace=False)
            df = df[df[CARD].isin(keep)].reset_index(drop=True)
    return df
