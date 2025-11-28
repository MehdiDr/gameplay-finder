#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py

Configuration centrale pour l'outil de recherche gameplay :
- chemins
- paramètres CLIP (open_clip)
- options d'indexation
- constantes pour l'UI
- gestion d'une bibliothèque de projets d'index
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List

import torch


class Config:
    # Dossier racine (là où se trouve ce fichier)
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Fichier de settings (bibliothèque + projet courant + projets connus)
    SETTINGS_PATH: Path = BASE_DIR / "settings.json"

    # Dossier de bibliothèque d'index (racine où se trouvent tous les projets)
    LIBRARY_DIR: Path = Path(r"D:\GameplayIndex")

    # Projet courant (chemin relatif depuis LIBRARY_DIR, peut contenir des sous-dossiers)
    # Exemple : "The Witcher" ou "Resident Evil/RE1"
    CURRENT_PROJECT: str = "The Witcher"

    # Liste des projets connus (persistance, même sans index.faiss encore présent)
    KNOWN_PROJECTS: List[str] = []

    # Dossier des captures (HDD)
    # Pour l'instant commun à tous les projets
    SOURCE_DIR: str = r"D:\CLIP_WITCHER\captures"

    # Fichiers d'index (dans le dossier du projet)
    INDEX_FILENAME: str = "index.faiss"
    METADATA_FILENAME: str = "metadata.json"

    @property
    def OUTPUT_DIR(self) -> Path:
        """Dossier d'index du projet courant."""
        return self.LIBRARY_DIR / Path(self.CURRENT_PROJECT)

    @property
    def INDEX_PATH(self) -> Path:
        return self.OUTPUT_DIR / self.INDEX_FILENAME

    @property
    def META_PATH(self) -> Path:
        return self.OUTPUT_DIR / self.METADATA_FILENAME

    # Modèle CLIP / open_clip (mode "FAST")
    MODEL_NAME: str = "ViT-B-32"
    PRETRAINED: str = "laion2b-s34b-b79k"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    FP16: bool = True

    # Vidéo / échantillonnage
    INTERVAL_SEC: float = 3.0
    MAX_FRAMES = None  # None = pas de limite

    VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")

    # Batch & logs
    BATCH_SIZE: int = 256
    LOG_EVERY: int = 1000

    # UI : preview
    PREVIEW_MAX_WIDTH: int = 640
    PREVIEW_MAX_HEIGHT: int = 360

    # VLC : à adapter si besoin
    VLC_PATH: str = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
    RESOLVE_PATH = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe"


def seconds_to_timecode(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = math.floor(sec // 3600)
    m = math.floor((sec % 3600) // 60)
    s = math.floor(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==========
# Settings : persistence bibliothèque / projet
# ==========

def load_settings(cfg: Config) -> None:
    """Charge LIBRARY_DIR, CURRENT_PROJECT et KNOWN_PROJECTS depuis settings.json si présent."""
    path = cfg.SETTINGS_PATH
    if not path.is_file():
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    lib = data.get("library_dir")
    proj = data.get("current_project")
    known = data.get("known_projects", [])

    if lib:
        cfg.LIBRARY_DIR = Path(lib)
    if proj:
        cfg.CURRENT_PROJECT = proj
    if isinstance(known, list):
        # On ne garde que des strings
        cfg.KNOWN_PROJECTS = [str(x) for x in known]


def save_settings(cfg: Config) -> None:
    """Sauvegarde LIBRARY_DIR, CURRENT_PROJECT et KNOWN_PROJECTS dans settings.json."""
    path = cfg.SETTINGS_PATH
    data = {
        "library_dir": str(cfg.LIBRARY_DIR),
        "current_project": cfg.CURRENT_PROJECT,
        "known_projects": cfg.KNOWN_PROJECTS,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # Pas dramatique si ça foire, l'app reste utilisable.
        pass


def os_walk_safe(root: Path):
    """Wrapper léger sur os.walk pour travailler avec Path."""
    for r, dirs, files in os.walk(root):
        yield r, dirs, files


def scan_projects(library_dir: Path, known_projects: List[str] | None = None) -> List[str]:
    """
    Scanne la bibliothèque et retourne la liste des projets détectés.

    Sources :
      - known_projects (settings) : projets déclarés, même sans index encore présent
      - les dossiers qui contiennent index.faiss + metadata.json

    Retourne des chemins relatifs au library_dir, au format POSIX :
        ex: "The Witcher"
            "Resident Evil/RE1"
    """
    projects_set = set()

    library_dir = Path(library_dir)

    # 1) Projets connus dans les settings (on garde seulement ceux dont le dossier existe)
    if known_projects:
        for rel in known_projects:
            rel_path = Path(rel)
            full = library_dir / rel_path
            if full.is_dir():
                projects_set.add(rel_path.as_posix())

    # 2) Dossiers contenant un index déjà existant
    if library_dir.is_dir():
        for root, dirs, files in os_walk_safe(library_dir):
            files_set = set(files)
            if "index.faiss" in files_set and "metadata.json" in files_set:
                proj_path = Path(root)
                try:
                    rel = proj_path.relative_to(library_dir)
                except ValueError:
                    continue
                projects_set.add(rel.as_posix())

    projects = sorted(projects_set)
    return projects
