#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recherche texte -> vidéo dans l'index CLIP/FAISS généré par index_video.py.

Usage :
    python search_video.py
    puis taper des requêtes (de préférence en anglais).
"""

import json
from math import floor
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import torch
import numpy as np
import open_clip


class Config:
    # Dossier où index_video.py écrit ses fichiers
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "index"

    INDEX_PATH = OUTPUT_DIR / "index.faiss"
    META_PATH = OUTPUT_DIR / "metadata.json"

    # Même modèle que dans index_video.py
    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "laion2b-s32b-b82k"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True


def log(msg: str) -> None:
    print(msg, flush=True)


def seconds_to_timecode(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = floor(sec // 3600)
    m = floor((sec % 3600) // 60)
    s = floor(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_index_and_meta(cfg: Config) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not cfg.INDEX_PATH.is_file():
        raise FileNotFoundError(f"Index FAISS introuvable : {cfg.INDEX_PATH}")
    if not cfg.META_PATH.is_file():
        raise FileNotFoundError(f"Métadonnées introuvables : {cfg.META_PATH}")

    log(f"Chargement de l'index FAISS : {cfg.INDEX_PATH}")
    index = faiss.read_index(str(cfg.INDEX_PATH))

    log(f"Chargement des métadonnées : {cfg.META_PATH}")
    with open(cfg.META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        log(f"[WARN] index.ntotal={index.ntotal} != len(metadata)={len(metadata)}")

    return index, metadata


def load_clip_text_encoder(cfg: Config):
    log("Chargement du modèle CLIP (open_clip)...")
    model, _, _ = open_clip.create_model_and_transforms(
        cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
    )
    model.to(cfg.DEVICE)
    model.eval()

    if cfg.FP16 and cfg.DEVICE == "cuda":
        model = model.half()

    log(f"Modèle CLIP : {cfg.MODEL_NAME} ({cfg.PRETRAINED})")
    log(f"Device       : {cfg.DEVICE} (fp16={cfg.FP16})")
    return model


def encode_text_query(model, cfg: Config, query: str) -> np.ndarray:
    """
    Encode une requête texte en feature CLIP normalisée (1, D).
    """
    with torch.no_grad(), torch.inference_mode():
        # Les tokens restent en int64, on ne les convertit surtout PAS en half
        tokens = open_clip.tokenize([query]).to(cfg.DEVICE)

        text_features = model.encode_text(tokens)

        # -> (1, D), repasse en float32 CPU
        text_features = text_features.float().cpu().numpy()
        # Normalisation L2 pour cosine similarity (comme l'index)
        faiss.normalize_L2(text_features)
        return text_features  # shape (1, D)


def main():
    cfg = Config()

    index, metadata = load_index_and_meta(cfg)
    model = load_clip_text_encoder(cfg)

    print("\n=== Moteur de recherche gameplay (texte -> images) ===")
    print("Utilise de préférence des requêtes en anglais (le modèle est entraîné dessus).")
    print("Exemples :")
    print("  fight in the snow")
    print("  Geralt on horseback")
    print("  tavern at night")
    print("Tape 'quit' pour sortir.\n")

    while True:
        query = input("Requête > ").strip()
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            break

        q_vec = encode_text_query(model, cfg, query)

        k = 10
        scores, idxs = index.search(q_vec, k)
        scores = scores[0]
        idxs = idxs[0]

        print(f"\nTop {k} résultats pour '{query}' :")
        for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
            if i < 0 or i >= len(metadata):
                continue

            meta = metadata[int(i)]
            video_path = meta.get("video_path", "unknown")

            # selon la version de metadata, on peut avoir timestamp_sec ou autre
            if "timecode" in meta:
                tc = meta["timecode"]
            else:
                ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
                tc = seconds_to_timecode(float(ts))

            print(f"{rank:02d}. {video_path} @ {tc} (score: {score:.3f})")

        print("")


if __name__ == "__main__":
    main()
