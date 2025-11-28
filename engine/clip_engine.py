#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Encodeur texte CLIP + chargement FAISS pour la recherche.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import open_clip
import torch

from config import Config


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class SearchResult:
    score: float
    index: int
    meta: Dict[str, Any]


class ClipSearchEngine:
    """
    Moteur de recherche texte -> frames indexées.
    Charge :
      - index.faiss
      - metadata.json
      - modèle CLIP (encodeur texte uniquement)
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.index, self.metadata = self._load_index_and_meta()
        self.text_model = self._load_clip_text_encoder()
        self.device = cfg.DEVICE

    # ---------- Chargements ----------

    def _load_index_and_meta(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        idx_path: Path = self.cfg.INDEX_PATH
        meta_path: Path = self.cfg.META_PATH

        if not idx_path.is_file():
            raise FileNotFoundError(f"Index FAISS introuvable : {idx_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Métadonnées introuvables : {meta_path}")

        log(f"Chargement de l'index FAISS : {idx_path}")
        index = faiss.read_index(str(idx_path))

        log(f"Chargement des métadonnées : {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.ntotal != len(metadata):
            log(f"[WARN] index.ntotal={index.ntotal} != len(metadata)={len(metadata)}")

        return index, metadata

    def _load_clip_text_encoder(self):
        cfg = self.cfg
        log("Chargement du modèle CLIP (encodeur texte)…")
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

    # ---------- Recherche ----------

    def _encode_text_query(self, query: str) -> np.ndarray:
        """
        Encode une requête texte en feature CLIP normalisée (1, D).
        """
        cfg = self.cfg
        with torch.no_grad(), torch.inference_mode():
            tokens = open_clip.tokenize([query]).to(cfg.DEVICE)

            text_features = self.text_model.encode_text(tokens)

            text_features = text_features.float().cpu().numpy()
            faiss.normalize_L2(text_features)
            return text_features  # (1, D)

    @property
    def num_items(self) -> int:
        return int(self.index.ntotal)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index non chargé.")

        vec = self._encode_text_query(query)
        k = max(1, min(top_k, self.index.ntotal))

        scores, idxs = self.index.search(vec, k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[SearchResult] = []
        for i, score in zip(idxs, scores):
            if i < 0 or i >= len(self.metadata):
                continue
            meta = self.metadata[int(i)]
            results.append(SearchResult(score=float(score), index=int(i), meta=meta))
        return results
