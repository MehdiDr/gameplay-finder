#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Backend CLIP / FAISS

- ClipSearchEngine : chargement modèle texte + index, recherche texte -> images.
- index_videos(cfg, progress_cb=None) : indexation d'un projet (vidéos dans cfg.OUTPUT_DIR).

Le progress_cb doit avoir la signature :
    progress_cb(current: int, total: int, message: str | None)
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import open_clip
import torch
from tqdm import tqdm

from config import Config


def log(msg: str) -> None:
    print(msg, flush=True)


# ===============================
#       RECHERCHE
# ===============================

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
        idx_path = self.cfg.INDEX_PATH
        meta_path = self.cfg.META_PATH

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


# ===============================
#       INDEXATION
# ===============================

def _find_videos(source_dir: Path, exts) -> List[Path]:
    videos: List[Path] = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(exts):
                videos.append(Path(root) / f)
    return sorted(videos)


def _compute_sources_snapshot(source_dir: Path, exts) -> List[Dict[str, Any]]:
    """
    Retourne une liste triée décrivant les vidéos dans le dossier :
    [{ "rel_path": "...", "size": ..., "mtime": ... }, ...]
    """
    sources: List[Dict[str, Any]] = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            p = Path(root) / f
            try:
                st = p.stat()
            except OSError:
                continue
            rel = p.relative_to(source_dir).as_posix()
            sources.append({
                "rel_path": rel,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            })
    sources.sort(key=lambda d: d["rel_path"])
    return sources


def _load_clip_image_model(cfg: Config):
    log("Chargement du modèle CLIP (encodeur image)…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
    )
    model.to(cfg.DEVICE)
    model.eval()

    if cfg.FP16 and cfg.DEVICE == "cuda":
        model = model.half()

    log(f"Modèle CLIP : {cfg.MODEL_NAME} ({cfg.PRETRAINED})")
    log(f"Device       : {cfg.DEVICE} (fp16={cfg.FP16})")
    return model, preprocess


def _frame_to_preprocessed_tensor(frame_bgr, preprocess) -> torch.Tensor:
    import cv2
    from PIL import Image

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return preprocess(img)  # 3 x H x W, float32


def index_videos(cfg: Config, progress_cb=None) -> None:
    """
    Indexation d'un projet.

    - Les vidéos sont cherchées dans cfg.OUTPUT_DIR.
    - L'index et les métadonnées sont écrits dans cfg.INDEX_PATH / cfg.META_PATH.
    - Un snapshot des sources est écrit dans sources_snapshot.json pour
      détecter si rien n'a changé au prochain lancement.
    """
    project_dir: Path = cfg.OUTPUT_DIR
    source_dir: Path = project_dir  # pour l'instant : vidéos dans le dossier du projet
    project_dir.mkdir(parents=True, exist_ok=True)

    index_path = cfg.INDEX_PATH
    meta_path = cfg.META_PATH
    snapshot_path = project_dir / "sources_snapshot.json"

    log("============================================================")
    log(" INDEXATION CLIP / FAISS")
    log("============================================================")
    log(f"Projet      : {cfg.CURRENT_PROJECT}")
    log(f"Bibliothèque: {cfg.LIBRARY_DIR}")
    log(f"Dossier src : {source_dir}")
    log(f"Index       : {index_path}")
    log(f"Métadonnées : {meta_path}")
    log(f"Intervalle  : {cfg.INTERVAL_SEC:.1f}s")
    log(f"Batch size  : {cfg.BATCH_SIZE}")
    log("")

    # ---------- Sources & détection de changement ----------

    current_sources = _compute_sources_snapshot(source_dir, cfg.VIDEO_EXT)
    if not current_sources:
        msg = "Aucune vidéo trouvée."
        log(msg)
        if progress_cb:
            progress_cb(100, 100, msg)
        return

    if snapshot_path.is_file():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                old_sources = json.load(f)
        except Exception:
            old_sources = None
    else:
        old_sources = None

    if old_sources == current_sources and index_path.is_file() and meta_path.is_file():
        msg = "Aucun changement détecté, index déjà à jour."
        log(msg)
        if progress_cb:
            progress_cb(100, 100, msg)
        return

    # ---------- Scan vidéos pour le comptage de frames ----------

    videos = [source_dir / s["rel_path"] for s in current_sources]
    per_video_info: List[Dict[str, Any]] = []
    total_sampled_frames = 0

    import cv2

    log("Préparation de l'indexation…")
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            log(f"[WARN] Impossible d'ouvrir la vidéo, ignorée : {v}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        if fps <= 0 or duration <= 0:
            log(f"[WARN] Vidéo invalide, ignorée : {v}")
            continue

        n_samples = int(math.floor(duration / cfg.INTERVAL_SEC)) + 1
        per_video_info.append({
            "path": v,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "n_samples": n_samples,
        })
        total_sampled_frames += n_samples

    if total_sampled_frames == 0:
        msg = "Aucune frame échantillonnée, abandon."
        log(msg)
        if progress_cb:
            progress_cb(100, 100, msg)
        return

    if cfg.MAX_FRAMES is not None:
        total_sampled_frames = min(total_sampled_frames, cfg.MAX_FRAMES)

    log(f"[Info] Frames échantillonnées totales ≈ {total_sampled_frames}")
    log("")

    if progress_cb:
        progress_cb(0, total_sampled_frames, "Extraction des frames…")

    # ---------- Modèle CLIP image ----------

    model, preprocess = _load_clip_image_model(cfg)
    feature_dim = model.visual.output_dim
    log(f"Dimension des features : {feature_dim}")
    log("")

    # ---------- Extraction + encodage ----------

    all_features: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    global_frame_counter = 0
    pbar = tqdm(total=total_sampled_frames, unit="frame", desc="Encodage")

    with torch.no_grad(), torch.inference_mode():
        batch_tensors: List[torch.Tensor] = []
        batch_meta: List[Dict[str, Any]] = []

        import cv2

        for info in per_video_info:
            vpath: Path = info["path"]
            fps = info["fps"]
            duration = info["duration"]
            n_samples = info["n_samples"]

            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                log(f"[WARN] Impossible d'ouvrir la vidéo (skip) : {vpath}")
                continue

            k = 0
            while True:
                timestamp = k * cfg.INTERVAL_SEC
                if timestamp > duration:
                    break
                if cfg.MAX_FRAMES is not None and global_frame_counter >= cfg.MAX_FRAMES:
                    break

                frame_index = int(round(timestamp * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                tensor = _frame_to_preprocessed_tensor(frame, preprocess)
                if cfg.FP16 and cfg.DEVICE == "cuda":
                    tensor = tensor.half()

                batch_tensors.append(tensor)
                batch_meta.append({
                    "video_path": str(vpath),
                    "rel_video_path": str(vpath.relative_to(source_dir)),
                    "timestamp_sec": float(timestamp),
                    "frame_index": int(frame_index),
                })

                global_frame_counter += 1
                pbar.update(1)
                if progress_cb:
                    progress_cb(global_frame_counter, total_sampled_frames, None)

                if len(batch_tensors) >= cfg.BATCH_SIZE:
                    features = _encode_batch(model, batch_tensors, cfg)
                    all_features.append(features)
                    metadata.extend(batch_meta)
                    batch_tensors = []
                    batch_meta = []

                k += 1

            cap.release()

            if cfg.MAX_FRAMES is not None and global_frame_counter >= cfg.MAX_FRAMES:
                break

        if batch_tensors:
            features = _encode_batch(model, batch_tensors, cfg)
            all_features.append(features)
            metadata.extend(batch_meta)

    pbar.close()
    log(f"[Info] Total frames encodées : {len(metadata)}")
    if not metadata:
        msg = "Aucune feature produite, abandon."
        log(msg)
        if progress_cb:
            progress_cb(100, 100, msg)
        return

    features_np = np.concatenate(all_features, axis=0).astype("float32")
    assert features_np.shape[0] == len(metadata)

    faiss.normalize_L2(features_np)

    # ---------- FAISS ----------

    log("")
    log("Construction de l'index FAISS…")
    index = faiss.IndexFlatIP(feature_dim)
    index.add(features_np)
    log(f"[Info] Index FAISS construit avec {index.ntotal} vecteurs.")

    # ---------- Sauvegarde ----------

    log("Sauvegarde de l'index et des métadonnées…")
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # snapshot
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(current_sources, f, ensure_ascii=False, indent=2)

    msg = "Indexation terminée."
    log(f"[OK] {msg}")
    if progress_cb:
        progress_cb(total_sampled_frames, total_sampled_frames, msg)


def _encode_batch(model, batch_tensors: List[torch.Tensor], cfg: Config) -> np.ndarray:
    batch = torch.stack(batch_tensors, dim=0)
    batch = batch.to(cfg.DEVICE, non_blocking=True)
    image_features: torch.Tensor = model.encode_image(batch)
    image_features = image_features.float().cpu().numpy()
    return image_features
