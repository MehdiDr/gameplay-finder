#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indexation de vidéos avec CLIP + FAISS
- Extraction de frames toutes les N secondes
- Encodage avec CLIP (open_clip)
- Construction d'un index FAISS (cosine similarity)
- Sauvegarde d'un fichier index + métadonnées JSON

Dépendances :
    pip install torch torchvision open_clip_torch faiss-cpu opencv-python tqdm pillow
"""

import os
import sys
import math
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import faiss
import open_clip


# ==========================
# CONFIG
# ==========================

class Config:
    # Dossiers
    SOURCE_DIR = r"D:\CLIP_WITCHER\captures"     # HDD
    OUTPUT_DIR = r"D:\CLIP_WITCHER\index"        # index + metadata

    # Échantillonnage vidéo
    INTERVAL_SEC = 3.0                           # une frame toutes les 3 secondes
    MAX_FRAMES = None                            # limite debug (None = pas de limite)

    # CLIP / open_clip
    # MODE "FAST" : ViT-B-32 = bien plus rapide, suffisant pour ton usage
    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "laion2b-s34b-b79k"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True                                  # encode_image en fp16 sur GPU

    # Batch
    BATCH_SIZE = 256

    # Divers
    VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")
    METADATA_FILENAME = "metadata.json"
    INDEX_FILENAME = "index.faiss"
    LOG_EVERY = 1000                             # log toutes les N frames


# ==========================
# UTILITAIRES
# ==========================

def log(msg: str) -> None:
    print(msg, flush=True)


def find_videos(source_dir: str, exts=(".mp4", ".mkv", ".mov", ".avi")) -> List[Path]:
    source = Path(source_dir)
    videos = []
    for root, _, files in os.walk(source):
        for f in files:
            if f.lower().endswith(exts):
                videos.append(Path(root) / f)
    return sorted(videos)


def get_video_info(path: Path) -> Tuple[float, int, float]:
    """
    Retourne (fps, total_frames, duration_sec)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0
    cap.release()
    return fps, total, duration


def load_clip_model(cfg: Config):
    log("Chargement du modèle CLIP (open_clip)...")
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


def frame_to_preprocessed_tensor(frame_bgr: np.ndarray, preprocess) -> torch.Tensor:
    """
    Convertit une frame OpenCV (BGR, uint8) en tensor préprocessé pour CLIP.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img = Image.fromarray(frame_rgb)
    return preprocess(img)  # 3 x H x W, float32


# ==========================
# PIPELINE PRINCIPAL
# ==========================

def index_videos(cfg: Config):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    log("============================================================")
    log(" INDEXATION CLIP / FAISS (MODE FAST : ViT-B-32)")
    log("============================================================")
    log(f"Torch version : {torch.__version__}")
    log(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU : {torch.cuda.get_device_name(0)}")
    log("")
    log(f"Intervalle : {cfg.INTERVAL_SEC:.1f}s")
    log(f"Batch size : {cfg.BATCH_SIZE}")
    log(f"Source     : {cfg.SOURCE_DIR}")
    log(f"Output     : {cfg.OUTPUT_DIR}")
    log("")

    # Étape 1 : scan des vidéos
    log("Étape 1/4 : scan des vidéos sources...")
    videos = find_videos(cfg.SOURCE_DIR, cfg.VIDEO_EXT)
    if not videos:
        log("Aucune vidéo trouvée, abandon.")
        return

    log(f" -> {len(videos)} vidéo(s) détectée(s).")

    per_video_info = []
    total_sampled_frames = 0

    for v in videos:
        fps, total_frames, duration = get_video_info(v)
        if fps <= 0 or duration <= 0:
            log(f"[WARN] Vidéo invalide, skip : {v}")
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
        log("Aucune frame échantillonnée, abandon.")
        return

    if cfg.MAX_FRAMES is not None:
        total_sampled_frames = min(total_sampled_frames, cfg.MAX_FRAMES)

    log(f"[Info] Frames échantillonnées totales ≈ {total_sampled_frames}")
    log("")

    # Étape 2 : chargement du modèle
    model, preprocess = load_clip_model(cfg)
    feature_dim = model.visual.output_dim
    log(f"Dimension des features : {feature_dim}")
    log("")

    # Étape 3 : extraction + encodage des frames
    log("Étape 2/4 : extraction et encodage des frames...")
    all_features = []
    metadata: List[Dict[str, Any]] = []

    global_frame_counter = 0
    pbar = tqdm(total=total_sampled_frames, unit="frame", desc="Encodage")

    with torch.no_grad(), torch.inference_mode():
        batch_tensors: List[torch.Tensor] = []
        batch_meta: List[Dict[str, Any]] = []

        for video_idx, info in enumerate(per_video_info):
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

                tensor = frame_to_preprocessed_tensor(frame, preprocess)
                if cfg.FP16 and cfg.DEVICE == "cuda":
                    tensor = tensor.half()

                batch_tensors.append(tensor)
                batch_meta.append({
                    "video_path": str(vpath),
                    "rel_video_path": str(vpath.relative_to(cfg.SOURCE_DIR)),
                    "timestamp_sec": float(timestamp),
                    "frame_index": int(frame_index),
                })

                global_frame_counter += 1
                pbar.update(1)

                if len(batch_tensors) >= cfg.BATCH_SIZE:
                    features = encode_batch(model, batch_tensors, cfg)
                    all_features.append(features)
                    metadata.extend(batch_meta)
                    batch_tensors = []
                    batch_meta = []

                    if global_frame_counter % cfg.LOG_EVERY == 0:
                        log(f"[Info] {global_frame_counter} frames encodées...")

                k += 1

            cap.release()

            if cfg.MAX_FRAMES is not None and global_frame_counter >= cfg.MAX_FRAMES:
                break

        if batch_tensors:
            features = encode_batch(model, batch_tensors, cfg)
            all_features.append(features)
            metadata.extend(batch_meta)

    pbar.close()
    log(f"[Info] Total frames encodées : {len(metadata)}")
    if not metadata:
        log("Aucune feature produite, abandon.")
        return

    features_np = np.concatenate(all_features, axis=0).astype("float32")
    assert features_np.shape[0] == len(metadata)

    faiss.normalize_L2(features_np)

    # Étape 4 : FAISS
    log("")
    log("Étape 3/4 : construction de l'index FAISS...")
    index = faiss.IndexFlatIP(feature_dim)
    index.add(features_np)
    log(f"[Info] Index FAISS construit avec {index.ntotal} vecteurs.")

    # Sauvegarde
    log("Étape 4/4 : sauvegarde de l'index et des métadonnées...")
    index_path = Path(cfg.OUTPUT_DIR) / cfg.INDEX_FILENAME
    meta_path = Path(cfg.OUTPUT_DIR) / cfg.METADATA_FILENAME

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log(f"[OK] Index sauvegardé : {index_path}")
    log(f"[OK] Métadonnées sauvegardées : {meta_path}")
    log("Terminé.")


def encode_batch(model, batch_tensors: List[torch.Tensor], cfg: Config) -> np.ndarray:
    batch = torch.stack(batch_tensors, dim=0)
    batch = batch.to(cfg.DEVICE, non_blocking=True)
    image_features: torch.Tensor = model.encode_image(batch)
    image_features = image_features.float().cpu().numpy()
    return image_features


# ==========================
# MAIN
# ==========================

def main():
    cfg = Config()
    start = time.time()
    try:
        index_videos(cfg)
    except KeyboardInterrupt:
        log("\n[STOP] Interruption manuelle.")
    except Exception as e:
        log(f"[ERREUR] {e}")
        raise
    finally:
        elapsed = time.time() - start
        log(f"Temps total : {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()
