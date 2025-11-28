#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import json
from math import floor
from pathlib import Path
from typing import List, Dict, Any, Tuple
import subprocess

import faiss
import torch
import numpy as np
import open_clip
import cv2
from PIL import Image, ImageTk


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "index"

    INDEX_PATH = OUTPUT_DIR / "index.faiss"
    META_PATH = OUTPUT_DIR / "metadata.json"

    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "laion2b-s34b-b79k"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True

    WINDOW_WIDTH = 1100
    WINDOW_HEIGHT = 600

    # Preview volontairement petite pour limiter le coût
    PREVIEW_MAX_WIDTH = 320
    PREVIEW_MAX_HEIGHT = 180

    # Chemin VLC (à adapter si besoin)
    VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"


def seconds_to_timecode(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_index_and_meta(cfg: Config) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not cfg.INDEX_PATH.is_file():
        raise FileNotFoundError(f"Index FAISS introuvable : {cfg.INDEX_PATH}")
    if not cfg.META_PATH.is_file():
        raise FileNotFoundError(f"Métadonnées introuvables : {cfg.META_PATH}")

    index = faiss.read_index(str(cfg.INDEX_PATH))
    with open(cfg.META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def load_clip_text_encoder(cfg: Config):
    model, _, _ = open_clip.create_model_and_transforms(
        cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
    )
    model.to(cfg.DEVICE)
    model.eval()

    if cfg.FP16 and cfg.DEVICE == "cuda":
        model = model.half()

    return model


class SearchApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.cfg = Config()

        self.master.title("Recherche gameplay (CLIP)")
        self.master.geometry(f"{self.cfg.WINDOW_WIDTH}x{self.cfg.WINDOW_HEIGHT}")
        self.master.minsize(900, 500)

        # ========= THEME SOMBRE =========
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg_main = "#1f1f23"
        bg_frame = "#25252b"
        bg_list = "#18181b"
        fg_text = "#f5f5f5"
        accent = "#3b82f6"
        accent_soft = "#111827"

        self.master.configure(bg=bg_main)
        style.configure(
            ".",
            background=bg_frame,
            foreground=fg_text,
            fieldbackground=bg_frame,
            font=("Segoe UI", 9),
        )
        style.configure("TFrame", background=bg_frame)
        style.configure("TLabel", background=bg_frame, foreground=fg_text)
        style.configure("TButton", background=accent, foreground="#f9fafb", padding=6)
        style.map("TButton", background=[("active", "#2563eb")])
        style.configure("Status.TLabel", background=accent_soft, foreground="#e5e7eb")

        # ========= STATUS =========
        status_frame = ttk.Frame(master)
        status_frame.pack(fill="x", padx=0, pady=(0, 4))

        self.status = tk.StringVar(value="Chargement du modèle et de l'index...")
        self._status_default = "Prêt."
        self._status_after_id = None

        status_label = ttk.Label(
            status_frame,
            textvariable=self.status,
            style="Status.TLabel",
            anchor="w",
        )
        status_label.pack(fill="x", padx=8, pady=4)

        master.update_idletasks()

        # ========= CHARGEMENT =========
        try:
            self.model = load_clip_text_encoder(self.cfg)
            self.index, self.metadata = load_index_and_meta(self.cfg)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger CLIP / index :\n{e}")
            master.destroy()
            return

        self.current_results: List[Tuple[int, float]] = []
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.set_status(f"Index chargé ({len(self.metadata)} images indexées).", transient=True, timeout_ms=1500)

        # ========= RECHERCHE =========
        frame_query = ttk.Frame(master)
        frame_query.pack(fill="x", padx=8, pady=(6, 4))

        label_title = ttk.Label(frame_query, text="Recherche de gameplay", font=("Segoe UI", 12, "bold"))
        label_title.pack(anchor="w", pady=(0, 4))

        label_hint = ttk.Label(
            frame_query,
            text="Tape une requête (en anglais de préférence), puis Entrée ou clique sur Rechercher.",
            font=("Segoe UI", 9),
        )
        label_hint.pack(anchor="w")

        self.entry = ttk.Entry(frame_query, font=("Segoe UI", 10))
        self.entry.pack(fill="x", pady=(4, 4))
        self.entry.bind("<Return>", lambda e: self.search())

        frame_opts = ttk.Frame(frame_query)
        frame_opts.pack(fill="x", pady=(0, 2))

        self.btn_search = ttk.Button(frame_opts, text="Rechercher", command=self.search)
        self.btn_search.pack(side="left")

        ttk.Label(frame_opts, text="  Top K :").pack(side="left", padx=(8, 0))
        self.topk_var = tk.IntVar(value=15)
        spin = ttk.Spinbox(frame_opts, from_=1, to=100, textvariable=self.topk_var, width=5)
        spin.pack(side="left")

        ttk.Label(frame_opts, text="  Filtre chemin :").pack(side="left", padx=(12, 0))
        self.filter_var = tk.StringVar(value="")
        self.filter_entry = ttk.Entry(frame_opts, textvariable=self.filter_var, width=25)
        self.filter_entry.pack(side="left", padx=(2, 0))

        self.autopreview_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(
            frame_opts,
            text="Auto preview",
            variable=self.autopreview_var,
            onvalue=True,
            offvalue=False,
            command=self.toggle_autopreview,
        )
        chk.pack(side="right")

        # ========= ZONE PRINCIPALE =========
        frame_main = ttk.Frame(master)
        frame_main.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # LISTE + BOUTONS
        frame_left = ttk.Frame(frame_main)
        frame_left.pack(side="left", fill="both", expand=True)

        frame_list = ttk.Frame(frame_left)
        frame_list.pack(side="top", fill="both", expand=True)

        bg_listbox = bg_list

        self.listbox = tk.Listbox(
            frame_list,
            width=60,
            font=("Consolas", 9),
            bg=bg_listbox,
            fg=fg_text,
            selectbackground="#374151",
            selectforeground="#f9fafb",
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        self.listbox.bind("<<ListboxSelect>>", self.on_select_result)
        self.listbox.bind("<Double-Button-1>", self.copy_selected_to_clipboard)

        # boutons sous la liste
        frame_actions = ttk.Frame(frame_left)
        frame_actions.pack(side="bottom", fill="x", pady=(4, 0))

        btn_copy = ttk.Button(frame_actions, text="Copier chemin + timecode", command=self.copy_selected_to_clipboard)
        btn_copy.pack(side="left")

        btn_vlc = ttk.Button(frame_actions, text="Ouvrir dans VLC", command=self.open_in_vlc)
        btn_vlc.pack(side="left", padx=(8, 0))

        # PREVIEW
        frame_preview = ttk.Frame(frame_main)
        frame_preview.pack(side="right", fill="both", expand=True, padx=(8, 0))

        ttk.Label(frame_preview, text="Prévisualisation", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.preview_label = ttk.Label(frame_preview, anchor="center")
        self.preview_label.pack(fill="both", expand=True, pady=(4, 4))

        self.info_label = ttk.Label(
            frame_preview,
            text="",
            anchor="w",
            justify="left",
            font=("Segoe UI", 9),
        )
        self.info_label.pack(fill="x")

        # ========= RACCOURCIS =========
        self.master.bind_all("<Control-l>", lambda e: self.entry.focus_set())
        self.master.bind_all("<Escape>", self.clear_search)

        self.set_status(self._status_default)

    # ========= STATUS =========

    def set_status(self, msg: str, transient: bool = False, timeout_ms: int = 1500):
        self.status.set(msg)
        if self._status_after_id is not None:
            self.master.after_cancel(self._status_after_id)
            self._status_after_id = None
        if transient:
            self._status_after_id = self.master.after(
                timeout_ms, lambda: self.status.set(self._status_default)
            )

    # ========= CLIP TEXTE =========

    def encode_query(self, text: str) -> np.ndarray:
        with torch.no_grad(), torch.inference_mode():
            tokens = open_clip.tokenize([text]).to(self.cfg.DEVICE)
            text_features = self.model.encode_text(tokens)
            text_features = text_features.float().cpu().numpy()
            faiss.normalize_L2(text_features)
            return text_features  # (1, D)

    # ========= RECHERCHE =========

    def search(self):
        query = self.entry.get().strip()
        if not query:
            return

        self.set_status("Recherche en cours…")
        self.master.update_idletasks()

        try:
            q_vec = self.encode_query(query)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur encodage requête :\n{e}")
            self.set_status("Erreur.")
            return

        k = max(1, int(self.topk_var.get() or 1))
        try:
            scores, idxs = self.index.search(q_vec, k)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la recherche dans l'index :\n{e}")
            self.set_status("Erreur.")
            return

        scores = scores[0]
        idxs = idxs[0]

        filter_text = self.filter_var.get().strip().lower()

        self.listbox.delete(0, tk.END)
        self.current_results = []

        for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
            if i < 0 or i >= len(self.metadata):
                continue
            meta = self.metadata[int(i)]

            path = meta.get("video_path", "unknown")
            if filter_text and filter_text not in path.lower():
                continue

            ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
            tc = seconds_to_timecode(float(ts))

            short_path = path
            if len(short_path) > 60:
                short_path = "..." + short_path[-57:]

            line = f"{rank:02d} | {tc} | {score:6.3f} | {short_path}"
            self.listbox.insert(tk.END, line)

            self.current_results.append((int(i), float(score)))

        if not self.current_results:
            self.set_status(f"Aucun résultat pour « {query} » (ou filtré).", transient=True, timeout_ms=2000)
            self.preview_label.config(image="", text="Aucun résultat")
            self.info_label.config(text="")
            return

        self.set_status(f"Recherche terminée pour « {query} ».", transient=True, timeout_ms=2000)

        # auto-sélection du premier résultat
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(0)
        self.listbox.event_generate("<<ListboxSelect>>")

    # ========= PREVIEW =========

    def on_select_result(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx_in_results = sel[0]
        if idx_in_results < 0 or idx_in_results >= len(self.current_results):
            return

        meta_index, score = self.current_results[idx_in_results]
        meta = self.metadata[meta_index]

        path = meta.get("video_path", "unknown")
        ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
        tc = seconds_to_timecode(float(ts))
        frame_index = meta.get("frame_index", None)

        if self.autopreview_var.get():
            self.load_preview_frame(path, frame_index)

        info = f"{path}\nTimecode : {tc}   |   Score : {score:.3f}"
        self.info_label.config(text=info)

    def load_preview_frame(self, video_path: str, frame_index: int | None):
        self.preview_photo = None
        self.preview_label.config(image="", text="")

        if not video_path or frame_index is None:
            self.preview_label.config(text="Aucune frame")
            return

        if not Path(video_path).is_file():
            self.preview_label.config(text="Fichier introuvable")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.preview_label.config(text="Impossible d'ouvrir la vidéo")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.preview_label.config(text="Impossible de lire la frame")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        img.thumbnail((self.cfg.PREVIEW_MAX_WIDTH, self.cfg.PREVIEW_MAX_HEIGHT), Image.BICUBIC)

        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self.preview_photo, text="")

    def toggle_autopreview(self):
        # Quand on décoche : on efface la preview
        if not self.autopreview_var.get():
            self.preview_label.config(image="", text="")
            self.preview_photo = None
        else:
            # Quand on recoche : on recharge la preview du résultat sélectionné (si il y en a un)
            sel = self.listbox.curselection()
            if not sel:
                return
            idx_in_results = sel[0]
            if 0 <= idx_in_results < len(self.current_results):
                meta_index, _ = self.current_results[idx_in_results]
                meta = self.metadata[meta_index]
                path = meta.get("video_path", "unknown")
                frame_index = meta.get("frame_index", None)
                self.load_preview_frame(path, frame_index)

    # ========= ACTIONS =========

    def get_current_meta(self):
        sel = self.listbox.curselection()
        if not sel:
            return None, None
        idx_in_results = sel[0]
        if idx_in_results < 0 or idx_in_results >= len(self.current_results):
            return None, None
        meta_index, score = self.current_results[idx_in_results]
        return self.metadata[meta_index], score

    def copy_selected_to_clipboard(self, event=None):
        meta, _ = self.get_current_meta()
        if meta is None:
            return

        path = meta.get("video_path", "unknown")
        ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
        tc = seconds_to_timecode(float(ts))

        line = f"{path} @ {tc}"

        self.master.clipboard_clear()
        self.master.clipboard_append(line)

        self.set_status("Chemin + timecode copiés dans le presse-papier.", transient=True, timeout_ms=1500)

    def open_in_vlc(self):
        meta, _ = self.get_current_meta()
        if meta is None:
            return

        path = meta.get("video_path", "unknown")
        ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))

        if not Path(path).is_file():
            messagebox.showerror("Erreur", f"Fichier introuvable :\n{path}")
            return

        vlc_path = Path(self.cfg.VLC_PATH)
        if not vlc_path.is_file():
            messagebox.showerror(
                "Erreur",
                f"VLC introuvable.\nChemin configuré :\n{self.cfg.VLC_PATH}\n\n"
                f"Modifie VLC_PATH dans le script si besoin.",
            )
            return

        try:
            subprocess.Popen([
                str(vlc_path),
                f"--start-time={int(ts)}",
                path,
            ])
            self.set_status("Ouverture dans VLC…", transient=True, timeout_ms=1500)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lancer VLC :\n{e}")

    def clear_search(self, event=None):
        self.entry.delete(0, tk.END)
        self.listbox.delete(0, tk.END)
        self.preview_label.config(image="", text="")
        self.info_label.config(text="")
        self.current_results = []
        self.set_status(self._status_default)


if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
