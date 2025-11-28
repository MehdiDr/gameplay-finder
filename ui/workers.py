#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from config import Config
from engine import ClipSearchEngine, index_videos


class SearchWorker(QObject):
    """
    Worker asynchrone pour la recherche CLIP.
    Déporté ici pour éviter de mélanger logique et UI.
    """
    finished = Signal()
    error = Signal(str)
    results_ready = Signal(list)          # list[SearchResult]
    status_message = Signal(str, int)

    def __init__(self, engine: Optional[ClipSearchEngine], query: str, top_k: int) -> None:
        super().__init__()
        self.engine = engine
        self.query = query
        self.top_k = top_k

    @Slot()
    def run(self) -> None:
        try:
            if self.engine is None:
                self.error.emit("Aucun index chargé.")
                return

            self.status_message.emit("Recherche en cours…", 0)
            results = self.engine.search(self.query, self.top_k)
            self.results_ready.emit(results)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class IndexWorker(QObject):
    """
    Worker asynchrone pour l’indexation CLIP/FAISS.
    Copié depuis index_tab.py pour centraliser la logique.
    """
    progress_changed = Signal(int)   # 0-100
    status_changed = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            def progress_cb(current: int, total: int, message: Optional[str]):
                if self._cancelled:
                    raise RuntimeError("Indexation annulée par l'utilisateur.")
                if total <= 0:
                    percent = 0
                else:
                    percent = int(current * 100 / max(total, 1))
                self.progress_changed.emit(percent)
                if message:
                    self.status_changed.emit(message)

            self.status_changed.emit("Démarrage de l'indexation…")
            index_videos(self.cfg, progress_cb=progress_cb)
            if not self._cancelled:
                self.progress_changed.emit(100)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        self._cancelled = True
