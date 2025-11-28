#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
from cv2 import cvtColor, COLOR_BGR2RGB

from PySide6.QtCore import (
    Qt,
    QSize,
    QPropertyAnimation,
    QEasingCurve,
    QAbstractAnimation,
    Signal,
    Slot,
    QThread,
)
from PySide6.QtGui import QPixmap, QImage, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QSplitter,
    QComboBox,
    QApplication,
    QMessageBox,
)

from config import Config, seconds_to_timecode
from engine import ClipSearchEngine, SearchResult
from resolve_bridge import write_resolve_bridge_command
from utils_resolve import is_resolve_running
from .ui_helpers import create_card, option_label, info_label, section_title
from .workers import SearchWorker
from .results_panel import ResultsPanel
from .timeline_widget import TimelineWidget


class SearchTab(QWidget):
    """
    Onglet Recherche : texte -> résultats -> preview + mini-timeline + actions (VLC / Resolve).
    """
    status_message = Signal(str, int)
    project_selected = Signal(str)       # émis quand on change de projet depuis l'onglet recherche

    def __init__(self, cfg: Config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.cfg = cfg

        self.engine: Optional[ClipSearchEngine] = None
        self.current_results: List[SearchResult] = []
        self.preview_pixmap: Optional[QPixmap] = None

        # Worker de recherche asynchrone
        self.search_thread: Optional[QThread] = None
        self.search_worker: Optional[SearchWorker] = None
        self._last_query: str = ""

        # Cache de preview : (video_path, frame_index) -> QPixmap
        self.preview_cache: Dict[Tuple[str, int], QPixmap] = {}
        self.preview_cache_max_size: int = 32

        # Cache FPS pour éviter de rouvrir sans cesse les vidéos
        self._fps_cache: Dict[str, float] = {}

        # Timestamp courant par ligne (pour coller à la mini-timeline)
        self._current_ts_for_row: Dict[int, float] = {}

        # Widgets principaux
        self.query_edit: QLineEdit | None = None
        self.search_button: QPushButton | None = None
        self.topk_spin: QSpinBox | None = None
        self.autopreview_check: QCheckBox | None = None
        self.project_combo: QComboBox | None = None

        self.preview_label: QLabel | None = None
        self.info_label: QLabel | None = None

        self.preview_container: QWidget | None = None
        self.preview_anim: QPropertyAnimation | None = None
        self.preview_expanded_width: int = 440

        self.results_panel: ResultsPanel | None = None
        self.timeline_widget: TimelineWidget | None = None

        self._build_ui()

        # Focus initial dans la barre de recherche
        if self.query_edit is not None:
            self.query_edit.setFocus()

    # ========= UI =========

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Ligne projet ---
        proj_row = QHBoxLayout()
        proj_row.setSpacing(8)

        proj_label = option_label("Projet")
        proj_row.addWidget(proj_label)

        self.project_combo = QComboBox()
        self.project_combo.setMaximumWidth(260)
        self.project_combo.setMinimumContentsLength(12)
        self.project_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.project_combo.currentIndexChanged.connect(self._on_project_changed_from_search)
        proj_row.addWidget(self.project_combo)

        proj_row.addStretch(1)

        info_proj = info_label("Gestion des dossiers → onglet Index.")
        proj_row.addWidget(info_proj)

        layout.addLayout(proj_row)

        # --- Carte recherche ---
        search_card, search_card_layout = create_card(
            object_name="SearchCard",
            parent=self,
            margins=(14, 10, 14, 10),
            spacing=8,
        )

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText(
            "Exemples : boss in the desert, snowy village at night, dialogue in tavern…"
        )
        self.query_edit.returnPressed.connect(self.on_search_clicked)
        search_row.addWidget(self.query_edit, stretch=1)

        self.search_button = QPushButton("Rechercher")
        self.search_button.setObjectName("PrimaryButton")
        self.search_button.clicked.connect(self.on_search_clicked)
        search_row.addWidget(self.search_button)

        search_card_layout.addLayout(search_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(12)

        lbl_topk = option_label("Top K")
        options_row.addWidget(lbl_topk)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(15)
        self.topk_spin.setFixedWidth(70)
        options_row.addWidget(self.topk_spin)

        options_row.addStretch()

        self.autopreview_check = QCheckBox("Auto preview")
        self.autopreview_check.setChecked(True)
        self.autopreview_check.stateChanged.connect(self.on_autopreview_toggled)
        options_row.addWidget(self.autopreview_check)

        search_card_layout.addLayout(options_row)

        layout.addWidget(search_card)

        # --- Carte principale (résultats + preview) ---
        main_card, main_card_layout = create_card(
            object_name="MainCard",
            parent=self,
            margins=(10, 10, 10, 10),
            spacing=8,
        )

        splitter = QSplitter(Qt.Horizontal)
        main_card_layout.addWidget(splitter, stretch=1)

        # Panneau gauche : résultats
        self.results_panel = ResultsPanel(self)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(0, 3)

        # Panneau droit : preview + infos + mini-timeline
        right_widget = QWidget()
        self.preview_container = right_widget
        self.preview_container.setMaximumWidth(self.preview_expanded_width)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 0, 0, 0)
        right_layout.setSpacing(8)

        right_title = section_title("Prévisualisation")
        right_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        right_layout.addWidget(right_title)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(QSize(380, 220))
        self.preview_label.setObjectName("PreviewLabel")
        right_layout.addWidget(self.preview_label, stretch=1)

        self.info_label = info_label("")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        # Mini-timeline
        self.timeline_widget = TimelineWidget(self)
        right_layout.addWidget(self.timeline_widget)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(main_card, stretch=1)

        # --- Raccourcis clavier globaux pour l'onglet ---
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.on_copy_clicked)
        QShortcut(QKeySequence("E"), self, activated=self.on_open_vlc_clicked)

        # Connexions internes
        self.results_panel.row_selected.connect(self.on_result_selected_row)
        self.results_panel.copy_requested.connect(self._on_copy_requested_from_panel)
        self.results_panel.open_vlc_requested.connect(self._on_open_vlc_requested_from_panel)
        self.results_panel.open_resolve_requested.connect(self._on_open_resolve_requested_from_panel)

        self.timeline_widget.frame_changed.connect(self.on_timeline_frame_changed)

        # État initial du panneau de preview
        if self.autopreview_check is not None:
            self.animate_preview(self.autopreview_check.isChecked())

    # ========= API publique =========

    def set_engine(self, engine: Optional[ClipSearchEngine]) -> None:
        self.engine = engine
        self._reset_results_view()
        if self.engine is None:
            self.status_message.emit("Aucun index chargé pour ce projet.", 4000)
        else:
            self.status_message.emit(
                f"Index chargé ({self.engine.num_items} images indexées).", 3000
            )

    def update_projects(self, projects: List[str], current: str) -> None:
        if self.project_combo is None:
            return
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for p in projects:
            self.project_combo.addItem(p)
        if current in projects:
            idx = projects.index(current)
            self.project_combo.setCurrentIndex(idx)
        self.project_combo.blockSignals(False)

    # ========= interne =========

    def _reset_results_view(self) -> None:
        if self.results_panel is not None:
            self.results_panel.clear()
        self.current_results = []
        self._current_ts_for_row = {}
        if self.preview_label is not None:
            self.preview_label.clear()
        if self.info_label is not None:
            self.info_label.setText("")
        self.preview_pixmap = None

    def _get_video_fps(self, path: str) -> float:
        if not path:
            return 0.0
        if path in self._fps_cache:
            return self._fps_cache[path]
        if not Path(path).is_file():
            self._fps_cache[path] = 0.0
            return 0.0
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._fps_cache[path] = 0.0
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            fps = 0.0
        self._fps_cache[path] = fps
        return fps

    def get_current_row(self) -> int:
        if self.results_panel is None:
            return -1
        return self.results_panel.current_row()

    def get_current_result(self) -> Optional[SearchResult]:
        row = self.get_current_row()
        if row < 0 or row >= len(self.current_results):
            return None
        return self.current_results[row]

    @Slot(int)
    def _on_project_changed_from_search(self, index: int) -> None:
        if index < 0 or self.project_combo is None:
            return
        name = self.project_combo.currentText().strip()
        if not name:
            return
        self._reset_results_view()
        self.project_selected.emit(name)

    # ========= Animation preview =========

    def animate_preview(self, show: bool) -> None:
        if self.preview_container is None:
            return

        if self.preview_anim is not None and \
                self.preview_anim.state() == QAbstractAnimation.Running:
            self.preview_anim.stop()

        self.preview_anim = QPropertyAnimation(self.preview_container, b"maximumWidth", self)
        self.preview_anim.setDuration(220)
        self.preview_anim.setEasingCurve(QEasingCurve.InOutCubic)

        if show:
            self.preview_container.setVisible(True)
            self.preview_anim.setStartValue(0)
            self.preview_anim.setEndValue(self.preview_expanded_width)
        else:
            self.preview_anim.setStartValue(self.preview_container.width())
            self.preview_anim.setEndValue(0)

            def hide():
                self.preview_container.setVisible(False)

            self.preview_anim.finished.connect(hide)

        self.preview_anim.start()

    # ========= Recherche (asynchrone) =========

    @Slot()
    def on_search_clicked(self) -> None:
        query = self.query_edit.text().strip() if self.query_edit else ""
        self._reset_results_view()

        if not query:
            return
        if self.engine is None:
            self.status_message.emit("Aucun index chargé pour ce projet.", 4000)
            return

        if self.search_thread is not None:
            self.status_message.emit("Recherche déjà en cours…", 2000)
            return

        k = max(1, int(self.topk_spin.value())) if self.topk_spin else 10
        self._last_query = query

        self.search_thread = QThread(self)
        self.search_worker = SearchWorker(self.engine, query, k)
        self.search_worker.moveToThread(self.search_thread)

        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.results_ready.connect(self._on_search_results)
        self.search_worker.error.connect(self._on_search_error)
        self.search_worker.status_message.connect(self.status_message)
        self.search_worker.finished.connect(self._on_search_finished)

        self.search_worker.finished.connect(self.search_thread.quit)
        self.search_thread.finished.connect(self.search_thread.deleteLater)

        if self.search_button is not None:
            self.search_button.setEnabled(False)
        if self.query_edit is not None:
            self.query_edit.setEnabled(False)

        self.search_thread.start()

    # ========= UI des résultats : liste texte lisible =========

    @Slot(list)
    def _on_search_results(self, results: list) -> None:
        """
        results : list[SearchResult]
        """
        self.current_results = []
        self._current_ts_for_row = {}

        if self.results_panel is not None:
            self.results_panel.clear()

        lines: List[str] = []
        tooltips: List[str] = []

        for idx, result in enumerate(results):
            meta = result.meta or {}
            path = meta.get("video_path", "unknown")
            rel = meta.get("rel_video_path", Path(path).name)

            ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))
            tc = seconds_to_timecode(ts)

            display_path = rel
            if len(display_path) > 70:
                display_path = "…" + display_path[-67:]

            line = f"{tc:>8}   {result.score:5.3f}   {display_path}"
            lines.append(line)
            tooltips.append(path)

            self.current_results.append(result)
            self._current_ts_for_row[idx] = ts

        if self.results_panel is not None:
            self.results_panel.set_items(lines, tooltips)

        if not self.current_results:
            self.status_message.emit(f"Aucun résultat pour « {self._last_query} ».", 3000)
        else:
            self.status_message.emit(
                f"Recherche terminée pour « {self._last_query} ». "
                f"{len(self.current_results)} résultats.",
                2500,
            )
            if self.results_panel is not None:
                self.results_panel.set_current_row(0)
                self.results_panel.focus_list()

    @Slot(str)
    def _on_search_error(self, message: str) -> None:
        self.status_message.emit(f"Erreur de recherche : {message}", 4000)

    @Slot()
    def _on_search_finished(self) -> None:
        if self.search_button is not None:
            self.search_button.setEnabled(True)
        if self.query_edit is not None:
            self.query_edit.setEnabled(True)

        self.search_worker = None
        self.search_thread = None

    # ========= Résultats & preview =========

    @Slot(int)
    def on_result_selected_row(self, row: int) -> None:
        if row < 0 or row >= len(self.current_results):
            return

        result = self.current_results[row]
        meta = result.meta or {}

        path = meta.get("video_path", "unknown")
        ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))
        tc = seconds_to_timecode(ts)
        frame_index = meta.get("frame_index", None)

        # FPS pour la mini-timeline
        fps = self._get_video_fps(path)
        if self.timeline_widget is not None:
            self.timeline_widget.set_context(
                video_path=path,
                fps=fps,
                center_frame=int(frame_index) if frame_index is not None else 0,
                center_ts=ts,
                interval_sec=self.cfg.INTERVAL_SEC,
            )

        if self.autopreview_check.isChecked():
            self.load_preview_frame(path, frame_index)

        info = f"{path}\nTimecode : {tc}   |   Score : {result.score:.3f}"
        self.info_label.setText(info)

    def load_preview_frame(self, video_path: str, frame_index: int | None) -> None:
        self.preview_pixmap = None
        self.preview_label.clear()

        if not video_path or frame_index is None:
            self.preview_label.setText("Aucune frame")
            return

        key = (video_path, int(frame_index))

        # Cache hit
        if key in self.preview_cache:
            pix = self.preview_cache[key]
            self.preview_pixmap = pix
            self.preview_label.setPixmap(self.preview_pixmap)
            return

        if not Path(video_path).is_file():
            self.preview_label.setText("Fichier vidéo introuvable")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.preview_label.setText("Impossible d'ouvrir la vidéo")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.preview_label.setText("Impossible de lire la frame")
            return

        frame = cvtColor(frame, COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        pix = pix.scaled(
            self.cfg.PREVIEW_MAX_WIDTH,
            self.cfg.PREVIEW_MAX_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview_pixmap = pix
        self.preview_label.setPixmap(self.preview_pixmap)

        # Mise à jour du cache
        self.preview_cache[key] = pix
        if len(self.preview_cache) > self.preview_cache_max_size:
            old_key = next(iter(self.preview_cache.keys()))
            if old_key in self.preview_cache:
                del self.preview_cache[old_key]

    @Slot(int)
    def on_autopreview_toggled(self, state: int) -> None:
        enabled = self.autopreview_check.isChecked()
        self.animate_preview(enabled)

        if not enabled:
            self.preview_label.clear()
            self.preview_pixmap = None
        else:
            row = self.get_current_row()
            if 0 <= row < len(self.current_results):
                result = self.current_results[row]
                meta = result.meta or {}
                path = meta.get("video_path", "unknown")
                frame_index = meta.get("frame_index", None)
                self.load_preview_frame(path, frame_index)

    # ========= Mini-timeline =========

    @Slot(int, float)
    def on_timeline_frame_changed(self, frame_index: int, ts: float) -> None:
        """
        Appelé quand l'utilisateur bouge le slider de mini-timeline.
        On met à jour la preview + le timecode courant pour la ligne sélectionnée.
        """
        row = self.get_current_row()
        if row < 0 or row >= len(self.current_results):
            return

        result = self.current_results[row]
        meta = result.meta or {}
        path = meta.get("video_path", "unknown")
        if not path:
            return

        self._current_ts_for_row[row] = ts

        # Preview
        self.load_preview_frame(path, frame_index)

        tc = seconds_to_timecode(ts)
        info = f"{path}\nTimecode : {tc}   |   Score : {result.score:.3f}"
        self.info_label.setText(info)

    # ========= Utilitaires =========

    @Slot()
    def on_copy_clicked(self) -> None:
        row = self.get_current_row()
        if row < 0 or row >= len(self.current_results):
            return

        result = self.current_results[row]
        meta = result.meta or {}
        path = meta.get("video_path", "unknown")
        base_ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))
        ts = self._current_ts_for_row.get(row, base_ts)
        tc = seconds_to_timecode(ts)

        line = f"{path} @ {tc}"
        QApplication.clipboard().setText(line)
        self.status_message.emit(
            "Chemin + timecode copiés dans le presse-papier.", 2000
        )

    @Slot()
    def on_open_vlc_clicked(self) -> None:
        row = self.get_current_row()
        if row < 0 or row >= len(self.current_results):
            return

        result = self.current_results[row]
        meta = result.meta or {}
        path = meta.get("video_path", "unknown")
        base_ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))
        ts = float(self._current_ts_for_row.get(row, base_ts))

        if not Path(path).is_file():
            self.status_message.emit("Fichier vidéo introuvable.", 3000)
            return

        vlc_path = Path(self.cfg.VLC_PATH)
        if not vlc_path.is_file():
            self.status_message.emit(
                "VLC introuvable. Modifie VLC_PATH dans config.py.", 4000
            )
            return

        try:
            subprocess.Popen([
                str(vlc_path),
                f"--start-time={int(ts)}",
                path,
            ])
            self.status_message.emit("Ouverture dans VLC…", 2000)
        except Exception as e:
            self.status_message.emit(
                "Erreur lors du lancement de VLC.", 3000
            )
            print("Erreur VLC :", e)

    @Slot()
    def on_open_in_resolve_clicked(self) -> None:
        """
        Slot appelé quand on clique sur le bouton 'Ouvrir dans Resolve'.
        Récupère le SearchResult sélectionné et envoie la commande à Resolve.
        """
        row = self.get_current_row()
        if row < 0 or row >= len(self.current_results):
            QMessageBox.warning(
                self,
                "Aucun résultat",
                "Veuillez sélectionner un extrait dans la liste avant d'ouvrir dans Resolve.",
            )
            return

        if not is_resolve_running():
            QMessageBox.warning(
                self,
                "Resolve non détecté",
                "DaVinci Resolve ne semble pas être lancé.\n"
                "Lance Resolve et ouvre ton projet / timeline avant d'utiliser cette fonction.",
            )
            return

        result = self.current_results[row]
        meta = getattr(result, "meta", None) or {}
        video_path = meta.get("video_path") or meta.get("path")
        base_ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))
        timestamp_sec = float(self._current_ts_for_row.get(row, base_ts))

        if not video_path:
            QMessageBox.warning(
                self,
                "Données manquantes",
                "Impossible de récupérer le chemin de la vidéo pour ce résultat.",
            )
            return

        write_resolve_bridge_command(video_path, timestamp_sec)

        self.status_message.emit(
            "Commande envoyée à DaVinci Resolve (insertion dans la timeline courante).",
            3000,
        )

    # ========= Callbacks depuis le panneau de résultats =========

    @Slot(int)
    def _on_copy_requested_from_panel(self, row: int) -> None:
        if self.results_panel is not None:
            self.results_panel.set_current_row(row)
        self.on_copy_clicked()

    @Slot(int)
    def _on_open_vlc_requested_from_panel(self, row: int) -> None:
        if self.results_panel is not None:
            self.results_panel.set_current_row(row)
        self.on_open_vlc_clicked()

    @Slot(int)
    def _on_open_resolve_requested_from_panel(self, row: int) -> None:
        if self.results_panel is not None:
            self.results_panel.set_current_row(row)
        self.on_open_in_resolve_clicked()
