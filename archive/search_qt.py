#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
from cv2 import cvtColor, COLOR_BGR2RGB

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QFrame,
    QTabWidget,
    QComboBox,
)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QAbstractAnimation
from PySide6.QtGui import QPixmap, QImage, QFont

from config import Config, seconds_to_timecode
from engine import ClipSearchEngine, SearchResult
from index_tab import IndexTab


class SearchWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.cfg = Config()
        self.engine: Optional[ClipSearchEngine] = None
        self.current_results: List[SearchResult] = []
        self.preview_pixmap: Optional[QPixmap] = None

        # Widgets recherche
        self.query_edit: QLineEdit | None = None
        self.search_button: QPushButton | None = None
        self.topk_spin: QSpinBox | None = None
        self.autopreview_check: QCheckBox | None = None
        self.results_list: QListWidget | None = None
        self.copy_button: QPushButton | None = None
        self.vlc_button: QPushButton | None = None
        self.preview_label: QLabel | None = None
        self.info_label: QLabel | None = None

        self.project_combo_search: QComboBox | None = None

        self.preview_container: QWidget | None = None
        self.preview_anim: QPropertyAnimation | None = None
        self.preview_expanded_width: int = 420

        self.index_tab: IndexTab | None = None

        # Fenêtre
        self.setWindowTitle("Gameplay Finder (CLIP)")
        self.resize(1200, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)

        # ===== HEADER =====
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title = QLabel("Gameplay Finder")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setObjectName("MainTitle")
        header_layout.addWidget(title)

        subtitle = QLabel("Recherche visuelle ultra-rapide dans tes captures (CLIP + FAISS).")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setObjectName("Subtitle")
        header_layout.addWidget(subtitle)

        proj_row = QHBoxLayout()
        proj_row.setSpacing(8)

        proj_label = QLabel("Projet")
        proj_label.setObjectName("OptionLabel")
        proj_row.addWidget(proj_label)

        self.project_combo_search = QComboBox()
        self.project_combo_search.setMaximumWidth(260)
        self.project_combo_search.setMinimumContentsLength(12)
        self.project_combo_search.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        proj_row.addWidget(self.project_combo_search)

        info_proj = QLabel("Gestion des dossiers dans l’onglet Index.")
        info_proj.setObjectName("InfoLabel")
        proj_row.addWidget(info_proj)

        proj_row.addStretch(1)
        header_layout.addLayout(proj_row)

        main_layout.addLayout(header_layout)

        # ===== TABS =====
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # --- Onglet Recherche ---
        search_tab = QWidget()
        search_tab_layout = QVBoxLayout(search_tab)
        search_tab_layout.setContentsMargins(0, 0, 0, 0)
        search_tab_layout.setSpacing(8)

        # Carte recherche
        search_card = QFrame()
        search_card.setObjectName("SearchCard")
        search_card_layout = QVBoxLayout(search_card)
        search_card_layout.setContentsMargins(12, 10, 12, 10)
        search_card_layout.setSpacing(8)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText(
            "Exemples : fight in the snow, Geralt on horseback, tavern at night…"
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

        lbl_topk = QLabel("Top K")
        lbl_topk.setObjectName("OptionLabel")
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

        search_tab_layout.addWidget(search_card)

        # Carte principale (résultats + preview)
        main_card = QFrame()
        main_card.setObjectName("MainCard")
        main_card_layout = QVBoxLayout(main_card)
        main_card_layout.setContentsMargins(10, 10, 10, 10)
        main_card_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_card_layout.addWidget(splitter, stretch=1)

        # Résultats à gauche
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.results_list = QListWidget()
        self.results_list.setObjectName("ResultsList")
        self.results_list.setFont(QFont("Consolas", 9))
        self.results_list.currentRowChanged.connect(self.on_result_selected)
        self.results_list.itemDoubleClicked.connect(self.on_copy_clicked)
        left_layout.addWidget(self.results_list, stretch=1)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)
        left_layout.addLayout(buttons_row)

        self.copy_button = QPushButton("Copier chemin + timecode")
        self.copy_button.clicked.connect(self.on_copy_clicked)
        buttons_row.addWidget(self.copy_button)

        self.vlc_button = QPushButton("Ouvrir dans VLC")
        self.vlc_button.clicked.connect(self.on_open_vlc_clicked)
        buttons_row.addWidget(self.vlc_button)

        buttons_row.addStretch()

        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 3)

        # Preview à droite
        right_widget = QWidget()
        self.preview_container = right_widget
        self.preview_container.setMaximumWidth(self.preview_expanded_width)

        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        right_title = QLabel("Prévisualisation")
        right_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        right_title.setObjectName("SectionTitle")
        right_layout.addWidget(right_title)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(QSize(360, 200))
        self.preview_label.setObjectName("PreviewLabel")
        right_layout.addWidget(self.preview_label, stretch=1)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("InfoLabel")
        right_layout.addWidget(self.info_label)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        search_tab_layout.addWidget(main_card, stretch=1)

        self.tabs.addTab(search_tab, "Recherche")

        # --- Onglet Index ---
        self.index_tab = IndexTab(self.cfg)
        self.tabs.addTab(self.index_tab, "Index")

        # Status bar
        status = QStatusBar()
        self.setStatusBar(status)

        # Style
        self.apply_dark_style()

        # Focus
        self.query_edit.setFocus()

        # Connexions Index <-> Recherche
        self.index_tab.engine_changed.connect(self._on_engine_changed)
        self.index_tab.status_message.connect(self._on_status_message)
        self.index_tab.projects_changed.connect(self._on_projects_changed)

        self.project_combo_search.currentIndexChanged.connect(
            self._on_project_selected_from_search
        )

        # Chargement backend initial
        self.statusBar().showMessage("Chargement du modèle et de l'index…")
        QApplication.processEvents()

        try:
            self.engine = ClipSearchEngine(self.cfg)
            self.index_tab.set_engine(self.engine)
            self.statusBar().showMessage(
                f"Index chargé ({self.engine.num_items} images indexées).", 3000
            )
        except Exception as e:
            self.engine = None
            self.index_tab.set_engine(None)
            self.statusBar().showMessage("Aucun index chargé (lance une indexation).", 5000)
            print("Impossible de charger CLIP / index au démarrage :", e)

    # ====== STYLE ======

    def apply_dark_style(self) -> None:
        self.setStyleSheet("""
        QMainWindow {
            background-color: #111111;
        }
        QWidget {
            background-color: #111111;
            color: #F2F2F7;
            font-family: "Segoe UI", sans-serif;
            font-size: 10pt;
        }

        QTabWidget::pane {
            border: none;
        }
        QTabBar::tab {
            background-color: #1C1C1E;
            color: #C7C7CC;
            padding: 6px 12px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #2C2C2E;
            color: #FFFFFF;
        }
        QTabBar::tab:hover {
            background-color: #2A2A2C;
        }

        QLabel#MainTitle {
            color: #F2F2F7;
        }
        QLabel#Subtitle {
            color: #C7C7CC;
        }
        QLabel#SectionTitle {
            color: #F2F2F7;
        }
        QLabel#OptionLabel {
            color: #D1D1D6;
            font-size: 9pt;
        }
        QLabel#InfoLabel {
            color: #C7C7CC;
        }

        QFrame#SearchCard, QFrame#MainCard {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #1F1F23,
                stop:1 #101014
            );
            border-radius: 10px;
            border: 1px solid rgba(70, 70, 90, 0.8);
        }

        QLineEdit, QSpinBox, QComboBox {
            background-color: #2C2C2E;
            border: 1px solid #3A3A3C;
            border-radius: 6px;
            padding: 4px 8px;
            color: #F2F2F7;
            selection-background-color: #0A84FF;
            selection-color: #FFFFFF;
        }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #0A84FF;
        }

        QPushButton {
            background-color: #2C2C2E;
            border-radius: 6px;
            border: 1px solid #3A3A3C;
            padding: 6px 12px;
            color: #F2F2F7;
        }
        QPushButton:hover {
            background-color: #3A3A3C;
        }
        QPushButton:pressed {
            background-color: #0A84FF;
            border: 1px solid #0A84FF;
            color: #FFFFFF;
        }
        QPushButton#PrimaryButton {
            background-color: #0A84FF;
            border: 1px solid #0A84FF;
            color: #FFFFFF;
            font-weight: 600;
        }
        QPushButton#PrimaryButton:hover {
            background-color: #1C92FF;
        }
        QPushButton#PrimaryButton:pressed {
            background-color: #0060DF;
        }

        QPushButton#SecondaryButton {
            background-color: rgba(44, 44, 46, 0.7);
            border-radius: 8px;
            border: 1px solid rgba(70, 70, 90, 0.8);
            padding: 6px 10px;
            color: #E5E5EA;
            font-weight: 500;
        }
        QPushButton#SecondaryButton:hover {
            background-color: rgba(60, 60, 78, 0.9);
        }

        QCheckBox {
            spacing: 6px;
            color: #D1D1D6;
        }

        QListWidget#ResultsList {
            background-color: #18181B;
            border: 1px solid #3A3A3C;
            border-radius: 8px;
            padding: 4px;
        }
        QListWidget#ResultsList::item {
            padding: 4px 6px;
            color: #F2F2F7;
            height: 22px;
        }
        QListWidget#ResultsList::item:selected {
            background-color: #0A84FF;
            color: #FFFFFF;
        }
        QListWidget#ResultsList::item:hover {
            background-color: #27272F;
        }

        QLabel#PreviewLabel {
            background-color: #101015;
            border: 1px solid #333333;
            border-radius: 8px;
        }

        QStatusBar {
            background-color: #1C1C1E;
            color: #C7C7CC;
            border-top: 1px solid #2C2C2E;
        }
        QStatusBar::item {
            border: none;
        }

        QSplitter::handle {
            background-color: #1C1C1E;
        }
        QSplitter::handle:horizontal {
            width: 4px;
        }
        """)

    # ====== CALLBACKS INDEX ======

    def _reset_results_view(self) -> None:
        if self.results_list is not None:
            self.results_list.clear()
        self.current_results = []
        if self.preview_label is not None:
            self.preview_label.clear()
        if self.info_label is not None:
            self.info_label.setText("")
        self.preview_pixmap = None

    def _on_engine_changed(self, engine: Optional[ClipSearchEngine]) -> None:
        self.engine = engine
        self._reset_results_view()
        if self.engine is None:
            self.statusBar().showMessage("Aucun index chargé pour ce projet.", 4000)
        else:
            self.statusBar().showMessage(
                f"Index chargé ({self.engine.num_items} images indexées).", 4000
            )

    def _on_status_message(self, msg: str, timeout: int) -> None:
        self.statusBar().showMessage(msg, timeout)

    def _on_projects_changed(self, projects: List[str], current: str) -> None:
        if self.project_combo_search is None:
            return
        self.project_combo_search.blockSignals(True)
        self.project_combo_search.clear()
        for p in projects:
            self.project_combo_search.addItem(p)
        if current in projects:
            idx = projects.index(current)
            self.project_combo_search.setCurrentIndex(idx)
        self.project_combo_search.blockSignals(False)

    def _on_project_selected_from_search(self, index: int) -> None:
        if index < 0 or self.project_combo_search is None or self.index_tab is None:
            return
        name = self.project_combo_search.currentText().strip()
        if not name:
            return
        self.index_tab.select_project(name)

    # ====== ANIMATION PREVIEW ======

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

    # ====== RECHERCHE ======

    def on_search_clicked(self) -> None:
        query = self.query_edit.text().strip() if self.query_edit else ""
        self._reset_results_view()

        if not query:
            return
        if self.engine is None:
            self.statusBar().showMessage("Aucun index chargé pour ce projet.", 4000)
            return

        self.statusBar().showMessage("Recherche en cours…")
        QApplication.processEvents()

        k = max(1, int(self.topk_spin.value())) if self.topk_spin else 10

        try:
            results = self.engine.search(query, k)
        except Exception as e:
            self.statusBar().showMessage("Erreur de recherche.", 4000)
            print("Erreur de recherche :", e)
            return

        for result in results:
            meta = result.meta
            path = meta.get("video_path", "unknown")
            rel = meta.get("rel_video_path", Path(path).name)

            ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
            tc = seconds_to_timecode(float(ts))

            display_path = rel
            if len(display_path) > 70:
                display_path = "…" + display_path[-67:]

            line = f"[{tc}]  {result.score:5.3f}  {display_path}"
            item = QListWidgetItem(line)
            item.setToolTip(path)
            self.results_list.addItem(item)

            self.current_results.append(result)

        if not self.current_results:
            self.statusBar().showMessage(
                f"Aucun résultat pour « {query} ».", 3000
            )
            return

        self.statusBar().showMessage(
            f"Recherche terminée pour « {query} ». {len(self.current_results)} résultats.",
            2500,
        )
        self.results_list.setCurrentRow(0)

    def on_result_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.current_results):
            return

        result = self.current_results[row]
        meta = result.meta

        path = meta.get("video_path", "unknown")
        ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
        tc = seconds_to_timecode(float(ts))
        frame_index = meta.get("frame_index", None)

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

    def on_autopreview_toggled(self, state: int) -> None:
        enabled = self.autopreview_check.isChecked()
        self.animate_preview(enabled)

        if not enabled:
            self.preview_label.clear()
            self.preview_pixmap = None
        else:
            row = self.results_list.currentRow()
            if 0 <= row < len(self.current_results):
                result = self.current_results[row]
                meta = result.meta
                path = meta.get("video_path", "unknown")
                frame_index = meta.get("frame_index", None)
                self.load_preview_frame(path, frame_index)

    # ====== UTILITAIRES ======

    def get_current_result(self) -> Optional[SearchResult]:
        row = self.results_list.currentRow()
        if row < 0 or row >= len(self.current_results):
            return None
        return self.current_results[row]

    def on_copy_clicked(self) -> None:
        result = self.get_current_result()
        if result is None:
            return

        meta = result.meta
        path = meta.get("video_path", "unknown")
        ts = meta.get("timestamp_sec", meta.get("timestamp", 0.0))
        tc = seconds_to_timecode(float(ts))

        line = f"{path} @ {tc}"
        QApplication.clipboard().setText(line)
        self.statusBar().showMessage(
            "Chemin + timecode copiés dans le presse-papier.", 2000
        )

    def on_open_vlc_clicked(self) -> None:
        result = self.get_current_result()
        if result is None:
            return

        meta = result.meta
        path = meta.get("video_path", "unknown")
        ts = float(meta.get("timestamp_sec", meta.get("timestamp", 0.0)))

        if not Path(path).is_file():
            self.statusBar().showMessage("Fichier vidéo introuvable.", 3000)
            return

        vlc_path = Path(self.cfg.VLC_PATH)
        if not vlc_path.is_file():
            self.statusBar().showMessage(
                "VLC introuvable. Modifie VLC_PATH dans config.py.", 4000
            )
            return

        try:
            subprocess.Popen([
                str(vlc_path),
                f"--start-time={int(ts)}",
                path,
            ])
            self.statusBar().showMessage("Ouverture dans VLC…", 2000)
        except Exception as e:
            self.statusBar().showMessage(
                "Erreur lors du lancement de VLC.", 3000
            )
            print("Erreur VLC :", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SearchWindow()
    win.show()
    sys.exit(app.exec())
