#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QUrl, QThread, QObject, Signal, Slot
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QComboBox,
)

from config import Config, save_settings, scan_projects
from engine import ClipSearchEngine, index_videos


class IndexWorker(QObject):
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


class IndexTab(QWidget):
    engine_changed = Signal(object)          # ClipSearchEngine | None
    status_message = Signal(str, int)
    projects_changed = Signal(list, str)     # (projects, current)

    def __init__(self, cfg: Config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.cfg = cfg

        self.engine: ClipSearchEngine | None = None
        self.projects: List[str] = []

        self.index_thread: QThread | None = None
        self.index_worker: IndexWorker | None = None

        self.library_edit: QLineEdit | None = None
        self.project_combo: QComboBox | None = None
        self.new_project_edit: QLineEdit | None = None

        self.index_info_label: QLabel | None = None
        self.index_log: QTextEdit | None = None
        self.index_button: QPushButton | None = None
        self.cancel_button: QPushButton | None = None
        self.open_index_button: QPushButton | None = None
        self.index_progress: QProgressBar | None = None

        self._build_ui()
        self.refresh_projects()

    # ========= UI =========

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Carte projet + bibliothèque
        project_card = QFrame()
        project_card.setObjectName("SearchCard")
        project_layout = QVBoxLayout(project_card)
        project_layout.setContentsMargins(16, 12, 16, 12)
        project_layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)

        # Projet
        proj_label = QLabel("Projet")
        proj_label.setObjectName("OptionLabel")
        row.addWidget(proj_label)

        self.project_combo = QComboBox()
        self.project_combo.setMaximumWidth(260)
        self.project_combo.setMinimumContentsLength(12)
        self.project_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        row.addWidget(self.project_combo, stretch=0)

        self.new_project_edit = QLineEdit()
        self.new_project_edit.setPlaceholderText("Nouveau projet…")
        self.new_project_edit.setMaximumWidth(220)
        row.addWidget(self.new_project_edit, stretch=0)

        new_project_button = QPushButton("Créer")
        new_project_button.setObjectName("SecondaryButton")
        new_project_button.clicked.connect(self.on_create_project_clicked)
        row.addWidget(new_project_button)

        row.addStretch(1)

        # Bibliothèque
        lib_label = QLabel("Bibliothèque")
        lib_label.setObjectName("OptionLabel")
        row.addWidget(lib_label)

        self.library_edit = QLineEdit()
        self.library_edit.setText(str(self.cfg.LIBRARY_DIR))
        self.library_edit.setMaximumWidth(260)
        row.addWidget(self.library_edit, stretch=0)

        browse_button = QPushButton("Modifier…")
        browse_button.setObjectName("SecondaryButton")
        browse_button.clicked.connect(self.on_browse_library_clicked)
        row.addWidget(browse_button)

        project_layout.addLayout(row)
        layout.addWidget(project_card)

        # Carte infos index
        index_info_card = QFrame()
        index_info_card.setObjectName("SearchCard")
        index_info_layout = QVBoxLayout(index_info_card)
        index_info_layout.setContentsMargins(16, 12, 16, 12)
        index_info_layout.setSpacing(6)

        index_title = QLabel("Index actuel")
        index_title.setObjectName("SectionTitle")
        index_info_layout.addWidget(index_title)

        self.index_info_label = QLabel("")
        self.index_info_label.setObjectName("InfoLabel")
        self.index_info_label.setWordWrap(True)
        index_info_layout.addWidget(self.index_info_label)

        layout.addWidget(index_info_card)

        # Carte actions index
        index_actions_card = QFrame()
        index_actions_card.setObjectName("SearchCard")
        index_actions_layout = QHBoxLayout(index_actions_card)
        index_actions_layout.setContentsMargins(16, 12, 16, 12)
        index_actions_layout.setSpacing(8)

        self.index_button = QPushButton("Lancer l'indexation")
        self.index_button.setObjectName("PrimaryButton")
        self.index_button.clicked.connect(self.on_start_indexing_clicked)
        index_actions_layout.addWidget(self.index_button)

        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setObjectName("SecondaryButton")
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.on_cancel_indexing_clicked)
        index_actions_layout.addWidget(self.cancel_button)

        self.open_index_button = QPushButton("Ouvrir le dossier du projet")
        self.open_index_button.setObjectName("SecondaryButton")
        self.open_index_button.clicked.connect(self.on_open_index_folder_clicked)
        index_actions_layout.addWidget(self.open_index_button)

        index_actions_layout.addStretch()
        layout.addWidget(index_actions_card)

        # Progression (barre seule)
        self.index_progress = QProgressBar()
        self.index_progress.setRange(0, 100)
        self.index_progress.setValue(0)
        layout.addWidget(self.index_progress)

        # Log indexation
        self.index_log = QTextEdit()
        self.index_log.setReadOnly(True)
        self.index_log.setObjectName("LogView")
        layout.addWidget(self.index_log, stretch=1)

        # Connexions
        self.project_combo.currentIndexChanged.connect(self.on_project_changed)

    # ========= API publique =========

    def set_engine(self, engine: Optional[ClipSearchEngine]) -> None:
        self.engine = engine
        self.update_index_info_label()

    def select_project(self, name: str) -> None:
        if self.project_combo is None or not self.projects:
            return
        if name not in self.projects:
            return
        idx = self.projects.index(name)
        self.project_combo.setCurrentIndex(idx)

    # ========= GESTION PROJETS / BIBLIOTHÈQUE =========

    def refresh_projects(self) -> None:
        if self.library_edit is not None:
            self.library_edit.setText(str(self.cfg.LIBRARY_DIR))

        disk_projects = scan_projects(self.cfg.LIBRARY_DIR, self.cfg.KNOWN_PROJECTS)
        merged = set(disk_projects) | set(self.cfg.KNOWN_PROJECTS)
        if self.cfg.CURRENT_PROJECT:
            merged.add(self.cfg.CURRENT_PROJECT)

        self.projects = sorted(merged)
        self.cfg.KNOWN_PROJECTS = list(self.projects)
        save_settings(self.cfg)

        if self.project_combo is None:
            return

        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        for p in self.projects:
            self.project_combo.addItem(p)

        if self.cfg.CURRENT_PROJECT in self.projects:
            idx = self.projects.index(self.cfg.CURRENT_PROJECT)
            self.project_combo.setCurrentIndex(idx)
        elif self.projects:
            self.cfg.CURRENT_PROJECT = self.projects[0]
            self.project_combo.setCurrentIndex(0)

        self.project_combo.blockSignals(False)

        save_settings(self.cfg)
        self.update_index_info_label()

        self.projects_changed.emit(self.projects, self.cfg.CURRENT_PROJECT)

    @Slot()
    def on_browse_library_clicked(self) -> None:
        base_dir = str(self.cfg.LIBRARY_DIR) if self.cfg.LIBRARY_DIR else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choisir la bibliothèque d'index", base_dir)
        if not chosen:
            return
        self.cfg.LIBRARY_DIR = Path(chosen)
        save_settings(self.cfg)
        self.refresh_projects()
        self.status_message.emit(f"Bibliothèque : {self.cfg.LIBRARY_DIR}", 3000)

    @Slot(int)
    def on_project_changed(self, index: int) -> None:
        if self.project_combo is None or index < 0:
            return

        project = self.project_combo.currentText().strip()
        if not project:
            return

        self.cfg.CURRENT_PROJECT = project
        if project not in self.cfg.KNOWN_PROJECTS:
            self.cfg.KNOWN_PROJECTS.append(project)
        save_settings(self.cfg)

        if self.index_progress is not None:
            self.index_progress.setValue(0)
        if self.index_log is not None:
            self.index_log.clear()

        try:
            self.engine = ClipSearchEngine(self.cfg)
            self.status_message.emit(
                f"Projet changé : {project} ({self.engine.num_items} images indexées).",
                4000,
            )
        except Exception as e:
            print("Erreur lors du chargement de l'index pour le projet :", e)
            self.engine = None
            self.status_message.emit(
                "Index introuvable pour ce projet. Lance une indexation.",
                5000,
            )

        self.update_index_info_label()
        self.projects_changed.emit(self.projects, self.cfg.CURRENT_PROJECT)
        self.engine_changed.emit(self.engine)

    @Slot()
    def on_create_project_clicked(self) -> None:
        if self.new_project_edit is None:
            return
        name = self.new_project_edit.text().strip()
        if not name:
            return

        rel = Path(name.replace("\\", "/"))
        project_dir = self.cfg.LIBRARY_DIR / rel
        project_dir.mkdir(parents=True, exist_ok=True)

        rel_str = rel.as_posix()

        self.cfg.CURRENT_PROJECT = rel_str
        if rel_str not in self.cfg.KNOWN_PROJECTS:
            self.cfg.KNOWN_PROJECTS.append(rel_str)
        save_settings(self.cfg)

        self.new_project_edit.clear()
        self.refresh_projects()

        self.status_message.emit(f"Projet créé : {self.cfg.CURRENT_PROJECT}", 4000)

    # ========= INDEX : INFOS & LOG =========

    def update_index_info_label(self) -> None:
        if self.index_info_label is None:
            return

        index_path = self.cfg.INDEX_PATH
        meta_path = self.cfg.META_PATH

        if self.engine is None:
            txt = (
                f"Projet : {self.cfg.CURRENT_PROJECT}\n"
                f"Bibliothèque : {self.cfg.LIBRARY_DIR}\n\n"
                f"Fichier index : {index_path}\n"
                f"Métadonnées : {meta_path}\n"
                f"Index non chargé."
            )
        else:
            txt = (
                f"Projet : {self.cfg.CURRENT_PROJECT}\n"
                f"Bibliothèque : {self.cfg.LIBRARY_DIR}\n\n"
                f"Fichier index : {index_path}\n"
                f"Métadonnées : {meta_path}\n"
                f"Images indexées : {self.engine.num_items}"
            )
        self.index_info_label.setText(txt)

    def append_index_log(self, text: str) -> None:
        if self.index_log is None:
            return
        self.index_log.append(text)
        cursor = self.index_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.index_log.setTextCursor(cursor)

    # ========= ACTIONS : INDEX =========

    @Slot()
    def on_open_index_folder_clicked(self) -> None:
        idx_dir = self.cfg.OUTPUT_DIR
        idx_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(idx_dir)))

    @Slot()
    def on_start_indexing_clicked(self) -> None:
        if self.index_thread is not None:
            self.status_message.emit("Indexation déjà en cours.", 3000)
            return

        if self.index_log is not None:
            self.index_log.clear()

        self.append_index_log(
            f"[INFO] Lancement de l'indexation (projet : {self.cfg.CURRENT_PROJECT})."
        )
        self.status_message.emit("Indexation en cours…", 2000)

        if self.index_button is not None:
            self.index_button.setEnabled(False)
        if self.cancel_button is not None:
            self.cancel_button.setVisible(True)
            self.cancel_button.setEnabled(True)

        if self.index_progress is not None:
            self.index_progress.setValue(0)

        self.index_thread = QThread(self)
        self.index_worker = IndexWorker(self.cfg)
        self.index_worker.moveToThread(self.index_thread)

        self.index_thread.started.connect(self.index_worker.run)
        self.index_worker.progress_changed.connect(self.on_index_progress_changed)
        self.index_worker.status_changed.connect(self.on_index_status_changed)
        self.index_worker.error.connect(self.on_index_error)
        self.index_worker.finished.connect(self.on_index_finished)

        self.index_worker.finished.connect(self.index_thread.quit)
        self.index_thread.finished.connect(self.index_thread.deleteLater)

        self.index_thread.start()

    @Slot()
    def on_cancel_indexing_clicked(self) -> None:
        if self.index_worker is None:
            return
        self.index_worker.cancel()
        self.append_index_log("[INFO] Annulation demandée…")
        self.status_message.emit("Annulation de l'indexation en cours…", 3000)
        if self.cancel_button is not None:
            self.cancel_button.setEnabled(False)

    @Slot(int)
    def on_index_progress_changed(self, value: int) -> None:
        if self.index_progress is not None:
            self.index_progress.setValue(value)

    @Slot(str)
    def on_index_status_changed(self, message: str) -> None:
        self.append_index_log(message)

    @Slot(str)
    def on_index_error(self, message: str) -> None:
        self.append_index_log(f"[ERREUR] {message}")
        self.status_message.emit("Erreur lors de l'indexation.", 4000)
        if self.index_button is not None:
            self.index_button.setEnabled(True)
        if self.cancel_button is not None:
            self.cancel_button.setVisible(False)
            self.cancel_button.setEnabled(False)
        self.index_thread = None
        self.index_worker = None
        self.engine_changed.emit(self.engine)

    @Slot()
    def on_index_finished(self) -> None:
        self.append_index_log("[INFO] Indexation terminée.")
        if self.index_button is not None:
            self.index_button.setEnabled(True)
        if self.cancel_button is not None:
            self.cancel_button.setVisible(False)
            self.cancel_button.setEnabled(False)

        try:
            self.engine = ClipSearchEngine(self.cfg)
        except Exception as e:
            self.append_index_log(f"[ERREUR] Impossible de charger l'index : {e}")
            self.engine = None

        self.refresh_projects()
        self.update_index_info_label()
        self.status_message.emit("Indexation terminée.", 4000)
        self.engine_changed.emit(self.engine)

        self.index_thread = None
        self.index_worker = None
