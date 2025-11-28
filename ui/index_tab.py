#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
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
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QMessageBox,
)

from config import Config, save_settings, scan_projects
from engine import ClipSearchEngine, index_videos, is_index_up_to_date
from .workers import IndexWorker


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

        # Gestion des fichiers
        self.files_list: QListWidget | None = None
        self.add_files_button: QPushButton | None = None
        self.rename_file_button: QPushButton | None = None
        self.delete_file_button: QPushButton | None = None

        self._build_ui()
        # IMPORTANT : pas de refresh_projects() ici.
        # Il est appelé par MainWindow une fois les signaux câblés.

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

        # Carte fichiers du projet
        files_card = QFrame()
        files_card.setObjectName("SearchCard")
        files_layout = QVBoxLayout(files_card)
        files_layout.setContentsMargins(16, 12, 16, 12)
        files_layout.setSpacing(6)

        files_title = QLabel("Fichiers du projet")
        files_title.setObjectName("SectionTitle")
        files_layout.addWidget(files_title)

        files_subtitle = QLabel(
            "Les vidéos à indexer doivent être présentes dans ce dossier de projet."
        )
        files_subtitle.setObjectName("InfoLabel")
        files_subtitle.setWordWrap(True)
        files_layout.addWidget(files_subtitle)

        self.files_list = QListWidget()
        self.files_list.setObjectName("FilesList")
        self.files_list.setSelectionMode(QListWidget.SingleSelection)
        files_layout.addWidget(self.files_list, stretch=1)

        files_buttons_row = QHBoxLayout()
        files_buttons_row.setSpacing(8)

        self.add_files_button = QPushButton("Ajouter des vidéos…")
        self.add_files_button.setObjectName("SecondaryButton")
        self.add_files_button.clicked.connect(self.on_add_files_clicked)
        files_buttons_row.addWidget(self.add_files_button)

        self.rename_file_button = QPushButton("Renommer")
        self.rename_file_button.setObjectName("SecondaryButton")
        self.rename_file_button.clicked.connect(self.on_rename_file_clicked)
        files_buttons_row.addWidget(self.rename_file_button)

        self.delete_file_button = QPushButton("Supprimer")
        self.delete_file_button.setObjectName("SecondaryButton")
        self.delete_file_button.clicked.connect(self.on_delete_file_clicked)
        files_buttons_row.addWidget(self.delete_file_button)

        files_buttons_row.addStretch()
        files_layout.addLayout(files_buttons_row)

        layout.addWidget(files_card, stretch=1)

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

        # Progression
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
        self.refresh_files_list()

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
        self.refresh_files_list()
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

    # ========= GESTION DES FICHIERS =========

    def refresh_files_list(self) -> None:
        if self.files_list is None:
            return

        self.files_list.clear()
        project_dir = self.cfg.OUTPUT_DIR
        project_dir.mkdir(parents=True, exist_ok=True)

        exts = tuple(ext.lower() for ext in self.cfg.VIDEO_EXT)

        for path in sorted(project_dir.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            rel = path.name
            item = QListWidgetItem(rel)
            item.setData(Qt.UserRole, str(path))
            self.files_list.addItem(item)

    @Slot()
    def on_add_files_clicked(self) -> None:
        project_dir = self.cfg.OUTPUT_DIR
        project_dir.mkdir(parents=True, exist_ok=True)

        exts_filter = "Vidéos (*.mp4 *.mkv *.mov *.avi);;Tous les fichiers (*.*)"
        base_dir = str(project_dir)

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Ajouter des vidéos au projet",
            base_dir,
            exts_filter,
        )
        if not files:
            return

        added = 0
        skipped = 0

        for f in files:
            src = Path(f)
            if not src.is_file():
                continue

            dest = project_dir / src.name

            # Doublon exact (nom + taille) -> ignoré
            if dest.is_file():
                try:
                    if dest.stat().st_size == src.stat().st_size:
                        self.append_index_log(
                            f"[INFO] Ignoré (déjà présent) : {src.name}"
                        )
                        skipped += 1
                        continue
                except OSError:
                    pass

                # Même nom mais taille différente -> on choisit un nouveau nom
                stem = src.stem
                suffix = src.suffix
                i = 2
                while True:
                    candidate = project_dir / f"{stem} ({i}){suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    i += 1

            try:
                shutil.copy2(src, dest)
                self.append_index_log(f"[INFO] Ajouté : {dest.name}")
                added += 1
            except Exception as e:
                self.append_index_log(f"[ERREUR] Copie impossible pour {src} : {e}")

        self.refresh_files_list()

        msg_parts = []
        if added:
            msg_parts.append(f"{added} fichier(s) ajouté(s)")
        if skipped:
            msg_parts.append(f"{skipped} déjà présent(s)")
        if not msg_parts:
            msg_parts.append("Aucun fichier ajouté")

        self.status_message.emit(" / ".join(msg_parts), 4000)

    @Slot()
    def on_rename_file_clicked(self) -> None:
        if self.files_list is None:
            return
        item = self.files_list.currentItem()
        if item is None:
            return

        old_path = Path(item.data(Qt.UserRole))
        if not old_path.exists():
            self.status_message.emit("Fichier introuvable sur le disque.", 4000)
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Renommer le fichier",
            "Nouveau nom de fichier :",
            text=old_path.name,
        )
        if not ok:
            return

        new_name = new_name.strip()
        if not new_name:
            return
        if any(sep in new_name for sep in ("/", "\\")):
            QMessageBox.warning(
                self,
                "Nom invalide",
                "Le nom ne doit pas contenir de séparateur de chemin.",
            )
            return

        new_path = old_path.parent / new_name
        if new_path.exists() and new_path != old_path:
            QMessageBox.warning(
                self,
                "Nom déjà utilisé",
                "Un fichier avec ce nom existe déjà dans le projet.",
            )
            return

        try:
            old_path.rename(new_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible de renommer le fichier : {e}",
            )
            return

        self.append_index_log(f"[INFO] Renommé : {old_path.name} -> {new_path.name}")
        self.refresh_files_list()
        self.status_message.emit("Fichier renommé.", 3000)

    @Slot()
    def on_delete_file_clicked(self) -> None:
        if self.files_list is None:
            return
        item = self.files_list.currentItem()
        if item is None:
            return

        path = Path(item.data(Qt.UserRole))
        if not path.exists():
            self.status_message.emit("Fichier introuvable sur le disque.", 4000)
            self.refresh_files_list()
            return

        answer = QMessageBox.question(
            self,
            "Supprimer le fichier",
            f"Supprimer définitivement ce fichier du projet ?\n\n{path.name}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            path.unlink()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible de supprimer le fichier : {e}",
            )
            return

        self.append_index_log(f"[INFO] Supprimé : {path.name}")
        self.refresh_files_list()
        self.status_message.emit("Fichier supprimé.", 3000)

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
        """
        Lance l’indexation dans un QThread, sauf si :
        - l’index est déjà à jour
        - ou s’il n’y a aucune vidéo dans le projet
        """
        if self.index_thread is not None:
            self.status_message.emit("Indexation déjà en cours.", 3000)
            return

        # Check préalable : index déjà à jour ?
        up_to_date, reason = is_index_up_to_date(self.cfg)

        if up_to_date:
            # Rien à faire, on log juste et on recharge l’engine au cas où
            self.append_index_log(f"[INFO] {reason}")
            self.status_message.emit(reason, 4000)
            try:
                self.engine = ClipSearchEngine(self.cfg)
            except Exception as e:
                print("Erreur lors du rechargement de l'index :", e)
                self.engine = None
            self.update_index_info_label()
            self.engine_changed.emit(self.engine)
            return

        # Pas à jour : cas particulier "aucune vidéo"
        if "Aucune vidéo trouvée" in reason:
            self.append_index_log(f"[INFO] {reason}")
            self.status_message.emit(reason, 4000)
            return

        # Sinon, on lance une vraie indexation
        if self.index_log is not None:
            self.index_log.clear()

        self.append_index_log(
            f"[INFO] Lancement de l'indexation (projet : {self.cfg.CURRENT_PROJECT})."
        )
        self.append_index_log(f"[INFO] {reason}")
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
