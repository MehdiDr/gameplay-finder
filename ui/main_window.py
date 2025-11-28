#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, List

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QStatusBar,
)
from PySide6.QtGui import QFont

from config import Config, load_settings
from engine import ClipSearchEngine
from .style import apply_dark_style
from .index_tab import IndexTab
from .search_tab import SearchTab


class MainWindow(QMainWindow):
    """
    Fenêtre principale de Gameplay Finder :
    - en-tête (titre + sous-titre)
    - onglets Recherche / Index
    - barre de statut partagée
    """

    def __init__(self) -> None:
        super().__init__()

        self.cfg = Config()
        load_settings(self.cfg)
        self.engine: Optional[ClipSearchEngine] = None

        self.tabs: QTabWidget | None = None
        self.index_tab: IndexTab | None = None
        self.search_tab: SearchTab | None = None

        self._build_ui()
        self._wire_signals()

        # Maintenant que les signaux sont câblés, on peut rafraîchir les projets
        # (cela va aussi propager la liste à l'onglet Recherche).
        if self.index_tab is not None:
            self.index_tab.refresh_projects()

        # Puis on charge l'engine initial pour le projet courant (si un index est dispo)
        self._load_initial_engine()

    # ========= construction UI =========

    def _build_ui(self) -> None:
        self.setWindowTitle("Gameplay Finder (CLIP)")
        self.resize(1280, 760)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)

        # Header
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

        main_layout.addLayout(header_layout)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        self.search_tab = SearchTab(self.cfg)
        self.index_tab = IndexTab(self.cfg)

        self.tabs.addTab(self.search_tab, "Recherche")
        self.tabs.addTab(self.index_tab, "Index")

        # Status bar
        status = QStatusBar()
        self.setStatusBar(status)

        # Style global
        apply_dark_style(self)

    def _wire_signals(self) -> None:
        if self.index_tab is None or self.search_tab is None:
            return

        # Index -> MainWindow
        self.index_tab.engine_changed.connect(self._on_engine_changed)
        self.index_tab.status_message.connect(self._on_status_message)
        self.index_tab.projects_changed.connect(self._on_projects_changed)

        # Search -> MainWindow
        self.search_tab.status_message.connect(self._on_status_message)
        self.search_tab.project_selected.connect(self._on_project_selected_from_search)

    def _load_initial_engine(self) -> None:
        if self.index_tab is None or self.search_tab is None:
            return

        self.statusBar().showMessage("Chargement du modèle et de l'index…")
        try:
            self.engine = ClipSearchEngine(self.cfg)
            self.index_tab.set_engine(self.engine)
            self.search_tab.set_engine(self.engine)
            if self.engine is not None:
                self.statusBar().showMessage(
                    f"Index chargé ({self.engine.num_items} images indexées).",
                    4000,
                )
            else:
                self.statusBar().showMessage(
                    "Aucun index chargé (lance une indexation).",
                    5000,
                )
        except Exception as e:
            print("Impossible de charger CLIP / index au démarrage :", e)
            self.engine = None
            self.index_tab.set_engine(None)
            self.search_tab.set_engine(None)
            self.statusBar().showMessage("Aucun index chargé (lance une indexation).", 5000)

    # ========= slots internes =========

    def _on_engine_changed(self, engine: Optional[ClipSearchEngine]) -> None:
        self.engine = engine
        if self.search_tab is not None:
            self.search_tab.set_engine(engine)

    def _on_status_message(self, msg: str, timeout: int) -> None:
        self.statusBar().showMessage(msg, timeout)

    def _on_projects_changed(self, projects: List[str], current: str) -> None:
        if self.search_tab is not None:
            self.search_tab.update_projects(projects, current)

    def _on_project_selected_from_search(self, name: str) -> None:
        if self.index_tab is not None:
            self.index_tab.select_project(name)
