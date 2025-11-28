# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)


class ResultsPanel(QWidget):
    """
    Panneau qui affiche la liste des résultats de recherche + boutons d'action.
    Il ne connaît pas la structure des SearchResult, seulement des lignes de texte.
    """

    row_selected = Signal(int)
    copy_requested = Signal(int)
    open_vlc_requested = Signal(int)
    open_resolve_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.currentRowChanged.connect(self._on_current_row_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)

        self.btn_copy = QPushButton("Copier chemin + timecode")
        self.btn_copy.setObjectName("SecondaryButton")
        self.btn_copy.clicked.connect(self._on_copy_clicked)

        self.btn_vlc = QPushButton("Ouvrir dans VLC")
        self.btn_vlc.setObjectName("SecondaryButton")
        self.btn_vlc.clicked.connect(self._on_open_vlc_clicked)

        self.btn_resolve = QPushButton("Ouvrir dans Resolve")
        self.btn_resolve.setObjectName("SecondaryButton")
        self.btn_resolve.clicked.connect(self._on_open_resolve_clicked)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        main_layout.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_vlc)
        btn_row.addWidget(self.btn_resolve)
        btn_row.addStretch(1)

        main_layout.addLayout(btn_row)

    # ===== API publique =====

    def clear(self) -> None:
        self._list.clear()

    def set_items(self, lines: List[str], tooltips: List[str] | None = None) -> None:
        self._list.clear()
        if tooltips is None:
            tooltips = [""] * len(lines)

        for text, tip in zip(lines, tooltips):
            item = QListWidgetItem(text)
            if tip:
                item.setToolTip(tip)
            self._list.addItem(item)

    def current_row(self) -> int:
        return self._list.currentRow()

    def set_current_row(self, row: int) -> None:
        if 0 <= row < self._list.count():
            self._list.setCurrentRow(row)

    def focus_list(self) -> None:
        self._list.setFocus(Qt.OtherFocusReason)

    # ===== Slots internes =====

    @Slot(int)
    def _on_current_row_changed(self, row: int) -> None:
        self.row_selected.emit(row)

    @Slot()
    def _on_item_double_clicked(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.copy_requested.emit(row)

    @Slot()
    def _on_copy_clicked(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.copy_requested.emit(row)

    @Slot()
    def _on_open_vlc_clicked(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.open_vlc_requested.emit(row)

    @Slot()
    def _on_open_resolve_clicked(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.open_resolve_requested.emit(row)
