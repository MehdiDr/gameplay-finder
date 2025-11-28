#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from PySide6.QtWidgets import QWidget


def apply_dark_style(widget: QWidget) -> None:
    """
    Style global de l'application (look "app mac" sombre, propre).
    """
    widget.setStyleSheet("""
    /* FOND GÉNÉRAL *****************************************************/

    QMainWindow {
        background-color: #0B0B0D;
    }

    QWidget {
        background-color: #0B0B0D;
        color: #F2F2F7;
        font-family: "Segoe UI", "SF Pro Text", sans-serif;
        font-size: 10pt;
    }

    /* TABS *************************************************************/

    QTabWidget::pane {
        border: none;
        padding-top: 2px;
    }

    QTabBar::tab {
        background-color: #1C1C1E;
        color: #C7C7CC;
        padding: 6px 14px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
        margin-top: 2px;
    }

    QTabBar::tab:selected {
        background-color: #2C2C2E;
        color: #FFFFFF;
    }

    QTabBar::tab:hover {
        background-color: #2A2A2C;
    }

    /* TITRES / LABELS **************************************************/

    QLabel#MainTitle {
        color: #F2F2F7;
        font-size: 18pt;
        font-weight: 650;
    }

    QLabel#Subtitle {
        color: #C7C7CC;
        font-size: 9.5pt;
    }

    QLabel#SectionTitle {
        color: #F2F2F7;
        font-size: 11pt;
        font-weight: 600;
        padding-bottom: 2px;
    }

    QLabel#OptionLabel {
        color: #D1D1D6;
        font-size: 9pt;
    }

    QLabel#InfoLabel {
        color: #C7C7CC;
        font-size: 9pt;
    }

    /* CARTES ***********************************************************/

    QFrame#SearchCard, 
    QFrame#MainCard {
        background: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 #1A1A1E,
            stop:1 #101014
        );
        border-radius: 12px;
        border: 1px solid rgba(70, 70, 90, 0.9);
    }

    /* CHAMPS ***********************************************************/

    QLineEdit, 
    QSpinBox, 
    QComboBox {
        background-color: #2C2C2E;
        border: 1px solid #3A3A3C;
        border-radius: 8px;
        padding: 5px 10px;
        color: #F2F2F7;
        selection-background-color: #0A84FF;
        selection-color: #FFFFFF;
    }

    QLineEdit:focus, 
    QSpinBox:focus, 
    QComboBox:focus {
        border: 1px solid #0A84FF;
    }

    QComboBox QAbstractItemView {
        background-color: #1C1C1E;
        border: 1px solid #3A3A3C;
        selection-background-color: #0A84FF;
        selection-color: #FFFFFF;
    }

    /* BOUTONS **********************************************************/

    QPushButton {
        background-color: #2C2C2E;
        border-radius: 8px;
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
        border: 1px solid rgba(70, 70, 90, 0.9);
        padding: 6px 10px;
        color: #E5E5EA;
        font-weight: 500;
    }

    QPushButton#SecondaryButton:hover {
        background-color: rgba(60, 60, 78, 0.95);
    }

    QPushButton#SecondaryButton:pressed {
        background-color: #0A84FF;
        border-color: #0A84FF;
        color: #FFFFFF;
    }

    /* CHECKBOX *********************************************************/

    QCheckBox {
        spacing: 6px;
        color: #D1D1D6;
    }

    QCheckBox::indicator {
        width: 14px;
        height: 14px;
    }

    QCheckBox::indicator:unchecked {
        border-radius: 4px;
        border: 1px solid #3A3A3C;
        background-color: #1C1C1E;
    }

    QCheckBox::indicator:checked {
        border-radius: 4px;
        border: 1px solid #0A84FF;
        background-color: #0A84FF;
    }

    /* LISTES DE RÉSULTATS / FICHIERS **********************************/

    QListWidget#ResultsList, 
    QListWidget#FilesList {
        background-color: #18181B;
        border: 1px solid #3A3A3C;
        border-radius: 10px;
        padding: 4px;
    }

    QListWidget#ResultsList::item,
    QListWidget#FilesList::item {
        padding: 4px 6px;
        color: #F2F2F7;
        height: 22px;
    }

    QListWidget#ResultsList::item:selected,
    QListWidget#FilesList::item:selected {
        background-color: #0A84FF;
        color: #FFFFFF;
    }

    QListWidget#ResultsList::item:hover,
    QListWidget#FilesList::item:hover {
        background-color: #27272F;
    }

    /* PREVIEW **********************************************************/

    QLabel#PreviewLabel {
        background-color: #050508;
        border: 1px solid #333333;
        border-radius: 12px;
    }

    /* JOURNAL **********************************************************/

    QTextEdit#LogView {
        background-color: #000000;
        border-radius: 10px;
        border: 1px solid #2C2C2E;
        font-family: "Consolas", "JetBrains Mono", monospace;
        font-size: 9pt;
    }

    /* BARRE DE STATUT **************************************************/

    QStatusBar {
        background-color: #111111;
        color: #C7C7CC;
        border-top: 1px solid #2C2C2E;
    }

    QStatusBar::item {
        border: none;
    }

    /* SPLITTER *********************************************************/

    QSplitter::handle {
        background-color: #1C1C1E;
    }

    QSplitter::handle:horizontal {
        width: 4px;
    }
    """)
