#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple, Optional

from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QLabel


def create_card(
    parent: Optional[QWidget] = None,
    object_name: str = "SearchCard",
    margins: Tuple[int, int, int, int] = (16, 12, 16, 12),
    spacing: int = 8,
) -> tuple[QFrame, QVBoxLayout]:
    """
    Crée un QFrame stylé comme une “carte” + son QVBoxLayout configuré.

    - object_name permet d'utiliser des styles différents (SearchCard, MainCard…)
    """
    card = QFrame(parent)
    card.setObjectName(object_name)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return card, layout


def section_title(text: str, parent: Optional[QWidget] = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("SectionTitle")
    return label


def option_label(text: str, parent: Optional[QWidget] = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("OptionLabel")
    return label


def info_label(text: str, parent: Optional[Widget] = None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("InfoLabel")
    return label
