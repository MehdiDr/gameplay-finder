# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
)

from .ui_helpers import info_label, option_label


class TimelineWidget(QWidget):
    """
    Mini-timeline : permet de naviguer autour de la frame trouvée par l'index.
    On travaille en 'pas' autour du centre (de -N à +N), interprétés comme des
    multiples de l'intervalle d'échantillonnage (cfg.INTERVAL_SEC).

    Signal:
        frame_changed(frame_index: int, timestamp_sec: float)
    """

    frame_changed = Signal(int, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._video_path: str = ""
        self._fps: float = 0.0
        self._center_frame: int = 0
        self._center_ts: float = 0.0
        self._interval_sec: float = 0.0

        self._half_window: int = 15  # pas de -15 à +15 par défaut

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 4, 0, 0)
        main_layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        title = option_label("Mini-timeline")
        header_row.addWidget(title)

        self.range_spin = QSpinBox()
        self.range_spin.setRange(1, 200)
        self.range_spin.setValue(self._half_window * 2)
        self.range_spin.setFixedWidth(70)
        self.range_spin.setToolTip(
            "Nombre de pas autour de la frame centrale.\n"
            "Chaque pas ~ 1 échantillon (cfg.INTERVAL_SEC)."
        )
        self.range_spin.valueChanged.connect(self._on_range_changed)
        header_row.addWidget(self.range_spin)

        lbl_steps = QLabel("pas")
        header_row.addWidget(lbl_steps)

        header_row.addStretch(1)
        main_layout.addLayout(header_row)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._half_window * 2)
        self.slider.setValue(self._half_window)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        main_layout.addWidget(self.slider)

        self.hint_label = info_label("Sélectionne un résultat pour activer la mini-timeline.")
        main_layout.addWidget(self.hint_label)

    # ===== API publique =====

    def set_context(
        self,
        video_path: str,
        fps: float,
        center_frame: int,
        center_ts: float,
        interval_sec: float,
    ) -> None:
        """
        Initialise la mini-timeline pour un nouveau résultat.
        """
        self._video_path = video_path or ""
        self._fps = float(fps) if fps else 0.0
        self._center_frame = max(0, int(center_frame))
        self._center_ts = max(0.0, float(center_ts))
        self._interval_sec = max(0.0, float(interval_sec))

        has_context = bool(self._video_path)

        self.slider.setEnabled(has_context)
        self.range_spin.setEnabled(has_context)
        self.hint_label.setText(
            "Navigue autour de la frame trouvée (gauche/droite) "
            "ou augmente la plage de recherche."
            if has_context
            else "Sélectionne un résultat pour activer la mini-timeline."
        )

        # On se recale au centre
        self._half_window = max(1, self.range_spin.value() // 2)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._half_window * 2)
        self.slider.setValue(self._half_window)
        self.slider.blockSignals(False)

    # ===== Slots internes =====

    @Slot(int)
    def _on_range_changed(self, value: int) -> None:
        # Nombre total de pas => moitié de chaque côté
        self._half_window = max(1, value // 2)
        cur_offset = self.slider.value() - self._half_window
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._half_window * 2)
        # On garde tant que possible la position relative
        new_val = self._half_window + cur_offset
        new_val = max(0, min(new_val, self._half_window * 2))
        self.slider.setValue(new_val)
        self.slider.blockSignals(False)
        # Et on réémet la frame correspondante
        self._emit_current_frame()

    @Slot(int)
    def _on_slider_changed(self, value: int) -> None:
        self._emit_current_frame()

    def _emit_current_frame(self) -> None:
        if not self._video_path:
            return

        offset = self.slider.value() - self._half_window  # entre -half_window et +half_window

        # On reste quantifié sur les échantillons de l'index
        if self._interval_sec > 0.0:
            ts = self._center_ts + offset * self._interval_sec
        else:
            # fallback très grossier : 1 / fps
            step = 1.0 / self._fps if self._fps > 0 else 1.0
            ts = self._center_ts + offset * step

        ts = max(0.0, ts)

        if self._fps > 0.0:
            frame_index = int(round(ts * self._fps))
        else:
            frame_index = max(0, self._center_frame + offset)

        self.frame_changed.emit(frame_index, ts)
