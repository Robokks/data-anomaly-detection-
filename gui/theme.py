"""Centralized light/dark theming: color tokens, QSS, QPalette, and a
``ThemeManager`` that applies/toggles/persists the active mode.

Everything the rest of the GUI needs to look "modern" lives here rather
than scattered across each panel: a small palette of semantic color
tokens, one QSS template parameterized by those tokens, a matching
``QPalette`` (QSS alone doesn't reliably restyle native dialogs like
``QFileDialog``/``QMessageBox``), and pyqtgraph curve-color lists (pyqtgraph
has its own styling system, entirely separate from Qt widget QSS -- panels
using it read ``PLOT_COLORS_LIGHT``/``PLOT_COLORS_DARK`` directly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ThemeMode = Literal["light", "dark"]

_SETTINGS_KEY = "theme/mode"


@dataclass(frozen=True)
class Palette:
    bg: str
    surface: str
    border: str
    text_primary: str
    text_secondary: str
    accent: str
    accent_hover: str
    success: str
    warning: str
    danger: str
    disabled: str


LIGHT = Palette(
    bg="#f5f6f8",
    surface="#ffffff",
    border="#dde1e6",
    text_primary="#1a1d23",
    text_secondary="#5b6270",
    accent="#3b6fed",
    accent_hover="#2f5bd1",
    success="#1e9e6b",
    warning="#b8860b",
    danger="#d64545",
    disabled="#a6acb8",
)

DARK = Palette(
    bg="#1b1e24",
    surface="#242832",
    border="#343a46",
    text_primary="#e8eaed",
    text_secondary="#a1a8b5",
    accent="#5b8dfb",
    accent_hover="#7ba1fc",
    success="#3ec98a",
    warning="#e0b13e",
    danger="#f0685f",
    disabled="#565d6b",
)

PLOT_COLORS_LIGHT = ["#3b6fed", "#e07b39", "#1e9e6b", "#d64545", "#8e5fd6", "#9c6b4c", "#d65fa0", "#6b7280"]
PLOT_COLORS_DARK = ["#5b8dfb", "#f0a05f", "#3ec98a", "#f0685f", "#ad8bee", "#c99a7a", "#f08bc4", "#9aa1ad"]


def plot_colors(mode: ThemeMode) -> list[str]:
    return PLOT_COLORS_DARK if mode == "dark" else PLOT_COLORS_LIGHT


_QSS_TEMPLATE = """
QMainWindow, QDialog {{
    background: {bg};
}}
QWidget {{
    color: {text_primary};
    font-size: 13px;
}}
QWidget#card {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel {{
    background: transparent;
}}
QToolBar {{
    background: {surface};
    border: none;
    border-bottom: 1px solid {border};
    padding: 6px;
    spacing: 4px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {text_primary};
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
}}
QToolBar QToolButton:hover {{
    background: {bg};
}}
QToolBar QToolButton:checked {{
    background: {accent};
    color: #ffffff;
}}
QStatusBar {{
    background: {surface};
    border-top: 1px solid {border};
    color: {text_secondary};
}}
QPushButton {{
    background: {surface};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {bg};
}}
QPushButton:disabled {{
    color: {disabled};
    border-color: {border};
}}
QDialogButtonBox QPushButton[text="Train"], QPushButton:default {{
    background: {accent};
    color: #ffffff;
    border: none;
}}
QDialogButtonBox QPushButton[text="Train"]:hover, QPushButton:default:hover {{
    background: {accent_hover};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {surface};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {accent};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {accent};
}}
QComboBox::drop-down {{
    border: none;
}}
QGroupBox {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {text_secondary};
}}
QTableWidget, QTreeWidget {{
    background: {surface};
    alternate-background-color: {bg};
    color: {text_primary};
    gridline-color: {border};
    border: 1px solid {border};
    border-radius: 6px;
}}
QHeaderView::section {{
    background: {surface};
    color: {text_secondary};
    border: none;
    border-bottom: 1px solid {border};
    padding: 6px;
    font-weight: 600;
}}
QTableWidget::item:selected, QTreeWidget::item:selected {{
    background: {accent};
    color: #ffffff;
}}
QSplitter::handle {{
    background: {bg};
    border-left: 1px solid {border};
    border-right: 1px solid {border};
}}
QSplitter::handle:hover {{
    background: {accent};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    border: none;
    width: 10px;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle {{
    background: {disabled};
    border-radius: 5px;
    min-height: 20px;
    min-width: 20px;
}}
QScrollBar::handle:hover {{
    background: {text_secondary};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border};
    border-radius: 3px;
    background: {surface};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}
"""


def build_stylesheet(palette: Palette) -> str:
    return _QSS_TEMPLATE.format(
        bg=palette.bg,
        surface=palette.surface,
        border=palette.border,
        text_primary=palette.text_primary,
        text_secondary=palette.text_secondary,
        accent=palette.accent,
        accent_hover=palette.accent_hover,
        disabled=palette.disabled,
    )


def build_qpalette(palette: Palette) -> QPalette:
    qp = QPalette()
    qp.setColor(QPalette.ColorRole.Window, QColor(palette.bg))
    qp.setColor(QPalette.ColorRole.WindowText, QColor(palette.text_primary))
    qp.setColor(QPalette.ColorRole.Base, QColor(palette.surface))
    qp.setColor(QPalette.ColorRole.AlternateBase, QColor(palette.bg))
    qp.setColor(QPalette.ColorRole.Text, QColor(palette.text_primary))
    qp.setColor(QPalette.ColorRole.Button, QColor(palette.surface))
    qp.setColor(QPalette.ColorRole.ButtonText, QColor(palette.text_primary))
    qp.setColor(QPalette.ColorRole.Highlight, QColor(palette.accent))
    qp.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    qp.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette.surface))
    qp.setColor(QPalette.ColorRole.ToolTipText, QColor(palette.text_primary))
    qp.setColor(QPalette.ColorRole.PlaceholderText, QColor(palette.text_secondary))
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(palette.disabled))
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(palette.disabled))
    return qp


def palette_for(mode: ThemeMode) -> Palette:
    return DARK if mode == "dark" else LIGHT


def load_theme_preference(settings: QSettings | None = None) -> ThemeMode:
    settings = settings or QSettings()
    value = settings.value(_SETTINGS_KEY, "light")
    return "dark" if value == "dark" else "light"


def save_theme_preference(mode: ThemeMode, settings: QSettings | None = None) -> None:
    settings = settings or QSettings()
    settings.setValue(_SETTINGS_KEY, mode)


class ThemeManager(QObject):
    """Owns the active theme mode, applies it to a QApplication, and persists
    the choice via QSettings across launches."""

    themeChanged = Signal(str)  # ThemeMode

    def __init__(self, mode: ThemeMode | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.mode: ThemeMode = mode or load_theme_preference()

    @property
    def current_palette(self) -> Palette:
        return palette_for(self.mode)

    def apply(self, app: QApplication) -> None:
        palette = self.current_palette
        app.setPalette(build_qpalette(palette))
        app.setStyleSheet(build_stylesheet(palette))

    def set_mode(self, mode: ThemeMode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        save_theme_preference(mode)
        app = QApplication.instance()
        if app is not None:
            self.apply(app)
        self.themeChanged.emit(mode)

    def toggle(self) -> None:
        self.set_mode("dark" if self.mode == "light" else "light")
