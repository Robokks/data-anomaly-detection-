"""Flat single-color SVG glyphs, embedded as strings (no asset files -- keeps
PyInstaller packaging untouched, see ``packaging/app.spec``), plus a helper
that renders and recolors one at the current theme's icon color so the same
glyph set works for both light and dark mode.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_STROKE = 'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'

ICONS: dict[str, str] = {
    "folder": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M3 6a1 1 0 0 1 1-1h4l2 2h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/></svg>',
    "train": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M8 5l11 7-11 7z"/></svg>',
    "signature": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M3 15c2-1 3-4 4-4s1 3 2 3 2-6 3-6 1 5 2 5 2-3 3-3 1 2 4 2"/></svg>',
    "live": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<circle cx="12" cy="12" r="2.5"/><path d="M7.5 7.5a6.5 6.5 0 0 0 0 9M16.5 7.5a6.5 6.5 0 0 1 0 9M4.5 4.5a11 11 0 0 0 0 15M19.5 4.5a11 11 0 0 1 0 15"/></svg>',
    "save": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/></svg>',
    "export": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M12 3v12M8 7l4-4 4 4"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></svg>',
    "start": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M6 4l14 8-14 8z"/></svg>',
    "stop": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<rect x="5" y="5" width="14" height="14" rx="2"/></svg>',
    "sun": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8l1.8-1.8M18 6l1.8-1.8"/></svg>',
    "moon": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {_STROKE}>'
    '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5z"/></svg>',
}


def colored_icon(svg_str: str, color: QColor | str, size: int = 20) -> QIcon:
    """Renders ``svg_str`` (using ``currentColor`` for its stroke) tinted to
    ``color``, at ``size``x``size`` px. Re-tinting at render time (rather than
    keeping separate light/dark icon sets) is what lets one glyph set work
    for both themes.
    """
    if isinstance(color, str):
        color = QColor(color)

    colored_svg = svg_str.replace("currentColor", color.name())
    renderer = QSvgRenderer(QByteArray(colored_svg.encode("utf-8")))

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    return QIcon(pixmap)


def icon(name: str, color: QColor | str, size: int = 20) -> QIcon:
    return colored_icon(ICONS[name], color, size)
