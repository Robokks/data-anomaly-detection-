from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon, QPalette

from gui import theme
from gui.icons import ICONS, colored_icon, icon


def test_build_stylesheet_contains_palette_colors():
    css = theme.build_stylesheet(theme.LIGHT)
    assert theme.LIGHT.accent in css
    assert theme.LIGHT.surface in css

    dark_css = theme.build_stylesheet(theme.DARK)
    assert dark_css != css
    assert theme.DARK.accent in dark_css


def test_build_qpalette_maps_tokens_to_roles():
    qp = theme.build_qpalette(theme.DARK)
    assert qp.color(QPalette.ColorRole.Window).name() == theme.DARK.bg
    assert qp.color(QPalette.ColorRole.Highlight).name() == theme.DARK.accent


def test_plot_colors_differ_by_mode():
    assert theme.plot_colors("light") == theme.PLOT_COLORS_LIGHT
    assert theme.plot_colors("dark") == theme.PLOT_COLORS_DARK
    assert theme.plot_colors("light") != theme.plot_colors("dark")


def test_theme_manager_toggle_emits_signal(qtbot):
    manager = theme.ThemeManager(mode="light")
    assert manager.mode == "light"

    with qtbot.waitSignal(manager.themeChanged, timeout=1000) as blocker:
        manager.toggle()

    assert blocker.args == ["dark"]
    assert manager.mode == "dark"

    manager.toggle()
    assert manager.mode == "light"


def test_theme_manager_apply_sets_app_style(qapp):
    manager = theme.ThemeManager(mode="dark")
    manager.apply(qapp)
    assert qapp.styleSheet() != ""
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == theme.DARK.bg


def test_theme_preference_round_trips_via_qsettings(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    assert theme.load_theme_preference(settings) == "light"

    theme.save_theme_preference("dark", settings)
    assert theme.load_theme_preference(settings) == "dark"


def test_colored_icon_returns_nonnull_icon_for_every_glyph():
    for name in ICONS:
        result = icon(name, "#ffffff", size=16)
        assert isinstance(result, QIcon)
        assert not result.isNull()


def test_colored_icon_accepts_qcolor_and_hex_string():
    from PySide6.QtGui import QColor

    a = colored_icon(ICONS["sun"], QColor("#123456"))
    b = colored_icon(ICONS["sun"], "#123456")
    assert not a.isNull()
    assert not b.isNull()
