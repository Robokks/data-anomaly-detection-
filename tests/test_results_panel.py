import pandas as pd

from gui.results_panel import ResultsPanel


def _make_scores():
    return pd.DataFrame(
        {
            "anomaly_score": [0.1, 5.0, 0.2, 4.5],
            "is_anomaly": [False, True, False, True],
        },
        index=pd.Index([0, 1, 2, 3], name="window"),
    )


def test_results_panel_shows_summary_and_enables_export(qtbot):
    panel = ResultsPanel()
    qtbot.addWidget(panel)

    panel.set_scores(_make_scores())

    assert "2 / 4" in panel.summary_label.text()
    assert panel.export_button.isEnabled()


def test_results_panel_table_only_shows_flagged_rows(qtbot):
    panel = ResultsPanel()
    qtbot.addWidget(panel)

    panel.set_scores(_make_scores())

    assert panel.table.rowCount() == 2
    window_values = {panel.table.item(row, 0).text() for row in range(panel.table.rowCount())}
    assert window_values == {"1", "3"}


def test_results_panel_export_writes_csv(qtbot, tmp_path, monkeypatch):
    panel = ResultsPanel()
    qtbot.addWidget(panel)
    panel.set_scores(_make_scores())

    out_path = tmp_path / "out.csv"
    monkeypatch.setattr(
        "gui.results_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(out_path), "CSV files (*.csv)"),
    )

    panel._on_export_clicked()

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert len(written) == 4
