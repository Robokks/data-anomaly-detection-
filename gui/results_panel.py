"""Results panel: flagged-window table after training/scoring, with CSV export."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scores: pd.DataFrame | None = None

        layout = QVBoxLayout(self)

        self.summary_label = QLabel("No results yet -- train a model to see anomaly scores here.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 0)
        layout.addWidget(self.table)

        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export CSV...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_clicked)
        export_row.addWidget(self.export_button)
        layout.addLayout(export_row)

    def set_scores(self, scores: pd.DataFrame) -> None:
        self._scores = scores
        n_anom = int(scores["is_anomaly"].sum())
        rate = n_anom / len(scores) if len(scores) else 0.0
        self.summary_label.setText(f"{n_anom} / {len(scores)} windows flagged as anomalous ({rate:.1%}).")
        self.export_button.setEnabled(True)
        self._populate_table(scores)

    def _populate_table(self, scores: pd.DataFrame) -> None:
        flagged = scores[scores["is_anomaly"]]
        columns = list(scores.columns)
        self.table.setColumnCount(len(columns) + 1)
        self.table.setHorizontalHeaderLabels(["window"] + columns)
        self.table.setRowCount(len(flagged))
        for row, (idx, values) in enumerate(flagged.iterrows()):
            self.table.setItem(row, 0, QTableWidgetItem(str(idx)))
            for col_i, col in enumerate(columns, start=1):
                self.table.setItem(row, col_i, QTableWidgetItem(f"{values[col]}"))

    def _on_export_clicked(self) -> None:
        if self._scores is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export scores", "scores.csv", "CSV files (*.csv)")
        if path:
            self._scores.to_csv(path, index_label=self._scores.index.name or "window")
