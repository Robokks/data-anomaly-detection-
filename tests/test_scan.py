import pandas as pd

from src.data_loader import scan_directory, scan_directory_with_info
from src.synthetic_tdms import _make_normal_signals, write_tdms


def _touch_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"vibration": [1.0, 2.0, 3.0]}).to_csv(path, index=False)


def test_scan_directory_finds_mixed_extensions_any_depth_and_naming(tmp_path):
    _touch_csv(tmp_path / "run_001.csv")
    _touch_csv(tmp_path / "sub" / "weird name (final) v2.csv")
    _touch_csv(tmp_path / "sub" / "deeper" / "unrelated_name.csv")
    write_tdms(tmp_path / "sub" / "rig_data.tdms", _make_normal_signals(n_samples=100, seed=0))
    (tmp_path / "readme.txt").write_text("not a data file")

    found = scan_directory(tmp_path, recursive=True)
    names = {p.name for p in found}

    assert names == {"run_001.csv", "weird name (final) v2.csv", "unrelated_name.csv", "rig_data.tdms"}


def test_scan_directory_non_recursive_only_top_level(tmp_path):
    _touch_csv(tmp_path / "top.csv")
    _touch_csv(tmp_path / "sub" / "nested.csv")

    found = scan_directory(tmp_path, recursive=False)
    assert {p.name for p in found} == {"top.csv"}


def test_scan_directory_empty_returns_empty_list(tmp_path):
    assert scan_directory(tmp_path) == []


def test_scan_directory_with_info_probes_each_file(tmp_path):
    _touch_csv(tmp_path / "a.csv")
    _touch_csv(tmp_path / "b.csv")

    infos = scan_directory_with_info(tmp_path)
    assert len(infos) == 2
    assert all(info.error is None for info in infos)
    assert all(info.channels == ["vibration"] for info in infos)
