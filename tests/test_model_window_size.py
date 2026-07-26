import joblib

from src.data_loader import load_dataframe
from src.pipeline import get_model_window_size, score_model, train_model
from src.synthetic_tdms import _make_normal_signals, write_tdms


def _load_normal_df(tmp_path, n_samples=4000, seed=0):
    path = tmp_path / "normal.tdms"
    write_tdms(path, _make_normal_signals(n_samples=n_samples, seed=seed))
    return load_dataframe(path)


def test_train_model_stashes_window_size_classic(tmp_path):
    df = _load_normal_df(tmp_path)
    model = train_model("classic", [df], window_size=200, step=None)

    assert model.window_size_ == 200
    assert model.step_ == 200  # step defaults to window_size
    assert get_model_window_size(model, "classic") == (200, 200)


def test_train_model_stashes_window_size_classic_explicit_step(tmp_path):
    df = _load_normal_df(tmp_path)
    model = train_model("classic", [df], window_size=200, step=100)

    assert model.window_size_ == 200
    assert model.step_ == 100


def test_get_model_window_size_deep(tmp_path):
    df = _load_normal_df(tmp_path, n_samples=2000)
    model = train_model("deep", [df], window_size=100, step=50, epochs=2)
    assert get_model_window_size(model, "deep") == (100, 50)


def test_get_model_window_size_returns_none_for_legacy_pickle_missing_attribute(tmp_path):
    df = _load_normal_df(tmp_path)
    model = train_model("classic", [df], window_size=200)

    # Simulate a model pickled *before* window_size_/step_ existed on
    # AnomalyDetector: delete them from this instance's __dict__ so they're
    # absent from the pickled state, the same as a genuinely old pickle would
    # be. (They still resolve via attribute lookup after unpickling --
    # dataclass fields with a plain `= None` default are also class
    # attributes, so a missing instance entry falls back to the class-level
    # default rather than raising AttributeError. get_model_window_size's
    # getattr(..., None) is extra insurance, not strictly required for this
    # case, but keeps the contract explicit regardless of dataclass internals.)
    del model.window_size_
    del model.step_
    legacy_path = tmp_path / "legacy_model.joblib"
    joblib.dump(model, legacy_path)

    loaded = joblib.load(legacy_path)
    assert "window_size_" not in loaded.__dict__
    assert get_model_window_size(loaded, "classic") == (None, None)

    # And scoring against this "legacy" model still works fine despite the
    # missing attributes -- score() never touches window_size_/step_.
    scores = score_model(loaded, "classic", [df], window_size=200)
    assert "anomaly_score" in scores.columns
