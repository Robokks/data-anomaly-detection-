from pathlib import Path

from src.data_loader import scan_directory
from src.session_grouping import (
    MachineType,
    Phase,
    group_files,
    load_session,
)
from src.synthetic_sessions import (
    write_endurance_session,
    write_motor_test_bench_session,
    write_transmission_session,
)
from src.synthetic_tdms import _make_normal_signals, write_tdms


def test_general_each_file_is_its_own_session(tmp_path):
    for i in range(3):
        write_tdms(tmp_path / f"run_{i}.tdms", _make_normal_signals(n_samples=50, seed=i))

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.GENERAL)

    assert len(sessions) == 3
    for session in sessions:
        assert len(session.files) == 1
        assert session.uut_id == session.files[0].stem
        assert session.phase_map == {}
        assert session.pocket_map == {}


def test_transmission_groups_one_uut_with_all_phases_tagged(tmp_path):
    write_transmission_session(
        tmp_path,
        uut_id="unit001",
        n_samples={"ramp_up": 100, "steady_state": 300, "coasting": 200, "ramp_down": 150},
    )

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.TRANSMISSION)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.uut_id == "unit001"
    assert len(session.files) == 4
    assert set(session.phase_map.values()) == {
        Phase.RAMP_UP,
        Phase.RAMP_DOWN,
        Phase.STEADY_STATE,
        Phase.COASTING,
    }


def test_transmission_load_session_filters_to_steady_and_coasting(tmp_path):
    write_transmission_session(
        tmp_path,
        uut_id="unit001",
        n_samples={"ramp_up": 100, "steady_state": 300, "coasting": 200, "ramp_down": 150},
    )

    files = scan_directory(tmp_path)
    session = group_files(files, MachineType.TRANSMISSION)[0]

    df = load_session(session, phases={Phase.STEADY_STATE, Phase.COASTING})

    assert len(df) == 300 + 200
    assert set(df["phase"].unique()) == {Phase.STEADY_STATE.value, Phase.COASTING.value}


def test_transmission_unmatched_filename_included_as_unknown(tmp_path):
    write_transmission_session(tmp_path, uut_id="unit001", n_samples=50)
    # An extra file that doesn't match any phase keyword.
    write_tdms(
        tmp_path / "unit001" / "unit001_misc_extra.tdms",
        _make_normal_signals(n_samples=50, seed=99),
    )

    files = scan_directory(tmp_path)
    session = group_files(files, MachineType.TRANSMISSION)[0]

    assert len(session.files) == 5
    assert Phase.UNKNOWN in session.phase_map.values()

    df = load_session(session)
    assert len(df) == 50 * 5
    assert Phase.UNKNOWN.value in set(df["phase"].unique())


def test_motor_test_bench_groups_by_pocket_and_filters(tmp_path):
    write_motor_test_bench_session(
        tmp_path,
        uut_id="unit001",
        pockets=[1, 2],
        n_samples={1: 120, 2: 180},
    )

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.MOTOR_TEST_BENCH)

    assert len(sessions) == 1
    session = sessions[0]
    assert session.uut_id == "unit001"
    assert set(session.pocket_map.values()) == {"1", "2"}

    df = load_session(session, pockets={"1"})
    assert len(df) == 120


def test_motor_test_bench_fallback_pocket_id_for_unmatched_filename(tmp_path):
    out_dir = tmp_path / "unit001"
    write_tdms(out_dir / "unit001_weird_name.tdms", _make_normal_signals(n_samples=40, seed=1))

    files = scan_directory(tmp_path)
    session = group_files(files, MachineType.MOTOR_TEST_BENCH)[0]

    assert len(session.files) == 1
    # No pocket/step pattern found -> fallback pocket id is the filename stem.
    assert session.pocket_map[session.files[0]] == "unit001_weird_name"


def test_endurance_orders_chronologically_by_timestamp_and_concatenates(tmp_path):
    write_endurance_session(
        tmp_path,
        uut_id="unit001",
        n_chunks=3,
        n_samples=[100, 150, 120],
        naming="timestamp",
    )

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.ENDURANCE)

    assert len(sessions) == 1
    session = sessions[0]
    assert len(session.order) == 3
    # chronological order == the order chunks were written (0min, 1min, 2min apart)
    assert [p.name for p in session.order] == sorted(p.name for p in session.files)

    df = load_session(session)
    assert len(df) == 100 + 150 + 120


def test_endurance_orders_by_index_suffix_when_no_timestamp(tmp_path):
    write_endurance_session(
        tmp_path,
        uut_id="unit001",
        n_chunks=3,
        n_samples=[80, 90, 70],
        naming="index",
    )

    files = scan_directory(tmp_path)
    session = group_files(files, MachineType.ENDURANCE)[0]

    assert [p.name for p in session.order] == [
        "unit001_chunk001.tdms",
        "unit001_chunk002.tdms",
        "unit001_chunk003.tdms",
    ]

    df = load_session(session)
    assert len(df) == 80 + 90 + 70


def test_endurance_falls_back_to_mtime_when_no_pattern_matches(tmp_path):
    out_dir = tmp_path / "unit001"
    p1 = out_dir / "unit001_first.tdms"
    p2 = out_dir / "unit001_second.tdms"
    write_tdms(p1, _make_normal_signals(n_samples=30, seed=1))
    write_tdms(p2, _make_normal_signals(n_samples=30, seed=2))

    files = scan_directory(tmp_path)
    session = group_files(files, MachineType.ENDURANCE)[0]

    # Neither filename has a timestamp/index pattern -> mtime fallback, and
    # both files must still be present in the order (written p1 before p2).
    assert len(session.order) == 2
    assert session.order[0].stat().st_mtime <= session.order[1].stat().st_mtime

    df = load_session(session)
    assert len(df) == 60


def test_two_uuts_in_separate_subfolders_produce_two_sessions(tmp_path):
    write_transmission_session(tmp_path, uut_id="unit001", n_samples=50)
    write_transmission_session(tmp_path, uut_id="unit002", n_samples=50)

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.TRANSMISSION)

    assert len(sessions) == 2
    uut_ids = {s.uut_id for s in sessions}
    assert uut_ids == {"unit001", "unit002"}
    for s in sessions:
        assert len(s.files) == 4


def test_two_uuts_with_uut_pattern_in_same_folder(tmp_path):
    write_tdms(tmp_path / "unitA_pocket1_step1.tdms", _make_normal_signals(n_samples=40, seed=1))
    write_tdms(tmp_path / "unitA_pocket2_step1.tdms", _make_normal_signals(n_samples=40, seed=2))
    write_tdms(tmp_path / "unitB_pocket1_step1.tdms", _make_normal_signals(n_samples=40, seed=3))

    files = scan_directory(tmp_path)
    sessions = group_files(files, MachineType.MOTOR_TEST_BENCH, uut_pattern=r"(unit[A-Za-z]+)")

    assert len(sessions) == 2
    by_uid = {s.uut_id: s for s in sessions}
    assert set(by_uid) == {"unitA", "unitB"}
    assert len(by_uid["unitA"].files) == 2
    assert len(by_uid["unitB"].files) == 1
    assert by_uid["unitB"].pocket_map[by_uid["unitB"].files[0]] == "1"
