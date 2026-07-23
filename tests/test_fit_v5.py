from __future__ import annotations

import subprocess
import sys

from analysis import run_fit_v5


def test_quarter_turns_can_be_overridden_for_one_capture_role() -> None:
    config = {
        "quarter_turns_ccw": 0,
        "role_quarter_turns_ccw": {"black_start": 3},
    }

    assert run_fit_v5._quarter_turns_for_role(config, "black_start") == 3
    assert run_fit_v5._quarter_turns_for_role(config, "train_A") == 0


def test_run_fit_v6_can_be_loaded_from_its_direct_script_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys; "
                "sys.path=[p for p in sys.path if p not in ('', '.') and not p.endswith('/glasses')]; "
                "sys.path.insert(0, 'analysis'); "
                "runpy.run_path('analysis/run_fit_v6.py', run_name='fit_v6_test')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
