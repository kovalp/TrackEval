"""."""

from pathlib import Path

import pytest

from trackeval.cli.run_kitti import run


def test_run_kitti(files_dir: Path, capsys: pytest.CaptureFixture) -> None:
    """."""
    run(['--GT_FOLDER', str(files_dir / 'data/gt/kitti/kitti_2d_box_train'),
         '--TRACKERS_FOLDER', str(files_dir / 'data/trackers/kitti/kitti_2d_box_train')])
    stdout = capsys.readouterr().out
    ref = 'COMBINED                           45.891    69.229    47.232    71.014    74.912'
    assert ref in stdout

