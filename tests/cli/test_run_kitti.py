"""."""

from pathlib import Path

from trackeval.cli.run_kitti import run


def test_run_kitti(files_dir: Path) -> None:
    """."""
    run(['--GT_FOLDER', str(files_dir / 'data/gt/kitti/kitti_2d_box_train'),
         '--TRACKERS_FOLDER', str(files_dir / 'data/trackers/kitti/kitti_2d_box_train')])
