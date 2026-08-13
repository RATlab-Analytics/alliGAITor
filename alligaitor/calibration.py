"""Multi-camera calibration for the alliGAITor rig, built on aniposelib.

Calibration is performed once against the fixed three-camera rig (left
side, right side, bottom-up) using a printed ChArUco board, and the
resulting camera parameters are reused for triangulating every session
recorded with that rig. Re-run calibration if a camera is repositioned.
"""

from __future__ import annotations

from pathlib import Path

from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

from alligaitor.config import CAMERA_ROLES, CalibrationConfig

# Board geometry, inferred from the printed board's filename
# (calib.io_CHARUCO_200x150_8x8_15_11_DICT_4X4.pdf): an 8x8-square ChArUco
# board on a 200x150 mm sheet, 15 mm checker squares, 11 mm ArUco markers,
# from the DICT_4X4_50 dictionary (aniposelib's default for marker_bits=4,
# dict_size=50). Verify these values against the calib.io generation
# settings before calibrating -- an incorrect square/marker length will not
# raise an error, it will silently produce an incorrectly scaled
# calibration.
BOARD_SQUARES_X = 8
BOARD_SQUARES_Y = 8
BOARD_SQUARE_LENGTH_MM = 15.0
BOARD_MARKER_LENGTH_MM = 11.0


def build_board() -> CharucoBoard:
    """Construct the ChArUco board used for calibrating the alliGAITor rig."""
    return CharucoBoard(
        squaresX=BOARD_SQUARES_X,
        squaresY=BOARD_SQUARES_Y,
        square_length=BOARD_SQUARE_LENGTH_MM,
        marker_length=BOARD_MARKER_LENGTH_MM,
    )


def calibrate(config: CalibrationConfig, board: CharucoBoard | None = None) -> CameraGroup:
    """Calibrate the three-camera rig from role-named ChArUco board videos.

    Args:
        config: Calibration video paths and the output path for the saved
            calibration.
        board: ChArUco board definition. Defaults to :func:`build_board`.

    Returns:
        The calibrated :class:`~aniposelib.cameras.CameraGroup`, already
        saved to ``config.output_path``.
    """
    if board is None:
        board = build_board()

    cgroup = CameraGroup.from_names(list(CAMERA_ROLES))
    videos = [[str(config.videos[role])] for role in CAMERA_ROLES]

    error, _ = cgroup.calibrate_videos(videos, board)
    print(f"Calibration complete. Mean reprojection error: {error:.4f} px")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    cgroup.dump(str(config.output_path))
    return cgroup


def load(config: CalibrationConfig) -> CameraGroup:
    """Load a previously saved calibration for the three-camera rig."""
    if not config.output_path.exists():
        raise FileNotFoundError(
            f"No calibration found at {config.output_path}. Run calibration first."
        )
    return CameraGroup.load(str(config.output_path))
