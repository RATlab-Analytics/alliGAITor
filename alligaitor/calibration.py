"""Multi-camera calibration for the alliGAITor rig, built on aniposelib.

Calibration is performed once against the fixed three-camera rig (left
side, right side, bottom-up) using a printed ChArUco board, and the
resulting camera parameters are reused for triangulating every session
recorded with that rig. Re-run calibration if a camera is repositioned.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

from alligaitor.config import CAMERA_ROLES, CalibrationConfig

# Board geometry, inferred from the printed board's filename
# (calib.io_CHARUCO_200x150_8x8_15_11_DICT_4X4.pdf): an 8x8-square ChArUco
# board on a 200x150 mm sheet, 15 mm checker squares, 11 mm ArUco markers,
# from the DICT_4X4_50 dictionary (aniposelib's default for marker_bits=4,
# dict_size=50). Both the 8x8 square count and the DICT_4X4 family have
# been directly verified against real footage: an OpenCV chessboard-corner
# search on a sharp frame confirmed a 7x7 internal-corner grid (= 8x8
# squares), and raw ArUco marker decoding on that same frame found real,
# in-range marker IDs (0-31) consistent with this exact board config.
BOARD_SQUARES_X = 8
BOARD_SQUARES_Y = 8
BOARD_SQUARE_LENGTH_MM = 15.0
BOARD_MARKER_LENGTH_MM = 11.0


def build_board() -> CharucoBoard:
    """Construct the ChArUco board used for calibrating the alliGAITor rig."""
    board = CharucoBoard(
        squaresX=BOARD_SQUARES_X,
        squaresY=BOARD_SQUARES_Y,
        square_length=BOARD_SQUARE_LENGTH_MM,
        marker_length=BOARD_MARKER_LENGTH_MM,
    )

    # OpenCV's CharucoParameters.checkMarkers (on by default) runs an extra
    # geometric-consistency check on each detected marker before letting it
    # contribute to corner interpolation. Verified directly against real
    # footage: on a sharp, well-lit test frame, raw ArUco detection found 11
    # genuine, correctly-decoded, in-range markers, yet detectBoard() still
    # produced zero charuco corners with checkMarkers on -- disabling it
    # alone recovered 28 corners from those same 11 markers, and a clean
    # synthetic reference board still detects all 49/49 corners with it
    # off, so this isn't loosening things enough to accept garbage. Likely
    # too strict for this footage's compression/lighting-induced corner
    # localization noise. aniposelib's CharucoBoard doesn't expose this
    # setting itself, so it's patched in here after construction.
    charuco_params = board.charuco_detector.getCharucoParameters()
    charuco_params.checkMarkers = False
    board.charuco_detector.setCharucoParameters(charuco_params)

    return board


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
