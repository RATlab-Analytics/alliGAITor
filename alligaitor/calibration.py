"""Multi-camera calibration for the alliGAITor rig, built on aniposelib.

Calibration is performed once against the fixed three-camera rig (left
side, right side, bottom-up) using a printed ChArUco board, and the
resulting camera parameters are reused for triangulating every session
recorded with that rig. Re-run calibration if a camera is repositioned.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
from aniposelib.boards import CharucoBoard, extract_points, extract_rtvecs, get_video_params, merge_rows
from aniposelib.cameras import CameraGroup
from aniposelib.utils import get_connections, get_initial_extrinsics

from alligaitor.config import CAMERA_ROLES, CalibrationConfig
from alligaitor.timing import shared_frame_key, video_fps

# Width, in seconds, of the shared time bucket used to match board
# detections across cameras (see alligaitor.timing). Chosen to comfortably
# exceed one frame period at every observed camera frame rate on this rig
# (as low as ~13 fps), so genuinely simultaneous detections land in the
# same or an adjacent bucket, while still being short enough to distinguish
# separate dwell positions during a calibration recording.
CALIBRATION_GRID_FPS = 10.0

# Matches aniposelib.cameras.CameraGroup.calibrate_rows' own default for
# min_corners_intrinsic, duplicated here to check it up front with a clear
# error message instead of letting calibrate_rows fail on an empty zip.
MIN_CORNERS_INTRINSIC = 9

# aniposelib.cameras.CameraGroup.calibrate_rows hardcodes 8 as the minimum
# ChArUco corners a frame needs to link two cameras' poses in the initial
# calibration graph (and to contribute a point to bundle adjustment) --
# this is not exposed as a parameter. calibrate() below reimplements that
# method with this made configurable (see _calibrate_rows), since the
# bottom camera's slit view can force a tradeoff between corner count and
# staying in a side camera's field of view: on some recordings, no frame
# reaches 8 corners on the bottom camera while also being visible to a
# side camera at all. 4 is the practical floor (cv2.aruco's ChArUco pose
# estimation needs at least 4 corners to solve for a pose); only lower a
# given recording's CalibrationConfig.min_corners_extrinsic toward that
# floor after confirming the recording has no frames clearing 8, since
# lower corner counts give a noisier initial pose estimate for that frame
# (bundle adjustment still refines it afterward, but a bad initial
# extrinsics guess can still leave it in a bad local optimum).
MIN_CORNERS_EXTRINSIC = 8

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

# A narrow alternative board geometry for the bottom camera's slit view,
# sized from a measured ~0.7 px/mm scale at that camera's working distance
# (derived from real detected marker corners against the above board's
# known layout, 2026-08-17). At this 35mm square size, markers project to
# ~19px per side -- well above the ~8px markers the 8-wide board produces
# there, which are too small to decode regardless of lighting or contrast
# (a DICT_4X4_50 marker encodes a 6x6 cell grid; at 8px total that is
# under one pixel per cell). 4x5 squares gives (4-1)*(5-1) = 12 usable
# ChArUco corners -- calibrate_rows() needs >=9 on a frame to initialize a
# camera's intrinsics and >=8 to use a frame for pose/extrinsics (see
# MIN_CORNERS_INTRINSIC below), so this has real margin above both
# thresholds rather than sitting exactly on one. Sized to 140mm x 175mm so
# the printed backing fits a Bambu Lab A1 mini's 180mm x 180mm build
# plate; this is the largest square size that keeps that margin within a
# single plate. Confirm 140mm still fits the slit width before printing.
# Requires printing a new physical target -- does not apply to
# already-recorded footage.
STRIP_BOARD_SQUARES_X = 4
STRIP_BOARD_SQUARES_Y = 5
STRIP_BOARD_SQUARE_LENGTH_MM = 35.0
STRIP_BOARD_MARKER_LENGTH_MM = 25.7

# Named board geometries, keyed by the ``board_preset`` a
# CalibrationConfig can specify -- different recordings may have used
# different physical printed boards. Each value is
# (squares_x, squares_y, square_length_mm, marker_length_mm).
BOARD_PRESETS = {
    "original": (BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_LENGTH_MM, BOARD_MARKER_LENGTH_MM),
    "strip": (
        STRIP_BOARD_SQUARES_X,
        STRIP_BOARD_SQUARES_Y,
        STRIP_BOARD_SQUARE_LENGTH_MM,
        STRIP_BOARD_MARKER_LENGTH_MM,
    ),
}

# aniposelib's CharucoBoard already overrides ArUco's default
# DetectorParameters with adaptiveThreshWinSizeMin/Max/Step tuned for
# markers that fill much of the frame (50/700/50, vs. OpenCV's own
# defaults of 3/23/10). The bottom camera's markers are far smaller than
# that in frame -- as small as ~10-12px per cell, from a board that only
# occupies a narrow strip of the tunnel-slit view -- so these values (and
# the related minimum-marker-size/shape-tolerance settings below) are
# relaxed toward the small-marker end. Verified against real footage: with
# aniposelib's stock parameters, a bottom-camera window with a clearly
# visible board (confirmed by eye) produced zero detected markers on every
# frame; with these relaxed values, the same window recovered real markers
# on most frames (up to 7 of 32 on the best frame tested). This alone does
# not guarantee a usable calibration frame (>=8 ChArUco corners) on any
# given recording -- corner interpolation still requires enough of those
# markers to be found on the same frame.
BOTTOM_CAMERA_MIN_MARKER_PERIMETER_RATE = 0.01
BOTTOM_CAMERA_POLYGONAL_APPROX_ACCURACY_RATE = 0.06
BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_MIN = 3
BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_MAX = 53
BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_STEP = 4
BOTTOM_CAMERA_MIN_CORNER_DISTANCE_RATE = 0.02


def build_board(
    relax_for_small_markers: bool = True,
    squares_x: int = BOARD_SQUARES_X,
    squares_y: int = BOARD_SQUARES_Y,
    square_length_mm: float = BOARD_SQUARE_LENGTH_MM,
    marker_length_mm: float = BOARD_MARKER_LENGTH_MM,
) -> CharucoBoard:
    """Construct the ChArUco board used for calibrating the alliGAITor rig.

    Args:
        relax_for_small_markers: Relax ArUco marker-size/shape detection
            tolerances for the bottom camera's small, distant view of the
            board (see the module-level ``BOTTOM_CAMERA_*`` constants).
            Applies to detection on every camera, not just the bottom one,
            since the same :class:`CharucoBoard` is shared across all
            three; this has not been observed to cause false positives on
            the side cameras' much larger, closer view of the board.
        squares_x: Board squares across its width. Defaults to the
            existing printed board's geometry (``BOARD_SQUARES_X``); pass
            ``STRIP_BOARD_SQUARES_X`` for the narrow, long alternative
            geometry sized for the bottom camera (see its constants'
            docstring above). Only change this for a board that has
            actually been printed to match.
        squares_y: Board squares along its length. See ``squares_x``.
        square_length_mm: Printed square edge length, in mm. See
            ``squares_x``.
        marker_length_mm: Printed marker edge length, in mm. See
            ``squares_x``.
    """
    board = CharucoBoard(
        squaresX=squares_x,
        squaresY=squares_y,
        square_length=square_length_mm,
        marker_length=marker_length_mm,
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

    if relax_for_small_markers:
        detector_params = board.charuco_detector.getDetectorParameters()
        detector_params.minMarkerPerimeterRate = BOTTOM_CAMERA_MIN_MARKER_PERIMETER_RATE
        detector_params.polygonalApproxAccuracyRate = BOTTOM_CAMERA_POLYGONAL_APPROX_ACCURACY_RATE
        detector_params.adaptiveThreshWinSizeMin = BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_MIN
        detector_params.adaptiveThreshWinSizeMax = BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_MAX
        detector_params.adaptiveThreshWinSizeStep = BOTTOM_CAMERA_ADAPTIVE_THRESH_WIN_SIZE_STEP
        detector_params.minCornerDistanceRate = BOTTOM_CAMERA_MIN_CORNER_DISTANCE_RATE
        board.charuco_detector.setDetectorParameters(detector_params)

    return board


# Whether relax_for_small_markers should default on for each preset. The
# BOTTOM_CAMERA_* relaxed detector parameters (in particular
# minMarkerPerimeterRate=0.01, versus ArUco's own default of ~0.03) were
# tuned for the original board's ~5-9px markers at the bottom camera's
# working distance. Applied to the much larger "strip" board's markers
# (~20px+), that low a perimeter floor does not improve detection -- it
# was verified to make ArUco consider a much larger set of small, mostly
# spurious low-detail contours as marker candidates across the whole
# frame, which measured at ~4s/frame (worst case observed: hangs a full
# calibration run) versus ~0.02s/frame with stock parameters on the same
# footage, for no corner-count benefit (stock parameters found frames
# with the board's full 12 corners on the same recording).
PRESET_DEFAULT_RELAX_FOR_SMALL_MARKERS = {
    "original": True,
    "strip": False,
}


def build_board_for_preset(preset: str, relax_for_small_markers: bool | None = None) -> CharucoBoard:
    """Construct the ChArUco board for a named geometry in :data:`BOARD_PRESETS`.

    Args:
        preset: Key into :data:`BOARD_PRESETS`.
        relax_for_small_markers: Passed through to :func:`build_board`.
            Defaults to this preset's entry in
            :data:`PRESET_DEFAULT_RELAX_FOR_SMALL_MARKERS` when left as
            ``None`` -- only the original board's markers are small enough
            to need it, and applying it unnecessarily is not just wasted
            effort but pathologically slow (see that dict's docstring).
    """
    if preset not in BOARD_PRESETS:
        raise ValueError(f"Unknown board preset '{preset}'; expected one of {sorted(BOARD_PRESETS)}")
    if relax_for_small_markers is None:
        relax_for_small_markers = PRESET_DEFAULT_RELAX_FOR_SMALL_MARKERS[preset]
    squares_x, squares_y, square_length_mm, marker_length_mm = BOARD_PRESETS[preset]
    return build_board(
        relax_for_small_markers=relax_for_small_markers,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
    )


def _detect_video_by_time(
    board: CharucoBoard, video_path: Path, fps: float, grid_fps: float, skip: int = 1
) -> List[dict]:
    """Detect ChArUco corners per frame, keyed by shared time bucket.

    Equivalent to :meth:`CharucoBoard.detect_video`, except each detection
    row's ``framenum`` is a time bucket shared across cameras (see
    :mod:`alligaitor.timing`) rather than this video's own raw frame index,
    so that :meth:`~aniposelib.cameras.CameraGroup.calibrate_rows` (which
    matches rows across cameras by exact ``framenum`` equality) lines up
    detections by estimated recording time instead.

    Args:
        board: ChArUco board to detect.
        video_path: Video to scan.
        fps: This video's own frame rate, used to convert its frame indices
            to recording time.
        grid_fps: Shared time-bucket rate (see
            :func:`alligaitor.timing.shared_frame_key`).
        skip: Process every ``skip``-th frame.

    Returns:
        Detection rows in the format
        :meth:`~aniposelib.cameras.CameraGroup.calibrate_rows` expects.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            try:
                corners, ids = board.detect_image(frame)
            except cv2.error:
                # OpenCV's CharucoDetector.detectBoard() can throw
                # (rather than return an empty result) on some frames --
                # observed on the larger-marker "strip" board geometry,
                # not just fail to find a board. Treat it as a miss on
                # this frame rather than aborting the whole scan.
                corners = ids = None
            if corners is not None and len(corners) > 0:
                key = shared_frame_key(frame_idx, fps, grid_fps)
                rows.append({"framenum": key, "corners": corners, "ids": ids})
        frame_idx += 1
    cap.release()

    return board.fill_points_rows(rows)


def _calibrate_rows(
    cgroup: CameraGroup,
    all_rows: List[List[dict]],
    board: CharucoBoard,
    min_corners_intrinsic: int = MIN_CORNERS_INTRINSIC,
    min_corners_extrinsic: int = MIN_CORNERS_EXTRINSIC,
    verbose: bool = True,
    **kwargs,
) -> float:
    """Equivalent to :meth:`CameraGroup.calibrate_rows`, with a configurable
    extrinsics corner-count threshold (see :data:`MIN_CORNERS_EXTRINSIC`).

    Adapted from ``aniposelib.cameras.CameraGroup.calibrate_rows``
    (BSD 2-Clause License, Copyright (c) 2019-2023 Lili Karashchuk); see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    for rows, camera in zip(all_rows, cgroup.cameras):
        size = camera.get_size()
        assert size is not None, f"Camera with name {camera.get_name()} has no specified frame size"
        objp, imgp = board.get_all_calibration_points(rows)
        mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= min_corners_intrinsic]
        objp, imgp = zip(*mixed)
        matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
        camera.set_camera_matrix(matrix.copy())
        camera.zero_distortions()

    for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)

    new_rows = [[r for r in rows if r["ids"].size >= min_corners_extrinsic] for rows in all_rows]
    merged = merge_rows(new_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    rtvecs = extract_rtvecs(merged)
    if verbose:
        print(get_connections(rtvecs, cgroup.get_names()))
    rvecs, tvecs = get_initial_extrinsics(rtvecs, cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    return cgroup.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)


def calibrate(
    config: CalibrationConfig,
    board: CharucoBoard | None = None,
    grid_fps: float = CALIBRATION_GRID_FPS,
    skip: int = 1,
) -> CameraGroup:
    """Calibrate the three-camera rig from role-named ChArUco board videos.

    Detections are matched across cameras by estimated recording time
    (frame index divided by each video's own frame rate), not by raw frame
    index, since the rig's cameras do not run at a matched frame rate. See
    :mod:`alligaitor.timing`.

    Args:
        config: Calibration video paths, output path, which physical board
            (``config.board_preset``) this recording used, and the
            extrinsics corner-count floor (``config.min_corners_extrinsic``)
            for this recording.
        board: ChArUco board definition. Defaults to
            ``build_board_for_preset(config.board_preset)``; pass this
            explicitly to override the config's preset.
        grid_fps: Shared time-bucket rate used to match detections across
            cameras. Defaults to :data:`CALIBRATION_GRID_FPS`.
        skip: Process every ``skip``-th frame of each video.

    Returns:
        The calibrated :class:`~aniposelib.cameras.CameraGroup`, already
        saved to ``config.output_path``.
    """
    if board is None:
        board = build_board_for_preset(config.board_preset)

    cgroup = CameraGroup.from_names(list(CAMERA_ROLES))

    all_rows = []
    for role, camera in zip(CAMERA_ROLES, cgroup.cameras):
        video_path = config.videos[role]
        params = get_video_params(str(video_path))
        camera.set_size((params["width"], params["height"]))
        fps = video_fps(video_path)
        all_rows.append(_detect_video_by_time(board, video_path, fps, grid_fps, skip=skip))

    for role, rows in zip(CAMERA_ROLES, all_rows):
        if not any(len(row["ids"]) >= MIN_CORNERS_INTRINSIC for row in rows):
            raise ValueError(
                f"No usable ChArUco detections for camera '{role}': no frame reached "
                f"the {MIN_CORNERS_INTRINSIC}-corner minimum needed for intrinsic "
                "calibration. Check board visibility and framing for this camera."
            )

    error = _calibrate_rows(
        cgroup, all_rows, board, min_corners_extrinsic=config.min_corners_extrinsic
    )
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
