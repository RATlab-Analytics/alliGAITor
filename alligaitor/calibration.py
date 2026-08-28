"""Multi-camera calibration for the alliGAITor rig, built on aniposelib.

Calibration is performed once against the fixed three-camera rig (left
side, right side, bottom-up) using a printed calibration board -- either a
ChArUco board or, for recordings where the bottom camera can only see the
board at a steep angle through the tunnel slit, a flat AprilTag marker-grid
board (see :class:`AprilGridBoard`) -- and the resulting camera parameters
are reused for triangulating every session recorded with that rig. Re-run
calibration if a camera is repositioned.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from aniposelib.boards import (
    CalibrationObject,
    CharucoBoard,
    extract_points,
    extract_rtvecs,
    get_video_params,
    merge_rows,
)
from aniposelib.cameras import CameraGroup
from aniposelib.utils import (
    find_calibration_pairs,
    get_calibration_graph,
    get_connections,
    get_rtvec,
    make_M,
    mean_transform,
    select_matrices,
)
from scipy.linalg import inv
from tqdm import tqdm

from alligaitor.config import CAMERA_ROLES, CalibrationConfig
from alligaitor.timing import shared_frame_key, video_fps

# Width, in seconds, of the shared time bucket used to match board
# detections across cameras (see alligaitor.timing). Wide enough to exceed
# one frame period at any camera's frame rate while still distinguishing
# separate dwell positions during a calibration recording.
CALIBRATION_GRID_FPS = 10.0

# Matches aniposelib's own default for min_corners_intrinsic, checked here
# up front with a clear error instead of letting calibrate_rows fail on an
# empty zip.
MIN_CORNERS_INTRINSIC = 9

# Minimum matched points for AprilGridBoard extrinsics pose estimation (4
# is the practical floor cv2.solvePnP needs). ChArUco recordings instead
# always use aniposelib's own hardcoded 7/8 floors.
MIN_CORNERS_EXTRINSIC = 8

# Board geometry for the printed ChArUco board: 8x8 squares on a 200x150mm
# sheet, 15mm checker squares, 11mm ArUco markers, DICT_4X4_50 dictionary.
BOARD_SQUARES_X = 8
BOARD_SQUARES_Y = 8
BOARD_SQUARE_LENGTH_MM = 15.0
BOARD_MARKER_LENGTH_MM = 11.0

# A narrow alternative board geometry for the bottom camera's slit view,
# where the standard board's markers project too small to decode.
# Requires printing a new physical target.
STRIP_BOARD_SQUARES_X = 4
STRIP_BOARD_SQUARES_Y = 5
STRIP_BOARD_SQUARE_LENGTH_MM = 35.0
STRIP_BOARD_MARKER_LENGTH_MM = 25.7

# A flat AprilTag marker-grid board -- replaces the "strip" ChArUco board
# for the bottom camera's steep viewing angle, since each marker's corners
# decode directly without ChArUco's unreliable corner-interpolation step.
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
APRILTAG_MARKERS_X = 2
APRILTAG_MARKERS_Y = 3
APRILTAG_MARKER_LENGTH_MM = 46.0
APRILTAG_MARKER_SEPARATION_MM = 8.0
APRILTAG_MARGIN_MM = 10.0

# Named ChArUco board geometries keyed by ``board_preset``: each value is
# (squares_x, squares_y, square_length_mm, marker_length_mm). The
# ``"apriltag"`` preset is handled separately in build_board_for_preset().
BOARD_PRESETS = {
    "original": (BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_LENGTH_MM, BOARD_MARKER_LENGTH_MM),
    "strip": (
        STRIP_BOARD_SQUARES_X,
        STRIP_BOARD_SQUARES_Y,
        STRIP_BOARD_SQUARE_LENGTH_MM,
        STRIP_BOARD_MARKER_LENGTH_MM,
    ),
}

# Relaxed ArUco marker detector parameters for the bottom camera's small,
# distant markers; aniposelib's stock parameters (tuned for markers that
# fill much of the frame) miss them entirely.
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

    # Disable OpenCV's extra geometric-consistency check on detected
    # markers before corner interpolation -- too strict for this footage's
    # compression/lighting noise, and not exposed by aniposelib's
    # CharucoBoard so it's patched in here.
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
# BOTTOM_CAMERA_* relaxed parameters are tuned for the original board's
# tiny markers; applied to the larger "strip" board's markers they add no
# detection benefit and are pathologically slow.
PRESET_DEFAULT_RELAX_FOR_SMALL_MARKERS = {
    "original": True,
    "strip": False,
}


def build_apriltag_board(
    markers_x: int = APRILTAG_MARKERS_X,
    markers_y: int = APRILTAG_MARKERS_Y,
    marker_length_mm: float = APRILTAG_MARKER_LENGTH_MM,
    marker_separation_mm: float = APRILTAG_MARKER_SEPARATION_MM,
    aruco_dict: int = APRILTAG_DICTIONARY,
    min_points_extrinsic: int = MIN_CORNERS_EXTRINSIC,
) -> "AprilGridBoard":
    """Construct the flat AprilTag marker-grid board (see
    :data:`APRILTAG_MARKERS_X` and neighbors for the rationale behind the
    default geometry).

    Args:
        min_points_extrinsic: Passed through to :class:`AprilGridBoard`;
            see its docstring.
    """
    return AprilGridBoard(
        markers_x=markers_x,
        markers_y=markers_y,
        marker_length=marker_length_mm,
        marker_separation=marker_separation_mm,
        aruco_dict=aruco_dict,
        min_points_extrinsic=min_points_extrinsic,
    )


def build_board_for_preset(
    preset: str,
    relax_for_small_markers: bool | None = None,
    min_corners_extrinsic: int = MIN_CORNERS_EXTRINSIC,
) -> Union[CharucoBoard, "AprilGridBoard"]:
    """Construct the calibration board for a named geometry.

    Args:
        preset: ``"apriltag"`` for the flat AprilTag marker-grid board (see
            :func:`build_apriltag_board`), or a key into
            :data:`BOARD_PRESETS` for a named ChArUco board geometry.
        relax_for_small_markers: Passed through to :func:`build_board` for
            ChArUco presets; ignored for ``"apriltag"``. Defaults to this
            preset's entry in :data:`PRESET_DEFAULT_RELAX_FOR_SMALL_MARKERS`
            when left as ``None`` -- only the original board's markers are
            small enough to need it, and applying it unnecessarily is not
            just wasted effort but pathologically slow (see that dict's
            docstring).
        min_corners_extrinsic: Passed through to :func:`build_apriltag_board`
            as ``min_points_extrinsic`` for ``"apriltag"``; ignored for
            ChArUco presets, which use aniposelib's own hardcoded floor
            instead (see :func:`calibrate`).
    """
    if preset == "apriltag":
        return build_apriltag_board(min_points_extrinsic=min_corners_extrinsic)
    if preset not in BOARD_PRESETS:
        raise ValueError(
            f"Unknown board preset '{preset}'; expected 'apriltag' or one of {sorted(BOARD_PRESETS)}"
        )
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


class AprilGridBoard(CalibrationObject):
    """A flat grid of independently-decodable AprilTag markers, with no
    checkerboard/corner-interpolation step -- see the module-level
    ``APRILTAG_*`` constants for why this replaces ChArUco for the bottom
    camera's fixed 45-degree slit view.

    Implements aniposelib's :class:`~aniposelib.boards.CalibrationObject`
    interface, so it plugs into aniposelib's existing machinery unmodified.
    Each marker has no interpolated corner, so each of the board's
    ``n_markers`` markers occupies 4 consecutive point slots (its 4
    corners): marker id ``i`` occupies slots ``[4*i, 4*i+4)``.
    """

    def __init__(
        self,
        markers_x: int,
        markers_y: int,
        marker_length: float,
        marker_separation: float,
        aruco_dict: int = APRILTAG_DICTIONARY,
        manually_verify: bool = False,
        min_points_extrinsic: int = MIN_CORNERS_EXTRINSIC,
    ):
        self.markers_x = markers_x
        self.markers_y = markers_y
        self.marker_length = marker_length
        self.marker_separation = marker_separation
        self.manually_verify = manually_verify
        # Minimum matched points estimate_pose_points() requires before
        # attempting solvePnP. Stored on the board rather than passed as a
        # call argument since aniposelib's estimate_pose_rows() has no way
        # to pass extra arguments through.
        self.min_points_extrinsic = min_points_extrinsic

        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.board = cv2.aruco.GridBoard(
            [markers_x, markers_y], marker_length, marker_separation, self.dictionary
        )

        # AprilTag's own quad-detection + decoding is a different algorithm
        # from the generic ArUco corner-refinement CharucoBoard uses above
        # (CORNER_REFINE_CONTOUR) -- CORNER_REFINE_APRILTAG is OpenCV's
        # refinement mode tuned specifically for it.
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # Build the fixed per-point object-point template by marker id
        # (not by getObjPoints()'s array order, which happened to match id
        # order in testing but isn't documented to be guaranteed) -- see
        # this class's docstring for the 4-points-per-marker-id layout.
        ids = np.asarray(self.board.getIds()).ravel()
        self.n_markers = len(ids)
        obj_points_by_marker = self.board.getObjPoints()
        self.objPoints = np.zeros((self.n_markers * 4, 3), dtype=np.float64)
        for marker_id, pts in zip(ids, obj_points_by_marker):
            self.objPoints[4 * marker_id : 4 * marker_id + 4] = np.asarray(pts, dtype=np.float64).reshape(4, 3)

        self.empty_detection = np.full((self.n_markers * 4, 1, 2), np.nan)

    def get_size(self):
        return (self.markers_x, self.markers_y)

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def get_object_points(self):
        return self.objPoints

    def draw(self, size, margin_mm: float | None = None):
        """Render a printable board image. ``size`` is ``(width, height)``
        in pixels; ``margin_mm`` defaults to :data:`APRILTAG_MARGIN_MM`
        (the outer white quiet zone AprilTag detection needs -- see that
        constant's docstring). Caller is responsible for choosing ``size``
        with a pixel scale consistent with this board's mm dimensions."""
        if margin_mm is None:
            margin_mm = APRILTAG_MARGIN_MM
        board_width_mm = self.markers_x * self.marker_length + (self.markers_x - 1) * self.marker_separation
        px_per_mm = size[0] / (board_width_mm + 2 * margin_mm)
        margin_px = int(round(margin_mm * px_per_mm))
        return self.board.generateImage(size, marginSize=margin_px, borderBits=1)

    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or ids is None or len(corners) == 0:
            return out
        for marker_id, marker_corners in zip(np.asarray(ids).ravel(), corners):
            if 0 <= marker_id < self.n_markers:
                out[4 * marker_id : 4 * marker_id + 4] = np.asarray(marker_corners, dtype=np.float64).reshape(4, 1, 2)
        return out

    def detect_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        try:
            corners, ids, _ = self.detector.detectMarkers(gray)
        except cv2.error:
            corners = ids = None

        if ids is None or len(ids) == 0:
            return [], []

        if self.manually_verify and not self.manually_verify_board_detection(gray, corners, ids):
            return [], []

        return corners, ids

    def manually_verify_board_detection(self, image, corners, ids=None):
        height, width = image.shape[:2]
        image = cv2.aruco.drawDetectedMarkers(np.asarray(image).copy(), corners, ids)
        cv2.putText(
            image, '(a) Accept (d) Reject', (int(width / 1.35), int(height / 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA,
        )
        cv2.imshow('verify_detection', image)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                return True
            elif key == ord('d'):
                return False

    def estimate_pose_points(self, camera, corners, ids):
        """Satisfies :meth:`CalibrationObject.estimate_pose_points`. Called
        by aniposelib's own (unmodified) ``CalibrationObject.estimate_pose_rows()``
        during calibration -- see :func:`_calibrate_rows`.

        ``self.board.matchImagePoints()`` (OpenCV's ``cv2.aruco.Board``
        method) resolves detected marker corners/ids to this board's fixed
        object-point layout; this is the same correspondence lookup a
        ChArUco board's own ``matchImagePoints()`` would do, just against a
        marker grid instead of interpolated checkerboard corners. Requires
        ``self.min_points_extrinsic`` matched points (4 is the practical
        floor -- ``solvePnP`` needs at least 4 point correspondences) rather
        than a fixed number, since a good calibration wants a much higher
        bar than the bare minimum where the data supports it; see
        :data:`MIN_CORNERS_EXTRINSIC`.
        """
        if corners is None or ids is None or len(corners) == 0:
            return None, None
        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or len(obj_points) < self.min_points_extrinsic:
            return None, None
        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        if ret:
            return rvec, tvec
        return None, None


def _detect_video_by_time(
    board: CalibrationObject,
    video_path: Path,
    fps: float,
    grid_fps: float,
    skip: int = 1,
    progress_desc: str | None = None,
) -> List[dict]:
    """Detect board corners/markers per frame, keyed by shared time bucket.

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
        progress_desc: Label shown on the frames-processed progress bar
            (e.g. the camera role); pass ``None`` to omit the bar.

    Returns:
        Detection rows in the format
        :meth:`~aniposelib.cameras.CameraGroup.calibrate_rows` expects.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Some containers report an inaccurate/zero frame count from metadata
    # alone; fall back to an unbounded bar (still shows a live count and
    # rate, just no percentage/ETA) rather than a misleading total.
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = n_frames if n_frames > 0 else None

    rows = []
    frame_idx = 0
    with tqdm(total=total, desc=progress_desc, unit="frame", disable=progress_desc is None) as pbar:
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
            pbar.update(1)
    cap.release()

    return board.fill_points_rows(rows)


def _mean_transform_robust(M_list, approx=None, error: float = 0.5):
    """Equivalent to ``aniposelib.utils.mean_transform_robust``, except it
    falls back to the un-filtered mean instead of crashing
    (``mean_transform([])``) when every candidate transform in ``M_list``
    falls outside ``error`` of ``approx``. Used only for the
    ``"apriltag"`` calibration standard (see :func:`_calibrate_rows`).

    Adapted from ``aniposelib.utils.mean_transform_robust`` (BSD 2-Clause
    License, Copyright (c) 2019-2023 Lili Karashchuk); see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = [M for M in M_list if np.max(np.abs((M - approx)[:3, :3])) < error]
    if not M_list_robust:
        M_list_robust = M_list
    return mean_transform(M_list_robust)


def _get_transform(rtvecs, left: int, right: int):
    """Equivalent to ``aniposelib.utils.get_transform``, using
    :func:`_mean_transform_robust` instead of the upstream function that
    can crash on a small, scattered sample (see its docstring).

    Adapted from ``aniposelib.utils.get_transform`` (BSD 2-Clause
    License, Copyright (c) 2019-2023 Lili Karashchuk); see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    L = []
    for dix in range(rtvecs.shape[1]):
        d = rtvecs[:, dix]
        good = ~np.isnan(d[:, 0])
        if good[left] and good[right]:
            M_left = make_M(d[left, 0:3], d[left, 3:6])
            M_right = make_M(d[right, 0:3], d[right, 3:6])
            M = np.matmul(M_left, inv(M_right))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    return _mean_transform_robust(L, M_mean, error=0.5)


def _get_initial_extrinsics(rtvecs, cam_names=None):
    """Equivalent to ``aniposelib.utils.get_initial_extrinsics``, using
    :func:`_get_transform` for each camera pair instead of the upstream
    function.

    Adapted from ``aniposelib.utils.get_initial_extrinsics`` (BSD 2-Clause
    License, Copyright (c) 2019-2023 Lili Karashchuk); see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    graph = get_calibration_graph(rtvecs, cam_names)
    pairs = find_calibration_pairs(graph, source=0)

    extrinsics = dict()
    source = pairs[0][0]
    extrinsics[source] = np.identity(4)
    for a, b in pairs:
        ext = _get_transform(rtvecs, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])

    n_cams = rtvecs.shape[0]
    rvecs = []
    tvecs = []
    for cnum in range(n_cams):
        rvec, tvec = get_rtvec(extrinsics[cnum])
        rvecs.append(rvec)
        tvecs.append(tvec)
    return np.array(rvecs), np.array(tvecs)


def _calibrate_rows(
    cgroup: CameraGroup,
    all_rows: List[List[dict]],
    board: CalibrationObject,
    min_corners_intrinsic: int = MIN_CORNERS_INTRINSIC,
    verbose: bool = True,
    **kwargs,
) -> float:
    """Equivalent to :meth:`CameraGroup.calibrate_rows`, except the
    extrinsics-linking row filter checks whether pose estimation actually
    succeeded (``row["rvec"] is not None``) instead of upstream's
    ``row["ids"].size >= 8``, which undercounts for a marker-grid board
    (1 marker = 4 points, not 1). Used only for the ``"apriltag"``
    calibration standard; ChArUco recordings use aniposelib's own
    :meth:`CameraGroup.calibrate_rows` instead.

    Adapted from ``aniposelib.cameras.CameraGroup.calibrate_rows``
    (BSD 2-Clause License, Copyright (c) 2019-2023 Lili Karashchuk); see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    for rows, camera in zip(all_rows, cgroup.cameras):
        size = camera.get_size()
        assert size is not None, f"Camera with name {camera.get_name()} has no specified frame size"
        objp, imgp = board.get_all_calibration_points(rows, min_points=min_corners_intrinsic)
        matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
        camera.set_camera_matrix(matrix.copy())
        camera.zero_distortions()

    for i, (rows, cam) in enumerate(zip(all_rows, cgroup.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, rows)

    new_rows = [[r for r in rows if r["rvec"] is not None] for rows in all_rows]
    merged = merge_rows(new_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    rtvecs = extract_rtvecs(merged)
    if verbose:
        print(get_connections(rtvecs, cgroup.get_names()))
    rvecs, tvecs = _get_initial_extrinsics(rtvecs, cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    return cgroup.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)


def calibrate(
    config: CalibrationConfig,
    board: CalibrationObject | None = None,
    grid_fps: float = CALIBRATION_GRID_FPS,
    skip: int = 1,
) -> CameraGroup:
    """Calibrate the three-camera rig from role-named calibration board videos.

    Detections are matched across cameras by estimated recording time
    (frame index divided by each video's own frame rate), not by raw frame
    index, since the rig's cameras do not run at a matched frame rate. See
    :mod:`alligaitor.timing`.

    Args:
        config: Calibration video paths, output path, which physical board
            (``config.board_preset``) this recording used, which extrinsics
            algorithm to run (``config.calibration_standard``, derived from
            ``board_preset``), and the extrinsics point-count floor
            (``config.min_corners_extrinsic``) used only for the
            ``"apriltag"`` standard (see :class:`AprilGridBoard`).
        board: Calibration board definition (a ChArUco board or
            :class:`AprilGridBoard`). Defaults to
            ``build_board_for_preset(config.board_preset, min_corners_extrinsic=config.min_corners_extrinsic)``;
            pass this explicitly to override the config's preset.
        grid_fps: Shared time-bucket rate used to match detections across
            cameras. Defaults to :data:`CALIBRATION_GRID_FPS`.
        skip: Process every ``skip``-th frame of each video.

    Returns:
        The calibrated :class:`~aniposelib.cameras.CameraGroup`, already
        saved to ``config.output_path``.
    """
    if board is None:
        board = build_board_for_preset(
            config.board_preset, min_corners_extrinsic=config.min_corners_extrinsic
        )

    cgroup = CameraGroup.from_names(list(CAMERA_ROLES))

    all_rows = []
    for role, camera in zip(CAMERA_ROLES, cgroup.cameras):
        video_path = config.videos[role]
        params = get_video_params(str(video_path))
        camera.set_size((params["width"], params["height"]))
        fps = video_fps(video_path)
        all_rows.append(
            _detect_video_by_time(
                board, video_path, fps, grid_fps, skip=skip, progress_desc=f"{role} ({video_path.name})"
            )
        )

    for role, rows in zip(CAMERA_ROLES, all_rows):
        objp, _ = board.get_all_calibration_points(rows, min_points=MIN_CORNERS_INTRINSIC)
        if not objp:
            raise ValueError(
                f"No usable board detections for camera '{role}': no frame reached "
                f"the {MIN_CORNERS_INTRINSIC}-point minimum needed for intrinsic "
                "calibration. Check board visibility and framing for this camera."
            )

    if config.calibration_standard == "apriltag":
        error = _calibrate_rows(cgroup, all_rows, board)
    else:
        error = cgroup.calibrate_rows(all_rows, board, min_corners_intrinsic=MIN_CORNERS_INTRINSIC)
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


def world_up_direction(cgroup: CameraGroup, side_roles: tuple = ("left", "right")) -> np.ndarray:
    """Estimate true world "up" in a calibrated rig's 3D reference frame.

    The reconstruction's coordinate frame tracks wherever the calibration
    board was held, not gravity, so no fixed axis can be assumed to point
    "up". The side cameras are mounted level, so each one's own
    image-vertical axis (``-y`` in camera space, or ``R.T @ [0, -1, 0]``
    in world space) reliably points along true vertical; this averages
    that estimate across available side cameras, used by
    :mod:`alligaitor.gait` to tell a planted paw from a raised one.

    Args:
        cgroup: Calibrated camera group.
        side_roles: Which camera roles to treat as "side" cameras for
            this estimate (the bottom camera is excluded by default: it
            looks up through the platform at a steep, only roughly-known
            angle, rather than level).

    Returns:
        A unit vector in the calibration's 3D reference frame.
    """
    names = cgroup.get_names()
    ups = []
    for role in side_roles:
        if role not in names:
            continue
        cam = cgroup.cameras[names.index(role)]
        rotation, _ = cv2.Rodrigues(cam.get_rotation())
        cam_up_in_world = rotation.T @ np.array([0.0, -1.0, 0.0])
        ups.append(cam_up_in_world / np.linalg.norm(cam_up_in_world))

    if not ups:
        raise ValueError(
            f"No side camera ({side_roles}) found in calibration (has {names}); "
            "cannot estimate world up direction."
        )

    if len(ups) > 1:
        agreement = float(np.clip(np.dot(ups[0], ups[1]), -1.0, 1.0))
        disagreement_deg = np.degrees(np.arccos(agreement))
        if disagreement_deg > 25:
            warnings.warn(
                f"Side cameras disagree on world 'up' by {disagreement_deg:.1f} degrees; "
                "check for a rolled or mismounted camera."
            )

    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)
