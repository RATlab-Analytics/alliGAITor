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

from pathlib import Path
from typing import List, Tuple, Union

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

# aniposelib hardcodes 8 as the minimum ChArUco corners a frame needs to
# be considered for linking two cameras' poses in the initial calibration
# graph (CameraGroup.calibrate_rows), AND separately hardcodes 7 as the
# minimum inside CharucoBoard.estimate_pose_points -- below that, pose
# estimation for that frame silently returns (None, None) regardless of
# how many corners were actually detected. Neither is exposed as a
# parameter. calibrate() below reimplements both call sites with this made
# configurable (see _calibrate_rows and _estimate_pose_points), since the
# bottom camera's slit view can force a tradeoff between corner count and
# staying in a side camera's field of view: on some recordings, no frame
# reaches 7-8 corners on the bottom camera while also being visible to a
# side camera at all. 4 is the practical floor (cv2.solvePnP needs at
# least 4 point correspondences); only lower a given recording's
# CalibrationConfig.min_corners_extrinsic toward that floor after
# confirming the recording has no frames clearing 8, since lower corner
# counts give a noisier initial pose estimate for that frame (bundle
# adjustment still refines it afterward, but a bad initial extrinsics
# guess can still leave it in a bad local optimum).
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

# A flat AprilTag marker-grid board -- the replacement for the "strip"
# ChArUco board above at the bottom camera's fixed 45-degree viewing angle
# (the rig's cameras are physically perpendicular, so this angle is not
# adjustable; a rigid or folded two-face target was ruled out, so this
# stays a single flat plane like every other board here).
#
# ChArUco calibration needs a clean geometric corner *interpolation* step
# (fitting a checkerboard corner from the surrounding squares) on top of
# marker decoding, and that interpolation is what was failing on real
# 45-degree/blurred/compressed bottom-camera footage even when markers
# were individually legible (see calib_big_01/calib_big_red_01
# measurements, 2026-08-17/18). A bare marker grid has no such step: each
# AprilTag marker's 4 corners come directly from decoding that one marker,
# so there is nothing here that depends on multiple markers being crisp
# simultaneously. AprilTag's own decoder (DICT_APRILTAG_36H11 has the
# largest inter-code Hamming distance of the AprilTag families OpenCV
# ships) is also substantially more blur/compression-tolerant than
# ChArUco's checkerboard-corner search: a synthetic test replicating this
# rig's approximate bottom-camera conditions -- foreshortened ~cos(45deg),
# downscaled to a ~15mm-wide on-sensor footprint, Gaussian-blurred, and
# JPEG-recompressed at quality 60 -- still decoded all 6 markers cleanly
# (2026-08-18).
#
# 2 columns x 3 rows of 46mm markers is the largest size that fits the
# same <=140mm x 175mm footprint the printed "strip" ChArUco board above
# used -- that board's physical footprint is a hard constraint (it's sized
# to the paddle the board mounts on to go through the slit, not just the
# 3D-printer-plate check the strip board's own comment describes), so this
# board must match or beat it, not just clear the plate/slit checks
# independently. 8mm marker separation and a 10mm outer margin are both
# required white "quiet zone" -- AprilTag detection needs white space
# *around* a marker to find its border in the first place; a synthetic
# test with zero outer margin (markers flush to the printed edge) failed
# to detect 2 of 6 markers outright, while every margin tested from 4mm up
# detected all 6 (2026-08-18). 10mm was chosen over the tested minimum
# (4mm) for extra tolerance against real-world lighting/glare at the
# board's physical edge, which a synthetic test can't reproduce. With
# sep=8mm and margin=10mm fixed, 46mm is the largest whole-mm marker size
# that keeps the long axis (3 markers) at or under 175mm; solving the
# short axis (2 markers) for the same size leaves it well under 140mm.
#
# Footprint: (2*46 + 1*8 + 2*10) = 120mm wide x (3*46 + 2*8 + 2*10) = 174mm
# long -- at or under the 140mm x 175mm target on both axes, and still
# ~79% larger than the strip board's 25.7mm squares (to compensate for the
# same 45-degree foreshortening: 46mm * cos(45deg) ~= 32.5mm effective,
# still bigger than the strip board's un-foreshortened 25.7mm). Not yet
# validated against real footage (no AprilTag board has been printed or
# recorded against yet); adjust APRILTAG_MARKER_LENGTH_MM/
# APRILTAG_MARKER_SEPARATION_MM and re-check real detections before
# committing to a final print.
#
# ChArUco calibration needs a clean geometric corner *interpolation* step
# (fitting a checkerboard corner from the surrounding squares) on top of
# marker decoding, and that interpolation is what was failing on real
# 45-degree/blurred/compressed bottom-camera footage even when markers
# were individually legible (see calib_big_01/calib_big_red_01
# measurements, 2026-08-17/18). A bare marker grid has no such step: each
# AprilTag marker's 4 corners come directly from decoding that one marker,
# so there is nothing here that depends on multiple markers being crisp
# simultaneously. AprilTag's own decoder (DICT_APRILTAG_36H11 has the
# largest inter-code Hamming distance of the AprilTag families OpenCV
# ships) is also substantially more blur/compression-tolerant than
# ChArUco's checkerboard-corner search: a synthetic test replicating this
# rig's approximate bottom-camera conditions -- foreshortened ~cos(45deg),
# downscaled to a ~15mm-wide on-sensor footprint, Gaussian-blurred, and
# JPEG-recompressed at quality 60 -- still decoded all 6 markers cleanly
# (2026-08-18); re-verified at 46mm after this resize with the same
# result.
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
APRILTAG_MARKERS_X = 2
APRILTAG_MARKERS_Y = 3
APRILTAG_MARKER_LENGTH_MM = 46.0
APRILTAG_MARKER_SEPARATION_MM = 8.0
APRILTAG_MARGIN_MM = 10.0

# Named board geometries, keyed by the ``board_preset`` a
# CalibrationConfig can specify -- different recordings may have used
# different physical printed boards. Each ChArUco value is
# (squares_x, squares_y, square_length_mm, marker_length_mm); the
# ``"apriltag"`` preset is handled separately in build_board_for_preset()
# since AprilGridBoard doesn't fit that same tuple shape (see its
# docstring).
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


def build_apriltag_board(
    markers_x: int = APRILTAG_MARKERS_X,
    markers_y: int = APRILTAG_MARKERS_Y,
    marker_length_mm: float = APRILTAG_MARKER_LENGTH_MM,
    marker_separation_mm: float = APRILTAG_MARKER_SEPARATION_MM,
    aruco_dict: int = APRILTAG_DICTIONARY,
) -> "AprilGridBoard":
    """Construct the flat AprilTag marker-grid board (see
    :data:`APRILTAG_MARKERS_X` and neighbors for the rationale behind the
    default geometry)."""
    return AprilGridBoard(
        markers_x=markers_x,
        markers_y=markers_y,
        marker_length=marker_length_mm,
        marker_separation=marker_separation_mm,
        aruco_dict=aruco_dict,
    )


def build_board_for_preset(
    preset: str, relax_for_small_markers: bool | None = None
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
    """
    if preset == "apriltag":
        return build_apriltag_board()
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
    interface the same way :class:`~aniposelib.boards.CharucoBoard` does,
    so it plugs into aniposelib's existing ``merge_rows``/``extract_points``
    machinery, :meth:`get_all_calibration_points`, etc. unmodified -- this
    is an original implementation of that public abstract interface, not a
    port of any CharucoBoard/GridBoard logic, so it isn't covered by the
    aniposelib attribution in ``THIRD_PARTY_NOTICES.md``.

    Every possible point on the board needs a fixed, stable index for
    aniposelib's correspondence machinery (``get_empty_detection()`` /
    ``get_object_points()`` must be the same length and order on every
    call). ChArUco gets this for free -- one interpolated corner per index.
    A bare marker grid doesn't have interpolated corners, so this class
    treats each of the board's ``n_markers`` markers as 4 consecutive point
    slots (its 4 corners, in ``cv2.aruco.GridBoard``'s own per-marker
    corner order): marker id ``i`` occupies slots ``[4*i, 4*i+4)``.
    """

    def __init__(
        self,
        markers_x: int,
        markers_y: int,
        marker_length: float,
        marker_separation: float,
        aruco_dict: int = APRILTAG_DICTIONARY,
        manually_verify: bool = False,
    ):
        self.markers_x = markers_x
        self.markers_y = markers_y
        self.marker_length = marker_length
        self.marker_separation = marker_separation
        self.manually_verify = manually_verify

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
        """Generic base-class-satisfying implementation (not used by this
        module's own calibration path, which calls :func:`_match_points` /
        :func:`_estimate_pose_points` directly so the minimum-point floor
        stays configurable end to end -- see :data:`MIN_CORNERS_EXTRINSIC`).
        """
        if corners is None or ids is None or len(corners) == 0:
            return None, None
        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or len(obj_points) < 4:
            return None, None
        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        if ret:
            return rvec, tvec
        return None, None


def _detect_video_by_time(
    board: CalibrationObject, video_path: Path, fps: float, grid_fps: float, skip: int = 1
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


def _match_points(board: CalibrationObject, corners, ids) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Match detected corners/markers to a board's fixed object-point
    layout, via OpenCV's own ``Board.matchImagePoints()`` -- exposed
    identically on the ``cv2.aruco.CharucoBoard`` and ``cv2.aruco.GridBoard``
    instances underlying, respectively, aniposelib's ``CharucoBoard`` and
    this module's :class:`AprilGridBoard` (both store it as ``board.board``).
    This is what lets both board types share one pose-estimation/
    point-counting code path below, rather than needing board-type-specific
    correspondence lookup logic (as aniposelib's own
    ``CharucoBoard.estimate_pose_points`` has, via ``getChessboardCorners()``
    -- verified numerically equivalent to that manual lookup for ChArUco,
    2026-08-18).

    Returns ``(obj_points, img_points)`` as ``(N, 3)``/``(N, 2)`` float32
    arrays, or ``(None, None)`` if there is nothing to match.
    """
    if corners is None or ids is None or len(corners) == 0 or len(ids) == 0:
        return None, None
    obj_points, img_points = board.board.matchImagePoints(corners, ids)
    if obj_points is None or len(obj_points) == 0:
        return None, None
    return obj_points.reshape(-1, 3).astype(np.float32), img_points.reshape(-1, 2).astype(np.float32)


def _n_matched_points(board: CalibrationObject, row: dict) -> int:
    """Number of points :func:`_match_points` can actually resolve for one
    detection row -- the board-agnostic point count. Needed because raw
    ``row["ids"].size`` means different things per board type: for ChArUco,
    one id is one interpolated corner (one point); for :class:`AprilGridBoard`,
    one id is one decoded marker (4 points). Using the raw id count as a
    stand-in for point count -- as aniposelib itself does -- silently
    undercounts AprilGridBoard detections by ~4x, making
    ``min_corners_intrinsic``/``min_corners_extrinsic`` mean a different
    number of points depending on which board a recording used.
    """
    obj_points, _ = _match_points(board, row["corners"], row["ids"])
    return 0 if obj_points is None else len(obj_points)


def _estimate_pose_points(camera, corners, ids, board: CalibrationObject, min_points: int):
    """Board-agnostic equivalent of ``CharucoBoard.estimate_pose_points``,
    via :func:`_match_points`, with the minimum *point* count (not corner-id
    count -- see :func:`_n_matched_points`) needed to attempt a pose solve
    made configurable (hardcoded to 7 corners upstream; see
    :data:`MIN_CORNERS_EXTRINSIC`).

    Loosely adapted from ``aniposelib.boards.CharucoBoard.estimate_pose_points``
    (BSD 2-Clause License, Copyright (c) 2019-2023 Lili Karashchuk) --
    the corner-count floor and ``solvePnP`` call are the same shape, but the
    correspondence lookup itself is now ``_match_points()`` rather than a
    port of that method's manual ``getChessboardCorners()`` loop; see
    ``THIRD_PARTY_NOTICES.md`` for the full license text.
    """
    obj_points, img_points = _match_points(board, corners, ids)
    if obj_points is None or len(obj_points) < min_points:
        return None, None

    K = camera.get_camera_matrix()
    D = camera.get_distortions()
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
    if ret:
        return rvec, tvec
    return None, None


def _estimate_pose_rows(camera, rows: List[dict], board: CalibrationObject, min_points: int) -> List[dict]:
    """Equivalent to :meth:`CharucoBoard.estimate_pose_rows`, calling
    :func:`_estimate_pose_points` instead of the board's own method so the
    minimum-point floor is configurable end to end. Also stashes each row's
    board-agnostic matched-point count (see :func:`_n_matched_points`) as
    ``row["n_points"]``, since it needs the same ``_match_points()`` result
    used here -- callers that need to filter rows by point count (e.g.
    :func:`_calibrate_rows`) should use that instead of ``row["ids"].size``.
    """
    for row in rows:
        rvec, tvec = _estimate_pose_points(camera, row["corners"], row["ids"], board, min_points)
        row["rvec"] = rvec
        row["tvec"] = tvec
        row["n_points"] = _n_matched_points(board, row)
    return rows


def _mean_transform_robust(M_list, approx=None, error: float = 0.5):
    """Equivalent to ``aniposelib.utils.mean_transform_robust``, except it
    falls back to the un-filtered mean instead of crashing when every
    candidate transform in ``M_list`` falls outside ``error`` of
    ``approx``. Observed on this rig's bottom-camera/side-camera pairs:
    their few, oblique-angle pose estimates can scatter widely enough
    that none clear aniposelib's fixed 0.5 threshold, even though the
    pair is otherwise usable. aniposelib's own ``mean_transform_robust``
    calls ``mean_transform(M_list_robust)`` unconditionally;
    ``mean_transform([])`` crashes deep inside ``cv2.Rodrigues`` with a
    confusing, unrelated-looking shape error instead of an empty-input
    error, which is what actually surfaced this.

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
        all_rows[i] = _estimate_pose_rows(cam, row, board, min_corners_extrinsic)

    # Point count (see _n_matched_points), not raw id count -- for
    # AprilGridBoard, one id is one marker (4 points), so id count alone
    # would silently undercount by ~4x here.
    new_rows = [[r for r in rows if r["n_points"] >= min_corners_extrinsic] for rows in all_rows]
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
            (``config.board_preset``) this recording used, and the
            extrinsics corner-count floor (``config.min_corners_extrinsic``)
            for this recording.
        board: Calibration board definition (a ChArUco board or
            :class:`AprilGridBoard`). Defaults to
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
        # Point count (see _n_matched_points), not raw id count -- see
        # _calibrate_rows's new_rows filter for why.
        if not any(_n_matched_points(board, row) >= MIN_CORNERS_INTRINSIC for row in rows):
            raise ValueError(
                f"No usable board detections for camera '{role}': no frame reached "
                f"the {MIN_CORNERS_INTRINSIC}-point minimum needed for intrinsic "
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
