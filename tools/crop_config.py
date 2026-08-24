"""
Crop defaults for alliGAITor, mirroring RATlab-NOR's config.py pattern
(CROP_TARGET_WIDTH/HEIGHT) -- editable per-crop later in the GUI dialog,
these are just the pre-filled defaults.

The tunnel occupies a horizontal strip roughly 1/10 the source frame's
height, hence the wide/short target size below (vs. NOR's roughly-square
294x292 crop for its arena footage).
"""

from __future__ import annotations

from pathlib import Path

CROP_TARGET_WIDTH = 1280
CROP_TARGET_HEIGHT = 170

# Fallback side-crop target size, used only when side_crop_size_for_model()
# can't read a size from the selected model's own training data (see its
# docstring for why that's the source of truth instead of the model's
# config). Matches CROP_TARGET_WIDTH/HEIGHT above since that's also what
# every side model shipped with this repo turns out to have been trained
# on -- not because side and bottom footage share a target size in
# general, just current fact for this rig.
_FALLBACK_SIDE_CROP_WIDTH = 1280
_FALLBACK_SIDE_CROP_HEIGHT = 170


def side_crop_size_for_model(model_dir) -> tuple[int, int]:
    """(width, height) the side model at ``model_dir`` was actually
    trained on, read from its ``labels_gt.train.0.slp``.

    These models are trained at ``scale: 1.0`` with ``crop_size: null``
    (see e.g. ``models/*/initial_config.yaml``) -- i.e. there's no fixed
    input size recorded in the model's own config, since the
    architecture is resolution-agnostic. The training labels file's
    cached video frame shape is the only place the size the model
    actually saw during training is recorded, so that's what this reads.

    Falls back to a hardcoded default if ``model_dir`` doesn't have that
    file, or ``sleap_io`` can't load it (e.g. a model directory copied
    without its labels).
    """
    labels_path = Path(model_dir) / "labels_gt.train.0.slp"
    if labels_path.exists():
        try:
            import sleap_io as sio

            labels = sio.load_slp(str(labels_path))
            _, height, width, _ = labels.videos[0].shape
            return int(width), int(height)
        except Exception:
            pass
    return _FALLBACK_SIDE_CROP_WIDTH, _FALLBACK_SIDE_CROP_HEIGHT
