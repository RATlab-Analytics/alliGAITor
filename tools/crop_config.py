"""
Crop defaults for alliGAITor, mirroring RATlab-NOR's config.py pattern
(CROP_TARGET_WIDTH/HEIGHT) -- editable per-crop later in the GUI dialog,
these are just the pre-filled defaults.

The tunnel occupies a horizontal strip roughly 1/10 the source frame's
height, hence the wide/short target size below (vs. NOR's roughly-square
294x292 crop for its arena footage).
"""

CROP_TARGET_WIDTH = 1280
CROP_TARGET_HEIGHT = 170
