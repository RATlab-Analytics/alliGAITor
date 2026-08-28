"""alliGAITor: multi-camera 3D gait reconstruction pipeline for rats.

Combines per-camera 2D pose estimation (SLEAP-NN) with multi-camera
calibration and triangulation (aniposelib) to reconstruct 3D keypoint
trajectories from three fixed cameras (left side, right side, bottom-up).
"""

__version__ = "1.0.0"
