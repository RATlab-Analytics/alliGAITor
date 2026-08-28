# RATlab alliGAITor: an open-source rodent gait analysis pipeline for research
# Copyright (C) 2026 Mitchell Carson
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""alliGAITor: multi-camera 3D gait reconstruction pipeline for rats.

Combines per-camera 2D pose estimation (SLEAP-NN) with multi-camera
calibration and triangulation (aniposelib) to reconstruct 3D keypoint
trajectories from three fixed cameras (left side, right side, bottom-up).
"""

__version__ = "1.0.2"
