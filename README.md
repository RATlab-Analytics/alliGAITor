# alliGAITor

A 3D gait reconstruction pipeline for rats, filmed with three fixed cameras (left side, right side, bottom-up through a tunnel).

## Pipeline

1. **2D pose estimation** — a trained SLEAP-NN model predicts 2D keypoints per camera view. One model handles both side cameras (left and right); a separate model handles the bottom camera.
2. **Camera calibration** — [aniposelib](https://github.com/lambdaloop/aniposelib) calibrates the three-camera rig from synchronized ChArUco board recordings.
3. **Triangulation** — per-camera 2D keypoints are combined with the camera calibration to reconstruct 3D keypoint trajectories.

The `alligaitor` package (`alligaitor/`) implements this pipeline; usage is below.

## Setup

```
pip install -r requirements.txt
```

`sleap-nn` is intentionally not pinned in `requirements.txt` — install it separately with the PyTorch/accelerator build appropriate for this machine (see https://nn.sleap.ai). Inference also requires the `ffmpeg` binary on `PATH` (e.g. `brew install ffmpeg`); see the note in `requirements.txt`.

Both models were trained on achromatic footage, so `alligaitor.inference.run_inference()` always re-encodes video to true grayscale content before prediction (`alligaitor/preprocessing.py`), independent of the channel count (`ensure_rgb`/`ensure_grayscale`) each model itself expects — see that module's docstring for why this can't be left to `sleap-nn`'s own flags.

## Configuration

A pipeline run is defined by a single YAML config: model directories, one ChArUco calibration video per camera role, and one or more sessions, each with a video per camera role. See `configs/session_example.yaml` for the schema.

Camera role (`left` / `right` / `bottom`) is assigned per session rather than by a fixed camera index, since which physical camera lands on a given recording's `cam0`/`cam1`/`cam2` is not consistent across sessions.

## Running

```
python -m alligaitor.cli calibrate configs/my_config.yaml
python -m alligaitor.cli run configs/my_config.yaml --device mps
```

`calibrate` runs camera calibration and saves it; `run` calibrates if needed and triangulates every configured session, writing one `<session_name>.pose_3d.csv` per session (columns: `frame`, `node`, `x`, `y`, `z`, `reprojection_error_px`).

## Status

Camera calibration footage has not yet been recorded. The calibration and triangulation code is complete and unit-verified against synthetic camera geometry, but has not been run against the real rig — `calibrate` will fail until ChArUco recordings exist for all three camera roles.

## License

Copyright (C) 2026 Mitchell Carson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
