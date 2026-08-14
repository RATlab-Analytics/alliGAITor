"""Generate an alliGAITor pipeline config YAML.

Writes a config file matching the schema in ``alligaitor.config.PipelineConfig``
(see ``configs/session_example.yaml`` for a hand-written reference). Paths you
pass on the command line are written as given -- use paths relative to the
output config's directory (matching the convention used elsewhere in this
repo) or absolute paths.

Examples:

    # Calibration-only config, no gait sessions yet.
    python3 tools/calibration/make_config.py configs/calibration_260813.yaml \\
        --calibration-left calibration_video_260813-132431/coded/calibration_cam0_coded.mp4 \\
        --calibration-right calibration_video_260813-132431/coded/calibration_cam1_coded.mp4 \\
        --calibration-bottom calibration_video_260813-132431/coded/calibration_cam2_coded.mp4

    # Same, plus one gait session.
    python3 tools/calibration/make_config.py configs/my_config.yaml \\
        --calibration-left calib/left.mp4 --calibration-right calib/right.mp4 \\
        --calibration-bottom calib/bottom.mp4 \\
        --session 359a-BL side-training-data/359a-BL_cam0_coded.mp4 \\
                           side-training-data/359a-BL_cam2_coded.mp4 \\
                           bottom_training_data_cropped/359a-BL_cam1_coded.mp4

Re-run against an existing output path to overwrite it; this script does
not merge into an existing config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_SIDE_MODEL = "models/alliGAITor_side_ConvNeXt_slim_v1.0.0.n=315"
DEFAULT_BOTTOM_MODEL = "models/alliGAITor_bottom_slim_v0.4.1-beta"


def build_config(args: argparse.Namespace) -> dict:
    config = {
        "models": {
            "side_model_dir": args.side_model,
            "bottom_model_dir": args.bottom_model,
        },
        "calibration": {
            "videos": {
                "left": args.calibration_left,
                "right": args.calibration_right,
                "bottom": args.calibration_bottom,
            },
            "output_path": args.calibration_output,
        },
        "sessions": [
            {
                "name": name,
                "videos": {"left": left, "right": right, "bottom": bottom},
                "output_dir": f"predictions_3d/{name}",
            }
            for name, left, right, bottom in args.session
        ],
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Path to write the generated config YAML to.")
    parser.add_argument("--side-model", default=DEFAULT_SIDE_MODEL, help="Side-camera model directory.")
    parser.add_argument("--bottom-model", default=DEFAULT_BOTTOM_MODEL, help="Bottom-camera model directory.")
    parser.add_argument("--calibration-left", required=True, help="Left-camera ChArUco calibration video.")
    parser.add_argument("--calibration-right", required=True, help="Right-camera ChArUco calibration video.")
    parser.add_argument("--calibration-bottom", required=True, help="Bottom-camera ChArUco calibration video.")
    parser.add_argument(
        "--calibration-output",
        default="calibration/calibration.toml",
        help="Where to save/load the resulting camera calibration.",
    )
    parser.add_argument(
        "--session",
        nargs=4,
        action="append",
        default=[],
        metavar=("NAME", "LEFT", "RIGHT", "BOTTOM"),
        help="Add a gait-recording session: name, then its left/right/bottom video paths. "
        "Repeat --session for multiple sessions.",
    )
    args = parser.parse_args()

    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
