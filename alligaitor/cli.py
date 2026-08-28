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

"""Command-line entry point for the alliGAITor calibration and triangulation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from alligaitor import calibration, pipeline
from alligaitor.config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("config", type=Path, help="Path to a pipeline config YAML file.")
    common.add_argument(
        "--device",
        default="auto",
        help="Torch device for SLEAP-NN inference: auto, cpu, cuda, or mps.",
    )
    common.add_argument(
        "--tracking",
        action="store_true",
        help="Run SLEAP-NN's tracker during inference, in addition to per-frame prediction.",
    )

    parser = argparse.ArgumentParser(
        prog="alligaitor",
        description="alliGAITor calibration and 3D triangulation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "calibrate",
        parents=[common],
        help="Run camera calibration from the configured ChArUco videos and save it.",
    )
    subparsers.add_parser(
        "run",
        parents=[common],
        help=(
            "Run calibration (if not already saved), triangulate every configured session, "
            "and write the group's gait-metrics workbook."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig.from_yaml(args.config)

    if args.command == "calibrate":
        calibration.calibrate(config.calibration)
    elif args.command == "run":
        output_xlsx = pipeline.run_group(config, device=args.device, tracking=args.tracking)
        print(output_xlsx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
