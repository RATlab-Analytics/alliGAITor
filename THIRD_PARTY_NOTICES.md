# Third-Party Notices

alliGAITor is licensed under the GNU General Public License v3.0 (see
`LICENSE`). It depends on and, in the specific files noted below, adapts
code from the following permissively-licensed projects. Their license
terms are reproduced here to satisfy each project's attribution
requirement; incorporating BSD/MIT-licensed code into a GPLv3 project does
not affect alliGAITor's own licensing.

## aniposelib

Adapted in `alligaitor/calibration.py`:

- `_calibrate_rows()` is a line-for-line port of
  `aniposelib.cameras.CameraGroup.calibrate_rows()`, with the minimum
  ChArUco corner count required to link two cameras' poses (hardcoded to
  8 upstream) exposed as a parameter.
- `_estimate_pose_points()` / `_estimate_pose_rows()` are loosely adapted
  from `aniposelib.boards.CharucoBoard.estimate_pose_points()` /
  `estimate_pose_rows()`: the corner-count floor (7, inside
  `estimate_pose_points` upstream) is exposed as a parameter, same as
  before, but the correspondence lookup itself was replaced with a
  board-agnostic `_match_points()` helper (built on OpenCV's own
  `Board.matchImagePoints()`, verified numerically equivalent to
  upstream's manual `getChessboardCorners()`-based lookup for ChArUco)
  so the same code path also serves `AprilGridBoard` (see below), whose
  markers aren't one-point-per-id the way ChArUco corners are.
- `AprilGridBoard` (the flat AprilTag marker-grid board added
  2026-08-18) is an original implementation of aniposelib's public
  `CalibrationObject` abstract interface — modeled on the same
  interface `CharucoBoard` implements, but not a port of any of
  `CharucoBoard`'s or `cv2.aruco.GridBoard`'s own method bodies — so it
  isn't separately listed here.
- `_mean_transform_robust()` / `_get_transform()` /
  `_get_initial_extrinsics()` port the corresponding functions in
  `aniposelib.utils`, with one behavior change: if every candidate
  transform for a camera pair falls outside the (fixed, upstream)
  robust-averaging error threshold, the port falls back to the
  unfiltered mean instead of crashing (upstream calls
  `mean_transform([])` on the empty result, which fails deep inside
  `cv2.Rodrigues` with a confusing, unrelated-looking shape error).

Everywhere else, alliGAITor calls aniposelib as an unmodified library
dependency.

> BSD 2-Clause License
>
> Copyright (c) 2019-2023, Lili Karashchuk
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
> IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
> TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
> PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
> TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
> PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
> LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
> NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## RATlab-NOR

Ported into `tools/video_crop.py`, `tools/crop_setup_dialog.py`,
`tools/crop_runner.py`, `tools/crop_worker_process.py`, and
`tools/frame_utils.py` (see each file's module docstring for what, if
anything, changed in the port). RATlab-NOR is the author's own prior
project (github.com/RATlab-Analytics/RATlab-NOR), released under the MIT
License below.

> MIT License
>
> Copyright (c) 2026 Mitchell Carson
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be included
> in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
> OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
