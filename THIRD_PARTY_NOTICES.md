# Third-Party Notices

alliGAITor is licensed under the GNU General Public License v3.0 (see
`LICENSE`). It depends on and, in the specific files noted below, adapts
code from the following open-source projects:

## aniposelib

Adapted in `alligaitor/calibration.py`, used only for the `"apriltag"`
calibration standard (ChArUco calibration calls aniposelib's own
`CameraGroup.calibrate_rows()` unmodified):

- `_calibrate_rows()` is adapted from
  `aniposelib.cameras.CameraGroup.calibrate_rows()`: the row filter that
  decides whether a frame counts toward linking two cameras' poses is
  replaced (upstream's `row['ids'].size >= 8` undercounts `AprilGridBoard`'s
  per-marker ids by ~4x relative to points). Every other call it makes
  (`estimate_pose_rows`, `get_all_calibration_points`, `merge_rows`,
  `extract_points`, `extract_rtvecs`, `bundle_adjust_iter`) is unmodified
  aniposelib.
- `_mean_transform_robust()` / `_get_transform()` /
  `_get_initial_extrinsics()` port the corresponding functions in
  `aniposelib.utils`, with one behavior change: if every candidate
  transform for a camera pair falls outside the (fixed, upstream)
  robust-averaging error threshold, the port falls back to the
  unfiltered mean instead of crashing (upstream calls
  `mean_transform([])` on the empty result, which fails deep inside
  `cv2.Rodrigues`).

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


