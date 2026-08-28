# RATlab alliGAITor

An automated 3-D rodent gait analysis pipeline for camera-based setups.

## Architecture

The alliGAITor app uses a three-stage pipeline to extract gait metrics from
synchronized videos:

### SLEAP 2-D Inference

alliGAITor uses SLEAP-based image-recognition models to generate 2-D 
predictions from each camera angle. The two side cameras use one model,
while the bottom camera uses another model.

### 3-D Triangulation (Anipose)

The predictions from the SLEAP models are then fed into `aniposelib`, which
converts them into 3-D coordinates using known spatial relations between
cameras. In the event of a failure of triangulation due to loss of side
camera tracking, alliGAITor can fall back to 2-D predictions from the bottom
camera to fill in missing strides (this can be disabled in settings).

### Scoring

alliGAITor detects crossings within a video by observing when the animal is
moving vs. standing still. Gait metrics are computed from consecutive runs of
at least five strides within each crossing. Stances are detected using a speed
threshold, and other metrics are computed from the locations of the stances. 

alliGAITor calculates the following metrics for each paw, as well as crossing
time and average speed: 

- Stride length (distance between placements of the same paw)
- Step length (forward distance from last placement of contralateral paw)
- Ground contact time
  
## Data Capture

### Setup

alliGAITor is designed to be used with DIY camera-based gait test rigs.
In order to capture videos for analysis, your setup will need three 
cameras recording in sync (start and end at the same time, same or
very similar frame rate) around the platform across which the rodent
will walk: two cameras looking horizontally from the left and right 
sides of the platform, and one looking up from below. For best results,
the cameras should all be roughly equidistant from the center of the platform.

### Calibration

In order to use triangulation, `aniposelib` needs to know the relative positions
and orientations of the cameras. These are determined by calibration. 

This repository includes a printable AprilGrid calibration standard, which should
be affixed to a rigid flat object such as a clipboard or the included 3D-printable
calibration paddle. To calibrate your rig, record a session under uniform lighting
where you move the calibration board around the test area, making sure that it is
visible to every pair of cameras multiple times. For best results, hold the paddle 
stationary for several seconds at each position. **If any part of your rig moves, you
will need to repeat calibration.**

### Capturing Data

alliGAITor calculates metrics for each time a given animal crosses the platform
and averages them to produce a final output table for each animal in the group.
For best results, you should record multiple crossings per animal per session, 
which can be recorded in a single video or multiple. alliGAITor automatically
detects individual crossings within each set of videos and combines them into 
one spreadsheet.

## Using the App

### Installation

To install the app, download the correct package from the latest release,
located [here](https://github.com/RATlab-Analytics/alliGAITor/releases/latest).

In order to train your models and run inference, you will also need to install
SLEAP. For the model training GUI, follow the instructions 
[here](https://docs.sleap.ai/latest/installation/). In order to run inference
within alliGAITor, install sleap-nn by running the following command in your
terminal:

```uv tool install sleap-nn --torch-backend auto```

### Model Training

Once SLEAP is installed, follow the steps at <https://docs.sleap.ai/latest/tutorial/overview/>
to train your models. When creating your model skeletons, select "load from file"
and use [minimal_skeleton.json](minimal_skeleton.json) to ensure that the model
output can be recognized by alliGAITor.

When you have trained suitable models, point alliGAITor at the `/models` directory
that SLEAP has produced in the preferences panel.

### Setting Up a Job

alliGAITor processes videos in groups called jobs. Each job pulls its source videos
from an input folder and outputs its data Excel workbook and validation videos
(if generated) to its own output folder. Each job can be configured based on how 
the files from each camera are labelled, which calibration it should use, whether
crossings for each animal are contained in one session or spread across multiple,
and whether scoring should use the 2-D-only fallback if triangulation fails.

To add a job to the queue, click "Load Jobs..." and select your input and output
folders. Make sure the camera roles are correctly configured and that your calibration
videos or file are current. 

Once you save the job, you will be prompted to crop your videos. Each video should be cropped 
to the size that your models were trained on, which can be set in the top left of the crop 
dialog. You will also have the option to apply color grading to the bottom videos to increase
contrast if your lighting is very intense. Once you have configured a suitable crop, you can
click "Use This Position for All Remaining" to crop all videos in the job.

### Running a Batch

Once your job(s) are ready, click "Run All" to analyze all the videos in the queue. 
If you have a lot of data, you can leave it to run overnight or while you do other tasks.
alliGAITor will automatically run each job in the order in which they were created.

### Validation

Once your job has finished, double-click on it to open the validation report. This
will show you which videos were unable to produce usable data. If you have generated
validation videos, you can watch them by double-clicking on a session.

## AI Notice

This app was built with the help of Claude Sonnet 5 and Opus 5 (Anthropic PBC). Design
was directed by a human author and all functionality and output was tested for 
reliability and accuracy by human reviewers.

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

Portions of this codebase are adapted from permissively-licensed
third-party projects; see `THIRD_PARTY_NOTICES.md` for attribution and
the relevant license texts.
