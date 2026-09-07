# Waymo VQA pipeline v1

This directory adapts the NuRisk-style video-risk pipeline to Waymo Open Dataset
v2 modular parquet files. It keeps the same intermediate CSV/JSON schemas used
by the nuScenes pipeline after Stage 1, so Stage 2-4, Stage 5b, Stage 6, Stage 7,
and Stage 8 can mostly reuse the existing logic.

Required Waymo components under `<dataroot>/<split>/`:

```text
camera_image/
camera_box/
camera_calibration/
camera_to_lidar_box_association/
lidar_box/
vehicle_pose/
```

Raw LiDAR point clouds are not required. Do not download these unless a future
task needs point-level data:

```text
lidar/
lidar_pose/
lidar_calibration/
lidar_camera_projection/
lidar_segmentation/
lidar_hkp/
```

Default dataroot:

```text
/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/waymo_test
```

## Workflow

Stage 1 reads Waymo `vehicle_pose` and `lidar_box` parquet files, samples them at
2Hz by default, transforms obstacle boxes to world coordinates, and writes:

```text
<dataroot>/nurisk_style/<segment>/ego_trajectory.csv
<dataroot>/nurisk_style/<segment>/dynamic_obstacles.csv
```

Stage 2-4 then compute relative metrics, close-agent filtering, and enhanced
risk scores from those CSVs.

Stage 5a reads Waymo `camera_image`, `camera_box`, `camera_calibration`,
`camera_to_lidar_box_association`, and `lidar_box` parquet files, samples camera
JPEGs at the same 2Hz cadence, encodes 5-frame multi-camera mp4 clips, and writes:

```text
<dataroot>/video_clip_dataset/videos/<segment>/<camera>/*.mp4
<dataroot>/video_clip_dataset/metadata/<segment>_clips.json
<dataroot>/video_clip_dataset/metadata/clips.jsonl
```

Waymo cameras are named:

```text
WAYMO_FRONT
WAYMO_FRONT_LEFT
WAYMO_FRONT_RIGHT
WAYMO_SIDE_LEFT
WAYMO_SIDE_RIGHT
```

## Run

Run Stage 1-4:

```bash
python scripts/pipelin_v1_waymo/run_until_stage4.py \
  --dataroot data/sets/waymo_test \
  --split validation \
  --overwrite
```

Run Stage 5a + 5b:

```bash
python scripts/pipelin_v1_waymo/run_stage5.py \
  --dataroot data/sets/waymo_test \
  --split validation \
  --overwrite-videos
```

Run Stage 6:

```bash
python scripts/pipelin_v1_waymo/stage6_build_future_risk_groundtruth.py \
  --dataroot data/sets/waymo_test
```

For a small smoke test:

```bash
python scripts/pipelin_v1_waymo/run_until_stage4.py \
  --dataroot data/sets/waymo_test \
  --split validation \
  --max-scenes 1 \
  --max-keyframes 12 \
  --overwrite

python scripts/pipelin_v1_waymo/run_stage5.py \
  --dataroot data/sets/waymo_test \
  --split validation \
  --max-scenes 1 \
  --max-keyframes 12 \
  --overwrite-videos
```

## Notes

- This adapter assumes Waymo box centers and velocities are in the vehicle frame
  for each frame and transforms them with `vehicle_pose.world_from_vehicle`.
- Risk scoring remains the same offline NuRisk-style approximation as the
  nuScenes pipeline. It is not Waymo's official metric.
- The directory name intentionally matches the requested path:
  `scripts/pipelin_v1_waymo`.
