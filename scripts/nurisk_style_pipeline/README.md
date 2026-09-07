# NuRisk-style nuScenes pipeline

This folder contains a NuRisk-style preprocessing workflow for nuScenes data.

Stage 1 reads raw nuScenes metadata from `<dataroot>/<version>/*.json` and
uses only keyframe `sample` records.

nuScenes keyframes are treated as 2Hz data by default. Time labels in JSON
outputs use `frame_index * 0.5`, e.g. frame 0 -> `At 0.0 seconds`, frame 1 ->
`At 0.5 seconds`. Override this with `--keyframe-interval-seconds` only if the
input is not keyframe-only 2Hz data.

Default output:

```text
<dataroot>/nurisk_style/
```

Workflow:

```text
Stage 1: stage1_extract_nuscenes_trajectories.py
  raw nuScenes metadata
  -> <scene>/ego_trajectory.csv
  -> <scene>/dynamic_obstacles.csv

Stage 2: stage2_compute_relative_metrics.py
  <scene>/ego_trajectory.csv + dynamic_obstacles.csv
  -> <scene>/relative_metrics.csv
  -> <scene>/output.json

Stage 3a: stage3a_filter_close_obstacles.py
  <scene>/ego_trajectory.csv + dynamic_obstacles.csv
  -> <scene>/close_dynamic_obstacles.csv

Stage 3b: stage3b_extract_close_relative_metrics.py
  <scene>/close_dynamic_obstacles.csv + relative_metrics.csv
  -> <scene>/close_relative_metrics.csv

Stage 4: stage4_compute_risk_scores_enhanced.py
  <scene>/close_relative_metrics.csv
  -> <scene>/risk_scores_close_relative_metrics_enhanced.csv
  -> <scene>/risk_scores_output_enhanced.json
  -> risk_score_summary_enhanced.csv

Stage 5a: stage5a_build_video_clips.py
  raw nuScenes samples camera frames
  -> video_clip_dataset/videos/<scene>/<camera>/*.mp4
  -> video_clip_dataset/metadata/<scene>_clips.json

Stage 5b: stage5b_align_video_clip_groundtruth.py
  video_clip_dataset/metadata/<scene>_clips.json
  + <scene>/risk_scores_output_enhanced.json
  -> <scene>/video_clip_groundtruth.json
  -> video_clip_groundtruth.jsonl

Stage 6: stage6_build_future_risk_groundtruth.py
  <scene>/video_clip_groundtruth.json
  + <scene>/relative_metrics.csv
  + <scene>/ego_trajectory.csv
  -> <scene>/video_future_groundtruth.json
  -> video_future_groundtruth.jsonl

Stage 7: stage7_create_qwen_future_vqa_dataset.py
  <scene>/video_future_groundtruth.json
  -> <scene>/qwen_future_vqa_dataset.json
  -> qwen_future_vqa_dataset.json
  -> qwen_future_vqa_dataset.jsonl

Stage 8: stage8_prepare_dataset_splits.py
  qwen_future_vqa_dataset.json
  -> dataset_splits/train.json
  -> dataset_splits/validation.json
  -> dataset_splits/dataset_stats.json
```

Run through Stage 4 on mini:

```bash
python scripts/nurisk_style_pipeline/run_until_stage4.py \
  --dataroot data/sets/nuscenes_mini \
  --version v1.0-mini \
  --overwrite
```

Run Stage 5 only on mini after Stage 4:

```bash
python scripts/nurisk_style_pipeline/run_stage5.py \
  --dataroot data/sets/nuscenes_mini \
  --version v1.0-mini \
  --overwrite-videos
```

Run Stage 6 future-risk labels after Stage 5b:

```bash
python scripts/nurisk_style_pipeline/stage6_build_future_risk_groundtruth.py \
  --dataroot data/sets/nuscenes_mini
```

Run Stage 7 Qwen/LLaVA-style VQA conversations after Stage 6:

```bash
python scripts/nurisk_style_pipeline/stage7_create_qwen_future_vqa_dataset.py \
  --dataroot data/sets/nuscenes_mini
```

Run Stage 8 train/validation split after Stage 7:

```bash
python scripts/nurisk_style_pipeline/stage8_prepare_dataset_splits.py \
  --dataroot data/sets/nuscenes_mini
```
