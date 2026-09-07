# Bench2Drive VQA pipeline v1

This directory adapts the NuRisk-style video-risk VQA pipeline to
Bench2Drive-V0.0.4.

Required Bench2Drive layout:

```text
<dataroot>/raw_camera_anno/<scene>/anno/*.json.gz
<dataroot>/raw_camera_anno/<scene>/camera/rgb_front/*.jpg
<dataroot>/raw_camera_anno/<scene>/camera/rgb_front_left/*.jpg
<dataroot>/raw_camera_anno/<scene>/camera/rgb_front_right/*.jpg
<dataroot>/raw_camera_anno/<scene>/camera/rgb_back/*.jpg
<dataroot>/raw_camera_anno/<scene>/camera/rgb_back_left/*.jpg
<dataroot>/raw_camera_anno/<scene>/camera/rgb_back_right/*.jpg
```

Default dataroot:

```text
/mnt/d/Bench2Drive-V0.0.4
```

## Workflow

```text
Stage 1: stage1_extract_bench2drive_trajectories.py
  raw_camera_anno/<scene>/anno/*.json.gz
  -> <dataroot>/nurisk_style/<scene>/ego_trajectory.csv
  -> <dataroot>/nurisk_style/<scene>/dynamic_obstacles.csv

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

Stage 5a: stage5a_build_bench2drive_video_clips.py
  raw_camera_anno/<scene>/camera/rgb_*/*.jpg + anno/*.json.gz
  -> raw_video_clip_dataset_v1/videos/<scene>/<camera>/*.mp4
  -> raw_video_clip_dataset_v1/metadata/<scene>_clips.json

Stage 5b: stage5b_align_video_clip_groundtruth.py
  raw_video_clip_dataset_v1/metadata/<scene>_clips.json
  + <scene>/risk_scores_output_enhanced.json
  -> <scene>/video_clip_groundtruth.json

Stage 6: stage6_build_future_risk_groundtruth.py
  <scene>/video_clip_groundtruth.json
  + <scene>/relative_metrics.csv
  + <scene>/ego_trajectory.csv
  -> <scene>/video_future_groundtruth.json

Stage 7: stage7_create_qwen_future_vqa_dataset.py
  <scene>/video_future_groundtruth.json
  -> <scene>/qwen_future_vqa_dataset.json
  -> qwen_future_vqa_dataset.json/jsonl

Stage 8: stage8_prepare_dataset_splits.py
  qwen_future_vqa_dataset.json
  -> dataset_splits/train.json
  -> dataset_splits/validation.json
```

Bench2Drive actor ids are used only for internal joins and metadata. VQA
questions use clip-level reference names such as:

```text
the closest car ahead-left in the adjacent lane at the last observed frame
```

All close agents that appear in the Stage 4 reference-frame risk rows are named
with category, final-frame relative position, distance bucket, and rank when
needed.

## Run

Run through Stage 4:

```bash
python scripts/pipeline_v1_Bench2Drive/run_until_stage4.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --overwrite
```

Run Stage 5a + 5b:

```bash
python scripts/pipeline_v1_Bench2Drive/run_stage5.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --overwrite-videos
```

Run Stage 6:

```bash
python scripts/pipeline_v1_Bench2Drive/stage6_build_future_risk_groundtruth.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4
```

Run Stage 7:

```bash
python scripts/pipeline_v1_Bench2Drive/stage7_create_qwen_future_vqa_dataset.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4
```

Run Stage 8:

```bash
python scripts/pipeline_v1_Bench2Drive/stage8_prepare_dataset_splits.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4
```

## High-risk subset workflow

After Stage 1-4 have completed, select high-risk clip candidates before
encoding videos:

```bash
python scripts/pipeline_v1_Bench2Drive/stage4b_select_high_risk_clip_candidates.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --target-vqa-samples 10000 \
  --estimated-samples-per-clip 3.0
```

Generate only selected videos:

```bash
python scripts/pipeline_v1_Bench2Drive/run_stage5.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --clip-selection /mnt/d/Bench2Drive-V0.0.4/nurisk_style/high_risk_selection/selected_clips.jsonl \
  --overwrite-videos
```

Then run Stage 6 and Stage 7. If Stage 7 produces more than the desired final
count, sample a balanced fixed-size high-risk dataset:

```bash
python scripts/pipeline_v1_Bench2Drive/stage7b_sample_high_risk_vqa_dataset.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --target-samples 10000
```

Small smoke test:

```bash
python scripts/pipeline_v1_Bench2Drive/run_until_stage4.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --max-scenes 1 \
  --max-keyframes 12 \
  --overwrite

python scripts/pipeline_v1_Bench2Drive/run_stage5.py \
  --dataroot /mnt/d/Bench2Drive-V0.0.4 \
  --max-scenes 1 \
  --max-keyframes 12 \
  --overwrite-videos
```
