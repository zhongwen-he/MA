# Raw nuScenes VQA pipeline v1

This folder is an isolated copy/adaptation of `scripts/pipeline`. It keeps the
Stage 1-4 NuRisk-style trajectory and risk logic, but changes the video/VQA
identity policy:

- generated videos contain raw camera pixels only;
- nuScenes annotations are used offline for agent category/token indexing and
  projected-box visibility metadata;
- final target references are generated per clip from the last observed
  reference frame, using close agents that appear in the Stage 4 risk
  rows;
- synthetic display ids are not rendered, stored as training fields, or used in
  questions;
- color/YOLO are not primary naming inputs;
- vehicles, pedestrians, movable objects, and static objects can be named when
  they appear as close agents.

Default outputs:

```text
<dataroot>/nurisk_style/
<dataroot>/raw_video_clip_dataset_v1/
```

Reference-name policy:

```text
canonical_agent_name: car
clip_reference_name: the closest car ahead-left in the adjacent lane at the last observed frame
target_reference.primary_expression: the closest car ahead-left in the adjacent lane at the last observed frame
instance_token: raw nuScenes id, metadata/internal join only
risk_agent_id: Obstacle <token>, metadata/internal join only
```

Stage 5a validates categories against nuScenes `category_name` and indexes all
annotated instances. It does not generate final scene-level natural-language
names. Stage 5b builds the final clip-level target reference only for close
agents present in the reference frame risk rows.

Clip-level target references use this policy:

```text
category + final-frame relative position
category + final-frame relative position + distance bucket
category + final-frame relative position + distance rank
```

When multiple close agents share the same category and final-frame position,
the reference adds distance rank, such as `closest`, `2nd closest`, or `4th
closest`. Rank, reference-frame visibility, and projected box area are retained
as metadata for audit, but they do not filter targets. Agent ids remain
available only in metadata for traceability.

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
  raw nuScenes camera keyframes
  -> raw_video_clip_dataset_v1/videos/<scene>/<camera>/*.mp4
  -> raw_video_clip_dataset_v1/metadata/<scene>_clips.json
  Note: builds raw clips, projected-box visibility metadata, and vehicle
        category/token indexes for all agents; no color naming.

Stage 5b: stage5b_align_video_clip_groundtruth.py
  raw_video_clip_dataset_v1/metadata/<scene>_clips.json
  + <scene>/risk_scores_output_enhanced.json
  -> <scene>/video_clip_groundtruth.json
  -> video_clip_groundtruth.jsonl
  Note: creates reference-frame target references for close agents. Rank,
        visibility, and projected-box area are metadata only.

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

Run through Stage 4:

```bash
python scripts/pipeline_v1/run_until_stage4.py \
  --dataroot data/sets/nuscenes_full \
  --version v1.0-trainval \
  --overwrite
```

Run raw-video Stage 5:

```bash
python scripts/pipeline_v1/run_stage5.py \
  --dataroot data/sets/nuscenes_full \
  --version v1.0-trainval \
  --overwrite-videos
```

Run Stage 6:

```bash
python scripts/pipeline_v1/stage6_build_future_risk_groundtruth.py \
  --dataroot data/sets/nuscenes_full
```

Run Stage 7:

```bash
python scripts/pipeline_v1/stage7_create_qwen_future_vqa_dataset.py \
  --dataroot data/sets/nuscenes_full
```

Stage 7 writes clean training conversations to `qwen_future_vqa_dataset.json`
and `qwen_future_vqa_dataset.jsonl`. Training entries keep `video` paths as
model inputs and use vehicle reference names in questions and answers. Raw
nuScenes ids are stored only in `qwen_future_vqa_metadata.json` and
`qwen_future_vqa_metadata.jsonl` for traceability.

Run Stage 8:

```bash
python scripts/pipeline_v1/stage8_prepare_dataset_splits.py \
  --dataroot data/sets/nuscenes_full
```

For small tests, pass `--scene-name` or `--scene-names`, and use a separate
`--output-dir` / `--clip-dir` under `data/sets/test`.
