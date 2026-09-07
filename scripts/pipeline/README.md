# Annotated nuScenes VQA pipeline

This folder is an isolated copy/adaptation of `scripts/nurisk_style_pipeline`.
It does not modify nuScenes SDK code.

Stages 1-4 and Stage 8 keep the original NuRisk-style logic. Stages 5a, 5b,
6, and 7 carry scene-local readable agent aliases so the generated video clips
and VQA questions refer to visible ids such as `A003`, while metadata preserves
the raw nuScenes `instance_token` for exact joins.

Default outputs:

```text
<dataroot>/nurisk_style/
<dataroot>/annotated_video_clip_dataset/
```

Identity policy:

```text
display_agent_id: A001, A002, ...     # shown on video and used in VQA
scene_agent_id:   scene-0001_A001     # globally readable display key
instance_token:   raw nuScenes id     # durable original id
risk_agent_id:    Obstacle <token>    # Stage 1-4/6 risk key
```

Aliases are assigned once per scene in first-observed annotation order and are
reused for every clip from that scene. A bare `A001` is scene-local; use
`scene_name + agent_id` or `scene_agent_id` when a global key is needed.

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
  -> annotated_video_clip_dataset/videos/<scene>/<camera>/*.mp4
  -> annotated_video_clip_dataset/metadata/<scene>_clips.json

Stage 5b: stage5b_align_video_clip_groundtruth.py
  annotated_video_clip_dataset/metadata/<scene>_clips.json
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

Run through Stage 4:

```bash
python scripts/pipeline/run_until_stage4.py \
  --dataroot data/sets/nuscenes_full \
  --version v1.0-trainval \
  --overwrite
```

Run annotated Stage 5:

```bash
python scripts/pipeline/run_stage5.py \
  --dataroot data/sets/nuscenes_full \
  --version v1.0-trainval \
  --overwrite-videos
```

Run Stage 6:

```bash
python scripts/pipeline/stage6_build_future_risk_groundtruth.py \
  --dataroot data/sets/nuscenes_full
```

Run Stage 7:

```bash
python scripts/pipeline/stage7_create_qwen_future_vqa_dataset.py \
  --dataroot data/sets/nuscenes_full
```

Stage 7 writes clean training conversations to `qwen_future_vqa_dataset.json`
and `qwen_future_vqa_dataset.jsonl`. The training entries keep `video` paths as
model inputs and use scene-local aliases such as `A001` in questions and
answers. Raw nuScenes ids are not part of the conversation target; they are
stored only in `qwen_future_vqa_metadata.json` and
`qwen_future_vqa_metadata.jsonl` for traceability.

Run Stage 8:

```bash
python scripts/pipeline/stage8_prepare_dataset_splits.py \
  --dataroot data/sets/nuscenes_full
```

For small tests, pass `--scene-name` or `--scene-names`, and use a separate
`--output-dir` / `--clip-dir` under `data/sets/test`.
