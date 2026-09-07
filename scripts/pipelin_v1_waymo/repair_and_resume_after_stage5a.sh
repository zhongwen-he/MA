#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dellpro2/zhongwen/nuscenes-devkit}"
DATAROOT="${DATAROOT:-$REPO_ROOT/data/sets/waymo}"
WAYMO_SPLIT="${WAYMO_SPLIT:-training}"
CONDA_SH="${CONDA_SH:-/home/dellpro2/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cr37}"
LOG_FILE="${LOG_FILE:-$DATAROOT/waymo_r6_0p5_repair_resume.log}"
LOCK_DIR="${LOCK_DIR:-$DATAROOT/.waymo_r6_0p5_pipeline.lock}"

cd "$REPO_ROOT"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another Waymo pipeline appears to be running."
  echo "Lock dir: $LOCK_DIR"
  echo "Check running processes first:"
  echo "  ps -ef | grep -E 'stage5a_build_waymo|stage5b_align|stage6_build_future|stage7_create_qwen|stage8_prepare' | grep -v grep"
  echo "If no pipeline is running, remove the lock dir and retry:"
  echo "  rm -rf '$LOCK_DIR'"
  exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

log_section() {
  echo
  echo "===== $(date '+%F %T') :: $1 ====="
}

git_guard() {
  log_section "$1"
  git check-ignore -v \
    "$DATAROOT" \
    "$DATAROOT/video_clip_dataset" \
    "$DATAROOT/nurisk_style" \
    || {
      echo "ERROR: Waymo data paths are not ignored by Git. Stop to avoid .git bloat."
      exit 1
    }
  du -sh .git || true
  git count-objects -vH || true
  df -h . || true
}

bad_manifest_segments() {
  python - <<'PY'
from pathlib import Path
import json
root = Path("/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/waymo/video_clip_dataset/metadata")
bad = []
for p in sorted(root.glob("*_clips.json")):
    try:
        with p.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        bad.append(p.name[:-len("_clips.json")])
print(",".join(bad))
PY
}

count_bad_manifests() {
  python - <<'PY'
from pathlib import Path
import json
root = Path("/home/dellpro2/zhongwen/nuscenes-devkit/data/sets/waymo/video_clip_dataset/metadata")
count = 0
for p in sorted(root.glob("*_clips.json")):
    try:
        with p.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        count += 1
print(count)
PY
}

log_section "Activate environment"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
unset PYTHONPATH

python - <<'PY'
import importlib.util
missing = [name for name in ("pandas", "pyarrow") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python modules: {', '.join(missing)}")
print("Python dependency check passed: pandas, pyarrow")
PY

git_guard "Start guard"

BAD_SEGMENTS="$(bad_manifest_segments)"
BAD_COUNT="$(count_bad_manifests)"
echo "Bad manifest count before repair: $BAD_COUNT"

if [[ "$BAD_COUNT" != "0" ]]; then
  echo "Repairing bad manifest segments only."
  python scripts/pipelin_v1_waymo/stage5a_build_waymo_video_clips.py \
    --dataroot "$DATAROOT" \
    --split "$WAYMO_SPLIT" \
    --channels all \
    --fps 2.0 \
    --clip-len 5 \
    --clip-stride 1 \
    --sample-interval-seconds 0.5 \
    --segment-names "$BAD_SEGMENTS"
fi

BAD_COUNT_AFTER="$(count_bad_manifests)"
echo "Bad manifest count after repair: $BAD_COUNT_AFTER"
if [[ "$BAD_COUNT_AFTER" != "0" ]]; then
  echo "ERROR: manifest repair failed; do not continue to Stage5b."
  exit 1
fi

git_guard "After manifest repair"

log_section "Stage5b: rebuild clip groundtruth from repaired manifests"
python scripts/pipelin_v1_waymo/stage5b_align_video_clip_groundtruth.py \
  --dataroot "$DATAROOT" \
  --keyframe-interval-seconds 0.5
git_guard "After Stage5b"

log_section "Stage6: build R6 future-risk labels, horizon=0.5s"
python scripts/pipelin_v1_waymo/stage6_build_future_risk_groundtruth.py \
  --dataroot "$DATAROOT" \
  --future-horizon-seconds 0.5
git_guard "After Stage6"

log_section "Stage7: build Qwen/LLaVA-style VQA dataset"
python scripts/pipelin_v1_waymo/stage7_create_qwen_future_vqa_dataset.py \
  --dataroot "$DATAROOT"
git_guard "After Stage7"

log_section "Stage8: prepare train/validation splits"
python scripts/pipelin_v1_waymo/stage8_prepare_dataset_splits.py \
  --dataroot "$DATAROOT"
git_guard "After Stage8"

log_section "Waymo repair/resume complete"
echo "Log file: $LOG_FILE"
