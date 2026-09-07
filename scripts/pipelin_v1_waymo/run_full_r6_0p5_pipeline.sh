#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/dellpro2/zhongwen/nuscenes-devkit}"
DATAROOT="${DATAROOT:-$REPO_ROOT/data/sets/waymo}"
WAYMO_SPLIT="${WAYMO_SPLIT:-training}"
CONDA_SH="${CONDA_SH:-/home/dellpro2/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cr37}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/data/sets/waymo/waymo_r6_0p5_pipeline.log}"
LOCK_DIR="${LOCK_DIR:-$DATAROOT/.waymo_r6_0p5_pipeline.lock}"

cd "$REPO_ROOT"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another Waymo pipeline appears to be running."
  echo "Lock dir: $LOCK_DIR"
  echo "If you are sure no pipeline is running, remove the lock dir and retry:"
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

log_section "Stage5: build/resume video clips and clip groundtruth"
python scripts/pipelin_v1_waymo/run_stage5.py \
  --dataroot "$DATAROOT" \
  --split "$WAYMO_SPLIT"
git_guard "After Stage5"

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

log_section "Waymo R6 0.5s full pipeline complete"
echo "Log file: $LOG_FILE"
