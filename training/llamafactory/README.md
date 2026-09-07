# Stage 1 LLaMA-Factory Integration

This folder contains the Stage 1 Qwen3-VL LoRA SFT setup. Stage 1 learns the
fixed structured JSON answer format directly with LLaMA-Factory.

Export the current NuRisk VQA splits to LLaMA-Factory's multimodal ShareGPT
video format:

```bash
PYTHONPATH=training python training/scripts/export_llamafactory_nurisk.py \
  --train-json data/sets/training_test/nurisk_style/dataset_splits/train_nuscenes.json \
  --validation-json data/sets/training_test/nurisk_style/dataset_splits/validation_nuscenes.json \
  --vqa-root data/sets/training_test/nurisk_style \
  --output-dir training/llamafactory/data
```

Then run the mature LLaMA-Factory trainer:

```bash
llamafactory-cli train training/llamafactory/stage1_qwen3vl_json_sft_final.yaml
```

This stage trains the heavy Qwen3-VL LoRA adapter with LLaMA-Factory's SFT
pipeline: optimizer, scheduler, mixed precision, distributed training,
checkpointing, logging, and evaluation are delegated to LLaMA-Factory.

The exported dataset keeps the visible VQA supervision used by Stage 1:

```text
messages: short JSON answer without reasoning_summary
videos: six synchronized camera video paths, matched by six <video> tags
```

Run prediction or validation with the Stage 1 helper scripts:

```bash
PYTHONPATH=training python training/scripts/run_llamafactory_predict.py
PYTHONPATH=training python training/scripts/validate_llamafactory_adapter.py
```

The current completed Stage 1 adapter is stored at:

```text
training/outputs/llamafactory/stage1_qwen3vl_json_sft_final
```
