# Training Code Status

This directory keeps the completed Stage 1 Qwen3-VL JSON SFT workflow and the
new custom Stage 2--5 risk-reasoning workflow.

## Stage 1: Qwen3-VL JSON SFT

Stage 1 uses LLaMA-Factory to fine-tune Qwen3-VL with LoRA on the visible VQA
question and fixed structured JSON answer.

Export VQA data:

```bash
PYTHONPATH=training python training/scripts/export_llamafactory_nurisk.py \
  --train-json data/sets/training_test/nurisk_style/dataset_splits/train_nuscenes.json \
  --validation-json data/sets/training_test/nurisk_style/dataset_splits/validation_nuscenes.json \
  --vqa-root data/sets/training_test/nurisk_style \
  --output-dir training/llamafactory/data
```

Train:

```bash
llamafactory-cli train training/llamafactory/stage1_qwen3vl_json_sft_final.yaml
```

Completed Stage 1 adapter:

```text
training/outputs/llamafactory/stage1_qwen3vl_json_sft_final
```

Prediction outputs:

```text
training/outputs/llamafactory/stage1_qwen3vl_json_sft_predict_validate20
training/outputs/llamafactory/stage1_qwen3vl_json_sft_predict_validate5_long
```

Field-level validation outputs:

```text
training/outputs/validation/stage1_qwen3vl_json_sft_fields_validate20
training/outputs/validation/stage1_qwen3vl_json_sft_fields_validate5_long
```

## Stage 2--5: risk-reasoning workflow

The custom trainer starts from the completed Stage 1 LoRA adapter and adds:

```text
Qwen vision -> pretrained BLIP-2 Q-Former -> 32 visual tokens
    -> Qwen + 8 risk latent tokens -> H_R
    -> recurrent latent CoT z1,z2,z3,z4
    -> risk heads + Qwen native LM draft
    -> structured JSON assembly
```

The Stage 4/5 risk heads read recurrent CoT states, not the raw `H_R`, so the
implicit CoT must support the main risk objective.

Recommended configs:

```text
training/configs/stage2_qformer_answer_alignment.yaml
training/configs/stage3_risk_latent_warmup.yaml
training/configs/stage4a_recurrent_latent_cot_risk.yaml
training/configs/stage4b_cot_decoder_warmup.yaml
training/configs/stage5_joint_answer_risk_cot.yaml
```

Losses:

```text
Stage 2:  L_answer^CE
Stage 3:  L_answer^CE + lambda_r L_risk
Stage 4A: L_risk + optional L_answer^CE
Stage 4B: L_CoT^CE
Stage 5:  lambda_a L_answer^CE + lambda_r L_risk + lambda_c L_CoT^CE
```

Risk scores 0--5 are treated as classification targets with cross entropy, not
as default MSE regression. TTC/DTC/speed/acceleration/lateral offset remain
regression targets; longitudinal and lateral ego meta-actions are separate
classification heads.

Run a lightweight smoke test before any real training:

```bash
PYTHONPATH=training python training/train_risk_trainer.py \
  --config training/configs/stage2_qformer_answer_alignment.yaml \
  --forward-only \
  --max-train-samples 1 \
  --max-validation-samples 1
```
