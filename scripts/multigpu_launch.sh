#!/usr/bin/env bash
# Multi-GPU parallel training launcher for BraTS segmentation models
# Runs 4 models simultaneously on 4 RTX 5090 GPUs for maximum utilization
#
# Usage: bash scripts/multigpu_launch.sh

set -euo pipefail

PROJECT_DIR="/home/tangzh/bmeaiproj2"
VENV_PYTHON="/home/tangzh/venv314/bin/python"
DATA_ROOT="/home/tangzh/bmeaiproj2/data/brats2023"
CACHE_DIR="${PROJECT_DIR}/artifacts/preprocessed_cache"
SPLITS="${CACHE_DIR}/splits.json"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"

echo "============================================================"
echo " Multi-GPU BraTS Training Launcher (4x RTX 5090)"
echo " Epochs per model: ${TOTAL_EPOCHS}"
echo "============================================================"

# ── GPU 0: SAM ViT-B (frozen encoder, cached data) ─────────────────
echo "[GPU 0] Launching SAM ViT-B (frozen enc, bs=48, cached)..."
CUDA_VISIBLE_DEVICES=0 ${VENV_PYTHON} scripts/finetune_sam.py \
    --source vit \
    --data-root "${DATA_ROOT}" \
    --output-dir artifacts/sam_vit_gpu0 \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size 48 \
    --lr 1e-4 \
    --amp \
    --grad-accum-steps 2 \
    --cache-dir "${CACHE_DIR}" \
    --num-workers 0 \
    > /tmp/sam_gpu0.log 2>&1 &
PID_GPU0=$!
echo "   PID: ${PID_GPU0}"

# ── GPU 1: SAM ViT-B (unfrozen encoder, cached data) ───────────────
echo "[GPU 1] Launching SAM ViT-B (full FT, bs=32, cached)..."
CUDA_VISIBLE_DEVICES=1 ${VENV_PYTHON} scripts/finetune_sam.py \
    --source vit \
    --data-root "${DATA_ROOT}" \
    --output-dir artifacts/sam_vit_gpu1 \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size 32 \
    --lr 1e-4 \
    --amp \
    --grad-accum-steps 1 \
    --unfreeze-encoder \
    --cache-dir "${CACHE_DIR}" \
    --num-workers 0 \
    > /tmp/sam_gpu1.log 2>&1 &
PID_GPU1=$!
echo "   PID: ${PID_GPU1}"

# ── GPU 2: SwinUNETR training (cached) ─────────────────────────────
echo "[GPU 2] Launching SwinUNETR (bs=64, cached)..."
CUDA_VISIBLE_DEVICES=2 ${VENV_PYTHON} scripts/train_all_models.py \
    --model swin_unetr \
    --data-root "${DATA_ROOT}" \
    --output-dir artifacts/swin_unetr_gpu2 \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size 64 \
    --lr 1e-3 \
    --amp \
    --splits "${SPLITS}" \
    --cache-dir "${CACHE_DIR}" \
    --num-workers 0 \
    > /tmp/swin_gpu2.log 2>&1 &
PID_GPU2=$!
echo "   PID: ${PID_GPU2}"

# ── GPU 3: SegFormer LoRA (PEFT) ────────────────────────────────────
echo "[GPU 3] Launching SegFormer-B0 LoRA (bs=32, PEFT)..."
CUDA_VISIBLE_DEVICES=3 ${VENV_PYTHON} scripts/finetune_segformer.py \
    --model nvidia/mit-b0 \
    --data-root "${DATA_ROOT}" \
    --output-dir artifacts/segformer_gpu3 \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size 32 \
    --lr 5e-4 \
    --mixed-precision bf16 \
    --grad-accum 2 \
    --num-workers 0 \
    > /tmp/segformer_gpu3.log 2>&1 &
PID_GPU3=$!
echo "   PID: ${PID_GPU3}"

echo ""
echo "============================================================"
echo " All 4 GPUs launched! PIDs: ${PID_GPU0} ${PID_GPU1} ${PID_GPU2} ${PID_GPU3}"
echo " Check GPU utilization: watch -n 1 nvidia-smi"
echo " Check logs: tail -f /tmp/sam_gpu0.log"
echo "============================================================"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
wait ${PID_GPU0} ${PID_GPU1} ${PID_GPU2} ${PID_GPU3} 2>/dev/null || true

echo ""
echo "============================================================"
echo " All training jobs completed!"
echo "============================================================"

# Print summary of all artifacts
for dir in artifacts/sam_vit_gpu0 artifacts/sam_vit_gpu1 artifacts/swin_unetr_gpu2 artifacts/segformer_gpu3; do
    if [ -f "${dir}/history.csv" ]; then
        echo ""
        echo "--- ${dir} ---"
        echo "Best metrics:"
        tail -1 "${dir}/history.csv" 2>/dev/null || echo "(empty)"
        if [ -f "${dir}/best_metrics.json" ]; then
            python3 -c "import json; d=json.load(open('${dir}/best_metrics.json')); print(f'Dice WT={d[\"dice\"][\"WT\"]:.3f} TC={d[\"dice\"][\"TC\"]:.3f} ET={d[\"dice\"][\"ET\"]:.3f}')" 2>/dev/null || true
        fi
    fi
done
