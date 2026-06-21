#!/bin/bash
# Wait until enough GPU memory is free, then launch the experiment.

REQUIRED_MB=20000       # minimum free VRAM per GPU required (MB); 7B model needs ~18GB
CHECK_INTERVAL=60       # seconds between checks
CMD="python geo_pipeline/evaluate.py --out /cvhci/temp/szuo/geo_results/run_fix1.json"

echo "[$(date)] Waiting for ${REQUIRED_MB}MB free VRAM..."

while true; do
    # Get maximum free memory across all GPUs
    MAX_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               | sort -n | tail -1)

    echo "[$(date)] Max free VRAM: ${MAX_FREE}MB"

    if [ "$MAX_FREE" -ge "$REQUIRED_MB" ]; then
        echo "[$(date)] GPU free! Launching experiment..."
        source /home/szuo/.local/opt/miniconda3/etc/profile.d/conda.sh
        conda activate /cvhci/temp/szuo/vllm-env
        MLLM_BACKEND=vllm MODEL_PATH=/cvhci/temp/szuo/models/qwen2.5-vl-7b $CMD
        echo "[$(date)] Experiment finished."
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
