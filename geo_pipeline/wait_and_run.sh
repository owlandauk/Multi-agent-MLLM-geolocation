#!/bin/bash
# Wait until ALL required GPUs are free, then launch the full 4K eval.
#
# Old bug: previous version took max free across GPUs, so when 1/4 cards
# were busy the script still fired and vLLM TP=4 OOM'd on the busy one.
# New logic: every required GPU must clear BOTH thresholds (free VRAM and
# utilization), and must stay clear for N consecutive checks (stability
# window) before launch — protects against a neighbour briefly releasing
# memory mid-job and grabbing it back.

set -u

# ── Config ────────────────────────────────────────────────────────────────────
GPUS="${GPUS:-0,1,2,3}"          # which GPUs we need (comma-separated indices)
REQUIRED_FREE_MB=10500           # min free VRAM per GPU. 11GB cards: leaves room
                                 # for a neighbour to grab <250MB without us OOMing
MAX_UTIL=5                       # max SM utilization % to consider idle
CHECK_INTERVAL=30                # seconds between probes
STABLE_CHECKS=3                  # consecutive idle probes before launching
                                 # (3 × 30s = 90s of confirmed idle)
FINAL_RECHECK_FREE_MB=10500      # last-second free VRAM check right before launch
                                 # (catches neighbour grabbing memory during the
                                 #  gap between final probe and vLLM init)

OUT="${OUT:-results/full_v3.json}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"     # e.g. EXTRA_ARGS="--limit 100 --start 0"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"    # vLLM --gpu-memory-utilization. 0.60 on
                                        # an 11GB card ≈ 6.6GB budget. Slower,
                                        # but less likely to die when another
                                        # user grabs a little memory after launch.
MAX_RETRIES="${MAX_RETRIES:-999}"       # if eval is killed/OOMs, wait again and retry
RETRY_SLEEP="${RETRY_SLEEP:-60}"        # pause before re-entering the wait loop
LOCK_FILE="${LOCK_FILE:-/tmp/geo_pipeline_wait_${USER}_${GPUS//,/}.lock}"

# Cooperative single-user lock. This prevents two jobs launched by the same
# account from racing each other. It cannot stop root, other users, or jobs that
# ignore this script.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "[$(date)] Another wait_and_run.sh appears to hold $LOCK_FILE; exit."
        exit 2
    fi
    echo "[$(date)] Acquired lock $LOCK_FILE"
fi

# ── Probe helpers ─────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "$GPUS"

probe_all_idle() {
    # Returns 0 if every GPU in GPU_LIST is idle (free VRAM >= REQUIRED_FREE_MB
    # AND utilization <= MAX_UTIL). Prints a per-GPU status line either way.
    local all_ok=1
    local report=""
    for g in "${GPU_LIST[@]}"; do
        # nvidia-smi -i <id> returns one line: "free_mb, util_pct"
        local line
        line=$(nvidia-smi -i "$g" \
               --query-gpu=memory.free,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null)
        if [ -z "$line" ]; then
            report+="  GPU $g: query failed\n"
            all_ok=0
            continue
        fi
        # strip spaces, split on comma
        local free util
        free=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
        util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')

        local ok="ok"
        if [ "$free" -lt "$REQUIRED_FREE_MB" ] || [ "$util" -gt "$MAX_UTIL" ]; then
            ok="BUSY"
            all_ok=0
        fi
        report+="  GPU $g: free=${free}MB util=${util}%  [$ok]\n"
    done
    printf '%b' "$report"
    return $((1 - all_ok))   # 0 = success when all_ok=1
}

wait_for_idle() {
    echo "[$(date)] Waiting for GPUs [$GPUS] to be idle"
    echo "  thresholds: free >= ${REQUIRED_FREE_MB} MB  AND  util <= ${MAX_UTIL}%"
    echo "  must stay idle for $STABLE_CHECKS consecutive checks (${CHECK_INTERVAL}s each)"

    streak=0
    while true; do
        echo
        echo "[$(date)] probe (streak=$streak/$STABLE_CHECKS)"
        if probe_all_idle; then
            streak=$((streak + 1))
            echo "  → all idle (streak=$streak/$STABLE_CHECKS)"
            if [ "$streak" -ge "$STABLE_CHECKS" ]; then
                break
            fi
        else
            if [ "$streak" -gt 0 ]; then
                echo "  → a GPU is busy, streak reset"
            fi
            streak=0
        fi
        sleep "$CHECK_INTERVAL"
    done

    echo
    echo "[$(date)] All GPUs stable. Final recheck before launch..."

    # Last-second recheck: between the final probe and vLLM init there's a window
    # where a neighbour can grab memory back. If any GPU dropped below the
    # threshold, restart the wait loop instead of OOMing inside vLLM.
    final_ok=1
    for g in "${GPU_LIST[@]}"; do
        free=$(nvidia-smi -i "$g" --query-gpu=memory.free \
               --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -z "$free" ] || [ "$free" -lt "$FINAL_RECHECK_FREE_MB" ]; then
            echo "  GPU $g: free=${free:-?}MB — dropped below ${FINAL_RECHECK_FREE_MB}MB"
            final_ok=0
        fi
    done
    return $((1 - final_ok))
}

source /home/szuo/.local/opt/miniconda3/etc/profile.d/conda.sh
conda activate /cvhci/temp/szuo/vllm-env

cd "$(dirname "$0")"   # cd into geo_pipeline/

attempt=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    if ! wait_for_idle; then
        echo "[$(date)] Final recheck failed. Re-entering wait loop after ${RETRY_SLEEP}s."
        sleep "$RETRY_SLEEP"
        continue
    fi

    echo "[$(date)] Launching eval (attempt $attempt/$MAX_RETRIES)."
    echo "  out=$OUT  batch_size=$BATCH_SIZE  gpu_mem_util=$GPU_MEM_UTIL  extra=$EXTRA_ARGS"

    MLLM_BACKEND=vllm \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    MODEL_PATH=/cvhci/temp/szuo/models/qwen2.5-vl-7b \
    VLLM_GPU_MEMORY_UTILIZATION="$GPU_MEM_UTIL" \
    python evaluate.py --batch_size "$BATCH_SIZE" --out "$OUT" $EXTRA_ARGS
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] eval done."
        exit 0
    fi

    echo "[$(date)] eval exited with code $rc. Waiting ${RETRY_SLEEP}s, then retrying."
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
done

echo "[$(date)] exhausted MAX_RETRIES=$MAX_RETRIES; exit 1."
exit 1
