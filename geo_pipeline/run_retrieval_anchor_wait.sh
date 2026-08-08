#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NAME="${NAME:-retrieval_country_anchor_w010_limit1500}"
OUT="${OUT:-results/retrieval_country_anchor_w010_limit1500.json}"
RUNLOG="${RUNLOG:-results/retrieval_country_anchor_w010_limit1500.log}"
WAITLOG="${WAITLOG:-results/retrieval_country_anchor_w010_limit1500.wait.log}"
ANALYSIS="${ANALYSIS:-results/retrieval_country_anchor_w010_limit1500.analysis.txt}"
AUTO_FULL="${AUTO_FULL:-1}"
FULL_NAME="${FULL_NAME:-retrieval_country_anchor_w010_full}"
FULL_OUT="${FULL_OUT:-results/retrieval_country_anchor_w010_full.json}"
FULL_RUNLOG="${FULL_RUNLOG:-results/retrieval_country_anchor_w010_full.log}"
FULL_ANALYSIS="${FULL_ANALYSIS:-results/retrieval_country_anchor_w010_full.analysis.txt}"
GATE_COUNTRY="${GATE_COUNTRY:-45.5}"
GATE_CONTINENT="${GATE_CONTINENT:-72.0}"
LIMIT="${LIMIT:-1500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPUS="${GPUS:-0 1}"
REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-90000}"
MAX_UTIL="${MAX_UTIL:-5}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
PYTHON_BIN="${PYTHON_BIN:-/cvhci/temp/szuo/vllm-blackwell-env/bin/python}"

mkdir -p results
exec >> "$WAITLOG" 2>&1

echo "wait_start $(date --iso-8601=seconds) name=$NAME gpus=[$GPUS]"

chosen=""
streak_gpu=""
streak=0
while [ -z "$chosen" ]; do
    for gpu in $GPUS; do
        line=$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
        free=${line%%,*}
        util=${line##*,}
        free=${free// /}
        util=${util// /}
        echo "probe $(date --iso-8601=seconds) gpu=$gpu free=${free:-?} util=${util:-?} streak_gpu=${streak_gpu:-none} streak=$streak"
        if [ -n "$free" ] && [ -n "$util" ] && [ "$free" -ge "$REQUIRED_FREE_MB" ] && [ "$util" -le "$MAX_UTIL" ]; then
            if [ "$streak_gpu" = "$gpu" ]; then
                streak=$((streak + 1))
            else
                streak_gpu="$gpu"
                streak=1
            fi
            if [ "$streak" -ge "$STABLE_CHECKS" ]; then
                chosen="$gpu"
                break
            fi
        elif [ "$streak_gpu" = "$gpu" ]; then
            streak_gpu=""
            streak=0
        fi
    done
    if [ -z "$chosen" ]; then
        sleep "$CHECK_INTERVAL"
    fi
done

echo "launch $(date --iso-8601=seconds) gpu=$chosen"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "missing_python $(date --iso-8601=seconds) path=$PYTHON_BIN"
    exit 1
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

export CUDA_VISIBLE_DEVICES="$chosen"
export VLLM_TP=1
export MLLM_BACKEND="${MLLM_BACKEND:-vllm}"
export MODEL_PATH="${MODEL_PATH:-/cvhci/temp/szuo/models/qwen2.5-vl-7b}"
export WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-0}"
export RETRIEVAL_PRIOR_ENABLED="${RETRIEVAL_PRIOR_ENABLED:-1}"
export RETRIEVAL_PRIOR_PATH="${RETRIEVAL_PRIOR_PATH:-results/geoclip_prior_full.json}"
export RETRIEVAL_PRIOR_WEIGHT="${RETRIEVAL_PRIOR_WEIGHT:-0.10}"
export RETRIEVAL_COUNTRY_ANCHOR_ENABLED="${RETRIEVAL_COUNTRY_ANCHOR_ENABLED:-1}"
export RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP="${RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP:-0.55}"
export RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP="${RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP:-0.15}"
export RETRIEVAL_COUNTRY_ANCHOR_WEIGHT="${RETRIEVAL_COUNTRY_ANCHOR_WEIGHT:-1.0}"
export ENABLE_CONTINENT_LEVEL="${ENABLE_CONTINENT_LEVEL:-0}"

"$PYTHON_BIN" run_experiment.py \
    --name "$NAME" \
    --limit "$LIMIT" \
    --batch_size "$BATCH_SIZE" \
    --out "$OUT" \
    --log "$RUNLOG" \
    --notes "pipeline_retrieval_country_anchor_w010_limit${LIMIT}"
rc=$?
echo "run_done $(date --iso-8601=seconds) rc=$rc"

if [ "$rc" -eq 0 ]; then
    "$PYTHON_BIN" analyze_results.py --pred "$OUT" > "$ANALYSIS" 2>&1
    echo "analyzed $(date --iso-8601=seconds) $ANALYSIS"
fi

if [ "$rc" -eq 0 ] && [ "$AUTO_FULL" = "1" ]; then
    if "$PYTHON_BIN" - "$OUT" "$GATE_COUNTRY" "$GATE_CONTINENT" <<'PY'
import json
import sys
from analyze_results import analyze

out, gate_country, gate_continent = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open(out, encoding="utf-8") as f:
    report = analyze(json.load(f).get("records", []))
country = float(report["accuracy"]["750"])
continent = float(report["accuracy"]["2500"])
print(f"gate country={country:.2f}/{gate_country:.2f} continent={continent:.2f}/{gate_continent:.2f}")
sys.exit(0 if country >= gate_country and continent >= gate_continent else 1)
PY
    then
        echo "gate_pass $(date --iso-8601=seconds); launching full"
        "$PYTHON_BIN" run_experiment.py \
            --name "$FULL_NAME" \
            --batch_size "$BATCH_SIZE" \
            --out "$FULL_OUT" \
            --log "$FULL_RUNLOG" \
            --notes "pipeline_retrieval_country_anchor_w010_full_after_limit${LIMIT}_gate"
        full_rc=$?
        echo "full_done $(date --iso-8601=seconds) rc=$full_rc"
        if [ "$full_rc" -eq 0 ]; then
            "$PYTHON_BIN" analyze_results.py --pred "$FULL_OUT" > "$FULL_ANALYSIS" 2>&1
            echo "full_analyzed $(date --iso-8601=seconds) $FULL_ANALYSIS"
        fi
        exit "$full_rc"
    else
        echo "gate_fail $(date --iso-8601=seconds); full not started"
    fi
fi

exit "$rc"
