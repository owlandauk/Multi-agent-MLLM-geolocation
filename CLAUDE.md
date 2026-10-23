# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` still holds the canonical coding-style, testing, and security rules — read it for those. It predates the GeoCLIP-rerank pivot, so trust this file for architecture and commands.

## What this project is

Master's thesis extending GeoBayes (image geo-localization as MAP estimation over geographic hypotheses, updated by an MLLM). Two lines of work coexist in `geo_pipeline/`:

1. **Full pipeline** (`pipeline.py` → `GeoPipeline.predict` / `predict_batch`): coarse-to-fine Country → City → Street inference, extending GeoBayes with three modules in `modules/`:
   - `sl.py` (`SLModule`): samples the MLLM N times per evidence item, shrinks likelihood toward neutral when variance is high.
   - `dst.py` (`DSTModule`): Dempster-Shafer evidence fusion; falls back to Yager's cautious rule when conflict mass K exceeds `DST_CONFLICT_THR`.
   - `pomdp.py` (`POMDPModule`): LLM-as-policy (or expected-information-gain) selection of the next verification task.
2. **GeoCLIP-rerank line** (current thesis direction): GeoCLIP supplies top-k GPS candidates; Qwen only *reranks within* those candidates (never invents a new location); the answer falls back to GeoCLIP rank-1 unless the reranked posterior clears an override margin. Scripts: `build_geoclip_candidate_cache.py` → `evaluate_geoclip_qwen_rerank.py` / `evaluate_geoclip_candidates.py`. Helpers in `geoclip_candidates.py`. The goal is to approach GeoCLIP oracle@5/@10, not to beat GeoCLIP from scratch.

`config.py` centralizes every hyperparameter and path; nearly all are overridable via environment variables (used for ablations). `country_aliases.py` collapses country-name variants to canonical form.

## Backends (`models/mllm_client.py`)

One `MLLMClient` facade over three backends, selected by `MLLM_BACKEND`:
- `dashscope` — API, for small smoke tests (needs `DASHSCOPE_API_KEY`).
- `local` — transformers, `device_map="auto"` (slow, pipeline-parallel).
- `vllm` — tensor-parallel + PagedAttention (default for GPU servers). `VLLM_TP` sets tensor-parallel size, `VLLM_GPU_MEMORY_UTILIZATION` the memory budget.

All backends expose the same surface: `generate` / `sample_n` / `batch_generate` / `batch_sample_n`.

## Commands

Python **3.10+ is required** (code uses `X | None` syntax). macOS default `python3` is often 3.9 and will fail to import — use the server env.

Run tests (unittest; each test injects `geo_pipeline/` into `sys.path`):

```bash
python3 -m unittest discover -s tests -v
# single test:
python3 -m unittest tests.test_sl_support_scores -v
```

Smoke test (API):

```bash
export MLLM_BACKEND=dashscope DASHSCOPE_API_KEY=...
python3 geo_pipeline/evaluate.py --limit 10 --out geo_pipeline/results/test_run.json
```

Full-pipeline eval (vLLM). Run from inside `geo_pipeline/`:

```bash
MLLM_BACKEND=vllm CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/qwen2.5-vl-7b VLLM_TP=4 \
python evaluate.py --batch_size 20 --out results/run.json
# resume from a checkpoint index:
python evaluate.py --start 1000 --out results/run2.json
```

GeoCLIP rerank line (build cache once, then rerank):

```bash
python build_geoclip_candidate_cache.py --top_k 25 --out results/geoclip_candidates.json
python evaluate_geoclip_qwen_rerank.py --candidate_cache results/geoclip_candidates.json \
  --limit 500 --batch_size 8 --top_k 5 --out results/rerank.json
```

Bookkept run (wraps `evaluate.py`, appends metrics + git SHA + env snapshot to `results/experiment_runs.{jsonl,csv}`):

```bash
python run_experiment.py --name my_ablation --limit 500 --batch_size 8
```

Unattended GPU runs (`wait_and_run.sh` waits for all GPUs to clear thresholds for N consecutive probes before launching; `run_retrieval_anchor_wait.sh` similar). Check syntax before use: `bash -n geo_pipeline/wait_and_run.sh`.

## Evaluation

Distance-threshold accuracy on YFCC4K: `EVAL_THRESHOLDS = [1, 25, 200, 750, 2500]` km = Street / City / Region / Country / Continent. `analyze_results.py` (`analyze()`) turns a records JSON into the accuracy report and diagnostics.

## Working notes

- Current full-pipeline diagnosis and the v7 rationale live in `geo_pipeline/results/v6b_diagnosis_notes.md`; experiment lessons and the optimization/thesis route in `docs/`.
- `results/` holds run artifacts — avoid committing large transient runs; only intentional experiment outputs.
- Keep model weights, datasets, and caches outside the repo; never commit API keys (`.env` is gitignored).
- Server paths differ from local. On the KIT Blackwell server (i14s48) the venv is `/cvhci/temp/szuo/vllm-blackwell-env`; the older `vllm-env` has a PyTorch too old for sm_120.
