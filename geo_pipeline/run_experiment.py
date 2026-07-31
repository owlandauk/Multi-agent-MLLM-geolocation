"""Run one evaluation, analyze it, and append a durable experiment record.

This is a thin wrapper around evaluate.py. It does not change inference logic;
it only keeps experiment bookkeeping honest during iterative server runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from analyze_results import analyze


ENV_KEYS = (
    "MODEL_PATH",
    "MLLM_BACKEND",
    "CUDA_VISIBLE_DEVICES",
    "VLLM_TP",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_LEVELS",
    "VERIFY_SUPPORT_FORMAT",
    "STRICT_COUNTRY_ALIAS_MATCH",
    "COUNTRY_CUE_ENSEMBLE",
    "POMDP_POLICY",
    "POMDP_EIG_LEVELS",
    "POMDP_OBS_SAMPLES",
    "POMDP_MAX_ACTIONS",
    "POMDP_REWARD_MODE",
    "TRANSITION_THR",
    "SL_N_SAMPLES",
    "ENABLE_CONTINENT_LEVEL",
    "CONTINENT_REG_MIN_TOP",
    "CONTINENT_REG_STRENGTH",
    "CONTINENT_REG_FLOOR",
)


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "experiment"


def _git_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _run_and_log(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()


def _load_report(out_path: Path) -> dict:
    with out_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return analyze(data.get("records", []))


def _accuracy(report: dict, threshold: int) -> float | None:
    return (report.get("accuracy") or {}).get(str(threshold))


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _append_csv(path: Path, record: dict) -> None:
    fields = [
        "name",
        "status",
        "git_sha",
        "start_time",
        "end_time",
        "duration_sec",
        "limit",
        "start_index",
        "batch_size",
        "out",
        "log",
        "street_1km",
        "city_25km",
        "region_200km",
        "country_750km",
        "continent_2500km",
        "unknown_country_rate",
        "country_child_conflict_rate",
        "country_stable_rate",
        "continent_stable_rate",
        "pomdp_policy",
        "notes",
    ]
    flat = {field: record.get(field) for field in fields}
    flat["pomdp_policy"] = ";".join(
        f"{k}:{v}" for k, v in (record.get("pomdp_policy") or {}).items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--strict_child_geocode", action="store_true")
    parser.add_argument("--disable_bare_city_geocode", action="store_true")
    parser.add_argument("--records_jsonl", default="results/experiment_runs.jsonl")
    parser.add_argument("--records_csv", default="results/experiment_runs.csv")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    slug = _slug(args.name)
    out_path = Path(args.out or f"results/{timestamp}_{slug}.json")
    log_path = Path(args.log or out_path.with_suffix(".log"))

    cmd = [
        sys.executable,
        "evaluate.py",
        "--start",
        str(args.start),
        "--batch_size",
        str(args.batch_size),
        "--out",
        str(out_path),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.strict_child_geocode:
        cmd.append("--strict_child_geocode")
    if args.disable_bare_city_geocode:
        cmd.append("--disable_bare_city_geocode")

    env_snapshot = {key: os.environ.get(key) for key in ENV_KEYS if os.environ.get(key) is not None}
    start_wall = time.time()
    start_time = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "ok"

    print(f"[RUN] {args.name}")
    print(f"[RUN] start_time={start_time}")
    print(f"[RUN] out={out_path} log={log_path}")
    print(f"[RUN] cmd={' '.join(cmd)}")
    print(f"[RUN] env={json.dumps(env_snapshot, sort_keys=True)}")

    return_code = _run_and_log(cmd, cwd, log_path)
    if return_code != 0:
        status = f"failed:{return_code}"

    report = _load_report(cwd / out_path) if (cwd / out_path).exists() else {}
    end_time = datetime.now().astimezone().isoformat(timespec="seconds")
    duration_sec = round(time.time() - start_wall, 2)

    record = {
        "name": args.name,
        "status": status,
        "git_sha": _git_sha(cwd),
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": duration_sec,
        "limit": args.limit,
        "start_index": args.start,
        "batch_size": args.batch_size,
        "out": str(out_path),
        "log": str(log_path),
        "env": env_snapshot,
        "pomdp_policy": report.get("pomdp_policy"),
        "street_1km": _accuracy(report, 1),
        "city_25km": _accuracy(report, 25),
        "region_200km": _accuracy(report, 200),
        "country_750km": _accuracy(report, 750),
        "continent_2500km": _accuracy(report, 2500),
        "unknown_country_rate": report.get("unknown_country_rate"),
        "country_child_conflict_rate": report.get("country_child_conflict_rate"),
        "country_stable_rate": report.get("country_stable_rate"),
        "continent_stable_rate": report.get("continent_stable_rate"),
        "notes": args.notes,
    }
    _append_jsonl(cwd / args.records_jsonl, record)
    _append_csv(cwd / args.records_csv, record)

    print("[RUN] end_time=" + end_time)
    print(f"[RUN] duration_sec={duration_sec}")
    print(
        "[RUN] accuracy "
        f"street={record['street_1km']} city={record['city_25km']} "
        f"region={record['region_200km']} country={record['country_750km']} "
        f"continent={record['continent_2500km']}"
    )
    print(f"[RUN] appended={args.records_jsonl} {args.records_csv}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
