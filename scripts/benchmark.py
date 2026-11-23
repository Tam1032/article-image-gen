"""Benchmark a running deployment: latency per stage and peak GPU memory.

Uses only the standard library, so it runs with the server's system python3:

    python3 scripts/benchmark.py                      # default article set
    python3 scripts/benchmark.py --images 2 --runs 2  # more load

Prints a Markdown table (paste it into the README) and saves raw results as JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ARTICLES = [
    {"title": "Metro số 1 TP.HCM chính thức vận hành thương mại", "size": "featured"},
    {"title": "5 cách tiết kiệm điện cho gia đình trong mùa nắng nóng", "size": "inline"},
    {"title": "Giá cà phê xuất khẩu tăng mạnh, nông dân Tây Nguyên phấn khởi", "size": "featured"},
    {"title": "How small cafés use AI to plan their weekly menu", "size": "square"},
    {"title": "Du lịch Đà Lạt mùa hoa dã quỳ: lịch trình 3 ngày 2 đêm", "size": "inline"},
]


def request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


class GpuMonitor:
    """Samples used GPU memory (MiB) with nvidia-smi in a background thread."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                used = max(int(x) for x in out.split())
                self.peak_mib = max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMonitor":
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()


def run_job(api: str, payload: dict, poll_s: float = 0.25, timeout_s: float = 1800) -> dict:
    start = time.monotonic()
    job_id = request("POST", f"{api}/v1/jobs", payload)["job_id"]
    planning_done = None
    while True:
        job = request("GET", f"{api}/v1/jobs/{job_id}")
        now = time.monotonic() - start
        if job["status"] == "generating" and planning_done is None:
            planning_done = now
        if job["status"] in ("completed", "failed"):
            break
        if now > timeout_s:
            raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")
        time.sleep(poll_s)
    total = time.monotonic() - start
    n = payload["num_images"]
    planning = planning_done if planning_done is not None else float("nan")
    return {
        "job_id": job_id,
        "title": payload["title"],
        "size": payload["size"],
        "num_images": n,
        "status": job["status"],
        "error": job.get("error"),
        "total_s": round(total, 1),
        "planning_s": round(planning, 1),
        "per_image_s": round((total - planning) / n, 1),
    }


def fmt(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.median(values):.1f} ({min(values):.1f}–{max(values):.1f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--images", type=int, default=1, help="images per job (1-4)")
    parser.add_argument("--runs", type=int, default=1, help="passes over the article set")
    parser.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up job")
    parser.add_argument("--out", default="data/benchmark.json")
    args = parser.parse_args()

    health = request("GET", f"{args.api}/health")
    if health.get("status") != "ok":
        raise SystemExit(f"Service not healthy: {health}")

    if not args.no_warmup:
        print("Warm-up job (loads models, not counted)...", flush=True)
        warm = run_job(args.api, {**ARTICLES[0], "num_images": 1})
        print(f"  cold start: {warm['total_s']} s", flush=True)

    results = []
    with GpuMonitor() as gpu:
        for r in range(args.runs):
            for article in ARTICLES:
                res = run_job(args.api, {**article, "num_images": args.images})
                results.append(res)
                print(f"  [{res['status']}] {res['total_s']:>6.1f} s  "
                      f"plan {res['planning_s']:>5.1f} s  "
                      f"{res['per_image_s']:>5.1f} s/img  {res['title'][:50]}", flush=True)

    ok = [r for r in results if r["status"] == "completed"]
    summary = {
        "jobs": len(results),
        "failed": len(results) - len(ok),
        "images_per_job": args.images,
        "cold_start_s": None if args.no_warmup else warm["total_s"],
        "peak_vram_mib": gpu.peak_mib if gpu.available else None,
        "health": health,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nMedian (min–max), warm models:\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Jobs (images/job) | {len(ok)} ok / {len(results)} ({args.images}) |")
    print(f"| End-to-end per job (s) | {fmt([r['total_s'] for r in ok])} |")
    print(f"| LLM planning (s) | {fmt([r['planning_s'] for r in ok])} |")
    print(f"| FLUX + post-processing per image (s) | {fmt([r['per_image_s'] for r in ok])} |")
    if summary["cold_start_s"] is not None:
        print(f"| Cold start, first job (s) | {summary['cold_start_s']:.1f} |")
    if summary["peak_vram_mib"]:
        print(f"| Peak GPU memory (GiB) | {summary['peak_vram_mib'] / 1024:.1f} |")
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    main()
