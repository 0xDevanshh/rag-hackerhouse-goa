"""Concurrency smoke benchmark for the text RAG route.

Usage:
    python benchmarks/run_concurrency.py --levels 1 5 10 25 --requests 25

The script uses the live FastAPI app through ASGITransport, so it measures
actual request orchestration without requiring a second server process. It
reports client-observed P50/P70/P95/P100 and fails if the route is unavailable.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import api  # noqa: E402


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(percentile: int) -> float:
        position = (len(ordered) - 1) * percentile / 100
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {f"p{p}": at(p) for p in (50, 70, 95, 100)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--requests", type=int, default=25)
    args = parser.parse_args()

    app = api.app
    manager = app.router.lifespan_context(app)
    await manager.__aenter__()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=120) as client:
            for concurrency in args.levels:
                semaphore = asyncio.Semaphore(concurrency)

                async def one(request_number: int) -> float:
                    async with semaphore:
                        started = time.perf_counter()
                        response = await client.post(
                            "/query/text",
                            json={"query": f"what is retrieval augmented generation load{concurrency}-{request_number}"},
                        )
                        elapsed = (time.perf_counter() - started) * 1000
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                        return elapsed

                values = await asyncio.gather(*(one(request_number) for request_number in range(args.requests)))
                metrics = percentiles(values)
                print(
                    f"concurrency={concurrency} requests={len(values)} "
                    + " ".join(f"{key.upper()}={value:.1f}ms" for key, value in metrics.items())
                )
    finally:
        await manager.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
