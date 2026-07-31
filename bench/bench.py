from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_pyentrp import entropy as mojo  # noqa: E402
from pyentrp import entropy as upstream  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def benchmark(name, ours, theirs, repeat=3):
    ours()
    theirs()
    mojo_time = timeit(ours, repeat)
    upstream_time = timeit(theirs, repeat)
    return name, mojo_time, upstream_time


def main():
    rng = np.random.default_rng(42)
    sample_data = np.ascontiguousarray(rng.normal(size=6_000))
    periodic_data = np.ascontiguousarray(
        np.sin(np.linspace(0, 500 * np.pi, 4_000))
    )
    permutation_data = np.ascontiguousarray(rng.normal(size=1_000_000))
    weighted_data = np.ascontiguousarray(rng.normal(size=200_000))

    sample_tolerance = 0.1 * np.std(sample_data)
    periodic_tolerance = 0.1 * np.std(periodic_data)
    cases = [
        (
            "sample_entropy random m=4 (6k)",
            lambda: mojo.sample_entropy(
                sample_data, 4, sample_tolerance
            ),
            lambda: upstream.sample_entropy(
                sample_data, 4, sample_tolerance
            ),
            3,
        ),
        (
            "sample_entropy periodic m=4 (4k)",
            lambda: mojo.sample_entropy(
                periodic_data, 4, periodic_tolerance
            ),
            lambda: upstream.sample_entropy(
                periodic_data, 4, periodic_tolerance
            ),
            3,
        ),
        (
            "permutation_entropy order=3 (1M)",
            lambda: mojo.permutation_entropy(permutation_data, order=3),
            lambda: upstream.permutation_entropy(
                permutation_data, order=3
            ),
            3,
        ),
        (
            "permutation_entropy order=5 (1M)",
            lambda: mojo.permutation_entropy(permutation_data, order=5),
            lambda: upstream.permutation_entropy(
                permutation_data, order=5
            ),
            3,
        ),
        (
            "weighted_permutation_entropy order=3 (200k)",
            lambda: mojo.weighted_permutation_entropy(
                weighted_data, order=3
            ),
            lambda: upstream.weighted_permutation_entropy(
                weighted_data, order=3
            ),
            3,
        ),
    ]

    results = [benchmark(*case) for case in cases]
    print(
        f"Machine: {cpu_name()} "
        f"({platform.system()} {platform.machine()})"
    )
    print()
    print("| measure | mojo-pyentrp | pyentrp 2.1.0 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, upstream_time in results:
        ratio = upstream_time / mojo_time
        label = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
