# mojo-pyentrp

The sample- and permutation-entropy family from
[pyentrp](https://github.com/nikdon/pyEntropy), implemented as native
[Mojo](https://www.modular.com/mojo) kernels behind a NumPy-friendly Python
API.

The covered functions keep pyentrp 2.1.0's public names and signatures, so
only the import changes:

```python
import numpy as np
from mojo_pyentrp import entropy as ent

rng = np.random.default_rng(42)
signal = rng.normal(size=10_000)

sample = ent.sample_entropy(signal, sample_length=4)
permutation = ent.permutation_entropy(
    signal, order=3, delay=1, normalize=True
)
print(sample)
print(permutation)
```

## Coverage

| family | functions |
| --- | --- |
| Sample entropy | `sample_entropy`, `multiscale_entropy`, `composite_multiscale_entropy` |
| Permutation entropy | `permutation_entropy`, `multiscale_permutation_entropy`, `weighted_permutation_entropy` |
| Supporting utilities | `time_delay_embedding`, `util_pattern_space`, `util_granulate_time_series` |

This covers every function in `pyentrp.entropy` except `shannon_entropy`.
That omission is intentional: NumPy's unique/count reduction already performs
its compute-heavy work, so wrapping that thin operation in Mojo would not be a
useful port. This project does not cover other pyentrp modules, multidimensional
signals, a GPU backend, or non-NumPy array protocols.

Inputs to the native entropy paths are one-dimensional, real numeric arrays.
They are converted to contiguous `float64`; complex values and integers that
cannot be represented exactly are rejected instead of silently narrowed.
Permutation order is supported through 20, the largest factoradic rank that
fits in the signed 64-bit scratch representation. The multiscale functions
retain upstream's coarse-graining and tolerance conventions.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo compilation unit into
`dist/libmojo-pyentrp.so`. The Python wrapper also builds the library on first
import if it is absent. This is a source-checkout install: Mojo and Linux
x86-64 are required, and no prebuilt wheel is currently published.

The 54-test suite compares every covered public function and signature against
the real `pyentrp` 2.1.0 package. It includes random, periodic, tied, explicit
tolerance, multiscale, SIMD-tail, high-order fallback, upstream
published-vector, dtype/stride, and error cases.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(Linux x86-64). Each implementation is warmed first and the table reports the
best of three runs on identical contiguous float64 data.

| measure | mojo-pyentrp | pyentrp 2.1.0 | result |
| --- | ---: | ---: | ---: |
| `sample_entropy` random m=4 (6k) | 40.56 ms | 641.61 ms | 15.82x faster |
| `sample_entropy` periodic m=4 (4k) | 15.48 ms | 393.39 ms | 25.41x faster |
| `permutation_entropy` order=3 (1M) | 4.96 ms | 145.50 ms | 29.36x faster |
| `permutation_entropy` order=5 (1M) | 21.62 ms | 1318.99 ms | 61.00x faster |
| `weighted_permutation_entropy` order=3 (200k) | 6.19 ms | 386.35 ms | 62.45x faster |

The largest gain is weighted permutation entropy: pyentrp groups every motif
through a Python dictionary, while Mojo accumulates weights in a native
open-addressed table. Sample entropy benefits from a fused pairwise prefix
scan that avoids repeatedly allocating NumPy search and hit arrays.

No GPU path is provided. These kernels have effective arithmetic intensity
below roughly two floating-point operations per byte and make irregular
histogram updates; transfer and launch overhead would dominate. The benchmarked
CPU implementation is already faster than upstream, so this port stays
deliberately small and CPU-only.

## How it works

Python validates inputs and makes one C-contiguous float64 view. It owns the
result and all scratch arrays, checks non-empty buffers for null addresses, and
keeps every NumPy owner alive for the duration of the synchronous call.
ctypes declares every argument and return type, then passes addresses as
64-bit integers; the exported `abi("C")` functions reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside Mojo. No allocation or
ownership crosses the FFI boundary. Python also rejects impossible native
result lengths and status values before slicing output buffers.

Sample entropy writes the exact prefix-match counts used by pyentrp into an
int64 buffer. The hot permutation counter compares a SIMD-width batch of
independent windows and encodes their pairwise ordering, with a scalar tail.
Its caller-owned hash table is bounded by the smaller of the window count and
the factorial motif count instead of by the full input size. Weighted and
high-order kernels retain the general factoradic rank. Mojo compacts the small
final count/weight set and NumPy performs the last `log2` reduction,
preserving upstream rounding.

pyentrp deliberately uses NumPy's unstable quicksort to order equal values.
The native path detects tied or NaN-containing windows and sends those rare
cases through a faithful NumPy fallback so its tie behavior remains exact.
Ordinary continuous signals stay entirely on the native path.

## License

MIT
