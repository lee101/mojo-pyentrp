from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-pyentrp.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpy_sample_counts": ([I, I, I, F, I], None),
    "mpy_permutation_entropy": ([I] * 9, I),
    "mpy_weighted_permutation_entropy": ([I] * 9, I),
}

_lib: ctypes.CDLL | None = None


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB_PATH):
            build()
        _lib = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty array.")
    return address


def f64(array) -> np.ndarray:
    source = np.asarray(array)
    if not (
        np.issubdtype(source.dtype, np.number)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise TypeError("time_series must contain real numeric values.")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex-valued time series are not supported.")

    result = np.ascontiguousarray(source, dtype=np.float64)
    # Float64 is the native ABI type.  Refuse conversions which can silently
    # merge distinct integer samples and therefore change ordinal motifs.
    if np.issubdtype(source.dtype, np.integer):
        converted_back = result.astype(source.dtype)
        if not np.array_equal(converted_back, source):
            raise ValueError(
                "integer samples must be exactly representable as float64."
            )
    elif (
        np.issubdtype(source.dtype, np.floating)
        and source.dtype.itemsize > np.dtype(np.float64).itemsize
        and not np.array_equal(
            result.astype(source.dtype), source, equal_nan=True
        )
    ):
        raise ValueError(
            "floating-point samples must be exactly representable as float64."
        )
    return result
