from __future__ import annotations

import math
import operator

import numpy as np

from ._lib import addr, f64, lib


def time_delay_embedding(time_series, embedding_dimension, delay):
    series_length = len(time_series)
    embedded_series = np.empty(
        (
            embedding_dimension,
            series_length - (embedding_dimension - 1) * delay,
        )
    )
    for i in range(embedding_dimension):
        embedded_series[i] = time_series[
            i * delay : i * delay + embedded_series.shape[1]
        ]
    return embedded_series.T


def util_pattern_space(time_series, lag, dim):
    n = len(time_series)
    if lag < 1:
        raise ValueError("Lag should be greater than or equal to 1.")
    if lag * dim > n:
        raise ValueError(
            "Result matrix size limit exceeded. Adjust the lag or dim value."
        )
    pattern_space = np.zeros((n - lag * (dim - 1), dim))
    for i in range(dim):
        pattern_space[:, i] = time_series[
            i * lag : i * lag + n - lag * (dim - 1)
        ]
    return pattern_space


def util_granulate_time_series(time_series, scale):
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)
    n = time_series.shape[0]
    b = n // scale
    return np.mean(time_series[: b * scale].reshape(b, scale), axis=1)


def _series_1d(time_series) -> np.ndarray:
    array = f64(time_series)
    if array.ndim != 1:
        raise ValueError("time_series must be one-dimensional.")
    return array


def sample_entropy(time_series, sample_length, tolerance=None):
    array = _series_1d(time_series)
    sample_length = operator.index(sample_length)
    if sample_length < 0:
        raise ValueError("negative dimensions are not allowed")
    if tolerance is None:
        tolerance = 0.1 * np.std(array)

    counts = np.zeros(sample_length + 1, dtype=np.int64)
    lib().mpy_sample_counts(
        addr(array),
        array.size,
        sample_length,
        float(tolerance),
        addr(counts),
    )
    return -np.log(counts[1:] / counts[:-1])


def multiscale_entropy(
    time_series, sample_length, tolerance=None, maxscale=None
):
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)
    if maxscale is None:
        maxscale = len(time_series)
    mse = np.zeros(maxscale)
    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i + 1)
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
    return mse


def _permutation_workspace(n: int, order: int, delay: int):
    windows = n - (order - 1) * delay
    if windows <= 0:
        raise ValueError("The signal is too short for the given order and delay.")
    if order > 20:
        raise ValueError("order must be 20 or less.")
    motifs = min(windows, math.factorial(order))
    capacity = 1
    while capacity < 2 * motifs:
        capacity *= 2
    keys = np.empty(capacity, dtype=np.int64)
    work = np.empty(order, dtype=np.int64)
    status = np.zeros(1, dtype=np.int64)
    return windows, capacity, keys, work, status


def _permutation_reference(array, order, delay):
    hashmult = np.power(order, np.arange(order))
    sorted_idx = time_delay_embedding(
        array, embedding_dimension=order, delay=delay
    ).argsort(kind="quicksort")
    hashval = np.multiply(sorted_idx, hashmult).sum(1)
    _, counts = np.unique(hashval, return_counts=True)
    probabilities = np.true_divide(counts, counts.sum())
    return -np.multiply(probabilities, np.log2(probabilities)).sum()


def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    array = _series_1d(time_series)
    order = operator.index(order)
    delay = operator.index(delay)
    if order < 1:
        raise ValueError("order must be at least 1.")
    if delay < 0:
        raise ValueError("delay must be non-negative.")

    _, capacity, keys, work, status = _permutation_workspace(
        array.size, order, delay
    )
    counts = np.empty(capacity, dtype=np.int64)
    used = lib().mpy_permutation_entropy(
        addr(array),
        array.size,
        order,
        delay,
        addr(keys),
        addr(counts),
        addr(work),
        capacity,
        addr(status),
    )
    if used < 0 or used > capacity or status[0] not in (0, 1):
        raise RuntimeError("native permutation kernel returned invalid output.")
    if status[0]:
        value = _permutation_reference(array, order, delay)
    else:
        probabilities = np.true_divide(
            counts[:used], counts[:used].sum()
        )
        value = -np.multiply(
            probabilities, np.log2(probabilities)
        ).sum()
    if normalize:
        value /= np.log2(math.factorial(order))
    return np.float64(value)


def multiscale_permutation_entropy(time_series, m, delay, scale):
    mspe = np.empty(scale)
    for i in range(scale):
        coarse_time_series = util_granulate_time_series(time_series, i + 1)
        mspe[i] = permutation_entropy(
            coarse_time_series, order=m, delay=delay
        )
    return mspe


def _weighted_permutation_reference(array, order, delay):
    embedded = time_delay_embedding(
        array, embedding_dimension=order, delay=delay
    )
    weights = np.var(embedded, axis=1)
    sorted_idx = embedded.argsort(kind="quicksort", axis=1)
    motif_weights = {}
    for weight, indices in zip(weights, sorted_idx):
        motif = tuple(indices)
        motif_weights[motif] = motif_weights.get(motif, 0.0) + weight
    probabilities = np.array(list(motif_weights.values()))
    probabilities /= weights.sum()
    return -np.dot(probabilities, np.log2(probabilities))


def weighted_permutation_entropy(
    time_series, order=2, delay=1, normalize=False
):
    array = _series_1d(time_series)
    order = operator.index(order)
    delay = operator.index(delay)
    if order < 1:
        raise ValueError("order must be at least 1.")
    if delay < 0:
        raise ValueError("delay must be non-negative.")

    _, capacity, keys, work, status = _permutation_workspace(
        array.size, order, delay
    )
    weights = np.empty(capacity, dtype=np.float64)
    used = lib().mpy_weighted_permutation_entropy(
        addr(array),
        array.size,
        order,
        delay,
        addr(keys),
        addr(weights),
        addr(work),
        capacity,
        addr(status),
    )
    if used < 0 or used > capacity or status[0] not in (0, 1):
        raise RuntimeError(
            "native weighted permutation kernel returned invalid output."
        )
    if status[0]:
        value = _weighted_permutation_reference(array, order, delay)
    else:
        probabilities = weights[:used]
        probabilities /= probabilities.sum()
        value = -np.dot(probabilities, np.log2(probabilities))
    if normalize:
        value /= np.log2(math.factorial(order))
    return np.float64(value)


def composite_multiscale_entropy(
    time_series, sample_length, scale, tolerance=None
):
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)
    cmse = np.zeros(scale)
    for i in range(scale):
        for j in range(i + 1):
            temp = util_granulate_time_series(time_series[j:], i + 1)
            value = sample_entropy(temp, sample_length, tolerance)[-1]
            cmse[i] += value / (i + 1)
    return cmse


__all__ = [
    "sample_entropy",
    "multiscale_entropy",
    "composite_multiscale_entropy",
    "permutation_entropy",
    "multiscale_permutation_entropy",
    "weighted_permutation_entropy",
    "time_delay_embedding",
    "util_pattern_space",
    "util_granulate_time_series",
]
