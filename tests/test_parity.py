import inspect

import numpy as np
import pytest
from pyentrp import entropy as upstream

from mojo_pyentrp import entropy as mojo


SAMPLE_VECTOR = np.array(
    [
        1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3, 9, 1,
        2, 3, 4, 2, 9, 6, 7, 4, 9, 2, 9, 9, 6, 5, 1, 3, 8, 1, 5, 3,
        8, 4, 1, 2, 2, 1, 6, 5, 3, 6, 5, 4, 8, 9, 6, 7, 5, 3, 2, 5,
        4, 2, 5, 1, 6, 5, 3, 5, 6, 7, 8, 5, 2, 8, 6, 3, 8, 2, 7, 1,
        7, 3, 5, 6, 2, 1, 3, 7, 3, 5, 3, 7, 6, 7, 7, 2, 3, 1, 7, 8,
    ],
    dtype=np.float64,
)
BANDT_VECTOR = np.array([4, 7, 9, 10, 6, 11, 3], dtype=np.float64)


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(1234567)
    random = rng.normal(size=900)
    periodic = np.sin(np.linspace(0, 24 * np.pi, 900))
    tied = rng.integers(0, 7, size=900)
    return random, periodic, tied


@pytest.mark.parametrize(
    "sample_length,tolerance",
    [(1, None), (2, None), (4, None), (3, 0.15), (5, 0.4)],
)
def test_sample_entropy_random_parity(signals, sample_length, tolerance):
    expected = upstream.sample_entropy(
        signals[0], sample_length, tolerance
    )
    actual = mojo.sample_entropy(signals[0], sample_length, tolerance)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("index", [1, 2])
def test_sample_entropy_structured_parity(signals, index):
    tolerance = 0.2 * np.std(signals[index])
    expected = upstream.sample_entropy(signals[index], 4, tolerance)
    actual = mojo.sample_entropy(signals[index], 4, tolerance)
    assert np.allclose(actual, expected, equal_nan=True)


def test_sample_entropy_upstream_vector():
    tolerance = 0.2 * np.std(SAMPLE_VECTOR)
    expected = np.array(
        [2.26881823, 2.11119024, 2.33537492, 1.79175947]
    )
    assert np.allclose(
        mojo.sample_entropy(SAMPLE_VECTOR, 4, tolerance), expected
    )


@pytest.mark.parametrize(
    "order,delay,normalize",
    [
        (2, 1, False),
        (3, 1, True),
        (4, 3, False),
        (5, 2, True),
        (7, 1, False),
    ],
)
def test_permutation_entropy_parity(
    signals, order, delay, normalize
):
    actual = mojo.permutation_entropy(
        signals[0], order=order, delay=delay, normalize=normalize
    )
    expected = upstream.permutation_entropy(
        signals[0], order=order, delay=delay, normalize=normalize
    )
    assert actual == pytest.approx(expected, abs=5e-12)


@pytest.mark.parametrize("order", [3, 4, 5, 7])
def test_permutation_entropy_tie_parity(signals, order):
    actual = mojo.permutation_entropy(signals[2], order=order)
    expected = upstream.permutation_entropy(signals[2], order=order)
    assert actual == expected


@pytest.mark.parametrize("order,delay", [(1, 1), (3, 0)])
def test_permutation_entropy_upstream_edge_parity(order, delay):
    x = np.arange(20, dtype=np.float64)
    assert mojo.permutation_entropy(
        x, order=order, delay=delay
    ) == upstream.permutation_entropy(x, order=order, delay=delay)


@pytest.mark.parametrize("size,order,delay", [(37, 5, 2), (42, 7, 1)])
def test_permutation_entropy_simd_tail_parity(size, order, delay):
    rng = np.random.default_rng(size)
    x = rng.normal(size=size)
    actual = mojo.permutation_entropy(x, order=order, delay=delay)
    expected = upstream.permutation_entropy(x, order=order, delay=delay)
    assert actual == pytest.approx(expected, abs=5e-12)


def test_permutation_entropy_high_order_scalar_parity():
    rng = np.random.default_rng(12)
    x = rng.normal(size=31)
    actual = mojo.permutation_entropy(x, order=12, delay=1)
    expected = upstream.permutation_entropy(x, order=12, delay=1)
    assert actual == pytest.approx(expected, abs=5e-12)


def test_permutation_entropy_max_supported_order():
    rng = np.random.default_rng(20)
    x = rng.normal(size=45)
    actual = mojo.permutation_entropy(x, order=20, delay=1)
    expected = upstream.permutation_entropy(x, order=20, delay=1)
    assert actual == pytest.approx(expected, abs=5e-12)


@pytest.mark.parametrize("order,expected_capacity", [(3, 16), (5, 256)])
def test_permutation_workspace_uses_motif_bound(order, expected_capacity):
    _, capacity, *_ = mojo._permutation_workspace(1_000_000, order, 1)
    assert capacity == expected_capacity


@pytest.mark.parametrize(
    "order,delay,normalize",
    [(2, 1, False), (3, 2, True), (4, 1, False), (6, 3, True)],
)
def test_weighted_permutation_entropy_parity(
    signals, order, delay, normalize
):
    actual = mojo.weighted_permutation_entropy(
        signals[0], order=order, delay=delay, normalize=normalize
    )
    expected = upstream.weighted_permutation_entropy(
        signals[0], order=order, delay=delay, normalize=normalize
    )
    assert actual == pytest.approx(expected, abs=5e-12)


def test_weighted_permutation_entropy_tie_parity(signals):
    for order in (3, 4, 5):
        actual = mojo.weighted_permutation_entropy(signals[2], order=order)
        expected = upstream.weighted_permutation_entropy(
            signals[2], order=order
        )
        assert actual == expected


def test_bandt_published_examples():
    assert mojo.permutation_entropy(BANDT_VECTOR, order=2) == pytest.approx(
        0.918, rel=1e-3
    )
    assert mojo.permutation_entropy(BANDT_VECTOR, order=3) == pytest.approx(
        1.522, rel=1e-3
    )
    assert mojo.weighted_permutation_entropy(
        BANDT_VECTOR, order=3
    ) == pytest.approx(1.414, rel=1e-3)


def test_multiscale_entropy_parity():
    rng = np.random.default_rng(44)
    x = rng.random(1000)
    actual = mojo.multiscale_entropy(x, 4, maxscale=4)
    expected = upstream.multiscale_entropy(x, 4, maxscale=4)
    assert np.allclose(actual, expected, equal_nan=True)


def test_multiscale_permutation_entropy_parity():
    actual = mojo.multiscale_permutation_entropy(
        SAMPLE_VECTOR, 3, 5, 3
    )
    expected = upstream.multiscale_permutation_entropy(
        SAMPLE_VECTOR, 3, 5, 3
    )
    assert np.allclose(actual, expected, atol=5e-12)


def test_composite_multiscale_entropy_parity():
    signal = np.cos(np.linspace(start=0, stop=30, num=100))
    actual = mojo.composite_multiscale_entropy(
        signal, sample_length=3, scale=3
    )
    expected = upstream.composite_multiscale_entropy(
        signal, sample_length=3, scale=3
    )
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "name,args",
    [
        ("time_delay_embedding", ([1, 2, 3, 4, 5], 3, 1)),
        ("util_pattern_space", ([1, 1, 1, 2, 3, 4, 5], 2, 3)),
        ("util_granulate_time_series", (np.arange(12), 3)),
    ],
)
def test_utility_parity(name, args):
    actual = getattr(mojo, name)(*args)
    expected = getattr(upstream, name)(*args)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "name",
    [
        "sample_entropy",
        "multiscale_entropy",
        "composite_multiscale_entropy",
        "permutation_entropy",
        "multiscale_permutation_entropy",
        "weighted_permutation_entropy",
        "time_delay_embedding",
        "util_pattern_space",
        "util_granulate_time_series",
    ],
)
def test_signature_parity(name):
    assert inspect.signature(getattr(mojo, name)) == inspect.signature(
        getattr(upstream, name)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: mojo.sample_entropy(np.ones((3, 3)), 2),
        lambda: mojo.permutation_entropy([1, 2, 3], order=4),
        lambda: mojo.permutation_entropy([1, 2, 3], delay=-1),
        lambda: mojo.weighted_permutation_entropy([1, 2, 3], order=21),
    ],
)
def test_validation(call):
    with pytest.raises(ValueError):
        call()


def test_strided_and_float32_input_parity():
    x = np.linspace(-2, 2, 301, dtype=np.float32)[::3]
    assert mojo.permutation_entropy(x, order=4) == pytest.approx(
        upstream.permutation_entropy(x, order=4), abs=5e-12
    )
    assert np.allclose(
        mojo.sample_entropy(x, 3), upstream.sample_entropy(x, 3)
    )


def test_complex_input_is_not_silently_narrowed():
    with pytest.raises(TypeError, match="complex"):
        mojo.permutation_entropy(np.array([1 + 2j, 2 + 0j, 3 + 0j]))


def test_inexact_large_integer_input_is_not_silently_narrowed():
    x = np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64)
    with pytest.raises(ValueError, match="representable"):
        mojo.permutation_entropy(x)


def test_non_numeric_input_is_rejected():
    with pytest.raises(TypeError, match="numeric"):
        mojo.sample_entropy(["1", "2", "3"], 2)
