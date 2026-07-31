from std.math import abs, isnan
from std.memory import UnsafePointer
from std.sys.info import simd_width_of


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def sample_counts(
    x: FPtr,
    n: Int,
    sample_length: Int,
    tolerance: Float64,
    counts: IPtr,
):
    for i in range(sample_length + 1):
        counts[i] = 0
    counts[0] = Int64(n) * Int64(n - 1) // 2

    # pyentrp compares every prefix over the set of starts that can hold the
    # longest requested template, rather than changing that set per prefix.
    var starts = n - sample_length + 1
    for i in range(starts - 1):
        for j in range(i + 1, starts):
            for length in range(sample_length):
                if abs(x[i + length] - x[j + length]) >= tolerance:
                    break
                counts[length + 1] += 1


def ordinal_rank(
    x: FPtr,
    start: Int,
    order: Int,
    delay: Int,
    work: IPtr,
    status: IPtr,
) -> Int64:
    for i in range(order):
        work[i] = Int64(i)

    for i in range(1, order):
        var j = i
        while j > 0:
            var left = Int(work[j - 1])
            var right = Int(work[j])
            var left_value = x[start + left * delay]
            var right_value = x[start + right * delay]
            if isnan(left_value) or isnan(right_value) or left_value == right_value:
                status[0] = 1
            var ordered = False
            if isnan(right_value):
                ordered = True
            elif not isnan(left_value) and left_value <= right_value:
                ordered = True
            if ordered:
                break
            work[j - 1] = Int64(right)
            work[j] = Int64(left)
            j -= 1

    var rank = Int64(0)
    for i in range(order):
        var smaller = Int64(0)
        for j in range(i + 1, order):
            if work[j] < work[i]:
                smaller += 1
        rank = rank * Int64(order - i) + smaller
    return rank


def pairwise_rank(
    x: FPtr,
    start: Int,
    order: Int,
    delay: Int,
    status: IPtr,
) -> Int64:
    var rank = Int64(0)
    var bit = Int64(1)
    for i in range(order - 1):
        var left_value = x[start + i * delay]
        for j in range(i + 1, order):
            var right_value = x[start + j * delay]
            if (
                isnan(left_value)
                or isnan(right_value)
                or left_value == right_value
            ):
                status[0] = 1
            if left_value > right_value:
                rank |= bit
            bit <<= 1
    return rank


def add_count(rank: Int64, keys: IPtr, counts: IPtr, capacity: Int):
    var slot = Int(rank) & (capacity - 1)
    while keys[slot] != -1 and keys[slot] != rank:
        slot = (slot + 1) & (capacity - 1)
    if keys[slot] == -1:
        keys[slot] = rank
    counts[slot] += 1


def permutation_counts(
    x: FPtr,
    n: Int,
    order: Int,
    delay: Int,
    keys: IPtr,
    counts: IPtr,
    work: IPtr,
    capacity: Int,
    status: IPtr,
) -> Int:
    for i in range(capacity):
        keys[i] = -1
        counts[i] = 0
    status[0] = 0

    var windows = n - (order - 1) * delay
    if order <= 11:
        comptime W = simd_width_of[DType.float64]()
        var start = 0
        while start + W <= windows:
            var ranks = SIMD[DType.int64, W](0)
            var bit = Int64(1)
            for i in range(order - 1):
                var left_values = x.load[width=W](start + i * delay)
                for j in range(i + 1, order):
                    var right_values = x.load[width=W](
                        start + j * delay
                    )
                    var invalid = (
                        left_values.ne(left_values)
                        | right_values.ne(right_values)
                        | left_values.eq(right_values)
                    )
                    if invalid.cast[DType.int64]().reduce_add() != 0:
                        status[0] = 1
                    ranks = left_values.gt(right_values).select(
                        ranks | SIMD[DType.int64, W](bit), ranks
                    )
                    bit <<= 1
            for lane in range(W):
                add_count(ranks[lane], keys, counts, capacity)
            start += W
        while start < windows:
            add_count(
                pairwise_rank(x, start, order, delay, status),
                keys,
                counts,
                capacity,
            )
            start += 1
    else:
        for start in range(windows):
            add_count(
                ordinal_rank(x, start, order, delay, work, status),
                keys,
                counts,
                capacity,
            )

    var used = 0
    for i in range(capacity):
        if counts[i] > 0:
            counts[used] = counts[i]
            used += 1
    return used


def window_variance(
    x: FPtr, start: Int, order: Int, delay: Int
) -> Float64:
    var mean = Float64(0)
    for i in range(order):
        mean += x[start + i * delay]
    mean /= Float64(order)

    var moment = Float64(0)
    for i in range(order):
        var delta = x[start + i * delay] - mean
        moment += delta * delta
    return moment / Float64(order)


def weighted_permutation_weights(
    x: FPtr,
    n: Int,
    order: Int,
    delay: Int,
    keys: IPtr,
    weights: FPtr,
    work: IPtr,
    capacity: Int,
    status: IPtr,
) -> Int:
    for i in range(capacity):
        keys[i] = -1
        weights[i] = 0.0
    status[0] = 0

    var windows = n - (order - 1) * delay
    for start in range(windows):
        var rank = ordinal_rank(x, start, order, delay, work, status)
        var weight = window_variance(x, start, order, delay)
        var slot = Int(rank) & (capacity - 1)
        while keys[slot] != -1 and keys[slot] != rank:
            slot = (slot + 1) & (capacity - 1)
        if keys[slot] == -1:
            keys[slot] = rank
        weights[slot] += weight

    var used = 0
    for i in range(capacity):
        if keys[i] != -1:
            weights[used] = weights[i]
            used += 1
    return used


@export("mpy_sample_counts")
def mpy_sample_counts(
    x_addr: Int,
    n: Int,
    sample_length: Int,
    tolerance: Float64,
    counts_addr: Int,
) abi("C"):
    sample_counts(fp(x_addr), n, sample_length, tolerance, ip(counts_addr))


@export("mpy_permutation_entropy")
def mpy_permutation_entropy(
    x_addr: Int,
    n: Int,
    order: Int,
    delay: Int,
    keys_addr: Int,
    counts_addr: Int,
    work_addr: Int,
    capacity: Int,
    status_addr: Int,
) abi("C") -> Int:
    return permutation_counts(
        fp(x_addr),
        n,
        order,
        delay,
        ip(keys_addr),
        ip(counts_addr),
        ip(work_addr),
        capacity,
        ip(status_addr),
    )


@export("mpy_weighted_permutation_entropy")
def mpy_weighted_permutation_entropy(
    x_addr: Int,
    n: Int,
    order: Int,
    delay: Int,
    keys_addr: Int,
    weights_addr: Int,
    work_addr: Int,
    capacity: Int,
    status_addr: Int,
) abi("C") -> Int:
    return weighted_permutation_weights(
        fp(x_addr),
        n,
        order,
        delay,
        ip(keys_addr),
        fp(weights_addr),
        ip(work_addr),
        capacity,
        ip(status_addr),
    )
