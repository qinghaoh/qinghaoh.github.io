"""Exact computations for the Ziyuan section of the SGS probability post."""

from collections import defaultdict
from functools import lru_cache
from fractions import Fraction
from math import comb, factorial
import sys


TARGET = 13
MAX_VALUE = 13
FULL_MASK = (1 << (TARGET + 1)) - 1
TARGET_BIT = 1 << TARGET


def mask_step(mask: int, value: int) -> int:
    return mask | ((mask << value) & FULL_MASK)


def reachability_rows(limit: int):
    dp = {1: 1}
    rows = [(0, 0.0, 0.0)]
    previous = 0.0
    for n in range(1, limit + 1):
        nxt = defaultdict(int)
        for mask, count in dp.items():
            for value in range(1, MAX_VALUE + 1):
                nxt[mask_step(mask, value)] += count
        dp = nxt
        success = sum(count for mask, count in dp.items() if mask & TARGET_BIT)
        p = success / MAX_VALUE**n
        hazard = (p - previous) / (1 - previous)
        rows.append((n, p, hazard))
        previous = p
    return rows


@lru_cache(None)
def hitting_moments(mask: int):
    """Exact first and second moments of additional draws to reach TARGET."""
    if mask & TARGET_BIT:
        return Fraction(0), Fraction(0)

    self_loops = 0
    strict = []
    for value in range(1, MAX_VALUE + 1):
        nxt = mask_step(mask, value)
        if nxt == mask:
            self_loops += 1
        else:
            strict.append(nxt)

    child = [hitting_moments(nxt) for nxt in strict]
    denominator = MAX_VALUE - self_loops
    mean = (MAX_VALUE + sum((item[0] for item in child), Fraction(0))) / denominator
    second = (
        MAX_VALUE * (2 * mean - 1)
        + sum((item[1] for item in child), Fraction(0))
    ) / denominator
    return mean, second


def capped_solution_rows(limit: int):
    """Distribution of N=0, N=1, N>=2, with every count capped at two."""
    initial = (1,) + (0,) * TARGET
    dp = {initial: 1}
    rows = []
    for n in range(1, limit + 1):
        nxt_dp = defaultdict(int)
        for state, weight in dp.items():
            for value in range(1, MAX_VALUE + 1):
                nxt = list(state)
                for total in range(value, TARGET + 1):
                    nxt[total] = min(2, state[total] + state[total - value])
                nxt_dp[tuple(nxt)] += weight
        dp = nxt_dp
        denominator = MAX_VALUE**n
        probabilities = [
            sum(weight for state, weight in dp.items() if state[TARGET] == category)
            / denominator
            for category in range(3)
        ]
        rows.append((n, *probabilities, len(dp)))
    return rows


def cardinality_state_rows(limit: int, selected: set[int], verbose: bool = False):
    """Track attainable subset cardinalities for every sum."""
    initial = (1,) + (0,) * TARGET  # bit zero: empty subset reaches sum zero
    dp = {initial: 1}
    rows = {}
    variance_min = []
    for n in range(1, limit + 1):
        nxt_dp = defaultdict(int)
        for state, weight in dp.items():
            for value in range(1, MAX_VALUE + 1):
                nxt = list(state)
                for total in range(value, TARGET + 1):
                    nxt[total] = state[total] | (state[total - value] << 1)
                nxt_dp[tuple(nxt)] += weight
        dp = nxt_dp
        denominator = MAX_VALUE**n
        joint = defaultdict(int)
        first = second = 0
        for state, weight in dp.items():
            bits = state[TARGET]
            if bits:
                minimum = (bits & -bits).bit_length() - 1
                maximum = bits.bit_length() - 1
            else:
                minimum = maximum = 0
            joint[(minimum, maximum)] += weight
            first += minimum * weight
            second += minimum * minimum * weight
        mean = first / denominator
        variance = second / denominator - mean * mean
        variance_min.append((n, mean, variance, len(dp)))
        if verbose:
            print("minvar", n, mean, variance, len(dp), flush=True)
        if n in selected:
            rows[n] = (joint, denominator)
            if verbose:
                for label, selector in (
                    ("min", lambda pair: pair[0]),
                    ("max", lambda pair: pair[1]),
                    ("gap", lambda pair: pair[1] - pair[0]),
                ):
                    distribution = defaultdict(int)
                    for pair, weight in joint.items():
                        distribution[selector(pair)] += weight
                    values = [
                        (value, weight / denominator)
                        for value, weight in sorted(distribution.items())
                    ]
                    print("distribution", n, label, values, flush=True)
    return rows, variance_min


def extremum_rows(limit: int, mode: str):
    """Track only a minimum or maximum cardinality, allowing larger n."""
    unreachable = 14 if mode == "min" else -1
    initial = (0,) + (unreachable,) * TARGET
    dp = {initial: 1}
    rows = []
    for n in range(1, limit + 1):
        nxt_dp = defaultdict(int)
        for state, weight in dp.items():
            for value in range(1, MAX_VALUE + 1):
                nxt = list(state)
                for total in range(value, TARGET + 1):
                    previous = state[total - value]
                    if previous == unreachable:
                        continue
                    candidate = previous + 1
                    if nxt[total] == unreachable:
                        nxt[total] = candidate
                    elif mode == "min":
                        nxt[total] = min(nxt[total], candidate)
                    else:
                        nxt[total] = max(nxt[total], candidate)
                nxt_dp[tuple(nxt)] += weight
        dp = nxt_dp
        denominator = MAX_VALUE**n
        moments = [0, 0]
        distribution = defaultdict(int)
        for state, weight in dp.items():
            value = state[TARGET] if state[TARGET] != unreachable else 0
            distribution[value] += weight
            moments[0] += value * weight
            moments[1] += value * value * weight
        mean = moments[0] / denominator
        variance = moments[1] / denominator - mean * mean
        rows.append((n, mean, variance, len(dp), distribution, denominator))
        print("extremum", mode, n, mean, variance, len(dp), flush=True)
    return rows


def solution_count_distribution(n: int):
    """Enumerate value multiplicities; return exact weighted statistics."""
    histogram = defaultdict(int)
    size_profile = [0] * (TARGET + 1)
    overlap_total = Fraction(0)
    nonempty_weight = 0
    counterexample = None
    n_factorial = factorial(n)

    def visit(index: int, remaining: int, counts: list[int]):
        nonlocal overlap_total, nonempty_weight, counterexample
        if index == MAX_VALUE:
            counts.append(remaining)
            process(counts)
            counts.pop()
            return
        for count in range(remaining + 1):
            counts.append(count)
            visit(index + 1, remaining - count, counts)
            counts.pop()

    def process(counts: list[int]):
        nonlocal overlap_total, nonempty_weight, counterexample
        weight = n_factorial
        for count in counts:
            weight //= factorial(count)

        # poly[sum][size] counts indexed subsets.
        poly = [[0] * (TARGET + 1) for _ in range(TARGET + 1)]
        poly[0][0] = 1
        for value, count in enumerate(counts, 1):
            for _ in range(count):
                old = [row[:] for row in poly]
                for total in range(value, TARGET + 1):
                    for size in range(1, TARGET + 1):
                        poly[total][size] += old[total - value][size - 1]
        profile = poly[TARGET]
        number = sum(profile)
        histogram[number] += weight
        for size, count in enumerate(profile):
            size_profile[size] += weight * count

        positive = [profile[k] for k in range(1, TARGET + 1)]
        if counterexample is None:
            for k in range(1, TARGET - 1):
                if positive[k] * positive[k] < positive[k - 1] * positive[k + 1]:
                    counterexample = (tuple(counts), tuple(profile))
                    break

        if number:
            nonempty_weight += weight
            overlap_numerator = 0
            for value, count in enumerate(counts, 1):
                if not count:
                    continue
                reduced = counts[:]
                reduced[value - 1] -= 1
                ways = [0] * (TARGET + 1)
                ways[0] = 1
                for other_value, other_count in enumerate(reduced, 1):
                    for _ in range(other_count):
                        for total in range(TARGET, other_value - 1, -1):
                            ways[total] += ways[total - other_value]
                containing_one_fixed_card = ways[TARGET - value]
                overlap_numerator += count * containing_one_fixed_card**2
            overlap_total += weight * Fraction(overlap_numerator, number * number)

    visit(1, n, [])
    denominator = MAX_VALUE**n
    mean = Fraction(sum(k * w for k, w in histogram.items()), denominator)
    second = Fraction(sum(k * k * w for k, w in histogram.items()), denominator)
    variance = second - mean * mean
    overlap = overlap_total / denominator
    overlap_given_reachable = overlap_total / nonempty_weight
    return {
        "histogram": histogram,
        "denominator": denominator,
        "mean": mean,
        "variance": variance,
        "size_profile": size_profile,
        "overlap": overlap,
        "overlap_given_reachable": overlap_given_reachable,
        "counterexample": counterexample,
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"
    if mode == "hitting":
        mean, second = hitting_moments(1)
        print("mean", float(mean), mean)
        print("variance", float(second - mean * mean))
        for row in reachability_rows(12):
            print("reach", *row)
    elif mode == "unique":
        for row in capped_solution_rows(30):
            print("unique", *row)
    elif mode == "extrema":
        cardinality_state_rows(10, {5, 8, 10}, verbose=True)
    elif mode == "minonly":
        extremum_rows(25, "min")
    elif mode == "maxonly":
        extremum_rows(14, "max")
    elif mode == "counts":
        sample_size = 5
        result = solution_count_distribution(sample_size)
        print("count", sample_size, "mean", float(result["mean"]),
              "var", float(result["variance"]), flush=True)
        print("hist", sample_size,
              sorted(result["histogram"].items()), flush=True)
        print("profile", sample_size, result["size_profile"], flush=True)
        print("overlap", sample_size, float(result["overlap"]),
              float(result["overlap_given_reachable"]), flush=True)
        print("counterexample", sample_size,
              result["counterexample"], flush=True)
    else:
        print("usage: analyze_sgs_ziyuan.py "
              "{hitting|unique|counts|extrema|minonly|maxonly}")
