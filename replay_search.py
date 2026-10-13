"""Canonical replay used by both the CLI tables and the paper figure.

Run this file to reproduce fig4_search_bands_statistics.json without plotting.
"""
import argparse
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPTHS = tuple(range(3, 10))
BUDGETS = tuple(range(10, 201, 10))
METHODS = ("local search", "random")
TRIALS = 300
BASE_SEED = 42
IMPROVEMENT_TOLERANCE = 1e-12
STATISTICS_FILENAME = "fig4_search_bands_statistics.json"


def load_landscape(database=HERE / "search.db"):
    """Preserve released row order, which determines uniform sampling indices."""
    database = Path(database).resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        return [(tuple(map(int, schedule.split(","))), float(accuracy))
                for schedule, accuracy in connection.execute(
                    "SELECT schedule_key, val_top1 FROM search_runs ORDER BY rowid")]


def trial_seed(method, depth, trial):
    """Disjoint consecutive seed ranges; the first range is 42,...,341."""
    if not 0 <= trial < TRIALS:
        raise ValueError("Trial index must be in 0,...,299")
    return BASE_SEED + trial + TRIALS * (
        DEPTHS.index(depth) + len(DEPTHS) * METHODS.index(method))


def neighbors(schedule):
    kept = [i for i, value in enumerate(schedule) if value]
    omitted = [i for i, value in enumerate(schedule) if not value]
    for i in kept:
        for j in omitted:
            candidate = list(schedule)
            candidate[i], candidate[j] = 0, 1
            yield tuple(candidate)


def local_search_trace(scores, depth, budget, seed):
    """Best measured score after each lookup, including restarts and repeats."""
    if budget < 1:
        raise ValueError("Budget must be positive")
    rng = random.Random(seed)
    num_blocks = len(next(iter(scores)))
    if not 0 < depth <= num_blocks:
        raise ValueError("Invalid retained depth")
    trace, best = [], float("-inf")

    def evaluate(schedule):
        nonlocal best
        value = scores[schedule]
        best = max(best, value)
        trace.append(best)
        return value

    while len(trace) < budget:
        retained = set(rng.sample(range(num_blocks), depth))
        current = tuple(int(i in retained) for i in range(num_blocks))
        current_value = evaluate(current)
        while len(trace) < budget:
            candidates = list(neighbors(current))
            rng.shuffle(candidates)
            moved = False
            for candidate in candidates:
                if len(trace) == budget:
                    return trace
                value = evaluate(candidate)
                if value > current_value + IMPROVEMENT_TOLERANCE:
                    current, current_value = candidate, value
                    moved = True
                    break
            if not moved:
                break  # Start a new random subset; charge its evaluation.
    return trace


def random_search_trace(population, budget, seed):
    """Uniform draws with replacement; every draw consumes one evaluation."""
    if budget < 1:
        raise ValueError("Budget must be positive")
    rng = random.Random(seed)
    trace, best = [], float("-inf")
    for _ in range(budget):
        _, value = rng.choice(population)
        best = max(best, value)
        trace.append(best)
    return trace


def run_replay(database=HERE / "search.db", *, budgets=BUDGETS):
    budgets = tuple(budgets)
    if not budgets or any(not isinstance(b, int) or b < 1 for b in budgets):
        raise ValueError("Budgets must be positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Budgets must be distinct and increasing")
    database = Path(database).resolve()
    rows = load_landscape(database)
    scores = dict(rows)
    population = defaultdict(list)
    for schedule, value in rows:
        population[sum(schedule)].append((schedule, value))
    optimum = {k: max(value for _, value in population[k]) for k in DEPTHS}
    records = {
        "schema_version": 2, "trials": TRIALS, "depths": list(DEPTHS),
        "budgets": list(budgets), "base_seed": BASE_SEED,
        "seed_rule": "42 + trial + 300 * ((k - 3) + 7 * method_index)",
        "method_order": list(METHODS), "percentiles": [10, 90],
        "percentile_method": "linear", "nested_budgets": True,
        "improvement_tolerance": IMPROVEMENT_TOLERANCE,
        "aggregation": "mean gap across depths per repetition",
        "interpretation": "search randomness with fixed measured accuracies",
        "evaluation_accounting": "every lookup, including restarts and repeats; no cache",
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "optimum_by_depth": {str(k): optimum[k] for k in DEPTHS},
        "methods": {},
    }
    for method in METHODS:
        seeds = [[trial_seed(method, k, trial) for trial in range(TRIALS)]
                 for k in DEPTHS]
        traces = np.asarray([
            [local_search_trace(scores, k, max(budgets), seed) if method == "local search"
             else random_search_trace(population[k], max(budgets), seed)
             for seed in depth_seeds]
            for k, depth_seeds in zip(DEPTHS, seeds)
        ])
        records["methods"][method] = []
        for budget in budgets:
            gaps = traces[:, :, budget - 1] - np.asarray([optimum[k] for k in DEPTHS])[:, None]
            averaged = gaps.mean(axis=0)
            low, high = np.percentile(averaged, [10, 90], method="linear")
            assert gaps.shape == (len(DEPTHS), TRIALS)
            assert np.all(gaps <= 1e-10)
            records["methods"][method].append({
                "budget": budget, "seeds_by_depth_and_trial": seeds,
                "mean": float(averaged.mean()), "p10": float(low), "p90": float(high),
                "gaps_by_depth_and_trial": gaps.tolist(),
            })
    return records


def write_records(records, path=HERE / STATISTICS_FILENAME):
    Path(path).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def print_summary(records):
    print("300 trials per depth; seeds start at 42; first-improvement local search.")
    print("Gaps are selected minus best observed accuracy (pp); higher is better.")
    print("method          B      mean       p10       p90")
    for method, entries in records["methods"].items():
        for entry in entries:
            print(f"{method:12s} {entry['budget']:4d} {entry['mean']:9.5f} "
                  f"{entry['p10']:9.5f} {entry['p90']:9.5f}")
    print("\nPer-depth means and optimum-hit rates:")
    print("method          B   k  mean gap  optimum hit (%)")
    for method, entries in records["methods"].items():
        for entry in entries:
            for k, values in zip(DEPTHS, entry["gaps_by_depth_and_trial"]):
                values = np.asarray(values)
                print(f"{method:12s} {entry['budget']:4d} {k:3d} "
                      f"{values.mean():9.5f} {100*np.mean(np.abs(values)<1e-9):16.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / STATISTICS_FILENAME)
    args = parser.parse_args()
    results = run_replay()
    write_records(results, args.output)
    print_summary(results)
