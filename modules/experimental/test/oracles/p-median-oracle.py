#!/usr/bin/env python3
"""Oracle generator for the GPULocationAllocation p-median property fixture.

Reference implementation: the exact p-median integer program (ReVelle and Swain 1970) solved by
scipy.optimize.milp (HiGHS branch and cut). Nothing about the GPU heuristic is reimplemented: the
fixture records the true optimum, so the GPU greedy-plus-swap result can be checked against it.

Formulation, for demand rows i (weight w_i), candidate facilities j and budget p:
  minimize   sum_ij w_i c_ij x_ij + M sum_i w_i u_i
  subject to sum_j x_ij + u_i = 1        (each row is assigned or left unreachable)
             x_ij <= y_j                  (only to an open facility)
             sum_j y_j = p                (exactly p facilities; GPULocationAllocation opens p)
             x_ij = 0 where c_ij = +inf   (unreachable pairs)
             x, y, u binary
M is larger than every finite total cost, so the objective is lexicographic: first the demand
weight left unreachable, then the reachable weighted cost. That is GPULocationAllocation's
documented p-median order (unreachable rows are never summed into the cost).

Instances use integer costs in [0, 50] and integer weights in [1, 5], so every objective is an
integer below 2^24 and float32 sums are exact: the recorded tolerance is 0. About a third of the
instances also have +inf entries, and some have a row unreachable from every facility.

Every instance is also solved by brute force over all p-subsets here, and the two optima must
agree exactly before the fixture is written.

Usage (writes ../gpu-spatial-analysis/location-allocation/p-median-fixture.json):
  python3 -I p-median-oracle.py
Needs numpy and scipy (versions are recorded in the fixture).
"""
import itertools
import json
import os
import sys
from importlib.metadata import version

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

SEED = 20261014
INSTANCE_COUNT = 24
TOLERANCE = 0


def solve_milp(costs, weights, p):
    demand_count, facility_count = costs.shape
    reachable = np.isfinite(costs)
    finite_costs = np.where(reachable, costs, 0.0)
    penalty = float((weights[:, None] * finite_costs).sum() + 1.0)
    # Variable order: x (demand-major), y, u.
    x_count = demand_count * facility_count
    variable_count = x_count + facility_count + demand_count
    objective = np.concatenate(
        [(weights[:, None] * finite_costs).ravel(), np.zeros(facility_count), penalty * weights]
    )
    rows = []
    lower = []
    upper = []
    for i in range(demand_count):
        row = np.zeros(variable_count)
        row[i * facility_count : (i + 1) * facility_count] = 1.0
        row[x_count + facility_count + i] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for i in range(demand_count):
        for j in range(facility_count):
            row = np.zeros(variable_count)
            row[i * facility_count + j] = 1.0
            row[x_count + j] = -1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)
    row = np.zeros(variable_count)
    row[x_count : x_count + facility_count] = 1.0
    rows.append(row)
    lower.append(float(p))
    upper.append(float(p))
    variable_upper = np.ones(variable_count)
    variable_upper[:x_count] = reachable.ravel().astype(float)
    result = milp(
        objective,
        constraints=LinearConstraint(np.array(rows), lower, upper),
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), variable_upper),
    )
    if not result.success:
        raise RuntimeError(result.message)
    solution = np.round(result.x).astype(int)
    unreachable_weight = int((weights * solution[x_count + facility_count :]).sum())
    cost = int(round(float((weights[:, None] * finite_costs).ravel() @ solution[:x_count])))
    return unreachable_weight, cost


def solve_brute_force(costs, weights, p):
    best = None
    for subset in itertools.combinations(range(costs.shape[1]), p):
        nearest = costs[:, subset].min(axis=1)
        reachable = np.isfinite(nearest)
        unreachable_weight = int(weights[~reachable].sum())
        cost = int(weights[reachable] @ nearest[reachable])
        if best is None or (unreachable_weight, cost) < best:
            best = (unreachable_weight, cost)
    return best


def main():
    generator = np.random.default_rng(SEED)
    instances = []
    for instance in range(INSTANCE_COUNT):
        demand_count = int(generator.integers(12, 41))
        facility_count = int(generator.integers(6, 13))
        p = int(generator.integers(1, min(5, facility_count) + 1))
        costs = generator.integers(0, 51, size=(demand_count, facility_count)).astype(float)
        weights = generator.integers(1, 6, size=demand_count).astype(float)
        if instance % 3 == 2:
            costs[generator.random(costs.shape) < 0.3] = np.inf
            costs[0, :] = np.inf
        milp_optimum = solve_milp(costs, weights, p)
        brute_optimum = solve_brute_force(costs, weights, p)
        if milp_optimum != brute_optimum:
            raise RuntimeError(f'instance {instance}: milp {milp_optimum} != brute force {brute_optimum}')
        instances.append(
            {
                'demandCount': demand_count,
                'facilityCount': facility_count,
                'maximumFacilityCount': p,
                # JSON has no infinity: null marks an unreachable pair.
                'costs': [None if not np.isfinite(value) else int(value) for value in costs.ravel()],
                'demandWeights': [int(value) for value in weights],
                'optimalUnreachableWeight': milp_optimum[0],
                'optimalCost': milp_optimum[1],
            }
        )
    fixture = {
        'generator': os.path.basename(__file__),
        'reference': 'scipy.optimize.milp (HiGHS), p-median ILP, cross-checked by brute force',
        'versions': {
            'python': sys.version.split()[0],
            'numpy': version('numpy'),
            'scipy': version('scipy'),
        },
        'seed': SEED,
        'tolerance': TOLERANCE,
        'instances': instances,
    }
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'gpu-spatial-analysis', 'location-allocation', 'p-median-fixture.json',
    )
    path = os.path.normpath(path)
    with open(path, 'w') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {len(instances)} instances to {path}')


if __name__ == '__main__':
    main()
