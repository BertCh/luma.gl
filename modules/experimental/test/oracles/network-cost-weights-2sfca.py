# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference two-step (and three-step) floating catchment area scores on a network cost matrix, from PySAL
`access`.

Writes fixtures/network-cost-weights-2sfca.json next to this script (consumed by
test/gpu-network/network-cost-weights/gpu-network-cost-weights.spec.ts, which builds the facility
and demand weights with GPUNetworkCostWeights and scores them with GPUCatchmentAccessibility). Run
from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/network-cost-weights-2sfca.py
in a venv with numpy, pandas and access (versions are recorded in the fixture).

Reference: access.fca.two_stage_fca(demand_df, supply_df, cost_df, max_cost, weight_fn), which is
Luo and Wang (2003) with the distance-decay extension of Luo and Qi (2009):
R_j = S_j / sum_i P_i f(c_ij) over demand i with c_ij <= max_cost, then A_i = sum_j R_j f(c_ij).
The decays are the GPUNetworkCostWeights definitions: none (1), exponential exp(-beta c) and the
Dai (2010) Gaussian (exp(-c^2 / (2 L^2)) - exp(-1/2)) / (1 - exp(-1/2)) with L = max_cost.

Scene: 7 facilities and 40 demand locations on a 20 x 20 network grid; costs are float32 network
distances (Manhattan path lengths of random float32 street weights in [1, 2), computed with
Dijkstra here) with a few unreachable entries (+Infinity) and one facility at a demand location
(cost 0). The cost limit keeps every demand location within reach of at least one facility.

Tolerance: the GPU sums in float32 in slot order. Each A_i is a sum of at most F = 7 products of a
ratio and a decay; each ratio is S_j over a float32 sum of at most D = 40 terms. With f32 decay
error at most 16 ulp (WGSL exp accuracy, 3 + 2|x| ulp for |x| <= 4, plus the Gaussian division),
the relative error of A_i is bounded by (16 + D + F + 2) 2^-24 < 2^-18; the spec uses 1e-5.
"""

import heapq
import json
import math
import os
import sys

import access
import numpy
import pandas
from access import fca

GRID = 20
FACILITY_COUNT = 7
DEMAND_COUNT = 40
COST_LIMIT = 14.0
BETA = 0.15


def to_float32(value):
    return float(numpy.float32(value))


def build_grid(random):
    """Returns adjacency lists of a GRID x GRID street grid with float32 weights in [1, 2)."""
    node_count = GRID * GRID
    edges = [[] for _ in range(node_count)]
    for row in range(GRID):
        for column in range(GRID):
            node = row * GRID + column
            for d_row, d_column in ((0, 1), (1, 0)):
                next_row, next_column = row + d_row, column + d_column
                if next_row < GRID and next_column < GRID:
                    neighbor = next_row * GRID + next_column
                    weight = to_float32(1 + random.random())
                    edges[node].append((neighbor, weight))
                    edges[neighbor].append((node, weight))
    return edges


def dijkstra(edges, start):
    costs = [math.inf] * len(edges)
    costs[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs[node]:
            continue
        for neighbor, weight in edges[node]:
            candidate = to_float32(cost + weight)
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return costs


def gaussian(cost):
    edge = math.exp(-0.5)
    ratio = cost / COST_LIMIT
    return (math.exp(-0.5 * ratio * ratio) - edge) / (1 - edge)


DECAYS = {
    'none': lambda cost: 1.0,
    'exponential': lambda cost: math.exp(-BETA * cost),
    'gaussian': gaussian
}


def main():
    random = numpy.random.default_rng(20261014)
    edges = build_grid(random)
    nodes = random.permutation(GRID * GRID)
    demand_nodes = [int(node) for node in nodes[:DEMAND_COUNT]]
    # Facility 0 sits on demand location 5 (cost 0).
    facility_nodes = [demand_nodes[5]] + [
        int(node) for node in nodes[DEMAND_COUNT : DEMAND_COUNT + FACILITY_COUNT - 1]
    ]
    costs = []
    for facility in facility_nodes:
        node_costs = dijkstra(edges, facility)
        costs.append([node_costs[node] for node in demand_nodes])
    # A few unreachable entries.
    for facility, demand in ((1, 3), (2, 17), (6, 31)):
        costs[facility][demand] = math.inf
    supply = [to_float32(value) for value in random.uniform(5, 50, FACILITY_COUNT)]
    demand = [to_float32(value) for value in random.uniform(100, 1000, DEMAND_COUNT)]

    rows = [
        {'origin': i, 'dest': j, 'cost': costs[j][i]}
        for j in range(FACILITY_COUNT)
        for i in range(DEMAND_COUNT)
        if math.isfinite(costs[j][i])
    ]
    cost_df = pandas.DataFrame(rows)
    demand_df = pandas.DataFrame({'geoid': range(DEMAND_COUNT), 'demand': demand})
    supply_df = pandas.DataFrame({'geoid': range(FACILITY_COUNT), 'supply': supply}).set_index(
        'geoid'
    )

    scenes = {}
    three_stage = {}
    for name, decay in DECAYS.items():
        series = fca.two_stage_fca(
            demand_df,
            supply_df,
            cost_df,
            max_cost=COST_LIMIT,
            weight_fn=decay,
        )
        scores = [float(series.get(i, 0.0)) for i in range(DEMAND_COUNT)]
        assert all(math.isfinite(score) and score > 0 for score in scores), name
        scenes[name] = scores
        # access computes the 3SFCA selection weight G_ij = W_ij / sum_k W_ik over every cost row of
        # demand i, before the max_cost filter, so pass only the in-range rows (Wan et al. 2012
        # sum over the catchment).
        in_range = cost_df[cost_df['cost'] <= COST_LIMIT].copy()
        series = fca.three_stage_fca(
            demand_df,
            supply_df,
            in_range,
            max_cost=COST_LIMIT,
            weight_fn=decay,
        )
        three_stage[name] = [float(series.get(i, 0.0)) for i in range(DEMAND_COUNT)]

    fixture = {
        'versions': {
            'python': sys.version.split()[0],
            'numpy': numpy.__version__,
            'pandas': pandas.__version__,
            'access': access.__version__
        },
        'facilityCount': FACILITY_COUNT,
        'demandCount': DEMAND_COUNT,
        'costLimit': COST_LIMIT,
        'beta': BETA,
        # Row-major [facility x demand]; null is +Infinity (unreachable).
        'costs': [None if not math.isfinite(cost) else cost for row in costs for cost in row],
        'supply': supply,
        'demand': demand,
        'accessibility': scenes,
        # Wan, Zou and Sternberg (2012) 3SFCA from access: R_j = S_j / sum_i G_ij W_ij P_i and
        # A_i = sum_j G_ij W_ij R_j. Not asserted by the spec yet (see the NET2 handoff).
        'threeStageAccessibility': three_stage,
        'relativeTolerance': 1e-5
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
    with open(os.path.join(path, 'network-cost-weights-2sfca.json'), 'w') as file:
        json.dump(fixture, file, indent=1)
        file.write('\n')


if __name__ == '__main__':
    main()
