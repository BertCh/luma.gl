# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX weighted Louvain/greedy quality references, not optimizer label parity.

Preserve original source multiplicity, loops, direction and f32-exact weights of
round-2 community inputs. OpenFlights includes all airports/edges. GPU local
moving is single-level and limited to 64 accepted moves; NetworkX is multilevel.
Four NetworkX seeds supply independent Louvain partitions, weighted asynchronous
LPA warm starts and vertex-ID permutations. Four full-graph singleton starts make
bounded cold-start quality visible. Small graphs use singleton starts, resolutions
0.5, 1, 2, loops, multiedges and islands. Compare all label feasibility and exact
objective evaluation; report per-run and distribution quality ratios (profile H),
never assert equality of different heuristics. For small converged outputs check
all one-vertex moves with the published modularity definition (composite reference,
Newman 2006 https://doi.org/10.1073/pnas.0601602103), not a CPU optimizer replica.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import networkx
import numpy

HERE = Path(__file__).resolve().parent
VERSIONS = {name: importlib.metadata.version(name) for name in ['networkx', 'numpy']}
VERSIONS['python'] = platform.python_version()


def canonical(communities, count):
    labels = [0] * count
    for community in communities:
        for node in community:
            labels[node] = min(community)
    return labels


def score(graph, labels, resolution):
    partition = [{node for node, label in enumerate(labels) if label == community}
                 for community in sorted(set(labels))]
    return networkx.community.modularity(graph, partition, weight='weight', resolution=resolution)


def maximum_gain(graph, labels, resolution):
    baseline = score(graph, labels, resolution)
    maximum = 0.0
    for node in graph:
        for label in range(len(labels)):
            candidate = labels.copy()
            candidate[node] = label
            maximum = max(maximum, score(graph, candidate, resolution) - baseline)
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ['undirected', 'directed', 'cliques', 'openflights']:
        path = HERE/'fixtures'/f'parity-wp3-graph-communities-{name}.json'
        source = json.loads(path.read_text())
        count = source['nodeCount']
        graph = networkx.MultiDiGraph() if source['directed'] else networkx.MultiGraph()
        graph.add_nodes_from(range(count))
        graph.add_weighted_edges_from(source['edges'])
        runs = []
        for resolution in ([1] if name == 'openflights' else [0.5, 1, 2]):
            greedy = list(networkx.community.greedy_modularity_communities(graph, resolution=resolution, weight='weight'))
            greedyLabels = canonical(greedy, count)
            for seed in [3, 17, 41, 97]:
                reference = canonical(networkx.community.louvain_communities(
                    graph, weight='weight', resolution=resolution, seed=seed), count)
                initial = (canonical(networkx.community.asyn_lpa_communities(graph, weight='weight', seed=seed), count)
                           if name == 'openflights' else list(range(count)))
                runs.append(dict(seed=seed, resolution=resolution, initial=initial,
                    permutation=numpy.random.default_rng(seed).permutation(count).tolist(),
                    labels=reference, score=score(graph, reference, resolution),
                    initialScore=score(graph, initial, resolution),
                    initialMaximumGain=maximum_gain(graph, initial, resolution) if count < 20 else -1,
                    greedyLabels=greedyLabels, greedyScore=score(graph, greedyLabels, resolution),
                    iterations=64, start='weighted-lpa' if name == 'openflights' else 'singleton'))
        if name == 'openflights':
            for warmRun in runs.copy():
                run = warmRun.copy()
                run.update(initial=list(range(count)), start='singleton',
                           initialScore=score(graph, list(range(count)), 1))
                runs.append(run)
        fixture = dict(case=name, nodeCount=count, edges=source['edges'], directed=source['directed'],
                       components=source['components'], runs=runs, source=source['source'],
                       inputFixtureSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       versions=VERSIONS, generator=Path(__file__).name, conventions=__doc__,
                       reference='networkx.community.louvain_communities; greedy_modularity_communities; modularity',
                       tolerance=dict(absolute=1e-6, relative=1e-4))
        target = output/f'parity-wp3-modularity-optimization-{name}.json'
        target.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False)+'\n')
        assert target.stat().st_size <= 1_500_000
        print(target.name, target.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
