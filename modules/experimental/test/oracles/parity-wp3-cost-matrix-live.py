# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX references for the transit single-start/default-initial-cost mode
and live network weight changes including a closed edge. Uses unquantized f32
seconds; no path tie/ID claims. Fourteen hub rows, every finite/infinite cost.
NetworkX removes +Infinity closure arcs and computes each hub's Dijkstra costs.
S/L error budget: atol=1e-6*max(1,expected cost), rtol=1e-4, unchanged profile.
"""
import importlib.metadata
import json
import math
from pathlib import Path
import networkx
import numpy
HERE=Path(__file__).resolve().parent
original=json.loads((HERE/'fixtures/parity-wp3-cost-matrix-synthetic-input.json').read_text())
starts=[index%original['nodeCount'] for index in range(14)]
cases=[]
for name,factor,closed in [('baseline',1,False),('slower',1.7,False),('closed',1.7,True)]:
    weights=numpy.asarray([(value+.13)*100*factor for value in original['weights']],dtype=numpy.float32).astype(float)
    if closed:weights[2]=math.inf
    graph=networkx.MultiDiGraph()
    graph.add_nodes_from(range(original['nodeCount']))
    for source in graph:
        for edge in range(original['offsets'][source],original['offsets'][source+1]):
            if math.isfinite(weights[edge]):graph.add_edge(source,original['neighbors'][edge],weight=weights[edge])
    costs=[]
    for source in starts:
        distances=networkx.single_source_dijkstra_path_length(graph,source)
        costs.extend(distances.get(node,math.inf) for node in graph)
    encode=lambda values:[value if math.isfinite(value) else 'Infinity' for value in values]
    cases.append(dict(case=name,weights=encode(weights.tolist()),costs=encode(costs),tolerance=dict(absolute=1e-6*max(1,*(value for value in costs if math.isfinite(value))),relative=1e-4)))
fixture=dict(case='transit-and-live-weights',reference='networkx.single_source_dijkstra_path_length',versions={name:importlib.metadata.version(name) for name in ['networkx','numpy']},conventions=__doc__,generator=Path(__file__).name,nodeCount=original['nodeCount'],offsets=original['offsets'],neighbors=original['neighbors'],startNodes=starts,cases=cases)
path=HERE/'fixtures/parity-wp3-cost-matrix-live.json'
path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
print(path.name,path.stat().st_size)
