# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX Euler paths over independently generated scikit-image contour edges.

Reference global edge identities, levels and directed endpoint coordinates come
from the hashed Round 5 composite scikit-image fixtures. NetworkX connected
components and Euler paths determine whole chains independently of GPU output.
Consecutive equal endpoints collapse; closed chains repeat their first point.
No ordering requirement across chains or starting vertex of cycles. G remains
8 f32 ulps at coordinate scale, exact level/closure/vertex-count topology.
https://networkx.org/documentation/stable/reference/algorithms/euler.html
"""
import base64,gzip,hashlib,json,platform
from pathlib import Path
import networkx as nx
import numpy as np
HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp4-contours.json'
original=json.loads(source.read_text())
cases=[]
for case in original['lineCases']:
    segments=np.asarray(json.loads(gzip.decompress(base64.b64decode(case['segments'])))).reshape(-1,8)
    chains=[]
    for level in range(len(case['levels'])):
        graph=nx.DiGraph()
        for segment in segments[segments[:,1]==level]:
            graph.add_edge(int(segment[2]),int(segment[3]),points=segment[4:].reshape(2,2).tolist())
        for nodes in nx.weakly_connected_components(graph):
            component=graph.subgraph(nodes)
            route=list(nx.eulerian_path(component))
            points=[component.edges[route[0]]['points'][0]]+[component.edges[edge]['points'][1] for edge in route]
            compact=[points[0]]
            for point in points[1:]:
                if point!=compact[-1]: compact.append(point)
            chains.append(dict(level=level,closed=int(nx.is_eulerian(component)),positions=np.asarray(compact).ravel().tolist()))
    cases.append(dict(name=case['name'],chains=chains))
fixture=dict(reference='networkx.weakly_connected_components/eulerian_path over independent skimage contours',
 versions=dict(python=platform.python_version(),numpy=np.__version__,networkx=nx.__version__,skimage=original['versions']['skimage']),
 source=dict(path=str(source.relative_to(HERE.parents[3])),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),cases=cases)
output=HERE/'fixtures/parity-wp4-stitched.json';output.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
assert output.stat().st_size<1_500_000
print(output.name,output.stat().st_size)
