# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Exact rational shared-edge triage, independent of GPU execution.

Enumerate ALL horizontal pixel-centre crossings of the existing f32 Chicago
geometry. A crossing less than half an f32 ULP from a centre can round ONTO
that centre even with a perfectly evaluated interpolant. This is an explicit
E-profile boundary ambiguity, not a global tolerance or changed ownership rule.
Retain only shared edges; report exact rational distances, both feature IDs,
and pixel IDs. No raster outputs or cached GPU results supply expected values.
"""
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
source = HERE/'fixtures/parity-wp4-polygons.json'
parent = json.loads(source.read_text())
scene = parent['cases'][0]
points = np.array(scene['positions']).reshape(-1, 2)
rings, polygons, features = (scene[key] for key in ['ringOffsets', 'polygonOffsets', 'featureOffsets'])
ambiguous = {}
for feature in range(len(features)-1):
    for polygon in range(features[feature], features[feature+1]):
        for ring in range(polygons[polygon], polygons[polygon+1]):
            vertices = points[rings[ring]:rings[ring+1]]
            for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
                lower, upper = sorted([first, second], key=lambda point: point[1])
                for row in range(max(0, int(np.ceil(lower[1]-.5))), min(scene['height'], int(np.ceil(upper[1]-.5)))):
                    crossing = Fraction(lower[0]) + (Fraction(upper[0])-Fraction(lower[0])) * (Fraction(row+.5)-Fraction(lower[1])) / (Fraction(upper[1])-Fraction(lower[1]))
                    column = round(crossing-Fraction(1, 2))
                    centre = Fraction(column)+Fraction(1, 2)
                    margin = crossing-centre
                    half_ulp = float(np.spacing(np.float32(float(centre))))/2
                    if abs(margin) <= Fraction(half_ulp) and 0 <= column < scene['width']:
                        index = row*scene['width']+column
                        entry = ambiguous.setdefault(index, dict(index=index, row=row, column=column,
                            centre=[column+.5, row+.5], crossing=float(crossing), exactMargin=str(margin),
                            halfUlp=half_ulp, edge=[lower.tolist(), upper.tolist()], features=[]))
                        assert entry['edge'] == [lower.tolist(), upper.tolist()]
                        entry['features'].append(feature)
assert all(len(entry['features']) == 2 for entry in ambiguous.values())
output = dict(versions=dict(python=platform.python_version(), numpy=np.__version__),
    reference='fractions.Fraction exact line intersection; IEEE-754 round-to-nearest half-ULP interval',
    sources=[dict(path=str(source.relative_to(HERE.parents[3])), sha256=hashlib.sha256(source.read_bytes()).hexdigest()), *parent['sources']],
    case=scene['name'], cells=list(ambiguous.values()))
(HERE/'fixtures/parity-wp4-polygon-centres.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))
