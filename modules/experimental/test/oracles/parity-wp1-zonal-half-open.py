# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GEOS half-open membership adapter for orthogonal boundaries.

An infinitesimal northeast point displacement implements left/bottom inclusion,
right/top exclusion for these axis-aligned rectangles (including shared corners).
The displacement is for the independent GEOS predicate ONLY, not GPU uploads.
Pandas supplies the same unweighted/weighted summaries as the original generator.
No claim for arbitrary sloped boundary conventions.
"""
from pathlib import Path
import importlib.util
import json,hashlib
import numpy as np
path=Path(__file__).with_name('parity-wp1-round4.py');spec=importlib.util.spec_from_file_location('reference',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
features=[[[[[0,0],[4,0],[4,4],[0,4]]]],[[[[4,0],[8,0],[8,4],[4,4]]]],[[[[12,0],[16,0],[16,4],[12,4]]]]]
positions=np.array([[0,0],[4,0],[8,0],[0,4],[4,4],[8,4],[2,2],[4,2],[6,2],[2,0],[12,2],[14,2],[18,0]],dtype=float)
cases=[]
for weighted in [False,True]:
 case=module.makeZonal('half-open-'+str(weighted),features,positions+2**-18,np.arange(1,len(positions)+1,dtype=float),np.linspace(.5,2,len(positions)) if weighted else None,'open',{'adapter':'orthogonal boundaries, northeast 2^-18 displacement for GEOS only'})
 case['inputFloat32Sha256']=hashlib.sha256(positions.astype('<f4').tobytes()+np.asarray(case['values'],dtype='<f4').tobytes()).hexdigest();case['positions']=positions.ravel();case['boundaryRule']='half-open';cases.append(case)
output=Path(__file__).parent/'fixtures/parity-wp1-round8-zonal-half-open.json';output.write_text(json.dumps(module.clean(dict(generator=Path(__file__).name,versions=module.VERSIONS,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-zonal-half-open.py',cases=cases)),separators=(',',':'))+'\n');print(output)
