# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Showcase 12/52/73-slice cubes, including cooperative >=64-slice trend.

Reuses the independent esda.G_Local expanded-cube, SciPy Kendall and statsmodels
FDR reference. Integer cubes exercise uint32 input and binary CSR/lattice paths.
Nonuniform weighted Gi* is covered separately by parity-wp1-emerging-weighted.py;
esda's analytic variance uses cardinality rather than sum(w*w).
"""
import importlib.util,json
from pathlib import Path
import numpy as np
DIRECTORY=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reference',DIRECTORY/'parity-wp1-emerging-hot-spots.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
random=np.random.default_rng(8203);cases=[]
for slices in [12,52,73]:
 cube=random.poisson(30,(6,slices));cube[0]+=np.arange(slices)//2;cube[5]+=np.arange(slices)[::-1]//2
 for method in ['none','benjamini-hochberg','benjamini-yekutieli']:
  cases.append(reference.makeCase(f'showcase-{slices}-{method}',cube,3,2,2,[1]*6,method,'require',{'source':'synthetic Poisson cube, numpy PCG64 seed 8203; rising/falling cells'}))
path=DIRECTORY/'fixtures/parity-wp1-emerging-lengths.json';path.write_text(json.dumps(reference.encode(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-emerging-lengths.py',versions={'numpy':np.__version__,'esda':reference.esda.__version__,'scipy':reference.scipy.__version__,'statsmodels':reference.statsmodels.__version__},cases=cases)),separators=(',',':'))+'\n');assert path.stat().st_size<1500000;print(path)
