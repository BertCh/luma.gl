# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Showcase zonal output layouts: six statistics, two density denominators.

Independent Shapely memberships/pandas aggregations are reused. Half-open uses
the documented axis-aligned northeast perturbation only in Shapely; original
coordinates are retained for the GPU. Counts do not depend on values. Population
is caller-supplied synthetic area metadata; reference density is count/population.
"""
import importlib.util,json,copy
from pathlib import Path
import numpy as np
DIRECTORY=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reference',DIRECTORY/'parity-wp1-round4.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
prototype=json.loads((DIRECTORY/'fixtures/parity-wp1-round8-zonal-half-open.json').read_text())['cases'][0]
cases=[]
for boundary in ['open','closed','half-open']:
 base=copy.deepcopy(prototype)
 if boundary!='half-open':
  base=reference.makeZonal('rectangles-'+boundary,base['features'],np.array(base['positions']).reshape(-1,2),base['values'],None,boundary,{'source':'synthetic adjacent rectangles'})
 base['boundaryRule']=boundary
 for statistic in ['count','sum','mean','minimum','maximum','density-area','density-population']:
  for sumOrder in ['atomic','sorted']:
   case=copy.deepcopy(base);case.update(name=f'{boundary}-{statistic}-{sumOrder}',statistic=statistic,sumOrder=sumOrder)
   key={'count':'counts','sum':'sums','mean':'means','minimum':'minima','maximum':'maxima','density-area':'densities','density-population':'densities'}[statistic]
   case['areas']=[2+index for index in range(base['featureCount'])] if statistic=='density-population' else []
   expected=np.array(case['expected'][key],dtype=float)
   if case['areas']:expected=np.array(case['expected']['counts'])/case['areas'];case['expected']['densities']=expected
   finite=expected[np.isfinite(expected)];case['expected']['extent']=[float(finite.min()),float(finite.max())]
   cases.append(case)
output=DIRECTORY/'fixtures/parity-wp1-round8-zonal-showcase.json';output.write_text(json.dumps(reference.clean(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-zonal-showcase.py',versions=reference.VERSIONS,cases=cases)),separators=(',',':'))+'\n');print(output,len(cases))
