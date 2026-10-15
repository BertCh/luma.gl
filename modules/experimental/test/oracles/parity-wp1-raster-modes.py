# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""All 720 focal parameter combinations reachable by the rainfall showcase.

radius=0..8, circle/square, minimumCount=1..20, propagateCenterNoData on/off;
all six outputs on a 7x7 grid with sentinel/NaN/mask holes. SciPy generic_filter
provides the reference through the already independently validated Round 4 helper.
Also GDAL invdist/invdistnn endpoint powers and k=12/32 showcase capacities.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
DIRECTORY=Path(__file__).parent

def load(name):
 spec=importlib.util.spec_from_file_location(name,DIRECTORY/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
round4=load('parity-wp1-round4');idw=load('parity-wp1-inverse-distance-weighting')
values=np.arange(49,dtype=float).reshape(7,7);values[1,2]=-9999;values[4,3]=np.nan
validity=np.ones((7,7),dtype=int);validity[3,3]=0;cases=[]
for radius in range(9):
 for shape in ['circle','square']:
  for minimum in range(1,21):
   for propagate in [False,True]:
    settings=dict(radius=radius,shape=shape,minimumCount=minimum,propagateCenterNoData=propagate)
    cases.append(round4.makeFocal(f'rainfall-{radius}-{shape}-{minimum}-{propagate}',values,validity,settings,8,{'source':'synthetic 7x7, all discrete showcase focal parameter combinations'}))
round4.writeFixture(DIRECTORY/'fixtures','round8-focal',dict(cases=cases))
wrapperPath=DIRECTORY/'fixtures/parity-wp1-round8-focal.json';wrapper=json.loads(wrapperPath.read_text());wrapper['generator']=Path(__file__).name;wrapper['command']='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-raster-modes.py';wrapperPath.write_text(json.dumps(wrapper,separators=(',',':'))+'\n')
positions=np.array([(column,row) for row in range(6) for column in range(6)],dtype=float)
dataset=idw.makeDataset(positions,np.arange(1,37),[1]*36,{'source':'synthetic integer lattice, all queries have exact f32 centers'})
library=idw.loadGDAL();cases=[]
for power in [0,.5,6]:
 for neighbors in [0,1,12,32]:
  for minimum in [1,8]:
   if neighbors==1 and minimum==8:continue
   cases.append(idw.makeCase(library,f'power-{power}-k-{neighbors}-min-{minimum}','lattice',dataset,[0,0,8,8],4,4,16,power,neighbors,minimum))
output=DIRECTORY/'fixtures/parity-wp1-idw-modes.json';output.write_text(json.dumps(idw.encode(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-raster-modes.py',versions={'GDAL':library.GDALVersionInfo(b'RELEASE_NAME').decode(),'numpy':np.__version__},datasets={'lattice':dataset},cases=cases)),separators=(',',':'))+'\n');print(output)
