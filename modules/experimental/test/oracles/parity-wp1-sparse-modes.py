# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Additional documented CSR modes, using existing independent reference helpers.

SciPy sparse dot handles rectangular lag, uint source conversion and IEEE missing
values. A source value outside stored CSR entries does not poison other rows.
Transpose tolerance is exact absolute |W-W.T| on f32-uploaded weights, with missing
reverse edges always asymmetric. Neighborhood references remain pandas/SciPy,
including focal weights 1/2/4 crossed with raw/row-normalized metadata.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix
DIRECTORY=Path(__file__).parent

def load(name):
 spec=importlib.util.spec_from_file_location(name,DIRECTORY/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
round5=load('parity-wp1-round5');round6=load('parity-wp1-round6')
# Rectangular shape includes an entirely unused source row with NaN and an island.
matrix=csr_matrix(np.array([[1,2,0,0,0,0],[0,0,1,0,2,0],[0,0,0,0,0,0],[0,1,0,1,0,0]],dtype=float))
lag=[]
for integer in [False,True]:
 for missing in ([False] if integer else [False,True]):
  for normalize in [False,True]:
   values=np.arange(1,13,dtype=float).reshape(6,2)
   if missing:values[1,1]=np.nan;values[5,:]=np.nan
   sums=np.asarray(matrix.sum(axis=1)).ravel();result=matrix.dot(values)
   if normalize:result=np.divide(result,sums[:,None],out=np.zeros_like(result),where=sums[:,None]!=0)
   lag.append(dict(name=f'rectangular-integer={integer}-missing={missing}-normalize={normalize}',reference='scipy.sparse.csr_matrix.dot',n=4,columns=2,integer=integer,normalize=normalize,values=values.ravel(),expected=result.ravel(),**round5.packMatrix(matrix)))
output=DIRECTORY/'fixtures/parity-wp1-lag-modes.json';output.write_text(json.dumps(round5.clean(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-sparse-modes.py',versions={'numpy':np.__version__,'scipy':scipy.__version__},cases=lag)),separators=(',',':'))+'\n')
small=csr_matrix(np.array([[0,1,2,0],[1.0625,0,0,1],[2.5,0,0,1],[0,0,1.125,0]],dtype=float))
transpose=[]
for tolerance in [.0625,.125,.5]:
 case=round6.makeTranspose(f'symmetry-absolute-{tolerance}',small,4,{'source':'synthetic, exact dyadic differences'},checkSymmetry=True)
 support=round6.pattern(small);missing=(support-support.T)!=0;unequal=abs(small-small.T)>tolerance
 case['asymmetricSlots']=int(missing.maximum(unequal).count_nonzero());case['symmetryTolerance']=tolerance;transpose.append(case)
round5.writeFixture(DIRECTORY/'fixtures','round8-transpose',dict(cases=transpose))
neighborhood=[]
base=csr_matrix(np.array([[0,1,2,1],[1,0,0,3],[2,0,0,1],[1,3,1,0]],dtype=float))
for normalize in [False,True]:
 for focal in [0,1,2,4]:
  neighborhood.append(round5.makeNeighborhood(f'focal-{focal}-normalize-{normalize}',base,[2,4,8,16],[0,1,0,1],[1]*4,normalize,focal>0,max(1,focal),64,False,{'source':'synthetic'}))
large=csr_matrix(np.ones((65,65))-np.eye(65))
neighborhood.append(round5.makeNeighborhood('maximum-64-with-focal-overflow',large,np.arange(65),np.arange(65)%3,[1]*65,False,True,1,64,False,{'source':'65-row complete graph'}))
round5.writeFixture(DIRECTORY/'fixtures','round8-neighborhood',dict(cases=neighborhood))
for name in ['round8-transpose','round8-neighborhood']:
 path=DIRECTORY/f'fixtures/parity-wp1-{name}.json';wrapper=json.loads(path.read_text());wrapper['generator']=Path(__file__).name;wrapper['command']='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-sparse-modes.py';path.write_text(json.dumps(wrapper,separators=(',',':'))+'\n')
print(output)
