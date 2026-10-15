# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Complete showcase operation/rule, order/cumulative and self-weight controls.

SciPy sparse set arithmetic and shortest_path supply the reference, reusing the
Round 6 independent implementation. Difference/symmetric difference have no
shared slots: all six non-binary rules necessarily agree, while binary sets 1.
All self weights 0..3 step .25 use the showcase's per-row buffer representation.
"""
import importlib.util,json,copy,base64,zlib,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
DIRECTORY=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reference',DIRECTORY/'parity-wp1-round6.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
left=csr_matrix((np.arange(1,9)/4,(np.arange(8),np.roll(np.arange(8),-1))),shape=(8,8))
right=csr_matrix((np.arange(1,9)/8,(np.arange(8),np.roll(np.arange(8),-2))),shape=(8,8))
right=right+csr_matrix(([0.5],([0],[1])),shape=(8,8))
dataset,mask=reference.makeAlgebraDataset('eight-zone-directed-cycle',left,right,np.arange(8)%3,{'source':'synthetic directed cycle and two-step partner'})
cases=reference.algebraCases(0,dataset,left,right,mask)
cases=[case for case in cases if case['operation'] not in ['selfWeight','higherOrder']]
for operation in ['difference','symmetricDifference']:
 prototype=next(case for case in cases if case['operation']==operation and case['weightRule']=='left')
 for rule in ['right','sum','min','max','product']:
  case=copy.deepcopy(prototype);case.update(name=dataset['name']+'/'+operation+'-'+rule,weightRule=rule);cases.append(case)
def add(name,operation,result,**settings):
 result.sort_indices();cases.append(dict(name=name,reference='scipy.sparse setdiag; scipy.sparse.csgraph.shortest_path',n=8,datasetIndex=0,operation=operation,expected=reference.pack(result),rowSums=np.asarray(result.sum(axis=1)).ravel(),**settings))
for weight in np.arange(13)/4:
 result=left.copy();result.setdiag(weight);add(f'per-row-self-{weight}','selfWeight',result,selfWeight=weight)
distances=shortest_path(reference.pattern(left),directed=True,unweighted=True)
for order in range(2,9):
 for cumulative in [False,True]:
  support=(distances>0)&(distances<=order if cumulative else distances==order)
  add(f'order-{order}-cumulative-{cumulative}','higherOrder',csr_matrix(support.astype(float)),order=order,cumulative=cumulative,workCapacity=64)
payload=json.dumps(reference.clean(dict(datasets=[dataset],cases=cases)),separators=(',',':')).encode()
output=DIRECTORY/'fixtures/parity-wp1-round8-algebra.json';output.write_text(json.dumps(dict(versions=reference.VERSIONS,encoding='zlib-base64-json',sha256=hashlib.sha256(payload).hexdigest(),data=base64.b64encode(zlib.compress(payload,9)).decode())))
path=DIRECTORY/'fixtures/parity-wp1-round8-algebra.json';wrapper=json.loads(path.read_text());wrapper.update(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-algebra-modes.py');path.write_text(json.dumps(wrapper,separators=(',',':'))+'\n');print(len(cases))
