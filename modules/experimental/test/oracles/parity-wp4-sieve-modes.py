# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent skimage RAG + regionprops reference for the documented one-pass sieve.

GDAL differs on ties and all-small chains, so these are explicitly a composite
reference: skimage.graph.RAG supplies 4/8 adjacency, regionprops supplies integer
areas, and a lexicographic (-area,label) ranking defines the documented largest
eligible neighbor. No recursive merging; background cannot receive a patch.
The public unfinished-label gate is asserted separately, not attributed to
skimage. Full outputs E; synthetic ties, islands, all-small chains, capacity,
validity, componentCount, convergence/overflow and restoration in one graph.
https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.RAG
"""
import json,platform
from pathlib import Path
import numpy as np
import skimage
from skimage.graph import RAG
from skimage.measure import regionprops
HERE=Path(__file__).resolve().parent
labels=np.array([[1,1,1,2,3,3,3],[1,1,1,2,3,3,3],[0,0,4,0,5,0,0],
                 [6,0,0,0,0,0,7],[0,0,0,0,0,8,7]],dtype=np.uint32)
cases=[]
for mode in ['remove','merge']:
 for connectivity in [4,8]:
  frames=[]
  for minimum,limit,valid,converged,overflow in [(3,8,True,1,0),(2,8,True,1,0),(100,8,True,1,0),(3,3,True,1,0),(3,8,False,1,0),(3,8,True,0,0),(3,8,True,1,1),(3,8,True,1,0)]:
   effective=np.where((labels<=limit)&valid,labels,0).astype(np.uint32)
   sizes={region.label:int(region.area) for region in regionprops(effective)}
   adjacency=RAG(effective,connectivity=1 if connectivity==4 else 2)
   targets=np.zeros(8,dtype=np.uint32)
   for label,size in sizes.items():
    if size>=minimum:targets[label-1]=label
    elif mode=='merge':
     choices=[other for other in adjacency.neighbors(label) if other>0 and sizes.get(other,0)>=minimum]
     if choices:targets[label-1]=min(choices,key=lambda other:(-sizes[other],other))
   result=np.zeros_like(labels)
   for label in sizes:result[effective==label]=targets[label-1]
   count=sum(size<minimum for size in sizes.values())
   if not converged or overflow: result.fill(2**32-1);targets.fill(2**32-1);count=2**32-1
   frames.append(dict(minimumPixels=minimum,componentCount=limit,valid=int(valid),converged=converged,overflow=overflow,
                      expected=dict(labels=result.ravel().tolist(),patchTargets=targets.tolist(),sievedCount=[count])))
  cases.append(dict(name=f'{mode}-{connectivity}-ties-islands-gates',width=7,height=5,labels=labels.ravel().tolist(),capacity=8,mode=mode,connectivity=connectivity,frames=frames))
fixture=dict(reference='composite skimage.graph.RAG + skimage.measure.regionprops + documented one-pass size ranking',
 versions=dict(python=platform.python_version(),numpy=np.__version__,skimage=skimage.__version__),cases=cases)
output=HERE/'fixtures/parity-wp4-sieve-modes.json';output.write_text(json.dumps(fixture,separators=(',',':'))+'\n');print(output.name,output.stat().st_size)
