# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy composite weighted linear combination for site-suitability.

Five layers use the exact scene topology: two clipped linear criteria, one
left-closed eight-break cover table, two inverted linear criteria. All eight
normalization x missingness x restriction toggle combinations have separate
references. Each repeats after stack and table edits and restores its input.
NumPy digitize defines cover intervals; clip defines linear utility; sum of
weighted utility defines score, optionally divided by sum of absolute available
weights. NaN utility is a restriction even when a weight is zero. Missing input
is ignored or propagates according to the setting. Definition: Malczewski,
GIS and Multicriteria Decision Analysis (1999), weighted linear combination,
with the public API's explicit restriction/missingness convention. This is a
composite reference, not an invocation of our TS/WGSL implementation.
Inputs, weights, bounds and remap values are exactly f32-representable. Full
257-cell arrays include table break equalities, masked layers and empty cells.
"""
import json,platform
from pathlib import Path
import numpy as np

count=257
stack=np.vstack([(np.arange(count)*7%41)/16,(np.arange(count)*3%65)/2,
                 np.resize([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],count),
                 (np.arange(count)*13%129)*16,(np.arange(count)*5%33)/16]).astype(np.float32).astype(float)
stack[:,0]=np.nan;stack[0,1::17]=np.nan;stack[1,2::19]=-9999
breaks=np.zeros((5,8));breaks[2]=[15,25,35,45,55,65,75,85]
layers=[dict(weight=.5,inputMinimum=0,inputMaximum=2),dict(weight=.25,inputMinimum=0,inputMaximum=32),
 dict(weight=1,mode='table',breakCount=8,closed='left'),dict(weight=.125,inputMinimum=0,inputMaximum=2048,invert=True),
 dict(weight=.5,inputMinimum=0,inputMaximum=2,invert=True)]
cases=[]
for normalize in [False,True]:
 for policy in ['ignore','propagate']:
  for restricted in [False,True]:
   frames=[]
   for edit in [False,True,False]:
    values=stack.copy()
    if edit:values[0,3:]+=0.125;values[2]=np.roll(values[2],1)
    table=np.zeros((5,9));table[2]=[.75,.5,1,.25,np.nan if restricted else .125,.5,0,np.nan if restricted else 0,1]
    if edit:table[2,0]=.25;table[2,3]=.75
    usable=~np.isnan(values)&(values!=-9999)
    mapped=np.empty_like(values)
    for index,layer in enumerate(layers):
     if layer.get('mode')=='table':mapped[index]=table[index,np.digitize(values[index],breaks[index],right=False)]
     else:
      mapped[index]=np.clip((values[index]-layer['inputMinimum'])/(layer['inputMaximum']-layer['inputMinimum']),0,1)
      if layer.get('invert'):mapped[index]=1-mapped[index]
    weights=np.array([layer['weight'] for layer in layers])[:,None]
    denominator=np.sum(np.where(usable,abs(weights),0),axis=0)
    with np.errstate(invalid='ignore',divide='ignore'):
     score=np.sum(np.where(usable,mapped*weights,0),axis=0)
     if normalize:score/=denominator
    defined=np.any(usable,axis=0)&~np.any(usable&np.isnan(mapped),axis=0)
    if policy=='propagate':defined &= np.all(usable,axis=0)
    if normalize:defined &= denominator>0
    score[~defined]=np.nan;finite=score[defined]
    frames.append(dict(stack=values.ravel().tolist(),remapValues=table.ravel().tolist(),expected=dict(score=score.tolist(),scoreRange=[float(np.min(finite)),float(np.max(finite))])))
   cases.append(dict(name=f'normalized-{normalize}-{policy}-restricted-{restricted}',settings=dict(layers=layers,normalizeWeights=normalize,noDataPolicy=policy),frames=frames))
def encode(value):
 if isinstance(value,dict):return {key:encode(item) for key,item in value.items()}
 if isinstance(value,list):return [encode(item) for item in value]
 if isinstance(value,float) and np.isnan(value):return None
 return value
fixture=dict(reference='composite: numpy.clip/digitize/sum/min/max; Malczewski (1999) weighted linear combination',versions=dict(python=platform.python_version(),numpy=np.__version__),cellCount=count,layerCount=5,remapBreaks=breaks.ravel().tolist(),cases=cases)
output=Path(__file__).resolve().parent/'fixtures/parity-wp4-overlay-showcase.json'
output.write_text(json.dumps(encode(fixture),separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<1_500_000
print(output.name,output.stat().st_size)
