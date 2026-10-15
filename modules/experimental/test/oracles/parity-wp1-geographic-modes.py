# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent geometric median optimization and circular statistics.

scipy.optimize.minimize(Powell) minimizes sum(w*Euclidean distance), with analytic
subgradient residual validation, independently of GPU Weiszfeld iterations.
SciPy circmean/circvar on integer-weight replicated directions supplies directed
and axial means; norm supplies lengths. No duplicated GPU numerical algorithm.
Finite positive weights and nondegenerate lines; convergence cases use 256 steps,
1e-6 tolerance. This is a converged-solution comparison, not 24-step trajectory.
"""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import circmean,circvar
positions=np.array([[-4,0],[0,-3],[4,0],[0,5],[-2,3],[3,2],[-3,-2],[2,-2]],dtype=float)
ends=positions+np.array([[3,0],[0,4],[-3,4],[4,3],[3,4],[4,-3],[0,2],[2,0]])
cases=[]
for weighted in [False,True]:
 weights=np.array([1,2,1,2,1,1,2,1] if weighted else [1]*8)
 objective=lambda center: np.sum(weights*np.linalg.norm(positions-center,axis=1))
 result=minimize(objective,np.average(positions,axis=0,weights=weights),method='Powell',options={'xtol':1e-13,'ftol':1e-13,'maxiter':10000})
 direction=(positions-result.x)/np.linalg.norm(positions-result.x,axis=1)[:,None]
 assert np.linalg.norm(np.sum(weights[:,None]*direction,axis=0))<1e-6
 delta=ends-positions; angles=np.arctan2(delta[:,1],delta[:,0]);lengths=np.linalg.norm(delta,axis=1)
 for axial in [False,True]:
  multiplier=2 if axial else 1; repeated=np.repeat(angles*multiplier,weights)
  mean=float(circmean(repeated,low=-np.pi,high=np.pi))/multiplier
  # The unweighted axial resultant cancels exactly. SciPy documents an
  # implementation-dependent circmean when it is zero; the GPU API uses NaN.
  # Certify zero using rational double-angle identities on these integer deltas.
  if axial:
   cosine=sum(Fraction(int(weight)*(int(deltaX)**2-int(deltaY)**2),int(deltaX)**2+int(deltaY)**2) for (deltaX,deltaY),weight in zip(delta,weights))
   sine=sum(Fraction(2*int(weight)*int(deltaX)*int(deltaY),int(deltaX)**2+int(deltaY)**2) for (deltaX,deltaY),weight in zip(delta,weights))
   if cosine==0 and sine==0: mean=None
  cases.append(dict(name=f'median-lines-weighted={weighted}-axial={axial}',reference='scipy.optimize.minimize Powell; scipy.stats.circmean/circvar; numpy.linalg.norm',positions=positions.ravel().tolist(),lineEnds=ends.ravel().tolist(),weights=weights.tolist(),orientationOnly=axial,median=result.x.tolist(),medianObjective=float(result.fun),directional=[mean,float(circvar(repeated)),float(np.average(lengths,weights=weights))]))
path=Path(__file__).parent/'fixtures/parity-wp1-round8-geographic-modes.json';path.write_text(json.dumps(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-geographic-modes.py',versions={'numpy':np.__version__,'scipy':scipy.__version__},cases=cases),separators=(',',':'))+'\n');print(path)
