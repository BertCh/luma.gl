# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy constant-field trajectories for every wind particle-count/trail topology.

Each GPU invocation receives the same f32 initial point and uniform field.
Broadcasting this independently solved trajectory is exact, not sampling a GPU
array: the spec compares every particle and every trail slot. Ten graphs cover
10000/30000/60000 x 8/16/32 and terrain 40000 x 24. Four frames cover viewport/full spawn parameters x
fixed/zoom-adjusted time steps; random lifecycle is separately tested with
randomgen.Philox. No random respawns on these dispatch/stride cases.
"""
import json,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
position=np.array([4.,4.]);frames=[]
for adaptive in [False,True]:
 for viewport in [False,True]:
  step=240 if adaptive else 120
  previous=position.copy();position=solve_ivp(lambda time,point:[1/4096,0],[0,step],position,method='DOP853',rtol=1e-12,atol=1e-12).y[:,-1]
  frames.append(dict(timeStep=step,viewport=viewport,adaptive=adaptive,position=position.tolist(),previous=previous.tolist(),speed=1/4096))
fixture=dict(reference='scipy.integrate.solve_ivp (constant ODE; exact independent broadcast)',
 versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),topologies=[dict(count=count,trailLength=length) for count in [10000,30000,60000] for length in [8,16,32]]+[dict(count=40000,trailLength=24)],frames=frames)
output=Path(__file__).resolve().parent/'fixtures/parity-wp4-particle-topology.json';output.write_text(json.dumps(fixture,separators=(',',':'))+'\n');print(output.name,output.stat().st_size)
