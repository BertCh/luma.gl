# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reproduce independent force-layout references at the airline scene defaults.
Uses SciPy optimization and the existing published-energy reference, no GPU solver.
The fixed 14x14 grid and [-6,6]^2 extent match airline-network.compute.ts.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
specification=importlib.util.spec_from_file_location('energy',HERE/'parity-wp3-force-layout.py')
energy=importlib.util.module_from_spec(specification)
specification.loader.exec_module(energy)
energy.generate(HERE/'fixtures','showcase',8,[(0,1,1),(1,2,1),(2,0,1),(2,3,1),(4,5,1),(5,6,1)],
    dict(description=__doc__),dict(repulsion=.0002,attraction=.045,gravity=.025,damping=.85,timeStep=1,maxVelocity=.045))
subprocess.run([sys.executable,'-I',str(HERE/'parity-wp3-spatial-force-layout.py'),'--names','showcase','--grid-size','14','--bound','6'],check=True)
# The scene supplies an all-zero pin mask. Keep the constrained variant too, and
# use the unconstrained optimizers for this unpinned comparison, at the same seeds.
import json
source=json.loads((HERE/'fixtures/parity-wp3-force-layout-showcase.json').read_text())
spatial=json.loads((HERE/'fixtures/parity-wp3-spatial-force-layout-showcase.json').read_text())
spatial.update(case='unpinned-showcase',pinned=[0]*source['nodeCount'],generator=Path(__file__).name,
    reference='networkx.spring_layout; scipy.optimize.minimize; scipy.stats.binned_statistic_2d; composite Noack energy')
spatial['runs']=[dict(run,gridReference=grid['gridReference']) for run,grid in zip(source['runs'],spatial['runs'])]
(HERE/'fixtures/parity-wp3-spatial-force-layout-showcase-unpinned.json').write_text(json.dumps(spatial,separators=(',',':'),allow_nan=False)+'\n')
