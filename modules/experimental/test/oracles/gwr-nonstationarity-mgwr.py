# Generator of MGWR_NONSTATIONARITY_REFERENCE (test/gpu-spatial-analysis/spatial-regression/mgwr-reference.ts).
# Input JSON: positions/predictors/response of createNonstationarityScene(12, 77) and
# assignments = getNonstationarityAssignment(144, 12345, p) for p = 1..19 (dump them from a node spec).
# Usage: python -I gwr-nonstationarity-mgwr.py scene.json  (mgwr 2.2.1, numpy 2.5.4)
import json, sys
import numpy as np
import mgwr
from mgwr.gwr import GWR

scene = json.load(open(sys.argv[1]))
coords = np.array(scene['positions']).reshape(-1, 2)
X = np.array(scene['predictors']).reshape(-1, 2)
y = np.array(scene['response']).reshape(-1, 1)
assignments = [np.array(a) for a in scene['assignments']]

def sd(c, bw, fixed):
    params = GWR(c, y, X, bw, kernel='bisquare', fixed=fixed, constant=True).fit().params
    return np.std(params, axis=0)

out = {}
for name, bw, fixed in [('fixed', 2.2, True), ('adaptive', 20, False)]:
    rows = [sd(coords, bw, fixed)]
    for a in assignments:
        # location j receives observation a[j]: observation i sits at coordinate argsort(a)[i].
        rows.append(sd(coords[np.argsort(a)], bw, fixed))
    rows = np.array(rows)
    exceed = [int(np.sum(rows[1:, c] >= rows[0, c])) for c in range(rows.shape[1])]
    out[name] = {'standardDeviations': [[float('%.12g' % v) for v in r] for r in rows], 'exceedances': exceed}
print('mgwr', mgwr.__version__, 'numpy', np.__version__)
print(json.dumps(out))
