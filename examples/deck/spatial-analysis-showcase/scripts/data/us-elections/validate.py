"""Read back us-elections bins: lengths, ranges, NaN coverage."""
import json, os, numpy as np
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-elections'))
m = json.load(open(f'{D}/manifest.json')); n = m['count']
for k, c in m['columns'].items():
    a = np.fromfile(f"{D}/{c['file']}", dtype=np.float32); assert len(a) == n, k
    if k.startswith(('dem', 'rep')) and 'Swing' not in k and 'TwoParty' not in k: assert np.nanmin(a) >= 0 and np.nanmax(a) <= 1.0001, (k, np.nanmin(a), np.nanmax(a))
for y in m['properties']['years']:
    t = np.fromfile(f'{D}/totalVotes{y}.bin', dtype=np.float32)
    print(y, 'counties', int((~np.isnan(t)).sum()), 'votes %.1fM' % (np.nansum(t) / 1e6))
print('validate OK')
