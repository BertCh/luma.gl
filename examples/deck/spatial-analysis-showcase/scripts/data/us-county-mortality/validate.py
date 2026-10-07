import json, os, numpy as np
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-county-mortality'))
m = json.load(open(f'{D}/manifest.json')); n = m['count']
c = {k: np.fromfile(f"{D}/{v['file']}", dtype=np.float32) for k, v in m['columns'].items()}
assert all(len(a) == n for a in c.values())
assert abs(sum(c[f'deaths{y}'] for y in range(2017, 2024)) - c['deaths']).max() < 1e-3
assert (c['deaths'] >= 0).all() and not np.isnan(c['deaths']).any()
print('NaN population:', int(np.isnan(c['population']).sum()), 'deaths total', int(c['deaths'].sum()), 'zero-death counties', int((c['deaths'] == 0).sum()))
print('validate OK')
