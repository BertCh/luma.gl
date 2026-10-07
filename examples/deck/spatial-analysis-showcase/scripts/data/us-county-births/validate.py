import json, os, numpy as np
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-county-births'))
m = json.load(open(f'{D}/manifest.json')); n = m['count']
c = {k: np.fromfile(f"{D}/{v['file']}", dtype=np.float32) for k, v in m['columns'].items()}
assert all(len(a) == n for a in c.values())
assert abs(sum(c[f'births{y}'] for y in (2021, 2022, 2023)) - c['births']).max() < 1e-2
assert abs(sum(c[f'women{y}'] for y in (2021, 2022, 2023)) - c['womenYears']).max() < 1
assert (c['births'] >= 0).all() and (c['womenYears'] > 0).all() and not np.isnan(c['rawRate']).any()
print('births', int(c['births'].sum()), 'zero-birth counties', int((c['births'] == 0).sum()))
print('validate OK')
