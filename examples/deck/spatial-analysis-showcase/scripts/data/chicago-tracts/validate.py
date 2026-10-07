"""Reads back every Chicago dataset: lengths, bbox, NaN, index ranges, polygon offsets."""
import json, os, sys, numpy as np
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
IDS = ['chicago-nature', 'chicago-tracts', 'chicago-community-areas', 'chicago-places', 'chicago-facilities', 'chicago-lodes-od', 'chicago-311-rats']
for i in IDS:
    d = f'{APP}/public/data/{i}'; m = json.load(open(f'{d}/manifest.json')); cols = {}
    for k, c in m['columns'].items():
        a = np.fromfile(f"{d}/{c['file']}", dtype='<' + {'float32': 'f4', 'uint32': 'u4', 'uint8': 'u1', 'uint16': 'u2'}[c['dtype']])
        assert a.size == c['length'] * c['components'], (i, k, a.size)
        cols[k] = a.reshape(-1, c['components']) if c['components'] > 1 else a
    nan = {k: int(np.isnan(v).sum()) for k, v in cols.items() if v.dtype.kind == 'f' and np.isnan(v).any()}
    if 'position' in cols:
        p = cols['position']; b = m['bbox']
        assert p[:, 0].min() >= b[0] - 1e-3 and p[:, 0].max() <= b[2] + 1e-3 and p[:, 1].min() >= b[1] - 1e-3 and p[:, 1].max() <= b[3] + 1e-3
        assert -88 < p[:, 0].min() and p[:, 0].max() < -87.5 and 41.6 < p[:, 1].min() and p[:, 1].max() < 42.1
        for k in m['columns']:
            assert m['columns'][k]['length'] == m['count'], (i, k)
        if 'tract' in cols: print('  tract unmatched:', int((cols['tract'] == 65535).sum()))
        if 'category' in cols: assert cols['category'].max() < len(m['columns']['category']['categories'])
    if 'vertices' in cols:
        ro, pr = cols['ringOffsets'], cols['polygonRingOffsets']
        assert ro[-1] == len(cols['vertices']) and pr[-1] == len(ro) - 1 and (np.diff(ro) >= 4).all()
        assert cols['partFeature'].max() == m['count'] - 1
    if 'origin' in cols:
        assert cols['origin'].max() < len(cols['locations']) and cols['destination'].max() < len(cols['locations'])
    print(i, 'OK', m['count'], 'NaN:', nan)
