import json, time, urllib.request, sys
OUT = sys.argv[1]
base = ('https://api.inaturalist.org/v1/observations?place_id=49906&d1=2023-01-01&d2=2023-12-31'
        '&licensed=true&preferred_place_id=49906&order_by=id&order=asc&per_page=200&id_above=')
rows, last = [], 0
while True:
    for attempt in range(5):
        try:
            req = urllib.request.Request(base + str(last), headers={'User-Agent': 'luma-showcase-data-build'})
            d = json.load(urllib.request.urlopen(req, timeout=60)); break
        except Exception as e:
            print('retry', e, flush=True); time.sleep(5 * (attempt + 1))
    res = d['results']
    if not res: break
    for o in res:
        t = o.get('taxon') or {}
        rows.append({'id': o['id'], 'location': o.get('location'), 'observed_on': o.get('observed_on'),
                     'time_observed_at': o.get('time_observed_at'), 'quality_grade': o.get('quality_grade'),
                     'license': o.get('license_code'), 'captive': o.get('captive'), 'obscured': o.get('obscured'),
                     'accuracy': o.get('positional_accuracy'), 'iconic': t.get('iconic_taxon_name'),
                     'taxon_id': t.get('id'), 'name': t.get('name'), 'common': t.get('preferred_common_name'),
                     'rank': t.get('rank'), 'native': t.get('native'), 'introduced': t.get('introduced'),
                     'user': (o.get('user') or {}).get('login')})
    last = res[-1]['id']
    print(len(rows), d['total_results'], flush=True)
    time.sleep(1.1)
json.dump(rows, open(OUT, 'w'))
