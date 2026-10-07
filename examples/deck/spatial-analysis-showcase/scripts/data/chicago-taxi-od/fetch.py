"""Fetch server-side aggregates of Chicago Taxi Trips 2023 from the Socrata API (wrvz-psew), per month."""
import json, sys, time, pathlib, requests
from concurrent.futures import ThreadPoolExecutor
RAW = pathlib.Path(sys.argv[1]); RAW.mkdir(parents=True, exist_ok=True)
URL = 'https://data.cityofchicago.org/resource/wrvz-psew.json'
def month_bounds(m):
    a = f'2023-{m:02d}-01T00:00:00'
    b = f'2023-{m+1:02d}-01T00:00:00' if m < 12 else '2024-01-01T00:00:00'
    return a, b
def q(name, m, select, group):
    out = RAW / f'{name}-{m:02d}.json'
    if out.exists(): return
    a, b = month_bounds(m)
    params = {'$select': select, '$group': group, '$limit': 500000,
              '$where': f"trip_start_timestamp >= '{a}' AND trip_start_timestamp < '{b}'"}
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=600); r.raise_for_status()
            out.write_text(r.text); print(name, m, len(r.json()), flush=True); return
        except Exception as e:
            print('retry', name, m, e, flush=True); time.sleep(5 * (attempt + 1))
jobs = []
for m in range(1, 13):
    jobs.append(('od', m, "pickup_community_area as p, dropoff_community_area as d, date_extract_dow(trip_start_timestamp) as dow, date_extract_hh(trip_start_timestamp) as h, count(*) as n, avg(fare) as fare, avg(trip_seconds) as secs, avg(trip_miles) as miles", "p,d,dow,h"))
    jobs.append(('daily', m, "date_trunc_ymd(trip_start_timestamp) as day, pickup_community_area as p, count(*) as n, avg(fare) as fare", "day,p"))
with ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda j: q(*j), jobs))
