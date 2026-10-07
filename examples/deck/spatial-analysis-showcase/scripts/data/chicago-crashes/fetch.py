"""Fetch Chicago Traffic Crashes 2023 (Socrata 85ca-t3if) as CSV."""
import sys, pathlib, requests
out = pathlib.Path(sys.argv[1]) / 'crashes2023.csv'
params = {'$select': 'crash_record_id,crash_date,latitude,longitude,injuries_total,injuries_fatal,injuries_incapacitating,injuries_non_incapacitating,crash_type,first_crash_type,prim_contributory_cause,posted_speed_limit,street_name,num_units,most_severe_injury',
          '$where': "crash_date >= '2023-01-01T00:00:00' AND crash_date < '2024-01-01T00:00:00'", '$limit': 200000, '$order': 'crash_date'}
r = requests.get('https://data.cityofchicago.org/resource/85ca-t3if.csv', params=params, timeout=600); r.raise_for_status()
out.write_bytes(r.content); print(len(r.content))
