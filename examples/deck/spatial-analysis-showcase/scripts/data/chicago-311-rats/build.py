import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import *
import pandas as pd, numpy as np

ID = 'chicago-311-rats'
c = pd.read_csv(f'{RAW}/chicago-311-rats/rats.csv')
b = CHICAGO_BBOX
c = c[c.longitude.between(b[0], b[2]) & c.latitude.between(b[1], b[3])].drop_duplicates('sr_number').copy()
c['ts'] = (pd.to_datetime(c.created_date) - pd.Timestamp('2023-01-01')).dt.total_seconds().astype('int64')
c = c.sort_values('ts').reset_index(drop=True)
tract, _ = assign_tract(c.longitude, c.latitude)
n = len(c); d = out_dir(ID)
m = {'id': ID, 'version': 1, 'kind': 'points', 'count': n, 'bbox': [round(float(c.longitude.min()), 5), round(float(c.latitude.min()), 5), round(float(c.longitude.max()), 5), round(float(c.latitude.max()), 5)], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'position', np.c_[c.longitude, c.latitude], 'float32', 'position.bin', 2)
write_col(d, m, 'timestamp', c.ts, 'uint32', 'time.bin', unit='seconds since 2023-01-01T00:00:00 (local clock time, stored as if UTC)')
write_col(d, m, 'tract', tract, 'uint16', 'tract.bin', note='index into chicago-tracts features; 65535 = none')
dt = pd.to_datetime(c.created_date); mon = dt.dt.month.value_counts().sort_index()
tc = pd.Series(tract[tract != 65535]).value_counts()
m['properties'] = {
  'description': 'Chicago 311 service requests of type "Rodent Baiting/Rat Complaint" created in 2023 (server-side filtered), a second point process for bivariate / Knox stories.',
  'source': 'https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy',
  'storyNotes': [
    f'{n:,} rodent complaints in 2023.',
    f'Peak month {mon.idxmax()} ({mon.max():,}); quietest {mon.idxmin()} ({mon.min():,}).',
    f'Busiest tract (index {tc.index[0]}) logged {tc.iloc[0]} complaints; complaint counts reflect reporting propensity as well as rats.',
    'Complaints are residential-address based and are good for testing co-location with other point processes such as wildlife sightings (cross-K, bivariate Moran) at tract level.']}
write_manifest(d, m)
print(m['properties']['storyNotes'])
