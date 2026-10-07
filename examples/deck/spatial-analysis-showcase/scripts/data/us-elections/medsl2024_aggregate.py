"""Aggregate MEDSL 2024 precinct returns (CC0) to county x candidate x mode. Slow (~2.9 GB CSV); output cached in RAW.

MIT Election Data + Science Lab, "U.S. President Precinct-Level Returns 2024"
(github.com/MEDSL/2024-elections-official, Harvard Dataverse collection 2024_precincts).
The MIT county-level 2000-2024 file on Dataverse requires a guestbook form for download, so the 2024 county
totals are rebuilt from the precinct files, which are openly downloadable.
"""
import glob, os, subprocess, zipfile
import pandas as pd
SC = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
RAW = f'{SC}/raw/us-elections'
STATES = 'al ak az ar ca co ct dc de fl ga hi ia id il in ks ky la ma md me mi mn mo ms mt nc nd ne nh nj nm nv ny oh ok or pa ri sc sd tn tx ut va vt wa wi wv wy'.split()
BASE = 'https://raw.githubusercontent.com/MEDSL/2024-elections-official/main/'
os.makedirs(f'{RAW}/medsl2024', exist_ok=True); os.makedirs(f'{RAW}/medsl2024x', exist_ok=True)
def get(url, dest):
    if not os.path.exists(dest): subprocess.check_call(['curl', '-sSL', '-m', '900', '-o', dest, url])
get(BASE + '2024-president-state.csv', f'{RAW}/medsl2024/pres-state.csv')
out = f'{RAW}/pres2024_cand_mode.csv'
if os.path.exists(out): print('cached', out); raise SystemExit
frames = []
for st in STATES:
    z = f'{RAW}/medsl2024/{st}24.zip'; get(BASE + f'individual_states/{st}24.zip', z)
    csv = f'{RAW}/medsl2024x/{st}24.csv'
    if not os.path.exists(csv): zipfile.ZipFile(z).extractall(f'{RAW}/medsl2024x')
    hdr = pd.read_csv(csv, nrows=0).columns.tolist()
    use = [c for c in ['office', 'candidate', 'mode', 'votes', 'county_fips', 'county_name'] if c in hdr]
    parts = []
    for ch in pd.read_csv(csv, usecols=use, dtype=str, chunksize=500000, low_memory=False):
        ch = ch[ch.office == 'US PRESIDENT']
        if not len(ch): continue
        ch['v'] = pd.to_numeric(ch.votes, errors='coerce'); ch['nstar'] = (ch.votes == '*')
        for c in ['mode', 'county_fips', 'candidate']: ch[c] = ch[c].fillna('NA')
        parts.append(ch.groupby(['county_fips', 'county_name', 'candidate', 'mode']).agg(v=('v', 'sum'), nstar=('nstar', 'sum')).reset_index())
    p = pd.concat(parts).groupby(['county_fips', 'county_name', 'candidate', 'mode']).sum().reset_index(); p['st'] = st
    frames.append(p); print(st, len(p), flush=True)
pd.concat(frames).to_csv(out, index=False)
