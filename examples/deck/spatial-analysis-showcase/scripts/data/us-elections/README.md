# us-elections

Source: MIT Election Data + Science Lab (CC0), doi:10.7910/DVN/VOQCHQ. 2000-2020 from a verbatim copy of countypres_2000-2020.csv (the Dataverse file download needs a guestbook form). 2024 is rebuilt from MEDSL precinct files (`medsl2024_aggregate.py`, ~2.9 GB unzipped, slow, cached); a state is kept only if its county sum reconciles with MEDSL's state totals within 1.5%, otherwise NaN (AK, IN, LA, NJ, NY). Aligned to us-counties.

Run (needs the geo-venv; downloads are cached under scratchpad raw/): `python -I build.py` then `python -I validate.py`.
Output: `public/data/us-elections/` (manifest.json + raw little-endian .bin columns, one float32 column per attribute).
