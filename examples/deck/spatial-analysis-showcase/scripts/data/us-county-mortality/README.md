# us-county-mortality

Source: NHTSA FARS 2017-2023 ACCIDENT files (public domain). Deaths per county of crash plus 2020 Census population: unstable small-county rates for empirical Bayes. Aligned to us-counties.

Run (needs the geo-venv; downloads are cached under scratchpad raw/): `python -I build.py` then `python -I validate.py`.
Output: `public/data/us-county-mortality/` (manifest.json + raw little-endian .bin columns, one float32 column per attribute).
