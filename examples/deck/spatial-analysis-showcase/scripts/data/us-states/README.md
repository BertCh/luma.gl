# us-states

Dissolved from the us-counties geometry so edges match exactly.

Run (needs the geo-venv; downloads are cached under scratchpad raw/): `python -I build.py` then `python -I validate.py`.
Output: `public/data/us-states/` (manifest.json + raw little-endian .bin columns, one float32 column per attribute).
