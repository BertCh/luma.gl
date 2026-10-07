# us-counties

Source: Census cb_2022_us_county_20m (public domain); CDC/ATSDR SVI 2022 county CSV; CDC PLACES county data 2024 release (Socrata fu4u-a9bh, public domain); USDA ERS RUCC 2023 and Unemployment/Median Household Income workbook. Contiguous US + DC, sorted by FIPS. Binary polygons are GeoArrow-style (countyPolygonOffsets, polygonRingOffsets, ringOffsets, vertices); adjacent counties share identical vertices. `story.py` computes story notes (Moran, LISA) into the manifest.

Run (needs the geo-venv; downloads are cached under scratchpad raw/): `python -I build.py` then `python -I validate.py`.
Output: `public/data/us-counties/` (manifest.json + raw little-endian .bin columns, one float32 column per attribute).
