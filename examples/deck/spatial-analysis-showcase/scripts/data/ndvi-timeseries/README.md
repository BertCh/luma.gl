# ndvi-timeseries

16 cloud-screened summer NDVI snapshots (uint8, 100 m, 256 x 256) across the 2021 Dixie Fire burn scar and its recovery.

- Licence: Copernicus Sentinel data: free, full and open
- Attribution: Contains modified Copernicus Sentinel data 2017-2024
- Source: https://registry.opendata.aws/sentinel-2-l2a-cogs/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/ndvi-timeseries/`; re-runnable).
- Output: `public/data/ndvi-timeseries/` (see manifest.json `properties.storyNotes`).
