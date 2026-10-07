# gfs-wind

Hourly 10 m and 3-hourly 250 hPa u/v wind from the 2024-09-26 12Z NOAA GFS run on a 0.25 degree grid over the Gulf, Southeast US and Atlantic, quantised into RG PNGs.

- Licence: US Government work (NOAA), public domain
- Attribution: NOAA / NCEP Global Forecast System
- Source: https://registry.opendata.aws/noaa-gfs-bdp-pds/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/gfs-wind/`; re-runnable).
- Output: `public/data/gfs-wind/` (see manifest.json `properties.storyNotes`).
