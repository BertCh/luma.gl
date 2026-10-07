# ghcn-stations

6,135 weather stations with 2024-09-27 precipitation (and previous day, TMAX where reported) plus elevation for the southeastern US.

- Licence: NOAA NCEI GHCN-Daily, public domain
- Attribution: NOAA National Centers for Environmental Information, GHCN-Daily
- Source: https://registry.opendata.aws/noaa-ghcn/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/ghcn-stations/`; re-runnable).
- Output: `public/data/ghcn-stations/` (see manifest.json `properties.storyNotes`).
