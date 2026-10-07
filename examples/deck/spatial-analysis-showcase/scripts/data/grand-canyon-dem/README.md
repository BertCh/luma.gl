# grand-canyon-dem

Terrarium DEM of the central Grand Canyon at about 15.5 m ground resolution (2048 x 2048, Web Mercator): rims, side canyons and the Colorado River gorge.

- Licence: AWS Open Data Terrain Tiles (public bucket); underlying USGS 3DEP/NED is public domain, other sources require attribution per Tilezen list
- Attribution: Terrain Tiles on AWS (Mapzen/Tilezen): USGS 3DEP/NED, SRTM and other sources; see https://github.com/tilezen/joerd/blob/master/docs/attribution.md
- Source: https://registry.opendata.aws/terrain-tiles/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/grand-canyon-dem/`; re-runnable).
- Output: `public/data/grand-canyon-dem/` (see manifest.json `properties.storyNotes`).
