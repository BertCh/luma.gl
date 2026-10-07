# dixie-fire

15 km window at 20 m on one UTM 10N grid: Sentinel-2 L2A red/NIR/SWIR22 reflectance and scene classification for 2021-07-13 and 2021-09-21, ESA WorldCover 2021 classes and a Terrarium DEM.

- Licence: Copernicus Sentinel data: free, full and open; ESA WorldCover: CC BY 4.0; terrain: AWS Terrain Tiles
- Attribution: Contains modified Copernicus Sentinel data 2021; (c) ESA WorldCover 2021 (CC BY 4.0); terrain from AWS Terrain Tiles / USGS 3DEP
- Source: https://registry.opendata.aws/sentinel-2-l2a-cogs/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/dixie-fire/`; re-runnable).
- Output: `public/data/dixie-fire/` (see manifest.json `properties.storyNotes`).
