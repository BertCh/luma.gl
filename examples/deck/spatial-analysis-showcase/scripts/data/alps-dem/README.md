# alps-dem

Terrarium-encoded DEM of the Matterhorn, Gornergrat and Gorner Glacier at 6.6 m ground resolution (2048 x 2048, Web Mercator, 0.25 m quantisation).

- Licence: Mapterhorn terrain tiles built from swisstopo swissALTI3D (OGD, free use with attribution) and Copernicus GLO-30
- Attribution: (c) swisstopo (swissALTI3D), (c) Mapterhorn; contains modified Copernicus data
- Source: https://mapterhorn.com/
- Rebuild: `geo-venv/bin/python -I build.py` (downloads are cached in scratchpad `showcase/raw/alps-dem/`; re-runnable).
- Output: `public/data/alps-dem/` (see manifest.json `properties.storyNotes`).
