# alps-dem-wide

Terrarium-encoded DEM of the whole Gornergrat view (Zermatt, Matterhorn, Monte Rosa, Weisshorn, Dom): 2331 x 2181 px, Web Mercator zoom 12 (19.1 m per pixel, about 13.3 m ground resolution at 46 N), 0.25 m vertical quantisation. A wider sibling of `alps-dem`, which is unchanged.

- Window: lon 7.58 to 7.98, lat 45.86 to 46.12, snapped outward to whole Mercator pixels (manifest `bounds` / `boundsMercator`).
- Source: Mapterhorn terrain tiles, `https://tiles.mapterhorn.com/12/x/y.webp` (tile size 512, Terrarium). Re-encoded to a PNG after quantising to 0.25 m (the PNG is 4.2 MB, so the 0.5 m fallback was not needed and the south edge was not cropped).
- Licence and attribution, from `https://mapterhorn.com/attribution/` (the page loads `https://download.mapterhorn.com/attribution.json`, 151 sources):
  - swissALTI3D, swisstopo, licence "Open Government Data" (free use, attribution required), Switzerland.
  - COPERNICUS GLO-30, "provided under COPERNICUS by the European Union and ESA. All rights reserved", licence "COPERNICUS full, free and open license".
  - Italian regional models in the catalogue that can cover the Italian side of the window: Valle d'Aosta DTM (`itaosta`, CC BY 4.0, 2 m) and Regione Piemonte DTM 5 (`itpiemonte`, CC BY 4.0, 5 m); also TINITALY (`tinitaly`, INGV, CC BY 4.0, 10 m).
  - Not verified: which source Mapterhorn actually serves at each pixel. The attribution page lists the sources, not their footprints (the coverage map is a vector tileset that was not decoded). The Swiss side is swissALTI3D; the Italian side is one of the sources above. The listed attributions are all carried in the dataset attribution.
- Rebuild: `FID/venv/bin/python -I build.py [south] [quantisation_m]` (defaults 45.86 and 0.25; set `RAW` to the tile cache directory). Tiles are cached in the raw directory; requests use the User-Agent `luma-showcase-data-build/1.0 (https://github.com/visgl/luma.gl)`.
- Output: `public/data/alps-dem-wide/` (`dem.png`, `manifest.json`; same schema as `alps-dem`; observers Gornergrat, Matterhorn, Zermatt, Dufourspitze with DEM elevation).
