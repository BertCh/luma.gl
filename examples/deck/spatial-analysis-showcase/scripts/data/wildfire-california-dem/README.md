# wildfire-california-dem

Terrarium DEM of northern and central California (lon -123.30 to -120.10, lat 36.40 to 41.00) for the wildfire terrain scene.

- Source: AWS Open Data `elevation-tiles-prod` (Mapzen/Tilezen Terrain Tiles), no key. Tiles are
  `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/9/{x}/{y}.png`.
- Licence: public domain and open source inputs (USGS NED, SRTM, GMTED2010, ETOPO1 and others); attribution required, see
  https://github.com/tilezen/joerd/blob/master/docs/attribution.md
- Rebuild: `RAW=<empty dir> python3 -I scripts/data/wildfire-california-dem/build.py` (Python with numpy and Pillow; tiles are cached in RAW).
- Output: `public/data/wildfire-california-dem/` (1166 x 2148 Terrarium PNG, 1 m quantisation, 2.4 MB). Web Mercator pixels of
  305.7 m at the equator, about 239 m on the ground at 38.7 N. Cells at or below 0 m are noData (-9999).
- 37 isolated spikes (more than 500 m from the 3x3 median, up to 8420 m) are replaced by the median.
