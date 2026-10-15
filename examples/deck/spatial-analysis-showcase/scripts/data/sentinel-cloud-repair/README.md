# Sentinel-2 cloud repair pairs

Two aligned natural-colour target/prior pairs plus cloud and cloud-shadow masks, read as Cloud Optimized
GeoTIFF windows from the public Element 84 Earth Search Sentinel-2 Level-2A Collection 1 archive and written
to `public/data/sentinel-cloud-repair/` on a shared 1536 × 1536 grid.

- Hamersley Range, Pilbara (MGRS 50KPA): 4 February 2026 target, 30 January 2026 clear prior; 25.6 km at 16.7 m.
- Colorado Plateau, Utah (MGRS 12SWF): 29 July 2026 target, 27 July 2026 clear prior; 20.5 km at 13.3 m.

Mask values are 0 clear, 1 cloud, 2 cloud shadow, built on the 20 m grid with the s2cloudless-style recipe:
Sen2Cor cloud probability (`CLD_20m`) ≥ 50 %, shadow where NIR is dark inside the cloud footprint projected
away from the sun for cloud tops up to 8 km, 40 m erosion and 100 m buffer. Two multitemporal checks against the
clear prior (as in MAJA and Tmask) catch thin fringes (blue reflectance up more than 0.06 within 600 m of a
probability cloud) and keep naturally dark terrain out of the shadow class (target NIR below 70 % of the prior).
The build fails if the prior is not clear under the same recipe. SCL is not used for cloud: it labelled
sun-glinted ocean as cirrus in the earlier Oʻahu crop.

From this directory, with a Python environment that has `numpy`, `scipy`, `rasterio`, `pyproj` and `Pillow`:

```bash
GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR python -I build.py
```

Sources:

- Earth Search catalog and public Sentinel COGs: <https://element84.com/earth-search/>
- Sentinel-2 Level-2A, SCL and cloud probability:
  <https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html>
- s2cloudless cloud and shadow recipe:
  <https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless>
