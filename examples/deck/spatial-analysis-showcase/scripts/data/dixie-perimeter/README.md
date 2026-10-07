# dixie-perimeter
Final 2021 Dixie Fire perimeter (California, 963,405 acres).

- Source: the `DIXIE` feature (2021) of `public/data/poopdeck-wildfires` (NIFC Open Data, public domain, packaged by poopdeck.gl); no new download. Attribution as that dataset.
- Processing (`python -I build.py`): rebuild the 179 shell and 21 hole rings with shapely, union, drop parts < 1 ha, simplify 20 m in EPSG:32610 (topology kept), snap to 1e-5 degrees. 32 parts, 3,897 km2 (963,066 acres after simplification; `properties.acres` keeps the NIFC value).
- Output (334 KB): `dixie-perimeter.geojson` and GeoArrow binary columns (`vertices`, `ringOffsets`, `polygonRingOffsets`, `partFeature`, `acres`).
