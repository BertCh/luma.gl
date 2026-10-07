# chicago-boundary
Derived Chicago geography: city limit, Lake Michigan and a land mask.

- Sources: `public/data/chicago-community-areas/community-areas.geojson` (City of Chicago Data Portal Terms of Use; build that dataset first) and Natural Earth 1:10m lakes (`ne_10m_lakes.geojson`, public domain, downloaded 2026-10-07 by the `natural-earth` build raw folder). Attribution `City of Chicago; Made with Natural Earth`.
- Processing (`python -I build.py [ne_10m_lakes.geojson]`): (a) dissolve the 77 areas (`unary_union` in EPSG:26916), close gaps with a +1.5/-1.5 m buffer, simplify 2 m, holes < 0.1 ha dropped -> `city.geojson` (3 parts: the main city and two small O'Hare outliers, 597.7 km2); (b) Natural Earth "Lake Michigan" clipped to `[-88.0, 41.55, -87.2, 42.15]` -> `lake.geojson` (1,928 km2); (c) that box minus the lake -> `land-mask.geojson`. All snapped to 1e-5 degrees.
- Output (65 KB): the three GeoJSON files, binary columns unprefixed (city), `lake*` and `landMask*` (`Vertices`, `RingOffsets`, `PolygonRingOffsets`, `PartFeature`), `areaKm2`.
- Caveats: the Natural Earth shoreline is generalised (~100 m); it does not follow the real Chicago shoreline, harbours or piers, so it can overlap or leave a gap against the city limit. Along the lake the city limit is the community-area edge.
