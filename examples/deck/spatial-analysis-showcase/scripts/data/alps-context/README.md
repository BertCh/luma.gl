# alps-context

OpenStreetMap context for the Zermatt / Gornergrat window (lon 7.58 to 7.98, lat 45.86 to 46.12), the same window as `alps-dem-wide`.

- Source: OpenStreetMap via the Overpass API (`https://overpass-api.de/api/interpreter`), one query, `out geom;`, timeout 180, User-Agent `luma-showcase-data-build/1.0 (https://github.com/visgl/luma.gl)`. Server data timestamp (`osm3s.timestamp_osm_base`): 2026-10-07T11:42:41Z.
- Query: `query.overpassql` in this folder (glaciers, peaks, saddles, place / station / hut nodes, railway station and halt ways, aerialway station ways, water, the Gornergratbahn route relation). Save the result as `osm.json` and run `python -I build.py path/to/osm.json`.
- Licence: ODbL 1.0. Attribution: `© OpenStreetMap contributors (ODbL)`.
- Processing: glacier and lake multipolygons rebuilt from outer/inner ways; clipped to the window; simplified 5 m in EPSG:32632 with topology kept; parts < 0.5 ha and holes < 0.1 ha dropped; coordinates rounded to 1e-5 degrees. Rivers, streams and canals are excluded from lakes. Stations mapped as building ways use the way centroid. The railway is the members of the `Gornergratbahn` relation (OSM has no `railway=rack` ways here), merged and clipped; parallel and siding ways remain separate LineStrings. Places keep only named features.
- Output (about 483 KB): `manifest.json` (kind `polygons`, GeoArrow columns for the glaciers, `name`, `areaKm2`, `osmId`, `osmType`) and GeoJSON side files:
  - `glaciers.geojson` 133 features: name, nameDe, areaKm2, osmId
  - `peaks.geojson` 295: name, nameDe, elevationM, prominenceM, osmId
  - `saddles.geojson` 169: name, elevationM, osmId
  - `places.geojson` 365: name, kind (settlement, rail-station, lift-station, hut), elevationM, osmId, place (settlements only)
  - `lakes.geojson` 73: name, areaHa, osmId
  - `railway.geojson` 12 LineStrings: name, osmId
- Caveats: the Gornergrat station is mapped as a building way; Klein Matterhorn lift station is named "Matterhorn Glacier Paradise". Overpass was busy on one attempt and the query was re-run.
