# poopdeck-nyc-taxi-paths

9,000 taxi trips routed by OSRM that were on the road between 08:00 and 08:30 on Friday 2 January 2015
(local time stored as UTC). Path pieces are clipped at the window start and end, so a trip that started at
07:50 appears from 08:00.

- Source: poopdeck.gl archive `nyc-taxi-paths`, `https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json`
  (NYC TLC January 2015 yellow-taxi records, routed with OSRM on OpenStreetMap data).
- Licence: NYC Open Data terms; OSM geometry ODbL, "© OpenStreetMap contributors".
- Command: `scripts/data/poopdeck-nyc-taxi-paths/build.sh` (wraps `poopdeck/stt-export.mjs` with `--stitch
  trip_id`; 17,797 trips are in the window and a seeded 9,000 are kept).
- Time: `timestamp` is `uint32` seconds since `properties.timeOriginMs` (07:59:00).
- Caveats: routes and per-vertex times are derived by a routing engine between the recorded endpoints; they
  are not GPS traces, so speeds describe the road network model, not observed traffic.
