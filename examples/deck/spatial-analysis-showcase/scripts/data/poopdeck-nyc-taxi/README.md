# poopdeck-nyc-taxi

440,000 yellow-taxi trips as origin and destination pairs, 1 January 2015 00:00 to 2 January 2015 about 15:00
(local New York time, stored as if UTC).

- Source: poopdeck.gl archive `nyc-taxi-paths` (500,000 OSRM-routed trips from the NYC TLC January 2015
  yellow-taxi records). `https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json`.
- Licence: NYC Open Data terms (TLC records); route endpoints from OSRM on OpenStreetMap data (ODbL,
  "© OpenStreetMap contributors").
- Why this archive: the `nyc-rideshare` archive holds only 50,000 trips (a trip is a pickup, a dozen
  `enroute` fixes and a dropoff, with a trip id that repeats every 50,000), so it cannot give 300k+ pairs.
  `nyc-taxi-paths` holds 500,000 trips; the first and last vertex of each path are the pickup and dropoff.
- Command (needs `~/Documents/GitHub/poopdeck.gl` built; takes about two minutes and downloads about 600 MB):

  ```sh
  node scripts/data/poopdeck-nyc-taxi/build.mjs --out public/data/poopdeck-nyc-taxi \
    --raw <scratch>/trips-cache.json --max-trips 440000
  ```

  `--raw` caches the per-trip first and last vertices so the sampling step can be re-run offline.
- Filter: both ends inside -74.05..-73.70 / 40.60..40.90, duration 0 to 3.5 h, distance 0.05 to 40 miles,
  fare 0 to 250 USD. 499,805 of 500,000 trips pass; a seeded sample of 440,000 fits the 6 MB budget.
- Format (`manifest.json`, kind `flows`): `origin` and `destination` are interleaved `uint16` pairs across
  `properties.quantBbox` (about 0.5 m resolution); `pickupTime` is `uint16` in 4 s steps since
  `properties.timeOriginMs`; `duration` is `uint8` in 15 s steps; `distance` `uint8` in 0.1 mile; `fare`
  `uint8` in 0.5 USD; `passengers` `uint8`.
- Caveats: yellow taxis only, before ride-hail dominated. Dropoff time is the OSRM-derived route time,
  not the metered one. The archive stops mid-afternoon on 2 January, so the timeline is 38 hours, not a
  full cycle of days.
