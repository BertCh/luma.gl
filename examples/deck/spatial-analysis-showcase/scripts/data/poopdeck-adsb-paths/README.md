# poopdeck-adsb-paths

One UTC day (Monday 2020-01-06) of jet flights over the contiguous United States, as 3-D trajectories.

- Source: OpenSky Network ADS-B state vectors, read through the poopdeck.gl `flights` archive
  (`https://tiles.poopdeck.gl/data/flights/manifest.json`, zoom 0). The poopdeck `adsb-paths` archive
  holds the same day as 2-D polylines only (no altitude, bounds -124.7..-65, 25..50), so it cannot
  give altitude or ground speed; both archives stop at the edge of the contiguous US, so there is no
  North Atlantic or Europe coverage.
- Licence: OpenSky terms, research and non-commercial use with attribution (not verified here, the
  OpenSky site returned 403 to the fetch tool). Treat as non-commercial.
- Command (needs `~/Documents/GitHub/poopdeck.gl/packages/core/dist/index.js`):

```sh
node scripts/data/poopdeck-adsb-paths/build.mjs \
  --out public/data/poopdeck-adsb-paths \
  --cache <fresh empty dir>
```

- Method: pings grouped by icao24 and sorted by time; pings under 60 m or without altitude are
  ground and split flights; ping gaps over 600 s split a flight; pings that repeat the previous
  position (ADS-B stale carry-forward) are dropped; each flight is simplified with a synchronised
  Euclidean distance (space-time Douglas-Peucker, 10 km tolerance with altitude weighted 4:1 and no
  step longer than 600 s) so ground speed stays measurable; flights shorter than 20 min, with fewer
  than 5 vertices or that never reach 7,500 m are dropped.
- Output: 38,135 flights, 454,800 vertices, 6.97 MB. Columns: `pathOffsets` (u32), `vertices` (f32
  lon, lat), `timestamp` (u32 s since 2020-01-06T00:00:00Z), `altitude` (u16 metres),
  `groundSpeed` (u8, 3-knot units, as reported by the aircraft).
