# poopdeck-osm-nyc

Source: poopdeck.gl archive `osm-nyc-nodes` (https://tiles.poopdeck.gl/data/osm-nyc-nodes/manifest.json), node creations in the
OpenStreetMap full history for the New York City box (Geofabrik internal extract, 2007-06 to 2026-05; 901,827 nodes at the
lowest zoom, one copy each). Licence ODbL 1.0, "© OpenStreetMap contributors".

Steps: `node build.mjs [--raw DIR]`. It runs `../poopdeck/stt-export.mjs` (z8, attrs `kind,uid`) into DIR (default in the OS temp
dir), then:

- ranks contributors by nodes over the full history and ships only the rank (`contributor.bin`, uint16, 0 = most active). User
  names and numeric ids are never written. Ranks over 11,361 contributors are anonymous by construction.
- takes a seeded uniform sample of 400,000 nodes sorted by time (positions float32 lon/lat, `time.bin` uint32 seconds since
  `timeOriginMs` = 2007-06-27T20:05:28Z, `kind.bin` uint8: land other transport infra poi building).
- writes exact full-history aggregates into `manifest.properties`: monthly counts (total and per kind, 228 months from
  2007-06), the share of each month held by its most active contributor, top-200 cumulative contributor shares, Gini.

Size: 6.0 MB (15 bytes per node). `sampleFraction` (0.4435) scales sample counts back to full-history counts.
