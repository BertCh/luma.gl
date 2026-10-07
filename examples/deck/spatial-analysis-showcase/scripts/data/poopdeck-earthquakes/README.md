# poopdeck-earthquakes

Source: USGS Earthquake Catalog (ComCat), magnitude 4.0 and above, 2020-01-01 to 2024-12-30, as served by the
poopdeck.gl archive `earthquakes-v2` (https://tiles.poopdeck.gl/data/earthquakes-v2/manifest.json).
Licence: public domain (US Government work). 77,231 events, about 1.7 MB.

```sh
node scripts/data/poopdeck-earthquakes/build.mjs \
  --out public/data/poopdeck-earthquakes \
  --cache <fresh empty dir under the scratchpad>/poopdeck-earthquakes
```

The build runs `scripts/data/poopdeck/stt-export.mjs` over the whole archive at zoom 0 (the archive's lowest zoom,
where every event appears once) and sorts the rows by time. `timestamp` is `uint32` seconds since
`properties.timeOriginMs` (five years is 1.6e8 s, inside `uint32`; scenes convert to float32 days).
Place names are dropped. Includes a few non-earthquake events (`type`: mining explosion, volcanic eruption, ...).
