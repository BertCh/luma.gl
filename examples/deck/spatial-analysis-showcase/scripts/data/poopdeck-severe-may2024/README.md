# poopdeck severe weather, 21-22 May 2024

Five showcase datasets rebuilt from poopdeck.gl STT archives (`https://tiles.poopdeck.gl/data/<id>/manifest.json`)
with `build.mjs`. It needs the sibling `~/Documents/GitHub/poopdeck.gl` checkout (prebuilt
`packages/core/dist`) and network access; nothing is stored outside `public/data`.

```sh
node scripts/data/poopdeck-severe-may2024/build.mjs            # all five
node scripts/data/poopdeck-severe-may2024/build.mjs --only tracks,warnings
```

Window: 2024-05-21T12:00Z to 2024-05-22T06:00Z. Every dataset stores `uint32` seconds since
`2024-05-21T12:00:00Z` (`properties.timeOriginMs` in each manifest), so the scenes combine them
without conversion.

| Dataset | Archive | What the build does | Licence |
| --- | --- | --- | --- |
| `poopdeck-mrms-precip-tracks` | `mrms-precip-tracks` z0 | Hourly pieces (distinct feature ids) are chained by matching end and start vertices; chains under 3 vertices or 15 minutes dropped | public domain |
| `poopdeck-goes-glm-lightning` | `goes-glm-lightning` z5 raw layer | Seeded 150,000-flash sample (the archive's low zooms are summary tiles) | public domain |
| `poopdeck-mrms-storm3d-reports` | `mrms-storm3d-reports` z3 | De-duplicates zoom replicas; kind codes tornado, wind, hail, flood, damage, other | public domain |
| `poopdeck-mrms-storm3d-warnings` | `mrms-storm3d-warnings` z3 | Polygon versions with valid-from and valid-until; versions of one warning chained to get `issueTime` and a `warning` id | public domain |
| `poopdeck-mrms-storm3d-outages` | `mrms-storm3d-outages` z3 | Polygons dropped; one row per county FIPS and 15-minute snapshot (geometry from `us-counties`) | CC BY 4.0 (DOE/ORNL EAGLE-I) |

Not used: `mrms-precip-cells`, `storm4d-*` subsets (smaller duplicates of the same storm), the
2020-08-10 derecho set (`storm-cells`, `storm-tracks`).
