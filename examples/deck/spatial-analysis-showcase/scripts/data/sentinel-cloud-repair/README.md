# Sentinel-2 cloud repair pairs

This builder fetches small Cloud Optimized GeoTIFF windows from the public Element 84 Earth Search
Sentinel-2 Level-2A archive. It writes two aligned natural-colour target/prior pairs plus masks to
`public/data/sentinel-cloud-repair/`.

- Waimānalo, Oʻahu: 29 September 2026 target; 3 February 2026 clear prior.
- Venice Lagoon: 25 August 2026 target; 24 August 2026 clear prior.
- Each crop is 512 × 512 at 10 m ground sampling distance.
- The 20 m Scene Classification Layer is nearest-neighbour upsampled and masks classes 1, 3, 7,
  8, 9 and 10, following Copernicus invalid-observation guidance. A two-pixel halo captures mixed
  cloud edges.

From the repository root:

```bash
npm install --no-save geotiff
node examples/deck/spatial-analysis-showcase/scripts/data/sentinel-cloud-repair/build.mjs
```

`pngjs` is already a repository development dependency. `geotiff` is build-only and intentionally
not shipped in the browser application.

Sources:

- Earth Search catalog and public Sentinel COGs: <https://element84.com/earth-search/>
- Sentinel-2 Level-2A/SCL documentation:
  <https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html>
