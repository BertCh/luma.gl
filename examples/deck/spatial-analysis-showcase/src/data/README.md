# Showcase data

Every dataset has two parts: a **catalog entry** (`src/data/datasets/<id>.dataset.ts`) and the
**bytes** (`public/data/<id>/manifest.json` plus the files it names). Datasets are discovered with
`import.meta.glob('./datasets/*.dataset.ts')`; nothing else needs editing to add one.

## Catalog entry

```ts
import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-nature',                         // kebab-case, equals the folder under public/data/
  title: 'Chicago nature observations, 2023',
  description: 'One or two sentences.',
  license: 'CC0 / CC BY 4.0 / CC BY-NC 4.0 (per observation)',
  attribution: 'iNaturalist contributors',
  sourceUrl: 'https://www.inaturalist.org/observations?place_id=49906',
  approxBytes: 4_500_000,
  bbox: [-87.94, 41.64, -87.52, 42.02]          // [west, south, east, north]
} satisfies DatasetInfo;
```

The `#/data` page lists every entry with licence, size, bbox, source and the stories that use it.
Each scene that declares the dataset shows its attribution in the "Data" section of the panel.

## Shipped bytes: `manifest.json`

```json
{
  "id": "chicago-nature",
  "version": 1,
  "kind": "points | polygons | lines | network | trajectories | raster | table | flows",
  "count": 43557,
  "bbox": [minLng, minLat, maxLng, maxLat],
  "crs": "EPSG:4326",
  "columns": {
    "position":  {"file": "position.bin", "dtype": "float32", "components": 2, "length": 43557},
    "timestamp": {"file": "time.bin", "dtype": "uint32", "length": 43557, "unit": "seconds since ..."},
    "category":  {"file": "category.bin", "dtype": "uint8", "length": 43557, "categories": ["Plants", "Birds"]}
  },
  "geometry": {"type": "polygons", "file": "tracts.geojson"},
  "raster": {"file": "dem.png", "encoding": "terrarium | mapbox | uint8-classes | rg-uv-8bit | float32-bin | uint8-bin | uint16-bin | int16-bin",
             "width": 1024, "height": 1024, "bounds": [w, s, e, n], "noData": -9999, "unit": "m"},
  "properties": {"free-form": "metadata such as field descriptions"}
}
```

- **Binary columns** are raw little-endian typed arrays with no header. `dtype` is one of
  `float32 float64 int32 uint32 int16 uint16 uint8 int8`. `components` greater than 1 means
  interleaved rows (`position` is `lng, lat, lng, lat, ...`). Coordinates are longitude and latitude
  degrees; the loader projects them to local meters on request.
- **Variable-length geometry** uses GeoArrow-style offset columns (`uint32`, length n+1) that index a
  vertex column: polygons `ringOffsets`, `polygonRingOffsets`, `vertices`; lines and trajectories
  `pathOffsets` plus `vertices` (plus per-vertex `timestamp`); networks `nodes`, `edgeSource`,
  `edgeTarget`, `edgeLength`, `edgeClass`; flows `origin`, `destination`, `count`.
- **GeoJSON** is fine for small polygon sets (`geometry.file`).
- **Rasters** (`raster`, plus extra named ones in `rasters`) are PNGs (`uint8-classes`, `terrarium`,
  `mapbox`, `rg-uv-8bit` wind with `uRange`/`vRange` in the spec or in the `properties` group that
  lists the file) or headerless arrays (`float32-bin`, `uint8-bin`, `uint16-bin`, `int16-bin`,
  `int32-bin`, `uint32-bin`). A `*-bin` raster may have a `depth` (time or band stack, laid out
  `[t][row][col]`, row 0 north); the loader checks `width * height * depth` against the file size.
  `scale`, `offset` and `noData` are passed through (`raster.scale`, `raster.offset`,
  `raster.noData`) but not applied: raw values keep their stored dtype. Extra georeferencing
  fields (for example `projection`) stay on `dataset.raster.spec`.
  `dataset.loadRaster('red_before')` loads an entry of `manifest.rasters`; `dataset.loadRasterFile(file)`
  loads another file with the primary raster's encoding (frame sequences).
- **Tables**: `dataset.loadTable()` fetches and parses the manifest's CSV (`table.file`,
  `attributesCsv`, `namesCsv`) into `{header, rows, records}`.
- A column whose file size says it is stored one integer width wider than the manifest dtype is
  decoded with the wider dtype and a console warning (the manifest should be fixed).

Column names are the dataset's own: look at the manifest of the dataset you use.

## Loading in a scene

Declare `datasets: [{id: 'chicago-nature', role: 'observation points'}]` in the scene. The shell loads them
before `create` and hands them over:

```ts
const nature = ctx.datasets.get('chicago-nature');   // LoadedDataset
const origin = nature.defaultOrigin;                  // bbox center, [lng, lat]
const meters = nature.projectColumn('position');      // Float32Array of x, y meters around origin
const category = nature.column<Uint8Array>('category');
const names = nature.categories('category');
const tracts = nature.geojson;                        // parsed GeoJSON, or null
const dem = nature.raster;                            // {width, height, values, bounds, unit, spec}, or null
```

Use `nature.defaultOrigin` as the layers' `coordinateOrigin`. `projectColumn` is memoized per column
and origin; pass an explicit origin to put two datasets in one frame:
`a.projectColumn('position', origin)`.

`LoadedDataset` has `info`, `manifest`, `origin` (`'bundled' | 'remote' | 'synthetic'`), `count`,
`properties`, `columns`, `column()`, `hasColumn()`, `categories()`, `getProjection()`.

## Helpers (`loaders.ts`)

`fetchJson`, `fetchText`, `fetchBytes`, `fetchGeoJson`, `fetchWithFallback(urls, parse)` (first
working URL wins), `parseCsv` + `csvColumnToFloat32`, `decodeColumn`, and
`decodeRasterImage(bytes, 'terrarium' | 'mapbox' | 'uint8-classes' | 'rg-uv-8bit', ranges?)`,
`decodeRasterBinary(bytes, 'uint16-bin' | ..., {width, height, depth})`. `getDataFileUrl(id, file)` gives
the URL of a shipped file and respects Vite's `base`.

A per-session memo cache in the catalog means two scenes that use one dataset load it once.

## Synthetic data

`?data=synthetic` makes the exemplar `ny-taxi-trips` and `sf-bike-parking` datasets use deterministic
generators (`builtin-datasets.ts`) so the site works offline and in headless captures. Shipped
datasets under `public/data` are local files and do not need a fallback. If a dataset is fetched from a
remote host, add a builtin loader with a synthetic generator.

## Build scripts

`scripts/data/<dataset-id>/` holds the re-runnable script that produces `public/data/<dataset-id>/`
and a short README with the source, licence and steps. Keep each dataset within its budget.
