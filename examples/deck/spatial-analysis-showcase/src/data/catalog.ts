// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {LocalMetricProjection, getBboxCenter, projectLngLatArray} from '../engine/projection';
import {BUILTIN_DATASETS} from './builtin-datasets';
import type {DatasetInfo} from './dataset-types';
import {
  decodeColumn,
  decodeRasterBinary,
  decodeRasterImage,
  isBinaryRasterEncoding,
  fetchBytes,
  type FetchOptions,
  fetchGeoJson,
  fetchJson,
  fetchText,
  parseCsv,
  getDtypeConstructor,
  getDataFileUrl,
  type ColumnArray,
  type ColumnDtype,
  type DecodedRaster,
  type GeoJsonCollection,
  type RasterEncoding,
  type RasterImageEncoding,
  type UvRanges
} from './loaders';

/** One binary column of a manifest (see `src/data/README.md`). */
export type ManifestColumn = {
  file: string;
  dtype: ColumnDtype;
  /** Interleaved components per row. Defaults to 1. */
  components?: number;
  /** Row count. */
  length: number;
  unit?: string;
  /** Names of dense category indexes, for `uint8` category columns. */
  categories?: string[];
};

/** One raster file of a manifest: `manifest.raster` or an entry of `manifest.rasters`. */
export type RasterSpec = {
  file: string;
  encoding: RasterEncoding;
  width: number;
  height: number;
  /** Stacked slices for `*-bin` encodings (time or band), laid out `[t][row][col]`. Default 1. */
  depth?: number;
  /** `[west, south, east, north]` degrees. */
  bounds: [number, number, number, number];
  noData?: number | null;
  unit?: string;
  /** Multiply raw values by this (reflectance bands use 0.0001). */
  scale?: number;
  /** Added after `scale`. */
  offset?: number;
  /** `rg-uv-8bit` decode ranges; when absent they are read from `properties`. */
  uRange?: [number, number];
  vRange?: [number, number];
  /** Extra georeferencing the data builder recorded, for example `projection: 'EPSG:3857'`. */
  [key: string]: unknown;
};

/** `manifest.json` of a shipped dataset. */
export type DatasetManifest = {
  id: string;
  version: number;
  kind: 'points' | 'polygons' | 'lines' | 'network' | 'trajectories' | 'raster' | 'table' | 'flows';
  count?: number;
  /** `[west, south, east, north]` degrees. */
  bbox: [number, number, number, number];
  crs?: string;
  columns?: Record<string, ManifestColumn>;
  geometry?: {type: string; file: string};
  /** The primary raster. */
  raster?: RasterSpec;
  /** Additional named rasters (for example the bands and masks of `dixie-fire`). */
  rasters?: Record<string, RasterSpec>;
  properties?: Record<string, unknown>;
};

/** Where a loaded dataset came from. */
export type DatasetOrigin = 'bundled' | 'remote' | 'synthetic';

/** A decoded raster with its georeferencing. */
export type LoadedRaster = DecodedRaster & {
  /** `[west, south, east, north]` degrees. */
  bounds: [number, number, number, number];
  noData?: number | null;
  unit?: string;
  /** The raster block of the manifest, including builder-specific georeferencing fields. */
  spec: RasterSpec;
  /** Same as `spec.scale`: `value * scale + offset` converts `*-bin` raw values. */
  scale?: number;
  offset?: number;
};

/** A decoded column. */
export type LoadedColumn = {
  data: ColumnArray;
  /** Interleaved components per row. */
  components: number;
  categories?: string[];
  unit?: string;
};

/** A parsed CSV file of a dataset. */
export type LoadedTable = {
  header: string[];
  rows: string[][];
  /** One object per row, keyed by header name. */
  records: Record<string, string>[];
};

/** Data of a dataset before it is wrapped by {@link LoadedDataset}. */
export type DatasetPayload = {
  manifest: DatasetManifest;
  columns: Record<string, LoadedColumn>;
  geojson: GeoJsonCollection | null;
  raster: LoadedRaster | null;
};

/**
 * A fully loaded dataset: the manifest, its typed-array columns, GeoJSON and raster if any, and
 * helpers that project longitude/latitude columns to planar meters around a local origin (the
 * coordinate frame the GPU contributors and `METER_OFFSETS` layers use).
 */
export class LoadedDataset {
  readonly info: DatasetInfo;
  readonly manifest: DatasetManifest;
  readonly origin: DatasetOrigin;
  /** Columns by name. Prefer {@link column}. */
  readonly columns: Record<string, LoadedColumn>;
  /** Parsed GeoJSON of `manifest.geometry`, or `null`. */
  readonly geojson: GeoJsonCollection | null;
  /** Decoded raster of `manifest.raster`, or `null`. */
  readonly raster: LoadedRaster | null;
  private readonly projections = new Map<string, LocalMetricProjection>();
  private readonly projected = new Map<string, Float32Array>();
  private readonly tablePromises = new Map<string, Promise<LoadedTable>>();
  private readonly rasterPromises = new Map<string, Promise<LoadedRaster>>();

  constructor(info: DatasetInfo, payload: DatasetPayload, origin: DatasetOrigin) {
    this.info = info;
    this.manifest = payload.manifest;
    this.columns = payload.columns;
    this.geojson = payload.geojson;
    this.raster = payload.raster;
    this.origin = origin;
  }

  /** Every extra raster of the manifest (`manifest.rasters`), by name. */
  get rasterSpecs(): Readonly<Record<string, RasterSpec>> {
    return this.manifest.rasters ?? {};
  }

  /**
   * Loads one of the extra rasters declared in `manifest.rasters` (for example `'red_before'` in
   * `dixie-fire`). Memoized per dataset. Only available for shipped (bundled) datasets.
   */
  loadRaster(name: string, signal?: AbortSignal): Promise<LoadedRaster> {
    let promise = this.rasterPromises.get(name);
    if (!promise) {
      const spec = this.rasterSpecs[name];
      if (!spec) {
        return Promise.reject(
          new Error(
            `Dataset "${this.info.id}" has no raster "${name}" (has: ${Object.keys(this.rasterSpecs).join(', ') || 'none'})`
          )
        );
      }
      promise = decodeRasterSpec(this.info.id, spec, this.manifest.properties, signal);
      promise.catch(() => this.rasterPromises.delete(name));
      this.rasterPromises.set(name, promise);
    }
    return promise;
  }

  /**
   * Loads another file with the encoding and size of the primary raster, for frame sequences such
   * as the hourly `gfs-wind` PNGs listed in `properties.wind10m.files`. Not memoized.
   */
  loadRasterFile(file: string, signal?: AbortSignal): Promise<LoadedRaster> {
    const spec = this.manifest.raster;
    if (!spec) return Promise.reject(new Error(`Dataset "${this.info.id}" has no raster`));
    return decodeRasterSpec(this.info.id, {...spec, file}, this.manifest.properties, signal);
  }

  /**
   * Loads a CSV that ships with the dataset and parses it. Without `file` it picks the manifest's
   * `table.file` (`openflights` airports), `attributesCsv` (`chicago-tracts`) or `namesCsv`
   * (`chicago-facilities`). Row order matches the dataset's rows, as the manifest notes.
   * Memoized per file. Only available for shipped (bundled) datasets.
   */
  loadTable(file?: string, signal?: AbortSignal): Promise<LoadedTable> {
    const manifest = this.manifest as DatasetManifest & {
      table?: {file: string};
      attributesCsv?: string;
      namesCsv?: string;
    };
    const resolved = file ?? manifest.table?.file ?? manifest.attributesCsv ?? manifest.namesCsv;
    if (!resolved) {
      return Promise.reject(new Error(`Dataset "${this.info.id}" has no CSV table`));
    }
    let promise = this.tablePromises.get(resolved);
    if (!promise) {
      promise = fetchText(getDataFileUrl(this.info.id, resolved), signal).then(text => {
        const table = parseCsv(text);
        const records = table.rows.map(row =>
          Object.fromEntries(table.header.map((name, index) => [name, row[index] ?? '']))
        );
        return {...table, records};
      });
      promise.catch(() => this.tablePromises.delete(resolved));
      this.tablePromises.set(resolved, promise);
    }
    return promise;
  }

  /** URL of a file in this dataset's folder, for scenes that fetch extra files themselves. */
  fileUrl(file: string): string {
    return getDataFileUrl(this.info.id, file);
  }

  /** Free-form `manifest.properties`. */
  get properties(): Record<string, unknown> {
    return this.manifest.properties ?? {};
  }

  /** Row count (`manifest.count`, else the first column's length). */
  get count(): number {
    if (this.manifest.count !== undefined) return this.manifest.count;
    const first = Object.values(this.columns)[0];
    return first ? first.data.length / first.components : 0;
  }

  /** Whether a column exists. */
  hasColumn(name: string): boolean {
    return name in this.columns;
  }

  /** Returns a column's typed array, typed by the caller. Throws when missing. */
  column<T extends ColumnArray = Float32Array>(name: string): T {
    const column = this.columns[name];
    if (!column) {
      throw new Error(
        `Dataset "${this.info.id}" has no column "${name}" (has: ${Object.keys(this.columns).join(', ')})`
      );
    }
    return column.data as T;
  }

  /** Category names of a column, or an empty list. */
  categories(name: string): readonly string[] {
    return this.columns[name]?.categories ?? [];
  }

  /**
   * Local metric projection. The origin defaults to the bbox center and is shared by every
   * projected column, so positions from different columns of one dataset line up.
   */
  getProjection(origin?: readonly [number, number]): LocalMetricProjection {
    const resolved = origin ?? getBboxCenter(this.manifest.bbox ?? this.info.bbox);
    const key = `${resolved[0]},${resolved[1]}`;
    let projection = this.projections.get(key);
    if (!projection) {
      projection = new LocalMetricProjection(resolved);
      this.projections.set(key, projection);
    }
    return projection;
  }

  /** `[longitude, latitude]` origin used when none is given. */
  get defaultOrigin(): [number, number] {
    return getBboxCenter(this.manifest.bbox ?? this.info.bbox);
  }

  /**
   * Projects a lon/lat column (components 2) to interleaved `x, y` meters around `origin`
   * (default: the bbox center). Memoized per column and origin.
   */
  projectColumn(name = 'position', origin?: readonly [number, number]): Float32Array {
    const resolved = origin ?? this.defaultOrigin;
    const key = `${name}@${resolved[0]},${resolved[1]}`;
    let meters = this.projected.get(key);
    if (!meters) {
      const column = this.columns[name];
      if (!column) throw new Error(`Dataset "${this.info.id}" has no column "${name}"`);
      meters = projectLngLatArray(this.getProjection(resolved), column.data, column.components);
      this.projected.set(key, meters);
    }
    return meters;
  }
}

/** Options of {@link createDataCatalog}. */
export type DataCatalogOptions = {
  /** Skip every network request and use deterministic synthetic data where available. */
  forceSynthetic?: boolean;
};

/** Lazily loads, decodes and memoizes datasets for one session. */
export type DataCatalog = {
  readonly forceSynthetic: boolean;
  /**
   * Loads one dataset by id (memoized). Rejects for an unknown id or a missing file.
   * `onProgress` reports aggregate download progress of the files; a second caller that joins an
   * in-flight load gets the same promise and no progress, and builtin remote/synthetic datasets
   * report only a final update.
   */
  load: (
    id: string,
    signal?: AbortSignal,
    onProgress?: LoadProgressCallback
  ) => Promise<LoadedDataset>;
};

const dataModules = import.meta.glob<{default: DatasetInfo}>('./datasets/*.dataset.ts');

let infoPromise: Promise<DatasetInfo[]> | null = null;

/** Every dataset registered in `src/data/datasets/*.dataset.ts`, sorted by id. */
export function loadDatasetInfos(): Promise<DatasetInfo[]> {
  infoPromise ??= Promise.all(Object.values(dataModules).map(load => load())).then(modules =>
    modules.map(module => module.default).sort((a, b) => a.id.localeCompare(b.id))
  );
  return infoPromise;
}

/** One dataset's info by id. */
export async function loadDatasetInfo(id: string): Promise<DatasetInfo | undefined> {
  return (await loadDatasetInfos()).find(info => info.id === id);
}

/**
 * Checks a column file against its manifest. When the byte size says the file is stored one
 * integer width wider than declared (`uint16` declared, `uint32` on disk), warns and decodes it
 * with the width the bytes imply instead of returning garbage.
 */
function resolveColumnDtype(
  datasetId: string,
  name: string,
  spec: ManifestColumn,
  byteLength: number
): ColumnDtype {
  const values = spec.length * (spec.components ?? 1);
  const declared = getDtypeConstructor(spec.dtype).BYTES_PER_ELEMENT;
  if (values === 0 || byteLength === values * declared) return spec.dtype;
  const wider: Partial<Record<ColumnDtype, ColumnDtype>> = {
    uint8: 'uint16',
    uint16: 'uint32',
    int8: 'int16',
    int16: 'int32'
  };
  const candidate = wider[spec.dtype];
  const message = `Dataset "${datasetId}" column "${name}": ${byteLength} bytes on disk, manifest declares ${values} ${spec.dtype} values`;
  if (candidate && byteLength === values * getDtypeConstructor(candidate).BYTES_PER_ELEMENT) {
    // biome-ignore lint/suspicious/noConsole: surfaces failures that would otherwise be swallowed
    console.warn(`${message}; decoding as ${candidate}. Fix the manifest dtype.`);
    return candidate;
  }
  // biome-ignore lint/suspicious/noConsole: surfaces failures that would otherwise be swallowed
  console.warn(`${message}.`);
  return spec.dtype;
}

/**
 * Fetches and decodes one raster file of a shipped dataset: PNG encodings (`terrarium`, `mapbox`,
 * `uint8-classes`, `rg-uv-8bit`) and binary arrays (`float32-bin`, `uint8-bin`, `uint16-bin`,
 * `int16-bin`, `int32-bin`, `uint32-bin`, optionally a `depth`-slice stack).
 */
export async function decodeRasterSpec(
  datasetId: string,
  spec: RasterSpec,
  properties?: Record<string, unknown>,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<LoadedRaster> {
  const bytes = await fetchBytes(getDataFileUrl(datasetId, spec.file), signalOrOptions);
  const decoded = isBinaryRasterEncoding(spec.encoding)
    ? decodeRasterBinary(bytes, spec.encoding, spec)
    : await decodeRasterImage(
        bytes,
        spec.encoding as RasterImageEncoding,
        spec.encoding === 'rg-uv-8bit' ? findUvRanges(spec, properties) : undefined
      );
  return {
    ...decoded,
    bounds: spec.bounds,
    noData: spec.noData,
    unit: spec.unit,
    spec,
    scale: spec.scale,
    offset: spec.offset
  };
}

/** `uRange` and `vRange` of a spec, or of the `properties` group that lists the spec's file. */
function findUvRanges(spec: RasterSpec, properties: Record<string, unknown> = {}): UvRanges {
  if (spec.uRange && spec.vRange) return {uRange: spec.uRange, vRange: spec.vRange};
  const groups = Object.values(properties).filter(
    (value): value is {files?: string[]; uRange: [number, number]; vRange: [number, number]} =>
      typeof value === 'object' && value !== null && 'uRange' in value && 'vRange' in value
  );
  const group = groups.find(candidate => candidate.files?.includes(spec.file)) ?? groups[0];
  if (!group) throw new Error(`rg-uv-8bit raster "${spec.file}" has no uRange/vRange`);
  return {uRange: group.uRange, vRange: group.vRange};
}

/** Download progress of a dataset load, aggregated over all of its files. */
export type LoadProgress = {
  /** Bytes received so far. */
  loadedBytes: number;
  /** Expected total: `DatasetInfo.approxBytes`, else the sum of reported `Content-Length`s. */
  totalBytes: number | null;
  /** `loadedBytes / totalBytes` clamped to `[0, 1]`, or `null` when the total is unknown. */
  fraction: number | null;
};

/** Receives {@link LoadProgress} updates while a dataset downloads. */
export type LoadProgressCallback = (progress: LoadProgress) => void;

/**
 * Sums per-file progress into one {@link LoadProgress}. `expectedBytes` (the catalog's
 * `approxBytes`) is a stable denominator; without it the reported content lengths are summed.
 */
function createProgressAggregator(
  expectedBytes: number | undefined,
  onProgress: LoadProgressCallback
) {
  const files = new Map<string, {loaded: number; total: number | null}>();
  const report = () => {
    let loadedBytes = 0;
    let reportedTotal: number | null = 0;
    for (const file of files.values()) {
      loadedBytes += file.loaded;
      reportedTotal =
        reportedTotal === null || file.total === null ? null : reportedTotal + file.total;
    }
    const total = expectedBytes && expectedBytes > 0 ? expectedBytes : reportedTotal || null;
    const totalBytes = total === null ? null : Math.max(total, loadedBytes);
    onProgress({
      loadedBytes,
      totalBytes,
      fraction: totalBytes ? Math.min(1, loadedBytes / totalBytes) : null
    });
  };
  return {
    /** Options for one file's fetch; the key identifies the file. */
    track(key: string): Pick<FetchOptions, 'onProgress'> {
      return {
        onProgress(loaded, total) {
          files.set(key, {loaded, total});
          report();
        }
      };
    },
    /** Reports the final state: everything received. */
    finish() {
      let loadedBytes = 0;
      for (const file of files.values()) loadedBytes += file.loaded;
      onProgress({loadedBytes, totalBytes: loadedBytes, fraction: 1});
    }
  };
}

/**
 * Fetches and decodes a shipped dataset described by `public/data/<id>/manifest.json`. With
 * `onProgress`, reports aggregate download progress over the manifest, columns, GeoJSON and
 * primary raster; `expectedBytes` is the denominator when the server sends no `Content-Length`.
 */
export async function loadBundledPayload(
  id: string,
  signal?: AbortSignal,
  onProgress?: LoadProgressCallback,
  expectedBytes?: number
): Promise<DatasetPayload> {
  const aggregator = onProgress && createProgressAggregator(expectedBytes, onProgress);
  const options = (file: string): FetchOptions => ({signal, ...aggregator?.track(file)});
  const manifest = await fetchJson<DatasetManifest>(
    getDataFileUrl(id, 'manifest.json'),
    options('manifest.json')
  );
  const columns: Record<string, LoadedColumn> = {};
  await Promise.all(
    Object.entries(manifest.columns ?? {}).map(async ([name, spec]) => {
      const bytes = await fetchBytes(getDataFileUrl(id, spec.file), options(spec.file));
      columns[name] = {
        data: decodeColumn(bytes, resolveColumnDtype(id, name, spec, bytes.byteLength)),
        components: spec.components ?? 1,
        categories: spec.categories,
        unit: spec.unit
      };
    })
  );
  const geojson = manifest.geometry
    ? await fetchGeoJson(
        getDataFileUrl(id, manifest.geometry.file),
        options(manifest.geometry.file)
      )
    : null;
  const raster = manifest.raster
    ? await decodeRasterSpec(
        id,
        manifest.raster,
        manifest.properties,
        options(manifest.raster.file)
      )
    : null;
  aggregator?.finish();
  return {manifest, columns, geojson, raster};
}

/** Creates the session catalog. */
export function createDataCatalog(options: DataCatalogOptions = {}): DataCatalog {
  const forceSynthetic = Boolean(options.forceSynthetic);
  const memo = new Map<string, Promise<LoadedDataset>>();
  return {
    forceSynthetic,
    load(id, signal, onProgress) {
      let promise = memo.get(id);
      if (!promise) {
        promise = loadDataset(id, forceSynthetic, signal, onProgress);
        // Do not memoize failures: a later scene may retry after the network recovers.
        promise.catch(() => memo.delete(id));
        memo.set(id, promise);
      }
      return promise;
    }
  };
}

async function loadDataset(
  id: string,
  forceSynthetic: boolean,
  signal?: AbortSignal,
  onProgress?: LoadProgressCallback
): Promise<LoadedDataset> {
  const info = await loadDatasetInfo(id);
  if (!info) throw new Error(`Unknown dataset "${id}"`);
  const builtin = BUILTIN_DATASETS[id];
  if (builtin) {
    const done = (dataset: LoadedDataset) => {
      onProgress?.({loadedBytes: info.approxBytes, totalBytes: info.approxBytes, fraction: 1});
      return dataset;
    };
    if (forceSynthetic) return done(new LoadedDataset(info, builtin.synthetic(), 'synthetic'));
    try {
      return done(new LoadedDataset(info, await builtin.load(signal), 'remote'));
    } catch (error) {
      if (signal?.aborted) throw error;
      return done(new LoadedDataset(info, builtin.synthetic(), 'synthetic'));
    }
  }
  return new LoadedDataset(
    info,
    await loadBundledPayload(id, signal, onProgress, info.approxBytes),
    'bundled'
  );
}
