// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Typed-array dtypes of binary columns. */
export type ColumnDtype =
  | 'float32'
  | 'float64'
  | 'int32'
  | 'uint32'
  | 'int16'
  | 'uint16'
  | 'uint8'
  | 'int8';

/** Any typed array a column can decode to. */
export type ColumnArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Uint8Array
  | Int8Array;

/** Raster encodings stored as PNG pixels; decoded by {@link decodeRasterImage}. */
export type RasterImageEncoding = 'terrarium' | 'mapbox' | 'uint8-classes' | 'rg-uv-8bit' | 'rgba8';

/** Raster encodings stored as headerless little-endian arrays; decoded by {@link decodeRasterBinary}. */
export type RasterBinaryEncoding =
  | 'float32-bin'
  | 'uint8-bin'
  | 'uint16-bin'
  | 'int16-bin'
  | 'int32-bin'
  | 'uint32-bin';

/** Every raster encoding a manifest can declare. */
export type RasterEncoding = RasterImageEncoding | RasterBinaryEncoding;

/** Whether `encoding` is stored as a binary array rather than a PNG. */
export function isBinaryRasterEncoding(encoding: string): encoding is RasterBinaryEncoding {
  return encoding.endsWith('-bin');
}

/** Base URL of shipped data: `<base>data/<dataset-id>/<file>`. */
export function getDataFileUrl(datasetId: string, file: string): string {
  return `${import.meta.env.BASE_URL}data/${datasetId}/${file}`;
}

/** Options of the fetch helpers. */
export type FetchOptions = {
  /** Aborts the request. */
  signal?: AbortSignal;
  /**
   * Called as the body streams in. `total` is the `Content-Length` when the server sent one
   * (`null` otherwise; with transfer compression `loaded` can exceed it). Setting it makes the
   * helper read the body as a stream.
   */
  onProgress?: (loaded: number, total: number | null) => void;
};

/** Normalizes the legacy `signal` argument of the fetch helpers to options. */
function toFetchOptions(signalOrOptions?: AbortSignal | FetchOptions): FetchOptions {
  if (!signalOrOptions) return {};
  return 'aborted' in signalOrOptions ? {signal: signalOrOptions} : signalOrOptions;
}

/** Fetches a URL and throws on a non-OK status. */
export async function fetchChecked(
  url: string,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<Response> {
  const response = await fetch(url, {signal: toFetchOptions(signalOrOptions).signal});
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

/** Fetches JSON. */
export async function fetchJson<T>(
  url: string,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<T> {
  const options = toFetchOptions(signalOrOptions);
  if (options.onProgress) return JSON.parse(await fetchText(url, options)) as T;
  return (await (await fetchChecked(url, options)).json()) as T;
}

/** Fetches text. */
export async function fetchText(
  url: string,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<string> {
  const options = toFetchOptions(signalOrOptions);
  if (options.onProgress) return new TextDecoder().decode(await fetchBytes(url, options));
  return (await fetchChecked(url, options)).text();
}

/**
 * Fetches raw bytes. With `onProgress` the body is read as a stream so progress can be reported
 * (falls back to `arrayBuffer()` when the response has no readable body). The callback always
 * fires once more with the final size.
 */
export async function fetchBytes(
  url: string,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<ArrayBuffer> {
  const options = toFetchOptions(signalOrOptions);
  const response = await fetchChecked(url, options);
  const {onProgress} = options;
  const reader = onProgress ? response.body?.getReader() : undefined;
  if (!onProgress || !reader) return response.arrayBuffer();
  const header = response.headers.get('content-length');
  const total = header !== null && Number.isFinite(Number(header)) ? Number(header) : null;
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress(loaded, total ?? loaded);
  return bytes.buffer;
}

/**
 * Tries each URL in order and returns the first successful `parse` result. Use it for a CDN mirror
 * list or a bundled copy behind a remote original; throws the last error when all fail.
 */
export async function fetchWithFallback<T>(
  urls: readonly string[],
  parse: (response: Response, url: string) => Promise<T>,
  signal?: AbortSignal
): Promise<T> {
  let lastError: unknown = new Error('No URLs given');
  for (const url of urls) {
    try {
      return await parse(await fetchChecked(url, signal), url);
    } catch (error) {
      if (signal?.aborted) throw error;
      lastError = error;
    }
  }
  throw lastError;
}

/** Fetches a GeoJSON FeatureCollection (or a single Feature, wrapped). */
export async function fetchGeoJson(
  url: string,
  signalOrOptions?: AbortSignal | FetchOptions
): Promise<GeoJsonCollection> {
  const json = await fetchJson<GeoJsonCollection | GeoJsonFeature>(url, signalOrOptions);
  return json.type === 'FeatureCollection' ? json : {type: 'FeatureCollection', features: [json]};
}

/** Minimal GeoJSON geometry. */
export type GeoJsonGeometry = {type: string; coordinates: unknown};
/** Minimal GeoJSON feature. */
export type GeoJsonFeature = {
  type: 'Feature';
  id?: string | number;
  properties: Record<string, unknown> | null;
  geometry: GeoJsonGeometry | null;
};
/** Minimal GeoJSON feature collection. */
export type GeoJsonCollection = {type: 'FeatureCollection'; features: GeoJsonFeature[]};

/** Returns the typed-array constructor for a dtype. */
export function getDtypeConstructor(dtype: ColumnDtype) {
  switch (dtype) {
    case 'float32':
      return Float32Array;
    case 'float64':
      return Float64Array;
    case 'int32':
      return Int32Array;
    case 'uint32':
      return Uint32Array;
    case 'int16':
      return Int16Array;
    case 'uint16':
      return Uint16Array;
    case 'uint8':
      return Uint8Array;
    case 'int8':
      return Int8Array;
    default:
      throw new Error(`Unsupported dtype "${String(dtype)}"`);
  }
}

/** Decodes little-endian bytes (no header) into a typed array. Copies when misaligned. */
export function decodeColumn(bytes: ArrayBuffer, dtype: ColumnDtype): ColumnArray {
  const Constructor = getDtypeConstructor(dtype);
  const length = Math.floor(bytes.byteLength / Constructor.BYTES_PER_ELEMENT);
  return new Constructor(bytes, 0, length);
}

/**
 * Parses CSV text with a header row. Handles quoted fields, escaped quotes and CRLF. Values stay
 * strings; use {@link csvColumnToFloat32} for numeric columns.
 */
export function parseCsv(text: string): {header: string[]; rows: string[][]} {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index++;
        } else {
          quoted = false;
        }
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ',') {
      row.push(field);
      field = '';
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && text[index + 1] === '\n') index++;
      row.push(field);
      field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else {
      field += character;
    }
  }
  if (field !== '' || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  const header = rows.shift() ?? [];
  return {header, rows};
}

/** Extracts one CSV column as float32 (`NaN` for blanks and non-numbers). */
export function csvColumnToFloat32(
  table: {header: string[]; rows: string[][]},
  name: string
): Float32Array {
  const index = table.header.indexOf(name);
  if (index < 0) throw new Error(`CSV has no column "${name}"`);
  const values = new Float32Array(table.rows.length);
  table.rows.forEach((row, rowIndex) => {
    const number = Number.parseFloat(row[index]);
    values[rowIndex] = Number.isFinite(number) ? number : Number.NaN;
  });
  return values;
}

/** A decoded raster: row 0 at the north edge. */
export type DecodedRaster = {
  width: number;
  height: number;
  /**
   * Pixel values, `width * height * depth * bands` long.
   *
   * - `terrarium`, `mapbox`: elevations in meters (`Float32Array`).
   * - `uint8-classes`: class ids (`Uint8Array`).
   * - `rgba8`: interleaved red, green, blue and alpha bytes (`Uint8Array`).
   * - `rg-uv-8bit`: decoded `[u, v]` pairs in the manifest unit (`Float32Array`, `bands` is 2).
   * - `*-bin`: the raw array in its stored dtype. Apply `scale` and `offset` yourself
   *   (`value * scale + offset`), and treat `noData` as missing. Time stacks are `[t][row][col]`.
   */
  values: Float32Array | Uint8Array | Uint16Array | Int16Array | Int32Array | Uint32Array;
  encoding: RasterEncoding;
  /** Slices of a time or band stack (`1` for a single layer). */
  depth: number;
  /** Interleaved values per pixel (`4` for `rgba8`, `2` for `rg-uv-8bit`, otherwise `1`). */
  bands: number;
};

/** Value ranges that map an 8-bit `rg-uv-8bit` pixel back to `u` and `v`. */
export type UvRanges = {uRange: readonly [number, number]; vRange: readonly [number, number]};

/**
 * Decodes the pixels of a PNG into RGBA bytes, elevations (Terrarium, Mapbox RGB), 8-bit classes,
 * or wind vectors (`rg-uv-8bit`, which needs `ranges`: `u = uMin + R/255 * (uMax - uMin)`, likewise
 * `v` from G).
 */
export async function decodeRasterImage(
  bytes: ArrayBuffer,
  encoding: RasterImageEncoding,
  ranges?: UvRanges
): Promise<DecodedRaster> {
  const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none'
  });
  const {width, height} = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', {willReadFrequently: true});
  if (!context) throw new Error('2D canvas unavailable for raster decoding');
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height).data;
  const count = width * height;
  if (encoding === 'rgba8') {
    return {
      width,
      height,
      values: new Uint8Array(pixels),
      encoding,
      depth: 1,
      bands: 4
    };
  }
  if (encoding === 'uint8-classes') {
    const classes = new Uint8Array(count);
    for (let index = 0; index < count; index++) classes[index] = pixels[index * 4];
    return {width, height, values: classes, encoding, depth: 1, bands: 1};
  }
  if (encoding === 'rg-uv-8bit') {
    if (!ranges)
      throw new Error('rg-uv-8bit needs uRange and vRange (manifest raster or properties)');
    const uv = new Float32Array(count * 2);
    const [uMin, uMax] = ranges.uRange;
    const [vMin, vMax] = ranges.vRange;
    for (let index = 0; index < count; index++) {
      uv[index * 2] = uMin + (pixels[index * 4] / 255) * (uMax - uMin);
      uv[index * 2 + 1] = vMin + (pixels[index * 4 + 1] / 255) * (vMax - vMin);
    }
    return {width, height, values: uv, encoding, depth: 1, bands: 2};
  }
  const elevation = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const red = pixels[index * 4];
    const green = pixels[index * 4 + 1];
    const blue = pixels[index * 4 + 2];
    elevation[index] =
      encoding === 'terrarium'
        ? red * 256 + green + blue / 256 - 32768
        : -10000 + (red * 65536 + green * 256 + blue) * 0.1;
  }
  return {width, height, values: elevation, encoding, depth: 1, bands: 1};
}

const BINARY_DTYPES: Record<RasterBinaryEncoding, ColumnDtype> = {
  'float32-bin': 'float32',
  'uint8-bin': 'uint8',
  'uint16-bin': 'uint16',
  'int16-bin': 'int16',
  'int32-bin': 'int32',
  'uint32-bin': 'uint32'
};

/**
 * Decodes a headerless little-endian raster array. `depth` is the number of stacked slices (time
 * or band, `[t][row][col]`). Throws when the byte length does not match `width * height * depth`.
 */
export function decodeRasterBinary(
  bytes: ArrayBuffer,
  encoding: RasterBinaryEncoding,
  size: {width: number; height: number; depth?: number}
): DecodedRaster {
  const dtype = BINARY_DTYPES[encoding];
  if (!dtype) throw new Error(`Unsupported raster encoding "${encoding}"`);
  const depth = size.depth ?? 1;
  const values = decodeColumn(bytes, dtype) as DecodedRaster['values'];
  const expected = size.width * size.height * depth;
  if (values.length !== expected) {
    throw new Error(
      `Raster ${encoding}: expected ${size.width}x${size.height}x${depth} = ${expected} values, got ${values.length}`
    );
  }
  return {width: size.width, height: size.height, values, encoding, depth, bands: 1};
}
