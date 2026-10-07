// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The relief ground of the terrain chapter: paper, warm-lit and cool-shaded multidirectional
 * hillshade, the quiet Alpine tint and pale-blue glaciers, built once on the CPU with
 * `buildReliefImage` (`engine/relief.ts`) and drawn first in the deck canvas as a
 * `SpatialAnalysisRasterLayer`. Every analysis layer of the chapter sits above it.
 *
 * Why the images are kept: a relief is a deterministic product of (DEM, ground tone, recipe), and
 * a theme switch or a return to an earlier step must not cost a second CPU build. The ground
 * therefore keeps the images it has built, keyed by tone and recipe (`getCachedBuildCount` says
 * how many), and builds each at most once per DEM. A new recipe (`rebuild(options)`) is a new
 * build. A few entries are kept (light and dark of the last two recipes); older ones are
 * destroyed after the layer that last drew them has been replaced.
 */

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Device} from '@luma.gl/core';
import type {LoadedDataset} from '../../data/catalog';
import {fetchGeoJson, type GeoJsonCollection} from '../../data/loaders';
import {getInputPolygons, type PolygonRings} from '../../cartography/picking';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {
  buildReliefImage,
  createReliefLayerProps,
  type ReliefImage,
  type ReliefLayerProps,
  type ReliefOptions
} from '../../engine/relief';
import type {TerrainDem} from './cpu-dem';

/** The chapter's relief recipe: every field is optional and the defaults are the Imhof-style ground. */
export type TerrainGroundOptions = {
  /** Five lights spread around the azimuth (weights 1-2-3-2-1). Defaults to `true`. */
  multidirectional?: boolean;
  /** Light azimuth, degrees clockwise from north. Defaults to 315. */
  azimuthDegrees?: number;
  /** Light altitude, degrees. Defaults to 40. */
  altitudeDegrees?: number;
  /** Share of the hypsometric tint multiplied into the shade, 0-1. Defaults to 0.3 (0 = no tint). */
  tintStrength?: number;
  /**
   * Contrast by height (haze on low ground). Defaults to the Alps recipe 1800-4200 m at 0.35;
   * `null` switches it off.
   */
  aerialPerspective?: ReliefOptions['aerialPerspective'];
  /** Paint the glaciers (needs a glacier mask). Defaults to `true`. */
  glacier?: boolean;
  /** Vertical exaggeration. Defaults to 1. */
  zFactor?: number;
  /** Shade scale. Defaults to the relief helper's 1.2. */
  contrast?: number;
};

/** Options of {@link createTerrainGround}. */
export type CreateTerrainGroundOptions = {
  /** The DEM, from `createTerrainDemFromGrid` or `createTerrainDemFromTerrain`. */
  dem: TerrainDem;
  /** The device that owns the relief buffers. */
  device: Device;
  /** The ground tone to start with (`ctx.ground()`). Defaults to `'light'`. */
  ground?: 'light' | 'dark';
  /** Non-zero where a cell is glacier, from {@link rasterizeGlacierMask} or {@link loadGlacierMask}. */
  glacierMask?: Uint8Array | null;
  /** The recipe. Defaults to the chapter recipe. */
  options?: TerrainGroundOptions;
  /** Cells over which the image edge fades out. Defaults to 8. */
  featherCells?: number;
  /** Layer id. Defaults to `'terrain-ground'`. */
  id?: string;
};

/** The relief ground: a layer factory plus the rebuild hooks. */
export type TerrainGround = {
  /** The ground tone currently drawn. */
  readonly ground: 'light' | 'dark';
  /** The recipe currently drawn, defaults filled in. */
  readonly options: Readonly<
    Required<
      Pick<
        TerrainGroundOptions,
        'multidirectional' | 'azimuthDegrees' | 'altitudeDegrees' | 'tintStrength' | 'glacier'
      >
    >
  > &
    TerrainGroundOptions;
  /** CPU images built so far (all tones and recipes). */
  readonly buildCount: number;
  /** Wall-clock milliseconds of the last build. */
  readonly lastBuildMilliseconds: number;
  /**
   * The raster layer that draws the ground, built (synchronously, tens to hundreds of ms) when the
   * current tone and recipe have not been built yet. Draw it first. `opacity` defaults to 1.
   */
  getLayer(layerOptions?: {opacity?: number}): Layer;
  /**
   * Builds the current tone and recipe now, after yielding one task so a "loading" state can
   * paint first. Resolves when the next `getLayer()` is free of CPU work.
   */
  prepare(): Promise<void>;
  /** Switches the tone (`onGroundChange` with `ctx.ground()`). Builds it the first time only. */
  setGround(ground: 'light' | 'dark'): void;
  /**
   * Switches to another recipe (merged over the defaults, not over the previous recipe). The
   * same recipe and tone as an earlier build reuses it; a new one is built.
   */
  rebuild(options?: TerrainGroundOptions): void;
  /** Replaces the glacier mask (or removes it with `null`); every kept image is dropped. */
  setGlacierMask(mask: Uint8Array | null): void;
  /** Destroys every relief buffer. The layer must not be drawn afterwards. */
  destroy(): void;
};

type Entry = {
  key: string;
  props: ReliefLayerProps;
  destroy: () => void;
  lastUsed: number;
};

/** Light and dark of two recipes. */
const MAXIMUM_ENTRIES = 4;

function resolveOptions(options: TerrainGroundOptions | undefined) {
  return {
    ...options,
    multidirectional: options?.multidirectional ?? true,
    azimuthDegrees: options?.azimuthDegrees ?? 315,
    altitudeDegrees: options?.altitudeDegrees ?? 40,
    tintStrength: options?.tintStrength ?? 0.3,
    glacier: options?.glacier ?? true
  };
}

/**
 * Creates the chapter relief ground for a DEM.
 *
 * The relief is `buildReliefImage(dem, {tints: 'alpine', ground, glacierMask, cellSizeMeters,
 * azimuth 315, altitude 40, multidirectional, aerial perspective 1800-4200 m, featherCells 8})`
 * uploaded with `createReliefLayerProps` and drawn by a `SpatialAnalysisRasterLayer` with
 * `coordinateOrigin` = the DEM origin and `METER_OFFSETS`, like the other terrain layers.
 *
 * @example
 * const dem = createTerrainDemFromGrid(grid);
 * const ground = createTerrainGround({dem, device: ctx.device, ground: ctx.ground(), glacierMask});
 * session.setGround(ground);      // TerrainSession draws it when `underlay` is true
 * // onGroundChange: ground.setGround(ctx.ground()); ctx.requestLayers();
 */
export function createTerrainGround(config: CreateTerrainGroundOptions): TerrainGround {
  const {dem, device} = config;
  const layerId = config.id ?? 'terrain-ground';
  let glacierMask = config.glacierMask ?? null;
  let tone: 'light' | 'dark' = config.ground ?? 'light';
  let options = resolveOptions(config.options);
  const entries = new Map<string, Entry>();
  const pendingDestroy: Entry[] = [];
  let lastReturned: Entry | null = null;
  let clock = 0;
  let buildCount = 0;
  let lastBuildMilliseconds = 0;
  let destroyed = false;
  let glacierRevision = 0;

  const getKey = () =>
    `${tone}|${glacierRevision}|${JSON.stringify(options, (_, value) =>
      value === undefined ? null : value
    )}`;

  const build = (): Entry => {
    const key = getKey();
    const existing = entries.get(key);
    if (existing) {
      existing.lastUsed = ++clock;
      return existing;
    }
    const started = performance.now();
    const reliefOptions: ReliefOptions = {
      tints: 'alpine',
      ground: tone,
      azimuthDegrees: options.azimuthDegrees,
      altitudeDegrees: options.altitudeDegrees,
      multidirectional: options.multidirectional,
      tintStrength: options.tintStrength,
      cellSizeMeters: dem.groundCellSize,
      featherCells: config.featherCells ?? 8,
      ...(options.aerialPerspective !== undefined
        ? {aerialPerspective: options.aerialPerspective}
        : {}),
      ...(options.zFactor !== undefined ? {zFactor: options.zFactor} : {}),
      ...(options.contrast !== undefined ? {contrast: options.contrast} : {}),
      ...(options.glacier && glacierMask ? {glacierMask} : {})
    };
    const image: ReliefImage = buildReliefImage(
      {width: dem.width, height: dem.height, values: dem.values, bounds: dem.lngLatBounds},
      reliefOptions
    );
    const props = createReliefLayerProps(device, image, dem.layerBounds, {id: `${layerId}-pixels`});
    // The pixels now live on the GPU; the CPU copy is not kept.
    const entry: Entry = {
      key,
      props,
      lastUsed: ++clock,
      destroy: () => (props.values as {destroy?: () => void}).destroy?.()
    };
    entries.set(key, entry);
    buildCount++;
    lastBuildMilliseconds = performance.now() - started;
    // Keep the most recently used entries; the oldest others are destroyed later (see getLayer).
    while (entries.size > MAXIMUM_ENTRIES) {
      let oldest: Entry | null = null;
      for (const candidate of entries.values()) {
        if (candidate !== entry && (!oldest || candidate.lastUsed < oldest.lastUsed)) {
          oldest = candidate;
        }
      }
      if (!oldest) break;
      entries.delete(oldest.key);
      pendingDestroy.push(oldest);
    }
    return entry;
  };

  const flushPending = () => {
    for (let index = pendingDestroy.length - 1; index >= 0; index--) {
      // The layer deck still holds may use this buffer until it has been replaced.
      if (pendingDestroy[index] === lastReturned) continue;
      pendingDestroy[index].destroy();
      pendingDestroy.splice(index, 1);
    }
  };

  return {
    get ground() {
      return tone;
    },
    get options() {
      return options;
    },
    get buildCount() {
      return buildCount;
    },
    get lastBuildMilliseconds() {
      return lastBuildMilliseconds;
    },
    getLayer(layerOptions = {}) {
      if (destroyed) throw new Error('terrain ground is destroyed');
      flushPending();
      const entry = build();
      lastReturned = entry;
      flushPending();
      return new SpatialAnalysisRasterLayer({
        id: layerId,
        coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
        coordinateOrigin: [dem.origin[0], dem.origin[1], 0],
        ...entry.props,
        opacity: layerOptions.opacity ?? 1
      });
    },
    prepare() {
      return new Promise<void>(resolve => {
        setTimeout(() => {
          if (!destroyed) build();
          resolve();
        }, 0);
      });
    },
    setGround(next) {
      tone = next;
    },
    rebuild(next) {
      options = resolveOptions(next);
    },
    setGlacierMask(mask) {
      glacierMask = mask;
      glacierRevision++;
      for (const entry of entries.values()) pendingDestroy.push(entry);
      entries.clear();
    },
    destroy() {
      destroyed = true;
      for (const entry of entries.values()) entry.destroy();
      for (const entry of pendingDestroy) entry.destroy();
      entries.clear();
      pendingDestroy.length = 0;
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Glacier mask
// ---------------------------------------------------------------------------------------------

/**
 * Rasterises glacier polygons onto the DEM grid: 1 where a cell centre is inside a glacier
 * (even-odd within each polygon, so nunataks stay open; polygons are united), else 0.
 *
 * **Mercator rows.** The DEM rows are Web Mercator, not linear in latitude (across the 0.26 degree
 * wide window a linear latitude mapping misplaces the south edge of a glacier by about 17 m, one
 * cell of the wide DEM). `cartography/masks.ts` `rasterizePolygonMask` is linear in latitude, so it
 * is not used. Instead every ring vertex is converted to pixel coordinates with the DEM's own
 * `getPixelCoordinates` (the grid's Web Mercator mapping) and a small even-odd scanline fills in
 * pixel space, where rows are linear by construction. Polygons outside the DEM cost nothing.
 *
 * @param glaciers GeoJSON polygons (`glaciers.geojson` of the `alps-context` dataset).
 */
export function rasterizeGlacierMask(
  glaciers: GeoJsonCollection,
  dem: Pick<TerrainDem, 'width' | 'height' | 'getPixelCoordinates'>
): Uint8Array {
  const {width, height} = dem;
  const mask = new Uint8Array(width * height);
  for (const {polygon} of getInputPolygons(glaciers)) {
    fillPolygon(mask, width, height, polygon, dem.getPixelCoordinates);
  }
  return mask;
}

/** Even-odd scanline fill of one polygon (outer ring then holes) into a mask, in pixel space. */
function fillPolygon(
  mask: Uint8Array,
  width: number,
  height: number,
  polygon: PolygonRings,
  getPixelCoordinates: (longitude: number, latitude: number) => [number, number]
): void {
  const rings = polygon.map(ring => ring.map(vertex => getPixelCoordinates(vertex[0], vertex[1])));
  let minimumRow = Number.POSITIVE_INFINITY;
  let maximumRow = Number.NEGATIVE_INFINITY;
  for (const ring of rings) {
    for (const [, y] of ring) {
      minimumRow = Math.min(minimumRow, y);
      maximumRow = Math.max(maximumRow, y);
    }
  }
  const firstRow = Math.max(0, Math.ceil(minimumRow - 0.5));
  const lastRow = Math.min(height - 1, Math.floor(maximumRow - 0.5));
  if (!(lastRow >= firstRow)) return;
  const crossings: number[][] = Array.from({length: lastRow - firstRow + 1}, () => []);
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [x0, y0] = ring[j];
      const [x1, y1] = ring[i];
      if (y0 === y1) continue;
      // Rows whose centre y = row + 0.5 lies in [min, max) of the edge.
      const rowStart = Math.max(firstRow, Math.ceil(Math.min(y0, y1) - 0.5));
      const rowEnd = Math.min(lastRow, Math.ceil(Math.max(y0, y1) - 0.5) - 1);
      for (let row = rowStart; row <= rowEnd; row++) {
        crossings[row - firstRow].push(x0 + ((row + 0.5 - y0) / (y1 - y0)) * (x1 - x0));
      }
    }
  }
  crossings.forEach((xs, offset) => {
    xs.sort((a, b) => a - b);
    const row = firstRow + offset;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const columnStart = Math.max(0, Math.ceil(xs[k] - 0.5));
      const columnEnd = Math.min(width - 1, Math.ceil(xs[k + 1] - 0.5) - 1);
      if (columnEnd >= columnStart) {
        mask.fill(1, row * width + columnStart, row * width + columnEnd + 1);
      }
    }
  });
}

/**
 * Fetches `glaciers.geojson` of the `alps-context` dataset (a scene lists
 * `{id: 'alps-context', role: 'glaciers, peaks, places'}` in its `datasets`) and rasterises it onto
 * the DEM grid with {@link rasterizeGlacierMask}. Pass a URL string instead of the dataset to read
 * another file. When the scene also needs the glacier polygons for labels, load them once with
 * `loadAlpsContext` (`terrain-places.ts`) and call `rasterizeGlacierMask` on those.
 */
export async function loadGlacierMask(
  source: LoadedDataset | string,
  dem: Pick<TerrainDem, 'width' | 'height' | 'getPixelCoordinates'>,
  signal?: AbortSignal
): Promise<Uint8Array> {
  const url = typeof source === 'string' ? source : source.fileUrl('glaciers.geojson');
  const glaciers = await fetchGeoJson(url, signal);
  return rasterizeGlacierMask(glaciers, dem);
}
