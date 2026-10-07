// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainVectorRuggednessParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainRuggedness,
  GPUTerrainVectorRuggedness
} from '@luma.gl/experimental/gpu-terrain';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {ALPS} from '../../cartography/gazetteer';
import type {MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAlpsGrid} from './b14a-grid';
import {ColorRasterLayer} from './b14a-layers';
import {TerrainSession, type ProductBuild, type ValueStats} from './b14a-session';
import {createDemProbe, createTerrainDemFromGrid} from './cpu-dem';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {demSampleLine} from './terrain-furniture';
import {
  ASPECT_RAMP,
  ASPECT_SLOPE_ALPHA,
  ELEVATION_BREAKS,
  SLOPE_BREAKS,
  SLOPE_CONTINUOUS
} from './terrain-palettes';
import {glacierLabels, loadAlpsContext, snapLngLatToHighestCell} from './terrain-places';
import {
  createElevationPipeline,
  getMissingPatchBounds,
  type ElevationPipeline
} from './terrain-basics-elevation';
import {
  formatMillions,
  getDataEdgeFrame,
  getPatchOutline,
  getSteepestNote,
  getSummitNote
} from './terrain-basics-labels';
import {
  addPaintPass,
  getPaintParameterValues,
  PAINT_PARAMETER_LENGTH,
  type PaintSettings
} from './terrain-basics-paint';
import {
  getAspectRoseChart,
  getBasicsTooltip,
  getShareFromClass,
  getSlopeHistogramChart,
  summarizeAspect,
  summarizeElevation,
  summarizeSlope,
  type AspectSummary,
  type HoverCell,
  type SlopeSummary
} from './terrain-basics-summary';
import {
  ASPECT_ALPHA,
  DECISION_SLOPE_DEGREES,
  getCartoucheSubtitle,
  getHatchColor,
  getWindowMeters,
  makeBasicsTables,
  PRODUCT_ALPHA,
  RUGGEDNESS_PROBABILITIES,
  STEEP_FACE_DEGREES,
  type BasicsOptions,
  type BasicsProduct,
  type BasicsTables,
  type RuggednessBreaks
} from './terrain-basics.style';

export type {BasicsOptions, BasicsProduct};

const REBUILD_DELAY_MILLISECONDS = 220;
/** Class index of the first class at or above a slope threshold (the breaks are the thresholds). */
const DECISION_CLASS = SLOPE_BREAKS.indexOf(DECISION_SLOPE_DEGREES) + 1;
const STEEP_CLASS = SLOPE_BREAKS.indexOf(STEEP_FACE_DEGREES) + 1;
/** The colorize pass of the session is unused here (the paint pass writes the colours). */
const UNUSED_PAINT = {mode: 'ramp', ramp: 'grayscale', low: 0, high: 1, alpha: 0} as const;

/** What a product leaves behind: its build, the colours it paints and the buffers worth reading. */
type Entry = {
  /** `'decoded'` or the product. */
  product: BasicsProduct | 'decoded';
  configKey: string;
  /** The build shown for the ground cell model (and every other product). */
  build: ProductBuild;
  /** The Mercator-pixel slope build that shares the slope stage. */
  mercatorBuild: ProductBuild | null;
  /** Packed RGBA8 cells the layer draws. */
  paintBuffer: Buffer;
  /** Raw rasters for the CPU summaries, by name. */
  buffers: Record<string, Buffer>;
  /** Run count of the stage when its summary was last requested. */
  seenRunCount: number;
  /** Summary version when it was last requested. */
  seenSummaryVersion: number;
};

type BulkJob = {
  key: string;
  buffer: Buffer;
  accept: (values: Float32Array) => void;
};

function getConfigKey(state: BasicsOptions, product: BasicsProduct): string {
  switch (product) {
    case 'slope':
    case 'aspect':
      return state.borderMode;
    case 'tpi':
      return state.edgeMode;
    case 'tri':
      return `${state.triAlgorithm}|${state.edgeMode}`;
    case 'vrm':
      return `${state.vrmRadius}|${state.borderMode}`;
  }
}

/**
 * Terrain basics on the Matterhorn tile. Every step is GPU work on one 2048 x 2048 raster:
 *
 * 1. `GPUTerrainRGBDecode` turns the Terrarium PNG into float32 heights (with `GPUTerrainSpikeRepair`
 *    as an expert sub-block); see `terrain-basics-elevation.ts`.
 * 2. One graph per product (`GPUTerrainDerivatives`, `GPUTerrainRuggedness`,
 *    `GPUTerrainVectorRuggedness`) reads the heights, and a paint pass in the same graph classes
 *    the result into packed colours (`terrain-basics-paint.ts`).
 * 3. The relief ground (`terrain-ground.ts`) is drawn under the colours; the chapter ground is a
 *    CPU product of the same DEM.
 *
 * Graphs are compiled the first time a product is shown, and again only when a compile-time option
 * of that product changes. Class tables, the cell model and ground changes are parameter writes.
 */
export async function createTerrainBasics(
  ctx: SceneContext<BasicsOptions>
): Promise<SceneInstance<BasicsOptions>> {
  const {device} = ctx;
  ctx.setStatus('Loading the elevation tile and its OpenStreetMap context');
  const [grid, context] = await Promise.all([
    loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal),
    loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const dem = createTerrainDemFromGrid(grid);
  const demProbe = createDemProbe(dem);
  const glacierMask = rasterizeGlacierMask(context.glacierGeoJson, dem);
  const quantum = Number(
    (ctx.datasets.get('alps-dem').raster?.spec as {verticalQuantisationM?: number} | undefined)
      ?.verticalQuantisationM ?? 0.25
  );

  const resources = new SpatialAnalysisResources(device, 'terrain-basics');
  const session = new TerrainSession(ctx, resources, grid);
  const ground = createTerrainGround({dem, device, ground: ctx.ground(), glacierMask});
  session.setGround(ground);
  ctx.setStatus('Building the shaded relief');
  await ground.prepare();
  ctx.signal.throwIfAborted();

  const pipeline: ElevationPipeline = createElevationPipeline(ctx, resources, session, grid);
  const cellMeters = grid.groundCellSize;
  const mercatorMeters = grid.mercatorCellSize;
  const cellSettings = grid.cellSettings;
  const glacierAnnotations = glacierLabels(context, {
    window: grid.lngLatBounds,
    names: ['Gornergletscher'],
    max: 1
  });

  // --- Class tables, frozen ruggedness breaks and the state they depend on --------------------
  /** Quantile breaks of TRI and VRM, measured once per product and configuration, then frozen. */
  const frozenBreaks = new Map<string, number[]>();
  const getFrozenKey = (state: BasicsOptions): string | null =>
    state.product === 'tri' || state.product === 'vrm'
      ? `${state.product}|${getConfigKey(state, state.product)}`
      : null;
  const getRuggedness = (): RuggednessBreaks | null => {
    const state = ctx.options;
    const key = getFrozenKey(state);
    const breaks = key ? frozenBreaks.get(key) : undefined;
    return breaks && (state.product === 'tri' || state.product === 'vrm')
      ? {product: state.product, breaks}
      : null;
  };
  let tables: BasicsTables = makeBasicsTables(ctx.ground(), getRuggedness());
  const refreshTables = (): void => {
    const ruggedness = getRuggedness();
    tables = makeBasicsTables(ctx.ground(), ruggedness);
    ctx.setLegendData('tables', tables);
    ctx.setLegendData('ruggednessBreaks', ruggedness);
  };

  const getPaintSettings = (kind: BasicsProduct | 'decoded'): PaintSettings => {
    const state = ctx.options;
    switch (kind) {
      case 'decoded':
        return {mode: 'classes', breaks: ELEVATION_BREAKS, colors: tables.decoded.colors};
      case 'slope': {
        const useSecond = state.cellModel === 'mercator';
        if (state.slopeDisplay === 'continuous') {
          return {
            mode: 'ramp',
            ramp: SLOPE_CONTINUOUS.ramp,
            low: SLOPE_CONTINUOUS.domain[0],
            high: SLOPE_CONTINUOUS.domain[1],
            floor: SLOPE_CONTINUOUS.transparentBelowDegrees,
            alpha: PRODUCT_ALPHA,
            useSecond
          };
        }
        return {
          mode: 'classes',
          breaks: tables.slope.breaks,
          colors: tables.slope.colors,
          useSecond
        };
      }
      case 'aspect':
        return {
          mode: 'aspect',
          ramp: ASPECT_RAMP,
          alpha: ASPECT_ALPHA,
          flatDegrees: ASPECT_SLOPE_ALPHA.flatDegrees,
          fullDegrees: ASPECT_SLOPE_ALPHA.fullDegrees
        };
      case 'tpi':
        return {mode: 'classes', breaks: tables.tpi.breaks, colors: tables.tpi.colors};
      case 'tri':
      case 'vrm':
        // Until the tile is measured the classes are not known: draw nothing, not a wrong map.
        return getRuggedness()
          ? {mode: 'classes', breaks: tables.rugged.breaks, colors: tables.rugged.colors}
          : {mode: 'classes', breaks: [], colors: [[0, 0, 0, 0]]};
    }
  };

  // --- Products ----------------------------------------------------------------------------------
  const entries = new Map<string, Entry>();
  let activeEntry: Entry | null = null;
  let compiledProducts = 0;
  let summaryVersion = 0;
  let preparedVersion = 0;
  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];

  const makeEntry = (
    product: Entry['product'],
    configKey: string,
    build: ProductBuild,
    paintBuffer: Buffer,
    buffers: Record<string, Buffer>,
    mercatorBuild: ProductBuild | null = null
  ): Entry => ({
    product,
    configKey,
    build,
    mercatorBuild,
    paintBuffer,
    buffers,
    seenRunCount: -1,
    seenSummaryVersion: -1
  });

  function buildDecoded(): Entry {
    const builder = session.builder('decoded');
    const settings = builder.settings(PAINT_PARAMETER_LENGTH);
    const paint = builder.words('paint');
    addPaintPass(builder.graph, {
      id: 'decoded-paint',
      count: pixelCount,
      first: importGraphBuffer(
        builder.graph,
        'elevation',
        session.elevationBuffer,
        'float32',
        pixelCount
      ),
      settings: settings.view,
      output: paint
    });
    const stage = builder.finishStage({
      write: () => settings.parameters.write(getPaintParameterValues(getPaintSettings('decoded')))
    });
    const build = session.createBuild('decoded', stage, session.elevationBuffer, 'float32');
    return makeEntry('decoded', '', build, builder.getBuffer('paint'), {});
  }

  function buildSlope(state: BasicsOptions): Entry {
    const builder = session.builder('slope');
    const elevation = builder.elevation();
    const settings = builder.settings(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
    const slopeOnGround = builder.floats('ground');
    const slopeOnMercator = builder.floats('mercator');
    const common = {
      width,
      height,
      elevation,
      settings: settings.view,
      rowDirection: 'south' as const,
      borderMode: state.borderMode
    };
    // The same heights, two cell models: the ground size of each row, and the Mercator pixel
    // taken as metres (the wrong one the story warns about). One settings buffer serves both.
    builder.graph.add(
      new GPUTerrainDerivatives({
        ...common,
        id: 'derivatives-ground',
        slope: slopeOnGround,
        cellSizeMode: 'web-mercator'
      })
    );
    builder.graph.add(
      new GPUTerrainDerivatives({
        ...common,
        id: 'derivatives-mercator',
        slope: slopeOnMercator,
        cellSizeMode: 'uniform'
      })
    );
    const paintSettings = builder.settings(PAINT_PARAMETER_LENGTH);
    const paint = builder.words('paint');
    addPaintPass(builder.graph, {
      id: 'slope-paint',
      count: pixelCount,
      first: slopeOnGround,
      second: slopeOnMercator,
      settings: paintSettings.view,
      output: paint
    });
    const build = builder.finish({
      value: 'ground',
      format: 'float32',
      write: () => {
        settings.parameters.write(
          getGPUTerrainDerivativesParameterValues({...cellSettings, zFactor: ctx.options.zFactor})
        );
        paintSettings.parameters.write(getPaintParameterValues(getPaintSettings('slope')));
      }
    });
    const mercatorBuild = session.createBuild(
      'slope-mercator',
      build.stage,
      builder.getBuffer('mercator'),
      'float32'
    );
    return makeEntry(
      'slope',
      getConfigKey(state, 'slope'),
      build,
      builder.getBuffer('paint'),
      {ground: builder.getBuffer('ground'), mercator: builder.getBuffer('mercator')},
      mercatorBuild
    );
  }

  function buildAspect(state: BasicsOptions): Entry {
    const builder = session.builder('aspect');
    const elevation = builder.elevation();
    const settings = builder.settings(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
    const aspect = builder.floats('aspect');
    const slope = builder.floats('slope');
    builder.graph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation,
        settings: settings.view,
        slope,
        aspect,
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        borderMode: state.borderMode
      })
    );
    const paintSettings = builder.settings(PAINT_PARAMETER_LENGTH);
    const paint = builder.words('paint');
    addPaintPass(builder.graph, {
      id: 'aspect-paint',
      count: pixelCount,
      first: aspect,
      second: slope,
      settings: paintSettings.view,
      output: paint
    });
    const build = builder.finish({
      value: 'aspect',
      format: 'float32',
      write: () => {
        settings.parameters.write(
          getGPUTerrainDerivativesParameterValues({...cellSettings, zFactor: ctx.options.zFactor})
        );
        paintSettings.parameters.write(getPaintParameterValues(getPaintSettings('aspect')));
      }
    });
    return makeEntry('aspect', getConfigKey(state, 'aspect'), build, builder.getBuffer('paint'), {
      aspect: builder.getBuffer('aspect'),
      slope: builder.getBuffer('slope')
    });
  }

  function buildRuggedness(state: BasicsOptions, product: 'tpi' | 'tri'): Entry {
    const builder = session.builder(product);
    const elevation = builder.elevation();
    const output = builder.floats(product);
    builder.graph.add(
      new GPUTerrainRuggedness({
        id: 'ruggedness',
        width,
        height,
        elevation,
        ...(product === 'tpi'
          ? {topographicPositionIndex: output}
          : {terrainRuggednessIndex: output}),
        terrainRuggednessAlgorithm: state.triAlgorithm,
        edgeMode: state.edgeMode
      })
    );
    const paintSettings = builder.settings(PAINT_PARAMETER_LENGTH);
    const paint = builder.words('paint');
    addPaintPass(builder.graph, {
      id: `${product}-paint`,
      count: pixelCount,
      first: output,
      settings: paintSettings.view,
      output: paint
    });
    const build = builder.finish({
      value: product,
      format: 'float32',
      write: () =>
        paintSettings.parameters.write(getPaintParameterValues(getPaintSettings(product)))
    });
    return makeEntry(product, getConfigKey(state, product), build, builder.getBuffer('paint'), {});
  }

  function buildVectorRuggedness(state: BasicsOptions): Entry {
    const builder = session.builder('vrm');
    const elevation = builder.elevation();
    const settings = builder.settings(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH);
    const output = builder.floats('vrm');
    builder.graph.add(
      new GPUTerrainVectorRuggedness({
        id: 'vector-ruggedness',
        width,
        height,
        elevation,
        settings: settings.view,
        radius: state.vrmRadius,
        vectorRuggedness: output,
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        borderMode: state.borderMode
      })
    );
    const paintSettings = builder.settings(PAINT_PARAMETER_LENGTH);
    const paint = builder.words('paint');
    addPaintPass(builder.graph, {
      id: 'vrm-paint',
      count: pixelCount,
      first: output,
      settings: paintSettings.view,
      output: paint
    });
    const build = builder.finish({
      value: 'vrm',
      format: 'float32',
      minIntervalMs: 120,
      write: () => {
        settings.parameters.write(
          getGPUTerrainVectorRuggednessParameterValues({
            ...cellSettings,
            zFactor: ctx.options.zFactor
          })
        );
        paintSettings.parameters.write(getPaintParameterValues(getPaintSettings('vrm')));
      }
    });
    return makeEntry('vrm', getConfigKey(state, 'vrm'), build, builder.getBuffer('paint'), {});
  }

  const decodedEntry = buildDecoded();
  session.addBuild(decodedEntry.build, '');

  /** The entry of a product, built the first time and again when a compile option changed. */
  function getEntry(state: BasicsOptions): Entry {
    const {product} = state;
    const key = getConfigKey(state, product);
    const existing = entries.get(product);
    if (existing && existing.configKey === key) return existing;
    let entry: Entry;
    switch (product) {
      case 'slope':
        entry = buildSlope(state);
        break;
      case 'aspect':
        entry = buildAspect(state);
        break;
      case 'tpi':
      case 'tri':
        entry = buildRuggedness(state, product);
        break;
      case 'vrm':
        entry = buildVectorRuggedness(state);
        break;
    }
    session.addBuild(entry.build, key);
    if (entry.mercatorBuild) session.addBuild(entry.mercatorBuild, key);
    entries.set(product, entry);
    compiledProducts++;
    ctx.setReadout('rebuilds', `${compiledProducts} product graphs compiled`);
    return entry;
  }

  // --- Summaries read back from the GPU rasters ---------------------------------------------------
  const slopeSummaries: Partial<Record<'ground' | 'mercator', SlopeSummary>> = {};
  let aspectSummary: AspectSummary | null = null;
  const jobs: BulkJob[] = [];
  let bulkBusy = false;
  const aspectParts: {aspect?: Float32Array; slope?: Float32Array} = {};

  function enqueue(job: BulkJob): void {
    const existing = jobs.findIndex(candidate => candidate.key === job.key);
    if (existing >= 0) jobs.splice(existing, 1);
    jobs.push(job);
  }

  function pumpJobs(commandEncoder: Parameters<TerrainSession['encode']>[0]): void {
    if (bulkBusy || jobs.length === 0) return;
    const job = jobs[0];
    const requested = session.readBulk(commandEncoder, job.buffer, bytes => {
      bulkBusy = false;
      if (destroyed) return;
      job.accept(new Float32Array(bytes.buffer, bytes.byteOffset, pixelCount));
    });
    if (requested) {
      bulkBusy = true;
      jobs.shift();
    }
  }

  const formatShare = (share: number): string =>
    Number.isFinite(share) ? `${(share * 100).toFixed(1)}%` : '-';

  function publishSlope(): void {
    const state = ctx.options;
    const shown = slopeSummaries[state.cellModel];
    ctx.setLegendData('slopeCounts', {
      ground: slopeSummaries.ground?.classCounts,
      mercator: slopeSummaries.mercator?.classCounts
    });
    ctx.setLegendData('slopeHistogram', shown?.histogram ?? null);
    ctx.setReadout(
      'steeperThan30',
      shown ? formatShare(getShareFromClass(shown, DECISION_CLASS)) : null
    );
    ctx.setReadout(
      'steeperThan45',
      shown ? formatShare(getShareFromClass(shown, STEEP_CLASS)) : null
    );
    ctx.setChart(
      'slopeHistogram',
      shown ? getSlopeHistogramChart(shown, tables.slope, DECISION_SLOPE_DEGREES) : null
    );
    const onGround = slopeSummaries.ground;
    const onMercator = slopeSummaries.mercator;
    ctx.setReadout(
      'share30Ground',
      onGround ? formatShare(getShareFromClass(onGround, DECISION_CLASS)) : null
    );
    ctx.setReadout(
      'share30Mercator',
      onMercator ? formatShare(getShareFromClass(onMercator, DECISION_CLASS)) : null
    );
    ctx.setReadout('maxSlopeGround', onGround ? `${onGround.maximum.toFixed(1)}°` : null);
    publishAnnotations();
  }

  function publishAspect(): void {
    if (!aspectSummary) {
      ctx.setChart('aspectRose', null);
      ctx.setReadout('northShare', null);
      return;
    }
    ctx.setChart('aspectRose', getAspectRoseChart(aspectSummary));
    ctx.setReadout(
      'northShare',
      aspectSummary.steepCount > 0
        ? formatShare(aspectSummary.northCount / aspectSummary.steepCount)
        : '-'
    );
  }

  function requestSummaries(entry: Entry): void {
    const version = summaryVersion;
    const isCurrent = () => version === summaryVersion;
    if (entry.product === 'slope') {
      for (const model of ['ground', 'mercator'] as const) {
        enqueue({
          key: `slope-${model}`,
          buffer: entry.buffers[model],
          accept: values => {
            if (!isCurrent()) return;
            slopeSummaries[model] = summarizeSlope(values);
            publishSlope();
          }
        });
      }
    } else if (entry.product === 'aspect') {
      aspectParts.aspect = undefined;
      aspectParts.slope = undefined;
      for (const name of ['aspect', 'slope'] as const) {
        enqueue({
          key: `aspect-${name}`,
          buffer: entry.buffers[name],
          accept: values => {
            if (!isCurrent()) return;
            aspectParts[name] = values;
            if (aspectParts.aspect && aspectParts.slope) {
              aspectSummary = summarizeAspect(aspectParts.aspect, aspectParts.slope);
              aspectParts.aspect = undefined;
              aspectParts.slope = undefined;
              publishAspect();
            }
          }
        });
      }
    }
  }

  function requestElevationCheck(): void {
    const state = ctx.options;
    const comparable =
      state.encoding === 'terrarium' &&
      state.spikeDensity === 0 &&
      !state.missingTile &&
      !state.clampBathymetry;
    const version = preparedVersion;
    enqueue({
      key: 'elevation',
      buffer: session.elevationBuffer,
      accept: heights => {
        if (version !== preparedVersion) return;
        const summary = summarizeElevation(heights, comparable ? grid.cpuElevation : null);
        ctx.setReadout(
          'validPixels',
          `${((summary.valid / pixelCount) * 100).toFixed(2)}% (${summary.valid.toLocaleString('en-US')} px)`
        );
        ctx.setReadout(
          'relief',
          summary.valid > 0
            ? `${Math.round(summary.maximum - summary.minimum).toLocaleString('en-US')} m`
            : null
        );
        ctx.setReadout('cells', summary.valid > 0 ? formatMillions(summary.valid, 'cells') : null);
        ctx.setReadout(
          'summitHeight',
          summary.valid > 0 ? `${Math.round(summary.maximum).toLocaleString('en-US')} m` : null
        );
        ctx.setReadout(
          'decodeDifference',
          summary.maximumDifference === null
            ? 'n/a (input modified)'
            : summary.maximumDifference === 0
              ? 'exactly 0 m (bit-identical)'
              : `${summary.maximumDifference.toExponential(2)} m`
        );
        elevationMaximum = summary.valid > 0 ? summary.maximum : null;
        publishAnnotations();
      }
    });
  }
  let elevationMaximum: number | null = null;

  // --- Statistics of the displayed float raster (ruggedness classes, 98th percentile) -------------
  function onStats(_id: string, stats: ValueStats): void {
    const state = ctx.options;
    if (state.view !== 'analysis' || stats.kind !== 'float') {
      ctx.setReadout('p98', null);
      return;
    }
    const key = getFrozenKey(state);
    if (key && !frozenBreaks.has(key) && stats.count > 0) {
      // Quantile classes of this tile, measured once and frozen so a slider cannot move them.
      frozenBreaks.set(
        key,
        RUGGEDNESS_PROBABILITIES.map(probability =>
          Number(stats.quantile(probability).toPrecision(4))
        )
      );
      refreshTables();
      session.markDirty(activeEntry?.build);
      ctx.requestLayers();
    }
    const p98 = stats.quantile(0.98);
    switch (state.product) {
      case 'slope':
        ctx.setReadout('p98', `${p98.toFixed(1)}°`);
        break;
      case 'tpi':
      case 'tri':
        ctx.setReadout('p98', `${p98.toFixed(1)} m`);
        break;
      case 'vrm':
        ctx.setReadout('p98', p98.toFixed(3));
        break;
      default:
        ctx.setReadout('p98', null);
    }
  }
  session.onStats = onStats;

  // --- Hover probe ---------------------------------------------------------------------------------
  let lastProbe: HoverCell | null = null;
  session.describeHover = cell => {
    lastProbe = cell;
    ctx.refreshTooltip();
    return null;
  };
  const getWindowBounds = (
    column: number,
    row: number,
    radius = 1
  ): [number, number, number, number] => {
    // getLongitudeLatitude returns pixel centres: half a pixel out to the outer cell edges.
    const [west, north] = grid.getLongitudeLatitude(column - radius - 0.5, row - radius - 0.5);
    const [east, south] = grid.getLongitudeLatitude(column + radius + 0.5, row + radius + 0.5);
    return [west, south, east, north];
  };

  // --- Furniture and annotations -------------------------------------------------------------------
  function publishFurniture(): void {
    const state = ctx.options;
    const ticks =
      state.view === 'analysis' && state.product === 'vrm'
        ? [getWindowMeters(state.vrmRadius, cellMeters)]
        : [];
    ctx.setFurniture({
      title: {
        subtitle: getCartoucheSubtitle(state, cellMeters, mercatorMeters),
        sample: demSampleLine(grid)
      },
      scaleBar: {units: 'metric', ticks}
    });
    ctx.setReadout(
      'taps',
      state.product === 'vrm'
        ? `${(2 * state.vrmRadius + 1) ** 2} taps per cell, ${(((2 * state.vrmRadius + 1) ** 2 * pixelCount) / 1e6).toFixed(0)} M in all`
        : null
    );
  }

  function publishAnnotations(): void {
    const state = ctx.options;
    const list: MapAnnotation[] = [];
    const isSlope = state.view === 'analysis' && state.product === 'slope';
    if (state.view !== 'decoded') list.push(...glacierAnnotations);
    if (state.view === 'relief' || isSlope) list.push(getDataEdgeFrame(grid.lngLatBounds));
    if (state.view === 'relief' && elevationMaximum !== null) {
      const summit = snapLngLatToHighestCell(dem, ALPS.places.matterhorn.lngLat);
      if (summit) {
        list.push(getSummitNote(summit.lngLat, elevationMaximum, cellMeters, quantum));
      }
    }
    if (state.view === 'decoded' && state.missingTile) {
      list.push(getPatchOutline(getMissingPatchBounds(grid)));
    }
    if (isSlope) {
      const shown = slopeSummaries[state.cellModel];
      if (shown && Number.isFinite(shown.maximum) && shown.maximumIndex >= 0) {
        const column = shown.maximumIndex % width;
        const row = Math.floor(shown.maximumIndex / width);
        list.push(
          getSteepestNote(grid.getLongitudeLatitude(column, row), shown.maximum, state.cellModel)
        );
      }
    }
    ctx.setAnnotations('basics', list);
  }

  // --- Showing a product ---------------------------------------------------------------------------
  function selectBuild(): void {
    const state = ctx.options;
    let entry: Entry;
    let build: ProductBuild;
    if (state.view === 'analysis') {
      entry = getEntry(state);
      build =
        state.product === 'slope' && state.cellModel === 'mercator' && entry.mercatorBuild
          ? entry.mercatorBuild
          : entry.build;
    } else {
      entry = decodedEntry;
      build = entry.build;
    }
    activeEntry = entry;
    session.activate(build, UNUSED_PAINT);
    // The paint parameters depend on the display options: re-run the stage that paints.
    session.markDirty(entry.build);
    refreshTables();
    publishSlope();
    publishAspect();
    publishFurniture();
    publishAnnotations();
    ctx.requestLayers();
  }

  function scheduleRebuild(): void {
    timers.push(setTimeout(() => !destroyed && selectBuild(), REBUILD_DELAY_MILLISECONDS));
  }

  // --- Static readouts ---------------------------------------------------------------------------
  ctx.setReadout('decisionSlope', `${DECISION_SLOPE_DEGREES}°`);
  ctx.setReadout('groundCell', `${cellMeters.toFixed(1)} m`);
  ctx.setReadout('mercatorCell', `${mercatorMeters.toFixed(1)} m`);
  ctx.setReadout(
    'aspectFade',
    `${ASPECT_SLOPE_ALPHA.flatDegrees}° to ${ASPECT_SLOPE_ALPHA.fullDegrees}°`
  );
  ctx.setReadout(
    'naive45',
    `${STEEP_FACE_DEGREES}° reads as ${(
      (Math.atan(Math.tan((STEEP_FACE_DEGREES * Math.PI) / 180) * (cellMeters / mercatorMeters)) *
        180) /
        Math.PI
    ).toFixed(1)}°`
  );
  ctx.setReadout('rebuilds', 'none yet');

  selectBuild();
  requestElevationCheck();

  return {
    getCompiledGraphs: () =>
      [
        ...pipeline.getCompiledGraphs(),
        ...session.getCompiledGraphs()
      ] as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'view':
        case 'product':
        case 'cellModel':
        case 'slopeDisplay':
          selectBuild();
          return;
        case 'encoding':
        case 'validRange':
        case 'alphaNoData':
        case 'clampBathymetry':
          pipeline.decodeOptionChanged();
          break;
        case 'missingTile':
        case 'spikeDensity':
          pipeline.inputChanged();
          break;
        case 'repairSpikes':
          pipeline.selectionChanged();
          break;
        case 'borderMode':
        case 'triAlgorithm':
        case 'edgeMode':
        case 'vrmRadius':
          scheduleRebuild();
          break;
        case 'zFactor':
          summaryVersion++;
          session.markAllDirty();
          break;
      }
      publishFurniture();
      publishAnnotations();
    },

    onAction(id) {
      if (id !== 'measure') return;
      ctx.setReadout('timing', 'measuring...');
      void session.measure().then(results => {
        if (destroyed) return;
        const total = results.reduce((sum, result) => sum + result.milliseconds, 0);
        ctx.setReadout(
          'timing',
          results.length === 0
            ? 'static raster (no graph)'
            : `${total.toFixed(2)} ms for ${(pixelCount / 1e6).toFixed(1)} M cells (${results[0].method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'})`
        );
      });
    },

    onGroundChange(next) {
      ground.setGround(next);
      refreshTables();
      session.markAllDirty();
      publishSlope();
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      session.getTooltip(event);
      if (!event.coordinate) return null;
      const pixel = grid.getPixel(event.coordinate[0], event.coordinate[1]);
      const cell = lastProbe;
      if (!pixel || !cell) return null;
      // The probe lags the pointer by a frame; a value from a neighbouring pixel is still honest.
      if (Math.abs(cell.column - pixel[0]) > 3 || Math.abs(cell.row - pixel[1]) > 3) return null;
      const state = ctx.options;
      return getBasicsTooltip({
        state,
        tables,
        cell,
        probe: demProbe,
        groundCellMeters: cellMeters,
        mercatorCellMeters: mercatorMeters,
        onGlacier: glacierMask[cell.row * width + cell.column] === 1,
        getWindowBounds: (column, row) =>
          getWindowBounds(column, row, state.product === 'vrm' ? state.vrmRadius : 1)
      });
    },

    encode(commandEncoder) {
      if (pipeline.encode(commandEncoder)) {
        preparedVersion++;
        summaryVersion++;
        session.elevationChanged();
        requestElevationCheck();
      }
      session.encode(commandEncoder);
      const entry = activeEntry;
      const state = ctx.options;
      if (entry && state.view === 'analysis') {
        const {runCount} = entry.build.stage;
        if (entry.seenRunCount !== runCount) {
          entry.seenRunCount = runCount;
          if (entry.seenSummaryVersion !== summaryVersion) {
            entry.seenSummaryVersion = summaryVersion;
            requestSummaries(entry);
          }
        }
      }
      pumpJobs(commandEncoder);
    },

    getLayers(): Layer[] {
      const state = ctx.options;
      const layers = session.getLayers({
        underlay: state.view !== 'decoded',
        underlayAlpha: 1,
        alpha: 1,
        showProduct: false
      });
      if (state.view !== 'relief' && activeEntry) {
        layers.push(
          new ColorRasterLayer({
            id: 'terrain-basics-product',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            colors: activeEntry.paintBuffer,
            alpha: 1,
            ...(state.view === 'decoded' ? {noDataHatch: getHatchColor(ctx.ground())} : {})
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      pipeline.stop();
      session.destroy();
      ground.destroy();
      resources.destroy();
    }
  };
}
