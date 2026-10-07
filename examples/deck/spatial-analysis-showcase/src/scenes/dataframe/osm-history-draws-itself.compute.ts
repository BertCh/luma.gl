// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  getGPUTimeWindowParameterValues,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTemporalReduction,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createViewImporter,
  dayToYear,
  formatDate,
  loadOsmHistory,
  OSM_DATASET_ID,
  OSM_KIND_COLORS,
  scaleToFull
} from './osm-history-data';

/** Option state of the "city draws itself" scene. */
export type OsmDrawsItselfOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  showPoints: boolean;
  colorBy: 'year' | 'kind';
  ramp: Exclude<RampName, 'grayscale' | 'diverging'>;
  pointSize: number;
  showGlow: boolean;
  glowDays: number;
  showCells: boolean;
  cellSize: '1000' | '500' | '250';
  bucket: 'quarter' | 'half' | 'year';
  cellMode: 'cumulative' | 'bucket';
  ceiling: number;
  cellOpacity: number;
};

/** Bucket width in days for each `bucket` option; the compiled bucket count covers all of them. */
export const BUCKET_DAYS: Record<OsmDrawsItselfOptions['bucket'], number> = {
  quarter: 365.2425 / 4,
  half: 365.2425 / 2,
  year: 365.2425
};

/** Compile-time bucket count: a quarter-year bucket over the 19 years of data, with headroom. */
const BUCKET_COUNT = 80;
const STATUS_INTERVAL_FRAMES = 12;

type CellVariant = {
  cellSize: number;
  columns: number;
  rows: number;
  cellCount: number;
  bounds: [number, number, number, number];
  cellValues: Buffer;
  reduce: CompiledGPUCommandGraph<void>;
  values: CompiledGPUCommandGraph<void>;
  /** Buffers owned by this variant, released after its graphs. */
  buffers: Buffer[];
  reduced: boolean;
  summary: SummaryReader;
  /** Per-bucket totals and occupied-slot count, read once per reduction. */
  totals: SummaryReader;
};

/**
 * The city draws itself. `GPUTemporalReduction` folds the 400k sampled node creations into one
 * record per (cell, time bucket); a kernel then turns the buckets up to the playhead into one
 * density value per cell for a raster layer. Two `GPUTimeWindowFilter` graphs pick the nodes
 * created before the playhead (the ink) and in the last few days (the glow) without recompiling.
 * The cell size is compile-time (it changes the dense slot table); the bucket width, the playhead
 * and every display option are parameter writes.
 */
export async function createOsmDrawsItself(
  ctx: SceneContext<OsmDrawsItselfOptions>
): Promise<SceneInstance<OsmDrawsItselfOptions>> {
  const {device} = ctx;
  const history = loadOsmHistory(ctx.datasets.get(OSM_DATASET_ID));
  const count = history.count;
  const resources = new SpatialAnalysisResources(device, 'osm-draws');
  const coordinateOrigin: [number, number, number] = [history.origin[0], history.origin[1], 0];
  let destroyed = false;

  // ---- Static inputs ------------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', history.positions);
  const daysBuffer = resources.createBuffer('days', history.days);
  const kindBuffer = resources.createBuffer('kind', history.kind);
  const onesBuffer = resources.createBuffer('ones', new Float32Array(count).fill(1));
  const reduceParameters = resources.createParameterBuffer(
    'reduce',
    'float32',
    GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH
  );
  const valueParameters = resources.createParameterBuffer('cell-values', 'uint32', 4);

  // ---- Playback and window graphs -----------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, Math.floor(history.maxDay)], secondsPerLoop: 60, step: 7}
  );

  function createWindowGraph(name: string) {
    const ids = resources.createBuffer(`${name}-ids`, count * 4);
    const countBuffer = resources.createBuffer(`${name}-count`, 4);
    const overflow = resources.createBuffer(`${name}-overflow`, 4);
    const parameters = resources.createParameterBuffer(
      `${name}-window`,
      'float32',
      GPU_TIME_WINDOW_PARAMETER_LENGTH
    );
    const draw = resources.track(
      new DrawCommandBuffer(device, {
        id: `osm-${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const graph = new GPUCommandGraph<void>(device, {id: `osm-${name}`});
    const v = createViewImporter(graph, name);
    graph.add(
      new GPUTimeWindowFilter({
        id: `${name}-window`,
        timestamps: v('days', daysBuffer, 'float32', count),
        window: parameters.importToGraph(graph),
        output: {
          ids: v('ids', ids, 'uint32', count),
          count: v('count', countBuffer, 'uint32', 1),
          overflow: v('overflow', overflow, 'uint32', 1)
        },
        drawInstanceCount: graph.importGPUData(`${name}-draw-count`, draw.getInstanceCountData(0))
      })
    );
    return {ids, countBuffer, parameters, draw, compiled: resources.track(graph.compile())};
  }
  const ink = createWindowGraph('ink');
  const glow = createWindowGraph('glow');

  // ---- Cell variant (compile-time cell size) ------------------------------------------------------
  let variant: CellVariant | null = null;
  let valuesDirty = true;
  let windowsDirty = true;
  let lastBucket = -1;
  let totalsScaled: Float32Array | null = null;
  let usedBuckets = 0;
  let peakText = '';
  let occupiedText = '';

  function buildVariant(cellSize: number): CellVariant {
    const [minX, minY, maxX, maxY] = history.bounds;
    const columns = Math.ceil((maxX - minX) / cellSize) + 1;
    const rows = Math.ceil((maxY - minY) / cellSize) + 1;
    const cellCount = columns * rows;
    const slotCount = cellCount * BUCKET_COUNT;
    const tag = `${cellSize}`;
    const cellIds = resources.createBuffer(`cell-ids-${tag}`, count * 4);
    const counts = resources.createBuffer(`slot-counts-${tag}`, slotCount * 4);
    const slotMin = resources.createBuffer(`slot-min-${tag}`, slotCount * 4);
    const slotMax = resources.createBuffer(`slot-max-${tag}`, slotCount * 4);
    const slotFirst = resources.createBuffer(`slot-first-${tag}`, slotCount * 4);
    const slotLast = resources.createBuffer(`slot-last-${tag}`, slotCount * 4);
    const occupiedIds = resources.createBuffer(
      `occupied-ids-${tag}`,
      Math.min(slotCount, count) * 4
    );
    const occupiedCount = resources.createBuffer(`occupied-count-${tag}`, 4);
    const occupiedOverflow = resources.createBuffer(`occupied-overflow-${tag}`, 4);
    const bucketTotals = resources.createBuffer(`bucket-totals-${tag}`, BUCKET_COUNT * 4);
    const cellValues = resources.createBuffer(`cell-values-${tag}`, cellCount * 4);

    // Graph 1: node -> cell, then the dense (cell, bucket) reduction and a total per bucket.
    const reduceGraph = new GPUCommandGraph<void>(device, {id: `osm-reduce-${tag}`});
    const r = createViewImporter(reduceGraph, 'r');
    const cellIdsView = r('cell-ids', cellIds, 'uint32', count);
    addKernelPass(reduceGraph, {
      id: `osm-cell-ids-${tag}`,
      invocationCount: count,
      declarations: `const MIN_X: f32 = ${minX.toFixed(3)};
const MIN_Y: f32 = ${minY.toFixed(3)};
const CELL: f32 = ${cellSize.toFixed(1)};
const COLUMNS: u32 = ${columns}u;
const ROWS: u32 = ${rows}u;`,
      bindings: [
        {
          name: 'positions',
          view: r('positions', positionsBuffer, 'float32x2', count),
          type: 'f32',
          access: 'read'
        },
        {name: 'cellIds', view: cellIdsView, type: 'u32', access: 'read_write'}
      ],
      body: `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let column = u32(max(floor((x - MIN_X) / CELL), 0.0));
  let row = u32(max(floor((y - MIN_Y) / CELL), 0.0));
  cellIds[cellIdsOffset + index] = select(0xffffffffu, row * COLUMNS + column, column < COLUMNS && row < ROWS);`
    });
    const countsView = r('counts', counts, 'uint32', slotCount);
    reduceGraph.add(
      new GPUTemporalReduction({
        id: `osm-reduction-${tag}`,
        cellIds: cellIdsView,
        timestamps: r('days', daysBuffer, 'float32', count),
        values: r('ones', onesBuffer, 'float32', count),
        parameters: reduceParameters.importToGraph(reduceGraph),
        cellCount,
        bucketCount: BUCKET_COUNT,
        output: {
          counts: countsView,
          min: r('slot-min', slotMin, 'float32', slotCount),
          max: r('slot-max', slotMax, 'float32', slotCount),
          first: r('slot-first', slotFirst, 'float32', slotCount),
          last: r('slot-last', slotLast, 'float32', slotCount),
          occupiedSlots: {
            ids: r('occupied-ids', occupiedIds, 'uint32', Math.min(slotCount, count)),
            count: r('occupied-count', occupiedCount, 'uint32', 1),
            overflow: r('occupied-overflow', occupiedOverflow, 'uint32', 1)
          }
        }
      })
    );
    addKernelPass(reduceGraph, {
      id: `osm-bucket-totals-${tag}`,
      invocationCount: BUCKET_COUNT,
      declarations: `const CELLS: u32 = ${cellCount}u;
const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {name: 'counts', view: countsView, type: 'u32', access: 'read'},
        {
          name: 'totals',
          view: r('bucket-totals', bucketTotals, 'float32', BUCKET_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `var sum = 0u;
  for (var cell = 0u; cell < CELLS; cell = cell + 1u) {
    sum = sum + counts[countsOffset + cell * BUCKETS + index];
  }
  totals[totalsOffset + index] = f32(sum);`
    });

    // Graph 2: one density value per cell from the buckets up to the playhead's bucket.
    const valuesGraph = new GPUCommandGraph<void>(device, {id: `osm-cell-values-${tag}`});
    const w = createViewImporter(valuesGraph, 'w');
    addKernelPass(valuesGraph, {
      id: `osm-cell-sum-${tag}`,
      invocationCount: cellCount,
      declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {
          name: 'counts',
          view: w('counts', counts, 'uint32', slotCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'parameters',
          view: valueParameters.importToGraph(valuesGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'values',
          view: w('cell-values', cellValues, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let current = min(parameters[parametersOffset], BUCKETS - 1u);
  let cumulative = parameters[parametersOffset + 1u] == 0u;
  let base = index * BUCKETS;
  var sum = 0u;
  if (cumulative) {
    for (var bucket = 0u; bucket <= current; bucket = bucket + 1u) {
      sum = sum + counts[countsOffset + base + bucket];
    }
  } else {
    sum = counts[countsOffset + base + current];
  }
  values[valuesOffset + index] = f32(sum);`
    });

    const summary = new SummaryReader(
      resources,
      `osm-cells-${tag}`,
      [
        {buffer: ink.countBuffer, size: 4},
        {buffer: glow.countBuffer, size: 4},
        {buffer: cellValues, size: cellCount * 4}
      ],
      bytes => {
        if (destroyed || variant?.cellValues !== cellValues) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        let peak = 0;
        let occupied = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          const value = floats[2 + cell];
          if (value > 0) occupied++;
          if (value > peak) peak = value;
        }
        const areaKm2 = (cellSize / 1000) ** 2;
        ctx.setReadout('shown', `${formatCount(scaleToFull(history, words[0]))} nodes`);
        ctx.setReadout('glowing', `${formatCount(scaleToFull(history, words[1]))} nodes`);
        ctx.setReadout('cells', `${formatCount(occupied)} of ${formatCount(cellCount)}`);
        peakText = `${formatCount(scaleToFull(history, peak))} nodes in one ${cellSize} m cell (${formatCount(scaleToFull(history, peak) / areaKm2)} per km2)`;
        ctx.setReadout('peak', peakText);
      }
    );
    const totalsReader = new SummaryReader(
      resources,
      `osm-totals-${tag}`,
      [
        {buffer: occupiedCount, size: 4},
        {buffer: bucketTotals, size: BUCKET_COUNT * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes, 0, 1);
        const floats = new Float32Array(bytes, 4, BUCKET_COUNT);
        const width = BUCKET_DAYS[ctx.options.bucket];
        usedBuckets = Math.min(BUCKET_COUNT, Math.ceil((history.maxDay + 1) / width));
        totalsScaled = Float32Array.from(
          floats.subarray(0, usedBuckets),
          v => v / history.full.sampleFraction
        );
        occupiedText = `${formatCount(words[0])} (cell, bucket) records from ${formatCount(count)} nodes`;
        ctx.setReadout('records', occupiedText);
        lastBucket = -1;
        publishChart();
      }
    );

    return {
      cellSize,
      columns,
      rows,
      cellCount,
      bounds: [minX, minY, minX + columns * cellSize, minY + rows * cellSize],
      cellValues,
      reduce: resources.track(reduceGraph.compile()),
      values: resources.track(valuesGraph.compile()),
      buffers: [
        cellIds,
        counts,
        slotMin,
        slotMax,
        slotFirst,
        slotLast,
        occupiedIds,
        occupiedCount,
        occupiedOverflow,
        bucketTotals,
        cellValues
      ],
      reduced: false,
      summary,
      totals: totalsReader
    };
  }

  function disposeVariant(old: CellVariant): void {
    old.summary.stop();
    old.totals.stop();
    resources.release(old.reduce);
    resources.release(old.values);
    for (const buffer of old.buffers) resources.release(buffer);
  }

  function setVariant(cellSize: number): void {
    const old = variant;
    variant = buildVariant(cellSize);
    if (old) disposeVariant(old);
    valuesDirty = true;
    windowsDirty = true;
    ctx.setReadout('grid', `${variant.columns} x ${variant.rows} cells of ${cellSize} m`);
    ctx.requestLayers();
  }

  function currentBucket(playhead: number): number {
    return Math.min(
      Math.max(Math.floor(playhead / BUCKET_DAYS[ctx.options.bucket]), 0),
      BUCKET_COUNT - 1
    );
  }

  function publishChart(): void {
    if (!totalsScaled) return;
    const width = BUCKET_DAYS[ctx.options.bucket];
    const bucket = currentBucket(clock.time);
    const startYear = dayToYear(history, 0);
    ctx.setChart('perBucket', {
      kind: 'histogram',
      values: totalsScaled,
      xDomain: [startYear, startYear + (usedBuckets * width) / 365.2425],
      xLabel: 'year',
      yLabel: 'nodes created per bucket (estimate)',
      formatX: value => value.toFixed(0),
      height: 120,
      markers: [{x: dayToYear(history, bucket * width)}],
      description:
        'Nodes created per time bucket across the whole city, scaled from the sample to the full history; the vertical rule is the playhead.'
    });
  }

  ctx.setReadout('rows', `${formatCount(count)} of ${formatCount(history.full.fullCount)} nodes`);
  setVariant(Number(ctx.options.cellSize));

  return {
    getCompiledGraphs: () =>
      variant ? [ink.compiled, glow.compiled, variant.reduce, variant.values] : [],

    setOption(id, _value, state) {
      switch (id) {
        case 'cellSize':
          setVariant(Number(state.cellSize));
          break;
        case 'bucket':
          if (variant) variant.reduced = false;
          valuesDirty = true;
          break;
        case 'cellMode':
          valuesDirty = true;
          break;
        case 'glowDays':
        case 'showGlow':
          windowsDirty = true;
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!variant) return;
      const options = ctx.options;
      const playhead = clock.advance(frame);
      const width = BUCKET_DAYS[options.bucket];
      ctx.setReadout('clock', formatDate(history, playhead));

      if (!variant.reduced) {
        reduceParameters.write(getGPUTemporalReductionParameterValues(0, width));
        variant.reduce.encode(commandEncoder, {parameters: undefined});
        variant.reduced = true;
        valuesDirty = true;
        variant.totals.request(commandEncoder);
      } else {
        variant.totals.flush(commandEncoder);
      }

      const bucket = currentBucket(playhead);
      if (bucket !== lastBucket) {
        lastBucket = bucket;
        valuesDirty = true;
        publishChart();
      }
      if (valuesDirty) {
        valueParameters.write(
          Uint32Array.of(bucket, options.cellMode === 'cumulative' ? 0 : 1, 0, 0)
        );
        variant.values.encode(commandEncoder, {parameters: undefined});
        valuesDirty = false;
        variant.summary.markStale();
      }

      if (clock.moved || windowsDirty) {
        ink.parameters.write(getGPUTimeWindowParameterValues({start: -1, end: playhead}));
        ink.compiled.encode(commandEncoder, {parameters: undefined});
        if (options.showGlow) {
          glow.parameters.write(
            getGPUTimeWindowParameterValues({start: playhead - options.glowDays, end: playhead})
          );
          glow.compiled.encode(commandEncoder, {parameters: undefined});
        } else {
          glow.parameters.write(getGPUTimeWindowParameterValues({start: 1e9, end: 1e9 + 1}));
          glow.compiled.encode(commandEncoder, {parameters: undefined});
        }
        windowsDirty = false;
        variant.summary.markStale();
      }
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0 && clock.moved)
        variant.summary.markStale();
      variant.summary.flush(commandEncoder);
    },

    getLayers() {
      if (!variant) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const areaKm2 = (variant.cellSize / 1000) ** 2;
      if (options.showCells) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `osm-cells-${variant.cellSize}`,
            coordinateOrigin,
            gridSize: [variant.columns, variant.rows],
            bounds: variant.bounds,
            values: variant.cellValues,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [0, options.ceiling],
            valueScale: 1 / areaKm2,
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 255],
            opacity: options.cellOpacity
          })
        );
      }
      if (options.showPoints) {
        const byYear = options.colorBy === 'year';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-ink',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: ink.ids,
            drawCommands: ink.draw,
            values: byYear ? daysBuffer : kindBuffer,
            valueFormat: byYear ? 'float32' : 'uint32',
            colormap: byYear ? options.ramp : 'category',
            valueRange: [0, history.maxDay],
            palette: OSM_KIND_COLORS,
            radiusPixels: options.pointSize,
            opacity: options.showCells ? 0.8 : 0.9
          })
        );
      }
      if (options.showGlow) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-glow',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: glow.ids,
            drawCommands: glow.draw,
            colormap: 'uniform',
            color: dark ? [255, 250, 220, 255] : [30, 30, 40, 255],
            radiusPixels: options.pointSize + 1.6,
            opacity: 0.95
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      variant?.summary.stop();
      variant?.totals.stop();
      resources.destroy();
    }
  };
}
