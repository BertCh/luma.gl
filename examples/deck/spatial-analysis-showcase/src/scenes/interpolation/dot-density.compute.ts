// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUDotDensityParameterValues,
  GPU_DOT_DENSITY_PARAMETER_LENGTH,
  GPUDotDensity,
  GPURandomPointsInPolygon
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  fitRasterGrid,
  makeDasymetricWeights,
  rasterizePointCount,
  rasterizeStreetLength
} from './b7-ancillary';
import {DOT_VALUE_SETS, type DotValueSet} from './b7-dot-style';
import {formatSignificant} from './b7-format';
import {loadB7Polygons} from './b7-polygons';

/** Option state of the dot-density scene. */
export type DotDensityOptions = {
  valueSet: DotValueSet;
  logUnitsPerDot: number;
  zoomCoupling: number;
  mask: 'none' | 'streets' | 'places';
  dotRadius: number;
  display: 'dots' | 'random';
  randomCounts: 'match' | 'fixed';
  randomFixed: number;
  showTractOutlines: boolean;
};

const CATEGORY_COUNT = 5;
/** Dot capacity of the outputs. Compile-time. */
const DOT_CAPACITY = 1 << 20;
/** Fraction of the capacity the zoom coupling may use before dots per unit is capped. */
const CAPACITY_FILL = 0.9;
const MAXIMUM_RANDOM_PER_TRACT = 4000;
const REFERENCE_ZOOM = 10;
const MASK_WIDTH = 480;
const MASK_HEIGHT = 576;
const SETTLE_MILLISECONDS = 300;
const INITIAL_SEED = 20_260;

/**
 * Racial dot map of Chicago. `GPUDotDensity` places `value * dotsPerUnit` dots per (tract,
 * category), optionally thinned by a dasymetric weight raster, with counter-based random numbers:
 * a dot is a pure function of (seed, tract, category, rank), so changing the dot value (or tying it
 * to the zoom) only appends or removes dots. `GPURandomPointsInPolygon` places uniform points in the
 * same polygons for comparison. Both graphs compile once; every control is a buffer write.
 */
export async function createDotDensity(
  ctx: SceneContext<DotDensityOptions>
): Promise<SceneInstance<DotDensityOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const roads = ctx.datasets.get('chicago-roads');
  const places = ctx.datasets.get('chicago-places');
  const {device} = ctx;
  const origin = tracts.defaultOrigin;
  const projection = tracts.getProjection(origin);
  const polygons = loadB7Polygons(tracts, origin);
  const featureCount = polygons.featureCount;
  const slotCount = featureCount * CATEGORY_COUNT;
  const outlineCount = polygons.outlineSegments.length / 4;
  const geoids = (tracts.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown> | null)?.GEOID ?? '')
  );
  const population = tracts.column<Float32Array>('population');
  const households = tracts.column<Float32Array>('households');

  /** Feature-major category counts of one value set, padded to five categories. */
  const valueTables = new Map<DotValueSet, Float32Array>();
  function getValueTable(set: DotValueSet): Float32Array {
    let table = valueTables.get(set);
    if (!table) {
      table = new Float32Array(slotCount);
      const categories = DOT_VALUE_SETS[set].categories;
      for (let tract = 0; tract < featureCount; tract++) {
        categories.forEach((category, index) => {
          let value: number;
          if (category.column === 'aboveP150') {
            value = Math.max(
              0,
              population[tract] - tracts.column<Float32Array>('poverty150')[tract]
            );
          } else if (category.column === 'withVehicle') {
            value = Math.max(
              0,
              households[tract] - tracts.column<Float32Array>('noVehicle')[tract]
            );
          } else {
            value = tracts.column<Float32Array>(category.column)[tract];
          }
          table![tract * CATEGORY_COUNT + index] = value;
        });
      }
      valueTables.set(set, table);
    }
    return table;
  }
  const totalOf = (table: Float32Array) => table.reduce((sum, value) => sum + value, 0);

  // --- Dasymetric mask rasters ----------------------------------------------------------------
  const grid = fitRasterGrid(polygons.bounds, MASK_WIDTH, MASK_HEIGHT, 0.01);
  const maskRasters = {
    none: new Float32Array(MASK_WIDTH * MASK_HEIGHT).fill(1),
    streets: makeDasymetricWeights(
      rasterizeStreetLength(roads, origin, grid),
      MASK_WIDTH,
      MASK_HEIGHT,
      1,
      0.03
    ),
    places: makeDasymetricWeights(
      rasterizePointCount(places, origin, grid),
      MASK_WIDTH,
      MASK_HEIGHT,
      2,
      0.03
    )
  };
  const maskExtent = [grid.originX, grid.originY, grid.cellSize, grid.cellSize] as const;

  // --- Buffers --------------------------------------------------------------------------------
  const resources = new SpatialAnalysisResources(device, 'dot-density');
  const polygonPositions = resources.createBuffer('polygon-positions', polygons.polygonPositions);
  const featureOffsets = resources.createBuffer('feature-offsets', polygons.featureOffsets);
  const polygonOffsets = resources.createBuffer('polygon-offsets', polygons.polygonOffsets);
  const ringOffsets = resources.createBuffer('ring-offsets', polygons.ringOffsets);
  const outlineSegments = resources.createBuffer('outline-segments', polygons.outlineSegments);
  const valuesBuffer = resources.createBuffer('values', getValueTable(ctx.options.valueSet));
  const maskBuffer = resources.createBuffer('mask', maskRasters[ctx.options.mask]);
  const parameters = resources.createParameterBuffer(
    'dot-parameters',
    'uint32',
    GPU_DOT_DENSITY_PARAMETER_LENGTH
  );
  const randomParameters = resources.createParameterBuffer(
    'random-parameters',
    'uint32',
    GPU_DOT_DENSITY_PARAMETER_LENGTH
  );
  const randomCountsBuffer = resources.createBuffer('random-counts', new Uint32Array(featureCount));
  const dotPositions = resources.createBuffer('dot-positions', DOT_CAPACITY * 8);
  const dotIds = resources.createBuffer('dot-ids', DOT_CAPACITY * 4);
  const dotCategories = resources.createBuffer('dot-categories', DOT_CAPACITY * 4);
  const dotOverflow = resources.createBuffer('dot-overflow', 4);
  const dotTotal = resources.createBuffer('dot-total', 4);
  const dotFailed = resources.createBuffer('dot-failed', 4);
  const slotCounts = resources.createBuffer('slot-counts', slotCount * 4);
  const slotOffsets = resources.createBuffer('slot-offsets', slotCount * 4);
  const dotDrawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'dot-density-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const randomPositions = resources.createBuffer('random-positions', DOT_CAPACITY * 8);
  const randomIds = resources.createBuffer('random-ids', DOT_CAPACITY * 4);
  const randomOverflow = resources.createBuffer('random-overflow', 4);
  const randomFailed = resources.createBuffer('random-failed', 4);
  const randomDrawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'random-points-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  const importPolygons = <Parameters>(graph: GPUCommandGraph<Parameters>) => ({
    polygonPositions: importGraphBuffer(
      graph,
      'polygon-positions',
      polygonPositions,
      'float32x2',
      polygons.polygonPositions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      featureOffsets,
      'uint32',
      polygons.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      polygonOffsets,
      'uint32',
      polygons.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      ringOffsets,
      'uint32',
      polygons.ringOffsets.length
    )
  });
  const importMask = <Parameters>(graph: GPUCommandGraph<Parameters>) => ({
    weights: importGraphBuffer(graph, 'mask', maskBuffer, 'float32', MASK_WIDTH * MASK_HEIGHT),
    width: MASK_WIDTH,
    height: MASK_HEIGHT
  });

  const dotGraph = new GPUCommandGraph<void>(device, {id: 'dot-density'});
  dotGraph.add(
    new GPUDotDensity({
      id: 'dots',
      ...importPolygons(dotGraph),
      values: importGraphBuffer(dotGraph, 'values', valuesBuffer, 'float32', slotCount),
      categoryCount: CATEGORY_COUNT,
      parameters: parameters.importToGraph(dotGraph),
      mask: importMask(dotGraph),
      output: {
        positions: importGraphBuffer(
          dotGraph,
          'dot-positions',
          dotPositions,
          'float32x2',
          DOT_CAPACITY
        ),
        dots: {
          ids: importGraphBuffer(dotGraph, 'dot-ids', dotIds, 'uint32', DOT_CAPACITY),
          count: dotGraph.importGPUData('dot-count', dotDrawCommands.getInstanceCountData(0)),
          overflow: importGraphBuffer(dotGraph, 'dot-overflow', dotOverflow, 'uint32', 1),
          requiredCount: importGraphBuffer(dotGraph, 'dot-total', dotTotal, 'uint32', 1)
        },
        categories: importGraphBuffer(
          dotGraph,
          'dot-categories',
          dotCategories,
          'uint32',
          DOT_CAPACITY
        ),
        failedCount: importGraphBuffer(dotGraph, 'dot-failed', dotFailed, 'uint32', 1),
        slotCounts: importGraphBuffer(dotGraph, 'slot-counts', slotCounts, 'uint32', slotCount),
        slotOffsets: importGraphBuffer(dotGraph, 'slot-offsets', slotOffsets, 'uint32', slotCount)
      }
    })
  );
  const compiledDots = resources.track(dotGraph.compile());

  const randomGraph = new GPUCommandGraph<void>(device, {id: 'random-points'});
  randomGraph.add(
    new GPURandomPointsInPolygon({
      id: 'random-points',
      ...importPolygons(randomGraph),
      counts: importGraphBuffer(
        randomGraph,
        'random-counts',
        randomCountsBuffer,
        'uint32',
        featureCount
      ),
      parameters: randomParameters.importToGraph(randomGraph),
      mask: importMask(randomGraph),
      output: {
        positions: importGraphBuffer(
          randomGraph,
          'random-positions',
          randomPositions,
          'float32x2',
          DOT_CAPACITY
        ),
        dots: {
          ids: importGraphBuffer(randomGraph, 'random-ids', randomIds, 'uint32', DOT_CAPACITY),
          count: randomGraph.importGPUData(
            'random-count',
            randomDrawCommands.getInstanceCountData(0)
          ),
          overflow: importGraphBuffer(randomGraph, 'random-overflow', randomOverflow, 'uint32', 1)
        },
        failedCount: importGraphBuffer(randomGraph, 'random-failed', randomFailed, 'uint32', 1)
      }
    })
  );
  const compiledRandom = resources.track(randomGraph.compile());

  // --- State ----------------------------------------------------------------------------------
  let destroyed = false;
  let seed = INITIAL_SEED;
  let busy = false;
  let readStale = true;
  let lastChangeTime = performance.now();
  let lastDotsPerUnit = 0;
  let totalValue = totalOf(getValueTable(ctx.options.valueSet));
  let dotsDirty = true;
  let randomDirty = true;
  let lastZoom = Number.NaN;
  let capped = false;

  const markChanged = () => {
    readStale = true;
    lastChangeTime = performance.now();
  };

  /** Dots per unit at `zoom`: the base value, doubled every `1 / coupling` zoom, capped by capacity. */
  function getDotsPerUnit(zoom: number): number {
    const base =
      2 ** (ctx.options.zoomCoupling * (zoom - REFERENCE_ZOOM)) / 10 ** ctx.options.logUnitsPerDot;
    const limit = (CAPACITY_FILL * DOT_CAPACITY) / Math.max(totalValue, 1);
    capped = base > limit;
    return Math.min(base, limit);
  }

  function writeDotParameters(dotsPerUnit: number): void {
    parameters.write(getGPUDotDensityParameterValues({seed, dotsPerUnit, maskExtent}));
  }

  function writeRandomCounts(dotsPerUnit: number): void {
    const o = ctx.options;
    const table = getValueTable(o.valueSet);
    const counts = new Uint32Array(featureCount);
    for (let tract = 0; tract < featureCount; tract++) {
      if (o.randomCounts === 'fixed') {
        counts[tract] = o.randomFixed;
      } else {
        let total = 0;
        for (let category = 0; category < CATEGORY_COUNT; category++) {
          total += table[tract * CATEGORY_COUNT + category];
        }
        counts[tract] = Math.min(MAXIMUM_RANDOM_PER_TRACT, Math.round(total * dotsPerUnit));
      }
    }
    randomCountsBuffer.write(counts);
    randomParameters.write(getGPUDotDensityParameterValues({seed: seed + 1, maskExtent}));
  }

  // --- Summary read-back ----------------------------------------------------------------------
  const summaryReader = new SummaryReader(
    resources,
    'dot-density-summary',
    [
      {buffer: dotDrawCommands.buffer, size: 16},
      {buffer: dotOverflow, size: 4},
      {buffer: dotTotal, size: 4},
      {buffer: dotFailed, size: 4},
      {buffer: randomDrawCommands.buffer, size: 16},
      {buffer: randomFailed, size: 4},
      {buffer: slotCounts, size: slotCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      // Draw-command records are four words: vertexCount, instanceCount, first vertex, first instance.
      const dots = words[1];
      const overflow = words[4];
      const total = words[5];
      const failed = words[6];
      const randomCount = words[8];
      const randomFailedCount = words[11];
      const slots = words.subarray(12);
      const set = DOT_VALUE_SETS[ctx.options.valueSet];
      const totals = new Array<number>(CATEGORY_COUNT).fill(0);
      for (let slot = 0; slot < slotCount; slot++) totals[slot % CATEGORY_COUNT] += slots[slot];
      ctx.setReadout('dots', `${formatCount(dots)} of ${formatCount(DOT_CAPACITY)}`);
      ctx.setReadout(
        'perCategory',
        set.categories
          .map(
            (category, index) => `${category.label.split(' (')[0]}: ${formatCount(totals[index])}`
          )
          .join(' · ')
      );
      ctx.setReadout(
        'failed',
        `${formatCount(failed)} dots could not be placed${overflow ? `; OVERFLOW (needed ${formatCount(total)})` : ''}`
      );
      ctx.setReadout(
        'randomPoints',
        ctx.options.display === 'random'
          ? `${formatCount(randomCount)} points, ${formatCount(randomFailedCount)} failed`
          : 'off (choose "Uniform random points" in Show)'
      );
    }
  );

  // --- Checks and timing (outside the frame) --------------------------------------------------
  async function runDotsOutsideFrame(dotsPerUnit: number) {
    writeDotParameters(dotsPerUnit);
    const encoder = device.createCommandEncoder();
    compiledDots.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const [positions, counts, offsets] = await Promise.all([
      dotPositions.readAsync(),
      slotCounts.readAsync(),
      slotOffsets.readAsync()
    ]);
    return {
      positions: new Float32Array(positions.buffer, positions.byteOffset, positions.byteLength / 4),
      counts: new Uint32Array(counts.buffer, counts.byteOffset, counts.byteLength / 4),
      offsets: new Uint32Array(offsets.buffer, offsets.byteOffset, offsets.byteLength / 4)
    };
  }

  async function checkStability(): Promise<void> {
    if (busy || destroyed) return;
    busy = true;
    ctx.setReadout('stability', 'checking...');
    try {
      const base = Math.min(getDotsPerUnit(REFERENCE_ZOOM), (0.35 * DOT_CAPACITY) / totalValue);
      const coarse = await runDotsOutsideFrame(base);
      const fine = await runDotsOutsideFrame(base * 2.5);
      let compared = 0;
      let moved = 0;
      let removed = 0;
      for (let slot = 0; slot < slotCount; slot++) {
        if (fine.counts[slot] < coarse.counts[slot])
          removed += coarse.counts[slot] - fine.counts[slot];
        const shared = Math.min(coarse.counts[slot], fine.counts[slot]);
        for (let rank = 0; rank < shared; rank++) {
          const a = (coarse.offsets[slot] + rank) * 2;
          const b = (fine.offsets[slot] + rank) * 2;
          compared++;
          if (
            !Object.is(coarse.positions[a], fine.positions[b]) ||
            !Object.is(coarse.positions[a + 1], fine.positions[b + 1])
          ) {
            moved++;
          }
        }
      }
      const coarseTotal = coarse.offsets[slotCount - 1] + coarse.counts[slotCount - 1];
      const fineTotal = fine.offsets[slotCount - 1] + fine.counts[slotCount - 1];
      ctx.setReadout(
        'stability',
        `${formatCount(compared)} dots compared (${formatCount(coarseTotal)} → ${formatCount(fineTotal)} dots): ${moved} moved, ${removed} removed`
      );
    } catch (error) {
      ctx.setReadout('stability', `failed: ${(error as Error).message}`);
    } finally {
      busy = false;
      dotsDirty = true;
    }
  }

  async function measureGraphs(): Promise<void> {
    if (busy || destroyed) return;
    busy = true;
    ctx.setReadout('graphTime', 'measuring...');
    try {
      writeDotParameters(getDotsPerUnit(REFERENCE_ZOOM));
      randomParameters.write(getGPUDotDensityParameterValues({seed: seed + 1, maskExtent}));
      const options = {parameters: undefined, completionBuffer: dotOverflow, signal: ctx.signal};
      const dotTiming = await measureCompiledGraph(device, compiledDots, options);
      const randomTiming = await measureCompiledGraph(device, compiledRandom, options);
      ctx.setReadout(
        'graphTime',
        `GPUDotDensity ${compiledDots.stats.nodeOrder.length} nodes, ${formatCompiledGraphTiming(dotTiming)}; GPURandomPointsInPolygon ${compiledRandom.stats.nodeOrder.length} nodes, ${formatCompiledGraphTiming(randomTiming)}`
      );
    } catch (error) {
      ctx.setReadout('graphTime', `failed: ${(error as Error).message}`);
    } finally {
      busy = false;
      dotsDirty = true;
      randomDirty = true;
    }
  }

  ctx.setReadout(
    'tracts',
    `${formatCount(featureCount)} census tracts, ${formatCount(totalOf(population))} residents`
  );
  ctx.setReadout(
    'mask',
    `${MASK_WIDTH} × ${MASK_HEIGHT} weights of ${grid.cellSize.toFixed(0)} m (${((MASK_WIDTH * MASK_HEIGHT * 4) / 1e6).toFixed(1)} MB)`
  );
  ctx.setReadout('stability', 'press the button to check');
  ctx.setReadout('graphTime', 'press the button to time');
  ctx.setReadout('seed', String(seed));

  return {
    getCompiledGraphs: () => [compiledDots, compiledRandom] as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'valueSet':
          valuesBuffer.write(getValueTable(state.valueSet));
          totalValue = totalOf(getValueTable(state.valueSet));
          dotsDirty = true;
          randomDirty = true;
          break;
        case 'mask':
          maskBuffer.write(maskRasters[state.mask]);
          dotsDirty = true;
          randomDirty = true;
          break;
        case 'logUnitsPerDot':
        case 'zoomCoupling':
        case 'randomCounts':
        case 'randomFixed':
          dotsDirty = true;
          randomDirty = true;
          break;
        default:
          break;
      }
      markChanged();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reseed') {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        ctx.setReadout('seed', String(seed));
        dotsDirty = true;
        randomDirty = true;
        markChanged();
      } else if (id === 'check') {
        void checkStability();
      } else if (id === 'measure') {
        void measureGraphs();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const tract = polygons.locate(x, y);
      if (tract < 0) return null;
      const set = DOT_VALUE_SETS[ctx.options.valueSet];
      const table = getValueTable(ctx.options.valueSet);
      let total = 0;
      for (let category = 0; category < set.categories.length; category++) {
        total += table[tract * CATEGORY_COUNT + category];
      }
      const parts = set.categories.map((category, index) => {
        const value = table[tract * CATEGORY_COUNT + index];
        return `${category.label.split(' (')[0]} ${formatCount(value)} (${total > 0 ? ((100 * value) / total).toFixed(0) : 0}%)`;
      });
      return `Tract ${geoids[tract] ?? tract}: ${formatCount(total)} ${set.unit}\n${parts.join('\n')}`;
    },

    encode(commandEncoder, frame) {
      if (busy) return;
      const zoom = frame.viewport.zoom;
      const o = ctx.options;
      if (zoom !== lastZoom) {
        lastZoom = zoom;
        dotsDirty = true;
        randomDirty = true;
        markChanged();
      }
      const dotsPerUnit = getDotsPerUnit(zoom);
      if (dotsPerUnit !== lastDotsPerUnit) {
        lastDotsPerUnit = dotsPerUnit;
        const unit = DOT_VALUE_SETS[o.valueSet].unit;
        ctx.setReadout(
          'dotValue',
          `1 dot = ${formatSignificant(1 / dotsPerUnit, 3)} ${unit} (zoom ${zoom.toFixed(1)})${capped ? ', capped by the dot capacity' : ''}`
        );
      }
      if (o.display === 'dots') {
        if (dotsDirty) {
          writeDotParameters(dotsPerUnit);
          compiledDots.encode(commandEncoder, {parameters: undefined});
          dotsDirty = false;
          markChanged();
        }
      } else if (randomDirty) {
        writeRandomCounts(dotsPerUnit);
        compiledRandom.encode(commandEncoder, {parameters: undefined});
        randomDirty = false;
        markChanged();
      }
      if (readStale && performance.now() - lastChangeTime > SETTLE_MILLISECONDS) {
        if (!summaryReader.isPending) {
          summaryReader.request(commandEncoder);
          readStale = false;
        }
      } else {
        summaryReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const o = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const set = DOT_VALUE_SETS[o.valueSet];
      const layers: Layer[] = [];
      if (o.showTractOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'dot-density-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: outlineCount,
            widthPixels: 0.7,
            color: dark ? [255, 255, 255, 60] : [20, 20, 20, 70]
          })
        );
      }
      if (o.display === 'dots') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `dot-density-dots-${o.valueSet}`,
            coordinateOrigin,
            positions: dotPositions,
            drawCommands: dotDrawCommands,
            values: dotCategories,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: set.categories.map(category => category.color),
            radiusPixels: o.dotRadius
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'dot-density-random',
            coordinateOrigin,
            positions: randomPositions,
            drawCommands: randomDrawCommands,
            radiusPixels: o.dotRadius,
            color: dark ? [255, 224, 102, 235] : [200, 120, 0, 235]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      summaryReader.stop();
      resources.destroy();
    }
  };
}
