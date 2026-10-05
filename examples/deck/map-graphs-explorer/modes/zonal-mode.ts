// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Zonal statistics: one `GPUZonalStatistics` graph joins the San Francisco bike-parking points to
 * the ZIP-code polygons and reduces them to per-ZIP counts, densities, mean spaces per rack and
 * sums, plus the extent of the selected statistic, all on the GPU.
 *
 * The choropleth is drawn without triangulating the polygons: at create time the ZIP polygons are
 * rasterized once on the CPU into a grid of feature rows, and a `MapGraphsRasterLayer` colors each
 * cell by looking up the selected statistic output through that grid (`valueIndices`). The GPU
 * extent buffer drives the color range. Every output buffer has one extra sentinel row that cells
 * outside every ZIP point at.
 *
 * Compile-time options (rebuild the graph): the extent statistic and `sumOrder` (`atomic` vs
 * `sorted`). Per-frame: the animated point positions (re-aggregated every frame, no recompile).
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUZonalStatistics,
  importGraphBuffer,
  type GPUZonalStatisticsExtentStatistic,
  type GPUZonalStatisticsSumOrder
} from '@luma.gl/experimental/map-graphs';
import {LocalMetricProjection, type MapGraphsPolygons} from '../map-graphs-data';
import {
  MapGraphsPointLayer,
  MapGraphsRasterLayer,
  MapGraphsSegmentLayer
} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Peak circular drift of each point while animating, in meters. */
const DRIFT_RADIUS_METERS = 150;
/** Frames between summary readbacks. */
const READBACK_INTERVAL = 15;
/** Frame index at which the atomic-vs-sorted measurement runs once automatically. */
const AUTOMATIC_TIMING_FRAME = 40;
/** Columns of the CPU-rasterized feature-row grid. */
const GRID_COLUMNS = 512;
/** Float sentinel stored in the extra row of float outputs; discarded by the raster layer. */
const FLOAT_SENTINEL = -1e30;
const FLOAT_DISCARD_AT_OR_BELOW = -1e29;
const UINT32_SENTINEL = 0xffffffff;
/** Multiplier of the optional non-integer value mode. */
const FRACTIONAL_VALUE_SCALE = 0.37;

type Statistic = 'count' | 'density' | 'mean';

const STATISTICS: Record<
  Statistic,
  {
    extent: GPUZonalStatisticsExtentStatistic;
    label: string;
    unit: string;
    valueScale: number;
    digits: number;
  }
> = {
  count: {extent: 'count', label: 'Count', unit: 'racks', valueScale: 1, digits: 0},
  density: {extent: 'density', label: 'Density', unit: 'racks/km²', valueScale: 1e6, digits: 1},
  mean: {extent: 'mean', label: 'Mean spaces per rack', unit: 'spaces', valueScale: 1, digits: 2}
};

/**
 * Words of the readback summary: counts, means, densities, sums (per feature), extent, overflow,
 * uncertain point-in-polygon pairs.
 */
type Summary = {
  counts: Uint32Array;
  means: Float32Array;
  densities: Float32Array;
  sums: Float32Array;
  sumBits: Uint32Array;
  extent: [number, number];
  overflow: boolean;
  uncertainCount: number;
};

/** Cell-to-feature grid produced by {@link rasterizeFeatureRows}. */
type FeatureRowGrid = {
  columns: number;
  rows: number;
  cellSize: number;
  bounds: [number, number, number, number];
  /** `row * columns + column` to feature row, or the feature count outside every polygon. */
  featureRows: Uint32Array;
};

export const zonalMode: MapGraphsModeDefinition = {
  id: 'zonal',
  title: 'Zonal stats',
  recipes: ['GPUZonalStatistics'],
  description:
    'Bike-parking racks are aggregated per ZIP code on the GPU: counts, density per km², mean ' +
    'spaces per rack. The choropleth is a CPU-rasterized grid of ZIP rows, not triangles. ' +
    'Switch the statistic or the sum order (atomic vs reproducible sorted) and hover a ZIP.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 12.2},

  async create(context) {
    const [parking, zips] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new MapGraphsResources(device, 'zonal');
    const pointCount = parking.positions.length / 2;
    const featureCount = zips.featureOffsets.length - 1;
    const segmentCount = zips.outlineSegments.length / 4;
    const projection = new LocalMetricProjection(zips.origin);
    const grid = rasterizeFeatureRows(zips);

    // Per-point values. `fractional` swaps in non-integer values so float sums become
    // order-sensitive (integer sums below 2^24 are exact in any order).
    const integerValues = parking.spaces;
    const fractionalValues = Float32Array.from(
      integerValues,
      value => value * FRACTIONAL_VALUE_SCALE
    );

    const positionsBuffer = resources.createBuffer('positions', parking.positions);
    const valuesBuffer = resources.createBuffer('values', integerValues);
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const cellFeatureRows = resources.createBuffer('cell-feature-rows', grid.featureRows);

    // Output buffers: featureCount + 1 rows; the extra row is a sentinel the graph never writes.
    const createFloatOutput = (name: string) => {
      const initial = new Float32Array(featureCount + 1);
      initial[featureCount] = FLOAT_SENTINEL;
      return resources.createBuffer(name, initial);
    };
    const countInitial = new Uint32Array(featureCount + 1);
    countInitial[featureCount] = UINT32_SENTINEL;
    const countsBuffer = resources.createBuffer('counts', countInitial);
    const meansBuffer = createFloatOutput('means');
    const densitiesBuffer = createFloatOutput('densities');
    const sumsBuffer = createFloatOutput('sums');
    const extentBuffer = resources.createBuffer('extent', 8);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const uncertainCountBuffer = resources.createBuffer('uncertain-count', 4);

    let graphGeneration = 0;
    /** Builds and compiles the zonal graph over the shared caller-owned buffers. */
    const buildGraph = (
      sumOrder: GPUZonalStatisticsSumOrder,
      statistic: Statistic
    ): CompiledGPUCommandGraph<void> => {
      const id = `zonal-${sumOrder}-${statistic}-${graphGeneration++}`;
      const graph = new GPUCommandGraph<void>(device, {id});
      const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      graph.add(
        new GPUZonalStatistics({
          id,
          sumOrder,
          features: {
            kind: 'polygons',
            polygonPositions: view(
              'polygon-positions',
              polygonPositions,
              'float32x2',
              zips.polygonPositions.length / 2
            ),
            featureOffsets: view(
              'feature-offsets',
              featureOffsets,
              'uint32',
              zips.featureOffsets.length
            ),
            polygonOffsets: view(
              'polygon-offsets',
              polygonOffsets,
              'uint32',
              zips.polygonOffsets.length
            ),
            ringOffsets: view('ring-offsets', ringOffsets, 'uint32', zips.ringOffsets.length),
            candidateCapacity: Math.max(1024, pointCount * 4)
          },
          points: view('points', positionsBuffer, 'float32x2', pointCount),
          values: view('values', valuesBuffer, 'float32', pointCount),
          output: {
            counts: view('counts', countsBuffer, 'uint32', featureCount),
            means: view('means', meansBuffer, 'float32', featureCount),
            densities: view('densities', densitiesBuffer, 'float32', featureCount),
            sums: view('sums', sumsBuffer, 'float32', featureCount),
            extent: view('extent', extentBuffer, 'float32', 2),
            extentStatistic: STATISTICS[statistic].extent,
            overflow: view('overflow', overflowBuffer, 'uint32', 1),
            uncertainCount: view('uncertain-count', uncertainCountBuffer, 'uint32', 1)
          }
        })
      );
      return graph.compile();
    };

    let statistic: Statistic = 'count';
    let sumOrder: GPUZonalStatisticsSumOrder = 'atomic';
    let compiled = resources.track(buildGraph(sumOrder, statistic));

    const summaryWords = featureCount * 4 + 4;
    const readbackRing = new GPUReadbackRing(device, {
      id: 'zonal-readback',
      byteLength: summaryWords * 4
    });
    resources.track({destroy: () => readbackRing.destroy()});

    // Deterministic per-point drift phases and the animated positions scratch array.
    const phases = new Float32Array(pointCount);
    for (let index = 0; index < pointCount; index++) {
      phases[index] = (((index * 2654435761) >>> 0) / 4294967296) * Math.PI * 2;
    }
    const animated = new Float32Array(pointCount * 2);

    let animate = false;
    let fractional = false;
    let destroyed = false;
    let readbackPending = false;
    let measuring = false;
    let automaticTimingStarted = false;
    let summary: Summary | null = null;
    let hoveredRow = -1;
    let previousSumBits: Uint32Array | null = null;
    let identicalReadbacks = 0;
    let readbackGeneration = 0;
    let colorRange: [number, number] = [0, 1];
    const timings: Partial<Record<GPUZonalStatisticsSumOrder, CompiledGraphTiming>> = {};

    // Controls -------------------------------------------------------------------------------
    const statisticSelect = context.controls.addSelect<Statistic>({
      label: 'Statistic (compile-time: extent statistic)',
      options: [
        {value: 'count', label: 'Count (racks)'},
        {value: 'density', label: 'Density (racks per km²)'},
        {value: 'mean', label: 'Mean spaces per rack'}
      ],
      value: statistic,
      onChange: value => {
        statistic = value;
        rebuildGraph();
      }
    });
    void statisticSelect;
    context.controls.addSelect<GPUZonalStatisticsSumOrder>({
      label: 'Sum order (compile-time: rebuilds graph)',
      options: [
        {value: 'atomic', label: 'atomic (fast, order varies)'},
        {value: 'sorted', label: 'sorted (reproducible)'}
      ],
      value: sumOrder,
      onChange: value => {
        sumOrder = value;
        rebuildGraph();
      }
    });
    context.controls.addToggle({
      label: 'Animate points (per-frame re-aggregation)',
      value: animate,
      onChange: value => {
        animate = value;
        if (!animate) positionsBuffer.write(parking.positions);
        resetReproducibility();
        updateReproducibilityReadout();
      }
    });
    context.controls.addToggle({
      label: `Non-integer values (spaces × ${FRACTIONAL_VALUE_SCALE}; float sums become order-sensitive)`,
      value: fractional,
      onChange: value => {
        fractional = value;
        valuesBuffer.write(fractional ? fractionalValues : integerValues);
        resetReproducibility();
      }
    });
    context.controls.addLegend({
      title: 'Selected statistic per ZIP (viridis, GPU extent range)',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: 'min',
        maximumLabel: 'max'
      }
    });
    const rangeReadout = context.controls.addReadout('Color range', '...');
    context.controls.addReadout('Points', formatCount(pointCount));
    context.controls.addReadout('Polygons (ZIP codes)', formatCount(featureCount));
    const overflowReadout = context.controls.addReadout('Overflow', '...');
    // Points the join left unassigned (point count minus the sum of counts); every real rack lies
    // inside exactly one ZIP, so this should read 0. Uncertain pairs are the join's exact-predicate
    // fallbacks (`output.uncertainCount`).
    const unassignedReadout = context.controls.addReadout('Unassigned points', '...');
    const uncertainReadout = context.controls.addReadout('Uncertain point-in-polygon pairs', '...');
    const extentReadout = context.controls.addReadout('Statistic extent', '...');
    const topReadout = context.controls.addReadout('Top ZIP', '...');
    const hoverReadout = context.controls.addReadout('Hovered ZIP', 'hover the map');
    const reproducibilityReadout = context.controls.addReadout(
      'Sums bitwise identical across last N readbacks',
      '...'
    );
    context.controls.addButton({
      label: 'Measure atomic vs sorted',
      onClick: () => void measureSumOrders()
    });
    const atomicReadout = context.controls.addReadout('Atomic sums', '...');
    const sortedReadout = context.controls.addReadout('Sorted sums', '...');
    const costReadout = context.controls.addReadout('Cost of reproducible order', '...');
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise ' +
        'wall clock / 8 repetitions (upper bound).'
    );
    context.controls.addReadout('Data', `${parking.attribution}; ${zips.attribution}`);

    // Helpers --------------------------------------------------------------------------------
    function formatStatistic(value: number, which: Statistic = statistic): string {
      if (!Number.isFinite(value)) return 'n/a';
      const {digits} = STATISTICS[which];
      return value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      });
    }

    function getDisplayValue(row: number, which: Statistic = statistic): number {
      if (!summary) return NaN;
      if (which === 'count') return summary.counts[row];
      if (which === 'density') return summary.densities[row] * 1e6;
      return summary.means[row];
    }

    function describeFeature(row: number): string {
      if (!summary) return '...';
      const name = zips.featureNames[row] ?? String(zips.featureIds[row]);
      return (
        `ZIP ${name}: ${formatCount(summary.counts[row])} racks, ` +
        `${formatStatistic(getDisplayValue(row, 'density'), 'density')} racks/km², ` +
        `${formatStatistic(getDisplayValue(row, 'mean'), 'mean')} mean spaces`
      );
    }

    function resetReproducibility(): void {
      previousSumBits = null;
      identicalReadbacks = 0;
      readbackGeneration++;
    }

    function updateReproducibilityReadout(changedCount = 0): void {
      if (animate) {
        reproducibilityReadout.setValue('n/a (points moving)');
      } else if (!previousSumBits) {
        reproducibilityReadout.setValue('...');
      } else if (changedCount > 0) {
        reproducibilityReadout.setValue(
          `no (${changedCount} ZIP sums changed between readbacks, ${sumOrder} order)`
        );
      } else {
        reproducibilityReadout.setValue(
          `yes (${identicalReadbacks + 1} readbacks, ${sumOrder} order)`
        );
      }
    }

    function updateTimingReadouts(): void {
      if (destroyed) return;
      const atomic = timings.atomic;
      const sorted = timings.sorted;
      atomicReadout.setValue(atomic ? formatCompiledGraphTiming(atomic) : '...');
      sortedReadout.setValue(sorted ? formatCompiledGraphTiming(sorted) : '...');
      costReadout.setValue(
        atomic && sorted
          ? `sorted vs atomic: ${formatSpeedup(atomic.milliseconds, sorted.milliseconds)}`
          : '...'
      );
    }

    /** Recompiles the graph for a compile-time option change and retires the old one safely. */
    function rebuildGraph(): void {
      const previous = compiled;
      compiled = resources.track(buildGraph(sumOrder, statistic));
      resetReproducibility();
      summary = null;
      updateReproducibilityReadout();
      // Deck may still encode layers that read these buffers this frame; wait two frames.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          if (!destroyed) resources.release(previous);
        })
      );
      context.updateLayers();
    }

    /** Times the current graph and a temporary graph of the other sum order, outside the frame. */
    async function measureSumOrders(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      const other: GPUZonalStatisticsSumOrder = sumOrder === 'atomic' ? 'sorted' : 'atomic';
      const temporary = buildGraph(other, statistic);
      const current = compiled;
      const currentOrder = sumOrder;
      try {
        const options = {
          parameters: undefined,
          completionBuffer: overflowBuffer,
          signal: context.signal
        };
        timings[currentOrder] = await measureCompiledGraph(device, current, options);
        if (destroyed) return;
        timings[other] = await measureCompiledGraph(device, temporary, options);
        updateTimingReadouts();
      } catch {
        // Aborted or destroyed while measuring.
      } finally {
        measuring = false;
        temporary.destroy();
      }
    }

    async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const generation = readbackGeneration;
      const rowBytes = featureCount * 4;
      let offset = 0;
      for (const [sourceBuffer, size] of [
        [countsBuffer, rowBytes],
        [meansBuffer, rowBytes],
        [densitiesBuffer, rowBytes],
        [sumsBuffer, rowBytes],
        [extentBuffer, 8],
        [overflowBuffer, 4],
        [uncertainCountBuffer, 4]
      ] as const) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size
        });
        offset += size;
      }
      ticket.markEncoded({byteLength: summaryWords * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + summaryWords * 4);
        const words = new Uint32Array(copy);
        const floats = new Float32Array(copy);
        const next: Summary = {
          counts: words.subarray(0, featureCount),
          means: floats.subarray(featureCount, featureCount * 2),
          densities: floats.subarray(featureCount * 2, featureCount * 3),
          sums: floats.subarray(featureCount * 3, featureCount * 4),
          sumBits: words.subarray(featureCount * 3, featureCount * 4),
          extent: [floats[featureCount * 4], floats[featureCount * 4 + 1]],
          overflow: words[featureCount * 4 + 2] !== 0,
          uncertainCount: words[featureCount * 4 + 3]
        };
        summary = next;
        // Reads in flight across a graph or value change are shown but not compared.
        if (generation === readbackGeneration) {
          if (!animate) {
            let changedCount = 0;
            if (previousSumBits) {
              for (let row = 0; row < featureCount; row++) {
                if (previousSumBits[row] !== next.sumBits[row]) changedCount++;
              }
              identicalReadbacks = changedCount === 0 ? identicalReadbacks + 1 : 0;
            }
            previousSumBits = next.sumBits;
            updateReproducibilityReadout(changedCount);
          }
        }
        updateSummaryReadouts();
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    function updateSummaryReadouts(): void {
      if (!summary) return;
      const info = STATISTICS[statistic];
      overflowReadout.setValue(summary.overflow ? 'YES' : 'no');
      let assigned = 0;
      for (let row = 0; row < featureCount; row++) assigned += summary.counts[row];
      unassignedReadout.setValue(
        assigned === pointCount
          ? '0'
          : `${formatCount(pointCount - assigned)} (outside every ZIP or unresolved)`
      );
      uncertainReadout.setValue(formatCount(summary.uncertainCount));
      const minimum = summary.extent[0] * info.valueScale;
      const maximum = summary.extent[1] * info.valueScale;
      extentReadout.setValue(
        `${info.label}: ${formatStatistic(minimum)} to ${formatStatistic(maximum)} ${info.unit}`
      );
      rangeReadout.setValue(
        `${formatStatistic(minimum)} (dark) to ${formatStatistic(maximum)} (bright) ${info.unit}`
      );
      let topRow = -1;
      for (let row = 0; row < featureCount; row++) {
        const value = getDisplayValue(row);
        if (Number.isFinite(value) && (topRow < 0 || value > getDisplayValue(topRow))) {
          topRow = row;
        }
      }
      topReadout.setValue(
        topRow < 0
          ? 'n/a'
          : `${zips.featureNames[topRow] ?? zips.featureIds[topRow]} (${formatStatistic(
              getDisplayValue(topRow)
            )} ${info.unit}; ${info.label.toLowerCase()})`
      );
      if (summary.extent[0] !== colorRange[0] || summary.extent[1] !== colorRange[1]) {
        colorRange = [summary.extent[0], summary.extent[1]];
      }
      if (hoveredRow >= 0) hoverReadout.setValue(describeFeature(hoveredRow));
    }

    function getFeatureRowAt(coordinate: readonly [number, number] | null): number {
      if (!coordinate) return -1;
      const [x, y] = projection.project(coordinate[0], coordinate[1]);
      const column = Math.floor((x - grid.bounds[0]) / grid.cellSize);
      const row = Math.floor((y - grid.bounds[1]) / grid.cellSize);
      if (column < 0 || row < 0 || column >= grid.columns || row >= grid.rows) return -1;
      const featureRow = grid.featureRows[row * grid.columns + column];
      return featureRow < featureCount ? featureRow : -1;
    }

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (animate) {
          const angularSpeed = 1.2;
          for (let index = 0; index < pointCount; index++) {
            const angle = phases[index] + frame.timeSeconds * angularSpeed;
            animated[index * 2] =
              parking.positions[index * 2] + Math.cos(angle) * DRIFT_RADIUS_METERS;
            animated[index * 2 + 1] =
              parking.positions[index * 2 + 1] + Math.sin(angle) * DRIFT_RADIUS_METERS;
          }
          positionsBuffer.write(animated);
        }
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 0) {
          void readSummary(commandEncoder);
        }
        if (!automaticTimingStarted && frame.frameIndex >= AUTOMATIC_TIMING_FRAME) {
          automaticTimingStarted = true;
          void measureSumOrders();
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const info = STATISTICS[statistic];
        const isCount = statistic === 'count';
        const values =
          statistic === 'count'
            ? countsBuffer
            : statistic === 'density'
              ? densitiesBuffer
              : meansBuffer;
        const layers: Layer[] = [
          new MapGraphsRasterLayer({
            id: `zonal-choropleth-${statistic}`,
            coordinateOrigin,
            gridSize: [grid.columns, grid.rows],
            bounds: grid.bounds,
            rowOrigin: 'south',
            values,
            valueFormat: isCount ? 'uint32' : 'float32',
            valueIndices: cellFeatureRows,
            extent: extentBuffer,
            valueScale: info.valueScale,
            colormap: 'viridis',
            noDataValue: UINT32_SENTINEL,
            noDataColor: isCount ? [0, 0, 0, 0] : [150, 150, 150, 110],
            ...(isCount ? {} : {discardAtOrBelow: FLOAT_DISCARD_AT_OR_BELOW * info.valueScale}),
            opacity: 0.75
          }),
          new MapGraphsSegmentLayer({
            id: 'zonal-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: segmentCount,
            color: [225, 230, 240, 200],
            widthPixels: 1
          }),
          new MapGraphsPointLayer({
            id: 'zonal-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            color: [255, 255, 255, 90],
            radiusPixels: 1.5
          })
        ];
        return layers;
      },
      getTooltip(event) {
        const row = getFeatureRowAt(event.coordinate);
        hoveredRow = row;
        if (row < 0) return null;
        const text = describeFeature(row);
        hoverReadout.setValue(text);
        return text;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};

/**
 * Rasterizes polygon features into a square-cell grid of feature rows, CPU only. A cell takes the
 * lowest feature row whose polygons contain the cell center (even-odd across all rings of each
 * feature, matching the join's "smallest row wins"). Cells in no polygon get `featureCount`.
 * Row 0 is the south edge, as `MapGraphsRasterLayer` expects with `rowOrigin: 'south'`.
 */
function rasterizeFeatureRows(polygons: MapGraphsPolygons): FeatureRowGrid {
  const {polygonPositions, featureOffsets, polygonOffsets, ringOffsets} = polygons;
  const featureCount = featureOffsets.length - 1;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < polygonPositions.length; index += 2) {
    minX = Math.min(minX, polygonPositions[index]);
    maxX = Math.max(maxX, polygonPositions[index]);
    minY = Math.min(minY, polygonPositions[index + 1]);
    maxY = Math.max(maxY, polygonPositions[index + 1]);
  }
  const cellSize = (maxX - minX) / GRID_COLUMNS;
  const columns = GRID_COLUMNS;
  const rows = Math.max(1, Math.ceil((maxY - minY) / cellSize));
  const featureRows = new Uint32Array(columns * rows).fill(featureCount);
  const crossings: number[][] = Array.from({length: rows}, () => []);

  for (let feature = 0; feature < featureCount; feature++) {
    for (const rowCrossings of crossings) rowCrossings.length = 0;
    let firstRow = rows;
    let lastRow = -1;
    for (
      let ring = polygonOffsets[featureOffsets[feature]];
      ring < polygonOffsets[featureOffsets[feature + 1]];
      ring++
    ) {
      const start = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      const length = end - start;
      for (let vertex = 0; vertex < length; vertex++) {
        const from = (start + vertex) * 2;
        const to = (start + ((vertex + 1) % length)) * 2;
        const x0 = polygonPositions[from];
        const y0 = polygonPositions[from + 1];
        const x1 = polygonPositions[to];
        const y1 = polygonPositions[to + 1];
        if (y0 === y1) continue;
        const lowY = Math.min(y0, y1);
        const highY = Math.max(y0, y1);
        // Rows whose center `minY + (row + 0.5) * cellSize` lies in [lowY, highY).
        const rowStart = Math.max(0, Math.ceil((lowY - minY) / cellSize - 0.5));
        const rowEnd = Math.min(rows - 1, Math.ceil((highY - minY) / cellSize - 0.5) - 1);
        for (let row = rowStart; row <= rowEnd; row++) {
          const centerY = minY + (row + 0.5) * cellSize;
          crossings[row].push(x0 + ((centerY - y0) / (y1 - y0)) * (x1 - x0));
        }
        if (rowStart <= rowEnd) {
          firstRow = Math.min(firstRow, rowStart);
          lastRow = Math.max(lastRow, rowEnd);
        }
      }
    }
    for (let row = firstRow; row <= lastRow; row++) {
      const rowCrossings = crossings[row].sort((left, right) => left - right);
      for (let index = 0; index + 1 < rowCrossings.length; index += 2) {
        const columnStart = Math.max(0, Math.ceil((rowCrossings[index] - minX) / cellSize - 0.5));
        const columnEnd = Math.min(
          columns - 1,
          Math.ceil((rowCrossings[index + 1] - minX) / cellSize - 0.5) - 1
        );
        for (let column = columnStart; column <= columnEnd; column++) {
          const cell = row * columns + column;
          if (featureRows[cell] === featureCount) featureRows[cell] = feature;
        }
      }
    }
  }
  return {
    columns,
    rows,
    cellSize,
    bounds: [minX, minY, minX + columns * cellSize, minY + rows * cellSize],
    featureRows
  };
}
