// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Quadbin cell pyramid over every New York trip vertex. One graph (`GPUCellPyramid`) keys the
 * points into Quadbin cells at the finest resolution and rolls the table up level by level; it is
 * encoded only when the point mask changes. A second tiny graph (`GPUCellLevelSelection`) runs
 * every frame: the map zoom (or a manual override) is written into a one-word parameter buffer and
 * the selection node publishes that level's cell count and indirect draw record. The layer decodes
 * the two-word Quadbin keys in its vertex shader, so zooming never rebuilds or re-aggregates.
 *
 * Roll-up graph: `GPUCellRollup` reads the finest table (resolution 17, with per-point fixed-point
 * sums of the trip time) and rolls it up to a coarser parent resolution chosen with a select. The
 * resolution is a kernel constant, so each choice has its own tiny graph, all compiled in `create`;
 * picking one is a re-encode, not a rebuild. The map can show the parents coloured by count
 * density or by mean trip time, and readouts compare the finest totals with the parent totals:
 * counts and fixed-point sums are integers, so the roll-up conserves them exactly.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellLevelSelection,
  GPUCellPyramid,
  GPUCellRollup,
  GPU_CELL_DEFAULT_SUM_SCALE
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import type {Buffer} from '@luma.gl/core';
import {LocalMetricProjection} from '../spatial-analysis-data';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {QuadbinCellLayer} from './cell-pyramid-layers';

/** Finest and coarsest Quadbin resolution of the pyramid (inclusive). */
const FINEST_RESOLUTION = 17;
const COARSEST_RESOLUTION = 4;
const LEVEL_COUNT = FINEST_RESOLUTION - COARSEST_RESOLUTION + 1;
/** Rows per level table; rounded so every level slab starts on a 256-byte boundary. */
const MAXIMUM_TABLE_CAPACITY = 1 << 18;
const READBACK_INTERVAL_FRAMES = 15;
/** Default number of Quadbin resolutions finer than the map zoom (cell edge = 512 / 2^offset px). */
const DEFAULT_RESOLUTION_OFFSET = 4;
/** Parent resolutions offered by the roll-up select; each has its own precompiled graph. */
const ROLLUP_RESOLUTIONS = [16, 14, 12, 10, 8, 6];
const DEFAULT_ROLLUP_RESOLUTION = 12;
const DEFAULT_DENSITY_LOG_RANGE: readonly [number, number] = [0, 4];

/** Resolution of pyramid level `levelIndex` (level 0 is the finest). */
function getLevelResolution(levelIndex: number): number {
  return FINEST_RESOLUTION - levelIndex;
}

/** Deterministic per-point hash in [0, 1) used to thin the mask. */
function hashToUnit(index: number): number {
  let hash = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}

export const cellPyramidMode: SpatialAnalysisModeDefinition = {
  id: 'cell-pyramid',
  title: 'Cells',
  contributors: ['GPUCellAggregation', 'GPUCellPyramid', 'GPUCellLevelSelection', 'GPUCellRollup'],
  description:
    'New York trip vertices aggregated into Quadbin cells, one table per zoom level. Zoom the ' +
    'map: the level is chosen on the GPU each frame from a one-word buffer; the pyramid is only ' +
    'rebuilt (re-encoded) when the point sample changes. Switch on the roll-up to see the finest ' +
    'table rolled up to a parent resolution you pick, with the totals compared.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 11.5},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'cell-pyramid');
    const pointCount = trips.vertexTimestamps.length;
    const capacity = Math.min(Math.ceil(pointCount / 1024) * 1024, MAXIMUM_TABLE_CAPACITY);
    // A roll-up may not read and write the same buffer, so even and odd levels alternate between
    // two slabs (and two scalar buffers); each slab holds half of the levels.
    const slabRows = capacity * Math.ceil(LEVEL_COUNT / 2);

    // Quadbin keys need longitude/latitude; the dataset stores planar meters around its origin.
    const lngLat = new Float32Array(pointCount * 2);
    for (let point = 0; point < pointCount; point++) {
      const [longitude, latitude] = projection.unproject(
        trips.vertexPositions[point * 2],
        trips.vertexPositions[point * 2 + 1]
      );
      lngLat[point * 2] = longitude;
      lngLat[point * 2 + 1] = latitude;
    }
    const positionsBuffer = resources.createBuffer('lng-lat', lngLat);
    // Per-point value summed per cell: minutes since the first trip timestamp.
    const pointValues = new Float32Array(pointCount);
    for (let point = 0; point < pointCount; point++) {
      pointValues[point] = (trips.vertexTimestamps[point] - trips.timeRange[0]) / 60;
    }
    const valuesBuffer = resources.createBuffer('point-values', pointValues);
    const maximumMeanValue = Math.max(1, (trips.timeRange[1] - trips.timeRange[0]) / 60);
    const maskValues = new Uint32Array(pointCount).fill(1);
    const maskBuffer = resources.createBuffer('mask', maskValues);

    // All levels share one slab per column so the layer binds once and the selection node only
    // publishes the active level's row count and first row.
    const cellsSlabs = [0, 1].map(parity =>
      resources.createBuffer(`cells-slab-${parity}`, slabRows * 8)
    );
    const countsSlabs = [0, 1].map(parity =>
      resources.createBuffer(`counts-slab-${parity}`, slabRows * 4)
    );
    const tableCountBuffers = [0, 1].map(parity =>
      resources.createBuffer(`table-counts-${parity}`, LEVEL_COUNT * 4)
    );
    const levelOverflowBuffers = [0, 1].map(parity =>
      resources.createBuffer(`level-overflow-${parity}`, LEVEL_COUNT * 4)
    );
    // Fixed-point sums of the finest level only; coarser pyramid levels do not need them.
    const finestSumsBuffer = resources.createBuffer('finest-sums', capacity * 8);
    // Roll-up output table (shared by every parent resolution; only one graph is encoded at a time).
    const rollupCellsBuffer = resources.createBuffer('rollup-cells', capacity * 8);
    const rollupCountsBuffer = resources.createBuffer('rollup-counts', capacity * 4);
    const rollupSumsBuffer = resources.createBuffer('rollup-sums', capacity * 8);
    const rollupSumValuesBuffer = resources.createBuffer('rollup-sum-values', capacity * 4);
    const rollupOverflowBuffer = resources.createBuffer('rollup-overflow', 4);
    // Words the roll-up layer reads as "active level" and "first row": always zero.
    const zeroWordBuffer = resources.createBuffer('zero-word', 4);
    const rollupDrawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'cell-rollup-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const levelCountsBuffer = resources.createBuffer('level-counts', LEVEL_COUNT * 4);
    const levelTotalsBuffer = resources.createBuffer('level-totals', 4);
    const activeLevel = resources.createParameterBuffer('active-level', 'uint32', 1);
    const activeFirstRow = resources.createBuffer('active-first-row', 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'cell-pyramid-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    // level counts, even-level overflow flags, odd-level overflow flags, finest total, indirect record head
    const readbackByteLength = (LEVEL_COUNT * 3 + 3) * 4;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'cell-pyramid-summary', byteLength: readbackByteLength})
    );

    // Pyramid graph: encoded when the mask changes.
    const pyramidGraph = new GPUCommandGraph<void>(device, {id: 'cell-pyramid'});
    const importSlice = (
      buffer: Buffer,
      format: 'uint32' | 'uint32x2',
      rowByteLength: number,
      firstRow: number,
      length: number
    ): GraphDataView<'uint32'> & GraphDataView<'uint32x2'> => {
      const handle =
        importedHandles.get(buffer) ??
        pyramidGraph.importBuffer(
          {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
      importedHandles.set(buffer, handle);
      return pyramidGraph.createDataView(handle, {
        format,
        length,
        byteOffset: firstRow * rowByteLength
      }) as GraphDataView<'uint32'> & GraphDataView<'uint32x2'>;
    };
    const importedHandles = new Map<Buffer, ReturnType<typeof pyramidGraph.importBuffer>>();
    const levels = Array.from({length: LEVEL_COUNT}, (_, levelIndex) => {
      const parity = levelIndex & 1;
      const slabRow = (levelIndex >> 1) * capacity;
      return {
        resolution: getLevelResolution(levelIndex),
        output: {
          cells: importSlice(cellsSlabs[parity], 'uint32x2', 8, slabRow, capacity),
          counts: importSlice(countsSlabs[parity], 'uint32', 4, slabRow, capacity),
          count: importSlice(tableCountBuffers[parity], 'uint32', 4, levelIndex, 1),
          overflow: importSlice(levelOverflowBuffers[parity], 'uint32', 4, levelIndex, 1),
          ...(levelIndex === 0
            ? {
                totalCount: importSlice(levelTotalsBuffer, 'uint32', 4, 0, 1),
                sums: importSlice(finestSumsBuffer, 'uint32x2', 8, 0, capacity)
              }
            : {})
        }
      };
    });
    const pyramid = new GPUCellPyramid({
      id: 'trips-pyramid',
      family: 'quadbin',
      positions: importGraphBuffer(
        pyramidGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      ),
      mask: importGraphBuffer(pyramidGraph, 'mask', maskBuffer, 'uint32', pointCount),
      values: importGraphBuffer(pyramidGraph, 'values', valuesBuffer, 'float32', pointCount),
      levels,
      levelCounts: importGraphBuffer(
        pyramidGraph,
        'level-counts',
        levelCountsBuffer,
        'uint32',
        LEVEL_COUNT
      )
    });
    pyramidGraph.add(pyramid);
    const compiledPyramid = resources.track(pyramidGraph.compile());

    // Selection graph: encoded every frame. It never touches the points or the tables.
    const selectionGraph = new GPUCommandGraph<void>(device, {id: 'cell-level-selection'});
    const drawView = drawCommands.importToGraph(selectionGraph);
    selectionGraph.add(
      new GPUCellLevelSelection({
        id: 'trips-level',
        levelCounts: importGraphBuffer(
          selectionGraph,
          'level-counts',
          levelCountsBuffer,
          'uint32',
          LEVEL_COUNT
        ),
        activeLevel: activeLevel.importToGraph(selectionGraph),
        levelFirstRows: pyramid.levelFirstRows,
        output: {
          // The record's firstInstance stays 0: the shell does not request the
          // indirect-first-instance device feature, so the layer adds the first row itself.
          count: selectionGraph.createDataView(drawView.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: Uint32Array.BYTES_PER_ELEMENT
          }),
          firstRow: importGraphBuffer(
            selectionGraph,
            'active-first-row',
            activeFirstRow,
            'uint32',
            1
          )
        }
      })
    );
    const compiledSelection = resources.track(selectionGraph.compile());

    // Roll-up graphs, one per parent resolution (the resolution is a kernel constant). They are
    // compiled up front, so picking a resolution is only a choice of which graph to encode.
    const rollupGraphs = new Map<number, CompiledGPUCommandGraph<void>>();
    const createRollupGraph = (parentResolution: number): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `cell-rollup-${parentResolution}`});
      const handles = new Map<Buffer, ReturnType<typeof graph.importBuffer>>();
      const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
        buffer: Buffer,
        format: Format,
        rowByteLength: number,
        length: number,
        firstRow = 0
      ): GraphDataView<Format> => {
        let handle = handles.get(buffer);
        if (!handle) {
          handle = graph.importBuffer(
            {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          );
          handles.set(buffer, handle);
        }
        return graph.createDataView(handle, {
          format,
          length,
          byteOffset: firstRow * rowByteLength
        });
      };
      const drawView = rollupDrawCommands.importToGraph(graph);
      graph.add(
        new GPUCellRollup({
          id: `trips-rollup-${parentResolution}`,
          family: 'quadbin',
          sourceResolution: FINEST_RESOLUTION,
          resolution: parentResolution,
          source: {
            cells: view(cellsSlabs[0], 'uint32x2', 8, capacity),
            counts: view(countsSlabs[0], 'uint32', 4, capacity),
            sums: view(finestSumsBuffer, 'uint32x2', 8, capacity),
            count: view(tableCountBuffers[0], 'uint32', 4, 1),
            overflow: view(levelOverflowBuffers[0], 'uint32', 4, 1)
          },
          sumScale: GPU_CELL_DEFAULT_SUM_SCALE,
          output: {
            cells: view(rollupCellsBuffer, 'uint32x2', 8, capacity),
            counts: view(rollupCountsBuffer, 'uint32', 4, capacity),
            sums: view(rollupSumsBuffer, 'uint32x2', 8, capacity),
            sumValues: view(rollupSumValuesBuffer, 'float32', 4, capacity),
            // The record's instanceCount is the parent cell count (the clamped output count).
            count: graph.createDataView(drawView.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: Uint32Array.BYTES_PER_ELEMENT
            }),
            overflow: view(rollupOverflowBuffer, 'uint32', 4, 1)
          }
        })
      );
      return resources.track(graph.compile());
    };
    for (const parentResolution of ROLLUP_RESOLUTIONS) {
      rollupGraphs.set(parentResolution, createRollupGraph(parentResolution));
    }

    let rollupEnabled = false;
    let rollupResolution = DEFAULT_ROLLUP_RESOLUTION;
    let rollupColorByMeanValue = false;
    let rollupDirty = false;
    let totalsWanted = false;
    let totalsPending = false;
    let manual = false;
    let manualResolution = 12;
    let resolutionOffset = DEFAULT_RESOLUTION_OFFSET;
    let keptPercent = 100;
    let densityLogRange = DEFAULT_DENSITY_LOG_RANGE;
    let pyramidDirty = true;
    let readbackPending = false;
    let destroyed = false;

    context.controls.addSlider({
      label: 'Cell resolutions finer than zoom (auto level)',
      min: 0,
      max: 6,
      step: 1,
      value: resolutionOffset,
      format: value => `+${value} (${(512 / 2 ** value).toFixed(0)} px cells)`,
      onChange: value => {
        resolutionOffset = value;
      }
    });
    context.controls.addToggle({
      label: 'Manual level override',
      value: manual,
      onChange: value => {
        manual = value;
      }
    });
    context.controls.addSlider({
      label: 'Manual Quadbin resolution',
      min: COARSEST_RESOLUTION,
      max: FINEST_RESOLUTION,
      step: 1,
      value: manualResolution,
      onChange: value => {
        manualResolution = value;
      }
    });
    context.controls.addSlider({
      label: 'Sampled points (re-encodes the pyramid)',
      min: 5,
      max: 100,
      step: 5,
      value: keptPercent,
      format: value => `${value}%`,
      onChange: value => {
        keptPercent = value;
        for (let point = 0; point < pointCount; point++) {
          maskValues[point] = hashToUnit(point) * 100 < keptPercent ? 1 : 0;
        }
        maskBuffer.write(maskValues);
        pyramidDirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Roll up the finest table (GPUCellRollup)',
      value: rollupEnabled,
      onChange: value => {
        rollupEnabled = value;
        rollupDirty = value;
        totalsWanted = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<string>({
      label: 'Roll-up parent resolution (one precompiled graph each)',
      options: ROLLUP_RESOLUTIONS.map(resolution => ({
        value: String(resolution),
        label: `res ${resolution} (cell edge ${(40075 / 2 ** resolution).toFixed(1)} km at the equator)`
      })),
      value: String(rollupResolution),
      onChange: value => {
        rollupResolution = Number(value);
        rollupDirty = rollupEnabled;
        totalsWanted = rollupEnabled;
      }
    });
    context.controls.addSelect<'density' | 'mean'>({
      label: 'Color the roll-up by',
      options: [
        {value: 'density', label: 'Points per km²'},
        {value: 'mean', label: `Mean trip time (0 to ${maximumMeanValue.toFixed(0)} min)`}
      ],
      value: 'density',
      onChange: value => {
        rollupColorByMeanValue = value === 'mean';
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Color ceiling (log10 points per km²)',
      min: 1,
      max: 6,
      step: 0.5,
      value: densityLogRange[1],
      onChange: value => {
        densityLogRange = [densityLogRange[0], value];
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Points per km² (log scale, comparable across levels)',
      gradient: {
        colors: [
          [0, 0, 4],
          [66, 10, 104],
          [148, 38, 103],
          [237, 105, 37],
          [252, 255, 164]
        ],
        minimumLabel: `10^${DEFAULT_DENSITY_LOG_RANGE[0]}`,
        maximumLabel: 'ceiling'
      }
    });
    context.controls.addReadout('Points', formatCount(pointCount));
    context.controls.addReadout(
      'Pyramid',
      `${LEVEL_COUNT} levels, res ${FINEST_RESOLUTION} to ${COARSEST_RESOLUTION}, ` +
        `${formatCount(capacity)} rows each`
    );
    const zoomReadout = context.controls.addReadout('Map zoom / level');
    const activeReadout = context.controls.addReadout('Active cells');
    const perLevelReadouts = [0, 1, 2].map(group =>
      context.controls.addReadout(
        `Cells at res ${getLevelResolution(group * 5)} to ${getLevelResolution(
          Math.min(group * 5 + 4, LEVEL_COUNT - 1)
        )}`
      )
    );
    const overflowReadout = context.controls.addReadout('Overflowed levels');
    const pyramidReadout = context.controls.addReadout('Pyramid encodes');
    const rollupCellsReadout = context.controls.addReadout('Roll-up parent cells');
    const rollupCountReadout = context.controls.addReadout('Roll-up count total (fine → parent)');
    const rollupSumReadout = context.controls.addReadout('Roll-up time sum (fine → parent)');
    context.controls.addReadout('Data', trips.attribution);
    context.controls.addNote(
      'Levels are tables in one buffer slab; the level choice, instance count and first row ' +
        'are written by GPUCellLevelSelection from a one-word buffer, so zooming is a buffer write.'
    );

    let pyramidEncodes = 0;

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0],
      zoomLevelIndex: number
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const sources: [Buffer, number][] = [
        [levelCountsBuffer, LEVEL_COUNT * 4],
        [levelOverflowBuffers[0], LEVEL_COUNT * 4],
        [levelOverflowBuffers[1], LEVEL_COUNT * 4],
        [levelTotalsBuffer, 4],
        [drawCommands.buffer, 8] // vertex count, then the GPU-written instance count
      ];
      let offset = 0;
      for (const [sourceBuffer, size] of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size
        });
        offset += size;
      }
      ticket.markEncoded({byteOffset: 0, byteLength: readbackByteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const counts = words.subarray(0, LEVEL_COUNT);
        const overflowsByParity = [
          words.subarray(LEVEL_COUNT, LEVEL_COUNT * 2),
          words.subarray(LEVEL_COUNT * 2, LEVEL_COUNT * 3)
        ];
        const finestTotal = words[LEVEL_COUNT * 3];
        perLevelReadouts.forEach((readout, group) => {
          const parts: string[] = [];
          for (let level = group * 5; level < Math.min(group * 5 + 5, LEVEL_COUNT); level++) {
            parts.push(formatCount(counts[level]));
          }
          readout.setValue(parts.join(' / '));
        });
        const overflowed: number[] = [];
        for (let level = 0; level < LEVEL_COUNT; level++) {
          if (overflowsByParity[level & 1][level]) overflowed.push(getLevelResolution(level));
        }
        overflowReadout.setValue(
          overflowed.length
            ? `res ${overflowed.join(', ')} (unclamped ${formatCount(finestTotal)} at finest)`
            : 'none'
        );
        activeReadout.setValue(
          `${formatCount(words[LEVEL_COUNT * 3 + 2])} cells at res ${getLevelResolution(zoomLevelIndex)}`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    /** Header words: fine rows, fine overflow, draw record (2 words), parent overflow. */
    const TOTALS_HEADER_BYTES = 20;
    const totalsByteLength = TOTALS_HEADER_BYTES + capacity * (4 + 8) * 2;
    const totalsRing = resources.track(
      new GPUReadbackRing(device, {id: 'cell-rollup-totals', byteLength: totalsByteLength})
    );
    const sumWords = (words: Uint32Array, rows: number): bigint => {
      let total = 0n;
      for (let row = 0; row < rows; row++) {
        total += BigInt.asIntN(64, (BigInt(words[row * 2 + 1]) << 32n) | BigInt(words[row * 2]));
      }
      return total;
    };
    const sumCounts = (words: Uint32Array, rows: number): number => {
      let total = 0;
      for (let row = 0; row < rows; row++) total += words[row];
      return total;
    };
    // Compares the finest table with the parent table: counts and fixed-point sums must match.
    const readTotals = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0],
      parentResolution: number
    ) => {
      const ticket = totalsRing.tryAcquire();
      if (!ticket) return;
      const sources: [Buffer, number, number][] = [
        [tableCountBuffers[0], 4, 0],
        [levelOverflowBuffers[0], 4, 4],
        [rollupDrawCommands.buffer, 8, 8],
        [rollupOverflowBuffer, 4, 16],
        [countsSlabs[0], capacity * 4, TOTALS_HEADER_BYTES],
        [finestSumsBuffer, capacity * 8, TOTALS_HEADER_BYTES + capacity * 4],
        [rollupCountsBuffer, capacity * 4, TOTALS_HEADER_BYTES + capacity * 12],
        [rollupSumsBuffer, capacity * 8, TOTALS_HEADER_BYTES + capacity * 16]
      ];
      for (const [sourceBuffer, size, destinationOffset] of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      }
      ticket.markEncoded({byteOffset: 0, byteLength: totalsByteLength});
      totalsPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copy = bytes.slice();
        const header = new Uint32Array(copy.buffer, copy.byteOffset, 5);
        const [fineRows, fineOverflow, , , rollupOverflow] = header;
        const parentRows = new Uint32Array(copy.buffer, copy.byteOffset + 8, 2)[1];
        const arrayWords = (byteOffset: number, length: number) =>
          new Uint32Array(copy.buffer, copy.byteOffset + byteOffset, length);
        const fineCount = sumCounts(arrayWords(TOTALS_HEADER_BYTES, capacity), fineRows);
        const fineSum = sumWords(
          arrayWords(TOTALS_HEADER_BYTES + capacity * 4, capacity * 2),
          fineRows
        );
        const parentCount = sumCounts(
          arrayWords(TOTALS_HEADER_BYTES + capacity * 12, capacity),
          parentRows
        );
        const parentSum = sumWords(
          arrayWords(TOTALS_HEADER_BYTES + capacity * 16, capacity * 2),
          parentRows
        );
        const toMinutes = (sum: bigint) => Number(sum) / GPU_CELL_DEFAULT_SUM_SCALE;
        rollupCellsReadout.setValue(
          `${formatCount(parentRows)} cells at res ${parentResolution} ` +
            `(from ${formatCount(fineRows)} at res ${FINEST_RESOLUTION})` +
            (fineOverflow || rollupOverflow ? ' (overflowed)' : '')
        );
        rollupCountReadout.setValue(
          `${formatCount(fineCount)} → ${formatCount(parentCount)} ` +
            (fineCount === parentCount ? '(conserved)' : '(DIFFERS)')
        );
        rollupSumReadout.setValue(
          `${toMinutes(fineSum).toFixed(1)} → ${toMinutes(parentSum).toFixed(1)} min ` +
            (fineSum === parentSum ? '(bit-exact)' : '(DIFFERS)')
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        totalsPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () =>
        [
          compiledPyramid,
          compiledSelection,
          ...rollupGraphs.values()
        ] as CompiledGPUCommandGraph<never>[],
      encode(commandEncoder, frame) {
        // The pyramid depends only on the points and the mask, never on the camera.
        if (pyramidDirty) {
          compiledPyramid.encode(commandEncoder, {parameters: undefined});
          pyramidDirty = false;
          rollupDirty = rollupEnabled;
          totalsWanted = rollupEnabled;
          pyramidEncodes++;
          pyramidReadout.setValue(String(pyramidEncodes));
        }
        if (rollupEnabled && rollupDirty) {
          rollupGraphs.get(rollupResolution)!.encode(commandEncoder, {parameters: undefined});
          rollupDirty = false;
        }
        if (rollupEnabled && totalsWanted && !totalsPending) {
          totalsWanted = false;
          void readTotals(commandEncoder, rollupResolution);
        }
        const zoom = frame.viewport.zoom;
        const targetResolution = manual ? manualResolution : Math.round(zoom) + resolutionOffset;
        const clampedResolution = Math.min(
          FINEST_RESOLUTION,
          Math.max(COARSEST_RESOLUTION, targetResolution)
        );
        const levelIndex = FINEST_RESOLUTION - clampedResolution;
        activeLevel.write(Uint32Array.of(levelIndex));
        zoomReadout.setValue(`${zoom.toFixed(2)} / res ${clampedResolution}`);
        compiledSelection.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 1 && !readbackPending) {
          void readSummary(commandEncoder, levelIndex);
        }
      },
      getLayers(): Layer[] {
        if (rollupEnabled) {
          return [
            new QuadbinCellLayer({
              id: 'cell-rollup-cells',
              cellsEven: rollupCellsBuffer,
              cellsOdd: rollupCellsBuffer,
              countsEven: rollupCountsBuffer,
              countsOdd: rollupCountsBuffer,
              values: rollupSumValuesBuffer,
              colorByMeanValue: rollupColorByMeanValue,
              meanValueRange: [0, maximumMeanValue],
              activeLevel: zeroWordBuffer,
              firstRow: zeroWordBuffer,
              drawCommands: rollupDrawCommands,
              densityLogRange,
              inset: 0.04
            })
          ];
        }
        return [
          new QuadbinCellLayer({
            id: 'cell-pyramid-cells',
            cellsEven: cellsSlabs[0],
            cellsOdd: cellsSlabs[1],
            countsEven: countsSlabs[0],
            countsOdd: countsSlabs[1],
            activeLevel: activeLevel.buffer,
            firstRow: activeFirstRow,
            drawCommands,
            densityLogRange
          })
        ];
      },
      destroy: () => {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
