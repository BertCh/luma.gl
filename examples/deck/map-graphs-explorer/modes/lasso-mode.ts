// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Lasso statistics: `GPURegionStatistics` over ~2.5k San Francisco bike-parking locations. The
 * selection shape (a drawn lasso polygon or a circle) lives in parameter buffers rewritten every
 * frame; the shape KIND is compile-time, so there is one compiled graph per kind and switching
 * kinds never recompiles. The GPU writes a 0/1 selection mask (drawn as highlighted points) and a
 * packed summary (count, value statistics, histogram, flags) that is read back every few frames
 * through `GPURegionStatisticsReadback` for the readouts.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUGridIndex,
  type CompiledGPUCommandGraph,
  type GPUGridIndexView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPURegionStatisticsSummaryLength, GPURegionStatistics, GPURegionStatisticsReadback, type GPURegionStatisticsResult} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer, submitGraph} from '@luma.gl/experimental/UNRESOLVED';
import {createSeededRandom, LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsPointLayer, MapGraphsSegmentLayer} from '../map-graphs-layers';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {
  type CompiledGraphTiming,
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph
} from './vector-timing';

/** Compile-time lasso vertex capacity. */
const VERTEX_CAPACITY = 256;
const CIRCLE_SEGMENTS = 64;
const OUTLINE_CAPACITY = VERTEX_CAPACITY;
const HISTOGRAM_BINS = 12;
/** Fixed histogram domain in parking spaces; larger values count as "outside". */
const HISTOGRAM_DOMAIN: readonly [number, number] = [0, 24];
const READBACK_INTERVAL_FRAMES = 10;
/** Minimum pointer travel in CSS pixels before a new lasso vertex is appended. */
const MINIMUM_VERTEX_SPACING_PIXELS = 5;
/** Cells per axis of the grid index over the (padded) point bounds. */
const GRID_SIZE = [256, 256] as const;
/** Point count of the replicated source. */
const JITTERED_POINT_COUNT = 1_000_000;
/** Maximum per-axis jitter of replicated points, in meters. */
const JITTER_METERS = 60;
/** Above this many points the point layers draw smaller, fainter dots. */
const DENSE_POINT_COUNT = 100_000;
const CPU_CHECK_IMMEDIATE_LIMIT = 100_000;
const CPU_CHECK_DEBOUNCE_MILLISECONDS = 300;
/** Selectable grid candidate capacities as a share of the point count. */
const CANDIDATE_FRACTIONS = [1, 0.5, 0.25, 0.1];
const DEFAULT_CANDIDATE_FRACTION = 0.5;
/** Frames after (re)creation before the automatic grid on/off measurement. */
const AUTO_MEASURE_FRAME = 40;
const SPARKLINE_BLOCKS = '▁▂▃▄▅▆▇█';

/** Lasso around Market St between the Embarcadero and Van Ness, as `[longitude, latitude]`. */
const PRESET_LASSO: readonly (readonly [number, number])[] = [
  [-122.395, 37.8],
  [-122.406, 37.7945],
  [-122.416, 37.787],
  [-122.4235, 37.781],
  [-122.4215, 37.771],
  [-122.412, 37.772],
  [-122.402, 37.779],
  [-122.393, 37.784],
  [-122.3905, 37.792]
];

type SelectionKind = 'polygon' | 'radius';
type PointSource = 'racks' | 'jittered';

/** Buffers, grid index and compiled graphs for one point source. */
type PointSet = {
  count: number;
  positions: Float32Array;
  spaces: Float32Array;
  bounds: [number, number, number, number];
  /** Owns every GPU resource below plus the graphs compiled over them. */
  resources: MapGraphsResources;
  positionsBuffer: Buffer;
  valuesBuffer: Buffer;
  maskBuffer: Buffer;
  cellOffsets: Buffer;
  objectIds: Buffer;
  indexCount: Buffer;
  indexOverflow: Buffer;
  compiledByKind: Map<SelectionKind, CompiledGPUCommandGraph<void>>;
};
type Point = readonly [number, number];

export const lassoMode: MapGraphsModeDefinition = {
  id: 'lasso',
  title: 'Lasso stats',
  recipes: ['GPURegionStatistics', 'GPURegionStatisticsReadback'],
  description:
    'Bike-parking racks inside a lasso or circle, summarized on the GPU: count, spaces sum/mean/' +
    'min/max and a histogram. Use "Draw lasso" then drag on the map, or switch to a circle and ' +
    'click. The shape is a per-frame parameter; only a small summary is read back.',
  initialViewState: {longitude: -122.415, latitude: 37.78, zoom: 13.5},

  async create(context) {
    const bikeParking = await context.data.getSanFranciscoBikeParking();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(bikeParking.origin);
    const resources = new MapGraphsResources(device, 'lasso');
    const summaryLength = getGPURegionStatisticsSummaryLength(HISTOGRAM_BINS);

    const summaryBuffer = resources.createBuffer('summary', summaryLength * 4);
    const outlineBuffer = resources.createBuffer('outline', OUTLINE_CAPACITY * 16);
    const vertexBuffer = resources.createBuffer('vertices', VERTEX_CAPACITY * 8);
    const vertexCount = resources.createParameterBuffer('vertex-count', 'uint32', 1);
    const circleParameters = resources.createParameterBuffer('circle', 'float32', 3);
    const readback = resources.track(
      new GPURegionStatisticsReadback(device, {id: 'lasso-readback', binCount: HISTOGRAM_BINS})
    );

    // Interaction state (meters around the dataset origin).
    const presetLasso = getPresetLasso(bikeParking.positions, projection);
    let kind: SelectionKind = 'polygon';
    let vertices: Point[] = [...presetLasso];
    let circleCenter: Point = getCentroid(vertices);
    let circleRadius = 700;
    let circleCleared = false;
    let drawingArmed = false;
    let drawing = false;
    let lastVertexPixel: Point | null = null;
    let destroyed = false;
    let cpuCount = 0;
    let cpuSignature = '';
    let cpuTimer: ReturnType<typeof setTimeout> | undefined;
    let latestResult: GPURegionStatisticsResult | null = null;

    // Compile-time options.
    let pointSource: PointSource = 'racks';
    let useGridIndex = false;
    let useMask = true;
    let candidateFraction = DEFAULT_CANDIDATE_FRACTION;
    let pointSetCount = 0;
    let pointSet = createPointSet(pointSource);
    let compiledGraphs = [...pointSet.compiledByKind.values()];

    /** Positions, values, mask and grid index for one point source, plus the graphs over them. */
    function createPointSet(source: PointSource): PointSet {
      const setResources = new MapGraphsResources(device, `lasso-set-${pointSetCount++}`);
      const {positions, spaces} =
        source === 'racks'
          ? {positions: bikeParking.positions, spaces: bikeParking.spaces}
          : createJitteredCopies(bikeParking.positions, bikeParking.spaces);
      const count = spaces.length;
      const bounds = getPaddedBounds(positions);
      const positionsBuffer = setResources.createBuffer('positions', positions);
      const valuesBuffer = setResources.createBuffer('spaces', spaces);
      const maskBuffer = setResources.createBuffer('mask', count * 4);
      const cellCount = GRID_SIZE[0] * GRID_SIZE[1];
      const cellOffsets = setResources.createBuffer('cell-offsets', (cellCount + 1) * 4);
      const objectIds = setResources.createBuffer('object-ids', count * 4);
      const indexCount = setResources.createBuffer('index-count', 4);
      const indexOverflow = setResources.createBuffer('index-overflow', 4);
      const set: PointSet = {
        count,
        positions,
        spaces,
        bounds,
        resources: setResources,
        positionsBuffer,
        valuesBuffer,
        maskBuffer,
        cellOffsets,
        objectIds,
        indexCount,
        indexOverflow,
        compiledByKind: new Map()
      };
      // The positions are static, so the grid index is built once in its own graph, submitted on
      // its own encoder here (outside Deck's frame) and shared by every region graph.
      const indexGraph = new GPUCommandGraph<void>(device, {id: `lasso-index-${pointSetCount}`});
      const view = importGridIndexView(indexGraph, set);
      indexGraph.add(
        new GPUGridIndex({
          id: `lasso-grid-index-${pointSetCount}`,
          positions: importGraphBuffer(
            indexGraph,
            'positions',
            positionsBuffer,
            'float32x2',
            count
          ),
          gridSize: GRID_SIZE,
          bounds,
          cellOffsets: view.cellOffsets,
          objectIds: view.objectIds,
          count: view.count,
          overflow: view.overflow
        })
      );
      const compiledIndex = setResources.track(indexGraph.compile());
      submitGraph(device, compiledIndex, undefined);
      for (const [graphKind, graph] of compileGraphs(set, useGridIndex, useMask, 'main')) {
        set.compiledByKind.set(graphKind, graph);
      }
      return set;
    }

    function importGridIndexView(graph: GPUCommandGraph<void>, set: PointSet): GPUGridIndexView {
      const cellCount = GRID_SIZE[0] * GRID_SIZE[1];
      return {
        gridSize: GRID_SIZE,
        bounds: set.bounds,
        cellOffsets: importGraphBuffer(
          graph,
          'cell-offsets',
          set.cellOffsets,
          'uint32',
          cellCount + 1
        ),
        objectIds: importGraphBuffer(graph, 'object-ids', set.objectIds, 'uint32', set.count),
        count: importGraphBuffer(graph, 'index-count', set.indexCount, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'index-overflow', set.indexOverflow, 'uint32', 1)
      } as GPUGridIndexView;
    }

    /**
     * Compiles one region-statistics graph per selection kind (or only `kinds`) over `set`, with
     * or without the grid candidate path and the selection mask. Every graph reuses the same
     * caller-owned buffers.
     */
    function compileGraphs(
      set: PointSet,
      withGridIndex: boolean,
      withMask: boolean,
      tag: string,
      kinds: readonly SelectionKind[] = ['polygon', 'radius']
    ): Map<SelectionKind, CompiledGPUCommandGraph<void>> {
      const compiled = new Map<SelectionKind, CompiledGPUCommandGraph<void>>();
      for (const graphKind of kinds) {
        const id = `lasso-${graphKind}-${tag}-${withGridIndex ? 'grid' : 'brute'}`;
        const graph = new GPUCommandGraph<void>(device, {id});
        graph.add(
          new GPURegionStatistics({
            id,
            selection:
              graphKind === 'polygon'
                ? {
                    kind: 'polygon',
                    vertices: importGraphBuffer(
                      graph,
                      'vertices',
                      vertexBuffer,
                      'float32x2',
                      VERTEX_CAPACITY
                    ),
                    vertexCount: vertexCount.importToGraph(graph)
                  }
                : {kind: 'radius', circle: circleParameters.importToGraph(graph)},
            positions: importGraphBuffer(
              graph,
              'positions',
              set.positionsBuffer,
              'float32x2',
              set.count
            ),
            values: importGraphBuffer(graph, 'spaces', set.valuesBuffer, 'float32', set.count),
            histogram: {binCount: HISTOGRAM_BINS, domain: HISTOGRAM_DOMAIN},
            outputMask: withMask
              ? importGraphBuffer(graph, 'mask', set.maskBuffer, 'uint32', set.count)
              : undefined,
            summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'uint32', summaryLength),
            spatialIndex: withGridIndex
              ? {
                  kind: 'grid',
                  index: importGridIndexView(graph, set),
                  // Statistics kernels shrink to this many rows; a region whose cells hold more
                  // candidates sets `candidatesTruncated` (shown in Flags).
                  candidateCapacity: Math.max(1, Math.ceil(set.count * candidateFraction))
                }
              : undefined
          })
        );
        compiled.set(graphKind, set.resources.track(graph.compile()));
      }
      return compiled;
    }

    /** Recompiles both graphs for the current options; the old ones are freed after two frames. */
    function rebuildGraphs(): void {
      const old = [...pointSet.compiledByKind.values()];
      pointSet.compiledByKind = compileGraphs(pointSet, useGridIndex, useMask, 'main');
      compiledGraphs = [...pointSet.compiledByKind.values()];
      const owner = pointSet;
      destroyAfterFrames(() => {
        for (const graph of old) owner.resources.release(graph);
      });
      updateOptionReadouts();
    }

    function destroyAfterFrames(callback: () => void): void {
      // Deck applies layer changes on its next animation frame; wait two frames before freeing
      // what the previous frame still used.
      requestAnimationFrame(() => requestAnimationFrame(callback));
    }

    function replacePointSet(source: PointSource): void {
      measurement.abort();
      const previous = pointSet;
      const previousDone = measurement.done;
      pointSource = source;
      pointSet = createPointSet(source);
      compiledGraphs = [...pointSet.compiledByKind.values()];
      cpuSignature = '';
      latestResult = null;
      framesSinceBuild = 0;
      autoMeasurePending = true;
      resetTimingReadouts();
      pointCountReadout.setValue(formatCount(pointSet.count));
      updateOptionReadouts();
      context.updateLayers();
      void previousDone.then(() =>
        destroyAfterFrames(() => {
          previous.resources.destroy();
        })
      );
    }

    // Timing: grid index on vs off for the ACTIVE kind, outside Deck's frame.
    let framesSinceBuild = 0;
    let autoMeasurePending = true;
    const measurement = {
      running: false,
      controller: new AbortController(),
      done: Promise.resolve(),
      abort() {
        this.controller.abort();
      }
    };

    async function measureGridIndex(): Promise<void> {
      if (measurement.running || destroyed) return;
      measurement.running = true;
      measurement.controller = new AbortController();
      const {signal} = measurement.controller;
      const target = pointSet;
      const measuredKind = kind;
      const gridIsActive = useGridIndex;
      const current = target.compiledByKind.get(measuredKind);
      let temporary: CompiledGPUCommandGraph<void> | undefined;
      let finish: () => void = () => {};
      measurement.done = new Promise<void>(resolve => (finish = resolve));
      try {
        if (!current) return;
        timingStatusReadout.setValue('measuring...');
        // Temporary graph for the other setting; not returned from getCompiledGraphs.
        temporary = compileGraphs(target, !gridIsActive, useMask, 'measure', [measuredKind]).get(
          measuredKind
        );
        if (!temporary) return;
        const options = {parameters: undefined, completionBuffer: summaryBuffer, signal};
        // Two alternating rounds, keeping the faster of each, so GPU clock ramp-up and the order
        // of measurement do not bias either side.
        const warmUp = {...options, warmUpRuns: 4};
        const firstCurrent = await measureCompiledGraph(device, current, warmUp);
        const firstOther = await measureCompiledGraph(device, temporary, warmUp);
        const secondCurrent = await measureCompiledGraph(device, current, options);
        const secondOther = await measureCompiledGraph(device, temporary, options);
        const currentTiming = getFaster(firstCurrent, secondCurrent);
        const otherTiming = getFaster(firstOther, secondOther);
        if (destroyed || signal.aborted) return;
        const gridTiming = gridIsActive ? currentTiming : otherTiming;
        const bruteTiming = gridIsActive ? otherTiming : currentTiming;
        gridOnReadout.setValue(formatCompiledGraphTiming(gridTiming));
        gridOffReadout.setValue(formatCompiledGraphTiming(bruteTiming));
        speedupReadout.setValue(
          `${formatSpeedup(bruteTiming.milliseconds, gridTiming.milliseconds)} (${formatCount(target.count)} points, ${measuredKind})`
        );
        timingStatusReadout.setValue('done');
      } catch (error) {
        if (!destroyed && !signal.aborted) timingStatusReadout.setValue(`failed: ${error}`);
      } finally {
        if (temporary) target.resources.release(temporary);
        measurement.running = false;
        finish();
      }
    }

    function resetTimingReadouts(): void {
      gridOnReadout.setValue('...');
      gridOffReadout.setValue('...');
      speedupReadout.setValue('...');
      timingStatusReadout.setValue('pending');
    }

    // Controls.
    context.controls.addSelect<SelectionKind>({
      label: 'Selection shape (one compiled graph each)',
      options: [
        {value: 'polygon', label: 'Lasso polygon'},
        {value: 'radius', label: 'Circle'}
      ],
      value: kind,
      onChange: value => {
        kind = value;
        drawingArmed = false;
        context.setMapDragEnabled(true);
        circleCleared = false;
        updateStatus();
      }
    });
    context.controls.addSlider({
      label: 'Circle radius (per-frame parameter)',
      min: 100,
      max: 2500,
      step: 50,
      value: circleRadius,
      format: value => `${value} m`,
      onChange: value => {
        circleRadius = value;
        circleCleared = false;
      }
    });
    context.controls.addButton({
      label: 'Draw lasso',
      onClick: () => {
        kind = 'polygon';
        drawingArmed = true;
        context.setMapDragEnabled(false);
        updateStatus();
      }
    });
    context.controls.addButton({
      label: 'Restore preset lasso',
      onClick: () => {
        vertices = [...presetLasso];
        circleCenter = getCentroid(vertices);
        circleCleared = false;
        updateStatus();
      }
    });
    context.controls.addButton({
      label: 'Clear selection',
      onClick: () => {
        vertices = [];
        circleCleared = true;
        updateStatus();
      }
    });
    context.controls.addSelect<PointSource>({
      label: 'Points (compile-time)',
      options: [
        {value: 'racks', label: '2.5k racks (data)'},
        {value: 'jittered', label: '1M jittered copies'}
      ],
      value: pointSource,
      onChange: value => {
        if (value !== pointSource) replacePointSet(value);
      }
    });
    context.controls.addToggle({
      label: 'Grid index (spatialIndex, compile-time: rebuilds graphs)',
      value: useGridIndex,
      onChange: value => {
        useGridIndex = value;
        rebuildGraphs();
      }
    });
    context.controls.addSelect<string>({
      label: 'Candidate capacity (compile-time, share of points)',
      options: CANDIDATE_FRACTIONS.map(fraction => ({
        value: String(fraction),
        label: fraction === 1 ? 'exact (all points)' : `${fraction * 100}% of points`
      })),
      value: String(candidateFraction),
      onChange: value => {
        candidateFraction = Number(value);
        rebuildGraphs();
      }
    });
    context.controls.addToggle({
      label: 'Selection mask output (compile-time: off = statistics only)',
      value: useMask,
      onChange: value => {
        useMask = value;
        rebuildGraphs();
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Measure grid index on vs off',
      onClick: () => void measureGridIndex()
    });
    context.controls.addLegend({
      title: 'Bike-parking locations',
      entries: [
        {color: [120, 150, 200, 150], label: 'All locations'},
        {color: [255, 190, 60, 255], label: 'Selected (GPU mask)'},
        {color: [90, 240, 220, 255], label: 'Selection outline'}
      ]
    });
    const selectedReadout = context.controls.addReadout('Selected');
    const cpuReadout = context.controls.addReadout('CPU check');
    const valueReadout = context.controls.addReadout('Valid values');
    const sumReadout = context.controls.addReadout('Spaces sum / mean');
    const rangeReadout = context.controls.addReadout('Spaces min / max');
    const histogramReadout = context.controls.addReadout(`Histogram 0-${HISTOGRAM_DOMAIN[1]}`);
    const outsideReadout = context.controls.addReadout('Above histogram');
    const flagReadout = context.controls.addReadout('Flags');
    const pointCountReadout = context.controls.addReadout('Locations', formatCount(pointSet.count));
    const optionReadout = context.controls.addReadout('Region path');
    const gridOnReadout = context.controls.addReadout('Grid index on', '...');
    const gridOffReadout = context.controls.addReadout('Grid index off', '...');
    const speedupReadout = context.controls.addReadout('Speedup', '...');
    const timingStatusReadout = context.controls.addReadout('Timing', 'pending');
    context.controls.addReadout('Data', bikeParking.attribution);
    context.controls.addNote(
      'Summary readback runs every 10 frames through a bounded GPUReadbackRing; the CPU check ' +
        'recounts the same shape on the CPU.'
    );
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise ' +
        'wall clock / 8 repetitions (upper bound); best of two alternating rounds. The grid index is built once per point set ' +
        'and excluded from these numbers; with the mask on, the mask scatter stays O(N).'
    );

    function updateOptionReadouts(): void {
      optionReadout.setValue(
        (useGridIndex
          ? `grid ${GRID_SIZE[0]}x${GRID_SIZE[1]}, capacity ${formatCount(Math.max(1, Math.ceil(pointSet.count * candidateFraction)))}`
          : 'brute force') + (useMask ? ' + mask' : ', statistics only')
      );
    }
    updateOptionReadouts();

    function updateStatus(): void {
      if (drawingArmed) {
        context.setStatus(
          drawing ? 'Drawing lasso... release to close' : 'Drag on the map to draw a lasso'
        );
      } else if (kind === 'radius') {
        context.setStatus('Click or drag on the map to move the circle');
      } else {
        context.setStatus('');
      }
    }
    updateStatus();

    /** Uploads the active shape and rebuilds the CPU outline. */
    function writeSelection(): void {
      const outline = new Float32Array(OUTLINE_CAPACITY * 4).fill(Number.NaN);
      if (kind === 'polygon') {
        const count = Math.min(vertices.length, VERTEX_CAPACITY);
        const packed = new Float32Array(VERTEX_CAPACITY * 2);
        vertices.slice(0, count).forEach((vertex, index) => packed.set(vertex, index * 2));
        vertexBuffer.write(packed);
        vertexCount.write(Uint32Array.of(count));
        if (count >= 2) {
          for (let index = 0; index < count; index++) {
            const next = vertices[(index + 1) % count];
            outline.set([...vertices[index], ...next], index * 4);
          }
        }
        const polygon = vertices.slice(0, count);
        const owner = pointSet;
        updateCpuCount(`polygon:${polygon.join(',')}`, () =>
          countInsidePolygon(owner.positions, polygon)
        );
      } else {
        const radius = circleCleared ? Number.NaN : circleRadius;
        circleParameters.write(Float32Array.of(circleCenter[0], circleCenter[1], radius));
        if (!circleCleared) {
          for (let index = 0; index < CIRCLE_SEGMENTS; index++) {
            const angle0 = (index / CIRCLE_SEGMENTS) * Math.PI * 2;
            const angle1 = ((index + 1) / CIRCLE_SEGMENTS) * Math.PI * 2;
            outline.set(
              [
                circleCenter[0] + Math.cos(angle0) * circleRadius,
                circleCenter[1] + Math.sin(angle0) * circleRadius,
                circleCenter[0] + Math.cos(angle1) * circleRadius,
                circleCenter[1] + Math.sin(angle1) * circleRadius
              ],
              index * 4
            );
          }
        }
        const center = circleCenter;
        const owner = pointSet;
        updateCpuCount(`radius:${circleCleared ? 'none' : `${center},${circleRadius}`}`, () =>
          circleCleared ? 0 : countInsideCircle(owner.positions, center, circleRadius)
        );
      }
      outlineBuffer.write(outline);
      if (latestResult) showResult(latestResult);
    }

    /**
     * Recounts on the CPU only when the shape changes. Large point sets are recounted after the
     * shape has been still for a moment so dragging stays smooth.
     */
    function updateCpuCount(signature: string, count: () => number): void {
      if (signature === cpuSignature) return;
      cpuSignature = signature;
      clearTimeout(cpuTimer);
      if (pointSet.count <= CPU_CHECK_IMMEDIATE_LIMIT) {
        cpuCount = count();
        return;
      }
      const owner = pointSet;
      cpuCount = -1;
      cpuTimer = setTimeout(() => {
        if (destroyed || owner !== pointSet || signature !== cpuSignature) return;
        cpuCount = count();
        if (latestResult) showResult(latestResult);
      }, CPU_CHECK_DEBOUNCE_MILLISECONDS);
    }

    function showResult(result: GPURegionStatisticsResult): void {
      latestResult = result;
      const matches = result.selectedCount === cpuCount;
      const pointCount = pointSet.count;
      selectedReadout.setValue(
        `${formatCount(result.selectedCount)} of ${formatCount(pointCount)}`
      );
      cpuReadout.setValue(
        cpuCount < 0
          ? 'recounting...'
          : `${formatCount(cpuCount)} ${matches ? '(match)' : '(differs or in flight)'}`
      );
      valueReadout.setValue(formatCount(result.valueCount));
      sumReadout.setValue(`${formatCount(result.sum)} / ${result.mean.toFixed(2)}`);
      rangeReadout.setValue(`${result.minimum} / ${result.maximum}`);
      histogramReadout.setValue(formatSparkline(result.histogram));
      outsideReadout.setValue(formatCount(result.histogramOutsideCount));
      const flags = [
        result.regionTruncated ? 'region truncated' : '',
        result.selectionTruncated ? 'selection truncated' : '',
        result.candidatesTruncated ? 'candidates truncated' : ''
      ].filter(Boolean);
      flagReadout.setValue(flags.length > 0 ? flags.join(', ') : 'none');
    }

    function toMeters(event: MapGraphsPointerEvent): Point | null {
      return event.coordinate ? projection.project(event.coordinate[0], event.coordinate[1]) : null;
    }

    function setCircleCenter(event: MapGraphsPointerEvent): boolean {
      const center = toMeters(event);
      if (!center) return false;
      circleCenter = center;
      circleCleared = false;
      return true;
    }

    function appendVertex(event: MapGraphsPointerEvent, force = false): void {
      const point = toMeters(event);
      if (!point || vertices.length >= VERTEX_CAPACITY) return;
      if (!force && lastVertexPixel) {
        const distance = Math.hypot(
          event.pixel[0] - lastVertexPixel[0],
          event.pixel[1] - lastVertexPixel[1]
        );
        if (distance < MINIMUM_VERTEX_SPACING_PIXELS) return;
      }
      vertices.push(point);
      lastVertexPixel = event.pixel;
    }

    function finishDrawing(): void {
      drawing = false;
      drawingArmed = false;
      lastVertexPixel = null;
      context.setMapDragEnabled(true);
      updateStatus();
    }

    let nextTicketFrame = 0;
    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => compiledGraphs,
      encode(commandEncoder, frame) {
        const compiled = pointSet.compiledByKind.get(kind);
        if (!compiled) return;
        if (autoMeasurePending && ++framesSinceBuild >= AUTO_MEASURE_FRAME) {
          autoMeasurePending = false;
          void measureGridIndex();
        }
        // Per-frame parameters: shape vertices or circle, rewritten without recompiling.
        writeSelection();
        compiled.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex >= nextTicketFrame) {
          const ticket = readback.encodeRead(commandEncoder, summaryBuffer);
          if (ticket) {
            nextTicketFrame = frame.frameIndex + READBACK_INTERVAL_FRAMES;
            readback
              .read(ticket)
              .then(result => {
                if (!destroyed) showResult(result);
              })
              .catch(() => {
                // The mode was destroyed or the ring was torn down while the read was in flight.
              });
          }
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [
          bikeParking.origin[0],
          bikeParking.origin[1],
          0
        ];
        const {positionsBuffer, maskBuffer, count: pointCount} = pointSet;
        const dense = pointCount > DENSE_POINT_COUNT;
        const layers: Layer[] = [
          new MapGraphsPointLayer({
            id: 'lasso-all-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: dense ? 1 : 2.4,
            color: dense ? [120, 150, 200, 40] : [120, 150, 200, 150]
          }),
          ...(useMask
            ? [
                new MapGraphsPointLayer({
                  id: 'lasso-selected-points',
                  coordinateOrigin,
                  positions: positionsBuffer,
                  instanceCount: pointCount,
                  radiusPixels: dense ? 1.2 : 4,
                  values: maskBuffer,
                  valueFormat: 'uint32',
                  colormap: 'mask',
                  color: dense ? [255, 190, 60, 80] : [255, 190, 60, 255],
                  noDataColor: [0, 0, 0, 0]
                })
              ]
            : []),
          new MapGraphsSegmentLayer({
            id: 'lasso-outline',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: OUTLINE_CAPACITY,
            widthPixels: 2.5,
            color: [90, 240, 220, 255]
          })
        ];
        return layers;
      },
      onClick(event) {
        if (kind === 'radius') return setCircleCenter(event);
        return false;
      },
      onDragStart(event) {
        if (kind === 'radius') return setCircleCenter(event);
        if (!drawingArmed) return false;
        drawing = true;
        vertices = [];
        lastVertexPixel = null;
        appendVertex(event, true);
        updateStatus();
        return true;
      },
      onDrag(event) {
        if (kind === 'radius') {
          setCircleCenter(event);
        } else if (drawing) {
          appendVertex(event);
          if (vertices.length >= VERTEX_CAPACITY) finishDrawing();
        }
      },
      onDragEnd(event) {
        if (kind === 'polygon' && drawing) {
          appendVertex(event);
          finishDrawing();
        }
      },
      destroy() {
        destroyed = true;
        measurement.abort();
        clearTimeout(cpuTimer);
        context.setMapDragEnabled(true);
        const owner = pointSet;
        void measurement.done.then(() => {
          owner.resources.destroy();
          resources.destroy();
        });
      }
    };
    return instance;
  }
};

/**
 * Returns the preset lasso in meters. When it selects almost nothing (for example for synthetic
 * data) it falls back to a ring around the point with the most neighbors within 800 m.
 */
function getPresetLasso(positions: Float32Array, projection: LocalMetricProjection): Point[] {
  const preset = PRESET_LASSO.map(([longitude, latitude]) =>
    projection.project(longitude, latitude)
  ) as Point[];
  if (countInsidePolygon(positions, preset) >= 30) return preset;
  const neighborRadiusSquared = 800 * 800;
  let bestIndex = 0;
  let bestNeighbors = -1;
  const pointCount = positions.length / 2;
  for (let index = 0; index < pointCount; index++) {
    let neighbors = 0;
    for (let other = 0; other < pointCount; other++) {
      const dx = positions[other * 2] - positions[index * 2];
      const dy = positions[other * 2 + 1] - positions[index * 2 + 1];
      if (dx * dx + dy * dy <= neighborRadiusSquared) neighbors++;
    }
    if (neighbors > bestNeighbors) {
      bestNeighbors = neighbors;
      bestIndex = index;
    }
  }
  const center = [positions[bestIndex * 2], positions[bestIndex * 2 + 1]];
  return Array.from({length: 10}, (_, index) => {
    const angle = (index / 10) * Math.PI * 2;
    const radius = index % 2 === 0 ? 1000 : 700;
    return [center[0] + Math.cos(angle) * radius, center[1] + Math.sin(angle) * radius] as Point;
  });
}

function getCentroid(polygon: readonly Point[]): Point {
  const sum = polygon.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  return [sum[0] / Math.max(polygon.length, 1), sum[1] / Math.max(polygon.length, 1)];
}

/** CPU even-odd point-in-polygon count, used only to cross-check the GPU summary. */
function countInsidePolygon(positions: Float32Array, polygon: readonly Point[]): number {
  if (polygon.length < 3) return 0;
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    let inside = false;
    for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
      const [xi, yi] = polygon[index];
      const [xj, yj] = polygon[previous];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside) count++;
  }
  return count;
}

function countInsideCircle(positions: Float32Array, center: Point, radius: number): number {
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    if (Math.hypot(positions[row * 2] - center[0], positions[row * 2 + 1] - center[1]) <= radius) {
      count++;
    }
  }
  return count;
}

/** Renders histogram bins as unicode blocks scaled to the tallest bin, plus the tallest count. */
function formatSparkline(histogram: Uint32Array): string {
  const maximum = Math.max(...histogram, 1);
  const blocks = Array.from(histogram, count =>
    count === 0
      ? '·'
      : SPARKLINE_BLOCKS[
          Math.min(
            SPARKLINE_BLOCKS.length - 1,
            Math.floor((count / maximum) * SPARKLINE_BLOCKS.length)
          )
        ]
  ).join('');
  return `${blocks} (max bin ${formatCount(maximum)})`;
}

function getFaster(first: CompiledGraphTiming, second: CompiledGraphTiming): CompiledGraphTiming {
  return second.milliseconds < first.milliseconds ? second : first;
}

/** Replicates the racks (and their spaces) to a million points with deterministic jitter. */
function createJitteredCopies(
  positions: Float32Array,
  spaces: Float32Array
): {positions: Float32Array; spaces: Float32Array} {
  const sourceCount = spaces.length;
  const random = createSeededRandom(4242);
  const jitteredPositions = new Float32Array(JITTERED_POINT_COUNT * 2);
  const jitteredSpaces = new Float32Array(JITTERED_POINT_COUNT);
  for (let index = 0; index < JITTERED_POINT_COUNT; index++) {
    const source = index % sourceCount;
    jitteredPositions[index * 2] = positions[source * 2] + (random() * 2 - 1) * JITTER_METERS;
    jitteredPositions[index * 2 + 1] =
      positions[source * 2 + 1] + (random() * 2 - 1) * JITTER_METERS;
    jitteredSpaces[index] = spaces[source];
  }
  return {positions: jitteredPositions, spaces: jitteredSpaces};
}

/** Point bounds padded by a few percent so cells never clip the outermost points. */
function getPaddedBounds(positions: Float32Array): [number, number, number, number] {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const padding = 0.02 * Math.max(maximumX - minimumX, maximumY - minimumY, 1);
  return [minimumX - padding, minimumY - padding, maximumX + padding, maximumY + padding];
}
