// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  encodeGPUMapMatchingParameters,
  GPUMapMatching,
  GPU_MAP_MATCHING_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {GPULineMerge} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {ACCURACY_COLORS} from './b10-scene-constants';
import {
  buildDenseDirectedGraph,
  buildRoadGraph,
  ROAD_CLASS_NAMES,
  SegmentIndex
} from './b10-road-graph';

/** Option state of the map-matching scene. */
export type MapMatchingOptions = {
  extraNoise: number;
  noiseSeed: number;
  sigma: number;
  beta: number;
  searchRadius: number;
  routeFactor: number;
  routeSlack: number;
  candidateCount: '2' | '4' | '6' | '8';
  routeNodeBudget: '16' | '32' | '64' | '128';
  cellSize: '40' | '60' | '80' | '100';
  trackFocus: number;
  showRaw: boolean;
  showTruth: boolean;
  showMatched: boolean;
  matchedColor: 'accuracy' | 'plain';
  roadStyle: 'plain' | 'chains' | 'off';
};

const REMATCH_INTERVAL_SECONDS = 0.15;
const NONE = 0xffffffff;

/** Gaussian noise per point from a hash of the point index and a seed (Box-Muller). */
const JITTER_DECLARATIONS = /* wgsl */ `
fn hashValue(value: u32) -> u32 {
  let state = value * 747796405u + 2891336653u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn uniformValue(value: u32) -> f32 {
  return (f32(hashValue(value) >> 8u) + 0.5) / 16777216.0;
}`;

type MatchBuild = {
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};

/**
 * HMM map matching (`GPUMapMatching`) of 200 simulated Chicago GPS traces on a polyline-dense
 * version of the street graph, scored against the ground-truth edge of every fix. Emission sigma,
 * transition beta, search radius, route allowances, the extra jitter and the track focus are
 * buffer writes; candidate count, route node budget and edge-grid cell size are compile-time and
 * rebuild the graph. `GPULineMerge` joins the street segments into chains between junctions.
 */
export async function createMapMatching(
  ctx: SceneContext<MapMatchingOptions>
): Promise<SceneInstance<MapMatchingOptions>> {
  const roads = ctx.datasets.get('chicago-roads');
  const traces = ctx.datasets.get('chicago-gps-traces');
  const {device} = ctx;
  const origin = roads.defaultOrigin;
  const roadGraph = buildRoadGraph(roads, origin);
  const dense = buildDenseDirectedGraph(roads, roadGraph);
  const projection = roads.getProjection(origin);
  const segmentIndex = new SegmentIndex(roadGraph.segments, roadGraph.bounds);
  const gpsPositions = traces.projectColumn('vertices', origin);
  const truthPositions = traces.projectColumn('truthPosition', origin);
  const truthEdges = traces.column<Uint32Array>('truthEdge');
  const trackOffsets = traces.column<Uint32Array>('pathOffsets');
  const pointCount = truthEdges.length;
  const trackCount = trackOffsets.length - 1;
  const resources = new SpatialAnalysisResources(device, 'map-matching');

  // Per-point track index and "next point is in the same track".
  const trackOfPoint = new Uint32Array(pointCount);
  const hasNext = new Uint32Array(pointCount);
  for (let track = 0; track < trackCount; track++) {
    for (let point = trackOffsets[track]; point < trackOffsets[track + 1]; point++) {
      trackOfPoint[point] = track;
      hasNext[point] = point + 1 < trackOffsets[track + 1] ? 1 : 0;
    }
  }

  // Edge grid bounds cover the network and every fix (matcher clamps outside points).
  let [minX, minY, maxX, maxY] = roadGraph.bounds;
  minX -= 200;
  minY -= 200;
  maxX += 200;
  maxY += 200;

  const originalBuffer = resources.createBuffer('original-points', gpsPositions);
  const noisyBuffer = resources.createBuffer('noisy-points', pointCount * 8);
  const truthBuffer = resources.createBuffer('truth-points', truthPositions);
  const trackOfPointBuffer = resources.createBuffer('track-of-point', trackOfPoint);
  const hasNextBuffer = resources.createBuffer('has-next', hasNext);
  const truthEdgeBuffer = resources.createBuffer('truth-edges', truthEdges);
  const reverseBuffer = resources.createBuffer('edge-reverse', roadGraph.edgeReverse);
  const rowEdgeBuffer = resources.createBuffer('row-edge', dense.rowEdge);
  const trackOffsetsBuffer = resources.createBuffer('track-offsets', trackOffsets);
  const nodePositionsBuffer = resources.createBuffer('node-positions', dense.nodePositions);
  const offsetsBuffer = resources.createBuffer('offsets', dense.offsets);
  const targetsBuffer = resources.createBuffer('targets', dense.targets);
  const roadSegmentsBuffer = resources.createBuffer('road-segments', roadGraph.segments);
  const jitterParameters = resources.createParameterBuffer('jitter', 'float32', 2);
  const focusParameter = resources.createParameterBuffer('focus', 'uint32', 1);
  const matchingParameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    GPU_MAP_MATCHING_PARAMETER_LENGTH
  );
  const matchedEdges = resources.createBuffer('matched-edges', pointCount * 4);
  const snapDistances = resources.createBuffer('snap-distances', pointCount * 4);
  const snappedPositions = resources.createBuffer('snapped-positions', pointCount * 8);
  const breaks = resources.createBuffer('breaks', pointCount * 4);
  const matchedCount = resources.createBuffer('matched-count', 4);
  const breakCount = resources.createBuffer('break-count', 4);
  const overflow = resources.createBuffer('overflow', 4);
  const rawSegments = resources.createBuffer('raw-segments', pointCount * 16);
  const rawWeights = resources.createBuffer('raw-weights', pointCount * 4);
  const truthSegments = resources.createBuffer('truth-segments', pointCount * 16);
  const truthWeights = resources.createBuffer('truth-weights', pointCount * 4);
  const matchedSegments = resources.createBuffer('matched-segments', pointCount * 16);
  const matchedWeights = resources.createBuffer('matched-weights', pointCount * 4);
  const matchedClasses = resources.createBuffer('matched-classes', pointCount * 4);

  let destroyed = false;
  let dirty = true;
  let lastMatchTime = -Infinity;
  let measuring = false;
  let build: MatchBuild | null = null;
  let chainsEncoded = false;
  let chainCountValue = 0;

  const maxSearchRadius = () => 2 * Number(ctx.options.cellSize);
  function writeParameters(): void {
    const {extraNoise, noiseSeed, sigma, beta, searchRadius, routeFactor, routeSlack, trackFocus} =
      ctx.options;
    jitterParameters.write(Float32Array.of(extraNoise, noiseSeed));
    focusParameter.write(Uint32Array.of(trackFocus));
    matchingParameters.write(
      encodeGPUMapMatchingParameters({
        sigma,
        beta,
        searchRadius: Math.min(searchRadius, maxSearchRadius()),
        routeFactor,
        routeSlack
      })
    );
    dirty = true;
  }

  // ---- Road chains (GPULineMerge): every drawn segment is a two-vertex line ----
  const segmentCount = roadGraph.segmentCount;
  const chainOffsets = resources.createBuffer('chain-offsets', (segmentCount + 1) * 4);
  const chainPositions = resources.createBuffer('chain-positions', segmentCount * 16);
  const chainCount = resources.createBuffer('chain-count', 4);
  const chainSegments = resources.createBuffer('chain-segments', segmentCount * 2 * 16);
  const chainWeights = resources.createBuffer('chain-weights', segmentCount * 2 * 4);
  const chainValues = resources.createBuffer('chain-values', segmentCount * 2 * 4);
  const lineOffsetValues = new Uint32Array(segmentCount + 1);
  for (let line = 0; line <= segmentCount; line++) lineOffsetValues[line] = line * 2;
  const lineOffsets = resources.createBuffer('line-offsets', lineOffsetValues);
  const mergeGraph = new GPUCommandGraph<void>(device, {id: 'map-matching-chains'});
  const chainOffsetsView = importGraphBuffer(
    mergeGraph,
    'chain-offsets',
    chainOffsets,
    'uint32',
    segmentCount + 1
  );
  const chainCountView = importGraphBuffer(mergeGraph, 'chain-count', chainCount, 'uint32', 1);
  const chainPositionsView = importGraphBuffer(
    mergeGraph,
    'chain-positions',
    chainPositions,
    'float32x2',
    segmentCount * 2
  );
  mergeGraph.add(
    new GPULineMerge({
      id: 'chains',
      positions: importGraphBuffer(
        mergeGraph,
        'road-vertices',
        roadSegmentsBuffer,
        'float32x2',
        segmentCount * 2
      ),
      lineOffsets: importGraphBuffer(
        mergeGraph,
        'line-offsets',
        lineOffsets,
        'uint32',
        segmentCount + 1
      ),
      output: {chainOffsets: chainOffsetsView, positions: chainPositionsView, count: chainCountView}
    })
  );
  addKernelPass(mergeGraph, {
    id: 'chain-segments',
    bindings: [
      {name: 'positions', view: chainPositionsView, type: 'f32', access: 'read'},
      {name: 'chainOffsets', view: chainOffsetsView, type: 'u32', access: 'read'},
      {name: 'chainCount', view: chainCountView, type: 'u32', access: 'read'},
      {
        name: 'segments',
        view: importGraphBuffer(
          mergeGraph,
          'chain-segments',
          chainSegments,
          'float32',
          segmentCount * 8
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'weights',
        view: importGraphBuffer(
          mergeGraph,
          'chain-weights',
          chainWeights,
          'float32',
          segmentCount * 2
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'values',
        view: importGraphBuffer(
          mergeGraph,
          'chain-values',
          chainValues,
          'uint32',
          segmentCount * 2
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: segmentCount * 2,
    declarations: `const VERTEX_CAPACITY: u32 = ${segmentCount * 2}u;`,
    body: `let count = chainCount[chainCountOffset];
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (chainOffsets[chainOffsetsOffset + middle + 1u] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let isDrawn = low < count && index + 1u < chainOffsets[chainOffsetsOffset + low + 1u];
  let next = min(index + 1u, VERTEX_CAPACITY - 1u);
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + next * 2u + 1u];
  weights[weightsOffset + index] = select(0.0, 1.0, isDrawn);
  values[valuesOffset + index] = low;`
  });
  const compiledMerge: CompiledGPUCommandGraph<void> = resources.track(mergeGraph.compile());
  const chainReader = new SummaryReader(
    resources,
    'chains',
    [{buffer: chainCount, size: 4}],
    bytes => {
      if (destroyed) return;
      chainCountValue = new Uint32Array(bytes)[0];
      ctx.setReadout(
        'chains',
        `${formatCount(segmentCount)} segments into ${formatCount(chainCountValue)} chains`
      );
    }
  );

  /**
   * Edge-grid entries needed when every edge is listed in each cell its box, grown by the largest
   * search radius (twice the cell size), touches. The default `16 * edgeCount` is too small for
   * the long highway edges of this network.
   */
  function getEntryCapacity(cellSize: number): number {
    const reach = 4 * cellSize;
    let entries = 0;
    for (let node = 0; node < dense.nodeCount; node++) {
      for (let row = dense.offsets[node]; row < dense.offsets[node + 1]; row++) {
        const target = dense.targets[row];
        const dx = Math.abs(dense.nodePositions[target * 2] - dense.nodePositions[node * 2]);
        const dy = Math.abs(
          dense.nodePositions[target * 2 + 1] - dense.nodePositions[node * 2 + 1]
        );
        entries +=
          (Math.floor((dx + reach) / cellSize) + 2) * (Math.floor((dy + reach) / cellSize) + 2);
      }
    }
    return Math.min(entries, 40_000_000);
  }

  function buildMatchGraph(): void {
    if (build) {
      build.reader.stop();
      resources.release(build.compiled);
      build = null;
    }
    const cellSize = Math.max(
      Number(ctx.options.cellSize),
      Math.sqrt(((maxX - minX) * (maxY - minY)) / 2e6)
    );
    const graph = new GPUCommandGraph<void>(device, {id: 'map-matching'});
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const originalView = view('original-points', originalBuffer, 'float32x2', pointCount);
    const noisyView = view('noisy-points', noisyBuffer, 'float32x2', pointCount);
    const matchedEdgesView = view('matched-edges', matchedEdges, 'uint32', pointCount);
    const snappedView = view('snapped-positions', snappedPositions, 'float32x2', pointCount);
    const breaksView = view('breaks', breaks, 'uint32', pointCount);
    const hasNextView = view('has-next', hasNextBuffer, 'uint32', pointCount);
    const trackOfPointView = view('track-of-point', trackOfPointBuffer, 'uint32', pointCount);
    const focusView = focusParameter.importToGraph(graph);
    const truthView = view('truth-points', truthBuffer, 'float32x2', pointCount);

    addKernelPass(graph, {
      id: 'extra-noise',
      bindings: [
        {name: 'original', view: originalView, type: 'f32', access: 'read'},
        {name: 'jitter', view: jitterParameters.importToGraph(graph), type: 'f32', access: 'read'},
        {name: 'noisy', view: noisyView, type: 'f32', access: 'read_write'}
      ],
      invocationCount: pointCount,
      declarations: JITTER_DECLARATIONS,
      body: `let seed = u32(jitter[jitterOffset + 1u]) * 2654435761u;
  let radius = sqrt(-2.0 * log(uniformValue(index * 2u + seed)));
  let angle = 6.2831853 * uniformValue(index * 2u + 1u + seed);
  noisy[noisyOffset + index * 2u] = original[originalOffset + index * 2u] + jitter[jitterOffset] * radius * cos(angle);
  noisy[noisyOffset + index * 2u + 1u] = original[originalOffset + index * 2u + 1u] + jitter[jitterOffset] * radius * sin(angle);`
    });
    graph.add(
      new GPUMapMatching({
        id: 'matching',
        points: noisyView,
        trackOffsets: view('track-offsets', trackOffsetsBuffer, 'uint32', trackCount + 1),
        nodePositions: view('node-positions', nodePositionsBuffer, 'float32x2', dense.nodeCount),
        offsets: view('offsets', offsetsBuffer, 'uint32', dense.nodeCount + 1),
        edgeTargets: view('targets', targetsBuffer, 'uint32', dense.rowCount),
        parameters: matchingParameters.importToGraph(graph),
        candidateCount: Number(ctx.options.candidateCount),
        routeNodeBudget: Number(ctx.options.routeNodeBudget),
        cellSize,
        bounds: {minimum: [minX, minY], maximum: [maxX, maxY]},
        entryCapacity: getEntryCapacity(cellSize),
        output: {
          matchedEdges: matchedEdgesView,
          snapDistances: view('snap-distances', snapDistances, 'float32', pointCount),
          snappedPositions: snappedView,
          breaks: breaksView,
          matchedCount: view('matched-count', matchedCount, 'uint32', 1),
          breakCount: view('break-count', breakCount, 'uint32', 1),
          overflow: view('overflow', overflow, 'uint32', 1)
        }
      })
    );
    const segmentRows = (
      id: string,
      points: typeof noisyView,
      segmentsBuffer: Buffer,
      weightsBuffer: Buffer
    ) => {
      addKernelPass(graph, {
        id,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'hasNext', view: hasNextView, type: 'u32', access: 'read'},
          {name: 'trackOfPoint', view: trackOfPointView, type: 'u32', access: 'read'},
          {name: 'focus', view: focusView, type: 'u32', access: 'read'},
          {
            name: 'segments',
            view: view(`${id}-segments`, segmentsBuffer, 'float32', pointCount * 4),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'weights',
            view: view(`${id}-weights`, weightsBuffer, 'float32', pointCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pointCount,
        declarations: `const POINT_COUNT: u32 = ${pointCount}u;
`,
        body: `let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  let inFocus = focus[focusOffset] == 0u || trackOfPoint[trackOfPointOffset + index] + 1u == focus[focusOffset];
  weights[weightsOffset + index] = select(0.0, 1.0, hasNext[hasNextOffset + index] != 0u && inFocus);`
      });
    };
    segmentRows('raw-rows', noisyView, rawSegments, rawWeights);
    segmentRows('truth-rows', truthView, truthSegments, truthWeights);
    const reverseView = view('edge-reverse', reverseBuffer, 'uint32', roadGraph.edgeCount);
    const rowEdgeView = view('row-edge', rowEdgeBuffer, 'uint32', dense.rowCount);
    const truthEdgeView = view('truth-edges', truthEdgeBuffer, 'uint32', pointCount);
    addKernelPass(graph, {
      id: 'matched-classes',
      bindings: [
        {name: 'matched', view: matchedEdgesView, type: 'u32', access: 'read'},
        {name: 'rowEdge', view: rowEdgeView, type: 'u32', access: 'read'},
        {name: 'truthEdge', view: truthEdgeView, type: 'u32', access: 'read'},
        {name: 'reverse', view: reverseView, type: 'u32', access: 'read'},
        {
          name: 'classes',
          view: view('matched-classes', matchedClasses, 'uint32', pointCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: pointCount,
      declarations: `const NONE: u32 = ${NONE}u;`,
      body: `var accuracy = 0u;
  if (matched[matchedOffset + index] != NONE) {
    let edge = rowEdge[rowEdgeOffset + matched[matchedOffset + index]];
    let truth = truthEdge[truthEdgeOffset + index];
    if (edge == truth) {
      accuracy = 1u;
    } else if (truth != NONE && edge == reverse[reverseOffset + truth]) {
      accuracy = 2u;
    }
  }
  classes[classesOffset + index] = accuracy;`
    });
    addKernelPass(graph, {
      id: 'matched-rows',
      bindings: [
        {name: 'points', view: snappedView, type: 'f32', access: 'read'},
        {name: 'matched', view: matchedEdgesView, type: 'u32', access: 'read'},
        {name: 'breaks', view: breaksView, type: 'u32', access: 'read'},
        {name: 'hasNext', view: hasNextView, type: 'u32', access: 'read'},
        {name: 'trackOfPoint', view: trackOfPointView, type: 'u32', access: 'read'},
        {name: 'focus', view: focusView, type: 'u32', access: 'read'},
        {
          name: 'segments',
          view: view('matched-segments', matchedSegments, 'float32', pointCount * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'weights',
          view: view('matched-weights', matchedWeights, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: pointCount,
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;
const NONE: u32 = ${NONE}u;`,
      body: `let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  let inFocus = focus[focusOffset] == 0u || trackOfPoint[trackOfPointOffset + index] + 1u == focus[focusOffset];
  let isDrawn = hasNext[hasNextOffset + index] != 0u && inFocus && matched[matchedOffset + index] != NONE &&
    matched[matchedOffset + next] != NONE && breaks[breaksOffset + next] == 0u;
  weights[weightsOffset + index] = select(0.0, 1.0, isDrawn);`
    });
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      'match',
      [
        {buffer: matchedCount, size: 4},
        {buffer: breakCount, size: 4},
        {buffer: overflow, size: 4},
        {buffer: matchedEdges, size: pointCount * 4},
        {buffer: snapDistances, size: pointCount * 4}
      ],
      bytes => {
        if (destroyed || build?.reader !== reader) return;
        handleSummary(bytes);
      }
    );
    build = {compiled, reader};
    dirty = true;
  }

  function handleSummary(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes, 0, 3 + pointCount);
    const [matched, breakTotal, overflowFlag] = words;
    const rows = words.subarray(3);
    const distances = new Float32Array(bytes, (3 + pointCount) * 4, pointCount);
    let exact = 0;
    let reversed = 0;
    let distanceSum = 0;
    let distanceCount = 0;
    for (let point = 0; point < pointCount; point++) {
      const row = rows[point];
      if (row === NONE) continue;
      const edge = dense.rowEdge[row];
      const truth = truthEdges[point];
      if (edge === truth) exact++;
      else if (truth !== NONE && edge === roadGraph.edgeReverse[truth]) reversed++;
      if (distances[point] >= 0) {
        distanceSum += distances[point];
        distanceCount++;
      }
    }
    ctx.setReadout(
      'matched',
      `${formatCount(matched)} of ${formatCount(pointCount)} (${((100 * matched) / pointCount).toFixed(1)}%)`
    );
    ctx.setReadout('breaks', breakTotal);
    ctx.setReadout('overflow', overflowFlag ? 'edge grid overflowed' : 'none');
    ctx.setReadout('exact', `${((100 * exact) / pointCount).toFixed(1)}%`);
    ctx.setReadout('sameStreet', `${((100 * (exact + reversed)) / pointCount).toFixed(1)}%`);
    ctx.setReadout('snap', distanceCount ? `${(distanceSum / distanceCount).toFixed(1)} m` : null);
  }

  async function measure(): Promise<void> {
    if (measuring || !build) return;
    measuring = true;
    try {
      const timing = await measureCompiledGraph(device, build.compiled, {
        parameters: undefined,
        completionBuffer: matchedCount,
        runs: 5,
        repetitions: 2
      });
      if (!destroyed) {
        ctx.setReadout(
          'time',
          `${timing.milliseconds.toFixed(1)} ms (${timing.method === 'gpu-timestamps' ? 'GPU' : 'wall clock'})`
        );
      }
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  let totalNoise = 0;
  {
    // RMS error of the dataset's own noise, for the readout.
    let sum = 0;
    for (let point = 0; point < pointCount; point++) {
      const dx = gpsPositions[point * 2] - truthPositions[point * 2];
      const dy = gpsPositions[point * 2 + 1] - truthPositions[point * 2 + 1];
      sum += dx * dx + dy * dy;
    }
    totalNoise = Math.sqrt(sum / pointCount);
  }
  ctx.setReadout('fixes', `${formatCount(trackCount)} tracks, ${formatCount(pointCount)} fixes`);
  ctx.setReadout(
    'network',
    `${formatCount(dense.nodeCount)} nodes, ${formatCount(dense.rowCount)} directed edges`
  );
  ctx.setReadout('noise', `${totalNoise.toFixed(1)} m RMS (data)`);

  writeParameters();
  buildMatchGraph();

  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];

  return {
    getCompiledGraphs: () =>
      (build
        ? [build.compiled, compiledMerge]
        : [compiledMerge]) as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'candidateCount':
        case 'routeNodeBudget':
        case 'cellSize':
          // The search radius is capped at twice the cell size.
          writeParameters();
          buildMatchGraph();
          break;
        case 'extraNoise':
        case 'noiseSeed':
        case 'sigma':
        case 'beta':
        case 'searchRadius':
        case 'routeFactor':
        case 'routeSlack':
        case 'trackFocus':
          writeParameters();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const segment = segmentIndex.nearest(x, y, 30);
      if (segment < 0) return null;
      const edge = roadGraph.segmentEdge[segment];
      return `${ROAD_CLASS_NAMES[roadGraph.edgeClass[edge]]}, ${Math.round(roadGraph.edgeLength[edge])} m, ${roadGraph.edgeSpeed[edge]} km/h`;
    },

    encode(commandEncoder, frame) {
      if (!chainsEncoded) {
        compiledMerge.encode(commandEncoder, {parameters: undefined});
        chainsEncoded = true;
        chainReader.request(commandEncoder);
      }
      chainReader.flush(commandEncoder);
      if (!build) return;
      if (dirty && frame.timeSeconds - lastMatchTime >= REMATCH_INTERVAL_SECONDS) {
        lastMatchTime = frame.timeSeconds;
        build.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        build.reader.markStale();
      }
      build.reader.flush(commandEncoder);
    },

    getLayers() {
      const {roadStyle, showRaw, showTruth, showMatched, matchedColor} = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (roadStyle === 'plain') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'mm-roads',
            coordinateOrigin,
            segments: roadSegmentsBuffer,
            instanceCount: segmentCount,
            color: dark ? [140, 152, 185, 150] : [96, 108, 135, 140],
            widthPixels: 1.2
          })
        );
      } else if (roadStyle === 'chains') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'mm-chains',
            coordinateOrigin,
            segments: chainSegments,
            instanceCount: segmentCount * 2,
            weights: chainWeights,
            values: chainValues,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: dark ? CHAIN_COLORS_DARK : CHAIN_COLORS_LIGHT,
            widthPixels: 2
          })
        );
      }
      if (showTruth) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'mm-truth',
            coordinateOrigin,
            segments: truthSegments,
            instanceCount: pointCount,
            weights: truthWeights,
            color: dark ? [200, 205, 215, 190] : [60, 65, 80, 190],
            widthPixels: 4
          })
        );
      }
      if (showRaw) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'mm-raw',
            coordinateOrigin,
            segments: rawSegments,
            instanceCount: pointCount,
            weights: rawWeights,
            color: [255, 140, 60, 170],
            widthPixels: 1.4
          })
        );
      }
      if (showMatched) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'mm-matched',
            coordinateOrigin,
            segments: matchedSegments,
            instanceCount: pointCount,
            weights: matchedWeights,
            ...(matchedColor === 'accuracy'
              ? {
                  values: matchedClasses,
                  valueFormat: 'uint32' as const,
                  colormap: 'category' as const,
                  palette: ACCURACY_COLORS
                }
              : {color: [40, 205, 255, 240] as const}),
            widthPixels: 2.4
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      build?.reader.stop();
      chainReader.stop();
      resources.destroy();
    }
  };
}

const CHAIN_COLORS_LIGHT = [
  [31, 119, 180, 255],
  [214, 95, 18, 255],
  [44, 160, 44, 255],
  [148, 90, 170, 255],
  [190, 160, 20, 255],
  [200, 55, 85, 255],
  [30, 150, 160, 255],
  [110, 110, 120, 255]
] as const;
const CHAIN_COLORS_DARK = [
  [96, 165, 220, 255],
  [255, 150, 70, 255],
  [110, 205, 110, 255],
  [190, 140, 220, 255],
  [235, 205, 70, 255],
  [245, 105, 135, 255],
  [80, 205, 215, 255],
  [170, 170, 180, 255]
] as const;
