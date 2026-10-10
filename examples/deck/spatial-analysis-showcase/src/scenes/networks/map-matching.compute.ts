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
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {buildDenseDirectedGraph, buildRoadGraph, SegmentIndex} from './b10-road-graph';

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
  fixFocus: number;
  evidenceMode: 'raw' | 'nearest' | 'candidates' | 'tradeoff' | 'stress';
};

const REMATCH_INTERVAL_SECONDS = 0.15;
const NONE = 0xffffffff;
const NOISE_LADDER = [0, 10, 20, 30, 40, 50, 60] as const;

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

type SweepTag = {generation: number; noise: number};

/**
 * HMM map matching (`GPUMapMatching`) of 200 simulated Chicago GPS traces on a polyline-dense
 * version of the street graph, scored against the ground-truth edge of every fix. Emission sigma,
 * transition beta, search radius, route allowances, the extra jitter and the track focus are
 * buffer writes; candidate count, route node budget and edge-grid cell size are compile-time and
 * rebuild the graph. The nearest-edge baseline below is deliberately independent of the HMM.
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
  const maximumBaselineDistance = Math.hypot(
    roadGraph.bounds[2] - roadGraph.bounds[0],
    roadGraph.bounds[3] - roadGraph.bounds[1]
  );
  const gpsPositions = traces.projectColumn('vertices', origin);
  const truthPositions = traces.projectColumn('truthPosition', origin);
  const truthEdges = traces.column<Uint32Array>('truthEdge');
  const trackOffsets = traces.column<Uint32Array>('pathOffsets');
  const timestamps = traces.column<Uint32Array>('timestamp');
  const sampleIntervals = traces.column<Uint8Array>('sampleInterval');
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
  const wrongStreetWeights = resources.createBuffer('wrong-street-weights', pointCount * 4);
  const correctWeights = resources.createBuffer('correct-weights', pointCount * 4);
  const reverseWeights = resources.createBuffer('reverse-weights', pointCount * 4);
  const segmentCount = roadGraph.segmentCount;
  const EVIDENCE_CAPACITY = 256;
  const CANDIDATE_CAPACITY = 8;
  const selectedFixPositions = resources.createBuffer(
    'selected-fix-positions',
    EVIDENCE_CAPACITY * 8
  );
  const candidateSegments = resources.createBuffer('candidate-segments', CANDIDATE_CAPACITY * 16);
  const snapLeaders = resources.createBuffer('snap-leaders', CANDIDATE_CAPACITY * 16);
  const breakPositions = resources.createBuffer('break-positions', EVIDENCE_CAPACITY * 8);
  const sigmaPosition = resources.createBuffer('sigma-position', 8);
  const searchPosition = resources.createBuffer('search-position', 8);
  const nearestSegments = resources.createBuffer('nearest-segments', EVIDENCE_CAPACITY * 16);
  const nearestWeights = resources.createBuffer('nearest-weights', EVIDENCE_CAPACITY * 4);
  const selectedRawSegments = resources.createBuffer(
    'selected-raw-segments',
    EVIDENCE_CAPACITY * 16
  );
  const selectedRawWeights = resources.createBuffer('selected-raw-weights', EVIDENCE_CAPACITY * 4);
  const baselineEdges = new Uint32Array(pointCount);
  const baselineDistances = new Float32Array(pointCount);
  const noisyCpuPositions = new Float32Array(pointCount * 2);
  let latestMatchedRows = new Uint32Array(pointCount).fill(NONE);
  let latestBreaks = new Uint32Array(pointCount);
  let latestSnapDistances = new Float32Array(pointCount);
  let selectedEvidenceCount = 0;
  let candidateEvidenceCount = 0;
  let breakEvidenceCount = 0;
  let activeSweepNoise: number | null = null;
  let noiseSweepIndex = -1;
  const hmmNoiseLadder = new Map<number, number>();
  let sweepGeneration = 0;
  let parameterSweepTag: SweepTag | null = null;
  let uncopiedSweepTag: SweepTag | null = null;
  let pendingSweepTag: SweepTag | null = null;

  let destroyed = false;
  let dirty = true;
  let lastMatchTime = -Infinity;
  let measuring = false;
  let build: MatchBuild | null = null;

  const maxSearchRadius = () => 2 * Number(ctx.options.cellSize);
  function writeParameters(
    extraNoise = ctx.options.extraNoise,
    sweepTag: SweepTag | null = null
  ): void {
    const {noiseSeed, sigma, beta, searchRadius, routeFactor, routeSlack, trackFocus} = ctx.options;
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
    parameterSweepTag = sweepTag;
    dirty = true;
    updateNearestBaseline();
    updateEvidence();
  }

  /** Runs the actual compiled HMM once per fixed noise level; values arrive through its readback. */
  function startHmmNoiseSweep(): void {
    sweepGeneration++;
    activeSweepNoise = null;
    noiseSweepIndex = -1;
    parameterSweepTag = null;
    uncopiedSweepTag = null;
    pendingSweepTag = null;
    if (ctx.options.evidenceMode !== 'stress') return;
    hmmNoiseLadder.clear();
    noiseSweepIndex = 0;
    activeSweepNoise = NOISE_LADDER[noiseSweepIndex];
    writeParameters(activeSweepNoise, {generation: sweepGeneration, noise: activeSweepNoise});
  }

  function continueHmmNoiseSweep(completed: SweepTag): void {
    if (
      activeSweepNoise === null ||
      completed.generation !== sweepGeneration ||
      completed.noise !== activeSweepNoise
    )
      return;
    noiseSweepIndex++;
    if (noiseSweepIndex < NOISE_LADDER.length) {
      activeSweepNoise = NOISE_LADDER[noiseSweepIndex];
      writeParameters(activeSweepNoise, {generation: sweepGeneration, noise: activeSweepNoise});
    } else {
      activeSweepNoise = null;
      noiseSweepIndex = -1;
      writeParameters();
    }
  }

  function hashValue(value: number): number {
    const state = (Math.imul(value, 747796405) + 2891336653) >>> 0;
    const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
  }

  function uniformValue(value: number): number {
    return ((hashValue(value) >>> 8) + 0.5) / 16777216;
  }

  function getNoisyPosition(point: number): readonly [number, number] {
    const seed = Math.imul(ctx.options.noiseSeed, 2654435761) >>> 0;
    const radius = Math.sqrt(-2 * Math.log(uniformValue(Math.imul(point, 2) + seed)));
    const angle = 2 * Math.PI * uniformValue(Math.imul(point, 2) + 1 + seed);
    return [
      gpsPositions[point * 2] + ctx.options.extraNoise * radius * Math.cos(angle),
      gpsPositions[point * 2 + 1] + ctx.options.extraNoise * radius * Math.sin(angle)
    ];
  }

  function nearestPointOnSegment(
    segment: number,
    x: number,
    y: number
  ): readonly [number, number, number] {
    const offset = segment * 4;
    const x0 = roadGraph.segments[offset];
    const y0 = roadGraph.segments[offset + 1];
    const dx = roadGraph.segments[offset + 2] - x0;
    const dy = roadGraph.segments[offset + 3] - y0;
    const lengthSquared = dx * dx + dy * dy;
    const fraction = lengthSquared
      ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / lengthSquared))
      : 0;
    const px = x0 + dx * fraction;
    const py = y0 + dy * fraction;
    return [px, py, Math.hypot(x - px, y - py)];
  }

  /** Honest geometric nearest-edge baseline, evaluated independently of GPUMapMatching. */
  function updateNearestBaseline(): void {
    let correct = 0;
    let reverse = 0;
    let mismatch = 0;
    let selectedMismatch = 0;
    let firstMismatch = -1;
    const selectedTrace = Math.max(0, Math.min(trackCount - 1, ctx.options.trackFocus - 1));
    const selectedStart = trackOffsets[selectedTrace];
    const selectedEnd = trackOffsets[selectedTrace + 1];
    for (let point = 0; point < pointCount; point++) {
      const [x, y] = getNoisyPosition(point);
      noisyCpuPositions[point * 2] = x;
      noisyCpuPositions[point * 2 + 1] = y;
      const segment = segmentIndex.nearest(x, y, maximumBaselineDistance);
      const edge = segment >= 0 ? roadGraph.segmentEdge[segment] : NONE;
      baselineEdges[point] = edge;
      baselineDistances[point] =
        segment >= 0 ? nearestPointOnSegment(segment, x, y)[2] : Number.NaN;
      const truth = truthEdges[point];
      if (edge === truth) correct++;
      else if (truth !== NONE && edge === roadGraph.edgeReverse[truth]) reverse++;
      if (latestMatchedRows[point] !== NONE && dense.rowEdge[latestMatchedRows[point]] !== edge) {
        mismatch++;
        if (firstMismatch < 0) firstMismatch = point;
        if (point >= selectedStart && point < selectedEnd) selectedMismatch++;
      }
    }
    ctx.setReadout(
      'baseline',
      `${((100 * (correct + reverse)) / pointCount).toFixed(1)}% (either direction)`
    );
    const firstMismatchLabel =
      firstMismatch < 0
        ? 'none'
        : `trace ${trackOfPoint[firstMismatch] + 1}, fix ${firstMismatch - trackOffsets[trackOfPoint[firstMismatch]] + 1}`;
    ctx.setReadout(
      'mismatch',
      `${formatCount(mismatch)} disagreements; selected trace: ${selectedMismatch}; first: ${firstMismatchLabel}`
    );
  }

  /** Selected trace evidence; candidates are CPU display geometry, not GPU output. */
  function updateEvidence(): void {
    const track = Math.max(0, Math.min(trackCount - 1, ctx.options.trackFocus - 1));
    const start = trackOffsets[track];
    const end = Math.min(trackOffsets[track + 1], start + EVIDENCE_CAPACITY);
    const count = end - start;
    const tracePositions = new Float32Array(EVIDENCE_CAPACITY * 2);
    const traceSegments = new Float32Array(EVIDENCE_CAPACITY * 4);
    const traceWeights = new Float32Array(EVIDENCE_CAPACITY);
    const baselineLines = new Float32Array(EVIDENCE_CAPACITY * 4);
    const baselineLineWeights = new Float32Array(EVIDENCE_CAPACITY);
    const breaksForTrace = new Float32Array(EVIDENCE_CAPACITY * 2);
    let breakCountForTrace = 0;
    for (let local = 0; local < count; local++) {
      const point = start + local;
      const x = noisyCpuPositions[point * 2];
      const y = noisyCpuPositions[point * 2 + 1];
      tracePositions[local * 2] = x;
      tracePositions[local * 2 + 1] = y;
      if (local + 1 < count) {
        traceSegments.set(
          [x, y, noisyCpuPositions[(point + 1) * 2], noisyCpuPositions[(point + 1) * 2 + 1]],
          local * 4
        );
        traceWeights[local] = 1;
      }
      const segment = segmentIndex.nearest(x, y, maximumBaselineDistance);
      if (segment >= 0) {
        const [px, py] = nearestPointOnSegment(segment, x, y);
        baselineLines.set([x, y, px, py], local * 4);
        baselineLineWeights[local] = 1;
      }
      if (latestBreaks[point]) {
        breaksForTrace[breakCountForTrace * 2] = x;
        breaksForTrace[breakCountForTrace * 2 + 1] = y;
        breakCountForTrace++;
      }
    }
    selectedFixPositions.write(tracePositions);
    selectedEvidenceCount = count;
    selectedRawSegments.write(traceSegments);
    selectedRawWeights.write(traceWeights);
    nearestSegments.write(baselineLines);
    nearestWeights.write(baselineLineWeights);
    breakPositions.write(breaksForTrace);
    breakEvidenceCount = breakCountForTrace;
    const focused = Math.min(end - 1, start + Math.max(0, ctx.options.fixFocus - 1));
    const fx = noisyCpuPositions[focused * 2];
    const fy = noisyCpuPositions[focused * 2 + 1];
    sigmaPosition.write(Float32Array.of(fx, fy));
    searchPosition.write(Float32Array.of(fx, fy));
    const nearby: {segment: number; distance: number}[] = [];
    const limit = Math.min(ctx.options.searchRadius, maxSearchRadius());
    for (let segment = 0; segment < segmentCount; segment++) {
      const distance = nearestPointOnSegment(segment, fx, fy)[2];
      if (distance <= limit) nearby.push({segment, distance});
    }
    nearby.sort((a, b) => a.distance - b.distance);
    const candidateRows = new Float32Array(CANDIDATE_CAPACITY * 4);
    const leaderRows = new Float32Array(CANDIDATE_CAPACITY * 4);
    const countCandidates = Math.min(CANDIDATE_CAPACITY, nearby.length);
    for (let index = 0; index < countCandidates; index++) {
      const segment = nearby[index].segment;
      candidateRows.set(roadGraph.segments.subarray(segment * 4, segment * 4 + 4), index * 4);
      const [px, py] = nearestPointOnSegment(segment, fx, fy);
      leaderRows.set([fx, fy, px, py], index * 4);
    }
    candidateSegments.write(candidateRows);
    candidateEvidenceCount = countCandidates;
    snapLeaders.write(leaderRows);
    ctx.requestLayers();
  }

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
    const matchedSegmentsView = view(
      'matched-segments',
      matchedSegments,
      'float32',
      pointCount * 4
    );
    const matchedWeightsView = view('matched-weights', matchedWeights, 'float32', pointCount);
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
          view: matchedSegmentsView,
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'weights',
          view: matchedWeightsView,
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
    addKernelPass(graph, {
      id: 'matched-outcomes',
      bindings: [
        {name: 'matched', view: matchedEdgesView, type: 'u32', access: 'read'},
        {name: 'weights', view: matchedWeightsView, type: 'f32', access: 'read'},
        {name: 'rowEdge', view: rowEdgeView, type: 'u32', access: 'read'},
        {name: 'truthEdge', view: truthEdgeView, type: 'u32', access: 'read'},
        {name: 'reverse', view: reverseView, type: 'u32', access: 'read'},
        {
          name: 'wrongWeights',
          view: view('wrong-street-weights', wrongStreetWeights, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'correctWeights',
          view: view('correct-weights', correctWeights, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'reverseWeights',
          view: view('reverse-weights', reverseWeights, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: pointCount,
      declarations: `const NONE: u32 = ${NONE}u;`,
      body: `let isDrawn = weights[weightsOffset + index] != 0.0;
  var outcome = 0u;
  if (isDrawn) {
    let edge = rowEdge[rowEdgeOffset + matched[matchedOffset + index]];
    let truth = truthEdge[truthEdgeOffset + index];
    if (edge == truth) {
      outcome = 1u;
    } else if (truth != NONE && edge == reverse[reverseOffset + truth]) {
      outcome = 2u;
    }
  }
  wrongWeights[wrongWeightsOffset + index] = select(0.0, 1.0, isDrawn && outcome == 0u);
  correctWeights[correctWeightsOffset + index] = select(0.0, 1.0, outcome == 1u);
  reverseWeights[reverseWeightsOffset + index] = select(0.0, 1.0, outcome == 2u);`
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
        {buffer: snapDistances, size: pointCount * 4},
        {buffer: breaks, size: pointCount * 4}
      ],
      bytes => {
        if (destroyed || build?.reader !== reader) return;
        const sweepTag = pendingSweepTag;
        pendingSweepTag = null;
        handleSummary(bytes, sweepTag);
      }
    );
    build = {compiled, reader};
    dirty = true;
  }

  function handleSummary(bytes: ArrayBuffer, sweepTag: SweepTag | null): void {
    const words = new Uint32Array(bytes, 0, 3 + pointCount);
    const [matched, breakTotal, overflowFlag] = words;
    const rows = words.subarray(3);
    const distances = new Float32Array(bytes, (3 + pointCount) * 4, pointCount);
    latestMatchedRows = rows.slice();
    latestBreaks = new Uint32Array(bytes, (3 + pointCount * 2) * 4, pointCount).slice();
    latestSnapDistances = distances.slice();
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
    const hmmAccuracy = (100 * (exact + reversed)) / pointCount;
    ctx.setReadout('hmm', `${hmmAccuracy.toFixed(1)}% (either direction)`);
    const isCurrentSweepResult = Boolean(
      sweepTag && sweepTag.generation === sweepGeneration && sweepTag.noise === activeSweepNoise
    );
    if (isCurrentSweepResult) hmmNoiseLadder.set(sweepTag!.noise, hmmAccuracy);
    updateNearestBaseline();
    updateEvidence();
    const bins = new Array<number>(8).fill(0);
    for (const distance of distances) {
      if (Number.isFinite(distance) && distance >= 0)
        bins[Math.min(7, Math.floor(distance / 10))]++;
    }
    ctx.setChart('snapHistogram', {
      kind: 'histogram',
      values: bins,
      xDomain: [0, 80],
      markers: [{x: ctx.options.sigma, label: 'sigma'}],
      xLabel: 'snap distance (m)',
      yLabel: 'fixes',
      description:
        'HMM snap-distance histogram from the loaded traces; marker is the current emission sigma.'
    });
    ctx.setChart('noiseComparison', {
      kind: 'line',
      xLabel: 'added isotropic noise (m)',
      yLabel: 'accuracy (%)',
      series: [
        {
          label: 'Nearest baseline',
          x: NOISE_LADDER,
          y: makeBaselineNoiseLadder(NOISE_LADDER),
          color: 1,
          dashed: true,
          points: true
        },
        {
          label: 'HMM',
          x: NOISE_LADDER.slice(0, hmmNoiseLadder.size),
          y: NOISE_LADDER.slice(0, hmmNoiseLadder.size).map(
            level => hmmNoiseLadder.get(level) ?? 0
          ),
          color: 0,
          points: true
        }
      ],
      description:
        'Nearest-edge and HMM accuracy are measured from the loaded traces over the same fixed deterministic added-noise ladder.'
    });
    if (isCurrentSweepResult) continueHmmNoiseSweep(sweepTag!);
  }

  function makeBaselineNoiseLadder(levels: readonly number[]): number[] {
    return levels.map(extraNoise => {
      let correct = 0;
      for (let point = 0; point < pointCount; point++) {
        const seed = Math.imul(ctx.options.noiseSeed, 2654435761) >>> 0;
        const radius = Math.sqrt(-2 * Math.log(uniformValue(Math.imul(point, 2) + seed)));
        const angle = 2 * Math.PI * uniformValue(Math.imul(point, 2) + 1 + seed);
        const x = gpsPositions[point * 2] + extraNoise * radius * Math.cos(angle);
        const y = gpsPositions[point * 2 + 1] + extraNoise * radius * Math.sin(angle);
        const segment = segmentIndex.nearest(x, y, maximumBaselineDistance);
        const edge = segment >= 0 ? roadGraph.segmentEdge[segment] : NONE;
        const truth = truthEdges[point];
        if (edge === truth || (truth !== NONE && edge === roadGraph.edgeReverse[truth])) correct++;
      }
      return (100 * correct) / pointCount;
    });
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
  const intervalValues = Array.from(sampleIntervals);
  intervalValues.sort((a, b) => a - b);
  const medianInterval = intervalValues[Math.floor(intervalValues.length / 2)] ?? 0;
  function updateFurniture(): void {
    const selectedTrace = Math.max(0, Math.min(trackCount - 1, ctx.options.trackFocus - 1));
    const traceSigma = traces.column<Float32Array>('noiseSigma')[selectedTrace];
    ctx.setReadout(
      'interval',
      `${medianInterval} s nominal (selected sigma ${traceSigma.toFixed(0)} m)`
    );
    ctx.setFurniture({
      title: {
        title: 'Simulated GPS',
        subtitle: `${traceSigma.toFixed(0)} m sigma + ${ctx.options.extraNoise.toFixed(0)} m added · ${medianInterval} s nominal samples`
      },
      scaleBar: {units: 'metric'},
      credit: 'Synthetic traces derived from OSM, ODbL; not observed vehicles.'
    });
  }
  updateFurniture();
  void timestamps;

  writeParameters();
  buildMatchGraph();

  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];

  return {
    getCompiledGraphs: () => (build ? [build.compiled] : []) as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'candidateCount':
        case 'routeNodeBudget':
        case 'cellSize':
          // The search radius is capped at twice the cell size.
          writeParameters();
          buildMatchGraph();
          startHmmNoiseSweep();
          break;
        case 'extraNoise':
        case 'noiseSeed':
        case 'sigma':
        case 'beta':
        case 'searchRadius':
        case 'routeFactor':
        case 'routeSlack':
        case 'trackFocus':
        case 'fixFocus':
          writeParameters();
          updateFurniture();
          startHmmNoiseSweep();
          break;
        case 'evidenceMode':
          writeParameters();
          startHmmNoiseSweep();
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
      let nearestFix = -1;
      let nearestDistance = 35 * 35;
      for (let point = 0; point < pointCount; point++) {
        const dx = noisyCpuPositions[point * 2] - x;
        const dy = noisyCpuPositions[point * 2 + 1] - y;
        const distance = dx * dx + dy * dy;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestFix = point;
        }
      }
      if (nearestFix < 0) return null;
      let trace = 0;
      while (trace + 1 < trackOffsets.length && trackOffsets[trace + 1] <= nearestFix) trace++;
      const fix = nearestFix - trackOffsets[trace] + 1;
      const rawTruthDistance = Math.hypot(
        noisyCpuPositions[nearestFix * 2] - truthPositions[nearestFix * 2],
        noisyCpuPositions[nearestFix * 2 + 1] - truthPositions[nearestFix * 2 + 1]
      );
      const row = latestMatchedRows[nearestFix];
      const matchedEdge = row === NONE ? NONE : dense.rowEdge[row];
      const truth = truthEdges[nearestFix];
      const outcome =
        matchedEdge === NONE
          ? 'unmatched'
          : matchedEdge === truth
            ? 'truth edge'
            : truth !== NONE && matchedEdge === roadGraph.edgeReverse[truth]
              ? 'reverse direction'
              : 'wrong street';
      return {
        title: `Trace ${trace + 1}, fix ${fix}`,
        subtitle: `t + ${timestamps[nearestFix]} s`,
        rows: [
          {label: 'Raw to truth', value: rawTruthDistance.toFixed(1), unit: 'm'},
          {label: 'Nearest snap', value: baselineDistances[nearestFix].toFixed(1), unit: 'm'},
          {
            label: 'HMM snap',
            value: row === NONE ? '—' : latestSnapDistances[nearestFix].toFixed(1),
            unit: row === NONE ? undefined : 'm'
          },
          {label: 'Outcome', value: outcome, emphasis: true}
        ]
      };
    },

    encode(commandEncoder, frame) {
      if (!build) return;
      if (dirty && frame.timeSeconds - lastMatchTime >= REMATCH_INTERVAL_SECONDS) {
        lastMatchTime = frame.timeSeconds;
        build.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        build.reader.markStale();
        uncopiedSweepTag = parameterSweepTag;
        parameterSweepTag = null;
      }
      const wasPending = build.reader.isPending;
      build.reader.flush(commandEncoder);
      if (!wasPending && build.reader.isPending) {
        pendingSweepTag = uncopiedSweepTag;
        uncopiedSweepTag = null;
      }
    },

    getLayers() {
      const dark = ctx.ground() === 'dark';
      const showRawEvidence =
        ctx.options.evidenceMode === 'raw' || ctx.options.evidenceMode === 'nearest';
      const showNearest =
        ctx.options.evidenceMode === 'nearest' || ctx.options.evidenceMode === 'tradeoff';
      const showCandidates = ctx.options.evidenceMode === 'candidates';
      const showBreaks =
        ctx.options.evidenceMode === 'tradeoff' || ctx.options.evidenceMode === 'stress';
      const layers: Layer[] = [];
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'mm-roads',
          coordinateOrigin,
          segments: roadSegmentsBuffer,
          instanceCount: segmentCount,
          color: dark ? [174, 164, 150, 115] : [130, 120, 105, 120],
          widthPixels: 1
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-truth',
          coordinateOrigin,
          segments: truthSegments,
          instanceCount: pointCount,
          weights: truthWeights,
          color: dark ? [200, 205, 215, 85] : [60, 65, 80, 80],
          widthPixels: 7,
          outlineColor: dark ? [20, 23, 28, 160] : [250, 247, 239, 190],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-raw-connector',
          coordinateOrigin,
          segments: selectedRawSegments,
          instanceCount: showRawEvidence ? EVIDENCE_CAPACITY : 0,
          weights: selectedRawWeights,
          color: [135, 83, 180, 190],
          widthPixels: 0.8,
          dashArray: [3, 3]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-nearest-baseline',
          coordinateOrigin,
          segments: nearestSegments,
          instanceCount: showNearest ? EVIDENCE_CAPACITY : 0,
          weights: nearestWeights,
          color: dark ? [200, 185, 160, 165] : [105, 92, 75, 185],
          widthPixels: 2,
          dashArray: [4, 3]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-hmm-wrong-street',
          coordinateOrigin,
          segments: matchedSegments,
          instanceCount: pointCount,
          weights: wrongStreetWeights,
          color: [205, 76, 47, 255],
          widthPixels: 4.5
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-hmm-correct',
          coordinateOrigin,
          segments: matchedSegments,
          instanceCount: pointCount,
          weights: correctWeights,
          color: [40, 125, 210, 255],
          widthPixels: 3.5
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-hmm-reverse',
          coordinateOrigin,
          segments: matchedSegments,
          instanceCount: pointCount,
          weights: reverseWeights,
          color: [122, 185, 232, 255],
          widthPixels: 3.5,
          dashArray: [5, 3]
        }),
        new SpatialAnalysisPointLayer({
          id: 'mm-raw-halos',
          coordinateOrigin,
          positions: selectedFixPositions,
          instanceCount: showRawEvidence ? selectedEvidenceCount : 0,
          radiusPixels: 3.5,
          color: [135, 83, 180, 255],
          outlineColor: dark ? [28, 22, 34, 220] : [255, 248, 255, 230],
          outlineWidthPixels: 2
        }),
        new SpatialAnalysisPointLayer({
          id: 'mm-sigma-ring',
          coordinateOrigin,
          positions: sigmaPosition,
          instanceCount: showRawEvidence ? 1 : 0,
          radiusMeters: ctx.options.sigma,
          radiusMinPixels: 3,
          shape: 'ring',
          color: [135, 83, 180, 255],
          fillOpacity: 0
        }),
        new SpatialAnalysisPointLayer({
          id: 'mm-search-ring',
          coordinateOrigin,
          positions: searchPosition,
          instanceCount: showCandidates ? 1 : 0,
          radiusMeters: Math.min(ctx.options.searchRadius, maxSearchRadius()),
          shape: 'ring',
          color: [212, 163, 45, 255],
          fillOpacity: 0
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-candidate-segments',
          coordinateOrigin,
          segments: candidateSegments,
          instanceCount: showCandidates ? candidateEvidenceCount : 0,
          color: [212, 163, 45, 245],
          widthPixels: 3
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'mm-snap-leaders',
          coordinateOrigin,
          segments: snapLeaders,
          instanceCount: showCandidates ? candidateEvidenceCount : 0,
          color: [212, 163, 45, 180],
          widthPixels: 1,
          dashArray: [2, 2]
        }),
        new SpatialAnalysisPointLayer({
          id: 'mm-break-symbols',
          coordinateOrigin,
          positions: breakPositions,
          instanceCount: showBreaks ? breakEvidenceCount : 0,
          radiusPixels: 6,
          shape: 'cross',
          color: [205, 76, 47, 255],
          outlineColor: dark ? [22, 22, 22, 255] : [255, 255, 255, 255],
          outlineWidthPixels: 1
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      build?.reader.stop();
      resources.destroy();
    }
  };
}
