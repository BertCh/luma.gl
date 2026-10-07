// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  encodeGPUNetworkAccessibilityParameters,
  GPU_NETWORK_SNAPPING_NONE,
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  recommendLaneCount,
  type GPUNetworkAccessibilityParameters
} from '@luma.gl/experimental/gpu-network';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  buildAccessGraph,
  writeAccessWeights,
  writeRoadWalkCosts,
  type AccessGraph
} from './b9-access-graph';
import {buildRoadNetwork} from './b9-road-network';
import {formatInteger, getRoadColors, sliceSections} from './b9-shared';
import {buildTractDemand} from './b9-tracts';

/** Option state of the job-accessibility scene. */
export type JobAccessibilityOptions = {
  transit: boolean;
  walkSpeed: number;
  waitFactor: number;
  matrixLimitMinutes: number;
  measure: 'cumulative' | 'gravity-exponential' | 'gravity-power' | 'two-step';
  thresholdMinutes: number;
  beta: number;
  powerExponent: number;
  minimumCostSeconds: number;
  maxSnapDistance: number;
  opportunityRows: string;
  laneCount: string;
  snapSearch: 'exact' | 'bvh';
  seedDirection: 'both' | 'forward' | 'reverse';
  showTransit: boolean;
  showStops: boolean;
  showOpportunities: boolean;
  showSnaps: boolean;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
};

const MAXIMUM_ITERATIONS = 48;
const LOCAL_ITERATIONS = 16;
const SCORE_PERCENTILE = 0.98;
const MATRIX_DEBOUNCE_MILLISECONDS = 280;
const TIMING_DEBOUNCE_MILLISECONDS = 600;
const BVH_CANDIDATE_CAPACITY = 1 << 20;
const LINE_COLORS: Record<string, readonly [number, number, number, number]> = {
  red: [198, 12, 48, 255],
  blue: [0, 161, 222, 255],
  brown: [98, 54, 27, 255],
  green: [0, 155, 58, 255],
  orange: [249, 70, 28, 255],
  purple: [82, 35, 152, 255],
  pink: [226, 126, 166, 255],
  yellow: [249, 227, 0, 255]
};

type Built = {
  matrix: CompiledGPUCommandGraph<void>;
  score: CompiledGPUCommandGraph<void>;
  rowCount: number;
};

/**
 * Jobs reachable on foot and by CTA, per intersection. Two compiled graphs split the work by how
 * often their inputs change:
 *
 * - matrix: `GPUNetworkSnapping` snaps each opportunity (a job-rich census tract) onto the nearest
 *   walkable street edge, then `GPUNetworkCostMatrix` runs one bounded search per opportunity over
 *   the reversed walk-plus-transit graph. It re-runs only when the network, the opportunity set or
 *   the cost limit changes (debounced);
 * - score: `GPUNetworkAccessibility` scores the retained matrix (cumulative opportunities, gravity
 *   with exponential or power decay, 2SFCA). Threshold, decay and measure are a four-word
 *   parameter buffer, so moving them re-runs a few linear passes and no search.
 */
export async function createJobAccessibility(
  ctx: SceneContext<JobAccessibilityOptions>
): Promise<SceneInstance<JobAccessibilityOptions>> {
  const {device} = ctx;
  const network = buildRoadNetwork(ctx.datasets.get('chicago-roads'));
  const transit = ctx.datasets.get('cta-transit');
  const demand = buildTractDemand(ctx.datasets.get('chicago-tracts'), network.origin);
  const access: AccessGraph = buildAccessGraph(network, transit);
  const resources = new SpatialAnalysisResources(device, 'access');
  const {nodeCount, edgeCount, roadNodeCount, stopCount} = access;
  const segmentCount = network.segmentEdges.length;

  // Opportunities ordered by jobs; the compile-time row count keeps the largest.
  const order = Array.from({length: demand.count}, (_, index) => index).sort(
    (a, b) => demand.jobs[b] - demand.jobs[a]
  );
  let totalJobs = 0;
  for (const jobs of demand.jobs) totalJobs += jobs;
  const maximumRows = Math.min(
    384,
    order.length,
    Math.floor(device.limits.maxStorageBufferBindingSize / 4 / nodeCount)
  );

  // Demand per node for 2SFCA: workers live at the nearest road node of their tract.
  const nodeDemand = new Float32Array(nodeCount);
  for (let tract = 0; tract < demand.count; tract++) {
    const node = network.findNearestNode(
      demand.centroids[tract * 2],
      demand.centroids[tract * 2 + 1]
    );
    nodeDemand[node] += demand.residentWorkers[tract];
  }

  // ---- Static buffers ---------------------------------------------------------------------
  const offsetsBuffer = resources.createBuffer('offsets', access.offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', access.neighbors);
  const weightsBuffer = resources.createBuffer('weights', edgeCount * 4);
  const roadNodePositions = resources.createBuffer('road-node-positions', network.nodePositions);
  const roadSourcesBuffer = resources.createBuffer('road-sources', access.roadSources);
  const roadTargetsBuffer = resources.createBuffer('road-targets', access.roadTargets);
  const roadCostsBuffer = resources.createBuffer('road-costs', access.roadMeters.length * 4);
  const demandBuffer = resources.createBuffer('node-demand', nodeDemand);
  const segmentsBuffer = resources.createBuffer('segments', network.segments);
  const majorSegmentsBuffer = resources.createBuffer('major-segments', network.majorSegments);
  const segmentTargetsBuffer = resources.createBuffer(
    'segment-targets',
    network.segmentTargetNodes
  );
  const stopPositionsBuffer = resources.createBuffer('stop-positions', access.stopPositions);
  const stopModeBuffer = resources.createBuffer('stop-modes', Uint32Array.from(access.stopMode));
  const maxSnapParameter = resources.createParameterBuffer('max-snap', 'float32', 1);
  const costLimitParameter = resources.createParameterBuffer('cost-limit', 'float32', 1);
  const scoring = resources.createParameterBuffer('scoring', 'float32', 4);

  // Transit route shapes: rail lines in the CTA colors, buses faint.
  const routes = (transit.properties.routes ?? []) as {
    type: string;
    color: string;
    short: string;
    name: string;
  }[];
  const shapePathOffsets = transit.column<Uint32Array>('shapePathOffsets');
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeVertices = transit.projectColumn('shapeVertices', network.origin);
  const railSegments: number[] = [];
  const railLine: number[] = [];
  const busSegments: number[] = [];
  const lineNames = ['red', 'blue', 'brown', 'green', 'orange', 'purple', 'pink', 'yellow'];
  const lineOfRoute = (route: {short: string; name: string}) => {
    const text = `${route.short} ${route.name}`.toLowerCase();
    const index = lineNames.findIndex(name => text.includes(name));
    return index >= 0 ? index : 0;
  };
  for (let shape = 0; shape < shapeRoute.length; shape++) {
    const route = routes[shapeRoute[shape]];
    const target = route?.type === 'rail' ? railSegments : busSegments;
    for (let vertex = shapePathOffsets[shape]; vertex + 1 < shapePathOffsets[shape + 1]; vertex++) {
      target.push(
        shapeVertices[vertex * 2],
        shapeVertices[vertex * 2 + 1],
        shapeVertices[vertex * 2 + 2],
        shapeVertices[vertex * 2 + 3]
      );
      if (route?.type === 'rail') railLine.push(lineOfRoute(route));
    }
  }
  const railBuffer = resources.createBuffer('rail-segments', Float32Array.from(railSegments));
  const railLineBuffer = resources.createBuffer('rail-lines', Uint32Array.from(railLine));
  const busBuffer = resources.createBuffer('bus-segments', Float32Array.from(busSegments));

  // ---- Per-build resources ----------------------------------------------------------------
  let perBuild: {destroy: () => void}[] = [];
  let built: Built | null = null;
  let opportunityPositions!: Buffer;
  let opportunityWeights!: Buffer;
  let snapSegments!: Buffer;
  let snappedPositions!: Buffer;
  let snappedEdges!: Buffer;
  let snapDistances!: Buffer;
  let matrixBuffer!: Buffer;
  let matrixConverged!: Buffer;
  let snapOverflow!: Buffer;
  let ratiosBuffer!: Buffer;
  const cumulativeBuffer = resources.createBuffer('cumulative', nodeCount * 4);
  const gravityBuffer = resources.createBuffer('gravity', nodeCount * 4);
  const twoStepBuffer = resources.createBuffer('two-step', nodeCount * 4);
  let snapReader: SummaryReader | null = null;

  let destroyed = false;
  let matrixDirty = 2;
  let matrixPendingAt = 0;
  let scoreDirty = 2;
  let scoreMaximum = 1;
  let matrixEncodeCount = 0;
  let scoreEncodeCount = 0;
  let timingTimer: ReturnType<typeof setTimeout> | undefined;
  let timingPending = false;
  let matrixTiming: CompiledGraphTiming | null = null;
  let matrixTimingStale = true;
  let rowCount = 0;
  let builtKey = '';
  let rowsJobs = 0;

  const writeWeights = () => {
    const options = ctx.options;
    const weights = new Float32Array(edgeCount);
    writeAccessWeights(
      access,
      {walkSpeed: options.walkSpeed, transit: options.transit, waitFactor: options.waitFactor},
      weights
    );
    weightsBuffer.write(weights);
    const roadCosts = new Float32Array(access.roadMeters.length);
    writeRoadWalkCosts(access, options.walkSpeed, roadCosts);
    roadCostsBuffer.write(roadCosts);
  };

  const writeScalars = () => {
    const options = ctx.options;
    costLimitParameter.write(Float32Array.of(options.matrixLimitMinutes * 60));
    maxSnapParameter.write(Float32Array.of(options.maxSnapDistance));
  };

  const markMatrixDirty = () => {
    matrixPendingAt = performance.now() + MATRIX_DEBOUNCE_MILLISECONDS;
    matrixDirty = Math.max(matrixDirty, 1);
    scoreDirty = Math.max(scoreDirty, 1);
    matrixTimingStale = true;
    scheduleTiming();
  };

  const writeScoring = () => {
    const options = ctx.options;
    const parameters: GPUNetworkAccessibilityParameters = {
      threshold: Math.min(options.thresholdMinutes, options.matrixLimitMinutes) * 60,
      decay: 'none'
    };
    if (options.measure === 'gravity-exponential') {
      // beta is "decay per 10 minutes"; the contributor takes it per second.
      Object.assign(parameters, {decay: 'exponential', beta: options.beta / 600});
    } else if (options.measure === 'gravity-power') {
      Object.assign(parameters, {
        decay: 'power',
        beta: options.powerExponent,
        minimumCost: options.minimumCostSeconds
      });
    }
    scoring.write(encodeGPUNetworkAccessibilityParameters(parameters));
    scoreDirty = Math.max(scoreDirty, 1);
    scheduleTiming();
  };

  const getScoreBuffer = (): Buffer =>
    ctx.options.measure === 'cumulative'
      ? cumulativeBuffer
      : ctx.options.measure === 'two-step'
        ? twoStepBuffer
        : gravityBuffer;

  // ---- Graph construction -----------------------------------------------------------------
  function buildGraphs(): void {
    if (built) {
      snapReader?.stop();
      resources.release(built.matrix);
      resources.release(built.score);
      for (const resource of perBuild) resources.release(resource);
      perBuild = [];
    }
    const options = ctx.options;
    rowCount = Math.min(Number(options.opportunityRows), maximumRows);
    const laneCount = Math.min(
      Number(options.laneCount) || recommendLaneCount({rowCount, nodeCount, edgeCount}),
      rowCount
    );
    const matrixLength = rowCount * nodeCount;
    const track = <T extends {destroy: () => void}>(resource: T): T => {
      perBuild.push(resource);
      return resource;
    };
    const rows = order.slice(0, rowCount);
    const positions = new Float32Array(rowCount * 2);
    const weights = new Float32Array(rowCount);
    rowsJobs = 0;
    rows.forEach((tract, row) => {
      positions[row * 2] = demand.centroids[tract * 2];
      positions[row * 2 + 1] = demand.centroids[tract * 2 + 1];
      weights[row] = demand.jobs[tract];
      rowsJobs += demand.jobs[tract];
    });
    opportunityPositions = track(resources.createBuffer('opportunity-positions', positions));
    opportunityWeights = track(resources.createBuffer('opportunity-weights', weights));
    const segmentsInit = new Float32Array(rowCount * 4);
    for (let row = 0; row < rowCount; row++) {
      segmentsInit[row * 4] = positions[row * 2];
      segmentsInit[row * 4 + 1] = positions[row * 2 + 1];
      segmentsInit[row * 4 + 2] = positions[row * 2];
      segmentsInit[row * 4 + 3] = positions[row * 2 + 1];
    }
    snapSegments = track(resources.createBuffer('snap-segments', segmentsInit));
    snappedPositions = track(resources.createBuffer('snapped-positions', rowCount * 8));
    snappedEdges = track(resources.createBuffer('snapped-edges', rowCount * 4));
    snapDistances = track(resources.createBuffer('snap-distances', rowCount * 4));
    const snapFractions = track(resources.createBuffer('snap-fractions', rowCount * 4));
    const seedNodes = track(resources.createBuffer('seed-nodes', rowCount * 8));
    const seedCosts = track(resources.createBuffer('seed-costs', rowCount * 8));
    snapOverflow = track(resources.createBuffer('snap-overflow', 4));
    matrixBuffer = track(resources.createBuffer('matrix', matrixLength * 4));
    matrixConverged = track(resources.createBuffer('matrix-converged', 4));
    ratiosBuffer = track(resources.createBuffer('facility-ratios', rowCount * 4));

    const importer = (graph: GPUCommandGraph<void>) => {
      const cache = new Map<Buffer, GraphDataView>();
      return <Format extends 'uint32' | 'float32' | 'float32x2'>(
        buffer: Buffer,
        format: Format,
        length: number
      ) => {
        let view = cache.get(buffer);
        if (!view) {
          view = importGraphBuffer(graph, buffer.id, buffer, format, length) as GraphDataView;
          cache.set(buffer, view);
        }
        return view as unknown as GraphDataView<Format>;
      };
    };

    const matrixGraph = new GPUCommandGraph<void>(device, {id: 'access-matrix'});
    {
      const view = importer(matrixGraph);
      const seedNodesView = view(seedNodes, 'uint32', rowCount * 2);
      const seedCostsView = view(seedCosts, 'float32', rowCount * 2);
      matrixGraph.add(
        new GPUNetworkSnapping({
          id: 'snapping',
          points: view(opportunityPositions, 'float32x2', rowCount),
          nodePositions: view(roadNodePositions, 'float32x2', roadNodeCount),
          edgeSources: view(roadSourcesBuffer, 'uint32', access.roadSources.length),
          edgeTargets: view(roadTargetsBuffer, 'uint32', access.roadTargets.length),
          edgeCosts: view(roadCostsBuffer, 'float32', access.roadMeters.length),
          maxSnapDistance: maxSnapParameter.importToGraph(matrixGraph),
          ...(options.snapSearch === 'bvh'
            ? {candidateCapacity: BVH_CANDIDATE_CAPACITY, spatialSort: true}
            : {}),
          seedDirection: options.seedDirection,
          snappedEdges: view(snappedEdges, 'uint32', rowCount),
          snapFractions: view(snapFractions, 'float32', rowCount),
          snapDistances: view(snapDistances, 'float32', rowCount),
          snappedPositions: view(snappedPositions, 'float32x2', rowCount),
          seedNodes: seedNodesView,
          seedCosts: seedCostsView,
          ...(options.snapSearch === 'bvh' ? {overflow: view(snapOverflow, 'uint32', 1)} : {})
        })
      );
      matrixGraph.add(
        new GPUNetworkCostMatrix({
          id: 'cost-matrix',
          offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
          neighbors: view(neighborsBuffer, 'uint32', edgeCount),
          weights: view(weightsBuffer, 'float32', edgeCount),
          seedNodes: seedNodesView,
          seedCosts: seedCostsView,
          seedsPerRow: 2,
          costLimit: costLimitParameter.importToGraph(matrixGraph),
          laneCount,
          maxIterations: MAXIMUM_ITERATIONS,
          localIterations: LOCAL_ITERATIONS,
          costs: view(matrixBuffer, 'float32', matrixLength),
          converged: view(matrixConverged, 'uint32', 1)
        })
      );
    }
    const scoreGraph = new GPUCommandGraph<void>(device, {id: 'access-score'});
    {
      const view = importer(scoreGraph);
      scoreGraph.add(
        new GPUNetworkAccessibility({
          id: 'score',
          costs: view(matrixBuffer, 'float32', matrixLength),
          opportunityWeights: view(opportunityWeights, 'float32', rowCount),
          parameters: scoring.importToGraph(scoreGraph),
          cumulative: view(cumulativeBuffer, 'float32', nodeCount),
          gravity: view(gravityBuffer, 'float32', nodeCount),
          catchment: {
            demand: view(demandBuffer, 'float32', nodeCount),
            output: view(twoStepBuffer, 'float32', nodeCount),
            ratios: view(ratiosBuffer, 'float32', rowCount)
          }
        })
      );
    }
    built = {
      matrix: resources.track(matrixGraph.compile()),
      score: resources.track(scoreGraph.compile()),
      rowCount
    };
    builtKey = getBuildKey();

    snapReader = new SummaryReader(
      resources,
      `access-snap-${rowCount}-${laneCount}`,
      [
        {buffer: matrixConverged, size: 4},
        {buffer: snapOverflow, size: 4},
        {buffer: snappedEdges, size: rowCount * 4},
        {buffer: snapDistances, size: rowCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const [converged, overflow, edges, distances] = sliceSections(bytes, [
          4,
          4,
          rowCount * 4,
          rowCount * 4
        ]);
        const edgeWords = new Uint32Array(edges);
        const distanceValues = new Float32Array(distances);
        let snapped = 0;
        let sum = 0;
        let maximum = 0;
        for (let row = 0; row < rowCount; row++) {
          if (edgeWords[row] !== GPU_NETWORK_SNAPPING_NONE) {
            snapped++;
            sum += distanceValues[row];
            maximum = Math.max(maximum, distanceValues[row]);
          }
        }
        ctx.setReadout('snapped', `${snapped} of ${rowCount}`);
        ctx.setReadout(
          'snapDistance',
          snapped ? `${(sum / snapped).toFixed(0)} m / ${maximum.toFixed(0)} m` : 'n/a'
        );
        ctx.setReadout(
          'converged',
          (new Uint32Array(converged)[0] ? 'yes' : `no (${MAXIMUM_ITERATIONS}-round limit)`) +
            (new Uint32Array(overflow)[0] ? ', snapping candidates overflowed' : '')
        );
      }
    );
    ctx.setReadout(
      'matrix',
      `${rowCount} x ${formatInteger(nodeCount)} = ${formatInteger(matrixLength)} (${(matrixLength / 262144).toFixed(0)} MiB), ${laneCount} lanes`
    );
    ctx.setReadout(
      'opportunityShare',
      `${((100 * rowsJobs) / totalJobs).toFixed(0)}% of ${formatInteger(totalJobs)} jobs`
    );
    matrixDirty = 2;
    scoreDirty = 2;
    matrixTimingStale = true;
    matrixEncodeCount = 0;
    scoreEncodeCount = 0;
    scheduleTiming();
  }

  function getBuildKey(): string {
    const options = ctx.options;
    return [
      options.opportunityRows,
      options.laneCount,
      options.snapSearch,
      options.seedDirection
    ].join('|');
  }

  // Score statistics: percentiles over the road nodes of the displayed measure.
  const scoreRing = createScoreHolder();
  function readScores(
    commandEncoder: Parameters<SceneInstance<JobAccessibilityOptions>['encode']>[0]
  ) {
    scoreRing.request(commandEncoder);
  }
  function createScoreHolder() {
    let reader: SummaryReader | null = null;
    let readerBuffer: Buffer | null = null;
    const process = (bytes: ArrayBuffer) => {
      if (destroyed) return;
      const scores = new Float32Array(bytes).subarray(0, roadNodeCount);
      let positive = 0;
      let maximum = 0;
      let sum = 0;
      for (const score of scores) {
        if (score > 0) positive++;
        if (score > maximum) maximum = score;
        sum += score;
      }
      const sorted = Float32Array.from(scores).sort();
      const median = sorted[Math.floor(sorted.length / 2)];
      const percentile =
        sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * SCORE_PERCENTILE))];
      const range = percentile > 0 ? percentile : maximum || 1;
      const measure = ctx.options.measure;
      const format = (value: number) =>
        measure === 'two-step'
          ? value.toExponential(2)
          : measure === 'cumulative'
            ? formatInteger(value)
            : formatInteger(value);
      ctx.setReadout(
        'reached',
        `${((100 * positive) / roadNodeCount).toFixed(0)}% of intersections`
      );
      ctx.setReadout('median', format(median));
      ctx.setReadout('mean', format(sum / roadNodeCount));
      ctx.setReadout('top', `${format(percentile)} (98th percentile), max ${format(maximum)}`);
      if (Math.abs(range - scoreMaximum) > 0.01 * scoreMaximum) {
        scoreMaximum = range;
        ctx.setLegendExtent('score', [0, range]);
        ctx.requestLayers();
      }
    };
    return {
      request(commandEncoder: Parameters<SceneInstance<JobAccessibilityOptions>['encode']>[0]) {
        const buffer = getScoreBuffer();
        if (!reader || readerBuffer !== buffer) {
          reader?.stop();
          readerBuffer = buffer;
          reader = new SummaryReader(
            resources,
            `access-scores-${buffer.id}`,
            [{buffer, size: nodeCount * 4}],
            process
          );
        }
        reader.request(commandEncoder);
      },
      flush(commandEncoder: Parameters<SceneInstance<JobAccessibilityOptions>['encode']>[0]) {
        reader?.flush(commandEncoder);
      },
      stop() {
        reader?.stop();
        reader = null;
      }
    };
  }

  // ---- Timing -----------------------------------------------------------------------------
  function scheduleTiming(): void {
    clearTimeout(timingTimer);
    timingTimer = setTimeout(() => void measureGraphs(), TIMING_DEBOUNCE_MILLISECONDS);
  }

  async function measureGraphs(): Promise<void> {
    if (destroyed || !built) return;
    if (timingPending || matrixDirty > 0 || scoreDirty > 0) {
      scheduleTiming();
      return;
    }
    timingPending = true;
    const current = built;
    try {
      if (!matrixTiming || matrixTimingStale) {
        ctx.setReadout('matrixTime', 'measuring...');
        matrixTiming = await measureCompiledGraph(device, current.matrix, {
          parameters: undefined,
          completionBuffer: matrixConverged,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 1,
          signal: ctx.signal
        });
        matrixTimingStale = false;
      }
      const scoreTiming = await measureCompiledGraph(device, current.score, {
        parameters: undefined,
        completionBuffer: matrixConverged,
        signal: ctx.signal
      });
      if (!destroyed && built === current) {
        ctx.setReadout(
          'matrixTime',
          `${formatCompiledGraphTiming(matrixTiming)} (${current.matrix.stats.nodeOrder.length} graph nodes)`
        );
        ctx.setReadout(
          'scoreTime',
          `${formatCompiledGraphTiming(scoreTiming)} (${current.score.stats.nodeOrder.length} graph nodes)`
        );
      }
    } catch {
      // Aborted by a scene switch or a rebuild.
    } finally {
      timingPending = false;
    }
  }

  // ---- Initial state ----------------------------------------------------------------------
  ctx.setReadout(
    'network',
    `${formatInteger(roadNodeCount)} intersections + ${formatInteger(stopCount)} stops (${formatInteger(nodeCount)} nodes), ${formatInteger(edgeCount)} edges`
  );
  writeWeights();
  writeScalars();
  writeScoring();
  buildGraphs();

  const railPalette = lineNames.map(name => LINE_COLORS[name]);

  return {
    getCompiledGraphs: () => (built ? [built.matrix, built.score] : []),

    setOption(id, _value, state) {
      switch (id) {
        case 'opportunityRows':
        case 'laneCount':
        case 'snapSearch':
        case 'seedDirection':
          if (builtKey !== getBuildKey()) {
            buildGraphs();
            ctx.requestLayers();
          }
          break;
        case 'transit':
        case 'walkSpeed':
        case 'waitFactor':
          writeWeights();
          markMatrixDirty();
          ctx.requestLayers();
          break;
        case 'matrixLimitMinutes':
        case 'maxSnapDistance':
          writeScalars();
          writeScoring();
          markMatrixDirty();
          break;
        case 'measure':
        case 'thresholdMinutes':
        case 'beta':
        case 'powerExponent':
        case 'minimumCostSeconds':
          writeScoring();
          scoreRing.stop();
          scoreMaximum = 0;
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
      void state;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip() {
      return null;
    },

    encode(commandEncoder) {
      if (!built) return;
      if (matrixDirty > 0 && performance.now() >= matrixPendingAt) {
        built.matrix.encode(commandEncoder, {parameters: undefined});
        for (let row = 0; row < built.rowCount; row++) {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: snappedPositions,
            sourceOffset: row * 8,
            destinationBuffer: snapSegments,
            destinationOffset: row * 16 + 8,
            size: 8
          });
        }
        matrixDirty--;
        matrixEncodeCount++;
        snapReader?.markStale();
        if (matrixDirty === 0) scoreDirty = Math.max(scoreDirty, 1);
      }
      if (scoreDirty > 0 && matrixDirty === 0 && matrixEncodeCount > 0) {
        built.score.encode(commandEncoder, {parameters: undefined});
        scoreDirty--;
        scoreEncodeCount++;
        ctx.setReadout('encodes', `${matrixEncodeCount} / ${scoreEncodeCount}`);
        if (scoreDirty === 0) readScores(commandEncoder);
      }
      snapReader?.flush(commandEncoder);
      scoreRing.flush(commandEncoder);
    },

    getLayers() {
      if (!built) return [];
      const options = ctx.options;
      const colors = getRoadColors(ctx.theme());
      const coordinateOrigin: [number, number, number] = [network.origin[0], network.origin[1], 0];
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'access-roads-base',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1,
          color: colors.minor
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'access-roads-major',
          coordinateOrigin,
          segments: majorSegmentsBuffer,
          instanceCount: network.majorSegments.length / 4,
          widthPixels: 1.5,
          color: colors.major
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'access-roads-score',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 2.4,
          values: getScoreBuffer(),
          valueFormat: 'float32',
          valueIndices: segmentTargetsBuffer,
          colormap: options.ramp,
          valueRange: [0, scoreMaximum],
          sqrtScale: true,
          color: [255, 255, 255, 240],
          noDataColor: [0, 0, 0, 0]
        })
      ];
      if (options.showTransit) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'access-bus-lines',
            coordinateOrigin,
            segments: busBuffer,
            instanceCount: busSegments.length / 4,
            widthPixels: 0.8,
            color: ctx.theme() === 'dark' ? [200, 205, 220, 40] : [60, 70, 100, 40]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'access-rail-halo',
            coordinateOrigin,
            segments: railBuffer,
            instanceCount: railSegments.length / 4,
            widthPixels: 6.5,
            color: colors.halo
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'access-rail-lines',
            coordinateOrigin,
            segments: railBuffer,
            instanceCount: railSegments.length / 4,
            widthPixels: 3.4,
            values: railLineBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: railPalette
          })
        );
      }
      if (options.showStops) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'access-stops',
            coordinateOrigin,
            positions: stopPositionsBuffer,
            instanceCount: stopCount,
            radiusPixels: 2,
            values: stopModeBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              ctx.theme() === 'dark' ? [220, 225, 240, 150] : [40, 50, 80, 140],
              ctx.theme() === 'dark' ? [255, 255, 255, 255] : [20, 24, 36, 255]
            ]
          })
        );
      }
      if (options.showSnaps) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'access-snap-segments',
            coordinateOrigin,
            segments: snapSegments,
            instanceCount: built.rowCount,
            widthPixels: 2.5,
            color: [255, 140, 60, 255]
          })
        );
      }
      if (options.showOpportunities) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'access-opportunity-halo',
            coordinateOrigin,
            positions: opportunityPositions,
            instanceCount: built.rowCount,
            radiusPixels: 6,
            color: colors.halo
          }),
          new SpatialAnalysisPointLayer({
            id: 'access-opportunities',
            coordinateOrigin,
            positions: opportunityPositions,
            instanceCount: built.rowCount,
            radiusPixels: 4,
            color: [90, 220, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'access-snapped',
            coordinateOrigin,
            positions: snappedPositions,
            instanceCount: built.rowCount,
            radiusPixels: 2.5,
            color: [255, 255, 255, 255]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      clearTimeout(timingTimer);
      snapReader?.stop();
      scoreRing.stop();
      resources.destroy();
    }
  };
}
