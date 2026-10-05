// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Walking accessibility over the New York road network.
 *
 * A sample of points of interest are the opportunities (jobs). Two compiled graphs split the work
 * by how often its inputs change:
 *
 * - The matrix graph runs `GPUNetworkSnapping` (each opportunity snaps to its nearest street edge
 *   and becomes two seeds) and `GPUNetworkCostMatrix` (one bounded shortest-path search per
 *   opportunity over the street graph, lane-batched). It is iterative and expensive, so it is
 *   encoded only when the opportunity set changes ("Resample opportunities"), never for a slider.
 * - The score graph runs `GPUNetworkAccessibility` over the retained matrix: cumulative
 *   opportunities, gravity (exponential or power decay) and 2SFCA. Measure, threshold and beta are
 *   a 4-word parameter buffer, so moving them re-runs a few linear passes and no search. Results
 *   persist in buffers, so the score graph is encoded only on frames where a parameter changed.
 *
 * The street graph lists both directions of every street, so the forward CSR is also the reverse
 * CSR that the opportunity-side matrix needs.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_NETWORK_SNAPPING_NONE,
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  encodeGPUNetworkAccessibilityParameters,
  type GPUNetworkAccessibilityParameters
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {createSeededRandom} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {sortEdgesBySource, WALK_SPEED} from './road-network-utils';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

type Measure = 'cumulative' | 'gravity-exponential' | 'gravity-power' | 'two-step';

/** Opportunities (matrix rows). Compile-time: it sets the matrix shape and the graph size. */
const OPPORTUNITY_COUNT = 192;
/** Matrix cost limit in seconds. The threshold slider may not exceed it. */
const COST_LIMIT_SECONDS = 20 * 60;
/** Relaxation rounds per lane batch (each covers up to 16 hops); the batch stops early on the GPU. */
const MAXIMUM_ITERATIONS = 40;
const LANE_COUNT = 32;
/** Power decay floor in seconds, so zero-cost entries stay finite. */
const POWER_MINIMUM_COST_SECONDS = 60;
const SCORE_PERCENTILE = 0.98;
const TIMING_DEBOUNCE_MILLISECONDS = 350;
const FACILITY_COLOR = [90, 220, 255, 255] as const;
const SNAP_COLOR = [255, 255, 255, 255] as const;
const SNAP_SEGMENT_COLOR = [255, 140, 60, 255] as const;
const INFERNO_COLORS = [
  [0, 0, 4],
  [87, 16, 110],
  [188, 55, 84],
  [249, 142, 9],
  [252, 255, 164]
] as const;

export const accessibilityMode: SpatialAnalysisModeDefinition = {
  id: 'accessibility',
  title: 'Access',
  contributors: ['GPUNetworkSnapping', 'GPUNetworkCostMatrix', 'GPUNetworkAccessibility'],
  description:
    'Walking accessibility to sampled points of interest over the New York street graph. ' +
    'Opportunities snap to street edges; the cost matrix is computed once per sample, and the ' +
    'measure, time threshold and decay re-score it without any new search.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [roads, pois] = await Promise.all([
      context.data.getNewYorkRoads(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'accessibility');
    const nodeCount = roads.nodePositions.length / 2;
    const segmentCount = roads.segmentNodes.length;
    const edges = sortEdgesBySource(roads);
    const edgeCount = edges.sources.length;
    const poiCount = pois.positions.length / 2;
    const rowCount = Math.min(OPPORTUNITY_COUNT, poiCount);
    const matrixLength = rowCount * nodeCount;
    if (matrixLength * 4 > device.limits.maxStorageBufferBindingSize) {
      throw new Error(
        `accessibility matrix ${rowCount} x ${nodeCount} exceeds maxStorageBufferBindingSize`
      );
    }

    // Walking seconds per sorted edge; sorted edge e is CSR edge e. Both directions of every
    // street are listed, so this CSR is its own reverse.
    const edgeSeconds = new Float32Array(edgeCount);
    const csrOffsets = new Uint32Array(nodeCount + 1);
    for (let edge = 0; edge < edgeCount; edge++) {
      edgeSeconds[edge] = Math.max(edges.lengths[edge], 0.5) / WALK_SPEED;
      csrOffsets[edges.sources[edge] + 1]++;
    }
    for (let node = 0; node < nodeCount; node++) csrOffsets[node + 1] += csrOffsets[node];
    // Demand per node for 2SFCA: a deterministic 1-5 pseudo-random population.
    const demand = new Float32Array(nodeCount);
    const demandRandom = createSeededRandom(23);
    for (let node = 0; node < nodeCount; node++) demand[node] = 1 + Math.floor(demandRandom() * 5);

    let measure: Measure = 'gravity-exponential';
    let thresholdMinutes = 15;
    let betaValue = 1;
    let sampleSeed = 1;
    let matrixDirty = true;
    let scoreDirty = true;
    let scoreReadbackWanted = true;
    let scoreReadbackPending = false;
    let matrixReadbackWanted = true;
    let matrixReadbackPending = false;
    let timingPending = false;
    let timingTimer: ReturnType<typeof setTimeout> | undefined;
    let destroyed = false;
    let matrixEncodeCount = 0;
    let scoreEncodeCount = 0;
    let scoreMaximum = 1;
    let matrixTiming: CompiledGraphTiming | null = null;
    let scoreTiming: CompiledGraphTiming | null = null;
    let matrixTimingStale = true;

    // --- Buffers -------------------------------------------------------------------------------
    const nodePositions = resources.createBuffer('node-positions', roads.nodePositions);
    const edgeSources = resources.createBuffer('edge-sources', edges.sources);
    const edgeTargets = resources.createBuffer('edge-targets', edges.targets);
    const edgeCosts = resources.createBuffer('edge-costs', edgeSeconds);
    const offsetsBuffer = resources.createBuffer('csr-offsets', csrOffsets);
    const facilityPositions = resources.createBuffer('facility-positions', rowCount * 8);
    const opportunityWeights = resources.createBuffer('opportunity-weights', rowCount * 4);
    const snappedEdges = resources.createBuffer('snapped-edges', rowCount * 4);
    const snapDistances = resources.createBuffer('snap-distances', rowCount * 4);
    const snappedPositions = resources.createBuffer('snapped-positions', rowCount * 8);
    const seedNodes = resources.createBuffer('seed-nodes', rowCount * 8);
    const seedCosts = resources.createBuffer('seed-costs', rowCount * 8);
    // Facility point then snapped point per opportunity: x0, y0, x1, y1 rows for the segment layer.
    const snapSegments = resources.createBuffer('snap-segments', rowCount * 16);
    const matrix = resources.createBuffer('matrix', matrixLength * 4);
    const matrixConverged = resources.createBuffer('matrix-converged', 4);
    const demandBuffer = resources.createBuffer('demand', demand);
    const cumulative = resources.createBuffer('cumulative', nodeCount * 4);
    const gravity = resources.createBuffer('gravity', nodeCount * 4);
    const twoStep = resources.createBuffer('two-step', nodeCount * 4);
    const facilityRatios = resources.createBuffer('facility-ratios', rowCount * 4);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const segmentNodes = resources.createBuffer('segment-nodes', roads.segmentNodes);
    const costLimit = resources.createParameterBuffer(
      'cost-limit',
      'float32',
      1,
      Float32Array.of(COST_LIMIT_SECONDS)
    );
    const scoring = resources.createParameterBuffer('scoring', 'float32', 4);
    const snapSummaryBytes = 4 + rowCount * 8;
    const snapRing = resources.track(
      new GPUReadbackRing(device, {id: 'accessibility-snap-summary', byteLength: snapSummaryBytes})
    );
    const scoreRing = resources.track(
      new GPUReadbackRing(device, {id: 'accessibility-scores', byteLength: nodeCount * 4})
    );

    // --- Matrix graph: snapping, then the many-to-all cost matrix ------------------------------
    const matrixGraph = new GPUCommandGraph<void>(device, {id: 'accessibility-matrix'});
    const edgeTargetsView = importGraphBuffer(
      matrixGraph,
      'edge-targets',
      edgeTargets,
      'uint32',
      edgeCount
    );
    const edgeCostsView = importGraphBuffer(
      matrixGraph,
      'edge-costs',
      edgeCosts,
      'float32',
      edgeCount
    );
    const seedNodesView = importGraphBuffer(
      matrixGraph,
      'seed-nodes',
      seedNodes,
      'uint32',
      rowCount * 2
    );
    const seedCostsView = importGraphBuffer(
      matrixGraph,
      'seed-costs',
      seedCosts,
      'float32',
      rowCount * 2
    );
    matrixGraph.add(
      new GPUNetworkSnapping({
        id: 'snapping',
        points: importGraphBuffer(
          matrixGraph,
          'facilities',
          facilityPositions,
          'float32x2',
          rowCount
        ),
        nodePositions: importGraphBuffer(
          matrixGraph,
          'node-positions',
          nodePositions,
          'float32x2',
          nodeCount
        ),
        edgeSources: importGraphBuffer(
          matrixGraph,
          'edge-sources',
          edgeSources,
          'uint32',
          edgeCount
        ),
        edgeTargets: edgeTargetsView,
        edgeCosts: edgeCostsView,
        seedDirection: 'both',
        snappedEdges: importGraphBuffer(
          matrixGraph,
          'snapped-edges',
          snappedEdges,
          'uint32',
          rowCount
        ),
        snapDistances: importGraphBuffer(
          matrixGraph,
          'snap-distances',
          snapDistances,
          'float32',
          rowCount
        ),
        snappedPositions: importGraphBuffer(
          matrixGraph,
          'snapped-positions',
          snappedPositions,
          'float32x2',
          rowCount
        ),
        seedNodes: seedNodesView,
        seedCosts: seedCostsView
      })
    );
    matrixGraph.add(
      new GPUNetworkCostMatrix({
        id: 'cost-matrix',
        offsets: importGraphBuffer(
          matrixGraph,
          'csr-offsets',
          offsetsBuffer,
          'uint32',
          nodeCount + 1
        ),
        neighbors: edgeTargetsView,
        weights: edgeCostsView,
        seedNodes: seedNodesView,
        seedCosts: seedCostsView,
        seedsPerRow: 2,
        costLimit: costLimit.importToGraph(matrixGraph),
        laneCount: Math.min(LANE_COUNT, rowCount),
        maxIterations: MAXIMUM_ITERATIONS,
        costs: importGraphBuffer(matrixGraph, 'matrix', matrix, 'float32', matrixLength),
        converged: importGraphBuffer(matrixGraph, 'matrix-converged', matrixConverged, 'uint32', 1)
      })
    );
    const compiledMatrix: CompiledGPUCommandGraph<void> = resources.track(matrixGraph.compile());

    // --- Score graph: all three measures every encoding; the layer picks one -------------------
    const scoreGraph = new GPUCommandGraph<void>(device, {id: 'accessibility-score'});
    scoreGraph.add(
      new GPUNetworkAccessibility({
        id: 'score',
        costs: importGraphBuffer(scoreGraph, 'matrix', matrix, 'float32', matrixLength),
        opportunityWeights: importGraphBuffer(
          scoreGraph,
          'opportunity-weights',
          opportunityWeights,
          'float32',
          rowCount
        ),
        parameters: scoring.importToGraph(scoreGraph),
        cumulative: importGraphBuffer(scoreGraph, 'cumulative', cumulative, 'float32', nodeCount),
        gravity: importGraphBuffer(scoreGraph, 'gravity', gravity, 'float32', nodeCount),
        catchment: {
          demand: importGraphBuffer(scoreGraph, 'demand', demandBuffer, 'float32', nodeCount),
          output: importGraphBuffer(scoreGraph, 'two-step', twoStep, 'float32', nodeCount),
          ratios: importGraphBuffer(
            scoreGraph,
            'facility-ratios',
            facilityRatios,
            'float32',
            rowCount
          )
        }
      })
    );
    const compiledScore: CompiledGPUCommandGraph<void> = resources.track(scoreGraph.compile());

    // --- Controls ------------------------------------------------------------------------------
    const getScoreBuffer = (): Buffer =>
      measure === 'cumulative' ? cumulative : measure === 'two-step' ? twoStep : gravity;

    const writeScoring = () => {
      const parameters: GPUNetworkAccessibilityParameters = {
        threshold: thresholdMinutes * 60,
        decay: 'none'
      };
      if (measure === 'gravity-exponential') {
        // Exponent per 10 minutes of walking, converted to per second.
        Object.assign(parameters, {decay: 'exponential', beta: betaValue / 600});
      } else if (measure === 'gravity-power') {
        Object.assign(parameters, {
          decay: 'power',
          beta: betaValue,
          minimumCost: POWER_MINIMUM_COST_SECONDS
        });
      }
      scoring.write(encodeGPUNetworkAccessibilityParameters(parameters));
      scoreDirty = true;
      scoreReadbackWanted = true;
      scheduleTiming();
    };

    context.controls.addSelect<Measure>({
      label: 'Measure (decay code in the parameter buffer)',
      options: [
        {value: 'cumulative', label: 'Cumulative opportunities'},
        {value: 'gravity-exponential', label: 'Gravity, exponential decay'},
        {value: 'gravity-power', label: 'Gravity, power decay'},
        {value: 'two-step', label: '2SFCA (supply / demand, no decay)'}
      ],
      value: measure,
      onChange: value => {
        measure = value;
        writeScoring();
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Walking threshold (per-frame parameter)',
      min: 2,
      max: COST_LIMIT_SECONDS / 60,
      step: 1,
      value: thresholdMinutes,
      format: value => `${value} min`,
      onChange: value => {
        thresholdMinutes = value;
        writeScoring();
      }
    });
    context.controls.addSlider({
      label: 'Beta (exp: per 10 min, power: exponent)',
      min: 0.1,
      max: 4,
      step: 0.05,
      value: betaValue,
      format: value => value.toFixed(2),
      onChange: value => {
        betaValue = value;
        writeScoring();
      }
    });
    context.controls.addButton({
      label: 'Resample opportunities (re-runs the matrix)',
      onClick: () => {
        sampleSeed++;
        writeSample();
      }
    });
    context.controls.addLegend({
      title: 'Accessibility score (clipped at the 98th percentile)',
      gradient: {
        colors: INFERNO_COLORS,
        minimumLabel: '0',
        maximumLabel: 'high'
      }
    });
    context.controls.addNote(
      'The cost matrix is iterative (one search per opportunity), so it is encoded only when the ' +
        'opportunity sample changes. Measure, threshold and beta only write a 4-word parameter ' +
        'buffer and re-score the retained matrix. Orange segments join each opportunity to its ' +
        'snapped point on the nearest street edge. 2SFCA ignores beta (classic catchment).'
    );
    context.controls.addReadout(
      'Network',
      `${formatCount(nodeCount)} nodes, ${formatCount(edgeCount)} edges`
    );
    const snappedReadout = context.controls.addReadout('Snapped opportunities');
    const snapDistanceReadout = context.controls.addReadout('Snap distance (mean / max)');
    context.controls.addReadout(
      'Cost matrix',
      `${formatCount(rowCount)} × ${formatCount(nodeCount)} = ${formatCount(matrixLength)} ` +
        `(${(matrixLength / 262144).toFixed(1)} MiB)`
    );
    const convergedReadout = context.controls.addReadout('Matrix converged');
    const matrixTimeReadout = context.controls.addReadout('Matrix time (snap + search)');
    const scoreTimeReadout = context.controls.addReadout('Re-score time');
    const encodeCountReadout = context.controls.addReadout('Encodes (matrix / score)');
    const reachedReadout = context.controls.addReadout('Nodes with score > 0');
    const scoreRangeReadout = context.controls.addReadout('Score max / 98th pct');
    context.controls.addReadout('Data', `${roads.attribution}; ${pois.attribution}`);

    // --- Opportunity sample --------------------------------------------------------------------
    function writeSample(): void {
      const random = createSeededRandom(sampleSeed * 7919);
      const chosen = new Set<number>();
      while (chosen.size < rowCount) chosen.add(Math.floor(random() * poiCount));
      const positions = new Float32Array(rowCount * 2);
      const weights = new Float32Array(rowCount);
      const segments = new Float32Array(rowCount * 4);
      let row = 0;
      for (const poi of chosen) {
        positions[row * 2] = pois.positions[poi * 2];
        positions[row * 2 + 1] = pois.positions[poi * 2 + 1];
        segments[row * 4] = positions[row * 2];
        segments[row * 4 + 1] = positions[row * 2 + 1];
        weights[row] = 1 + Math.floor(random() * 5);
        row++;
      }
      facilityPositions.write(positions);
      opportunityWeights.write(weights);
      // The snapped half of each segment is copied from the snapping output after it is encoded.
      snapSegments.write(segments);
      matrixDirty = true;
      matrixReadbackWanted = true;
      scoreDirty = true;
      scoreReadbackWanted = true;
      scheduleTiming();
    }

    // --- Timing and readback -------------------------------------------------------------------
    function scheduleTiming(): void {
      clearTimeout(timingTimer);
      timingTimer = setTimeout(() => void measureGraphs(), TIMING_DEBOUNCE_MILLISECONDS);
    }

    async function measureGraphs(): Promise<void> {
      if (destroyed || timingPending || matrixDirty) {
        if (!destroyed) scheduleTiming();
        return;
      }
      timingPending = true;
      try {
        // Timing re-encodes outside Deck's frame; the graphs are deterministic so results are equal.
        if (!matrixTiming || matrixTimingStale) {
          matrixTimeReadout.setValue('measuring...');
          matrixTiming = await measureCompiledGraph(device, compiledMatrix, {
            parameters: undefined,
            completionBuffer: matrixConverged,
            runs: 3,
            warmUpRuns: 1,
            repetitions: 1,
            signal: context.signal
          });
          matrixTimingStale = false;
        }
        scoreTiming = await measureCompiledGraph(device, compiledScore, {
          parameters: undefined,
          completionBuffer: matrixConverged,
          signal: context.signal
        });
        if (!destroyed) {
          matrixTimeReadout.setValue(
            `${formatCompiledGraphTiming(matrixTiming)} · ${compiledMatrix.stats.nodeOrder.length} nodes`
          );
          scoreTimeReadout.setValue(
            `${formatCompiledGraphTiming(scoreTiming)} · ${compiledScore.stats.nodeOrder.length} nodes`
          );
        }
      } catch {
        // Aborted by a mode switch.
      } finally {
        timingPending = false;
      }
    }
    const readSnapSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = snapRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: matrixConverged,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: snappedEdges,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4,
        size: rowCount * 4
      });
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: snapDistances,
        destinationBuffer: ticket.buffer,
        destinationOffset: 4 + rowCount * 4,
        size: rowCount * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: snapSummaryBytes});
      matrixReadbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const distances = new Float32Array(
          bytes.buffer,
          bytes.byteOffset + 4 + rowCount * 4,
          rowCount
        );
        let snapped = 0;
        let distanceSum = 0;
        let distanceMaximum = 0;
        for (let row = 0; row < rowCount; row++) {
          if (words[1 + row] !== GPU_NETWORK_SNAPPING_NONE) {
            snapped++;
            distanceSum += distances[row];
            distanceMaximum = Math.max(distanceMaximum, distances[row]);
          }
        }
        snappedReadout.setValue(`${snapped} of ${rowCount}`);
        snapDistanceReadout.setValue(
          snapped
            ? `${(distanceSum / snapped).toFixed(0)} m / ${distanceMaximum.toFixed(0)} m`
            : 'n/a'
        );
        convergedReadout.setValue(words[0] ? 'yes' : `no (${MAXIMUM_ITERATIONS}-round limit)`);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        matrixReadbackPending = false;
      }
    };

    const readScores = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = scoreRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: getScoreBuffer(),
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: nodeCount * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: nodeCount * 4});
      scoreReadbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const scores = Float32Array.from(
          new Float32Array(bytes.buffer, bytes.byteOffset, nodeCount)
        );
        let positive = 0;
        let maximum = 0;
        for (const score of scores) {
          if (score > 0) positive++;
          if (score > maximum) maximum = score;
        }
        const sorted = scores.sort();
        const percentile =
          sorted[Math.min(nodeCount - 1, Math.floor(nodeCount * SCORE_PERCENTILE))];
        const range = percentile > 0 ? percentile : maximum || 1;
        reachedReadout.setValue(`${formatCount(positive)} of ${formatCount(nodeCount)}`);
        scoreRangeReadout.setValue(`${maximum.toPrecision(3)} / ${percentile.toPrecision(3)}`);
        if (Math.abs(range - scoreMaximum) > 0.01 * scoreMaximum) {
          scoreMaximum = range;
          context.updateLayers();
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        scoreReadbackPending = false;
      }
    };

    writeSample();
    writeScoring();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledMatrix, compiledScore],
      encode(commandEncoder) {
        if (matrixDirty) {
          // Iterative many-to-all search: only when the opportunity set changed.
          compiledMatrix.encode(commandEncoder, {parameters: undefined});
          for (let row = 0; row < rowCount; row++) {
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: snappedPositions,
              sourceOffset: row * 8,
              destinationBuffer: snapSegments,
              destinationOffset: row * 16 + 8,
              size: 8
            });
          }
          matrixDirty = false;
          matrixTimingStale = true;
          matrixEncodeCount++;
        }
        if (scoreDirty) {
          // Results persist in buffers, so rescoring is only needed after a parameter changed.
          compiledScore.encode(commandEncoder, {parameters: undefined});
          scoreDirty = false;
          scoreEncodeCount++;
          encodeCountReadout.setValue(`${matrixEncodeCount} / ${scoreEncodeCount}`);
        }
        if (matrixReadbackWanted && !matrixReadbackPending && matrixEncodeCount > 0) {
          matrixReadbackWanted = false;
          void readSnapSummary(commandEncoder);
        }
        if (scoreReadbackWanted && !scoreReadbackPending && scoreEncodeCount > 0) {
          scoreReadbackWanted = false;
          void readScores(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'accessibility-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.5,
            values: getScoreBuffer(),
            valueFormat: 'float32',
            valueIndices: segmentNodes,
            colormap: 'inferno',
            valueRange: [0, scoreMaximum],
            sqrtScale: true,
            noDataColor: [90, 92, 105, 60]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'accessibility-snap-segments',
            coordinateOrigin,
            segments: snapSegments,
            instanceCount: rowCount,
            widthPixels: 2,
            colormap: 'uniform',
            color: SNAP_SEGMENT_COLOR
          }),
          new SpatialAnalysisPointLayer({
            id: 'accessibility-opportunities',
            coordinateOrigin,
            positions: facilityPositions,
            instanceCount: rowCount,
            radiusPixels: 4,
            colormap: 'uniform',
            color: FACILITY_COLOR
          }),
          new SpatialAnalysisPointLayer({
            id: 'accessibility-snapped',
            coordinateOrigin,
            positions: snappedPositions,
            instanceCount: rowCount,
            radiusPixels: 2.5,
            colormap: 'uniform',
            color: SNAP_COLOR
          })
        ];
        return layers;
      },
      destroy() {
        destroyed = true;
        clearTimeout(timingTimer);
        resources.destroy();
      }
    };
    return instance;
  }
};
