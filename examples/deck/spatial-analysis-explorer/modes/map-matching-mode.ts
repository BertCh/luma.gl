// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  encodeGPUMapMatchingParameters,
  GPUMapMatching,
  GPU_MAP_MATCHING_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {GPULineMerge} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {sortEdgesBySource} from './road-network-utils';
import {SummaryReader} from './summary-reader';

/** Candidate edges per GPS point. Both directions of a two-way road take a slot. */
const CANDIDATE_COUNT = 6;
/** Edge grid cell size in meters; the search radius is capped at twice this. */
const CELL_SIZE = 60;
const MAXIMUM_SEARCH_RADIUS = 2 * CELL_SIZE;
/** Route search table size (nodes settled per search); larger is closer to an unlimited search. */
const ROUTE_NODE_BUDGET = 64;
/** Minimum seconds between re-matches while sliders move; the last change always lands. */
const REMATCH_INTERVAL_SECONDS = 0.15;
const ROUTE_FACTOR = 3;
const ROUTE_SLACK = 100;

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

/**
 * Hidden Markov model map matching of noisy taxi GPS traces onto the New York road network.
 * The "raw" trip vertices get Gaussian jitter from a slider (a per-frame parameter), then
 * `GPUMapMatching` snaps every track to the directed road graph with a Viterbi pass per track
 * (one thread per track); emission sigma, transition beta and the search radius are per-frame
 * parameters too, so the matching re-runs on the GPU as sliders move without recompiling.
 * `GPULineMerge` joins the road segments into maximal chains between junctions, shown as a
 * toggle that colors each chain.
 */
export const mapMatchingMode: SpatialAnalysisModeDefinition = {
  id: 'map-matching',
  title: 'Map matching',
  contributors: ['GPUMapMatching', 'GPULineMerge'],
  description:
    'Noisy GPS taxi traces (orange) matched to the road graph (cyan) by a GPU hidden Markov ' +
    'model. Raise the GPS jitter, then tune emission sigma, transition beta and the search ' +
    'radius: matching re-runs per frame with no recompile, and the matched fraction and break ' +
    'count update. Merge road chains colors the roads as maximal junction-to-junction chains.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [trips, roads] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkRoads()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'map-matching');
    const pointCount = trips.vertexTimestamps.length;
    const trackCount = trips.tripOffsets.length - 1;
    const sorted = sortEdgesBySource(roads);
    const nodeCount = roads.nodePositions.length / 2;
    const edgeCount = sorted.sources.length;
    const roadSegmentCount = roads.segments.length / 4;

    const offsets = new Uint32Array(nodeCount + 1);
    for (let edge = 0; edge < edgeCount; edge++) offsets[sorted.sources[edge] + 1]++;
    for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let node = 0; node < nodeCount; node++) {
      minX = Math.min(minX, roads.nodePositions[node * 2]);
      maxX = Math.max(maxX, roads.nodePositions[node * 2]);
      minY = Math.min(minY, roads.nodePositions[node * 2 + 1]);
      maxY = Math.max(maxY, roads.nodePositions[node * 2 + 1]);
    }
    for (let vertex = 0; vertex < pointCount; vertex++) {
      minX = Math.min(minX, trips.vertexPositions[vertex * 2] - 4 * CELL_SIZE);
      maxX = Math.max(maxX, trips.vertexPositions[vertex * 2] + 4 * CELL_SIZE);
      minY = Math.min(minY, trips.vertexPositions[vertex * 2 + 1] - 4 * CELL_SIZE);
      maxY = Math.max(maxY, trips.vertexPositions[vertex * 2 + 1] + 4 * CELL_SIZE);
    }
    // Cap the grid at about two million cells by widening cells on very large extents.
    const cellSize = Math.max(CELL_SIZE, Math.sqrt(((maxX - minX) * (maxY - minY)) / 2e6));

    // 1 when the next point belongs to the same track, so a segment row may be drawn.
    const hasNext = new Uint32Array(pointCount);
    for (let track = 0; track < trackCount; track++) {
      for (
        let vertex = trips.tripOffsets[track];
        vertex < trips.tripOffsets[track + 1] - 1;
        vertex++
      ) {
        hasNext[vertex] = 1;
      }
    }

    const originalPoints = resources.createBuffer('original-points', trips.vertexPositions);
    const noisyPoints = resources.createBuffer('noisy-points', pointCount * 8);
    const hasNextBuffer = resources.createBuffer('has-next', hasNext);
    const trackOffsets = resources.createBuffer('track-offsets', trips.tripOffsets);
    const nodePositions = resources.createBuffer('node-positions', roads.nodePositions);
    const adjacencyOffsets = resources.createBuffer('offsets', offsets);
    const edgeTargets = resources.createBuffer('edge-targets', sorted.targets);
    const roadSegments = resources.createBuffer('road-segments', roads.segments);
    const jitterParameters = resources.createParameterBuffer('jitter', 'float32', 2);
    const matchingParameters = resources.createParameterBuffer(
      'parameters',
      'float32',
      GPU_MAP_MATCHING_PARAMETER_LENGTH
    );

    const matchedEdges = resources.createBuffer('matched-edges', pointCount * 4);
    const snappedPositions = resources.createBuffer('snapped-positions', pointCount * 8);
    const breaks = resources.createBuffer('breaks', pointCount * 4);
    const matchedCount = resources.createBuffer('matched-count', 4);
    const breakCount = resources.createBuffer('break-count', 4);
    const overflow = resources.createBuffer('overflow', 4);
    const rawSegments = resources.createBuffer('raw-segments', pointCount * 16);
    const rawWeights = resources.createBuffer('raw-weights', pointCount * 4);
    const matchedSegments = resources.createBuffer('matched-segments', pointCount * 16);
    const matchedWeights = resources.createBuffer('matched-weights', pointCount * 4);

    const chainOffsets = resources.createBuffer('chain-offsets', (roadSegmentCount + 1) * 4);
    const chainPositions = resources.createBuffer('chain-positions', roadSegmentCount * 16);
    const chainCount = resources.createBuffer('chain-count', 4);
    const chainSegments = resources.createBuffer('chain-segments', roadSegmentCount * 2 * 16);
    const chainWeights = resources.createBuffer('chain-weights', roadSegmentCount * 2 * 4);
    const chainValues = resources.createBuffer('chain-values', roadSegmentCount * 2 * 4);

    let jitter = 10;
    let jitterSeed = 1;
    let sigma = 20;
    let beta = 40;
    let searchRadius = 60;
    let showRaw = true;
    let showMatched = true;
    let showRoads = true;
    let mergeChains = false;
    let dirty = true;
    let mergeEncoded = false;
    let lastMatchTime = -Infinity;
    let destroyed = false;

    function writeParameters(): void {
      jitterParameters.write(Float32Array.of(jitter, jitterSeed));
      matchingParameters.write(
        encodeGPUMapMatchingParameters({
          sigma,
          beta,
          searchRadius: Math.min(searchRadius, MAXIMUM_SEARCH_RADIUS),
          routeFactor: ROUTE_FACTOR,
          routeSlack: ROUTE_SLACK
        })
      );
      dirty = true;
    }
    writeParameters();

    const matchingGraph = new GPUCommandGraph<void>(device, {id: 'map-matching'});
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      buffer: Parameters<typeof importGraphBuffer>[2],
      format: Format,
      length: number
    ) => importGraphBuffer(matchingGraph, name, buffer, format, length);
    const originalView = view('original-points', originalPoints, 'float32x2', pointCount);
    const noisyView = view('noisy-points', noisyPoints, 'float32x2', pointCount);
    const matchedEdgesView = view('matched-edges', matchedEdges, 'uint32', pointCount);
    const snappedView = view('snapped-positions', snappedPositions, 'float32x2', pointCount);
    const breaksView = view('breaks', breaks, 'uint32', pointCount);
    const hasNextView = view('has-next', hasNextBuffer, 'uint32', pointCount);
    const jitterView = jitterParameters.importToGraph(matchingGraph);
    addKernelPass(matchingGraph, {
      id: 'map-matching-jitter',
      bindings: [
        {name: 'original', view: originalView, type: 'f32', access: 'read'},
        {name: 'jitter', view: jitterView, type: 'f32', access: 'read'},
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
    matchingGraph.add(
      new GPUMapMatching({
        id: 'matching',
        points: noisyView,
        trackOffsets: view('track-offsets', trackOffsets, 'uint32', trackCount + 1),
        nodePositions: view('node-positions', nodePositions, 'float32x2', nodeCount),
        offsets: view('offsets', adjacencyOffsets, 'uint32', nodeCount + 1),
        edgeTargets: view('edge-targets', edgeTargets, 'uint32', edgeCount),
        parameters: matchingParameters.importToGraph(matchingGraph),
        candidateCount: CANDIDATE_COUNT,
        routeNodeBudget: ROUTE_NODE_BUDGET,
        cellSize,
        bounds: {minimum: [minX, minY], maximum: [maxX, maxY]},
        output: {
          matchedEdges: matchedEdgesView,
          snappedPositions: snappedView,
          breaks: breaksView,
          matchedCount: view('matched-count', matchedCount, 'uint32', 1),
          breakCount: view('break-count', breakCount, 'uint32', 1),
          overflow: view('overflow', overflow, 'uint32', 1)
        }
      })
    );
    // Segment rows (point i to point i + 1) for drawing; hidden rows get weight 0.
    addKernelPass(matchingGraph, {
      id: 'map-matching-raw-segments',
      bindings: [
        {name: 'points', view: noisyView, type: 'f32', access: 'read'},
        {name: 'hasNext', view: hasNextView, type: 'u32', access: 'read'},
        {
          name: 'segments',
          view: view('raw-segments', rawSegments, 'float32', pointCount * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'weights',
          view: view('raw-weights', rawWeights, 'float32', pointCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: pointCount,
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
      body: `let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  weights[weightsOffset + index] = select(0.0, 1.0, hasNext[hasNextOffset + index] != 0u);`
    });
    addKernelPass(matchingGraph, {
      id: 'map-matching-matched-segments',
      bindings: [
        {name: 'points', view: snappedView, type: 'f32', access: 'read'},
        {name: 'matched', view: matchedEdgesView, type: 'u32', access: 'read'},
        {name: 'breaks', view: breaksView, type: 'u32', access: 'read'},
        {name: 'hasNext', view: hasNextView, type: 'u32', access: 'read'},
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
const NONE: u32 = 0xffffffffu;`,
      body: `let next = min(index + 1u, POINT_COUNT - 1u);
  segments[segmentsOffset + index * 4u] = points[pointsOffset + index * 2u];
  segments[segmentsOffset + index * 4u + 1u] = points[pointsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = points[pointsOffset + next * 2u];
  segments[segmentsOffset + index * 4u + 3u] = points[pointsOffset + next * 2u + 1u];
  let isDrawn = hasNext[hasNextOffset + index] != 0u && matched[matchedOffset + index] != NONE &&
    matched[matchedOffset + next] != NONE && breaks[breaksOffset + next] == 0u;
  weights[weightsOffset + index] = select(0.0, 1.0, isDrawn);`
    });
    const compiledMatching = resources.track(matchingGraph.compile());

    // Road chains: every road segment is a two-vertex line; chains join them between junctions.
    const mergeGraph = new GPUCommandGraph<void>(device, {id: 'map-matching-chains'});
    const lineOffsetValues = new Uint32Array(roadSegmentCount + 1);
    for (let line = 0; line <= roadSegmentCount; line++) lineOffsetValues[line] = line * 2;
    const lineOffsets = resources.createBuffer('line-offsets', lineOffsetValues);
    const chainOffsetsView = importGraphBuffer(
      mergeGraph,
      'chain-offsets',
      chainOffsets,
      'uint32',
      roadSegmentCount + 1
    );
    const chainCountView = importGraphBuffer(mergeGraph, 'chain-count', chainCount, 'uint32', 1);
    const chainPositionsView = importGraphBuffer(
      mergeGraph,
      'chain-positions',
      chainPositions,
      'float32x2',
      roadSegmentCount * 2
    );
    mergeGraph.add(
      new GPULineMerge({
        id: 'chains',
        positions: importGraphBuffer(
          mergeGraph,
          'road-vertices',
          roadSegments,
          'float32x2',
          roadSegmentCount * 2
        ),
        lineOffsets: importGraphBuffer(
          mergeGraph,
          'line-offsets',
          lineOffsets,
          'uint32',
          roadSegmentCount + 1
        ),
        output: {
          chainOffsets: chainOffsetsView,
          positions: chainPositionsView,
          count: chainCountView
        }
      })
    );
    addKernelPass(mergeGraph, {
      id: 'map-matching-chain-segments',
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
            roadSegmentCount * 8
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
            roadSegmentCount * 2
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
            roadSegmentCount * 2
          ),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: roadSegmentCount * 2,
      declarations: `const VERTEX_CAPACITY: u32 = ${roadSegmentCount * 2}u;`,
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
    const compiledMerge = resources.track(mergeGraph.compile());

    const summaryReader = new SummaryReader(
      resources,
      'map-matching',
      [
        {buffer: matchedCount, size: 4},
        {buffer: breakCount, size: 4},
        {buffer: overflow, size: 4},
        {buffer: chainCount, size: 4}
      ],
      bytes => {
        if (destroyed) return;
        const [matched, breakTotal, overflowFlag, chains] = new Uint32Array(bytes, 0, 4);
        matchedReadout.setValue(
          `${formatCount(matched)} of ${formatCount(pointCount)} (${((100 * matched) / Math.max(pointCount, 1)).toFixed(1)}%)`
        );
        breakReadout.setValue(formatCount(breakTotal));
        overflowReadout.setValue(overflowFlag ? 'edge grid overflowed' : 'none');
        chainReadout.setValue(
          mergeEncoded
            ? `${formatCount(roadSegmentCount)} segments into ${formatCount(chains)} chains`
            : '...'
        );
      }
    );

    context.controls.addSlider({
      label: 'GPS jitter sigma (m, noise added to trips)',
      min: 0,
      max: 40,
      step: 1,
      value: jitter,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        jitter = value;
        writeParameters();
      }
    });
    context.controls.addButton({
      label: 'Re-roll jitter noise',
      onClick: () => {
        jitterSeed++;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Emission sigma (m)',
      min: 2,
      max: 60,
      step: 1,
      value: sigma,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        sigma = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Transition beta (m)',
      min: 5,
      max: 200,
      step: 5,
      value: beta,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        beta = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: `Candidate search radius (m, at most ${MAXIMUM_SEARCH_RADIUS})`,
      min: 10,
      max: MAXIMUM_SEARCH_RADIUS,
      step: 5,
      value: searchRadius,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        searchRadius = value;
        writeParameters();
      }
    });
    context.controls.addToggle({
      label: 'Show raw (jittered) traces',
      value: showRaw,
      onChange: value => {
        showRaw = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show matched traces',
      value: showMatched,
      onChange: value => {
        showMatched = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show road network',
      value: showRoads,
      onChange: value => {
        showRoads = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Merge road chains (GPULineMerge)',
      value: mergeChains,
      onChange: value => {
        mergeChains = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Traces',
      entries: [
        {color: [255, 140, 60], label: 'Raw GPS (with jitter)'},
        {color: [60, 220, 255], label: 'Matched to road (snapped points)'}
      ]
    });
    context.controls.addReadout(
      'Tracks / points',
      `${formatCount(trackCount)} / ${formatCount(pointCount)}`
    );
    const matchedReadout = context.controls.addReadout('Matched points', '...');
    const breakReadout = context.controls.addReadout('Breaks and unmatched points', '...');
    const overflowReadout = context.controls.addReadout('Edge grid overflow', '...');
    const chainReadout = context.controls.addReadout('Road chains', '...');
    context.controls.addReadout(
      'Network',
      `${formatCount(nodeCount)} nodes, ${formatCount(edgeCount)} directed edges`
    );
    context.controls.addNote(
      'Matched lines join snapped positions of consecutive matched points and are hidden across breaks. Candidates come from a GPU edge grid; route distances use a bounded per-source Dijkstra with a fixed node budget (an approximation in dense networks).'
    );
    context.controls.addReadout('Data', `${trips.attribution}; ${roads.attribution}`);

    const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () =>
        [compiledMatching, compiledMerge] as CompiledGPUCommandGraph<never>[],
      encode(commandEncoder, frame) {
        if (!mergeEncoded) {
          compiledMerge.encode(commandEncoder, {parameters: undefined});
          mergeEncoded = true;
          summaryReader.markStale();
        }
        if (dirty && frame.timeSeconds - lastMatchTime >= REMATCH_INTERVAL_SECONDS) {
          lastMatchTime = frame.timeSeconds;
          compiledMatching.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          summaryReader.markStale();
        }
        summaryReader.flush(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [];
        if (showRoads) {
          layers.push(
            mergeChains
              ? new SpatialAnalysisSegmentLayer({
                  id: 'map-matching-chains',
                  coordinateOrigin,
                  segments: chainSegments,
                  instanceCount: roadSegmentCount * 2,
                  weights: chainWeights,
                  values: chainValues,
                  valueFormat: 'uint32',
                  colormap: 'category',
                  widthPixels: 2.5
                })
              : new SpatialAnalysisSegmentLayer({
                  id: 'map-matching-roads',
                  coordinateOrigin,
                  segments: roadSegments,
                  instanceCount: roadSegmentCount,
                  color: [120, 135, 165, 150],
                  widthPixels: 1.2
                })
          );
        }
        if (showRaw) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'map-matching-raw',
              coordinateOrigin,
              segments: rawSegments,
              instanceCount: pointCount,
              weights: rawWeights,
              color: [255, 140, 60, 150],
              widthPixels: 1.5
            })
          );
        }
        if (showMatched) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'map-matching-matched',
              coordinateOrigin,
              segments: matchedSegments,
              instanceCount: pointCount,
              weights: matchedWeights,
              color: [60, 220, 255, 235],
              widthPixels: 2.5
            })
          );
        }
        return layers;
      },
      destroy: () => {
        destroyed = true;
        summaryReader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
