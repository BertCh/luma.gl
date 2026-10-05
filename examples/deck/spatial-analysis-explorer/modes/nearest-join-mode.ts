// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Nearest road: `GPUNearestFeatureJoin` snaps every New York point of interest to its nearest road
 * segment within a per-frame radius. A mode-local layer draws a connector from each POI to the
 * closest point on its matched segment straight from the join output, POIs are colored by snap
 * distance, unmatched POIs are red, and segments that received at least one POI are highlighted
 * from `featureCounts`. Only a small summary (matched count, overflow) is read back.
 *
 * `spatialSort` is a compile-time option (the toggle rebuilds the graph). "Shuffle road rows" applies
 * one seeded permutation to every per-segment buffer so the road table becomes spatially incoherent
 * without changing the picture; it is a plain buffer rewrite. "Measure spatialSort on vs off" times
 * the active graph and a temporary graph of the other setting outside the frame.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNearestFeatureJoin} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {createSeededRandom} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {formatCompiledGraphTiming, formatSpeedup, measureCompiledGraph} from './vector-timing';
import {NearestSnapLayer} from './nearest-join-layers';

const MAXIMUM_RADIUS = 300;
const DEFAULT_RADIUS = 60;
/** Frames between summary readbacks. */
const READBACK_INTERVAL = 15;
const CANDIDATES_PER_POINT = 256;
/** Frames after creation before the automatic spatialSort measurement runs. */
const AUTO_MEASURE_FRAME = 40;
const SHUFFLE_SEED = 20240611;

export const nearestJoinMode: SpatialAnalysisModeDefinition = {
  id: 'nearest-join',
  title: 'Nearest road',
  contributors: ['GPUNearestFeatureJoin'],
  description:
    'Each point of interest snaps to its nearest road segment within a radius, computed on the GPU. ' +
    'Drag the radius slider: the join re-runs every frame without recompiling.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 15},

  async create(context) {
    const [pois, roads] = await Promise.all([
      context.data.getNewYorkPointsOfInterest(),
      context.data.getNewYorkRoads()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'nearest-join');
    const pointCount = pois.positions.length / 2;
    const segmentCount = roads.segments.length / 4;

    // The contributor takes packed starts and ends; the float32x4 rows are kept for the layers.
    const starts = new Float32Array(segmentCount * 2);
    const ends = new Float32Array(segmentCount * 2);
    for (let row = 0; row < segmentCount; row++) {
      starts.set(roads.segments.subarray(row * 4, row * 4 + 2), row * 2);
      ends.set(roads.segments.subarray(row * 4 + 2, row * 4 + 4), row * 2);
    }
    const positionsBuffer = resources.createBuffer('positions', pois.positions);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const startsBuffer = resources.createBuffer('starts', starts);
    const endsBuffer = resources.createBuffer('ends', ends);
    const radius = resources.createParameterBuffer(
      'radius',
      'float32',
      1,
      Float32Array.of(DEFAULT_RADIUS)
    );
    const nearestFeatureIds = resources.createBuffer('nearest-feature-ids', pointCount * 4);
    const nearestDistances = resources.createBuffer('nearest-distances', pointCount * 4);
    const featureCounts = resources.createBuffer('feature-counts', segmentCount * 4);
    const overflow = resources.createBuffer('overflow', 4);
    const candidateCount = resources.createBuffer('candidate-count', 4);
    const matchIds = resources.createBuffer('match-ids', pointCount * 4);
    const matchCount = resources.createBuffer('match-count', 4);
    const matchOverflow = resources.createBuffer('match-overflow', 4);

    const buildGraph = (spatialSort: boolean): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `nearest-join-${spatialSort ? 'sorted' : 'unsorted'}`
      });
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'nearest-join',
          points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
          features: {
            kind: 'segments',
            starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', segmentCount),
            ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', segmentCount)
          },
          radius: radius.importToGraph(graph),
          // Sized for the maximum radius; the candidate readout shows the real demand.
          candidateCapacity: pointCount * CANDIDATES_PER_POINT,
          spatialSort,
          nearestFeatureIds: importGraphBuffer(
            graph,
            'nearest-feature-ids',
            nearestFeatureIds,
            'uint32',
            pointCount
          ),
          nearestDistances: importGraphBuffer(
            graph,
            'nearest-distances',
            nearestDistances,
            'float32',
            pointCount
          ),
          featureCounts: importGraphBuffer(
            graph,
            'feature-counts',
            featureCounts,
            'uint32',
            segmentCount
          ),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1),
          matches: {
            ids: importGraphBuffer(graph, 'match-ids', matchIds, 'uint32', pointCount),
            count: importGraphBuffer(graph, 'match-count', matchCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'match-overflow', matchOverflow, 'uint32', 1)
          }
        })
      );
      return graph.compile();
    };
    let spatialSort = false;
    let compiled = resources.track(buildGraph(spatialSort));

    const readbackRing = new GPUReadbackRing(device, {id: 'nearest-join-readback', byteLength: 16});
    resources.track({destroy: () => readbackRing.destroy()});

    let currentRadius = DEFAULT_RADIUS;
    let showConnectors = true;
    let destroyed = false;
    let readbackPending = false;
    let measuring = false;
    let measureAgain = false;
    let encodedFrames = 0;
    let autoMeasured = false;

    // One deterministic Fisher-Yates permutation; shuffled row r holds original row permutation[r].
    // starts, ends and the float32x4 segments are permuted together, so the nearestFeatureIds and
    // featureCounts the join writes keep indexing matching rows in every layer.
    let shuffledRows: {segments: Float32Array; starts: Float32Array; ends: Float32Array} | null =
      null;
    const getShuffledRows = () => {
      if (shuffledRows) return shuffledRows;
      const random = createSeededRandom(SHUFFLE_SEED);
      const permutation = Uint32Array.from({length: segmentCount}, (_, row) => row);
      for (let row = segmentCount - 1; row > 0; row--) {
        const other = Math.floor(random() * (row + 1));
        [permutation[row], permutation[other]] = [permutation[other], permutation[row]];
      }
      const segments = new Float32Array(segmentCount * 4);
      const shuffledStarts = new Float32Array(segmentCount * 2);
      const shuffledEnds = new Float32Array(segmentCount * 2);
      for (let row = 0; row < segmentCount; row++) {
        const source = permutation[row];
        segments.set(roads.segments.subarray(source * 4, source * 4 + 4), row * 4);
        shuffledStarts.set(starts.subarray(source * 2, source * 2 + 2), row * 2);
        shuffledEnds.set(ends.subarray(source * 2, source * 2 + 2), row * 2);
      }
      shuffledRows = {segments, starts: shuffledStarts, ends: shuffledEnds};
      return shuffledRows;
    };

    context.controls.addSlider({
      label: 'Search radius (per-frame parameter)',
      min: 10,
      max: MAXIMUM_RADIUS,
      step: 5,
      value: currentRadius,
      format: value => `${value} m`,
      onChange: value => {
        currentRadius = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'spatialSort (compile-time: rebuilds graph)',
      value: spatialSort,
      onChange: value => {
        spatialSort = value;
        const previous = compiled;
        compiled = resources.track(buildGraph(spatialSort));
        // The old graph is freed once no frame can encode it any more.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) resources.release(previous);
          })
        );
        if (measuring) measureAgain = true;
      }
    });
    context.controls.addToggle({
      label: 'Shuffle road rows (per-frame buffer rewrite, no recompile)',
      value: false,
      onChange: value => {
        const rows = value ? getShuffledRows() : {segments: roads.segments, starts, ends};
        segmentsBuffer.write(rows.segments);
        startsBuffer.write(rows.starts);
        endsBuffer.write(rows.ends);
        if (measuring) measureAgain = true;
        else void measureSpatialSort();
      }
    });
    context.controls.addButton({
      label: 'Measure spatialSort on vs off',
      onClick: () => void measureSpatialSort()
    });
    context.controls.addToggle({
      label: 'Show snap connectors',
      value: showConnectors,
      onChange: value => {
        showConnectors = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'POI color: distance to nearest road',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0 m',
        maximumLabel: 'radius'
      }
    });
    context.controls.addLegend({
      title: 'Overlays',
      entries: [
        {color: [255, 255, 255], label: 'Snap connector to nearest road point'},
        {color: [255, 170, 60], label: 'Road that received a POI'},
        {color: [255, 70, 70], label: 'POI with no road in radius'}
      ]
    });
    context.controls.addReadout('Points of interest', formatCount(pointCount));
    context.controls.addReadout('Road segments', formatCount(segmentCount));
    const matchedReadout = context.controls.addReadout('Matched', '...');
    const overflowReadout = context.controls.addReadout('Overflow', '...');
    const candidateReadout = context.controls.addReadout('BVH candidates', '...');
    const sortedReadout = context.controls.addReadout('spatialSort on', '...');
    const unsortedReadout = context.controls.addReadout('spatialSort off', '...');
    const speedupReadout = context.controls.addReadout('Speedup', '...');
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise wall clock / 8 repetitions (upper bound). ' +
        'Sorting helps most when road rows are spatially incoherent; try shuffling the rows.'
    );
    context.controls.addReadout('Data', `${pois.attribution}; ${roads.attribution}`);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      [matchCount, overflow, candidateCount].forEach((sourceBuffer, index) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: index * 4,
          size: 4
        });
      });
      ticket.markEncoded({byteLength: 12});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 3);
        matchedReadout.setValue(`${formatCount(words[0])} of ${formatCount(pointCount)}`);
        overflowReadout.setValue(words[1] ? 'YES' : 'no');
        candidateReadout.setValue(formatCount(words[2]));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    /** Times the active graph and a temporary graph of the other spatialSort setting. */
    const measureSpatialSort = async () => {
      if (measuring || destroyed) return;
      measuring = true;
      measureAgain = false;
      sortedReadout.setValue('measuring...');
      unsortedReadout.setValue('measuring...');
      speedupReadout.setValue('...');
      let temporary: CompiledGPUCommandGraph<void> | null = null;
      try {
        const activeSort = spatialSort;
        const active = compiled;
        temporary = buildGraph(!activeSort);
        const options = {parameters: undefined, completionBuffer: overflow, signal: context.signal};
        const activeTiming = await measureCompiledGraph(device, active, options);
        const otherTiming = await measureCompiledGraph(device, temporary, options);
        if (destroyed) return;
        const sorted = activeSort ? activeTiming : otherTiming;
        const unsorted = activeSort ? otherTiming : activeTiming;
        sortedReadout.setValue(formatCompiledGraphTiming(sorted));
        unsortedReadout.setValue(formatCompiledGraphTiming(unsorted));
        speedupReadout.setValue(formatSpeedup(unsorted.milliseconds, sorted.milliseconds));
      } catch {
        if (!destroyed) {
          sortedReadout.setValue('interrupted');
          unsortedReadout.setValue('interrupted');
          measureAgain = true;
        }
      } finally {
        temporary?.destroy();
        measuring = false;
        if (measureAgain && !destroyed) void measureSpatialSort();
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        encodedFrames++;
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureSpatialSort();
        }
        radius.write(Float32Array.of(currentRadius));
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 0) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'nearest-join-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            color: [150, 170, 200, 110],
            widthPixels: 1.5
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'nearest-join-used-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: featureCounts,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 170, 60, 130],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 2.5
          })
        ];
        if (showConnectors) {
          layers.push(
            new NearestSnapLayer({
              id: 'nearest-join-connectors',
              coordinateOrigin,
              positions: positionsBuffer,
              nearestFeatureIds,
              segments: segmentsBuffer,
              instanceCount: pointCount,
              color: [255, 255, 255, 235],
              widthPixels: 1.5
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'nearest-join-unmatched',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: nearestFeatureIds,
            valueFormat: 'uint32',
            colormap: 'category',
            // A single transparent palette entry hides matched points; the no-data sentinel is red.
            palette: [[0, 0, 0, 0]],
            noDataColor: [255, 70, 70, 255],
            radiusPixels: 4
          }),
          new SpatialAnalysisPointLayer({
            id: 'nearest-join-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: nearestDistances,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, currentRadius],
            // Unmatched points carry the distance -1 and are drawn by the red layer instead.
            discardAtOrBelow: -0.5,
            radiusPixels: 4
          })
        );
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
