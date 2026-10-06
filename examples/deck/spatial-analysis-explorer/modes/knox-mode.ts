// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Space-time interaction of New York taxi pickups. Every trip start is an event with a location
 * and a time. A radius-mode `GPUNeighborSearch` lists the pairs closer than the spatial
 * threshold; `GPUKnoxTest` counts how many of those pairs are also close in time and compares
 * the count with `P` permutations of the event times over the fixed locations, and
 * `GPUMantelTest` correlates pair distance with time difference over the same pairs.
 *
 * The spatial threshold (the search radius), the time threshold, the permutation count and the
 * seed are all buffer writes: one compiled graph is re-encoded when a slider moves and the
 * rebuild counter stays 0. The summary (about 60 words) and the two permutation distributions
 * come back through one ring-buffered readback.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  getKnoxPoissonPValue,
  GPUKnoxTest,
  GPUMantelTest,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

const MAXIMUM_PERMUTATIONS = 999;
/** Pair slots per event; a spatial threshold that lists more pairs overflows (reported). */
const SLOTS_PER_EVENT = 192;
const HISTOGRAM_BINS = 28;
const SUMMARY = GPU_SPACE_TIME_SUMMARY;

function formatNumber(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '–';
}

function formatProbability(value: number): string {
  if (!Number.isFinite(value)) return '–';
  return value < 0.001 ? '< 0.001' : value.toFixed(3);
}

/** Text sparkline of a histogram, with the bin holding `marker` flagged below it. */
function formatHistogram(values: ArrayLike<number>, count: number, marker: number): string {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < count; index++) {
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  minimum = Math.min(minimum, marker);
  maximum = Math.max(maximum, marker);
  const span = Math.max(maximum - minimum, 1e-9);
  const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
  for (let index = 0; index < count; index++) {
    bins[
      Math.min(HISTOGRAM_BINS - 1, Math.floor(((values[index] - minimum) / span) * HISTOGRAM_BINS))
    ]++;
  }
  const bars = '▁▂▃▄▅▆▇█';
  const peak = Math.max(...bins, 1);
  const spark = bins
    .map(bin => (bin === 0 ? '·' : bars[Math.min(7, Math.round((bin / peak) * 7))]))
    .join('');
  const markerBin = Math.min(
    HISTOGRAM_BINS - 1,
    Math.floor(((marker - minimum) / span) * HISTOGRAM_BINS)
  );
  return `${spark}\n${' '.repeat(markerBin)}▲ observed`;
}

export const knoxMode: SpatialAnalysisModeDefinition = {
  id: 'knox',
  title: 'Knox',
  contributors: ['GPUKnoxTest', 'GPUMantelTest', 'GPUNeighborSearch'],
  description:
    'Are pickups that are near in space also near in time? Trip starts are events; pairs ' +
    'closer than the spatial threshold are tested against permuted times. Orange lines are ' +
    'pairs close in both; move the thresholds and read observed versus expected counts.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'knox');

    // Events: the first vertex of every trip, inside the central 99% of the city (a few trips
    // carry coordinates far outside it).
    const tripCount = trips.tripOffsets.length - 1;
    const xs: number[] = [];
    const ys: number[] = [];
    const times: number[] = [];
    for (let trip = 0; trip < tripCount; trip++) {
      const vertex = trips.tripOffsets[trip];
      if (trips.tripOffsets[trip + 1] <= vertex) continue;
      xs.push(trips.vertexPositions[vertex * 2]);
      ys.push(trips.vertexPositions[vertex * 2 + 1]);
      times.push(trips.vertexTimestamps[vertex]);
    }
    const center = (values: number[]): [number, number] => {
      const sorted = Float32Array.from(values).sort();
      return [
        sorted[Math.floor(sorted.length * 0.005)],
        sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.995))]
      ];
    };
    const [minimumX, maximumX] = center(xs);
    const [minimumY, maximumY] = center(ys);
    const keptIndices = xs
      .map((_, index) => index)
      .filter(
        index =>
          xs[index] >= minimumX &&
          xs[index] <= maximumX &&
          ys[index] >= minimumY &&
          ys[index] <= maximumY
      );
    const eventCount = keptIndices.length;
    const positions = new Float32Array(eventCount * 2);
    const eventTimes = new Float32Array(eventCount);
    keptIndices.forEach((source, index) => {
      positions[index * 2] = xs[source];
      positions[index * 2 + 1] = ys[source];
      eventTimes[index] = times[source];
    });
    let minimumTime = Infinity;
    let maximumTime = -Infinity;
    for (const time of eventTimes) {
      minimumTime = Math.min(minimumTime, time);
      maximumTime = Math.max(maximumTime, time);
    }
    const slotCapacity = eventCount * SLOTS_PER_EVENT;

    let spatialThreshold = 250;
    let timeThreshold = 120;
    let permutations = 499;
    let seed = 1;
    let showSpatialOnly = false;
    let dirty = true;
    let summary: Float32Array | null = null;
    let knoxStatistics: Uint32Array | null = null;
    let overflow = 0;

    const positionsBuffer = resources.createBuffer('positions', positions);
    const timesBuffer = resources.createBuffer('times', eventTimes);
    const offsetsBuffer = resources.createBuffer('offsets', (eventCount + 1) * 4);
    const neighborsBuffer = resources.createBuffer('neighbors', slotCapacity * 4);
    const weightsBuffer = resources.createBuffer('weights', slotCapacity * 4);
    const distancesBuffer = resources.createBuffer('distances', slotCapacity * 4);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const knoxStatisticsBuffer = resources.createBuffer(
      'knox-statistics',
      (MAXIMUM_PERMUTATIONS + 1) * 4
    );
    const knoxSummaryBuffer = resources.createBuffer(
      'knox-summary',
      GPU_SPACE_TIME_SUMMARY_LENGTH * 4
    );
    const mantelStatisticsBuffer = resources.createBuffer(
      'mantel-statistics',
      (MAXIMUM_PERMUTATIONS + 1) * 4
    );
    const mantelSummaryBuffer = resources.createBuffer(
      'mantel-summary',
      GPU_SPACE_TIME_SUMMARY_LENGTH * 4
    );
    const segmentsBuffer = resources.createBuffer('segments', slotCapacity * 16);
    const segmentFadeBuffer = resources.createBuffer('segment-fade', slotCapacity * 4);

    const searchParameters = resources.createParameterBuffer(
      'search-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    const testParameters = resources.createParameterBuffer(
      'test-parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    // Draw parameters: time threshold and the alpha of pairs that are close in space only.
    const drawParameters = resources.createParameterBuffer('draw-parameters', 'float32', 2);
    const searchBounds: [number, number, number, number] = [
      minimumX - 50,
      minimumY - 50,
      maximumX + 50,
      maximumY + 50
    ];

    function writeParameters(): void {
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius: spatialThreshold,
          weightKind: 'binary'
        })
      );
      testParameters.write(getGPUSpaceTimeParameterValues({seed, permutations, timeThreshold}));
      drawParameters.write(Float32Array.of(timeThreshold, showSpatialOnly ? 0.12 : 0));
    }
    writeParameters();

    const graph = new GPUCommandGraph<void>(device, {id: 'knox'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      buffer: ReturnType<SpatialAnalysisResources['createBuffer']>,
      format: Format,
      length?: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const positionsView = view('positions', positionsBuffer, 'float32x2', eventCount);
    const timesView = view('times', timesBuffer, 'float32', eventCount);
    const pairs = {
      offsets: view('offsets', offsetsBuffer, 'uint32', eventCount + 1),
      neighbors: view('neighbors', neighborsBuffer, 'uint32', slotCapacity),
      weights: view('weights', weightsBuffer, 'float32', slotCapacity),
      distances: view('distances', distancesBuffer, 'float32', slotCapacity)
    };
    const testParametersView = testParameters.importToGraph(graph);
    graph.add(
      new GPUNeighborSearch({
        id: 'knox-search',
        mode: 'radius',
        positions: positionsView,
        parameters: searchParameters.importToGraph(graph),
        gridSize: [48, 48],
        weights: pairs,
        overflow: view('overflow', overflowBuffer, 'uint32', 1)
      })
    );
    graph.add(
      new GPUKnoxTest({
        id: 'knox-test',
        pairs,
        times: timesView,
        parameters: testParametersView,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        statistics: view(
          'knox-statistics',
          knoxStatisticsBuffer,
          'uint32',
          MAXIMUM_PERMUTATIONS + 1
        ),
        summary: view('knox-summary', knoxSummaryBuffer, 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
      })
    );
    graph.add(
      new GPUMantelTest({
        id: 'mantel-test',
        pairs,
        times: timesView,
        parameters: testParametersView,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        statistics: view(
          'mantel-statistics',
          mantelStatisticsBuffer,
          'float32',
          MAXIMUM_PERMUTATIONS + 1
        ),
        summary: view(
          'mantel-summary',
          mantelSummaryBuffer,
          'float32',
          GPU_SPACE_TIME_SUMMARY_LENGTH
        )
      })
    );
    // One thread per pair slot finds its row by binary search and writes the drawable segment of
    // each unordered pair (entries j > i); pairs close in time are fully opaque.
    addKernelPass(graph, {
      id: 'knox-segments',
      invocationCount: slotCapacity,
      declarations: `const EVENT_COUNT: u32 = ${eventCount}u;`,
      bindings: [
        {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'times', view: timesView, type: 'f32', access: 'read'},
        {
          name: 'drawParameters',
          view: drawParameters.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'segments',
          view: view('segments', segmentsBuffer, 'float32', slotCapacity * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'fade',
          view: view('segment-fade', segmentFadeBuffer, 'float32', slotCapacity),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  segments[segmentsOffset + index * 4u] = 0.0;
  segments[segmentsOffset + index * 4u + 1u] = 0.0;
  segments[segmentsOffset + index * 4u + 2u] = 0.0;
  segments[segmentsOffset + index * 4u + 3u] = 0.0;
  fade[fadeOffset + index] = 0.0;
  if (index >= offsets[offsetsOffset + EVENT_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  if (neighbor <= low) {
    return;
  }
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  let timeGap = abs(times[timesOffset + low] - times[timesOffset + neighbor]);
  fade[fadeOffset + index] = select(drawParameters[drawParametersOffset + 1u], 1.0, timeGap <= drawParameters[drawParametersOffset]);`
    });
    const compiled = resources.track(graph.compile());

    const reader = new SummaryReader(
      resources,
      'knox',
      [
        {buffer: knoxSummaryBuffer, size: GPU_SPACE_TIME_SUMMARY_LENGTH * 4},
        {buffer: mantelSummaryBuffer, size: GPU_SPACE_TIME_SUMMARY_LENGTH * 4},
        {buffer: overflowBuffer, size: 4},
        {buffer: knoxStatisticsBuffer, size: (MAXIMUM_PERMUTATIONS + 1) * 4}
      ],
      bytes => {
        const words = GPU_SPACE_TIME_SUMMARY_LENGTH;
        summary = new Float32Array(bytes.slice(0, words * 8));
        overflow = new Uint32Array(bytes.slice(words * 8, words * 8 + 4))[0];
        knoxStatistics = new Uint32Array(bytes.slice(words * 8 + 4));
        showSummary();
      }
    );

    // Controls.
    context.controls.addSlider({
      label: 'Spatial threshold (neighbor-search radius)',
      min: 50,
      max: 600,
      step: 25,
      value: spatialThreshold,
      format: value => `${value} m`,
      onChange: value => {
        spatialThreshold = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Time threshold',
      min: 10,
      max: 900,
      step: 10,
      value: timeThreshold,
      format: value => `${value} s`,
      onChange: value => {
        timeThreshold = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Permutations',
      min: 19,
      max: MAXIMUM_PERMUTATIONS,
      step: 10,
      value: permutations,
      onChange: value => {
        permutations = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addButton({
      label: 'New permutation seed',
      onClick: () => {
        seed++;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Show pairs close in space only',
      value: showSpatialOnly,
      onChange: value => {
        showSpatialOnly = value;
        writeParameters();
        dirty = true;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Event start time (points)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: `${Math.round(minimumTime)} s`,
        maximumLabel: `${Math.round(maximumTime)} s`
      }
    });
    context.controls.addReadout('Events', formatCount(eventCount));
    const pairsReadout = context.controls.addReadout('Spatial pairs');
    const timeCloseReadout = context.controls.addReadout('Time-close pairs (all)');
    const observedReadout = context.controls.addReadout('Knox observed');
    const expectedReadout = context.controls.addReadout('Knox expected');
    const permutedReadout = context.controls.addReadout('Permutation mean ± sd');
    const pValueReadout = context.controls.addReadout('Pseudo p (X ≥ obs)');
    const poissonReadout = context.controls.addReadout('Poisson p (classic)');
    const mantelReadout = context.controls.addReadout('Mantel r (observed)');
    const mantelPermutedReadout = context.controls.addReadout('Mantel perm. mean ± sd');
    const mantelPReadout = context.controls.addReadout('Mantel pseudo p');
    const overflowReadout = context.controls.addReadout('Pair overflow');
    context.controls.addReadout('Data', trips.attribution);
    // The panel has no preformatted block, so the histogram goes straight into the readout area.
    const histogramBlock = document.createElement('pre');
    histogramBlock.style.cssText =
      'margin:8px 0 0;color:#a9b8d0;font:10px/1.2 ui-monospace,monospace';
    document.querySelector('[data-mode-readouts]')?.appendChild(histogramBlock);

    function showSummary(): void {
      if (!summary) return;
      const words = GPU_SPACE_TIME_SUMMARY_LENGTH;
      const knox = summary.subarray(0, words);
      const mantel = summary.subarray(words, words * 2);
      pairsReadout.setValue(formatCount(knox[SUMMARY.pairCount]));
      timeCloseReadout.setValue(formatCount(knox[SUMMARY.timeClosePairs]));
      observedReadout.setValue(formatCount(knox[SUMMARY.observed]));
      expectedReadout.setValue(formatNumber(knox[SUMMARY.expected]));
      permutedReadout.setValue(
        `${formatNumber(knox[SUMMARY.permutationMean])} ± ${formatNumber(Math.sqrt(knox[SUMMARY.permutationVariance]))}`
      );
      pValueReadout.setValue(formatProbability(knox[SUMMARY.pseudoPGreater]));
      poissonReadout.setValue(
        formatProbability(getKnoxPoissonPValue(knox[SUMMARY.observed], knox[SUMMARY.expected]))
      );
      mantelReadout.setValue(formatNumber(mantel[SUMMARY.observed], 4));
      mantelPermutedReadout.setValue(
        `${formatNumber(mantel[SUMMARY.permutationMean], 4)} ± ${formatNumber(Math.sqrt(mantel[SUMMARY.permutationVariance]), 4)}`
      );
      mantelPReadout.setValue(formatProbability(mantel[SUMMARY.pseudoPGreater]));
      overflowReadout.setValue(overflow ? 'yes (pairs truncated; lower the radius)' : 'no');
      if (knoxStatistics) {
        histogramBlock.textContent =
          `Knox count under permuted times (${permutations} permutations):\n` +
          formatHistogram(
            Array.from(knoxStatistics.subarray(1, permutations + 1)),
            permutations,
            knox[SUMMARY.observed]
          );
      }
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.request(commandEncoder);
        }
        reader.flush(commandEncoder);
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        return [
          new SpatialAnalysisSegmentLayer({
            id: 'knox-pairs',
            coordinateOrigin,
            segments: segmentsBuffer,
            weights: segmentFadeBuffer,
            instanceCount: slotCapacity,
            widthPixels: 1.3,
            colormap: 'uniform',
            color: [255, 160, 50, 200]
          }),
          new SpatialAnalysisPointLayer({
            id: 'knox-events',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: eventCount,
            radiusPixels: 3,
            values: timesBuffer,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [minimumTime, maximumTime],
            color: [255, 255, 255, 235]
          })
        ];
      },
      destroy() {
        reader.stop();
        histogramBlock.remove();
        resources.destroy();
      }
    };
    return instance;
  }
};
