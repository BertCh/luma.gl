// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Encounters between trips, and how alike the encountering trips are. Encounters need every trip on
 * one clock, which raw taxi trips do not share (each has its own pickup and drop-off time).
 * `addClockEncounters` resamples every trip onto one shared clock (`GPUTrajectoryResample` with
 * `spacing: 'clock'`: a trip is absent, NaN, outside its own span) and feeds
 * `GPUTrajectoryEncounters`, which lists the pairs within the distance in the same time bucket.
 * The distance and the clock step are per-frame buffer writes. `GPUTrackSimilarity` then scores each
 * encounter pair (Hausdorff) over the full trips, and a second instance scores the clicked trip
 * against every trip to color the whole trip layer. Two small kernels written with the explorer's
 * kernel helper gather connector segments and vehicle positions straight from the dense sample
 * table.
 */

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_TRACK_SIMILARITY_STATUS,
  GPUTrackSimilarity
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {addClockEncounters} from '@luma.gl/experimental/gpu-spatial-analysis';
import {getGPUTrajectoryClockParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

/** Time buckets of the common clock. */
const BUCKET_COUNT = 192;
/** Compile-time lattice cell size in meters; the distance slider may not exceed it. */
const CELL_SIZE = 150;
const HIT_CAPACITY = 1 << 20;
const PAIR_CAPACITY = 16384;
const MAXIMUM_FRECHET_VERTICES = 256;
const SIMILARITY_RANGE_METERS = 2500;
const NO_TRACK = 0xffffffff;
const SIMILARITY_COLORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
] as const;

type Similarity = 'hausdorff' | 'frechet';
type ConnectorTime = 'first' | 'bucket';

/** Central 98 percent of a strided coordinate sample, for the lattice bounds. */
function getCentralRange(values: Float32Array, offset: number): [number, number] {
  const sample: number[] = [];
  for (let index = offset; index < values.length; index += 2 * 3) sample.push(values[index]);
  sample.sort((a, b) => a - b);
  return [sample[Math.floor(sample.length * 0.01)], sample[Math.floor(sample.length * 0.99)]];
}

function formatClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/** Encounter pairs of taxi trips and trip-to-trip similarity. */
export const encountersMode: SpatialAnalysisModeDefinition = {
  id: 'encounters',
  title: 'Encounters',
  contributors: ['GPUTrajectoryEncounters', 'GPUTrackSimilarity', 'GPUTrajectoryResample'],
  description:
    'Pairs of taxi trips within a distance of each other in the same time bucket, drawn as ' +
    'connectors. Drag the distance, clock step and time sliders (all buffer writes); click a ' +
    'trip to color every trip by its Hausdorff or Frechet distance to it.',
  initialViewState: {longitude: -73.985, latitude: 40.74, zoom: 12},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new SpatialAnalysisResources(device, 'encounters');

    const trackCount = trips.vendors.length;
    const rowCount = trips.vertexTimestamps.length;
    const epoch = trips.timeRange[0];
    const duration = Math.max(1, trips.timeRange[1] - epoch);
    const rebasedTimestamps = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      rebasedTimestamps[row] = trips.vertexTimestamps[row] - epoch;
    }
    const [minX, maxX] = getCentralRange(trips.vertexPositions, 0);
    const [minY, maxY] = getCentralRange(trips.vertexPositions, 1);
    const bounds: [number, number, number, number] = [
      minX - CELL_SIZE,
      minY - CELL_SIZE,
      maxX + CELL_SIZE,
      maxY + CELL_SIZE
    ];
    const sampleCount = trackCount * BUCKET_COUNT;

    // Static segments of every trip (consecutive vertices of one trip) with the trip index.
    const tripSegments: number[] = [];
    const tripSegmentTracks: number[] = [];
    let longestTrip = 0;
    for (let track = 0; track < trackCount; track++) {
      const first = trips.tripOffsets[track];
      const last = trips.tripOffsets[track + 1] - 1;
      longestTrip = Math.max(longestTrip, last - first + 1);
      for (let row = first; row < last; row++) {
        tripSegments.push(
          trips.vertexPositions[row * 2],
          trips.vertexPositions[row * 2 + 1],
          trips.vertexPositions[row * 2 + 2],
          trips.vertexPositions[row * 2 + 3]
        );
        tripSegmentTracks.push(track);
      }
    }
    const tripSegmentCount = tripSegmentTracks.length;

    const defaultClockStep = duration / (BUCKET_COUNT - 1);
    let clockStep = Math.round(defaultClockStep);
    let distance = 50;
    let similarity: Similarity = 'hausdorff';
    let connectorTime: ConnectorTime = 'bucket';
    let bucket = Math.floor(BUCKET_COUNT / 3);
    let showVehicles = true;
    let selectedTrack = NO_TRACK;
    let encountersDirty = true;
    let viewDirty = true;
    let selectionDirty = true;
    let destroyed = false;

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const offsetsBuffer = resources.createBuffer('offsets', trips.tripOffsets);
    const timestampsBuffer = resources.createBuffer('timestamps', rebasedTimestamps);
    const samplesBuffer = resources.createBuffer('samples', sampleCount * 8);
    const tripSegmentsBuffer = resources.createBuffer(
      'trip-segments',
      Float32Array.from(tripSegments)
    );
    const tripSegmentTracksBuffer = resources.createBuffer(
      'trip-segment-tracks',
      Uint32Array.from(tripSegmentTracks)
    );
    const selectedSegmentsBuffer = resources.createBuffer('selected-segments', longestTrip * 16);
    const pairIds = resources.createBuffer('pair-ids', PAIR_CAPACITY * 4);
    const pairPartners = resources.createBuffer('pair-partners', PAIR_CAPACITY * 4);
    const pairFirstBuckets = resources.createBuffer('pair-first-buckets', PAIR_CAPACITY * 4);
    const pairMinimumDistances = resources.createBuffer(
      'pair-minimum-distances',
      PAIR_CAPACITY * 4
    );
    const pairBucketCounts = resources.createBuffer('pair-bucket-counts', PAIR_CAPACITY * 4);
    const pairCount = resources.createBuffer('pair-count', 4);
    const pairOverflow = resources.createBuffer('pair-overflow', 4);
    const pairHausdorff = resources.createBuffer('pair-hausdorff', PAIR_CAPACITY * 4);
    const pairFrechet = resources.createBuffer('pair-frechet', PAIR_CAPACITY * 4);
    const pairStatus = resources.createBuffer('pair-status', PAIR_CAPACITY * 4);
    const connectors = resources.createBuffer('connectors', PAIR_CAPACITY * 16);
    const vehicles = resources.createBuffer('vehicles', trackCount * 8);
    const selectedPairA = resources.createBuffer('selected-pair-a', trackCount * 4);
    const selectedPairB = resources.createBuffer('selected-pair-b', trackCount * 4);
    const selectedHausdorff = resources.createBuffer('selected-hausdorff', trackCount * 4);
    const selectedFrechet = resources.createBuffer('selected-frechet', trackCount * 4);
    const selectedStatus = resources.createBuffer('selected-status', trackCount * 4);
    const distanceParameter = resources.createParameterBuffer(
      'distance',
      'float32',
      1,
      Float32Array.of(distance)
    );
    const clockParameter = resources.createParameterBuffer(
      'clock',
      'float32',
      4,
      getGPUTrajectoryClockParameterValues({start: 0, step: clockStep})
    );
    const viewParameters = resources.createParameterBuffer('view', 'float32', 4);
    const selectionParameter = resources.createParameterBuffer(
      'selection',
      'uint32',
      1,
      Uint32Array.of(0)
    );

    // --- Graphs --------------------------------------------------------------------------------
    const encountersGraph = new GPUCommandGraph<void>(device, {id: 'encounters-pairs'});
    const pairView = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: typeof pairIds,
      format: Format,
      length = PAIR_CAPACITY
    ) => importGraphBuffer(encountersGraph, name, buffer, format, length);
    const trackPositions = importGraphBuffer(
      encountersGraph,
      'positions',
      positionsBuffer,
      'float32x2',
      rowCount
    );
    const trackOffsets = importGraphBuffer(
      encountersGraph,
      'offsets',
      offsetsBuffer,
      'uint32',
      trackCount + 1
    );
    const pairIdsView = pairView('pair-ids', pairIds, 'uint32');
    const pairPartnersView = pairView('pair-partners', pairPartners, 'uint32');
    const pairCountView = pairView('pair-count', pairCount, 'uint32', 1);
    addClockEncounters(encountersGraph, {
      id: 'encounters',
      positions: trackPositions,
      timestamps: importGraphBuffer(
        encountersGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        rowCount
      ),
      trackOffsets,
      clock: clockParameter.importToGraph(encountersGraph),
      bucketCount: BUCKET_COUNT,
      samples: importGraphBuffer(
        encountersGraph,
        'samples',
        samplesBuffer,
        'float32x2',
        sampleCount
      ),
      distance: distanceParameter.importToGraph(encountersGraph),
      cellSize: CELL_SIZE,
      bounds,
      hitCapacity: HIT_CAPACITY,
      pairs: {
        output: {
          ids: pairIdsView,
          count: pairCountView,
          overflow: pairView('pair-overflow', pairOverflow, 'uint32', 1)
        },
        partners: pairPartnersView,
        firstBuckets: pairView('pair-first-buckets', pairFirstBuckets, 'uint32'),
        minimumDistances: pairView('pair-minimum-distances', pairMinimumDistances, 'float32'),
        bucketCounts: pairView('pair-bucket-counts', pairBucketCounts, 'uint32')
      }
    });
    encountersGraph.add(
      new GPUTrackSimilarity({
        id: 'encounters-pair-similarity',
        positionsA: trackPositions,
        offsetsA: trackOffsets,
        pairA: pairIdsView,
        pairB: pairPartnersView,
        activePairCount: pairCountView,
        hausdorff: pairView('pair-hausdorff', pairHausdorff, 'float32'),
        frechet: pairView('pair-frechet', pairFrechet, 'float32'),
        status: pairView('pair-status', pairStatus, 'uint32'),
        maxFrechetVertices: MAXIMUM_FRECHET_VERTICES
      })
    );
    const encountersCompiled = resources.track(encountersGraph.compile());

    // Connector segments and vehicle positions gathered from the dense table.
    const viewGraph = new GPUCommandGraph<void>(device, {id: 'encounters-view'});
    const viewView = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: typeof pairIds,
      format: Format,
      length: number
    ) => importGraphBuffer(viewGraph, name, buffer, format, length);
    const samplesFloats = viewView('samples', samplesBuffer, 'float32', sampleCount * 2);
    const viewParameterView = viewParameters.importToGraph(viewGraph);
    addKernelPass(viewGraph, {
      id: 'encounters-connectors',
      invocationCount: PAIR_CAPACITY,
      declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {
          name: 'pairIds',
          view: viewView('pair-ids', pairIds, 'uint32', PAIR_CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'pairPartners',
          view: viewView('pair-partners', pairPartners, 'uint32', PAIR_CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'firstBuckets',
          view: viewView('pair-first-buckets', pairFirstBuckets, 'uint32', PAIR_CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'pairCount',
          view: viewView('pair-count', pairCount, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {name: 'samples', view: samplesFloats, type: 'f32', access: 'read'},
        {name: 'viewParameters', view: viewParameterView, type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: viewView('connectors', connectors, 'float32', PAIR_CAPACITY * 4),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var valid = index < pairCount[pairCountOffset];
  var segment = vec4<f32>(nan);
  if (valid) {
    let useBucket = viewParameters[viewParametersOffset] > 0.5;
    let bucket = select(firstBuckets[firstBucketsOffset + index], u32(viewParameters[viewParametersOffset + 1u]), useBucket);
    let a = (pairIds[pairIdsOffset + index] * BUCKETS + bucket) * 2u;
    let b = (pairPartners[pairPartnersOffset + index] * BUCKETS + bucket) * 2u;
    let pa = vec2<f32>(samples[samplesOffset + a], samples[samplesOffset + a + 1u]);
    let pb = vec2<f32>(samples[samplesOffset + b], samples[samplesOffset + b + 1u]);
    // At a chosen time, only pairs that are within the distance at that bucket are connected.
    if (!useBucket || distance(pa, pb) <= viewParameters[viewParametersOffset + 2u] + 1e-3) {
      segment = vec4<f32>(pa, pb);
    }
  }
  segments[segmentsOffset + index * 4u] = segment.x;
  segments[segmentsOffset + index * 4u + 1u] = segment.y;
  segments[segmentsOffset + index * 4u + 2u] = segment.z;
  segments[segmentsOffset + index * 4u + 3u] = segment.w;`
    });
    addKernelPass(viewGraph, {
      id: 'encounters-vehicles',
      invocationCount: trackCount,
      declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {name: 'samples', view: samplesFloats, type: 'f32', access: 'read'},
        {name: 'viewParameters', view: viewParameterView, type: 'f32', access: 'read'},
        {
          name: 'vehicles',
          view: viewView('vehicles', vehicles, 'float32', trackCount * 2),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let bucket = u32(viewParameters[viewParametersOffset + 1u]);
  let source = (index * BUCKETS + bucket) * 2u;
  vehicles[vehiclesOffset + index * 2u] = samples[samplesOffset + source];
  vehicles[vehiclesOffset + index * 2u + 1u] = samples[samplesOffset + source + 1u];`
    });
    const viewCompiled = resources.track(viewGraph.compile());

    // The clicked trip against every trip.
    const selectionGraph = new GPUCommandGraph<void>(device, {id: 'encounters-selection'});
    const selectionView = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: typeof pairIds,
      format: Format
    ) => importGraphBuffer(selectionGraph, name, buffer, format, trackCount);
    const selectedPairAView = selectionView('selected-pair-a', selectedPairA, 'uint32');
    const selectedPairBView = selectionView('selected-pair-b', selectedPairB, 'uint32');
    addKernelPass(selectionGraph, {
      id: 'encounters-selection-pairs',
      invocationCount: trackCount,
      bindings: [
        {
          name: 'selection',
          view: selectionParameter.importToGraph(selectionGraph),
          type: 'u32',
          access: 'read'
        },
        {name: 'pairA', view: selectedPairAView, type: 'u32', access: 'read_write'},
        {name: 'pairB', view: selectedPairBView, type: 'u32', access: 'read_write'}
      ],
      body: `pairA[pairAOffset + index] = selection[selectionOffset];
  pairB[pairBOffset + index] = index;`
    });
    selectionGraph.add(
      new GPUTrackSimilarity({
        id: 'encounters-selected-similarity',
        positionsA: importGraphBuffer(
          selectionGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          rowCount
        ),
        offsetsA: importGraphBuffer(
          selectionGraph,
          'offsets',
          offsetsBuffer,
          'uint32',
          trackCount + 1
        ),
        pairA: selectedPairAView,
        pairB: selectedPairBView,
        hausdorff: selectionView('selected-hausdorff', selectedHausdorff, 'float32'),
        frechet: selectionView('selected-frechet', selectedFrechet, 'float32'),
        status: selectionView('selected-status', selectedStatus, 'uint32'),
        maxFrechetVertices: MAXIMUM_FRECHET_VERTICES
      })
    );
    const selectionCompiled = resources.track(selectionGraph.compile());

    // --- Controls ------------------------------------------------------------------------------
    const writeView = () => {
      viewParameters.write(
        Float32Array.of(connectorTime === 'bucket' ? 1 : 0, bucket, distance, 0)
      );
      viewDirty = true;
    };
    const selectTrack = (track: number) => {
      selectedTrack = track;
      selectionParameter.write(Uint32Array.of(track));
      const first = trips.tripOffsets[track];
      const last = trips.tripOffsets[track + 1] - 1;
      const segments = new Float32Array(longestTrip * 4).fill(Number.NaN);
      for (let row = first; row < last; row++) {
        segments.set(trips.vertexPositions.subarray(row * 2, row * 2 + 4), (row - first) * 4);
      }
      selectedSegmentsBuffer.write(segments);
      selectedReadout.setValue(`Trip ${track} (${last - first + 1} vertices)`);
      selectionDirty = true;
    };

    const selectedReadout = context.controls.addReadout('Selected trip', 'none');
    context.controls.addSlider({
      label: 'Encounter distance (per-frame buffer)',
      min: 10,
      max: CELL_SIZE,
      step: 5,
      value: distance,
      format: value => `${value} m`,
      onChange: value => {
        distance = value;
        distanceParameter.write(Float32Array.of(distance));
        encountersDirty = true;
        writeView();
      }
    });
    const clockReadout = context.controls.addReadout('Common clock');
    const describeClock = () =>
      clockReadout.setValue(
        `${BUCKET_COUNT} buckets of ${clockStep} s from 0:00 to ${formatClock((BUCKET_COUNT - 1) * clockStep)} (NaN outside a trip's span)`
      );
    describeClock();
    context.controls.addSlider({
      label: 'Clock step (per-frame buffer)',
      min: Math.max(2, Math.round(defaultClockStep / 4)),
      max: Math.round(defaultClockStep * 2),
      step: 1,
      value: clockStep,
      format: value => `${value} s`,
      onChange: value => {
        clockStep = value;
        clockParameter.write(getGPUTrajectoryClockParameterValues({start: 0, step: clockStep}));
        describeClock();
        encountersDirty = true;
        writeView();
      }
    });
    context.controls.addSelect<ConnectorTime>({
      label: 'Connectors drawn at',
      options: [
        {value: 'bucket', label: 'The time bucket below'},
        {value: 'first', label: "Each pair's first encounter (many)"}
      ],
      value: connectorTime,
      onChange: value => {
        connectorTime = value;
        writeView();
      }
    });
    context.controls.addSlider({
      label: 'Time bucket',
      min: 0,
      max: BUCKET_COUNT - 1,
      step: 1,
      value: bucket,
      format: value => formatClock(value * clockStep),
      onChange: value => {
        bucket = value;
        writeView();
      }
    });
    context.controls.addToggle({
      label: 'Show vehicles at the time bucket',
      value: showVehicles,
      onChange: value => {
        showVehicles = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<Similarity>({
      label: 'Trip color: distance to the clicked trip',
      options: [
        {value: 'hausdorff', label: 'Discrete Hausdorff'},
        {value: 'frechet', label: 'Discrete Frechet'}
      ],
      value: similarity,
      onChange: value => {
        similarity = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: `Trip distance to the selected trip: 0 to ${SIMILARITY_RANGE_METERS} m (gray: not computed)`,
      gradient: {colors: SIMILARITY_COLORS, minimumLabel: 'alike', maximumLabel: 'different'}
    });
    context.controls.addNote(
      'Click near a trip to select it. Orange connectors join encountering pairs; the pair ' +
        'readout scores how alike each pair of full trips is.'
    );
    context.controls.addReadout('Trips', formatCount(trackCount));
    const pairReadout = context.controls.addReadout('Encounter pairs');
    const pairSimilarityReadout = context.controls.addReadout('Pair Hausdorff');
    const selectedSimilarityReadout = context.controls.addReadout('Within 500 m of selected');
    context.controls.addReadout('Data', trips.attribution);

    const pairSummary = new SummaryReader(
      resources,
      'encounters-pairs',
      [
        {buffer: pairCount, size: 4},
        {buffer: pairOverflow, size: 4},
        {buffer: pairHausdorff, size: PAIR_CAPACITY * 4},
        {buffer: pairStatus, size: PAIR_CAPACITY * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const count = Math.min(words[0], PAIR_CAPACITY);
        let finite = 0;
        let sum = 0;
        let together = 0;
        for (let pair = 0; pair < count; pair++) {
          const value = floats[2 + pair];
          if (Number.isFinite(value)) {
            finite++;
            sum += value;
            if (value < 500) together++;
          }
        }
        pairReadout.setValue(
          `${formatCount(words[0])} of ${formatCount(PAIR_CAPACITY)}` +
            `${words[1] ? ' OVERFLOW (hit scratch or pair capacity)' : ''}`
        );
        pairSimilarityReadout.setValue(
          finite
            ? `mean ${(sum / finite).toFixed(0)} m, ${formatCount(together)} of ${formatCount(finite)} under 500 m (traveling together)`
            : '–'
        );
      }
    );
    const selectionSummary = new SummaryReader(
      resources,
      'encounters-selection',
      [
        {buffer: selectedStatus, size: trackCount * 4},
        {buffer: selectedHausdorff, size: trackCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        let near = 0;
        let capExceeded = 0;
        for (let track = 0; track < trackCount; track++) {
          if (words[track] & GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded) capExceeded++;
          if (track !== selectedTrack && floats[trackCount + track] < 500) near++;
        }
        selectedSimilarityReadout.setValue(
          `${formatCount(near)} trips${capExceeded ? `; ${formatCount(capExceeded)} too long for Frechet` : ''}`
        );
      }
    );

    // Start with the trip nearest the middle of the lattice.
    let startTrack = 0;
    let nearest = Infinity;
    const centerX = (bounds[0] + bounds[2]) / 2;
    const centerY = (bounds[1] + bounds[3]) / 2;
    for (let row = 0, track = 0; row < rowCount; row++) {
      while (row >= trips.tripOffsets[track + 1]) track++;
      const away =
        (trips.vertexPositions[row * 2] - centerX) ** 2 +
        (trips.vertexPositions[row * 2 + 1] - centerY) ** 2;
      if (away < nearest) {
        nearest = away;
        startTrack = track;
      }
    }
    selectTrack(startTrack);
    writeView();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [encountersCompiled, viewCompiled, selectionCompiled],
      encode(commandEncoder: CommandEncoder) {
        if (encountersDirty) {
          encountersCompiled.encode(commandEncoder, {parameters: undefined});
          encountersDirty = false;
          viewDirty = true;
          pairSummary.markStale();
        }
        if (viewDirty) {
          viewCompiled.encode(commandEncoder, {parameters: undefined});
          viewDirty = false;
        }
        if (selectionDirty) {
          selectionCompiled.encode(commandEncoder, {parameters: undefined});
          selectionDirty = false;
          selectionSummary.markStale();
        }
        pairSummary.flush(commandEncoder);
        selectionSummary.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'encounters-trips',
            coordinateOrigin,
            segments: tripSegmentsBuffer,
            instanceCount: tripSegmentCount,
            widthPixels: 1.3,
            values: similarity === 'hausdorff' ? selectedHausdorff : selectedFrechet,
            valueFormat: 'float32',
            valueIndices: tripSegmentTracksBuffer,
            colormap: 'viridis',
            valueRange: [0, SIMILARITY_RANGE_METERS],
            noDataColor: [120, 125, 140, 90],
            color: [255, 255, 255, 190],
            opacity: 0.45
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'encounters-selected',
            coordinateOrigin,
            segments: selectedSegmentsBuffer,
            instanceCount: longestTrip,
            widthPixels: 4,
            color: [255, 255, 255, 255]
          })
        ];
        if (showVehicles) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'encounters-vehicles',
              coordinateOrigin,
              positions: vehicles,
              instanceCount: trackCount,
              radiusPixels: 2.2,
              color: [200, 225, 255, 200]
            })
          );
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'encounters-connectors',
            coordinateOrigin,
            segments: connectors,
            instanceCount: PAIR_CAPACITY,
            widthPixels: 6,
            color: [255, 140, 60, 255]
          })
        );
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        let nearestTrack = NO_TRACK;
        let nearestDistance = Infinity;
        for (let track = 0, row = 0; track < trackCount; track++) {
          for (; row < trips.tripOffsets[track + 1]; row++) {
            const away =
              (trips.vertexPositions[row * 2] - x) ** 2 +
              (trips.vertexPositions[row * 2 + 1] - y) ** 2;
            if (away < nearestDistance) {
              nearestDistance = away;
              nearestTrack = track;
            }
          }
        }
        if (nearestTrack === NO_TRACK) return false;
        selectTrack(nearestTrack);
        context.updateLayers();
        return true;
      },
      destroy() {
        destroyed = true;
        pairSummary.stop();
        selectionSummary.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
