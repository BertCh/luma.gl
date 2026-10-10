// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {TrajectoryStopLayer} from './trajectories-layers';

/** Capacity of the bounded stop list. `overflow` reports a larger result. */
const STOP_CAPACITY = 8192;
/** Frames between summary readbacks (a slider change also requests one). */
const READBACK_INTERVAL_FRAMES = 20;
const METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR = 3.6;
const DEFAULT_STOP_SPEED_THRESHOLD = 3;
const DEFAULT_STOP_MINIMUM_DURATION = 20;
/** Viridis samples for the legend gradient. */
const VIRIDIS_COLORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
] as const;

type ColorMetric = 'average' | 'maximum' | 'speed' | 'heading' | 'acceleration';

/** Fixed color ranges of each trail metric: m/s, m/s, m/s, radians, m/s squared. */
const METRIC_RANGES: Record<ColorMetric, readonly [number, number]> = {
  average: [2, 14],
  maximum: [5, 35],
  speed: [0, 20],
  heading: [-Math.PI, Math.PI],
  acceleration: [-2, 2]
};

const METRIC_UNITS: Record<ColorMetric, string> = {
  average: 'm/s',
  maximum: 'm/s',
  speed: 'm/s',
  heading: 'rad',
  acceleration: 'm/s²'
};

const METRIC_DESCRIPTIONS: Record<ColorMetric, string> = {
  average: 'average speed of each trip (per-track column)',
  maximum: 'maximum step speed of each trip (per-track column)',
  speed: 'speed of each step (per-vertex column)',
  heading: 'heading of each step, radians from east (per-vertex column)',
  acceleration: 'acceleration of each step (per-vertex column)'
};

/**
 * Trajectory analytics for taxi trips. One compiled `GPUTrajectoryMetrics` computes per-trip path
 * length, average speed, maximum speed, per-trip stop counts and a bounded list of stops (dwells:
 * maximal runs of steps slower than a speed threshold that last at least a minimum duration, with
 * their centroid and duration). The stop thresholds are per-frame parameters, so moving the
 * sliders never recompiles. Trails are colored either by their trip's metric (per-track columns
 * through a per-segment track index) or by the per-step speed, heading or acceleration columns
 * (through a per-segment end-vertex index). The contributor writes the stop count straight into
 * the indirect draw record (`drawInstanceCount`); only a small, throttled per-track summary is
 * read back for the panel.
 *
 * Timestamps are uploaded as float32 seconds since the first sample of the dataset: the contributor
 * has no double-single time support, so an application epoch must keep values small enough for f32.
 */
export const trajectoriesMode: SpatialAnalysisModeDefinition = {
  id: 'trajectories',
  title: 'Trajectories',
  contributors: ['GPUTrajectoryMetrics'],
  description:
    'Per-trip metrics and stop detection for taxi trips. Trails are colored by trip average or ' +
    'maximum speed, or by per-step speed, heading or acceleration (pick in the selector); ' +
    'circles mark stops, sized by duration. Stop speed threshold and minimum duration are ' +
    'per-frame parameters, so the stop list updates on the GPU without recompiling.',
  initialViewState: {longitude: -73.985, latitude: 40.74, zoom: 12.6},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'trajectories');

    const tripCount = trips.vendors.length;
    const vertexCount = trips.vertexTimestamps.length;

    // Timestamps rebased to a local epoch so f32 keeps sub-millisecond precision.
    const epoch = trips.timeRange[0];
    const rebasedTimestamps = new Float32Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      rebasedTimestamps[vertex] = trips.vertexTimestamps[vertex] - epoch;
    }

    // One drawn row per consecutive vertex pair inside a trip, plus its trip index.
    let segmentCount = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      segmentCount += Math.max(0, trips.tripOffsets[trip + 1] - trips.tripOffsets[trip] - 1);
    }
    const segments = new Float32Array(segmentCount * 4);
    const segmentTrips = new Uint32Array(segmentCount);
    // The per-step columns describe the step that ends at a vertex: a segment uses its end vertex.
    const segmentEndVertices = new Uint32Array(segmentCount);
    let row = 0;
    for (let trip = 0; trip < tripCount; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      for (let vertex = first; vertex < last; vertex++, row++) {
        segments.set(trips.vertexPositions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
        segmentTrips[row] = trip;
        segmentEndVertices[row] = vertex + 1;
      }
    }

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const timestampsBuffer = resources.createBuffer('timestamps', rebasedTimestamps);
    const offsetsBuffer = resources.createBuffer('track-offsets', trips.tripOffsets);
    const segmentsBuffer = resources.createBuffer('segments', segments);
    const segmentTripsBuffer = resources.createBuffer('segment-trips', segmentTrips);
    const segmentEndVerticesBuffer = resources.createBuffer(
      'segment-end-vertices',
      segmentEndVertices
    );
    const stepSpeedsBuffer = resources.createBuffer('step-speeds', vertexCount * 4);
    const stepHeadingsBuffer = resources.createBuffer('step-headings', vertexCount * 4);
    const stepAccelerationsBuffer = resources.createBuffer('step-accelerations', vertexCount * 4);
    const trackLengthsBuffer = resources.createBuffer('track-lengths', tripCount * 4);
    const trackDurationsBuffer = resources.createBuffer('track-durations', tripCount * 4);
    const averageSpeedsBuffer = resources.createBuffer('average-speeds', tripCount * 4);
    const maximumSpeedsBuffer = resources.createBuffer('maximum-speeds', tripCount * 4);
    const trackStopCountsBuffer = resources.createBuffer('track-stop-counts', tripCount * 4);
    const stopIdsBuffer = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
    const stopStartRowsBuffer = resources.createBuffer('stop-start-rows', STOP_CAPACITY * 4);
    const stopEndRowsBuffer = resources.createBuffer('stop-end-rows', STOP_CAPACITY * 4);
    const stopCentroidsBuffer = resources.createBuffer('stop-centroids', STOP_CAPACITY * 8);
    const stopDurationsBuffer = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
    const stopCountBuffer = resources.createBuffer('stop-count', 4);
    const stopOverflowBuffer = resources.createBuffer('stop-overflow', 4);
    const stopTotalBuffer = resources.createBuffer('stop-total', 4);
    const parameterBuffer = resources.createParameterBuffer(
      'parameters',
      'float32',
      GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
    );
    // The contributor writes the clamped stop count into the record's instance-count word.
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'trajectories-stop-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    // Summary ring layout: 4 header words, then four per-track columns, then stop durations.
    const summaryByteLength = 16 + tripCount * 4 * 4 + STOP_CAPACITY * 4;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'trajectories-summary', byteLength: summaryByteLength})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'trajectories'});
    graph.add(
      new GPUTrajectoryMetrics({
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        id: 'trajectories',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: importGraphBuffer(
          graph,
          'track-offsets',
          offsetsBuffer,
          'uint32',
          tripCount + 1
        ),
        parameters: parameterBuffer.importToGraph(graph),
        trackLengths: importGraphBuffer(
          graph,
          'track-lengths',
          trackLengthsBuffer,
          'float32',
          tripCount
        ),
        trackDurations: importGraphBuffer(
          graph,
          'track-durations',
          trackDurationsBuffer,
          'float32',
          tripCount
        ),
        averageSpeeds: importGraphBuffer(
          graph,
          'average-speeds',
          averageSpeedsBuffer,
          'float32',
          tripCount
        ),
        maximumSpeeds: importGraphBuffer(
          graph,
          'maximum-speeds',
          maximumSpeedsBuffer,
          'float32',
          tripCount
        ),
        stepSpeeds: importGraphBuffer(
          graph,
          'step-speeds',
          stepSpeedsBuffer,
          'float32',
          vertexCount
        ),
        stepHeadings: importGraphBuffer(
          graph,
          'step-headings',
          stepHeadingsBuffer,
          'float32',
          vertexCount
        ),
        stepAccelerations: importGraphBuffer(
          graph,
          'step-accelerations',
          stepAccelerationsBuffer,
          'float32',
          vertexCount
        ),
        trackStopCounts: importGraphBuffer(
          graph,
          'track-stop-counts',
          trackStopCountsBuffer,
          'uint32',
          tripCount
        ),
        stops: {
          output: {
            ids: importGraphBuffer(graph, 'stop-ids', stopIdsBuffer, 'uint32', STOP_CAPACITY),
            count: importGraphBuffer(graph, 'stop-count', stopCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'stop-overflow', stopOverflowBuffer, 'uint32', 1),
            requiredCount: importGraphBuffer(graph, 'stop-total', stopTotalBuffer, 'uint32', 1)
          },
          drawInstanceCount: graph.importGPUData(
            'stop-draw-count',
            drawCommands.getInstanceCountData(0)
          ),
          startRows: importGraphBuffer(
            graph,
            'stop-start-rows',
            stopStartRowsBuffer,
            'uint32',
            STOP_CAPACITY
          ),
          endRows: importGraphBuffer(
            graph,
            'stop-end-rows',
            stopEndRowsBuffer,
            'uint32',
            STOP_CAPACITY
          ),
          centroids: importGraphBuffer(
            graph,
            'stop-centroids',
            stopCentroidsBuffer,
            'float32x2',
            STOP_CAPACITY
          ),
          durations: importGraphBuffer(
            graph,
            'stop-durations',
            stopDurationsBuffer,
            'float32',
            STOP_CAPACITY
          )
        }
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    let stopSpeedThreshold = DEFAULT_STOP_SPEED_THRESHOLD;
    let stopMinimumDuration = DEFAULT_STOP_MINIMUM_DURATION;
    let colorMetric: ColorMetric = 'average';
    let readbackPending = false;
    let readbackRequested = true;
    let destroyed = false;
    const parameterValues = new Float32Array(GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH);

    context.controls.addSlider({
      label: 'Stop speed threshold (per-frame)',
      min: 0.5,
      max: 8,
      step: 0.25,
      value: stopSpeedThreshold,
      format: value => `${value.toFixed(2)} m/s`,
      onChange: value => {
        stopSpeedThreshold = value;
        readbackRequested = true;
      }
    });
    context.controls.addSlider({
      label: 'Minimum stop duration (per-frame)',
      min: 0,
      max: 180,
      step: 5,
      value: stopMinimumDuration,
      format: value => `${value} s`,
      onChange: value => {
        stopMinimumDuration = value;
        readbackRequested = true;
      }
    });
    const rangeNote = context.controls.addNote('');
    const describeRange = () => {
      const [minimum, maximum] = METRIC_RANGES[colorMetric];
      rangeNote.setValue(
        `Trail color scale: ${minimum.toFixed(1)} to ${maximum.toFixed(1)} ${METRIC_UNITS[colorMetric]}, ` +
          `${METRIC_DESCRIPTIONS[colorMetric]}.`
      );
    };
    describeRange();
    context.controls.addSelect<ColorMetric>({
      label: 'Colour trails by',
      options: [
        {value: 'average', label: 'Trip average speed'},
        {value: 'maximum', label: 'Trip maximum speed'},
        {value: 'speed', label: 'Step speed (per vertex)'},
        {value: 'heading', label: 'Step heading (per vertex)'},
        {value: 'acceleration', label: 'Step acceleration (per vertex)'}
      ],
      value: colorMetric,
      onChange: value => {
        colorMetric = value;
        describeRange();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Trail color: selected metric (low to high, ranges in the note above)',
      gradient: {
        colors: VIRIDIS_COLORS,
        minimumLabel: 'low',
        maximumLabel: 'high'
      }
    });
    context.controls.addLegend({
      title: 'Stop circles: radius and color grow with dwell time',
      entries: [
        {color: [255, 199, 224], label: 'Short stop'},
        {color: [255, 41, 128], label: 'Medium stop'},
        {color: [191, 0, 51], label: 'Long stop (10 min or more)'}
      ]
    });

    context.controls.addReadout('Trips', formatCount(tripCount));
    const lengthReadout = context.controls.addReadout('Total path length');
    const meanSpeedReadout = context.controls.addReadout('Mean trip average speed');
    const medianSpeedReadout = context.controls.addReadout('Median trip average speed');
    const fastestReadout = context.controls.addReadout('Fastest trip average');
    const stepSpeedReadout = context.controls.addReadout('Max step speed');
    const stopTotalReadout = context.controls.addReadout('Total stops (per-trip sum)');
    const stopTracksReadout = context.controls.addReadout('Trips with at least one stop');
    const stopListReadout = context.controls.addReadout('Stops listed');
    const longestStopReadout = context.controls.addReadout('Longest stop');
    context.controls.addNote(
      `Timestamps are float32 seconds since the dataset's first sample (${epoch.toFixed(0)} s): ` +
        'the contributor has no double-single time, so keep an application epoch for long recordings.'
    );
    context.controls.addReadout('Data', trips.attribution);

    const formatSpeed = (metersPerSecond: number) =>
      `${metersPerSecond.toFixed(2)} m/s (${(metersPerSecond * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR).toFixed(1)} km/h)`;

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copy = (source: typeof stopCountBuffer, destinationOffset: number, size: number) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      };
      const columnBytes = tripCount * 4;
      copy(stopCountBuffer, 0, 4);
      copy(stopOverflowBuffer, 4, 4);
      copy(stopTotalBuffer, 8, 4);
      copy(trackLengthsBuffer, 16, columnBytes);
      copy(averageSpeedsBuffer, 16 + columnBytes, columnBytes);
      copy(maximumSpeedsBuffer, 16 + columnBytes * 2, columnBytes);
      copy(trackStopCountsBuffer, 16 + columnBytes * 3, columnBytes);
      copy(stopDurationsBuffer, 16 + columnBytes * 4, STOP_CAPACITY * 4);
      ticket.markEncoded({byteOffset: 0, byteLength: summaryByteLength});
      readbackPending = true;
      readbackRequested = false;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const columnWords = tripCount;
        const lengths = floats.subarray(4, 4 + columnWords);
        const averages = floats.subarray(4 + columnWords, 4 + columnWords * 2);
        const maxima = floats.subarray(4 + columnWords * 2, 4 + columnWords * 3);
        const stopCounts = words.subarray(4 + columnWords * 3, 4 + columnWords * 4);
        const durations = floats.subarray(4 + columnWords * 4, 4 + columnWords * 4 + STOP_CAPACITY);

        let totalLength = 0;
        let averageSum = 0;
        let fastestAverage = 0;
        let maximumStepSpeed = 0;
        let totalStops = 0;
        let tracksWithStops = 0;
        for (let trip = 0; trip < tripCount; trip++) {
          totalLength += lengths[trip];
          averageSum += averages[trip];
          fastestAverage = Math.max(fastestAverage, averages[trip]);
          maximumStepSpeed = Math.max(maximumStepSpeed, maxima[trip]);
          totalStops += stopCounts[trip];
          if (stopCounts[trip] > 0) tracksWithStops++;
        }
        const sortedAverages = Float32Array.from(averages).sort();
        const median =
          tripCount % 2 === 1
            ? sortedAverages[(tripCount - 1) / 2]
            : (sortedAverages[tripCount / 2 - 1] + sortedAverages[tripCount / 2]) / 2;
        let longestStop = 0;
        for (let stop = 0; stop < Math.min(words[0], STOP_CAPACITY); stop++) {
          longestStop = Math.max(longestStop, durations[stop]);
        }

        lengthReadout.setValue(`${(totalLength / 1000).toFixed(1)} km`);
        meanSpeedReadout.setValue(formatSpeed(averageSum / tripCount));
        medianSpeedReadout.setValue(formatSpeed(median));
        fastestReadout.setValue(formatSpeed(fastestAverage));
        stepSpeedReadout.setValue(formatSpeed(maximumStepSpeed));
        stopTotalReadout.setValue(formatCount(totalStops));
        stopTracksReadout.setValue(formatCount(tracksWithStops));
        stopListReadout.setValue(
          `${formatCount(words[0])} of ${formatCount(words[2])}${words[1] ? ' (overflow, capacity ' + formatCount(STOP_CAPACITY) + ')' : ''}`
        );
        longestStopReadout.setValue(`${longestStop.toFixed(0)} s`);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        parameterBuffer.write(
          getGPUTrajectoryMetricsParameterValues(
            {stopSpeedThreshold, stopMinimumDuration},
            parameterValues
          )
        );
        compiled.encode(commandEncoder, {parameters: undefined});
        if (
          !readbackPending &&
          (readbackRequested || frame.frameIndex % READBACK_INTERVAL_FRAMES === 0)
        ) {
          void readSummary(commandEncoder);
        }
      },
      getLayers(): Layer[] {
        const getTrailValues = () =>
          colorMetric === 'average' || colorMetric === 'maximum'
            ? {
                values: colorMetric === 'average' ? averageSpeedsBuffer : maximumSpeedsBuffer,
                valueIndices: segmentTripsBuffer
              }
            : {
                values: {
                  speed: stepSpeedsBuffer,
                  heading: stepHeadingsBuffer,
                  acceleration: stepAccelerationsBuffer
                }[colorMetric],
                valueIndices: segmentEndVerticesBuffer
              };
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        return [
          new SpatialAnalysisSegmentLayer({
            id: 'trajectories-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            ...getTrailValues(),
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: METRIC_RANGES[colorMetric],
            color: [255, 255, 255, 215],
            widthPixels: 2.5
          }),
          new TrajectoryStopLayer({
            id: 'trajectories-stops',
            coordinateOrigin,
            centroids: stopCentroidsBuffer,
            durations: stopDurationsBuffer,
            drawCommands
          })
        ];
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
