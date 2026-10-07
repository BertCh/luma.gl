// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUTrajectoryMetrics} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FLIGHT_DATASET_ID,
  getMedian,
  KNOTS_PER_METER_SECOND,
  loadFlights,
  type FlightSet
} from './flight-corridors-data';
import {FlightSegmentLayer} from './flight-corridors-layers';

/** Option state of the jet stream scene. */
export type JetStreamOptions = {
  show: 'all' | 'east' | 'west';
  minAltitude: number;
  coneDegrees: number;
  speedSource: 'derived' | 'reported';
  speedRange: readonly [number, number];
  ramp: RampName;
  widthPixels: number;
  chartBy: 'longitude' | 'latitude' | 'altitude';
};

const HISTOGRAM_LOW = 250;
const HISTOGRAM_HIGH = 650;
const HISTOGRAM_BIN = 10;
const MINIMUM_BIN_SAMPLES = 30;

const PROFILES = {
  longitude: {low: -122, high: -70, step: 4, label: 'longitude of the step (deg)'},
  latitude: {low: 27, high: 49, step: 2, label: 'latitude of the step (deg)'},
  altitude: {low: 6000, high: 13000, step: 500, label: 'altitude (m)'}
} as const;

type Snapshot = {speeds: Float32Array; headings: Float32Array};

/**
 * Jet stream from aircraft alone. One `GPUTrajectoryMetrics` graph measures the speed and heading
 * of every step of every flight (in azimuthal-equidistant metres, once, since the data is static).
 * The map colors the steps straight from those GPU buffers with a shader-side cruise-altitude and
 * direction filter; the statistics read the same two columns back once and bin them on the CPU.
 */
export async function createJetStream(
  ctx: SceneContext<JetStreamOptions>
): Promise<SceneInstance<JetStreamOptions>> {
  const flights: FlightSet = loadFlights(ctx.datasets.get(FLIGHT_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = flights;
  const resources = new SpatialAnalysisResources(device, 'jet-stream');

  const positionsBuffer = resources.createBuffer('positions', flights.positions);
  const lngLatBuffer = resources.createBuffer('lng-lat', flights.lngLat);
  const altitudeBuffer = resources.createBuffer('altitude', flights.altitude);
  const timestampsBuffer = resources.createBuffer('timestamps', flights.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', flights.offsets);
  const endVerticesBuffer = resources.createBuffer('segment-ends', flights.segmentEndVertices);
  const reportedBuffer = resources.createBuffer('reported-knots', flights.reportedKnots);
  const stepSpeeds = resources.createBuffer('step-speeds', vertexCount * 4);
  const stepHeadings = resources.createBuffer('step-headings', vertexCount * 4);

  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'jet-stream-metrics'});
  metricsGraph.add(
    new GPUTrajectoryMetrics({
      id: 'metrics',
      positions: importGraphBuffer(
        metricsGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(
        metricsGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        metricsGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      stepSpeeds: importGraphBuffer(
        metricsGraph,
        'step-speeds',
        stepSpeeds,
        'float32',
        vertexCount
      ),
      stepHeadings: importGraphBuffer(
        metricsGraph,
        'step-headings',
        stepHeadings,
        'float32',
        vertexCount
      )
    })
  );
  const metricsCompiled = resources.track(metricsGraph.compile());

  let destroyed = false;
  let snapshot: Snapshot | null = null;
  let encoded = false;

  ctx.setReadout('flights', `${formatCount(trackCount)} flights`);
  ctx.setReadout('steps', `${formatCount(segmentCount)} steps measured on the GPU`);

  const reader = new SummaryReader(
    resources,
    'jet-stream-steps',
    [
      {buffer: stepSpeeds, size: vertexCount * 4},
      {buffer: stepHeadings, size: vertexCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      snapshot = {
        speeds: floats.slice(0, vertexCount),
        headings: floats.slice(vertexCount, vertexCount * 2)
      };
      recompute();
    }
  );

  /** Bins the measured steps with the current filters and fills the readouts and charts. */
  function recompute(): void {
    if (!snapshot || destroyed) return;
    const {minAltitude, coneDegrees, speedSource, chartBy} = ctx.options;
    const cosine = Math.cos((coneDegrees * Math.PI) / 180);
    const east = new Float32Array(vertexCount);
    const west = new Float32Array(vertexCount);
    let eastCount = 0;
    let westCount = 0;
    const profile = PROFILES[chartBy];
    const binCount = Math.round((profile.high - profile.low) / profile.step);
    const eastBins: number[][] = Array.from({length: binCount}, () => []);
    const westBins: number[][] = Array.from({length: binCount}, () => []);
    const differences: number[] = [];
    const histogramBins = Math.round((HISTOGRAM_HIGH - HISTOGRAM_LOW) / HISTOGRAM_BIN);
    const eastHistogram = new Float64Array(histogramBins);
    const westHistogram = new Float64Array(histogramBins);
    for (let track = 0; track < trackCount; track++) {
      for (let vertex = flights.offsets[track] + 1; vertex < flights.offsets[track + 1]; vertex++) {
        const altitude = Math.min(flights.altitude[vertex], flights.altitude[vertex - 1]);
        if (altitude < minAltitude) continue;
        const derived = snapshot.speeds[vertex] * KNOTS_PER_METER_SECOND;
        const reported = flights.reportedKnots[vertex];
        if (derived <= 0) continue;
        if (reported > 0) differences.push(Math.abs(derived - reported));
        const speed = speedSource === 'derived' ? derived : reported;
        if (speed <= 0) continue;
        const heading = Math.cos(snapshot.headings[vertex]);
        const isEast = heading > cosine;
        const isWest = heading < -cosine;
        if (!isEast && !isWest) continue;
        const value =
          chartBy === 'longitude'
            ? flights.lngLat[vertex * 2]
            : chartBy === 'latitude'
              ? flights.lngLat[vertex * 2 + 1]
              : flights.altitude[vertex];
        const bin = Math.floor((value - profile.low) / profile.step);
        const histogramBin = Math.floor((speed - HISTOGRAM_LOW) / HISTOGRAM_BIN);
        if (isEast) {
          east[eastCount++] = speed;
          if (bin >= 0 && bin < binCount) eastBins[bin].push(speed);
          if (histogramBin >= 0 && histogramBin < histogramBins) eastHistogram[histogramBin]++;
        } else {
          west[westCount++] = speed;
          if (bin >= 0 && bin < binCount) westBins[bin].push(speed);
          if (histogramBin >= 0 && histogramBin < histogramBins) westHistogram[histogramBin]++;
        }
      }
    }
    const eastMedian = getMedian(east, eastCount);
    const westMedian = getMedian(west, westCount);
    const difference = eastMedian - westMedian;
    ctx.setReadout('eastMedian', eastCount ? `${eastMedian.toFixed(0)} kn` : 'no steps');
    ctx.setReadout('westMedian', westCount ? `${westMedian.toFixed(0)} kn` : 'no steps');
    ctx.setReadout(
      'difference',
      eastCount && westCount
        ? `${difference.toFixed(0)} kn (${(difference * 1.852).toFixed(0)} km/h) faster eastbound`
        : 'n/a'
    );
    ctx.setReadout(
      'airspeed',
      eastCount && westCount ? `${((eastMedian + westMedian) / 2).toFixed(0)} kn` : 'n/a'
    );
    ctx.setReadout(
      'wind',
      eastCount && westCount ? `${(difference / 2).toFixed(0)} kn along the flight axis` : 'n/a'
    );
    ctx.setReadout(
      'samples',
      `${formatCount(eastCount)} eastbound, ${formatCount(westCount)} westbound steps`
    );
    ctx.setReadout(
      'agreement',
      differences.length
        ? `median |derived - reported| = ${getMedian(differences).toFixed(1)} kn over ${formatCount(differences.length)} steps`
        : 'n/a'
    );

    const centers = Array.from(
      {length: histogramBins},
      (_, bin) => HISTOGRAM_LOW + (bin + 0.5) * HISTOGRAM_BIN
    );
    const toPercent = (counts: Float64Array, total: number) =>
      Float64Array.from(counts, count => (total ? (100 * count) / total : 0));
    ctx.setChart('speedChart', {
      kind: 'line',
      height: 150,
      xLabel: `ground speed (kn, ${speedSource === 'derived' ? 'measured from positions' : 'ADS-B reported'})`,
      yLabel: 'share of steps (%)',
      xDomain: [HISTOGRAM_LOW, HISTOGRAM_HIGH],
      series: [
        {
          label: 'eastbound',
          x: centers,
          y: toPercent(eastHistogram, eastCount),
          color: 0,
          area: true
        },
        {
          label: 'westbound',
          x: centers,
          y: toPercent(westHistogram, westCount),
          color: 1,
          area: true
        }
      ],
      markers: [
        ...(eastCount ? [{x: eastMedian, label: `E ${eastMedian.toFixed(0)}`}] : []),
        ...(westCount ? [{x: westMedian, label: `W ${westMedian.toFixed(0)}`}] : [])
      ],
      formatX: value => `${Math.round(value)}`,
      formatY: value => `${value.toFixed(0)}`,
      description:
        'Distribution of cruise ground speed for eastbound and westbound steps, with the median of each marked.'
    });

    const profileCenters = Array.from(
      {length: binCount},
      (_, bin) => profile.low + (bin + 0.5) * profile.step
    );
    const eastProfile = eastBins.map(values =>
      values.length >= MINIMUM_BIN_SAMPLES ? getMedian(values) : Number.NaN
    );
    const westProfile = westBins.map(values =>
      values.length >= MINIMUM_BIN_SAMPLES ? getMedian(values) : Number.NaN
    );
    const wind = eastProfile.map((value, bin) => (value - westProfile[bin]) / 2);
    ctx.setChart('profileChart', {
      kind: 'line',
      height: 150,
      xLabel: profile.label,
      yLabel: 'median ground speed (kn)',
      xDomain: [profile.low, profile.high],
      series: [
        {label: 'eastbound', x: profileCenters, y: eastProfile, color: 0},
        {label: 'westbound', x: profileCenters, y: westProfile, color: 1},
        {label: 'half the difference', x: profileCenters, y: wind, color: 2, dashed: true}
      ],
      formatX: value => (chartBy === 'altitude' ? `${Math.round(value)}` : value.toFixed(0)),
      formatY: value => `${Math.round(value)}`,
      description: `Median ground speed of eastbound and westbound steps by ${chartBy}; the dashed line is half their difference, the average tailwind component.`
    });
  }

  return {
    getCompiledGraphs: () => [metricsCompiled],

    setOption(id) {
      if (
        id === 'minAltitude' ||
        id === 'coneDegrees' ||
        id === 'speedSource' ||
        id === 'chartBy'
      ) {
        recompute();
      }
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (!encoded) {
        // The tracks never change, so the metrics graph runs once and its two columns stay resident.
        metricsCompiled.encode(commandEncoder, {parameters: undefined});
        encoded = true;
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const derived = options.speedSource === 'derived';
      const layers: Layer[] = [
        new FlightSegmentLayer({
          id: 'jet-stream-steps',
          lngLat: lngLatBuffer,
          elevations: altitudeBuffer,
          endVertices: endVerticesBuffer,
          instanceCount: segmentCount,
          values: derived ? stepSpeeds : reportedBuffer,
          valuesAreFloat: true,
          valueScale: derived ? KNOTS_PER_METER_SECOND : 1,
          colorMode: 'value',
          ramp: options.ramp,
          valueRange: options.speedRange,
          minAltitude: options.minAltitude,
          headings: stepHeadings,
          directionFilter: options.show,
          directionHalfAngle: (options.coneDegrees * Math.PI) / 180,
          widthPixels: options.widthPixels,
          opacity: 0.9
        })
      ];
      return layers;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
