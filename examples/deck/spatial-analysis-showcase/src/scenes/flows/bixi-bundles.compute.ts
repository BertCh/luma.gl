// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  createGPUEdgeBundlingParameterValues,
  GPUEdgeBundling,
  type GPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {createTrackSet} from '../movement/b12-tracks';
import {BundledPathLayer} from './b11-flow-layers';
import {getApproximateKilometres} from './b11-flight-data';
import {findStationNearPixel, formatCompact, readBixiFlows} from './bixi-data';
import {buildUndirectedEdges} from './bixi-graph';

/** Option state of the bixi-bundles scene. */
export type BixiBundlesOptions = {
  time: number;
  play: boolean;
  speed: number;
  loop: boolean;
  showRides: boolean;
  trailMinutes: number;
  tailFade: number;
  bikeSize: number;
  edges: number;
  minRides: number;
  iterations: number;
  kernelRadius: number;
  decay: number;
  stiffness: number;
  stepScale: number;
  pointsPerEdge: '8' | '16' | '24' | '32';
  densityResolution: '128' | '256' | '512';
  colorBy: 'rides' | 'length' | 'crossing' | 'plain';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  opacity: number;
  showStraight: boolean;
  showStations: boolean;
};

/** Compile-time iteration capacity; the slider picks how many run. */
export const BUNDLE_MAXIMUM_ITERATIONS = 32;
/** Compile-time edge capacity: the busiest station pairs. */
export const BUNDLE_EDGE_CAPACITY = 30000;
/** Length of the ride window in seconds (07:30 to 10:00). */
export const RIDE_WINDOW_SECONDS = 9000;
const HUB_COUNT = 24;
const COVERAGE_CELL_DEGREES = 0.002;
const RETIRE_FRAMES = 4;
const SETTLE_MILLISECONDS = 350;
const STATUS_INTERVAL_FRAMES = 12;

type BundlingGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  pointsPerEdge: number;
  paths: Buffer;
  reader: SummaryReader;
};

/**
 * Edge-bundled station flows with the morning's rides running over them. One compiled
 * `GPUEdgeBundling` graph turns the busiest 30,000 BIXI station pairs into bundled polylines (the
 * iteration count, kernel radius, decay, stiffness and step scale are one parameter buffer; the
 * edge mask applies the ride filters). A `GPUTrajectoryPlayhead` graph interpolates 8,365 routed
 * rides at the clock and a `GPUTimeWindowFilter` graph selects the trail segments, both driven by
 * parameter buffers.
 */
export async function createBixiBundles(
  ctx: SceneContext<BixiBundlesOptions>
): Promise<SceneInstance<BixiBundlesOptions>> {
  const {device} = ctx;
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const rideData = ctx.datasets.get('poopdeck-bixi-rides');
  const allEdges = buildUndirectedEdges(flows);
  const edgeCount = Math.min(BUNDLE_EDGE_CAPACITY, allEdges.count);
  const resources = new SpatialAnalysisResources(device, 'bixi-bundles');
  const stationCount = flows.stationCount;

  // ---- Edge inputs ------------------------------------------------------------------------------
  const lengthKm = new Float32Array(edgeCount);
  const crossing = new Float32Array(edgeCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    const a = allEdges.a[edge];
    const b = allEdges.b[edge];
    lengthKm[edge] = getApproximateKilometres(
      flows.lngLat[a * 2],
      flows.lngLat[a * 2 + 1],
      flows.lngLat[b * 2],
      flows.lngLat[b * 2 + 1]
    );
    crossing[edge] = flows.borough[a] === flows.borough[b] ? 0 : 1;
  }
  const rideShare = (() => {
    let covered = 0;
    for (let edge = 0; edge < edgeCount; edge++) covered += allEdges.rides[edge];
    return covered / allEdges.totalRides;
  })();
  const ranges: Record<'rides' | 'length' | 'crossing', [number, number]> = {
    rides: [
      Math.max(1, allEdges.rides[Math.floor(edgeCount * 0.9)]),
      allEdges.rides[Math.floor(edgeCount * 0.01)]
    ],
    length: [0, percentile(lengthKm, 0.95)],
    crossing: [0, 1]
  };
  const hubRows = (() => {
    const order = Array.from({length: stationCount}, (_, station) => station).sort(
      (x, y) => flows.departures[y] + flows.arrivals[y] - flows.departures[x] - flows.arrivals[x]
    );
    return Uint32Array.from(order.slice(0, HUB_COUNT));
  })();
  const positions = resources.createBuffer('positions', flows.lngLat);
  const sources = resources.createBuffer('sources', allEdges.a.slice(0, edgeCount));
  const targets = resources.createBuffer('targets', allEdges.b.slice(0, edgeCount));
  const values = resources.createBuffer('values', lengthKm);
  const mask = resources.createBuffer('mask', new Uint32Array(edgeCount).fill(1));
  const maskWeights = resources.createBuffer('mask-weights', new Float32Array(edgeCount).fill(1));
  const straight = resources.createBuffer('straight', allEdges.segments.slice(0, edgeCount * 4));
  const hubs = resources.createBuffer('hubs', hubRows);
  const parameterBuffer = resources.createParameterBuffer('bundling-parameters', 'uint32', 5);

  let bundling: BundlingGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let lastChange = performance.now();
  let statsStale = true;
  let liveEdges = 0;
  let liveMask = new Uint32Array(edgeCount);
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    statsStale = true;
    lastChange = performance.now();
  }

  function getParameters(): GPUEdgeBundlingParameterValues {
    const {iterations, kernelRadius, decay, stiffness, stepScale} = ctx.options;
    return {
      activeIterations: iterations,
      kernelRadius,
      lambda: decay,
      smoothing: stiffness,
      stepScale
    };
  }

  function writeParameters(): void {
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    markChanged();
  }

  function writeMask(): void {
    const {edges: limit, minRides} = ctx.options;
    const live = new Uint32Array(edgeCount);
    const weights = new Float32Array(edgeCount);
    liveEdges = 0;
    let rides = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      const on = edge < limit && allEdges.rides[edge] >= minRides ? 1 : 0;
      live[edge] = on;
      weights[edge] = on;
      liveEdges += on;
      if (on) rides += allEdges.rides[edge];
    }
    mask.write(live);
    maskWeights.write(weights);
    liveMask = live;
    ctx.setReadout('edges', `${formatCount(liveEdges)} of ${formatCount(edgeCount)}`);
    ctx.setReadout('controlPoints', liveEdges * (bundling?.pointsPerEdge ?? 16));
    ctx.setReadout('ridesCovered', rides / allEdges.totalRides);
    markChanged();
  }

  function writeValues(): void {
    const {colorBy} = ctx.options;
    if (colorBy === 'plain') return;
    values.write(
      colorBy === 'rides'
        ? allEdges.rides.slice(0, edgeCount)
        : colorBy === 'length'
          ? lengthKm
          : crossing
    );
    ctx.setLegendExtent('edge-value', ranges[colorBy]);
  }

  function buildBundling(): BundlingGraph {
    const pointsPerEdge = Number(ctx.options.pointsPerEdge);
    const densityResolution = Number(ctx.options.densityResolution);
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `bixi-bundling-${id}`);
    const paths = graphResources.createBuffer('paths', edgeCount * pointsPerEdge * 8);
    const commandGraph = new GPUCommandGraph<void>(device, {id: `bixi-bundling-${id}`});
    commandGraph.add(
      new GPUEdgeBundling({
        id: 'bundling',
        positions: importGraphBuffer(
          commandGraph,
          'positions',
          positions,
          'float32x2',
          stationCount
        ),
        sourceVertices: importGraphBuffer(commandGraph, 'sources', sources, 'uint32', edgeCount),
        targetVertices: importGraphBuffer(commandGraph, 'targets', targets, 'uint32', edgeCount),
        edgeMask: importGraphBuffer(commandGraph, 'mask', mask, 'uint32', edgeCount),
        geographic: true,
        pointsPerEdge,
        iterations: BUNDLE_MAXIMUM_ITERATIONS,
        densityResolution,
        parameters: parameterBuffer.importToGraph(commandGraph),
        paths: importGraphBuffer(
          commandGraph,
          'paths',
          paths,
          'float32x2',
          edgeCount * pointsPerEdge
        )
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    const built: BundlingGraph = {
      resources: graphResources,
      compiled,
      pointsPerEdge,
      paths,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `bixi-bundling-${id}`,
      [{buffer: paths, size: edgeCount * pointsPerEdge * 8}],
      bytes => {
        if (!destroyed && bundling === built) processPaths(built, new Float32Array(bytes));
      }
    );
    return built;
  }

  function rebuildBundling(): void {
    if (bundling) retired.push({resources: bundling.resources, frames: 0});
    bundling = buildBundling();
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    writeMask();
    writeValues();
    markChanged();
  }

  /** Path stretch and map coverage of the bundled polylines against the straight edges. */
  function processPaths(current: BundlingGraph, paths: Float32Array): void {
    const {pointsPerEdge} = current;
    const stretches: number[] = [];
    let straightKilometres = 0;
    let bundledKilometres = 0;
    const straightCells = new Set<number>();
    const bundledCells = new Set<number>();
    const mark = (cells: Set<number>, ax: number, ay: number, bx: number, by: number) => {
      const steps = Math.max(
        1,
        Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(by - ay)) / (COVERAGE_CELL_DEGREES * 0.5))
      );
      for (let step = 0; step <= steps; step++) {
        const t = step / steps;
        const x = Math.floor((ax + (bx - ax) * t) / COVERAGE_CELL_DEGREES);
        const y = Math.floor((ay + (by - ay) * t) / COVERAGE_CELL_DEGREES);
        cells.add((x + 100000) * 200000 + (y + 100000));
      }
    };
    for (let edge = 0; edge < edgeCount; edge++) {
      if (liveMask[edge] !== 1) continue;
      const first = edge * pointsPerEdge * 2;
      const last = first + (pointsPerEdge - 1) * 2;
      const direct = getApproximateKilometres(
        paths[first],
        paths[first + 1],
        paths[last],
        paths[last + 1]
      );
      straightKilometres += direct;
      mark(straightCells, paths[first], paths[first + 1], paths[last], paths[last + 1]);
      let along = 0;
      for (let point = 0; point < pointsPerEdge - 1; point++) {
        const a = first + point * 2;
        along += getApproximateKilometres(paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
        mark(bundledCells, paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      }
      bundledKilometres += along;
      if (direct > 0.3) stretches.push(along / direct);
    }
    ctx.setChart(
      'stretchChart',
      stretches.length
        ? histogramChart(binValues(stretches, 1, 1.6, 24), 1, 1.6, {
            xLabel: 'path length / straight length (60% and over in the last bin)',
            yLabel: 'pairs',
            color: 3,
            formatX: value => value.toFixed(2),
            formatY: formatCompact,
            description:
              'Histogram of how much longer each bundled pair is than its straight line. Most pairs stay within a few percent; the tail is pairs pulled into a nearby corridor.'
          })
        : null
    );
    ctx.setReadout(
      'stretch',
      straightKilometres > 0 ? bundledKilometres / straightKilometres : null
    );
    ctx.setReadout('coverageStraight', straightCells.size);
    ctx.setReadout('coverageBundled', bundledCells.size);
    ctx.setReadout(
      'inkSaved',
      straightCells.size > 0 ? 1 - bundledCells.size / straightCells.size : null
    );
  }

  // ---- Rides: playhead and trail graphs ---------------------------------------------------------
  const origin = rideData.defaultOrigin;
  const projection = rideData.getProjection(origin);
  const trackSet = createTrackSet({
    offsets: rideData.column<Uint32Array>('pathOffsets'),
    positions: rideData.projectColumn('vertices', origin),
    lngLat: rideData.column<Float32Array>('vertices'),
    timestamps: Float32Array.from(rideData.column<Uint32Array>('timestamp')),
    origin,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    unproject: (x, y) => projection.unproject(x, y),
    drawInDegrees: false
  });
  const {trackCount, vertexCount, segmentCount} = trackSet;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const ridePositions = resources.createBuffer('ride-positions', trackSet.positions);
  const rideTimes = resources.createBuffer('ride-times', trackSet.timestamps);
  const rideOffsets = resources.createBuffer('ride-offsets', trackSet.offsets);
  const rideSegments = resources.createBuffer('ride-segments', trackSet.segments);
  const segmentStartTimes = resources.createBuffer(
    'segment-start-times',
    trackSet.segmentStartTimes
  );
  const segmentEndTimes = resources.createBuffer('segment-end-times', trackSet.segmentEndTimes);
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const speeds = resources.createBuffer('speeds', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const bikeDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'bixi-bike-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'bixi-playhead'});
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'positions',
        ridePositions,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(playheadGraph, 'timestamps', rideTimes, 'float32', vertexCount),
      trackOffsets: importGraphBuffer(
        playheadGraph,
        'offsets',
        rideOffsets,
        'uint32',
        trackCount + 1
      ),
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: importGraphBuffer(
        playheadGraph,
        'current',
        currentPositions,
        'float32x2',
        trackCount
      ),
      speeds: importGraphBuffer(playheadGraph, 'speeds', speeds, 'float32', trackCount),
      activeTracks: {
        ids: importGraphBuffer(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
        count: importGraphBuffer(playheadGraph, 'active-count', activeCount, 'uint32', 1),
        overflow: importGraphBuffer(playheadGraph, 'active-overflow', activeOverflow, 'uint32', 1)
      },
      drawInstanceCount: playheadGraph.importGPUData(
        'bike-draw-count',
        bikeDraw.getInstanceCountData(0)
      )
    })
  );
  const playheadCompiled = resources.track(playheadGraph.compile());

  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'bixi-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'bixi-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start',
        segmentStartTimes,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end',
        segmentEndTimes,
        'float32',
        segmentCount
      ),
      window: windowParameters.importToGraph(trailGraph),
      output: {
        ids: importGraphBuffer(trailGraph, 'trail-ids', trailIds, 'uint32', segmentCount),
        count: importGraphBuffer(trailGraph, 'trail-count', trailCount, 'uint32', 1),
        overflow: importGraphBuffer(trailGraph, 'trail-overflow', trailOverflow, 'uint32', 1)
      },
      fadeWeights: importGraphBuffer(
        trailGraph,
        'fade-weights',
        fadeWeights,
        'float32',
        segmentCount
      ),
      clipFractions: importGraphBuffer(
        trailGraph,
        'clip-fractions',
        clipFractions,
        'float32x2',
        segmentCount
      ),
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  const statusReader = new SummaryReader(
    resources,
    'bixi-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: trailCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      ctx.setReadout('bikes', words[0]);
      ctx.setReadout('trailSegments', words[1]);
    }
  );

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, RIDE_WINDOW_SECONDS], rate: 60, step: 30, notify: false}
  );

  // Rides in progress per minute, from the ride start and end times.
  const concurrent = (() => {
    const minutes = RIDE_WINDOW_SECONDS / 60;
    const counts = new Float64Array(minutes + 1);
    for (let track = 0; track < trackCount; track++) {
      const start = trackSet.timestamps[trackSet.offsets[track]];
      const end = trackSet.timestamps[trackSet.offsets[track + 1] - 1];
      for (
        let minute = Math.max(0, Math.floor(start / 60));
        minute <= Math.min(minutes, Math.floor(end / 60));
        minute++
      ) {
        counts[minute]++;
      }
    }
    return counts;
  })();
  let lastChartMinute = -1;
  let statusStale = true;

  function formatRideClock(seconds: number): string {
    const total = 7.5 * 3600 + seconds;
    const hour = Math.floor(total / 3600);
    const minute = Math.floor((total % 3600) / 60);
    return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
  }

  function updateRideChart(playhead: number): void {
    const minute = Math.round(playhead / 60);
    if (minute === lastChartMinute) return;
    lastChartMinute = minute;
    const x = Array.from(concurrent, (_, index) => 7.5 + index / 60);
    ctx.setChart('ridesChart', {
      kind: 'line',
      series: [{label: 'rides in progress', x, y: concurrent, area: true, color: 0}],
      xDomain: [7.5, 10],
      markers: [{x: 7.5 + playhead / 3600, label: formatRideClock(playhead)}],
      xLabel: 'time of day (local)',
      yLabel: 'rides in progress',
      height: 120,
      formatX: value =>
        `${Math.floor(value)}:${String(Math.round((value % 1) * 60)).padStart(2, '0')}`,
      formatY: formatCompact,
      description:
        'Routed BIXI rides in progress each minute between 07:30 and 10:00 on 15 August 2024; the rule is the playhead. Rides already underway at 07:30 are counted from the start.'
    });
  }

  ctx.setReadout('rides', trackCount);
  ctx.setReadout('ridesShare', rideShare);
  rebuildBundling();

  return {
    getCompiledGraphs: () => [
      ...(bundling ? [bundling.compiled as CompiledGPUCommandGraph<never>] : []),
      playheadCompiled as CompiledGPUCommandGraph<never>,
      trailCompiled as CompiledGPUCommandGraph<never>
    ],

    setOption(id) {
      switch (id) {
        case 'pointsPerEdge':
        case 'densityResolution':
          rebuildBundling();
          ctx.requestLayers();
          break;
        case 'edges':
        case 'minRides':
          writeMask();
          break;
        case 'iterations':
        case 'kernelRadius':
        case 'decay':
        case 'stiffness':
        case 'stepScale':
          writeParameters();
          break;
        case 'colorBy':
          writeValues();
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          statusStale = true;
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel
      );
      if (station < 0) return null;
      return `${flows.names[station]}\n${flows.boroughNames[flows.borough[station]]}\n${formatCompact(flows.departures[station])} departures, ${formatCompact(flows.arrivals[station])} arrivals in August`;
    },

    encode(commandEncoder, frame) {
      if (!bundling) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      const options = ctx.options;
      const playhead = clock.advance(frame);
      // Paths persist in their buffer, so the bundling only re-encodes after a change.
      if (encodeFrames > 0) {
        bundling.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      if (
        statsStale &&
        encodeFrames === 0 &&
        performance.now() - lastChange > SETTLE_MILLISECONDS
      ) {
        if (!bundling.reader.isPending) {
          statsStale = false;
          bundling.reader.request(commandEncoder);
        }
      } else {
        bundling.reader.flush(commandEncoder);
      }
      ctx.setReadout('clock', formatRideClock(playhead));
      if (frame.frameIndex % 4 === 0) updateRideChart(playhead);
      if (options.showRides) {
        playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
        playheadCompiled.encode(commandEncoder, {parameters: undefined});
        const trailSeconds = options.trailMinutes * 60;
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - trailSeconds,
            end: playhead,
            startFadeDuration: trailSeconds * options.tailFade
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
        if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
          statusReader.markStale();
          statusStale = false;
        }
        statusReader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!bundling) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const colored = options.colorBy !== 'plain';
      const layers: Layer[] = [];
      if (options.showStraight) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-straight',
            coordinateSystem: lngLat,
            segments: straight,
            weights: maskWeights,
            instanceCount: edgeCount,
            widthPixels: 0.8,
            color: dark ? [180, 195, 230, 40] : [60, 70, 100, 45]
          })
        );
      }
      layers.push(
        new BundledPathLayer({
          id: `bixi-bundles-${bundling.pointsPerEdge}`,
          paths: bundling.paths,
          pointsPerPath: bundling.pointsPerEdge,
          pathCount: edgeCount,
          values: colored ? values : null,
          valueRange: ranges[options.colorBy === 'plain' ? 'rides' : options.colorBy],
          ramp: options.ramp,
          sqrtScale: options.colorBy === 'rides',
          edgeMask: mask,
          startColor: dark ? [96, 214, 255, 255] : [20, 110, 190, 255],
          endColor: dark ? [96, 214, 255, 255] : [20, 110, 190, 255],
          opacity: options.opacity
        })
      );
      if (options.showStations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-bundle-stations',
            coordinateSystem: lngLat,
            positions,
            instanceCount: stationCount,
            radiusPixels: 1.6,
            color: dark ? [235, 240, 255, 150] : [30, 40, 70, 150]
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bundle-hubs',
            coordinateSystem: lngLat,
            positions,
            ids: hubs,
            instanceCount: hubRows.length,
            radiusPixels: 4,
            color: [255, 184, 64, 235]
          })
        );
      }
      if (options.showRides) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-ride-trails',
            coordinateOrigin,
            segments: rideSegments,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            color: dark ? [255, 120, 150, 255] : [210, 40, 90, 255],
            widthPixels: 2.2
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bike-halo',
            coordinateOrigin,
            ids: activeIds,
            positions: currentPositions,
            drawCommands: bikeDraw,
            radiusPixels: options.bikeSize + 1.4,
            color: dark ? [8, 10, 18, 235] : [250, 250, 253, 235]
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bikes',
            coordinateOrigin,
            ids: activeIds,
            positions: currentPositions,
            drawCommands: bikeDraw,
            values: speeds,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: [0, 7],
            radiusPixels: options.bikeSize
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      bundling?.reader.stop();
      statusReader.stop();
      bundling?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}

function percentile(values: ArrayLike<number>, fraction: number): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? 1;
}
