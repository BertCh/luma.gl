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
import {dissolveBoundaries, toSegmentPairs} from '../../cartography/boundaries';
import {MONTREAL} from '../../cartography/gazetteer';
import {getLocalProjector} from '../../cartography/segments';
import {formatCount, formatDistance, formatPercent, liveText} from '../../cartography/live-text';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {directionFor} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {MapAnnotation, SceneContext, SceneInstance} from '../scene';
import {createTrackSet} from '../movement/b12-tracks';
import {BundleRibbonLayer, RideTrailLayer} from './bixi-bundles-layers';
import {
  BIKE_HALO,
  BIKE_INK,
  BIKE_RADIUS_PIXELS,
  BIKE_SPEED_EXTENT,
  BIKE_SPEED_RANGE,
  DEMOTED_BUNDLE_WIDTH_PIXELS,
  formatRideClock,
  getCrossingPalette,
  getDemotedBundleInk,
  getRideClassTable,
  GHOST_INK,
  GHOST_WIDTH_PIXELS,
  HUB_NAME_COUNT,
  HUB_RING_COUNT,
  HUB_RING_INK,
  HUB_RING_RADIUS_PIXELS,
  RIDE_CLASS_BREAKS,
  RIDE_TRAIL_INK,
  RIDE_TRAIL_WIDTH_PIXELS,
  RIDE_WINDOW_SECONDS,
  STATION_INK,
  STATION_RADIUS_PIXELS,
  STREET_DISTANCE_METERS,
  TRUNK_WIDTH_MAX_PIXELS,
  TRUNK_WIDTH_MIN_PIXELS
} from './bixi-bundles-style';
import {
  binValues,
  createStreetIndex,
  getKilometres,
  getMedian,
  getParetoSeries,
  getStationLabel,
  getWorkBoxSideMeters,
  measureBundles,
  type StreetIndex
} from './bixi-bundles-stats';
import {findStationNearPixel, formatCompact, readBixiFlows} from './bixi-data';
import {buildUndirectedEdges} from './bixi-graph';
import {CONTEXT_INK, inkFor} from './flows-style';

/** Option state of the bixi-bundles scene. */
export type BixiBundlesOptions = {
  time: number;
  play: boolean;
  speed: number;
  loop: boolean;
  showRides: boolean;
  trailMinutes: number;
  tailFade: number;
  bikeSpeed: boolean;
  edges: number;
  minRides: number;
  iterations: number;
  kernelRadius: number;
  decay: number;
  stiffness: number;
  stepScale: number;
  pointsPerEdge: '8' | '16' | '24' | '32';
  densityResolution: '128' | '256' | '512';
  colorBy: 'rides' | 'crossing';
  radiusRing: 'round' | 'start';
  compareStraight: boolean;
  showStraight: boolean;
  showBoroughs: boolean;
  showStations: boolean;
};

/** Compile-time iteration capacity; the slider picks how many run. */
export const BUNDLE_MAXIMUM_ITERATIONS = 32;
/** Compile-time edge capacity: the busiest station pairs. */
export const BUNDLE_EDGE_CAPACITY = 30000;
const RETIRE_FRAMES = 4;
const SETTLE_MILLISECONDS = 350;
const STATUS_INTERVAL_FRAMES = 12;
const STRETCH_BINS = 24;
const STRETCH_MAXIMUM = 1.6;
const DOWNTOWN = MONTREAL.places.downtown.lngLat;

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
 * edge mask applies the ride filters) and a scene-local ribbon layer draws them with the width
 * following the rides on the pair. A `GPUTrajectoryPlayhead` graph interpolates 8,365 routed
 * rides at the clock and a `GPUTimeWindowFilter` graph selects the trail segments, both driven by
 * parameter buffers. The CPU adds what the story needs to say: the kernel radius in metres, the
 * Pareto curve of the pairs, and how far the bundles lie from the streets the rides used.
 */
export async function createBixiBundles(
  ctx: SceneContext<BixiBundlesOptions>
): Promise<SceneInstance<BixiBundlesOptions>> {
  const {device} = ctx;
  const flowDataset = ctx.datasets.get('bixi-flows');
  const flows = readBixiFlows(flowDataset);
  const sourceProperties = flowDataset.properties as {
    droppedShortRides?: number;
    pairMin?: number;
  };
  const rideData = ctx.datasets.get('poopdeck-bixi-rides');
  const allEdges = buildUndirectedEdges(flows);
  const edgeCount = Math.min(BUNDLE_EDGE_CAPACITY, allEdges.count);
  const resources = new SpatialAnalysisResources(device, 'bixi-bundles');
  const stationCount = flows.stationCount;
  // Rides between two different stations: the base of every share the story quotes.
  const stationToStationRides = flows.totalRides - flows.sameStationRides;
  const maximumRides = allEdges.rides[0];
  ctx.setAnnotationHalo('heavy');

  // ---- Edge inputs ------------------------------------------------------------------------------
  const lengthKm = new Float32Array(allEdges.count);
  const crossing = new Float32Array(edgeCount);
  const rideClass = new Float32Array(edgeCount);
  const cumulativeRides = new Float64Array(allEdges.count + 1);
  for (let edge = 0; edge < allEdges.count; edge++) {
    const a = allEdges.a[edge];
    const b = allEdges.b[edge];
    lengthKm[edge] = getKilometres(
      flows.lngLat[a * 2],
      flows.lngLat[a * 2 + 1],
      flows.lngLat[b * 2],
      flows.lngLat[b * 2 + 1]
    );
    cumulativeRides[edge + 1] = cumulativeRides[edge] + allEdges.rides[edge];
    if (edge < edgeCount) {
      crossing[edge] = flows.borough[a] === flows.borough[b] ? 0 : 1;
      rideClass[edge] = RIDE_CLASS_BREAKS.filter(limit => allEdges.rides[edge] >= limit).length;
    }
  }
  const pareto = getParetoSeries(cumulativeRides, stationToStationRides);
  const eligiblePairRides = cumulativeRides[allEdges.count];

  // The busiest partner of every station: edges are ranked by rides, so the first one seen wins.
  const partner = new Int32Array(stationCount).fill(-1);
  const partnerRides = new Float32Array(stationCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    for (const [from, to] of [
      [allEdges.a[edge], allEdges.b[edge]],
      [allEdges.b[edge], allEdges.a[edge]]
    ]) {
      if (partner[from] < 0) {
        partner[from] = to;
        partnerRides[from] = allEdges.rides[edge];
      }
    }
  }

  const hubRows = (() => {
    const order = Array.from({length: stationCount}, (_, station) => station).sort(
      (x, y) => flows.departures[y] + flows.arrivals[y] - flows.departures[x] - flows.arrivals[x]
    );
    return Uint32Array.from(order.slice(0, HUB_RING_COUNT));
  })();
  const hubAnnotations: MapAnnotation[] = Array.from(hubRows.slice(0, HUB_NAME_COUNT)).map(
    (station, rank) => ({
      kind: 'point',
      id: `hub-${rank}`,
      coordinate: [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]],
      text: getStationLabel(flows.names[station]),
      rank: 'context',
      marker: 'none',
      minZoom: 11.6,
      priority: 3
    })
  );
  ctx.setAnnotations('hubs', hubAnnotations);

  const positions = resources.createBuffer('positions', flows.lngLat);
  const sources = resources.createBuffer('sources', allEdges.a.slice(0, edgeCount));
  const targets = resources.createBuffer('targets', allEdges.b.slice(0, edgeCount));
  const rideValues = resources.createBuffer('rides', allEdges.rides.slice(0, edgeCount));
  const rideClasses = resources.createBuffer('ride-classes', rideClass);
  const crossingClasses = resources.createBuffer('crossing-classes', crossing);
  const mask = resources.createBuffer('mask', new Uint32Array(edgeCount).fill(1));
  const hubs = resources.createBuffer('hubs', hubRows);

  // Faint borough outlines (context): shared edges once, the outer edge of the union once.
  const boroughs = ctx.datasets.get('montreal-boroughs');
  const outlineRows = (() => {
    if (!boroughs.geojson) return new Float32Array(0);
    const {interior, exterior} = dissolveBoundaries(boroughs.geojson, (_feature, index) => index);
    const joined = new Float64Array(interior.length + exterior.length);
    joined.set(interior);
    joined.set(exterior, interior.length);
    return toSegmentPairs(joined, getLocalProjector(boroughs.defaultOrigin));
  })();
  const outlineCount = outlineRows.length / 4;
  const outlineBuffer = resources.createBuffer('borough-outline', outlineRows);
  const outlineOrigin: [number, number, number] = [
    boroughs.defaultOrigin[0],
    boroughs.defaultOrigin[1],
    0
  ];
  const parameterBuffer = resources.createParameterBuffer('bundling-parameters', 'uint32', 5);

  let bundling: BundlingGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let lastChange = performance.now();
  let statsStale = true;
  let liveEdges = 0;
  let liveMask = new Uint32Array(edgeCount);
  let boxSideMeters = 0;
  let legendClasses: readonly number[] | null = null;
  let ringKey = '';
  let trunkShown = false;
  let streets: StreetIndex | null = null;
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    markStatsStale();
    if (trunkShown) {
      ctx.setAnnotations('trunk', null);
      trunkShown = false;
    }
  }

  function markStatsStale(): void {
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

  /** The kernel in metres: the readouts, the dashed ring at downtown and the scale-bar tick. */
  function updateKernelGeometry(): void {
    const {iterations, kernelRadius, decay, radiusRing, showRides} = ctx.options;
    const startMeters = kernelRadius * boxSideMeters;
    // The radius is multiplied by the decay after every round, so round n used r0 * decay^(n-1).
    const roundMeters = startMeters * decay ** Math.max(iterations - 1, 0);
    ctx.setReadout('radiusStart', startMeters > 0 ? startMeters : null);
    ctx.setReadout('radiusNow', iterations > 0 && roundMeters > 0 ? roundMeters : null);
    ctx.setReadout('boxSide', boxSideMeters > 0 ? boxSideMeters : null);
    const ringMeters = radiusRing === 'start' ? startMeters : roundMeters;
    const visible = iterations > 0 && !showRides && ringMeters > 0;
    const key = visible ? `${Math.round(ringMeters)}:${radiusRing}` : 'hidden';
    if (key === ringKey) return;
    ringKey = key;
    ctx.setAnnotations(
      'kernel',
      visible
        ? [
            {
              kind: 'ring',
              id: 'kernel-radius',
              coordinate: DOWNTOWN,
              radiusMeters: ringMeters,
              text: `kernel radius ${formatDistance(ringMeters)}`,
              dashed: true,
              tone: 'signal'
            }
          ]
        : null
    );
    ctx.setFurniture({scaleBar: {units: 'metric', ticks: visible ? [ringMeters] : []}});
  }

  function writeParameters(): void {
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    ctx.setReadout('iteration', ctx.options.iterations);
    updateKernelGeometry();
    publishCost();
    markChanged();
  }

  function publishCost(): void {
    const controlPoints =
      liveEdges * (bundling?.pointsPerEdge ?? Number(ctx.options.pointsPerEdge));
    ctx.setCost({
      records: controlPoints,
      // Nodes of the bundling graph: four, then three per active round.
      passes: 4 + 3 * ctx.options.iterations,
      note: 'encoded again only when a parameter changes'
    });
    ctx.setReadout('controlPoints', controlPoints);
  }

  /** The ride filters, the readouts that depend on them and the Pareto curve with its cutoff. */
  function writeMask(): void {
    const {edges: limit, minRides} = ctx.options;
    const live = new Uint32Array(edgeCount);
    const classCounts = new Array<number>(RIDE_CLASS_BREAKS.length + 1).fill(0);
    const drawnLengths: number[] = [];
    const leftOutLengths: number[] = [];
    liveEdges = 0;
    let rides = 0;
    let crossingRides = 0;
    for (let edge = 0; edge < allEdges.count; edge++) {
      const on = edge < edgeCount && edge < limit && allEdges.rides[edge] >= minRides;
      if (!on) {
        leftOutLengths.push(lengthKm[edge]);
        continue;
      }
      live[edge] = 1;
      liveEdges++;
      rides += allEdges.rides[edge];
      if (crossing[edge] === 1) crossingRides += allEdges.rides[edge];
      classCounts[rideClass[edge]]++;
      drawnLengths.push(lengthKm[edge]);
    }
    mask.write(live);
    liveMask = live;
    boxSideMeters = getWorkBoxSideMeters(flows.lngLat, allEdges.a, allEdges.b, live, edgeCount);
    ctx.setReadout('drawn', `${formatCount(liveEdges)} of ${formatCount(allEdges.count)}`);
    ctx.setReadout('ridesShare', rides / stationToStationRides);
    ctx.setReadout('crossingShare', rides > 0 ? crossingRides / rides : null);
    ctx.setReadout(
      'lengthBias',
      liveEdges > 0 && leftOutLengths.length > 0
        ? `${getMedian(drawnLengths).toFixed(1)} km vs ${getMedian(leftOutLengths).toFixed(1)} km`
        : null
    );
    ctx.setChart('paretoChart', {
      kind: 'line',
      series: [
        {label: 'cumulative share of rides', x: pareto.x, y: pareto.y, color: 0, area: true}
      ],
      xScale: 'log',
      xDomain: [1, allEdges.count],
      yDomain: [0, 1],
      xLabel: 'pairs ranked by rides (log scale)',
      yLabel: 'share of rides between stations',
      formatX: formatCompact,
      formatY: value => formatPercent(value),
      link: {option: 'edges', label: value => `${formatCompact(value)} pairs`},
      height: 130,
      description:
        'Pareto curve of the station pairs ranked by rides in August 2024: the busiest few thousand pairs carry a large share of all rides between stations and the rest add little each. The marker is the number of pairs drawn.'
    });
    ctx.setLegendData('rideClassCounts', classCounts);
    ctx.setLegendData('maximumRides', maximumRides);
    updateKernelGeometry();
    publishCost();
    markChanged();
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
    ctx.setReadout('iteration', ctx.options.iterations);
    writeMask();
  }

  function getStreets(): StreetIndex {
    streets ??= createStreetIndex(
      rideData.column<Float32Array>('vertices'),
      rideData.column<Uint32Array>('pathOffsets'),
      [rideData.defaultOrigin[0], rideData.defaultOrigin[1]]
    );
    return streets;
  }

  /** What the bundles did to the map, read back once the parameters have settled. */
  function processPaths(current: BundlingGraph, paths: Float32Array): void {
    const options = ctx.options;
    const measures = measureBundles({
      paths,
      pointsPerEdge: current.pointsPerEdge,
      edgeCount,
      mask: liveMask,
      // Distance from the streets needs the rides' index; only the rides step asks for it.
      streets: options.showRides ? getStreets() : null,
      streetMeters: STREET_DISTANCE_METERS
    });
    ctx.setChart(
      'stretchChart',
      measures.stretches.length
        ? {
            kind: 'histogram',
            values: binValues(measures.stretches, 1, STRETCH_MAXIMUM, STRETCH_BINS),
            xDomain: [1, STRETCH_MAXIMUM],
            xLabel: 'bundled length / straight length (longer ones in the last bin)',
            yLabel: 'pairs',
            color: 1,
            formatX: value => value.toFixed(2),
            formatY: formatCompact,
            height: 110,
            description:
              'Histogram of how much longer each bundled pair is than its straight line. Most pairs stay close to 1; the tail is pairs pulled into a nearby corridor.'
          }
        : null
    );
    ctx.setReadout('stretch', measures.meanStretch);
    ctx.setReadout('areaStraight', measures.straightAreaKm2 > 0 ? measures.straightAreaKm2 : null);
    ctx.setReadout('areaBundled', measures.bundledAreaKm2 > 0 ? measures.bundledAreaKm2 : null);
    ctx.setReadout('offStreetBundled', measures.offStreetBundled);
    ctx.setReadout('offStreetStraight', measures.offStreetStraight);
    const showTrunk =
      measures.busiestEdge !== null &&
      measures.busiestMidpoint !== null &&
      options.iterations > 0 &&
      !options.showRides &&
      !options.compareStraight;
    if (showTrunk) {
      const edge = measures.busiestEdge as number;
      ctx.setAnnotations('trunk', [
        {
          kind: 'note',
          id: 'busiest-trunk',
          coordinate: measures.busiestMidpoint as [number, number],
          title: liveText('{rides:integer} rides', {rides: allEdges.rides[edge]}),
          text: `${getStationLabel(flows.names[allEdges.a[edge]])} and ${getStationLabel(flows.names[allEdges.b[edge]])}`,
          tone: 'accent',
          priority: 8
        }
      ]);
    }
    trunkShown = showTrunk;
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
  const concurrentSeconds = Array.from(concurrent, (_, minute) => minute * 60);
  let lastChartMinute = -1;
  let statusStale = true;

  function updateRideChart(playhead: number): void {
    const minute = Math.round(playhead / 60);
    if (minute === lastChartMinute) return;
    lastChartMinute = minute;
    ctx.setChart('ridesChart', {
      kind: 'timeline',
      x: concurrentSeconds,
      y: concurrent,
      mode: 'area',
      playhead,
      xDomain: [0, RIDE_WINDOW_SECONDS],
      xLabel: 'Montreal local time',
      yLabel: 'rides in progress',
      formatX: formatRideClock,
      formatY: formatCompact,
      link: {option: 'time', label: formatRideClock},
      height: 110,
      description:
        'Routed BIXI rides in progress each minute between 07:30 and 10:00 on 15 August 2024; the marker is the playhead. Rides already under way at 07:30 are counted from the start.'
    });
  }

  ctx.setReadout('rides', trackCount);
  ctx.setReadout('streetDistance', STREET_DISTANCE_METERS);
  ctx.setReadout(
    'eligibleCoverage',
    stationToStationRides > 0 ? eligiblePairRides / stationToStationRides : null
  );
  ctx.setReadout(
    'sourceAudit',
    `${formatCount(flows.totalRides)} retained rides after ${formatCount(sourceProperties.droppedShortRides ?? 0)} trips under 60 seconds were dropped; ${formatCount(flows.sameStationRides)} same-station rides do not form a pair. Bundling starts from ${formatCount(allEdges.count)} undirected pairs whose directed source pair had at least ${formatCount(sourceProperties.pairMin ?? 3)} rides; the drawn share still uses every retained inter-station ride as its denominator.`
  );
  ctx.setReadout(
    'routeEvidence',
    `${formatCount(trackCount)} OSRM bicycle-profile routes from one morning (15 August, 07:30–10:00), not GPS traces and not a complete street inventory.`
  );
  ctx.setFurniture({
    title: {
      sample: `${formatCount(edgeCount)} busiest station pairs of ${formatCount(allEdges.count)}, August 2024`
    }
  });
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
        case 'radiusRing':
          updateKernelGeometry();
          break;
        case 'compareStraight':
          // The busiest-trunk note is only drawn on the bundled map, not across a divider.
          markStatsStale();
          ctx.requestLayers();
          break;
        case 'showRides':
          // The off-street shares are only measured while the rides are shown.
          updateKernelGeometry();
          markStatsStale();
          statusStale = true;
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

    onGroundChange() {
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendClasses = classes;
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
      const best = partner[station];
      return {
        title: getStationLabel(flows.names[station]),
        subtitle: flows.boroughNames[flows.borough[station]],
        rows: [
          {
            label: 'Departures, August 2024',
            value: formatCount(flows.departures[station]),
            unit: 'rides',
            emphasis: true
          },
          {label: 'Arrivals', value: formatCount(flows.arrivals[station]), unit: 'rides'},
          ...(best >= 0
            ? [
                {
                  label: 'Busiest pair',
                  value: getStationLabel(flows.names[best]),
                  unit: `${formatCount(partnerRides[station])} rides`
                }
              ]
            : [])
        ]
      };
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
      const ground = ctx.ground();
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const crossingMode = options.colorBy === 'crossing';
      const palette = crossingMode
        ? getCrossingPalette(ground)
        : getRideClassTable(ground).colors.map(
            color => [color[0], color[1], color[2], color[3] ?? 255] as const
          );
      const ribbon = {
        paths: bundling.paths,
        pointsPerPath: bundling.pointsPerEdge,
        pathCount: edgeCount,
        values: rideValues,
        classes: crossingMode ? crossingClasses : rideClasses,
        edgeMask: mask,
        palette,
        maximumValue: maximumRides,
        widthMinPixels: TRUNK_WIDTH_MIN_PIXELS,
        widthMaxPixels: TRUNK_WIDTH_MAX_PIXELS,
        widthByValue: true,
        highlightClasses: crossingMode ? null : legendClasses
      };
      const layers: Layer[] = [];
      if (options.showBoroughs && outlineCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-borough-outline',
            coordinateOrigin: outlineOrigin,
            segments: outlineBuffer,
            instanceCount: outlineCount,
            widthPixels: 0.5,
            color: inkFor(CONTEXT_INK, ground)
          })
        );
      }
      if (options.showStraight && !options.compareStraight) {
        layers.push(
          new BundleRibbonLayer({
            ...ribbon,
            id: `bixi-straight-${bundling.pointsPerEdge}`,
            straight: true,
            flatColor: inkFor(GHOST_INK, ground),
            widthByValue: false,
            widthMinPixels: GHOST_WIDTH_PIXELS,
            heaviestLast: false,
            highlightClasses: null
          })
        );
      }
      if (options.compareStraight) {
        // Same widths, same classes, same legend: only the geometry differs across the divider.
        layers.push(
          new BundleRibbonLayer({
            ...ribbon,
            id: `bixi-compare-straight-${bundling.pointsPerEdge}`,
            straight: true,
            compareSide: 'a'
          }),
          new BundleRibbonLayer({
            ...ribbon,
            id: `bixi-compare-bundled-${bundling.pointsPerEdge}`,
            compareSide: 'b'
          })
        );
      } else if (options.showRides) {
        layers.push(
          new BundleRibbonLayer({
            ...ribbon,
            id: `bixi-bundles-demoted-${bundling.pointsPerEdge}`,
            flatColor: getDemotedBundleInk(ground),
            widthByValue: false,
            widthMinPixels: DEMOTED_BUNDLE_WIDTH_PIXELS,
            highlightClasses: null
          })
        );
      } else {
        layers.push(
          new BundleRibbonLayer({...ribbon, id: `bixi-bundles-${bundling.pointsPerEdge}`})
        );
      }
      if (options.showStations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-bundle-stations',
            coordinateSystem: lngLat,
            positions,
            instanceCount: stationCount,
            radiusPixels: STATION_RADIUS_PIXELS,
            color: inkFor(STATION_INK, ground)
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bundle-hubs',
            coordinateSystem: lngLat,
            positions,
            ids: hubs,
            instanceCount: hubRows.length,
            shape: 'ring',
            radiusPixels: HUB_RING_RADIUS_PIXELS,
            outlineWidthPixels: 1.5,
            color: inkFor(HUB_RING_INK, ground)
          })
        );
      }
      if (options.showRides) {
        layers.push(
          new RideTrailLayer({
            id: 'bixi-ride-trails',
            coordinateOrigin,
            segments: rideSegments,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            color: inkFor(RIDE_TRAIL_INK, ground),
            widthPixels: RIDE_TRAIL_WIDTH_PIXELS
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bike-halo',
            coordinateOrigin,
            ids: activeIds,
            positions: currentPositions,
            drawCommands: bikeDraw,
            radiusPixels: BIKE_RADIUS_PIXELS + 1.4,
            color: BIKE_HALO
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-bikes',
            coordinateOrigin,
            ids: activeIds,
            positions: currentPositions,
            drawCommands: bikeDraw,
            radiusPixels: BIKE_RADIUS_PIXELS,
            ...(options.bikeSpeed
              ? {
                  values: speeds,
                  valueFormat: 'float32' as const,
                  colormap: 'magma' as const,
                  valueRange: BIKE_SPEED_EXTENT,
                  rampRange: BIKE_SPEED_RANGE,
                  reverseRamp: directionFor(ground, 'magma').reverse
                }
              : {color: BIKE_INK})
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
