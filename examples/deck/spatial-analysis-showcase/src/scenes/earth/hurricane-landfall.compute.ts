// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {HURRICANE_CLASS} from '../../cartography/hue-registry';
import type {Buffer} from '@luma.gl/core';
import {
  GPUDistanceField,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  getGPUDistanceFieldParameterValues
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPULineDensityParameterValues,
  GPULineDensity,
  GPUZoneEvents,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPU_ZONE_EVENT_TYPE
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildZoneSet, rasterizeZones, type ZoneSet} from '../movement/b12-zones';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {
  getCategoryOfWind,
  getStormLabel,
  HURRICANE_CATEGORIES,
  loadHurricaneTracks
} from './hurricane-data';

/** Option state of the hurricane landfall scene. */
export type HurricaneLandfallOptions = {
  view: 'density' | 'coast' | 'landfalls' | 'none';
  intensity: 'all' | 'storm' | 'hurricane' | 'major' | 'weighted';
  ramp: 'magma' | 'inferno' | 'cividis';
  sqrtScale: boolean;
  rasterOpacity: number;
  coastSearchKm: number;
  distanceMode: 'exact' | 'jump-flood';
  stateMetric: 'count' | 'strongest';
  landfallCounting: 'every-crossing' | 'first-per-storm';
  minimumLandfallWind: number;
  eventsPerStorm: '4' | '8' | '16' | '32';
  showTracks: boolean;
  trackOpacity: number;
  showLandfalls: boolean;
  showCoastline: boolean;
  showStateOutlines: boolean;
};

/** Origin of the shared planar system (deck.gl meters) of storms, coast, states and grids. */
const ORIGIN: readonly [number, number] = [-56, 28];
const COLUMNS = 360;
const ROWS = 232;
const CHOROPLETH_SCALE = 2;
/** Window of the analysis grids, `[west, south, east, north]`. */
const GRID_WINDOW = [-100, 8, -8, 50] as const;
/** Wind thresholds of the four density layers (knots): all fixes, tropical storm, hurricane, major. */
const INTENSITY_THRESHOLDS = [0, 34, 64, 96] as const;
const INTENSITY_INDEX: Record<HurricaneLandfallOptions['intensity'], number> = {
  all: 0,
  storm: 1,
  hurricane: 2,
  major: 3,
  weighted: 4
};
const CANDIDATE_CAPACITY = 1 << 19;
const MAXIMUM_EVENTS_PER_STORM = 32;
const SEED_SPACING_METERS = 11000;
const BACK_STEP_METERS = 6000;
const COAST_NEAR_KM = 100;

type EventsVariant = {
  key: string;
  eventsPerStorm: number;
  compiled: CompiledGPUCommandGraph<void>;
  encoded: boolean;
};

type Landfall = {
  track: number;
  zone: number;
  wind: number;
  x: number;
  y: number;
};

/**
 * Hurricane landfall: four `GPULineDensity` layers (all fixes, tropical storm, hurricane and major
 * hurricane strength) show where tracks pass, a `GPUDistanceField` from the Natural Earth coastline
 * gives the distance to the nearest coast of every cell, and `GPUZoneEvents` finds every entry of
 * every storm into a US state. The CPU keeps the entries that come from open water, interpolates
 * the wind at the crossing and counts them per state.
 *
 * Everything is planar in deck.gl meters about 28 N, 56 W so the rasters line up with the map. At
 * these latitudes a meter in that system is not a true meter, so a small kernel multiplies each
 * row by cos(latitude) / cos(28 N) before lengths, areas and distances are displayed.
 */
export async function createHurricaneLandfall(
  ctx: SceneContext<HurricaneLandfallOptions>
): Promise<SceneInstance<HurricaneLandfallOptions>> {
  const storms = loadHurricaneTracks(ctx.datasets.get('ibtracs-north-atlantic'), {
    projection: 'mercator',
    origin: ORIGIN,
    timeBase: 'storm'
  });
  const statesDataset = ctx.datasets.get('us-states');
  const coastDataset = ctx.datasets.get('naturalearth-atlantic-coast');
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = storms;
  const resources = new SpatialAnalysisResources(device, 'hurricane-landfall');
  const coordinateOrigin: [number, number, number] = [ORIGIN[0], ORIGIN[1], 0];
  const lngLatProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const cellCount = COLUMNS * ROWS;

  // ---- Grid ----------------------------------------------------------------------------------
  const corners = [
    storms.project(GRID_WINDOW[0], GRID_WINDOW[1]),
    storms.project(GRID_WINDOW[2], GRID_WINDOW[1]),
    storms.project(GRID_WINDOW[0], GRID_WINDOW[3]),
    storms.project(GRID_WINDOW[2], GRID_WINDOW[3])
  ];
  const bounds: [number, number, number, number] = [
    Math.min(...corners.map(corner => corner[0])),
    Math.min(...corners.map(corner => corner[1])),
    Math.max(...corners.map(corner => corner[0])),
    Math.max(...corners.map(corner => corner[1]))
  ];
  const cellWidth = (bounds[2] - bounds[0]) / COLUMNS;
  const cellHeight = (bounds[3] - bounds[1]) / ROWS;
  const rowScale = new Float32Array(ROWS);
  for (let row = 0; row < ROWS; row++) {
    const [, latitude] = storms.unproject(
      (bounds[0] + bounds[2]) / 2,
      bounds[1] + (row + 0.5) * cellHeight
    );
    rowScale[row] = Math.cos((latitude * Math.PI) / 180) / Math.cos((ORIGIN[1] * Math.PI) / 180);
  }
  const rowScaleBuffer = resources.createBuffer('row-scale', rowScale);

  // ---- States --------------------------------------------------------------------------------
  const zones: ZoneSet = buildZoneSet(statesDataset.geojson!, storms.project);
  const zoneCount = zones.zoneCount;
  const edgeCount = zones.edgeZones.length;
  const maximumZoneEdges = (() => {
    const counts = new Uint32Array(zoneCount);
    for (const zone of zones.edgeZones) counts[zone]++;
    return Math.max(...counts);
  })();
  const zoneRaster = rasterizeZones(
    zones,
    bounds,
    COLUMNS * CHOROPLETH_SCALE,
    ROWS * CHOROPLETH_SCALE
  );
  const zoneRasterBuffer = resources.createBuffer('zone-raster', zoneRaster);
  const outlineBuffer = resources.createBuffer('state-outlines', zones.outlineSegments);
  const selectedOutline = resources.createBuffer(
    'selected-state',
    new Float32Array(maximumZoneEdges * 4).fill(Number.NaN)
  );
  const zoneBounds = Array.from({length: zoneCount}, (_, zone) => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const ring of zones.zoneRings[zone]) {
      for (let index = 0; index < ring.length; index += 2) {
        minX = Math.min(minX, ring[index]);
        maxX = Math.max(maxX, ring[index]);
        minY = Math.min(minY, ring[index + 1]);
        maxY = Math.max(maxY, ring[index + 1]);
      }
    }
    return [minX, minY, maxX, maxY] as const;
  });
  /** Even-odd test of a point against all rings of one state. */
  const isInsideState = (zone: number, x: number, y: number): boolean => {
    let inside = false;
    for (const ring of zones.zoneRings[zone]) {
      const count = ring.length / 2;
      for (let index = 0, previous = count - 1; index < count; previous = index++) {
        const x0 = ring[index * 2];
        const y0 = ring[index * 2 + 1];
        const x1 = ring[previous * 2];
        const y1 = ring[previous * 2 + 1];
        if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
      }
    }
    return inside;
  };
  /** State containing a point (the smallest where states overlap), or -1. */
  const findStateAt = (x: number, y: number): number => {
    let best = -1;
    for (let zone = 0; zone < zoneCount; zone++) {
      const box = zoneBounds[zone];
      if (x < box[0] || x > box[2] || y < box[1] || y > box[3]) continue;
      if (best >= 0 && zones.areas[zone] >= zones.areas[best]) continue;
      if (isInsideState(zone, x, y)) best = zone;
    }
    return best;
  };

  // ---- Coast ---------------------------------------------------------------------------------
  const coastLngLat = coastDataset.column<Float32Array>('vertices');
  const coastOffsets = coastDataset.column<Uint32Array>('pathOffsets');
  const seedList: number[] = [];
  const coastSegments: number[] = [];
  for (let path = 0; path < coastOffsets.length - 1; path++) {
    let previous: [number, number] | null = null;
    for (let vertex = coastOffsets[path]; vertex < coastOffsets[path + 1]; vertex++) {
      const point = storms.project(coastLngLat[vertex * 2], coastLngLat[vertex * 2 + 1]);
      if (previous) {
        const length = Math.hypot(point[0] - previous[0], point[1] - previous[1]);
        const steps = Math.max(1, Math.ceil(length / SEED_SPACING_METERS));
        for (let step = 1; step <= steps; step++) {
          seedList.push(
            previous[0] + ((point[0] - previous[0]) * step) / steps,
            previous[1] + ((point[1] - previous[1]) * step) / steps
          );
        }
        coastSegments.push(
          coastLngLat[(vertex - 1) * 2],
          coastLngLat[(vertex - 1) * 2 + 1],
          coastLngLat[vertex * 2],
          coastLngLat[vertex * 2 + 1]
        );
      } else {
        seedList.push(point[0], point[1]);
      }
      previous = point;
    }
  }
  const seedPositions = Float32Array.from(seedList);
  const seedTotal = seedPositions.length / 2;
  const seedBuffer = resources.createBuffer('coast-seeds', seedPositions);
  const coastSegmentBuffer = resources.createBuffer(
    'coast-segments',
    Float32Array.from(coastSegments)
  );
  const coastSegmentCount = coastSegments.length / 4;

  // ---- Static storm buffers ------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', storms.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', storms.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', storms.offsets);
  const segmentsBuffer = resources.createBuffer('segments', storms.segments);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', storms.segmentEndVertices);
  const categoryBuffer = resources.createBuffer('category', storms.category);

  // ---- Line density: one graph, four intensity layers -----------------------------------------
  const densityParameters = resources.createParameterBuffer(
    'density-parameters',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH,
    getGPULineDensityParameterValues({
      minX: bounds[0],
      minY: bounds[1],
      cellWidth,
      cellHeight
    })
  );
  const densityGraph = new GPUCommandGraph<void>(device, {id: 'hurricane-density'});
  const densityParameterView = densityParameters.importToGraph(densityGraph);
  const rowScaleView = importGraphBuffer(
    densityGraph,
    'row-scale',
    rowScaleBuffer,
    'float32',
    ROWS
  );
  const displayBuffers: Buffer[] = [];
  const displayViews: ReturnType<typeof importGraphBuffer<'float32', void>>[] = [];
  const densityFlagBuffers: {overflow: Buffer; total: Buffer}[] = [];
  const densityLengths: Buffer[] = [];
  const unit = 1e7 / storms.seasonCount;
  const cellArea = cellWidth * cellHeight;
  INTENSITY_THRESHOLDS.forEach((threshold, variant) => {
    const positions: number[] = [];
    const offsets: number[] = [0];
    let pieces = 0;
    let run: number[] = [];
    const flush = () => {
      if (run.length >= 4) {
        positions.push(...run);
        offsets.push(positions.length / 2);
        for (let index = 2; index < run.length; index += 2) {
          pieces +=
            Math.abs(run[index] - run[index - 2]) / cellWidth +
            Math.abs(run[index + 1] - run[index - 1]) / cellHeight +
            2;
        }
      }
      run = [];
    };
    for (let track = 0; track < trackCount; track++) {
      for (let vertex = storms.offsets[track]; vertex < storms.offsets[track + 1]; vertex++) {
        if (storms.wind[vertex] >= threshold) {
          run.push(storms.positions[vertex * 2], storms.positions[vertex * 2 + 1]);
        } else flush();
      }
      flush();
    }
    const positionBuffer = resources.createBuffer(
      `density-positions-${variant}`,
      Float32Array.from(positions)
    );
    const offsetBuffer = resources.createBuffer(
      `density-offsets-${variant}`,
      Uint32Array.from(offsets)
    );
    const lengths = resources.createBuffer(`density-lengths-${variant}`, cellCount * 4);
    const display = resources.createBuffer(`density-display-${variant}`, cellCount * 4);
    densityLengths.push(lengths);
    displayBuffers.push(display);
    const overflow = resources.createBuffer(`density-overflow-${variant}`, 4);
    const total = resources.createBuffer(`density-total-${variant}`, 4);
    densityFlagBuffers.push({overflow, total});
    const lengthsView = importGraphBuffer(
      densityGraph,
      `density-lengths-${variant}`,
      lengths,
      'float32',
      cellCount
    );
    densityGraph.add(
      new GPULineDensity({
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        id: `density-${variant}`,
        positions: importGraphBuffer(
          densityGraph,
          `density-positions-${variant}`,
          positionBuffer,
          'float32x2',
          positions.length / 2
        ),
        pathOffsets: importGraphBuffer(
          densityGraph,
          `density-offsets-${variant}`,
          offsetBuffer,
          'uint32',
          offsets.length
        ),
        columns: COLUMNS,
        rows: ROWS,
        maximumRecords: Math.ceil(pieces * 1.1) + 1024,
        parameters: densityParameterView,
        output: {
          lengths: lengthsView,
          overflow: importGraphBuffer(
            densityGraph,
            `density-overflow-${variant}`,
            overflow,
            'uint32',
            1
          ),
          totalRecords: importGraphBuffer(
            densityGraph,
            `density-total-${variant}`,
            total,
            'uint32',
            1
          )
        }
      })
    );
    const displayView = importGraphBuffer(
      densityGraph,
      `density-display-${variant}`,
      display,
      'float32',
      cellCount
    );
    displayViews.push(displayView);
    addKernelPass(densityGraph, {
      id: `density-display-${variant}`,
      invocationCount: cellCount,
      declarations: `const COLUMNS: u32 = ${COLUMNS}u;
const AREA_UNIT: f32 = ${Math.fround(unit / cellArea)};`,
      bindings: [
        {name: 'lengths', view: lengthsView, type: 'f32', access: 'read'},
        {name: 'rowScale', view: rowScaleView, type: 'f32', access: 'read'},
        {
          name: 'display',
          view: displayView,
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let row = index / COLUMNS;
  display[displayOffset + index] = lengths[lengthsOffset + index] * AREA_UNIT / rowScale[rowScaleOffset + row];`
    });
  });
  // The weighted layer adds the four layers: a fix counts once for each wind threshold it reaches.
  const weightedDisplay = resources.createBuffer('density-display-weighted', cellCount * 4);
  displayBuffers.push(weightedDisplay);
  addKernelPass(densityGraph, {
    id: 'density-weighted',
    invocationCount: cellCount,
    bindings: [
      ...displayViews.map((view, variant) => ({
        name: `layer${variant}`,
        view,
        type: 'f32' as const,
        access: 'read' as const
      })),
      {
        name: 'total',
        view: importGraphBuffer(
          densityGraph,
          'density-weighted',
          weightedDisplay,
          'float32',
          cellCount
        ),
        type: 'f32' as const,
        access: 'read_write' as const
      }
    ],
    body: `total[totalOffset + index] = layer0[layer0Offset + index] + layer1[layer1Offset + index] + layer2[layer2Offset + index] + layer3[layer3Offset + index];`
  });
  const densityCompiled = resources.track(densityGraph.compile());

  // ---- Distance to the coast ---------------------------------------------------------------------
  const distanceSettings = resources.createParameterBuffer(
    'distance-settings',
    'float32',
    GPU_DISTANCE_FIELD_PARAMETER_LENGTH
  );
  const seedCountParameter = resources.createParameterBuffer(
    'seed-count',
    'uint32',
    1,
    Uint32Array.of(seedTotal)
  );
  const distancesBuffer = resources.createBuffer('coast-distances', cellCount * 4);
  const coastKmBuffer = resources.createBuffer('coast-km', cellCount * 4);
  function buildCoastGraph(mode: 'exact' | 'jump-flood'): CompiledGPUCommandGraph<void> {
    const graph = new GPUCommandGraph<void>(device, {id: `hurricane-coast-${mode}`});
    const distancesView = importGraphBuffer(
      graph,
      'coast-distances',
      distancesBuffer,
      'float32',
      cellCount
    );
    graph.add(
      new GPUDistanceField({
        id: `coast-${mode}`,
        width: COLUMNS,
        height: ROWS,
        mode,
        settings: distanceSettings.importToGraph(graph),
        seedPositions: importGraphBuffer(graph, 'coast-seeds', seedBuffer, 'float32x2', seedTotal),
        seedCount: seedCountParameter.importToGraph(graph),
        output: {distances: distancesView}
      })
    );
    addKernelPass(graph, {
      id: `coast-km-${mode}`,
      invocationCount: cellCount,
      declarations: `const COLUMNS: u32 = ${COLUMNS}u;`,
      bindings: [
        {name: 'distances', view: distancesView, type: 'f32', access: 'read'},
        {
          name: 'rowScale',
          view: importGraphBuffer(graph, 'row-scale', rowScaleBuffer, 'float32', ROWS),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'kilometers',
          view: importGraphBuffer(graph, 'coast-km', coastKmBuffer, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let row = index / COLUMNS;
  kilometers[kilometersOffset + index] = distances[distancesOffset + index] * rowScale[rowScaleOffset + row] * 0.001;`
    });
    return resources.track(graph.compile());
  }
  const coastGraphs = {
    exact: buildCoastGraph('exact'),
    'jump-flood': buildCoastGraph('jump-flood')
  };

  // ---- Landfall events ------------------------------------------------------------------------
  const eventCapacity = trackCount * MAXIMUM_EVENTS_PER_STORM;
  const eventTracks = resources.createBuffer('event-tracks', eventCapacity * 4);
  const eventZones = resources.createBuffer('event-zones', eventCapacity * 4);
  const eventTypes = resources.createBuffer('event-types', eventCapacity * 4);
  const eventTimes = resources.createBuffer('event-times', eventCapacity * 4);
  const eventRows = resources.createBuffer('event-rows', eventCapacity * 4);
  const eventPositions = resources.createBuffer('event-positions', eventCapacity * 8);
  const eventCountBuffer = resources.createBuffer('event-count', 4);
  const eventOverflowBuffer = resources.createBuffer('event-overflow', 4);
  const candidateCountBuffer = resources.createBuffer('candidate-count', 4);
  const candidateOverflowBuffer = resources.createBuffer('candidate-overflow', 4);
  const trackOverflowBuffer = resources.createBuffer('track-overflow', 4);
  const eventListOverflowBuffer = resources.createBuffer('event-list-overflow', 4);
  const edgeStartsBuffer = resources.createBuffer('edge-starts', zones.edgeStarts);
  const edgeEndsBuffer = resources.createBuffer('edge-ends', zones.edgeEnds);
  const edgeZonesBuffer = resources.createBuffer('edge-zones', zones.edgeZones);

  function buildEvents(eventsPerStorm: number): EventsVariant {
    const key = `events-${eventsPerStorm}`;
    const capacity = trackCount * eventsPerStorm;
    const graph = new GPUCommandGraph<void>(device, {id: `hurricane-${key}`});
    graph.add(
      new GPUZoneEvents({
        id: 'landfalls',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
        edgeStarts: importGraphBuffer(
          graph,
          'edge-starts',
          edgeStartsBuffer,
          'float32x2',
          edgeCount
        ),
        edgeEnds: importGraphBuffer(graph, 'edge-ends', edgeEndsBuffer, 'float32x2', edgeCount),
        edgeZones: importGraphBuffer(graph, 'edge-zones', edgeZonesBuffer, 'uint32', edgeCount),
        zoneCount,
        candidateCapacity: CANDIDATE_CAPACITY,
        maxEventsPerTrack: eventsPerStorm,
        events: {
          output: {
            ids: importGraphBuffer(graph, 'event-tracks', eventTracks, 'uint32', capacity),
            count: importGraphBuffer(graph, 'event-count', eventCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'event-overflow', eventOverflowBuffer, 'uint32', 1)
          },
          eventZones: importGraphBuffer(graph, 'event-zones-out', eventZones, 'uint32', capacity),
          eventTypes: importGraphBuffer(graph, 'event-types', eventTypes, 'uint32', capacity),
          eventTimes: importGraphBuffer(graph, 'event-times', eventTimes, 'float32', capacity),
          eventRows: importGraphBuffer(graph, 'event-rows', eventRows, 'uint32', capacity),
          eventPositions: importGraphBuffer(
            graph,
            'event-positions',
            eventPositions,
            'float32x2',
            capacity
          )
        },
        diagnostics: {
          candidateCount: importGraphBuffer(
            graph,
            'candidate-count',
            candidateCountBuffer,
            'uint32',
            1
          ),
          candidateOverflow: importGraphBuffer(
            graph,
            'candidate-overflow',
            candidateOverflowBuffer,
            'uint32',
            1
          ),
          trackOverflow: importGraphBuffer(
            graph,
            'track-overflow',
            trackOverflowBuffer,
            'uint32',
            1
          ),
          eventOverflow: importGraphBuffer(
            graph,
            'event-list-overflow',
            eventListOverflowBuffer,
            'uint32',
            1
          )
        }
      })
    );
    return {key, eventsPerStorm, compiled: resources.track(graph.compile()), encoded: false};
  }

  // ---- State ---------------------------------------------------------------------------------
  let destroyed = false;
  let eventsVariant: EventsVariant | null = null;
  let eventsDirty = true;
  let coastDirty = true;
  let densityEncoded = false;
  let coastKm: Float32Array | null = null;
  let displayMaxima = new Float32Array(5);
  let eventSnapshot: {
    count: number;
    candidates: number;
    candidateOverflow: boolean;
    trackOverflow: boolean;
    listOverflow: boolean;
    tracks: Uint32Array;
    zonesOut: Uint32Array;
    types: Uint32Array;
    times: Float32Array;
    rows: Uint32Array;
    positions: Float32Array;
  } | null = null;
  let landfalls: Landfall[] = [];
  let stateCounts = new Float64Array(zoneCount);
  let stateStrongest = new Float64Array(zoneCount);
  let stateStrongestStorm = new Int32Array(zoneCount).fill(-1);
  let selectedState = -1;
  let dotCount = 0;
  let landfallMaximum = 1;

  const stateDisplayBuffer = resources.createBuffer('state-display', (zoneCount + 1) * 4);
  const dotPositions = resources.createBuffer('landfall-positions', eventCapacity * 8);
  const dotCategories = resources.createBuffer('landfall-categories', eventCapacity * 4);

  function replaceEvents(eventsPerStorm: number): void {
    const previous = eventsVariant;
    eventsVariant = buildEvents(eventsPerStorm);
    if (previous) resources.release(previous.compiled);
    eventsDirty = true;
  }

  // ---- Readers -------------------------------------------------------------------------------
  const densityReader = new SummaryReader(
    resources,
    'hurricane-density',
    [
      ...displayBuffers.map(buffer => ({buffer, size: cellCount * 4})),
      ...densityFlagBuffers.flatMap(({overflow, total}) => [
        {buffer: overflow, size: 4},
        {buffer: total, size: 4}
      ])
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      displayMaxima = new Float32Array(5);
      for (let layer = 0; layer < 5; layer++) {
        let maximum = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          const value = floats[layer * cellCount + cell];
          if (Number.isFinite(value) && value > maximum) maximum = value;
        }
        displayMaxima[layer] = maximum;
      }
      let overflowed = false;
      let pieces = 0;
      for (let layer = 0; layer < 4; layer++) {
        overflowed ||= words[5 * cellCount + layer * 2] !== 0;
        pieces += words[5 * cellCount + layer * 2 + 1];
      }
      ctx.setReadout(
        'density',
        `${formatCount(pieces)} segment-cell pieces in four layers${overflowed ? ', capacity exceeded: lengths are low' : ', no overflow'}`
      );
      updateLegendExtents();
    }
  );

  const coastReader = new SummaryReader(
    resources,
    'hurricane-coast',
    [{buffer: coastKmBuffer, size: cellCount * 4}],
    bytes => {
      if (destroyed) return;
      coastKm = new Float32Array(bytes).slice();
      updateCoastChart();
    }
  );

  const eventReader = new SummaryReader(
    resources,
    'hurricane-events',
    [
      {buffer: eventCountBuffer, size: 4},
      {buffer: candidateCountBuffer, size: 4},
      {buffer: candidateOverflowBuffer, size: 4},
      {buffer: trackOverflowBuffer, size: 4},
      {buffer: eventListOverflowBuffer, size: 4},
      {buffer: eventOverflowBuffer, size: 4},
      {buffer: eventTracks, size: eventCapacity * 4},
      {buffer: eventZones, size: eventCapacity * 4},
      {buffer: eventTypes, size: eventCapacity * 4},
      {buffer: eventRows, size: eventCapacity * 4},
      {buffer: eventTimes, size: eventCapacity * 4},
      {buffer: eventPositions, size: eventCapacity * 8}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const head = 6;
      eventSnapshot = {
        count: words[0],
        candidates: words[1],
        candidateOverflow: words[2] !== 0,
        trackOverflow: words[3] !== 0,
        listOverflow: words[4] !== 0 || words[5] !== 0,
        tracks: words.slice(head, head + eventCapacity),
        zonesOut: words.slice(head + eventCapacity, head + 2 * eventCapacity),
        types: words.slice(head + 2 * eventCapacity, head + 3 * eventCapacity),
        rows: words.slice(head + 3 * eventCapacity, head + 4 * eventCapacity),
        times: floats.slice(head + 4 * eventCapacity, head + 5 * eventCapacity),
        positions: floats.slice(head + 5 * eventCapacity, head + 7 * eventCapacity)
      };
      processEvents();
    }
  );

  // ---- CPU post-processing: from entries to landfalls ------------------------------------------
  function processEvents(): void {
    const snapshot = eventSnapshot;
    if (!snapshot) return;
    const capacity = (eventsVariant?.eventsPerStorm ?? 16) * trackCount;
    const listed = Math.min(snapshot.count, capacity);
    const found: Landfall[] = [];
    for (let event = 0; event < listed; event++) {
      if (snapshot.types[event] !== GPU_ZONE_EVENT_TYPE.enter) continue;
      const track = snapshot.tracks[event];
      const zone = snapshot.zonesOut[event];
      const row = snapshot.rows[event];
      if (zone >= zoneCount || row === 0xffffffff || row <= storms.offsets[track]) continue;
      const x = snapshot.positions[event * 2];
      const y = snapshot.positions[event * 2 + 1];
      // An entry from another state is an inland border crossing. Step back along the segment: a
      // landfall comes from open water (or from a country outside the state set).
      const x0 = storms.positions[(row - 1) * 2];
      const y0 = storms.positions[(row - 1) * 2 + 1];
      const x1 = storms.positions[row * 2];
      const y1 = storms.positions[row * 2 + 1];
      const length = Math.hypot(x1 - x0, y1 - y0) || 1;
      const back = Math.min(BACK_STEP_METERS, length * 0.5);
      if (findStateAt(x - ((x1 - x0) / length) * back, y - ((y1 - y0) / length) * back) >= 0) {
        continue;
      }
      const t0 = storms.timestamps[row - 1];
      const t1 = storms.timestamps[row];
      const fraction =
        t1 > t0 ? Math.min(1, Math.max(0, (snapshot.times[event] - t0) / (t1 - t0))) : 0;
      const wind = storms.wind[row - 1] * (1 - fraction) + storms.wind[row] * fraction;
      found.push({track, zone, wind, x, y});
    }
    found.sort((a, b) => a.track - b.track);
    const firstOnly = new Set<number>();
    landfalls = found.filter(entry => {
      if (ctx.options.landfallCounting === 'every-crossing') return true;
      if (firstOnly.has(entry.track)) return false;
      firstOnly.add(entry.track);
      return true;
    });
    // Events are ordered by track, then time, so the first kept entry of a track is its first landfall.
    ctx.setReadout(
      'events',
      `${formatCount(snapshot.count)} state entries found, ${formatCount(found.length)} from open water${snapshot.trackOverflow || snapshot.listOverflow ? ', list truncated: raise Events kept per storm' : ''}`
    );
    ctx.setReadout(
      'candidates',
      `${formatCount(snapshot.candidates)} of ${formatCount(CANDIDATE_CAPACITY)} segment-edge candidates${snapshot.candidateOverflow ? ' (overflow: events may be missing)' : ''}`
    );
    updateLandfallViews();
  }

  function updateLandfallViews(): void {
    const options = ctx.options;
    const kept = landfalls.filter(entry => entry.wind >= options.minimumLandfallWind);
    stateCounts = new Float64Array(zoneCount);
    stateStrongest = new Float64Array(zoneCount);
    stateStrongestStorm = new Int32Array(zoneCount).fill(-1);
    const stormsWithLandfall = new Set<number>();
    let strongest: Landfall | null = null;
    for (const entry of kept) {
      stateCounts[entry.zone]++;
      stormsWithLandfall.add(entry.track);
      if (entry.wind > stateStrongest[entry.zone]) {
        stateStrongest[entry.zone] = entry.wind;
        stateStrongestStorm[entry.zone] = entry.track;
      }
      if (!strongest || entry.wind > strongest.wind) strongest = entry;
    }
    const display = new Float32Array(zoneCount + 1).fill(Number.NaN);
    let maximum = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      const value = options.stateMetric === 'count' ? stateCounts[zone] : stateStrongest[zone];
      display[zone] = value > 0 ? value : Number.NaN;
      maximum = Math.max(maximum, value);
    }
    stateDisplayBuffer.write(display);
    landfallMaximum = Math.max(1, maximum);
    ctx.setLegendData('landfallMaximum', landfallMaximum);
    // Landfall dots: positions and class of the crossing wind.
    const positions = new Float32Array(eventCapacity * 2).fill(Number.NaN);
    const categories = new Uint32Array(eventCapacity);
    kept.slice(0, eventCapacity).forEach((entry, index) => {
      positions[index * 2] = entry.x;
      positions[index * 2 + 1] = entry.y;
      categories[index] = getCategoryOfWind(entry.wind);
    });
    dotPositions.write(positions);
    dotCategories.write(categories);
    dotCount = Math.min(kept.length, eventCapacity);

    ctx.setReadout(
      'landfalls',
      `${formatCount(kept.length)} landfalls by ${formatCount(stormsWithLandfall.size)} of ${formatCount(trackCount)} storms`
    );
    ctx.setReadout(
      'strongest',
      strongest
        ? `${getStormLabel(storms, strongest.track)}: ${strongest.wind.toFixed(0)} kt (${HURRICANE_CATEGORIES[getCategoryOfWind(strongest.wind)]}) entering ${zones.names[strongest.zone]}`
        : 'none'
    );
    // Charts.
    const ranked = Array.from({length: zoneCount}, (_, zone) => zone)
      .filter(zone => stateCounts[zone] > 0)
      .sort((a, b) => stateCounts[b] - stateCounts[a])
      .slice(0, 12);
    ctx.setChart(
      'stateChart',
      ranked.length
        ? {
            kind: 'bars',
            values: ranked.map(zone => stateCounts[zone]),
            labels: ranked.map(zone => stateAbbreviations[zone] ?? zones.names[zone].slice(0, 3)),
            highlight:
              selectedState >= 0 && ranked.includes(selectedState)
                ? [ranked.indexOf(selectedState)]
                : [0],
            height: 140,
            yLabel: options.landfallCounting === 'first-per-storm' ? 'storms' : 'landfalls',
            formatY: value => value.toFixed(0),
            description:
              'Landfalls per state from open water, the twelve states with the most, at the chosen minimum wind.'
          }
        : null
    );
    describeSelectedState();
    ctx.requestLayers();
  }

  const stateAbbreviations: readonly string[] =
    (statesDataset.properties.abbr as string[] | undefined) ?? [];

  // ---- Charts that depend on the intensity -----------------------------------------------------
  function updateCoastChart(): void {
    if (!coastKm) return;
    const options = ctx.options;
    const threshold = INTENSITY_THRESHOLDS[Math.min(3, INTENSITY_INDEX[options.intensity])];
    const distances: number[] = [];
    let near = 0;
    let beyond = 0;
    const windowWidth = bounds[2] - bounds[0];
    const windowHeight = bounds[3] - bounds[1];
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      if (storms.wind[vertex] < Math.max(threshold, 1)) continue;
      const column = Math.floor(
        ((storms.positions[vertex * 2] - bounds[0]) / windowWidth) * COLUMNS
      );
      const row = Math.floor(
        ((storms.positions[vertex * 2 + 1] - bounds[1]) / windowHeight) * ROWS
      );
      if (column < 0 || column >= COLUMNS || row < 0 || row >= ROWS) continue;
      const value = coastKm[row * COLUMNS + column];
      if (!Number.isFinite(value)) {
        beyond++;
        continue;
      }
      distances.push(value);
      if (value <= COAST_NEAR_KM) near++;
    }
    const total = distances.length + beyond;
    ctx.setReadout(
      'nearCoast',
      total
        ? `${((100 * near) / total).toFixed(0)}% of ${formatCount(total)} fixes within ${COAST_NEAR_KM} km of a coast${beyond ? `, ${formatCount(beyond)} beyond the ${options.coastSearchKm} km search limit` : ''}`
        : 'n/a'
    );
    const limit = Math.max(200, options.coastSearchKm);
    ctx.setChart(
      'coastChart',
      histogramChart(binValues(distances, 0, limit, 30), 0, limit, {
        xLabel: 'distance from a fix to the nearest coast (km)',
        yLabel: 'fixes',
        markers: [{x: COAST_NEAR_KM, label: `${COAST_NEAR_KM} km`}],
        formatX: value => `${Math.round(value / 50) * 50}`,
        description:
          'Histogram of the distance to the nearest coast at every fix of the chosen intensity, read from the GPU distance field.'
      })
    );
  }

  function updateLegendExtents(): void {
    const options = ctx.options;
    ctx.setLegendExtent('density', [
      0,
      Math.max(1e-6, displayMaxima[INTENSITY_INDEX[options.intensity]])
    ]);
    ctx.requestLayers();
  }

  function describeSelectedState(): void {
    if (selectedState < 0) {
      ctx.setReadout('selectedState', 'click a state');
      return;
    }
    ctx.setReadout('selectedState', describeState(selectedState));
  }

  function describeState(zone: number): string {
    const count = stateCounts[zone];
    if (!count) return `${zones.names[zone]}: no landfall at this minimum wind`;
    const storm = stateStrongestStorm[zone];
    return `${zones.names[zone]}: ${formatCount(count)} landfalls, strongest ${stateStrongest[zone].toFixed(0)} kt (${storm >= 0 ? getStormLabel(storms, storm) : 'n/a'})`;
  }

  function writeSelectedOutline(): void {
    const outline = new Float32Array(maximumZoneEdges * 4).fill(Number.NaN);
    if (selectedState >= 0) {
      let row = 0;
      for (let edge = 0; edge < edgeCount; edge++) {
        if (zones.edgeZones[edge] !== selectedState) continue;
        outline.set(zones.outlineSegments.subarray(edge * 4, edge * 4 + 4), row * 4);
        row++;
      }
    }
    selectedOutline.write(outline);
  }

  function writeCoastSettings(): void {
    distanceSettings.write(
      getGPUDistanceFieldParameterValues({
        bounds,
        gridSize: [COLUMNS, ROWS],
        maxDistance: ctx.options.coastSearchKm * 1000
      })
    );
    coastDirty = true;
  }

  // ---- Initial work ----------------------------------------------------------------------------
  ctx.setReadout(
    'grid',
    `${COLUMNS} x ${ROWS} cells of ${(cellWidth / 1000).toFixed(0)} km; ${formatCount(seedTotal)} coast seeds`
  );
  writeCoastSettings();
  replaceEvents(Number(ctx.options.eventsPerStorm));
  describeSelectedState();

  // ---- Instance -------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      densityCompiled,
      coastGraphs[ctx.options.distanceMode],
      ...(eventsVariant ? [eventsVariant.compiled] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'intensity':
          updateCoastChart();
          updateLegendExtents();
          break;
        case 'coastSearchKm':
          writeCoastSettings();
          ctx.requestLayers();
          break;
        case 'distanceMode':
          coastDirty = true;
          break;
        case 'eventsPerStorm':
          replaceEvents(Number(state.eventsPerStorm));
          break;
        case 'stateMetric':
        case 'landfallCounting':
        case 'minimumLandfallWind':
          if (id === 'landfallCounting') processEvents();
          else updateLandfallViews();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (!densityEncoded) {
        densityCompiled.encode(commandEncoder, {parameters: undefined});
        densityEncoded = true;
        densityReader.request(commandEncoder);
      }
      densityReader.flush(commandEncoder);

      if (coastDirty) {
        coastGraphs[ctx.options.distanceMode].encode(commandEncoder, {parameters: undefined});
        coastDirty = false;
        coastReader.markStale();
      }
      coastReader.flush(commandEncoder);

      if (eventsVariant && eventsDirty) {
        eventsVariant.compiled.encode(commandEncoder, {parameters: undefined});
        eventsVariant.encoded = true;
        eventsDirty = false;
        eventReader.request(commandEncoder);
      }
      eventReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const layerIndex = INTENSITY_INDEX[options.intensity];
      if (options.view === 'density') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `landfall-density-${layerIndex}`,
            coordinateOrigin,
            gridSize: [COLUMNS, ROWS],
            bounds,
            rowOrigin: 'south',
            tessellation: 48,
            values: displayBuffers[layerIndex],
            valueFormat: 'float32',
            valueRange: [0, Math.max(1e-6, displayMaxima[layerIndex])],
            colormap: options.ramp,
            sqrtScale: options.sqrtScale,
            discardAtOrBelow: 0,
            noDataColor: [0, 0, 0, 0],
            opacity: options.rasterOpacity
          })
        );
      } else if (options.view === 'coast') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `landfall-coast-${options.coastSearchKm}`,
            coordinateOrigin,
            gridSize: [COLUMNS, ROWS],
            bounds,
            rowOrigin: 'south',
            tessellation: 48,
            values: coastKmBuffer,
            valueFormat: 'float32',
            valueRange: [0, options.coastSearchKm],
            colormap: options.ramp,
            noDataColor: [0, 0, 0, 0],
            opacity: options.rasterOpacity
          })
        );
      } else if (options.view === 'landfalls') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `landfall-states-${options.stateMetric}`,
            coordinateOrigin,
            gridSize: [COLUMNS * CHOROPLETH_SCALE, ROWS * CHOROPLETH_SCALE],
            bounds,
            rowOrigin: 'south',
            tessellation: 48,
            values: stateDisplayBuffer,
            valueFormat: 'float32',
            valueIndices: zoneRasterBuffer,
            valueRange: [options.stateMetric === 'count' ? 0 : 34, landfallMaximum],
            colormap: options.ramp,
            noDataColor: [0, 0, 0, 0],
            opacity: options.rasterOpacity
          })
        );
      }
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'landfall-tracks',
            ...lngLatProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentEndsBuffer,
            colormap: 'category',
            palette: HURRICANE_CLASS[ctx.ground()],
            widthPixels: 1,
            opacity: options.trackOpacity
          })
        );
      }
      if (options.showCoastline) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'landfall-coastline',
            ...lngLatProps,
            segments: coastSegmentBuffer,
            instanceCount: coastSegmentCount,
            widthPixels: 1,
            color: dark ? [200, 215, 235, 150] : [40, 55, 80, 170]
          })
        );
      }
      if (options.showStateOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'landfall-state-outlines',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: edgeCount,
            widthPixels: 1,
            color: dark ? [225, 230, 245, 110] : [30, 40, 60, 120]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'landfall-selected-state',
          coordinateOrigin,
          segments: selectedOutline,
          instanceCount: maximumZoneEdges,
          widthPixels: 3,
          color: dark ? [255, 255, 255, 245] : [10, 15, 25, 245]
        })
      );
      if (options.showLandfalls && dotCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'landfall-dots',
            coordinateOrigin,
            positions: dotPositions,
            instanceCount: dotCount,
            values: dotCategories,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: HURRICANE_CLASS[ctx.ground()],
            radiusPixels: 4.5,
            opacity: 0.95
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = storms.project(event.coordinate[0], event.coordinate[1]);
      const zone = findStateAt(x, y);
      return zone < 0 ? null : describeState(zone);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = storms.project(event.coordinate[0], event.coordinate[1]);
      const zone = findStateAt(x, y);
      selectedState = zone === selectedState ? -1 : zone;
      writeSelectedOutline();
      describeSelectedState();
      updateLandfallViews();
      return zone >= 0;
    },

    destroy() {
      destroyed = true;
      densityReader.stop();
      coastReader.stop();
      eventReader.stop();
      resources.destroy();
    }
  };
}
