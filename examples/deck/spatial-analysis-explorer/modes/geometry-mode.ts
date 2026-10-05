// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Geometry operations (turf and PostGIS parity) on the GPU, as eight views of one mode:
 *
 * - Great-circle arcs (North Atlantic camera, worldwide destinations): `GPUGeodesicPairs` (WGS84 distances) and `GPUGreatCircleArcs`
 *   from a hub to 20,000 destinations, plus a `GPUGeodesicDestination` range ring. Segment count,
 *   maximum segment length, hub and ring radius are buffer writes.
 * - Densify, Chaikin smooth, chunk, substring: `GPULineSegmentize`, `GPULineSmooth`,
 *   `GPULineChunk` on the New York road paths (or trips). Resolution, ratio, chunk length and
 *   measure range are per-frame parameters.
 * - Measures (San Francisco ZIP codes): `GPUGeometryMeasures` with planar, spherical and WGS84
 *   nodes side by side, and group totals.
 * - Snap and Locate: `GPULinearReferencing` snaps POIs to roads (radius per frame) and
 *   `GPULineLocate` slides events along roads by a measure parameter.
 *
 * Every view compiles its graphs once in `create`; each slider is a buffer write. The camera is
 * outside a mode's control, so the view select re-enters the mode with that view's camera (Atlantic,
 * city or bay), which is navigation, not a parameter change: the footer's rebuild counter restarts
 * at 0 for the new view. The outputs are drawn directly from the contributors' storage buffers by the
 * layers in `geometry-layers.ts` (path outputs through an indirect record whose instance count is
 * the GPU-written vertex count).
 */

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineChunkParameterValues,
  getGPULineLocateParameterValues,
  getGPULineSegmentizeParameterValues,
  getGPULineSmoothParameterValues,
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGeometryMeasures,
  GPUGreatCircleArcs,
  GPULineChunk,
  GPULineLocate,
  GPULineSegmentize,
  GPULineSmooth,
  GPULinearReferencing,
  GPU_LINE_CHUNK_PARAMETER_LENGTH,
  GPU_LINE_LOCATE_PARAMETER_LENGTH,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH,
  GPU_LINE_SMOOTH_PARAMETER_LENGTH
} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '../graph-buffers';
import {
  createSeededRandom,
  LocalMetricProjection,
  type SpatialAnalysisPolygons,
  type SpatialAnalysisRoadNetwork
} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisFrame,
  SpatialAnalysisModeContext,
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisViewState
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {PairSegmentLayer, PathOutputLayer, PolygonFanLayer} from './geometry-layers';
import {SummaryReader} from './summary-reader';

type GeometryView =
  | 'arcs'
  | 'densify'
  | 'smooth'
  | 'chunk'
  | 'substring'
  | 'snap'
  | 'locate'
  | 'measures';
type GeometryScene = 'atlantic' | 'city' | 'midtown' | 'block' | 'bay';

const SCENE_VIEW_STATES: Record<GeometryScene, SpatialAnalysisViewState> = {
  // The shell clamps the zoom to 3, so the widest camera spans about 80 degrees of longitude.
  atlantic: {longitude: -55, latitude: 46, zoom: 3},
  city: {longitude: -73.985, latitude: 40.745, zoom: 12.4},
  midtown: {longitude: -73.985, latitude: 40.755, zoom: 14.3},
  block: {longitude: -73.985, latitude: 40.755, zoom: 15.4},
  bay: {longitude: -122.44, latitude: 37.76, zoom: 11.4}
};

const VIEWS: readonly {id: GeometryView; label: string; scene: GeometryScene}[] = [
  {id: 'arcs', label: 'Great-circle arcs (North Atlantic camera)', scene: 'atlantic'},
  {id: 'densify', label: 'Densify roads (GPULineSegmentize)', scene: 'city'},
  {id: 'smooth', label: 'Chaikin-smooth trips (GPULineSmooth)', scene: 'midtown'},
  {id: 'chunk', label: 'Chunk roads (GPULineChunk)', scene: 'city'},
  {id: 'substring', label: 'Road substrings (GPULineChunk)', scene: 'city'},
  {id: 'snap', label: 'Snap POIs to roads (GPULinearReferencing)', scene: 'block'},
  {id: 'locate', label: 'Events along roads (GPULineLocate)', scene: 'midtown'},
  {id: 'measures', label: 'Zone measures (GPUGeometryMeasures, SF)', scene: 'bay'}
];

/** Reads the `view` URL parameter once at load so the first camera matches the view. */
function readViewFromUrl(): GeometryView {
  const parameter =
    typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get('view');
  return VIEWS.find(entry => entry.id === parameter)?.id ?? 'arcs';
}

/** View chosen by the last navigation, or by `?view=` at load. */
let requestedView: GeometryView = readViewFromUrl();

function getSceneOf(view: GeometryView): GeometryScene {
  return VIEWS.find(entry => entry.id === view)!.scene;
}

/** The view for a create call. */
function resolveView(): GeometryView {
  return requestedView;
}

/** Switches view by re-entering the mode with that view's camera. */
function navigate(view: GeometryView): void {
  requestedView = view;
  geometryMode.initialViewState = SCENE_VIEW_STATES[getSceneOf(view)];
  globalThis.spatialAnalysisExplorer?.selectMode('geometry');
}

/** What a view returns: the mode instance minus `destroy`, which the mode owns. */
type ViewInstance = Omit<SpatialAnalysisModeInstance, 'destroy'>;

/** Frames between summary readbacks when nothing changed. */
const READBACK_INTERVAL_FRAMES = 45;

/**
 * Geometry operations: great-circle arcs, densify, smooth, chunk, substring, zone measures,
 * linear referencing and event placement. See the file comment.
 */
export const geometryMode: SpatialAnalysisModeDefinition = {
  id: 'geometry',
  title: 'Geometry',
  contributors: [
    'GPULineSegmentize',
    'GPUGreatCircleArcs',
    'GPULineSmooth',
    'GPULineChunk',
    'GPUGeometryMeasures',
    'GPUGeodesicPairs',
    'GPUGeodesicDestination',
    'GPULinearReferencing',
    'GPULineLocate'
  ],
  description:
    'Turf and PostGIS style geometry on the GPU. Pick a view: great-circle arcs from a hub, ' +
    'densified, smoothed or chunked roads, zone area and perimeter, POIs snapped to roads, or ' +
    'events sliding along roads. Every slider rewrites a parameter buffer; nothing recompiles.',
  initialViewState: SCENE_VIEW_STATES[getSceneOf(requestedView)],

  async create(context) {
    const view = resolveView();
    const resources = new SpatialAnalysisResources(context.device, `geometry-${view}`);
    context.controls.addSelect<GeometryView>({
      label: 'View (re-enters the mode with this view’s camera)',
      options: VIEWS.map(entry => ({value: entry.id, label: entry.label})),
      value: view,
      onChange: navigate
    });
    let instance: ViewInstance;
    switch (view) {
      case 'arcs':
        instance = createArcsView(context, resources);
        break;
      case 'densify':
        instance = await createDensifyView(context, resources);
        break;
      case 'smooth':
        instance = await createSmoothView(context, resources);
        break;
      case 'chunk':
        instance = await createChunkView(context, resources, 'chunk');
        break;
      case 'substring':
        instance = await createChunkView(context, resources, 'substring');
        break;
      case 'snap':
        instance = await createSnapView(context, resources);
        break;
      case 'locate':
        instance = await createLocateView(context, resources);
        break;
      case 'measures':
        instance = await createMeasuresView(context, resources);
        break;
    }
    return {...instance, destroy: () => resources.destroy()};
  }
};

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/** Road polylines rebuilt from the road network's consecutive segments. */
type RoadPaths = {
  /** `x, y` meters per vertex. */
  positions: Float32Array;
  /** `pathCount + 1` vertex offsets. */
  offsets: Uint32Array;
  /** Planar length per path in meters. */
  lengths: Float32Array;
  /** Sum of every path length. */
  totalLength: number;
};

/** The road network stores each OSM polyline as consecutive segments; stitch them back. */
function buildRoadPaths(roads: SpatialAnalysisRoadNetwork): RoadPaths {
  const segments = roads.segments;
  const segmentCount = segments.length / 4;
  const positions: number[] = [];
  const offsets: number[] = [0];
  const lengths: number[] = [];
  let pathLength = 0;
  let previousEndX = Number.NaN;
  let previousEndY = Number.NaN;
  for (let segment = 0; segment < segmentCount; segment++) {
    const [x0, y0, x1, y1] = segments.subarray(segment * 4, segment * 4 + 4);
    if (x0 !== previousEndX || y0 !== previousEndY) {
      if (positions.length > 0) {
        offsets.push(positions.length / 2);
        lengths.push(pathLength);
      }
      pathLength = 0;
      positions.push(x0, y0);
    }
    positions.push(x1, y1);
    pathLength += Math.hypot(x1 - x0, y1 - y0);
    previousEndX = x1;
    previousEndY = y1;
  }
  offsets.push(positions.length / 2);
  lengths.push(pathLength);
  return {
    positions: Float32Array.from(positions),
    offsets: Uint32Array.from(offsets),
    lengths: Float32Array.from(lengths),
    totalLength: lengths.reduce((total, length) => total + length, 0)
  };
}

/** A path layout drawn without a contributor: static buffers plus an indirect record. */
type StaticPaths = {
  positions: Buffer;
  offsets: Buffer;
  vertexCount: Buffer;
  offsetCount: number;
  drawCommands: DrawCommandBuffer;
};

function createStaticPaths(
  resources: SpatialAnalysisResources,
  name: string,
  positions: Float32Array,
  offsets: Uint32Array
): StaticPaths {
  const vertexCount = positions.length / 2;
  return {
    positions: resources.createBuffer(`${name}-positions`, positions),
    offsets: resources.createBuffer(`${name}-offsets`, offsets),
    vertexCount: resources.createBuffer(`${name}-count`, Uint32Array.of(vertexCount)),
    offsetCount: offsets.length,
    drawCommands: resources.track(
      new DrawCommandBuffer(resources.device, {
        id: `geometry-${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: vertexCount}]
      })
    )
  };
}

/** A contributor path output: caller-owned buffers plus the indirect record fed from `count`. */
type PathOutputBuffers = {
  positions: Buffer;
  offsets: Buffer;
  count: Buffer;
  overflow: Buffer;
  totalCount: Buffer;
  pathCount: Buffer;
  measures: Buffer;
  sourcePaths: Buffer;
  offsetCount: number;
  drawCommands: DrawCommandBuffer;
};

function createPathOutputBuffers(
  resources: SpatialAnalysisResources,
  name: string,
  vertexCapacity: number,
  pathCapacity: number
): PathOutputBuffers {
  return {
    positions: resources.createBuffer(`${name}-positions`, vertexCapacity * 8),
    offsets: resources.createBuffer(`${name}-offsets`, (pathCapacity + 1) * 4),
    count: resources.createBuffer(`${name}-count`, 4),
    overflow: resources.createBuffer(`${name}-overflow`, 4),
    totalCount: resources.createBuffer(`${name}-total`, 4),
    pathCount: resources.createBuffer(`${name}-path-count`, 4),
    measures: resources.createBuffer(`${name}-measures`, vertexCapacity * 4),
    sourcePaths: resources.createBuffer(`${name}-source-paths`, pathCapacity * 4),
    offsetCount: pathCapacity + 1,
    drawCommands: resources.track(
      new DrawCommandBuffer(resources.device, {
        id: `geometry-${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    )
  };
}

/** Copies the GPU-written vertex count into word 1 (instance count) of an indirect record. */
function copyCountToDrawRecord(
  commandEncoder: CommandEncoder,
  count: Buffer,
  drawCommands: DrawCommandBuffer
): void {
  commandEncoder.copyBufferToBuffer({
    sourceBuffer: count,
    sourceOffset: 0,
    destinationBuffer: drawCommands.buffer,
    destinationOffset: 4,
    size: 4
  });
}

/** Binds a path output's views into a graph and returns the contributor `output` object. */
function importPathOutput(
  graph: GPUCommandGraph<void>,
  output: PathOutputBuffers,
  vertexCapacity: number,
  pathCapacity: number,
  options: {measures?: boolean; pathCount?: boolean; sourcePaths?: boolean} = {}
) {
  return {
    positions: importGraphBuffer(
      graph,
      'out-positions',
      output.positions,
      'float32x2',
      vertexCapacity
    ),
    pathOffsets: importGraphBuffer(
      graph,
      'out-offsets',
      output.offsets,
      'uint32',
      pathCapacity + 1
    ),
    count: importGraphBuffer(graph, 'out-count', output.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'out-overflow', output.overflow, 'uint32', 1),
    totalCount: importGraphBuffer(graph, 'out-total', output.totalCount, 'uint32', 1),
    ...(options.pathCount
      ? {pathCount: importGraphBuffer(graph, 'out-path-count', output.pathCount, 'uint32', 1)}
      : {}),
    ...(options.sourcePaths
      ? {
          sourcePaths: importGraphBuffer(
            graph,
            'out-source-paths',
            output.sourcePaths,
            'uint32',
            pathCapacity
          )
        }
      : {}),
    ...(options.measures
      ? {
          measures: importGraphBuffer(
            graph,
            'out-measures',
            output.measures,
            'float32',
            vertexCapacity
          )
        }
      : {})
  };
}

function formatKilometers(meters: number): string {
  return `${(meters / 1000).toLocaleString('en-US', {maximumFractionDigits: 0})} km`;
}

function formatMeters(meters: number): string {
  return meters < 10 ? `${meters.toFixed(1)} m` : `${Math.round(meters)} m`;
}

// ---------------------------------------------------------------------------------------------
// View: great-circle arcs
// ---------------------------------------------------------------------------------------------

const HUBS = [
  {id: 'new-york', label: 'New York', position: [-74.0, 40.7]},
  {id: 'london', label: 'London', position: [-0.13, 51.5]},
  {id: 'reykjavik', label: 'Reykjavik', position: [-21.9, 64.1]},
  {id: 'lisbon', label: 'Lisbon', position: [-9.14, 38.7]}
] as const;

/** `[longitude, latitude, weight]` of the synthetic destination clusters. */
const CITY_ANCHORS: readonly (readonly [number, number, number])[] = [
  [-74, 40.7, 3],
  [-118.2, 34, 2],
  [-87.6, 41.9, 1.5],
  [-99.1, 19.4, 2],
  [-46.6, -23.5, 3],
  [-58.4, -34.6, 1.5],
  [-77, -12, 1],
  [-0.1, 51.5, 3],
  [2.3, 48.9, 2],
  [13.4, 52.5, 1.5],
  [37.6, 55.7, 2],
  [28.9, 41, 1.5],
  [31.2, 30, 1.5],
  [3.4, 6.5, 2],
  [18.4, -33.9, 1],
  [36.8, -1.3, 1],
  [77.2, 28.6, 3],
  [72.8, 19, 2],
  [90.4, 23.8, 2],
  [100.5, 13.7, 1.5],
  [106.8, -6.2, 2],
  [121.5, 31.2, 3],
  [116.4, 39.9, 3],
  [126.9, 37.6, 1.5],
  [139.7, 35.7, 3],
  [103.8, 1.35, 1.5],
  [151.2, -33.9, 1.5],
  [174.8, -36.8, 0.7],
  [55.3, 25.2, 1],
  [51.4, 35.7, 1.5],
  [-123.1, 49.3, 1],
  [-79.4, 43.7, 1.2]
];

const ARC_PAIR_COUNT = 20000;
const ARC_MAXIMUM_SEGMENTS = 64;
const RING_VERTEX_COUNT = 361;
const WORLD_COPIES = [-360, 0, 360] as const;

/** Synthetic world destinations: Gaussian scatter around weighted city anchors. */
function makeWorldDestinations(count: number): Float32Array {
  const random = createSeededRandom(2026);
  const totalWeight = CITY_ANCHORS.reduce((total, anchor) => total + anchor[2], 0);
  const destinations = new Float32Array(count * 2);
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(random(), 1e-9))) * Math.cos(2 * Math.PI * random());
  for (let index = 0; index < count; index++) {
    let pick = random() * totalWeight;
    let anchor = CITY_ANCHORS[0];
    for (const candidate of CITY_ANCHORS) {
      pick -= candidate[2];
      if (pick <= 0) {
        anchor = candidate;
        break;
      }
    }
    const latitude = Math.max(-70, Math.min(75, anchor[1] + gaussian() * 3));
    const longitude =
      anchor[0] + (gaussian() * 3) / Math.max(0.3, Math.cos((latitude * Math.PI) / 180));
    destinations[index * 2] = ((((longitude + 180) % 360) + 360) % 360) - 180;
    destinations[index * 2 + 1] = latitude;
  }
  return destinations;
}

function createArcsView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): ViewInstance {
  const {device} = context;
  const destinations = makeWorldDestinations(ARC_PAIR_COUNT);
  const vertexCapacity = ARC_PAIR_COUNT * (ARC_MAXIMUM_SEGMENTS + 1);
  let hubIndex = 0;
  let minimumSegments = 24;
  let maximumLengthKilometers = 0;
  let ringKilometers = 3000;
  let showRing = true;
  let dirty = true;

  const sources = resources.createBuffer('sources', new Float32Array(ARC_PAIR_COUNT * 2));
  const targets = resources.createBuffer('targets', destinations);
  const distances = resources.createBuffer('distances', ARC_PAIR_COUNT * 4);
  const converged = resources.createBuffer('converged', ARC_PAIR_COUNT * 4);
  const arcs = createPathOutputBuffers(resources, 'arcs', vertexCapacity, ARC_PAIR_COUNT);
  const arcParameters = resources.createParameterBuffer(
    'arc-parameters',
    'float32',
    GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
  );
  const hubPosition = resources.createBuffer('hub-position', 8);
  const ringOrigins = resources.createBuffer('ring-origins', RING_VERTEX_COUNT * 8);
  const ringBearings = resources.createBuffer(
    'ring-bearings',
    Float32Array.from({length: RING_VERTEX_COUNT}, (_, index) => index)
  );
  const ringDistances = resources.createBuffer('ring-distances', RING_VERTEX_COUNT * 4);
  const ringDestinations = resources.createBuffer('ring-destinations', RING_VERTEX_COUNT * 8);
  const ringPaths = createStaticPaths(
    resources,
    'ring-paths',
    new Float32Array(RING_VERTEX_COUNT * 2),
    Uint32Array.of(0, RING_VERTEX_COUNT)
  );

  const arcGraph = new GPUCommandGraph<void>(device, {id: 'geometry-arcs'});
  const arcSources = importGraphBuffer(arcGraph, 'sources', sources, 'float32x2', ARC_PAIR_COUNT);
  const arcTargets = importGraphBuffer(arcGraph, 'targets', targets, 'float32x2', ARC_PAIR_COUNT);
  arcGraph.add(
    new GPUGeodesicPairs({
      id: 'arc-distances',
      origins: arcSources,
      targets: arcTargets,
      model: 'wgs84',
      output: {
        distances: importGraphBuffer(arcGraph, 'distances', distances, 'float32', ARC_PAIR_COUNT),
        converged: importGraphBuffer(arcGraph, 'converged', converged, 'uint32', ARC_PAIR_COUNT)
      }
    })
  );
  arcGraph.add(
    new GPUGreatCircleArcs({
      id: 'arcs',
      sources: arcSources,
      targets: arcTargets,
      maximumSegments: ARC_MAXIMUM_SEGMENTS,
      parameters: arcParameters.importToGraph(arcGraph),
      output: importPathOutput(arcGraph, arcs, vertexCapacity, ARC_PAIR_COUNT)
    })
  );
  const compiledArcs = resources.track(arcGraph.compile());

  const ringGraph = new GPUCommandGraph<void>(device, {id: 'geometry-ring'});
  ringGraph.add(
    new GPUGeodesicDestination({
      id: 'range-ring',
      origins: importGraphBuffer(ringGraph, 'origins', ringOrigins, 'float32x2', RING_VERTEX_COUNT),
      bearings: importGraphBuffer(
        ringGraph,
        'bearings',
        ringBearings,
        'float32',
        RING_VERTEX_COUNT
      ),
      distances: importGraphBuffer(
        ringGraph,
        'distances',
        ringDistances,
        'float32',
        RING_VERTEX_COUNT
      ),
      model: 'wgs84',
      output: {
        destinations: importGraphBuffer(
          ringGraph,
          'destinations',
          ringDestinations,
          'float32x2',
          RING_VERTEX_COUNT
        )
      }
    })
  );
  const compiledRing = resources.track(ringGraph.compile());

  const writeHub = () => {
    const [longitude, latitude] = HUBS[hubIndex].position;
    sources.write(
      Float32Array.from({length: ARC_PAIR_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? longitude : latitude
      )
    );
    ringOrigins.write(
      Float32Array.from({length: RING_VERTEX_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? longitude : latitude
      )
    );
    hubPosition.write(Float32Array.of(longitude, latitude));
    dirty = true;
  };
  const writeArcParameters = () => {
    arcParameters.write(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength: maximumLengthKilometers * 1000,
        minimumSegments
      })
    );
    dirty = true;
  };
  const writeRing = () => {
    ringDistances.write(new Float32Array(RING_VERTEX_COUNT).fill(ringKilometers * 1000));
    dirty = true;
  };

  context.controls.addSelect<string>({
    label: 'Hub (rewrites the origin column)',
    options: HUBS.map(hub => ({value: hub.id, label: hub.label})),
    value: HUBS[hubIndex].id,
    onChange: value => {
      hubIndex = HUBS.findIndex(hub => hub.id === value);
      writeHub();
    }
  });
  context.controls.addSlider({
    label: 'Minimum segments per arc (parameter)',
    min: 1,
    max: ARC_MAXIMUM_SEGMENTS,
    step: 1,
    value: minimumSegments,
    format: value => (value === 1 ? '1 (a chord in lon/lat)' : String(value)),
    onChange: value => {
      minimumSegments = value;
      writeArcParameters();
    }
  });
  context.controls.addSlider({
    label: 'Maximum segment length (parameter, 0 = off)',
    min: 0,
    max: 3000,
    step: 100,
    value: maximumLengthKilometers,
    format: value => (value === 0 ? 'off' : `${value} km`),
    onChange: value => {
      maximumLengthKilometers = value;
      writeArcParameters();
    }
  });
  context.controls.addSlider({
    label: 'Range ring radius (GPUGeodesicDestination)',
    min: 500,
    max: 10000,
    step: 500,
    value: ringKilometers,
    format: value => `${value} km`,
    onChange: value => {
      ringKilometers = value;
      writeRing();
    }
  });
  context.controls.addToggle({
    label: 'Show range ring',
    value: showRing,
    onChange: value => {
      showRing = value;
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Arc color: WGS84 geodesic distance (GPUGeodesicPairs)',
    gradient: {
      colors: [
        [0, 0, 4],
        [120, 28, 109],
        [237, 105, 37],
        [252, 255, 164]
      ],
      minimumLabel: '0 km',
      maximumLabel: '20,000 km'
    }
  });
  context.controls.addNote(
    'Arcs keep unwrapped longitudes (continuous across the antimeridian), so each layer is drawn ' +
      'three times at 0 and ±360 degrees. The shell limits the zoom to 3, so most of the ' +
      'world is off screen; at 1 segment every arc is a straight chord in lon/lat, and at 24 ' +
      'the great-circle bulge toward the pole appears.'
  );
  context.controls.addReadout('Destinations', formatCount(ARC_PAIR_COUNT));
  const vertexReadout = context.controls.addReadout('Arc vertices');
  const overflowReadout = context.controls.addReadout('Overflow');
  context.controls.addReadout('Data', 'Synthetic world cities (seeded)');

  writeHub();
  writeArcParameters();
  writeRing();

  const reader = new SummaryReader(
    resources,
    'arcs',
    [
      {buffer: arcs.count, size: 4},
      {buffer: arcs.overflow, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      vertexReadout.setValue(
        `${formatCount(words[0])} of ${formatCount(vertexCapacity)} (${(words[0] / ARC_PAIR_COUNT).toFixed(1)} per arc)`
      );
      overflowReadout.setValue(words[1] ? 'yes' : 'no');
    }
  );

  return {
    getCompiledGraphs: () => [compiledArcs, compiledRing],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiledArcs.encode(commandEncoder, {parameters: undefined});
        compiledRing.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, arcs.count, arcs.drawCommands);
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [];
      for (const offset of WORLD_COPIES) {
        layers.push(
          new PathOutputLayer({
            id: `arcs-${offset}`,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: arcs.positions,
            pathOffsets: arcs.offsets,
            pathOffsetCount: arcs.offsetCount,
            vertexCount: arcs.count,
            drawCommands: arcs.drawCommands,
            colorSource: 'path-value',
            values: distances,
            colormap: 'inferno',
            valueRange: [0, 2.0e7],
            color: [255, 255, 255, 70],
            widthPixels: 1,
            positionOffset: [offset, 0]
          })
        );
        if (showRing) {
          layers.push(
            new PathOutputLayer({
              id: `ring-${offset}`,
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              positions: ringDestinations,
              pathOffsets: ringPaths.offsets,
              pathOffsetCount: ringPaths.offsetCount,
              vertexCount: ringPaths.vertexCount,
              drawCommands: ringPaths.drawCommands,
              color: [90, 235, 255, 255],
              widthPixels: 2.5,
              positionOffset: [offset, 0]
            })
          );
        }
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'arcs-hub',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: hubPosition,
          instanceCount: 1,
          radiusPixels: 7,
          color: [255, 255, 255, 255]
        })
      );
      return layers;
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Views: densify, smooth, chunk, substring (roads and trips)
// ---------------------------------------------------------------------------------------------

const ROAD_COLOR = [150, 170, 215, 110] as const;

/** Faint static road paths drawn under the contributor output. */
function getRoadLayer(id: string, roadPaths: StaticPaths, origin: readonly [number, number]) {
  return new PathOutputLayer({
    id,
    coordinateOrigin: [origin[0], origin[1], 0],
    positions: roadPaths.positions,
    pathOffsets: roadPaths.offsets,
    pathOffsetCount: roadPaths.offsetCount,
    vertexCount: roadPaths.vertexCount,
    drawCommands: roadPaths.drawCommands,
    color: [...ROAD_COLOR],
    widthPixels: 1.5
  });
}

const MINIMUM_DENSIFY_LENGTH = 6;
const DENSIFY_MAXIMUM_PIECES = 64;

async function createDensifyView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): Promise<ViewInstance> {
  const roads = await context.data.getNewYorkRoads();
  context.signal.throwIfAborted();
  const {device} = context;
  const paths = buildRoadPaths(roads);
  const inputVertexCount = paths.positions.length / 2;
  const pathCount = paths.offsets.length - 1;
  // Capacity at the finest slider setting.
  let capacity = inputVertexCount;
  for (let path = 0; path < pathCount; path++) {
    for (let row = paths.offsets[path]; row + 1 < paths.offsets[path + 1]; row++) {
      const length = Math.hypot(
        paths.positions[row * 2 + 2] - paths.positions[row * 2],
        paths.positions[row * 2 + 3] - paths.positions[row * 2 + 1]
      );
      capacity +=
        Math.min(DENSIFY_MAXIMUM_PIECES, Math.max(1, Math.ceil(length / MINIMUM_DENSIFY_LENGTH))) -
        1;
    }
  }
  capacity += 16;
  let maximumLength = 40;
  let dirty = true;

  const roadPaths = createStaticPaths(resources, 'roads', paths.positions, paths.offsets);
  const output = createPathOutputBuffers(resources, 'densify', capacity, pathCount);
  const parameters = resources.createParameterBuffer(
    'densify-parameters',
    'float32',
    GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
  );
  const graph = new GPUCommandGraph<void>(device, {id: 'geometry-densify'});
  graph.add(
    new GPULineSegmentize({
      id: 'densify',
      positions: importGraphBuffer(
        graph,
        'positions',
        roadPaths.positions,
        'float32x2',
        inputVertexCount
      ),
      pathOffsets: importGraphBuffer(graph, 'offsets', roadPaths.offsets, 'uint32', pathCount + 1),
      coordinateSystem: 'planar',
      maximumPiecesPerSegment: DENSIFY_MAXIMUM_PIECES,
      parameters: parameters.importToGraph(graph),
      output: importPathOutput(graph, output, capacity, pathCount, {measures: true})
    })
  );
  const compiled = resources.track(graph.compile());

  const writeParameters = () => {
    parameters.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: maximumLength}));
    dirty = true;
  };
  context.controls.addSlider({
    label: 'Maximum segment length (parameter, log scale)',
    min: Math.log10(MINIMUM_DENSIFY_LENGTH),
    max: 3,
    step: 0.02,
    value: Math.log10(maximumLength),
    format: value => formatMeters(10 ** value),
    onChange: value => {
      maximumLength = 10 ** value;
      writeParameters();
    }
  });
  context.controls.addLegend({
    title: 'Vertices: cumulative distance along the road (output measures)',
    gradient: {
      colors: [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
      ],
      minimumLabel: '0 m',
      maximumLabel: '1 km+'
    }
  });
  context.controls.addNote(
    'Planar densify: every input vertex is kept and long segments gain equal pieces. The cap ' +
      `of ${DENSIFY_MAXIMUM_PIECES} pieces per segment is compile-time.`
  );
  context.controls.addReadout('Road paths', formatCount(pathCount));
  context.controls.addReadout('Input vertices', formatCount(inputVertexCount));
  const outputReadout = context.controls.addReadout('Output vertices');
  const spacingReadout = context.controls.addReadout('Mean spacing');
  const overflowReadout = context.controls.addReadout('Overflow');
  context.controls.addReadout('Total road length', formatKilometers(paths.totalLength));
  context.controls.addReadout('Data', roads.attribution);
  writeParameters();

  const reader = new SummaryReader(
    resources,
    'densify',
    [
      {buffer: output.count, size: 4},
      {buffer: output.overflow, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      outputReadout.setValue(
        `${formatCount(words[0])} (${(words[0] / inputVertexCount).toFixed(2)}x)`
      );
      spacingReadout.setValue(formatMeters(paths.totalLength / Math.max(1, words[0] - pathCount)));
      overflowReadout.setValue(words[1] ? 'yes' : 'no');
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, output.count, output.drawCommands);
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const origin = roads.origin;
      return [
        getRoadLayer('densify-roads', roadPaths, origin),
        new SpatialAnalysisPointLayer({
          id: 'densify-vertices',
          coordinateOrigin: [origin[0], origin[1], 0],
          positions: output.positions,
          drawCommands: output.drawCommands,
          values: output.measures,
          valueFormat: 'float32',
          colormap: 'viridis',
          valueRange: [0, 1000],
          radiusPixels: 2.2,
          color: [255, 255, 255, 235]
        })
      ];
    }
  };
}

const SMOOTH_ITERATION_OPTIONS = [1, 2, 3, 4] as const;

async function createSmoothView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): Promise<ViewInstance> {
  const trips = await context.data.getNewYorkTrips();
  context.signal.throwIfAborted();
  const {device} = context;
  const vertexCount = trips.vertexPositions.length / 2;
  const tripCount = trips.vendors.length;
  const tripPaths = createStaticPaths(resources, 'trips', trips.vertexPositions, trips.tripOffsets);
  const parameters = resources.createParameterBuffer(
    'smooth-parameters',
    'float32',
    GPU_LINE_SMOOTH_PARAMETER_LENGTH
  );
  // One prebuilt graph per iteration count (the count is compile-time).
  const levels = SMOOTH_ITERATION_OPTIONS.map(iterations => {
    const capacity = vertexCount * 2 ** iterations + 16;
    const output = createPathOutputBuffers(resources, `smooth-${iterations}`, capacity, tripCount);
    const graph = new GPUCommandGraph<void>(device, {id: `geometry-smooth-${iterations}`});
    graph.add(
      new GPULineSmooth({
        id: `smooth-${iterations}`,
        positions: importGraphBuffer(
          graph,
          'positions',
          tripPaths.positions,
          'float32x2',
          vertexCount
        ),
        pathOffsets: importGraphBuffer(
          graph,
          'offsets',
          tripPaths.offsets,
          'uint32',
          tripCount + 1
        ),
        iterations,
        closed: false,
        parameters: parameters.importToGraph(graph),
        output: importPathOutput(graph, output, capacity, tripCount)
      })
    );
    return {iterations, capacity, output, compiled: resources.track(graph.compile())};
  });

  let levelIndex = 2;
  let ratio = 0.25;
  let showOriginal = true;
  let dirty = true;

  // Original trip segments for the faint underlay.
  let segmentCount = 0;
  for (let trip = 0; trip < tripCount; trip++) {
    segmentCount += Math.max(0, trips.tripOffsets[trip + 1] - trips.tripOffsets[trip] - 1);
  }
  const segments = new Float32Array(segmentCount * 4);
  let row = 0;
  for (let trip = 0; trip < tripCount; trip++) {
    const first = trips.tripOffsets[trip];
    const last = trips.tripOffsets[trip + 1] - 1;
    for (let vertex = first; vertex < last; vertex++, row++) {
      segments.set(trips.vertexPositions.subarray(vertex * 2, vertex * 2 + 4), row * 4);
    }
  }
  const segmentsBuffer = resources.createBuffer('trip-segments', segments);

  const writeParameters = () => {
    parameters.write(getGPULineSmoothParameterValues({ratio}));
    dirty = true;
  };
  context.controls.addSelect<string>({
    label: 'Chaikin iterations (compile-time: one prebuilt graph each)',
    options: SMOOTH_ITERATION_OPTIONS.map(iterations => ({
      value: String(iterations),
      label: `${iterations} iteration${iterations > 1 ? 's' : ''} (${2 ** iterations}x vertices)`
    })),
    value: String(SMOOTH_ITERATION_OPTIONS[levelIndex]),
    onChange: value => {
      levelIndex = SMOOTH_ITERATION_OPTIONS.indexOf(Number(value) as 1 | 2 | 3 | 4);
      dirty = true;
      context.updateLayers();
    }
  });
  context.controls.addSlider({
    label: 'Cut ratio (parameter; 0.5 = polyline of midpoints)',
    min: 0.05,
    max: 0.5,
    step: 0.01,
    value: ratio,
    format: value => value.toFixed(2),
    onChange: value => {
      ratio = value;
      writeParameters();
    }
  });
  context.controls.addToggle({
    label: 'Show original trips',
    value: showOriginal,
    onChange: value => {
      showOriginal = value;
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Lines',
    entries: [
      {color: [255, 150, 40, 255], label: 'Chaikin-smoothed'},
      {color: [120, 170, 255, 150], label: 'Original GPS trips'}
    ]
  });
  context.controls.addReadout(
    'Trips / vertices',
    `${formatCount(tripCount)} / ${formatCount(vertexCount)}`
  );
  const outputReadout = context.controls.addReadout('Smoothed vertices');
  const overflowReadout = context.controls.addReadout('Overflow');
  context.controls.addReadout('Data', trips.attribution);
  writeParameters();

  let readerLevel = -1;
  const readers = levels.map(
    (level, index) =>
      new SummaryReader(
        resources,
        `smooth-${level.iterations}`,
        [
          {buffer: level.output.count, size: 4},
          {buffer: level.output.overflow, size: 4}
        ],
        bytes => {
          if (index !== levelIndex) return;
          const words = new Uint32Array(bytes);
          outputReadout.setValue(
            `${formatCount(words[0])} (${(words[0] / vertexCount).toFixed(1)}x)`
          );
          overflowReadout.setValue(words[1] ? 'yes' : 'no');
        }
      )
  );

  return {
    getCompiledGraphs: () => levels.map(level => level.compiled),
    encode(commandEncoder, frame) {
      const level = levels[levelIndex];
      if (dirty || frame.frameIndex < 3 || readerLevel !== levelIndex) {
        level.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, level.output.count, level.output.drawCommands);
        dirty = false;
        readerLevel = levelIndex;
        readers[levelIndex].markStale();
      }
      readers[levelIndex].flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const origin = trips.origin;
      const level = levels[levelIndex];
      const layers: Layer[] = [];
      if (showOriginal) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'smooth-original',
            coordinateOrigin: [origin[0], origin[1], 0],
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.5,
            color: [120, 170, 255, 150]
          })
        );
      }
      layers.push(
        new PathOutputLayer({
          id: `smooth-output-${level.iterations}`,
          coordinateOrigin: [origin[0], origin[1], 0],
          positions: level.output.positions,
          pathOffsets: level.output.offsets,
          pathOffsetCount: level.output.offsetCount,
          vertexCount: level.output.count,
          drawCommands: level.output.drawCommands,
          color: [255, 150, 40, 255],
          widthPixels: 1.6
        })
      );
      return layers;
    }
  };
}

const MINIMUM_CHUNK_LENGTH = 25;

async function createChunkView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources,
  mode: 'chunk' | 'substring'
): Promise<ViewInstance> {
  const roads = await context.data.getNewYorkRoads();
  context.signal.throwIfAborted();
  const {device} = context;
  const paths = buildRoadPaths(roads);
  const inputVertexCount = paths.positions.length / 2;
  const pathCount = paths.offsets.length - 1;
  const chunkMode = mode === 'chunk';
  // Chunk capacity at the finest slider setting; substring emits one piece per road.
  const pieceCapacity = chunkMode
    ? Math.ceil(paths.totalLength / MINIMUM_CHUNK_LENGTH) + pathCount + 16
    : pathCount;
  const vertexCapacity = chunkMode
    ? 2 * pieceCapacity + inputVertexCount
    : inputVertexCount + 2 * pathCount;
  let chunkLength = 200;
  let startMeasure = 40;
  let endMeasure = 160;
  let dirty = true;

  const roadPaths = createStaticPaths(resources, 'roads', paths.positions, paths.offsets);
  const output = createPathOutputBuffers(resources, mode, vertexCapacity, pieceCapacity);
  const parameters = resources.createParameterBuffer(
    `${mode}-parameters`,
    'float32',
    GPU_LINE_CHUNK_PARAMETER_LENGTH
  );
  const graph = new GPUCommandGraph<void>(device, {id: `geometry-${mode}`});
  graph.add(
    new GPULineChunk({
      id: mode,
      positions: importGraphBuffer(
        graph,
        'positions',
        roadPaths.positions,
        'float32x2',
        inputVertexCount
      ),
      pathOffsets: importGraphBuffer(graph, 'offsets', roadPaths.offsets, 'uint32', pathCount + 1),
      mode,
      coordinateSystem: 'planar',
      parameters: parameters.importToGraph(graph),
      output: importPathOutput(graph, output, vertexCapacity, pieceCapacity, {
        pathCount: true,
        sourcePaths: chunkMode
      })
    })
  );
  const compiled = resources.track(graph.compile());

  const writeParameters = () => {
    parameters.write(
      getGPULineChunkParameterValues(
        chunkMode ? {chunkLength} : {startMeasure, endMeasure: Math.max(startMeasure, endMeasure)}
      )
    );
    dirty = true;
  };

  if (chunkMode) {
    context.controls.addSlider({
      label: 'Chunk length (parameter, log scale)',
      min: Math.log10(MINIMUM_CHUNK_LENGTH),
      max: 3,
      step: 0.02,
      value: Math.log10(chunkLength),
      format: value => formatMeters(10 ** value),
      onChange: value => {
        chunkLength = 10 ** value;
        writeParameters();
      }
    });
    context.controls.addLegend({
      title: 'Chunks (color cycles with the output piece index)',
      entries: [
        {color: [79, 201, 255, 255], label: 'piece 0, 6, 12...'},
        {color: [255, 148, 72, 255], label: '1, 7...'},
        {color: [189, 122, 255, 255], label: '2, 8...'},
        {color: [87, 235, 168, 255], label: '3, 9...'},
        {color: [255, 105, 168, 255], label: '4, 10...'},
        {color: [245, 220, 87, 255], label: '5, 11...'}
      ]
    });
    context.controls.addNote(
      'Consecutive chunks of one road share their boundary point. The piece count is data ' +
        'dependent: pathCount reports it and the offsets past it equal the vertex count.'
    );
  } else {
    context.controls.addSlider({
      label: 'Start measure (parameter)',
      min: 0,
      max: 1000,
      step: 10,
      value: startMeasure,
      format: value => formatMeters(value),
      onChange: value => {
        startMeasure = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'End measure (parameter)',
      min: 0,
      max: 1000,
      step: 10,
      value: endMeasure,
      format: value => formatMeters(value),
      onChange: value => {
        endMeasure = value;
        writeParameters();
      }
    });
    context.controls.addLegend({
      title: 'Lines',
      entries: [
        {color: [255, 150, 40, 255], label: 'ST_LineSubstring(road, start, end)'},
        {color: [150, 170, 215, 110], label: 'Full road'}
      ]
    });
    context.controls.addNote(
      'Measures are clamped to each road; a start past the road end gives an empty path.'
    );
  }
  context.controls.addReadout('Road paths', formatCount(pathCount));
  const piecesReadout = context.controls.addReadout(chunkMode ? 'Chunks' : 'Substrings');
  const vertexReadout = context.controls.addReadout('Output vertices');
  const overflowReadout = context.controls.addReadout('Overflow');
  context.controls.addReadout('Total road length', formatKilometers(paths.totalLength));
  context.controls.addReadout('Data', roads.attribution);
  writeParameters();

  const reader = new SummaryReader(
    resources,
    mode,
    [
      {buffer: output.count, size: 4},
      {buffer: output.overflow, size: 4},
      {buffer: output.pathCount, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      vertexReadout.setValue(formatCount(words[0]));
      overflowReadout.setValue(words[1] ? 'yes (capacity)' : 'no');
      piecesReadout.setValue(`${formatCount(words[2])} of ${formatCount(pieceCapacity)} slots`);
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, output.count, output.drawCommands);
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const origin = roads.origin;
      return [
        ...(chunkMode ? [] : [getRoadLayer('roads', roadPaths, origin)]),
        new PathOutputLayer({
          id: `${mode}-output`,
          coordinateOrigin: [origin[0], origin[1], 0],
          positions: output.positions,
          pathOffsets: output.offsets,
          pathOffsetCount: output.offsetCount,
          vertexCount: output.count,
          drawCommands: output.drawCommands,
          colorSource: chunkMode ? 'path-index' : 'uniform',
          color: chunkMode ? [255, 255, 255, 235] : [255, 150, 40, 255],
          widthPixels: chunkMode ? 2.4 : 2.5
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Views: snap and locate (linear referencing)
// ---------------------------------------------------------------------------------------------

type SnapColor = 'side' | 'measure' | 'distance';

async function createSnapView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): Promise<ViewInstance> {
  const [roads, pois] = await Promise.all([
    context.data.getNewYorkRoads(),
    context.data.getNewYorkPointsOfInterest()
  ]);
  context.signal.throwIfAborted();
  const {device} = context;
  const paths = buildRoadPaths(roads);
  const inputVertexCount = paths.positions.length / 2;
  const pathCount = paths.offsets.length - 1;
  const pointCount = pois.positions.length / 2;
  const sortedLengths = Float32Array.from(paths.lengths).sort();
  const measureRange = sortedLengths[Math.floor(sortedLengths.length * 0.95)] || 1;
  let radius = 60;
  let colorBy: SnapColor = 'side';
  let dirty = true;

  const roadPaths = createStaticPaths(resources, 'roads', paths.positions, paths.offsets);
  const points = resources.createBuffer('points', pois.positions);
  const radiusBuffer = resources.createParameterBuffer('radius', 'float32', 1);
  const footPoints = resources.createBuffer('foot-points', pointCount * 8);
  const distances = resources.createBuffer('distances', pointCount * 4);
  const measures = resources.createBuffer('measures', pointCount * 4);
  const signedOffsets = resources.createBuffer('signed-offsets', pointCount * 4);
  const pathIndices = resources.createBuffer('path-indices', pointCount * 4);
  const overflow = resources.createBuffer('overflow', 4);
  const candidateCount = resources.createBuffer('candidate-count', 4);
  const candidateCapacity = 1 << 20;

  const graph = new GPUCommandGraph<void>(device, {id: 'geometry-snap'});
  graph.add(
    new GPULinearReferencing({
      id: 'snap',
      points: importGraphBuffer(graph, 'points', points, 'float32x2', pointCount),
      positions: importGraphBuffer(
        graph,
        'positions',
        roadPaths.positions,
        'float32x2',
        inputVertexCount
      ),
      pathOffsets: importGraphBuffer(graph, 'offsets', roadPaths.offsets, 'uint32', pathCount + 1),
      radius: radiusBuffer.importToGraph(graph),
      candidateCapacity,
      spatialSort: true,
      output: {
        footPoints: importGraphBuffer(graph, 'foot-points', footPoints, 'float32x2', pointCount),
        distances: importGraphBuffer(graph, 'distances', distances, 'float32', pointCount),
        measures: importGraphBuffer(graph, 'measures', measures, 'float32', pointCount),
        signedOffsets: importGraphBuffer(
          graph,
          'signed-offsets',
          signedOffsets,
          'float32',
          pointCount
        ),
        pathIndices: importGraphBuffer(graph, 'path-indices', pathIndices, 'uint32', pointCount)
      },
      overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
      candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)
    })
  );
  const compiled = resources.track(graph.compile());

  const colorNote = context.controls.addNote('');
  const describeColor = () =>
    colorNote.setValue(
      colorBy === 'side'
        ? 'Side of the road direction: blue is left (positive offset), orange is right; the ' +
            'extremes are 40 m or the search radius if smaller.'
        : colorBy === 'measure'
          ? `Measure along the road to the foot point: 0 to ${Math.round(measureRange)} m (95th percentile of road length).`
          : 'Distance from the POI to its foot point: dark is on the road, light is 40 m or more.'
    );
  context.controls.addSlider({
    label: 'Search radius (per-frame parameter)',
    min: 5,
    max: 300,
    step: 5,
    value: radius,
    format: value => `${value} m`,
    onChange: value => {
      radius = value;
      radiusBuffer.write(Float32Array.of(radius));
      dirty = true;
      context.updateLayers();
    }
  });
  context.controls.addSelect<SnapColor>({
    label: 'Snap lines colored by',
    options: [
      {value: 'side', label: 'Side (signed offset)'},
      {value: 'measure', label: 'Measure along the road'},
      {value: 'distance', label: 'Snap distance'}
    ],
    value: colorBy,
    onChange: value => {
      colorBy = value;
      describeColor();
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Layers',
    entries: [
      {color: [255, 255, 255, 255], label: 'POI'},
      {color: [60, 230, 255, 255], label: 'Foot point on the road'},
      {color: [150, 170, 215, 110], label: 'Roads'}
    ]
  });
  describeColor();
  context.controls.addReadout(
    'POIs / roads',
    `${formatCount(pointCount)} / ${formatCount(pathCount)}`
  );
  const matchedReadout = context.controls.addReadout('Matched');
  const sideReadout = context.controls.addReadout('Left / right of road');
  const meanReadout = context.controls.addReadout('Mean snap distance');
  const candidateReadout = context.controls.addReadout('Candidates');
  context.controls.addReadout('Data', `${pois.attribution}; ${roads.attribution}`);
  radiusBuffer.write(Float32Array.of(radius));

  const reader = new SummaryReader(
    resources,
    'snap',
    [
      {buffer: signedOffsets, size: pointCount * 4},
      {buffer: distances, size: pointCount * 4},
      {buffer: overflow, size: 4},
      {buffer: candidateCount, size: 4}
    ],
    bytes => {
      const offsets = new Float32Array(bytes, 0, pointCount);
      const distanceValues = new Float32Array(bytes, pointCount * 4, pointCount);
      const words = new Uint32Array(bytes, pointCount * 8, 2);
      let matched = 0;
      let left = 0;
      let right = 0;
      let totalDistance = 0;
      for (let index = 0; index < pointCount; index++) {
        if (Number.isFinite(offsets[index])) {
          matched++;
          totalDistance += distanceValues[index];
          if (offsets[index] > 0) left++;
          else if (offsets[index] < 0) right++;
        }
      }
      matchedReadout.setValue(`${formatCount(matched)} / ${formatCount(pointCount)}`);
      sideReadout.setValue(`${formatCount(left)} / ${formatCount(right)}`);
      meanReadout.setValue(matched ? formatMeters(totalDistance / matched) : 'n/a');
      candidateReadout.setValue(
        `${formatCount(words[1])} of ${formatCount(candidateCapacity)}${words[0] ? ' (overflow)' : ''}`
      );
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const origin = roads.origin;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const valueRange: [number, number] =
        colorBy === 'side'
          ? [-Math.min(radius, 40), Math.min(radius, 40)]
          : colorBy === 'measure'
            ? [0, measureRange]
            : [0, Math.min(radius, 40)];
      return [
        getRoadLayer('snap-roads', roadPaths, origin),
        new PairSegmentLayer({
          id: 'snap-lines',
          coordinateOrigin,
          starts: points,
          ends: footPoints,
          instanceCount: pointCount,
          values: colorBy === 'side' ? signedOffsets : colorBy === 'measure' ? measures : distances,
          colormap:
            colorBy === 'side' ? 'diverging' : colorBy === 'measure' ? 'viridis' : 'inferno',
          valueRange,
          color: [255, 255, 255, 255],
          widthPixels: 3
        }),
        new SpatialAnalysisPointLayer({
          id: 'snap-foot-points',
          coordinateOrigin,
          positions: footPoints,
          instanceCount: pointCount,
          radiusPixels: 1.8,
          color: [60, 230, 255, 255]
        }),
        new SpatialAnalysisPointLayer({
          id: 'snap-points',
          coordinateOrigin,
          positions: points,
          instanceCount: pointCount,
          radiusPixels: 2.4,
          color: [255, 255, 255, 255]
        })
      ];
    }
  };
}

const EVENT_COUNT = 700;
const SWEEP_SECONDS = 10;

async function createLocateView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): Promise<ViewInstance> {
  const roads = await context.data.getNewYorkRoads();
  context.signal.throwIfAborted();
  const {device} = context;
  const paths = buildRoadPaths(roads);
  const inputVertexCount = paths.positions.length / 2;
  const pathCount = paths.offsets.length - 1;
  // Events ride the longer roads so the motion is visible.
  const random = createSeededRandom(31);
  const eligible: number[] = [];
  for (let path = 0; path < pathCount; path++) {
    if (paths.lengths[path] >= 90) eligible.push(path);
  }
  const eventPaths = new Uint32Array(EVENT_COUNT);
  const eventSpread = new Float32Array(EVENT_COUNT);
  for (let event = 0; event < EVENT_COUNT; event++) {
    eventPaths[event] = eligible.length
      ? eligible[Math.floor(random() * eligible.length)]
      : Math.floor(random() * pathCount);
    eventSpread[event] = random();
  }
  let spread = 1;
  let position = 0;
  let lateralOffset = 8;
  let animate = true;
  let parametersDirty = true;
  let offsetsDirty = true;

  const roadPaths = createStaticPaths(resources, 'roads', paths.positions, paths.offsets);
  const eventPathBuffer = resources.createBuffer('event-paths', eventPaths);
  const eventMeasureBuffer = resources.createBuffer('event-measures', eventSpread);
  const eventOffsetBuffer = resources.createBuffer('event-offsets', EVENT_COUNT * 4);
  const parameters = resources.createParameterBuffer(
    'locate-parameters',
    'float32',
    GPU_LINE_LOCATE_PARAMETER_LENGTH
  );
  const eventPositions = resources.createBuffer('event-positions', EVENT_COUNT * 8);
  const eventTangents = resources.createBuffer('event-tangents', EVENT_COUNT * 8);
  const eventStatuses = resources.createBuffer('event-statuses', EVENT_COUNT * 4);

  const graph = new GPUCommandGraph<void>(device, {id: 'geometry-locate'});
  graph.add(
    new GPULineLocate({
      id: 'locate',
      positions: importGraphBuffer(
        graph,
        'positions',
        roadPaths.positions,
        'float32x2',
        inputVertexCount
      ),
      pathOffsets: importGraphBuffer(graph, 'offsets', roadPaths.offsets, 'uint32', pathCount + 1),
      eventPaths: importGraphBuffer(graph, 'event-paths', eventPathBuffer, 'uint32', EVENT_COUNT),
      eventMeasures: importGraphBuffer(
        graph,
        'event-measures',
        eventMeasureBuffer,
        'float32',
        EVENT_COUNT
      ),
      eventOffsets: importGraphBuffer(
        graph,
        'event-offsets',
        eventOffsetBuffer,
        'float32',
        EVENT_COUNT
      ),
      measureMode: 'fraction',
      parameters: parameters.importToGraph(graph),
      output: {
        positions: importGraphBuffer(
          graph,
          'event-positions',
          eventPositions,
          'float32x2',
          EVENT_COUNT
        ),
        tangents: importGraphBuffer(
          graph,
          'event-tangents',
          eventTangents,
          'float32x2',
          EVENT_COUNT
        ),
        statuses: importGraphBuffer(graph, 'event-statuses', eventStatuses, 'uint32', EVENT_COUNT)
      }
    })
  );
  const compiled = resources.track(graph.compile());

  const writeParameters = () => {
    parameters.write(
      getGPULineLocateParameterValues({measureScale: spread, measureOffset: position})
    );
    parametersDirty = true;
  };
  const writeOffsets = () => {
    // Alternate left and right lanes so the lateral offset reads as two traffic directions.
    eventOffsetBuffer.write(
      Float32Array.from(
        {length: EVENT_COUNT},
        (_, event) => (event % 2 === 0 ? 1 : -1) * lateralOffset
      )
    );
    offsetsDirty = true;
  };
  const positionSlider = context.controls.addSlider({
    label: 'Position offset along each road (measure = base * spread + offset)',
    min: -0.5,
    max: 1.5,
    step: 0.01,
    value: position,
    format: value => `${Math.round(value * 100)}% of the road`,
    onChange: value => {
      position = value;
      writeParameters();
    }
  });
  context.controls.addSlider({
    label: 'Spread of the base fractions (measure scale)',
    min: 0,
    max: 1,
    step: 0.05,
    value: spread,
    format: value => value.toFixed(2),
    onChange: value => {
      spread = value;
      writeParameters();
    }
  });
  context.controls.addSlider({
    label: 'Lateral offset (events in alternating lanes)',
    min: 0,
    max: 40,
    step: 1,
    value: lateralOffset,
    format: value => `${value} m`,
    onChange: value => {
      lateralOffset = value;
      writeOffsets();
    }
  });
  context.controls.addToggle({
    label: 'Sweep along the roads (animated parameter)',
    value: animate,
    onChange: value => {
      animate = value;
    }
  });
  context.controls.addLegend({
    title: 'Layers',
    entries: [
      {color: [255, 150, 40, 255], label: 'Event positions (ST_LineInterpolatePoint)'},
      {color: [255, 255, 255, 230], label: 'Tangent tick (25 m)'},
      {color: [150, 170, 215, 110], label: 'Roads'}
    ]
  });
  context.controls.addNote(
    'Fractions are clamped to each road: events reaching an end stop there (status clamped).'
  );
  context.controls.addReadout(
    'Events / roads',
    `${formatCount(EVENT_COUNT)} / ${formatCount(pathCount)}`
  );
  const statusReadout = context.controls.addReadout('Clamped / invalid');
  context.controls.addReadout('Data', roads.attribution);
  writeParameters();
  writeOffsets();

  const reader = new SummaryReader(
    resources,
    'locate',
    [{buffer: eventStatuses, size: EVENT_COUNT * 4}],
    bytes => {
      const statuses = new Uint32Array(bytes);
      let clamped = 0;
      let invalid = 0;
      for (const status of statuses) {
        if (status === 1) clamped++;
        else if (status === 2) invalid++;
      }
      statusReadout.setValue(`${formatCount(clamped)} / ${formatCount(invalid)}`);
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame: SpatialAnalysisFrame) {
      if (animate) {
        // Triangle wave 0 -> 1 -> 0 so events sweep each road back and forth.
        const phase = (frame.timeSeconds / SWEEP_SECONDS) % 1;
        const wave = 1 - Math.abs(2 * phase - 1);
        position = wave * (1 - 0.5 * spread);
        parameters.write(
          getGPULineLocateParameterValues({measureScale: spread, measureOffset: position})
        );
        positionSlider.setValue(position);
        parametersDirty = true;
      }
      if (parametersDirty || offsetsDirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        parametersDirty = false;
        offsetsDirty = false;
      }
      if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
      reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const origin = roads.origin;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      return [
        getRoadLayer('locate-roads', roadPaths, origin),
        new PairSegmentLayer({
          id: 'locate-ticks',
          coordinateOrigin,
          starts: eventPositions,
          ends: eventTangents,
          directionScale: 25,
          instanceCount: EVENT_COUNT,
          color: [255, 255, 255, 230],
          widthPixels: 1.6
        }),
        new SpatialAnalysisPointLayer({
          id: 'locate-events',
          coordinateOrigin,
          positions: eventPositions,
          instanceCount: EVENT_COUNT,
          radiusPixels: 4,
          color: [255, 150, 40, 255]
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: zone measures
// ---------------------------------------------------------------------------------------------

type MeasureSystem = 'planar' | 'spherical' | 'wgs84';
type MeasureQuantity = 'area' | 'perimeter';

const MEASURE_SYSTEMS: readonly {id: MeasureSystem; label: string}[] = [
  {id: 'planar', label: 'Planar (local meters)'},
  {id: 'spherical', label: 'Spherical (haversine, Chamberlain-Duquette)'},
  {id: 'wgs84', label: 'WGS84 (Vincenty, authalic area)'}
];
const GROUP_COUNT = 3;

async function createMeasuresView(
  context: SpatialAnalysisModeContext,
  resources: SpatialAnalysisResources
): Promise<ViewInstance> {
  const zones: SpatialAnalysisPolygons = await context.data.getSanFranciscoZipCodes();
  context.signal.throwIfAborted();
  const {device} = context;
  const projection = new LocalMetricProjection(zones.origin);
  const featureCount = zones.featureIds.length;
  const vertexCount = zones.polygonPositions.length / 2;
  const edgeCount = zones.outlineSegments.length / 4;
  const geographic = new Float32Array(zones.polygonPositions.length);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const [longitude, latitude] = projection.unproject(
      zones.polygonPositions[vertex * 2],
      zones.polygonPositions[vertex * 2 + 1]
    );
    geographic[vertex * 2] = longitude;
    geographic[vertex * 2 + 1] = latitude;
  }
  // Features own rings through their polygon (one polygon per feature in these datasets).
  const featureRingOffsets = new Uint32Array(featureCount + 1);
  for (let feature = 0; feature <= featureCount; feature++) {
    featureRingOffsets[feature] = zones.polygonOffsets[zones.featureOffsets[feature]];
  }
  // Three latitude bands by planar centroid, for the group totals.
  const centroidY = new Float32Array(featureCount);
  for (let feature = 0; feature < featureCount; feature++) {
    const first = zones.ringOffsets[featureRingOffsets[feature]];
    const last = zones.ringOffsets[featureRingOffsets[feature + 1]];
    let sum = 0;
    for (let vertex = first; vertex < last; vertex++) sum += zones.polygonPositions[vertex * 2 + 1];
    centroidY[feature] = sum / Math.max(1, last - first);
  }
  const sortedCentroids = Array.from(centroidY).sort((left, right) => left - right);
  const groupIds = Uint32Array.from(centroidY, value => {
    const rank = sortedCentroids.indexOf(value);
    return Math.min(GROUP_COUNT - 1, Math.floor((rank * GROUP_COUNT) / featureCount));
  });

  let system: MeasureSystem = 'wgs84';
  let quantity: MeasureQuantity = 'area';
  let totals = {area: [0, 0, 0], perimeter: [0, 0, 0]};
  let columns = {area: new Float32Array(0), perimeter: new Float32Array(0)};
  let valueRange: [number, number] = [0, 1];

  const planarPositions = resources.createBuffer('planar-positions', zones.polygonPositions);
  const geographicPositions = resources.createBuffer('geographic-positions', geographic);
  const ringOffsets = resources.createBuffer('ring-offsets', zones.ringOffsets);
  const featureRings = resources.createBuffer('feature-rings', featureRingOffsets);
  const groupBuffer = resources.createBuffer('group-ids', groupIds);
  const segmentsBuffer = resources.createBuffer('edge-segments', zones.outlineSegments);
  const featureRowsBuffer = resources.createBuffer('edge-feature-rows', zones.outlineFeatureRows);
  const centroids = resources.createBuffer('centroids', featureCount * 8);
  const outputs = MEASURE_SYSTEMS.map(entry => ({
    id: entry.id,
    areas: resources.createBuffer(`${entry.id}-areas`, featureCount * 4),
    lengths: resources.createBuffer(`${entry.id}-lengths`, featureCount * 4)
  }));
  const groupAreas = resources.createBuffer('group-areas', GROUP_COUNT * 4);
  const groupLengths = resources.createBuffer('group-lengths', GROUP_COUNT * 4);
  const groupCounts = resources.createBuffer('group-counts', GROUP_COUNT * 4);

  const graph = new GPUCommandGraph<void>(device, {id: 'geometry-measures'});
  const featureRingView = importGraphBuffer(
    graph,
    'feature-rings',
    featureRings,
    'uint32',
    featureCount + 1
  );
  const ringView = importGraphBuffer(
    graph,
    'ring-offsets',
    ringOffsets,
    'uint32',
    zones.ringOffsets.length
  );
  for (const entry of MEASURE_SYSTEMS) {
    const buffers = outputs.find(output => output.id === entry.id)!;
    const planar = entry.id === 'planar';
    graph.add(
      new GPUGeometryMeasures({
        id: `measures-${entry.id}`,
        positions: importGraphBuffer(
          graph,
          `positions-${entry.id}`,
          planar ? planarPositions : geographicPositions,
          'float32x2',
          vertexCount
        ),
        geometryType: 'polygons',
        ringOffsets: ringView,
        featureRingOffsets: featureRingView,
        coordinateSystem: entry.id,
        holeRule: 'first-ring-exterior',
        output: {
          areas: importGraphBuffer(
            graph,
            `${entry.id}-areas`,
            buffers.areas,
            'float32',
            featureCount
          ),
          lengths: importGraphBuffer(
            graph,
            `${entry.id}-lengths`,
            buffers.lengths,
            'float32',
            featureCount
          ),
          ...(planar
            ? {
                centroids: importGraphBuffer(
                  graph,
                  'centroids',
                  centroids,
                  'float32x2',
                  featureCount
                )
              }
            : {})
        },
        ...(entry.id === 'wgs84'
          ? {
              groupIds: importGraphBuffer(graph, 'group-ids', groupBuffer, 'uint32', featureCount),
              groupCount: GROUP_COUNT,
              groupOutput: {
                areas: importGraphBuffer(graph, 'group-areas', groupAreas, 'float32', GROUP_COUNT),
                lengths: importGraphBuffer(
                  graph,
                  'group-lengths',
                  groupLengths,
                  'float32',
                  GROUP_COUNT
                ),
                featureCounts: importGraphBuffer(
                  graph,
                  'group-counts',
                  groupCounts,
                  'uint32',
                  GROUP_COUNT
                )
              }
            }
          : {})
      })
    );
  }
  const compiled = resources.track(graph.compile());

  const updateRange = () => {
    const offset = MEASURE_SYSTEMS.findIndex(entry => entry.id === system) * featureCount;
    const slice = columns[quantity].subarray(offset, offset + featureCount);
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const value of slice) {
      if (Number.isFinite(value)) {
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    }
    valueRange = minimum < maximum ? [minimum, maximum] : [0, 1];
  };

  context.controls.addSelect<MeasureSystem>({
    label: 'Coordinate system (three prebuilt nodes; the select picks a column)',
    options: MEASURE_SYSTEMS.map(entry => ({value: entry.id, label: entry.label})),
    value: system,
    onChange: value => {
      system = value;
      updateRange();
      context.updateLayers();
    }
  });
  context.controls.addSelect<MeasureQuantity>({
    label: 'Choropleth',
    options: [
      {value: 'area', label: 'Area'},
      {value: 'perimeter', label: 'Perimeter'}
    ],
    value: quantity,
    onChange: value => {
      quantity = value;
      updateRange();
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Zone color: selected measure (min to max of the zones)',
    gradient: {
      colors: [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
      ],
      minimumLabel: 'smallest',
      maximumLabel: 'largest'
    }
  });
  context.controls.addNote(
    'Zones are filled as triangle fans from the GPU centroid column, exact for polygons that are ' +
      'star-shaped around it. White dots are the planar centroids.'
  );
  context.controls.addReadout(
    'Zones / vertices',
    `${formatCount(featureCount)} / ${formatCount(vertexCount)}`
  );
  const areaReadouts = MEASURE_SYSTEMS.map(entry =>
    context.controls.addReadout(`Total area, ${entry.id}`)
  );
  const perimeterReadout = context.controls.addReadout('Total perimeter (planar / WGS84)');
  const differenceReadout = context.controls.addReadout('WGS84 vs planar area');
  const groupReadout = context.controls.addReadout('WGS84 groups S / C / N');
  context.controls.addReadout('Data', zones.attribution);

  const groupBytes = GROUP_COUNT * 4;
  const reader = new SummaryReader(
    resources,
    'measures',
    [
      ...outputs.flatMap(output => [
        {buffer: output.areas, size: featureCount * 4},
        {buffer: output.lengths, size: featureCount * 4}
      ]),
      {buffer: groupAreas, size: groupBytes},
      {buffer: groupCounts, size: groupBytes}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      const area = new Float32Array(featureCount * 3);
      const perimeter = new Float32Array(featureCount * 3);
      outputs.forEach((_, index) => {
        area.set(
          floats.subarray(index * 2 * featureCount, (index * 2 + 1) * featureCount),
          index * featureCount
        );
        perimeter.set(
          floats.subarray((index * 2 + 1) * featureCount, (index * 2 + 2) * featureCount),
          index * featureCount
        );
      });
      columns = {area, perimeter};
      const sum = (values: Float32Array, index: number) =>
        values
          .subarray(index * featureCount, (index + 1) * featureCount)
          .reduce((a, b) => a + b, 0);
      totals = {
        area: [0, 1, 2].map(index => sum(area, index)),
        perimeter: [0, 1, 2].map(index => sum(perimeter, index))
      };
      areaReadouts.forEach((readout, index) =>
        readout.setValue(`${(totals.area[index] / 1e6).toFixed(3)} km²`)
      );
      perimeterReadout.setValue(
        `${(totals.perimeter[0] / 1000).toFixed(1)} / ${(totals.perimeter[2] / 1000).toFixed(1)} km`
      );
      differenceReadout.setValue(
        `${(((totals.area[2] - totals.area[0]) / totals.area[0]) * 100).toFixed(3)} %`
      );
      const groupBase = outputs.length * 2 * featureCount;
      const groupArea = floats.subarray(groupBase, groupBase + GROUP_COUNT);
      const groupCountWords = new Uint32Array(bytes, (groupBase + GROUP_COUNT) * 4, GROUP_COUNT);
      groupReadout.setValue(
        Array.from(
          groupArea,
          (value, index) => `${(value / 1e6).toFixed(1)} (${groupCountWords[index]})`
        ).join(' / ') + ` = ${(groupArea.reduce((a, b) => a + b, 0) / 1e6).toFixed(2)} km²`
      );
      updateRange();
      context.updateLayers();
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      // The zones are static, so the graph runs for the first frames and results persist.
      if (frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
      }
      if (frame.frameIndex === 4) reader.markStale();
      reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const origin = zones.origin;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const chosen = outputs[MEASURE_SYSTEMS.findIndex(entry => entry.id === system)];
      const values = quantity === 'area' ? chosen.areas : chosen.lengths;
      return [
        new PolygonFanLayer({
          id: 'measures-fill',
          coordinateOrigin,
          segments: segmentsBuffer,
          featureRows: featureRowsBuffer,
          centroids,
          instanceCount: edgeCount,
          values,
          colormap: 'viridis',
          valueRange,
          color: [255, 255, 255, 190]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'measures-outline',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: edgeCount,
          widthPixels: 1.5,
          color: [15, 20, 35, 230]
        }),
        new SpatialAnalysisPointLayer({
          id: 'measures-centroids',
          coordinateOrigin,
          positions: centroids,
          instanceCount: featureCount,
          radiusPixels: 3.5,
          color: [255, 255, 255, 255]
        })
      ];
    }
  };
}
