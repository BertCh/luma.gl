// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUBufferSelection
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColormap
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {formatInteger} from './b2-geometry';

/** Option state of the near-transit scene. */
export type NearTransitOptions = {
  points: 'observations' | 'places' | 'crashes';
  features: 'stations' | 'lines' | 'bus';
  line: string;
  distance: number;
  spatialSort: boolean;
  colorBy: 'distance' | 'line' | 'single';
  ramp: RampName;
  showBuffer: boolean;
  showOutside: boolean;
  pointSize: number;
  opacity: number;
};

/** The eight L routes in `properties.routes` order, with their official colors. */
export const L_LINES = [
  {id: 'Red', label: 'Red Line', color: [198, 12, 48]},
  {id: 'P', label: 'Purple Line', color: [150, 90, 230]},
  {id: 'Y', label: 'Yellow Line', color: [249, 227, 0]},
  {id: 'Blue', label: 'Blue Line', color: [0, 161, 222]},
  {id: 'Pink', label: 'Pink Line', color: [226, 126, 166]},
  {id: 'G', label: 'Green Line', color: [0, 155, 58]},
  {id: 'Org', label: 'Orange Line', color: [249, 110, 40]},
  {id: 'Brn', label: 'Brown Line', color: [160, 100, 60]}
] as const;

type PointSource = {
  id: NearTransitOptions['points'];
  label: string;
  positions: Float32Array;
  count: number;
  categories: Uint8Array;
  categoryNames: readonly string[];
  buffer: Buffer;
};

type FeatureKind = {
  kind: NearTransitOptions['features'];
  /** Largest distance the candidate capacity supports, meters. */
  maximumDistance: number;
  candidatesPerPoint: number;
};

const FEATURE_KINDS: Record<NearTransitOptions['features'], FeatureKind> = {
  stations: {kind: 'stations', maximumDistance: 2000, candidatesPerPoint: 24},
  lines: {kind: 'lines', maximumDistance: 1000, candidatesPerPoint: 40},
  bus: {kind: 'bus', maximumDistance: 500, candidatesPerPoint: 24}
};

const SETTLE_FRAMES = 4;
const NO_FEATURE = GPU_SPATIAL_JOIN_NO_FEATURE;
const LINE_PALETTE = L_LINES.map(line => [...line.color, 255] as const);

type Variant = {
  key: string;
  points: PointSource;
  features: NearTransitOptions['features'];
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  mask: Buffer;
  ids: Buffer;
  distances: Buffer;
  pointLine: Buffer;
};

/**
 * Select-by-distance of points around CTA features. `GPUBufferSelection` joins every point to the
 * nearest feature within a per-frame distance and publishes a 0/1 mask, the ordered ids of the
 * selected points and an instance count that a GPU-written indirect draw reads directly, so the
 * selection is drawn without a readback. The feature subset (which L line) is chosen by rewriting
 * the feature buffers: unselected rows are parked far outside the data, never recompiled.
 */
export async function createNearTransit(
  ctx: SceneContext<NearTransitOptions>
): Promise<SceneInstance<NearTransitOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const places = ctx.datasets.get('chicago-places');
  const crashes = ctx.datasets.get('chicago-crashes');
  const transit = ctx.datasets.get('cta-transit');
  const origin = transit.defaultOrigin;

  const resources = new SpatialAnalysisResources(device, 'near-transit');

  const pointSources: Record<NearTransitOptions['points'], PointSource> = {
    observations: makeSource('observations', 'Nature observations', observations, 'category'),
    places: makeSource('places', 'Places', places, 'category'),
    crashes: makeSource('crashes', 'Traffic crashes', crashes, 'severity')
  };
  function makeSource(
    id: NearTransitOptions['points'],
    label: string,
    dataset: typeof observations,
    categoryColumn: string
  ): PointSource {
    const positions = dataset.projectColumn('position', origin);
    return {
      id,
      label,
      positions,
      count: positions.length / 2,
      categories: dataset.column<Uint8Array>(categoryColumn),
      categoryNames: dataset.categories(categoryColumn),
      buffer: resources.createBuffer(`${id}-positions`, positions)
    };
  }

  // --- features: stations, lines (deduplicated segments) and bus stops --------------------------
  const stopPositions = transit.projectColumn('stopPosition', origin);
  const stopMode = transit.column<Uint8Array>('stopMode');
  const stopRoutes = transit.column<Uint16Array>('stopRoutes');
  const stopRouteOffsets = transit.column<Uint32Array>('stopRouteOffsets');
  const routes = (transit.properties.routes as {type: string; id: string}[]) ?? [];
  const lineRouteIndex = new Map<number, number>();
  routes.forEach((route, row) => {
    if (route.type === 'rail') {
      const lineIndex = L_LINES.findIndex(line => line.id === route.id);
      if (lineIndex >= 0) lineRouteIndex.set(row, lineIndex);
    }
  });

  const stationRows: number[] = [];
  const busRows: number[] = [];
  stopMode.forEach((mode, row) => (mode === 1 ? stationRows : busRows).push(row));
  const stationLines: number[][] = stationRows.map(row => {
    const lines: number[] = [];
    for (let slot = stopRouteOffsets[row]; slot < stopRouteOffsets[row + 1]; slot++) {
      const line = lineRouteIndex.get(stopRoutes[slot]);
      if (line !== undefined && !lines.includes(line)) lines.push(line);
    }
    return lines;
  });
  const stationPrimaryLine = Uint32Array.from(stationLines.map(lines => lines[0] ?? 0));
  const stationPositions = new Float32Array(stationRows.length * 2);
  stationRows.forEach((row, index) =>
    stationPositions.set(stopPositions.subarray(row * 2, row * 2 + 2), index * 2)
  );
  const busPositions = new Float32Array(busRows.length * 2);
  busRows.forEach((row, index) =>
    busPositions.set(stopPositions.subarray(row * 2, row * 2 + 2), index * 2)
  );

  const shapeVertices = transit.projectColumn('shapeVertices', origin);
  const shapeOffsets = transit.column<Uint32Array>('shapePathOffsets');
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeDirection = transit.column<Uint8Array>('shapeDirection');
  const forwardRoutes = new Set<number>();
  for (let shape = 0; shape < shapeRoute.length; shape++) {
    if (lineRouteIndex.has(shapeRoute[shape]) && shapeDirection[shape] === 0)
      forwardRoutes.add(shapeRoute[shape]);
  }
  const segmentList: number[] = [];
  const segmentLine: number[] = [];
  const seen = new Set<string>();
  for (let shape = 0; shape < shapeRoute.length; shape++) {
    const route = shapeRoute[shape];
    const line = lineRouteIndex.get(route);
    if (line === undefined) continue;
    if (forwardRoutes.has(route) && shapeDirection[shape] !== 0) continue;
    for (let vertex = shapeOffsets[shape]; vertex + 1 < shapeOffsets[shape + 1]; vertex++) {
      const x0 = shapeVertices[vertex * 2];
      const y0 = shapeVertices[vertex * 2 + 1];
      const x1 = shapeVertices[vertex * 2 + 2];
      const y1 = shapeVertices[vertex * 2 + 3];
      // Segments shared by several routes (the Loop) or by both directions are kept once.
      const key = [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)]
        .sort((a, b) => a - b)
        .join(',');
      const reverse = `${Math.round(x1)},${Math.round(y1)},${Math.round(x0)},${Math.round(y0)}`;
      if (seen.has(key) || seen.has(reverse)) continue;
      seen.add(key);
      segmentList.push(x0, y0, x1, y1);
      segmentLine.push(line);
    }
  }
  const lineSegments = Float32Array.from(segmentList);
  const lineSegmentCount = segmentLine.length;
  const lineSegmentLine = Uint32Array.from(segmentLine);

  // Parking position far beyond the data keeps unselected rows out of every distance.
  let parkedX = -Infinity;
  let parkedY = -Infinity;
  for (const source of Object.values(pointSources)) {
    for (let index = 0; index < source.positions.length; index += 2) {
      parkedX = Math.max(parkedX, source.positions[index]);
      parkedY = Math.max(parkedY, source.positions[index + 1]);
    }
  }
  parkedX += 20000;
  parkedY += 20000;

  const stationPositionsLive = Float32Array.from(stationPositions);
  const busPositionsLive = Float32Array.from(busPositions);
  const lineStarts = new Float32Array(lineSegmentCount * 2);
  const lineEnds = new Float32Array(lineSegmentCount * 2);
  const stationBuffer = resources.createBuffer('stations', stationPositionsLive);
  const busBuffer = resources.createBuffer('bus', busPositionsLive);
  const startsBuffer = resources.createBuffer('line-starts', lineStarts);
  const endsBuffer = resources.createBuffer('line-ends', lineEnds);
  const lineSegmentsBuffer = resources.createBuffer('line-segments', lineSegments);
  const lineSegmentLineBuffer = resources.createBuffer('line-segment-line', lineSegmentLine);
  const stationLineBuffer = resources.createBuffer('station-line', stationPrimaryLine);
  const selectedSegmentsBuffer = resources.createBuffer(
    'selected-segments',
    lineSegments.length * 4 || 16
  );
  const selectedSegmentLineBuffer = resources.createBuffer(
    'selected-segment-line',
    Math.max(lineSegmentCount, 1) * 4
  );
  const selectedStationsBuffer = resources.createBuffer(
    'selected-stations',
    Math.max(stationRows.length, 1) * 8
  );
  const selectedStationLineBuffer = resources.createBuffer(
    'selected-station-line',
    Math.max(stationRows.length, 1) * 4
  );
  let selectedSegmentCount = 0;
  let selectedStationCount = 0;

  const distanceParameter = resources.createParameterBuffer(
    'distance',
    'float32',
    1,
    Float32Array.of(ctx.options.distance)
  );
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'near-transit-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // --- state -----------------------------------------------------------------------------------
  let destroyed = false;
  let variant: Variant;
  let dirty = true;
  let settle = 0;
  let measuring = false;
  let lastZoom = NaN;
  let selectedFeatureCount = 0;

  const getKey = (state: NearTransitOptions) =>
    `${state.points}|${state.features}|${state.spatialSort}`;

  /** Rewrites the feature buffers for the chosen L line; unselected rows are parked. */
  function writeSubset(state: NearTransitOptions): void {
    const lineIndex = state.line === 'all' ? -1 : L_LINES.findIndex(line => line.id === state.line);
    const compactSegments: number[] = [];
    const compactSegmentLine: number[] = [];
    let selectedSegments = 0;
    for (let segment = 0; segment < lineSegmentCount; segment++) {
      const selected = lineIndex < 0 || lineSegmentLine[segment] === lineIndex;
      if (selected) {
        selectedSegments++;
        lineStarts.set(lineSegments.subarray(segment * 4, segment * 4 + 2), segment * 2);
        lineEnds.set(lineSegments.subarray(segment * 4 + 2, segment * 4 + 4), segment * 2);
        compactSegments.push(...lineSegments.subarray(segment * 4, segment * 4 + 4));
        compactSegmentLine.push(lineSegmentLine[segment]);
      } else {
        lineStarts[segment * 2] = lineEnds[segment * 2] = parkedX;
        lineStarts[segment * 2 + 1] = lineEnds[segment * 2 + 1] = parkedY;
      }
    }
    startsBuffer.write(lineStarts);
    endsBuffer.write(lineEnds);
    selectedSegmentCount = compactSegmentLine.length;
    selectedSegmentsBuffer.write(Float32Array.from(compactSegments));
    selectedSegmentLineBuffer.write(Uint32Array.from(compactSegmentLine));

    const compactStations: number[] = [];
    const compactStationLine: number[] = [];
    let selectedStations = 0;
    stationRows.forEach((_, station) => {
      const selected = lineIndex < 0 || stationLines[station].includes(lineIndex);
      if (selected) {
        selectedStations++;
        stationPositionsLive.set(
          stationPositions.subarray(station * 2, station * 2 + 2),
          station * 2
        );
        compactStations.push(stationPositions[station * 2], stationPositions[station * 2 + 1]);
        compactStationLine.push(lineIndex >= 0 ? lineIndex : stationPrimaryLine[station]);
      } else {
        stationPositionsLive[station * 2] = parkedX;
        stationPositionsLive[station * 2 + 1] = parkedY;
      }
    });
    stationBuffer.write(stationPositionsLive);
    selectedStationCount = compactStationLine.length;
    selectedStationsBuffer.write(Float32Array.from(compactStations));
    selectedStationLineBuffer.write(Uint32Array.from(compactStationLine));
    selectedFeatureCount =
      state.features === 'lines'
        ? selectedSegments
        : state.features === 'stations'
          ? selectedStations
          : busRows.length;
    ctx.setReadout(
      'features',
      state.features === 'lines'
        ? `${formatInteger(selectedSegments)} of ${formatInteger(lineSegmentCount)} track segments`
        : state.features === 'stations'
          ? `${formatInteger(selectedStations)} of ${formatInteger(stationRows.length)} L stations`
          : `${formatInteger(busRows.length)} bus stops`
    );
  }

  /** Builds the compiled graph and buffers of one (points, features, sort) combination. */
  function buildVariant(state: NearTransitOptions, forTiming = false): Variant {
    const source = pointSources[state.points];
    const key = getKey(state);
    const own = new SpatialAnalysisResources(device, `nt-${key}`);
    const graph = new GPUCommandGraph<void>(device, {id: `nt-${key}`});
    const kind = FEATURE_KINDS[state.features];
    const count = source.count;
    const mask = own.createBuffer('mask', count * 4);
    const ids = own.createBuffer('ids', count * 4);
    const outCount = own.createBuffer('count', 4);
    const outOverflow = own.createBuffer('output-overflow', 4);
    const outTotal = own.createBuffer('total', 4);
    const distances = own.createBuffer('distances', count * 4);
    const nearestIds = own.createBuffer('nearest-ids', count * 4);
    const joinOverflow = own.createBuffer('join-overflow', 4);
    const pointLine = own.createBuffer('point-line', count * 4);
    const nearestView = importGraphBuffer(graph, 'nearest-ids', nearestIds, 'uint32', count);
    const features =
      state.features === 'lines'
        ? {
            kind: 'segments' as const,
            starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', lineSegmentCount),
            ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', lineSegmentCount)
          }
        : {
            kind: 'points' as const,
            positions: importGraphBuffer(
              graph,
              'feature-positions',
              state.features === 'stations' ? stationBuffer : busBuffer,
              'float32x2',
              state.features === 'stations' ? stationRows.length : busRows.length
            )
          };
    graph.add(
      new GPUBufferSelection({
        id: 'near-transit',
        points: importGraphBuffer(graph, 'points', source.buffer, 'float32x2', count),
        features,
        distance: distanceParameter.importToGraph(graph),
        candidateCapacity: count * kind.candidatesPerPoint,
        spatialSort: state.spatialSort,
        outputMask: importGraphBuffer(graph, 'mask', mask, 'uint32', count),
        output: {
          ids: importGraphBuffer(graph, 'ids', ids, 'uint32', count),
          count: importGraphBuffer(graph, 'count', outCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'output-overflow', outOverflow, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'total', outTotal, 'uint32', 1)
        },
        // The clamped selected count lands directly in the indirect draw record.
        drawInstanceCount: graph.importGPUData(
          'draw-instance-count',
          drawCommands.getInstanceCountData(0)
        ),
        distances: importGraphBuffer(graph, 'distances', distances, 'float32', count),
        nearestFeatureIds: nearestView,
        overflow: importGraphBuffer(graph, 'join-overflow', joinOverflow, 'uint32', 1)
      })
    );
    // The L line of each point's nearest feature, for coloring by line.
    addKernelPass(graph, {
      id: 'point-line',
      invocationCount: count,
      bindings: [
        {name: 'nearest', view: nearestView, type: 'u32', access: 'read'},
        ...(state.features === 'bus'
          ? []
          : [
              {
                name: 'rowLine',
                view: importGraphBuffer(
                  graph,
                  'row-line',
                  state.features === 'lines' ? lineSegmentLineBuffer : stationLineBuffer,
                  'uint32',
                  state.features === 'lines'
                    ? Math.max(lineSegmentCount, 1)
                    : Math.max(stationRows.length, 1)
                ),
                type: 'u32' as const,
                access: 'read' as const
              }
            ]),
        {
          name: 'pointLine',
          view: importGraphBuffer(graph, 'point-line', pointLine, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `
  let nearestRow = nearest[nearestOffset + index];
  if (nearestRow == ${NO_FEATURE}u) { pointLine[pointLineOffset + index] = ${NO_FEATURE}u; return; }
  pointLine[pointLineOffset + index] = ${state.features === 'bus' ? '0u' : 'rowLine[rowLineOffset + nearestRow]'};`
    });

    const reader = new SummaryReader(
      own,
      `summary-${key}`,
      [
        {buffer: outCount, size: 4},
        {buffer: outOverflow, size: 4},
        {buffer: joinOverflow, size: 4},
        {buffer: mask, size: count * 4}
      ],
      bytes => {
        if (destroyed || variant?.key !== key) return;
        const header = new Uint32Array(bytes, 0, 3);
        const mask32 = new Uint32Array(bytes, 12, count);
        const selected = header[0];
        ctx.setReadout(
          'selected',
          `${formatInteger(selected)} of ${formatInteger(count)} (${((100 * selected) / count).toFixed(1)}%)`
        );
        ctx.setReadout('overflow', header[1] || header[2] ? 'YES (capacity)' : 'no');
        // Category composition: which categories are over-represented inside the buffer.
        const names = source.categoryNames;
        const inside = new Float64Array(names.length);
        const all = new Float64Array(names.length);
        for (let index = 0; index < count; index++) {
          const category = source.categories[index];
          all[category]++;
          if (mask32[index]) inside[category]++;
        }
        const lifts = names
          .map((name, category) => ({
            name,
            lift:
              selected > 0 && all[category] > 0
                ? inside[category] / selected / (all[category] / count)
                : 0,
            count: inside[category]
          }))
          .filter(entry => entry.count >= 50)
          .sort((a, b) => b.lift - a.lift);
        ctx.setReadout(
          'composition',
          selected === 0 || lifts.length === 0
            ? 'n/a'
            : `Over-represented: ${lifts
                .slice(0, 3)
                .map(entry => `${entry.name.toLowerCase()} ${entry.lift.toFixed(2)}x`)
                .join(', ')}. Under: ${lifts
                .slice(-2)
                .reverse()
                .map(entry => `${entry.name.toLowerCase()} ${entry.lift.toFixed(2)}x`)
                .join(', ')}`
        );
      }
    );
    const compiled = own.track(graph.compile());
    void forTiming;
    void kind;
    return {
      key,
      points: source,
      features: state.features,
      resources: own,
      compiled,
      reader,
      mask,
      ids,
      distances,
      pointLine
    };
  }

  function replaceVariant(state: NearTransitOptions): void {
    const previous = variant;
    variant = buildVariant(state);
    dirty = true;
    if (previous) {
      previous.reader.stop();
      requestAnimationFrame(() => requestAnimationFrame(() => previous.resources.destroy()));
    }
    writeSubset(state);
    updateStatic(state);
  }

  function updateStatic(state: NearTransitOptions): void {
    const kind = FEATURE_KINDS[state.features];
    ctx.setReadout(
      'pointsTotal',
      `${formatInteger(variant.points.count)} ${variant.points.label.toLowerCase()}`
    );
    ctx.setReadout(
      'distanceUsed',
      state.distance > kind.maximumDistance
        ? `${formatInteger(kind.maximumDistance)} m (capped for this feature layer's candidate capacity)`
        : `${formatInteger(state.distance)} m`
    );
  }

  writeSubset(ctx.options);
  variant = buildVariant(ctx.options);
  updateStatic(ctx.options);

  /** Times the active selection graph and the other `spatialSort` choice outside the frame. */
  async function measureSort(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timeSort', 'measuring...');
    const state = ctx.options;
    const results: string[] = [];
    try {
      for (const spatialSort of [false, true]) {
        const built = buildVariant({...state, spatialSort}, true);
        try {
          const timing = await measureCompiledGraph(device, built.compiled, {
            parameters: undefined,
            completionBuffer: built.mask,
            signal: ctx.signal
          });
          results.push(
            `${spatialSort ? 'sorted' : 'unsorted'} ${formatCompiledGraphTiming(timing).split(' · ')[0]}`
          );
        } finally {
          built.compiled.destroy();
          built.reader.stop();
          built.resources.destroy();
        }
      }
      if (!destroyed) ctx.setReadout('timeSort', results.join(' | '));
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
      dirty = true;
    }
  }

  const getMetersPerPixel = (): number => {
    const viewport = ctx.getViewport();
    if (!viewport) return 20;
    const [, latitude] = viewport.unproject([viewport.width / 2, viewport.height / 2]);
    return (40075016.686 * Math.cos((latitude * Math.PI) / 180)) / (512 * 2 ** viewport.zoom);
  };

  return {
    getCompiledGraphs: () => [variant.compiled as CompiledGPUCommandGraph<never>],

    setOption(id, _value, state) {
      switch (id) {
        case 'points':
        case 'features':
        case 'spatialSort':
          if (getKey(state) !== variant.key) replaceVariant(state);
          else dirty = true;
          if (id === 'features') writeSubset(state);
          break;
        case 'line':
          writeSubset(state);
          dirty = true;
          break;
        case 'distance':
          dirty = true;
          updateStatic(state);
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureSort();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const state = ctx.options;
      const kind = FEATURE_KINDS[state.features];
      if (dirty) {
        distanceParameter.write(Float32Array.of(Math.min(state.distance, kind.maximumDistance)));
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        settle = SETTLE_FRAMES;
      } else if (settle > 0) {
        settle--;
        if (settle === 0) variant.reader.request(commandEncoder);
      }
      variant.reader.flush(commandEncoder);
      // The buffer width is in pixels: refresh the layers when the zoom changed noticeably.
      const zoom = ctx.getViewport()?.zoom;
      if (state.showBuffer && zoom !== undefined && Math.abs(zoom - lastZoom) > 0.04) {
        lastZoom = zoom;
        ctx.requestLayers();
      }
    },

    getLayers() {
      const state = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const kind = FEATURE_KINDS[state.features];
      const distance = Math.min(state.distance, kind.maximumDistance);
      const bufferPixels = Math.max(2, (2 * distance) / getMetersPerPixel());
      const layers: Layer[] = [];

      // Buffer footprint around the selected features, under everything.
      if (state.showBuffer && state.features === 'lines') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'buffer-lines',
            coordinateOrigin,
            segments: selectedSegmentsBuffer,
            instanceCount: selectedSegmentCount,
            color: dark ? [90, 200, 255, 38] : [30, 120, 220, 40],
            widthPixels: bufferPixels
          })
        );
      } else if (state.showBuffer && state.features !== 'lines') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'buffer-points',
            coordinateOrigin,
            positions: state.features === 'stations' ? selectedStationsBuffer : busBuffer,
            instanceCount: state.features === 'stations' ? selectedStationCount : busRows.length,
            color: dark ? [90, 200, 255, 38] : [30, 120, 220, 40],
            radiusPixels: Math.max(1.5, distance / getMetersPerPixel())
          })
        );
      }
      if (state.showOutside) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'outside-points',
            coordinateOrigin,
            positions: variant.points.buffer,
            instanceCount: variant.points.count,
            values: variant.mask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [0, 0, 0, 0],
            noDataColor: dark ? [140, 150, 170, 80] : [90, 100, 120, 80],
            radiusPixels: Math.max(0.6, state.pointSize * 0.6)
          })
        );
      }
      // Selected points: drawn through the compact ids with the GPU-written instance count.
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'selected-points',
          coordinateOrigin,
          positions: variant.points.buffer,
          ids: variant.ids,
          drawCommands,
          ...(state.colorBy === 'distance'
            ? {
                values: variant.distances,
                valueFormat: 'float32' as const,
                colormap: state.ramp as SpatialAnalysisColormap,
                valueRange: [0, Math.max(distance, 1)] as const
              }
            : state.colorBy === 'line' && state.features !== 'bus'
              ? {
                  values: variant.pointLine,
                  valueFormat: 'uint32' as const,
                  colormap: 'category' as const,
                  palette: LINE_PALETTE,
                  noDataValue: NO_FEATURE,
                  noDataColor: [150, 150, 150, 255] as const
                }
              : {color: [255, 120, 60, 255] as const}),
          radiusPixels: state.pointSize,
          opacity: state.opacity
        })
      );
      // The features themselves: official line colors under white-ringed stations.
      if (state.features === 'lines') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'rail-all',
            coordinateOrigin,
            segments: lineSegmentsBuffer,
            instanceCount: lineSegmentCount,
            color: dark ? [200, 205, 215, 70] : [60, 70, 90, 80],
            widthPixels: 1.5
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'rail-selected',
            coordinateOrigin,
            segments: selectedSegmentsBuffer,
            instanceCount: selectedSegmentCount,
            values: selectedSegmentLineBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: LINE_PALETTE,
            widthPixels: 3
          })
        );
      } else if (state.features === 'stations') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'station-halo',
            coordinateOrigin,
            positions: selectedStationsBuffer,
            instanceCount: selectedStationCount,
            color: [255, 255, 255, 255],
            radiusPixels: 6.5
          }),
          new SpatialAnalysisPointLayer({
            id: 'stations',
            coordinateOrigin,
            positions: selectedStationsBuffer,
            instanceCount: selectedStationCount,
            values: selectedStationLineBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: LINE_PALETTE,
            radiusPixels: 4.5
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bus-stops',
            coordinateOrigin,
            positions: busBuffer,
            instanceCount: busRows.length,
            color: dark ? [255, 255, 255, 230] : [20, 30, 50, 230],
            radiusPixels: 1.6
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      variant.reader.stop();
      variant.resources.destroy();
      void selectedFeatureCount;
      resources.destroy();
    }
  };
}
