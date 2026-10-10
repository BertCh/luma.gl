// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights,
  GPUSpatialJoinCandidates,
  type GPUNearestFeatureGeometry
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
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
import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createLineSet,
  createPolygonSet,
  formatCompact,
  formatInteger,
  getDistanceToFeature,
  type LineSet
} from './b2-geometry';
import {
  createZoneRaster,
  importLines,
  importPolygons,
  uploadLines,
  uploadPolygons,
  type LineBuffers,
  type ZoneRaster
} from './b2-graph';
import {LinkLayer, ZoneFillLayer} from './b2-layers';

/** Option state of the nearest-facility scene. */
export type NearestFacilityOptions = {
  mode: 'nearest' | 'neighbors' | 'weights' | 'candidates';
  featureSet: 'hospital' | 'library' | 'school' | 'fire' | 'rail' | 'bus' | 'rail-lines';
  placeCategory: string;
  radius: number;
  maxDistance: number;
  kShown: number;
  ties: 'lowest-id' | 'all';
  spatialSort: boolean;
  weightK: number;
  weightType: 'binary' | 'inverse-distance';
  weightPower: number;
  candidateDistance: string;
  showLinks: boolean;
  linkStride: number;
  distanceRange: number;
  ramp: RampName;
  opacity: number;
  pointSize: number;
};

type FeatureSet = {
  id: NearestFacilityOptions['featureSet'];
  label: string;
  kind: 'points' | 'lines';
  /** Planar positions `x, y` (points) or line vertices. */
  positions: Float32Array;
  count: number;
  names: (row: number) => string;
  /** Largest search radius the candidate capacity supports, meters. */
  maximumRadius: number;
  /** Bounding-box candidates reserved per place. */
  candidatesPerPoint: number;
  lines?: LineSet;
};

const NEIGHBOR_CAPACITY = 8;
const NEIGHBOR_TIE_CAPACITY = 16;
const UNMATCHED_VALUE = 3.4e38;
const HIDDEN_VALUE = -1;
const SETTLE_FRAMES = 4;

type Variant = {
  key: string;
  mode: NearestFacilityOptions['mode'];
  set: FeatureSet;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  limit: ReturnType<SpatialAnalysisResources['createParameterBuffer']> | null;
  featureCounts: Buffer | null;
  nearestIds: Buffer | null;
  neighborIds: Buffer | null;
  footPoints: Buffer | null;
  neighborCapacity: number;
  weights: {offsets: Buffer; neighbors: Buffer; weights: Buffer; k: number} | null;
  weightsReader: SummaryReader | null;
  candidates: {pairs: Buffer; capacity: number; reader: SummaryReader} | null;
};

/**
 * Nearest-feature joins of 105,808 Chicago places to facilities and CTA stops. One variant (a
 * compiled graph plus its own buffers) exists per compile-time choice: the join mode, the feature
 * set, `spatialSort`, `ties`, the weights rule and the candidate distance. The search radius, the
 * distance limit, the shown neighbour count and every display choice are parameter writes.
 */
export async function createNearestFacility(
  ctx: SceneContext<NearestFacilityOptions>
): Promise<SceneInstance<NearestFacilityOptions>> {
  const {device} = ctx;
  const placesData = ctx.datasets.get('chicago-places');
  const facilitiesData = ctx.datasets.get('chicago-facilities');
  const transit = ctx.datasets.get('cta-transit');
  const tractsData = ctx.datasets.get('chicago-tracts');
  const origin = placesData.defaultOrigin;
  const projection = placesData.getProjection(origin);

  const placePositions = placesData.projectColumn('position', origin);
  const placeCount = placePositions.length / 2;
  const placeCategories = placesData.column<Uint8Array>('category');
  const categoryNames = placesData.categories('category');

  // --- feature sets ----------------------------------------------------------------------------
  const facilityPositions = facilitiesData.projectColumn('position', origin);
  const facilityKinds = facilitiesData.column<Uint8Array>('category');
  const facilityKindNames = facilitiesData.categories('category');
  let facilityNames: string[] = [];
  try {
    const csv = parseCsv(
      await fetchText(getDataFileUrl('chicago-facilities', 'names.csv'), ctx.signal)
    );
    facilityNames = csv.rows.map(row => row[1] ?? '');
  } catch {
    facilityNames = [];
  }
  const stopPositions = transit.projectColumn('stopPosition', origin);
  const stopMode = transit.column<Uint8Array>('stopMode');
  const stopRouteCount = transit.column<Uint8Array>('stopRouteCount');
  const stopTrips = transit.column<Uint16Array>('stopWeekdayTrips');

  const pickRows = (rows: number[], source: Float32Array): Float32Array => {
    const out = new Float32Array(rows.length * 2);
    rows.forEach((row, index) => out.set(source.subarray(row * 2, row * 2 + 2), index * 2));
    return out;
  };
  const facilitySet = (
    id: FeatureSet['id'],
    label: string,
    kindName: string,
    maximumRadius: number,
    candidatesPerPoint: number
  ): FeatureSet => {
    const rows: number[] = [];
    const kindIndex = facilityKindNames.indexOf(kindName);
    facilityKinds.forEach((kind, row) => kind === kindIndex && rows.push(row));
    return {
      id,
      label,
      kind: 'points',
      positions: pickRows(rows, facilityPositions),
      count: rows.length,
      names: index => facilityNames[rows[index]] || label,
      maximumRadius,
      candidatesPerPoint
    };
  };
  const stopSet = (
    id: FeatureSet['id'],
    label: string,
    mode: number,
    maximumRadius: number,
    candidatesPerPoint: number
  ): FeatureSet => {
    const rows: number[] = [];
    stopMode.forEach((value, row) => value === mode && rows.push(row));
    return {
      id,
      label,
      kind: 'points',
      positions: pickRows(rows, stopPositions),
      count: rows.length,
      names: index =>
        `${label} (${stopRouteCount[rows[index]]} routes, ${formatInteger(stopTrips[rows[index]])} weekday departures)`,
      maximumRadius,
      candidatesPerPoint
    };
  };

  // Rail lines from the route shapes of the L routes (one direction where both are published).
  const shapeVertices = transit.projectColumn('shapeVertices', origin);
  const shapeOffsets = transit.column<Uint32Array>('shapePathOffsets');
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeDirection = transit.column<Uint8Array>('shapeDirection');
  const routes = (transit.properties.routes as {type: string; name: string}[]) ?? [];
  const railRoutes = new Set(
    routes.flatMap((route, index) => (route.type === 'rail' ? [index] : []))
  );
  const routeHasForward = new Set<number>();
  for (let shape = 0; shape < shapeRoute.length; shape++) {
    if (railRoutes.has(shapeRoute[shape]) && shapeDirection[shape] === 0)
      routeHasForward.add(shapeRoute[shape]);
  }
  const railPolylines: Float32Array[] = [];
  const railPolylineRoute: number[] = [];
  for (let shape = 0; shape < shapeRoute.length; shape++) {
    const route = shapeRoute[shape];
    if (!railRoutes.has(route)) continue;
    if (routeHasForward.has(route) && shapeDirection[shape] !== 0) continue;
    railPolylines.push(shapeVertices.slice(shapeOffsets[shape] * 2, shapeOffsets[shape + 1] * 2));
    railPolylineRoute.push(route);
  }
  const railLines = createLineSet(railPolylines);

  const featureSets: Record<FeatureSet['id'], FeatureSet> = {
    hospital: facilitySet('hospital', 'Hospital', 'hospital', 8000, 32),
    library: facilitySet('library', 'Library', 'library', 6000, 32),
    school: facilitySet('school', 'CPS school', 'cps_school', 4000, 64),
    fire: facilitySet('fire', 'Fire station', 'fire_station', 6000, 32),
    rail: stopSet('rail', 'L station', 1, 6000, 32),
    bus: stopSet('bus', 'Bus stop', 0, 1000, 64),
    'rail-lines': {
      id: 'rail-lines',
      label: 'L line',
      kind: 'lines',
      positions: railLines.positions,
      count: railLines.featureCount,
      names: index => routes[railPolylineRoute[index]]?.name ?? 'L line',
      maximumRadius: 4000,
      candidatesPerPoint: 64,
      lines: railLines
    }
  };

  // --- shared resources ------------------------------------------------------------------------
  ctx.setStatus('Rasterizing the tract fill');
  const resources = new SpatialAnalysisResources(device, 'nearest-facility');
  const tractSet = createPolygonSet(tractsData, origin);
  const tractBuffers = uploadPolygons(resources, 'tracts', tractSet);
  const tractRaster: ZoneRaster = await createZoneRaster(
    resources,
    'tract-fill',
    tractBuffers,
    tractSet.bounds,
    900
  );
  ctx.signal.throwIfAborted();

  const placesBuffer = resources.createBuffer('places', placePositions);
  const placeMask = new Uint32Array(placeCount).fill(1);
  const placeMaskBuffer = resources.createBuffer('place-mask', placeMask);
  const displayBuffer = resources.createBuffer('display', placeCount * 4);
  const tractFlags = new Uint32Array(tractSet.featureCount);
  const tractFlagsBuffer = resources.createBuffer('tract-flags', tractFlags);
  const linkSegments = new Float32Array(20000 * 4);
  const linkSegmentsBuffer = resources.createBuffer('candidate-links', linkSegments);
  let linkSegmentCount = 0;

  const featureBufferCache = new Map<
    FeatureSet['id'],
    {positions: Buffer; starts?: Buffer; ends?: Buffer; lines?: LineBuffers}
  >();
  const getFeatureBuffers = (set: FeatureSet) => {
    let cached = featureBufferCache.get(set.id);
    if (cached) return cached;
    if (set.lines) {
      const starts = new Float32Array((set.lines.segments.length / 4) * 2);
      const ends = new Float32Array(starts.length);
      for (let segment = 0; segment < set.lines.segments.length / 4; segment++) {
        starts.set(set.lines.segments.subarray(segment * 4, segment * 4 + 2), segment * 2);
        ends.set(set.lines.segments.subarray(segment * 4 + 2, segment * 4 + 4), segment * 2);
      }
      cached = {
        positions: resources.createBuffer(`${set.id}-positions`, set.positions),
        starts: resources.createBuffer(`${set.id}-starts`, starts),
        ends: resources.createBuffer(`${set.id}-ends`, ends),
        lines: uploadLines(resources, `${set.id}-lines`, set.lines)
      };
    } else {
      cached = {positions: resources.createBuffer(`${set.id}-positions`, set.positions)};
    }
    featureBufferCache.set(set.id, cached);
    return cached;
  };

  // --- state -----------------------------------------------------------------------------------
  let destroyed = false;
  let variant: Variant;
  let dirty = true;
  let settle = 0;
  let measuring = false;
  let displayDistances = new Float32Array(placeCount);
  let catchmentCounts = new Uint32Array(0);
  let maximumCatchment = 1;
  let tractCandidateFlags = false;
  let encodedFrames = 0;

  const getEffectiveSet = (state: NearestFacilityOptions): FeatureSet => {
    // Lines have no point queries: weights and candidates use the stations instead.
    if (
      state.featureSet === 'rail-lines' &&
      (state.mode === 'weights' || state.mode === 'candidates')
    ) {
      return featureSets.rail;
    }
    return featureSets[state.featureSet];
  };

  const getVariantKey = (state: NearestFacilityOptions): string => {
    const set = getEffectiveSet(state);
    switch (state.mode) {
      case 'nearest':
        return `nearest|${set.id}|${state.spatialSort}`;
      case 'neighbors':
        return `neighbors|${set.id}|${state.ties}|${state.spatialSort}`;
      case 'weights':
        return `weights|${set.id}|${state.weightK}|${state.weightType}|${state.weightPower}`;
      default:
        return `candidates|${set.id}|${state.candidateDistance}`;
    }
  };

  /** Builds the compiled graph and buffers of one compile-time configuration. */
  function buildVariant(state: NearestFacilityOptions, forTiming = false): Variant {
    const set = getEffectiveSet(state);
    const key = getVariantKey(state);
    const own = new SpatialAnalysisResources(device, `nf-${key}`);
    const graph = new GPUCommandGraph<void>(device, {id: `nf-${key}`});
    const featureBuffers = getFeatureBuffers(set);
    const places = importGraphBuffer(graph, 'places', placesBuffer, 'float32x2', placeCount);
    const display = importGraphBuffer(graph, 'display', displayBuffer, 'float32', placeCount);
    const maskView = importGraphBuffer(graph, 'mask', placeMaskBuffer, 'uint32', placeCount);
    const overflow = own.createBuffer('overflow', 4);
    const candidateCount = own.createBuffer('candidate-count', 4);
    const result: Variant = {
      key,
      mode: state.mode,
      set,
      resources: own,
      compiled: undefined as never,
      reader: undefined as never,
      limit: null,
      featureCounts: null,
      nearestIds: null,
      neighborIds: null,
      footPoints: null,
      neighborCapacity: 1,
      weights: null,
      weightsReader: null,
      candidates: null
    };
    let readerSources: {buffer: Buffer; size: number}[] = [];

    if (state.mode === 'nearest') {
      const limit = own.createParameterBuffer(
        'radius',
        'float32',
        1,
        Float32Array.of(state.radius)
      );
      const nearestIds = own.createBuffer('nearest-ids', placeCount * 4);
      const distances = own.createBuffer('distances', placeCount * 4);
      const featureTotal = set.kind === 'lines' ? set.lines!.segments.length / 4 : set.count;
      const featureCounts = own.createBuffer('feature-counts', Math.max(featureTotal, 1) * 4);
      const features: GPUNearestFeatureGeometry =
        set.kind === 'lines'
          ? {
              kind: 'segments',
              starts: importGraphBuffer(
                graph,
                'starts',
                featureBuffers.starts!,
                'float32x2',
                set.lines!.segments.length / 4
              ),
              ends: importGraphBuffer(
                graph,
                'ends',
                featureBuffers.ends!,
                'float32x2',
                set.lines!.segments.length / 4
              )
            }
          : {
              kind: 'points',
              positions: importGraphBuffer(
                graph,
                'features',
                featureBuffers.positions,
                'float32x2',
                set.count
              )
            };
      const distancesView = importGraphBuffer(graph, 'distances', distances, 'float32', placeCount);
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'nearest',
          points: places,
          features,
          radius: limit.importToGraph(graph),
          candidateCapacity: placeCount * set.candidatesPerPoint,
          spatialSort: state.spatialSort,
          nearestFeatureIds: importGraphBuffer(
            graph,
            'nearest-ids',
            nearestIds,
            'uint32',
            placeCount
          ),
          nearestDistances: distancesView,
          featureCounts: importGraphBuffer(
            graph,
            'feature-counts',
            featureCounts,
            'uint32',
            Math.max(featureTotal, 1)
          ),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)
        })
      );
      addKernelPass(graph, {
        id: 'nearest-display',
        invocationCount: placeCount,
        bindings: [
          {name: 'distances', view: distancesView, type: 'f32', access: 'read'},
          {name: 'mask', view: maskView, type: 'u32', access: 'read'},
          {name: 'display', view: display, type: 'f32', access: 'read_write'}
        ],
        body: `
  if (mask[maskOffset + index] == 0u) { display[displayOffset + index] = ${HIDDEN_VALUE}.0; return; }
  let distance = distances[distancesOffset + index];
  display[displayOffset + index] = select(${UNMATCHED_VALUE}, distance, distance >= 0.0);`
      });
      result.limit = limit;
      result.featureCounts = featureCounts;
      result.nearestIds = nearestIds;
      readerSources = [
        {buffer: displayBuffer, size: placeCount * 4},
        {buffer: featureCounts, size: Math.max(featureTotal, 1) * 4},
        {buffer: overflow, size: 4},
        {buffer: candidateCount, size: 4}
      ];
    } else if (state.mode === 'neighbors') {
      const capacity = state.ties === 'all' ? NEIGHBOR_TIE_CAPACITY : NEIGHBOR_CAPACITY;
      const limit = own.createParameterBuffer(
        'max-distance',
        'float32',
        1,
        Float32Array.of(state.maxDistance)
      );
      const neighborIds = own.createBuffer('neighbor-ids', placeCount * capacity * 4);
      const neighborCounts = own.createBuffer('neighbor-counts', placeCount * 4);
      const neighborDistances = own.createBuffer('neighbor-distances', placeCount * capacity * 4);
      const footPoints = own.createBuffer('foot-points', placeCount * capacity * 8);
      const features: GPUNearestFeatureGeometry =
        set.kind === 'lines'
          ? importLines(graph, 'rail', featureBuffers.lines!)
          : {
              kind: 'points',
              positions: importGraphBuffer(
                graph,
                'features',
                featureBuffers.positions,
                'float32x2',
                set.count
              )
            };
      const neighborIdsView = importGraphBuffer(
        graph,
        'neighbor-ids',
        neighborIds,
        'uint32',
        placeCount * capacity
      );
      const neighborCountsView = importGraphBuffer(
        graph,
        'neighbor-counts',
        neighborCounts,
        'uint32',
        placeCount
      );
      const neighborDistancesView = importGraphBuffer(
        graph,
        'neighbor-distances',
        neighborDistances,
        'float32',
        placeCount * capacity
      );
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'neighbors',
          points: places,
          features,
          k: NEIGHBOR_CAPACITY,
          neighborCapacity: capacity,
          ties: state.ties,
          maxDistance: limit.importToGraph(graph),
          spatialSort: state.spatialSort,
          neighborIds: neighborIdsView,
          neighborCounts: neighborCountsView,
          neighborDistances: neighborDistancesView,
          neighborFootPoints: importGraphBuffer(
            graph,
            'foot-points',
            footPoints,
            'float32x2',
            placeCount * capacity
          ),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        })
      );
      addKernelPass(graph, {
        id: 'neighbors-display',
        invocationCount: placeCount,
        declarations: `const CAPACITY: u32 = ${capacity}u;`,
        bindings: [
          {name: 'counts', view: neighborCountsView, type: 'u32', access: 'read'},
          {name: 'distances', view: neighborDistancesView, type: 'f32', access: 'read'},
          {name: 'mask', view: maskView, type: 'u32', access: 'read'},
          {name: 'display', view: display, type: 'f32', access: 'read_write'}
        ],
        body: `
  if (mask[maskOffset + index] == 0u) { display[displayOffset + index] = ${HIDDEN_VALUE}.0; return; }
  if (counts[countsOffset + index] == 0u) { display[displayOffset + index] = ${UNMATCHED_VALUE}; return; }
  display[displayOffset + index] = distances[distancesOffset + index * CAPACITY];`
      });
      result.limit = limit;
      result.neighborIds = neighborIds;
      result.footPoints = footPoints;
      result.neighborCapacity = capacity;
      readerSources = [
        {buffer: displayBuffer, size: placeCount * 4},
        {buffer: overflow, size: 4}
      ];
    } else if (state.mode === 'weights') {
      const k = state.weightK;
      const capacity = k + 1;
      const rows = set.count;
      const featuresView = importGraphBuffer(
        graph,
        'features',
        featureBuffers.positions,
        'float32x2',
        rows
      );
      const neighborIds = own.createBuffer('neighbor-ids', rows * capacity * 4);
      const neighborCounts = own.createBuffer('neighbor-counts', rows * 4);
      const neighborDistances = own.createBuffer('neighbor-distances', rows * capacity * 4);
      const csrOffsets = own.createBuffer('csr-offsets', (rows + 1) * 4);
      const csrNeighbors = own.createBuffer('csr-neighbors', rows * k * 4);
      const csrWeights = own.createBuffer('csr-weights', rows * k * 4);
      const csrDistances = own.createBuffer('csr-distances', rows * k * 4);
      const weightsOverflow = own.createBuffer('weights-overflow', 4);
      const neighborIdsView = importGraphBuffer(
        graph,
        'neighbor-ids',
        neighborIds,
        'uint32',
        rows * capacity
      );
      const neighborCountsView = importGraphBuffer(
        graph,
        'neighbor-counts',
        neighborCounts,
        'uint32',
        rows
      );
      const neighborDistancesView = importGraphBuffer(
        graph,
        'neighbor-distances',
        neighborDistances,
        'float32',
        rows * capacity
      );
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'weights-join',
          points: featuresView,
          features: {kind: 'points', positions: featuresView},
          k: capacity,
          neighborIds: neighborIdsView,
          neighborCounts: neighborCountsView,
          neighborDistances: neighborDistancesView,
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        })
      );
      graph.add(
        new GPUNearestFeatureWeights({
          id: 'nearest-weights',
          neighborIds: neighborIdsView,
          neighborCounts: neighborCountsView,
          neighborDistances: neighborDistancesView,
          slotCapacity: capacity,
          k,
          excludeSelf: true,
          weightType: state.weightType,
          power: state.weightPower,
          weights: {
            offsets: importGraphBuffer(graph, 'csr-offsets', csrOffsets, 'uint32', rows + 1),
            neighbors: importGraphBuffer(graph, 'csr-neighbors', csrNeighbors, 'uint32', rows * k),
            weights: importGraphBuffer(graph, 'csr-weights', csrWeights, 'float32', rows * k),
            distances: importGraphBuffer(graph, 'csr-distances', csrDistances, 'float32', rows * k)
          },
          overflow: importGraphBuffer(graph, 'weights-overflow', weightsOverflow, 'uint32', 1)
        })
      );
      addKernelPass(graph, {
        id: 'weights-display',
        invocationCount: placeCount,
        bindings: [
          {name: 'mask', view: maskView, type: 'u32', access: 'read'},
          {name: 'display', view: display, type: 'f32', access: 'read_write'}
        ],
        body: `display[displayOffset + index] = select(${HIDDEN_VALUE}.0, ${UNMATCHED_VALUE}, mask[maskOffset + index] != 0u);`
      });
      result.neighborIds = neighborIds;
      result.neighborCapacity = capacity;
      result.weights = {offsets: csrOffsets, neighbors: csrNeighbors, weights: csrWeights, k};
      readerSources = [{buffer: overflow, size: 4}];
      result.weightsReader = new SummaryReader(
        own,
        'weights',
        [
          {buffer: csrOffsets, size: (rows + 1) * 4},
          {buffer: csrNeighbors, size: rows * k * 4},
          {buffer: csrWeights, size: rows * k * 4},
          {buffer: weightsOverflow, size: 4}
        ],
        bytes => {
          if (destroyed || variant !== result) return;
          const offsets = new Uint32Array(bytes, 0, rows + 1);
          const neighbors = new Uint32Array(bytes, (rows + 1) * 4, rows * k);
          const weights = new Float32Array(bytes, (rows + 1) * 4 + rows * k * 4, rows * k);
          const nonZero = offsets[rows];
          let minimum = Infinity;
          let maximum = 0;
          let reciprocal = 0;
          for (let row = 0; row < rows; row++) {
            for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
              minimum = Math.min(minimum, weights[slot]);
              maximum = Math.max(maximum, weights[slot]);
              const other = neighbors[slot];
              for (let back = offsets[other]; back < offsets[other + 1]; back++) {
                if (neighbors[back] === row) {
                  reciprocal++;
                  break;
                }
              }
            }
          }
          ctx.setReadout(
            'weightsRows',
            `${formatInteger(rows)} rows, ${formatInteger(nonZero)} links (${(nonZero / rows).toFixed(1)} per row)`
          );
          ctx.setReadout(
            'weightsRange',
            state.weightType === 'binary'
              ? 'all weights are 1 (binary KNN)'
              : `${formatCompact(minimum)} to ${formatCompact(maximum)} per metre^${state.weightPower}`
          );
          ctx.setReadout(
            'weightsSymmetry',
            `${nonZero ? ((100 * reciprocal) / nonZero).toFixed(0) : 0}% of links are reciprocal (KNN is directed)`
          );
        }
      );
    } else {
      // Candidates: facility bounding boxes expanded by a compile-time distance against tract polygons.
      const distance = Number(state.candidateDistance);
      const left = importGraphBuffer(
        graph,
        'features',
        featureBuffers.positions,
        'float32x2',
        set.count
      );
      const capacity = Math.max(2048, set.count * 64);
      const leftIds = own.createBuffer('pairs-left', capacity * 4);
      const rightIds = own.createBuffer('pairs-right', capacity * 4);
      const count = own.createBuffer('pairs-count', 4);
      const total = own.createBuffer('pairs-total', 4);
      const polygons = importPolygons(graph, 'tracts', tractBuffers);
      graph.add(
        new GPUSpatialJoinCandidates({
          id: 'candidates',
          left: {kind: 'points', positions: left},
          right: polygons,
          distance,
          pairs: {
            leftIds: importGraphBuffer(graph, 'pairs-left', leftIds, 'uint32', capacity),
            rightIds: importGraphBuffer(graph, 'pairs-right', rightIds, 'uint32', capacity),
            count: importGraphBuffer(graph, 'pairs-count', count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
            requiredCount: importGraphBuffer(graph, 'pairs-total', total, 'uint32', 1)
          }
        })
      );
      addKernelPass(graph, {
        id: 'candidates-display',
        invocationCount: placeCount,
        bindings: [{name: 'display', view: display, type: 'f32', access: 'read_write'}],
        body: `display[displayOffset + index] = ${HIDDEN_VALUE}.0;`
      });
      const pairsReader = new SummaryReader(
        own,
        'pairs',
        [
          {buffer: count, size: 4},
          {buffer: total, size: 4},
          {buffer: overflow, size: 4},
          {buffer: leftIds, size: capacity * 4},
          {buffer: rightIds, size: capacity * 4}
        ],
        bytes => {
          if (destroyed || variant !== result) return;
          const header = new Uint32Array(bytes, 0, 3);
          const pairCount = header[0];
          const lefts = new Uint32Array(bytes, 12, capacity);
          const rights = new Uint32Array(bytes, 12 + capacity * 4, capacity);
          tractCandidateFlags = true;
          tractFlags.fill(0);
          let exact = 0;
          linkSegmentCount = 0;
          for (let pair = 0; pair < pairCount; pair++) {
            const tract = rights[pair];
            tractFlags[tract] = 1;
            const x = set.positions[lefts[pair] * 2];
            const y = set.positions[lefts[pair] * 2 + 1];
            if (getDistanceToFeature(tractSet, tract, x, y) <= distance) exact++;
            if (linkSegmentCount < 20000) {
              linkSegments.set(
                [x, y, tractSet.centroids[tract * 2], tractSet.centroids[tract * 2 + 1]],
                linkSegmentCount * 4
              );
              linkSegmentCount++;
            }
          }
          tractFlagsBuffer.write(tractFlags);
          linkSegmentsBuffer.write(linkSegments);
          const flagged = tractFlags.reduce((sum, flag) => sum + flag, 0);
          ctx.setReadout(
            'candidatePairs',
            `${formatInteger(header[1])} pairs (${header[2] ? 'overflowed' : 'complete'})`
          );
          ctx.setReadout(
            'candidateExact',
            `${formatInteger(exact)} of ${formatInteger(pairCount)} pairs are truly within ${distance} m (${pairCount ? ((100 * exact) / pairCount).toFixed(0) : 0}%)`
          );
          ctx.setReadout(
            'candidateTracts',
            `${formatInteger(flagged)} of ${tractSet.featureCount} tracts touch a box`
          );
          ctx.requestLayers();
        }
      );
      result.candidates = {pairs: leftIds, capacity, reader: pairsReader};
      readerSources = [
        {buffer: overflow, size: 4},
        {buffer: candidateCount, size: 4}
      ];
    }

    result.reader = new SummaryReader(own, `summary-${key}`, readerSources, bytes => {
      if (destroyed || variant !== result) return;
      handleSummary(result, bytes);
    });
    result.compiled = own.track(graph.compile());
    if (forTiming) return result;
    void candidateCount;
    return result;
  }

  /** Statistics of the places over the display distances, plus the per-mode readouts. */
  function handleSummary(source: Variant, bytes: ArrayBuffer): void {
    const state = ctx.options;
    if (source.mode === 'candidates' || source.mode === 'weights') {
      ctx.setReadout('shown', 'n/a in this mode');
      ctx.setReadout('matched', 'n/a in this mode');
      ctx.setReadout('median', 'n/a in this mode');
      ctx.setReadout('walkable', 'n/a in this mode');
      ctx.setReadout('overflow', new Uint32Array(bytes, 0, 1)[0] ? 'YES (capacity)' : 'no');
      return;
    }
    const distances = new Float32Array(bytes, 0, placeCount);
    displayDistances = Float32Array.from(distances);
    let shown = 0;
    const matchedDistances: number[] = [];
    for (let index = 0; index < placeCount; index++) {
      const value = distances[index];
      if (value === HIDDEN_VALUE) continue;
      shown++;
      if (value < 3e38) matchedDistances.push(value);
    }
    matchedDistances.sort((a, b) => a - b);
    const matched = matchedDistances.length;
    const percentile = (fraction: number) =>
      matchedDistances[Math.min(matched - 1, Math.floor(fraction * matched))];
    const within = (limit: number) => {
      let low = 0;
      let high = matched;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (matchedDistances[middle] <= limit) low = middle + 1;
        else high = middle;
      }
      return low;
    };
    ctx.setReadout('shown', `${formatInteger(shown)} of ${formatInteger(placeCount)} places`);
    ctx.setReadout(
      'matched',
      shown
        ? `${formatInteger(matched)} (${((100 * matched) / shown).toFixed(1)}%) have a ${source.set.label.toLowerCase()} within ${formatInteger(source.mode === 'nearest' ? state.radius : state.maxDistance)} m`
        : 'none'
    );
    ctx.setReadout(
      'median',
      matched
        ? `${formatInteger(percentile(0.5))} m median, ${formatInteger(percentile(0.9))} m at the 90th percentile`
        : 'n/a'
    );
    ctx.setReadout(
      'walkable',
      shown
        ? `${((100 * within(400)) / shown).toFixed(1)}% within 400 m, ${((100 * within(800)) / shown).toFixed(1)}% within 800 m (of all shown places)`
        : 'n/a'
    );
    const tailOffset = placeCount * 4;
    if (source.mode === 'nearest') {
      const counts = new Uint32Array(
        bytes.slice(
          tailOffset,
          tailOffset +
            Math.max(
              source.set.kind === 'lines'
                ? source.set.lines!.segments.length / 4
                : source.set.count,
              1
            ) *
              4
        )
      );
      catchmentCounts = counts;
      let maximum = 0;
      let top = -1;
      for (let row = 0; row < counts.length; row++) {
        if (counts[row] > maximum) {
          maximum = counts[row];
          top = row;
        }
      }
      maximumCatchment = Math.max(maximum, 1);
      ctx.setLegendExtent('catchment', [0, maximumCatchment]);
      const tail = new Uint32Array(bytes.slice(tailOffset + counts.length * 4));
      ctx.setReadout('overflow', tail[0] ? 'YES (capacity)' : 'no');
      ctx.setReadout('candidates', formatInteger(tail[1]));
      ctx.setReadout(
        'catchment',
        top >= 0 && source.set.kind === 'points'
          ? `${source.set.names(top)}: nearest to ${formatInteger(maximum)} places`
          : maximum
            ? `${formatInteger(maximum)} places on the largest segment`
            : 'none'
      );
    } else {
      const tail = new Uint32Array(bytes.slice(tailOffset));
      ctx.setReadout('overflow', tail[0] ? 'YES (tie capacity)' : 'no');
      ctx.setReadout('candidates', 'n/a (branch-and-bound traversal)');
      ctx.setReadout('catchment', 'n/a in this mode');
    }
  }

  function writeMask(state: NearestFacilityOptions): void {
    if (state.placeCategory === 'all') {
      placeMask.fill(1);
    } else {
      const index = categoryNames.indexOf(state.placeCategory);
      for (let place = 0; place < placeCount; place++)
        placeMask[place] = placeCategories[place] === index ? 1 : 0;
    }
    placeMaskBuffer.write(placeMask);
  }

  function replaceVariant(state: NearestFacilityOptions): void {
    const previous = variant;
    variant = buildVariant(state);
    dirty = true;
    tractCandidateFlags = state.mode === 'candidates' ? tractCandidateFlags : false;
    if (previous) {
      previous.reader.stop();
      previous.weightsReader?.stop();
      previous.candidates?.reader.stop();
      // Deck may still hold the previous frame; free two frames later.
      requestAnimationFrame(() => requestAnimationFrame(() => previous.resources.destroy()));
    }
    updateStaticReadouts(state);
  }

  function updateStaticReadouts(state: NearestFacilityOptions): void {
    const set = variant.set;
    ctx.setReadout('places', placeCount);
    ctx.setReadout(
      'features',
      `${formatInteger(set.count)} ${set.kind === 'lines' ? 'lines' : 'points'}: ${set.label}`
    );
    if (state.mode === 'nearest') {
      const capped = Math.min(state.radius, set.maximumRadius);
      ctx.setReadout(
        'radiusUsed',
        capped < state.radius
          ? `${formatInteger(capped)} m (capped for this feature set's candidate capacity)`
          : `${formatInteger(capped)} m`
      );
    } else {
      ctx.setReadout(
        'radiusUsed',
        state.mode === 'neighbors'
          ? `${formatInteger(state.maxDistance)} m search limit`
          : 'n/a in this mode'
      );
    }
    if (state.mode !== 'weights') {
      ctx.setReadout('weightsRows', 'n/a in this mode');
      ctx.setReadout('weightsRange', 'n/a in this mode');
      ctx.setReadout('weightsSymmetry', 'n/a in this mode');
    }
    if (state.mode !== 'candidates') {
      ctx.setReadout('candidatePairs', 'n/a in this mode');
      ctx.setReadout('candidateExact', 'n/a in this mode');
      ctx.setReadout('candidateTracts', 'n/a in this mode');
    }
  }

  writeMask(ctx.options);
  variant = buildVariant(ctx.options);
  updateStaticReadouts(ctx.options);

  /** Times the active variant and the other `spatialSort` choice outside the frame. */
  async function measureSort(): Promise<void> {
    if (measuring || destroyed) return;
    const state = ctx.options;
    if (state.mode !== 'nearest' && state.mode !== 'neighbors') {
      ctx.setReadout('timeSort', 'switch to nearest or k nearest mode');
      return;
    }
    measuring = true;
    ctx.setReadout('timeSort', 'measuring...');
    const results: string[] = [];
    try {
      for (const spatialSort of [false, true]) {
        const built = buildVariant({...state, spatialSort}, true);
        try {
          const timing = await measureCompiledGraph(device, built.compiled, {
            parameters: undefined,
            completionBuffer: displayBuffer,
            signal: ctx.signal
          });
          results.push(
            `${spatialSort ? 'sorted' : 'unsorted'} ${formatCompiledGraphTiming(timing).split(' · ')[0]}`
          );
        } finally {
          built.compiled.destroy();
          built.resources.destroy();
        }
      }
      if (!destroyed) ctx.setReadout('timeSort', results.join(' | '));
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  function getMetersPerPixel(latitude: number): number {
    const viewport = ctx.getViewport();
    const zoom = viewport?.zoom ?? 10;
    return (40075016.686 * Math.cos((latitude * Math.PI) / 180)) / (512 * 2 ** zoom);
  }

  return {
    getCompiledGraphs: () => [variant.compiled as CompiledGPUCommandGraph<never>],

    setOption(id, _value, state) {
      switch (id) {
        case 'mode':
        case 'featureSet':
        case 'spatialSort':
        case 'ties':
        case 'weightK':
        case 'weightType':
        case 'weightPower':
        case 'candidateDistance':
          if (getVariantKey(state) !== variant.key) replaceVariant(state);
          else updateStaticReadouts(state);
          break;
        case 'placeCategory':
          writeMask(state);
          dirty = true;
          break;
        case 'radius':
        case 'maxDistance':
          dirty = true;
          updateStaticReadouts(state);
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
      encodedFrames++;
      if (variant.limit) {
        const value =
          state.mode === 'nearest'
            ? Math.min(state.radius, variant.set.maximumRadius)
            : state.maxDistance;
        variant.limit.write(Float32Array.of(value));
      }
      if (dirty) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        settle = SETTLE_FRAMES;
      } else if (settle > 0) {
        settle--;
        if (settle === 0) {
          variant.reader.request(commandEncoder);
          variant.weightsReader?.request(commandEncoder);
          variant.candidates?.reader.request(commandEncoder);
        }
      }
      variant.reader.flush(commandEncoder);
      variant.weightsReader?.flush(commandEncoder);
      variant.candidates?.reader.flush(commandEncoder);
      void encodedFrames;
    },

    getLayers() {
      const state = ctx.options;
      const set = variant.set;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (state.mode === 'candidates') {
        layers.push(
          new ZoneFillLayer({
            id: 'candidate-tracts',
            coordinateOrigin,
            gridSize: [tractRaster.width, tractRaster.height],
            bounds: tractRaster.bounds,
            rowOrigin: 'south',
            valueIndices: tractRaster.zones,
            values: tractFlagsBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 170, 60, 110],
            noDataColor: [0, 0, 0, 0],
            opacity: tractCandidateFlags ? 1 : 0
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'tract-outline',
            coordinateOrigin,
            segments: tractBuffers.outline,
            instanceCount: tractSet.outline.length / 4,
            color: dark ? [230, 235, 245, 70] : [30, 40, 60, 80],
            widthPixels: 0.7
          })
        );
      }
      if (state.mode === 'nearest' || state.mode === 'neighbors') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'places',
            coordinateOrigin,
            positions: placesBuffer,
            instanceCount: placeCount,
            values: displayBuffer,
            valueFormat: 'float32',
            colormap: state.ramp as SpatialAnalysisColormap,
            valueRange: [0, state.distanceRange],
            discardAtOrBelow: HIDDEN_VALUE,
            color: dark ? [150, 160, 180, 90] : [90, 100, 120, 90],
            noDataColor: dark ? [255, 120, 100, 170] : [210, 60, 50, 170],
            radiusPixels: state.pointSize,
            opacity: state.opacity
          })
        );
      }
      if (
        state.showLinks &&
        state.mode === 'nearest' &&
        variant.nearestIds &&
        set.kind === 'points'
      ) {
        layers.push(
          new LinkLayer({
            id: 'snap-links',
            coordinateOrigin,
            queries: placesBuffer,
            targets: getFeatureBuffers(set).positions,
            targetIds: variant.nearestIds,
            queryCount: placeCount,
            queryStride: state.linkStride,
            color: dark ? [255, 255, 255, 150] : [20, 30, 50, 150],
            widthPixels: 1
          })
        );
      }
      if (state.showLinks && state.mode === 'neighbors' && variant.footPoints) {
        layers.push(
          new LinkLayer({
            id: 'neighbor-links',
            coordinateOrigin,
            queries: placesBuffer,
            targets: variant.footPoints,
            queryCount: placeCount,
            slotCapacity: variant.neighborCapacity,
            slotCount: state.kShown,
            queryStride: state.linkStride,
            color: dark ? [255, 255, 255, 190] : [20, 30, 50, 190],
            widthPixels: 1.1,
            fade: 0.55
          })
        );
      }
      if (state.mode === 'weights' && variant.neighborIds) {
        const buffers = getFeatureBuffers(set);
        layers.push(
          new LinkLayer({
            id: 'weight-links',
            coordinateOrigin,
            queries: buffers.positions,
            targets: buffers.positions,
            targetIds: variant.neighborIds,
            queryCount: set.count,
            slotCapacity: variant.neighborCapacity,
            slotCount: state.weightK,
            firstSlot: 1,
            color: dark ? [255, 200, 90, 210] : [200, 110, 0, 210],
            widthPixels: 1.6,
            fade: 0.6
          })
        );
      }
      if (state.mode === 'candidates') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'candidate-links',
            coordinateOrigin,
            segments: linkSegmentsBuffer,
            instanceCount: linkSegmentCount,
            color: dark ? [255, 255, 255, 120] : [20, 30, 50, 120],
            widthPixels: 0.8
          })
        );
      }
      // Features on top: lines, or points sized and colored by the places that choose them.
      if (set.kind === 'lines') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'feature-lines',
            coordinateOrigin,
            segments: getFeatureBuffers(set).lines!.segments,
            instanceCount: set.lines!.segments.length / 4,
            color: [255, 160, 40, 255],
            widthPixels: 2.6
          })
        );
      } else {
        const buffers = getFeatureBuffers(set);
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'feature-halo',
            coordinateOrigin,
            positions: buffers.positions,
            instanceCount: set.count,
            color: dark ? [255, 255, 255, 255] : [255, 255, 255, 255],
            radiusPixels: set.id === 'bus' ? 3.2 : 8
          }),
          new SpatialAnalysisPointLayer({
            id: 'features',
            coordinateOrigin,
            positions: buffers.positions,
            instanceCount: set.count,
            ...(state.mode === 'nearest' && variant.featureCounts
              ? {
                  values: variant.featureCounts,
                  valueFormat: 'uint32' as const,
                  colormap: 'cividis' as const,
                  valueRange: [0, maximumCatchment] as const,
                  sqrtScale: true
                }
              : {color: [30, 30, 40, 255] as const}),
            radiusPixels: set.id === 'bus' ? 2 : 5.5
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const set = variant.set;
      if (set.kind !== 'points') return null;
      const reach = 12 * getMetersPerPixel(event.coordinate[1]);
      let best = -1;
      let bestDistance = reach;
      for (let row = 0; row < set.count; row++) {
        const distance = Math.hypot(set.positions[row * 2] - x, set.positions[row * 2 + 1] - y);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = row;
        }
      }
      if (best < 0) return null;
      const lines = [set.names(best)];
      if (variant.mode === 'nearest' && catchmentCounts.length > best) {
        lines.push(
          `Nearest facility for ${formatInteger(catchmentCounts[best])} places within the radius`
        );
      }
      return lines.join('\n');
    },

    destroy() {
      destroyed = true;
      variant.reader.stop();
      variant.weightsReader?.stop();
      variant.candidates?.reader.stop();
      variant.resources.destroy();
      void displayDistances;
      resources.destroy();
    }
  };
}
