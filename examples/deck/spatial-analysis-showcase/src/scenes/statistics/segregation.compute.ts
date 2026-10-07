// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSegregationLayout,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPUNeighborSearch,
  GPUSegregation,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatOrdinal, formatPercent, liveText} from '../../cartography/live-text';
import {createFeatureLocator, getGeometryPolygons} from '../../cartography/picking';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {ChartColor, ChartData, LineChartData} from '../chart-types';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {createByteReader} from './b5-common';
import {getPolygonLayout} from './b5-geometry';
import {
  countClasses,
  countDominanceCells,
  getAspatialIndices,
  getClassIndex,
  getDominanceClasses,
  getLargestGroup,
  getPercentile,
  getRadiusMembers,
  getWeightedMean,
  shuffleRows,
  sumRowsByUnit
} from './segregation.stats';
import {
  DIVERSITY_BREAKS,
  DOMINANCE_TIER_LABELS,
  formatIndex,
  getAreaLineStyle,
  getCasingColor,
  getDiversityTable,
  getDominanceColors,
  getDominanceTier,
  getGroupColor,
  getInkColor,
  getRelativeTable,
  GROUP_COUNT,
  RELATIVE_BREAKS,
  RELATIVE_CLIP,
  SEGREGATION_GROUPS
} from './segregation.style';

/** Option state of the segregation scene. */
export type SegregationOptions = {
  localIndex: 'dominant' | 'relative' | 'diversity';
  focalGroup: string;
  units: 'tracts' | 'areas';
  scale: number;
  bandwidth: number;
  shuffle: boolean;
  shuffleSeed: number;
  showRing: boolean;
  focusArea: string;
  weights: 'band' | GPUSpatialWeightsKernel;
  rowStandardize: boolean;
  spatialForm: 'environment' | 'smoothed-population';
  selfWeight: string;
  atkinsonB: string;
};

/** Data shared with `legends(state, data)` through `ctx.setLegendData` (key `segregation`). */
export type SegregationLegendData = {
  groundIsDark: boolean;
  /** Tracts per class of the largest-group map, `registryGroup * 3 + tier`. */
  dominanceCells: number[];
  /** Tracts without residents. */
  emptyTracts: number;
  /** Units per class of the relative map and of the diversity map (current units and scale). */
  relativeCounts: number[] | null;
  diversityCounts: number[] | null;
  /** Citywide share of the focal group and the citywide diversity (entropy over ln 5). */
  focalCityShare: number;
  cityDiversity: number;
  /** Label of the current environment: `the tract` or `everyone within 4 km`. */
  environmentLabel: string;
  focalLabel: string;
};

/** Ladder of radii as multiples of the base radius (the `bandwidth` option, in km). */
const RADIUS_LADDER = [1, 2, 4, 8, 16] as const;
const SCALE_COUNT = RADIUS_LADDER.length + 1;
const LOG_GROUP_COUNT = Math.log(GROUP_COUNT);
/** Mode of the display kernel: the log2 ratio of the focal share, or the normalised entropy. */
const DISPLAY_MODE = {relative: 0, diversity: 1, dominant: 0} as const;
/** Residents a tract needs to be named in a note (a tract of a few dozen people proves nothing). */
const NOTE_MINIMUM_RESIDENTS = 1000;
/** Community areas named in the mixing notes of the diversity step. */
const MIXING_AREAS = ['Rogers Park', 'Uptown', 'Albany Park'] as const;
/** Layer opacity of the tract fill on the light and the dark city ground. */
const FILL_OPACITY = {light: 0.88, dark: 0.9} as const;

type Variant = {key: string; compiled: CompiledGPUCommandGraph<void>};

/** Everything read back from the GPU after the graphs ran. */
type Latest = {
  indices: Float32Array;
  shuffledIndices: Float32Array;
  display: Float32Array;
  environment: Float32Array;
  theil: Float32Array;
  dissimilarity: Float32Array;
  sortedDisplay: Float64Array;
};

/** Bits of a float32 as a uint32, to carry a float through the uint32 parameter buffer. */
function getFloatBits(value: number): number {
  return new Uint32Array(Float32Array.of(value).buffer)[0];
}

/**
 * Residential segregation of Chicago census tracts by race and ethnicity. A `GPUNeighborSearch`
 * distance band per radius writes a weights CSR, an optional `GPUSpatialWeightsTransform` turns it
 * into a kernel or row-standardises it, and two `GPUSegregation` nodes read every scale: one on the
 * real counts and one on the same rows shuffled between tracts (the checkerboard null), each
 * writing the global indices; the real one also writes the per-tract terms. The radius, the
 * shuffle seed, the focal group and the map are parameter or buffer writes; the weights variant,
 * self weight, spatial form and Atkinson parameter are compile-time and compile on demand.
 */
export async function createSegregation(
  ctx: SceneContext<SegregationOptions>
): Promise<SceneInstance<SegregationOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const areas = ctx.datasets.get('chicago-community-areas');
  const tractGeojson = tracts.geojson;
  const areaGeojson = areas.geojson;
  if (!tractGeojson || !areaGeojson) {
    throw new Error('segregation needs the tract and community-area polygons');
  }
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'segregation');
  const layout = getPolygonLayout(tracts);
  const n = layout.featureCount;
  const origin = tracts.defaultOrigin;
  const projection = tracts.getProjection(origin);
  const project = (lng: number, lat: number) => projection.project(lng, lat);
  const layerOrigin: [number, number, number] = [origin[0], origin[1], 0];

  // Geometry: planar-metre meshes of the tracts and the community areas (shared origin).
  const tractMesh = buildPolygonMesh(tractGeojson, project);
  const tractPolygons = createPolygonMeshBuffers(resources, tractMesh, 'tracts');
  const areaMesh = buildPolygonMesh(areaGeojson, project);
  const areaPolygons = createPolygonMeshBuffers(resources, areaMesh, 'areas');
  const tractLocator = createFeatureLocator(tractGeojson);
  const areaLocator = createFeatureLocator(areaGeojson);
  const tractFeatures = tractGeojson.features;
  const areaFeatures = areaGeojson.features;
  const areaCount = areaFeatures.length;

  // One point per tract: the mean of its outline vertices in local metres (not an area centroid
  // and not population-weighted; a few hundred metres on irregular tracts, negligible at 2 km+).
  const points = new Float32Array(n * 2);
  const pointLngLat: LngLat[] = [];
  {
    const {vertices, ringOffsets, featureRingOffsets} = layout;
    for (let row = 0; row < n; row++) {
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      const first = ringOffsets[featureRingOffsets[row]];
      const last = ringOffsets[featureRingOffsets[row + 1]];
      for (let vertex = first; vertex < last; vertex++) {
        const [x, y] = project(vertices[vertex * 2], vertices[vertex * 2 + 1]);
        sumX += x;
        sumY += y;
        count++;
      }
      points[row * 2] = sumX / Math.max(count, 1);
      points[row * 2 + 1] = sumY / Math.max(count, 1);
      pointLngLat.push(projection.unproject(points[row * 2], points[row * 2 + 1]));
    }
  }
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < n; row++) {
    minimumX = Math.min(minimumX, points[row * 2]);
    maximumX = Math.max(maximumX, points[row * 2]);
    minimumY = Math.min(minimumY, points[row * 2 + 1]);
    maximumY = Math.max(maximumY, points[row * 2 + 1]);
  }
  const margin = 2000;
  const searchBounds: [number, number, number, number] = [
    minimumX - margin,
    minimumY - margin,
    maximumX + margin,
    maximumY + margin
  ];

  // Counts and the facts that need no GPU.
  const groupCountsHost = new Float32Array(n * GROUP_COUNT);
  const groupColumns = SEGREGATION_GROUPS.map(group => tracts.column<Float32Array>(group.id));
  const totals = new Float64Array(n);
  for (let row = 0; row < n; row++) {
    for (let group = 0; group < GROUP_COUNT; group++) {
      const value = groupColumns[group][row];
      const count = Number.isFinite(value) && value > 0 ? value : 0;
      groupCountsHost[row * GROUP_COUNT + group] = count;
      totals[row] += count;
    }
  }
  const communityArea = tracts.column<Uint8Array>('communityArea');
  const areaOf = Int32Array.from(communityArea, id => id - 1);
  const areaName = (index: number): string =>
    String(areaFeatures[index]?.properties?.name ?? `Area ${index + 1}`);
  const tractAreaName = (row: number): string => areaName(areaOf[row]);
  const citywide = getAspatialIndices(groupCountsHost, n);
  const areaCountsHost = sumRowsByUnit(groupCountsHost, n, areaOf, areaCount);
  const areaCitywide = getAspatialIndices(areaCountsHost, areaCount);
  const areaTotals = new Float64Array(areaCount);
  for (let area = 0; area < areaCount; area++) {
    for (let group = 0; group < GROUP_COUNT; group++) {
      areaTotals[area] += areaCountsHost[area * GROUP_COUNT + group];
    }
  }
  const cityDiversity =
    citywide.shares.reduce((sum, share) => sum + (share > 0 ? share * Math.log(1 / share) : 0), 0) /
    LOG_GROUP_COUNT;
  const classReal = getDominanceClasses(groupCountsHost, n);
  const dominance = countDominanceCells(classReal);
  const supermajorityCount = [2, 5, 8, 11, 14].reduce(
    (sum, offset) => sum + dominance.cells[offset],
    0
  );
  const noMajorityCount = [0, 3, 6, 9, 12].reduce(
    (sum, offset) => sum + dominance.cells[offset],
    0
  );
  let shuffledCountsHost = shuffleRows(groupCountsHost, n, ctx.options.shuffleSeed);
  let classShuffled = getDominanceClasses(shuffledCountsHost, n);

  // The tract each focus place stands for: its most populous tract.
  const getFocusRow = (name: string): number => {
    const area = areaFeatures.findIndex(feature => feature.properties?.name === name);
    let best = -1;
    for (let row = 0; row < n; row++) {
      if (areaOf[row] === area && (best < 0 || totals[row] > totals[best])) best = row;
    }
    return best;
  };

  // Outline segments per tract, for the selection and the members of a radius.
  const outlineStart = new Uint32Array(n + 1);
  for (const feature of tractMesh.outlineFeatures) outlineStart[feature + 1]++;
  for (let row = 0; row < n; row++) outlineStart[row + 1] += outlineStart[row];
  const collectSegments = (rows: Iterable<number>): Float32Array => {
    const parts: Float32Array[] = [];
    let length = 0;
    for (const row of rows) {
      const part = tractMesh.outlineSegments.subarray(
        outlineStart[row] * 4,
        outlineStart[row + 1] * 4
      );
      parts.push(part);
      length += part.length;
    }
    const segments = new Float32Array(length);
    let offset = 0;
    for (const part of parts) {
      segments.set(part, offset);
      offset += part.length;
    }
    return segments;
  };
  const getRings = (geometryOf: {geometry: unknown} | undefined): LngLat[][] =>
    getGeometryPolygons(geometryOf?.geometry as Parameters<typeof getGeometryPolygons>[0]).flatMap(
      polygon => polygon.map(ring => ring.map(point => [point[0], point[1]] as LngLat))
    );

  // GPU buffers.
  const layoutIndices = getGPUSegregationLayout(GROUP_COUNT);
  const positionsBuffer = resources.createBuffer('positions', points);
  const groupCounts = resources.createBuffer('group-counts', groupCountsHost);
  const shuffledCounts = resources.createBuffer('group-counts-shuffled', shuffledCountsHost);
  const indices = resources.createBuffer('indices', SCALE_COUNT * layoutIndices.stride * 4);
  const shuffledIndices = resources.createBuffer(
    'indices-shuffled',
    SCALE_COUNT * layoutIndices.stride * 4
  );
  const localEnvironment = resources.createBuffer(
    'local-environment',
    SCALE_COUNT * n * GROUP_COUNT * 4
  );
  const localEntropy = resources.createBuffer('local-entropy', SCALE_COUNT * n * 4);
  const localDissimilarity = resources.createBuffer(
    'local-dissimilarity',
    SCALE_COUNT * n * GROUP_COUNT * 4
  );
  const localTheil = resources.createBuffer('local-theil', SCALE_COUNT * n * 4);
  const display = resources.createBuffer('display', n * 4);
  const displayParameter = resources.createParameterBuffer('display-parameter', 'uint32', 6);
  const classRealBuffer = resources.createBuffer('class-real', classReal);
  const classShuffledBuffer = resources.createBuffer('class-shuffled', classShuffled);
  const areaValues = resources.createBuffer('area-values', areaCount * 4);
  const highlightChannel = resources.createBuffer('highlight-channel', new Float32Array(n));
  const selectionBuffer = resources.createBuffer('selection', tractMesh.outlineSegments.byteLength);
  const membersBuffer = resources.createBuffer('members', tractMesh.outlineSegments.byteLength);
  let selectionCount = 0;
  let membersCount = 0;
  const neighborCapacity = n * n;
  const scaleBuffers = RADIUS_LADDER.map((_, scale) => ({
    capacity: neighborCapacity,
    searchParameter: resources.createParameterBuffer(
      `search-parameter-${scale}`,
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    ),
    offsets: resources.createBuffer(`offsets-${scale}`, (n + 1) * 4),
    neighbors: resources.createBuffer(`neighbors-${scale}`, neighborCapacity * 4),
    weights: resources.createBuffer(`weights-${scale}`, neighborCapacity * 4),
    distances: resources.createBuffer(`distances-${scale}`, neighborCapacity * 4),
    overflow: resources.createBuffer(`overflow-${scale}`, 4)
  }));

  const importWeights = (graph: GPUCommandGraph<void>, scale: number): GPUSpatialWeights => {
    const buffers = scaleBuffers[scale];
    return {
      offsets: importGraphBuffer(graph, `offsets-${scale}`, buffers.offsets, 'uint32', n + 1),
      neighbors: importGraphBuffer(
        graph,
        `neighbors-${scale}`,
        buffers.neighbors,
        'uint32',
        buffers.capacity
      ),
      weights: importGraphBuffer(
        graph,
        `weights-${scale}`,
        buffers.weights,
        'float32',
        buffers.capacity
      ),
      distances: importGraphBuffer(
        graph,
        `distances-${scale}`,
        buffers.distances,
        'float32',
        buffers.capacity
      )
    };
  };

  // Search graph: one distance band per scale; the radius is a parameter write.
  const searchGraph = new GPUCommandGraph<void>(device, {id: 'segregation-search'});
  const positionsView = importGraphBuffer(
    searchGraph,
    'positions',
    positionsBuffer,
    'float32x2',
    n
  );
  RADIUS_LADDER.forEach((_, scale) => {
    const buffers = scaleBuffers[scale];
    searchGraph.add(
      new GPUNeighborSearch({
        id: `search-${scale}`,
        mode: 'radius',
        gridSize: [32, 32],
        positions: positionsView,
        parameters: buffers.searchParameter.importToGraph(searchGraph),
        weights: importWeights(searchGraph, scale),
        overflow: importGraphBuffer(searchGraph, `overflow-${scale}`, buffers.overflow, 'uint32', 1)
      })
    );
  });
  const searchCompiled = resources.track(searchGraph.compile());

  const getKey = (options: SegregationOptions) =>
    [
      options.weights,
      options.rowStandardize ? 'row' : '-',
      options.spatialForm,
      options.selfWeight,
      options.atkinsonB
    ].join('|');

  // Variant graph: the weights transforms, then two GPUSegregation nodes over the same weights.
  const compileVariant = (options: SegregationOptions): Variant => {
    const key = getKey(options);
    const graph = new GPUCommandGraph<void>(device, {id: `segregation-${key}`});
    const scaleWeights: (GPUSpatialWeights | null)[] = [null];
    RADIUS_LADDER.forEach((_, scale) => {
      const weights = importWeights(graph, scale);
      if (options.weights !== 'band') {
        graph.add(
          new GPUSpatialWeightsTransform({
            id: `kernel-${scale}`,
            operation: 'kernel',
            weights,
            kernel: options.weights,
            bandwidth: 'adaptive'
          })
        );
      }
      if (options.rowStandardize) {
        graph.add(new GPUSpatialWeightsTransform({id: `row-${scale}`, operation: 'row', weights}));
      }
      scaleWeights.push(weights);
    });
    const view = (
      name: string,
      buffer: typeof localEntropy,
      length: number
    ): GraphDataView<'float32'> => importGraphBuffer(graph, name, buffer, 'float32', length);
    const common = {
      unitCount: n,
      groupCount: GROUP_COUNT,
      scales: scaleWeights,
      selfWeight: Number(options.selfWeight),
      spatialForm: options.spatialForm,
      atkinsonB: Number(options.atkinsonB)
    } as const;
    graph.add(
      new GPUSegregation({
        id: 'segregation',
        ...common,
        groupCounts: view('group-counts', groupCounts, n * GROUP_COUNT),
        indices: view('indices', indices, SCALE_COUNT * layoutIndices.stride),
        local: {
          environment: view('local-environment', localEnvironment, SCALE_COUNT * n * GROUP_COUNT),
          entropy: view('local-entropy', localEntropy, SCALE_COUNT * n),
          dissimilarity: view(
            'local-dissimilarity',
            localDissimilarity,
            SCALE_COUNT * n * GROUP_COUNT
          ),
          theil: view('local-theil', localTheil, SCALE_COUNT * n)
        }
      })
    );
    // The null: the same weights, the rows of the count table moved between tracts.
    graph.add(
      new GPUSegregation({
        id: 'segregation-shuffled',
        ...common,
        groupCounts: view('group-counts-shuffled', shuffledCounts, n * GROUP_COUNT),
        indices: view('indices-shuffled', shuffledIndices, SCALE_COUNT * layoutIndices.stride)
      })
    );
    return {key, compiled: resources.track(graph.compile())};
  };

  const variants = new Map<string, Variant>();
  const getVariant = (options: SegregationOptions): Variant => {
    const key = getKey(options);
    let variant = variants.get(key);
    if (!variant) {
      variant = compileVariant(options);
      variants.set(key, variant);
    }
    return variant;
  };

  // Display graph: one value per tract from the chosen scale and group.
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'segregation-display'});
  const importDisplay = (name: string, buffer: typeof display, length: number) =>
    importGraphBuffer(displayGraph, name, buffer, 'float32', length);
  addKernelPass(displayGraph, {
    id: 'segregation-display',
    invocationCount: n,
    bindings: [
      {
        name: 'parameters',
        view: displayParameter.importToGraph(displayGraph),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'counts',
        view: importDisplay('group-counts', groupCounts, n * GROUP_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'environment',
        view: importDisplay('local-environment', localEnvironment, SCALE_COUNT * n * GROUP_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'entropy',
        view: importDisplay('local-entropy', localEntropy, SCALE_COUNT * n),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'display',
        view: importDisplay('display', display, n),
        type: 'f32',
        access: 'read_write'
      }
    ],
    declarations: `const UNITS: u32 = ${n}u;
const GROUPS: u32 = ${GROUP_COUNT}u;
const LOG_GROUPS: f32 = ${LOG_GROUP_COUNT.toFixed(8)};
const CLIP: f32 = ${RELATIVE_CLIP.toFixed(1)};`,
    body: /* wgsl */ `
  let mode = parameters[parametersOffset];
  let scale = parameters[parametersOffset + 1u];
  let group = parameters[parametersOffset + 2u];
  let cityShare = bitcast<f32>(parameters[parametersOffset + 3u]);
  var total = 0.0;
  for (var member = 0u; member < GROUPS; member++) {
    total += counts[countsOffset + index * GROUPS + member];
  }
  let row = scale * UNITS + index;
  var environmentTotal = 0.0;
  for (var member = 0u; member < GROUPS; member++) {
    environmentTotal += environment[environmentOffset + row * GROUPS + member];
  }
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (parameters[parametersOffset + 4u] >> 31u));
  if (total > 0.0 && environmentTotal > 0.0) {
    if (mode == 0u) {
      let share = environment[environmentOffset + row * GROUPS + group];
      value = clamp(log2(max(share, 0.000001) / max(cityShare, 0.000001)), -CLIP, CLIP);
    } else {
      value = entropy[entropyOffset + row] / LOG_GROUPS;
    }
  }
  display[displayOffset + index] = value;`
  });
  const displayCompiled = resources.track(displayGraph.compile());

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------

  let active = getVariant(ctx.options);
  let analysisDirty = true;
  let displayDirty = true;
  let latest: Latest | null = null;
  // False from a change of map, scale or focal group until the readback of that display arrives.
  let displayCurrent = true;
  // Set once the instance exists: layers are requested only after that.
  let created = false;
  let legendHighlight: number[] | null = null;
  let selectedRow = getFocusRow(ctx.options.focusArea);
  let clickedSelection = false;
  let memberRows: number[] = [];

  const getFocalIndex = () =>
    Math.max(
      0,
      SEGREGATION_GROUPS.findIndex(group => group.id === ctx.options.focalGroup)
    );
  const getFocalLabel = () => SEGREGATION_GROUPS[getFocalIndex()].label;
  const getRadiusMeters = (scale: number) =>
    scale <= 0 ? 0 : RADIUS_LADDER[scale - 1] * ctx.options.bandwidth * 1000;
  const formatRadius = (scale: number) => {
    if (scale <= 0) return 'aspatial';
    const kilometers = Number((getRadiusMeters(scale) / 1000).toFixed(2));
    return `${kilometers} km`;
  };
  const getFocalCityShare = () => citywide.shares[getFocalIndex()];
  const isAreaMap = () => ctx.options.units === 'areas' && ctx.options.localIndex === 'relative';
  const isRingShown = () => ctx.options.showRing && ctx.options.scale > 0;

  const writeSearchParameters = () => {
    scaleBuffers.forEach((buffers, index) => {
      buffers.searchParameter.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius: RADIUS_LADDER[index] * ctx.options.bandwidth * 1000,
          weightKind: 'binary'
        })
      );
    });
    analysisDirty = true;
    reader.markStale();
  };
  const writeDisplayParameter = () => {
    const {localIndex, scale} = ctx.options;
    displayParameter.write(
      Uint32Array.of(
        DISPLAY_MODE[localIndex],
        scale,
        getFocalIndex(),
        getFloatBits(getFocalCityShare()),
        0,
        0
      )
    );
    displayDirty = true;
    reader.markStale();
  };
  // Over- and under-representation of the focal group by community area (aspatial, CPU).
  const writeAreaValues = () => {
    const focal = getFocalIndex();
    const cityShare = getFocalCityShare();
    const values = new Float32Array(areaCount);
    for (let area = 0; area < areaCount; area++) {
      const share =
        areaTotals[area] > 0 ? areaCountsHost[area * GROUP_COUNT + focal] / areaTotals[area] : NaN;
      values[area] = Number.isFinite(share)
        ? Math.max(
            -RELATIVE_CLIP,
            Math.min(RELATIVE_CLIP, Math.log2(Math.max(share, 1e-6) / Math.max(cityShare, 1e-6)))
          )
        : Number.NaN;
    }
    areaValues.write(values);
    return values;
  };
  let areaValueHost = writeAreaValues();

  // -------------------------------------------------------------------------------------------
  // Selection, ring and notes
  // -------------------------------------------------------------------------------------------

  const refreshSelection = () => {
    if (selectedRow < 0) {
      selectionCount = 0;
      membersCount = 0;
      memberRows = [];
      ctx.setAnnotations('segregation-ring', null);
    } else {
      const own = collectSegments([selectedRow]);
      selectionBuffer.write(own);
      selectionCount = own.length / 4;
      memberRows = isRingShown()
        ? getRadiusMembers(points, n, selectedRow, getRadiusMeters(ctx.options.scale))
        : [];
      const around = collectSegments(memberRows);
      membersBuffer.write(around);
      membersCount = around.length / 4;
      const channel = new Float32Array(n);
      channel[selectedRow] = 1;
      for (const member of memberRows) channel[member] = 1;
      highlightChannel.write(channel);
      ctx.setAnnotations(
        'segregation-ring',
        isRingShown()
          ? [
              {
                kind: 'ring',
                id: 'segregation-ring',
                coordinate: pointLngLat[selectedRow],
                radiusMeters: getRadiusMeters(ctx.options.scale),
                text: `${formatRadius(ctx.options.scale)} environment`,
                dashed: true,
                tone: 'signal'
              }
            ]
          : null
      );
    }
    publishSelection();
    if (created) ctx.requestLayers();
  };

  // One call carries every runtime furniture field: each call replaces the previous one.
  const updateFurniture = () => {
    const {scale} = ctx.options;
    ctx.setFurniture({
      title: {sample: `${formatCount(n)} census tracts, ACS 2018-2022 via CDC SVI 2022`},
      scaleBar: scale > 0 ? {units: 'metric', ticks: [getRadiusMeters(scale)]} : false
    });
  };

  const updateNotes = () => {
    const {localIndex, scale, showRing} = ctx.options;
    const notes: MapAnnotation[] = [];
    if (latest && localIndex === 'relative' && !showRing && !isAreaMap()) {
      // The tract that most over-represents the focal group among tracts large enough to name.
      let best = -1;
      for (let row = 0; row < n; row++) {
        if (totals[row] < NOTE_MINIMUM_RESIDENTS || !Number.isFinite(latest.display[row])) continue;
        if (best < 0 || latest.display[row] > latest.display[best]) best = row;
      }
      if (best >= 0) {
        const focal = getFocalIndex();
        notes.push({
          kind: 'note',
          id: 'segregation-note-over',
          coordinate: pointLngLat[best],
          priority: 5,
          title: liveText('{ratio:fixed:1} times the city share', {
            ratio: 2 ** latest.display[best]
          }),
          text: `${tractAreaName(best)}: ${formatPercent(
            groupCountsHost[best * GROUP_COUNT + focal] / totals[best]
          )} ${getFocalLabel()}${scale > 0 ? ` within ${formatRadius(scale)}` : ''}`
        });
      }
    }
    if (latest && localIndex === 'diversity' && scale > 0) {
      const rowsOfArea = (area: number) =>
        Array.from({length: n}, (_, row) => row).filter(row => areaOf[row] === area);
      for (const name of MIXING_AREAS) {
        const area = areaFeatures.findIndex(feature => feature.properties?.name === name);
        const point = areaMesh.labelPoints[area];
        if (area < 0 || !point || !Number.isFinite(point[0])) continue;
        const mean = getWeightedMean(latest.display, totals, rowsOfArea(area));
        notes.push({
          kind: 'note',
          id: `segregation-note-${area}`,
          coordinate: [point[0], point[1]],
          priority: 4,
          title: liveText('{value:fixed:2} of the maximum', {value: mean}),
          text: `${name}, ${formatRadius(scale)} environments`
        });
      }
    }
    ctx.setAnnotations('segregation-notes', notes.length ? notes : null);
  };

  // -------------------------------------------------------------------------------------------
  // Readback, readouts, charts
  // -------------------------------------------------------------------------------------------

  const column = (source: Float32Array, scale: number, field: number) =>
    source[scale * layoutIndices.stride + field];

  const getProfileChart = (): LineChartData | null => {
    if (!latest) return null;
    const labels = Array.from({length: SCALE_COUNT}, (_, scale) => formatRadius(scale));
    const x = Array.from({length: SCALE_COUNT}, (_, scale) => scale);
    const real = x.map(scale =>
      column(latest!.indices, scale, layoutIndices.multiGroupDissimilarity)
    );
    const series: LineChartData['series'][number][] = [
      {label: 'Real tracts', x, y: real, color: 5, directLabel: true, points: true}
    ];
    if (ctx.options.shuffle) {
      series.push({
        label: 'Tracts shuffled',
        x,
        y: x.map(scale =>
          column(latest!.shuffledIndices, scale, layoutIndices.multiGroupDissimilarity)
        ),
        dashed: true,
        ghost: true,
        directLabel: true
      });
    }
    return {
      kind: 'line',
      title: 'Multigroup dissimilarity D by environment radius',
      series,
      xDomain: [0, SCALE_COUNT - 1],
      yDomain: [0, 0.9],
      xLabel: 'Environment radius (each step doubles)',
      yLabel: 'D',
      formatX: value => (Number.isInteger(value) ? (labels[value] ?? '') : ''),
      formatY: value => value.toFixed(1),
      link: {option: 'scale', label: value => labels[Math.round(value)] ?? ''},
      description:
        'Multigroup dissimilarity D falls as the environment widens; the same tracts shuffled across the city start at the same D and fall much faster.'
    };
  };

  const getGroupProfileChart = (): LineChartData | null => {
    if (!latest) return null;
    const labels = Array.from({length: SCALE_COUNT}, (_, scale) => formatRadius(scale));
    const x = Array.from({length: SCALE_COUNT}, (_, scale) => scale);
    // Registry order White, Black, Hispanic, Asian takes chart slots 0 to 3 (blue, orange, green,
    // purple), the nearest the chart palette has to the group hues; Other is a grey ghost line.
    const order = [1, 0, 2, 3, 4];
    return {
      kind: 'line',
      title: 'Dissimilarity D of each group by environment radius',
      series: order.map((group, position) => ({
        label: SEGREGATION_GROUPS[group].label.split(' ')[0],
        x,
        y: x.map(scale => column(latest!.indices, scale, layoutIndices.dissimilarity + group)),
        color: position,
        ghost: position === 4,
        directLabel: true
      })),
      xDomain: [0, SCALE_COUNT - 1],
      yDomain: [0, 0.9],
      xLabel: 'Environment radius (each step doubles)',
      yLabel: 'D',
      formatX: value => (Number.isInteger(value) ? (labels[value] ?? '') : ''),
      formatY: value => value.toFixed(1),
      link: {option: 'scale', label: value => labels[Math.round(value)] ?? ''},
      description:
        'Dissimilarity of each group by radius: the Black curve stays high at wide radii, the Hispanic curve falls to near zero.'
    };
  };

  const getCompositionChart = (): ChartData | null => {
    if (!latest || selectedRow < 0) return null;
    const ground = ctx.ground();
    const {scale} = ctx.options;
    const segments = (shares: (group: number) => number) =>
      SEGREGATION_GROUPS.map((group, index) => ({
        label: group.label.split(' ')[0],
        value: shares(index),
        color: getGroupColor(index, ground) as ChartColor
      }));
    const own = segments(group => groupCountsHost[selectedRow * GROUP_COUNT + group]);
    const environment = segments(
      group => latest!.environment[(scale * n + selectedRow) * GROUP_COUNT + group]
    );
    const city = segments(group => citywide.shares[group]);
    return {
      kind: 'multiples',
      columns: 1,
      shareDomains: false,
      titles: [
        `This tract (${tractAreaName(selectedRow)})`,
        scale > 0 ? `Within ${formatRadius(scale)}` : 'Its environment (aspatial: itself)',
        'Chicago'
      ],
      charts: [
        {kind: 'stacked', segments: own},
        {kind: 'stacked', segments: environment},
        {kind: 'stacked', segments: city}
      ],
      description:
        'Three stacked share bars: the composition of the selected tract, of everyone within the radius, and of the whole city.'
    };
  };

  const publishLegendData = () => {
    const {scale} = ctx.options;
    const areaMap = isAreaMap();
    const relativeValues = areaMap ? areaValueHost : latest?.display;
    ctx.setLegendData('segregation', {
      groundIsDark: ctx.ground() === 'dark',
      dominanceCells: dominance.cells,
      emptyTracts: dominance.empty,
      relativeCounts:
        relativeValues && (areaMap || displayCurrent) && ctx.options.localIndex === 'relative'
          ? countClasses(relativeValues, RELATIVE_BREAKS)
          : null,
      diversityCounts:
        latest && displayCurrent && ctx.options.localIndex === 'diversity'
          ? countClasses(latest.display, DIVERSITY_BREAKS)
          : null,
      focalCityShare: getFocalCityShare(),
      cityDiversity,
      environmentLabel: areaMap
        ? 'the community area'
        : scale === 0
          ? 'the tract'
          : `everyone within ${formatRadius(scale)}`,
      focalLabel: getFocalLabel()
    } satisfies SegregationLegendData);
  };

  const publishStatic = () => {
    ctx.setReadout('residents', citywide.total);
    ctx.setReadout('supermajority', supermajorityCount);
    ctx.setReadout('noMajority', noMajorityCount);
    ctx.setReadout(
      'capacity',
      `${formatCount(neighborCapacity)} slots per radius, 4 arrays, ${RADIUS_LADDER.length} radii (capacity = tracts squared; a count pass would size it)`
    );
  };

  const publishFocal = () => {
    const focal = getFocalIndex();
    ctx.setReadout('focalLabel', getFocalLabel());
    ctx.setReadout('cityShare', formatPercent(getFocalCityShare()));
    ctx.setReadout('dAreas', formatIndex(areaCitywide.multigroup, 3));
    ctx.setReadout('dFocalAreas', formatIndex(areaCitywide.perGroup[focal], 3));
  };

  const publishSelection = () => {
    if (selectedRow < 0) {
      ctx.setReadout('selected', null);
      ctx.setReadout('members', null);
      ctx.setReadout('ownShare', null);
      ctx.setReadout('envShare', null);
      return;
    }
    const focal = getFocalIndex();
    const {scale} = ctx.options;
    const geoid = tractFeatures[selectedRow]?.properties?.GEOID ?? selectedRow;
    ctx.setReadout(
      'selected',
      `${tractAreaName(selectedRow)}, tract ${geoid}: ${formatCount(totals[selectedRow])} residents`
    );
    ctx.setReadout('members', isRingShown() ? memberRows.length : null);
    ctx.setReadout(
      'ownShare',
      formatPercent(groupCountsHost[selectedRow * GROUP_COUNT + focal] / totals[selectedRow])
    );
    ctx.setReadout(
      'envShare',
      latest
        ? formatPercent(latest.environment[(scale * n + selectedRow) * GROUP_COUNT + focal])
        : null
    );
    ctx.setChart('composition', getCompositionChart());
  };

  const publishReadouts = () => {
    if (!latest) return;
    const {scale} = ctx.options;
    const focal = getFocalIndex();
    const radiusMid = 3;
    const radiusWide = 4;
    ctx.setReadout('radius', formatRadius(scale));
    ctx.setReadout('radiusWide', formatRadius(radiusWide));
    ctx.setReadout('radiusMid', formatRadius(radiusMid));
    ctx.setReadout(
      'dAspatial',
      formatIndex(column(latest.indices, 0, layoutIndices.multiGroupDissimilarity), 3)
    );
    ctx.setReadout(
      'dAspatialShuffled',
      formatIndex(column(latest.shuffledIndices, 0, layoutIndices.multiGroupDissimilarity), 3)
    );
    ctx.setReadout(
      'dFocal',
      formatIndex(column(latest.indices, 0, layoutIndices.dissimilarity + focal), 3)
    );
    ctx.setReadout(
      'dMidReal',
      formatIndex(column(latest.indices, radiusMid, layoutIndices.multiGroupDissimilarity), 3)
    );
    ctx.setReadout(
      'dMidShuffled',
      formatIndex(
        column(latest.shuffledIndices, radiusMid, layoutIndices.multiGroupDissimilarity),
        3
      )
    );
    ctx.setReadout(
      'dScale',
      formatIndex(column(latest.indices, scale, layoutIndices.multiGroupDissimilarity), 3)
    );
    ctx.setReadout(
      'entropyIndex',
      formatIndex(column(latest.indices, scale, layoutIndices.entropy), 3)
    );
    ctx.setReadout(
      'isolation',
      formatIndex(column(latest.indices, scale, layoutIndices.isolation + focal), 3)
    );
    ctx.setReadout(
      'atkinson',
      formatIndex(column(latest.indices, scale, layoutIndices.atkinson + focal), 3)
    );
    const blackIndex = SEGREGATION_GROUPS.findIndex(group => group.id === 'nhBlack');
    const hispanicIndex = SEGREGATION_GROUPS.findIndex(group => group.id === 'hispanic');
    ctx.setReadout(
      'blackWide',
      formatIndex(column(latest.indices, radiusWide, layoutIndices.dissimilarity + blackIndex), 3)
    );
    ctx.setReadout(
      'hispanicWide',
      formatIndex(
        column(latest.indices, radiusWide, layoutIndices.dissimilarity + hispanicIndex),
        3
      )
    );
    ctx.setReadout('cityDiversity', formatIndex(cityDiversity, 2));
    let localDissimilaritySum = 0;
    let localTheilSum = 0;
    for (let unit = 0; unit < n; unit++) {
      localTheilSum += latest.theil[scale * n + unit];
      localDissimilaritySum += latest.dissimilarity[(scale * n + unit) * GROUP_COUNT + focal];
    }
    ctx.setReadout(
      'localSum',
      `local H sums to ${localTheilSum.toFixed(3)} (global ${column(latest.indices, scale, layoutIndices.entropy).toFixed(3)}); local D of ${getFocalLabel()} sums to ${localDissimilaritySum.toFixed(3)} (global ${column(latest.indices, scale, layoutIndices.dissimilarity + focal).toFixed(3)})`
    );
  };

  const publishCharts = () => {
    ctx.setChart('profile', getProfileChart());
    ctx.setChart('profileAll', getGroupProfileChart());
    ctx.setChart('composition', getCompositionChart());
  };

  const reader = new SummaryReader(
    resources,
    'segregation',
    [
      {buffer: indices, size: SCALE_COUNT * layoutIndices.stride * 4},
      {buffer: shuffledIndices, size: SCALE_COUNT * layoutIndices.stride * 4},
      ...scaleBuffers.map(buffers => ({buffer: buffers.overflow, size: 4})),
      {buffer: display, size: n * 4},
      {buffer: localEnvironment, size: SCALE_COUNT * n * GROUP_COUNT * 4},
      {buffer: localTheil, size: SCALE_COUNT * n * 4},
      {buffer: localDissimilarity, size: SCALE_COUNT * n * GROUP_COUNT * 4}
    ],
    bytes => {
      const read = createByteReader(bytes);
      const indexValues = read.floats(SCALE_COUNT * layoutIndices.stride);
      const shuffledValues = read.floats(SCALE_COUNT * layoutIndices.stride);
      const flags = read.words(RADIUS_LADDER.length);
      const displayValues = read.floats(n);
      const environment = read.floats(SCALE_COUNT * n * GROUP_COUNT);
      const theil = read.floats(SCALE_COUNT * n);
      const dissimilarity = read.floats(SCALE_COUNT * n * GROUP_COUNT);
      const finite: number[] = [];
      for (const value of displayValues) if (Number.isFinite(value)) finite.push(value);
      latest = {
        indices: indexValues,
        shuffledIndices: shuffledValues,
        display: displayValues,
        environment,
        theil,
        dissimilarity,
        sortedDisplay: Float64Array.from(finite).sort()
      };
      ctx.setReadout(
        'overflow',
        flags.some(flag => flag) ? `YES in radii ${Array.from(flags).join(', ')}` : 'no'
      );
      displayCurrent = true;
      publishReadouts();
      publishCharts();
      publishLegendData();
      publishSelection();
      updateNotes();
      ctx.requestLayers();
    }
  );

  ctx.setCost({records: n, note: `${RADIUS_LADDER.length} radii, real and shuffled counts`});
  publishStatic();
  publishFocal();
  publishLegendData();
  writeSearchParameters();
  writeDisplayParameter();
  updateFurniture();
  refreshSelection();
  created = true;

  // -------------------------------------------------------------------------------------------
  // Layers
  // -------------------------------------------------------------------------------------------

  const getTable = (): ClassTable =>
    ctx.options.localIndex === 'diversity'
      ? getDiversityTable(ctx.ground())
      : getRelativeTable(ctx.ground());

  const getLayers = (): Layer[] => {
    const o = ctx.options;
    const ground = ctx.ground();
    const layers: Layer[] = [];
    const noData = NO_DATA_COLOR[ground];
    const areaMap = isAreaMap();
    const common = {
      coordinateOrigin: layerOrigin,
      opacity: FILL_OPACITY[ground],
      noDataColor: noData,
      hatchNoData: true,
      hatchColor: getInkColor(ground, 90),
      highlightClasses: legendHighlight
    } as const;
    const tractGeometry = {
      triangles: tractPolygons.triangles,
      features: tractPolygons.triangleFeatures,
      vertexCount: tractPolygons.vertexCount
    };
    const ringShown = isRingShown();
    const channels = {
      instanceChannels: highlightChannel,
      channelStride: 1,
      channels: {highlight: 0},
      highlightActive: ringShown,
      dimOpacity: 0.28
    } as const;
    if (areaMap) {
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: 'segregation-area-fill',
          ...common,
          triangles: areaPolygons.triangles,
          features: areaPolygons.triangleFeatures,
          vertexCount: areaPolygons.vertexCount,
          values: areaValues,
          valueFormat: 'float32',
          colormap: 'puor',
          ...getClassTableLayerProps(getTable())
        })
      );
    } else if (o.localIndex === 'dominant') {
      const palette = getDominanceColors(ground);
      const sides: {id: string; values: typeof classRealBuffer; side?: 'a' | 'b'}[] = o.shuffle
        ? [
            {id: 'segregation-class-real', values: classRealBuffer, side: 'a'},
            {id: 'segregation-class-shuffled', values: classShuffledBuffer, side: 'b'}
          ]
        : [{id: 'segregation-class-real', values: classRealBuffer}];
      for (const {id, values, side} of sides) {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id,
            ...common,
            ...tractGeometry,
            ...channels,
            values,
            valueFormat: 'uint32',
            colormap: 'category',
            palette,
            highlightClasses: null,
            compareSide: side
          })
        );
      }
    } else {
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: 'segregation-tract-fill',
          ...common,
          ...tractGeometry,
          ...channels,
          values: display,
          valueFormat: 'float32',
          colormap: o.localIndex === 'relative' ? 'puor' : 'greys',
          ...getClassTableLayerProps(getTable())
        })
      );
    }
    if (!areaMap) {
      // Tier 3: tract hairlines, thin and in the ground colour.
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'segregation-hairlines',
          coordinateOrigin: layerOrigin,
          segments: tractPolygons.outline,
          instanceCount: tractPolygons.outlineCount,
          widthPixels: 0.5,
          color: ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 161]
        })
      );
    }
    // Community-area boundaries: context over the tracts, the subject when areas are the unit.
    const areaLine = getAreaLineStyle(ground, areaMap);
    layers.push(
      new SpatialAnalysisSegmentLayer({
        id: 'segregation-area-lines',
        coordinateOrigin: layerOrigin,
        segments: areaPolygons.outline,
        instanceCount: areaPolygons.outlineCount,
        widthPixels: areaLine.widthPixels,
        color: areaLine.color,
        ...(areaMap
          ? {
              outlineColor: areaLine.casing,
              outlineWidthPixels: (areaLine.casingPixels - areaLine.widthPixels) / 2
            }
          : {})
      })
    );
    if (!areaMap && selectedRow >= 0 && (ringShown || clickedSelection)) {
      if (ringShown && membersCount) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'segregation-members',
            coordinateOrigin: layerOrigin,
            segments: membersBuffer,
            instanceCount: membersCount,
            widthPixels: 1.2,
            color: getInkColor(ground)
          })
        );
      }
      // Selection: ink core 2.5 px over a 4.5 px ground casing.
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'segregation-selection',
          coordinateOrigin: layerOrigin,
          segments: selectionBuffer,
          instanceCount: selectionCount,
          widthPixels: 2.5,
          color: getInkColor(ground),
          outlineColor: getCasingColor(ground),
          outlineWidthPixels: 1
        })
      );
    }
    return layers;
  };

  // -------------------------------------------------------------------------------------------
  // Tooltip
  // -------------------------------------------------------------------------------------------

  const swatchOf = (table: ClassTable, value: number) => {
    const color = table.colors[Math.max(0, getClassIndex(value, table.breaks))];
    return [color[0], color[1], color[2], color[3] ?? 255] as const;
  };

  const shareRows = (counts: ArrayLike<number>, row: number, total: number): TooltipRow[] =>
    SEGREGATION_GROUPS.map((group, index) => ({
      label: group.label.split(' ')[0],
      value: formatPercent(counts[row * GROUP_COUNT + index] / total),
      swatch: getGroupColor(index, ctx.ground())
    }));

  const describeTract = (row: number): TooltipContent | null => {
    const o = ctx.options;
    const ground = ctx.ground();
    const total = totals[row];
    const geoid = tractFeatures[row]?.properties?.GEOID ?? row;
    const labelPoint = tractMesh.labelPoints[row];
    const rows: TooltipRow[] = [];
    if (total <= 0) {
      rows.push({label: 'Residents', value: 0});
    } else if (o.localIndex === 'dominant') {
      const largest = getLargestGroup(groupCountsHost, row);
      if (largest) {
        const group = SEGREGATION_GROUPS[largest.group];
        const tier = getDominanceTier(largest.share);
        rows.push(
          {
            label: 'Largest group',
            value: group.label.split(' ')[0],
            unit: `${formatPercent(largest.share)} of residents`,
            swatch: getDominanceColors(ground)[group.registryIndex * 3 + tier],
            emphasis: true
          },
          {label: 'Share of the largest group', value: DOMINANCE_TIER_LABELS[tier]}
        );
      }
    } else if (latest) {
      const value = latest.display[row];
      const table = getTable();
      const percentile = getPercentile(latest.sortedDisplay, value);
      if (o.localIndex === 'relative') {
        rows.push({
          label: `${getFocalLabel().split(' ')[0]} share of ${o.scale === 0 ? 'the tract' : `everyone within ${formatRadius(o.scale)}`}`,
          value: Number.isFinite(value) ? `${(2 ** value).toFixed(1)}x` : 'n/a',
          unit: 'the city share',
          swatch: swatchOf(table, value),
          emphasis: true
        });
      } else {
        rows.push({
          label: 'Local diversity',
          value: Number.isFinite(value) ? value.toFixed(2) : 'n/a',
          unit: 'of the maximum',
          swatch: swatchOf(table, value),
          emphasis: true
        });
      }
      if (Number.isFinite(percentile)) {
        rows.push({
          label: 'Rank',
          value: `${formatOrdinal(Math.round(percentile * 100))} percentile`,
          unit: `of ${formatCount(latest.sortedDisplay.length)} tracts`
        });
      }
    }
    if (total > 0) {
      rows.push({label: 'Residents', value: formatCount(total)});
      rows.push(...shareRows(groupCountsHost, row, total));
      if (latest && o.scale > 0 && o.localIndex !== 'dominant') {
        const focal = getFocalIndex();
        rows.push({
          label: `${getFocalLabel().split(' ')[0]} within ${formatRadius(o.scale)}`,
          value: formatPercent(latest.environment[(o.scale * n + row) * GROUP_COUNT + focal])
        });
      }
    }
    return {
      title: tractAreaName(row),
      subtitle: `Census tract ${geoid}`,
      rows,
      anchor:
        labelPoint && Number.isFinite(labelPoint[0])
          ? ([labelPoint[0], labelPoint[1]] as LngLat)
          : undefined,
      highlight: {kind: 'polygon', rings: getRings(tractFeatures[row])}
    };
  };

  const describeArea = (index: number): TooltipContent | null => {
    const total = areaTotals[index];
    const value = areaValueHost[index];
    const table = getRelativeTable(ctx.ground());
    const rows: TooltipRow[] = [
      {
        label: `${getFocalLabel().split(' ')[0]} share, relative to the city`,
        value: Number.isFinite(value) ? `${(2 ** value).toFixed(1)}x` : 'n/a',
        unit: 'the city share',
        swatch: swatchOf(table, value),
        emphasis: true
      },
      {label: 'Residents', value: formatCount(total)},
      ...shareRows(areaCountsHost, index, Math.max(total, 1))
    ];
    const labelPoint = areaMesh.labelPoints[index];
    return {
      title: areaName(index),
      subtitle: 'Community area',
      rows,
      anchor:
        labelPoint && Number.isFinite(labelPoint[0])
          ? ([labelPoint[0], labelPoint[1]] as LngLat)
          : undefined,
      highlight: {kind: 'polygon', rings: getRings(areaFeatures[index])}
    };
  };

  return {
    getCompiledGraphs: () => [
      searchCompiled,
      ...[...variants.values()].map(variant => variant.compiled),
      displayCompiled
    ],

    setOption(id) {
      if (['weights', 'rowStandardize', 'spatialForm', 'selfWeight', 'atkinsonB'].includes(id)) {
        active = getVariant(ctx.options);
        analysisDirty = true;
        reader.markStale();
      } else if (id === 'bandwidth') {
        writeSearchParameters();
        updateFurniture();
        refreshSelection();
      } else if (id === 'shuffleSeed') {
        shuffledCountsHost = shuffleRows(groupCountsHost, n, ctx.options.shuffleSeed);
        classShuffled = getDominanceClasses(shuffledCountsHost, n);
        shuffledCounts.write(shuffledCountsHost);
        classShuffledBuffer.write(classShuffled);
        analysisDirty = true;
        reader.markStale();
      } else if (id === 'shuffle') {
        publishCharts();
      } else if (id === 'focalGroup') {
        displayCurrent = false;
        areaValueHost = writeAreaValues();
        publishFocal();
        writeDisplayParameter();
        publishSelection();
        publishLegendData();
        ctx.setAnnotations('segregation-notes', null);
      } else if (id === 'localIndex' || id === 'scale') {
        displayCurrent = false;
        legendHighlight = null;
        writeDisplayParameter();
        updateFurniture();
        if (id === 'scale') refreshSelection();
        publishLegendData();
        ctx.setAnnotations('segregation-notes', null);
      } else if (id === 'units') {
        legendHighlight = null;
        publishLegendData();
        updateNotes();
      } else if (id === 'focusArea') {
        const row = getFocusRow(ctx.options.focusArea);
        if (row >= 0) selectedRow = row;
        refreshSelection();
      } else if (id === 'showRing') {
        if (ctx.options.showRing && selectedRow < 0) {
          selectedRow = getFocusRow(ctx.options.focusArea);
        }
        refreshSelection();
        updateNotes();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (analysisDirty || frame.frameIndex < 2) {
        searchCompiled.encode(commandEncoder, {parameters: undefined});
        active.compiled.encode(commandEncoder, {parameters: undefined});
        analysisDirty = false;
        displayDirty = true;
      }
      if (displayDirty || frame.frameIndex < 2) {
        displayCompiled.encode(commandEncoder, {parameters: undefined});
        displayDirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers,

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange() {
      publishLegendData();
      publishSelection();
      publishCharts();
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      if (isAreaMap()) {
        const found = areaLocator.find(event.coordinate);
        return found ? describeArea(found.index) : null;
      }
      const found = tractLocator.find(event.coordinate);
      return found ? describeTract(found.index) : null;
    },

    onClick(event) {
      if (!event.coordinate || isAreaMap()) return false;
      const found = tractLocator.find(event.coordinate);
      const row = found ? found.index : -1;
      selectedRow = row === selectedRow && clickedSelection ? -1 : row;
      clickedSelection = selectedRow >= 0;
      refreshSelection();
      return true;
    },

    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
