// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addPointsInPolygonsChoroplethRecipe,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared,
  GPUZonalStatistics,
  type GPUZonalStatisticsExtentStatistic
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColormap
} from '../../engine/layers';
import {RAMP_STOPS, type RampName} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createPolygonSet,
  findPolygonAt,
  formatCompact,
  formatInteger,
  SegmentGridIndex
} from './b2-geometry';
import {
  createZoneRaster,
  getStopsPalette,
  importPolygons,
  readUint32,
  uploadPolygons,
  type ZoneRaster
} from './b2-graph';
import {PackedZoneFillLayer, ZoneFillLayer} from './b2-layers';

/** Option state of the observations-by-tract scene. */
export type ObservationsByTractOptions = {
  statistic: 'count' | 'sum' | 'mean' | 'minimum' | 'maximum' | 'density';
  valueKind: 'researchGrade' | 'introduced' | 'animal' | 'weekend' | 'category';
  groupType: string;
  denominator: 'area' | 'population';
  sumOrder: 'sorted' | 'atomic';
  includeBoundary: boolean;
  prepared: boolean;
  spatialSort: boolean;
  invalidateEveryFrame: boolean;
  rerunEveryFrame: boolean;
  display: 'ramp' | 'classes';
  backend: 'zonal' | 'group';
  classMethod: string;
  classCount: number;
  ramp: RampName;
  sqrtScale: boolean;
  opacity: number;
  pointsMode: 'off' | 'joined' | 'outside';
  outlines: boolean;
};

/** Categorical colors of the joined-tract point display (cycled by tract row). */
export const TRACT_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
];

const MAXIMUM_CLASSES = 7;
const CLASS_METHODS = [
  'equal-interval',
  'quantile',
  'natural-breaks',
  'standard-deviation',
  'head-tail',
  'box-plot'
] as const;
const ZONE_RASTER_LONG_SIDE = 1500;
const CANDIDATES_PER_POINT = 8;
const SETTLE_FRAMES = 3;

type AssignGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  prepared: GPUSpatialJoinPrepared | null;
};

type ZonalGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  /** Per-feature statistic shown on the map. */
  statistic: Buffer;
  statisticFormat: 'uint32' | 'float32';
};

type RecipeGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  usedClasses: number;
};

/** True for Saturday and Sunday; 2023-01-01 (second 0) is a Sunday. */
function isWeekend(seconds: number): boolean {
  const weekday = Math.floor(seconds / 86400) % 7;
  return weekday === 0 || weekday === 6;
}

/** Observation groups that are not wildlife: plants, fungi and the catch-all group. */
const NON_ANIMAL_GROUPS = new Set(['Plants', 'Fungi', 'Other life']);

/**
 * Nature observations joined to census tracts. One prepared tract index, three compiled graphs that share the
 * observation points: the assignment join (`GPUPointInPolygonJoin`), the zonal reduction
 * (`GPUZonalStatistics`) and the classified choropleth recipe. Everything the analyst can steer
 * without a rebuild (value column, group, ramp, classes, denominator values) is a buffer write.
 */
export async function createObservationsByTract(
  ctx: SceneContext<ObservationsByTractOptions>
): Promise<SceneInstance<ObservationsByTractOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const tractsData = ctx.datasets.get('chicago-tracts');
  const areasData = ctx.datasets.get('chicago-community-areas');
  const origin = tractsData.defaultOrigin;
  const projection = tractsData.getProjection(origin);

  const positions = observations.projectColumn('position', origin);
  const pointCount = positions.length / 2;
  const timestamps = observations.column<Uint32Array>('timestamp');
  const researchGrade = observations.column<Uint8Array>('researchGrade');
  const introduced = observations.column<Uint8Array>('introduced');
  const categories = observations.column<Uint8Array>('category');
  const observationTract = observations.column<Uint16Array>('tract');
  const categoryNames = observations.categories('category');
  const isAnimalGroup = categoryNames.map(name => !NON_ANIMAL_GROUPS.has(name));

  const tractSet = createPolygonSet(tractsData, origin);
  const featureCount = tractSet.featureCount;
  const edgeIndex = new SegmentGridIndex(tractSet.outline, 200, 10);
  const population = tractsData.column<Float32Array>('population');
  const areaKm2 = tractsData.column<Float32Array>('areaKm2');
  const dataObservationCounts = tractsData.column<Float32Array>('natureObs2023');
  const tractCommunityArea = tractsData.column<Uint8Array>('communityArea');
  const geoids = (tractsData.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown>)?.GEOID ?? '')
  );
  const areaNames = (areasData.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown>)?.name ?? '')
  );

  ctx.setStatus('Rasterizing the tract fill');
  const resources = new SpatialAnalysisResources(device, 'observations-by-tract');
  const polygons = uploadPolygons(resources, 'tracts', tractSet);
  const zoneRaster: ZoneRaster = await createZoneRaster(
    resources,
    'tract-fill',
    polygons,
    tractSet.bounds,
    ZONE_RASTER_LONG_SIDE
  );
  ctx.signal.throwIfAborted();

  // Caller-owned inputs and outputs shared by every graph variant.
  const positionsBuffer = resources.createBuffer('positions', positions);
  const values = new Float32Array(pointCount);
  const valuesBuffer = resources.createBuffer('values', values);
  const areasBuffer = resources.createBuffer('areas', featureCount * 4);
  const pointFeatures = resources.createBuffer('point-features', pointCount * 4);
  const assignCounts = resources.createBuffer('assign-counts', featureCount * 4);
  const assignOverflow = resources.createBuffer('assign-overflow', 4);
  const assignCandidates = resources.createBuffer('assign-candidates', 4);
  const assignUncertain = resources.createBuffer('assign-uncertain', 4);

  const zonalCounts = resources.createBuffer('zonal-counts', featureCount * 4);
  const zonalSums = resources.createBuffer('zonal-sums', featureCount * 4);
  const zonalMeans = resources.createBuffer('zonal-means', featureCount * 4);
  const zonalMinima = resources.createBuffer('zonal-minima', featureCount * 4);
  const zonalMaxima = resources.createBuffer('zonal-maxima', featureCount * 4);
  const zonalDensities = resources.createBuffer('zonal-densities', featureCount * 4);
  const zonalAreas = resources.createBuffer('zonal-areas', featureCount * 4);
  const zonalExtent = resources.createBuffer('zonal-extent', 8);
  const zonalOverflow = resources.createBuffer('zonal-overflow', 4);
  const zonalUncertain = resources.createBuffer('zonal-uncertain', 4);

  const recipeCounts = resources.createBuffer('recipe-counts', featureCount * 4);
  const recipeValues = resources.createBuffer('recipe-values', featureCount * 4);
  const recipeOverflow = resources.createBuffer('recipe-overflow', 4);
  const recipeBreaks = resources.createBuffer('recipe-breaks', (MAXIMUM_CLASSES + 1) * 4);
  const recipeClassCount = resources.createBuffer('recipe-class-count', 4);
  const recipeColors = resources.createBuffer('recipe-colors', featureCount * 4);
  const classBreaksParameters = resources.createParameterBuffer(
    'class-breaks-parameters',
    'float32',
    8 + MAXIMUM_CLASSES + 1
  );
  const colorScaleParameters = resources.createParameterBuffer(
    'color-scale-parameters',
    'float32',
    16
  );
  const palette = resources.createBuffer('palette', MAXIMUM_CLASSES * 4);

  // --- state -----------------------------------------------------------------------------------
  let destroyed = false;
  let assign: AssignGraph;
  let zonal: ZonalGraph;
  let recipe: RecipeGraph | null = null;
  const zonalCache = new Map<string, ZonalGraph>();
  const recipeCache = new Map<string, RecipeGraph>();
  const assignCache = new Map<string, AssignGraph>();
  let dirty = true;
  let valuesDirty = true;
  let settle = 0;
  let encodings = 0;
  let measuring = false;
  let statisticValues = new Float32Array(featureCount);
  let tractCounts = new Uint32Array(featureCount);
  let legendScale = 1;
  let displayedClasses = 0;

  const getAssignKey = (
    state: Pick<ObservationsByTractOptions, 'prepared' | 'spatialSort' | 'includeBoundary'>
  ) =>
    `${state.prepared ? 'prepared' : 'plain'}|${state.spatialSort ? 'sorted' : 'unsorted'}|${state.includeBoundary ? 'boundary' : 'open'}`;
  const getZonalKey = (state: ObservationsByTractOptions) =>
    `${state.statistic}|${state.sumOrder}|${state.includeBoundary ? 'boundary' : 'open'}|${state.statistic === 'density' ? state.denominator : '-'}`;
  const getRecipeStatistic = (state: ObservationsByTractOptions) =>
    state.statistic === 'density' ? 'count' : state.statistic;
  const getRecipeKey = (state: ObservationsByTractOptions) =>
    `${state.backend}|${getRecipeStatistic(state)}|${state.includeBoundary ? 'boundary' : 'open'}`;

  /** Builds the assignment join (and its prepared handle) for the compile-time choices. */
  function buildAssignGraph(
    state: Pick<ObservationsByTractOptions, 'prepared' | 'spatialSort' | 'includeBoundary'>,
    idSuffix = ''
  ): AssignGraph {
    const graph = new GPUCommandGraph<void>(device, {id: `observations-assign${idSuffix}`});
    const views = importPolygons(graph, 'tracts', polygons);
    const prepared = state.prepared
      ? new GPUSpatialJoinPrepared({
          id: 'tract-index',
          geometry: views,
          spatialSort: state.spatialSort
        })
      : null;
    if (prepared) graph.add(prepared);
    graph.add(
      new GPUPointInPolygonJoin({
        id: 'tract-join',
        points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        candidateCapacity: pointCount * CANDIDATES_PER_POINT,
        includeBoundary: state.includeBoundary,
        ...(prepared ? {prepared} : {spatialSort: state.spatialSort}),
        pointFeatureIds: importGraphBuffer(
          graph,
          'point-features',
          pointFeatures,
          'uint32',
          pointCount
        ),
        featureCounts: importGraphBuffer(
          graph,
          'assign-counts',
          assignCounts,
          'uint32',
          featureCount
        ),
        overflow: importGraphBuffer(graph, 'assign-overflow', assignOverflow, 'uint32', 1),
        candidateCount: importGraphBuffer(
          graph,
          'assign-candidates',
          assignCandidates,
          'uint32',
          1
        ),
        uncertainCount: importGraphBuffer(graph, 'assign-uncertain', assignUncertain, 'uint32', 1)
      })
    );
    return {
      key: getAssignKey(state),
      compiled: graph.compile(),
      prepared
    };
  }

  function getAssignGraph(state: ObservationsByTractOptions): AssignGraph {
    const key = getAssignKey(state);
    let built = assignCache.get(key);
    if (!built) {
      built = buildAssignGraph(state);
      assignCache.set(key, built);
      // Destroy the compiled graph before the prepared handle whose buffers it imports.
      const handle = built;
      if (handle.prepared) resources.track({destroy: () => handle.prepared?.destroy()});
      resources.track(handle.compiled);
    }
    return built;
  }

  /** Builds the zonal reduction: the statistic selects which outputs exist (compile-time). */
  function getZonalGraph(state: ObservationsByTractOptions): ZonalGraph {
    const key = getZonalKey(state);
    let built = zonalCache.get(key);
    if (built) return built;
    const {statistic} = state;
    const graph = new GPUCommandGraph<void>(device, {id: `observations-zonal-${statistic}`});
    const views = importPolygons(graph, 'tracts', polygons);
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      buffer: Buffer,
      format: Format
    ) => importGraphBuffer(graph, name, buffer, format, featureCount);
    const usesValues = statistic !== 'count' && statistic !== 'density';
    const output: ConstructorParameters<typeof GPUZonalStatistics>[0]['output'] = {
      counts: view('counts', zonalCounts, 'uint32'),
      extent: importGraphBuffer(graph, 'extent', zonalExtent, 'float32', 2),
      extentStatistic: (statistic === 'count'
        ? 'count'
        : statistic === 'density'
          ? 'density'
          : statistic === 'sum'
            ? 'sum'
            : statistic === 'mean'
              ? 'mean'
              : statistic) as GPUZonalStatisticsExtentStatistic,
      overflow: importGraphBuffer(graph, 'zonal-overflow', zonalOverflow, 'uint32', 1),
      uncertainCount: importGraphBuffer(graph, 'zonal-uncertain', zonalUncertain, 'uint32', 1)
    };
    let statisticBuffer = zonalCounts;
    let statisticFormat: 'uint32' | 'float32' = 'uint32';
    if (statistic === 'sum') {
      output.sums = view('sums', zonalSums, 'float32');
      statisticBuffer = zonalSums;
      statisticFormat = 'float32';
    } else if (statistic === 'mean') {
      output.means = view('means', zonalMeans, 'float32');
      statisticBuffer = zonalMeans;
      statisticFormat = 'float32';
    } else if (statistic === 'minimum') {
      output.minima = view('minima', zonalMinima, 'float32');
      statisticBuffer = zonalMinima;
      statisticFormat = 'float32';
    } else if (statistic === 'maximum') {
      output.maxima = view('maxima', zonalMaxima, 'float32');
      statisticBuffer = zonalMaxima;
      statisticFormat = 'float32';
    } else if (statistic === 'density') {
      output.densities = view('densities', zonalDensities, 'float32');
      statisticBuffer = zonalDensities;
      statisticFormat = 'float32';
      // The GPU polygon area is only produced when the caller supplies no areas.
      if (state.denominator === 'area')
        output.featureAreas = view('areas-out', zonalAreas, 'float32');
    }
    graph.add(
      new GPUZonalStatistics({
        id: 'tract-zonal',
        features: {
          kind: 'polygons',
          polygonPositions: views.positions,
          featureOffsets: views.featureOffsets,
          polygonOffsets: views.polygonOffsets,
          ringOffsets: views.ringOffsets,
          candidateCapacity: pointCount * CANDIDATES_PER_POINT,
          includeBoundary: state.includeBoundary
        },
        points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
        ...(usesValues
          ? {values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount)}
          : {}),
        ...(statistic === 'density' && state.denominator === 'population'
          ? {areas: view('areas-in', areasBuffer, 'float32')}
          : {}),
        sumOrder: state.sumOrder,
        output
      })
    );
    built = {
      key,
      compiled: resources.track(graph.compile()),
      statistic: statisticBuffer,
      statisticFormat
    };
    zonalCache.set(key, built);
    return built;
  }

  /** Builds the choropleth recipe: zonal or join + group statistics, class breaks, color scale. */
  function getRecipeGraph(state: ObservationsByTractOptions): RecipeGraph {
    const key = getRecipeKey(state);
    let built = recipeCache.get(key);
    if (built) return built;
    const statistic = getRecipeStatistic(state);
    const graph = new GPUCommandGraph<void>(device, {id: `observations-recipe-${state.backend}`});
    const views = importPolygons(graph, 'tracts', polygons);
    recipeResultInfo = addPointsInPolygonsChoroplethRecipe(graph, {
      id: 'tract-choropleth',
      backend: state.backend,
      points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
      ...(statistic === 'count'
        ? {}
        : {values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount)}),
      statistic,
      polygons: {
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        candidateCapacity: pointCount * CANDIDATES_PER_POINT,
        includeBoundary: state.includeBoundary
      },
      outputs: {
        counts: importGraphBuffer(graph, 'recipe-counts', recipeCounts, 'uint32', featureCount),
        ...(statistic === 'count'
          ? {}
          : {
              featureValues: importGraphBuffer(
                graph,
                'recipe-values',
                recipeValues,
                'float32',
                featureCount
              )
            }),
        overflow: importGraphBuffer(graph, 'recipe-overflow', recipeOverflow, 'uint32', 1),
        color: {
          breaks: importGraphBuffer(
            graph,
            'recipe-breaks',
            recipeBreaks,
            'float32',
            MAXIMUM_CLASSES + 1
          ),
          classCount: importGraphBuffer(graph, 'recipe-class-count', recipeClassCount, 'uint32', 1),
          colors: importGraphBuffer(graph, 'recipe-colors', recipeColors, 'uint32', featureCount)
        }
      },
      color: {
        classBreaksParameters: classBreaksParameters.importToGraph(graph),
        maximumClassCount: MAXIMUM_CLASSES,
        methods: [...CLASS_METHODS],
        colorScaleParameters: colorScaleParameters.importToGraph(graph),
        palette: importGraphBuffer(graph, 'palette', palette, 'uint32', MAXIMUM_CLASSES),
        maximumPaletteCount: MAXIMUM_CLASSES
      }
    });
    built = {key, compiled: resources.track(graph.compile()), usedClasses: 0};
    recipeCache.set(key, built);
    return built;
  }
  let recipeResultInfo: {contributors: readonly unknown[]} | null = null;

  // --- summaries -------------------------------------------------------------------------------
  const statisticReader = new SummaryReader(
    resources,
    'observations-statistic',
    [
      {buffer: zonalCounts, size: featureCount * 4},
      {buffer: zonalExtent, size: 8},
      {buffer: zonalOverflow, size: 4},
      {buffer: zonalUncertain, size: 4}
    ],
    () => undefined
  );
  statisticReader.stop();
  // The statistic buffer changes with the compiled variant, so the reader is rebuilt per variant.
  let activeReader: SummaryReader | null = null;
  const makeStatisticReader = (graph: ZonalGraph): SummaryReader =>
    new SummaryReader(
      resources,
      `observations-statistic-${graph.key}`,
      [
        {buffer: graph.statistic, size: featureCount * 4},
        {buffer: zonalCounts, size: featureCount * 4},
        {buffer: zonalExtent, size: 8},
        {buffer: zonalOverflow, size: 4},
        {buffer: zonalUncertain, size: 4},
        {buffer: zonalAreas, size: featureCount * 4}
      ],
      bytes => {
        if (destroyed || zonal !== graph) return;
        const statistics =
          graph.statisticFormat === 'uint32'
            ? Float32Array.from(new Uint32Array(bytes, 0, featureCount))
            : new Float32Array(bytes.slice(0, featureCount * 4));
        statisticValues = statistics;
        tractCounts = new Uint32Array(bytes.slice(featureCount * 4, featureCount * 8));
        const tail = new Float32Array(bytes.slice(featureCount * 8, featureCount * 8 + 8));
        const flags = new Uint32Array(bytes.slice(featureCount * 8 + 8, featureCount * 8 + 16));
        const gpuAreas = new Float32Array(
          bytes.slice(featureCount * 8 + 16, featureCount * 12 + 16)
        );
        ctx.setLegendExtent('statistic', [tail[0] * legendScale, tail[1] * legendScale]);
        ctx.setReadout('zonalOverflow', flags[0] ? 'YES (capacity)' : 'no');
        ctx.setReadout('zonalUncertain', formatInteger(flags[1]));
        updateReadoutsFromStatistics(gpuAreas);
      }
    );

  let assignReader: SummaryReader | null = null;
  const assignSummary = new SummaryReader(
    resources,
    'observations-assign',
    [
      {buffer: pointFeatures, size: pointCount * 4},
      {buffer: assignCounts, size: featureCount * 4},
      {buffer: assignOverflow, size: 4},
      {buffer: assignCandidates, size: 4},
      {buffer: assignUncertain, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const ids = new Uint32Array(bytes, 0, pointCount);
      let joined = 0;
      let agree = 0;
      let outside = 0;
      let differ = 0;
      let differNearEdge = 0;
      for (let index = 0; index < pointCount; index++) {
        const id = ids[index];
        if (id === GPU_SPATIAL_JOIN_NO_FEATURE) outside++;
        else joined++;
        const reference = observationTract[index];
        const referenceRow = reference === 65535 ? GPU_SPATIAL_JOIN_NO_FEATURE : reference;
        if (referenceRow === id) {
          agree++;
        } else {
          differ++;
          const [x, y] = [positions[index * 2], positions[index * 2 + 1]];
          if (edgeIndex.getDistance(x, y) <= 2) differNearEdge++;
        }
      }
      const tail = new Uint32Array(bytes.slice(pointCount * 4 + featureCount * 4));
      ctx.setReadout('joined', `${formatInteger(joined)} of ${formatInteger(pointCount)}`);
      ctx.setReadout(
        'outside',
        `${formatInteger(outside)} (${((100 * outside) / pointCount).toFixed(2)}%)`
      );
      ctx.setReadout(
        'agreement',
        `${((100 * agree) / pointCount).toFixed(2)}% (${formatInteger(differ)} differ)`
      );
      ctx.setReadout(
        'differenceNearEdge',
        differ === 0
          ? 'no differences'
          : `${((100 * differNearEdge) / differ).toFixed(1)}% of the ${formatInteger(differ)} differences lie within 2 m of a tract edge`
      );
      ctx.setReadout('joinOverflow', tail[0] ? 'YES (capacity)' : 'no');
      ctx.setReadout('candidates', formatInteger(tail[1]));
      ctx.setReadout('joinUncertain', formatInteger(tail[2]));
      void assignReader;
    }
  );

  const recipeReader = new SummaryReader(
    resources,
    'observations-recipe',
    [
      {buffer: recipeBreaks, size: (MAXIMUM_CLASSES + 1) * 4},
      {buffer: recipeClassCount, size: 4},
      {buffer: recipeOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const edges = new Float32Array(bytes.slice(0, (MAXIMUM_CLASSES + 1) * 4));
      const usedClasses = new Uint32Array(
        bytes.slice((MAXIMUM_CLASSES + 1) * 4, (MAXIMUM_CLASSES + 2) * 4)
      )[0];
      displayedClasses = usedClasses;
      ctx.setReadout(
        'classBreaks',
        Array.from(edges.subarray(0, usedClasses + 1), formatCompact).join(' | ')
      );
      ctx.setReadout('classesUsed', `${usedClasses} (${ctx.options.classMethod})`);
    }
  );

  /** Fills the per-tract readouts that need the CPU copy of the statistic. */
  function updateReadoutsFromStatistics(gpuAreas: Float32Array): void {
    let hottest = -1;
    let hottestValue = -Infinity;
    let gpuAreaTotal = 0;
    let dataAreaTotal = 0;
    for (let row = 0; row < featureCount; row++) {
      const value = statisticValues[row];
      if (Number.isFinite(value) && value > hottestValue) {
        hottestValue = value;
        hottest = row;
      }
      if (Number.isFinite(gpuAreas[row])) gpuAreaTotal += gpuAreas[row];
      dataAreaTotal += areaKm2[row];
    }
    const options = ctx.options;
    const scale = options.statistic === 'density' && options.denominator === 'area' ? 1e6 : 1;
    ctx.setReadout(
      'top',
      hottest >= 0
        ? `${geoids[hottest]} (${areaNames[tractCommunityArea[hottest] - 1] ?? 'no community area'}): ${formatCompact(hottestValue * scale)}${options.statistic === 'count' ? '' : ` · N=${formatInteger(tractCounts[hottest])}`}`
        : 'n/a'
    );
    let observationsTotal = 0;
    let emptyTracts = 0;
    for (let row = 0; row < featureCount; row++) {
      observationsTotal += tractCounts[row];
      if (tractCounts[row] === 0) emptyTracts++;
    }
    const cityDensity = observationsTotal / dataAreaTotal;
    let crowdedTracts = 0;
    for (let row = 0; row < featureCount; row++) {
      if (areaKm2[row] > 0 && tractCounts[row] / areaKm2[row] > 10 * cityDensity) crowdedTracts++;
    }
    ctx.setReadout('cityRate', `${formatCompact(cityDensity)} observations per km2`);
    ctx.setReadout(
      'unstableRates',
      `${emptyTracts} tracts have no observation; ${crowdedTracts} have over 10x the city density`
    );
    if (options.statistic === 'density' && options.denominator === 'area' && gpuAreaTotal > 0) {
      ctx.setReadout(
        'areaCheck',
        `${formatCompact(gpuAreaTotal / 1e6)} km2 from the GPU vs ${formatCompact(dataAreaTotal)} km2 in the data`
      );
    } else {
      ctx.setReadout('areaCheck', 'GPU polygon area is used when the denominator is area');
    }
    // Parity with the CPU pipeline that built the tract table.
    let absoluteDifference = 0;
    for (let row = 0; row < featureCount; row++) {
      absoluteDifference += Math.abs(tractCounts[row] - dataObservationCounts[row]);
    }
    ctx.setReadout(
      'tractParity',
      absoluteDifference === 0
        ? 'exact: every tract matches the GeoPandas table'
        : `${formatInteger(absoluteDifference)} observations differ from the GeoPandas table`
    );
  }

  // --- parameter writes ------------------------------------------------------------------------
  function writeValues(state: ObservationsByTractOptions): void {
    const groupIndex = categoryNames.indexOf(state.groupType);
    for (let index = 0; index < pointCount; index++) {
      let value = 0;
      switch (state.valueKind) {
        case 'researchGrade':
          value = researchGrade[index] ? 1 : 0;
          break;
        case 'introduced':
          value = introduced[index] ? 1 : 0;
          break;
        case 'animal':
          value = isAnimalGroup[categories[index]] ? 1 : 0;
          break;
        case 'weekend':
          value = isWeekend(timestamps[index]) ? 1 : 0;
          break;
        case 'category':
          value = categories[index] === groupIndex ? 1 : 0;
          break;
      }
      values[index] = value;
    }
    valuesBuffer.write(values);
    valuesDirty = false;
  }

  function writeAreas(): void {
    const areas = new Float32Array(featureCount);
    for (let row = 0; row < featureCount; row++) {
      // Per 1,000 residents: counts / (population / 1000). Zero population gives no data.
      areas[row] = population[row] > 0 ? population[row] / 1000 : 0;
    }
    areasBuffer.write(areas);
  }

  function writeClassification(state: ObservationsByTractOptions): void {
    const classCount = state.classMethod === 'box-plot' ? 6 : state.classCount;
    classBreaksParameters.write(
      getGPUClassBreaksParameterValues(
        {method: state.classMethod as (typeof CLASS_METHODS)[number], classCount},
        MAXIMUM_CLASSES
      )
    );
    colorScaleParameters.write(
      getGPUColorScaleParameterValues({
        scale: 'quantile',
        domainCount: classCount + 1,
        paletteCount: classCount,
        noDataColor: packGPUColor(0, 0, 0, 0)
      })
    );
    palette.write(getStopsPalette(RAMP_STOPS[state.ramp], classCount));
    ctx.setReadout('classesUsed', `${classCount} requested (${state.classMethod})`);
  }

  // --- initial build ---------------------------------------------------------------------------
  const initial = ctx.options;
  writeValues(initial);
  writeAreas();
  writeClassification(initial);
  assign = getAssignGraph(initial);
  zonal = getZonalGraph(initial);
  activeReader = makeStatisticReader(zonal);
  if (initial.display === 'classes') recipe = getRecipeGraph(initial);
  ctx.setReadout('points', pointCount);
  ctx.setReadout('tracts', featureCount);
  ctx.setReadout('zoneRaster', `${zoneRaster.width} x ${zoneRaster.height} cells (fill only)`);

  const updateScale = (state: ObservationsByTractOptions) => {
    legendScale = state.statistic === 'density' && state.denominator === 'area' ? 1e6 : 1;
  };
  updateScale(initial);

  function currentGraphs(state: ObservationsByTractOptions): CompiledGPUCommandGraph<never>[] {
    const graphs = [assign.compiled, zonal.compiled];
    if (state.display === 'classes' && recipe) graphs.push(recipe.compiled);
    return graphs as CompiledGPUCommandGraph<never>[];
  }

  function releaseLater(release: () => void): void {
    // Deck may still hold the previous frame; free two frames later.
    requestAnimationFrame(() => requestAnimationFrame(() => !destroyed && release()));
  }
  void releaseLater;

  /** Times the three assignment variants outside the frame. */
  async function measureJoins(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timePlain', 'measuring...');
    ctx.setReadout('timeSorted', 'measuring...');
    ctx.setReadout('timePrepared', 'measuring...');
    const variants = [
      {id: 'timePlain', state: {prepared: false, spatialSort: false}},
      {id: 'timeSorted', state: {prepared: false, spatialSort: true}},
      {id: 'timePrepared', state: {prepared: true, spatialSort: true}}
    ] as const;
    try {
      for (const variant of variants) {
        const built = buildAssignGraph(
          {...variant.state, includeBoundary: ctx.options.includeBoundary},
          `-time-${variant.id}`
        );
        try {
          const timing = await measureCompiledGraph(device, built.compiled, {
            parameters: undefined,
            completionBuffer: assignOverflow,
            signal: ctx.signal
          });
          if (destroyed) return;
          const builds = built.prepared
            ? ` (index built ${built.prepared.encodedBuildCount}x)`
            : '';
          ctx.setReadout(variant.id, `${formatCompiledGraphTiming(timing)}${builds}`);
        } finally {
          built.compiled.destroy();
          built.prepared?.destroy();
        }
      }
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  return {
    getCompiledGraphs: () => currentGraphs(ctx.options),

    setOption(id, _value, state) {
      switch (id) {
        case 'includeBoundary':
        case 'prepared':
        case 'spatialSort':
          assign = getAssignGraph(state);
          zonal = getZonalGraph(state);
          activeReader?.stop();
          activeReader = makeStatisticReader(zonal);
          if (state.display === 'classes') recipe = getRecipeGraph(state);
          dirty = true;
          break;
        case 'statistic':
        case 'sumOrder':
        case 'denominator':
          zonal = getZonalGraph(state);
          activeReader?.stop();
          activeReader = makeStatisticReader(zonal);
          if (state.display === 'classes') recipe = getRecipeGraph(state);
          updateScale(state);
          dirty = true;
          break;
        case 'valueKind':
        case 'groupType':
          valuesDirty = true;
          dirty = true;
          break;
        case 'display':
        case 'backend':
          if (state.display === 'classes') recipe = getRecipeGraph(state);
          writeClassification(state);
          dirty = true;
          break;
        case 'classMethod':
        case 'classCount':
          writeClassification(state);
          dirty = true;
          break;
        case 'ramp':
          writeClassification(state);
          break;
        default:
          break;
      }
      if (valuesDirty) writeValues(state);
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureJoins();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const state = ctx.options;
      if (state.invalidateEveryFrame) assign.prepared?.invalidate();
      const run = dirty || state.rerunEveryFrame;
      if (run) {
        assign.compiled.encode(commandEncoder, {parameters: undefined});
        zonal.compiled.encode(commandEncoder, {parameters: undefined});
        if (state.display === 'classes' && recipe) {
          recipe.compiled.encode(commandEncoder, {parameters: undefined});
        }
        encodings++;
        dirty = false;
        settle = SETTLE_FRAMES;
      }
      if (state.rerunEveryFrame && run && encodings % 30 === 0) {
        // Streaming mode never settles: refresh the summaries a few times per second instead.
        activeReader?.request(commandEncoder);
        assignSummary.request(commandEncoder);
        if (state.display === 'classes') recipeReader.request(commandEncoder);
      }
      if (settle > 0 && !run) {
        settle--;
        if (settle === 0) {
          activeReader?.request(commandEncoder);
          assignSummary.request(commandEncoder);
          if (state.display === 'classes') recipeReader.request(commandEncoder);
        }
      }
      activeReader?.flush(commandEncoder);
      assignSummary.flush(commandEncoder);
      recipeReader.flush(commandEncoder);
      if (state.prepared && assign.prepared) {
        ctx.setReadout(
          'indexBuilds',
          `${assign.prepared.encodedBuildCount} build${assign.prepared.encodedBuildCount === 1 ? '' : 's'} in ${formatInteger(encodings)} join runs`
        );
      } else {
        ctx.setReadout('indexBuilds', `index rebuilt in all ${formatInteger(encodings)} join runs`);
      }
    },

    getLayers() {
      const state = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (state.display === 'classes') {
        layers.push(
          new PackedZoneFillLayer({
            id: 'tract-classes',
            coordinateOrigin,
            gridSize: [zoneRaster.width, zoneRaster.height],
            bounds: zoneRaster.bounds,
            rowOrigin: 'south',
            valueIndices: zoneRaster.zones,
            values: recipeColors,
            valueFormat: 'uint32',
            opacity: state.opacity
          })
        );
      } else {
        const scale = state.statistic === 'density' && state.denominator === 'area' ? 1e6 : 1;
        layers.push(
          new ZoneFillLayer({
            id: 'tract-fill',
            coordinateOrigin,
            gridSize: [zoneRaster.width, zoneRaster.height],
            bounds: zoneRaster.bounds,
            rowOrigin: 'south',
            valueIndices: zoneRaster.zones,
            values: zonal.statistic,
            valueFormat: zonal.statisticFormat,
            extent: zonalExtent,
            valueScale: scale,
            colormap: state.ramp as SpatialAnalysisColormap,
            sqrtScale: state.sqrtScale,
            noDataColor: [128, 128, 128, 70],
            opacity: state.opacity
          })
        );
      }
      if (state.outlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'tract-outline',
            coordinateOrigin,
            segments: polygons.outline,
            instanceCount: tractSet.outline.length / 4,
            color: ctx.theme() === 'dark' ? [230, 235, 245, 120] : [30, 40, 60, 130],
            widthPixels: 0.8
          })
        );
      }
      if (state.pointsMode !== 'off') {
        const outsideOnly = state.pointsMode === 'outside';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'observation-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: pointFeatures,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: outsideOnly
              ? TRACT_PALETTE.map(color => [color[0], color[1], color[2], 0] as const)
              : TRACT_PALETTE,
            noDataValue: GPU_SPATIAL_JOIN_NO_FEATURE,
            noDataColor: outsideOnly ? [255, 40, 40, 255] : [150, 150, 150, 200],
            radiusPixels: outsideOnly ? 3 : 1.3,
            opacity: outsideOnly ? 1 : 0.9
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const row = findPolygonAt(tractSet, x, y);
      if (row < 0) return null;
      const options = ctx.options;
      const scale = options.statistic === 'density' && options.denominator === 'area' ? 1e6 : 1;
      const value = statisticValues[row];
      const label = {
        count: 'observations',
        sum: 'sum of value',
        mean: 'mean of value',
        minimum: 'minimum',
        maximum: 'maximum',
        density:
          options.denominator === 'area'
            ? 'observations per km2'
            : 'observations per 1,000 residents'
      }[options.statistic];
      const area = areaNames[tractCommunityArea[row] - 1];
      return [
        `Tract ${geoids[row]}${area ? ` (${area})` : ''}`,
        `${formatInteger(tractCounts[row])} observations, ${formatInteger(population[row])} residents`,
        `${label}: ${Number.isFinite(value) ? formatCompact(value * scale) : 'no data'}`
      ].join('\n');
    },

    destroy() {
      destroyed = true;
      activeReader?.stop();
      assignSummary.stop();
      recipeReader.stop();
      void recipeResultInfo;
      void displayedClasses;
      void readUint32;
      resources.destroy();
    }
  };
}
