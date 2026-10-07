// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  GPUGroupStatistics,
  GPUKeyJoin,
  type GPUKeyJoinKind
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassCounts, getClassIndex} from '../../cartography/breaks';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatOrdinal, liveText} from '../../cartography/live-text';
import {createFeatureLocator, getGeometryPolygons} from '../../cartography/picking';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {getSizeLegendEntries} from '../../cartography/proportional';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {getSortedOrder} from '../../engine/draw-order';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {ChartColor, ChartData} from '../chart-types';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {
  DERIVED,
  GROUP_METRICS,
  PERCENTILE_COUNT,
  SLOT,
  SLOT_COUNT,
  ZONE_CAPACITY,
  ZONE_COUNT,
  type MetricDefinition
} from './b5-group-metrics';
import {createGraphImporter} from './b5-classify';
import {createByteReader} from './b5-common';
import {SlotSlices} from './b5-slots';
import {
  formatClockHour,
  getDescendingOrder,
  getFrozenBreaks,
  getHourCounts,
  getMedian,
  getPercentileRank,
  getReferenceTable,
  getSpearmanCorrelation,
  NO_KEY,
  type Classification,
  type FrozenBreaks,
  type ReferenceTable,
  type RowFilter
} from './group-statistics.stats';
import {
  COUNT_RADIUS_MINIMUM_PIXELS,
  COUNT_RADIUS_PIXELS,
  FILL_OPACITY,
  getAreaLinePasses,
  getCasingColor,
  getCircleColor,
  getDotColor,
  getHatchColor,
  getHourClass,
  getHourColors,
  getInkColor,
  getMetricTable,
  getMutedSectorColor,
  getSuppressedBarColor,
  getTractHairlineColor,
  getTractRateTable,
  getUnfilledAreaColor,
  RATE_BASIS,
  TRACT_RATE_BREAKS
} from './group-statistics.style';

/** Option state of the group-statistics scene. */
export type GroupStatisticsOptions = {
  /** Circles sized by the count, or a fill coloured by the rate. */
  view: 'counts' | 'rates';
  metric: string;
  classification: Classification;
  /** Community areas, census tracts, or both with a swipe divider. */
  zoning: 'areas' | 'tracts' | 'swipe';
  natureGroup: string;
  hours: readonly [number, number];
  gradeFilter: 'all' | 'research' | 'unconfirmed';
  variance: 'sample' | 'population';
  lowerFraction: number;
  upperFraction: number;
  joinKind: GPUKeyJoinKind;
  minimumObservations: number;
};

/** Data shared with `legends(state, data)` through `ctx.setLegendData`. */
export type GroupStatisticsLegend = {
  /** The class table the map draws now (the area metric, or the tract rate table). */
  table: ClassTable;
  /** Features per class of `table`; areas, or tracts when the map draws tracts. */
  counts: number[];
  /** Nested-circle legend entries, largest first. */
  circleEntries: {radiusPixels: number; label: string}[];
  circleColor: readonly [number, number, number, number];
  hourColors: readonly (readonly [number, number, number, number])[];
  noDataColor: readonly [number, number, number, number];
  /** Areas withheld by the minimum-observations rule. */
  withheldAreas: number;
};

const FRAME_ORIGIN_ELEVATION = 0;
const DESCRIBED_NOTE =
  'Filters, thresholds and statistics are parameter writes; only the variance and join kinds rebuild.';

/** Metric parameter kind codes of the display kernel. */
const KERNEL_KIND = {slot: 0, unsigned: 1, percentile: 2, derived: 3} as const;

/**
 * Nature observations grouped by community area with `GPUGroupStatistics` (dense keys: counts,
 * hour statistics, grade and introduced shares, distinct taxa) and `GPUKeyJoin` (a 1:n join sums
 * tract population and income-times-population onto the areas, a gather join paints the area value
 * back on the tracts). A GPU kernel builds the row mask from the options each frame, a second
 * kernel picks the displayed statistic and withholds small groups.
 *
 * The maps are drawn by `SpatialAnalysisPolygonLayer` straight from the contributor buffers, with
 * one frozen class table per metric (breaks come from the unfiltered table once), proportional
 * circles for counts, a hatched "withheld" class and a swipe between areas and tracts.
 */
export async function createGroupStatistics(
  ctx: SceneContext<GroupStatisticsOptions>
): Promise<SceneInstance<GroupStatisticsOptions>> {
  const observations = ctx.datasets.get('chicago-nature');
  const areas = ctx.datasets.get('chicago-community-areas');
  const tracts = ctx.datasets.get('chicago-tracts');
  const areaGeojson = areas.geojson;
  const tractGeojson = tracts.geojson;
  if (!areaGeojson || !tractGeojson)
    throw new Error('group-statistics needs the area and tract polygons');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'group-statistics');
  const areaFeatures = areaGeojson.features;
  const tractFeatures = tractGeojson.features;
  const areaName = (zone: number) =>
    String(areaFeatures[zone]?.properties?.name ?? `Area ${zone + 1}`);
  const categoryNames = observations.categories('category');

  // -------------------------------------------------------------------------------------------
  // Columns
  // -------------------------------------------------------------------------------------------

  const observationCount = observations.count;
  const timestamps = observations.column<Uint32Array>('timestamp');
  const categoryColumn = observations.column<Uint8Array>('category');
  const gradeColumn = observations.column<Uint8Array>('researchGrade');
  const introducedColumn = observations.column<Uint8Array>('introduced');
  const speciesColumn = observations.column<Uint32Array>('species');
  const areaColumn = observations.column<Uint8Array>('communityArea');
  const hourHost = new Float32Array(observationCount);
  const categoryHost = new Float32Array(observationCount);
  const speciesHost = new Float32Array(observationCount);
  const gradeHost = new Float32Array(observationCount);
  const introducedHost = new Float32Array(observationCount);
  const keyHost = new Uint32Array(observationCount);
  for (let row = 0; row < observationCount; row++) {
    hourHost[row] = Math.floor((timestamps[row] % 86400) / 3600);
    categoryHost[row] = categoryColumn[row];
    speciesHost[row] = speciesColumn[row];
    gradeHost[row] = gradeColumn[row];
    introducedHost[row] = introducedColumn[row];
    keyHost[row] = areaColumn[row] >= 1 ? areaColumn[row] - 1 : NO_KEY;
  }

  const tractCount = tractFeatures.length;
  const tractArea = tracts.column<Uint8Array>('communityArea');
  const tractZone = Uint32Array.from(tractArea, area => (area >= 1 ? area - 1 : NO_KEY));
  const populationColumn = tracts.column<Float32Array>('population');
  const incomeColumn = tracts.column<Float32Array>('perCapitaIncome');
  const tractObservationColumn = tracts.column<Float32Array>('natureObs2023');
  // Income is joined as income x residents (in thousands of dollars, so float32 sums stay exact)
  // and as the residents whose income is known: the area mean is then weighted by people.
  const incomeProductHost = new Float32Array(tractCount);
  const knownPopulationHost = new Float32Array(tractCount);
  const tractRateHost = new Float32Array(tractCount);
  for (let tract = 0; tract < tractCount; tract++) {
    const residents = populationColumn[tract];
    const income = incomeColumn[tract];
    if (Number.isFinite(income) && residents > 0) {
      incomeProductHost[tract] = (income / 1000) * residents;
      knownPopulationHost[tract] = residents;
    }
    tractRateHost[tract] =
      residents > 0 ? (tractObservationColumn[tract] * 1000) / residents : Number.NaN;
  }
  const areaKeys = Uint32Array.from({length: ZONE_COUNT}, (_, index) => index);

  // The unfiltered table freezes the class breaks once; nothing on the map reads it.
  const reference: ReferenceTable = getReferenceTable({
    areaCount: ZONE_COUNT,
    observationArea: areaColumn,
    researchGrade: gradeColumn,
    introduced: introducedColumn,
    species: speciesColumn,
    tractArea,
    population: populationColumn,
    perCapitaIncome: incomeColumn
  });
  const frozenBreaks: FrozenBreaks = getFrozenBreaks(reference);
  const fullMaximumCount = Math.max(...reference.counts);
  const tractClassCounts = getClassCounts(tractRateHost, TRACT_RATE_BREAKS);
  const hourColumns = {hour: hourHost, category: categoryHost, grade: gradeHost, key: keyHost};

  // -------------------------------------------------------------------------------------------
  // Geometry: planar-metre meshes for the fills, outline segments for boundaries and selection
  // -------------------------------------------------------------------------------------------

  const areaOrigin = areas.defaultOrigin;
  const areaProjection = areas.getProjection(areaOrigin);
  const areaMesh = buildPolygonMesh(areaGeojson, (lng, lat) => areaProjection.project(lng, lat));
  const areaPolygons = createPolygonMeshBuffers(resources, areaMesh, 'areas');
  const areaLayerOrigin: [number, number, number] = [
    areaOrigin[0],
    areaOrigin[1],
    FRAME_ORIGIN_ELEVATION
  ];
  const tractOrigin = tracts.defaultOrigin;
  const tractProjection = tracts.getProjection(tractOrigin);
  const tractMesh = buildPolygonMesh(tractGeojson, (lng, lat) => tractProjection.project(lng, lat));
  const tractPolygons = createPolygonMeshBuffers(resources, tractMesh, 'tracts');
  const tractLayerOrigin: [number, number, number] = [
    tractOrigin[0],
    tractOrigin[1],
    FRAME_ORIGIN_ELEVATION
  ];
  const areaLocator = createFeatureLocator(areaGeojson);
  const tractLocator = createFeatureLocator(tractGeojson);

  const outlineStart = new Uint32Array(ZONE_COUNT + 1);
  for (const feature of areaMesh.outlineFeatures) outlineStart[feature + 1]++;
  for (let zone = 0; zone < ZONE_COUNT; zone++) outlineStart[zone + 1] += outlineStart[zone];
  const getAreaSegments = (zone: number): Float32Array =>
    areaMesh.outlineSegments.slice(outlineStart[zone] * 4, outlineStart[zone + 1] * 4);
  const selectionBuffer = resources.createBuffer('selection', areaMesh.outlineSegments.byteLength);
  let selectionCount = 0;

  const getRings = (features: typeof areaFeatures, row: number): LngLat[][] =>
    getGeometryPolygons(features[row]?.geometry).flatMap(polygon =>
      polygon.map(ring => ring.map(point => [point[0], point[1]] as LngLat))
    );
  const getLabelPoint = (zone: number): LngLat => {
    const point = areaMesh.labelPoints[zone];
    return [point[0], point[1]] as LngLat;
  };

  // Proportional circles sit at the area label points, biggest drawn first so small ones stay visible.
  const circlePositions = new Float32Array(ZONE_COUNT * 2);
  for (let zone = 0; zone < ZONE_COUNT; zone++) {
    const [lng, lat] = areaMesh.labelPoints[zone];
    const [x, y] = areaProjection.project(lng, lat);
    circlePositions[zone * 2] = Number.isFinite(x) ? x : 0;
    circlePositions[zone * 2 + 1] = Number.isFinite(y) ? y : 0;
  }
  const circlePositionBuffer = resources.createBuffer('circle-positions', circlePositions);
  const circleOrderBuffer = resources.createBuffer(
    'circle-order',
    getSortedOrder(reference.counts, 'descending')
  );

  // -------------------------------------------------------------------------------------------
  // GPU buffers
  // -------------------------------------------------------------------------------------------

  const hourBuffer = resources.createBuffer('hour', hourHost);
  const categoryBuffer = resources.createBuffer('category', categoryHost);
  const speciesBuffer = resources.createBuffer('species', speciesHost);
  const gradeBuffer = resources.createBuffer('grade', gradeHost);
  const introducedBuffer = resources.createBuffer('introduced', introducedHost);
  const keysBuffer = resources.createBuffer('nature-keys', keyHost);
  const maskBuffer = resources.createBuffer('mask', observationCount * 4);
  const metricsBuffer = resources.createBuffer('metrics', SLOT_COUNT * ZONE_CAPACITY * 4);
  const outputKeysBuffer = resources.createBuffer('output-keys', ZONE_COUNT * 4);
  const outputCountBuffer = resources.createBuffer('output-count', 4);
  const outputOverflowBuffer = resources.createBuffer('output-overflow', 4);
  const maskParameters = resources.createParameterBuffer('mask-parameters', 'uint32', 4);
  const percentileParameters = resources.createParameterBuffer(
    'percentiles',
    'float32',
    PERCENTILE_COUNT
  );
  const metricParameters = resources.createParameterBuffer('metric-parameters', 'uint32', 4);
  const minimumParameters = resources.createParameterBuffer('minimum-parameters', 'uint32', 4);

  const areaKeysBuffer = resources.createBuffer('area-keys', areaKeys);
  const tractZoneBuffer = resources.createBuffer('tract-zone', tractZone);
  const tractPopulation = resources.createBuffer('tract-population', populationColumn);
  const tractIncomeProduct = resources.createBuffer('tract-income-product', incomeProductHost);
  const tractKnownPopulation = resources.createBuffer(
    'tract-known-population',
    knownPopulationHost
  );
  const tractRateBuffer = resources.createBuffer('tract-rate', tractRateHost);
  const populationSumBuffer = resources.createBuffer('area-population', ZONE_COUNT * 4);
  const incomeSumBuffer = resources.createBuffer('area-income-sum', ZONE_COUNT * 4);
  const incomePopulationBuffer = resources.createBuffer('area-income-population', ZONE_COUNT * 4);
  const tractCountBuffer = resources.createBuffer('area-tract-count', ZONE_COUNT * 4);
  const areaMetricBuffer = resources.createBuffer('area-metric', ZONE_COUNT * 4);
  const areaCountBuffer = resources.createBuffer('area-count', ZONE_COUNT * 4);
  const areaKeepBuffer = resources.createBuffer('area-keep', ZONE_COUNT * 4);
  const tractMetricBuffer = resources.createBuffer('tract-metric', tractCount * 4);
  const tractMatchedBuffer = resources.createBuffer('tract-matched', tractCount * 4);
  const innerRowsBuffer = resources.createBuffer('inner-rows', tractCount * 4);
  const innerCountBuffer = resources.createBuffer('inner-count', 4);
  const innerOverflowBuffer = resources.createBuffer('inner-overflow', 4);

  // -------------------------------------------------------------------------------------------
  // The graph: mask, group statistics, join onto areas, display kernel, gather onto tracts
  // -------------------------------------------------------------------------------------------

  type Variant = {key: string; compiled: CompiledGPUCommandGraph<void>};
  const variants = new Map<string, Variant>();

  const compileVariant = (options: GroupStatisticsOptions): Variant => {
    const key = `${options.variance}|${options.joinKind}`;
    const graph = new GPUCommandGraph<void>(device, {id: `group-statistics-${key}`});
    // A graph rejects two imports of one buffer, so each buffer is imported once per graph.
    const bind = createGraphImporter(graph);
    const maskView = bind.word(maskBuffer, observationCount);
    const hour = bind.float(hourBuffer, observationCount);
    const category = bind.float(categoryBuffer, observationCount);
    const grade = bind.float(gradeBuffer, observationCount);

    // The filter mask is built on the GPU from the options (a four-word parameter write).
    addKernelPass(graph, {
      id: 'nature-mask',
      invocationCount: observationCount,
      bindings: [
        {
          name: 'parameters',
          view: maskParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {name: 'category', view: category, type: 'f32', access: 'read'},
        {name: 'hour', view: hour, type: 'f32', access: 'read'},
        {name: 'grade', view: grade, type: 'f32', access: 'read'},
        {name: 'mask', view: maskView, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let categoryFilter = parameters[parametersOffset];
  let hourLow = parameters[parametersOffset + 1u];
  let hourHigh = parameters[parametersOffset + 2u];
  let gradeMode = parameters[parametersOffset + 3u];
  var keep = true;
  if (categoryFilter != 0xffffffffu && u32(category[categoryOffset + index]) != categoryFilter) {
    keep = false;
  }
  let hourValue = u32(hour[hourOffset + index]);
  if (hourValue < hourLow || hourValue >= hourHigh) {
    keep = false;
  }
  let confirmed = grade[gradeOffset + index] > 0.5;
  if ((gradeMode == 1u && !confirmed) || (gradeMode == 2u && confirmed)) {
    keep = false;
  }
  mask[maskOffset + index] = select(0u, 1u, keep);`
    });

    const slots = new SlotSlices(graph, metricsBuffer, ZONE_CAPACITY);
    graph.add(
      new GPUGroupStatistics({
        id: 'nature-statistics',
        keys: bind.word(keysBuffer, observationCount),
        mask: maskView,
        keyCount: ZONE_COUNT,
        variance: options.variance,
        percentiles: percentileParameters.importToGraph(graph),
        columns: [
          {
            values: hour,
            statistics: ['mean', 'median', 'standardDeviation', 'percentiles', 'mode'],
            output: {
              means: slots.view(SLOT.hourMean, 'float32', ZONE_COUNT),
              medians: slots.view(SLOT.hourMedian, 'float32', ZONE_COUNT),
              standardDeviations: slots.view(SLOT.hourSd, 'float32', ZONE_COUNT),
              percentiles: slots.view(
                SLOT.hourPercentiles,
                'float32',
                ZONE_COUNT * PERCENTILE_COUNT
              ),
              modes: slots.view(SLOT.hourMode, 'float32', ZONE_COUNT)
            }
          },
          {
            values: grade,
            statistics: ['sum', 'mean'],
            output: {
              sumValues: slots.view(SLOT.gradeSum, 'float32', ZONE_COUNT),
              means: slots.view(SLOT.gradeMean, 'float32', ZONE_COUNT)
            }
          },
          {
            values: category,
            statistics: ['mode', 'uniqueCount'],
            output: {
              modes: slots.view(SLOT.categoryMode, 'float32', ZONE_COUNT),
              uniqueCounts: slots.view(SLOT.categoryUnique, 'uint32', ZONE_COUNT)
            }
          },
          {
            values: bind.float(speciesBuffer, observationCount),
            statistics: ['uniqueCount'],
            output: {uniqueCounts: slots.view(SLOT.speciesUnique, 'uint32', ZONE_COUNT)}
          },
          {
            values: bind.float(introducedBuffer, observationCount),
            statistics: ['mean'],
            output: {means: slots.view(SLOT.introducedMean, 'float32', ZONE_COUNT)}
          }
        ],
        output: {
          keys: bind.word(outputKeysBuffer, ZONE_COUNT),
          counts: slots.view(SLOT.counts, 'uint32', ZONE_COUNT),
          count: bind.word(outputCountBuffer, 1),
          overflow: bind.word(outputOverflowBuffer, 1)
        }
      })
    );

    // 1:n join: sum residents, income x residents and the residents with a known income, and
    // count the census tracts, over the tracts inside every community area.
    const populationSum = bind.float(populationSumBuffer, ZONE_COUNT);
    const incomeSum = bind.float(incomeSumBuffer, ZONE_COUNT);
    const incomePopulation = bind.float(incomePopulationBuffer, ZONE_COUNT);
    graph.add(
      new GPUKeyJoin({
        id: 'tracts-into-areas',
        leftKeys: bind.word(areaKeysBuffer, ZONE_COUNT),
        rightKeys: bind.word(tractZoneBuffer, tractCount),
        aggregates: [
          {
            operation: 'sum',
            column: bind.float(tractPopulation, tractCount),
            output: populationSum
          },
          {
            operation: 'sum',
            column: bind.float(tractIncomeProduct, tractCount),
            output: incomeSum
          },
          {
            operation: 'sum',
            column: bind.float(tractKnownPopulation, tractCount),
            output: incomePopulation
          },
          {operation: 'count', output: bind.float(tractCountBuffer, ZONE_COUNT)}
        ],
        output: {}
      })
    );

    // The displayed statistic per area (feature-aligned), the record count that sizes the circles
    // and the keep mask of the minimum-observations rule.
    const areaMetric = bind.float(areaMetricBuffer, ZONE_COUNT);
    const areaKeep = bind.word(areaKeepBuffer, ZONE_COUNT);
    addKernelPass(graph, {
      id: 'area-metric',
      invocationCount: ZONE_COUNT,
      bindings: [
        {
          name: 'selection',
          view: metricParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'minimum',
          view: minimumParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'metrics',
          view: slots.whole(SLOT_COUNT * ZONE_CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {name: 'populations', view: populationSum, type: 'f32', access: 'read'},
        {name: 'incomeSums', view: incomeSum, type: 'f32', access: 'read'},
        {name: 'incomePopulations', view: incomePopulation, type: 'f32', access: 'read'},
        {name: 'areaMetric', view: areaMetric, type: 'f32', access: 'read_write'},
        {
          name: 'areaCount',
          view: bind.float(areaCountBuffer, ZONE_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  // selection: [kind, slot or index, scale bits, suppress flag]; kind 0 float slot, 1 unsigned
  // slot, 2 percentile index, 3 derived column.
  let kind = selection[selectionOffset];
  let slot = selection[selectionOffset + 1u];
  let scale = bitcast<f32>(selection[selectionOffset + 2u]);
  let suppress = selection[selectionOffset + 3u] != 0u;
  let observationRows = metrics[metricsOffset + ${SLOT.counts}u * ${ZONE_CAPACITY}u + index];
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  let nan = bitcast<f32>(0x7fc00000u | (kind >> 31u));
  var value = nan;
  if (kind == 0u) {
    value = bitcast<f32>(metrics[metricsOffset + slot * ${ZONE_CAPACITY}u + index]) * scale;
  } else if (kind == 1u) {
    value = f32(metrics[metricsOffset + slot * ${ZONE_CAPACITY}u + index]);
  } else if (kind == 2u) {
    value = bitcast<f32>(metrics[metricsOffset + ${SLOT.hourPercentiles}u * ${ZONE_CAPACITY}u + index * ${PERCENTILE_COUNT}u + slot]);
  } else {
    let population = populations[populationsOffset + index];
    if (slot == ${DERIVED.perThousand}u) {
      value = select(nan, f32(observationRows) * 1000.0 / population, population > 0.0);
    } else if (slot == ${DERIVED.income}u) {
      // incomeSums holds thousands of dollars times residents.
      let known = incomePopulations[incomePopulationsOffset + index];
      value = select(nan, incomeSums[incomeSumsOffset + index] * 1000.0 / known, known > 0.0);
    } else {
      value = population;
    }
  }
  if (suppress && observationRows < minimum[minimumOffset]) {
    value = nan;
  }
  areaMetric[areaMetricOffset + index] = value;
  areaCount[areaCountOffset + index] = f32(observationRows);`
    });
    addKernelPass(graph, {
      id: 'area-keep',
      invocationCount: ZONE_COUNT,
      bindings: [
        {
          name: 'minimum',
          view: minimumParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'metrics',
          view: slots.whole(SLOT_COUNT * ZONE_CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {name: 'areaKeep', view: areaKeep, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let observationRows = metrics[metricsOffset + ${SLOT.counts}u * ${ZONE_CAPACITY}u + index];
  areaKeep[areaKeepOffset + index] = select(0u, 1u, observationRows >= minimum[minimumOffset]);`
    });

    // Gather join: paint the area statistic on every tract of the area (left or inner join).
    graph.add(
      new GPUKeyJoin({
        id: 'areas-onto-tracts',
        kind: options.joinKind,
        leftKeys: bind.word(tractZoneBuffer, tractCount),
        rightKeys: bind.word(areaKeysBuffer, ZONE_COUNT),
        rightMask: areaKeep,
        gather: [{column: areaMetric, output: bind.float(tractMetricBuffer, tractCount)}],
        output: {
          matched: bind.word(tractMatchedBuffer, tractCount),
          ...(options.joinKind === 'inner'
            ? {
                rows: {
                  ids: bind.word(innerRowsBuffer, tractCount),
                  count: bind.word(innerCountBuffer, 1),
                  overflow: bind.word(innerOverflowBuffer, 1)
                }
              }
            : {})
        }
      })
    );
    return {key, compiled: resources.track(graph.compile())};
  };

  const getVariant = (options: GroupStatisticsOptions): Variant => {
    const key = `${options.variance}|${options.joinKind}`;
    let variant = variants.get(key);
    if (!variant) {
      variant = compileVariant(options);
      variants.set(key, variant);
    }
    return variant;
  };
  let active = getVariant(ctx.options);

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------

  let dirty = true;
  let selectedZone = -1;
  let legendHighlight: number[] | null = null;

  /** Everything read back from the GPU after the graph ran, plus arrays derived from it. */
  type Latest = {
    metrics: Float32Array;
    metricWords: Uint32Array;
    areaMetric: Float32Array;
    population: Float32Array;
    incomeSum: Float32Array;
    incomePopulation: Float32Array;
    tractCounts: Float32Array;
    matchedTracts: number;
    innerCount: number;
    overflow: number;
    groups: number;
    /** Records per area under the filter. */
    counts: Float64Array;
    /** Unsuppressed records per 1,000 residents. */
    rate: Float64Array;
    /** Population-weighted per-capita income in dollars. */
    income: Float64Array;
    /** Areas with enough observations (all of them when the minimum is 0). */
    kept: Uint8Array;
  };
  let latest: Latest | null = null;
  let latestMinimum = 0;

  /** The statistic on the map: tracts and the swipe always draw the rate. */
  const getEffectiveMetric = (): MetricDefinition => {
    const {metric, zoning} = ctx.options;
    const id = zoning === 'areas' ? metric : 'perThousand';
    return GROUP_METRICS.find(entry => entry.id === id) ?? GROUP_METRICS[0];
  };

  const getRowFilter = (): RowFilter => {
    const options = ctx.options;
    const categoryIndex = categoryNames.findIndex(name => name === options.natureGroup);
    return {
      category: categoryIndex >= 0 ? categoryIndex : NO_KEY,
      hourLow: options.hours[0],
      hourHigh: options.hours[1],
      gradeMode: options.gradeFilter === 'all' ? 0 : options.gradeFilter === 'research' ? 1 : 2
    };
  };

  const writeParameters = () => {
    const options = ctx.options;
    const filter = getRowFilter();
    maskParameters.write(
      Uint32Array.of(filter.category, filter.hourLow, filter.hourHigh, filter.gradeMode)
    );
    percentileParameters.write(Float32Array.of(options.lowerFraction, 0.5, options.upperFraction));
    const metric = getEffectiveMetric();
    const parameter =
      metric.kind === 'percentile' || metric.kind === 'derived'
        ? (metric.index ?? 0)
        : (metric.slot ?? 0);
    metricParameters.write(
      Uint32Array.of(
        KERNEL_KIND[metric.kind],
        parameter,
        new Uint32Array(Float32Array.of(metric.scale ?? 1).buffer)[0],
        metric.suppressed ? 1 : 0
      )
    );
    minimumParameters.write(Uint32Array.of(options.minimumObservations, 0, 0, 0));
    dirty = true;
    reader.markStale();
  };

  // -------------------------------------------------------------------------------------------
  // Tables
  // -------------------------------------------------------------------------------------------

  const getTable = (): ClassTable =>
    getMetricTable({
      metric: getEffectiveMetric(),
      ground: ctx.ground(),
      breaks: frozenBreaks,
      classification: ctx.options.classification,
      minimumObservations: latestMinimum,
      zoning: ctx.options.zoning
    });

  /** Class index of an area value in the table the map draws. */
  const getClassOf = (table: ClassTable, value: number) =>
    Number.isFinite(value) ? getClassIndex(value, table.breaks) : -1;

  const toChartColor = (color: readonly [number, number, number, number?]): ChartColor => [
    color[0],
    color[1],
    color[2],
    color[3] ?? 255
  ];

  const getDefaultZone = (): number => {
    if (!latest) return -1;
    let best = -1;
    for (let zone = 0; zone < ZONE_COUNT; zone++) {
      if (!latest.kept[zone]) continue;
      if (best < 0 || latest.counts[zone] > latest.counts[best]) best = zone;
    }
    return best;
  };
  const getFocusZone = () => (selectedZone >= 0 ? selectedZone : getDefaultZone());

  // -------------------------------------------------------------------------------------------
  // Publishing: legend data, readouts, charts, notes
  // -------------------------------------------------------------------------------------------

  const publishLegendData = () => {
    if (!latest) return;
    const ground = ctx.ground();
    const table = getTable();
    const metric = getEffectiveMetric();
    const drawsTracts = ctx.options.zoning !== 'areas';
    const values = Array.from(latest.areaMetric);
    const counts = drawsTracts ? tractClassCounts : getClassCounts(values, table.breaks);
    const radiusOptions = {
      count: 3,
      minRadiusPixels: COUNT_RADIUS_MINIMUM_PIXELS,
      format: (value: number) => formatCount(value)
    };
    const withheld = metric.suppressed
      ? Array.from(latest.kept).filter(keep => keep === 0).length
      : 0;
    const legendData: GroupStatisticsLegend = {
      table,
      counts,
      circleEntries: getSizeLegendEntries(fullMaximumCount, COUNT_RADIUS_PIXELS, radiusOptions),
      circleColor: getCircleColor(ground) as GroupStatisticsLegend['circleColor'],
      hourColors: getHourColors() as GroupStatisticsLegend['hourColors'],
      noDataColor: NO_DATA_COLOR[ground] as GroupStatisticsLegend['noDataColor'],
      withheldAreas: withheld
    };
    ctx.setLegendData('groupStatistics', legendData);
  };

  const describeSelected = () => {
    const zone = selectedZone;
    if (!latest || zone < 0) return null;
    const rate = latest.rate[zone];
    return `${areaName(zone)}: ${formatCount(latest.counts[zone])} records, ${formatCount(latest.population[zone])} residents${Number.isFinite(rate) ? `, ${rate.toFixed(1)} per 1,000` : ''}`;
  };

  const publishReadouts = () => {
    if (!latest) return;
    const options = ctx.options;
    const zoneRange = Array.from({length: ZONE_COUNT}, (_, zone) => zone);
    let total = 0;
    let totalPopulation = 0;
    for (const zone of zoneRange) {
      total += latest.counts[zone];
      totalPopulation += latest.population[zone];
    }
    ctx.setReadout(
      'observations',
      `${formatCount(total)} of ${formatCount(observationCount)} records`
    );
    ctx.setReadout(
      'citywideRate',
      totalPopulation > 0 ? `${((total * 1000) / totalPopulation).toFixed(1)} per 1,000` : null
    );
    ctx.setReadout('medianRate', `${getMedian(latest.rate).toFixed(1)} per 1,000`);
    ctx.setReadout('threshold', `${formatCount(options.minimumObservations)} records`);

    let withheldAreas = 0;
    let withheldRecords = 0;
    let withheldPeople = 0;
    for (const zone of zoneRange) {
      if (latest.kept[zone]) continue;
      withheldAreas++;
      withheldRecords += latest.counts[zone];
      withheldPeople += latest.population[zone];
    }
    ctx.setReadout('suppressedAreas', `${withheldAreas} of ${ZONE_COUNT} areas`);
    ctx.setReadout(
      'suppressedShare',
      total > 0 ? `${((100 * withheldRecords) / total).toFixed(1)}% of the records` : null
    );
    ctx.setReadout('suppressedPeople', `${formatCount(withheldPeople)} residents`);
    ctx.setReadout(
      'groups',
      `${latest.groups} groups${latest.overflow ? ' (CAPACITY OVERFLOW)' : ''}`
    );
    ctx.setReadout('selected', describeSelected());

    // Median hour: the earliest and the latest area among those that are not withheld.
    let earliest = -1;
    let latestZone = -1;
    for (const zone of zoneRange) {
      const median = latest.metrics[SLOT.hourMedian * ZONE_CAPACITY + zone];
      if (!latest.kept[zone] || !Number.isFinite(median)) continue;
      if (earliest < 0 || median < latest.metrics[SLOT.hourMedian * ZONE_CAPACITY + earliest]) {
        earliest = zone;
      }
      if (latestZone < 0 || median > latest.metrics[SLOT.hourMedian * ZONE_CAPACITY + latestZone]) {
        latestZone = zone;
      }
    }
    const medianText = (zone: number) =>
      zone < 0
        ? null
        : `${areaName(zone)}, ${formatClockHour(latest!.metrics[SLOT.hourMedian * ZONE_CAPACITY + zone])}`;
    ctx.setReadout('earliestMedian', medianText(earliest));
    ctx.setReadout('latestMedian', medianText(latestZone));
    const focus = getFocusZone();
    if (focus >= 0) {
      const base = SLOT.hourPercentiles * ZONE_CAPACITY + focus * PERCENTILE_COUNT;
      ctx.setReadout(
        'hourSpan',
        `${areaName(focus)}: ${formatClockHour(latest.metrics[base])} to ${formatClockHour(latest.metrics[base + 2])} (P${Math.round(options.lowerFraction * 100)} to P${Math.round(options.upperFraction * 100)})`
      );
    } else {
      ctx.setReadout('hourSpan', null);
    }

    ctx.setReadout(
      'joinResult',
      options.joinKind === 'inner'
        ? `inner join keeps ${formatCount(latest.innerCount)} of ${formatCount(tractCount)} tracts`
        : `left join: ${formatCount(latest.matchedTracts)} of ${formatCount(tractCount)} tracts match an area with enough records`
    );

    // Tract scale (the zoning step): how well the area class describes the tracts inside it.
    const topBreak = TRACT_RATE_BREAKS[TRACT_RATE_BREAKS.length - 1];
    let insideTotal = 0;
    let insideBelow = 0;
    let outsideTop = 0;
    let emptyTracts = 0;
    let maxTract = -1;
    for (let tract = 0; tract < tractCount; tract++) {
      const rate = tractRateHost[tract];
      if (tractObservationColumn[tract] === 0) emptyTracts++;
      if (Number.isFinite(rate) && (maxTract < 0 || rate > tractRateHost[maxTract]))
        maxTract = tract;
      const zone = tractZone[tract];
      const areaRate = zone < ZONE_COUNT ? reference.perThousand[zone] : Number.NaN;
      const areaIsTop = areaRate >= topBreak;
      const tractIsTop = rate >= topBreak;
      if (areaIsTop) {
        insideTotal++;
        if (!tractIsTop) insideBelow++;
      } else if (tractIsTop) {
        outsideTop++;
      }
    }
    ctx.setReadout(
      'tractMax',
      maxTract >= 0
        ? `${formatCount(tractRateHost[maxTract])} per 1,000, in ${tractZone[maxTract] < ZONE_COUNT ? areaName(tractZone[maxTract]) : 'no area'}`
        : null
    );
    ctx.setReadout('notTopInside', `${insideBelow} of ${insideTotal} tracts`);
    ctx.setReadout('topOutside', `${outsideTop} tracts`);
    ctx.setReadout(
      'emptyTracts',
      `${formatCount(emptyTracts)} of ${formatCount(tractCount)} tracts`
    );
  };

  /** The run chart: records per area, largest first, withheld areas in grey. */
  const publishRunChart = () => {
    if (!latest) return;
    const ground = ctx.ground();
    const order = getDescendingOrder(latest.counts);
    const kept = toChartColor(getDotColor(ground));
    const withheld = toChartColor(getSuppressedBarColor(ground));
    const minimum = ctx.options.minimumObservations;
    const chart: ChartData = {
      kind: 'bars',
      title: 'Records per area, largest first',
      yLabel: 'Records (log scale)',
      yScale: 'log',
      values: order.map(zone => (latest!.counts[zone] > 0 ? latest!.counts[zone] : Number.NaN)),
      labels: order.map(zone => areaName(zone)),
      colors: order.map(zone => (latest!.kept[zone] ? kept : withheld)),
      guides: minimum > 0 ? [{y: minimum, label: 'Threshold'}] : undefined,
      description:
        'Bar chart of the records in each of the 77 community areas, sorted from the largest to the smallest on a log scale; areas under the threshold are grey. Click a bar to outline the area.',
      onBarClick: index => selectZone(order[index], true)
    };
    ctx.setChart('runs', chart);
  };

  /** The rose chart: records by hour of the focus area against the citywide shape. */
  const publishHourChart = () => {
    if (!latest) return;
    const zone = getFocusZone();
    if (zone < 0) {
      ctx.setChart('hours', null);
      return;
    }
    const ground = ctx.ground();
    const options = ctx.options;
    const filter = getRowFilter();
    const areaCounts = getHourCounts(hourColumns, filter, zone);
    const cityCounts = getHourCounts(hourColumns, filter, -1);
    const areaTotal = areaCounts.reduce((sum, value) => sum + value, 0);
    const cityTotal = cityCounts.reduce((sum, value) => sum + value, 0);
    const base = SLOT.hourPercentiles * ZONE_CAPACITY + zone * PERCENTILE_COUNT;
    const low = latest.metrics[base];
    const high = latest.metrics[base + 2];
    const classColors = getHourColors();
    const muted = getMutedSectorColor(ground);
    const colors: ChartColor[] = Array.from({length: 24}, (_, hour) =>
      hour + 1 > low && hour <= high
        ? toChartColor(classColors[getHourClass(hour)])
        : toChartColor(muted)
    );
    ctx.setChart('hours', {
      kind: 'rose',
      title: `${areaName(zone)}: records by hour`,
      values: Array.from(areaCounts),
      baseline: Array.from(cityCounts, value =>
        cityTotal > 0 ? (value * areaTotal) / cityTotal : 0
      ),
      labels: ['0:00', '6:00', '12:00', '18:00'],
      colors,
      description: `Polar bar chart of the records of ${areaName(zone)} by hour of the day, coloured between the P${Math.round(options.lowerFraction * 100)} and P${Math.round(options.upperFraction * 100)} hours and grey outside, with the citywide shape as a ghost outline.`
    });
  };

  /** The join chart: weighted income against the rate, one dot per area that is not withheld. */
  const publishIncomeChart = () => {
    if (!latest) return;
    const ground = ctx.ground();
    const zones: number[] = [];
    const x: number[] = [];
    const y: number[] = [];
    for (let zone = 0; zone < ZONE_COUNT; zone++) {
      const rate = latest.rate[zone];
      const income = latest.income[zone];
      if (!latest.kept[zone] || !(rate > 0) || !Number.isFinite(income)) continue;
      zones.push(zone);
      x.push(income);
      y.push(rate);
    }
    const rho = getSpearmanCorrelation(x, y);
    ctx.setReadout(
      'spearman',
      Number.isFinite(rho) ? `${rho.toFixed(2)} across ${zones.length} areas` : null
    );
    const focusIndex = zones.indexOf(selectedZone);
    ctx.setChart('incomeScatter', {
      kind: 'scatter',
      title: 'Income against records per resident, one dot per area',
      x,
      y,
      xLabel: 'Population-weighted per-capita income',
      yLabel: 'Records per 1,000 residents',
      yScale: 'log',
      formatX: value => `$${Math.round(value / 1000)}k`,
      palette: [toChartColor(getDotColor(ground))],
      colorIndex: x.map(() => 0),
      ringed: focusIndex >= 0 ? [focusIndex] : undefined,
      radius: 2.6,
      opacity: 0.8,
      description:
        'Scatter plot of the community areas: population-weighted per-capita income on the x axis, observations per 1,000 residents on a log y axis. Click a dot to find the area on the map.',
      onPointClick: index => selectZone(zones[index], true)
    });
  };

  /** Finding notes, chosen by what the map shows now. At most three. */
  const updateNotes = () => {
    if (!latest) {
      ctx.setAnnotations('group-notes', null);
      return;
    }
    const options = ctx.options;
    const metric = getEffectiveMetric();
    const notes: MapAnnotation[] = [];
    const filtered =
      options.natureGroup !== 'ALL' ||
      options.hours[0] !== 0 ||
      options.hours[1] !== 24 ||
      options.gradeFilter !== 'all';
    const argmax = (values: ArrayLike<number>, eligible: (zone: number) => boolean) => {
      let best = -1;
      for (let zone = 0; zone < ZONE_COUNT; zone++) {
        if (!eligible(zone) || !Number.isFinite(values[zone])) continue;
        if (best < 0 || values[zone] > values[best]) best = zone;
      }
      return best;
    };
    const argmin = (values: ArrayLike<number>, eligible: (zone: number) => boolean) => {
      let best = -1;
      for (let zone = 0; zone < ZONE_COUNT; zone++) {
        if (!eligible(zone) || !Number.isFinite(values[zone])) continue;
        if (best < 0 || values[zone] < values[best]) best = zone;
      }
      return best;
    };
    const kept = (zone: number) => latest!.kept[zone] === 1;
    if (options.zoning !== 'areas') {
      let maxTract = -1;
      for (let tract = 0; tract < tractCount; tract++) {
        const rate = tractRateHost[tract];
        if (Number.isFinite(rate) && (maxTract < 0 || rate > tractRateHost[maxTract])) {
          maxTract = tract;
        }
      }
      if (maxTract >= 0) {
        const point = tractMesh.labelPoints[maxTract];
        notes.push({
          kind: 'note',
          coordinate: [point[0], point[1]] as LngLat,
          priority: 5,
          title: liveText('{rate:integer} per 1,000 in one tract', {
            rate: tractRateHost[maxTract]
          }),
          text: `${formatCount(tractObservationColumn[maxTract])} records, ${formatCount(populationColumn[maxTract])} residents`
        });
      }
    } else if (metric.family === 'hour') {
      const medians = Array.from(
        {length: ZONE_COUNT},
        (_, zone) => latest!.metrics[SLOT.hourMedian * ZONE_CAPACITY + zone]
      );
      for (const zone of [argmin(medians, kept), argmax(medians, kept)]) {
        if (zone < 0) continue;
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(zone),
          priority: 5,
          title: `${formatClockHour(medians[zone])} median hour`,
          text: areaName(zone)
        });
      }
    } else if (metric.family === 'income') {
      const richest = argmax(latest.income, () => true);
      const poorest = argmin(latest.income, () => true);
      const hottest = argmax(latest.rate, kept);
      for (const [zone, text] of [
        [richest, 'highest income'],
        [poorest, 'lowest income']
      ] as const) {
        if (zone < 0) continue;
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(zone),
          priority: 5,
          title: `$${formatCount(latest.income[zone])} per person`,
          text: `${areaName(zone)}: ${text}`
        });
      }
      if (hottest >= 0 && hottest !== richest && hottest !== poorest) {
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(hottest),
          priority: 4,
          title: liveText('{rate:fixed:0} per 1,000', {rate: latest.rate[hottest]}),
          text: `${areaName(hottest)}: highest rate`
        });
      }
    } else if (filtered) {
      // The filter step speaks through the legend and the readouts, not through notes.
    } else if (options.minimumObservations > 0 && options.view === 'rates') {
      const withheld = argmax(latest.population, zone => !kept(zone));
      if (withheld >= 0) {
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(withheld),
          priority: 5,
          title: liveText('{n:integer} records, {people:integer} residents', {
            n: latest.counts[withheld],
            people: latest.population[withheld]
          }),
          text: `${areaName(withheld)}: withheld`
        });
      }
    } else {
      const busiest = argmax(latest.counts, () => true);
      const highest = argmax(latest.rate, () => true);
      let cityRecords = 0;
      let cityResidents = 0;
      for (let zone = 0; zone < ZONE_COUNT; zone++) {
        cityRecords += latest.counts[zone];
        cityResidents += latest.population[zone];
      }
      const cityPerThousand = (cityRecords * 1000) / cityResidents;
      const populous = argmax(
        latest.population,
        zone => latest!.rate[zone] < cityPerThousand && zone !== busiest && zone !== highest
      );
      if (busiest >= 0) {
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(busiest),
          priority: 5,
          title: liveText('{n:integer} records', {n: latest.counts[busiest]}),
          text: `${areaName(busiest)}: the most records`
        });
      }
      if (highest >= 0) {
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(highest),
          priority: 5,
          title: liveText('{rate:fixed:0} per 1,000', {rate: latest.rate[highest]}),
          text: `${areaName(highest)}: the highest rate`
        });
      }
      if (populous >= 0) {
        notes.push({
          kind: 'note',
          coordinate: getLabelPoint(populous),
          priority: 4,
          title: liveText('{people:integer} residents, {rate:fixed:1} per 1,000', {
            people: latest.population[populous],
            rate: latest.rate[populous]
          }),
          text: `${areaName(populous)}: populous, below the citywide rate`
        });
      }
    }
    ctx.setAnnotations('group-notes', notes.length ? notes : null);
  };

  const publish = () => {
    publishLegendData();
    publishReadouts();
    publishRunChart();
    publishHourChart();
    publishIncomeChart();
    updateNotes();
    ctx.setCost({
      records: observationCount,
      passes: active.compiled.stats.nodeOrder.length,
      note: DESCRIBED_NOTE
    });
    ctx.requestLayers();
  };

  // -------------------------------------------------------------------------------------------
  // Selection
  // -------------------------------------------------------------------------------------------

  const writeSelection = () => {
    if (selectedZone < 0) {
      selectionCount = 0;
    } else {
      const segments = getAreaSegments(selectedZone);
      selectionBuffer.write(segments);
      selectionCount = segments.length / 4;
    }
  };

  function selectZone(zone: number, pulse = false): void {
    selectedZone = zone === selectedZone && !pulse ? -1 : zone;
    writeSelection();
    ctx.setReadout('selected', describeSelected());
    publishReadouts();
    publishHourChart();
    publishIncomeChart();
    if (pulse && selectedZone >= 0) {
      ctx.setHighlight({kind: 'polygon', rings: getRings(areaFeatures, selectedZone), pulse: true});
    } else {
      ctx.setHighlight(null);
    }
    ctx.requestLayers();
  }

  // -------------------------------------------------------------------------------------------
  // Readback
  // -------------------------------------------------------------------------------------------

  const reader = new SummaryReader(
    resources,
    'group-statistics',
    [
      {buffer: metricsBuffer, size: SLOT_COUNT * ZONE_CAPACITY * 4},
      {buffer: areaMetricBuffer, size: ZONE_COUNT * 4},
      {buffer: populationSumBuffer, size: ZONE_COUNT * 4},
      {buffer: incomeSumBuffer, size: ZONE_COUNT * 4},
      {buffer: incomePopulationBuffer, size: ZONE_COUNT * 4},
      {buffer: tractCountBuffer, size: ZONE_COUNT * 4},
      {buffer: tractMatchedBuffer, size: tractCount * 4},
      {buffer: innerCountBuffer, size: 4},
      {buffer: outputOverflowBuffer, size: 4},
      {buffer: outputCountBuffer, size: 4}
    ],
    bytes => {
      const read = createByteReader(bytes.slice(0));
      const metricWords = read.words(SLOT_COUNT * ZONE_CAPACITY).slice();
      const metrics = new Float32Array(metricWords.buffer.slice(0));
      const areaMetric = read.floats(ZONE_COUNT).slice();
      const population = read.floats(ZONE_COUNT).slice();
      const incomeSum = read.floats(ZONE_COUNT).slice();
      const incomePopulation = read.floats(ZONE_COUNT).slice();
      const tractCounts = read.floats(ZONE_COUNT).slice();
      const matched = read.words(tractCount);
      const innerCount = read.words(1)[0];
      const overflow = read.words(1)[0];
      const groups = read.words(1)[0];
      const minimum = ctx.options.minimumObservations;
      const counts = new Float64Array(ZONE_COUNT);
      const rate = new Float64Array(ZONE_COUNT);
      const income = new Float64Array(ZONE_COUNT);
      const kept = new Uint8Array(ZONE_COUNT);
      let matchedTracts = 0;
      for (let tract = 0; tract < tractCount; tract++) matchedTracts += matched[tract] ? 1 : 0;
      for (let zone = 0; zone < ZONE_COUNT; zone++) {
        counts[zone] = metricWords[SLOT.counts * ZONE_CAPACITY + zone];
        rate[zone] = population[zone] > 0 ? (counts[zone] * 1000) / population[zone] : Number.NaN;
        income[zone] =
          incomePopulation[zone] > 0
            ? (incomeSum[zone] * 1000) / incomePopulation[zone]
            : Number.NaN;
        kept[zone] = counts[zone] >= minimum ? 1 : 0;
      }
      latestMinimum = minimum;
      latest = {
        metrics,
        metricWords,
        areaMetric,
        population,
        incomeSum,
        incomePopulation,
        tractCounts,
        matchedTracts,
        innerCount,
        overflow,
        groups,
        counts,
        rate,
        income,
        kept
      };
      publish();
    }
  );

  ctx.setFurniture({
    title: {
      sample: `${formatCount(observationCount)} iNaturalist records, ${ZONE_COUNT} community areas, ${formatCount(tractCount)} census tracts`
    }
  });
  writeParameters();
  ctx.setReadout('selected', null);

  // -------------------------------------------------------------------------------------------
  // Layers
  // -------------------------------------------------------------------------------------------

  const getFillLayer = (
    id: string,
    geometry: typeof areaPolygons,
    origin: [number, number, number],
    style: SpatialAnalysisStyleProps,
    side?: 'a' | 'b'
  ) =>
    new SpatialAnalysisPolygonLayer({
      id,
      coordinateOrigin: origin,
      triangles: geometry.triangles,
      features: geometry.triangleFeatures,
      vertexCount: geometry.vertexCount,
      ...style,
      compareSide: side,
      opacity: FILL_OPACITY
    });

  const getLineLayer = (
    id: string,
    geometry: typeof areaPolygons,
    origin: [number, number, number],
    widthPixels: number,
    color: readonly [number, number, number, number],
    side?: 'a' | 'b'
  ) =>
    new SpatialAnalysisSegmentLayer({
      id,
      coordinateOrigin: origin,
      segments: geometry.outline,
      instanceCount: geometry.outlineCount,
      widthPixels,
      color,
      compareSide: side
    });

  const getLayers = (): Layer[] => {
    if (!latest) return [];
    const options = ctx.options;
    const ground = ctx.ground();
    const table = getTable();
    const layers: Layer[] = [];
    const classStyle = (values: typeof areaMetricBuffer): SpatialAnalysisStyleProps => ({
      values,
      valueFormat: 'float32',
      colormap: 'greys',
      ...getClassTableLayerProps(table),
      hatchNoData: true,
      hatchColor: getHatchColor(ground),
      noDataColor: NO_DATA_COLOR[ground],
      highlightClasses: legendHighlight
    });
    const contextLines = getAreaLinePasses(ground, 'context');
    const subjectLines = getAreaLinePasses(ground, 'subject');
    const counting = options.zoning === 'areas' && options.view === 'counts';

    if (counting) {
      layers.push(
        getFillLayer('group-unfilled', areaPolygons, areaLayerOrigin, {
          colormap: 'uniform',
          color: getUnfilledAreaColor(ground)
        })
      );
      contextLines.forEach((pass, index) => {
        layers.push(
          getLineLayer(
            `group-area-lines-${index}`,
            areaPolygons,
            areaLayerOrigin,
            pass.widthPixels,
            pass.color
          )
        );
      });
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'group-circles',
          coordinateOrigin: areaLayerOrigin,
          positions: circlePositionBuffer,
          ids: circleOrderBuffer,
          instanceCount: ZONE_COUNT,
          sizeValues: areaCountBuffer,
          sizeMaximumValue: fullMaximumCount,
          sizeScale: 'sqrt',
          radiusPixels: COUNT_RADIUS_PIXELS,
          radiusMinPixels: COUNT_RADIUS_MINIMUM_PIXELS,
          shape: 'circle',
          colormap: 'uniform',
          color: getCircleColor(ground),
          fillOpacity: 0.55,
          outlineColor: getCasingColor(ground, 235),
          outlineWidthPixels: 1
        })
      );
    } else {
      const swipe = options.zoning === 'swipe';
      if (options.zoning !== 'tracts') {
        layers.push(
          getFillLayer(
            'group-area-fill',
            areaPolygons,
            areaLayerOrigin,
            classStyle(areaMetricBuffer),
            swipe ? 'a' : undefined
          )
        );
      }
      if (options.zoning !== 'areas') {
        layers.push(
          getFillLayer(
            'group-tract-fill',
            tractPolygons,
            tractLayerOrigin,
            classStyle(tractRateBuffer),
            swipe ? 'b' : undefined
          )
        );
        layers.push(
          getLineLayer(
            'group-tract-hairlines',
            tractPolygons,
            tractLayerOrigin,
            0.4,
            getTractHairlineColor(ground),
            swipe ? 'b' : undefined
          )
        );
        subjectLines.forEach((pass, index) => {
          layers.push(
            getLineLayer(
              `group-area-subject-${index}`,
              areaPolygons,
              areaLayerOrigin,
              pass.widthPixels,
              pass.color,
              swipe ? 'b' : undefined
            )
          );
        });
      }
      if (options.zoning !== 'tracts') {
        contextLines.forEach((pass, index) => {
          layers.push(
            getLineLayer(
              `group-area-lines-${index}`,
              areaPolygons,
              areaLayerOrigin,
              pass.widthPixels,
              pass.color,
              swipe ? 'a' : undefined
            )
          );
        });
      }
    }

    if (selectedZone >= 0 && selectionCount > 0) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'group-selection',
          coordinateOrigin: areaLayerOrigin,
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
  // Tooltips
  // -------------------------------------------------------------------------------------------

  /** Whether the pointer is over the tract side of the map. */
  const isTractSide = (pixelX: number): boolean => {
    const {zoning} = ctx.options;
    if (zoning === 'tracts') return true;
    if (zoning === 'areas') return false;
    const compare = ctx.getCompare();
    const width = ctx.getViewport()?.width ?? 0;
    if (!compare || width <= 0) return false;
    if (compare.showing === 'b') return true;
    if (compare.showing === 'a') return false;
    return pixelX / width > compare.position;
  };

  const describeArea = (zone: number): TooltipContent | null => {
    if (!latest) return null;
    const options = ctx.options;
    const metric = getEffectiveMetric();
    const table = getTable();
    const counting = options.zoning === 'areas' && options.view === 'counts';
    const records = latest.counts[zone];
    const residents = latest.population[zone];
    const rate = latest.rate[zone];
    const withheld = !counting && metric.suppressed && !latest.kept[zone];
    const value = latest.areaMetric[zone];
    const swatch = toChartColor;
    const rows: TooltipRow[] = [];
    const sorted = Float64Array.from(latest.areaMetric).filter(Number.isFinite).sort();
    const rank = (shown: number): TooltipRow => ({
      label: 'Rank',
      value: `${formatOrdinal(Math.round(100 * getPercentileRank(sorted, shown)))} percentile`,
      unit: `of ${sorted.length} areas`
    });
    if (counting) {
      rows.push({
        label: 'Records',
        value: formatCount(records),
        unit: 'in the filter',
        swatch: swatch(getCircleColor(ctx.ground())),
        emphasis: true
      });
      const countSorted = Float64Array.from(latest.counts).sort();
      rows.push({
        label: 'Rank',
        value: `${formatOrdinal(Math.round(100 * getPercentileRank(countSorted, records)))} percentile`,
        unit: `of ${ZONE_COUNT} areas`
      });
    } else if (withheld || !Number.isFinite(value)) {
      rows.push({label: metric.label, value: withheld ? 'withheld' : 'no data', emphasis: true});
    } else {
      const classIndex = getClassOf(table, value);
      const formatted =
        metric.family === 'hour'
          ? formatClockHour(value)
          : metric.family === 'income'
            ? `$${formatCount(value)}`
            : metric.family === 'richness'
              ? formatCount(value)
              : value.toFixed(metric.family === 'share' ? 0 : 1);
      rows.push({
        label: metric.label,
        value: formatted,
        unit:
          metric.family === 'rate'
            ? RATE_BASIS
            : metric.family === 'share'
              ? '% of the records'
              : metric.family === 'richness'
                ? 'taxa'
                : undefined,
        swatch: classIndex >= 0 ? swatch(table.colors[classIndex]) : undefined,
        emphasis: true
      });
      rows.push(rank(value));
    }
    rows.push(
      {label: 'Records', value: formatCount(records), unit: 'in the filter'},
      {
        label: 'Residents',
        value: formatCount(residents),
        unit: `in ${latest.tractCounts[zone]} tracts`
      }
    );
    const cityRecords = latest.counts.reduce((sum, count) => sum + count, 0);
    const cityResidents = latest.population.reduce((sum, count) => sum + count, 0);
    const cityRate = cityResidents > 0 ? (cityRecords * 1000) / cityResidents : Number.NaN;
    if (Number.isFinite(rate) && Number.isFinite(cityRate) && cityRate > 0) {
      rows.push({
        label: 'Against the city',
        value: `${(rate / cityRate).toFixed(1)}x`,
        unit: `the citywide rate of ${cityRate.toFixed(1)}`
      });
    }
    if (records > 0) {
      rows.push(
        {
          label: 'Research grade',
          value: `${(100 * latest.metrics[SLOT.gradeMean * ZONE_CAPACITY + zone]).toFixed(0)}%`,
          unit: 'of the records'
        },
        {
          label: 'Median hour',
          value: formatClockHour(latest.metrics[SLOT.hourMedian * ZONE_CAPACITY + zone]),
          unit: `spread (sd) ${latest.metrics[SLOT.hourSd * ZONE_CAPACITY + zone].toFixed(1)} h`
        }
      );
    }
    return {
      title: areaName(zone),
      subtitle: 'Community area',
      rows,
      note: withheld
        ? `Only ${formatCount(records)} records: this statistic is withheld below ${formatCount(options.minimumObservations)}`
        : undefined,
      anchor: getLabelPoint(zone),
      highlight: {kind: 'polygon', rings: getRings(areaFeatures, zone)}
    };
  };

  const describeTract = (tract: number): TooltipContent | null => {
    if (!latest) return null;
    const table = getTractRateTable(ctx.ground());
    const rate = tractRateHost[tract];
    const zone = tractZone[tract];
    const classIndex = getClassOf(table, rate);
    const geoid = tractFeatures[tract]?.properties?.GEOID;
    const rows: TooltipRow[] = [
      {
        label: 'Records per 1,000 residents',
        value: Number.isFinite(rate) ? rate.toFixed(1) : 'no residents',
        swatch: classIndex >= 0 ? toChartColor(table.colors[classIndex]) : undefined,
        emphasis: true
      },
      {label: 'Records', value: formatCount(tractObservationColumn[tract]), unit: 'in 2023'},
      {label: 'Residents', value: formatCount(populationColumn[tract])}
    ];
    if (zone < ZONE_COUNT) {
      rows.push({
        label: `${areaName(zone)} as a whole`,
        value: Number.isFinite(latest.rate[zone]) ? latest.rate[zone].toFixed(1) : 'n/a',
        unit: RATE_BASIS
      });
    }
    const point = tractMesh.labelPoints[tract];
    return {
      title: `Tract ${geoid ?? tract}`,
      subtitle: zone < ZONE_COUNT ? `Census tract in ${areaName(zone)}` : 'Census tract',
      rows,
      anchor: [point[0], point[1]] as LngLat,
      highlight: {kind: 'polygon', rings: getRings(tractFeatures, tract)}
    };
  };

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (id === 'variance' || id === 'joinKind') {
        active = getVariant(ctx.options);
        dirty = true;
        reader.markStale();
      } else if (id === 'classification' || id === 'view') {
        legendHighlight = null;
        publishLegendData();
        updateNotes();
      } else {
        legendHighlight = null;
        writeParameters();
        if (id === 'zoning') {
          publishLegendData();
          updateNotes();
        }
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 2) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers,

    onGroundChange() {
      publish();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      if (isTractSide(event.pixel[0])) {
        const found = tractLocator.find(event.coordinate);
        if (found) return describeTract(found.index);
      }
      const found = areaLocator.find(event.coordinate);
      return found ? describeArea(found.index) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const found = areaLocator.find(event.coordinate);
      selectZone(found ? found.index : -1);
      return true;
    },

    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
