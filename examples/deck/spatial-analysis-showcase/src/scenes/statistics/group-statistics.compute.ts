// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  GPUGroupStatistics,
  GPUKeyJoin,
  type GPUKeyJoinKind
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createByteReader,
  createChoroplethGeometry,
  formatNumber,
  getOutlineColor,
  getQuantile,
  getSortedFinite
} from './b5-common';
import {createGraphImporter} from './b5-classify';
import {
  GROUP_METRICS,
  PERCENTILE_COUNT,
  SLOT,
  SLOT_COUNT,
  ZONE_CAPACITY,
  ZONE_COUNT
} from './b5-group-metrics';
import {clearLegendData, formatCompact, setLegendData} from './b5-legend-bus';
import {NATURE_CATEGORY_COLORS} from './b5-palettes';
import {SlotSlices} from './b5-slots';

/** Option state of the group-statistics scene. */
export type GroupStatisticsOptions = {
  level: 'areas' | 'tracts';
  metric: string;
  natureGroup: string;
  hours: readonly [number, number];
  gradeFilter: 'all' | 'research' | 'unconfirmed';
  variance: 'sample' | 'population';
  lowerFraction: number;
  upperFraction: number;
  joinKind: GPUKeyJoinKind;
  minimumObservations: number;
  showObservations: boolean;
  outlines: boolean;
};

const NO_KEY = 0xffffffff;

/** Legend data for the scene. */
export type GroupStatisticsLegend = {
  range: [number, number];
  modalGroups: {index: number; name: string; areas: number}[];
};

/**
 * Nature observation statistics per community area with `GPUGroupStatistics` (count, mean, median, percentiles,
 * deviation, skewness, kurtosis, mode, distinct count, sums and per-observation z-scores over hour,
 * grade, category and introduced columns) and `GPUKeyJoin`: a 1:n join attaches the sums and means
 * of the census tracts inside each area, and a gather join paints the area statistic on every
 * tract, as a left or inner join. A GPU kernel builds the filter mask from the options each frame.
 */
export async function createGroupStatistics(
  ctx: SceneContext<GroupStatisticsOptions>
): Promise<SceneInstance<GroupStatisticsOptions>> {
  const observations = ctx.datasets.get('chicago-nature');
  const areas = ctx.datasets.get('chicago-community-areas');
  const tracts = ctx.datasets.get('chicago-tracts');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'group-statistics');
  const areaGeometry = createChoroplethGeometry(resources, areas);
  const tractGeometry = createChoroplethGeometry(resources, tracts);
  const tractCount = tractGeometry.featureCount;
  const areaFeatures = areas.geojson?.features ?? [];
  const tractFeatures = tracts.geojson?.features ?? [];
  const areaName = (zone: number) =>
    String(areaFeatures[zone]?.properties?.name ?? `Area ${zone + 1}`);
  const categoryNames = observations.categories('category');
  const formatCategory = (name: string) => name;

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

  const tractArea = tracts.column<Uint8Array>('communityArea');
  const tractZone = Uint32Array.from(tractArea, area => (area >= 1 ? area - 1 : NO_KEY));
  const populationColumn = tracts.column<Float32Array>('population');
  const incomeColumn = tracts.column<Float32Array>('perCapitaIncome');
  const povertyColumn = tracts.column<Float32Array>('poverty150Pct');
  const areaKeys = Uint32Array.from({length: ZONE_COUNT}, (_, index) => index);

  const hourBuffer = resources.createBuffer('hour', hourHost);
  const categoryBuffer = resources.createBuffer('category', categoryHost);
  const speciesBuffer = resources.createBuffer('species', speciesHost);
  const gradeBuffer = resources.createBuffer('grade', gradeHost);
  const introducedBuffer = resources.createBuffer('introduced', introducedHost);
  const keysBuffer = resources.createBuffer('nature-keys', keyHost);
  const maskBuffer = resources.createBuffer('mask', observationCount * 4);
  const zScoreBuffer = resources.createBuffer('hour-z-scores', observationCount * 4);
  const positionsBuffer = resources.createBuffer(
    'nature-positions',
    observations.column<Float32Array>('position')
  );
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
  const tractIncome = resources.createBuffer('tract-income', incomeColumn);
  const tractPoverty = resources.createBuffer('tract-poverty', povertyColumn);
  const populationSumBuffer = resources.createBuffer('area-population', ZONE_COUNT * 4);
  const incomeMeanBuffer = resources.createBuffer('area-income', ZONE_COUNT * 4);
  const povertyMeanBuffer = resources.createBuffer('area-poverty', ZONE_COUNT * 4);
  const tractCountBuffer = resources.createBuffer('area-tract-count', ZONE_COUNT * 4);
  const areaMetricBuffer = resources.createBuffer('area-metric', ZONE_COUNT * 4);
  const areaCategoryBuffer = resources.createBuffer('area-category', ZONE_COUNT * 4);
  const areaKeepBuffer = resources.createBuffer('area-keep', ZONE_COUNT * 4);
  const tractMetricBuffer = resources.createBuffer('tract-metric', tractCount * 4);
  const tractCategoryBuffer = resources.createBuffer('tract-category', tractCount * 4);
  const tractMatchedBuffer = resources.createBuffer('tract-matched', tractCount * 4);
  const innerRowsBuffer = resources.createBuffer('inner-rows', tractCount * 4);
  const innerCountBuffer = resources.createBuffer('inner-count', 4);
  const innerOverflowBuffer = resources.createBuffer('inner-overflow', 4);

  type Variant = {key: string; compiled: CompiledGPUCommandGraph<void>};
  const variants = new Map<string, Variant>();

  const compileVariant = (options: GroupStatisticsOptions): Variant => {
    const key = `${options.variance}|${options.joinKind}`;
    const graph = new GPUCommandGraph<void>(device, {id: `group-statistics-${key}`});
    // A graph rejects two imports of one buffer, so each buffer is imported once per graph.
    const bind = createGraphImporter(graph);
    const float = (_id: string, buffer: typeof hourBuffer, length: number) =>
      bind.float(buffer, length);
    const word = (_id: string, buffer: typeof hourBuffer, length: number) =>
      bind.word(buffer, length);
    const maskView = word('mask', maskBuffer, observationCount);
    const hour = float('hour', hourBuffer, observationCount);
    const category = float('category', categoryBuffer, observationCount);
    const grade = float('grade', gradeBuffer, observationCount);

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
    const percentileView = slots.view(
      SLOT.hourPercentiles,
      'float32',
      ZONE_COUNT * PERCENTILE_COUNT
    );
    graph.add(
      new GPUGroupStatistics({
        id: 'nature-statistics',
        keys: word('nature-keys', keysBuffer, observationCount),
        mask: maskView,
        keyCount: ZONE_COUNT,
        variance: options.variance,
        percentiles: percentileParameters.importToGraph(graph),
        columns: [
          {
            values: hour,
            statistics: [
              'mean',
              'median',
              'standardDeviation',
              'percentiles',
              'mode',
              'uniqueCount',
              'skewness',
              'kurtosis',
              'minimum',
              'maximum',
              'zScore'
            ],
            output: {
              means: slots.view(SLOT.hourMean, 'float32', ZONE_COUNT),
              medians: slots.view(SLOT.hourMedian, 'float32', ZONE_COUNT),
              standardDeviations: slots.view(SLOT.hourSd, 'float32', ZONE_COUNT),
              percentiles: percentileView,
              modes: slots.view(SLOT.hourMode, 'float32', ZONE_COUNT),
              uniqueCounts: slots.view(SLOT.hourUnique, 'uint32', ZONE_COUNT),
              skewness: slots.view(SLOT.hourSkew, 'float32', ZONE_COUNT),
              kurtosis: slots.view(SLOT.hourKurtosis, 'float32', ZONE_COUNT),
              minimums: slots.view(SLOT.hourMinimum, 'float32', ZONE_COUNT),
              maximums: slots.view(SLOT.hourMaximum, 'float32', ZONE_COUNT),
              zScores: float('hour-z-scores', zScoreBuffer, observationCount)
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
            values: float('species', speciesBuffer, observationCount),
            statistics: ['uniqueCount'],
            output: {uniqueCounts: slots.view(SLOT.speciesUnique, 'uint32', ZONE_COUNT)}
          },
          {
            values: float('introduced', introducedBuffer, observationCount),
            statistics: ['mean'],
            output: {means: slots.view(SLOT.introducedMean, 'float32', ZONE_COUNT)}
          }
        ],
        output: {
          keys: word('output-keys', outputKeysBuffer, ZONE_COUNT),
          counts: slots.view(SLOT.counts, 'uint32', ZONE_COUNT),
          count: word('output-count', outputCountBuffer, 1),
          overflow: word('output-overflow', outputOverflowBuffer, 1)
        }
      })
    );

    // 1:n join: attach the sums and means of the census tracts inside every community area.
    const populationSum = float('area-population', populationSumBuffer, ZONE_COUNT);
    const incomeMean = float('area-income', incomeMeanBuffer, ZONE_COUNT);
    const povertyMean = float('area-poverty', povertyMeanBuffer, ZONE_COUNT);
    const tractCountView = float('area-tract-count', tractCountBuffer, ZONE_COUNT);
    graph.add(
      new GPUKeyJoin({
        id: 'tracts-into-areas',
        leftKeys: word('area-keys', areaKeysBuffer, ZONE_COUNT),
        rightKeys: word('tract-zone', tractZoneBuffer, tractCount),
        aggregates: [
          {
            operation: 'sum',
            column: float('tract-population', tractPopulation, tractCount),
            output: populationSum
          },
          {
            operation: 'mean',
            column: float('tract-income', tractIncome, tractCount),
            output: incomeMean
          },
          {
            operation: 'mean',
            column: float('tract-poverty', tractPoverty, tractCount),
            output: povertyMean
          },
          {operation: 'count', output: tractCountView}
        ],
        output: {}
      })
    );

    // The chosen metric per area (feature-aligned), a keep mask and the categorical copy.
    const areaMetric = float('area-metric', areaMetricBuffer, ZONE_COUNT);
    const areaKeep = word('area-keep', areaKeepBuffer, ZONE_COUNT);
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
        {name: 'incomes', view: incomeMean, type: 'f32', access: 'read'},
        {name: 'poverties', view: povertyMean, type: 'f32', access: 'read'},
        {name: 'tractCounts', view: tractCountView, type: 'f32', access: 'read'},
        {name: 'areaMetric', view: areaMetric, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  // selection: [kind, slot or index, scale bits, unused]; kind 0 float slot, 1 unsigned slot,
  // 2 percentile index, 3 derived index.
  let kind = selection[selectionOffset];
  let slot = selection[selectionOffset + 1u];
  let scale = bitcast<f32>(selection[selectionOffset + 2u]);
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
    if (slot == 0u) {
      value = select(nan, f32(observationRows) * 1000.0 / population, population > 0.0);
    } else if (slot == 1u) {
      value = incomes[incomesOffset + index];
    } else if (slot == 2u) {
      value = poverties[povertiesOffset + index];
    } else if (slot == 3u) {
      value = tractCounts[tractCountsOffset + index];
    } else {
      value = population;
    }
  }
  if (observationRows < minimum[minimumOffset]) {
    value = nan;
  }
  areaMetric[areaMetricOffset + index] = value;`
    });
    addKernelPass(graph, {
      id: 'area-category-and-keep',
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
        {
          name: 'areaCategory',
          view: word('area-category', areaCategoryBuffer, ZONE_COUNT),
          type: 'u32',
          access: 'read_write'
        },
        {name: 'areaKeep', view: areaKeep, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let observationRows = metrics[metricsOffset + ${SLOT.counts}u * ${ZONE_CAPACITY}u + index];
  let mode = bitcast<f32>(metrics[metricsOffset + ${SLOT.categoryMode}u * ${ZONE_CAPACITY}u + index]);
  var category = 0xffffffffu;
  if (observationRows >= minimum[minimumOffset] && observationRows > 0u) {
    category = u32(max(mode, 0.0));
  }
  areaCategory[areaCategoryOffset + index] = category;
  areaKeep[areaKeepOffset + index] = select(0u, 1u, observationRows >= minimum[minimumOffset]);`
    });

    // Gather join: paint the area statistic on every tract of the area (left or inner join).
    graph.add(
      new GPUKeyJoin({
        id: 'areas-onto-tracts',
        kind: options.joinKind,
        leftKeys: word('tract-zone-left', tractZoneBuffer, tractCount),
        rightKeys: word('area-keys-right', areaKeysBuffer, ZONE_COUNT),
        rightMask: areaKeep,
        gather: [
          {column: areaMetric, output: float('tract-metric', tractMetricBuffer, tractCount)},
          {
            column: word('area-category', areaCategoryBuffer, ZONE_COUNT),
            output: word('tract-category', tractCategoryBuffer, tractCount)
          }
        ],
        output: {
          matched: word('tract-matched', tractMatchedBuffer, tractCount),
          ...(options.joinKind === 'inner'
            ? {
                rows: {
                  ids: word('inner-rows', innerRowsBuffer, tractCount),
                  count: word('inner-count', innerCountBuffer, 1),
                  overflow: word('inner-overflow', innerOverflowBuffer, 1)
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

  let dirty = true;
  let mapRange: [number, number] = [0, 1];
  let selectedZone = -1;
  type Latest = {
    metrics: Float32Array;
    metricWords: Uint32Array;
    areaMetric: Float32Array;
    population: Float32Array;
    income: Float32Array;
    poverty: Float32Array;
    tractCount: Float32Array;
  };
  let latest: Latest | null = null;

  const getMetric = () =>
    GROUP_METRICS.find(entry => entry.id === ctx.options.metric) ?? GROUP_METRICS[0];

  const writeParameters = () => {
    const options = ctx.options;
    const categoryIndex = categoryNames.findIndex(name => name === options.natureGroup);
    maskParameters.write(
      Uint32Array.of(
        categoryIndex >= 0 ? categoryIndex : NO_KEY,
        options.hours[0],
        options.hours[1],
        options.gradeFilter === 'all' ? 0 : options.gradeFilter === 'research' ? 1 : 2
      )
    );
    percentileParameters.write(Float32Array.of(options.lowerFraction, 0.5, options.upperFraction));
    const metric = getMetric();
    const kind = {slot: 0, unsigned: 1, percentile: 2, derived: 3, category: 0}[metric.kind];
    const parameter =
      metric.kind === 'percentile' || metric.kind === 'derived'
        ? (metric.index ?? 0)
        : (metric.slot ?? 0);
    metricParameters.write(
      Uint32Array.of(
        kind,
        parameter,
        new Uint32Array(Float32Array.of(metric.scale ?? 1).buffer)[0],
        0
      )
    );
    minimumParameters.write(Uint32Array.of(options.minimumObservations, 0, 0, 0));
    dirty = true;
    reader.markStale();
  };

  const describeArea = (zone: number): string => {
    const lines = [areaName(zone)];
    if (!latest) return lines.join('\n');
    const f = (slot: number) => latest!.metrics[slot * ZONE_CAPACITY + zone];
    const u = (slot: number) => latest!.metricWords[slot * ZONE_CAPACITY + zone];
    const rows = u(SLOT.counts);
    lines.push(`${formatNumber(rows)} observations in the filter`);
    if (rows === 0) return lines.join('\n');
    const percentileBase = SLOT.hourPercentiles * ZONE_CAPACITY + zone * PERCENTILE_COUNT;
    const pl = latest.metrics[percentileBase];
    const ph = latest.metrics[percentileBase + 2];
    lines.push(
      `Hour: mean ${f(SLOT.hourMean).toFixed(1)}, median ${f(SLOT.hourMedian).toFixed(1)}, mode ${f(SLOT.hourMode).toFixed(0)}, sd ${f(SLOT.hourSd).toFixed(1)}`,
      `Hour P${Math.round(ctx.options.lowerFraction * 100)} ${pl.toFixed(1)}, P${Math.round(ctx.options.upperFraction * 100)} ${ph.toFixed(1)}; range ${f(SLOT.hourMinimum)}-${f(SLOT.hourMaximum)}; skew ${f(SLOT.hourSkew).toFixed(2)}, kurtosis ${f(SLOT.hourKurtosis).toFixed(2)}`,
      `Research grade ${formatNumber(f(SLOT.gradeSum))} (${(100 * f(SLOT.gradeMean)).toFixed(1)}%), introduced ${(100 * f(SLOT.introducedMean)).toFixed(1)}%`,
      `${u(SLOT.categoryUnique)} groups, ${u(SLOT.speciesUnique)} taxa, most common group: ${formatCategory(categoryNames[f(SLOT.categoryMode)] ?? '?')}`
    );
    const population = latest.population[zone];
    lines.push(
      `Population ${formatNumber(population)} in ${latest.tractCount[zone]} tracts: ${formatNumber((rows * 1000) / Math.max(population, 1), 1)} observations per 1,000, income ${formatCompact(latest.income[zone])}, poverty ${latest.poverty[zone].toFixed(1)}%`
    );
    return lines.join('\n');
  };

  const reader = new SummaryReader(
    resources,
    'group-statistics',
    [
      {buffer: metricsBuffer, size: SLOT_COUNT * ZONE_CAPACITY * 4},
      {buffer: areaMetricBuffer, size: ZONE_COUNT * 4},
      {buffer: populationSumBuffer, size: ZONE_COUNT * 4},
      {buffer: incomeMeanBuffer, size: ZONE_COUNT * 4},
      {buffer: povertyMeanBuffer, size: ZONE_COUNT * 4},
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
      const income = read.floats(ZONE_COUNT).slice();
      const poverty = read.floats(ZONE_COUNT).slice();
      const tractCounts = read.floats(ZONE_COUNT).slice();
      const matched = read.words(tractCount);
      const innerCount = read.words(1)[0];
      const overflow = read.words(1)[0];
      const groups = read.words(1)[0];
      latest = {
        metrics,
        metricWords,
        areaMetric,
        population,
        income,
        poverty,
        tractCount: tractCounts
      };
      const options = ctx.options;
      const metric = getMetric();
      const sorted = getSortedFinite(areaMetric);
      mapRange =
        metric.kind === 'category'
          ? [0, 1]
          : [getQuantile(sorted, 0.02), getQuantile(sorted, 0.98)];
      if (!(mapRange[1] > mapRange[0])) mapRange = [mapRange[0], mapRange[0] + 1e-6];
      const modalCounts = new Map<number, number>();
      if (metric.kind === 'category') {
        for (let zone = 0; zone < ZONE_COUNT; zone++) {
          if (Number.isFinite(areaMetric[zone])) {
            const index = Math.round(areaMetric[zone]);
            modalCounts.set(index, (modalCounts.get(index) ?? 0) + 1);
          }
        }
      }
      setLegendData<GroupStatisticsLegend>('group-statistics', {
        range: mapRange,
        modalGroups: [...modalCounts.entries()]
          .sort((a, b) => b[1] - a[1])
          .map(([index, areasCount]) => ({
            index,
            name: formatCategory(categoryNames[index] ?? '?'),
            areas: areasCount
          }))
      });
      ctx.setLegendExtent('value', mapRange);
      const total = metricWords
        .slice(SLOT.counts * ZONE_CAPACITY, SLOT.counts * ZONE_CAPACITY + ZONE_COUNT)
        .reduce((sum, value) => sum + value, 0);
      ctx.setReadout(
        'observations',
        `${formatCount(total)} of ${formatCount(observationCount)} observations pass the filter`
      );
      ctx.setReadout(
        'groups',
        `${groups} community areas in the table${overflow ? ' (CAPACITY OVERFLOW)' : ''}`
      );
      const ranked = [...areaMetric.keys()]
        .filter(zone => Number.isFinite(areaMetric[zone]))
        .sort((a, b) => areaMetric[b] - areaMetric[a]);
      const show = (value: number) =>
        metric.kind === 'category'
          ? formatCategory(categoryNames[Math.round(value)] ?? '?')
          : `${formatCompact(value)} ${metric.unit}`;
      ctx.setReadout(
        'highest',
        ranked
          .slice(0, 3)
          .map(zone => `${areaName(zone)} ${show(areaMetric[zone])}`)
          .join('; ') || '-'
      );
      ctx.setReadout(
        'lowest',
        ranked
          .slice(-3)
          .reverse()
          .map(zone => `${areaName(zone)} ${show(areaMetric[zone])}`)
          .join('; ') || '-'
      );
      const matchedCount = matched.reduce((sum, value) => sum + value, 0);
      ctx.setReadout(
        'join',
        options.joinKind === 'inner'
          ? `inner join keeps ${formatNumber(innerCount)} of ${formatNumber(tractCount)} tracts`
          : `left join: ${formatNumber(matchedCount)} of ${formatNumber(tractCount)} tracts matched an area with at least ${options.minimumObservations} observations`
      );
      if (selectedZone >= 0)
        ctx.setReadout('selected', describeArea(selectedZone).replace(/\n/g, ' | '));
      ctx.requestLayers();
    }
  );

  writeParameters();
  ctx.setReadout('selected', null);

  const getZoneOfTract = (row: number) => tractZone[row];

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (id === 'variance' || id === 'joinKind') {
        active = getVariant(ctx.options);
        dirty = true;
        reader.markStale();
      }
      if (id === 'level' || id === 'showObservations' || id === 'outlines') {
        ctx.requestLayers();
      } else {
        writeParameters();
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

    getLayers() {
      const {level, outlines, showObservations} = ctx.options;
      const metric = getMetric();
      const isCategory = metric.kind === 'category';
      const geometry = level === 'areas' ? areaGeometry : tractGeometry;
      const values =
        level === 'areas'
          ? isCategory
            ? areaCategoryBuffer
            : areaMetricBuffer
          : isCategory
            ? tractCategoryBuffer
            : tractMetricBuffer;
      const layers: Layer[] = [
        geometry.createFillLayer(`group-fill-${level}`, {
          values,
          mode: isCategory ? 'category' : 'ramp',
          ramp:
            metric.id === 'hourMean' || metric.id === 'hourMedian' || metric.id === 'hourMode'
              ? 'cividis'
              : 'magma',
          valueRange: mapRange,
          palette: NATURE_CATEGORY_COLORS,
          noDataColor: [128, 128, 128, 55],
          fillOpacity: showObservations ? 0.55 : 0.9
        }),
        geometry.createOutlineLayer(
          `group-outline-${level}`,
          getOutlineColor(ctx.theme(), level === 'areas' || outlines ? 120 : 40),
          level === 'areas' ? 1 : 0.5
        )
      ];
      if (showObservations) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'group-observations',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: positionsBuffer,
            instanceCount: observationCount,
            values: zScoreBuffer,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange: [-2.5, 2.5],
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 1.4,
            color: [255, 255, 255, 170]
          })
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const zone = areaGeometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      if (zone < 0) return null;
      if (ctx.options.level === 'tracts') {
        const row = tractGeometry.locator.locate(event.coordinate[0], event.coordinate[1]);
        if (row >= 0) {
          const geoid = tractFeatures[row]?.properties?.GEOID;
          return `Tract ${geoid ?? row} (${formatNumber(populationColumn[row])} residents)\n${describeArea(getZoneOfTract(row) < ZONE_COUNT ? getZoneOfTract(row) : zone)}`;
        }
      }
      return describeArea(zone);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const zone = areaGeometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      selectedZone = zone === selectedZone ? -1 : zone;
      ctx.setReadout(
        'selected',
        selectedZone >= 0 ? describeArea(selectedZone).replace(/\n/g, ' | ') : null
      );
      return true;
    },

    destroy() {
      reader.stop();
      clearLegendData('group-statistics');
      resources.destroy();
    }
  };
}
