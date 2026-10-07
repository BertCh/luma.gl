// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {EFFORT_CAVEAT} from '../../cartography/hue-registry';
import {defineScene, type LegendSpec} from '../scene';
import {GROUP_METRICS} from './b5-group-metrics';
import type {GroupStatisticsLegend, GroupStatisticsOptions} from './group-statistics.compute';
import {getMetricLegendCopy} from './group-statistics.style';

const NATURE_GROUPS = [
  'Plants',
  'Birds',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles',
  'Snails and mussels',
  'Fish',
  'Other life'
] as const;

/** The lakefront of the North Side, where the busiest areas are. */
const NORTH_LAKEFRONT_BOUNDS = [-87.72, 41.86, -87.58, 42.02] as const;
/** The North Side, wide enough to show tracts of the busiest areas. */
const NORTH_SIDE_BOUNDS = [-87.8, 41.9, -87.6, 42.02] as const;

const CHICAGO_FRAME = {...CITY_FRAMES.chicago, transitionMs: 1400};

const CREDIT = joinCredits(
  CREDITS.iNaturalist,
  CREDITS.cityOfChicago,
  'US Census ACS 2018-2022 via CDC SVI 2022 (public domain)',
  CREDITS.colorBrewer
);

/** The cartouche of one step (the standing sample line is set from the data at create). */
const cartouche = (
  title: string,
  subtitle: string,
  chips: readonly string[] = ['Observer effort']
) => ({title, subtitle, chips});

/** Names that orient the reader in every step, from the Chicago gazetteer (drawn above the data). */
const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan']);

function getLegends(
  state: GroupStatisticsOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const shared = data['groupStatistics'] as GroupStatisticsLegend | undefined;
  if (!shared) return [];
  const legends: LegendSpec[] = [];
  const metric =
    GROUP_METRICS.find(
      entry => entry.id === (state.zoning === 'areas' ? state.metric : 'perThousand')
    ) ?? GROUP_METRICS[0];
  if (state.zoning === 'areas' && state.view === 'counts') {
    legends.push({
      kind: 'size',
      title: 'Records per area',
      layout: 'nested',
      entries: shared.circleEntries,
      color: shared.circleColor,
      unit: 'records',
      note: 'Circle area is proportional to the count, on one scale for every area.'
    });
    return legends;
  }
  const copy = getMetricLegendCopy(metric);
  if (metric.family === 'hour' && state.zoning === 'areas') {
    legends.push({
      kind: 'cyclic',
      title: 'Median hour of the day',
      colors: shared.hourColors,
      labels: ['0:00', '6:00', '12:00', '18:00'],
      note: 'Three-hour classes; the ring wraps at midnight.'
    });
    if (state.minimumObservations > 0) {
      legends.push({
        kind: 'categories',
        title: 'Withheld',
        entries: [
          {
            color: shared.noDataColor,
            label: `Fewer than ${state.minimumObservations} observations`,
            shape: 'hatch'
          }
        ]
      });
    }
    return legends;
  }
  const swipeNote =
    state.zoning === 'swipe'
      ? ' Counts are for the tracts on the right; both sides share the classes.'
      : state.zoning === 'tracts'
        ? ' Counts are tracts.'
        : '';
  legends.push(
    getClassTableLegend(shared.table, {
      title: copy.title,
      id: 'classes',
      basis: copy.basis,
      counts: shared.counts,
      interactive: true,
      layout: 'list',
      note: `${shared.table.method ?? ''}${copy.note ? ` ${copy.note}` : ''}${swipeNote}`.trim()
    })
  );
  return legends;
}

function getSnippet(state: GroupStatisticsOptions): string {
  return `import {GPUGroupStatistics, GPUKeyJoin} from '@luma.gl/experimental/gpu-dataframe';

// keys: community area index per record (0xffffffff = none); a dense table of 77 groups.
// mask: one word per record, rewritten from the filters (group ${state.natureGroup}, hours ${state.hours[0]} to ${state.hours[1]}, identification ${state.gradeFilter}).
graph.add(new GPUGroupStatistics({
  keys: observationArea, mask: filterMask, keyCount: 77,
  variance: '${state.variance}', percentiles,        // fractions [${state.lowerFraction}, 0.5, ${state.upperFraction}] are a per-frame buffer
  columns: [
    {values: hour, statistics: ['mean', 'median', 'standardDeviation', 'percentiles', 'mode'], output: hourOutput},
    {values: researchGrade, statistics: ['sum', 'mean'], output: gradeOutput},
    {values: category, statistics: ['mode', 'uniqueCount'], output: categoryOutput},
    {values: taxon, statistics: ['uniqueCount'], output: taxonOutput},
    {values: introduced, statistics: ['mean'], output: introducedOutput}
  ],
  output: {keys, counts, count, overflow}
}));

// 1:n join: sum what lives in the tracts of each area. Income is joined as income x residents,
// so the area mean is weighted by people.
graph.add(new GPUKeyJoin({
  leftKeys: areaIds, rightKeys: tractArea,
  aggregates: [
    {operation: 'sum', column: population, output: populationSum},
    {operation: 'sum', column: incomeTimesPopulation, output: incomeSum},
    {operation: 'sum', column: populationWithIncome, output: incomePopulation},
    {operation: 'count', output: tractCount}
  ],
  output: {}
}));

// Gather the area value onto every tract (${state.joinKind} join). Areas under ${state.minimumObservations} records are masked out.
graph.add(new GPUKeyJoin({
  kind: '${state.joinKind}', leftKeys: tractArea, rightKeys: areaIds,
  rightMask: areaHasEnoughRecords,
  gather: [{column: areaValue, output: tractValue}],
  output: {matched${state.joinKind === 'inner' ? ',\n    rows: {ids, count, overflow}' : ''}}
}));`;
}

export default defineScene<GroupStatisticsOptions>({
  id: 'group-statistics',
  title: 'Wildlife by community area, joined to who lives there',
  chapter: 'statistics',
  order: 3,
  summary:
    'Group iNaturalist records by community area on the GPU, divide them by the residents a key join adds up, withhold the small groups, and see how the answer changes with the filter and with the zoning.',
  contributors: ['GPUGroupStatistics', 'GPUKeyJoin'],
  datasets: [
    {
      id: 'chicago-nature',
      role: 'iNaturalist records with hour, group, taxon, research grade, introduced flag and community area'
    },
    {id: 'chicago-community-areas', role: 'the 77 groups'},
    {id: 'chicago-tracts', role: 'census tracts with population, income and records per tract'}
  ],
  initialView: {...CITY_FRAMES.chicago},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Show',
      group: 'Map',
      apply: 'param',
      display: 'segmented',
      default: 'rates',
      disabledWhen: state => state.zoning !== 'areas',
      help: 'Counts are drawn as circles sized by the records of each area, over unfilled areas. The statistic below is drawn as a classed fill. Both read the same GPU group table; this only picks which one is drawn.',
      options: [
        {value: 'counts', label: 'Records'},
        {value: 'rates', label: 'Per 1,000 residents'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Statistic',
      group: 'Map',
      apply: 'param',
      default: 'perThousand',
      disabledWhen: state => state.zoning !== 'areas',
      help: 'Every statistic is computed in one graph run; a small kernel picks the slot the layer reads, so switching never recompiles. Tracts always show the rate.',
      options: GROUP_METRICS.map(metric => ({
        value: metric.id,
        label: metric.label,
        help: metric.help
      }))
    },
    {
      kind: 'select',
      id: 'classification',
      label: 'Rate classes',
      group: 'Map',
      apply: 'param',
      display: 'segmented',
      default: 'manual',
      disabledWhen: state => state.metric !== 'perThousand' || state.zoning !== 'areas',
      help: 'Manual breaks are roughly logarithmic because the rates are heavy-tailed. Quantile and equal-interval breaks are computed once from the unfiltered rates and then frozen, so filters change colours, never classes.',
      options: [
        {value: 'manual', label: 'Manual'},
        {value: 'quantile', label: 'Quantile'},
        {value: 'equal', label: 'Equal'}
      ]
    },
    {
      kind: 'select',
      id: 'zoning',
      label: 'Zoning',
      group: 'Map',
      apply: 'param',
      display: 'segmented',
      default: 'areas',
      help: 'Community areas are the groups. Tracts show the records of each tract over its residents, on the same classes. The swipe draws both with a divider.',
      options: [
        {value: 'areas', label: 'Areas'},
        {value: 'tracts', label: 'Tracts'},
        {value: 'swipe', label: 'Swipe'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumObservations',
      label: 'Minimum observations',
      group: 'Small groups',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 0,
      unit: 'records',
      describe: value =>
        value === 0
          ? 'No area is withheld'
          : `Areas with fewer than ${value} records are withheld and hatched`,
      help: 'A statistic computed from a handful of records says little. Areas under this many records in the filter are withheld (NaN in the display kernel), hatched on the map, and masked out of the tract join.'
    },
    {
      kind: 'select',
      id: 'natureGroup',
      label: 'Group',
      group: 'Filter (GPU mask)',
      apply: 'param',
      default: 'ALL',
      help: 'A kernel builds the row mask from these options; GPUGroupStatistics skips masked rows. A handful of parameter words, no rebuild.',
      options: [
        {value: 'ALL', label: 'All groups'},
        ...NATURE_GROUPS.map(group => ({value: group, label: group}))
      ]
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hour of day',
      group: 'Filter (GPU mask)',
      apply: 'param',
      min: 0,
      max: 24,
      step: 1,
      default: [0, 24],
      format: value => `${value}:00`,
      help: 'Keep records from the first hour up to, not including, the second.'
    },
    {
      kind: 'select',
      id: 'gradeFilter',
      label: 'Identification',
      group: 'Filter (GPU mask)',
      apply: 'param',
      default: 'all',
      help: 'Keep only records whose identification the community has confirmed (research grade), or only those not yet confirmed.',
      options: [
        {value: 'all', label: 'All records'},
        {value: 'research', label: 'Research grade'},
        {value: 'unconfirmed', label: 'Not yet confirmed'}
      ]
    },
    {
      kind: 'slider',
      id: 'lowerFraction',
      label: 'Lower percentile',
      group: 'Hour statistics',
      apply: 'param',
      min: 0.05,
      max: 0.45,
      step: 0.05,
      default: 0.1,
      format: value => `P${Math.round(value * 100)}`,
      help: 'The lower percentile of the observation hour. The fractions of GPUGroupStatistics are a per-frame parameter buffer; the rose chart colours the hours between the two percentiles.'
    },
    {
      kind: 'slider',
      id: 'upperFraction',
      label: 'Upper percentile',
      group: 'Hour statistics',
      apply: 'param',
      min: 0.55,
      max: 0.95,
      step: 0.05,
      default: 0.9,
      format: value => `P${Math.round(value * 100)}`,
      help: 'The upper percentile of the observation hour.'
    },
    {
      kind: 'select',
      id: 'variance',
      label: 'Variance kind',
      group: 'Hour statistics',
      apply: 'compile',
      display: 'segmented',
      default: 'sample',
      expert: true,
      help: 'Sample (n - 1, as d3 and kepler.gl) or population (n) variance behind the spread of the hour in the tooltip. Compile-time: compiles another graph.',
      options: [
        {value: 'sample', label: 'Sample (n - 1)'},
        {value: 'population', label: 'Population (n)'}
      ]
    },
    {
      kind: 'select',
      id: 'joinKind',
      label: 'Join kind (areas onto tracts)',
      group: 'Key join',
      apply: 'compile',
      display: 'segmented',
      default: 'left',
      expert: true,
      help: 'Left keeps every tract aligned with its row; inner also compacts the matched tract rows into a list. Both gather the area statistic, and areas below the minimum are masked out. Compile-time.',
      options: [
        {value: 'left', label: 'Left'},
        {value: 'inner', label: 'Inner'}
      ]
    }
  ],

  readouts: [
    {
      id: 'observations',
      label: 'Records in the filter',
      help: 'Rows that pass the GPU mask and carry an area key. Records outside every area get no key and are skipped.'
    },
    {
      id: 'citywideRate',
      label: 'Citywide rate',
      emphasis: 'tile',
      help: 'All records in the filter over all residents: the number every area is compared with.'
    },
    {
      id: 'medianRate',
      label: 'Median area rate',
      help: 'The middle of the area rates: the typical area, usually below the citywide rate because a few busy areas pull the city figure up.'
    },
    {id: 'threshold', label: 'Minimum observations'},
    {
      id: 'suppressedAreas',
      label: 'Areas withheld',
      emphasis: 'tile',
      help: 'Areas with fewer records in the filter than the minimum.'
    },
    {
      id: 'suppressedShare',
      label: 'Share of records in them',
      help: 'The records of the withheld areas, as a share of all records in the filter.'
    },
    {
      id: 'suppressedPeople',
      label: 'Residents of the withheld areas',
      emphasis: 'tile',
      help: 'Residents the join adds up over the tracts of the withheld areas.'
    },
    {
      id: 'runs',
      label: 'Records per area',
      kind: 'chart',
      help: 'The sorted runs: one bar per group, largest first, on a log scale. Grey bars are withheld. Click a bar to outline the area.'
    },
    {
      id: 'hours',
      label: 'Records by hour',
      kind: 'chart',
      help: 'The selected area, or the busiest one when none is selected, against the citywide shape. Click an area on the map to change it.'
    },
    {
      id: 'hourSpan',
      label: 'Hours between the percentiles',
      help: 'The lower and upper percentile of the hour in the selected area, from the percentile sliders.'
    },
    {
      id: 'earliestMedian',
      label: 'Earliest median hour',
      help: 'Among the areas that are not withheld.'
    },
    {
      id: 'latestMedian',
      label: 'Latest median hour',
      help: 'Among the areas that are not withheld.'
    },
    {
      id: 'spearman',
      label: 'Rank correlation, income and rate',
      emphasis: 'tile',
      help: "Spearman's rho between population-weighted income and records per 1,000 residents, over the areas that are not withheld. An association between areas, not a cause."
    },
    {
      id: 'joinResult',
      label: 'Key join result',
      help: 'How many tracts the gather join matched to an area that passes the minimum (left join), or kept (inner join).'
    },
    {
      id: 'incomeScatter',
      label: 'Income against rate',
      kind: 'chart',
      help: 'One dot per area that is not withheld: weighted income on x, records per 1,000 residents on a log y axis. Click a dot to find the area.'
    },
    {
      id: 'tractMax',
      label: 'Highest tract rate',
      emphasis: 'tile',
      help: 'The largest records-per-1,000 value of any tract. Tracts with few residents swing widely.'
    },
    {
      id: 'notTopInside',
      label: 'Tracts below the top class inside top-class areas',
      help: 'Tracts whose rate is under the top class although their community area is in it.'
    },
    {
      id: 'topOutside',
      label: 'Top-class tracts outside top-class areas',
      help: 'Tracts in the top class that sit in an area whose average rate is lower.'
    },
    {id: 'emptyTracts', label: 'Tracts with no record at all'},
    {
      id: 'selected',
      label: 'Selected area',
      help: 'Click an area to pin it; hover shows its statistics too.'
    },
    {
      id: 'groups',
      label: 'Groups in the table',
      hood: true,
      help: 'Rows of the dense group table (community areas 1 to 77), and the capacity flag.'
    }
  ],

  pipeline: [
    {id: 'mask', label: 'Mask', detail: 'A kernel turns the filters into one mask word per record'},
    {
      id: 'sort',
      label: 'Sort by area',
      detail: 'A stable radix sort puts each area in one run of rows'
    },
    {
      id: 'runs',
      label: 'Runs',
      detail: 'One binary search per key finds the run; its length is the count'
    },
    {
      id: 'stats',
      label: 'Stats',
      detail: 'Exact counts and sums; a value sort per group for median, percentiles and mode'
    },
    {
      id: 'join',
      label: 'Join',
      detail: 'Sort the tracts by key, sum onto the areas, gather the area value back'
    }
  ],

  legends: getLegends,

  basemap: ground('paperCity'),
  furniture: {
    title: cartouche('Where is nature logged, per resident?', 'Records per area, 2023'),
    credit: CREDIT,
    caveat: EFFORT_CAVEAT
  },
  annotations: ORIENTATION,

  snippet: getSnippet,

  about: {
    what: '`GPUGroupStatistics` groups rows by a 32- or 64-bit key and computes statistics per group on the GPU: counts, sums, means, deviations, median, percentiles, mode and distinct counts. Here the key is the community area of a record and the table is dense, so empty areas keep a row. `GPUKeyJoin` attaches a right table to a left table by key, as a 1:1 gather or a 1:n aggregate, left or inner: it adds up the residents and the income of the tracts of each area, and paints the area value back onto the tracts.',
    why: 'Almost every dashboard is "group by, then join": records by area, rates by population, areas back onto tracts. Doing it on the GPU keeps the table next to the data, so a filter regroups every record in a frame and only a few kilobytes are read back.',
    howToRead:
      'Circles are counts, sized by area; fills are rates, classed. Hatched areas are withheld because too few records stand behind them. Class breaks are computed once from the unfiltered table, so colours change with a filter and classes do not. The hour is treated as a plain number, not a circle, which is fine for a daytime pastime and wrong for night-time groups. Population comes from the American Community Survey (estimates), the denominator of every rate: records follow observers, so a rate here measures how much a place is watched as well as what lives there. Matches pandas groupby and merge, numpy percentiles and d3 statistics.'
  },

  create: async ctx => (await import('./group-statistics.compute')).createGroupStatistics(ctx),

  story: [
    {
      id: 'counts-vs-rates',
      title: 'Counting is not comparing',
      headline: 'Counts follow parks and people; rates divide',
      textAlternative:
        'Map of the Chicago community areas with proportional circles for the number of nature records in each: the largest circles lie on the North Side. Switching to rates fills each area from pale yellow to dark blue.',
      body: 'Each circle is a community area, sized by the records logged there (**{{observations}}**). Switch **Show** to *Per 1,000 residents*: the same records divided by the people who live there, against **{{citywideRate}}** citywide, with a median area at **{{medianRate}}**. The busiest area is not the most watched per resident.\n\n*Counts need symbols; rates earn a fill.*',
      options: {view: 'counts', metric: 'perThousand', zoning: 'areas', minimumObservations: 0},
      optionsMode: 'fresh',
      controls: ['view'],
      readouts: ['observations', 'citywideRate', 'medianRate'],
      camera: CHICAGO_FRAME,
      basemap: ground('paperCity'),
      furniture: {
        title: cartouche(
          'Where is nature logged, per resident?',
          'Records per area, then per 1,000 residents, 2023'
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop'], {loop: {minZoom: 10.2}}),
      stage: 'runs'
    },
    {
      id: 'group-by',
      title: 'One sort turns rows into groups',
      headline: 'Small groups are too small to trust',
      textAlternative:
        'Chicago community areas in green by research-grade share; hatched grey areas are withheld because they hold too few records. A bar chart ranks the areas by records.',
      body: "`GPUGroupStatistics` sorts the records by area key, so every area becomes one run, then reads each run. The map shows each area's research-grade share. Some runs are short: drag **Minimum observations** and areas under **{{threshold}}** are withheld and hatched, **{{suppressedAreas}}**, holding **{{suppressedShare}}** but **{{suppressedPeople}}**. Why small numbers mislead is in *Small numbers, loud maps*.\n\n*Withhold what the data cannot support.*",
      options: {view: 'rates', metric: 'researchShare', zoning: 'areas', minimumObservations: 30},
      optionsMode: 'fresh',
      controls: ['minimumObservations'],
      readouts: ['suppressedAreas', 'suppressedShare', 'suppressedPeople', 'runs'],
      camera: CHICAGO_FRAME,
      furniture: {
        title: cartouche(
          'Which groups are too small to trust?',
          'Research-grade share of the records; small areas withheld'
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      stage: 'sort'
    },
    {
      id: 'time-of-day',
      title: 'When do people go out looking?',
      headline: 'Neighbourhoods look at nature at different hours',
      textAlternative:
        'North Side areas coloured by median hour of day on a cyclic ring of three-hour classes, with hatched grey areas withheld; a rose chart shows the records of one area by hour.',
      body: 'Each area gets its median hour, coloured on a ring because late evening and early morning are neighbours. It runs from **{{earliestMedian}}** to **{{latestMedian}}**; areas under **{{threshold}}** are withheld, so no median rests on a handful of records. Click an area, then set **Lower percentile** and **Upper percentile**: the rose colours **{{hourSpan}}**.\n\n*Time of day is cyclic: its legend is a ring.*',
      options: {
        view: 'rates',
        metric: 'hourMedian',
        zoning: 'areas',
        minimumObservations: 100,
        lowerFraction: 0.1,
        upperFraction: 0.9
      },
      optionsMode: 'fresh',
      controls: ['lowerFraction', 'upperFraction'],
      readouts: ['hours', 'hourSpan', 'earliestMedian', 'latestMedian'],
      camera: {bounds: NORTH_LAKEFRONT_BOUNDS, transitionMs: 1600},
      furniture: {
        title: cartouche(
          'When do people go out looking?',
          'Median hour of day per area; small areas withheld'
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      stage: 'stats'
    },
    {
      id: 'join',
      title: 'Who lives where people look?',
      headline: 'Wealthier areas tend to log more per resident',
      textAlternative:
        'Chicago community areas in blue by population-weighted per-capita income, with a scatter plot of income against records per 1,000 residents, one dot per area.',
      body: '`GPUKeyJoin` sums residents and income times residents over the tracts of each area, so income is weighted by people: a plain mean of tract incomes would count a tiny tract like a crowded one. Flip **Statistic** to compare maps, and raise **Minimum observations**. Rank correlation with the rate: **{{spearman}}**; the gather back onto tracts: **{{joinResult}}**.\n\n*An area average is not a person (ecological fallacy).*',
      options: {view: 'rates', metric: 'income', zoning: 'areas', minimumObservations: 30},
      optionsMode: 'fresh',
      controls: ['metric', 'minimumObservations'],
      readouts: ['spearman', 'joinResult', 'incomeScatter'],
      camera: CHICAGO_FRAME,
      furniture: {
        title: cartouche(
          'Who lives where people look?',
          'Population-weighted per-capita income, ACS 2018-2022',
          ['Observer effort', 'Estimates']
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      stage: 'join'
    },
    {
      id: 'zoning',
      title: 'Do smaller zones tell the same story?',
      headline: 'The same records read differently at tract scale',
      textAlternative:
        'North Side split by a divider: community areas on the left and census tracts on the right, both classed by records per 1,000 residents on the same blue scale; many tracts inside dark areas are paler.',
      body: 'Same records, same classes, different zones. Drag the divider between areas and tracts (**Zoning** holds the other views): inside the darkest-class areas, **{{notTopInside}}** are themselves in a lighter class, while **{{topOutside}}** outside them reach the top class. The busiest tract reaches **{{tractMax}}**. This is the **modifiable areal unit problem**: the answer depends on the zoning.\n\n*The unit of analysis is a cartographic decision.*',
      options: {view: 'rates', metric: 'perThousand', zoning: 'swipe', minimumObservations: 0},
      optionsMode: 'fresh',
      controls: ['zoning'],
      readouts: ['tractMax', 'notTopInside', 'topOutside', 'emptyTracts'],
      camera: {bounds: NORTH_SIDE_BOUNDS, transitionMs: 1600},
      compare: {labels: ['Community areas', 'Census tracts'], position: 0.5},
      furniture: {
        title: cartouche(
          'Do smaller zones tell the same story?',
          'Records per 1,000 residents: areas | tracts, same classes',
          ['Observer effort', 'Estimates']
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      stage: 'join'
    },
    {
      id: 'filters',
      title: 'What if you only count some records?',
      headline: 'Filters change the rows; the groups follow',
      textAlternative:
        'Chicago community areas classed by records per 1,000 residents for birds logged in the early morning; most areas are hatched grey because too few records remain.',
      body: 'Choose a **Group**, an **Hour of day** range or an **Identification** grade: a kernel rewrites the row mask from a few parameter words and `GPUGroupStatistics` regroups **{{observations}}** in the next frame, with no rebuild. Rates move, and **{{suppressedAreas}}** now fall under the threshold.\n\n*Every filter is a new question; the groups answer it again.*',
      options: {
        view: 'rates',
        metric: 'perThousand',
        zoning: 'areas',
        natureGroup: 'Birds',
        hours: [5, 10],
        minimumObservations: 30
      },
      optionsMode: 'fresh',
      controls: ['natureGroup', 'hours', 'gradeFilter'],
      readouts: ['observations', 'suppressedAreas', 'citywideRate'],
      camera: CHICAGO_FRAME,
      furniture: {
        title: cartouche(
          'What if you only count some records?',
          'Records per 1,000 residents in the filtered slice'
        ),
        credit: CREDIT,
        caveat: EFFORT_CAVEAT
      },
      stage: 'mask'
    }
  ]
});
