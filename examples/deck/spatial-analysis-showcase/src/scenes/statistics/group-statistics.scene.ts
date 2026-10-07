// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {GROUP_METRICS} from './b5-group-metrics';
import {formatCompact, getLegendData} from './b5-legend-bus';
import {NATURE_CATEGORY_COLORS} from './b5-palettes';
import type {GroupStatisticsLegend, GroupStatisticsOptions} from './group-statistics.compute';

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

export default defineScene<GroupStatisticsOptions>({
  id: 'group-statistics',
  title: 'Wildlife by community area, joined to who lives there',
  chapter: 'statistics',
  order: 6,
  summary:
    'Group 43,600 Chicago nature observations by community area and compute a dozen statistics per area on the GPU, then join census attributes by key to turn counts into rates and paint area results back onto tracts.',
  contributors: ['GPUGroupStatistics', 'GPUKeyJoin'],
  datasets: [
    {
      id: 'chicago-nature',
      role: '43,557 observations with hour, group, species, research grade, introduced flag and community area'
    },
    {id: 'chicago-community-areas', role: 'the 77 groups'},
    {id: 'chicago-tracts', role: '791 tracts with population, income and poverty'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},

  options: [
    {
      kind: 'select',
      id: 'metric',
      label: 'Statistic mapped',
      group: 'Map',
      apply: 'param',
      default: 'perThousand',
      help: 'All statistics are computed in one graph run; a small kernel picks the slot the layer reads, so switching never recompiles.',
      options: GROUP_METRICS.map(metric => ({
        value: metric.id,
        label: metric.label,
        help: metric.help
      }))
    },
    {
      kind: 'select',
      id: 'level',
      label: 'Draw',
      group: 'Map',
      apply: 'param',
      default: 'areas',
      help: 'Community areas are the groups. Tracts show the result of a second GPUKeyJoin that gathers the area value onto every tract of the area.',
      options: [
        {value: 'areas', label: 'Community areas (the groups)'},
        {value: 'tracts', label: 'Census tracts (via key join)'}
      ]
    },
    {
      kind: 'select',
      id: 'natureGroup',
      label: 'Group',
      group: 'Filter (GPU mask)',
      apply: 'param',
      default: 'ALL',
      help: 'A GPU kernel builds the row mask from these options; GPUGroupStatistics skips masked rows. A four-word parameter write.',
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
      help: 'Keep observations from the first hour up to (not including) the second. Birders start early; insects and plants peak around midday.'
    },
    {
      kind: 'select',
      id: 'gradeFilter',
      label: 'Identification',
      group: 'Filter (GPU mask)',
      apply: 'param',
      default: 'all',
      help: 'Restrict to observations whose identification the community has confirmed (research grade), or not yet.',
      options: [
        {value: 'all', label: 'All observations'},
        {value: 'research', label: 'Research grade'},
        {value: 'unconfirmed', label: 'Not yet confirmed'}
      ]
    },
    {
      kind: 'select',
      id: 'variance',
      label: 'Variance kind',
      group: 'Statistics',
      apply: 'compile',
      default: 'sample',
      help: 'Sample (n - 1, as d3 and kepler.gl) or population (n) variance for the standard deviation and the per-row z-score. Compile-time: compiles another graph.',
      options: [
        {value: 'sample', label: 'Sample (n - 1)'},
        {value: 'population', label: 'Population (n)'}
      ]
    },
    {
      kind: 'slider',
      id: 'lowerFraction',
      label: 'Lower percentile of the hour',
      group: 'Statistics',
      apply: 'param',
      min: 0.05,
      max: 0.45,
      step: 0.05,
      default: 0.1,
      format: value => `P${Math.round(value * 100)}`,
      help: 'The percentiles fractions of GPUGroupStatistics are a per-frame parameter buffer.'
    },
    {
      kind: 'slider',
      id: 'upperFraction',
      label: 'Upper percentile of the hour',
      group: 'Statistics',
      apply: 'param',
      min: 0.55,
      max: 0.95,
      step: 0.05,
      default: 0.9,
      format: value => `P${Math.round(value * 100)}`,
      help: 'Upper quantile of the observation hour.'
    },
    {
      kind: 'select',
      id: 'joinKind',
      label: 'Join kind (areas onto tracts)',
      group: 'Key join',
      apply: 'compile',
      default: 'left',
      help: 'Left keeps every tract aligned with its row; inner also compacts the matched tract rows into a list. Both gather the area statistic. Compile-time.',
      options: [
        {value: 'left', label: 'Left join (every tract keeps its row)'},
        {value: 'inner', label: 'Inner join (compact the matched tracts)'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumObservations',
      label: 'Right-side mask: areas with at least',
      group: 'Key join',
      apply: 'param',
      min: 0,
      max: 5000,
      step: 250,
      default: 0,
      unit: 'observations',
      help: 'The area table is the right side of the tract join. Areas below this many observations are masked out: their tracts do not match and the area is drawn gray.'
    },
    {
      kind: 'toggle',
      id: 'showObservations',
      label: 'Show observations colored by hour z-score',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'GPUGroupStatistics also writes a z-score per observation: how unusual its hour is inside its own area. Blue is earlier than the area average, red later.'
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'Tract outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draw tract boundaries more boldly when drawing tracts.'
    }
  ],

  readouts: [
    {id: 'observations', label: 'Observations in the filter'},
    {
      id: 'groups',
      label: 'Groups',
      help: 'Rows of the dense group table (community areas 1 to 77).'
    },
    {id: 'highest', label: 'Highest areas'},
    {id: 'lowest', label: 'Lowest areas'},
    {id: 'join', label: 'Key join result'},
    {
      id: 'selected',
      label: 'Selected area',
      help: 'Click an area to pin all its statistics; hover shows them too.'
    }
  ],

  legends: state => {
    const data = getLegendData<GroupStatisticsLegend>('group-statistics');
    const metric = GROUP_METRICS.find(entry => entry.id === state.metric) ?? GROUP_METRICS[0];
    const legends: LegendSpec[] = [];
    if (metric.kind === 'category') {
      legends.push({
        kind: 'categories',
        title: metric.label,
        entries: (data?.modalGroups ?? []).map(type => ({
          color: NATURE_CATEGORY_COLORS[type.index] ?? [150, 150, 150, 255],
          label: `${type.name}: ${type.areas} areas`
        })),
        note: 'The mode of the category column per area.'
      });
    } else {
      legends.push({
        kind: 'ramp',
        id: 'value',
        title: metric.label,
        ramp:
          metric.id.startsWith('hour') &&
          metric.id !== 'hourSd' &&
          metric.id !== 'hourSkew' &&
          metric.id !== 'hourKurtosis'
            ? 'cividis'
            : 'magma',
        extent: 'gpu',
        unit: metric.unit,
        format: value => formatCompact(value)
      });
    }
    if (state.showObservations) {
      legends.push({
        kind: 'ramp',
        title: 'Observation hour vs. area average (z-score)',
        ramp: 'diverging',
        extent: [-2.5, 2.5],
        labels: ['earlier', 'later']
      });
    }
    return legends;
  },

  snippet:
    state => `import {GPUGroupStatistics, GPUKeyJoin} from '@luma.gl/experimental/gpu-dataframe';

// keys: community area index per observation (0xffffffff = none); dense table of 77 groups.
graph.add(new GPUGroupStatistics({
  keys: observationArea, mask: filterMask, keyCount: 77,
  variance: '${state.variance}', percentiles,         // per-frame fractions [${state.lowerFraction}, 0.5, ${state.upperFraction}]
  columns: [
    {values: hour, statistics: ['mean', 'median', 'standardDeviation', 'percentiles',
      'mode', 'uniqueCount', 'skewness', 'kurtosis', 'minimum', 'maximum', 'zScore'], output: hourOutput},
    {values: grade, statistics: ['sum', 'mean'], output: gradeOutput},
    {values: category, statistics: ['mode', 'uniqueCount'], output: categoryOutput},
    {values: species, statistics: ['uniqueCount'], output: speciesOutput},
    {values: introduced, statistics: ['mean'], output: introducedOutput}
  ],
  output: {keys, counts, count, overflow}
}));

// 1:n join: sums and means of the tracts inside each area.
graph.add(new GPUKeyJoin({
  leftKeys: areaIds, rightKeys: tractArea,
  aggregates: [
    {operation: 'sum', column: population, output: populationSum},
    {operation: 'mean', column: perCapitaIncome, output: incomeMean},
    {operation: 'count', output: tractCount}
  ],
  output: {}
}));

// Gather the area statistic onto every tract (${state.joinKind} join).
graph.add(new GPUKeyJoin({
  kind: '${state.joinKind}', leftKeys: tractArea, rightKeys: areaIds,
  rightMask: areaHasEnoughObservations,                     // areas below ${state.minimumObservations} observations are ignored
  gather: [{column: areaStatistic, output: tractStatistic}],
  output: {matched${state.joinKind === 'inner' ? ',\n    rows: {ids, count, overflow}' : ''}}
}));`,

  about: {
    what: '`GPUGroupStatistics` groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and value column on the GPU: count, sum, mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles, mode, distinct count and a per-row z-score. `GPUKeyJoin` attaches a right table to a left table by key, as a 1:1 gather or a 1:n aggregate, left or inner.',
    why: 'Almost every dashboard is "group by and join": sightings by area, rates by population, areas back onto tracts. Doing it on the GPU keeps the table next to the data, so changing a filter recomputes 43,600 rows in a frame and nothing is read back except a few kilobytes of results.',
    howToRead:
      'Each community area is one group. Its colour is the chosen statistic of the observations inside it. Switch to tracts to see the same value gathered by key. Hover for the full statistics of an area. Matches pandas groupby/merge, numpy percentiles and d3 statistics.'
  },

  create: async ctx => (await import('./group-statistics.compute')).createGroupStatistics(ctx),

  story: [
    {
      id: 'question',
      title: 'Where is nature watched the most, per resident?',
      body: 'People logged **43,557 wild plants, animals and fungi** in Chicago on iNaturalist in 2023. Counting them by community area is a *group by*, and the raw count rewards big parks: Lincoln Park has the most (5,526), then Uptown (4,887) and Lincoln Square (4,142).\n\n`GPUGroupStatistics` groups every observation by its community area on the GPU. `GPUKeyJoin` then attaches the **population** of the 791 tracts inside each area by key, so with **Statistic mapped** set below the map shows **observations per 1,000 residents**. Citywide that is 16; North Park stands out at 165, ten times the average, and Lincoln Square (98), Uptown (85) and South Deering (82) follow. Observations follow observers, so a rate here measures how much a place is watched as much as what lives there: always ask what the denominator is.',
      options: {metric: 'perThousand'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.9},
      controls: ['metric'],
      readouts: ['highest', 'lowest'],
      highlight: {readout: 'highest'}
    },
    {
      id: 'many-statistics',
      title: 'One pass, a dozen statistics',
      body: 'The graph computes, per area and in one run: count, mean, median, percentiles, standard deviation, skewness, kurtosis, mode, distinct count, research-grade observations, the introduced share and the number of distinct taxa. Set **Statistic mapped** to **Research-grade share**: the mean of the 0/1 research-grade column, the observations whose identification other iNaturalist users confirmed. Citywide 63.0% reach research grade; Woodlawn (80%) and Uptown (78%) run well above that, while Lincoln Park (51%) sits near the bottom of the busy areas.\n\nHover any area to read every statistic at once. Rows are sorted by key with a radix sort, counts and sums use exact 64-bit integer atomics, and the moments use a fixed-order reduction, so results are bitwise reproducible.',
      options: {metric: 'researchShare'},
      controls: ['metric'],
      readouts: ['highest', 'selected']
    },
    {
      id: 'richness',
      title: 'How many different species?',
      body: 'Set **Statistic mapped** to **Species richness (distinct taxa)**: `GPUGroupStatistics` counts the unique values of the taxon column per area. Lincoln Square (1,435 taxa), Lincoln Park (1,379) and North Park (1,316) lead, then Uptown (1,030): citywide the 43,557 observations cover 4,823 taxa.\n\nRichness climbs with the number of observations, so compare it with the **Observations** map before reading it as biodiversity. Then try **Introduced share**: 15.7% of observations are non-native, but Logan Square reaches 41% and the Loop 29%, while Washington Park and South Shore sit near 5%.',
      options: {metric: 'richness'},
      controls: ['metric'],
      readouts: ['highest', 'selected']
    },
    {
      id: 'time-of-day',
      title: 'When do people go out looking?',
      body: "Set **Statistic mapped** to **Median hour of day**, then use the **Lower percentile of the hour** and **Upper percentile of the hour** sliders: the range between them says how concentrated an area's observations are in time. The percentile fractions are a parameter buffer; drag the sliders and watch the map update with no rebuild.\n\nCitywide the median observation is made at 13:00 and the middle 80% fall between 8:00 and 18:00. A few areas run later (Lincoln Square's median is 16:00): the story is in the spread, which depends on who goes where and when. Try **Most common hour (mode)** to see each area's busiest hour.",
      options: {metric: 'hourMedian', lowerFraction: 0.1, upperFraction: 0.9},
      controls: ['metric', 'lowerFraction', 'upperFraction']
    },
    {
      id: 'mode',
      title: 'The most common group',
      body: 'The **mode** of the group column per area is the kind of life seen most often, and **distinct groups** counts how many of the ten groups appear. Plants lead in 48 of the 77 areas, birds in 16 (Uptown alone has 2,564 bird observations, thanks to the lakefront), insects in 11 and fungi in two. All ten groups are seen in 18 areas.\n\nThe mode is exact: ties resolve to the smallest category code. Switch **Statistic mapped** to *Distinct groups* to see which areas see them all.',
      options: {metric: 'modalGroup'},
      controls: ['metric']
    },
    {
      id: 'filters',
      title: 'Filter on the GPU, recompute in a frame',
      body: 'Pick **Group** *Birds*, restrict the **Hour of day** to 5:00 to 9:00, or keep only **Identification** *Research grade*. A kernel builds the row mask from four words and `GPUGroupStatistics` regroups all 43,600 rows. Early morning belongs to birders: 4,642 observations were made between 5:00 and 9:00 and 62% of them are birds, against 26% of all observations.\n\nSet **Statistic mapped** to **Observations** or **Research-grade share** for that slice, and turn on **Show observations colored by hour z-score** to draw the per-row z-score the contributor also writes: each observation coloured by how early or late it is compared with its own area.',
      options: {metric: 'count', natureGroup: 'Birds', hours: [5, 9]},
      controls: ['natureGroup', 'hours', 'gradeFilter', 'metric', 'showObservations'],
      readouts: ['observations']
    },
    {
      id: 'join',
      title: 'Joins by key: tracts and areas',
      body: 'Switch **Draw** to *Census tracts*. A second `GPUKeyJoin` takes the area table as its **right side** and gathers the chosen statistic onto each of the 791 tracts by community-area key. Set **Statistic mapped** to **Mean tract per-capita income** or **Mean tract poverty** (the other join: a 1:n mean of the tracts in each area) to see how wildlife watching and socio-economics line up, remembering that observations also follow access to parks.\n\nRaise **Right-side mask: areas with at least** to drop areas with few observations: their tracts no longer match. Change **Join kind** to *Inner*: the join then also compacts the matched tract rows, reported in **Key join result**.',
      options: {
        level: 'tracts',
        metric: 'poverty',
        natureGroup: 'ALL',
        hours: [0, 24],
        minimumObservations: 500
      },
      controls: ['level', 'metric', 'minimumObservations', 'joinKind'],
      readouts: ['join']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: 'iNaturalist counts are opportunistic: they measure where people look and upload, not how much wildlife there is, and an area with a popular nature walk can outweigh a larger wild one. Per-resident rates mislead where few people live; population is the 2018-2022 ACS count. Averaging tract incomes ignores tract size.\n\nTry: **Skewness of the hour** and **Excess kurtosis of the hour** in **Statistic mapped**; **Variance kind** *Population*; a different **Upper percentile of the hour** for fungi; **Join kind** *Inner* with a high right-side mask; **Introduced share** for plants only.',
      options: {
        metric: 'introducedShare',
        natureGroup: 'Plants',
        hours: [0, 24],
        level: 'areas',
        minimumObservations: 0
      },
      controls: ['metric', 'natureGroup', 'variance', 'upperFraction', 'joinKind']
    }
  ]
});
