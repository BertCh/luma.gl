// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {EMERGING_CATEGORY_COLORS} from './b13-category-layer';
import type {EmergingHotSpotsOptions} from './emerging-hot-spots.compute';

const GROUPS = [
  {value: 'all', label: 'All observations'},
  {value: '0', label: 'Plants'},
  {value: '1', label: 'Birds'},
  {value: '2', label: 'Insects'},
  {value: '3', label: 'Fungi'},
  {value: '4', label: 'Mammals'},
  {value: '5', label: 'Spiders and kin'},
  {value: '6', label: 'Amphibians and reptiles'},
  {value: '7', label: 'Snails and mussels'},
  {value: '8', label: 'Fish'},
  {value: '9', label: 'Other life'}
];

const color = (code: number, alpha = 235) =>
  [...EMERGING_CATEGORY_COLORS[code], alpha] as [number, number, number, number];

const CATEGORY_ENTRIES = [
  {color: color(1), label: 'New hot spot'},
  {color: color(2), label: 'Consecutive hot spot'},
  {color: color(3), label: 'Intensifying hot spot'},
  {color: color(4), label: 'Persistent hot spot'},
  {color: color(5), label: 'Diminishing hot spot'},
  {color: color(6), label: 'Sporadic hot spot'},
  {color: color(7), label: 'Oscillating hot spot'},
  {color: color(8), label: 'Historical hot spot'},
  {color: color(9), label: 'New cold spot'},
  {color: color(10), label: 'Consecutive cold spot'},
  {color: color(11), label: 'Intensifying cold spot'},
  {color: color(12), label: 'Persistent cold spot'},
  {color: color(13), label: 'Diminishing cold spot'},
  {color: color(14), label: 'Sporadic cold spot'},
  {color: color(15), label: 'Oscillating cold spot'},
  {color: color(16), label: 'Historical cold spot'}
];

export default defineScene<EmergingHotSpotsOptions>({
  id: 'emerging-hot-spots',
  title: 'Where do Chicago nature hot spots persist, and which are seasonal?',
  chapter: 'time',
  order: 1,
  summary:
    'A space-time cube of 43,557 iNaturalist observations from 2023 on 500 m cells, classified on the GPU into the 17 emerging hot spot categories: space-time Gi*, a Mann-Kendall trend per cell, and ArcGIS-style new, intensifying, persistent, diminishing, sporadic and historical patterns.',
  contributors: ['GPUEmergingHotSpots', 'addSpaceTimeHotSpotsRecipe', 'GPUTemporalReduction'],
  datasets: [{id: 'chicago-nature', role: 'timestamped nature observations'}],
  initialView: {longitude: -87.68, latitude: 41.835, zoom: 9.9},

  options: [
    {
      kind: 'select',
      id: 'groupType',
      label: 'Group of life',
      group: 'Space-time cube',
      apply: 'param',
      default: 'all',
      options: GROUPS,
      help: 'Which groups of life are counted. This rewrites the per-event mask buffer; the graph is not rebuilt.'
    },
    {
      kind: 'select',
      id: 'cube',
      label: 'Time slices',
      group: 'Space-time cube',
      apply: 'compile',
      default: 'weeks',
      options: [
        {
          value: 'weeks',
          label: '52 weeks (GPUTemporalReduction)',
          help: 'Seven-day buckets from 1 January. The 365th day (31 December) falls outside 52 whole weeks and is dropped.'
        },
        {
          value: 'months',
          label: '12 calendar months (recipe)',
          help: 'GPUCalendarBuckets decodes each timestamp to a month and the recipe counts events per cell and month.'
        },
        {
          value: 'hours',
          label: '24 hours of the day (recipe)',
          help: 'A cyclic cube: slice 0 is midnight. Trends over the day are read loosely because 23:00 and 00:00 are neighbours.'
        }
      ],
      help: 'How events are cut into slices. The slice count is fixed when the graph is compiled, so changing it rebuilds (each variant is cached).'
    },
    {
      kind: 'slider',
      id: 'minimumEvents',
      label: 'Study-area threshold',
      group: 'Space-time cube',
      apply: 'param',
      min: 1,
      max: 400,
      step: 1,
      default: 20,
      unit: 'observations / year',
      help: 'Cells with fewer observations (of any group) in 2023 are masked out, so lake and industrial cells do not enter the statistics. The cell mask is a buffer write.'
    },
    {
      kind: 'select',
      id: 'neighborhood',
      label: 'Neighborhood source',
      group: 'Neighborhood',
      apply: 'compile',
      default: 'lattice',
      options: [
        {
          value: 'lattice',
          label: 'Lattice radius (cells mode)',
          help: 'The kernel walks lattice offsets with dx^2 + dy^2 <= r^2. Binary weights.'
        },
        {
          value: 'weights',
          label: 'Neighbor-search weights (weights mode)',
          help: 'A GPUNeighborSearch distance band over cell centers supplies the neighbors and their weights.'
        }
      ],
      help: 'Cells mode needs a regular lattice; weights mode accepts any spatial weights (H3 cells, polygons, points). With binary weights the two give the same map.'
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Neighborhood radius',
      group: 'Neighborhood',
      apply: 'param',
      min: 0,
      max: 4,
      step: 0.5,
      default: 2,
      unit: 'cells',
      format: value => `${value} cells (${value * 500} m)`,
      help: 'Cells within this distance count as neighbors of a bin. 0 uses the bin alone. The maximum of 4 cells is a compile-time bound.'
    },
    {
      kind: 'select',
      id: 'weightKind',
      label: 'Weights in weights mode',
      group: 'Neighborhood',
      apply: 'param',
      default: 'binary',
      disabledWhen: state => state.neighborhood === 'lattice',
      options: [
        {value: 'binary', label: 'Binary (every neighbor 1)'},
        {value: 'inverse-distance', label: 'Inverse distance (1 / d)'},
        {value: 'kernel', label: 'Triangular kernel (1 - d / radius)'}
      ],
      help: 'How neighbors are weighted in the Gi* sum. Distance decay makes nearer cells count more; only weights mode can do this.'
    },
    {
      kind: 'slider',
      id: 'temporalWindow',
      label: 'Temporal window',
      group: 'Neighborhood',
      apply: 'param',
      min: 0,
      max: 6,
      step: 1,
      default: 2,
      unit: 'previous slices',
      help: 'Each bin also gathers its neighbors in this many previous slices (the space-time neighborhood). 0 is a purely spatial Gi* per slice.'
    },
    {
      kind: 'select',
      id: 'confidence',
      label: 'Hot spot confidence',
      group: 'Significance',
      apply: 'param',
      default: '0.95',
      options: [
        {value: '0.9', label: '90% (z 1.64)'},
        {value: '0.95', label: '95% (z 1.96)'},
        {value: '0.99', label: '99% (z 2.58)'}
      ],
      help: 'A bin is hot when its Gi* z-score reaches the critical value, cold when it reaches its negative.'
    },
    {
      kind: 'toggle',
      id: 'tieTrend',
      label: 'Trend level follows confidence',
      group: 'Significance',
      apply: 'param',
      default: true,
      help: 'The Mann-Kendall trend must be significant at one minus the confidence level (a p-value of 0.05 at 95%). Turn off to set it separately.'
    },
    {
      kind: 'slider',
      id: 'trendLevel',
      label: 'Trend p-value threshold',
      group: 'Significance',
      apply: 'param',
      min: 0.01,
      max: 0.2,
      step: 0.01,
      default: 0.05,
      disabledWhen: state => state.tieTrend,
      help: 'Two-sided p-value at or below which a cell is called intensifying or diminishing.'
    },
    {
      kind: 'slider',
      id: 'persistentFraction',
      label: 'Persistence share',
      group: 'Significance',
      apply: 'param',
      min: 0.6,
      max: 1,
      step: 0.05,
      default: 0.9,
      format: value => `${Math.round(value * 100)}% of slices`,
      help: 'A cell is persistent, intensifying or diminishing (or historical) when it is hot in at least this share of the slices. Lower it to promote sporadic cells.'
    },
    {
      kind: 'select',
      id: 'mapView',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'category',
      options: [
        {value: 'category', label: 'Emerging hot spot category'},
        {value: 'gi', label: 'Gi* z-score in one slice'},
        {value: 'events', label: 'Observation count in one slice'},
        {value: 'trend', label: 'Mann-Kendall trend z'},
        {value: 'hot-count', label: 'Number of hot slices'},
        {value: 'cold-count', label: 'Number of cold slices'}
      ],
      help: 'Every view reads a different output buffer of the same run; switching never recomputes anything.'
    },
    {
      kind: 'slider',
      id: 'slice',
      label: 'Slice shown',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 52,
      step: 1,
      default: 1,
      disabledWhen: state => !['gi', 'events'].includes(state.mapView),
      help: 'Which time slice the Gi* and event-count views display (clamped to the cube, 12, 24 or 52).'
    },
    {
      kind: 'toggle',
      id: 'play',
      label: 'Play through the slices',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => !['gi', 'events'].includes(state.mapView),
      help: 'Steps the slice every half second by rewriting a small index buffer.'
    },
    {
      kind: 'toggle',
      id: 'showEvents',
      label: 'Show observation points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the selected observations as faint dots under the cells.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Lower it to read streets under the cells.'
    }
  ],

  readouts: [
    {id: 'events', label: 'Events counted', help: 'Observations passing the group filter.'},
    {id: 'grid', label: 'Lattice'},
    {id: 'cells', label: 'Study-area cells', help: 'Cells above the study-area threshold.'},
    {id: 'cubeEvents', label: 'Cube'},
    {id: 'busiest', label: 'Busiest slice'},
    {id: 'sliceLabel', label: 'Slice shown'},
    {id: 'hot', label: 'Hot spot cells', format: 'integer', help: 'Categories 1 to 8.'},
    {id: 'cold', label: 'Cold spot cells', format: 'integer', help: 'Categories 9 to 16.'},
    {id: 'new', label: 'New hot spots', format: 'integer'},
    {id: 'intensifying', label: 'Intensifying hot spots', format: 'integer'},
    {id: 'persistent', label: 'Persistent hot spots', format: 'integer'},
    {id: 'diminishing', label: 'Diminishing hot spots', format: 'integer'},
    {id: 'sporadic', label: 'Sporadic hot spots', format: 'integer'}
  ],

  legends: state => {
    switch (state.mapView) {
      case 'category':
        return [
          {
            kind: 'categories',
            title: 'Emerging hot spot pattern',
            entries: CATEGORY_ENTRIES,
            note: 'Cells with no pattern are transparent. Warm = hot, cool = cold; the final slice decides the track.'
          }
        ];
      case 'gi':
        return [
          {
            kind: 'ramp',
            title: 'Space-time Gi* z-score',
            ramp: 'diverging',
            extent: [-4, 4],
            labels: ['cold (z = -4)', 'hot (z = +4)'],
            unit: 'z'
          }
        ];
      case 'trend':
        return [
          {
            kind: 'ramp',
            title: 'Mann-Kendall trend of Gi*',
            ramp: 'diverging',
            extent: [-4, 4],
            labels: ['falling', 'rising'],
            unit: 'z'
          }
        ];
      case 'hot-count':
        return [
          {
            kind: 'ramp',
            title: 'Hot slices per cell',
            ramp: 'inferno',
            extent: [0, state.cube === 'months' ? 12 : state.cube === 'hours' ? 24 : 52],
            unit: 'slices'
          }
        ];
      case 'cold-count':
        return [
          {
            kind: 'ramp',
            title: 'Cold slices per cell',
            ramp: 'viridis',
            extent: [0, state.cube === 'months' ? 12 : state.cube === 'hours' ? 24 : 52],
            unit: 'slices'
          }
        ];
      default:
        return [
          {
            kind: 'ramp',
            id: 'count',
            title: 'Events per cell and slice',
            ramp: 'magma',
            extent: 'gpu',
            sqrtScale: true,
            unit: 'events',
            format: value => value.toFixed(0)
          }
        ];
    }
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  addSpaceTimeHotSpotsRecipe,
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTemporalReduction} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'emerging'});
${
  state.cube === 'weeks'
    ? `// 52 seven-day buckets of event counts per ${'500 m'} cell
graph.add(new GPUTemporalReduction({
  cellIds, timestamps: seconds, values: ones, mask: eventMask,
  parameters: bucketParameters,          // [origin, width] = [0, 7 * 86400]
  cellCount, bucketCount: 52,
  output: {counts, min, max, first, last, occupiedSlots}
}));
graph.add(new GPUEmergingHotSpots({
  values: counts, ${state.neighborhood === 'lattice' ? 'gridWidth, gridHeight, maximumRadius: 4' : 'weights, selfWeight: 1'},
  sliceCount: 52, mask: cellMask, parameters: hotSpotParameters,
  giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount
}));`
    : `// calendar ${state.cube} -> cube -> Gi* -> categories in one recipe
addSpaceTimeHotSpotsRecipe(graph, {
  timestamps, mask: eventMask, calendarParameters,
  slices: {field: '${state.cube === 'months' ? 'month' : 'hour'}', firstValue: ${state.cube === 'months' ? 1 : 0}, count: ${state.cube === 'months' ? 12 : 24}},
  cells: ${
    state.neighborhood === 'lattice'
      ? `{kind: 'lattice', positions, width, height, bounds, maximumRadius: 4}`
      : `{kind: 'ids', cellIds, cellCount, weights, selfWeight: 1}`
  },
  cellMask, parameters: hotSpotParameters,
  cube, giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount
});`
}
const compiled = graph.compile();            // once
// per change: write parameters (no rebuild), then encode
hotSpotParameters.write(getGPUEmergingHotSpotParameterValues({
  radius: ${state.radius}, temporalWindow: ${state.temporalWindow}, confidenceLevel: ${state.confidence},
  persistentFraction: ${state.persistentFraction}${state.tieTrend ? '' : `, trendSignificanceLevel: ${state.trendLevel}`}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: "`GPUEmergingHotSpots` takes a space-time cube of counts (cells by time slices) and computes, for every bin, a Gi* z-score over its space-time neighborhood; then a Mann-Kendall trend test over each cell's Gi* series; then assigns one of 17 categories. The cube comes from `GPUTemporalReduction` (equal-width time buckets) or from `addSpaceTimeHotSpotsRecipe` (calendar months or hours).",
    why: 'A map of one period tells you where observations are, not whether a place is a lasting refuge, a seasonal flush or a newly discovered patch. The categories separate a spot that lit up for one migration week from one that has drawn observers all year, which is the difference between a seasonal event and a core habitat. Because the data are community-science records, hot spots mean wildlife and observers together.',
    howToRead:
      'Warm colors are hot spots and cool colors are cold spots. The final slice picks the track: hot now (new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating) or not hot now but hot before (historical). Cells with no significant pattern are transparent.'
  },

  create: async ctx => (await import('./emerging-hot-spots.compute')).createEmergingHotSpots(ctx),

  story: [
    {
      id: 'question',
      title: 'Where is nature-watching new, and where is it entrenched?',
      body: 'Chicagoans posted 43,557 wild observations to iNaturalist in 2023. A heat map of the year shows where, not when. **`GPUEmergingHotSpots`** adds the time axis: it counts observations per 500 m cell and per week (a *space-time cube*), tests every bin with a Gi* hot spot statistic, then classifies each cell by how its hot or cold status evolved over the year.\n\nWarm cells are hot at the end of the year, cool cells cold, and pale cells less settled than saturated ones. Hover a cell for its counts and trend; **Group of life** and **Map shows**, below, change what is counted and what is drawn. The legend lists all 16 categories; "no pattern" is transparent.',
      controls: ['groupType', 'mapView'],
      readouts: ['events', 'hot', 'cold']
    },
    {
      id: 'the-cube',
      title: 'First, the cube: a count per cell and week',
      body: 'The cube here comes from **`GPUTemporalReduction`**: every observation is assigned to a cell and to a seven-day bucket counted from 1 January (52 buckets; 31 December falls outside 52 whole weeks and is dropped). That is a plain count per bin, drawn for one week.\n\nThe **Busiest slice** readout names the week with the most observations: for all groups it is week 19, 7 to 13 May, with 1,811, the height of spring migration. Move the **Slice shown** slider or turn on **Play through the slices** to watch the city bloom in April and May, hold through the summer and fade by November. The study area is the roughly 300 cells with at least 20 observations in the year (**Study-area threshold**); the lake, rail yards and industrial land are masked.',
      options: {mapView: 'events', slice: 19},
      controls: ['slice', 'play', 'minimumEvents'],
      readouts: ['busiest', 'sliceLabel', 'cells']
    },
    {
      id: 'gi-star',
      title: 'Space-time Gi*: is this bin hotter than chance?',
      body: 'For every bin, Gi* sums the counts of its **space-time neighborhood** (cells within the radius, in the current and the previous two weeks, set by **Temporal window**) and compares the sum with what random arrangement of all bins would give. The result is a z-score: `z = sum(x_j - mean) / (S * sqrt((n k - k^2) / (n - 1)))` where `k` is the number of bins in the neighborhood.\n\nRed is significantly high, blue significantly low, white unremarkable. A bin is a *hot spot* when z reaches the critical value of the **Hot spot confidence** (1.96 at 95%). Move **Slice shown** to see the pattern change week to week. The persistent dark red is the north lakefront: Lincoln Park, Montrose Point and Uptown, where neighborhood after neighborhood stays far above the citywide mean.',
      options: {mapView: 'gi', slice: 29},
      controls: ['slice', 'confidence'],
      readouts: ['sliceLabel']
    },
    {
      id: 'categories',
      title: 'A Mann-Kendall trend turns z-scores into 16 categories',
      body: 'Each cell now has a series of 52 Gi* scores. The **Mann-Kendall** test asks whether the series rises or falls, without assuming a straight line. Together with how often the cell was hot, it gives the ArcGIS-style categories: **new** (hot only in the last week), **consecutive**, **intensifying** (hot most of the time and getting hotter), **persistent** (hot most of the time, no trend), **diminishing**, **sporadic**, **oscillating** and **historical**; cold spots mirror them.\n\nThe readouts count cells per class. The cell around Montrose Point, the busiest of all, has observations in all 52 weeks of 2023, so it is the natural candidate for a long-running hot spot. Sporadic ones flare for a migration week or a bioblitz and go quiet. Hover cells for their hot-week counts, and try one **Group of life** at a time.',
      options: {mapView: 'category'},
      callout: {coordinate: [-87.6325, 41.9625], text: 'Montrose Point: observed every week'},
      controls: ['mapView'],
      readouts: ['intensifying', 'persistent', 'sporadic']
    },
    {
      id: 'persistence',
      title: 'Persistence share: how settled must a spot be?',
      body: 'A cell is "persistent", "intensifying" or "diminishing" only when it is hot in at least the **Persistence share** of the weeks (90% by default). Lower it to 60% and sporadic cells with a long record are promoted into the entrenched classes; raise **Hot spot confidence** to 99% and weak hot spots drop out entirely.\n\nThese are parameters of the classification, not of the cube, so the map updates without recomputing the counts. The trend p-value follows the confidence level unless you untick **Trend level follows confidence**.',
      options: {persistentFraction: 0.6, confidence: '0.99'},
      controls: ['persistentFraction', 'confidence', 'tieTrend'],
      readouts: ['persistent', 'sporadic']
    },
    {
      id: 'neighborhood',
      title: 'Bigger neighborhoods, longer memory',
      body: 'The **Neighborhood radius** (cells) and the **Temporal window** (previous slices) define the space-time neighborhood. A radius of 3 cells (1.5 km) and a window of 4 weeks smooths over single sites and merges nearby hot spots into broad regions, such as the whole lakefront; a radius of 0 and a window of 0 tests each bin alone and is noisy, especially here, where most bins hold zero or one observation.\n\nThe radius is a per-frame parameter; only its maximum (4 cells) is compiled into the graph. Try 1 cell: the hot areas fragment into individual parks and preserves.',
      options: {radius: 3, temporalWindow: 4, confidence: '0.95', persistentFraction: 0.9},
      controls: ['radius', 'temporalWindow'],
      readouts: ['hot']
    },
    {
      id: 'weights-mode',
      title: 'Same answer from a weights table, plus distance decay',
      body: 'In **weights mode** (**Neighborhood source** set to *Neighbor-search weights*) the neighbors do not come from lattice offsets but from a `GPUNeighborSearch` distance band over the cell centers, handed to the contributor as a weights table. With binary weights it reproduces the lattice result (a handful of cells on the significance boundary can flip because of floating-point order). That is the path for H3 cells, polygons or points, which have no lattice.\n\nOnly weights mode can decay neighbors with distance: set **Weights in weights mode** to *Triangular kernel* and nearer cells count more. This is a compile-time choice, so the control shows a rebuild badge; the variant is cached afterwards.',
      options: {neighborhood: 'weights', weightKind: 'kernel', radius: 2.5, temporalWindow: 2},
      controls: ['neighborhood', 'weightKind', 'radius'],
      readouts: ['hot']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      body: 'These are observations by volunteers, not a census: where and when people look shapes the map, and a hot spot may be a popular trailhead rather than the richest habitat. The City Nature Challenge weekend (28 April to 1 May 2023) alone adds a visible pulse. The cube treats a bin with zero observations as a valid low value, which is why the study-area threshold matters. Gi* runs hundreds of tests, so some hot cells are chance; the Mann-Kendall test on 52 slices is also sensitive to autocorrelation. Hours of the day are a cyclic cube, so read those trends loosely.\n\nTry: switch **Group of life** to *Insects* or *Fungi*; set **Time slices** to *12 calendar months* (the recipe path with `GPUCalendarBuckets`) and compare; or to *24 hours of the day* to see where the morning and afternoon hot spots sit.',
      options: {groupType: '2', cube: 'months', neighborhood: 'lattice', mapView: 'category'},
      controls: ['groupType', 'cube']
    }
  ]
});
