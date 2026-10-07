// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import {getGiTable, getMoranQuadrantColors} from '../weights/hot-spots.style';
import type {RateSmoothingLegendData, RateSmoothingOptions} from './rate-smoothing.compute';
import {getInkColor, RATE_BASIS} from './rate-smoothing.style';

/** The contiguous US, the frame of every national county step. */
const CONUS_BOUNDS = [-124.8, 24.4, -66.9, 49.4] as const;
/** The plains from Texas to Montana, where the smallest counties are. */
const PLAINS_BOUNDS = [-109.5, 30.5, -94, 49.2] as const;
/** The Mountain West, where spatial pooling matters most. */
const MOUNTAIN_BOUNDS = [-117.5, 31, -102, 49] as const;

/** The cartouche of one step (steps replace the whole `title`; the standing sample line is set at create). */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  sample: '3,109 counties of the contiguous US; births 2021-2023, Census population estimates',
  chips: ['Estimates'] as const
});

const COUNTY_CREDIT = joinCredits(
  'US Census Bureau Population Estimates, Vintage 2023 (public domain)',
  'US Census boundaries',
  CREDITS.colorBrewer
);

const MAP_LABELS: Record<RateSmoothingOptions['map'], string> = {
  raw: 'Raw rate',
  smoothed: 'Empirical Bayes',
  spatial: 'Spatial EB',
  standardized: 'Standardised z',
  cluster: 'Clusters'
};

function getLegends(
  state: RateSmoothingOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const shared = data['rateSmoothing'] as RateSmoothingLegendData | undefined;
  if (!shared) return [];
  const {rateTable, zTable, rateCounts, zCounts, clusterCounts, groundIsDark} = shared;
  const effective =
    state.swipe === 'raw-eb' ? 'smoothed' : state.swipe === 'eb-spatial' ? 'spatial' : state.map;
  const legends: LegendSpec[] = [];
  if (effective === 'raw' || effective === 'smoothed' || effective === 'spatial') {
    const swipeNote =
      state.swipe === 'off'
        ? ''
        : ' Counts are for the right-hand map; both sides share the table.';
    legends.push(
      getClassTableLegend(rateTable, {
        title: 'General fertility rate',
        id: 'rate-classes',
        basis: RATE_BASIS,
        counts: rateCounts[effective],
        interactive: true,
        layout: 'list',
        note: `${rateTable.method}. Pooled rate ${shared.pooledPerThousand.toFixed(1)} per 1,000.${swipeNote}`
      })
    );
    if (state.fadeUnreliable) {
      legends.push({
        kind: 'alpha',
        title: "Weight on the county's own births",
        colors: [rateTable.colors[5]],
        ends: ['mostly the pooled prior', 'mostly its own births'],
        steps: 4,
        note: 'Faded counties borrow most of their estimate from the prior.'
      });
    }
    if (state.showExtremes) {
      const ink = getInkColor(groundIsDark ? 'dark' : 'light');
      legends.push({
        kind: 'line',
        title: 'Extreme raw rates',
        entries: [
          {color: ink, widthPixels: 1.6, label: 'Highest tail of the raw rates'},
          {color: ink, widthPixels: 1.6, label: 'Lowest tail of the raw rates', dashed: true}
        ]
      });
    }
  } else if (effective === 'standardized') {
    legends.push(
      getClassTableLegend(zTable, {
        title: 'Standardised rate',
        id: 'z-classes',
        basis: '(rate - pooled rate) / its standard error',
        counts: zCounts,
        interactive: true,
        layout: 'list',
        note: '0 is as expected from the pooled rate; red is above it, blue below.'
      })
    );
  } else {
    const colors = getMoranQuadrantColors(getGiTable(groundIsDark ? 'dark' : 'light', 'No data'));
    legends.push({
      kind: 'categories',
      id: 'moran-classes',
      title: 'Local Moran cluster of the smoothed rate',
      layout: 'list',
      interactive: true,
      entries: [
        {color: colors.highHigh, label: 'High-high cluster', count: clusterCounts[1]},
        {color: colors.lowLow, label: 'Low-low cluster', count: clusterCounts[3]},
        {color: colors.highLow, label: 'High-low outlier', count: clusterCounts[4]},
        {color: colors.lowHigh, label: 'Low-high outlier', count: clusterCounts[2]},
        {color: colors.notSignificant, label: 'Not significant', count: clusterCounts[0]}
      ],
      note:
        state.gating === 'permutation'
          ? `Conditional permutation, ${state.permutations} permutations, p <= ${state.significance}${state.falseDiscoveryRate ? ', Benjamini-Hochberg corrected' : ''}.`
          : `Analytic p <= ${state.significance}.`
    });
  }
  return legends;
}

function getSnippet(state: RateSmoothingOptions): string {
  const permutation =
    state.gating === 'permutation'
      ? `  permutation: {                              // GPULocalPermutationTest
    parameters: permutationParameters,        // seed ${state.seed}, ${state.permutations} permutations
    maximumPermutations: 999,
    alternative: '${state.alternative}',
    falseDiscoveryRate: ${state.falseDiscoveryRate}
  },
`
      : '  // no permutation: GPULocalMoran gates analytically\n';
  return `import {
  addRateClusterMapRecipe,
  GPUSpatialEmpiricalBayesRates
} from '@luma.gl/experimental/gpu-spatial-analysis';

// births and womenYears are float32 views; vertices, ringOffsets and
// featureRingOffsets are the county polygons in GeoArrow layout.
const rates = addRateClusterMapRecipe(graph, {
  events: births, populations: womenYears,
  positions: vertices, ringOffsets, polygonOffsets: featureRingOffsets,
  criterion: '${state.criterion}',            // GPUContiguityWeights
  analyze: '${state.analyze}',                // GPUEmpiricalBayesRates output
  neighborCapacity, parameters, palette,
${permutation}  standardizedRates, smoothedRates, rawRates, summary
});

// Spatial EB reuses the recipe's weights.
graph.add(new GPUSpatialEmpiricalBayesRates({
  events: births, populations: womenYears,
  weights: rates.weights, smoothedRates: spatialRates
}));

// summary holds the pooled rate m and the prior variance a; the weight
// w = a / (a + m / b) is not an output, so the map computes it from them.`;
}

export default defineScene<RateSmoothingOptions>({
  id: 'rate-smoothing',
  title: 'Small numbers, loud maps',
  chapter: 'statistics',
  order: 2,
  summary:
    'Raw county birth rates are loudest where counties are smallest. Empirical Bayes shrinks each rate toward what its population can support, spatial EB borrows from neighbours, and a rate cluster map finds the regions that are more than noise.',
  contributors: [
    'GPUEmpiricalBayesRates',
    'GPUSpatialEmpiricalBayesRates',
    'addRateClusterMapRecipe',
    'GPUContiguityWeights',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'GPUGlobalSpatialStatistics'
  ],
  datasets: [
    {id: 'us-counties', role: 'county polygons (queen contiguity from shared vertices)'},
    {id: 'us-county-births', role: 'births 2021-2023 and women aged 15-44 at risk, by county'},
    {id: 'us-states', role: 'state lines over the counties'}
  ],
  initialView: {longitude: -96, latitude: 38.3, zoom: 3.9},

  options: [
    {
      kind: 'select',
      id: 'map',
      label: 'Map shows',
      group: 'Rate variable',
      apply: 'param',
      display: 'chips',
      default: 'raw',
      help: 'Every variable is computed in one graph run; this only picks which output buffer the layer reads. The three rate maps share one set of class breaks, so a change of colour is a change of estimate.',
      options: (Object.keys(MAP_LABELS) as RateSmoothingOptions['map'][]).map(value => ({
        value,
        label: MAP_LABELS[value]
      }))
    },
    {
      kind: 'toggle',
      id: 'fadeUnreliable',
      label: 'Fade unreliable counties',
      group: 'Rate variable',
      apply: 'param',
      default: false,
      help: 'Value-by-alpha: a county the data speaks little for fades toward the paper. Opacity is 0.25 + 0.75 w, where w = a / (a + m / b) is the weight on its own births.'
    },
    {
      kind: 'toggle',
      id: 'showExtremes',
      label: 'Ring the extremes',
      group: 'Rate variable',
      apply: 'param',
      default: false,
      help: 'Solid rings mark the highest tail of the raw rates, dashed rings the lowest. The rings follow the raw rate on every map.'
    },
    {
      kind: 'select',
      id: 'swipe',
      label: 'Compare side by side',
      group: 'Rate variable',
      apply: 'param',
      display: 'segmented',
      default: 'off',
      expert: true,
      help: 'Draws two rate maps on the same classes with a swipe divider: raw and empirical Bayes, or empirical Bayes and spatial empirical Bayes. Story steps set it.',
      options: [
        {value: 'off', label: 'Off'},
        {value: 'raw-eb', label: 'Raw | EB'},
        {value: 'eb-spatial', label: 'EB | Spatial'}
      ]
    },
    {
      kind: 'select',
      id: 'criterion',
      label: 'Neighbour definition',
      group: 'Neighbours',
      apply: 'compile',
      display: 'segmented',
      default: 'queen',
      help: 'GPUContiguityWeights finds neighbours from shared boundary vertices. Queen: any shared vertex. Rook: a shared edge. Used by spatial EB and by local Moran. Compile-time: switches to another graph.',
      options: [
        {value: 'queen', label: 'Queen (shared vertex)'},
        {value: 'rook', label: 'Rook (shared edge)'}
      ]
    },
    {
      kind: 'select',
      id: 'analyze',
      label: 'Rate analysed by local Moran',
      group: 'Cluster map',
      apply: 'compile',
      display: 'segmented',
      default: 'smoothed',
      help: 'The empirical-Bayes rate (the default here: it is not smoothed over neighbours, so clusters are not built in) or the standardised rate of esda Moran_Local_Rate.',
      options: [
        {value: 'smoothed', label: 'Empirical Bayes rate'},
        {value: 'standardized', label: 'Standardised rate'}
      ]
    },
    {
      kind: 'select',
      id: 'gating',
      label: 'Significance test',
      group: 'Cluster map',
      apply: 'compile',
      default: 'permutation',
      expert: true,
      help: 'Analytic: normal-theory p-value of local Moran. Permutation: conditional permutation pseudo p-values from GPULocalPermutationTest. Both graphs are compiled on demand.',
      options: [
        {value: 'permutation', label: 'Conditional permutation test'},
        {value: 'analytic', label: 'Analytic p-value'}
      ]
    },
    {
      kind: 'select',
      id: 'alternative',
      label: 'Permutation tail',
      group: 'Cluster map',
      apply: 'compile',
      default: 'directed',
      expert: true,
      disabledWhen: state => state.gating !== 'permutation',
      help: 'Which tail of the simulated distribution counts as extreme. Directed is the esda default; folded compares absolute deviations from the mean.',
      options: [
        {value: 'directed', label: 'Directed (esda default)'},
        {value: 'two-sided', label: 'Two-sided'},
        {value: 'greater', label: 'Greater (unusually high)'},
        {value: 'lesser', label: 'Lesser (unusually low)'},
        {value: 'folded', label: 'Folded'}
      ]
    },
    {
      kind: 'toggle',
      id: 'falseDiscoveryRate',
      label: 'Benjamini-Hochberg FDR',
      group: 'Cluster map',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.gating !== 'permutation',
      help: 'Controls the share of false discoveries over thousands of simultaneous tests. Expect fewer, more trustworthy clusters.'
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Permutations',
      group: 'Cluster map',
      apply: 'param',
      min: 49,
      max: 999,
      step: 50,
      default: 499,
      disabledWhen: state => state.gating !== 'permutation',
      help: 'More permutations resolve smaller pseudo p-values at proportional GPU cost: the smallest p is 1 / (permutations + 1).'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Permutation seed',
      group: 'Cluster map',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 1,
      expert: true,
      disabledWhen: state => state.gating !== 'permutation',
      help: 'The same seed gives identical pseudo p-values on every run.'
    },
    {
      kind: 'slider',
      id: 'significance',
      label: 'Significance level',
      group: 'Cluster map',
      apply: 'param',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      default: 0.05,
      format: value => `p <= ${value.toFixed(3)}`,
      help: 'Counties whose p-value is above this level are drawn as not significant.'
    }
  ],

  readouts: [
    {id: 'counties', label: 'Counties', format: 'integer', emphasis: 'tile'},
    {
      id: 'pooledRate',
      label: 'Pooled national rate',
      emphasis: 'tile',
      help: 'Sum of births over sum of woman-years: the prior mean every county is shrunk toward.'
    },
    {
      id: 'rawRange',
      label: 'Raw rates range',
      help: 'The lowest and highest raw county rates, per 1,000 women aged 15-44 per year.'
    },
    {
      id: 'rawTopClass',
      label: 'Raw: counties in the darkest class',
      format: 'integer',
      help: 'The classes are septiles of the smoothed rate, so the smoothed map puts about one seventh of the counties in each. The raw map piles counties into the end classes.'
    },
    {
      id: 'rawBottomClass',
      label: 'Raw: counties in the palest class',
      format: 'integer'
    },
    {
      id: 'darkestPopulation',
      label: 'Median county in the darkest raw class',
      help: 'Median residents of the counties in the darkest class of the raw map.'
    },
    {
      id: 'funnel',
      label: 'Rate against county size',
      kind: 'chart',
      help: 'Every county as a dot. Dotted curves are control limits around the pooled rate: counties outside them differ from it by more than chance. Click a dot to find the county on the map.'
    },
    {
      id: 'extremePopulation',
      label: 'Median county with the highest raw rates',
      help: 'Median residents of the counties ringed solid.'
    },
    {id: 'allPopulation', label: 'Median county, all counties'},
    {
      id: 'lowPopulation',
      label: 'Median county with the lowest raw rates',
      help: 'Median residents of the counties ringed dashed.'
    },
    {
      id: 'fewBirths',
      label: 'Counties with under 30 births in three years',
      format: 'integer',
      help: 'With so few events, one birth more or less visibly moves the rate.'
    },
    {
      id: 'trueSpread',
      label: 'Spread of true county rates',
      emphasis: 'tile',
      help: 'The square root of the prior variance a, the estimated variance of the true rates, per 1,000.'
    },
    {id: 'smallLabel', label: 'Small counties'},
    {
      id: 'smallRaw',
      label: 'Small counties, raw (5th to 95th percentile)',
      help: 'The spread of raw rates in the counties under 5,000 residents.'
    },
    {
      id: 'smallEb',
      label: 'Small counties, smoothed (5th to 95th percentile)',
      help: 'The same counties after empirical Bayes smoothing.'
    },
    {
      id: 'neighbourMean',
      label: 'Neighbours per county',
      help: 'Average number of contiguity neighbours under the chosen definition.'
    },
    {
      id: 'moranEb',
      label: "Moran's I, empirical Bayes",
      help: 'Global spatial autocorrelation of the empirical-Bayes rate, row-standardised weights, from GPUGlobalSpatialStatistics.'
    },
    {
      id: 'moranSpatial',
      label: "Moran's I, spatial EB",
      help: 'Global spatial autocorrelation of the spatial empirical-Bayes rate. Pooling with neighbours builds similarity in, so it is higher by construction.'
    },
    {
      id: 'localPool',
      label: 'Local pooled rate of the selected county',
      help: 'Click a county: the rate of the county together with its neighbours, the local prior mean of spatial empirical Bayes.'
    },
    {
      id: 'beyond196',
      label: 'Counties beyond 1.96',
      format: 'integer',
      emphasis: 'tile',
      help: 'Counties whose standardised rate is more than 1.96 standard deviations from the pooled rate.'
    },
    {id: 'above196', label: 'Above the pooled rate', format: 'integer'},
    {id: 'below196', label: 'Below the pooled rate', format: 'integer'},
    {
      id: 'testName',
      label: 'Significance test in use',
      help: 'The conditional permutation test with its permutation count, or the analytic p-value.'
    },
    {id: 'hhCount', label: 'High-high counties', format: 'integer', emphasis: 'tile'},
    {id: 'llCount', label: 'Low-low counties', format: 'integer', emphasis: 'tile'},
    {id: 'outlierCount', label: 'Outliers (high-low and low-high)', format: 'integer'},
    {
      id: 'notSignificantShare',
      label: 'Not significant',
      help: 'Share of counties whose local Moran quadrant is not significant.'
    },
    {id: 'hhRegion', label: 'States with most high-high counties'},
    {id: 'llRegion', label: 'States with most low-low counties'},
    {
      id: 'clusterBars',
      label: 'Counties by cluster type',
      kind: 'chart'
    },
    {id: 'selected', label: 'Selected county', help: 'Click a county to pin its numbers here.'},
    {
      id: 'priorVariance',
      label: 'Prior variance a',
      hood: true,
      format: 'decimal',
      help: 'Estimated variance of the true rates, in births per 1,000 women a year squared (esda does not clamp it).'
    },
    {id: 'priorCheck', label: 'Prior variance sign', hood: true},
    {id: 'weightSpread', label: 'Weight on own births', hood: true},
    {id: 'moranRaw', label: "Moran's I, raw", hood: true},
    {id: 'moranZ', label: "Moran's I, standardised", hood: true},
    {id: 'neighbourDetail', label: 'Contiguity detail', hood: true}
  ],

  pipeline: [
    {
      id: 'rates',
      label: 'Rates',
      detail: 'Births over woman-years of every county',
      show: {option: 'map', value: 'raw'}
    },
    {
      id: 'prior',
      label: 'Prior',
      detail: 'Pooled rate and the variance of the true rates, then z',
      show: {option: 'map', value: 'standardized'}
    },
    {
      id: 'shrink',
      label: 'Shrink',
      detail: 'Weight w pulls small counties toward the pooled rate',
      show: {option: 'map', value: 'smoothed'}
    },
    {
      id: 'neighbours',
      label: 'Neighbours',
      detail: 'Queen contiguity, row-standardised, and local pooling',
      show: {option: 'map', value: 'spatial'}
    },
    {
      id: 'clusters',
      label: 'Clusters',
      detail: 'Local Moran with a permutation test',
      show: {option: 'map', value: 'cluster'}
    }
  ],

  legends: getLegends,

  basemap: ground('paperSheet'),
  furniture: {
    ...NATIONAL_FURNITURE,
    title: cartouche('Where is the birth rate highest?', 'General fertility rate by county'),
    credit: COUNTY_CREDIT
  },

  snippet: getSnippet,

  about: {
    what: '`GPUEmpiricalBayesRates` computes the raw rate, the empirical-Bayes smoothed rate and the Assuncao-Reis standardised rate of every county from births and the women at risk. `GPUSpatialEmpiricalBayesRates` pools each county with its neighbours instead of the nation. `addRateClusterMapRecipe` chains the rates into queen contiguity, local Moran and a permutation test.',
    why: 'A county with a handful of births can show almost any rate: one birth more or less visibly moves it. Smoothing and standardising separate signal from small-number noise before anyone ranks or maps the rates.',
    howToRead:
      'Darker is a higher rate; faded counties borrow most of their estimate from the prior. The same classes are used on every rate map, so a change of colour is a change of estimate. In the funnel chart, counties outside the dotted curves differ from the pooled rate by more than chance. On the cluster map, red counties are high-rate clusters and blue low-rate ones. Limits: births are counted by the mother\'s residence; women aged 15-44 is an estimate, and in college towns students inflate it and lower the rate, which smoothing cannot fix; and smoothed maps can still mislead (Gelman and Price, "All maps of parameter estimates are misleading"). Matches PySAL esda `Empirical_Bayes`, `Spatial_Empirical_Bayes`, `assuncao_rate` and `Moran_Local_Rate`.'
  },

  create: async ctx => (await import('./rate-smoothing.compute')).createRateSmoothing(ctx),

  story: [
    {
      id: 'raw-rate',
      title: 'Where is the birth rate highest?',
      headline: 'The darkest counties tend to be the smallest',
      textAlternative:
        'Map of US counties in seven shades of brown by raw birth rate: dark counties are scattered across the Plains and the interior West, many of them very small.',
      body: 'Births per 1,000 women aged 15-44 a year, against a national rate of **{{pooledRate}}**. The classes are septiles of the smoothed rate, so each should hold about the same number of counties. The raw rate puts **{{rawTopClass}}** in the darkest class, whose median county has **{{darkestPopulation}}**, and **{{rawBottomClass}}** in the palest. Flip **Map shows** to compare.\n\n*Equal classes, unequal counties.*',
      options: {map: 'raw'},
      optionsMode: 'fresh',
      controls: ['map'],
      readouts: ['pooledRate', 'counties', 'rawTopClass', 'rawBottomClass'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Where is the birth rate highest?',
          'Raw general fertility rate, per 1,000 women'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, ['state-sd', 'state-tx']),
      stage: 'rates'
    },
    {
      id: 'small-numbers',
      title: 'Which of those highs can you believe?',
      headline: 'The loudest rates come from small counties',
      textAlternative:
        'The Plains with the highest raw rates ringed in solid ink and the lowest dashed; a funnel chart shows the rates fanning out for small counties.',
      body: 'Solid rings mark the highest raw rates. Those counties have a median of **{{extremePopulation}}** against **{{allPopulation}}** overall, and in the chart they sit on the wide left of the funnel. Dashed rings mark the lowest: larger than typical (**{{lowPopulation}}**), and partly college towns whose students inflate the denominator. Toggle **Ring the extremes**.\n\n*Small numbers make loud maps.*',
      options: {map: 'raw', showExtremes: true},
      optionsMode: 'fresh',
      controls: ['showExtremes'],
      readouts: ['funnel', 'extremePopulation', 'allPopulation', 'fewBirths'],
      camera: {bounds: PLAINS_BOUNDS, transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche('Which rates can you believe?', 'Raw rate, extreme counties ringed'),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, [
        'state-mt',
        'state-nd',
        'state-sd',
        'state-ne',
        'state-ks',
        'state-tx'
      ]),
      stage: 'prior'
    },
    {
      id: 'empirical-bayes',
      title: 'Shrink each rate by how much it can be trusted',
      headline: 'Shrinkage tames small counties and spares large ones',
      textAlternative:
        'The same Plains map split by a divider: raw rates on the left, empirical-Bayes rates on the right on identical classes; the rings stay where they were but the extreme colours fade.',
      body: 'Empirical Bayes estimates how far true county rates really differ (a spread of **{{trueSpread}}**) and pulls each raw rate toward the pooled rate, more for smaller counties. Drag the divider: in **{{smallLabel}}** the raw rates spread **{{smallRaw}}** but the smoothed ones **{{smallEb}}**. **Fade unreliable counties** shows how much of each estimate is its own.\n\n*Shrinkage trades a little bias for much less noise.*',
      options: {map: 'smoothed', swipe: 'raw-eb', showExtremes: true},
      optionsMode: 'fresh',
      controls: ['fadeUnreliable'],
      readouts: ['funnel', 'trueSpread', 'smallRaw', 'smallEb'],
      camera: {bounds: PLAINS_BOUNDS, transitionMs: 1000},
      compare: {labels: ['Raw rate', 'Empirical Bayes'], position: 0.5},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Shrink each rate toward what its size supports',
          'Raw | empirical Bayes, same classes'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, ['state-mt', 'state-sd']),
      stage: 'shrink'
    },
    {
      id: 'spatial-eb',
      title: 'Borrow strength from neighbours, not the nation',
      headline: 'Neighbours supply a local prior',
      textAlternative:
        'Mountain West counties split by a divider: national empirical Bayes on the left, spatial empirical Bayes on the right; a clicked county has its neighbourhood outlined.',
      body: "Spatial empirical Bayes pools each county with its queen neighbours (**{{neighbourMean}}** on average) instead of the nation. Drag the divider, then click a county to outline its neighbourhood and read its local rate (**{{localPool}}**). Under **Neighbour definition**, rook counts shared edges only. Moran's I rises from **{{moranEb}}** to **{{moranSpatial}}**, partly because pooling builds similarity in.\n\n*Pooling with neighbours smooths the map and builds similarity into it.*",
      options: {map: 'spatial', swipe: 'eb-spatial'},
      optionsMode: 'fresh',
      controls: ['criterion'],
      readouts: ['neighbourMean', 'moranEb', 'moranSpatial', 'localPool'],
      camera: {bounds: MOUNTAIN_BOUNDS, transitionMs: 1600},
      compare: {labels: ['Empirical Bayes', 'Spatial EB'], position: 0.5},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Borrow strength from neighbours',
          'Empirical Bayes | spatial EB, same classes'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, [
        'state-mt',
        'state-id',
        'state-wy',
        'state-nv',
        'state-ut',
        'state-co'
      ]),
      stage: 'neighbours'
    },
    {
      id: 'standardized',
      title: 'How surprising is each rate?',
      headline: 'Standardising asks how surprising each rate is',
      textAlternative:
        'US county map in seven red-blue classes by standardised rate: most counties are pale and as expected, with red counties above and blue counties below the pooled rate.',
      body: 'Each rate is compared with the pooled rate in units of its own sampling noise: z = (rate - pooled) / sqrt(a + m / b). **{{beyond196}}** counties lie beyond 1.96 (**{{above196}}** above, **{{below196}}** below), the dots outside the curves in the chart. Tiny counties need a huge excess to qualify. Flip **Map shows** to compare rate and z.\n\n*z answers how surprising, not how high.*',
      options: {map: 'standardized'},
      optionsMode: 'fresh',
      controls: ['map'],
      readouts: ['funnel', 'beyond196', 'above196', 'below196'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche('How surprising is each rate?', 'Standardised rate (Assuncao-Reis z)'),
        credit: COUNTY_CREDIT
      },
      stage: 'prior'
    },
    {
      id: 'clusters',
      title: 'Where high and low birth rates cluster',
      headline: 'High and low birth rates cluster by region',
      textAlternative:
        'US county cluster map: red high-high clusters and blue low-low clusters in separate regions, with most counties not significant and drawn as a pale ghost.',
      body: "Local Moran on the smoothed rate, tested by **{{testName}}**, finds **{{hhCount}}** high-high counties, led by **{{hhRegion}}**, and **{{llCount}}** low-low, led by **{{llRegion}}**; **{{notSignificantShare}}** are not significant. Try **Benjamini-Hochberg FDR** and **Significance level**. Limits: births are counted by mother's residence, and students inflate college-town denominators. LISA itself is in *Hot spots beyond chance*.\n\n*A cluster map says where; the test says whether.*",
      options: {map: 'cluster', gating: 'permutation', permutations: 499},
      optionsMode: 'fresh',
      controls: ['falseDiscoveryRate', 'significance'],
      readouts: ['hhCount', 'llCount', 'notSignificantShare', 'clusterBars'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Where do high and low rates cluster?',
          'Local Moran of the empirical-Bayes rate'
        ),
        credit: COUNTY_CREDIT
      },
      stage: 'clusters'
    }
  ]
});
