// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import type {ClassTable} from '../../cartography/types';
import {defineScene, type LegendSpec} from '../scene';
import {getVariableInfo, VARIABLES} from './b4-geography';
import type {GlobalAutocorrelationOptions} from './global-autocorrelation.compute';
import {
  BLACK_JOIN,
  getBivariateColors,
  getQuadrantColors,
  WHITE_JOIN
} from './global-autocorrelation.style';

const MAXIMUM_BANDS_OPTION = 32;

/** The conterminous US as one frame (the national county frame of the chapter). */
const CONUS_BOUNDS = [-124.8, 24.4, -66.9, 49.4] as const;

/** The South-East, for the bivariate step. */
const SOUTH_EAST_BOUNDS = [-98.5, 25, -74.5, 40.5] as const;

const ALTERNATIVES = [
  {
    value: 'directed',
    label: 'Directed (esda default)',
    help: 'The tail on the side of the observed statistic, as esda p_sim.'
  },
  {value: 'two-sided', label: 'Two-sided', help: 'Doubles the smaller tail.'},
  {value: 'greater', label: 'Greater', help: 'Tests for clustering stronger than chance.'},
  {
    value: 'lesser',
    label: 'Less',
    help: 'Tests for dispersion (neighbours less alike than chance).'
  },
  {
    value: 'folded',
    label: 'Folded',
    help: 'Compares |simulated - mean| with |observed - mean|; symmetric.'
  }
] as const;

const VARIABLE_OPTIONS = VARIABLES.map(variable => ({
  value: variable.id,
  label: variable.label,
  help: variable.help
}));

/** Credit of every step: the data, the boundaries and the colour tables. */
const CREDIT = joinCredits(
  'CDC PLACES, 2024 release (public domain)',
  CREDITS.usCensus,
  'Census SAIPE 2022 (income)',
  CREDITS.colorBrewer
);

/** Cartouche of one step: a claim or question, the variable and method, and the honesty chip. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  sample: 'CDC PLACES 2024 model-based estimates; 3,109 counties of the contiguous US',
  chips: ['Modelled'] as const
});

/** Annotations of the national county steps: regions of the gazetteer, shown from the national zoom. */
const regionLabels = (ids: readonly string[]) =>
  labelsFor(US, ids, Object.fromEntries(ids.map(id => [id, {minZoom: 3, tone: 'ink' as const}])));

function getLegends(
  state: GlobalAutocorrelationOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const first = getVariableInfo(state.variable);
  const second = getVariableInfo(state.secondVariable);
  switch (state.display) {
    case 'value':
    case 'lag': {
      const table = data['valueTable'] as ClassTable | undefined;
      if (!table) return [];
      const lagged = state.display === 'lag';
      return [
        getClassTableLegend(table, {
          title: lagged ? `${first.label}, neighbour average` : first.label,
          id: 'value-classes',
          counts: data[lagged ? 'lagCounts' : 'valueCounts'] as number[] | undefined,
          interactive: true,
          layout: 'list',
          note: lagged
            ? 'The same classes as the values map: each county shows the average of its neighbours, so the map is smoother.'
            : `${table.method}. The lag and shuffled maps keep these breaks.`
        })
      ];
    }
    case 'second': {
      const table = data['secondTable'] as ClassTable | undefined;
      if (!table) return [];
      return [
        getClassTableLegend(table, {
          title: second.label,
          id: 'second-classes',
          counts: data['secondCounts'] as number[] | undefined,
          interactive: true,
          layout: 'list'
        })
      ];
    }
    case 'bivariate': {
      const invert = state.secondVariable === 'income';
      return [
        {
          kind: 'bivariate',
          title: `${first.label} with ${second.label.toLowerCase()}`,
          size: 3,
          colors:
            (data['bivariateColors'] as ReturnType<typeof getBivariateColors> | undefined) ??
            getBivariateColors('light'),
          xLabel: first.label,
          yLabel: invert ? `${second.label}, inverted` : second.label,
          xEnds: ['lower', 'higher'],
          yEnds: invert ? ['higher', 'lower'] : ['lower', 'higher'],
          note: `Tertiles of each variable. The darkest cell is high ${first.label.toLowerCase()} with ${invert ? 'low' : 'high'} ${second.label.toLowerCase()}.`
        }
      ];
    }
    case 'quadrant': {
      const colors = getQuadrantColors('light');
      const counts = data['quadrantCounts'] as number[] | undefined;
      const quadrantColors =
        (data['quadrantColors'] as ReturnType<typeof getQuadrantColors> | undefined) ?? colors;
      return [
        {
          kind: 'categories',
          id: 'quadrant-classes',
          title: 'Moran scatterplot quadrant (signs only, not tested)',
          layout: 'list',
          interactive: true,
          entries: [
            {color: quadrantColors.highHigh, label: 'High-High', count: counts?.[0]},
            {color: quadrantColors.lowLow, label: 'Low-Low', count: counts?.[2]},
            {color: quadrantColors.highLow, label: 'High-Low', count: counts?.[3]},
            {color: quadrantColors.lowHigh, label: 'Low-High', count: counts?.[1]},
            {
              color: (data['noDataColor'] as [number, number, number, number] | undefined) ?? [
                217, 217, 217, 178
              ],
              label: 'No neighbours (island)',
              count: counts?.[4]
            }
          ],
          note: 'High or low against the mean, for the county and for the average of its neighbours. No significance test: see Hot spots beyond chance.'
        }
      ];
    }
    default:
      return [
        {
          kind: 'categories',
          title: 'Join-count colouring',
          entries: [
            {color: BLACK_JOIN, label: `Black: above the ${state.joinPercentile}th percentile`},
            {color: WHITE_JOIN, label: 'White: the rest'}
          ],
          note: 'A join is a pair of neighbours; BB, BW and WW count the pairs by colour.'
        }
      ];
  }
}

function getSnippet(state: GlobalAutocorrelationOptions): string {
  const statistic = state.statistic;
  const permutationLine =
    statistic === 'joinCount'
      ? '// Join counts have an analytic test only; there is no permutation variant.'
      : `graph.add(new GPUGlobalPermutationTest({
  weights, values${statistic === 'bivariateMoran' ? ', secondValues' : ''}, mask,
  statistic: '${statistic}', alternative: '${state.alternative}',
  parameters,                       // seed + permutations: a buffer write
  maximumPermutations: 999, results, histogram
}));
parameters.write(getGPUPermutationParameterValues({seed: ${state.seed}, permutations: ${state.permutations}}));`;
  return `import {
  GPUSpatialLag, GPUGlobalSpatialStatistics, GPUGlobalPermutationTest, getGPUPermutationParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUSpatialCorrelogram, getGPUSpatialCorrelogramParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// The neighbour average of every county: the y axis of the Moran scatterplot.
graph.add(new GPUSpatialLag({weights, values, mask, normalize: true, output: lag}));

// Analytic expectation, variance, z and p of every global statistic, one pass.
graph.add(new GPUGlobalSpatialStatistics({
  weights, values, secondValues, mask,
  statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran'],   // or 'joinCount' on a 0/1 column
  results                           // laid out by GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
}));

// Permutation reference distribution (esda p_sim, z_sim) and its histogram.
${permutationLine}

// Moran's I at a ladder of distance bands.
graph.add(new GPUSpatialCorrelogram({
  positions: centroids, values, mask, parameters: correlogramParameters,
  gridSize: [128, 128], bandCount: ${state.bandCount}, bandMode: '${state.bandMode}',
  moransI, zScores, pValues, peakBands
}));
correlogramParameters.write(getGPUSpatialCorrelogramParameterValues({
  bounds, maximumDistance: ${state.maxDistanceFactor} * medianSpacing,
  varianceAssumption: '${state.varianceAssumption}'
}));`;
}

/**
 * Global autocorrelation of health and income: one number, the Moran scatterplot behind it, the
 * permutation null, the neighbour rule and the distance. GPU work is in
 * `global-autocorrelation.compute.ts`.
 */
export default defineScene<GlobalAutocorrelationOptions>({
  id: 'global-autocorrelation',
  title: 'Is it clustered at all?',
  chapter: 'weights',
  order: 2,
  summary:
    "Moran's I of county diabetes and income as the slope of the Moran scatterplot, with a permutation null, the neighbour rule, a bivariate version and a correlogram of the distance at which clustering fades.",
  contributors: [
    'GPUGlobalSpatialStatistics',
    'GPUGlobalPermutationTest',
    'GPUSpatialCorrelogram',
    'GPUNeighborSearch',
    'GPUContiguityWeights',
    'GPUSpatialLag',
    'GPUSpatialWeightsTransform'
  ],
  datasets: [
    {id: 'us-counties', role: 'counties with CDC PLACES health measures and income'},
    {id: 'us-states', role: 'state lines over the counties'},
    {id: 'chicago-tracts', role: 'tracts with PLACES health measures (alternate geography)'},
    {id: 'chicago-community-areas', role: 'names of the community areas'}
  ],
  initialView: {longitude: -96, latitude: 38.3, zoom: 3.9},

  options: [
    {
      kind: 'select',
      id: 'geography',
      label: 'Geography',
      group: 'Variable',
      apply: 'compile',
      default: 'us-counties',
      help: 'Switching rebuilds every graph for the new row count.',
      options: [
        {value: 'us-counties', label: 'US counties (3,109)'},
        {value: 'chicago-tracts', label: 'Chicago census tracts (791)'}
      ]
    },
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable x',
      group: 'Variable',
      apply: 'param',
      default: 'diabetes',
      help: 'The variable the univariate statistics, the permutation test, the join counts and the correlogram describe.',
      options: VARIABLE_OPTIONS
    },
    {
      kind: 'select',
      id: 'secondVariable',
      label: 'Variable y (bivariate Moran)',
      group: 'Variable',
      apply: 'param',
      default: 'income',
      help: "The second variable of bivariate Moran's I: is x here similar to y around here?",
      options: VARIABLE_OPTIONS
    },
    {
      kind: 'select',
      id: 'source',
      label: 'Neighbours are...',
      group: 'Weights',
      apply: 'compile',
      default: 'queen',
      help: 'Every statistic reads the same weights matrix W. Change the rule and watch the statistics move.',
      options: [
        {value: 'queen', label: 'Queen contiguity'},
        {value: 'rook', label: 'Rook contiguity'},
        {value: 'knn', label: 'k nearest centroids'},
        {value: 'band', label: 'Distance band'}
      ]
    },
    {
      kind: 'slider',
      id: 'k',
      label: 'Nearest neighbours (k)',
      group: 'Weights',
      apply: 'compile',
      min: 2,
      max: 16,
      step: 1,
      default: 6,
      disabledWhen: state => state.source !== 'knn',
      help: 'k of the nearest-neighbour rule. Compile-time.'
    },
    {
      kind: 'slider',
      id: 'bandFactor',
      label: 'Distance band',
      group: 'Weights',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 1.5,
      disabledWhen: state => state.source !== 'band',
      format: value => `${value} x spacing`,
      help: 'Band radius in multiples of the median nearest-centroid distance.'
    },
    {
      kind: 'select',
      id: 'transform',
      label: 'Weight transform',
      group: 'Weights',
      apply: 'compile',
      default: 'row',
      help: "The statistics use the weights as given, so the transform matters. Row standardisation is the convention for Moran's I and makes the scatterplot slope equal I; the join counts always use the binary pattern.",
      options: [
        {value: 'row', label: 'Row standardised (R)'},
        {value: 'none', label: 'Binary (as produced)'},
        {value: 'double', label: 'Double standardised (D)'},
        {value: 'variance', label: 'Variance stabilising (V)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRuleSweep',
      label: 'Compare the four neighbour rules',
      group: 'Weights',
      apply: 'compile',
      default: false,
      help: "Computes Moran's I (row standardised) under queen, rook, k nearest and distance band one after the other, on a second weights set that never touches the map. Each rule's weights graph is compiled once and kept."
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic to test by permutation',
      group: 'Permutation test',
      apply: 'compile',
      default: 'moran',
      help: 'Which global statistic GPUGlobalPermutationTest simulates. Compile-time: each statistic and tail is its own compiled graph, built on first use.',
      options: [
        {value: 'moran', label: "Moran's I"},
        {value: 'geary', label: "Geary's C"},
        {value: 'getisOrdG', label: 'Getis-Ord General G'},
        {value: 'bivariateMoran', label: "Bivariate Moran's I (x with the lag of y)"},
        {value: 'joinCount', label: 'Join counts (analytic only)'}
      ]
    },
    {
      kind: 'select',
      id: 'alternative',
      label: 'Tail of the p-value',
      group: 'Permutation test',
      apply: 'compile',
      default: 'directed',
      disabledWhen: state => state.statistic === 'joinCount',
      help: 'Which tail counts as evidence. Compile-time.',
      options: ALTERNATIVES
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Permutations',
      group: 'Permutation test',
      apply: 'param',
      min: 99,
      max: 999,
      step: 100,
      default: 999,
      disabledWhen: state => state.statistic === 'joinCount',
      help: 'More permutations give a finer pseudo p-value: the smallest possible is 1 / (P + 1). A parameter write.'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Shuffle seed',
      group: 'Permutation test',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      display: 'stepper',
      format: value => `seed ${value}`,
      help: 'The same seed always gives the same null distribution (a counter-based Philox generator) and the same shuffled map.'
    },
    {
      kind: 'button',
      id: 'shuffle',
      label: 'Shuffle again (next seed)',
      group: 'Permutation test',
      help: 'Moves to the next seed: a new shuffled map and a new null distribution.'
    },
    {
      kind: 'toggle',
      id: 'showShuffled',
      label: 'Show a shuffled map beside the real one',
      group: 'Permutation test',
      apply: 'param',
      default: false,
      help: 'Deals the same county values out at random (seeded, on the CPU) and draws that map left of the swipe divider. The classes are the real map classes.'
    },
    {
      kind: 'slider',
      id: 'joinPercentile',
      label: 'Join counts: black above the percentile',
      group: 'Join counts',
      apply: 'param',
      min: 50,
      max: 95,
      step: 5,
      default: 80,
      unit: '%',
      help: 'Join counts need a binary variable: places above this percentile of x are black, the rest white.'
    },
    {
      kind: 'slider',
      id: 'bandCount',
      label: 'Correlogram: distance bands',
      group: 'Correlogram',
      apply: 'compile',
      min: 4,
      max: MAXIMUM_BANDS_OPTION,
      step: 2,
      default: 16,
      expert: true,
      help: 'Number of distance bands of GPUSpatialCorrelogram (up to 64 in the contributor; 32 here). Compile-time.'
    },
    {
      kind: 'select',
      id: 'bandMode',
      label: 'Band type',
      group: 'Correlogram',
      apply: 'compile',
      default: 'cumulative',
      display: 'segmented',
      help: 'Cumulative bands hold every pair up to their distance (PySAL DistanceBand thresholds, the ArcGIS incremental autocorrelation). Annulus bands hold only the pairs between the previous distance and theirs.',
      options: [
        {value: 'cumulative', label: 'Cumulative'},
        {value: 'annulus', label: 'Annulus'}
      ]
    },
    {
      kind: 'slider',
      id: 'maxDistanceFactor',
      label: 'Maximum distance',
      group: 'Correlogram',
      apply: 'param',
      min: 4,
      max: 60,
      step: 2,
      default: 30,
      format: value => `${value} x spacing`,
      help: 'Upper distance of the last band, in median centroid spacings. The readout shows it in kilometres. A parameter write.'
    },
    {
      kind: 'select',
      id: 'varianceAssumption',
      label: 'Correlogram: variance assumption',
      group: 'Correlogram',
      apply: 'param',
      default: 'randomization',
      help: "Null hypothesis behind each band's z-score: values permuted over places (esda default), or independent normal draws.",
      options: [
        {value: 'randomization', label: 'Randomisation'},
        {value: 'normality', label: 'Normality'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'value',
      help: 'Which buffer the fill reads. All value maps use the same frozen quantile classes.',
      options: [
        {value: 'value', label: 'Variable x'},
        {value: 'lag', label: 'Neighbour average (spatial lag of x)'},
        {value: 'quadrant', label: 'Moran scatterplot quadrants'},
        {value: 'bivariate', label: 'x with y (bivariate)'},
        {value: 'second', label: 'Variable y'},
        {value: 'binary', label: 'Join-count colours (black / white)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBands',
      label: 'Draw the correlogram distance rings',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Two rings around the focus county: dashed at the longest distance, solid where the correlogram z first peaks (or is highest). Click the map to move the focus.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Draw boundaries',
      group: 'Display',
      apply: 'param',
      default: true
    }
  ],

  readouts: [
    {id: 'places', label: 'Places analysed', help: 'Places with a finite value of x.'},
    {
      id: 'moran',
      label: "Moran's I",
      emphasis: 'tile',
      help: 'I = n / S0 * sum w z_i z_j / sum z^2. The slope of the Moran scatterplot when the weights are row standardised.'
    },
    {
      id: 'moranZ',
      label: 'z-score (randomisation)',
      help: "Moran's I less its expectation, in standard deviations under randomisation."
    },
    {
      id: 'expected',
      label: 'Expected I with no pattern',
      hood: true,
      help: 'E[I] = -1 / (n - 1).'
    },
    {
      id: 'quadrantShare',
      label: 'In High-High or Low-Low',
      emphasis: 'tile',
      help: 'Share of counties with neighbours that sit with the same sign as their neighbours (the two cornered quadrants).'
    },
    {
      id: 'moranScatter',
      label: 'Moran scatterplot',
      kind: 'chart',
      help: 'Value against neighbour average, both as standard scores. Drag to select counties.'
    },
    {
      id: 'pSim',
      label: 'Pseudo p-value',
      help: 'p_sim = (larger + 1) / (P + 1) for the selected statistic and tail.'
    },
    {
      id: 'pFloor',
      label: 'Smallest possible p',
      help: '1 / (P + 1): even a perfect result cannot go below it.'
    },
    {
      id: 'permutations',
      label: 'Shuffles',
      help: 'Permutations P of the selected statistic.'
    },
    {id: 'exceedances', label: 'Shuffles beyond observed', help: 'The count behind p_sim.'},
    {
      id: 'shuffledI',
      label: 'I of the shuffled map shown',
      help: "Moran's I of the seeded shuffled map, computed by a second GPUGlobalSpatialStatistics pass."
    },
    {
      id: 'nullDistribution',
      label: 'Null distribution',
      kind: 'chart',
      help: 'Histogram of the statistic over the shuffles, with the observed value marked.'
    },
    {
      id: 'zSim',
      label: 'z of the observed value among shuffles',
      hood: true,
      help: 'esda z_sim: (observed - simulated mean) / simulated standard deviation.'
    },
    {
      id: 'iByW',
      label: "Moran's I by neighbour rule",
      kind: 'chart',
      help: "Row-standardised Moran's I under each rule; the current rule is highlighted."
    },
    {
      id: 'iRange',
      label: 'Range of I over the rules',
      help: 'Lowest and highest I of the four rules.'
    },
    {
      id: 'bivariate',
      label: "Bivariate Moran's I (x, y)",
      emphasis: 'tile',
      help: 'x here against the neighbour average of y.'
    },
    {
      id: 'darkCorner',
      label: 'Counties in the darkest class',
      help: 'High x with the unfavourable end of y (low income).'
    },
    {id: 'geary', label: "Geary's C", hood: true, help: 'Below 1 means similar neighbours.'},
    {
      id: 'getis',
      label: 'General G',
      hood: true,
      help: 'Above its expectation means high values cluster.'
    },
    {id: 'joins', label: 'Joins BB / BW / WW', hood: true},
    {id: 'joinTest', label: 'Join-count tests', hood: true},
    {
      id: 'correlogram',
      label: "Moran's I by distance",
      kind: 'chart',
      help: "Global Moran's I of neighbours within each distance, with the z-score on the right axis."
    },
    {id: 'peak', label: 'z peak (distance at the focus county)'},
    {id: 'maxDistance', label: 'Longest distance (at the focus county)'},
    {id: 'bandSpan', label: 'Correlogram span', hood: true},
    {
      id: 'distanceNote',
      label: 'Distance units',
      hood: true,
      help: 'The correlogram measures distance in the planar metres of the map frame.'
    }
  ],

  pipeline: [
    {
      id: 'weights',
      label: 'Weights',
      detail: 'Who is whose neighbour, row standardised',
      show: {option: 'showRuleSweep', value: true}
    },
    {
      id: 'lag',
      label: 'Lag',
      detail: 'The average of the neighbours of every county',
      show: {option: 'display', value: 'lag'}
    },
    {
      id: 'sums',
      label: 'Global sums',
      detail: 'One pass: mean, S0, S1, S2, then I, z and p',
      show: {option: 'display', value: 'value'}
    },
    {
      id: 'permutation',
      label: 'Permutations',
      detail: 'Shuffles by a keyed bijection, no n x P table',
      show: {option: 'showShuffled', value: true}
    },
    {
      id: 'correlogram',
      label: 'Correlogram',
      detail: 'Every distance band in one pass over the pairs',
      show: {option: 'showBands', value: true}
    }
  ],

  legends: getLegends,
  snippet: getSnippet,

  basemap: ground('paperSheet'),
  furniture: {
    ...NATIONAL_FURNITURE,
    title: cartouche(
      'Is diabetes clustered?',
      'Age-adjusted prevalence, CDC PLACES 2024, quantile classes'
    ),
    credit: CREDIT
  },
  annotations: [],

  about: {
    what: "`GPUGlobalSpatialStatistics` computes one number for the whole map: Moran's I, Geary's C, Getis-Ord General G, bivariate Moran's I and join counts, each with its expectation, variance, z-score and p-value. `GPUGlobalPermutationTest` re-computes the statistic for hundreds of random rearrangements to build the reference distribution, and `GPUSpatialCorrelogram` repeats Moran's I at growing distances.",
    why: 'Before mapping local hot spots or fitting a regression, an analyst needs to know that there is a spatial pattern at all, how strong it is, and at what distance it fades.',
    howToRead:
      "Moran's I is the slope of the Moran scatterplot: well above its expectation (about 0) means similar values sit next to each other. The permutation histogram shows what shuffled maps give; the observed value is marked on it. The correlogram shows I by distance: the solid ring marks where the z-score peaks."
  },

  create: async ctx =>
    (await import('./global-autocorrelation.compute')).createGlobalAutocorrelation(ctx),

  story: [
    {
      id: 'clustered',
      title: 'Is diabetes clustered across US counties?',
      headline: 'A belt of high diabetes crosses the South',
      textAlternative:
        'Map of the contiguous US in five classes of county diabetes prevalence, pale yellow to dark brown: the darkest counties form a belt across the South and Appalachia, the palest lie in the Rockies and the Upper Midwest.',
      body: "Five equal-count classes show a belt of high prevalence across the South and Appalachia, and low values from the Rockies to the Upper Midwest.\n\n**Moran's I** puts one number on the pattern: **{{moran}}** over {{places}} counties, z = {{moranZ}}. With no pattern it would be close to zero. Compare another **Variable**.\n\n*Near things are more alike: Moran's I measures how much.*",
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        source: 'queen',
        transform: 'row',
        statistic: 'moran',
        display: 'value'
      },
      controls: ['variable'],
      readouts: ['moran', 'moranZ', 'places'],
      stage: 'sums',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Is diabetes clustered?',
          'Age-adjusted prevalence, CDC PLACES 2024, quantile classes'
        ),
        credit: CREDIT
      },
      annotations: regionLabels([
        'black-belt',
        'mississippi-delta',
        'appalachia',
        'rio-grande-valley',
        'front-range',
        'great-plains'
      ])
    },
    {
      id: 'scatterplot',
      title: "Moran's I is a slope",
      headline: "Moran's I is the slope of a scatterplot",
      textAlternative:
        'Map of US counties coloured by Moran quadrant, beside a scatterplot of each county against the average of its neighbours: the cloud rises from lower left to upper right along a fitted line.',
      body: "Each dot is a county: its value (across) against the average of its neighbours (up), both as standard scores. The line through the cloud has slope **{{moran}}**, Moran's I; **{{quadrantShare}}** of counties sit in the High-High or Low-Low corners.\n\nDrag on the chart to outline counties on the map; **Map shows** switches between values, neighbour average and quadrants (signs only, not tested).\n\n*Autocorrelation is a county against its surroundings.*",
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        source: 'queen',
        transform: 'row',
        statistic: 'moran',
        display: 'quadrant'
      },
      controls: ['display'],
      readouts: ['moranScatter', 'moran', 'quadrantShare'],
      stage: 'lag',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          "Moran's I is a slope",
          'Quadrants of value and neighbour average, signs only'
        ),
        credit: CREDIT
      },
      annotations: regionLabels(['black-belt', 'front-range'])
    },
    {
      id: 'permutation',
      title: 'Shuffled maps never look like this',
      headline: 'No shuffled map comes close to the real one',
      textAlternative:
        'A swipe between a shuffled county map, which looks like salt and pepper, on the left and the real map with its southern belt on the right, above a histogram of Moran I over the shuffles with the observed value far to the right.',
      body: 'Deal the county values out at random and the belt vanishes: left is one shuffle (I = **{{shuffledI}}**), right the real map (I = **{{moran}}**). **{{permutations}}** shuffles built the histogram; **{{exceedances}}** reached the real I, so p_sim is **{{pSim}}** and cannot go below **{{pFloor}}**, 1 / (shuffles + 1).\n\nStep the **Shuffle seed**; change **Permutations**.\n\n*Beyond chance means beyond shuffling.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        source: 'queen',
        transform: 'row',
        statistic: 'moran',
        display: 'value',
        showShuffled: true,
        permutations: 999,
        seed: 1
      },
      controls: ['seed', 'permutations'],
      readouts: ['nullDistribution', 'pSim', 'pFloor', 'shuffledI'],
      stage: 'permutation',
      compare: {mode: 'swipe', labels: ['Shuffled', 'Real'], position: 0.5},
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Shuffled maps never look like this',
          'Same values, dealt at random to counties'
        ),
        credit: CREDIT
      },
      annotations: []
    },
    {
      id: 'which-w',
      title: 'Does the neighbour rule matter?',
      headline: 'Every rule finds the belt; the number moves',
      textAlternative:
        'A horizontal bar chart of Moran I under queen, rook, nearest-neighbour and distance-band rules, all strongly positive, beside the quadrant map under the chosen rule.',
      body: "Moran's I depends on who counts as a neighbour. Under four rules it runs over **{{iRange}}**; the highlighted bar is the rule chosen in **Neighbours are...**.\n\nEach rule is its own weights graph, compiled once and reused (see the cost line); the data never change.\n\n*A statistic belongs to its weights matrix.*",
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        source: 'queen',
        transform: 'row',
        statistic: 'moran',
        display: 'quadrant',
        showRuleSweep: true
      },
      controls: ['source'],
      readouts: ['iByW', 'moran', 'iRange'],
      stage: 'weights',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Does the neighbour rule matter?',
          "Moran's I under four weights rules, row standardised"
        ),
        credit: CREDIT
      },
      annotations: regionLabels(['black-belt'])
    },
    {
      id: 'two-variables',
      title: 'High diabetes sits beside low income',
      headline: 'High diabetes sits beside low income',
      textAlternative:
        'Bivariate map of the South-East: counties coloured by tertiles of diabetes and of income, with the darkest blue-violet for high diabetes and low income covering the Black Belt, Appalachia and the Rio Grande Valley.',
      body: "Counties are cut into tertiles of diabetes (across) and income (up, inverted), so the darkest colour is high diabetes with low income: **{{darkCorner}}** counties, dominating the Black Belt, Appalachia and the Rio Grande Valley. Bivariate Moran's I is **{{bivariate}}**, p_sim **{{pSim}}**: x here against y around here, not within a county.\n\nTry **Variable y** or **Map shows**.\n\n*A pattern across places is not a correlation within them.*",
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        secondVariable: 'income',
        source: 'queen',
        transform: 'row',
        statistic: 'bivariateMoran',
        display: 'bivariate'
      },
      controls: ['secondVariable', 'display'],
      readouts: ['bivariate', 'pSim', 'darkCorner'],
      stage: 'sums',
      camera: {bounds: SOUTH_EAST_BOUNDS, transitionMs: 1600},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche('Diabetes and income together', 'Tertiles of each, income inverted'),
        credit: CREDIT
      },
      annotations: regionLabels(['black-belt', 'appalachia', 'rio-grande-valley'])
    },
    {
      id: 'distance',
      title: 'How far does the clustering reach?',
      headline: 'Clustering fades with distance',
      textAlternative:
        'Map of US counties with two rings around Cook County: a dashed ring at the longest distance and a solid ring where the correlogram peaks, above a line chart of Moran I falling as the distance grows.',
      body: "Moran's I for neighbours within growing distances: z peaks near **{{peak}}**, the solid ring around the focus county (click the map to move it); the dashed ring is the longest distance, **{{maxDistance}}**.\n\nWiden **Maximum distance**, switch **Band type**, or set **Geography** to tracts: the zoning changes the answer. Every setting is open in All controls.\n\n*County values are model estimates, so part of the clustering may come from the model.*",
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        variable: 'diabetes',
        source: 'queen',
        transform: 'row',
        statistic: 'moran',
        display: 'value',
        showBands: true
      },
      controls: ['maxDistanceFactor', 'bandMode', 'geography'],
      readouts: ['correlogram', 'peak', 'maxDistance'],
      stage: 'correlogram',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'How far does the clustering reach?',
          "Moran's I by distance band, planar metres of the map frame"
        ),
        credit: CREDIT
      },
      annotations: []
    }
  ]
});
