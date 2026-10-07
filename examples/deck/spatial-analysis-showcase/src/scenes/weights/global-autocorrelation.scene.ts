// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {RAMP_NAMES} from '../../engine/ramps';
import {BLACK_JOIN_COLOR, MORAN_COLORS, WHITE_JOIN_COLOR} from './b4-colors';
import {getVariableInfo, VARIABLES} from './b4-geography';
import type {GlobalAutocorrelationOptions} from './global-autocorrelation.compute';

const MAXIMUM_BANDS_OPTION = 32;

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

function getDisplayLegend(state: GlobalAutocorrelationOptions): LegendSpec {
  const first = getVariableInfo(state.variable);
  const second = getVariableInfo(state.secondVariable);
  switch (state.display) {
    case 'quadrant':
      return {
        kind: 'categories',
        title: 'Moran scatterplot quadrant (not tested)',
        entries: [
          {color: MORAN_COLORS[1], label: 'High, high neighbours'},
          {color: MORAN_COLORS[3], label: 'Low, low neighbours'},
          {color: MORAN_COLORS[4], label: 'High among low'},
          {color: MORAN_COLORS[2], label: 'Low among high'}
        ],
        note: 'Compares the place and its weighted neighbours with the mean. The global I is the slope through this cloud. No significance test is applied here: see Local Moran in Hot spots.'
      };
    case 'binary':
      return {
        kind: 'categories',
        title: 'Join-count colouring',
        entries: [
          {color: BLACK_JOIN_COLOR, label: `Black: above the ${state.joinPercentile}th percentile`},
          {color: WHITE_JOIN_COLOR, label: 'White: the rest'}
        ],
        note: 'A join is a pair of neighbours; BB, BW and WW count the pairs by colour.'
      };
    case 'second':
      return {
        kind: 'ramp',
        id: 'display',
        title: second.label,
        ramp: state.ramp,
        extent: 'gpu',
        unit: second.unit,
        format: value => value.toFixed(second.digits)
      };
    case 'lag':
      return {
        kind: 'ramp',
        id: 'display',
        title: `Spatial lag of ${first.label.toLowerCase()}`,
        ramp: state.ramp,
        extent: 'gpu',
        unit: first.unit,
        format: value => value.toFixed(first.digits)
      };
    default:
      return {
        kind: 'ramp',
        id: 'display',
        title: first.label,
        ramp: state.ramp,
        extent: 'gpu',
        unit: first.unit,
        format: value => value.toFixed(first.digits)
      };
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
  GPUGlobalSpatialStatistics, GPUGlobalPermutationTest, getGPUPermutationParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUSpatialCorrelogram, getGPUSpatialCorrelogramParameterValues} from '@luma.gl/experimental/gpu-dataframe';

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
 * Global autocorrelation of health and income. GPU work is in
 * `global-autocorrelation.compute.ts`.
 */
export default defineScene<GlobalAutocorrelationOptions>({
  id: 'global-autocorrelation',
  title: 'Is it clustered at all?',
  chapter: 'weights',
  order: 2,
  summary:
    "Moran's I, Geary's C, General G, bivariate Moran and join counts for county health and income, with permutation reference distributions and a spatial correlogram of the distance at which clustering peaks.",
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
    {id: 'chicago-tracts', role: 'tracts with PLACES health measures (alternate geography)'},
    {id: 'chicago-community-areas', role: 'names of the community areas'}
  ],
  initialView: {longitude: -95.5, latitude: 38.2, zoom: 3.9},

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
      help: "The statistics use the weights as given, so the transform matters. Row standardisation is the convention for Moran's I; the join counts always use the binary pattern.",
      options: [
        {value: 'row', label: 'Row standardised (R)'},
        {value: 'none', label: 'Binary (as produced)'},
        {value: 'double', label: 'Double standardised (D)'},
        {value: 'variance', label: 'Variance stabilising (V)'}
      ]
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
      label: 'Random seed',
      group: 'Permutation test',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      disabledWhen: state => state.statistic === 'joinCount',
      help: 'The same seed always gives the same null distribution (a counter-based Philox generator).'
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
      help: 'Number of distance bands of GPUSpatialCorrelogram (up to 64 in the contributor; 32 here). Compile-time.'
    },
    {
      kind: 'select',
      id: 'bandMode',
      label: 'Correlogram: band type',
      group: 'Correlogram',
      apply: 'compile',
      default: 'cumulative',
      help: 'Cumulative bands hold every pair up to their distance (PySAL DistanceBand thresholds, the ArcGIS incremental autocorrelation). Annulus bands hold only the pairs between the previous distance and theirs.',
      options: [
        {value: 'cumulative', label: 'Cumulative (within d)'},
        {value: 'annulus', label: 'Annulus (between d and the previous band)'}
      ]
    },
    {
      kind: 'slider',
      id: 'maxDistanceFactor',
      label: 'Correlogram: maximum distance',
      group: 'Correlogram',
      apply: 'param',
      min: 4,
      max: 60,
      step: 2,
      default: 30,
      format: value => `${value} x spacing`,
      help: 'Upper distance of the last band, in median centroid spacings. A parameter write.'
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
      help: 'Which buffer the fill reads.',
      options: [
        {value: 'value', label: 'Variable x'},
        {value: 'second', label: 'Variable y'},
        {value: 'lag', label: 'Spatial lag of x'},
        {value: 'quadrant', label: 'Moran scatterplot quadrants'},
        {value: 'binary', label: 'Join-count colours (black / white)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.display === 'quadrant' || state.display === 'binary',
      options: RAMP_NAMES.filter(name => name !== 'diverging').map(name => ({
        value: name,
        label: name.charAt(0).toUpperCase() + name.slice(1)
      }))
    },
    {
      kind: 'toggle',
      id: 'showBands',
      label: 'Draw the correlogram distance rings',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Rings around the focus place at every band edge; the orange ring marks the first peak of the correlogram. Click the map to move the centre.'
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
    {id: 'places', label: 'Places analysed'},
    {id: 'summary', label: 'Mean of x, S0'},
    {
      id: 'moran',
      label: "Moran's I",
      help: 'I = n / S0 * sum w z_i z_j / sum z^2. Expected value -1 / (n - 1) under no pattern.'
    },
    {id: 'moranTest', label: 'Moran z (randomisation / normality), p'},
    {
      id: 'geary',
      label: "Geary's C",
      help: 'Squared differences between neighbours; below 1 means similar neighbours.'
    },
    {
      id: 'getis',
      label: 'General G',
      help: 'Cross-product of values within neighbourhoods; above its expectation means high values cluster.'
    },
    {id: 'bivariate', label: 'Bivariate Moran I (x, y)'},
    {id: 'joins', label: 'Joins BB / BW / WW'},
    {id: 'joinTest', label: 'Join-count tests'},
    {
      id: 'permutation',
      label: 'Permutation test',
      help: 'The selected statistic: observed value, p_sim = (larger + 1) / (P + 1) and z_sim.'
    },
    {
      id: 'nullDistribution',
      label: 'Null distribution',
      help: 'Histogram of the simulated statistic; compare the range with the observed value above.'
    },
    {id: 'correlogram', label: "Moran's I by distance"},
    {id: 'correlogramZ', label: 'z-score by distance'},
    {id: 'peak', label: 'First peak of the correlogram'},
    {id: 'bandSpan', label: 'Correlogram span'}
  ],

  legends: state => [getDisplayLegend(state)],

  snippet: getSnippet,

  about: {
    what: "`GPUGlobalSpatialStatistics` computes one number for the whole map: Moran's I, Geary's C, Getis-Ord General G, bivariate Moran's I and join counts, each with its expectation, variance, z-score and p-value. `GPUGlobalPermutationTest` re-computes the statistic for hundreds of random rearrangements to build the reference distribution, and `GPUSpatialCorrelogram` repeats Moran's I at growing distances.",
    why: 'Before mapping local hot spots or fitting a regression, an analyst needs to know that there is a spatial pattern at all, how strong it is, and at what distance it fades.',
    howToRead:
      "A Moran's I well above its expectation (about 0) means similar values sit next to each other; a Geary's C below 1 says the same. The permutation test shows whether the observed value could come from a random arrangement: compare it with the range of the null distribution. The correlogram shows I by distance: the orange ring marks where clustering first peaks."
  },

  create: async ctx =>
    (await import('./global-autocorrelation.compute')).createGlobalAutocorrelation(ctx),

  story: [
    {
      id: 'clustered',
      title: 'Is diabetes clustered across US counties?',
      body: "The map shows age-adjusted diabetes prevalence (CDC PLACES) for the 3,109 counties of the contiguous United States. The eye sees a belt across the South. **`GPUGlobalSpatialStatistics`** puts a number on it: **Moran's I**, `I = (n / S0) * sum w_ij z_i z_j / sum z_i^2`, where `z` is the value minus the mean and `W` is the queen contiguity weights, row standardised (**Neighbours are...** and **Weight transform** below set that). With no pattern the expected value is `-1 / (n - 1)`, close to 0.\n\nRead **Moran's I** and its **z** below: for these counties I is about 0.70 and z is above 60, so clustering is overwhelming. The z and p come from analytic variances under *randomisation* and *normality*, as in esda `Moran`, and the whole statistic is one deterministic GPU pass.",
      options: {statistic: 'moran', display: 'value', source: 'queen', transform: 'row'},
      camera: {longitude: -95.5, latitude: 38.2, zoom: 3.9, transitionMs: 1400},
      highlight: {readout: 'moran'},
      controls: ['variable', 'source', 'transform'],
      readouts: ['moran', 'moranTest']
    },
    {
      id: 'scatterplot',
      title: 'The Moran scatterplot, drawn as a map',
      body: "Moran's I is the slope of the **Moran scatterplot**: each county's value `z` against the weighted mean of its neighbours (`GPUSpatialLag`). Colouring each county by its quadrant gives a map: red is **high with high neighbours**, blue **low with low neighbours**, and the two lighter colours are places unlike their surroundings.\n\nA strongly positive I means most counties sit in the red and blue quadrants. This map is *not* tested for significance: it only shows the signs. The next contributors in this chapter (Local Moran, in the hot-spots story) test each county. **Map shows** below switches back to the plain values.",
      options: {display: 'quadrant'},
      camera: {longitude: -95.5, latitude: 38.2, zoom: 3.9},
      highlight: {readout: 'moran'},
      controls: ['display'],
      readouts: ['moran']
    },
    {
      id: 'permutation',
      title: 'Check the formula with a permutation test',
      body: 'The analytic p-value assumes a distribution. A **permutation test** assumes nothing: shuffle the values over the counties 999 times, recompute I each time, and count how often chance beats the observed value. **`GPUGlobalPermutationTest`** does this without a 999 x 3,109 table: a keyed bijection (Philox counter generator and a Feistel network) relabels the values inside the kernel, so the result depends only on the seed.\n\nThe pseudo p-value is `(larger + 1) / (P + 1)`, so with 999 permutations the smallest possible is 0.001. Read **Permutation test** and the **Null distribution** sparkline below: simulated values sit around 0 and the observed 0.70 is far outside. Try another **Random seed** and a different **Tail of the p-value**: the result is reproducible and the tail changes how extremes count (esda `Moran` `p_sim`).',
      options: {statistic: 'moran', permutations: 999, seed: 1},
      highlight: {readout: 'permutation'},
      controls: ['permutations', 'seed', 'alternative'],
      readouts: ['permutation', 'nullDistribution']
    },
    {
      id: 'geary-g',
      title: "Geary's C and the General G look at different things",
      body: "**Geary's C** uses squared differences between neighbours, so it reacts to *local* contrasts: below 1 means neighbours are similar. The **General G** adds the products `x_i x_j` over neighbours: it exceeds its expectation when **high** values cluster and falls below it when **low** values do, something Moran's I cannot separate.\n\nHere G is above its expectation: diabetes clusters at the high end, as the red belt suggests. The permutation test is now running for G (**Tail of the p-value** set to *Greater* tests exactly that). Both match esda `Geary` and `G`.",
      options: {statistic: 'getisOrdG', alternative: 'greater', display: 'value'},
      highlight: {readout: 'getis'},
      controls: ['statistic', 'alternative'],
      readouts: ['geary', 'getis', 'permutation']
    },
    {
      id: 'bivariate',
      title: 'Do high diabetes counties border poor counties?',
      body: "**Bivariate Moran's I** relates one variable to the neighbours of another: `I_xy = (n / S0) * sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`. Here x is diabetes prevalence and y is median household income. A negative value means counties with a lot of diabetes are surrounded by counties with low income, even before asking whether the two are correlated within a county.\n\nThe map shows income. The permutation test now permutes y, as esda `Moran_BV` does, and the analytic bivariate variance is exact (esda only has the permutation version). Try swapping **Variable x** and **Variable y**, or setting **Geography** to *Chicago census tracts*.",
      options: {
        statistic: 'bivariateMoran',
        secondVariable: 'income',
        display: 'second',
        alternative: 'directed'
      },
      highlight: {readout: 'bivariate'},
      controls: ['variable', 'secondVariable', 'geography'],
      readouts: ['bivariate']
    },
    {
      id: 'join-counts',
      title: 'Join counts: a binary question',
      body: 'Sometimes the question is yes or no: is this county in the top fifth for diabetes? A **join count** looks at pairs of neighbours: **BB** (both black), **BW**, **WW**. If black counties were scattered at random, BB would equal its expectation; far more BB joins mean clustering. The map colours counties black or white at the percentile you set with **Join counts: black above the percentile**.\n\nRead **Joins BB / BW / WW** and **Join-count tests** below: BB is many times its expectation and BW far below it, so the top fifth form blocks rather than being sprinkled across the map. The counts are exact integers from the same pass (esda `Join_Counts`).',
      options: {statistic: 'joinCount', display: 'binary', joinPercentile: 80},
      highlight: {readout: 'joins'},
      controls: ['joinPercentile'],
      readouts: ['joins', 'joinTest']
    },
    {
      id: 'distance',
      title: 'At what distance does clustering fade?',
      body: "**`GPUSpatialCorrelogram`** repeats Moran's I with a distance band: every pair of centroids within d are neighbours. It runs all bands in one pass over the pairs. The **Moran's I by distance** sparkline below shows I falling as the band widens; the **orange ring** marks the first peak of the z-score (the ArcGIS *incremental spatial autocorrelation* rule) and thin rings mark the band edges. Click anywhere to move the rings.\n\n**Limits.** Moran's I depends on the weights (change **Neighbours are...** and compare), on how areas are drawn (the modifiable areal unit problem), and county values are modelled estimates, not measurements. Distances are Web Mercator metres. **Try it:** raise **Correlogram: maximum distance**, set **Correlogram: band type** to *Annulus*, or set **Geography** to *Chicago census tracts*.",
      options: {display: 'value', showBands: true, statistic: 'moran'},
      camera: {longitude: -95.5, latitude: 38.2, zoom: 3.9},
      highlight: {readout: 'peak'},
      controls: ['showBands', 'source', 'maxDistanceFactor', 'bandMode', 'geography'],
      readouts: ['correlogram', 'correlogramZ', 'peak']
    }
  ]
});
