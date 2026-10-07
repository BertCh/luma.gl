// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {RAMP_NAMES} from '../../engine/ramps';
import type {RateSmoothingOptions} from './rate-smoothing.compute';

/** Quadrant colors, duplicated from the compute module so this file stays light. */
const CLUSTER_LEGEND = [
  {color: [215, 48, 39, 235], label: 'High-high: high rates among high-rate neighbours'},
  {color: [49, 54, 149, 235], label: 'Low-low: low rates among low-rate neighbours'},
  {color: [253, 174, 97, 235], label: 'High-low outlier'},
  {color: [145, 191, 219, 235], label: 'Low-high outlier'},
  {color: [176, 182, 194, 120], label: 'Not significant'}
] as const;

const MAP_LABELS: Record<RateSmoothingOptions['map'], string> = {
  raw: 'Raw rate (deaths / person-years)',
  smoothed: 'Empirical Bayes rate',
  spatial: 'Spatial empirical Bayes rate',
  standardized: 'Standardized rate (Assuncao-Reis z)',
  cluster: 'Local Moran cluster map'
};

export default defineScene<RateSmoothingOptions>({
  id: 'rate-smoothing',
  title: 'Traffic deaths without the small-number noise',
  chapter: 'statistics',
  order: 1,
  summary:
    'Raw county death rates are dominated by tiny counties. Empirical Bayes shrinks them toward what their population can support, spatial EB borrows from neighbours, and a rate cluster map finds the places that are genuinely dangerous.',
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
    {id: 'us-county-mortality', role: 'traffic deaths 2017-2023 and population at risk'}
  ],
  initialView: {longitude: -96.5, latitude: 38.2, zoom: 3.55},

  options: [
    {
      kind: 'select',
      id: 'map',
      label: 'Map shows',
      group: 'Rate variable',
      apply: 'param',
      default: 'smoothed',
      help: 'Every variable is computed in one graph run; this only picks which output buffer the layer reads.',
      options: (Object.keys(MAP_LABELS) as RateSmoothingOptions['map'][]).map(value => ({
        value,
        label: MAP_LABELS[value]
      }))
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Rate variable',
      apply: 'param',
      default: 'magma',
      help: 'Applies to the three rate maps. They share one color range (the central 96% of the smoothed rates), so raw outliers saturate and the maps stay comparable.',
      disabledWhen: state => state.map === 'standardized' || state.map === 'cluster',
      options: RAMP_NAMES.filter(name => name !== 'diverging').map(value => ({
        value,
        label: value.charAt(0).toUpperCase() + value.slice(1)
      }))
    },
    {
      kind: 'select',
      id: 'criterion',
      label: 'Neighbour definition',
      group: 'Neighbours',
      apply: 'compile',
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
      default: 'standardized',
      help: 'The standardized rate is esda Moran_Local_Rate (adjusted): each rate is centred and scaled by its own sampling variance. The smoothed rate is the alternative the recipe supports.',
      options: [
        {value: 'standardized', label: 'Standardized rate (esda Moran_Local_Rate)'},
        {value: 'smoothed', label: 'Empirical Bayes smoothed rate'}
      ]
    },
    {
      kind: 'select',
      id: 'gating',
      label: 'Significance test',
      group: 'Cluster map',
      apply: 'compile',
      default: 'permutation',
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
      help: 'Controls the share of false discoveries over 3,109 simultaneous tests. Expect fewer, more trustworthy clusters.'
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
      help: 'More permutations resolve smaller pseudo p-values at proportional GPU cost. 499 resolves p = 0.002.'
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
      help: 'Counties whose p-value is above this level are drawn gray.'
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'County outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Thin boundaries help at regional zoom; at national zoom they hide the colors.'
    }
  ],

  readouts: [
    {id: 'counties', label: 'Counties', format: 'integer'},
    {
      id: 'pooled',
      label: 'Pooled national rate',
      help: 'Sum of deaths over sum of person-years: the prior mean every county is shrunk toward.'
    },
    {
      id: 'shrinkageWeight',
      label: 'Between-county variance',
      help: 'The estimated variance of the true rates, a. The shrinkage weight of a county is a / (a + m / b): large populations keep their own rate, small ones fall back to the pooled rate.'
    },
    {
      id: 'smallCounties',
      label: 'Counties under 5,000 residents (5th-95th pct, per 100k)',
      help: 'The spread of small-county rates before and after empirical Bayes smoothing.'
    },
    {
      id: 'moran',
      label: "Moran's I: raw / EB / spatial EB / standardized",
      help: 'Global spatial autocorrelation of each variable from GPUGlobalSpatialStatistics, on row-standardised weights.'
    },
    {id: 'clusters', label: 'Clusters HH / LH / LL / HL'},
    {id: 'notSignificant', label: 'Not significant'},
    {id: 'neighbors', label: 'Neighbours per county'},
    {id: 'selected', label: 'Selected county', help: 'Click a county to pin its numbers here.'}
  ],

  legends: state => {
    if (state.map === 'cluster') {
      return [
        {
          kind: 'categories',
          title: 'Local Moran cluster of the traffic-death rate',
          entries: CLUSTER_LEGEND,
          note:
            state.gating === 'permutation'
              ? `Conditional permutation, ${state.permutations} permutations, p <= ${state.significance}${state.falseDiscoveryRate ? ', FDR corrected' : ''}.`
              : `Analytic p <= ${state.significance}.`
        }
      ];
    }
    if (state.map === 'standardized') {
      return [
        {
          kind: 'ramp',
          title: 'Standardized rate (Assuncao-Reis z)',
          ramp: 'diverging',
          extent: [-3, 3],
          unit: 'standard deviations',
          labels: ['below expected', 'above expected']
        }
      ];
    }
    return [
      {
        kind: 'ramp',
        id: 'rate',
        title: MAP_LABELS[state.map],
        ramp: state.ramp,
        extent: 'gpu',
        unit: 'deaths per 100,000 per year',
        format: value => value.toFixed(1)
      }
    ];
  },

  snippet: state => `import {
  addRateClusterMapRecipe,
  GPUSpatialEmpiricalBayesRates
} from '@luma.gl/experimental/gpu-spatial-analysis';

// deaths and personYears are float32 views; vertices, ringOffsets and
// featureRingOffsets are the county polygons in GeoArrow layout.
const rates = addRateClusterMapRecipe(graph, {
  events: deaths, populations: personYears,
  positions: vertices, ringOffsets, polygonOffsets: featureRingOffsets,
  criterion: '${state.criterion}',            // GPUContiguityWeights
  analyze: '${state.analyze}',                // GPUEmpiricalBayesRates output
  neighborCapacity, parameters, palette,
${
  state.gating === 'permutation'
    ? `  permutation: {                              // GPULocalPermutationTest
    parameters: permutationParameters,        // seed ${state.seed}, ${state.permutations} permutations
    maximumPermutations: 999,
    alternative: '${state.alternative}',
    falseDiscoveryRate: ${state.falseDiscoveryRate}
  },
`
    : `  // no permutation: GPULocalMoran gates analytically\n`
}  standardizedRates, smoothedRates, rawRates, summary
});

// Spatial EB reuses the recipe's weights.
graph.add(new GPUSpatialEmpiricalBayesRates({
  events: deaths, populations: personYears,
  weights: rates.weights, smoothedRates: spatialRates
}));

// rates.colors is a packed rgba8 column the layer reads directly.`,

  about: {
    what: '`GPUEmpiricalBayesRates` computes the raw rate, the global empirical Bayes smoothed rate and the Assuncao-Reis standardized rate of every county from deaths and population at risk. `GPUSpatialEmpiricalBayesRates` pools each county with its neighbours instead. `addRateClusterMapRecipe` chains the rates into queen contiguity, local Moran and a permutation test.',
    why: 'A county with 64 residents and one fatal crash has a rate of 6,000 per 100,000. That is not danger, it is arithmetic. Smoothing and standardising separate real signal from small-number noise before anyone ranks or maps the rates.',
    howToRead:
      'Compare the raw and smoothed maps at the same color range: speckle in the thinly populated West disappears and the broad Southern and Mountain belts remain. On the cluster map, red counties are high-rate clusters (high-high), blue are low-rate clusters. Matches PySAL esda `Empirical_Bayes`, `Spatial_Empirical_Bayes`, `assuncao_rate` and `Moran_Local_Rate`.'
  },

  create: async ctx => (await import('./rate-smoothing.compute')).createRateSmoothing(ctx),

  story: [
    {
      id: 'question',
      title: 'Where is it really most dangerous to drive?',
      body: "Between 2017 and 2023, 275,368 people died in motor-vehicle crashes in the United States. Dividing each county's deaths by its residents seems to answer the question *where is driving most dangerous?*\n\nThis map starts with exactly that raw rate (**Map shows** below). Notice how the loudest colors sit on **tiny rural counties**: that is the problem this scene solves.",
      options: {map: 'raw'},
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['map']
    },
    {
      id: 'small-numbers',
      title: 'Small numbers make wild rates',
      body: 'Loving County, Texas has about 64 residents. One fatal crash on a pipeline road gives it a raw rate near 6,000 deaths per 100,000 people per year, five hundred times the national rate. Counties under 5,000 residents range from 7 to 174 deaths per 100k (5th to 95th percentile), while counties over 100,000 stay between 5 and 21.\n\nThe map clamps colors at the central 96% of the smoothed rates, so the outliers saturate. **Click any county** to see its numbers.',
      options: {map: 'raw'},
      controls: ['map'],
      readouts: ['smallCounties', 'selected'],
      camera: {longitude: -103.3, latitude: 31.9, zoom: 5.2},
      callout: {coordinate: [-103.58, 31.85], text: 'Loving County, TX (about 64 residents)'}
    },
    {
      id: 'empirical-bayes',
      title: 'Empirical Bayes: shrink toward what the population supports',
      body: '`GPUEmpiricalBayesRates` estimates how much true county rates really vary (the prior variance *a*) and shrinks each raw rate *y* toward the pooled national rate *m*: `r = w y + (1 - w) m` with `w = a / (a + m / b)` and *b* the person-years at risk. Big counties keep their own rate (*w* near 1); tiny counties collapse to the pooled rate.\n\nSame colors, same range: the loudest speckle of the thinly populated West is tamed, while the broad Southern and Mountain belts survive because they rest on many deaths. The readout *Counties under 5,000 residents* shows the small-county spread shrinking from 7-174 to about 9-67. This matches esda `Empirical_Bayes`.',
      options: {map: 'smoothed'},
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['map'],
      readouts: ['smallCounties', 'shrinkageWeight'],
      highlight: {readout: 'smallCounties'}
    },
    {
      id: 'spatial-eb',
      title: 'Borrow from neighbours instead of the nation',
      body: "`GPUSpatialEmpiricalBayesRates` uses each county plus its queen neighbours as the reference population: the prior mean is the neighbourhood's pooled rate. A sparse county in the Navajo Nation is pulled toward its neighbours, not toward Ohio.\n\nThe neighbours come from `GPUContiguityWeights`, built from shared boundary vertices. Switch **Neighbour definition** to rook (shared edge) to see how little the answer depends on it; this is a compile-time option, so the panel marks it as a rebuild.",
      options: {map: 'spatial'},
      controls: ['map', 'criterion'],
      readouts: ['neighbors'],
      camera: {longitude: -108, latitude: 35.8, zoom: 4.6},
      callout: {coordinate: [-110.32, 35.4], text: 'Navajo County, AZ'}
    },
    {
      id: 'standardized',
      title: 'Standardise: how surprising is this rate?',
      body: "The **Assuncao-Reis standardized rate** divides each county's excess over the pooled rate by its own sampling uncertainty, `z = (y - m) / sqrt(a + m / b)`. A tiny county needs a huge excess to be surprising; a big county needs only a small one. Red is more deaths than chance explains, blue fewer.\n\nThis is the input esda feeds into `Moran_Local_Rate`.",
      options: {map: 'standardized'},
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['map']
    },
    {
      id: 'cluster-map',
      title: 'Cluster map: where is danger concentrated?',
      body: "`GPULocalMoran` compares each standardized rate with its neighbours' and `GPULocalPermutationTest` shuffles the values 499 times to decide which of the 3,109 counties are significant clusters. Red counties are high-rate clusters (high-high), blue are low-rate clusters; gray is noise.\n\nTry **Permutation tail**, **Significance level** and **Benjamini-Hochberg FDR**. The *Moran's I* readout shows how much more spatially structured the standardized rate is than the raw one.",
      options: {map: 'cluster', gating: 'permutation', permutations: 499},
      controls: ['alternative', 'significance', 'falseDiscoveryRate'],
      readouts: ['clusters', 'moran'],
      highlight: {readout: 'clusters'}
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: 'Deaths are counted where the crash happened, population is residents, so tourist and through-traffic counties look worse than they are. Counts are 2017-2023 totals; empirical Bayes assumes one shared prior, which is rough for a country mixing urban and rural driving. FDR is stricter than the unadjusted test.\n\nTry: turn on **Benjamini-Hochberg FDR**; set **Permutation tail** to *Lesser* to find only low-rate clusters; switch **Rate analysed by local Moran** to the smoothed rate; click Robeson County, NC (a high rate that survives smoothing because 130,000 people live there).',
      options: {map: 'cluster', falseDiscoveryRate: true},
      controls: ['falseDiscoveryRate', 'alternative', 'analyze'],
      readouts: ['clusters', 'selected'],
      callout: {coordinate: [-79.1, 34.64], text: 'Robeson County, NC: rate survives smoothing'}
    }
  ]
});
