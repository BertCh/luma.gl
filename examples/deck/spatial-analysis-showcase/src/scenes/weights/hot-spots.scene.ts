// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {RAMP_NAMES} from '../../engine/ramps';
import {GI_COLORS, MORAN_COLORS, NEUTRAL} from './b4-colors';
import {getVariableInfo, VARIABLES} from './b4-geography';
import type {HotSpotsOptions} from './hot-spots.compute';

const CATEGORY_NAMES = [
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
];

const ALTERNATIVES = [
  {
    value: 'directed',
    label: 'Directed (esda default)',
    help: 'The tail on the side of the observed statistic.'
  },
  {value: 'two-sided', label: 'Two-sided', help: 'Doubles the smaller tail.'},
  {value: 'greater', label: 'Greater', help: 'Tests for an unusually large statistic: hot spots.'},
  {value: 'lesser', label: 'Less', help: 'Tests for an unusually small statistic: cold spots.'},
  {
    value: 'folded',
    label: 'Folded',
    help: 'Compares |simulated - mean| with |observed - mean|; symmetric.'
  }
] as const;

const isCells = (state: HotSpotsOptions) => state.source === 'nature-cells';

function getLegends(state: HotSpotsOptions): LegendSpec[] {
  const variable = getVariableInfo(state.variable);
  if (state.display === 'values') {
    return [
      {
        kind: 'ramp',
        id: 'display',
        title: isCells(state) ? 'Observations per cell' : variable.label,
        ramp: state.ramp,
        extent: 'gpu',
        unit: isCells(state) ? 'observations' : variable.unit,
        sqrtScale: isCells(state),
        format: value =>
          isCells(state) ? Math.round(value).toString() : value.toFixed(variable.digits)
      }
    ];
  }
  if (state.display === 'zscore') {
    return [
      {
        kind: 'ramp',
        title: state.statistic === 'gi-star' ? 'Gi* z-score' : 'Local Moran z-score',
        ramp: 'diverging',
        extent: [-5, 5],
        labels: ['cold / dissimilar', 'hot / similar'],
        unit: 'standard deviations'
      }
    ];
  }
  const neutralEntry = {color: NEUTRAL, label: 'Not significant'};
  if (state.statistic === 'gi-star') {
    return [
      {
        kind: 'categories',
        title: 'Gi* confidence (hot to cold)',
        entries: [
          {color: GI_COLORS[3], label: 'Hot spot, 99%'},
          {color: GI_COLORS[2], label: 'Hot spot, 95%'},
          {color: GI_COLORS[1], label: 'Hot spot, 90%'},
          neutralEntry,
          {color: GI_COLORS[7], label: 'Cold spot, 90%'},
          {color: GI_COLORS[6], label: 'Cold spot, 95%'},
          {color: GI_COLORS[5], label: 'Cold spot, 99%'}
        ],
        note:
          (state.inference === 'permutation'
            ? 'Bins are kept only where the conditional permutation test confirms the cell. '
            : '') +
          (state.falseDiscoveryRate
            ? 'Bins use Benjamini-Hochberg corrected p-values.'
            : 'Two-sided confidence of the Gi* z-score.')
      }
    ];
  }
  return [
    {
      kind: 'categories',
      title: 'Local Moran quadrant',
      entries: [
        {color: MORAN_COLORS[1], label: 'High-High cluster'},
        {color: MORAN_COLORS[3], label: 'Low-Low cluster'},
        {color: MORAN_COLORS[4], label: 'High-Low outlier'},
        {color: MORAN_COLORS[2], label: 'Low-High outlier'},
        neutralEntry
      ],
      note:
        state.inference === 'analytic'
          ? 'Quadrants shown where the analytic p-value is within the significance level.'
          : state.inference === 'none'
            ? 'Ungated: every place gets its quadrant.'
            : 'Quadrants gated by a conditional permutation test.'
    }
  ];
}

function getSnippet(state: HotSpotsOptions): string {
  const isGi = state.statistic === 'gi-star';
  const permutation =
    state.inference === 'permutation'
      ? `
graph.add(new GPULocalPermutationTest({
  weights, values, mask,
  statistic: '${isGi ? (state.selfWeight === 'include' ? 'localGStar' : 'localG') : 'localMoran'}', alternative: '${state.alternative}',
  parameters: permutationParameters,       // seed, permutations, significance: buffer writes
  maximumPermutations: 999, maximumNeighbors: ${state.maximumNeighbors},
  falseDiscoveryRate: ${state.falseDiscoveryRate},
  exceedances, pseudoPValues, significant
}));`
      : '';
  if (isCells(state)) {
    return `import {addHotSpotAnalysisRecipe, GPULocalMoran, getGPUNeighborSearchParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

// One call adds the whole chain: points -> ${state.family} cells -> counts -> neighbour search -> Gi* ${state.inference === 'permutation' ? '-> permutation test' : ''}
const hotSpots = addHotSpotAnalysisRecipe(graph, {
  source: {
    kind: 'points', positions, mask,           // lon/lat degrees; the mask is a category and hour filter
    family: '${state.family}', resolution: ${state.family === 'quadbin' ? state.resolution : state.h3Resolution},
    tableCapacity: 16384, neighborCapacity: 16384 * 40, gridSize: [128, 128],
    neighborSearchParameters               // radius ${state.radiusCells} cells: a buffer write
  },
  parameters,                               // significance level
  selfWeight: ${state.selfWeight === 'include' ? 1 : 0},                         // 1 is Gi*, 0 is Gi
  falseDiscoveryRate: ${state.falseDiscoveryRate},
  zScores, bins, pValues, weights${
    state.inference === 'permutation' && isGi
      ? `,
  permutation: {parameters: permutationParameters, maximumPermutations: 999,
    alternative: '${state.alternative}', maximumNeighbors: ${state.maximumNeighbors}, significant}`
      : ''
  }
});
${
  isGi
    ? ''
    : `
// Local Moran on the recipe's cell weights.
graph.add(new GPULocalMoran({
  weights: hotSpots.weights, values, mask, parameters, zScores, quadrants,
  quadrantGating: '${state.inference === 'analytic' ? 'analytic' : 'none'}'
}));${permutation}`
}`;
  }
  return `import {GPUHotSpotAnalysis, GPULocalMoran, GPULocalPermutationTest} from '@luma.gl/experimental/gpu-spatial-analysis';

// weights: queen contiguity (GPUContiguityWeights) or a neighbour search, ${state.weightTransform === 'row' ? 'row-standardised' : 'binary'}.
${
  isGi
    ? `graph.add(new GPUHotSpotAnalysis({
  weights, values, mask, parameters,
  selfWeight: ${state.selfWeight === 'include' ? 1 : 0},                         // 1 is Gi*, 0 is Gi
  falseDiscoveryRate: ${state.falseDiscoveryRate},
  zScores, bins, pValues
}));`
    : `graph.add(new GPULocalMoran({
  weights, values, mask, parameters, zScores, pValues, quadrants,
  quadrantGating: '${state.inference === 'analytic' ? 'analytic' : 'none'}'${
    state.inference === 'analytic'
      ? `,
  falseDiscoveryRate: ${state.falseDiscoveryRate}`
      : ''
  }
}));`
}${permutation}`;
}

/**
 * Hot spots: Gi* and local Moran of nature observations aggregated to cells and of county or tract health
 * measures. GPU work is in `hot-spots.compute.ts`.
 */
export default defineScene<HotSpotsOptions>({
  id: 'hot-spots',
  title: 'Hot spots beyond chance',
  chapter: 'weights',
  order: 3,
  summary:
    'Where do wildlife sightings in Chicago and diabetes across US counties cluster beyond what chance would produce? Getis-Ord Gi* and local Moran, with analytic or conditional permutation significance and false-discovery control.',
  contributors: [
    'GPUHotSpotAnalysis',
    'GPULocalMoran',
    'GPULocalPermutationTest',
    'addHotSpotAnalysisRecipe',
    'GPUNeighborSearch',
    'GPUContiguityWeights'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations aggregated into cells'},
    {id: 'us-counties', role: 'counties with CDC PLACES health measures'},
    {
      id: 'chicago-tracts',
      role: 'tracts (outlines of the cell map, and an alternate polygon source)'
    },
    {id: 'chicago-community-areas', role: 'names of the community areas'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.7},

  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Analyse',
      group: 'Data',
      apply: 'compile',
      default: 'nature-cells',
      help: 'Chicago nature observations aggregated to cells (a point pattern), or health measures of areas with a spatial-weights matrix.',
      options: [
        {
          value: 'nature-cells',
          label: 'Chicago nature observations in cells',
          help: '43,557 observations of 2023 binned into Quadbin or H3 cells by addHotSpotAnalysisRecipe.'
        },
        {value: 'us-counties', label: 'US counties (3,109)'},
        {value: 'chicago-tracts', label: 'Chicago tracts (791)'}
      ]
    },
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Observations',
      apply: 'param',
      default: 'all',
      disabledWhen: state => !isCells(state),
      help: 'Which observations are binned. A per-point mask: a buffer write.',
      options: [
        {value: 'all', label: 'All observations'},
        ...CATEGORY_NAMES.map((label, index) => ({value: String(index), label}))
      ]
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hours of the day',
      group: 'Observations',
      apply: 'param',
      min: 0,
      max: 24,
      step: 1,
      default: [0, 24],
      unit: 'h',
      disabledWhen: state => !isCells(state),
      help: 'Only observations from this local clock window are counted (0 to 24 is the whole day). Try 5 to 9 for the early-morning birders.'
    },
    {
      kind: 'select',
      id: 'family',
      label: 'Cell grid',
      group: 'Observations',
      apply: 'compile',
      default: 'quadbin',
      disabledWhen: state => !isCells(state),
      help: 'Square Quadbin tiles or H3 hexagons. Compile-time: the aggregation is specialised per family and resolution.',
      options: [
        {value: 'quadbin', label: 'Quadbin squares'},
        {value: 'h3', label: 'H3 hexagons'}
      ]
    },
    {
      kind: 'slider',
      id: 'resolution',
      label: 'Quadbin level',
      group: 'Observations',
      apply: 'compile',
      min: 12,
      max: 17,
      step: 1,
      default: 15,
      disabledWhen: state => !isCells(state) || state.family !== 'quadbin',
      format: value => `level ${value}`,
      help: 'Zoom level of the square tiles: 15 is about 0.9 km, 16 about 0.45 km, 17 about 0.2 km at this latitude. Compile-time; the table holds up to 16,384 occupied cells.'
    },
    {
      kind: 'slider',
      id: 'h3Resolution',
      label: 'H3 resolution',
      group: 'Observations',
      apply: 'compile',
      min: 5,
      max: 9,
      step: 1,
      default: 8,
      disabledWhen: state => !isCells(state) || state.family !== 'h3',
      format: value => `resolution ${value}`,
      help: 'H3 hexagon size: 8 is about 0.8 km across (edge 0.46 km), 9 about 0.3 km. Compile-time.'
    },
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable',
      group: 'Areas',
      apply: 'param',
      default: 'diabetes',
      disabledWhen: state => isCells(state),
      help: 'The health or social measure analysed.',
      options: VARIABLES.map(variable => ({
        value: variable.id,
        label: variable.label,
        help: variable.help
      }))
    },
    {
      kind: 'select',
      id: 'weights',
      label: 'Neighbours are...',
      group: 'Areas',
      apply: 'compile',
      default: 'queen',
      disabledWhen: state => isCells(state),
      help: 'The weights matrix of the areas (see "Who is my neighbour?").',
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
      group: 'Areas',
      apply: 'compile',
      min: 2,
      max: 16,
      step: 1,
      default: 6,
      disabledWhen: state => isCells(state) || state.weights !== 'knn',
      help: 'k of the nearest-neighbour rule. Compile-time.'
    },
    {
      kind: 'slider',
      id: 'bandFactor',
      label: 'Distance band',
      group: 'Areas',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 1.5,
      disabledWhen: state => isCells(state) || state.weights !== 'band',
      format: value => `${value} x spacing`,
      help: 'Radius in multiples of the median nearest-centroid distance.'
    },
    {
      kind: 'slider',
      id: 'radiusCells',
      label: 'Neighbourhood radius',
      group: 'Neighbours',
      apply: 'param',
      min: 1,
      max: 3.5,
      step: 0.25,
      default: 1.5,
      disabledWhen: state => !isCells(state),
      format: value => `${value} cells`,
      help: 'Cells whose centres lie within this many cell widths are neighbours (the search runs on the cell centres in degrees, so the band is about 15% longer north-south than east-west at this latitude). A parameter write.'
    },
    {
      kind: 'select',
      id: 'weightTransform',
      label: 'Weights',
      group: 'Neighbours',
      apply: 'compile',
      default: 'binary',
      help: 'Binary weights give the classic Gi* of ArcGIS. Row standardisation is the esda default for local Moran; with it, choose Gi (exclude the place itself).',
      options: [
        {value: 'binary', label: 'Binary'},
        {value: 'row', label: 'Row standardised'}
      ]
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic',
      group: 'Statistic',
      apply: 'compile',
      default: 'gi-star',
      help: 'Gi* finds concentrations of high or low values; local Moran finds places similar to (or unlike) their neighbours. Compile-time: each statistic is its own graph.',
      options: [
        {
          value: 'gi-star',
          label: 'Getis-Ord Gi* hot and cold spots',
          help: 'Is the sum of values in the neighbourhood unusually high or low?'
        },
        {
          value: 'local-moran',
          label: 'Local Moran clusters and outliers',
          help: 'Is a place similar to its neighbours (HH, LL) or an outlier (HL, LH)?'
        }
      ]
    },
    {
      kind: 'select',
      id: 'selfWeight',
      label: 'Gi* or Gi',
      group: 'Statistic',
      apply: 'compile',
      default: 'include',
      disabledWhen: state => state.statistic !== 'gi-star',
      help: 'Gi* includes the place itself in its neighbourhood (self weight 1, the ArcGIS Hot Spot Analysis); Gi leaves it out. Compile-time.',
      options: [
        {value: 'include', label: 'Gi*: include the place itself'},
        {value: 'exclude', label: 'Gi: exclude the place itself'}
      ]
    },
    {
      kind: 'select',
      id: 'inference',
      label: 'Significance from',
      group: 'Significance',
      apply: 'compile',
      default: 'analytic',
      help: 'How a result is kept: the analytic normal p-value, a conditional permutation test, or (local Moran only) nothing. Compile-time.',
      options: [
        {value: 'analytic', label: 'Analytic p-value (normal approximation)'},
        {
          value: 'permutation',
          label: 'Conditional permutation test',
          help: 'GPULocalPermutationTest shuffles the other values around each place (esda p_sim).'
        },
        {value: 'none', label: 'None: every quadrant (local Moran only)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'falseDiscoveryRate',
      label: 'Benjamini-Hochberg FDR correction',
      group: 'Significance',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.statistic === 'local-moran' && state.inference === 'none',
      help: 'Controls the share of false discoveries when thousands of tests run at once: fewer, more trustworthy spots. Compile-time (it adds a sort).'
    },
    {
      kind: 'slider',
      id: 'significance',
      label: 'Significance level',
      group: 'Significance',
      apply: 'param',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      default: 0.05,
      format: value => `p <= ${value.toFixed(3)}`,
      disabledWhen: state => state.statistic === 'gi-star' && state.inference !== 'permutation',
      help: 'Local Moran: places above this p-value are not significant. With the permutation test it is the level of the confirmation (Gi* bins always use 90, 95 and 99 percent).'
    },
    {
      kind: 'select',
      id: 'alternative',
      label: 'Permutation tail',
      group: 'Significance',
      apply: 'compile',
      default: 'directed',
      disabledWhen: state => state.inference !== 'permutation',
      help: 'Which tail of the simulated distribution counts as evidence. Compile-time.',
      options: ALTERNATIVES
    },
    {
      kind: 'select',
      id: 'maximumNeighbors',
      label: 'Neighbour limit of the permutation test',
      group: 'Significance',
      apply: 'compile',
      default: '32',
      disabledWhen: state => state.inference !== 'permutation',
      help: 'Rows with more neighbours than this are not tested. Larger limits cost memory per row. Compile-time.',
      options: [
        {value: '16', label: '16'},
        {value: '32', label: '32'},
        {value: '64', label: '64'}
      ]
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Permutations',
      group: 'Significance',
      apply: 'param',
      min: 49,
      max: 999,
      step: 50,
      default: 199,
      disabledWhen: state => state.inference !== 'permutation',
      help: 'More permutations give finer pseudo p-values at proportional GPU cost. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Permutation seed',
      group: 'Significance',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      disabledWhen: state => state.inference !== 'permutation',
      help: 'The same seed always gives the same pseudo p-values.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'classes',
      help: 'The significant classes, the raw z-scores, or the analysed values.',
      options: [
        {value: 'classes', label: 'Significant hot spots / clusters'},
        {value: 'zscore', label: 'z-score of the statistic'},
        {value: 'values', label: 'The analysed values'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp (values)',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      disabledWhen: state => state.display !== 'values',
      options: RAMP_NAMES.filter(name => name !== 'diverging').map(name => ({
        value: name,
        label: name.charAt(0).toUpperCase() + name.slice(1)
      }))
    },
    {
      kind: 'toggle',
      id: 'showNotSignificant',
      label: 'Show not-significant places',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.display !== 'classes',
      help: 'Draws the gray places that are indistinguishable from chance. Turn off to see only the significant ones.'
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
    {
      id: 'hot',
      label: 'Hot 99 / 95 / 90% (or HH)',
      help: 'Gi*: places per confidence bin. Local Moran: the High-High quadrant.'
    },
    {id: 'cold', label: 'Cold 99 / 95 / 90% (or LL)'},
    {id: 'outliers', label: 'Outliers LH / HL'},
    {id: 'notSignificant', label: 'Not significant'},
    {id: 'permutation', label: 'Permutation test'},
    {id: 'moments', label: 'Mean / std. deviation', help: 'Of the analysed values.'},
    {id: 'scale', label: 'Scale'},
    {id: 'capacity', label: 'Capacity'}
  ],

  legends: getLegends,

  snippet: getSnippet,

  about: {
    what: "`GPUHotSpotAnalysis` computes the Getis-Ord Gi* z-score of every place: is the sum of the values in its neighbourhood higher or lower than random placement would give? `GPULocalMoran` computes local Moran's I and its quadrant (cluster or outlier). `GPULocalPermutationTest` replaces the normal approximation by a conditional permutation, and `addHotSpotAnalysisRecipe` chains point binning, neighbour search, Gi* and the permutation test in one call.",
    why: 'Mapping raw counts or rates shows where people are, not where something is unusual. A local test says whether a concentration exceeds what chance could produce, and false-discovery control keeps thousands of simultaneous tests honest.',
    howToRead:
      'Red places are hot spots (high values with high neighbours), blue are cold spots, gray places are not significant. For local Moran, **High-High** and **Low-Low** are clusters of similar values; **High-Low** and **Low-High** are outliers. Darker means higher confidence.'
  },

  create: async ctx => (await import('./hot-spots.compute')).createHotSpots(ctx),

  story: [
    {
      id: 'counts',
      title: 'Where in Chicago do people watch nature?',
      body: 'People logged **43,557 wild plants, animals and fungi** in Chicago on iNaturalist in 2023. This map bins them into about 0.9 km Quadbin cells and colours each cell by its count. It shows where people go to look: the lakefront at Montrose Point, Lincoln Park and the North Side parks and river corridors.\n\nThat is not yet an answer to *where sightings concentrate beyond what the surroundings explain*. **`addHotSpotAnalysisRecipe`** adds the whole chain in one call: points to cells, cell counts, a neighbour search on the cell centres, Gi*, an optional permutation test. The next step turns on its significance classes. **Group** and **Hours of the day** below change what is counted.',
      options: {source: 'nature-cells', category: 'all', display: 'values', ramp: 'inferno'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.7, transitionMs: 1400},
      callout: {coordinate: [-87.6325, 41.9625], text: 'Montrose Point'},
      highlight: {readout: 'scale'},
      controls: ['category', 'hours'],
      readouts: ['places', 'scale']
    },
    {
      id: 'gi-star',
      title: 'Getis-Ord Gi*: hot and cold spots',
      body: '**`GPUHotSpotAnalysis`** compares, for every cell, the sum of the counts in its neighbourhood (itself included, the star) with the sum random placement would give: `Gi* = (sum_j w_ij x_j - mean * sum_j w_ij) / (s * sqrt((n * sum_j w_ij^2 - (sum_j w_ij)^2) / (n - 1)))`. A large positive z is a **hot spot**, a large negative one a **cold spot**; the bins are the 90, 95 and 99 percent confidence of the two-sided normal test (esda `G_Local` with `star=True`, ArcGIS Hot Spot Analysis).\n\nRed cells are neighbourhoods with more sightings than their surroundings explain; most of the city is gray. The readouts count the cells per bin from a GPU histogram. **Map shows** flips between these classes, the raw *z-score* behind them and the analysed counts.',
      options: {display: 'classes'},
      highlight: {readout: 'hot'},
      controls: ['display', 'selfWeight', 'weightTransform'],
      readouts: ['hot', 'cold', 'notSignificant']
    },
    {
      id: 'scale',
      title: 'The answer depends on the scale',
      body: 'A hot spot is relative to the cells and the neighbourhood you choose. Set **Quadbin level** to 16 (about 0.45 km cells, already on) or slide **Neighbourhood radius** below: small cells and a small radius find block-scale pockets, large ones merge them into districts.\n\nThe resolution is a compile-time option (the cell aggregation is specialised per resolution), so the first use rebuilds one graph; the radius is a parameter-buffer write that only re-runs the neighbour search. Note that positions are precise, so finer cells work, but counts per cell follow where observers walk: a single trail can light up a cell.',
      options: {resolution: 16, radiusCells: 1.5},
      camera: {longitude: -87.66, latitude: 41.86, zoom: 10.6, transitionMs: 1600},
      highlight: {readout: 'hot'},
      controls: ['resolution', 'radiusCells'],
      readouts: ['hot', 'scale']
    },
    {
      id: 'fdr',
      title: 'Thousands of tests at once',
      body: 'This map tests thousands of cells at the 95 percent level, so some will be flagged by chance. The **Benjamini-Hochberg false discovery rate** correction (`falseDiscoveryRate`, esda `fdr`) raises the bar so that the expected share of false discoveries stays at the stated level: fewer, more trustworthy spots survive, mostly the strongest cores.\n\nIt is a compile-time option of the contributor (it adds a 31-bit sort), so the graph with it was compiled once and the toggle switches to it. Toggle **Benjamini-Hochberg FDR correction** below and compare **Hot 99 / 95 / 90% (or HH)** with and without.',
      options: {falseDiscoveryRate: true},
      highlight: {readout: 'hot'},
      controls: ['falseDiscoveryRate'],
      readouts: ['hot']
    },
    {
      id: 'permutation',
      title: 'Check the normal approximation with a permutation test',
      body: 'The z-score assumes a normal distribution, which is shaky for counts with few neighbours. **`GPULocalPermutationTest`** needs no assumption: around each cell it shuffles the other values 199 times (conditional randomisation, esda `p_sim`) and counts how often chance beats the observed Gi*. Only cells it confirms keep their colour (**Significance from** is now *Conditional permutation test*); **Permutation test** below reports how many.\n\nChange **Permutations**, the **Permutation seed** (results are reproducible: a counter-based Philox generator), **Permutation tail** or **Neighbour limit of the permutation test** (rows with more neighbours are skipped). All of them keep the GPU busy for milliseconds, where the CPU takes minutes for thousands of rows.',
      options: {inference: 'permutation', permutations: 199, seed: 1, resolution: 15},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.7, transitionMs: 1400},
      highlight: {readout: 'permutation'},
      controls: ['inference', 'permutations', 'seed', 'alternative', 'maximumNeighbors'],
      readouts: ['permutation', 'hot']
    },
    {
      id: 'local-moran',
      title: 'Local Moran: clusters and outliers',
      body: 'Gi* finds concentrations of high or low values. **`GPULocalMoran`** asks a different question: does this cell resemble its neighbours? **High-High** and **Low-Low** are clusters of similar counts; **High-Low** and **Low-High** are **outliers**, a busy cell in a quiet area (esda `Moran_Local`).\n\nThe quadrants here are gated by the same conditional permutation test. Set **Significance from** to *Analytic p-value*: that gives the normal approximation, and *None* shows every quadrant, which is how a Moran scatterplot looks before testing.',
      options: {statistic: 'local-moran', inference: 'permutation', falseDiscoveryRate: false},
      highlight: {readout: 'outliers'},
      controls: ['statistic', 'inference'],
      readouts: ['hot', 'cold', 'outliers']
    },
    {
      id: 'counties',
      title: 'The diabetes belt across US counties',
      body: 'The same contributors work on areas. Set **Analyse** to *US counties (3,109)* with queen contiguity (**Neighbours are...**) and *Row standardised* **Weights**: local Moran of age-adjusted diabetes prevalence finds large **High-High** blocks across Texas, the Deep South and Appalachia, and **Low-Low** blocks in the Mountain West and the Upper Midwest, with scattered outliers between them.\n\n**Limits.** A significant cell is a statement about a *pattern*, not a cause; county values are model-based estimates; results move with the weights, the cell size or the aggregation (the modifiable areal unit problem); counts per cell mix wildlife and observer effort (where people walk and upload). **Try it:** set **Statistic** to Gi*, compare row and binary **Weights**, pick a *Distance band* in **Neighbours are...**, or set **Analyse** to *Chicago tracts*.',
      options: {
        source: 'us-counties',
        variable: 'diabetes',
        weightTransform: 'row',
        statistic: 'local-moran',
        inference: 'analytic',
        selfWeight: 'exclude',
        display: 'classes'
      },
      camera: {longitude: -95.5, latitude: 38.2, zoom: 3.9, transitionMs: 1800},
      highlight: {readout: 'hot'},
      controls: ['source', 'variable', 'weights', 'weightTransform', 'statistic'],
      readouts: ['hot', 'cold', 'outliers']
    }
  ]
});
