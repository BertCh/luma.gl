// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CLASSIFICATION_METHOD_INFO} from '../../cartography/breaks';
import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CHICAGO, CITY_FRAMES, labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {EFFORT_CAVEAT} from '../../cartography/hue-registry';
import {formatCount} from '../../cartography/live-text';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import type {ClassTable} from '../../cartography/types';
import {getRampOptions} from '../../engine/ramps';
import {defineScene, type LegendSpec} from '../scene';
import {getVariableInfo, VARIABLES} from './b4-geography';
import type {HotSpotsOptions, HotSpotValueClasses} from './hot-spots.compute';
import {
  getGiTable,
  getMoranQuadrantColors,
  GI_CRITICAL_Z,
  Z_DISPLAY_RANGE
} from './hot-spots.style';

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

/** The cartouche of one step (steps replace the whole `title` object, so the sample repeats). */
const cartouche = (title: string, subtitle: string, national = false) => ({
  title,
  subtitle,
  chips: national ? (['Modelled'] as const) : (['Observer effort'] as const)
});

/** Credits of the Chicago steps and of the national step. */
const CHICAGO_CREDIT = joinCredits(
  CREDITS.iNaturalist,
  CREDITS.cityOfChicago,
  CREDITS.usCensus,
  CREDITS.openStreetMap
);
const COUNTY_CREDIT = joinCredits('CDC PLACES (public domain)', CREDITS.usCensus);

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

function getLegends(state: HotSpotsOptions, data: Readonly<Record<string, unknown>>): LegendSpec[] {
  const variable = getVariableInfo(state.variable);
  const counts = data['classCounts'] as number[] | undefined;
  const noDataLabel = isCells(state) ? 'No observations' : 'No data';
  if (state.display === 'values') {
    const title = isCells(state) ? 'Records per cell' : variable.label;
    const unit = isCells(state) ? 'iNaturalist records' : variable.unit;
    const classes = data['valueClasses'] as HotSpotValueClasses | null | undefined;
    if (state.classification !== 'continuous' && classes) {
      const alternate = classes.alternate
        ? ` Left of the divider: ${classes.alternate.table.method}.`
        : '';
      return [
        getClassTableLegend(classes.table, {
          title,
          id: 'value-classes',
          basis: isCells(state) ? (data['basis'] as string | undefined) : undefined,
          counts: classes.counts,
          interactive: true,
          layout: 'list',
          // Rule 9: the legend stays on one table so a colour change is a data change.
          note: `${classes.table.method}.${alternate}`
        })
      ];
    }
    return [
      {
        kind: 'ramp',
        id: 'display',
        title,
        ramp: state.ramp,
        extent: 'gpu',
        unit,
        sqrtScale: isCells(state),
        format: isCells(state) ? formatCount : (value: number) => value.toFixed(variable.digits)
      }
    ];
  }
  if (state.display === 'zscore') {
    return [
      {
        kind: 'ramp',
        title: state.statistic === 'gi-star' ? 'Gi* z-score' : 'Local Moran z-score',
        ramp: 'rdbu',
        extent: Z_DISPLAY_RANGE,
        midpoint: 0,
        midpointLabel: '0 = random',
        ticks: [-GI_CRITICAL_Z[1], GI_CRITICAL_Z[1]],
        unit: 'standard deviations',
        format: value => value.toFixed(2)
      }
    ];
  }
  if (state.statistic === 'gi-star') {
    const table = (data['giTable'] as ClassTable | undefined) ?? getGiTable('light', noDataLabel);
    return [
      getClassTableLegend(table, {
        title: 'Gi* hot and cold spots',
        id: 'gi-classes',
        counts,
        interactive: true,
        layout: 'list',
        note:
          (state.inference === 'permutation'
            ? 'Kept only where the conditional permutation test confirms the cell. '
            : '') +
          (state.falseDiscoveryRate
            ? 'Benjamini-Hochberg corrected p-values.'
            : 'Two-sided normal test of the z-score: |z| above 1.65, 1.96, 2.58.')
      })
    ];
  }
  const table = (data['giTable'] as ClassTable | undefined) ?? getGiTable('light', noDataLabel);
  const colors = getMoranQuadrantColors(table);
  const moranCount = (code: number) => counts?.[code];
  return [
    {
      kind: 'categories',
      id: 'moran-classes',
      title: 'Local Moran quadrant',
      layout: 'list',
      interactive: true,
      entries: [
        {color: colors.highHigh, label: 'High-High cluster', count: moranCount(1)},
        {color: colors.lowLow, label: 'Low-Low cluster', count: moranCount(3)},
        {color: colors.highLow, label: 'High-Low outlier', count: moranCount(4)},
        {color: colors.lowHigh, label: 'Low-High outlier', count: moranCount(2)},
        {color: colors.notSignificant, label: 'Not significant', count: moranCount(0)}
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
    {id: 'chicago-community-areas', role: 'names of the community areas'},
    {id: 'us-states', role: 'state lines over the counties'}
  ],
  initialView: {...CITY_FRAMES.chicago},

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
      id: 'classification',
      label: 'Classification',
      group: 'Display',
      apply: 'param',
      default: 'natural-breaks',
      disabledWhen: state => state.display !== 'values',
      help: 'How the analysed values become colour classes (computed on the CPU from the values read back). The same data looks very different under each method; the legend reports the goodness of variance fit.',
      options: [
        ...(['natural-breaks', 'quantile', 'equal-interval', 'head-tail'] as const).map(method => ({
          value: method,
          label: CLASSIFICATION_METHOD_INFO[method].label,
          help: CLASSIFICATION_METHOD_INFO[method].help
        })),
        {value: 'continuous', label: 'Unclassed (continuous ramp)'}
      ]
    },
    {
      kind: 'slider',
      id: 'classCount',
      label: 'Classes',
      group: 'Display',
      apply: 'param',
      min: 3,
      max: 7,
      step: 1,
      default: 5,
      disabledWhen: state => state.display !== 'values' || state.classification === 'continuous',
      help: 'Number of colour classes. Five to seven is what readers can tell apart on a map.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp (values)',
      group: 'Display',
      apply: 'param',
      default: 'bupu',
      disabledWhen: state => state.display !== 'values' || state.classification !== 'continuous',
      help: 'For the unclassed map only: classed maps use the exact ColorBrewer tables of the hue registry. Light is little, dark is much; kept away from red and blue, which this map reserves for hot and cold spots, and from green, the park fill.',
      options: getRampOptions(['bupu', 'ylgnbu', 'purples', 'batlow'])
    },
    {
      kind: 'toggle',
      id: 'compareBreaks',
      label: 'Compare equal interval and natural breaks',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.display !== 'values' || state.classification === 'continuous',
      help: 'Draws the same counts twice with a swipe divider: equal-interval classes on the left, natural breaks on the right. The legend stays on natural breaks.'
    },
    {
      kind: 'toggle',
      id: 'labelHotSpots',
      label: 'Label the strongest hot spots',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.display === 'values',
      help: 'Names the three strongest hot spots (highest z among the significant places), one per cluster, read back from the GPU.'
    },
    {
      kind: 'toggle',
      id: 'showBand',
      label: 'Show the neighbourhood',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => !isCells(state) || state.display === 'values',
      help: 'Draws the neighbourhood radius as a ring around the strongest hot spot: every cell centre inside it is a neighbour.'
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
    {
      id: 'places',
      label: 'Places analysed',
      help: 'Cells with at least one record, or areas with data.'
    },
    {id: 'cellSize', label: 'Cell size', help: 'Edge of one cell.'},
    {
      id: 'hotTotal',
      label: 'Hot spots',
      help: 'Gi*: places in the 90, 95 and 99 percent hot classes. Local Moran: High-High places.'
    },
    {id: 'coldTotal', label: 'Cold spots', help: 'Gi*: cold classes. Local Moran: Low-Low places.'},
    {
      id: 'notSignificantShare',
      label: 'Not significant',
      help: 'Share of places chance could explain.'
    },
    {
      id: 'confirmed',
      label: 'Confirmed by permutation',
      help: 'Places the conditional permutation test keeps.'
    },
    {id: 'gvfEqual', label: 'Fit of equal intervals', help: 'Goodness of variance fit, 0 to 1.'},
    {id: 'gvfNatural', label: 'Fit of natural breaks', help: 'Goodness of variance fit, 0 to 1.'},
    {id: 'outliers', label: 'Outliers LH / HL'},
    {
      id: 'zHistogram',
      label: 'Distribution of z-scores',
      kind: 'chart',
      help: 'Places by z-score; the guides are the 90, 95 and 99 percent critical values.'
    },
    {
      id: 'resultBalance',
      label: 'Results by test outcome',
      kind: 'chart',
      help: 'Tested places split into hot, cold and not significant results, or local Moran cluster types.'
    },
    {
      id: 'hot',
      label: 'Hot 99 / 95 / 90% (or HH)',
      hood: true,
      help: 'Gi*: places per confidence bin. Local Moran: the High-High quadrant.'
    },
    {id: 'cold', label: 'Cold 99 / 95 / 90% (or LL)', hood: true},
    {id: 'notSignificant', label: 'Not significant (count)', hood: true},
    {id: 'permutation', label: 'Permutation test', hood: true},
    {id: 'moments', label: 'Mean / std. deviation', hood: true, help: 'Of the analysed values.'},
    {id: 'scale', label: 'Scale', hood: true},
    {id: 'capacity', label: 'Capacity', hood: true}
  ],

  pipeline: [
    {id: 'weights', label: 'Weights', detail: 'Who is whose neighbour: a radius or contiguity'},
    {id: 'sums', label: 'Local sums', detail: 'The sum of the values in each neighbourhood'},
    {
      id: 'z',
      label: 'z-scores',
      detail: 'How far each sum is from what random placement gives',
      show: {option: 'display', value: 'zscore'}
    },
    {
      id: 'test',
      label: 'Test and bins',
      detail: 'Normal or permutation p-values, 90, 95, 99 percent'
    },
    {
      id: 'classes',
      label: 'Classes',
      detail: 'Hot, cold and not significant',
      show: {option: 'display', value: 'classes'}
    }
  ],

  legends: getLegends,

  basemap: ground('paperCity'),
  furniture: {
    title: cartouche('Where do observers cluster?', 'Hot and cold spots of records'),
    scaleBar: {units: 'metric'},
    credit: CHICAGO_CREDIT,
    // Rule 15: hot spots of counts measure observer effort.
    caveat: EFFORT_CAVEAT
  },
  annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop'], {loop: {minZoom: 10.4}}),

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
      title: 'Where Chicago looks at nature',
      headline: 'Counts show where people look, not wildlife',
      textAlternative:
        'Map of Chicago with square cells shaded in five purple classes by record count: the lakefront and the large parks are darkest.',
      body: '**{{places}}** cells hold at least one iNaturalist record, each **{{cellSize}}** across, coloured in five classes. The darkest cells are the lakefront and the large parks.\n\nA count measures observer effort, not nature: more people looking means more records. **Category** and **Hours of the day** change what is counted.\n\n*Counts follow people.*',
      optionsMode: 'fresh',
      options: {
        source: 'nature-cells',
        display: 'values',
        classification: 'natural-breaks',
        classCount: 5
      },
      controls: ['category', 'hours'],
      readouts: ['places', 'cellSize'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {title: cartouche('Where do observers go?', 'Records per cell, five classes')},
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park', 'jackson-park']),
      stage: 'sums'
    },
    {
      id: 'classes-are-a-choice',
      title: 'Classification is a choice',
      headline: 'The same counts make two different maps',
      textAlternative:
        'The same cell map split by a divider: equal-interval classes on the left leave most cells in the palest class; natural breaks on the right separate them.',
      body: 'Drag the divider. On the left, **equal intervals** leave most cells in the lowest class (fit **{{gvfEqual}}**). On the right, **natural breaks** put the breaks at the gaps in the data (fit **{{gvfNatural}}**).\n\nThe legend stays on natural breaks, so only the breaks change between the two sides. **Classes** sets how many.\n\n*Classification is an editorial choice.*',
      optionsMode: 'fresh',
      options: {
        source: 'nature-cells',
        display: 'values',
        classification: 'natural-breaks',
        classCount: 5,
        compareBreaks: true
      },
      controls: ['classCount'],
      readouts: ['gvfEqual', 'gvfNatural'],
      compare: {labels: ['Equal interval', 'Natural breaks'], position: 0.5},
      camera: {...CITY_FRAMES.chicago, transitionMs: 1000},
      furniture: {title: cartouche('Same counts, two maps', 'Records per cell')},
      annotations: labelsFor(CHICAGO, ['montrose-point'])
    },
    {
      id: 'gi-star',
      title: 'Getis-Ord Gi*: hot and cold spots',
      headline: 'Hot spots beat what their neighbours predict',
      textAlternative:
        'Map of Chicago cells in seven classes from blue cold spots to red hot spots; most cells are faint and not significant, and red cells cluster on the lakefront.',
      body: "**Gi\\*** compares the sum of counts in each cell's neighbourhood with what random placement would give. **{{hotTotal}}** cells are hot spots and **{{coldTotal}}** cold; **{{notSignificantShare}}** are not significant, drawn as a ghost so the parks show through.\n\nHover a cell for its z-score and p-value, or click a legend class to isolate it. These hot spots still measure observer effort.",
      evidence:
        'The z-score distribution shows the tested signal, while the result balance counts **{{hotTotal}}** hot, **{{coldTotal}}** cold and the remaining non-significant cells.',
      caveat:
        'A hot spot here is a concentration of records, which can be caused by observer access and effort as well as wildlife.',
      optionsMode: 'fresh',
      options: {source: 'nature-cells', display: 'classes', statistic: 'gi-star'},
      controls: ['display', 'selfWeight', 'labelHotSpots'],
      readouts: ['resultBalance', 'zHistogram', 'hotTotal', 'notSignificantShare'],
      stage: 'z',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: cartouche('Which cells are hot?', 'Gi* z-score, 90 / 95 / 99 % confidence')
      }
    },
    {
      id: 'scale',
      title: 'The answer depends on the scale',
      headline: 'Change the neighbourhood, change the answer',
      textAlternative:
        'Closer map of the lakefront with a dashed ring around the strongest hot spot showing the neighbourhood that was tested.',
      body: 'A hot spot is relative to the cells and the neighbourhood you choose. The dashed ring is the neighbourhood of the strongest hot spot; cells are **{{cellSize}}** across and **{{hotTotal}}** are hot. Slide **Neighbourhood radius**: small ones find block-scale pockets, large ones merge them into districts.\n\nCells beside the lake have no neighbours in the water, an edge effect.\n\n*Near things are compared with near things; "near" is your choice.*',
      optionsMode: 'fresh',
      options: {
        source: 'nature-cells',
        display: 'classes',
        statistic: 'gi-star',
        resolution: 16,
        radiusCells: 1.5,
        showBand: true
      },
      controls: ['resolution', 'radiusCells', 'showBand'],
      readouts: ['cellSize', 'hotTotal'],
      stage: 'sums',
      camera: {longitude: -87.64, latitude: 41.92, zoom: 11.2, transitionMs: 1600},
      furniture: {title: cartouche('How near is near?', 'Gi* with a chosen neighbourhood')}
    },
    {
      id: 'tests',
      title: 'Thousands of tests at once',
      headline: 'Correct for testing thousands of cells at once',
      textAlternative:
        'The same hot-spot map with fewer coloured cells: only cells that survive the false discovery correction and the permutation test keep their colour.',
      body: 'Every cell is a test, so some pass by chance. The **Benjamini-Hochberg** correction and a **permutation test** (shuffle the other counts and see how often chance beats the cell) raise the bar: **{{hotTotal}}** hot cells survive and **{{confirmed}}** are confirmed.\n\nToggle **Benjamini-Hochberg FDR correction** or change **Permutations** and watch the counts fall.\n\n*The more tests, the stricter the bar.*',
      evidence:
        'The balance chart shows how many cells retain a named result; **{{confirmed}}** pass the conditional permutation test.',
      caveat:
        'FDR controls the expected false-discovery share across the family of tests; it does not guarantee that every coloured cell is real.',
      optionsMode: 'fresh',
      options: {
        source: 'nature-cells',
        display: 'classes',
        statistic: 'gi-star',
        inference: 'permutation',
        permutations: 199,
        seed: 1,
        falseDiscoveryRate: true
      },
      controls: ['falseDiscoveryRate', 'inference', 'permutations'],
      readouts: ['resultBalance', 'hotTotal', 'confirmed', 'zHistogram'],
      stage: 'test',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {title: cartouche('Which hot spots are real?', 'Gi* with false discovery control')}
    },
    {
      id: 'counties',
      title: 'The diabetes belt across US counties',
      headline: 'High diabetes clusters in the South and Appalachia',
      textAlternative:
        'Map of US counties coloured by local Moran quadrant: High-High clusters across the South and Appalachia, Low-Low clusters in the Mountain West and Upper Midwest.',
      body: 'The same machinery works on areas. Local Moran of diabetes prevalence finds **{{hotTotal}}** High-High counties, mostly across the South and Appalachia, and **{{coldTotal}}** Low-Low counties in the Mountain West and Upper Midwest: the belt that *Is it clustered at all?* measured with one number, now located county by county.\n\nTry **Statistic**, **Variable** or **Neighbours are...** below.\n\n*A cluster describes a pattern, not a cause; county values are model-based.*',
      evidence:
        'The local result profile separates **{{hotTotal}}** High-High counties, **{{coldTotal}}** Low-Low counties and the spatial outliers.',
      caveat:
        'County prevalence is model-based and local Moran depends on the chosen neighbour graph; neither establishes a causal regional effect.',
      optionsMode: 'fresh',
      options: {
        source: 'us-counties',
        variable: 'diabetes',
        weights: 'queen',
        weightTransform: 'row',
        statistic: 'local-moran',
        inference: 'analytic',
        selfWeight: 'exclude',
        display: 'classes'
      },
      controls: ['source', 'variable', 'statistic', 'weights'],
      readouts: ['resultBalance', 'hotTotal', 'coldTotal', 'outliers'],
      stage: 'classes',
      // An atlas page: no tiles, a flat sheet, state lines over the counties.
      basemap: ground('paperSheet'),
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche('Where does diabetes cluster?', 'Local Moran of prevalence', true),
        credit: COUNTY_CREDIT
      },
      // The chapter's CONUS frame (same bounds as the other county stories).
      camera: {bounds: [-124.8, 24.4, -66.9, 49.4], transitionMs: 1800},
      annotations: labelsFor(US, ['appalachia', 'mississippi-delta', 'great-plains', 'corn-belt'])
    }
  ]
});
