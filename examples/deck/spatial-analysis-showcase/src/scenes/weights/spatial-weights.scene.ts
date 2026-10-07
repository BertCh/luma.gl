// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CHICAGO, CITY_FRAMES, labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import {getVariableInfo, VARIABLES} from './b4-geography';
import {getVertexDiagram} from './spatial-weights.charts';
import type {SpatialWeightsOptions, WeightsLegendData} from './spatial-weights.compute';
import {
  CONUS_BOUNDS,
  getCardinalityTable,
  getContextSwatch,
  getEffectiveDisplay,
  getFocusPalette,
  getOneWayTable,
  getWeightTable,
  MOUNTAIN_WEST_BOUNDS,
  type Ground
} from './spatial-weights.style';

const KERNEL_OPTIONS = [
  {value: 'gaussian', label: 'Gaussian', help: 'exp(-z^2 / 2) / sqrt(2 pi); never reaches zero.'},
  {
    value: 'triangular',
    label: 'Triangular',
    help: '1 - z; falls linearly to zero at the bandwidth.'
  },
  {
    value: 'epanechnikov',
    label: 'Epanechnikov',
    help: '3/4 (1 - z^2); PySAL calls it quadratic.'
  },
  {value: 'bisquare', label: 'Bisquare', help: '15/16 (1 - z^2)^2; PySAL calls it quartic.'},
  {value: 'uniform', label: 'Uniform', help: '1/2 inside the bandwidth, 0 outside.'}
] as const;

const SUMMARY_OPTIONS = [
  {value: 'count', label: 'Neighbours counted', help: 'How many members the neighbourhood has.'},
  {value: 'weightSum', label: 'Sum of weights', help: 'W, the total weight of the neighbourhood.'},
  {value: 'sum', label: 'Weighted sum (the lag)', help: 'sum of w x over the neighbours.'},
  {value: 'mean', label: 'Weighted mean', help: 'sum / W.'},
  {value: 'min', label: 'Minimum', help: 'Smallest neighbour value.'},
  {value: 'max', label: 'Maximum', help: 'Largest neighbour value.'},
  {
    value: 'standardDeviation',
    label: 'Standard deviation',
    help: 'Weighted spread of the neighbourhood.'
  },
  {value: 'median', label: 'Median', help: 'Unweighted median of the neighbour values.'},
  {
    value: 'entropy',
    label: 'Group diversity (entropy)',
    help: 'Shannon entropy of the states (counties) or community areas (tracts) among the neighbours. Zero inside a group, high on borders.'
  }
] as const;

/** The cartouche of one step (steps replace the whole `title` object; the sample line is runtime). */
const cartouche = (title: string, subtitle: string, modelled = false) => ({
  title,
  subtitle,
  ...(modelled ? {chips: ['Modelled'] as const} : {})
});

const COUNTY_CREDIT = joinCredits(
  CREDITS.usCensus,
  'CDC PLACES (public domain)',
  CREDITS.colorBrewer
);
const TRACT_CREDIT = joinCredits(
  CREDITS.usCensus,
  CREDITS.cityOfChicago,
  'CDC PLACES (public domain)',
  CREDITS.colorBrewer
);

/** A `[west, south, east, north]` frame around a gazetteer place. */
const around = (
  place: readonly [number, number],
  halfWidth: number,
  halfHeight: number
): [number, number, number, number] => [
  place[0] - halfWidth,
  place[1] - halfHeight,
  place[0] + halfWidth,
  place[1] + halfHeight
];

const isContiguity = (state: SpatialWeightsOptions) =>
  !state.lattice && (state.source === 'queen' || state.source === 'rook');
const isDistance = (state: SpatialWeightsOptions) =>
  !state.lattice && (state.source === 'knn' || state.source === 'band');

function getLegends(
  state: SpatialWeightsOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const legendData = data['weights'] as WeightsLegendData | undefined;
  const ground: Ground = legendData?.ground ?? 'light';
  const tables = legendData?.tables;
  const counts = legendData?.counts;
  const variable = getVariableInfo(state.variable);
  const display = getEffectiveDisplay(state);
  switch (display) {
    case 'focus': {
      const palette = getFocusPalette(ground);
      return [
        {
          kind: 'categories',
          title: 'Membership in W',
          layout: 'list',
          entries: [
            {color: palette[2], label: 'Focus place (click to move it)'},
            {color: palette[1], label: 'Neighbour of the focus'},
            {color: getContextSwatch(ground), label: 'Not a neighbour'}
          ],
          note: 'The line bundle joins the centroid of the focus to each neighbour.'
        }
      ];
    }
    case 'weights':
      return [
        getClassTableLegend(tables?.weights ?? getWeightTable(ground), {
          title: 'Weight in the focus row',
          id: 'weight-classes',
          layout: 'list',
          interactive: true,
          note: 'Classes of the weight divided by the largest weight of the row; line width follows the weight.'
        })
      ];
    case 'neighbors':
      return [
        getClassTableLegend(tables?.cardinality ?? getCardinalityTable(ground), {
          title: 'Neighbours per place',
          id: 'cardinality-classes',
          counts: counts?.cardinality,
          layout: 'list',
          interactive: true,
          note: 'Islands (no neighbours) are unfilled and ringed.'
        })
      ];
    case 'oneWay':
      return [
        getClassTableLegend(tables?.oneWay ?? getOneWayTable(ground), {
          title: 'One-way links per place',
          id: 'one-way-classes',
          counts: counts?.oneWay,
          layout: 'list',
          interactive: true,
          note: 'A link is one-way when the other place does not list this one back.'
        })
      ];
    case 'value':
    case 'lag':
    case 'difference':
    case 'summary': {
      const table =
        display === 'difference'
          ? tables?.difference
          : display === 'summary'
            ? (tables?.summary ?? tables?.value)
            : tables?.value;
      if (!table) {
        return [
          {kind: 'ramp', title: variable.label, ramp: 'ylorbr', extent: [0, 1], unit: variable.unit}
        ];
      }
      const entry = SUMMARY_OPTIONS.find(option => option.value === state.summary);
      const title =
        display === 'lag'
          ? `${variable.label}: value and neighbourhood`
          : display === 'difference'
            ? `Neighbourhood minus place: ${variable.label.toLowerCase()}`
            : display === 'summary'
              ? `Neighbourhood: ${entry?.label.toLowerCase() ?? state.summary}`
              : variable.label;
      return [
        getClassTableLegend(table, {
          title,
          id: `${display}-classes`,
          counts:
            display === 'summary'
              ? counts?.summary
              : display === 'difference'
                ? counts?.difference
                : counts?.value,
          layout: 'list',
          interactive: true,
          note:
            display === 'difference'
              ? '0 = the neighbourhood equals the place; orange = the neighbourhood is higher, purple = lower.'
              : display === 'lag'
                ? 'The lag uses the classes of the variable, so a change of colour is smoothing.'
                : table.method
        })
      ];
    }
    default:
      return [];
  }
}

function getSnippet(state: SpatialWeightsOptions): string {
  const kind = state.lattice ? 'lattice' : state.source;
  const imports = [
    kind === 'queen' || kind === 'rook'
      ? 'GPUContiguityWeights'
      : kind === 'lattice'
        ? 'GPULatticeWeights'
        : 'GPUNeighborSearch, getGPUNeighborSearchParameterValues',
    state.transform !== 'none' ? 'GPUSpatialWeightsTransform' : '',
    state.combine !== 'none' ? 'GPUSpatialWeightsAlgebra' : '',
    'GPUSpatialWeightsSummary',
    'GPUSpatialWeightsTranspose',
    'GPUSpatialLag',
    'GPUNeighborhoodSummary'
  ].filter(Boolean);
  const producer =
    kind === 'queen' || kind === 'rook'
      ? `// 1. Which polygons touch? One output row per polygon, binary weights.
graph.add(new GPUContiguityWeights({
  criterion: '${kind}',${Number(state.snapTolerance) > 0 ? `\n  snapTolerance: ${state.snapTolerance},          // metres, grid snap` : ''}
  positions: vertices, ringOffsets, polygonOffsets,
  weights: {offsets, neighbors, weights}, overflow
}));`
      : kind === 'lattice'
        ? `// 1. Neighbours on a raster grid; a mask removes cells outside the map.
graph.add(new GPULatticeWeights({
  width, height, criterion: '${state.latticeCriterion}', radius: ${state.latticeRadius},
  mask, cellSize: [cellMetres, cellMetres],
  weights: {offsets, neighbors, weights}, overflow
}));`
        : `// 1. ${kind === 'knn' ? `The ${state.k} nearest centroids of every place` : 'Every centroid within the distance band'}.
graph.add(new GPUNeighborSearch({
  mode: '${kind === 'knn' ? 'knn' : 'radius'}',${kind === 'knn' ? `\n  k: ${state.k},` : ''}
  gridSize: [256, 256], positions: centroids,
  parameters,                          // radius, weightKind, kernel, rowStandardize: buffer writes
  weights: {offsets, neighbors, weights, distances}, overflow
}));
parameters.write(getGPUNeighborSearchParameterValues({
  bounds, ${kind === 'band' ? `radius: ${state.bandFactor} * medianSpacing, ` : ''}weightKind: '${state.weightKind}'${state.weightKind === 'kernel' ? `, kernel: '${state.kernel}'` : ''}${state.rowStandardize ? ', rowStandardize: true' : ''}
}));`;
  const transform =
    state.transform === 'none'
      ? ''
      : `\n\n// 2. Rewrite the weights in place.
graph.add(new GPUSpatialWeightsTransform({
  operation: '${state.transform}',${state.transform === 'kernel' ? `\n  kernel: '${state.transformKernel}', bandwidth: '${state.bandwidth === 'adaptive' ? 'adaptive' : `${state.bandwidth} x median spacing`}',` : ''}${state.transform === 'double' ? `\n  doubleSum: '${state.doubleSum}',` : ''}
  weights: {offsets, neighbors, weights${state.transform === 'kernel' ? ', distances' : ''}}
}));`;
  const algebra =
    state.combine === 'none'
      ? ''
      : `\n\n// 3. Combine neighbourhoods on the GPU.
graph.add(new GPUSpatialWeightsAlgebra({
  operation: '${state.combine}',${
    ['union', 'intersection', 'difference', 'symmetricDifference'].includes(state.combine)
      ? `\n  left: queenWeights, right: nearestWeights, weightRule: '${state.weightRule}',`
      : state.combine === 'higherOrder'
        ? `\n  weights, order: ${state.order}, cumulative: ${state.cumulative},`
        : state.combine === 'selfWeight'
          ? '\n  weights, selfWeight: perRowSelfWeight,   // a per-row view, so it is a buffer write'
          : state.combine === 'subgraph'
            ? '\n  weights, mask: populationMask,'
            : `\n  groupIds, groupCount: ${state.geography === 'us-counties' ? 49 : 78},`
  }
  output: analysisWeights, overflow
}));`;
  return `import {${imports.join(', ')}} from '@luma.gl/experimental/gpu-spatial-analysis';

${producer}${transform}${algebra}

// Diagnostics, lag and neighbourhood statistics read the same CSR.
graph.add(new GPUSpatialWeightsSummary({weights, statistics, counts, cardinality}));
graph.add(new GPUSpatialWeightsTranspose({weights, output: transposed, asymmetricSlots}));
graph.add(new GPUSpatialLag({weights, values, normalize: ${state.lagNormalize}, output: lag}));
graph.add(new GPUNeighborhoodSummary({
  weights, values, categories: stateIds, includeFocal: ${state.includeFocal},${state.includeFocal ? ` focalWeight: ${state.focalWeight},` : ''}
  statistics: ['count', 'weightSum', 'sum', 'mean', 'min', 'max', 'standardDeviation', 'median'],
  output: table, modes, entropy, maximumNeighbors: 64, overflow
}));`;
}

const COOK = US.places['cook-county'].lngLat;
const FOUR_CORNERS = US.places['four-corners'].lngLat;

/**
 * Spatial weights on US counties and Chicago tracts: contiguity, nearest neighbours, distance
 * bands, kernels, algebra, lags and lattices. GPU work is in `spatial-weights.compute.ts`.
 */
export default defineScene<SpatialWeightsOptions>({
  id: 'spatial-weights',
  title: 'Who is my neighbour?',
  chapter: 'weights',
  order: 1,
  summary:
    'Build the neighbourhood every spatial statistic starts from: queen and rook contiguity of US counties, nearest neighbours, distance bands, kernels, weight algebra and the spatial lag of Chicago tracts, all on the GPU.',
  contributors: [
    'GPUContiguityWeights',
    'GPULatticeWeights',
    'GPUNeighborSearch',
    'GPUSpatialWeightsTransform',
    'GPUSpatialWeightsAlgebra',
    'GPUSpatialWeightsSummary',
    'GPUSpatialWeightsTranspose',
    'GPUSpatialLag',
    'GPUNeighborhoodSummary'
  ],
  datasets: [
    {id: 'us-counties', role: 'polygons and health variables (default geography)'},
    {id: 'us-states', role: 'state lines over the counties'},
    {id: 'chicago-tracts', role: 'polygons and health variables (alternate geography)'},
    {id: 'chicago-community-areas', role: 'community-area lines over the tracts, and their names'}
  ],
  initialView: {longitude: -96, latitude: 38.3, zoom: 3.9},

  options: [
    {
      kind: 'select',
      id: 'geography',
      label: 'Geography',
      group: 'Place',
      apply: 'compile',
      default: 'us-counties',
      help: 'The polygon coverage. Switching rebuilds every graph for the new row count.',
      options: [
        {
          value: 'us-counties',
          label: 'US counties (contiguous states)',
          help: 'Contiguous United States and DC; Connecticut as 9 planning regions.'
        },
        {
          value: 'chicago-tracts',
          label: 'Chicago census tracts',
          help: "Whole 2020 tracts inside the city, including the O'Hare island."
        }
      ]
    },
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable',
      group: 'Place',
      apply: 'param',
      default: 'diabetes',
      help: 'The column the lag and the neighbourhood summary are computed from. Switching it is a buffer write.',
      options: VARIABLES.map(variable => ({
        value: variable.id,
        label: variable.label,
        help: variable.help
      }))
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Focus place (or click the map)',
      group: 'Place',
      apply: 'param',
      default: 'typical',
      help: 'The place whose neighbours are highlighted and listed. Click any place to move it.',
      options: [
        {value: 'typical', label: 'A typical place (Cook County, IL / the Loop)'},
        {
          value: 'corner',
          label: 'Corner case (San Juan, NM / Woodlawn)',
          help: 'Queen and rook disagree here: the place touches neighbours only at a corner.'
        },
        {
          value: 'island',
          label: "An island (Nantucket, MA / O'Hare)",
          help: 'A place no polygon touches, so contiguity gives it no neighbours.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'source',
      label: 'Neighbours are...',
      group: 'Neighbours',
      apply: 'compile',
      default: 'queen',
      display: 'segmented',
      disabledWhen: state => state.lattice,
      help: 'The rule that decides who is a neighbour. Each rule is its own contributor; the first use of a rule compiles it once.',
      options: [
        {
          value: 'queen',
          label: 'Queen',
          help: 'Polygons that share at least one boundary vertex (GPUContiguityWeights).'
        },
        {
          value: 'rook',
          label: 'Rook',
          help: 'Polygons that share a boundary edge, not just a corner (GPUContiguityWeights).'
        },
        {
          value: 'knn',
          label: 'Nearest k',
          help: 'The k closest centroids, whatever the borders (GPUNeighborSearch, kNN).'
        },
        {
          value: 'band',
          label: 'Band',
          help: 'Every centroid within a radius (GPUNeighborSearch, radius).'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'showQueenOnly',
      label: 'Outline corner-only neighbours',
      group: 'Neighbours',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.lattice,
      help: "Builds queen and rook side by side (two producers, compiled once) and draws the focus place's queen-only neighbours as a dashed outline, whichever rule is on."
    },
    {
      kind: 'toggle',
      id: 'lattice',
      label: 'Use a regular grid (lattice) instead',
      group: 'Neighbours',
      apply: 'compile',
      default: false,
      help: 'The map rasterised to square cells; neighbours are adjacent cells (GPULatticeWeights, libpysal lat2W). Cells outside the map are masked.'
    },
    {
      kind: 'select',
      id: 'snapTolerance',
      label: 'Vertex snap tolerance',
      group: 'Neighbours',
      apply: 'compile',
      default: '0',
      expert: true,
      disabledWhen: state => !isContiguity(state),
      help: 'Contiguity compares vertices exactly. A tolerance snaps vertices to a grid first, which repairs digitised boundaries that nearly touch. Compile-time.',
      options: [
        {value: '0', label: 'Exact (shared vertices only)'},
        {value: '500', label: '500 m grid'},
        {
          value: '5000',
          label: '5 km grid',
          help: 'Coarse: joins places whose corners fall in one 5 km cell.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'k',
      label: 'Nearest neighbours (k)',
      group: 'Neighbours',
      apply: 'compile',
      min: 1,
      max: 16,
      step: 1,
      default: 6,
      display: 'stepper',
      disabledWhen: state => state.lattice || state.source !== 'knn',
      help: 'How many nearest centroids each place lists. Compile-time: the search keeps a private sorted list of k.'
    },
    {
      kind: 'slider',
      id: 'knnCapFactor',
      label: 'kNN maximum distance',
      group: 'Neighbours',
      apply: 'param',
      min: 0,
      max: 6,
      step: 0.25,
      default: 0,
      disabledWhen: state => state.lattice || state.source !== 'knn',
      format: value => (value === 0 ? 'no limit' : `${value} x spacing`),
      help: 'Optional cap: a neighbour farther than this many typical centroid spacings is dropped, so remote places keep fewer than k.'
    },
    {
      kind: 'slider',
      id: 'bandFactor',
      label: 'Distance band',
      group: 'Neighbours',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 1.5,
      disabledWhen: state => state.lattice || state.source !== 'band',
      format: value => `${value} x spacing`,
      help: 'Radius as a multiple of the median distance from a centroid to its nearest other centroid (see the readout). Only a parameter write.'
    },
    {
      kind: 'select',
      id: 'latticeCriterion',
      label: 'Lattice neighbourhood',
      group: 'Lattice',
      apply: 'compile',
      default: 'queen',
      disabledWhen: state => !state.lattice,
      help: 'Rook: Manhattan distance up to the radius (4 neighbours at radius 1). Queen: Chebyshev distance (8 neighbours).',
      options: [
        {value: 'rook', label: 'Rook (edge-adjacent cells)'},
        {value: 'queen', label: 'Queen (edge or corner)'}
      ]
    },
    {
      kind: 'slider',
      id: 'latticeRadius',
      label: 'Lattice radius',
      group: 'Lattice',
      apply: 'compile',
      min: 1,
      max: 4,
      step: 1,
      default: 1,
      unit: 'cells',
      disabledWhen: state => !state.lattice,
      help: 'Cells within this many steps are neighbours (the contributor supports up to 32; the map caps it at 4 so the focus links stay drawable).'
    },
    {
      kind: 'toggle',
      id: 'latticeMask',
      label: 'Mask cells outside the map',
      group: 'Lattice',
      apply: 'param',
      default: true,
      disabledWhen: state => !state.lattice,
      help: 'On: sea and Canada (or the suburbs) are removed as rows and as neighbours. Off: they stay in the grid.'
    },
    {
      kind: 'select',
      id: 'weightKind',
      label: 'Weights inside the search',
      group: 'Distance weights',
      apply: 'param',
      default: 'binary',
      disabledWhen: state => !isDistance(state),
      help: 'What number a neighbour link carries.',
      options: [
        {value: 'binary', label: 'Binary (1)', help: 'Every neighbour counts the same.'},
        {
          value: 'inverseDistance',
          label: 'Inverse distance',
          help: 'max(d, floor)^-power: near neighbours weigh more.'
        },
        {value: 'kernel', label: 'Kernel', help: 'K(d / h): a smooth bump of distance.'}
      ]
    },
    {
      kind: 'select',
      id: 'kernel',
      label: 'Kernel',
      group: 'Distance weights',
      apply: 'param',
      default: 'triangular',
      disabledWhen: state => !isDistance(state) || state.weightKind !== 'kernel',
      help: 'With a distance band h is the radius; with kNN it is the distance of the k-th neighbour (an adaptive bandwidth, as in PySAL).',
      options: KERNEL_OPTIONS
    },
    {
      kind: 'slider',
      id: 'power',
      label: 'Inverse-distance power',
      group: 'Distance weights',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.5,
      default: 1,
      disabledWhen: state => !isDistance(state) || state.weightKind !== 'inverseDistance',
      help: 'Exponent p of d^-p. Larger values concentrate weight on the nearest neighbour.'
    },
    {
      kind: 'slider',
      id: 'distanceFloorFactor',
      label: 'Distance floor',
      group: 'Distance weights',
      apply: 'param',
      min: 0,
      max: 0.5,
      step: 0.05,
      default: 0,
      disabledWhen: state => !isDistance(state) || state.weightKind !== 'inverseDistance',
      format: value => `${value.toFixed(2)} x spacing`,
      help: 'Distances below the floor count as the floor, which keeps coincident points finite.'
    },
    {
      kind: 'toggle',
      id: 'rowStandardize',
      label: 'Row-standardise inside the search',
      group: 'Distance weights',
      apply: 'param',
      default: false,
      disabledWhen: state => !isDistance(state),
      help: 'Divides each row of weights by its sum. The Transform below does the same for any source.'
    },
    {
      kind: 'select',
      id: 'transform',
      label: 'Transform weights',
      group: 'Transform',
      apply: 'compile',
      default: 'none',
      disabledWhen: state => state.lattice,
      help: 'Rewrites the weights in place after they are produced (GPUSpatialWeightsTransform), like PySAL w.transform.',
      options: [
        {value: 'none', label: 'None (as produced)'},
        {
          value: 'row',
          label: 'Row standardisation (R)',
          help: 'w / row sum: every row sums to 1, so a lag is a weighted mean.'
        },
        {value: 'binary', label: 'Binary (B)', help: 'Any positive weight becomes 1.'},
        {
          value: 'double',
          label: 'Double standardisation (D)',
          help: 'w / S0: all weights together sum to 1 (or to n, see below).'
        },
        {
          value: 'variance',
          label: 'Variance stabilising (V)',
          help: 'Tiefelsdorf, Boots and Kozak 1999: s = w / sqrt(sum w^2), rescaled to sum to n.'
        },
        {
          value: 'symmetrize',
          label: 'Symmetrise ((w + w^T) / 2)',
          help: 'Averages each link with its reverse; a missing reverse counts as 0.'
        },
        {
          value: 'kernel',
          label: 'Distance kernel',
          help: 'Replaces weights by K(d / h). Needs distances, so it applies to kNN and distance-band sources only.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'transformKernel',
      label: 'Transform kernel',
      group: 'Transform',
      apply: 'compile',
      default: 'bisquare',
      disabledWhen: state => state.transform !== 'kernel',
      help: 'Kernel profile of the kernel transform.',
      options: KERNEL_OPTIONS
    },
    {
      kind: 'select',
      id: 'bandwidth',
      label: 'Kernel bandwidth',
      group: 'Transform',
      apply: 'compile',
      default: 'adaptive',
      disabledWhen: state => state.transform !== 'kernel',
      help: "Adaptive uses each row's farthest neighbour as h; a fixed bandwidth is the same for every place.",
      options: [
        {value: 'adaptive', label: 'Adaptive (per row)'},
        {value: '1.5', label: 'Fixed, 1.5 x spacing'},
        {value: '3', label: 'Fixed, 3 x spacing'},
        {value: '6', label: 'Fixed, 6 x spacing'}
      ]
    },
    {
      kind: 'select',
      id: 'doubleSum',
      label: 'Double standardisation total',
      group: 'Transform',
      apply: 'compile',
      default: 'one',
      disabledWhen: state => state.transform !== 'double',
      help: 'Scale so all weights sum to 1 (libpysal D) or to the number of places (Anselin).',
      options: [
        {value: 'one', label: 'Sum to 1'},
        {value: 'rows', label: 'Sum to n'}
      ]
    },
    {
      kind: 'select',
      id: 'combine',
      label: 'Combine neighbourhoods',
      group: 'Weights algebra',
      apply: 'compile',
      default: 'none',
      disabledWhen: state => state.lattice,
      help: 'Builds a new neighbourhood from the chosen one (A) with GPUSpatialWeightsAlgebra. Binary operations use the k nearest centroids as the partner (B).',
      options: [
        {value: 'none', label: 'None'},
        {value: 'union', label: 'A or B (union)', help: 'A neighbour in either list.'},
        {
          value: 'intersection',
          label: 'A and B (intersection)',
          help: 'A neighbour in both lists.'
        },
        {value: 'difference', label: 'A but not B (difference)'},
        {value: 'symmetricDifference', label: 'In exactly one (symmetric difference)'},
        {
          value: 'higherOrder',
          label: 'Higher order (neighbours of neighbours)',
          help: 'Places exactly n steps away along A.'
        },
        {
          value: 'selfWeight',
          label: 'Add self weight',
          help: 'Puts the place itself into its own neighbourhood: the G* of Getis-Ord.'
        },
        {
          value: 'subgraph',
          label: 'Only well-populated places',
          help: 'Drops places below a population percentile and every link to them.'
        },
        {
          value: 'block',
          label: 'Block weights (same state / community area)',
          help: 'Everyone in the same group is a neighbour.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'partnerK',
      label: 'Partner neighbours (B = k nearest)',
      group: 'Weights algebra',
      apply: 'compile',
      min: 2,
      max: 12,
      step: 1,
      default: 6,
      disabledWhen: state =>
        !['union', 'intersection', 'difference', 'symmetricDifference'].includes(state.combine),
      help: 'k of the second neighbourhood used by the binary operations.'
    },
    {
      kind: 'select',
      id: 'weightRule',
      label: 'Weight where both agree',
      group: 'Weights algebra',
      apply: 'compile',
      default: 'left',
      disabledWhen: state =>
        !['union', 'intersection', 'difference', 'symmetricDifference'].includes(state.combine),
      help: 'How the two weights of a shared link combine. "Binary" reproduces libpysal 4.15, which gives every set-operation link weight 1.',
      options: [
        {value: 'left', label: 'Weight of A'},
        {value: 'right', label: 'Weight of B'},
        {value: 'sum', label: 'Sum'},
        {value: 'min', label: 'Minimum'},
        {value: 'max', label: 'Maximum'},
        {value: 'product', label: 'Product'},
        {value: 'binary', label: 'Binary (1)'}
      ]
    },
    {
      kind: 'slider',
      id: 'order',
      label: 'Order (steps)',
      group: 'Weights algebra',
      apply: 'compile',
      min: 2,
      max: 8,
      step: 1,
      default: 2,
      disabledWhen: state => state.combine !== 'higherOrder',
      help: 'Neighbours exactly this many steps away along A. Compile-time (one expansion pass per order).'
    },
    {
      kind: 'toggle',
      id: 'cumulative',
      label: 'Include lower orders',
      group: 'Weights algebra',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.combine !== 'higherOrder',
      help: 'Lists every place within n steps instead of exactly n (libpysal lower_order=True).'
    },
    {
      kind: 'slider',
      id: 'selfWeight',
      label: 'Self weight',
      group: 'Weights algebra',
      apply: 'param',
      min: 0,
      max: 3,
      step: 0.25,
      default: 1,
      disabledWhen: state => state.combine !== 'selfWeight',
      help: 'Weight of a place in its own neighbourhood. A per-row view, so moving it is a buffer write.'
    },
    {
      kind: 'slider',
      id: 'populationPercentile',
      label: 'Population percentile kept',
      group: 'Weights algebra',
      apply: 'param',
      min: 0,
      max: 90,
      step: 5,
      default: 50,
      disabledWhen: state => state.combine !== 'subgraph',
      format: value => `top ${100 - value}%`,
      help: 'Places below this population percentile are masked out of the neighbourhood. The mask is a buffer.'
    },
    {
      kind: 'toggle',
      id: 'lagNormalize',
      label: 'Normalise the lag by the weight sum',
      group: 'Lag and neighbourhood',
      apply: 'compile',
      default: true,
      help: 'On: the lag is a weighted average (what row standardisation gives). Off: the raw sum of w x.'
    },
    {
      kind: 'toggle',
      id: 'includeFocal',
      label: 'Include the place itself in the summary',
      group: 'Lag and neighbourhood',
      apply: 'compile',
      default: false,
      help: 'Adds the focal place as a member of its own neighbourhood (momepy include_self).'
    },
    {
      kind: 'select',
      id: 'focalWeight',
      label: 'Weight of the place itself',
      group: 'Lag and neighbourhood',
      apply: 'compile',
      default: '1',
      disabledWhen: state => !state.includeFocal,
      help: 'Weight of the focal member when it is included. Compile-time.',
      options: [
        {value: '1', label: '1 (as one neighbour)'},
        {value: '2', label: '2'},
        {value: '4', label: '4'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'focus',
      help: 'Which GPU buffer the fill reads. All of them are computed every time the weights change.',
      options: [
        {value: 'focus', label: 'Neighbours of the focus place'},
        {
          value: 'weights',
          label: 'Weights of the focus row',
          help: 'Each neighbour classed by its weight as a share of the row maximum.'
        },
        {value: 'neighbors', label: 'Neighbour count per place'},
        {
          value: 'oneWay',
          label: 'One-way links per place',
          help: 'Links a place lists that the other place does not return.'
        },
        {value: 'value', label: 'The variable'},
        {
          value: 'lag',
          label: 'Variable and lag (swipe)',
          help: 'The variable and its spatial lag on one set of classes; a swipe in the story, the lag alone otherwise.'
        },
        {
          value: 'difference',
          label: 'Lag minus value',
          help: 'Where the neighbourhood differs from the place: orange when the neighbours are higher.'
        },
        {value: 'summary', label: 'Neighbourhood statistic'}
      ]
    },
    {
      kind: 'select',
      id: 'summary',
      label: 'Neighbourhood statistic',
      group: 'Display',
      apply: 'param',
      default: 'mean',
      disabledWhen: state => state.display !== 'summary',
      help: 'Which column of GPUNeighborhoodSummary the map shows.',
      options: SUMMARY_OPTIONS
    },
    {
      kind: 'select',
      id: 'matrix',
      label: 'Links drawn',
      group: 'Display',
      apply: 'param',
      default: 'weights',
      disabledWhen: state => state.lattice && state.display !== 'focus',
      help: 'W as built, its transpose W^T (who lists me), or the union W or W^T that makes any list symmetric.',
      options: [
        {value: 'weights', label: 'W (who I list)'},
        {value: 'transpose', label: 'W transpose (who lists me)'},
        {value: 'union', label: 'W union W transpose (symmetrised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showLinks',
      label: 'Draw every link',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.lattice,
      help: "Faint lines between centroids for the whole matrix, visible once you zoom in. The focus place's links are always drawn."
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Draw boundaries',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Outlines of the polygons and the state or community-area lines.'
    }
  ],

  readouts: [
    {id: 'rows', label: 'Places', help: 'Rows of the weights matrix.'},
    {
      id: 'links',
      label: 'Links (non-zero entries of W)',
      help: 'Directed neighbour slots; a symmetric rule counts each pair twice.'
    },
    {
      id: 'meanNeighbors',
      label: 'Mean neighbours',
      unit: 'per place',
      help: 'Links divided by the places that have at least one neighbour.'
    },
    {
      id: 'focusDegree',
      label: 'Neighbours of the focus',
      emphasis: 'tile',
      help: 'Neighbours of the focus place in the matrix drawn.'
    },
    {id: 'focusName', label: 'Focus place'},
    {
      id: 'islands',
      label: 'Islands (no neighbours)',
      help: 'Places the statistics cannot use unless the rule changes.'
    },
    {id: 'islandNames', label: 'Island names'},
    {
      id: 'edgeMean',
      label: 'Outer-edge places: mean neighbours',
      unit: 'neighbours',
      help: 'Places with a boundary edge that no other place shares (coast, lake shore, border, city limit), mean neighbours among those with any.'
    },
    {
      id: 'interiorMean',
      label: 'Interior places: mean neighbours',
      unit: 'neighbours',
      help: 'Every other place, mean neighbours among those with any.'
    },
    {
      id: 'cardinality',
      label: 'Neighbours per place',
      kind: 'chart',
      help: 'Places by neighbour count, coloured like the map; the marker is the common rule of thumb of eight.'
    },
    {
      id: 'asymmetric',
      label: 'Asymmetric slots',
      help: 'Positions of W that differ from W transpose, counted per slot (an unreturned pair counts as 2).'
    },
    {
      id: 'oneWayPlaces',
      label: 'Places with a one-way link',
      help: 'Places that list at least one neighbour that does not list them back.'
    },
    {
      id: 'cornerLinks',
      label: 'Corner-only links',
      help: 'Pairs that queen lists and rook does not (queen links minus rook links, halved).'
    },
    {
      id: 'cornerFocus',
      label: 'Corner-only neighbours of the focus',
      help: 'Places the focus touches at a corner but not along an edge.'
    },
    {id: 'cornerCount', label: 'Corner-only neighbours of the focus (count)'},
    {
      id: 'band',
      label: 'Distance band radius',
      help: 'The radius of the band as ground distance at the focus (the search itself runs in planar metres).'
    },
    {
      id: 'weightSum',
      label: 'Weight sum of the focus row',
      help: 'The weights of the focus row added up; 1 after row standardisation.'
    },
    {
      id: 'kernelCurve',
      label: 'Kernel and focus weights',
      kind: 'chart',
      help: 'The kernel curve K(d / h) scaled to the focus row, with the weight of each neighbour of the focus on it.'
    },
    {
      id: 'spacing',
      label: 'Median centroid spacing',
      help: 'Distance from a centroid to its nearest other centroid: the unit of the band and kNN cap sliders.'
    },
    {
      id: 'lagSlope',
      label: 'Slope of lag on value',
      help: 'Least-squares slope of the neighbourhood value on the place value, over places with a lag.'
    },
    {
      id: 'lagScatter',
      label: 'Lag against value',
      kind: 'chart',
      help: 'Each place: its value against the weighted mean of its neighbours, with the 1:1 line and the fitted slope.'
    },
    {
      id: 'lagRange',
      label: 'Lag range',
      hood: true,
      help: 'Smallest and largest spatial lag of the variable.'
    },
    {
      id: 'sums',
      label: 'S0 / S1 / S2',
      hood: true,
      help: 'Sums libpysal and every autocorrelation test use: S0 = sum w, S1 = 1/2 sum (w + w^T)^2, S2 = sum (row + column sums)^2.'
    },
    {
      id: 'union',
      label: 'W union W transpose',
      hood: true,
      help: 'Slots of the symmetrised neighbourhood.'
    },
    {
      id: 'capacity',
      label: 'Capacity',
      hood: true,
      help: 'Overflow flag of the producer; slots beyond the capacity are dropped.'
    },
    {
      id: 'numerics',
      label: 'Numerical notes',
      hood: true,
      layout: 'block',
      help: 'What the distances are measured in, and what that means at the focus.'
    }
  ],

  pipeline: [
    {
      id: 'sort',
      label: 'Keys and sort',
      detail:
        'Contiguity: one 64-bit key per ring vertex, radix-sorted. Nearest and band rules bucket the centroids into a grid instead.'
    },
    {
      id: 'pairs',
      label: 'Pairs',
      detail:
        'Runs of equal keys emit ordered pairs (queen: a shared vertex, rook: a shared edge); a kNN row keeps a private sorted list of k.'
    },
    {
      id: 'csr',
      label: 'CSR',
      detail: 'Dedupe, count and scan to offsets: row i lists its neighbours and their weights.',
      show: {option: 'display', value: 'neighbors'}
    },
    {
      id: 'transform',
      label: 'Transform',
      detail: 'Row standardisation and kernels rewrite the weight of every slot, one thread each.',
      show: {option: 'display', value: 'weights'}
    },
    {
      id: 'lag',
      label: 'Lag',
      detail: 'One thread per row sums w times the neighbour value in slot order.',
      show: {option: 'display', value: 'lag'}
    }
  ],

  legends: getLegends,

  basemap: ground('paperSheet'),
  furniture: {
    title: cartouche('Who counts as a neighbour?', 'Queen contiguity of counties'),
    credit: COUNTY_CREDIT
  },

  snippet: getSnippet,

  about: {
    what: 'A spatial-weights matrix W lists, for every place, its neighbours and how much each counts. `GPUContiguityWeights` builds it from shared polygon boundaries, `GPUNeighborSearch` from distances between centroids, `GPULatticeWeights` from a regular grid. `GPUSpatialWeightsTransform` rescales it, `GPUSpatialWeightsAlgebra` combines neighbourhoods, `GPUSpatialWeightsSummary` and `GPUSpatialWeightsTranspose` check it, and `GPUSpatialLag` and `GPUNeighborhoodSummary` use it to describe what surrounds each place.',
    why: "Moran's I, Gi*, regression with spatial lags, GWR and smoothing all take W as an input, and each answer changes with it. Looking at W first (who is connected, who is isolated, whether it is symmetric) is the cheapest way to avoid a wrong conclusion.",
    howToRead:
      'Green is membership in W: the neighbours of the focus place, or places classed by how many neighbours they list. The ink outline is the focus; the ink bundle joins it to its neighbours. Rings mark islands. Orange and brown classes show the variable and its lag; purple to orange shows where the neighbourhood differs from the place. The numbers on the right are the diagnostics of the matrix actually analysed.'
  },

  create: async ctx => (await import('./spatial-weights.compute')).createSpatialWeights(ctx),

  story: [
    {
      id: 'neighbours',
      title: 'Which counties touch Cook County?',
      headline: 'A neighbour is a shared boundary point',
      textAlternative:
        'Map of the Midwest: Cook County, Illinois is outlined in ink and its queen neighbours are filled green; every other county is left blank.',
      body: 'Every spatial statistic starts with one decision: who counts as a neighbour. Under the **queen** rule, counties that share any boundary point are neighbours. **{{focusName}}** has **{{focusDegree}}** of them, among **{{links}}** links in all, **{{meanNeighbors}}** per county on average. Choose a **Focus place** or click the map.\n\n*A weights matrix is a model of nearness, not a fact.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        source: 'queen',
        display: 'focus',
        focus: 'typical',
        showLinks: false
      },
      controls: ['focus'],
      readouts: ['focusDegree', 'links', 'meanNeighbors'],
      stage: 'sort',
      basemap: ground('paperSheet'),
      camera: {bounds: around(COOK, 2.5, 2.5), transitionMs: 1400},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche('Who touches Cook County?', 'Queen contiguity: a shared boundary point'),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, ['state-il', 'state-in', 'state-wi', 'state-mi']),
      diagram: getVertexDiagram()
    },
    {
      id: 'rook',
      title: 'Corners count for queen, not for rook',
      headline: 'Corners count for queen, not rook',
      textAlternative:
        'Map of the Four Corners: San Juan County, New Mexico is outlined in ink; its rook neighbours are green and San Juan County, Utah, which it touches only at a corner, has a dashed outline.',
      body: 'A **rook** neighbour shares an edge; a **queen** neighbour may share only a corner. Next to the focus, **{{cornerFocus}}** touches it at a single point, so only queen lists it (dashed outline). Across the map, queen adds **{{cornerLinks}}** corner-only links to rook. Switch **Neighbours are...** and watch **{{focusDegree}}** change.\n\n*Rook is always a subset of queen.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        source: 'rook',
        display: 'focus',
        focus: 'corner',
        showQueenOnly: true
      },
      controls: ['source'],
      readouts: ['focusDegree', 'links', 'cornerLinks'],
      stage: 'pairs',
      basemap: ground('paperSheet'),
      camera: {bounds: around(FOUR_CORNERS, 1.7, 1.4), transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Corners count for queen, not rook',
          'Edge or corner: the focus at the Four Corners'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, ['four-corners'])
    },
    {
      id: 'cardinality',
      title: 'How many neighbours is normal?',
      headline: 'Border and coast counties list fewer neighbours',
      textAlternative:
        'Map of the contiguous United States with counties in five green classes by neighbour count; counties along the coasts and borders are paler, and two island counties are ringed.',
      body: 'Counties list **{{meanNeighbors}}** neighbours on average, and a common rule of thumb asks for eight. Counties on a coast, lake shore or border list only **{{edgeMean}}**, against **{{interiorMean}}** inside the map, and **{{islands}}** list none. Try the other rules under **Neighbours are...**.\n\n*Edge effect: places at the boundary of the data have fewer neighbours.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        source: 'queen',
        display: 'neighbors',
        showLinks: false
      },
      controls: ['source'],
      readouts: ['meanNeighbors', 'edgeMean', 'islands', 'cardinality'],
      stage: 'csr',
      basemap: ground('paperSheet'),
      camera: {bounds: [...CONUS_BOUNDS], transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'How many neighbours is normal?',
          'Neighbour count of each county, fixed classes'
        ),
        credit: COUNTY_CREDIT
      }
    },
    {
      id: 'nearest',
      title: 'Nearest neighbours: no islands, but one-way',
      headline: 'Nearest neighbours: nobody isolated, but links go one way',
      textAlternative:
        'Map of the Mountain West with counties in classes by the number of one-way links they list; the large, sparse western counties carry the most.',
      body: 'Listing the **k** nearest centroids leaves **{{islands}}** counties isolated. But nearness is not mutual: **{{oneWayPlaces}}** counties list a neighbour that does not list them back (**{{asymmetric}}** unreturned slots). Change **Nearest neighbours (k)**, or draw W, its transpose or their union with **Links drawn**. A union with queen reconnects islands too.\n\n*Nearness by distance is not symmetric.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        source: 'knn',
        k: 6,
        display: 'oneWay',
        matrix: 'weights'
      },
      controls: ['k', 'matrix'],
      readouts: ['islands', 'oneWayPlaces', 'asymmetric'],
      stage: 'pairs',
      basemap: ground('paperSheet'),
      camera: {bounds: [...MOUNTAIN_WEST_BOUNDS], transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Nearest neighbours go one way',
          'One-way links of the k nearest centroids'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, [
        'state-mt',
        'state-wy',
        'state-co',
        'state-nm',
        'state-ut',
        'state-id'
      ])
    },
    {
      id: 'kernels',
      title: 'Distance fades: kernels and row standardisation',
      headline: 'Weights fade with distance and sum to one',
      textAlternative:
        'Map around Cook County with a dashed ring for the distance band; neighbours inside it are classed green by weight and the line widths follow the weights.',
      body: 'A distance band lists every centroid within **{{band}}** of the focus: **Distance band** times the **{{spacing}}** median spacing. A bisquare kernel weights each neighbour by distance, and row standardisation makes the row sum to **{{weightSum}}**. Line width is the weight. Drag **Distance band** or change **Kernel**.\n\n*A kernel makes W graded, not 0 or 1.*\n\n*Numerical note: the search runs in planar metres; the ring is the same disc at ground distance.*',
      optionsMode: 'fresh',
      options: {
        geography: 'us-counties',
        source: 'band',
        bandFactor: 2.5,
        weightKind: 'kernel',
        kernel: 'bisquare',
        transform: 'row',
        display: 'weights',
        focus: 'typical'
      },
      controls: ['bandFactor', 'kernel'],
      readouts: ['band', 'weightSum', 'spacing', 'kernelCurve'],
      stage: 'transform',
      basemap: ground('paperSheet'),
      camera: {bounds: around(COOK, 1.9, 1.5), transitionMs: 1600},
      furniture: {
        ...NATIONAL_FURNITURE,
        title: cartouche(
          'Weights fade with distance',
          'Bisquare kernel, row-standardised distance band'
        ),
        credit: COUNTY_CREDIT
      },
      annotations: labelsFor(US, ['state-il', 'state-in', 'state-wi'])
    },
    {
      id: 'lag',
      title: 'Does the neighbourhood agree with the place?',
      headline: 'The neighbourhood mostly agrees with the place',
      textAlternative:
        'Map of Chicago census tracts in five orange classes with a swipe divider: on the left each tract value, on the right the average of its neighbours, which is smoother.',
      body: "Now Chicago: **{{rows}}** tracts. The spatial lag averages each tract's neighbours; swipe from the tract itself to its neighbourhood on the same classes. The lag rises with the value at a slope of **{{lagSlope}}**; *Is it clustered at all?* tests that against chance. Change **Neighbours are...** and the lag moves. A lattice grid is the alternative.\n\n*Every later statistic inherits W.*",
      optionsMode: 'fresh',
      options: {
        geography: 'chicago-tracts',
        source: 'queen',
        display: 'lag',
        variable: 'diabetes',
        lagNormalize: true
      },
      controls: ['source', 'variable', 'display'],
      readouts: ['rows', 'meanNeighbors', 'lagSlope', 'lagScatter'],
      stage: 'lag',
      basemap: ground('paperCity'),
      camera: {...CITY_FRAMES.chicago, transitionMs: 1800},
      compare: {mode: 'swipe', labels: ['Tract value', 'Neighbourhood (lag)'], position: 0.5},
      furniture: {
        title: cartouche(
          'Does the neighbourhood agree?',
          'Diabetes prevalence, tract and neighbourhood',
          true
        ),
        credit: TRACT_CREDIT,
        scaleBar: false,
        caveat: ''
      },
      annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop', 'ohare'])
    }
  ]
});
