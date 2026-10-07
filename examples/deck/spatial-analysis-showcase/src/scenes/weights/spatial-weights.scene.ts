// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {RAMP_NAMES} from '../../engine/ramps';
import {FOCUS_COLORS, ISLAND_COLOR} from './b4-colors';
import {getVariableInfo, VARIABLES} from './b4-geography';
import type {SpatialWeightsOptions} from './spatial-weights.compute';

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

function getDisplayLegend(state: SpatialWeightsOptions): LegendSpec {
  const variable = getVariableInfo(state.variable);
  switch (state.display) {
    case 'focus':
      return {
        kind: 'categories',
        title: 'Neighbours of the focus place',
        entries: [
          {color: FOCUS_COLORS[2], label: 'Focus place (click to move it)'},
          {color: FOCUS_COLORS[1], label: 'Neighbour in W'},
          {color: FOCUS_COLORS[0], label: 'Not a neighbour'}
        ],
        note: 'Lines join centroids along every link of the chosen matrix.'
      };
    case 'neighbors':
      return {
        kind: 'ramp',
        id: 'display',
        title: 'Neighbours per place',
        ramp: state.ramp,
        extent: 'gpu',
        unit: 'places',
        format: value => Math.round(value).toString()
      };
    case 'oneWay':
      return {
        kind: 'ramp',
        id: 'display',
        title: 'One-way links per place',
        ramp: state.ramp,
        extent: 'gpu',
        unit: 'links that are not returned',
        format: value => Math.round(value).toString()
      };
    case 'lag':
      return {
        kind: 'ramp',
        id: 'display',
        title: `Spatial lag of ${variable.label.toLowerCase()}`,
        ramp: state.ramp,
        extent: 'gpu',
        unit: state.lagNormalize ? variable.unit : `${variable.unit} x weight`,
        format: value => value.toFixed(variable.digits)
      };
    case 'summary': {
      const entry = SUMMARY_OPTIONS.find(option => option.value === state.summary);
      return {
        kind: 'ramp',
        id: 'display',
        title: `Neighbourhood: ${entry?.label.toLowerCase() ?? state.summary}`,
        ramp: state.ramp,
        extent: 'gpu',
        unit: state.summary === 'entropy' ? 'nats' : variable.unit,
        format: value => value.toFixed(state.summary === 'count' ? 0 : 2)
      };
    }
    default:
      return {
        kind: 'ramp',
        id: 'display',
        title: variable.label,
        ramp: state.ramp,
        extent: 'gpu',
        unit: variable.unit,
        format: value => value.toFixed(variable.digits)
      };
  }
}

function getSnippet(state: SpatialWeightsOptions): string {
  const imports = [
    state.source === 'queen' || state.source === 'rook'
      ? 'GPUContiguityWeights'
      : state.source === 'lattice'
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
    state.source === 'queen' || state.source === 'rook'
      ? `// 1. Which polygons touch? One output row per polygon, binary weights.
graph.add(new GPUContiguityWeights({
  criterion: '${state.source}',${Number(state.snapTolerance) > 0 ? `\n  snapTolerance: ${state.snapTolerance},          // metres, grid snap` : ''}
  positions: vertices, ringOffsets, polygonOffsets,
  weights: {offsets, neighbors, weights}, overflow
}));`
      : state.source === 'lattice'
        ? `// 1. Neighbours on a raster grid; a mask removes cells outside the map.
graph.add(new GPULatticeWeights({
  width, height, criterion: '${state.latticeCriterion}', radius: ${state.latticeRadius},
  mask, cellSize: [cellMetres, cellMetres],
  weights: {offsets, neighbors, weights}, overflow
}));`
        : `// 1. ${state.source === 'knn' ? `The ${state.k} nearest centroids of every place` : 'Every centroid within the distance band'}.
graph.add(new GPUNeighborSearch({
  mode: '${state.source === 'knn' ? 'knn' : 'radius'}',${state.source === 'knn' ? `\n  k: ${state.k},` : ''}
  gridSize: [256, 256], positions: centroids,
  parameters,                          // radius, weightKind, kernel, rowStandardize: buffer writes
  weights: {offsets, neighbors, weights, distances}, overflow
}));
parameters.write(getGPUNeighborSearchParameterValues({
  bounds, ${state.source === 'band' ? `radius: ${state.bandFactor} * medianSpacing, ` : ''}weightKind: '${state.weightKind}'${state.weightKind === 'kernel' ? `, kernel: '${state.kernel}'` : ''}${state.rowStandardize ? ', rowStandardize: true' : ''}
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
    'Build the neighbourhood every spatial statistic starts from: queen and rook contiguity of 3,109 US counties, nearest neighbours, distance bands, kernels, weight algebra, spatial lags and grid lattices, all on the GPU.',
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
    {id: 'chicago-tracts', role: 'polygons and health variables (alternate geography)'},
    {id: 'chicago-community-areas', role: 'names of the community areas that group tracts'}
  ],
  initialView: {longitude: -95.5, latitude: 38.2, zoom: 3.9},

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
          label: 'US counties (3,109)',
          help: 'Contiguous United States; Connecticut as 9 planning regions.'
        },
        {
          value: 'chicago-tracts',
          label: 'Chicago census tracts (791)',
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
      help: 'The rule that decides who is a neighbour. Each rule is its own contributor; the first use of a rule compiles it once.',
      options: [
        {
          value: 'queen',
          label: 'Queen contiguity',
          help: 'Polygons that share at least one boundary vertex (GPUContiguityWeights).'
        },
        {
          value: 'rook',
          label: 'Rook contiguity',
          help: 'Polygons that share a boundary edge, not just a corner (GPUContiguityWeights).'
        },
        {
          value: 'knn',
          label: 'k nearest centroids',
          help: 'The k closest centroids, whatever the borders (GPUNeighborSearch, kNN).'
        },
        {
          value: 'band',
          label: 'Distance band',
          help: 'Every centroid within a radius (GPUNeighborSearch, radius).'
        },
        {
          value: 'lattice',
          label: 'Regular grid (lattice)',
          help: 'The map rasterised to square cells; neighbours are adjacent cells (GPULatticeWeights).'
        }
      ]
    },
    {
      kind: 'select',
      id: 'snapTolerance',
      label: 'Vertex snap tolerance',
      group: 'Neighbours',
      apply: 'compile',
      default: '0',
      disabledWhen: state => state.source !== 'queen' && state.source !== 'rook',
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
      disabledWhen: state => state.source !== 'knn',
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
      disabledWhen: state => state.source !== 'knn',
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
      disabledWhen: state => state.source !== 'band',
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
      disabledWhen: state => state.source !== 'lattice',
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
      disabledWhen: state => state.source !== 'lattice',
      help: 'Cells within this many steps are neighbours (the contributor supports up to 32; the map caps it at 4 so the focus links stay drawable).'
    },
    {
      kind: 'toggle',
      id: 'latticeMask',
      label: 'Mask cells outside the map',
      group: 'Lattice',
      apply: 'param',
      default: true,
      disabledWhen: state => state.source !== 'lattice',
      help: 'On: sea and Canada (or the suburbs) are removed as rows and as neighbours. Off: they stay in the grid.'
    },
    {
      kind: 'select',
      id: 'weightKind',
      label: 'Weights inside the search',
      group: 'Distance weights',
      apply: 'param',
      default: 'binary',
      disabledWhen: state => state.source !== 'knn' && state.source !== 'band',
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
      label: 'Search kernel',
      group: 'Distance weights',
      apply: 'param',
      default: 'triangular',
      disabledWhen: state =>
        (state.source !== 'knn' && state.source !== 'band') || state.weightKind !== 'kernel',
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
      disabledWhen: state =>
        (state.source !== 'knn' && state.source !== 'band') ||
        state.weightKind !== 'inverseDistance',
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
      disabledWhen: state =>
        (state.source !== 'knn' && state.source !== 'band') ||
        state.weightKind !== 'inverseDistance',
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
      disabledWhen: state => state.source !== 'knn' && state.source !== 'band',
      help: 'Divides each row of weights by its sum. The Transform below does the same for any source.'
    },
    {
      kind: 'select',
      id: 'transform',
      label: 'Transform weights',
      group: 'Transform',
      apply: 'compile',
      default: 'none',
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
      disabledWhen: state => state.source === 'lattice',
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
        {value: 'neighbors', label: 'Neighbour count per place'},
        {
          value: 'oneWay',
          label: 'One-way links per place',
          help: 'Links a place lists that the other place does not return.'
        },
        {value: 'value', label: 'The variable'},
        {value: 'lag', label: 'Spatial lag of the variable'},
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
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.display === 'focus',
      help: 'The legend uses the same ramp table.',
      options: RAMP_NAMES.filter(name => name !== 'diverging').map(name => ({
        value: name,
        label: name.charAt(0).toUpperCase() + name.slice(1)
      }))
    },
    {
      kind: 'select',
      id: 'matrix',
      label: 'Links drawn',
      group: 'Display',
      apply: 'param',
      default: 'weights',
      disabledWhen: state => state.source === 'lattice' && state.display !== 'focus',
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
      default: true,
      disabledWhen: state => state.source === 'lattice',
      help: "Lines between centroids for the whole matrix. The focus place's links are always drawn."
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Draw boundaries',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Outlines of the polygons.'
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
      id: 'neighbors',
      label: 'Neighbours per place',
      help: 'Mean (over places that have any), minimum and maximum.'
    },
    {
      id: 'islands',
      label: 'Islands (no neighbours)',
      help: 'Places the statistics cannot use unless the rule changes.'
    },
    {
      id: 'asymmetric',
      label: 'Asymmetric slots',
      help: 'Positions of W that differ from W transpose, counted per slot (an unreturned pair counts as 2).'
    },
    {
      id: 'sums',
      label: 'S0 / S1 / S2',
      help: 'Sums libpysal and every autocorrelation test use: S0 = sum w, S1 = 1/2 sum (w + w^T)^2, S2 = sum (row + column sums)^2.'
    },
    {id: 'union', label: 'W union W transpose', help: 'Slots of the symmetrised neighbourhood.'},
    {id: 'lagRange', label: 'Lag range', help: 'Smallest and largest spatial lag of the variable.'},
    {
      id: 'spacing',
      label: 'Median centroid spacing',
      help: 'Distance from a centroid to its nearest other centroid: the unit of the band and kNN cap sliders.'
    },
    {id: 'band', label: 'Distance band radius'},
    {id: 'focus', label: 'Focus place', help: 'Neighbours of the focus place in the matrix drawn.'},
    {
      id: 'capacity',
      label: 'Capacity',
      help: 'Overflow flag of the producer; slots beyond the capacity are dropped.'
    }
  ],

  legends: state => {
    const legends: LegendSpec[] = [getDisplayLegend(state)];
    if (state.source !== 'lattice') {
      legends.push({
        kind: 'categories',
        title: 'Special places',
        entries: [{color: ISLAND_COLOR, label: 'Island: no neighbours'}]
      });
    }
    return legends;
  },

  snippet: getSnippet,

  about: {
    what: 'A spatial-weights matrix W lists, for every place, its neighbours and how much each counts. `GPUContiguityWeights` builds it from shared polygon boundaries, `GPUNeighborSearch` from distances between centroids, `GPULatticeWeights` from a regular grid. `GPUSpatialWeightsTransform` rescales it, `GPUSpatialWeightsAlgebra` combines neighbourhoods, `GPUSpatialWeightsSummary` and `GPUSpatialWeightsTranspose` check it, and `GPUSpatialLag` and `GPUNeighborhoodSummary` use it to describe what surrounds each place.',
    why: "Moran's I, Gi*, regression with spatial lags, GWR and smoothing all take W as an input, and each answer changes with it. Looking at W first (who is connected, who is isolated, whether it is symmetric) is the cheapest way to avoid a wrong conclusion.",
    howToRead:
      'Orange places are neighbours of the red focus place; lines join centroids along every link. Switch **Map shows** to count neighbours, find one-way links, or read the spatial lag. Red dots are islands. The numbers on the right are the diagnostics of the matrix actually analysed.'
  },

  create: async ctx => (await import('./spatial-weights.compute')).createSpatialWeights(ctx),

  story: [
    {
      id: 'neighbours',
      title: 'Which counties touch Cook County?',
      body: 'Every spatial statistic begins with one decision: **who counts as a neighbour**? The map shows the 3,109 counties of the contiguous United States. The red county is Cook County, Illinois; orange counties are its neighbours, and lines join the centroids of every pair the rule links.\n\n**`GPUContiguityWeights`** builds the *queen* rule on the GPU: two polygons are neighbours when they share at least one boundary vertex. It sorts 64-bit vertex keys and groups equal ones, so it is exact and deterministic, and it matches libpysal `Queen`. The readouts give the size of the matrix (9,102 pairs, about 5.9 neighbours per county). Click any county or pick a **Focus place** below to move the focus.',
      options: {source: 'queen', display: 'focus', focus: 'typical'},
      camera: {longitude: -88.2, latitude: 40.6, zoom: 5.4, transitionMs: 1400},
      callout: {coordinate: [-87.65, 41.84], text: 'Focus: Cook County, IL'},
      highlight: {readout: 'neighbors'},
      controls: ['source', 'focus'],
      readouts: ['links', 'neighbors']
    },
    {
      id: 'rook',
      title: 'Corners count for queen, not for rook',
      body: 'Set **Neighbours are...** to **rook** contiguity: polygons are neighbours only if they share a boundary *edge*. At the Four Corners, Utah, Colorado, Arizona and New Mexico meet at a single point. Under queen, San Juan County, New Mexico lists San Juan County, Utah as a neighbour; under rook it does not.\n\nThe choice changes the answer a statistic gives: rook always lists a subset of queen, so every degree falls or stays. Both rules are exact on this boundary data (**Vertex snap tolerance** only matters for digitised boundaries that nearly touch).',
      options: {source: 'rook', focus: 'corner'},
      camera: {longitude: -108.6, latitude: 37.0, zoom: 6.4, transitionMs: 1600},
      callout: {coordinate: [-109.045, 37.0], text: 'Four Corners'},
      highlight: {readout: 'links'},
      controls: ['source', 'focus', 'snapTolerance'],
      readouts: ['links', 'neighbors']
    },
    {
      id: 'nearest',
      title: 'Nearest neighbours ignore borders, but are one-way',
      body: "Contiguity leaves **islands**: Nantucket and San Juan County, Washington (red dots) touch nothing. **`GPUNeighborSearch`** uses distances between centroids instead. In kNN mode each county lists its **k** nearest centroids (6 here), so nobody is isolated; it writes exact neighbours with a bucket grid and deterministic tie-breaking (libpysal `KNN`).\n\nThe fill now counts **one-way links**: a county that lists B while B does not list it back. kNN is not symmetric, because a small county in a dense area is among many neighbours' nearest six. **`GPUSpatialWeightsTranspose`** reverses the direction (W^T); set **Links drawn** to the union of W and W^T to see the symmetrised matrix, and watch **Asymmetric slots** below.",
      options: {source: 'knn', k: 6, display: 'oneWay'},
      camera: {longitude: -96, latitude: 38.5, zoom: 3.9, transitionMs: 1600},
      highlight: {readout: 'asymmetric'},
      controls: ['k', 'display', 'matrix'],
      readouts: ['asymmetric', 'islands']
    },
    {
      id: 'kernels',
      title: 'Distance bands and kernel weights',
      body: 'A **distance band** lists every centroid within a radius; **Distance band** is measured in typical centroid spacings, so it works for counties and tracts alike. Here each link also carries a **kernel weight**, `K(d / h)` with the bisquare profile `15/16 (1 - z^2)^2`: close neighbours count most and the weight fades to zero at the band edge (libpysal `DistanceBand` and `Kernel`).\n\n**`GPUSpatialWeightsTransform`** then **row-standardises** the weights, `w_ij / sum_j w_ij`, so every row sums to 1 and a lag becomes a weighted mean. Try the other **Transform weights** choices, double standardisation, variance stabilising or symmetrise: the **Focus place** readout below shows the weight sum of the row.',
      options: {
        source: 'band',
        bandFactor: 1.5,
        weightKind: 'kernel',
        kernel: 'bisquare',
        transform: 'row',
        display: 'neighbors'
      },
      camera: {longitude: -92, latitude: 38.5, zoom: 4.6, transitionMs: 1600},
      highlight: {readout: 'focus'},
      controls: ['bandFactor', 'weightKind', 'kernel', 'transform'],
      readouts: ['focus', 'band', 'spacing']
    },
    {
      id: 'algebra',
      title: 'Combine neighbourhoods with set algebra',
      body: 'Neighbourhoods can be built from other neighbourhoods. **`GPUSpatialWeightsAlgebra`** takes the queen contiguity (A) and the 6 nearest centroids (B) and keeps a neighbour listed by **either** (union). Nantucket and the San Juan Islands are connected again, while every county that touches stays linked.\n\nThe same contributor does intersection, difference, **higher order** (neighbours of neighbours, libpysal `higher_order`), self weights (the Getis-Ord G* neighbourhood), a **subgraph** of well-populated places, and **block** weights (everyone in one state). **Combine neighbourhoods** (below) sets the operation, **Partner neighbours** sets B, and **Weight where both agree** set to *Binary (1)* reproduces libpysal 4.15, which gives every set-operation link weight 1.',
      options: {
        source: 'queen',
        weightKind: 'binary',
        transform: 'none',
        combine: 'union',
        partnerK: 6,
        display: 'neighbors',
        focus: 'island'
      },
      camera: {longitude: -72.5, latitude: 41.8, zoom: 5.6, transitionMs: 1600},
      callout: {coordinate: [-70.05, 41.28], text: 'Nantucket'},
      highlight: {readout: 'islands'},
      controls: ['combine', 'partnerK', 'weightRule'],
      readouts: ['islands', 'links']
    },
    {
      id: 'lag',
      title: 'What surrounds each tract?',
      body: "Now Chicago: 791 census tracts, queen contiguity (6.6 neighbours on average; O'Hare is the one island). **`GPUSpatialLag`** computes `sum_j w_ij x_j`, the value of the neighbourhood, and with normalisation the weighted mean of the neighbours. The map shows the lag of diabetes prevalence (**Variable** and **Normalise the lag by the weight sum** are below): it smooths the tract-to-tract noise, and plotted against the tract's own value it is the Moran scatterplot.\n\n**`GPUNeighborhoodSummary`** goes further: set **Map shows** to *Neighbourhood statistic*, then use **Neighbourhood statistic** for the median, the standard deviation, or the **entropy of community areas** among the neighbours, which is zero inside a community area and high on the borders (momepy `describe`).",
      options: {
        geography: 'chicago-tracts',
        combine: 'none',
        source: 'queen',
        display: 'lag',
        focus: 'typical'
      },
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.8, transitionMs: 1800},
      callout: {coordinate: [-87.63, 41.88], text: 'The Loop'},
      highlight: {readout: 'lagRange'},
      controls: ['variable', 'lagNormalize', 'display'],
      readouts: ['lagRange']
    },
    {
      id: 'lattice',
      title: 'A grid instead of polygons, and where to be careful',
      body: "Set **Neighbours are...** to the **lattice**. The tracts are rasterised to square cells and **`GPULatticeWeights`** links adjacent cells (libpysal `lat2W`): rook means 4 neighbours, queen 8 (**Lattice neighbourhood**), and **Lattice radius** extends the reach. Cells outside the city are masked out as rows *and* as neighbours; turn **Mask cells outside the map** off to see the lake and the suburbs join the grid.\n\n**Limits.** Distances here are in flattened Web Mercator metres, fine for choosing neighbours, not for survey-grade work. Centroid rules ignore the shape of a place, and contiguity depends on how boundaries were digitised. Weights are a modelling choice: try a different rule and check that a conclusion survives. Next, run Moran's I and Gi* on these weights.",
      options: {
        source: 'lattice',
        display: 'neighbors',
        latticeCriterion: 'queen',
        latticeRadius: 1
      },
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.8, transitionMs: 1400},
      highlight: {readout: 'neighbors'},
      controls: ['source', 'latticeCriterion', 'latticeRadius', 'latticeMask'],
      readouts: ['neighbors', 'rows']
    }
  ]
});
