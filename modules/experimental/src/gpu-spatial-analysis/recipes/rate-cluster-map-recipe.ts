// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GPULocalPermutationTest} from '../permutation-inference/index';
import type {GPUPermutationAlternative} from '../permutation-inference/index';
import {GPUEmpiricalBayesRates} from '../rate-smoothing/index';
import {GPULocalMoran} from '../spatial-autocorrelation/index';
import {GPUContiguityWeights, GPUSpatialWeightsTransform} from '../spatial-weights/index';
import type {GPUContiguityCriterion, GPUSpatialWeights} from '../spatial-weights/index';
import {
  createTransientSpatialWeights,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeResult
} from './recipe-utils';

/** Number of palette entries the quadrant colors read: index 0 is "not significant". */
export const GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH = 5;

/** Which empirical-Bayes output is analysed. */
export type GPURateClusterMapRate = 'standardized' | 'smoothed';

/** Properties for {@link addRateClusterMapRecipe}. */
export type GPURateClusterMapRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'rate-cluster-map'`. */
  id?: string;
  /** Event count per polygon. */
  events: GraphDataView<'float32'>;
  /** Population at risk per polygon (rows with a population of 0 are excluded). */
  populations: GraphDataView<'float32'>;
  /** Optional polygon selection. */
  mask?: GraphDataView<'uint32'>;
  /** Flattened polygon ring vertices (GeoArrow layout). */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry; the row count is its length minus one. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** `'queen'` (default) or `'rook'`. */
  criterion?: GPUContiguityCriterion;
  /** Vertex snap tolerance of the contiguity search. */
  snapTolerance?: number;
  /** Slots of the contiguity CSR. */
  neighborCapacity: number;
  /** Intermediate pair capacity of `GPUContiguityWeights`. */
  pairCapacity?: number;
  /** Rate analysed by local Moran. Defaults to `'standardized'` (esda `Moran_Local_Rate`, adjusted). */
  analyze?: GPURateClusterMapRate;
  /** `getGPUSpatialAutocorrelationParameterValues` view. */
  parameters: GraphDataView<'float32'>;
  /** Permutation inference gating the colors; analytic local Moran quadrants are used when absent. */
  permutation?: {
    /** `getGPUPermutationParameterValues` view. */
    parameters: GraphDataView<'uint32'>;
    /** Compile-time upper bound of the per-frame permutation count. */
    maximumPermutations: number;
    /** Tail of the pseudo p-value. Defaults to the contributor default. */
    alternative?: GPUPermutationAlternative;
    /** Compile-time neighbor bound, up to 64. */
    maximumNeighbors?: number;
    /** Benjamini-Hochberg control of `significant`. */
    falseDiscoveryRate?: boolean;
    /** Caller-owned pseudo p-values. */
    pseudoPValues?: GraphDataView<'float32'>;
    /** Caller-owned significance mask. */
    significant?: GraphDataView<'uint32'>;
  };
  /**
   * Packed rgba8 palette of `GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH` entries indexed by quadrant:
   * 0 not significant (grey), 1 high-high, 2 low-high, 3 low-low, 4 high-low.
   */
  palette: GraphDataView<'uint32'>;
  /** Caller-owned standardized rates. */
  standardizedRates?: GraphDataView<'float32'>;
  /** Caller-owned smoothed rates. */
  smoothedRates?: GraphDataView<'float32'>;
  /** Caller-owned raw rates. */
  rawRates?: GraphDataView<'float32'>;
  /** Caller-owned `GPU_EMPIRICAL_BAYES_SUMMARY` row. */
  summary?: GraphDataView<'float32'>;
  /** Caller-owned weights CSR pieces. */
  weights?: Partial<GPUSpatialWeights>;
  /** Caller-owned weights overflow flag. */
  weightsOverflow?: GraphDataView<'uint32'>;
  /** Caller-owned local Moran z-scores. */
  zScores?: GraphDataView<'float32'>;
  /** Caller-owned local Moran I. */
  localI?: GraphDataView<'float32'>;
  /** Caller-owned spatial lag of the centered rate. */
  spatialLag?: GraphDataView<'float32'>;
  /**
   * Caller-owned `GPULocalMoran` quadrants: gated by the analytic level without `permutation`,
   * ungated with it.
   */
  analyticQuadrants?: GraphDataView<'uint32'>;
  /** Caller-owned final quadrant code per polygon (0 not significant, 1 HH, 2 LH, 3 LL, 4 HL). */
  quadrants?: GraphDataView<'uint32'>;
  /** Caller-owned rgba8 color per polygon. */
  colors?: GraphDataView<'uint32'>;
};

/** Named outputs of {@link addRateClusterMapRecipe}. */
export type GPURateClusterMapRecipeResult = GPURecipeResult & {
  rowCount: number;
  /** The analysed rate (standardized or smoothed). */
  analyzedRates: GraphDataView<'float32'>;
  standardizedRates: GraphDataView<'float32'>;
  smoothedRates: GraphDataView<'float32'>;
  /** Row-standardized contiguity weights. */
  weights: GPUSpatialWeights;
  weightsOverflow: GraphDataView<'uint32'>;
  zScores: GraphDataView<'float32'>;
  spatialLag: GraphDataView<'float32'>;
  quadrants: GraphDataView<'uint32'>;
  colors: GraphDataView<'uint32'>;
  /** Present when `permutation` was requested. */
  permutation?: {
    exceedances: GraphDataView<'uint32'>;
    pseudoPValues: GraphDataView<'float32'>;
    significant: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
  };
};

/**
 * Rate cluster map recipe: counts and populations to significant local Moran clusters of
 * empirical-Bayes rates, colored by quadrant (esda `Moran_Local_Rate`).
 *
 * Chain: `GPUEmpiricalBayesRates` -> `GPUContiguityWeights` -> `GPUSpatialWeightsTransform`
 * (`'row'`, in place) -> `GPULocalMoran` -> `GPULocalPermutationTest` -> quadrant colors. The last
 * stage maps quadrants to palette colors. With `permutation`, `GPULocalMoran` runs with
 * `quadrantGating: 'none'` and the quadrant is kept only where the permutation `significant` mask
 * is nonzero; without it the analytic gate applies. Excluded rows (NaN rate) get quadrant 0.
 */
export function addRateClusterMapRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURateClusterMapRecipeProps
): GPURateClusterMapRecipeResult {
  const id = props.id ?? 'rate-cluster-map';
  const builder = new RecipeBuilder(graph);
  const rowCount = props.polygonOffsets.length - 1;
  if (rowCount < 3) {
    throw new Error('addRateClusterMapRecipe needs at least three polygons');
  }
  if (props.events.length !== rowCount || props.populations.length !== rowCount) {
    throw new Error('addRateClusterMapRecipe events and populations need one row per polygon');
  }
  const standardizedRates = getOrCreateView(
    graph,
    `${id}-standardized`,
    'float32',
    rowCount,
    props.standardizedRates
  );
  const smoothedRates = getOrCreateView(
    graph,
    `${id}-smoothed`,
    'float32',
    rowCount,
    props.smoothedRates
  );
  builder.add(
    new GPUEmpiricalBayesRates({
      id: `${id}-empirical-bayes`,
      events: props.events,
      populations: props.populations,
      mask: props.mask,
      standardizedRates,
      smoothedRates,
      rawRates: props.rawRates,
      summary: props.summary
    })
  );
  const analyzedRates = props.analyze === 'smoothed' ? smoothedRates : standardizedRates;

  const weights = createTransientSpatialWeights(
    graph,
    `${id}-weights`,
    rowCount,
    props.neighborCapacity,
    props.weights
  );
  const weightsOverflow = getOrCreateView(
    graph,
    `${id}-weights-overflow`,
    'uint32',
    1,
    props.weightsOverflow
  );
  builder.add(
    new GPUContiguityWeights({
      id: `${id}-contiguity`,
      criterion: props.criterion ?? 'queen',
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      polygonOffsets: props.polygonOffsets,
      snapTolerance: props.snapTolerance,
      weights,
      overflow: weightsOverflow,
      pairCapacity: props.pairCapacity
    })
  );
  builder.add(
    new GPUSpatialWeightsTransform({id: `${id}-row-standardize`, operation: 'row', weights})
  );

  const zScores = getOrCreateView(graph, `${id}-z-scores`, 'float32', rowCount, props.zScores);
  const spatialLag = getOrCreateView(
    graph,
    `${id}-spatial-lag`,
    'float32',
    rowCount,
    props.spatialLag
  );
  // With permutation inference the analytic quadrant gate is switched off, so the permutation
  // result is the only gate; without it the analytic gate applies.
  const localQuadrants = getOrCreateView(
    graph,
    `${id}-local-quadrants`,
    'uint32',
    rowCount,
    props.analyticQuadrants
  );
  builder.add(
    new GPULocalMoran({
      id: `${id}-local-moran`,
      weights,
      values: analyzedRates,
      parameters: props.parameters,
      zScores,
      localI: props.localI,
      spatialLag,
      quadrants: localQuadrants,
      quadrantGating: props.permutation ? 'none' : 'analytic'
    })
  );

  let permutation: GPURateClusterMapRecipeResult['permutation'];
  if (props.permutation) {
    const options = props.permutation;
    permutation = {
      exceedances: getOrCreateView(graph, `${id}-exceedances`, 'uint32', rowCount),
      pseudoPValues: getOrCreateView(
        graph,
        `${id}-pseudo-p-values`,
        'float32',
        rowCount,
        options.pseudoPValues
      ),
      significant: getOrCreateView(
        graph,
        `${id}-significant`,
        'uint32',
        rowCount,
        options.significant
      ),
      overflow: getOrCreateView(graph, `${id}-permutation-overflow`, 'uint32', 1)
    };
    builder.add(
      new GPULocalPermutationTest({
        id: `${id}-permutation`,
        weights,
        values: analyzedRates,
        statistic: 'localMoran',
        alternative: options.alternative,
        parameters: options.parameters,
        maximumPermutations: options.maximumPermutations,
        maximumNeighbors: options.maximumNeighbors,
        falseDiscoveryRate: options.falseDiscoveryRate,
        ...permutation
      })
    );
  }

  const quadrants = getOrCreateView(graph, `${id}-quadrants`, 'uint32', rowCount, props.quadrants);
  const colors = getOrCreateView(graph, `${id}-colors`, 'uint32', rowCount, props.colors);
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-quadrant-colors`,
      operation: 'GPURateClusterMapQuadrantColors',
      bindings: [
        {name: 'localQuadrants', view: localQuadrants, type: 'u32', access: 'read'},
        {
          name: 'gate',
          view: permutation ? permutation.significant : localQuadrants,
          type: 'u32',
          access: 'read'
        },
        {name: 'palette', view: props.palette, type: 'u32', access: 'read'},
        {name: 'quadrantOutput', view: quadrants, type: 'u32', access: 'read_write'},
        {name: 'colorOutput', view: colors, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      body: `let quadrant = select(0u, localQuadrants[localQuadrantsOffset + index], gate[gateOffset + index] != 0u);
  quadrantOutput[quadrantOutputOffset + index] = quadrant;
  colorOutput[colorOutputOffset + index] = palette[paletteOffset + quadrant];`
    })
  );
  return {
    contributors: builder.contributors,
    rowCount,
    analyzedRates,
    standardizedRates,
    smoothedRates,
    weights,
    weightsOverflow,
    zScores,
    spatialLag,
    quadrants,
    colors,
    permutation
  };
}
