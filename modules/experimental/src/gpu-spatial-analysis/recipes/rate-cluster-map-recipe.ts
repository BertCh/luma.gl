// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUInferenceMetadata, GPUMultipleTestingMethod} from '../contracts/index';
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
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

/** Number of palette entries the quadrant colors read: index 0 is "not significant". */
export const GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH = 5;

/**
 * Default of {@link GPURateClusterMapRecipeProps.multipleTesting}: Benjamini-Hochberg false
 * discovery rate control over all tested polygons.
 */
export const GPU_RATE_CLUSTER_MAP_DEFAULT_MULTIPLE_TESTING: GPUMultipleTestingMethod =
  'benjamini-hochberg';

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
  /**
   * Rate analysed by local Moran. Defaults to `'standardized'` (Assuncao-Reis, esda
   * `Moran_Local_Rate`, adjusted). `'smoothed'` analyses the global empirical-Bayes smoothed rate,
   * which uses the clamped prior variance of `clampPriorVariance`; the standardized path is
   * unaffected by that option.
   */
  analyze?: GPURateClusterMapRate;
  /**
   * Compile-time. Clamp the estimated empirical-Bayes prior variance at zero (Marshall 1991,
   * GeoDa) in the `'smoothed'` rates, passed to `GPUEmpiricalBayesRates`. Defaults to `true`;
   * `false` reproduces esda `Empirical_Bayes` exactly (a negative variance pushes smoothed rates
   * away from the pooled rate). Has no effect on `'standardized'` rates.
   */
  clampPriorVariance?: boolean;
  /**
   * Multiple-testing correction of the cluster significance, applied to whichever inference runs:
   * the analytic `GPULocalMoran` p-values, or the permutation pseudo p-values when `permutation`
   * is set. Defaults to `'benjamini-hochberg'`. The test family is every polygon with a valid
   * rate and at least one neighbor, never a viewport.
   *
   * Legend contract, stable across this setting: palette index 0 is "not significant" and indices
   * 1..4 are the quadrants high-high, low-high, low-low, high-low. The correction changes only
   * which polygons reach significance, never what a color means. `result.inference` states the
   * alternative, correction and permutation count in force.
   *
   * Without `permutation` the p-values are analytic normal approximations (two-sided), a quick
   * preview; permutation inference is the reference (999 permutations recommended, the esda
   * default). The alternative defaults to `'two-sided'`.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /**
   * CPU-side significance level recorded in `result.inference.significanceLevel` (default 0.05);
   * the GPU reads the level from `parameters` (analytic) or `permutation.parameters`. Keep them
   * equal.
   */
  significanceLevel?: number;
  /** `getGPUSpatialAutocorrelationParameterValues` view. */
  parameters: GraphDataView<'float32'>;
  /**
   * Permutation inference gating the colors (the reference method; 999 permutations
   * recommended). Analytic local Moran quadrants, a preview, are used when absent.
   */
  permutation?: {
    /** `getGPUPermutationParameterValues` view. */
    parameters: GraphDataView<'uint32'>;
    /** Compile-time upper bound of the per-frame permutation count. */
    maximumPermutations: number;
    /** Tail of the pseudo p-value. Defaults to `'two-sided'`. */
    alternative?: GPUPermutationAlternative;
    /** Compile-time neighbor bound, up to 64. */
    maximumNeighbors?: number;
    /**
     * CPU-side permutation count recorded in `result.inference`; the GPU reads the count from
     * `parameters`. Defaults to `maximumPermutations`.
     */
    permutations?: number;
    /** CPU-side seed recorded in `result.inference`; the GPU reads it from `parameters`. */
    seed?: number;
  };
  /**
   * Packed rgba8 palette of `GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH` entries indexed by quadrant:
   * 0 not significant (grey), 1 high-high, 2 low-high, 3 low-low, 4 high-low.
   */
  palette: GraphDataView<'uint32'>;
} & GPURecipeOverrides<
  Record<never, never>,
  {
    standardizedRates?: GraphDataView<'float32'>;
    smoothedRates?: GraphDataView<'float32'>;
    zScores?: GraphDataView<'float32'>;
    localI?: GraphDataView<'float32'>;
    quadrants?: GraphDataView<'uint32'>;
    colors?: GraphDataView<'uint32'>;
    weightsOverflow?: GraphDataView<'uint32'>;
    analytic?: {
      pValues?: GraphDataView<'float32'>;
      adjustedPValues?: GraphDataView<'float32'>;
    };
    permutation?: {
      exceedances?: GraphDataView<'uint32'>;
      pseudoPValues?: GraphDataView<'float32'>;
      adjustedPValues?: GraphDataView<'float32'>;
      significant?: GraphDataView<'uint32'>;
      overflow?: GraphDataView<'uint32'>;
    };
  },
  {
    rawRates?: GraphDataView<'float32'>;
    summary?: GraphDataView<'float32'>;
    weights?: Partial<GPUSpatialWeights>;
    spatialLag?: GraphDataView<'float32'>;
    analyticQuadrants?: GraphDataView<'uint32'>;
  }
>;

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
  /** How the p-values and the significance gate were produced, for building a legend. */
  inference: GPUInferenceMetadata;
  /**
   * Present without `permutation`: analytic two-sided p-values and their `multipleTesting`
   * adjusted values (NaN for polygons that are not tested).
   */
  analytic?: {
    pValues: GraphDataView<'float32'>;
    adjustedPValues: GraphDataView<'float32'>;
  };
  /** Present when `permutation` was requested. */
  permutation?: {
    exceedances: GraphDataView<'uint32'>;
    pseudoPValues: GraphDataView<'float32'>;
    /** Pseudo p-values adjusted by `multipleTesting` (NaN where not tested). */
    adjustedPValues: GraphDataView<'float32'>;
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
 * is nonzero; without it the analytic gate applies. Excluded rows (NaN rate, or islands) get
 * quadrant 0 and a NaN p-value: they are not part of the tested family, so they are never counted
 * as tested but not significant.
 *
 * Defaults: `multipleTesting` is `'benjamini-hochberg'` over all tested polygons and the
 * alternative is `'two-sided'`. Analytic inference (no `permutation`) is a preview; permutation
 * inference (999 permutations recommended) is the reference. The palette meaning is fixed (index 0
 * not significant, 1..4 the quadrants) whatever the correction; `result.inference` describes it.
 * `analyze: 'smoothed'` uses the clamped empirical-Bayes prior variance (`clampPriorVariance`).
 */
export function addRateClusterMapRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURateClusterMapRecipeProps
): GPURateClusterMapRecipeResult {
  const id = props.id ?? 'rate-cluster-map';
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const rowCount = props.polygonOffsets.length - 1;
  const multipleTesting = props.multipleTesting ?? GPU_RATE_CLUSTER_MAP_DEFAULT_MULTIPLE_TESTING;
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
    outputs.standardizedRates
  );
  const smoothedRates = getOrCreateView(
    graph,
    `${id}-smoothed`,
    'float32',
    rowCount,
    outputs.smoothedRates
  );
  builder.add(
    new GPUEmpiricalBayesRates({
      id: `${id}-empirical-bayes`,
      events: props.events,
      populations: props.populations,
      mask: props.mask,
      standardizedRates,
      smoothedRates,
      clampPriorVariance: props.clampPriorVariance,
      rawRates: scratch.rawRates,
      summary: scratch.summary
    })
  );
  const analyzedRates = props.analyze === 'smoothed' ? smoothedRates : standardizedRates;

  const weights = createTransientSpatialWeights(
    graph,
    `${id}-weights`,
    rowCount,
    props.neighborCapacity,
    scratch.weights
  );
  const weightsOverflow = getOrCreateView(
    graph,
    `${id}-weights-overflow`,
    'uint32',
    1,
    outputs.weightsOverflow
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

  const zScores = getOrCreateView(graph, `${id}-z-scores`, 'float32', rowCount, outputs.zScores);
  const spatialLag = getOrCreateView(
    graph,
    `${id}-spatial-lag`,
    'float32',
    rowCount,
    scratch.spatialLag
  );
  // With permutation inference the analytic quadrant gate is switched off, so the permutation
  // result is the only gate; without it the analytic gate applies.
  const localQuadrants = getOrCreateView(
    graph,
    `${id}-local-quadrants`,
    'uint32',
    rowCount,
    scratch.analyticQuadrants
  );
  // Analytic inference only: the p-values and their correction. With permutation inference the
  // permutation test is the only gate and Moran rejects a correction combined with gating 'none'.
  const analytic: GPURateClusterMapRecipeResult['analytic'] = props.permutation
    ? undefined
    : {
        pValues: getOrCreateView(
          graph,
          `${id}-analytic-p-values`,
          'float32',
          rowCount,
          outputs.analytic?.pValues
        ),
        adjustedPValues: getOrCreateView(
          graph,
          `${id}-analytic-adjusted-p-values`,
          'float32',
          rowCount,
          outputs.analytic?.adjustedPValues
        )
      };
  const localMoran = builder.add(
    new GPULocalMoran({
      id: `${id}-local-moran`,
      weights,
      values: analyzedRates,
      parameters: props.parameters,
      zScores,
      localI: outputs.localI,
      spatialLag,
      quadrants: localQuadrants,
      quadrantGating: props.permutation ? 'none' : 'analytic',
      multipleTesting: props.permutation ? 'none' : multipleTesting,
      pValues: analytic?.pValues,
      adjustedPValues: analytic?.adjustedPValues
    })
  );

  let permutation: GPURateClusterMapRecipeResult['permutation'];
  let permutationTest: GPULocalPermutationTest | undefined;
  if (props.permutation) {
    const options = props.permutation;
    const provided = outputs.permutation ?? {};
    permutation = {
      exceedances: getOrCreateView(
        graph,
        `${id}-exceedances`,
        'uint32',
        rowCount,
        provided.exceedances
      ),
      pseudoPValues: getOrCreateView(
        graph,
        `${id}-pseudo-p-values`,
        'float32',
        rowCount,
        provided.pseudoPValues
      ),
      adjustedPValues: getOrCreateView(
        graph,
        `${id}-permutation-adjusted-p-values`,
        'float32',
        rowCount,
        provided.adjustedPValues
      ),
      significant: getOrCreateView(
        graph,
        `${id}-significant`,
        'uint32',
        rowCount,
        provided.significant
      ),
      overflow: getOrCreateView(graph, `${id}-permutation-overflow`, 'uint32', 1, provided.overflow)
    };
    permutationTest = builder.add(
      new GPULocalPermutationTest({
        id: `${id}-permutation`,
        weights,
        values: analyzedRates,
        statistic: 'localMoran',
        alternative: options.alternative,
        parameters: options.parameters,
        maximumPermutations: options.maximumPermutations,
        maximumNeighbors: options.maximumNeighbors,
        multipleTesting,
        ...permutation
      })
    );
  }

  const quadrants = getOrCreateView(
    graph,
    `${id}-quadrants`,
    'uint32',
    rowCount,
    outputs.quadrants
  );
  const colors = getOrCreateView(graph, `${id}-colors`, 'uint32', rowCount, outputs.colors);
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
  // The metadata comes from the node that produces the p-values, so it cannot drift from the
  // settings packed for the GPU. The recipe records the seed only when the caller states it.
  let inference: GPUInferenceMetadata;
  if (permutationTest && props.permutation) {
    const {seed, permutations, maximumPermutations} = props.permutation;
    inference = permutationTest.getInferenceMetadata({
      seed: seed ?? 0,
      permutations: permutations ?? maximumPermutations,
      significanceLevel: props.significanceLevel
    });
    if (seed === undefined) {
      delete inference.seed;
    }
  } else {
    inference = localMoran.getInferenceMetadata(props.significanceLevel);
  }
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
    inference,
    analytic,
    permutation,
    outputs: {
      analyzedRates,
      standardizedRates,
      smoothedRates,
      zScores,
      quadrants,
      colors,
      ...(analytic ? {analytic} : {}),
      ...(permutation ? {permutation} : {})
    },
    intermediates: {weights, spatialLag},
    status: {
      stages: [
        {stage: 'weights', status: {overflow: weightsOverflow}},
        ...(permutation ? [{stage: 'permutation', status: {overflow: permutation.overflow}}] : [])
      ]
    }
  };
}
