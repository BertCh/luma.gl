// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {
  GPUInferenceAlternative,
  GPUInferenceMetadata,
  GPUMultipleTestingMethod,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPULocalPermutationTest
} from '../permutation-inference/index';
import {GPUEmpiricalBayesRates} from '../rate-smoothing/index';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPULocalMoran
} from '../spatial-autocorrelation/index';
import {GPUContiguityWeights, GPUSpatialWeightsTransform} from '../spatial-weights/index';
import type {
  GPUContiguityCriterion,
  GPUFloat32OrUint32Values,
  GPUSpatialWeights,
  GPUSpatialWeightsOutput
} from '../spatial-weights/index';
import {
  assertRecipe,
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

/**
 * Per-frame inference settings of {@link addRateClusterMapRecipe}. The recipe packs these settings
 * into its `GPUParameterBuffer`s and derives `result.inference` from the same object, so the
 * legend cannot describe a test other than the one that runs.
 */
export type GPURateClusterMapRecipeSettings = {
  /** Significance level of the cluster gate, analytic or permutation. Defaults to 0.05. */
  significanceLevel?: number;
  /** Permutation seed, with `permutation`. Defaults to 0. */
  seed?: number;
  /** Per-frame permutation count, with `permutation`. Defaults to `maximumPermutations`. */
  permutationCount?: number;
};

/** Which empirical-Bayes output is analysed. */
export type GPURateClusterMapRate = 'standardized' | 'smoothed';

/** Properties for {@link addRateClusterMapRecipe}. */
export type GPURateClusterMapRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'rate-cluster-map'`. */
  id?: string;
  /**
   * Event count per feature, `float32` or `uint32` (for example the `counts` of
   * `addPointsInPolygonsChoroplethRecipe`; exact in f32 below 2^24).
   */
  events: GPUFloat32OrUint32Values;
  /**
   * Population at risk per feature, `float32` or `uint32` (rows with a population of 0 are
   * excluded).
   */
  populations: GPUFloat32OrUint32Values;
  /** Optional feature selection. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Polygon or multipolygon features in the four-level `GPUPolygonGeometryPort` layout; the rows
   * of every output are the features (`featureOffsets.length - 1`), so the `counts` of a
   * points-in-polygons recipe on the same features chain in directly.
   */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
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
   * Inference settings: the recipe packs them into `parameters` (and `permutation.parameters`)
   * and builds `result.inference` from them; change them per frame with `result.setSettings`.
   * Defaults to `{}` (every setting at its default).
   */
  settings?: GPURateClusterMapRecipeSettings;
  /**
   * Local Moran parameters: a `GPUParameterBuffer` of at least
   * `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` float32 values that the recipe fills from
   * `settings`.
   */
  parameters: GPUParameterBuffer<'float32'>;
  /**
   * Permutation inference gating the colors (the reference method; 999 permutations
   * recommended). Analytic local Moran quadrants, a preview, are used when absent.
   */
  permutation?: {
    /**
     * A `GPUParameterBuffer` of at least `GPU_PERMUTATION_PARAMETER_LENGTH` uint32 values that the
     * recipe fills from `settings`.
     */
    parameters: GPUParameterBuffer<'uint32'>;
    /** Compile-time upper bound of the per-frame permutation count. */
    maximumPermutations: number;
    /** Tail of the pseudo p-value. Defaults to `'two-sided'`. */
    alternative?: GPUInferenceAlternative;
    /** Compile-time neighbor bound, up to 64. */
    maximumNeighbors?: number;
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
    /** Mirrors `result.status`. */
    status?: {
      weightsOverflow?: GPUStatusScalar;
      /** One-row unclamped neighbor-slot total of the contiguity weights (stage `weights`). */
      weightsRequiredCount?: GPUStatusScalar;
    };
    analytic?: {
      pValues?: GraphDataView<'float32'>;
      adjustedPValues?: GraphDataView<'float32'>;
    };
    permutation?: {
      exceedances?: GraphDataView<'uint32'>;
      pseudoPValues?: GraphDataView<'float32'>;
      adjustedPValues?: GraphDataView<'float32'>;
      significant?: GraphDataView<'uint32'>;
      status?: {overflow?: GPUStatusScalar};
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
  /** Row-standardized contiguity weights, with their `status`. */
  weights: GPUSpatialWeightsOutput;
  zScores: GraphDataView<'float32'>;
  spatialLag: GraphDataView<'float32'>;
  quadrants: GraphDataView<'uint32'>;
  colors: GraphDataView<'uint32'>;
  /**
   * How the p-values and the significance gate were produced, for building a legend. It is
   * derived from the packed `settings`.
   */
  inference: GPUInferenceMetadata;
  /**
   * Writes `settings` into the parameter buffers for the next submitted encoding and updates
   * `inference` from the same object. Throws, writing nothing, on invalid settings.
   */
  setSettings: (settings: GPURateClusterMapRecipeSettings) => void;
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
    status: {overflow: GPUStatusScalar};
  };
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the contiguity weights overflowed `neighborCapacity` (stage `weights`). */
    weightsOverflow: GPUStatusScalar;
    /**
     * Unclamped neighbor-slot total: feed it to `getGPUSpatialWeightsCapacityPlan` to size
     * `neighborCapacity` (stage `weights`).
     */
    weightsRequiredCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
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
  const rowCount = props.polygons.featureOffsets.length - 1;
  const multipleTesting = props.multipleTesting ?? GPU_RATE_CLUSTER_MAP_DEFAULT_MULTIPLE_TESTING;
  assertRecipe(id, rowCount >= 3, 'needs at least three features');
  assertRecipe(
    id,
    props.events.length === rowCount && props.populations.length === rowCount,
    'events and populations need one row per feature'
  );
  // One source of truth (rule 19): the recipe packs `settings` itself and derives the inference
  // metadata from the same object.
  const parameterBuffer = props.parameters;
  const permutationBuffer = props.permutation?.parameters ?? null;
  assertRecipe(
    id,
    parameterBuffer.length >= GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    `parameters needs ${GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} values`
  );
  assertRecipe(
    id,
    !permutationBuffer || permutationBuffer.length >= GPU_PERMUTATION_PARAMETER_LENGTH,
    `permutation.parameters needs ${GPU_PERMUTATION_PARAMETER_LENGTH} values`
  );
  const parameters = parameterBuffer.importToGraph(graph, `${id}-parameters`);
  const permutationParameters = permutationBuffer?.importToGraph(
    graph,
    `${id}-permutation-parameters`
  );
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
      clampPriorVariance: props.clampPriorVariance,
      output: {
        standardizedRates,
        smoothedRates,
        rawRates: scratch.rawRates,
        summary: scratch.summary
      }
    })
  );
  const analyzedRates = props.analyze === 'smoothed' ? smoothedRates : standardizedRates;

  const weights = createTransientSpatialWeights(
    graph,
    `${id}-weights`,
    rowCount,
    props.neighborCapacity,
    {
      ...scratch.weights,
      status: {
        overflow: outputs.status?.weightsOverflow,
        requiredCount: outputs.status?.weightsRequiredCount
      }
    }
  );
  const weightsOverflow = weights.status.overflow;
  const weightsRequiredCount = weights.status.requiredCount!;
  builder.add(
    new GPUContiguityWeights({
      id: `${id}-contiguity`,
      criterion: props.criterion ?? 'queen',
      polygons: props.polygons,
      snapTolerance: props.snapTolerance,
      output: weights,
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
      parameters,
      quadrantGating: props.permutation ? 'none' : 'analytic',
      multipleTesting: props.permutation ? 'none' : multipleTesting,
      output: {
        zScores,
        localI: outputs.localI,
        spatialLag,
        quadrants: localQuadrants,
        pValues: analytic?.pValues,
        adjustedPValues: analytic?.adjustedPValues
      }
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
      status: {
        overflow: getOrCreateView(
          graph,
          `${id}-permutation-overflow`,
          'uint32',
          1,
          provided.status?.overflow
        )
      }
    };
    permutationTest = builder.add(
      new GPULocalPermutationTest({
        id: `${id}-permutation`,
        weights,
        values: analyzedRates,
        statistic: 'localMoran',
        alternative: options.alternative,
        parameters: permutationParameters!,
        maximumPermutations: options.maximumPermutations,
        maximumNeighbors: options.maximumNeighbors,
        multipleTesting,
        output: {
          exceedances: permutation.exceedances,
          pseudoPValues: permutation.pseudoPValues,
          adjustedPValues: permutation.adjustedPValues,
          significant: permutation.significant,
          status: permutation.status
        }
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
  // The metadata comes from the node that produces the p-values; the same settings object is
  // packed for the GPU and described here.
  const getSettingsInference = (
    settings: GPURateClusterMapRecipeSettings
  ): GPUInferenceMetadata => {
    if (permutationTest && props.permutation) {
      return permutationTest.getInferenceMetadata({
        seed: settings.seed ?? 0,
        permutationCount: settings.permutationCount ?? props.permutation.maximumPermutations,
        significanceLevel: settings.significanceLevel
      });
    }
    return localMoran.getInferenceMetadata({significanceLevel: settings.significanceLevel});
  };
  const writeSettings = (settings: GPURateClusterMapRecipeSettings): GPUInferenceMetadata => {
    // Pack (and validate) both views before writing either.
    const values = getGPUSpatialAutocorrelationParameterValues({
      significanceLevel: settings.significanceLevel
    });
    const permutationValues =
      permutationTest && props.permutation
        ? permutationTest.getParameterValues({
            seed: settings.seed ?? 0,
            permutationCount: settings.permutationCount ?? props.permutation.maximumPermutations,
            significanceLevel: settings.significanceLevel
          })
        : null;
    const metadata = getSettingsInference(settings);
    parameterBuffer.write(values);
    if (permutationValues) {
      permutationBuffer!.write(permutationValues);
    }
    return metadata;
  };
  const inference = writeSettings(props.settings ?? {});
  const result: GPURateClusterMapRecipeResult = {
    contributors: builder.contributors,
    rowCount,
    analyzedRates,
    standardizedRates,
    smoothedRates,
    weights,
    zScores,
    spatialLag,
    quadrants,
    colors,
    inference,
    setSettings: settings => {
      result.inference = writeSettings(settings);
    },
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
      weightsOverflow,
      weightsRequiredCount,
      stages: [
        {
          stage: 'weights',
          status: {overflow: weightsOverflow, requiredCount: weightsRequiredCount}
        },
        ...(permutation ? [{stage: 'permutation', status: permutation.status}] : [])
      ]
    }
  };
  return result;
}
