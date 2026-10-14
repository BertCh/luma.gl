// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPULineGeometryFields} from '../../analysis-core/contracts/index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {
  createLineImportanceInitNode,
  createLineImportanceRoundNodes,
  createLineImportanceUnresolvedNode,
  createLineKeepMaskNode,
  createLineKeptRangesNode,
  type GPULineSimplificationMetric,
  type LineImportanceScratch
} from './line-simplification-kernels';
import {GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} from './line-simplification-parameters';
import {
  createVisvalingamInitNode,
  createVisvalingamRoundNodes,
  createVisvalingamUnresolvedNode,
  type VisvalingamScratch
} from './visvalingam-kernels';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPULineSimplification';

/** Splits `output` into the row selection (present when `ids` is given) and the status bag. */
function getOutputParts(props: GPULineSimplificationProps): {
  selection: GPULineSimplificationSelectionOutput | undefined;
  status: GPULineSimplificationStatus | undefined;
} {
  const {output} = props;
  return {
    selection: output?.ids ? (output as GPULineSimplificationSelectionOutput) : undefined,
    status: output?.status
  };
}

/** Default compile-time cap on Douglas-Peucker iterations. */
export const GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS = 64;

/**
 * Default `finishSpanLimit`: Douglas-Peucker intervals with at most this many interior rows are
 * solved whole by one lane instead of one level per round.
 */
export const GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT = 32;

/** Largest accepted `finishSpanLimit`. */
export const GPU_LINE_SIMPLIFICATION_MAXIMUM_FINISH_SPAN_LIMIT = 128;

/** Largest accepted compile-time iteration cap. */
export const GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS = 1024;

/** Default compile-time cap on Visvalingam-Whyatt iterations (`method: 'visvalingam'`). */
export const GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS = 256;

/** Default `neighborhoodRadius` of `method: 'visvalingam'`. */
export const GPU_LINE_SIMPLIFICATION_DEFAULT_NEIGHBORHOOD_RADIUS = 3;

/** Largest accepted `neighborhoodRadius`. */
export const GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS = 8;

/**
 * Simplification algorithm. `'douglas-peucker'` (default) is exact recursive Douglas-Peucker (or
 * TD-TR with the `'time-ratio'` metric); `'visvalingam'` is Visvalingam-Whyatt effective-area
 * simplification, approximated by parallel rounds (see {@link GPULineSimplification}).
 */
export type GPULineSimplificationMethod = 'douglas-peucker' | 'visvalingam';

/** Optional scalar diagnostics of the importance pass. */
export type GPULineSimplificationStatus = {
  /** One-row scalar: `1` when every row was decided within `maximumIterations`, otherwise `0`. */
  converged?: GraphDataView<'uint32'>;
  /** One-row scalar receiving the number of iterations (importance rounds) that ran. */
  iterationCount?: GraphDataView<'uint32'>;
};

/**
 * Per-frame selection outputs of {@link GPULineSimplification}, all written from the current
 * tolerance in `parameters`. The kept rows are the {@link GPUCompactOutput} fields: `ids` lists them
 * in ascending order (line order, then row order), `count` is clamped to `ids.length` so it can
 * drive an indirect draw, and `overflow` is 1 when more rows were kept.
 */
export type GPULineSimplificationSelectionOutput = GPUCompactOutput & {
  /** Optional per-row `1`/`0` mask of the kept rows, one row per position. */
  outputMask?: GraphDataView<'uint32'>;
  /** Optional kept-row count of each line, `lineCount` rows. Not clamped by the capacity. */
  lineCounts?: GraphDataView<'uint32'>;
  /**
   * Optional position of each line's first kept row inside `ids`, `lineCount` rows. A line
   * with `lineStarts + lineCounts > ids.length` is truncated by the capacity.
   */
  lineStarts?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned outputs of {@link GPULineSimplification}. Give the selection fields (`ids`,
 * `count`, `overflow`, ...) to select the kept rows at the current tolerance, and `status` to
 * report the importance pass; at least one of them is needed unless `importance` is computed.
 */
export type GPULineSimplificationOutput = {
  /** Optional convergence diagnostics; only with `computeImportance`. */
  status?: GPULineSimplificationStatus;
} & (GPULineSimplificationSelectionOutput | {ids?: undefined});

/**
 * Properties for {@link GPULineSimplification}.
 *
 * Per-frame (no recompile): the contents of `parameters` (the tolerance) and of every input
 * buffer. Compile-time (needs a new graph): view lengths, the line count (`lineOffsets.length - 1`),
 * `metric`, `maximumIterations`, `computeImportance`, the output capacity, and which optional views are
 * present.
 */
export type GPULineSimplificationProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-simplification'`. */
  id?: string;
  /**
   * Native planar coordinates (default), or `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}`: positions are `[longitude, latitude]` degrees, the Douglas-Peucker distances
   * (and so `tolerance` and `importance`) are great-circle meters on a sphere of `sphereRadius`
   * (default `GPU_EARTH_MEAN_RADIUS_METERS`), and chords are straight in longitude/latitude, as in
   * the rest of the package. The distance from a vertex to its chord is the haversine distance to
   * the closest point of the chord, found in the local equirectangular frame of the vertex (exact
   * to first order for chords of kilometers; simplify long edges after densifying them). Kilometers
   * and other metrics throw. `method: 'visvalingam'` is planar only and throws in a great-circle
   * context: its tolerance is an area, its parallel rounds mirror a bit-exact planar triangle
   * area, and no spherical-excess variant is implemented; project to a planar frame first, for
   * example with `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context. Great-circle distances use f32
   * transcendental functions, so the bit-exactness described on {@link GPULineSimplification}
   * holds for planar contexts only.
   */
  spatialContext?: GPUSpatialContext;
  // `positions` (one row per vertex, sorted by line and then vertex order; use tile- or
  // view-local coordinates since distances are f32) and `lineOffsets` come from
  // `GPULineGeometryFields`. Rows outside `[lineOffsets[0], lineOffsets[lineCount])` get
  // importance 0 and are never kept.
  /**
   * Packed float32 times aligned with `positions`. Required for `metric: 'time-ratio'`, ignored
   * otherwise.
   */
  timestamps?: GraphDataView<'float32'>;
  /**
   * Simplification algorithm. Default `'douglas-peucker'`. With `'visvalingam'` the importance of a
   * vertex is its effective triangle area in squared position units, the tolerance is an area, the
   * `'segment'` metric applies, and `timestamps` are ignored.
   */
  method?: GPULineSimplificationMethod;
  /**
   * `method: 'visvalingam'` only. A vertex is removed in a round when its triangle area is the
   * smallest among the surviving vertices within this many steps on both sides, in `[1, 8]`.
   * Default 3. Larger values follow the sequential smallest-area-first order more closely and need
   * more rounds; `1` is fastest and deviates the most. Compile-time.
   */
  neighborhoodRadius?: number;
  /**
   * Distance used by the importance pass. `'segment'` (default) is the Euclidean distance from a
   * vertex to the chord segment between the interval's anchors (perpendicular distance when the
   * vertex projects inside the chord, distance to the nearer anchor otherwise, as in GEOS and
   * simplify-js). `'time-ratio'` is TD-TR's synchronized Euclidean distance: the distance from a
   * vertex to the chord position linearly interpolated at the vertex's timestamp.
   */
  metric?: GPULineSimplificationMetric;
  /**
   * Per-vertex importance column, `positions.length` rows. Written when `computeImportance` is
   * true, otherwise read as the result of an earlier encoding (for example a one-shot graph).
   */
  importance: GraphDataView<'float32'>;
  /**
   * Whether this contributor computes `importance`. Default `true`. Set to `false` in a per-frame graph
   * that only selects rows from an importance column computed once by another graph.
   */
  computeImportance?: boolean;
  /**
   * Compile-time cap on level-synchronous iterations (rounds), in `[1, 1024]`. Default 64 (256 for
   * `method: 'visvalingam'`). A balanced split
   * tree needs about `log2(n)` rounds; adversarial shapes (spirals) need up to `n - 2`. Each round
   * adds five nodes, and rounds after convergence are skipped on the GPU.
   */
  maximumIterations?: number;
  /**
   * Douglas-Peucker only. An open interval with at most this many interior rows is solved
   * completely by one lane (explicit-stack recursion, same distances, ties and importance as the
   * level rounds) instead of one split level per round, which removes the long tail of rounds
   * that a few small intervals otherwise need and so also lowers `iterationCount`. In `[0, 128]`,
   * default {@link GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT}; `0` runs rounds only.
   * Converged importance is identical for every value; only the iteration count and what a
   * `maximumIterations` cap leaves undecided change. Compile-time.
   */
  finishSpanLimit?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH}
   * elements written with `getGPULineSimplificationParameterValues`. Required when `output`
   * selects rows.
   */
  parameters?: GraphDataView<'float32'>;
  /** Optional selection at the current tolerance and importance diagnostics. */
  output?: GPULineSimplificationOutput;
};

/**
 * Douglas-Peucker (and TD-TR) simplification of many polylines or tracks as a per-vertex
 * importance column plus a per-frame tolerance selection.
 *
 * **Importance.** Each line starts as one open interval between its endpoints (importance
 * `+Infinity`). Every round, for every open interval at once, a segmented argmax finds the row
 * farthest from the interval's chord (`atomicMax` on the non-negative f32 distance bits, then
 * `atomicMin` on the row among rows with that key, so ties go to the smallest row), decides it,
 * and splits the interval there. The decided row's importance is
 * `min(distance, min(importance[a], importance[b]))` for anchors `a` and `b`, which is the
 * importance of the split that created the interval, so importance never increases from parent to
 * child. An interval whose largest distance is 0 is decided at once with importance 0. Rounds are
 * GPU gated: after each round a one-thread gate stops the indirect dispatch when no row remains
 * undecided (converged) or `maximumIterations` ran. Rows left undecided by the cap get the importance
 * of their interval's split, which is at least their true importance.
 *
 * **Selection.** A row is kept when `importance > tolerance` (line endpoints always). The
 * tolerance lives in `parameters`, so changing it re-runs only the mask, a stable
 * `GPUCompaction`, a per-line binary search, and the bounded publish, never the rounds.
 *
 * **Exactness.** The split tree does not depend on the tolerance, and a row survives classic
 * recursive Douglas-Peucker at tolerance `e` (split when `dmax > e`, first maximum on ties) exactly
 * when every split on its ancestor chain, including its own, has distance `> e`, which is
 * `importance > e` with the min clamp. So when `converged` is 1 the kept set equals recursive
 * Douglas-Peucker at every tolerance under the same metric; when it is 0 the kept set is a
 * superset. Distances use only correctly rounded `+`, `-`, and `*` to decide results: division and
 * square root are seeded by WGSL and then corrected to the largest f32 whose rounded product does
 * not exceed the operand, so a CPU oracle with `Math.fround` reproduces importance bit for bit.
 * Inputs must be finite and avoid the subnormal range (GPUs may flush subnormals).
 *
 * **Visvalingam-Whyatt** (`method: 'visvalingam'`). The importance of a vertex is its effective
 * area: the area of the triangle with its surviving neighbors when it is removed, raised to the
 * largest effective area of any neighbor removed before it (so importance is monotone along removal
 * order). That is the order of the `geo` crate and the Python `simplification` package, which
 * remove the smallest triangle first and keep a vertex when its area exceeds the tolerance.
 * Instead of a heap, every round removes each surviving vertex whose `(area, row)` is smaller than
 * that of all surviving vertices within `neighborhoodRadius` steps on both sides: an independent
 * set that is unlinked in parallel. Areas use only correctly rounded f32 `+`, `-`, `*`, so a CPU
 * mirror reproduces importance bit for bit. Rounds needed grow slowly with the line length (about
 * 55 for 3000 random-walk vertices at radius 3). When `converged` is 0, undecided rows get
 * importance `+Infinity`, so the kept set is a superset.
 *
 * **Visvalingam exactness.** Heap order is inherently sequential: removing a small triangle two or
 * more steps away can lower a neighbor's area below a vertex that a local test already removed, so
 * the parallel result equals the sequential one only up to that effect. Measured against
 * `simplification.cutil.simplify_coords_vw` on random walks of 3000 vertices at five tolerances,
 * the kept set differs in 3.1 percent of vertices at `neighborhoodRadius` 1, 0.65 percent at 2 and
 * 0.13 percent at 3 (the default); on smooth noisy curves 0.9, 0.09 and 0 percent. Small lines
 * (up to 120 vertices) match exactly at radius 3. Treat the result as an approximation of
 * Visvalingam-Whyatt, not as bit-identical to the heap.
 *
 * Non-goals: topology preservation (simplified lines may self-intersect or cross each other),
 * topology-preserving Visvalingam (`simplify_coords_vwp`), geodesic Visvalingam areas (project first), ellipsoidal distances, double-single or Int64 timestamps for
 * `'time-ratio'`, and chunked inputs.
 */
export class GPULineSimplification implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineSimplificationProps;
  /** Resolved spatial context: native planar, or longitude-latitude with the great-circle metric. */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in meters of a great-circle context, `undefined` for planar. */
  readonly sphereRadius: number | undefined;
  /** Simplification algorithm. */
  readonly method: GPULineSimplificationMethod;
  /** Visvalingam neighborhood radius. */
  readonly neighborhoodRadius: number;
  /** Distance metric. */
  readonly metric: GPULineSimplificationMetric;
  /** Compile-time iteration cap. */
  readonly maximumIterations: number;
  /** Largest open interval solved whole by one lane (Douglas-Peucker). */
  readonly finishSpanLimit: number;
  /** Whether the importance rounds are part of this contributor. */
  readonly computeImportance: boolean;

  constructor(props: GPULineSimplificationProps) {
    this.id = props.id ?? 'line-simplification';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
    );
    this.sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    this.props = props;
    this.metric = props.metric ?? 'segment';
    this.method = props.method ?? 'douglas-peucker';
    this.neighborhoodRadius =
      props.neighborhoodRadius ?? GPU_LINE_SIMPLIFICATION_DEFAULT_NEIGHBORHOOD_RADIUS;
    this.maximumIterations =
      props.maximumIterations ??
      (this.method === 'visvalingam'
        ? GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS
        : GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS);
    this.computeImportance = props.computeImportance ?? true;
    this.finishSpanLimit =
      props.finishSpanLimit ?? GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT;
    validateCapacity(
      this.id,
      'finishSpanLimit',
      this.finishSpanLimit,
      0,
      GPU_LINE_SIMPLIFICATION_MAXIMUM_FINISH_SPAN_LIMIT
    );
    const {id, metric, maximumIterations, method, neighborhoodRadius} = this;
    const {selection, status} = getOutputParts(props);
    if (method !== 'douglas-peucker' && method !== 'visvalingam') {
      throw new Error(`${id} method must be 'douglas-peucker' or 'visvalingam'`);
    }
    if (method === 'visvalingam') {
      if (this.sphereRadius !== undefined) {
        throw new Error(
          `${id} method 'visvalingam' is planar only; project positions to a planar frame first`
        );
      }
      if (metric !== 'segment') {
        throw new Error(`${id} method 'visvalingam' does not support metric '${metric}'`);
      }
      validateCapacity(
        id,
        'neighborhoodRadius',
        neighborhoodRadius,
        1,
        GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS
      );
    }
    if (metric !== 'segment' && metric !== 'time-ratio') {
      throw new Error(`${id} metric must be 'segment' or 'time-ratio'`);
    }
    validateCapacity(
      id,
      'maximumIterations',
      maximumIterations,
      1,
      GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS
    );
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    const lineCount = props.lineOffsets.length - 1;
    validatePackedView(props.importance, ['float32'], `${id} importance`);
    if (props.importance.length !== rowCount) {
      throw new Error(`${id} importance length must equal positions length`);
    }
    if (metric === 'time-ratio' && this.computeImportance) {
      if (!props.timestamps) {
        throw new Error(`${id} metric 'time-ratio' requires timestamps`);
      }
      validatePackedView(props.timestamps, ['float32'], `${id} timestamps`);
      if (props.timestamps.length !== rowCount) {
        throw new Error(`${id} timestamps length must equal positions length`);
      }
    }
    if ((status?.converged || status?.iterationCount) && !this.computeImportance) {
      throw new Error(`${id} output.status.converged and iterationCount require computeImportance`);
    }
    for (const [name, scalar] of [
      ['converged', status?.converged],
      ['iterationCount', status?.iterationCount]
    ] as const) {
      if (scalar) {
        validatePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.status.${name} needs one row`);
        }
      }
    }
    if (selection) {
      if (!props.parameters) {
        throw new Error(`${id} parameters are required to select rows`);
      }
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH} float32 values`
        );
      }
      validateCompactOutput(id, selection);
      if (selection.outputMask) {
        validatePackedView(selection.outputMask, ['uint32'], `${id} output.outputMask`);
        if (selection.outputMask.length !== rowCount) {
          throw new Error(`${id} output.outputMask length must equal positions length`);
        }
      }
      for (const [name, view] of [
        ['lineCounts', selection.lineCounts],
        ['lineStarts', selection.lineStarts]
      ] as const) {
        if (view) {
          validatePackedView(view, ['uint32'], `${id} output.${name}`);
          if (view.length !== lineCount) {
            throw new Error(`${id} output.${name} length must equal the line count`);
          }
        }
      }
    }
    if (!this.computeImportance && !selection) {
      throw new Error(`${id} requires computeImportance or selected output`);
    }
    const inputs = [props.positions, props.lineOffsets, props.timestamps, props.parameters];
    if (!this.computeImportance) {
      inputs.push(props.importance);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        this.computeImportance ? props.importance : undefined,
        status?.converged,
        status?.iterationCount,
        selection?.ids,
        selection?.status.count,
        selection?.status.overflow,
        selection?.status.requiredCount,
        selection?.outputMask,
        selection?.lineCounts,
        selection?.lineStarts
      ],
      inputs
    );
  }

  /**
   * Returns, when computing importance, the init node, `maximumIterations` gated rounds of reset,
   * distance, argmax, split, and gate nodes, an unresolved-row node and an optional status node;
   * then, with `selection`, the mask, compaction, publish, and optional per-line range nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {selection, status} = getOutputParts(props);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.timestamps,
      props.importance,
      props.parameters,
      status?.converged,
      status?.iterationCount,
      selection?.ids,
      selection?.status.count,
      selection?.status.overflow,
      selection?.status.requiredCount,
      selection?.outputMask,
      selection?.lineCounts,
      selection?.lineStarts
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (this.computeImportance) {
      nodes.push(...this._getImportanceNodes(graph));
    }
    if (selection && props.parameters) {
      nodes.push(...this._getSelectionNodes(graph, selection, props.parameters));
    }
    return nodes;
  }

  private _getVisvalingamNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id, maximumIterations} = this;
    const rowCount = props.positions.length;
    const scratch: VisvalingamScratch = {
      previousRows: createTransientView(graph, `${id}-previous-rows`, 'uint32', rowCount),
      nextRows: createTransientView(graph, `${id}-next-rows`, 'uint32', rowCount),
      areaKeys: createTransientView(graph, `${id}-area-keys`, 'uint32', rowCount),
      floorKeys: createTransientView(graph, `${id}-floor-keys`, 'uint32', rowCount),
      pendingFlags: createTransientView(graph, `${id}-pending-flags`, 'uint32', rowCount)
    };
    const state = createRasterIterationState(graph, `${id}-rounds`, OPERATION, rowCount, id);
    const nodes: GPUCommandNode<Parameters>[] = [
      createRasterIterationResetNode<Parameters>(graph, {
        id: `${id}-rounds-reset`,
        operation: OPERATION,
        state
      }),
      createVisvalingamInitNode<Parameters>(graph, {
        id: `${id}-init`,
        lineOffsets: props.lineOffsets,
        scratch,
        importance: props.importance
      })
    ];
    for (let round = 0; round < maximumIterations; round++) {
      const roundId = `${id}-round-${round}`;
      nodes.push(
        ...createVisvalingamRoundNodes<Parameters>(graph, {
          id: roundId,
          neighborhoodRadius: this.neighborhoodRadius,
          positions: props.positions,
          scratch,
          importance: props.importance,
          status: state.status,
          gate: getRasterIterationCondition<Parameters>(state, roundId)
        }),
        createRasterIterationGateNode<Parameters>(graph, {
          id: `${roundId}-gate`,
          operation: OPERATION,
          state,
          maxIterations: maximumIterations
        })
      );
    }
    nodes.push(
      createVisvalingamUnresolvedNode<Parameters>(graph, {
        id: `${id}-unresolved`,
        scratch,
        importance: props.importance
      })
    );
    const status = props.output?.status;
    if (status?.converged || status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          state,
          converged: status.converged,
          iterationCount: status.iterationCount
        })
      );
    }
    return nodes;
  }

  private _getImportanceNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    if (this.method === 'visvalingam') {
      return this._getVisvalingamNodes(graph);
    }
    const {props, id, maximumIterations} = this;
    const rowCount = props.positions.length;
    const scratch: LineImportanceScratch = {
      leftAnchors: createTransientView(graph, `${id}-left-anchors`, 'uint32', rowCount),
      rightAnchors: createTransientView(graph, `${id}-right-anchors`, 'uint32', rowCount),
      distanceKeys: createTransientView(graph, `${id}-distance-keys`, 'uint32', rowCount),
      bestKeys: createTransientView(graph, `${id}-best-keys`, 'uint32', rowCount),
      bestRows: createTransientView(graph, `${id}-best-rows`, 'uint32', rowCount)
    };
    const state = createRasterIterationState(graph, `${id}-rounds`, OPERATION, rowCount, id);
    const nodes: GPUCommandNode<Parameters>[] = [
      createRasterIterationResetNode<Parameters>(graph, {
        id: `${id}-rounds-reset`,
        operation: OPERATION,
        state
      }),
      createLineImportanceInitNode<Parameters>(graph, {
        id: `${id}-init`,
        lineOffsets: props.lineOffsets,
        scratch,
        importance: props.importance
      })
    ];
    for (let round = 0; round < maximumIterations; round++) {
      const roundId = `${id}-round-${round}`;
      nodes.push(
        ...createLineImportanceRoundNodes<Parameters>(graph, {
          id: roundId,
          metric: this.metric,
          positions: props.positions,
          timestamps: props.timestamps,
          scratch,
          importance: props.importance,
          status: state.status,
          gate: getRasterIterationCondition<Parameters>(state, roundId),
          finishSpanLimit: this.finishSpanLimit,
          sphereRadius: this.sphereRadius
        }),
        createRasterIterationGateNode<Parameters>(graph, {
          id: `${roundId}-gate`,
          operation: OPERATION,
          state,
          maxIterations: maximumIterations
        })
      );
    }
    nodes.push(
      createLineImportanceUnresolvedNode<Parameters>(graph, {
        id: `${id}-unresolved`,
        scratch,
        importance: props.importance
      })
    );
    const status = props.output?.status;
    if (status?.converged || status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          state,
          converged: status.converged,
          iterationCount: status.iterationCount
        })
      );
    }
    return nodes;
  }

  private _getSelectionNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    selection: GPULineSimplificationSelectionOutput,
    parameters: GraphDataView<'float32'>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rowCount = props.positions.length;
    const outputMask =
      selection.outputMask ?? createTransientView(graph, `${id}-keep-mask`, 'uint32', rowCount);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const compactRows = createTransientView(graph, `${id}-compact-rows`, 'uint32', rowCount);
    const requiredCount = createTransientView(graph, `${id}-kept-total`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      createLineKeepMaskNode<Parameters>(graph, {
        id: `${id}-keep-mask`,
        lineOffsets: props.lineOffsets,
        importance: props.importance,
        parameters,
        outputMask,
        rowIds
      }),
      ...new GPUCompaction({
        id: `${id}-compaction`,
        input: rowIds,
        flags: outputMask,
        output: compactRows,
        count: requiredCount
      }).getCommandNodes(graph),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount,
        compactIds: compactRows,
        output: selection
      })
    ];
    if (selection.lineCounts || selection.lineStarts) {
      nodes.push(
        createLineKeptRangesNode<Parameters>(graph, {
          id: `${id}-line-ranges`,
          lineOffsets: props.lineOffsets,
          compactRows,
          requiredCount,
          lineCounts: selection.lineCounts,
          lineStarts: selection.lineStarts
        })
      );
    }
    return nodes;
  }
}
