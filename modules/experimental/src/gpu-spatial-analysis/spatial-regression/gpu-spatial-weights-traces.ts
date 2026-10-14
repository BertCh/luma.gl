// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createProbeBatches,
  getAlignedSubView,
  padToAlignedElementCount
} from './spatial-maximum-likelihood-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  GPUSpatialWeightsTranspose,
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/index';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT,
  GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_LENGTH,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER,
  GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH,
  getSpatialWeightsTracesProbeWGSL,
  GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';
import type {GPUStatusScalar} from '../../analysis-core/contracts/status';

const OPERATION = 'GPUSpatialWeightsTraces';
const TILE_WORKGROUP_SIZE = 64;
const MINIMUM_DEFAULT_TILE_ROWS = 32;
const DEFAULT_TILE_COUNT = 1024;

/** Caller-owned outputs of {@link GPUSpatialWeightsTraces}. */
export type GPUSpatialWeightsTracesOutput = {
  /** At least `maximumPower` float32 values: `tr((W / s)^q)` at index `q - 1`, `s = infinityNorm`. */
  powerTraces: GraphDataView<'float32'>;
  /** At least `maximumPower` float32 values: `1' (W / s)^q 1` at index `q - 1` (always exact). */
  powerSums: GraphDataView<'float32'>;
  /**
   * At least 4 float32 values `[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]` of the unscaled
   * `W`; required exactly
   * when `includeGramTraces` is true.
   */
  gramTraces?: GraphDataView<'float32'>;
  /** Optional status words. */
  status?: {
    /** One uint32 scalar: 1 when the traces are stochastic estimates, 0 when exact. */
    approximation?: GPUStatusScalar;
  };
  /**
   * At least one float32 value: `s = ||W||_inf`, the largest absolute row sum (1 when `W` has no
   * links). `powerTraces` and `powerSums` are the traces and sums of `(W / s)^q`; consumers need
   * `s` to rescale the series parameter (`rho * s`). Required by `GPUSpatialLagImpacts` and the
   * maximum-likelihood estimators.
   */
  infinityNorm?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialWeightsTraces}.
 *
 * Topology (needs a new graph): the weights views, `maximumPower`, `mode`, `probeCount`,
 * `includeGramTraces`, `tileRowCount` and which outputs are present. Per-frame: the weights'
 * contents and of `parameters` (the probe seed).
 */
export type GPUSpatialWeightsTracesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-traces'`. */
  id?: string;
  /**
   * Square weights `W`. Any sparsity pattern and values work: no symmetry, no row standardization,
   * and self-loops (`w_ii != 0`) are allowed. Slots with a neighbor id `>= rows` are dropped.
   */
  weights: GPUSpatialWeights;
  /** Highest power `Q`, compile-time, an integer in `[1, 256]`. */
  maximumPower: number;
  /**
   * `'exact'` runs one unit-vector probe per row (exact up to float32 rounding, `O(rows)` chains)
   * and `'stochastic'` runs Hutchinson estimation with `probeCount` Rademacher probes. Defaults to
   * `'exact'` when `rows <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT` (4096) and to
   * `'stochastic'` otherwise. An explicit `'exact'` is accepted up to 16384 rows.
   */
  mode?: GPUSpatialWeightsTracesMode;
  /** Probes of the stochastic mode. Defaults to 64. */
  probeCount?: number;
  /**
   * Optional per-frame uint32 view of at least `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` words
   * written with `getGPUSpatialWeightsTracesParameterValues({seed})`: the Philox seed of the
   * stochastic probes, read on the GPU. Omitted means seed 0.
   */
  parameters?: GraphDataView<'uint32'>;
  /** Also compute `output.gramTraces`. Defaults to whether `output.gramTraces` is present. */
  includeGramTraces?: boolean;
  /**
   * Rows reduced by one invocation of the dot-product passes. Defaults to
   * `max(32, ceil(rows / 1024))`.
   */
  tileRowCount?: number;
  /**
   * Upper bound on the probes held by one batch, for low-memory devices and tests. The actual
   * batch is also limited by the 64 MiB scratch budget, so this can only shrink it. Results are
   * identical for every batch size.
   */
  probeBatchSize?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialWeightsTracesOutput;
};

/**
 * Validates an optional per-frame trace parameters view (`[seed]`, see
 * `getGPUSpatialWeightsTracesParameterValues`).
 * @internal
 */
export function validateGPUSpatialWeightsTracesParameters(
  id: string,
  name: string,
  view: GraphDataView<'uint32'> | undefined
): void {
  if (!view) {
    return;
  }
  validatePackedView(view, ['uint32'], `${id} ${name}`);
  if (view.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH) {
    throw new Error(
      `${id} ${name} must hold ${GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH} uint32`
    );
  }
}

/**
 * Powers of a spatial weights matrix: the traces `tr(W^q)` and the sums `1' W^q 1` for
 * `q = 1..Q`, the inputs of spatial-lag log-determinants and of direct/indirect/total impacts
 * (see {@link getSpatialLogDeterminantWGSL} and `GPUSpatialLagImpacts`).
 *
 * - Scaling: every power chain runs with `W / s`, `s = ||W||_inf` (the largest absolute row sum,
 *   written to `output.infinityNorm`; 1 for an empty `W`). The scaled matrix has spectral radius
 *   at most 1, so `|tr((W/s)^q)| <= n` and the float32 powers never overflow, whatever the scale
 *   of `W`. Series therefore run in the scaled parameter `t = rho * s`:
 *   `ln|I - rho W| = -sum_q t^q tr((W/s)^q) / q`. Row-standardized weights have `s = 1`.
 * - `powerSums` is exact: one chain `u_q = (W/s) u_{q-1}` from the ones vector.
 * - `powerTraces[q-1] = tr((W/s)^q) = E[z' (W/s)^q z]`. Exact mode uses the unit probes `e_i`, so each
 *   chain `v_q = W v_{q-1}` from `e_i` contributes its diagonal entry `v_q[i]`; there is no
 *   sampling error, only float32 rounding. Stochastic mode (Hutchinson) uses `probeCount`
 *   Rademacher probes from Philox (counter `(probe, row / 4, 0, 'PROB')`, key `(seed, 'TRAC')`, so a
 *   seeded run replays exactly and matches `getSpatialWeightsTracesOnCPU`) and averages
 *   `z' W^q z`. Its standard error for `tr(A)` is at most `sqrt(2 / probeCount) * ||A||_F`.
 *   `approximation` is 1 for stochastic runs.
 * - `gramTraces` (optional, unscaled `W`) are `tr(W'W) = sum w_ij^2` and `sum_i c_i^2` with `c_i = sum_l w_li^2`
 *   (exact, the latter through `GPUSpatialWeightsTranspose`), and `tr(W'W W) = E[(Wz)'(WWz)]`,
 *   `tr((W'W)^2) = E[||W'(W z)||^2]` over the same probes (exact in exact mode). They are the
 *   moments of the Kelejian-Prucha / Anselin-Kelejian variance formulas.
 *
 * Probes run in batches sized so the two ping-pong vectors stay within 64 MiB
 * (`max(1, floor(8M / rows))` probes per batch). Every dot product is a fixed-order tile sum
 * followed by a Kahan merge, so results are deterministic and no float atomics are used. Nothing
 * is read back. The shaders do not depend on the power, the batch or the probe index (rule 12):
 * per-power and per-batch tables are 64-element aligned sub-views, and a uint32 batch table holds
 * each batch's first probe and probe count, so the pipeline count is independent of
 * `maximumPower` and of the number of batches (the short last batch runs the full lane count
 * behind a guard). Cost: `ceil(probes / batch) * (Q + 2)` dispatches of `batch * rows * nnz / rows`
 * work; see `GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT` for the exact-mode default limit.
 *
 */
export class GPUSpatialWeightsTraces implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsTracesProps;
  /** Number of rows of `W`. */
  readonly rowCount: number;
  /** Resolved mode. */
  readonly mode: GPUSpatialWeightsTracesMode;
  /** Total probes: `rowCount` in exact mode, `probeCount` in stochastic mode. */
  readonly probeTotal: number;
  /** Probes held by one batch. */
  readonly batchProbeCount: number;
  /** Rows reduced by one dot-product invocation. */
  readonly tileRowCount: number;
  /** Dot-product tiles per vector. */
  readonly tileCount: number;
  /** Whether the Gram traces are computed. */
  readonly includeGramTraces: boolean;

  constructor(props: GPUSpatialWeightsTracesProps) {
    const id = props.id ?? 'spatial-weights-traces';
    this.id = id;
    this.props = props;
    const {output, maximumPower} = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    validateCapacity(id, 'maximumPower', maximumPower, 1, GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER);
    if (props.mode !== undefined && props.mode !== 'exact' && props.mode !== 'stochastic') {
      throw new Error(`${id} mode must be 'exact' or 'stochastic'`);
    }
    const mode =
      props.mode ??
      (rows <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic');
    if (mode === 'exact' && rows > GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT) {
      throw new Error(
        `${id} exact mode supports at most ${GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT} rows; use mode 'stochastic'`
      );
    }
    const probeCount = props.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT;
    validateCapacity(id, 'probeCount', probeCount);
    validateGPUSpatialWeightsTracesParameters(id, 'parameters', props.parameters);
    if (props.tileRowCount !== undefined) {
      validateCapacity(id, 'tileRowCount', props.tileRowCount);
    }
    validatePackedView(output.powerTraces, ['float32'], `${id} output.powerTraces`);
    if (output.powerTraces.length < maximumPower) {
      throw new Error(`${id} output.powerTraces must hold maximumPower float32 values`);
    }
    validatePackedView(output.powerSums, ['float32'], `${id} output.powerSums`);
    if (output.powerSums.length < maximumPower) {
      throw new Error(`${id} output.powerSums must hold maximumPower float32 values`);
    }
    const includeGram = props.includeGramTraces ?? output.gramTraces !== undefined;
    if (includeGram !== (output.gramTraces !== undefined)) {
      throw new Error(`${id} output.gramTraces is required exactly when includeGramTraces is true`);
    }
    if (output.gramTraces) {
      validatePackedView(output.gramTraces, ['float32'], `${id} output.gramTraces`);
      if (output.gramTraces.length < GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_LENGTH) {
        throw new Error(`${id} output.gramTraces must hold 4 float32 values`);
      }
    }
    if (output.infinityNorm) {
      validatePackedView(output.infinityNorm, ['float32'], `${id} output.infinityNorm`);
      if (output.infinityNorm.length < 1) {
        throw new Error(`${id} output.infinityNorm must hold one float32 value`);
      }
    }
    if (output.status?.approximation) {
      validatePackedView(
        output.status.approximation,
        ['uint32'],
        `${id} output.status.approximation`
      );
      if (output.status.approximation.length < 1) {
        throw new Error(`${id} output.status.approximation must hold one uint32 row`);
      }
    }
    this.rowCount = rows;
    this.mode = mode;
    this.includeGramTraces = includeGram;
    this.probeTotal = mode === 'exact' ? rows : probeCount;
    if (props.probeBatchSize !== undefined) {
      validateCapacity(id, 'probeBatchSize', props.probeBatchSize);
    }
    this.batchProbeCount = Math.min(
      this.probeTotal,
      props.probeBatchSize ?? Number.POSITIVE_INFINITY,
      Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / rows))
    );
    this.tileRowCount =
      props.tileRowCount ??
      Math.max(MINIMUM_DEFAULT_TILE_ROWS, Math.ceil(rows / DEFAULT_TILE_COUNT));
    this.tileCount = Math.ceil(rows / this.tileRowCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.powerTraces,
        output.powerSums,
        output.gramTraces,
        output.status?.approximation,
        output.infinityNorm
      ],
      [props.weights.offsets, props.weights.neighbors, props.weights.weights, props.parameters]
    );
  }

  /**
   * Returns the command nodes in dependency order: the ones-vector chain for `powerSums`, the
   * Gram transpose and exact moments (when requested), the batch table, the probe batches (per
   * step one matrix-vector tile pass and, per target row of the dot table, one per-probe Kahan
   * merge), and the final probe merge that writes `powerTraces` and `gramTraces`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, probeTotal, batchProbeCount, tileRowCount, tileCount} = this;
    const {weights, output, maximumPower} = props;
    const stochastic = this.mode === 'stochastic';
    const includeGram = this.includeGramTraces;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.powerTraces,
      output.powerSums,
      output.gramTraces,
      output.status?.approximation,
      output.infinityNorm,
      props.parameters
    ]);
    // Steps run to 2 when the Gram traces need `W W z` even if `maximumPower` is 1.
    const stepCount = includeGram ? Math.max(maximumPower, 2) : maximumPower;
    const powerRowCount = stepCount;
    const gramPairRow = powerRowCount;
    const gramSquareRow = powerRowCount + 1;
    const dotRowCount = powerRowCount + (includeGram ? 2 : 0);
    const batches = createProbeBatches(
      graph,
      id,
      OPERATION,
      probeTotal,
      batchProbeCount,
      props.parameters
    );
    const sumsStride = padToAlignedElementCount(tileCount);
    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const constants = `const ROWS: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const PROBE_TOTAL: u32 = ${probeTotal}u;`;
    const f32 = (name: string, view: GraphDataView, access: 'read' | 'read_write') =>
      ({name, view, type: 'f32', access}) as WGSLKernelBinding;
    const u32 = (name: string, view: GraphDataView, access: 'read' | 'read_write') =>
      ({name, view, type: 'u32', access}) as WGSLKernelBinding;
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      declarations: string,
      body: string,
      constantValues?: Record<string, number>,
      workgroupSize?: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name.replace(/-\d+(-\d+)?$/, ''),
        bindings,
        invocationCount,
        declarations,
        body,
        ...(constantValues ? {constants: constantValues} : {}),
        ...(workgroupSize ? {workgroupSize} : {})
      });
    const csrBindings = (
      prefix: string,
      csr: {offsets: GraphDataView; neighbors: GraphDataView; weights: GraphDataView}
    ): WGSLKernelBinding[] => [
      u32(prefix ? `${prefix}Offsets` : 'offsets', csr.offsets, 'read'),
      u32(prefix ? `${prefix}Neighbors` : 'neighbors', csr.neighbors, 'read'),
      f32(prefix ? `${prefix}Weights` : 'weights', csr.weights, 'read')
    ];
    const kahanLoop = (
      source: string,
      indexExpression: string,
      stride: string,
      count: string
    ) => `var sum = 0.0;
  var compensation = 0.0;
  for (var item = 0u; item < ${count}; item++) {
    let term = ${source}[${source}Offset + ${indexExpression} + item * ${stride}] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const weightBindings = csrBindings('', weights);

    // ---- s = ||W||_inf and the scaled weights W / s used by every power chain. ----
    const infinityNorm = output.infinityNorm ?? transient('infinity-norm', 1);
    const normPartial = transient('norm-partial', tileCount);
    const scaledWeights = transient('scaled-weights', weights.weights.length);
    nodes.push(
      kernel(
        'norm-tiles',
        [...weightBindings, f32('normPartial', normPartial, 'read_write')],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      sum += abs(weights[weightsOffset + slot]);
    }
    largest = max(largest, sum);
  }
  normPartial[normPartialOffset + index] = largest;`,
        undefined,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'norm-merge',
        [f32('normPartial', normPartial, 'read'), f32('infinityNorm', infinityNorm, 'read_write')],
        1,
        constants,
        `var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, normPartial[normPartialOffset + tile]);
  }
  // An empty or non-finite W scales by 1 (non-finite weights then propagate through the chains).
  infinityNorm[infinityNormOffset] = select(1.0, largest, largest > 0.0 && largest < 3.0e38);`
      ),
      kernel(
        'scale-weights',
        [
          f32('weights', weights.weights, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('scaledWeights', scaledWeights, 'read_write')
        ],
        weights.weights.length,
        '',
        `scaledWeights[scaledWeightsOffset + index] = weights[weightsOffset + index] / infinityNorm[infinityNormOffset];`
      )
    );
    const scaledBindings = csrBindings('', {
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      weights: scaledWeights
    });

    // ---- powerSums: one chain from the ones vector, tile partial sums then a Kahan merge. ----
    // One 64-element aligned row per power, so each step binds its own row as a sub-view.
    const sumsPartial = transient('sums-partial', maximumPower * sumsStride);
    const onesVectors = [transient('ones-a', rowCount), transient('ones-b', rowCount)];
    for (let power = 1; power <= maximumPower; power++) {
      const source = onesVectors[(power - 2 + 2) % 2];
      const target = onesVectors[(power - 1) % 2];
      const first = power === 1;
      nodes.push(
        kernel(
          first ? 'sums-step-first' : `sums-step-${power}`,
          [
            ...scaledBindings,
            ...(first ? [] : [f32('vectorIn', source, 'read')]),
            f32('vectorOut', target, 'read_write'),
            f32(
              'sumsPartial',
              getAlignedSubView(graph, sumsPartial, (power - 1) * sumsStride, tileCount),
              'read_write'
            )
          ],
          tileCount,
          constants,
          `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var tileSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${first ? '1.0' : 'vectorIn[vectorInOffset + neighbor]'};
    }
    vectorOut[vectorOutOffset + row] = sum;
    tileSum += sum;
  }
  sumsPartial[sumsPartialOffset + index] = tileSum;`,
          undefined,
          TILE_WORKGROUP_SIZE
        )
      );
    }
    nodes.push(
      kernel(
        'sums-merge',
        [f32('sumsPartial', sumsPartial, 'read'), f32('powerSums', output.powerSums, 'read_write')],
        maximumPower,
        `${constants}
const SUMS_STRIDE: u32 = ${sumsStride}u;`,
        `${kahanLoop('sumsPartial', 'index * SUMS_STRIDE', '1u', 'TILE_COUNT')}
  powerSums[powerSumsOffset + index] = sum;`
      )
    );

    // ---- Gram moments that need no probes: tr(W'W) and sum_i c_i^2 (column sums of squares). ----
    let transposed: GPUSpatialWeights | undefined;
    const gramPartial = includeGram ? transient('gram-partial', tileCount * 2) : undefined;
    if (includeGram && gramPartial) {
      transposed = {
        offsets: createTransientView(graph, `${id}-transpose-offsets`, 'uint32', rowCount + 1),
        neighbors: createTransientView(
          graph,
          `${id}-transpose-neighbors`,
          'uint32',
          weights.neighbors.length
        ),
        weights: transient('transpose-weights', weights.neighbors.length)
      };
      nodes.push(
        ...new GPUSpatialWeightsTranspose({
          id: `${id}-transpose`,
          weights: {
            offsets: weights.offsets,
            neighbors: weights.neighbors,
            weights: weights.weights
          },
          output: transposed
        }).getCommandNodes(graph)
      );
      nodes.push(
        kernel(
          'gram-moments-tiles',
          [
            ...weightBindings,
            ...csrBindings('transposed', transposed),
            f32('gramPartial', gramPartial, 'read_write')
          ],
          tileCount,
          constants,
          `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  var columnSquareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      squareSum += weight * weight;
    }
    var columnSquares = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      if (transposedNeighbors[transposedNeighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = transposedWeights[transposedWeightsOffset + slot];
      columnSquares += weight * weight;
    }
    columnSquareSum += columnSquares * columnSquares;
  }
  gramPartial[gramPartialOffset + index * 2u] = squareSum;
  gramPartial[gramPartialOffset + index * 2u + 1u] = columnSquareSum;`,
          undefined,
          TILE_WORKGROUP_SIZE
        )
      );
    }

    // ---- Probe batches. ----
    // The dot table has one row per dot product; a row holds one 64-element aligned block per batch.
    const dotsRowStride = batches.batchCount * batches.blockStride;
    const dots = transient('dots', dotRowCount * dotsRowStride);
    const getDotsBlock = (row: number, batchIndex: number) =>
      batches.getBlock(
        getAlignedSubView(graph, dots, row * dotsRowStride, dotsRowStride),
        batchIndex
      );
    // Tile partials of the power dots and of the Gram dots (the latter hold the pair dot in step 2
    // and the squared norm of W' (W z) from the gram-square pass).
    const partialPower = stochastic
      ? transient('partial-power', batchProbeCount * tileCount)
      : undefined;
    const partialGram = includeGram
      ? transient('partial-gram', batchProbeCount * tileCount)
      : undefined;
    const scratch = [
      transient('scratch-a', batchProbeCount * rowCount),
      transient('scratch-b', batchProbeCount * rowCount)
    ];
    nodes.push(batches.node);
    const probeDeclarations = getSpatialWeightsTracesProbeWGSL(stochastic);
    // Reduces the tile partials of every probe of a batch into one aligned block of one dot row.
    const mergeNode = (
      name: string,
      batchIndex: number,
      partial: GraphDataView,
      dotRow: number
    ): GPUCommandNode<Parameters> =>
      kernel(
        name,
        [
          f32('partial', partial, 'read'),
          f32('dots', getDotsBlock(dotRow, batchIndex), 'read_write'),
          u32('batchInformation', batches.getBatchInformation(batchIndex), 'read')
        ],
        batchProbeCount,
        constants,
        `if (index >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  ${kahanLoop('partial', 'index * TILE_COUNT', '1u', 'TILE_COUNT')}
  dots[dotsOffset + index] = sum;`,
        undefined,
        TILE_WORKGROUP_SIZE
      );
    for (let batchIndex = 0; batchIndex < batches.batchCount; batchIndex++) {
      const batchInformation = u32(
        'batchInformation',
        batches.getBatchInformation(batchIndex),
        'read'
      );
      for (let step = 1; step <= stepCount; step++) {
        const first = step === 1;
        const gramStep = includeGram && step === 2;
        const powerDot = stochastic && step <= maximumPower;
        const writesPartial = powerDot || gramStep;
        const target = scratch[(step - 1) % 2];
        const source = scratch[(step - 2 + 2) % 2];
        const exactDiagonal = !stochastic && step <= maximumPower;
        const variant = `${first ? 'first' : 'next'}${gramStep ? '-gram' : ''}${
          writesPartial ? '-dot' : ''
        }${exactDiagonal ? '-diagonal' : ''}`;
        const bindings: WGSLKernelBinding[] = [
          ...scaledBindings,
          ...(first ? [] : [f32('vectorIn', source, 'read')]),
          f32('vectorOut', target, 'read_write'),
          ...(powerDot ? [f32('partialPower', partialPower as GraphDataView, 'read_write')] : []),
          ...(gramStep ? [f32('partialGram', partialGram as GraphDataView, 'read_write')] : []),
          ...(exactDiagonal ? [f32('dots', getDotsBlock(step - 1, batchIndex), 'read_write')] : []),
          batchInformation
        ];
        const value = first
          ? 'getProbeValue(probeSeed, globalProbe, neighbor)'
          : 'vectorIn[vectorInOffset + base + neighbor]';
        nodes.push(
          kernel(
            `step-${variant}-${batchIndex}-${step}`,
            bindings,
            batchProbeCount * tileCount,
            `${constants}
${probeDeclarations}`,
            `let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var powerDot = 0.0;
  var gramDot = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${value};
    }
    vectorOut[vectorOutOffset + base + row] = sum;
${
  exactDiagonal
    ? `    if (row == globalProbe) {
      dots[dotsOffset + probe] = sum;
    }
`
    : ''
}${powerDot ? '    powerDot += getProbeValue(probeSeed, globalProbe, row) * sum;\n' : ''}${
  gramStep ? '    gramDot += vectorIn[vectorInOffset + base + row] * sum;\n' : ''
}  }
${powerDot ? '  partialPower[partialPowerOffset + probe * TILE_COUNT + tile] = powerDot;\n' : ''}${
  gramStep ? '  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = gramDot;\n' : ''
}  _ = powerDot;
  _ = gramDot;`,
            undefined,
            TILE_WORKGROUP_SIZE
          )
        );
        if (powerDot) {
          nodes.push(
            mergeNode(
              `probe-merge-${batchIndex}-${step}`,
              batchIndex,
              partialPower as GraphDataView,
              step - 1
            )
          );
        }
        if (gramStep) {
          nodes.push(
            mergeNode(
              `probe-merge-pair-${batchIndex}`,
              batchIndex,
              partialGram as GraphDataView,
              gramPairRow
            )
          );
        }
        if (gramStep && transposed) {
          // W' (W z) from the first-step vector, still held by the source buffer.
          nodes.push(
            kernel(
              `gram-square-${batchIndex}`,
              [
                ...csrBindings('transposed', transposed),
                f32('vectorIn', source, 'read'),
                f32('partialGram', partialGram as GraphDataView, 'read_write'),
                batchInformation
              ],
              batchProbeCount * tileCount,
              constants,
              `let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      let neighbor = transposedNeighbors[transposedNeighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += transposedWeights[transposedWeightsOffset + slot] * vectorIn[vectorInOffset + base + neighbor];
    }
    squareSum += sum * sum;
  }
  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = squareSum;`,
              undefined,
              TILE_WORKGROUP_SIZE
            )
          );
          nodes.push(
            mergeNode(
              `probe-merge-square-${batchIndex}`,
              batchIndex,
              partialGram as GraphDataView,
              gramSquareRow
            )
          );
        }
      }
    }

    // ---- Final merge over probes (and tiles of the exact Gram moments). ----
    const finalBindings: WGSLKernelBinding[] = [
      f32('dots', dots, 'read'),
      f32('infinityNorm', infinityNorm, 'read'),
      ...(gramPartial ? [f32('gramPartial', gramPartial, 'read')] : []),
      f32('powerTraces', output.powerTraces, 'read_write'),
      ...(output.gramTraces ? [f32('gramTraces', output.gramTraces, 'read_write')] : [])
    ];
    const scale = stochastic ? `1.0 / f32(PROBE_TOTAL)` : '1.0';
    nodes.push(
      kernel(
        'final-merge',
        finalBindings,
        dotRowCount + (includeGram ? 2 : 0),
        `${constants}
const POWER_COUNT: u32 = ${maximumPower}u;
const DOT_ROWS: u32 = ${dotRowCount}u;
const POWER_ROWS: u32 = ${powerRowCount}u;`,
        `if (index < DOT_ROWS) {
    ${batches.getSumWGSL('dots', 'sum', `index * ${dotsRowStride}u`)}
    let value = sum * ${scale};
    if (index < POWER_COUNT) {
      powerTraces[powerTracesOffset + index] = value;
    }${
      includeGram
        ? `
    if (index >= POWER_ROWS) {
      // The chains hold W / s: tr(W'W W) = s^3 E[v1' v2] and tr((W'W)^2) = s^2 E[|W' v1|^2].
      let norm = infinityNorm[infinityNormOffset];
      gramTraces[gramTracesOffset + 1u + index - POWER_ROWS] =
        value * select(norm * norm * norm, norm * norm, index > POWER_ROWS);
    }`
        : ''
    }
  }${
    includeGram
      ? `
  if (index >= DOT_ROWS) {
    let slot = index - DOT_ROWS;
    ${kahanLoop('gramPartial', 'slot', '2u', 'TILE_COUNT')}
    gramTraces[gramTracesOffset + select(0u, 3u, slot == 1u)] = sum;
  }`
      : ''
  }`
      )
    );
    if (output.status?.approximation) {
      nodes.push(
        kernel(
          'approximation',
          [u32('approximation', output.status.approximation, 'read_write')],
          1,
          '',
          `approximation[approximationOffset] = ${stochastic ? '1u' : '0u'};`
        )
      );
    }
    return nodes;
  }
}
