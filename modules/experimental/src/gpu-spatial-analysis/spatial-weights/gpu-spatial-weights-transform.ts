// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUSpatialWeightsTransform';

/** Kernel profile `K(z)` of a `'kernel'` transform, with `z = d / h`. */
export type GPUSpatialWeightsKernel =
  | 'gaussian'
  | 'triangular'
  | 'epanechnikov'
  | 'bisquare'
  | 'uniform';

/** Transform applied to the weights of a {@link GPUSpatialWeights}. */
export type GPUSpatialWeightsTransformOperation =
  | 'row'
  | 'binary'
  | 'kernel'
  | 'symmetrize'
  | 'double'
  | 'variance';

/**
 * Properties for {@link GPUSpatialWeightsTransform}.
 *
 * Compile-time: every prop. Per-frame: the contents of `weights` and `output`.
 */
export type GPUSpatialWeightsTransformProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-weights-transform'`. */
  id?: string;
  /**
   * - `'row'`: row standardization `w_ij / sum_j w_ij` (the PySAL `R` transform); zero-sum or
   *   non-finite-sum rows become all zero.
   * - `'binary'`: `w_ij > 0 ? 1 : 0` (PySAL `B`). Slots are kept, so a zero weight stays an explicit slot.
   * - `'kernel'`: replace each weight by `K(d_ij / h_i)` from `weights.distances` (see `kernel`, `bandwidth`).
   * - `'symmetrize'`: `(w_ij + w_ji) / 2`, with a missing reverse slot counting as 0. The sparsity pattern
   *   is not changed, so it needs a symmetric pattern (contiguity, lattice, distance band) to give
   *   a symmetric result. `output` must be a different view than `weights.weights`.
   * - `'double'`: double standardization `w_ij / S0` with `S0 = sum_ij w_ij` (the PySAL `D`
   *   transform, all weights sum to 1; see `doubleSum` for the sum-to-`n` variant). A zero or
   *   non-finite `S0` gives all-zero weights.
   * - `'variance'`: variance-stabilizing transform (PySAL `V`, Tiefelsdorf, Boots and Kozak
   *   1999): `s_ij = w_ij / sqrt(sum_j w_ij^2)`, `Q = sum_ij s_ij`, `w'_ij = s_ij * n / Q` with `n`
   *   the row count. Rows with no positive weight stay zero; a zero `Q` gives all-zero weights.
   *
   * `'double'` and `'variance'` need one global sum: per-row partials, a fixed-order `GPUReduction`
   * sum, then an apply pass, so results are deterministic.
   */
  operation: GPUSpatialWeightsTransformOperation;
  /** Weights to transform. Only `weights.weights` is written (to `output`). */
  weights: GPUSpatialWeights;
  /**
   * Destination for the transformed weights, aligned with `weights.neighbors`. Defaults to
   * `weights.weights` (in place). Only slots below `offsets[rows]` are written. Use a separate
   * view to keep the original weights, and a graph view with its own buffer when `weights`
   * is also read by other nodes in the same graph.
   */
  output?: GraphDataView<'float32'>;
  /**
   * Total of the `'double'` transform: `'one'` (default) scales by `1 / S0` so all weights sum to
   * 1, as the libpysal `W.transform = 'D'` formula; `'rows'` scales by `n / S0` so they sum to the
   * row count `n` (the Anselin double standardization, same mean row sum as `'row'`).
   */
  doubleSum?: 'one' | 'rows';
  /** Kernel profile for `'kernel'`. Defaults to `'triangular'`. */
  kernel?: GPUSpatialWeightsKernel;
  /**
   * Bandwidth `h` of `'kernel'`: a positive number (fixed, same for every row) or `'adaptive'`
   * (per row, `h_i` is the largest distance in row `i`, which for a kNN row is the distance of the
   * k-th neighbor, PySAL's adaptive kernel bandwidth). Defaults to `'adaptive'`.
   */
  bandwidth?: number | 'adaptive';
};

/**
 * Transforms spatial weights on the GPU: row standardization, binarization, distance kernels and
 * symmetrization of a {@link GPUSpatialWeights} (the PySAL `transform` and `Kernel` equivalents).
 *
 * Kernel formulas, with `z = d / h` (`z = 0` when `h <= 0`), match PySAL's `Kernel` and
 * `GPUNeighborSearch`:
 * - `gaussian`: `exp(-z^2 / 2) / sqrt(2 pi)`, no cutoff.
 * - `triangular`: `max(1 - z, 0)`.
 * - `epanechnikov` (PySAL `quadratic`): `3/4 max(1 - z^2, 0)`.
 * - `bisquare` (PySAL `quartic`): `15/16 max(1 - z^2, 0)^2`.
 * - `uniform`: `1/2` for `z <= 1`, else 0.
 *
 * Non-finite or negative results are written as 0, so the non-negativity invariant holds.
 * Row sums run in slot order and results are deterministic. `row`, `binary` and `kernel` work in
 * place or to `output`; chain two transforms (for example `kernel` then `row`) by adding two
 * instances in order.
 */
export class GPUSpatialWeightsTransform implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsTransformProps;

  constructor(props: GPUSpatialWeightsTransformProps) {
    const id = props.id ?? 'spatial-weights-transform';
    this.id = id;
    this.props = props;
    if (
      !['row', 'binary', 'kernel', 'symmetrize', 'double', 'variance'].includes(props.operation)
    ) {
      throw new Error(
        `${id} operation must be 'row', 'binary', 'kernel', 'symmetrize', 'double' or 'variance'`
      );
    }
    validateGPUSpatialWeights(id, props.weights);
    if (props.output) {
      validatePackedView(props.output, ['float32'], `${id} output`);
      if (props.output.length < props.weights.neighbors.length) {
        throw new Error(`${id} output must hold at least weights.neighbors.length values`);
      }
    }
    if (props.operation === 'kernel') {
      if (!props.weights.distances) {
        throw new Error(`${id} the 'kernel' operation requires weights.distances`);
      }
      const bandwidth = props.bandwidth ?? 'adaptive';
      if (bandwidth !== 'adaptive' && !(Number.isFinite(bandwidth) && bandwidth > 0)) {
        throw new Error(`${id} bandwidth must be 'adaptive' or a positive finite number`);
      }
      const kernel = props.kernel ?? 'triangular';
      if (!['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform'].includes(kernel)) {
        throw new Error(`${id} unknown kernel ${kernel}`);
      }
    }
    if (props.doubleSum !== undefined && !['one', 'rows'].includes(props.doubleSum)) {
      throw new Error(`${id} doubleSum must be 'one' or 'rows'`);
    }
    if (props.operation === 'symmetrize' && this.isInPlace()) {
      throw new Error(`${id} 'symmetrize' needs an output view separate from weights.weights`);
    }
  }

  /** Returns whether the transform writes `weights.weights` itself. */
  isInPlace(): boolean {
    const {output, weights} = this.props;
    return !output || output.buffer === weights.weights.buffer;
  }

  /** Returns the transform nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, operation} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      props.output
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const inPlace = this.isInPlace();
    // A kernel that reads and writes one buffer binds it once, read-write.
    const source = inPlace ? 'values' : 'source';
    const bindings = [
      {name: 'offsets', view: weights.offsets, type: 'u32' as const, access: 'read' as const},
      ...(inPlace
        ? [
            {
              name: 'values',
              view: weights.weights,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [
            {name: 'source', view: weights.weights, type: 'f32' as const, access: 'read' as const},
            {
              name: 'values',
              view: props.output!,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ])
    ];
    const read = (slot: string) => `${source}[${source}Offset + ${slot}]`;
    const write = (slot: string, value: string) => `values[valuesOffset + ${slot}] = ${value};`;
    const rowBounds = `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];`;
    const finite = `fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`;

    if (operation === 'row') {
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-row`,
          operation: OPERATION,
          variant: 'row',
          bindings,
          invocationCount: rows,
          declarations: finite,
          body: `${rowBounds}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${read('slot')};
  }
  let scale = select(0.0, 1.0 / sum, sum > 0.0 && isFiniteFloat(sum));
  for (var slot = begin; slot < end; slot++) {
    ${write('slot', `${read('slot')} * scale`)}
  }`
        })
      ];
    }
    if (operation === 'double' || operation === 'variance') {
      // Per-row partial, fixed-order global sum, then one apply pass over the rows.
      const partials = createTransientView(graph, `${id}-partials`, 'float32', rows);
      const total = createTransientView(graph, `${id}-total`, 'float32', 1);
      const variance = operation === 'variance';
      const squareSum = `var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${read('slot')};
    sum += value;
    squares += value * value;
  }
  let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);`;
      const factor = variance
        ? `let totalValue = total[totalOffset];
  let scale = select(0.0, f32(ROWS) / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`
        : `let totalValue = total[totalOffset];
  let scale = select(0.0, ${
    props.doubleSum === 'rows' ? 'f32(ROWS)' : '1.0'
  } / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`;
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-partials`,
          operation: OPERATION,
          variant: `${operation}-partials`,
          bindings: [
            ...bindings,
            {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: finite,
          body: `${rowBounds}
  ${
    variance
      ? `${squareSum}
  partials[partialsOffset + index] = select(0.0, sum / norm, valid);`
      : `var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${read('slot')};
  }
  partials[partialsOffset + index] = select(0.0, sum, isFiniteFloat(sum));`
  }`
        }),
        ...new GPUReduction({
          id: `${id}-reduce`,
          input: partials,
          output: total,
          operation: 'sum'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply`,
          operation: OPERATION,
          variant: `${operation}-apply`,
          bindings: [...bindings, {name: 'total', view: total, type: 'f32', access: 'read'}],
          invocationCount: rows,
          declarations: `${finite}\nconst ROWS: u32 = ${rows}u;`,
          body: `${rowBounds}
  ${factor}
  ${
    variance
      ? `${squareSum}
  let rowScale = select(0.0, scale / norm, valid);`
      : 'let rowScale = scale;'
  }
  for (var slot = begin; slot < end; slot++) {
    let result = ${read('slot')} * rowScale;
    ${write('slot', 'select(0.0, result, isFiniteFloat(result) && result >= 0.0)')}
  }`
        })
      ];
    }
    if (operation === 'binary') {
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-binary`,
          operation: OPERATION,
          variant: 'binary',
          bindings,
          invocationCount: rows,
          body: `${rowBounds}
  for (var slot = begin; slot < end; slot++) {
    ${write('slot', `select(0.0, 1.0, ${read('slot')} > 0.0)`)}
  }`
        })
      ];
    }
    if (operation === 'kernel') {
      const kernel = props.kernel ?? 'triangular';
      const bandwidth = props.bandwidth ?? 'adaptive';
      // The kernel reads distances, not the old weights, so the source binding is not needed.
      const kernelBindings = [
        bindings[0],
        {
          name: 'distances',
          view: weights.distances!,
          type: 'f32' as const,
          access: 'read' as const
        },
        {
          name: 'values',
          view: props.output ?? weights.weights,
          type: 'f32' as const,
          access: 'read_write' as const
        }
      ];
      const profile = {
        gaussian: 'exp(-0.5 * z * z) * 0.3989422804014327',
        triangular: 'max(1.0 - z, 0.0)',
        epanechnikov: '0.75 * max(1.0 - z * z, 0.0)',
        bisquare: '0.9375 * max(1.0 - z * z, 0.0) * max(1.0 - z * z, 0.0)',
        uniform: 'select(0.0, 0.5, z <= 1.0)'
      }[kernel];
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-kernel`,
          operation: OPERATION,
          variant: 'kernel',
          bindings: kernelBindings,
          invocationCount: rows,
          declarations: finite,
          body: `${rowBounds}
  ${
    bandwidth === 'adaptive'
      ? `var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }`
      : `let bandwidth = ${getWGSLFloatLiteral(bandwidth)};`
  }
  for (var slot = begin; slot < end; slot++) {
    let z = select(0.0, distances[distancesOffset + slot] / bandwidth, bandwidth > 0.0);
    let weight = ${profile};
    ${write('slot', 'select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0)')}
  }`
        })
      ];
    }
    // symmetrize: one invocation per slot; both rows are found by binary search.
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-symmetrize`,
        operation: OPERATION,
        variant: 'symmetrize',
        bindings: [
          bindings[0],
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          bindings[1],
          bindings[2]
        ],
        invocationCount: capacity,
        declarations: `const ROWS: u32 = ${rows}u;`,
        body: `if (index >= offsets[offsetsOffset + ROWS]) {
    return;
  }
  var row = 0u;
  var high = ROWS;
  while (row + 1u < high) {
    let middle = (row + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      row = middle;
    } else {
      high = middle;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  var reverse = 0.0;
  if (neighbor < ROWS) {
    var low = offsets[offsetsOffset + neighbor];
    var upper = offsets[offsetsOffset + neighbor + 1u];
    while (low < upper) {
      let middle = (low + upper) / 2u;
      let candidate = neighbors[neighborsOffset + middle];
      if (candidate == row) {
        reverse = source[sourceOffset + middle];
        break;
      }
      if (candidate < row) {
        low = middle + 1u;
      } else {
        upper = middle;
      }
    }
  }
  values[valuesOffset + index] = 0.5 * (source[sourceOffset + index] + reverse);`
      })
    ];
  }
}
