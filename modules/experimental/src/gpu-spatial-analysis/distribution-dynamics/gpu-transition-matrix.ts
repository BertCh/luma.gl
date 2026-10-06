// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPUTransitionMatrix';
const MAXIMUM_CLASS_COUNT = 64;

/** Caller-owned outputs of {@link GPUTransitionMatrix}. */
export type GPUTransitionMatrixOutput = {
  /**
   * Exact transition counts, `conditionCount * classCount * classCount` rows: row
   * `(c * classCount + from) * classCount + to` counts transitions from class `from` at period `t`
   * to class `to` at period `t + periodLag` among rows whose condition class at `t` is `c`.
   */
  counts: GraphDataView<'uint32'>;
  /**
   * Optional row-normalized probabilities, same layout as `counts`: `counts / rowTotal`, and 0 for
   * a row with no transitions.
   */
  probabilities?: GraphDataView<'float32'>;
  /** Optional transitions out of each `(condition, from)` state, `conditionCount * classCount` rows. */
  rowTotals?: GraphDataView<'uint32'>;
  /**
   * Optional single row counting row-period pairs that were not counted: masked rows and any pair
   * whose classes or condition class are out of range (for example NaN or masked classes).
   */
  ignored?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTransitionMatrix}.
 *
 * Topology (new graph): `rows`, `periods`, `periodLag`, `classCount`, `conditionCount` and which
 * optional views exist. Per-frame: the contents of `classes`, `conditionClasses` and `mask`.
 */
export type GPUTransitionMatrixProps = {
  /** Prefix for generated node IDs. Defaults to `'transition-matrix'`. */
  id?: string;
  /**
   * Class per row and period, period-major: period `t` of row `i` is at `t * rows + i`. Length
   * `rows * periods`. Values at or above `classCount` (such as `0xffffffff`) are not counted.
   */
  classes: GraphDataView<'uint32'>;
  /** Number of rows (locations) per period. */
  rows: number;
  /** Number of periods, at least `periodLag + 1`. */
  periods: number;
  /** Number of classes `K`, 1 to 64. */
  classCount: number;
  /** Periods between the two ends of a transition. Defaults to 1. */
  periodLag?: number;
  /**
   * Optional class used to condition the counts, same layout as `classes`: the transition from
   * period `t` is counted in the slice of `conditionClasses[t * rows + i]`. Values at or above
   * `conditionCount` are not counted.
   */
  conditionClasses?: GraphDataView<'uint32'>;
  /** Number of condition classes `C`. Defaults to 1 (and must be 1 without `conditionClasses`). */
  conditionCount?: number;
  /** Optional row selection (`rows` entries): zero rows are skipped in every period. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUTransitionMatrixOutput;
};

/**
 * Exact class transition counts across periods (giddy `Markov`): how many locations moved from
 * class `a` at period `t` to class `b` at period `t + periodLag`, summed over all `t`, with
 * row-normalized probabilities.
 *
 * Counts are exact `uint32` integer atomics, independent of dispatch order, so a time slider can
 * recompute them each frame with no drift. With `conditionClasses` the counts are split by the
 * condition's class at the start of each transition, which is the spatial Markov of Rey (2001)
 * (see {@link GPUSpatialMarkov}). Probabilities are the f32 quotient of two exact counts.
 *
 * Transitions are counted for `t = 0 .. periods - periodLag - 1`. The matrix is the maximum
 * likelihood estimate of the Markov transition probabilities, `p_ab = n_ab / n_a`.
 */
export class GPUTransitionMatrix implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTransitionMatrixProps;
  /** Resolved condition class count. */
  readonly conditionCount: number;
  /** Resolved period lag. */
  readonly periodLag: number;

  constructor(props: GPUTransitionMatrixProps) {
    const id = props.id ?? 'transition-matrix';
    this.id = id;
    this.props = props;
    this.periodLag = props.periodLag ?? 1;
    this.conditionCount = props.conditionCount ?? 1;
    const {rows, periods, classCount, output} = props;
    if (!Number.isInteger(rows) || rows < 1) {
      throw new Error(`${id} rows must be a positive integer`);
    }
    if (!Number.isInteger(this.periodLag) || this.periodLag < 1) {
      throw new Error(`${id} periodLag must be a positive integer`);
    }
    if (!Number.isInteger(periods) || periods < this.periodLag + 1) {
      throw new Error(`${id} periods must be an integer of at least periodLag + 1`);
    }
    if (!Number.isInteger(classCount) || classCount < 1 || classCount > MAXIMUM_CLASS_COUNT) {
      throw new Error(`${id} classCount must be an integer in [1, ${MAXIMUM_CLASS_COUNT}]`);
    }
    if (!Number.isInteger(this.conditionCount) || this.conditionCount < 1) {
      throw new Error(`${id} conditionCount must be a positive integer`);
    }
    if (!props.conditionClasses && this.conditionCount !== 1) {
      throw new Error(`${id} conditionCount needs conditionClasses`);
    }
    validatePackedUint32View(props.classes, `${id} classes`);
    if (props.classes.length !== rows * periods) {
      throw new Error(`${id} classes length must equal rows * periods`);
    }
    if (props.conditionClasses) {
      validatePackedUint32View(props.conditionClasses, `${id} conditionClasses`);
      if (props.conditionClasses.length !== rows * periods) {
        throw new Error(`${id} conditionClasses length must equal rows * periods`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal rows`);
      }
    }
    const cellCount = this.conditionCount * classCount * classCount;
    validatePackedUint32View(output.counts, `${id} output.counts`);
    if (output.counts.length < cellCount) {
      throw new Error(`${id} output.counts must hold conditionCount * classCount^2 rows`);
    }
    if (output.probabilities) {
      validatePackedView(output.probabilities, ['float32'], `${id} output.probabilities`);
      if (output.probabilities.length < cellCount) {
        throw new Error(`${id} output.probabilities must hold conditionCount * classCount^2 rows`);
      }
    }
    if (output.rowTotals) {
      validatePackedUint32View(output.rowTotals, `${id} output.rowTotals`);
      if (output.rowTotals.length < this.conditionCount * classCount) {
        throw new Error(`${id} output.rowTotals must hold conditionCount * classCount rows`);
      }
    }
    if (output.ignored) {
      validatePackedUint32View(output.ignored, `${id} output.ignored`);
      if (output.ignored.length < 1) {
        throw new Error(`${id} output.ignored must hold one row`);
      }
    }
    const outputs = [output.counts, output.probabilities, output.rowTotals, output.ignored];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.classes,
      props.conditionClasses,
      props.mask
    ]);
    const buffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(buffers).size !== buffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns `clear`, `count` and, when requested, `normalize` nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, conditionCount, periodLag} = this;
    const {classes, conditionClasses, mask, output, rows, periods, classCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      classes,
      conditionClasses,
      mask,
      output.counts,
      output.probabilities,
      output.rowTotals,
      output.ignored
    ]);
    const cellCount = conditionCount * classCount * classCount;
    const stateCount = conditionCount * classCount;
    const constants = `const ROWS: u32 = ${rows}u;
const CLASSES: u32 = ${classCount}u;
const CONDITIONS: u32 = ${conditionCount}u;
const CELLS: u32 = ${cellCount}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [
          {name: 'counts', view: output.counts, type: 'u32', access: 'read_write'},
          ...(output.ignored
            ? [
                {
                  name: 'ignored',
                  view: output.ignored,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: cellCount,
        declarations: constants,
        body: `counts[countsOffset + index] = 0u;
  ${output.ignored ? 'if (index == 0u) { ignored[ignoredOffset] = 0u; }' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          {name: 'classes', view: classes, type: 'u32', access: 'read'},
          ...(conditionClasses
            ? [
                {
                  name: 'conditions',
                  view: conditionClasses,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'counts', view: output.counts, type: 'atomic<u32>', access: 'read_write'},
          ...(output.ignored
            ? [
                {
                  name: 'ignored',
                  view: output.ignored,
                  type: 'atomic<u32>' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows * (periods - periodLag),
        declarations: constants,
        body: `let period = index / ROWS;
  let row = index - period * ROWS;
  let fromClass = classes[classesOffset + index];
  let toClass = classes[classesOffset + index + ${periodLag}u * ROWS];
  let condition = ${conditionClasses ? 'conditions[conditionsOffset + index]' : '0u'};
  let selected = ${mask ? 'mask[maskOffset + row] != 0u' : 'true'};
  if (selected && fromClass < CLASSES && toClass < CLASSES && condition < CONDITIONS) {
    atomicAdd(&counts[countsOffset + (condition * CLASSES + fromClass) * CLASSES + toClass], 1u);
  } else {
    ${output.ignored ? 'atomicAdd(&ignored[ignoredOffset], 1u);' : ''}
  }`
      })
    ];
    if (output.probabilities || output.rowTotals) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-normalize`,
          operation: OPERATION,
          variant: 'normalize',
          bindings: [
            {name: 'counts', view: output.counts, type: 'u32', access: 'read'},
            ...(output.probabilities
              ? [
                  {
                    name: 'probabilities',
                    view: output.probabilities,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(output.rowTotals
              ? [
                  {
                    name: 'rowTotals',
                    view: output.rowTotals,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: stateCount,
          declarations: constants,
          body: `var total = 0u;
  for (var to = 0u; to < CLASSES; to++) {
    total += counts[countsOffset + index * CLASSES + to];
  }
  for (var to = 0u; to < CLASSES; to++) {
    ${
      output.probabilities
        ? `probabilities[probabilitiesOffset + index * CLASSES + to] = select(0.0, f32(counts[countsOffset + index * CLASSES + to]) / f32(total), total > 0u);`
        : ''
    }
  }
  ${output.rowTotals ? 'rowTotals[rowTotalsOffset + index] = total;' : ''}`
        })
      );
    }
    return nodes;
  }
}
