// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {GPUClassBreaks} from '../../gpu-dataframe/column-classification/gpu-class-breaks';
import type {GPUClassBreaksMethod} from '../../gpu-dataframe/column-classification/class-breaks-parameters';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPUSpatialLag} from '../spatial-weights/gpu-spatial-lag';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPUClassAssignment} from './gpu-class-assignment';
import {GPUTransitionMatrix, type GPUTransitionMatrixOutput} from './gpu-transition-matrix';
import {getPeriodView} from './period-views';

/** Caller-owned outputs of {@link GPUSpatialMarkov}. */
export type GPUSpatialMarkovOutput = GPUTransitionMatrixOutput & {
  /**
   * Pooled class edges of the spatial lag, `lagMaximumClassCount + 1` rows (see
   * `GPUClassBreaks.output.breaks`). One legend for every period.
   */
  lagBreaks: GraphDataView<'float32'>;
  /** One row receiving the produced lag class count (`GPUClassBreaks.output.classCount`). */
  lagClassCount: GraphDataView<'uint32'>;
  /** Optional spatial lag per row and period, period-major (`rows * periods` entries). */
  lagValues?: GraphDataView<'float32'>;
  /** Optional lag class per row and period, period-major (`rows * periods` entries). */
  lagClasses?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSpatialMarkov}.
 *
 * Topology: `rows`, `periods`, `classCount`, `lagMaximumClassCount`, `lagMethods`, `periodLag`,
 * `normalize` and which optional views exist. Per-frame: every view's contents, including
 * `lagParameters` (lag classification method and class count).
 */
export type GPUSpatialMarkovProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-markov'`. */
  id?: string;
  /** Analysis values per row and period, period-major (`rows * periods` entries). */
  values: GraphDataView<'float32'>;
  /** Class of `values` per row and period, same layout, for example from {@link GPUClassAssignment}. */
  classes: GraphDataView<'uint32'>;
  /** Number of classes `K` of `classes`, 1 to 64. */
  classCount: number;
  /** Square self-join weights over the `rows` locations, constant across periods. */
  weights: GPUSpatialWeights;
  /** Number of periods. */
  periods: number;
  /** Periods between the two ends of a transition. Defaults to 1. */
  periodLag?: number;
  /**
   * When true (default), the lag is the weighted mean of the neighbors (row-standardized weights,
   * giddy's default); otherwise the weighted sum.
   */
  normalize?: boolean;
  /** Compile-time class capacity of the lag classification, 1 to 64. Also the condition count. */
  lagMaximumClassCount: number;
  /** `GPUClassBreaks` per-frame parameters for the lag (`getGPUClassBreaksParameterValues`). */
  lagParameters: GraphDataView<'float32'>;
  /** Methods compiled into the lag classification. Defaults to `['quantile']`. */
  lagMethods?: readonly GPUClassBreaksMethod[];
  /** Optional row selection (`rows` entries) applied to every period. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUSpatialMarkovOutput;
};

/**
 * Spatial Markov (Rey 2001, giddy `Spatial_Markov`): transition counts of a variable's classes
 * conditioned on the class of the spatial lag at the start of each transition.
 *
 * Chain: one `GPUSpatialLag` per period, then a single `GPUClassBreaks` over the pooled lag column
 * of all periods (so the lag legend is one set of edges), a `GPUClassAssignment` of the lag, and a
 * `GPUTransitionMatrix` with the lag class as condition. The counts tensor is
 * `lagMaximumClassCount * classCount * classCount`; condition `c` is the `c`th lag class, so slices
 * past the produced `lagClassCount` stay zero. Counts are exact integers. The lag values and edges
 * are f32 (fixed-order sums), so a lag within f32 rounding of an edge may fall in the neighboring
 * lag class compared with a double-precision evaluation.
 *
 * Not included: giddy's `relative` rescaling by the period mean (divide the values per period
 * before classification) and its homogeneity tests, which consume the counts on the CPU.
 */
export class GPUSpatialMarkov implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialMarkovProps;
  /** Rows per period. */
  readonly rows: number;

  constructor(props: GPUSpatialMarkovProps) {
    const id = props.id ?? 'spatial-markov';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.weights);
    const total = this.rows * props.periods;
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedUint32View(props.classes, `${id} classes`);
    if (!Number.isInteger(props.periods) || props.periods < 2) {
      throw new Error(`${id} periods must be an integer of at least 2`);
    }
    if (props.values.length !== total || props.classes.length !== total) {
      throw new Error(`${id} values and classes length must equal rows * periods`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rows) {
        throw new Error(`${id} mask length must equal the weights row count`);
      }
    }
    if (props.output.lagValues && props.output.lagValues.length !== total) {
      throw new Error(`${id} output.lagValues length must equal rows * periods`);
    }
    if (props.output.lagClasses && props.output.lagClasses.length !== total) {
      throw new Error(`${id} output.lagClasses length must equal rows * periods`);
    }
  }

  /** Returns the lag, pooled-breaks, assignment and transition nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {output, weights, periods, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.classes,
      props.lagParameters,
      mask,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.counts,
      output.probabilities,
      output.rowTotals,
      output.ignored,
      output.lagBreaks,
      output.lagClassCount,
      output.lagValues,
      output.lagClasses
    ]);
    const total = rows * periods;
    const lagValues =
      output.lagValues ?? createTransientView(graph, `${id}-lag-values`, 'float32', total);
    const lagClasses =
      output.lagClasses ?? createTransientView(graph, `${id}-lag-classes`, 'uint32', total);
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (let period = 0; period < periods; period++) {
      nodes.push(
        ...new GPUSpatialLag({
          id: `${id}-lag-${period}`,
          values: getPeriodView(graph, props.values, period, rows),
          weights,
          mask,
          normalize: props.normalize ?? true,
          output: getPeriodView(graph, lagValues, period, rows)
        }).getCommandNodes(graph)
      );
    }
    let pooledMask: GraphDataView<'uint32'> | undefined;
    if (mask) {
      pooledMask = createTransientView(graph, `${id}-pooled-mask`, 'uint32', total);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pooled-mask`,
          operation: 'GPUSpatialMarkov',
          variant: 'pooled-mask',
          bindings: [
            {name: 'mask', view: mask, type: 'u32', access: 'read'},
            {name: 'pooled', view: pooledMask, type: 'u32', access: 'read_write'}
          ],
          invocationCount: total,
          body: `pooled[pooledOffset + index] = mask[maskOffset + index % ${rows}u];`
        })
      );
    }
    nodes.push(
      ...new GPUClassBreaks({
        id: `${id}-lag-breaks`,
        values: lagValues,
        mask: pooledMask,
        parameters: props.lagParameters,
        maximumClassCount: props.lagMaximumClassCount,
        methods: props.lagMethods ?? ['quantile'],
        output: {breaks: output.lagBreaks, classCount: output.lagClassCount}
      }).getCommandNodes(graph),
      ...new GPUClassAssignment({
        id: `${id}-lag-classes`,
        values: lagValues,
        breaks: output.lagBreaks,
        classCount: output.lagClassCount,
        mask: pooledMask,
        output: lagClasses
      }).getCommandNodes(graph),
      ...new GPUTransitionMatrix({
        id: `${id}-transitions`,
        classes: props.classes,
        rows,
        periods,
        classCount: props.classCount,
        periodLag: props.periodLag,
        conditionClasses: lagClasses,
        conditionCount: props.lagMaximumClassCount,
        mask,
        output: {
          counts: output.counts,
          probabilities: output.probabilities,
          rowTotals: output.rowTotals,
          ignored: output.ignored
        }
      }).getCommandNodes(graph)
    );
    return nodes;
  }
}
