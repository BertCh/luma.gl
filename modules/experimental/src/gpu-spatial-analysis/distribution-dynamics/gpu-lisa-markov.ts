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
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPULocalMoran} from '../spatial-autocorrelation/gpu-local-moran';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../spatial-autocorrelation/spatial-autocorrelation-parameters';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPUTransitionMatrix, type GPUTransitionMatrixOutput} from './gpu-transition-matrix';
import {getPeriodView} from './period-views';

/** Number of LISA states: 0 not significant, then HH, LH, LL, HL (`GPU_LOCAL_MORAN_QUADRANT`). */
export const GPU_LISA_MARKOV_STATE_COUNT = 5;

/** Caller-owned outputs of {@link GPULISAMarkov}. */
export type GPULISAMarkovOutput = GPUTransitionMatrixOutput & {
  /** Optional LISA state per row and period, period-major (`rows * periods` entries). */
  quadrants?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULISAMarkov}.
 *
 * Topology: `rows`, `periods`, `periodLag` and which optional views exist. Per-frame: every
 * view's contents, including `parameters` (significance level).
 */
export type GPULISAMarkovProps = {
  /** Prefix for generated node IDs. Defaults to `'lisa-markov'`. */
  id?: string;
  /** Analysis values per row and period, period-major (`rows * periods` entries). */
  values: GraphDataView<'float32'>;
  /** Square self-join weights over the `rows` locations, constant across periods. */
  weights: GPUSpatialWeights;
  /** Number of periods, at least `periodLag + 1`. */
  periods: number;
  /** Periods between the two ends of a transition. Defaults to 1. */
  periodLag?: number;
  /** `GPULocalMoran` per-frame parameters (`getGPUSpatialAutocorrelationParameterValues`). */
  parameters: GraphDataView<'float32'>;
  /** Optional row selection (`rows` entries) applied to every period. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPULISAMarkovOutput;
};

/**
 * LISA Markov (Rey 2001, giddy `LISA_Markov`): transition counts between local Moran quadrants of
 * consecutive periods. Each period runs `GPULocalMoran` on its slice of `values` (moments and
 * significance per period, as esda would per year), and a `GPUTransitionMatrix` counts moves
 * between the {@link GPU_LISA_MARKOV_STATE_COUNT} states.
 *
 * States: 0 not significant at the per-frame level, 1 HH, 2 LH, 3 LL, 4 HL. Rows excluded by
 * `mask` are never counted. A row with a non-finite value or no neighbors is state 0 and is
 * counted. Quadrants use the analytic z-score of `GPULocalMoran`, so p-values differ from esda's
 * permutation inference. Counts are exact integers.
 */
export class GPULISAMarkov implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULISAMarkovProps;
  /** Rows per period. */
  readonly rows: number;

  constructor(props: GPULISAMarkovProps) {
    const id = props.id ?? 'lisa-markov';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} words`
      );
    }
    if (!Number.isInteger(props.periods) || props.periods < (props.periodLag ?? 1) + 1) {
      throw new Error(`${id} periods must be an integer of at least periodLag + 1`);
    }
    if (props.values.length !== this.rows * props.periods) {
      throw new Error(`${id} values length must equal rows * periods`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rows) {
        throw new Error(`${id} mask length must equal the weights row count`);
      }
    }
    if (props.output.quadrants) {
      validatePackedUint32View(props.output.quadrants, `${id} output.quadrants`);
      if (props.output.quadrants.length !== this.rows * props.periods) {
        throw new Error(`${id} output.quadrants length must equal rows * periods`);
      }
    }
  }

  /** Returns one local Moran chain per period, then the transition nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {output, weights, periods, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.parameters,
      mask,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.counts,
      output.probabilities,
      output.rowTotals,
      output.ignored,
      output.quadrants
    ]);
    const total = rows * periods;
    const quadrants =
      output.quadrants ?? createTransientView(graph, `${id}-quadrants`, 'uint32', total);
    const zScores = createTransientView(graph, `${id}-z-scores`, 'float32', total);
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (let period = 0; period < periods; period++) {
      nodes.push(
        ...new GPULocalMoran({
          id: `${id}-moran-${period}`,
          weights,
          values: getPeriodView(graph, props.values, period, rows),
          parameters: props.parameters,
          mask,
          zScores: getPeriodView(graph, zScores, period, rows),
          quadrants: getPeriodView(graph, quadrants, period, rows)
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      ...new GPUTransitionMatrix({
        id: `${id}-transitions`,
        classes: quadrants,
        rows,
        periods,
        classCount: GPU_LISA_MARKOV_STATE_COUNT,
        periodLag: props.periodLag,
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
