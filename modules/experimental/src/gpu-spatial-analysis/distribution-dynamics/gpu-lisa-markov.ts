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
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUInferenceMetadata} from '../contracts/index';
import {GPULocalMoran} from '../spatial-autocorrelation/gpu-local-moran';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../spatial-autocorrelation/spatial-autocorrelation-parameters';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_CLASS_ASSIGNMENT_NO_CLASS} from './gpu-class-assignment';
import {GPUTransitionMatrix, type GPUTransitionMatrixOutput} from './gpu-transition-matrix';
import {getPeriodView} from './period-views';

const OPERATION = 'GPULISAMarkov';

/**
 * Number of LISA states: 0 not significant, then HH, LH, LL, HL (`GPU_LOCAL_MORAN_QUADRANT`).
 * The layout is the same for both {@link GPULISAMarkovGating} modes; state 0 is only populated
 * under `'significance'`.
 */
export const GPU_LISA_MARKOV_STATE_COUNT = 5;

/**
 * How {@link GPULISAMarkov} forms the states: `'none'` (default) uses the four local Moran
 * quadrants of every valid row, as giddy `LISA_Markov` does; `'significance'` puts rows whose
 * analytic local Moran p-value exceeds the significance level in state 0.
 */
export type GPULISAMarkovGating = 'significance' | 'none';

/** Caller-owned outputs of {@link GPULISAMarkov}. */
export type GPULISAMarkovOutput = GPUTransitionMatrixOutput & {
  /**
   * Optional LISA state per row and period, period-major (`rows * periods` entries): 0 to 4 as in
   * {@link GPU_LISA_MARKOV_STATE_COUNT}, or `GPU_CLASS_ASSIGNMENT_NO_CLASS` (`0xffffffff`) for a
   * row-period that is not part of the analysis.
   */
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
  /**
   * Compile-time. {@link GPULISAMarkovGating}; defaults to `'none'`, the four-quadrant chain of
   * giddy `LISA_Markov`.
   */
  gating?: GPULISAMarkovGating;
  /**
   * Compile-time. A `(from)` state with fewer transitions than this is counted in
   * `output.stateSparsity`. Defaults to 5.
   */
  minimumTransitions?: number;
  /** Caller-owned outputs. */
  output: GPULISAMarkovOutput;
};

/**
 * LISA Markov (Rey 2001, giddy `LISA_Markov`): transition counts between local Moran quadrants of
 * consecutive periods. Each period runs `GPULocalMoran` on its slice of `values` (moments and
 * significance per period, as esda would per year), and a `GPUTransitionMatrix` counts moves
 * between the {@link GPU_LISA_MARKOV_STATE_COUNT} states.
 *
 * States: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL; the 5-state layout is the same for both
 * `gating` modes so a legend never changes with the option.
 * - `gating: 'none'` (default, giddy `LISA_Markov` parity): every valid row is in its quadrant, the
 *   signs of the centered value and of the spatial lag, so state 0 stays empty and the chain is a
 *   four-state chain. A row exactly at the mean or with an exactly zero lag has no quadrant here
 *   (esda would call it low) and is not counted.
 * - `gating: 'significance'`: a row whose analytic local Moran p-value is above the per-frame
 *   significance level is state 0. The test is the per-period analytic normal z of `GPULocalMoran`
 *   at an uncorrected level, not esda's permutation p-value, and every period is gated on its own,
 *   so a transition into or out of state 0 can be a threshold crossing of a similar local value
 *   rather than a change of the local pattern. Use it as a map of significant clusters, not as a
 *   test of transition dynamics. Under multiplicity (many rows tested per period) expect false
 *   0-to-quadrant moves; `getInferenceMetadata` describes the test.
 *
 * Quadrant numbering: `GPU_LOCAL_MORAN_QUADRANT` is esda's default numbering (`Moran_Local` with
 * `geoda_quads=False`, which `LISA_Markov` uses by default): 1 HH, 2 LH, 3 LL, 4 HL. With
 * `geoda_quads=True` esda swaps 2 and 3 (1 HH, 2 LL, 3 LH, 4 HL); map the rows and columns of
 * `counts` when comparing with such a run.
 *
 * Missing data: a row-period whose value is not finite, that is masked, or that has no neighbor
 * (positive weight, other than itself, among the rows that are finite and unmasked in that period)
 * has no state: it is written as `GPU_CLASS_ASSIGNMENT_NO_CLASS` in `output.quadrants` and a
 * transition with such a state at either end is not counted but included in `output.ignored`. It
 * is never counted as state 0. Each period excludes its invalid rows from the mean and the
 * neighbor sums (`GPULocalMoran`'s default island policy `'exclude'`), so a row valid in only some
 * periods changes the neighbors of the others in those periods. Counts are exact integers.
 */
export class GPULISAMarkov implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULISAMarkovProps;
  /** Rows per period. */
  readonly rows: number;
  /** Resolved gating mode. */
  readonly gating: GPULISAMarkovGating;

  constructor(props: GPULISAMarkovProps) {
    const id = props.id ?? 'lisa-markov';
    this.id = id;
    this.props = props;
    this.rows = validateGPUSpatialWeights(id, props.weights);
    this.gating = props.gating ?? 'none';
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

  /**
   * Describes the significance gating of the states, or `undefined` for `gating: 'none'` (no
   * p-value is used). Two-sided analytic normal p-values, no multiple-testing correction.
   *
   * @param significanceLevel Level of the per-frame `parameters`; defaults to 0.05.
   */
  getInferenceMetadata(significanceLevel = 0.05): GPUInferenceMetadata | undefined {
    if (this.gating === 'none') {
      return undefined;
    }
    return {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: 'none',
      significanceLevel,
      testedRowsDefinition:
        'per period, unmasked rows with a finite value and at least one neighbor among such rows, a nonzero centered value and a defined z-score; each period tested separately'
    };
  }

  /** Returns one local Moran chain per period, the state kernel, then the transition nodes. */
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
      output.quadrants,
      output.homogeneity,
      output.steadyStates,
      output.stateSparsity,
      output.status?.converged
    ]);
    const total = rows * periods;
    const quadrants =
      output.quadrants ?? createTransientView(graph, `${id}-quadrants`, 'uint32', total);
    const zScores = createTransientView(graph, `${id}-z-scores`, 'float32', total);
    const neighborCounts = createTransientView(graph, `${id}-neighbor-counts`, 'uint32', total);
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
          quadrants: getPeriodView(graph, quadrants, period, rows),
          neighborCounts: getPeriodView(graph, neighborCounts, period, rows),
          quadrantGating: this.gating === 'none' ? 'none' : 'analytic',
          minimumNeighbors: 1
        }).getCommandNodes(graph)
      );
    }
    // GPULocalMoran writes quadrant 0 for rows outside the analysis. Those rows (masked,
    // non-finite, no neighbor) and, ungated, rows without a quadrant get the out-of-range class
    // that GPUTransitionMatrix skips, so they are counted as ignored and never as state 0.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-states`,
        operation: OPERATION,
        variant: 'states',
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'neighborCounts', view: neighborCounts, type: 'u32', access: 'read'},
          {name: 'quadrants', view: quadrants, type: 'u32', access: 'read_write'}
        ],
        invocationCount: total,
        declarations: IS_FINITE_BITS_WGSL,
        body: `var state = quadrants[quadrantsOffset + index];
  if (!isFiniteBits(values[valuesOffset + index]) || neighborCounts[neighborCountsOffset + index] == 0u${
    this.gating === 'none' ? ' || state == 0u' : ''
  }) {
    state = ${GPU_CLASS_ASSIGNMENT_NO_CLASS}u;
  }
  quadrants[quadrantsOffset + index] = state;`
      })
    );
    nodes.push(
      ...new GPUTransitionMatrix({
        id: `${id}-transitions`,
        classes: quadrants,
        rows,
        periods,
        classCount: GPU_LISA_MARKOV_STATE_COUNT,
        periodLag: props.periodLag,
        mask,
        minimumTransitions: props.minimumTransitions,
        output: {
          counts: output.counts,
          probabilities: output.probabilities,
          rowTotals: output.rowTotals,
          ignored: output.ignored,
          homogeneity: output.homogeneity,
          steadyStates: output.steadyStates,
          stateSparsity: output.stateSparsity,
          status: output.status
        }
      }).getCommandNodes(graph)
    );
    return nodes;
  }
}
