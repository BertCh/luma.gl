// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createAccessibilityColumnGatherNode,
  createAccessibilityRowReduceNode
} from './network-accessibility-passes';

const OPERATION = 'GPUNetworkAccessibility';

/** Number of `float32` words in the per-frame parameter view of {@link GPUNetworkAccessibility}. */
export const GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH = 4;

/**
 * Distance-decay function of {@link GPUNetworkAccessibility}:
 * - `'none'`: every opportunity within the threshold counts fully (classic 2SFCA catchment).
 * - `'exponential'`: negative exponential `exp(-beta * cost)`.
 * - `'power'`: power decay `max(cost, minimumCost) ^ -beta`.
 */
export type GPUNetworkAccessibilityDecay = 'none' | 'exponential' | 'power';

/** Per-frame scoring parameters of {@link GPUNetworkAccessibility}. */
export type GPUNetworkAccessibilityParameters = {
  /**
   * Cost cutoff: only matrix entries with a finite cost `<= threshold` contribute. `Infinity`
   * keeps every reached entry. Must not exceed the `costLimit` the matrix was built with.
   */
  threshold: number;
  /** Decay applied to the `gravity` and two-step floating catchment outputs. Defaults to `'none'`. */
  decay?: GPUNetworkAccessibilityDecay;
  /** Decay rate. Defaults to 0. */
  beta?: number;
  /** Power-decay floor so zero costs stay finite. Must be positive. Defaults to 1. */
  minimumCost?: number;
};

const DECAY_CODES: Record<GPUNetworkAccessibilityDecay, number> = {
  none: 0,
  exponential: 1,
  power: 2
};

/**
 * Encodes per-frame scoring parameters as the four `float32` words read by
 * {@link GPUNetworkAccessibility}: threshold, decay code, beta, minimum cost.
 */
export function encodeGPUNetworkAccessibilityParameters(
  parameters: GPUNetworkAccessibilityParameters
): Float32Array {
  return Float32Array.from([
    parameters.threshold,
    DECAY_CODES[parameters.decay ?? 'none'],
    parameters.beta ?? 0,
    parameters.minimumCost ?? 1
  ]);
}

/**
 * Two-step floating catchment area (2SFCA) inputs and outputs of {@link GPUNetworkAccessibility}.
 * Only for `orientation: 'opportunity-rows'`, where rows are facilities and `opportunityWeights`
 * holds their supply.
 */
export type GPUNetworkAccessibilityCatchment = {
  /** Demand (for example population) per column node. */
  demand: GraphDataView<'float32'>;
  /** Per-node accessibility `A_i = sum_j R_j f(c_ij)` over facilities within the threshold. */
  output: GraphDataView<'float32'>;
  /**
   * Optional per-facility supply-to-demand ratio `R_j = S_j / sum_i P_i f(c_ij)`, 0 when no demand
   * is in its catchment.
   */
  ratios?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUNetworkAccessibility}.
 *
 * Compile-time: matrix shape, orientation, and which outputs exist. Per-frame: the contents of
 * `costs`, weights, demand, and `parameters` (threshold, decay, beta, minimum cost).
 */
export type GPUNetworkAccessibilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-accessibility'`. */
  id?: string;
  /**
   * Row-major `[rowCount x nodeCount]` cost matrix, for example the `costs` of
   * `GPUNetworkCostMatrix`. Unreached entries are `+Infinity`.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Which side the matrix rows are on. Defaults to `'opportunity-rows'`.
   * - `'opportunity-rows'`: rows are opportunity (or facility) searches over the reverse network,
   *   columns are origin nodes. `opportunityWeights` has one row per matrix row and outputs have one
   *   row per node.
   * - `'origin-rows'`: rows are origin searches over the forward network, columns are nodes that
   *   hold opportunities. `opportunityWeights` has one row per node and outputs one row per matrix
   *   row.
   */
  orientation?: 'opportunity-rows' | 'origin-rows';
  /** Opportunity weight (jobs, beds, supply) per opportunity row or node. */
  opportunityWeights: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: `GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH` `float32` words written with
   * {@link encodeGPUNetworkAccessibilityParameters}.
   */
  parameters: GraphDataView<'float32'>;
  /** Optional cumulative opportunities within the threshold, per origin. */
  cumulative?: GraphDataView<'float32'>;
  /** Optional gravity accessibility `sum_j O_j f(c_ij)` within the threshold, per origin. */
  gravity?: GraphDataView<'float32'>;
  /** Optional two-step floating catchment area. */
  catchment?: GPUNetworkAccessibilityCatchment;
};

/**
 * Scores per-origin accessibility from a retained network cost matrix: cumulative opportunities
 * within a per-frame cost threshold, gravity accessibility with negative-exponential or power decay,
 * and the two-step floating catchment area (2SFCA, with the decay applied inside each catchment).
 *
 * The contributor only reads the matrix, so changing the threshold, decay, or beta re-encodes a few
 * linear passes and never re-runs a shortest-path search: build the matrix with
 * `GPUNetworkCostMatrix` in one graph, encode it when the network or the opportunity set changes,
 * and encode this contributor's graph every frame.
 *
 * Sums are deterministic without float atomics: per-node outputs gather the rows in ascending order
 * in one invocation, and per-row outputs reduce each row in one workgroup with a fixed tree.
 */
export class GPUNetworkAccessibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkAccessibilityProps;
  /** Resolved matrix orientation. */
  readonly orientation: 'opportunity-rows' | 'origin-rows';
  /** Matrix rows. */
  readonly rowCount: number;
  /** Matrix columns (network nodes). */
  readonly nodeCount: number;

  constructor(props: GPUNetworkAccessibilityProps) {
    this.id = props.id ?? 'network-accessibility';
    this.props = props;
    this.orientation = props.orientation ?? 'opportunity-rows';
    const {id} = this;
    const {catchment} = props;
    for (const [name, view] of [
      ['costs', props.costs],
      ['opportunityWeights', props.opportunityWeights],
      ['parameters', props.parameters],
      ['cumulative', props.cumulative],
      ['gravity', props.gravity],
      ['catchment.demand', catchment?.demand],
      ['catchment.output', catchment?.output],
      ['catchment.ratios', catchment?.ratios]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (props.parameters.length !== GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH} rows`
      );
    }
    const weightCount = props.opportunityWeights.length;
    if (weightCount < 1 || props.costs.length % weightCount !== 0) {
      throw new Error(`${id} costs length must be a multiple of opportunityWeights length`);
    }
    const otherCount = props.costs.length / weightCount;
    if (otherCount < 1) {
      throw new Error(`${id} costs must not be empty`);
    }
    const isOpportunityRows = this.orientation === 'opportunity-rows';
    this.rowCount = isOpportunityRows ? weightCount : otherCount;
    this.nodeCount = isOpportunityRows ? otherCount : weightCount;
    const originCount = isOpportunityRows ? this.nodeCount : this.rowCount;
    for (const [name, view] of [
      ['cumulative', props.cumulative],
      ['gravity', props.gravity]
    ] as const) {
      if (view && view.length !== originCount) {
        throw new Error(`${id} ${name} length must equal the origin count`);
      }
    }
    if (catchment) {
      if (!isOpportunityRows) {
        throw new Error(`${id} catchment requires orientation 'opportunity-rows'`);
      }
      if (catchment.demand.length !== this.nodeCount) {
        throw new Error(`${id} catchment.demand length must equal the node count`);
      }
      if (catchment.output.length !== this.nodeCount) {
        throw new Error(`${id} catchment.output length must equal the node count`);
      }
      if (catchment.ratios && catchment.ratios.length !== this.rowCount) {
        throw new Error(`${id} catchment.ratios length must equal the row count`);
      }
    }
    if (!props.cumulative && !props.gravity && !catchment) {
      throw new Error(`${id} needs at least one of cumulative, gravity, or catchment`);
    }
    const outputs = [props.cumulative, props.gravity, catchment?.output, catchment?.ratios];
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns one `scores` node when `cumulative` or `gravity` is requested, and two catchment nodes
   * (`catchment-ratios`, `catchment-scores`) for 2SFCA.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, nodeCount} = this;
    const {catchment} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getInputs(props),
      props.cumulative,
      props.gravity,
      catchment?.output,
      catchment?.ratios
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (props.cumulative || props.gravity) {
      nodes.push(
        this.orientation === 'opportunity-rows'
          ? createAccessibilityColumnGatherNode<Parameters>(graph, {
              id: `${id}-scores`,
              operation: OPERATION,
              rowCount,
              columnCount: nodeCount,
              costs: props.costs,
              rowWeights: props.opportunityWeights,
              parameters: props.parameters,
              cumulative: props.cumulative,
              gravity: props.gravity
            })
          : createAccessibilityRowReduceNode<Parameters>(graph, {
              id: `${id}-scores`,
              operation: OPERATION,
              variant: 'row-reduce',
              rowCount,
              columnCount: nodeCount,
              costs: props.costs,
              columnWeights: props.opportunityWeights,
              parameters: props.parameters,
              cumulative: props.cumulative,
              gravity: props.gravity
            })
      );
    }
    if (catchment) {
      const ratios =
        catchment.ratios ??
        createTransientView(graph, `${id}-catchment-ratios`, 'float32', rowCount);
      nodes.push(
        createAccessibilityRowReduceNode<Parameters>(graph, {
          id: `${id}-catchment-ratios`,
          operation: OPERATION,
          variant: 'catchment-ratios',
          rowCount,
          columnCount: nodeCount,
          costs: props.costs,
          columnWeights: catchment.demand,
          parameters: props.parameters,
          rowNumerators: props.opportunityWeights,
          ratios
        }),
        createAccessibilityColumnGatherNode<Parameters>(graph, {
          id: `${id}-catchment-scores`,
          operation: OPERATION,
          rowCount,
          columnCount: nodeCount,
          costs: props.costs,
          rowWeights: ratios,
          parameters: props.parameters,
          gravity: catchment.output
        })
      );
    }
    return nodes;
  }
}

/** Returns every read-only view of an accessibility contributor. */
function getInputs(props: GPUNetworkAccessibilityProps): (GraphDataView | undefined)[] {
  return [props.costs, props.opportunityWeights, props.parameters, props.catchment?.demand];
}
