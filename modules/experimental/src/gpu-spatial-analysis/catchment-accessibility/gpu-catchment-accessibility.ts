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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUCatchmentAccessibility';

/** Floating catchment method of {@link GPUCatchmentAccessibility}. */
export type GPUCatchmentMethod = '2sfca' | '3sfca';

/**
 * Properties for {@link GPUCatchmentAccessibility}.
 *
 * Compile-time: `method`, view lengths and which optional outputs exist. Per-frame: the contents of
 * `supply`, `demand` and both weights.
 */
export type GPUCatchmentAccessibilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'catchment-accessibility'`. */
  id?: string;
  /**
   * `'2sfca'` (Luo and Wang 2003) or `'3sfca'` (Wan, Zou and Sternberg 2012, which splits each
   * demand location's population over the facilities it can reach by selection weights).
   */
  method: GPUCatchmentMethod;
  /** Supply (beds, doctors, capacity) per facility, one row per facility. */
  supply: GraphDataView<'float32'>;
  /** Demand (population) per demand location, one row per demand location. */
  demand: GraphDataView<'float32'>;
  /**
   * Facility-to-demand weights: row `j` lists the demand locations `i` inside facility `j`'s
   * catchment, and `w_ji` is the distance-decay value `f(c_ji)` (1 for the classic binary
   * catchment, a kernel value otherwise). Producers such as `GPUNeighborSearch` write it. Its row
   * count is the facility count and its neighbor IDs index `demand`.
   */
  facilityWeights: GPUSpatialWeights;
  /**
   * Demand-to-facility weights: row `i` lists the facilities `j` that demand location `i` can
   * reach, with weight `w_ij`. Defaults to `facilityWeights`, which is correct when facilities and
   * demand locations are the same set of places and the weights are symmetric (a self-join
   * distance band or kernel, `w_ij = w_ji`). Supply a separate cross weights otherwise, for example
   * the transpose or a second `GPUNeighborSearch` from demand to facilities.
   */
  demandWeights?: GPUSpatialWeights;
  /** Caller-owned accessibility `A_i` per demand location. */
  accessibility: GraphDataView<'float32'>;
  /**
   * Optional caller-owned supply-to-demand ratio per facility: `R_j = S_j / sum_i P_i w_ji` for
   * `'2sfca'`, with `P_i / T_i` in place of `P_i` for `'3sfca'`. 0 when no demand is in range.
   */
  ratios?: GraphDataView<'float32'>;
  /** Optional caller-owned count of facilities each demand location reaches (the row length of the demand weights). */
  reachableFacilities?: GraphDataView<'uint32'>;
};

/**
 * Two-step and three-step floating catchment area accessibility on any {@link GPUSpatialWeights}
 * (Euclidean kernel, network-cost kernel, binary band), generalising the 2SFCA of
 * `GPUNetworkAccessibility`, which reads a dense network cost matrix.
 *
 * With `w` the decay weights, facility supply `S_j` and demand `P_i`:
 * - `'2sfca'`: `R_j = S_j / sum_i P_i w_ji` over the demand in facility `j`'s row (0 when the sum is
 *   0), then `A_i = sum_j w_ij R_j` over the facilities in demand row `i`.
 * - `'3sfca'`: selection weights `G_ij = w_ij / T_i` with `T_i = sum_k w_ik` over demand row `i`.
 *   `R_j = S_j / sum_i P_i G_ij`, `A_i = sum_j G_ij R_j`. Demand locations with `T_i = 0` are skipped.
 *
 * Both methods conserve supply, `sum_i P_i A_i = sum_j S_j` over facilities with demand in range,
 * whenever the demand weights are the transpose of the facility weights, for any decay function.
 *
 * Each output row accumulates its neighbors in slot order in one invocation, so results are
 * deterministic and match a sequential CPU sum within f32 rounding. Neighbor IDs outside the
 * partner row space are skipped.
 */
export class GPUCatchmentAccessibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCatchmentAccessibilityProps;
  /** Number of facilities. */
  readonly facilityCount: number;
  /** Number of demand locations. */
  readonly demandCount: number;

  constructor(props: GPUCatchmentAccessibilityProps) {
    const id = props.id ?? 'catchment-accessibility';
    this.id = id;
    this.props = props;
    if (props.method !== '2sfca' && props.method !== '3sfca') {
      throw new Error(`${id} method must be '2sfca' or '3sfca'`);
    }
    this.facilityCount = validateGPUSpatialWeights(id, props.facilityWeights, 'facilityWeights');
    this.demandCount = props.demandWeights
      ? validateGPUSpatialWeights(id, props.demandWeights, 'demandWeights')
      : this.facilityCount;
    if (!props.demandWeights && props.supply.length !== props.demand.length) {
      throw new Error(`${id} needs demandWeights unless supply and demand have equal length`);
    }
    for (const [name, view] of [
      ['supply', props.supply],
      ['demand', props.demand],
      ['accessibility', props.accessibility],
      ['ratios', props.ratios]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (props.reachableFacilities) {
      validatePackedUint32View(props.reachableFacilities, `${id} reachableFacilities`);
    }
    if (props.supply.length !== this.facilityCount) {
      throw new Error(`${id} supply length must equal the facilityWeights row count`);
    }
    if (props.demand.length !== this.demandCount) {
      throw new Error(`${id} demand length must equal the demandWeights row count`);
    }
    if (props.accessibility.length !== this.demandCount) {
      throw new Error(`${id} accessibility length must equal the demand count`);
    }
    if (props.ratios && props.ratios.length !== this.facilityCount) {
      throw new Error(`${id} ratios length must equal the facility count`);
    }
    if (props.reachableFacilities && props.reachableFacilities.length !== this.demandCount) {
      throw new Error(`${id} reachableFacilities length must equal the demand count`);
    }
    const weightViews = getWeightViews(props);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.accessibility, props.ratios, props.reachableFacilities],
      [props.supply, props.demand, ...weightViews]
    );
    const outputBuffers = [props.accessibility, props.ratios, props.reachableFacilities]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the nodes in order: `selection` (3SFCA only), `ratios`, `scores`, and `reachable` when
   * requested.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, facilityCount, demandCount} = this;
    const {supply, demand, facilityWeights, accessibility} = props;
    const demandWeights = props.demandWeights ?? facilityWeights;
    validateGraphViewsBelongToGraph(id, graph, [
      supply,
      demand,
      accessibility,
      props.ratios,
      props.reachableFacilities,
      ...getWeightViews(props)
    ]);
    const isThreeStep = props.method === '3sfca';
    const ratios =
      props.ratios ?? createTransientView(graph, `${id}-ratios`, 'float32', facilityCount);
    const selection = isThreeStep
      ? createTransientView(graph, `${id}-selection`, 'float32', demandCount)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (selection) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-selection`,
          operation: OPERATION,
          variant: 'selection',
          bindings: [
            {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: demandWeights.weights, type: 'f32', access: 'read'},
            {name: 'selection', view: selection, type: 'f32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: `const PARTNERS: u32 = ${facilityCount}u;`,
          body: `var total = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      total += weights[weightsOffset + slot];
    }
  }
  selection[selectionOffset + index] = total;`
        })
      );
    }
    const selectionBinding = selection
      ? [{name: 'selection', view: selection, type: 'f32' as const, access: 'read' as const}]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ratios`,
        operation: OPERATION,
        variant: 'ratios',
        bindings: [
          {name: 'offsets', view: facilityWeights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: facilityWeights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: facilityWeights.weights, type: 'f32', access: 'read'},
          {name: 'demand', view: demand, type: 'f32', access: 'read'},
          {name: 'supply', view: supply, type: 'f32', access: 'read'},
          ...selectionBinding,
          {name: 'ratios', view: ratios, type: 'f32', access: 'read_write'}
        ],
        invocationCount: facilityCount,
        declarations: `const PARTNERS: u32 = ${demandCount}u;`,
        body: `var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      ${
        isThreeStep
          ? `let total = selection[selectionOffset + partner];
      if (total > 0.0) {
        captured += demand[demandOffset + partner] * weights[weightsOffset + slot] / total;
      }`
          : 'captured += demand[demandOffset + partner] * weights[weightsOffset + slot];'
      }
    }
  }
  ratios[ratiosOffset + index] = select(0.0, supply[supplyOffset + index] / captured, captured > 0.0);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scores`,
        operation: OPERATION,
        variant: 'scores',
        bindings: [
          {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: demandWeights.weights, type: 'f32', access: 'read'},
          {name: 'ratios', view: ratios, type: 'f32', access: 'read'},
          ...selectionBinding,
          {name: 'accessibility', view: accessibility, type: 'f32', access: 'read_write'}
        ],
        invocationCount: demandCount,
        declarations: `const PARTNERS: u32 = ${facilityCount}u;`,
        body: `var score = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      score += weights[weightsOffset + slot] * ratios[ratiosOffset + partner];
    }
  }
  ${
    isThreeStep
      ? `let total = selection[selectionOffset + index];
  score = select(0.0, score / total, total > 0.0);`
      : ''
  }
  accessibility[accessibilityOffset + index] = score;`
      })
    );
    if (props.reachableFacilities) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-reachable`,
          operation: OPERATION,
          variant: 'reachable',
          bindings: [
            {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'reachable', view: props.reachableFacilities, type: 'u32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: `const PARTNERS: u32 = ${facilityCount}u;`,
          body: `var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      count++;
    }
  }
  reachable[reachableOffset + index] = count;`
        })
      );
    }
    return nodes;
  }
}

/** Returns every weights view read by a catchment contributor. */
function getWeightViews(props: GPUCatchmentAccessibilityProps): GraphDataView[] {
  const views: GraphDataView[] = [];
  for (const weights of [props.facilityWeights, props.demandWeights]) {
    if (weights) {
      views.push(weights.offsets, weights.neighbors, weights.weights);
    }
  }
  return views;
}
