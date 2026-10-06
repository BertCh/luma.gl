// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';
import {
  createCSRStages,
  getContainsFunctionWGSL,
  getMergeVisitWGSL,
  type CSRStageAttribute
} from './spatial-weights-algebra-wgsl';

const OPERATION = 'GPUSpatialWeightsAlgebra';

/** Maximum `order` of the `'higherOrder'` operation (one expansion level is added to the graph per order). */
export const GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER = 32;

/**
 * How `'union'` and `'intersection'` combine the weights of a neighbor that both operands list.
 * A neighbor listed by one operand only keeps that operand's weight.
 *
 * - `'left'` (default): the weight of `left`.
 * - `'right'`: the weight of `right`.
 * - `'sum'`, `'min'`, `'max'`, `'product'`: the named combination of the two weights.
 * - `'binary'`: every output weight is 1 for every operation, including `difference` and
 *   `symmetricDifference`. This is what libpysal 4.15 `Graph.union/intersection/difference/
 *   symmetric_difference` return (verified: they ignore the operands' weights and give 1.0).
 */
export type GPUSpatialWeightsCombineRule =
  | 'left'
  | 'right'
  | 'sum'
  | 'min'
  | 'max'
  | 'product'
  | 'binary';

/** Properties shared by every {@link GPUSpatialWeightsAlgebra} operation. */
export type GPUSpatialWeightsAlgebraBaseProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-weights-algebra'`. */
  id?: string;
  /**
   * Destination CSR with `rows + 1` offsets and the result capacity (`neighbors.length`). It must
   * not share buffers with the inputs. `output.distances` is written only when present, and is
   * not supported by `'higherOrder'` and `'block'` (they have no geometric distances).
   */
  output: GPUSpatialWeights;
  /**
   * One-element uint32 flag. Set to 1 when the result needs more than the output capacity, else
   * 0. On overflow the output is still a valid CSR: offsets are clamped to the capacity, so the
   * leading rows are complete, the row in which the capacity runs out is a sorted prefix and the
   * following rows are empty.
   */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-element uint32 view receiving the slot count the final result needs, even past the capacity. */
  totalNeighbors?: GraphDataView<'uint32'>;
};

/**
 * Set operation of two weights on the same row set. Row `i` of the result is the sorted merge of
 * row `i` of `left` and `right`, keeping the neighbor IDs that satisfy the operation:
 * - `'union'`: IDs in either operand.
 * - `'intersection'`: IDs in both.
 * - `'difference'`: IDs in `left` but not `right` (weights from `left`).
 * - `'symmetricDifference'`: IDs in exactly one operand (weights from the operand that has it).
 *
 * Distances come from `left` where it lists the neighbor, else from `right`.
 */
export type GPUSpatialWeightsBinaryProps = GPUSpatialWeightsAlgebraBaseProps & {
  operation: 'union' | 'intersection' | 'difference' | 'symmetricDifference';
  /** Left operand. */
  left: GPUSpatialWeights;
  /** Right operand, with the same row count as `left`. */
  right: GPUSpatialWeights;
  /** Weight rule for neighbors both operands list (only `'union'` and `'intersection'`). Defaults to `'left'`. */
  weightRule?: GPUSpatialWeightsCombineRule;
};

/**
 * Exactly-`order`-step neighbors (libpysal `Graph.higher_order`): `j` is in row `i` when the
 * shortest directed path from `i` to `j` in `weights` has exactly `order` steps. Row `i` itself is
 * never listed, even when `weights` lists self weights or `j` can return to `i`. Every weight is 1.
 * With `cumulative` the result lists all neighbors within `order` steps (libpysal `lower_order=True`).
 *
 * Each level expands the previous frontier through `weights` and drops IDs seen at lower orders,
 * then insertion-sorts the row. Work grows with the squared frontier degree, which suits the
 * sparse contiguity and k-nearest-neighbor weights this entry produces, not dense graphs.
 */
export type GPUSpatialWeightsHigherOrderProps = GPUSpatialWeightsAlgebraBaseProps & {
  operation: 'higherOrder';
  /** Base weights (square). Only the sparsity pattern is used. */
  weights: GPUSpatialWeights;
  /** Step count, an integer in `[1, GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER]`. */
  order: number;
  /** Include all lower orders too. Defaults to false. */
  cumulative?: boolean;
  /**
   * Slot capacity of the intermediate frontier and visited sets. Defaults to the output capacity.
   * Exact lower-order sets can be larger than an exact-`order` output, so raise it when `overflow`
   * is set although the final result would fit. `overflow` also reports intermediate overflow.
   */
  workCapacity?: number;
};

/**
 * Adds or replaces the self entry of every row (the diagonal `w_ii`). This relaxes the
 * `w_ii = 0` invariant of {@link GPUSpatialWeights}, see the class documentation for which
 * consumers accept the result. The self distance is 0.
 */
export type GPUSpatialWeightsSelfWeightProps = GPUSpatialWeightsAlgebraBaseProps & {
  operation: 'selfWeight';
  /** Weights to extend. */
  weights: GPUSpatialWeights;
  /**
   * Self weight: one non-negative finite number for every row, or a per-row float32 view (so it
   * can change between encodings without recompiling). A row that already lists itself has that
   * weight replaced.
   */
  selfWeight: number | GraphDataView<'float32'>;
};

/**
 * Restricts weights to the rows selected by `mask` (libpysal `Graph.subgraph`). Row and neighbor
 * IDs are not renumbered: a masked-out row becomes empty and every slot pointing to a masked-out
 * row is dropped, so the result still indexes the same row space. Renumber with `GPUCompaction`
 * when a dense subset is needed.
 */
export type GPUSpatialWeightsSubgraphProps = GPUSpatialWeightsAlgebraBaseProps & {
  operation: 'subgraph';
  /** Weights to restrict. */
  weights: GPUSpatialWeights;
  /** One uint32 per row; nonzero keeps the row. */
  mask: GraphDataView<'uint32'>;
};

/**
 * Block weights from group IDs (libpysal `Graph.build_block_contiguity`): row `i` lists every
 * other row with the same group ID, with weight 1. Rows whose ID is `>= groupCount` have no group
 * and stay empty. The output size is the sum of squared group sizes minus the row count.
 */
export type GPUSpatialWeightsBlockProps = GPUSpatialWeightsAlgebraBaseProps & {
  operation: 'block';
  /** Group ID per row, one uint32 each. The row count is `groupIds.length`. */
  groupIds: GraphDataView<'uint32'>;
  /** Group IDs are in `[0, groupCount)`; larger values mean no group. */
  groupCount: number;
};

/** Properties for {@link GPUSpatialWeightsAlgebra}. */
export type GPUSpatialWeightsAlgebraProps =
  | GPUSpatialWeightsBinaryProps
  | GPUSpatialWeightsHigherOrderProps
  | GPUSpatialWeightsSelfWeightProps
  | GPUSpatialWeightsSubgraphProps
  | GPUSpatialWeightsBlockProps;

const COMBINE_RULES: Record<GPUSpatialWeightsCombineRule, string> = {
  left: 'leftValue',
  right: 'rightValue',
  sum: 'leftValue + rightValue',
  min: 'min(leftValue, rightValue)',
  max: 'max(leftValue, rightValue)',
  product: 'leftValue * rightValue',
  binary: '1.0'
};

/**
 * Algebra on {@link GPUSpatialWeights}: set operations, higher orders, self weights, subgraphs and
 * block weights, producing a new CSR on the GPU with a bounded output and an overflow flag. This
 * is the libpysal `Graph` algebra (`union`, `intersection`, `difference`, `symmetric_difference`,
 * `higher_order`, `subgraph`, `build_block_contiguity`, `fill_diagonal`) over our CSR.
 *
 * Invariants preserved in the output: `offsets` non-decreasing from 0 with
 * `offsets[rows] <= capacity`, strictly ascending neighbor IDs per row (no duplicates), finite
 * non-negative weights. Binary operations are per-row two-pointer merges, so they are
 * deterministic and need both operands to satisfy those invariants.
 *
 * Weight rules: libpysal 4.15 set operations return weight 1 for every link (use
 * `weightRule: 'binary'` to match). By default we keep information instead: `union` and
 * `intersection` keep the left operand's weight where both list a link (see `weightRule`),
 * `difference` keeps left weights, `symmetricDifference` keeps the weight of the operand that has
 * the link. `higherOrder` and `block` give weight 1 (as libpysal `higher_order`); `subgraph`
 * keeps weights unchanged. For a binary operation the left operand's row count defines the row
 * space; both operands need the same count.
 *
 * Self weights: `'selfWeight'` writes `w_ii != 0`, which relaxes the `w_ii = 0` invariant. Consumers
 * that sum over slots keep the self term and accept it: `GPUSpatialLag` (a lag with the self
 * weight gives Getis-Ord G* style sums), `GPUSpatialWeightsTransform` (row, binary, kernel,
 * symmetrize, double, variance) and this class's own operations. The permutation tests,
 * `GPUGlobalSpatialStatistics` and the local spatial-autocorrelation kernels skip `neighbor ==
 * row` entries by design, so passing self weights to them silently ignores the diagonal.
 *
 * Capacity: row counts run first, then an exclusive scan and clamped offsets, so an operation
 * that fits reproduces the CPU result exactly and an overflowing one is flagged and truncated.
 */
export class GPUSpatialWeightsAlgebra implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsAlgebraProps;

  constructor(props: GPUSpatialWeightsAlgebraProps) {
    const id = props.id ?? 'spatial-weights-algebra';
    this.id = id;
    this.props = props;
    const outputRows = validateGPUSpatialWeights(id, props.output, 'output');
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    if (props.totalNeighbors) {
      validatePackedUint32View(props.totalNeighbors, `${id} totalNeighbors`);
      if (props.totalNeighbors.length < 1) {
        throw new Error(`${id} totalNeighbors must hold one uint32`);
      }
    }
    const inputs = this.getInputWeights();
    const checkRows = (weights: GPUSpatialWeights, name: string) => {
      const rows = validateGPUSpatialWeights(id, weights, name);
      if (rows !== outputRows) {
        throw new Error(`${id} ${name}.offsets length must equal output.offsets length`);
      }
    };
    switch (props.operation) {
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        checkRows(props.left, 'left');
        checkRows(props.right, 'right');
        if (props.weightRule && !(props.weightRule in COMBINE_RULES)) {
          throw new Error(`${id} unknown weightRule ${props.weightRule}`);
        }
        if (props.output.distances) {
          const needsRight =
            props.operation === 'union' || props.operation === 'symmetricDifference';
          if (!props.left.distances || (needsRight && !props.right.distances)) {
            throw new Error(
              `${id} output.distances requires distances on the operands that contribute`
            );
          }
        }
        break;
      case 'higherOrder':
        checkRows(props.weights, 'weights');
        if (
          !Number.isInteger(props.order) ||
          props.order < 1 ||
          props.order > GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER
        ) {
          throw new Error(
            `${id} order must be an integer in [1, ${GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER}]`
          );
        }
        if (
          props.workCapacity !== undefined &&
          !(Number.isInteger(props.workCapacity) && props.workCapacity >= 1)
        ) {
          throw new Error(`${id} workCapacity must be a positive integer`);
        }
        break;
      case 'selfWeight':
        checkRows(props.weights, 'weights');
        if (typeof props.selfWeight === 'number') {
          if (!(Number.isFinite(props.selfWeight) && props.selfWeight >= 0)) {
            throw new Error(`${id} selfWeight must be a non-negative finite number`);
          }
        } else if (props.selfWeight.length !== outputRows) {
          throw new Error(`${id} selfWeight view length must equal the row count`);
        }
        break;
      case 'subgraph':
        checkRows(props.weights, 'weights');
        validatePackedUint32View(props.mask, `${id} mask`);
        if (props.mask.length !== outputRows) {
          throw new Error(`${id} mask length must equal the row count`);
        }
        break;
      case 'block':
        validatePackedUint32View(props.groupIds, `${id} groupIds`);
        if (props.groupIds.length !== outputRows) {
          throw new Error(`${id} groupIds length must equal the output row count`);
        }
        if (!(Number.isInteger(props.groupCount) && props.groupCount >= 1)) {
          throw new Error(`${id} groupCount must be a positive integer`);
        }
        break;
      default:
        throw new Error(
          `${id} operation must be one of union, intersection, difference, symmetricDifference, higherOrder, selfWeight, subgraph or block`
        );
    }
    if (
      props.output.distances &&
      (props.operation === 'higherOrder' || props.operation === 'block')
    ) {
      throw new Error(`${id} '${props.operation}' has no distances: omit output.distances`);
    }
    if (
      props.output.distances &&
      (props.operation === 'selfWeight' || props.operation === 'subgraph') &&
      !props.weights.distances
    ) {
      throw new Error(`${id} output.distances requires weights.distances`);
    }
    const outputViews = [
      props.output.offsets,
      props.output.neighbors,
      props.output.weights,
      props.output.distances,
      props.overflow,
      props.totalNeighbors
    ];
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      ...inputs.flatMap(getViews),
      props.operation === 'subgraph' ? props.mask : undefined,
      props.operation === 'block' ? props.groupIds : undefined,
      props.operation === 'selfWeight' && typeof props.selfWeight !== 'number'
        ? props.selfWeight
        : undefined
    ]);
  }

  /** Returns every input weights structure. */
  private getInputWeights(): GPUSpatialWeights[] {
    const {props} = this;
    switch (props.operation) {
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        return [props.left, props.right];
      case 'higherOrder':
      case 'selfWeight':
      case 'subgraph':
        return [props.weights];
      default:
        return [];
    }
  }

  /** Returns the algebra nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.getInputWeights().flatMap(getViews),
      ...getViews(props.output),
      props.overflow,
      props.totalNeighbors,
      props.operation === 'subgraph' ? props.mask : undefined,
      props.operation === 'block' ? props.groupIds : undefined,
      props.operation === 'selfWeight' && typeof props.selfWeight !== 'number'
        ? props.selfWeight
        : undefined
    ]);
    switch (props.operation) {
      case 'higherOrder':
        return this.getHigherOrderNodes(graph, props);
      case 'selfWeight':
        return this.getSelfWeightNodes(graph, props);
      case 'subgraph':
        return this.getSubgraphNodes(graph, props);
      case 'block':
        return this.getBlockNodes(graph, props);
      default:
        return this.getBinaryNodes(graph, props);
    }
  }

  private getBinaryNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUSpatialWeightsBinaryProps
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const {left, right, output, operation} = props;
    const rows = output.offsets.length - 1;
    const rule = props.weightRule ?? 'left';
    const bindings = [
      {name: 'leftOffsets', view: left.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'leftNeighbors', view: left.neighbors, type: 'u32' as const, access: 'read' as const},
      {name: 'rightOffsets', view: right.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'rightNeighbors', view: right.neighbors, type: 'u32' as const, access: 'read' as const}
    ];
    const attributes: CSRStageAttribute[] = [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {
        name: 'weights',
        output: output.weights,
        type: 'f32',
        value: 'combineWeight(leftSlot, rightSlot)',
        bindings: [
          {name: 'leftWeights', view: left.weights, type: 'f32', access: 'read'},
          {name: 'rightWeights', view: right.weights, type: 'f32', access: 'read'}
        ],
        declarations: `fn combineWeight(leftSlot: u32, rightSlot: u32) -> f32 {
  ${rule === 'binary' ? 'return 1.0;' : ''}
  if (rightSlot == NO_SLOT) {
    return leftWeights[leftWeightsOffset + leftSlot];
  }
  if (leftSlot == NO_SLOT) {
    return rightWeights[rightWeightsOffset + rightSlot];
  }
  let leftValue = leftWeights[leftWeightsOffset + leftSlot];
  let rightValue = rightWeights[rightWeightsOffset + rightSlot];
  return ${COMBINE_RULES[rule]};
}`
      }
    ];
    if (output.distances) {
      const rightDistances =
        right.distances && operation !== 'intersection' && operation !== 'difference';
      attributes.push({
        name: 'distances',
        output: output.distances,
        type: 'f32',
        value: 'pickDistance(leftSlot, rightSlot)',
        bindings: [
          {name: 'leftDistances', view: left.distances!, type: 'f32', access: 'read'},
          ...(rightDistances
            ? [
                {
                  name: 'rightDistances',
                  view: right.distances!,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `fn pickDistance(leftSlot: u32, rightSlot: u32) -> f32 {
  if (leftSlot != NO_SLOT) {
    return leftDistances[leftDistancesOffset + leftSlot];
  }
  ${rightDistances ? 'return rightDistances[rightDistancesOffset + rightSlot];' : 'return 0.0;'}
}`
      });
    }
    return createCSRStages(graph, {
      id,
      operation: OPERATION,
      rows,
      targetOffsets: output.offsets,
      capacity: output.neighbors.length,
      overflow: props.overflow,
      initializesOverflow: true,
      totalNeighbors: props.totalNeighbors,
      bindings,
      visit: onItem => getMergeVisitWGSL(operation, onItem),
      attributes
    });
  }

  private getHigherOrderNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUSpatialWeightsHigherOrderProps
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const {weights, output, order} = props;
    const cumulative = props.cumulative ?? false;
    const rows = output.offsets.length - 1;
    const workCapacity = props.workCapacity ?? output.neighbors.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    type Pattern = {offsets: GraphDataView<'uint32'>; neighbors: GraphDataView<'uint32'>};
    const createPattern = (name: string): Pattern => ({
      offsets: createTransientView(graph, `${id}-${name}-offsets`, 'uint32', rows + 1),
      neighbors: createTransientView(graph, `${id}-${name}-neighbors`, 'uint32', workCapacity)
    });
    const outputPattern: Pattern = {offsets: output.offsets, neighbors: output.neighbors};
    let initializesOverflow = true;
    const stageBase = (name: string, target: Pattern, last: boolean) => {
      const stage = {
        id: `${id}-${name}`,
        operation: OPERATION,
        rows,
        targetOffsets: target.offsets,
        capacity: target.neighbors.length,
        overflow: props.overflow,
        initializesOverflow,
        totalNeighbors: last ? props.totalNeighbors : undefined,
        attributes: [
          {name: 'neighbors', output: target.neighbors, type: 'u32' as const, value: 'id'}
        ]
      };
      initializesOverflow = false;
      return stage;
    };
    const readBinding = (name: string, view: GraphDataView) => ({
      name,
      view,
      type: 'u32' as const,
      access: 'read' as const
    });

    // Level 1: the pattern of `weights` without the diagonal.
    const first = order === 1 ? outputPattern : createPattern('exact-1');
    nodes.push(
      ...createCSRStages(graph, {
        ...stageBase('exact-1', first, order === 1),
        bindings: [
          readBinding('baseOffsets', weights.offsets),
          readBinding('baseNeighbors', weights.neighbors)
        ],
        visit:
          onItem => `for (var slot = baseOffsets[baseOffsetsOffset + index]; slot < baseOffsets[baseOffsetsOffset + index + 1u]; slot++) {
    let id = baseNeighbors[baseNeighborsOffset + slot];
    if (id != index && id < ROWS) {
      ${onItem}
    }
  }`
      })
    );
    let frontier = first;
    let visited = first;
    for (let level = 2; level <= order; level++) {
      const isLast = level === order;
      const frontierTarget =
        isLast && !cumulative ? outputPattern : createPattern(`exact-${level}`);
      nodes.push(
        ...createCSRStages(graph, {
          ...stageBase(`exact-${level}`, frontierTarget, isLast && !cumulative),
          bindings: [
            readBinding('baseOffsets', weights.offsets),
            readBinding('baseNeighbors', weights.neighbors),
            readBinding('frontierOffsets', frontier.offsets),
            readBinding('frontierNeighbors', frontier.neighbors),
            readBinding('visitedOffsets', visited.offsets),
            readBinding('visitedNeighbors', visited.neighbors)
          ],
          declarations: `${getContainsFunctionWGSL('baseContains', 'baseNeighbors')}
${getContainsFunctionWGSL('visitedContains', 'visitedNeighbors')}`,
          // Candidates c of row i come from the frontier rows. Keep c when it is not i, not
          // visited at a lower order, and not already produced by an earlier frontier neighbor.
          visit: onItem => `let frontierBegin = frontierOffsets[frontierOffsetsOffset + index];
  let frontierEnd = frontierOffsets[frontierOffsetsOffset + index + 1u];
  let visitedBegin = visitedOffsets[visitedOffsetsOffset + index];
  let visitedEnd = visitedOffsets[visitedOffsetsOffset + index + 1u];
  for (var frontierSlot = frontierBegin; frontierSlot < frontierEnd; frontierSlot++) {
    let middle = frontierNeighbors[frontierNeighborsOffset + frontierSlot];
    if (middle >= ROWS) {
      continue;
    }
    for (var slot = baseOffsets[baseOffsetsOffset + middle]; slot < baseOffsets[baseOffsetsOffset + middle + 1u]; slot++) {
      let id = baseNeighbors[baseNeighborsOffset + slot];
      if (id == index || id >= ROWS || visitedContains(visitedBegin, visitedEnd, id)) {
        continue;
      }
      var seen = false;
      for (var earlier = frontierBegin; earlier < frontierSlot; earlier++) {
        let earlierMiddle = frontierNeighbors[frontierNeighborsOffset + earlier];
        if (earlierMiddle < ROWS && baseContains(baseOffsets[baseOffsetsOffset + earlierMiddle], baseOffsets[baseOffsetsOffset + earlierMiddle + 1u], id)) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        ${onItem}
      }
    }
  }`,
          sortRows: true
        })
      );
      if (isLast && !cumulative) {
        frontier = frontierTarget;
        break;
      }
      const visitedTarget = isLast ? outputPattern : createPattern(`visited-${level}`);
      nodes.push(
        ...createCSRStages(graph, {
          ...stageBase(`visited-${level}`, visitedTarget, isLast),
          bindings: [
            readBinding('leftOffsets', visited.offsets),
            readBinding('leftNeighbors', visited.neighbors),
            readBinding('rightOffsets', frontierTarget.offsets),
            readBinding('rightNeighbors', frontierTarget.neighbors)
          ],
          visit: onItem => getMergeVisitWGSL('union', onItem)
        })
      );
      frontier = frontierTarget;
      visited = visitedTarget;
    }
    nodes.push(createFillWeightsNode(graph, `${id}-weights`, output));
    return nodes;
  }

  private getSelfWeightNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUSpatialWeightsSelfWeightProps
  ): GPUCommandNode<Parameters>[] {
    const {weights, output, selfWeight} = props;
    const rows = output.offsets.length - 1;
    const selfValue =
      typeof selfWeight === 'number'
        ? getWGSLFloatLiteral(selfWeight)
        : 'selfWeights[selfWeightsOffset + index]';
    const attributes: CSRStageAttribute[] = [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {
        name: 'weights',
        output: output.weights,
        type: 'f32',
        value: `select(sourceWeights[sourceWeightsOffset + sourceSlot], ${selfValue}, id == index)`,
        bindings: [
          {name: 'sourceWeights', view: weights.weights, type: 'f32', access: 'read'},
          ...(typeof selfWeight === 'number'
            ? []
            : [
                {
                  name: 'selfWeights',
                  view: selfWeight,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ])
        ]
      }
    ];
    if (output.distances) {
      attributes.push({
        name: 'distances',
        output: output.distances,
        type: 'f32',
        value: 'select(sourceDistances[sourceDistancesOffset + sourceSlot], 0.0, id == index)',
        bindings: [{name: 'sourceDistances', view: weights.distances!, type: 'f32', access: 'read'}]
      });
    }
    return createCSRStages(graph, {
      id: this.id,
      operation: OPERATION,
      rows,
      targetOffsets: output.offsets,
      capacity: output.neighbors.length,
      overflow: props.overflow,
      initializesOverflow: true,
      totalNeighbors: props.totalNeighbors,
      bindings: [
        {name: 'sourceOffsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'sourceNeighbors', view: weights.neighbors, type: 'u32', access: 'read'}
      ],
      // The source slot of the inserted or replaced self entry is clamped to a valid slot by
      // `select`; the value is only read for `id != index`.
      visit: onItem => `var inserted = false;
  for (var slot = sourceOffsets[sourceOffsetsOffset + index]; slot < sourceOffsets[sourceOffsetsOffset + index + 1u]; slot++) {
    let neighbor = sourceNeighbors[sourceNeighborsOffset + slot];
    if (!inserted && neighbor >= index) {
      inserted = true;
      {
        let id = index;
        let sourceSlot = slot;
        ${onItem}
      }
      if (neighbor == index) {
        continue;
      }
    }
    {
      let id = neighbor;
      let sourceSlot = slot;
      ${onItem}
    }
  }
  if (!inserted) {
    let id = index;
    let sourceSlot = 0u;
    ${onItem}
  }`,
      attributes
    });
  }

  private getSubgraphNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUSpatialWeightsSubgraphProps
  ): GPUCommandNode<Parameters>[] {
    const {weights, output, mask} = props;
    const rows = output.offsets.length - 1;
    const attributes: CSRStageAttribute[] = [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {
        name: 'weights',
        output: output.weights,
        type: 'f32',
        value: 'sourceWeights[sourceWeightsOffset + slot]',
        bindings: [{name: 'sourceWeights', view: weights.weights, type: 'f32', access: 'read'}]
      }
    ];
    if (output.distances) {
      attributes.push({
        name: 'distances',
        output: output.distances,
        type: 'f32',
        value: 'sourceDistances[sourceDistancesOffset + slot]',
        bindings: [{name: 'sourceDistances', view: weights.distances!, type: 'f32', access: 'read'}]
      });
    }
    return createCSRStages(graph, {
      id: this.id,
      operation: OPERATION,
      rows,
      targetOffsets: output.offsets,
      capacity: output.neighbors.length,
      overflow: props.overflow,
      initializesOverflow: true,
      totalNeighbors: props.totalNeighbors,
      bindings: [
        {name: 'sourceOffsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'sourceNeighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'mask', view: mask, type: 'u32', access: 'read'}
      ],
      visit: onItem => `if (mask[maskOffset + index] != 0u) {
    for (var slot = sourceOffsets[sourceOffsetsOffset + index]; slot < sourceOffsets[sourceOffsetsOffset + index + 1u]; slot++) {
      let id = sourceNeighbors[sourceNeighborsOffset + slot];
      if (id < ROWS && mask[maskOffset + id] != 0u) {
        ${onItem}
      }
    }
  }`,
      attributes
    });
  }

  private getBlockNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUSpatialWeightsBlockProps
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const {output, groupIds, groupCount} = props;
    const rows = output.offsets.length - 1;
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', rows);
    const sortRowIds = createTransientView(graph, `${id}-sort-rows`, 'uint32', rows);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-prepare`,
        operation: OPERATION,
        variant: 'sort-prepare',
        bindings: [
          {name: 'groupIds', view: groupIds, type: 'u32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortRows', view: sortRowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const GROUP_COUNT: u32 = ${groupCount}u;`,
        body: `sortKeys[sortKeysOffset + index] = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortRows[sortRowsOffset + index] = index;`
      }),
      // The sort is stable, so each group lists its rows in ascending ID order.
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortRowIds,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits: getSortKeyBits(groupCount)
      }).getCommandNodes(graph),
      ...createCSRStages(graph, {
        id,
        operation: OPERATION,
        rows,
        targetOffsets: output.offsets,
        capacity: output.neighbors.length,
        overflow: props.overflow,
        initializesOverflow: true,
        totalNeighbors: props.totalNeighbors,
        bindings: [
          {name: 'groupIds', view: groupIds, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'}
        ],
        declarations: `const GROUP_COUNT: u32 = ${groupCount}u;
fn lowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ${rows}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
        visit: onItem => `let group = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  if (group < GROUP_COUNT) {
    let groupEnd = lowerBound(group + 1u);
    for (var position = lowerBound(group); position < groupEnd; position++) {
      let id = sortedRows[sortedRowsOffset + position];
      if (id != index) {
        ${onItem}
      }
    }
  }`,
        attributes: [
          {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
          {name: 'weights', output: output.weights, type: 'f32', value: '1.0'}
        ]
      })
    ];
  }
}

function getViews(weights: GPUSpatialWeights): (GraphDataView | undefined)[] {
  return [weights.offsets, weights.neighbors, weights.weights, weights.distances];
}

/** Writes weight 1 into every used slot of `output`. */
function createFillWeightsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  output: GPUSpatialWeights
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'weights',
    bindings: [
      {name: 'offsets', view: output.offsets, type: 'u32', access: 'read'},
      {name: 'weights', view: output.weights, type: 'f32', access: 'read_write'}
    ],
    invocationCount: output.offsets.length - 1,
    body: `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    weights[weightsOffset + slot] = 1.0;
  }`
  });
}
