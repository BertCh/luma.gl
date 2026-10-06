// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from './spatial-join-types';

const OPERATION = 'GPUNearestFeatureWeights';

/** Largest per-query slot count of the nearest join (`neighborCapacity` is at most 64). */
const MAXIMUM_SLOTS = 64;

/**
 * Properties for {@link GPUNearestFeatureWeights}.
 *
 * The `neighbor*` views are the outputs of a {@link GPUNearestFeatureJoin} in neighbors mode,
 * passed with the join's `neighborCapacity` as `slotCapacity`. Compile-time: every prop except the
 * contents of the input views.
 */
export type GPUNearestFeatureWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-weights'`. */
  id?: string;
  /** `neighborIds` of the join: `queryCount * slotCapacity` feature IDs, padded with `GPU_SPATIAL_JOIN_NO_FEATURE`. */
  neighborIds: GraphDataView<'uint32'>;
  /** `neighborCounts` of the join: filled slots per query. Its length is the row count. */
  neighborCounts: GraphDataView<'uint32'>;
  /**
   * `neighborDistances` of the join. Required for `'inverse-distance'` weights and for
   * `weights.distances`.
   */
  neighborDistances?: GraphDataView<'float32'>;
  /** The join's `neighborCapacity`: slots per query in the input layout. An integer in `[1, 64]`. */
  slotCapacity: number;
  /**
   * Neighbors kept per row, taken in the join's `(distance, feature row)` order. Defaults to
   * `slotCapacity`, or to `slotCapacity - 1` with `excludeSelf`. Rows with fewer valid slots keep
   * fewer (no padding entries are written).
   */
  k?: number;
  /**
   * Self-join: drop the slot whose feature ID equals the query row (libpysal `KNN` never lists a
   * row as its own neighbor). Run the join with `k + 1` neighbors so `k` remain. Valid when the
   * join's feature IDs are feature rows (no `featureIds`), or IDs that equal the query row index.
   * Defaults to false (cross weights: query rows to feature IDs).
   */
  excludeSelf?: boolean;
  /**
   * Weight rule. `'binary'`: 1 per neighbor (libpysal `KNN` default). `'inverse-distance'`:
   * `1 / max(d, distanceFloor)^power`. Defaults to `'binary'`.
   */
  weightType?: 'binary' | 'inverse-distance';
  /** Exponent of the inverse-distance rule. A positive finite number. Defaults to 1. */
  power?: number;
  /**
   * Lower bound on the distance in the inverse-distance rule, so coincident features get a finite
   * weight (libpysal would divide by zero). A positive finite number. Defaults to `1e-6`.
   */
  distanceFloor?: number;
  /**
   * Caller-owned output CSR with `neighborCounts.length + 1` offsets. Neighbor IDs ascend within
   * each row. `distances` is filled when present. The capacity (`neighbors.length`) must be at least
   * `1`; `rows * (k)` always suffices.
   */
  weights: GPUSpatialWeights;
  /** Caller-owned one-row flag: 1 when the kept neighbors did not fit `weights` capacity. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Adapter from the k-nearest output of {@link GPUNearestFeatureJoin} to a {@link GPUSpatialWeights}
 * CSR: libpysal 4.15 `KNN` / `Graph.build_knn` for features of any kind (points, lines, polygons),
 * with the join's exact geometry distances.
 *
 * Per row it keeps the first `k` valid slots in the join's `(distance, feature row)` order, drops
 * padding (`GPU_SPATIAL_JOIN_NO_FEATURE`, slots past `neighborCounts`) and, with `excludeSelf`, the
 * row's own ID, then sorts the kept IDs ascending as the CSR invariants require. Rows with fewer
 * than `k` available neighbors are simply shorter. The result is deterministic and exact.
 *
 * The weights are directed: a nearest-neighbor relation is generally not symmetric, so this
 * output does not meet the symmetric-pattern requirement of the `'symmetrize'` transform. Take the
 * union of the weights with their transpose (`GPUSpatialWeightsAlgebra` `'union'` with
 * `GPUSpatialWeightsTranspose`) first. Deviation from libpysal: binary weights are 1 and
 * inverse-distance weights floor the distance instead of dividing by zero.
 */
export class GPUNearestFeatureWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNearestFeatureWeightsProps;
  /** Number of rows (queries). */
  readonly rows: number;
  /** Resolved neighbors kept per row. */
  readonly k: number;

  constructor(props: GPUNearestFeatureWeightsProps) {
    const id = props.id ?? 'nearest-feature-weights';
    this.id = id;
    this.props = props;
    const {slotCapacity} = props;
    if (!Number.isInteger(slotCapacity) || slotCapacity < 1 || slotCapacity > MAXIMUM_SLOTS) {
      throw new Error(`${id} slotCapacity must be an integer in [1, ${MAXIMUM_SLOTS}]`);
    }
    this.k = props.k ?? (props.excludeSelf ? Math.max(slotCapacity - 1, 1) : slotCapacity);
    if (!Number.isInteger(this.k) || this.k < 1 || this.k > slotCapacity) {
      throw new Error(`${id} k must be an integer in [1, slotCapacity]`);
    }
    validatePackedUint32View(props.neighborIds, `${id} neighborIds`);
    validatePackedUint32View(props.neighborCounts, `${id} neighborCounts`);
    this.rows = props.neighborCounts.length;
    if (this.rows < 1) {
      throw new Error(`${id} neighborCounts must hold at least one row`);
    }
    if (props.neighborIds.length !== this.rows * slotCapacity) {
      throw new Error(`${id} neighborIds length must equal neighborCounts.length * slotCapacity`);
    }
    const weightType = props.weightType ?? 'binary';
    if (weightType !== 'binary' && weightType !== 'inverse-distance') {
      throw new Error(`${id} weightType must be 'binary' or 'inverse-distance'`);
    }
    if (props.neighborDistances) {
      validatePackedView(props.neighborDistances, ['float32'], `${id} neighborDistances`);
      if (props.neighborDistances.length !== props.neighborIds.length) {
        throw new Error(`${id} neighborDistances length must equal neighborIds length`);
      }
    }
    if (
      (weightType === 'inverse-distance' || props.weights.distances) &&
      !props.neighborDistances
    ) {
      throw new Error(`${id} neighborDistances is required for inverse-distance and distances`);
    }
    for (const [name, value] of [
      ['power', props.power],
      ['distanceFloor', props.distanceFloor]
    ] as const) {
      if (value !== undefined && !(Number.isFinite(value) && value > 0)) {
        throw new Error(`${id} ${name} must be a positive finite number`);
      }
    }
    if (validateGPUSpatialWeights(id, props.weights) !== this.rows) {
      throw new Error(`${id} weights.offsets length must equal neighborCounts.length + 1`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.distances,
        props.overflow
      ],
      [props.neighborIds, props.neighborCounts, props.neighborDistances]
    );
  }

  /** Returns the adapter nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows, k} = this;
    const {weights, slotCapacity} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.neighborIds,
      props.neighborCounts,
      props.neighborDistances,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      props.overflow
    ]);
    const capacity = weights.neighbors.length;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rows);
    const inverse = (props.weightType ?? 'binary') === 'inverse-distance';
    const declarations = `const SLOTS: u32 = ${slotCapacity}u;
const KEEP: u32 = ${k}u;
const EXCLUDE_SELF: bool = ${props.excludeSelf === true};
const NO_FEATURE: u32 = ${GPU_SPATIAL_JOIN_NO_FEATURE}u;
const POWER: f32 = ${getWGSLFloatLiteral(props.power ?? 1)};
const FLOOR: f32 = ${getWGSLFloatLiteral(props.distanceFloor ?? 1e-6)};`;
    const inputBindings = [
      {name: 'ids', view: props.neighborIds, type: 'u32' as const, access: 'read' as const},
      {
        name: 'filled',
        view: props.neighborCounts,
        type: 'u32' as const,
        access: 'read' as const
      },
      ...(props.neighborDistances
        ? [
            {
              name: 'slotDistances',
              view: props.neighborDistances,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ];
    // Visits the valid slots of row `index` in join order, stopping after KEEP kept neighbors.
    const visit = (onKeep: string) => `let filledCount = min(filled[filledOffset + index], SLOTS);
  var kept = 0u;
  for (var slot = 0u; slot < filledCount && kept < KEEP; slot++) {
    let neighbor = ids[idsOffset + index * SLOTS + slot];
    if (neighbor == NO_FEATURE || (EXCLUDE_SELF && neighbor == index)) {
      continue;
    }
    ${onKeep}
    kept++;
  }`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          ...inputBindings.slice(0, 2),
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `${visit('')}
  counts[countsOffset + index] = kept;`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows + 1,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
    offsets[offsetsOffset + ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        variant: 'emit',
        bindings: [
          ...inputBindings,
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
          {name: 'values', view: weights.weights, type: 'f32', access: 'read_write'},
          ...(weights.distances
            ? [
                {
                  name: 'outDistances',
                  view: weights.distances,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations,
        body: `var keptIds: array<u32, ${MAXIMUM_SLOTS}>;
  var keptDistances: array<f32, ${MAXIMUM_SLOTS}>;
  ${visit(`var position = kept;
    let distance = ${props.neighborDistances ? 'slotDistances[slotDistancesOffset + index * SLOTS + slot]' : '0.0'};
    // Insertion sort by neighbor ID keeps the CSR rows ascending.
    while (position > 0u && keptIds[position - 1u] > neighbor) {
      keptIds[position] = keptIds[position - 1u];
      keptDistances[position] = keptDistances[position - 1u];
      position--;
    }
    keptIds[position] = neighbor;
    keptDistances[position] = distance;`)}
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var position = 0u; position < kept && begin + position < end; position++) {
    let outSlot = begin + position;
    neighbors[neighborsOffset + outSlot] = keptIds[position];
    ${
      inverse
        ? 'let safeDistance = max(keptDistances[position], FLOOR);\n    let weight = 1.0 / pow(safeDistance, POWER);\n    values[valuesOffset + outSlot] = select(0.0, weight, weight < 3.0e38);'
        : 'values[valuesOffset + outSlot] = 1.0;'
    }
    ${weights.distances ? 'outDistances[outDistancesOffset + outSlot] = keptDistances[position];' : ''}
  }`
      })
    ];
  }
}
