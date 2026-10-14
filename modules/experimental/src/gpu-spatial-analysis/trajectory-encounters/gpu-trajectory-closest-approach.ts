// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  validateCapacity,
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  CLOSEST_APPROACH_ENTRY_WORDS,
  CLOSEST_APPROACH_HIT_WORDS,
  CLOSEST_APPROACH_SEGMENT_WORDS,
  CLOSEST_APPROACH_STATE_WORDS,
  createClosestApproachEntriesNode,
  createClosestApproachEntryClearNode,
  createClosestApproachPairsNode,
  createClosestApproachReduceNodes,
  createClosestApproachSegmentsNode,
  createClosestApproachSortedHitsNode,
  createClosestApproachSortKeyNode,
  createClosestApproachStatusNode,
  type ClosestApproachShape
} from './closest-approach-kernels';

const OPERATION = 'GPUTrajectoryClosestApproach';

/**
 * Capacity-bounded pair list of {@link GPUTrajectoryClosestApproach}: a {@link GPUCompactOutput}
 * (`ids` holds the lower track index of each pair) with optional per-pair columns, each with
 * exactly `ids.length` rows. Pairs are ordered by `(track, partner)` with `track < partner`. Rows at
 * and after `count` hold `0xffffffff` for IDs and 0 for values. Times are seconds relative to the
 * first sample of the lower track (`ids`), like the event times of `GPUZoneEvents`.
 */
export type GPUTrajectoryClosestApproachPairs = GPUCompactOutput & {
  /** Optional higher track index of each pair. */
  partners?: GraphDataView<'uint32'>;
  /** Optional smallest distance between the two tracks over their common time, at most `distance`. */
  minimumDistances?: GraphDataView<'float32'>;
  /** Optional time of the smallest distance (the earliest when tied). */
  closestTimes?: GraphDataView<'float32'>;
  /** Optional first time at which the tracks are within `distance` of each other. */
  firstTimes?: GraphDataView<'float32'>;
  /** Optional total time during which the tracks are within `distance`. */
  durations?: GraphDataView<'float32'>;
};

/** Caller-owned outputs of {@link GPUTrajectoryClosestApproach}. */
export type GPUTrajectoryClosestApproachOutput = {
  /** Bounded pair list. `pairs.status.overflow` is 1 only when the pair capacity is too small. */
  pairs: GPUTrajectoryClosestApproachPairs;
  /** One-row status words. After either overflow, pairs may be missing or incomplete. */
  status: {
    /**
     * 1 when the space-time cell entries exceeded `entryCapacity`, or a segment covers more than
     * 4096 cells (raise `cellSize` or `timeCellSize`).
     */
    entryOverflow: GraphDataView<'uint32'>;
    /** 1 when the segment-pair hits exceeded `hitCapacity`. */
    hitOverflow: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUTrajectoryClosestApproach}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `distance`.
 * Compile-time: view lengths, `cellSize`, `timeCellSize`, `entryCapacity`, `hitCapacity`, the pair
 * capacity, the timestamp format and which optional views are present.
 */
export type GPUTrajectoryClosestApproachProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-closest-approach'`. */
  id?: string;
  /**
   * Planar only (the default). A longitude-latitude context throws: project positions first, for
   * example with `createGPULocalFrame` and `GPULocalFrameProjection`, and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Optional per-track flags, `trackCount` rows. Zero ignores the track. */
  trackValid?: GraphDataView<'uint32'>;
  /**
   * Per-frame distance in position units, one float32 row. Two tracks are an encounter while
   * their linearly interpolated positions are at most this far apart. Negative, NaN or infinite
   * matches nothing.
   */
  distance: GraphDataView<'float32'>;
  /**
   * Compile-time spatial cell width of the space-time grid, in position units. Cells of about the
   * typical `distance` plus segment length keep the per-cell candidate lists short. A segment may
   * cover at most 4096 cells (counting time cells); more raises `entryOverflow`.
   */
  cellSize: number;
  /** Compile-time time cell length of the space-time grid, in seconds. */
  timeCellSize: number;
  /**
   * Maximum `(cell, segment)` entries per encoding; also the sorted entry scratch length. Its
   * 16-byte records must fit one storage binding of the device (8388608 entries on a default
   * device, checked when the nodes are declared).
   */
  entryCapacity: number;
  /**
   * Maximum `(segment, segment)` hits per encoding; also the sorted hit scratch length. Its 24-byte
   * records must fit one storage binding (5592405 hits on a default device).
   */
  hitCapacity: number;
  /** Caller-owned pair list and status. */
  output: GPUTrajectoryClosestApproachOutput;
};

/**
 * Exact encounters between trajectories, the MobilityDB `tdwithin` / `nearestApproachDistance`
 * question on the raw samples: tracks move linearly between their samples, and two tracks are an
 * encounter while their positions at the same instant are at most `distance` apart.
 *
 * Unlike the clock (bucket) mode of `GPUTrajectoryEncounters` and `addClockEncounters`, which
 * compares positions only at shared bucket instants and misses tracks that pass each other between
 * buckets, this solves the closest point of approach of every pair of overlapping segments.
 * Use the clock mode as a fast preview over resampled tables, and this contributor for the exact
 * answer.
 *
 * Composition: one kernel turns every sample row into a segment to the next sample of its track
 * with a space-time box (`x`, `y` padded by half the distance, time measured from row 0), counts
 * the cells of a hashed space-time grid (`cellSize` by `cellSize` by `timeCellSize`) that the box
 * covers, a `GPUScan` gives the entry offsets, and one kernel writes a `(cell hash, segment, cell)`
 * entry per covered cell, sorted with `GPUSort`. Each entry tests the later entries of its cell
 * that belong to other tracks; a segment pair is tested once, in the cell that holds the minimum
 * corner of its box intersection. The test solves the separation `D + V tau` over the common time
 * interval: closest approach at `tau = -(D . V) / |V|^2` and time within the distance between the
 * roots of `|D + V tau|^2 = distance^2`. Hits are sorted by `(track, partner, enter time)`, runs
 * are compacted with `GPUCompaction`, and reduction kernels write the per-pair minimum distance,
 * its time, the first enter time and the total time within the distance. Times are measured from
 * a segment's own start row, so `uint32x2` word timestamps stay exact until the final f32
 * difference.
 *
 * Bounds: `entryCapacity` (cell entries) and `hitCapacity` (segment-pair hits) are scratch
 * capacities reported in `output.status`; `output.pairs.status.overflow` reports the pair capacity.
 */
export class GPUTrajectoryClosestApproach implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryClosestApproachProps;
  /** Resolved coordinate context; always planar. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of tracks. */
  readonly trackCount: number;

  constructor(props: GPUTrajectoryClosestApproachProps) {
    this.id = props.id ?? 'trajectory-closest-approach';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    if (props.timestamps.length !== props.positions.length) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    this.trackCount = props.trackOffsets.length - 1;
    if (props.trackValid) {
      validatePackedView(props.trackValid, ['uint32'], `${id} trackValid`);
      if (props.trackValid.length !== this.trackCount) {
        throw new Error(`${id} trackValid length must equal the track count`);
      }
    }
    validatePackedView(props.distance, ['float32'], `${id} distance`);
    if (props.distance.length < 1) {
      throw new Error(`${id} distance must contain one float32 row`);
    }
    for (const [name, value] of [
      ['cellSize', props.cellSize],
      ['timeCellSize', props.timeCellSize]
    ] as const) {
      if (!(Number.isFinite(value) && value > 0)) {
        throw new Error(`${id} ${name} must be a positive finite number`);
      }
    }
    validateCapacity(id, 'entryCapacity', props.entryCapacity);
    validateCapacity(id, 'hitCapacity', props.hitCapacity);
    const {pairs, status} = props.output;
    validateCompactOutput(id, pairs);
    for (const [name, view] of [
      ['status.entryOverflow', status.entryOverflow],
      ['status.hitOverflow', status.hitOverflow]
    ] as const) {
      validatePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} must hold one row`);
      }
    }
    const capacity = pairs.ids.length;
    for (const [name, view, format] of [
      ['partners', pairs.partners, 'uint32'],
      ['minimumDistances', pairs.minimumDistances, 'float32'],
      ['closestTimes', pairs.closestTimes, 'float32'],
      ['firstTimes', pairs.firstTimes, 'float32'],
      ['durations', pairs.durations, 'float32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} pairs.${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} pairs.${name} length must equal the capacity`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputs(), this.getInputs());
  }

  private getInputs(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.positions,
      props.timestamps,
      props.trackOffsets,
      props.trackValid,
      props.distance
    ];
  }

  private getOutputs(): (GraphDataView | undefined)[] {
    const {pairs, status} = this.props.output;
    return [
      pairs.ids,
      pairs.status.count,
      pairs.status.overflow,
      pairs.status.requiredCount,
      pairs.partners,
      pairs.minimumDistances,
      pairs.closestTimes,
      pairs.firstTimes,
      pairs.durations,
      status.entryOverflow,
      status.hitOverflow
    ];
  }

  /** Returns the segment, grid, entry sort, pair test, hit sort, run and reduction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, trackCount} = this;
    const {pairs, status} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputs(), ...this.getOutputs()]);
    const rowCount = props.positions.length;
    // Each scratch table is one storage binding on this device.
    for (const [name, byteLength, hint] of [
      ['segment records', rowCount * CLOSEST_APPROACH_SEGMENT_WORDS * 4, 'split the tracks'],
      [
        'cell entries',
        props.entryCapacity * CLOSEST_APPROACH_ENTRY_WORDS * 4,
        'lower entryCapacity'
      ],
      ['segment-pair hits', props.hitCapacity * CLOSEST_APPROACH_HIT_WORDS * 4, 'lower hitCapacity']
    ] as const) {
      validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, hint);
    }
    const shape: ClosestApproachShape = {
      rowCount,
      trackCount,
      cellSize: props.cellSize,
      timeCellSize: props.timeCellSize,
      entryCapacity: props.entryCapacity,
      hitCapacity: props.hitCapacity,
      pairCapacity: pairs.ids.length,
      isWordMode: props.timestamps.format === 'uint32x2'
    };
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const state = u32('state', CLOSEST_APPROACH_STATE_WORDS);
    const segmentRecords = u32('segment-records', rowCount * CLOSEST_APPROACH_SEGMENT_WORDS);
    const segmentCounts = u32('segment-counts', rowCount);
    const segmentStarts = u32('segment-starts', rowCount);
    const entryKeys = u32('entry-keys', props.entryCapacity);
    const entryIdentity = u32('entry-identity', props.entryCapacity);
    const entryRecords = u32('entry-records', props.entryCapacity * CLOSEST_APPROACH_ENTRY_WORDS);
    const sortedEntryKeys = u32('sorted-entry-keys', props.entryCapacity);
    const sortedEntries = u32('sorted-entries', props.entryCapacity);
    const hits = u32('hits', props.hitCapacity * CLOSEST_APPROACH_HIT_WORDS);
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        view: state,
        type: 'u32',
        value: '0u'
      }),
      createClosestApproachSegmentsNode<Parameters>(graph, {
        id: `${id}-segments`,
        shape,
        positions: props.positions,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        trackValid: props.trackValid,
        distance: props.distance,
        segmentRecords,
        segmentCounts,
        state
      }),
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: segmentCounts,
        output: segmentStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: `${id}-entry`,
        operation: OPERATION,
        counts: segmentCounts,
        offsets: segmentStarts,
        rowCount,
        total: state,
        totalRow: 1
      }),
      createClosestApproachEntryClearNode<Parameters>(graph, {
        id: `${id}-entry-clear`,
        entryKeys,
        entryIdentity
      }),
      createClosestApproachEntriesNode<Parameters>(graph, {
        id: `${id}-entries`,
        shape,
        segmentRecords,
        segmentCounts,
        segmentStarts,
        entryKeys,
        entryRecords
      }),
      ...new GPUSort({
        id: `${id}-entry-sort`,
        keys: entryKeys,
        values: entryIdentity,
        outputKeys: sortedEntryKeys,
        outputValues: sortedEntries
      }).getCommandNodes(graph),
      createClosestApproachPairsNode<Parameters>(graph, {
        id: `${id}-pairs`,
        shape,
        sortedEntries,
        entryRecords,
        segmentRecords,
        positions: props.positions,
        timestamps: props.timestamps,
        distance: props.distance,
        state,
        hits
      })
    ];

    // Stable LSD chain: enter time, then partner, then track gives (track, partner, enter) order.
    const capacity = props.hitCapacity;
    const identity = u32('hit-identity', capacity);
    const trackBits = getKeyBits(trackCount);
    let order: GraphDataView<'uint32'> | undefined;
    let sortedLow: GraphDataView<'uint32'> | undefined;
    for (const [variant, keyBits] of [
      ['enter', 32],
      ['high', trackBits],
      ['low', trackBits]
    ] as const) {
      const keys = u32(`${variant}-keys`, capacity);
      const sortedKeys = u32(`${variant}-sorted-keys`, capacity);
      const nextOrder = u32(`${variant}-order`, capacity);
      nodes.push(
        createClosestApproachSortKeyNode<Parameters>(graph, {
          id: `${id}-${variant}-keys`,
          shape,
          variant,
          state,
          hits,
          order,
          identity: order ? undefined : identity,
          keys
        }),
        ...new GPUSort({
          id: `${id}-${variant}-sort`,
          keys,
          values: order ?? identity,
          outputKeys: sortedKeys,
          outputValues: nextOrder,
          keyBits
        }).getCommandNodes(graph)
      );
      order = nextOrder;
      sortedLow = sortedKeys;
    }
    // The loop always runs: the last pass sorts by the low track.
    const finalOrder = order as GraphDataView<'uint32'>;
    const finalLow = sortedLow as GraphDataView<'uint32'>;
    const sortedHigh = u32('sorted-high', capacity);
    const sortedValues = f32('sorted-values', capacity * 4);
    const runFlags = u32('run-flags', capacity);
    const rowIds = u32('row-ids', capacity);
    const runStarts = u32('run-starts', capacity);
    const runCount = u32('run-count', 1);
    nodes.push(
      createClosestApproachSortedHitsNode<Parameters>(graph, {
        id: `${id}-sorted-hits`,
        shape,
        state,
        order: finalOrder,
        hits,
        sortedLow: finalLow,
        sortedHigh,
        sortedValues,
        flags: runFlags,
        rowIds
      }),
      ...new GPUCompaction({
        id: `${id}-runs`,
        input: rowIds,
        flags: runFlags,
        output: runStarts,
        count: runCount
      }).getCommandNodes(graph),
      ...createClosestApproachReduceNodes<Parameters>(graph, {
        id: `${id}-reduce`,
        shape,
        state,
        runStarts,
        runCount,
        sortedLow: finalLow,
        sortedHigh,
        sortedValues,
        columns: {
          ids: pairs.ids,
          partners: pairs.partners,
          minimumDistances: pairs.minimumDistances,
          closestTimes: pairs.closestTimes,
          firstTimes: pairs.firstTimes,
          durations: pairs.durations
        }
      }),
      createClosestApproachStatusNode<Parameters>(graph, {
        id: `${id}-status`,
        shape,
        state,
        entryOverflow: status.entryOverflow,
        hitOverflow: status.hitOverflow
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: runCount,
        output: pairs,
        overflowSources: []
      })
    );
    return nodes;
  }
}

/** Bits needed to sort keys in `[0, maximumKey]`. */
function getKeyBits(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}
