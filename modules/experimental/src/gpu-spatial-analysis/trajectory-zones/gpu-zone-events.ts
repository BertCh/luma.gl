// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createSpatialJoinBoundsNode,
  getNextPowerOfTwo,
  getSortedFeatureBVHNodes,
  isPowerOfTwo
} from '../spatial-join/spatial-join-passes';
import {
  createZoneClearNode,
  createZoneCloseNode,
  createZoneCrossingNode,
  createZoneDiagnosticsNode,
  createZoneDwellNode,
  createZoneEventTimeNode,
  createZoneGatherNode,
  createZoneInitialStateNode,
  createZoneKeepFlagsNode,
  createZoneOverflowNode,
  createZoneProbeNode,
  createZoneSeedNode,
  createZoneSortedColumnsNode,
  createZoneSortedPositionsNode,
  createZoneSortKeyNode,
  createZoneSpanNode,
  createZoneTableFlagsNode,
  createZoneTableGatherNode,
  createZoneWalkNode,
  getZoneTableColumnBindingCount,
  type ZoneEventsShape,
  type ZoneGatherColumn,
  type ZoneTableColumn
} from './zone-events-kernels';

const OPERATION = 'GPUZoneEvents';

/** Values of the `eventTypes` column of {@link GPUZoneEvents}. */
export const GPU_ZONE_EVENT_TYPE = {
  /** The track enters the zone. */
  enter: 0,
  /** The track leaves the zone. */
  exit: 1
} as const;

/**
 * Capacity-bounded event list of {@link GPUZoneEvents}.
 *
 * `output.ids.length` is the event capacity, fixed at compile time. Every column that is present
 * must have exactly that many rows. Events are ordered by track, then time, then edge row. Rows at
 * and after `output.count` hold sentinels: `0xffffffff` for IDs, zones and rows, `0` for times and
 * types. `output.ids` holds the track index of each event.
 */
export type GPUZoneEventOutput = {
  /** Bounded compact result. `ids` holds the track index of each event. */
  output: GPUCompactOutput;
  /** Optional zone index (`edgeZones` value) of each event. */
  eventZones?: GraphDataView<'uint32'>;
  /** Optional {@link GPU_ZONE_EVENT_TYPE} code of each event. */
  eventTypes?: GraphDataView<'uint32'>;
  /** Optional interpolated crossing time relative to the track's first timestamp. */
  eventTimes?: GraphDataView<'float32'>;
  /**
   * Optional row of the segment's end sample. The crossing lies on the segment between row `r - 1`
   * and row `r`.
   */
  eventRows?: GraphDataView<'uint32'>;
  /**
   * Optional interpolated crossing position (x, y) of each event, in the units of `positions`:
   * `positions[r - 1] + s * (positions[r] - positions[r - 1])` with the same intersection
   * parameter `s` as the crossing time. Zero at and after `output.count`.
   */
  eventPositions?: GraphDataView<'float32x2'>;
};

/**
 * Capacity-bounded sparse `(track, zone)` table of {@link GPUZoneEvents}: one row for every pair
 * with at least one visit (an interval inside the zone, including one that is open at the start
 * or end of the track).
 *
 * `output.ids.length` is the row capacity, fixed at compile time. Every column that is present
 * must have exactly that many rows. Rows are ordered by track, then zone. `output.ids` holds the
 * track index. Rows at and after `output.count` hold sentinels: `0xffffffff` for IDs, zones and
 * visits, `0` for times. `output.overflow` is 1 when more pairs have visits than rows, and
 * `output.totalCount` (when present) receives the unclamped number of such pairs.
 */
export type GPUZoneVisitTableOutput = {
  /** Bounded compact result. `ids` holds the track index of each row. */
  output: GPUCompactOutput;
  /** Optional zone index of each row. */
  zones?: GraphDataView<'uint32'>;
  /** Optional number of maximal inside intervals, as `visitCounts`. */
  visits?: GraphDataView<'uint32'>;
  /** Optional total dwell time, as `dwellTimes`. */
  dwellTimes?: GraphDataView<'float32'>;
  /**
   * Optional time of the first enter relative to the track's first timestamp. A track that starts
   * inside the zone reports 0.
   */
  firstEnterTimes?: GraphDataView<'float32'>;
  /**
   * Optional time of the last exit relative to the track's first timestamp. A visit that is still
   * open at the end of the track reports the track's duration.
   */
  lastExitTimes?: GraphDataView<'float32'>;
};

/**
 * Optional one-row outputs that split `events.output.overflow` into its causes and report the
 * capacity the candidate scratch needed. Every view is optional and independent.
 */
export type GPUZoneEventsDiagnostics = {
  /**
   * Unclamped number of `(row, edge)` bounding-box candidates found (including one ray query per
   * track). `candidateCapacity` must be at least this for exact results, so callers can size it.
   */
  candidateCount?: GraphDataView<'uint32'>;
  /** 1 when `candidateCount` exceeded `candidateCapacity`; every result may then miss events. */
  candidateOverflow?: GraphDataView<'uint32'>;
  /** 1 when a track had more events than `maxEventsPerTrack` (the list is truncated, dwell is exact). */
  trackOverflow?: GraphDataView<'uint32'>;
  /** 1 when more events were kept than `events.output.ids.length`. */
  eventOverflow?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUZoneEvents}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * track and edge counts, `zoneCount`, `candidateCapacity`, `maxEventsPerTrack`, `leafCapacity`,
 * `spatialSort`, the time format, and which optional views are present.
 */
export type GPUZoneEventsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'zone-events'`. */
  id?: string;
  /** Packed planar positions, one row per sample, sorted by track and then time. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Packed sample times, non-decreasing inside each track: `float32` relative to an application
   * epoch, or `uint32x2` Int64 `(low, high)` words (differences are exact before the f32 round).
   */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** `trackCount + 1` monotonic row offsets, the layout of `GPUTrajectoryMetrics`. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Start point of each zone boundary edge. Edges of a ring must form a closed loop. */
  edgeStarts: GraphDataView<'float32x2'>;
  /** End point of each zone boundary edge, aligned with `edgeStarts`. */
  edgeEnds: GraphDataView<'float32x2'>;
  /**
   * Zone index in `[0, zoneCount)` of each edge. Rings of one zone (holes, extra parts) share an
   * index. Edges with a larger value or a non-finite endpoint are ignored.
   */
  edgeZones: GraphDataView<'uint32'>;
  /** Number of zones, at least 1. The dwell and visit matrices have `trackCount * zoneCount` rows. */
  zoneCount: number;
  /**
   * Maximum `(row, edge)` bounding-box candidates per encoding, including one ray query per track.
   * Also the length of the sorted event scratch. A larger value costs sort time every encoding.
   */
  candidateCapacity: number;
  /** Events kept per track in `events`, at least 1. Dwell and visits always use every event. */
  maxEventsPerTrack: number;
  /** Bounded event list. */
  events: GPUZoneEventOutput;
  /**
   * Optional dense `[trackCount * zoneCount]` dwell time, row `track * zoneCount + zone`, in the
   * timestamps' unit. Includes time before the first exit when a track starts inside a zone and
   * time after the last enter up to the track's last sample.
   */
  dwellTimes?: GraphDataView<'float32'>;
  /**
   * Optional dense `[trackCount * zoneCount]` count of maximal inside intervals. A track that
   * starts inside a zone counts that first interval.
   */
  visitCounts?: GraphDataView<'uint32'>;
  /**
   * Optional sparse `(track, zone)` table of visits, dwell, first enter and last exit. It is
   * built from the same dense per-cell state as `dwellTimes` and `visitCounts` (which stay
   * transient when not requested), so it bounds the output size, not the per-encoding memory.
   */
  visitTable?: GPUZoneVisitTableOutput;
  /** Optional per-track number of events found, not clamped by `maxEventsPerTrack`. */
  trackEventCounts?: GraphDataView<'uint32'>;
  /**
   * Optional split overflow flags and the required candidate capacity.
   * `events.output.overflow` stays the OR of the three flags.
   */
  diagnostics?: GPUZoneEventsDiagnostics;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the edge count. */
  leafCapacity?: number;
  /** Morton-sorts edges before the BVH build (compile-time). Results are identical. Default false. */
  spatialSort?: boolean;
};

/**
 * Detects when trajectories enter and leave polygon zones, with interpolated times.
 *
 * Zones are boundary edges tagged with a zone index. Every segment between consecutive samples of
 * a track queries a `GPUBVH` over the edge bounds; each bounding-box candidate is tested with an
 * exact segment-segment intersection, with half-open parameter ranges so a segment through a shared
 * vertex counts once. The crossing time is `t[r - 1] + s * (t[r] - t[r - 1])` where `s` is the
 * intersection parameter, stored relative to the track's first timestamp (a duplicate-timestamp
 * segment yields its start time).
 *
 * Whether a crossing enters or leaves does not depend on ring orientation. The first sample of each
 * track probes the BVH with a ray toward +x (even-odd rule) to find which zones contain it; every
 * crossing then toggles the `(track, zone)` state. Holes (extra rings of the same zone) therefore
 * work, while overlapping rings of one zone do not. Orientation, ties and degenerate inputs: a
 * vertex lying exactly on a track sample or segment is resolved by half-open rules, never by
 * epsilons, and a track sample exactly on an edge may disagree with the CPU parity by one event.
 *
 * Composition: bounds and BVH build, one probe kernel per row, an exact crossing kernel, a ray
 * parity kernel, an event-time kernel, three stable `GPUSort` passes (edge row, time, track) giving
 * `(track, time, edge row)` order, one sequential walk per track that alternates enter and exit and
 * ranks events, one dwell kernel per track, a close kernel for still-open visits, `GPUCompaction`
 * of the first `maxEventsPerTrack` events of each track, gather kernels, and one publish kernel.
 *
 * Bounds and cost: the event scratch is `candidateCapacity` rows and every encoding sorts all of
 * them. The walk and dwell kernels use one invocation per track (so very few tracks with very many
 * events is slow), and the matrices are `trackCount * zoneCount` rows (so very large zone counts need
 * a sparse approach). `events.output.overflow` is 1 when the candidate scratch overflowed, when a
 * track has more than `maxEventsPerTrack` events, or when more events were kept than
 * `events.output.ids.length`; `diagnostics` reports each cause separately plus the candidate
 * count the scratch needed. After a candidate overflow every result may be missing events.
 *
 * Crossing positions: `events.eventPositions` holds the interpolated crossing point of each event.
 * Sparse table: `visitTable` compacts the `(track, zone)` cells that have a visit into a bounded
 * list with visits, total dwell, first enter and last exit, so large zone counts do not need to
 * read dense matrices back.
 *
 * Non-goals: geodesic crossings, zone polygons with orientation semantics, point-in-polygon tolerance for samples exactly on a
 * boundary, and Double-single timestamps.
 */
export class GPUZoneEvents implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUZoneEventsProps;
  /** Number of tracks. */
  readonly trackCount: number;
  /** Number of boundary edges. */
  readonly edgeCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;

  constructor(props: GPUZoneEventsProps) {
    this.id = props.id ?? 'zone-events';
    this.props = props;
    const {id} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedUint32View(props.trackOffsets, `${id} trackOffsets`);
    validatePackedView(props.edgeStarts, ['float32x2'], `${id} edgeStarts`);
    validatePackedView(props.edgeEnds, ['float32x2'], `${id} edgeEnds`);
    validatePackedUint32View(props.edgeZones, `${id} edgeZones`);
    if (props.timestamps.length !== props.positions.length) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    this.trackCount = props.trackOffsets.length - 1;
    this.edgeCount = props.edgeStarts.length;
    if (this.edgeCount < 1) {
      throw new Error(`${id} needs at least one edge`);
    }
    if (props.edgeEnds.length !== this.edgeCount || props.edgeZones.length !== this.edgeCount) {
      throw new Error(`${id} edgeStarts, edgeEnds and edgeZones must have the same length`);
    }
    for (const [name, value] of [
      ['zoneCount', props.zoneCount],
      ['candidateCapacity', props.candidateCapacity],
      ['maxEventsPerTrack', props.maxEventsPerTrack]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    if (props.positions.length >= 0x80000000) {
      throw new Error(`${id} supports fewer than 2^31 rows`);
    }
    const cellCount = this.trackCount * props.zoneCount;
    if (cellCount > 0xffffffff) {
      throw new Error(`${id} trackCount * zoneCount must fit in 32 bits`);
    }
    for (const [name, view, format] of [
      ['dwellTimes', props.dwellTimes, 'float32'],
      ['visitCounts', props.visitCounts, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} length must equal trackCount * zoneCount`);
        }
      }
    }
    if (props.trackEventCounts) {
      validatePackedUint32View(props.trackEventCounts, `${id} trackEventCounts`);
      if (props.trackEventCounts.length !== this.trackCount) {
        throw new Error(`${id} trackEventCounts length must equal the track count`);
      }
    }
    const {events} = props;
    validateCompactOutput(id, events.output);
    const capacity = events.output.ids.length;
    for (const [name, view, format] of [
      ['eventZones', events.eventZones, 'uint32'],
      ['eventTypes', events.eventTypes, 'uint32'],
      ['eventTimes', events.eventTimes, 'float32'],
      ['eventRows', events.eventRows, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} events.${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} events.${name} length must equal the event capacity`);
        }
      }
    }
    if (events.eventPositions) {
      validatePackedView(events.eventPositions, ['float32x2'], `${id} events.eventPositions`);
      if (events.eventPositions.length !== capacity) {
        throw new Error(`${id} events.eventPositions length must equal the event capacity`);
      }
    }
    const {visitTable} = props;
    if (visitTable) {
      validateCompactOutput(`${id} visitTable`, visitTable.output);
      const tableCapacity = visitTable.output.ids.length;
      for (const [name, view, format] of [
        ['zones', visitTable.zones, 'uint32'],
        ['visits', visitTable.visits, 'uint32'],
        ['dwellTimes', visitTable.dwellTimes, 'float32'],
        ['firstEnterTimes', visitTable.firstEnterTimes, 'float32'],
        ['lastExitTimes', visitTable.lastExitTimes, 'float32']
      ] as const) {
        if (view) {
          validatePackedView(view, [format], `${id} visitTable.${name}`);
          if (view.length !== tableCapacity) {
            throw new Error(`${id} visitTable.${name} length must equal the table capacity`);
          }
        }
      }
    }
    for (const [name, view] of Object.entries(props.diagnostics ?? {})) {
      if (view) {
        validatePackedUint32View(view, `${id} diagnostics.${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} diagnostics.${name} must have one row`);
        }
      }
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.edgeCount);
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        events.output.ids,
        events.output.count,
        events.output.overflow,
        events.output.totalCount,
        events.eventZones,
        events.eventTypes,
        events.eventTimes,
        events.eventRows,
        events.eventPositions,
        ...this.getTableViews(),
        props.dwellTimes,
        props.visitCounts,
        props.trackEventCounts,
        ...Object.values(props.diagnostics ?? {})
      ],
      [
        props.positions,
        props.timestamps,
        props.trackOffsets,
        props.edgeStarts,
        props.edgeEnds,
        props.edgeZones
      ]
    );
  }

  /** Returns every node of the pipeline described in the class documentation. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, trackCount, edgeCount, leafCapacity} = this;
    const {events} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.timestamps,
      props.trackOffsets,
      props.edgeStarts,
      props.edgeEnds,
      props.edgeZones,
      events.output.ids,
      events.output.count,
      events.output.overflow,
      events.output.totalCount,
      events.eventZones,
      events.eventTypes,
      events.eventTimes,
      events.eventRows,
      events.eventPositions,
      ...this.getTableViews(),
      props.dwellTimes,
      props.visitCounts,
      props.trackEventCounts,
      ...Object.values(props.diagnostics ?? {})
    ]);
    const rowCount = props.positions.length;
    const capacity = props.candidateCapacity;
    const cellCount = trackCount * props.zoneCount;
    const isWordMode = props.timestamps.format === 'uint32x2';
    const shape: ZoneEventsShape = {
      rowCount,
      trackCount,
      edgeCount,
      zoneCount: props.zoneCount,
      candidateCapacity: capacity,
      maximumEventsPerTrack: props.maxEventsPerTrack
    };
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    const minima = createTransientView(graph, `${id}-edge-minima`, 'float32x2', edgeCount);
    const maxima = createTransientView(graph, `${id}-edge-maxima`, 'float32x2', edgeCount);
    nodes.push(
      createSpatialJoinBoundsNode<Parameters>(graph, {
        id: `${id}-bounds`,
        operation: OPERATION,
        featureCount: edgeCount,
        source: {kind: 'segments', starts: props.edgeStarts, ends: props.edgeEnds},
        minima,
        maxima
      })
    );
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes(
      graph,
      id,
      OPERATION,
      minima,
      maxima,
      leafCapacity,
      props.spatialSort ?? false
    );
    nodes.push(...bvhNodes);

    const state = u32('state', 4);
    const initialInside = u32('initial-inside', cellCount);
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      capacity
    );
    const candidateKinds = u32('candidate-kinds', capacity);
    const candidateParameters = f32('candidate-parameters', capacity);
    const candidateTracks = u32('candidate-tracks', capacity);
    const candidateTimes = f32('candidate-times', capacity);
    nodes.push(
      createZoneClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        state,
        initialInside,
        cellCount
      }),
      createZoneProbeNode<Parameters>(graph, {
        id: `${id}-probe`,
        shape,
        positions: props.positions,
        trackOffsets: props.trackOffsets,
        bvh,
        state,
        candidatePairs
      }),
      createZoneCrossingNode<Parameters>(graph, {
        id: `${id}-crossing`,
        shape,
        positions: props.positions,
        edgeStarts: props.edgeStarts,
        edgeEnds: props.edgeEnds,
        edgeZones: props.edgeZones,
        state,
        candidatePairs,
        candidateKinds,
        candidateParameters
      }),
      createZoneInitialStateNode<Parameters>(graph, {
        id: `${id}-initial-state`,
        shape,
        trackOffsets: props.trackOffsets,
        edgeZones: props.edgeZones,
        candidatePairs,
        candidateKinds,
        initialInside
      }),
      createZoneEventTimeNode<Parameters>(graph, {
        id: `${id}-event-time`,
        shape,
        isWordMode,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        state,
        candidatePairs,
        candidateKinds,
        candidateParameters,
        candidateTracks,
        candidateTimes
      })
    );

    // Stable LSD chain: edge row, then time, then track gives (track, time, edge row) order.
    const identity = u32('identity', capacity);
    const edgeKeys = u32('edge-keys', capacity);
    const edgeSortedKeys = u32('edge-sorted-keys', capacity);
    const edgeOrder = u32('edge-order', capacity);
    const timeKeys = u32('time-keys', capacity);
    const timeSortedKeys = u32('time-sorted-keys', capacity);
    const timeOrder = u32('time-order', capacity);
    const trackKeys = u32('track-keys', capacity);
    const sortedTracks = u32('sorted-tracks', capacity);
    const order = u32('order', capacity);
    nodes.push(
      createZoneSortKeyNode<Parameters>(graph, {
        id: `${id}-edge-keys`,
        shape,
        variant: 'edge',
        candidatePairs,
        candidateKinds,
        keys: edgeKeys,
        identity
      }),
      ...new GPUSort({
        id: `${id}-edge-sort`,
        keys: edgeKeys,
        values: identity,
        outputKeys: edgeSortedKeys,
        outputValues: edgeOrder,
        keyBits: getKeyBits(edgeCount - 1)
      }).getCommandNodes(graph),
      createZoneSortKeyNode<Parameters>(graph, {
        id: `${id}-time-keys`,
        shape,
        variant: 'time',
        order: edgeOrder,
        candidateTimes,
        keys: timeKeys
      }),
      ...new GPUSort({
        id: `${id}-time-sort`,
        keys: timeKeys,
        values: edgeOrder,
        outputKeys: timeSortedKeys,
        outputValues: timeOrder,
        keyBits: 32
      }).getCommandNodes(graph),
      createZoneSortKeyNode<Parameters>(graph, {
        id: `${id}-track-keys`,
        shape,
        variant: 'track',
        order: timeOrder,
        candidateTracks,
        keys: trackKeys
      }),
      ...new GPUSort({
        id: `${id}-track-sort`,
        keys: trackKeys,
        values: timeOrder,
        outputKeys: sortedTracks,
        outputValues: order,
        keyBits: getKeyBits(trackCount)
      }).getCommandNodes(graph)
    );

    const sortedZones = u32('sorted-zones', capacity);
    const sortedTimes = f32('sorted-times', capacity);
    const sortedRows = u32('sorted-rows', capacity);
    const eventTypes = u32('event-types', capacity);
    const eventRanks = u32('event-ranks', capacity);
    const walkState = u32('walk-state', cellCount);
    const enterTimes = f32('enter-times', cellCount);
    const {visitTable} = props;
    const needsSpans = Boolean(visitTable?.firstEnterTimes || visitTable?.lastExitTimes);
    const firstEnterTimes = needsSpans ? f32('first-enter-times', cellCount) : undefined;
    const lastExitTimes = needsSpans ? f32('last-exit-times', cellCount) : undefined;
    const dwellTimes = props.dwellTimes ?? f32('dwell-times', cellCount);
    const visitCounts = props.visitCounts ?? u32('visit-counts', cellCount);
    const trackEventCounts = props.trackEventCounts ?? u32('track-event-counts', trackCount);
    nodes.push(
      createZoneSortedColumnsNode<Parameters>(graph, {
        id: `${id}-sorted-columns`,
        shape,
        state,
        order,
        candidatePairs,
        edgeZones: props.edgeZones,
        candidateTimes,
        sortedZones,
        sortedTimes,
        sortedRows
      }),
      createZoneSeedNode<Parameters>(graph, {
        id: `${id}-seed`,
        cellCount,
        initialInside,
        walkState,
        enterTimes,
        dwellTimes,
        visitCounts,
        firstEnterTimes,
        lastExitTimes
      }),
      createZoneWalkNode<Parameters>(graph, {
        id: `${id}-walk`,
        shape,
        state,
        sortedTracks,
        sortedZones,
        walkState,
        eventTypes,
        eventRanks,
        trackEventCounts
      }),
      createZoneDwellNode<Parameters>(graph, {
        id: `${id}-dwell`,
        shape,
        state,
        sortedTracks,
        sortedZones,
        sortedTimes,
        eventTypes,
        enterTimes,
        dwellTimes,
        visitCounts
      }),
      createZoneCloseNode<Parameters>(graph, {
        id: `${id}-close`,
        shape,
        isWordMode,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        walkState,
        enterTimes,
        dwellTimes,
        lastExitTimes
      })
    );
    if (firstEnterTimes && lastExitTimes) {
      // Runs before the close node, which overwrites the last exit of still-open visits.
      nodes.splice(
        nodes.length - 1,
        0,
        createZoneSpanNode<Parameters>(graph, {
          id: `${id}-span`,
          shape,
          state,
          sortedTracks,
          sortedZones,
          sortedTimes,
          eventTypes,
          firstEnterTimes,
          lastExitTimes
        })
      );
    }

    let sortedPositions: GraphDataView<'float32x2'> | undefined;
    if (events.eventPositions) {
      sortedPositions = createTransientView(
        graph,
        `${id}-sorted-positions`,
        'float32x2',
        Math.max(capacity, 1)
      );
      nodes.push(
        createZoneSortedPositionsNode<Parameters>(graph, {
          id: `${id}-sorted-positions`,
          shape,
          state,
          order,
          candidatePairs,
          candidateParameters,
          positions: props.positions,
          sortedPositions
        })
      );
    }
    const keepFlags = u32('keep-flags', capacity);
    const rowIds = u32('row-ids', capacity);
    const keptRows = u32('kept-rows', capacity);
    const keptTotal = u32('kept-total', 1);
    const overflowFlag = u32('overflow-flag', 1);
    nodes.push(
      createZoneKeepFlagsNode<Parameters>(graph, {
        id: `${id}-keep-flags`,
        shape,
        state,
        eventRanks,
        keepFlags,
        rowIds
      }),
      ...new GPUCompaction({
        id: `${id}-kept`,
        input: rowIds,
        flags: keepFlags,
        output: keptRows,
        count: keptTotal
      }).getCommandNodes(graph),
      createZoneOverflowNode<Parameters>(graph, {
        id: `${id}-overflow`,
        shape,
        state,
        flag: overflowFlag
      })
    );
    if (props.diagnostics && Object.values(props.diagnostics).some(Boolean)) {
      nodes.push(
        createZoneDiagnosticsNode<Parameters>(graph, {
          id: `${id}-diagnostics`,
          shape,
          eventCapacity: events.output.ids.length,
          state,
          keptTotal,
          ...props.diagnostics
        })
      );
    }
    const columns: ZoneGatherColumn[] = [
      {source: sortedTracks, destination: events.output.ids, sentinel: '0xffffffffu'}
    ];
    if (events.eventZones) {
      columns.push({source: sortedZones, destination: events.eventZones, sentinel: '0xffffffffu'});
    }
    if (events.eventRows) {
      columns.push({source: sortedRows, destination: events.eventRows, sentinel: '0xffffffffu'});
    }
    if (events.eventTypes) {
      columns.push({source: eventTypes, destination: events.eventTypes, sentinel: '0u'});
    }
    if (events.eventTimes) {
      columns.push({source: sortedTimes, destination: events.eventTimes, sentinel: '0u'});
    }
    if (events.eventPositions && sortedPositions) {
      columns.push({
        source: sortedPositions,
        destination: events.eventPositions,
        sentinel: '0u',
        stride: 2
      });
    }
    for (let first = 0, part = 0; first < columns.length; first += 3, part++) {
      nodes.push(
        createZoneGatherNode<Parameters>(graph, {
          id: `${id}-gather-${part}`,
          capacity: events.output.ids.length,
          keptTotal,
          keptRows,
          columns: columns.slice(first, first + 3)
        })
      );
    }
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        totalCount: keptTotal,
        output: events.output,
        overflowSources: [overflowFlag]
      })
    );
    if (visitTable) {
      nodes.push(
        ...this.getTableNodes(graph, {
          cellCount,
          visitCounts,
          dwellTimes,
          firstEnterTimes,
          lastExitTimes
        })
      );
    }
    return nodes;
  }

  /** Every output view of the optional visit table. */
  private getTableViews(): (GraphDataView | undefined)[] {
    const table = this.props.visitTable;
    return table
      ? [
          table.output.ids,
          table.output.count,
          table.output.overflow,
          table.output.totalCount,
          table.zones,
          table.visits,
          table.dwellTimes,
          table.firstEnterTimes,
          table.lastExitTimes
        ]
      : [];
  }

  /** Compaction of the visited `(track, zone)` cells, gather kernels and the publish kernel. */
  private getTableNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    cells: {
      cellCount: number;
      visitCounts: GraphDataView<'uint32'>;
      dwellTimes: GraphDataView<'float32'>;
      firstEnterTimes?: GraphDataView<'float32'>;
      lastExitTimes?: GraphDataView<'float32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const table = this.props.visitTable!;
    const {cellCount} = cells;
    const capacity = table.output.ids.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const flags = u32('table-flags', cellCount);
    const cellIds = u32('table-cell-ids', cellCount);
    const keptCells = u32('table-kept-cells', cellCount);
    const keptTotal = u32('table-kept-total', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      createZoneTableFlagsNode<Parameters>(graph, {
        id: `${id}-table-flags`,
        cellCount,
        visitCounts: cells.visitCounts,
        flags,
        cellIds
      }),
      ...new GPUCompaction({
        id: `${id}-table-cells`,
        input: cellIds,
        flags,
        output: keptCells,
        count: keptTotal
      }).getCommandNodes(graph)
    ];
    const columns: ZoneTableColumn[] = [{kind: 'track', destination: table.output.ids}];
    if (table.zones) {
      columns.push({kind: 'zone', destination: table.zones});
    }
    if (table.visits) {
      columns.push({kind: 'value', source: cells.visitCounts, destination: table.visits});
    }
    if (table.dwellTimes) {
      columns.push({kind: 'value', source: cells.dwellTimes, destination: table.dwellTimes});
    }
    if (table.firstEnterTimes && cells.firstEnterTimes) {
      columns.push({
        kind: 'value',
        source: cells.firstEnterTimes,
        destination: table.firstEnterTimes
      });
    }
    if (table.lastExitTimes && cells.lastExitTimes) {
      columns.push({kind: 'value', source: cells.lastExitTimes, destination: table.lastExitTimes});
    }
    // Pack columns into kernels of at most 8 storage bindings (keptTotal and keptCells use 2).
    let chunk: ZoneTableColumn[] = [];
    let bindingCount = 2;
    let part = 0;
    const flush = () => {
      if (chunk.length > 0) {
        nodes.push(
          createZoneTableGatherNode<Parameters>(graph, {
            id: `${id}-table-gather-${part++}`,
            capacity,
            zoneCount: this.props.zoneCount,
            keptTotal,
            keptCells,
            columns: chunk
          })
        );
        chunk = [];
        bindingCount = 2;
      }
    };
    for (const column of columns) {
      const columnBindings = getZoneTableColumnBindingCount(column);
      if (bindingCount + columnBindings > 8) {
        flush();
      }
      chunk.push(column);
      bindingCount += columnBindings;
    }
    flush();
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-table-publish`,
        operation: OPERATION,
        totalCount: keptTotal,
        output: table.output
      })
    );
    return nodes;
  }
}

/** Bits needed to sort keys in `[0, maximumKey]`. */
function getKeyBits(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}
