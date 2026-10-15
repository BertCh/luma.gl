// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {createLoadBalancedSegmentSumNodes} from '../../utils/load-balanced-for-each';
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import type {
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPUTrajectoryGeometryPort
} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createSpatialJoinBoundsNode,
  getNextPowerOfTwo,
  getSortedFeatureBVHNodes
} from '../spatial-join/spatial-join-passes';
import {
  createZoneCellEventsNode,
  createZoneCellFinalizeNode,
  createZoneCellKeysNode,
  createZoneCellSpansNode,
  createZoneClearNode,
  createZoneCrossingNode,
  createZoneDiagnosticsNode,
  createZoneEdgesNode,
  createZoneEventRanksNode,
  createZoneEventTimeNode,
  createZoneGatherNode,
  createZoneInitialStateNode,
  createZoneKeepFlagsNode,
  createZoneOverflowNode,
  createZoneProbeNode,
  createZoneSortedColumnsNode,
  createZoneSortedPositionsNode,
  createZoneSortKeyNode,
  createZoneTableFlagsNode,
  createZoneTableGatherNode,
  createZoneTrackCountsNode,
  getZoneTableColumnBindingCount,
  ZONE_DWELL_CONTRIBUTION_WGSL,
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
 * Capacity-bounded event list of {@link GPUZoneEvents}: a {@link GPUCompactOutput} extended with
 * optional per-event columns.
 *
 * `ids.length` is the event capacity, fixed at compile time. Every column that is present must
 * have exactly that many rows. Events are ordered by track, then time, then edge row. Rows at and
 * after `count` hold sentinels: `0xffffffff` for IDs, zones and rows, `0` for times and types.
 * `ids` holds the track index of each event.
 */
export type GPUZoneEventOutput = GPUCompactOutput & {
  /** Optional zone index (feature row of `zones`) of each event. */
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
   * parameter `s` as the crossing time. Zero at and after `count`.
   */
  eventPositions?: GraphDataView<'float32x2'>;
};

/**
 * Capacity-bounded sparse `(track, zone)` table of {@link GPUZoneEvents}: one row for every pair
 * with at least one visit (an interval inside the zone, including one that is open at the start
 * or end of the track). A {@link GPUCompactOutput} extended with optional per-row columns.
 *
 * `ids.length` is the row capacity, fixed at compile time. Every column that is present must have
 * exactly that many rows. Rows are ordered by track, then zone. `ids` holds the track index. Rows
 * at and after `count` hold sentinels: `0xffffffff` for IDs, zones and visits, `0` for times.
 * `overflow` is 1 when more pairs have visits than rows, and `requiredCount` (when present)
 * receives the unclamped number of such pairs.
 */
export type GPUZoneVisitTableOutput = GPUCompactOutput & {
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
 * Optional one-row status words of {@link GPUZoneEvents} that split `output.events.overflow` into
 * its causes and report the capacity the candidate scratch needed. Every view is optional and
 * independent.
 */
export type GPUZoneEventsStatus = {
  /**
   * Unclamped number of `(row, edge)` bounding-box candidates found (including one ray query per
   * track). `candidateCapacity` must be at least this for exact results, so callers can size it.
   */
  candidateCount?: GraphDataView<'uint32'>;
  /** 1 when `candidateCount` exceeded `candidateCapacity`; every result may then miss events. */
  candidateOverflow?: GraphDataView<'uint32'>;
  /** 1 when a track had more events than `trackEventCapacity` (the list is truncated, dwell is exact). */
  trackOverflow?: GraphDataView<'uint32'>;
  /** 1 when more events were kept than `events.ids.length`. */
  eventOverflow?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUZoneEvents}. */
export type GPUZoneEventsOutput = {
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
  /** Optional per-track number of events found, not clamped by `trackEventCapacity`. */
  trackEventCounts?: GraphDataView<'uint32'>;
  /** Optional split overflow flags and the required candidate capacity. */
  status?: GPUZoneEventsStatus;
};

/**
 * Properties for {@link GPUZoneEvents}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * track count, the zone count (`zones.featureOffsets.length - 1`), `candidateCapacity`,
 * `trackEventCapacity`, `leafCapacity`, `spatialSort`, the time format, and which optional views
 * are present.
 */
export type GPUZoneEventsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'zone-events'`. */
  id?: string;
  /**
   * Coordinate space of positions and zones. Planar only: segments and zone edges are straight
   * lines in the coordinate plane, which is not the geometry of longitude/latitude degrees (a great
   * circle segment is not a straight line in degrees, and edges cross the antimeridian). Any
   * longitude-latitude context throws; project positions and zones to a planar frame first.
   * Defaults to native planar.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `zones` when declared separately from `spatialContext`. Must be compatible in
   * datum, coordinate space, metric, units and planar frame.
   * Omitted means `spatialContext` describes both inputs.
   */
  zonesSpatialContext?: GPUSpatialContext;
  /**
   * Tagged trajectory geometry: `positions` are packed planar rows sorted by track and then time,
   * `timestamps` are non-decreasing inside each track (`float32` relative to an application epoch,
   * or `uint32x2` Int64 `(low, high)` words; differences are exact before the f32 round), and
   * `trackOffsets` are `trackCount + 1` monotonic row offsets, the layout of `GPUTrajectoryMetrics`.
   */
  tracks: GPUPackedGeometryPort<GPUTrajectoryGeometryPort>;
  /**
   * Tagged polygon geometry in the four-level layout, one feature per zone: zone `z` is feature
   * `z` and the first ring of each polygon is its shell. Rings close implicitly (a repeated closing
   * vertex is tolerated). Rings of one zone (holes, extra parts) share the zone index, so holes
   * work and overlapping rings of one zone do not. Vertices with a non-finite coordinate ignore
   * the edges they start or end. The dwell and visit matrices have `trackCount * zoneCount` rows.
   */
  zones: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /**
   * Maximum `(row, edge)` bounding-box candidates per encoding, including one ray query per track.
   * Also the length of the sorted event scratch. A larger value costs sort time every encoding.
   */
  candidateCapacity: number;
  /** Events kept per track in `output.events`, at least 1. Dwell and visits always use every event. */
  trackEventCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the edge count. */
  leafCapacity?: number;
  /** Morton-sorts edges before the BVH build (compile-time). Results are identical. Default false. */
  spatialSort?: boolean;
  /** Caller-owned output views. */
  output: GPUZoneEventsOutput;
};

/**
 * Detects when trajectories enter and leave polygon zones, with interpolated times.
 *
 * Zones are the features of a polygon geometry: every ring edge is tagged with its feature index. Every segment between consecutive samples of
 * a track queries a `GPUBVH` over the edge bounds; each bounding-box candidate is tested with an
 * exact crossing test built on `orientSign`, so decisions never depend on f32 rounding. The crossing time is `t[r - 1] + s * (t[r] - t[r - 1])` where `s` is the
 * intersection parameter, stored relative to the track's first timestamp (a duplicate-timestamp
 * segment yields its start time).
 *
 * Whether a crossing enters or leaves does not depend on ring orientation. The first sample of each
 * track probes the BVH with a ray toward +x (even-odd rule) to find which zones contain it; every
 * crossing then toggles the `(track, zone)` state. Holes (extra rings of the same zone) therefore
 * work, while overlapping rings of one zone do not.
 *
 * Boundary rule: every track sample `(x, y)` is treated as the point `(x - e^2, y - e)` for an
 * infinitesimal `e > 0`, applied to the whole track and not to the polygon, with the same rule for
 * the initial-state ray and the segment crossings. The crossings of a track against a zone are
 * therefore always consistent with the inside flag: a track that only touches a polygon vertex or
 * edge from outside has no event, one that runs through a vertex into the interior has one, and a
 * sample exactly on an edge counts as just below it (so inside when the interior lies below the
 * edge) and on a vertical edge as just west of it. Segments can still produce several events at
 * one time (a track passing a vertex that connects edges on both sides of it).
 *
 * Equal-time crossings toggle in polygon edge-row order, including zero-duration segments and
 * boundary touches. Enter/exit labels describe the ordered event stream, so at a tied time they
 * need not describe the spatial direction at that particular edge. Dwell uses the same order.
 *
 * Composition: bounds and BVH build, one probe kernel per row, an exact crossing kernel, a ray
 * parity kernel, an event-time kernel, three stable `GPUSort` passes (edge row, time, track) giving
 * `(track, time, edge row)` order, then a fourth stable sort by `(track, zone)` cell so the events
 * of every cell form a time-ordered run. Every event of a run toggles the cell's seeded inside state,
 * so its type follows from its position in the run in one step; ranks inside a track and per-track
 * counts come from binary searches; dwell is a load-balanced segmented sum
 * (`createLoadBalancedSegmentSumNodes`) of the enter/exit pair differences; visit counts, first
 * enters, last exits and the closing of still-open visits are O(1) per cell. Then `GPUCompaction` of
 * the first `trackEventCapacity` events of each track, gather kernels, and one publish kernel.
 *
 * Bounds and cost: the event scratch is `candidateCapacity` rows and every encoding sorts all of
 * them. No kernel walks a track's events serially, so a few tracks with very many events cost the
 * same as many tracks with few (dwell sums are reduced in a fixed order that depends only on the
 * cell offsets, so they are reproducible; they can differ from a sequential sum in the last bits).
 * The matrices are `trackCount * zoneCount` rows (so very large zone counts need a sparse approach). `output.events.overflow` is 1 when the candidate scratch overflowed, when a
 * track has more than `trackEventCapacity` events, or when more events were kept than
 * `output.events.ids.length`; `output.status` reports each cause separately plus the candidate
 * count the scratch needed. After a candidate overflow every result may be missing events.
 *
 * Crossing positions: `output.events.eventPositions` holds the interpolated crossing point of each event.
 * Sparse table: `output.visitTable` compacts the `(track, zone)` cells that have a visit into a bounded
 * list with visits, total dwell, first enter and last exit, so large zone counts do not need to
 * read dense matrices back.
 *
 * Spatial context: planar only (see `GPUZoneEventsProps.spatialContext`). Construction with a
 * longitude-latitude context throws instead of intersecting segments in degree space.
 *
 * Non-goals: geodesic crossings, zone polygons with orientation semantics, point-in-polygon tolerance for samples exactly on a
 * boundary, and Double-single timestamps.
 */
export class GPUZoneEvents implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUZoneEventsProps;
  /** Resolved coordinate context; always planar. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of tracks. */
  readonly trackCount: number;
  /** Number of zones (features of `zones`). */
  readonly zoneCount: number;
  /** Number of boundary edges, one per zone vertex. */
  readonly edgeCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;

  constructor(props: GPUZoneEventsProps) {
    this.id = props.id ?? 'zone-events';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    resolveGPUSecondarySpatialContext(
      this.id,
      'zones',
      props.zonesSpatialContext,
      this.spatialContext
    );
    const {tracks, zones, output} = props;
    validatePackedView(tracks.positions, ['float32x2'], `${id} tracks.positions`);
    validatePackedView(tracks.timestamps, ['float32', 'uint32x2'], `${id} tracks.timestamps`);
    validatePackedView(tracks.trackOffsets, ['uint32'], `${id} tracks.trackOffsets`);
    validatePackedView(zones.positions, ['float32x2'], `${id} zones.positions`);
    validatePackedView(zones.featureOffsets, ['uint32'], `${id} zones.featureOffsets`);
    validatePackedView(zones.polygonOffsets, ['uint32'], `${id} zones.polygonOffsets`);
    validatePackedView(zones.ringOffsets, ['uint32'], `${id} zones.ringOffsets`);
    if (tracks.timestamps.length !== tracks.positions.length) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (tracks.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    if (zones.featureOffsets.length < 2) {
      throw new Error(`${id} zones need at least one feature`);
    }
    if (zones.polygonOffsets.length < 1 || zones.ringOffsets.length < 1) {
      throw new Error(`${id} zones polygonOffsets and ringOffsets must not be empty`);
    }
    this.trackCount = tracks.trackOffsets.length - 1;
    this.zoneCount = zones.featureOffsets.length - 1;
    // One edge per zone vertex: from the vertex to the next one of its ring, closing the ring.
    this.edgeCount = zones.positions.length;
    if (this.edgeCount < 1) {
      throw new Error(`${id} zones need at least one vertex`);
    }
    validateCapacity(id, 'candidateCapacity', props.candidateCapacity);
    validateCapacity(id, 'trackEventCapacity', props.trackEventCapacity);
    if (tracks.positions.length >= 0x80000000) {
      throw new Error(`${id} supports fewer than 2^31 rows`);
    }
    const cellCount = this.trackCount * this.zoneCount;
    if (cellCount > 0xffffffff) {
      throw new Error(`${id} trackCount * zoneCount must fit in 32 bits`);
    }
    for (const [name, view, format] of [
      ['dwellTimes', output.dwellTimes, 'float32'],
      ['visitCounts', output.visitCounts, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} length must equal trackCount * zoneCount`);
        }
      }
    }
    if (output.trackEventCounts) {
      validatePackedView(output.trackEventCounts, ['uint32'], `${id} trackEventCounts`);
      if (output.trackEventCounts.length !== this.trackCount) {
        throw new Error(`${id} trackEventCounts length must equal the track count`);
      }
    }
    const {events} = output;
    validateCompactOutput(id, events);
    const capacity = events.ids.length;
    for (const [name, view, format] of [
      ['eventZones', events.eventZones, 'uint32'],
      ['eventTypes', events.eventTypes, 'uint32'],
      ['eventTimes', events.eventTimes, 'float32'],
      ['eventRows', events.eventRows, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} events.${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} events.${name} length must equal the capacity`);
        }
      }
    }
    if (events.eventPositions) {
      validatePackedView(events.eventPositions, ['float32x2'], `${id} events.eventPositions`);
      if (events.eventPositions.length !== capacity) {
        throw new Error(`${id} events.eventPositions length must equal the capacity`);
      }
    }
    const {visitTable} = output;
    if (visitTable) {
      validateCompactOutput(`${id} visitTable`, visitTable);
      const tableCapacity = visitTable.ids.length;
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
            throw new Error(`${id} visitTable.${name} length must equal the capacity`);
          }
        }
      }
    }
    for (const [name, view] of Object.entries(output.status ?? {})) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} status.${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} status.${name} must have one row`);
        }
      }
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.edgeCount);
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        events.ids,
        events.status.count,
        events.status.overflow,
        events.status.requiredCount,
        events.eventZones,
        events.eventTypes,
        events.eventTimes,
        events.eventRows,
        events.eventPositions,
        ...this.getTableViews(),
        output.dwellTimes,
        output.visitCounts,
        output.trackEventCounts,
        ...Object.values(output.status ?? {})
      ],
      [
        tracks.positions,
        tracks.timestamps,
        tracks.trackOffsets,
        zones.positions,
        zones.featureOffsets,
        zones.polygonOffsets,
        zones.ringOffsets
      ]
    );
  }

  /** Returns every node of the pipeline described in the class documentation. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, trackCount, edgeCount, leafCapacity} = this;
    const {tracks, zones, output} = props;
    const {events} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      tracks.positions,
      tracks.timestamps,
      tracks.trackOffsets,
      zones.positions,
      zones.featureOffsets,
      zones.polygonOffsets,
      zones.ringOffsets,
      events.ids,
      events.status.count,
      events.status.overflow,
      events.status.requiredCount,
      events.eventZones,
      events.eventTypes,
      events.eventTimes,
      events.eventRows,
      events.eventPositions,
      ...this.getTableViews(),
      output.dwellTimes,
      output.visitCounts,
      output.trackEventCounts,
      ...Object.values(output.status ?? {})
    ]);
    const rowCount = tracks.positions.length;
    const capacity = props.candidateCapacity;
    const cellCount = trackCount * this.zoneCount;
    const isWordMode = tracks.timestamps.format === 'uint32x2';
    const shape: ZoneEventsShape = {
      rowCount,
      trackCount,
      edgeCount,
      zoneCount: this.zoneCount,
      candidateCapacity: capacity,
      maximumEventsPerTrack: props.trackEventCapacity
    };
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    // One edge per zone vertex, closing each ring. Degenerate edges and edges of rings with fewer
    // than three vertices get a NaN start, which the bounds and crossing kernels ignore.
    const edgeStarts = createTransientView(graph, `${id}-edge-starts`, 'float32x2', edgeCount);
    const edgeEnds = createTransientView(graph, `${id}-edge-ends`, 'float32x2', edgeCount);
    const edgeZones = u32('edge-zones', edgeCount);
    nodes.push(
      createZoneEdgesNode<Parameters>(graph, {
        id: `${id}-edges`,
        zones,
        edgeStarts,
        edgeEnds,
        edgeZones
      })
    );
    const minima = createTransientView(graph, `${id}-edge-minima`, 'float32x2', edgeCount);
    const maxima = createTransientView(graph, `${id}-edge-maxima`, 'float32x2', edgeCount);
    nodes.push(
      createSpatialJoinBoundsNode<Parameters>(graph, {
        id: `${id}-bounds`,
        operation: OPERATION,
        featureCount: edgeCount,
        source: {kind: 'segments', starts: edgeStarts, ends: edgeEnds},
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
        positions: tracks.positions,
        trackOffsets: tracks.trackOffsets,
        bvh,
        state,
        candidatePairs
      }),
      createZoneCrossingNode<Parameters>(graph, {
        id: `${id}-crossing`,
        shape,
        positions: tracks.positions,
        edgeStarts: edgeStarts,
        edgeEnds: edgeEnds,
        edgeZones: edgeZones,
        state,
        candidatePairs,
        candidateKinds,
        candidateParameters
      }),
      createZoneInitialStateNode<Parameters>(graph, {
        id: `${id}-initial-state`,
        shape,
        trackOffsets: tracks.trackOffsets,
        edgeZones: edgeZones,
        candidatePairs,
        candidateKinds,
        initialInside
      }),
      createZoneEventTimeNode<Parameters>(graph, {
        id: `${id}-event-time`,
        shape,
        isWordMode,
        timestamps: tracks.timestamps,
        trackOffsets: tracks.trackOffsets,
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
    const {visitTable} = output;
    const needsSpans = Boolean(visitTable?.firstEnterTimes || visitTable?.lastExitTimes);
    const firstEnterTimes = needsSpans ? f32('first-enter-times', cellCount) : undefined;
    const lastExitTimes = needsSpans ? f32('last-exit-times', cellCount) : undefined;
    const dwellTimes = output.dwellTimes ?? f32('dwell-times', cellCount);
    const visitCounts = output.visitCounts ?? u32('visit-counts', cellCount);
    const trackEventCounts = output.trackEventCounts ?? u32('track-event-counts', trackCount);
    // Parallel replacement of the per-track walk: a stable sort by (track, zone) cell turns the
    // events of each cell into a time-ordered run, so event types, visit counts, spans and the
    // closing of open visits are O(1) per event or cell, and dwell is a load-balanced segmented
    // sum of the enter/exit pair differences.
    const cellKeys = u32('cell-keys', capacity);
    const cellIdentity = u32('cell-identity', capacity);
    const sortedCellKeys = u32('sorted-cell-keys', capacity);
    const cellOrder = u32('cell-order', capacity);
    const cellOffsets = u32('cell-offsets', cellCount + 1);
    const cellTimes = f32('cell-times', capacity);
    nodes.push(
      createZoneSortedColumnsNode<Parameters>(graph, {
        id: `${id}-sorted-columns`,
        shape,
        state,
        order,
        candidatePairs,
        edgeZones: edgeZones,
        candidateTimes,
        sortedZones,
        sortedTimes,
        sortedRows
      }),
      createZoneEventRanksNode<Parameters>(graph, {
        id: `${id}-event-ranks`,
        shape,
        state,
        sortedTracks,
        eventRanks
      }),
      createZoneTrackCountsNode<Parameters>(graph, {
        id: `${id}-track-counts`,
        shape,
        state,
        sortedTracks,
        trackEventCounts
      }),
      createZoneCellKeysNode<Parameters>(graph, {
        id: `${id}-cell-keys`,
        shape,
        state,
        sortedTracks,
        sortedZones,
        cellKeys,
        identity: cellIdentity
      }),
      ...new GPUSort({
        id: `${id}-cell-sort`,
        keys: cellKeys,
        values: cellIdentity,
        outputKeys: sortedCellKeys,
        outputValues: cellOrder,
        keyBits: getKeyBits(cellCount)
      }).getCommandNodes(graph),
      createSortedSegmentOffsetsNode<Parameters>(graph, {
        id: `${id}-cell-offsets`,
        operation: OPERATION,
        variant: 'cell-offsets',
        segmentCount: cellCount,
        sortedKeys: sortedCellKeys,
        segmentOffsets: cellOffsets
      }),
      createZoneCellEventsNode<Parameters>(graph, {
        id: `${id}-cell-events`,
        shape,
        state,
        cellOrder,
        sortedCellKeys,
        cellOffsets,
        initialInside,
        sortedTimes,
        eventTypes,
        cellTimes
      }),
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-dwell`,
        operation: OPERATION,
        variant: 'dwell-sum',
        offsets: cellOffsets,
        itemCapacity: capacity,
        accumulatorType: 'f32',
        output: dwellTimes,
        bindings: [
          {name: 'initialInside', view: initialInside, type: 'u32', access: 'read'},
          {name: 'cellTimes', view: cellTimes, type: 'f32', access: 'read'}
        ],
        declarations: ZONE_DWELL_CONTRIBUTION_WGSL
      }),
      createZoneCellFinalizeNode<Parameters>(graph, {
        id: `${id}-cell-finalize`,
        shape,
        isWordMode,
        timestamps: tracks.timestamps,
        trackOffsets: tracks.trackOffsets,
        cellOffsets,
        initialInside,
        cellTimes,
        dwellTimes,
        visitCounts
      })
    );
    if (firstEnterTimes && lastExitTimes) {
      nodes.push(
        createZoneCellSpansNode<Parameters>(graph, {
          id: `${id}-cell-spans`,
          shape,
          isWordMode,
          timestamps: tracks.timestamps,
          trackOffsets: tracks.trackOffsets,
          cellOffsets,
          initialInside,
          cellTimes,
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
          positions: tracks.positions,
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
    if (output.status && Object.values(output.status).some(Boolean)) {
      nodes.push(
        createZoneDiagnosticsNode<Parameters>(graph, {
          id: `${id}-diagnostics`,
          shape,
          eventCapacity: events.ids.length,
          state,
          keptTotal,
          ...output.status
        })
      );
    }
    const columns: ZoneGatherColumn[] = [
      {source: sortedTracks, destination: events.ids, sentinel: '0xffffffffu'}
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
          capacity: events.ids.length,
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
        requiredCount: keptTotal,
        output: events,
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
    const table = this.props.output.visitTable;
    return table
      ? [
          table.ids,
          table.status.count,
          table.status.overflow,
          table.status.requiredCount,
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
    const table = this.props.output.visitTable!;
    const {cellCount} = cells;
    const capacity = table.ids.length;
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
    const columns: ZoneTableColumn[] = [{kind: 'track', destination: table.ids}];
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
            zoneCount: this.zoneCount,
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
        requiredCount: keptTotal,
        output: table
      })
    );
    return nodes;
  }
}

/** Bits needed to sort keys in `[0, maximumKey]`. */
function getKeyBits(maximumKey: number): number {
  return Math.max(1, 32 - Math.clz32(Math.max(maximumKey, 1)));
}
