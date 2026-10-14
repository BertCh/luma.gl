// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar,
  GPUTrajectoryGeometryPort
} from '../../analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {GPUTrajectoryMetrics} from '../trajectory-analysis/index';
import {GPUZoneEvents} from '../trajectory-zones/index';
import type {
  GPUZoneEventOutput,
  GPUZoneEventsStatus,
  GPUZoneVisitTableOutput
} from '../trajectory-zones/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUFleetDwellRecipe';

/**
 * Caller-owned per-zone statistics table of the fleet dwell recipes. Every view is optional.
 *
 * The table is dense: `GPUGroupStatistics` runs with `keyCount` equal to the table rows, so row `k`
 * is zone `k` for every zone, including zones nothing visited (`counts` 0, `sumValues` 0, NaN for
 * `means` and `maximums`). It can be drawn per zone without an expansion step.
 */
export type GPUFleetDwellZoneTable = {
  /** Zone keys: row `k` holds `k`. */
  keys?: GraphDataView<'uint32'>;
  /** Rows per zone: stops (stops variant) or visiting tracks (zone events variant). */
  counts?: GraphDataView<'uint32'>;
  /** Optional one-row status words of the zone statistics stage. */
  status?: {
    /** Number of table rows (the zone count). */
    count?: GPUStatusScalar;
    /** Overflow flag (always 0 in dense mode). */
    overflow?: GPUStatusScalar;
    /** Unclamped group count (equals the row count in dense mode). */
    requiredCount?: GPUStatusScalar;
  };
  /** Total dwell time per zone. */
  sumValues?: GraphDataView<'float32'>;
  /** Mean dwell per stop (stops variant) or per visiting track (zone events variant). */
  means?: GraphDataView<'float32'>;
  /** Longest dwell per zone. */
  maximums?: GraphDataView<'float32'>;
};

/** Named per-zone table of the fleet dwell recipes. */
export type GPUFleetDwellZoneTableViews = {
  keys: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
  /** One-row status words of the zone statistics stage. */
  status: {
    count: GPUStatusScalar;
    overflow: GPUStatusScalar;
    /** Unclamped group count (equals the row count in dense mode). */
    requiredCount: GPUStatusScalar;
  };
  sumValues: GraphDataView<'float32'>;
  means: GraphDataView<'float32'>;
  maximums: GraphDataView<'float32'>;
};

function createZoneTable<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  capacity: number,
  provided: GPUFleetDwellZoneTable = {}
): GPUFleetDwellZoneTableViews {
  return {
    keys: getOrCreateView(graph, `${id}-zone-keys`, 'uint32', capacity, provided.keys),
    counts: getOrCreateView(graph, `${id}-zone-counts`, 'uint32', capacity, provided.counts),
    status: {
      count: getOrCreateView(graph, `${id}-zone-count`, 'uint32', 1, provided.status?.count),
      overflow: getOrCreateView(
        graph,
        `${id}-zone-overflow`,
        'uint32',
        1,
        provided.status?.overflow
      ),
      requiredCount: getOrCreateView(
        graph,
        `${id}-zone-required-count`,
        'uint32',
        1,
        provided.status?.requiredCount
      )
    },
    sumValues: getOrCreateView(graph, `${id}-zone-sums`, 'float32', capacity, provided.sumValues),
    means: getOrCreateView(graph, `${id}-zone-means`, 'float32', capacity, provided.means),
    maximums: getOrCreateView(graph, `${id}-zone-maximums`, 'float32', capacity, provided.maximums)
  };
}

function addZoneStatistics<Parameters>(
  builder: RecipeBuilder<Parameters>,
  id: string,
  keys: GraphDataView<'uint32'>,
  mask: GraphDataView<'uint32'>,
  values: GraphDataView<'float32'>,
  table: GPUFleetDwellZoneTableViews
): void {
  const capacity = table.keys.length;
  builder.add(
    new GPUGroupStatistics({
      id: `${id}-zone-statistics`,
      keys,
      keyCount: capacity,
      mask,
      columns: [
        {
          values,
          statistics: ['sum', 'mean', 'maximum'],
          output: {
            sums: getOrCreateView(builder.graph, `${id}-zone-sum-words`, 'uint32x2', capacity),
            sumValues: table.sumValues,
            means: table.means,
            maximums: table.maximums
          }
        }
      ],
      output: {
        keys: table.keys,
        counts: table.counts,
        status: table.status
      }
    })
  );
}

/**
 * Polygon zones in the four-level `GPUPolygonGeometryPort` layout (one packed `positions` view),
 * one zone per feature. Optional `sourceIds` are stable zone IDs used as the table rows instead of
 * feature rows; each must be below `zoneCapacity`.
 */
export type GPUFleetDwellPolygonZones = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Maximum `(stop, zone)` bounding-box candidates per encoding. */
  candidateCapacity: number;
};

/**
 * Tracks as the movement contributors take them (`GPUPackedGeometryPort<GPUTrajectoryGeometryPort>`: packed `positions`
 * sorted by track then time, `trackOffsets` with `trackCount + 1` entries, float32 or Int64
 * `uint32x2` `timestamps`).
 */
export type GPUFleetDwellTrajectories = GPUPackedGeometryPort<GPUTrajectoryGeometryPort> & {
  /** Low parts of double-single `float32` timestamps (stops variant only). */
  timestampsLow?: GraphDataView<'float32'>;
};

/** Properties for {@link addFleetDwellRecipe}. */
export type GPUFleetDwellRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    stops?: {
      ids?: GraphDataView<'uint32'>;
      status?: {
        count?: GPUStatusScalar;
        overflow?: GPUStatusScalar;
        requiredCount?: GPUStatusScalar;
      };
      centroids?: GraphDataView<'float32x2'>;
      durations?: GraphDataView<'float32'>;
      startRows?: GraphDataView<'uint32'>;
      endRows?: GraphDataView<'uint32'>;
    };
    table?: GPUFleetDwellZoneTable;
    /** Mirrors `result.status`. */
    status?: {
      joinOverflow?: GPUStatusScalar;
      joinUncertainCount?: GPUStatusScalar;
    };
  },
  {stopZones?: GraphDataView<'uint32'>; stopMask?: GraphDataView<'uint32'>}
> & {
  /** Prefix for every node and transient ID. Defaults to `'fleet-dwell'`. */
  id?: string;
  /**
   * Coordinate space of the positions, stop centroids and zone polygons. The stops variant is planar only: the stop-to-zone join and the zone edges are straight lines in the coordinate plane,
   * so degrees would give wrong containment near the antimeridian and poles, even though dwell is
   * topological. A
   * longitude-latitude context throws; project first with `createGPULocalFrame` and pass its
   * `spatialContext` (planar, meters). Omitted means native planar. The resolved context is
   * returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** Planar tracks. */
  trajectories: GPUFleetDwellTrajectories;
  /** `getGPUTrajectoryMetricsParameterValues` view (stop speed threshold and minimum duration). */
  parameters: GraphDataView<'float32'>;
  /** Stop capacity of the bounded stop list. */
  stopCapacity: number;
  /** Zones joined to the stop centroids. */
  zones: GPUFleetDwellPolygonZones;
  /** Rows of the per-zone table; defaults to the zone count. */
  zoneCapacity?: number;
};

/** Named outputs of {@link addFleetDwellRecipe}. */
export type GPUFleetDwellRecipeResult = GPURecipeResult & {
  /** Resolved coordinate context shared by the metrics, join and zone stages. */
  spatialContext: GPUSpatialContext;
  stops: {
    ids: GraphDataView<'uint32'>;
    centroids: GraphDataView<'float32x2'>;
    durations: GraphDataView<'float32'>;
    /** One-row status words of the stop list (stage `metrics`). */
    status: {
      count: GPUStatusScalar;
      overflow: GPUStatusScalar;
      /** Unclamped stop count, for sizing `stopCapacity`. */
      requiredCount: GPUStatusScalar;
    };
  };
  stopZones: GraphDataView<'uint32'>;
  /** Per-zone stop count (`counts`), total, mean and longest dwell. */
  table: GPUFleetDwellZoneTableViews;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the stop-to-zone join was incomplete (stage `zone-join`). */
    joinOverflow: GPUStatusScalar;
    /**
     * One-row count of stop-to-zone classifications resolved as uncertain (those stops get no
     * zone), stage `zone-join`.
     */
    joinUncertainCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Fleet dwell recipe (stops variant): where do vehicles stop and for how long, per zone.
 *
 * Chain: `GPUTrajectoryMetrics` stops -> `GPUPointInPolygonJoin` of the stop centroids against the
 * zones -> adapter mask (stop slots past the GPU-written stop count are excluded) ->
 * dense `GPUGroupStatistics` (`keyCount` = zone rows) of stop duration keyed by zone, one row per
 * zone. Stops outside every zone carry the reserved no-zone key and are skipped.
 *
 * Spatial context: the recipe resolves one planar `spatialContext` (default native planar; a
 * `createGPULocalFrame` context for meters) and passes the same object to the metrics, join and
 * zone stages. Previously it always emitted a native planar context to the metrics stage regardless
 * of the caller's data. Stop durations use the timestamps' unit; the metrics stage speed threshold
 * uses context length units per time unit. Longitude-latitude throws.
 */
export function addFleetDwellRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUFleetDwellRecipeProps
): GPUFleetDwellRecipeResult {
  const id = props.id ?? 'fleet-dwell';
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const capacity = props.stopCapacity;
  const zoneCount = props.zones.featureOffsets.length - 1;
  const {trajectories} = props;
  assertRecipe(ID, capacity >= 1, 'stopCapacity must be positive');
  assertRecipe(ID, zoneCount >= 1, 'needs at least one zone');
  const provided = outputs.stops ?? {};
  const stops = {
    ids: getOrCreateView(graph, `${id}-stop-ids`, 'uint32', capacity, provided.ids),
    status: {
      count: getOrCreateView(graph, `${id}-stop-count`, 'uint32', 1, provided.status?.count),
      overflow: getOrCreateView(
        graph,
        `${id}-stop-overflow`,
        'uint32',
        1,
        provided.status?.overflow
      ),
      requiredCount: getOrCreateView(
        graph,
        `${id}-stop-required-count`,
        'uint32',
        1,
        provided.status?.requiredCount
      )
    },
    centroids: getOrCreateView(
      graph,
      `${id}-stop-centroids`,
      'float32x2',
      capacity,
      provided.centroids
    ),
    durations: getOrCreateView(
      graph,
      `${id}-stop-durations`,
      'float32',
      capacity,
      provided.durations
    )
  };
  builder.add(
    new GPUTrajectoryMetrics({
      spatialContext,
      id: `${id}-metrics`,
      positions: trajectories.positions,
      timestamps: trajectories.timestamps,
      timestampsLow: trajectories.timestampsLow,
      trackOffsets: trajectories.trackOffsets,
      parameters: props.parameters,
      output: {
        stops: {
          ids: stops.ids,
          status: stops.status,
          centroids: stops.centroids,
          durations: stops.durations,
          startRows: provided.startRows,
          endRows: provided.endRows
        }
      }
    })
  );

  const stopZones = getOrCreateView(
    graph,
    `${id}-stop-zones`,
    'uint32',
    capacity,
    scratch.stopZones
  );
  const joinOverflow = getOrCreateView(
    graph,
    `${id}-join-overflow`,
    'uint32',
    1,
    outputs.status?.joinOverflow
  );
  const joinUncertainCount = getOrCreateView(
    graph,
    `${id}-join-uncertain-count`,
    'uint32',
    1,
    outputs.status?.joinUncertainCount
  );
  builder.add(
    new GPUPointInPolygonJoin({
      id: `${id}-join`,
      spatialContext,
      points: {kind: 'points', positions: stops.centroids},
      polygons: props.zones,
      candidateCapacity: props.zones.candidateCapacity,
      output: {
        pointFeatureIds: stopZones,
        status: {overflow: joinOverflow, uncertainCount: joinUncertainCount}
      }
    })
  );

  // Adapter: the stop list is compacted at the front; slots past the GPU-written count are stale.
  const mask = getOrCreateView(graph, `${id}-stop-mask`, 'uint32', capacity, scratch.stopMask);
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-stop-mask-kernel`,
      operation: 'GPURecipeFleetDwell',
      variant: 'stop-mask',
      bindings: [
        {name: 'stopCount', view: stops.status.count, type: 'u32', access: 'read'},
        {name: 'stopMask', view: mask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: capacity,
      body: 'stopMask[stopMaskOffset + index] = select(0u, 1u, index < stopCount[stopCountOffset]);'
    })
  );

  const table = createZoneTable(graph, id, props.zoneCapacity ?? zoneCount, outputs.table);
  addZoneStatistics(builder, id, stopZones, mask, stops.durations, table);
  return {
    spatialContext,
    contributors: builder.contributors,
    stops,
    stopZones,
    table,
    outputs: {stops, table},
    intermediates: {stopZones},
    status: {
      joinOverflow,
      joinUncertainCount,
      stages: [
        {
          stage: 'metrics',
          status: stops.status
        },
        {
          stage: 'zone-join',
          status: {overflow: joinOverflow, uncertainCount: joinUncertainCount}
        },
        {
          stage: 'zone-statistics',
          status: table.status
        }
      ]
    }
  };
}

/** Properties for {@link addFleetDwellZoneEventsRecipe}. */
export type GPUFleetDwellZoneEventsRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    events?: Omit<Partial<GPUZoneEventOutput>, 'status'> & {
      status?: Partial<GPUZoneEventOutput['status']>;
    };
    dwellTimes?: GraphDataView<'float32'>;
    visitCounts?: GraphDataView<'uint32'>;
    trackEventCounts?: GraphDataView<'uint32'>;
    /** Mirrors `result.status`. */
    status?: GPUZoneEventsStatus;
    table?: GPUFleetDwellZoneTable;
  },
  {
    visitTable?: GPUZoneVisitTableOutput;
    cellZones?: GraphDataView<'uint32'>;
    cellMask?: GraphDataView<'uint32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'fleet-dwell-zone-events'`. */
  id?: string;
  /**
   * Coordinate space of the positions and zone edges. The zone events variant is planar only: zone edges and track segments are straight lines in the coordinate plane,
   * which is not the geometry of degrees. A
   * longitude-latitude context throws; project first with `createGPULocalFrame` and pass its
   * `spatialContext` (planar, meters). Omitted means native planar. The resolved context is
   * returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** Planar tracks (`timestampsLow` is not used by this variant). */
  trajectories: GPUFleetDwellTrajectories;
  /**
   * Zones in the four-level `GPUPolygonGeometryPort` layout (one packed `positions` view), one zone
   * per feature; holes and extra parts share the zone index.
   */
  zones: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /** Segment-edge candidate scratch rows. */
  candidateCapacity: number;
  /** Compile-time number of events kept per track. */
  trackEventCapacity: number;
  /**
   * Event capacity (`events.ids` rows) when `outputs.events.ids` is not given. Defaults to
   * `trackCount * trackEventCapacity`.
   */
  eventCapacity?: number;
};

/** Named outputs of {@link addFleetDwellZoneEventsRecipe}. */
export type GPUFleetDwellZoneEventsRecipeResult = GPURecipeResult & {
  /** Resolved coordinate context of the zone events stage. */
  spatialContext: GPUSpatialContext;
  events: {
    ids: GraphDataView<'uint32'>;
    /** One-row status words of the event list (stage `zone-events`). */
    status: {
      count: GPUStatusScalar;
      overflow: GPUStatusScalar;
      /** Unclamped event count, for sizing `eventCapacity`. */
      requiredCount: GPUStatusScalar;
    };
    zones: GraphDataView<'uint32'>;
    types: GraphDataView<'uint32'>;
    times: GraphDataView<'float32'>;
    rows: GraphDataView<'uint32'>;
    /** Crossing position per event; present only when `props.events.eventPositions` was given. */
    positions?: GraphDataView<'float32x2'>;
  };
  /** The caller's sparse visit table, when `props.visitTable` was given. */
  visitTable?: GPUZoneVisitTableOutput;
  dwellTimes: GraphDataView<'float32'>;
  visitCounts: GraphDataView<'uint32'>;
  trackEventCounts: GraphDataView<'uint32'>;
  /** Per-zone count of visiting tracks and their total, mean and longest dwell. */
  table: GPUFleetDwellZoneTableViews;
  /**
   * One-row `GPUZoneEvents` status words next to the per-stage `stages`: `candidateCount` is the
   * unclamped candidate count (the `candidateCapacity` needed), and `candidateOverflow`,
   * `trackOverflow` and `eventOverflow` split `events.status.overflow` into candidate scratch,
   * per-track bound and event list capacity.
   */
  status: {
    candidateCount: GPUStatusScalar;
    candidateOverflow: GPUStatusScalar;
    trackOverflow: GPUStatusScalar;
    eventOverflow: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Fleet dwell recipe (zone events variant): enter and exit events of tracks against zones, dwell
 * per (track, zone) and a per-zone roll-up.
 *
 * Chain: `GPUZoneEvents` -> adapter (zone key and visited mask per `(track, zone)` cell) ->
 * dense `GPUGroupStatistics` of dwell time keyed by zone, one row per zone. Unlike the stops variant this counts any time
 * spent inside a zone, moving or not, and needs no point-in-polygon join. `table.counts` is the
 * number of tracks that visited the zone. `events.eventPositions` and `visitTable` are optional
 * pass-through outputs of `GPUZoneEvents` (crossing positions and a sparse `(track, zone)` table).
 *
 * Status: stages `zone-events` (event count, unclamped `requiredCount`, event overflow and
 * `candidateOverflow`; the unclamped candidate count is `status.candidateCount`),
 * `visit-table` (with `scratch.visitTable`) and `zone-statistics`.
 */
export function addFleetDwellZoneEventsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUFleetDwellZoneEventsRecipeProps
): GPUFleetDwellZoneEventsRecipeResult {
  const id = props.id ?? 'fleet-dwell-zone-events';
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const {trajectories, zones} = props;
  const trackCount = trajectories.trackOffsets.length - 1;
  const zoneCount = zones.featureOffsets.length - 1;
  assertRecipe(ID, trackCount >= 1 && zoneCount >= 1, 'needs tracks and zones');
  const cellCount = trackCount * zoneCount;
  const eventCapacity = props.eventCapacity ?? trackCount * props.trackEventCapacity;
  const given = outputs.events ?? {};
  const events = {
    ids: getOrCreateView(graph, `${id}-event-ids`, 'uint32', eventCapacity, given.ids),
    status: {
      count: getOrCreateView(graph, `${id}-event-count`, 'uint32', 1, given.status?.count),
      overflow: getOrCreateView(graph, `${id}-event-overflow`, 'uint32', 1, given.status?.overflow),
      requiredCount: getOrCreateView(
        graph,
        `${id}-event-required-count`,
        'uint32',
        1,
        given.status?.requiredCount
      )
    },
    zones: getOrCreateView(graph, `${id}-event-zones`, 'uint32', eventCapacity, given.eventZones),
    types: getOrCreateView(graph, `${id}-event-types`, 'uint32', eventCapacity, given.eventTypes),
    times: getOrCreateView(graph, `${id}-event-times`, 'float32', eventCapacity, given.eventTimes),
    rows: getOrCreateView(graph, `${id}-event-rows`, 'uint32', eventCapacity, given.eventRows),
    positions: given.eventPositions
  };
  const dwellTimes = getOrCreateView(
    graph,
    `${id}-dwell-times`,
    'float32',
    cellCount,
    outputs.dwellTimes
  );
  const visitCounts = getOrCreateView(
    graph,
    `${id}-visit-counts`,
    'uint32',
    cellCount,
    outputs.visitCounts
  );
  const trackEventCounts = getOrCreateView(
    graph,
    `${id}-track-event-counts`,
    'uint32',
    trackCount,
    outputs.trackEventCounts
  );
  const givenDiagnostics = outputs.status ?? {};
  const diagnostics = {
    candidateCount: getOrCreateView(
      graph,
      `${id}-candidate-count`,
      'uint32',
      1,
      givenDiagnostics.candidateCount
    ),
    candidateOverflow: getOrCreateView(
      graph,
      `${id}-candidate-overflow`,
      'uint32',
      1,
      givenDiagnostics.candidateOverflow
    ),
    trackOverflow: getOrCreateView(
      graph,
      `${id}-track-overflow`,
      'uint32',
      1,
      givenDiagnostics.trackOverflow
    ),
    eventOverflow: getOrCreateView(
      graph,
      `${id}-event-overflow-flag`,
      'uint32',
      1,
      givenDiagnostics.eventOverflow
    )
  };
  builder.add(
    new GPUZoneEvents({
      id: `${id}-events`,
      spatialContext,
      tracks: {
        kind: 'trajectories',
        positions: trajectories.positions,
        timestamps: trajectories.timestamps,
        trackOffsets: trajectories.trackOffsets
      },
      zones,
      candidateCapacity: props.candidateCapacity,
      trackEventCapacity: props.trackEventCapacity,
      output: {
        events: {
          ids: events.ids,
          status: events.status,
          eventZones: events.zones,
          eventTypes: events.types,
          eventTimes: events.times,
          eventRows: events.rows,
          eventPositions: events.positions
        },
        visitTable: scratch.visitTable,
        dwellTimes,
        visitCounts,
        trackEventCounts,
        status: diagnostics
      }
    })
  );

  // Adapter: one group row per (track, zone) cell, keyed by zone, masked to visited cells.
  const cellZones = getOrCreateView(
    graph,
    `${id}-cell-zones`,
    'uint32',
    cellCount,
    scratch.cellZones
  );
  const cellMask = getOrCreateView(graph, `${id}-cell-mask`, 'uint32', cellCount, scratch.cellMask);
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-keys-kernel`,
      operation: 'GPURecipeFleetDwell',
      variant: 'cell-keys',
      bindings: [
        {name: 'visitCounts', view: visitCounts, type: 'u32', access: 'read'},
        {name: 'cellZones', view: cellZones, type: 'u32', access: 'read_write'},
        {name: 'cellMask', view: cellMask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: cellCount,
      declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;`,
      body: `cellZones[cellZonesOffset + index] = index % ZONE_COUNT;
  cellMask[cellMaskOffset + index] = select(0u, 1u, visitCounts[visitCountsOffset + index] > 0u);`
    })
  );
  const table = createZoneTable(graph, id, zoneCount, outputs.table);
  addZoneStatistics(builder, id, cellZones, cellMask, dwellTimes, table);
  return {
    spatialContext,
    contributors: builder.contributors,
    events,
    visitTable: scratch.visitTable,
    dwellTimes,
    visitCounts,
    trackEventCounts,
    table,
    outputs: {events, dwellTimes, visitCounts, trackEventCounts, table},
    intermediates: {visitTable: scratch.visitTable, cellZones, cellMask},
    status: {
      ...diagnostics,
      stages: [
        {
          stage: 'zone-events',
          status: {...events.status, candidateOverflow: diagnostics.candidateOverflow}
        },
        ...(scratch.visitTable
          ? [
              {
                stage: 'visit-table',
                status: scratch.visitTable.status
              }
            ]
          : []),
        {
          stage: 'zone-statistics',
          status: table.status
        }
      ]
    }
  };
}
