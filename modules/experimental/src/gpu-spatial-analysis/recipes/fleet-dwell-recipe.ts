// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {GPUTrajectoryMetrics} from '../trajectory-analysis/index';
import {GPUZoneEvents} from '../trajectory-zones/index';
import type {
  GPUZoneEventOutput,
  GPUZoneEventsDiagnostics,
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
  /** One-row number of table rows (the zone count). */
  count?: GraphDataView<'uint32'>;
  /** One-row overflow flag (always 0 in dense mode). */
  overflow?: GraphDataView<'uint32'>;
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
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
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
    count: getOrCreateView(graph, `${id}-zone-count`, 'uint32', 1, provided.count),
    overflow: getOrCreateView(graph, `${id}-zone-overflow`, 'uint32', 1, provided.overflow),
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
        count: table.count,
        overflow: table.overflow
      }
    })
  );
}

/** Polygon zones as a GeoArrow-style polygon set (static topology). */
export type GPUFleetDwellPolygonZones = {
  /** Flattened polygon vertices. */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets, `zoneCount + 1` entries. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Optional stable zone ids used as the table rows instead of feature rows; each must be below `zoneCapacity`. */
  featureIds?: GraphDataView<'uint32'>;
  /** Maximum `(stop, zone)` bounding-box candidates per encoding. */
  candidateCapacity: number;
};

/** Properties for {@link addFleetDwellRecipe}. */
export type GPUFleetDwellRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    stops?: {
      ids?: GraphDataView<'uint32'>;
      count?: GraphDataView<'uint32'>;
      overflow?: GraphDataView<'uint32'>;
      centroids?: GraphDataView<'float32x2'>;
      durations?: GraphDataView<'float32'>;
      startRows?: GraphDataView<'uint32'>;
      endRows?: GraphDataView<'uint32'>;
    };
    table?: GPUFleetDwellZoneTable;
    joinOverflow?: GraphDataView<'uint32'>;
  },
  {stopZones?: GraphDataView<'uint32'>; stopMask?: GraphDataView<'uint32'>}
> & {
  /** Prefix for every node and transient ID. Defaults to `'fleet-dwell'`. */
  id?: string;
  /** Planar sample positions sorted by track then time. */
  positions: GraphDataView<'float32x2'>;
  /** Sample times: float32 relative times (or float32 high parts with `timestampsLow`), or Int64 words. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Low parts of double-single `float32` timestamps. */
  timestampsLow?: GraphDataView<'float32'>;
  /** `trackCount + 1` row offsets. */
  trackOffsets: GraphDataView<'uint32'>;
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
  stops: {
    ids: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    centroids: GraphDataView<'float32x2'>;
    durations: GraphDataView<'float32'>;
  };
  stopZones: GraphDataView<'uint32'>;
  joinOverflow: GraphDataView<'uint32'>;
  /** Per-zone stop count (`counts`), total, mean and longest dwell. */
  table: GPUFleetDwellZoneTableViews;
};

/**
 * Fleet dwell recipe (stops variant): where do vehicles stop and for how long, per zone.
 *
 * Chain: `GPUTrajectoryMetrics` stops -> `GPUPointInPolygonJoin` of the stop centroids against the
 * zones -> adapter mask (stop slots past the GPU-written stop count are excluded) ->
 * dense `GPUGroupStatistics` (`keyCount` = zone rows) of stop duration keyed by zone, one row per
 * zone. Stops outside every zone carry the reserved no-zone key and are skipped.
 */
export function addFleetDwellRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUFleetDwellRecipeProps
): GPUFleetDwellRecipeResult {
  const id = props.id ?? 'fleet-dwell';
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const capacity = props.stopCapacity;
  const zoneCount = props.zones.featureOffsets.length - 1;
  assertRecipe(ID, capacity >= 1, 'stopCapacity must be positive');
  assertRecipe(ID, zoneCount >= 1, 'needs at least one zone');
  const provided = outputs.stops ?? {};
  const stops = {
    ids: getOrCreateView(graph, `${id}-stop-ids`, 'uint32', capacity, provided.ids),
    count: getOrCreateView(graph, `${id}-stop-count`, 'uint32', 1, provided.count),
    overflow: getOrCreateView(graph, `${id}-stop-overflow`, 'uint32', 1, provided.overflow),
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
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      id: `${id}-metrics`,
      positions: props.positions,
      timestamps: props.timestamps,
      timestampsLow: props.timestampsLow,
      trackOffsets: props.trackOffsets,
      parameters: props.parameters,
      stops: {
        output: {ids: stops.ids, count: stops.count, overflow: stops.overflow},
        centroids: stops.centroids,
        durations: stops.durations,
        startRows: provided.startRows,
        endRows: provided.endRows
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
    outputs.joinOverflow
  );
  builder.add(
    new GPUPointInPolygonJoin({
      id: `${id}-join`,
      points: stops.centroids,
      polygonPositions: props.zones.polygonPositions,
      featureOffsets: props.zones.featureOffsets,
      polygonOffsets: props.zones.polygonOffsets,
      ringOffsets: props.zones.ringOffsets,
      featureIds: props.zones.featureIds,
      candidateCapacity: props.zones.candidateCapacity,
      pointFeatureIds: stopZones,
      overflow: joinOverflow
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
        {name: 'stopCount', view: stops.count, type: 'u32', access: 'read'},
        {name: 'stopMask', view: mask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: capacity,
      body: 'stopMask[stopMaskOffset + index] = select(0u, 1u, index < stopCount[stopCountOffset]);'
    })
  );

  const table = createZoneTable(graph, id, props.zoneCapacity ?? zoneCount, outputs.table);
  addZoneStatistics(builder, id, stopZones, mask, stops.durations, table);
  return {
    contributors: builder.contributors,
    stops,
    stopZones,
    joinOverflow,
    table,
    outputs: {stops, table},
    intermediates: {stopZones},
    status: {
      stages: [
        {stage: 'metrics', status: {count: stops.count, overflow: stops.overflow}},
        {stage: 'zone-join', status: {overflow: joinOverflow}},
        {stage: 'zone-statistics', status: {count: table.count, overflow: table.overflow}}
      ]
    }
  };
}

/** Properties for {@link addFleetDwellZoneEventsRecipe}. */
export type GPUFleetDwellZoneEventsRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    events?: Omit<Partial<GPUZoneEventOutput>, 'output'> & {
      output?: Partial<GPUZoneEventOutput['output']>;
    };
    dwellTimes?: GraphDataView<'float32'>;
    visitCounts?: GraphDataView<'uint32'>;
    trackEventCounts?: GraphDataView<'uint32'>;
    diagnostics?: GPUZoneEventsDiagnostics;
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
  /** Planar sample positions sorted by track then time. */
  positions: GraphDataView<'float32x2'>;
  /** Float32 relative times or Int64 words. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** `trackCount + 1` row offsets. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Zone boundary edge starts (rings and holes of one zone share its index). */
  edgeStarts: GraphDataView<'float32x2'>;
  /** Zone boundary edge ends. */
  edgeEnds: GraphDataView<'float32x2'>;
  /** Zone index per edge. */
  edgeZones: GraphDataView<'uint32'>;
  /** Number of zones. */
  zoneCount: number;
  /** Segment-edge candidate scratch rows. */
  candidateCapacity: number;
  /** Compile-time event bound per track. */
  maxEventsPerTrack: number;
  /** Event capacity (`events.output.ids` rows) when `events` is not given. Defaults to `trackCount * maxEventsPerTrack`. */
  eventCapacity?: number;
};

/** Named outputs of {@link addFleetDwellZoneEventsRecipe}. */
export type GPUFleetDwellZoneEventsRecipeResult = GPURecipeResult & {
  events: {
    ids: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
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
  /**
   * One-row `GPUZoneEvents` diagnostics: `candidateCount` is the unclamped candidate count (the
   * `candidateCapacity` needed), and `candidateOverflow`, `trackOverflow` and `eventOverflow` split
   * `events.overflow` into candidate scratch, per-track bound and event list capacity.
   */
  diagnostics: {
    candidateCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
    trackOverflow: GraphDataView<'uint32'>;
    eventOverflow: GraphDataView<'uint32'>;
  };
  /** Per-zone count of visiting tracks and their total, mean and longest dwell. */
  table: GPUFleetDwellZoneTableViews;
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
 */
export function addFleetDwellZoneEventsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUFleetDwellZoneEventsRecipeProps
): GPUFleetDwellZoneEventsRecipeResult {
  const id = props.id ?? 'fleet-dwell-zone-events';
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const trackCount = props.trackOffsets.length - 1;
  const {zoneCount} = props;
  assertRecipe(ID, trackCount >= 1 && zoneCount >= 1, 'needs tracks and zones');
  const cellCount = trackCount * zoneCount;
  const eventCapacity = props.eventCapacity ?? trackCount * props.maxEventsPerTrack;
  const given = outputs.events ?? {};
  const events = {
    ids: getOrCreateView(graph, `${id}-event-ids`, 'uint32', eventCapacity, given.output?.ids),
    count: getOrCreateView(graph, `${id}-event-count`, 'uint32', 1, given.output?.count),
    overflow: getOrCreateView(graph, `${id}-event-overflow`, 'uint32', 1, given.output?.overflow),
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
  const givenDiagnostics = outputs.diagnostics ?? {};
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
      positions: props.positions,
      timestamps: props.timestamps,
      trackOffsets: props.trackOffsets,
      edgeStarts: props.edgeStarts,
      edgeEnds: props.edgeEnds,
      edgeZones: props.edgeZones,
      zoneCount,
      candidateCapacity: props.candidateCapacity,
      maxEventsPerTrack: props.maxEventsPerTrack,
      events: {
        output: {ids: events.ids, count: events.count, overflow: events.overflow},
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
      diagnostics
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
    contributors: builder.contributors,
    events,
    visitTable: scratch.visitTable,
    dwellTimes,
    visitCounts,
    trackEventCounts,
    diagnostics,
    table,
    outputs: {events, dwellTimes, visitCounts, trackEventCounts, table},
    intermediates: {visitTable: scratch.visitTable, diagnostics, cellZones, cellMask},
    status: {
      stages: [
        {
          stage: 'zone-events',
          status: {
            count: events.count,
            overflow: events.overflow,
            candidateOverflow: diagnostics.candidateOverflow,
            requiredCount: diagnostics.candidateCount
          }
        },
        {stage: 'zone-statistics', status: {count: table.count, overflow: table.overflow}}
      ]
    }
  };
}
