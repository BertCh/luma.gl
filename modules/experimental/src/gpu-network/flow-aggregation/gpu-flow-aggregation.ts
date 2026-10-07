// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GPUHashIndex,
  GPUHashIndexQuery,
  GPUReduction,
  GPUScan,
  GPUSort,
  GPU_HASH_INDEX_STATISTICS_LENGTH,
  GPU_HASH_QUERY_STATISTICS_LENGTH,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput, GPUFloat32Positions} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {GPU_TIME_WINDOW_PARAMETER_LENGTH} from '../../gpu-dataframe/time-window-filter/time-window-parameters';
import {GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH} from '../../gpu-dataframe/time-window-filter/time-words';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {getTimeWindowClassifyNodes} from '../../gpu-dataframe/time-window-filter/time-window-classify-node';
import {
  createFlowContributionNodes,
  createFlowDecodeZonesNode,
  createFlowExtentToFloatNode,
  createFlowGatherIdsNode,
  createFlowGatherValuesNode,
  createFlowPairKeysNode,
  createFlowPairOverflowNode,
  createFlowPairSortPrepareNode,
  createFlowPairSumNode,
  createFlowSlotKeysNode,
  createFlowSortedGatherNode,
  createFlowWeightKeysNode,
  getFlowWeightKeyBits,
  createFlowZoneNode,
  type FlowResolvedBounds
} from './flow-aggregation-kernels';
import {GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT} from './flow-aggregation-pairs';

/**
 * Inclusive `[minX, minY, maxX, maxY]` domain of a grid or hexagon zone lattice.
 *
 * A literal is compile-time topology. A GPU view is per-frame: one `float32x4` row or four packed
 * `float32` rows, the shape of a `GPUParameterBuffer` with `format: 'float32', length: 4`.
 */
export type GPUFlowAggregationBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>
  | GraphDataView<'float32'>;

/**
 * Per-frame lattice size of grid and hexagon zones: a packed `uint32` view of at least two rows
 * `[columns, rows]`, for example a `GPUParameterBuffer` with `format: 'uint32'`.
 *
 * Each value is clamped to `1..gridSize` of the compile-time capacity. Zone IDs are row-major in
 * the active size (`row * columns + column`), cells outside it are rejected, and zone outputs keep
 * their capacity length: rows at and after `columns * rows` stay zero. Pair keys are built with the
 * capacity zone count, so ranking and ties are unaffected by the active size beyond the zone IDs.
 * Pair this with a per-frame hexagon `radius` so the lattice always covers the bounds.
 */
export type GPUFlowAggregationActiveGridSize = GraphDataView<'uint32'>;

/**
 * How rows are assigned origin and destination zones.
 *
 * `kind`, `gridSize`, and `zoneCount` are compile-time. Literal bounds and a literal radius are
 * compile-time; GPU bounds and a GPU radius are per-frame. `gridSize` is the capacity of the
 * lattice: with `activeGridSize` the lattice size is per-frame and `gridSize` is only its upper
 * bound.
 */
export type GPUFlowAggregationZones =
  | {
      /** Row-major rectangular cells, same cell formula as `GPUGridBinning`. */
      kind: 'grid';
      /** Domain covered by the cells. */
      bounds: GPUFlowAggregationBounds;
      /** `[columns, rows]` positive integers. Compile-time capacity. */
      gridSize: readonly [number, number];
      /** Optional per-frame `[columns, rows]` of the lattice in use. See {@link GPUFlowAggregationActiveGridSize}. */
      activeGridSize?: GraphDataView<'uint32'>;
    }
  | {
      /** Pointy-top odd-r hexagons, same lattice as `GPUPointDensity`. */
      kind: 'hexagon';
      /** Domain covered by the lattice. Points outside it are rejected. */
      bounds: GPUFlowAggregationBounds;
      /** `[columns, rows]` positive integers. Compile-time capacity. */
      gridSize: readonly [number, number];
      /** Optional per-frame `[columns, rows]` of the lattice in use. See {@link GPUFlowAggregationActiveGridSize}. */
      activeGridSize?: GraphDataView<'uint32'>;
      /** Center-to-vertex radius. A number is compile-time; a one-row float32 view is per-frame. */
      radius: number | GraphDataView<'float32'>;
    }
  | {
      /** Caller-supplied zone IDs per row (`originZoneIds` and `destinationZoneIds`). */
      kind: 'ids';
      /** Number of zones. IDs at or above it are rejected. Compile-time. */
      zoneCount: number;
    };

/**
 * Optional per-frame instant time gate (a row is accepted when its timestamp is inside the window).
 *
 * Two modes, chosen by the `timestamps` format:
 * - Float: packed `float32` timestamps, optional `timestampsLow`, and a `float32` window written with
 *   `getGPUTimeWindowParameterValues`.
 * - Int64 words: packed `uint32x2` `(low, high)` timestamps (for example epoch milliseconds from an
 *   Arrow `Int64` column) and a packed `uint32` window written with
 *   `getGPUTimeWindowWordParameterValues`. Acceptance is exact.
 */
export type GPUFlowAggregationTimeWindow = {
  /** Packed `float32` timestamp per row, or packed `uint32x2` Int64 words per row. Per-frame contents. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Optional double-single low parts from `splitTimestamps`. Float timestamps only. Per-frame contents. */
  timestampsLow?: GraphDataView<'float32'>;
  /**
   * Float timestamps: packed window of at least 8 `float32` rows from `getGPUTimeWindowParameterValues`.
   * Word timestamps: packed window of at least 8 `uint32` rows from `getGPUTimeWindowWordParameterValues`.
   * Per-frame.
   */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
};

/**
 * Order in which float weight sums are accumulated.
 *
 * - `'sorted'` (default): rows are stably sorted by key and reduced per key in a fixed tree, so sums
 *   are bitwise identical across runs and devices. Costs extra sort, scan, and gather passes; about
 *   13-24 ms for 1M rows whether flows are concentrated or spread over 4096 pairs.
 * - `'atomic'`: compare-exchange float addition, rounding order depends on GPU scheduling. Skips the
 *   sorts, and was about 4x faster (21 ms against 85 ms for 1M rows) only when almost every row has
 *   its own pair (about 600k distinct pairs). Lanes that share a pair or zone retry on one address
 *   and serialize: 1M rows over 4096 pairs took 70 ms against 13 ms, and 1M rows with 90% on one
 *   pair took 12-17 seconds against 22 ms.
 */
export type GPUFlowAggregationSumOrder = 'atomic' | 'sorted';

/**
 * Properties for {@link GPUFlowAggregation}.
 *
 * Compile-time: zone kind, grid size, zone count, `pairCapacity`, `maxProbeCount`,
 * `excludeSelfFlows`, literal bounds and radius, all lengths, and which optional views exist.
 * Per-frame: the contents of every input view (positions, zone IDs, weights, mask, timestamps, time
 * window, GPU bounds and radius).
 */
export type GPUFlowAggregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'flow-aggregation'`. */
  id?: string;
  /** Zone assignment. Compile-time. */
  zones: GPUFlowAggregationZones;
  /** Origin positions for grid and hexagon zones (single view or chunked vector). Per-frame contents. */
  origins?: GPUFloat32Positions;
  /** Destination positions, same length as `origins`. Per-frame contents. */
  destinations?: GPUFloat32Positions;
  /** Packed origin zone IDs for `'ids'` zones. IDs `>= zoneCount` reject the row. Per-frame contents. */
  originZoneIds?: GraphDataView<'uint32'>;
  /** Packed destination zone IDs for `'ids'` zones, same length as `originZoneIds`. Per-frame contents. */
  destinationZoneIds?: GraphDataView<'uint32'>;
  /**
   * Float weight sum order for `flowWeights`, `zoneOutWeights`, and `zoneInWeights`. Compile-time.
   * Defaults to `'sorted'`.
   */
  sumOrder?: GPUFlowAggregationSumOrder;
  /** Optional per-row weight with one row per flow row. Non-finite weights are ignored in sums. Per-frame contents. */
  weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional packed per-row accept mask; nonzero accepts. Presence is compile-time, contents per-frame. */
  mask?: GraphDataView<'uint32'>;
  /** Optional instant time gate. Presence is compile-time, contents per-frame. */
  timeWindow?: GPUFlowAggregationTimeWindow;
  /** Reject rows whose origin and destination zones are equal. Compile-time. Defaults to `false`. */
  excludeSelfFlows?: boolean;
  /** Hash table slots, a power of two: the maximum number of distinct pairs retained. Compile-time. */
  pairCapacity: number;
  /** Forwarded to `GPUHashIndex` and its query. Defaults to `pairCapacity`. Compile-time. */
  maxProbeCount?: number;
  /** Top-K output. `ids` receives pair keys; `ids.length` is K. */
  output: GPUCompactOutput;
  /** Optional K rows of origin zone IDs aligned with `output.ids`. */
  flowOriginZoneIds?: GraphDataView<'uint32'>;
  /** Optional K rows of destination zone IDs aligned with `output.ids`. */
  flowDestinationZoneIds?: GraphDataView<'uint32'>;
  /** Optional K rows: accepted rows in each reported pair. */
  flowCounts?: GraphDataView<'uint32'>;
  /** Optional K rows: sum of finite weights per pair, or the count as float32 without `weights`. */
  flowWeights?: GraphDataView<'float32'>;
  /** Optional one row: 1 when an accepted row's pair could not be stored in the table. */
  pairOverflow?: GraphDataView<'uint32'>;
  /** Optional `zoneCount` rows: accepted rows per origin zone. */
  zoneOutCounts?: GraphDataView<'uint32'>;
  /** Optional `zoneCount` rows: accepted rows per destination zone. */
  zoneInCounts?: GraphDataView<'uint32'>;
  /** Optional `zoneCount` rows: sum of finite weights per origin zone. Requires `weights`. */
  zoneOutWeights?: GraphDataView<'float32'>;
  /** Optional `zoneCount` rows: sum of finite weights per destination zone. Requires `weights`. */
  zoneInWeights?: GraphDataView<'float32'>;
  /** Optional one row receiving the clamped flow count, for an indirect instanced draw. */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /**
   * Optional two `float32` rows `[minimum, maximum]` of the nonzero per-origin-zone counts (both 0
   * when no zone has a count). Computed on the GPU, so a color range needs no readback. Layers can
   * take it directly as a `[min, max]` extent buffer.
   */
  zoneOutCountExtent?: GraphDataView<'float32'>;
  /** Same as `zoneOutCountExtent` for per-destination-zone counts. */
  zoneInCountExtent?: GraphDataView<'float32'>;
};

/**
 * Aggregates origin-destination movements into a weight-sorted top-K flow list and per-zone totals.
 *
 * Each source row is one movement. Rows are assigned origin and destination zones (grid cells,
 * hexagons, or caller IDs), accepted rows are grouped by `originZone * zoneCount + destinationZone`
 * in a fixed-capacity `GPUHashIndex`, counts and weight sums are accumulated with
 * `GPUGroupAggregation` over each pair's lowest source row, and the table slots are ranked with two
 * stable `GPUSort` passes.
 *
 * Accepted row: both zones valid (below `zoneCount`; for grid and hexagon, a finite position inside
 * the inclusive bounds and inside the lattice), mask nonzero when given, time window accepts when
 * given, and not a self flow when `excludeSelfFlows`.
 *
 * Output: flows are sorted by weight descending, ties by pair key ascending (origin zone, then
 * destination zone); without `weights` the weight is the count. `output.totalCount` is the number
 * of distinct pairs retained in the table, `output.count = min(totalCount, K)`, and
 * `output.overflow` is 1 when `totalCount > K` or `pairOverflow` is set. `totalCount > K` alone means
 * the list is a top-K truncation but every aggregate is exact. `pairOverflow` means distinct pairs
 * exceeded `pairCapacity` (or the probe limit): aggregates are incomplete and which pairs were
 * retained is unspecified (the `GPUHashIndex` contract). Rows `k >= count` of every K-row output are
 * rewritten each encoding with sentinels (`0xffffffff` IDs, zero counts and weights). Zone totals
 * count every accepted row, including rows whose pair overflowed the table.
 *
 * Counts are exact. With `sumOrder: 'sorted'` (default) every float sum (`flowWeights`, `zoneOutWeights`, `zoneInWeights`) is accumulated in source row
 * order within each key by a fixed tree, so it is bitwise identical across runs and devices, at the
 * cost of a stable sort, a scan, and gather passes per sum (about 3 sorts for pair and zone sums).
 * With `sumOrder: 'atomic'` float sums use compare-exchange atomic addition, so their rounding order
 * is not deterministic (small integer weights stay exact), and rows that share a pair or zone
 * serialize on one address; use it only when nearly every row has a distinct pair.
 * Weight ranking uses the final float sum either way.
 *
 * The time gate accepts float timestamps (optionally double-single) or exact Int64 word timestamps
 * (`uint32x2`), see {@link GPUFlowAggregationTimeWindow}.
 *
 * Pair keys are one uint32, so `zoneCount * zoneCount` must stay below `0xffffffff`, which caps
 * `zoneCount` at 65535. `GPUHashIndex` also requires `rows * maxProbeCount` to fit in a uint32.
 *
 * Non-goals: geodesic or great-circle zones, more than 65535 zones (64-bit pair keys), flow
 * bundling or edge routing, chunked (vector) zone IDs, masks, or timestamps, and table resizing.
 */
export class GPUFlowAggregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUFlowAggregationProps;
  /** Number of zones. */
  readonly zoneCount: number;
  /** Number of source rows. */
  readonly rowCount: number;
  /** Float weight sum order. */
  readonly sumOrder: GPUFlowAggregationSumOrder;

  constructor(props: GPUFlowAggregationProps) {
    this.id = props.id ?? 'flow-aggregation';
    this.props = props;
    const id = this.id;
    const {zones, output, weights} = props;
    this.sumOrder = props.sumOrder ?? 'sorted';
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }

    if (zones.kind === 'ids') {
      this.zoneCount = zones.zoneCount;
    } else if (zones.kind === 'grid' || zones.kind === 'hexagon') {
      const [columns, rows] = zones.gridSize;
      if (
        !Number.isSafeInteger(columns) ||
        !Number.isSafeInteger(rows) ||
        columns < 1 ||
        rows < 1
      ) {
        throw new Error(`${id} gridSize must be two positive integers`);
      }
      this.zoneCount = columns * rows;
    } else {
      throw new Error(`${id} zones.kind must be grid, hexagon, or ids`);
    }
    if (
      !Number.isSafeInteger(this.zoneCount) ||
      this.zoneCount < 1 ||
      this.zoneCount > GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT
    ) {
      throw new Error(
        `${id} zone count must be an integer from 1 to ${GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT}`
      );
    }

    if (zones.kind === 'ids') {
      if (!props.originZoneIds || !props.destinationZoneIds) {
        throw new Error(`${id} ids zones require originZoneIds and destinationZoneIds`);
      }
      if (props.origins || props.destinations) {
        throw new Error(`${id} ids zones do not accept origins or destinations`);
      }
      validatePackedUint32View(props.originZoneIds, `${id} originZoneIds`);
      validatePackedUint32View(props.destinationZoneIds, `${id} destinationZoneIds`);
      this.rowCount = props.originZoneIds.length;
      if (props.destinationZoneIds.length !== this.rowCount) {
        throw new Error(`${id} destinationZoneIds length must equal originZoneIds length`);
      }
    } else {
      if (!props.origins || !props.destinations) {
        throw new Error(`${id} ${zones.kind} zones require origins and destinations`);
      }
      if (props.originZoneIds || props.destinationZoneIds) {
        throw new Error(`${id} ${zones.kind} zones do not accept zone IDs`);
      }
      for (const chunk of getGraphViewChunks(props.origins)) {
        validatePackedView(chunk, ['float32x2'], `${id} origins`);
      }
      for (const chunk of getGraphViewChunks(props.destinations)) {
        validatePackedView(chunk, ['float32x2'], `${id} destinations`);
      }
      this.rowCount = props.origins.length;
      if (props.destinations.length !== this.rowCount) {
        throw new Error(`${id} destinations length must equal origins length`);
      }
      validateFlowBounds(id, zones.bounds);
      if (zones.activeGridSize) {
        validatePackedUint32View(zones.activeGridSize, `${id} activeGridSize`);
        if (zones.activeGridSize.length < 2) {
          throw new Error(`${id} activeGridSize must contain two uint32 rows [columns, rows]`);
        }
      }
      if (zones.kind === 'hexagon') {
        const radius = zones.radius;
        if (typeof radius === 'number') {
          if (!Number.isFinite(radius) || radius <= 0) {
            throw new Error(`${id} hexagon radius must be positive and finite`);
          }
        } else {
          validatePackedView(radius, ['float32'], `${id} hexagon radius`);
          if (radius.length < 1) {
            throw new Error(`${id} hexagon radius must contain one float32 row`);
          }
        }
      }
    }

    if (weights) {
      for (const chunk of getGraphViewChunks(weights)) {
        validatePackedView(chunk, ['float32'], `${id} weights`);
      }
      if (weights.length !== this.rowCount) {
        throw new Error(`${id} weights length must equal the row count`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    if (props.timeWindow) {
      const {timestamps, timestampsLow, window} = props.timeWindow;
      validatePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
      if (timestamps.length !== this.rowCount) {
        throw new Error(`${id} timestamps length must equal the row count`);
      }
      const isWordTime = timestamps.format === 'uint32x2';
      if (timestampsLow) {
        if (isWordTime) {
          throw new Error(`${id} timestampsLow requires float32 timestamps`);
        }
        validatePackedView(timestampsLow, ['float32'], `${id} timestampsLow`);
        if (timestampsLow.length !== this.rowCount) {
          throw new Error(`${id} timestampsLow length must equal the row count`);
        }
      }
      if (isWordTime) {
        validatePackedView(window, ['uint32'], `${id} word window`);
        if (window.length < GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH) {
          throw new Error(
            `${id} word window must hold ${GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH} uint32 values`
          );
        }
      } else {
        validatePackedView(window, ['float32'], `${id} window`);
        if (window.length < GPU_TIME_WINDOW_PARAMETER_LENGTH) {
          throw new Error(
            `${id} window must hold ${GPU_TIME_WINDOW_PARAMETER_LENGTH} float32 values`
          );
        }
      }
    }

    const capacity = props.pairCapacity;
    if (!Number.isSafeInteger(capacity) || capacity < 1 || (capacity & (capacity - 1)) !== 0) {
      throw new Error(`${id} pairCapacity must be a positive power of two`);
    }
    if (capacity > 0x80000000) {
      throw new Error(`${id} pairCapacity must fit in uint32`);
    }
    const maxProbeCount = props.maxProbeCount ?? capacity;
    if (!Number.isSafeInteger(maxProbeCount) || maxProbeCount < 1 || maxProbeCount > capacity) {
      throw new Error(`${id} maxProbeCount must be an integer from one through pairCapacity`);
    }

    validateCompactOutput(id, output);
    const topCount = output.ids.length;
    if (topCount < 1) {
      throw new Error(`${id} output.ids must contain at least one row`);
    }
    for (const [name, view] of [
      ['flowOriginZoneIds', props.flowOriginZoneIds],
      ['flowDestinationZoneIds', props.flowDestinationZoneIds],
      ['flowCounts', props.flowCounts]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== topCount) {
          throw new Error(`${id} ${name} length must equal output.ids length`);
        }
      }
    }
    if (props.flowWeights) {
      validatePackedView(props.flowWeights, ['float32'], `${id} flowWeights`);
      if (props.flowWeights.length !== topCount) {
        throw new Error(`${id} flowWeights length must equal output.ids length`);
      }
    }
    for (const [name, view] of [
      ['pairOverflow', props.pairOverflow],
      ['drawInstanceCount', props.drawInstanceCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutCounts', props.zoneOutCounts],
      ['zoneInCounts', props.zoneInCounts]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== this.zoneCount) {
          throw new Error(`${id} ${name} length must equal the zone count`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutCountExtent', props.zoneOutCountExtent],
      ['zoneInCountExtent', props.zoneInCountExtent]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== 2) {
          throw new Error(`${id} ${name} must contain two float32 rows`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutWeights', props.zoneOutWeights],
      ['zoneInWeights', props.zoneInWeights],
      ['flowWeights', props.flowWeights]
    ] as const) {
      if (!view) {
        continue;
      }
      if (name !== 'flowWeights' && !weights) {
        throw new Error(`${id} ${name} requires weights`);
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (name !== 'flowWeights' && view.length !== this.zoneCount) {
        throw new Error(`${id} ${name} length must equal the zone count`);
      }
    }

    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  /**
   * Returns zone, time, pair-key, hash, aggregation, ranking, gather, publish, and zone-total
   * nodes in order. Declares work only.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zoneCount, rowCount} = this;
    const {zones, weights, output} = props;
    const capacity = props.pairCapacity;
    const maxProbeCount = props.maxProbeCount ?? capacity;
    const topCount = output.ids.length;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);

    const nodes: GPUCommandNode<Parameters>[] = [];
    let originZones: GraphDataView<'uint32'>;
    let destinationZones: GraphDataView<'uint32'>;
    if (zones.kind === 'ids') {
      originZones = props.originZoneIds as GraphDataView<'uint32'>;
      destinationZones = props.destinationZoneIds as GraphDataView<'uint32'>;
    } else {
      originZones = createTransientView(graph, `${id}-origin-zone-ids`, 'uint32', rowCount);
      destinationZones = createTransientView(
        graph,
        `${id}-destination-zone-ids`,
        'uint32',
        rowCount
      );
      const boundsView = Array.isArray(zones.bounds) ? undefined : (zones.bounds as GraphDataView);
      const bounds: FlowResolvedBounds =
        boundsView && boundsView.format === 'float32'
          ? graph.createDataView(boundsView.buffer, {
              format: 'float32x4',
              length: 1,
              byteOffset: boundsView.byteOffset
            })
          : (zones.bounds as FlowResolvedBounds);
      for (const [name, positions, target] of [
        ['origin', props.origins as GPUFloat32Positions, originZones],
        ['destination', props.destinations as GPUFloat32Positions, destinationZones]
      ] as const) {
        const chunks = getGraphViewChunks(positions);
        let zoneStart = 0;
        for (const [chunkIndex, chunk] of chunks.entries()) {
          if (chunk.length > 0) {
            nodes.push(
              createFlowZoneNode(graph, {
                id: `${id}-${name}-zones${chunks.length > 1 ? `-chunk-${chunkIndex}` : ''}`,
                kind: zones.kind,
                positions: chunk,
                zones: target,
                zoneStart,
                gridSize: zones.gridSize,
                bounds,
                radius: zones.kind === 'hexagon' ? zones.radius : undefined,
                activeGridSize: zones.activeGridSize
              })
            );
          }
          zoneStart += chunk.length;
        }
      }
    }

    let timeMask: GraphDataView<'uint32'> | undefined;
    if (props.timeWindow && rowCount > 0) {
      timeMask = createTransientView(graph, `${id}-time-mask`, 'uint32', rowCount);
      nodes.push(
        ...getTimeWindowClassifyNodes(graph, {
          id: `${id}-time-classify`,
          timestamps: props.timeWindow.timestamps as GraphDataView<'float32'>,
          timestampsLow: props.timeWindow.timestampsLow,
          window: props.timeWindow.window,
          mask: timeMask
        })
      );
    }

    const pairKeys = createTransientView(graph, `${id}-pair-keys`, 'uint32', rowCount);
    const acceptedOriginZones = createTransientView(
      graph,
      `${id}-accepted-origin-zones`,
      'uint32',
      rowCount
    );
    const acceptedDestinationZones = createTransientView(
      graph,
      `${id}-accepted-destination-zones`,
      'uint32',
      rowCount
    );
    if (rowCount > 0) {
      nodes.push(
        createFlowPairKeysNode(graph, {
          id: `${id}-pair-keys`,
          originZones,
          destinationZones,
          mask: props.mask,
          timeMask,
          pairKeys,
          acceptedOriginZones,
          acceptedDestinationZones,
          zoneCount,
          excludeSelfFlows: props.excludeSelfFlows ?? false
        })
      );
    }

    const tableKeys = createTransientView(graph, `${id}-table-keys`, 'uint32', capacity);
    const tableValues = createTransientView(graph, `${id}-table-values`, 'uint32', capacity);
    const buildStatistics = createTransientView(
      graph,
      `${id}-build-statistics`,
      'uint32',
      GPU_HASH_INDEX_STATISTICS_LENGTH
    );
    nodes.push(
      ...new GPUHashIndex({
        id: `${id}-pair-index`,
        keys: pairKeys,
        firstValue: 0,
        tableKeys,
        tableValues,
        statistics: buildStatistics,
        maxProbeCount
      }).getCommandNodes(graph)
    );

    const representativeRows = createTransientView(
      graph,
      `${id}-representative-rows`,
      'uint32',
      rowCount
    );
    if (rowCount > 0) {
      nodes.push(
        ...new GPUHashIndexQuery({
          id: `${id}-pair-query`,
          index: {
            tableKeys,
            tableValues,
            statistics: buildStatistics,
            maxProbeCount
          },
          keys: pairKeys,
          values: representativeRows,
          found: createTransientView(graph, `${id}-query-found`, 'uint32', rowCount),
          probes: createTransientView(graph, `${id}-query-probes`, 'uint32', rowCount),
          statistics: createTransientView(
            graph,
            `${id}-query-statistics`,
            'uint32',
            GPU_HASH_QUERY_STATISTICS_LENGTH
          ),
          maxProbeCount
        }).getCommandNodes(graph)
      );
    }
    const rowCounts = createTransientView(
      graph,
      `${id}-row-counts`,
      'uint32',
      Math.max(rowCount, 1)
    );
    nodes.push(
      ...new GPUGroupAggregation({
        id: `${id}-pair-counts`,
        keys: representativeRows,
        output: rowCounts
      }).getCommandNodes(graph)
    );
    const useSortedSums = this.sumOrder === 'sorted' && rowCount > 0;
    // Finite weights, or 0, one per source row; shared by every sorted sum.
    let contributions: GraphDataView<'float32'> | undefined;
    if (weights && useSortedSums) {
      contributions = createTransientView(graph, `${id}-weight-contributions`, 'float32', rowCount);
      nodes.push(...createFlowContributionNodes(graph, {id, weights, contributions}));
    }
    let rowWeights: GraphDataView<'float32'> | undefined;
    if (weights) {
      rowWeights = createTransientView(
        graph,
        `${id}-row-weights`,
        'float32',
        Math.max(rowCount, 1)
      );
    }
    if (weights && rowWeights && contributions) {
      // Stable sort of rows by representative row, offsets from the exact per-pair counts, gather
      // in sorted order, and one fixed-tree segment sum per table slot.
      const sortKeys = createTransientView(graph, `${id}-pair-sum-sort-keys`, 'uint32', rowCount);
      const sortIndices = createTransientView(
        graph,
        `${id}-pair-sum-sort-indices`,
        'uint32',
        rowCount
      );
      const sortedKeys = createTransientView(
        graph,
        `${id}-pair-sum-sorted-keys`,
        'uint32',
        rowCount
      );
      const sortedIndices = createTransientView(
        graph,
        `${id}-pair-sum-sorted-indices`,
        'uint32',
        rowCount
      );
      const rowOffsets = createTransientView(graph, `${id}-pair-sum-offsets`, 'uint32', rowCount);
      const sortedContributions = createTransientView(
        graph,
        `${id}-pair-sum-sorted-contributions`,
        'float32',
        rowCount
      );
      nodes.push(
        createFlowPairSortPrepareNode(graph, {
          id: `${id}-pair-sum-sort-prepare`,
          representativeRows,
          sortKeys,
          sortIndices
        }),
        ...new GPUSort({
          id: `${id}-pair-sum-sort`,
          keys: sortKeys,
          values: sortIndices,
          outputKeys: sortedKeys,
          outputValues: sortedIndices,
          keyBits: Math.max(1, rowCount.toString(2).length)
        }).getCommandNodes(graph),
        ...new GPUScan({
          id: `${id}-pair-sum-scan`,
          input: rowCounts,
          output: rowOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createFlowSortedGatherNode(graph, {
          id: `${id}-pair-sum-gather`,
          sortedIndices,
          contributions,
          sortedContributions
        }),
        createFlowPairSumNode(graph, {
          id: `${id}-pair-sum`,
          tableKeys,
          tableValues,
          rowCounts,
          rowOffsets,
          sortedContributions,
          rowWeights
        })
      );
    } else if (weights && rowWeights) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-pair-weights`,
          keys: representativeRows,
          values: weights,
          output: rowWeights,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    // Two stable ascending sorts form an LSD composite key: pair key first, then weight descending.
    const slotKeys = createTransientView(graph, `${id}-slot-keys`, 'uint32', capacity);
    const slotIndices = createTransientView(graph, `${id}-slot-indices`, 'uint32', capacity);
    const sortedPairKeys = createTransientView(graph, `${id}-sorted-pair-keys`, 'uint32', capacity);
    const slotsByPair = createTransientView(graph, `${id}-slots-by-pair`, 'uint32', capacity);
    const weightKeys = createTransientView(graph, `${id}-weight-keys`, 'uint32', capacity);
    const sortedWeightKeys = createTransientView(
      graph,
      `${id}-sorted-weight-keys`,
      'uint32',
      capacity
    );
    const rankedSlots = createTransientView(graph, `${id}-ranked-slots`, 'uint32', capacity);
    nodes.push(
      createFlowSlotKeysNode(graph, {
        id: `${id}-slot-keys`,
        tableKeys,
        slotKeys,
        slotIndices,
        zoneCount
      }),
      ...new GPUSort({
        id: `${id}-sort-pairs`,
        keys: slotKeys,
        values: slotIndices,
        outputKeys: sortedPairKeys,
        outputValues: slotsByPair,
        // Pair keys are below zoneCount^2, which empty slots take (see createFlowSlotKeysNode).
        keyBits: Math.max(1, (zoneCount * zoneCount).toString(2).length)
      }).getCommandNodes(graph),
      createFlowWeightKeysNode(graph, {
        id: `${id}-weight-keys`,
        slotsByPair,
        tableKeys,
        tableValues,
        rowCounts,
        rowWeights,
        weightKeys
      }),
      ...new GPUSort({
        id: `${id}-sort-weights`,
        keys: weightKeys,
        values: slotsByPair,
        outputKeys: sortedWeightKeys,
        outputValues: rankedSlots,
        keyBits: getFlowWeightKeyBits(Boolean(rowWeights), rowCounts.length)
      }).getCommandNodes(graph)
    );

    nodes.push(
      createFlowGatherIdsNode(graph, {
        id: `${id}-gather-ids`,
        statistics: buildStatistics,
        rankedSlots,
        tableKeys,
        ids: output.ids
      })
    );
    if (props.flowCounts || props.flowWeights) {
      nodes.push(
        createFlowGatherValuesNode(graph, {
          id: `${id}-gather-values`,
          capacity: topCount,
          statistics: buildStatistics,
          rankedSlots,
          tableValues,
          rowCounts,
          rowWeights,
          flowCounts: props.flowCounts,
          flowWeights: props.flowWeights
        })
      );
    }
    if (props.flowOriginZoneIds || props.flowDestinationZoneIds) {
      nodes.push(
        createFlowDecodeZonesNode(graph, {
          id: `${id}-decode-zones`,
          zoneCount,
          ids: output.ids,
          originZoneIds: props.flowOriginZoneIds,
          destinationZoneIds: props.flowDestinationZoneIds
        })
      );
    }

    const pairOverflow =
      props.pairOverflow ?? createTransientView(graph, `${id}-pair-overflow`, 'uint32', 1);
    nodes.push(
      createFlowPairOverflowNode(graph, {
        id: `${id}-pair-overflow`,
        statistics: buildStatistics,
        pairOverflow
      }),
      createPublishNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUFlowAggregation',
        totalCount: graph.createDataView(buildStatistics.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: buildStatistics.byteOffset
        }),
        output,
        overflowSources: [pairOverflow],
        extraCounts: props.drawInstanceCount ? [props.drawInstanceCount] : []
      })
    );

    for (const [name, keys, requestedCounts, sums, extent] of [
      [
        'out',
        acceptedOriginZones,
        props.zoneOutCounts,
        props.zoneOutWeights,
        props.zoneOutCountExtent
      ],
      [
        'in',
        acceptedDestinationZones,
        props.zoneInCounts,
        props.zoneInWeights,
        props.zoneInCountExtent
      ]
    ] as const) {
      const counts =
        requestedCounts ??
        (extent
          ? createTransientView(graph, `${id}-zone-${name}-extent-counts`, 'uint32', zoneCount)
          : undefined);
      if (counts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-zone-${name}-counts`,
            keys,
            output: counts
          }).getCommandNodes(graph)
        );
      }
      if (counts && extent) {
        // Extent over occupied zones only: any nonzero count selects a row.
        const countExtent = createTransientView(graph, `${id}-zone-${name}-extent`, 'uint32', 2);
        nodes.push(
          ...new GPUReduction({
            id: `${id}-zone-${name}-extent`,
            input: counts,
            mask: counts,
            output: countExtent,
            operation: 'extent'
          }).getCommandNodes(graph),
          createFlowExtentToFloatNode(graph, {
            id: `${id}-zone-${name}-extent-float`,
            extent: countExtent,
            output: extent
          })
        );
      }
      if (sums && weights && contributions) {
        nodes.push(
          ...getSortedSegmentSumNodes<Parameters>(graph, {
            id: `${id}-zone-${name}-sorted`,
            operation: 'GPUFlowAggregation',
            segmentCount: zoneCount,
            segmentKeys: keys,
            sumContributions: contributions,
            sums
          })
        );
      } else if (sums && weights) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-zone-${name}-weights`,
            keys,
            values: weights,
            output: sums,
            operation: 'sum'
          }).getCommandNodes(graph)
        );
      }
    }
    return nodes;
  }

  private getInputViews(): (GraphDataView | GraphVectorView | undefined)[] {
    const {props} = this;
    const {zones} = props;
    return [
      props.origins,
      props.destinations,
      props.originZoneIds,
      props.destinationZoneIds,
      props.weights,
      props.mask,
      props.timeWindow?.timestamps,
      props.timeWindow?.timestampsLow,
      props.timeWindow?.window,
      zones.kind !== 'ids' && !Array.isArray(zones.bounds)
        ? (zones.bounds as GraphDataView)
        : undefined,
      zones.kind === 'hexagon' && typeof zones.radius !== 'number' ? zones.radius : undefined,
      zones.kind !== 'ids' ? zones.activeGridSize : undefined
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    const {output} = props;
    return [
      output.ids,
      output.count,
      output.overflow,
      output.totalCount,
      props.flowOriginZoneIds,
      props.flowDestinationZoneIds,
      props.flowCounts,
      props.flowWeights,
      props.pairOverflow,
      props.zoneOutCounts,
      props.zoneInCounts,
      props.zoneOutWeights,
      props.zoneInWeights,
      props.drawInstanceCount,
      props.zoneOutCountExtent,
      props.zoneInCountExtent
    ];
  }
}

/** Validates literal or GPU-resident zone bounds. */
function validateFlowBounds(id: string, bounds: GPUFlowAggregationBounds): void {
  if (Array.isArray(bounds)) {
    const [minX, minY, maxX, maxY] = bounds as readonly number[];
    if (bounds.length !== 4 || !bounds.every(Number.isFinite) || minX > maxX || minY > maxY) {
      throw new Error(`${id} bounds must be finite [minX, minY, maxX, maxY]`);
    }
    return;
  }
  const view = bounds as GraphDataView;
  validatePackedView(view, ['float32x4', 'float32'], `${id} bounds`);
  if (
    (view.format === 'float32x4' && view.length !== 1) ||
    (view.format === 'float32' && view.length !== 4)
  ) {
    throw new Error(`${id} bounds view must be one float32x4 row or four float32 rows`);
  }
}
