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
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
  GPU_POLYGON_RASTERIZATION_NO_ZONE,
  POLYGON_RASTER_EXTENT_WGSL
} from './polygon-rasterization-parameters';

const OPERATION = 'GPURasterJoin';

/**
 * Caller-owned outputs of {@link GPURasterJoin}, rewritten on every encoding. At least one is
 * required. Per-zone columns hold exactly `zoneCount` rows; per-point columns one row per point;
 * scalars one row.
 */
export type GPURasterJoinOutput = {
  /** Per zone: points whose cell holds the zone. Exact integer atomics. */
  counts?: GraphDataView<'uint32'>;
  /**
   * Per zone: sum of finite `values` of the zone's points (non-finite values contribute 0).
   * Requires `values`. Reduced in sorted order, so bitwise reproducible across encodings.
   */
  sums?: GraphDataView<'float32'>;
  /**
   * Per zone: points counted in `counts[zone]` that fall in a boundary cell. Requires `boundary`.
   * These are the only points whose zone may differ from an exact point-in-polygon join.
   */
  boundaryCounts?: GraphDataView<'uint32'>;
  /**
   * One row: boundary-cell points that joined no zone. Requires `boundary`. Together with
   * `boundaryCounts` this bounds the error: the exact count of zone `k` among points inside the
   * raster lies in
   * `[counts[k] - boundaryCounts[k], counts[k] - boundaryCounts[k] + allBoundaryPoints]` where
   * `allBoundaryPoints = sum(boundaryCounts) + unassignedBoundaryCount`.
   */
  unassignedBoundaryCount?: GraphDataView<'uint32'>;
  /** One row: points outside the raster or with a non-finite position. They join no zone. */
  outsideCount?: GraphDataView<'uint32'>;
  /** Per point: joined zone, or `GPU_POLYGON_RASTERIZATION_NO_ZONE`. */
  pointZones?: GraphDataView<'uint32'>;
  /**
   * Per point: 1 when the point's cell is a boundary cell, otherwise 0. Requires `boundary`.
   * Feed it as a mask to an exact join (for example `GPUPointInPolygonJoin`) for a hybrid join.
   */
  pointBoundaryMask?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterJoin}.
 *
 * Per-frame (no recompile): the contents of `extent`, `points`, `values`, `zones`, and
 * `boundary`. Topology: `width`, `height`, `zoneCount`, view lengths, and which views exist.
 */
export type GPURasterJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-join'`. */
  id?: string;
  /** Raster width in cells. Compile-time. */
  width: number;
  /** Raster height in cells. Compile-time. */
  height: number;
  /**
   * Per-frame raster placement `[originX, originY, cellWidth, cellHeight]`, the same view the
   * zone raster was built with (see `getGPUPolygonRasterizationExtentValues`).
   */
  extent: GraphDataView<'float32'>;
  /** Packed planar points in the coordinate system of the extent. */
  points: GraphDataView<'float32x2'>;
  /** Optional per-point values for `sums`. */
  values?: GraphDataView<'float32'>;
  /** Packed row-major zone raster (`width * height`), for example from `GPUPolygonRasterization`. */
  zones: GraphDataView<'uint32'>;
  /** Optional boundary flags (`width * height`), for example from `GPUPolygonRasterization`. */
  boundary?: GraphDataView<'uint32'>;
  /** Number of dense zones. Zone IDs at or above it join no zone. Compile-time. */
  zoneCount: number;
  /** Result views. */
  output: GPURasterJoinOutput;
};

/**
 * Approximate polygon aggregation of points (a raster join).
 *
 * Each point is binned to its raster cell (`floor((p - origin) / cellSize)`, half-open cells) and
 * takes the zone stored in that cell, so the per-point cost is O(1) and independent of polygon
 * complexity. Counts accumulate with integer atomics; sums go through the shared sorted segmented
 * reduction, so every output is deterministic.
 *
 * Accuracy: with a zone raster from `GPUPolygonRasterization`, a point in a cell without the
 * boundary flag joins exactly the feature an exact point-in-polygon join (smallest containing
 * feature row) would give. Only points in boundary cells can differ; they are reported per zone in
 * `boundaryCounts`, in `unassignedBoundaryCount`, and per point in `pointBoundaryMask`, so callers
 * can bound the error or re-join those points exactly.
 *
 * Cost per encoding: two passes over the points plus, for `sums`, the radix sort, scan, gather,
 * and one 256-thread workgroup per zone of the shared sorted reduction.
 */
export class GPURasterJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterJoinProps;

  constructor(props: GPURasterJoinProps) {
    this.id = props.id ?? 'raster-join';
    this.props = props;
    const {id} = this;
    const {width, height, output} = props;
    for (const [name, value] of [
      ['width', width],
      ['height', height]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    if (width * height > 0xffffffff) {
      throw new Error(`${id} width * height must fit in uint32`);
    }
    if (
      !Number.isSafeInteger(props.zoneCount) ||
      props.zoneCount < 1 ||
      props.zoneCount >= GPU_POLYGON_RASTERIZATION_NO_ZONE
    ) {
      throw new Error(`${id} zoneCount must be a positive integer below the no-zone sentinel`);
    }
    validatePackedView(props.extent, ['float32'], `${id} extent`);
    if (props.extent.length < GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH) {
      throw new Error(
        `${id} extent must contain ${GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH} float32 rows`
      );
    }
    validatePackedView(props.points, ['float32x2'], `${id} points`);
    const pointCount = props.points.length;
    if (pointCount < 1) {
      throw new Error(`${id} points must not be empty`);
    }
    if (props.values) {
      validatePackedView(props.values, ['float32'], `${id} values`);
      if (props.values.length !== pointCount) {
        throw new Error(`${id} values must contain one row per point`);
      }
    }
    const cellCount = width * height;
    for (const [name, view] of [
      ['zones', props.zones],
      ['boundary', props.boundary]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} must contain width * height rows`);
        }
      }
    }
    const outputViews = Object.values(output).filter(Boolean);
    if (outputViews.length === 0) {
      throw new Error(`${id} requires at least one output`);
    }
    if (output.sums && !props.values) {
      throw new Error(`${id} sums require values`);
    }
    if (
      (output.boundaryCounts || output.unassignedBoundaryCount || output.pointBoundaryMask) &&
      !props.boundary
    ) {
      throw new Error(`${id} boundary outputs require boundary`);
    }
    for (const [name, view, length] of [
      ['counts', output.counts, props.zoneCount],
      ['boundaryCounts', output.boundaryCounts, props.zoneCount],
      ['unassignedBoundaryCount', output.unassignedBoundaryCount, 1],
      ['outsideCount', output.outsideCount, 1],
      ['pointZones', output.pointZones, pointCount],
      ['pointBoundaryMask', output.pointBoundaryMask, pointCount]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain ${length} uint32 rows`);
      }
    }
    if (output.sums) {
      validatePackedView(output.sums, ['float32'], `${id} sums`);
      if (output.sums.length !== props.zoneCount) {
        throw new Error(`${id} sums must contain zoneCount rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.extent,
      props.points,
      props.values,
      props.zones,
      props.boundary
    ]);
  }

  /** Returns lookup, aggregation, and optional sorted-sum nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, zoneCount, output, boundary} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.extent,
      props.points,
      props.values,
      props.zones,
      boundary,
      ...Object.values(output)
    ]);
    const pointCount = props.points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const declarations = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const ZONE_COUNT: u32 = ${zoneCount}u;
const NO_ZONE: u32 = ${GPU_POLYGON_RASTERIZATION_NO_ZONE}u;
// pointStatus values: inside a non-boundary cell, inside a boundary cell, outside the raster.
const STATUS_INTERIOR: u32 = 0u;
const STATUS_BOUNDARY: u32 = 1u;
const STATUS_OUTSIDE: u32 = 2u;`;

    // 1. Lookup: per point zone and status.
    const pointZones =
      output.pointZones ?? createTransientView(graph, `${id}-point-zones`, 'uint32', pointCount);
    const pointStatus = createTransientView(graph, `${id}-point-status`, 'uint32', pointCount);
    const lookupBindings: WGSLKernelBinding[] = [
      {name: 'points', view: props.points, type: 'f32', access: 'read'},
      {name: 'extent', view: props.extent, type: 'f32', access: 'read'},
      {name: 'zones', view: props.zones, type: 'u32', access: 'read'},
      {name: 'pointZones', view: pointZones, type: 'u32', access: 'read_write'},
      {name: 'pointStatus', view: pointStatus, type: 'u32', access: 'read_write'}
    ];
    if (boundary) {
      lookupBindings.push({name: 'boundary', view: boundary, type: 'u32', access: 'read'});
    }
    if (output.pointBoundaryMask) {
      lookupBindings.push({
        name: 'pointBoundaryMask',
        view: output.pointBoundaryMask,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-lookup`,
        operation: OPERATION,
        variant: 'lookup',
        bindings: lookupBindings,
        invocationCount: pointCount,
        declarations: `${declarations}
${POLYGON_RASTER_EXTENT_WGSL}`,
        body: `let rasterExtent = getRasterExtent();
  let position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var zone = NO_ZONE;
  var status = STATUS_OUTSIDE;
  if (rasterExtent.valid && isFiniteValue(position.x) && isFiniteValue(position.y)) {
    let cell = floor((position - rasterExtent.origin) / rasterExtent.cellSize);
    if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < f32(WIDTH) && cell.y < f32(HEIGHT)) {
      let cellIndex = u32(cell.y) * WIDTH + u32(cell.x);
      zone = zones[zonesOffset + cellIndex];
      if (zone >= ZONE_COUNT) {
        zone = NO_ZONE;
      }
      status = STATUS_INTERIOR;
      ${boundary ? 'if (boundary[boundaryOffset + cellIndex] != 0u) {\n        status = STATUS_BOUNDARY;\n      }' : ''}
    }
  }
  pointZones[pointZonesOffset + index] = zone;
  pointStatus[pointStatusOffset + index] = status;
  ${output.pointBoundaryMask ? 'pointBoundaryMask[pointBoundaryMaskOffset + index] = select(0u, 1u, status == STATUS_BOUNDARY);' : ''}`
      })
    );

    // 2. Aggregation with integer atomics.
    // The sorted sums below find zone sizes by binary search, so counts are built on request only.
    const counts = output.counts;
    const tallies =
      output.unassignedBoundaryCount || output.outsideCount
        ? createTransientView(graph, `${id}-tallies`, 'uint32', 2)
        : undefined;
    const aggregateBindings: WGSLKernelBinding[] = [
      {name: 'pointZones', view: pointZones, type: 'u32', access: 'read'},
      {name: 'pointStatus', view: pointStatus, type: 'u32', access: 'read'}
    ];
    for (const [name, view] of [
      ['counts', counts],
      ['boundaryCounts', output.boundaryCounts],
      ['tallies', tallies]
    ] as const) {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-reset`,
            operation: OPERATION,
            view,
            type: 'u32',
            value: '0u'
          })
        );
        aggregateBindings.push({name, view, type: 'atomic<u32>', access: 'read_write'});
      }
    }
    if (aggregateBindings.length > 2) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-aggregate`,
          operation: OPERATION,
          variant: 'aggregate',
          bindings: aggregateBindings,
          invocationCount: pointCount,
          declarations,
          body: `let zone = pointZones[pointZonesOffset + index];
  let status = pointStatus[pointStatusOffset + index];
  if (zone != NO_ZONE) {
    ${counts ? 'atomicAdd(&counts[countsOffset + zone], 1u);' : ''}
    ${output.boundaryCounts ? 'if (status == STATUS_BOUNDARY) {\n      atomicAdd(&boundaryCounts[boundaryCountsOffset + zone], 1u);\n    }' : ''}
  }
  ${
    tallies
      ? `if (zone == NO_ZONE && status == STATUS_BOUNDARY) {
    atomicAdd(&tallies[talliesOffset], 1u);
  }
  if (status == STATUS_OUTSIDE) {
    atomicAdd(&tallies[talliesOffset + 1u], 1u);
  }`
      : ''
  }`
        })
      );
    }
    if (tallies) {
      const publishBindings: WGSLKernelBinding[] = [
        {name: 'tallies', view: tallies, type: 'u32', access: 'read'}
      ];
      if (output.unassignedBoundaryCount) {
        publishBindings.push({
          name: 'unassignedOut',
          view: output.unassignedBoundaryCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (output.outsideCount) {
        publishBindings.push({
          name: 'outsideOut',
          view: output.outsideCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-tallies-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: publishBindings,
          invocationCount: 1,
          body: `${output.unassignedBoundaryCount ? 'unassignedOut[unassignedOutOffset] = tallies[talliesOffset];' : ''}
  ${output.outsideCount ? 'outsideOut[outsideOutOffset] = tallies[talliesOffset + 1u];' : ''}`
        })
      );
    }

    // 3. Deterministic sums: stable sort by zone, then a fixed-shape reduction per zone.
    if (output.sums) {
      const contributions = createTransientView(
        graph,
        `${id}-contributions`,
        'float32',
        pointCount
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-contributions`,
          operation: OPERATION,
          variant: 'contributions',
          bindings: [
            {name: 'pointZones', view: pointZones, type: 'u32', access: 'read'},
            {name: 'values', view: props.values!, type: 'f32', access: 'read'},
            {name: 'contributions', view: contributions, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pointCount,
          declarations: `const NO_ZONE: u32 = ${GPU_POLYGON_RASTERIZATION_NO_ZONE}u;
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }`,
          body: `let value = values[valuesOffset + index];
  let isContributing = pointZones[pointZonesOffset + index] != NO_ZONE && isFiniteValue(value);
  contributions[contributionsOffset + index] = select(0.0, value, isContributing);`
        })
      );
      nodes.push(
        ...getSortedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-sorted`,
          operation: OPERATION,
          segmentCount: zoneCount,
          segmentKeys: pointZones,
          reductions: [{name: 'sums', contributions, output: output.sums}]
        })
      );
    }
    return nodes;
  }
}
