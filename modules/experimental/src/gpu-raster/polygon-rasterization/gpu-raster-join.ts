// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
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
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {validateRasterGrid, validateRasterParameters} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';
import {
  GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
  GPU_POLYGON_RASTERIZATION_NO_ZONE,
  POLYGON_RASTER_EXTENT_WGSL
} from './polygon-rasterization-parameters';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPURasterJoin';

/**
 * Largest `zoneCount` of {@link GPURasterJoin} without `output.sums` on `device`: each per-zone
 * column holds 4 bytes per zone in one storage binding, so 33554432 on a default device (128 MiB
 * bindings).
 */
export function getGPURasterJoinMaximumZoneCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest `zoneCount` of {@link GPURasterJoin} with `output.sums` on `device`: the sorted reduction
 * sums each zone with one 256-invocation workgroup, and the total invocation count must fit a
 * uint32 (`floor((2^32 - 1) / 256)`); its `zoneCount + 1` uint32 segment offsets must also fit one
 * storage binding. 16777215 on a default device (128 MiB bindings).
 */
export function getGPURasterJoinMaximumSummedZoneCount(device: Device): number {
  return Math.min(Math.floor(0xffffffff / 256), getStorageBindingElementLimit(device, 4, 1));
}

/** Workgroup size of the aggregate kernel, which reduces its tallies per workgroup. */
const AGGREGATE_WORKGROUP_SIZE = 256;

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
 * Per-frame (no recompile): the contents of `parameters`, `points`, `values`, `zones`, and
 * `boundary`. Topology: the `grid` size, `zoneCount`, view lengths, and which views exist.
 */
export type GPURasterJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-join'`. */
  id?: string;
  /** Planar by default; longitude/latitude accepts metric 'none' and linear edges. */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `zones` when declared separately from `spatialContext`. Must be compatible in
   * datum, coordinate space, metric, units and planar frame; longitude/latitude needs linear edges.
   * Declares the context zones were rasterized in, e.g. `GPUPolygonRasterization.spatialContext`.
   * Omitted means `spatialContext` describes both inputs.
   */
  zonesSpatialContext?: GPUSpatialContext;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Per-frame raster placement, the same view the zone raster was built with (see
   * `getGPUPolygonRasterizationParameterValues`).
   */
  parameters: GraphDataView<'float32'>;
  /** Packed planar points in the coordinate system of the raster. */
  points: GraphDataView<'float32x2'>;
  /** Optional per-point values for `sums`. */
  values?: GraphDataView<'float32'>;
  /** Packed row-major zone raster (`grid.width * grid.height`), for example from `GPUPolygonRasterization`. */
  zones: GraphDataView<'uint32'>;
  /** Optional boundary flags (`grid.width * grid.height`), for example from `GPUPolygonRasterization`. */
  boundary?: GraphDataView<'uint32'>;
  /**
   * Number of dense zones. Zone IDs at or above it join no zone. Compile-time. At most
   * {@link getGPURasterJoinMaximumZoneCount} (33554432 on a default device), or
   * {@link getGPURasterJoinMaximumSummedZoneCount} (16777215) with `output.sums`.
   */
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
 * Longitude/latitude contexts declare linear edges and metric 'none'. Use the same context for
 * the upstream `GPUPolygonRasterization`; this contributor consumes its zone raster. Points must
 * already use the grid's longitude branch. Prepare crossing or pole polygons with
 * `GPULonLatPolygonPreparation` before rasterization.
 *
 * Cost per encoding: two passes over the points plus, for `sums`, the radix sort, scan, gather,
 * and one 256-thread workgroup per zone of the shared sorted reduction.
 */
export class GPURasterJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated coordinate and edge model declaration. */
  readonly spatialContext: GPUSpatialContext;
  /** Validated properties. */
  readonly props: GPURasterJoinProps;

  constructor(props: GPURasterJoinProps) {
    this.id = props.id ?? 'raster-join';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      coordinateSpaces: ['planar', 'longitude-latitude'],
      metrics: ['none', 'native'],
      edgeModels: GPU_LINEAR_EDGE_MODELS
    });
    resolveGPUSecondarySpatialContext(
      this.id,
      'zones',
      props.zonesSpatialContext,
      this.spatialContext,
      GPU_LINEAR_EDGE_MODELS
    );
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    if (
      !Number.isSafeInteger(props.zoneCount) ||
      props.zoneCount < 1 ||
      props.zoneCount >= GPU_POLYGON_RASTERIZATION_NO_ZONE
    ) {
      throw new Error(`${id} zoneCount must be a positive integer below the no-zone sentinel`);
    }
    validateRasterParameters(id, props.parameters, GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH);
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
    for (const [name, view] of [
      ['zones', props.zones],
      ['boundary', props.boundary]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} must contain one row per cell`);
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
      validatePackedView(view, ['uint32'], `${id} output.${name}`);
      if (view.length !== length) {
        throw new Error(`${id} output.${name} must contain ${length} uint32 rows`);
      }
    }
    if (output.sums) {
      validatePackedView(output.sums, ['float32'], `${id} output.sums`);
      if (output.sums.length !== props.zoneCount) {
        throw new Error(`${id} output.sums must contain zoneCount rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.parameters,
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
    const {zoneCount, output, boundary} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.points,
      props.values,
      props.zones,
      boundary,
      ...Object.values(output)
    ]);
    // The sorted sums add one 256-invocation workgroup per zone, which lowers the maximum.
    validateCapacity(
      id,
      'zoneCount',
      zoneCount,
      1,
      output.sums
        ? getGPURasterJoinMaximumSummedZoneCount(graph.device)
        : getGPURasterJoinMaximumZoneCount(graph.device)
    );
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
      {name: 'extent', view: props.parameters, type: 'f32', access: 'read'},
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
      let cellIndex = getStoredRow(u32(cell.y), rasterExtent) * WIDTH + u32(cell.x);
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
          // The two tallies are reduced per workgroup first (library rule 14): one global atomic
          // per workgroup and tally instead of one per point on the same two words.
          workgroupSize: AGGREGATE_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
${tallies ? 'var<workgroup> workgroupTallies: array<atomic<u32>, 2>;' : ''}`,
          body: `${
            tallies
              ? `if (localInvocationIndex < 2u) { atomicStore(&workgroupTallies[localInvocationIndex], 0u); }
  workgroupBarrier();`
              : ''
          }
  if (index < INVOCATION_COUNT) {
    let zone = pointZones[pointZonesOffset + index];
    let status = pointStatus[pointStatusOffset + index];
    if (zone != NO_ZONE) {
      ${counts ? 'atomicAdd(&counts[countsOffset + zone], 1u);' : ''}
      ${output.boundaryCounts ? 'if (status == STATUS_BOUNDARY) {\n        atomicAdd(&boundaryCounts[boundaryCountsOffset + zone], 1u);\n      }' : ''}
    }
    ${
      tallies
        ? `if (zone == NO_ZONE && status == STATUS_BOUNDARY) {
      atomicAdd(&workgroupTallies[0], 1u);
    }
    if (status == STATUS_OUTSIDE) {
      atomicAdd(&workgroupTallies[1], 1u);
    }`
        : ''
    }
  }
  ${
    tallies
      ? `workgroupBarrier();
  if (localInvocationIndex < 2u) {
    let tally = atomicLoad(&workgroupTallies[localInvocationIndex]);
    if (tally > 0u) { atomicAdd(&tallies[talliesOffset + localInvocationIndex], tally); }
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
${IS_FINITE_VALUE_WGSL}`,
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
