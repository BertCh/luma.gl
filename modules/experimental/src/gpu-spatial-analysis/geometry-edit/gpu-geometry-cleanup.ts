// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  createCleanupEmitNode,
  createCleanupFlagNode,
  createCleanupPointsNode,
  createCleanupPublishNode
} from './geometry-edit-kernels';

const OPERATION = 'GPUGeometryCleanup';

/** Number of float32 elements in a `GPUGeometryCleanup` parameter buffer. */
export const GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPUGeometryCleanup`. */
export type GPUGeometryCleanupParameters = {
  /**
   * Grid size to snap every vertex to, `round(x / gridSize) * gridSize` with ties rounding up as
   * in GEOS (`shapely.set_precision(mode='pointwise')`). `0` (default) disables snapping.
   */
  gridSize?: number;
  /**
   * Distance within which a vertex counts as a repeat of the previously kept vertex
   * (`shapely.remove_repeated_points(tolerance)`). `0` (default) removes exact repeats only.
   * Ignored for point input because every row is an independent feature.
   */
  tolerance?: number;
  /**
   * Whether to drop repeated vertices after snapping. Default true. `false` with a `gridSize`
   * reproduces `set_precision(mode='pointwise')`, which snaps but keeps repeated vertices.
   * Ignored for point input.
   */
  removeRepeatedPoints?: boolean;
};

/**
 * Packs `GPUGeometryCleanup` parameters into the 4-element float32 layout
 * `[gridSize, tolerance, removeRepeatedPoints, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `gridSize` or `tolerance` is negative or NaN, or `target` is too short.
 */
export function getGPUGeometryCleanupParameterValues(
  parameters: GPUGeometryCleanupParameters = {},
  target: Float32Array = new Float32Array(GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH) {
    throw new Error(
      `Geometry cleanup target must hold ${GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH} elements`
    );
  }
  const gridSize = parameters.gridSize ?? 0;
  const tolerance = parameters.tolerance ?? 0;
  if (!Number.isFinite(gridSize) || !Number.isFinite(tolerance) || gridSize < 0 || tolerance < 0) {
    throw new Error('Geometry cleanup gridSize and tolerance must be finite and non-negative');
  }
  target.set([gridSize, tolerance, parameters.removeRepeatedPoints === false ? 0 : 1, 0]);
  return target;
}

/** Outputs shared by every {@link GPUGeometryCleanup} geometry type. */
type GPUGeometryCleanupBaseOutput = {
  /** Compacted output vertices, `capacity` rows. Rows at or past `count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving the clamped vertex count. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when the vertex capacity overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped vertex count. */
  requiredCount?: GraphDataView<'uint32'>;
};

/** Caller-owned, capacity-bounded point outputs of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupPointOutput = GPUGeometryCleanupBaseOutput & {
  /** Point features have no ring offsets. */
  ringOffsets?: never;
  /** Point features cannot collapse. */
  collapsedRings?: never;
};

/** Caller-owned, capacity-bounded line or polygon outputs of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupOutput = GPUGeometryCleanupBaseOutput & {
  /**
   * Republished ring offsets, `ringCount + 1` rows, clamped to the capacity. Every input ring
   * keeps its index (collapsed rings become empty), so feature and polygon offsets of the input
   * stay valid.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Optional one-row scalar receiving the number of polygon rings emptied because fewer than 3
   * vertices survived (GEOS raises on these rings).
   */
  collapsedRings?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeometryCleanup}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, `geometryType`, the output capacity, and which optional outputs are present.
 */
export type GPUGeometryCleanupProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-cleanup'`. */
  id?: string;
  /** Packed vertex positions. Polygon rings repeat their first vertex as the last (GeoArrow). */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH}
   * elements written with `getGPUGeometryCleanupParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
} & (
  | {
      /** Independent Point features, one per position row. */
      geometryType: 'points';
      /** Point features do not use ring offsets. */
      ringOffsets?: never;
      /** Point output views. `output.positions.length` is the point capacity. */
      output: GPUGeometryCleanupPointOutput;
    }
  | {
      /** Paths or closed polygon rings; polygon rings that collapse are emptied. */
      geometryType: 'lines' | 'polygons';
      /** `ringCount + 1` monotonic offsets; ring `r` owns `[offsets[r], offsets[r + 1])`. */
      ringOffsets: GraphDataView<'uint32'>;
      /** Output views. `output.positions.length` is the vertex capacity. */
      output: GPUGeometryCleanupOutput;
    }
);

/**
 * Snaps coordinates to a grid and removes repeated line or polygon vertices on the GPU
 * (`shapely.set_precision(mode='pointwise')`, `shapely.remove_repeated_points`, PostGIS
 * `ST_SnapToGrid`, `ST_RemoveRepeatedPoints`).
 *
 * Snapping is applied first (when `gridSize > 0`), then repeats are removed by walking each ring
 * in order exactly like GEOS: the first and last vertex always survive; an interior vertex is
 * dropped when it lies within `tolerance` (inclusive) of the previously kept vertex; and a kept
 * interior vertex within `tolerance` of the last vertex is replaced by it. Rings and paths with
 * fewer than 2 vertices are untouched. A polygon ring left with fewer than 3 vertices is emptied
 * and counted in `collapsedRings`; the polygon, its feature and the other rings keep their index
 * (GEOS raises for these rings, and `set_precision(mode='pointwise')` alone never removes
 * vertices). Vertices outside every ring are dropped. In point mode each position row is an
 * independent Point feature: snapping applies, but `tolerance` and `removeRepeatedPoints` are
 * ignored so coincident features remain distinct.
 *
 * Composition: points use one direct per-row kernel. Lines and polygons use a per-ring kernel to
 * flag surviving vertices (sequential, deterministic, latency grows with the largest ring), an
 * exclusive `GPUScan` to rank them, a per-vertex emit kernel, and a publish kernel.
 *
 * Precision: f32. Snapping computes `floor(x / gridSize + 0.5) * gridSize`, exact for
 * `|x / gridSize| < 2^24`; ties in f32 may differ from the f64 GEOS decision for grid sizes that are
 * not powers of two.
 */
export class GPUGeometryCleanup implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryCleanupProps;

  constructor(props: GPUGeometryCleanupProps) {
    this.id = props.id ?? 'geometry-cleanup';
    this.props = props;
    const {id} = this;
    const ringOffsets = 'ringOffsets' in props ? props.ringOffsets : undefined;
    const outputRingOffsets = 'ringOffsets' in props.output ? props.output.ringOffsets : undefined;
    const outputCollapsedRings =
      'collapsedRings' in props.output ? props.output.collapsedRings : undefined;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(
        Object.entries(props.output).map(([key, view]) => [`output.${key}`, view])
      )
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (
      props.geometryType !== 'points' &&
      props.geometryType !== 'lines' &&
      props.geometryType !== 'polygons'
    ) {
      throw new Error(`${id} geometryType must be 'points', 'lines' or 'polygons'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (props.geometryType === 'points') {
      if (ringOffsets || outputRingOffsets || outputCollapsedRings) {
        throw new Error(`${id} point geometry must not provide ring outputs or offsets`);
      }
    } else {
      validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
      if (props.ringOffsets.length < 2) {
        throw new Error(`${id} ringOffsets must contain at least two rows`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH} float32 values`
      );
    }
    const {output} = props;
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must hold at least one row`);
    }
    if (props.geometryType !== 'points') {
      validatePackedUint32View(props.output.ringOffsets, `${id} output.ringOffsets`);
      if (props.output.ringOffsets.length !== props.ringOffsets.length) {
        throw new Error(`${id} output.ringOffsets length must equal ringOffsets length`);
      }
    }
    for (const [name, scalar] of [
      ['count', output.count],
      ['overflow', output.overflow],
      ['requiredCount', output.requiredCount],
      ['collapsedRings', outputCollapsedRings]
    ] as const) {
      if (scalar) {
        validatePackedUint32View(scalar, `${id} output.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.positions,
        outputRingOffsets,
        output.count,
        output.overflow,
        output.requiredCount,
        outputCollapsedRings
      ],
      [props.positions, ringOffsets, props.parameters]
    );
  }

  /**
   * Returns the point kernel, or the line/polygon clear, flag, scan, emit and publish nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    if (props.geometryType === 'points') {
      const {output} = props;
      validateGraphViewsBelongToGraph(id, graph, [
        props.positions,
        props.parameters,
        output.positions,
        output.count,
        output.overflow,
        output.requiredCount
      ]);
      return [
        createCleanupPointsNode<Parameters>(graph, {
          id: `${id}-points`,
          operation: OPERATION,
          positions: props.positions,
          parameters: props.parameters,
          outputPositions: output.positions,
          count: output.count,
          overflow: output.overflow,
          requiredCount: output.requiredCount
        })
      ];
    }
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.parameters,
      output.positions,
      output.ringOffsets,
      output.count,
      output.overflow,
      output.requiredCount,
      output.collapsedRings
    ]);
    const vertexCount = props.positions.length;
    const keepFlags = createTransientView(graph, `${id}-keep-flags`, 'uint32', vertexCount);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', vertexCount);
    const collapsedCount =
      output.collapsedRings ?? createTransientView(graph, `${id}-collapsed`, 'uint32', 1);
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-flags`,
        operation: OPERATION,
        view: keepFlags,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-collapsed`,
        operation: OPERATION,
        view: collapsedCount,
        type: 'u32',
        value: '0u',
        componentCount: 1
      }),
      createCleanupFlagNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        parameters: props.parameters,
        isPolygon: props.geometryType === 'polygons',
        keepFlags,
        collapsedCount
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: keepFlags,
        output: ranks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createCleanupEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        positions: props.positions,
        parameters: props.parameters,
        keepFlags,
        ranks,
        outputPositions: output.positions
      }),
      createCleanupPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        ringOffsets: props.ringOffsets,
        keepFlags,
        ranks,
        outputRingOffsets: output.ringOffsets,
        count: output.count,
        overflow: output.overflow,
        requiredCount: output.requiredCount,
        capacity: output.positions.length
      })
    ];
  }
}
