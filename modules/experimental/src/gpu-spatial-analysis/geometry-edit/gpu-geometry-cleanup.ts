// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {assert} from '@luma.gl/core';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  createCleanupEmitNode,
  createCleanupFlagNode,
  createCleanupPointsNode,
  createCleanupPublishNode,
  createCleanupRepeatApplyNode,
  createCleanupRepeatFlagNode,
  createCleanupRepeatGate,
  createCleanupRepeatGateNode,
  createCleanupRepeatResolveNode,
  CLEANUP_RING_STATE_WORDS,
  getCleanupRepeatGate
} from './geometry-edit-kernels';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUGeometryCleanup';

/** Number of float32 elements in a `GPUGeometryCleanup` parameter buffer. */
export const GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame settings of `GPUGeometryCleanup`. */
export type GPUGeometryCleanupSettings = {
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
 * Packs `GPUGeometryCleanup` settings into the 4-element float32 layout
 * `[gridSize, tolerance, removeRepeatedPoints, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `gridSize` or `tolerance` is negative or NaN, or `target` is too short.
 */
export function getGPUGeometryCleanupParameterValues(
  settings: GPUGeometryCleanupSettings = {},
  target: Float32Array = new Float32Array(GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH) {
    throw new Error(
      `Geometry cleanup target must hold ${GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH} elements`
    );
  }
  const gridSize = settings.gridSize ?? 0;
  const tolerance = settings.tolerance ?? 0;
  if (!Number.isFinite(gridSize) || !Number.isFinite(tolerance) || gridSize < 0 || tolerance < 0) {
    throw new Error('Geometry cleanup gridSize and tolerance must be finite and non-negative');
  }
  target.set([gridSize, tolerance, settings.removeRepeatedPoints === false ? 0 : 1, 0]);
  return target;
}

/** Completeness scalars of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupStatus = {
  /** One-row scalar receiving the clamped vertex count. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when the vertex capacity overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped vertex count. */
  requiredCount?: GraphDataView<'uint32'>;
};

/** Caller-owned, capacity-bounded point outputs of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupPointOutput = {
  /** Compacted output positions, `capacity` rows. Rows at or past `status.count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /** Point features have no line or ring offsets. */
  lineOffsets?: never;
  ringOffsets?: never;
  status: GPUGeometryCleanupStatus;
};

/** Caller-owned, capacity-bounded line outputs of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupLineOutput = {
  /** Compacted output vertices, `capacity` rows. Rows at or past `status.count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Republished line offsets, `lineCount + 1` rows, clamped to the capacity. Every input line
   * keeps its index, so the input `featureOffsets` stay valid.
   */
  lineOffsets: GraphDataView<'uint32'>;
  status: GPUGeometryCleanupStatus;
};

/** Caller-owned, capacity-bounded polygon outputs of {@link GPUGeometryCleanup}. */
export type GPUGeometryCleanupPolygonOutput = {
  /** Compacted output vertices, `capacity` rows. Rows at or past `status.count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Republished ring offsets, `ringCount + 1` rows, clamped to the capacity. Every input ring
   * keeps its index (collapsed rings become empty), so the input `featureOffsets` and
   * `polygonOffsets` stay valid: together with `positions` this output is a full polygon layout.
   */
  ringOffsets: GraphDataView<'uint32'>;
  status: GPUGeometryCleanupStatus & {
    /**
     * Optional one-row scalar receiving the number of polygon rings emptied because fewer than 3
     * vertices survived (GEOS raises on these rings).
     */
    collapsedCount?: GraphDataView<'uint32'>;
  };
};

/** Caller-owned output of {@link GPUGeometryCleanup}; the shape follows `geometryType`. */
export type GPUGeometryCleanupOutput =
  | GPUGeometryCleanupPointOutput
  | GPUGeometryCleanupLineOutput
  | GPUGeometryCleanupPolygonOutput;

/** Props shared by every geometry type of {@link GPUGeometryCleanup}. */
type GPUGeometryCleanupSharedProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-cleanup'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed vertex positions. Polygon rings repeat their first vertex as the last (GeoArrow). */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH}
   * elements written with `getGPUGeometryCleanupParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compiles only the sequential per-ring repeat pass even at `tolerance = 0`, for A/B timing
   * against the vertex-parallel pass.
   *
   * @internal
   */
  forceSequentialRepeatRemoval?: boolean;
};

/**
 * Properties for {@link GPUGeometryCleanup}, discriminated by `geometryType`.
 *
 * - `'points'`: independent Point features, one per position row; `output.positions.length` is the
 *   point capacity.
 * - `'lines'`: {@link GPULineGeometryFields}; `output.lineOffsets` republishes the line offsets.
 * - `'polygons'`: the four-level layout of {@link GPUPolygonGeometryFields}; polygon rings that
 *   collapse are emptied and `output.ringOffsets` republishes the ring offsets, so the input
 *   `featureOffsets` and `polygonOffsets` describe the output too.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, `geometryType`, the output capacity, and which optional outputs are present.
 */
export type GPUGeometryCleanupProps = GPUGeometryCleanupSharedProps &
  (
    | {geometryType: 'points'; output: GPUGeometryCleanupPointOutput}
    | ({geometryType: 'lines'} & GPULineGeometryFields & {output: GPUGeometryCleanupLineOutput})
    | ({geometryType: 'polygons'} & GPUPolygonGeometryFields & {
          output: GPUGeometryCleanupPolygonOutput;
        })
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
 * and counted in `status.collapsedCount`; the polygon, its feature and the other rings keep their index
 * (GEOS raises for these rings, and `set_precision(mode='pointwise')` alone never removes
 * vertices). Vertices outside every ring are dropped. In point mode each position row is an
 * independent Point feature: snapping applies, but `tolerance` and `removeRepeatedPoints` are
 * ignored so coincident features remain distinct.
 *
 * Composition: points use one direct per-row kernel. Lines and polygons flag surviving vertices,
 * rank them with an exclusive `GPUScan`, then run a per-vertex emit kernel and a publish kernel.
 * Flagging has two compiled paths and the GPU picks one per frame from the `tolerance` word, with
 * indirect dispatch gates: a per-ring kernel that walks each ring in order (used when
 * `tolerance > 0`; latency grows with the largest ring), and, for `tolerance = 0` (exact repeats
 * after snapping), a vertex-parallel path whose work per vertex is constant. In the parallel path
 * a vertex repeats when its snapped position equals the previous vertex's (`-0` equals `+0`; a
 * NaN or infinite coordinate never repeats, as in the sequential path), then a per-ring step
 * applies the closing rule and the polygon collapse rule, and a per-vertex step clears the
 * affected flags. Both paths give bitwise identical output at `tolerance = 0`. Positions with
 * subnormal differences are outside this guarantee if the device flushes subnormals.
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
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUGeometryCleanupProps) {
    this.id = props.id ?? 'geometry-cleanup';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
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
    const {output} = props;
    const layout = getCleanupLayout(props);
    if (props.geometryType === 'points') {
      const inputs = props as {lineOffsets?: unknown; ringOffsets?: unknown};
      if (
        inputs.lineOffsets ||
        inputs.ringOffsets ||
        props.output.lineOffsets ||
        props.output.ringOffsets
      ) {
        throw new Error(`${id} point geometry must not provide offsets`);
      }
    } else {
      // Lines and polygons always resolve both offsets views.
      assert(layout.ringOffsets && layout.outputRingOffsets);
      validatePackedView(layout.ringOffsets, ['uint32'], `${id} ringOffsets`);
      if (layout.ringOffsets.length < 2) {
        throw new Error(`${id} ringOffsets must contain at least two rows`);
      }
      validatePackedView(layout.outputRingOffsets, ['uint32'], `${id} output ring offsets`);
      if (layout.outputRingOffsets.length !== layout.ringOffsets.length) {
        throw new Error(`${id} output offsets length must equal input offsets length`);
      }
    }
    if (props.geometryType === 'polygons') {
      for (const name of ['polygonOffsets', 'featureOffsets'] as const) {
        validatePackedView(props[name], ['uint32'], `${id} ${name}`);
        if (props[name].length < 2) {
          throw new Error(`${id} ${name} must contain at least two rows`);
        }
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_GEOMETRY_CLEANUP_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must hold at least one row`);
    }
    const collapsedCount =
      'collapsedCount' in output.status ? output.status.collapsedCount : undefined;
    for (const [name, scalar] of [
      ['count', output.status.count],
      ['overflow', output.status.overflow],
      ['requiredCount', output.status.requiredCount],
      ['collapsedCount', collapsedCount]
    ] as const) {
      if (scalar) {
        validatePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.positions,
        layout.outputRingOffsets,
        output.status.count,
        output.status.overflow,
        output.status.requiredCount,
        collapsedCount
      ],
      [
        props.positions,
        layout.ringOffsets,
        layout.polygonOffsets,
        layout.featureOffsets,
        props.parameters
      ]
    );
  }

  /**
   * Returns the point kernel, or the line/polygon clear, flag (sequential and tolerance-0 parallel, GPU-gated), scan, emit and publish nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const layout = getCleanupLayout(props);
    if (props.geometryType === 'points') {
      const {output} = props;
      validateGraphViewsBelongToGraph(id, graph, [
        props.positions,
        props.parameters,
        output.positions,
        output.status.count,
        output.status.overflow,
        output.status.requiredCount
      ]);
      return [
        createCleanupPointsNode<Parameters>(graph, {
          id: `${id}-points`,
          operation: OPERATION,
          positions: props.positions,
          parameters: props.parameters,
          outputPositions: output.positions,
          count: output.status.count,
          overflow: output.status.overflow,
          requiredCount: output.status.requiredCount
        })
      ];
    }
    const {output} = props;
    const {ringOffsets, outputRingOffsets} = layout;
    // Lines and polygons always resolve both offsets views.
    assert(ringOffsets && outputRingOffsets);
    const collapsedRings =
      'collapsedCount' in output.status ? output.status.collapsedCount : undefined;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      ringOffsets,
      layout.polygonOffsets,
      layout.featureOffsets,
      props.parameters,
      output.positions,
      outputRingOffsets,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      collapsedRings
    ]);
    const vertexCount = props.positions.length;
    const keepFlags = createTransientView(graph, `${id}-keep-flags`, 'uint32', vertexCount);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', vertexCount);
    const collapsedCount =
      collapsedRings ?? createTransientView(graph, `${id}-collapsed`, 'uint32', 1);
    const isPolygon = props.geometryType === 'polygons';
    const ringCount = ringOffsets.length - 1;
    const parallel = !props.forceSequentialRepeatRemoval;
    const repeatGate = parallel
      ? createCleanupRepeatGate<Parameters>(graph, {
          id: `${id}-repeat-dispatch`,
          operation: OPERATION,
          vertexCount,
          ringCount
        })
      : undefined;
    const ringState = parallel
      ? createTransientView(
          graph,
          `${id}-ring-state`,
          'uint32',
          CLEANUP_RING_STATE_WORDS * ringCount
        )
      : undefined;
    const sequentialGate = repeatGate
      ? getCleanupRepeatGate(repeatGate, 'sequential', `${id}-flags`)
      : undefined;
    const flagNodes: GPUCommandNode<Parameters>[] = [
      ...(repeatGate && ringState
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-clear-ring-state`,
              operation: OPERATION,
              view: ringState,
              type: 'u32',
              value: '0u'
            }),
            createCleanupRepeatGateNode<Parameters>(graph, {
              id: `${id}-repeat-gate`,
              operation: OPERATION,
              parameters: props.parameters,
              gate: repeatGate
            })
          ]
        : []),
      createCleanupFlagNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets,
        parameters: props.parameters,
        isPolygon,
        keepFlags,
        collapsedCount,
        condition: sequentialGate?.condition,
        extraResources: sequentialGate?.extraResources
      })
    ];
    if (repeatGate && ringState) {
      const shared = (pass: 'flag' | 'resolve' | 'apply', nodeId: string) => ({
        id: nodeId,
        operation: OPERATION,
        ringOffsets,
        ringState,
        ...getCleanupRepeatGate(repeatGate, pass, nodeId)
      });
      flagNodes.push(
        createCleanupRepeatFlagNode<Parameters>(graph, {
          ...shared('flag', `${id}-repeat-flags`),
          positions: props.positions,
          parameters: props.parameters,
          keepFlags
        }),
        createCleanupRepeatResolveNode<Parameters>(graph, {
          ...shared('resolve', `${id}-repeat-resolve`),
          positions: props.positions,
          parameters: props.parameters,
          isPolygon,
          collapsedCount
        }),
        createCleanupRepeatApplyNode<Parameters>(graph, {
          ...shared('apply', `${id}-repeat-apply`),
          positionCount: vertexCount,
          keepFlags
        })
      );
    }
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
      ...flagNodes,
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
        ringOffsets,
        keepFlags,
        ranks,
        outputRingOffsets,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        capacity: output.positions.length
      })
    ];
  }
}

/** Input and output ring offsets of the props, plus the polygon levels above them. */
function getCleanupLayout(props: GPUGeometryCleanupProps): {
  ringOffsets?: GraphDataView<'uint32'>;
  outputRingOffsets?: GraphDataView<'uint32'>;
  polygonOffsets?: GraphDataView<'uint32'>;
  featureOffsets?: GraphDataView<'uint32'>;
} {
  switch (props.geometryType) {
    case 'lines':
      return {ringOffsets: props.lineOffsets, outputRingOffsets: props.output.lineOffsets};
    case 'polygons':
      return {
        ringOffsets: props.ringOffsets,
        outputRingOffsets: props.output.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        featureOffsets: props.featureOffsets
      };
    default:
      return {};
  }
}
