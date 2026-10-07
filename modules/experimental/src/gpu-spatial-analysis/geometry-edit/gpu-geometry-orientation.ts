// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {createRingOrientationNode, createRingReverseNode} from './geometry-edit-kernels';

const OPERATION = 'GPUGeometryOrientation';

/** Number of float32 elements in a `GPUGeometryOrientation` parameter buffer. */
export const GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPUGeometryOrientation`. */
export type GPUGeometryOrientationParameters = {
  /**
   * Winding of polygon exteriors in a y-up coordinate system: `false` (default) is
   * counter-clockwise exteriors with clockwise holes (GeoJSON RFC 7946, `exterior_cw=False`),
   * `true` is clockwise exteriors with counter-clockwise holes (shapefile, `exterior_cw=True`).
   */
  exteriorClockwise?: boolean;
};

/**
 * Packs `GPUGeometryOrientation` parameters into the 4-element float32 layout
 * `[exteriorClockwise, 0, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `target` is too short.
 */
export function getGPUGeometryOrientationParameterValues(
  parameters: GPUGeometryOrientationParameters = {},
  target: Float32Array = new Float32Array(GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH) {
    throw new Error(
      `Geometry orientation target must hold ${GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH} elements`
    );
  }
  target.set([parameters.exteriorClockwise ? 1 : 0, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUGeometryOrientation}. */
export type GPUGeometryOrientationOutput = {
  /** Output vertices, same length and ring layout as the input `positions`. */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-ring flags, `ringCount` rows: 1 when the ring's vertex order was reversed. */
  reversedRings?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeometryOrientation}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, `mode`, and which optional outputs are present.
 */
export type GPUGeometryOrientationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-orientation'`. */
  id?: string;
  /** Packed vertex positions. Rings may repeat their first vertex as the last. */
  positions: GraphDataView<'float32x2'>;
  /** `ringCount + 1` monotonic vertex offsets; ring (or path) `r` owns `[ringOffsets[r], ringOffsets[r + 1])`. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * `'reverse'` reverses the vertex order of every ring and path (`shapely.reverse`).
   * `'orient-polygons'` reverses only rings that wind the wrong way for their role, exterior or
   * hole (`shapely.orient_polygons`).
   */
  mode: 'reverse' | 'orient-polygons';
  /**
   * `polygonCount + 1` monotonic ring offsets for `'orient-polygons'` (required there): polygon `p`
   * owns rings `[polygonOffsets[p], polygonOffsets[p + 1])` and its first ring is the exterior. A
   * multi-polygon is several polygons. Ignored by `'reverse'`.
   */
  polygonOffsets?: GraphDataView<'uint32'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH}
   * elements written with `getGPUGeometryOrientationParameterValues`. Required for
   * `'orient-polygons'`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Output views. */
  output: GPUGeometryOrientationOutput;
};

/**
 * Reverses rings or orients polygons on the GPU (`shapely.reverse`, `shapely.orient_polygons`,
 * PostGIS `ST_Reverse`, `ST_ForcePolygonCW`, `ST_ForcePolygonCCW`).
 *
 * The vertex count and ring layout never change, so `ringOffsets` (and any feature offsets) stay
 * valid for the output. A ring's winding comes from the sign of its shoelace area, summed in
 * f32 in coordinates relative to the ring's first vertex; rings with fewer than 3 vertices or
 * exactly zero area are left alone. For valid simple rings this agrees with GEOS; for
 * self-intersecting rings GEOS uses a different test (the highest vertex).
 *
 * Composition: a per-ring kernel decides which rings to flip, then a per-vertex kernel copies the
 * positions, mirroring the vertex order inside flipped rings (binary search of `ringOffsets`).
 * Deterministic. Latency of the first kernel grows with the largest ring.
 */
export class GPUGeometryOrientation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryOrientationProps;

  constructor(props: GPUGeometryOrientationProps) {
    this.id = props.id ?? 'geometry-orientation';
    this.props = props;
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      polygonOffsets: props.polygonOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(
        Object.entries(props.output).map(([key, view]) => [`output.${key}`, view])
      )
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (props.mode !== 'reverse' && props.mode !== 'orient-polygons') {
      throw new Error(`${id} mode must be 'reverse' or 'orient-polygons'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must contain at least two rows`);
    }
    if (props.mode === 'orient-polygons') {
      if (!props.polygonOffsets || !props.parameters) {
        throw new Error(`${id} 'orient-polygons' requires polygonOffsets and parameters`);
      }
      validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
      if (props.polygonOffsets.length < 2) {
        throw new Error(`${id} polygonOffsets must contain at least two rows`);
      }
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    const {output} = props;
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length !== props.positions.length) {
      throw new Error(`${id} output.positions length must equal positions length`);
    }
    if (output.reversedRings) {
      validatePackedUint32View(output.reversedRings, `${id} output.reversedRings`);
      if (output.reversedRings.length !== props.ringOffsets.length - 1) {
        throw new Error(`${id} output.reversedRings must hold one row per ring`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.positions, output.reversedRings],
      [props.positions, props.ringOffsets, props.polygonOffsets, props.parameters]
    );
  }

  /** Returns the per-ring orientation node and the per-vertex copy node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.parameters,
      output.positions,
      output.reversedRings
    ]);
    const ringFlags =
      output.reversedRings ??
      createTransientView(graph, `${id}-ring-flags`, 'uint32', props.ringOffsets.length - 1);
    return [
      createRingOrientationNode<Parameters>(graph, {
        id: `${id}-orientation`,
        operation: OPERATION,
        mode: props.mode,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        polygonOffsets: props.polygonOffsets,
        parameters: props.parameters,
        ringFlags
      }),
      createRingReverseNode<Parameters>(graph, {
        id: `${id}-reverse`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        ringFlags,
        outputPositions: output.positions
      })
    ];
  }
}
