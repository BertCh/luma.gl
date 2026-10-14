// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {createRingOrientationNode, createRingReverseNode} from './geometry-edit-kernels';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUGeometryOrientation';

/** Number of float32 elements in a `GPUGeometryOrientation` parameter buffer. */
export const GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame settings of `GPUGeometryOrientation`. */
export type GPUGeometryOrientationSettings = {
  /**
   * Winding of polygon exteriors in a y-up coordinate system: `false` (default) is
   * counter-clockwise exteriors with clockwise holes (GeoJSON RFC 7946, `exterior_cw=False`),
   * `true` is clockwise exteriors with counter-clockwise holes (shapefile, `exterior_cw=True`).
   */
  exteriorClockwise?: boolean;
};

/**
 * Packs `GPUGeometryOrientation` settings into the 4-element float32 layout
 * `[exteriorClockwise, 0, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `target` is too short.
 */
export function getGPUGeometryOrientationParameterValues(
  settings: GPUGeometryOrientationSettings = {},
  target: Float32Array = new Float32Array(GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH) {
    throw new Error(
      `Geometry orientation target must hold ${GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH} elements`
    );
  }
  target.set([settings.exteriorClockwise ? 1 : 0, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUGeometryOrientation}. */
export type GPUGeometryOrientationOutput = {
  /** Output vertices, same length and ring layout as the input `positions`. */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-ring flags, `ringCount` rows: 1 when the ring's vertex order was reversed. */
  reversedRings?: GraphDataView<'uint32'>;
};

/** Props shared by both modes of {@link GPUGeometryOrientation}. */
type GPUGeometryOrientationSharedProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-orientation'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Output views. */
  output: GPUGeometryOrientationOutput;
};

/**
 * Properties for {@link GPUGeometryOrientation}, discriminated by `mode`.
 *
 * - `'orient-polygons'` reverses only rings that wind the wrong way for their role, exterior or
 *   hole (`shapely.orient_polygons`). Takes the four-level layout of
 *   {@link GPUPolygonGeometryFields} (the first ring of every polygon is the exterior) and
 *   `parameters`, a per-frame packed float32 view of at least
 *   {@link GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH} elements written with
 *   `getGPUGeometryOrientationParameterValues`. Winding is judged at the ring's lexicographically
 *   smallest vertex with exact orientation, so sliver rings are not misjudged by f32 area
 *   cancellation.
 * - `'reverse'` reverses the vertex order of every ring and path (`shapely.reverse`). Takes
 *   polygons (the four-level layout) or {@link GPULineGeometryFields}.
 *
 * The output keeps the input layout: pass the input `featureOffsets`, `polygonOffsets` and
 * `ringOffsets` (or `lineOffsets`) together with `output.positions` to the next contributor.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, `mode`, and which optional outputs are present.
 */
export type GPUGeometryOrientationProps = GPUGeometryOrientationSharedProps &
  (
    | ({mode: 'orient-polygons'} & GPUPolygonGeometryFields & {
          /** Per-frame packed float32 view, see {@link GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH}. */
          parameters: GraphDataView<'float32'>;
        })
    | ({mode: 'reverse'; parameters?: undefined} & (
        | GPUPolygonGeometryFields
        | GPULineGeometryFields
      ))
  );

/**
 * Reverses rings or orients polygons on the GPU (`shapely.reverse`, `shapely.orient_polygons`,
 * PostGIS `ST_Reverse`, `ST_ForcePolygonCW`, `ST_ForcePolygonCCW`).
 *
 * The vertex count and ring layout never change, so `ringOffsets` (and any feature offsets) stay
 * valid for the output. A ring's winding comes from the exact orientation
 * sign (`orientSign`, no f32 sum) of the triangle at the ring's lexicographically smallest vertex
 * (lowest x, then lowest y; the first one on ties) and its nearest distinct neighbours; rings with
 * fewer than 3 vertices or no two distinct neighbours are left alone, as is a ring whose sign is
 * zero or uncertain. This is the GEOS extreme-vertex test, so a self-intersecting ring is judged by
 * that vertex rather than by its signed area.
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
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUGeometryOrientationProps) {
    this.id = props.id ?? 'geometry-orientation';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    if (props.mode !== 'reverse' && props.mode !== 'orient-polygons') {
      throw new Error(`${id} mode must be 'reverse' or 'orient-polygons'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    const {ringOffsets, polygonOffsets} = getOrientationLayout(props);
    validatePackedView(ringOffsets, ['uint32'], `${id} ringOffsets`);
    if (ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must contain at least two rows`);
    }
    if (props.mode === 'orient-polygons') {
      if (!('featureOffsets' in props) || !polygonOffsets) {
        throw new Error(`${id} 'orient-polygons' requires the four-level polygon layout`);
      }
      for (const name of ['polygonOffsets', 'featureOffsets'] as const) {
        validatePackedView(props[name], ['uint32'], `${id} ${name}`);
        if (props[name].length < 2) {
          throw new Error(`${id} ${name} must contain at least two rows`);
        }
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
      validatePackedView(output.reversedRings, ['uint32'], `${id} output.reversedRings`);
      if (output.reversedRings.length !== ringOffsets.length - 1) {
        throw new Error(`${id} output.reversedRings must hold one row per ring`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.positions, output.reversedRings],
      [props.positions, ringOffsets, polygonOffsets, props.parameters]
    );
  }

  /** Returns the per-ring orientation node and the per-vertex copy node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const {ringOffsets, polygonOffsets} = getOrientationLayout(props);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      ringOffsets,
      polygonOffsets,
      props.parameters,
      output.positions,
      output.reversedRings
    ]);
    const ringFlags =
      output.reversedRings ??
      createTransientView(graph, `${id}-ring-flags`, 'uint32', ringOffsets.length - 1);
    return [
      createRingOrientationNode<Parameters>(graph, {
        id: `${id}-orientation`,
        operation: OPERATION,
        mode: props.mode,
        positions: props.positions,
        ringOffsets,
        polygonOffsets: props.mode === 'orient-polygons' ? polygonOffsets : undefined,
        parameters: props.parameters,
        ringFlags
      }),
      createRingReverseNode<Parameters>(graph, {
        id: `${id}-reverse`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets,
        ringFlags,
        outputPositions: output.positions
      })
    ];
  }
}

/** Ring offsets (rings or lines) and, for polygons, polygon-to-ring offsets of the props. */
function getOrientationLayout(props: GPUGeometryOrientationProps): {
  ringOffsets: GraphDataView<'uint32'>;
  polygonOffsets?: GraphDataView<'uint32'>;
} {
  return 'lineOffsets' in props
    ? {ringOffsets: props.lineOffsets}
    : {ringOffsets: props.ringOffsets, polygonOffsets: props.polygonOffsets};
}
