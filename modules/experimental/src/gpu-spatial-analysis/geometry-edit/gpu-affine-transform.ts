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
import {GPUGeometryMeasures} from '../geometry-measures/gpu-geometry-measures';
import type {GPUGeometryHoleRule} from '../geometry-measures/geometry-measures-kernels';
import {createAffineTransformNode} from './geometry-edit-kernels';

const OPERATION = 'GPUAffineTransform';

/** Number of float32 elements in a `GPUAffineTransform` parameter buffer. */
export const GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH = 12;

/** Per-feature origin kinds that {@link GPUAffineTransform} can compute on the GPU. */
export type GPUAffineTransformOrigin = 'center' | 'centroid';

/** CPU description of the per-frame parameters of `GPUAffineTransform`. */
export type GPUAffineTransformParameters = {
  /** Rotation angle, counter-clockwise in a y-up system. Degrees unless `useRadians`. Default 0. */
  rotate?: number;
  /** Interpret `rotate`, `skew` as radians. Default false (degrees, as shapely). */
  useRadians?: boolean;
  /** Scale factors `[x, y]`, or one number for both. Default 1. */
  scale?: number | readonly [number, number];
  /** Shear angles `[x, y]` (`shapely.affinity.skew`: `xs`, `ys`). Default 0. */
  skew?: readonly [number, number];
  /** Translation `[x, y]` applied last. Default `[0, 0]`. */
  translate?: readonly [number, number];
  /**
   * Fixed point of the scale, skew and rotation: a coordinate, the per-feature bounding box
   * `'center'`, or the per-feature `'centroid'` of the input geometry. Default `[0, 0]` (shapely
   * defaults to `'center'`; ask for it explicitly). `'center'` and `'centroid'` only work when the
   * contributor was created with the matching entry in `origins`; otherwise they act as `[0, 0]`.
   */
  origin?: 'center' | 'centroid' | readonly [number, number];
};

/**
 * Packs `GPUAffineTransform` parameters into the 12-element float32 layout
 * `[a, b, d, e, translateX, translateY, originX, originY, originMode, 0, 0, 0]` where
 * `originMode` is 0 (point), 1 (`'center'`) or 2 (`'centroid'`), and every vertex maps to
 * `[a b; d e] * (p - origin) + origin + translate`. The matrix is the composition shapely applies
 * when calling `scale`, then `skew`, then `rotate`, then `translate` about the same origin:
 * `R * K * S` with `S = diag(scale)`, `K = [1 tan(xs); tan(ys) 1]`, `R = [cos -sin; sin cos]`,
 * evaluated in f64 before narrowing to f32.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 12 elements.
 * @throws If a number is not finite, or `target` is too short.
 */
export function getGPUAffineTransformParameters(
  parameters: GPUAffineTransformParameters = {},
  target: Float32Array = new Float32Array(GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH) {
    throw new Error(
      `Affine transform target must hold ${GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH} elements`
    );
  }
  const angleScale = parameters.useRadians ? 1 : Math.PI / 180;
  const rotate = (parameters.rotate ?? 0) * angleScale;
  const scale = parameters.scale ?? 1;
  const [scaleX, scaleY] = typeof scale === 'number' ? [scale, scale] : scale;
  const [skewX, skewY] = parameters.skew ?? [0, 0];
  const [translateX, translateY] = parameters.translate ?? [0, 0];
  const origin = parameters.origin ?? [0, 0];
  const originPoint = typeof origin === 'string' ? [0, 0] : origin;
  const originMode = origin === 'center' ? 1 : origin === 'centroid' ? 2 : 0;
  if (
    ![rotate, scaleX, scaleY, skewX, skewY, translateX, translateY, ...originPoint].every(
      Number.isFinite
    )
  ) {
    throw new Error('Affine transform parameters must be finite numbers');
  }
  const tanX = Math.tan(skewX * angleScale);
  const tanY = Math.tan(skewY * angleScale);
  const cosine = Math.cos(rotate);
  const sine = Math.sin(rotate);
  // M = R * K * S, with K * S = [[sx, tanX * sy], [tanY * sx, sy]].
  const ksa = scaleX;
  const ksb = tanX * scaleY;
  const ksd = tanY * scaleX;
  const kse = scaleY;
  target.set([
    cosine * ksa - sine * ksd,
    cosine * ksb - sine * kse,
    sine * ksa + cosine * ksd,
    sine * ksb + cosine * kse,
    translateX,
    translateY,
    originPoint[0],
    originPoint[1],
    originMode,
    0,
    0,
    0
  ]);
  return target;
}

/** Caller-owned output of {@link GPUAffineTransform}. */
export type GPUAffineTransformOutput = {
  /** Transformed vertices, same length and layout as the input `positions`. */
  positions: GraphDataView<'float32x2'>;
};

/**
 * Properties for {@link GPUAffineTransform}.
 *
 * Per-frame (no recompile): the contents of `parameters` or `featureTransforms` and of every
 * input buffer. Compile-time: view lengths, `geometryType`, `holeRule`, `origins`, and which of
 * `parameters` or `featureTransforms` is used.
 */
export type GPUAffineTransformProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'affine-transform'`. */
  id?: string;
  /** Packed planar vertex positions. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional `ringCount + 1` monotonic vertex offsets. Required with `featureTransforms` and with
   * per-feature `origins`; ring `r` owns `[ringOffsets[r], ringOffsets[r + 1])`.
   */
  ringOffsets?: GraphDataView<'uint32'>;
  /**
   * Optional `featureCount + 1` monotonic ring offsets; when omitted every ring is its own
   * feature. Requires `ringOffsets`.
   */
  featureRingOffsets?: GraphDataView<'uint32'>;
  /**
   * Geometry type used for `'centroid'` origins (length-weighted for lines, area-weighted for
   * polygons). `'points'` supports only the `'center'` origin. Default `'lines'`.
   */
  geometryType?: 'points' | 'lines' | 'polygons';
  /** How polygon rings combine for the centroid, see `GPUGeometryMeasures`. Default `'winding'`. */
  holeRule?: GPUGeometryHoleRule;
  /** Per-feature origins that `parameters` may select. Default none. */
  origins?: readonly GPUAffineTransformOrigin[];
  /**
   * Per-frame packed float32 view of at least {@link GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH}
   * elements written with {@link getGPUAffineTransformParameters}: one matrix for every feature.
   * Exactly one of `parameters` and `featureTransforms` is required.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * `6 * featureCount` float32 rows of `[a, b, d, e, xoff, yoff]` per feature, with
   * `x' = a x + b y + xoff` and `y' = d x + e y + yoff` (`shapely.affinity.affine_transform`
   * order), applied to absolute coordinates.
   */
  featureTransforms?: GraphDataView<'float32'>;
  /** Output view. */
  output: GPUAffineTransformOutput;
};

/**
 * Applies 2D affine transforms to many features at once on the GPU (`shapely.affinity.affine_transform`,
 * `translate`, `scale`, `rotate`, `skew`; PostGIS `ST_Affine`, `ST_Translate`, `ST_Scale`,
 * `ST_Rotate`).
 *
 * Two forms. **Global** (`parameters`): one per-frame matrix with translation and an origin,
 * packed by {@link getGPUAffineTransformParameters}; sliders and animation rewrite the buffer
 * without recompiling. The origin may be a point, or computed per feature from the input geometry
 * (`'center'` bounding-box center, `'centroid'`) by composing `GPUGeometryMeasures`. Vertices are
 * transformed relative to the origin, so rotating large projected coordinates about a feature's
 * own center keeps f32 precision. **Per feature** (`featureTransforms`): one `[a b d e xoff yoff]`
 * row set per feature, applied to absolute coordinates.
 *
 * Unlike shapely, which recomputes `'center'` after every step, the origin is computed once from
 * the input geometry; the two agree for scale, rotate and `'centroid'`, and differ for skew about
 * `'center'` (shapely re-centers on the sheared bounding box). Pin shapely to a fixed origin to
 * compare. The vertex count and
 * ring layout never change, so the input offsets stay valid for the output. Vertices outside every
 * ring use the nearest ring's feature.
 */
export class GPUAffineTransform implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAffineTransformProps;
  /** Number of features, or 0 when no ring offsets are given. */
  readonly featureCount: number;
  /** Resolved geometry type. */
  readonly geometryType: 'points' | 'lines' | 'polygons';
  /** Resolved per-feature origins. */
  readonly origins: readonly GPUAffineTransformOrigin[];

  constructor(props: GPUAffineTransformProps) {
    this.id = props.id ?? 'affine-transform';
    this.props = props;
    this.geometryType = props.geometryType ?? 'lines';
    this.origins = props.origins ?? [];
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureRingOffsets: props.featureRingOffsets,
      parameters: props.parameters,
      featureTransforms: props.featureTransforms,
      ...Object.fromEntries(
        Object.entries(props.output).map(([key, view]) => [`output.${key}`, view])
      )
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (!['points', 'lines', 'polygons'].includes(this.geometryType)) {
      throw new Error(`${id} geometryType must be 'points', 'lines' or 'polygons'`);
    }
    for (const origin of this.origins) {
      if (origin !== 'center' && origin !== 'centroid') {
        throw new Error(`${id} origins must contain only 'center' or 'centroid'`);
      }
    }
    if (this.origins.includes('centroid') && this.geometryType === 'points') {
      throw new Error(`${id} the 'centroid' origin needs geometryType 'lines' or 'polygons'`);
    }
    if (Boolean(props.parameters) === Boolean(props.featureTransforms)) {
      throw new Error(`${id} needs exactly one of parameters and featureTransforms`);
    }
    if (props.featureTransforms && this.origins.length > 0) {
      throw new Error(`${id} origins apply only to parameters, not featureTransforms`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (props.ringOffsets) {
      validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
      if (props.ringOffsets.length < 2) {
        throw new Error(`${id} ringOffsets must contain at least two rows`);
      }
    }
    if (props.featureRingOffsets) {
      if (!props.ringOffsets) {
        throw new Error(`${id} featureRingOffsets requires ringOffsets`);
      }
      validatePackedUint32View(props.featureRingOffsets, `${id} featureRingOffsets`);
      if (props.featureRingOffsets.length < 2) {
        throw new Error(`${id} featureRingOffsets must contain at least two rows`);
      }
    }
    this.featureCount = props.featureRingOffsets
      ? props.featureRingOffsets.length - 1
      : props.ringOffsets
        ? props.ringOffsets.length - 1
        : 0;
    if ((props.featureTransforms || this.origins.length > 0) && !props.ringOffsets) {
      throw new Error(`${id} per-feature transforms and origins require ringOffsets`);
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_AFFINE_TRANSFORM_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    if (props.featureTransforms) {
      validatePackedView(props.featureTransforms, ['float32'], `${id} featureTransforms`);
      if (props.featureTransforms.length !== 6 * this.featureCount) {
        throw new Error(`${id} featureTransforms must hold 6 rows per feature`);
      }
    }
    validatePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    if (props.output.positions.length !== props.positions.length) {
      throw new Error(`${id} output.positions length must equal positions length`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions],
      [
        props.positions,
        props.ringOffsets,
        props.featureRingOffsets,
        props.parameters,
        props.featureTransforms
      ]
    );
  }

  /** Returns the optional measurement nodes and the per-vertex transform node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount, origins} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.featureRingOffsets,
      props.parameters,
      props.featureTransforms,
      props.output.positions
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    let bounds: GraphDataView<'float32x4'> | undefined;
    let centroids: GraphDataView<'float32x2'> | undefined;
    if (origins.length > 0) {
      const wantsCenter = origins.includes('center');
      const wantsCentroid = origins.includes('centroid');
      // Centroids are only valid for lines and polygons; bounds work for any geometry.
      bounds = wantsCenter
        ? createTransientView(graph, `${id}-bounds`, 'float32x4', featureCount)
        : undefined;
      centroids = wantsCentroid
        ? createTransientView(graph, `${id}-centroids`, 'float32x2', featureCount)
        : undefined;
      nodes.push(
        ...new GPUGeometryMeasures({
          spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
          id: `${id}-measures`,
          positions: props.positions,
          geometryType: this.geometryType === 'polygons' ? 'polygons' : 'lines',
          ringOffsets: props.ringOffsets!,
          featureRingOffsets: props.featureRingOffsets,
          holeRule: props.holeRule,
          output: {bounds, centroids}
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createAffineTransformNode<Parameters>(graph, {
        id: `${id}-transform`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        featureRingOffsets: props.featureRingOffsets,
        parameters: props.parameters,
        featureTransforms: props.featureTransforms,
        bounds,
        centroids,
        featureCount,
        outputPositions: props.output.positions
      })
    );
    return nodes;
  }
}
