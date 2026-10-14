// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
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
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {GPUGeometryMeasures} from '../geometry-measures/index';
import {
  createCompactnessNode,
  createConvexityNode,
  createMomentsNode,
  createOrientationNode,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH
} from './shape-descriptors-kernels';
import {canUseMonotoneChainConvexity, getMonotoneChainConvexityNodes} from './convexity-hull';

const OPERATION = 'GPUShapeDescriptors';

export {GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH};

/** Default sliver threshold: Polsby-Popper below this flags a sliver (about a 1:60 rectangle). */
export const GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD = 0.05;

/** CPU description of the per-frame settings of {@link GPUShapeDescriptors}. */
export type GPUShapeDescriptorsSettings = {
  /**
   * Polsby-Popper compactness below which a polygon is flagged as a sliver. Default
   * {@link GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD}.
   */
  sliverThreshold?: number;
};

/**
 * Packs {@link GPUShapeDescriptorsSettings} into the 4-element float32 layout
 * `[sliverThreshold, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the threshold without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the threshold is not a non-negative finite number, or `target` is too short.
 */
export function getGPUShapeDescriptorsParameterValues(
  settings: GPUShapeDescriptorsSettings = {},
  target: Float32Array = new Float32Array(GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
    throw new Error(
      `Shape descriptors target must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} elements`
    );
  }
  const sliverThreshold =
    settings.sliverThreshold ?? GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD;
  if (!Number.isFinite(sliverThreshold) || sliverThreshold < 0) {
    throw new Error('Shape descriptors sliverThreshold must be a non-negative finite number');
  }
  target.set([sliverThreshold, 0, 0, 0]);
  return target;
}

/**
 * How {@link GPUShapeDescriptors} computes `convexity`: `'gift-wrapping'` marches a private hull
 * per feature in one invocation (`O(vertices * hull vertices)` each, no extra passes, best for many
 * small features); `'monotone-chain'` builds every feature's hull with `GPUGroupConvexHull`
 * (`O(n log n)` total, balanced for large or skewed features, but about thirty small passes);
 * `'auto'` picks `'monotone-chain'` when features average at least
 * {@link GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES} vertices.
 */
export type GPUShapeDescriptorsConvexityMethod = 'auto' | 'gift-wrapping' | 'monotone-chain';

/** Average vertices per feature from which `convexityMethod: 'auto'` uses the monotone chain. */
export const GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES = 256;

/** Per-feature output columns of {@link GPUShapeDescriptors}. Every column is optional. */
export type GPUShapeDescriptorsOutput = {
  /** Polygon area (`|signed area|`), the same column `GPUGeometryMeasures` writes. */
  areas?: GraphDataView<'float32'>;
  /** Perimeter over every ring, the same column `GPUGeometryMeasures` writes as `lengths`. */
  perimeters?: GraphDataView<'float32'>;
  /** Polsby-Popper compactness `4 pi A / P^2`: 1 for a circle, toward 0 for thin or ragged shapes. */
  polsbyPopper?: GraphDataView<'float32'>;
  /** Schwartzberg compactness `P_circle / P = 2 sqrt(pi A) / P`, equal to `sqrt(polsbyPopper)`. */
  schwartzberg?: GraphDataView<'float32'>;
  /**
   * Elongation `1 - sqrt(lambda2 / lambda1)` from the eigenvalues of the second central moments:
   * 0 for a circle or square, toward 1 for a line.
   */
  elongation?: GraphDataView<'float32'>;
  /**
   * Orientation of the major axis in radians in `(-pi / 2, pi / 2]`, counter-clockwise from +x.
   * 0 for shapes with no distinguished axis.
   */
  orientation?: GraphDataView<'float32'>;
  /**
   * Convexity `A / hull area` in `(0, 1]` over the hull of every vertex of the feature (1 for a
   * convex polygon).
   */
  convexity?: GraphDataView<'float32'>;
  /** `1` when the first (exterior) ring is clockwise in the plane (y up), else `0`. */
  clockwise?: GraphDataView<'uint32'>;
  /** `1` when the perimeter is positive and Polsby-Popper is below the threshold or area is 0. */
  sliver?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUShapeDescriptors}.
 *
 * Per-frame (no recompile): the contents of `parameters` (sliver threshold) and every input
 * buffer. Compile-time: view lengths and which outputs are present.
 *
 * Polygons are the four-level layout of {@link GPUPolygonGeometryFields}, one row per feature. Ring
 * role is positional: the first ring of each polygon is a shell and later rings are holes,
 * whatever their winding, so multi-polygons with mixed winding and holed polygons measure
 * correctly. The `clockwise` output reports the first ring of each feature.
 */
export type GPUShapeDescriptorsProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'shape-descriptors'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   * Measures both distance and area, so neither local frame is exact: use an
   * azimuthal-equidistant frame for study areas small enough that its area distortion is
   * negligible, or a projected CRS.
   */
  spatialContext?: GPUSpatialContext;
  /** Hull algorithm behind `output.convexity`. Default `'auto'`. */
  convexityMethod?: GPUShapeDescriptorsConvexityMethod;
  /** Per-frame packed float32 view of at least 4 elements, written with {@link getGPUShapeDescriptorsParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Per-feature output columns; at least one is required. */
  output: GPUShapeDescriptorsOutput;
};

/**
 * Per-polygon shape descriptors: Polsby-Popper and Schwartzberg compactness, convexity,
 * elongation and orientation, plus clockwise and sliver flags (momepy and PySAL `esda`
 * compactness, turf `polygonSmooth` diagnostics, PostGIS `ST_IsPolygonCW`, QGIS "Polygon
 * compactness").
 *
 * Composition: an internal `GPUGeometryMeasures` supplies area, perimeter (including holes) and
 * centroid; a moments node integrates second central moments with Green's theorem about the
 * centroid in Neumaier-compensated f32 (one invocation per feature, deterministic); small nodes
 * derive the descriptors. Convexity runs a private gift-wrapping hull per feature with O(1)
 * memory, so its cost is O(vertices * hull vertices) per feature and grows with the largest
 * feature; `convexityMethod: 'monotone-chain'` (chosen by `'auto'` for large average features)
 * uses `GPUGroupConvexHull` instead, which is O(n log n) and balanced across skewed features.
 * Orientation and elongation weight holes negatively, so a polygon with a large hole is
 * described by its area, not its outline. Planar only: a longitude/latitude `spatialContext` throws
 * at construction, so project first (`createGPULocalFrame` with `GPULocalFrameProjection`). The
 * resolved `spatialContext`, including any `frame` identity, is passed to the internal
 * `GPUGeometryMeasures`, so it propagates to the measures instead of a fixed planar descriptor.
 *
 * Degenerate features (zero area or perimeter) report NaN for ratios, 0 for the clockwise flag,
 * and `sliver = 1` when the perimeter is positive.
 *
 * Precision: f32. Moments are relative to the centroid, so far-from-origin coordinates keep their
 * small-scale precision; the descriptors match f64 references to about 1e-4 relative.
 */
export class GPUShapeDescriptors implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUShapeDescriptorsProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features. */
  readonly featureCount: number;
  /** Whether convexity uses the sort-based monotone chain hull instead of gift wrapping. */
  readonly useMonotoneChainConvexity: boolean;

  constructor(props: GPUShapeDescriptorsProps) {
    this.id = props.id ?? 'shape-descriptors';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const {id} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
      validatePackedView(props[name], ['uint32'], `${id} ${name}`);
      if (props[name].length < 2) {
        throw new Error(`${id} ${name} must contain at least two rows`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    const convexityMethod = props.convexityMethod ?? 'auto';
    if (!['auto', 'gift-wrapping', 'monotone-chain'].includes(convexityMethod)) {
      throw new Error(`${id} convexityMethod must be 'auto', 'gift-wrapping' or 'monotone-chain'`);
    }
    const canChain = canUseMonotoneChainConvexity(props.positions.length, this.featureCount);
    if (convexityMethod === 'monotone-chain' && !canChain) {
      throw new Error(`${id} monotone-chain convexity supports fewer than 2^24 rows and features`);
    }
    this.useMonotoneChainConvexity =
      canChain &&
      (convexityMethod === 'monotone-chain' ||
        (convexityMethod === 'auto' &&
          props.positions.length >=
            GPU_SHAPE_DESCRIPTORS_MONOTONE_CHAIN_AVERAGE_VERTICES * this.featureCount));
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} float32 values`
      );
    }
    const columns = Object.entries(props.output).filter(([, view]) => view);
    if (columns.length === 0) {
      throw new Error(`${id} requires at least one output column`);
    }
    for (const [name, view] of columns) {
      validatePackedView(
        view as GraphDataView,
        name === 'clockwise' || name === 'sliver' ? ['uint32'] : ['float32'],
        `${id} output.${name}`
      );
      if ((view as GraphDataView).length !== this.featureCount) {
        throw new Error(`${id} output.${name} must hold ${this.featureCount} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      columns.map(([, view]) => view as GraphDataView),
      [
        props.positions,
        props.ringOffsets,
        props.featureOffsets,
        props.polygonOffsets,
        props.parameters
      ]
    );
  }

  /**
   * Returns the internal measures nodes, then the moments, compactness, orientation and convexity
   * nodes for the requested outputs.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.featureOffsets,
      props.polygonOffsets,
      props.parameters,
      ...Object.values(output)
    ]);
    const needsMoments = Boolean(output.elongation || output.orientation || output.clockwise);
    const needsCompactness = Boolean(output.polsbyPopper || output.schwartzberg || output.sliver);
    const needsAreas = needsCompactness || Boolean(output.convexity) || Boolean(output.areas);
    const needsPerimeters = needsCompactness || Boolean(output.perimeters);
    const areas =
      output.areas ??
      (needsAreas ? createTransientView(graph, `${id}-areas`, 'float32', featureCount) : undefined);
    const perimeters =
      output.perimeters ??
      (needsPerimeters
        ? createTransientView(graph, `${id}-perimeters`, 'float32', featureCount)
        : undefined);
    const centroids = needsMoments
      ? createTransientView(graph, `${id}-centroids`, 'float32x2', featureCount)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (areas || perimeters || centroids) {
      nodes.push(
        ...new GPUGeometryMeasures({
          spatialContext: this.spatialContext,
          id: `${id}-measures`,
          positions: props.positions,
          geometryType: 'polygons',
          ringOffsets: props.ringOffsets,
          featureOffsets: props.featureOffsets,
          polygonOffsets: props.polygonOffsets,
          output: {lengths: perimeters, areas, centroids}
        }).getCommandNodes(graph)
      );
    }
    const ringInputs = {
      operation: OPERATION,
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      featureCount
    };
    if (needsMoments && centroids) {
      const moments = createTransientView(graph, `${id}-moments`, 'float32x4', featureCount);
      nodes.push(
        createMomentsNode<Parameters>(graph, {
          ...ringInputs,
          id: `${id}-moments`,
          centroids,
          moments
        }),
        createOrientationNode<Parameters>(graph, {
          id: `${id}-orientation`,
          operation: OPERATION,
          featureCount,
          moments,
          elongation: output.elongation,
          orientation: output.orientation,
          clockwise: output.clockwise
        })
      );
    }
    if (needsCompactness && areas && perimeters) {
      nodes.push(
        createCompactnessNode<Parameters>(graph, {
          id: `${id}-compactness`,
          operation: OPERATION,
          featureCount,
          areas,
          perimeters,
          parameters: props.parameters,
          polsbyPopper: output.polsbyPopper,
          schwartzberg: output.schwartzberg,
          sliver: output.sliver
        })
      );
    }
    if (output.convexity && areas) {
      if (this.useMonotoneChainConvexity) {
        nodes.push(
          ...getMonotoneChainConvexityNodes<Parameters>(graph, {
            ...ringInputs,
            id: `${id}-convexity`,
            areas,
            convexities: output.convexity
          })
        );
      } else {
        nodes.push(
          createConvexityNode<Parameters>(graph, {
            ...ringInputs,
            id: `${id}-convexity`,
            areas,
            convexities: output.convexity
          })
        );
      }
    }
    return nodes;
  }
}
