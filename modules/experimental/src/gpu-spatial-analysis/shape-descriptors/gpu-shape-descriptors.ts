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
import {GPUGeometryMeasures, type GPUGeometryHoleRule} from '../geometry-measures/index';
import {
  createCompactnessNode,
  createConvexityNode,
  createMomentsNode,
  createOrientationNode,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH
} from './shape-descriptors-kernels';

const OPERATION = 'GPUShapeDescriptors';

export {GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH};

/** Default sliver threshold: Polsby-Popper below this flags a sliver (about a 1:60 rectangle). */
export const GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD = 0.05;

/** CPU description of the per-frame parameters of {@link GPUShapeDescriptors}. */
export type GPUShapeDescriptorsParameters = {
  /**
   * Polsby-Popper compactness below which a polygon is flagged as a sliver. Default
   * {@link GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD}.
   */
  sliverThreshold?: number;
};

/**
 * Packs {@link GPUShapeDescriptorsParameters} into the 4-element float32 layout
 * `[sliverThreshold, 0, 0, 0]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the threshold without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the threshold is not a non-negative finite number, or `target` is too short.
 */
export function getGPUShapeDescriptorsParameterValues(
  parameters: GPUShapeDescriptorsParameters = {},
  target: Float32Array = new Float32Array(GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
    throw new Error(
      `Shape descriptors target must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} elements`
    );
  }
  const sliverThreshold =
    parameters.sliverThreshold ?? GPU_SHAPE_DESCRIPTORS_DEFAULT_SLIVER_THRESHOLD;
  if (!Number.isFinite(sliverThreshold) || sliverThreshold < 0) {
    throw new Error('Shape descriptors sliverThreshold must be a non-negative finite number');
  }
  target.set([sliverThreshold, 0, 0, 0]);
  return target;
}

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
 * buffer. Compile-time: view lengths, `holeRule`, and which outputs are present.
 */
export type GPUShapeDescriptorsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'shape-descriptors'`. */
  id?: string;
  /** Packed vertex positions in a planar (projected) coordinate system. */
  positions: GraphDataView<'float32x2'>;
  /** `ringCount + 1` monotonic vertex offsets, as in `GPUGeometryMeasures`. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Optional `featureCount + 1` monotonic ring offsets. When omitted every ring is its own feature.
   */
  featureRingOffsets?: GraphDataView<'uint32'>;
  /** How rings combine; see `GPUGeometryMeasures`. Default `'winding'`. */
  holeRule?: GPUGeometryHoleRule;
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
 * feature (a shared per-label hull from `GPUGroupConvexHull` can replace it when available).
 * Orientation and elongation weight holes by the hole rule, so a polygon with a large hole is
 * described by its area, not its outline. Use planar coordinates: for longitude/latitude input,
 * project first (geographic descriptors are open).
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
  /** Number of features. */
  readonly featureCount: number;
  /** Resolved hole rule. */
  readonly holeRule: GPUGeometryHoleRule;

  constructor(props: GPUShapeDescriptorsProps) {
    this.id = props.id ?? 'shape-descriptors';
    this.props = props;
    this.holeRule = props.holeRule ?? 'winding';
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureRingOffsets: props.featureRingOffsets,
      parameters: props.parameters,
      ...props.output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (this.holeRule !== 'winding' && this.holeRule !== 'first-ring-exterior') {
      throw new Error(`${id} holeRule must be 'winding' or 'first-ring-exterior'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must contain at least two rows`);
    }
    if (props.featureRingOffsets) {
      validatePackedUint32View(props.featureRingOffsets, `${id} featureRingOffsets`);
      if (props.featureRingOffsets.length < 2) {
        throw new Error(`${id} featureRingOffsets must contain at least two rows`);
      }
    }
    this.featureCount = props.featureRingOffsets
      ? props.featureRingOffsets.length - 1
      : props.ringOffsets.length - 1;
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
      [props.positions, props.ringOffsets, props.featureRingOffsets, props.parameters]
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
      props.featureRingOffsets,
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
          id: `${id}-measures`,
          positions: props.positions,
          geometryType: 'polygons',
          ringOffsets: props.ringOffsets,
          featureRingOffsets: props.featureRingOffsets,
          holeRule: this.holeRule,
          output: {lengths: perimeters, areas, centroids}
        }).getCommandNodes(graph)
      );
    }
    const ringInputs = {
      operation: OPERATION,
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureRingOffsets: props.featureRingOffsets,
      featureCount
    };
    if (needsMoments && centroids) {
      const moments = createTransientView(graph, `${id}-moments`, 'float32x4', featureCount);
      nodes.push(
        createMomentsNode<Parameters>(graph, {
          ...ringInputs,
          id: `${id}-moments`,
          holeRule: this.holeRule,
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
      nodes.push(
        createConvexityNode<Parameters>(graph, {
          ...ringInputs,
          id: `${id}-convexity`,
          areas,
          convexities: output.convexity
        })
      );
    }
    return nodes;
  }
}
