// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
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
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {GPU_GEODESIC_MEAN_EARTH_RADIUS} from './geodesic-wgsl';
import {
  createFeatureMeasuresNode,
  createGroupOffsetsNode,
  createGroupReduceNode,
  createGroupSortPrepareNode,
  createMeasuresScatterNode,
  GEOMETRY_STATS_STRIDE,
  type GeometryMeasureColumns,
  type GPUGeometryCoordinateSystem,
  type GPUGeometryHoleRule
} from './geometry-measures-kernels';

const OPERATION = 'GPUGeometryMeasures';

/**
 * Per-feature or per-group measurement columns of {@link GPUGeometryMeasures}. Every column is
 * optional and has one row per feature (or group).
 */
export type GPUGeometryMeasuresOutput = {
  /** Line length, or polygon perimeter over every ring including closing edges. */
  lengths?: GraphDataView<'float32'>;
  /** Polygon area, `|signedArea|`. Polygons only. */
  areas?: GraphDataView<'float32'>;
  /**
   * Polygon signed area, positive for counter-clockwise exteriors (in longitude/latitude order for
   * geographic coordinates). With `holeRule: 'first-ring-exterior'` it is always non-negative.
   * Polygons only.
   */
  signedAreas?: GraphDataView<'float32'>;
  /** Area-weighted (polygons) or length-weighted (lines) centroid. */
  centroids?: GraphDataView<'float32x2'>;
  /** `[minX, minY, maxX, maxY]`. Geographic longitudes are unwrapped from the first vertex. */
  bounds?: GraphDataView<'float32x4'>;
  /** Number of vertex rows. */
  vertexCounts?: GraphDataView<'uint32'>;
};

/** Per-group outputs of {@link GPUGeometryMeasures}; `groupCount` rows each. */
export type GPUGeometryMeasuresGroupOutput = GPUGeometryMeasuresOutput & {
  /** Number of features with this group ID. */
  featureCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGeometryMeasures}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * feature and group counts, `geometryType`, `coordinateSystem`, `radius`, `holeRule`, and which
 * outputs are present.
 */
export type GPUGeometryMeasuresProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-measures'`. */
  id?: string;
  /**
   * Packed vertex positions: planar coordinates, or longitude/latitude degrees for `'spherical'`
   * and `'wgs84'`.
   */
  positions: GraphDataView<'float32x2'>;
  /** `'lines'` (paths) or `'polygons'` (closed rings). */
  geometryType: 'lines' | 'polygons';
  /**
   * `ringCount + 1` monotonic vertex offsets; ring (or path) `r` owns rows
   * `[ringOffsets[r], ringOffsets[r + 1])`. Rings close implicitly; a repeated closing vertex just
   * adds a zero-length edge.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Optional `featureCount + 1` monotonic ring offsets; feature `f` owns rings
   * `[featureRingOffsets[f], featureRingOffsets[f + 1])` (multi-polygons, polygons with holes,
   * multi-lines). When omitted every ring is its own feature.
   */
  featureRingOffsets?: GraphDataView<'uint32'>;
  /**
   * `'planar'` (default): Euclidean lengths and shoelace areas in position units.
   * `'spherical'`: haversine lengths and Chamberlain-Duquette areas on a sphere of `radius`.
   * `'wgs84'`: Vincenty edge lengths on the WGS84 ellipsoid (Lambert's formula where Vincenty does
   * not converge) and areas on the WGS84 authalic sphere.
   */
  coordinateSystem?: GPUGeometryCoordinateSystem;
  /**
   * Sphere radius of `'spherical'`. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}; pass
   * `GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS` for turf `area` parity.
   */
  radius?: number;
  /**
   * How rings of one polygon feature combine. `'winding'` (default) sums signed ring areas, so
   * holes subtract when they wind opposite to the exterior (GeoJSON RFC 7946). `'first-ring-exterior'`
   * counts the first ring of each feature positive and every other ring as a hole, whatever the
   * winding (one polygon per feature).
   */
  holeRule?: GPUGeometryHoleRule;
  /** Per-feature outputs. */
  output?: GPUGeometryMeasuresOutput;
  /** Optional per-feature group IDs; IDs at or above `groupCount` are ignored. */
  groupIds?: GraphDataView<'uint32'>;
  /** Number of groups. Required with `groupIds`. */
  groupCount?: number;
  /** Per-group outputs. Requires `groupIds`. */
  groupOutput?: GPUGeometryMeasuresGroupOutput;
};

/**
 * Measures many line or polygon features at once: length or perimeter, area, signed area,
 * centroid, bounds and vertex count per feature, and optionally per group (turf `area`, `length`,
 * `centroid`/`centerOfMass`, `bbox`; PostGIS `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid`,
 * `ST_Envelope`, `ST_NPoints`).
 *
 * **Features.** One invocation per feature walks its rings in row order with Neumaier-compensated
 * f32 sums in coordinates relative to the feature's first vertex, so coordinates far from the
 * origin (projected meters, Web Mercator) keep their small-scale precision, and results are
 * deterministic. Latency grows with the largest feature's vertex count.
 *
 * **Geographic modes.** Areas are shoelace areas in the Lambert cylindrical equal-area projection
 * (`x` = longitude in radians, `y` = sine of latitude), the Chamberlain-Duquette formula turf uses:
 * exact for longitude/latitude boxes, and edges are straight in that projection rather than
 * geodesics, which matters only for long edges. `'wgs84'` replaces latitude by authalic latitude
 * and the radius by the authalic radius (exact equal-area mapping of the ellipsoid). Polygon
 * centroids are area-weighted in the same projection; line centroids are length-weighted
 * longitude/latitude midpoints. Rings that enclose a pole or span 180 degrees of longitude per
 * edge are not supported.
 *
 * **Groups.** Features are stably sorted by group ID; one workgroup per group reduces its
 * features with a fixed tree (bitwise reproducible). Group centroids are the area- or
 * length-weighted mean of feature centroids; geographic group centroids average longitudes
 * naively and are wrong for groups straddling the antimeridian.
 *
 * Empty features report zero length/area, zero vertices and NaN centroid and bounds.
 */
export class GPUGeometryMeasures implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryMeasuresProps;
  /** Number of features. */
  readonly featureCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: GPUGeometryCoordinateSystem;
  /** Resolved hole rule. */
  readonly holeRule: GPUGeometryHoleRule;
  /** Resolved sphere radius. */
  readonly radius: number;

  constructor(props: GPUGeometryMeasuresProps) {
    this.id = props.id ?? 'geometry-measures';
    this.props = props;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.holeRule = props.holeRule ?? 'winding';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    const {id} = this;
    const output = props.output ?? {};
    const groupOutput = props.groupOutput ?? {};
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureRingOffsets: props.featureRingOffsets,
      groupIds: props.groupIds,
      ...output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (props.geometryType !== 'lines' && props.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
    if (!['planar', 'spherical', 'wgs84'].includes(this.coordinateSystem)) {
      throw new Error(`${id} coordinateSystem must be 'planar', 'spherical' or 'wgs84'`);
    }
    if (this.holeRule !== 'winding' && this.holeRule !== 'first-ring-exterior') {
      throw new Error(`${id} holeRule must be 'winding' or 'first-ring-exterior'`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`${id} radius must be a positive finite number`);
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
    const isPolygon = props.geometryType === 'polygons';
    validateMeasureColumns(id, 'output', output, this.featureCount, isPolygon);
    const hasGroupOutput = Object.values(groupOutput).some(Boolean);
    if (props.groupIds) {
      validatePackedUint32View(props.groupIds, `${id} groupIds`);
      if (props.groupIds.length !== this.featureCount) {
        throw new Error(`${id} groupIds length must equal the feature count`);
      }
      if (!Number.isSafeInteger(props.groupCount) || (props.groupCount ?? 0) < 1) {
        throw new Error(`${id} groupCount must be a positive integer with groupIds`);
      }
      if (!hasGroupOutput) {
        throw new Error(`${id} groupIds require at least one groupOutput column`);
      }
      validateMeasureColumns(id, 'groupOutput', groupOutput, props.groupCount ?? 0, isPolygon);
    } else if (hasGroupOutput) {
      throw new Error(`${id} groupOutput requires groupIds`);
    }
    if (!Object.values(output).some(Boolean) && !hasGroupOutput) {
      throw new Error(`${id} requires at least one output column`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [...Object.values(output), ...Object.values(groupOutput)],
      [props.positions, props.ringOffsets, props.featureRingOffsets, props.groupIds]
    );
  }

  /**
   * Returns the feature node and feature scatter node, then with groups the sort, offsets, reduce
   * and group scatter nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, featureCount} = this;
    const output = props.output ?? {};
    const groupOutput = props.groupOutput ?? {};
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.featureRingOffsets,
      props.groupIds,
      ...Object.values(output),
      ...Object.values(groupOutput)
    ]);
    const isPolygon = props.geometryType === 'polygons';
    const featureStats = createTransientView(
      graph,
      `${id}-feature-stats`,
      'uint32',
      featureCount * GEOMETRY_STATS_STRIDE
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createFeatureMeasuresNode<Parameters>(graph, {
        id: `${id}-features`,
        operation: OPERATION,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        featureRingOffsets: props.featureRingOffsets,
        featureCount,
        isPolygon,
        holeRule: this.holeRule,
        coordinateSystem: this.coordinateSystem,
        radius: this.radius,
        featureStats
      })
    ];
    if (Object.values(output).some(Boolean)) {
      nodes.push(
        createMeasuresScatterNode<Parameters>(graph, {
          id: `${id}-feature-output`,
          operation: OPERATION,
          rowCount: featureCount,
          stats: featureStats,
          columns: output
        })
      );
    }
    if (props.groupIds && props.groupCount) {
      nodes.push(...this._getGroupNodes(graph, props.groupIds, props.groupCount, featureStats));
    }
    return nodes;
  }

  private _getGroupNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    groupIds: GraphDataView<'uint32'>,
    groupCount: number,
    featureStats: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, featureCount} = this;
    const sortKeys = createTransientView(graph, `${id}-group-keys`, 'uint32', featureCount);
    const sortValues = createTransientView(graph, `${id}-group-values`, 'uint32', featureCount);
    const sortedKeys = createTransientView(
      graph,
      `${id}-group-sorted-keys`,
      'uint32',
      featureCount
    );
    const sortedIndices = createTransientView(
      graph,
      `${id}-group-sorted-indices`,
      'uint32',
      featureCount
    );
    const groupOffsets = createTransientView(
      graph,
      `${id}-group-offsets`,
      'uint32',
      groupCount + 1
    );
    const groupStats = createTransientView(
      graph,
      `${id}-group-stats`,
      'uint32',
      groupCount * GEOMETRY_STATS_STRIDE
    );
    return [
      createGroupSortPrepareNode<Parameters>(graph, {
        id: `${id}-group-sort-prepare`,
        operation: OPERATION,
        groupIds,
        groupCount,
        sortKeys,
        sortValues
      }),
      ...new GPUSort({
        id: `${id}-group-sort`,
        keys: sortKeys,
        values: sortValues,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(groupCount)
      }).getCommandNodes(graph),
      createGroupOffsetsNode<Parameters>(graph, {
        id: `${id}-group-offsets`,
        operation: OPERATION,
        sortedKeys,
        groupCount,
        groupOffsets
      }),
      createGroupReduceNode<Parameters>(graph, {
        id: `${id}-group-reduce`,
        operation: OPERATION,
        isPolygon: this.props.geometryType === 'polygons',
        groupCount,
        sortedIndices,
        groupOffsets,
        featureStats,
        groupStats
      }),
      createMeasuresScatterNode<Parameters>(graph, {
        id: `${id}-group-output`,
        operation: OPERATION,
        rowCount: groupCount,
        stats: groupStats,
        columns: this.props.groupOutput as GeometryMeasureColumns
      })
    ];
  }
}

function validateMeasureColumns(
  id: string,
  name: string,
  columns: GPUGeometryMeasuresGroupOutput,
  rowCount: number,
  isPolygon: boolean
): void {
  const formats = {
    lengths: 'float32',
    areas: 'float32',
    signedAreas: 'float32',
    centroids: 'float32x2',
    bounds: 'float32x4',
    vertexCounts: 'uint32',
    featureCounts: 'uint32'
  } as const;
  for (const [column, view] of Object.entries(columns) as [
    keyof typeof formats,
    GraphDataView | undefined
  ][]) {
    if (!view) {
      continue;
    }
    if (!(column in formats)) {
      throw new Error(`${id} ${name}.${column} is not a measure column`);
    }
    validatePackedView(view, [formats[column]], `${id} ${name}.${column}`);
    if (view.length !== rowCount) {
      throw new Error(`${id} ${name}.${column} must hold ${rowCount} rows`);
    }
    if (!isPolygon && (column === 'areas' || column === 'signedAreas')) {
      throw new Error(`${id} ${name}.${column} requires geometryType 'polygons'`);
    }
  }
}
