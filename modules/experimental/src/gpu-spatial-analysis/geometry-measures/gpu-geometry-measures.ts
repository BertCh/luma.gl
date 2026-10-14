// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields,
  GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {
  getGPUAreaUnit,
  validateGPUSpatialContext,
  type GPUAreaModel,
  type GPULengthModel,
  type GPUMeasureDescription,
  type GPUSpatialContext,
  getGPUSphereRadius
} from '../../analysis-core/contracts/spatial-context';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {assert} from '@luma.gl/core';
import {COOPERATIVE_RING_ROWS} from './feature-measures-cooperative';
import {
  createFeatureMeasuresNode,
  createGroupReduceNode,
  createGroupSortPrepareNode,
  createMeasuresScatterNode,
  GEOMETRY_STATS_STRIDE,
  type GeometryMeasureColumns,
  type GPUGeometryCoordinateSystem
} from './geometry-measures-kernels';

const OPERATION = 'GPUGeometryMeasures';

/**
 * Per-feature or per-group measurement columns of {@link GPUGeometryMeasures}. Every column is
 * optional and has one row per feature (or group).
 */
export type GPUGeometryMeasuresOutput = {
  /** Line length, or polygon perimeter over every ring including closing edges. */
  lengths?: GraphDataView<'float32'>;
  /**
   * Polygon area, the sum of the shell areas minus the sum of the hole areas. Ring role is
   * positional (the first ring of each polygon is the shell and later rings are holes, whatever
   * their winding), so it is never negative. Polygons only.
   */
  areas?: GraphDataView<'float32'>;
  /** Area-weighted (polygons) or length-weighted (lines) centroid. */
  centroids?: GraphDataView<'float32x2'>;
  /** `[minX, minY, maxX, maxY]`. Geographic longitudes are unwrapped from the first vertex. */
  bounds?: GraphDataView<'float32x4'>;
  /** Number of vertex rows. */
  vertexCounts?: GraphDataView<'uint32'>;
  /**
   * Per-feature extreme vertices, `[argMinX, argMinY, argMaxX, argMaxY]` as global rows of
   * `positions` (the vertex holding the smallest x, smallest y, largest x and largest y; the
   * lowest row wins ties; geographic x is the unwrapped longitude of `bounds`). Empty features
   * report `0xffffffff` in every word. Per-feature only.
   */
  extremeVertices?: GraphDataView<'uint32x4'>;
  /**
   * Optional status. `invalidCount` receives the exact number of polygon features rejected because
   * a ring encloses a pole or has an edge spanning 180 degrees of longitude (geographic contexts).
   * Rejected features report NaN `areas` and `centroids`; lengths, bounds and
   * vertex counts stay valid. Written every encoding.
   */
  status?: {invalidCount?: GPUStatusScalar};
};

/** Per-group outputs of {@link GPUGeometryMeasures}; `groupCount` rows each. */
export type GPUGeometryMeasuresGroupOutput = Omit<
  GPUGeometryMeasuresOutput,
  'extremeVertices' | 'status'
> & {
  /** Number of features with this group ID. */
  featureCounts?: GraphDataView<'uint32'>;
};

/** Props shared by every geometry type of {@link GPUGeometryMeasures}. */
type GPUGeometryMeasuresSharedProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-measures'`. */
  id?: string;
  /** Coordinate, metric and output-unit contract of the geometry. */
  spatialContext: GPUSpatialContext;
  /**
   * Ellipsoidal polygon-edge interpretation, for `metric: 'ellipsoidal'` contexts only. Defaults to
   * `'geodesic'`: edges of 20 km or more follow geodesics, as `pyproj.Geod.geometry_area_perimeter`.
   * `'coordinate-linear'`: straight edges in the equal-area mapping of the WGS84 authalic sphere.
   */
  ellipsoidalEdgeModel?: 'coordinate-linear' | 'geodesic';
  /**
   * Compile-time. A feature owning a ring of more than this many vertex rows is measured by one
   * 64-lane workgroup instead of one invocation (see {@link GPUGeometryMeasures}); every other
   * feature keeps the serial path. Default 512; 0 disables the cooperative path.
   */
  cooperativeRingRows?: number;
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
 * Properties for {@link GPUGeometryMeasures}, discriminated by `geometryType`.
 *
 * - `'polygons'`: the four-level layout of {@link GPUPolygonGeometryFields}, one row per feature.
 *   Ring role is positional: the first ring of every polygon is a shell and later rings are
 *   holes, whatever their winding (a multi-polygon with mixed winding measures correctly).
 * - `'lines'`: {@link GPULineGeometryFields} (one feature per line), or with `featureOffsets`
 *   `featureCount + 1` monotonic line offsets so a feature owns several lines (multi-lines).
 * - `'points'`: `positions` only (one feature per point), or with `featureOffsets`
 *   `featureCount + 1` monotonic point offsets so a feature owns several points (multi-points,
 *   centroid = vertex mean as shapely's `MultiPoint.centroid`; lengths are zero and `areas` is
 *   unavailable).
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * feature and group counts, `geometryType`, `spatialContext`, `ellipsoidalEdgeModel`, and which
 * outputs are present.
 */
export type GPUGeometryMeasuresProps = GPUGeometryMeasuresSharedProps &
  (
    | ({geometryType: 'polygons'} & GPUPolygonGeometryFields)
    | ({geometryType: 'lines'} & GPULineGeometryFields & {
          /** Optional `featureCount + 1` monotonic line offsets (multi-lines). */
          featureOffsets?: GraphDataView<'uint32'>;
        })
    | {
        geometryType: 'points';
        /** Packed point positions. */
        positions: GraphDataView<'float32x2'>;
        /** Optional `featureCount + 1` monotonic point offsets (multi-points). */
        featureOffsets?: GraphDataView<'uint32'>;
      }
  );

/**
 * Measures many line or polygon features at once: length or perimeter, area, signed area,
 * centroid, bounds and vertex count per feature, and optionally per group (turf `area`, `length`,
 * `centroid`/`centerOfMass`, `bbox`; PostGIS `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid`,
 * `ST_Envelope`, `ST_NPoints`), plus per-feature extreme vertices (argmin and argmax of x and y).
 *
 * **Features.** One invocation per feature walks its rings in row order with Neumaier-compensated
 * f32 sums in coordinates relative to the feature's first vertex, so coordinates far from the
 * origin (projected meters, Web Mercator) keep their small-scale precision, and results are
 * deterministic. Latency grows with the largest feature's vertex count.
 *
 * **Large rings.** A feature that owns a ring of more than `cooperativeRingRows` (default 512)
 * vertices is measured by one 64-lane workgroup: each lane runs the same loop over a contiguous
 * chunk of the ring (geographic longitudes unwrapped by an exclusive scan of the chunks' steps)
 * and the lane partials are merged in lane order with Neumaier addition. A million-vertex ring
 * then costs a chain of 16k vertices instead of 1M. Such features can differ from the serial
 * walk by f32 summation order only; all other features are unchanged bit for bit.
 *
 * **Geographic modes.** Areas are shoelace areas in the Lambert cylindrical equal-area projection
 * (`x` = longitude in radians, `y` = sine of latitude), the Chamberlain-Duquette formula turf uses:
 * exact for longitude/latitude boxes, and edges are straight in that projection rather than
 * geodesics, which matters only for long edges. `'wgs84'` replaces latitude by authalic latitude
 * and the radius by the authalic radius (exact equal-area mapping of the ellipsoid). Polygon
 * centroids are area-weighted in the same projection; line centroids are length-weighted
 * longitude/latitude midpoints. Rings that enclose a pole or span 180 degrees of longitude per
 * edge have no well-defined interior: they are detected (see **Invalid geographic polygons**),
 * not measured.
 *
 * **Invalid geographic polygons.** Longitude steps are wrapped per edge, so a ring that crosses
 * the antimeridian (an edge with a longitude difference above 180 degrees, such as Fiji's
 * 179 to -179) is measured correctly. A ring whose wrapped steps sum to +-360 degrees encloses a
 * pole, and an edge whose wrapped step reaches 180 degrees (or whose raw step is a full 360)
 * is ambiguous; features owning such a ring report NaN area and centroid and are counted in
 * `output.status.invalidCount`. Split such polygons at the antimeridian or project them first.
 *
 * **Measure model.** {@link GPUGeometryMeasures.measures} describes the length and area models and
 * units of the outputs (planar: native units; geographic: meters and square meters unless the
 * context asks for kilometers on a sphere).
 *
 * **Groups.** Features are stably sorted by group ID; one workgroup per group reduces its
 * features with a fixed tree (bitwise reproducible). Group centroids are the area- or
 * length-weighted mean of feature centroids. Geographic group centroids are the weighted mean of
 * the feature centroids as 3D unit vectors (a spherical centroid, correct across the
 * antimeridian); an antipodal group with no mean direction reports NaN. Planar groups are
 * unchanged. A group that contains an invalid feature reports NaN area and centroid.
 *
 * **Geodesic areas.** An `'ellipsoidal'` context with the default `ellipsoidalEdgeModel` replaces every edge of 20 km or more by its
 * geodesic, sampled with Vincenty's direct solution at the ends and midpoint of spans of at most
 * 250 km (at most 16 spans) and summed as a chord plus its exact parabolic-segment bulge, in the
 * same equal-area coordinates (local origin) as `'wgs84'`, so the f32 sum never subtracts two
 * large areas to the equator. Against GeographicLib (`pyproj.Geod.geometry_area_perimeter`) this
 * measured within 1e-6 relative on regional and continental polygons (straight equal-area edges are off by up to 30% on the longest); edges shorter than 20 km keep straight
 * equal-area edges (their bulge is below f32 position noise). Perimeter is the Vincenty length.
 *
 * **Points.** `geometryType: 'points'` reports vertex counts, bounds, extreme vertices and the
 * vertex-mean centroid; group centroids weight features by vertex count, so they equal the mean
 * of all member points.
 *
 * Empty features report zero length/area, zero vertices and NaN centroid and bounds.
 */
export class GPUGeometryMeasures implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryMeasuresProps;
  /** Canonical coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features. */
  readonly featureCount: number;
  /** Machine-readable length and area models and units of the outputs. */
  readonly measures: GPUMeasureDescription;
  /** Internal kernel model resolved from `spatialContext`; not a prop. */
  readonly coordinateSystem: GPUGeometryCoordinateSystem;
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Resolved ring size above which a feature is measured cooperatively (0 disables). */
  readonly cooperativeRingRows: number;

  constructor(props: GPUGeometryMeasuresProps) {
    this.id = props.id ?? 'geometry-measures';
    this.props = props;
    this.spatialContext = props.spatialContext;
    validateGPUSpatialContext(this.id, this.spatialContext);
    this.coordinateSystem = resolveGeometryMeasureCoordinateSystem(
      this.id,
      this.spatialContext,
      props.ellipsoidalEdgeModel
    );
    this.measures = describeGeometryMeasures(this.coordinateSystem, this.spatialContext);
    this.radius = getGPUSphereRadius(this.spatialContext);
    this.cooperativeRingRows = props.cooperativeRingRows ?? COOPERATIVE_RING_ROWS;
    const {id} = this;
    const {status, ...output} = props.output ?? {};
    const groupOutput = props.groupOutput ?? {};
    if (!['lines', 'polygons', 'points'].includes(props.geometryType)) {
      throw new Error(`${id} geometryType must be 'lines', 'polygons' or 'points'`);
    }
    validateCapacity(id, 'cooperativeRingRows', this.cooperativeRingRows, 0);
    // The radius comes from a validated spatial context.
    assert(Number.isFinite(this.radius) && this.radius > 0);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    const layout = getMeasuredLayout(props);
    for (const name of ['ringOffsets', 'featureOffsets', 'polygonOffsets'] as const) {
      const view = layout[name];
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 2) {
          throw new Error(`${id} ${name} must contain at least two rows`);
        }
      }
    }
    this.featureCount = layout.featureOffsets
      ? layout.featureOffsets.length - 1
      : layout.ringOffsets
        ? layout.ringOffsets.length - 1
        : props.positions.length;
    const isPolygon = props.geometryType === 'polygons';
    validateMeasureColumns(id, 'output', output, this.featureCount, isPolygon);
    const hasGroupOutput = Object.values(groupOutput).some(Boolean);
    if (props.groupIds) {
      validatePackedView(props.groupIds, ['uint32'], `${id} groupIds`);
      if (props.groupIds.length !== this.featureCount) {
        throw new Error(`${id} groupIds length must equal the feature count`);
      }
      validateCapacity(id, 'groupCount', props.groupCount ?? 0);
      if (!hasGroupOutput) {
        throw new Error(`${id} groupIds require at least one groupOutput column`);
      }
      validateMeasureColumns(id, 'groupOutput', groupOutput, props.groupCount ?? 0, isPolygon);
    } else if (hasGroupOutput) {
      throw new Error(`${id} groupOutput requires groupIds`);
    }
    const invalidCount = status?.invalidCount;
    if (invalidCount) {
      validatePackedView(invalidCount, ['uint32'], `${id} output.status.invalidCount`);
      if (invalidCount.length < 1) {
        throw new Error(`${id} output.status.invalidCount must contain one uint32 row`);
      }
    }
    if (!Object.values(output).some(Boolean) && !hasGroupOutput) {
      throw new Error(`${id} requires at least one output column`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [...Object.values(output), ...Object.values(groupOutput), invalidCount],
      [
        props.positions,
        layout.ringOffsets,
        layout.featureOffsets,
        layout.polygonOffsets,
        props.groupIds
      ]
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
    const {status, ...output} = props.output ?? {};
    const groupOutput = props.groupOutput ?? {};
    const layout = getMeasuredLayout(props);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      layout.ringOffsets,
      layout.featureOffsets,
      layout.polygonOffsets,
      props.groupIds,
      status?.invalidCount,
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
    const featureNodeProps = {
      id: `${id}-features`,
      operation: OPERATION,
      positions: props.positions,
      ringOffsets: layout.ringOffsets,
      featureOffsets: layout.featureOffsets,
      polygonOffsets: layout.polygonOffsets,
      featureCount,
      isPolygon,
      isPoint: props.geometryType === 'points',
      coordinateSystem: this.coordinateSystem,
      radius: this.radius,
      featureStats
    };
    const invalidCount = status?.invalidCount;
    // Rings of at most the threshold can never make a feature cooperative.
    const hasCooperativeFeatures =
      this.cooperativeRingRows > 0 && props.positions.length > this.cooperativeRingRows;
    const nodes: GPUCommandNode<Parameters>[] = [
      createFeatureMeasuresNode<Parameters>(graph, {
        ...featureNodeProps,
        cooperativeRingRows: hasCooperativeFeatures ? this.cooperativeRingRows : 0
      })
    ];
    if (hasCooperativeFeatures) {
      nodes.push(
        createFeatureMeasuresNode<Parameters>(graph, {
          ...featureNodeProps,
          id: `${id}-features-cooperative`,
          cooperative: true,
          cooperativeRingRows: this.cooperativeRingRows
        })
      );
    }
    if (invalidCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-invalid-count`,
          operation: OPERATION,
          view: invalidCount,
          type: 'u32',
          value: '0u'
        }),
        createInvalidCountNode<Parameters>(graph, id, featureCount, featureStats, invalidCount)
      );
    }
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
      createSortedSegmentOffsetsNode<Parameters>(graph, {
        id: `${id}-group-offsets`,
        operation: OPERATION,
        variant: 'group-offsets',
        segmentCount: groupCount,
        sortedKeys,
        segmentOffsets: groupOffsets
      }),
      createGroupReduceNode<Parameters>(graph, {
        id: `${id}-group-reduce`,
        operation: OPERATION,
        isPolygon: this.props.geometryType === 'polygons',
        isPoint: this.props.geometryType === 'points',
        isGeographic: this.coordinateSystem !== 'planar',
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

/**
 * Maps the public layout to the kernel's ring layout: polygons keep the four levels; lines use
 * `lineOffsets` as ring offsets and `featureOffsets` as feature-to-ring; a point set is one ring
 * spanning its points, so `featureOffsets` doubles as the ring offsets.
 */
function getMeasuredLayout(props: GPUGeometryMeasuresProps): {
  ringOffsets?: GraphDataView<'uint32'>;
  featureOffsets?: GraphDataView<'uint32'>;
  polygonOffsets?: GraphDataView<'uint32'>;
} {
  switch (props.geometryType) {
    case 'polygons':
      return {
        ringOffsets: props.ringOffsets,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets
      };
    case 'lines':
      return {ringOffsets: props.lineOffsets, featureOffsets: props.featureOffsets};
    default:
      return {ringOffsets: props.featureOffsets};
  }
}

function describeGeometryMeasures(
  coordinateSystem: GPUGeometryCoordinateSystem,
  spatialContext: GPUSpatialContext
): GPUMeasureDescription {
  const areaModels: Record<GPUGeometryCoordinateSystem, GPUAreaModel> = {
    planar: 'planar',
    spherical: 'sphere-equal-area',
    wgs84: 'wgs84-authalic',
    geodesic: 'geodesic'
  };
  const lengthModels: Record<GPUGeometryCoordinateSystem, GPULengthModel> = {
    planar: 'planar',
    spherical: 'great-circle',
    wgs84: 'ellipsoidal-geodesic',
    geodesic: 'ellipsoidal-geodesic'
  };
  const lengthUnits = coordinateSystem === 'planar' ? 'native' : (spatialContext.units ?? 'meters');
  return {
    lengthModel: lengthModels[coordinateSystem],
    areaModel: areaModels[coordinateSystem],
    lengthUnits,
    areaUnits: getGPUAreaUnit(lengthUnits)
  };
}

function createInvalidCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  featureCount: number,
  featureStats: GraphDataView<'uint32'>,
  invalidCount: GPUStatusScalar
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'featureStats', view: featureStats, type: 'u32', access: 'read'},
    {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
  ];
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-invalid-count`,
    operation: OPERATION,
    variant: 'invalid-count',
    bindings,
    invocationCount: featureCount,
    declarations: `const STATS_STRIDE: u32 = ${GEOMETRY_STATS_STRIDE}u;`,
    body: `if (featureStats[featureStatsOffset + index * STATS_STRIDE + 15u] > 0u) {
    atomicAdd(&invalidCount[invalidCountOffset], 1u);
  }`
  });
}

function resolveGeometryMeasureCoordinateSystem(
  id: string,
  spatialContext: GPUSpatialContext,
  ellipsoidalEdgeModel: 'coordinate-linear' | 'geodesic' | undefined
): GPUGeometryCoordinateSystem {
  if (
    spatialContext.coordinateSpace === 'planar' &&
    (spatialContext.metric === 'native' || spatialContext.metric === 'none')
  ) {
    if (ellipsoidalEdgeModel) {
      throw new Error(`${id} ellipsoidalEdgeModel requires an ellipsoidal spatial context`);
    }
    return 'planar';
  }
  if (
    spatialContext.coordinateSpace === 'longitude-latitude' &&
    spatialContext.metric === 'great-circle'
  ) {
    if (ellipsoidalEdgeModel) {
      throw new Error(`${id} ellipsoidalEdgeModel requires an ellipsoidal spatial context`);
    }
    return 'spherical';
  }
  if (
    spatialContext.coordinateSpace === 'longitude-latitude' &&
    spatialContext.metric === 'ellipsoidal'
  ) {
    if (spatialContext.units === 'kilometers') {
      throw new Error(`${id} ellipsoidal geometry measures currently publish meters`);
    }
    return ellipsoidalEdgeModel === 'coordinate-linear' ? 'wgs84' : 'geodesic';
  }
  throw new Error(`${id} spatial context is not supported by geometry measures`);
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
    centroids: 'float32x2',
    bounds: 'float32x4',
    vertexCounts: 'uint32',
    extremeVertices: 'uint32x4',
    featureCounts: 'uint32'
  } as const;
  for (const [column, view] of Object.entries(columns) as [
    keyof typeof formats,
    GraphDataView | undefined
  ][]) {
    if (!view) {
      continue;
    }
    if (!(column in formats) || (name === 'groupOutput' && column === 'extremeVertices')) {
      throw new Error(`${id} ${name}.${column} is not a measure column`);
    }
    validatePackedView(view, [formats[column]], `${id} ${name}.${column}`);
    if (view.length !== rowCount) {
      throw new Error(`${id} ${name}.${column} must hold ${rowCount} rows`);
    }
    if (!isPolygon && column === 'areas') {
      throw new Error(`${id} ${name}.${column} requires geometryType 'polygons'`);
    }
  }
}
