// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPUSegmentIntersection, type GPUSegmentGeometry} from '../segment-intersection/index';
import {SEGMENT_PREDICATES_WGSL} from '../segment-intersection/segment-intersection-wgsl';
import type {GPUSpatialJoinGeometry} from '../spatial-join/index';
import {
  getPredicateFeatureCount,
  getPredicateGeometryBindings,
  getPredicateGeometryViews,
  getPredicateGeometryWGSL
} from './geometry-predicates-access';
import {GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH} from './geometry-predicates-parameters';

const OPERATION = 'GPUGeometryPredicates';

/** Value written to the pair layout count when two features differ in structure. */
const STRUCTURE_MISMATCH = 0xffffffff;

/**
 * Properties for {@link GPUGeometryPredicates}.
 *
 * Per-frame: the contents of every input buffer and of `parameters` (the `equalsExact`
 * tolerance). Topology: view lengths, geometry kinds, `intersectionCapacity`, `leafCapacity`, and
 * which optional output views exist.
 */
export type GPUGeometryPredicatesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geometry-predicates'`. */
  id?: string;
  /**
   * Point, linestring or polygon/multipolygon features in the GeoArrow layout of
   * `GPUSpatialPredicateJoin`. Each output column has one `uint32` row per feature, holding `0` or
   * `1` for the boolean columns.
   */
  geometry: GPUSpatialJoinGeometry;
  /**
   * Shapely `is_simple`: 1 when the feature has no self-intersection. Linestrings use
   * {@link GPUSegmentIntersection} in self mode (a closed linestring may meet itself only at its
   * ends). Polygons are simple when no ring crosses or touches itself; rings that meet each other
   * are not checked (`GPUGeometryValidity` does that). Points are always simple. Needs `overflow`
   * and `intersectionCapacity`.
   */
  isSimple?: GraphDataView<'uint32'>;
  /** Shapely `is_ring` for linestrings: closed and simple. Always 0 for points and polygons. */
  isRing?: GraphDataView<'uint32'>;
  /** Shapely `is_closed` for linestrings: two or more vertices, the last equal to the first. Always 0 otherwise. */
  isClosed?: GraphDataView<'uint32'>;
  /**
   * Shapely `is_ccw` for linestrings, evaluated like GEOS 3.13 `Orientation::isCCW` on the vertex
   * list as stored (the last vertex is taken as the ring closure, so an unclosed line is judged
   * without its last vertex). Always 0 for points and polygons, like Shapely.
   */
  isCcw?: GraphDataView<'uint32'>;
  /** Shapely `get_num_points`: vertex count of a linestring, `0` for points and polygons. */
  numPoints?: GraphDataView<'uint32'>;
  /** Shapely `count_coordinates` per feature: stored vertex count, including closing duplicates. */
  numCoordinates?: GraphDataView<'uint32'>;
  /**
   * Shapely `get_num_interior_rings`: holes of a feature with exactly one polygon; `0` for
   * multipolygons (as Shapely), points and linestrings.
   */
  numInteriorRings?: GraphDataView<'uint32'>;
  /**
   * Shapely `get_num_geometries`: polygon count of a polygon feature (0 when it has none), `1` for
   * points and linestrings.
   */
  numGeometries?: GraphDataView<'uint32'>;
  /** Capacity of the internal list of same-feature segment intersections. Required with `isSimple` or `isRing`. */
  intersectionCapacity?: number;
  /** One-row flag set to 1 when the intersection list or BVH overflowed; `isSimple` may then be 1 for some non-simple features. Required with `isSimple` or `isRing`. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row unclamped number of same-feature segment intersections, for sizing `intersectionCapacity`. */
  intersectionCount?: GraphDataView<'uint32'>;
  /** Power-of-two BVH leaf slots over segments. Defaults to the next power of two of the vertex count. */
  leafCapacity?: number;
  /**
   * Second geometry for element-wise comparison: feature `i` of `geometry` against feature `i` of
   * `other`. Must have the same kind and feature count. Used by `equalsExact` and `equalsIdentical`.
   */
  other?: GPUSpatialJoinGeometry;
  /**
   * Per-frame packed float32 view of at least {@link GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH}
   * elements written with `getGPUGeometryPredicatesParameterValues`. Required with `equalsExact`.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Shapely `equals_exact(a, b, tolerance)` per feature pair: same structure (polygon, ring and
   * vertex counts) and every vertex pair within the tolerance. NaN coordinates never match. Needs
   * `other` and `parameters`.
   */
  equalsExact?: GraphDataView<'uint32'>;
  /**
   * Shapely `equals_identical` per feature pair: same structure and bit-identical coordinates, where
   * `0` equals `-0` and NaN equals NaN. Needs `other`.
   */
  equalsIdentical?: GraphDataView<'uint32'>;
  /**
   * Shapely `extract_unique_points`: the distinct vertices of every feature in first-occurrence
   * order, compacted. Capacity must be at least the vertex count so nothing can overflow.
   * `-0` and `0` are the same point.
   */
  uniquePositions?: GraphDataView<'float32x2'>;
  /**
   * `featureCount + 1` offsets into `uniquePositions`: feature `i` owns rows
   * `[uniqueOffsets[i], uniqueOffsets[i + 1])`. The last entry is the total unique count.
   * Required with `uniquePositions`.
   */
  uniqueOffsets?: GraphDataView<'uint32'>;
};

/**
 * Per-feature geometry predicates and counts: the GPU analogs of the Shapely 2 / GeoPandas
 * attribute columns `is_simple`, `is_ring`, `is_closed`, `is_ccw`, `get_num_points`,
 * `count_coordinates`, `get_num_interior_rings`, `get_num_geometries`, the element-wise
 * `equals_exact` / `equals_identical` of two aligned geometry columns, and
 * `extract_unique_points`. Every output is optional and costs only its own kernels.
 *
 * `isSimple` and `isRing` run {@link GPUSegmentIntersection} in same-feature self mode, so they are
 * exact for finite f32 coordinates. The self-mode successor search looks 1024 vertices ahead, so a
 * line with more than 1024 consecutive repeated vertices may be misjudged. `extractUniquePoints`
 * sorts vertices by (feature, x, y) with three stable radix passes, marks the first occurrence of
 * every distinct point in each feature, and compacts in vertex order, so the output order is
 * Shapely's. Nothing is read back. Row-wise columns use one thread per feature; a single feature
 * with millions of vertices therefore runs the `equalsExact`, `equalsIdentical` and `isCcw`
 * loops serially.
 *
 * Empty features follow Shapely where the layout can express them (a linestring with no vertices
 * is not closed, a polygon feature with no polygons has zero geometries).
 */
export class GPUGeometryPredicates implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeometryPredicatesProps;
  /** Number of features. */
  readonly featureCount: number;
  /** Number of vertices of `geometry`. */
  readonly vertexCount: number;

  constructor(props: GPUGeometryPredicatesProps) {
    this.id = props.id ?? 'geometry-predicates';
    this.props = props;
    const {id} = this;
    const {geometry} = props;
    this.featureCount = getPredicateFeatureCount(geometry);
    this.vertexCount = geometry.positions.length;
    this.validateGeometry('geometry', geometry);
    if (this.featureCount < 1) {
      throw new Error(`${id} geometry must contain at least one feature`);
    }
    if (this.vertexCount < 1) {
      throw new Error(`${id} geometry must contain at least one vertex`);
    }
    for (const [name, view] of [
      ['isSimple', props.isSimple],
      ['isRing', props.isRing],
      ['isClosed', props.isClosed],
      ['isCcw', props.isCcw],
      ['numPoints', props.numPoints],
      ['numCoordinates', props.numCoordinates],
      ['numInteriorRings', props.numInteriorRings],
      ['numGeometries', props.numGeometries],
      ['equalsExact', props.equalsExact],
      ['equalsIdentical', props.equalsIdentical]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} length must equal the feature count`);
        }
      }
    }
    if (props.isSimple || props.isRing) {
      if (!props.overflow || !props.intersectionCapacity) {
        throw new Error(`${id} isSimple and isRing require overflow and intersectionCapacity`);
      }
      if (!Number.isSafeInteger(props.intersectionCapacity) || props.intersectionCapacity < 1) {
        throw new Error(`${id} intersectionCapacity must be a positive integer`);
      }
    }
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['intersectionCount', props.intersectionCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.equalsExact || props.equalsIdentical) {
      const {other} = props;
      if (!other) {
        throw new Error(`${id} equalsExact and equalsIdentical require other`);
      }
      this.validateGeometry('other', other);
      if (other.kind !== geometry.kind || getPredicateFeatureCount(other) !== this.featureCount) {
        throw new Error(`${id} other must have the kind and feature count of geometry`);
      }
    }
    if (props.equalsExact) {
      if (!props.parameters) {
        throw new Error(`${id} equalsExact requires parameters`);
      }
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH} elements`
        );
      }
    }
    if (props.uniquePositions || props.uniqueOffsets) {
      if (!props.uniquePositions || !props.uniqueOffsets) {
        throw new Error(`${id} uniquePositions and uniqueOffsets must be given together`);
      }
      validatePackedView(props.uniquePositions, ['float32x2'], `${id} uniquePositions`);
      validatePackedUint32View(props.uniqueOffsets, `${id} uniqueOffsets`);
      if (props.uniquePositions.length < this.vertexCount) {
        throw new Error(`${id} uniquePositions must hold at least the vertex count`);
      }
      if (props.uniqueOffsets.length !== this.featureCount + 1) {
        throw new Error(`${id} uniqueOffsets must have featureCount + 1 rows`);
      }
    }
  }

  private validateGeometry(name: string, geometry: GPUSpatialJoinGeometry): void {
    const {id} = this;
    validatePackedView(geometry.positions, ['float32x2'], `${id} ${name}.positions`);
    for (const view of getPredicateGeometryViews(geometry).slice(1)) {
      validatePackedUint32View(view, `${id} ${name} offsets`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} offsets require a terminal entry`);
      }
    }
  }

  private getViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      ...getPredicateGeometryViews(props.geometry),
      ...(props.other ? getPredicateGeometryViews(props.other) : []),
      props.isSimple,
      props.isRing,
      props.isClosed,
      props.isCcw,
      props.numPoints,
      props.numCoordinates,
      props.numInteriorRings,
      props.numGeometries,
      props.overflow,
      props.intersectionCount,
      props.parameters,
      props.equalsExact,
      props.equalsIdentical,
      props.uniquePositions,
      props.uniqueOffsets
    ];
  }

  /** Returns the column, segment intersection, pair comparison and unique point nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount} = this;
    validateGraphViewsBelongToGraph(id, graph, this.getViews());
    const {geometry} = props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const geometryBindings = getPredicateGeometryBindings('g', geometry);
    const geometryWGSL = `${COMMON_WGSL}\n${getPredicateGeometryWGSL('g', geometry)}`;
    const isLines = geometry.kind === 'lines';

    const addColumn = (
      name: string,
      output: GraphDataView<'uint32'>,
      body: string,
      options: {
        declarations?: string;
        bindings?: WGSLKernelBinding[];
      } = {}
    ) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${name}`,
          operation: OPERATION,
          variant: name,
          bindings: [
            ...geometryBindings,
            ...(options.bindings ?? []),
            {
              name: 'output',
              view: output,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: featureCount,
          declarations: `${geometryWGSL}\n${options.declarations ?? ''}`,
          body
        })
      );
    };
    const vertexRange = `let start = gVertexStart(index);
  let end = gVertexStart(index + 1u);`;
    const write = (expression: string) => `output[outputOffset + index] = ${expression};`;

    if (props.isClosed) {
      addColumn(
        'is-closed',
        props.isClosed,
        isLines ? `${vertexRange}\n  ${write(CLOSED_EXPRESSION)}` : write('0u')
      );
    }
    if (props.numPoints) {
      addColumn(
        'num-points',
        props.numPoints,
        isLines ? `${vertexRange}\n  ${write('end - start')}` : write('0u')
      );
    }
    if (props.numCoordinates) {
      addColumn(
        'num-coordinates',
        props.numCoordinates,
        `${vertexRange}\n  ${write('end - start')}`
      );
    }
    if (props.numInteriorRings) {
      addColumn(
        'num-interior-rings',
        props.numInteriorRings,
        geometry.kind === 'polygons'
          ? `let firstPolygon = gFeatureOffsets[gFeatureOffsetsOffset + index];
  let polygonCount = gFeatureOffsets[gFeatureOffsetsOffset + index + 1u] - firstPolygon;
  var holes = 0u;
  if (polygonCount == 1u) {
    let ringCount = gPolygonOffsets[gPolygonOffsetsOffset + firstPolygon + 1u] - gPolygonOffsets[gPolygonOffsetsOffset + firstPolygon];
    holes = select(0u, ringCount - 1u, ringCount > 0u);
  }
  ${write('holes')}`
          : write('0u')
      );
    }
    if (props.numGeometries) {
      addColumn(
        'num-geometries',
        props.numGeometries,
        geometry.kind === 'polygons'
          ? write(
              'gFeatureOffsets[gFeatureOffsetsOffset + index + 1u] - gFeatureOffsets[gFeatureOffsetsOffset + index]'
            )
          : write('1u')
      );
    }
    if (props.isCcw) {
      addColumn(
        'is-ccw',
        props.isCcw,
        isLines ? `${vertexRange}\n  ${write('getLineIsCcw(start, end - start)')}` : write('0u'),
        {declarations: `${SEGMENT_PREDICATES_WGSL}\n${CCW_WGSL}`}
      );
    }

    // Simplicity: segment intersections within each feature, folded into an AND per feature.
    if ((props.isSimple || props.isRing) && geometry.kind !== 'points') {
      const simple =
        props.isSimple ?? createTransientView(graph, `${id}-simple`, 'uint32', featureCount);
      nodes.push(...this.getSimplicityNodes(graph, geometry, simple));
      if (props.isRing) {
        addColumn(
          'is-ring',
          props.isRing,
          isLines
            ? `${vertexRange}\n  ${write(`select(0u, simple[simpleOffset + index], ${CLOSED_EXPRESSION} == 1u)`)}`
            : write('0u'),
          {bindings: [{name: 'simple', view: simple, type: 'u32', access: 'read'}]}
        );
      }
    } else {
      if (props.isSimple) {
        addColumn('is-simple', props.isSimple, write('1u'));
      }
      if (props.isRing) {
        addColumn('is-ring', props.isRing, write('0u'));
      }
    }

    if (props.equalsExact || props.equalsIdentical) {
      nodes.push(...this.getPairNodes(graph));
    }
    if (props.uniquePositions && props.uniqueOffsets) {
      nodes.push(...this.getUniquePointNodes(graph, geometryBindings, geometryWGSL));
    }
    return nodes;
  }

  private getSimplicityNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    geometry: GPUSegmentGeometry,
    simple: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props, featureCount} = this;
    const capacity = props.intersectionCapacity as number;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-simple-clear`,
        operation: OPERATION,
        variant: 'simple-clear',
        bindings: [{name: 'output', view: simple, type: 'u32', access: 'read_write'}],
        invocationCount: featureCount,
        body: 'output[outputOffset + index] = 1u;'
      })
    );
    const pairLeft = createTransientView(graph, `${id}-pair-left`, 'uint32', capacity);
    const pairRight = createTransientView(graph, `${id}-pair-right`, 'uint32', capacity);
    const pairCount = createTransientView(graph, `${id}-pair-count`, 'uint32', 1);
    const leftFeatures = createTransientView(graph, `${id}-left-features`, 'uint32', capacity);
    const leftRings = createTransientView(graph, `${id}-left-rings`, 'uint32', capacity);
    const rightRings = createTransientView(graph, `${id}-right-rings`, 'uint32', capacity);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-intersection`,
        left: geometry,
        sameFeatureOnly: true,
        leafCapacity: props.leafCapacity,
        pairs: {
          leftIds: pairLeft,
          rightIds: pairRight,
          count: pairCount,
          overflow: props.overflow as GraphDataView<'uint32'>,
          requiredCount: props.intersectionCount
        },
        leftFeatures,
        leftRings,
        rightRings
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-simple-fold`,
        operation: OPERATION,
        variant: 'simple-fold',
        bindings: [
          {name: 'pairCount', view: pairCount, type: 'u32', access: 'read'},
          {name: 'leftFeatures', view: leftFeatures, type: 'u32', access: 'read'},
          {name: 'leftRings', view: leftRings, type: 'u32', access: 'read'},
          {name: 'rightRings', view: rightRings, type: 'u32', access: 'read'},
          {name: 'simple', view: simple, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `const SAME_RING_ONLY: bool = ${geometry.kind === 'polygons'};`,
        // Polygons only count a ring meeting itself; lines count every same-feature pair.
        body: `if (index >= pairCount[pairCountOffset]) { return; }
  if (SAME_RING_ONLY && leftRings[leftRingsOffset + index] != rightRings[rightRingsOffset + index]) { return; }
  atomicAnd(&simple[simpleOffset + leftFeatures[leftFeaturesOffset + index]], 0u);`
      })
    );
    return nodes;
  }

  private getPairNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {id, props, featureCount} = this;
    const geometry = props.geometry;
    const other = props.other as GPUSpatialJoinGeometry;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const layout = createTransientView(graph, `${id}-pair-layout`, 'uint32', featureCount * 3);
    const accessWGSL = `${getPredicateGeometryWGSL('a', geometry, false)}\n${getPredicateGeometryWGSL('b', other, false)}`;
    let structureBody: string;
    if (geometry.kind === 'polygons') {
      structureBody = `let firstA = aFeatureOffsets[aFeatureOffsetsOffset + index];
  let lastA = aFeatureOffsets[aFeatureOffsetsOffset + index + 1u];
  let firstB = bFeatureOffsets[bFeatureOffsetsOffset + index];
  let lastB = bFeatureOffsets[bFeatureOffsetsOffset + index + 1u];
  var matches = (lastA - firstA) == (lastB - firstB);
  for (var k = 0u; matches && k < lastA - firstA; k++) {
    let ringsStartA = aPolygonOffsets[aPolygonOffsetsOffset + firstA + k];
    let ringsEndA = aPolygonOffsets[aPolygonOffsetsOffset + firstA + k + 1u];
    let ringsStartB = bPolygonOffsets[bPolygonOffsetsOffset + firstB + k];
    let ringsEndB = bPolygonOffsets[bPolygonOffsetsOffset + firstB + k + 1u];
    if (ringsEndA - ringsStartA != ringsEndB - ringsStartB) { matches = false; break; }
    for (var r = 0u; r < ringsEndA - ringsStartA; r++) {
      let countA = aRingOffsets[aRingOffsetsOffset + ringsStartA + r + 1u] - aRingOffsets[aRingOffsetsOffset + ringsStartA + r];
      let countB = bRingOffsets[bRingOffsetsOffset + ringsStartB + r + 1u] - bRingOffsets[bRingOffsetsOffset + ringsStartB + r];
      if (countA != countB) { matches = false; break; }
    }
  }`;
    } else {
      structureBody =
        'let matches = (aVertexStart(index + 1u) - aVertexStart(index)) == (bVertexStart(index + 1u) - bVertexStart(index));';
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-structure`,
        operation: OPERATION,
        variant: 'pair-structure',
        bindings: [
          ...getPredicateGeometryBindings('a', geometry, false),
          ...getPredicateGeometryBindings('b', other, false),
          {name: 'pairLayout', view: layout, type: 'u32', access: 'read_write'}
        ],
        invocationCount: featureCount,
        declarations: `${accessWGSL}\nconst STRUCTURE_MISMATCH: u32 = ${STRUCTURE_MISMATCH}u;`,
        body: `${structureBody}
  let startA = aVertexStart(index);
  let startB = bVertexStart(index);
  pairLayout[pairLayoutOffset + index * 3u] = startA;
  pairLayout[pairLayoutOffset + index * 3u + 1u] = startB;
  pairLayout[pairLayoutOffset + index * 3u + 2u] = select(STRUCTURE_MISMATCH, aVertexStart(index + 1u) - startA, matches);`
      })
    );
    const {equalsExact, equalsIdentical, parameters} = props;
    const bindings: WGSLKernelBinding[] = [
      {name: 'aPositions', view: geometry.positions, type: 'f32', access: 'read'},
      {name: 'bPositions', view: other.positions, type: 'f32', access: 'read'},
      {name: 'pairLayout', view: layout, type: 'u32', access: 'read'}
    ];
    if (equalsExact && parameters) {
      bindings.push(
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'exactOutput', view: equalsExact, type: 'u32', access: 'read_write'}
      );
    }
    if (equalsIdentical) {
      bindings.push({
        name: 'identicalOutput',
        view: equalsIdentical,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-compare`,
        operation: OPERATION,
        variant: 'pair-compare',
        bindings,
        invocationCount: featureCount,
        declarations: `${COMMON_WGSL}
const STRUCTURE_MISMATCH: u32 = ${STRUCTURE_MISMATCH}u;
const WRITE_EXACT: bool = ${Boolean(equalsExact)};
const WRITE_IDENTICAL: bool = ${Boolean(equalsIdentical)};
fn readA(vertex: u32) -> vec2f { return vec2f(aPositions[aPositionsOffset + vertex * 2u], aPositions[aPositionsOffset + vertex * 2u + 1u]); }
fn readB(vertex: u32) -> vec2f { return vec2f(bPositions[bPositionsOffset + vertex * 2u], bPositions[bPositionsOffset + vertex * 2u + 1u]); }`,
        body: `let count = pairLayout[pairLayoutOffset + index * 3u + 2u];
  var exactMatch = count != STRUCTURE_MISMATCH;
  var identicalMatch = exactMatch;
  if (exactMatch) {
    let startA = pairLayout[pairLayoutOffset + index * 3u];
    let startB = pairLayout[pairLayoutOffset + index * 3u + 1u];
    ${equalsExact ? 'let tolerance = parameters[parametersOffset];' : ''}
    for (var k = 0u; k < count; k++) {
      let a = readA(startA + k);
      let b = readB(startB + k);
      ${equalsExact ? 'if (exactMatch && !withinTolerance(a, b, tolerance)) { exactMatch = false; }' : ''}
      ${equalsIdentical ? 'if (identicalMatch && !identicalPosition(a, b)) { identicalMatch = false; }' : ''}
      if ((!WRITE_EXACT || !exactMatch) && (!WRITE_IDENTICAL || !identicalMatch)) { break; }
    }
  }
  ${equalsExact ? 'exactOutput[exactOutputOffset + index] = select(0u, 1u, exactMatch);' : ''}
  ${equalsIdentical ? 'identicalOutput[identicalOutputOffset + index] = select(0u, 1u, identicalMatch);' : ''}`
      })
    );
    return nodes;
  }

  private getUniquePointNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    geometryBindings: WGSLKernelBinding[],
    geometryWGSL: string
  ): GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, vertexCount} = this;
    const uniquePositions = props.uniquePositions as GraphDataView<'float32x2'>;
    const uniqueOffsets = props.uniqueOffsets as GraphDataView<'uint32'>;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const rows = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', vertexCount);
    const keysY = rows('keys-y');
    const identity = rows('identity');
    const sortedY = rows('sorted-y');
    const orderByY = rows('order-y');
    const keysX = rows('keys-x');
    const sortedX = rows('sorted-x');
    const orderByX = rows('order-x');
    const keysFeature = rows('keys-feature');
    const sortedFeature = rows('sorted-feature');
    const order = rows('order');
    const flags = rows('first-flags');
    const starts = rows('first-starts');
    const addKernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      body: string,
      invocationCount: number = vertexCount
    ) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-unique-${name}`,
          operation: OPERATION,
          variant: `unique-${name}`,
          bindings: [...geometryBindings, ...bindings],
          invocationCount,
          declarations: `${geometryWGSL}\n${ORDER_KEY_WGSL}`,
          body
        })
      );
    };
    const read = (name: string, view: GraphDataView<'uint32'>) =>
      ({name, view, type: 'u32', access: 'read'}) as const;
    const readWrite = (name: string, view: GraphDataView<'uint32'>) =>
      ({name, view, type: 'u32', access: 'read_write'}) as const;
    const sort = (
      name: string,
      keys: GraphDataView<'uint32'>,
      values: GraphDataView<'uint32'>,
      outputKeys: GraphDataView<'uint32'>,
      outputValues: GraphDataView<'uint32'>,
      keyBits: number = 32
    ) =>
      nodes.push(
        ...new GPUSort({
          id: `${id}-unique-sort-${name}`,
          keys,
          values,
          outputKeys,
          outputValues,
          keyBits
        }).getCommandNodes(graph)
      );

    addKernel(
      'keys-y',
      [readWrite('keysOut', keysY), readWrite('idsOut', identity)],
      `keysOut[keysOutOffset + index] = getOrderKey(gVertex(index).y);
  idsOut[idsOutOffset + index] = index;`
    );
    sort('y', keysY, identity, sortedY, orderByY);
    addKernel(
      'keys-x',
      [read('orderIn', orderByY), readWrite('keysOut', keysX)],
      'keysOut[keysOutOffset + index] = getOrderKey(gVertex(orderIn[orderInOffset + index]).x);'
    );
    sort('x', keysX, orderByY, sortedX, orderByX);
    addKernel(
      'keys-feature',
      [read('orderIn', orderByX), readWrite('keysOut', keysFeature)],
      'keysOut[keysOutOffset + index] = min(gFeatureOfVertex(orderIn[orderInOffset + index]), g_FEATURE_COUNT);'
    );
    // Feature keys are at most featureCount, so the last (most significant) radix sort only needs
    // ceil(log2(featureCount + 1)) bits instead of 32: 2 to 3 passes instead of 8 for typical sizes.
    const featureKeyBits = Math.max(1, Math.ceil(Math.log2(featureCount + 1)));
    sort('feature', keysFeature, orderByX, sortedFeature, order, featureKeyBits);
    addKernel(
      'mark',
      [read('sortedFeature', sortedFeature), read('order', order), readWrite('flags', flags)],
      `let vertex = order[orderOffset + index];
  var first = index == 0u;
  if (!first) {
    let previous = order[orderOffset + index - 1u];
    first = sortedFeature[sortedFeatureOffset + index - 1u] != sortedFeature[sortedFeatureOffset + index]
      || !samePoint(gVertex(vertex), gVertex(previous));
  }
  flags[flagsOffset + vertex] = select(0u, 1u, first);`
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-unique-scan`,
        input: flags,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    addKernel(
      'emit',
      [
        read('flags', flags),
        read('starts', starts),
        {name: 'uniquePositions', view: uniquePositions, type: 'f32', access: 'read_write'}
      ],
      `if (flags[flagsOffset + index] == 1u) {
    let point = gVertex(index);
    let slot = starts[startsOffset + index];
    uniquePositions[uniquePositionsOffset + slot * 2u] = point.x;
    uniquePositions[uniquePositionsOffset + slot * 2u + 1u] = point.y;
  }`
    );
    addKernel(
      'offsets',
      [read('flags', flags), read('starts', starts), readWrite('uniqueOffsets', uniqueOffsets)],
      `let vertex = gVertexStart(index);
  var prefix = 0u;
  if (vertex >= g_VERTEX_COUNT) {
    prefix = starts[startsOffset + g_VERTEX_COUNT - 1u] + flags[flagsOffset + g_VERTEX_COUNT - 1u];
  } else {
    prefix = starts[startsOffset + vertex];
  }
  uniqueOffsets[uniqueOffsetsOffset + index] = prefix;`,
      featureCount + 1
    );
    return nodes;
  }
}

/** WGSL helpers shared by the column and comparison kernels. */
const COMMON_WGSL = /* wgsl */ `
fn isNaNBits(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn canonicalBits(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits, 0u, (bits & 0x7fffffffu) == 0u);
}
fn samePoint(first: vec2f, second: vec2f) -> bool {
  return canonicalBits(first.x) == canonicalBits(second.x) && canonicalBits(first.y) == canonicalBits(second.y);
}
fn identicalValue(first: f32, second: f32) -> bool {
  return canonicalBits(first) == canonicalBits(second) || (isNaNBits(first) && isNaNBits(second));
}
fn identicalPosition(first: vec2f, second: vec2f) -> bool {
  return identicalValue(first.x, second.x) && identicalValue(first.y, second.y);
}
// Distance at most tolerance, scaled so that tiny and huge offsets neither underflow nor overflow.
fn withinTolerance(first: vec2f, second: vec2f, tolerance: f32) -> bool {
  if (isNaNBits(first.x) || isNaNBits(first.y) || isNaNBits(second.x) || isNaNBits(second.y)) { return false; }
  let delta = first - second;
  let scale = max(abs(delta.x), abs(delta.y));
  if (scale == 0.0) { return true; }
  let scaled = delta / scale;
  return scale * sqrt(scaled.x * scaled.x + scaled.y * scaled.y) <= tolerance;
}
`;

/** Closed linestring: at least two vertices and the last equals the first. Uses `start`, `end`. */
const CLOSED_EXPRESSION =
  'select(0u, 1u, end >= start + 2u && samePoint(gVertex(start), gVertex(end - 1u)))';

/** Orderable uint32 key of an f32, with `-0` and `0` equal. */
const ORDER_KEY_WGSL = /* wgsl */ `
fn getOrderKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) == 0u) { bits = 0u; }
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;

/**
 * GEOS 3.13 `Orientation::isCCW` (JTS 1.19 algorithm: first highest point reached by a rising
 * segment, so flat tops are decided by their direction), on a vertex list judged as stored.
 */
const CCW_WGSL = /* wgsl */ `
fn getLineIsCcw(start: u32, size: u32) -> u32 {
  if (size < 4u) { return 0u; }
  let pointCount = size - 1u;
  var upHigh = gVertex(start);
  var upLow = upHigh;
  var previousY = upHigh.y;
  var upHighIndex = 0u;
  for (var i = 1u; i <= pointCount; i++) {
    let point = gVertex(start + i);
    if (point.y > previousY && point.y >= upHigh.y) {
      upHigh = point;
      upHighIndex = i;
      upLow = gVertex(start + i - 1u);
    }
    previousY = point.y;
  }
  if (upHighIndex == 0u) { return 0u; }
  var downLowIndex = upHighIndex;
  for (var step = 0u; step < size; step++) {
    downLowIndex = (downLowIndex + 1u) % pointCount;
    if (!(downLowIndex != upHighIndex && gVertex(start + downLowIndex).y == upHigh.y)) { break; }
  }
  let downLow = gVertex(start + downLowIndex);
  let downHigh = gVertex(start + select(pointCount - 1u, downLowIndex - 1u, downLowIndex > 0u));
  if (samePoint(upHigh, downHigh)) {
    if (samePoint(upLow, upHigh) || samePoint(downLow, upHigh) || samePoint(upLow, downLow)) { return 0u; }
    return select(0u, 1u, orientSign(upLow, upHigh, downLow) == 1);
  }
  return select(0u, 1u, downHigh.x < upHigh.x);
}
`;
