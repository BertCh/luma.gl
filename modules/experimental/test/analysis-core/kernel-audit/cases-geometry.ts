// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULonLatPolygonPreparation} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import {createLonLatPolygonCheckNode} from '../../../src/analysis-core/lon-lat-ring-wgsl';

import {
  GPUAffineTransform,
  GPUDelaunayTessellation,
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGeometryPredicates,
  GPUGroupConvexHull,
  GPUGroupGeometry,
  GPULabelPoint,
  GPUGreatCircleArcs,
  GPULineChunk,
  GPULineClipByPolygon,
  GPULineLocate,
  getGPUBufferSurfaceMaximumBoundaryRowCount,
  GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT,
  getGPULineLengthPerPolygonMaximumPairCapacity,
  getGPULineLengthPerPolygonMaximumPositionCount,
  getGPULineLocateMaximumEventCount,
  getGPULinearReferencingMaximumPlanarPositionCount,
  getGPULinearReferencingMaximumQueryCount,
  getGPULinearReferencingMaximumSphericalPositionCount,
  getGPUPolygonTriangulationMaximumNodeCount,
  getGPUSegmentIntersectionMaximumVertexCount,
  getPolygonTriangulationNodeCount,
  GPULineSegmentize,
  GPULineSimplification,
  GPULineSmooth,
  GPULinearReferencing,
  GPUBufferSurface,
  GPUCoverageDissolve,
  GPUCoverageSimplification,
  GPUCoverageValidity,
  GPUMapColoring,
  GPUMinimumBounds,
  GPUMinimumClearance,
  GPUOffsetCurve,
  GPUOffsetExpansion,
  GPUOutlineGeometry,
  GPUPolygonOverlay,
  GPUPolygonTriangulation,
  GPURectangleClip,
  GPUSegmentIntersection,
  GPUShapeDescriptors,
  GPUVertexSnap,
  GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH,
  getGPULineClipByPolygonMaximumSubPieceBound,
  getGPULineSplitMaximumPieceBound,
  getGPUSharedPathsMaximumIntersectionCapacity,
  GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS,
  GPU_OFFSET_CURVE_PARAMETER_LENGTH,
  GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
  GPU_VERTEX_SNAP_PARAMETER_LENGTH,
  getGPUOffsetCurveRowsPerVertex,
  getGPUOutlineGeometryVerticesPerInput,
  getPolygonTriangulationIndexCount,
  type GPUOffsetCurveJoinStyle,
  type GPUPolygonOverlayOperation,
  type GPUPolygonOverlayFillRule,
  GPU_LINE_CHUNK_PARAMETER_LENGTH,
  GPU_LINE_LOCATE_PARAMETER_LENGTH,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS,
  GPU_LINE_SMOOTH_PARAMETER_LENGTH,
  GPU_MAP_COLORING_PARAMETER_LENGTH,
  GPULineDensity,
  GPULineLengthPerPolygon,
  GPULineMerge,
  GPULineSplit,
  GPUSharedPaths,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPUGeometryValidity,
  GPUGridGenerator,
  GPULineNoding,
  GPUMakeValid,
  GPUPolygonize,
  GPUShapeGenerator,
  GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH,
  GPU_GRID_GENERATOR_PARAMETER_LENGTH,
  GPU_SHAPE_GENERATOR_PARAMETER_LENGTH,
  getGPUGridCellCount,
  getGPUGridCornerCount,
  getGPUGridVerticesPerCell,
  getGPUShapeVertexCount,
  type GPUGridType,
  GPUGeometryCleanup,
  GPUGeometryMeasures,
  GPUGeometryOrientation,
  GPULocalFrameProjection,
  GPUSegmentRingAssembly,
  GPUVoronoiDiagram,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_PLANAR_SPATIAL_CONTEXT,
  createGPULocalFrame,
  type GPUSpatialContext
} from '../../../src/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';
import {
  createLinePrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode
} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';

// Shared fixtures. Nothing is allocated: every view is a transient description, so the sizes only
// have to be consistent with each other.

/** Rows of a moderate "every option" case. */
const ROWS = 300;
/** Vertices of a moderate polygon/line case. */
const VERTICES = 1200;

/** Features per operand and crossing capacity of the `GPUPolygonOverlay` limit cases. */
const OVERLAY_AUDIT_FEATURES = 100000;
const OVERLAY_AUDIT_INTERSECTIONS = 100000;

/**
 * Largest vertex count per operand that `GPUPolygonOverlay` accepts with `intersections` crossings:
 * both operands add one closing vertex per ring, and vertices plus crossings (hot pixels) must fit
 * a quarter of the segment table (four diagonal vertices per hot pixel).
 */
function getOverlayMaximumOperandVertices(context: KernelAuditContext, intersections: number) {
  const quarter = Math.floor(getGPUSegmentIntersectionMaximumVertexCount(context.device) / 4);
  // polygonGeometry(…, features) has two rings per feature.
  return Math.floor((quarter - intersections - 4 * OVERLAY_AUDIT_FEATURES) / 2);
}

const ELLIPSOIDAL_CONTEXT: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'ellipsoidal',
  units: 'meters'
};
const RHUMB_CONTEXT: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'rhumb',
  units: 'meters'
};

/** Four-level polygon layout fields (features own polygons own rings own vertices). */
function polygonFields(context: KernelAuditContext, vertices = VERTICES, features = ROWS) {
  return {
    positions: context.view('float32x2', vertices, 'positions'),
    featureOffsets: context.view('uint32', features + 1, 'feature-offsets'),
    polygonOffsets: context.view('uint32', features + 1, 'polygon-offsets'),
    ringOffsets: context.view('uint32', features * 2 + 1, 'ring-offsets')
  };
}

/** Line layout fields, one feature per line. */
function lineFields(context: KernelAuditContext, vertices = VERTICES, lines = ROWS) {
  return {
    positions: context.view('float32x2', vertices, 'positions'),
    lineOffsets: context.view('uint32', lines + 1, 'line-offsets')
  };
}

/**
 * Vertex count of the audit polygon layout (8 vertices and two rings per feature) whose
 * triangulation scratch holds the most nodes at or below `maximumNodeCount`, or one vertex more
 * (the first layout above the maximum) when `above` is true.
 */
function getTriangulationVertexCount(maximumNodeCount: number, above: boolean): number {
  const getNodeCount = (vertices: number) =>
    getPolygonTriangulationNodeCount(
      vertices,
      2 * Math.floor(vertices / 8),
      Math.floor(vertices / 8)
    );
  let vertices = Math.floor((maximumNodeCount * 2) / 7);
  while (getNodeCount(vertices) > maximumNodeCount) vertices--;
  while (getNodeCount(vertices + 1) <= maximumNodeCount) vertices++;
  return above ? vertices + 1 : vertices;
}

/** One-row uint32 status scalar. */
function scalar(context: KernelAuditContext, name = 'scalar') {
  return context.view('uint32', 1, name);
}

/** Cardinality and overflow status plus the exact required count. */
function countStatus(context: KernelAuditContext) {
  return {
    count: scalar(context, 'count'),
    requiredCount: scalar(context, 'required-count'),
    overflow: scalar(context, 'overflow')
  };
}

/** Status with every optional bounded-result scalar wired. */
function fullStatus(context: KernelAuditContext) {
  return {
    ...countStatus(context),
    candidateOverflow: scalar(context, 'candidate-overflow'),
    uncertainCount: scalar(context, 'uncertain-count'),
    invalidCount: scalar(context, 'invalid-count'),
    iterationCount: scalar(context, 'iteration-count'),
    converged: scalar(context, 'converged'),
    iterationLimitReached: scalar(context, 'iteration-limit-reached'),
    approximation: scalar(context, 'approximation')
  };
}

/** Packed point features in the layout of `GPUSpatialJoinGeometry`. */
function pointGeometry(context: KernelAuditContext, vertices = ROWS) {
  return {kind: 'points', positions: context.view('float32x2', vertices, 'positions')} as const;
}

/** Packed line features in the layout of `GPUSpatialJoinGeometry`. */
function lineGeometry(context: KernelAuditContext, vertices = VERTICES, features = ROWS) {
  return {kind: 'lines', ...lineFields(context, vertices, features)} as const;
}

/** Packed polygon features in the layout of `GPUSpatialJoinGeometry`. */
function polygonGeometry(context: KernelAuditContext, vertices = VERTICES, features = ROWS) {
  return {kind: 'polygons', ...polygonFields(context, vertices, features)} as const;
}

/** Line or polygon geometry in the layout of `GPUSegmentGeometry`. */
function segmentGeometry(context: KernelAuditContext, kind: 'lines' | 'polygons') {
  return kind === 'lines' ? lineGeometry(context) : polygonGeometry(context);
}

/** Geometry of the given kind in the layout of `GPUSpatialJoinGeometry`. */
function joinGeometry(context: KernelAuditContext, kind: 'points' | 'lines' | 'polygons') {
  if (kind === 'points') return pointGeometry(context);
  return kind === 'lines' ? lineGeometry(context) : polygonGeometry(context);
}

/** Topology status with every scalar wired. */
function topologyStatus(context: KernelAuditContext) {
  return {
    ...countStatus(context),
    candidateOverflow: scalar(context, 'candidate-overflow'),
    uncertainCount: scalar(context, 'uncertain-count')
  };
}

/** Noded segment port of `capacity` segments. */
function nodedSegments(context: KernelAuditContext, capacity: number) {
  const {view} = context;
  return {
    endpoints: view('float32x4', capacity, 'noded-endpoints'),
    sourceFeatureIds: view('uint32', capacity, 'noded-features'),
    sourceRingIds: view('uint32', capacity, 'noded-rings'),
    sourceEdgeIds: view('uint32', capacity, 'noded-edges'),
    sourceStartParameters: view('float32', capacity, 'noded-start'),
    sourceEndParameters: view('float32', capacity, 'noded-end'),
    status: topologyStatus(context)
  };
}

/** Full `GPUPolygonizeOutput` for `segments` noded segments and `rings` output rings. */
function polygonizeOutput(
  context: KernelAuditContext,
  segments: number,
  rings: number,
  vertices: number
) {
  const {view} = context;
  return {
    nodedSegments: nodedSegments(context, segments),
    directedEdges: {
      endpoints: view('float32x4', 2 * segments, 'directed-endpoints'),
      sourceSegmentIds: view('uint32', 2 * segments, 'directed-segments'),
      sourceFeatureIds: view('uint32', 2 * segments, 'directed-features'),
      twinIds: view('uint32', 2 * segments, 'directed-twins'),
      ringIds: view('uint32', 2 * segments, 'directed-rings'),
      flags: view('uint32', 2 * segments, 'directed-flags'),
      status: topologyStatus(context)
    },
    polygons: {
      kind: 'polygons' as const,
      positions: view('float32x2', vertices, 'polygon-positions'),
      featureOffsets: view('uint32', rings + 1, 'polygon-features'),
      polygonOffsets: view('uint32', rings + 1, 'polygon-polygons'),
      ringOffsets: view('uint32', rings + 1, 'polygon-rings'),
      sourceIds: view('uint32', rings, 'polygon-source-ids')
    },
    diagnostics: {
      edgeClasses: view('uint32', segments, 'edge-classes'),
      closedRingCount: scalar(context, 'closed-ring-count'),
      openChainCount: scalar(context, 'open-chain-count'),
      cutEdgeCount: scalar(context, 'cut-edge-count'),
      dangleCount: scalar(context, 'dangle-count'),
      unusedEdgeCount: scalar(context, 'unused-edge-count')
    },
    status: topologyStatus(context)
  };
}

const PRECISION = {
  vertexTolerance: 1e-5,
  predicates: 'exact-or-uncertain',
  storage: 'float32'
} as const;

/** Line pieces output (`GPULineSplitPieces`) with every optional column wired. */
function linePieces(
  context: KernelAuditContext,
  pieceCapacity: number,
  vertexCapacity: number,
  withSourceEdges = true
) {
  const {view} = context;
  return {
    geometry: {
      kind: 'lines' as const,
      positions: view('float32x2', vertexCapacity, 'piece-positions'),
      lineOffsets: view('uint32', pieceCapacity + 1, 'piece-offsets')
    },
    sourceIds: view('uint32', pieceCapacity, 'piece-source-ids'),
    status: {
      ...countStatus(context),
      candidateOverflow: scalar(context, 'candidate-overflow'),
      uncertainCount: scalar(context, 'uncertain-count'),
      vertexCount: scalar(context, 'vertex-count'),
      requiredVertexCount: scalar(context, 'required-vertex-count')
    },
    vertexSourceEdges: withSourceEdges ? view('uint32', vertexCapacity, 'source-edges') : undefined
  };
}

/** `GPULineGeometryOutput` with every optional column wired. */
function lineGeometryOutput(
  context: KernelAuditContext,
  lineCapacity: number,
  vertexCapacity: number,
  columns: {sourceRows?: boolean; measures?: boolean} = {sourceRows: true, measures: true}
) {
  const {view} = context;
  return {
    positions: view('float32x2', vertexCapacity, 'line-positions'),
    lineOffsets: view('uint32', lineCapacity + 1, 'line-offsets'),
    status: {...countStatus(context), lineCount: scalar(context, 'line-count')},
    sourceLineIndices: view('uint32', lineCapacity, 'line-sources'),
    sourceRows: columns.sourceRows ? view('uint32', vertexCapacity, 'line-source-rows') : undefined,
    measures: columns.measures ? view('float32', vertexCapacity, 'line-measures') : undefined
  };
}

/** `GPUPolygonOverlayOutput` with every optional column wired. */
function overlayOutput(
  context: KernelAuditContext,
  boundaryCapacity: number,
  rings: number,
  vertices: number
) {
  const {view} = context;
  const sourceIds = view('uint32', rings, 'overlay-source-ids');
  return {
    geometry: {
      kind: 'polygons' as const,
      positions: view('float32x2', vertices, 'overlay-positions'),
      featureOffsets: view('uint32', rings + 1, 'overlay-features'),
      polygonOffsets: view('uint32', rings + 1, 'overlay-polygons'),
      ringOffsets: view('uint32', rings + 1, 'overlay-rings'),
      sourceIds
    },
    status: topologyStatus(context),
    boundary: {
      endpoints: view('float32x4', boundaryCapacity, 'boundary-endpoints'),
      operandIds: view('uint32', boundaryCapacity, 'boundary-operands'),
      sourceFeatureIds: view('uint32', boundaryCapacity, 'boundary-features'),
      status: topologyStatus(context)
    }
  };
}

const AFFINE_PARAMETER_LENGTH = 12;
/** Largest accepted `GPULineSimplification` `finishSpanLimit` (not exported by the barrel). */
const MAXIMUM_FINISH_SPAN_LIMIT = 128;

/** Kernel-audit cases for geometry. */
export const GEOMETRY_CASES: KernelAuditCase[] = [
  // GPUSegmentRingAssembly
  ...(['left', 'right'] as const).map(
    (interiorSide): KernelAuditCase => ({
      exportName: 'GPUSegmentRingAssembly',
      name: `every option, ${interiorSide}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUSegmentRingAssembly({
            endpoints: view('float32x4', ROWS),
            count: scalar(context),
            groups: view('uint32', ROWS),
            pairedOpposites: interiorSide === 'left',
            vertexTolerance: 1e-5,
            interiorSide,
            normalizeWinding: true,
            spatialContext: GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
            cancelOpposingSegments: true,
            splitTouchingRings: true,
            output: {
              ringOffsets: view('uint32', 101),
              positions: view('float32x2', ROWS + 100),
              ringAreas: view('float32', 100),
              ringIsHole: view('uint32', 100),
              ringShells: view('uint32', 100),
              ringGroups: view('uint32', 100),
              segmentRings: view('uint32', ROWS),
              segmentVertices: view('uint32', ROWS),
              segmentFlags: view('uint32', ROWS),
              polygons: {
                kind: 'polygons',
                positions: view('float32x2', ROWS + 100),
                ringOffsets: view('uint32', 101),
                polygonOffsets: view('uint32', 101),
                featureOffsets: view('uint32', 101),
                sourceIds: view('uint32', 100)
              },
              status: {
                ...countStatus(context),
                openSegmentCount: scalar(context, 'open'),
                touchingSegmentCount: scalar(context, 'touching')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUSegmentRingAssembly',
    name: 'segments at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const segments = maximumRows(16);
      const rings = Math.floor(segments / 3);
      graph.add(
        new GPUSegmentRingAssembly({
          endpoints: view('float32x4', segments),
          groups: view('uint32', segments),
          splitTouchingRings: true,
          cancelOpposingSegments: true,
          output: {
            ringOffsets: view('uint32', rings + 1),
            positions: view('float32x2', segments + rings),
            ringAreas: view('float32', rings),
            ringIsHole: view('uint32', rings),
            ringShells: view('uint32', rings),
            ringGroups: view('uint32', rings),
            segmentRings: view('uint32', segments),
            segmentVertices: view('uint32', segments),
            segmentFlags: view('uint32', segments),
            status: countStatus(context)
          }
        })
      );
    }
  },

  // GPUDelaunayTessellation
  {
    exportName: 'GPUDelaunayTessellation',
    name: 'every output',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUDelaunayTessellation({
          positions: view('float32x2', ROWS),
          output: {
            triangles: view('uint32x3', 2 * ROWS),
            status: {...fullStatus(context), duplicateCount: scalar(context, 'duplicates')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUDelaunayTessellation',
    // The graph holds several nodes per site (one set per insertion), so the binding-limited point
    // count would build millions of nodes; this is the largest set that stays practical.
    name: 'a few thousand points',
    build: context => {
      const {graph, view} = context;
      const points = 2048;
      graph.add(
        new GPUDelaunayTessellation({
          positions: view('float32x2', points),
          output: {
            triangles: view('uint32x3', 2 * points),
            status: countStatus(context)
          }
        })
      );
    }
  },

  // GPUVoronoiDiagram
  {
    exportName: 'GPUVoronoiDiagram',
    name: 'every output',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUVoronoiDiagram({
          positions: view('float32x2', ROWS),
          triangles: view('uint32x3', 2 * ROWS),
          triangleCount: scalar(context),
          clipBounds: view('float32x4', 1),
          output: {
            segments: view('float32x4', 3 * ROWS),
            siteIds: view('uint32x2', 3 * ROWS),
            status: fullStatus(context)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUVoronoiDiagram',
    name: 'sites at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      // 3n segments of 16 bytes and 2n padded triangles of 16 bytes.
      const sites = maximumRows(48);
      graph.add(
        new GPUVoronoiDiagram({
          positions: view('float32x2', sites),
          triangles: view('uint32x3', 2 * sites),
          triangleCount: scalar(context),
          clipBounds: view('float32x4', 1),
          output: {
            segments: view('float32x4', 3 * sites),
            siteIds: view('uint32x2', 3 * sites),
            status: countStatus(context)
          }
        })
      );
    }
  },

  // GPUAffineTransform
  ...(['parameters', 'featureTransforms'] as const).map(
    (form): KernelAuditCase => ({
      exportName: 'GPUAffineTransform',
      name: `lines, ${form}`,
      build: context => {
        const {graph, view} = context;
        const featureCount = 100;
        graph.add(
          new GPUAffineTransform({
            geometryType: 'lines',
            ...lineFields(context),
            featureOffsets: view('uint32', featureCount + 1),
            origins: form === 'parameters' ? ['center', 'centroid'] : undefined,
            parameters:
              form === 'parameters' ? view('float32', AFFINE_PARAMETER_LENGTH) : undefined,
            featureTransforms:
              form === 'featureTransforms' ? view('float32', 6 * featureCount) : undefined,
            output: {positions: view('float32x2', VERTICES)}
          })
        );
      }
    })
  ),
  ...(['center', 'centroid'] as const).map(
    (origin): KernelAuditCase => ({
      exportName: 'GPUAffineTransform',
      name: `polygons, ${origin} origin`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUAffineTransform({
            geometryType: 'polygons',
            ...polygonFields(context),
            origins: [origin],
            parameters: view('float32', AFFINE_PARAMETER_LENGTH),
            output: {positions: view('float32x2', VERTICES)}
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUAffineTransform',
    name: 'polygons, featureTransforms',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUAffineTransform({
          geometryType: 'polygons',
          ...polygonFields(context),
          featureTransforms: view('float32', 6 * ROWS),
          output: {positions: view('float32x2', VERTICES)}
        })
      );
    }
  },
  {
    exportName: 'GPUAffineTransform',
    name: 'points, center origin',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUAffineTransform({
          geometryType: 'points',
          positions: view('float32x2', VERTICES),
          featureOffsets: view('uint32', ROWS + 1),
          origins: ['center'],
          parameters: view('float32', AFFINE_PARAMETER_LENGTH),
          output: {positions: view('float32x2', VERTICES)}
        })
      );
    }
  },
  {
    exportName: 'GPUAffineTransform',
    name: 'polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUAffineTransform({
          geometryType: 'polygons',
          ...polygonFields(context, vertices, features),
          origins: ['center', 'centroid'],
          parameters: view('float32', AFFINE_PARAMETER_LENGTH),
          output: {positions: view('float32x2', vertices)}
        })
      );
    }
  },

  // GPUGeometryOrientation
  {
    exportName: 'GPUGeometryOrientation',
    name: 'orient polygons, reversed rings',
    build: context => {
      const {graph, view} = context;
      const fields = polygonFields(context);
      graph.add(
        new GPUGeometryOrientation({
          mode: 'orient-polygons',
          ...fields,
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', VERTICES),
            reversedRings: view('uint32', fields.ringOffsets.length - 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryOrientation',
    name: 'reverse polygons',
    build: context => {
      const {graph, view} = context;
      const fields = polygonFields(context);
      graph.add(
        new GPUGeometryOrientation({
          mode: 'reverse',
          ...fields,
          output: {
            positions: view('float32x2', VERTICES),
            reversedRings: view('uint32', fields.ringOffsets.length - 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryOrientation',
    name: 'reverse lines',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUGeometryOrientation({
          mode: 'reverse',
          ...lineFields(context),
          output: {
            positions: view('float32x2', VERTICES),
            reversedRings: view('uint32', ROWS)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryOrientation',
    name: 'orient polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUGeometryOrientation({
          mode: 'orient-polygons',
          ...fields,
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', vertices),
            reversedRings: view('uint32', fields.ringOffsets.length - 1)
          }
        })
      );
    }
  },

  // GPUGeometryCleanup
  {
    exportName: 'GPUGeometryCleanup',
    name: 'points',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUGeometryCleanup({
          geometryType: 'points',
          positions: view('float32x2', VERTICES),
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', VERTICES),
            status: countStatus(context)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryCleanup',
    name: 'lines',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUGeometryCleanup({
          geometryType: 'lines',
          ...lineFields(context),
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', VERTICES),
            lineOffsets: view('uint32', ROWS + 1),
            status: countStatus(context)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryCleanup',
    name: 'polygons, collapsed count',
    build: context => {
      const {graph, view} = context;
      const fields = polygonFields(context);
      graph.add(
        new GPUGeometryCleanup({
          geometryType: 'polygons',
          ...fields,
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', VERTICES),
            ringOffsets: view('uint32', fields.ringOffsets.length),
            status: {...countStatus(context), collapsedCount: scalar(context, 'collapsed')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryCleanup',
    name: 'polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUGeometryCleanup({
          geometryType: 'polygons',
          ...fields,
          parameters: view('float32', 4),
          output: {
            positions: view('float32x2', vertices),
            ringOffsets: view('uint32', fields.ringOffsets.length),
            status: {...countStatus(context), collapsedCount: scalar(context, 'collapsed')}
          }
        })
      );
    }
  },

  // GPUGeometryMeasures
  ...(
    [
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT, undefined],
      ['great-circle', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT, undefined],
      [
        'great-circle polygon edges',
        {...GPU_GREAT_CIRCLE_SPATIAL_CONTEXT, edgeModel: 'great-circle'},
        undefined
      ],
      ['ellipsoidal geodesic', ELLIPSOIDAL_CONTEXT, 'geodesic'],
      ['ellipsoidal coordinate-linear', ELLIPSOIDAL_CONTEXT, 'coordinate-linear']
    ] as const
  ).map(
    ([label, spatialContext, ellipsoidalEdgeModel]): KernelAuditCase => ({
      exportName: 'GPUGeometryMeasures',
      name: `polygons, ${label}`,
      build: context => {
        const {graph, view} = context;
        const groupCount = 16;
        graph.add(
          new GPUGeometryMeasures({
            geometryType: 'polygons',
            ...polygonFields(context),
            spatialContext,
            ellipsoidalEdgeModel,
            cooperativeRingRows: 512,
            groupIds: view('uint32', ROWS),
            groupCount,
            output: {
              lengths: view('float32', ROWS),
              areas: view('float32', ROWS),
              centroids: view('float32x2', ROWS),
              bounds: view('float32x4', ROWS),
              vertexCounts: view('uint32', ROWS),
              extremeVertices: view('uint32x4', ROWS),
              status: {invalidCount: scalar(context, 'invalid')}
            },
            groupOutput: {
              lengths: view('float32', groupCount),
              areas: view('float32', groupCount),
              centroids: view('float32x2', groupCount),
              bounds: view('float32x4', groupCount),
              vertexCounts: view('uint32', groupCount),
              featureCounts: view('uint32', groupCount)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGeometryMeasures',
    name: 'polygons, cooperative path off',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUGeometryMeasures({
          geometryType: 'polygons',
          ...polygonFields(context),
          spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
          cooperativeRingRows: 0,
          output: {
            lengths: view('float32', ROWS),
            areas: view('float32', ROWS),
            centroids: view('float32x2', ROWS)
          }
        })
      );
    }
  },
  ...(['planar', 'geographic'] as const).map(
    (label): KernelAuditCase => ({
      exportName: 'GPUGeometryMeasures',
      name: `lines, ${label}`,
      build: context => {
        const {graph, view} = context;
        const groupCount = 16;
        graph.add(
          new GPUGeometryMeasures({
            geometryType: 'lines',
            ...lineFields(context),
            featureOffsets: view('uint32', 101),
            spatialContext:
              label === 'planar' ? GPU_PLANAR_SPATIAL_CONTEXT : GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
            groupIds: view('uint32', 100),
            groupCount,
            output: {
              lengths: view('float32', 100),
              centroids: view('float32x2', 100),
              bounds: view('float32x4', 100),
              vertexCounts: view('uint32', 100),
              extremeVertices: view('uint32x4', 100),
              status: {invalidCount: scalar(context, 'invalid')}
            },
            groupOutput: {
              lengths: view('float32', groupCount),
              centroids: view('float32x2', groupCount),
              bounds: view('float32x4', groupCount),
              vertexCounts: view('uint32', groupCount),
              featureCounts: view('uint32', groupCount)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGeometryMeasures',
    name: 'points',
    build: context => {
      const {graph, view} = context;
      const groupCount = 16;
      graph.add(
        new GPUGeometryMeasures({
          geometryType: 'points',
          positions: view('float32x2', VERTICES),
          featureOffsets: view('uint32', ROWS + 1),
          spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
          groupIds: view('uint32', ROWS),
          groupCount,
          output: {
            lengths: view('float32', ROWS),
            centroids: view('float32x2', ROWS),
            bounds: view('float32x4', ROWS),
            vertexCounts: view('uint32', ROWS),
            extremeVertices: view('uint32x4', ROWS)
          },
          groupOutput: {
            centroids: view('float32x2', groupCount),
            bounds: view('float32x4', groupCount),
            vertexCounts: view('uint32', groupCount),
            featureCounts: view('uint32', groupCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryMeasures',
    name: 'ellipsoidal polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const features = Math.floor(vertices / 8);
      const groupCount = 1024;
      graph.add(
        new GPUGeometryMeasures({
          geometryType: 'polygons',
          ...polygonFields(context, vertices, features),
          spatialContext: ELLIPSOIDAL_CONTEXT,
          groupIds: view('uint32', features),
          groupCount,
          output: {
            lengths: view('float32', features),
            areas: view('float32', features),
            centroids: view('float32x2', features),
            bounds: view('float32x4', features),
            vertexCounts: view('uint32', features),
            extremeVertices: view('uint32x4', features),
            status: {invalidCount: scalar(context, 'invalid')}
          },
          groupOutput: {
            areas: view('float32', groupCount),
            centroids: view('float32x2', groupCount),
            featureCounts: view('uint32', groupCount)
          }
        })
      );
    }
  },

  // GPUGeodesicPairs
  ...(
    [
      ['sphere', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT],
      ['wgs84', ELLIPSOIDAL_CONTEXT],
      ['rhumb', RHUMB_CONTEXT]
    ] as const
  ).map(
    ([label, spatialContext]): KernelAuditCase => ({
      exportName: 'GPUGeodesicPairs',
      name: `${label}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUGeodesicPairs({
            spatialContext,
            origins: view('float32x2', ROWS),
            targets: view('float32x2', ROWS),
            maximumIterations: 200,
            output: {
              distances: view('float32', ROWS),
              initialBearings: view('float32', ROWS),
              finalBearings: view('float32', ROWS),
              midpoints: view('float32x2', ROWS),
              status: label === 'wgs84' ? {converged: view('uint32', ROWS)} : undefined
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGeodesicPairs',
    name: 'wgs84 at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(32);
      graph.add(
        new GPUGeodesicPairs({
          spatialContext: ELLIPSOIDAL_CONTEXT,
          origins: view('float32x2', rows),
          targets: view('float32x2', rows),
          maximumIterations: 200,
          output: {
            distances: view('float32', rows),
            initialBearings: view('float32', rows),
            finalBearings: view('float32', rows),
            midpoints: view('float32x2', rows),
            status: {converged: view('uint32', rows)}
          }
        })
      );
    }
  },

  // GPUGeodesicDestination
  ...(
    [
      ['sphere', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT],
      ['wgs84', ELLIPSOIDAL_CONTEXT],
      ['rhumb', RHUMB_CONTEXT]
    ] as const
  ).map(
    ([label, spatialContext]): KernelAuditCase => ({
      exportName: 'GPUGeodesicDestination',
      name: `${label}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUGeodesicDestination({
            spatialContext,
            origins: view('float32x2', ROWS),
            bearings: view('float32', ROWS),
            distances: view('float32', ROWS),
            maximumIterations: 200,
            output: {
              destinations: view('float32x2', ROWS),
              finalBearings: view('float32', ROWS),
              status: label === 'wgs84' ? {converged: view('uint32', ROWS)} : undefined
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGeodesicDestination',
    name: 'wgs84 at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(16);
      graph.add(
        new GPUGeodesicDestination({
          spatialContext: ELLIPSOIDAL_CONTEXT,
          origins: view('float32x2', rows),
          bearings: view('float32', rows),
          distances: view('float32', rows),
          maximumIterations: 200,
          output: {
            destinations: view('float32x2', rows),
            finalBearings: view('float32', rows),
            status: {converged: view('uint32', rows)}
          }
        })
      );
    }
  },

  // GPULocalFrameProjection
  ...(['azimuthal-equidistant', 'lambert-equal-area'] as const).map(
    (kind): KernelAuditCase => ({
      exportName: 'GPULocalFrameProjection',
      name: `${kind}, double precision input`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULocalFrameProjection({
            frame: createGPULocalFrame({origin: [8.5417, 47.3769], kind}),
            positions: view('uint32x4', ROWS),
            output: view('float32x2', ROWS),
            bounds: [7.5, 46.5, 9.5, 48]
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULocalFrameProjection',
    name: 'rows at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(16);
      graph.add(
        new GPULocalFrameProjection({
          frame: createGPULocalFrame({origin: [8.5417, 47.3769]}),
          positions: view('uint32x4', rows),
          output: view('float32x2', rows),
          bounds: [7.5, 46.5, 9.5, 48]
        })
      );
    }
  },

  // GPUGeometryPredicates
  ...(['points', 'lines', 'polygons'] as const).map(
    (kind): KernelAuditCase => ({
      exportName: 'GPUGeometryPredicates',
      name: `${kind}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUGeometryPredicates({
            geometry: joinGeometry(context, kind),
            other: joinGeometry(context, kind),
            parameters: view('float32', GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH),
            intersectionCapacity: 4096,
            leafCapacity: 2048,
            output: {
              isSimple: view('uint32', ROWS),
              isRing: view('uint32', ROWS),
              isClosed: view('uint32', ROWS),
              isCcw: view('uint32', ROWS),
              numPoints: view('uint32', ROWS),
              numCoordinates: view('uint32', ROWS),
              numInteriorRings: view('uint32', ROWS),
              numGeometries: view('uint32', ROWS),
              equalsExact: view('uint32', ROWS),
              equalsIdentical: view('uint32', ROWS),
              uniquePositions: view('float32x2', VERTICES),
              uniqueOffsets: view('uint32', ROWS + 1),
              status: {
                overflow: scalar(context, 'overflow'),
                intersectionCount: scalar(context, 'count')
              }
            }
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUGeometryPredicates',
      name: excess ? 'polygons above the maximum' : 'polygons at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess;
        const features = Math.floor(vertices / 8);
        graph.add(
          new GPUGeometryPredicates({
            geometry: polygonGeometry(context, vertices, features),
            other: polygonGeometry(context, vertices, features),
            parameters: view('float32', GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH),
            intersectionCapacity: maximumRows(16),
            output: {
              isSimple: view('uint32', features),
              isRing: view('uint32', features),
              isClosed: view('uint32', features),
              isCcw: view('uint32', features),
              numPoints: view('uint32', features),
              numCoordinates: view('uint32', features),
              numInteriorRings: view('uint32', features),
              numGeometries: view('uint32', features),
              equalsExact: view('uint32', features),
              equalsIdentical: view('uint32', features),
              uniquePositions: view('float32x2', vertices),
              uniqueOffsets: view('uint32', features + 1),
              status: {
                overflow: scalar(context, 'overflow'),
                intersectionCount: scalar(context, 'count')
              }
            }
          })
        );
      }
    })
  ),

  // GPULineNoding
  ...[false, true].map(
    (spatialSort): KernelAuditCase => ({
      exportName: 'GPULineNoding',
      name: `every option, spatialSort ${spatialSort}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineNoding({
            lines: {...lineGeometry(context), sourceIds: view('uint32', ROWS)},
            intersectionCapacity: 4096,
            spatialSort,
            leafCapacity: 2048,
            output: nodedSegments(context, 4096)
          })
        );
      }
    })
  ),
  ...[
    ['at the maximum', 0, 0],
    ['vertices above the maximum', 1, 0],
    ['piece bound above the maximum', 0, 1]
  ].map(
    ([name, excessVertices, excessCapacity]): KernelAuditCase => ({
      exportName: 'GPULineNoding',
      name: name as string,
      build: context => {
        const {graph} = context;
        const capacity =
          Math.floor((getGPULineSplitMaximumPieceBound(context.device) - 100000) / 4) +
          (excessCapacity as number);
        graph.add(
          new GPULineNoding({
            lines: lineGeometry(
              context,
              getGPUSegmentIntersectionMaximumVertexCount(context.device) +
                (excessVertices as number),
              100000
            ),
            intersectionCapacity: capacity,
            output: nodedSegments(context, capacity)
          })
        );
      }
    })
  ),

  // GPUPolygonize
  ...[false, true].map(
    (grouped): KernelAuditCase => ({
      exportName: 'GPUPolygonize',
      name: `every option, groups ${grouped}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUPolygonize({
            lines: {...lineGeometry(context), sourceIds: view('uint32', ROWS)},
            groupIds: grouped ? view('uint32', ROWS) : undefined,
            intersectionCapacity: 4096,
            precision: PRECISION,
            spatialSort: grouped,
            leafCapacity: 2048,
            output: polygonizeOutput(context, 4096, 1024, 4096)
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUPolygonize',
      name: excess ? 'vertices above the maximum' : 'at the maximum',
      build: context => {
        const {graph} = context;
        const segments = Math.floor(
          (getGPULineSplitMaximumPieceBound(context.device) - 100000) / 4
        );
        graph.add(
          new GPUPolygonize({
            lines: lineGeometry(
              context,
              getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess,
              100000
            ),
            intersectionCapacity: segments,
            precision: PRECISION,
            output: polygonizeOutput(context, segments, Math.floor(segments / 3), segments)
          })
        );
      }
    })
  ),

  // GPUMakeValid
  ...[false, true].map(
    (spatialSort): KernelAuditCase => ({
      exportName: 'GPUMakeValid',
      name: `every option, spatialSort ${spatialSort}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUMakeValid({
            polygons: {...polygonGeometry(context), sourceIds: view('uint32', ROWS)},
            intersectionCapacity: 4096,
            precision: PRECISION,
            spatialSort,
            leafCapacity: 2048,
            output: polygonizeOutput(context, 4096, 1024, 4096)
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUMakeValid',
      name: excess ? 'vertices above the maximum' : 'at the maximum',
      build: context => {
        const {graph} = context;
        const segments = Math.floor(
          (getGPULineSplitMaximumPieceBound(context.device) - 200000) / 4
        );
        graph.add(
          new GPUMakeValid({
            polygons: polygonGeometry(
              context,
              getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess,
              100000
            ),
            intersectionCapacity: segments,
            precision: PRECISION,
            output: polygonizeOutput(context, segments, Math.floor(segments / 3), segments)
          })
        );
      }
    })
  ),

  // GPUGeometryValidity
  ...(
    [
      ['implicit', 'counter-clockwise-shell'],
      ['explicit', 'clockwise-shell'],
      ['implicit', 'ignore']
    ] as const
  ).map(
    ([ringClosure, orientation]): KernelAuditCase => ({
      exportName: 'GPUGeometryValidity',
      name: `polygons, ${ringClosure} closure, ${orientation}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUGeometryValidity({
            polygons: polygonGeometry(context),
            intersectionCapacity: 4096,
            leafCapacity: 2048,
            ringClosure,
            orientation,
            output: {
              mask: view('uint32', ROWS),
              status: {
                overflow: scalar(context, 'overflow'),
                intersectionCount: scalar(context, 'count')
              }
            }
          })
        );
      }
    })
  ),
  ...(['lines', 'points'] as const).map(
    (kind): KernelAuditCase => ({
      exportName: 'GPUGeometryValidity',
      name: kind,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUGeometryValidity({
            [kind]: joinGeometry(context, kind),
            output: {
              mask: view('uint32', ROWS),
              status: {
                overflow: scalar(context, 'overflow'),
                intersectionCount: scalar(context, 'count')
              }
            }
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUGeometryValidity',
      name: excess ? 'polygons above the maximum' : 'polygons at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess;
        const features = Math.floor(vertices / 8);
        graph.add(
          new GPUGeometryValidity({
            polygons: polygonGeometry(context, vertices, features),
            intersectionCapacity: maximumRows(16),
            ringClosure: 'explicit',
            output: {
              mask: view('uint32', features),
              status: {
                overflow: scalar(context, 'overflow'),
                intersectionCount: scalar(context, 'count')
              }
            }
          })
        );
      }
    })
  ),

  // Scratch-limited sizes: inputs and capacities at which the internal tables still fit a buffer.
  {
    exportName: 'GPUGeometryPredicates',
    name: 'polygons at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = Math.floor(maximumRows(32) / 2);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUGeometryPredicates({
          geometry: polygonGeometry(context, vertices, features),
          other: polygonGeometry(context, vertices, features),
          parameters: view('float32', GPU_GEOMETRY_PREDICATES_PARAMETER_LENGTH),
          intersectionCapacity: maximumRows(64),
          output: {
            isSimple: view('uint32', features),
            isRing: view('uint32', features),
            equalsExact: view('uint32', features),
            equalsIdentical: view('uint32', features),
            uniquePositions: view('float32x2', vertices),
            uniqueOffsets: view('uint32', features + 1),
            status: {
              overflow: scalar(context, 'overflow'),
              intersectionCount: scalar(context, 'count')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeometryValidity',
    name: 'polygons at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = Math.floor(maximumRows(32) / 2);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUGeometryValidity({
          polygons: polygonGeometry(context, vertices, features),
          intersectionCapacity: maximumRows(64),
          ringClosure: 'explicit',
          output: {
            mask: view('uint32', features),
            status: {
              overflow: scalar(context, 'overflow'),
              intersectionCount: scalar(context, 'count')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineNoding',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const capacity = maximumRows(144);
      graph.add(
        new GPULineNoding({
          lines: lineGeometry(context, maximumRows(32), 100000),
          intersectionCapacity: capacity,
          output: nodedSegments(context, capacity)
        })
      );
    }
  },
  {
    exportName: 'GPUPolygonize',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const segments = maximumRows(144);
      graph.add(
        new GPUPolygonize({
          lines: lineGeometry(context, maximumRows(32), 100000),
          intersectionCapacity: segments,
          precision: PRECISION,
          output: polygonizeOutput(context, segments, Math.floor(segments / 3), segments)
        })
      );
    }
  },
  {
    exportName: 'GPUMakeValid',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const segments = maximumRows(144);
      graph.add(
        new GPUMakeValid({
          polygons: polygonGeometry(context, maximumRows(32), 100000),
          intersectionCapacity: segments,
          precision: PRECISION,
          output: polygonizeOutput(context, segments, Math.floor(segments / 3), segments)
        })
      );
    }
  },

  // GPUGridGenerator
  ...(['square', 'hex', 'triangle', 'point'] as const).flatMap((gridType): KernelAuditCase[] =>
    (gridType === 'hex' ? (['pointy', 'flat'] as const) : ([undefined] as const)).map(
      (hexOrientation): KernelAuditCase => ({
        exportName: 'GPUGridGenerator',
        name: `${gridType}${hexOrientation ? ` ${hexOrientation}` : ''}, extent masks`,
        build: context => {
          const {graph, view} = context;
          const columns = 30;
          const rows = 20;
          const cells = getGPUGridCellCount(gridType, columns, rows);
          const cellVertices = getGPUGridVerticesPerCell(gridType);
          const hasCorners = gridType === 'square' || gridType === 'hex';
          const corners = hasCorners ? getGPUGridCornerCount(gridType, columns, rows) : 0;
          graph.add(
            new GPUGridGenerator({
              gridType,
              columns,
              rows,
              hexOrientation,
              extent: {kind: 'polygons', ...polygonFields(context, 64, 4)},
              parameters: view('float32', GPU_GRID_GENERATOR_PARAMETER_LENGTH),
              output: {
                positions:
                  gridType === 'point' ? undefined : view('float32x2', cells * cellVertices),
                centers: view('float32x2', cells),
                corners: hasCorners ? view('float32x2', corners) : undefined,
                cornerIntersects: hasCorners ? view('uint32', corners) : undefined,
                intersects: view('uint32', cells)
              }
            })
          );
        }
      })
    )
  ),
  ...(['square', 'hex', 'triangle', 'point'] as GPUGridType[]).map(
    (gridType): KernelAuditCase => ({
      exportName: 'GPUGridGenerator',
      name: `${gridType} at the binding limit`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const cellVertices = Math.max(getGPUGridVerticesPerCell(gridType), 1);
        const side = Math.floor(Math.sqrt(maximumRows(8) / cellVertices / 2)) - 2;
        const cells = getGPUGridCellCount(gridType, side, side);
        const hasCorners = gridType === 'square' || gridType === 'hex';
        graph.add(
          new GPUGridGenerator({
            gridType,
            columns: side,
            rows: side,
            parameters: view('float32', GPU_GRID_GENERATOR_PARAMETER_LENGTH),
            output: {
              positions: gridType === 'point' ? undefined : view('float32x2', cells * cellVertices),
              centers: view('float32x2', cells),
              corners: hasCorners
                ? view('float32x2', getGPUGridCornerCount(gridType, side, side))
                : undefined
            }
          })
        );
      }
    })
  ),

  // GPUShapeGenerator
  ...(
    [
      ['circle', 'planar', undefined],
      ['circle', 'geodesic', undefined],
      ['sector', 'planar', undefined],
      ['sector', 'geodesic', undefined],
      ['ellipse', 'planar', 'parameter'],
      ['ellipse', 'geodesic', 'arc-length'],
      ['ellipse', 'planar', 'arc-length']
    ] as const
  ).map(
    ([shape, system, ellipseSpacing]): KernelAuditCase => ({
      exportName: 'GPUShapeGenerator',
      name: `${shape} ${system}${ellipseSpacing ? ` ${ellipseSpacing}` : ''}, every input`,
      build: context => {
        const {graph, view} = context;
        const maximumSegments = 256;
        graph.add(
          new GPUShapeGenerator({
            shape,
            spatialContext:
              system === 'geodesic' ? GPU_GREAT_CIRCLE_SPATIAL_CONTEXT : GPU_PLANAR_SPATIAL_CONTEXT,
            ellipseSpacing,
            maximumSegments,
            centers: view('float32x2', ROWS),
            radii: shape === 'ellipse' ? view('float32x2', ROWS) : view('float32', ROWS),
            bearings: shape === 'sector' ? view('float32x2', ROWS) : undefined,
            rotations: shape === 'ellipse' ? view('float32', ROWS) : undefined,
            parameters: view('float32', GPU_SHAPE_GENERATOR_PARAMETER_LENGTH),
            output: {
              positions: view('float32x2', ROWS * getGPUShapeVertexCount(shape, maximumSegments)),
              offsets: view('uint32', ROWS + 1),
              vertexCount: scalar(context, 'vertex-count')
            }
          })
        );
      }
    })
  ),
  ...(['circle', 'sector', 'ellipse'] as const).map(
    (shape): KernelAuditCase => ({
      exportName: 'GPUShapeGenerator',
      name: `${shape} at the maximum segments`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const maximumSegments = 65536;
        const perFeature = getGPUShapeVertexCount(shape, maximumSegments);
        const features = Math.floor(maximumRows(8) / perFeature);
        graph.add(
          new GPUShapeGenerator({
            shape,
            spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
            ellipseSpacing: shape === 'ellipse' ? 'arc-length' : undefined,
            maximumSegments,
            centers: view('float32x2', features),
            radii: shape === 'ellipse' ? view('float32x2', features) : view('float32', features),
            bearings: shape === 'sector' ? view('float32x2', features) : undefined,
            rotations: shape === 'ellipse' ? view('float32', features) : undefined,
            parameters: view('float32', GPU_SHAPE_GENERATOR_PARAMETER_LENGTH),
            output: {
              positions: view('float32x2', features * perFeature),
              offsets: view('uint32', features + 1),
              vertexCount: scalar(context, 'vertex-count')
            }
          })
        );
      }
    })
  ),

  // GPUGroupGeometry
  ...[4096, 2 ** 24 - 1].map(
    (medoidMaximumGroupSize): KernelAuditCase => ({
      exportName: 'GPUGroupGeometry',
      name: `every output, medoid group size ${medoidMaximumGroupSize}`,
      build: context => {
        const {graph, view} = context;
        const groups = 16;
        graph.add(
          new GPUGroupGeometry({
            positions: view('float32x2', ROWS),
            labels: view('uint32', ROWS),
            groupCount: groups,
            noiseLabel: 3,
            weights: view('float32', ROWS),
            parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
            medoidMaximumGroupSize,
            output: {
              counts: view('uint32', groups),
              bounds: view('float32x4', groups),
              meanCenters: view('float32x2', groups),
              weightedCenters: view('float32x2', groups),
              weightSums: view('float32', groups),
              medoidIndices: view('uint32', groups),
              ellipses: view('float32', groups * 3),
              standardDistances: view('float32', groups),
              status: {overflow: scalar(context, 'overflow')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGroupGeometry',
    name: 'rows at the maximum',
    build: context => {
      const {graph, view} = context;
      const rows = 2 ** 24 - 1;
      const groups = 1 << 20;
      graph.add(
        new GPUGroupGeometry({
          positions: view('float32x2', rows),
          labels: view('uint32', rows),
          groupCount: groups,
          weights: view('float32', rows),
          parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
          output: {
            counts: view('uint32', groups),
            bounds: view('float32x4', groups),
            meanCenters: view('float32x2', groups),
            weightedCenters: view('float32x2', groups),
            weightSums: view('float32', groups),
            medoidIndices: view('uint32', groups),
            ellipses: view('float32', groups * 3),
            standardDistances: view('float32', groups),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },

  // GPUGroupConvexHull
  ...[0, 2, 4].map(
    (prefilterLevels): KernelAuditCase => ({
      exportName: 'GPUGroupConvexHull',
      name: `every output, prefilter levels ${prefilterLevels}`,
      build: context => {
        const {graph, view} = context;
        const groups = 16;
        graph.add(
          new GPUGroupConvexHull({
            positions: view('float32x2', ROWS),
            labels: view('uint32', ROWS),
            groupCount: groups,
            noiseLabel: 3,
            maximumVerticesPerGroup: 64,
            vertexCapacity: 1024,
            prefilterLevels,
            output: {
              vertexIndices: view('uint32', 1024),
              vertexPositions: view('float32x2', 1024),
              offsets: view('uint32', groups + 1),
              counts: view('uint32', groups),
              sizes: view('uint32', groups),
              status: {overflow: scalar(context, 'overflow')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGroupConvexHull',
    name: 'rows and capacities at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = 2 ** 24 - 1;
      const groups = 1 << 20;
      const total = maximumRows(8);
      graph.add(
        new GPUGroupConvexHull({
          positions: view('float32x2', rows),
          labels: view('uint32', rows),
          groupCount: groups,
          maximumVerticesPerGroup: 0xffffffff,
          vertexCapacity: total,
          prefilterLevels: 4,
          output: {
            vertexIndices: view('uint32', total),
            vertexPositions: view('float32x2', total),
            offsets: view('uint32', groups + 1),
            counts: view('uint32', groups),
            sizes: view('uint32', groups),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },

  // GPULabelPoint
  {
    exportName: 'GPULabelPoint',
    name: 'every output',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPULabelPoint({
          ...polygonFields(context),
          initialGridSize: 16,
          refinementGridSize: 8,
          refinementRounds: 6,
          refinementCandidates: 16,
          smallFeatureVertexLimit: 32,
          output: {
            points: view('float32x2', ROWS),
            distances: view('float32', ROWS),
            degenerate: view('uint32', ROWS)
          }
        })
      );
    }
  },
  ...(
    [
      ['large grids', 256, 64, 32],
      ['workgroup only', 16, 8, 0],
      ['thread only', 16, 8, 0xffffffff]
    ] as const
  ).map(
    ([label, initialGridSize, refinementGridSize, smallFeatureVertexLimit]): KernelAuditCase => ({
      exportName: 'GPULabelPoint',
      name: label,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULabelPoint({
            ...polygonFields(context),
            initialGridSize,
            refinementGridSize,
            refinementRounds: 64,
            refinementCandidates: 16,
            smallFeatureVertexLimit,
            output: {
              points: view('float32x2', ROWS),
              distances: view('float32', ROWS),
              degenerate: view('uint32', ROWS)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULabelPoint',
    name: 'at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPULabelPoint({
          ...polygonFields(context, vertices, features),
          output: {
            points: view('float32x2', features),
            distances: view('float32', features),
            degenerate: view('uint32', features)
          }
        })
      );
    }
  },

  // GPULineClipByPolygon
  ...(['inside', 'outside'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPULineClipByPolygon',
      name: `${mode}, every option`,
      build: context => {
        const {graph} = context;
        graph.add(
          new GPULineClipByPolygon({
            lines: lineGeometry(context),
            polygons: polygonGeometry(context),
            mode,
            intersectionCapacity: 4096,
            candidateCapacity: 4096,
            spatialSort: true,
            leafCapacity: 2048,
            output: {
              pieces: linePieces(context, 2048, 4096),
              status: {uncertainCount: scalar(context, 'uncertain')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineClipByPolygon',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const capacity = maximumRows(640);
      graph.add(
        new GPULineClipByPolygon({
          lines: lineGeometry(context, maximumRows(128), 100000),
          polygons: polygonGeometry(context, maximumRows(128), 100000),
          intersectionCapacity: capacity,
          candidateCapacity: capacity,
          output: {pieces: linePieces(context, capacity, capacity)}
        })
      );
    }
  },
  ...[
    ['at the maximum', 0, 0, 0],
    ['line vertices above the maximum', 1, 0, 0],
    ['polygon vertices above the maximum', 0, 1, 0],
    ['sub-piece bound above the maximum', 0, 0, 1]
  ].map(
    ([name, excessLines, excessPolygons, excessCapacity]): KernelAuditCase => ({
      exportName: 'GPULineClipByPolygon',
      name: name as string,
      build: context => {
        const {graph} = context;
        const lineVertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) / 2;
        const capacity =
          (getGPULineClipByPolygonMaximumSubPieceBound(context.device) - lineVertices) / 2 +
          (excessCapacity as number);
        graph.add(
          new GPULineClipByPolygon({
            lines: lineGeometry(context, lineVertices + (excessLines as number), 100000),
            polygons: polygonGeometry(
              context,
              getGPUSegmentIntersectionMaximumVertexCount(context.device) +
                (excessPolygons as number),
              100000
            ),
            intersectionCapacity: capacity,
            candidateCapacity: capacity,
            output: {pieces: linePieces(context, capacity, capacity)}
          })
        );
      }
    })
  ),

  // GPUSharedPaths
  {
    exportName: 'GPUSharedPaths',
    name: 'every option',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUSharedPaths({
          left: lineGeometry(context),
          right: lineGeometry(context),
          intersectionCapacity: 4096,
          spatialSort: true,
          leafCapacity: 2048,
          output: {
            runs: {
              leftLineIds: view('uint32', 1024),
              rightLineIds: view('uint32', 1024),
              forward: view('uint32', 1024),
              offsets: view('uint32', 1025),
              positions: view('float32x2', 4096)
            },
            status: {
              count: scalar(context, 'count'),
              vertexCount: scalar(context, 'vertex-count'),
              overflow: scalar(context, 'overflow'),
              requiredCount: scalar(context, 'required-count'),
              requiredVertexCount: scalar(context, 'required-vertex-count'),
              uncertainCount: scalar(context, 'uncertain')
            }
          }
        })
      );
    }
  },
  ...[
    ['at the maximum', 0, 0],
    ['vertices above the maximum', 1, 0],
    ['intersections above the maximum', 0, 1]
  ].map(
    ([name, excessVertices, excessCapacity]): KernelAuditCase => ({
      exportName: 'GPUSharedPaths',
      name: name as string,
      build: context => {
        const {graph, view, maximumRows} = context;
        const runs = maximumRows(16);
        const capacity =
          getGPUSharedPathsMaximumIntersectionCapacity(context.device) + (excessCapacity as number);
        const vertices =
          getGPUSegmentIntersectionMaximumVertexCount(context.device) + (excessVertices as number);
        graph.add(
          new GPUSharedPaths({
            left: lineGeometry(context, vertices, 100000),
            right: lineGeometry(context, vertices, 100000),
            intersectionCapacity: capacity,
            output: {
              runs: {
                leftLineIds: view('uint32', runs),
                rightLineIds: view('uint32', runs),
                forward: view('uint32', runs),
                offsets: view('uint32', runs + 1),
                positions: view('float32x2', capacity)
              },
              status: {
                count: scalar(context, 'count'),
                vertexCount: scalar(context, 'vertex-count'),
                overflow: scalar(context, 'overflow'),
                requiredCount: scalar(context, 'required-count'),
                requiredVertexCount: scalar(context, 'required-vertex-count'),
                uncertainCount: scalar(context, 'uncertain')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUSharedPaths',
    name: 'at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const capacity = maximumRows(144);
      graph.add(
        new GPUSharedPaths({
          left: lineGeometry(context, maximumRows(32), 100000),
          right: lineGeometry(context, maximumRows(32), 100000),
          intersectionCapacity: capacity,
          output: {
            runs: {
              leftLineIds: view('uint32', capacity),
              rightLineIds: view('uint32', capacity),
              forward: view('uint32', capacity),
              offsets: view('uint32', capacity + 1),
              positions: view('float32x2', capacity)
            },
            status: {count: scalar(context, 'count')}
          }
        })
      );
    }
  },

  // GPULineSplit
  ...[false, true].map(
    (spatialSort): KernelAuditCase => ({
      exportName: 'GPULineSplit',
      name: `every option, spatialSort ${spatialSort}`,
      build: context => {
        const {graph} = context;
        graph.add(
          new GPULineSplit({
            lines: lineGeometry(context),
            intersectionCapacity: 4096,
            spatialSort,
            leafCapacity: 2048,
            output: {
              pieces: linePieces(context, 2048, 8192),
              status: {uncertainCount: scalar(context, 'uncertain')}
            }
          })
        );
      }
    })
  ),
  ...[
    ['at the maximum', 0, 0],
    ['vertices above the maximum', 1, 0],
    ['piece bound above the maximum', 0, 1]
  ].map(
    ([name, excessVertices, excessCapacity]): KernelAuditCase => ({
      exportName: 'GPULineSplit',
      name: name as string,
      build: context => {
        const {graph} = context;
        const capacity =
          Math.floor((getGPULineSplitMaximumPieceBound(context.device) - 100000) / 4) +
          (excessCapacity as number);
        graph.add(
          new GPULineSplit({
            lines: lineGeometry(
              context,
              getGPUSegmentIntersectionMaximumVertexCount(context.device) +
                (excessVertices as number),
              100000
            ),
            intersectionCapacity: capacity,
            output: {pieces: linePieces(context, capacity, capacity)}
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineSplit',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const capacity = maximumRows(144);
      graph.add(
        new GPULineSplit({
          lines: lineGeometry(context, maximumRows(32), 100000),
          intersectionCapacity: capacity,
          output: {pieces: linePieces(context, capacity, capacity)}
        })
      );
    }
  },

  // GPULineDensity
  ...(
    [
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT],
      ['spherical', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT]
    ] as const
  ).map(
    ([label, spatialContext]): KernelAuditCase => ({
      exportName: 'GPULineDensity',
      name: `${label}, every output`,
      build: context => {
        const {graph, view} = context;
        const columns = 64;
        const rows = 32;
        graph.add(
          new GPULineDensity({
            ...lineFields(context),
            spatialContext,
            columns,
            rows,
            recordCapacity: 8192,
            parameters: view('float32', GPU_LINE_DENSITY_PARAMETER_LENGTH),
            output: {
              lengths: view('float32', columns * rows),
              densities: view('float32', columns * rows),
              status: {
                overflow: scalar(context, 'overflow'),
                requiredCount: scalar(context, 'required')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineDensity',
    name: 'grid at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const columns = 4097;
      const rows = GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT / columns;
      graph.add(
        new GPULineDensity({
          ...lineFields(context, maximumRows(8), 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          columns,
          rows,
          recordCapacity: maximumRows(16),
          parameters: view('float32', GPU_LINE_DENSITY_PARAMETER_LENGTH),
          output: {
            lengths: view('float32', columns * rows),
            densities: view('float32', columns * rows),
            status: {
              overflow: scalar(context, 'overflow'),
              requiredCount: scalar(context, 'required')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineDensity',
    name: 'grid above the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const columns = 4096;
      const rows = Math.floor((GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT + 1) / columns);
      graph.add(
        new GPULineDensity({
          ...lineFields(context, maximumRows(8), 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          columns,
          rows,
          recordCapacity: maximumRows(16),
          parameters: view('float32', GPU_LINE_DENSITY_PARAMETER_LENGTH),
          output: {
            lengths: view('float32', columns * rows),
            densities: view('float32', columns * rows),
            status: {
              overflow: scalar(context, 'overflow'),
              requiredCount: scalar(context, 'required')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineDensity',
    name: 'grid just below the segment-sum dispatch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const cells = maximumRows(8);
      const columns = 4095;
      const rows = Math.min(4095, Math.floor(cells / columns));
      graph.add(
        new GPULineDensity({
          ...lineFields(context, maximumRows(8), 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          columns,
          rows,
          recordCapacity: maximumRows(16),
          parameters: view('float32', GPU_LINE_DENSITY_PARAMETER_LENGTH),
          output: {
            lengths: view('float32', columns * rows),
            densities: view('float32', columns * rows),
            status: {
              overflow: scalar(context, 'overflow'),
              requiredCount: scalar(context, 'required')
            }
          }
        })
      );
    }
  },

  // GPULineLengthPerPolygon
  ...[64, 1024, 1025, 0xffffffff].map(
    (maximumCrossingCount): KernelAuditCase => ({
      exportName: 'GPULineLengthPerPolygon',
      name: `every output, maximum crossings ${maximumCrossingCount}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineLengthPerPolygon({
            ...lineFields(context),
            lineWeights: view('float32', ROWS),
            polygons: polygonGeometry(context),
            spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
            pairCapacity: 8192,
            maximumCrossingCount,
            output: {
              lengths: view('float32', ROWS),
              weightedLengths: view('float32', ROWS),
              segmentCounts: view('uint32', ROWS),
              status: {overflow: scalar(context, 'overflow')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'positions and pairs at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const features = 100000;
      graph.add(
        new GPULineLengthPerPolygon({
          ...lineFields(
            context,
            getGPULineLengthPerPolygonMaximumPositionCount(context.device),
            100000
          ),
          lineWeights: view('float32', 100000),
          polygons: polygonGeometry(context, maximumRows(8), features),
          pairCapacity: getGPULineLengthPerPolygonMaximumPairCapacity(context.device),
          output: {
            lengths: view('float32', features),
            weightedLengths: view('float32', features),
            segmentCounts: view('uint32', features),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'positions above the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const features = 100000;
      graph.add(
        new GPULineLengthPerPolygon({
          ...lineFields(
            context,
            getGPULineLengthPerPolygonMaximumPositionCount(context.device) + 1,
            100000
          ),
          lineWeights: view('float32', 100000),
          polygons: polygonGeometry(context, maximumRows(8), features),
          pairCapacity: 8192,
          output: {
            lengths: view('float32', features),
            weightedLengths: view('float32', features),
            segmentCounts: view('uint32', features),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'pairs above the maximum',
    build: context => {
      const {graph, view} = context;
      const features = 100000;
      graph.add(
        new GPULineLengthPerPolygon({
          ...lineFields(context, 4096, 100),
          lineWeights: view('float32', 100),
          polygons: polygonGeometry(context, 4096, features),
          pairCapacity: getGPULineLengthPerPolygonMaximumPairCapacity(context.device) + 1,
          output: {
            lengths: view('float32', features),
            weightedLengths: view('float32', features),
            segmentCounts: view('uint32', features),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'pairs at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const features = 100000;
      graph.add(
        new GPULineLengthPerPolygon({
          ...lineFields(context, maximumRows(32), 100000),
          lineWeights: view('float32', 100000),
          polygons: polygonGeometry(context, maximumRows(32), features),
          pairCapacity: maximumRows(64),
          output: {
            lengths: view('float32', features),
            weightedLengths: view('float32', features),
            segmentCounts: view('uint32', features),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },

  // GPULineMerge
  ...[false, true].map(
    (directed): KernelAuditCase => ({
      exportName: 'GPULineMerge',
      name: `every output, directed ${directed}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineMerge({
            ...lineFields(context),
            directed,
            output: {
              chainOffsets: view('uint32', ROWS + 1),
              positions: view('float32x2', VERTICES),
              status: {count: scalar(context, 'count')},
              lineChains: view('uint32', ROWS),
              lineOrders: view('uint32', ROWS),
              lineReversed: view('uint32', ROWS)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineMerge',
    name: 'at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const lines = maximumRows(8);
      graph.add(
        new GPULineMerge({
          ...lineFields(context, lines, lines - 1),
          output: {
            chainOffsets: view('uint32', lines),
            positions: view('float32x2', lines),
            status: {count: scalar(context, 'count')},
            lineChains: view('uint32', lines - 1),
            lineOrders: view('uint32', lines - 1),
            lineReversed: view('uint32', lines - 1)
          }
        })
      );
    }
  },

  // GPULineSegmentize
  ...([GPU_PLANAR_SPATIAL_CONTEXT, GPU_GREAT_CIRCLE_SPATIAL_CONTEXT] as const).map(
    (spatialContext): KernelAuditCase => ({
      exportName: 'GPULineSegmentize',
      name: `closed ${spatialContext.coordinateSpace}, every output`,
      build: context => {
        context.graph.add(
          new GPULineSegmentize({
            ...lineFields(context),
            spatialContext,
            closedLines: true,
            parameters: context.view('float32', GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH),
            output: lineGeometryOutput(context, ROWS, 65536)
          })
        );
      }
    })
  ),
  ...(['linear-degrees', 'great-circle'] as const).map(
    (interpolation): KernelAuditCase => ({
      exportName: 'GPULineSegmentize',
      name: `closed degree-step ${interpolation}, integer turns and approximation`,
      build: context => {
        const {graph, view} = context;
        const fields = {
          ...lineFields(context),
          operation: 'GPULineSegmentize',
          coordinateSystem: 'spherical' as const,
          radius: 1,
          closedLines: true
        };
        const counts = view('uint32', VERTICES);
        const starts = view('uint32', VERTICES);
        const rowTurns = view('sint32', VERTICES);
        graph.add(createLinePrefixNode(graph, {...fields, id: 'degree-prefix', rowTurns}));
        graph.add(
          createSegmentizeCountNode(graph, {
            ...fields,
            id: 'degree-count',
            counts,
            parameters: view('float32', 4),
            parameterIndex: 2,
            pieceCountModel: 'degree-step',
            interpolation,
            maximumPieceCount: 65536,
            approximation: view('uint32', 1)
          })
        );
        graph.add(
          createSegmentizeEmitNode(graph, {
            ...fields,
            id: 'degree-emit',
            counts,
            starts,
            rowTurns,
            interpolation,
            wrapOutputLongitudes: true,
            outputPositions: view('float32x2', 65536),
            outputTurns: view('sint32', 65536),
            outputSourceRows: view('uint32', 65536)
          })
        );
      }
    })
  ),
  ...(
    [
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT],
      ['great-circle', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT]
    ] as const
  ).map(
    ([label, spatialContext]): KernelAuditCase => ({
      exportName: 'GPULineSegmentize',
      name: `${label}, every output, maximum piece count`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineSegmentize({
            ...lineFields(context),
            spatialContext,
            maximumPieceCount: Math.floor(0xffffffff / VERTICES),
            parameters: view('float32', GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH),
            output: lineGeometryOutput(context, ROWS, 65536)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineSegmentize',
    name: 'vertices at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      graph.add(
        new GPULineSegmentize({
          ...lineFields(context, vertices, 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          maximumPieceCount: Math.floor(0xffffffff / vertices),
          parameters: view('float32', GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, 100000, maximumRows(16))
        })
      );
    }
  },

  // GPUGreatCircleArcs
  {
    exportName: 'GPUGreatCircleArcs',
    name: 'every output, maximum segment count',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUGreatCircleArcs({
          sources: view('float32x2', ROWS),
          targets: view('float32x2', ROWS),
          maximumSegmentCount: Math.floor(0xffffffff / ROWS) - 1,
          parameters: view('float32', GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, ROWS, 65536)
        })
      );
    }
  },
  {
    exportName: 'GPUGreatCircleArcs',
    name: 'pairs at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const pairs = maximumRows(16);
      graph.add(
        new GPUGreatCircleArcs({
          sources: view('float32x2', pairs),
          targets: view('float32x2', pairs),
          maximumSegmentCount: Math.floor(0xffffffff / pairs) - 1,
          parameters: view('float32', GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, pairs, maximumRows(16))
        })
      );
    }
  },

  // GPULineSmooth
  ...[false, true].map(
    (closed): KernelAuditCase => ({
      exportName: 'GPULineSmooth',
      name: `maximum iterations, closed ${closed}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineSmooth({
            ...lineFields(context),
            iterationCount: GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS,
            closed,
            parameters: view('float32', GPU_LINE_SMOOTH_PARAMETER_LENGTH),
            output: lineGeometryOutput(context, ROWS, 1 << 20, {})
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineSmooth',
    name: 'vertices at the 32-bit limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = Math.floor((0xffffffff - 100000) / 2 ** GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS);
      graph.add(
        new GPULineSmooth({
          ...lineFields(context, vertices, 100000),
          iterationCount: GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS,
          closed: true,
          parameters: view('float32', GPU_LINE_SMOOTH_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, 100000, maximumRows(8), {})
        })
      );
    }
  },

  // GPULineChunk
  ...(
    [
      ['chunk', GPU_PLANAR_SPATIAL_CONTEXT],
      ['chunk', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT],
      ['substring', GPU_PLANAR_SPATIAL_CONTEXT],
      ['substring', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT]
    ] as const
  ).map(
    ([mode, spatialContext]): KernelAuditCase => ({
      exportName: 'GPULineChunk',
      name: `${mode}, ${spatialContext.coordinateSpace}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineChunk({
            ...lineFields(context),
            mode,
            spatialContext,
            parameters: view('float32', GPU_LINE_CHUNK_PARAMETER_LENGTH),
            output: lineGeometryOutput(context, mode === 'chunk' ? 4096 : ROWS, 8192)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineChunk',
    name: 'chunk at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertexCapacity = maximumRows(8);
      graph.add(
        new GPULineChunk({
          ...lineFields(context, maximumRows(8), 100000),
          mode: 'chunk',
          parameters: view('float32', GPU_LINE_CHUNK_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, Math.floor(0xffffffff / 100000) - 2, vertexCapacity)
        })
      );
    }
  },

  // GPULineSimplification
  ...(
    [
      ['douglas-peucker segment', 'douglas-peucker', 'segment'],
      ['douglas-peucker time-ratio', 'douglas-peucker', 'time-ratio'],
      ['visvalingam', 'visvalingam', 'segment']
    ] as const
  ).flatMap(([label, method, metric]): KernelAuditCase[] =>
    [GPU_PLANAR_SPATIAL_CONTEXT, GPU_GREAT_CIRCLE_SPATIAL_CONTEXT]
      .filter(spatialContext => method !== 'visvalingam' || spatialContext.metric === 'native')
      .map(
        (spatialContext): KernelAuditCase => ({
          exportName: 'GPULineSimplification',
          name: `${label}, ${spatialContext.coordinateSpace}, every output`,
          build: context => {
            const {graph, view} = context;
            graph.add(
              new GPULineSimplification({
                ...lineFields(context),
                spatialContext,
                method,
                metric,
                timestamps: metric === 'time-ratio' ? view('float32', VERTICES) : undefined,
                neighborhoodRadius:
                  method === 'visvalingam'
                    ? GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS
                    : undefined,
                importance: view('float32', VERTICES),
                computeImportance: true,
                maximumIterations: GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
                finishSpanLimit:
                  method === 'douglas-peucker' ? MAXIMUM_FINISH_SPAN_LIMIT : undefined,
                parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
                output: {
                  ids: view('uint32', VERTICES),
                  status: {
                    ...countStatus(context),
                    converged: scalar(context, 'converged'),
                    iterationCount: scalar(context, 'iterations')
                  },
                  outputMask: view('uint32', VERTICES),
                  lineCounts: view('uint32', ROWS),
                  lineStarts: view('uint32', ROWS)
                }
              })
            );
          }
        })
      )
  ),
  {
    exportName: 'GPULineSimplification',
    name: 'selection only, no finish span',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPULineSimplification({
          ...lineFields(context),
          importance: view('float32', VERTICES),
          computeImportance: false,
          finishSpanLimit: 0,
          parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
          output: {
            ids: view('uint32', VERTICES),
            status: countStatus(context),
            outputMask: view('uint32', VERTICES),
            lineCounts: view('uint32', ROWS),
            lineStarts: view('uint32', ROWS)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineSimplification',
    name: 'vertices at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      graph.add(
        new GPULineSimplification({
          ...lineFields(context, vertices, 100000),
          method: 'visvalingam',
          importance: view('float32', vertices),
          maximumIterations: GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
          neighborhoodRadius: GPU_LINE_SIMPLIFICATION_MAXIMUM_NEIGHBORHOOD_RADIUS,
          parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
          output: {
            ids: view('uint32', vertices),
            status: countStatus(context),
            outputMask: view('uint32', vertices),
            lineCounts: view('uint32', 100000),
            lineStarts: view('uint32', 100000)
          }
        })
      );
    }
  },

  // GPULinearReferencing
  ...(
    [
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT],
      ['great-circle', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT]
    ] as const
  ).map(
    ([label, spatialContext]): KernelAuditCase => ({
      exportName: 'GPULinearReferencing',
      name: `${label}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULinearReferencing({
            ...lineFields(context),
            spatialContext,
            queryPositions: view('float32x2', ROWS),
            radius: view('float32', 1),
            candidateCapacity: 8192,
            leafCapacity: 2048,
            spatialSort: true,
            output: {
              lineIndices: view('uint32', ROWS),
              segmentIndices: view('uint32', ROWS),
              fractions: view('float32', ROWS),
              footPoints: view('float32x2', ROWS),
              distances: view('float32', ROWS),
              measures: view('float32', ROWS),
              normalizedMeasures: view('float32', ROWS),
              sides: view('sint32', ROWS),
              signedOffsets: view('float32', ROWS),
              vertexMeasures: view('float32', VERTICES),
              status: {
                overflow: scalar(context, 'overflow'),
                candidateCount: scalar(context, 'candidates')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULinearReferencing',
    name: 'planar queries at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const queries = getGPULinearReferencingMaximumQueryCount(context.device);
      graph.add(
        new GPULinearReferencing({
          ...lineFields(
            context,
            getGPULinearReferencingMaximumPlanarPositionCount(context.device),
            100000
          ),
          queryPositions: view('float32x2', queries),
          radius: view('float32', 1),
          candidateCapacity: maximumRows(8),
          output: {
            lineIndices: view('uint32', queries),
            segmentIndices: view('uint32', queries),
            fractions: view('float32', queries),
            footPoints: view('float32x2', queries),
            distances: view('float32', queries),
            measures: view('float32', queries),
            normalizedMeasures: view('float32', queries),
            sides: view('sint32', queries),
            signedOffsets: view('float32', queries),
            status: {
              overflow: scalar(context, 'overflow'),
              candidateCount: scalar(context, 'candidates')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'planar positions above the maximum',
    build: context => {
      const {graph, view} = context;
      const queries = ROWS;
      graph.add(
        new GPULinearReferencing({
          ...lineFields(
            context,
            getGPULinearReferencingMaximumPlanarPositionCount(context.device) + 1,
            100000
          ),
          queryPositions: view('float32x2', queries),
          radius: view('float32', 1),
          candidateCapacity: 8192,
          output: {
            lineIndices: view('uint32', queries),
            segmentIndices: view('uint32', queries),
            fractions: view('float32', queries),
            footPoints: view('float32x2', queries),
            distances: view('float32', queries),
            measures: view('float32', queries),
            normalizedMeasures: view('float32', queries),
            sides: view('sint32', queries),
            signedOffsets: view('float32', queries),
            status: {
              overflow: scalar(context, 'overflow'),
              candidateCount: scalar(context, 'candidates')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'planar queries above the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const queries = getGPULinearReferencingMaximumQueryCount(context.device) + 1;
      graph.add(
        new GPULinearReferencing({
          ...lineFields(
            context,
            getGPULinearReferencingMaximumPlanarPositionCount(context.device),
            100000
          ),
          queryPositions: view('float32x2', queries),
          radius: view('float32', 1),
          candidateCapacity: maximumRows(8),
          output: {
            lineIndices: view('uint32', queries),
            segmentIndices: view('uint32', queries),
            fractions: view('float32', queries),
            footPoints: view('float32x2', queries),
            distances: view('float32', queries),
            measures: view('float32', queries),
            normalizedMeasures: view('float32', queries),
            sides: view('sint32', queries),
            signedOffsets: view('float32', queries),
            status: {
              overflow: scalar(context, 'overflow'),
              candidateCount: scalar(context, 'candidates')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'great-circle positions at the maximum',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPULinearReferencing({
          ...lineFields(
            context,
            getGPULinearReferencingMaximumSphericalPositionCount(context.device),
            100000
          ),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          queryPositions: view('float32x2', ROWS),
          radius: view('float32', 1),
          output: {
            lineIndices: view('uint32', ROWS),
            segmentIndices: view('uint32', ROWS),
            fractions: view('float32', ROWS),
            footPoints: view('float32x2', ROWS),
            distances: view('float32', ROWS),
            measures: view('float32', ROWS),
            normalizedMeasures: view('float32', ROWS),
            sides: view('sint32', ROWS),
            signedOffsets: view('float32', ROWS),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'great-circle positions above the maximum',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPULinearReferencing({
          ...lineFields(
            context,
            getGPULinearReferencingMaximumSphericalPositionCount(context.device) + 1,
            100000
          ),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          queryPositions: view('float32x2', ROWS),
          radius: view('float32', 1),
          output: {
            lineIndices: view('uint32', ROWS),
            segmentIndices: view('uint32', ROWS),
            fractions: view('float32', ROWS),
            footPoints: view('float32x2', ROWS),
            distances: view('float32', ROWS),
            measures: view('float32', ROWS),
            normalizedMeasures: view('float32', ROWS),
            sides: view('sint32', ROWS),
            signedOffsets: view('float32', ROWS),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'planar at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const queries = maximumRows(64);
      graph.add(
        new GPULinearReferencing({
          ...lineFields(context, maximumRows(64), 100000),
          queryPositions: view('float32x2', queries),
          radius: view('float32', 1),
          candidateCapacity: maximumRows(64),
          output: {
            lineIndices: view('uint32', queries),
            segmentIndices: view('uint32', queries),
            fractions: view('float32', queries),
            footPoints: view('float32x2', queries),
            distances: view('float32', queries),
            measures: view('float32', queries),
            normalizedMeasures: view('float32', queries),
            sides: view('sint32', queries),
            signedOffsets: view('float32', queries),
            status: {
              overflow: scalar(context, 'overflow'),
              candidateCount: scalar(context, 'candidates')
            }
          }
        })
      );
    }
  },

  // GPULineLocate
  ...(
    [
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT, 'distance'],
      ['planar', GPU_PLANAR_SPATIAL_CONTEXT, 'fraction'],
      ['great-circle', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT, 'distance'],
      ['great-circle', GPU_GREAT_CIRCLE_SPATIAL_CONTEXT, 'fraction']
    ] as const
  ).map(
    ([label, spatialContext, measureMode]): KernelAuditCase => ({
      exportName: 'GPULineLocate',
      name: `${label}, ${measureMode}, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPULineLocate({
            ...lineFields(context),
            spatialContext,
            eventLineIndices: view('uint32', ROWS),
            eventMeasures: view('float32', ROWS),
            eventOffsets: view('float32', ROWS),
            measureMode,
            parameters: view('float32', GPU_LINE_LOCATE_PARAMETER_LENGTH),
            output: {
              positions: view('float32x2', ROWS),
              segmentIndices: view('uint32', ROWS),
              tangents: view('float32x2', ROWS),
              angles: view('float32', ROWS),
              statuses: view('uint32', ROWS),
              vertexMeasures: view('float32', VERTICES)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULineLocate',
    name: 'events above the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const events = getGPULineLocateMaximumEventCount(context.device) + 1;
      graph.add(
        new GPULineLocate({
          ...lineFields(context, maximumRows(8), 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          eventLineIndices: view('uint32', events),
          eventMeasures: view('float32', events),
          eventOffsets: view('float32', events),
          parameters: view('float32', GPU_LINE_LOCATE_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', events),
            segmentIndices: view('uint32', events),
            tangents: view('float32x2', events),
            angles: view('float32', events),
            statuses: view('uint32', events),
            vertexMeasures: view('float32', maximumRows(8))
          }
        })
      );
    }
  },
  {
    exportName: 'GPULineLocate',
    name: 'events at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const events = getGPULineLocateMaximumEventCount(context.device);
      graph.add(
        new GPULineLocate({
          ...lineFields(context, maximumRows(8), 100000),
          spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
          eventLineIndices: view('uint32', events),
          eventMeasures: view('float32', events),
          eventOffsets: view('float32', events),
          parameters: view('float32', GPU_LINE_LOCATE_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', events),
            segmentIndices: view('uint32', events),
            tangents: view('float32x2', events),
            angles: view('float32', events),
            statuses: view('uint32', events),
            vertexMeasures: view('float32', maximumRows(8))
          }
        })
      );
    }
  },

  // GPUMapColoring
  {
    exportName: 'GPUMapColoring',
    name: 'every output',
    build: context => {
      const {graph, view, weights} = context;
      graph.add(
        new GPUMapColoring({
          weights: weights(ROWS, 8),
          parameters: view('uint32', GPU_MAP_COLORING_PARAMETER_LENGTH),
          maximumIterations: 1024,
          output: {
            colors: view('uint32', ROWS),
            colorCount: scalar(context, 'color-count'),
            conflictCount: scalar(context, 'conflict-count'),
            status: {
              converged: scalar(context, 'converged'),
              iterationCount: scalar(context, 'iterations')
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUMapColoring',
    name: 'rows at the binding limit',
    build: context => {
      const {graph, view, weights, maximumRows} = context;
      const rows = Math.floor(maximumRows(4) / 9);
      graph.add(
        new GPUMapColoring({
          weights: weights(rows, 8),
          parameters: view('uint32', GPU_MAP_COLORING_PARAMETER_LENGTH),
          maximumIterations: 1024,
          output: {
            colors: view('uint32', rows),
            colorCount: scalar(context, 'color-count'),
            conflictCount: scalar(context, 'conflict-count'),
            status: {
              converged: scalar(context, 'converged'),
              iterationCount: scalar(context, 'iterations')
            }
          }
        })
      );
    }
  },

  // GPUOffsetExpansion
  {
    exportName: 'GPUOffsetExpansion',
    name: 'every output with owner map',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUOffsetExpansion({
          offsets: view('uint32', ROWS + 1),
          ownerMap: view('uint32', ROWS),
          output: {
            owners: view('uint32', VERTICES),
            localIndex: view('uint32', VERTICES),
            status: countStatus(context)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUOffsetExpansion',
    name: 'owners and children at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const owners = maximumRows(4);
      const children = maximumRows(4);
      graph.add(
        new GPUOffsetExpansion({
          offsets: view('uint32', owners),
          ownerMap: view('uint32', owners - 1),
          output: {
            owners: view('uint32', children),
            localIndex: view('uint32', children),
            status: countStatus(context)
          }
        })
      );
    }
  },

  // GPUOutlineGeometry
  ...(['points', 'lines', 'polygons'] as const).flatMap((geometryType): KernelAuditCase[] =>
    [GPU_PLANAR_SPATIAL_CONTEXT, GPU_GREAT_CIRCLE_SPATIAL_CONTEXT].map(
      (spatialContext): KernelAuditCase => ({
        exportName: 'GPUOutlineGeometry',
        name: `${geometryType}, ${spatialContext.coordinateSpace}`,
        build: context => {
          const {graph, view} = context;
          const joinSegments = 64;
          const output = {
            positions: view(
              'float32x2',
              VERTICES * getGPUOutlineGeometryVerticesPerInput(joinSegments)
            )
          };
          const parameters = view('float32', GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH);
          if (geometryType === 'points') {
            graph.add(
              new GPUOutlineGeometry({
                geometryType,
                positions: view('float32x2', VERTICES),
                spatialContext,
                joinSegments,
                parameters,
                output
              })
            );
          } else if (geometryType === 'lines') {
            graph.add(
              new GPUOutlineGeometry({
                geometryType,
                ...lineFields(context),
                spatialContext,
                joinSegments,
                parameters,
                output
              })
            );
          } else {
            graph.add(
              new GPUOutlineGeometry({
                geometryType,
                ...polygonFields(context),
                spatialContext,
                joinSegments,
                parameters,
                output
              })
            );
          }
        }
      })
    )
  ),
  {
    exportName: 'GPUOutlineGeometry',
    name: 'one point with the largest join',
    build: context => {
      const {graph, view, maximumRows} = context;
      const joinSegments = Math.floor(maximumRows(8) / 3) - 2;
      graph.add(
        new GPUOutlineGeometry({
          geometryType: 'points',
          positions: view('float32x2', 1),
          joinSegments,
          parameters: view('float32', GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', getGPUOutlineGeometryVerticesPerInput(joinSegments))
          }
        })
      );
    }
  },
  {
    exportName: 'GPUOutlineGeometry',
    name: 'polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const joinSegments = 16;
      const k = getGPUOutlineGeometryVerticesPerInput(joinSegments);
      const vertices = Math.floor(maximumRows(8) / k);
      graph.add(
        new GPUOutlineGeometry({
          geometryType: 'polygons',
          ...polygonFields(context, vertices, Math.floor(vertices / 8)),
          joinSegments,
          parameters: view('float32', GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH),
          output: {positions: view('float32x2', vertices * k)}
        })
      );
    }
  },

  // GPUOffsetCurve
  ...(['round', 'mitre', 'bevel'] as const).flatMap((joinStyle): KernelAuditCase[] =>
    (['intersection', 'buffer'] as const).flatMap((insideTurns): KernelAuditCase[] =>
      (['lines', 'polygons'] as const).map(
        (geometryType): KernelAuditCase => ({
          exportName: 'GPUOffsetCurve',
          name: `${geometryType}, ${joinStyle}, ${insideTurns}`,
          build: context => {
            const {graph, view} = context;
            const quadSegments = 32;
            const k = getGPUOffsetCurveRowsPerVertex(joinStyle, quadSegments, insideTurns);
            const common = {
              joinStyle,
              quadSegments,
              insideTurns,
              parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
              output: {positions: view('float32x2', VERTICES * k), counts: view('uint32', VERTICES)}
            };
            if (geometryType === 'lines') {
              graph.add(new GPUOffsetCurve({geometryType, ...lineFields(context), ...common}));
            } else {
              graph.add(new GPUOffsetCurve({geometryType, ...polygonFields(context), ...common}));
            }
          }
        })
      )
    )
  ),
  {
    exportName: 'GPUOffsetCurve',
    name: 'one vertex with the largest round join',
    build: context => {
      const {graph, view} = context;
      const quadSegments = GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS;
      graph.add(
        new GPUOffsetCurve({
          positions: view('float32x2', 1),
          lineOffsets: view('uint32', 2),
          joinStyle: 'round',
          quadSegments,
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', getGPUOffsetCurveRowsPerVertex('round', quadSegments)),
            counts: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUOffsetCurve',
    name: 'one vertex with quadSegments above the maximum',
    build: context => {
      const {graph, view} = context;
      const quadSegments = GPU_OFFSET_CURVE_MAXIMUM_QUAD_SEGMENTS + 1;
      graph.add(
        new GPUOffsetCurve({
          positions: view('float32x2', 1),
          lineOffsets: view('uint32', 2),
          joinStyle: 'round',
          quadSegments,
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', getGPUOffsetCurveRowsPerVertex('round', quadSegments)),
            counts: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUOffsetCurve',
    name: 'polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const quadSegments = 8;
      const k = getGPUOffsetCurveRowsPerVertex('round', quadSegments, 'buffer');
      const vertices = Math.floor(maximumRows(8) / k);
      graph.add(
        new GPUOffsetCurve({
          geometryType: 'polygons',
          ...polygonFields(context, vertices, Math.floor(vertices / 8)),
          quadSegments,
          insideTurns: 'buffer',
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          output: {positions: view('float32x2', vertices * k), counts: view('uint32', vertices)}
        })
      );
    }
  },

  // GPUBufferSurface
  ...(
    [
      ['lines', 'round', 'round'],
      ['lines', 'mitre', 'flat'],
      ['lines', 'bevel', 'square'],
      ['polygons', 'round', 'round'],
      ['polygons', 'mitre', 'flat'],
      ['polygons', 'bevel', 'square']
    ] as const
  ).map(
    ([geometryType, joinStyle, capStyle]): KernelAuditCase => ({
      exportName: 'GPUBufferSurface',
      name: `${geometryType}, ${joinStyle} joins, ${capStyle} caps`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUBufferSurface({
            geometry: geometryType === 'lines' ? lineGeometry(context) : polygonGeometry(context),
            parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
            joinStyle: joinStyle as GPUOffsetCurveJoinStyle,
            capStyle,
            quadSegments: 16,
            vertexTolerance: 1e-5,
            capacity: {intersections: 4096, nodedSegments: 4096},
            spatialSort: true,
            leafCapacity: 4096,
            output: overlayOutput(context, 4096, 1024, 4096)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUBufferSurface',
    name: 'polygons at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const capacity = maximumRows(768);
      graph.add(
        new GPUBufferSurface({
          // Raw rows 35 per vertex plus 3 per ring (two rings per feature) must fit the quarter table.
          geometry: polygonGeometry(
            context,
            Math.floor(
              (getGPUBufferSurfaceMaximumBoundaryRowCount(context.device) - capacity - 600000) / 35
            ),
            100000
          ),
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          vertexTolerance: 1e-5,
          capacity: {intersections: capacity, nodedSegments: capacity},
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'polygons at the maximum',
    build: context => {
      const {graph, view} = context;
      const capacity = 1 << 18;
      graph.add(
        new GPUBufferSurface({
          // Raw rows: (1 + 2 rowsPerVertex) per vertex and 3 per ring (two rings per feature).
          geometry: polygonGeometry(
            context,
            Math.floor(
              (getGPUBufferSurfaceMaximumBoundaryRowCount(context.device) - capacity - 6 * 1000) /
                (1 + 2 * getGPUOffsetCurveRowsPerVertex('round', 8, 'buffer'))
            ),
            1000
          ),
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          vertexTolerance: 1e-5,
          capacity: {intersections: capacity, nodedSegments: capacity},
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'polygons above the maximum',
    build: context => {
      const {graph, view} = context;
      const capacity = 1 << 18;
      graph.add(
        new GPUBufferSurface({
          geometry: polygonGeometry(
            context,
            Math.floor(
              (getGPUBufferSurfaceMaximumBoundaryRowCount(context.device) - capacity - 6 * 1000) /
                (1 + 2 * getGPUOffsetCurveRowsPerVertex('round', 8, 'buffer'))
            ) + 1,
            1000
          ),
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          vertexTolerance: 1e-5,
          capacity: {intersections: capacity, nodedSegments: capacity},
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'lines at the maximum',
    build: context => {
      const {graph, view} = context;
      const capacity = 1 << 18;
      const lines = 1000;
      // Round caps add 2 * quadSegments - 1 rows at each end of every line, plus a closing vertex.
      const vertices = Math.floor(
        (getGPUBufferSurfaceMaximumBoundaryRowCount(context.device) -
          capacity -
          (2 * 15 + 1) * lines) /
          (2 * getGPUOffsetCurveRowsPerVertex('round', 8, 'buffer'))
      );
      graph.add(
        new GPUBufferSurface({
          geometry: lineGeometry(context, vertices, lines),
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          vertexTolerance: 1e-5,
          capacity: {intersections: capacity, nodedSegments: capacity},
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'lines above the maximum',
    build: context => {
      const {graph, view} = context;
      const capacity = 1 << 18;
      const lines = 1000;
      // Round caps add 2 * quadSegments - 1 rows at each end of every line, plus a closing vertex.
      const vertices =
        Math.floor(
          (getGPUBufferSurfaceMaximumBoundaryRowCount(context.device) -
            capacity -
            (2 * 15 + 1) * lines) /
            (2 * getGPUOffsetCurveRowsPerVertex('round', 8, 'buffer'))
        ) + 1;
      graph.add(
        new GPUBufferSurface({
          geometry: lineGeometry(context, vertices, lines),
          parameters: view('float32', GPU_OFFSET_CURVE_PARAMETER_LENGTH),
          vertexTolerance: 1e-5,
          capacity: {intersections: capacity, nodedSegments: capacity},
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },

  // GPUPolygonOverlay
  ...(
    [
      'intersection',
      'union',
      'difference',
      'symmetric-difference',
      'dissolve'
    ] as GPUPolygonOverlayOperation[]
  ).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPUPolygonOverlay',
      name: `${operation}, every option`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUPolygonOverlay({
            left: {...polygonGeometry(context), sourceIds: view('uint32', ROWS)},
            right:
              operation === 'dissolve'
                ? undefined
                : {...polygonGeometry(context), sourceIds: view('uint32', ROWS)},
            operation,
            fillRule: 'even-odd',
            capacity: {intersections: 4096, nodedSegments: 4096},
            vertexTolerance: 1e-5,
            spatialSort: true,
            leafCapacity: 2048,
            output: overlayOutput(context, 4096, 1024, 4096)
          })
        );
      }
    })
  ),
  ...(['positive', 'negative', 'nonzero'] as GPUPolygonOverlayFillRule[]).map(
    (fillRule): KernelAuditCase => ({
      exportName: 'GPUPolygonOverlay',
      name: `union, ${fillRule} fill rule`,
      build: context => {
        const {graph} = context;
        graph.add(
          new GPUPolygonOverlay({
            left: polygonGeometry(context),
            right: polygonGeometry(context),
            operation: 'union',
            fillRule,
            capacity: {intersections: 4096, nodedSegments: 4096},
            vertexTolerance: 1e-5,
            output: overlayOutput(context, 4096, 1024, 4096)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUPolygonOverlay',
    name: 'at the scratch limit',
    build: context => {
      const {graph, maximumRows} = context;
      const capacity = maximumRows(144);
      // Snap rounding bounds hot pixels (vertices + crossings) by a quarter of the segment table.
      const vertices = getOverlayMaximumOperandVertices(context, OVERLAY_AUDIT_INTERSECTIONS);
      graph.add(
        new GPUPolygonOverlay({
          left: polygonGeometry(context, vertices, OVERLAY_AUDIT_FEATURES),
          right: polygonGeometry(context, vertices, OVERLAY_AUDIT_FEATURES),
          operation: 'symmetric-difference',
          capacity: {intersections: OVERLAY_AUDIT_INTERSECTIONS, nodedSegments: capacity},
          vertexTolerance: 1e-5,
          output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
        })
      );
    }
  },
  ...[
    ['at the maximum', 0, 0],
    ['vertices above the maximum', 1, 0],
    ['hot pixels above the maximum', 0, 1]
  ].map(
    ([name, excessVertices, excessCapacity]): KernelAuditCase => ({
      exportName: 'GPUPolygonOverlay',
      name: name as string,
      build: context => {
        const {graph} = context;
        const capacity = OVERLAY_AUDIT_INTERSECTIONS + (excessCapacity as number);
        // Vertices fitting the segment table, or one more than it holds (rejected first).
        const vertices = excessVertices
          ? getGPUSegmentIntersectionMaximumVertexCount(context.device) / 2
          : getOverlayMaximumOperandVertices(context, OVERLAY_AUDIT_INTERSECTIONS);
        graph.add(
          new GPUPolygonOverlay({
            left: polygonGeometry(
              context,
              vertices + (excessVertices as number),
              OVERLAY_AUDIT_FEATURES
            ),
            right: polygonGeometry(context, vertices, OVERLAY_AUDIT_FEATURES),
            operation: 'symmetric-difference',
            capacity: {intersections: capacity, nodedSegments: capacity},
            vertexTolerance: 1e-5,
            output: overlayOutput(context, capacity, Math.floor(capacity / 3), capacity)
          })
        );
      }
    })
  ),

  // GPUCoverageSimplification
  ...[true, false].map(
    (simplifyBoundary): KernelAuditCase => ({
      exportName: 'GPUCoverageSimplification',
      name: `every output, simplifyBoundary ${simplifyBoundary}`,
      build: context => {
        const {graph, view} = context;
        const fields = polygonFields(context);
        graph.add(
          new GPUCoverageSimplification({
            ...fields,
            snapTolerance: 1e-4,
            parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
            simplifyBoundary,
            maximumIterations: GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS,
            topologyIterationCount: 16,
            topologyPairCapacity: 8192,
            output: {
              positions: view('float32x2', VERTICES),
              ringOffsets: view('uint32', fields.ringOffsets.length),
              outputMask: view('uint32', VERTICES),
              topologyStatistics: view('uint32', 4),
              status: {
                converged: scalar(context, 'converged'),
                overflow: scalar(context, 'overflow'),
                requiredCount: scalar(context, 'required')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCoverageSimplification',
    name: 'no topology repair',
    build: context => {
      const {graph, view} = context;
      const fields = polygonFields(context);
      graph.add(
        new GPUCoverageSimplification({
          ...fields,
          parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
          topologyIterationCount: 0,
          output: {
            positions: view('float32x2', VERTICES),
            ringOffsets: view('uint32', fields.ringOffsets.length),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUCoverageSimplification',
      name: excess ? 'vertices above the maximum' : 'vertices at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess;
        const fields = polygonFields(context, vertices, Math.floor(vertices / 8));
        graph.add(
          new GPUCoverageSimplification({
            ...fields,
            parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
            topologyIterationCount: 16,
            topologyPairCapacity: maximumRows(16),
            output: {
              positions: view('float32x2', vertices),
              ringOffsets: view('uint32', fields.ringOffsets.length),
              outputMask: view('uint32', vertices),
              topologyStatistics: view('uint32', 4),
              status: {
                converged: scalar(context, 'converged'),
                overflow: scalar(context, 'overflow'),
                requiredCount: scalar(context, 'required')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCoverageSimplification',
    name: 'vertices at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(32);
      const fields = polygonFields(context, vertices, Math.floor(vertices / 8));
      graph.add(
        new GPUCoverageSimplification({
          ...fields,
          parameters: view('float32', GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH),
          topologyIterationCount: 16,
          topologyPairCapacity: maximumRows(16),
          output: {
            positions: view('float32x2', vertices),
            ringOffsets: view('uint32', fields.ringOffsets.length),
            outputMask: view('uint32', vertices),
            topologyStatistics: view('uint32', 4),
            status: {
              converged: scalar(context, 'converged'),
              overflow: scalar(context, 'overflow'),
              requiredCount: scalar(context, 'required')
            }
          }
        })
      );
    }
  },

  // GPUCoverageDissolve
  ...[true, false].map(
    (splitTouchingRings): KernelAuditCase => ({
      exportName: 'GPUCoverageDissolve',
      name: `every output, splitTouchingRings ${splitTouchingRings}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUCoverageDissolve({
            ...polygonFields(context),
            labels: view('uint32', ROWS),
            snapTolerance: 1e-4,
            vertexTolerance: 1e-5,
            splitTouchingRings,
            output: {
              ringOffsets: view('uint32', 101),
              positions: view('float32x2', 4096),
              ringAreas: view('float32', 100),
              ringIsHole: view('uint32', 100),
              ringShells: view('uint32', 100),
              ringGroups: view('uint32', 100),
              segmentRings: view('uint32', VERTICES),
              segmentVertices: view('uint32', VERTICES),
              segmentFlags: view('uint32', VERTICES),
              polygons: {
                kind: 'polygons',
                positions: view('float32x2', 4096),
                ringOffsets: view('uint32', 101),
                polygonOffsets: view('uint32', 101),
                featureOffsets: view('uint32', 101),
                sourceIds: view('uint32', 100)
              },
              status: {
                ...countStatus(context),
                openSegmentCount: scalar(context, 'open'),
                touchingSegmentCount: scalar(context, 'touching'),
                boundarySegmentCount: scalar(context, 'boundary')
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCoverageDissolve',
    name: 'vertices at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(16);
      const rings = Math.floor(vertices / 3);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUCoverageDissolve({
          ...polygonFields(context, vertices, features),
          labels: view('uint32', features),
          output: {
            ringOffsets: view('uint32', rings + 1),
            positions: view('float32x2', vertices),
            ringAreas: view('float32', rings),
            ringIsHole: view('uint32', rings),
            ringShells: view('uint32', rings),
            ringGroups: view('uint32', rings),
            segmentRings: view('uint32', vertices),
            segmentVertices: view('uint32', vertices),
            segmentFlags: view('uint32', vertices),
            status: {...countStatus(context), boundarySegmentCount: scalar(context, 'boundary')}
          }
        })
      );
    }
  },

  // GPUCoverageValidity
  ...[false, true].map(
    (spatialSort): KernelAuditCase => ({
      exportName: 'GPUCoverageValidity',
      name: `every output, spatialSort ${spatialSort}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUCoverageValidity({
            ...polygonFields(context),
            parameters: view('float32', GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH),
            spatialSort,
            leafCapacity: 2048,
            output: {
              segmentFlags: view('uint32', VERTICES),
              featureInvalidCounts: view('uint32', ROWS),
              isValid: scalar(context, 'is-valid'),
              status: {invalidCount: scalar(context, 'invalid')}
            }
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUCoverageValidity',
      name: excess ? 'vertices above the maximum' : 'vertices at the maximum',
      build: context => {
        const {graph, view} = context;
        const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess;
        const features = Math.floor(vertices / 8);
        graph.add(
          new GPUCoverageValidity({
            ...polygonFields(context, vertices, features),
            parameters: view('float32', GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH),
            output: {
              segmentFlags: view('uint32', vertices),
              featureInvalidCounts: view('uint32', features),
              isValid: scalar(context, 'is-valid'),
              status: {invalidCount: scalar(context, 'invalid')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCoverageValidity',
    name: 'vertices at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(32);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUCoverageValidity({
          ...polygonFields(context, vertices, features),
          parameters: view('float32', GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH),
          output: {
            segmentFlags: view('uint32', vertices),
            featureInvalidCounts: view('uint32', features),
            isValid: scalar(context, 'is-valid'),
            status: {invalidCount: scalar(context, 'invalid')}
          }
        })
      );
    }
  },

  // GPUPolygonTriangulation
  ...[true, false].map(
    (useZOrderHash): KernelAuditCase => ({
      exportName: 'GPUPolygonTriangulation',
      name: `every output, z-order hash ${useZOrderHash}`,
      build: context => {
        const {graph, view} = context;
        const fields = polygonFields(context);
        const indices = getPolygonTriangulationIndexCount(
          fields.positions.length,
          fields.ringOffsets.length - 1,
          fields.polygonOffsets.length - 1
        );
        graph.add(
          new GPUPolygonTriangulation({
            ...fields,
            maximumWork: 0xffffffff,
            useZOrderHash,
            output: {
              indices: view('uint32', indices),
              valid: view('uint32', ROWS),
              triangleCount: view('uint32', ROWS)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices at the maximum',
    build: context => {
      const {graph, view} = context;
      const vertices = getTriangulationVertexCount(
        getGPUPolygonTriangulationMaximumNodeCount(context.device),
        false
      );
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUPolygonTriangulation({
          ...fields,
          maximumWork: 0xffffffff,
          output: {
            indices: view(
              'uint32',
              getPolygonTriangulationIndexCount(
                vertices,
                fields.ringOffsets.length - 1,
                fields.polygonOffsets.length - 1
              )
            ),
            valid: view('uint32', features),
            triangleCount: view('uint32', features)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices above the maximum',
    build: context => {
      const {graph, view} = context;
      const vertices = getTriangulationVertexCount(
        getGPUPolygonTriangulationMaximumNodeCount(context.device),
        true
      );
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUPolygonTriangulation({
          ...fields,
          maximumWork: 0xffffffff,
          output: {
            indices: view(
              'uint32',
              getPolygonTriangulationIndexCount(
                vertices,
                fields.ringOffsets.length - 1,
                fields.polygonOffsets.length - 1
              )
            ),
            valid: view('uint32', features),
            triangleCount: view('uint32', features)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices at the maximum without z-order hash',
    build: context => {
      const {graph, view} = context;
      const vertices = getTriangulationVertexCount(
        getGPUPolygonTriangulationMaximumNodeCount(context.device, false),
        false
      );
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUPolygonTriangulation({
          ...fields,
          maximumWork: 0xffffffff,
          useZOrderHash: false,
          output: {
            indices: view(
              'uint32',
              getPolygonTriangulationIndexCount(
                vertices,
                fields.ringOffsets.length - 1,
                fields.polygonOffsets.length - 1
              )
            ),
            valid: view('uint32', features),
            triangleCount: view('uint32', features)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices above the maximum without z-order hash',
    build: context => {
      const {graph, view} = context;
      const vertices = getTriangulationVertexCount(
        getGPUPolygonTriangulationMaximumNodeCount(context.device, false),
        true
      );
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUPolygonTriangulation({
          ...fields,
          maximumWork: 0xffffffff,
          useZOrderHash: false,
          output: {
            indices: view(
              'uint32',
              getPolygonTriangulationIndexCount(
                vertices,
                fields.ringOffsets.length - 1,
                fields.polygonOffsets.length - 1
              )
            ),
            valid: view('uint32', features),
            triangleCount: view('uint32', features)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = Math.floor(maximumRows(112));
      const features = Math.floor(vertices / 8);
      const fields = polygonFields(context, vertices, features);
      graph.add(
        new GPUPolygonTriangulation({
          ...fields,
          maximumWork: 0xffffffff,
          output: {
            indices: view(
              'uint32',
              getPolygonTriangulationIndexCount(
                vertices,
                fields.ringOffsets.length - 1,
                fields.polygonOffsets.length - 1
              )
            ),
            valid: view('uint32', features),
            triangleCount: view('uint32', features)
          }
        })
      );
    }
  },

  // GPURectangleClip
  {
    exportName: 'GPURectangleClip',
    name: 'lines, every output',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPURectangleClip({
          geometryType: 'lines',
          ...lineFields(context),
          parameters: view('float32', GPU_RECTANGLE_CLIP_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, 2048, 4096, {sourceRows: false, measures: false})
        })
      );
    }
  },
  {
    exportName: 'GPURectangleClip',
    name: 'polygons',
    build: context => {
      const {graph, view} = context;
      const fields = polygonFields(context);
      graph.add(
        new GPURectangleClip({
          geometryType: 'polygons',
          ...fields,
          parameters: view('float32', GPU_RECTANGLE_CLIP_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', 4096),
            lineOffsets: view('uint32', fields.ringOffsets.length),
            status: {...countStatus(context), lineCount: scalar(context, 'line-count')}
          }
        })
      );
    }
  },
  {
    exportName: 'GPURectangleClip',
    name: 'lines at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const lineCapacity = maximumRows(8) / 4;
      graph.add(
        new GPURectangleClip({
          geometryType: 'lines',
          ...lineFields(context, maximumRows(8), 100000),
          parameters: view('float32', GPU_RECTANGLE_CLIP_PARAMETER_LENGTH),
          output: lineGeometryOutput(context, lineCapacity, maximumRows(8), {
            sourceRows: false,
            measures: false
          })
        })
      );
    }
  },
  {
    exportName: 'GPURectangleClip',
    name: 'polygons at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const vertices = maximumRows(8);
      const fields = polygonFields(context, vertices, Math.floor(vertices / 8));
      graph.add(
        new GPURectangleClip({
          geometryType: 'polygons',
          ...fields,
          parameters: view('float32', GPU_RECTANGLE_CLIP_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', vertices),
            lineOffsets: view('uint32', fields.ringOffsets.length),
            status: countStatus(context)
          }
        })
      );
    }
  },

  // GPUSegmentIntersection
  ...(
    [
      ['lines', 'lines'],
      ['polygons', 'polygons'],
      ['lines', 'polygons'],
      ['polygons', undefined],
      ['lines', undefined]
    ] as const
  ).flatMap(([leftKind, rightKind]): KernelAuditCase[] =>
    [false, true].map(
      (spatialSort): KernelAuditCase => ({
        exportName: 'GPUSegmentIntersection',
        name: `${leftKind} x ${rightKind ?? 'self'}, spatialSort ${spatialSort}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUSegmentIntersection({
              left: segmentGeometry(context, leftKind),
              right: rightKind ? segmentGeometry(context, rightKind) : undefined,
              sameFeatureOnly: rightKind === undefined,
              spatialSort,
              leafCapacity: 2048,
              output: {
                pairs: {
                  leftIds: view('uint32', 4096),
                  rightIds: view('uint32', 4096),
                  status: {
                    count: scalar(context, 'count'),
                    overflow: scalar(context, 'overflow'),
                    requiredCount: scalar(context, 'required'),
                    candidateOverflow: scalar(context, 'candidate-overflow')
                  }
                },
                kinds: view('uint32', 4096),
                points: view('float32x2', 4096),
                endPoints: view('float32x2', 4096),
                leftFeatures: view('uint32', 4096),
                rightFeatures: view('uint32', 4096),
                leftRings: view('uint32', 4096),
                rightRings: view('uint32', 4096),
                status: {uncertainCount: scalar(context, 'uncertain')}
              }
            })
          );
        }
      })
    )
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUSegmentIntersection',
      name: excess ? 'vertices above the maximum' : 'pairs and vertices at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const pairs = maximumRows(8);
        const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + excess;
        graph.add(
          new GPUSegmentIntersection({
            left: polygonGeometry(context, vertices, 100000),
            right: polygonGeometry(context, vertices, 100000),
            output: {
              pairs: {
                leftIds: view('uint32', pairs),
                rightIds: view('uint32', pairs),
                status: {
                  count: scalar(context, 'count'),
                  overflow: scalar(context, 'overflow'),
                  requiredCount: scalar(context, 'required'),
                  candidateOverflow: scalar(context, 'candidate-overflow')
                }
              },
              kinds: view('uint32', pairs),
              points: view('float32x2', pairs),
              endPoints: view('float32x2', pairs),
              leftFeatures: view('uint32', pairs),
              rightFeatures: view('uint32', pairs),
              leftRings: view('uint32', pairs),
              rightRings: view('uint32', pairs),
              status: {uncertainCount: scalar(context, 'uncertain')}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUSegmentIntersection',
    name: 'pairs at the scratch limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const pairs = maximumRows(32);
      graph.add(
        new GPUSegmentIntersection({
          left: polygonGeometry(context, maximumRows(32), 100000),
          right: polygonGeometry(context, maximumRows(32), 100000),
          output: {
            pairs: {
              leftIds: view('uint32', pairs),
              rightIds: view('uint32', pairs),
              status: {
                count: scalar(context, 'count'),
                overflow: scalar(context, 'overflow'),
                requiredCount: scalar(context, 'required'),
                candidateOverflow: scalar(context, 'candidate-overflow')
              }
            },
            kinds: view('uint32', pairs),
            points: view('float32x2', pairs),
            endPoints: view('float32x2', pairs),
            leftFeatures: view('uint32', pairs),
            rightFeatures: view('uint32', pairs),
            leftRings: view('uint32', pairs),
            rightRings: view('uint32', pairs)
          }
        })
      );
    }
  },

  // GPUShapeDescriptors
  ...(['auto', 'gift-wrapping', 'monotone-chain'] as const).map(
    (convexityMethod): KernelAuditCase => ({
      exportName: 'GPUShapeDescriptors',
      name: `every output, ${convexityMethod}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUShapeDescriptors({
            ...polygonFields(context),
            convexityMethod,
            parameters: view('float32', GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH),
            output: {
              areas: view('float32', ROWS),
              perimeters: view('float32', ROWS),
              polsbyPopper: view('float32', ROWS),
              schwartzberg: view('float32', ROWS),
              elongation: view('float32', ROWS),
              orientation: view('float32', ROWS),
              convexity: view('float32', ROWS),
              clockwise: view('uint32', ROWS),
              sliver: view('uint32', ROWS)
            }
          })
        );
      }
    })
  ),
  ...(['gift-wrapping', 'monotone-chain'] as const).map(
    (convexityMethod): KernelAuditCase => ({
      exportName: 'GPUShapeDescriptors',
      name: `${convexityMethod} at the maximum size`,
      build: context => {
        const {graph, view} = context;
        const vertices = 2 ** 24 - 2;
        const features = Math.floor(vertices / 8);
        graph.add(
          new GPUShapeDescriptors({
            ...polygonFields(context, vertices, features),
            convexityMethod,
            parameters: view('float32', GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH),
            output: {
              areas: view('float32', features),
              perimeters: view('float32', features),
              polsbyPopper: view('float32', features),
              schwartzberg: view('float32', features),
              elongation: view('float32', features),
              orientation: view('float32', features),
              convexity: view('float32', features),
              clockwise: view('uint32', features),
              sliver: view('uint32', features)
            }
          })
        );
      }
    })
  ),

  // GPUVertexSnap
  ...[false, true].map(
    (pairwise): KernelAuditCase => ({
      exportName: 'GPUVertexSnap',
      name: pairwise ? 'pairwise, every output' : 'global, every output',
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUVertexSnap({
            positions: view('float32x2', VERTICES),
            referencePositions: view('float32x2', 500),
            featureOffsets: pairwise ? view('uint32', ROWS + 1) : undefined,
            referenceOffsets: pairwise ? view('uint32', ROWS + 1) : undefined,
            parameters: view('float32', GPU_VERTEX_SNAP_PARAMETER_LENGTH),
            output: {
              positions: view('float32x2', VERTICES),
              referenceRows: view('uint32', VERTICES)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUVertexSnap',
    name: 'pairwise at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      graph.add(
        new GPUVertexSnap({
          positions: view('float32x2', rows),
          referencePositions: view('float32x2', rows),
          featureOffsets: view('uint32', Math.floor(rows / 8)),
          referenceOffsets: view('uint32', Math.floor(rows / 8)),
          parameters: view('float32', GPU_VERTEX_SNAP_PARAMETER_LENGTH),
          output: {positions: view('float32x2', rows), referenceRows: view('uint32', rows)}
        })
      );
    }
  },

  // GPUMinimumBounds
  ...(['polygons', 'labelled points'] as const).flatMap((kind): KernelAuditCase[] =>
    [16, 256].map(
      (maximumHullVertices): KernelAuditCase => ({
        exportName: 'GPUMinimumBounds',
        name: `${kind}, every output, hulls up to ${maximumHullVertices}`,
        build: context => {
          const {graph, view} = context;
          const groups = ROWS;
          graph.add(
            new GPUMinimumBounds({
              ...(kind === 'polygons'
                ? polygonFields(context)
                : {
                    positions: view('float32x2', VERTICES),
                    labels: view('uint32', VERTICES),
                    groupCount: groups
                  }),
              maximumHullVertices,
              output: {
                rectangleCorners: view('float32x2', 4 * groups),
                rectangleSizes: view('float32x4', groups),
                circles: view('float32x3', groups),
                longestLines: view('float32x4', groups),
                diameters: view('float32', groups),
                bounds: view('float32x4', groups),
                hullSizes: view('uint32', groups),
                status: {overflow: scalar(context, 'overflow')}
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUMinimumBounds',
    name: 'labelled points at the maximum size and hull',
    build: context => {
      const {graph, view} = context;
      const rows = 2 ** 24 - 1;
      const groups = 1 << 20;
      graph.add(
        new GPUMinimumBounds({
          positions: view('float32x2', rows),
          labels: view('uint32', rows),
          groupCount: groups,
          maximumHullVertices: 0xffffffff,
          output: {
            rectangleCorners: view('float32x2', 4 * groups),
            rectangleSizes: view('float32x4', groups),
            circles: view('float32x3', groups),
            longestLines: view('float32x4', groups),
            diameters: view('float32', groups),
            bounds: view('float32x4', groups),
            hullSizes: view('uint32', groups),
            status: {overflow: scalar(context, 'overflow')}
          }
        })
      );
    }
  },

  // GPUMinimumClearance
  ...(['lines', 'polygons'] as const).flatMap((kind): KernelAuditCase[] =>
    [false, true].map(
      (spatialSort): KernelAuditCase => ({
        exportName: 'GPUMinimumClearance',
        name: `${kind}, every output, spatialSort ${spatialSort}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUMinimumClearance({
              geometry: segmentGeometry(context, kind),
              leafCapacity: 2048,
              spatialSort,
              output: {
                clearances: view('float32', ROWS),
                lines: view('float32x4', ROWS),
                vertexIds: view('uint32', ROWS)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUMinimumClearance',
    name: 'polygons above the maximum',
    build: context => {
      const {graph, view} = context;
      const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device) + 1;
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUMinimumClearance({
          geometry: polygonGeometry(context, vertices, features),
          output: {
            clearances: view('float32', features),
            lines: view('float32x4', features),
            vertexIds: view('uint32', features)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUMinimumClearance',
    name: 'polygons at the maximum',
    build: context => {
      const {graph, view} = context;
      const vertices = getGPUSegmentIntersectionMaximumVertexCount(context.device);
      const features = Math.floor(vertices / 8);
      graph.add(
        new GPUMinimumClearance({
          geometry: polygonGeometry(context, vertices, features),
          output: {
            clearances: view('float32', features),
            lines: view('float32x4', features),
            vertexIds: view('uint32', features)
          }
        })
      );
    }
  }
];

// Internal analysis-core consumer check, including the maximum position binding and optional ports.
GEOMETRY_CASES.push(
  ...[0, 1, 2, 3].map(
    (mask): KernelAuditCase => ({
      exportName: 'GPUPointInPolygonJoin',
      name: `lon-lat consumer check status mask ${mask}, positions at the binding limit`,
      build: context => {
        context.graph.add(
          createLonLatPolygonCheckNode(context.graph, {
            id: 'lon-lat-check',
            operation: 'GPUPointInPolygonJoin',
            positions: context.view('float32x2', context.maximumRows(8)),
            ringOffsets: context.view('uint32', ROWS + 1),
            status: {
              ...(mask & 1 ? {antimeridianUncertainCount: scalar(context, 'crossing')} : {}),
              ...(mask & 2 ? {poleUncertainCount: scalar(context, 'pole')} : {})
            }
          })
        );
      }
    })
  )
);

// Preparation builds every internal kernel, both edge models, all clipping stages and optional ports.
GEOMETRY_CASES.push(
  ...(['linear', 'great-circle'] as const).flatMap(edgeModel =>
    [false, true].map(
      (allOutputs): KernelAuditCase => ({
        exportName: 'GPULonLatPolygonPreparation',
        name: `${edgeModel}, optional outputs ${allOutputs}`,
        build: ({graph, view}) => {
          graph.add(
            new GPULonLatPolygonPreparation({
              positions: view('float32x2', 12),
              featureOffsets: view('uint32', 3),
              polygonOffsets: view('uint32', 4),
              ringOffsets: view('uint32', 5),
              parameters: view('float32', 4),
              spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel},
              maximumPieceCount: 4,
              antimeridian: allOutputs ? 'reject' : 'split',
              poleRings: allOutputs ? 'reject' : 'cap',
              output: {
                positions: view('float32x2', 64),
                featureOffsets: view('uint32', 3),
                polygonOffsets: view('uint32', 7),
                ringOffsets: view('uint32', 9),
                ...(allOutputs ? {ringFlags: view('uint32', 4)} : {}),
                status: {
                  overflow: view('uint32', 1),
                  ...(allOutputs
                    ? {
                        count: view('uint32', 1),
                        requiredCount: view('uint32', 1),
                        invalidCount: view('uint32', 1),
                        approximation: view('uint32', 1)
                      }
                    : {})
                }
              }
            })
          );
        }
      })
    )
  )
);
