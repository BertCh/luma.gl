// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// H3 boundary construction is derived from Uber's Apache-2.0 H3 (faceijk.c).

/**
 * WGSL port of H3's `_faceIjkToCellBoundary` and `_faceIjkPentToCellBoundary` (H3 C
 * `faceijk.c`), complete for every H3 cell: hexagons and pentagons, cells that cross icosahedron
 * edges (face overage), and the extra "distortion" vertices H3 inserts where a Class III cell
 * edge crosses an icosahedron edge. Boundaries have 6 to 10 vertices in the order of h3-js
 * `cellToBoundary`, as `(longitude, latitude)` degrees.
 *
 * Concatenate AFTER `dggs.source` from `@luma.gl/shadertools`. It reuses the dggs face/IJK helpers
 * (`dggs_h3_get_center_face_ijk`, `dggs_h3_get_face_neighbor`, `dggs_h3_get_vertex_offset`,
 * `dggs_h3_down_ap3`, `dggs_h3_down_ap3r`, `dggs_h3_down_ap7r`, `dggs_h3_hex2d_to_lnglat`) and
 * defines, all prefixed `cellIndexH3Boundary` or `CELL_INDEX_H3_BOUNDARY_`:
 *
 * - `struct CellIndexH3Boundary {count: u32, points: array<vec2f, 10>}`
 * - `fn cellIndexH3GetBoundary(cell: vec2u) -> CellIndexH3Boundary` (one pass, all vertices)
 * - `fn cellIndexH3GetBoundaryVertexCount(cell: vec2u) -> u32` (0 for invalid cells)
 * - `fn cellIndexH3GetBoundaryVertex(cell: vec2u, vertexIndex: u32) -> vec2f`
 *
 * `cell` is the canonical `vec2u(high, low)` H3 index. The per-vertex functions recompute the whole
 * boundary, so prefer `cellIndexH3GetBoundary` when reading several vertices.
 *
 * Precision is f32: the hex2d coordinates of the finest resolutions exceed the f32 mantissa
 * (resolution 14 and 15), so vertices there carry errors of a fraction of a cell.
 */
export const H3_BOUNDARY_WGSL = /* wgsl */ `
const CELL_INDEX_H3_BOUNDARY_MAX_VERTICES: u32 = 10u;
const CELL_INDEX_H3_BOUNDARY_NO_OVERAGE: u32 = 0u;
const CELL_INDEX_H3_BOUNDARY_FACE_EDGE: u32 = 1u;
const CELL_INDEX_H3_BOUNDARY_NEW_FACE: u32 = 2u;

struct CellIndexH3Boundary {
  count : u32,
  points : array<vec2f, 10>,
};

// H3 _adjustOverageClassII, including the substrate-grid variant used for boundary vertices.
fn cellIndexH3BoundaryAdjustOverage(
  input: DggsH3FaceIJK,
  resolution: u32,
  pentagonLeading4: bool,
  substrate: bool
) -> DggsH3FaceIJKOverage {
  var faceIJK = input;
  let unscaledDimension = dggs_h3_get_max_dim_by_cii_resolution(resolution);
  var maximumDimension = unscaledDimension;
  if (substrate) {
    maximumDimension *= 3;
  }
  let dimension = faceIJK.coord.x + faceIJK.coord.y + faceIJK.coord.z;
  if (substrate && dimension == maximumDimension) {
    return DggsH3FaceIJKOverage(faceIJK, CELL_INDEX_H3_BOUNDARY_FACE_EDGE);
  }
  if (dimension <= maximumDimension) {
    return DggsH3FaceIJKOverage(faceIJK, CELL_INDEX_H3_BOUNDARY_NO_OVERAGE);
  }

  var quadrant = 1u;
  if (faceIJK.coord.z > 0) {
    quadrant = select(2u, 3u, faceIJK.coord.y > 0);
    if (quadrant == 2u && pentagonLeading4) {
      let origin = vec3i(maximumDimension, 0, 0);
      faceIJK.coord = dggs_h3_ijk_rotate_60_cw(faceIJK.coord - origin) + origin;
    }
  }
  let neighbor = dggs_h3_get_face_neighbor(faceIJK.face, quadrant);
  faceIJK.face = neighbor.face;
  for (var rotation = 0u; rotation < neighbor.valid; rotation++) {
    faceIJK.coord = dggs_h3_ijk_rotate_60_ccw(faceIJK.coord);
  }
  var unitScale = unscaledDimension / 2;
  if (substrate) {
    unitScale *= 3;
  }
  faceIJK.coord = dggs_h3_ijk_normalize(faceIJK.coord + neighbor.coord * unitScale);
  var overage = CELL_INDEX_H3_BOUNDARY_NEW_FACE;
  if (substrate && faceIJK.coord.x + faceIJK.coord.y + faceIJK.coord.z == maximumDimension) {
    overage = CELL_INDEX_H3_BOUNDARY_FACE_EDGE;
  }
  return DggsH3FaceIJKOverage(faceIJK, overage);
}

// Quadrant (1 = IJ, 2 = KI, 3 = JK) of the edge of face \`fromFace\` that borders \`toFace\`, or 0.
fn cellIndexH3BoundaryGetFaceDirection(fromFace: u32, toFace: u32) -> u32 {
  for (var quadrant = 1u; quadrant <= 3u; quadrant++) {
    if (dggs_h3_get_face_neighbor(fromFace, quadrant).face == toFace) {
      return quadrant;
    }
  }
  return 0u;
}

// Intersection of the segment p0-p1 with the line through edge0-edge1 (H3 _v2dIntersect).
fn cellIndexH3BoundaryIntersect(p0: vec2f, p1: vec2f, edge0: vec2f, edge1: vec2f) -> vec2f {
  let s1 = p1 - p0;
  let s2 = edge1 - edge0;
  let t = (s2.x * (p0.y - edge0.y) - s2.y * (p0.x - edge0.x)) / (-s2.x * s1.y + s1.x * s2.y);
  return p0 + t * s1;
}

// Intersection of the hex2d segment p0-p1 with the face edge in \`quadrant\` of an icosahedron face
// at the given adjusted (Class II) resolution.
fn cellIndexH3BoundaryIntersectFaceEdge(
  p0: vec2f,
  p1: vec2f,
  adjustedResolution: u32,
  quadrant: u32
) -> vec2f {
  let maximumDimension = f32(dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution));
  let v0 = vec2f(3.0 * maximumDimension, 0.0);
  let v1 = vec2f(-1.5 * maximumDimension, 3.0 * DGGS_H3_SQRT3_2 * maximumDimension);
  let v2 = vec2f(-1.5 * maximumDimension, -3.0 * DGGS_H3_SQRT3_2 * maximumDimension);
  if (quadrant == 1u) {
    return cellIndexH3BoundaryIntersect(p0, p1, v0, v1);
  }
  if (quadrant == 3u) {
    return cellIndexH3BoundaryIntersect(p0, p1, v1, v2);
  }
  return cellIndexH3BoundaryIntersect(p0, p1, v2, v0);
}

// True when a substrate coordinate lies exactly on an icosahedron face edge.
fn cellIndexH3BoundaryIsOnFaceEdge(coord: vec3i, adjustedResolution: u32) -> bool {
  return coord.x + coord.y + coord.z == 3 * dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution);
}

// Substrate hex2d point on \`face\` to lng/lat degrees (H3 _hex2dToGeo with substrate = 1). Projects
// through the face's unit-vector basis instead of dggs_h3_hex2d_to_lnglat, whose sin/cos/atan2
// azimuth-distance chain loses about 1e-4 degrees in f32.
fn cellIndexH3BoundaryHex2dToLngLat(point: vec2f, face: u32, adjustedResolution: u32) -> vec2f {
  var scale = DGGS_H3_RES0_U_GNOMONIC * DGGS_H3_ONETHIRD;
  for (var level = 0u; level < adjustedResolution; level++) {
    scale *= DGGS_H3_RSQRT7;
  }
  // \`adjustedResolution\` is always Class II here, so no extra aperture-7 factor applies.
  let unitVector = normalize(
    dggs_h3_get_face_unit_vector_basis(face, 0u) +
    scale * (
      point.x * dggs_h3_get_face_unit_vector_basis(face, 1u) +
      point.y * dggs_h3_get_face_unit_vector_basis(face, 2u)
    )
  );
  return vec2f(
    atan2(unitVector.y, unitVector.x) * DGGS_RADIANS_TO_DEGREES,
    atan2(unitVector.z, length(unitVector.xy)) * DGGS_RADIANS_TO_DEGREES
  );
}

fn cellIndexH3BoundaryPush(boundary: ptr<function, CellIndexH3Boundary>, point: vec2f) {
  if ((*boundary).count < CELL_INDEX_H3_BOUNDARY_MAX_VERTICES) {
    (*boundary).points[(*boundary).count] = point;
    (*boundary).count += 1u;
  }
}

// H3 _faceIjkToCellBoundary / _faceIjkPentToCellBoundary for the whole loop. \`center\` is the
// overage-adjusted face/IJK of the cell (dggs_h3_get_center_face_ijk).
fn cellIndexH3BoundaryFromFaceIJK(
  center: DggsH3FaceIJK,
  resolution: u32,
  isPentagon: bool
) -> CellIndexH3Boundary {
  var boundary: CellIndexH3Boundary;
  let isClassIII = dggs_h3_is_resolution_class_iii(resolution);
  let vertexTotal = select(6u, 5u, isPentagon);

  // _faceIjkToVerts: vertices on the aperture 33r (7r for Class III) substrate grid.
  var adjustedResolution = resolution;
  var centerCoord = dggs_h3_down_ap3r(dggs_h3_down_ap3(center.coord));
  if (isClassIII) {
    centerCoord = dggs_h3_down_ap7r(centerCoord);
    adjustedResolution += 1u;
  }
  var vertexCoords: array<vec3i, 6>;
  for (var vertex = 0u; vertex < vertexTotal; vertex++) {
    vertexCoords[vertex] = dggs_h3_ijk_normalize(
      centerCoord + dggs_h3_get_vertex_offset(resolution, vertex)
    );
  }

  var lastFace = 0xffffffffu;
  var lastOverage = CELL_INDEX_H3_BOUNDARY_NO_OVERAGE;
  var lastCoord = vec3i(0);
  // One extra iteration tests the last edge for a distortion vertex.
  for (var vert = 0u; vert <= vertexTotal; vert++) {
    let vertex = vert % vertexTotal;
    var adjusted = cellIndexH3BoundaryAdjustOverage(
      DggsH3FaceIJK(center.face, vertexCoords[vertex], 1u),
      adjustedResolution,
      false,
      true
    );
    if (isPentagon) {
      // _adjustPentVertOverage: repeat until the vertex settles on a face.
      for (var settlePass = 0u; settlePass < 4u && adjusted.overage == CELL_INDEX_H3_BOUNDARY_NEW_FACE; settlePass++) {
        adjusted = cellIndexH3BoundaryAdjustOverage(
          adjusted.faceIJK,
          adjustedResolution,
          false,
          true
        );
      }
    }
    let current = adjusted.faceIJK;

    if (isClassIII && vert > 0u) {
      if (isPentagon) {
        // All Class III pentagon edges cross an icosahedron edge: re-express the current vertex on
        // the previous vertex's face and intersect with the edge between the two faces.
        let currentToLast = cellIndexH3BoundaryGetFaceDirection(current.face, lastFace);
        let lastToCurrent = cellIndexH3BoundaryGetFaceDirection(lastFace, current.face);
        if (currentToLast != 0u && lastToCurrent != 0u) {
          let orient = dggs_h3_get_face_neighbor(current.face, currentToLast);
          var transformed = current.coord;
          for (var rotation = 0u; rotation < orient.valid; rotation++) {
            transformed = dggs_h3_ijk_rotate_60_ccw(transformed);
          }
          let unitScale = (dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution) / 2) * 3;
          transformed = dggs_h3_ijk_normalize(transformed + orient.coord * unitScale);
          let point = cellIndexH3BoundaryIntersectFaceEdge(
            dggs_h3_ijk_to_hex2d(lastCoord),
            dggs_h3_ijk_to_hex2d(transformed),
            adjustedResolution,
            lastToCurrent
          );
          cellIndexH3BoundaryPush(
            &boundary,
            cellIndexH3BoundaryHex2dToLngLat(point, lastFace, adjustedResolution)
          );
        }
      } else if (current.face != lastFace && lastOverage != CELL_INDEX_H3_BOUNDARY_FACE_EDGE) {
        // The crossed edge borders whichever of the two faces is not the center face.
        let neighborFace = select(lastFace, current.face, lastFace == center.face);
        let quadrant = cellIndexH3BoundaryGetFaceDirection(center.face, neighborFace);
        let lastVertex = (vertex + 5u) % 6u;
        // An intersection exactly at a hexagon vertex needs no extra vertex.
        let isAtVertex =
          cellIndexH3BoundaryIsOnFaceEdge(vertexCoords[vertex], adjustedResolution) ||
          cellIndexH3BoundaryIsOnFaceEdge(vertexCoords[lastVertex], adjustedResolution);
        if (quadrant != 0u && !isAtVertex) {
          let point = cellIndexH3BoundaryIntersectFaceEdge(
            dggs_h3_ijk_to_hex2d(vertexCoords[lastVertex]),
            dggs_h3_ijk_to_hex2d(vertexCoords[vertex]),
            adjustedResolution,
            quadrant
          );
          cellIndexH3BoundaryPush(
            &boundary,
            cellIndexH3BoundaryHex2dToLngLat(point, center.face, adjustedResolution)
          );
        }
      }
    }

    if (vert < vertexTotal) {
      cellIndexH3BoundaryPush(
        &boundary,
        cellIndexH3BoundaryHex2dToLngLat(
          dggs_h3_ijk_to_hex2d(current.coord),
          current.face,
          adjustedResolution
        )
      );
    }
    lastFace = current.face;
    lastOverage = adjusted.overage;
    lastCoord = current.coord;
  }
  return boundary;
}

// All boundary vertices of an H3 cell; \`count\` is 0 for invalid cells.
fn cellIndexH3GetBoundary(cell: vec2u) -> CellIndexH3Boundary {
  var boundary: CellIndexH3Boundary;
  if (!dggs_h3_is_valid_cell_id(cell)) {
    return boundary;
  }
  let center = dggs_h3_get_center_face_ijk(cell);
  if (center.valid == 0u) {
    return boundary;
  }
  let isPentagon = dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) &&
    dggs_h3_get_leading_non_zero_digit(cell) == 0u;
  return cellIndexH3BoundaryFromFaceIJK(center, dggs_h3_get_resolution(cell), isPentagon);
}

fn cellIndexH3GetBoundaryVertexCount(cell: vec2u) -> u32 {
  return cellIndexH3GetBoundary(cell).count;
}

fn cellIndexH3GetBoundaryVertex(cell: vec2u, vertexIndex: u32) -> vec2f {
  let boundary = cellIndexH3GetBoundary(cell);
  if (vertexIndex >= boundary.count) {
    return vec2f(0.0);
  }
  return boundary.points[vertexIndex];
}
`;
