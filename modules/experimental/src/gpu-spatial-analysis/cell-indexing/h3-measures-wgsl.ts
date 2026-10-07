// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// H3 cell measures use the face/IJK geometry of Uber's Apache-2.0 H3 (faceijk.c, latLng.c).

/**
 * WGSL for the exact spherical area, perimeter and edge lengths of an H3 cell.
 *
 * Concatenate AFTER `dggs.source` from `@luma.gl/shadertools` and {@link H3_BOUNDARY_WGSL}.
 * Defines, all prefixed `cellMeasures` or `CELL_MEASURES_`:
 *
 * - `struct CellMeasures {count: u32, area: f32, perimeter: f32, edges: array<f32, 10>, isExact: u32}`
 *   with `area` in steradians (square radians on the unit sphere), `perimeter` and `edges[i]` in
 *   radians, where `edges[i]` joins boundary vertex `i` to vertex `i + 1` of H3 `cellToBoundary`.
 * - `fn cellMeasuresGetH3(cell: vec2u) -> CellMeasures` (`count` 0 for invalid cells).
 *
 * Two evaluations, both equal to H3 `cellAreaRads2` (center plus the boundary vertices joined by
 * great-circle arcs) up to f32 rounding:
 *
 * - Exact (`isExact` 1): hexagons whose six vertices stay on the center's icosahedron face. Every
 *   vertex is an integer lattice offset from the center on the gnomonic face plane, so the triangle
 *   determinants and edge differences are exact integers times powers of `1/sqrt(7)` and the only
 *   floating-point work is well conditioned: measured relative error at most 1e-6 at every
 *   resolution 0-15.
 * - Fallback (`isExact` 0): pentagons and cells crossing an icosahedron edge, from the unit vectors
 *   of {@link H3_BOUNDARY_WGSL}. The vertices are f32 unit vectors, so the triangle determinants
 *   cancel: measured worst relative error 6e-7 at resolution 0, 2e-5 at 2, 5e-5 at 3, 2.5e-4 at 4,
 *   1e-2 at 6, growing about 40x per two resolutions (unusable beyond resolution 5). These cells
 *   are under 1% of the cells of resolution 5 and rarer above.
 *
 * @internal
 */
export const H3_MEASURES_WGSL = /* wgsl */ `
const CELL_MEASURES_MAX_EDGES: u32 = 10u;

struct CellMeasures {
  count : u32,
  area : f32,
  perimeter : f32,
  edges : array<f32, 10>,
  isExact : u32,
};

// Hexagon center and per-vertex substrate offsets, or valid = 0 when the hexagon is not on one face.
struct CellMeasuresHexagon {
  valid : u32,
  scale : f32,
  center : vec2f,
  offsets : array<vec3i, 6>,
};

fn cellMeasuresGetHex2d(offset: vec3i) -> vec2f {
  let i = f32(offset.x - offset.z);
  let j = f32(offset.y - offset.z);
  return vec2f(i - 0.5 * j, j * DGGS_H3_SQRT3_2);
}

fn cellMeasuresGetHexagon(cell: vec2u, resolution: u32) -> CellMeasuresHexagon {
  var hexagon: CellMeasuresHexagon;
  hexagon.valid = 0u;
  let center = dggs_h3_get_center_face_ijk(cell);
  if (center.valid == 0u) {
    return hexagon;
  }
  var adjustedResolution = resolution;
  var centerCoord = dggs_h3_down_ap3r(dggs_h3_down_ap3(center.coord));
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    centerCoord = dggs_h3_down_ap7r(centerCoord);
    adjustedResolution += 1u;
  }
  let maximumDimension = 3 * dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution);
  for (var vertex = 0u; vertex < 6u; vertex++) {
    let offset = dggs_h3_get_vertex_offset(resolution, vertex);
    let coord = dggs_h3_ijk_normalize(centerCoord + offset);
    if (coord.x + coord.y + coord.z > maximumDimension) {
      return hexagon;
    }
    hexagon.offsets[vertex] = offset;
  }
  var scale = DGGS_H3_RES0_U_GNOMONIC * DGGS_H3_ONETHIRD;
  for (var level = 0u; level < adjustedResolution; level++) {
    scale *= DGGS_H3_RSQRT7;
  }
  hexagon.scale = scale;
  hexagon.center = scale * cellMeasuresGetHex2d(centerCoord);
  hexagon.valid = 1u;
  return hexagon;
}

// Length of the vector from the sphere center to a point of the gnomonic plane at distance 1.
fn cellMeasuresGetPlaneNorm(point: vec2f) -> f32 {
  return sqrt(1.0 + dot(point, point));
}

fn cellMeasuresGetExactHexagon(hexagon: CellMeasuresHexagon) -> CellMeasures {
  var measures: CellMeasures;
  measures.count = 6u;
  measures.isExact = 1u;
  let scale = hexagon.scale;
  let centerNorm = cellMeasuresGetPlaneNorm(hexagon.center);
  var points: array<vec2f, 6>;
  for (var vertex = 0u; vertex < 6u; vertex++) {
    points[vertex] = hexagon.center + scale * cellMeasuresGetHex2d(hexagon.offsets[vertex]);
  }
  var area = 0.0;
  var perimeter = 0.0;
  for (var vertex = 0u; vertex < 6u; vertex++) {
    let next = (vertex + 1u) % 6u;
    let offsetA = hexagon.offsets[vertex];
    let offsetB = hexagon.offsets[next];
    let pointA = points[vertex];
    let pointB = points[next];
    let latticeA = vec2f(f32(offsetA.x - offsetA.z), f32(offsetA.y - offsetA.z));
    let latticeB = vec2f(f32(offsetB.x - offsetB.z), f32(offsetB.y - offsetB.z));
    // Twice the planar area of the triangle (center, A, B): an exact lattice cross product.
    let determinant = abs(scale * scale * DGGS_H3_SQRT3_2 * (latticeA.x * latticeB.y - latticeA.y * latticeB.x));
    let normA = cellMeasuresGetPlaneNorm(pointA);
    let normB = cellMeasuresGetPlaneNorm(pointB);
    let dotAB = 1.0 + dot(pointA, pointB);
    let dotBC = 1.0 + dot(pointB, hexagon.center);
    let dotCA = 1.0 + dot(hexagon.center, pointA);
    let denominator = normA * normB * centerNorm + dotAB * centerNorm + dotBC * normA + dotCA * normB;
    area += 2.0 * atan2(determinant, denominator);

    // Great-circle arc A-B: |wA x wB| = sqrt(|d|^2 + (qA x d)^2) with d = qB - qA exact.
    let latticeDelta = latticeB - latticeA;
    let delta = scale * vec2f(latticeDelta.x - 0.5 * latticeDelta.y, latticeDelta.y * DGGS_H3_SQRT3_2);
    let crossProduct = pointA.x * delta.y - pointA.y * delta.x;
    let edge = atan2(sqrt(dot(delta, delta) + crossProduct * crossProduct), dotAB);
    measures.edges[vertex] = edge;
    perimeter += edge;
  }
  measures.area = area;
  measures.perimeter = perimeter;
  return measures;
}

fn cellMeasuresGetBoundaryMeasures(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  measures.isExact = 0u;
  let boundary = cellIndexH3GetBoundary(cell);
  measures.count = boundary.count;
  if (boundary.count < 3u) {
    return measures;
  }
  let center = dggs_h3_get_center_unit_vector(cell);
  var area = 0.0;
  var perimeter = 0.0;
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    let a = boundary.units[vertex];
    let b = boundary.units[(vertex + 1u) % boundary.count];
    let determinant = abs(dot(center, cross(a, b)));
    area += 2.0 * atan2(determinant, 1.0 + dot(a, b) + dot(b, center) + dot(center, a));
    let edge = atan2(length(cross(a, b)), dot(a, b));
    measures.edges[vertex] = edge;
    perimeter += edge;
  }
  measures.area = area;
  measures.perimeter = perimeter;
  return measures;
}

fn cellMeasuresIsPentagon(cell: vec2u) -> bool {
  return dggs_h3_is_valid_cell_id(cell) &&
    dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) &&
    dggs_h3_get_leading_non_zero_digit(cell) == 0u;
}

fn cellMeasuresGetH3(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  if (!dggs_h3_is_valid_cell_id(cell)) {
    return measures;
  }
  if (!cellMeasuresIsPentagon(cell)) {
    let hexagon = cellMeasuresGetHexagon(cell, dggs_h3_get_resolution(cell));
    if (hexagon.valid != 0u) {
      return cellMeasuresGetExactHexagon(hexagon);
    }
  }
  return cellMeasuresGetBoundaryMeasures(cell);
}
`;
