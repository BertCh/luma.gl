var e=`
fn sphericalExcessTriangle(center: vec3f, first: vec3f, second: vec3f) -> f32 {
  let determinant = dot(center, cross(first, second));
  return 2.0 * atan2(determinant, 1.0 + dot(first, second) + dot(second, center) + dot(center, first));
}
`,t=`
${e}
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
    area += abs(sphericalExcessTriangle(center, a, b));
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
`;export{e as n,t};