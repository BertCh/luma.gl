import{E as e,d as t,f as n}from"./analysis-frame-Blu0tYdw.js";import{a as r,c as i,i as a,o}from"./cell-topology-wgsl-BT8_s49b.js";import{_ as s,h as c,y as l}from"./resources-CwvhEOBe.js";import{s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as d}from"./wgsl-common-BNRVzgd8.js";var f=[16,18,24,33,30,800,305,816,-1,8,1285,1290,22,-1,-1,297,-1,-1,4,1280,-1,271,-1,-1,-1,-1,-1,2,6,14,10,11,785,280,791,-1,0,1281,1289,5,-1,-1,274,-1,-1,260,1283,-1,264,-1,-1,-1,-1,-1,7,21,38,9,19,802,270,788,-1,3,1293,1309,1,-1,-1,262,-1,-1,516,1292,-1,256,-1,-1,-1,-1,-1,26,42,58,29,43,830,294,815,-1,12,1308,1324,13,-1,-1,277,-1,-1,772,1295,-1,259,-1,-1,-1,-1,-1,31,41,49,44,53,829,314,833,-1,15,1302,1313,28,-1,-1,298,-1,-1,1028,1288,-1,268,-1,-1,-1,-1,-1,50,48,817,32,798,801,792,786,-1,70,67,834,820,-1,-1,805,-1,-1,83,855,-1,842,-1,-1,-1,-1,-1,25,23,792,17,779,778,782,774,-1,45,39,805,803,-1,-1,795,-1,-1,63,827,-1,824,-1,-1,-1,-1,-1,36,20,782,34,787,777,806,789,-1,55,40,795,822,-1,-1,819,-1,-1,72,828,-1,841,-1,-1,-1,-1,-1,64,47,806,62,811,797,826,810,-1,84,69,819,850,-1,-1,844,-1,-1,97,857,-1,866,-1,-1,-1,-1,-1,75,65,826,61,821,812,817,809,-1,94,86,844,849,-1,-1,834,-1,-1,107,872,-1,869,-1,-1,-1,-1,-1,57,59,831,74,846,847,851,860,-1,37,807,813,52,-1,-1,838,-1,-1,24,791,-1,800,-1,-1,-1,-1,-1,46,60,840,56,836,848,831,845,-1,27,808,823,35,-1,-1,813,-1,-1,14,788,-1,785,-1,-1,-1,-1,-1,71,89,865,73,859,871,840,856,-1,51,837,852,54,-1,-1,823,-1,-1,38,815,-1,802,-1,-1,-1,-1,-1,96,104,875,98,878,883,865,879,-1,76,854,862,82,-1,-1,852,-1,-1,58,833,-1,830,-1,-1,-1,-1,-1,85,87,851,101,870,868,875,880,-1,66,835,838,81,-1,-1,862,-1,-1,49,816,-1,829,-1,-1,-1,-1,-1,95,92,83,79,78,842,319,827,-1,109,108,1380,349,-1,-1,333,-1,-1,1141,1398,-1,362,-1,-1,-1,-1,-1,90,77,63,80,68,824,328,828,-1,106,93,1359,355,-1,-1,344,-1,-1,885,1389,-1,369,-1,-1,-1,-1,-1,105,88,72,103,91,841,353,857,-1,113,99,1360,372,-1,-1,367,-1,-1,629,1386,-1,377,-1,-1,-1,-1,-1,119,111,97,115,110,866,363,872,-1,121,116,1383,376,-1,-1,368,-1,-1,373,1393,-1,374,-1,-1,-1,-1,-1,114,112,107,100,102,869,339,855,-1,118,120,1395,364,-1,-1,348,-1,-1,117,1401,-1,365,-1,-1,-1,-1,-1],p=[[4,-1,-1],[14,2,6],[24,1,5],[38,3,7],[49,0,9],[58,4,8],[63,11,15],[72,12,16],[83,10,19],[97,13,17],[107,14,18],[117,-1,-1]],m=f.map(e=>e<0?`65535u`:`${e}u`).join(`, `),h=p.filter(([,e])=>e>=0).map(([e,t,n])=>`(baseCell == ${e}u && (face == ${t}u || face == ${n}u))`).join(` ||
    `),g=`\
const CELL_INDEX_H3_BASE_CELL_TABLE = array<u32, 540>(${m});
const CELL_INDEX_H3_SQRT7_POWERS = array<f32, 16>(${Array.from({length:16},(e,t)=>Math.fround(Math.sqrt(7)**t).toPrecision(9)).join(`, `)});
const CELL_INDEX_H3_ROTATE_CCW = array<u32, 8>(0u, 5u, 3u, 1u, 6u, 4u, 2u, 7u);
const CELL_INDEX_H3_NO_CELL: u32 = 65535u;
// cos and sin of the aperture-7 rotation: 5 / (2 sqrt 7) and sqrt 3 / (2 sqrt 7).
const CELL_INDEX_H3_AP7_COS: f32 = 0.944911182523068;
const CELL_INDEX_H3_AP7_SIN: f32 = 0.3273268353539886;

fn cellIndexH3IsCwOffset(baseCell: u32, face: u32) -> bool {
  return ${h};
}

// (sin, cos) of an angle in degrees. The angle is reduced to [-45, 45] degrees exactly in degrees,
// then a Taylor polynomial gives full f32 accuracy independent of the builtin sin/cos precision.
fn cellIndexH3SinCosDegrees(degrees: f32) -> vec2f {
  let quadrant = round(degrees * (1.0 / 90.0));
  let reduced = (degrees - quadrant * 90.0) * 0.017453292519943295;
  let x2 = reduced * reduced;
  let sine = reduced * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 +
    x2 * (1.0 / 362880.0 + x2 * (-1.0 / 39916800.0))))));
  let cosine = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 +
    x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0))))));
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 1u) {
    return vec2f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec2f(-sine, -cosine);
  }
  if (turn == 3u) {
    return vec2f(-cosine, sine);
  }
  return vec2f(sine, cosine);
}

fn cellIndexH3GetNearestFace(point: vec3f) -> u32 {
  var face = 0u;
  var bestAlignment = -2.0;
  for (var candidate = 0u; candidate < 20u; candidate++) {
    let alignment = dot(point, dggs_h3_get_face_unit_vector_basis(candidate, 0u));
    if (alignment > bestAlignment) {
      bestAlignment = alignment;
      face = candidate;
    }
  }
  return face;
}

// H3 _hex2dToCoordIJK: quantizes a hex2d point to the containing cell's normalized IJK.
fn cellIndexH3Hex2dToIjk(point: vec2f) -> vec3i {
  let a1 = abs(point.x);
  let a2 = abs(point.y);
  let x2 = a2 / DGGS_H3_SQRT3_2;
  let x1 = a1 + x2 * 0.5;
  let m1 = i32(x1);
  let m2 = i32(x2);
  let r1 = x1 - f32(m1);
  let r2 = x2 - f32(m2);
  var i = 0;
  var j = 0;
  if (r1 < 0.5) {
    if (r1 < 1.0 / 3.0) {
      i = m1;
      j = select(m2 + 1, m2, r2 < (1.0 + r1) * 0.5);
    } else {
      j = select(m2 + 1, m2, r2 < 1.0 - r1);
      i = select(m1, m1 + 1, (1.0 - r1) <= r2 && r2 < 2.0 * r1);
    }
  } else if (r1 < 2.0 / 3.0) {
    j = select(m2 + 1, m2, r2 < 1.0 - r1);
    i = select(m1 + 1, m1, (2.0 * r1 - 1.0) < r2 && r2 < 1.0 - r1);
  } else {
    i = m1 + 1;
    j = select(m2 + 1, m2, r2 < r1 * 0.5);
  }
  if (point.x < 0.0) {
    if ((j & 1) == 0) {
      i -= 2 * (i - j / 2);
    } else {
      i -= 2 * (i - (j + 1) / 2) + 1;
    }
  }
  if (point.y < 0.0) {
    i -= (2 * j + 1) / 2;
    j = -j;
  }
  return dggs_h3_ijk_normalize(vec3i(i, j, 0));
}

// H3 _upAp7: Class III child to Class II parent (counter-clockwise aperture 7). Upstream dggs
// only has dggs_h3_up_ap7r (clockwise, (2i+j)/7); this one is (3i-j)/7, so it is not a duplicate.
fn cellIndexH3UpAp7(coord: vec3i) -> vec3i {
  let i = coord.x - coord.z;
  let j = coord.y - coord.z;
  return dggs_h3_ijk_normalize(vec3i(
    dggs_h3_divide_round_nearest_seven(3 * i - j),
    dggs_h3_divide_round_nearest_seven(i + 2 * j),
    0
  ));
}

fn cellIndexH3RotateDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      CELL_INDEX_H3_ROTATE_CCW[digit]
    );
  }
  return result;
}

fn cellIndexH3RotateDigitsCw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      dggs_h3_rotate_digit_60_cw(digit)
    );
  }
  return result;
}

// H3 _h3RotatePent60ccw: rotates counter-clockwise, skipping the deleted K subsequence.
fn cellIndexH3RotatePentagonDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  var foundFirstNonZeroDigit = false;
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = CELL_INDEX_H3_ROTATE_CCW[dggs_h3_get_digit(result, digitResolution)];
    result = dggs_u64_set_bits(result, dggs_h3_digit_bit_offset(digitResolution), 3u, digit);
    if (!foundFirstNonZeroDigit && digit != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellIndexH3RotateDigitsCcw(result);
      }
    }
  }
  return result;
}

// H3 _faceIjkToH3 for an IJK at 'resolution' on 'face'. Returns vec2u(0u) when no base cell fits.
fn cellIndexH3FaceIjkToIndex(face: u32, coord: vec3i, resolution: u32) -> vec2u {
  // Header with base cell 0 and every digit unused (7); digits and base cell are filled in below.
  var index = vec2u((1u << 27u) | (resolution << 20u) | 0x1FFFu, 0xFFFFFFFFu);
  var current = coord;
  for (var parentResolution = resolution; parentResolution > 0u; parentResolution--) {
    let last = current;
    var lastCenter = vec3i(0);
    if (dggs_h3_is_resolution_class_iii(parentResolution)) {
      current = cellIndexH3UpAp7(current);
      lastCenter = dggs_h3_down_ap7(current);
    } else {
      current = dggs_h3_up_ap7r(current);
      lastCenter = dggs_h3_down_ap7r(current);
    }
    let difference = dggs_h3_ijk_normalize(last - lastCenter);
    index = dggs_u64_set_bits(
      index,
      dggs_h3_digit_bit_offset(parentResolution),
      3u,
      u32(4 * difference.x + 2 * difference.y + difference.z)
    );
  }
  if (any(current < vec3i(0)) || any(current > vec3i(2))) {
    return vec2u(0u);
  }
  let packed = CELL_INDEX_H3_BASE_CELL_TABLE[
    face * 27u + u32(current.x) * 9u + u32(current.y) * 3u + u32(current.z)
  ];
  if (packed == CELL_INDEX_H3_NO_CELL) {
    return vec2u(0u);
  }
  let baseCell = packed & 0xFFu;
  let rotations = packed >> 8u;
  index = dggs_u64_set_bits(index, 45u, 7u, baseCell);
  if (dggs_h3_is_base_cell_pentagon(baseCell)) {
    if (dggs_h3_get_leading_non_zero_digit(index) == 1u) {
      if (cellIndexH3IsCwOffset(baseCell, face)) {
        index = cellIndexH3RotateDigitsCw(index);
      } else {
        index = cellIndexH3RotateDigitsCcw(index);
      }
    }
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotatePentagonDigitsCcw(index);
    }
  } else {
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotateDigitsCcw(index);
    }
  }
  return index;
}

fn cellIndexH3FromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u {
  // Exponent-bit tests catch NaN and infinity without a NaN compare, which Metal may fold.
  let exponentBits = vec2u(bitcast<u32>(lngLatDegrees.x), bitcast<u32>(lngLatDegrees.y)) & vec2u(0x7f800000u);
  if (
    any(exponentBits == vec2u(0x7f800000u)) ||
    abs(lngLatDegrees.x) >= 3.0e38 ||
    abs(lngLatDegrees.y) >= 3.0e38 ||
    resolution > DGGS_H3_MAX_RESOLUTION
  ) {
    return vec2u(0u);
  }
  let longitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.x);
  let latitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.y);
  let point = vec3f(
    latitudeSinCos.y * longitudeSinCos.y,
    latitudeSinCos.y * longitudeSinCos.x,
    latitudeSinCos.x
  );
  let face = cellIndexH3GetNearestFace(point);
  let faceCenter = dggs_h3_get_face_unit_vector_basis(face, 0u);
  let inverseScale = 1.0 / (dot(point, faceCenter) * DGGS_H3_RES0_U_GNOMONIC);
  var hexPoint = vec2f(
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 1u)),
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 2u))
  ) * inverseScale;
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    hexPoint = vec2f(
      CELL_INDEX_H3_AP7_COS * hexPoint.x + CELL_INDEX_H3_AP7_SIN * hexPoint.y,
      -CELL_INDEX_H3_AP7_SIN * hexPoint.x + CELL_INDEX_H3_AP7_COS * hexPoint.y
    );
  }
  hexPoint *= CELL_INDEX_H3_SQRT7_POWERS[resolution];
  return cellIndexH3FaceIjkToIndex(face, cellIndexH3Hex2dToIjk(hexPoint), resolution);
}
`,_=`
const CELL_INDEX_H3_BOUNDARY_MAX_VERTICES: u32 = 10u;
const CELL_INDEX_H3_BOUNDARY_NO_OVERAGE: u32 = 0u;
const CELL_INDEX_H3_BOUNDARY_FACE_EDGE: u32 = 1u;
const CELL_INDEX_H3_BOUNDARY_NEW_FACE: u32 = 2u;

struct CellIndexH3Boundary {
  count : u32,
  points : array<vec2f, 10>,
  units : array<vec3f, 10>,
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

// Substrate hex2d point on \`face\` to a unit vector (H3 _hex2dToGeo with substrate = 1). Projects
// through the face's unit-vector basis instead of dggs_h3_hex2d_to_lnglat, whose sin/cos/atan2
// azimuth-distance chain loses about 1e-4 degrees in f32.
fn cellIndexH3BoundaryHex2dToUnitVector(point: vec2f, face: u32, adjustedResolution: u32) -> vec3f {
  var scale = DGGS_H3_RES0_U_GNOMONIC * DGGS_H3_ONETHIRD;
  for (var level = 0u; level < adjustedResolution; level++) {
    scale *= DGGS_H3_RSQRT7;
  }
  // \`adjustedResolution\` is always Class II here, so no extra aperture-7 factor applies.
  return normalize(
    dggs_h3_get_face_unit_vector_basis(face, 0u) +
    scale * (
      point.x * dggs_h3_get_face_unit_vector_basis(face, 1u) +
      point.y * dggs_h3_get_face_unit_vector_basis(face, 2u)
    )
  );
}

fn cellIndexH3BoundaryUnitVectorToLngLat(unitVector: vec3f) -> vec2f {
  return vec2f(
    atan2(unitVector.y, unitVector.x) * DGGS_RADIANS_TO_DEGREES,
    atan2(unitVector.z, length(unitVector.xy)) * DGGS_RADIANS_TO_DEGREES
  );
}

fn cellIndexH3BoundaryPush(boundary: ptr<function, CellIndexH3Boundary>, unitVector: vec3f) {
  if ((*boundary).count < CELL_INDEX_H3_BOUNDARY_MAX_VERTICES) {
    (*boundary).points[(*boundary).count] = cellIndexH3BoundaryUnitVectorToLngLat(unitVector);
    (*boundary).units[(*boundary).count] = unitVector;
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
            cellIndexH3BoundaryHex2dToUnitVector(point, lastFace, adjustedResolution)
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
            cellIndexH3BoundaryHex2dToUnitVector(point, center.face, adjustedResolution)
          );
        }
      }
    }

    if (vert < vertexTotal) {
      cellIndexH3BoundaryPush(
        &boundary,
        cellIndexH3BoundaryHex2dToUnitVector(
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
`,v={quadbin:{minimum:0,maximum:26},h3:{minimum:0,maximum:15},quadkey:{minimum:1,maximum:29},geohash:{minimum:1,maximum:12},s2:{minimum:0,maximum:30}},y=[`quadbin`,`h3`,`quadkey`,`geohash`,`s2`];function b(e){return typeof e==`string`&&e in v}function x(e,t,n){if(!b(t))throw Error(`${e} family must be one of ${y.join(`, `)}`);let{minimum:r,maximum:i}=v[t];if(!Number.isInteger(n)||n<r||n>i)throw Error(`${e} ${t} resolution must be an integer in [${r}, ${i}]`)}var S=[[0,1,3,2],[0,2,3,1],[3,2,0,1],[3,1,0,2]],C=[1,0,0,3];function w(){let e=[];for(let t of S){let n=[0,0,0,0];t.forEach((e,t)=>{n[e]=t}),e.push(...n)}return e}var T=`
${a}
${r}
fn cellIndexQuadkeyFromLngLat(longitude: f32, latitude: f32, zoom: u32) -> vec2u {
  let x = quadbinGetTileX(longitude, zoom);
  let y = quadbinGetTileY(latitude, zoom);
  let morton = quadbinGetCompactKey(x, y);
  return vec2u(morton.x | (zoom << 26u), morton.y);
}
`,E=`
${a}
${r}
fn cellIndexGeohashFromLngLat(longitude: f32, latitude: f32, length: u32) -> vec2u {
  let totalBits = 5u * length;
  let longitudeBits = (totalBits + 1u) / 2u;
  let latitudeBits = totalBits / 2u;
  var longitudeIndex = cellMaskLow32(longitudeBits);
  if (longitude < 180.0) {
    longitudeIndex = quadbinGetTileX(longitude, longitudeBits);
  }
  var latitudeIndex = cellMaskLow32(latitudeBits);
  if (latitude < 90.0) {
    latitudeIndex = quadbinGetTileX(2.0 * max(latitude, -90.0), latitudeBits);
  }
  var interleaved = quadbinGetCompactKey(longitudeIndex, latitudeIndex);
  if ((totalBits & 1u) == 0u) {
    interleaved = quadbinGetCompactKey(latitudeIndex, longitudeIndex);
  }
  return vec2u(interleaved.x | (length << 28u), interleaved.y);
}
`,D=`
${a}
const CELL_INDEX_S2_IJ_TO_POSITION = array<u32, 16>(${w().map(e=>`${e}u`).join(`, `)});
const CELL_INDEX_S2_POSITION_TO_ORIENTATION = array<u32, 4>(${C.map(e=>`${e}u`).join(`, `)});
const CELL_INDEX_S2_RADIANS_PER_DEGREE: f32 = 0.017453292519943295;

/** (sin, cos) of an angle in degrees, exact quadrant reduction plus polynomials on [-45, 45]. */
fn cellIndexS2SinCosDegrees(degrees: f32) -> vec2f {
  let quadrant = round(degrees / 90.0);
  let x = (degrees - quadrant * 90.0) * CELL_INDEX_S2_RADIANS_PER_DEGREE;
  let x2 = x * x;
  let sine = x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))));
  let cosine = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0)))));
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 0u) {
    return vec2f(sine, cosine);
  }
  if (turn == 1u) {
    return vec2f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec2f(-sine, -cosine);
  }
  return vec2f(-cosine, sine);
}

fn cellIndexS2UvToSt(uv: f32) -> f32 {
  if (uv >= 0.0) {
    return 0.5 * sqrt(1.0 + 3.0 * uv);
  }
  return 1.0 - 0.5 * sqrt(1.0 - 3.0 * uv);
}

fn cellIndexS2StToIj(st: f32) -> u32 {
  let scaled = floor(st * 1073741824.0);
  if (scaled >= 1073741824.0) {
    return 1073741823u;
  }
  return u32(max(scaled, 0.0));
}

fn cellIndexS2FromLngLat(longitude: f32, latitude: f32, level: u32) -> vec2u {
  let lngTrig = cellIndexS2SinCosDegrees(longitude);
  let latTrig = cellIndexS2SinCosDegrees(clamp(latitude, -90.0, 90.0));
  let cosLatitude = max(latTrig.y, 0.0);
  let p = vec3f(cosLatitude * lngTrig.y, cosLatitude * lngTrig.x, latTrig.x);
  let magnitude = abs(p);
  var axis = 2u;
  if (magnitude.x > magnitude.y) {
    if (magnitude.x > magnitude.z) {
      axis = 0u;
    }
  } else if (magnitude.y > magnitude.z) {
    axis = 1u;
  }
  let component = select(p.z, select(p.y, p.x, axis == 0u), axis < 2u);
  let face = axis + select(0u, 3u, component < 0.0);
  var uv: vec2f;
  if (face == 0u) {
    uv = vec2f(p.y / p.x, p.z / p.x);
  } else if (face == 1u) {
    uv = vec2f(-p.x / p.y, p.z / p.y);
  } else if (face == 2u) {
    uv = vec2f(-p.x / p.z, -p.y / p.z);
  } else if (face == 3u) {
    uv = vec2f(p.z / p.x, p.y / p.x);
  } else if (face == 4u) {
    uv = vec2f(p.z / p.y, -p.x / p.y);
  } else {
    uv = vec2f(-p.y / p.z, -p.x / p.z);
  }
  let i = cellIndexS2StToIj(cellIndexS2UvToSt(uv.x));
  let j = cellIndexS2StToIj(cellIndexS2UvToSt(uv.y));
  var orientation = face & 1u;
  // Face in bits 61..63 and the level marker at bit 60 - 2 level.
  var key = cellShiftLeft(vec2u(0u, face), 61u) | cellShiftLeft(vec2u(0u, 1u), 60u - 2u * level);
  for (var step = 0u; step < level; step++) {
    let bit = 29u - step;
    let ij = (((i >> bit) & 1u) << 1u) | ((j >> bit) & 1u);
    let position = CELL_INDEX_S2_IJ_TO_POSITION[orientation * 4u + ij];
    orientation = orientation ^ CELL_INDEX_S2_POSITION_TO_ORIENTATION[position];
    key = key | cellShiftLeft(vec2u(0u, position), 59u - 2u * step);
  }
  return key;
}
`,O=`GPUPointToCell`,k=class{id;props;spatialContext;constructor(r){this.id=r.id??`point-to-cell`,this.props=r,this.spatialContext=e(this.id,r.spatialContext??t,n);let i=this.id;x(i,r.family,r.resolution),l(r.positions,[`float32x2`],`${i} positions`);let a=r.positions.length;if(a<1)throw Error(`${i} needs at least one row`);if(r.mask&&(l(r.mask,[`uint32`],`${i} mask`),r.mask.length!==a))throw Error(`${i} mask length must equal the row count`);if(l(r.output.cells,[`uint32x2`],`${i} output.cells`),r.output.cells.length!==a)throw Error(`${i} output.cells length must equal the row count`);if(r.output.validity&&(l(r.output.validity,[`uint32`],`${i} output.validity`),r.output.validity.length!==a))throw Error(`${i} output.validity length must equal the row count`);c(i,[r.output.cells,r.output.validity],[r.positions,r.mask])}getCommandNodes(e){let{id:t,props:n}=this,{family:r,resolution:i,output:a}=n;s(t,e,[n.positions,n.mask,a.cells,a.validity]);let o=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},...n.mask?[{name:`rowMask`,view:n.mask,type:`u32`,access:`read`}]:[],{name:`cellsOut`,view:a.cells,type:`u32`,access:`read_write`},...a.validity?[{name:`validityOut`,view:a.validity,type:`u32`,access:`read_write`}]:[]],c=j(r,i);return[u(e,{id:`${t}-keys`,operation:O,variant:r,bindings:o,invocationCount:n.positions.length,declarations:`const RESOLUTION: u32 = ${i}u;
${A(r)}
${d}`,body:`let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${n.mask?`isValid = isValid && rowMask[rowMaskOffset + index] != 0u;`:``}
  var key = vec2u(0u);
  if (isValid) {
    key = ${c};
  }
  isValid = isValid && any(key != vec2u(0u));
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${a.validity?`validityOut[validityOutOffset + index] = select(0u, 1u, isValid);`:``}`})]}};function A(e){switch(e){case`quadbin`:return`${a}\n${r}`;case`quadkey`:return T;case`geohash`:return E;case`s2`:return D;case`h3`:return`${i.source}\n${g}`}}function j(e,t){switch(e){case`quadbin`:{let e=o(`quadbin`,t);return`cellGetKey(
      quadbinGetCompactKey(quadbinGetTileX(longitude, RESOLUTION), quadbinGetTileY(latitude, RESOLUTION)),
      ${e.headerHigh}u, RESOLUTION, ${e.lowBit}u)`}case`quadkey`:return`cellIndexQuadkeyFromLngLat(longitude, latitude, RESOLUTION)`;case`geohash`:return`cellIndexGeohashFromLngLat(longitude, latitude, RESOLUTION)`;case`s2`:return`cellIndexS2FromLngLat(longitude, latitude, RESOLUTION)`;case`h3`:return`cellIndexH3FromLngLat(vec2f(longitude, latitude), RESOLUTION)`}}export{_ as n,g as r,k as t};