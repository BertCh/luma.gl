import{t as e}from"./status-BwNARr9H.js";import{s as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var n=`
const LON_LAT_POLE_LATITUDE_TOLERANCE: f32 = 1e-6;
const LON_LAT_AMBIGUOUS_STEP_TOLERANCE: f32 = 1e-4;
const LON_LAT_POLE_WINDING_LIMIT: f32 = 180.0;
const LON_LAT_RADIANS_PER_DEGREE: f32 = 0.0174532925199433;

fn lonLatWrapLongitudeStep(rawStep: f32) -> f32 {
  return rawStep - 360.0 * round(rawStep / 360.0);
}

fn lonLatIsPoleEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return abs(a.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE &&
    abs(b.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE && a.y * b.y > 0.0;
}

fn lonLatRingStep(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let rawStep = b.x - a.x;
  if (lonLatIsPoleEdge(a, b)) { return rawStep; }
  return lonLatWrapLongitudeStep(rawStep);
}

fn lonLatIsCrossingEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return !lonLatIsPoleEdge(a, b) && abs(b.x - a.x) > 180.0;
}

fn lonLatIsAmbiguousLinearEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  return !lonLatIsPoleEdge(a, b) &&
    abs(lonLatWrapLongitudeStep(b.x - a.x)) >= 180.0 - LON_LAT_AMBIGUOUS_STEP_TOLERANCE;
}

fn lonLatIsAmbiguousGreatCircleEdge(a: vec2<f32>, b: vec2<f32>) -> bool {
  if (lonLatIsPoleEdge(a, b)) { return false; }
  // Rotate the first longitude to zero. Colatitudes and the supplementary longitude keep
  // exact pole/180-degree inputs exact, avoiding sin(pi)'s f32 residual at the v1 threshold.
  let longitude = lonLatWrapLongitudeStep(b.x - a.x);
  let longitudeRadians = longitude * LON_LAT_RADIANS_PER_DEGREE;
  let longitudeSine = sign(longitude) * sin(min(abs(longitude), 180.0 - abs(longitude)) * LON_LAT_RADIANS_PER_DEGREE);
  let firstCosine = sin((90.0 - abs(a.y)) * LON_LAT_RADIANS_PER_DEGREE);
  let secondCosine = sin((90.0 - abs(b.y)) * LON_LAT_RADIANS_PER_DEGREE);
  let first = vec3<f32>(firstCosine, 0.0, sin(a.y * LON_LAT_RADIANS_PER_DEGREE));
  let second = vec3<f32>(secondCosine * cos(longitudeRadians), secondCosine * longitudeSine, sin(b.y * LON_LAT_RADIANS_PER_DEGREE));
  if (length(first + second) <= 1e-6) { return true; }
  if (abs(a.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE ||
      abs(b.y) >= 90.0 - LON_LAT_POLE_LATITUDE_TOLERANCE) { return true; }
  let normal = cross(first, second);
  let normalLength = length(normal);
  if (normalLength == 0.0) { return false; }
  let unitNormal = normal / normalLength;
  if (abs(unitNormal.z) > sin(LON_LAT_POLE_LATITUDE_TOLERANCE * LON_LAT_RADIANS_PER_DEGREE)) { return false; }
  let north = normalize(vec3<f32>(0.0, 0.0, 1.0) - unitNormal.z * unitNormal);
  for (var hemisphere = 0u; hemisphere < 2u; hemisphere++) {
    let pole = select(north, -north, hemisphere == 1u);
    if (dot(cross(first, pole), normal) >= 0.0 && dot(cross(pole, second), normal) >= 0.0) { return true; }
  }
  return false;
}

fn lonLatIsPoleWinding(winding: f32) -> bool {
  return abs(winding) > LON_LAT_POLE_WINDING_LIMIT;
}
`;function r(e){return!!(e?.antimeridianUncertainCount||e?.poleUncertainCount)}function i(r,i){e(i.operation,i.status);let a=[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:i.ringOffsets,type:`u32`,access:`read`}];for(let e of[`antimeridianUncertainCount`,`poleUncertainCount`]){let t=i.status[e];t&&a.push({name:e,view:t,type:`atomic<u32>`,access:`read_write`})}return t(r,{id:i.id,operation:i.operation,variant:`lon-lat-polygon-check`,bindings:a,invocationCount:Math.max(0,i.ringOffsets.length-1),declarations:`${n}
fn lonLatCheckPosition(vertex: u32) -> vec2<f32> {
  let positionOffset = positionsOffset + 2u * vertex;
  return vec2<f32>(positions[positionOffset], positions[positionOffset + 1u]);
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  var end = ringOffsets[ringOffsetsOffset + index + 1u];
  if (end <= begin + 2u) { return; }
  let first = lonLatCheckPosition(begin);
  if (all(first == lonLatCheckPosition(end - 1u))) { end--; }
  if (end <= begin + 2u) { return; }
  // Find three distinct positions without quadratic work or per-ring scratch storage.
  var second = first;
  var hasSecond = false;
  var hasThird = false;
  var winding = 0.0;
  var crossing = false;
  for (var vertex = begin; vertex < end; vertex++) {
    let position = lonLatCheckPosition(vertex);
    if (any(position != first)) {
      if (!hasSecond) { second = position; hasSecond = true; }
      else if (any(position != second)) { hasThird = true; }
    }
    var next = first;
    if (vertex + 1u < end) { next = lonLatCheckPosition(vertex + 1u); }
    winding += lonLatRingStep(position, next);
    crossing = crossing || lonLatIsCrossingEdge(position, next);
  }
  if (!hasThird) { return; }
  ${i.status.antimeridianUncertainCount?`if (crossing) { atomicAdd(&antimeridianUncertainCount[antimeridianUncertainCountOffset], 1u); }`:``}
  ${i.status.poleUncertainCount?`if (lonLatIsPoleWinding(winding)) { atomicAdd(&poleUncertainCount[poleUncertainCountOffset], 1u); }`:``}`})}export{i as n,r,n as t};