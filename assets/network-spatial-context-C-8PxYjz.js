import{t as e}from"./geodesic-wgsl-DS0CJZ42.js";function t(t){return t===void 0?`
fn networkDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let delta = a - b;
  return sqrt(dot(delta, delta));
}
fn networkSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  let segment = end - start;
  let denominator = dot(segment, segment);
  if (denominator > 0.0) {
    return clamp(dot(point - start, segment) / denominator, 0.0, 1.0);
  }
  return 0.0;
}
fn networkInterpolate(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  return start + fraction * (end - start);
}
fn networkSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  let offset = point - (start + fraction * (end - start));
  return sqrt(dot(offset, offset));
}`:`${e}
const NETWORK_EARTH_RADIUS: f32 = ${n(t)};
fn networkDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, NETWORK_EARTH_RADIUS);
}
fn networkSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  return geodesicSegmentFraction(point, start, end);
}
fn networkInterpolate(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  let delta = vec2<f32>(geodesicLongitudeDelta(start.x, end.x), end.y - start.y);
  let position = start + fraction * delta;
  return vec2<f32>(geodesicWrapLongitudeDelta(position.x), position.y);
}
fn networkSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  return geodesicSegmentDistance(point, start, end, fraction, NETWORK_EARTH_RADIUS);
}`}function n(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}export{t};