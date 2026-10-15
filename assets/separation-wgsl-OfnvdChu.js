import{E as e,p as t,w as n}from"./analysis-frame-Blu0tYdw.js";import{l as r}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as i}from"./geodesic-wgsl-DS0CJZ42.js";var a=Object.freeze({...t,measure:`distance`});function o(t,r){let i=e(t,r,a);return{context:i,geographic:i.coordinateSpace===`longitude-latitude`,sphereRadius:n(i)}}function s(e){return e.geographic?`
${i}
const SPHERE_RADIUS: f32 = ${r(e.sphereRadius)};
struct QueryBox { minimum: vec2f, maximum: vec2f, }
// The key is the haversine term h = sin^2(d / 2R): monotonic in distance, so ordering and radius
// tests need no inverse trigonometry.
fn getSeparationKey(position: vec2f, query: vec2f) -> f32 {
  return geodesicHaversineTerm(query, position);
}
fn getRadiusKey(radius: f32) -> f32 {
  let halfAngle = min(0.5 * radius / SPHERE_RADIUS, 1.5707963);
  let sine = sin(halfAngle);
  return sine * sine;
}
fn getSeparationMeters(key: f32) -> f32 {
  return 2.0 * SPHERE_RADIUS * asin(sqrt(clamp(key, 0.0, 1.0)));
}
fn getSquaredMeters(key: f32) -> f32 {
  let meters = getSeparationMeters(key);
  return meters * meters;
}
fn getPairMeters(a: vec2f, b: vec2f) -> f32 { return geodesicHaversineDistance(a, b, SPHERE_RADIUS); }
fn isExactHit(key: f32, position: vec2f, query: vec2f) -> bool { return all(position == query); }
// Conservative lon/lat box around the cap of the given radius: the latitude half-span is r / R,
// the longitude half-span is that divided by the cosine of the highest latitude the box reaches
// (1 percent slack for f32), and a box reaching a pole spans every longitude. No antimeridian
// wrap: a cap crossing +-180 is cut at the index domain edge.
fn getQueryBox(center: vec2f, radius: f32) -> QueryBox {
  let latitudeSpan = min(radius / SPHERE_RADIUS * 57.29577951308232 * 1.01, 180.0);
  let highestLatitude = min(abs(center.y) + latitudeSpan, 90.0);
  var longitudeSpan = 360.0;
  if (highestLatitude < 89.9) {
    longitudeSpan = min(latitudeSpan / geodesicCosLatitude(highestLatitude), 360.0);
  }
  let span = vec2f(longitudeSpan, latitudeSpan);
  return QueryBox(center - span, center + span);
}`:`
struct QueryBox { minimum: vec2f, maximum: vec2f, }
fn getSeparationKey(position: vec2f, query: vec2f) -> f32 {
  let delta = position - query;
  return dot(delta, delta);
}
fn getRadiusKey(radius: f32) -> f32 { return radius * radius; }
fn getSeparationMeters(key: f32) -> f32 { return sqrt(key); }
fn getSquaredMeters(key: f32) -> f32 { return key; }
fn getPairMeters(a: vec2f, b: vec2f) -> f32 {
  let separation = b - a;
  return sqrt(dot(separation, separation));
}
fn isExactHit(key: f32, position: vec2f, query: vec2f) -> bool { return key == 0.0; }
fn getQueryBox(center: vec2f, radius: f32) -> QueryBox {
  return QueryBox(center - vec2f(radius), center + vec2f(radius));
}`}export{o as n,s as t};