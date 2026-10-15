import{E as e,w as t,x as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,d as i,h as a,y as o}from"./resources-CwvhEOBe.js";import{l as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as l}from"./geodesic-wgsl-DS0CJZ42.js";import{n as u,t as d}from"./geodesic-kernels-Cg1Javw7.js";var f=`
const RHUMB_MAX_LATITUDE: f32 = 89.9999;

struct RhumbDestination {
  destination: vec2<f32>,
  finalBearing: f32,
}

fn rhumbClampLatitude(latitudeDegrees: f32) -> f32 {
  return clamp(latitudeDegrees, -RHUMB_MAX_LATITUDE, RHUMB_MAX_LATITUDE);
}

// cos of the mean latitude. Within one hemisphere the mean colatitude is formed from the exact
// colatitudes, so near a pole it keeps full relative precision (a + b would round to ~1e-5 degree).
fn rhumbMeanCosLatitude(a: f32, b: f32) -> f32 {
  if (a * b >= 0.0) {
    // colatitude(a) - (|b| - |a|) / 2, with the exact difference so nothing cancels near the pole.
    return sin(((90.0 - abs(a)) - 0.5 * (abs(b) - abs(a))) * GEODESIC_DEGREES_TO_RADIANS);
  }
  return cos(0.5 * (a + b) * GEODESIC_DEGREES_TO_RADIANS);
}

// Short edge: dPsi = dPhi / cos(mean latitude) to relative 1e-8, which keeps meter-scale edges
// exact where atanh of a ~1e-7 argument has no relative precision.
fn rhumbIsShortEdge(a: f32, b: f32) -> bool {
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  return abs(deltaPhi) < 1e-4 * rhumbMeanCosLatitude(a, b);
}

// Stretched (isometric) latitude difference psi(b) - psi(a), with a and b in degrees.
fn rhumbStretchedDelta(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbMeanCosLatitude(a, b);
  }
  let halfDelta = 0.5 * (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfDelta = sin(halfDelta);
  let denominator = 2.0 * sinHalfDelta * sinHalfDelta +
    geodesicCosLatitude(a) * geodesicCosLatitude(b);
  // sin(b) - sin(a) = 2 cos(mean) sin(delta / 2), with the mean cosine from the exact colatitude.
  let sinDelta = 2.0 * rhumbMeanCosLatitude(a, b) * sinHalfDelta;
  let argument = clamp(sinDelta / denominator, -0.999999, 0.999999);
  // Native atanh can lower to log((1+x)/(1-x))/2, losing relative precision near zero.
  // For |x| < 1/8, the odd series through x^7 has relative remainder bounded by
  // x^8 / (9 * (1 - x^2)) < 6.8e-9, below one f32 rounding error.
  if (abs(argument) < 0.125) {
    let squaredArgument = argument * argument;
    return argument * (1.0 + squaredArgument * (1.0 / 3.0 +
      squaredArgument * (1.0 / 5.0 + squaredArgument / 7.0)));
  }
  return atanh(argument);
}

// dPhi / dPsi, with the east-west limit cos(latitude) when the latitudes (nearly) coincide.
fn rhumbStretchRatio(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return rhumbMeanCosLatitude(a, b);
  }
  return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbStretchedDelta(a, b);
}

// Rhumb central angle (distance / radius).
fn rhumbAngle(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x) * GEODESIC_DEGREES_TO_RADIANS;
  let q = rhumbStretchRatio(a, b);
  return sqrt(deltaPhi * deltaPhi + q * q * deltaLambda * deltaLambda);
}

// Constant bearing in degrees clockwise from north, (-180, 180].
fn rhumbBearingDegrees(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x) * GEODESIC_DEGREES_TO_RADIANS;
  var deltaPsi = rhumbStretchedDelta(a, b);
  // Keep atan2's second argument a positive zero (Metal mishandles -0.0).
  deltaPsi = select(deltaPsi, 0.0, deltaPsi == 0.0);
  return atan2(deltaLambda, deltaPsi) * GEODESIC_RADIANS_TO_DEGREES;
}

// Midpoint of the rhumb line: latitude and (unwrapped) longitude are both linear along it.
fn rhumbMidpoint(origin: vec2<f32>, targetPosition: vec2<f32>) -> vec2<f32> {
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x);
  return vec2<f32>(origin.x + 0.5 * deltaLambda, origin.y + 0.5 * (targetPosition.y - origin.y));
}

// Destination after travelling 'angle' radians along a constant bearing.
fn rhumbDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> RhumbDestination {
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  var latitude = origin.y + angle * cos(theta) * GEODESIC_RADIANS_TO_DEGREES;
  var bearing = geodesicWrapLongitudeDelta(bearingDegrees);
  if (latitude > 90.0) {
    latitude = 180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  } else if (latitude < -90.0) {
    latitude = -180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  }
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(latitude);
  let q = rhumbStretchRatio(a, b);
  let deltaLambda = angle * sin(theta) / q;
  return RhumbDestination(
    vec2<f32>(origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES, latitude),
    bearing
  );
}
`,p=`GPUGeodesicPairs`,m=class{id;props;spatialContext;model;radius;ellipsoid;maximumIterations;constructor(e){this.id=e.id??`geodesic-pairs`,this.props=e,this.spatialContext=e.spatialContext;let t=_(this.id,this.spatialContext);this.model=t.model,this.radius=t.radius,this.ellipsoid=t.ellipsoid,this.maximumIterations=e.maximumIterations??16,v(this.id,this.model,this.radius,this.maximumIterations);let{id:n}=this,{output:r}=e;for(let[t,r]of[[`origins`,e.origins],[`targets`,e.targets]])o(r,[`float32x2`],`${n} ${t}`);let i=e.origins.length;if(i<1||e.targets.length!==i)throw Error(`${n} needs at least one pair and targets aligned with origins`);if(b(n,r,{distances:`float32`,initialBearings:`float32`,finalBearings:`float32`,midpoints:`float32x2`,"status.converged":`uint32`},i),r.status?.converged&&this.model!==`wgs84`)throw Error(`${n} output.status.converged requires model 'wgs84'`);a(n,Object.values(y(r)),[e.origins,e.targets])}getCommandNodes(e){let{props:t,id:n}=this,{output:i}=t;r(n,e,[t.origins,t.targets,...Object.values(y(i))]);let a=[{name:`origins`,view:t.origins,type:`f32`,access:`read`},{name:`targets`,view:t.targets,type:`f32`,access:`read`}],o=[],m=this.model===`rhumb`,h=(e,t,n)=>{let r=e===`converged`?i.status?.converged:i[e];r&&(a.push({name:e,view:r,type:t,access:`read_write`}),o.push(n))};h(`distances`,`f32`,`distances[distancesOffset + index] = distance;`),h(`initialBearings`,`f32`,`initialBearings[initialBearingsOffset + index] = initialBearing;`),h(`finalBearings`,`f32`,`finalBearings[finalBearingsOffset + index] = finalBearing;`),h(`midpoints`,`f32`,`let midpoint = ${m?`rhumbMidpoint(origin, targetPosition)`:`geodesicInterpolate(origin, targetPosition, geodesicCentralAngle(origin, targetPosition), 0.5)`};
  midpoints[midpointsOffset + 2u * index] = midpoint.x;
  midpoints[midpointsOffset + 2u * index + 1u] = midpoint.y;`),h(`converged`,`u32`,`converged[convergedOffset + index] = select(0u, 1u, isConverged);`);let g=this.model===`wgs84`;return[c(e,{id:`${n}-pairs`,operation:p,variant:this.model,bindings:a,invocationCount:t.origins.length,declarations:`${l}
${g?u(this.maximumIterations,this.ellipsoid):``}
${d}
${m?f:``}
const RADIUS: f32 = ${s(this.radius)};`,body:`let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let targetPosition = vec2<f32>(targets[targetsOffset + 2u * index], targets[targetsOffset + 2u * index + 1u]);
  var distance = ${m?`rhumbAngle(origin, targetPosition)`:`geodesicCentralAngle(origin, targetPosition)`} * RADIUS;
  var initialBearing = ${m?`rhumbBearingDegrees(origin, targetPosition)`:`geodesicInitialBearingDegrees(origin, targetPosition)`};
  var finalBearing = ${m?`initialBearing`:`sphereFinalBearingDegrees(origin, targetPosition)`};
  var isConverged = true;
  ${g?`let inverse = vincentyInverse(origin, targetPosition);
  isConverged = inverse.converged;
  if (isConverged) {
    distance = inverse.distance;
    initialBearing = inverse.initialBearing;
    finalBearing = inverse.finalBearing;
  }`:``}
  ${o.join(`
  `)}`})]}},h=Object.freeze({measure:`distance`,coordinateSpaces:Object.freeze([`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({"longitude-latitude":Object.freeze([`great-circle`,`rhumb`,`ellipsoidal`])})}),g={"great-circle":`sphere`,rhumb:`rhumb`,ellipsoidal:`wgs84`};function _(r,i){let a=e(r,i,h),o=g[a.metric],s=n(a);return{model:o,radius:t(a),ellipsoid:a.units===`kilometers`?{...s,semiMajorAxis:s.semiMajorAxis/1e3}:s}}function v(e,t,n,r){if(t!==`sphere`&&t!==`wgs84`&&t!==`rhumb`)throw Error(`${e} model must be 'sphere', 'wgs84' or 'rhumb'`);if(!Number.isFinite(n)||n<=0)throw Error(`${e} radius must be a positive finite number`);i(e,`maximumIterations`,r,1,200)}function y(e){let{status:t,...n}=e;return{...n,"status.converged":t?.converged}}function b(e,t,n,r){let i=0;for(let[a,s]of Object.entries(y(t))){if(!s)continue;let t=n[a];if(!t)throw Error(`${e} output.${a} is not a known column`);if(o(s,[t],`${e} output.${a}`),s.length!==r)throw Error(`${e} output.${a} must hold ${r} rows`);i++}if(i===0)throw Error(`${e} requires at least one output column`)}export{v as a,b as i,y as n,f as o,_ as r,m as t};