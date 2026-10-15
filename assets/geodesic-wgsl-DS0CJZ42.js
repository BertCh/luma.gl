var e=`
const GEODESIC_PI: f32 = 3.14159265358979;
const GEODESIC_DEGREES_TO_RADIANS: f32 = 0.0174532925199433;
const GEODESIC_RADIANS_TO_DEGREES: f32 = 57.2957795130823;

fn geodesicWrapLongitudeDelta(delta: f32) -> f32 {
  return delta - 360.0 * round(delta / 360.0);
}

// Subtract near the shared antimeridian before forming the small angular separation.
// Wrapping (destination - source) afterward cannot recover bits lost by that large subtraction:
// 179.341 -> -179.877 loses about 1.6 meters on the OpenFlights sphere fixture.
fn geodesicLongitudeDelta(source: f32, destination: f32) -> f32 {
  // WGSL permits floating-point reassociation: the two seam-relative subtractions below
  // can become (destination - source) - 2 * halfTurn, restoring the cancellation.
  // For opposite seam quadrants, every f32 endpoint is exactly a multiple of 2^-17
  // degrees (the smallest ulp on [90, 180]). Scale by that power of two and wrap in
  // i32: magnitudes stay below 2^26, with no quantization or integer overflow. Only
  // the final, small difference is converted back to f32, so reassociation cannot
  // reintroduce a large floating-point subtraction.
  if (abs(source) > 90.0 && abs(source) <= 180.0 &&
      abs(destination) > 90.0 && abs(destination) <= 180.0 &&
      ((source < 0.0) != (destination < 0.0))) {
    let sourceUnits = i32(source * 131072.0);
    let destinationUnits = i32(destination * 131072.0);
    let turnUnits = select(-47185920, 47185920, source > 0.0);
    return f32(destinationUnits - sourceUnits + turnUnits) * 0.00000762939453125;
  }
  let halfTurn = 180.0 * round((destination - source) / 360.0);
  return (destination - halfTurn) - (source + halfTurn);
}

// cos(latitude) as sin(colatitude): 90 - |lat| is exact in f32, so the cosine keeps its relative
// precision near the poles where cos(lat * pi / 180) would not.
fn geodesicCosLatitude(latitudeDegrees: f32) -> f32 {
  return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS);
}

fn geodesicUnitVector(lngLat: vec2<f32>) -> vec3<f32> {
  let lambda = lngLat.x * GEODESIC_DEGREES_TO_RADIANS;
  let phi = lngLat.y * GEODESIC_DEGREES_TO_RADIANS;
  let cosPhi = geodesicCosLatitude(lngLat.y);
  return vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sin(phi));
}

fn geodesicFromUnitVector(vector: vec3<f32>) -> vec2<f32> {
  let horizontal = sqrt(vector.x * vector.x + vector.y * vector.y);
  return vec2<f32>(
    atan2(vector.y, vector.x) * GEODESIC_RADIANS_TO_DEGREES,
    atan2(vector.z, horizontal) * GEODESIC_RADIANS_TO_DEGREES
  );
}

fn geodesicHaversineTerm(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfPhi = sin(0.5 * deltaPhi);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let term = sinHalfPhi * sinHalfPhi +
    geodesicCosLatitude(a.y) * geodesicCosLatitude(b.y) *
    sinHalfLambda * sinHalfLambda;
  return clamp(term, 0.0, 1.0);
}

fn geodesicCentralAngle(a: vec2<f32>, b: vec2<f32>) -> f32 {
  // Vincenty's special case for the sphere: well conditioned from a millimeter to antipodal,
  // unlike the haversine form whose sqrt(1 - h) loses precision near antipodes.
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi1 = sin(phi1);
  let cosPhi1 = geodesicCosLatitude(a.y);
  let cosPhi2 = geodesicCosLatitude(b.y);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let halfVersine = 2.0 * sinHalfLambda * sinHalfLambda;
  let east = cosPhi2 * sin(deltaLambda);
  let north = sin(deltaPhi) + sinPhi1 * cosPhi2 * halfVersine;
  let up = sinPhi1 * sin(phi2) + cosPhi1 * cosPhi2 * (1.0 - halfVersine);
  return atan2(sqrt(east * east + north * north), up);
}

fn geodesicInitialBearingDegrees(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicLongitudeDelta(a.x, b.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let cosPhi2 = geodesicCosLatitude(b.y);
  let y = sin(deltaLambda) * cosPhi2;
  // cos(phi1) sin(phi2) - sin(phi1) cos(phi2) cos(deltaLambda), without cancellation.
  let x = sin(deltaPhi) + 2.0 * sin(phi1) * cosPhi2 * sinHalfLambda * sinHalfLambda;
  // Coincident points have no azimuth. Pin the conventional zero bearing instead of
  // evaluating atan2(0, 0), whose result is unbounded by WGSL (and can be NaN).
  if (x == 0.0 && y == 0.0) {
    return 0.0;
  }
  return atan2(y, x) * GEODESIC_RADIANS_TO_DEGREES;
}

fn geodesicDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> vec2<f32> {
  let phi1 = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi1 = sin(phi1);
  let cosPhi1 = geodesicCosLatitude(origin.y);
  let sinHalfAngle = sin(0.5 * angle);
  // sin(phi2) - sin(phi1), from the half-angle form so short moves keep their precision.
  let sinDelta = -2.0 * sinPhi1 * sinHalfAngle * sinHalfAngle + cosPhi1 * sin(angle) * cos(theta);
  let sinPhi2 = clamp(sinPhi1 + sinDelta, -1.0, 1.0);
  let roughPhi2 = asin(sinPhi2);
  // sin(phi2) - sin(phi1) = 2 cos((phi1 + phi2) / 2) sin((phi2 - phi1) / 2); the midpoint cosine
  // is insensitive to the rough phi2, so the latitude change is recovered to relative precision.
  let midpointCosine = cos(0.5 * (phi1 + roughPhi2));
  var deltaPhi = roughPhi2 - phi1;
  if (abs(midpointCosine) > 1e-3) {
    deltaPhi = 2.0 * asin(clamp(sinDelta / (2.0 * midpointCosine), -1.0, 1.0));
  }
  let deltaLambda = atan2(
    sin(theta) * sin(angle) * cosPhi1,
    cos(angle) - sinPhi1 * sinPhi2
  );
  return vec2<f32>(
    origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES,
    origin.y + deltaPhi * GEODESIC_RADIANS_TO_DEGREES
  );
}

fn geodesicInterpolate(a: vec2<f32>, b: vec2<f32>, angle: f32, fraction: f32) -> vec2<f32> {
  let unitA = geodesicUnitVector(a);
  let unitB = geodesicUnitVector(b);
  var vector: vec3<f32>;
  if (angle < 1e-6) {
    vector = normalize(mix(unitA, unitB, fraction));
  } else {
    let sinAngle = sin(angle);
    vector = unitA * (sin((1.0 - fraction) * angle) / sinAngle) +
      unitB * (sin(fraction * angle) / sinAngle);
  }
  let lngLat = geodesicFromUnitVector(vector);
  // Keep longitudes continuous with the start point (unwrapped across the antimeridian).
  return vec2<f32>(a.x + geodesicWrapLongitudeDelta(lngLat.x - a.x), lngLat.y);
}

fn geodesicSinLatitudeDelta(latitudeDegrees: f32, originLatitudeDegrees: f32) -> f32 {
  let halfSum = 0.5 * (latitudeDegrees + originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  let halfDifference = 0.5 * (latitudeDegrees - originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  return 2.0 * cos(halfSum) * sin(halfDifference);
}

fn geodesicHaversineDistance(a: vec2<f32>, b: vec2<f32>, radius: f32) -> f32 {
  return radius * geodesicCentralAngle(a, b);
}

// Offset of position from origin in the local equirectangular frame at origin, in degrees of
// latitude (east offsets scaled by cos(latitude of origin), longitudes wrapped at +-180).
fn geodesicLocalOffset(origin: vec2<f32>, position: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    geodesicLongitudeDelta(origin.x, position.x) * geodesicCosLatitude(origin.y),
    position.y - origin.y
  );
}

// Fraction in [0, 1] of the point of segment start-end closest to point, in the local
// equirectangular frame at point: exact to first order for segments of road length.
fn geodesicSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  // Pin coincident endpoints before projection: contracting a scaled offset subtraction
  // into an FMA can leave a nonzero residual even when both endpoints are identical.
  if (all(start == end)) {
    return 0.0;
  }
  let startOffset = geodesicLocalOffset(point, start);
  let segment = geodesicLocalOffset(point, end) - startOffset;
  let denominator = dot(segment, segment);
  if (denominator > 0.0) {
    return clamp(-dot(startOffset, segment) / denominator, 0.0, 1.0);
  }
  return 0.0;
}

// Great-circle distance (units of radius) from point to the position at fraction along the
// segment, built from coordinate differences so short distances keep their precision near +-180.
fn geodesicSegmentDistance(
  point: vec2<f32>,
  start: vec2<f32>,
  end: vec2<f32>,
  fraction: f32,
  radius: f32
) -> f32 {
  // Use the same seam-preserving differences as edge lengths and local projection. Mixing
  // rounded seam offsets with precise edge lengths biases map-matching likelihoods.
  let startOffset = vec2<f32>(geodesicLongitudeDelta(point.x, start.x), start.y - point.y);
  let delta = vec2<f32>(geodesicLongitudeDelta(start.x, end.x), end.y - start.y);
  let offset = startOffset + fraction * delta;
  let sinHalfLatitude = sin(0.5 * offset.y * GEODESIC_DEGREES_TO_RADIANS);
  let sinHalfLongitude = sin(0.5 * offset.x * GEODESIC_DEGREES_TO_RADIANS);
  let term = sinHalfLatitude * sinHalfLatitude +
    geodesicCosLatitude(point.y) * geodesicCosLatitude(point.y + offset.y) *
    sinHalfLongitude * sinHalfLongitude;
  return 2.0 * radius * asin(sqrt(clamp(term, 0.0, 1.0)));
}
`;export{e as t};