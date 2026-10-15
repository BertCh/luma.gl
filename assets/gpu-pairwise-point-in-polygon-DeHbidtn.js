import{O as e}from"./layer-DFQ5HqjV.js";import{E as t,T as n,u as r}from"./analysis-frame-Blu0tYdw.js";import{a as i}from"./arithmetic-expression-Kknn3j_i.js";import{A as a,D as o,M as s,O as c,T as l,i as u,k as d}from"./resources-CwvhEOBe.js";import{u as f}from"./interleave-brZZCO-f.js";import{l as p}from"./wgsl-common-BNRVzgd8.js";var m=[`float32x2`,`uint32x4`],h=4294967295,g=Math.floor(h/256)+1,_={name:e.name,source:e.source};function v(e){return Array.isArray(e.data)}function y(e){return v(e)?e.data:[e]}function b(e,t,n){for(let r of y(e)){if(s(r,t,n),r.byteOffset%r.rowByteLength!==0)throw Error(`${n} must be naturally aligned to its row format`);if(r.format!==e.format)throw Error(`${n} chunks must use the declared vector format`)}}function x(e,t,n){if(v(e)!==v(t))throw Error(`${n} must use the same view kind`);if(v(e)&&v(t))a(e,t,n);else if(e.length!==t.length)throw Error(`${n} must contain the same number of rows`)}function S(e,t,n){let r=t.flatMap(([e,t])=>y(t).map(t=>[e,t])),i=[];for(let[t,a]of n){let n=y(a);for(let a of n){for(let[n,i]of r)if(C(a,i))throw Error(`${e} output ${t} and ${n} must not overlap`);for(let[n,r]of i)if(C(a,r))throw Error(`${e} output ${t} and output ${n} must not overlap`)}i.push(...n.map(e=>[t,e]))}}function C(e,t){if(l(e,t))return!0;let n=w(e),r=w(t);if(e.buffer!==t.buffer&&n!==void 0&&n===r)return!0;if(e.buffer!==t.buffer)return!1;let i=c(e),a=c(t),o=i.offset+i.size,s=a.offset+a.size;return i.offset<s&&a.offset<o}function w(e){let t=e.buffer.defaultBuffer;return t instanceof i?t.buffer:t}function T(e,t,n){if(t.some(t=>y(t).some(t=>t.buffer.graph!==e)))throw Error(`${n} views must belong to the target graph`)}function E(e,t){let n=d(t);return t.format===`float32x2`?{declaration:`const ${e.toUpperCase()}_OFFSET: u32 = ${n}u;\n@group(0) @binding(auto) var<storage, read> ${e}: array<vec2f>;`,read:t=>`${e}[${e.toUpperCase()}_OFFSET / 2u + (${t})]`,precise:!1}:{declaration:`const ${e.toUpperCase()}_OFFSET: u32 = ${n}u;\n@group(0) @binding(auto) var<storage, read> ${e}: array<u32>;`,read:t=>{let n=`${e.toUpperCase()}_OFFSET + (${t}) * 4u`;return`makeRawPoint(${e}[${n}], ${e}[${n} + 1u], ${e}[${n} + 2u], ${e}[${n} + 3u])`},precise:!0}}var D=`
struct RawPoint { x: vec2u, y: vec2u }

fn makeRawPoint(xLow: u32, xHigh: u32, yLow: u32, yHigh: u32) -> RawPoint {
  // Browser Float64Array words are low/high; fp64 helpers consume high/low.
  return RawPoint(
    vec2u(xHigh, xLow),
    vec2u(yHigh, yLow)
  );
}

fn rawScalarIsFinite(value: vec2u) -> bool {
  return ((value.x >> 20u) & 0x7ffu) != 0x7ffu;
}

fn rawPointIsFinite(point: RawPoint) -> bool {
  return rawScalarIsFinite(point.x) && rawScalarIsFinite(point.y);
}

fn rawPointToF32(point: RawPoint) -> vec2f {
  let zero = vec2u(0u, 0u);
  return vec2f(
    sub_fp64u32_to_f32(point.x, zero),
    sub_fp64u32_to_f32(point.y, zero)
  );
}
`;function O(e,t){if(!Number.isSafeInteger(e)||e<0||e>h)throw Error(`geospatial element count must be a non-negative uint32`);let n=Math.floor(t);if(!Number.isSafeInteger(n)||n<1)throw Error(`maxComputeWorkgroupsPerDimension must be a positive integer`);let r=Math.max(1,Math.ceil(e/256)),i=Math.min(r,n),a=Math.min(Math.ceil(r/i),n),o=Math.ceil(r/i/a);if(o>n)throw Error(`geospatial operation requires ${r} workgroups, exceeding the 3D dispatch limit of ${n} per dimension`);return{x:i,y:a,z:o}}function k(e){return`let workgroupIndex = (workgroupId.z * ${e.y}u + workgroupId.y) * ${e.x}u + workgroupId.x;
  if (workgroupIndex >= ${g}u) { return; }
  let index = workgroupIndex * 256u + localId.x;`}function A(e,t){if(!t.precise&&t.fp64Profile!==void 0)throw Error(`geospatial fp64 profiles require precise arithmetic`);e.addComputePass({id:t.id,resources:t.resources,compile:({device:e})=>{let n=t.precise?[_]:[],r=t.fp64Profile??`full`,i=t.precise?{LUMA_FP64_INTEGER_ARITHMETIC:!0,...r===`full`?{}:{LUMA_FP64_PREDICATE_ONLY:!0},...r===`predicate-f32`?{LUMA_FP64_F32_INPUT_ONLY:!0}:{}}:{},a=new f(e,{id:t.id,source:t.source,modules:n,defines:i,shaderLayout:{bindings:Object.keys(t.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:n})=>{let r={};for(let[e,i]of Object.entries(t.bindings))r[e]=o(i,n);a.setBindings(r),a.dispatch(e,t.dispatchLayout.x,t.dispatchLayout.y,t.dispatchLayout.z)},destroy:()=>a.destroy()}}})}var j=`
// Predicate arithmetic produces normalized high/low pairs. Keep these helpers
// on that narrower contract so software backends do not repeatedly compile the
// generic arbitrary-limb normalizer into every sign and finite check.
fn geospatial_max_abs_fp64(first: vec2f, second: vec2f) -> f32 {
  return max(abs(first.x), abs(second.x));
}

fn geospatial_div_fp64_f32(value: vec2f, divisor: f32) -> vec2f {
  return twoSum(value.x / divisor, value.y / divisor);
}

fn geospatial_is_finite_normalized_fp64(value: vec2f) -> bool {
  let highMagnitude = bitcast<u32>(value.x) & 0x7fffffffu;
  let lowMagnitude = bitcast<u32>(value.y) & 0x7fffffffu;
  return highMagnitude < 0x7f800000u && lowMagnitude < 0x7f800000u;
}

fn geospatial_sign_normalized_fp64(value: vec2f) -> i32 {
  let highBits = bitcast<u32>(value.x);
  let lowBits = bitcast<u32>(value.y);
  let highMagnitude = highBits & 0x7fffffffu;
  let lowMagnitude = lowBits & 0x7fffffffu;
  if (highMagnitude > 0x7f800000u || lowMagnitude > 0x7f800000u) { return 0; }
  if (highMagnitude != 0u) { return select(1, -1, (highBits >> 31u) == 1u); }
  if (lowMagnitude != 0u) { return select(1, -1, (lowBits >> 31u) == 1u); }
  return 0;
}
`,M={outside:0,inside:1,boundary:2,uncertain:3},N=class{id;spatialContext;queryPositions;positions;featureOffsets;polygonOffsets;ringOffsets;output;constructor(e){if(this.id=e.id??`gpu-pairwise-point-in-polygon`,this.spatialContext=t(this.id,e.spatialContext,{coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`],edgeModels:r}),n(this.id,`polygons`,e.polygonsSpatialContext,this.spatialContext,r),this.queryPositions=e.queryPositions,this.positions=e.positions,this.featureOffsets=e.featureOffsets,this.polygonOffsets=e.polygonOffsets,this.ringOffsets=e.ringOffsets,this.output=e.output,b(this.queryPositions,[`float32x2`,`uint32x4`],`${this.id} queryPositions`),b(this.positions,[`float32x2`,`uint32x4`],`${this.id} positions`),this.queryPositions.format!==this.positions.format)throw Error(`${this.id} point and polygon position formats must match`);for(let[e,t]of[[`featureOffsets`,this.featureOffsets],[`polygonOffsets`,this.polygonOffsets],[`ringOffsets`,this.ringOffsets]])b(t,[`uint32`],`${this.id} ${e}`);if(this.featureOffsets.length!==this.queryPositions.length+1)throw Error(`${this.id} featureOffsets.length must equal queryPositions.length + 1`);if(this.polygonOffsets.length<1||this.ringOffsets.length<1)throw Error(`${this.id} polygonOffsets and ringOffsets require a terminal entry`);if(b(this.output,[`uint32`],`${this.id} output`),this.output.length!==this.queryPositions.length)throw Error(`${this.id} output.length must equal queryPositions.length`);S(this.id,[[`queryPositions`,this.queryPositions],[`positions`,this.positions],[`featureOffsets`,this.featureOffsets],[`polygonOffsets`,this.polygonOffsets],[`ringOffsets`,this.ringOffsets]],[[`output`,this.output]])}getCommandNodes(e){return u(e,()=>this.addNodes(e))}addNodes(e){T(e,[this.queryPositions,this.positions,this.featureOffsets,this.polygonOffsets,this.ringOffsets,this.output],this.id),this.queryPositions.length!==0&&this.addPass(e)}addPass(e){let t=E(`points`,this.queryPositions),n=E(`polygonPositions`,this.positions),r=O(this.queryPositions.length,e.device.limits.maxComputeWorkgroupsPerDimension),i=this.queryPositions.format===`uint32x4`,a=`
${i?D:``}
${j}
const OUTSIDE: u32 = ${M.outside}u;
const INSIDE: u32 = ${M.inside}u;
const BOUNDARY: u32 = ${M.boundary}u;
const UNCERTAIN: u32 = ${M.uncertain}u;
const ELEMENT_COUNT: u32 = ${this.queryPositions.length}u;
const POLYGON_COUNT: u32 = ${this.polygonOffsets.length-1}u;
const RING_COUNT: u32 = ${this.ringOffsets.length-1}u;
const VERTEX_COUNT: u32 = ${this.positions.length}u;
const GEOMETRY_OFFSETS_OFFSET: u32 = ${d(this.featureOffsets)}u;
const POLYGON_OFFSETS_OFFSET: u32 = ${d(this.polygonOffsets)}u;
const RING_OFFSETS_OFFSET: u32 = ${d(this.ringOffsets)}u;
const OUTPUT_OFFSET: u32 = ${d(this.output)}u;
${t.declaration}
${n.declaration}
@group(0) @binding(auto) var<storage, read> geometryOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> polygonOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> ringOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read_write> classifications: array<u32>;

${P(n,i)}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  ${k(r)}
  if (index >= ELEMENT_COUNT) { return; }
  let point = ${t.read(`index`)};
  classifications[OUTPUT_OFFSET + index] = classifyGeometry(point, index);
}`;A(e,{id:this.id,source:a,resources:[{buffer:this.queryPositions,usage:`storage-read`},{buffer:this.positions,usage:`storage-read`},{buffer:this.featureOffsets,usage:`storage-read`},{buffer:this.polygonOffsets,usage:`storage-read`},{buffer:this.ringOffsets,usage:`storage-read`},{buffer:this.output,usage:`storage-write`}],bindings:{points:this.queryPositions,polygonPositions:this.positions,geometryOffsets:this.featureOffsets,polygonOffsets:this.polygonOffsets,ringOffsets:this.ringOffsets,classifications:this.output},dispatchLayout:r,precise:!0,fp64Profile:i?`predicate-raw`:`predicate-f32`})}};function P(e,t){let n=t?`RawPoint`:`vec2f`;return`
struct RelativePoint {
  x: vec2f,
  y: vec2f,
  finite: bool,
  exactZeroX: bool,
  exactZeroY: bool,
  underresolved: bool,
}

struct Orientation {
  sign: i32,
  uncertain: bool,
}

struct EdgeClassification {
  crossing: bool,
  boundary: bool,
  uncertain: bool,
}

${p}

${t?`fn rawScalarEqual(first: vec2u, second: vec2u) -> bool {
  let firstDecoded = fp64_decode_bits(first);
  let secondDecoded = fp64_decode_bits(second);
  if (firstDecoded.isNan || secondDecoded.isNan) { return false; }
  if (firstDecoded.isZero && secondDecoded.isZero) { return true; }
  return all(first == second);
}

fn sourcePointsEqual(first: RawPoint, second: RawPoint) -> bool {
  return rawScalarEqual(first.x, second.x) && rawScalarEqual(first.y, second.y);
}`:`fn sourcePointsEqual(first: vec2f, second: vec2f) -> bool {
  return all(first == second);
}`}

fn pointIsFinite(point: ${n}) -> bool {
  ${t?`return rawPointIsFinite(point);`:`return isFiniteValue(point.x) && isFiniteValue(point.y);`}
}

fn makeRelativePoint(vertex: ${n}, point: ${n}) -> RelativePoint {
  ${t?`let exactZeroX = rawScalarEqual(vertex.x, point.x);
  let exactZeroY = rawScalarEqual(vertex.y, point.y);
  let deltaX = sub_fp64u32_to_fp64(vertex.x, point.x);
  let deltaY = sub_fp64u32_to_fp64(vertex.y, point.y);`:`let exactZeroX = vertex.x == point.x;
  let exactZeroY = vertex.y == point.y;
  let deltaX = twoSub(vertex.x, point.x);
  let deltaY = twoSub(vertex.y, point.y);`}
  let finite =
    geospatial_is_finite_normalized_fp64(deltaX) &&
    geospatial_is_finite_normalized_fp64(deltaY);
  let underresolved =
    (!exactZeroX && geospatial_sign_normalized_fp64(deltaX) == 0) ||
    (!exactZeroY && geospatial_sign_normalized_fp64(deltaY) == 0);
  return RelativePoint(deltaX, deltaY, finite, exactZeroX, exactZeroY, underresolved);
}

fn readPolygonPoint(vertexIndex: u32) -> ${n} {
  return ${e.read(`vertexIndex`)};
}

fn readRelativePoint(vertexIndex: u32, point: ${n}) -> RelativePoint {
  return makeRelativePoint(readPolygonPoint(vertexIndex), point);
}

fn bracketsZero(first: vec2f, second: vec2f) -> bool {
  let firstSign = geospatial_sign_normalized_fp64(first);
  let secondSign = geospatial_sign_normalized_fp64(second);
  return firstSign == 0 || secondSign == 0 || firstSign != secondSign;
}

fn getOrientation(start: RelativePoint, end: RelativePoint) -> Orientation {
  if (!start.finite || !end.finite || start.underresolved || end.underresolved) {
    return Orientation(0, true);
  }
  let scale = max(
    geospatial_max_abs_fp64(start.x, start.y),
    geospatial_max_abs_fp64(end.x, end.y)
  );
  if (scale == 0.0) { return Orientation(0, false); }
  let startX = geospatial_div_fp64_f32(start.x, scale);
  let startY = geospatial_div_fp64_f32(start.y, scale);
  let endX = geospatial_div_fp64_f32(end.x, scale);
  let endY = geospatial_div_fp64_f32(end.y, scale);
  let firstProduct = mul_fp64(startX, endY);
  let secondProduct = mul_fp64(startY, endX);
  let crossProduct = sub_fp64(firstProduct, secondProduct);
  let approximateFirst = (startX.x + startX.y) * (endY.x + endY.y);
  let approximateSecond = (startY.x + startY.y) * (endX.x + endX.y);
  let approximateCross = approximateFirst - approximateSecond;
  let errorBound =
    (abs(approximateFirst) + abs(approximateSecond)) * 9.5367431640625e-7;
  let orientationSign = geospatial_sign_normalized_fp64(crossProduct);
  let nearErrorBound = orientationSign != 0 && abs(approximateCross) <= errorBound;
  let underresolved =
    (geospatial_sign_normalized_fp64(start.x) != 0 &&
      geospatial_sign_normalized_fp64(startX) == 0) ||
    (geospatial_sign_normalized_fp64(start.y) != 0 &&
      geospatial_sign_normalized_fp64(startY) == 0) ||
    (geospatial_sign_normalized_fp64(end.x) != 0 &&
      geospatial_sign_normalized_fp64(endX) == 0) ||
    (geospatial_sign_normalized_fp64(end.y) != 0 &&
      geospatial_sign_normalized_fp64(endY) == 0) ||
    (geospatial_sign_normalized_fp64(startX) != 0 &&
      geospatial_sign_normalized_fp64(endY) != 0 &&
      geospatial_sign_normalized_fp64(firstProduct) == 0) ||
    (geospatial_sign_normalized_fp64(startY) != 0 &&
      geospatial_sign_normalized_fp64(endX) != 0 &&
      geospatial_sign_normalized_fp64(secondProduct) == 0);
  return Orientation(
    orientationSign,
    underresolved ||
      nearErrorBound ||
      !geospatial_is_finite_normalized_fp64(firstProduct) ||
      !geospatial_is_finite_normalized_fp64(secondProduct) ||
      !geospatial_is_finite_normalized_fp64(crossProduct)
  );
}

fn ringHasProvableArea(ringStart: u32, ringEnd: u32) -> bool {
  if (ringEnd - ringStart < 3u) { return false; }
  let first = readPolygonPoint(ringStart);
  var second = first;
  var secondFound = false;
  for (var vertexIndex = ringStart + 1u; vertexIndex < ringEnd; vertexIndex++) {
    let candidate = readPolygonPoint(vertexIndex);
    if (!sourcePointsEqual(candidate, first)) {
      second = candidate;
      secondFound = true;
      break;
    }
  }
  if (!secondFound) { return false; }
  let relativeSecond = makeRelativePoint(second, first);
  for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
    let candidate = readPolygonPoint(vertexIndex);
    if (sourcePointsEqual(candidate, first) || sourcePointsEqual(candidate, second)) {
      continue;
    }
    let relativeCandidate = makeRelativePoint(candidate, first);
    let orientation = getOrientation(relativeSecond, relativeCandidate);
    if (!orientation.uncertain && orientation.sign != 0) { return true; }
  }
  return false;
}

fn classifyEdge(start: RelativePoint, end: RelativePoint) -> EdgeClassification {
  if (!start.finite || !end.finite || start.underresolved || end.underresolved) {
    return EdgeClassification(false, false, true);
  }
  let startIsPoint = start.exactZeroX && start.exactZeroY;
  let endIsPoint = end.exactZeroX && end.exactZeroY;
  if (startIsPoint || endIsPoint) {
    return EdgeClassification(false, true, false);
  }
  let startYSign = geospatial_sign_normalized_fp64(start.y);
  let endYSign = geospatial_sign_normalized_fp64(end.y);
  let upward = startYSign <= 0 && endYSign > 0;
  let downward = endYSign <= 0 && startYSign > 0;
  let straddlesY = upward || downward;
  // Signs of the relative coordinates are exact, so edges decided by signs alone never consult the
  // orientation envelope. An edge strictly on one side of the point's y can neither be crossed by
  // the +x ray nor contain the point, however close its supporting line passes to the point.
  if ((startYSign > 0 && endYSign > 0) || (startYSign < 0 && endYSign < 0)) {
    return EdgeClassification(false, false, false);
  }
  if (straddlesY) {
    let startXSign = geospatial_sign_normalized_fp64(start.x);
    let endXSign = geospatial_sign_normalized_fp64(end.x);
    if (startXSign > 0 && endXSign > 0) {
      return EdgeClassification(true, false, false);
    }
    if (startXSign < 0 && endXSign < 0) {
      return EdgeClassification(false, false, false);
    }
  }
  let orientation = getOrientation(start, end);
  if (orientation.uncertain) {
    return EdgeClassification(false, false, true);
  }
  if (orientation.sign == 0) {
    let withinSegment = bracketsZero(start.x, end.x) && bracketsZero(start.y, end.y);
    if (withinSegment) {
      let exactAxisBoundary =
        (start.exactZeroX && end.exactZeroX) ||
        (start.exactZeroY && end.exactZeroY);
      return EdgeClassification(false, exactAxisBoundary, !exactAxisBoundary);
    }
    return EdgeClassification(false, false, straddlesY);
  }
  let crossing = (upward && orientation.sign > 0) || (downward && orientation.sign < 0);
  return EdgeClassification(crossing, false, false);
}

fn offsetsAreGloballyValid() -> bool {
  return
    geometryOffsets[GEOMETRY_OFFSETS_OFFSET] == 0u &&
    geometryOffsets[GEOMETRY_OFFSETS_OFFSET + ELEMENT_COUNT] == POLYGON_COUNT &&
    polygonOffsets[POLYGON_OFFSETS_OFFSET] == 0u &&
    polygonOffsets[POLYGON_OFFSETS_OFFSET + POLYGON_COUNT] == RING_COUNT &&
    ringOffsets[RING_OFFSETS_OFFSET] == 0u &&
    ringOffsets[RING_OFFSETS_OFFSET + RING_COUNT] == VERTEX_COUNT;
}

fn classifyGeometry(point: ${n}, geometryIndex: u32) -> u32 {
  if (!pointIsFinite(point) || !offsetsAreGloballyValid()) { return UNCERTAIN; }
  let geometryStart = geometryOffsets[GEOMETRY_OFFSETS_OFFSET + geometryIndex];
  let geometryEnd = geometryOffsets[GEOMETRY_OFFSETS_OFFSET + geometryIndex + 1u];
  if (geometryStart > geometryEnd || geometryEnd > POLYGON_COUNT) { return UNCERTAIN; }

  var geometryInside = false;
  var boundary = false;
  var uncertain = false;
  for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
    let polygonStart = polygonOffsets[POLYGON_OFFSETS_OFFSET + polygonIndex];
    let polygonEnd = polygonOffsets[POLYGON_OFFSETS_OFFSET + polygonIndex + 1u];
    if (polygonStart >= polygonEnd || polygonEnd > RING_COUNT) {
      uncertain = true;
      continue;
    }
    var polygonInside = false;
    for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
      let ringStart = ringOffsets[RING_OFFSETS_OFFSET + ringIndex];
      let ringEnd = ringOffsets[RING_OFFSETS_OFFSET + ringIndex + 1u];
      if (ringStart > ringEnd || ringEnd > VERTEX_COUNT) {
        uncertain = true;
        continue;
      }
      if (!ringHasProvableArea(ringStart, ringEnd)) {
        uncertain = true;
        continue;
      }
      var previous = readRelativePoint(ringEnd - 1u, point);
      for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
        let current = readRelativePoint(vertexIndex, point);
        let edge = classifyEdge(previous, current);
        boundary = boundary || edge.boundary;
        uncertain = uncertain || edge.uncertain;
        if (edge.crossing) { polygonInside = !polygonInside; }
        previous = current;
      }
    }
    geometryInside = geometryInside || polygonInside;
  }
  if (uncertain) { return UNCERTAIN; }
  if (boundary) { return BOUNDARY; }
  return select(OUTSIDE, INSIDE, geometryInside);
}`}export{A as a,k as c,v as d,S as f,D as i,E as l,b as m,M as n,T as o,x as p,m as r,O as s,N as t,y as u};