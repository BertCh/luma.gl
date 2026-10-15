import{mt as e}from"./src-CNCKFW2b.js";import{E as t}from"./analysis-frame-Blu0tYdw.js";import{_ as n,d as r,m as i,w as a,x as o,y as s}from"./resources-CwvhEOBe.js";import{l as c,s as l}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as u}from"./wgsl-common-BNRVzgd8.js";var d=`GPUEdgeBundling`,f=-1e8,p=-1e7,m=1e-9,h=1e-9;function g(e){return`
const EDGE_COUNT: u32 = ${e.edgeCount}u;
const VERTEX_COUNT: u32 = ${e.vertexCount}u;
const POINTS: u32 = ${e.pointsPerEdge}u;
const RESOLUTION: u32 = ${e.densityResolution}u;
const RESOLUTION_F: f32 = ${c(e.densityResolution)};
const FIXED_POINT_SCALE: f32 = ${c(2**e.fixedPointExponent)};
const FIXED_POINT_INVERSE: f32 = ${c(2**-e.fixedPointExponent)};
const MAXIMUM_ITERATIONS: u32 = ${e.iterationCount}u;
const BOX_SCALE: f32 = ${c(1+2*e.boxPadding)};
const DEAD_COORDINATE: f32 = ${c(f)};
const DEAD_LIMIT: f32 = ${c(p)};
const LENGTH_EPSILON: f32 = ${c(m)};
const GRADIENT_MINIMUM: f32 = ${c(4)} * FIXED_POINT_INVERSE;
const RADIUS_EPSILON: f32 = ${c(h)};`}function _(e){let t;if(!e)t=`
fn readActiveIterations() -> u32 { return MAXIMUM_ITERATIONS; }
fn readInitialRadius() -> f32 { return 0.03; }
fn readLambda() -> f32 { return 0.85; }
fn readSmoothing() -> f32 { return 0.5; }
fn readStepScale() -> f32 { return 1.0; }`;else{let n=e.format===`float32`,r=e=>n?`parameters[parametersOffset + ${e}u]`:`bitcast<f32>(parameters[parametersOffset + ${e}u])`;t=`
fn readActiveIterations() -> u32 {
  ${n?`return u32(clamp(parameters[parametersOffset], 0.0, f32(MAXIMUM_ITERATIONS)));`:`return min(parameters[parametersOffset], MAXIMUM_ITERATIONS);`}
}
fn readInitialRadius() -> f32 { return max(${r(1)}, 0.0); }
fn readLambda() -> f32 { return clamp(${r(2)}, 0.5, 0.9); }
fn readSmoothing() -> f32 { return clamp(${r(3)}, 0.0, 1.0); }
fn readStepScale() -> f32 { return ${e.length>4?`max(${r(4)}, 0.0)`:`1.0`}; }`}return t}function v(){return`
fn getIterationRadius() -> f32 {
  let iteration = iterationInfo[iterationInfoOffset];
  var radius = readInitialRadius();
  let lambda = readLambda();
  for (var round = 0u; round < iteration; round++) {
    radius = radius * lambda;
  }
  return radius;
}`}function y(e,t){return l(e,{id:t.id,operation:d,variant:`iteration-info`,bindings:[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read_write`}],invocationCount:t.iterationCount,body:`iterationInfo[iterationInfoOffset + index * 64u] = index;`})}function b(e){return`
// Strict 3.0e38 bound (finite and not near overflow), unlike the shared isFiniteValue.
${u}
fn isFiniteValue(value: f32) -> bool { return isFiniteBits(value) && abs(value) < 3.0e38; }
fn getVertexPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(
    vertexPositions[vertexPositionsOffset + vertex * 2u],
    vertexPositions[vertexPositionsOffset + vertex * 2u + 1u]
  );
}
fn isEdgeLive(edge: u32) -> bool {
  let source = edgeSources[edgeSourcesOffset + edge];
  let targetVertex = edgeTargets[edgeTargetsOffset + edge];
  if (source >= VERTEX_COUNT || targetVertex >= VERTEX_COUNT) {
    return false;
  }
  ${e?`if (edgeMask[edgeMaskOffset + edge] == 0u) {
    return false;
  }`:``}
  let a = getVertexPosition(source);
  let b = getVertexPosition(targetVertex);
  return isFiniteValue(a.x) && isFiniteValue(a.y) && isFiniteValue(b.x) && isFiniteValue(b.y);
}`}function x(e){return`
fn decodeKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
fn getBox() -> vec4<f32> {
  if (boxKeys[boxKeysOffset] == 0xffffffffu) {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
  }
  let minX = decodeKey(boxKeys[boxKeysOffset]);
  let minY = decodeKey(boxKeys[boxKeysOffset + 1u]);
  let maxX = decodeKey(boxKeys[boxKeysOffset + 2u]);
  let maxY = decodeKey(boxKeys[boxKeysOffset + 3u]);
  ${e?`let xScale = max(cos(0.5 * (minY + maxY) * 0.017453292519943295), 0.01);`:`let xScale = 1.0;`}
  let extent = max((maxX - minX) * xScale, maxY - minY);
  let side = select(extent * BOX_SCALE, 1.0, !(extent > 0.0));
  return vec4<f32>(
    0.5 * (minX + maxX) * xScale - 0.5 * side,
    0.5 * (minY + maxY) - 0.5 * side,
    side,
    xScale
  );
}
fn toWork(workBox: vec4<f32>, position: vec2<f32>) -> vec2<f32> {
  return (vec2<f32>(position.x * workBox.w, position.y) - workBox.xy) / workBox.z;
}
fn fromWork(workBox: vec4<f32>, normalized: vec2<f32>) -> vec2<f32> {
  let scaled = workBox.xy + normalized * workBox.z;
  return vec2<f32>(scaled.x / workBox.w, scaled.y);
}`}function S(e){let t=[{name:`vertexPositions`,view:e.positions,type:`f32`,access:`read`},{name:`edgeSources`,view:e.sources,type:`u32`,access:`read`},{name:`edgeTargets`,view:e.targets,type:`u32`,access:`read`}];return e.mask&&t.push({name:`edgeMask`,view:e.mask,type:`u32`,access:`read`}),t}var C=256,w=64,T={clear:0,splat:1,update:2};function E(e,t,n){if(!e)return{};let r=e.view.byteOffset+4*(e.iteration*9+3*T[n]);return{condition:{id:`${t}-gate`,source:`gpu`,mode:`indirect`,buffer:e.view.buffer,byteOffset:r},extraResources:[{buffer:e.view,usage:`indirect`}]}}function D(e,t){let{constants:n}=t,r=e.device.limits.maxComputeWorkgroupsPerDimension,i={clear:o(d,n.densityResolution*n.densityResolution,256,r),splat:o(d,n.edgeCount*n.pointsPerEdge,256,r),update:o(d,n.edgeCount,w,r)},a=[`clear`,`splat`,`update`].map((e,t)=>`    gate[gateOffset + base + ${3*t}u] = select(0u, ${i[e].x}u, isActive);
    gate[gateOffset + base + ${3*t+1}u] = ${i[e].y}u;
    gate[gateOffset + base + ${3*t+2}u] = ${i[e].z}u;`).join(`
`);return l(e,{id:t.id,operation:d,variant:`gate`,bindings:[{name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`},{name:`gate`,view:t.gate,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${g(n)}
${_(t.parameters)}`,body:`let activeIterations = readActiveIterations();
  var radius = readInitialRadius();
  let lambda = readLambda();
  for (var iteration = 0u; iteration < MAXIMUM_ITERATIONS; iteration++) {
    let isActive = iteration < activeIterations && radius > RADIUS_EPSILON;
    let base = iteration * 9u;
${a}
    radius = radius * lambda;
  }`})}function O(e,t){return l(e,{id:t.id,operation:d,variant:`box-reset`,bindings:[{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read_write`}],invocationCount:4,body:`boxKeys[boxKeysOffset + index] = select(0u, 0xffffffffu, index < 2u);`})}function k(e,t){return l(e,{id:t.id,operation:d,variant:`box`,workgroupSize:C,bindings:[...S(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.constants.edgeCount,guardIndex:!1,declarations:`${g(t.constants)}
${b(!!t.inputs.mask)}
fn encodeKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
var<workgroup> sharedBox: array<u32, ${4*C}>;`,body:`var lowX = 0xffffffffu;
  var lowY = 0xffffffffu;
  var highX = 0u;
  var highY = 0u;
  if (index < EDGE_COUNT && isEdgeLive(index)) {
    let a = getVertexPosition(edgeSources[edgeSourcesOffset + index]);
    let b = getVertexPosition(edgeTargets[edgeTargetsOffset + index]);
    let low = min(a, b);
    let high = max(a, b);
    lowX = encodeKey(low.x);
    lowY = encodeKey(low.y);
    highX = encodeKey(high.x);
    highY = encodeKey(high.y);
  }
  sharedBox[localInvocationIndex] = lowX;
  sharedBox[${C}u + localInvocationIndex] = lowY;
  sharedBox[${2*C}u + localInvocationIndex] = highX;
  sharedBox[${3*C}u + localInvocationIndex] = highY;
  workgroupBarrier();
  for (var stride = ${C/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      let other = localInvocationIndex + stride;
      sharedBox[localInvocationIndex] = min(sharedBox[localInvocationIndex], sharedBox[other]);
      sharedBox[${C}u + localInvocationIndex] =
        min(sharedBox[${C}u + localInvocationIndex], sharedBox[${C}u + other]);
      sharedBox[${2*C}u + localInvocationIndex] =
        max(sharedBox[${2*C}u + localInvocationIndex], sharedBox[${2*C}u + other]);
      sharedBox[${3*C}u + localInvocationIndex] =
        max(sharedBox[${3*C}u + localInvocationIndex], sharedBox[${3*C}u + other]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    atomicMin(&boxKeys[boxKeysOffset], sharedBox[0]);
    atomicMin(&boxKeys[boxKeysOffset + 1u], sharedBox[${C}u]);
    atomicMax(&boxKeys[boxKeysOffset + 2u], sharedBox[${2*C}u]);
    atomicMax(&boxKeys[boxKeysOffset + 3u], sharedBox[${3*C}u]);
  }`})}function A(e,t){return l(e,{id:t.id,operation:d,variant:`initialize`,bindings:[...S(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read_write`}],invocationCount:t.constants.edgeCount*t.constants.pointsPerEdge,declarations:`${g(t.constants)}
${b(!!t.inputs.mask)}
${x(t.constants.geographic)}`,body:`let edge = index / POINTS;
  let point = index % POINTS;
  if (!isEdgeLive(edge)) {
    work[workOffset + index * 2u] = DEAD_COORDINATE;
    work[workOffset + index * 2u + 1u] = DEAD_COORDINATE;
    return;
  }
  let workBox = getBox();
  let a = toWork(workBox, getVertexPosition(edgeSources[edgeSourcesOffset + edge]));
  let b = toWork(workBox, getVertexPosition(edgeTargets[edgeTargetsOffset + edge]));
  let t = f32(point) / f32(POINTS - 1u);
  var placed = a + (b - a) * t;
  if (point == 0u) {
    placed = a;
  } else if (point == POINTS - 1u) {
    placed = b;
  }
  work[workOffset + index * 2u] = placed.x;
  work[workOffset + index * 2u + 1u] = placed.y;`})}function j(e,t){return l(e,{id:t.id,operation:d,variant:`fill`,record:t.record,bindings:[{name:`density`,view:t.density,type:`u32`,access:`read_write`}],invocationCount:t.density.length,body:`density[densityOffset + index] = 0u;`,...E(t.gate,t.id,`clear`)})}function M(e,t){let{constants:n}=t,r=[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read`},{name:`density`,view:t.density,type:`atomic<u32>`,access:`read_write`}];return t.parameters&&r.push({name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`}),l(e,{id:t.id,operation:d,variant:`splat`,record:t.record,bindings:r,invocationCount:n.edgeCount*n.pointsPerEdge,declarations:`${g(n)}
${_(t.parameters)}
${v()}`,body:`let radius = getIterationRadius();
  let px = work[workOffset + index * 2u];
  let py = work[workOffset + index * 2u + 1u];
  if (px < DEAD_LIMIT) {
    return;
  }
  let gx = px * RESOLUTION_F;
  let gy = py * RESOLUTION_F;
  let radiusCells = radius * RESOLUTION_F;
  let reach = i32(max(1.0, ceil(radiusCells)));
  let inverseSquare = 1.0 / (radiusCells * radiusCells);
  let cx = i32(floor(gx));
  let cy = i32(floor(gy));
  let maximumCell = i32(RESOLUTION) - 1;
  let x0 = max(cx - reach, 0);
  let x1 = min(cx + reach, maximumCell);
  let y0 = max(cy - reach, 0);
  let y1 = min(cy + reach, maximumCell);
  for (var y = y0; y <= y1; y++) {
    let dy = f32(y) + 0.5 - gy;
    for (var x = x0; x <= x1; x++) {
      let dx = f32(x) + 0.5 - gx;
      let q = (dx * dx + dy * dy) * inverseSquare;
      if (q < 1.0) {
        let weight = u32((1.0 - q) * FIXED_POINT_SCALE + 0.5);
        if (weight > 0u) {
          atomicAdd(&density[densityOffset + u32(y) * RESOLUTION + u32(x)], weight);
        }
      }
    }
  }`,...E(t.gate,t.id,`splat`)})}function N(e,t){let{constants:n}=t,r=[{name:`iterationInfo`,view:t.iterationInfo,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read_write`},{name:`density`,view:t.density,type:`u32`,access:`read`}];return t.parameters&&r.push({name:`parameters`,view:t.parameters,type:t.parameters.format===`float32`?`f32`:`u32`,access:`read`}),l(e,{id:t.id,operation:d,variant:`update`,record:t.record,bindings:r,invocationCount:n.edgeCount,workgroupSize:64,declarations:`${g(n)}
${_(t.parameters)}
${v()}
fn readDensity(x: i32, y: i32) -> f32 {
  return f32(density[densityOffset + u32(y) * RESOLUTION + u32(x)]) * FIXED_POINT_INVERSE;
}
/** Bilinear density at grid coordinates where integers are cell centers; edges clamp. */
fn sampleDensity(gx: f32, gy: f32) -> f32 {
  let maximumCell = f32(RESOLUTION) - 1.0;
  let cx = clamp(gx, 0.0, maximumCell);
  let cy = clamp(gy, 0.0, maximumCell);
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = min(x0 + 1, i32(RESOLUTION) - 1);
  let y1 = min(y0 + 1, i32(RESOLUTION) - 1);
  let fx = cx - f32(x0);
  let fy = cy - f32(y0);
  return readDensity(x0, y0) * (1.0 - fx) * (1.0 - fy) +
    readDensity(x1, y0) * fx * (1.0 - fy) +
    readDensity(x0, y1) * (1.0 - fx) * fy +
    readDensity(x1, y1) * fx * fy;
}`,body:`let radius = getIterationRadius();
  let base = index * POINTS;
  if (work[workOffset + base * 2u] < DEAD_LIMIT) {
    return;
  }
  var points: array<vec2<f32>, ${n.pointsPerEdge}>;
  var resampled: array<vec2<f32>, ${n.pointsPerEdge}>;
  for (var i = 0u; i < POINTS; i++) {
    points[i] = vec2<f32>(
      work[workOffset + (base + i) * 2u],
      work[workOffset + (base + i) * 2u + 1u]
    );
  }
  // Advect interior points one scaled bandwidth along the bilinearly sampled, normalized gradient.
  let advectionStep = radius * readStepScale();
  for (var i = 1u; i < POINTS - 1u; i++) {
    let gx = points[i].x * RESOLUTION_F - 0.5;
    let gy = points[i].y * RESOLUTION_F - 0.5;
    if (gx < 1.0 || gy < 1.0 || gx >= RESOLUTION_F - 2.0 || gy >= RESOLUTION_F - 2.0) {
      continue;
    }
    let gradientX = sampleDensity(gx + 1.0, gy) - sampleDensity(gx - 1.0, gy);
    let gradientY = sampleDensity(gx, gy + 1.0) - sampleDensity(gx, gy - 1.0);
    let gradientLength = sqrt(gradientX * gradientX + gradientY * gradientY);
    if (gradientLength < GRADIENT_MINIMUM) {
      continue;
    }
    points[i] = points[i] + vec2<f32>(gradientX / gradientLength, gradientY / gradientLength) * advectionStep;
  }
  // Resample to uniform arc length, keeping both endpoints.
  var total = 0.0;
  for (var i = 1u; i < POINTS; i++) {
    let delta = points[i] - points[i - 1u];
    total = total + sqrt(delta.x * delta.x + delta.y * delta.y);
  }
  var segment = 1u;
  var consumed = 0.0;
  var delta = points[1] - points[0];
  var segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
  for (var i = 0u; i < POINTS; i++) {
    let arcTarget = total * f32(i) / f32(POINTS - 1u);
    while (segment < POINTS - 1u && consumed + segmentLength < arcTarget) {
      consumed = consumed + segmentLength;
      segment = segment + 1u;
      delta = points[segment] - points[segment - 1u];
      segmentLength = sqrt(delta.x * delta.x + delta.y * delta.y);
    }
    let fraction = select((arcTarget - consumed) / segmentLength, 0.0, segmentLength < LENGTH_EPSILON);
    let previous = points[segment - 1u];
    resampled[i] = previous + (points[segment] - previous) * fraction;
  }
  resampled[0] = points[0];
  resampled[POINTS - 1u] = points[POINTS - 1u];
  // One simultaneous Laplacian pass from the resampled snapshot.
  let smoothing = readSmoothing();
  for (var i = 0u; i < POINTS; i++) {
    var position = resampled[i];
    if (i > 0u && i < POINTS - 1u) {
      let midpoint = 0.5 * (resampled[i - 1u] + resampled[i + 1u]) - position;
      position = position + smoothing * midpoint;
    }
    work[workOffset + (base + i) * 2u] = position.x;
    work[workOffset + (base + i) * 2u + 1u] = position.y;
  }`,...E(t.gate,t.id,`update`)})}function P(e,t){return l(e,{id:t.id,operation:d,variant:`finalize`,bindings:[...S(t.inputs),{name:`boxKeys`,view:t.boxKeys,type:`u32`,access:`read`},{name:`work`,view:t.work,type:`f32`,access:`read`},{name:`paths`,view:t.paths,type:`f32`,access:`read_write`}],invocationCount:t.constants.edgeCount*t.constants.pointsPerEdge,declarations:`${g(t.constants)}
${b(!!t.inputs.mask)}
${x(t.constants.geographic)}`,body:`let edge = index / POINTS;
  let point = index % POINTS;
  let source = edgeSources[edgeSourcesOffset + edge];
  var outputPoint = vec2<f32>(0.0, 0.0);
  if (!isEdgeLive(edge)) {
    if (source < VERTEX_COUNT) {
      outputPoint = getVertexPosition(source);
    }
  } else if (point == 0u) {
    outputPoint = getVertexPosition(source);
  } else if (point == POINTS - 1u) {
    outputPoint = getVertexPosition(edgeTargets[edgeTargetsOffset + edge]);
  } else {
    let workBox = getBox();
    let normalized = vec2<f32>(work[workOffset + index * 2u], work[workOffset + index * 2u + 1u]);
    outputPoint = fromWork(workBox, normalized);
  }
  paths[pathsOffset + index * 2u] = outputPoint.x;
  paths[pathsOffset + index * 2u + 1u] = outputPoint.y;`})}function F(e,t){let n=[];return t.startIndices&&n.push({name:`startIndices`,view:t.startIndices,type:`u32`,access:`read_write`}),t.drawRecord&&n.push({name:`drawRecord`,view:t.drawRecord,type:`u32`,access:`read_write`}),l(e,{id:t.id,operation:d,variant:`indices`,bindings:n,invocationCount:Math.max(t.constants.edgeCount+1,4),declarations:g(t.constants),body:`${t.startIndices?`if (index <= EDGE_COUNT) {
    startIndices[startIndicesOffset + index] = index * POINTS;
  }`:``}
  ${t.drawRecord?`if (index < 4u) {
    drawRecord[drawRecordOffset + index] = select(select(0u, EDGE_COUNT, index == 1u), POINTS, index == 0u);
  }`:``}`})}var I=.05,L={pointsPerEdge:16,maximumIterations:15,densityResolution:256,kernelRadius:.03,lambda:.85,smoothing:.5,stepScale:1},R={coordinateSpaces:[`planar`,`longitude-latitude`]},z=16,B=8,V=8;function H(e){let t=z;for(;t>0&&e*2**t>4294967295;)t--;return t}function U(e,t){let n=Float32Array.of(e.activeIterations??64,e.kernelRadius??L.kernelRadius,e.lambda??L.lambda,e.smoothing??L.smoothing,e.stepScale??L.stepScale);if(t===`float32`)return n;let r=new Uint32Array(n.buffer.slice(0));return r[0]=Math.max(0,Math.floor(e.activeIterations??64)),r}var W=class{id;props;pointsPerEdge;maximumIterations;densityResolution;edgeCount;fixedPointExponent;spatialContext;constructor(e){this.id=e.id??`edge-bundling`,this.props=e,this.pointsPerEdge=e.pointsPerEdge??L.pointsPerEdge,this.maximumIterations=e.maximumIterations??L.maximumIterations,this.densityResolution=e.densityResolution??L.densityResolution,this.edgeCount=e.sourceNodes.length;let{id:n,pointsPerEdge:i,maximumIterations:a,densityResolution:o,edgeCount:c}=this,{output:l}=e;this.spatialContext=t(n,e.spatialContext,R),s(e.nodePositions,[`float32x2`],`${n} nodePositions`),s(l.paths,[`float32x2`],`${n} output.paths`);for(let[t,r]of[[`sourceNodes`,e.sourceNodes],[`targetNodes`,e.targetNodes],[`edgeMask`,e.edgeMask],[`output.lineOffsets`,l.lineOffsets],[`output.drawRecord`,l.drawRecord]])r&&s(r,[`uint32`],`${n} ${t}`);if(e.parameters&&(s(e.parameters,[`uint32`,`float32`],`${n} parameters`),e.parameters.length<4))throw Error(`${n} parameters must contain at least 4 rows`);if(c<1)throw Error(`${n} must have at least one edge`);if(e.nodePositions.length<1)throw Error(`${n} nodePositions must contain at least one node`);if(e.targetNodes.length!==c)throw Error(`${n} targetNodes length must equal sourceNodes length`);if(e.edgeMask&&e.edgeMask.length!==c)throw Error(`${n} edgeMask length must equal the edge count`);if(r(n,`pointsPerEdge`,i,2,64),r(n,`maximumIterations`,a,1,64),r(n,`densityResolution`,o,V),l.paths.length!==c*i)throw Error(`${n} output.paths length must equal edgeCount * pointsPerEdge`);if(l.lineOffsets&&l.lineOffsets.length!==c+1)throw Error(`${n} output.lineOffsets length must equal edgeCount + 1`);if(l.drawRecord&&l.drawRecord.length!==4)throw Error(`${n} output.drawRecord must contain exactly four rows`);if(this.fixedPointExponent=H(c*i),this.fixedPointExponent<B)throw Error(`${n} edgeCount * pointsPerEdge is too large for 32-bit density`);let u=[l.paths,l.lineOffsets,l.drawRecord].filter(e=>e!==void 0).map(e=>e.buffer),d=[e.nodePositions,e.sourceNodes,e.targetNodes,e.edgeMask,e.parameters].filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(u).size!==u.length||u.some(e=>d.includes(e)))throw Error(`${n} outputs must use separate buffers from each other and from inputs`)}getCommandNodes(t){let{id:r,props:o,maximumIterations:s,pointsPerEdge:c,densityResolution:l,edgeCount:u}=this,{output:d}=o;if(n(r,t,[o.nodePositions,o.sourceNodes,o.targetNodes,o.edgeMask,o.parameters,d.paths,d.lineOffsets,d.drawRecord]),l*l*Uint32Array.BYTES_PER_ELEMENT>t.device.limits.maxStorageBufferBindingSize)throw Error(`${r} density grid exceeds maxStorageBufferBindingSize`);let f={edgeCount:u,vertexCount:o.nodePositions.length,pointsPerEdge:c,densityResolution:l,fixedPointExponent:this.fixedPointExponent,iterationCount:s,boxPadding:I,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`},p={positions:o.nodePositions,sources:o.sourceNodes,targets:o.targetNodes,mask:o.edgeMask},m=a(t,`${r}-box`,`uint32`,4),h=a(t,`${r}-density`,`uint32`,l*l);i(t.device,r,`work`,u*c*8,`lower pointsPerEdge or split the edges`);let g=a(t,`${r}-work`,`float32x2`,u*c),_=a(t,`${r}-iteration-info`,`uint32`,s*64),v=o.parameters?a(t,`${r}-gate`,`uint32`,s*9,e.STORAGE|e.INDIRECT):void 0,b=[O(t,{id:`${r}-box-reset`,boxKeys:m}),k(t,{id:`${r}-box`,constants:f,inputs:p,boxKeys:m}),A(t,{id:`${r}-initialize`,constants:f,inputs:p,boxKeys:m,work:g})];b.push(y(t,{id:`${r}-iteration-info`,iterationInfo:_,iterationCount:s})),v&&o.parameters&&b.push(D(t,{id:`${r}-gate`,constants:f,parameters:o.parameters,gate:v}));for(let e=0;e<s;e++){let n={constants:f,iterationInfo:t.createDataView(_.buffer,{format:`uint32`,length:1,byteOffset:_.byteOffset+4*e*64}),parameters:o.parameters,work:g,density:h,gate:v&&{view:v,iteration:e}},i={loop:{id:`${r}-iteration`,iteration:e,phase:v?`gated`:`unconditional`,maximumIterations:s}};b.push(j(t,{id:`${r}-clear-${e}`,density:h,gate:n.gate,record:i}),M(t,{id:`${r}-splat-${e}`,...n,record:i}),N(t,{id:`${r}-update-${e}`,...n,record:i}))}return b.push(P(t,{id:`${r}-finalize`,constants:f,inputs:p,boxKeys:m,work:g,paths:d.paths})),(d.lineOffsets||d.drawRecord)&&b.push(F(t,{id:`${r}-indices`,constants:f,startIndices:d.lineOffsets,drawRecord:d.drawRecord})),b}};export{I as n,U as r,W as t};