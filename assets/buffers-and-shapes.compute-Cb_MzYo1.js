import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{E as t,p as n,w as r}from"./analysis-frame-Blu0tYdw.js";import{N as i,_ as a,d as o,h as s,n as c,t as l,w as u,y as d}from"./resources-CwvhEOBe.js";import{t as f}from"./gpu-sort-Bl3Dztdb.js";import{l as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{f as h,l as g,s as _}from"./wgsl-common-BNRVzgd8.js";import{t as v}from"./bounds-reduction-cUiPIxi7.js";import{n as y,r as b,t as x}from"./gpu-grid-generator-2Y70aLan.js";import{n as S,r as C,t as w}from"./gpu-outline-geometry-DW4RTkmU.js";import{n as T,t as E}from"./gpu-rectangle-clip-D_EZ8d4-.js";import{a as D}from"./index-Bjk18V2I.js";import{t as O}from"./graph-swap-41Ru7LVu.js";import{t as k}from"./summary-reader-CsgWxENf.js";import{t as A}from"./mode-kernels-Dxm2GWvI.js";import{t as j}from"./b3-palette-APlD7x7U.js";import{a as M,i as N,o as P,r as F,t as I}from"./b3-layers-B2V1ljKW.js";import{a as L,i as R,l as z,o as B,p as V,s as H,t as U,u as W}from"./b3-common-BbHtVprn.js";import{i as ee,n as te,t as G}from"./b3-city-data-BBHr728V.js";var K=`GPUShapeGenerator`,ne=Object.freeze({...n,measure:`distance`});function re(e,t=new Float32Array(2)){if(t.length<2)throw Error(`Shape generator target must hold 2 elements`);let{segmentCount:n,radiusScale:r=1}=e;if(!Number.isFinite(n)||!Number.isFinite(r))throw Error(`Shape generator settings must be finite`);if(n<1||r<0)throw Error(`Shape generator segmentCount must be at least 1 and radiusScale >= 0`);return t.set([Math.floor(n),r]),t}function q(e){return e===`sector`?1:3}function J(e,t){return e===`sector`?t+3:t+1}var ie=class{id;props;spatialContext;coordinateSystem;sphereRadius;featureCount;maximumSegments;constructor(e){let n=e.id??`shape-generator`;if(this.id=n,this.props=e,e.shape!==`circle`&&e.shape!==`sector`&&e.shape!==`ellipse`)throw Error(`${n} unknown shape ${String(e.shape)}`);if(this.spatialContext=t(n,e.spatialContext,ne),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`geodesic`:`planar`,this.sphereRadius=this.coordinateSystem===`geodesic`?r(this.spatialContext):1,e.ellipseSpacing!==void 0&&(e.shape!==`ellipse`||e.ellipseSpacing!==`parameter`&&e.ellipseSpacing!==`arc-length`))throw Error(`${n} ellipseSpacing must be 'parameter' or 'arc-length' for ellipses`);this.maximumSegments=e.maximumSegments??64;let i=q(e.shape);if(o(n,`maximumSegments`,this.maximumSegments,i,65536),d(e.centers,[`float32x2`],`${n} centers`),this.featureCount=e.centers.length,this.featureCount<1)throw Error(`${n} centers must hold at least one row`);if(d(e.radii,[e.shape===`ellipse`?`float32x2`:`float32`],`${n} radii`),e.radii.length!==this.featureCount)throw Error(`${n} radii must have one row per center`);if(e.shape===`sector`){if(!e.bearings)throw Error(`${n} sector shapes need bearings`);if(d(e.bearings,[`float32x2`],`${n} bearings`),e.bearings.length!==this.featureCount)throw Error(`${n} bearings must have one row per center`)}else if(e.bearings)throw Error(`${n} ${e.shape} shapes have no bearings`);if(e.rotations){if(e.shape!==`ellipse`)throw Error(`${n} only ellipse shapes have rotations`);if(d(e.rotations,[`float32`],`${n} rotations`),e.rotations.length!==this.featureCount)throw Error(`${n} rotations must have one row per center`)}if(d(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<2)throw Error(`${n} parameters must hold 2 float32 values`);let{positions:a,offsets:c,vertexCount:l}=e.output;d(a,[`float32x2`],`${n} output.positions`);let u=this.featureCount*J(e.shape,this.maximumSegments);if(a.length<u)throw Error(`${n} output.positions must hold at least ${u} rows`);if(d(c,[`uint32`],`${n} output.offsets`),c.length!==this.featureCount+1)throw Error(`${n} output.offsets must hold featureCount + 1 rows`);if(l&&(d(l,[`uint32`],`${n} output.vertexCount`),l.length<1))throw Error(`${n} output.vertexCount must hold one uint32`);s(n,[a,c,l],[e.centers,e.radii,e.bearings,e.rotations,e.parameters])}getCommandNodes(e){let{props:t,id:n,featureCount:r,maximumSegments:i}=this,{positions:o,offsets:s,vertexCount:c}=t.output;a(n,e,[t.centers,t.radii,t.bearings,t.rotations,t.parameters,o,s,c]);let{shape:l}=t,d=this.coordinateSystem===`geodesic`,f=t.ellipseSpacing===`arc-length`?u(e,`${n}-arc-lengths`,`float32`,r*257):void 0,h=[],g=`
const QUARTER_TURN: f32 = 1.5707963267948966;
const ARC_INTERVALS: u32 = 256u;

fn getEllipseRadii(feature: u32) -> vec2<f32> {
  var axes = vec2<f32>(
    radii[radiiOffset + 2u * feature], radii[radiiOffset + 2u * feature + 1u]) *
    parameters[parametersOffset + 1u];
  if ((bitcast<u32>(axes.x) & 0x7f800000u) == 0x7f800000u || axes.x < 0.0) {
    axes.x = 0.0;
  }
  if ((bitcast<u32>(axes.y) & 0x7f800000u) == 0x7f800000u || axes.y < 0.0) {
    axes.y = 0.0;
  }
  return axes;
}`;f&&h.push(m(e,{id:`${n}-arc-lengths`,operation:K,variant:`ellipse-arc-lengths`,bindings:[{name:`radii`,view:t.radii,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`arcLengths`,view:f,type:`f32`,access:`read_write`}],invocationCount:r,declarations:g,body:`let axes = getEllipseRadii(index);
  // Normalize before squaring to keep the table independent of world-coordinate scale.
  let maximumRadius = max(axes.x, axes.y);
  var normalizedAxes = vec2<f32>(0.0);
  if (maximumRadius > 0.0) { normalizedAxes = axes / maximumRadius; }
  let base = arcLengthsOffset + index * (ARC_INTERVALS + 1u);
  arcLengths[base] = 0.0;
  var total = 0.0;
  for (var interval = 1u; interval <= ARC_INTERVALS; interval++) {
    // The exact chord uses the midpoint derivative times 2 sin(delta / 2), avoiding
    // cancellation from subtracting nearby ellipse coordinates. Quarter symmetry lets
    // every vertex reuse one table. The polygonal length converges quadratically.
    let halfStep = QUARTER_TURN / (2.0 * f32(ARC_INTERVALS));
    let midpoint = (2.0 * f32(interval) - 1.0) * halfStep;
    total += 2.0 * sin(halfStep) * length(normalizedAxes * vec2<f32>(sin(midpoint), cos(midpoint)));
    arcLengths[base + interval] = total;
  }`}));let v=J(l,i),y=`const FEATURE_COUNT: u32 = ${r}u;
const MAXIMUM_SEGMENTS: u32 = ${i}u;
const MINIMUM_SEGMENTS: u32 = ${q(l)}u;
const MAXIMUM_VERTICES: u32 = ${v}u;
const IS_SECTOR: bool = ${l===`sector`};

fn getSegmentCount(rawSegments: f32) -> u32 {
  return clamp(u32(max(rawSegments, 0.0)), MINIMUM_SEGMENTS, MAXIMUM_SEGMENTS);
}

fn getVertexCount(segments: u32) -> u32 {
  return select(segments + 1u, segments + 3u, IS_SECTOR);
}`,b=[{name:`centers`,view:t.centers,type:`f32`,access:`read`},{name:`radii`,view:t.radii,type:`f32`,access:`read`},...t.bearings?[{name:`bearings`,view:t.bearings,type:`f32`,access:`read`}]:[],...t.rotations?[{name:`rotations`,view:t.rotations,type:`f32`,access:`read`}]:[],{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},...f?[{name:`arcLengths`,view:f,type:`f32`,access:`read`}]:[],{name:`positionsOut`,view:o,type:`f32`,access:`read_write`}];return h.push(m(e,{id:`${n}-vertices`,operation:K,variant:`${l}-${d?`geodesic`:`planar`}${t.rotations?`-rotated`:``}${f?`-arc-length`:``}`,bindings:b,invocationCount:r*v,declarations:`${y}
const EARTH_RADIUS: f32 = ${p(this.sphereRadius)};
const DEGREES: f32 = 0.017453292519943295;
const TWO_PI: f32 = 6.283185307179586;
const NOT_A_NUMBER: u32 = 0x7fc00000u;
${l===`ellipse`?g:``}
${f?`
fn getEllipsePhase(feature: u32, step: u32, segments: u32, axes: vec2<f32>) -> f32 {
  let fraction = f32(step) / f32(segments);
  // Equal axes already have equal arc intervals. This also handles collapsed points.
  if (axes.x == axes.y) { return TWO_PI * fraction; }
  // Integer quadrant arithmetic keeps cardinal vertices and the closure exact.
  let quadrant = (4u * step) / segments;
  let quarterFraction = f32((4u * step) % segments) / f32(segments);
  let reverse = (quadrant & 1u) != 0u;
  let targetFraction = select(quarterFraction, 1.0 - quarterFraction, reverse);
  let base = arcLengthsOffset + feature * (ARC_INTERVALS + 1u);
  let targetLength = targetFraction * arcLengths[base + ARC_INTERVALS];
  var lower = 0u;
  var upper = ARC_INTERVALS;
  for (var iteration = 0u; iteration < 8u; iteration++) {
    let middle = (lower + upper) / 2u;
    if (arcLengths[base + middle] < targetLength) { lower = middle; } else { upper = middle; }
  }
  let start = arcLengths[base + lower];
  let span = arcLengths[base + upper] - start;
  var weight = 0.0;
  if (span > 0.0) { weight = clamp((targetLength - start) / span, 0.0, 1.0); }
  let phase = QUARTER_TURN * (f32(lower) + weight) / f32(ARC_INTERVALS);
  return f32(quadrant) * QUARTER_TURN + select(phase, QUARTER_TURN - phase, reverse);
}`:``}

${_}

// Point at the given distance (world units, or meters for geodesic) and bearing (radians
// clockwise from north) from the center.
fn getDestination(center: vec2<f32>, distance: f32, bearing: f32) -> vec2<f32> {
${d?`  let angularDistance = distance / EARTH_RADIUS;
  let latitude = center.y * DEGREES;
  let sinDistance = sin(angularDistance);
  let cosDistance = cos(angularDistance);
  let sinLatitude = sin(latitude);
  let cosLatitude = cos(latitude);
  let sinLatitude2 = clamp(
    sinLatitude * cosDistance + cosLatitude * sinDistance * cos(bearing), -1.0, 1.0);
  let latitude2 = asin(sinLatitude2);
  // Longitude difference directly, so the small offset is not lost to radians rounding.
  let longitudeDelta = atan2(
    sin(bearing) * sinDistance * cosLatitude, cosDistance - sinLatitude * sinLatitude2);
  return vec2<f32>(center.x + longitudeDelta / DEGREES, latitude2 / DEGREES);`:`  return center + distance * vec2<f32>(sin(bearing), cos(bearing));`}
}`,body:`let feature = index / MAXIMUM_VERTICES;
  let local = index % MAXIMUM_VERTICES;
  let segments = getSegmentCount(parameters[parametersOffset]);
  let vertices = getVertexCount(segments);
  if (local >= vertices) {
    return;
  }
  let radiusScale = parameters[parametersOffset + 1u];
  let center = vec2<f32>(
    centers[centersOffset + 2u * feature], centers[centersOffset + 2u * feature + 1u]);
  var position = center;
${l===`ellipse`?`  let radii2 = getEllipseRadii(feature);
  // Counter-clockwise from the x semi-axis, which points along bearing 90 degrees + rotation.
  let phase = ${f?`getEllipsePhase(feature, local % segments, segments, radii2)`:`TWO_PI * f32(local % segments) / f32(segments)`};
  let east = radii2.x * cos(phase);
  let north = radii2.y * sin(phase);
  ${t.rotations?`let tilt = rotations[rotationsOffset + feature] * DEGREES;`:`let tilt: f32 = 0.0;`}
  // Rotate the (east, north) offset clockwise by tilt.
  let rotatedEast = east * cos(tilt) + north * sin(tilt);
  let rotatedNorth = north * cos(tilt) - east * sin(tilt);
  ${f&&!d?`position = center + vec2<f32>(rotatedEast, rotatedNorth);`:`let distance = length(vec2<f32>(rotatedEast, rotatedNorth));
  var bearing = atan2(rotatedEast, rotatedNorth);
  ${f?`// Avoid backend atan2 signed-zero ambiguity for an east/west degenerate axis.
  if (rotatedNorth == 0.0) {
    bearing = select(-QUARTER_TURN, QUARTER_TURN, rotatedEast >= 0.0);
  }`:``}
  position = getDestination(center, distance, bearing);`}`:`  var radius = radii[radiiOffset + feature] * radiusScale;
  if (!isFiniteBits(radius) || radius < 0.0) { radius = 0.0; }
${l===`sector`?`  var bearing1 = bearings[bearingsOffset + 2u * feature];
  var bearing2 = bearings[bearingsOffset + 2u * feature + 1u];
  bearing1 = bearing1 - 360.0 * floor(bearing1 / 360.0);
  bearing2 = bearing2 - 360.0 * floor(bearing2 / 360.0);
  let sweep = select(bearing2 - bearing1 + 360.0, bearing2 - bearing1, bearing1 < bearing2);
  // Slot 0 and the last slot are the center; slots 1..=segments+1 are the arc.
  if (local != 0u && local != vertices - 1u) {
    let step = f32(local - 1u) / f32(segments);
    position = getDestination(center, radius, (bearing1 + sweep * step) * DEGREES);
  }`:`  // Counter-clockwise ring, bearings 0, -360 / n, ...; the last vertex repeats the first.
  let ringStep = f32(local % segments) / f32(segments);
  position = getDestination(center, radius, -ringStep * TWO_PI);`}`}
  if (!isFiniteBits(center.x) || !isFiniteBits(center.y)) {
    position = vec2<f32>(bitcast<f32>(NOT_A_NUMBER | (feature & 0u)));
  }
  let base = positionsOutOffset + 2u * (feature * vertices + local);
  positionsOut[base] = position.x;
  positionsOut[base + 1u] = position.y;`}),m(e,{id:`${n}-offsets`,operation:K,variant:`offsets`,bindings:[{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`offsetsOut`,view:s,type:`u32`,access:`read_write`},...c?[{name:`vertexCountOut`,view:c,type:`u32`,access:`read_write`}]:[]],invocationCount:r+1,declarations:y,body:`let vertices = getVertexCount(getSegmentCount(parameters[parametersOffset]));
  offsetsOut[offsetsOutOffset + index] = index * vertices;
  ${c?`if (index == FEATURE_COUNT) { vertexCountOut[vertexCountOutOffset] = index * vertices; }`:``}`})),h}},ae=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metrics:Object.freeze([`none`,`native`])}),Y=`GPUHilbertKeys`;function oe(e){return e>=16?4294967295:4**e}var X=class{id;props;spatialContext;order;itemCount;constructor(e){let n=e.id??`hilbert-keys`;this.id=n,this.props=e,this.spatialContext=t(this.id,e.spatialContext,ae),this.order=e.order??16,o(n,`order`,this.order,1,16);let{positions:r,minima:i,maxima:a,bounds:c}=e;if(r){if(i||a)throw Error(`${n} takes positions or minima and maxima, not both`);d(r,[`float32x2`],`${n} positions`),this.itemCount=r.length}else{if(!i||!a)throw Error(`${n} needs positions, or minima and maxima`);if(d(i,[`float32x2`],`${n} minima`),d(a,[`float32x2`],`${n} maxima`),i.length!==a.length)throw Error(`${n} minima and maxima must have the same length`);this.itemCount=i.length}if(this.itemCount<1)throw Error(`${n} needs at least one item`);if(c&&(d(c,[`float32`],`${n} bounds`),c.length<4))throw Error(`${n} bounds must hold 4 float32 values`);let{keys:l,sortedRows:u,sortedKeys:f}=e.output;for(let[e,t]of[[`keys`,l],[`sortedRows`,u],[`sortedKeys`,f]])if(t&&(d(t,[`uint32`],`${n} output.${e}`),t.length!==this.itemCount))throw Error(`${n} output.${e} must have one row per item`);if(f&&!u)throw Error(`${n} output.sortedKeys needs output.sortedRows`);s(n,[l,u,f],[r,i,a,c])}getCommandNodes(e){let{props:t,id:n,order:r,itemCount:i}=this,{positions:o,minima:s,maxima:c}=t,{keys:l,sortedRows:d,sortedKeys:p}=t.output;a(n,e,[o,s,c,t.bounds,l,d,p]);let _=[],y=o?[{name:`positions`,view:o,type:`f32`,access:`read`}]:[{name:`minima`,view:s,type:`f32`,access:`read`},{name:`maxima`,view:c,type:`f32`,access:`read`}],b=`${g}
// Returns (x, y, valid, 0) for an item: the point, or the box center (halved before the sum to
// avoid overflow near FLT_MAX).
fn readItem(row: u32) -> vec4f {
${o?`  let point = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  let valid = isFiniteValue(point.x) && isFiniteValue(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);`:`  let minimum = vec2f(minima[minimaOffset + row * 2u], minima[minimaOffset + row * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + row * 2u], maxima[maximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  return vec4f(minimum * 0.5 + maximum * 0.5, select(0.0, 1.0, valid), 0.0);`}
}`,x=t.bounds;x||(x=u(e,`${n}-bounds`,`float32`,4),_.push(...v(e,{id:n,operation:Y,variant:o?`bounds-points`:`bounds-boxes`,bindings:y,declarations:b,itemCount:i,output:x})));let S=d?u(e,`${n}-rows`,`uint32`,i):void 0;return _.push(m(e,{id:`${n}-keys`,operation:Y,variant:`${o?`points`:`boxes`}-order-${r}${S?`-rows`:``}`,bindings:[...y,{name:`bounds`,view:x,type:`f32`,access:`read`},{name:`keysOut`,view:l,type:`u32`,access:`read_write`},...S?[{name:`rowsOut`,view:S,type:`u32`,access:`read_write`}]:[]],invocationCount:i,declarations:`${b}
const ORDER: u32 = ${r}u;
const CELLS: u32 = ${2**r}u;

fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || !isFiniteValue(extent)) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(floor(normalized * f32(CELLS))), CELLS - 1u);
}

// Spreads the low 16 bits of a value to the even bit positions.
${h}

// The Hilbert curve index of cell (x, y) on a CELLS x CELLS grid, equal to the classic xy2d
// rotation loop. The loop carries a rotate/flip state from the top bit down, so its 16 steps
// are serial and branch on data. The state transform of each bit level is a small affine map over
// GF(2), so the same result comes from a parallel prefix over the levels (Hacker's Delight 16-5,
// as in rawrunprotected's "Hilbert curves in O(log n)"): four branch-free rounds on 16-bit lanes,
// then one interleave.
fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  let x = cellX << (16u - ORDER);
  let y = cellY << (16u - ORDER);
  let a0 = x ^ y;
  let b0 = 0xffffu ^ a0;
  let c0 = 0xffffu ^ (x | y);
  let d0 = x & (y ^ 0xffffu);
  var transformA = a0 | (b0 >> 1u);
  var transformB = (a0 >> 1u) ^ a0;
  var transformC = ((c0 >> 1u) ^ (b0 & (d0 >> 1u))) ^ c0;
  var transformD = ((a0 & (c0 >> 1u)) ^ (d0 >> 1u)) ^ d0;
  for (var shift = 2u; shift <= 4u; shift = shift << 1u) {
    let a = transformA;
    let b = transformB;
    let c = transformC;
    let d = transformD;
    transformA = (a & (a >> shift)) ^ (b & (b >> shift));
    transformB = (a & (b >> shift)) ^ (b & ((a ^ b) >> shift));
    transformC = transformC ^ ((a & (c >> shift)) ^ (b & (d >> shift)));
    transformD = transformD ^ ((b & (c >> shift)) ^ ((a ^ b) & (d >> shift)));
  }
  let a = transformA;
  let b = transformB;
  let c = transformC;
  let d = transformD;
  transformC = c ^ ((a & (c >> 8u)) ^ (b & (d >> 8u)));
  transformD = d ^ ((b & (c >> 8u)) ^ ((a ^ b) & (d >> 8u)));
  let low = transformC ^ (transformC >> 1u);
  let high = transformD ^ (transformD >> 1u);
  let indexLow = x ^ y;
  let indexHigh = high | (0xffffu ^ (indexLow | low));
  return ((spreadBits(indexHigh) << 1u) | spreadBits(indexLow)) >> (32u - 2u * ORDER);
}`,body:`${S?`rowsOut[rowsOutOffset + index] = index;`:``}
  let item = readItem(index);
  var key = ${oe(r)}u;
  if (item.z > 0.5) {
    let cellX = quantizeAxis(item.x, bounds[boundsOffset], bounds[boundsOffset + 2u]);
    let cellY = quantizeAxis(item.y, bounds[boundsOffset + 1u], bounds[boundsOffset + 3u]);
    key = getHilbertIndex(cellX, cellY);
  }
  keysOut[keysOutOffset + index] = key;`})),d&&S&&_.push(...new f({id:`${n}-sort`,keys:l,values:S,outputKeys:p??u(e,`${n}-sorted-keys`,`uint32`,i),outputValues:d,algorithm:`radix`,keyBits:Math.min(32,2*r+1)}).getCommandNodes(e)),_}},Z=e.METER_OFFSETS,se=e.LNGLAT,ce=(e,t)=>[e[0],e[1],e[2],t];async function le(e){let t=te(e.datasets.get(`cta-transit`),e.datasets.get(`chicago-community-areas`)),n=null,r={ctx:e,city:t,coordinateOrigin:[t.origin[0],t.origin[1],0],loadStreets:()=>(n??=ee(e.datasets.get(`chicago-roads`),t.projection),n)},i=new Map,a={buffers:ue,shapes:de,grid:fe,clip:pe},o=!1,s=new O,c=await a[e.options.view](r);i.set(e.options.view,c);let l=e.options.view,u=new Map,d=()=>c,f=t=>{s.request({build:async()=>i.get(t)??await a[t](r),commit:n=>{i.set(t,n),c=n,l=t;for(let r of u.get(t)??[])n.setOption(r,e.options);u.delete(t),e.requestLayers()},discard:e=>{i.get(t)!==e&&e.destroy()}})},p=(e,t)=>{d().setOption(e,t);for(let t of Object.keys(a)){if(t===l)continue;let n=u.get(t)??new Set;n.add(e),u.set(t,n)}};return{getCompiledGraphs:()=>d().getCompiledGraphs(),setOption(t,n,r){if(t===`view`){f(r.view);return}p(t,r),e.requestLayers()},onThemeChange:()=>e.requestLayers(),getTooltip:e=>d().getTooltip?.(e)??null,onClick:e=>d().onClick?.(e)??!1,onDragStart:e=>d().onDragStart?.(e)??!1,onDrag:e=>d().onDrag?.(e),onDragEnd:e=>d().onDragEnd?.(e),encode(e,t){o||d().encode(e,t)},getLayers:()=>d().getLayers(),destroy(){o=!0,s.destroy();for(let e of i.values())e.destroy();i.clear()}}}async function ue(e){let{ctx:t,city:n}=e,{device:r}=t,a=new l(r,`buffers`),o=()=>t.theme()===`dark`,s={stations:{type:`points`,local:a.createBuffer(`stations-local`,n.railStations.local),lngLat:a.createBuffer(`stations-lnglat`,n.railStations.lngLat),count:n.railStations.count,offsets:null,polygonOffsets:null,featureOffsets:null,pathCount:0},lines:{type:`lines`,local:a.createBuffer(`lines-local`,n.railLines.local),lngLat:a.createBuffer(`lines-lnglat`,n.railLines.lngLat),count:n.railLines.vertexCount,offsets:a.createBuffer(`lines-offsets`,n.railLines.offsets),polygonOffsets:null,featureOffsets:null,pathCount:n.railLines.pathCount},rings:{type:`polygons`,local:a.createBuffer(`rings-local`,n.areas.local),lngLat:a.createBuffer(`rings-lnglat`,n.areas.lngLat),count:n.areas.vertexCount,offsets:a.createBuffer(`rings-offsets`,n.areas.ringOffsets),polygonOffsets:a.createBuffer(`rings-polygon-offsets`,n.areas.polygonOffsets),featureOffsets:a.createBuffer(`rings-feature-offsets`,n.areas.featureOffsets),pathCount:n.areas.ringCount}},u=a.createParameterBuffer(`outline-parameters`,`float32`,4),d=H(a,`buffers-lines`,n.railLines.local,n.railLines.offsets),f=H(a,`buffers-rings`,n.areas.local,n.areas.ringOffsets),p=null,m=!0,h=async e=>{let t=`${e.bufferSource}|${e.bufferSystem}|${e.joinSegments}`;if(p?.key===t)return null;let o=s[e.bufferSource],c=e.bufferSystem===`spherical`,l=C(e.joinSegments),d=o.count*l,f=a.createBuffer(`triangles-${t}`,d*8),m=new i(r,{id:`outline-${t}`}),h=L(m),g=h(`positions`,c?o.lngLat:o.local,`float32x2`,o.count),_=o.type===`points`?{geometryType:`points`,positions:g}:o.type===`lines`?{geometryType:`lines`,positions:g,lineOffsets:h(`offsets`,o.offsets,`uint32`,o.pathCount+1)}:{geometryType:`polygons`,positions:g,featureOffsets:h(`feature-offsets`,o.featureOffsets,`uint32`,n.areas.featureCount+1),polygonOffsets:h(`polygon-offsets`,o.polygonOffsets,`uint32`,n.areas.partCount+1),ringOffsets:h(`offsets`,o.offsets,`uint32`,o.pathCount+1)};return m.add(new w({id:`outline`,..._,spatialContext:c?U:n.projection.spatialContext,joinSegments:e.joinSegments,parameters:u.importToGraph(m),output:{positions:h(`triangles`,f,`float32x2`,d)}})),{key:t,compiled:a.track(await m.compileAsync()),triangles:f,triangleCount:d/3}},g=(e,n)=>{if(!e)return;let r=p;p=e,r&&(a.release(r.compiled),a.release(r.triangles)),m=!0;let i=s[n.bufferSource],o=C(n.joinSegments),l=i.count*o;t.setReadout(`bufferInputs`,`${c(i.count)} vertices`),t.setReadout(`bufferTriangles`,`${c(l/3)} (${o} vertices each input)`)},_=e=>{u.write(S({distance:e.bufferDistance}));let n=e.bufferDistance*(1-Math.cos(Math.PI/e.joinSegments));t.setReadout(`bufferNarrowing`,`${z(n)} (${(100*(1-Math.cos(Math.PI/e.joinSegments))).toFixed(1)}%)`),m=!0};g(await h(t.options),t.options),_(t.options);let v=a.track(new O);return{getCompiledGraphs:()=>p?[p.compiled]:[],setOption(e,n){(e===`bufferSource`||e===`bufferSystem`||e===`joinSegments`)&&v.request({build:()=>h(n),commit:e=>{g(e,n),t.requestLayers()},discard:e=>{e&&(a.release(e.compiled),a.release(e.triangles))}}),(e===`bufferDistance`||e===`joinSegments`)&&_(n)},encode(e){p&&m&&(p.compiled.encode(e,{parameters:void 0}),m=!1)},getLayers(){if(!p)return[];let n=t.options,r=[new P({id:`buffer-triangles`,coordinateSystem:n.bufferSystem===`spherical`?se:Z,coordinateOrigin:e.coordinateOrigin,positions:p.triangles,triangleCount:p.triangleCount,color:ce((o(),j[0]),235),opacity:.55})];if(n.showSource){let t=o()?[255,255,255,235]:[20,30,50,235];if(n.bufferSource===`stations`)r.push(new D({id:`buffer-source-points`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:s.stations.local,instanceCount:s.stations.count,radiusPixels:2.6,color:t}));else{let i=n.bufferSource===`lines`?d:f;r.push(new N({id:`buffer-source-paths`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:i.positions,pathOffsets:i.offsets,pathOffsetCount:i.offsetCount,vertexCount:i.vertexCount,drawCommands:i.drawCommands,color:t,widthPixels:1.4}))}}return r},destroy:()=>a.destroy()}}async function de(e){let{ctx:t,city:n}=e,{device:r}=t,a=new l(r,`shapes`),o=n.railStations,s=o.count,c=n.projection.project(G[0],G[1]),u=new Float32Array(s),d=new Float32Array(s),f=new Float32Array(s);for(let e=0;e<s;e++){let t=c[0]-o.local[e*2],n=c[1]-o.local[e*2+1];u[e]=(Math.atan2(t,n)*180/Math.PI+360)%360,d[e]=Math.hypot(t,n)/1e3,f[e]=.65+.2*Math.min(o.routeCount[e],5)}let p=0;for(let e of f)p+=e;p/=s;for(let e=0;e<s;e++)f[e]/=p;let m=a.createBuffer(`centers-local`,o.local),h=a.createBuffer(`centers-lnglat`,o.lngLat),g=a.createBuffer(`circle-radii`,f),_=a.createBuffer(`ellipse-radii`,s*8),v=a.createBuffer(`rotations`,u),y=a.createBuffer(`sector-bearings`,s*8),b=a.createBuffer(`distance-values`,d),x=a.createParameterBuffer(`shape-parameters`,`float32`,2),S=new Map,C=null,w=!0,T=e=>{let t=new Float32Array(s*2);for(let n=0;n<s;n++)t[n*2]=u[n]-e.sectorSweep/2,t[n*2+1]=u[n]+e.sectorSweep/2;y.write(t)},E=e=>{let t=new Float32Array(s*2);for(let n=0;n<s;n++)t[n*2]=f[n],t[n*2+1]=f[n]*e.ellipseRatio;_.write(t)},k=(e,t,n)=>Math.min(Math.max(e.segmentCount,q(t)),n),j=e=>{x.write(re({segmentCount:e.segmentCount,radiusScale:e.shapeRadius})),w=!0},N=async e=>{let t=Number(e.maxSegments),o=`${e.shapeKind}|${e.shapeSystem}|${e.ellipseSpacing}|${t}`,c=S.get(o);if(!c){let l=e.shapeKind,u=e.shapeSystem===`geodesic`,d=s*J(l,t),f=a.createBuffer(`positions-${o}`,d*8),p=a.createBuffer(`offsets-${o}`,(s+1)*4),b=a.createBuffer(`total-${o}`,4),C=a.createBuffer(`corners-${o}`,d*3*8),w=a.createBuffer(`rows-${o}`,d*4),T=new i(r,{id:`shape-${o}`}),E=L(T),D=E(`positions`,f,`float32x2`,d),O=E(`offsets`,p,`uint32`,s+1),k=E(`total`,b,`uint32`,1);T.add(new ie({id:`shape-${l}`,shape:l,spatialContext:u?U:n.projection.spatialContext,ellipseSpacing:l===`ellipse`?e.ellipseSpacing:void 0,maximumSegments:t,centers:E(`centers`,u?h:m,`float32x2`,s),radii:l===`ellipse`?E(`radii`,_,`float32x2`,s):E(`radii`,g,`float32`,s),bearings:l===`sector`?E(`bearings`,y,`float32x2`,s):void 0,rotations:l===`ellipse`?E(`rotations`,v,`float32`,s):void 0,parameters:x.importToGraph(T),output:{positions:D,offsets:O,vertexCount:k}})),A(T,{id:`shape-fans`,invocationCount:d,bindings:[{name:`positions`,view:D,type:`f32`,access:`read`},{name:`offsets`,view:O,type:`u32`,access:`read`},{name:`total`,view:k,type:`u32`,access:`read`},{name:`centers`,view:E(`centers`,u?h:m,`float32x2`,s),type:`f32`,access:`read`},{name:`corners`,view:E(`corners`,C,`float32x2`,d*3),type:`f32`,access:`read_write`},{name:`rows`,view:E(`rows`,w,`uint32`,d),type:`u32`,access:`read_write`}],declarations:`const FEATURES: u32 = ${s}u;`,body:`
  let stride = max(offsets[offsetsOffset + 1u], 1u);
  let feature = min(index / stride, FEATURES - 1u);
  let center = vec2<f32>(centers[centersOffset + feature * 2u], centers[centersOffset + feature * 2u + 1u]);
  let zero = f32(index) * 0.0;
  var a = vec2<f32>(zero / zero);
  var b = a;
  var c = a;
  if (index < total[totalOffset] && (index + 1u) % stride != 0u) {
    a = center;
    b = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    c = vec2<f32>(positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
  }
  corners[cornersOffset + index * 6u] = a.x;
  corners[cornersOffset + index * 6u + 1u] = a.y;
  corners[cornersOffset + index * 6u + 2u] = b.x;
  corners[cornersOffset + index * 6u + 3u] = b.y;
  corners[cornersOffset + index * 6u + 4u] = c.x;
  corners[cornersOffset + index * 6u + 5u] = c.y;
  rows[rowsOffset + index] = feature;`}),c={key:o,shapeKind:e.shapeKind,shapeSystem:e.shapeSystem,compiled:a.track(await T.compileAsync()),positions:f,corners:C,featureRows:w,slotCount:d,maximumSegments:t},S.set(o,c)}return c},P=e=>{C=e,w=!0};T(t.options),E(t.options),j(t.options),P(await N(t.options)),t.setReadout(`shapeCount`,s);let F=a.track(new O),R=e=>{let n=e.shapeKind;if(!C)return;let r=k(e,n,C.maximumSegments);t.setReadout(`shapeVertices`,`${J(n,r)} per shape (${r} segments${r===e.segmentCount?``:`, clamped to ${C.maximumSegments} by maximumSegments`})`);let i=e.shapeRadius/1e3,a=n===`circle`?Math.PI*i**2:n===`sector`?Math.PI*i**2*e.sectorSweep/360:Math.PI*i**2*e.ellipseRatio;t.setReadout(`shapeArea`,`${W(a,2)} km² per average station`)};R(t.options);let z=()=>t.theme()===`dark`;return{getCompiledGraphs:()=>C?[C.compiled]:[],setOption(e,n){e===`sectorSweep`&&(T(n),w=!0),e===`ellipseRatio`&&(E(n),w=!0),(e===`segmentCount`||e===`shapeRadius`)&&j(n),(e===`shapeKind`||e===`shapeSystem`||e===`ellipseSpacing`||e===`maxSegments`)&&F.request({build:()=>N(n),commit:e=>{P(e),R(t.options),t.requestLayers()}}),R(n)},encode(e){C&&w&&(C.compiled.encode(e,{parameters:void 0}),w=!1)},getLayers(){if(!C)return[];let n={...t.options,shapeKind:C.shapeKind,shapeSystem:C.shapeSystem},r=n.shapeSystem===`geodesic`?se:Z,i=k(n,n.shapeKind,C.maximumSegments),a=J(n.shapeKind,i),o=[];return n.showShapeFill&&o.push(new I({id:`shape-fill`,coordinateSystem:r,coordinateOrigin:e.coordinateOrigin,corners:C.corners,featureRows:C.featureRows,instanceCount:C.slotCount,values:b,colormap:`ylgnbu`,valueRange:[0,20],color:[255,255,255,255],opacity:.42})),o.push(new M({id:`shape-rings`,coordinateSystem:r,coordinateOrigin:e.coordinateOrigin,positions:C.positions,stride:a,ringCount:s,values:b,colorSource:`slot-value`,colormap:`ylgnbu`,valueRange:[0,20],color:[255,255,255,255],widthPixels:1.4}),new D({id:`shape-centers`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:m,instanceCount:s,radiusPixels:2.6,color:z()?[255,255,255,255]:[20,30,50,255]})),o},destroy:()=>a.destroy()}}var Q=56,$=72;async function fe(e){let{ctx:t,city:n}=e,{device:r}=t,a=new l(r,`grid`),o=n.allStops.count,s=a.createParameterBuffer(`grid-parameters`,`float32`,4),u=a.createParameterBuffer(`grid-hilbert-bounds`,`float32`,4),d=a.createParameterBuffer(`stops-hilbert-bounds`,`float32`,4),f=a.createBuffer(`stop-positions`,n.allStops.local),p=null,m=null,h=!0,g=!0,_=null,v=1/0,S=1/0,C=-1/0,w=-1/0;for(let e=0;e<o;e++)v=Math.min(v,n.allStops.local[e*2]),C=Math.max(C,n.allStops.local[e*2]),S=Math.min(S,n.allStops.local[e*2+1]),w=Math.max(w,n.allStops.local[e*2+1]);d.write(Float32Array.of(v-100,S-100,C+100,w+100));let T=(e,t,n,r,i,a,o,s)=>{A(e,{id:`hilbert-curve`,invocationCount:n,bindings:[{name:`rows`,view:a,type:`u32`,access:`read`},{name:`keys`,view:i,type:`u32`,access:`read`},{name:`centers`,view:r,type:`f32`,access:`read`},{name:`starts`,view:t(`curve-starts`,o.starts,`float32`,n*2),type:`f32`,access:`read_write`},{name:`ends`,view:t(`curve-ends`,o.ends,`float32`,n*2),type:`f32`,access:`read_write`},{name:`values`,view:t(`values`,o.values,`float32`,n),type:`f32`,access:`read_write`}],declarations:`const COUNT: u32 = ${n}u;\nconst CELLS: f32 = ${(4**s).toFixed(1)};`,body:`
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (index + 1u < COUNT) {
    let first = rows[rowsOffset + index];
    let second = rows[rowsOffset + index + 1u];
    start = vec2<f32>(centers[centersOffset + first * 2u], centers[centersOffset + first * 2u + 1u]);
    end = vec2<f32>(centers[centersOffset + second * 2u], centers[centersOffset + second * 2u + 1u]);
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;
  values[valuesOffset + index] = f32(keys[keysOffset + rows[rowsOffset + index]]) / CELLS;`})},E=e=>{a.release(e.compiled);for(let t of[e.positions,e.centers,e.values,e.fanCorners,e.fanRows,e.curveStarts,e.curveEnds,e.keys,e.rows,e.curveValues])t&&a.release(t)},j=e=>{a.release(e.compiled);for(let t of[e.values,e.curveStarts,e.curveEnds,e.rows,e.keys,e.curveValues])a.release(t)},N=async e=>{let t=`${e.gridType}|${e.hilbertOrder}`;if(p?.key===t)return null;let o=e.gridType,c=b(o),l=Q*$*(o===`triangle`?2:1),d=c>0?a.createBuffer(`grid-positions-${t}`,l*c*8):null,f=a.createBuffer(`grid-centers-${t}`,l*8),m=a.createBuffer(`grid-keys-${t}`,l*4),h=a.createBuffer(`grid-rows-${t}`,l*4),g=a.createBuffer(`grid-curve-values-${t}`,l*4),_=a.createBuffer(`grid-curve-starts-${t}`,l*8),v=a.createBuffer(`grid-curve-ends-${t}`,l*8),y=a.createBuffer(`grid-cell-values-${t}`,l*4),S=c>0?l*c:0,C=S?a.createBuffer(`grid-fan-${t}`,S*3*8):null,w=new Uint32Array(Math.max(S,1));for(let e=0;e<S;e++)w[e]=Math.floor(e/Math.max(c,1));let E=S?a.createBuffer(`grid-fan-rows-${t}`,w):null,D=new i(r,{id:`grid-${t}`}),O=L(D),k=O(`centers`,f,`float32x2`,l);D.add(new x({spatialContext:n.projection.spatialContext,id:`grid`,gridType:o,columns:Q,rows:$,parameters:s.importToGraph(D),output:{...d?{positions:O(`positions`,d,`float32x2`,l*c)}:{},centers:k}}));let j=O(`keys`,m,`uint32`,l),M=O(`rows`,h,`uint32`,l);return D.add(new X({spatialContext:n.projection.spatialContext,id:`hilbert`,order:e.hilbertOrder,positions:k,bounds:u.importToGraph(D),output:{keys:j,sortedRows:M}})),T(D,O,l,k,j,M,{starts:_,ends:v,values:g},e.hilbertOrder),A(D,{id:`cell-values`,invocationCount:l,bindings:[{name:`keys`,view:j,type:`u32`,access:`read`},{name:`cellValues`,view:O(`cell-values`,y,`float32`,l),type:`f32`,access:`read_write`}],declarations:`const CELLS: f32 = ${(4**e.hilbertOrder).toFixed(1)};`,body:`cellValues[cellValuesOffset + index] = f32(keys[keysOffset + index]) / CELLS;`}),C&&d&&A(D,{id:`cell-fans`,invocationCount:S,bindings:[{name:`positions`,view:O(`positions`,d,`float32x2`,l*c),type:`f32`,access:`read`},{name:`centers`,view:k,type:`f32`,access:`read`},{name:`corners`,view:O(`fan`,C,`float32x2`,S*3),type:`f32`,access:`read_write`}],declarations:`const K: u32 = ${c}u;`,body:`
  let cell = index / K;
  let step = index % K;
  let next = (step + 1u) % K;
  let center = vec2<f32>(centers[centersOffset + cell * 2u], centers[centersOffset + cell * 2u + 1u]);
  let a = vec2<f32>(positions[positionsOffset + (cell * K + step) * 2u], positions[positionsOffset + (cell * K + step) * 2u + 1u]);
  let b = vec2<f32>(positions[positionsOffset + (cell * K + next) * 2u], positions[positionsOffset + (cell * K + next) * 2u + 1u]);
  corners[cornersOffset + index * 6u] = center.x;
  corners[cornersOffset + index * 6u + 1u] = center.y;
  corners[cornersOffset + index * 6u + 2u] = a.x;
  corners[cornersOffset + index * 6u + 3u] = a.y;
  corners[cornersOffset + index * 6u + 4u] = b.x;
  corners[cornersOffset + index * 6u + 5u] = b.y;`}),{key:t,compiled:a.track(await D.compileAsync()),curveValues:g,cellCount:l,verticesPerCell:c,positions:d,centers:f,values:y,fanCorners:C,fanRows:E,fanCount:S,curveStarts:_,curveEnds:v,keys:m,rows:h}},P=(e,n)=>{if(!e)return;let r=p;p=e,r&&E(r);let i=n.gridType,a=e.cellCount;h=!0,t.setReadout(`gridCells`,`${c(a)} ${i===`point`?`points`:i+` cells`}`),t.setReadout(`hilbertCells`,`${c(4**n.hilbertOrder)} curve cells (order ${n.hilbertOrder})`)},R=async e=>{let s=`${e.hilbertOrder}`;if(m?.key===s)return null;let c=a.createBuffer(`stops-keys-${s}`,o*4),l=a.createBuffer(`stops-rows-${s}`,o*4),u=a.createBuffer(`stops-values-${s}`,o*4),p=a.createBuffer(`stops-point-values-${s}`,o*4),h=a.createBuffer(`stops-curve-starts-${s}`,o*8),g=a.createBuffer(`stops-curve-ends-${s}`,o*8),_=new i(r,{id:`stops-${s}`}),v=L(_),y=v(`points`,f,`float32x2`,o),b=v(`keys`,c,`uint32`,o),x=v(`rows`,l,`uint32`,o);return _.add(new X({spatialContext:n.projection.spatialContext,id:`stops-hilbert`,order:e.hilbertOrder,positions:y,bounds:d.importToGraph(_),output:{keys:b,sortedRows:x}})),T(_,v,o,y,b,x,{starts:h,ends:g,values:u},e.hilbertOrder),A(_,{id:`stop-values`,invocationCount:o,bindings:[{name:`keys`,view:b,type:`u32`,access:`read`},{name:`out`,view:v(`point-values`,p,`float32`,o),type:`f32`,access:`read_write`}],declarations:`const CELLS: f32 = ${(4**e.hilbertOrder).toFixed(1)};`,body:`out[outOffset + index] = f32(keys[keysOffset + index]) / CELLS;`}),{build:{key:s,compiled:a.track(await _.compileAsync()),values:p,curveStarts:h,curveEnds:g,rows:l,keys:c,curveValues:u},reader:new k(a,`stops-rows-${s}`,[{buffer:l,size:o*4}],e=>{let r=new Uint32Array(e),i=0,a=0;for(let e=0;e+1<o;e++){let t=r[e],o=r[e+1];i+=Math.hypot(n.allStops.local[t*2]-n.allStops.local[o*2],n.allStops.local[t*2+1]-n.allStops.local[o*2+1]),a+=Math.hypot(n.allStops.local[e*2]-n.allStops.local[e*2+2],n.allStops.local[e*2+1]-n.allStops.local[e*2+3])}t.setReadout(`hop`,`${z(i/(o-1))} along the curve vs ${z(a/(o-1))} in file order`)})}},B=e=>{if(!e)return;let t=m;_?.stop(),m=e.build,_=e.reader,t&&j(t),g=!0},V=e=>{let n=e.gridType,r=e.cellSize,i=n===`triangle`?r*Math.sqrt(3)/2:r,a=n===`hex`?r/Math.sqrt(3)*1.5:i,o=Q*r+(n===`hex`||n===`triangle`?r/2:0),c=$*a,l=-o/2,d=-c/2;s.write(y({minX:l,minY:d,cellWidth:r,cellHeight:i})),u.write(Float32Array.of(l,d,l+o,d+c)),t.setReadout(`gridExtent`,`${(o/1e3).toFixed(1)} × ${(c/1e3).toFixed(1)} km`),h=!0};P(await N(t.options),t.options),B(await R(t.options)),V(t.options);let H=a.track(new O);return{getCompiledGraphs(){let e=[];return p&&e.push(p.compiled),m&&e.push(m.compiled),e},setOption(e,n){(e===`gridType`||e===`hilbertOrder`)&&H.request({build:async()=>({nextGrid:await N(n),nextStops:await R(n)}),commit:({nextGrid:e,nextStops:r})=>{P(e,n),B(r),V(n),t.requestLayers()},discard:({nextGrid:e,nextStops:t})=>{e&&E(e),t?.reader.stop(),t&&j(t.build)}}),e===`cellSize`&&V(n)},encode(e){p&&h&&(p.compiled.encode(e,{parameters:void 0}),h=!1),m&&g?(m.compiled.encode(e,{parameters:void 0}),_?.request(e),g=!1):_?.flush(e)},getLayers(){let n=t.options,r=t.theme()===`dark`,i=[],a=r?[255,255,255,120]:[20,30,50,120];return n.hilbertTarget===`grid`&&p&&(p.fanCorners&&p.fanRows?(i.push(new I({id:`grid-fill`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,corners:p.fanCorners,featureRows:p.fanRows,instanceCount:p.fanCount,values:p.values,colormap:`ylgnbu`,valueRange:[0,1],color:[255,255,255,255],opacity:.7})),i.push(new M({id:`grid-edges`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:p.positions,stride:p.verticesPerCell,ringCount:p.cellCount,color:a,widthPixels:.8}))):i.push(new D({id:`grid-points`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:p.centers,instanceCount:p.cellCount,radiusPixels:4,values:p.values,colormap:`ylgnbu`,valueRange:[0,1]})),n.showCurve&&i.push(new F({id:`grid-curve`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,starts:p.curveStarts,ends:p.curveEnds,instanceCount:p.cellCount-1,values:p.curveValues,colormap:`ylorbr`,valueRange:[0,1],color:[255,255,255,255],widthPixels:1.6}))),n.hilbertTarget===`stops`&&m&&(n.showCurve&&i.push(new F({id:`stops-curve`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,starts:m.curveStarts,ends:m.curveEnds,instanceCount:o-1,values:m.values,colormap:`ylgnbu`,valueRange:[0,1],color:[255,255,255,255],widthPixels:.8,opacity:.55})),i.push(new D({id:`stops-points`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:f,instanceCount:o,radiusPixels:2,values:m.values,colormap:`ylgnbu`,valueRange:[0,1]}))),i},destroy(){_?.stop(),a.destroy()}}}async function pe(e){let{ctx:t,city:n}=e,{device:r}=t,a=new l(r,`clip`),o=e.loadStreets(),s=a.createParameterBuffer(`clip-parameters`,`float32`,4),u=a.createBuffer(`rectangle-starts`,32),d=a.createBuffer(`rectangle-ends`,32),f=n.projection.project(G[0],G[1]),p=[f[0]-2500,f[1]+1500],m=!1,h=[0,0],g=!0,_=H(a,`clip-streets`,o.local,o.offsets),v=H(a,`clip-areas`,n.areas.local,n.areas.ringOffsets),y=a.createBuffer(`street-positions`,o.local),b=a.createBuffer(`street-offsets`,o.offsets),x=a.createBuffer(`area-positions`,n.areas.local),S=a.createBuffer(`area-offsets`,n.areas.ringOffsets),C=a.createBuffer(`area-polygon-offsets`,n.areas.polygonOffsets),w=a.createBuffer(`area-feature-offsets`,n.areas.featureOffsets),D=a.createBuffer(`street-classes`,Float32Array.from(o.classes??[])),j=new Map,M=async e=>{let l=j.get(e);if(l)return l;let u=e===`roads`,d=u?o.vertexCount:n.areas.vertexCount,f=u?o.pathCount:n.areas.ringCount,p=d*2+64,m=u?f*2:f,h=B(a,`clip-${e}`,p,m),g=u?a.createBuffer(`clip-path-values`,m*4):null,_=new i(r,{id:`clip-${e}`}),v=L(_),T=V(_,h,{sourceLineIndices:u,lineCount:u}),O=v(`positions`,u?y:x,`float32x2`,d),M=u?{geometryType:`lines`,positions:O,lineOffsets:v(`offsets`,b,`uint32`,f+1)}:{geometryType:`polygons`,positions:O,featureOffsets:v(`feature-offsets`,w,`uint32`,n.areas.featureCount+1),polygonOffsets:v(`polygon-offsets`,C,`uint32`,n.areas.partCount+1),ringOffsets:v(`offsets`,S,`uint32`,f+1)};_.add(new E({spatialContext:n.projection.spatialContext,id:`clip`,...M,parameters:s.importToGraph(_),output:T})),u&&g&&A(_,{id:`source-classes`,invocationCount:m,bindings:[{name:`sources`,view:T.sourceLineIndices,type:`u32`,access:`read`},{name:`classes`,view:v(`classes`,D,`float32`,f),type:`f32`,access:`read`},{name:`values`,view:v(`path-values`,g,`float32`,m),type:`f32`,access:`read_write`}],declarations:`const PATHS: u32 = ${f}u;`,body:`
  let source = sources[sourcesOffset + index];
  let zero = f32(index) * 0.0;
  var value = zero / zero;
  if (source < PATHS) {
    value = classes[classesOffset + source];
  }
  values[valuesOffset + index] = value;`});let N=new k(a,`clip-${e}`,[{buffer:h.count,size:4},{buffer:h.requiredCount,size:4},{buffer:h.overflow,size:4},{buffer:h.pathCount,size:4}],e=>{let n=new Uint32Array(e);t.setReadout(`clipVertices`,`${c(n[0])} of ${c(d)} input vertices`),t.setReadout(`clipTotal`,c(n[1])),t.setReadout(`clipOverflow`,n[2]?`yes: output truncated`:`no`),t.setReadout(`clipPaths`,u?`${c(n[3])} pieces from ${c(f)} streets`:`${c(f)} rings (one output ring per input ring)`)});return l={compiled:a.track(await _.compileAsync()),output:h,pathValues:g,reader:N},j.set(e,l),l},P=e=>{let t=e.clipSize*1e3,[n,r]=p;s.write(T({minX:n-t,minY:r-t,maxX:n+t,maxY:r+t})),u.write(Float32Array.of(n-t,r-t,n+t,r-t,n+t,r+t,n-t,r+t)),d.write(Float32Array.of(n+t,r-t,n+t,r+t,n-t,r+t,n-t,r-t)),g=!0};await M(t.options.clipGeometry);let I=t.options.clipGeometry,z=a.track(new O);P(t.options);let U=e=>e.coordinate?n.projection.project(e.coordinate[0],e.coordinate[1]):null,W=e=>{let n=t.options.clipSize*1e3;return Math.abs(e[0]-p[0])<=n&&Math.abs(e[1]-p[1])<=n};return{getCompiledGraphs:()=>[...j.values()].map(e=>e.compiled),setOption(e,n){e===`clipGeometry`&&z.request({build:()=>M(n.clipGeometry),commit:()=>{I=n.clipGeometry,g=!0,t.requestLayers()}}),e===`clipSize`&&P(n)},onClick(e){let n=U(e);return n?(p=n,P(t.options),!0):!1},onDragStart(e){let n=U(e);return!n||!W(n)?!1:(m=!0,h=[p[0]-n[0],p[1]-n[1]],t.setMapDragEnabled(!1),!0)},onDrag(e){let n=U(e);!m||!n||(p=[n[0]+h[0],n[1]+h[1]],P(t.options))},onDragEnd(){m=!1,t.setMapDragEnabled(!0)},encode(e){let t=j.get(I);t&&(g?(t.compiled.encode(e,{parameters:void 0}),R(e,t.output.count,t.output.drawCommands),t.reader.request(e),g=!1):t.reader.flush(e))},getLayers(){let n=t.options,r=j.get(I);if(!r)return[];let i=t.theme()===`dark`,a=I===`roads`,o=[];if(n.showOriginal){let t=a?_:v;o.push(new N({id:`clip-original`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:t.positions,pathOffsets:t.offsets,pathOffsetCount:t.offsetCount,vertexCount:t.vertexCount,drawCommands:t.drawCommands,color:i?[150,160,180,70]:[90,100,120,70],widthPixels:.8}))}return o.push(new N({id:`clip-output`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,positions:r.output.positions,pathOffsets:r.output.offsets,pathOffsetCount:r.output.offsetCount,vertexCount:r.output.count,drawCommands:r.output.drawCommands,closed:!a,values:a?r.pathValues:null,colorSource:a?`path-value`:`uniform`,valueMapping:`category`,color:a?[255,255,255,255]:[78,168,222,255],widthPixels:a?1.8:2.4}),new F({id:`clip-rectangle`,coordinateSystem:Z,coordinateOrigin:e.coordinateOrigin,starts:u,ends:d,instanceCount:4,color:i?[255,214,120,255]:[200,90,10,255],widthPixels:2.5})),o},destroy(){for(let e of j.values())e.reader.stop();a.destroy()}}}export{le as createBuffersAndShapes};