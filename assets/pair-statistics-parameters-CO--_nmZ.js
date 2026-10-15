import{J as e}from"./src-CNCKFW2b.js";import{_ as t}from"./analysis-frame-Blu0tYdw.js";import{d as n,j as r,l as i,w as a,y as o}from"./resources-CwvhEOBe.js";import{t as s}from"./gpu-scan-BuGhTLDQ.js";import{t as c}from"./gpu-sort-Bl3Dztdb.js";import{t as l}from"./gpu-group-aggregation-DrNKSCi8.js";import{n as u,s as d}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as f}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{t as p}from"./exact-geometry-wgsl-DOqxBQzc.js";import{a as m,n as h}from"./sorted-segment-sums-DwGkmbx2.js";import{i as g,n as _}from"./polygon-edge-slabs-Cmf_6bAe.js";Object.freeze({minX:0,minY:1,maxX:2,maxY:3,area:4,perimeter:5,edgeCount:6,flags:7});function v(e,n,r){if(n===void 0||n===`rectangle`)return`rectangle`;if(typeof n==`object`&&n&&n.kind===`polygon`&&typeof n.id==`string`&&n.spatialContext!==null&&typeof n.spatialContext==`object`&&typeof n.getViews==`function`)return t(`${e} studyWindow`,n.spatialContext,r),n;throw Error(`${e} studyWindow must be 'rectangle' or a GPUStudyWindowPreparation`)}var y=8;function b(e){let t=0+y,n=t+4*e.vertexCount,r=n+6,i=r+e.slabOffsetCount,a=i+2*e.slabEntryCount,o=a+1;return{summary:0,edges:t,slabParams:n,slabOffsets:r,slabEntries:i,sampleCellCount:a,sampleCells:o,length:o+e.sampleCellCapacity}}var x=2147483648;function S(e,t){return`
const STUDY_WINDOW_SAMPLE_COLUMNS: u32 = ${e[0]}u;
const STUDY_WINDOW_SAMPLE_ROWS: u32 = ${e[1]}u;

fn getStudyWindowSampleCell(index: u32) -> vec4f {
  let cell = index & 0x7fffffffu;
  let minimum = vec2f(${t}(0u), ${t}(1u));
  let maximum = vec2f(${t}(2u), ${t}(3u));
  let size = (maximum - minimum) / vec2f(f32(STUDY_WINDOW_SAMPLE_COLUMNS), f32(STUDY_WINDOW_SAMPLE_ROWS));
  let column = cell % STUDY_WINDOW_SAMPLE_COLUMNS;
  let row = cell / STUDY_WINDOW_SAMPLE_COLUMNS;
  return vec4f(minimum + size * vec2f(f32(column), f32(row)), size);
}
`}function C(e){return{name:`studyWindow`,view:e.packed,type:`u32`,access:`read`}}function w(e,t){let n=e.edges.length/4;return{layout:b({vertexCount:n,slabOffsetCount:e.slabs.slabOffsets.length,slabEntryCount:e.slabs.slabEntries.length,sampleCellCapacity:e.sampleCells.length}),vertexCount:n,entryCapacity:e.slabs.entryCapacity,sampleGrid:e.sampleGrid,kernel:t}}function T(e){let{layout:t}=e,n={params:{array:`studyWindow`,offset:`(studyWindowOffset + ${t.slabParams}u)`},offsets:{array:`studyWindow`,offset:`(studyWindowOffset + ${t.slabOffsets}u)`},entries:{array:`studyWindow`,offset:`(studyWindowOffset + ${t.slabEntries}u)`}},r=(e,t,r,i=!1)=>_({owner:`0u`,lowY:e,highY:t,storage:n,...i?{dedupe:{edgeLowY:`min(studyWindowEdge(edgeStart).y, studyWindowEdge(edgeStart).w)`}}:{},body:`let edge = studyWindowEdge(edgeStart);
    if (!studyWindowFinitePoint(edge.xy) || !studyWindowFinitePoint(edge.zw)) { continue; }
${r}`});return`
${g(e,n)}
const STUDY_WINDOW_TWO_PI: f32 = 6.283185307179586;
fn studyWindowSummaryValue(slot: u32) -> f32 {
  return bitcast<f32>(studyWindow[studyWindowOffset + ${t.summary}u + slot]);
}
fn studyWindowEdge(vertex: u32) -> vec4f {
  let base = studyWindowOffset + ${t.edges}u + 4u * vertex;
  return vec4f(bitcast<f32>(studyWindow[base]), bitcast<f32>(studyWindow[base + 1u]),
    bitcast<f32>(studyWindow[base + 2u]), bitcast<f32>(studyWindow[base + 3u]));
}
fn isStudyWindowUsable() -> bool {
  let area = studyWindowSummaryValue(4u);
  let flags = studyWindowSummaryValue(7u);
  return isFiniteBits(area) && area > 0.0 && isFiniteBits(flags) && (u32(flags) & 2u) == 0u;
}
fn studyWindowFinitePoint(point: vec2f) -> bool {
  return isFiniteBits(point.x) && isFiniteBits(point.y);
}
fn studyWindowCross(first: vec2f, second: vec2f) -> f32 {
  return first.x * second.y - first.y * second.x;
}
fn studyWindowSegmentDistance(point: vec2f, edge: vec4f) -> f32 {
  let direction = edge.zw - edge.xy;
  let squaredLength = dot(direction, direction);
  if (!(squaredLength > 0.0)) { return length(point - edge.xy); }
  let fraction = clamp(dot(point - edge.xy, direction) / squaredLength, 0.0, 1.0);
  return length(point - edge.xy - fraction * direction);
}
fn isInsideStudyWindow(point: vec2f) -> bool {
  if (!studyWindowFinitePoint(point) || !isStudyWindowUsable()) { return false; }
  if (point.x < studyWindowSummaryValue(0u) || point.y < studyWindowSummaryValue(1u) ||
      point.x > studyWindowSummaryValue(2u) || point.y > studyWindowSummaryValue(3u)) { return false; }
  let probe = ExactProbe(point, point, point, EXACT_PROBE_POINT);
  var parity = 0i;
  ${r(`point.y`,`point.y`,`parity ^= abs(exactProbeCrossing(probe, edge.xy, edge.zw).x);`)}
  return (parity & 1i) != 0i;
}
fn getStudyWindowBoundaryDistance(point: vec2f, limit: f32) -> f32 {
  // Runtime bits: WGSL rejects non-finite constant expressions.
  var infinityBits = 0x7f800000u;
  var distance = bitcast<f32>(infinityBits);
  if (!studyWindowFinitePoint(point) || !(limit >= 0.0)) { return distance; }
  ${r(`point.y - limit`,`point.y + limit`,`
    if (max(edge.x, edge.z) < point.x - limit || min(edge.x, edge.z) > point.x + limit) { continue; }
    let candidate = studyWindowSegmentDistance(point, edge);
    if (candidate <= limit) { distance = min(distance, candidate); }`)}
  return distance;
}
// Metric fan baseline. The ray test is only used for a floating-point integral, never membership.
// At a boundary point the full signed angular sum supplies the interior angular limit; zero-area
// fans contribute zero. The slower full walk is restricted to this exceptional case.
fn studyWindowFanBase(centre: vec2f) -> f32 {
  var parity = false;
  var boundary = false;
  ${r(`centre.y`,`centre.y`,`
    let start = edge.xy - centre;
    let end = edge.zw - centre;
    let cross = studyWindowCross(start, end);
    if (cross == 0.0 && dot(start, end) <= 0.0 && any(edge.xy != edge.zw)) { boundary = true; }
    if ((edge.y > centre.y) != (edge.w > centre.y)) {
      if ((cross > 0.0) == (edge.w > edge.y)) { parity = !parity; }
    }`)}
  if (!boundary) { return select(0.0, 1.0, parity); }
  var angle = 0.0;
  for (var vertex = 0u; vertex < EDGE_SLAB_VERTEX_COUNT; vertex++) {
    let edge = studyWindowEdge(vertex);
    if (!studyWindowFinitePoint(edge.xy) || !studyWindowFinitePoint(edge.zw)) { continue; }
    let start = edge.xy - centre;
    let end = edge.zw - centre;
    let cross = studyWindowCross(start, end);
    if (isFiniteBits(cross) && cross != 0.0) { angle += atan2(cross, dot(start, end)); }
  }
  return clamp(angle / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}
// (line distance, signed along-edge start, signed along-edge end, fan orientation).
fn studyWindowFan(edge: vec4f, centre: vec2f) -> vec4f {
  let direction = edge.zw - edge.xy;
  let edgeLength = length(direction);
  if (!(edgeLength > 0.0)) { return vec4f(0.0); }
  let start = edge.xy - centre;
  let cross = studyWindowCross(start, direction);
  let along = dot(start, direction / edgeLength);
  return vec4f(abs(cross) / edgeLength, along, along + edgeLength, sign(cross));
}
fn getStudyWindowCircleFraction(centre: vec2f, radius: f32) -> f32 {
  if (!studyWindowFinitePoint(centre) || !(radius > 0.0) || !isFiniteBits(radius)) { return 0.0; }
  var correction = 0.0;
  ${r(`centre.y - radius`,`centre.y + radius`,`
    let fan = studyWindowFan(edge, centre);
    if (!(fan.x > 0.0 && fan.x < radius)) { continue; }
    let ratio = fan.x / radius;
    let angle = atan2(sqrt(max(0.0, (1.0 - ratio) * (1.0 + ratio))), ratio);
    let low = max(atan2(fan.y, fan.x), -angle);
    let high = min(atan2(fan.z, fan.x), angle);
    correction += fan.w * max(0.0, high - low);`,!0)}
  return clamp(studyWindowFanBase(centre) - correction / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}
${e.kernel?E(e.kernel):``}
${e.kernel?`
fn getStudyWindowKernelMass(centre: vec2f, bandwidth: f32) -> f32 {
  if (!studyWindowFinitePoint(centre) || !(bandwidth > 0.0) || !isFiniteBits(bandwidth)) { return 0.0; }
  let radius = ${e.kernel===`gaussian`?`5.0`:`1.0`} * bandwidth;
  var correction = 0.0;
  ${r(`centre.y - radius`,`centre.y + radius`,`
    let fan = studyWindowFan(edge, centre);
    if (!(fan.x > 0.0 && fan.x < radius)) { continue; }
    correction += fan.w * studyWindowKernelIntegral(fan.x / bandwidth, fan.y / bandwidth, fan.z / bandwidth);`,!0)}
  return clamp(studyWindowFanBase(centre) - correction / STUDY_WINDOW_TWO_PI, 0.0, 1.0);
}`:``}
${e.sampleGrid?S(e.sampleGrid,`studyWindowSummaryValue`):``}
`}function E(e){return e===`gaussian`?`
// 16-point Gauss-Legendre on half-bandwidth panels, split additionally at x=p/h.
// Integrate the complement in x=(p/h)*tan(t): subtraction removes the angular boundary layer
// at tiny p/h. Only rho/h <= 5 is retained (Gaussian tail <= exp(-12.5)).
fn studyWindowGaussianCorrection(ratio: f32, low: f32, high: f32) -> f32 {
  var nodes = array<f32, 8>(0.09501250983763744, 0.2816035507792589, 0.4580167776572274,
    0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499);
  var weights = array<f32, 8>(0.1894506104550685, 0.1826034150449236, 0.16915651939500254,
    0.14959598881657673, 0.12462897125553387, 0.09515851168249278, 0.06225352393864789, 0.027152459411754096);
  var total = 0.0;
  for (var panel = 0u; panel < 10u; panel++) {
    let begin = max(low, f32(panel) * 0.5);
    let end = min(high, f32(panel + 1u) * 0.5);
    if (!(end > begin)) { continue; }
    let split = clamp(ratio, begin, end);
    for (var part = 0u; part < 2u; part++) {
      let left = select(begin, split, part == 1u);
      let right = select(split, end, part == 1u);
      if (!(right > left)) { continue; }
      let middle = (left + right) * 0.5;
      let half = (right - left) * 0.5;
      for (var node = 0u; node < 8u; node++) {
        for (var side = 0u; side < 2u; side++) {
          let value = middle + select(-1.0, 1.0, side == 1u) * half * nodes[node];
          let squaredValue = value * value;
          total += half * weights[node] * (1.0 - exp(-squaredValue * 0.5)) * ratio / (ratio * ratio + squaredValue);
        }
      }
    }
  }
  return total;
}
fn studyWindowKernelIntegral(ratio: f32, start: f32, end: f32) -> f32 {
  let extent = sqrt(max(0.0, 25.0 - ratio * ratio));
  let low = max(start, -extent);
  let high = min(end, extent);
  if (!(high > low)) { return 0.0; }
  let correction = studyWindowGaussianCorrection(ratio, max(0.0, low), max(0.0, high)) +
    studyWindowGaussianCorrection(ratio, max(0.0, -high), max(0.0, -low));
  return exp(-ratio * ratio * 0.5) * (atan2(high, ratio) - atan2(low, ratio) - correction);
}
`:`
fn studyWindowKernelPolynomial(value: f32, squaredRatio: f32) -> f32 {
  ${e===`epanechnikov`?`return (-2.0 + squaredRatio) * value + value * value * value / 3.0;`:`return (-3.0 + 3.0 * squaredRatio - squaredRatio * squaredRatio) * value +
    (1.0 - 2.0 * squaredRatio / 3.0) * value * value * value - value * value * value * value * value / 5.0;`}
}
fn studyWindowKernelIntegral(ratio: f32, start: f32, end: f32) -> f32 {
  let squaredRatio = ratio * ratio;
  let extent = sqrt(max(0.0, 1.0 - squaredRatio));
  let low = max(start, -extent);
  let high = min(end, extent);
  if (!(high > low)) { return 0.0; }
  // x = tan(t), evaluated as ratio*x = along/h to avoid huge powers near pi/2.
  return atan2(high, ratio) - atan2(low, ratio) + ratio *
    (studyWindowKernelPolynomial(high, squaredRatio) - studyWindowKernelPolynomial(low, squaredRatio));
}
`}function D(e){return Math.min(i(e,12),2**24-1)}function O(e,t,r){n(e,`positions length`,r,1,D(t))}var k=4096;function A(e,t){o(t.positions,[`float32x2`],`${e} positions`),o(t.parameters,[`float32`],`${e} parameters`);let i=t.positions.length;if(n(e,`positions length`,i,1,2**24-1),t.parameters.length<t.parameterLength)throw Error(`${e} parameters must hold ${t.parameterLength} float32 values`);let[a,s]=t.gridSize;if(!Number.isInteger(a)||!Number.isInteger(s)||a<1||s<1||a*s>=2**24)throw Error(`${e} gridSize must be two positive integers with columns * rows < 2^24`);if(t.values&&(o(t.values,[`float32`],`${e} values`),t.values.length!==i))throw Error(`${e} values length must equal positions length`);if(t.mask&&(r(t.mask,`${e} mask`),t.mask.length!==i))throw Error(`${e} mask length must equal positions length`);return i}var j=`
fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
fn getQuietNaN(seed: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

fn getOrderedFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn getOrderedFloatValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`;function M(e){return`
const COLUMNS: u32 = ${e[0]}u;
const ROWS: u32 = ${e[1]}u;
const CELL_COUNT: u32 = ${e[0]*e[1]}u;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  maximumDistance: f32,
  radiusSquared: f32
}

fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.radiusSquared = 0.0;
  let minimumX = readParameter(0u);
  let minimumY = readParameter(1u);
  let maximumX = readParameter(2u);
  let maximumY = readParameter(3u);
  let radius = readParameter(4u);
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let radiusSquared = radius * radius;
  let cellRadius = radius * CELL_MARGIN;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(radius) && isFiniteFloat(width) &&
    isFiniteFloat(height) && isFiniteFloat(radiusSquared) && isFiniteFloat(cellRadius) &&
    radius > 0.0 && width >= 0.0 && height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.maximumDistance = radius;
  lattice.radiusSquared = radiusSquared;
  return lattice;
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}

${j}
`}function N(e,t){let n=a(e,`${t.id}-window-parameters`,`float32`,t.parameterLength);return{parameters:n,nodes:[d(e,{id:`${t.id}-window-parameters`,operation:t.operation,variant:`window-parameters`,bindings:[{name:`sourceParameters`,view:t.parameters,type:`u32`,access:`read`},C(t.window),{name:`parameters`,view:n,type:`u32`,access:`read_write`}],invocationCount:t.parameterLength,declarations:`${f}
${p}
${T(w(t.window))}`,body:`var value = sourceParameters[sourceParametersOffset + index];
  if (index < 4u) {
    value = select(0x7fc00000u, bitcast<u32>(studyWindowSummaryValue(index)), isStudyWindowUsable());
  }
  parameters[parametersOffset + index] = value;`})]}}function P(e,t){let{id:n,operation:r,positions:i,values:o,parameters:u,gridSize:h,mask:g,studyWindow:_}=t,v=i.length,y=h[0]*h[1],b=M(h),x=[],S=a(e,`${n}-cell-keys`,`uint32`,v),E=a(e,`${n}-row-ids`,`uint32`,v),D=a(e,`${n}-cell-counts`,`uint32`,y),O=a(e,`${n}-cell-offsets`,`uint32`,y+1),A=a(e,`${n}-sorted-keys`,`uint32`,v),N=a(e,`${n}-sorted-rows`,`uint32`,v),P=o?a(e,`${n}-value-terms`,`float32`,v):void 0;if(x.push(d(e,{id:`${n}-cell-keys`,operation:r,variant:`cell-keys`,bindings:[..._?[C(_)]:[],{name:`positions`,view:i,type:`f32`,access:`read`},{name:`parameters`,view:u,type:`f32`,access:`read`},...o?[{name:`values`,view:o,type:`f32`,access:`read`}]:[],...g?[{name:`mask`,view:g,type:`u32`,access:`read`}]:[],{name:`cellKeys`,view:S,type:`u32`,access:`read_write`},{name:`rowIds`,view:E,type:`u32`,access:`read_write`},...P?[{name:`valueTerms`,view:P,type:`f32`,access:`read_write`}]:[]],invocationCount:v,declarations:_?`${b}
${f}
${p}
${T(w(_))}`:b,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${g?`mask[maskOffset + index] != 0u`:`true`};
  ${o?`let value = values[valuesOffset + index];`:``}
  let valid = included && ${o?`isFiniteFloat(value) &&`:``} isPointValid(lattice, x, y)${_?` && isInsideStudyWindow(vec2f(x, y))`:``};
  var key = CELL_COUNT;
  if (valid) {
    key = getCellRow(lattice, y) * lattice.columns + getCellColumn(lattice, x);
  }
  cellKeys[cellKeysOffset + index] = key;
  rowIds[rowIdsOffset + index] = index;
  ${o?`valueTerms[valueTermsOffset + index] = select(0.0, value, valid);`:``}`}),...new l({id:`${n}-cell-counts`,keys:S,output:D}).getCommandNodes(e),...new s({id:`${n}-cell-scan`,input:D,output:O,mode:`exclusive`}).getCommandNodes(e),d(e,{id:`${n}-cell-total`,operation:r,variant:`cell-total`,bindings:[{name:`counts`,view:D,type:`u32`,access:`read`},{name:`cellOffsets`,view:O,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LAST_CELL: u32 = ${y-1}u;`,body:`cellOffsets[cellOffsetsOffset + LAST_CELL + 1u] =
    cellOffsets[cellOffsetsOffset + LAST_CELL] + counts[countsOffset + LAST_CELL];`}),...new c({id:`${n}-cell-sort`,keys:S,values:E,outputKeys:A,outputValues:N,keyBits:m(y)}).getCommandNodes(e)),!o||!P)return{nodes:x,sortedRows:N,cellOffsets:O,cellKeys:S};let F=Math.ceil(v/k),L=a(e,`${n}-block-offsets`,`uint32`,F+1),R=a(e,`${n}-total-offsets`,`uint32`,2),z=a(e,`${n}-value-partials`,`float32`,F),B=a(e,`${n}-value-total`,`float32`,1),V=a(e,`${n}-square-terms`,`float32`,v),H=a(e,`${n}-square-partials`,`float32`,F),U=a(e,`${n}-square-total`,`float32`,1),W=a(e,`${n}-extremes`,`uint32`,2),G=a(e,`${n}-statistics`,`float32`,v+6),K=`const MOMENTS: u32 = ${v}u;
const CELL_COUNT: u32 = ${y}u;`;return x.push(d(e,{id:`${n}-block-offsets`,operation:r,variant:`block-offsets`,bindings:[{name:`blockOffsets`,view:L,type:`u32`,access:`read_write`},{name:`totalOffsets`,view:R,type:`u32`,access:`read_write`},{name:`extremes`,view:W,type:`u32`,access:`read_write`}],invocationCount:F+1,declarations:`const ROW_COUNT: u32 = ${v}u;
const BLOCK_ROWS: u32 = ${k}u;
const BLOCK_COUNT: u32 = ${F}u;`,body:`blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
    // Ordered keys: 0xffffffff is above every finite key, 0 below every finite key.
    extremes[extremesOffset] = 0xffffffffu;
    extremes[extremesOffset + 1u] = 0u;
  }`}),...I(e,{id:`${n}-value-sum`,operation:r,input:P,partials:z,output:B,blockOffsets:L,totalOffsets:R}),d(e,{id:`${n}-moments-mean`,operation:r,variant:`moments-mean`,bindings:[{name:`cellOffsets`,view:O,type:`u32`,access:`read`},{name:`valueTotal`,view:B,type:`f32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${K}
${j}`,body:`let count = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  statistics[statisticsOffset + MOMENTS] = count;
  statistics[statisticsOffset + MOMENTS + 1u] =
    select(getQuietNaN(index), valueTotal[valueTotalOffset] / count, count >= 1.0);`}),d(e,{id:`${n}-center`,operation:r,variant:`center`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},{name:`cellKeys`,view:S,type:`u32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`},{name:`squareTerms`,view:V,type:`f32`,access:`read_write`},{name:`extremes`,view:W,type:`atomic<u32>`,access:`read_write`}],invocationCount:v,declarations:`${K}
${j}`,body:`let valid = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  let value = values[valuesOffset + index];
  let centered = value - statistics[statisticsOffset + MOMENTS + 1u];
  // Excluded rows store quiet NaN, which every later kernel reads as "not included".
  statistics[statisticsOffset + index] = select(getQuietNaN(index), centered, valid);
  squareTerms[squareTermsOffset + index] = select(0.0, centered * centered, valid);
  if (valid) {
    let key = getOrderedFloatKey(value);
    atomicMin(&extremes[extremesOffset], key);
    atomicMax(&extremes[extremesOffset + 1u], key);
  }`}),...I(e,{id:`${n}-square-sum`,operation:r,input:V,partials:H,output:U,blockOffsets:L,totalOffsets:R}),d(e,{id:`${n}-moments`,operation:r,variant:`moments`,bindings:[{name:`squareTotal`,view:U,type:`f32`,access:`read`},{name:`extremes`,view:W,type:`u32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${K}
${j}`,body:`let count = statistics[statisticsOffset + MOMENTS];
  let mean = statistics[statisticsOffset + MOMENTS + 1u];
  let hasRows = count >= 1.0;
  let minimum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset]), hasRows);
  let maximum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset + 1u]), hasRows);
  statistics[statisticsOffset + MOMENTS + 2u] = select(getQuietNaN(index), squareTotal[squareTotalOffset], hasRows);
  statistics[statisticsOffset + MOMENTS + 3u] = minimum;
  statistics[statisticsOffset + MOMENTS + 4u] = maximum;
  statistics[statisticsOffset + MOMENTS + 5u] = max(maximum - mean, mean - minimum);`})),{nodes:x,sortedRows:N,cellOffsets:O,cellKeys:S,statistics:G}}function F(e,t){let{id:n,operation:r,positions:i,sortedRows:o,cellOffsets:s,gridSize:c}=t,l=i.length,u=a(e,`${n}-sorted-points`,`uint32`,l*3);return{sortedPoints:u,nodes:[d(e,{id:`${n}-sorted-points`,operation:r,variant:`sorted-points`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`sortedRows`,view:o,type:`u32`,access:`read`},{name:`cellOffsets`,view:s,type:`u32`,access:`read`},{name:`sortedPoints`,view:u,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const CELL_COUNT: u32 = ${c[0]*c[1]}u;`,body:`if (index >= cellOffsets[cellOffsetsOffset + CELL_COUNT]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedPoints[sortedPointsOffset + index * 3u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedPoints[sortedPointsOffset + index * 3u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedPoints[sortedPointsOffset + index * 3u + 2u] = row;`})]}}function I(e,t){return[h(e,{id:`${t.id}-blocks`,operation:t.operation,segmentCount:t.partials.length,input:t.input,segmentOffsets:t.blockOffsets,output:t.partials,offsetsWithinInput:!0}),h(e,{id:`${t.id}-total`,operation:t.operation,segmentCount:1,input:t.partials,segmentOffsets:t.totalOffsets,output:t.output,offsetsWithinInput:!0})]}function L(e,t){let{id:n,operation:r,input:i,output:o}=t,s=i.length,c=Math.max(1,Math.ceil(s/k)),l=a(e,`${n}-block-offsets`,`uint32`,c+1),u=a(e,`${n}-total-offsets`,`uint32`,2),f=a(e,`${n}-partials`,`float32`,c);return[d(e,{id:`${n}-block-offsets`,operation:r,variant:`block-offsets`,bindings:[{name:`blockOffsets`,view:l,type:`u32`,access:`read_write`},{name:`totalOffsets`,view:u,type:`u32`,access:`read_write`}],invocationCount:c+1,declarations:`const ROW_COUNT: u32 = ${s}u;
const BLOCK_ROWS: u32 = ${k}u;
const BLOCK_COUNT: u32 = ${c}u;`,body:`blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
  }`}),...I(e,{id:`${n}-sum`,operation:r,input:i,partials:f,output:o,blockOffsets:l,totalOffsets:u})]}function R(e,t,n,r){return u(e,{id:t,operation:n,view:r,type:`u32`,value:`0u`})}var z=2048,B=`4294967040.0`,V=`9223372036854775808.0`;function H(t,n,r){if(e(Number.isInteger(n)&&n>=1),e(Number.isInteger(r)&&r>=1),n*r>2048)throw Error(`${t} needs at most ${z} histogram accumulators (bins x channels)`)}function U(t,n){let{id:r,operation:i,positions:o,parameters:s,gridSize:c,sortedRows:l,cellOffsets:u}=n,{slotCount:f,channelCount:p}=n;H(r,f,p);let m=n.extraBindings??[];e(m.length<=3);let h=f*p,g=a(t,`${r}-accumulators`,`uint32`,h*2+1),_=o.length,v=n.pairOrder===`ordered`,y=F(t,{id:r,operation:i,positions:o,sortedRows:l,cellOffsets:u,gridSize:c}),b=(e,t)=>`for (var candidateSlot = ${e}; candidateSlot < ${t}; candidateSlot++) {
        let neighbor = sortedPoints[sortedPointsOffset + candidateSlot * 3u + 2u];
        if (${v?`neighbor != focus`:`true`}) {
          let neighborX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]);
          let neighborY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]);
          ${v?`let deltaX = neighborX - x;
          let deltaY = neighborY - y;`:`// Oriented from the lower to the higher row ID, whichever row is the focus.
          let focusIsLower = focus < neighbor;
          let deltaX = select(x - neighborX, neighborX - x, focusIsLower);
          let deltaY = select(y - neighborY, neighborY - y, focusIsLower);`}
          let distanceSquared = deltaX * deltaX + deltaY * deltaY;
          if (distanceSquared <= lattice.radiusSquared) {
            let pairDistance = sqrt(distanceSquared);
            ${n.pairAction}
          }
        }
      }`;return{nodes:[...y.nodes,R(t,`${r}-clear`,i,g),d(t,{id:`${r}-pairs`,operation:i,variant:`pair-histogram-${n.pairOrder}`,bindings:[{name:`parameters`,view:s,type:`f32`,access:`read`},{name:`sortedPoints`,view:y.sortedPoints,type:`u32`,access:`read`},{name:`cellOffsets`,view:u,type:`u32`,access:`read`},...m,{name:`accumulators`,view:g,type:`atomic<u32>`,access:`read_write`}],invocationCount:_,workgroupSize:64,guardIndex:!1,declarations:`${M(c)}
const SLOT_COUNT: u32 = ${f}u;
const CHANNEL_COUNT: u32 = ${p}u;
const ACCUMULATOR_COUNT: u32 = ${h}u;
var<workgroup> localLow: array<atomic<u32>, ${h}>;
var<workgroup> localHigh: array<atomic<u32>, ${h}>;

// Adds the 64-bit amount (highAmount * 2^32 + amount) to an accumulator. highAmount is at most
// 2^31. A carry out of bit 63 of the workgroup partial sets the global overflow flag.
fn accumulateWide(slot: u32, channel: u32, amount: u32, highAmount: u32) {
  if ((amount == 0u && highAmount == 0u) || slot >= SLOT_COUNT) {
    return;
  }
  let word = slot * CHANNEL_COUNT + channel;
  var highAddend = highAmount;
  if (amount != 0u) {
    let previous = atomicAdd(&localLow[word], amount);
    if (previous > 0xffffffffu - amount) {
      highAddend += 1u;
    }
  }
  if (highAddend != 0u) {
    let previousHigh = atomicAdd(&localHigh[word], highAddend);
    if (previousHigh > 0xffffffffu - highAddend) {
      atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    }
  }
}

fn accumulate(slot: u32, channel: u32, amount: u32) {
  accumulateWide(slot, channel, amount, 0u);
}

// Rounds a non-negative scaled term to a u32 amount; negative or NaN terms give 0.
fn quantizePairAmount(scaledValue: f32) -> u32 {
  return select(0u, u32(min(scaledValue + 0.5, ${B})), scaledValue > 0.0);
}

// Accumulates a non-negative scaled term without letting its magnitude set the rounding unit:
// a term up to 2^32 - 256 units is rounded to the nearest unit into smallChannel; a larger
// term (rare: an outlier or sentinel pair) is rounded to a multiple of 2^12 units (relative
// error below 2^-20, plus the f32 ulp) and added as an exact 64-bit amount to largeChannel,
// which readers scale back by 2^12. A term beyond 2^75 units is clamped and
// raises the overflow flag, as does any accumulator sum that carries out of 64 bits.
fn accumulateScaled(slot: u32, smallChannel: u32, largeChannel: u32, scaledValue: f32) {
  if (!(scaledValue > 0.0)) {
    return;
  }
  if (scaledValue <= ${B}) {
    accumulateWide(slot, smallChannel, u32(scaledValue + 0.5), 0u);
    return;
  }
  var coarse = scaledValue * ${1/2**12};
  if (!(coarse <= ${V})) {
    atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    coarse = ${V};
  }
  // Exact in f32: scaling by powers of two, and coarse - high * 2^32 has no more bits than coarse.
  let high = floor(coarse * 2.3283064365386963e-10);
  let low = coarse - high * 4294967296.0;
  accumulateWide(slot, largeChannel, u32(min(low + 0.5, ${B})), u32(high));
}

${n.declarations??``}`,body:`for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += 64u) {
    atomicStore(&localLow[word], 0u);
    atomicStore(&localHigh[word], 0u);
  }
  workgroupBarrier();
  let lattice = readLattice();
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  if (index < INVOCATION_COUNT && index < includedCount && lattice.valid) {
    let focus = sortedPoints[sortedPointsOffset + index * 3u + 2u];
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    ${n.focusPrologue??``}
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    ${v?`let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${b(`begin`,`end`)}
    }`:`// Forward half stencil: the cells after the focus cell in row-major cell order. The rest
    // of the focus row up to the right neighbor cell is one slot range, the row below another.
    {
      let rowEnd = cellOffsets[cellOffsetsOffset + row * lattice.columns + lastColumn + 1u];
      ${b(`index + 1u`,`rowEnd`)}
    }
    if (row + 1u < lattice.rows) {
      let rowBase = (row + 1u) * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${b(`begin`,`end`)}
    }`}
    ${n.focusEpilogue??``}
  }
  workgroupBarrier();
  for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += 64u) {
    let low = atomicLoad(&localLow[word]);
    let high = atomicLoad(&localHigh[word]);
    if (low != 0u || high != 0u) {
      let previous = atomicAdd(&accumulators[accumulatorsOffset + word * 2u], low);
      let carry = select(0u, 1u, previous > 0xffffffffu - low);
      let highAddend = high + carry;
      if (highAddend < high) {
        atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
      }
      if (highAddend != 0u) {
        let previousHigh = atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], highAddend);
        if (previousHigh > 0xffffffffu - highAddend) {
          atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
        }
      }
    }
  }
`})],accumulators:g,sortedPoints:y.sortedPoints}}function W(e,t,n=``,r){return`
fn readPairAccumulator${n}(slot: u32, channel: u32) -> f32 {
  let word = (slot * ${t}u + channel) * 2u;
  return f32(${e}[${e}Offset + word + 1u]) * 4294967296.0 +
    f32(${e}[${e}Offset + word]);
}

fn readPairCount${n}(slot: u32, channel: u32) -> u32 {
  let word = (slot * ${t}u + channel) * 2u;
  return select(
    ${e}[${e}Offset + word],
    0xffffffffu,
    ${e}[${e}Offset + word + 1u] != 0u
  );
}
${r===void 0?``:`
fn readPairOverflow${n}() -> bool {
  return ${e}[${e}Offset + ${r*t*2}u] != 0u;
}
`}`}function G(e,t,n){if(t!==`study-window`&&(typeof t==`string`||t.length!==4))throw Error(`${e} bounds must be [minX, minY, maxX, maxY] or 'study-window'`);for(let r of[...t===`study-window`?[]:t,n])if(!Number.isFinite(r))throw Error(`${e} bounds and maximumDistance must be finite`);if(t!==`study-window`&&(t[2]<t[0]||t[3]<t[1]))throw Error(`${e} bounds must satisfy minX <= maxX and minY <= maxY`);if(n<=0)throw Error(`${e} maximumDistance must be positive`)}export{S as _,j as a,v as b,M as c,N as d,A as f,b as g,C as h,H as i,F as l,x as m,U as n,R as o,O as p,W as r,P as s,G as t,L as u,T as v,w as y};