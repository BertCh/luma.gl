import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{J as e}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{i as t,n,o as r,r as i}from"./gpu-raster-extrema-pyramid-DhOnk6uD.js";import{M as a,N as o,_ as s,f as c,i as l,j as u,k as d,l as f,t as p,v as m,w as h,y as g}from"./resources-CwvhEOBe.js";import{u as _}from"./interleave-brZZCO-f.js";import{t as v}from"./gpu-scan-BuGhTLDQ.js";import{t as ee}from"./gpu-compaction-Cpx1OCRM.js";import{t as te}from"./draw-command-buffer-BeqRkhBa.js";import{n as y,r as ne,s as b,t as x}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as re}from"./wgsl-common-BNRVzgd8.js";import{c as S,f as C,i as w,n as ie,p as T,r as ae,s as oe,u as se}from"./raster-utils-DhdnkkOa.js";import{a as E,c as ce,i as le,n as D,o as ue,r as O,s as k,t as de,u as fe}from"./raster-band-utils-CA-sh8xt.js";import{a as A,c as j,r as pe,u as me}from"./raster-grid-D2bmCwkc.js";import{U as he,a as M,c as ge,o as _e,r as ve,t as ye}from"./index-Bjk18V2I.js";import{t as N}from"./graph-buffers-B4_80RlX.js";import{t as be}from"./graph-swap-41Ru7LVu.js";import{t as xe}from"./summary-reader-CsgWxENf.js";import"./anchors-D9E9ucGL.js";import"./gazetteer-OhisynFF.js";import"./class-table-BVEo3-Vs.js";import"./hue-registry-4FjIJ9xz.js";import{c as Se,n as P}from"./live-text-83XBWyr1.js";import"./picking-DRmnefrA.js";import{a as Ce,o as we,s as F,t as I}from"./b14b-terrain-C8XHhTMr.js";import{r as Te,t as Ee}from"./cpu-dem-DnM3p94o.js";import{_ as L,a as De,g as Oe,k as ke}from"./terrain-furniture-C6hPKNNK.js";import{a as Ae,c as je}from"./terrain-places-CXVUagwV.js";import{n as Me,t as Ne}from"./terrain-ground-BLUqElOt.js";import{c as R,d as Pe,f as Fe,i as Ie,l as z,n as Le,o as Re,p as ze,r as Be,u as Ve}from"./summits-style-7ERPqyoL.js";var He=class{id;width;height;input;level;cases;segmentCounts;cellCount;constructor(e){this.id=e.id??`gpu-raster-contour-classifier`,this.width=e.width,this.height=e.height,this.input=e.input,this.level=e.level,this.cases=e.cases,this.segmentCounts=e.segmentCounts,this.cellCount=We(this),Ge(this.cases,this.cellCount,`${this.id} cases`),Ge(this.segmentCounts,this.cellCount,`${this.id} segmentCounts`),V(this.id,this.input.storage.values.buffer.graph,[this.cases,this.segmentCounts]),Ke(this.id,[this.cases,this.segmentCounts],[this.input.storage.values,...this.input.validity?[this.input.validity]:[],...U(this.level)?[U(this.level)]:[]])}addToGraph(e){let t=U(this.level);if(H(this.id,e,[this.input.storage.values,this.cases,this.segmentCounts,...this.input.validity?[this.input.validity]:[],...t?[t]:[]]),this.cellCount===0)return;let[n,r]=ae(e.device,this.width-1,this.height-1,this.id),i=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:this.cases,usage:`storage-write`},{buffer:this.segmentCounts,usage:`storage-write`}];this.input.validity&&i.push({buffer:this.input.validity,usage:`storage-read`}),t&&i.push({buffer:t,usage:`storage-read`}),B(e,{id:this.id,resources:i,source:Je(this,t),bindings:[{name:`sourceValues`,view:this.input.storage.values,writable:!1},{name:`cellCases`,view:this.cases,writable:!0},{name:`cellSegmentCounts`,view:this.segmentCounts,writable:!0},...this.input.validity?[{name:`sourceValidity`,view:this.input.validity,writable:!1}]:[],...t?[{name:`levelValues`,view:t,writable:!1}]:[]],dispatch:[n,r]})}},Ue=class{id;width;height;input;level;vertices;segmentCount;overflow;requiredSegmentCount;draw;drawCommandIndex;capacity;metadata;cellCount;constructor(e){if(this.id=e.id??`gpu-raster-contours`,this.width=e.width,this.height=e.height,this.input=e.input,this.level=e.level,this.vertices=e.vertices,this.segmentCount=e.segmentCount,this.overflow=e.overflow,this.requiredSegmentCount=e.requiredSegmentCount,this.draw=e.draw,this.drawCommandIndex=e.drawCommandIndex??0,this.metadata=e.metadata,this.cellCount=We(this),this.capacity=e.capacity??this.vertices.length/2,a(this.vertices,[`float32x2`],`${this.id} vertices`),this.vertices.length%2!=0)throw Error(`${this.id} vertices must contain an even number of positions`);if(!Number.isSafeInteger(this.capacity)||this.capacity<0||this.capacity>this.vertices.length/2)throw Error(`${this.id} capacity must fit in the available vertex pairs`);if(T(this.segmentCount,`uint32`,1,`${this.id} segmentCount`),T(this.overflow,`uint32`,1,`${this.id} overflow`),this.requiredSegmentCount&&T(this.requiredSegmentCount,`uint32`,1,`${this.id} requiredSegmentCount`),this.metadata&&(C(this.metadata,`${this.id} metadata`),this.metadata.width!==this.width||this.metadata.height!==this.height))throw Error(`${this.id} metadata dimensions must match the raster grid`);if(!Number.isSafeInteger(this.drawCommandIndex)||this.drawCommandIndex<0)throw Error(`${this.id} drawCommandIndex must be a non-negative integer`);if(!this.draw&&this.drawCommandIndex!==0)throw Error(`${this.id} drawCommandIndex requires a draw command`);this.draw&&qe(this.id,this.draw,this.drawCommandIndex);let t=[this.vertices,this.segmentCount,this.overflow,...this.requiredSegmentCount?[this.requiredSegmentCount]:[],...this.draw?[this.draw.words]:[]];V(this.id,this.input.storage.values.buffer.graph,t),Ke(this.id,t,[this.input.storage.values,...this.input.validity?[this.input.validity]:[],...U(this.level)?[U(this.level)]:[]])}addToGraph(e){let t=U(this.level);if(H(this.id,e,[this.input.storage.values,this.vertices,this.segmentCount,this.overflow,...this.input.validity?[this.input.validity]:[],...t?[t]:[],...this.requiredSegmentCount?[this.requiredSegmentCount]:[],...this.draw?[this.draw.words]:[]]),this.cellCount>0&&(e.device.limits.maxComputeInvocationsPerWorkgroup<256||e.device.limits.maxComputeWorkgroupSizeX<256))throw Error(`${this.id} scan exceeds device workgroup limits`);let n=h(e,`${this.id}-cases`,`uint32`,this.cellCount),r=h(e,`${this.id}-segment-counts`,`uint32`,this.cellCount),i=h(e,`${this.id}-segment-offsets`,`uint32`,this.cellCount);new He({id:`${this.id}-classify`,width:this.width,height:this.height,input:this.input,level:this.level,cases:n,segmentCounts:r}).addToGraph(e),e.add(new v({id:`${this.id}-scan`,input:r,output:i,mode:`exclusive`})),this.cellCount>0&&this.capacity>0&&this.addScatterPass(e,n,i,t),this.addSummaryPass(e,r,i)}addScatterPass(e,t,n,r){let[i,a]=ae(e.device,this.width-1,this.height-1,`${this.id}-scatter`),o=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:t,usage:`storage-read`},{buffer:n,usage:`storage-read`},{buffer:this.vertices,usage:`storage-write`}];r&&o.push({buffer:r,usage:`storage-read`}),B(e,{id:`${this.id}-scatter`,resources:o,source:Ye(this,t,n,r),bindings:[{name:`sourceValues`,view:this.input.storage.values,writable:!1},{name:`cellCases`,view:t,writable:!1},{name:`cellOffsets`,view:n,writable:!1},{name:`outputVertices`,view:this.vertices,writable:!0},...r?[{name:`levelValues`,view:r,writable:!1}]:[]],dispatch:[i,a]})}addSummaryPass(e,t,n){let r=this.cellCount>0,i=[...r?[{name:`cellSegmentCounts`,view:t,writable:!1}]:[],...r?[{name:`cellOffsets`,view:n,writable:!1}]:[],{name:`outputSegmentCount`,view:this.segmentCount,writable:!0},{name:`outputOverflow`,view:this.overflow,writable:!0},...this.requiredSegmentCount?[{name:`outputRequiredCount`,view:this.requiredSegmentCount,writable:!0}]:[],...this.draw?[{name:`drawCommandWords`,view:this.draw.words,writable:!0}]:[]];B(e,{id:`${this.id}-publish`,resources:i.map(({view:e,writable:t})=>({buffer:e,usage:t?`storage-write`:`storage-read`})),source:W(this,t,n,i),bindings:i,dispatch:[1,1]})}};function B(e,t){e.addComputePass({id:t.id,resources:t.resources,compile:({device:e})=>{let n=t.bindings.map((e,t)=>({name:e.name,type:e.writable?`storage`:`read-only-storage`,group:0,location:t})),r=new _(e,{id:t.id,source:t.source,shaderLayout:{bindings:n}}),i={};for(let e of t.bindings)i[e.name]=e.view;let a=x(i);return{encode:({computePass:e,getBuffer:n})=>{let i=a(n);i&&r.setBindings(i),r.dispatch(e,t.dispatch[0],t.dispatch[1])},destroy:()=>r.destroy()}}})}function We(e){let t=e.id??`gpu-raster-contours`;if(!Number.isSafeInteger(e.width)||e.width<=0||!Number.isSafeInteger(e.height)||e.height<=0)throw Error(`${t} dimensions must be positive integers`);let n=e.width*e.height;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`${t} pixel count must fit in uint32`);let r=(e.width-1)*(e.height-1);if(r*2>4294967295)throw Error(`${t} maximum segment count must fit in uint32`);if(e.input.storage.kind!==`buffer`)throw Error(`${t} requires a buffer-backed raster band`);let i=se(e.input,e,`${t} input`);if(w(e.input.scale??1),w(e.input.offset??0),typeof e.level==`number`){if(!Number.isFinite(e.level))throw Error(`${t} contour level must be finite`);w(e.level)}else if(T(e.level,`float32`,1,`${t} level`),e.level.buffer.graph!==i)throw Error(`${t} contour level must belong to the same graph`);return r}function Ge(e,t,n){if(u(e,n),e.length!==t)throw Error(`${n} must contain exactly one value per raster cell`)}function V(e,t,n){if(n.some(e=>e.buffer.graph!==t))throw Error(`${e} resources must belong to the same graph`)}function Ke(e,t,n){let r=new Set(t.map(e=>e.buffer));if(r.size!==t.length||n.some(e=>r.has(e.buffer)))throw Error(`${e} input and output resources must use separate buffers`)}function qe(e,t,n){if(t.type!==`draw`||t.recordByteLength!==16)throw Error(`${e} requires non-indexed two-vertex indirect draw records`);if(n>=t.capacity)throw Error(`${e} drawCommandIndex exceeds the indirect draw capacity`);if(u(t.words,`${e} indirect command words`),t.words.buffer!==t.buffer||t.words.length<t.capacity*4)throw Error(`${e} indirect command words must span every draw record`)}function H(e,t,n){for(let r of n){if(r.buffer.graph!==t)throw Error(`${e} resources must belong to the target graph`);ie(t.device,r,`${e} ${r.buffer.id}`)}}function U(e){return typeof e==`number`?void 0:e}function Je(e,t){let n=e.input.validity,r=n?4:3,i=n?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;
const VALIDITY_OFFSET: u32 = ${d(n)}u;`:``,a=t?`@group(0) @binding(${r}) var<storage, read> levelValues: array<f32>;
const LEVEL_OFFSET: u32 = ${d(t)}u;`:``,o=t?`levelValues[LEVEL_OFFSET]`:w(e.level),s=e.input.noDataValue!==void 0&&!Number.isNaN(e.input.noDataValue)?` && rawValue != ${oe(e.input.noDataValue,e.input.format)}`:``,c=n?` && sourceValidity[VALIDITY_OFFSET + index] != 0u`:``,l=e.input.format===`float32`?` && isFiniteValue(rawValue)`:``;return`
const WIDTH: u32 = ${e.width}u;
const CELL_WIDTH: u32 = ${e.width-1}u;
const CELL_HEIGHT: u32 = ${e.height-1}u;
const SOURCE_OFFSET: u32 = ${d(e.input.storage.values)}u;
const CASE_OFFSET: u32 = ${d(e.cases)}u;
const COUNT_OFFSET: u32 = ${d(e.segmentCounts)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<${S(e.input.format)}>;
@group(0) @binding(1) var<storage, read_write> cellCases: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellSegmentCounts: array<u32>;
${i}
${a}

${re}

fn isValidSample(index: u32, rawValue: ${S(e.input.format)}, value: f32) -> bool {
  return isFiniteValue(value)${l}${s}${c};
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= CELL_WIDTH || globalId.y >= CELL_HEIGHT) { return; }
  let cellIndex = globalId.y * CELL_WIDTH + globalId.x;
  let topLeftIndex = globalId.y * WIDTH + globalId.x;
  let topRightIndex = topLeftIndex + 1u;
  let bottomLeftIndex = topLeftIndex + WIDTH;
  let bottomRightIndex = bottomLeftIndex + 1u;
  let topLeftRaw = sourceValues[SOURCE_OFFSET + topLeftIndex];
  let topRightRaw = sourceValues[SOURCE_OFFSET + topRightIndex];
  let bottomRightRaw = sourceValues[SOURCE_OFFSET + bottomRightIndex];
  let bottomLeftRaw = sourceValues[SOURCE_OFFSET + bottomLeftIndex];
  let topLeft = f32(topLeftRaw) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let topRight = f32(topRightRaw) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let bottomRight = f32(bottomRightRaw) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let bottomLeft = f32(bottomLeftRaw) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let contourLevel = ${o};
  if (!isFiniteValue(contourLevel) ||
      !isValidSample(topLeftIndex, topLeftRaw, topLeft) ||
      !isValidSample(topRightIndex, topRightRaw, topRight) ||
      !isValidSample(bottomRightIndex, bottomRightRaw, bottomRight) ||
      !isValidSample(bottomLeftIndex, bottomLeftRaw, bottomLeft)) {
    cellCases[CASE_OFFSET + cellIndex] = 0u;
    cellSegmentCounts[COUNT_OFFSET + cellIndex] = 0u;
    return;
  }

  let caseIndex = select(0u, 1u, topLeft >= contourLevel) |
    select(0u, 2u, topRight >= contourLevel) |
    select(0u, 4u, bottomRight >= contourLevel) |
    select(0u, 8u, bottomLeft >= contourLevel);
  var encodedCase = caseIndex;
  var segmentCount = select(1u, 0u, caseIndex == 0u || caseIndex == 15u);
  if (caseIndex == 5u || caseIndex == 10u) {
    let magnitude = max(1.0, max(abs(contourLevel),
      max(max(abs(topLeft), abs(topRight)), max(abs(bottomRight), abs(bottomLeft)))));
    let topLeftDelta = topLeft / magnitude - contourLevel / magnitude;
    let topRightDelta = topRight / magnitude - contourLevel / magnitude;
    let bottomRightDelta = bottomRight / magnitude - contourLevel / magnitude;
    let bottomLeftDelta = bottomLeft / magnitude - contourLevel / magnitude;
    let determinant = topLeftDelta * bottomRightDelta - topRightDelta * bottomLeftDelta;
    let topRightPair = determinant > 0.0 || (determinant == 0.0 && caseIndex == 5u);
    encodedCase = caseIndex | select(0u, 16u, topRightPair);
    segmentCount = 2u;
  }
  cellCases[CASE_OFFSET + cellIndex] = encodedCase;
  cellSegmentCounts[COUNT_OFFSET + cellIndex] = segmentCount;
}`}function Ye(e,t,n,r){let i=r?`@group(0) @binding(4) var<storage, read> levelValues: array<f32>;
const LEVEL_OFFSET: u32 = ${d(r)}u;`:``,a=r?`levelValues[LEVEL_OFFSET]`:w(e.level),o=e.metadata?.pixelInterpretation===`point`?`0.0`:`0.5`;return`
const WIDTH: u32 = ${e.width}u;
const CELL_WIDTH: u32 = ${e.width-1}u;
const CELL_HEIGHT: u32 = ${e.height-1}u;
const SEGMENT_CAPACITY: u32 = ${e.capacity}u;
const SOURCE_OFFSET: u32 = ${d(e.input.storage.values)}u;
const CASE_OFFSET: u32 = ${d(t)}u;
const CELL_OFFSET: u32 = ${d(n)}u;
const VERTEX_OFFSET: u32 = ${d(e.vertices)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<${S(e.input.format)}>;
@group(0) @binding(1) var<storage, read> cellCases: array<u32>;
@group(0) @binding(2) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputVertices: array<f32>;
${i}

${re}

fn getInterpolation(first: f32, second: f32, contourLevel: f32) -> f32 {
  let directNumerator = contourLevel - first;
  let directDenominator = second - first;
  if (isFiniteValue(directNumerator) && isFiniteValue(directDenominator) &&
      directDenominator != 0.0) {
    return clamp(directNumerator / directDenominator, 0.0, 1.0);
  }
  let magnitude = max(1.0, max(abs(contourLevel), max(abs(first), abs(second))));
  let numerator = contourLevel / magnitude - first / magnitude;
  let denominator = second / magnitude - first / magnitude;
  if (denominator == 0.0) { return 0.5; }
  return clamp(numerator / denominator, 0.0, 1.0);
}

fn getEdgePosition(
  edge: u32,
  topLeft: f32,
  topRight: f32,
  bottomRight: f32,
  bottomLeft: f32,
  contourLevel: f32,
  origin: vec2<f32>
) -> vec2<f32> {
  switch edge {
    case 0u: {
      return origin + vec2<f32>(getInterpolation(topLeft, topRight, contourLevel), 0.0);
    }
    case 1u: {
      return origin + vec2<f32>(1.0, getInterpolation(topRight, bottomRight, contourLevel));
    }
    case 2u: {
      return origin + vec2<f32>(getInterpolation(bottomLeft, bottomRight, contourLevel), 1.0);
    }
    default: {
      return origin + vec2<f32>(0.0, getInterpolation(topLeft, bottomLeft, contourLevel));
    }
  }
}

fn writeSegment(
  segmentIndex: u32,
  firstEdge: u32,
  secondEdge: u32,
  topLeft: f32,
  topRight: f32,
  bottomRight: f32,
  bottomLeft: f32,
  contourLevel: f32,
  origin: vec2<f32>
) {
  if (segmentIndex >= SEGMENT_CAPACITY) { return; }
  let first = getEdgePosition(
    firstEdge, topLeft, topRight, bottomRight, bottomLeft, contourLevel, origin
  );
  let second = getEdgePosition(
    secondEdge, topLeft, topRight, bottomRight, bottomLeft, contourLevel, origin
  );
  let outputIndex = VERTEX_OFFSET + segmentIndex * 4u;
  outputVertices[outputIndex] = first.x;
  outputVertices[outputIndex + 1u] = first.y;
  outputVertices[outputIndex + 2u] = second.x;
  outputVertices[outputIndex + 3u] = second.y;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= CELL_WIDTH || globalId.y >= CELL_HEIGHT) { return; }
  let cellIndex = globalId.y * CELL_WIDTH + globalId.x;
  let encodedCase = cellCases[CASE_OFFSET + cellIndex];
  let caseIndex = encodedCase & 15u;
  if (caseIndex == 0u || caseIndex == 15u) { return; }
  let segmentIndex = cellOffsets[CELL_OFFSET + cellIndex];
  if (segmentIndex >= SEGMENT_CAPACITY) { return; }
  let topLeftIndex = globalId.y * WIDTH + globalId.x;
  let topLeft = f32(sourceValues[SOURCE_OFFSET + topLeftIndex]) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let topRight = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + 1u]) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let bottomLeft = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + WIDTH]) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let bottomRight = f32(sourceValues[SOURCE_OFFSET + topLeftIndex + WIDTH + 1u]) * ${w(e.input.scale??1)} + ${w(e.input.offset??0)};
  let contourLevel = ${a};
  let origin = vec2<f32>(f32(globalId.x) + ${o}, f32(globalId.y) + ${o});
  var firstEdge = 0u;
  var secondEdge = 0u;
  switch caseIndex {
    case 1u, 14u: { firstEdge = 3u; secondEdge = 0u; }
    case 2u, 13u: { firstEdge = 0u; secondEdge = 1u; }
    case 3u, 12u: { firstEdge = 3u; secondEdge = 1u; }
    case 4u, 11u: { firstEdge = 1u; secondEdge = 2u; }
    case 6u, 9u: { firstEdge = 0u; secondEdge = 2u; }
    case 7u, 8u: { firstEdge = 2u; secondEdge = 3u; }
    default: {
      let topRightPair = (encodedCase & 16u) != 0u;
      firstEdge = select(3u, 0u, topRightPair);
      secondEdge = select(0u, 1u, topRightPair);
      let thirdEdge = select(1u, 2u, topRightPair);
      let fourthEdge = select(2u, 3u, topRightPair);
      writeSegment(segmentIndex + 1u, thirdEdge, fourthEdge, topLeft, topRight,
        bottomRight, bottomLeft, contourLevel, origin);
    }
  }
  writeSegment(segmentIndex, firstEdge, secondEdge, topLeft, topRight,
    bottomRight, bottomLeft, contourLevel, origin);
}`}function W(e,t,n,r){let i=r.map((e,t)=>`@group(0) @binding(${t}) var<storage, ${e.writable?`read_write`:`read`}> ${e.name}: array<u32>;`).join(`
`),a=e.cellCount>0?`cellOffsets[${d(n)}u + ${e.cellCount-1}u] + cellSegmentCounts[${d(t)}u + ${e.cellCount-1}u]`:`0u`,o=e.requiredSegmentCount?`outputRequiredCount[${d(e.requiredSegmentCount)}u] = requiredCount;`:``,s=``;if(e.draw){let t=d(e.draw.words)+e.drawCommandIndex*4;s=`drawCommandWords[${t}u] = 2u;
  drawCommandWords[${t+1}u] = clampedCount;
  drawCommandWords[${t+2}u] = 0u;
  drawCommandWords[${t+3}u] = 0u;`}return`
${i}

@compute @workgroup_size(1)
fn main() {
  let requiredCount = ${a};
  let clampedCount = min(requiredCount, ${e.capacity}u);
  outputSegmentCount[${d(e.segmentCount)}u] = clampedCount;
  outputOverflow[${d(e.overflow)}u] = select(0u, 1u, requiredCount > ${e.capacity}u);
  ${o}
  ${s}
}`}var G=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-contours`,this.props=e;let{id:t}=this;k(t,e.grid);let{overflow:n}=e.output.status;if(e.levels.length<1)throw Error(`${t} requires at least one contour level`);for(let[n,r]of e.levels.entries())Ze(t,n,r);if(g(n,[`uint32`],`${t} output.status.overflow`),n.length<1)throw Error(`${t} output.status.overflow must contain one uint32 row`);if([...D(e.elevation),...e.levels.flatMap(e=>[e.vertices,e.status.count,e.status.requiredCount,e.status.overflow,typeof e.level==`number`?void 0:e.level])].filter(e=>e!==void 0).map(e=>e.buffer).includes(n.buffer))throw Error(`${t} outputs must not share buffers with each other or with inputs`)}getCommandNodes(e){let{id:t,props:n}=this,{width:r,height:i}=n.grid,{overflow:a}=n.output.status;E(t,e,n.elevation,[]),s(t,e,[a,...n.levels.flatMap(e=>[e.vertices,e.status.count,e.status.requiredCount,e.status.overflow,typeof e.level==`number`?void 0:e.level,e.draw?.words])]);let o=O(e,t,n.elevation,n.grid,!1),c=[...o.nodes];for(let[s,u]of n.levels.entries()){let d=u.draw!==void 0&&!Xe(u),f=u.status.overflow??h(e,`${t}-level-${s}-overflow`,`uint32`,1);c.push(...l(e,()=>new Ue({id:`${t}-level-${s}`,width:r,height:i,input:o.band,level:u.level,vertices:u.vertices,segmentCount:u.status.count,overflow:f,requiredSegmentCount:u.status.requiredCount,draw:d?void 0:u.draw,drawCommandIndex:d?void 0:u.drawCommandIndex,capacity:u.capacity,metadata:n.grid}).addToGraph(e))),d&&u.draw&&c.push(K(e,t,s,u,u.draw)),c.push(b(e,{id:`${t}-overflow-${s}`,operation:`GPUTerrainContours`,variant:`overflow`,bindings:[{name:`levelOverflow`,view:f,type:`u32`,access:`read`},{name:`outputOverflow`,view:a,type:`u32`,access:`read_write`}],invocationCount:1,body:`let previous = ${s===0?`0u`:`outputOverflow[outputOverflowOffset]`};
  outputOverflow[outputOverflowOffset] =
    select(0u, 1u, previous != 0u || levelOverflow[levelOverflowOffset] != 0u);`}))}return c}};function Xe(e){return(e.drawLayout??`instanced`)===`instanced`&&(e.verticesPerInstance??2)===2}function Ze(e,t,n){let r=`${e} level ${t}`,{draw:i}=n;if(!i){if(n.drawCommandIndex!==void 0||n.drawLayout!==void 0||n.verticesPerInstance!==void 0)throw Error(`${r} drawCommandIndex, drawLayout and verticesPerInstance require draw`);return}let a=n.drawLayout??`instanced`;if(a!==`instanced`&&a!==`line-list`)throw Error(`${r} drawLayout must be 'instanced' or 'line-list'`);if(n.verticesPerInstance!==void 0){if(a!==`instanced`)throw Error(`${r} verticesPerInstance requires drawLayout 'instanced'`);if(!Number.isSafeInteger(n.verticesPerInstance)||n.verticesPerInstance<1)throw Error(`${r} verticesPerInstance must be a positive integer`)}let o=n.drawCommandIndex??0;if(!Number.isSafeInteger(o)||o<0)throw Error(`${r} drawCommandIndex must be a non-negative integer`);if(i.type!==`draw`||i.recordByteLength!==16)throw Error(`${r} draw must be a non-indexed indirect draw record view`);if(o>=i.capacity)throw Error(`${r} drawCommandIndex exceeds the indirect draw capacity`);if(g(i.words,[`uint32`],`${r} draw words`),i.words.buffer!==i.buffer||i.words.length<i.capacity*4)throw Error(`${r} draw words must span every draw record`)}function K(e,t,n,r,i){let a=r.capacity??Math.floor(r.vertices.length/2),o=(r.drawCommandIndex??0)*4,s=r.drawLayout===`line-list`;return b(e,{id:`${t}-level-${n}-draw`,operation:`GPUTerrainContours`,variant:`draw`,bindings:[{name:`segmentCount`,view:r.status.count,type:`u32`,access:`read`},{name:`drawWords`,view:i.words,type:`u32`,access:`read_write`}],invocationCount:1,body:`let count = min(segmentCount[segmentCountOffset], ${a}u);
  let base = drawWordsOffset + ${o}u;
  drawWords[base] = ${s?`2u * count`:`${r.verticesPerInstance??2}u`};
  drawWords[base + 1u] = ${s?`1u`:`count`};
  drawWords[base + 2u] = 0u;
  drawWords[base + 3u] = 0u;`})}var q=16,J=64;function Y(e,t){let n=t??q;if(!Number.isInteger(n)||n<1||n>J)throw Error(`${e} maximumRadiusPixels must be an integer in [1, ${J}]`);return n}function Qe(e,t){if(!t.every(e=>Number.isFinite(e)&&e>0))throw Error(`${e} cell size must be finite and positive`)}function X(e,t,n,r,i=!1){if(!Number.isFinite(n)||(i?n<=r:n<r))throw Error(`${e} ${t} must be finite and ${i?`greater than`:`at least`} ${r}`)}var $e=`
const PI: f32 = 3.141592653589793;
const NAN_BITS: u32 = 0x7fc00000u;
const INFINITY_BITS: u32 = 0x7f800000u;
// WGSL rejects constant-evaluated NaN and infinity; a runtime zero (seed & 0u) defers evaluation.
fn getNan(seed: u32) -> f32 { return bitcast<f32>(NAN_BITS | (seed & 0u)); }
fn getInfinity(seed: u32) -> f32 { return bitcast<f32>(INFINITY_BITS | (seed & 0u)); }`,et=`
fn isInDisc(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  let scaledX = (f32(pixel.x) - centre.x) * cell.x;
  let scaledY = (f32(pixel.y) - centre.y) * cell.y;
  return scaledX * scaledX + scaledY * scaledY <= radiusSquared;
}
fn isOnDiscRing(pixel: vec2<i32>, centre: vec2<f32>, cell: vec2<f32>, radiusSquared: f32) -> bool {
  for (var neighborY = -1; neighborY <= 1; neighborY++) {
    for (var neighborX = -1; neighborX <= 1; neighborX++) {
      if ((neighborX != 0 || neighborY != 0) &&
          !isInDisc(pixel + vec2<i32>(neighborX, neighborY), centre, cell, radiusSquared)) {
        return true;
      }
    }
  }
  return false;
}
fn getMaximumRadius(cell: vec2<f32>) -> f32 {
  return min(f32(MAXIMUM_RADIUS_PIXELS) * cell.x, f32(MAXIMUM_RADIUS_PIXELS) * cell.y);
}
fn isRadiusClamped(requested: f32, cell: vec2<f32>) -> bool {
  return requested > getMaximumRadius(cell);
}
fn getEffectiveRadius(requested: f32, cell: vec2<f32>) -> f32 {
  return min(requested, getMaximumRadius(cell));
}
fn getDiscExtent(radius: f32, cellSize: f32) -> i32 {
  return min(i32(floor(radius / cellSize + 0.5)) + 1, MAXIMUM_RADIUS_PIXELS);
}`,tt=`
fn isOffsetInDisc(dx: i32, dy: i32, cell: vec2<f32>, radiusSquared: f32) -> bool {
  return isInDisc(vec2<i32>(dx, dy), vec2<f32>(0.0, 0.0), cell, radiusSquared);
}
fn getDiscRowHalfWidth(dy: i32, cell: vec2<f32>, radiusSquared: f32) -> i32 {
  let scaledY = f32(dy) * cell.y;
  let remaining = max(radiusSquared - scaledY * scaledY, 0.0);
  var halfWidth = clamp(i32(floor(sqrt(remaining) / cell.x)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfWidth > 0 && !isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { halfWidth--; }
  while (halfWidth <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(halfWidth + 1, dy, cell, radiusSquared)) { halfWidth++; }
  if (!isOffsetInDisc(halfWidth, dy, cell, radiusSquared)) { return -1; }
  return halfWidth;
}
fn getDiscColumnHalfHeight(cell: vec2<f32>, radiusSquared: f32) -> i32 {
  var halfHeight = clamp(i32(floor(sqrt(radiusSquared) / cell.y)), 0, MAXIMUM_RADIUS_PIXELS + 1);
  while (halfHeight > 0 && !isOffsetInDisc(0, halfHeight, cell, radiusSquared)) { halfHeight--; }
  while (halfHeight <= MAXIMUM_RADIUS_PIXELS && isOffsetInDisc(0, halfHeight + 1, cell, radiusSquared)) { halfHeight++; }
  return halfHeight;
}`,nt={cellSizeIndex:2,northEdgeIndex:4};function rt(e,t,n=new Float32Array(8)){let r=`getGPUTerrainSummitsParameterValues`;return le(r,n,8),X(r,`radius`,e.radius,0,!0),X(r,`minimumDrop`,e.minimumDrop??0,0),Qe(r,pe(t).cellSize),n[0]=e.radius,n[1]=e.minimumDrop??0,n[6]=0,n[7]=0,me(n,t,nt),n}var it=class{id;props;pyramidLayout;constructor(e){this.id=e.id??`terrain-summits`,this.props=e;let{id:t}=this,{output:n}=e,r=n.status?.radiusClamped,a=k(t,e.grid);if(j(t,e.cellSizeMode??`uniform`),ce(t,e.grid,e.cellSizeMode),Y(t,e.maximumRadiusPixels),e.incompleteNeighborhood!==void 0&&e.incompleteNeighborhood!==`reject`&&e.incompleteNeighborhood!==`ignore`)throw Error(`${t} incompleteNeighborhood must be 'reject' or 'ignore'`);if(!n.summitMask&&!n.drop&&!n.summits)throw Error(`${t} requires at least one output`);if(fe(t,e.parameters,8),e.pyramid){if(e.incompleteNeighborhood===void 0||e.incompleteNeighborhood===`reject`)throw Error(`${t} pyramid requires incompleteNeighborhood 'ignore'`);this.pyramidLayout=st(t,e.pyramid,e.grid)}else this.pyramidLayout=i(e.grid,{firstBlockSize:at,footprint:`cell`});if(n.summitMask&&(g(n.summitMask,[`uint32`],`${t} output.summitMask`),n.summitMask.length!==a))throw Error(`${t} output.summitMask must contain one value per pixel`);if(n.drop&&(g(n.drop,[`float32`],`${t} output.drop`),n.drop.length!==a))throw Error(`${t} output.drop must contain one value per pixel`);if(n.summits&&c(t,n.summits),n.summitDrop){if(!n.summits)throw Error(`${t} output.summitDrop requires output.summits`);if(g(n.summitDrop,[`float32`],`${t} output.summitDrop`),n.summitDrop.length<n.summits.ids.length)throw Error(`${t} output.summitDrop must contain at least summits.ids.length rows`)}if(r&&(g(r,[`uint32`],`${t} output.status.radiusClamped`),r.length<1))throw Error(`${t} output.status.radiusClamped must contain one row`);ue(t,[n.summitMask,n.drop,r,n.summitDrop,n.summits?.ids,n.summits?.status.count,n.summits?.status.overflow,n.summits?.status.requiredCount],[...D(e.elevation),e.parameters,...e.pyramid?[e.pyramid.combined]:[]])}getCommandNodes(e){let{id:t,props:r}=this,{width:i,height:a}=r.grid,o=r.output.summits,c=r.output.status?.radiusClamped;E(t,e,r.elevation,[]),s(t,e,[r.parameters,r.pyramid?.combined,r.output.summitMask,r.output.drop,c,r.output.summitDrop,o?.ids,o?.status.count,o?.status.overflow,o?.status.requiredCount]);let l=i*a,u=r.cellSizeMode??`uniform`,d=Y(t,r.maximumRadiusPixels),f=O(e,t,r.elevation,r.grid,!0),p=[...f.nodes],m=f.band.validity,g=r.output.summitMask??(o?h(e,`${t}-summit-mask`,`uint32`,l):void 0),_=r.output.drop??(r.output.summitDrop?h(e,`${t}-drop`,`float32`,l):void 0),{pyramidLayout:v}=this,te=(r.incompleteNeighborhood??`reject`)===`reject`,x;if(r.pyramid)x=r.pyramid.combined;else{x=h(e,`${t}-pyramid-maximum`,`float32`,v.length);let r=f.band.storage.values,i=m;te&&(r=h(e,`${t}-pyramid-values`,`float32`,l),i=h(e,`${t}-pyramid-validity`,`uint32`,l),p.push(b(e,{id:`${t}-pyramid-input`,operation:`GPUTerrainSummits`,variant:`pyramid-input`,bindings:[{name:`sourceValues`,view:f.band.storage.values,type:`f32`,access:`read`},{name:`sourceValidity`,view:m,type:`u32`,access:`read`},{name:`blockValues`,view:r,type:`f32`,access:`read_write`},{name:`blockValidity`,view:i,type:`u32`,access:`read_write`}],invocationCount:l,body:`let isValid = sourceValidity[sourceValidityOffset + index] != 0u;
  blockValues[blockValuesOffset + index] = select(${ot}, sourceValues[sourceValuesOffset + index], isValid);
  blockValidity[blockValidityOffset + index] = 1u;`}))),p.push(...n(e,{id:`${t}-pyramid`,layout:v,values:r,validity:i,maximum:x}))}let re=[{name:`elevationValues`,view:f.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:m,type:`u32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`pyramid`,view:x,type:`f32`,access:`read`},...g?[{name:`maskValues`,view:g,type:`u32`,access:`read_write`}]:[],..._?[{name:`dropValues`,view:_,type:`f32`,access:`read_write`}]:[],...c?[{name:`overflowValues`,view:c,type:`atomic<u32>`,access:`read_write`}]:[]];if(c&&p.push(y(e,{id:`${t}-overflow-clear`,operation:`GPUTerrainSummits`,view:c,type:`u32`,value:`0u`,componentCount:1})),p.push(b(e,{id:`${t}-summits`,operation:`GPUTerrainSummits`,variant:`disc-${u}-${r.incompleteNeighborhood??`reject`}`,bindings:re,invocationCount:l,declarations:lt({width:i,height:a,maximumRadiusPixels:d,cellSizeMode:u,rejectIncomplete:(r.incompleteNeighborhood??`reject`)===`reject`,reportsClamp:!!c,layout:v}),body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  summitDrop = getNan(index);
  let isSummit = evaluateSummit(column, row, index);
  ${g?`maskValues[maskValuesOffset + index] = select(0u, 1u, isSummit);`:``}
  ${_?`dropValues[dropValuesOffset + index] = select(getNan(index), summitDrop, isSummit);`:``}`})),o&&g){let n=h(e,`${t}-pixel-ids`,`uint32`,l),i=h(e,`${t}-compact-rows`,`uint32`,l),a=h(e,`${t}-summit-total`,`uint32`,1);p.push(b(e,{id:`${t}-pixel-ids`,operation:`GPUTerrainSummits`,variant:`pixel-ids`,bindings:[{name:`pixelIds`,view:n,type:`u32`,access:`read_write`}],invocationCount:l,body:`pixelIds[pixelIdsOffset + index] = index;`}),...new ee({id:`${t}-compaction`,input:n,flags:g,output:i,count:a}).getCommandNodes(e),ne(e,{id:`${t}-publish`,operation:`GPUTerrainSummits`,requiredCount:a,compactIds:i,output:o})),r.output.summitDrop&&_&&p.push(b(e,{id:`${t}-output-drop`,operation:`GPUTerrainSummits`,variant:`gather-drop`,bindings:[{name:`idsIn`,view:o.ids,type:`u32`,access:`read`},{name:`countIn`,view:o.status.count,type:`u32`,access:`read`},{name:`dropIn`,view:_,type:`f32`,access:`read`},{name:`dropOut`,view:r.output.summitDrop,type:`f32`,access:`read_write`}],invocationCount:o.ids.length,body:`if (index < countIn[countInOffset]) {
    dropOut[dropOutOffset + index] = dropIn[dropInOffset + idsIn[idsInOffset + index]];
  } else {
    dropOut[dropOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
  }`}))}return p}},at=2,ot=`3.4028234663852886e38`;function st(e,t,n){let{layout:r,combined:i}=t;if(!r||!i)throw Error(`${e} pyramid must provide a layout and a combined view`);if(r.width!==n.width||r.height!==n.height)throw Error(`${e} pyramid covers ${r.width}x${r.height} pixels but the grid is ${n.width}x${n.height}`);if(r.footprint!==`cell`)throw Error(`${e} pyramid must use the 'cell' footprint`);let a=r.levels[r.levels.length-1];if(!a||a.width!==1||a.height!==1)throw Error(`${e} pyramid must reach a single top cell`);if(g(i,[`float32`],`${e} pyramid combined`),i.length!==2*r.length)throw Error(`${e} pyramid combined must contain ${2*r.length} values`);return r}function ct(e){return(3*e.levels.length+4)*3*4}function lt(n){let{width:r,height:i,maximumRadiusPixels:a,cellSizeMode:o,layout:s}=n,c=3*s.levels.length+4;e(ct(s)+256<8192);let l=s.levels.length-1;return`const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${i}u;
const MAXIMUM_RADIUS_PIXELS: i32 = ${a};
const REJECT_INCOMPLETE: bool = ${n.rejectIncomplete};
const NO_INDEX: u32 = 0xffffffffu;
const FLOAT_MAXIMUM: f32 = ${ot};
const STACK_SIZE: u32 = ${c}u;
${$e}
${de}
${A(o,nt)}
${et}
${tt}
${t(s,`pyramid`)}
var<private> summitDrop: f32;
var<private> stackLevel: array<u32, ${c}>;
var<private> stackColumn: array<u32, ${c}>;
var<private> stackRow: array<u32, ${c}>;
// Whether a pixel of height \`other\` beats the centre: >= where it has the smaller index, else >.
fn beatsCentre(other: f32, ownHeight: f32, hasSmallerIndex: bool) -> bool {
  return select((other > ownHeight), (other >= ownHeight), hasSmallerIndex);
}
// Returns true when no disc pixel of rows [rowLow, rowHigh] beats the centre pixel.
fn isHalfDiscDominated(column: i32, row: i32, rowLow: i32, rowHigh: i32, ownHeight: f32,
    hasSmallerIndex: bool, exact: bool, cell: vec2<f32>, radiusSquared: f32) -> bool {
  if (rowLow > rowHigh) { return true; }
  var depth = 1u;
  stackLevel[0] = ${l}u;
  stackColumn[0] = 0u;
  stackRow[0] = 0u;
  loop {
    if (depth == 0u) { break; }
    depth--;
    let level = stackLevel[depth];
    let cellColumn = i32(stackColumn[depth]);
    let cellRow = i32(stackRow[depth]);
    let blockSize = i32(pyramidLevelBlockSize(level));
    let x0 = cellColumn * blockSize;
    let y0 = cellRow * blockSize;
    let x1 = min(x0 + blockSize, i32(WIDTH)) - 1;
    let y1 = min(y0 + blockSize, i32(HEIGHT)) - 1;
    let ya = max(y0, rowLow);
    let yb = min(y1, rowHigh);
    if (ya > yb) { continue; }
    let dxLow = x0 - column;
    let dxHigh = x1 - column;
    let dyLow = ya - row;
    let dyHigh = yb - row;
    // Block pixel nearest the centre: disjoint from the disc when it is outside.
    var nearX = 0;
    if (dxLow > 0) { nearX = dxLow; } else if (dxHigh < 0) { nearX = dxHigh; }
    var nearY = 0;
    if (dyLow > 0) { nearY = dyLow; } else if (dyHigh < 0) { nearY = dyHigh; }
    if (!isOffsetInDisc(nearX, nearY, cell, radiusSquared)) { continue; }
    let farX = max(abs(dxLow), abs(dxHigh));
    let farY = max(abs(dyLow), abs(dyHigh));
    if (!exact && ya == y0 && yb == y1 && isOffsetInDisc(farX, farY, cell, radiusSquared)) {
      let blockMaximum = pyramid[pyramidOffset + pyramidLevelIndex(level, u32(x0), u32(y0))];
      if (beatsCentre(blockMaximum, ownHeight, hasSmallerIndex)) { return false; }
      continue;
    }
    if (level == 0u) {
      for (var y = ya; y <= yb; y++) {
        for (var x = x0; x <= x1; x++) {
          if (!isOffsetInDisc(x - column, y - row, cell, radiusSquared)) { continue; }
          let pixelIndex = u32(y) * WIDTH + u32(x);
          if (elevationValidity[elevationValidityOffset + pixelIndex] == 0u) {
            if (REJECT_INCOMPLETE) { return false; }
            continue;
          }
          if (beatsCentre(elevationValues[elevationValuesOffset + pixelIndex], ownHeight, hasSmallerIndex)) {
            return false;
          }
        }
      }
      continue;
    }
    let childSize = i32(pyramidLevelBlockSize(level - 1u));
    for (var child = 0; child < 4; child++) {
      let childColumn = 2 * cellColumn + (child & 1);
      let childRow = 2 * cellRow + (child >> 1);
      if (childColumn * childSize >= i32(WIDTH) || childRow * childSize >= i32(HEIGHT)) { continue; }
      stackLevel[depth] = level - 1u;
      stackColumn[depth] = u32(childColumn);
      stackRow[depth] = u32(childRow);
      depth++;
    }
  }
  return true;
}
// Returns whether the pixel is a summit and leaves its drop in summitDrop.
fn evaluateSummit(column: i32, row: i32, index: u32) -> bool {
  let cell = getGroundCellSize(u32(row));
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0)) {
    return false;
  }
  let requested = settings[settingsOffset];
  let minimumDrop = settings[settingsOffset + 1u];
  if (!(isFiniteValue(requested) && requested > 0.0)) { return false; }
  ${n.reportsClamp?`// The clamp depends only on the row's cell size: one pixel per row reports it, so at most
  // HEIGHT invocations touch the flag word instead of every pixel (review C-17).
  if (column == 0 && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }`:``}
  if (elevationValidity[elevationValidityOffset + index] == 0u) { return false; }
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let centre = vec2<f32>(f32(column), f32(row));
  let ownHeight = elevationValues[elevationValuesOffset + index];
  // Cheap rejection first: almost every pixel has a higher disc pixel among its 8 neighbors, so the
  // search below only runs for 3x3 local maxima. Every rejection here would also be reached by
  // the full search (higher pixel, or an incomplete neighborhood under 'reject'), so results match.
  for (var nearY = -1; nearY <= 1; nearY++) {
    for (var nearX = -1; nearX <= 1; nearX++) {
      let nearPixel = vec2<i32>(column + nearX, row + nearY);
      if ((nearX == 0 && nearY == 0) || !isInDisc(nearPixel, centre, cell, radiusSquared)) { continue; }
      if (nearPixel.x < 0 || nearPixel.y < 0 || nearPixel.x >= i32(WIDTH) || nearPixel.y >= i32(HEIGHT)) { continue; }
      let nearIndex = u32(nearPixel.y) * WIDTH + u32(nearPixel.x);
      if (elevationValidity[elevationValidityOffset + nearIndex] == 0u) { continue; }
      let nearHeight = elevationValues[elevationValuesOffset + nearIndex];
      if (nearHeight > ownHeight || (nearHeight == ownHeight && nearIndex < index)) { return false; }
    }
  }
  let ownRowHalfWidth = getDiscRowHalfWidth(0, cell, radiusSquared);
  let columnHalfHeight = getDiscColumnHalfHeight(cell, radiusSquared);
  // The widest row is dy = 0 and the tallest column is dx = 0, so these decide grid containment.
  if (REJECT_INCOMPLETE &&
      (column - ownRowHalfWidth < 0 || column + ownRowHalfWidth >= i32(WIDTH) ||
       row - columnHalfHeight < 0 || row + columnHalfHeight >= i32(HEIGHT))) {
    return false;
  }
  // Heights of FLT_MAX collide with the empty/invalid sentinels of block maxima.
  let exact = abs(ownHeight) >= FLOAT_MAXIMUM;
  // Own row: every |dx| <= ownRowHalfWidth is in the disc.
  for (var offsetX = -ownRowHalfWidth; offsetX <= ownRowHalfWidth; offsetX++) {
    let pixelColumn = column + offsetX;
    if (offsetX == 0 || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
    let neighborIndex = index + u32(offsetX) ;
    if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) {
      if (REJECT_INCOMPLETE) { return false; }
      continue;
    }
    if (beatsCentre(elevationValues[elevationValuesOffset + neighborIndex], ownHeight, offsetX < 0)) {
      return false;
    }
  }
  let lastRow = i32(HEIGHT) - 1;
  if (!isHalfDiscDominated(column, row, max(row - columnHalfHeight, 0), row - 1, ownHeight, true, exact, cell, radiusSquared) ||
      !isHalfDiscDominated(column, row, row + 1, min(row + columnHalfHeight, lastRow), ownHeight, false, exact, cell, radiusSquared)) {
    return false;
  }
  var ringHeight = 0.0;
  var hasRing = false;
  var previousHalfWidth = -1;
  var currentHalfWidth = getDiscRowHalfWidth(-columnHalfHeight, cell, radiusSquared);
  for (var offsetY = -columnHalfHeight; offsetY <= columnHalfHeight; offsetY++) {
    var nextHalfWidth = -1;
    if (offsetY < columnHalfHeight) { nextHalfWidth = getDiscRowHalfWidth(offsetY + 1, cell, radiusSquared); }
    let pixelRow = row + offsetY;
    if (pixelRow >= 0 && pixelRow <= lastRow) {
      let interiorHalfWidth = min(previousHalfWidth, min(currentHalfWidth, nextHalfWidth));
      // Interior pixels (all eight neighbors in the disc) have |dx| <= interiorHalfWidth - 1.
      let hasInterior = interiorHalfWidth > 0;
      for (var run = 0; run < 2; run++) {
        if (run == 1 && !hasInterior) { break; }
        var runStart = -currentHalfWidth;
        var runEnd = currentHalfWidth;
        if (hasInterior) {
          if (run == 0) { runEnd = -interiorHalfWidth; } else { runStart = interiorHalfWidth; }
        }
        for (var offsetX = runStart; offsetX <= runEnd; offsetX++) {
          let pixelColumn = column + offsetX;
          if ((offsetX == 0 && offsetY == 0) || pixelColumn < 0 || pixelColumn >= i32(WIDTH)) { continue; }
          let neighborIndex = u32(pixelRow) * WIDTH + u32(pixelColumn);
          if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { continue; }
          let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
          if (!hasRing || neighborHeight > ringHeight) {
            ringHeight = neighborHeight;
            hasRing = true;
          }
        }
      }
    }
    previousHalfWidth = currentHalfWidth;
    currentHalfWidth = nextHalfWidth;
  }
  if (hasRing) {
    let dropValue = ownHeight - ringHeight;
    if (dropValue < minimumDrop) { return false; }
    summitDrop = dropValue;
  } else {
    summitDrop = getInfinity(index);
  }
  return true;
}`}var ut={unchanged:0,snapped:1,onRing:2,rejectedMove:3,rejectedHeight:4,noData:5,outside:6};function dt(e){return f(e,20)}var ft={cellSizeIndex:3,northEdgeIndex:5};function pt(e,t,n=new Float32Array(8)){let r=`getGPUTerrainPeakSnapParameterValues`;le(r,n,8);let i=e.radius??150,a=e.maximumMove??300,o=e.maximumHeightChange??80;return X(r,`radius`,i,0,!0),X(r,`maximumMove`,a,0),X(r,`maximumHeightChange`,o,0),Qe(r,pe(t).cellSize),n[0]=i,n[1]=a,n[2]=o,n[7]=e.interior===!1?0:1,me(n,t,ft),n}var mt=class{id;props;constructor(e){this.id=e.id??`terrain-peak-snap`,this.props=e;let{id:t}=this;if(e.candidateVerticalDatum!==void 0&&!e.candidateHeights)throw Error(`${t} candidateVerticalDatum needs candidateHeights`);r(t,e.candidateVerticalDatum,e.elevation.verticalDatum,`candidate`);let{output:n}=e,i=n.status?.radiusClamped;k(t,e.grid),j(t,e.cellSizeMode??`uniform`),ce(t,e.grid,e.cellSizeMode),Y(t,e.maximumRadiusPixels),g(e.candidates,[`float32x2`],`${t} candidates`);let a=e.candidates.length;if(a<1)throw Error(`${t} candidates must contain at least one row`);fe(t,e.parameters,8);for(let[r,i]of[[`candidateHeights`,e.candidateHeights],[`candidateRadii`,e.candidateRadii],[`output.heights`,n.heights],[`output.snapDistance`,n.snapDistance]])if(i&&(g(i,[`float32`],`${t} ${r}`),i.length!==a))throw Error(`${t} ${r} must contain one value per candidate`);if(g(n.positions,[`float32x2`],`${t} output.positions`),n.positions.length!==a)throw Error(`${t} output.positions must contain one value per candidate`);if(g(n.snapStatus,[`uint32`],`${t} output.snapStatus`),n.snapStatus.length!==a)throw Error(`${t} output.snapStatus must contain one value per candidate`);if(i&&(g(i,[`uint32`],`${t} output.status.radiusClamped`),i.length<1))throw Error(`${t} output.status.radiusClamped must contain one row`);ue(t,[n.positions,n.heights,n.snapStatus,n.snapDistance,i],[...D(e.elevation),e.parameters,e.candidates,e.candidateHeights,e.candidateRadii])}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:a}=n.grid,o=r.status?.radiusClamped;E(t,e,n.elevation,[]),s(t,e,[n.parameters,n.candidates,n.candidateHeights,n.candidateRadii,r.positions,r.heights,r.snapStatus,r.snapDistance,o]);let c=n.candidates.length;m(t,`candidates rows`,c,dt(e.device));let l=n.cellSizeMode??`uniform`,u=Y(t,n.maximumRadiusPixels),d=O(e,t,n.elevation,n.grid,!0),f=[...d.nodes],p=d.band.validity,g=h(e,`${t}-results`,`uint32`,c*5);o&&f.push(y(e,{id:`${t}-overflow-clear`,operation:`GPUTerrainPeakSnap`,view:o,type:`u32`,value:`0u`,componentCount:1}));let _=[{name:`elevationValues`,view:d.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:p,type:`u32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`candidates`,view:n.candidates,type:`f32`,access:`read`},...n.candidateHeights?[{name:`candidateHeights`,view:n.candidateHeights,type:`f32`,access:`read`}]:[],...n.candidateRadii?[{name:`candidateRadii`,view:n.candidateRadii,type:`f32`,access:`read`}]:[],{name:`results`,view:g,type:`u32`,access:`read_write`},...o?[{name:`overflowValues`,view:o,type:`atomic<u32>`,access:`read_write`}]:[]],v=ut;return f.push(b(e,{id:`${t}-search`,operation:`GPUTerrainPeakSnap`,variant:`disc-${l}`,bindings:_,invocationCount:c*256,workgroupSize:256,guardIndex:!1,declarations:`const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
const CANDIDATE_COUNT: u32 = ${c}u;
var<workgroup> reduceIndex: array<u32, 256>;
var<workgroup> reduceHeight: array<f32, 256>;
// True when (height, index) beats the incumbent: higher first, lower index on ties, any pixel beats none.
fn isBetterPixel(height: f32, pixelIndex: u32, incumbentHeight: f32, incumbentIndex: u32) -> bool {
  if (pixelIndex == NO_INDEX) { return false; }
  return incumbentIndex == NO_INDEX || height > incumbentHeight ||
    (height == incumbentHeight && pixelIndex < incumbentIndex);
}
const MAXIMUM_RADIUS_PIXELS: i32 = ${u};
const NO_INDEX: u32 = 0xffffffffu;
const STATUS_UNCHANGED: u32 = ${v.unchanged}u;
const STATUS_SNAPPED: u32 = ${v.snapped}u;
const STATUS_ON_RING: u32 = ${v.onRing}u;
const STATUS_REJECTED_MOVE: u32 = ${v.rejectedMove}u;
const STATUS_REJECTED_HEIGHT: u32 = ${v.rejectedHeight}u;
const STATUS_NO_DATA: u32 = ${v.noData}u;
const STATUS_OUTSIDE: u32 = ${v.outside}u;
${$e}
${de}
${A(l,ft)}
${et}
var<private> snapStatus: u32;
var<private> snapPosition: vec2<f32>;
var<private> snapHeight: f32;
var<private> snapDistanceValue: f32;
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear DEM at an in-grid position; NaN unless all four surrounding pixels are valid.
fn sampleElevation(position: vec2<f32>, seed: u32) -> f32 {
  let base = vec2<u32>(floor(position));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = position - floor(position);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return getNan(seed);
  }
  let top = mix(getElevation(base.x, base.y), getElevation(next.x, base.y), fraction.x);
  let bottom = mix(getElevation(base.x, next.y), getElevation(next.x, next.y), fraction.x);
  return mix(top, bottom, fraction.y);
}
var<private> snapNearest: vec2<i32>;
var<private> snapCell: vec2<f32>;
var<private> snapRadiusSquared: f32;
var<private> snapExtent: vec2<i32>;
var<private> snapDemHeight: f32;
// Validates the candidate and sets up the disc search. Returns false when snapStatus is final.
fn prepareSnap(index: u32, candidate: vec2<f32>, isLeader: bool) -> bool {
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getNan(index);
  snapDistanceValue = 0.0;
  if (!(isFiniteValue(candidate.x) && isFiniteValue(candidate.y) &&
      candidate.x >= 0.0 && candidate.y >= 0.0 &&
      candidate.x <= f32(WIDTH - 1u) && candidate.y <= f32(HEIGHT - 1u))) {
    return false;
  }
  let demHeight = sampleElevation(candidate, index);
  snapHeight = demHeight;
  snapStatus = STATUS_NO_DATA;
  let nearest = vec2<i32>(i32(floor(candidate.x + 0.5)), i32(floor(candidate.y + 0.5)));
  let cell = getGroundCellSize(u32(nearest.y));
  var requested = settings[settingsOffset];
  ${n.candidateRadii?`let candidateRadius = candidateRadii[candidateRadiiOffset + index];
  if (isFiniteValue(candidateRadius) && candidateRadius > 0.0) { requested = candidateRadius; }`:``}
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0 &&
      isFiniteValue(requested) && requested > 0.0)) {
    return false;
  }
  ${o?`if (isLeader && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }`:``}
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let extentX = getDiscExtent(radius, cell.x);
  let extentY = getDiscExtent(radius, cell.y);
  snapNearest = nearest;
  snapCell = cell;
  snapRadiusSquared = radiusSquared;
  snapExtent = vec2<i32>(extentX, extentY);
  snapDemHeight = demHeight;
  return true;
}
// Applies the move, height, and ring policies to the disc maximum found by the cooperative scan.
fn finishSnap(index: u32, candidate: vec2<f32>, bestIndex: u32, bestHeight: f32) {
  let nearest = snapNearest;
  let cell = snapCell;
  let radiusSquared = snapRadiusSquared;
  let demHeight = snapDemHeight;
  if (bestIndex == NO_INDEX) { return; }
  let bestPixel = vec2<i32>(i32(bestIndex % WIDTH), i32(bestIndex / WIDTH));
  if (settings[settingsOffset + 7u] != 0.0 &&
      isOnDiscRing(bestPixel, candidate, cell, radiusSquared)) {
    snapStatus = STATUS_ON_RING;
    return;
  }
  if (bestPixel.x == nearest.x && bestPixel.y == nearest.y) {
    snapStatus = STATUS_UNCHANGED;
    return;
  }
  let moveX = (f32(bestPixel.x) - candidate.x) * cell.x;
  let moveY = (f32(bestPixel.y) - candidate.y) * cell.y;
  let moveSquared = moveX * moveX + moveY * moveY;
  let maximumMove = settings[settingsOffset + 1u];
  if (moveSquared > maximumMove * maximumMove) {
    snapStatus = STATUS_REJECTED_MOVE;
    return;
  }
  var reference = demHeight;
  ${n.candidateHeights?`let candidateHeight = candidateHeights[candidateHeightsOffset + index];
  if (isFiniteValue(candidateHeight)) { reference = candidateHeight; }`:``}
  if (isFiniteValue(reference) &&
      abs(bestHeight - reference) > settings[settingsOffset + 2u]) {
    snapStatus = STATUS_REJECTED_HEIGHT;
    return;
  }
  snapStatus = STATUS_SNAPPED;
  snapPosition = vec2<f32>(f32(bestPixel.x), f32(bestPixel.y));
  snapHeight = bestHeight;
  snapDistanceValue = sqrt(moveSquared);
}`,body:`let candidateIndex = workgroupIndex;
  let isLeader = localInvocationIndex == 0u;
  let inRange = candidateIndex < CANDIDATE_COUNT;
  var candidate = vec2<f32>(0.0, 0.0);
  if (inRange) {
    candidate = vec2<f32>(candidates[candidatesOffset + 2u * candidateIndex], candidates[candidatesOffset + 2u * candidateIndex + 1u]);
  }
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getNan(candidateIndex);
  snapDistanceValue = 0.0;
  let prepared = inRange && prepareSnap(candidateIndex, candidate, isLeader);
  // The whole workgroup scans one candidate's disc; the (height, -index) maximum is order independent.
  var bestIndex = NO_INDEX;
  var bestHeight = 0.0;
  if (prepared) {
    let extent = snapExtent;
    let spanX = u32(2 * extent.x + 1);
    let scanCount = spanX * u32(2 * extent.y + 1);
    for (var scan = localInvocationIndex; scan < scanCount; scan += 256u) {
      let offsetX = i32(scan % spanX) - extent.x;
      let offsetY = i32(scan / spanX) - extent.y;
      let pixel = snapNearest + vec2<i32>(offsetX, offsetY);
      if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(WIDTH) || pixel.y >= i32(HEIGHT)) { continue; }
      if (!isInDisc(pixel, candidate, snapCell, snapRadiusSquared)) { continue; }
      if (!isValidPixel(u32(pixel.x), u32(pixel.y))) { continue; }
      let pixelIndex = u32(pixel.y) * WIDTH + u32(pixel.x);
      let pixelHeight = elevationValues[elevationValuesOffset + pixelIndex];
      if (isBetterPixel(pixelHeight, pixelIndex, bestHeight, bestIndex)) {
        bestIndex = pixelIndex;
        bestHeight = pixelHeight;
      }
    }
  }
  reduceIndex[localInvocationIndex] = bestIndex;
  reduceHeight[localInvocationIndex] = bestHeight;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      let otherIndex = reduceIndex[localInvocationIndex + stride];
      let otherHeight = reduceHeight[localInvocationIndex + stride];
      if (isBetterPixel(otherHeight, otherIndex, reduceHeight[localInvocationIndex], reduceIndex[localInvocationIndex])) {
        reduceIndex[localInvocationIndex] = otherIndex;
        reduceHeight[localInvocationIndex] = otherHeight;
      }
    }
    workgroupBarrier();
  }
  if (isLeader && inRange) {
    if (prepared && reduceIndex[0] != NO_INDEX) {
      finishSnap(candidateIndex, candidate, reduceIndex[0], reduceHeight[0]);
    }
    let resultBase = resultsOffset + 5u * candidateIndex;
    results[resultBase] = snapStatus;
    results[resultBase + 1u] = bitcast<u32>(snapPosition.x);
    results[resultBase + 2u] = bitcast<u32>(snapPosition.y);
    results[resultBase + 3u] = bitcast<u32>(snapHeight);
    results[resultBase + 4u] = bitcast<u32>(snapDistanceValue);
  }`}),b(e,{id:`${t}-unpack`,operation:`GPUTerrainPeakSnap`,variant:`unpack`,bindings:[{name:`results`,view:g,type:`u32`,access:`read`},{name:`positionValues`,view:r.positions,type:`f32`,access:`read_write`},{name:`heightValues`,view:r.heights,type:`u32`,access:`read_write`},{name:`statusValues`,view:r.snapStatus,type:`u32`,access:`read_write`},...r.snapDistance?[{name:`distanceValues`,view:r.snapDistance,type:`u32`,access:`read_write`}]:[]],invocationCount:c,body:`let resultBase = resultsOffset + 5u * index;
  statusValues[statusValuesOffset + index] = results[resultBase];
  positionValues[positionValuesOffset + 2u * index] = bitcast<f32>(results[resultBase + 1u]);
  positionValues[positionValuesOffset + 2u * index + 1u] = bitcast<f32>(results[resultBase + 2u]);
  heightValues[heightValuesOffset + index] = results[resultBase + 3u];
  ${r.snapDistance?`distanceValues[distanceValuesOffset + index] = results[resultBase + 4u];`:``}`})),f}},Z={regular:0,peak:1,pit:2,saddle:3,boundary:4,noData:5},ht=class{id;props;constructor(e){this.id=e.id??`terrain-critical-points`,this.props=e;let{id:t}=this,{output:n}=e,r=k(t,e.grid);if(e.connectivity!==void 0&&e.connectivity!==8&&e.connectivity!==6)throw Error(`${t} connectivity must be 8 or 6`);if(g(n.classes,[`uint32`],`${t} output.classes`),n.classes.length!==r)throw Error(`${t} output.classes must contain one value per pixel`);if(n.signChanges&&(g(n.signChanges,[`uint32`],`${t} output.signChanges`),n.signChanges.length!==r))throw Error(`${t} output.signChanges must contain one value per pixel`);if(n.counts&&(g(n.counts,[`uint32`],`${t} output.counts`),n.counts.length!==6))throw Error(`${t} output.counts must contain one value per class (6)`);ue(t,[n.classes,n.signChanges,n.counts],D(e.elevation))}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:a}=n.grid;E(t,e,n.elevation,[]),s(t,e,[r.classes,r.signChanges,r.counts]);let o=O(e,t,n.elevation,n.grid,!0),c=[...o.nodes];r.counts&&c.push(y(e,{id:`${t}-clear-counts`,operation:`GPUTerrainCriticalPoints`,view:r.counts,type:`u32`,value:`0u`}));let l=(n.connectivity??8)===8?gt:_t,u=[{name:`elevationValues`,view:o.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:o.band.validity,type:`u32`,access:`read`},{name:`classValues`,view:r.classes,type:`u32`,access:`read_write`},...r.signChanges?[{name:`signChangeValues`,view:r.signChanges,type:`u32`,access:`read_write`}]:[],...r.counts?[{name:`classCounts`,view:r.counts,type:`atomic<u32>`,access:`read_write`}]:[]],d=l.map(([e,t],n)=>`let side${n} = getSide(column, row, ${e}, ${t}, centerHeight, index);`).join(`
  `),f=l.map((e,t)=>`side${t} == 2u`).join(` || `),p=l.map((e,t)=>`select(0u, 1u, side${t} != side${(t+1)%l.length})`).join(` +
    `),m=l.map((e,t)=>`side${t}`).join(` + `);return c.push(b(e,{id:`${t}-classify`,operation:`GPUTerrainCriticalPoints`,variant:`ring-${l.length}`,bindings:u,invocationCount:i*a,declarations:`const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
const RING_COUNT: u32 = ${l.length}u;
const REGULAR: u32 = ${Z.regular}u;
const PEAK: u32 = ${Z.peak}u;
const PIT: u32 = ${Z.pit}u;
const SADDLE: u32 = ${Z.saddle}u;
const BOUNDARY: u32 = ${Z.boundary}u;
const NO_DATA: u32 = ${Z.noData}u;
// 1 when the neighbor is higher (simulation of simplicity), 0 when lower, 2 when it is invalid.
// The neighbor is always inside the grid because edge pixels return earlier.
fn getSide(column: u32, row: u32, columnOffset: i32, rowOffset: i32, centerHeight: f32, centerIndex: u32) -> u32 {
  let neighborIndex = u32(i32(row) + rowOffset) * WIDTH + u32(i32(column) + columnOffset);
  if (elevationValidity[elevationValidityOffset + neighborIndex] == 0u) { return 2u; }
  let neighborHeight = elevationValues[elevationValuesOffset + neighborIndex];
  return select(0u, 1u, neighborHeight > centerHeight || (neighborHeight == centerHeight && neighborIndex > centerIndex));
}`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  var classCode = NO_DATA;
  var changeCount = 0u;
  if (elevationValidity[elevationValidityOffset + index] != 0u) {
    if (column == 0u || row == 0u || column == WIDTH - 1u || row == HEIGHT - 1u) {
      classCode = BOUNDARY;
    } else {
      let centerHeight = elevationValues[elevationValuesOffset + index];
      ${d}
      if (${f}) {
        classCode = BOUNDARY;
      } else {
        changeCount = ${p};
        let higherCount = ${m};
        classCode = select(SADDLE, REGULAR, changeCount == 2u);
        if (changeCount == 0u) {
          classCode = select(PEAK, PIT, higherCount == RING_COUNT);
        }
      }
    }
  }
  classValues[classValuesOffset + index] = classCode;
  ${r.signChanges?`signChangeValues[signChangeValuesOffset + index] = changeCount;`:``}
  ${r.counts?`atomicAdd(&classCounts[classCountsOffset + classCode], 1u);`:``}`})),c}},gt=[[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1]],_t=[[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]];function vt(e,t,n,r){let i=F(e,4),a=Math.min(i[0],t),o=F(e.slice(16),t),s=we(e.slice(16+t*4),t),c=[];for(let e=0;e<a;e++){let t=o[e];c.push({column:t%n,row:Math.floor(t/n),elevation:r[t],drop:s[e]})}return{candidates:c,total:i[1],listOverflow:i[2]!==0||i[1]>t,radiusClamped:i[3]!==0}}function yt(e,t){let n=[],r=[];for(let i of e)(i.drop>=t?n:r).push(i);return{survivors:n,rejected:r}}function bt(e){return L.breaks.filter(t=>e>=t).length}function xt(e,t,n){let r=Array(n).fill(0);for(let{drop:i}of e)Number.isFinite(i)&&r[Math.min(n-1,Math.max(0,Math.floor(i/t)))]++;return r}var St={"Ober Gabelhorn":`Obergabelhorn`,"Unter Gabelhorn":`Untergabelhorn`,"Mittler Gabelhorn":`Mittlergabelhorn`};function Q(e){let[t,n]=e.split(` / `),r=n?`${t.split(` `)[0]} ${n}`:e;return St[r]??r}function Ct(e,t){let n=[];for(let r of e.peaks){if(!r.name)continue;let[e,i]=t.getPixel(r.lngLat[0],r.lngLat[1]);e<1||i<1||e>t.width-2||i>t.height-2||n.push({name:Q(r.name),elevationMeters:r.elevationMeters,lngLat:r.lngLat,column:e,row:i,osmId:r.osmId})}return n.sort((e,t)=>(t.elevationMeters??0)-(e.elevationMeters??0))}function wt(e,t,n){let r=new Map,i=150/n;for(let n of t){let t=-1,a=i;for(let r=0;r<e.length;r++){let i=Math.hypot(e[r].column-n.column,e[r].row-n.row);i<=a&&(t=r,a=i)}if(t<0)continue;let o=r.get(t);(!o||a<o.distance)&&r.set(t,{peak:n,distance:a})}return new Map([...r].map(([e,{peak:t}])=>[e,t]))}function Tt(e,t,n){let r=Object.fromEntries(Re.map(e=>[e,0])),i=[];for(let a=0;a<n;a++){let n=Fe(e[a]);n&&(r[n]++,n===`snapped`&&i.push(t[a]))}i.sort((e,t)=>e-t);let a=Math.floor(i.length/2);return{counts:r,medianMoveMeters:i.length?i.length%2?i[a]:(i[a-1]+i[a])/2:null}}var Et=3e4;function Dt(e,t,n){return{ids:e.createBuffer(`${t}-ids`,n*4),drops:e.createBuffer(`${t}-drops`,n*4),count:e.createBuffer(`${t}-count`,4),total:e.createBuffer(`${t}-total`,4),overflow:e.createBuffer(`${t}-overflow`,4),clamped:e.createBuffer(`${t}-clamped`,4)}}async function Ot(e,t){let{buffers:n,capacity:r}=t,i=new o(e.device,{id:t.name});return i.add(new it({id:`summits`,grid:t.grid,elevation:I(i,`summits`,t.elevation,t.validity,t.pixelCount),cellSizeMode:`web-mercator`,maximumRadiusPixels:t.maximumRadiusPixels,incompleteNeighborhood:t.incompleteNeighborhood,parameters:t.settings.importToGraph(i),output:{summits:{ids:N(i,`ids`,n.ids,`uint32`,r),status:{count:N(i,`count`,n.count,`uint32`,1),overflow:N(i,`overflow`,n.overflow,`uint32`,1),requiredCount:N(i,`total`,n.total,`uint32`,1)}},summitDrop:N(i,`drops`,n.drops,`float32`,r),status:{radiusClamped:N(i,`clamped`,n.clamped,`uint32`,1)}}})),e.track(await i.compileAsync())}function kt(e){return{candidates:e.createBuffer(`candidates`,512),candidateHeights:e.createBuffer(`candidate-heights`,256),candidateRadii:e.createBuffer(`candidate-radii`,256),positions:e.createBuffer(`snap-positions`,512),heights:e.createBuffer(`snap-heights`,256),status:e.createBuffer(`snap-status`,256),distance:e.createBuffer(`snap-distance`,256),overflow:e.createBuffer(`snap-overflow`,4)}}async function At(e,t){let{buffers:n}=t,r=new o(e.device,{id:`summits-snap`});return r.add(new mt({id:`snap`,grid:t.grid,elevation:I(r,`snap`,t.elevation,t.validity,t.pixelCount),cellSizeMode:`web-mercator`,maximumRadiusPixels:16,candidates:N(r,`candidates`,n.candidates,`float32x2`,64),candidateHeights:N(r,`candidate-heights`,n.candidateHeights,`float32`,64),candidateRadii:N(r,`candidate-radii`,n.candidateRadii,`float32`,64),parameters:t.settings.importToGraph(r),output:{positions:N(r,`positions`,n.positions,`float32x2`,64),heights:N(r,`heights`,n.heights,`float32`,64),snapStatus:N(r,`status`,n.status,`uint32`,64),snapDistance:N(r,`distance`,n.distance,`float32`,64),status:{radiusClamped:N(r,`overflow`,n.overflow,`uint32`,1)}}})),e.track(await r.compileAsync())}async function jt(e,t){let n=new o(e.device,{id:`summits-critical-${t.connectivity}`});return n.add(new ht({id:`critical`,grid:t.grid,elevation:I(n,`critical`,t.elevation,t.validity,t.pixelCount),connectivity:t.connectivity,output:{classes:N(n,`classes`,t.classes,`uint32`,t.pixelCount),signChanges:N(n,`signs`,t.signChanges,`uint32`,t.pixelCount),counts:N(n,`counts`,t.counts,`uint32`,6)}})),e.track(await n.compileAsync())}function Mt(e){let t=[],n=[];for(let r=0;r<40;r++)t.push(e.createBuffer(`contour-vertices-${r}`,Et*16)),n.push(e.createBuffer(`contour-count-${r}`,4));return{levels:e.createParameterBuffer(`levels`,`float32`,40),overflow:e.createBuffer(`contour-overflow`,4),vertices:t,counts:n,drawCommands:e.track(new te(e.device,{id:`summits-contour-draw`,type:`draw`,commands:Array.from({length:40},()=>({vertexCount:6,instanceCount:0}))}))}}async function Nt(e,t){let{buffers:n}=t,r=new o(e.device,{id:`summits-contours`}),i=n.levels.importToGraph(r),a=n.drawCommands.importToGraph(r);return r.add(new G({id:`contours`,grid:t.grid,elevation:I(r,`contours`,t.elevation,t.validity,t.pixelCount),output:{status:{overflow:N(r,`overflow`,n.overflow,`uint32`,1)}},levels:Array.from({length:40},(e,t)=>({level:r.createDataView(i.buffer,{format:`float32`,length:1,byteOffset:t*4}),vertices:N(r,`vertices-${t}`,n.vertices[t],`float32x2`,Et*2),status:{count:N(r,`count-${t}`,n.counts[t],`uint32`,1)},draw:a,drawCommandIndex:t,drawLayout:`instanced`,verticesPerInstance:6,capacity:Et}))})),e.track(await r.compileAsync())}function Pt(e){let{candidate:t,peak:n,radiusMeters:r,colors:i}=e;return{title:n?.name??`Summit`,subtitle:`Highest within ${z(r)} m`,rows:[{label:`Drop to the ring`,value:z(t.drop),unit:`m`,swatch:i.summit,emphasis:!0},{label:`DEM cell, analysis grid`,value:z(t.elevation),unit:`m`},...n?.elevationMeters?[{label:`Published elevation`,value:z(n.elevationMeters),unit:`m`}]:[]],note:`A lower bound of prominence: the way out may drop further.`}}function Ft(e){let{candidate:t,minimumDrop:n,peak:r,colors:i}=e;return{title:r?.name??`Highest in its disc, not a summit`,subtitle:`Fails the drop test`,rows:[{label:`Drop to the ring`,value:z(t.drop),unit:`m`,swatch:i.rejected,emphasis:!0},{label:`Needed`,value:z(n),unit:`m`},{label:`DEM cell, analysis grid`,value:z(t.elevation),unit:`m`}]}}function It(e){let{test:t,elevation:n,radiusMeters:r,minimumDrop:i,center:a,colors:o}=e;return{title:t.isMax&&t.drop>=i?`A summit`:t.isMax?`Highest, but the drop is too small`:`Not a summit`,subtitle:`Disc of ${z(r)} m around this cell`,rows:[{label:`Highest in disc`,value:t.isMax?`yes`:`no`,swatch:t.isMax?o.summit:o.rejected,emphasis:!0},{label:`This cell`,value:z(n),unit:`m`},...Number.isFinite(t.ringMax)?[{label:`Highest ring cell`,value:z(t.ringMax),unit:`m`}]:[],...t.isMax&&Number.isFinite(t.drop)?[{label:`Drop to the ring`,value:z(t.drop),unit:`m`}]:[],...!t.isMax&&e.distanceToHigherMeters!==null?[{label:`Higher ground`,value:z(e.distanceToHigherMeters),unit:`m away`}]:[]],...t.incomplete?{note:`The disc leaves the DEM here.`}:{},highlight:{kind:`circle`,coordinate:a,radiusMeters:r}}}function Lt(e){let{peak:t,status:n,ground:r}=e,i=Oe.find(e=>e.id===n)?.label??n;return{title:t.name,subtitle:`OpenStreetMap peak`,rows:[{label:`Snap result`,value:i,swatch:ke(n,r),emphasis:!0},{label:`Catalogue elevation`,value:t.elevationMeters===null?`none`:z(t.elevationMeters),...t.elevationMeters===null?{}:{unit:`m`}},{label:`DEM cell, ${e.analysisCellMeters.toFixed(1)} m grid`,value:Number.isFinite(e.analysisCellHeight)?z(e.analysisCellHeight):`none`,...Number.isFinite(e.analysisCellHeight)?{unit:`m`}:{}},...e.fullCellHeight===null?[]:[{label:`DEM cell, ${e.fullCellMeters.toFixed(1)} m grid`,value:z(e.fullCellHeight),unit:`m`}],{label:`Moved by the snap`,value:z(e.moveMeters),unit:`m`},...e.errorMeters>0?[{label:`Error added`,value:z(e.errorMeters),unit:`m`}]:[]],note:`The published height and the DEM cell differ: cells average the summit with its flanks.`}}function Rt(e){return{title:`Elevation`,rows:[{label:`DEM cell`,value:z(e.elevation),unit:`m`,emphasis:!0},{label:`Cell size`,value:e.cellMeters.toFixed(1),unit:`m`}]}}var zt=4,$=65536,Bt=25,Vt=20,Ht=5,Ut=6,Wt=12,Gt=3.5;async function Kt(e){let{device:t}=e;e.setStatus(`Loading the DEM and its OpenStreetMap context`);let n=e.datasets.get(`alps-dem`),r=await Ae(e.datasets.get(`alps-context`),e.signal);e.signal.throwIfAborted();let i=Ce(n,1),a=Ce(n,zt),o=Te(i),s=Ee(Te(a)),c=Ct(r,a),l=Me(r.glacierGeoJson,o),u=Ne({dem:o,device:t,ground:e.ground(),glacierMask:l});e.setStatus(`Building the shaded relief`),await u.prepare(),e.signal.aborted&&(u.destroy(),e.signal.throwIfAborted());let{width:d,height:f,pixelCount:m}=a,h=[i.origin[0],i.origin[1],0],g=a.groundCellSize,_=new p(t,`summits`),v=!1,ee=_.createBuffer(`full-elevation`,i.elevation),te=_.createBuffer(`full-validity`,i.validity),y=_.createBuffer(`coarse-elevation`,a.elevation),ne=_.createBuffer(`coarse-validity`,a.validity),b=_.createParameterBuffer(`summit-settings`,`float32`,8),x=_.createParameterBuffer(`sweep-settings`,`float32`,8),re=_.createParameterBuffer(`snap-settings`,`float32`,8),S=Dt(_,`sweep`,16),C=kt(_),w=Mt(_),ie=Array.from({length:L.sizesPixels.length},(e,t)=>_.createBuffer(`summit-class-${t}`,R*8)),T=_.createBuffer(`rejected`,R*8),ae=_.createBuffer(`original-markers`,512),oe=_.createBuffer(`snapped-markers`,512),se=_.createBuffer(`snap-segments`,1024),E=_.createBuffer(`critical-classes`,m*4),ce=_.createBuffer(`critical-signs`,m*4),le=_.createBuffer(`critical-counts`,24),D=_.createBuffer(`critical-peaks`,$*8),ue=_.createBuffer(`critical-saddles`,$*8),O=_.createBuffer(`critical-pits`,$*8),k=_.createBuffer(`edge-band-triangles`,192),de=_.createBuffer(`edge-band-features`,new Uint32Array(24)),fe=_.createBuffer(`edge-band-values`,new Uint32Array(1)),A=()=>Number(e.options.summitMaximumRadius),j=new Map,pe=new Map,me=null,N=e=>j.get(`${A()}|${e}`);async function I(e){let t=A(),n=`${t}|${e}`,r=j.get(n);if(r)return r;let i=Dt(_,`variant-${n}`,R),o={key:n,mode:e,compiled:await Ot(_,{name:`summits-${e}-${t}`,grid:a.grid,pixelCount:m,elevation:y,validity:ne,maximumRadiusPixels:t,incompleteNeighborhood:e,settings:b,buffers:i,capacity:R}),buffers:i,dirty:!0,list:null,reader:new xe(_,`summits-${n}`,[{buffer:i.count,size:4},{buffer:i.total,size:4},{buffer:i.overflow,size:4},{buffer:i.clamped,size:4},{buffer:i.ids,size:R*4},{buffer:i.drops,size:R*4}],e=>ft(o,e))};return j.set(n,o),o}function Oe(){let t=N(e.options.incompleteNeighborhood);return[t,t?.mode===`reject`?N(`ignore`):void 0].filter(e=>!!e)}let Re=_.track(new be),He=_.track(new be);function Ue(){let t=e.options.incompleteNeighborhood;return!N(t)||t===`reject`&&!N(`ignore`)}function B(){return Re.request({build:async()=>{let t=e.options.incompleteNeighborhood;await I(t),t===`reject`&&await I(`ignore`)},commit:()=>{at(),mt(),e.requestLayers()}}).catch(t=>{v||e.setStatus(`Cannot build the summits: ${t.message}`)})}async function We(){let t=e.options.incompleteNeighborhood,n=`${A()}|${t}`,r=pe.get(n);if(!r){let e=await Ot(_,{name:`summits-sweep-${n}`,grid:a.grid,pixelCount:m,elevation:y,validity:ne,maximumRadiusPixels:A(),incompleteNeighborhood:t,settings:x,buffers:S,capacity:16});r=pe.get(n),r?_.release(e):(r=e,pe.set(n,r))}return r}let Ge=await At(_,{grid:a.grid,pixelCount:m,elevation:y,validity:ne,settings:re,buffers:C}),V=null,Ke=await Nt(_,{grid:i.grid,pixelCount:i.pixelCount,elevation:ee,validity:te,buffers:w}),qe=!0,H=!0,U=!0,Je=!1,Ye=new Uint32Array(40),W=[],G=[],Xe=new Map,Ze=new Map,K=Array(L.sizesPixels.length).fill(0),q=[],J=null,Y={peaks:0,saddles:0,pits:0},Qe=0,X=null,$e=he(7),et=c.map(()=>$e()*Math.PI*2),tt=a.observers.find(e=>e.name===`Gornergrat`),nt=tt?a.getPixel(tt.longitude,tt.latitude):[d/2,f/2],it=t=>{let n=e.options.catalogueError/g,r=c[t];return[r.column+Math.cos(et[t])*n,r.row+Math.sin(et[t])*n]};function at(){b.write(rt({radius:e.options.summitRadius,minimumDrop:0},a.grid));for(let e of j.values())e.dirty=!0;ot()}function ot(){let[t,n,r,a]=i.bounds,o=Math.min(e.options.summitRadius,(r-t)/2,(a-n)/2),s=[[t,n,t+o,a],[r-o,n,r,a],[t+o,n,r-o,n+o],[t+o,a-o,r-o,a]],c=new Float32Array(s.length*12);s.forEach(([e,t,n,r],i)=>{c.set([e,t,n,t,n,r,e,t,n,r,e,r],i*12)}),k.write(c)}function st(){let t=e.options,n=new Float32Array(128).fill(-5),r=new Float32Array(64).fill(NaN),i=new Float32Array(64).fill(NaN),o=new Float32Array(128).fill(1e7);c.forEach((e,s)=>{let[c,l]=it(s);if(n[s*2]=c,n[s*2+1]=l,t.snapCatalogueHeights&&e.elevationMeters!==null&&(r[s]=e.elevationMeters),t.snapDistanceRule){let e=Math.hypot(c-nt[0],l-nt[1])*g;i[s]=Math.min(250,60+.004*e)}let[u,d]=a.getMeters(c,l);o[s*2]=u,o[s*2+1]=d}),C.candidates.write(n),C.candidateHeights.write(r),C.candidateRadii.write(i),ae.write(o),qe=!0}function ct(){let t=e.options;re.write(pt({radius:t.snapRadius,maximumMove:t.snapMaximumMove,maximumHeightChange:t.snapMaximumHeightChange,interior:t.snapInterior},a.grid)),qe=!0}function lt(){let t=e.options.contourInterval,n=Math.floor(i.elevationRange[0]/t)*t;return Array.from({length:40},(e,r)=>n+r*t)}function ut(){w.levels.write(Float32Array.from(lt())),U=!0}function dt(){let t=e.options,n=`Highest within ${t.summitRadius} m, at least ${t.summitMinimumDrop} m above its ring · ${g.toFixed(1)} m analysis grid`,r=[t.summitRadius];t.showSnap?(n=`Catalogue peaks snapped within ${t.snapRadius} m · ${g.toFixed(1)} m analysis grid`,r=[t.snapRadius]):(t.showContours||t.showCritical)&&(n=`Contours every ${t.contourInterval} m · critical points on the ${g.toFixed(1)} m grid`,r=[]),e.setFurniture({title:{subtitle:n,sample:De(i)},scaleBar:{units:`metric`,ticks:r}}),e.setReadout(`radiusNow`,t.summitRadius),e.setReadout(`dropNow`,t.summitMinimumDrop)}function ft(e,t){e.list=vt(t,R,d,a.elevation),Oe().includes(e)&&mt()}function mt(){let t=e.options,n=N(t.incompleteNeighborhood)?.list??me;if(!n)return;me=n;let r=yt(n.candidates,t.summitMinimumDrop);W=r.survivors,G=r.rejected,Xe=wt(W,c,g),Ze=wt(G,c,g);let i=ie.map(()=>[]);K=K.map(()=>0);for(let e of W){let t=bt(e.drop),[n,r]=a.getMeters(e.column,e.row);i[t].push(n,r),K[t]++}i.forEach((e,t)=>{e.length&&ie[t].write(Float32Array.from(e))});let o=[];for(let e of G){let[t,n]=a.getMeters(e.column,e.row);o.push(t,n)}o.length&&T.write(Float32Array.from(o)),e.setReadout(`summitCount`,W.length),e.setReadout(`rejectedCount`,G.length),e.setReadout(`listUse`,n.listOverflow?`capped: ${P(R)} of ${P(n.total)} candidates listed`:`${P(n.total)} of ${P(R)} places`),e.setReadout(`radiusClamp`,n.radiusClamped?`clamped to ${z(A()*g)} m by the search bound`:`as requested`),e.setReadout(`edgeLost`,ht()),e.setChart(`dropHistogram`,{kind:`histogram`,values:xt(n.candidates,Bt,Vt),xDomain:[0,Bt*Vt],xLabel:`Drop to the ring (m); the last bin holds all larger drops`,yLabel:`Disc maxima`,height:130,link:{option:`summitMinimumDrop`,label:e=>`at least ${e} m`},description:`Histogram of the drop of every disc maximum the GPU found, with the minimum drop marked: candidates left of the mark fail the test.`}),nn(),e.requestLayers()}function ht(){let t=e.options,n=N(`ignore`)?.list,r=t.summitMinimumDrop;if(t.incompleteNeighborhood===`reject`){let e=N(`reject`)?.list;return!n||!e?null:yt(n.candidates,r).survivors.length-yt(e.candidates,r).survivors.length}if(!n)return null;let i=Math.ceil(t.summitRadius/(g*.98))+1,a=0;for(let e of yt(n.candidates,r).survivors)(e.column<=i||e.row<=i||e.column>=d-1-i||e.row>=f-1-i)&&s.discAndRing(e.column,e.row,t.summitRadius,`ignore`).incomplete&&a++;return a}function gt(){X&&clearTimeout(X),X=setTimeout(()=>void _t(),200)}async function _t(){let n=++Qe,r=e.options,i=await We();if(v||n!==Qe)return;let o=[];try{for(let e of Ie){x.write(rt({radius:e,minimumDrop:r.summitMinimumDrop},a.grid));let s=t.createCommandEncoder({id:`summits-sweep-encoder`});i.encode(s,{parameters:void 0}),t.submit(s.finish());let[c,l]=await Promise.all([S.total.readAsync(0,4),S.clamped.readAsync(0,4)]);if(v||n!==Qe)return;o.push(F(l,1)[0]?NaN:F(c,1)[0])}}catch{return}e.setChart(`countByRadius`,{kind:`line`,series:[{label:`Summits`,x:Ie,y:o,points:!0}],xLabel:`Radius (m)`,yLabel:`Summits`,xDomain:[Ie[0],Ie[Ie.length-1]],height:130,link:{option:`summitRadius`,label:e=>`r = ${e} m`},description:`Number of summits found at six radii with the current minimum drop: the count falls steeply as the radius grows. A radius beyond the search bound is left out.`})}let St=new xe(_,`snap`,[{buffer:C.positions,size:512},{buffer:C.heights,size:256},{buffer:C.status,size:256},{buffer:C.distance,size:256},{buffer:C.overflow,size:4}],e=>Kt(e)),Q=[];function Kt(t){let n=we(t,128),r=we(t.slice(512),64),i=F(t.slice(768),64),o=we(t.slice(1024),64);q=[],Q=[];let s=new Float32Array(128).fill(1e7),l=new Float32Array(256).fill(1e7);c.forEach((e,t)=>{q.push({status:i[t],distance:o[t],height:r[t]});let c=n[t*2],u=n[t*2+1];Q.push([c,u]);let[d,f]=a.getMeters(c,u);s[t*2]=d,s[t*2+1]=f;let[p,m]=it(t),[h,g]=a.getMeters(p,m);l.set([h,g,d,f],t*4)}),oe.write(s),se.write(l),J=Tt(i,o,c.length),e.setLegendData(`snapCounts`,J.counts),e.setReadout(`medianMove`,J.medianMoveMeters),e.setReadout(`catalogueSize`,c.length),qt(),nn(),e.requestLayers()}function qt(){if(!J)return;let t=e.ground(),n=[[`snapped`,`Snapped`],[`unchanged`,`Already on the summit`],[`flank`,`On a flank`],[`too-far`,`Too far`],[`height-mismatch`,`Height mismatch`]].filter(([e])=>J.counts[e]>0).map(([e,n])=>({label:n,value:J.counts[e],color:ke(e,t)}));e.setChart(`snapOutcomes`,{kind:`stacked`,format:`value`,segments:n,description:`How many catalogue peaks ended in each snap outcome, in the map colours.`})}let Jt=new xe(_,`critical`,[{buffer:E,size:m*4},{buffer:ce,size:m*4}],e=>Yt(e));function Yt(t){let n=F(t,m),r=F(t.slice(m*4),m),i=[],o=[],s=[],c=0,l=0;for(let e=0;e<m;e++){let t=n[e];if(t===0||t===Z.regular)continue;let[u,f]=a.getMeters(e%d,Math.floor(e/d));t===Z.peak?i.push(u,f):t===Z.pit?s.push(u,f):t===Z.saddle?(o.push(u,f),c+=r[e]/2-1):t===Z.boundary&&l++}let u=e=>Float32Array.from(e.slice(0,$*2));i.length&&D.write(u(i)),o.length&&ue.write(u(o)),s.length&&O.write(u(s)),Y.peaks=i.length/2,Y.saddles=o.length/2,Y.pits=s.length/2,e.setReadout(`criticalCounts`,`${P(Y.peaks)} peaks, ${P(Y.saddles)} saddles, ${P(Y.pits)} pits`),e.setReadout(`euler`,`peaks - saddles + pits = ${P(Y.peaks-c+Y.pits)} (saddles counted with multiplicity; ${P(l)} boundary cells unclassified)`),e.requestLayers()}async function Xt(e){return{key:e,compiled:await jt(_,{grid:a.grid,pixelCount:m,elevation:y,validity:ne,connectivity:e===`8`?8:6,classes:E,signChanges:ce,counts:le})}}function Zt(t){let n=V;V=t,n&&_.release(n.compiled),H=!0,e.requestLayers()}function Qt(){return He.request({build:()=>Xt(e.options.connectivity),commit:Zt,discard:e=>_.release(e.compiled)}).catch(t=>{v||e.setStatus(`Cannot build the critical points: ${t.message}`)})}let $t=new xe(_,`contours`,[{buffer:w.overflow,size:4},...w.counts.map(e=>({buffer:e,size:4}))],e=>en(e));function en(t){let n=F(t,1)[0];Ye=F(t.slice(4),40);let r=0,i=0;for(let e of Ye)r+=e,e>0&&i++;Je=!0,e.setReadout(`contourSummary`,`${i} levels with data, ${P(r)} segments`+(n?`; a level exceeded ${P(Et)} segments`:``)),e.requestLayers()}function tn(e){return a.getLongitudeLatitude(e.column,e.row)}function nn(){let t=e.options,n=[];if(t.showSummits&&[...Xe].map(([e,t])=>({candidate:W[e],peak:t})).sort((e,t)=>t.candidate.drop-e.candidate.drop).slice(0,Ht).forEach(({candidate:e,peak:t},r)=>{n.push({kind:`landform`,id:`summits:name:${t.osmId}`,coordinate:tn(e),text:t.name,marker:`none`,...t.elevationMeters===null?{}:{elevationMeters:t.elevationMeters},priority:Ht-r})}),t.showRejected&&t.summitMinimumDrop>L.minimumDropMeters){let e=[...Ze].map(([e,t])=>({candidate:G[e],peak:t})).filter(({candidate:e})=>e.drop>=L.minimumDropMeters).sort((e,t)=>t.candidate.drop-e.candidate.drop)[0];e&&n.push({kind:`note`,id:`summits:miss`,coordinate:tn(e.candidate),title:Se(`{name}: {drop:integer} m drop`,{name:e.peak.name,drop:e.candidate.drop}),text:Se(`Highest in its disc, but {needed:integer} m are needed`,{needed:t.summitMinimumDrop}),anchor:`se`,distance:50,priority:4})}t.showEdge&&n.push({kind:`frame`,id:`summits:edge`,bounds:i.lngLatBounds,text:`Data ends here`}),t.showSnap&&q.length&&n.push(...rn()),e.setAnnotations(`summits`,n)}function rn(){let e=c.map((e,t)=>t),t=e=>{let t=Fe(q[e].status);return t===`flank`||t===`too-far`||t===`height-mismatch`};return[...e.filter(t).slice(0,2),...e.filter(e=>!t(e))].slice(0,Ut).map((e,t)=>{let n=c[e],[r,i]=Q[e]??it(e);return{kind:`landform`,id:`summits:snap:${n.osmId}`,coordinate:a.getLongitudeLatitude(r,i),text:n.name,marker:`none`,...n.elevationMeters===null?{}:{elevationMeters:n.elevationMeters},priority:Ut-t}})}async function an(){try{e.setReadout(`timing`,`measuring...`);let n=async(n,r,i)=>`${i} ${ye(await ve(t,n,{parameters:void 0,completionBuffer:r,signal:e.signal,runs:3,warmUpRuns:1,repetitions:2})).replace(/ · .*/,``)}`;await Promise.all([B(),Qt()]);let r=N(e.options.incompleteNeighborhood);if(!r||V?.key!==e.options.connectivity)throw Error(`the graphs were rebuilt while measuring`);let i=[await n(r.compiled,r.buffers.count,`summits`),await n(V.compiled,le,`critical points`),await n(Ge,C.status,`snap`),await n(Ke,w.overflow,`40 contour levels`)];e.setReadout(`timing`,i.join(` · `))}catch(t){v||e.setReadout(`timing`,`failed: ${t.message}`)}}function on(e){if(!e.coordinate)return null;let[t,n]=i.projection.project(e.coordinate[0],e.coordinate[1]),[r,o,s,c]=i.bounds;return t<r||t>s||n<o||n>c?null:a.getPixelFromMeters(t,n)}function sn(t,n){let r=Wt*e.getMetersPerPixel()/g,i=-1,a=r;return t.forEach((e,t)=>{let r=Math.hypot(e.column-n[0],e.row-n[1]);r<a&&(i=t,a=r)}),i}return e.setLegendData(`ground`,e.ground()),e.setReadout(`grid`,`${d} × ${f} cells at ${g.toFixed(1)} m for summits; ${i.width} × ${i.height} at ${i.groundCellSize.toFixed(1)} m for contours and the relief`),at(),st(),ct(),ut(),await B(),dt(),gt(),{getCompiledGraphs:()=>{let e=[Ge,Ke,...Oe().map(e=>e.compiled)];return V&&e.push(V.compiled),e},setOption(t){switch(t){case`summitRadius`:at();break;case`summitMinimumDrop`:mt(),gt();break;case`summitMaximumRadius`:case`incompleteNeighborhood`:Ue()&&B(),at(),gt();break;case`catalogueError`:case`snapCatalogueHeights`:case`snapDistanceRule`:st();break;case`snapRadius`:case`snapMaximumMove`:case`snapMaximumHeightChange`:case`snapInterior`:ct();break;case`connectivity`:H=!0;break;case`contourInterval`:ut();break;default:break}dt(),nn(),e.requestLayers()},onAction(e){e===`measure`&&an()},encode(t){let n=e.options;!Re.pending&&Ue()&&B();for(let e of Oe())e.dirty&&=(e.compiled.encode(t,{parameters:void 0}),e.reader.request(t),!1),e.reader.flush(t);if(qe&&n.showSnap&&(Ge.encode(t,{parameters:void 0}),St.request(t),qe=!1),St.flush(t),n.showCritical){let e=n.connectivity;V?.key!==e&&!He.pending&&Qt(),H&&V?.key===e&&(V.compiled.encode(t,{parameters:void 0}),Jt.request(t),H=!1)}Jt.flush(t),U&&n.showContours&&(Ke.encode(t,{parameters:void 0}),$t.request(t),U=!1),$t.flush(t)},getLayers(){let t=e.options,n=e.ground(),r=ze(n),a=[u.getLayer()];if(t.showContours&&Je){let e=lt();for(let r=0;r<40;r++){if(Ye[r]===0)continue;let o=Ve(n,Math.round(e[r]/t.contourInterval)%t.contourIndexEvery===0);a.push(new ge({id:`summits-contour-${r}`,coordinateOrigin:h,segments:w.vertices[r],drawCommands:w.drawCommands,drawCommandIndex:r,positionScale:[i.layerCellSize[0],-i.layerCellSize[1]],positionOffset:[i.bounds[0],i.bounds[3]],widthPixels:o.widthPixels,color:o.color}))}}if(t.showEdge&&a.push(new _e({id:`summits-edge-band`,coordinateOrigin:h,triangles:k,features:de,vertexCount:24,values:fe,valueFormat:`uint32`,colormap:`category`,palette:[[0,0,0,0]],hatchClasses:[0],hatchColor:r.hatch,hatchSpacingPixels:6,hatchWidthPixels:1.2})),t.showCritical){let e={coordinateOrigin:h,opacityStops:[[Be-Le,0],[Be+Le,1]]};Y.pits>0&&a.push(new M({...e,id:`summits-critical-pits`,positions:O,instanceCount:Math.min(Y.pits,$),shape:`ring`,radiusPixels:4,outlineWidthPixels:1.5,color:r.pit})),Y.saddles>0&&a.push(new M({...e,id:`summits-critical-saddles`,positions:ue,instanceCount:Math.min(Y.saddles,$),shape:`diamond`,radiusPixels:4,outlineColor:r.summit,outlineWidthPixels:1,color:r.saddle})),Y.peaks>0&&a.push(new M({...e,id:`summits-critical-peaks`,positions:D,instanceCount:Math.min(Y.peaks,$),shape:`triangle`,radiusPixels:4,outlineColor:r.paper,outlineWidthPixels:1,color:r.criticalPeak}))}return t.showRejected&&G.length>0&&a.push(new M({id:`summits-rejected`,coordinateOrigin:h,positions:T,instanceCount:G.length,shape:`triangle`,radiusPixels:Gt,fillOpacity:0,outlineColor:r.rejected,outlineWidthPixels:1,color:r.rejected})),t.showSummits&&L.sizesPixels.forEach((e,t)=>{K[t]!==0&&a.push(new M({id:`summits-class-${t}`,coordinateOrigin:h,positions:ie[t],instanceCount:K[t],shape:`triangle`,radiusPixels:e/2,outlineColor:r.paper,outlineWidthPixels:1.5,color:r.summit}))}),t.showSnap&&q.length>0&&a.push(new ge({id:`summits-snap-segments`,coordinateOrigin:h,segments:se,instanceCount:c.length,widthPixels:1.5,color:r.connector}),new M({id:`summits-snap-original-edge`,coordinateOrigin:h,positions:ae,instanceCount:c.length,shape:`ring`,radiusPixels:5.25,outlineWidthPixels:3,color:r.summit}),new M({id:`summits-snap-original`,coordinateOrigin:h,positions:ae,instanceCount:c.length,shape:`ring`,radiusPixels:4.5,outlineWidthPixels:1.5,color:r.paper}),new M({id:`summits-snap-result`,coordinateOrigin:h,positions:oe,instanceCount:c.length,values:C.status,valueFormat:`uint32`,colormap:`category`,palette:Pe(n),shape:`circle`,radiusPixels:4})),a},getTooltip(t){let n=on(t);if(!n)return null;let r=e.options,l=e.ground(),u=ze(l);if(r.showSnap&&q.length){let t=Wt*e.getMetersPerPixel()/g,s=-1,u=t;c.forEach((e,t)=>{let[r,i]=it(t),a=Q[t]??[r,i],o=Math.min(Math.hypot(r-n[0],i-n[1]),Math.hypot(a[0]-n[0],a[1]-n[1]));o<u&&(s=t,u=o)});let d=s>=0?Fe(q[s].status):null;if(s>=0&&d){let e=Q[s],t=je(o,a.getLongitudeLatitude(e[0],e[1]),g*1.5);return Lt({peak:c[s],status:d,moveMeters:q[s].distance,errorMeters:r.catalogueError,analysisCellHeight:q[s].height,fullCellHeight:t?t.elevation:null,analysisCellMeters:g,fullCellMeters:i.groundCellSize,ground:l})}}if(r.showSummits){let e=sn(W,n);if(e>=0)return Pt({candidate:W[e],peak:Xe.get(e)??null,radiusMeters:r.summitRadius,colors:u})}if(r.showRejected){let e=sn(G,n);if(e>=0)return Ft({candidate:G[e],minimumDrop:r.summitMinimumDrop,peak:Ze.get(e)??null,colors:u})}let f=Math.round(n[0]),p=Math.round(n[1]),m=a.elevation[p*d+f];if(r.showProbe){let e=s.discAndRing(f,p,r.summitRadius,r.incompleteNeighborhood),t=e.beatenBy?Math.hypot(e.beatenBy[0]-f,e.beatenBy[1]-p)*s.getGroundCellSize(p):null;return It({test:e,elevation:m,radiusMeters:r.summitRadius,minimumDrop:r.summitMinimumDrop,center:a.getLongitudeLatitude(f,p),distanceToHigherMeters:t,colors:u})}return Rt({elevation:m,cellMeters:g,colors:u})},onGroundChange(t){u.setGround(t),e.setLegendData(`ground`,t),qt(),e.requestLayers()},onThemeChange(){e.requestLayers()},destroy(){v=!0,X&&clearTimeout(X);for(let e of j.values())e.reader.stop();St.stop(),Jt.stop(),$t.stop(),u.destroy(),_.destroy()}}}export{Kt as createSummits};