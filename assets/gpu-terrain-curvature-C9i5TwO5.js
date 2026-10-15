import{J as e}from"./src-CNCKFW2b.js";import{C as t}from"./arithmetic-expression-Kknn3j_i.js";import{C as n,D as r,E as i,I as a,M as o,R as s,S as c,T as l,_ as u,j as d,k as f,l as p,w as m,x as h,y as g,z as _}from"./resources-CwvhEOBe.js";import{a as v,i as y}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{t as b}from"./gpu-scan-BuGhTLDQ.js";import{l as x,s as S}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as C,c as w,i as T,n as ee,o as te,r as E,s as D,t as O,u as k}from"./raster-band-utils-CA-sh8xt.js";import{a as A,c as j,l as M}from"./raster-grid-D2bmCwkc.js";import{n as ne,t as N}from"./terrain-geomorphometry-utils-BlzhWTj6.js";var P=256,F=class{id;props;constructor(e){this.id=e.id??`gpu-scan-uint64`,this.props=Object.freeze({...e,id:this.id}),L(this.props)}getCommandNodes(e){let t=[],r=this.props;for(let t of[r.inputLow,r.inputHigh,r.outputLow,r.outputHigh])if(v(t).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(r.inputLow.length===0)return t;let a=i(e,r.outputLow,r.inputLow.length),o=i(e,r.outputHigh,r.inputLow.length);t.push(...new b({id:`${this.id}-low`,input:r.inputLow,output:a,mode:`inclusive`}).getCommandNodes(e));let c=r.inputHigh instanceof s?n(e,`${this.id}-adjusted-high`,r.inputHigh):m(e,`${this.id}-adjusted-high`,`uint32`,r.inputHigh.length),l=y(e,[r.inputHigh,a,c]),u;for(let[n,[r,i,a]]of l.entries())t.push(...I(e,`${this.id}-carry-${n}`,r,i,a,u)),u=e.createDataView(i.buffer,{format:`uint32`,length:1,byteOffset:i.byteOffset+(i.length-1)*4});return t.push(...new b({id:`${this.id}-high`,input:c,output:o,mode:`inclusive`}).getCommandNodes(e)),t}};function I(e,t,n,i,o,s){let l=[],u=n.length,d=h(`GPUScanUint64`,u,P,e.device.limits.maxComputeWorkgroupsPerDimension),p=`const LENGTH: u32 = ${u}u;
const INPUT_HIGH_OFFSET: u32 = ${f(n)}u;
const PREFIX_LOW_OFFSET: u32 = ${f(i)}u;
const OUTPUT_OFFSET: u32 = ${f(o)}u;
@group(0) @binding(0) var<storage, read> inputHigh: array<u32>;
@group(0) @binding(1) var<storage, read> prefixLow: array<u32>;
@group(0) @binding(2) var<storage, read_write> adjustedHigh: array<u32>;
${s?`@group(0) @binding(3) var<storage, read> previousLow: array<u32>;`:``}
@compute @workgroup_size(${P})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${c(d,P)}
  if (index >= LENGTH) {
    return;
  }
  var preceding = ${s?`previousLow[${f(s)}u]`:`0u`};
  if (index > 0u) {
    preceding = prefixLow[PREFIX_LOW_OFFSET + index - 1u];
  }
  let carry = select(0u, 1u, prefixLow[PREFIX_LOW_OFFSET + index] < preceding);
  adjustedHigh[OUTPUT_OFFSET + index] = inputHigh[INPUT_HIGH_OFFSET + index] + carry;
}`;return l.push(a({id:t,workload:{operation:`GPUScanUint64`,commandCount:1,maximumWorkgroupCount:Math.ceil(u/P),maximumInvocationCount:Math.ceil(u/P)*P,readByteLength:(u*2+(s?1:0))*Uint32Array.BYTES_PER_ELEMENT,writeByteLength:u*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:n,usage:`storage-read`},{buffer:i,usage:`storage-read`},{buffer:o,usage:`storage-write`},...s?[{buffer:s,usage:`storage-read`}]:[]],compile:({device:e})=>{let a=new _(e,{id:t,source:p,shaderLayout:{bindings:[{name:`inputHigh`,type:`read-only-storage`,group:0,location:0},{name:`prefixLow`,type:`read-only-storage`,group:0,location:1},{name:`adjustedHigh`,type:`storage`,group:0,location:2},...s?[{name:`previousLow`,type:`read-only-storage`,group:0,location:3}]:[]]}});return{encode:({computePass:e,getBuffer:t})=>{let c={inputHigh:r(n,t),prefixLow:r(i,t),adjustedHigh:r(o,t)};s&&(c.previousLow=r(s,t)),a.dispatch(e,{bindings:c,x:d.x,y:d.y,z:d.z})},destroy:()=>a.destroy()}}})),l}function L(e){for(let[t,n]of Object.entries({inputLow:e.inputLow,inputHigh:e.inputHigh,outputLow:e.outputLow,outputHigh:e.outputHigh}))for(let r of v(n))d(r,`${e.id} ${t}`);if(e.inputHigh.length!==e.inputLow.length||e.outputLow.length<e.inputLow.length||e.outputHigh.length<e.inputLow.length)throw Error(`${e.id} inputs must match and outputs must cover every input row`);let t=v(e.inputLow),n=v(e.inputHigh),r=v(e.outputLow),i=v(e.outputHigh);if(r.some(e=>[...t,...n,...i].some(t=>t.buffer===e.buffer))||i.some(e=>t.some(t=>t.buffer===e.buffer)))throw Error(`${e.id} split-word outputs must use safe separate buffers`);for(let t of[r,i])for(let[n,r]of t.entries())if(t.slice(0,n).some(e=>l(e,r)))throw Error(`${e.id} output chunks must not overlap`)}var R=class{id;input;output;rows;columns;stats;constructor(e){this.id=e.id??`gpu-transpose`,this.input=e.input,this.output=e.output,this.rows=e.rows,this.columns=e.columns,this.stats=z(e.rows,e.columns);for(let[e,t]of[[`input`,this.input],[`output`,this.output]])for(let n of v(t))o(n,[`uint32`,`sint32`,`float32`],`${this.id} ${e}`);if(this.output.format!==this.input.format)throw Error(`${this.id} input and output formats must match`);if(this.input.length<this.stats.elementCount)throw Error(`${this.id} input must contain at least rows * columns rows`);if(this.output.length<this.stats.elementCount)throw Error(`${this.id} output must contain at least rows * columns rows`);let t=new Set(v(this.input).map(e=>e.buffer)),n=v(this.output);if(n.some(e=>t.has(e.buffer)))throw Error(`${this.id} input and output must use separate buffers`);for(let[e,t]of n.entries())if(n.slice(0,e).some(e=>l(e,t)))throw Error(`${this.id} output chunks must not overlap`)}getCommandNodes(e){let n=[];if(W(e,this.input,`${this.id} input`),W(e,this.output,`${this.id} output`),this.stats.elementCount===0)return n;G(e.device,this.id);let r=i(e,this.input,this.stats.elementCount),a=i(e,this.output,this.stats.elementCount),o=t(v(r)).filter(e=>e.length),s=t(v(a)).filter(e=>e.length);for(let[t,r]of o.entries())for(let[i,a]of s.entries()){let c=q(r.offset,r.length,a.offset,a.length,this.rows,this.columns);if(!c)continue;let l=H(c.tileCount,e.device.limits.maxComputeWorkgroupsPerDimension);n.push(...V(e,{...this,id:o.length===1&&s.length===1?this.id:`${this.id}-input-${t}-output-${i}`,input:r.data,output:a.data},l,c))}return n}};function z(e,t){U(`rows`,e),U(`columns`,t);let n=e*t;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`GPUTranspose rows * columns must fit in a uint32 index range`);let r=Math.ceil(e/16),i=Math.ceil(t/16);return Object.freeze({rows:e,columns:t,elementCount:n,tileRowCount:r,tileColumnCount:i,tileCount:r*i,workgroupSize:Object.freeze([16,16,1])})}function B(e,t,n={inputOffset:0,outputOffset:0,tileRowOffset:0,tileColumnOffset:0,tileColumnCount:e.stats.tileColumnCount,tileCount:e.stats.tileCount,elementCount:e.stats.elementCount}){let r=n.inputOffset===0&&e.input.length>=e.stats.elementCount,i=n.outputOffset===0&&e.output.length>=e.stats.elementCount;return`const ROWS: u32 = ${e.rows}u;
const COLUMNS: u32 = ${e.columns}u;
const TILE_COLUMN_COUNT: u32 = ${n.tileColumnCount}u;
const INPUT_START: u32 = ${n.inputOffset}u;
const INPUT_LENGTH: u32 = ${e.input.length}u;
const OUTPUT_START: u32 = ${n.outputOffset}u;
const OUTPUT_LENGTH: u32 = ${e.output.length}u;
const INPUT_OFFSET: u32 = ${f(e.input)}u;
const OUTPUT_OFFSET: u32 = ${f(e.output)}u;

// Move raw words so signed zeros, NaN payloads, and integer bits survive unchanged.
@group(0) @binding(0) var<storage, read> inputValues: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
var<workgroup> tile: array<array<u32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localIdentifier: vec3u
) {
  let tileIndex = (workgroupId.z * ${t.y}u + workgroupId.y) *
    ${t.x}u + workgroupId.x;
  if (tileIndex >= ${n.tileCount}u) { return; }
  let localTileRow = tileIndex / TILE_COLUMN_COUNT;
  let tileRow = localTileRow + ${n.tileRowOffset}u;
  let tileColumn = tileIndex - localTileRow * TILE_COLUMN_COUNT + ${n.tileColumnOffset}u;
  let inputRow = tileRow * 16u + localIdentifier.y;
  let inputColumn = tileColumn * 16u + localIdentifier.x;
  let inputIndex = inputRow * COLUMNS + inputColumn;
  let destinationIndex = inputColumn * ROWS + inputRow;
  if (inputRow < ROWS && inputColumn < COLUMNS
      ${r?``:`&& inputIndex >= INPUT_START && inputIndex - INPUT_START < INPUT_LENGTH`}
      ${i?``:`&& destinationIndex >= OUTPUT_START && destinationIndex - OUTPUT_START < OUTPUT_LENGTH`}) {
    tile[localIdentifier.y][localIdentifier.x] =
      inputValues[INPUT_OFFSET + inputIndex - INPUT_START];
  }
  workgroupBarrier();
  let outputRow = tileColumn * 16u + localIdentifier.y;
  let outputColumn = tileRow * 16u + localIdentifier.x;
  let outputIndex = outputRow * ROWS + outputColumn;
  let sourceIndex = outputColumn * COLUMNS + outputRow;
  if (outputRow < COLUMNS && outputColumn < ROWS
      ${i?``:`&& outputIndex >= OUTPUT_START && outputIndex - OUTPUT_START < OUTPUT_LENGTH`}
      ${r?``:`&& sourceIndex >= INPUT_START && sourceIndex - INPUT_START < INPUT_LENGTH`}) {
    outputValues[OUTPUT_OFFSET + outputIndex - OUTPUT_START] =
      tile[localIdentifier.x][localIdentifier.y];
  }
}`}function V(e,t,n,i){let o=[],s=B(t,n,i);return o.push(a({id:t.id,workload:{operation:`GPUTranspose`,commandCount:1,maximumWorkgroupCount:i.tileCount,maximumInvocationCount:i.tileCount*16*16,readByteLength:i.elementCount*Uint32Array.BYTES_PER_ELEMENT,writeByteLength:i.elementCount*Uint32Array.BYTES_PER_ELEMENT},resources:[{buffer:t.input,usage:`storage-read`},{buffer:t.output,usage:`storage-write`}],compile:({device:e})=>{let i=new _(e,{id:t.id,source:s,shaderLayout:{bindings:[{name:`inputValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1}]}});return{encode:({computePass:e,getBuffer:a})=>{let o={inputValues:r(t.input,a),outputValues:r(t.output,a)};i.dispatch(e,{bindings:o,x:n.x,y:n.y,z:n.z})},destroy:()=>i.destroy()}}})),o}function H(e,t){return h(`GPUTranspose`,e*16*16,256,t)}function U(e,t){if(!Number.isSafeInteger(t)||t<0)throw Error(`GPUTranspose ${e} must be a non-negative safe integer`)}function W(e,t,n){if(v(t).some(t=>t.buffer.graph!==e))throw Error(`${n} belongs to a different GPUCommandGraph`)}function G(e,t){if(e.limits.maxComputeInvocationsPerWorkgroup<256||e.limits.maxComputeWorkgroupSizeX<16||e.limits.maxComputeWorkgroupSizeY<16)throw Error(`${t} requires 16 by 16 compute workgroups`)}function K(e,t,n){let r=Math.floor(e/n),i=Math.floor((e+t-1)/n),a=e%n,o=(e+t-1)%n+1;if(r===i)return[{rowStart:r,rowEnd:r+1,columnStart:a,columnEnd:o}];let s=[{rowStart:r,rowEnd:r+1,columnStart:a,columnEnd:n},{rowStart:i,rowEnd:i+1,columnStart:0,columnEnd:o}];return r+1<i&&s.push({rowStart:r+1,rowEnd:i,columnStart:0,columnEnd:n}),s}function q(e,t,n,r,i,a){let o=i,s=0,c=a,l=0,u=0;for(let d of K(e,t,a))for(let e of K(n,r,i)){let t=Math.max(d.rowStart,e.columnStart),n=Math.min(d.rowEnd,e.columnEnd),r=Math.max(d.columnStart,e.rowStart),i=Math.min(d.columnEnd,e.rowEnd);t>=n||r>=i||(u+=(n-t)*(i-r),o=Math.min(o,t),s=Math.max(s,n),c=Math.min(c,r),l=Math.max(l,i))}if(!u)return;let d=Math.floor(o/16),f=Math.floor(c/16),p=Math.ceil(l/16)-f;return{inputOffset:e,outputOffset:n,tileRowOffset:d,tileColumnOffset:f,tileColumnCount:p,tileCount:(Math.ceil(s/16)-d)*p,elementCount:u}}function J(e){return p(e,12)}var Y=`
${O}
const MAXIMUM_QUANTIZED: f32 = ${x(2147483520)};
fn add64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  return vec2<u32>(low, left.y + right.y + select(0u, 1u, low < left.x));
}
fn subtract64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x - right.x, left.y - right.y - select(0u, 1u, left.x < right.x));
}
fn multiplyWide32(left: u32, right: u32) -> vec2<u32> {
  let left0 = left & 0xffffu;
  let left1 = left >> 16u;
  let right0 = right & 0xffffu;
  let right1 = right >> 16u;
  let product00 = left0 * right0;
  let product01 = left0 * right1;
  let product10 = left1 * right0;
  let product11 = left1 * right1;
  let middle = (product00 >> 16u) + (product01 & 0xffffu) + (product10 & 0xffffu);
  return vec2<u32>(
    (product00 & 0xffffu) | (middle << 16u),
    product11 + (product01 >> 16u) + (product10 >> 16u) + (middle >> 16u)
  );
}
fn multiply64By32(left: vec2<u32>, right: u32) -> vec2<u32> {
  let wide = multiplyWide32(left.x, right);
  return vec2<u32>(wide.x, wide.y + left.y * right);
}
fn negate64(value: vec2<u32>) -> vec2<u32> {
  return subtract64(vec2<u32>(0u, 0u), value);
}
fn unsigned64ToFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}
// Converts the magnitude, not the two's-complement words: f32(hi) * 2^32 + f32(lo) of a small
// negative value cancels -2^32 against a low word rounded up to 2^32.
fn signed64ToFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0i) {
    return -unsigned64ToFloat(negate64(value));
  }
  return unsigned64ToFloat(value);
}
fn isQuantizable(elevation: f32) -> bool {
  return isFiniteValue(elevation) && abs(elevation * INVERSE_QUANTUM) <= MAXIMUM_QUANTIZED;
}
fn quantizeElevation(elevation: f32) -> i32 {
  return i32(round(elevation * INVERSE_QUANTUM));
}`;function X(t,n){let{id:r,width:i,height:a}=n,o=n.includeSquares??!0,s=o?3:2,c=o?2:1,l=s-1,u=i*a;e(u<=J(t.device));let d=e=>({low:m(t,`${r}-summed-area-${e}-low`,`uint32`,s*u),high:m(t,`${r}-summed-area-${e}-high`,`uint32`,c*u)}),f=d(`first`),p=d(`second`),h=(e,n)=>t.createDataView(e.buffer,{format:`uint32`,length:u,byteOffset:e.byteOffset+n*u*4}),g=n.elevation.validity;if(!g)throw Error(`${r} summed-area table requires a canonical elevation band`);let _=n.originRelative?m(t,`${r}-anchor-index`,`uint32`,1):void 0,v=n.originRelative?m(t,`${r}-anchor`,`float32`,1):void 0,y=[];_&&v&&y.push(S(t,{id:`${r}-anchor-reset`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`anchorIndex`,view:_,type:`u32`,access:`read_write`}],invocationCount:1,body:`anchorIndex[anchorIndexOffset] = 0xffffffffu;`}),S(t,{id:`${r}-anchor-select`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`values`,view:n.elevation.storage.values,type:`f32`,access:`read`},{name:`validity`,view:g,type:`u32`,access:`read`},{name:`anchorIndex`,view:_,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,declarations:O,body:`if (validity[validityOffset + index] != 0u && isFiniteValue(values[valuesOffset + index])) {
          atomicMin(&anchorIndex[anchorIndexOffset], index);
        }`}),S(t,{id:`${r}-anchor-publish`,operation:`GPUTerrainTopographicPosition`,bindings:[{name:`values`,view:n.elevation.storage.values,type:`f32`,access:`read`},{name:`anchorIndex`,view:_,type:`u32`,access:`read`},{name:`anchor`,view:v,type:`f32`,access:`read_write`}],invocationCount:1,body:`let first = anchorIndex[anchorIndexOffset];
          var origin = 0.0;
          if (first != 0xffffffffu) { origin = values[valuesOffset + first]; }
          // Snap only small anchors; large f32 values already lie on the quantum lattice.
          if (abs(origin) < ${x(n.quantum*2**23)}) {
            origin = round(origin * ${x(1/n.quantum)}) * ${x(n.quantum)};
          }
          anchor[anchorOffset] = origin;`})),y.push(S(t,{id:`${r}-summed-area-quantize`,operation:`GPUTerrainTopographicPosition`,variant:`quantize`,bindings:[{name:`elevationValues`,view:n.elevation.storage.values,type:`f32`,access:`read`},...v?[{name:`anchor`,view:v,type:`f32`,access:`read`}]:[],{name:`elevationValidity`,view:g,type:`u32`,access:`read`},{name:`lowPlanes`,view:f.low,type:`u32`,access:`read_write`},{name:`highPlanes`,view:f.high,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`const PIXEL_COUNT: u32 = ${u}u;
const INVERSE_QUANTUM: f32 = ${x(1/n.quantum)};
${Y}`,body:`let elevation = elevationValues[elevationValuesOffset + index]${v?` - anchor[anchorOffset]`:``};
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u && isQuantizable(elevation);
  var quantized = 0i;
  if (isValid) {
    quantized = quantizeElevation(elevation);
  }
  lowPlanes[lowPlanesOffset + index] = bitcast<u32>(quantized);
  lowPlanes[lowPlanesOffset + ${l}u * PIXEL_COUNT + index] = select(0u, 1u, isValid);
  highPlanes[highPlanesOffset + index] = select(0u, 0xffffffffu, quantized < 0i);${o?`
  let magnitude = u32(abs(quantized));
  let square = multiplyWide32(magnitude, magnitude);
  lowPlanes[lowPlanesOffset + PIXEL_COUNT + index] = square.x;
  highPlanes[highPlanesOffset + PIXEL_COUNT + index] = square.y;`:``}`}));let C=(e,n,i)=>{for(let[a,s]of o?[[`sum`,0],[`square`,1]]:[[`sum`,0]])y.push(...new F({id:`${r}-summed-area-${e}-${a}`,inputLow:h(n.low,s),inputHigh:h(n.high,s),outputLow:h(i.low,s),outputHigh:h(i.high,s)}).getCommandNodes(t));y.push(...new b({id:`${r}-summed-area-${e}-count`,input:h(n.low,l),output:h(i.low,l),mode:`inclusive`}).getCommandNodes(t))};C(`rows`,f,p);for(let[e,n]of[[`low`,s],[`high`,c]])for(let o=0;o<n;o++)y.push(...new R({id:`${r}-summed-area-transpose-${e}-${o}`,input:h(p[e],o),output:h(f[e],o),rows:a,columns:i}).getCommandNodes(t));return C(`columns`,f,p),{lowTable:p.low,highTable:p.high,anchor:v,nodes:y}}var re=`
struct TerrainBoxSums {
  sum: vec2<u32>,
  square: vec2<u32>,
  count: u32,
}
fn readTablePlane64(plane: u32, linearIndex: i32) -> vec2<u32> {
  if (linearIndex < 0i) {
    return vec2<u32>(0u, 0u);
  }
  let offset = plane * PIXEL_COUNT + u32(linearIndex);
  return vec2<u32>(lowTable[lowTableOffset + offset], highTable[highTableOffset + offset]);
}
fn readTableCount(linearIndex: i32) -> u32 {
  if (linearIndex < 0i) {
    return 0u;
  }
  return lowTable[lowTableOffset + 2u * PIXEL_COUNT + u32(linearIndex)];
}
fn readColumnRange(column: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  var upper: i32;
  var lower: i32;
  if (column >= 0i) {
    upper = column * i32(HEIGHT) + lastRow;
    lower = column * i32(HEIGHT) + firstRow - 1i;
  } else {
    let lastColumnStart = (i32(WIDTH) - 1i) * i32(HEIGHT);
    upper = lastColumnStart + lastRow - 1i;
    lower = lastColumnStart + max(firstRow, 1i) - 2i;
  }
  return TerrainBoxSums(
    subtract64(readTablePlane64(0u, upper), readTablePlane64(0u, lower)),
    subtract64(readTablePlane64(1u, upper), readTablePlane64(1u, lower)),
    readTableCount(upper) - readTableCount(lower)
  );
}
fn readBoxSums(firstColumn: i32, lastColumn: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  let right = readColumnRange(lastColumn, firstRow, lastRow);
  let left = readColumnRange(firstColumn - 1i, firstRow, lastRow);
  return TerrainBoxSums(
    subtract64(right.sum, left.sum),
    subtract64(right.square, left.square),
    right.count - left.count
  );
}`,ie=`
struct TerrainBoxSums {
  sum: vec2<u32>,
  count: u32,
}
fn readTableSum64(linearIndex: i32) -> vec2<u32> {
  if (linearIndex < 0i) {
    return vec2<u32>(0u, 0u);
  }
  return vec2<u32>(lowTable[lowTableOffset + u32(linearIndex)], highTable[highTableOffset + u32(linearIndex)]);
}
fn readTableCount(linearIndex: i32) -> u32 {
  if (linearIndex < 0i) {
    return 0u;
  }
  return lowTable[lowTableOffset + PIXEL_COUNT + u32(linearIndex)];
}
fn readColumnRange(column: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  var upper: i32;
  var lower: i32;
  if (column >= 0i) {
    upper = column * i32(HEIGHT) + lastRow;
    lower = column * i32(HEIGHT) + firstRow - 1i;
  } else {
    let lastColumnStart = (i32(WIDTH) - 1i) * i32(HEIGHT);
    upper = lastColumnStart + lastRow - 1i;
    lower = lastColumnStart + max(firstRow, 1i) - 2i;
  }
  return TerrainBoxSums(
    subtract64(readTableSum64(upper), readTableSum64(lower)),
    readTableCount(upper) - readTableCount(lower)
  );
}
fn readBoxSums(firstColumn: i32, lastColumn: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  let right = readColumnRange(lastColumn, firstRow, lastRow);
  let left = readColumnRange(firstColumn - 1i, firstRow, lastRow);
  return TerrainBoxSums(subtract64(right.sum, left.sum), right.count - left.count);
}`,Z=4,ae=1e-6,oe=[.55,.45,0,0],Q=5,$=[`profile`,`plan`,`tangential`,`mean`,`gaussian`,`minimal`,`maximal`,`unsphericity`,`difference`,`horizontal-excess`,`vertical-excess`,`accumulation`,`ring`,`rotor`,`laplacian`];function se(e,t,n=new Float32Array(12)){T(`getGPUTerrainCurvatureParameterValues`,n,12);let r=e.ringGains??oe;if(r.length>Z)throw Error(`Terrain curvature ringGains must hold at most 4 values`);n.fill(0,0,12),ne(e,t,n),n[5]=e.flatGradient??ae;for(let[e,t]of r.entries())n[6+e]=t;return n}var ce=class{id;props;requiredHalo;constructor(e){this.id=e.id??`terrain-curvature`,this.props=e;let{id:t}=this,n=D(t,e.grid),r=e.method??`evans-young`;if(![`evans-young`,`zevenbergen-thorne`,`florinsky`].includes(r))throw Error(`${t} method must be evans-young, zevenbergen-thorne, or florinsky`);let{output:i}=e,a=Object.keys(i.curvatures??{});for(let e of a)if(!$.includes(e))throw Error(`${t} unknown curvature kind ${e}`);let o=a.map(e=>i.curvatures?.[e]).filter(e=>e!==void 0);if(o.length===0&&!i.ringCurvature&&!i.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,r]of[...a.map(e=>[e,i.curvatures?.[e]]),[`ringCurvature`,i.ringCurvature]])if(r&&(g(r,[`float32`],`${t} output.${e}`),r.length!==n))throw Error(`${t} output.${e} must contain one value per pixel`);if(i.outputValidity&&(g(i.outputValidity,[`uint32`],`${t} output.outputValidity`),i.outputValidity.length!==n))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(k(t,e.parameters,12),j(t,e.cellSizeMode),w(t,e.grid,e.cellSizeMode),M(t,e.rowDirection),![`clamp`,`nodata`].includes(e.borderMode??`clamp`))throw Error(`${t} borderMode must be clamp or nodata`);if(![`none`,`pade-tanh`].includes(e.ringSquash??`none`))throw Error(`${t} ringSquash must be none or pade-tanh`);let s=e.ringRadii??[2,8];if(s.length<1||s.length>Z||s.some((e,t)=>!Number.isInteger(e)||e<1||t>0&&e<=s[t-1]))throw Error(`${t} ringRadii must be 1 to 4 strictly ascending integers of at least 1`);te(t,[...o,i.ringCurvature,i.outputValidity],[...ee(e.elevation),e.parameters]);let c=r===`florinsky`?2:1;this.requiredHalo=i.ringCurvature?Math.max(c,s[s.length-1]):c}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n;C(t,e,n.elevation,[]),u(t,e,[n.parameters,...$.map(e=>r.curvatures?.[e]),r.ringCurvature,r.outputValidity]);let i=E(e,t,n.elevation,n.grid,!0),a=[];for(let e of $){let t=r.curvatures?.[e];t&&a.push({kind:`curvature`,curvature:e,view:t})}r.ringCurvature&&a.push({kind:`ring`,view:r.ringCurvature}),r.outputValidity&&a.push({kind:`validity`,view:r.outputValidity});let o=a.some(e=>e.kind===`curvature`),s=!!r.ringCurvature,c=[...i.nodes];for(let r=0,l=0;r<a.length;r+=Q,l++)c.push(fe(e,{id:`${t}-curvature-${l}`,props:n,source:i,slots:a.slice(r,r+Q),hasWindowOutput:o,hasRingOutput:s}));return c}},le={profile:`profile`,plan:`plan`,tangential:`tangential`,mean:`meanCurvature`,gaussian:`gaussian`,minimal:`minimal`,maximal:`maximal`,unsphericity:`unsphericity`,difference:`differenceCurvature`,"horizontal-excess":`horizontalExcess`,"vertical-excess":`verticalExcess`,accumulation:`accumulation`,ring:`ringProduct`,rotor:`rotor`,laplacian:`laplacian`};function ue(e,t){let n=[],r=(e,r,i)=>{n.push(`let ${e} = readWindowDelta(column, row, ${r}, ${i*t}, centre);`)};if(e===`florinsky`){for(let e=0;e<25;e++)r(`z${e}`,e%5-2,2-Math.floor(e/5));return{samples:n.join(`
  `),partials:`let r = zFactor * (2.0 * (z0 + z4 + z5 + z9 + z10 + z14 + z15 + z19 + z20 + z24) - 2.0 * (z2 + z7 + z12 + z17 + z22) - z1 - z3 - z6 - z8 - z11 - z13 - z16 - z18 - z21 - z23) / (35.0 * wx * wx);
  let t = zFactor * (2.0 * (z0 + z1 + z2 + z3 + z4 + z20 + z21 + z22 + z23 + z24) - 2.0 * (z10 + z11 + z12 + z13 + z14) - z5 - z6 - z7 - z8 - z9 - z15 - z16 - z17 - z18 - z19) / (35.0 * wy * wy);
  let s = zFactor * (z8 + z16 - z6 - z18 + 4.0 * (z4 + z20 - z0 - z24) + 2.0 * (z3 + z9 + z15 + z21 - z1 - z5 - z19 - z23)) / (100.0 * wx * wy);
  let p = zFactor * (44.0 * (z3 + z23 - z1 - z21) + 31.0 * (z0 + z20 - z4 - z24 + 2.0 * (z8 + z18 - z6 - z16)) + 17.0 * (z14 - z10 + 4.0 * (z13 - z11)) + 5.0 * (z9 + z19 - z5 - z15)) / (420.0 * wx);
  let q = zFactor * (44.0 * (z5 + z9 - z15 - z19) + 31.0 * (z20 + z24 - z0 - z4 + 2.0 * (z6 + z8 - z16 - z18)) + 17.0 * (z2 - z22 + 4.0 * (z7 - z17)) + 5.0 * (z1 + z3 - z21 - z23)) / (420.0 * wy);`}}for(let[e,t,n]of[[1,-1,1],[2,0,1],[3,1,1],[4,-1,0],[6,1,0],[7,-1,-1],[8,0,-1],[9,1,-1]])r(`z${e}`,t,n);let i=`zFactor * (z3 + z7 - z1 - z9) / (4.0 * wx * wy)`;return e===`zevenbergen-thorne`?{samples:n.join(`
  `),partials:`let p = zFactor * (z6 - z4) / (2.0 * wx);
  let q = zFactor * (z2 - z8) / (2.0 * wy);
  let r = zFactor * (z4 + z6) / (wx * wx);
  let t = zFactor * (z2 + z8) / (wy * wy);
  let s = ${i};`}:{samples:n.join(`
  `),partials:`let p = zFactor * (z3 + z6 + z9 - z1 - z4 - z7) / (6.0 * wx);
  let q = zFactor * (z1 + z2 + z3 - z7 - z8 - z9) / (6.0 * wy);
  let r = zFactor * (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + z8)) / (3.0 * wx * wx);
  let t = zFactor * (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + z6)) / (3.0 * wy * wy);
  let s = ${i};`}}function de(e,t){let n=[],r=[];for(let[t,i]of e.entries()){let e=Math.max(1,Math.round(i/Math.SQRT2)),a=[[i,0],[-i,0],[0,i],[0,-i],[e,e],[e,-e],[-e,e],[-e,-e]].map(([e,t])=>`readRingDelta(column, row, ${e}, ${t}, centre)`).join(` + `);r.push(`let ringSum${t} = ${a};`),n.push(`settings[settingsOffset + ${6+t}u] * (-ringSum${t}) / (${x(8*i)} * cellMetres)`)}return`${r.join(`
  `)}
  var ringValue = zFactor * (${n.join(` + `)});
  ${t===`pade-tanh`?`let squashed = clamp(ringValue, -3.0, 3.0);
  ringValue = squashed * (27.0 + squashed * squashed) / (27.0 + 9.0 * squashed * squashed);`:``}`}function fe(e,t){let{props:n,source:r,slots:i}=t,{width:a,height:o}=n.grid,s=n.method??`evans-young`,c=(n.rowDirection??`south`)===`south`?-1:1,l=(n.borderMode??`clamp`)===`clamp`,u=i.some(e=>e.kind===`curvature`)||t.hasWindowOutput&&i.some(e=>e.kind===`validity`),d=i.some(e=>e.kind===`ring`)||t.hasRingOutput&&i.some(e=>e.kind===`validity`),f=r.band,p=[{name:`elevationValues`,view:f.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:f.validity,type:`u32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`}];for(let[e,t]of i.entries())p.push({name:`output${e}`,view:t.view,type:t.kind===`validity`?`u32`:`f32`,access:`read_write`});let m=ue(s,c),h=n.ringRadii??[2,8],g=i.map((e,n)=>{let r=`output${n}`;return e.kind===`validity`?`${r}[${r}Offset + index] = select(0u, 1u, centreValid && ${[t.hasWindowOutput?`windowValid`:`true`,t.hasRingOutput?`ringValid`:`true`].join(` && `)});`:e.kind===`ring`?`${r}[${r}Offset + index] = select(invalidValue, ringValue, ringValid && centreValid && isFiniteValue(ringValue));`:`${r}[${r}Offset + index] = select(invalidValue, ${le[e.curvature]}, windowValid && centreValid);`}).join(`
  `),_=`let row = index / WIDTH;
  let column = index - row * WIDTH;
  let centre = elevationValues[elevationValuesOffset + index];
  let centreValid = elevationValidity[elevationValidityOffset + index] != 0u && isFiniteValue(centre);
  let groundCell = getGroundCellSize(row);
  let wx = groundCell.x;
  let wy = groundCell.y;
  let zFactor = settings[settingsOffset + 2u];
  let cellsValid = wx > 0.0 && wy > 0.0 && isFiniteValue(wx) && isFiniteValue(wy) && isFiniteValue(zFactor);
  let cellMetres = sqrt(wx * wy);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  windowValid = cellsValid;
  ringValid = cellsValid;
  ${u?`${m.samples}
  ${m.partials}
  let gradientSquared = p * p + q * q;
  let weight = 1.0 + gradientSquared;
  let weightRoot = sqrt(weight);
  let weightPower = weight * weightRoot;
  let flatLimit = settings[settingsOffset + 5u];
  let isFlat = gradientSquared <= flatLimit * flatLimit;
  let safeGradient = select(gradientSquared, 1.0, isFlat);
  let safeGradientPower = safeGradient * sqrt(safeGradient);
  let planNumerator = q * q * r - 2.0 * p * q * s + p * p * t;
  let profileNumerator = p * p * r + 2.0 * p * q * s + q * q * t;
  let tangential = select(-planNumerator / (safeGradient * weightRoot), 0.0, isFlat);
  let plan = select(-planNumerator / safeGradientPower, 0.0, isFlat);
  let profile = select(-profileNumerator / (safeGradient * weightPower), 0.0, isFlat);
  let meanCurvature = -((1.0 + q * q) * r - 2.0 * p * q * s + (1.0 + p * p) * t) / (2.0 * weightPower);
  let gaussian = (r * t - s * s) / (weight * weight);
  let unsphericity = sqrt(max(meanCurvature * meanCurvature - gaussian, 0.0));
  let minimal = meanCurvature - unsphericity;
  let maximal = meanCurvature + unsphericity;
  let differenceCurvature = (profile - tangential) * 0.5;
  let horizontalExcess = select(tangential - minimal, 0.0, isFlat);
  let verticalExcess = select(profile - minimal, 0.0, isFlat);
  let accumulation = tangential * profile;
  let ringProduct = horizontalExcess * verticalExcess;
  let rotor = select(((p * p - q * q) * s - p * q * (r - t)) / safeGradientPower, 0.0, isFlat);
  let laplacian = r + t;
  windowValid = windowValid && isFiniteValue(p) && isFiniteValue(q) && isFiniteValue(r) &&
    isFiniteValue(s) && isFiniteValue(t) && isFiniteValue(gradientSquared);`:``}
  ${d?de(h,n.ringSquash??`none`):``}
  ${g}`;return S(e,{id:t.id,operation:`GPUTerrainCurvature`,variant:`${s}-${n.cellSizeMode??`uniform`}-${n.borderMode??`clamp`}`,bindings:p,invocationCount:a*o,declarations:`const WIDTH: u32 = ${a}u;
const HEIGHT: u32 = ${o}u;
const WIDTH_SIGNED: i32 = ${a};
const HEIGHT_SIGNED: i32 = ${o};
var<private> windowValid: bool;
var<private> ringValid: bool;
${O}
${A(n.cellSizeMode??`uniform`,N)}
// Elevation of an offset sample minus the centre elevation; flags the window invalid on failure.
fn readWindowDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  var sampleColumn = i32(column) + dx;
  var sampleRow = i32(row) + dy;
  ${l?`sampleColumn = clamp(sampleColumn, 0, WIDTH_SIGNED - 1);
  sampleRow = clamp(sampleRow, 0, HEIGHT_SIGNED - 1);`:`if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    windowValid = false;
    return 0.0;
  }`}
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    windowValid = false;
    return 0.0;
  }
  return value - centre;
}
// Ring samples never clamp: any sample outside the raster or invalid invalidates the ring.
fn readRingDelta(column: u32, row: u32, dx: i32, dy: i32, centre: f32) -> f32 {
  let sampleColumn = i32(column) + dx;
  let sampleRow = i32(row) + dy;
  if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= WIDTH_SIGNED || sampleRow >= HEIGHT_SIGNED) {
    ringValid = false;
    return 0.0;
  }
  let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  if (elevationValidity[elevationValidityOffset + sampleIndex] == 0u || !isFiniteValue(value)) {
    ringValid = false;
    return 0.0;
  }
  return value - centre;
}`,body:_})}export{Y as a,ie as i,se as n,X as o,re as r,J as s,ce as t};