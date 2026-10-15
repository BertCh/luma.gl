import{a as e,i as t,n}from"./gpu-vector-format-zKtof1xn.js";import{C as r,E as i,M as a,R as o,S as s,T as c,j as l,k as u,w as d,x as f}from"./resources-CwvhEOBe.js";import{a as p,i as m,o as h}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{n as g,t as _}from"./gpu-scan-BuGhTLDQ.js";import{n as v,r as y,t as b}from"./gpu-chunk-utils-Bz-6HA7y.js";import{i as x,r as S,t as C}from"./gpu-spatial-utils-Cp29aAUS.js";var w=256,T=Uint32Array.BYTES_PER_ELEMENT,E=class{id;source;indices;output;wordsPerRow;constructor(e){this.id=e.id??`gpu-scatter`,this.source=e.source,this.indices=e.indices,this.output=e.output;for(let e of p(this.source))O(e,`${this.id} source`);for(let e of p(this.indices))l(e,`${this.id} indices`);for(let e of p(this.output))O(e,`${this.id} output`);if(this.source.format!==this.output.format)throw Error(`${this.id} source and output must use the same format`);if(this.source.length<this.indices.length)throw Error(`${this.id} source must contain at least indices.length rows`);let t=new Set([...p(this.source).map(e=>e.buffer),...p(this.indices).map(e=>e.buffer)]);if(p(this.output).some(e=>t.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer`);this.wordsPerRow=this.source.rowByteLength/T}getCommandNodes(e){return D(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)}};function D(e,t,n){y(t,[e.source,e.indices],[e.output]);let r=[],a=m(t,[i(t,e.source,e.indices.length),e.indices]);for(let[i,[o,s]]of a.entries()){let c=f(e.id,s.length,w,n),l=0;for(let[n,u]of p(e.output).entries())u.length&&r.push(b(t,{id:a.length===1&&p(e.output).length===1?e.id:`${e.id}-${i}-${n}`,inputs:{sourceWords:o,indices:s},outputs:{outputWords:u},dispatch:c,source:k({source:o,indices:s,output:u,wordsPerRow:e.wordsPerRow,outputStart:l},c)})),l+=u.length}return r}function O(r,i){if(e(r.format)||t(r.format))throw Error(`${i} must use a fixed-width GPU data format`);let a=n(r.format);if(r.byteStride!==a.byteLength||r.rowByteLength!==a.byteLength||r.byteOffset%T!==0||r.rowByteLength%T!==0)throw Error(`${i} must be packed, uint32-aligned GPU data with 32-bit-word rows`)}function k(e,t){return`const INDEX_COUNT: u32 = ${e.indices.length}u;
const OUTPUT_LENGTH: u32 = ${e.output.length}u;
const WORDS_PER_ROW: u32 = ${e.wordsPerRow}u;
const SOURCE_OFFSET: u32 = ${u(e.source)}u;
const INDEX_OFFSET: u32 = ${u(e.indices)}u;
const OUTPUT_OFFSET: u32 = ${u(e.output)}u;
@group(0) @binding(0) var<storage, read> sourceWords: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputWords: array<u32>;
@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${s(t,w)}
  if (index >= INDEX_COUNT) { return; }
  let destinationIndex = indices[INDEX_OFFSET + index];
  if (destinationIndex < ${e.outputStart}u || destinationIndex - ${e.outputStart}u >= OUTPUT_LENGTH) { return; }

  let sourceBase = SOURCE_OFFSET + index * WORDS_PER_ROW;
  let outputBase = OUTPUT_OFFSET + (destinationIndex - ${e.outputStart}u) * WORDS_PER_ROW;
  for (var word = 0u; word < WORDS_PER_ROW; word++) {
    outputWords[outputBase + word] = sourceWords[sourceBase + word];
  }
}`}var A=256;function j(e,t,n){y(t,[e.positions,...e.sourceIds?[e.sourceIds]:[],...e.boundsBuffer?[e.boundsBuffer]:[]],[e.cellOffsets,e.objectIds,e.count,e.overflow]);let i=[],a=v(t,e.cellOffsets,0,e.cellCount),o=r(t,`${e.id}-counts`,a),s=r(t,`${e.id}-cursors`,a),c=p(e.positions),l=c.map((n,r)=>d(t,`${e.id}-ranks-${r}`,`uint32`,n.length)),f=e.sourceIds?void 0:c.map((n,r)=>d(t,`${e.id}-ids-${r}`,`uint32`,n.length)),m=e.firstSourceIndex;for(let[r,a]of c.entries()){if(a.length){let o=l[r],s=f?.[r],c=P(a.length,n);i.push(M(t,e,{id:`${e.id}-initialize-ranks-${r}`,outputs:{ranks:o,...s?{ids:s}:{}},dispatch:c,invocationCount:a.length,source:`
@group(0) @binding(0) var<storage, read_write> ranks: array<u32>;
${s?`@group(0) @binding(1) var<storage, read_write> ids: array<u32>;`:``}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${F(c)}
  if (index >= ${a.length}u) { return; }
  ranks[index] = 0xffffffffu;
  ${s?`ids[index] = ${m}u + index;`:``}
}`}))}m+=a.length}let h=0;for(let[r,a]of o.data.entries()){let o=s.data[r],l=P(a.length,n);i.push(M(t,e,{id:`${e.id}-initialize-${r}`,outputs:{cellCounts:a,cellCursors:o},dispatch:l,invocationCount:a.length,source:`
@group(0) @binding(0) var<storage, read_write> cellCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellCursors: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${F(l)}
  if (index >= ${a.length}u) { return; }
  cellCounts[index] = 0u;
  cellCursors[index] = 0u;
}`}));for(let[o,s]of c.entries()){if(!s.length)continue;let c=P(s.length,n);i.push(M(t,e,{id:`${e.id}-count-${r}-${o}`,inputs:{positions:s,...I(e)},outputs:{cellCounts:a},dispatch:c,invocationCount:s.length,source:N(e,s,c,e=>`@group(0) @binding(${e}) var<storage, read_write> cellCounts: array<atomic<u32>>;`,`
  if (accepted && cellIndex >= ${h}u && cellIndex - ${h}u < ${a.length}u) {
    atomicAdd(&cellCounts[cellIndex - ${h}u], 1u);
  }`)}))}h+=a.length}i.push(...g(new _({id:`${e.id}-scan`,input:o,output:a}),t,n));let b=o.data[o.data.length-1],x=a.data[a.data.length-1],S=v(t,e.cellOffsets,e.cellCount,1).data[0];i.push(M(t,e,{id:`${e.id}-finalize`,inputs:{counts:b,offsets:x},outputs:{count:e.count,overflow:e.overflow},dispatch:{x:1,y:1,z:1},invocationCount:1,workgroupSize:1,source:`
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> overflow: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let total = counts[${b.length-1}u] + offsets[${u(x)+x.length-1}u];
  count[${u(e.count)}u] = total;
  overflow[${u(e.overflow)}u] = select(0u, 1u, total > ${e.objectIds.length}u);
}`})),i.push(M(t,e,{id:`${e.id}-terminal`,inputs:{count:e.count},outputs:{terminal:S},dispatch:{x:1,y:1,z:1},invocationCount:1,workgroupSize:1,source:`
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> terminal: array<u32>;
@compute @workgroup_size(1)
fn main() {
  terminal[${u(S)}u] = count[${u(e.count)}u];
}`})),h=0;for(let[r,o]of a.data.entries()){let a=s.data[r];for(let[s,d]of c.entries()){if(!d.length||!e.objectIds.length)continue;let c=P(d.length,n);i.push(M(t,e,{id:`${e.id}-rank-${r}-${s}`,inputs:{positions:d,...I(e),cellOffsets:o},outputs:{cellCursors:a,ranks:l[s]},dispatch:c,invocationCount:d.length,source:N(e,d,c,e=>`
@group(0) @binding(${e}) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(${e+1}) var<storage, read_write> cellCursors: array<atomic<u32>>;
@group(0) @binding(${e+2}) var<storage, read_write> ranks: array<u32>;`,`
  if (accepted && cellIndex >= ${h}u && cellIndex - ${h}u < ${o.length}u) {
    let cell = cellIndex - ${h}u;
    ranks[index] = cellOffsets[${u(o)}u + cell] + atomicAdd(&cellCursors[cell], 1u);
  }`)}))}h+=o.length}let C=0;for(let[r,a]of c.entries())a.length&&e.objectIds.length&&i.push(...D(new E({id:`${e.id}-scatter-${r}`,source:e.sourceIds?v(t,e.sourceIds,C,a.length):f[r],indices:l[r],output:e.objectIds}),t,n)),C+=a.length;if(t.recording)for(let t of i)t.workload={...t.workload,operation:`GPUGridIndex`,variant:t.id.slice(e.id.length+1).replace(/-\d.*$/,``),readByteLength:(t.resources??[]).reduce((e,t)=>e+(`buffer`in t&&(t.usage===`storage-read`||t.usage===`storage-read-write`)?`length`in t.buffer?t.buffer.length*t.buffer.rowByteLength:t.buffer.byteLength:0),0),writeByteLength:(t.resources??[]).reduce((e,t)=>e+(`buffer`in t&&(t.usage===`storage-write`||t.usage===`storage-read-write`)?`length`in t.buffer?t.buffer.length*t.buffer.rowByteLength:t.buffer.byteLength:0),0)};return i}function M(e,t,n){let{invocationCount:r,...i}=n,a=b(e,i),o=h(e,()=>({op:`GPUGridIndex`,moduleId:`GPUGridIndex/${n.id.slice(t.id.length+1).replace(/-\d.*$/,``)}`,source:n.source,workgroupSize:[n.workgroupSize??A,1,1],workgroups:[n.dispatch.x,n.dispatch.y,n.dispatch.z],invocationCount:r,bindings:[...Object.entries(n.inputs??{}).map(([e,t])=>({name:e,view:t,access:`read`})),...Object.entries(n.outputs).map(([e,t])=>({name:e,view:t,access:`read_write`}))]}));return o&&(a.annotation=o),a}function N(e,t,n,r,i){let a=R(e.bounds,e.boundsBuffer,e.dimension,1),o=r(e.boundsBuffer?2:1),s=e.gridSize[0],c=e.gridSize[1],l=e.dimension===3?e.gridSize[2]:1,d=e.dimension===3?`let z = positions[POSITIONS_OFFSET + index * 3u + 2u];
    let finiteZ = z == z && abs(z) <= 3.402823466e+38;
    let inZ = z >= ${a.minimum[2]} && z <= ${a.maximum[2]};`:`let finiteZ = true;
    let inZ = true;`,f=e.dimension===3?`let layer = getGridCoordinate(z, ${a.minimum[2]}, ${a.maximum[2]}, DEPTH);`:`let layer = 0u;`;return`
const ELEMENT_COUNT: u32 = ${t.length}u;
const POSITIONS_OFFSET: u32 = ${u(t)}u;
const WIDTH: u32 = ${s}u;
const HEIGHT: u32 = ${c}u;
const DEPTH: u32 = ${l}u;
${a.declarations}@group(0) @binding(0) var<storage, read> positions: array<f32>;
${o}

${C}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${F(n)}
  var accepted = false;
  var cellIndex = 0u;
  if (index < ELEMENT_COUNT) {
    let x = positions[POSITIONS_OFFSET + index * ${e.dimension}u];
    let y = positions[POSITIONS_OFFSET + index * ${e.dimension}u + 1u];
    ${d}
    let finiteXY = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= ${a.minimum[0]} && x <= ${a.maximum[0]};
    let inY = y >= ${a.minimum[1]} && y <= ${a.maximum[1]};
    if (${a.validity}finiteXY && finiteZ && inX && inY && inZ) {
      accepted = true;
      let column = getGridCoordinate(x, ${a.minimum[0]}, ${a.maximum[0]}, WIDTH);
      let row = getGridCoordinate(y, ${a.minimum[1]}, ${a.maximum[1]}, HEIGHT);
      ${f}
      cellIndex = (layer * HEIGHT + row) * WIDTH + column;
    }
  }
  ${i}
}`}function P(e,t){return f(`GPUGridIndex`,e,A,t)}function F(e){return s(e,A)}function I(e){return e.boundsBuffer?{boundsBuffer:e.boundsBuffer}:{}}function L(e){let t=e.length/2;return e.every(x)&&Array.from({length:t},(e,t)=>t).every(n=>e[n]<=e[n+t])}function R(e,t,n,r){if(!t)return{minimum:Array.from({length:n},(t,n)=>S(e[n])),maximum:Array.from({length:n},(t,r)=>S(e[r+n])),declarations:``,validity:``};let i=e=>`boundsValues[BOUNDS_OFFSET + ${e}u]`,a=Array.from({length:n},(e,t)=>i(t)),o=Array.from({length:n},(e,t)=>i(t+n)),s=a.map((e,t)=>`${e} == ${e} && ${o[t]} == ${o[t]} && abs(${e}) <= 3.402823466e+38 && abs(${o[t]}) <= 3.402823466e+38 && ${e} <= ${o[t]}`);return{minimum:a,maximum:o,declarations:`const BOUNDS_OFFSET: u32 = ${u(t)}u;
@group(0) @binding(${r}) var<storage, read> boundsValues: array<f32>;
`,validity:`${s.join(` && `)} && `}}var z=4294967295,B=class{id;positions;sourceIds;firstSourceIndex;gridSize;bounds;boundsBuffer;cellOffsets;objectIds;count;overflow;dimension;cellCount;updatePolicy=`rebuild`;constructor(e){this.id=e.id??`gpu-grid-index`,this.positions=e.positions,this.sourceIds=e.sourceIds,this.firstSourceIndex=e.firstSourceIndex??0,this.gridSize=e.gridSize,this.bounds=e.bounds,this.boundsBuffer=e.boundsBuffer,this.cellOffsets=e.cellOffsets,this.objectIds=e.objectIds,this.count=e.count,this.overflow=e.overflow,this.dimension=this.positions.format===`float32x2`?2:3;let t=this.dimension===2?`float32x2`:`float32x3`;for(let e of U(this.positions))a(e,[t],`${this.id} positions`);if(this.boundsBuffer&&(a(this.boundsBuffer,[`float32`],`${this.id} boundsBuffer`),this.boundsBuffer.length!==this.dimension*2))throw Error(`${this.id} boundsBuffer must hold ${this.dimension*2} float32 values`);if(this.gridSize.length!==this.dimension||this.bounds.length!==this.dimension*2)throw Error(`${this.id} positions, gridSize, and bounds must have matching dimensions`);if(this.gridSize.some(e=>!Number.isSafeInteger(e)||e<=0))throw Error(`${this.id} gridSize must contain positive integers`);if(this.cellCount=this.gridSize.reduce((e,t)=>e*t,1),!Number.isSafeInteger(this.cellCount)||this.cellCount>=z)throw Error(`${this.id} cell count must fit in uint32`);if(!L(this.bounds))throw Error(`${this.id} bounds must contain finite ordered minima and maxima`);for(let e of W(this.cellOffsets))l(e,`${this.id} cellOffsets`);for(let e of W(this.objectIds))l(e,`${this.id} objectIds`);if(l(this.count,`${this.id} count`),l(this.overflow,`${this.id} overflow`),this.cellOffsets.length!==this.cellCount+1)throw Error(`${this.id} cellOffsets.length must equal cellCount + 1`);if(this.count.length<1||this.overflow.length<1)throw Error(`${this.id} count and overflow must each contain one uint32 row`);if(this.positions.length>z)throw Error(`${this.id} position count must fit in uint32`);if(!Number.isSafeInteger(this.firstSourceIndex)||this.firstSourceIndex<0||this.firstSourceIndex+this.positions.length>z+1)throw Error(`${this.id} generated source IDs must fit in uint32`);if(this.sourceIds&&e.firstSourceIndex!==void 0)throw Error(`${this.id} sourceIds and firstSourceIndex are mutually exclusive`);V(this.id,this.positions,this.sourceIds),H(this.id,this.positions,this.sourceIds,this.objectIds)}getCommandNodes(e){let t=[];return t.push(...j(this,e,e.device.limits.maxComputeWorkgroupsPerDimension)),t}};function V(e,t,n){if(n){for(let t of W(n))l(t,`${e} sourceIds`);if(t.length!==n.length)throw Error(`${e} sourceIds.length must equal positions.length`)}}function H(e,t,n,r){for(let n of U(t))if(W(r).some(e=>c(n,e)))throw Error(`${e} positions and objectIds must not overlap`);for(let t of W(n))if(W(r).some(e=>c(t,e)))throw Error(`${e} sourceIds and objectIds must not overlap`)}function U(e){return e instanceof o?e.data:[e]}function W(e){return e?e instanceof o?e.data:[e]:[]}export{j as n,B as t};