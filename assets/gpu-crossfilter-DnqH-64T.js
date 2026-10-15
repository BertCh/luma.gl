import{J as e,mt as t}from"./src-CNCKFW2b.js";import{D as n,R as r,k as i,w as a}from"./resources-CwvhEOBe.js";import{u as o}from"./interleave-brZZCO-f.js";import{n as s,t as c}from"./gpu-visibility-workflow-CuWho9Ae.js";import{t as l}from"./gpu-reduction-BD8l40jM.js";import{t as u}from"./gpu-histogram-C4GJppav.js";import{t as d}from"./gpu-group-aggregation-DrNKSCi8.js";import{l as f,n as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as h}from"./wgsl-common-BNRVzgd8.js";var g=256,_=5,v=Uint32Array.BYTES_PER_ELEMENT,y=-2147483648,ee=2147483647,te=4294967295,b=class{id;dimension;mask;graph;stateValues=new Uint32Array(_);stateBuffer;stateView;ownedBuffers=[];destroyed=!1;constructor(e,n,r={}){if(this.graph=e,this.dimension=n,this.id=r.id??`gpu-crossfilter-${n.id}`,n.kind===`external`){if(n.mask.format!==`uint32`)throw Error(`${this.id} mask must contain uint32 selection flags`);w(n.mask,e,`${this.id} mask`),this.mask=n.mask;return}let i=n.kind===`range`?n.input:n.x;if(w(i,e,`${this.id} input`),n.kind===`bounds`&&(w(n.y,e,`${this.id} y input`),T(i,n.y,`${this.id} bounds inputs`)),n.mask){if(n.mask.format!==`uint32`)throw Error(`${this.id} mask must contain uint32 selection flags`);w(n.mask,e,`${this.id} mask`),T(i,n.mask,`${this.id} mask`),re(n,n.mask,this.id),this.mask=n.mask}else this.mask=x(e,this.id,i,r.persistentMask?this.ownedBuffers:void 0);let a=e.device.createBuffer({id:`${this.id}-state`,data:this.stateValues,usage:t.STORAGE|t.COPY_DST});try{let t=e.importBuffer({id:`${this.id}-state`,byteLength:a.byteLength,usage:a.usage},a);this.stateView=e.createDataView(t,{format:`uint32`,length:_}),this.stateBuffer=a}catch(e){a.destroy();for(let e of this.ownedBuffers)e.destroy();throw e}}getControlView(){return e(this.stateView),this.stateView}setRange(e){if(this.dimension.kind!==`range`)throw Error(`${this.id} does not support scalar range selection`);E(e[0],e[1],`${this.id} range`);let t=D(e[0],this.dimension.input.format,this.id),n=D(e[1],this.dimension.input.format,this.id);this.stateValues[0]=1,this.stateValues[1]=t,this.stateValues[2]=n,this.writeState()}setBounds(e){if(this.dimension.kind!==`bounds`)throw Error(`${this.id} does not support two-dimensional bounds selection`);let[t,n,r,i]=e;E(t,r,`${this.id} horizontal bounds`),E(n,i,`${this.id} vertical bounds`);let a=D(t,this.dimension.x.format,this.id),o=D(r,this.dimension.x.format,this.id),s=D(n,this.dimension.y.format,this.id),c=D(i,this.dimension.y.format,this.id);this.stateValues[0]=1,this.stateValues[1]=a,this.stateValues[2]=o,this.stateValues[3]=s,this.stateValues[4]=c,this.writeState()}clear(){this.stateValues.fill(0),this.writeState()}addToGraph(e,t){if(this.destroyed)throw Error(`${this.id} selection has been destroyed`);if(e!==this.graph)throw Error(`${this.id} selection belongs to another command graph`);if(this.dimension.kind===`external`)return;let n=this.dimension.kind===`range`?this.dimension.input:this.dimension.x,i=C(n),a=this.dimension.kind===`bounds`?C(this.dimension.y):void 0,o=C(this.mask);for(let[s,c]of i.entries()){if(c.length===0)continue;let i=n instanceof r?`${this.id}-chunk-${s}`:this.id;ne(e,{id:i,condition:t?.(i),primaryInput:c,secondaryInput:a?.[s],rejectNonFinite:this.dimension.rejectNonFinite===!0,exclusiveMaximum:this.dimension.kind===`range`&&this.dimension.exclusiveMaximum===!0,state:this.getControlView(),output:o[s]})}}destroy(){if(!this.destroyed){this.destroyed=!0,this.stateBuffer?.destroy();for(let e of this.ownedBuffers)e.destroy()}}writeState(){if(this.destroyed)throw Error(`${this.id} selection has been destroyed`);this.stateBuffer?.write(this.stateValues)}};function x(e,t,n,i){let o=(t,n)=>i?S(e,t,`uint32`,n,i):a(e,t,`uint32`,n);if(!(n instanceof r))return o(`${t}-mask`,n.length);let s=n.data.map((e,n)=>o(`${t}-mask-chunk-${n}`,e.length));return new r({id:`${t}-mask`,name:`${t}-mask`,format:`uint32`,length:n.length,valueLength:n.length,stride:1,byteStride:v,rowByteLength:v,data:s})}function S(e,n,r,i,a){let o=e.device.createBuffer({id:n,byteLength:Math.max(i,1)*v,usage:t.STORAGE|t.COPY_DST});a.push(o);let s=e.importBuffer({id:n,byteLength:o.byteLength,usage:o.usage},o);return e.createDataView(s,{format:r,length:i})}function ne(e,t){let r=O(t.primaryInput.format),a=t.secondaryInput?O(t.secondaryInput.format):void 0,s=t.secondaryInput?2:1,c=s+1,l=t.secondaryInput?`const SECONDARY_OFFSET: u32 = ${i(t.secondaryInput)}u;
@group(0) @binding(1) var<storage, read> secondaryValues: array<${a}>;`:``,u=k(t.primaryInput.format,1),d=k(t.primaryInput.format,2),f=`primaryValue >= ${u} && primaryValue ${t.exclusiveMaximum?`<`:`<=`} ${d}`,p=t.secondaryInput?` && secondaryValue >= ${k(t.secondaryInput.format,3)} &&
      secondaryValue <= ${k(t.secondaryInput.format,4)}`:``,m=[t.primaryInput.format===`float32`?`primaryFinite`:void 0,t.secondaryInput?.format===`float32`?`secondaryFinite`:void 0].filter(Boolean),h=t.rejectNonFinite&&m.length>0?`let primaryFinite = ${t.primaryInput.format===`float32`?`(bitcast<u32>(primaryValue) & 0x7f800000u) != 0x7f800000u`:`true`};
  let secondaryFinite = ${t.secondaryInput?.format===`float32`?`(bitcast<u32>(secondaryValue) & 0x7f800000u) != 0x7f800000u`:`true`};
  let finite = ${m.join(` && `)};`:`let finite = true;`,_=`
const ELEMENT_COUNT: u32 = ${t.primaryInput.length}u;
const PRIMARY_OFFSET: u32 = ${i(t.primaryInput)}u;
const STATE_OFFSET: u32 = ${i(t.state)}u;
const OUTPUT_OFFSET: u32 = ${i(t.output)}u;
@group(0) @binding(0) var<storage, read> primaryValues: array<${r}>;
${l}
@group(0) @binding(${s}) var<storage, read> selectionState: array<u32>;
@group(0) @binding(${c}) var<storage, read_write> outputMask: array<u32>;

@compute @workgroup_size(${g})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= ELEMENT_COUNT) {
    return;
  }
  let primaryValue = primaryValues[PRIMARY_OFFSET + index];
  ${t.secondaryInput?`let secondaryValue = secondaryValues[SECONDARY_OFFSET + index];`:``}
  ${h}
  if (selectionState[STATE_OFFSET] == 0u) {
    outputMask[OUTPUT_OFFSET + index] = select(0u, 1u, finite);
    return;
  }
  let accepted = finite && ${f}${p};
  outputMask[OUTPUT_OFFSET + index] = select(0u, 1u, accepted);
}`,v=[{buffer:t.primaryInput,usage:`storage-read`},...t.secondaryInput?[{buffer:t.secondaryInput,usage:`storage-read`}]:[],{buffer:t.state,usage:`storage-read`},{buffer:t.output,usage:`storage-write`}],y={primaryValues:t.primaryInput,...t.secondaryInput?{secondaryValues:t.secondaryInput}:{},selectionState:t.state,outputMask:t.output};e.addComputePass({id:t.id,resources:v,condition:t.condition,compile:({device:e})=>{let r=new o(e,{id:t.id,source:_,shaderLayout:{bindings:Object.keys(y).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,t]of Object.entries(y))a[e]=n(t,i);r.setBindings(a),r.dispatch(e,Math.ceil(t.primaryInput.length/g))},destroy:()=>r.destroy()}}})}function C(e){return e instanceof r?e.data:[e]}function w(e,t,n){if(![`float32`,`sint32`,`uint32`].includes(e.format))throw Error(`${n} must contain packed 32-bit scalar GPU data`);for(let r of C(e))if(r.buffer.graph!==t||r.format!==e.format||r.byteStride!==v||r.rowByteLength!==v||r.byteOffset%v!==0)throw Error(`${n} must contain packed graph-local 32-bit scalar GPU data`)}function T(e,t,n){if(e instanceof r!=t instanceof r)throw Error(`${n} must use the same view kind`);let i=C(e),a=C(t);if(e.length!==t.length||i.length!==a.length||i.some((e,t)=>e.length!==a[t].length))throw Error(`${n} must preserve the same chunk topology`)}function re(e,t,n){let r=e.kind===`range`?C(e.input):[...C(e.x),...C(e.y)];if(C(t).some(e=>r.some(t=>t.buffer===e.buffer)))throw Error(`${n} input and mask must use separate buffers`)}function E(e,t,n){if(!Number.isFinite(e)||!Number.isFinite(t)||e>t)throw Error(`${n} must contain ordered finite endpoints`)}function D(e,t,n){if(t===`float32`){if(!Number.isFinite(Math.fround(e)))throw Error(`${n} selection endpoints must fit their input scalar format`);let t=new ArrayBuffer(v);return new Float32Array(t)[0]=e,new Uint32Array(t)[0]}let r=t===`sint32`?y:0,i=t===`sint32`?ee:te;if(!Number.isInteger(e)||e<r||e>i)throw Error(`${n} selection endpoints must fit their input scalar format`);return e>>>0}function O(e){switch(e){case`float32`:return`f32`;case`sint32`:return`i32`;case`uint32`:return`u32`}}function k(e,t){let n=`selectionState[STATE_OFFSET + ${t}u]`;switch(e){case`float32`:return`bitcast<f32>(${n})`;case`sint32`:return`bitcast<i32>(${n})`;case`uint32`:return n}}var A=2048,j=256,M=`0xffffffffu`;function N(e,t){return e*t}function P(e,t){let n=t.active.binning.binCount,r=t.linked.binning.binCount,i=N(n,r),a=[],o=(e,n=`read_write`)=>({name:`cube`,view:t.cube,type:e,access:n});a.push(m(e,{id:`${t.id}/clear`,operation:`GPUCrossfilterHistogramCube`,variant:`clear`,bindings:[o(`u32`)],invocationCount:i,condition:t.createCondition(`${t.id}/clear`),body:`cube[cubeOffset + index] = 0u;`}));let s=I(`active`,t.active.binning),c=I(`linked`,t.linked.binning),l=i<=A;for(let[n,u]of t.active.chunks.entries()){if(u.length===0)continue;let d=t.linked.chunks[n],f=t.masks?.[n],p=`${t.id}/accumulate-${n}`,g=Math.ceil(i/j),_=l?`
  let lane = localInvocationIndex;
  for (var round = 0u; round < ${g}u; round = round + 1u) {
    let cell = round * ${j}u + lane;
    if (cell < CELL_COUNT) { atomicStore(&localCells[cell], 0u); }
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    ${F(f!==void 0,`atomicAdd(&localCells[cellIndex], 1u);`)}
  }
  workgroupBarrier();
  for (var round = 0u; round < ${g}u; round = round + 1u) {
    let cell = round * ${j}u + lane;
    if (cell < CELL_COUNT) {
      let count = atomicLoad(&localCells[cell]);
      if (count != 0u) { atomicAdd(&cube[cubeOffset + cell], count); }
    }
  }`:`
  if (index < INVOCATION_COUNT) {
    ${F(f!==void 0,`atomicAdd(&cube[cubeOffset + cellIndex], 1u);`)}
  }`;a.push(m(e,{id:p,operation:`GPUCrossfilterHistogramCube`,variant:l?`accumulate-local`:`accumulate-global`,bindings:[{name:`activeValues`,view:u,type:z(t.active.binning.format),access:`read`},{name:`linkedValues`,view:d,type:z(t.linked.binning.format),access:`read`},...f?[{name:`rowMask`,view:f,type:`u32`,access:`read`}]:[],o(`atomic<u32>`),...s.bindings,...c.bindings],invocationCount:u.length,workgroupSize:j,guardIndex:!1,condition:t.createCondition(p),declarations:`
${h}
const CELL_COUNT: u32 = ${i}u;
const LINKED_BIN_COUNT: u32 = ${r}u;
const ACTIVE_STRIDE: u32 = ${L(u)}u;
const LINKED_STRIDE: u32 = ${L(d)}u;
${f?`const MASK_STRIDE: u32 = ${L(f)}u;`:``}
${s.declarations}
${c.declarations}
${l?`var<workgroup> localCells: array<atomic<u32>, ${i}>;`:``}`,body:_}))}let u=`${t.id}/prefix`;return a.push(m(e,{id:u,operation:`GPUCrossfilterHistogramCube`,variant:`prefix`,bindings:[o(`u32`)],invocationCount:r,condition:t.createCondition(u),declarations:`
const ACTIVE_BIN_COUNT: u32 = ${n}u;
const LINKED_BIN_COUNT: u32 = ${r}u;`,body:`
  var running = 0u;
  for (var activeBin = 0u; activeBin < ACTIVE_BIN_COUNT; activeBin = activeBin + 1u) {
    let cell = cubeOffset + activeBin * LINKED_BIN_COUNT + index;
    running = running + cube[cell];
    cube[cell] = running;
  }`})),a}function ie(e,t){let n=t.active.binning.binCount,r=t.linked.binning.binCount,i=t.active.binning.format,a=z(i),o=I(`active`,t.active.binning),s=e=>i===`float32`?`bitcast<f32>(brushState[brushStateOffset + ${e}u])`:i===`sint32`?`bitcast<i32>(brushState[brushStateOffset + ${e}u])`:`brushState[brushStateOffset + ${e}u]`;return m(e,{id:`${t.id}/readout`,operation:`GPUCrossfilterHistogramCube`,variant:`readout`,bindings:[{name:`cube`,view:t.cube,type:`u32`,access:`read`},{name:`brushState`,view:t.brushState,type:`u32`,access:`read`},{name:`outputCounts`,view:t.output,type:`u32`,access:`read_write`},...o.bindings],invocationCount:j,workgroupSize:j,condition:t.condition,declarations:`
${h}
const ACTIVE_BIN_COUNT: u32 = ${n}u;
const LINKED_BIN_COUNT: u32 = ${r}u;
${o.declarations}`,body:`
  let brushMinimum: ${a} = ${s(1)};
  let brushMaximum: ${a} = ${s(2)};
  let domainMinimum = activeMinimum();
  let domainMaximum = activeMaximum();
  var first = 0u;
  var last = ACTIVE_BIN_COUNT - 1u;
  var empty = !(domainMaximum >= domainMinimum) || brushMaximum < domainMinimum ||
    brushMinimum > domainMaximum;
  if (!empty) {
    if (brushMinimum > domainMinimum) { first = activeBin(brushMinimum); }
    if (brushMaximum < domainMaximum) { last = activeBin(brushMaximum); }
    empty = first == ${M} || last == ${M} || first > last;
  }
  for (var bin = index; bin < LINKED_BIN_COUNT; bin = bin + ${j}u) {
    var total = 0u;
    if (!empty) {
      total = cube[cubeOffset + last * LINKED_BIN_COUNT + bin];
      if (first > 0u) {
        total = total - cube[cubeOffset + (first - 1u) * LINKED_BIN_COUNT + bin];
      }
    }
    outputCounts[outputCountsOffset + bin] = total;
  }`})}function F(e,t){return`${e?`if (rowMask[rowMaskOffset + index * MASK_STRIDE] != 0u) {`:`{`}
      let activeBinIndex = activeBin(activeValues[activeValuesOffset + index * ACTIVE_STRIDE]);
      let linkedBinIndex = linkedBin(linkedValues[linkedValuesOffset + index * LINKED_STRIDE]);
      if (activeBinIndex != ${M} && linkedBinIndex != ${M}) {
        let cellIndex = activeBinIndex * LINKED_BIN_COUNT + linkedBinIndex;
        ${t}
      }
    }`}function I(e,t){let{format:n,binCount:r}=t,i=R(n),a=[],o=n===`float32`?`if (!isFiniteBits(value)) { return 0xffffffffu; }`:``,s=`${e.toUpperCase()}_BINS`;if(t.edges!==void 0){let c=t.edges,l;if(Array.isArray(c)){let t=c.map(e=>B(e,n));l=`const ${e}Edges = array<${i}, ${t.length}>(${t.join(`, `)});
fn ${e}Edge(index: u32) -> ${i} { return ${e}Edges[index]; }`}else a.push({name:`${e}EdgeValues`,view:c,type:z(n),access:`read`}),l=`fn ${e}Edge(index: u32) -> ${i} { return ${e}EdgeValues[${e}EdgeValuesOffset + index]; }`;return{bindings:a,declarations:`
const ${s}: u32 = ${r}u;
${l}
fn ${e}Minimum() -> ${i} { return ${e}Edge(0u); }
fn ${e}Maximum() -> ${i} { return ${e}Edge(${s}); }
fn ${e}Bin(value: ${i}) -> u32 {
  ${o}
  if (!(value >= ${e}Minimum() && value <= ${e}Maximum())) { return 0xffffffffu; }
  var lower = 0u;
  var upper = ${s};
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (value < ${e}Edge(middle + 1u)) {
      upper = middle;
    } else {
      lower = middle + 1u;
    }
  }
  return min(lower, ${s} - 1u);
}`}}let c=t.domain,l;Array.isArray(c)?l=`fn ${e}Minimum() -> ${i} { return ${B(c[0],n)}; }
fn ${e}Maximum() -> ${i} { return ${B(c[1],n)}; }`:(a.push({name:`${e}DomainValues`,view:c,type:z(n),access:`read`}),l=`fn ${e}Minimum() -> ${i} { return ${e}DomainValues[${e}DomainValuesOffset]; }
fn ${e}Maximum() -> ${i} { return ${e}DomainValues[${e}DomainValuesOffset + 1u]; }`);let u=32-Math.clz32(r),d=n===`float32`?``:`fn ${e}MultiplyDivideFloor(numerator: u32, multiplier: u32, denominator: u32) -> u32 {
  var quotient = 0u;
  var remainder = 0u;
  var bitIndex = ${u}u;
  loop {
    bitIndex = bitIndex - 1u;
    let doubledThreshold = denominator - remainder;
    if (remainder >= doubledThreshold) {
      remainder = remainder - doubledThreshold;
      quotient = quotient * 2u + 1u;
    } else {
      remainder = remainder * 2u;
      quotient = quotient * 2u;
    }
    if (((multiplier >> bitIndex) & 1u) != 0u) {
      let additionThreshold = denominator - numerator;
      if (remainder >= additionThreshold) {
        remainder = remainder - additionThreshold;
        quotient = quotient + 1u;
      } else {
        remainder = remainder + numerator;
      }
    }
    if (bitIndex == 0u) { break; }
  }
  return quotient;
}`,f=n===`float32`?`let ratio = (value - minimum) / (maximum - minimum);
  return min(u32(ratio * f32(${s})), ${s} - 1u);`:n===`uint32`?`return ${e}MultiplyDivideFloor(value - minimum, ${s}, maximum - minimum);`:`let orderedValue = bitcast<u32>(value) ^ 0x80000000u;
  let orderedMinimum = bitcast<u32>(minimum) ^ 0x80000000u;
  let orderedMaximum = bitcast<u32>(maximum) ^ 0x80000000u;
  return ${e}MultiplyDivideFloor(
    orderedValue - orderedMinimum, ${s}, orderedMaximum - orderedMinimum);`;return{bindings:a,declarations:`
const ${s}: u32 = ${r}u;
${l}
${d}
fn ${e}Bin(value: ${i}) -> u32 {
  let minimum = ${e}Minimum();
  let maximum = ${e}Maximum();
  if (!(maximum >= minimum)) { return 0xffffffffu; }
  ${o}
  if (value < minimum || value > maximum) { return 0xffffffffu; }
  if (maximum == minimum) { return 0u; }
  if (value == maximum) { return ${s} - 1u; }
  ${f}
}`}}function L(e){return e.byteStride/Uint32Array.BYTES_PER_ELEMENT}function R(e){return e===`float32`?`f32`:e===`sint32`?`i32`:`u32`}function z(e){return R(e)}function B(e,t){return t===`uint32`?`${e}u`:t===`sint32`?`${e}`:f(e)}var V=e=>`selection:${e}`,H=e=>`content:${e}`,U=e=>`view:${e}`,W=`live`,G=`all`,K=`cube`,ae=class{id;graph;dimensions;views;mask;selections=new Map;viewMasks=new Map;excludedDimensionMasks=new Map;liveMask;skipUnchanged;cubesEnabled;ownedBuffers=[];generations=new Map;brushActive=new Map;activeViewId;servedViewIds=new Set;suppressedSelectionKey;maySuppressMasks;addedToGraph=!1;destroyed=!1;constructor(e,t){if(this.id=t.id??`gpu-crossfilter`,this.graph=e,this.dimensions=t.dimensions,this.views=t.views??[],this.skipUnchanged=t.skipUnchangedDimensions===!0,this.cubesEnabled=t.histogramCubes===!0,this.maySuppressMasks=this.cubesEnabled&&t.outputMask===void 0&&this.dimensions.every(e=>e.mask===void 0)&&this.views.every(e=>e.kind===`histogram`),this.dimensions.length===0)throw Error(`${this.id} requires at least one selection dimension`);try{for(let t of this.dimensions){if(this.selections.has(t.id))throw Error(`${this.id} selection dimensions require unique identifiers`);let n=this.selections.values().next().value,r=new b(e,t,{id:`${this.id}-${t.id}`,persistentMask:this.skipUnchanged});this.selections.set(t.id,r),n&&$(n.mask,r.mask,`${this.id} ${t.id}`)}se(this.views,this.selections,this.id),t.liveMask&&($(this.selections.values().next().value.mask,t.liveMask,`${this.id} liveMask`),this.liveMask=t.liveMask);let n=this.selections.values().next().value.mask;this.mask=t.outputMask??this.createMaskLike(`${this.id}-mask`,n),$(n,this.mask,`${this.id} output mask`),this.cubesEnabled&&this.validateCubeMemory()}catch(e){for(let e of this.selections.values())e.destroy();for(let e of this.ownedBuffers)e.destroy();throw e}}setRange(e,t){return this.getSelection(e).setRange(t),this.noteSelectionChange(e,!0),this}setBounds(e,t){return this.getSelection(e).setBounds(t),this.noteSelectionChange(e,!0),this}setExternalActive(e,t){if(this.getSelection(e).dimension.kind!==`external`)throw Error(`${this.id} dimension "${e}" is not external`);return this.noteSelectionChange(e,t),this}clear(e){return this.getSelection(e).clear(),this.noteSelectionChange(e,!1),this}clearAll(){this.assertAvailable();for(let[e,t]of this.selections)t.clear(),this.bumpGeneration(V(e)),this.brushActive.set(e,!1);return this.updateCubeState(),this}invalidate(e){return this.assertAvailable(),e===void 0?this.bumpGeneration(G):(this.getSelection(e),this.bumpGeneration(V(e)),this.bumpGeneration(H(e))),this}activateView(e){if(this.assertAvailable(),!this.cubesEnabled)throw Error(`${this.id} activateView requires histogramCubes: true`);let t=this.views.find(t=>t.id===e);if(!t||!this.isCubeCandidate(t))throw Error(`${this.id} view "${e}" cannot be the active cube view`);return this.activeViewId=e,this.bumpGeneration(K),this.updateCubeState(),this}deactivateView(){return this.assertAvailable(),this.activeViewId=void 0,this.bumpGeneration(K),this.updateCubeState(),this}getDimensionMask(e){return this.getSelection(e).mask}getViewMask(e){if(this.assertAvailable(),!this.views.some(t=>t.id===e))throw Error(`${this.id} does not contain view "${e}"`);if(!this.addedToGraph)throw Error(`${this.id} view masks are available after addToGraph()`);return this.viewMasks.get(e)}addToGraph(e=this.graph){if(this.assertAvailable(),e!==this.graph)throw Error(`${this.id} selections must be added to their owning graph`);if(this.addedToGraph)throw Error(`${this.id} can only be added to its graph once`);for(let[t,n]of this.selections)n.addToGraph(e,e=>this.createCondition(e,[V(t),H(t)]));this.composeMasks(`${this.id}/controller/compose`,[...Array.from(this.selections.values(),e=>e.mask),...this.liveMask?[this.liveMask]:[]],this.mask,this.getMaskDependencies());for(let e of this.views){let t=this.getEffectiveMask(e);switch(this.viewMasks.set(e.id,t),e.kind){case`histogram`:this.addHistogramView(e,t);break;case`group`:this.addGroupView(e,t);break;case`visibility`:this.addVisibilityView(e,t);break;case`mask`:this.addMaskView(e,t);break;case`count`:this.addCountView(e,t);break}}this.cubesEnabled&&this.addHistogramCubes(),this.addedToGraph=!0}destroy(){if(!this.destroyed){for(let e of this.selections.values())e.destroy();for(let e of this.ownedBuffers)e.destroy();this.destroyed=!0}}getSelection(e){this.assertAvailable();let t=this.selections.get(e);if(!t)throw Error(`${this.id} does not contain selection dimension "${e}"`);return t}noteSelectionChange(e,t){this.bumpGeneration(V(e)),this.brushActive.set(e,t),this.updateCubeState()}bumpGeneration(e){this.generations.set(e,(this.generations.get(e)??0)+1)}createCondition(e,t,n={}){let r=n.track??this.skipUnchanged,i=this.maySuppressMasks&&t.some(e=>e.startsWith(`selection:`));if(!r&&!i&&!n.gate)return;let a=new Map,o=!1;return{id:`${e}/skip`,source:`cpu`,evaluate:()=>{if(n.gate&&!n.gate()||this.suppressedSelectionKey!==void 0&&t.includes(this.suppressedSelectionKey))return!1;if(!r)return!0;let e=!o;o=!0;for(let n of[G,...t]){let t=this.generations.get(n)??0;a.get(n)!==t&&(a.set(n,t),e=!0)}return e}}}addConditioned(e,t,n){oe(this.graph,e,e=>this.createCondition(e,t,n))}getMaskDependencies(e){let t=this.dimensions.filter(t=>t.id!==e).map(e=>V(e.id));return this.liveMask&&t.push(W),t}getViewDependencies(e){let t=this.getMaskDependencies(q(e)?e.dimension:void 0);return e.kind===`histogram`&&(e.dimension!==void 0&&t.push(H(e.dimension)),t.push(U(e.id))),t}createMaskLike(e,t){return Q(this.graph,e,t,this.skipUnchanged?this.ownedBuffers:void 0)}getEffectiveMask(e){if(!q(e)||!e.dimension)return this.mask;if(this.excludedDimensionMasks.has(e.dimension))return this.excludedDimensionMasks.get(e.dimension);let t=Array.from(this.selections.values()).filter(t=>t.dimension.id!==e.dimension).map(e=>e.mask);if(this.liveMask&&t.push(this.liveMask),t.length===0){this.excludedDimensionMasks.set(e.dimension,void 0);return}let n=this.createMaskLike(`${this.id}/controller/mask/without/${e.dimension}`,this.mask);return this.composeMasks(`${this.id}/controller/compose-without/${e.dimension}`,t,n,this.getMaskDependencies(e.dimension)),this.excludedDimensionMasks.set(e.dimension,n),n}addHistogramView(e,t){let n=`${this.id}/view/${e.id.length}:${e.id}`,r=this.getViewDependencies(e),i=this.cubesEnabled?{gate:()=>!this.servedViewIds.has(e.id)}:void 0;if(e.edges!==void 0){this.addConditioned(new u({id:n,input:e.input,output:e.output,edges:e.edges,mask:t}),r,i);return}let o=e.domain;if(o===`auto`&&this.liveMask){let t=a(this.graph,`${n}-live-extent`,e.input.format,2);this.addConditioned(new l({id:`${n}/extent`,input:e.input,mask:this.liveMask,output:t,operation:`extent`}),r,i),o=t}this.addConditioned(new u({id:n,input:e.input,output:e.output,domain:o,mask:t}),r,i)}addCountView(e,t){let n=this.getViewDependencies(e),r=t??this.createAllRowsMask(e.id,n);this.viewMasks.set(e.id,r),this.addConditioned(new l({id:`${this.id}/view/${e.id.length}:${e.id}`,input:r,output:e.output,operation:`sum`}),n)}composeMasks(e,t,n,r,i){let a=this.dimensions.filter(e=>e.kind===`external`&&r.includes(V(e.id))&&t.includes(e.mask));if(a.length>0){let o=t.filter(e=>!a.some(t=>t.mask===e)),c;o.length>0&&(c=Q(this.graph,`${e}/ordinary`,n),this.composeMasks(`${e}/ordinary`,o,c,r,i));for(let[t,o]of a.entries()){let l=`${e}/external-${t}`,u=t===a.length-1?n:Q(this.graph,l,n),d={...i,gate:()=>(!i?.gate||i.gate())&&this.brushActive.get(o.id)===!0},f={...i,gate:()=>(!i?.gate||i.gate())&&this.brushActive.get(o.id)!==!0};if(this.addConditioned(new s({id:`${l}/active`,inputs:c?[c,o.mask]:[o.mask],output:u}),r,d),c)this.addConditioned(new s({id:`${l}/inactive`,inputs:[c],output:u}),r,f);else for(let[e,t]of J(u).entries())t.length&&this.addConditioned(p(this.graph,{id:`${l}/inactive-${e}`,operation:`GPUCrossfilter`,view:t,type:`u32`,value:`1u`}),r,f);c=u}return}let o=Math.max(2,(this.graph.device.limits.maxStorageBuffersPerShaderStage||8)-1),c=t,l=0;for(;c.length>o;){let t=[];for(let a=0;a<c.length;a+=o){let u=c.slice(a,a+o);if(u.length===1){t.push(u[0]);continue}let d=Q(this.graph,`${e}/partial-${l}-${a}`,n);this.addConditioned(new s({id:`${e}/partial-${l}-${a}`,inputs:u,output:d}),r,i),t.push(d)}c=t,l++}this.addConditioned(new s({id:e,inputs:c,output:n}),r,i)}addGroupView(e,t){let n=`${this.id}/view/${e.id.length}:${e.id}`,r=this.getViewDependencies(e);if(!Z(e)){this.addConditioned(new d({id:n,keys:e.keys,output:e.output,operation:`count`,mask:t}),r);return}this.addConditioned(new d({id:n,keys:e.keys,values:e.values,output:e.output,operation:e.operation,mask:t}),r)}addVisibilityView(e,t){let n=this.getViewDependencies(e),r=t??this.createAllRowsMask(e.id,n);this.viewMasks.set(e.id,r),this.addConditioned(new c({id:`${this.id}/view/${e.id.length}:${e.id}`,predicates:[{kind:`selection`,mask:r}],output:e.output,count:e.count,outputMask:e.outputMask,sourceIds:e.sourceIds,firstSourceIndex:e.firstSourceIndex}),n)}addMaskView(e,t){let n=this.getViewDependencies(e),r=t??this.createAllRowsMask(e.id,n);r!==e.output&&this.addConditioned(new s({id:`${this.id}/view/${e.id.length}:${e.id}`,inputs:[r],output:e.output}),n),this.viewMasks.set(e.id,e.output)}createAllRowsMask(e,t){let n=`${this.id}/controller/mask/view/${e}/all`,r=`${this.id}/controller/view/${e}/all`,i=this.selections.values().next().value.mask;if(this.liveMask){let e=this.createMaskLike(n,i);return this.addConditioned(new s({id:r,inputs:[this.liveMask],output:e}),t),e}let a=Q(this.graph,`${n}-inverse`,i),o=this.createMaskLike(n,i);return this.addConditioned(new s({id:`${r}/not`,inputs:[i],output:a,operation:`not`}),t),this.addConditioned(new s({id:`${r}/compose`,inputs:[i,a],output:o,operation:`or`}),t),o}isCubeCandidate(e){if(e.kind!==`histogram`||e.dimension===void 0)return!1;let t=this.dimensions.find(t=>t.id===e.dimension);return t?.kind===`range`&&t.input===e.input&&t.exclusiveMaximum!==!0}isCubeLinked(e,t){return t.kind===`histogram`&&t.id!==e.id&&!(t.dimension===e.dimension&&q(t))}validateCubeMemory(){let{maxStorageBufferBindingSize:e,maxBufferSize:t}=this.graph.device.limits;for(let n of this.views)if(this.isCubeCandidate(n))for(let r of this.views){if(!this.isCubeLinked(n,r))continue;let i=N(n.output.length,r.output.length)*Uint32Array.BYTES_PER_ELEMENT;if(i>e||i>t)throw Error(`${this.id} cube "${n.id}" x "${r.id}" needs ${i} bytes, above the device limit`)}}updateCubeState(){if(!this.cubesEnabled)return;let e=new Set,t,n=this.views.find(e=>e.id===this.activeViewId);if(n&&this.isCubeCandidate(n)&&n.dimension!==void 0&&this.brushActive.get(n.dimension)){for(let t of this.views)this.isCubeLinked(n,t)&&e.add(t.id);let r=this.views.every(t=>e.has(t.id)||t.dimension===n.dimension&&q(t));this.maySuppressMasks&&r&&(t=V(n.dimension))}for(let t of this.views)e.has(t.id)!==this.servedViewIds.has(t.id)&&this.bumpGeneration(U(t.id));this.servedViewIds=e,this.suppressedSelectionKey=t}addHistogramCubes(){let e=this.views.filter(e=>e.kind===`histogram`),t=e.filter(e=>this.isCubeCandidate(e));if(t.length===0)return;let n=new Map;for(let t of e){if(t.edges!==void 0||t.domain!==`auto`)continue;let e=S(this.graph,`${this.id}/cube/${t.id.length}:${t.id}/extent`,t.input.format,2,this.ownedBuffers);n.set(t.id,e),this.addConditioned(new l({id:`${this.id}/cube/${t.id.length}:${t.id}/extent`,input:t.input,...this.liveMask?{mask:this.liveMask}:{},output:e,operation:`extent`}),[...this.liveMask?[W]:[],...t.dimension===void 0?[]:[H(t.dimension)]],{track:!0,gate:()=>this.activeViewId!==void 0})}let r=new Map;for(let n of t)for(let t of e)this.isCubeLinked(n,t)&&r.set(t.id,Math.max(r.get(t.id)??0,N(n.output.length,t.output.length)));let i=new Map;for(let[e,t]of r)i.set(e,S(this.graph,`${this.id}/cube/${e.length}:${e}/cells`,`uint32`,t,this.ownedBuffers));for(let r of t)for(let t of e)this.isCubeLinked(r,t)&&this.addHistogramCubePair(r,t,i.get(t.id),n)}addHistogramCubePair(e,t,n,r){let i=e.dimension,a=`${this.id}/cube/${e.id.length}:${e.id}/${t.id.length}:${t.id}`,o=new Set([i]);t.dimension!==void 0&&q(t)&&o.add(t.dimension);let s=this.dimensions.filter(e=>!o.has(e.id)),c=[...s.map(e=>this.selections.get(e.id).mask),...this.liveMask?[this.liveMask]:[]],l=[...s.map(e=>V(e.id)),...this.liveMask?[W]:[],K,H(i),...t.dimension===void 0?[]:[H(t.dimension)]],u={track:!0,gate:()=>this.activeViewId===e.id},d;c.length===1&&!s.some(e=>e.kind===`external`)?d=c[0]:c.length>0&&(d=Q(this.graph,`${a}/mask`,this.mask),this.composeMasks(`${a}/compose`,c,d,l,u));let f={chunks:J(e.input),binning:Y(e,r)},p={chunks:J(t.input),binning:Y(t,r)};for(let e of P(this.graph,{id:a,active:f,linked:p,cube:n,masks:d?J(d):void 0,createCondition:e=>this.createCondition(e,l,u)}))this.graph.add(e);this.graph.add(ie(this.graph,{id:a,active:f,linked:p,cube:n,brushState:this.selections.get(i).getControlView(),output:t.output,condition:this.createCondition(`${a}/readout`,[],{track:!1,gate:()=>this.activeViewId===e.id&&this.servedViewIds.has(t.id)})}))}assertAvailable(){if(this.destroyed)throw Error(`${this.id} has been destroyed`)}};function q(e){return e.dimension!==void 0&&(e.includeOwnSelection===!1||e.includeOwnSelection!==!0&&(e.kind===`histogram`||e.kind===`group`))}function J(e){return e instanceof r?e.data:[e]}function Y(e,t){let n={format:e.input.format,binCount:e.output.length};return e.edges===void 0?{...n,domain:e.domain===`auto`?t.get(e.id):e.domain}:{...n,edges:e.edges}}function oe(e,t,n){for(let r of X(e,t)){if(r.condition||r.publication)throw Error(`${r.id} cannot be skipped by a crossfilter`);let t=n(r.id);e.add(t?{...r,condition:t}:r)}}function X(e,t){return Array.isArray(t)?t.flatMap(t=>X(e,t)):`getNodes`in t?t.getNodes().flatMap(t=>X(e,t)):`getCommandNodes`in t?[...t.getCommandNodes(e)]:[t]}function Z(e){return e.operation!==void 0&&e.operation!==`count`}function Q(e,t,n,i){let o=(t,n)=>i?S(e,t,`uint32`,n,i):a(e,t,`uint32`,n);if(!(n instanceof r))return o(t,n.length);let s,c=n.data.map((e,n)=>e.length===0?(s??=o(`${t}-empty`,0),s):o(`${t}-chunk-${n}`,e.length));return new r({id:t,name:t,format:`uint32`,length:n.length,valueLength:n.length,stride:1,byteStride:Uint32Array.BYTES_PER_ELEMENT,rowByteLength:Uint32Array.BYTES_PER_ELEMENT,data:c})}function $(e,t,n){if(e instanceof r!=t instanceof r)throw Error(`${n} must preserve the same view kind`);if(e.length!==t.length)throw Error(`${n} must preserve the same row count`);if(e instanceof r&&t instanceof r&&(e.data.length!==t.data.length||e.data.some((e,n)=>e.length!==t.data[n].length)))throw Error(`${n} must preserve the same chunk topology`)}function se(e,t,n){let r=t.values().next().value?.mask,i=new Set;for(let a of e){if(i.has(a.id))throw Error(`${n} linked views require unique identifiers`);if(i.add(a.id),a.dimension!==void 0&&!t.has(a.dimension))throw Error(`${n} view "${a.id}" references an unknown dimension`);if(r)switch(a.kind){case`histogram`:$(r,a.input,`${n} view "${a.id}" input`);break;case`group`:$(r,a.keys,`${n} view "${a.id}" keys`),Z(a)&&$(r,a.values,`${n} view "${a.id}" values`);break;case`mask`:$(r,a.output,`${n} view "${a.id}" output`);break;case`count`:break;case`visibility`:a.outputMask&&$(r,a.outputMask,`${n} view "${a.id}" output mask`),a.sourceIds&&$(r,a.sourceIds,`${n} view "${a.id}" source IDs`);break}}}export{ae as t};