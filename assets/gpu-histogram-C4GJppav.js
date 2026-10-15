import{D as e,I as t,M as n,R as r,S as i,j as a,k as o,w as s,x as c,z as l}from"./resources-CwvhEOBe.js";import{i as u,n as d,o as f,r as p,t as m}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{t as h}from"./gpu-reduction-BD8l40jM.js";var g=256,_=256,v=16,y=257,b=Uint32Array.BYTES_PER_ELEMENT,x=[`uint32`,`sint32`,`float32`],S=class{id;input;output;mask;domain;edges;constructor(e){this.id=e.id??`gpu-histogram`,this.input=e.input,this.output=e.output,this.mask=e.mask,this.domain=e.domain,this.edges=e.edges;for(let[e,t]of C(this.input).entries()){let n=this.input instanceof r?` input chunk ${e}`:` input`;if(T(t,`${this.id}${n}`),t.format!==this.input.format)throw Error(`${this.id}${n} format must match the input format`)}if(a(this.output,`${this.id} output`),this.output.length===0)throw Error(`${this.id} output must contain at least one bin`);if(C(this.input).some(e=>e.buffer===this.output.buffer))throw Error(`${this.id} input and output must use separate buffers`);if(this.mask){for(let[e,t]of w(this.mask).entries())if(a(t,`${this.id} mask chunk ${e}`),t.buffer===this.output.buffer)throw Error(`${this.id} mask and output must use separate buffers`);if(this.input.length!==this.mask.length)throw Error(`${this.id} input and mask lengths must match`)}if(this.edges!==void 0)N(this.edges,this.input.format,this.output,this.id);else if(this.domain===`auto`&&C(this.input).some(e=>e.byteStride!==e.rowByteLength))throw Error(`${this.id} automatic domains require packed input; provide an explicit domain`);else if(F(this.domain)){if(n(this.domain,x,`${this.id} domain`),this.domain.format!==this.input.format||this.domain.length!==2)throw Error(`${this.id} GPU domain must contain two ${this.input.format} rows`);if(this.domain.buffer===this.output.buffer)throw Error(`${this.id} domain and output must use separate buffers`)}else Array.isArray(this.domain)&&P(this.domain,this.input.format,this.id)}getCommandNodes(e){let t=[],n=C(this.input),i=this.mask?w(this.mask):void 0;if(n.some(t=>t.buffer.graph!==e)||i?.some(t=>t.buffer.graph!==e)||this.output.buffer.graph!==e)throw Error(`${this.id} views must belong to the target graph`);if(this.mask){let t=u(e,[this.input,this.mask]);n=t.map(([e])=>e),i=t.map(([,e])=>e)}let a=this.domain,o=this.edges;if(I(o)&&o.buffer.graph!==e)throw Error(`${this.id} edges must belong to the target graph`);if(o!==void 0){let a=I(o)?s(e,`${this.id}-edge-validity`,`uint32`,1):void 0;I(o)&&a&&t.push(...O(e,this.id,o,a)),t.push(...D(e,this.id,this.output));let c=this.output.length<=_?`local`:`global`;return n.forEach((s,l)=>{s.length!==0&&t.push(...k(e,{id:this.input instanceof r||n.length>1?`${this.id}-chunk-${l}-edges-${c}`:`${this.id}-edges-${c}`,input:s,output:this.output,mask:i?.[l],edges:o,edgeValidity:a}))}),t}if(F(a)&&a.buffer.graph!==e)throw Error(`${this.id} domain must belong to the target graph`);if(a===`auto`){let n=s(e,`${this.id}-auto-domain`,this.input.format,2);t.push(...new h({id:`${this.id}-extent`,input:this.input,output:n,operation:`extent`}).getCommandNodes(e)),a=n}if(a===void 0)throw Error(`${this.id} requires either domain or edges`);t.push(...D(e,this.id,this.output));let c=this.output.length<=_?`local`:`global`;return n.forEach((o,s)=>{o.length!==0&&t.push(...A(e,{id:this.input instanceof r||n.length>1?`${this.id}-chunk-${s}-${c}`:`${this.id}-${c}`,input:o,output:this.output,mask:i?.[s],domain:a}))}),t}};function C(e){return e instanceof r?e.data:[e]}function w(e){return e instanceof r?e.data:[e]}function T(e,t){if(!x.includes(e.format)||e.rowByteLength!==b||e.byteStride<b||e.byteStride%b!==0||e.byteOffset%b!==0)throw Error(`${t} must be a uint32-aligned 32-bit scalar GPU data view`)}function E(e){return e.byteStride/b}function D(e,t,n){let r=[],a=`${t}-clear`,s=M(e,n.length),c=`
const BIN_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;
@compute @workgroup_size(${g}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${i(s,g)}
  if (index < BIN_COUNT) { atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u); }
}`;return r.push(...j(e,{id:a,source:c,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputCounts:n},dispatchLayout:s,record:{moduleId:`GPUHistogram/clear`,invocationCount:n.length}})),r}function O(e,t,n,r){let i=[],a=L(n.format),s=n.format===`float32`?`left == left && right == right && abs(left) <= 3.402823466e+38 && abs(right) <= 3.402823466e+38`:`true`,c=`
const EDGE_COUNT: u32 = ${n.length}u;
const EDGE_OFFSET: u32 = ${o(n)}u;
const VALIDITY_OFFSET: u32 = ${o(r)}u;
@group(0) @binding(0) var<storage, read> edgeValues: array<${a}>;
@group(0) @binding(1) var<storage, read_write> edgeValidity: array<atomic<u32>>;
@compute @workgroup_size(${g}) fn main(
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let lane = localId.x;
  if (lane == 0u) { atomicStore(&edgeValidity[VALIDITY_OFFSET], 1u); }
  storageBarrier();
  var edgeIndex = lane;
  loop {
    if (edgeIndex + 1u >= EDGE_COUNT) { break; }
    let left = edgeValues[EDGE_OFFSET + edgeIndex];
    let right = edgeValues[EDGE_OFFSET + edgeIndex + 1u];
    if (!(${s}) || left >= right) {
      atomicStore(&edgeValidity[VALIDITY_OFFSET], 0u);
    }
    edgeIndex = edgeIndex + ${g}u;
  }
}`;return i.push(...j(e,{id:`${t}-validate-edges`,source:c,resources:[{buffer:n,usage:`storage-read`},{buffer:r,usage:`storage-write`}],bindings:{edgeValues:n,edgeValidity:r},dispatchCount:1,record:{moduleId:`GPUHistogram/validate-edges`,invocationCount:n.length}})),i}function k(e,t){let n=M(e,t.input.length),r=t.output.length<=_,a=r&&t.output.length<=v&&m(e.device)===`subgroups`,s=L(t.input.format),c=I(t.edges),l=c?`@group(0) @binding(1) var<storage, read> edgeValues: array<${s}>;`:``,u=t.edgeValidity?`@group(0) @binding(2) var<storage, read> edgeValidity: array<u32>;`:``,f=c?3:1,h=t.mask?`@group(0) @binding(${f}) var<storage, read> selectionMask: array<u32>;`:``,y=f+Number(!!t.mask),b=t.edges,x=c?`fn getEdge(index: u32) -> ${s} {
  return edgeValues[EDGE_OFFSET + index];
}`:`const EDGES: array<${s}, ${b.length}> = array<${s}, ${b.length}>(
  ${b.map(e=>R(e,t.input.format)).join(`, `)}
);
fn getEdge(index: u32) -> ${s} { return EDGES[index]; }`,S=t.edgeValidity?`edgeValidity[${o(t.edgeValidity)}u] != 0u`:`true`,C=t.input.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`,w=a?p(`accepted`,`binIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[binIndex], 1u); }`,T=r?`${w}
  workgroupBarrier();
  if (lane < BIN_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + binIndex], 1u); }`,D=`
${a?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const BIN_COUNT: u32 = ${t.output.length}u;
const INPUT_OFFSET: u32 = ${o(t.input)}u;
const INPUT_STRIDE: u32 = ${E(t.input)}u;
${c?`const EDGE_OFFSET: u32 = ${o(t.edges)}u;`:``}
${t.mask?`const MASK_OFFSET: u32 = ${o(t.mask)}u;
const MASK_STRIDE: u32 = ${E(t.mask)}u;`:``}
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${s}>;
${l}
${u}
${h}
@group(0) @binding(${y}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${r?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${x}
${a?d():``}

@compute @workgroup_size(${g}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>${a?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${i(n,g)}
  let lane = localInvocationIndex;
  ${r?`if (lane < BIN_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var binIndex = 0u;
  if (index < ELEMENT_COUNT && ${S}${t.mask?` && selectionMask[MASK_OFFSET + index * MASK_STRIDE] != 0u`:``}) {
    let value = inputValues[INPUT_OFFSET + index * INPUT_STRIDE];
    let minimum = getEdge(0u);
    let maximum = getEdge(BIN_COUNT);
    if (${C} && value >= minimum && value <= maximum) {
      accepted = true;
      var lower = 0u;
      var upper = BIN_COUNT;
      loop {
        if (lower >= upper) { break; }
        let middle = lower + (upper - lower) / 2u;
        if (value < getEdge(middle + 1u)) {
          upper = middle;
        } else {
          lower = middle + 1u;
        }
      }
      binIndex = min(lower, BIN_COUNT - 1u);
    }
  }
  ${T}
}`,O=[{buffer:t.input,usage:`storage-read`},...c?[{buffer:t.edges,usage:`storage-read`}]:[],...t.edgeValidity?[{buffer:t.edgeValidity,usage:`storage-read`}]:[],...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return j(e,{id:t.id,source:D,resources:O,bindings:{inputValues:t.input,...c?{edgeValues:t.edges}:{},...t.edgeValidity?{edgeValidity:t.edgeValidity}:{},...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchLayout:n,record:{moduleId:`GPUHistogram/accumulate-edges/${r?`local`:`global`}`,invocationCount:t.input.length,formula:`gpu-core.histogram`}})}function A(e,t){let n=M(e,t.input.length),r=t.output.length<=_,a=r&&t.output.length<=v&&m(e.device)===`subgroups`,s=L(t.input.format),c=F(t.domain),l=t.domain,u=c?`@group(0) @binding(1) var<storage, read> domainValues: array<`+s+`>;`:``,f=c?2:1,h=t.mask?`@group(0) @binding(${f}) var<storage, read> selectionMask: array<u32>;`:``,y=f+Number(!!t.mask),b=c?`let minimum: ${s} = domainValues[DOMAIN_OFFSET];
  let maximum: ${s} = domainValues[DOMAIN_OFFSET + 1u];`:`let minimum: ${s} = ${R(l[0],t.input.format)};
  let maximum: ${s} = ${R(l[1],t.input.format)};`,x=t.input.format===`float32`?`value == value && abs(value) <= 3.402823466e+38`:`true`,S=32-Math.clz32(t.output.length),C=t.input.format===`float32`?``:`fn multiplyDivideFloor(numerator: u32, multiplier: u32, denominator: u32) -> u32 {
  var quotient = 0u;
  var remainder = 0u;
  var bitIndex = ${S}u;
  loop {
    bitIndex = bitIndex - 1u;
    // Double and add modulo denominator without overflowing u32.
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
}`,w=t.input.format===`float32`?`let ratio = (value - minimum) / (maximum - minimum);
          binIndex = min(u32(ratio * f32(BIN_COUNT)), BIN_COUNT - 1u);
          let width = (maximum - minimum) / f32(BIN_COUNT);
          // Subtraction can erase which side of an edge value occupies: for [-1, 8],
          // nextDown(1) - (-1) rounds to 2. Compare the original value against the
          // interval edges, not the rounded difference. Usually no correction is needed.
          if (width > 0.0 && (bitcast<u32>(width) & 0x7f800000u) != 0x7f800000u) {
            var lower = binIndex;
            var upper = binIndex + 1u;
            if (value < minimum + f32(binIndex) * width) {
              lower = 0u;
              upper = binIndex;
            } else if (binIndex + 1u < BIN_COUNT && value >= minimum + f32(binIndex + 1u) * width) {
              lower = binIndex + 1u;
              upper = BIN_COUNT;
            }
            // Binary correction bounds the work even when adjacent f32 edges coincide.
            while (lower + 1u < upper) {
              let middle = lower + (upper - lower) / 2u;
              if (value < minimum + f32(middle) * width) {
                upper = middle;
              } else {
                lower = middle;
              }
            }
            binIndex = lower;
          }`:t.input.format===`uint32`?`binIndex = multiplyDivideFloor(value - minimum, BIN_COUNT, maximum - minimum);`:`let orderedValue = bitcast<u32>(value) ^ 0x80000000u;
          let orderedMinimum = bitcast<u32>(minimum) ^ 0x80000000u;
          let orderedMaximum = bitcast<u32>(maximum) ^ 0x80000000u;
          binIndex = multiplyDivideFloor(
            orderedValue - orderedMinimum,
            BIN_COUNT,
            orderedMaximum - orderedMinimum
          );`,T=a?p(`accepted`,`binIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[binIndex], 1u); }`,D=r?`${T}
  workgroupBarrier();
  if (lane < BIN_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + binIndex], 1u); }`,O=`
${a?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.input.length}u;
const BIN_COUNT: u32 = ${t.output.length}u;
const INPUT_OFFSET: u32 = ${o(t.input)}u;
const INPUT_STRIDE: u32 = ${E(t.input)}u;
${c?`const DOMAIN_OFFSET: u32 = ${o(t.domain)}u;`:``}
${t.mask?`const MASK_OFFSET: u32 = ${o(t.mask)}u;
const MASK_STRIDE: u32 = ${E(t.mask)}u;`:``}
const OUTPUT_OFFSET: u32 = ${o(t.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${s}>;
${u}
${h}
@group(0) @binding(${y}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${r?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${C}
${a?d():``}

@compute @workgroup_size(${g}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>${a?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${i(n,g)}
  let lane = localInvocationIndex;
  ${b}
  ${r?`if (lane < BIN_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var binIndex = 0u;
  if (index < ELEMENT_COUNT && maximum >= minimum${t.mask?` && selectionMask[MASK_OFFSET + index * MASK_STRIDE] != 0u`:``}) {
    let value = inputValues[INPUT_OFFSET + index * INPUT_STRIDE];
    if (${x} && value >= minimum && value <= maximum) {
      if (maximum == minimum) {
        accepted = value == minimum;
      } else {
        accepted = true;
        if (value == maximum) {
          binIndex = BIN_COUNT - 1u;
        } else {
          ${w}
        }
      }
    }
  }
  ${D}
}`,k=[{buffer:t.input,usage:`storage-read`},...c?[{buffer:t.domain,usage:`storage-read`}]:[],...t.mask?[{buffer:t.mask,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];return j(e,{id:t.id,source:O,resources:k,bindings:{inputValues:t.input,...c?{domainValues:t.domain}:{},...t.mask?{selectionMask:t.mask}:{},outputCounts:t.output},dispatchLayout:n,record:{moduleId:`GPUHistogram/accumulate/${r?`local`:`global`}`,invocationCount:t.input.length,formula:`gpu-core.histogram`}})}function j(n,r){let i=[],a=r.dispatchLayout?r.dispatchLayout.x*r.dispatchLayout.y*r.dispatchLayout.z:r.dispatchCount??1;i.push(t({id:r.id,resources:r.resources,workload:{operation:`GPUHistogram`,commandCount:1,maximumWorkgroupCount:a,maximumInvocationCount:a*g},compile:({device:t})=>{let n=new l(t,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:t,getBuffer:i})=>{let a={};for(let[t,n]of Object.entries(r.bindings))a[t]=e(n,i);r.dispatchLayout?n.dispatch(t,{bindings:a,x:r.dispatchLayout.x,y:r.dispatchLayout.y,z:r.dispatchLayout.z}):n.dispatch(t,{bindings:a,x:r.dispatchCount??1})},destroy:()=>n.destroy()}}}));let o=f(n,()=>({op:`GPUHistogram`,moduleId:r.record.moduleId,source:r.source,workgroupSize:[g,1,1],workgroups:r.dispatchLayout?[r.dispatchLayout.x,r.dispatchLayout.y,r.dispatchLayout.z]:[r.dispatchCount??1,1,1],invocationCount:r.record.invocationCount,formula:r.record.formula,bindings:Object.entries(r.bindings).map(([e,t])=>{let n=r.resources.find(e=>e.buffer===t)?.usage;return{name:e,view:t,access:n===`storage-read`?`read`:n===`storage-read-write`?`read_write`:`write`}})}));return o&&(i[0].annotation=o),i}function M(e,t){return c(`GPUHistogram`,t,g,e.device.limits.maxComputeWorkgroupsPerDimension)}function N(e,t,r,i){if(I(e)){if(n(e,x,`${i} edges`),e.format!==t||e.length!==r.length+1)throw Error(`${i} GPU edges must contain output.length + 1 ${t} rows`);if(e.buffer===r.buffer)throw Error(`${i} edges and output must use separate buffers`);return}if(e.length!==r.length+1)throw Error(`${i} literal edges must contain output.length + 1 values`);if(e.length>y)throw Error(`${i} literal edges support at most ${y} values`);let a=t===`uint32`?0:-2147483648,o=t===`uint32`?4294967295:2147483647,s=t===`float32`?e.map(Math.fround):e;if((t===`float32`?s.some(e=>!Number.isFinite(e)):e.some(e=>!Number.isInteger(e)||e<a||e>o))||s.some((e,t)=>t>0&&e<=s[t-1]))throw Error(`${i} literal edges must be finite, representable, and strictly increasing`)}function P(e,t,n){if(e.length!==2||!e.every(Number.isFinite)||e[0]>e[1])throw Error(`${n} literal domain must be a finite [min, max] pair`);if(t!==`float32`){let r=t===`uint32`?0:-2147483648,i=t===`uint32`?4294967295:2147483647;if(!e.every(e=>Number.isInteger(e)&&e>=r&&e<=i))throw Error(`${n} literal domain values must fit ${t}`)}}function F(e){return e!==void 0&&e!==`auto`&&!Array.isArray(e)}function I(e){return e!==void 0&&!Array.isArray(e)}function L(e){return e===`uint32`?`u32`:e===`sint32`?`i32`:`f32`}function R(e,t){if(t===`uint32`)return`${e}u`;if(t===`sint32`)return`${e}`;let n=`${Math.fround(e)}`;return n.includes(`.`)||n.includes(`e`)?n:`${n}.0`}export{S as t};