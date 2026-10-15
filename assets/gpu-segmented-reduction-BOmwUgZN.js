import{D as e,I as t,M as n,j as r,k as i,z as a}from"./resources-CwvhEOBe.js";import{o}from"./gpu-subgroup-utils-BuV3X9Xp.js";var s=256,c=[`uint32`,`sint32`,`float32`],l=class{id;input;segmentOffsets;output;operation;constructor(e){if(this.id=e.id??`gpu-segmented-reduction`,this.input=e.input,this.segmentOffsets=e.segmentOffsets,this.output=e.output,this.operation=e.operation,n(this.input,c,`${this.id} input`),r(this.segmentOffsets,`${this.id} segmentOffsets`),n(this.output,c,`${this.id} output`),this.input.format!==this.output.format)throw Error(`${this.id} input and output formats must match`);if(this.segmentOffsets.length!==this.output.length+1)throw Error(`${this.id} segmentOffsets length must equal output.length + 1`);if(![`sum`,`min`,`max`].includes(this.operation))throw Error(`${this.id} operation must be sum, min, or max`);if(this.output.buffer===this.input.buffer||this.output.buffer===this.segmentOffsets.buffer)throw Error(`${this.id} output must use a separate buffer`)}getCommandNodes(n){let r=[];for(let e of[this.input,this.segmentOffsets,this.output])if(e.buffer.graph!==n)throw Error(`${this.id} views must belong to target graph`);if(this.output.length===0)return r;let i=u(this);r.push(t({id:this.id,workload:{operation:`GPUSegmentedReduction`,commandCount:1,maximumWorkgroupCount:this.output.length,maximumInvocationCount:this.output.length*s,readByteLength:this.input.length*4+this.segmentOffsets.length*4,writeByteLength:this.output.length*4},resources:[{buffer:this.input,usage:`storage-read`},{buffer:this.segmentOffsets,usage:`storage-read`},{buffer:this.output,usage:`storage-write`}],compile:({device:t})=>{let n=new a(t,{id:this.id,source:i,shaderLayout:{bindings:[{name:`inputValues`,type:`read-only-storage`,group:0,location:0},{name:`segmentOffsets`,type:`read-only-storage`,group:0,location:1},{name:`outputValues`,type:`storage`,group:0,location:2}]}});return{encode:({computePass:t,getBuffer:r})=>{let i={inputValues:e(this.input,r),segmentOffsets:e(this.segmentOffsets,r),outputValues:e(this.output,r)};n.dispatch(t,{bindings:i,x:this.output.length,y:1,z:1})},destroy:()=>n.destroy()}}}));let c=o(n,()=>({op:`GPUSegmentedReduction`,moduleId:`GPUSegmentedReduction/reduce`,source:i,workgroupSize:[s,1,1],workgroups:[this.output.length,1,1],invocationCount:this.output.length*s,formula:`gpu-core.segmented-reduction`,bindings:[{name:`inputValues`,view:this.input,access:`read`},{name:`segmentOffsets`,view:this.segmentOffsets,access:`read`},{name:`outputValues`,view:this.output,access:`write`}]}));return c&&(r[0].annotation=c),r}};function u(e){let t=e.input.format===`uint32`?`u32`:e.input.format===`sint32`?`i32`:`f32`,n=e.operation===`sum`?t===`f32`?`0.0`:`0`:e.operation===`min`?t===`f32`?`3.402823466e+38`:t===`u32`?`0xffffffffu`:`2147483647`:t===`f32`?`-3.402823466e+38`:t===`u32`?`0u`:`-2147483648`,r=e.operation===`sum`?`a + b`:e.operation===`min`?`min(a, b)`:`max(a, b)`;return`const INPUT_OFFSET: u32 = ${i(e.input)}u;
const SEGMENT_OFFSET: u32 = ${i(e.segmentOffsets)}u;
const OUTPUT_OFFSET: u32 = ${i(e.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<${t}>;
@group(0) @binding(1) var<storage, read> segmentOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<${t}>;
var<workgroup> scratch: array<${t}, ${s}>;
fn combine(a: ${t}, b: ${t}) -> ${t} { return ${r}; }
@compute @workgroup_size(${s})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let segment = wg.x;
  let begin = segmentOffsets[SEGMENT_OFFSET + segment];
  let end = segmentOffsets[SEGMENT_OFFSET + segment + 1u];
  var value: ${t} = ${n};
  var i = begin + lane;
  loop {
    if (i >= end) { break; }
    value = combine(value, inputValues[INPUT_OFFSET + i]);
    i += ${s}u;
  }
  scratch[lane] = value;
  workgroupBarrier();
  var stride = ${s/2}u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { scratch[lane] = combine(scratch[lane], scratch[lane + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    outputValues[OUTPUT_OFFSET + segment] = select(scratch[0], ${t}(${e.operation,`0`}), begin == end);
  }
}`}export{l as t};