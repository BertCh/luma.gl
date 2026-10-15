import{E as e,h as t}from"./analysis-frame-Blu0tYdw.js";import{C as n}from"./arithmetic-expression-Kknn3j_i.js";import{D as r,E as i,I as a,M as o,O as s,R as c,S as l,T as u,_ as d,a as f,d as p,i as m,j as h,k as g,s as _,v,w as y,x as b,y as x,z as S}from"./resources-CwvhEOBe.js";import{a as C,i as ee,n as w,o as T,r as te,t as E}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{t as ne}from"./gpu-reduction-BD8l40jM.js";import{t as re}from"./gpu-histogram-C4GJppav.js";import{a as ie,i as D,n as O,r as k,t as A}from"./gpu-spatial-utils-Cp29aAUS.js";import{t as j}from"./gpu-group-aggregation-DrNKSCi8.js";import{l as M,s as N}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as ae}from"./wgsl-common-BNRVzgd8.js";import{t as oe}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";import{R as se}from"./index-Bjk18V2I.js";var ce=2048,le=`
const GPU_FFT_PI: f32 = 3.14159265358979323846;

fn reverseLowBits(value: u32, bitCount: u32) -> u32 {
  var source = value;
  var reversed = 0u;
  for (var bitIndex = 0u; bitIndex < bitCount; bitIndex++) {
    reversed = (reversed << 1u) | (source & 1u);
    source = source >> 1u;
  }
  return reversed;
}

fn multiplyComplex(left: vec2f, right: vec2f) -> vec2f {
  return vec2f(
    left.x * right.x - left.y * right.y,
    left.x * right.y + left.y * right.x
  );
}
`;function P(e){let t=Math.log2(e);return[{kind:`bit-reversal`,stage:t},...Array.from({length:t},(e,t)=>({kind:`butterfly`,stage:t+1}))]}var F=class{id;input;kernel;output;width;height;kernelWidth;kernelHeight;strategy;boundary;stats;constructor(e){this.id=e.id??`gpu-convolution`,this.input=e.input,this.kernel=e.kernel,this.output=e.output,this.width=e.width,this.height=e.height,this.kernelWidth=e.kernelWidth,this.kernelHeight=e.kernelHeight,this.strategy=e.strategy??`auto`,this.boundary=e.boundary??`zero`,this.stats=I(e);for(let e of[this.input,this.kernel,this.output])for(let t of C(e))o(t,[`float32`],this.id);if(this.input.length<this.stats.elementCount)throw Error(`${this.id} input must contain at least width * height rows`);if(this.kernel.length<this.stats.kernelElementCount)throw Error(`${this.id} kernel must contain at least kernelWidth * kernelHeight rows`);if(this.output.length<this.stats.elementCount)throw Error(`${this.id} output must contain at least width * height rows`);let t=new Set(C(this.input).map(e=>e.buffer)),n=new Set(C(this.kernel).map(e=>e.buffer)),r=C(this.output);if(r.some(e=>t.has(e.buffer)||n.has(e.buffer)))throw Error(`${this.id} output must use a separate buffer from input and kernel`);if(C(this.input).some(e=>n.has(e.buffer)))throw Error(`${this.id} input and kernel must use separate buffers`);for(let[e,t]of r.entries())if(r.slice(0,e).some(e=>u(e,t)))throw Error(`${this.id} output chunks must not overlap`);if(![`auto`,`direct`,`fft`].includes(this.strategy))throw Error(`${this.id} strategy must be auto, direct, or fft`)}getCommandNodes(e){let t=[];U(e,this.input,`${this.id} input`),U(e,this.kernel,`${this.id} kernel`),U(e,this.output,`${this.id} output`);let n=ue(e.device,{width:this.width,height:this.height,kernelWidth:this.kernelWidth,kernelHeight:this.kernelHeight,strategy:this.strategy,boundary:this.boundary,input:this.input,kernel:this.kernel,output:this.output});if(!n.supported||!n.strategy)throw Error(n.reason);return n.strategy===`direct`?t.push(...de(e,this)):t.push(...fe(e,this)),t}};function I(e){be(e);let t=e.boundary??`zero`,n=e.width*e.height,r=e.kernelWidth*e.kernelHeight;if(!Number.isSafeInteger(n)||n>4294967295)throw Error(`GPUConvolution width * height must fit in a uint32 index range.`);if(!Number.isSafeInteger(r)||r>4294967295)throw Error(`GPUConvolution kernel dimensions must fit in a uint32 index range.`);let i=B(e.width,e.kernelWidth,t),a=B(e.height,e.kernelHeight,t),o=i*a,s=Math.log2(i)+Math.log2(a)+2,c=o*2*Float32Array.BYTES_PER_ELEMENT;return Object.freeze({width:e.width,height:e.height,kernelWidth:e.kernelWidth,kernelHeight:e.kernelHeight,boundary:t,elementCount:n,kernelElementCount:r,directMultiplyAddCount:n*r,fftWidth:i,fftHeight:a,fftElementCount:o,fftTransformPassCount:s,fftDispatchCount:3*s+3,fftComplexBufferByteLength:c,fftLogicalTransientByteLength:9*c})}function ue(e,t){let n;try{n=I(t)}catch(e){return{supported:!1,reason:e.message}}if(e.type!==`webgpu`)return{supported:!1,reason:`GPUConvolution requires WebGPU.`,stats:n};if(e.limits.maxStorageBuffersPerShaderStage<3)return{supported:!1,reason:`GPUConvolution requires three compute storage buffers.`,stats:n};if(e.limits.maxComputeInvocationsPerWorkgroup<256||e.limits.maxComputeWorkgroupSizeX<256)return{supported:!1,reason:`GPUConvolution requires 256 compute invocations.`,stats:n};let r=Math.max(n.elementCount,n.kernelElementCount)*Float32Array.BYTES_PER_ELEMENT,i=t.input&&t.kernel&&t.output;if(!i&&(r>e.limits.maxStorageBufferBindingSize||r>e.limits.maxBufferSize))return{supported:!1,reason:`GPUConvolution scalar data exceeds device buffer limits.`,stats:n};if(i)for(let[r,i]of[[t.input,n.elementCount],[t.kernel,n.kernelElementCount],[t.output,n.elementCount]]){let t=i;for(let i of C(r)){let r=Math.min(t,i.length);if(r&&(s(i).size-(i.length-r)*4>e.limits.maxStorageBufferBindingSize||i.buffer.byteLength>e.limits.maxBufferSize))return{supported:!1,reason:`GPUConvolution chunk exceeds device buffer limits.`,stats:n};if(t-=r,!t)break}}let a=n.elementCount;if(t.output){let e=n.elementCount;a=0;for(let n of C(t.output)){let t=Math.min(e,n.length);if(a=Math.max(a,t),e-=t,!e)break}}try{b(`GPUConvolution`,a,256,e.limits.maxComputeWorkgroupsPerDimension)}catch(e){return{supported:!1,reason:e.message,stats:n}}let o=t.strategy??`auto`;if(![`auto`,`direct`,`fft`].includes(o))return{supported:!1,reason:`GPUConvolution strategy must be auto, direct, or fft.`,stats:n};let c=xe(e,t,n),l=o===`auto`?n.kernelElementCount<=4096||c?`direct`:`fft`:o;return l===`fft`&&c?{supported:!1,reason:c,stats:n}:{supported:!0,stats:n,strategy:l}}function de(e,t){let n=[],r=L(e,t.input,t.stats.elementCount),i=L(e,t.kernel,t.stats.kernelElementCount),a=L(e,t.output,t.stats.elementCount);for(let o of a){let a=b(t.id,o.length,256,e.device.limits.maxComputeWorkgroupsPerDimension),s=!0;for(let c of i)for(let i of r){let r=_e({...t,input:i.data,kernel:c.data,output:o.data,inputLogicalOffset:i.offset,kernelLogicalOffset:c.offset,outputLogicalOffset:o.offset,initializeOutput:s},a);n.push(...z(e,{id:`${t.id}-direct-${n.length}`,source:r,bindings:{inputValues:i.data,kernelValues:c.data},outputs:{outputValues:o.data},dispatchLayout:a,operation:`GPUConvolution.direct`,readOutputs:!s,invocationCount:o.length,formula:`gpu-core.convolution`,readByteLength:o.length*c.length*8+(s?0:o.length*4),writeByteLength:o.length*4})),s=!1}}return n}function L(e,t,r){return n(C(i(e,t,r))).filter(e=>e.length)}function fe(e,t){let n=[],{fftElementCount:r}=t.stats,i=y(e,`${t.id}-fft-packed-input`,`float32x2`,r),a=y(e,`${t.id}-fft-packed-kernel`,`float32x2`,r),o=y(e,`${t.id}-fft-input-spectrum`,`float32x2`,r),s=y(e,`${t.id}-fft-kernel-spectrum`,`float32x2`,r),c=y(e,`${t.id}-fft-product-spectrum`,`float32x2`,r),l=y(e,`${t.id}-fft-inverse-spatial`,`float32x2`,r),u=y(e,`${t.id}-fft-input-scratch`,`float32x2`,r),d=y(e,`${t.id}-fft-kernel-scratch`,`float32x2`,r),f=y(e,`${t.id}-fft-inverse-scratch`,`float32x2`,r);return n.push(...pe(e,t,i,a)),n.push(...R(e,`${t.id}-fft-input`,i,o,u,t.stats.fftWidth,t.stats.fftHeight,`forward`)),n.push(...R(e,`${t.id}-fft-kernel`,a,s,d,t.stats.fftWidth,t.stats.fftHeight,`forward`)),n.push(...me(e,t,o,s,c)),n.push(...R(e,`${t.id}-fft-inverse`,c,l,f,t.stats.fftWidth,t.stats.fftHeight,`inverse`)),n.push(...he(e,t,l)),n}function pe(e,t,n,r){let i=[],a=L(e,t.input,t.stats.elementCount),o=L(e,t.kernel,t.stats.kernelElementCount),s=b(`${t.id} FFT pack`,t.stats.fftElementCount,256,e.device.limits.maxComputeWorkgroupsPerDimension);if(a.length>1||o.length>1){for(let[c,u,d]of[[`input`,a,n],[`kernel`,o,r]])for(let[n,r]of u.entries()){let a=c===`input`?`let sourceX = i32(coordinateX);
  let sourceY = i32(coordinateY);`:`var signedX = i32(coordinateX);
  var signedY = i32(coordinateY);
  if (coordinateX > ${Math.floor(t.kernelWidth/2)}u) {
    signedX -= i32(FFT_WIDTH);
  }
  if (coordinateY > ${Math.floor(t.kernelHeight/2)}u) {
    signedY -= i32(FFT_HEIGHT);
  }
  let sourceX = signedX + ${Math.floor(t.kernelWidth/2)};
  let sourceY = signedY + ${Math.floor(t.kernelHeight/2)};`,o=c===`input`?t.width:t.kernelWidth,u=c===`input`?t.height:t.kernelHeight,f=`const FFT_ELEMENT_COUNT: u32 = ${t.stats.fftElementCount}u;
const FFT_WIDTH: u32 = ${t.stats.fftWidth}u;
const FFT_HEIGHT: u32 = ${t.stats.fftHeight}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> packedValues: array<vec2f>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${l(s,256)}
  if (index >= FFT_ELEMENT_COUNT) {
    return;
  }
  ${n===0?`packedValues[index] = vec2f(0.0);`:``}
  let coordinateY = index / FFT_WIDTH;
  let coordinateX = index % FFT_WIDTH;
  ${a}
  if (sourceX >= 0 && sourceX < ${o} && sourceY >= 0 && sourceY < ${u}) {
    let sourceIndex = u32(sourceY) * ${o}u + u32(sourceX);
    if (sourceIndex >= ${r.offset}u && sourceIndex - ${r.offset}u < ${r.length}u) {
      packedValues[index] = vec2f(sourceValues[${g(r.data)}u + sourceIndex - ${r.offset}u], 0.0);
    }
  }
}`;i.push(...z(e,{id:`${t.id}-fft-pack-${c}-${n}`,source:f,bindings:{sourceValues:r.data},outputs:{packedValues:d},dispatchLayout:s,operation:`GPUConvolution.fft.pack`,readByteLength:r.length*4,writeByteLength:n===0?t.stats.fftComplexBufferByteLength:r.length*8}))}return i}let c=ve({...t,input:a[0].data,kernel:o[0].data},s);return i.push(...z(e,{id:`${t.id}-fft-pack`,source:c,bindings:{inputValues:a[0].data,kernelValues:o[0].data},outputs:{packedInput:n,packedKernel:r},dispatchLayout:s,operation:`GPUConvolution.fft.pack`,readByteLength:(t.stats.elementCount+t.stats.kernelElementCount)*Float32Array.BYTES_PER_ELEMENT,writeByteLength:2*t.stats.fftComplexBufferByteLength})),i}function me(e,t,n,r,i){let a=[],o=b(`${t.id} FFT multiply`,t.stats.fftElementCount,256,e.device.limits.maxComputeWorkgroupsPerDimension),s=`${le}
const ELEMENT_COUNT: u32 = ${t.stats.fftElementCount}u;
@group(0) @binding(0) var<storage, read> inputSpectrum: array<vec2f>;
@group(0) @binding(1) var<storage, read> kernelSpectrum: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> productSpectrum: array<vec2f>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(o,256)}
  if (index >= ELEMENT_COUNT) { return; }
  productSpectrum[index] = multiplyComplex(inputSpectrum[index], kernelSpectrum[index]);
}`;return a.push(...z(e,{id:`${t.id}-fft-multiply`,source:s,bindings:{inputSpectrum:n,kernelSpectrum:r},outputs:{productSpectrum:i},dispatchLayout:o,operation:`GPUConvolution.fft.multiply`,readByteLength:2*t.stats.fftComplexBufferByteLength,writeByteLength:t.stats.fftComplexBufferByteLength})),a}function he(e,t,n){let r=[],i=L(e,t.output,t.stats.elementCount);for(let[a,o]of i.entries()){let i=b(`${t.id} FFT crop`,o.length,256,e.device.limits.maxComputeWorkgroupsPerDimension),s=`const ELEMENT_COUNT: u32 = ${o.length}u;
const WIDTH: u32 = ${t.width}u;
const FFT_WIDTH: u32 = ${t.stats.fftWidth}u;
@group(0) @binding(0) var<storage, read> inverseSpatial: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(i,256)}
  if (index >= ELEMENT_COUNT) {
    return;
  }
  let row = ${o.offset}u + index;
  let outputY = row / WIDTH;
  let outputX = row % WIDTH;
  outputValues[${g(o.data)}u + index] =
    inverseSpatial[outputY * FFT_WIDTH + outputX].x;
}`;r.push(...z(e,{id:`${t.id}-fft-crop-${a}`,source:s,bindings:{inverseSpatial:n},outputs:{outputValues:o.data},dispatchLayout:i,operation:`GPUConvolution.fft.crop`,formula:`gpu-core.convolution`,readByteLength:o.length*2*Float32Array.BYTES_PER_ELEMENT,writeByteLength:o.length*Float32Array.BYTES_PER_ELEMENT}))}return r}function R(e,t,n,r,i,a,o,s){let c=[],l=[...P(a).map(e=>({axis:`horizontal`,...e})),...P(o).map(e=>({axis:`vertical`,...e}))],u=n;for(let[n,d]of l.entries()){let f=(l.length-n)%2==0?i:r;c.push(...ge(e,{id:`${t}-${d.axis}-${d.kind}-${d.stage}`,input:u,output:f,width:a,height:o,direction:s,pass:d,finalPass:n===l.length-1})),u=f}return c}function ge(e,t){let n=[],r=t.width*t.height,i=b(t.id,r,256,e.device.limits.maxComputeWorkgroupsPerDimension),a=ye(t,i);return n.push(...z(e,{id:t.id,source:a,bindings:{inputValues:t.input},outputs:{outputValues:t.output},dispatchLayout:i,operation:`GPUConvolution.fft.transform`,readByteLength:r*2*Float32Array.BYTES_PER_ELEMENT,writeByteLength:r*2*Float32Array.BYTES_PER_ELEMENT})),n}function z(e,t){let n=[],i=[...Object.entries(t.bindings),...Object.entries(t.outputs)];n.push(a({id:t.id,workload:{operation:t.operation,commandCount:1,maximumWorkgroupCount:t.dispatchLayout.x*t.dispatchLayout.y*t.dispatchLayout.z,maximumInvocationCount:t.dispatchLayout.x*t.dispatchLayout.y*t.dispatchLayout.z*256,readByteLength:t.readByteLength,writeByteLength:t.writeByteLength},resources:[...Object.values(t.bindings).map(e=>({buffer:e,usage:`storage-read`})),...Object.values(t.outputs).map(e=>({buffer:e,usage:t.readOutputs?`storage-read-write`:`storage-write`}))],compile:({device:e})=>{let n=new S(e,{id:t.id,source:t.source,shaderLayout:{bindings:i.map(([e],n)=>({name:e,type:n<Object.keys(t.bindings).length?`read-only-storage`:`storage`,group:0,location:n}))}});return{encode:({computePass:e,getBuffer:a})=>{let o={};for(let[e,t]of i)o[e]=r(t,a);n.dispatch(e,{bindings:o,x:t.dispatchLayout.x,y:t.dispatchLayout.y,z:t.dispatchLayout.z})},destroy:()=>n.destroy()}}}));let o=T(e,()=>({op:`GPUConvolution`,moduleId:t.operation.replace(`.`,`/`).replaceAll(`.`,`-`),source:t.source,workgroupSize:[256,1,1],workgroups:[t.dispatchLayout.x,t.dispatchLayout.y,t.dispatchLayout.z],invocationCount:t.invocationCount,formula:t.formula,bindings:[...Object.entries(t.bindings).map(([e,t])=>({name:e,view:t,access:`read`})),...Object.entries(t.outputs).map(([e,n])=>({name:e,view:n,access:t.readOutputs?`read_write`:`write`}))]}));return o&&(n[0].annotation=o),n}function _e(e,t){let n=e.boundary===`wrap`?`let wrappedX = ((sourceX % i32(WIDTH)) + i32(WIDTH)) % i32(WIDTH);
    let wrappedY = ((sourceY % i32(HEIGHT)) + i32(HEIGHT)) % i32(HEIGHT);
    let sourceIndex = u32(wrappedY) * WIDTH + u32(wrappedX);`:`if (sourceX < 0 || sourceX >= i32(WIDTH) || sourceY < 0 || sourceY >= i32(HEIGHT)) {
      continue;
    }
    let sourceIndex = u32(sourceY) * WIDTH + u32(sourceX);`;return`const ELEMENT_COUNT: u32 = ${Math.min(e.output.length,e.stats.elementCount)}u;
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const KERNEL_WIDTH: u32 = ${e.kernelWidth}u;
const KERNEL_HEIGHT: u32 = ${e.kernelHeight}u;
const KERNEL_CENTER_X: u32 = ${Math.floor(e.kernelWidth/2)}u;
const KERNEL_CENTER_Y: u32 = ${Math.floor(e.kernelHeight/2)}u;
const INPUT_OFFSET: u32 = ${g(e.input)}u;
const KERNEL_OFFSET: u32 = ${g(e.kernel)}u;
const OUTPUT_OFFSET: u32 = ${g(e.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> kernelValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<f32>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(t,256)}
  if (index >= ELEMENT_COUNT) {
    return;
  }
  let outputIndex = ${e.outputLogicalOffset??0}u + index;
  let outputY = outputIndex / WIDTH;
  let outputX = outputIndex - outputY * WIDTH;
  var total = 0.0;
  for (var kernelLocalIndex = 0u; kernelLocalIndex < ${Math.min(e.kernel.length,e.stats.kernelElementCount)}u; kernelLocalIndex++) {
    let kernelIndex = ${e.kernelLogicalOffset??0}u + kernelLocalIndex;
    let kernelY = kernelIndex / KERNEL_WIDTH;
    let kernelX = kernelIndex % KERNEL_WIDTH;
    let sourceY = i32(outputY) - (i32(kernelY) - i32(KERNEL_CENTER_Y));
    let sourceX = i32(outputX) - (i32(kernelX) - i32(KERNEL_CENTER_X));
    let kernelValue = kernelValues[KERNEL_OFFSET + kernelLocalIndex];
    ${n}
    if (sourceIndex >= ${e.inputLogicalOffset??0}u &&
        sourceIndex - ${e.inputLogicalOffset??0}u < ${Math.min(e.input.length,e.stats.elementCount)}u) {
      total += inputValues[INPUT_OFFSET + sourceIndex - ${e.inputLogicalOffset??0}u] * kernelValue;
    }
  }
  outputValues[OUTPUT_OFFSET + index] ${e.initializeOutput===!1?`+=`:`=`} total;
}`}function ve(e,t){return`const FFT_ELEMENT_COUNT: u32 = ${e.stats.fftElementCount}u;
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const FFT_WIDTH: u32 = ${e.stats.fftWidth}u;
const FFT_HEIGHT: u32 = ${e.stats.fftHeight}u;
const KERNEL_WIDTH: u32 = ${e.kernelWidth}u;
const KERNEL_HEIGHT: u32 = ${e.kernelHeight}u;
const KERNEL_CENTER_X: i32 = ${Math.floor(e.kernelWidth/2)};
const KERNEL_CENTER_Y: i32 = ${Math.floor(e.kernelHeight/2)};
const INPUT_OFFSET: u32 = ${g(e.input)}u;
const KERNEL_OFFSET: u32 = ${g(e.kernel)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> kernelValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> packedInput: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> packedKernel: array<vec2f>;
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(t,256)}
  if (index >= FFT_ELEMENT_COUNT) { return; }
  let coordinateY = index / FFT_WIDTH;
  let coordinateX = index - coordinateY * FFT_WIDTH;
  var inputValue = 0.0;
  if (coordinateX < WIDTH && coordinateY < HEIGHT) {
    inputValue = inputValues[INPUT_OFFSET + coordinateY * WIDTH + coordinateX];
  }
  var signedX = i32(coordinateX);
  var signedY = i32(coordinateY);
  if (coordinateX > u32(KERNEL_CENTER_X)) { signedX -= i32(FFT_WIDTH); }
  if (coordinateY > u32(KERNEL_CENTER_Y)) { signedY -= i32(FFT_HEIGHT); }
  let kernelX = signedX + KERNEL_CENTER_X;
  let kernelY = signedY + KERNEL_CENTER_Y;
  var kernelValue = 0.0;
  if (kernelX >= 0 && kernelX < i32(KERNEL_WIDTH) &&
      kernelY >= 0 && kernelY < i32(KERNEL_HEIGHT)) {
    kernelValue = kernelValues[KERNEL_OFFSET + u32(kernelY) * KERNEL_WIDTH + u32(kernelX)];
  }
  packedInput[index] = vec2f(inputValue, 0.0);
  packedKernel[index] = vec2f(kernelValue, 0.0);
}`}function ye(e,t){let n=e.width*e.height,r=e.pass.axis===`horizontal`,i=e.direction===`forward`?`-1.0`:`1.0`,a=e.direction===`inverse`&&e.finalPass?`${1/n}`:`1.0`,o=`const ELEMENT_COUNT: u32 = ${n}u;
const WIDTH: u32 = ${e.width}u;
const INPUT_OFFSET: u32 = ${H(e.input)}u;
const OUTPUT_OFFSET: u32 = ${H(e.output)}u;
@group(0) @binding(0) var<storage, read> inputValues: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec2f>;
${le}`,s=`let coordinateY = index / WIDTH;
  let coordinateX = index - coordinateY * WIDTH;`;if(e.pass.kind===`bit-reversal`){let n=r?`coordinateY * WIDTH + reverseLowBits(coordinateX, ${e.pass.stage}u)`:`reverseLowBits(coordinateY, ${e.pass.stage}u) * WIDTH + coordinateX`;return`${o}
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(t,256)}
  if (index >= ELEMENT_COUNT) { return; }
  ${s}
  outputValues[OUTPUT_OFFSET + index] = inputValues[INPUT_OFFSET + ${n}];
}`}let c=1<<e.pass.stage,u=c>>1,d=r?`coordinateX`:`coordinateY`,f=r?`coordinateY * WIDTH + firstCoordinate`:`firstCoordinate * WIDTH + coordinateX`,p=r?`coordinateY * WIDTH + secondCoordinate`:`secondCoordinate * WIDTH + coordinateX`;return`${o}
@compute @workgroup_size(256) fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(t,256)}
  if (index >= ELEMENT_COUNT) { return; }
  ${s}
  let butterflyOffset = ${d} & ${c-1}u;
  let twiddleIndex = butterflyOffset & ${u-1}u;
  let butterflyStart = ${d} - butterflyOffset;
  let firstCoordinate = butterflyStart + twiddleIndex;
  let secondCoordinate = firstCoordinate + ${u}u;
  let firstValue = inputValues[INPUT_OFFSET + ${f}];
  let secondValue = inputValues[INPUT_OFFSET + ${p}];
  let angle = ${i} * 2.0 * GPU_FFT_PI * f32(twiddleIndex) /
    f32(${c}u);
  let rotatedSecondValue = multiplyComplex(secondValue, vec2f(cos(angle), sin(angle)));
  let butterflyValue = select(
    firstValue + rotatedSecondValue,
    firstValue - rotatedSecondValue,
    butterflyOffset >= ${u}u
  );
  outputValues[OUTPUT_OFFSET + index] = butterflyValue * ${a};
}`}function be(e){for(let[t,n]of[[`width`,e.width],[`height`,e.height],[`kernelWidth`,e.kernelWidth],[`kernelHeight`,e.kernelHeight]])if(!Number.isSafeInteger(n)||n<=0)throw Error(`GPUConvolution ${t} must be a positive integer.`);if(e.kernelWidth%2==0||e.kernelHeight%2==0)throw Error(`GPUConvolution kernel dimensions must be odd.`);if(e.boundary!==void 0&&e.boundary!==`zero`&&e.boundary!==`wrap`)throw Error(`GPUConvolution boundary must be zero or wrap.`)}function B(e,t,n){let r=n===`zero`?e+t-1:e;return Math.max(2,2**Math.ceil(Math.log2(r)))}function xe(e,t,n){if(t.boundary===`wrap`){if(!V(t.width)||!V(t.height))return`GPUConvolution FFT wrap boundary requires power-of-two input dimensions.`;if(t.kernelWidth>t.width||t.kernelHeight>t.height)return`GPUConvolution FFT wrap boundary requires the kernel to fit inside the input field.`}if(n.fftWidth>2048||n.fftHeight>2048)return`GPUConvolution FFT dimensions must not exceed ${ce}.`;if(e.limits.maxStorageBuffersPerShaderStage<4)return`GPUConvolution FFT strategy requires four compute storage buffers.`;if(n.fftComplexBufferByteLength>e.limits.maxStorageBufferBindingSize||n.fftComplexBufferByteLength>e.limits.maxBufferSize)return`GPUConvolution FFT scratch exceeds device buffer limits.`;try{b(`GPUConvolution FFT`,n.fftElementCount,256,e.limits.maxComputeWorkgroupsPerDimension)}catch(e){return e.message}}function V(e){return(e&e-1)==0}function H(e){return e.byteOffset%256/(2*Float32Array.BYTES_PER_ELEMENT)}function U(e,t,n){if(C(t).some(t=>t.buffer.graph!==e))throw Error(`${n} belongs to a different GPUCommandGraph`)}var W=256,Se=256,Ce=16,we=class{id;positions;output;gridSize;bounds;constructor(e){this.id=e.id??`gpu-grid-binning`,this.positions=e.positions,this.output=e.output,this.gridSize=e.gridSize,this.bounds=e.bounds;for(let e of K(this.positions))o(e,[`float32x2`],`${this.id} positions`);for(let e of C(this.output))h(e,`${this.id} output`);ie(this.id,K(this.positions),C(this.output));let[t,n]=this.gridSize;if(!Number.isSafeInteger(t)||!Number.isSafeInteger(n)||t<=0||n<=0)throw Error(`${this.id} gridSize must contain two positive integers`);if(!Number.isSafeInteger(t*n)||t*n>4294967295||this.output.length!==t*n)throw Error(`${this.id} output.length must equal gridSize width * height`);if(K(this.positions).some(e=>C(this.output).some(t=>e.buffer===t.buffer)))throw Error(`${this.id} positions and output must use separate buffers`);if(Array.isArray(this.bounds)){let[e,t,n,r]=this.bounds;if(this.bounds.length!==4||!this.bounds.every(D)||e>n||t>r)throw Error(`${this.id} literal bounds must be finite [minX, minY, maxX, maxY]`)}else if(G(this.bounds)){if(o(this.bounds,[`float32x4`],`${this.id} bounds`),this.bounds.length!==1)throw Error(`${this.id} GPU bounds must contain one float32x4 row`);if(C(this.output).some(e=>e.buffer===this.bounds.buffer))throw Error(`${this.id} bounds and output must use separate buffers`)}}getCommandNodes(e){let t=[];if(K(this.positions).some(t=>t.buffer.graph!==e)||C(this.output).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(G(this.bounds)&&this.bounds.buffer.graph!==e)throw Error(`${this.id} bounds must belong to the target graph`);let n=0;for(let[r,i]of C(this.output).entries()){if(i.length){let a=`${this.id}-output-${r}`;t.push(...Te(e,a,i));for(let[r,o]of K(this.positions).entries())o.length&&t.push(...Ee(e,{id:`${a}-input-${r}`,positions:o,output:i,outputStart:n,gridSize:this.gridSize,bounds:this.bounds}))}n+=i.length}return t}};function Te(e,t,n){let r=[],i=`${t}-clear`,a=b(t,n.length,W,e.device.limits.maxComputeWorkgroupsPerDimension),o=`
const CELL_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${g(n)}u;
@group(0) @binding(0) var<storage, read_write> outputCounts: array<atomic<u32>>;
@compute @workgroup_size(${W}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(a,W)}
  if (index < CELL_COUNT) { atomicStore(&outputCounts[OUTPUT_OFFSET + index], 0u); }
}`;r.push(...O(e,{id:i,source:o,resources:[{buffer:n,usage:`storage-write`}],bindings:{outputCounts:n},dispatch:a}));let s=T(e,()=>({op:`GPUGridBinning`,moduleId:`GPUGridBinning/clear`,source:o,workgroupSize:[W,1,1],workgroups:[a.x,a.y,a.z],invocationCount:n.length,bindings:[{name:`outputCounts`,view:n,access:`write`}]}));if(s&&(r[0].annotation=s),e.recording)for(let e of r)e.workload={operation:`GPUGridBinning`,variant:`clear`,commandCount:1,maximumWorkgroupCount:a.x*a.y*a.z,maximumInvocationCount:a.x*a.y*a.z*W,writeByteLength:n.length*n.rowByteLength};return r}function Ee(e,t){let n=[],[r,i]=t.gridSize,a=b(t.id,t.positions.length,W,e.device.limits.maxComputeWorkgroupsPerDimension),o=t.output.length<=Se,s=o&&t.output.length<=Ce&&E(e.device)===`subgroups`,c=G(t.bounds),u=t.bounds,d=c?`@group(0) @binding(1) var<storage, read> boundsValues: array<f32>;`:``,f=c?2:1,p=c?`let minimumX = boundsValues[BOUNDS_OFFSET];
  let minimumY = boundsValues[BOUNDS_OFFSET + 1u];
  let maximumX = boundsValues[BOUNDS_OFFSET + 2u];
  let maximumY = boundsValues[BOUNDS_OFFSET + 3u];`:`let minimumX = ${k(u[0])};
  let minimumY = ${k(u[1])};
  let maximumX = ${k(u[2])};
  let maximumY = ${k(u[3])};`,m=s?te(`accepted`,`cellIndex`,`localCounts`,t.output.length):`  if (accepted) { atomicAdd(&localCounts[cellIndex], 1u); }`,h=o?`${m}
  workgroupBarrier();
  if (lane < CELL_COUNT) {
    atomicAdd(&outputCounts[OUTPUT_OFFSET + lane], atomicLoad(&localCounts[lane]));
  }`:`if (accepted) { atomicAdd(&outputCounts[OUTPUT_OFFSET + cellIndex], 1u); }`,_=`
${s?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.positions.length}u;
const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${i}u;
const CELL_COUNT: u32 = ${t.output.length}u;
const OUTPUT_START: u32 = ${t.outputStart}u;
const POSITIONS_OFFSET: u32 = ${g(t.positions)}u;
${c?`const BOUNDS_OFFSET: u32 = ${g(t.bounds)}u;`:``}
const OUTPUT_OFFSET: u32 = ${g(t.output)}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
${d}
@group(0) @binding(${f}) var<storage, read_write> outputCounts: array<atomic<u32>>;
${o?`var<workgroup> localCounts: array<atomic<u32>, ${t.output.length}>;`:``}
${s?w():``}

${A}

@compute @workgroup_size(${W}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(local_invocation_id) localId: vec3<u32>${s?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${l(a,W)}
  let lane = localId.x;
  ${p}
  ${o?`if (lane < CELL_COUNT) { atomicStore(&localCounts[lane], 0u); }
  workgroupBarrier();`:``}
  var accepted = false;
  var cellIndex = 0u;
  let finiteBounds = isFiniteGridValue(minimumX) && isFiniteGridValue(minimumY) &&
    isFiniteGridValue(maximumX) && isFiniteGridValue(maximumY);
  let orderedBounds = minimumX <= maximumX && minimumY <= maximumY;
  if (index < ELEMENT_COUNT && finiteBounds && orderedBounds) {
    let x = positions[POSITIONS_OFFSET + index * 2u];
    let y = positions[POSITIONS_OFFSET + index * 2u + 1u];
    let finite = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
    let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
    if (finite && inX && inY) {
      let column = getGridCoordinate(x, minimumX, maximumX, WIDTH);
      let row = getGridCoordinate(y, minimumY, maximumY, HEIGHT);
      let globalCell = row * WIDTH + column;
      accepted = globalCell >= OUTPUT_START && globalCell - OUTPUT_START < CELL_COUNT;
      if (accepted) { cellIndex = globalCell - OUTPUT_START; }
    }
  }
  ${h}
}`,v=[{buffer:t.positions,usage:`storage-read`},...c?[{buffer:t.bounds,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`}];n.push(...O(e,{id:`${t.id}-${o?`local`:`global`}`,source:_,resources:v,bindings:{positions:t.positions,...c?{boundsValues:t.bounds}:{},outputCounts:t.output},dispatch:a}));let y=T(e,()=>({op:`GPUGridBinning`,moduleId:`GPUGridBinning/${o?`local`:`global`}-accumulate${s?`/subgroups`:``}`,source:_,workgroupSize:[W,1,1],workgroups:[a.x,a.y,a.z],invocationCount:t.positions.length,formula:`gpu-core.grid-binning`,bindings:[{name:`positions`,view:t.positions,access:`read`},...c?[{name:`boundsValues`,view:t.bounds,access:`read`}]:[],{name:`outputCounts`,view:t.output,access:`read_write`}]}));if(y&&(n[0].annotation=y),e.recording)for(let e of n)e.workload={operation:`GPUGridBinning`,variant:o?`local-accumulate`:`global-accumulate`,commandCount:1,maximumWorkgroupCount:a.x*a.y*a.z,maximumInvocationCount:a.x*a.y*a.z*W,readByteLength:t.positions.length*t.positions.rowByteLength+(c?16:0)+t.output.length*t.output.rowByteLength,writeByteLength:t.output.length*t.output.rowByteLength};return n}function G(e){return!Array.isArray(e)}function K(e){return e instanceof c?e.data:[e]}var q=256,De=16,Oe=class{id;positions;weights;output;operation;gridSize;bounds;constructor(e){this.id=e.id??`gpu-grid-aggregation`,this.positions=e.positions,this.weights=e.weights,this.output=e.output,this.operation=e.operation??`sum`,this.gridSize=e.gridSize,this.bounds=e.bounds;for(let e of X(this.positions))o(e,[`float32x2`],`${this.id} positions`);for(let e of Z(this.weights))o(e,[`float32`],`${this.id} weights`);if(this.positions.length!==this.weights.length)throw Error(`${this.id} positions and weights must contain the same number of rows`);for(let e of C(this.output))o(e,[`float32`],`${this.id} output`);ie(this.id,[...X(this.positions),...Z(this.weights)],C(this.output));let[t,n]=this.gridSize;if(!Number.isSafeInteger(t)||!Number.isSafeInteger(n)||t<=0||n<=0)throw Error(`${this.id} gridSize must contain two positive integers`);if(!Number.isSafeInteger(t*n)||t*n>4294967295||this.output.length!==t*n)throw Error(`${this.id} output.length must equal gridSize width * height`);if(![`sum`,`min`,`max`,`mean`].includes(this.operation))throw Error(`${this.id} operation must be sum, min, max, or mean`);if(this.operation===`mean`&&this.positions.length>4294967295)throw Error(`${this.id} mean input length must fit in uint32 cell counts`);if(X(this.positions).some(e=>C(this.output).some(t=>e.buffer===t.buffer))||Z(this.weights).some(e=>C(this.output).some(t=>e.buffer===t.buffer)))throw Error(`${this.id} inputs and output must use separate buffers`);if(Array.isArray(this.bounds)){let[e,t,n,r]=this.bounds;if(this.bounds.length!==4||!this.bounds.every(D)||e>n||t>r)throw Error(`${this.id} literal bounds must be finite [minX, minY, maxX, maxY]`)}else if(Y(this.bounds)){if(o(this.bounds,[`float32x4`],`${this.id} bounds`),this.bounds.length!==1)throw Error(`${this.id} GPU bounds must contain one float32x4 row`);if(C(this.output).some(e=>e.buffer===this.bounds.buffer))throw Error(`${this.id} bounds and output must use separate buffers`)}}getCommandNodes(e){let t=[],n=X(this.positions),r=Z(this.weights);if(n.some(t=>t.buffer.graph!==e)||r.some(t=>t.buffer.graph!==e)||C(this.output).some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(Y(this.bounds)&&this.bounds.buffer.graph!==e)throw Error(`${this.id} bounds must belong to the target graph`);let i=ee(e,[this.positions,this.weights]),a=0;for(let[n,r]of C(this.output).entries()){if(r.length){let o=`${this.id}-output-${n}`,s=this.operation===`mean`?y(e,`${o}-counts`,`uint32`,r.length):void 0;t.push(...ke(e,o,r,this.operation,s));for(let[n,[c,l]]of i.entries())t.push(...Ae(e,{id:`${o}-input-${n}`,positions:c,weights:l,output:r,outputStart:a,operation:this.operation,counts:s,gridSize:this.gridSize,bounds:this.bounds}));this.operation!==`sum`&&t.push(...je(e,o,r,this.operation,s))}a+=r.length}return t}};function ke(e,t,n,r,i){let a=[],o=b(t,n.length,q,e.device.limits.maxComputeWorkgroupsPerDimension),s=r===`min`?`0xffffffffu`:`0u`,c=i?`@group(0) @binding(1) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,u=i?`atomicStore(&outputCounts[${g(i)}u + index], 0u);`:``,d=`
const CELL_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${g(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<atomic<u32>>;
${c}
@compute @workgroup_size(${q}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(o,q)}
  if (index < CELL_COUNT) {
    atomicStore(&outputValues[OUTPUT_OFFSET + index], ${s});
    ${u}
  }
}`;a.push(...O(e,{id:r===`sum`?`${t}-clear`:`${t}-initialize`,source:d,resources:[{buffer:n,usage:`storage-write`},...i?[{buffer:i,usage:`storage-write`}]:[]],bindings:{outputValues:n,...i?{outputCounts:i}:{}},dispatch:o}));let f=T(e,()=>({op:`GPUGridAggregation`,moduleId:`GPUGridAggregation/${r===`sum`?`clear`:`initialize`}`,source:d,workgroupSize:[q,1,1],workgroups:[o.x,o.y,o.z],invocationCount:n.length,bindings:[{name:`outputValues`,view:n,access:`write`},...i?[{name:`outputCounts`,view:i,access:`write`}]:[]]}));return f&&(a[0].annotation=f),a}function Ae(e,t){let n=[],[r,i]=t.gridSize,a=b(t.id,t.positions.length,q,e.device.limits.maxComputeWorkgroupsPerDimension),o=t.output.length<=De&&E(e.device)===`subgroups`,s=Y(t.bounds),c=t.bounds,u=s?`@group(0) @binding(2) var<storage, read> boundsValues: array<f32>;`:``,d=s?3:2,f=t.counts?`@group(0) @binding(${d+1}) var<storage, read_write> outputCounts: array<atomic<u32>>;`:``,p=s?`let minimumX = boundsValues[BOUNDS_OFFSET];
  let minimumY = boundsValues[BOUNDS_OFFSET + 1u];
  let maximumX = boundsValues[BOUNDS_OFFSET + 2u];
  let maximumY = boundsValues[BOUNDS_OFFSET + 3u];`:`let minimumX = ${k(c[0])};
  let minimumY = ${k(c[1])};
  let maximumX = ${k(c[2])};
  let maximumY = ${k(c[3])};`,m=o?Ne(t.operation,t.output.length,t.counts):`  if (accepted) {
    ${J(t.operation)}
    ${t.counts?`atomicAdd(&outputCounts[${g(t.counts)}u + cellIndex], 1u);`:``}
  }`,h=`
${o?`enable subgroups;`:``}
const ELEMENT_COUNT: u32 = ${t.positions.length}u;
const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${i}u;
const OUTPUT_START: u32 = ${t.outputStart}u;
const CELL_COUNT: u32 = ${t.output.length}u;
const POSITIONS_OFFSET: u32 = ${g(t.positions)}u;
const WEIGHTS_OFFSET: u32 = ${g(t.weights)}u;
${s?`const BOUNDS_OFFSET: u32 = ${g(t.bounds)}u;`:``}
const OUTPUT_OFFSET: u32 = ${g(t.output)}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
${u}
@group(0) @binding(${d}) var<storage, read_write> outputValues: array<atomic<u32>>;
${f}

${A}

${Me(t.operation)}
${o?w():``}

@compute @workgroup_size(${q}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32${o?`,
  @builtin(subgroup_invocation_id) subgroupInvocationId: u32`:``}
) {
  ${l(a,q)}
  ${p}
  var accepted = false;
  var cellIndex = 0u;
  var weight = 0.0;
  let finiteBounds = isFiniteGridValue(minimumX) && isFiniteGridValue(minimumY) &&
    isFiniteGridValue(maximumX) && isFiniteGridValue(maximumY);
  let orderedBounds = minimumX <= maximumX && minimumY <= maximumY;
  if (index < ELEMENT_COUNT && finiteBounds && orderedBounds) {
    let x = positions[POSITIONS_OFFSET + index * 2u];
    let y = positions[POSITIONS_OFFSET + index * 2u + 1u];
    weight = weights[WEIGHTS_OFFSET + index];
    let finitePosition = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let finiteWeight = weight == weight && abs(weight) <= 3.402823466e+38;
    let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
    let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
    if (finitePosition && finiteWeight && inX && inY) {
      let column = getGridCoordinate(x, minimumX, maximumX, WIDTH);
      let row = getGridCoordinate(y, minimumY, maximumY, HEIGHT);
      let globalCell = row * WIDTH + column;
      accepted = globalCell >= OUTPUT_START && globalCell - OUTPUT_START < CELL_COUNT;
      if (accepted) { cellIndex = globalCell - OUTPUT_START; }
    }
  }
${m}
}`,_=[{buffer:t.positions,usage:`storage-read`},{buffer:t.weights,usage:`storage-read`},...s?[{buffer:t.bounds,usage:`storage-read`}]:[],{buffer:t.output,usage:`storage-read-write`},...t.counts?[{buffer:t.counts,usage:`storage-read-write`}]:[]];n.push(...O(e,{id:`${t.id}-${t.operation}`,source:h,resources:_,bindings:{positions:t.positions,weights:t.weights,...s?{boundsValues:t.bounds}:{},outputValues:t.output,...t.counts?{outputCounts:t.counts}:{}},dispatch:a}));let v=T(e,()=>({op:`GPUGridAggregation`,moduleId:`GPUGridAggregation/accumulate-${t.operation}${o?`/subgroups`:``}`,source:h,workgroupSize:[q,1,1],workgroups:[a.x,a.y,a.z],invocationCount:t.positions.length,formula:`gpu-core.grid-aggregation`,bindings:[{name:`positions`,view:t.positions,access:`read`},{name:`weights`,view:t.weights,access:`read`},...s?[{name:`boundsValues`,view:t.bounds,access:`read`}]:[],{name:`outputValues`,view:t.output,access:`read_write`},...t.counts?[{name:`outputCounts`,view:t.counts,access:`read_write`}]:[]]}));return v&&(n[0].annotation=v),n}function je(e,t,n,r,i){let a=[],o=b(t,n.length,q,e.device.limits.maxComputeWorkgroupsPerDimension),s=i?`@group(0) @binding(1) var<storage, read> outputCounts: array<u32>;`:``,c=r===`mean`?`let count = outputCounts[${g(i)}u + index];
    if (count == 0u) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      let sum = bitcast<f32>(outputValues[OUTPUT_OFFSET + index]);
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(sum / f32(count));
    }`:`let orderedValue = outputValues[OUTPUT_OFFSET + index];
    if (orderedValue == ${r===`min`?`0xffffffffu`:`0u`}) {
      outputValues[OUTPUT_OFFSET + index] = 0x7fc00000u;
    } else {
      outputValues[OUTPUT_OFFSET + index] = bitcast<u32>(decodeOrderedFloat(orderedValue));
    }`,u=r===`mean`?``:`fn decodeOrderedFloat(value: u32) -> f32 {
  let bits = select(~value, value ^ 0x80000000u, (value & 0x80000000u) != 0u);
  return bitcast<f32>(bits);
}`,d=`
const CELL_COUNT: u32 = ${n.length}u;
const OUTPUT_OFFSET: u32 = ${g(n)}u;
@group(0) @binding(0) var<storage, read_write> outputValues: array<u32>;
${s}
${u}
@compute @workgroup_size(${q}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${l(o,q)}
  if (index < CELL_COUNT) {
    ${c}
  }
}`;a.push(...O(e,{id:`${t}-finalize`,source:d,resources:[{buffer:n,usage:`storage-read-write`},...i?[{buffer:i,usage:`storage-read`}]:[]],bindings:{outputValues:n,...i?{outputCounts:i}:{}},dispatch:o}));let f=T(e,()=>({op:`GPUGridAggregation`,moduleId:`GPUGridAggregation/finalize-${r}`,source:d,workgroupSize:[q,1,1],workgroups:[o.x,o.y,o.z],invocationCount:n.length,formula:`gpu-core.grid-aggregation`,bindings:[{name:`outputValues`,view:n,access:`read_write`},...i?[{name:`outputCounts`,view:i,access:`read`}]:[]]}));return f&&(a[0].annotation=f),a}function Me(e){return e===`min`||e===`max`?`fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`:`fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var oldBits = atomicLoad(destination);
  loop {
    let newBits = bitcast<u32>(bitcast<f32>(oldBits) + value);
    let result = atomicCompareExchangeWeak(destination, oldBits, newBits);
    if (result.exchanged) { break; }
    oldBits = result.old_value;
  }
}`}function J(e,t=`cellIndex`,n=`weight`){return e===`min`||e===`max`?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + ${t}], encodeOrderedFloat(${n}));`:`atomicAddFloat(&outputValues[OUTPUT_OFFSET + ${t}], ${n});`}function Ne(e,t,n){let r=e===`min`||e===`max`;return`
  var subgroupPending = accepted;
  for (var subgroupCell = 0u; subgroupCell < ${t}u; subgroupCell++) {
    let pendingBallot = subgroupBallot(subgroupPending);
    let hasPending = any(pendingBallot != vec4<u32>(0u));
    let leaderInvocation = getFirstBallotLane(pendingBallot);
    let leaderCell = subgroupShuffle(cellIndex, leaderInvocation);
    let matchingCell = hasPending && subgroupPending && cellIndex == leaderCell;
    let matchingBallot = subgroupBallot(matchingCell);
    let selectedWeight = ${r?`select(${e===`min`?`0xffffffffu`:`0u`}, encodeOrderedFloat(weight), matchingCell)`:`select(0.0, weight, matchingCell)`};
    let aggregatedWeight = ${e===`min`?`subgroupMin(selectedWeight)`:e===`max`?`subgroupMax(selectedWeight)`:`subgroupAdd(selectedWeight)`};
    if (hasPending && subgroupInvocationId == leaderInvocation) {
      ${r?`${e===`min`?`atomicMin`:`atomicMax`}(&outputValues[OUTPUT_OFFSET + leaderCell], aggregatedWeight);`:J(e,`leaderCell`,`aggregatedWeight`)}
      ${n?`atomicAdd(&outputCounts[${g(n)}u + leaderCell], getBallotLaneCount(matchingBallot));`:``}
    }
    subgroupPending = subgroupPending && !matchingCell;
  }`}function Y(e){return!Array.isArray(e)}function X(e){return e instanceof c?e.data:[e]}function Z(e){return e instanceof c?e.data:[e]}function Pe(e){if(!Number.isSafeInteger(e)||e<0||e>4294967295)throw Error(`fixed-point maximumTermCount must be an integer in [0, 2^32 - 1]`);return Math.max(1,e.toString(2).length)}var Q=`
fn getFixedPointExponentBits(value: f32) -> u32 {
  let magnitudeBits = bitcast<u32>(value) & 0x7fffffffu;
  if (magnitudeBits == 0u || magnitudeBits >= 0x7f800000u) { return 0u; }
  return max(magnitudeBits >> 23u, 1u);
}
fn getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  // A term with biased exponent E is below 2^(E - 126) = 2^(E - 127 + 1); 95 - h bits remain.
  return i32(max(maximumExponentBits, 1u)) - 127 + 1 + i32(headroomBits) - 95;
}
fn addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
  let low = a.x + b.x;
  let lowCarry = select(0u, 1u, low < a.x);
  let middlePartial = a.y + b.y;
  let middleCarry = select(0u, 1u, middlePartial < a.y);
  let middle = middlePartial + lowCarry;
  let middleRoundCarry = select(0u, 1u, middle < middlePartial);
  return vec3<u32>(low, middle, a.z + b.z + middleCarry + middleRoundCarry);
}
fn negateFixedPoint(value: vec3<u32>) -> vec3<u32> {
  return addFixedPoint(~value, vec3<u32>(1u, 0u, 0u));
}
fn shiftFixedPointLeft(value: u32, shift: u32) -> vec3<u32> {
  var words = array<u32, 4>(0u, 0u, 0u, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  words[word] = value << bit;
  // (value >> 1) >> (31 - bit) is value >> (32 - bit) without a shift by 32 when bit is 0.
  words[word + 1u] = (value >> 1u) >> (31u - bit);
  return vec3<u32>(words[0], words[1], words[2]);
}
fn shiftFixedPointRight(value: vec3<u32>, shift: u32) -> u32 {
  var words = array<u32, 4>(value.x, value.y, value.z, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  return (words[word] >> bit) | ((words[word + 1u] << 1u) << (31u - bit));
}
fn getFixedPointBit(value: vec3<u32>, position: u32) -> u32 {
  var words = array<u32, 3>(value.x, value.y, value.z);
  return (words[position / 32u] >> (position % 32u)) & 1u;
}
fn hasFixedPointBitsBelow(value: vec3<u32>, count: u32) -> bool {
  var words = array<u32, 3>(value.x, value.y, value.z);
  let word = count / 32u;
  var found = (words[word] & ((1u << (count % 32u)) - 1u)) != 0u;
  if (word > 0u) { found = found || words[0] != 0u; }
  if (word > 1u) { found = found || words[1] != 0u; }
  return found;
}
fn quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32> {
  let bits = bitcast<u32>(value);
  let exponentBits = (bits >> 23u) & 0xffu;
  if (exponentBits == 0xffu) { return vec3<u32>(0u); }
  var significand = bits & 0x7fffffu;
  var exponent = -149;
  if (exponentBits != 0u) {
    significand = significand | 0x800000u;
    exponent = i32(exponentBits) - 150;
  }
  // value = significand * 2^exponent; the term in units is significand * 2^shift.
  let shift = exponent - unitExponent;
  var magnitude = vec3<u32>(0u);
  if (shift >= 0) {
    // The scale bounds shift by 71 - h; the clamp only keeps a misused scale memory-safe.
    magnitude = shiftFixedPointLeft(significand, u32(min(shift, 71)));
  } else if (shift >= -24) {
    let right = u32(-shift);
    var quotient = significand >> right;
    let remainder = significand & ((1u << right) - 1u);
    let half = 1u << (right - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
      quotient = quotient + 1u;
    }
    magnitude = vec3<u32>(quotient, 0u, 0u);
  }
  return select(magnitude, negateFixedPoint(magnitude), (bits >> 31u) != 0u);
}
fn getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32 {
  let negative = (value.z >> 31u) != 0u;
  let magnitude = select(value, negateFixedPoint(value), negative);
  var leading = 0u;
  if (magnitude.z != 0u) {
    leading = 64u + firstLeadingBit(magnitude.z);
  } else if (magnitude.y != 0u) {
    leading = 32u + firstLeadingBit(magnitude.y);
  } else if (magnitude.x != 0u) {
    leading = firstLeadingBit(magnitude.x);
  } else {
    return 0.0;
  }
  var significand = magnitude.x;
  var exponent = unitExponent;
  if (leading >= 24u) {
    let shift = leading - 23u;
    significand = shiftFixedPointRight(magnitude, shift);
    let roundBit = getFixedPointBit(magnitude, shift - 1u);
    if (roundBit != 0u && (hasFixedPointBitsBelow(magnitude, shift - 1u) || (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
    exponent = unitExponent + i32(shift);
  }
  // significand <= 2^24, so f32(significand) is exact and the value is significand * 2^exponent.
  var infinityBits = 0x7f800000u;
  var result = bitcast<f32>(infinityBits);
  if (exponent < 104 || (exponent == 104 && significand < 0x1000000u)) {
    // Two steps keep the ldexp exponent in range for normal results below 2^-103.
    let leadingExponent = i32(firstLeadingBit(significand));
    result = ldexp(ldexp(f32(significand), -leadingExponent), exponent + leadingExponent);
  }
  return select(result, -result, negative);
}`;function Fe(e){return`
fn ${`addFixedPointTo${e[0].toUpperCase()}${e.slice(1)}`}(cell: u32, value: vec3<u32>) {
  let base = ${e}Offset + cell * 3u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&${e}[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&${e}[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&${e}[base + 2u], highAddend);
  }
}`}function Ie(e,t){let n=[N(e,{id:`${t.id}-clear`,operation:t.operation,variant:`fixed-point-scale-clear`,bindings:[{name:`fixedPointScale`,view:t.scale,type:`u32`,access:`read_write`}],invocationCount:1,body:`fixedPointScale[fixedPointScaleOffset] = 0u;`})],r=t.values instanceof c?t.values.data:[t.values];for(let[i,a]of r.entries())a.length!==0&&n.push(N(e,{id:`${t.id}-${i}`,operation:t.operation,variant:`fixed-point-scale`,bindings:[{name:`fixedPointValues`,view:a,type:`f32`,access:`read`},{name:`fixedPointScale`,view:t.scale,type:`atomic<u32>`,access:`read_write`}],invocationCount:a.length,guardIndex:!1,declarations:`${Q}
var<workgroup> workgroupExponentBits: atomic<u32>;`,body:`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&fixedPointScale[fixedPointScaleOffset], workgroupMaximum);
    }
  }`}));return n}function Le(e,t){return N(e,{id:t.id,operation:t.operation,variant:`fixed-point-clear`,bindings:[{name:`fixedPointWords`,view:t.words,type:`u32`,access:`read_write`}],invocationCount:t.words.length,workgroupSize:64,body:`fixedPointWords[fixedPointWordsOffset + index] = 0u;`})}function Re(e,t){if(t.words.length<t.output.length*3)throw Error(`${t.operation}: fixed-point words must hold three words per output`);return N(e,{id:t.id,operation:t.operation,variant:`fixed-point-decode`,bindings:[{name:`fixedPointWords`,view:t.words,type:`u32`,access:`read`},{name:`fixedPointScale`,view:t.scale,type:`u32`,access:`read`},{name:`fixedPointOutput`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:t.output.length,workgroupSize:64,declarations:`override FIXED_POINT_HEADROOM_BITS: u32;
${Q}`,constants:{FIXED_POINT_HEADROOM_BITS:Pe(t.maximumTermCount)},body:`let base = fixedPointWordsOffset + index * 3u;
  let unitExponent = getFixedPointUnitExponent(
    fixedPointScale[fixedPointScaleOffset],
    FIXED_POINT_HEADROOM_BITS
  );
  fixedPointOutput[fixedPointOutputOffset + index] = getFixedPointFloat(
    vec3<u32>(fixedPointWords[base], fixedPointWords[base + 1u], fixedPointWords[base + 2u]),
    unitExponent
  );`})}var $=`GPUPointDensity`;function ze(e,t,n){return N(e,{id:t,operation:$,variant:`clear-overflow`,bindings:[{name:`overflowFlags`,view:n,type:`u32`,access:`read_write`}],invocationCount:1,body:`overflowFlags[overflowFlagsOffset] = 0u;`})}function Be(e,t){let n=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`}];t.mask&&n.push({name:`rowMask`,view:t.mask,type:`u32`,access:`read`});let r=!Array.isArray(t.bounds);r&&n.push({name:`boundsValues`,view:t.bounds,type:`f32`,access:`read`}),typeof t.radius!=`number`&&n.push({name:`radiusValues`,view:t.radius,type:`f32`,access:`read`}),t.overflow&&n.push({name:`overflowFlags`,view:t.overflow,type:`atomic<u32>`,access:`read_write`});let i=r?void 0:t.bounds.map(M),a=i?`let minimumX = ${i[0]}; let minimumY = ${i[1]};
  let maximumX = ${i[2]}; let maximumY = ${i[3]};`:`let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`,o=typeof t.radius==`number`?`let radius = ${M(t.radius)};`:`let radius = radiusValues[radiusValuesOffset];`;return N(e,{id:t.id,operation:$,variant:`hexagon-keys`,bindings:n,invocationCount:t.positions.length,declarations:`const KEY_START: u32 = ${t.keyStart}u;
const COLUMNS: u32 = ${t.gridSize[0]}u;
const ROWS: u32 = ${t.gridSize[1]}u;
const MASK_START: u32 = ${t.maskStart??0}u;
const INVALID_KEY: u32 = 0xffffffffu;
const MAXIMUM_FLOAT: f32 = 3.402823466e+38;
${se}`,body:`${a}
  ${o}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = INVALID_KEY;
  let included = ${t.mask?`rowMask[rowMaskOffset + MASK_START + index] != 0u`:`true`};
  let finite = included && abs(x) <= MAXIMUM_FLOAT && abs(y) <= MAXIMUM_FLOAT;
  let finiteBounds = abs(minimumX) <= MAXIMUM_FLOAT && abs(minimumY) <= MAXIMUM_FLOAT &&
    abs(maximumX) <= MAXIMUM_FLOAT && abs(maximumY) <= MAXIMUM_FLOAT;
  let orderedBounds = minimumX <= maximumX && minimumY <= maximumY;
  let inside = x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY;
  let validRadius = radius > 0.0 && radius <= MAXIMUM_FLOAT;
  if (finite && finiteBounds && orderedBounds && inside && validRadius) {
    let cell = getPointDensityHexagonCell(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < COLUMNS && u32(cell.y) < ROWS) {
      key = u32(cell.y) * COLUMNS + u32(cell.x);
    } else {
      ${t.overflow?`atomicStore(&overflowFlags[overflowFlagsOffset], 1u);`:``}
    }
  }
  keys[keysOffset + KEY_START + index] = key;`})}function Ve(e,t){return N(e,{id:t.id,operation:$,variant:`mask-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`rowMask`,view:t.mask,type:`u32`,access:`read`},{name:`maskedPositions`,view:t.maskedPositions,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`const ROW_START: u32 = ${t.rowStart}u;`,body:`var nanBits = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let included = rowMask[rowMaskOffset + ROW_START + index] != 0u;
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let slot = maskedPositionsOffset + (ROW_START + index) * 2u;
  maskedPositions[slot] = select(nan, x, included);
  maskedPositions[slot + 1u] = select(nan, y, included);`})}function He(e,t){let n=[{name:`counts`,view:t.counts,type:`u32`,access:`read`}];t.sums&&n.push({name:`sums`,view:t.sums,type:`f32`,access:`read`}),n.push({name:`values`,view:t.values,type:`f32`,access:`read_write`}),t.means&&n.push({name:`means`,view:t.means,type:`f32`,access:`read_write`});let r=t.statistic===`count`?`f32(count)`:t.statistic===`sum`?`sum`:`mean`;return N(e,{id:t.id,operation:$,variant:`finalize`,record:t.statistic===`count`?{formula:`gpu-spatial-analysis.point-density`}:void 0,bindings:n,invocationCount:t.values.length,body:`let count = counts[countsOffset + index];
  let sum = ${t.sums?`sums[sumsOffset + index]`:`f32(count)`};
  let mean = select(0.0, sum / f32(max(count, 1u)), count > 0u);
  values[valuesOffset + index] = ${r};
  ${t.means?`means[meansOffset + index] = mean;`:``}`})}function Ue(e,t){let n=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`}],r=!Array.isArray(t.bounds);r&&n.push({name:`boundsValues`,view:t.bounds,type:`f32`,access:`read`});let i=r?void 0:t.bounds.map(M),a=i?`let minimumX = ${i[0]}; let minimumY = ${i[1]};
  let maximumX = ${i[2]}; let maximumY = ${i[3]};`:`let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;return N(e,{id:t.id,operation:$,variant:`grid-keys`,bindings:n,invocationCount:t.positions.length,declarations:`const KEY_START: u32 = ${t.keyStart}u;
const COLUMNS: u32 = ${t.gridSize[0]}u;
const ROWS: u32 = ${t.gridSize[1]}u;
${ae}
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,body:`${a}
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var key = 0xffffffffu;
  let finite = isFiniteBits(x) && isFiniteBits(y);
  let finiteBounds = isFiniteBits(minimumX) && isFiniteBits(minimumY) &&
    isFiniteBits(maximumX) && isFiniteBits(maximumY);
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  if (finiteBounds && maximumX >= minimumX && maximumY >= minimumY && finite && inX && inY) {
    key = getCoordinate(y, minimumY, maximumY, ROWS) * COLUMNS + getCoordinate(x, minimumX, maximumX, COLUMNS);
  }
  keys[keysOffset + KEY_START + index] = key;`})}function We(e,t){return N(e,{id:t.id,operation:$,variant:`workgroup-sums`,bindings:[{name:`weights`,view:t.weights,type:`f32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read`},{name:`scale`,view:t.scale,type:`u32`,access:`read`},{name:`sums`,view:t.fixedPointSums,type:`atomic<u32>`,access:`read_write`}],workgroupSize:256,invocationCount:t.weights.length,guardIndex:!1,declarations:`const KEY_START: u32 = ${t.keyStart}u;
const CELL_COUNT: u32 = ${t.fixedPointSums.length/3}u;
override HEADROOM_BITS: u32;
${ae}
${Q}
${Fe(`sums`)}
var<workgroup> sortKeys: array<u32, 256>;
var<workgroup> sortValues: array<vec3<u32>, 256>;`,constants:{HEADROOM_BITS:Pe(t.rowCount)},body:`let lane = localInvocationIndex;
  let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
  var key = 0xffffffffu;
  var value = vec3<u32>(0u);
  if (index < INVOCATION_COUNT) {
    let weight = weights[weightsOffset + index];
    let rowKey = keys[keysOffset + KEY_START + index];
    if (isFiniteBits(weight) && rowKey < CELL_COUNT) {
      key = rowKey;
      value = quantizeFixedPoint(weight, unitExponent);
    }
  }
  sortKeys[lane] = key;
  sortValues[lane] = value;
  workgroupBarrier();
  // Every row sums the rows of its workgroup that share its key (exact integer adds). Only the
  // first row of each key issues the global add, so no sort or scan is needed.
  var runSum = vec3<u32>(0u);
  var isFirst = true;
  for (var other = 0u; other < 256u; other = other + 1u) {
    if (sortKeys[other] == key) {
      runSum = addFixedPoint(runSum, sortValues[other]);
      isFirst = isFirst && other >= lane;
    }
  }
  if (isFirst && key < CELL_COUNT) {
    addFixedPointToSums(key, runSum);
  }`})}var Ge=Object.freeze({...t,measure:`area`}),Ke=f(12);function qe(e){return e===`atomic`?`atomic`:`workgroup`}function Je(e){return e?e.separableKernel?[e.separableKernel.horizontal,e.separableKernel.vertical]:e.kernel?[e.kernel]:[]:[]}var Ye=class{id;props;spatialContext;binning;statistic;sumAccumulation;cellCount;surface;constructor(t){this.id=t.id??`point-density`,this.props=t,this.spatialContext=e(this.id,t.spatialContext,Ge),this.binning=t.binning??`grid`,this.statistic=t.statistic??`count`;let n=this.id,[r,i]=t.gridSize;this.cellCount=r*i,this.sumAccumulation=qe(t.sumAccumulation),this.surface={values:t.output.values,columns:r,rows:i};let{output:a,weights:o,smoothing:s,mask:c}=t;if(![`grid`,`hexagon`].includes(this.binning))throw Error(`${n} binning must be grid or hexagon`);if(![`count`,`sum`,`mean`].includes(this.statistic))throw Error(`${n} statistic must be count, sum, or mean`);if(![`auto`,`workgroup`,`atomic`].includes(t.sumAccumulation??`auto`))throw Error(`${n} sumAccumulation must be auto, workgroup, or atomic`);if(p(n,`gridSize columns`,r),p(n,`gridSize rows`,i),this.cellCount>=4294967295)throw Error(`${n} gridSize cell count must be below 2^32 - 1`);o&&this.sumAccumulation===`workgroup`&&v(n,`gridSize cells`,this.cellCount,Ke);for(let e of _(t.positions))x(e,[`float32x2`],`${n} positions`);if(o){for(let e of _(o))x(e,[`float32`],`${n} weights`);if(o.length!==t.positions.length)throw Error(`${n} weights length must equal positions length`)}for(let[e,t]of[[this.statistic!==`count`,`statistic ${this.statistic}`],[!!a.sums,`output.sums`],[!!a.means,`output.means`]])if(e&&!o)throw Error(`${n} ${t} requires weights`);if(c&&(x(c,[`uint32`],`${n} mask`),c.length!==t.positions.length))throw Error(`${n} mask length must equal positions length`);if(Xe(n,t.bounds),this.binning===`hexagon`){let e=t.hexagonRadius;if(e===void 0)throw Error(`${n} hexagon binning requires hexagonRadius`);if(typeof e==`number`){if(!Number.isFinite(e)||e<=0)throw Error(`${n} hexagonRadius must be positive and finite`)}else if(x(e,[`float32`],`${n} hexagonRadius`),e.length<1)throw Error(`${n} hexagonRadius must contain one float32 row`);if(s)throw Error(`${n} smoothing requires grid binning`)}else if(t.hexagonRadius!==void 0)throw Error(`${n} hexagonRadius requires hexagon binning`);if(s){for(let e of[s.kernelWidth,s.kernelHeight])if(!Number.isSafeInteger(e)||e<1||e%2==0)throw Error(`${n} smoothing kernel sizes must be positive odd integers`);let{separableKernel:e}=s;if(e){if(x(e.horizontal,[`float32`],`${n} smoothing.separableKernel.horizontal`),x(e.vertical,[`float32`],`${n} smoothing.separableKernel.vertical`),e.horizontal.length<s.kernelWidth||e.vertical.length<s.kernelHeight)throw Error(`${n} smoothing.separableKernel is shorter than kernelWidth or kernelHeight`)}else{if(!s.kernel)throw Error(`${n} smoothing requires kernel or separableKernel`);if(x(s.kernel,[`float32`],`${n} smoothing.kernel`),s.kernel.length<s.kernelWidth*s.kernelHeight)throw Error(`${n} smoothing.kernel is shorter than kernelWidth * kernelHeight`)}}let l=[[`values`,a.values,`float32`],[`counts`,a.counts,`uint32`],[`sums`,a.sums,`float32`],[`means`,a.means,`float32`]];for(let[e,t,r]of l)if(t&&(x(t,[r],`${n} output.${e}`),t.length!==this.cellCount))throw Error(`${n} output.${e} must have one row per cell`);if(a.extent&&(x(a.extent,[`float32`],`${n} output.extent`),a.extent.length!==2))throw Error(`${n} output.extent must have two rows`);for(let[e,t]of[[`histogram`,a.histogram],[`status.overflow`,a.status?.overflow]])if(t&&(x(t,[`uint32`],`${n} output.${e}`),t.length<1))throw Error(`${n} output.${e} must have at least one row`);if(a.texture&&(a.texture.format!==`r32float`||a.texture.width!==r||a.texture.height!==i))throw Error(`${n} output.texture must be r32float with gridSize extent`);let u=[a.values,a.counts,a.sums,a.means,a.extent,a.histogram,a.status?.overflow].filter(e=>e!==void 0).map(e=>e.buffer),d=[..._(t.positions),...o?_(o):[],...c?[c]:[],...Array.isArray(t.bounds)?[]:[t.bounds],...typeof t.hexagonRadius==`object`?[t.hexagonRadius]:[],...Je(s)].map(e=>e.buffer);if(new Set(u).size!==u.length||u.some(e=>d.includes(e)))throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:n,statistic:r,sumAccumulation:i,cellCount:a}=this,{positions:o,weights:s,smoothing:c,output:l,gridSize:u,mask:f}=n,p=Array.isArray(n.bounds)?void 0:n.bounds;if(d(t,e,[o,s,f,p,typeof n.hexagonRadius==`object`?n.hexagonRadius:void 0,...Je(c),l.values,l.counts,l.sums,l.means,l.extent,l.histogram,l.status?.overflow]),l.texture&&l.texture.texture.graph!==e)throw Error(`${t} views must belong to the target graph`);let h=p&&p.format===`float32`?e.createDataView(p.buffer,{format:`float32x4`,length:1,byteOffset:p.byteOffset}):n.bounds,g=l.counts??y(e,`${t}-counts-scratch`,`uint32`,a),v=s&&(r!==`count`||l.sums||l.means)?l.sums??y(e,`${t}-sums-scratch`,`float32`,a):void 0,b=c?y(e,`${t}-unsmoothed-values`,`float32`,a):l.values,x=l.extent??(l.histogram?y(e,`${t}-extent-scratch`,`float32`,2):void 0),S=[];if(l.status?.overflow&&S.push(ze(e,`${t}-clear-overflow`,l.status.overflow)),this.binning===`hexagon`){let r=y(e,`${t}-hexagon-keys`,`uint32`,o.length),a=0;for(let[i,s]of _(o).entries())s.length>0&&S.push(Be(e,{id:`${t}-hexagon-keys-${i}`,positions:s,keys:r,keyStart:a,gridSize:u,bounds:h,radius:n.hexagonRadius,mask:f,maskStart:a,overflow:l.status?.overflow})),a+=s.length;S.push(...new j({id:`${t}-counts`,keys:r,output:g}).getCommandNodes(e)),v&&s&&(i===`atomic`?S.push(...new j({id:`${t}-sums`,keys:r,values:s,output:v,operation:`sum`}).getCommandNodes(e)):S.push(...Ze(e,`${t}-sums`,s,r,v)))}else{let n=o;if(f){let r=y(e,`${t}-masked-positions`,`float32x2`,o.length),i=0;for(let[n,a]of _(o).entries())a.length>0&&S.push(Ve(e,{id:`${t}-mask-positions-${n}`,positions:a,mask:f,maskedPositions:r,rowStart:i})),i+=a.length;n=r}if(S.push(...new we({id:`${t}-counts`,positions:n,output:g,gridSize:u,bounds:h}).getCommandNodes(e)),v&&s)if(i===`atomic`)S.push(...new Oe({id:`${t}-sums`,positions:n,weights:s,output:v,operation:`sum`,gridSize:u,bounds:h}).getCommandNodes(e));else{let r=y(e,`${t}-grid-keys`,`uint32`,o.length),i=0;for(let[a,o]of _(n).entries())o.length>0&&S.push(Ue(e,{id:`${t}-sums-grid-keys-${a}`,positions:o,keys:r,keyStart:i,gridSize:u,bounds:h})),i+=o.length;S.push(...Ze(e,`${t}-sums`,s,r,v))}}if(S.push(He(e,{id:`${t}-finalize`,statistic:r,counts:g,sums:v,values:b,means:l.means})),c?.separableKernel){let n=y(e,`${t}-smooth-rows`,`float32`,a),r={width:u[0],height:u[1],boundary:`zero`};S.push(...new F({...r,id:`${t}-smooth-horizontal`,input:b,kernel:c.separableKernel.horizontal,output:n,kernelWidth:c.kernelWidth,kernelHeight:1,strategy:`direct`}).getCommandNodes(e),...new F({...r,id:`${t}-smooth-vertical`,input:n,kernel:c.separableKernel.vertical,output:l.values,kernelWidth:1,kernelHeight:c.kernelHeight,strategy:`direct`}).getCommandNodes(e))}else c&&S.push(...new F({id:`${t}-smooth`,input:b,kernel:c.kernel,output:l.values,width:u[0],height:u[1],kernelWidth:c.kernelWidth,kernelHeight:c.kernelHeight,boundary:`zero`,strategy:c.strategy??`auto`}).getCommandNodes(e));let C=c?void 0:g;if(x&&S.push(...new ne({id:`${t}-extent`,input:l.values,mask:C,output:x,operation:`extent`}).getCommandNodes(e)),l.histogram&&x&&S.push(...new re({id:`${t}-histogram`,input:l.values,mask:C,output:l.histogram,domain:x}).getCommandNodes(e)),l.texture){let n=l.texture;S.push(...m(e,()=>new oe({id:`${t}-texture`,input:{id:`${t}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:l.values}},output:n}).addToGraph(e)))}return S}};function Xe(e,t){if(Array.isArray(t)){let[n,r,i,a]=t;if(t.length!==4||!t.every(e=>Number.isFinite(Math.fround(e)))||n>i||r>a)throw Error(`${e} bounds must be finite float32 [minX, minY, maxX, maxY]`);return}let n=t;if(x(n,[`float32x4`,`float32`],`${e} bounds`),n.format===`float32x4`&&n.length!==1||n.format===`float32`&&n.length!==4)throw Error(`${e} bounds view must be one float32x4 row or four float32 rows`)}function Ze(e,t,n,r,i){let a=y(e,`${t}-scale`,`uint32`,1),o=y(e,`${t}-fixed-point`,`uint32`,i.length*3),s=[...Ie(e,{id:`${t}-scale`,operation:`GPUPointDensity`,values:n,scale:a}),Le(e,{id:`${t}-clear`,operation:`GPUPointDensity`,words:o})],c=0;for(let[i,l]of _(n).entries())l.length>0&&s.push(We(e,{id:`${t}-workgroup-${i}`,weights:l,keys:r,keyStart:c,fixedPointSums:o,scale:a,rowCount:n.length})),c+=l.length;return s.push(Re(e,{id:`${t}-decode`,operation:`GPUPointDensity`,words:o,scale:a,maximumTermCount:n.length,output:i})),s}function Qe(e,t){if(!Number.isInteger(e)||e<0||e>32)throw Error(`Gaussian kernel radius must be an integer in [0, 32]`);let n=t??Math.max(e/2,.5);if(!Number.isFinite(n)||n<=0)throw Error(`Gaussian kernel sigma must be positive and finite`);let r=2*e+1,i=new Float64Array(r*r),a=0;for(let t=0;t<r;t++)for(let o=0;o<r;o++){let s=o-e,c=t-e,l=Math.exp(-(s*s+c*c)/(2*n*n));i[t*r+o]=l,a+=l}return Float32Array.from(i,e=>e/a)}function $e(e,t){if(!Number.isInteger(e)||e<0||e>32)throw Error(`Gaussian kernel radius must be an integer in [0, 32]`);let n=t??Math.max(e/2,.5);if(!Number.isFinite(n)||n<=0)throw Error(`Gaussian kernel sigma must be positive and finite`);let r=new Float64Array(2*e+1),i=0;for(let t=0;t<r.length;t++){let a=t-e;r[t]=Math.exp(-(a*a)/(2*n*n)),i+=r[t]}return Float32Array.from(r,e=>e/i)}export{$e as n,Ye as r,Qe as t};