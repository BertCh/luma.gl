import{M as e,_ as t,i as n,k as r,w as i}from"./resources-CwvhEOBe.js";import{u as a}from"./interleave-brZZCO-f.js";import{t as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as s}from"./wgsl-common-BNRVzgd8.js";import{c,g as l,i as u,l as d,n as f,p,r as m,s as h,u as g}from"./raster-utils-DhdnkkOa.js";var _=8,v=class{id;width;height;input;output;outputValidity;radius;horizontalRadius;verticalRadius;requiredHalo;kernel;borderMode;borderValue;noDataPolicy;normalize;kernelSum;constructor(e){if(this.id=e.id??`gpu-raster-neighborhood`,this.width=e.width,this.height=e.height,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.radius=e.radius,this.borderMode=e.borderMode??`clamp`,this.borderValue=e.borderValue??0,this.noDataPolicy=e.noDataPolicy??`propagate`,this.normalize=e.normalize??!1,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count must fit in uint32`);let n=typeof this.radius==`number`?[this.radius,this.radius]:this.radius;if(!Array.isArray(n)||n.length!==2||!n.every(e=>Number.isSafeInteger(e)&&e>=0&&e<=_))throw Error(`${this.id} neighborhood radii must be integers from zero through eight`);this.horizontalRadius=n[0],this.verticalRadius=n[1],this.requiredHalo=Math.max(this.horizontalRadius,this.verticalRadius);let r=(this.horizontalRadius*2+1)*(this.verticalRadius*2+1);if(!Array.isArray(e.kernel)||e.kernel.length!==r)throw Error(`${this.id} kernel must contain exactly one coefficient per neighborhood`);for(let t of e.kernel){if(!Number.isFinite(t))throw Error(`${this.id} kernel coefficients must be finite`);u(t)}if(this.kernel=Object.freeze(e.kernel.map(e=>Math.fround(e))),this.kernelSum=this.kernel.reduce((e,t)=>e+t,0),u(this.kernelSum),![`clamp`,`reflect`,`constant`,`nodata`].includes(this.borderMode))throw Error(`${this.id} border mode must be clamp, reflect, constant, or nodata`);if(!Number.isFinite(this.borderValue))throw Error(`${this.id} constant border value must be finite`);if(u(this.borderValue),![`propagate`,`ignore-renormalize`].includes(this.noDataPolicy))throw Error(`${this.id} nodata policy must be propagate or ignore-renormalize`);if(typeof this.normalize!=`boolean`)throw Error(`${this.id} normalize must be a boolean`);if(this.noDataPolicy===`ignore-renormalize`&&this.kernel.some(e=>e<0))throw Error(`${this.id} ignore-renormalize requires nonnegative kernel coefficients`);if((this.normalize||this.noDataPolicy===`ignore-renormalize`)&&this.kernelSum===0)throw Error(`${this.id} normalized kernels must have a nonzero coefficient sum`);if(this.input.storage.kind!==`buffer`)throw Error(`${this.id} requires a buffer-backed input band`);let i=g(this.input,this,`${this.id} input`);if(p(this.output,`float32`,t,`${this.id} output`),l(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==i||this.outputValidity.buffer.graph!==i)throw Error(`${this.id} resources must belong to the same graph`);let a=[this.input.storage.values.buffer,...this.input.validity?[this.input.validity.buffer]:[]];if(this.output.buffer===this.outputValidity.buffer||a.includes(this.output.buffer)||a.includes(this.outputValidity.buffer))throw Error(`${this.id} neighborhood inputs and outputs must use separate buffers`);u(this.input.scale??1),u(this.input.offset??0)}addToGraph(e){let t=[this.input.storage.values,this.output,this.outputValidity,...this.input.validity?[this.input.validity]:[]];for(let n of t){if(n.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);f(e.device,n,`${this.id} ${n.buffer.id}`)}let n=m(e.device,this.width,this.height,this.id);if((8+this.horizontalRadius*2)*(8+this.verticalRadius*2)*Float32Array.BYTES_PER_ELEMENT*2>e.device.limits.maxComputeWorkgroupStorageSize)throw Error(`${this.id} neighborhood exceeds the device workgroup storage limit`);if(t.length>e.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${this.id} neighborhood exceeds the device storage binding count`);let r=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.input.validity&&r.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:r,compile:({device:e})=>{let t=[{name:`sourceValues`,type:`read-only-storage`,group:0,location:0},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`outputValidity`,type:`storage`,group:0,location:2}];this.input.validity&&t.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let r=new a(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:t}}),i={sourceValues:this.input.storage.values,outputValues:this.output,outputValidity:this.outputValidity};this.input.validity&&(i.sourceValidity=this.input.validity);let s=o(i);return{encode:({computePass:e,getBuffer:t})=>{let i=s(t);i&&r.setBindings(i),r.dispatch(e,n[0],n[1])},destroy:()=>r.destroy()}}})}getShaderSource(){let e=8+this.horizontalRadius*2,t=e*(8+this.verticalRadius*2),n=this.input.validity?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${r(this.input.validity)}u;`:``,i=[`isFiniteValue(sample)`];this.input.format===`float32`&&i.push(`isFiniteValue(rawSample)`),this.input.validity&&i.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + sourceIndex] != 0u`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&i.push(`rawSample != ${h(this.input.noDataValue,this.input.format)}`);let a=this.kernel.map(u).join(`, `),o=y(this.borderMode,this.borderValue),l=this.noDataPolicy===`propagate`?`neighborhoodIsValid = false;`:``,d=this.normalize?`weightedSum / participatingWeight`:this.noDataPolicy===`ignore-renormalize`?`weightedSum * ${u(this.kernelSum)} / participatingWeight`:`weightedSum`;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const RADIUS_X: i32 = ${this.horizontalRadius}i;
const RADIUS_Y: i32 = ${this.verticalRadius}i;
const TILE_WIDTH: u32 = ${e}u;
const TILE_PIXEL_COUNT: u32 = ${t}u;
const SOURCE_OFFSET: u32 = ${r(this.input.storage.values)}u;
const OUTPUT_OFFSET: u32 = ${r(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${r(this.outputValidity)}u;
const KERNEL: array<f32, ${this.kernel.length}> = array<f32, ${this.kernel.length}>(${a});
@group(0) @binding(0) var<storage, read> sourceValues: array<${c(this.input.format)}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;
${n}
var<workgroup> neighborhoodValues: array<f32, ${t}>;
var<workgroup> neighborhoodValidity: array<u32, ${t}>;

${s}

fn reflectCoordinate(value: i32, length: i32) -> i32 {
  if (length <= 1i) { return 0i; }
  let period = (length - 1i) * 2i;
  let reflected = ((value % period) + period) % period;
  return select(reflected, period - reflected, reflected >= length);
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localIndex = localId.y * 8u + localId.x;
  let workgroupOrigin = vec2<i32>(
    i32(workgroupId.x * 8u),
    i32(workgroupId.y * 8u)
  );

  for (
    var tileIndex = localIndex;
    tileIndex < TILE_PIXEL_COUNT;
    tileIndex += 64u
  ) {
    var sourceColumn = workgroupOrigin.x + i32(tileIndex % TILE_WIDTH) - RADIUS_X;
    var sourceRow = workgroupOrigin.y + i32(tileIndex / TILE_WIDTH) - RADIUS_Y;
    var sample = ${u(this.borderValue)};
    var validSample = false;
    var sampleSource = true;
    if (
      sourceColumn < 0i || sourceColumn >= i32(WIDTH) ||
      sourceRow < 0i || sourceRow >= i32(HEIGHT)
    ) {
      ${o}
    }
    if (sampleSource) {
      let sourceIndex = u32(sourceRow) * WIDTH + u32(sourceColumn);
      let rawSample = sourceValues[SOURCE_OFFSET + sourceIndex];
      sample = f32(rawSample) * ${u(this.input.scale??1)} + ${u(this.input.offset??0)};
      validSample = ${i.join(` && `)};
    }
    neighborhoodValues[tileIndex] = sample;
    neighborhoodValidity[tileIndex] = select(0u, 1u, validSample);
  }
  workgroupBarrier();

  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let centerIndex = (localId.y + u32(RADIUS_Y)) * TILE_WIDTH + localId.x + u32(RADIUS_X);
  var neighborhoodIsValid = neighborhoodValidity[centerIndex] != 0u;
  var weightedSum = 0.0;
  var participatingWeight = 0.0;
  for (var kernelRow = 0u; kernelRow < ${this.verticalRadius*2+1}u; kernelRow++) {
    for (var kernelColumn = 0u; kernelColumn < ${this.horizontalRadius*2+1}u; kernelColumn++) {
      let kernelIndex = kernelRow * ${this.horizontalRadius*2+1}u + kernelColumn;
      let coefficient = KERNEL[kernelIndex];
      if (coefficient != 0.0) {
        let tileIndex = (localId.y + kernelRow) * TILE_WIDTH + localId.x + kernelColumn;
        if (neighborhoodValidity[tileIndex] != 0u) {
          weightedSum += neighborhoodValues[tileIndex] * coefficient;
          participatingWeight += coefficient;
        } else {
          ${l}
        }
      }
    }
  }
  ${this.normalize||this.noDataPolicy===`ignore-renormalize`?`neighborhoodIsValid = neighborhoodIsValid && participatingWeight != 0.0;`:``}
  let result = ${d};
  neighborhoodIsValid = neighborhoodIsValid && isFiniteValue(result);
  let invalidValue = bitcast<f32>(0x7fc00000u | (pixelIndex & 0u));
  outputValues[OUTPUT_OFFSET + pixelIndex] = select(invalidValue, result, neighborhoodIsValid);
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, neighborhoodIsValid);
}`}};function y(e,t){switch(e){case`clamp`:return`sourceColumn = clamp(sourceColumn, 0i, i32(WIDTH) - 1i);
      sourceRow = clamp(sourceRow, 0i, i32(HEIGHT) - 1i);`;case`reflect`:return`sourceColumn = reflectCoordinate(sourceColumn, i32(WIDTH));
      sourceRow = reflectCoordinate(sourceRow, i32(HEIGHT));`;case`constant`:return`sample = ${u(t)};\n      validSample = true;\n      sampleSource = false;`;case`nodata`:return`sampleSource = false;`}}var b=class{id;input;output;outputValidity;applyCalibration;width;height;constructor(e){if(this.id=e.id??`gpu-raster-texture-to-buffer`,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.applyCalibration=e.applyCalibration??!1,this.input.storage.kind!==`texture`)throw Error(`${this.id} requires a texture-backed input band`);this.width=this.input.storage.view.width,this.height=this.input.storage.view.height;let t=this.width*this.height,n=g(this.input,this,this.id),r=this.applyCalibration?`float32`:this.input.format;if(p(this.output,r,t,`${this.id} output`),l(this.outputValidity,t,`${this.id} output validity`),this.output.buffer.graph!==n||this.outputValidity.buffer.graph!==n)throw Error(`${this.id} resources must belong to the same graph`);if(this.output.buffer===this.outputValidity.buffer||this.input.validity?.buffer===this.output.buffer||this.input.validity?.buffer===this.outputValidity.buffer)throw Error(`${this.id} source validity and outputs must use separate buffers`);this.applyCalibration&&(u(this.input.scale??1),u(this.input.offset??0))}addToGraph(e){if(this.input.storage.view.texture.graph!==e||this.output.buffer.graph!==e||this.outputValidity.buffer.graph!==e||this.input.validity&&this.input.validity.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);f(e.device,this.output,`${this.id} output`),f(e.device,this.outputValidity,`${this.id} output validity`),this.input.validity&&f(e.device,this.input.validity,`${this.id} input validity`);let[t,n]=m(e.device,this.width,this.height,this.id),r=[{texture:this.input.storage.view,usage:`sampled`},{buffer:this.output,usage:`storage-write`},{buffer:this.outputValidity,usage:`storage-write`}];this.input.validity&&r.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:this.id,resources:r,compile:({device:e})=>{let r=[{name:`sourceTexture`,type:`texture`,group:0,location:0,sampleType:d(this.input.format)},{name:`outputValues`,type:`storage`,group:0,location:1},{name:`outputValidity`,type:`storage`,group:0,location:2}];this.input.validity&&r.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let i=new a(e,{id:this.id,source:this.getShaderSource(),shaderLayout:{bindings:r}}),s={outputValues:this.output,outputValidity:this.outputValidity};this.input.validity&&(s.sourceValidity=this.input.validity);let c=o(s,{sourceTexture:this.input.storage.view});return{encode:({computePass:e,getBuffer:r,getTextureView:a})=>{let o=c(r,a);o&&i.setBindings(o),i.dispatch(e,t,n)},destroy:()=>i.destroy()}}})}getShaderSource(){let e=c(this.input.format),t=c(this.output.format),n=[`x`,`y`,`z`,`w`][this.input.storage.channel??0],i=this.input.validity,a=i?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${r(i)}u;`:``,o=[];i&&o.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u`),this.input.format===`float32`&&o.push(`isFiniteValue(rawSample)`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&o.push(`rawSample != ${h(this.input.noDataValue,this.input.format)}`);let l=this.applyCalibration?`f32(rawSample) * ${u(this.input.scale??1)} + ${u(this.input.offset??0)}`:`rawSample`;this.applyCalibration&&o.push(`isFiniteValue(outputSample)`);let d=o.join(` && `)||`true`,f=this.applyCalibration?`let outputSample = ${l};`:``,p=this.output.format===`float32`?`bitcast<f32>(0x7fc00000u | (pixelIndex & 0u))`:this.output.format===`uint32`?`0u`:`0i`;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const OUTPUT_OFFSET: u32 = ${r(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${r(this.outputValidity)}u;
@group(0) @binding(0) var sourceTexture: texture_2d<${e}>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<${t}>;
@group(0) @binding(2) var<storage, read_write> outputValidity: array<u32>;
${a}

${s}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let rawSample = textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).${n};
  ${f}
  let validSample = ${d};
  outputValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, validSample);
  if (validSample) {
    outputValues[OUTPUT_OFFSET + pixelIndex] = ${this.applyCalibration?`outputSample`:`rawSample`};
  } else {
    outputValues[OUTPUT_OFFSET + pixelIndex] = ${p};
  }
}`}},x=s;function S(e,t){let{width:n,height:r,affine:i}=t;if(!Number.isSafeInteger(n)||!Number.isSafeInteger(r)||n<1||r<1||n*r>4294967295)throw Error(`${e} grid dimensions must be positive integers`);if(i.length!==6||!i.every(Number.isFinite))throw Error(`${e} grid affine must be six finite values`);return n*r}function C(e,t,n){if((n??`uniform`)!==`uniform`&&(t.affine[1]!==0||t.affine[3]!==0))throw Error(`${e} ${n} cell sizes need an axis-aligned grid`)}function w(t,n,r,i=`parameters`){if(e(n,[`float32`],`${t} ${i}`),n.length<r)throw Error(`${t} ${i} must contain at least ${r} values`)}function T(e,t,n){if(t.length<n)throw Error(`${e} target must hold ${n} values`)}function E(e,t,n){let r=t.filter(e=>e!==void 0).map(e=>e.buffer),i=n.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(r).size!==r.length||r.some(e=>i.includes(e)))throw Error(`${e} outputs must not share buffers with each other or with inputs`)}function D(e){return[...e.storage.kind===`buffer`?[e.storage.values]:[],...e.validity?[e.validity]:[]]}function O(e,n,r,i){t(e,n,D(r));let a=[...i,r.storage.kind===`texture`?r.storage.view:void 0];for(let t of a)if(t&&t.texture.graph!==n)throw Error(`${e} views must belong to the target graph`)}function k(e,t,n,r,i){if(n&&(!r.includes(n.format)||n.width!==i.width||n.height!==i.height))throw Error(`${e} ${t} must be ${r.join(` or `)} with the grid extent`)}function A(e,t,r,a,o){let{width:s,height:c}=a;if(r.storage.kind===`buffer`&&!o||r.storage.kind===`buffer`&&o===`reuse-canonical`&&r.format===`float32`&&r.validity&&r.scale===void 0&&r.offset===void 0&&r.noDataValue===void 0)return{band:r,nodes:[]};let l=s*c,u=i(e,`${t}-elevation-values`,`float32`,l),d=i(e,`${t}-elevation-validity`,`uint32`,l),f=r.storage.kind===`texture`?n(e,()=>new b({id:`${t}-elevation`,input:r,output:u,outputValidity:d,applyCalibration:!0}).addToGraph(e)):n(e,()=>new v({id:`${t}-elevation`,width:s,height:c,input:r,output:u,outputValidity:d,radius:0,kernel:[1],borderMode:`clamp`,noDataPolicy:`propagate`,normalize:!1}).addToGraph(e));return{band:{id:`${t}-elevation`,format:`float32`,storage:{kind:`buffer`,values:u},validity:d},nodes:f}}export{O as a,C as c,v as d,T as i,k as l,D as n,E as o,A as r,S as s,x as t,w as u};