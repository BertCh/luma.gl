import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as e}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,d as r,i,l as a,n as o,t as s,u as c,v as l,w as u,y as d}from"./resources-CwvhEOBe.js";import{l as f,s as p}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as m}from"./raster-utils-DhdnkkOa.js";import{a as h,c as g,i as _,n as v,o as y,r as b,s as x,t as S,u as C}from"./raster-band-utils-CA-sh8xt.js";import{a as w,c as T,l as E,s as D}from"./raster-grid-D2bmCwkc.js";import{U as O,m as k,y as A}from"./index-Bjk18V2I.js";import{t as j}from"./graph-buffers-B4_80RlX.js";import{t as M}from"./graph-swap-41Ru7LVu.js";import{t as N}from"./summary-reader-CsgWxENf.js";import{t as P}from"./mode-kernels-Dxm2GWvI.js";import"./anchors-D9E9ucGL.js";import{h as F}from"./gazetteer-OhisynFF.js";import{a as I,o as L}from"./class-table-BVEo3-Vs.js";import"./hue-registry-4FjIJ9xz.js";import{t as ee}from"./gpu-graph-connected-components-aw-M009a.js";import{r as te,t as ne}from"./network-topology-BXzNuj-N.js";import"./picking-DRmnefrA.js";import{n as R,t as z}from"./terrain-geomorphometry-utils-BlzhWTj6.js";import{n as B,t as V}from"./gpu-terrain-derivatives-BZk8xB3m.js";import{a as H,i as U,t as re}from"./b14a-session-CMNZQd9U.js";import{n as ie,t as ae}from"./cpu-dem-DnM3p94o.js";import{a as oe,c as se,d as ce,h as le,l as W,m as ue,x as de}from"./terrain-furniture-C6hPKNNK.js";import{a as fe,c as pe,i as me}from"./terrain-places-CXVUagwV.js";import{n as he,t as ge}from"./terrain-ground-BLUqElOt.js";import{_ as _e,a as ve,c as ye,d as be,f as xe,g as Se,i as Ce,l as we,m as Te,n as Ee,p as De,r as Oe,s as ke,v as Ae,y as je}from"./terrain-basics-labels-CHiq64H6.js";var Me=[-11e3,9e3],G=-12e3,K=`bitcast<f32>(0x7fc00000u | (pixelIndex & 0u))`,q=class{id;props;validRange;constructor(e){this.id=e.id??`terrain-rgb-decode`,this.props=e;let{id:t}=this,{output:n}=e,r=x(t,e.grid);if(e.encoding!==`terrarium`&&e.encoding!==`mapbox`)throw Error(`${t} encoding must be 'terrarium' or 'mapbox'`);let{texture:i,buffer:a}=e.input;if(!!i==!!a)throw Error(`${t} input requires exactly one of texture or buffer`);if(i){if(i.format!==`rgba8unorm`)throw Error(`${t} input texture must be rgba8unorm`);if(i.dimension!==`2d`||i.mipLevelCount!==1||i.arrayLayerCount!==1||i.width!==e.grid.width||i.height!==e.grid.height)throw Error(`${t} input texture must be one 2D mip with the grid extent`);if(e.validity)throw Error(`${t} validity requires a buffer input`)}if(a&&(d(a,[`uint32`],`${t} input buffer`),a.length!==r))throw Error(`${t} input buffer must contain one packed RGBA8 word per pixel`);if(e.validity&&(d(e.validity,[`uint32`],`${t} validity`),e.validity.length!==r))throw Error(`${t} validity must contain one value per pixel`);if(d(n.values,[`float32`],`${t} output.values`),d(n.outputValidity,[`uint32`],`${t} output.outputValidity`),n.values.length!==r||n.outputValidity.length!==r)throw Error(`${t} outputs must contain one value per pixel`);if(e.noDataRGB&&(e.noDataRGB.length!==3||!e.noDataRGB.every(e=>Number.isInteger(e)&&e>=0&&e<=255)))throw Error(`${t} noDataRGB must be three integer bytes`);this.validRange=e.validRange??Me;let[o,s]=this.validRange;if(Number.isNaN(o)||Number.isNaN(s)||o>s)throw Error(`${t} validRange must be [minimum, maximum] with minimum <= maximum`);y(t,[n.values,n.outputValidity],[a,e.validity])}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r;if(n(t,e,[r.input.buffer,r.validity,i.values,i.outputValidity]),r.input.texture){if(r.input.texture.texture.graph!==e)throw Error(`${t} views must belong to the target graph`);return[this.getTextureNode(e,r.input.texture)]}let a=[{name:`inputWords`,view:r.input.buffer,type:`u32`,access:`read`},...r.validity?[{name:`inputValidity`,view:r.validity,type:`u32`,access:`read`}]:[],{name:`outputValues`,view:i.values,type:`f32`,access:`read_write`},{name:`outputValidity`,view:i.outputValidity,type:`u32`,access:`read_write`}];return[p(e,{id:`${t}-decode`,operation:`GPUTerrainRGBDecode`,variant:`${r.encoding}-buffer`,bindings:a,invocationCount:r.grid.width*r.grid.height,declarations:this.getDecodeDeclarations(),body:`let pixelIndex = index;
  let word = inputWords[inputWordsOffset + pixelIndex];
  let red = word & 0xffu;
  let green = (word >> 8u) & 0xffu;
  let blue = (word >> 16u) & 0xffu;
  let alpha = word >> 24u;
  var isValid = isEncodedPixelValid(red, green, blue, alpha);
  ${r.validity?`isValid = isValid && inputValidity[inputValidityOffset + pixelIndex] != 0u;`:``}
  ${this.getDecodeStatements()}`})]}getDecodeDeclarations(){let{props:e,validRange:t}=this,[n,r]=t,i=e.noDataRGB,a=[Number.isFinite(n)?`height >= ${f(n)}`:``,Number.isFinite(r)?`height <= ${f(r)}`:``].filter(Boolean),o=e.encoding===`terrarium`?`f32(code) * 0.00390625 - 32768.0`:`f32(i32(code) - 100000) * ${f(.1)}`;return`const WIDTH: u32 = ${e.grid.width}u;
const HEIGHT: u32 = ${e.grid.height}u;
fn isEncodedPixelValid(red: u32, green: u32, blue: u32, alpha: u32) -> bool {
  ${e.alphaNoData===!1?``:`if (alpha == 0u) { return false; }`}
  ${i?`if (red == ${i[0]}u && green == ${i[1]}u && blue == ${i[2]}u) { return false; }`:``}
  return true;
}
// Integer code N = R * 65536 + G * 256 + B < 2^24, then exactly one float operation sequence.
fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {
  let code = (red << 16u) | (green << 8u) | blue;
  return ${o};
}
fn isHeightInRange(height: f32) -> bool {
  return ${a.join(` && `)||`true`};
}
fn applySeaClamp(height: f32) -> f32 {
  ${e.clampBathymetry?`if (height < 0.0 && height > ${f(G)}) { return 0.0; }`:``}
  return height;
}`}getDecodeStatements(){return`let height = decodeHeight(red, green, blue);
  isValid = isValid && isHeightInRange(height);
  outputValidity[outputValidityOffset + pixelIndex] = select(0u, 1u, isValid);
  outputValues[outputValuesOffset + pixelIndex] =
    select(${K}, applySeaClamp(height), isValid);`}getTextureNode(e,t){let{id:n,props:r}=this,{output:i}=r;return m(e.device,i.values,`${n} output.values`),m(e.device,i.outputValidity,`${n} output.outputValidity`),p(e,{id:`${n}-decode`,operation:`GPUTerrainRGBDecode`,variant:`${r.encoding}-texture`,bindings:[{name:`outputValues`,view:i.values,type:`f32`,access:`read_write`},{name:`outputValidity`,view:i.outputValidity,type:`u32`,access:`read_write`}],textures:[{name:`sourceTexture`,texture:t,sampleType:`float`}],invocationCount:r.grid.width*r.grid.height,declarations:this.getDecodeDeclarations(),body:`let pixelIndex = index;
  let texelCoordinate = vec2<i32>(i32(pixelIndex % WIDTH), i32(pixelIndex / WIDTH));
  // Exact for unorm8: the load is within half a code of k / 255.
  let texel = clamp(textureLoad(sourceTexture, texelCoordinate, 0), vec4<f32>(0.0), vec4<f32>(1.0));
  let bytes = vec4<u32>(round(texel * 255.0));
  let red = bytes.x;
  let green = bytes.y;
  let blue = bytes.z;
  var isValid = isEncodedPixelValid(red, green, blue, bytes.w);
  ${this.getDecodeStatements()}`})}},Ne={jumpCount:0,repairedPixelCount:1,shiftedComponentCount:2,remainingJumpCount:3,converged:4},J=4294967295;function Pe(e){return a(e,32)}var Fe=127,Y=16,X=8,Ie=0,Le=1,Re=2,Z=4,ze=`0xffffffffu`;function Be(e,t){let n=Math.round((e??t)*100);return Math.abs((e??t)*100-n)<1e-9?n:void 0}var Ve=class{id;props;maximumComponentIterations;agreementPercent;maximumComponentPercent;ownedBuffers;constructor(e){this.id=e.id??`terrain-spike-repair`,this.props=e;let{id:t}=this,{output:n}=e,i=x(t,e.grid),a=e.step??256,o=e.jump??200,s=e.tolerance??40;if(!Number.isFinite(a)||a<=0)throw Error(`${t} step must be finite and positive`);if(!Number.isFinite(o)||o<=0)throw Error(`${t} jump must be finite and positive`);if(!Number.isFinite(s)||s<=0||s>=a/2)throw Error(`${t} tolerance must be positive and less than step / 2`);let c=Be(e.agreement,.8);if(c===void 0||c<=50||c>100)throw Error(`${t} agreement must be a multiple of 0.01 in (0.5, 1]`);let l=Be(e.maximumComponentFraction,.25);if(l===void 0||l<1||l>100)throw Error(`${t} maximumComponentFraction must be a multiple of 0.01 in (0, 1]`);this.agreementPercent=c,this.maximumComponentPercent=l,this.maximumComponentIterations=e.maximumComponentIterations??32,r(t,`maximumComponentIterations`,this.maximumComponentIterations,1,1024);for(let[e,r,a]of[[`values`,n.values,`float32`],[`outputValidity`,n.outputValidity,`uint32`],[`labels`,n.labels,`uint32`]])if(r&&(d(r,[a],`${t} output.${e}`),r.length!==i))throw Error(`${t} output.${e} must contain one value per pixel`);if(n.statistics&&(d(n.statistics,[`uint32`],`${t} output.statistics`),n.statistics.length<5))throw Error(`${t} output.statistics must contain at least 5 rows`);y(t,[n.values,n.outputValidity,n.labels,n.statistics],v(e.elevation))}getCommandNodes(t){let{id:r,props:a}=this,{output:o}=a,{width:s,height:d}=a.grid,m=s*d;h(r,t,a.elevation,[]),n(r,t,[o.values,o.outputValidity,o.labels,o.statistics]),l(r,`grid cells`,m,Pe(t.device));let g=X*m,_=`GPUTerrainSpikeRepair`,v=a.step??256,y=a.jump??200,x=a.tolerance??40;if(!this.ownedBuffers){let n=e.STORAGE|e.COPY_SRC|e.COPY_DST,i=(e,i)=>t.device.createBuffer({id:`${r}-${e}`,byteLength:Math.max(i,1)*4,usage:n});this.ownedBuffers={offsets:i(`offsets`,m+1),neighbors:i(`neighbors`,2*m),labels:i(`labels`,m),converged:i(`converged`,1),status:i(`status`,4)}}let C=this.ownedBuffers,w=c(t,`${r}-offsets`,C.offsets,`uint32`,m+1),T=c(t,`${r}-neighbors`,C.neighbors,`uint32`,2*m),E=c(t,`${r}-labels`,C.labels,`uint32`,m),D=c(t,`${r}-converged`,C.converged,`uint32`,1),O=b(t,r,a.elevation,a.grid,!0),k=[...O.nodes],A=O.band.storage.values,j=O.band.validity,M=(e,n)=>u(t,`${r}-${e}`,`uint32`,n),N=M(`component-size`,m),P=M(`seam-count`,m),F=M(`vote-count`,m),I=M(`agree-count`,m),L=M(`bit-count`,g),R=M(`scalars`,Z),z=M(`shift-key`,m),B=o.statistics??M(`statistics`,5),V=Ne,H=`const WIDTH: u32 = ${s}u;
const HEIGHT: u32 = ${d}u;
const PIXEL_COUNT: u32 = ${m}u;
const NO_VERTEX: u32 = ${ze};
const JUMP: f32 = ${f(y)};
const STEP: f32 = ${f(v)};
const INVERSE_STEP: f32 = ${f(1/v)};
const TOLERANCE: f32 = ${f(x)};
const BIT_COUNT: u32 = ${X}u;
${S}`,U=`
fn isValidPixel(pixel: u32) -> bool {
  return validity[validityOffset + pixel] != 0u && isFiniteValue(values[valuesOffset + pixel]);
}
fn getNeighbor(pixel: u32, direction: u32) -> u32 {
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  if (direction == 0u) { return select(NO_VERTEX, pixel + 1u, column + 1u < WIDTH); }
  if (direction == 1u) { return select(NO_VERTEX, pixel - 1u, column > 0u); }
  if (direction == 2u) { return select(NO_VERTEX, pixel + WIDTH, row + 1u < HEIGHT); }
  return select(NO_VERTEX, pixel - WIDTH, row > 0u);
}`,re=`
// Biased step multiple k + 128 in 1..255 when the height difference votes, else 0.
fn getVoteKey(difference: f32) -> u32 {
  let multiple = round(difference * INVERSE_STEP);
  if (multiple == 0.0 || abs(multiple) > ${Fe}.0) { return 0u; }
  if (!(abs(difference - multiple * STEP) < TOLERANCE)) { return 0u; }
  return u32(i32(multiple) + 128);
}`;k.push(p(t,{id:`${r}-clear`,operation:_,variant:`clear`,bindings:[{name:`componentSize`,view:N,type:`u32`,access:`read_write`},{name:`seamCount`,view:P,type:`u32`,access:`read_write`},{name:`voteCount`,view:F,type:`u32`,access:`read_write`},{name:`agreeCount`,view:I,type:`u32`,access:`read_write`},{name:`bitCount`,view:L,type:`u32`,access:`read_write`},{name:`scalars`,view:R,type:`u32`,access:`read_write`},{name:`statistics`,view:B,type:`u32`,access:`read_write`}],invocationCount:g,declarations:`const PIXEL_COUNT: u32 = ${m}u;`,body:`bitCount[bitCountOffset + index] = 0u;
  if (index < PIXEL_COUNT) {
    componentSize[componentSizeOffset + index] = 0u;
    seamCount[seamCountOffset + index] = 0u;
    voteCount[voteCountOffset + index] = 0u;
    agreeCount[agreeCountOffset + index] = 0u;
  }
  if (index < ${Z}u) {
    scalars[scalarsOffset + index] = select(0u, 0xffffffffu, index == ${Le}u);
  }
  if (index < 5u) {
    statistics[statisticsOffset + index] = 0u;
  }`})),k.push(p(t,{id:`${r}-graph`,operation:_,variant:`graph`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`offsets`,view:w,type:`u32`,access:`read_write`},{name:`neighbors`,view:T,type:`u32`,access:`read_write`}],invocationCount:m+1,declarations:`${H}${U}
fn getEdge(pixel: u32, direction: u32) -> u32 {
  let neighbor = getNeighbor(pixel, direction);
  if (neighbor == NO_VERTEX || !isValidPixel(pixel) || !isValidPixel(neighbor)) {
    return NO_VERTEX;
  }
  let difference = values[valuesOffset + pixel] - values[valuesOffset + neighbor];
  return select(NO_VERTEX, neighbor, abs(difference) <= JUMP);
}`,body:`offsets[offsetsOffset + index] = 2u * index;
  if (index < PIXEL_COUNT) {
    neighbors[neighborsOffset + 2u * index] = getEdge(index, 0u);
    neighbors[neighborsOffset + 2u * index + 1u] = getEdge(index, 2u);
  }`})),k.push(...i(t,()=>{let e=ne({id:r,nodeCount:m,offsets:w,neighbors:T,status:C.status});new ee({id:`${r}-components`,topology:e,output:te(r,`component labels`,E),iterations:this.maximumComponentIterations,converged:te(r,`component converged`,D)}).addToGraph(t)})),k.push(p(t,{id:`${r}-sizes`,operation:_,variant:`sizes`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`labels`,view:E,type:`u32`,access:`read`},{name:`componentSize`,view:N,type:`atomic<u32>`,access:`read_write`},{name:`statistics`,view:B,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(m/Y),declarations:`${H}${U}
const RUN_LENGTH: u32 = ${Y}u;`,body:`var runLabel = NO_VERTEX;
  var runLength = 0u;
  var jumps = 0u;
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label < PIXEL_COUNT) {
      if (label == runLabel) {
        runLength++;
      } else {
        if (runLength > 0u) {
          atomicAdd(&componentSize[componentSizeOffset + runLabel], runLength);
        }
        runLabel = label;
        runLength = 1u;
      }
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction += 2u) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
          abs(value - values[valuesOffset + neighbor]) > JUMP) {
        jumps++;
      }
    }
  }
  if (runLength > 0u) {
    atomicAdd(&componentSize[componentSizeOffset + runLabel], runLength);
  }
  if (jumps > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${V.jumpCount}u], jumps);
  }`})),k.push(p(t,{id:`${r}-main-size`,operation:_,variant:`main-size`,bindings:[{name:`componentSize`,view:N,type:`u32`,access:`read`},{name:`scalars`,view:R,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,body:`let size = componentSize[componentSizeOffset + index];
  if (size > 0u) {
    atomicMax(&scalars[scalarsOffset + ${Ie}u], size);
    // Component sizes partition the valid pixels, so their sum is the valid pixel count: one
    // atomic per component instead of one per pixel on a single address.
    atomicAdd(&scalars[scalarsOffset + ${Re}u], size);
  }`}),p(t,{id:`${r}-main-label`,operation:_,variant:`main-label`,bindings:[{name:`componentSize`,view:N,type:`u32`,access:`read`},{name:`scalars`,view:R,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,body:`let size = componentSize[componentSizeOffset + index];
  if (size > 0u && size == atomicLoad(&scalars[scalarsOffset + ${Ie}u])) {
    atomicMin(&scalars[scalarsOffset + ${Le}u], index);
  }`})),k.push(p(t,{id:`${r}-votes`,operation:_,variant:`votes`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`labels`,view:E,type:`u32`,access:`read`},{name:`seamCount`,view:P,type:`atomic<u32>`,access:`read_write`},{name:`voteCount`,view:F,type:`atomic<u32>`,access:`read_write`},{name:`bitCount`,view:L,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(m/Y),declarations:`${H}${U}${re}
const RUN_LENGTH: u32 = ${Y}u;
var<private> runBits: array<u32, ${X}>;
var<private> runSeams: u32;
var<private> runVotes: u32;
// Adds the counters of the finished label run, one atomic per non-zero counter.
fn flushRun(label: u32) {
  if (runSeams > 0u) {
    atomicAdd(&seamCount[seamCountOffset + label], runSeams);
  }
  if (runVotes > 0u) {
    atomicAdd(&voteCount[voteCountOffset + label], runVotes);
    for (var bit = 0u; bit < BIT_COUNT; bit++) {
      if (runBits[bit] > 0u) {
        atomicAdd(&bitCount[bitCountOffset + label * BIT_COUNT + bit], runBits[bit]);
      }
    }
  }
  runSeams = 0u;
  runVotes = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    runBits[bit] = 0u;
  }
}`,body:`var runLabel = NO_VERTEX;
  runSeams = 0u;
  runVotes = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    runBits[bit] = 0u;
  }
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label >= PIXEL_COUNT) {
      continue;
    }
    if (label != runLabel) {
      if (runLabel != NO_VERTEX) {
        flushRun(runLabel);
      }
      runLabel = label;
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction++) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
          labels[labelsOffset + neighbor] == label) {
        continue;
      }
      runSeams++;
      let key = getVoteKey(value - values[valuesOffset + neighbor]);
      if (key == 0u) {
        continue;
      }
      runVotes++;
      for (var bit = 0u; bit < BIT_COUNT; bit++) {
        runBits[bit] += (key >> bit) & 1u;
      }
    }
  }
  if (runLabel != NO_VERTEX) {
    flushRun(runLabel);
  }`}));let ie=`
// Majority-bit candidate key of a component, 0 when no bit has a majority.
fn getCandidateKey(label: u32) -> u32 {
  let votes = voteCount[voteCountOffset + label];
  var key = 0u;
  for (var bit = 0u; bit < BIT_COUNT; bit++) {
    if (bitCount[bitCountOffset + label * BIT_COUNT + bit] * 2u > votes) {
      key |= 1u << bit;
    }
  }
  return key;
}`;return k.push(p(t,{id:`${r}-agree`,operation:_,variant:`agree`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`labels`,view:E,type:`u32`,access:`read`},{name:`voteCount`,view:F,type:`u32`,access:`read`},{name:`bitCount`,view:L,type:`u32`,access:`read`},{name:`agreeCount`,view:I,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(m/Y),declarations:`${H}${U}${re}${ie}
const RUN_LENGTH: u32 = ${Y}u;`,body:`var runLabel = NO_VERTEX;
  var candidate = 0u;
  var runAgree = 0u;
  let first = index * RUN_LENGTH;
  for (var offset = 0u; offset < RUN_LENGTH; offset++) {
    let pixel = first + offset;
    if (pixel >= PIXEL_COUNT) {
      break;
    }
    if (!isValidPixel(pixel)) {
      continue;
    }
    let label = labels[labelsOffset + pixel];
    if (label >= PIXEL_COUNT) {
      continue;
    }
    if (label != runLabel) {
      if (runAgree > 0u) {
        atomicAdd(&agreeCount[agreeCountOffset + runLabel], runAgree);
      }
      runAgree = 0u;
      runLabel = label;
      candidate = getCandidateKey(label);
    }
    if (candidate == 0u) {
      continue;
    }
    let value = values[valuesOffset + pixel];
    for (var direction = 0u; direction < 4u; direction++) {
      let neighbor = getNeighbor(pixel, direction);
      if (neighbor == NO_VERTEX || !isValidPixel(neighbor) ||
          labels[labelsOffset + neighbor] == label) {
        continue;
      }
      if (getVoteKey(value - values[valuesOffset + neighbor]) == candidate) {
        runAgree++;
      }
    }
  }
  if (runAgree > 0u) {
    atomicAdd(&agreeCount[agreeCountOffset + runLabel], runAgree);
  }`})),k.push(p(t,{id:`${r}-decide`,operation:_,variant:`decide`,bindings:[{name:`componentSize`,view:N,type:`u32`,access:`read`},{name:`seamCount`,view:P,type:`u32`,access:`read`},{name:`voteCount`,view:F,type:`u32`,access:`read`},{name:`agreeCount`,view:I,type:`u32`,access:`read`},{name:`bitCount`,view:L,type:`u32`,access:`read`},{name:`scalars`,view:R,type:`u32`,access:`read`},{name:`converged`,view:D,type:`u32`,access:`read`},{name:`shiftKey`,view:z,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`const PIXEL_COUNT: u32 = ${m}u;
const BIT_COUNT: u32 = ${X}u;
const AGREEMENT_PERCENT: u32 = ${this.agreementPercent}u;
const MAXIMUM_COMPONENT_PERCENT: u32 = ${this.maximumComponentPercent}u;${ie}
// Exact 64-bit product of a u32 and a factor below 2^16, as (low, high) words: the percentage
// tests below would wrap in u32 for components or valid counts above about 42.9 million pixels.
fn getWideProduct(value: u32, factor: u32) -> vec2<u32> {
  let low = (value & 0xffffu) * factor;
  let high = (value >> 16u) * factor;
  let lowWord = low + (high << 16u);
  return vec2<u32>(lowWord, (high >> 16u) + select(0u, 1u, lowWord < low));
}
fn isWideLessOrEqual(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.y < right.y || (left.y == right.y && left.x <= right.x);
}`,body:`let size = componentSize[componentSizeOffset + index];
  let seams = seamCount[seamCountOffset + index];
  let key = getCandidateKey(index);
  let validCount = scalars[scalarsOffset + ${Re}u];
  let mainLabel = scalars[scalarsOffset + ${Le}u];
  let shift = converged[convergedOffset] != 0u && size > 0u && index != mainLabel &&
    isWideLessOrEqual(getWideProduct(size, 100u), getWideProduct(validCount, MAXIMUM_COMPONENT_PERCENT)) &&
    seams > 0u && key != 0u &&
    isWideLessOrEqual(getWideProduct(seams, AGREEMENT_PERCENT), getWideProduct(agreeCount[agreeCountOffset + index], 100u));
  shiftKey[shiftKeyOffset + index] = select(0u, key, shift);`})),k.push(p(t,{id:`${r}-shift`,operation:_,variant:`shift`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`labels`,view:E,type:`u32`,access:`read`},{name:`shiftKey`,view:z,type:`u32`,access:`read`},{name:`converged`,view:D,type:`u32`,access:`read`},{name:`valuesOut`,view:o.values,type:`f32`,access:`read_write`},{name:`validityOut`,view:o.outputValidity,type:`u32`,access:`read_write`},{name:`statistics`,view:B,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,declarations:`${H}${U}`,body:`if (index == 0u) {
    atomicStore(&statistics[statisticsOffset + ${V.converged}u],
      select(0u, 1u, converged[convergedOffset] != 0u));
  }
  if (!isValidPixel(index)) {
    valuesOut[valuesOutOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    validityOut[validityOutOffset + index] = 0u;
    return;
  }
  if (converged[convergedOffset] == 0u) {
    // Labelling stopped early: nothing is repaired and the pixel is marked, not copied.
    var infinityBits = 0x7f800000u;
    valuesOut[valuesOutOffset + index] = bitcast<f32>(infinityBits);
    validityOut[validityOutOffset + index] = ${J}u;
    return;
  }
  validityOut[validityOutOffset + index] = 1u;
  var value = values[valuesOffset + index];
  let label = labels[labelsOffset + index];
  var key = 0u;
  if (label < PIXEL_COUNT) {
    key = shiftKey[shiftKeyOffset + label];
  }
  if (key != 0u) {
    value = value - STEP * f32(i32(key) - 128);
    atomicAdd(&statistics[statisticsOffset + ${V.repairedPixelCount}u], 1u);
    if (label == index) {
      atomicAdd(&statistics[statisticsOffset + ${V.shiftedComponentCount}u], 1u);
    }
  }
  valuesOut[valuesOutOffset + index] = value;`})),o.labels&&k.push(p(t,{id:`${r}-publish-labels`,operation:_,variant:`labels`,bindings:[{name:`values`,view:A,type:`f32`,access:`read`},{name:`validity`,view:j,type:`u32`,access:`read`},{name:`labels`,view:E,type:`u32`,access:`read`},{name:`converged`,view:D,type:`u32`,access:`read`},{name:`labelsOut`,view:o.labels,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${H}${U}`,body:`let unresolved = converged[convergedOffset] == 0u && isValidPixel(index);
  labelsOut[labelsOutOffset + index] =
    select(labels[labelsOffset + index], ${J}u, unresolved);`})),k.push(p(t,{id:`${r}-recount`,operation:_,variant:`recount`,bindings:[{name:`values`,view:o.values,type:`f32`,access:`read`},{name:`validity`,view:o.outputValidity,type:`u32`,access:`read`},{name:`statistics`,view:B,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,declarations:`${H}${U}`,body:`if (!isValidPixel(index)) {
    return;
  }
  let value = values[valuesOffset + index];
  var count = 0u;
  for (var direction = 0u; direction < 4u; direction += 2u) {
    let neighbor = getNeighbor(index, direction);
    if (neighbor != NO_VERTEX && isValidPixel(neighbor) &&
        abs(value - values[valuesOffset + neighbor]) > JUMP) {
      count++;
    }
  }
  if (count > 0u) {
    atomicAdd(&statistics[statisticsOffset + ${V.remainingJumpCount}u], count);
  }`})),k.push(p(t,{id:`${r}-recount-unresolved`,operation:_,variant:`recount-unresolved`,bindings:[{name:`converged`,view:D,type:`u32`,access:`read`},{name:`statistics`,view:B,type:`u32`,access:`read_write`}],invocationCount:1,body:`if (converged[convergedOffset] == 0u) {
    statistics[statisticsOffset + ${V.remainingJumpCount}u] =
      statistics[statisticsOffset + ${V.jumpCount}u];
  }`})),k}destroy(){if(this.ownedBuffers){for(let e of Object.values(this.ownedBuffers))e.destroy();this.ownedBuffers=void 0}}},He=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-ruggedness`,this.props=e;let{id:t}=this,{output:n}=e,{width:r,height:i}=e.grid,a=x(t,e.grid);if(!n.topographicPositionIndex&&!n.terrainRuggednessIndex&&!n.roughness&&!n.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,r]of[[`topographicPositionIndex`,n.topographicPositionIndex],[`terrainRuggednessIndex`,n.terrainRuggednessIndex],[`roughness`,n.roughness]])if(r&&(d(r,[`float32`],`${t} output.${e}`),r.length!==a))throw Error(`${t} output.${e} must contain one value per pixel`);if(n.outputValidity&&(d(n.outputValidity,[`uint32`],`${t} output.outputValidity`),n.outputValidity.length!==a))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(![`riley`,`wilson`].includes(e.terrainRuggednessAlgorithm??`riley`))throw Error(`${t} terrainRuggednessAlgorithm must be riley or wilson`);let o=e.edgeMode??`nodata`;if(![`nodata`,`extrapolate`].includes(o))throw Error(`${t} edgeMode must be nodata or extrapolate`);if(o===`extrapolate`&&(r<2||i<2))throw Error(`${t} edgeMode extrapolate requires width and height of at least 2`);y(t,[n.topographicPositionIndex,n.terrainRuggednessIndex,n.roughness,n.outputValidity],v(e.elevation))}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r,{width:a,height:o}=r.grid;h(t,e,r.elevation,[]),n(t,e,[i.topographicPositionIndex,i.terrainRuggednessIndex,i.roughness,i.outputValidity]);let s=b(e,t,r.elevation,r.grid,!0),c=[{name:`elevation`,view:s.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:s.band.validity,type:`u32`,access:`read`}],l=[[`tpiValues`,i.topographicPositionIndex,`f32`],[`rugValues`,i.terrainRuggednessIndex,`f32`],[`roughnessValues`,i.roughness,`f32`],[`validityValues`,i.outputValidity,`u32`]];for(let[e,t,n]of l)t&&c.push({name:e,view:t,type:n,access:`read_write`});let u=r.edgeMode??`nodata`,d=r.terrainRuggednessAlgorithm??`riley`,f=p(e,{id:`${t}-window`,operation:`GPUTerrainRuggedness`,variant:`${u}-${d}`,bindings:c,invocationCount:a*o,declarations:Ue(a,o,u),body:We(r,d)});return[...s.nodes,f]}};function Ue(e,t,n){let r=`const WIDTH: i32 = ${e};
const HEIGHT: i32 = ${t};
${S}
fn isSampleValid(row: i32, column: i32) -> bool {
  let sampleIndex = u32(row * WIDTH + column);
  return elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
    isFiniteValue(elevation[elevationOffset + sampleIndex]);
}
fn getSample(row: i32, column: i32) -> f32 {
  return elevation[elevationOffset + u32(row * WIDTH + column)];
}`;return n===`nodata`?r:`${r}
struct WindowSample { value: f32, valid: bool };
fn getExtrapolated(
  rowA: i32, columnA: i32, rowB: i32, columnB: i32
) -> WindowSample {
  let valid = isSampleValid(rowA, columnA) && isSampleValid(rowB, columnB);
  return WindowSample(2.0 * getSample(rowA, columnA) - getSample(rowB, columnB), valid);
}
fn getWindowSample(row: i32, column: i32, deltaRow: i32, deltaColumn: i32) -> WindowSample {
  if (row == 0 || row == HEIGHT - 1) {
    let sampleColumn = clamp(column + deltaColumn, 0, WIDTH - 1);
    if (row == 0 && deltaRow < 0) {
      return getExtrapolated(0, sampleColumn, 1, sampleColumn);
    }
    if (row == HEIGHT - 1 && deltaRow > 0) {
      return getExtrapolated(HEIGHT - 1, sampleColumn, HEIGHT - 2, sampleColumn);
    }
    let sampleRow = row + deltaRow;
    return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
  }
  let sampleRow = row + deltaRow;
  if (column == 0 && deltaColumn < 0) {
    return getExtrapolated(sampleRow, 0, sampleRow, 1);
  }
  if (column == WIDTH - 1 && deltaColumn > 0) {
    return getExtrapolated(sampleRow, WIDTH - 1, sampleRow, WIDTH - 2);
  }
  let sampleColumn = column + deltaColumn;
  return WindowSample(getSample(sampleRow, sampleColumn), isSampleValid(sampleRow, sampleColumn));
}`}function We(e,t){return`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  var windowValues: array<f32, 9>;
  let centerValue = getSample(row, column);
  ${(e.edgeMode??`nodata`)===`nodata`?`var isValid = row > 0 && row < HEIGHT - 1 && column > 0 && column < WIDTH - 1;
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let sampleRow = row + k / 3 - 1;
      let sampleColumn = column + k % 3 - 1;
      isValid = isValid && isSampleValid(sampleRow, sampleColumn);
      if (isValid) {
        windowValues[k] = getSample(sampleRow, sampleColumn);
      }
    }
  }`:`var isValid = isSampleValid(row, column);
  if (isValid) {
    for (var k = 0; k < 9; k++) {
      let windowSample = getWindowSample(row, column, k / 3 - 1, k % 3 - 1);
      // gdaldem replaces invalid window samples by the centre when computing edges.
      windowValues[k] = select(centerValue, windowSample.value, windowSample.valid);
    }
  }`}
  var differenceSum = 0.0;
  var absoluteSum = 0.0;
  var squareSum = 0.0;
  var minimum = windowValues[0];
  var maximum = windowValues[0];
  for (var k = 0; k < 9; k++) {
    // Differences against the centre are formed first so they stay exact near large offsets.
    let difference = windowValues[k] - centerValue;
    differenceSum += difference;
    absoluteSum += abs(difference);
    squareSum += difference * difference;
    minimum = min(minimum, windowValues[k]);
    maximum = max(maximum, windowValues[k]);
  }
  let topographicPosition = -differenceSum / 8.0;
  ${t===`riley`?`let ruggedness = sqrt(squareSum);`:`let ruggedness = absoluteSum / 8.0;`}
  let roughness = maximum - minimum;
  isValid = isValid && isFiniteValue(topographicPosition) && isFiniteValue(ruggedness) &&
    isFiniteValue(roughness);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${e.output.topographicPositionIndex?`tpiValues[tpiValuesOffset + index] = select(invalidValue, topographicPosition, isValid);`:``}
  ${e.output.terrainRuggednessIndex?`rugValues[rugValuesOffset + index] = select(invalidValue, ruggedness, isValid);`:``}
  ${e.output.roughness?`roughnessValues[roughnessValuesOffset + index] = select(invalidValue, roughness, isValid);`:``}
  ${e.output.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`}var Ge=3,Ke=5;function qe(e,t,n=new Float32Array(8)){if(_(`getGPUTerrainVectorRuggednessParameterValues`,n,8),e.radius!==void 0&&(!Number.isSafeInteger(e.radius)||e.radius<1))throw Error(`getGPUTerrainVectorRuggednessParameterValues radius must be an integer of at least 1`);return n.fill(0,0,8),R(e,t,n),n[Ke]=e.radius??0,n}var Je=class{id;props;requiredHalo;constructor(e){this.id=e.id??`terrain-vector-ruggedness`,this.props=e;let{id:t}=this,{output:n}=e,r=e.maximumRadius??1;if(!Number.isSafeInteger(r)||r<1)throw Error(`${t} maximumRadius must be an integer of at least 1`);this.requiredHalo=r+1;let i=x(t,e.grid);if(!n.vectorRuggedness&&!n.outputValidity)throw Error(`${t} requires at least one output`);if(n.vectorRuggedness&&(d(n.vectorRuggedness,[`float32`],`${t} output.vectorRuggedness`),n.vectorRuggedness.length!==i))throw Error(`${t} output.vectorRuggedness must contain one value per pixel`);if(n.outputValidity&&(d(n.outputValidity,[`uint32`],`${t} output.outputValidity`),n.outputValidity.length!==i))throw Error(`${t} output.outputValidity must contain one value per pixel`);if(C(t,e.parameters,8),T(t,e.cellSizeMode),g(t,e.grid,e.cellSizeMode),E(t,e.rowDirection),![`clamp`,`nodata`].includes(e.borderMode??`clamp`))throw Error(`${t} borderMode must be clamp or nodata`);y(t,[n.vectorRuggedness,n.outputValidity],[...v(e.elevation),e.parameters])}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r,{width:a,height:o}=r.grid;h(t,e,r.elevation,[]),n(t,e,[r.parameters,i.vectorRuggedness,i.outputValidity]);let s=a*o,c=b(e,t,r.elevation,r.grid,!0),l=[`x`,`y`,`z`].map(n=>u(e,`${t}-normal-${n}`,`float32`,s)),d=p(e,{id:`${t}-normals`,operation:`GPUTerrainVectorRuggedness`,variant:`normals-${r.cellSizeMode??`uniform`}-${r.borderMode??`clamp`}`,bindings:[{name:`elevation`,view:c.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:c.band.validity,type:`u32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`normalX`,view:l[0],type:`f32`,access:`read_write`},{name:`normalY`,view:l[1],type:`f32`,access:`read_write`},{name:`normalZ`,view:l[2],type:`f32`,access:`read_write`}],invocationCount:s,declarations:`const WIDTH: u32 = ${a}u;
const HEIGHT: u32 = ${o}u;
${D(r.rowDirection??`south`)}
${S}
${w(r.cellSizeMode??`uniform`,z)}`,body:`let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  var isValid = true;
  var windowValues: array<f32, 9>;
  for (var k = 0; k < 9; k++) {
    var sampleRow = row + k / 3 - 1;
    var sampleColumn = column + k % 3 - 1;
    ${(r.borderMode??`clamp`)===`nodata`?`if (sampleRow < 0 || sampleRow >= i32(HEIGHT) || sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
      isValid = false;
      continue;
    }`:``}
    sampleRow = clamp(sampleRow, 0, i32(HEIGHT) - 1);
    sampleColumn = clamp(sampleColumn, 0, i32(WIDTH) - 1);
    let sampleIndex = u32(sampleRow) * WIDTH + u32(sampleColumn);
    let sampleValue = elevation[elevationOffset + sampleIndex];
    isValid = isValid && elevationValidity[elevationValidityOffset + sampleIndex] != 0u &&
      isFiniteValue(sampleValue);
    windowValues[k] = sampleValue;
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Horn: weights 1-2-1 across three rows or columns, divided by 8 cell sizes.
  let eastGradient = zFactor *
    ((windowValues[2] + 2.0 * windowValues[5] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[3] + windowValues[6])) / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor *
    ((windowValues[6] + 2.0 * windowValues[7] + windowValues[8]) -
     (windowValues[0] + 2.0 * windowValues[1] + windowValues[2])) / (8.0 * groundCell.y);
  isValid = isValid && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  normalX[normalXOffset + index] = select(invalidValue, normal.x, isValid);
  normalY[normalYOffset + index] = select(invalidValue, normal.y, isValid);
  normalZ[normalZOffset + index] = select(invalidValue, normal.z, isValid);`}),f=r.maximumRadius??1,m={name:`settings`,view:r.parameters,type:`f32`,access:`read`},g=[{name:`normalX`,view:l[0],type:`f32`,access:`read`},{name:`normalY`,view:l[1],type:`f32`,access:`read`},{name:`normalZ`,view:l[2],type:`f32`,access:`read`},m];i.vectorRuggedness&&g.push({name:`vrmValues`,view:i.vectorRuggedness,type:`f32`,access:`read_write`}),i.outputValidity&&g.push({name:`validityValues`,view:i.outputValidity,type:`u32`,access:`read_write`});let _=`let radiusWord = settings[settingsOffset + ${Ke}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);`,v=e=>`let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let isValid = isFiniteValue(normalX[normalXOffset + index]);
  var vectorSum = vec3<f32>(0.0);
  var count = 0.0;
  if (isValid) {
    ${e}
  }
  // count >= 1 whenever the centre normal is valid.
  let ruggedness = clamp(1.0 - length(vectorSum) / max(count, 1.0), 0.0, 1.0);
  ${i.vectorRuggedness?`vrmValues[vrmValuesOffset + index] = select(invalidValue, ruggedness, isValid);`:``}
  ${i.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`;if(f<Ge){let n=p(e,{id:`${t}-window`,operation:`GPUTerrainVectorRuggedness`,variant:`window`,bindings:g,invocationCount:s,declarations:`const WIDTH: i32 = ${a};
const HEIGHT: i32 = ${o};
const RADIUS: i32 = ${f};
${S}`,body:`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${_}
  ${v(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
        let sampleIndex = u32(sampleRow * WIDTH + sampleColumn);
        let normalXValue = normalX[normalXOffset + sampleIndex];
        if (isFiniteValue(normalXValue)) {
          vectorSum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
          count += 1.0;
        }
      }
    }`)}`});return[...c.nodes,d,n]}let y=[`x`,`y`,`z`].map(n=>u(e,`${t}-row-sum-${n}`,`float32`,s)),x=u(e,`${t}-row-count`,`uint32`,s),C=p(e,{id:`${t}-window-rows`,operation:`GPUTerrainVectorRuggedness`,variant:`window-rows`,bindings:[{name:`normalX`,view:l[0],type:`f32`,access:`read`},{name:`normalY`,view:l[1],type:`f32`,access:`read`},{name:`normalZ`,view:l[2],type:`f32`,access:`read`},{name:`rowSumX`,view:y[0],type:`f32`,access:`read_write`},{name:`rowSumY`,view:y[1],type:`f32`,access:`read_write`},{name:`rowSumZ`,view:y[2],type:`f32`,access:`read_write`},{name:`rowCounts`,view:x,type:`u32`,access:`read_write`},m],invocationCount:s,declarations:`const WIDTH: i32 = ${a};
const RADIUS: i32 = ${f};
${S}`,body:`let column = i32(index) % WIDTH;
  let rowStart = i32(index) - column;
  ${_}
  var sum = vec3<f32>(0.0);
  var count = 0u;
  for (var sampleColumn = max(column - radius, 0); sampleColumn <= min(column + radius, WIDTH - 1); sampleColumn++) {
    let sampleIndex = u32(rowStart + sampleColumn);
    let normalXValue = normalX[normalXOffset + sampleIndex];
    if (isFiniteValue(normalXValue)) {
      sum += vec3<f32>(normalXValue, normalY[normalYOffset + sampleIndex], normalZ[normalZOffset + sampleIndex]);
      count += 1u;
    }
  }
  rowSumX[rowSumXOffset + index] = sum.x;
  rowSumY[rowSumYOffset + index] = sum.y;
  rowSumZ[rowSumZOffset + index] = sum.z;
  rowCounts[rowCountsOffset + index] = count;`}),T=p(e,{id:`${t}-window`,operation:`GPUTerrainVectorRuggedness`,variant:`window-columns`,bindings:[{name:`normalX`,view:l[0],type:`f32`,access:`read`},{name:`rowSumX`,view:y[0],type:`f32`,access:`read`},{name:`rowSumY`,view:y[1],type:`f32`,access:`read`},{name:`rowSumZ`,view:y[2],type:`f32`,access:`read`},{name:`rowCounts`,view:x,type:`u32`,access:`read`},...g.slice(3)],invocationCount:s,declarations:`const WIDTH: i32 = ${a};
const HEIGHT: i32 = ${o};
const RADIUS: i32 = ${f};
${S}`,body:`let row = i32(index) / WIDTH;
  let column = i32(index) % WIDTH;
  ${_}
  ${v(`for (var sampleRow = max(row - radius, 0); sampleRow <= min(row + radius, HEIGHT - 1); sampleRow++) {
      let sampleIndex = u32(sampleRow * WIDTH + column);
      vectorSum += vec3<f32>(rowSumX[rowSumXOffset + sampleIndex], rowSumY[rowSumYOffset + sampleIndex], rowSumZ[rowSumZOffset + sampleIndex]);
      count += f32(rowCounts[rowCountsOffset + sampleIndex]);
    }`)}`});return[...c.nodes,d,C,T]}},Ye=[1150,780,380,380];function Xe(e){let[t,n,r,i]=Ye,[a,o]=e.getLongitudeLatitude(t-.5,n-.5),[s,c]=e.getLongitudeLatitude(t+r-.5,n+i-.5);return[a,c,s,o]}function Ze(e,t,n){let r=e.packedWords.slice();if(t>0){let e=O(2024),n=t/100;for(let t=0;t<r.length;t++)if(e()<n){let n=r[t],i=n&255,a=e()<.5?i+1:i-1,o=a<0||a>255?i+(i<128?1:-1):a;r[t]=n&4294967040|o}}if(n){let[t,n,i,a]=Ye;for(let o=n;o<n+a;o++)for(let n=t;n<t+i;n++)r[o*e.width+n]&=16777215}return r}async function Qe(e,n,r,i){let{device:a}=e,{pixelCount:s}=i,c=n.createBuffer(`encoded`,Ze(i,e.options.spikeDensity,e.options.missingTile)),l=n.createBuffer(`decoded`,s*4),u=n.createBuffer(`decoded-validity`,s*4),d=n.createBuffer(`repaired`,s*4),f=n.createBuffer(`repaired-validity`,s*4),p=n.createBuffer(`repair-statistics`,32),m=n.createParameterBuffer(`select-flag`,`float32`,4),h=null,g=``,_=n.track(new M),v=e=>`${e.encoding}|${e.validRange}|${e.alphaNoData}|${e.clampBathymetry}`,y=async e=>{let r=new t(a,{id:`terrain-basics-decode`});return r.add(new q({id:`decode`,grid:i.grid,encoding:e.encoding,input:{buffer:j(r,`encoded`,c,`uint32`,s)},alphaNoData:e.alphaNoData,validRange:Te[e.validRange],clampBathymetry:e.clampBathymetry,output:{values:j(r,`decoded`,l,`float32`,s),outputValidity:j(r,`decoded-validity`,u,`uint32`,s)}})),n.track(await r.compileAsync())},b=e=>{let t=h;h=e,t&&n.release(t),O=!0},x=new t(a,{id:`terrain-basics-repair`});x.add(new Ve({id:`repair`,grid:i.grid,elevation:{id:`decoded`,format:`float32`,storage:{kind:`buffer`,values:j(x,`decoded`,l,`float32`,s)},validity:j(x,`decoded-validity`,u,`uint32`,s)},output:{values:j(x,`repaired`,d,`float32`,s),outputValidity:j(x,`repaired-validity`,f,`uint32`,s),statistics:j(x,`statistics`,p,`uint32`,5)}}));let S=n.track(await x.compileAsync()),C=new t(a,{id:`terrain-basics-select`});P(C,{id:`select-elevation`,invocationCount:s,bindings:[{name:`flag`,view:m.importToGraph(C),type:`f32`,access:`read`},{name:`rawHeights`,view:j(C,`decoded`,l,`float32`,s),type:`f32`,access:`read`},{name:`rawValid`,view:j(C,`decoded-validity`,u,`uint32`,s),type:`u32`,access:`read`},{name:`fixedHeights`,view:j(C,`repaired`,d,`float32`,s),type:`f32`,access:`read`},{name:`fixedValid`,view:j(C,`repaired-validity`,f,`uint32`,s),type:`u32`,access:`read`},{name:`outHeights`,view:j(C,`elevation`,r.elevationBuffer,`float32`,s),type:`f32`,access:`read_write`},{name:`outValid`,view:j(C,`validity`,r.validityBuffer,`uint32`,s),type:`u32`,access:`read_write`}],body:`
  if (flag[flagOffset] > 0.5) {
    outHeights[outHeightsOffset + index] = fixedHeights[fixedHeightsOffset + index];
    outValid[outValidOffset + index] = fixedValid[fixedValidOffset + index];
  } else {
    outHeights[outHeightsOffset + index] = rawHeights[rawHeightsOffset + index];
    outValid[outValidOffset + index] = rawValid[rawValidOffset + index];
  }`});let w=n.track(await C.compileAsync()),T=new N(n,`repair-statistics`,[{buffer:p,size:20}],t=>{let[n,r,i,a,s]=new Uint32Array(t);e.setReadout(`repairJumps`,`${o(n)} jumps found`),e.setReadout(`repairResult`,`${o(r)} px repaired in ${o(i)} components, ${o(a)} jumps left${s?``:` (did not converge)`}`)}),E=()=>m.write(Float32Array.of(e.options.repairSpikes?1:0,0,0,0));E();let D=!1,O=!0,k=!0;return g=v(e.options),b(await y(e.options)),{getCompiledGraphs:()=>[...h?[h]:[],S,w],decodeOptionChanged(){let t=v(e.options);t!==g&&(g=t,_.request({build:()=>y(e.options),commit:b,discard:e=>n.release(e)}))},inputChanged(){D=!0,O=!0},selectionChanged(){E(),k=!0},encode(t){D&&=(c.write(Ze(i,e.options.spikeDensity,e.options.missingTile)),!1);let n=!1;return O&&h?(h.encode(t,{parameters:void 0}),S.encode(t,{parameters:void 0}),w.encode(t,{parameters:void 0}),O=!1,k=!1,T.request(t),n=!0):k&&(w.encode(t,{parameters:void 0}),k=!1,n=!0),T.flush(t),n},stop(){T.stop()}}}var $e=12,et=20,tt=3e38;function Q(e,t=new Float32Array(52)){let n=e.colors??[],r=Math.min(8,Math.max(1,n.length));t.fill(0),t[0]=e.mode===`classes`?0:e.mode===`ramp`?1:2,t[1]=r,t[2]=e.useSecond?1:0,t[3]=e.ramp?k[e.ramp]:0,t[4]=e.low??0,t[5]=e.high??1,t[6]=e.floor??-tt,t[7]=e.alpha??1,t[8]=e.flatDegrees??5,t[9]=e.fullDegrees??25;for(let n=0;n<7;n++){let r=e.breaks?.[n];t[$e+n]=r!==void 0&&Number.isFinite(r)?r:tt}for(let e=0;e<r;e++){let r=n[e]??[0,0,0,0];t[et+e*4]=r[0]/255,t[et+e*4+1]=r[1]/255,t[et+e*4+2]=r[2]/255,t[et+e*4+3]=(r[3]??255)/255}return t}function nt(e,t){let n=t.second!==void 0;P(e,{id:t.id,invocationCount:t.count,bindings:[{name:`firstRaster`,view:t.first,type:`f32`,access:`read`},...t.second?[{name:`secondRaster`,view:t.second,type:`f32`,access:`read`}]:[],{name:`paint`,view:t.settings,type:`f32`,access:`read`},{name:`cells`,view:t.output,type:`u32`,access:`read_write`}],declarations:`
${A()}
const BREAKS_OFFSET: u32 = ${$e}u;
const COLORS_OFFSET: u32 = ${et}u;
const BREAK_COUNT: u32 = 7u;
fn isNotFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn getClassOf(value: f32, classCount: u32) -> u32 {
  var found = 0u;
  for (var breakIndex = 0u; breakIndex < BREAK_COUNT; breakIndex = breakIndex + 1u) {
    if (breakIndex + 1u < classCount && value >= paint[paintOffset + BREAKS_OFFSET + breakIndex]) {
      found = breakIndex + 1u;
    }
  }
  return found;
}`,body:`
  let mode = u32(paint[paintOffset]);
  let classCount = u32(paint[paintOffset + 1u]);
  let useSecond = paint[paintOffset + 2u] > 0.5;
  let rampIndex = u32(paint[paintOffset + 3u]);
  let rampLow = paint[paintOffset + 4u];
  let rampHigh = paint[paintOffset + 5u];
  let rampFloor = paint[paintOffset + 6u];
  let alpha = paint[paintOffset + 7u];
  let firstValue = firstRaster[firstRasterOffset + index];
  ${n?`let secondValue = secondRaster[secondRasterOffset + index];`:`let secondValue = firstValue;`}
  var color = vec4<f32>(0.0);
  if (mode == 2u) {
    // Aspect hue, fading in with the slope; flat ground (aspect -1) and no-data stay empty.
    if (!isNotFinite(firstValue) && !isNotFinite(secondValue) && firstValue >= 0.0) {
      let flatDegrees = paint[paintOffset + 8u];
      let fullDegrees = paint[paintOffset + 9u];
      let weight = clamp((secondValue - flatDegrees) / max(fullDegrees - flatDegrees, 1e-6), 0.0, 1.0);
      let hue = fract(firstValue / 360.0);
      color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, hue), alpha * weight);
    }
  } else {
    let value = select(firstValue, secondValue, useSecond);
    if (!isNotFinite(value)) {
      if (mode == 1u) {
        if (value >= rampFloor) {
          let fraction = clamp((value - rampLow) / max(rampHigh - rampLow, 1e-6), 0.0, 1.0);
          color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, fraction), alpha);
        }
      } else {
        let found = getClassOf(value, classCount);
        let base = paintOffset + COLORS_OFFSET + found * 4u;
        color = vec4<f32>(paint[base], paint[base + 1u], paint[base + 2u], paint[base + 3u]);
      }
    }
  }
  cells[cellsOffset + index] = pack4x8unorm(color);`})}var rt=[`north`,`north-east`,`east`,`south-east`,`south`,`south-west`,`west`,`north-west`];function it(e){return rt[Math.round((e%360+360)%360/45)%8]}function at(e,t=ue){let n=Array(t.length+1).fill(0),r=Array(90).fill(0),i=0,a=-1/0,o=-1;for(let s=0;s<e.length;s++){let c=e[s];if(!Number.isFinite(c))continue;i++,c>a&&(a=c,o=s);let l=0;for(;l<t.length&&c>=t[l];)l++;n[l]++,r[Math.min(89,Math.max(0,Math.floor(c)))]++}return{valid:i,classCounts:n,histogram:r,maximum:i>0?a:NaN,maximumIndex:o}}function ot(e,t){if(e.valid===0)return NaN;let n=0;for(let r=t;r<e.classCounts.length;r++)n+=e.classCounts[r];return n/e.valid}function st(e,t){let n=Array(16).fill(0),r=0,i=0,a=360/16;for(let o=0;o<e.length;o++){let s=e[o],c=t[o];!(s>=0)||!(c>=W.fullDegrees)||(r++,n[Math.floor((s+a/2)/a)%16]++,(s>=292.5||s<67.5)&&i++)}return{steepCount:r,sectorCounts:n,northCount:i}}function ct(e,t){let n=0,r=1/0,i=-1/0,a=0;for(let o=0;o<e.length;o++){let s=e[o];Number.isFinite(s)&&(n++,s<r&&(r=s),s>i&&(i=s),t&&(a=Math.max(a,Math.abs(s-t[o]))))}return{valid:n,minimum:r,maximum:i,maximumDifference:t?a:null}}function lt(e,t,n){let r=t.colors.map((e,t)=>t===0?[150,156,164,255]:[e[0],e[1],e[2],255]);return{kind:`histogram`,values:e.histogram,xDomain:[0,90],breaks:t.breaks,classColors:r,markers:[{x:n,label:`${n}°`}],xLabel:`Slope (degrees)`,yLabel:`Cells`,formatX:e=>`${e}°`,height:120,title:`Cells by slope, one bar per degree`,description:`Histogram of slope in one-degree bins, coloured by the map classes, with the decision threshold marked.`}}function ut(e){return{kind:`rose`,values:e.sectorCounts.map(t=>e.steepCount?t/e.steepCount:0),labels:[`N`,`NE`,`E`,`SE`,`S`,`SW`,`W`,`NW`],colors:e.sectorCounts.map((e,t)=>{let[n,r,i]=de(t*22.5);return[n,r,i,255]}),height:190,title:`Share of steep cells by the direction they face`,description:`Rose chart of steep cells in sixteen compass sectors, each coloured as on the map.`}}var dt=e=>[e[0],e[1],e[2],255],ft=[150,156,164,255];function pt(e){return e.toLocaleString(`en-US`,{maximumFractionDigits:1,minimumFractionDigits:1})}function mt(e,t){let n=I(e,t);if(n<0)return ft;let r=e.colors[n];return(r[3]??255)===0?ft:dt(r)}function ht(e){let{state:t,tables:n,cell:r,probe:i}=e,{value:a,elevation:o,column:s,row:c}=r;if(!Number.isFinite(o))return{title:`No data`,subtitle:`Decoded height`,rows:[{label:`Height`,value:`no data`,swatch:n.decoded.noData?.color}],note:`The decode marked this cell nodata: alpha 0, outside the valid range, or the wrong decoder.`};let l={label:`Elevation`,value:pt(o),unit:`m`};if(t.view!==`analysis`){let r=t.view===`decoded`?n.decoded:n.elevation,i=I(r,o);return{title:`Elevation`,subtitle:L(r,i)?`${L(r,i)} m class`:``,rows:[{...l,swatch:mt(r,o),emphasis:!0},...e.onGlacier&&t.view===`relief`?[{label:`Surface`,value:`glacier (OpenStreetMap outline)`}]:[]]}}let u=i.hornAt(s,c),d=u.neighbours.map(e=>Math.round(e).toLocaleString(`en-US`)),f=[{label:`Window, north row`,value:d.slice(0,3).join(`   `),unit:`m`},{label:`Window, middle row`,value:d.slice(3,6).join(`   `),unit:`m`},{label:`Window, south row`,value:d.slice(6,9).join(`   `),unit:`m`}],p={kind:`box`,bounds:e.getWindowBounds(s,c)},m={label:`Slope here`,value:u.slopeDeg.toFixed(1),unit:`°`};switch(t.product){case`slope`:{if(!Number.isFinite(a))return{title:`No slope`,rows:[l],note:`Edge cell or no data.`};let r=I(n.slope,a),i=e.groundCellMeters/e.mercatorCellMeters,o=Math.tan(a*Math.PI/180),s=Math.atan(t.cellModel===`ground`?o/i:o*i)*180/Math.PI;return{title:`${a.toFixed(1)}° slope`,subtitle:`Class ${L(n.slope,r)}`,rows:[{label:t.cellModel===`ground`?`Slope, ground cells`:`Slope, Mercator pixels`,value:a.toFixed(1),unit:`°`,swatch:mt(n.slope,a),emphasis:!0},{label:t.cellModel===`ground`?`On Mercator pixels`:`On ground cells`,value:s.toFixed(1),unit:`°`},{label:`Faces`,value:Number.isFinite(u.aspectDeg)?`${it(u.aspectDeg)} (${u.aspectDeg.toFixed(0)}°)`:`flat: no direction`},l,...f,{label:`dz/dx, dz/dy`,value:`${u.dzdx.toFixed(2)}, ${u.dzdy.toFixed(2)}`,unit:`m per m`}],highlight:p}}case`aspect`:{let e=Number.isFinite(a)&&a>=0,[t,n,r]=e?de(a):ft;return{title:e?`Faces ${it(a)}`:`Flat: no direction`,rows:[{label:`Aspect`,value:e?a.toFixed(0):`none`,unit:e?`°`:void 0,swatch:[t,n,r,255],emphasis:!0},m,l],note:u.slopeDeg<W.flatDegrees?`Flat ground is not coloured: its aspect is noise.`:void 0,highlight:p}}case`tpi`:{if(!Number.isFinite(a))return{title:`No TPI`,rows:[l]};let e=a>n.tpi.breaks[2]?`above its 8 neighbours (ridge or knoll)`:a<n.tpi.breaks[1]?`below its 8 neighbours (hollow)`:`about level with its neighbours`;return{title:`TPI ${a>=0?`+`:``}${a.toFixed(1)} m`,subtitle:e,rows:[{label:`TPI`,value:`${a>=0?`+`:``}${a.toFixed(1)}`,unit:`m`,swatch:mt(n.tpi,a),emphasis:!0},m,l],highlight:p}}case`tri`:case`vrm`:{if(!Number.isFinite(a))return{title:`No ruggedness`,rows:[l]};let e=t.product===`tri`;return{title:e?`TRI ${a.toFixed(1)} m`:`VRM ${a.toFixed(3)}`,subtitle:`Class ${L(n.rugged,I(n.rugged,a))}`,rows:[{label:e?`TRI`:`VRM`,value:e?a.toFixed(1):a.toFixed(3),unit:e?`m`:`index 0-1`,swatch:mt(n.rugged,a),emphasis:!0},m,l],highlight:p}}}}var gt=220,_t=ue.indexOf(we)+1,vt=ue.indexOf(De)+1,yt={mode:`ramp`,ramp:`grayscale`,low:0,high:1,alpha:0};function $(e,t){switch(t){case`slope`:case`aspect`:return e.borderMode;case`tpi`:return e.edgeMode;case`tri`:return`${e.triAlgorithm}|${e.edgeMode}`;case`vrm`:return`${e.vrmRadius}|${e.borderMode}`}}async function bt(e){let{device:t}=e;e.setStatus(`Loading the elevation tile and its OpenStreetMap context`);let[n,r]=await Promise.all([H(e.datasets.get(`alps-dem`),e.signal),fe(e.datasets.get(`alps-context`),e.signal)]);e.signal.throwIfAborted();let{width:i,height:a,pixelCount:o}=n,c=ie(n),l=ae(c),u=he(r.glacierGeoJson,c),d=Number(e.datasets.get(`alps-dem`).raster?.spec?.verticalQuantisationM??.25),f=new s(t,`terrain-basics`),p=new re(e,f,n),m=ge({dem:c,device:t,ground:e.ground(),glacierMask:u});p.setGround(m),e.setStatus(`Building the shaded relief`),await m.prepare(),e.signal.throwIfAborted();let h=await Qe(e,f,p,n),g=n.groundCellSize,_=n.mercatorCellSize,v=me(r,{window:n.lngLatBounds,names:[`Gornergletscher`],max:1}),y=new Map,b=e=>e.product===`tri`||e.product===`vrm`?`${e.product}|${$(e,e.product)}`:null,x=()=>{let t=e.options,n=b(t),r=n?y.get(n):void 0;return r&&(t.product===`tri`||t.product===`vrm`)?{product:t.product,breaks:r}:null},S=je(e.ground(),x()),C=()=>{let t=x();S=je(e.ground(),t),e.setLegendData(`tables`,S),e.setLegendData(`ruggednessBreaks`,t)},w=t=>{let n=e.options;switch(t){case`decoded`:return{mode:`classes`,breaks:ce,colors:S.decoded.colors};case`slope`:{let e=n.cellModel===`mercator`;return n.slopeDisplay===`continuous`?{mode:`ramp`,ramp:le.ramp,low:le.domain[0],high:le.domain[1],floor:le.transparentBelowDegrees,alpha:be,useSecond:e}:{mode:`classes`,breaks:S.slope.breaks,colors:S.slope.colors,useSecond:e}}case`aspect`:return{mode:`aspect`,ramp:se,alpha:ye,flatDegrees:W.flatDegrees,fullDegrees:W.fullDegrees};case`tpi`:return{mode:`classes`,breaks:S.tpi.breaks,colors:S.tpi.colors};case`tri`:case`vrm`:return x()?{mode:`classes`,breaks:S.rugged.breaks,colors:S.rugged.colors}:{mode:`classes`,breaks:[],colors:[[0,0,0,0]]}}},T=new Map,E=null,D=0,O=0,k=0,A=!1,N=[],P=(e,t,n,r,i,a=null)=>({product:e,configKey:t,build:n,mercatorBuild:a,paintBuffer:r,buffers:i,seenRunCount:-1,seenSummaryVersion:-1});async function I(){let e=p.builder(`decoded`),t=e.settings(52),n=e.words(`paint`);nt(e.graph,{id:`decoded-paint`,count:o,first:j(e.graph,`elevation`,p.elevationBuffer,`float32`,o),settings:t.view,output:n});let r=await e.finishStage({write:()=>t.parameters.write(Q(w(`decoded`)))});return P(`decoded`,``,p.createBuild(`decoded`,r,p.elevationBuffer,`float32`),e.getBuffer(`paint`),{})}async function L(t){let r=p.builder(`slope`),i=r.elevation(),a=r.settings(8),s=r.floats(`ground`),c=r.floats(`mercator`),l={grid:n.grid,elevation:i,parameters:a.view,rowDirection:`south`,borderMode:t.borderMode};r.graph.add(new V({...l,id:`derivatives-ground`,output:{slope:s},cellSizeMode:`web-mercator`})),r.graph.add(new V({...l,id:`derivatives-mercator`,output:{slope:c},cellSizeMode:`uniform`}));let u=r.settings(52),d=r.words(`paint`);nt(r.graph,{id:`slope-paint`,count:o,first:s,second:c,settings:u.view,output:d});let f=await r.finish({value:`ground`,format:`float32`,write:()=>{a.parameters.write(B({zFactor:e.options.zFactor},n.grid)),u.parameters.write(Q(w(`slope`)))}}),m=p.createBuild(`slope-mercator`,f.stage,r.getBuffer(`mercator`),`float32`);return P(`slope`,$(t,`slope`),f,r.getBuffer(`paint`),{ground:r.getBuffer(`ground`),mercator:r.getBuffer(`mercator`)},m)}async function ee(t){let r=p.builder(`aspect`),i=r.elevation(),a=r.settings(8),s=r.floats(`aspect`),c=r.floats(`slope`);r.graph.add(new V({id:`derivatives`,grid:n.grid,elevation:i,parameters:a.view,cellSizeMode:`web-mercator`,rowDirection:`south`,borderMode:t.borderMode,output:{slope:c,aspect:s}}));let l=r.settings(52),u=r.words(`paint`);nt(r.graph,{id:`aspect-paint`,count:o,first:s,second:c,settings:l.view,output:u});let d=await r.finish({value:`aspect`,format:`float32`,write:()=>{a.parameters.write(B({zFactor:e.options.zFactor},n.grid)),l.parameters.write(Q(w(`aspect`)))}});return P(`aspect`,$(t,`aspect`),d,r.getBuffer(`paint`),{aspect:r.getBuffer(`aspect`),slope:r.getBuffer(`slope`)})}async function te(e,t){let r=p.builder(t),i=r.elevation(),a=r.floats(t);r.graph.add(new He({id:`ruggedness`,grid:n.grid,elevation:i,output:t===`tpi`?{topographicPositionIndex:a}:{terrainRuggednessIndex:a},terrainRuggednessAlgorithm:e.triAlgorithm,edgeMode:e.edgeMode}));let s=r.settings(52),c=r.words(`paint`);nt(r.graph,{id:`${t}-paint`,count:o,first:a,settings:s.view,output:c});let l=await r.finish({value:t,format:`float32`,write:()=>s.parameters.write(Q(w(t)))});return P(t,$(e,t),l,r.getBuffer(`paint`),{})}async function ne(t){let r=p.builder(`vrm`),i=r.elevation(),a=r.settings(8),s=r.floats(`vrm`);r.graph.add(new Je({id:`vector-ruggedness`,grid:n.grid,elevation:i,parameters:a.view,maximumRadius:t.vrmRadius,cellSizeMode:`web-mercator`,rowDirection:`south`,borderMode:t.borderMode,output:{vectorRuggedness:s}}));let c=r.settings(52),l=r.words(`paint`);nt(r.graph,{id:`vrm-paint`,count:o,first:s,settings:c.view,output:l});let u=await r.finish({value:`vrm`,format:`float32`,minIntervalMs:120,write:()=>{a.parameters.write(qe({zFactor:e.options.zFactor},n.grid)),c.parameters.write(Q(w(`vrm`)))}});return P(`vrm`,$(t,`vrm`),u,r.getBuffer(`paint`),{})}let R=await I();await p.prepareBuild(R.build),p.addBuild(R.build,``);function z(e){let t=T.get(e.product);return t&&t.configKey===$(e,e.product)?t:void 0}async function ue(e){let t;switch(e.product){case`slope`:t=await L(e);break;case`aspect`:t=await ee(e);break;case`tpi`:case`tri`:t=await te(e,e.product);break;case`vrm`:t=await ne(e);break}try{await Promise.all([t.build,t.mercatorBuild].map(e=>e&&p.prepareBuild(e)))}catch(e){throw de(t),e}return t}function de(e){p.discardBuild(e.build),e.mercatorBuild&&p.discardBuild(e.mercatorBuild)}function Te(t){p.addBuild(t.build,t.configKey),t.mercatorBuild&&p.addBuild(t.mercatorBuild,t.configKey),T.set(t.product,t),D++,e.setReadout(`rebuilds`,`${D} product graphs compiled`)}let Me=f.track(new M),G={},K=null,q=[],Ne=!1,J={};function Pe(e){let t=q.findIndex(t=>t.key===e.key);t>=0&&q.splice(t,1),q.push(e)}function Fe(e){if(Ne||q.length===0)return;let t=q[0];p.readBulk(e,t.buffer,e=>{Ne=!1,!A&&t.accept(new Float32Array(e.buffer,e.byteOffset,o))})&&(Ne=!0,q.shift())}let Y=e=>Number.isFinite(e)?`${(e*100).toFixed(1)}%`:`-`;function X(){let t=G[e.options.cellModel];e.setLegendData(`slopeCounts`,{ground:G.ground?.classCounts,mercator:G.mercator?.classCounts}),e.setLegendData(`slopeHistogram`,t?.histogram??null),e.setReadout(`steeperThan30`,t?Y(ot(t,_t)):null),e.setReadout(`steeperThan45`,t?Y(ot(t,vt)):null),e.setChart(`slopeHistogram`,t?lt(t,S.slope,we):null);let n=G.ground,r=G.mercator;e.setReadout(`share30Ground`,n?Y(ot(n,_t)):null),e.setReadout(`share30Mercator`,r?Y(ot(r,_t)):null),e.setReadout(`maxSlopeGround`,n?`${n.maximum.toFixed(1)}°`:null),We()}function Ie(){if(!K){e.setChart(`aspectRose`,null),e.setReadout(`northShare`,null);return}e.setChart(`aspectRose`,ut(K)),e.setReadout(`northShare`,K.steepCount>0?Y(K.northCount/K.steepCount):`-`)}function Le(e){let t=O,n=()=>t===O;if(e.product===`slope`)for(let t of[`ground`,`mercator`])Pe({key:`slope-${t}`,buffer:e.buffers[t],accept:e=>{n()&&(G[t]=at(e),X())}});else if(e.product===`aspect`){J.aspect=void 0,J.slope=void 0;for(let t of[`aspect`,`slope`])Pe({key:`aspect-${t}`,buffer:e.buffers[t],accept:e=>{n()&&(J[t]=e,J.aspect&&J.slope&&(K=st(J.aspect,J.slope),J.aspect=void 0,J.slope=void 0,Ie()))}})}}function Re(){let t=e.options,r=t.encoding===`terrarium`&&t.spikeDensity===0&&!t.missingTile&&!t.clampBathymetry,i=k;Pe({key:`elevation`,buffer:p.elevationBuffer,accept:t=>{if(i!==k)return;let a=ct(t,r?n.cpuElevation:null);e.setReadout(`validPixels`,`${(a.valid/o*100).toFixed(2)}% (${a.valid.toLocaleString(`en-US`)} px)`),e.setReadout(`relief`,a.valid>0?`${Math.round(a.maximum-a.minimum).toLocaleString(`en-US`)} m`:null),e.setReadout(`cells`,a.valid>0?Ee(a.valid,`cells`):null),e.setReadout(`summitHeight`,a.valid>0?`${Math.round(a.maximum).toLocaleString(`en-US`)} m`:null),e.setReadout(`decodeDifference`,a.maximumDifference===null?`n/a (input modified)`:a.maximumDifference===0?`exactly 0 m (bit-identical)`:`${a.maximumDifference.toExponential(2)} m`),Z=a.valid>0?a.maximum:null,We()}})}let Z=null;function ze(t,n){let r=e.options;if(r.view!==`analysis`||n.kind!==`float`){e.setReadout(`p98`,null);return}let i=b(r);i&&!y.has(i)&&n.count>0&&(y.set(i,xe.map(e=>Number(n.quantile(e).toPrecision(4)))),C(),p.markDirty(E?.build),e.requestLayers());let a=n.quantile(.98);switch(r.product){case`slope`:e.setReadout(`p98`,`${a.toFixed(1)}°`);break;case`tpi`:case`tri`:e.setReadout(`p98`,`${a.toFixed(1)} m`);break;case`vrm`:e.setReadout(`p98`,a.toFixed(3));break;default:e.setReadout(`p98`,null)}}p.onStats=ze;let Be=null;p.describeHover=t=>(Be=t,e.refreshTooltip(),null);let Ve=(e,t,r=1)=>{let[i,a]=n.getLongitudeLatitude(e-r-.5,t-r-.5),[o,s]=n.getLongitudeLatitude(e+r+.5,t+r+.5);return[i,s,o,a]};function Ue(){let t=e.options,r=t.view===`analysis`&&t.product===`vrm`?[Ae(t.vrmRadius,g)]:[];e.setFurniture({title:{subtitle:Se(t,g,_),sample:oe(n)},scaleBar:{units:`metric`,ticks:r}}),e.setReadout(`taps`,t.product===`vrm`?`${(2*t.vrmRadius+1)**2} taps per cell, ${((2*t.vrmRadius+1)**2*o/1e6).toFixed(0)} M in all`:null)}function We(){let t=e.options,r=[],a=t.view===`analysis`&&t.product===`slope`;if(t.view!==`decoded`&&r.push(...v),(t.view===`relief`||a)&&r.push(Oe(n.lngLatBounds)),t.view===`relief`&&Z!==null){let e=pe(c,F.places.matterhorn.lngLat);e&&r.push(ke(e.lngLat,Z,g,d))}if(t.view===`decoded`&&t.missingTile&&r.push(Ce(Xe(n))),a){let e=G[t.cellModel];if(e&&Number.isFinite(e.maximum)&&e.maximumIndex>=0){let a=e.maximumIndex%i,o=Math.floor(e.maximumIndex/i);r.push(ve(n.getLongitudeLatitude(a,o),e.maximum,t.cellModel))}}e.setAnnotations(`basics`,r)}function Ge(){let t=e.options,n,r;if(t.view===`analysis`){if(n=z(t),!n){Me.request({build:()=>ue(e.options),commit:e=>{Te(e),Ge()},discard:de});return}r=t.product===`slope`&&t.cellModel===`mercator`&&n.mercatorBuild?n.mercatorBuild:n.build}else n=R,r=n.build;E=n,p.activate(r,yt),p.markDirty(n.build),C(),X(),Ie(),Ue(),We(),e.requestLayers()}function Ke(){N.push(setTimeout(()=>!A&&Ge(),gt))}return e.setReadout(`decisionSlope`,`${we}°`),e.setReadout(`groundCell`,`${g.toFixed(1)} m`),e.setReadout(`mercatorCell`,`${_.toFixed(1)} m`),e.setReadout(`aspectFade`,`${W.flatDegrees}° to ${W.fullDegrees}°`),e.setReadout(`naive45`,`${De}° reads as ${(Math.atan(Math.tan(De*Math.PI/180)*(g/_))*180/Math.PI).toFixed(1)}°`),e.setReadout(`rebuilds`,`none yet`),e.options.view===`analysis`&&!z(e.options)&&Te(await ue(e.options)),Ge(),Re(),{getCompiledGraphs:()=>[...h.getCompiledGraphs(),...p.getCompiledGraphs()],setOption(e){switch(e){case`view`:case`product`:case`cellModel`:case`slopeDisplay`:Ge();return;case`encoding`:case`validRange`:case`alphaNoData`:case`clampBathymetry`:h.decodeOptionChanged();break;case`missingTile`:case`spikeDensity`:h.inputChanged();break;case`repairSpikes`:h.selectionChanged();break;case`borderMode`:case`triAlgorithm`:case`edgeMode`:case`vrmRadius`:Ke();break;case`zFactor`:O++,p.markAllDirty();break}Ue(),We()},onAction(t){t===`measure`&&(e.setReadout(`timing`,`measuring...`),p.measure().then(t=>{if(A)return;let n=t.reduce((e,t)=>e+t.milliseconds,0);e.setReadout(`timing`,t.length===0?`static raster (no graph)`:`${n.toFixed(2)} ms for ${(o/1e6).toFixed(1)} M cells (${t[0].method===`gpu-timestamps`?`GPU timestamps`:`wall clock`})`)}))},onGroundChange(t){m.setGround(t),C(),p.markAllDirty(),X(),e.requestLayers()},onThemeChange(){e.requestLayers()},getTooltip(t){if(p.getTooltip(t),!t.coordinate)return null;let r=n.getPixel(t.coordinate[0],t.coordinate[1]),a=Be;if(!r||!a||Math.abs(a.column-r[0])>3||Math.abs(a.row-r[1])>3)return null;let o=e.options;return ht({state:o,tables:S,cell:a,probe:l,groundCellMeters:g,mercatorCellMeters:_,onGlacier:u[a.row*i+a.column]===1,getWindowBounds:(e,t)=>Ve(e,t,o.product===`vrm`?o.vrmRadius:1)})},encode(t){h.encode(t)&&(k++,O++,p.elevationChanged(),Re()),p.encode(t);let n=E,r=e.options;if(n&&r.view===`analysis`){let{runCount:e}=n.build.stage;n.seenRunCount!==e&&(n.seenRunCount=e,n.seenSummaryVersion!==O&&(n.seenSummaryVersion=O,Le(n)))}Fe(t)},getLayers(){let t=e.options,r=p.getLayers({underlay:t.view!==`decoded`,underlayAlpha:1,alpha:1,showProduct:!1});return t.view!==`relief`&&E&&r.push(new U({id:`terrain-basics-product`,coordinateOrigin:[n.origin[0],n.origin[1],0],gridSize:[i,a],bounds:n.bounds,colors:E.paintBuffer,alpha:1,...t.view===`decoded`?{noDataHatch:_e(e.ground())}:{}})),r},destroy(){A=!0;for(let e of N)clearTimeout(e);N=[],h.stop(),p.destroy(),m.destroy(),f.destroy()}}}export{bt as createTerrainBasics};