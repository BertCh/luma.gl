import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as r}from"./src-CNCKFW2b.js";import{n as i}from"./earth-xB16XgAa.js";import"./analysis-frame-Blu0tYdw.js";import{i as a}from"./convert-colors-D-TZjU1C.js";import{M as o,N as s,_ as c,f as l,h as u,i as d,j as f,n as p,t as m,w as h}from"./resources-CwvhEOBe.js";import{t as g}from"./gpu-scan-BuGhTLDQ.js";import{t as _}from"./draw-command-buffer-BeqRkhBa.js";import{n as ee,s as v}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as y}from"./raster-band-utils-CA-sh8xt.js";import{n as te,t as b}from"./philox-random-CHwyk0dd.js";import{a as ne,d as re,o as x,s as ie,t as ae}from"./raster-relaxation-BJ40xgro.js";import{t as S}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";import{i as C,n as oe,r as w,t as T}from"./gpu-particle-advection-B0dTzuCj.js";import{n as E}from"./gpu-raster-grid-BOSvDY5A.js";import{T as D,k as O,r as k}from"./index-Bjk18V2I.js";import{t as se}from"./graph-swap-41Ru7LVu.js";import{t as ce}from"./summary-reader-CsgWxENf.js";import{o as le}from"./b16-colors-Jr7XN5JQ.js";import{i as A,n as ue}from"./b16-common-3eXVh1Vw.js";var j=b.lineIntegralConvolutionNoise.id;function de(e,t=new Float32Array(12)){if(t.length<12)throw Error(`Line integral convolution parameter target must hold 12 elements`);return t.set([...e.fieldExtent,...e.outputExtent,e.stepLength??.5,e.minimumSpeed??0,e.phase??0,e.period??0]),t}function fe(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Line integral convolution uint32 parameter target must hold 4 elements`);let n=e.seed??0;if(!Number.isInteger(n)||n<0||n>4294967295)throw Error(`Line integral convolution seed must be a uint32 integer`);return t.set([n,0,0,0]),t}var M=`GPULineIntegralConvolution`,pe=256,me=class{id;stepCount;props;constructor(e){let t=e.id??`line-integral-convolution`;for(let[n,r]of[[`fieldWidth`,e.fieldWidth],[`fieldHeight`,e.fieldHeight],[`grid width`,e.grid.width],[`grid height`,e.grid.height]])if(!Number.isInteger(r)||r<1||r>65535)throw Error(`${t} ${n} must be an integer in [1, 65535]`);let n=e.stepCount??20;if(!Number.isInteger(n)||n<0||n>pe)throw Error(`${t} stepCount must be an integer in [0, ${pe}]`);if(o(e.velocities,[`float32x2`],`${t} velocities`),e.velocities.length!==e.fieldWidth*e.fieldHeight)throw Error(`${t} velocities must have fieldWidth * fieldHeight rows`);if(o(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<12)throw Error(`${t} parameters must contain 12 float32 rows`);if(f(e.uint32Parameters,`${t} uint32Parameters`),e.uint32Parameters.length<4)throw Error(`${t} uint32Parameters must contain 4 uint32 rows`);let r=y(t,e.grid),{output:i}=e;if(o(i.values,[`float32`],`${t} output.values`),i.values.length!==r)throw Error(`${t} output.values must have grid width * height rows`);if(i.speeds&&(o(i.speeds,[`float32`],`${t} output.speeds`),i.speeds.length!==r))throw Error(`${t} output.speeds must have grid width * height rows`);if(i.texture&&(i.texture.format!==`r32float`||i.texture.width!==e.grid.width||i.texture.height!==e.grid.height))throw Error(`${t} output.texture must be r32float with width x height`);u(t,[i.values,i.speeds],[e.velocities,e.parameters,e.uint32Parameters]),this.id=t,this.stepCount=n,this.props=e}getCommandNodes(e){let{id:t,props:n,stepCount:r}=this,{output:i}=n,{width:a,height:o}=n.grid;if(c(t,e,[n.velocities,n.parameters,n.uint32Parameters,i.values,i.speeds]),i.texture&&i.texture.texture.graph!==e)throw Error(`${t} views must belong to the target graph`);let s=a*o,l=h(e,`${t}-noise`,`float32`,s),u=[v(e,{id:`${t}-noise`,operation:M,variant:`noise`,bindings:[{name:`words`,view:n.uint32Parameters,type:`u32`,access:`read`},{name:`noise`,view:l,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`${te}
const WIDTH: u32 = ${a}u;`,body:`let seed = words[wordsOffset];
  let counter = vec4<u32>(index % WIDTH, index / WIDTH, 0u, 0u);
  noise[noiseOffset + index] = philoxUnitFloat(
    philox4x32(counter, vec2<u32>(seed, ${j}u)).x
  );`})],f=[{name:`velocities`,view:n.velocities,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`noise`,view:l,type:`f32`,access:`read`},{name:`values`,view:i.values,type:`f32`,access:`read_write`}];if(i.speeds&&f.push({name:`speeds`,view:i.speeds,type:`f32`,access:`read_write`}),u.push(v(e,{id:`${t}-convolve`,operation:M,variant:`convolve`,bindings:f,invocationCount:s,declarations:`${C(`velocities`,n.fieldWidth,n.fieldHeight)}
const WIDTH: u32 = ${a}u;
const HEIGHT: u32 = ${o}u;
const STEP_COUNT: u32 = ${r}u;
const PI: f32 = 3.14159265358979;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getWeight(signedStep: f32, phase: f32, period: f32) -> f32 {
  let window = 0.5 * (1.0 + cos(PI * signedStep / f32(STEP_COUNT + 1u)));
  var ripple = 1.0;
  if (period > 0.0) {
    ripple = 0.5 * (1.0 + cos(2.0 * PI * (signedStep / period - phase)));
  }
  return window * ripple;
}
// Unit flow direction times sign, or (0, 0, 0) where the trace must stop.
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}`,body:`
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let origin = vec2<f32>(readParameter(4u), readParameter(5u));
  let cell = vec2<f32>(readParameter(6u), readParameter(7u));
  let stepVector = readParameter(8u) * cell;
  let minimumSpeed = readParameter(9u);
  let phase = readParameter(10u);
  let period = readParameter(11u);
  let column = index % WIDTH;
  let row = index / WIDTH;
  let centre = origin + (vec2<f32>(f32(column), f32(row)) + vec2<f32>(0.5)) * cell;
  let centreSample = sampleField(centre, fieldExtent);
  if (centreSample.z == 0.0) {
    var nanBits = 0x7fc00000u;
    values[valuesOffset + index] = bitcast<f32>(nanBits);
    ${i.speeds?`speeds[speedsOffset + index] = bitcast<f32>(nanBits);`:``}
    return;
  }
  ${i.speeds?`speeds[speedsOffset + index] = length(centreSample.xy);`:``}
  var weightSum = getWeight(0.0, phase, period);
  var sum = weightSum * noise[noiseOffset + index];
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(1.0, -1.0, direction == 1u);
    var position = centre;
    for (var stepIndex = 1u; stepIndex <= STEP_COUNT; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * stepVector * first.xy, fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      position = position + stepVector * middle.xy;
      let pixel = floor((position - origin) / cell);
      if (pixel.x < 0.0 || pixel.y < 0.0 || pixel.x >= f32(WIDTH) || pixel.y >= f32(HEIGHT)) {
        break;
      }
      let weight = getWeight(directionSign * f32(stepIndex), phase, period);
      sum = sum + weight * noise[noiseOffset + u32(pixel.y) * WIDTH + u32(pixel.x)];
      weightSum = weightSum + weight;
    }
  }
  values[valuesOffset + index] = select(noise[noiseOffset + index], sum / weightSum, weightSum > 0.0);`})),i.texture){let n=i.texture;u.push(...d(e,()=>new S({id:`${t}-texture`,input:{id:`${t}-texture-band`,format:`float32`,storage:{kind:`buffer`,values:i.values}},output:n}).addToGraph(e)))}return u}},he=b.streamlinesJitter.id;function ge(e,t=new Float32Array(12)){if(t.length<12)throw Error(`Streamlines parameter target must hold 12 elements`);let[,,n,r]=e.gridExtent;if(!(n>0)||!(r>0))throw Error(`Streamlines grid cells must have a positive size`);return t.set([...e.fieldExtent,...e.gridExtent,1/Math.fround(n),1/Math.fround(r),e.stepLength,e.minimumSpeed??0]),t}function _e(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`Streamlines uint32 parameter target must hold 4 elements`);let n=e.seed??0,r=e.minimumPoints??2;for(let[e,t]of[[`seed`,n],[`minimumPoints`,r]])if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`Streamlines ${e} must be a uint32 integer`);return t.set([n,r,0,0]),t}var N=`GPUStreamlines`,P=65535,F=0,ve=1,I=2,ye=class{id;seedCount;maximumIterations;props;constructor(e){let t=e.id??`streamlines`;for(let[n,r]of[[`fieldWidth`,e.fieldWidth],[`fieldHeight`,e.fieldHeight],[`gridWidth`,e.gridWidth],[`gridHeight`,e.gridHeight],[`seedColumns`,e.seedColumns],[`seedRows`,e.seedRows]])if(!Number.isInteger(r)||r<1||r>65535)throw Error(`${t} ${n} must be an integer in [1, 65535]`);let n=e.seedColumns*e.seedRows;if(n>P)throw Error(`${t} seedColumns * seedRows must be at most ${P}`);if(!Number.isInteger(e.stepsPerDirection)||e.stepsPerDirection<1)throw Error(`${t} stepsPerDirection must be a positive integer`);let r=e.maximumIterations??32;if(!Number.isInteger(r)||r<1)throw Error(`${t} maximumIterations must be a positive integer`);if(o(e.velocities,[`float32x2`],`${t} velocities`),e.velocities.length!==e.fieldWidth*e.fieldHeight)throw Error(`${t} velocities must have fieldWidth * fieldHeight rows`);if(o(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<12)throw Error(`${t} parameters must contain 12 float32 rows`);if(f(e.uint32Parameters,`${t} uint32Parameters`),e.uint32Parameters.length<4)throw Error(`${t} uint32Parameters must contain 4 uint32 rows`);let{output:i}=e;if(l(t,i.lines),i.lines.ids.length<1)throw Error(`${t} output.lines.ids must hold at least one line`);if(f(i.lineOffsets,`${t} output.lineOffsets`),i.lineOffsets.length!==i.lines.ids.length+1)throw Error(`${t} output.lineOffsets must have lines.ids.length + 1 rows`);if(o(i.positions,[`float32x2`],`${t} output.positions`),i.positions.length<1)throw Error(`${t} output.positions must hold at least one point`);for(let[e,n]of[[`positionCount`,i.positionCount],[`status.iterationLimitReached`,i.status?.iterationLimitReached]])if(n&&(f(n,`${t} output.${e}`),n.length<1))throw Error(`${t} output.${e} must contain one uint32 row`);if(i.candidates&&(o(i.candidates.positions,[`float32x2`],`${t} candidates.positions`),f(i.candidates.spans,`${t} candidates.spans`),i.candidates.positions.length!==n*(2*e.stepsPerDirection+1)||i.candidates.spans.length!==n*4))throw Error(`${t} candidates must have seedCount * (2L + 1) points and seedCount * 4 spans`);u(t,[i.lines.ids,i.lines.status.count,i.lines.status.overflow,i.lines.status.requiredCount,i.lineOffsets,i.positions,i.positionCount,i.status?.iterationLimitReached,i.candidates?.positions,i.candidates?.spans],[e.velocities,e.parameters,e.uint32Parameters]),this.id=t,this.seedCount=n,this.maximumIterations=r,this.props=e}getCommandNodes(e){let{id:t,props:n,seedCount:r,maximumIterations:i}=this,{output:a}=n;c(t,e,[n.velocities,n.parameters,n.uint32Parameters,a.lines.ids,a.lines.status.count,a.lines.status.overflow,a.lines.status.requiredCount,a.lineOffsets,a.positions,a.positionCount,a.status?.iterationLimitReached,a.candidates?.positions,a.candidates?.spans]);let o=n.stepsPerDirection,s=2*o+1,l=n.gridWidth*n.gridHeight,u=a.lines.ids.length,d=a.positions.length,f=a.candidates?.positions??h(e,`${t}-candidates`,`float32x2`,r*s),p=a.candidates?.spans??h(e,`${t}-spans`,`uint32`,r*4),m=h(e,`${t}-keys`,`uint32`,r),_=h(e,`${t}-status`,`uint32`,r),y=h(e,`${t}-undecided`,`uint32`,l),b=h(e,`${t}-accepted`,`uint32`,l),S=h(e,`${t}-point-counts`,`uint32`,r),oe=h(e,`${t}-point-offsets`,`uint32`,r),w=h(e,`${t}-line-flags`,`uint32`,r),T=h(e,`${t}-line-ranks`,`uint32`,r),E=h(e,`${t}-state`,`uint32`,2),D=`
const GRID_WIDTH: u32 = ${n.gridWidth}u;
const GRID_HEIGHT: u32 = ${n.gridHeight}u;
const STEPS: u32 = ${o}u;
const STRIDE: u32 = ${s}u;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getGridLocal(position: vec2<f32>) -> vec2<f32> {
  return (position - vec2<f32>(readParameter(4u), readParameter(5u))) *
    vec2<f32>(readParameter(8u), readParameter(9u));
}
fn isInsideGrid(local: vec2<f32>) -> bool {
  return local.x >= 0.0 && local.y >= 0.0 && local.x < f32(GRID_WIDTH) && local.y < f32(GRID_HEIGHT);
}
fn getGridCell(position: vec2<f32>) -> u32 {
  let local = getGridLocal(position);
  let column = min(u32(max(local.x, 0.0)), GRID_WIDTH - 1u);
  let row = min(u32(max(local.y, 0.0)), GRID_HEIGHT - 1u);
  return row * GRID_WIDTH + column;
}
fn readCandidate(slot: u32) -> vec2<f32> {
  return vec2<f32>(candidates[candidatesOffset + 2u * slot], candidates[candidatesOffset + 2u * slot + 1u]);
}`,O=[];O.push(v(e,{id:`${t}-trace`,operation:N,variant:`trace`,bindings:[{name:`velocities`,view:n.velocities,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`words`,view:n.uint32Parameters,type:`u32`,access:`read`},{name:`candidates`,view:f,type:`f32`,access:`read_write`},{name:`spans`,view:p,type:`u32`,access:`read_write`},{name:`keys`,view:m,type:`u32`,access:`read_write`},{name:`status`,view:_,type:`u32`,access:`read_write`}],invocationCount:r,declarations:`${te}
${C(`velocities`,n.fieldWidth,n.fieldHeight)}
${D}
const SEED_COLUMNS: u32 = ${n.seedColumns}u;
const SEED_ROWS: u32 = ${n.seedRows}u;
fn writeCandidate(slot: u32, position: vec2<f32>) {
  candidates[candidatesOffset + 2u * slot] = position.x;
  candidates[candidatesOffset + 2u * slot + 1u] = position.y;
}
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}`,body:`
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let gridOrigin = vec2<f32>(readParameter(4u), readParameter(5u));
  let seedCell = vec2<f32>(readParameter(6u), readParameter(7u)) *
    vec2<f32>(f32(GRID_WIDTH), f32(GRID_HEIGHT)) / vec2<f32>(f32(SEED_COLUMNS), f32(SEED_ROWS));
  let stepLength = readParameter(10u);
  let minimumSpeed = readParameter(11u);
  let seed = words[wordsOffset];
  let random = philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${he}u));
  keys[keysOffset + index] = ((random.z >> 16u) << 16u) | (0xffffu - index);
  let lattice = vec2<f32>(f32(index % SEED_COLUMNS), f32(index / SEED_COLUMNS));
  let start = gridOrigin + (lattice + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y))) * seedCell;
  let base = index * STRIDE + STEPS;
  var nanBits = 0x7fc00000u;
  let nanPoint = vec2<f32>(bitcast<f32>(nanBits));
  for (var slot = 0u; slot < STRIDE; slot = slot + 1u) {
    writeCandidate(index * STRIDE + slot, nanPoint);
  }
  spans[spansOffset + 4u * index] = 0u;
  spans[spansOffset + 4u * index + 1u] = 0u;
  spans[spansOffset + 4u * index + 2u] = 0u;
  spans[spansOffset + 4u * index + 3u] = 0u;
  if (getDirection(start, fieldExtent, 1.0, minimumSpeed).z == 0.0 || !isInsideGrid(getGridLocal(start))) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  status[statusOffset + index] = ${F}u;
  writeCandidate(base, start);
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(-1.0, 1.0, direction == 1u);
    var position = start;
    var steps = 0u;
    for (var stepIndex = 1u; stepIndex <= STEPS; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * stepLength * first.xy, fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      let next = position + stepLength * middle.xy;
      if (!isInsideGrid(getGridLocal(next))) {
        break;
      }
      position = next;
      steps = stepIndex;
      if (direction == 1u) {
        writeCandidate(base + stepIndex, position);
      } else {
        writeCandidate(base - stepIndex, position);
      }
    }
    spans[spansOffset + 4u * index + direction] = steps;
  }`})),O.push(ee(e,{id:`${t}-clear-accepted`,operation:N,view:b,type:`u32`,value:`0u`}));let k=ie(e,`${t}-rounds`,N,r,t);O.push(x(e,{id:`${t}-rounds-reset`,operation:N,state:k}));let se=ae(e,{id:`${t}-round-gate`,operation:N,contributorId:t,groupSize:1});O.push(se.writeNode);let ce=Math.ceil(r/256)*256;for(let a=0;a<i;a++){let o=re(k,`${t}-round-${a}-claim`),s=re(k,`${t}-round-${a}-decide`),c=re(k,`${t}-round-${a}-clear`);O.push(v(e,{id:`${t}-round-${a}-clear`,operation:N,variant:`clear-grid`,bindings:[{name:`undecidedMaximum`,view:y,type:`u32`,access:`read_write`}],invocationCount:r,guardIndex:!1,condition:c.condition,extraResources:c.extraResources,body:`for (var cell = index; cell < ${l}u; cell = cell + ${ce}u) {
    undecidedMaximum[undecidedMaximumOffset + cell] = 0u;
  }`}),v(e,{id:`${t}-round-${a}-claim`,operation:N,variant:`claim`,condition:o.condition,extraResources:o.extraResources,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`words`,view:n.uint32Parameters,type:`u32`,access:`read`},{name:`candidates`,view:f,type:`f32`,access:`read`},{name:`spans`,view:p,type:`u32`,access:`read_write`},{name:`keys`,view:m,type:`u32`,access:`read`},{name:`status`,view:_,type:`u32`,access:`read_write`},{name:`accepted`,view:b,type:`u32`,access:`read`},{name:`undecidedMaximum`,view:y,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:D,body:`
  if (status[statusOffset + index] != ${F}u) {
    return;
  }
  // Occupancy only grows, so the line can keep at most the points before its first occupied
  // cell in each direction. Reject now when that cannot reach minimumPoints.
  let base = index * STRIDE + STEPS;
  if (accepted[acceptedOffset + getGridCell(readCandidate(base))] != 0u) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  let backward = spans[spansOffset + 4u * index];
  let forward = spans[spansOffset + 4u * index + 1u];
  var keptBackward = 0u;
  while (keptBackward < backward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base - keptBackward - 1u))] == 0u) {
    keptBackward = keptBackward + 1u;
  }
  var keptForward = 0u;
  while (keptForward < forward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base + keptForward + 1u))] == 0u) {
    keptForward = keptForward + 1u;
  }
  if (1u + keptBackward + keptForward < words[wordsOffset + 1u]) {
    status[statusOffset + index] = ${I}u;
    return;
  }
  spans[spansOffset + 4u * index + 2u] = keptBackward;
  spans[spansOffset + 4u * index + 3u] = keptForward;
  let key = keys[keysOffset + index];
  for (var slot = base - keptBackward; slot <= base + keptForward; slot = slot + 1u) {
    atomicMax(&undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))], key);
  }`}),v(e,{id:`${t}-round-${a}-decide`,operation:N,variant:`decide`,condition:s.condition,extraResources:s.extraResources,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`candidates`,view:f,type:`f32`,access:`read`},{name:`spans`,view:p,type:`u32`,access:`read`},{name:`keys`,view:m,type:`u32`,access:`read`},{name:`status`,view:_,type:`u32`,access:`read_write`},{name:`undecidedMaximum`,view:y,type:`u32`,access:`read`},{name:`accepted`,view:b,type:`u32`,access:`read_write`},{name:`iterationStatus`,view:k.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:D,body:`
  if (status[statusOffset + index] != ${F}u) {
    return;
  }
  let key = keys[keysOffset + index];
  let base = index * STRIDE + STEPS;
  let first = base - spans[spansOffset + 4u * index + 2u];
  let last = base + spans[spansOffset + 4u * index + 3u];
  for (var slot = first; slot <= last; slot = slot + 1u) {
    if (undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))] != key) {
      return;
    }
  }
  // Ready: no undecided line with a higher key can still occupy these cells, every decided one
  // already has, and no line decided in this dispatch touches them, so the claimed span is the
  // greedy trim.
  status[statusOffset + index] = ${ve}u;
  atomicOr(&iterationStatus[iterationStatusOffset], 1u);
  for (var slot = first; slot <= last; slot = slot + 1u) {
    accepted[acceptedOffset + getGridCell(readCandidate(slot))] = index + 1u;
  }`})),O.push(ne(e,{id:`${t}-round-gate-${a}`,operation:N,state:k,maxIterations:i,groupSize:se.getSlot(1)}))}O.push(ee(e,{id:`${t}-clear-state`,operation:N,view:E,type:`u32`,value:`0u`}),v(e,{id:`${t}-counts`,operation:N,variant:`counts`,bindings:[{name:`spans`,view:p,type:`u32`,access:`read`},{name:`status`,view:_,type:`u32`,access:`read`},{name:`pointCounts`,view:S,type:`u32`,access:`read_write`},{name:`lineFlags`,view:w,type:`u32`,access:`read_write`},{name:`state`,view:E,type:`u32`,access:`read_write`}],invocationCount:r,body:`
  let lineStatus = status[statusOffset + index];
  let isAccepted = lineStatus == ${ve}u;
  pointCounts[pointCountsOffset + index] = select(
    0u,
    1u + spans[spansOffset + 4u * index + 2u] + spans[spansOffset + 4u * index + 3u],
    isAccepted
  );
  lineFlags[lineFlagsOffset + index] = select(0u, 1u, isAccepted);
  if (lineStatus == ${F}u) {
    state[stateOffset + 1u] = 1u;
  }`}),...new g({id:`${t}-point-scan`,input:S,output:oe,mode:`exclusive`}).getCommandNodes(e),...new g({id:`${t}-line-scan`,input:w,output:T,mode:`inclusive`}).getCommandNodes(e));let le=`
const LINE_CAPACITY: u32 = ${u}u;
const POINT_CAPACITY: u32 = ${d}u;
fn getFittingRank(index: u32) -> u32 {
  if (lineFlags[lineFlagsOffset + index] == 0u) {
    return 0xffffffffu;
  }
  let rank = lineRanks[lineRanksOffset + index] - 1u;
  let end = pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  if (rank >= LINE_CAPACITY || end > POINT_CAPACITY) {
    return 0xffffffffu;
  }
  return rank;
}`;O.push(v(e,{id:`${t}-emit-points`,operation:N,variant:`emit-points`,bindings:[{name:`candidates`,view:f,type:`f32`,access:`read`},{name:`spans`,view:p,type:`u32`,access:`read`},{name:`lineFlags`,view:w,type:`u32`,access:`read`},{name:`lineRanks`,view:T,type:`u32`,access:`read`},{name:`pointCounts`,view:S,type:`u32`,access:`read`},{name:`pointOffsets`,view:oe,type:`u32`,access:`read`},{name:`points`,view:a.positions,type:`f32`,access:`read_write`}],invocationCount:r,declarations:`const STEPS: u32 = ${o}u;
const STRIDE: u32 = ${s}u;
${le}`,body:`
  if (getFittingRank(index) == 0xffffffffu) {
    return;
  }
  let first = index * STRIDE + STEPS - spans[spansOffset + 4u * index + 2u];
  let offset = pointOffsets[pointOffsetsOffset + index];
  let count = pointCounts[pointCountsOffset + index];
  for (var point = 0u; point < count; point = point + 1u) {
    points[pointsOffset + 2u * (offset + point)] = candidates[candidatesOffset + 2u * (first + point)];
    points[pointsOffset + 2u * (offset + point) + 1u] = candidates[candidatesOffset + 2u * (first + point) + 1u];
  }`}),v(e,{id:`${t}-emit-lines`,operation:N,variant:`emit-lines`,bindings:[{name:`lineFlags`,view:w,type:`u32`,access:`read`},{name:`lineRanks`,view:T,type:`u32`,access:`read`},{name:`pointCounts`,view:S,type:`u32`,access:`read`},{name:`pointOffsets`,view:oe,type:`u32`,access:`read`},{name:`ids`,view:a.lines.ids,type:`u32`,access:`read_write`},{name:`pathOffsets`,view:a.lineOffsets,type:`u32`,access:`read_write`},{name:`state`,view:E,type:`atomic<u32>`,access:`read_write`}],invocationCount:r,declarations:le,body:`
  let rank = getFittingRank(index);
  if (rank == 0xffffffffu) {
    return;
  }
  ids[idsOffset + rank] = index;
  pathOffsets[pathOffsetsOffset + rank + 1u] =
    pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  atomicMax(&state[stateOffset], rank + 1u);`}));let A=[{name:`lineRanks`,view:T,type:`u32`,access:`read`},{name:`state`,view:E,type:`u32`,access:`read`},{name:`pathOffsets`,view:a.lineOffsets,type:`u32`,access:`read_write`},{name:`countOut`,view:a.lines.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:a.lines.status.overflow,type:`u32`,access:`read_write`},{name:`pointCountOut`,view:a.positionCount,type:`u32`,access:`read_write`}],ue=[...a.lines.status.requiredCount?[{name:`totalOut`,view:a.lines.status.requiredCount,type:`u32`,access:`read_write`}]:[],...a.status?.iterationLimitReached?[{name:`unconvergedOut`,view:a.status?.iterationLimitReached,type:`u32`,access:`read_write`}]:[]];return O.push(v(e,{id:`${t}-finalize`,operation:N,variant:`finalize`,bindings:[...A,...ue],invocationCount:1,body:`
  let total = lineRanks[lineRanksOffset + ${r-1}u];
  let count = state[stateOffset];
  pathOffsets[pathOffsetsOffset] = 0u;
  countOut[countOutOffset] = count;
  overflowOut[overflowOutOffset] = select(0u, 1u, count < total) | select(0u, 2u, state[stateOffset + 1u] != 0u);
  pointCountOut[pointCountOutOffset] = select(0u, pathOffsets[pathOffsetsOffset + count], count > 0u);
  ${a.lines.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}
  ${a.status?.iterationLimitReached?`unconvergedOut[unconvergedOutOffset] = state[stateOffset + 1u];`:``}`})),O}},L={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},R=`
struct FieldStyle {
  ramp: array<vec4<f32>, 4>,
  fieldOrigin: vec2<f32>,
  fieldCell: vec2<f32>,
  fieldSize: vec2<u32>,
  speedRange: vec2<f32>,
};

fn projectWindPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteWind(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}

fn getRampColor(ramp: array<vec4<f32>, 4>, t: f32) -> vec3<f32> {
  var stops = ramp;
  let scaled = clamp(t, 0.0, 1.0) * 3.0;
  let index = min(u32(floor(scaled)), 2u);
  return mix(stops[index].rgb, stops[index + 1u].rgb, scaled - f32(index));
}
`,z=`
@group(0) @binding(auto) var<storage, read> speedField: array<f32>;

fn readSpeedCell(style: FieldStyle, column: i32, row: i32) -> f32 {
  let columns = i32(style.fieldSize.x);
  let rows = i32(style.fieldSize.y);
  return speedField[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

// Bilinear speed in meters per second at a longitude and latitude.
fn sampleWindSpeed(style: FieldStyle, lonLat: vec2<f32>) -> f32 {
  let grid = (lonLat - style.fieldOrigin) / style.fieldCell - vec2<f32>(0.5);
  let base = vec2<i32>(floor(grid));
  let fraction = grid - floor(grid);
  let a = readSpeedCell(style, base.x, base.y);
  let b = readSpeedCell(style, base.x + 1, base.y);
  let c = readSpeedCell(style, base.x, base.y + 1);
  let d = readSpeedCell(style, base.x + 1, base.y + 1);
  return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}
`,B=96;function V(e,t,n){n.ramp.forEach(([t,n,r],i)=>{e.set([t/255,n/255,r/255,1],i*4)}),e.set(n.field.extent.slice(0,2),16),e.set(n.field.extent.slice(2,4),18),t.set(n.field.size,20),e.set(n.speedRange,22)}var H=class extends n{static layerName=`WindModelLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:L};getAttributeManager(){return null}getVertexCount(){return 6}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:this.getStyleByteLength(),usage:r.UNIFORM|r.COPY_DST}),i=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST});this.setState({model:null,styleBuffer:n,placeholderBuffer:i});let o=new a(t,{...this.getShaders({modules:[e],source:this.getShaderSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.getVertexCount(),instanceCount:0,bufferLayout:[],bindings:{...this.getStorageBindings(i),layerStyle:n},parameters:L});this.setState({model:o})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t&&n&&r&&t.setBindings({...this.getStorageBindings(r),layerStyle:n})}getModels(){let e=this.state.model;return e?[e]:[]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;this.writeStyle(n);let{drawCommands:r,drawCommandIndex:i=0}=this.props;r?(t.setInstanceCount(0),t.draw(e),r.draw(e,i)):(t.setInstanceCount(this.getInstanceCount()),t.draw(e))}finalizeState(e){let t=this.state;t.model?.destroy(),t.styleBuffer?.destroy(),t.placeholderBuffer?.destroy(),this.setState({model:null,styleBuffer:null,placeholderBuffer:null}),super.finalizeState(e)}},U=`
var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);

// Expands the segment (startClip, endClip) into a screen-space quad corner; x is t along the
// segment, y the side across it.
fn expandSegment(startClip: vec4<f32>, endClip: vec4<f32>, corner: vec2<f32>, widthPixels: f32) -> vec4<f32> {
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  return vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
}
`,W=96,G=`
${R}
struct LayerStyle {
  field: FieldStyle,
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  gridSize: vec2<u32>,
  opacity: f32,
  contrast: f32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> licValues: array<f32>;
${z}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
);

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let strip = vertexIndex / 6u;
  let corner = QUAD_CORNERS[vertexIndex % 6u];
  let local = vec2<f32>(corner.x, (f32(strip) + corner.y) / ${W}.0);
  var output: VertexOutput;
  output.position = projectWindPosition(mix(layerStyle.minimum, layerStyle.maximum, local));
  output.local = local;
  return output;
}

fn readCell(column: i32, row: i32) -> f32 {
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  return licValues[u32(clamp(row, 0, rows - 1) * columns + clamp(column, 0, columns - 1))];
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Row 0 is the south edge, so the row index grows with latitude. Bilinear between centers.
  let grid = vec2<f32>(layerStyle.gridSize);
  let position = input.local * grid - vec2<f32>(0.5);
  let base = vec2<i32>(floor(position));
  let fraction = position - floor(position);
  let a = readCell(base.x, base.y);
  let b = readCell(base.x + 1, base.y);
  let c = readCell(base.x, base.y + 1);
  let d = readCell(base.x + 1, base.y + 1);
  if (isNonFiniteWind(a) || isNonFiniteWind(b) || isNonFiniteWind(c) || isNonFiniteWind(d)) {
    discard;
  }
  let value = mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
  let stretched = clamp((value - 0.5) * layerStyle.contrast + 0.5, 0.0, 1.0);
  let lonLat = mix(layerStyle.minimum, layerStyle.maximum, input.local);
  let speed = sampleWindSpeed(layerStyle.field, lonLat);
  let speedRatio = (speed - layerStyle.field.speedRange.x) /
    max(layerStyle.field.speedRange.y - layerStyle.field.speedRange.x, 1e-20);
  let alpha = layerStyle.opacity * stretched * stretched;
  return vec4<f32>(getRampColor(layerStyle.field.ramp, speedRatio), alpha);
}
`,be=class extends H{static layerName=`WindTextureLayer`;getShaderSource(){return G}getStyleByteLength(){return B+32}getInstanceCount(){return 1}getVertexCount(){return 6*W}getStorageBindings(e){return{licValues:this.props.values??e,speedField:this.props.speedField}}writeStyle(e){let{bounds:t,gridSize:n,opacity:r=.9,contrast:i=3}=this.props,a=new ArrayBuffer(B+32),o=new Float32Array(a),s=new Uint32Array(a);V(o,s,this.props);let c=B/4;o.set([t[0],t[1],t[2],t[3]],c),s.set(n,c+4),o[c+6]=r,o[c+7]=i,e.write(new Uint8Array(a))}},xe=`
${R}
struct LayerStyle {
  field: FieldStyle,
  widthPixels: f32,
  opacity: f32,
  ringLength: u32,
  particleCount: u32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailWords: array<u32>;
${z}
${U}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn hiddenOutput() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let segmentsPerParticle = layerStyle.ringLength - 1u;
  let particle = instanceIndex / segmentsPerParticle;
  let segment = instanceIndex % segmentsPerParticle;
  if (particle >= layerStyle.particleCount) {
    return hiddenOutput();
  }
  // Frame f writes slot f % L, so the oldest slot is (f + 1) % L; segment k joins the k-th and
  // (k + 1)-th slots in age order.
  let frame = trailWords[1];
  let startSlot = (frame + 1u + segment) % layerStyle.ringLength;
  let endSlot = (frame + 2u + segment) % layerStyle.ringLength;
  let start = trailPositions[particle * layerStyle.ringLength + startSlot];
  let end = trailPositions[particle * layerStyle.ringLength + endSlot];
  if (isNonFiniteWind(start.x) || isNonFiniteWind(end.x) ||
      isNonFiniteWind(start.y) || isNonFiniteWind(end.y)) {
    return hiddenOutput();
  }
  let startClip = projectWindPosition(start);
  let endClip = projectWindPosition(end);
  // A respawn fills the ring with one position: skip zero-length segments.
  let pixelLength = length((endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize);
  if (pixelLength < 0.02) {
    return hiddenOutput();
  }
  let age = f32(segment + 1u) / f32(segmentsPerParticle);
  let speed = sampleWindSpeed(layerStyle.field, end);
  let speedRatio = (speed - layerStyle.field.speedRange.x) /
    max(layerStyle.field.speedRange.y - layerStyle.field.speedRange.x, 1e-20);
  var output: VertexOutput;
  output.position = expandSegment(startClip, endClip, SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels);
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  output.color = vec4<f32>(getRampColor(layerStyle.field.ramp, speedRatio), age * age * layerStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`,Se=class extends H{static layerName=`WindTrailLayer`;getShaderSource(){return xe}getStyleByteLength(){return B+16}getInstanceCount(){return this.props.particleCount*(this.props.ringLength-1)}getStorageBindings(){return{trailPositions:this.props.trailPositions,trailWords:this.props.wordParameters,speedField:this.props.speedField}}writeStyle(e){let{widthPixels:t=1.4,opacity:n=.9,ringLength:r,particleCount:i}=this.props,a=new ArrayBuffer(B+16),o=new Float32Array(a),s=new Uint32Array(a);V(o,s,this.props);let c=B/4;o[c]=t,o[c+1]=n,s[c+2]=r,s[c+3]=i,e.write(new Uint8Array(a))}},Ce=`
${R}
struct LayerStyle {
  field: FieldStyle,
  color: vec4<f32>,
  widthPixels: f32,
  opacity: f32,
  padding: vec2<u32>,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> pathPoints: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathLineCount: array<u32>;
@group(0) @binding(auto) var<storage, read> pathPointCount: array<u32>;
${U}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  let next = instanceIndex + 1u;
  if (next >= pathPointCount[0]) {
    return output;
  }
  // Instance i joins points i and i + 1 unless i + 1 starts a line: binary search the line of
  // point i + 1 in the CSR offsets.
  var low = 0u;
  var high = pathLineCount[0];
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= next) {
      low = middle;
    } else {
      high = middle;
    }
  }
  if (pathOffsets[low] == next) {
    return output;
  }
  let start = pathPoints[instanceIndex];
  let end = pathPoints[next];
  if (isNonFiniteWind(start.x) || isNonFiniteWind(end.x)) {
    return output;
  }
  output.position = expandSegment(
    projectWindPosition(start), projectWindPosition(end),
    SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels
  );
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(layerStyle.color.rgb, layerStyle.color.a * layerStyle.opacity * coverage);
}
`,we=class extends H{static layerName=`WindPathLayer`;getShaderSource(){return Ce}getStyleByteLength(){return B+32}getInstanceCount(){return this.props.pointCapacity}getStorageBindings(){return{pathPoints:this.props.points,pathOffsets:this.props.pathOffsets,pathLineCount:this.props.lineCount,pathPointCount:this.props.pointCount}}writeStyle(e){let{color:t=[255,255,255,255],widthPixels:n=1,opacity:r=1}=this.props,i=new ArrayBuffer(B+32),a=new Float32Array(i);V(a,new Uint32Array(i),this.props);let o=B/4;a.set([t[0]/255,t[1]/255,t[2]/255,(t[3]??255)/255],o),a[o+4]=n,a[o+5]=r,e.write(new Uint8Array(i))}},Te=i,K=181,q=153,J=K*q,Y=[-105.125,11.875,.25,.25],X=896,Z=704,Ee=128,De=112,Oe=192,ke=160,Ae=48,je=16,Me=16384,Ne=262144,Pe=.1,Fe=.18,Ie=Date.UTC(2024,8,26,12),Le=1.2,Q=180/(Math.PI*Te);async function Re(e,t,n){let r=await createImageBitmap(new Blob([e],{type:`image/png`}),{premultiplyAlpha:`none`,colorSpaceConversion:`none`}),i=new OffscreenCanvas(r.width,r.height),a=i.getContext(`2d`,{willReadFrequently:!0});if(!a)throw Error(`2D canvas unavailable for wind decoding`);a.drawImage(r,0,0),r.close();let{width:o,height:s}=i,c=a.getImageData(0,0,o,s).data,l=new Float32Array(o*s*2);for(let e=0;e<s;e++){let r=s-1-e;for(let i=0;i<o;i++){let a=(e*o+i)*4,s=(r*o+i)*2;l[s]=t[0]+c[a]/255*(t[1]-t[0]),l[s+1]=n[0]+c[a+1]/255*(n[1]-n[0])}}return l}async function ze(e,t,n,r,i){return{frames:await Promise.all(e.map(async e=>Re(await D(O(`gfs-wind`,e),i),n,r))),hours:t}}async function Be(e){let t=e.datasets.get(`gfs-wind`),{device:n}=e;e.setStatus(`Decoding 32 GFS wind frames`);let r=t.manifest.properties,[i,a]=await Promise.all([ze(r.wind10m.files,r.wind10m.forecastHours,r.wind10m.uRange,r.wind10m.vRange,e.signal),ze(r.jet250hPa.files,r.jet250hPa.forecastHours,r.jet250hPa.uRange,r.jet250hPa.vRange,e.signal)]);e.signal.throwIfAborted();let o={"10m":i,"250hPa":a},c=new m(n,`wind`),l=new Float32Array(q);for(let e=0;e<q;e++){let t=12+e*.25;l[e]=Q/Math.cos(t*Math.PI/180)}let u=c.createBuffer(`velocity`,J*8),d=c.createBuffer(`speed`,J*4),f=new Float32Array(J*2),h=new Float32Array(J),g=new Float32Array(J*2),ee=c.createParameterBuffer(`particle-parameters`,`float32`,12),v=c.createParameterBuffer(`particle-words`,`uint32`,4),y=c.createParameterBuffer(`lic-parameters`,`float32`,12),te=c.createParameterBuffer(`lic-words`,`uint32`,4),b=c.createParameterBuffer(`stream-parameters`,`float32`,12),ne=c.createParameterBuffer(`stream-words`,`uint32`,4),re=c.createBuffer(`stream-ids`,Me*4),x=c.createBuffer(`stream-count`,4),ie=c.createBuffer(`stream-overflow`,4),ae=c.createBuffer(`stream-offsets`,(Me+1)*4),S=c.createBuffer(`stream-points`,Ne*8),C=c.createBuffer(`stream-point-count`,4),D=c.createBuffer(`stream-unconverged`,4),O=c.track(new _(n,{id:`wind-stream-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),j=new s(n,{id:`wind-streamlines`});{let e=ue(j);j.add(new ye({id:`streamlines`,velocities:e(`velocity`,u,`float32x2`,J),fieldWidth:K,fieldHeight:q,gridWidth:Ee,gridHeight:De,seedColumns:Oe,seedRows:ke,stepsPerDirection:Ae,maximumIterations:je,parameters:b.importToGraph(j),uint32Parameters:ne.importToGraph(j),output:{lines:{ids:e(`ids`,re,`uint32`,Me),status:{count:e(`count`,x,`uint32`,1),overflow:e(`overflow`,ie,`uint32`,1)}},lineOffsets:e(`offsets`,ae,`uint32`,Me+1),positions:e(`points`,S,`float32x2`,Ne),positionCount:e(`point-count`,C,`uint32`,1),status:{iterationLimitReached:e(`unconverged`,D,`uint32`,1)}}}))}let M=c.track(await j.compileAsync()),pe=0,he=async(e,t)=>{let r=new m(n,`wind-particles-${pe++}`),i=r.createBuffer(`positions`,e*8),a=r.createBuffer(`ages`,e*4),o=r.createBuffer(`generations`,e*4),c=r.createBuffer(`speeds`,e*4),l=r.createBuffer(`trails`,e*t*8),d=new s(n,{id:`wind-particles-${e}-${t}`}),f=ue(d);return d.add(new T({id:`particles`,velocities:f(`velocity`,u,`float32x2`,J),fieldWidth:K,fieldHeight:q,parameters:ee.importToGraph(d),uint32Parameters:v.importToGraph(d),state:{positions:f(`positions`,i,`float32x2`,e),ages:f(`ages`,a,`uint32`,e),generations:f(`generations`,o,`uint32`,e)},output:{speeds:f(`speeds`,c,`float32`,e),trails:{positions:f(`trails`,l,`float32x2`,e*t),length:t}}})),{resources:r,compiled:r.track(await d.compileAsync()),positions:i,trails:l,count:e,trailLength:t}},N=async e=>{let t=new m(n,`wind-lic-${pe++}`),r=t.createBuffer(`values`,X*Z*4),i=new s(n,{id:`wind-lic-${e}`}),a=ue(i);return i.add(new me({id:`lic`,velocities:a(`velocity`,u,`float32x2`,J),fieldWidth:K,fieldHeight:q,grid:E({width:X,height:Z}),stepCount:e,parameters:y.importToGraph(i),uint32Parameters:te.importToGraph(i),output:{values:a(`values`,r,`float32`,X*Z)}})),{resources:t,compiled:t.track(await i.compileAsync()),values:r,stepCount:e}},P=await he(Number(e.options.particleCount),Number(e.options.trailLength)),F=await N(Number(e.options.licSteps)),ve=c.track(new se),I=c.track(new se),L=!1,R=e.options.hour,z=NaN,B=``,V=-1/0,H=0,U=!0,W=!0,G=!0,xe=-1/0,Ce=``,Te=!1,Re=e.options.seed,Be=e.options.hour,Ve=``,He=``,Ue=``,$=[-105,12,-60,50],We=e=>{let{frames:t,hours:n}=o[e.level],r=t.length-1,i=Math.min(Math.max(R,n[0]),n[r]),a=n.findIndex(e=>e>=i);a<0&&(a=r);let s=Math.max(0,a-(n[a]>i?1:0)),c=n[a]-n[s],p=c>0?(i-n[s])/c:0,m=t[s],_=t[a];for(let e=0;e<q;e++){let t=l[e];for(let n=0;n<K;n++){let r=e*K+n,i=m[r*2]+(_[r*2]-m[r*2])*p,a=m[r*2+1]+(_[r*2+1]-m[r*2+1])*p;g[r*2]=i,g[r*2+1]=a,f[r*2]=i*t,f[r*2+1]=a*Q,h[r]=Math.hypot(i,a)}}u.write(f),d.write(h),z=R,B=e.level,W=!0,G=!0,Ge(e)},Ge=t=>{let n=`${new Date(Ie+R*36e5).toISOString().slice(0,16).replace(`T`,` `)} UTC (forecast hour ${R.toFixed(1)})`;e.setReadout(`time`,n);let i=0,a=0;for(let e=0;e<J;e++)h[e]>i&&(i=h[e],a=e);let o=a%K,s=Math.floor(a/K),c=-105+o*.25,l=12+s*.25,u=r.wind10m.maxSpeedMs,d=t.level===`10m`&&Math.abs(R-r.wind10m.maxSpeedFrame)<.26;e.setReadout(`peak`,`${A(i,1)} m/s at ${l.toFixed(2)}°N ${Math.abs(c).toFixed(2)}°W${d?` · GFS peak in the file: ${A(u,1)} m/s at ${r.wind10m.maxSpeedLonLat[1]}°N ${Math.abs(r.wind10m.maxSpeedLonLat[0])}°W`:``}`)},Ke=e=>{let[t,n,r,i]=e.getBounds();return{west:t,south:n,east:r,north:i}},qe=(e,t,n)=>{y.write(de({fieldExtent:Y,outputExtent:[t.west,t.south,(t.east-t.west)/X,(t.north-t.south)/Z],stepLength:e.licStep,minimumSpeed:e.licMinimumSpeed*Q,phase:n,period:e.licPeriod})),te.write(fe({seed:e.licSeed})),$[0]=t.west,$[1]=t.south,$[2]=t.east,$[3]=t.north},Je=(e,t)=>{let n=360/(512*2**t.zoom),r=e.spacing*n,i=r*Math.cos(t.latitude*Math.PI/180);b.write(ge({fieldExtent:Y,gridExtent:[t.longitude-Ee*r/2,t.latitude-De*i/2,r,i],stepLength:.4*r,minimumSpeed:e.streamMinimumSpeed*Q})),ne.write(_e({seed:e.streamSeed,minimumPoints:e.streamMinimumPoints}))},Ye=new ce(c,`wind-stream-summary`,[x,C,ie,D].map(e=>({buffer:e,size:4})),t=>{let n=new Uint32Array(t);e.setReadout(`streamlines`,`${p(n[0])} lines · ${p(n[1])} points · ${n[2]?`capacity overflow`:`no overflow`} · ${n[3]?`unconverged (undecided lines dropped)`:`converged`}`)}),Xe=()=>{e.setReadout(`particles`,`${p(P.count)} · trail ring ${P.trailLength} slots · ${(P.count*P.trailLength*8/1048576).toFixed(1)} MB`)};Xe(),e.setReadout(`field`,`${K} × ${q} cells, 0.25° (about 28 km), u and v as degrees per second`),e.setReadout(`licRaster`,`${X} × ${Z} px follows the camera, ${F.stepCount*2} taps`),e.setReadout(`streamlines`,`not encoded yet`),e.setReadout(`timing`,`press the button`),e.setStatus(``);async function Ze(){if(!(Te||L)){Te=!0;try{let t={parameters:void 0,completionBuffer:ie,signal:e.signal},r=await k(n,P.compiled,t),i=await k(n,F.compiled,t),a=await k(n,M,{...t,runs:3});if(L)return;e.setReadout(`timing`,`advection ${r.milliseconds.toFixed(2)} ms · LIC ${i.milliseconds.toFixed(2)} ms · streamlines ${a.milliseconds.toFixed(2)} ms`),G=!0}catch{}finally{Te=!1}}}let Qe=(e,t)=>{let n=(e+105)/.25,r=(t-12)/.25;if(n<0||r<0||n>K-1||r>q-1)return null;let i=Math.min(Math.floor(n),K-2),a=Math.min(Math.floor(r),q-2),o=n-i,s=r-a,c=e=>{let t=(t,n)=>g[(n*K+t)*2+e];return t(i,a)*(1-o)*(1-s)+t(i+1,a)*o*(1-s)+t(i,a+1)*(1-o)*s+t(i+1,a+1)*o*s};return{u:c(0),v:c(1)}},$e=e=>[`N`,`NNE`,`NE`,`ENE`,`E`,`ESE`,`SE`,`SSE`,`S`,`SSW`,`SW`,`WSW`,`W`,`WNW`,`NW`,`NNW`][Math.round(e/22.5)%16];return{getCompiledGraphs:()=>[P.compiled,F.compiled,M],setOption(t,n,r){if(t===`hour`||t===`level`)t===`hour`&&(R=r.hour,Be=r.hour),We(r),U=t===`level`?!0:U;else if(t===`particleCount`||t===`trailLength`){let t=Number(r.particleCount),n=Number(r.trailLength);ve.request({build:()=>he(t,n),commit:t=>{let n=P;P=t,n.resources.destroy(),U=!0,Xe(),e.requestLayers()},discard:e=>e.resources.destroy()})}else if(t===`licSteps`){let t=Number(r.licSteps);I.request({build:()=>N(t),commit:t=>{let n=F;F=t,n.resources.destroy(),W=!0,e.setReadout(`licRaster`,`${X} × ${Z} px follows the camera, ${F.stepCount*2} taps`),e.requestLayers()},discard:e=>e.resources.destroy()})}else t===`seed`&&(U=!0);e.requestLayers()},onAction(e){e===`reset`?U=!0:e===`measure`&&Ze()},encode(t,n){let r=e.options,i=n.viewport,a=i;if(r.hour!==Be&&(Be=r.hour,R=r.hour),r.play){let e=r.level===`10m`?23:21;R=(R+n.deltaSeconds*r.playSpeed)%(e+.001)}if((B!==r.level||Math.abs(R-z)>5e-4&&n.timeSeconds-V>Pe)&&(V=n.timeSeconds,We(r)),r.seed!==Re&&(Re=r.seed,U=!0),r.showParticles){let e=r.zoomAdaptive?2**(5-a.zoom):1,n=Ke(i),o=r.spawnInView?[Math.max(n.west,Y[0]),Math.max(n.south,Y[1]),Math.min(n.east,Y[0]+K*.25),Math.min(n.north,Y[1]+q*.25)]:void 0;ee.write(oe({fieldExtent:Y,timeStep:120*e,speedScale:r.speedScale,dropRate:r.dropRate,minimumSpeed:r.minimumSpeed*Q*r.speedScale,spawnBounds:o},[K,q])),v.write(w({seed:r.seed,frame:H,maximumAge:r.maximumAge,reset:U})),P.compiled.encode(t,{parameters:void 0}),H++,U=!1}if(r.showLic){let e=Ke(i),a=`${r.licStep}|${r.licMinimumSpeed}|${r.licPeriod}|${r.licSeed}`,o=`${e.west.toFixed(4)},${e.south.toFixed(4)},${e.east.toFixed(4)},${e.north.toFixed(4)}`;(r.animateLic||W||a!==Ve||o!==Ce)&&(Ve=a,qe(r,e,r.animateLic?-n.timeSeconds*Le:0),F.compiled.encode(t,{parameters:void 0}),W=!1),Ce=o}if(r.showStreamlines){let e=`${r.spacing}|${r.streamMinimumPoints}|${r.streamSeed}|${r.streamMinimumSpeed}`,i=`${a.longitude.toFixed(3)},${a.latitude.toFixed(3)},${a.zoom.toFixed(2)}`;e!==He&&(He=e,G=!0),(G||i!==Ue)&&n.timeSeconds-xe>Fe&&(Ue=i,xe=n.timeSeconds,Je(r,a),M.encode(t,{parameters:void 0}),t.copyBufferToBuffer({sourceBuffer:C,sourceOffset:0,destinationBuffer:O.buffer,destinationOffset:4,size:4}),G=!1,Ye.markStale())}Ye.flush(t)},getLayers(){let t=e.options,n=[],r={speedField:d,field:{extent:Y,size:[K,q]},ramp:le,speedRange:[0,t.speedMax]},i=e.theme()===`dark`;return t.showLic&&n.push(new be({...r,id:`wind-lic`,values:F.values,gridSize:[X,Z],bounds:$,contrast:t.licContrast,opacity:t.licOpacity})),t.showStreamlines&&n.push(new we({...r,id:`wind-streamlines`,points:S,pathOffsets:ae,lineCount:x,pointCount:C,pointCapacity:Ne,drawCommands:O,color:i?[255,255,255,255]:[20,30,60,255],widthPixels:t.streamWidth,opacity:.6})),t.showParticles&&n.push(new Se({...r,id:`wind-trails`,trailPositions:P.trails,wordParameters:v.buffer,ringLength:P.trailLength,particleCount:P.count,widthPixels:t.trailWidth,opacity:t.particleOpacity})),n},getTooltip(t){if(!t.coordinate)return null;let n=Qe(t.coordinate[0],t.coordinate[1]);if(!n)return null;let r=Math.hypot(n.u,n.v),i=(Math.atan2(-n.u,-n.v)*180/Math.PI+360)%360;return`${r.toFixed(1)} m/s (${(r*1.944).toFixed(0)} kn) from the ${$e(i)} (${Math.round(i)}°)\nu ${A(n.u,1)}, v ${A(n.v,1)} m/s · ${e.options.level===`10m`?`10 m above ground`:`250 hPa (about 10.5 km)`}`},onThemeChange(){e.requestLayers()},destroy(){L=!0,Ye.stop(),P.resources.destroy(),F.resources.destroy(),c.destroy()}}}export{Be as createWindFlow};