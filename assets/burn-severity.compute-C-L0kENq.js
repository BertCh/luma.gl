import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as e}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{D as t,N as n,_ as r,d as i,h as a,k as o,n as s,t as c,v as l,w as u,y as d}from"./resources-CwvhEOBe.js";import{u as f}from"./interleave-brZZCO-f.js";import{t as p}from"./gpu-scan-BuGhTLDQ.js";import{t as m}from"./gpu-reduction-BD8l40jM.js";import{l as h,n as g,s as _,t as v}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as y,l as b,t as x}from"./wgsl-common-BNRVzgd8.js";import{g as S,i as C,n as w,p as T,r as E,s as D,u as O}from"./raster-utils-DhdnkkOa.js";import{i as k,s as A}from"./raster-band-utils-CA-sh8xt.js";import{d as j,s as ee}from"./raster-relaxation-BJ40xgro.js";import{a as te,i as M,n as ne,o as N,r as re,s as P}from"./raster-algebra-utils-C7qGzplC.js";import{a as ie,n as ae,o as oe,r as se,s as ce,t as le}from"./gpu-raster-arithmetic-Dk0SdDgu.js";import{n as ue}from"./gpu-raster-grid-BOSvDY5A.js";import{t as de}from"./summary-reader-CsgWxENf.js";import{a as fe,i as pe,r as me}from"./b16-colors-Jr7XN5JQ.js";import{a as F,c as he,i as I,n as ge,o as L,s as _e,t as ve}from"./b16-common-3eXVh1Vw.js";var R=4294967295;function z(e,t){let n=A(e,t.grid);if(d(t.labels,[`uint32`],`${e} labels`),t.labels.length!==n)throw Error(`${e} labels length must equal grid width * height`);if(t.validity&&(d(t.validity,[`uint32`],`${e} validity`),t.validity.length!==n))throw Error(`${e} validity length must equal grid width * height`);for(let[n,r]of[[`status.converged`,t.status?.converged],[`status.componentCount`,t.status?.componentCount],[`status.overflow`,t.status?.overflow]])if(r&&(d(r,[`uint32`],`${e} ${n}`),r.length<1))throw Error(`${e} ${n} must hold one uint32`);return n}function B(e){return[e.labels,e.validity,e.status?.converged,e.status?.componentCount,e.status?.overflow].filter(e=>e!==void 0)}function V(e,t,n,r,i){let a=u(e,`${t}-keys`,`uint32`,r.labels.length),o=[{name:`labels`,view:r.labels,type:`u32`,access:`read`}],s=[];return r.validity&&(o.push({name:`labelValidity`,view:r.validity,type:`u32`,access:`read`}),s.push(`labelValidity[labelValidityOffset + index] != 0u`)),r.status?.converged&&(o.push({name:`converged`,view:r.status?.converged,type:`u32`,access:`read`}),s.push(`converged[convergedOffset] != 0u`)),r.status?.overflow&&(o.push({name:`overflow`,view:r.status?.overflow,type:`u32`,access:`read`}),s.push(`overflow[overflowOffset] == 0u`)),r.status?.componentCount&&(o.push({name:`componentCount`,view:r.status?.componentCount,type:`u32`,access:`read`}),s.push(`label <= componentCount[componentCountOffset]`)),o.push({name:`keys`,view:a,type:`u32`,access:`read_write`}),{node:_(e,{id:`${t}-keys`,operation:n,variant:`keys`,bindings:o,invocationCount:r.labels.length,body:`let label = labels[labelsOffset + index];
  let accepted = label != 0u && label <= ${i}u${s.map(e=>` && ${e}`).join(``)};
  keys[keysOffset + index] = select(0u, label, accepted);`}),keys:a}}function H(e,t,n,r,i){if(!r.status?.converged&&!r.status?.overflow||i.length===0)return[];let a=Math.max(...i.map(e=>e.view.length)),o=ee(e,`${t}-unresolved`,n,a,t),{x:s,y:c,z:l}=o.layout,u=[],d=[];r.status?.converged&&(u.push({name:`converged`,view:r.status?.converged,type:`u32`,access:`read`}),d.push(`converged[convergedOffset] == 0u`)),r.status?.overflow&&(u.push({name:`overflow`,view:r.status?.overflow,type:`u32`,access:`read`}),d.push(`overflow[overflowOffset] != 0u`)),u.push({name:`dispatch`,view:o.dispatch,type:`u32`,access:`read_write`});let f=[_(e,{id:`${t}-unresolved-arm`,operation:n,variant:`unresolved-arm`,bindings:u,invocationCount:1,body:`let unfinished = ${d.join(` || `)};
  dispatch[dispatchOffset] = select(0u, ${s}u, unfinished);
  dispatch[dispatchOffset + 1u] = ${c}u;
  dispatch[dispatchOffset + 2u] = ${l}u;`})];for(let[r,a]of i.entries()){let{condition:i,extraResources:s}=j(o,`${t}-unresolved-${r}`);f.push(_(e,{id:`${t}-unresolved-${r}`,operation:n,variant:a.type===`f32`?`unresolved-float`:`unresolved-word`,bindings:[{name:`unresolvedTarget`,view:a.view,type:a.type,access:`read_write`}],invocationCount:o.invocationCount,workgroupSize:256,body:a.type===`f32`?`if (index < ${a.view.length}u) {
    var nanBits = 0x7fc00000u;
    unresolvedTarget[unresolvedTargetOffset + index] = bitcast<f32>(nanBits);
  }`:`if (index < ${a.view.length}u) {
    unresolvedTarget[unresolvedTargetOffset + index] = ${R}u;
  }`,condition:i,extraResources:s}))}return f}function U(e,t){return t*Math.ceil(e/16)}function W(e,t){return`let segmentsPerRow = ${Math.ceil(e/16)}u;
  let row = index / segmentsPerRow;
  let segmentStart = (index % segmentsPerRow) * 16u;
  let segmentEnd = min(segmentStart + 16u, WIDTH);
  let rowBase = row * WIDTH;
  var runKey = 0u;
  var runStart = segmentStart;
  ${t.declareRunState??``}
  // The extra iteration at segmentEnd sees a sentinel key that flushes the last run.
  for (var column = segmentStart; column <= segmentEnd; column++) {
    var key = 0xffffffffu;
    if (column < segmentEnd) {
      key = keys[keysOffset + rowBase + column];
    }
    if (key != runKey) {
      if (runKey != 0u) {
        let runEnd = column;
        ${t.onRun}
      }
      runKey = key;
      runStart = column;
      ${t.onRunStart??``}
    }
    ${t.onPixel?`if (column < segmentEnd && key != 0u) {\n      ${t.onPixel}\n    }`:``}
  }`}var G=`GPURasterPatchMetrics`,K=`0xffffffffu`,ye=class{id;props;patchCapacity;constructor(e){let t=e.id??`raster-patch-metrics`;this.id=t,this.props=e,z(t,e);let n=Object.entries(e.output).filter(([,e])=>e!==void 0);if(n.length===0)throw Error(`${t} output must provide at least one column`);this.patchCapacity=n[0][1].length;for(let[e,r]of n)if(e===`areas`||e===`perimeters`?d(r,[`float32`],`${t} output.${e}`):d(r,[`uint32`],`${t} output.${e}`),r.length!==this.patchCapacity)throw Error(`${t} output columns must have identical lengths`);if(this.patchCapacity<1)throw Error(`${t} output columns must hold at least one patch row`);a(t,n.map(([,e])=>e),B(e))}getCommandNodes(e){let{id:t,props:n,patchCapacity:i}=this,{output:a}=n,{width:o,height:s}=n.grid;r(t,e,[...B(n),...Object.values(a).filter(e=>e!==void 0)]);let[c,l,,d,f]=n.grid.affine,p=Math.hypot(c,d),m=Math.hypot(l,f),v=Math.abs(c*f-l*d),y=n.countRasterBorder??!0,b=(n,r)=>u(e,`${t}-${n}`,`uint32`,r),{node:x,keys:S}=V(e,t,G,n,i),C=[x],w=a.pixelCounts??b(`pixel-counts`,i),T=`const WIDTH: u32 = ${o}u;
const HEIGHT: u32 = ${s}u;`;if(C.push(g(e,{id:`${t}-counts-clear`,operation:G,view:w,type:`u32`,value:`0u`}),_(e,{id:`${t}-counts`,operation:G,variant:`counts`,bindings:[{name:`keys`,view:S,type:`u32`,access:`read`},{name:`pixelCounts`,view:w,type:`atomic<u32>`,access:`read_write`}],invocationCount:U(o,s),declarations:T,body:W(o,{onRun:`atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);`})})),a.minColumns||a.minRows||a.maxColumns||a.maxRows){let n={minColumns:a.minColumns??b(`min-columns`,i),minRows:a.minRows??b(`min-rows`,i),maxColumns:a.maxColumns??b(`max-columns`,i),maxRows:a.maxRows??b(`max-rows`,i)};for(let r of[`minColumns`,`minRows`])C.push(g(e,{id:`${t}-${r}-clear`,operation:G,view:n[r],type:`u32`,value:K}));for(let r of[`maxColumns`,`maxRows`])C.push(g(e,{id:`${t}-${r}-clear`,operation:G,view:n[r],type:`u32`,value:`0u`}));C.push(_(e,{id:`${t}-extents`,operation:G,variant:`extents`,bindings:[{name:`keys`,view:S,type:`u32`,access:`read`},{name:`minColumns`,view:n.minColumns,type:`atomic<u32>`,access:`read_write`},{name:`minRows`,view:n.minRows,type:`atomic<u32>`,access:`read_write`},{name:`maxColumns`,view:n.maxColumns,type:`atomic<u32>`,access:`read_write`},{name:`maxRows`,view:n.maxRows,type:`atomic<u32>`,access:`read_write`}],invocationCount:U(o,s),declarations:T,body:W(o,{onRun:`atomicMin(&minColumns[minColumnsOffset + runKey - 1u], runStart);
        atomicMin(&minRows[minRowsOffset + runKey - 1u], row);
        atomicMax(&maxColumns[maxColumnsOffset + runKey - 1u], runEnd - 1u);
        atomicMax(&maxRows[maxRowsOffset + runKey - 1u], row);`})}),_(e,{id:`${t}-extents-empty`,operation:G,variant:`extents-empty`,bindings:[{name:`pixelCounts`,view:w,type:`u32`,access:`read`},{name:`minColumns`,view:n.minColumns,type:`u32`,access:`read_write`},{name:`minRows`,view:n.minRows,type:`u32`,access:`read_write`}],invocationCount:i,body:`if (pixelCounts[pixelCountsOffset + index] == 0u) {
    minColumns[minColumnsOffset + index] = 0u;
    minRows[minRowsOffset + index] = 0u;
  }`}))}if(a.perimeterFaceCounts||a.perimeters){let n=b(`faces-across-rows`,i),r=b(`faces-across-columns`,i);C.push(g(e,{id:`${t}-faces-rows-clear`,operation:G,view:n,type:`u32`,value:`0u`}),g(e,{id:`${t}-faces-columns-clear`,operation:G,view:r,type:`u32`,value:`0u`}),_(e,{id:`${t}-faces`,operation:G,variant:`faces`,bindings:[{name:`keys`,view:S,type:`u32`,access:`read`},{name:`acrossRows`,view:n,type:`atomic<u32>`,access:`read_write`},{name:`acrossColumns`,view:r,type:`atomic<u32>`,access:`read_write`}],invocationCount:U(o,s),declarations:`${T}
const COUNT_BORDER: bool = ${y};

// Whether the neighbor at (column + dx, row + dy) differs from \`key\`.
fn isExposed(column: u32, row: u32, dx: i32, dy: i32, key: u32) -> bool {
  let neighborColumn = i32(column) + dx;
  let neighborRow = i32(row) + dy;
  if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
    return COUNT_BORDER;
  }
  return keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)] != key;
}`,body:W(o,{declareRunState:`var runRowFaces = 0u;
  var runColumnFaces = 0u;`,onRunStart:`runRowFaces = 0u;
      runColumnFaces = 0u;`,onPixel:`if (isExposed(column, row, 0, -1, key)) { runRowFaces++; }
      if (isExposed(column, row, 0, 1, key)) { runRowFaces++; }
      if (isExposed(column, row, -1, 0, key)) { runColumnFaces++; }
      if (isExposed(column, row, 1, 0, key)) { runColumnFaces++; }`,onRun:`if (runRowFaces > 0u) { atomicAdd(&acrossRows[acrossRowsOffset + runKey - 1u], runRowFaces); }
        if (runColumnFaces > 0u) { atomicAdd(&acrossColumns[acrossColumnsOffset + runKey - 1u], runColumnFaces); }`})}));let c=[{name:`acrossRows`,view:n,type:`u32`,access:`read`},{name:`acrossColumns`,view:r,type:`u32`,access:`read`}];a.perimeterFaceCounts&&c.push({name:`faceCounts`,view:a.perimeterFaceCounts,type:`u32`,access:`read_write`}),a.perimeters&&c.push({name:`perimeters`,view:a.perimeters,type:`f32`,access:`read_write`}),C.push(_(e,{id:`${t}-perimeters`,operation:G,variant:`perimeters`,bindings:c,invocationCount:i,body:`let rowFaces = acrossRows[acrossRowsOffset + index];
  let columnFaces = acrossColumns[acrossColumnsOffset + index];
  ${a.perimeterFaceCounts?`faceCounts[faceCountsOffset + index] = rowFaces + columnFaces;`:``}
  ${a.perimeters?`perimeters[perimetersOffset + index] = f32(rowFaces) * ${h(p)} + f32(columnFaces) * ${h(m)};`:``}`}))}return a.areas&&C.push(_(e,{id:`${t}-areas`,operation:G,variant:`areas`,bindings:[{name:`pixelCounts`,view:w,type:`u32`,access:`read`},{name:`areas`,view:a.areas,type:`f32`,access:`read_write`}],invocationCount:i,body:`areas[areasOffset + index] = f32(pixelCounts[pixelCountsOffset + index]) * ${h(v)};`})),C.push(...H(e,t,G,n,Object.entries(a).filter(([,e])=>e!==void 0).map(([e,t])=>({view:t,type:e===`areas`||e===`perimeters`?`f32`:`u32`})))),C}};function be(e,t=new Uint32Array(4)){k(`getGPURasterSieveParameterValues`,t,4);let{minimumPixels:n}=e;if(!Number.isInteger(n)||n<1||n>4294967295)throw Error(`getGPURasterSieveParameterValues minimumPixels must be an integer in [1, 2^32 - 1]`);return t.set([n,0,0,0]),t}var q=`GPURasterSieve`,xe=`0xffffffffu`,Se=256,Ce=class{id;props;mode;connectivity;constructor(e){let t=e.id??`raster-sieve`;this.id=t,this.props=e,this.mode=e.mode??`remove`,this.connectivity=e.connectivity??4;let n=z(t,e);if(this.mode!==`remove`&&this.mode!==`merge`)throw Error(`${t} mode must be 'remove' or 'merge'`);if(this.connectivity!==4&&this.connectivity!==8)throw Error(`${t} connectivity must be 4 or 8`);if(i(t,`patchCapacity`,e.patchCapacity,1,n),d(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must contain at least 4 values`);let{output:r}=e;if(d(r.labels,[`uint32`],`${t} output.labels`),r.labels.length!==n)throw Error(`${t} output.labels length must equal grid width * height`);if(r.patchTargets&&(d(r.patchTargets,[`uint32`],`${t} output.patchTargets`),r.patchTargets.length!==e.patchCapacity))throw Error(`${t} output.patchTargets length must equal patchCapacity`);if(r.sievedCount&&(d(r.sievedCount,[`uint32`],`${t} output.sievedCount`),r.sievedCount.length<1))throw Error(`${t} output.sievedCount must hold one uint32`);a(t,[r.labels,r.patchTargets,r.sievedCount],[...B(e),e.parameters])}getCommandNodes(e){let{id:t,props:n,mode:i,connectivity:a}=this,{output:o,patchCapacity:s,parameters:c}=n,{width:l,height:d}=n.grid;r(t,e,[...B(n),c,o.labels,o.patchTargets,o.sievedCount]);let f=l*d,p=(n,r)=>u(e,`${t}-${n}`,`uint32`,r),{node:m,keys:h}=V(e,t,q,n,s),v=p(`pixel-counts`,s),y=o.patchTargets??p(`patch-targets`,s),b=`const WIDTH: u32 = ${l}u;
const HEIGHT: u32 = ${d}u;`,x=[m,g(e,{id:`${t}-counts-clear`,operation:q,view:v,type:`u32`,value:`0u`}),_(e,{id:`${t}-counts`,operation:q,variant:`counts`,bindings:[{name:`keys`,view:h,type:`u32`,access:`read`},{name:`pixelCounts`,view:v,type:`atomic<u32>`,access:`read_write`}],invocationCount:U(l,d),declarations:b,body:W(l,{onRun:`atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);`})})],S,C;if(i===`merge`){S=p(`best-counts`,s),C=p(`merge-targets`,s);let n=e=>`let key = keys[keysOffset + index];
  if (key != 0u && pixelCounts[pixelCountsOffset + key - 1u] < parameters[parametersOffset]) {
    let column = index % WIDTH;
    let row = index / WIDTH;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if ((dx == 0 && dy == 0) || (${a} == 4 && dx != 0 && dy != 0)) {
          continue;
        }
        let neighborColumn = i32(column) + dx;
        let neighborRow = i32(row) + dy;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)];
        if (neighbor == 0u || neighbor == key) {
          continue;
        }
        let neighborCount = pixelCounts[pixelCountsOffset + neighbor - 1u];
        if (neighborCount >= parameters[parametersOffset]) {
          ${e}
        }
      }
    }
  }`;x.push(g(e,{id:`${t}-best-counts-clear`,operation:q,view:S,type:`u32`,value:`0u`}),g(e,{id:`${t}-merge-targets-clear`,operation:q,view:C,type:`u32`,value:xe}),_(e,{id:`${t}-best-counts`,operation:q,variant:`best-counts`,bindings:[{name:`keys`,view:h,type:`u32`,access:`read`},{name:`pixelCounts`,view:v,type:`u32`,access:`read`},{name:`parameters`,view:c,type:`u32`,access:`read`},{name:`bestCounts`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:f,declarations:b,body:n(`atomicMax(&bestCounts[bestCountsOffset + key - 1u], neighborCount);`)}),_(e,{id:`${t}-merge-targets`,operation:q,variant:`merge-targets`,bindings:[{name:`keys`,view:h,type:`u32`,access:`read`},{name:`pixelCounts`,view:v,type:`u32`,access:`read`},{name:`parameters`,view:c,type:`u32`,access:`read`},{name:`bestCounts`,view:S,type:`u32`,access:`read`},{name:`targets`,view:C,type:`atomic<u32>`,access:`read_write`}],invocationCount:f,declarations:b,body:n(`if (neighborCount == bestCounts[bestCountsOffset + key - 1u]) {
            atomicMin(&targets[targetsOffset + key - 1u], neighbor);
          }`)}))}let w=o.sievedCount;return x.push(...w?[g(e,{id:`${t}-sieved-clear`,operation:q,view:w,type:`u32`,value:`0u`,componentCount:1})]:[],_(e,{id:`${t}-patch-targets`,operation:q,variant:`patch-targets`,bindings:[{name:`pixelCounts`,view:v,type:`u32`,access:`read`},{name:`parameters`,view:c,type:`u32`,access:`read`},{name:`patchTargets`,view:y,type:`u32`,access:`read_write`},...S&&C?[{name:`bestCounts`,view:S,type:`u32`,access:`read`},{name:`targets`,view:C,type:`u32`,access:`read`}]:[],...w?[{name:`sievedCount`,view:w,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:s,...w?{guardIndex:!1,workgroupSize:Se,declarations:`var<workgroup> workgroupSievedCount: atomic<u32>;`}:{},body:`${w?`if (localInvocationIndex == 0u) {
    atomicStore(&workgroupSievedCount, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {`:``}
  let count = pixelCounts[pixelCountsOffset + index];
  var finalLabel = 0u;
  if (count >= parameters[parametersOffset]) {
    finalLabel = index + 1u;
  } else if (count > 0u) {
    ${w?`atomicAdd(&workgroupSievedCount, 1u);`:``}
    ${i===`merge`?`if (bestCounts[bestCountsOffset + index] != 0u) {
      finalLabel = targets[targetsOffset + index];
    }`:``}
  }
  patchTargets[patchTargetsOffset + index] = finalLabel;${w?`
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupTotal = atomicLoad(&workgroupSievedCount);
    if (workgroupTotal > 0u) {
      atomicAdd(&sievedCount[sievedCountOffset], workgroupTotal);
    }
  }`:``}`}),_(e,{id:`${t}-relabel`,operation:q,variant:`relabel`,bindings:[{name:`keys`,view:h,type:`u32`,access:`read`},{name:`patchTargets`,view:y,type:`u32`,access:`read`},{name:`sievedLabels`,view:o.labels,type:`u32`,access:`read_write`}],invocationCount:f,body:`let key = keys[keysOffset + index];
  sievedLabels[sievedLabelsOffset + index] =
    select(0u, patchTargets[patchTargetsOffset + key - 1u], key != 0u);`})),x.push(...H(e,t,q,n,[{view:o.labels,type:`u32`},...o.patchTargets?[{view:o.patchTargets,type:`u32`}]:[],...w?[{view:w,type:`u32`}]:[]])),x}},we=1<<24;function Te(e,t){if(!Number.isFinite(e))throw Error(`getGPURasterStretchParameterValues window ${t} must be finite`);return Math.min(Math.max(Math.floor(e),0),we)}function Ee(e={},t=new Float32Array(16)){k(`getGPURasterStretchParameterValues`,t,16),t.fill(0,0,16);let n=e.window??[0,0,16777216,16777216];t[0]=Te(n[0],`column0`),t[1]=Te(n[1],`row0`),t[2]=Te(n[2],`column1`),t[3]=Te(n[3],`row1`);let r=e.domain??`auto`;if(r!==`auto`){let[e,n]=r;if(!Number.isFinite(e)||!Number.isFinite(n)||e>n)throw Error(`getGPURasterStretchParameterValues domain must be finite with min <= max`);if(!Number.isFinite(Math.fround(n-e))&&Number.isFinite(n-e))throw Error(`getGPURasterStretchParameterValues domain range must fit in float32`);t[4]=1,t[5]=e,t[6]=n}let i=e.mode??`linear`;if(i!==`linear`&&i!==`percentile`&&i!==`equalize`)throw Error(`getGPURasterStretchParameterValues mode ${String(i)} is not supported`);t[7]=i===`linear`?0:i===`percentile`?1:2;let[a,o]=e.percentiles??[2,98];if(!Number.isFinite(a)||!Number.isFinite(o)||a<0||o>100||a>o)throw Error(`getGPURasterStretchParameterValues percentiles must satisfy 0 <= low <= high <= 100`);t[8]=a/100,t[9]=o/100;let s=e.gamma??1;if(!Number.isFinite(s)||s<=0)throw Error(`getGPURasterStretchParameterValues gamma must be positive and finite`);t[10]=s;let c=e.sigmoidContrast??0;if(!Number.isFinite(c)||c<0||c>80)throw Error(`getGPURasterStretchParameterValues sigmoidContrast must be in [0, 80]`);t[11]=c;let l=e.sigmoidMidpoint??.5;if(!Number.isFinite(l)||l<=0||l>=1)throw Error(`getGPURasterStretchParameterValues sigmoidMidpoint must be in (0, 1)`);t[12]=l;let u=e.paletteInterpolation??`nearest`;if(u!==`nearest`&&u!==`linear`)throw Error(`getGPURasterStretchParameterValues paletteInterpolation ${String(u)} is not supported`);return t[13]=u===`linear`?1:0,t}var J=`GPURasterStretch`,De=1024,Oe=256,Y=65536,X=256,Z=16,ke=class{id;props;binCount;lutSize;constructor(e){this.id=e.id??`raster-stretch`,this.props=e;let{id:t}=this,{output:n}=e,r=A(t,e.grid);for(let[n,r]of[[`width`,e.grid.width],[`height`,e.grid.height]])i(t,`grid ${n}`,r,1,we);if(this.binCount=e.binCount??De,this.lutSize=e.lutSize??Oe,M(t,`binCount`,this.binCount,16,65536),M(t,`lutSize`,this.lutSize,2,65536),N(t,e.noDataValue),P(t,`values`,e.values,`float32`,r),!e.values)throw Error(`${t} needs values`);if(P(t,`validity`,e.validity,`uint32`,r),P(t,`regionMask`,e.regionMask,`uint32`,r),P(t,`parameters`,e.parameters,`float32`,16),!e.parameters)throw Error(`${t} needs parameters`);if(P(t,`palette`,e.palette,`uint32`,1),e.palette&&e.palette.length>Y)throw Error(`${t} palette must hold at most ${Y} colors`);if(!n||Object.values(n).every(e=>e===void 0))throw Error(`${t} needs at least one output`);if((n.colors||n.lutColors)&&!e.palette)throw Error(`${t} output.colors and output.lutColors require palette`);if(P(t,`output.stretched`,n.stretched,`float32`,r),P(t,`output.colors`,n.colors,`uint32`,r),P(t,`output.lut`,n.lut,`float32`,this.lutSize),P(t,`output.lutColors`,n.lutColors,`uint32`,this.lutSize),P(t,`output.histogram`,n.histogram,`uint32`,this.binCount),n.histogram&&n.histogram.length!==this.binCount)throw Error(`${t} output.histogram must hold exactly binCount (${this.binCount}) rows`);P(t,`output.statistics`,n.statistics,`float32`,8),re(t,[n.stretched,n.colors,n.lut,n.lutColors,n.histogram,n.statistics],[e.values,e.validity,e.regionMask,e.parameters,e.palette])}getCommandNodes(e){let{id:t,props:n,binCount:r,lutSize:i}=this,{output:a}=n,{width:o,height:s}=n.grid,c=o*s;te(t,e,[n.values,n.validity,n.regionMask,n.parameters,n.palette,a.stretched,a.colors,a.lut,a.lutColors,a.histogram,a.statistics]);let l=[],d=u(e,`${t}-keys`,`uint32`,4),f=u(e,`${t}-state`,`float32`,8),m=a.histogram??u(e,`${t}-histogram`,`uint32`,r),h=u(e,`${t}-cdf`,`uint32`,r),v=!!(a.stretched||a.colors),b=!!(a.lut||a.lutColors),S=n.palette?.length??0,C=r*4<=e.device.limits.maxComputeWorkgroupStorageSize,w=Math.min(64,Math.max(8,Math.ceil(r/32))),T=[{name:`values`,view:n.values,type:`f32`,access:`read`}];n.validity&&T.push({name:`validity`,view:n.validity,type:`u32`,access:`read`});let E=[...T];n.regionMask&&E.push({name:`regionMask`,view:n.regionMask,type:`u32`,access:`read`});let D={name:`params`,view:n.parameters,type:`f32`,access:`read`},O=`const WIDTH: u32 = ${o}u;
const HEIGHT: u32 = ${s}u;
const BIN_COUNT: u32 = ${r}u;
const LUT_SIZE: u32 = ${i}u;
const PALETTE_SIZE: u32 = ${S}u;`,k=`${O}
${ne(n.noDataValue)}
fn readCellValue(index: u32, value: ptr<function, f32>) -> bool {
  let sample = values[valuesOffset + index];
  *value = sample;
  var isValid = !isNoDataValue(sample);
  ${n.validity?`isValid = isValid && validity[validityOffset + index] != 0u;`:``}
  return isValid;
}`,A=`${k}
// Statistics include cells inside the window, in the region mask, valid and finite.
fn readStatisticsValue(index: u32, value: ptr<function, f32>) -> bool {
  let column = index % WIDTH;
  let row = index / WIDTH;
  let column0 = min(u32(params[paramsOffset]), WIDTH);
  let row0 = min(u32(params[paramsOffset + 1u]), HEIGHT);
  let column1 = min(u32(params[paramsOffset + 2u]), WIDTH);
  let row1 = min(u32(params[paramsOffset + 3u]), HEIGHT);
  if (column < column0 || column >= column1 || row < row0 || row >= row1) {
    return false;
  }
  ${n.regionMask?`if (regionMask[regionMaskOffset + index] == 0u) { return false; }`:``}
  var sample = 0.0;
  if (!readCellValue(index, &sample)) {
    return false;
  }
  *value = sample;
  return isFiniteValue(sample);
}`;l.push(_(e,{id:`${t}-clear-keys`,operation:J,variant:`clear-keys`,bindings:[{name:`keys`,view:d,type:`u32`,access:`read_write`}],invocationCount:1,body:`keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;
  keys[keysOffset + 2u] = 0u;
  keys[keysOffset + 3u] = 0u;`}),_(e,{id:`${t}-reduce`,operation:J,variant:`reduce`,bindings:[...E,D,{name:`keys`,view:d,type:`atomic<u32>`,access:`read_write`}],invocationCount:Math.ceil(c/Z),guardIndex:!1,workgroupSize:X,declarations:`${A}
const CELL_COUNT: u32 = ${c}u;
const CELLS_PER_THREAD: u32 = ${Z}u;
var<workgroup> tileMinimum: array<u32, ${X}>;
var<workgroup> tileMaximum: array<u32, ${X}>;
var<workgroup> tileCount: array<u32, ${X}>;
${y}`,body:`var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var includedCount = 0u;
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${X}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    var value = 0.0;
    if (readStatisticsValue(cell, &value)) {
      let key = getOrderedKey(value);
      minimumKey = min(minimumKey, key);
      maximumKey = max(maximumKey, key);
      includedCount++;
    }
  }
  tileMinimum[localInvocationIndex] = minimumKey;
  tileMaximum[localInvocationIndex] = maximumKey;
  tileCount[localInvocationIndex] = includedCount;
  workgroupBarrier();
  for (var stride = ${X/2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      tileMinimum[localInvocationIndex] = min(tileMinimum[localInvocationIndex], tileMinimum[localInvocationIndex + stride]);
      tileMaximum[localInvocationIndex] = max(tileMaximum[localInvocationIndex], tileMaximum[localInvocationIndex + stride]);
      tileCount[localInvocationIndex] = tileCount[localInvocationIndex] + tileCount[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && tileCount[0] > 0u) {
    atomicAdd(&keys[keysOffset + 2u], tileCount[0]);
    if (params[paramsOffset + 4u] == 0.0) {
      atomicMin(&keys[keysOffset], tileMinimum[0]);
      atomicMax(&keys[keysOffset + 1u], tileMaximum[0]);
    }
  }`}),_(e,{id:`${t}-domain`,operation:J,variant:`domain`,bindings:[D,{name:`keys`,view:d,type:`u32`,access:`read`},{name:`state`,view:f,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const BIN_COUNT: u32 = ${r}u;
${x}`,body:`let count = keys[keysOffset + 2u];
  var domainMinimum = 0.0;
  var domainMaximum = 0.0;
  if (params[paramsOffset + 4u] != 0.0) {
    domainMinimum = params[paramsOffset + 5u];
    domainMaximum = params[paramsOffset + 6u];
  } else if (count > 0u) {
    domainMinimum = decodeOrderedKey(keys[keysOffset]);
    domainMaximum = decodeOrderedKey(keys[keysOffset + 1u]);
  }
  let hasRange = domainMaximum > domainMinimum;
  let range = domainMaximum - domainMinimum;
  state[stateOffset] = domainMinimum;
  state[stateOffset + 1u] = domainMaximum;
  state[stateOffset + 2u] = select(0.0, f32(BIN_COUNT) / range, hasRange);
  state[stateOffset + 3u] = select(0.0, range / f32(BIN_COUNT), hasRange);
  state[stateOffset + 4u] = domainMinimum;
  state[stateOffset + 5u] = domainMaximum;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 7u] = f32(count);`}),g(e,{id:`${t}-clear-histogram`,operation:J,view:m,type:`u32`,value:`0u`}),_(e,{id:`${t}-histogram`,operation:J,variant:C?`histogram-private`:`histogram`,bindings:[...E,D,{name:`state`,view:f,type:`f32`,access:`read`},{name:`histogram`,view:m,type:`atomic<u32>`,access:`read_write`}],invocationCount:C?Math.ceil(c/w):c,guardIndex:!C,workgroupSize:C?X:void 0,declarations:C?`${A}
const CELL_COUNT: u32 = ${c}u;
const CELLS_PER_THREAD: u32 = ${w}u;
var<workgroup> localHistogram: array<atomic<u32>, ${r}>;`:A,body:C?`for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${X}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  let binScale = state[stateOffset + 2u];
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${X}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    var value = 0.0;
    if (readStatisticsValue(cell, &value) && value >= domainMinimum && value <= domainMaximum) {
      let bin = min(u32((value - domainMinimum) * binScale), BIN_COUNT - 1u);
      atomicAdd(&localHistogram[bin], 1u);
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${X}u) {
    let partial = atomicLoad(&localHistogram[bin]);
    if (partial != 0u) {
      atomicAdd(&histogram[histogramOffset + bin], partial);
    }
  }`:`var value = 0.0;
  if (!readStatisticsValue(index, &value)) {
    return;
  }
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  if (value < domainMinimum || value > domainMaximum) {
    return;
  }
  let position = (value - domainMinimum) * state[stateOffset + 2u];
  let bin = min(u32(position), BIN_COUNT - 1u);
  atomicAdd(&histogram[histogramOffset + bin], 1u);`}),...new p({id:`${t}-cdf-scan`,input:m,output:h,mode:`inclusive`}).getCommandNodes(e));let j=[D,{name:`state`,view:f,type:`f32`,access:`read_write`},{name:`cdf`,view:h,type:`u32`,access:`read`}];if(a.statistics&&j.push({name:`statisticsOut`,view:a.statistics,type:`f32`,access:`read_write`}),l.push(_(e,{id:`${t}-finalize`,operation:J,variant:`finalize`,bindings:j,invocationCount:1,declarations:`${O}
${ne(void 0)}
// Smallest bin whose cumulative count is positive and reaches target, then linear inside it.
fn getPercentileValue(rank: f32) -> f32 {
  var low = 0u;
  var high = BIN_COUNT - 1u;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let cumulative = cdf[cdfOffset + middle];
    if (cumulative > 0u && f32(cumulative) >= rank) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  var before = 0u;
  if (low > 0u) {
    before = cdf[cdfOffset + low - 1u];
  }
  let count = cdf[cdfOffset + low] - before;
  var fraction = 0.0;
  if (count > 0u) {
    fraction = clamp((rank - f32(before)) / f32(count), 0.0, 1.0);
  }
  let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  return clamp(domainMinimum + (f32(low) + fraction) * state[stateOffset + 3u], domainMinimum, domainMaximum);
}`,body:`let domainMinimum = state[stateOffset];
  let domainMaximum = state[stateOffset + 1u];
  let validCount = state[stateOffset + 7u];
  let total = cdf[cdfOffset + BIN_COUNT - 1u];
  let mode = u32(params[paramsOffset + 7u]);
  var lo = domainMinimum;
  var hi = domainMaximum;
  if (mode != 0u && total == 0u) {
    hi = lo;
  } else if (mode == 1u) {
    lo = getPercentileValue(params[paramsOffset + 8u] * f32(total));
    hi = getPercentileValue(params[paramsOffset + 9u] * f32(total));
  }
  state[stateOffset + 4u] = lo;
  state[stateOffset + 5u] = hi;
  state[stateOffset + 6u] = f32(total);
  ${a.statistics?`let isEmpty = params[paramsOffset + 4u] == 0.0 && validCount == 0.0;
  let nan = getNaN();
  statisticsOut[statisticsOutOffset] = select(domainMinimum, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 1u] = select(domainMaximum, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 2u] = select(lo, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 3u] = select(hi, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 4u] = validCount;
  statisticsOut[statisticsOutOffset + 5u] = select(state[stateOffset + 3u], nan, isEmpty);
  statisticsOut[statisticsOutOffset + 6u] = 0.0;
  statisticsOut[statisticsOutOffset + 7u] = 0.0;`:``}`})),!v&&!b)return l;let ee=`${Ae()}
${n.palette?je():``}`,M=[D,{name:`state`,view:f,type:`f32`,access:`read`},{name:`cdf`,view:h,type:`u32`,access:`read`}],N=n.palette?[{name:`palette`,view:n.palette,type:`u32`,access:`read`}]:[];if(b){let n=[...M];a.lut&&n.push({name:`lutOut`,view:a.lut,type:`f32`,access:`read_write`}),a.lutColors&&n.push({name:`lutColorsOut`,view:a.lutColors,type:`u32`,access:`read_write`}),n.push(...N),l.push(_(e,{id:`${t}-lut`,operation:J,variant:`lut`,bindings:n,invocationCount:i,declarations:`${O}
${ee}`,body:`let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  let value = lo + (f32(index) + 0.5) / f32(LUT_SIZE) * (hi - lo);
  let t = getStretchedValue(value);
  ${a.lut?`lutOut[lutOutOffset + index] = t;`:``}
  ${a.lutColors?`lutColorsOut[lutColorsOutOffset + index] = getPaletteColor(t);`:``}`}))}if(v){let n=[...T,...M];a.stretched&&n.push({name:`stretchedOut`,view:a.stretched,type:`f32`,access:`read_write`}),a.colors&&n.push({name:`colorsOut`,view:a.colors,type:`u32`,access:`read_write`}),n.push(...N),l.push(_(e,{id:`${t}-apply`,operation:J,variant:`apply`,bindings:n,invocationCount:c,declarations:`${k}
${ee}`,body:`var value = 0.0;
  if (!readCellValue(index, &value)) {
    ${a.stretched?`stretchedOut[stretchedOutOffset + index] = getNaN();`:``}
    ${a.colors?`colorsOut[colorsOutOffset + index] = 0u;`:``}
    return;
  }
  let t = getStretchedValue(value);
  ${a.stretched?`stretchedOut[stretchedOutOffset + index] = t;`:``}
  ${a.colors?`colorsOut[colorsOutOffset + index] = getPaletteColor(t);`:``}`}))}return l}};function Ae(){return`
fn applyCurve(unit: f32) -> f32 {
  var t = unit;
  let gamma = params[paramsOffset + 10u];
  if (gamma != 1.0 && t > 0.0 && t < 1.0) {
    t = pow(t, 1.0 / gamma);
  }
  let contrast = params[paramsOffset + 11u];
  if (contrast > 0.0) {
    let midpoint = params[paramsOffset + 12u];
    let lowEnd = 1.0 / (1.0 + exp(contrast * midpoint));
    let highEnd = 1.0 / (1.0 + exp(contrast * (midpoint - 1.0)));
    let denominator = highEnd - lowEnd;
    if (denominator > 0.0) {
      t = (1.0 / (1.0 + exp(contrast * (midpoint - t))) - lowEnd) / denominator;
    }
    t = clamp(t, 0.0, 1.0);
  }
  return t;
}

fn getStretchedValue(value: f32) -> f32 {
  let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  if (hi <= lo) {
    return select(select(0.5, 1.0, value > hi), 0.0, value < lo);
  }
  var unit = 0.0;
  if (u32(params[paramsOffset + 7u]) == 2u) {
    let domainMinimum = state[stateOffset];
    let domainMaximum = state[stateOffset + 1u];
    if (value <= domainMinimum) {
      unit = 0.0;
    } else if (value >= domainMaximum) {
      unit = 1.0;
    } else {
      let position = (value - domainMinimum) * state[stateOffset + 2u];
      let bin = min(u32(position), BIN_COUNT - 1u);
      let fraction = clamp(position - f32(bin), 0.0, 1.0);
      var before = 0u;
      if (bin > 0u) {
        before = cdf[cdfOffset + bin - 1u];
      }
      let count = cdf[cdfOffset + bin] - before;
      unit = (f32(before) + fraction * f32(count)) / state[stateOffset + 6u];
    }
  } else {
    unit = clamp((value - lo) / (hi - lo), 0.0, 1.0);
  }
  return applyCurve(unit);
}`}function je(){return`
fn unpackColor(color: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(color & 255u),
    f32((color >> 8u) & 255u),
    f32((color >> 16u) & 255u),
    f32((color >> 24u) & 255u)
  );
}

fn getPaletteColor(t: f32) -> u32 {
  if (params[paramsOffset + 13u] == 0.0 || PALETTE_SIZE == 1u) {
    let index = min(u32(t * f32(PALETTE_SIZE)), PALETTE_SIZE - 1u);
    return palette[paletteOffset + index];
  }
  let position = t * f32(PALETTE_SIZE - 1u);
  let lower = min(u32(position), PALETTE_SIZE - 2u);
  let fraction = position - f32(lower);
  let first = unpackColor(palette[paletteOffset + lower]);
  let second = unpackColor(palette[paletteOffset + lower + 1u]);
  let channels = vec4<u32>(clamp(floor(mix(first, second, fraction) + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
  return channels.x | (channels.y << 8u) | (channels.z << 16u) | (channels.w << 24u);
}`}var Me=256,Ne=Uint32Array.BYTES_PER_ELEMENT,Pe=256;function Fe(e,t={}){if(e.type!==`webgpu`)throw Error(`Raster compute requires a WebGPU device`);let n=t.workgroupSize??Me,r=t.bytesPerElement??Ne;if(!Number.isSafeInteger(n)||n<=0)throw Error(`Raster workgroup size must be a positive safe integer`);if(!Number.isSafeInteger(r)||r<=0)throw Error(`Raster bytes per element must be a positive safe integer`);let{maxTextureDimension2D:i,maxComputeInvocationsPerWorkgroup:a,maxComputeWorkgroupSizeX:o,maxComputeWorkgroupsPerDimension:s,maxStorageBufferBindingSize:c,maxBufferSize:l,minStorageBufferOffsetAlignment:u}=e.limits;if(n>a||n>o)throw Error(`Raster workgroup size ${n} exceeds device compute limits`);let d=s*n,f=Math.floor(c/r),p=Math.floor(l/r),m=Math.min(d,f,p);if(!Number.isSafeInteger(d)||m<1)throw Error(`Raster device limits cannot accommodate one packed element`);return{workgroupSize:n,bytesPerElement:r,maxTextureDimension2D:i,maxComputeInvocationsPerWorkgroup:a,maxComputeWorkgroupSizeX:o,maxComputeWorkgroupsPerDimension:s,maxStorageBufferBindingByteLength:c,maxBufferByteLength:l,storageBufferOffsetAlignment:Math.max(Pe,u),maxDispatchElementCount:d,maxStorageBufferBindingElementCount:f,maxBufferElementCount:p,maxStripeElementCount:m}}function Ie(e){return e.limits.maxComputeWorkgroupsPerDimension*256}var Le=class{id;width;height;input;count;sum;mean;extent;constructor(e){if(this.id=e.id??`gpu-raster-statistics`,this.width=e.width,this.height=e.height,this.input=e.input,this.count=e.count,this.sum=e.sum,this.mean=e.mean,this.extent=e.extent,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count must fit in uint32`);if(this.input.storage.kind!==`buffer`||this.input.format!==`float32`)throw Error(`${this.id} requires a float32 buffer-backed raster band`);let n=O(this.input,this,`${this.id} input`);T(this.count,`uint32`,1,`${this.id} count`),T(this.sum,`float32`,1,`${this.id} sum`),T(this.mean,`float32`,1,`${this.id} mean`),T(this.extent,`float32`,2,`${this.id} extent`);let r=[this.count,this.sum,this.mean,this.extent];if(r.some(e=>e.buffer.graph!==n))throw Error(`${this.id} resources must belong to the same graph`);let i=[this.input.storage.values.buffer,...this.input.validity?[this.input.validity.buffer]:[]];if(r.some((e,t)=>[...i,...r.slice(t+1).map(e=>e.buffer)].includes(e.buffer)))throw Error(`${this.id} inputs and outputs must use separate buffers`);C(this.input.scale??1),C(this.input.offset??0)}addToGraph(e){l(this.id,`width x height`,this.width*this.height,Ie(e.device));let t=[this.input.storage.values,...this.input.validity?[this.input.validity]:[]],n=[this.count,this.sum,this.mean,this.extent];for(let r of[...t,...n]){if(r.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);w(e.device,r,`${this.id} ${r.buffer.id}`)}let r=Fe(e.device),i=this.width*this.height;if(i>r.maxDispatchElementCount)throw Error(`${this.id} input exceeds one bounded scalar-analysis dispatch`);let[a,o]=E(e.device,this.width,this.height,this.id),s=u(e,`${this.id}-calibrated-values`,`float32`,i),c=u(e,`${this.id}-resolved-validity`,`uint32`,i);w(e.device,s,`${this.id} calibrated values`),w(e.device,c,`${this.id} resolved validity`),this.addPreparationPass(e,s,c,[a,o]),e.add(new m({id:`${this.id}-count`,input:c,output:this.count,operation:`sum`})),e.add(new m({id:`${this.id}-sum`,input:s,mask:c,output:this.sum,operation:`sum`})),e.add(new m({id:`${this.id}-extent`,input:s,mask:c,output:this.extent,operation:`extent`})),this.addMeanPass(e)}addPreparationPass(e,t,n,r){let i=[{buffer:this.input.storage.values,usage:`storage-read`},{buffer:t,usage:`storage-write`},{buffer:n,usage:`storage-write`}];this.input.validity&&i.push({buffer:this.input.validity,usage:`storage-read`}),e.addComputePass({id:`${this.id}-prepare`,resources:i,compile:({device:e})=>{let i=[{name:`sourceValues`,type:`read-only-storage`,group:0,location:0},{name:`calibratedValues`,type:`storage`,group:0,location:1},{name:`resolvedValidity`,type:`storage`,group:0,location:2}];this.input.validity&&i.push({name:`sourceValidity`,type:`read-only-storage`,group:0,location:3});let a=new f(e,{id:`${this.id}-prepare`,source:this.getPreparationShader(t,n),shaderLayout:{bindings:i}}),o={sourceValues:this.input.storage.values,calibratedValues:t,resolvedValidity:n};this.input.validity&&(o.sourceValidity=this.input.validity);let s=v(o);return{encode:({computePass:e,getBuffer:t})=>{let n=s(t);n&&a.setBindings(n),a.dispatch(e,r[0],r[1])},destroy:()=>a.destroy()}}})}getPreparationShader(e,t){let n=this.input.validity,r=n?`@group(0) @binding(3) var<storage, read> sourceValidity: array<u32>;\nconst SOURCE_VALIDITY_OFFSET: u32 = ${o(n)}u;`:``,i=[`isFiniteValue(rawSample)`,`isFiniteValue(calibratedSample)`];return n&&i.push(`sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u`),this.input.noDataValue!==void 0&&!Number.isNaN(this.input.noDataValue)&&i.push(`rawSample != ${C(this.input.noDataValue)}`),`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const SOURCE_OFFSET: u32 = ${o(this.input.storage.values)}u;
const CALIBRATED_OFFSET: u32 = ${o(e)}u;
const VALIDITY_OFFSET: u32 = ${o(t)}u;
@group(0) @binding(0) var<storage, read> sourceValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> calibratedValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> resolvedValidity: array<u32>;
${r}

${b}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let rawSample = sourceValues[SOURCE_OFFSET + pixelIndex];
  let calibratedSample = rawSample * ${C(this.input.scale??1)} + ${C(this.input.offset??0)};
  let validSample = ${i.join(` && `)};
  calibratedValues[CALIBRATED_OFFSET + pixelIndex] = select(0.0, calibratedSample, validSample);
  resolvedValidity[VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, validSample);
}`}addMeanPass(e){e.addComputePass({id:`${this.id}-mean`,resources:[{buffer:this.count,usage:`storage-read`},{buffer:this.sum,usage:`storage-read`},{buffer:this.mean,usage:`storage-write`}],compile:({device:e})=>{let t=new f(e,{id:`${this.id}-mean`,source:`
@group(0) @binding(0) var<storage, read> countValues: array<u32>;
@group(0) @binding(1) var<storage, read> sumValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> meanValues: array<f32>;

@compute @workgroup_size(1)
fn main() {
  let count = countValues[${o(this.count)}u];
  let sum = sumValues[${o(this.sum)}u];
  meanValues[${o(this.mean)}u] = select(0.0, sum / f32(max(count, 1u)), count > 0u);
}`,shaderLayout:{bindings:[{name:`countValues`,type:`read-only-storage`,group:0,location:0},{name:`sumValues`,type:`read-only-storage`,group:0,location:1},{name:`meanValues`,type:`storage`,group:0,location:2}]}}),n=v({countValues:this.count,sumValues:this.sum,meanValues:this.mean});return{encode:({computePass:e,getBuffer:r})=>{let i=n(r);i&&t.setBindings(i),t.dispatch(e,1)},destroy:()=>t.destroy()}}})}},Re=64,ze=32,Be=class{id;width;height;input;output;outputValidity;converged;iterationCount;connectivity;maximumIterations;constructor(e){if(this.id=e.id??`gpu-raster-connected-components`,this.width=e.width,this.height=e.height,this.input=e.input,this.output=e.output,this.outputValidity=e.outputValidity,this.converged=e.converged,this.iterationCount=e.iterationCount,this.connectivity=e.connectivity??4,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive safe integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count and sparse component roots must fit in uint32`);if(this.maximumIterations=e.maximumIterations??Math.max(1,Math.ceil(Math.log2(t))+2),this.connectivity!==4&&this.connectivity!==8)throw Error(`${this.id} connectivity must be four or eight`);if(!Number.isSafeInteger(this.maximumIterations)||this.maximumIterations<1||this.maximumIterations>Re)throw Error(`${this.id} maximum iterations must be an integer from one through 64`);if(this.input.storage.kind!==`buffer`||this.input.format!==`uint32`)throw Error(`${this.id} requires a buffer-backed uint32 foreground band`);if(this.input.scale!==void 0&&this.input.scale!==1||this.input.offset!==void 0&&this.input.offset!==0)throw Error(`${this.id} foreground masks require identity input calibration`);let n=O(this.input,this,`${this.id} input`);T(this.output,`uint32`,t,`${this.id} output labels`),S(this.outputValidity,t,`${this.id} output validity`),S(this.converged,1,`${this.id} convergence state`),this.iterationCount&&S(this.iterationCount,1,`${this.id} iteration count`);let r=this.getBorrowedViews();for(let[e,t]of r.entries()){if(t.buffer.graph!==n)throw Error(`${this.id} component inputs and outputs must belong to the same graph`);if(r.slice(e+1).some(e=>e.buffer===t.buffer))throw Error(`${this.id} component inputs and outputs must use separate buffers`)}}addToGraph(t){if(t.device.type!==`webgpu`)throw Error(`${this.id} connected-component labeling requires a WebGPU device`);for(let e of this.getBorrowedViews()){if(e.buffer.graph!==t)throw Error(`${this.id} resources must belong to the target graph`);w(t.device,e,`${this.id} ${e.buffer.id}`)}let n=E(t.device,this.width,this.height,this.id),r=this.width*this.height,i=u(t,`${this.id}-parents`,`uint32`,r),a=u(t,`${this.id}-changed`,`uint32`,1),o=u(t,`${this.id}-active-dispatch`,`uint32`,3,e.STORAGE|e.INDIRECT);w(t.device,i,`${this.id} atomic component parents`),w(t.device,a,`${this.id} change state`),w(t.device,o,`${this.id} indirect command`),this.addInitializationPass(t,i,a,o,n);for(let e=0;e<this.maximumIterations;e++)this.addHookPass(t,i,a,o,e),this.addCompressionPass(t,i,a,o,e),this.addConvergencePass(t,a,o,e);this.addPublicationPass(t,i,n)}addInitializationPass(e,t,n,r,i){let a=[{name:`sourceValues`,view:this.input.storage.values,usage:`storage-read`},{name:`parentValues`,view:t,usage:`storage-write`},{name:`outputValidity`,view:this.outputValidity,usage:`storage-write`},{name:`convergenceValues`,view:this.converged,usage:`storage-write`},{name:`changeValues`,view:n,usage:`storage-write`},{name:`dispatchValues`,view:r,usage:`storage-write`}];this.input.validity&&a.push({name:`sourceValidity`,view:this.input.validity,usage:`storage-read`}),this.iterationCount&&a.push({name:`iterationValues`,view:this.iterationCount,usage:`storage-write`}),Ve(e,`${this.id}-initialize`,a,e=>{let a=this.input.validity,s=this.iterationCount,c=a?`@group(0) @binding(${e.get(`sourceValidity`)}) var<storage, read> sourceValidity: array<u32>;
const SOURCE_VALIDITY_OFFSET: u32 = ${o(a)}u;`:``,l=a?` && sourceValidity[SOURCE_VALIDITY_OFFSET + pixelIndex] != 0u`:``,u=this.input.noDataValue===void 0?``:` && rawSample != ${D(this.input.noDataValue,`uint32`)}`,d=s?`@group(0) @binding(${e.get(`iterationValues`)}) var<storage, read_write> iterationValues: array<u32>;
const ITERATION_OFFSET: u32 = ${o(s)}u;`:``,f=s?`iterationValues[ITERATION_OFFSET] = 0u;`:``;return`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const SOURCE_OFFSET: u32 = ${o(this.input.storage.values)}u;
const PARENT_OFFSET: u32 = ${o(t)}u;
const VALIDITY_OFFSET: u32 = ${o(this.outputValidity)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
const CHANGED_OFFSET: u32 = ${o(n)}u;
const DISPATCH_OFFSET: u32 = ${o(r)}u;
@group(0) @binding(${e.get(`sourceValues`)}) var<storage, read> sourceValues: array<u32>;
@group(0) @binding(${e.get(`parentValues`)}) var<storage, read_write> parentValues: array<atomic<u32>>;
@group(0) @binding(${e.get(`outputValidity`)}) var<storage, read_write> outputValidity: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read_write> convergenceValues: array<u32>;
@group(0) @binding(${e.get(`changeValues`)}) var<storage, read_write> changeValues: array<atomic<u32>>;
@group(0) @binding(${e.get(`dispatchValues`)}) var<storage, read_write> dispatchValues: array<u32>;
${c}
${d}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let rawSample = sourceValues[SOURCE_OFFSET + pixelIndex];
  let observed = true${l}${u};
  let foreground = observed && rawSample != 0u;
  atomicStore(&parentValues[PARENT_OFFSET + pixelIndex], select(0u, pixelIndex + 1u, foreground));
  outputValidity[VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, observed);
  if (pixelIndex == 0u) {
    convergenceValues[CONVERGENCE_OFFSET] = 0u;
    atomicStore(&changeValues[CHANGED_OFFSET], 0u);
    dispatchValues[DISPATCH_OFFSET] = ${i[0]}u;
    dispatchValues[DISPATCH_OFFSET + 1u] = ${i[1]}u;
    dispatchValues[DISPATCH_OFFSET + 2u] = 1u;
    ${f}
  }
}`},i)}addHookPass(e,t,n,r,i){Ve(e,`${this.id}-hook-${i}`,[{name:`parentValues`,view:t,usage:`storage-read-write`},{name:`changeValues`,view:n,usage:`storage-read-write`}],e=>`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const PARENT_OFFSET: u32 = ${o(t)}u;
const CHANGED_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(${e.get(`parentValues`)}) var<storage, read_write> parentValues: array<atomic<u32>>;
@group(0) @binding(${e.get(`changeValues`)}) var<storage, read_write> changeValues: array<atomic<u32>>;

fn followRoot(label: u32) -> u32 {
  var root = label;
  for (var depth = 0u; depth < ${ze}u; depth++) {
    let parent = atomicLoad(&parentValues[PARENT_OFFSET + root - 1u]);
    if (parent == root) { break; }
    root = parent;
  }
  return root;
}

fn connectPixels(firstIndex: u32, secondIndex: u32) {
  let firstLabel = atomicLoad(&parentValues[PARENT_OFFSET + firstIndex]);
  let secondLabel = atomicLoad(&parentValues[PARENT_OFFSET + secondIndex]);
  if (firstLabel == 0u || secondLabel == 0u) { return; }
  let firstRoot = followRoot(firstLabel);
  let secondRoot = followRoot(secondLabel);
  if (firstRoot == secondRoot) { return; }
  let lowerRoot = min(firstRoot, secondRoot);
  let higherRoot = max(firstRoot, secondRoot);
  let previousRoot = atomicMin(&parentValues[PARENT_OFFSET + higherRoot - 1u], lowerRoot);
  if (previousRoot > lowerRoot) {
    atomicOr(&changeValues[CHANGED_OFFSET], 1u);
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  if (atomicLoad(&parentValues[PARENT_OFFSET + pixelIndex]) == 0u) { return; }
  if (globalId.x > 0u) { connectPixels(pixelIndex, pixelIndex - 1u); }
  if (globalId.y > 0u) {
    connectPixels(pixelIndex, pixelIndex - WIDTH);
    ${this.connectivity===8?`if (globalId.x > 0u) { connectPixels(pixelIndex, pixelIndex - WIDTH - 1u); }`:``}
    ${this.connectivity===8?`if (globalId.x + 1u < WIDTH) { connectPixels(pixelIndex, pixelIndex - WIDTH + 1u); }`:``}
  }
}`,void 0,r)}addCompressionPass(e,t,n,r,i){Ve(e,`${this.id}-compress-${i}`,[{name:`parentValues`,view:t,usage:`storage-read-write`},{name:`changeValues`,view:n,usage:`storage-read-write`}],e=>`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const PARENT_OFFSET: u32 = ${o(t)}u;
const CHANGED_OFFSET: u32 = ${o(n)}u;
@group(0) @binding(${e.get(`parentValues`)}) var<storage, read_write> parentValues: array<atomic<u32>>;
@group(0) @binding(${e.get(`changeValues`)}) var<storage, read_write> changeValues: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let initialParent = atomicLoad(&parentValues[PARENT_OFFSET + pixelIndex]);
  if (initialParent == 0u) { return; }
  var root = initialParent;
  for (var depth = 0u; depth < ${ze}u; depth++) {
    let parent = atomicLoad(&parentValues[PARENT_OFFSET + root - 1u]);
    if (parent == root) { break; }
    root = parent;
  }
  if (root < initialParent) {
    let previousRoot = atomicMin(&parentValues[PARENT_OFFSET + pixelIndex], root);
    if (previousRoot > root) {
      atomicOr(&changeValues[CHANGED_OFFSET], 1u);
    }
  }
}`,void 0,r)}addConvergencePass(e,t,n,r){let i=[{name:`changeValues`,view:t,usage:`storage-read-write`},{name:`dispatchValues`,view:n,usage:`storage-read-write`},{name:`convergenceValues`,view:this.converged,usage:`storage-read-write`}];this.iterationCount&&i.push({name:`iterationValues`,view:this.iterationCount,usage:`storage-read-write`}),Ve(e,`${this.id}-convergence-${r}`,i,e=>{let r=this.iterationCount,i=r?`@group(0) @binding(${e.get(`iterationValues`)}) var<storage, read_write> iterationValues: array<u32>;
const ITERATION_OFFSET: u32 = ${o(r)}u;`:``,a=r?`iterationValues[ITERATION_OFFSET] += 1u;`:``;return`
const CHANGED_OFFSET: u32 = ${o(t)}u;
const DISPATCH_OFFSET: u32 = ${o(n)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
@group(0) @binding(${e.get(`changeValues`)}) var<storage, read_write> changeValues: array<atomic<u32>>;
@group(0) @binding(${e.get(`dispatchValues`)}) var<storage, read_write> dispatchValues: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read_write> convergenceValues: array<u32>;
${i}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x != 0u || globalId.y != 0u || dispatchValues[DISPATCH_OFFSET] == 0u) {
    return;
  }
  ${a}
  if (atomicLoad(&changeValues[CHANGED_OFFSET]) == 0u) {
    convergenceValues[CONVERGENCE_OFFSET] = 1u;
    dispatchValues[DISPATCH_OFFSET] = 0u;
    dispatchValues[DISPATCH_OFFSET + 1u] = 0u;
    dispatchValues[DISPATCH_OFFSET + 2u] = 0u;
  } else {
    atomicStore(&changeValues[CHANGED_OFFSET], 0u);
  }
}`},[1,1])}addPublicationPass(e,t,n){Ve(e,`${this.id}-publish`,[{name:`parentValues`,view:t,usage:`storage-read`},{name:`convergenceValues`,view:this.converged,usage:`storage-read`},{name:`outputLabels`,view:this.output,usage:`storage-write`},{name:`outputValidity`,view:this.outputValidity,usage:`storage-read-write`}],e=>`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const PARENT_OFFSET: u32 = ${o(t)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
const OUTPUT_OFFSET: u32 = ${o(this.output)}u;
const VALIDITY_OFFSET: u32 = ${o(this.outputValidity)}u;
@group(0) @binding(${e.get(`parentValues`)}) var<storage, read> parentValues: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read> convergenceValues: array<u32>;
@group(0) @binding(${e.get(`outputLabels`)}) var<storage, read_write> outputLabels: array<u32>;
@group(0) @binding(${e.get(`outputValidity`)}) var<storage, read_write> outputValidity: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let accepted = convergenceValues[CONVERGENCE_OFFSET] != 0u;
  outputLabels[OUTPUT_OFFSET + pixelIndex] =
    select(0u, parentValues[PARENT_OFFSET + pixelIndex], accepted);
  outputValidity[VALIDITY_OFFSET + pixelIndex] =
    select(0u, outputValidity[VALIDITY_OFFSET + pixelIndex], accepted);
}`,n)}getBorrowedViews(){return[this.input.storage.values,...this.input.validity?[this.input.validity]:[],this.output,this.outputValidity,this.converged,...this.iterationCount?[this.iterationCount]:[]]}};function Ve(e,n,r,i,a,o){if(r.length>e.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${n} exceeds the device storage binding count`);let s=[],c=[],l=new Map;for(let[t,i]of r.entries()){if(i.view.buffer.graph!==e)throw Error(`${n} resources must belong to the target graph`);w(e.device,i.view,`${n} ${i.view.buffer.id}`),s.push({buffer:i.view,usage:i.usage}),c.push({name:i.name,type:i.usage===`storage-read`?`read-only-storage`:`storage`,group:0,location:t}),l.set(i.name,t)}o&&s.push({buffer:o.buffer,usage:`indirect`}),e.addComputePass({id:n,resources:s,compile:({device:e})=>{let s=new f(e,{id:n,source:i(l),shaderLayout:{bindings:c}});return{encode:({computePass:e,getBuffer:n})=>{let i={};for(let e of r)i[e.name]=t(e.view,n);s.setBindings(i),o?s.dispatchIndirect(e,n(o.buffer),o.byteOffset):a&&s.dispatch(e,a[0],a[1])},destroy:()=>s.destroy()}}})}var He=class{id;width;height;input;inputValidity;converged;output;outputValidity;componentCount;overflow;requiredComponentCount;capacity;constructor(e){if(this.id=e.id??`gpu-raster-dense-components`,this.width=e.width,this.height=e.height,this.input=e.input,this.inputValidity=e.inputValidity,this.converged=e.converged,this.output=e.output,this.outputValidity=e.outputValidity,this.componentCount=e.componentCount,this.overflow=e.overflow,this.requiredComponentCount=e.requiredComponentCount,!Number.isSafeInteger(this.width)||this.width<=0||!Number.isSafeInteger(this.height)||this.height<=0)throw Error(`${this.id} dimensions must be positive safe integers`);let t=this.width*this.height;if(!Number.isSafeInteger(t)||t>4294967295)throw Error(`${this.id} pixel count and dense component identifiers must fit in uint32`);if(this.capacity=e.capacity??t,!Number.isSafeInteger(this.capacity)||this.capacity<0||this.capacity>t)throw Error(`${this.id} component capacity must be an integer from zero through pixel count`);T(this.input,`uint32`,t,`${this.id} sparse labels`),S(this.inputValidity,t,`${this.id} sparse validity`),S(this.converged,1,`${this.id} convergence state`),T(this.output,`uint32`,t,`${this.id} dense labels`),S(this.outputValidity,t,`${this.id} dense validity`),S(this.componentCount,1,`${this.id} component count`),S(this.overflow,1,`${this.id} component overflow`),this.requiredComponentCount&&S(this.requiredComponentCount,1,`${this.id} required component count`);let n=this.getBorrowedViews(),r=this.input.buffer.graph;for(let[e,t]of n.entries()){if(t.buffer.graph!==r)throw Error(`${this.id} dense component inputs and outputs must belong to the same graph`);if(n.slice(e+1).some(e=>e.buffer===t.buffer))throw Error(`${this.id} dense component inputs and outputs must use separate buffers`)}}addToGraph(e){if(e.device.type!==`webgpu`)throw Error(`${this.id} dense component relabeling requires a WebGPU device`);for(let t of this.getBorrowedViews()){if(t.buffer.graph!==e)throw Error(`${this.id} resources must belong to the target graph`);w(e.device,t,`${this.id} ${t.buffer.id}`)}if(e.device.limits.maxComputeInvocationsPerWorkgroup<256||e.device.limits.maxComputeWorkgroupSizeX<256)throw Error(`${this.id} representative scan exceeds device workgroup limits`);let t=E(e.device,this.width,this.height,this.id),n=this.width*this.height,r=u(e,`${this.id}-root-flags`,`uint32`,n),i=u(e,`${this.id}-root-offsets`,`uint32`,n);w(e.device,r,`${this.id} representative flags`),w(e.device,i,`${this.id} representative offsets`),this.addMarkPass(e,r,t),e.add(new p({id:`${this.id}-scan`,input:r,output:i,mode:`exclusive`})),this.addScatterPass(e,r,i,t),this.addPublicationPass(e,r,i)}addMarkPass(e,t,n){Ue(e,`${this.id}-mark-roots`,[{name:`sparseLabels`,view:this.input,usage:`storage-read`},{name:`sparseValidity`,view:this.inputValidity,usage:`storage-read`},{name:`convergenceValues`,view:this.converged,usage:`storage-read`},{name:`rootFlags`,view:t,usage:`storage-write`}],e=>`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const INPUT_OFFSET: u32 = ${o(this.input)}u;
const VALIDITY_OFFSET: u32 = ${o(this.inputValidity)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
const ROOT_OFFSET: u32 = ${o(t)}u;
@group(0) @binding(${e.get(`sparseLabels`)}) var<storage, read> sparseLabels: array<u32>;
@group(0) @binding(${e.get(`sparseValidity`)}) var<storage, read> sparseValidity: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read> convergenceValues: array<u32>;
@group(0) @binding(${e.get(`rootFlags`)}) var<storage, read_write> rootFlags: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let isRepresentative = convergenceValues[CONVERGENCE_OFFSET] != 0u &&
    sparseValidity[VALIDITY_OFFSET + pixelIndex] != 0u &&
    sparseLabels[INPUT_OFFSET + pixelIndex] == pixelIndex + 1u;
  rootFlags[ROOT_OFFSET + pixelIndex] = select(0u, 1u, isRepresentative);
}`,n)}addScatterPass(e,t,n,r){Ue(e,`${this.id}-scatter`,[{name:`sparseLabels`,view:this.input,usage:`storage-read`},{name:`sparseValidity`,view:this.inputValidity,usage:`storage-read`},{name:`convergenceValues`,view:this.converged,usage:`storage-read`},{name:`rootFlags`,view:t,usage:`storage-read`},{name:`rootOffsets`,view:n,usage:`storage-read`},{name:`denseLabels`,view:this.output,usage:`storage-write`},{name:`denseValidity`,view:this.outputValidity,usage:`storage-write`}],e=>`
const WIDTH: u32 = ${this.width}u;
const HEIGHT: u32 = ${this.height}u;
const PIXEL_COUNT: u32 = ${this.width*this.height}u;
const CAPACITY: u32 = ${this.capacity}u;
const INPUT_OFFSET: u32 = ${o(this.input)}u;
const INPUT_VALIDITY_OFFSET: u32 = ${o(this.inputValidity)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
const ROOT_OFFSET: u32 = ${o(t)}u;
const SCAN_OFFSET: u32 = ${o(n)}u;
const OUTPUT_OFFSET: u32 = ${o(this.output)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${o(this.outputValidity)}u;
@group(0) @binding(${e.get(`sparseLabels`)}) var<storage, read> sparseLabels: array<u32>;
@group(0) @binding(${e.get(`sparseValidity`)}) var<storage, read> sparseValidity: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read> convergenceValues: array<u32>;
@group(0) @binding(${e.get(`rootFlags`)}) var<storage, read> rootFlags: array<u32>;
@group(0) @binding(${e.get(`rootOffsets`)}) var<storage, read> rootOffsets: array<u32>;
@group(0) @binding(${e.get(`denseLabels`)}) var<storage, read_write> denseLabels: array<u32>;
@group(0) @binding(${e.get(`denseValidity`)}) var<storage, read_write> denseValidity: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  let sparseLabel = sparseLabels[INPUT_OFFSET + pixelIndex];
  var observationIsValid = convergenceValues[CONVERGENCE_OFFSET] != 0u &&
    sparseValidity[INPUT_VALIDITY_OFFSET + pixelIndex] != 0u;
  var denseLabel = 0u;
  if (observationIsValid && sparseLabel != 0u) {
    if (sparseLabel <= PIXEL_COUNT) {
      let rootIndex = sparseLabel - 1u;
      if (rootFlags[ROOT_OFFSET + rootIndex] != 0u) {
        denseLabel = rootOffsets[SCAN_OFFSET + rootIndex] + 1u;
        observationIsValid = denseLabel <= CAPACITY;
      } else {
        observationIsValid = false;
      }
    } else {
      observationIsValid = false;
    }
  }
  denseLabels[OUTPUT_OFFSET + pixelIndex] = select(0u, denseLabel, observationIsValid);
  denseValidity[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, observationIsValid);
}`,r)}addPublicationPass(e,t,n){let r=[{name:`rootFlags`,view:t,usage:`storage-read`},{name:`rootOffsets`,view:n,usage:`storage-read`},{name:`convergenceValues`,view:this.converged,usage:`storage-read`},{name:`componentCounts`,view:this.componentCount,usage:`storage-write`},{name:`overflowValues`,view:this.overflow,usage:`storage-write`}];this.requiredComponentCount&&r.push({name:`requiredComponentCounts`,view:this.requiredComponentCount,usage:`storage-write`}),Ue(e,`${this.id}-publish`,r,e=>{let r=this.requiredComponentCount,i=r?`@group(0) @binding(${e.get(`requiredComponentCounts`)}) var<storage, read_write> requiredComponentCounts: array<u32>;
const REQUIRED_OFFSET: u32 = ${o(r)}u;`:``,a=r?`requiredComponentCounts[REQUIRED_OFFSET] = exactCount;`:``;return`
const LAST_INDEX: u32 = ${this.width*this.height-1}u;
const CAPACITY: u32 = ${this.capacity}u;
const ROOT_OFFSET: u32 = ${o(t)}u;
const SCAN_OFFSET: u32 = ${o(n)}u;
const CONVERGENCE_OFFSET: u32 = ${o(this.converged)}u;
const COUNT_OFFSET: u32 = ${o(this.componentCount)}u;
const OVERFLOW_OFFSET: u32 = ${o(this.overflow)}u;
@group(0) @binding(${e.get(`rootFlags`)}) var<storage, read> rootFlags: array<u32>;
@group(0) @binding(${e.get(`rootOffsets`)}) var<storage, read> rootOffsets: array<u32>;
@group(0) @binding(${e.get(`convergenceValues`)}) var<storage, read> convergenceValues: array<u32>;
@group(0) @binding(${e.get(`componentCounts`)}) var<storage, read_write> componentCounts: array<u32>;
@group(0) @binding(${e.get(`overflowValues`)}) var<storage, read_write> overflowValues: array<u32>;
${i}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x != 0u || globalId.y != 0u) { return; }
  let totalRoots = rootOffsets[SCAN_OFFSET + LAST_INDEX] + rootFlags[ROOT_OFFSET + LAST_INDEX];
  let exactCount = select(0u, totalRoots, convergenceValues[CONVERGENCE_OFFSET] != 0u);
  componentCounts[COUNT_OFFSET] = min(exactCount, CAPACITY);
  overflowValues[OVERFLOW_OFFSET] = select(0u, 1u, exactCount > CAPACITY);
  ${a}
}`},[1,1])}getBorrowedViews(){return[this.input,this.inputValidity,this.converged,this.output,this.outputValidity,this.componentCount,this.overflow,...this.requiredComponentCount?[this.requiredComponentCount]:[]]}};function Ue(e,n,r,i,a){if(r.length>e.device.limits.maxStorageBuffersPerShaderStage)throw Error(`${n} exceeds the device storage binding count`);let o=[],s=[],c=new Map;for(let[t,i]of r.entries()){if(i.view.buffer.graph!==e)throw Error(`${n} resources must belong to the target graph`);w(e.device,i.view,`${n} ${i.view.buffer.id}`),o.push({buffer:i.view,usage:i.usage}),s.push({name:i.name,type:i.usage===`storage-read`?`read-only-storage`:`storage`,group:0,location:t}),c.set(i.name,t)}e.addComputePass({id:n,resources:o,compile:({device:e})=>{let o=new f(e,{id:n,source:i(c),shaderLayout:{bindings:s}});return{encode:({computePass:e,getBuffer:n})=>{let i={};for(let e of r)i[e.name]=t(e.view,n);o.setBindings(i),o.dispatch(e,a[0],a[1])},destroy:()=>o.destroy()}}})}var We=[`dnbr`,`nbr-before`,`nbr-after`,`ndvi-before`,`ndvi-after`,`dndvi`],Q=8192,Ge=6,Ke=Ge+1,qe=1024,$=.04,Je=350,Ye=[`.`,`:`,`-`,`=`,`+`,`*`,`#`,`@`],Xe=[`pixelCounts`,`areas`,`perimeters`,`minColumns`,`minRows`,`maxColumns`,`maxRows`],Ze=[`dnbr`,`nbrBefore`,`nbrAfter`,`ndviBefore`,`ndviAfter`,`dndvi`,`burned`];async function Qe(e){let t=e.datasets.get(`dixie-fire`),{device:r}=e;e.setStatus(`Loading Sentinel-2 bands`);let i=await _e(t,e.signal);e.signal.throwIfAborted();let a=he(t),{width:o,height:l,cellCount:u,frame:d}=i,f=new c(r,`burn`);e.setReadout(`grid`,`${o} × ${l} px, 20 m`);let p=ue({width:o,height:l,cellSize:d.cellSize,origin:[-d.halfWidth,d.halfHeight]}),m={redBefore:f.createBuffer(`red-before`,i.redBefore),nirBefore:f.createBuffer(`nir-before`,i.nirBefore),swirBefore:f.createBuffer(`swir-before`,i.swirBefore),redAfter:f.createBuffer(`red-after`,i.redAfter),nirAfter:f.createBuffer(`nir-after`,i.nirAfter),swirAfter:f.createBuffer(`swir-after`,i.swirAfter),scl:f.createBuffer(`scl`,i.sclAfter)},h=e=>f.createBuffer(e,u*4),g=(e,t=u)=>f.createBuffer(e,t*4),_={nbrBefore:h(`nbr-before`),nbrAfter:h(`nbr-after`),ndviBefore:h(`ndvi-before`),ndviAfter:h(`ndvi-after`),dnbr:h(`dnbr`),dndvi:h(`dndvi`),cloudFlag:h(`cloud-flag`),cloudMask:g(`cloud-mask`),clean:h(`clean-dnbr`),burnedFlag:h(`burned-flag`),burnedMask:g(`burned-mask`),patchMask:g(`patch-mask`),classes:g(`classes`),classCounts:g(`class-counts`,Ke),stretched:h(`stretched`),stretchStatistics:f.createBuffer(`stretch-statistics`,32),histogram:g(`histogram`,qe)},v={sparse:g(`sparse`),sparseValidity:g(`sparse-validity`),sparseConverged:g(`sparse-converged`,1),iterations:g(`iterations`,1),labels:g(`labels`),labelValidity:g(`label-validity`),componentCount:g(`component-count`,1),overflow:g(`overflow`,1),columns:{pixelCounts:g(`pixel-counts`,Q),areas:g(`areas`,Q),perimeters:g(`perimeters`,Q),minColumns:g(`min-columns`,Q),minRows:g(`min-rows`,Q),maxColumns:g(`max-columns`,Q),maxRows:g(`max-rows`,Q)},sieves:{remove:{labels:g(`remove-labels`),targets:g(`remove-targets`,Q),count:g(`remove-count`,1)},merge:{labels:g(`merge-labels`),targets:g(`merge-targets`,Q),count:g(`merge-count`,1)}}},y=Ze.map(e=>({count:f.createBuffer(`${e}-count`,4),sum:f.createBuffer(`${e}-sum`,4),mean:f.createBuffer(`${e}-mean`,4),extent:f.createBuffer(`${e}-extent`,8)})),b=f.createParameterBuffer(`index-parameters`,`float32`,8),x=f.createParameterBuffer(`change-parameters`,`float32`,8),S=f.createParameterBuffer(`cloud-parameters`,`float32`,8),C=f.createParameterBuffer(`clean-parameters`,`float32`,8),w=f.createParameterBuffer(`burned-parameters`,`float32`,8),T=f.createParameterBuffer(`patch-mask-parameters`,`float32`,8),E=f.createParameterBuffer(`reclassify-parameters`,`float32`,4),D=f.createParameterBuffer(`breaks`,`float32`,Ge),O=f.createParameterBuffer(`stretch-parameters`,`float32`,16),k=f.createParameterBuffer(`sieve-parameters`,`uint32`,4),A=(e,t)=>({id:e,format:`float32`,storage:{kind:`buffer`,values:t}}),j=new n(r,{id:`burn-main`});{let e=ge(j),t=(t,n)=>e(t,n,`float32`,u),n=(t,n,r=u)=>e(t,n,`uint32`,r),r=b.importToGraph(j),i=x.importToGraph(j),a=[[`nbr-before`,m.nirBefore,m.swirBefore,_.nbrBefore],[`nbr-after`,m.nirAfter,m.swirAfter,_.nbrAfter],[`ndvi-before`,m.nirBefore,m.redBefore,_.ndviBefore],[`ndvi-after`,m.nirAfter,m.redAfter,_.ndviAfter]];for(let[e,n,i,o]of a)j.add(new le({id:e,cellCount:u,a:t(`${e}-a`,n),b:t(`${e}-b`,i),noDataValue:0,parameters:r,output:{values:t(`${e}-out`,o)}}));for(let[e,n,r,a]of[[`dnbr`,_.nbrBefore,_.nbrAfter,_.dnbr],[`dndvi`,_.ndviBefore,_.ndviAfter,_.dndvi]])j.add(new le({id:e,cellCount:u,a:t(`${e}-before`,n),b:t(`${e}-after`,r),parameters:i,output:{values:t(`${e}-out`,a)}}));j.add(new ae({id:`cloud`,cellCount:u,conditionValues:t(`scl`,m.scl),parameters:S.importToGraph(j),output:{values:t(`cloud-flag`,_.cloudFlag),outputMask:n(`cloud-mask`,_.cloudMask)}})),j.add(new ae({id:`clean`,cellCount:u,mask:n(`cloud-mask-in`,_.cloudMask),b:t(`dnbr-in`,_.dnbr),parameters:C.importToGraph(j),output:{values:t(`clean-out`,_.clean)}})),j.add(new ae({id:`burned`,cellCount:u,conditionValues:t(`clean-in`,_.clean),parameters:w.importToGraph(j),output:{values:t(`burned-flag`,_.burnedFlag),outputMask:n(`burned-mask`,_.burnedMask)}})),j.add(new ae({id:`patch-mask`,cellCount:u,conditionValues:t(`burned-flag-in`,_.burnedFlag),parameters:T.importToGraph(j),output:{values:t(`patch-flag`,h(`patch-flag`)),outputMask:n(`patch-mask`,_.patchMask)}})),j.add(new oe({id:`severity`,values:t(`clean-classify`,_.clean),breaks:D.importToGraph(j),parameters:E.importToGraph(j),output:{classes:n(`classes`,_.classes),classCounts:n(`class-counts`,_.classCounts,Ke)}}))}let ee=f.track(await j.compileAsync()),te={dnbr:_.clean,"nbr-before":_.nbrBefore,"nbr-after":_.nbrAfter,"ndvi-before":_.ndviBefore,"ndvi-after":_.ndviAfter,dndvi:_.dndvi},M={};for(let e of We){let t=new n(r,{id:`burn-stretch-${e}`}),i=ge(t);t.add(new ke({id:`stretch-${e}`,values:i(`values`,te[e],`float32`,u),grid:p,binCount:qe,parameters:O.importToGraph(t),output:{stretched:i(`stretched`,_.stretched,`float32`,u),statistics:i(`statistics`,_.stretchStatistics,`float32`,8),histogram:i(`histogram`,_.histogram,`uint32`,qe)}})),M[e]=f.track(await t.compileAsync())}let ne=new n(r,{id:`burn-statistics`});{let e=ge(ne);[_.clean,_.nbrBefore,_.nbrAfter,_.ndviBefore,_.ndviAfter,_.dndvi,_.burnedFlag].forEach((t,n)=>{let r=Ze[n],i=y[n];new Le({id:`stats-${r}`,width:o,height:l,input:A(r,e(`${r}-values`,t,`float32`,u)),count:e(`${r}-count`,i.count,`uint32`,1),sum:e(`${r}-sum`,i.sum,`float32`,1),mean:e(`${r}-mean`,i.mean,`float32`,1),extent:e(`${r}-extent`,i.extent,`float32`,2)}).addToGraph(ne)})}let N=f.track(await ne.compileAsync()),re=async e=>{let t=new n(r,{id:`burn-patches-${e}`}),i=ge(t),a=(e,t,n=u)=>i(e,t,`uint32`,n),s=a(`sparse`,v.sparse),c=a(`sparse-validity`,v.sparseValidity),d=a(`sparse-converged`,v.sparseConverged,1);new Be({id:`components`,width:o,height:l,input:{id:`patch-mask`,format:`uint32`,storage:{kind:`buffer`,values:a(`patch-mask`,_.patchMask)}},output:s,outputValidity:c,converged:d,iterationCount:a(`iterations`,v.iterations,1),connectivity:e}).addToGraph(t);let m=a(`labels`,v.labels),h=a(`label-validity`,v.labelValidity),g=a(`component-count`,v.componentCount,1),y=a(`overflow`,v.overflow,1);new He({id:`dense`,width:o,height:l,input:s,inputValidity:c,converged:d,output:m,outputValidity:h,componentCount:g,overflow:y,capacity:Q}).addToGraph(t);let b={grid:p,labels:m,validity:h,status:{converged:d,componentCount:g,overflow:y}},x=(e,t)=>i(e,v.columns[e],t,Q);t.add(new ye({...b,id:`metrics`,output:{pixelCounts:x(`pixelCounts`,`uint32`),areas:x(`areas`,`float32`),perimeters:x(`perimeters`,`float32`),minColumns:x(`minColumns`,`uint32`),minRows:x(`minRows`,`uint32`),maxColumns:x(`maxColumns`,`uint32`),maxRows:x(`maxRows`,`uint32`)}}));let S=k.importToGraph(t);for(let n of[`remove`,`merge`]){let r=v.sieves[n];t.add(new Ce({...b,id:`sieve-${n}`,patchCapacity:Q,parameters:S,mode:n,connectivity:e,output:{labels:a(`${n}-labels`,r.labels),patchTargets:a(`${n}-targets`,r.targets,Q),sievedCount:a(`${n}-count`,r.count,1)}}))}return f.track(await t.compileAsync())},P={4:await re(4),8:await re(8)},R=!1,z=!0,B=!0,V=!0,H=!0,U=performance.now(),W=``,G=0,K=null,q=null,xe=null,Se={},we=()=>{z=!0,U=performance.now()},Te=e=>e.classScheme===`usgs`?{values:[-.25,-.1,.1,.27,.44,.66],count:6}:e.classScheme===`binary`?{values:[e.moderateBreak],count:1}:{values:[e.lowBreak,e.moderateBreak,e.highBreak].sort((e,t)=>e-t),count:3},J=e=>e===`dnbr`||e===`dndvi`?[-.5,1.5]:[-1,1],De=e=>{let t=1/0,n=1/0,r=-1/0,i=-1/0;for(let[a,o]of[[0,0],[e.width,0],[e.width,e.height],[0,e.height]]){let[s,c]=e.unproject([a,o]),[l,u]=d.lngLatToCell(s,c);t=Math.min(t,l),r=Math.max(r,l),n=Math.min(n,u),i=Math.max(i,u)}let a=(e,t)=>Math.min(Math.max(Math.floor(e),0),t);return[a(t,o-1),a(n,l-1),a(r,o-1)+1,a(i,l-1)+1]},Oe=t=>{let n=e.options;b.write(se({operation:n.indexFormula,scaleA:1e-4,scaleB:1e-4})),x.write(se({operation:n.changeOperation})),S.write(ie({comparison:`>=`,threshold:n.maskClouds?n.cloudClass:99,constantA:1,constantB:0})),C.write(ie({constantA:NaN})),w.write(ie({comparison:n.comparison,threshold:n.threshold,upperThreshold:n.upperThreshold,constantA:1,constantB:0})),T.write(ie({comparison:`==`,threshold:n.patchTarget===`burned`?1:0}));let{values:r,count:i}=Te(n);D.write(Float32Array.from({length:Ge},(e,t)=>r[t]??1e9)),E.write(ce({breakCount:i,closed:n.closed}));let a=$e(n.layer);O.write(Ee({window:t,domain:n.domain===`fixed`?J(a):`auto`,mode:n.stretchMode,percentiles:[Math.min(n.percentiles[0],n.percentiles[1]),Math.max(...n.percentiles)],gamma:n.gamma,sigmoidContrast:n.sigmoidContrast,sigmoidMidpoint:n.sigmoidMidpoint})),k.write(be({minimumPixels:Math.max(1,Math.round(n.minimumPatchHa/$))}))},Y=new de(f,`burn-summary`,[{buffer:_.stretchStatistics,size:32},{buffer:_.histogram,size:qe*4},{buffer:_.classCounts,size:Ke*4},...y.flatMap(e=>[{buffer:e.count,size:4},{buffer:e.sum,size:4},{buffer:e.mean,size:4},{buffer:e.extent,size:8}]),{buffer:v.componentCount,size:4},{buffer:v.sparseConverged,size:4},{buffer:v.overflow,size:4},{buffer:v.sieves.remove.count,size:4},{buffer:v.sieves.merge.count,size:4},...Xe.map(e=>({buffer:v.columns[e],size:Q*4}))],e=>{R||(K=Ae(e),Fe())}),X=new de(f,`burn-inspect`,[{buffer:_.clean,size:u*4},{buffer:_.classes,size:u*4}],e=>{R||(q=new Float32Array(e,0,u),xe=new Uint32Array(e,u*4,u))}),Z={};for(let e of[`remove`,`merge`])Z[e]=new de(f,`burn-labels-${e}`,[{buffer:v.sieves[e].labels,size:u*4}],t=>{R||(Se[e]=new Uint32Array(t),G!==0&&Fe())});function Ae(e){let t=0,n=(n,r)=>{let i=new n(e,t,r);return t+=r*4,i},r=n(Float32Array,8),i=n(Uint32Array,qe),a=n(Uint32Array,Ke),o=Ze.map(()=>{let e=n(Uint32Array,1)[0];n(Float32Array,1);let t=n(Float32Array,1)[0],r=n(Float32Array,2);return{count:e,mean:t,minimum:r[0],maximum:r[1]}}),s=n(Uint32Array,5),c={pixelCounts:n(Uint32Array,Q),areas:n(Float32Array,Q),perimeters:n(Float32Array,Q),minColumns:n(Uint32Array,Q),minRows:n(Uint32Array,Q),maxColumns:n(Uint32Array,Q),maxRows:n(Uint32Array,Q)};return{stretch:r,histogram:i,classCounts:a,statistics:o,componentCount:s[0],converged:s[1],overflow:s[2],sievedRemove:s[3],sievedMerge:s[4],columns:c}}function je(e){return e.classScheme===`binary`?[`Below break`,`Burned`]:e.classScheme===`custom`?[`Unburned`,`Low`,`Moderate`,`High`]:[...fe]}function Me(e,t,n,r=``){if(!Number.isFinite(e))return`n/a`;let i=Math.abs(e-t),a=i<5*10**-(n+1)?`match`:`Δ ${i.toExponential(1)}`;return`${I(e,n)}${r} · CPU ${I(t,n)}${r} (${a})`}function Ne(e){if(!K)return null;let{columns:t,componentCount:n}=K,r=Math.min(n,Q),i=Math.max(1,Math.round(e.minimumPatchHa/$)),a=0,o=0,s=0,c=0,l=-1,u=0;for(let e=0;e<r;e++){let n=t.pixelCounts[e];u+=n,!(n<i)&&(a++,o+=n,s+=t.perimeters[e],n>c&&(c=n,l=e))}return{rows:r,kept:a,keptPixels:o,perimeter:s,largest:c,largestRow:l,totalPixels:u,minimumPixels:i}}function Pe(e){if(!K||e<1||e>Q)return`none`;let t=e-1,{columns:n}=K,r=n.pixelCounts[t];if(r===0)return`removed by the sieve`;let i=r*$,a=n.perimeters[t],o=(n.maxColumns[t]-n.minColumns[t]+1)*.02,s=(n.maxRows[t]-n.minRows[t]+1)*.02;return`#${e}: ${F(i)}, perimeter ${(a/1e3).toFixed(2)} km, box ${o.toFixed(1)} × ${s.toFixed(1)} km`}function Fe(){if(!K)return;let t=e.options,n=t.indexFormula===`normalizedDifference`&&t.changeOperation===`subtract`,[r,i,o,c,l,d,f]=K.statistics;e.setReadout(`validCells`,`${s(r.count)} of ${s(u)}`),e.setReadout(`meanDnbr`,n?Me(r.mean,a.meanDNBR,4):`${I(r.mean,4)} (the CPU reference needs the defaults)`);let p=(e,t,r,i)=>n?`${I(e,3)} / ${I(t,3)} (CPU ${I(r,3)} / ${I(i,3)})`:`${I(e,3)} / ${I(t,3)}`;e.setReadout(`meanNbr`,p(i.mean,o.mean,a.meanNBRBefore,a.meanNBRAfter)),e.setReadout(`meanNdvi`,p(c.mean,l.mean,a.meanNDVIBefore,a.meanNDVIAfter)),e.setReadout(`meanDndvi`,n?Me(d.mean,a.meanDNDVI,4):I(d.mean,4));let m=t.comparison===`>`&&Math.abs(t.threshold-.27)<1e-6;e.setReadout(`burnedShare`,n&&m&&t.maskClouds?`${L(f.mean,2)} · CPU ${L(a.moderateFraction,2)}`:`${L(f.mean,2)} (reference at dNBR > 0.27)`);let h=K.classCounts,g=je(t),_=g.length,v=0;for(let e=0;e<_;e++)v+=h[e];let y=v>0?h[_-1]/v:NaN;e.setReadout(`highShare`,n&&t.classScheme===`usgs`&&t.closed===`left`?`${L(y,2)} · CPU ${L(a.highFraction,2)}`:`${L(y,2)} (top class of this scheme)`),e.setReadout(`classShares`,g.map((e,t)=>`${e.split(`,`)[0].replace(` severity`,``)} ${L(v>0?h[t]/v:0,1)}`).join(` · `));let[b,x,S,C,w]=K.stretch;e.setReadout(`stretchRange`,`${I(S,3)} to ${I(C,3)} (data ${I(b,2)} to ${I(x,2)}, ${s(w)} cells)`),e.setLegendExtent(`index`,[S,C]),e.setReadout(`histogram`,Ie(K.histogram));let T=Ne(t);if(T){let n=K.converged?``:` (labelling did not converge)`,r=K.overflow?` · overflow: more than ${s(Q)} patches`:``;e.setReadout(`patchCount`,`${s(T.kept)} of ${s(T.rows)}${r}${n}`),e.setReadout(`patchArea`,`${F(T.keptPixels*$)} of ${F(T.totalPixels*$)}`),e.setReadout(`largestPatch`,T.largestRow>=0?F(T.largest*$):`none`),e.setReadout(`edgeDensity`,T.keptPixels>0?`${(T.perimeter/(T.keptPixels*$)).toFixed(1)} m / ha`:`n/a`);let i=t.sieveMode===`remove`?K.sievedRemove:K.sievedMerge;e.setReadout(`sieved`,`${s(i)} patches under ${F(T.minimumPixels*$)}`)}e.setReadout(`selectedPatch`,G?Pe(G):`click a patch`)}function Ie(e){let t=Array(40).fill(0),n=e.length/40;for(let r=0;r<e.length;r++)t[Math.min(39,Math.floor(r/n))]+=e[r];let r=Math.log1p(Math.max(...t));return r<=0?`no data`:t.map(e=>Ye[Math.min(Ye.length-1,Math.floor(Math.log1p(e)/r*(Ye.length-1)))]).join(``)}let Re=e=>{if(!e)return-1;let[t,n]=d.lngLatToCell(e[0],e[1]),r=Math.floor(t),i=Math.floor(n);return r<0||i<0||r>=o||i>=l?-1:i*o+r},ze=()=>Se[e.options.sieveMode];return e.setReadout(`selectedPatch`,`click a patch`),e.setStatus(``),{getCompiledGraphs:()=>[ee,...We.map(e=>M[e]),N,P[4],P[8]],setOption(t){we(),B=!0,V=!0,H=!0,e.requestLayers()},encode(t,n){let r=e.options,i;if(r.stretchToView){i=De(n.viewport);let e=i.join(`,`);e!==W&&(W=e,we())}else W!==``&&(W=``,we());(z||n.frameIndex<2)&&(Oe(i),ee.encode(t,{parameters:void 0}),M[$e(r.layer)].encode(t,{parameters:void 0}),N.encode(t,{parameters:void 0}),P[r.connectivity===`8`?8:4].encode(t,{parameters:void 0}),z=!1,B=!0,V=!0,H=!0),performance.now()-U>Je&&(B&&!Y.isPending&&(Y.request(t),B=!1),V&&!X.isPending&&(X.request(t),V=!1),H&&!Z[r.sieveMode].isPending&&(Z[r.sieveMode].request(t),H=!1)),Y.flush(t),X.flush(t),Z[r.sieveMode].flush(t)},getLayers(){let t=e.options,n={id:`burn-${t.layer}`,frame:d,opacity:t.opacity,noDataColor:[0,0,0,0]},r;if(t.layer===`severity`){let e=t.classScheme===`usgs`?pe:t.classScheme===`custom`?[2,3,5,6].map(e=>pe[e]):[2,6].map(e=>pe[e]);r=new ve({...n,values:_.classes,valueFormat:`uint32`,colormap:`category`,palette:e})}else r=t.layer===`burned`?new ve({...n,values:_.burnedMask,valueFormat:`uint32`,colormap:`mask`,color:[255,90,40,215]}):t.layer===`patches`?new ve({...n,values:v.sieves[t.sieveMode].labels,valueFormat:`uint32`,colormap:`category`,palette:me,noDataValue:0}):new ve({...n,values:_.stretched,valueFormat:`float32`,colormap:t.ramp,valueRange:[0,1]});return[r]},getTooltip(t){let n=Re(t.coordinate);if(n<0)return null;let r=e.options,a=(e,t)=>e[n]>0&&t[n]>0?(e[n]-t[n])/(e[n]+t[n]):NaN,o=a(i.nirBefore,i.swirBefore),s=a(i.nirAfter,i.swirAfter),c=[],l=q?q[n]:o-s;if(c.push(`dNBR ${I(l,3)}  (NBR ${I(o,2)} → ${I(s,2)})`),xe&&r.layer===`severity`){let e=xe[n],t=je(r);c.push(e<t.length?t[e]:`No data`)}if(r.layer===`patches`){let e=ze()?.[n]??0;c.push(e?`Patch ${Pe(e)}`:`Not in a kept patch`)}return c.push(`${Math.round(i.elevation[n]).toLocaleString(`en-US`)} m elevation`),c.join(`
`)},onClick(t){if(e.options.layer!==`patches`)return!1;let n=Re(t.coordinate);return n<0?!1:(G=ze()?.[n]??0,Fe(),!0)},onThemeChange(){e.requestLayers()},destroy(){R=!0,Y.stop(),X.stop();for(let e of Object.values(Z))e.stop();f.destroy()}}}function $e(e){return We.includes(e)?e:`dnbr`}export{Qe as createBurnSeverity};