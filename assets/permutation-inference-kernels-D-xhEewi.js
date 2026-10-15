import{d as e,j as t,w as n}from"./resources-CwvhEOBe.js";import{t as r}from"./gpu-scan-BuGhTLDQ.js";import{n as i,s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as o}from"./sorted-segment-sums-DwGkmbx2.js";import{c as s,o as c,r as l}from"./spatial-weights-D4bd9AQ6.js";import{n as u,t as d}from"./float32-values-C3OXugIo.js";import{t as f}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";var p=`two-sided`;function m(e,t){let n=t??`two-sided`;if(![`directed`,`two-sided`,`greater`,`lesser`,`folded`].includes(n))throw Error(`${e} alternative must be 'directed', 'two-sided', 'greater', 'lesser' or 'folded', got ${String(t)}`);return n}function h(e){switch(e){case`greater`:return`greater`;case`lesser`:return`lesser`;case`two-sided`:return`min(greater, lesser)`;case`folded`:return`folded`;default:return`min(greater, lesser)`}}function g(e){return e===`two-sided`?2:1}function _(t,n,r,i){if(n!==void 0){if(e(t,`stopAfterExceedances`,n,1,i),r===`folded`)throw Error(`${t} stopAfterExceedances does not support the folded alternative`);return n}}function v(t,n={}){let{seed:r,permutationCount:i}=t;if(!Number.isSafeInteger(r)||r<0)throw Error(`permutation seed must be a non-negative safe integer`);if(e(`getGPUPermutationParameterValues`,`permutationCount`,i,1,2**31-1),n.maximumPermutations!==void 0&&i>n.maximumPermutations)throw Error(`permutationCount exceeds maximumPermutations`);let a=t.significanceLevel??.05;if(!(a>0&&a<1))throw Error(`significanceLevel must be in (0, 1)`);let o=new Uint32Array(4);return o[0]=r%2**32,o[1]=Math.floor(r/2**32),o[2]=i,o[3]=new Uint32Array(new Float32Array([a]).buffer)[0],o}var y=4096,b=4294967295,x={count:0,sumX:1,sumY:2,sumSquares:3,sumSquaresY:4,s0:5,mean:6,meanY:7,s0Squared:8};function S(n){let{id:r}=n,i=s(r,n.weights);if(e(r,`weights row count`,i,2,2**24-1),u(n.values,`${r} values`)!==i)throw Error(`${r} values length must equal the weights row count`);if(n.secondValues&&u(n.secondValues,`${r} secondValues`)!==i)throw Error(`${r} secondValues length must equal the weights row count`);if(n.mask&&(t(n.mask,`${r} mask`),n.mask.length!==i))throw Error(`${r} mask length must equal the weights row count`);return n.islandCount&&(t(n.islandCount,`${r} output.islandCount`),e(r,`output.islandCount length`,n.islandCount.length)),n.weights.status?.overflow&&t(n.weights.status.overflow,`${r} weights.status.overflow`),t(n.parameters,`${r} parameters`),e(r,`parameters length`,n.parameters.length,4),e(r,`maximumPermutations`,n.maximumPermutations,1,2**20),i}function C(e){return`
const MAXIMUM_PERMUTATIONS: u32 = ${e}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn isPermutationCountValid() -> bool {
  let requested = parameters[parametersOffset + 2u];
  return requested >= 1u && requested <= MAXIMUM_PERMUTATIONS;
}
fn readValidPermutationCount() -> u32 {
  return select(0u, parameters[parametersOffset + 2u], isPermutationCountValid());
}
fn readSignificanceLevel() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`}var w=f;function T(e,t){let{id:o,operation:s,weights:u,mask:f}=t,p=d(e,`${o}-values`,t.values),m=t.secondValues?d(e,`${o}-second-values`,t.secondValues):void 0,h=p.view,g=m?.view,_=(t.islands??`exclude`)===`keep`,v=c(u),x=u.status?.overflow,S=h.length,C=Math.ceil(S/y),T=n(e,`${o}-matrix`,`float32`,8*S),D=n(e,`${o}-flags`,`uint32`,S),O=n(e,`${o}-analysis-flags`,`uint32`,S),k=n(e,`${o}-scanned`,`uint32`,S),A=n(e,`${o}-row-positions`,`uint32`,S+1),j=n(e,`${o}-compact-x`,`float32`,S),M=n(e,`${o}-compact-y`,`float32`,S),N=t.islandCount??n(e,`${o}-island-count`,`uint32`,1),P=v?n(e,`${o}-scaled-weights`,`float32`,u.neighbors.length):u.weights,F=n(e,`${o}-totals-a`,`float32`,2),I=n(e,`${o}-totals-b`,`float32`,2),L=n(e,`${o}-totals-c`,`float32`,4),R=n(e,`${o}-totals`,`float32`,9),z=`const ROWS: u32 = ${S}u;
const CAPACITY: u32 = ${u.neighbors.length}u;
const INVALID_POSITION: u32 = ${b}u;
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
${w}`,B=[{name:`offsets`,view:u.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`weights`,view:u.weights,type:`f32`,access:`read`}],V=g?[{name:`secondValues`,view:g,type:`f32`,access:`read`}]:[];return{nodes:[...p.nodes,...m?.nodes??[],a(e,{id:`${o}-flags`,operation:s,variant:`flags`,bindings:[{name:`values`,view:h,type:`f32`,access:`read`},...V,...f?[{name:`mask`,view:f,type:`u32`,access:`read`}]:[],{name:`flags`,view:D,type:`u32`,access:`read_write`}],invocationCount:S,declarations:z,body:`let x = values[valuesOffset + index];
  let y = ${g?`secondValues[secondValuesOffset + index]`:`0.0`};
  let included = ${f?`mask[maskOffset + index] != 0u &&`:``} isFiniteFloat(x) && isFiniteFloat(y);
  flags[flagsOffset + index] = select(0u, 1u, included);`}),i(e,{id:`${o}-island-clear`,operation:s,view:N,type:`u32`,value:`0u`,componentCount:1}),a(e,{id:`${o}-analysis`,operation:s,variant:_?`analysis-keep`:`analysis-exclude`,bindings:[...B,{name:`flags`,view:D,type:`u32`,access:`read`},{name:`analysisFlags`,view:O,type:`u32`,access:`read_write`},{name:`islandCount`,view:N,type:`atomic<u32>`,access:`read_write`},...x?[{name:`weightsOverflow`,view:x,type:`u32`,access:`read`}]:[]],invocationCount:S,declarations:z,body:`let included = flags[flagsOffset + index] != 0u;
  var hasNeighbor = false;
  if (included) {
    ${l(`flags[flagsOffset + neighbor] != 0u`,`hasNeighbor = true; break;`)}
  }
  if (included && !hasNeighbor) {
    atomicAdd(&islandCount[islandCountOffset], 1u);
  }
  let truncated = ${x?`weightsOverflow[weightsOverflowOffset] != 0u`:`false`};
  let analyzed = included && !truncated && ${_?`true`:`hasNeighbor`};
  analysisFlags[analysisFlagsOffset + index] = select(0u, 1u, analyzed);`}),a(e,{id:`${o}-moment-inputs`,operation:s,variant:`moment-inputs`,bindings:[{name:`values`,view:h,type:`f32`,access:`read`},...V,{name:`analysisFlags`,view:O,type:`u32`,access:`read`},{name:`matrix`,view:T,type:`f32`,access:`read_write`}],invocationCount:S,declarations:z,body:`let analyzed = analysisFlags[analysisFlagsOffset + index] != 0u;
  matrix[matrixOffset + getMatrixIndex(0u, index)] =
    select(0.0, values[valuesOffset + index], analyzed);
  matrix[matrixOffset + getMatrixIndex(1u, index)] =
    select(0.0, ${g?`secondValues[secondValuesOffset + index]`:`0.0`}, analyzed);`}),...E(e,{id:`${o}-sum-a`,operation:s,matrix:T,rows:S,blockCount:C,first:0,count:2,totals:F}),...new r({id:`${o}-scan`,input:O,output:k,mode:`exclusive`}).getCommandNodes(e),a(e,{id:`${o}-positions`,operation:s,variant:`positions`,bindings:[{name:`analysisFlags`,view:O,type:`u32`,access:`read`},{name:`scanned`,view:k,type:`u32`,access:`read`},{name:`rowPositions`,view:A,type:`u32`,access:`read_write`}],invocationCount:S+1,declarations:z,body:`if (index < ROWS) {
    rowPositions[rowPositionsOffset + index] =
      select(INVALID_POSITION, scanned[scannedOffset + index], analysisFlags[analysisFlagsOffset + index] != 0u);
  } else {
    rowPositions[rowPositionsOffset + ROWS] =
      scanned[scannedOffset + ROWS - 1u] + analysisFlags[analysisFlagsOffset + ROWS - 1u];
  }`}),a(e,{id:`${o}-deviations`,operation:s,variant:`deviations`,bindings:[{name:`rowPositions`,view:A,type:`u32`,access:`read`},{name:`totalsA`,view:F,type:`f32`,access:`read`},{name:`matrix`,view:T,type:`f32`,access:`read_write`}],invocationCount:S,declarations:z,body:`let included = rowPositions[rowPositionsOffset + index] != INVALID_POSITION;
  let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  matrix[matrixOffset + getMatrixIndex(2u, index)] =
    select(0.0, matrix[matrixOffset + getMatrixIndex(0u, index)] - totalsA[totalsAOffset] / count, included);
  matrix[matrixOffset + getMatrixIndex(3u, index)] =
    select(0.0, matrix[matrixOffset + getMatrixIndex(1u, index)] - totalsA[totalsAOffset + 1u] / count, included);`}),...E(e,{id:`${o}-sum-b`,operation:s,matrix:T,rows:S,blockCount:C,first:2,count:2,totals:I}),a(e,{id:`${o}-compact`,operation:s,variant:`compact`,bindings:[{name:`rowPositions`,view:A,type:`u32`,access:`read`},{name:`totalsB`,view:I,type:`f32`,access:`read`},{name:`compactX`,view:j,type:`f32`,access:`read_write`},{name:`compactY`,view:M,type:`f32`,access:`read_write`},{name:`matrix`,view:T,type:`f32`,access:`read_write`}],invocationCount:S,declarations:z,body:`let position = rowPositions[rowPositionsOffset + index];
  let included = position != INVALID_POSITION;
  let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  // Second-pass mean correction: the deviations from the first mean sum to count * (mean - mean0).
  let centeredX = matrix[matrixOffset + getMatrixIndex(2u, index)] - totalsB[totalsBOffset] / count;
  let centeredY = matrix[matrixOffset + getMatrixIndex(3u, index)] - totalsB[totalsBOffset + 1u] / count;
  if (included) {
    compactX[compactXOffset + position] = ${t.centerX?`centeredX`:`matrix[matrixOffset + getMatrixIndex(0u, index)]`};
    compactY[compactYOffset + position] = ${t.centerY===!1?`matrix[matrixOffset + getMatrixIndex(1u, index)]`:`centeredY`};
  }
  matrix[matrixOffset + getMatrixIndex(4u, index)] = select(0.0, centeredX * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(5u, index)] = select(0.0, centeredY * centeredY, included);`}),a(e,{id:`${o}-row-weights`,operation:s,variant:v?`row-weights-standardize`:`row-weights`,bindings:[...B,{name:`rowPositions`,view:A,type:`u32`,access:`read`},{name:`matrix`,view:T,type:`f32`,access:`read_write`},...v?[{name:`scaledWeights`,view:P,type:`f32`,access:`read_write`}]:[]],invocationCount:S,declarations:z,body:`var rowWeight = 0.0;
  if (rowPositions[rowPositionsOffset + index] != INVALID_POSITION) {
    ${l(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,`rowWeight += weight;`)}
  }
  // Row-standardized weights are re-standardized over the active slots: a row with any active slot sums to 1.
  matrix[matrixOffset + getMatrixIndex(6u, index)] = ${v?`select(0.0, 1.0, rowWeight > 0.0)`:`rowWeight`};
  matrix[matrixOffset + getMatrixIndex(7u, index)] = ${v?`select(0.0, 1.0, rowWeight > 0.0)`:`rowWeight * rowWeight`};
  ${v?`let scale = select(0.0, 1.0 / rowWeight, rowWeight > 0.0);
  let begin = min(offsets[offsetsOffset + index], CAPACITY);
  let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    scaledWeights[scaledWeightsOffset + slot] = weights[weightsOffset + slot] * scale;
  }`:``}`}),...E(e,{id:`${o}-sum-c`,operation:s,matrix:T,rows:S,blockCount:C,first:4,count:4,totals:L}),a(e,{id:`${o}-totals`,operation:s,variant:`totals`,bindings:[{name:`rowPositions`,view:A,type:`u32`,access:`read`},{name:`totalsA`,view:F,type:`f32`,access:`read`},{name:`totalsB`,view:I,type:`f32`,access:`read`},{name:`totalsC`,view:L,type:`f32`,access:`read`},{name:`totals`,view:R,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const ROWS: u32 = ${S}u;`,body:`let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  let meanX = totalsA[totalsAOffset] / count + totalsB[totalsBOffset] / count;
  let meanY = totalsA[totalsAOffset + 1u] / count + totalsB[totalsBOffset + 1u] / count;
  totals[totalsOffset] = count;
  totals[totalsOffset + 1u] = meanX * count;
  totals[totalsOffset + 2u] = meanY * count;
  totals[totalsOffset + 3u] = totalsC[totalsCOffset];
  totals[totalsOffset + 4u] = totalsC[totalsCOffset + 1u];
  totals[totalsOffset + 5u] = totalsC[totalsCOffset + 2u];
  totals[totalsOffset + 6u] = meanX;
  totals[totalsOffset + 7u] = meanY;
  totals[totalsOffset + 8u] = totalsC[totalsCOffset + 3u];`})],rowPositions:A,compactX:j,compactY:M,totals:R,islandCount:N,weights:P}}function E(e,t){let{id:r,operation:i,rows:s,blockCount:c,first:l,count:u}=t,d=u*c,f=n(e,`${r}-block-offsets`,`uint32`,d+1),p=n(e,`${r}-column-offsets`,`uint32`,u+1),m=n(e,`${r}-partials`,`float32`,d);return[a(e,{id:`${r}-offsets`,operation:i,variant:`sum-offsets`,bindings:[{name:`blockOffsets`,view:f,type:`u32`,access:`read_write`},{name:`columnOffsets`,view:p,type:`u32`,access:`read_write`}],invocationCount:d+1,declarations:`const ROWS: u32 = ${s}u;
const BLOCK_ROWS: u32 = ${y}u;
const BLOCK_COUNT: u32 = ${c}u;
const FIRST_COLUMN: u32 = ${l}u;
const COLUMN_COUNT: u32 = ${u}u;`,body:`let column = index / BLOCK_COUNT;
  let block = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(block * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`}),o(e,{id:`${r}-blocks`,operation:i,segmentCount:d,input:t.matrix,segmentOffsets:f,output:m,offsetsWithinInput:(l+u)*s<=t.matrix.length}),o(e,{id:`${r}-columns`,operation:i,segmentCount:u,input:m,segmentOffsets:p,output:t.totals,offsetsWithinInput:!0})]}export{E as a,S as c,h as d,m as f,x as i,v as l,p as m,w as n,T as o,_ as p,b as r,C as s,y as t,g as u};