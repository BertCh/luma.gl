import"./permutation-inference-kernels-D-xhEewi.js";import{_ as e,h as t,m as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{n as a,s as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as s}from"./wgsl-common-BNRVzgd8.js";import{n as c}from"./sorted-segment-sums-DwGkmbx2.js";import{c as l,o as u,r as d,s as f}from"./spatial-weights-D4bd9AQ6.js";import{n as p,t as m}from"./float32-values-C3OXugIo.js";import{t as ee}from"./slot-grouping-DuACcF2F.js";import{t as h}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";var g={summary:0,moran:8,geary:16,getisOrdG:24,bivariateMoran:32,joinCount:40,length:52},_={count:0,s0:1,s1:2,s2:3,mean:4,variance:5,blackCount:6,islandCount:7},v={statistic:0,expected:1,varianceNormality:2,zNormality:3,pNormality:4,varianceRandomization:5,zRandomization:6,pRandomization:7},y={blackBlack:0,blackWhite:1,whiteWhite:2,joins:3,expectedBlackBlack:4,varianceBlackBlack:5,zBlackBlack:6,pBlackBlack:7,expectedBlackWhite:8,varianceBlackWhite:9,zBlackWhite:10,pBlackWhite:11},b=`GPUGlobalSpatialStatistics`,x=4096,S=256,C=[`moran`,`geary`,`getisOrdG`,`bivariateMoran`,`joinCount`],w={included:0,x:1,y:2,centeredSquare:3,centeredCube:4,centeredFourth:5,centeredSquareY:6,black:7,rowWeight:8,s1:9,moranCross:10,gearySum:11,gCross:12,bivariateCross:13,binaryOutDegree:14,binaryS1:15,degree:16,binaryDegree:17,columnLag:18,island:19,degreeSquare:20,binaryDegreeSquare:21,degreeDeviation:22,binaryDegreeDeviation:23,columnLagDeviation:24,deviationX:25,deviationY:26},T=27,E={first:0,count:3},D={first:3,count:19},O={first:22,count:3},k={first:25,count:2},A=2**24,j=class{id;props;islands;alternative;constructor(e){let n=e.id??`global-spatial-statistics`;this.id=n,this.props=e,this.islands=f(n,e.islands),this.alternative=M(n,e.alternative);let r=l(n,e.weights);if(r>=A)throw Error(`${n} weights must hold fewer than 2^24 rows`);if(e.weights.status?.overflow&&i(e.weights.status.overflow,[`uint32`],`${n} weights.status.overflow`),p(e.values,`${n} values`)!==r)throw Error(`${n} values needs one row per weights row`);if(e.secondValues&&p(e.secondValues,`${n} secondValues`)!==r)throw Error(`${n} secondValues needs one row per weights row`);if(e.mask&&(i(e.mask,[`uint32`],`${n} mask`),e.mask.length!==r))throw Error(`${n} mask needs one row per weights row`);if(e.statistics.length===0)throw Error(`${n} statistics must name at least one statistic`);for(let t of e.statistics)if(!C.includes(t))throw Error(`${n} unknown statistic ${t}`);if(e.statistics.includes(`bivariateMoran`)&&!e.secondValues)throw Error(`${n} bivariateMoran requires secondValues`);let{output:a}=e;if(i(a.summary,[`float32`],`${n} output.summary`),a.summary.length<g.length)throw Error(`${n} output.summary needs ${g.length} values`);if(a.joinCounts&&(i(a.joinCounts,[`uint32`],`${n} output.joinCounts`),a.joinCounts.length<3))throw Error(`${n} output.joinCounts needs 3 values`);t(n,[a.summary,a.joinCounts],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.status?.overflow,e.values,e.secondValues,e.mask])}getInferenceMetadata(){return{alternative:this.alternative,permutationCount:0,multipleTesting:`none`,testedRowsDefinition:`included rows with finite values${this.islands===`exclude`?`, islands excluded`:` (islands kept, as esda)`}; one test per statistic`}}getCommandNodes(t){let{id:i,props:s}=this,{weights:c,mask:l,output:f}=s;e(i,t,[c.offsets,c.neighbors,c.weights,c.status?.overflow,s.values,s.secondValues,l,f.summary,f.joinCounts]);let p=m(t,`${i}-values`,s.values),g=s.secondValues?m(t,`${i}-second-values`,s.secondValues):void 0,_=p.view,v=g?.view,y=c.offsets.length-1,C=c.neighbors.length,A=c.status?.overflow,j=u(c),M=this.islands===`keep`,L=Math.ceil(y/x),R=`const ROWS: u32 = ${y}u;
const CAPACITY: u32 = ${C}u;
${Object.entries(w).map(([e,t])=>`const COLUMN_${te(e)}: u32 = ${t}u;`).join(`
`)}
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
// total / count with one remainder correction. WGSL division may be 2.5 ulp off (Metal computes
// it through a reciprocal), so 24 / 12 can miss 2; the correction makes the quotient correctly
// rounded in practice, so constant values center to exactly 0 and their statistics are NaN, not
// the ratio of rounding residues.
fn getMeanQuotient(total: f32, count: f32) -> f32 {
  let quotient = total / count;
  return quotient + fma(-quotient, count, total) / count;
}
${h}`,z=`isFiniteFloat(centered[centeredOffset + neighbor * 2u])`;n(t.device,i,`matrix`,T*y*4,`reduce the row count`);let B=r(t,`${i}-matrix`,`float32`,T*y),V=r(t,`${i}-centered`,`float32`,y*2),H=r(t,`${i}-totals-a`,`float32`,E.count),U=r(t,`${i}-totals-b`,`float32`,D.count),W=r(t,`${i}-totals-c`,`float32`,O.count),G=r(t,`${i}-totals-d`,`float32`,k.count),K=r(t,`${i}-base-states`,`uint32`,y),q=r(t,`${i}-counters`,`uint32`,5),J=r(t,`${i}-row-scale`,`float32`,y),Y=r(t,`${i}-slot-rows`,`uint32`,C),X=r(t,`${i}-transpose-keys`,`uint32`,C),Z=r(t,`${i}-slot-ids`,`uint32`,C),Q=r(t,`${i}-column-offsets`,`uint32`,y+1),re=[...v?[{name:`secondValues`,view:v,type:`f32`,access:`read`}]:[],...l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[]],$=ee(t,{id:`${i}-transpose`,operation:b,slotKeys:X,slotIndices:Z,columns:y,columnOffsets:Q}),{sortedSlots:ie}=$;return[...p.nodes,...g?.nodes??[],a(t,{id:`${i}-counters-clear`,operation:b,view:q,type:`u32`,value:`0u`}),o(t,{id:`${i}-include`,operation:b,variant:`include`,bindings:[{name:`values`,view:_,type:`f32`,access:`read`},...re,{name:`baseStates`,view:K,type:`u32`,access:`read_write`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,declarations:R,body:`let x = values[valuesOffset + index];
  let y = ${v?`secondValues[secondValuesOffset + index]`:`0.0`};
  let included = ${l?`mask[maskOffset + index] != 0u &&`:``} isFiniteFloat(x) && isFiniteFloat(y);
  baseStates[baseStatesOffset + index] = select(0u, 1u, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = select(0.0, x, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = select(0.0, y, included);`}),o(t,{id:`${i}-islands`,operation:b,variant:`islands`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`baseStates`,view:K,type:`u32`,access:`read`},{name:`counters`,view:q,type:`atomic<u32>`,access:`read_write`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,workgroupSize:S,guardIndex:!1,declarations:`${R}
var<workgroup> groupIncluded: atomic<u32>;
var<workgroup> groupNegative: atomic<u32>;`,body:`if (index < ROWS) {
  let base = baseStates[baseStatesOffset + index] != 0u;
  var hasNeighbor = false;
  if (base) {
    ${d(`baseStates[baseStatesOffset + neighbor] != 0u`,`hasNeighbor = true;`)}
  }
  let island = base && !hasNeighbor;
  let included = base && ${M?`true`:`!island`};
  if (included) {
    atomicAdd(&groupIncluded, 1u);
    if (matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] < 0.0) {
      atomicAdd(&groupNegative, 1u);
    }
  } else {
    // Missing rows and excluded islands leave the sums.
    matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = 0.0;
    matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = 0.0;
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] = select(0.0, 1.0, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ISLAND, index)] = select(0.0, 1.0, island);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset], atomicLoad(&groupIncluded));
    atomicAdd(&counters[countersOffset + 1u], atomicLoad(&groupNegative));
  }`}),...N(t,{id:`${i}-sum-a`,matrix:B,rows:y,blockCount:L,...E,totals:H}),o(t,{id:`${i}-deviate`,operation:b,variant:`deviate`,bindings:[{name:`totalsA`,view:H,type:`f32`,access:`read`},{name:`counters`,view:q,type:`u32`,access:`read`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,declarations:R,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)];
  let y = matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)];
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] =
    select(0.0, x - getMeanQuotient(totalsA[totalsAOffset + 1u], count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] =
    select(0.0, y - getMeanQuotient(totalsA[totalsAOffset + 2u], count), included);`}),...N(t,{id:`${i}-sum-d`,matrix:B,rows:y,blockCount:L,...k,totals:G}),o(t,{id:`${i}-center`,operation:b,variant:`center`,bindings:[{name:`totalsD`,view:G,type:`f32`,access:`read`},{name:`counters`,view:q,type:`u32`,access:`read`},{name:`centered`,view:V,type:`f32`,access:`read_write`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,declarations:R,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)];
  let centeredX = matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] - getMeanQuotient(totalsD[totalsDOffset], count);
  let centeredY = ${v?`matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] - getMeanQuotient(totalsD[totalsDOffset + 1u], count)`:`0.0`};
  let square = centeredX * centeredX;
  centered[centeredOffset + index * 2u] = select(getQuietNaN(index), centeredX, included);
  centered[centeredOffset + index * 2u + 1u] = select(getQuietNaN(index), centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE, index)] = select(0.0, square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_CUBE, index)] = select(0.0, square * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_FOURTH, index)] = select(0.0, square * square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE_Y, index)] = select(0.0, centeredY * centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BLACK, index)] =
    select(0.0, 1.0, included && x != 0.0);`}),...j?[o(t,{id:`${i}-row-scale`,operation:b,variant:`row-scale`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`centered`,view:V,type:`f32`,access:`read`},{name:`rowScale`,view:J,type:`f32`,access:`read_write`}],invocationCount:y,declarations:R,body:`var sum = 0.0;
  if (isFiniteFloat(centered[centeredOffset + index * 2u])) {
    ${d(z,`sum += weight;`)}
  }
  rowScale[rowScaleOffset + index] = select(0.0, 1.0 / sum, sum > 0.0);`})]:[],o(t,{id:`${i}-slot-rows`,operation:b,variant:`slot-rows`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`centered`,view:V,type:`f32`,access:`read`},{name:`slotRows`,view:Y,type:`u32`,access:`read_write`}],invocationCount:y,declarations:R,body:`let source = select(ROWS, index, isFiniteFloat(centered[centeredOffset + index * 2u]));
  let begin = min(offsets[offsetsOffset + index], CAPACITY);
  let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    slotRows[slotRowsOffset + slot] = source;
  }`}),o(t,{id:`${i}-transpose-keys`,operation:b,variant:`transpose-keys`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`slotRows`,view:Y,type:`u32`,access:`read`},{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`centered`,view:V,type:`f32`,access:`read`},{name:`transposeKeys`,view:X,type:`u32`,access:`read_write`},{name:`slotIds`,view:Z,type:`u32`,access:`read_write`}],invocationCount:C,declarations:R,body:`let used = index < min(offsets[offsetsOffset + ROWS], CAPACITY);
  let source = select(ROWS, slotRows[slotRowsOffset + index], used);
  let neighbor = neighbors[neighborsOffset + index];
  var key = ROWS;
  if (source < ROWS && neighbor < ROWS && neighbor != source &&
      weights[weightsOffset + index] > 0.0 &&
      isFiniteFloat(centered[centeredOffset + neighbor * 2u])) {
    key = neighbor;
  }
  transposeKeys[transposeKeysOffset + index] = key;
  slotIds[slotIdsOffset + index] = index;`}),...$.nodes,o(t,{id:`${i}-pairs`,operation:b,variant:`pairs`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:c.neighbors,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`values`,view:_,type:`f32`,access:`read`},...j?[{name:`rowScale`,view:J,type:`f32`,access:`read`}]:[],{name:`centered`,view:V,type:`f32`,access:`read`},{name:`matrix`,view:B,type:`f32`,access:`read_write`},{name:`counters`,view:q,type:`atomic<u32>`,access:`read_write`}],invocationCount:y,workgroupSize:S,guardIndex:!1,declarations:`${R}
var<workgroup> groupBlackBlack: atomic<u32>;
var<workgroup> groupBlackWhite: atomic<u32>;
var<workgroup> groupWhiteWhite: atomic<u32>;
fn getRowEnd(row: u32) -> u32 {
  return min(offsets[offsetsOffset + row + 1u], CAPACITY);
}

${P(j)}

// Effective weight w'_row,column by binary search of the ascending neighbor IDs of row, or -1 when
// the slot is absent or not active (zero weight).
fn findWeight(row: u32, column: u32) -> f32 {
  var low = min(offsets[offsetsOffset + row], CAPACITY);
  let end = getRowEnd(row);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    let weight = weights[weightsOffset + low];
    return select(-1.0, weight * getRowScale(row), weight > 0.0);
  }
  return -1.0;
}`,body:`var blackBlack = 0u;
  var blackWhite = 0u;
  var whiteWhite = 0u;
  if (index < ROWS) {
  let centeredX = centered[centeredOffset + index * 2u];
  var rowWeight = 0.0;
  var s1 = 0.0;
  var lag = 0.0;
  var gearySum = 0.0;
  var rawLag = 0.0;
  var lagY = 0.0;
  var outDegree = 0u;
  var binaryS1 = 0u;
  let x = values[valuesOffset + index];
  if (isFiniteFloat(centeredX)) {
    let black = x != 0.0;
    let scale = getRowScale(index);
    ${d(z,`let neighborX = centered[centeredOffset + neighbor * 2u];
      let scaled = weight * scale;
      let reverse = findWeight(neighbor, index);
      rowWeight += scaled;
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      lag += scaled * neighborX;
      let difference = centeredX - neighborX;
      gearySum += scaled * difference * difference;
      rawLag += scaled * values[valuesOffset + neighbor];
      lagY += scaled * centered[centeredOffset + neighbor * 2u + 1u];
      outDegree++;
      binaryS1 += select(1u, 2u, reverse > 0.0);
      let neighborBlack = values[valuesOffset + neighbor] != 0.0;
      if (black && neighborBlack) {
        blackBlack++;
      } else if (black != neighborBlack) {
        blackWhite++;
      } else {
        whiteWhite++;
      }`)}
  }
  let included = isFiniteFloat(centeredX);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] = rowWeight;
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_MORAN_CROSS, index)] = select(0.0, centeredX * lag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_GEARY_SUM, index)] = gearySum;
  matrix[matrixOffset + getMatrixIndex(COLUMN_G_CROSS, index)] = select(0.0, x * rawLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BIVARIATE_CROSS, index)] = select(0.0, centeredX * lagY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] = f32(outDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);
  }
  atomicAdd(&groupBlackBlack, blackBlack);
  atomicAdd(&groupBlackWhite, blackWhite);
  atomicAdd(&groupWhiteWhite, whiteWhite);
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset + 2u], atomicLoad(&groupBlackBlack));
    atomicAdd(&counters[countersOffset + 3u], atomicLoad(&groupBlackWhite));
    atomicAdd(&counters[countersOffset + 4u], atomicLoad(&groupWhiteWhite));
  }`}),o(t,{id:`${i}-columns`,operation:b,variant:`columns`,bindings:[{name:`columnOffsets`,view:Q,type:`u32`,access:`read`},{name:`sortedSlots`,view:ie,type:`u32`,access:`read`},{name:`slotRows`,view:Y,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},...j?[{name:`rowScale`,view:J,type:`f32`,access:`read`}]:[],{name:`centered`,view:V,type:`f32`,access:`read`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`${R}
${P(j)}`,body:`let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let begin = columnOffsets[columnOffsetsOffset + index];
  let end = columnOffsets[columnOffsetsOffset + index + 1u];
  var columnWeight = 0.0;
  var inDegree = 0u;
  var columnLag = 0.0;
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    // Only active slots are keyed into columns, so every weight here is positive.
    let source = slotRows[slotRowsOffset + slot];
    let weight = weights[weightsOffset + slot] * getRowScale(source);
    columnWeight += weight;
    inDegree++;
    columnLag += weight * centered[centeredOffset + source * 2u];
  }
  let degree = matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] + columnWeight;
  let binaryDegree = matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] + f32(inDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE, index)] = select(0.0, degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE, index)] = select(0.0, binaryDegree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG, index)] = select(0.0, columnLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_SQUARE, index)] = select(0.0, degree * degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_SQUARE, index)] =
    select(0.0, binaryDegree * binaryDegree, included);`}),...N(t,{id:`${i}-sum-b`,matrix:B,rows:y,blockCount:L,...D,totals:U}),o(t,{id:`${i}-deviations`,operation:b,variant:`deviations`,bindings:[{name:`totalsB`,view:U,type:`f32`,access:`read`},{name:`counters`,view:q,type:`u32`,access:`read`},{name:`matrix`,view:B,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`${R}
const BATCH_B_FIRST: u32 = ${D.first}u;
fn readTotalB(column: u32) -> f32 {
  return totalsB[totalsBOffset + column - BATCH_B_FIRST];
}
fn getSquaredDeviation(column: u32, row: u32, count: f32) -> f32 {
  let deviation = matrix[matrixOffset + getMatrixIndex(column, row)] - readTotalB(column) / count;
  return deviation * deviation;
}`,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_BINARY_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_COLUMN_LAG, index, count), included);`}),...N(t,{id:`${i}-sum-c`,matrix:B,rows:y,blockCount:L,...O,totals:W}),o(t,{id:`${i}-finalize`,operation:b,variant:`finalize`,bindings:[{name:`totalsA`,view:H,type:`f32`,access:`read`},{name:`totalsB`,view:U,type:`f32`,access:`read`},{name:`totalsC`,view:W,type:`f32`,access:`read`},{name:`totalsD`,view:G,type:`f32`,access:`read`},{name:`counters`,view:q,type:`u32`,access:`read`},{name:`results`,view:f.summary,type:`f32`,access:`read_write`},...f.joinCounts?[{name:`joinCounts`,view:f.joinCounts,type:`u32`,access:`read_write`}]:[],...A?[{name:`overflow`,view:A,type:`u32`,access:`read`}]:[]],invocationCount:1,declarations:`${R}
${ne}
const BATCH_B_FIRST: u32 = ${D.first}u;
const BATCH_C_FIRST: u32 = ${O.first}u;
const BATCH_D_FIRST: u32 = ${k.first}u;
fn readTotal(column: u32) -> f32 {
  if (column < BATCH_B_FIRST) {
    return totalsA[totalsAOffset + column];
  }
  if (column < BATCH_C_FIRST) {
    return totalsB[totalsBOffset + column - BATCH_B_FIRST];
  }
  if (column < BATCH_D_FIRST) {
    return totalsC[totalsCOffset + column - BATCH_C_FIRST];
  }
  return totalsD[totalsDOffset + column - BATCH_D_FIRST];
}
${F(this.alternative)}
fn writeBlock(offset: u32, block: array<f32, 8>, enabled: bool) {
  for (var field = 0u; field < 8u; field++) {
    results[resultsOffset + offset + field] = select(getQuietNaN(field), block[field], enabled);
  }
}`,body:I(s,!!A)})]}};function M(e,t){let n=t??`two-sided`;switch(n){case`two-sided`:case`greater`:return n;case`lesser`:return n;default:throw Error(`${e} alternative must be 'two-sided', 'greater' or 'lesser'`)}}function te(e){return e.replace(/([A-Z])/g,`_$1`).toUpperCase()}function N(e,t){let{id:n,rows:i,blockCount:a,first:s,count:l}=t,u=l*a,d=r(e,`${n}-block-offsets`,`uint32`,u+1),f=r(e,`${n}-column-offsets`,`uint32`,l+1),p=r(e,`${n}-partials`,`float32`,u);return[o(e,{id:`${n}-offsets`,operation:b,variant:`sum-offsets`,bindings:[{name:`blockOffsets`,view:d,type:`u32`,access:`read_write`},{name:`columnOffsets`,view:f,type:`u32`,access:`read_write`}],invocationCount:u+1,declarations:`const ROWS: u32 = ${i}u;
const BLOCK_ROWS: u32 = ${x}u;
const BLOCK_COUNT: u32 = ${a}u;
const FIRST_COLUMN: u32 = ${s}u;
const COLUMN_COUNT: u32 = ${l}u;`,body:`let column = index / BLOCK_COUNT;
  let block = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(block * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`}),c(e,{id:`${n}-blocks`,operation:b,segmentCount:u,input:t.matrix,segmentOffsets:d,output:p,offsetsWithinInput:(s+l)*i<=t.matrix.length}),c(e,{id:`${n}-columns`,operation:b,segmentCount:l,input:p,segmentOffsets:f,output:t.totals,offsetsWithinInput:!0})]}var ne=`
fn getFinite(value: f32) -> f32 {
  return select(getQuietNaN(0u), value, isFiniteFloat(value));
}

// Division that never relies on the implementation-defined result of x / 0.
fn divide(numerator: f32, denominator: f32) -> f32 {
  return select(getQuietNaN(0u), getFinite(numerator / denominator), denominator != 0.0 && isFiniteFloat(denominator));
}

fn getZScore(statistic: f32, expected: f32, variance: f32) -> f32 {
  let z = (statistic - expected) / sqrt(variance);
  return select(getQuietNaN(0u), z, variance > 0.0 && isFiniteFloat(z));
}

// esda.Moran: (expected, VI_norm, VI_rand).
fn getMoranMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec3<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let expected = -1.0 / (n - 1.0);
  let varianceNormality = (n2 * s1 - n * s2 + 3.0 * s02) / ((n2 - 1.0) * s02) - expected * expected;
  let a = n * ((n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
  let b = kurtosis * ((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
  let varianceRandomization =
    (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - expected * expected;
  return vec3<f32>(expected, varianceNormality, varianceRandomization);
}

// esda.Geary: (VC_norm, VC_rand).
fn getGearyMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec2<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let varianceNormality = ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02) / (2.0 * (n + 1.0) * s02);
  let a = (n - 1.0) * s1 * (n2 - 3.0 * n + 3.0 - (n - 1.0) * kurtosis);
  let b = 0.25 * (n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * kurtosis);
  let c = s02 * (n2 - 3.0 - (n - 1.0) * (n - 1.0) * kurtosis);
  let varianceRandomization = (a - b + c) / (n * (n - 2.0) * (n - 3.0) * s02);
  return vec2<f32>(varianceNormality, varianceRandomization);
}

// Randomization variance of x'Wx (zero diagonal) with x = mean + z, sum z = 0:
// mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz), d_i = w_i. + w_.i, Vd = sum (d - mean d)^2.
fn getQuadraticFormVariance(
  n: f32, s0: f32, s1: f32, s2: f32, degreeDeviation: f32, mean: f32,
  sumSquares: f32, sumCubes: f32, sumFourths: f32
) -> f32 {
  // A constant input has an invariant quadratic form under every permutation.
  if (sumSquares == 0.0) { return 0.0; }
  let kurtosis = n * sumFourths / (sumSquares * sumSquares);
  let moranVariance = getMoranMoments(n, s0, s1, s2, kurtosis).z;
  let scale = s0 * sumSquares / n;
  let linearVariance = sumSquares * degreeDeviation / (n - 1.0);
  let covariance = -sumCubes * degreeDeviation / ((n - 1.0) * (n - 2.0));
  return mean * mean * linearVariance + 2.0 * mean * covariance + moranVariance * scale * scale;
}
`;function P(e){return e?`fn getRowScale(row: u32) -> f32 {
  return rowScale[rowScaleOffset + row];
}`:`fn getRowScale(row: u32) -> f32 {
  return 1.0;
}`}function F(e){let t=`0.70710678118654752`;return`${s}
fn getPValue(z: f32) -> f32 {
  return select(getQuietNaN(0u), ${{"two-sided":`getTwoSidedPValue(z)`,greater:`min(0.5 * erfcValue(z * ${t}), 1.0)`,lesser:`min(0.5 * erfcValue(-z * ${t}), 1.0)`}[e]}, isFiniteFloat(z));
}`}function I(e,t){let n=t=>e.statistics.includes(t)?t===`getisOrdG`?`!truncated && !hasNegative`:`!truncated`:`false`,r=g;return`let n = f32(counters[countersOffset]);
  let hasNegative = counters[countersOffset + 1u] > 0u;
  let truncated = ${t?`overflow[overflowOffset] != 0u`:`false`};
  // Mean with the second-pass correction sum(x - mean) / n.
  let mean = divide(readTotal(COLUMN_X), n) + divide(readTotal(COLUMN_DEVIATION_X), n);
  let sumSquares = readTotal(COLUMN_CENTERED_SQUARE);
  let sumCubes = readTotal(COLUMN_CENTERED_CUBE);
  let sumFourths = readTotal(COLUMN_CENTERED_FOURTH);
  let s0 = readTotal(COLUMN_ROW_WEIGHT);
  let s1 = readTotal(COLUMN_S1);
  let s2 = readTotal(COLUMN_DEGREE_SQUARE);
  let blackCount = readTotal(COLUMN_BLACK);
  let degreeDeviation = readTotal(COLUMN_DEGREE_DEVIATION);
  let kurtosis = divide(n * sumFourths, sumSquares * sumSquares);
  let valid = n >= 4.0 && s0 > 0.0 && sumSquares > 0.0;

  results[resultsOffset + ${r.summary}u] = n;
  results[resultsOffset + ${r.summary+1}u] = select(s0, getQuietNaN(1u), truncated);
  results[resultsOffset + ${r.summary+2}u] = select(s1, getQuietNaN(2u), truncated);
  results[resultsOffset + ${r.summary+3}u] = select(s2, getQuietNaN(3u), truncated);
  results[resultsOffset + ${r.summary+4}u] = select(getFinite(mean), getQuietNaN(4u), truncated);
  results[resultsOffset + ${r.summary+5}u] = select(divide(sumSquares, n), getQuietNaN(5u), truncated);
  results[resultsOffset + ${r.summary+6}u] = blackCount;
  results[resultsOffset + ${r.summary+7}u] = readTotal(COLUMN_ISLAND);

  // Moran's I.
  {
    let statistic = divide(n * readTotal(COLUMN_MORAN_CROSS), s0 * sumSquares);
    let moments = getMoranMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.z), valid);
    let zNormality = getZScore(statistic, moments.x, varianceNormality);
    let zRandomization = getZScore(statistic, moments.x, varianceRandomization);
    // E[I] = -1 / (n - 1) is undefined below two rows (n = 0 would give 1).
    let expected = select(getQuietNaN(1u), getFinite(moments.x), n >= 2.0);
    writeBlock(${r.moran}u, array<f32, 8>(statistic, expected, varianceNormality,
      zNormality, getPValue(zNormality), varianceRandomization, zRandomization,
      getPValue(zRandomization)), ${n(`moran`)});
  }
  // Geary's C.
  {
    let statistic = divide((n - 1.0) * readTotal(COLUMN_GEARY_SUM), 2.0 * s0 * sumSquares);
    let moments = getGearyMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.x), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let zNormality = getZScore(statistic, 1.0, varianceNormality);
    let zRandomization = getZScore(statistic, 1.0, varianceRandomization);
    writeBlock(${r.geary}u, array<f32, 8>(statistic, select(getQuietNaN(1u), 1.0, n >= 2.0),
      varianceNormality, zNormality,
      getPValue(zNormality), varianceRandomization, zRandomization, getPValue(zRandomization)),
      ${n(`geary`)});
  }
  // Getis-Ord General G.
  {
    let denominator = n * (n - 1.0) * mean * mean - sumSquares;
    let statistic = divide(readTotal(COLUMN_G_CROSS), denominator);
    let expected = divide(s0, n * (n - 1.0));
    let variance = select(getQuietNaN(1u), getFinite(getQuadraticFormVariance(n, s0, s1, s2,
      degreeDeviation, mean, sumSquares, sumCubes, sumFourths) / (denominator * denominator)),
      n >= 4.0 && s0 > 0.0 && denominator > 0.0);
    let z = getZScore(statistic, expected, variance);
    // esda calls these VG / z_norm; retain the randomization aliases for compatibility.
    writeBlock(${r.getisOrdG}u, array<f32, 8>(statistic, expected, variance,
      z, getPValue(z), variance, z, getPValue(z)), ${n(`getisOrdG`)});
  }
  // Bivariate Moran's I, randomization of y.
  {
    let sumSquaresY = readTotal(COLUMN_CENTERED_SQUARE_Y);
    let scale = divide(n, s0);
    let statistic = divide(scale * readTotal(COLUMN_BIVARIATE_CROSS), sqrt(sumSquares * sumSquaresY));
    let variance = select(getQuietNaN(1u), getFinite(scale * scale *
      readTotal(COLUMN_COLUMN_LAG_DEVIATION) / ((n - 1.0) * sumSquares)), valid);
    let z = getZScore(statistic, 0.0, variance);
    writeBlock(${r.bivariateMoran}u, array<f32, 8>(statistic, 0.0, getQuietNaN(2u),
      getQuietNaN(2u), getQuietNaN(2u), variance, z, getPValue(z)), ${n(`bivariateMoran`)});
  }
  // Join counts over the binarized weights.
  {
    let enabled = ${n(`joinCount`)};
    let blackBlack2 = counters[countersOffset + 2u];
    let blackWhite2 = counters[countersOffset + 3u];
    let whiteWhite2 = counters[countersOffset + 4u];
    ${e.output.joinCounts?`joinCounts[joinCountsOffset] = select(0u, blackBlack2, enabled);
    joinCounts[joinCountsOffset + 1u] = select(0u, blackWhite2, enabled);
    joinCounts[joinCountsOffset + 2u] = select(0u, whiteWhite2, enabled);`:``}
    let b0 = readTotal(COLUMN_BINARY_OUT_DEGREE);
    let b1 = readTotal(COLUMN_BINARY_S1);
    let b2 = readTotal(COLUMN_BINARY_DEGREE_SQUARE);
    let p = divide(blackCount, n);
    let q = 1.0 - p;
    let blackSquares = n * p * q;
    let blackCubes = blackSquares * (q - p);
    let blackFourths = blackSquares * (p * p * p + q * q * q);
    let binaryValid = n >= 4.0 && b0 > 0.0;
    // A single colour makes every join count constant: zero variance.
    let oneColour = blackCount == 0.0 || blackCount == n;
    let expectedBlackBlack = getFinite(0.5 * b0 * blackCount * (blackCount - 1.0) / (n * (n - 1.0)));
    let varianceBlackBlack = select(getQuietNaN(3u), getFinite(0.25 * getQuadraticFormVariance(n,
      b0, b1, b2, readTotal(COLUMN_BINARY_DEGREE_DEVIATION), p, blackSquares, blackCubes,
      blackFourths)), binaryValid);
    let varianceBlackBlackOrZero = select(varianceBlackBlack, 0.0, binaryValid && oneColour);
    let expectedBlackWhite = getFinite(b0 * blackSquares / (n - 1.0));
    let blackKurtosis = n * blackFourths / (blackSquares * blackSquares);
    let blackWhiteScale = b0 * blackSquares / (n - 1.0);
    let varianceBlackWhite = select(getQuietNaN(3u), getFinite(getGearyMoments(n, b0, b1, b2,
      blackKurtosis).y * blackWhiteScale * blackWhiteScale), binaryValid && !oneColour);
    let varianceBlackWhiteOrZero = select(varianceBlackWhite, 0.0, binaryValid && oneColour);
    let blackBlack = 0.5 * f32(blackBlack2);
    let blackWhite = 0.5 * f32(blackWhite2);
    let zBlackBlack = getZScore(blackBlack, expectedBlackBlack, varianceBlackBlackOrZero);
    let zBlackWhite = getZScore(blackWhite, expectedBlackWhite, varianceBlackWhiteOrZero);
    let block = array<f32, 12>(blackBlack, blackWhite, 0.5 * f32(whiteWhite2), 0.5 * b0,
      expectedBlackBlack, varianceBlackBlackOrZero, zBlackBlack, getPValue(zBlackBlack),
      expectedBlackWhite, varianceBlackWhiteOrZero, zBlackWhite, getPValue(zBlackWhite));
    for (var field = 0u; field < 12u; field++) {
      results[resultsOffset + ${r.joinCount}u + field] = select(getQuietNaN(field), block[field], enabled);
    }
  }`}export{v as a,_ as i,y as n,g as r,j as t};