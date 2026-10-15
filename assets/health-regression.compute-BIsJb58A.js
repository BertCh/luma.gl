import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{J as t}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{_ as n,b as r,f as i,g as a,h as o,m as s,p as c,t as l,v as u,x as d,y as f}from"./gpu-geographically-weighted-regression-BYSqsAQL.js";import{N as p,_ as m,d as h,h as g,l as _,t as v,v as y,w as b,y as x}from"./resources-CwvhEOBe.js";import{l as S,s as C}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as w,n as T,r as E}from"./wgsl-common-BNRVzgd8.js";import{c as D}from"./spatial-weights-D4bd9AQ6.js";import{t as O}from"./gpu-contiguity-weights-BI6hkB9v.js";import{t as k}from"./gpu-spatial-weights-transform-B3cihvVH.js";import{t as A}from"./gpu-spatial-weights-transpose-DBog2XI3.js";import{d as j,f as M}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";import{t as N}from"./gpu-local-moran-CvNMe0Gu.js";import{n as P,t as F}from"./philox-random-CHwyk0dd.js";import{n as I,t as L}from"./gpu-neighbor-search-Fct8W8hB.js";import{r as R,s as z,t as B}from"./recipe-utils-CqprOQld.js";import{H as V,c as H}from"./index-Bjk18V2I.js";import{t as U}from"./graph-buffers-B4_80RlX.js";import{t as ee}from"./graph-swap-41Ru7LVu.js";import{_ as te,a as W,c as G,d as K,f as ne,g as re,i as ie,l as q,o as ae,p as oe,r as se,s as ce,t as le,u as ue,v as de,y as fe}from"./b6-polygons-Bf2djBQx.js";import{n as pe,t as me}from"./b6-fill-layer-CwuOmgDO.js";var J=`GPUSpatialRegressionDiagnostics`,he=64,ge=4096,_e=64,ve=7,ye=1;function be(e){return _(e,4)}function xe(e){return _(e,4)}var Se=class{id;props;rowCount;tileRowCount;tileCount;constructor(e){let t=e.id??`spatial-regression-diagnostics`;this.id=t,this.props=e;let{predictorCount:n,output:r}=e;h(t,`predictorCount`,n,1,15);let i=D(t,e.weights);if(x(e.predictors,[`float32`],`${t} predictors`),x(e.response,[`float32`],`${t} response`),x(e.residuals,[`float32`],`${t} residuals`),e.response.length!==i)throw Error(`${t} response length must equal the weights row count`);if(e.residuals.length!==i)throw Error(`${t} residuals length must equal the weights row count`);if(e.predictors.length!==i*n)throw Error(`${t} predictors length must equal the weights row count * predictorCount`);if(e.mask&&(x(e.mask,[`uint32`],`${t} mask`),e.mask.length!==i))throw Error(`${t} mask length must equal the weights row count`);if(x(r.tests,[`float32`],`${t} output.tests`),r.tests.length<18)throw Error(`${t} output.tests must hold 18 float32 values`);if(x(r.summary,[`float32`],`${t} output.summary`),r.summary.length<13)throw Error(`${t} output.summary must hold 13 float32 values`);if(x(r.status.code,[`uint32`],`${t} output.status.code`),r.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);e.tileRowCount!==void 0&&h(t,`tileRowCount`,e.tileRowCount),this.rowCount=i,this.tileRowCount=e.tileRowCount??Math.max(_e,Math.ceil(i/ge)),this.tileCount=Math.ceil(i/this.tileRowCount),g(t,[r.tests,r.summary,r.status.code],[e.predictors,e.response,e.residuals,e.mask,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(e){let{id:t,props:n,rowCount:r,tileRowCount:i,tileCount:a}=this,{weights:o,output:s,predictorCount:c}=n;m(t,e,[n.predictors,n.response,n.residuals,n.mask,o.offsets,o.neighbors,o.weights,s.tests,s.summary,s.status.code]);let l=c+1,u=5*l*l+ve+l+ye;y(t,`rows x (predictorCount + 3)`,r*(c+3),be(e.device)),y(t,`tiles x moments (rows, tileRowCount, predictorCount)`,a*u,xe(e.device));let d=n.restandardizeRows===!0,f=(n,r)=>b(e,`${t}-${n}`,`float32`,r),p=f(`partial-means`,a*(c+1)),h=f(`means`,c+1),g=f(`partial-moments`,a*u),_=f(`moments`,u),v=f(`row-table`,r*(l+2)),x=f(`transposed-lag-table`,r*l),S=d?f(`row-scale`,r):void 0,T={offsets:b(e,`${t}-transpose-offsets`,`uint32`,r+1),neighbors:b(e,`${t}-transpose-neighbors`,`uint32`,o.neighbors.length),weights:f(`transpose-weights`,o.neighbors.length)},E=new A({id:`${t}-transpose`,weights:o,output:T}).getCommandNodes(e),D=`const ROWS: u32 = ${r}u;
const K: u32 = ${c}u;
const TABLE_WIDTH: u32 = ${l+2}u;
const P: u32 = ${l}u;
const TILE_ROWS: u32 = ${i}u;
const TILE_COUNT: u32 = ${a}u;`,O=`${D}
${w}
fn isFittedRow(row: u32) -> bool {
  ${n.mask?`if (rowMask[rowMaskOffset + row] == 0u) {
    return false;
  }`:``}
  if (!isFiniteValue(response[responseOffset + row]) ||
      !isFiniteValue(residuals[residualsOffset + row])) {
    return false;
  }
  for (var column = 0u; column < K; column++) {
    if (!isFiniteValue(design[designOffset + row * K + column])) {
      return false;
    }
  }
  return true;
}`,k=n.mask?[{name:`rowMask`,view:n.mask,type:`u32`,access:`read`}]:[],j=[{name:`design`,view:n.predictors,type:`f32`,access:`read`},{name:`response`,view:n.response,type:`f32`,access:`read`},{name:`residuals`,view:n.residuals,type:`f32`,access:`read`},...k],M=S?[{name:`rowScale`,view:S,type:`f32`,access:`read`}]:[],N=S?`fn getRowScale(row: u32) -> f32 {
  return rowScale[rowScaleOffset + row];
}`:`fn getRowScale(row: u32) -> f32 {
  return 1.0;
}`,P=e=>`var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + tile * ${e}u + index] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`,F=[{name:`offsets`,view:o.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:o.neighbors,type:`u32`,access:`read`},{name:`weights`,view:o.weights,type:`f32`,access:`read`}],I={name:`table`,view:v,type:`f32`,access:`read`};return[C(e,{id:`${t}-means-tiles`,operation:J,variant:`means-tiles`,workgroupSize:he,bindings:[...j,{name:`partial`,view:p,type:`f32`,access:`read_write`}],invocationCount:a,declarations:O,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${c}>;
  var count = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isFittedRow(row)) {
      continue;
    }
    count = count + 1.0;
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
  }
  for (var column = 0u; column < K; column++) {
    partial[partialOffset + index * ${c+1}u + column] = sums[column];
  }
  partial[partialOffset + index * ${c+1}u + K] = count;`}),C(e,{id:`${t}-means-merge`,operation:J,variant:`means-merge`,bindings:[{name:`partial`,view:p,type:`f32`,access:`read`},{name:`means`,view:h,type:`f32`,access:`read_write`}],invocationCount:c+1,declarations:D,body:`${P(c+1)}
  var rows = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    rows = rows + partial[partialOffset + tile * ${c+1}u + K];
  }
  means[meansOffset + index] = select(sum, select(0.0, sum / rows, rows > 0.0), index < K);`}),C(e,{id:`${t}-row-table`,operation:J,variant:`row-table`,bindings:[...j,{name:`means`,view:h,type:`f32`,access:`read`},{name:`tableOut`,view:v,type:`f32`,access:`read_write`}],invocationCount:r,declarations:O,body:`let base = tableOutOffset + index * TABLE_WIDTH;
  if (!isFittedRow(index)) {
    for (var column = 0u; column < TABLE_WIDTH; column++) {
      tableOut[base + column] = 0.0;
    }
    return;
  }
  tableOut[base] = 1.0;
  for (var column = 0u; column < K; column++) {
    tableOut[base + 1u + column] =
      design[designOffset + index * K + column] - means[meansOffset + column];
  }
  tableOut[base + P] = response[responseOffset + index];
  tableOut[base + P + 1u] = residuals[residualsOffset + index];`}),...S?[C(e,{id:`${t}-row-scale`,operation:J,variant:`row-scale`,bindings:[...F,I,{name:`scaleOut`,view:S,type:`f32`,access:`read_write`}],invocationCount:r,declarations:D,body:`var total = 0.0;
  if (table[tableOffset + index * TABLE_WIDTH] > 0.5) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && table[tableOffset + neighbor * TABLE_WIDTH] > 0.5) {
        total += weights[weightsOffset + slot];
      }
    }
  }
  scaleOut[scaleOutOffset + index] = select(0.0, 1.0 / total, total > 0.0);`})]:[],...E,C(e,{id:`${t}-transposed-lag`,operation:J,variant:`transposed-lag`,bindings:[{name:`offsets`,view:T.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:T.neighbors,type:`u32`,access:`read`},{name:`weights`,view:T.weights,type:`f32`,access:`read`},I,...M,{name:`lagOut`,view:x,type:`f32`,access:`read_write`}],invocationCount:r,declarations:`${D}
${N}`,body:`var sums: array<f32, ${l}>;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    // The transposed link j -> i carries w_ji, scaled by the row scale of its source row j.
    let weight = weights[weightsOffset + slot] * getRowScale(neighbor);
    for (var column = 0u; column < P; column++) {
      sums[column] += weight * table[tableOffset + neighbor * TABLE_WIDTH + column];
    }
  }
  for (var column = 0u; column < P; column++) {
    lagOut[lagOutOffset + index * P + column] = sums[column];
  }`}),C(e,{id:`${t}-moments-tiles`,operation:J,variant:`moments-tiles`,workgroupSize:he,bindings:[...F,I,...M,{name:`transposedLag`,view:x,type:`f32`,access:`read`},{name:`partial`,view:g,type:`f32`,access:`read_write`}],invocationCount:a,declarations:`${D}
const PP: u32 = ${l*l}u;
const MOMENTS: u32 = ${u}u;
${N}
${Ce()}`,body:`let base = partialOffset + index * MOMENTS;
  for (var slot = 0u; slot < MOMENTS; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    accumulateRow(row, base);
  }`}),C(e,{id:`${t}-moments-merge`,operation:J,variant:`moments-merge`,bindings:[{name:`partial`,view:g,type:`f32`,access:`read`},{name:`moments`,view:_,type:`f32`,access:`read_write`}],invocationCount:u,declarations:D,body:`${P(u)}
  moments[momentsOffset + index] = sum;`}),C(e,{id:`${t}-finish`,operation:J,variant:`finish`,bindings:[{name:`moments`,view:_,type:`f32`,access:`read`},{name:`testsOut`,view:s.tests,type:`f32`,access:`read_write`},{name:`summaryOut`,view:s.summary,type:`f32`,access:`read_write`},{name:`statusOut`,view:s.status.code,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${D}
const PP: u32 = ${l*l}u;
${we(l)}`,body:`finish();`})]}};function Ce(){return`
fn tableValue(row: u32, column: u32) -> f32 {
  return table[tableOffset + row * TABLE_WIDTH + column];
}

fn accumulateRow(row: u32, base: u32) {
  // An excluded row (leading table entry 0) is outside the analysis.
  if (tableValue(row, 0u) < 0.5) {
    return;
  }
  let rowWeightScale = getRowScale(row);
  var z: array<f32, P>;
  var u: array<f32, P>;
  var v: array<f32, P>;
  for (var column = 0u; column < P; column++) {
    z[column] = tableValue(row, column);
    // (W'Z)_row, accumulated over the transposed CSR so one-way links are included.
    v[column] = transposedLag[transposedLagOffset + row * P + column];
  }
  var weightSum = 0.0;
  var squareSum = 0.0;
  var crossSum = 0.0;
  var lagResidual = 0.0;
  var lagResponse = 0.0;
  var lagFitted = 0.0;
  let residual = tableValue(row, P + 1u);
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    // Links to excluded rows are dropped from the restricted W.
    if (neighbor >= ROWS || tableValue(neighbor, 0u) < 0.5) {
      continue;
    }
    let weight = weights[weightsOffset + slot] * rowWeightScale;
    // Reverse weight w_ji by binary search of this row in the neighbor's sorted row; it is 0 for
    // a one-way link, which then contributes nothing to tr(WW).
    var reverse = 0.0;
    var low = offsets[offsetsOffset + neighbor];
    var high = offsets[offsetsOffset + neighbor + 1u];
    while (low < high) {
      let middle = (low + high) / 2u;
      let candidate = neighbors[neighborsOffset + middle];
      if (candidate == row) {
        reverse = weights[weightsOffset + middle] * getRowScale(neighbor);
        break;
      }
      if (candidate < row) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    weightSum += weight;
    squareSum += weight * weight;
    crossSum += weight * reverse;
    let neighborResponse = tableValue(neighbor, P);
    let neighborResidual = tableValue(neighbor, P + 1u);
    lagResponse += weight * neighborResponse;
    lagResidual += weight * neighborResidual;
    lagFitted += weight * (neighborResponse - neighborResidual);
    for (var column = 0u; column < P; column++) {
      u[column] += weight * tableValue(neighbor, column);
    }
  }
  for (var first = 0u; first < P; first++) {
    for (var second = 0u; second < P; second++) {
      let slot = first * P + second;
      partial[base + slot] += z[first] * z[second];
      partial[base + PP + slot] += z[first] * u[second];
      partial[base + 2u * PP + slot] += u[first] * u[second];
      partial[base + 3u * PP + slot] += v[first] * v[second];
      partial[base + 4u * PP + slot] += v[first] * u[second];
    }
  }
  let scalars = base + 5u * PP;
  partial[scalars] += weightSum;
  partial[scalars + 1u] += squareSum;
  partial[scalars + 2u] += crossSum;
  partial[scalars + 3u] += residual * residual;
  partial[scalars + 4u] += residual * lagResidual;
  partial[scalars + 5u] += residual * lagResponse;
  partial[scalars + 6u] += lagFitted * lagFitted;
  for (var column = 0u; column < P; column++) {
    partial[scalars + 7u + column] += z[column] * lagFitted;
  }
  partial[scalars + 7u + P] += 1.0;
}
`}function we(e){return`
var<private> columnScale: array<f32, ${e}>;

${E}

${w}

// Complementary error function, fractional error below 1.2e-7 (Numerical Recipes erfcc).
${T}

fn chiSquareSurvival(value: f32, degreesOfFreedom: u32) -> f32 {
  if (degreesOfFreedom == 1u) {
    return erfcValue(sqrt(max(value, 0.0) * 0.5));
  }
  return exp(-max(value, 0.0) * 0.5);
}

// Scaled (unit-diagonal Gram) entry of the p x p matrix stored at \`base\`.
fn scaledMoment(base: u32, row: u32, column: u32) -> f32 {
  return moments[momentsOffset + base + row * P + column] * columnScale[row] * columnScale[column];
}

${s(e)}

// result = (Z'Z)^-1 Q (or its transpose version) for the matrix Q stored at \`base\`.
fn solveMatrix(
  factor: ptr<function, array<f32, ${e*e}>>,
  base: u32,
  transposed: bool,
  result: ptr<function, array<f32, ${e*e}>>
) {
  for (var column = 0u; column < P; column++) {
    var rhs: array<f32, ${e}>;
    for (var row = 0u; row < P; row++) {
      rhs[row] = select(
        scaledMoment(base, row, column),
        scaledMoment(base, column, row),
        transposed
      );
    }
    choleskySolve_${e}(factor, &rhs);
    for (var row = 0u; row < P; row++) {
      (*result)[row * P + column] = rhs[row];
    }
  }
}

fn writeTest(row: u32, statistic: f32, degreesOfFreedom: u32) {
  testsOut[testsOutOffset + row * 3u] = statistic;
  testsOut[testsOutOffset + row * 3u + 1u] = f32(degreesOfFreedom);
  testsOut[testsOutOffset + row * 3u + 2u] = select(
    getNaN(),
    chiSquareSurvival(statistic, degreesOfFreedom),
    isFiniteValue(statistic)
  );
}

fn fail(status: u32, fittedRows: f32) {
  let nan = getNaN();
  for (var slot = 0u; slot < 18u; slot++) {
    testsOut[testsOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 13u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = fittedRows;
  summaryOut[summaryOutOffset + 12u] =
    f32(ROWS) - fittedRows;
  for (var row = 0u; row < 5u; row++) {
    testsOut[testsOutOffset + row * 3u + 1u] = select(1.0, 2.0, row == 4u);
  }
  testsOut[testsOutOffset + 16u] = 0.0;
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let scalars = 5u * PP;
  // Rows analyzed: finite and unmasked. Everything below runs over those rows only.
  let n = moments[momentsOffset + scalars + 7u + P];
  let weightSum = moments[momentsOffset + scalars];
  let squareSum = moments[momentsOffset + scalars + 1u];
  let crossSum = moments[momentsOffset + scalars + 2u];
  let residualSquares = moments[momentsOffset + scalars + 3u];
  let residualLag = moments[momentsOffset + scalars + 4u];
  let responseLag = moments[momentsOffset + scalars + 5u];
  let fittedLagSquares = moments[momentsOffset + scalars + 6u];
  if (!(n > f32(P) + 1.0)) {
    fail(2u, select(0.0, n, isFiniteValue(n)));
    return;
  }
  if (!isFiniteValue(residualSquares) || !isFiniteValue(residualLag) ||
      !isFiniteValue(responseLag) || !isFiniteValue(fittedLagSquares) ||
      !isFiniteValue(squareSum) || !isFiniteValue(crossSum) || !(residualSquares > 0.0)) {
    fail(4u, n);
    return;
  }
  let traceT = squareSum + crossSum;
  if (!(weightSum > 0.0) || !(traceT > 0.0)) {
    fail(3u, n);
    return;
  }
  for (var column = 0u; column < P; column++) {
    let diagonal = moments[momentsOffset + column * P + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      fail(1u, n);
      return;
    }
    columnScale[column] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      factor[row * P + column] = scaledMoment(0u, row, column);
    }
  }
  if (!choleskyFactor_${e}(&factor)) {
    fail(1u, n);
    return;
  }
  var inverseC: array<f32, ${e*e}>;
  var inverseCTransposed: array<f32, ${e*e}>;
  var inverseD: array<f32, ${e*e}>;
  var inverseE: array<f32, ${e*e}>;
  var inverseF: array<f32, ${e*e}>;
  solveMatrix(&factor, PP, false, &inverseC);
  solveMatrix(&factor, PP, true, &inverseCTransposed);
  solveMatrix(&factor, 2u * PP, false, &inverseD);
  solveMatrix(&factor, 3u * PP, false, &inverseE);
  solveMatrix(&factor, 4u * PP, false, &inverseF);
  var traceC = 0.0;
  var traceD = 0.0;
  var traceE = 0.0;
  var traceF = 0.0;
  var traceCC = 0.0;
  var traceCCTransposed = 0.0;
  for (var row = 0u; row < P; row++) {
    traceC += inverseC[row * P + row];
    traceD += inverseD[row * P + row];
    traceE += inverseE[row * P + row];
    traceF += inverseF[row * P + row];
    for (var column = 0u; column < P; column++) {
      traceCC += inverseC[row * P + column] * inverseC[column * P + row];
      traceCCTransposed += inverseC[row * P + column] * inverseCTransposed[column * P + row];
    }
  }
  // (W Zb)' M (W Zb) = v'v - (Z'v)' (Z'Z)^-1 (Z'v) with v = W (y - e).
  var projected: array<f32, ${e}>;
  var scaledProduct: array<f32, ${e}>;
  for (var column = 0u; column < P; column++) {
    scaledProduct[column] = moments[momentsOffset + scalars + 7u + column] * columnScale[column];
    projected[column] = scaledProduct[column];
  }
  choleskySolve_${e}(&factor, &projected);
  var quadratic = 0.0;
  for (var column = 0u; column < P; column++) {
    quadratic += scaledProduct[column] * projected[column];
  }
  let fittedLagResidualSquares = max(fittedLagSquares - quadratic, 0.0);

  let sigmaSquared = residualSquares / n;
  let errorGradient = residualLag / sigmaSquared;
  let lagGradient = responseLag / sigmaSquared;
  let information = fittedLagResidualSquares / sigmaSquared + traceT;
  let lagStatistic = lagGradient * lagGradient / information;
  let errorStatistic = errorGradient * errorGradient / traceT;
  let robustDenominator = traceT - traceT * traceT / information;
  var robustError = getNaN();
  var robustLag = getNaN();
  var sarma = getNaN();
  if (robustDenominator > 0.0 && information > traceT) {
    let numerator = errorGradient - traceT * lagGradient / information;
    robustError = numerator * numerator / robustDenominator;
    let difference = lagGradient - errorGradient;
    robustLag = difference * difference / (information - traceT);
    sarma = lagStatistic + robustError;
  }
  writeTest(0u, lagStatistic, 1u);
  writeTest(1u, errorStatistic, 1u);
  writeTest(2u, robustLag, 1u);
  writeTest(3u, robustError, 1u);
  writeTest(4u, sarma, 2u);

  let degrees = n - f32(P);
  let scaleFactor = n / weightSum;
  let moranI = scaleFactor * residualLag / residualSquares;
  let moranExpectation = -scaleFactor * traceC / degrees;
  let traceMWMWt = squareSum - traceE - traceD + traceCCTransposed;
  let traceMWMW = crossSum - 2.0 * traceF + traceCC;
  let traceMW = -traceC;
  let moranVariance = scaleFactor * scaleFactor * (traceMWMWt + traceMWMW + traceMW * traceMW) /
    (degrees * (degrees + 2.0)) - moranExpectation * moranExpectation;
  var moranZ = getNaN();
  var moranP = getNaN();
  if (moranVariance > 0.0) {
    moranZ = (moranI - moranExpectation) / sqrt(moranVariance);
    moranP = erfcValue(abs(moranZ) * 0.70710678);
  }
  testsOut[testsOutOffset + 15u] = moranZ;
  testsOut[testsOutOffset + 16u] = 0.0;
  testsOut[testsOutOffset + 17u] = moranP;

  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = traceT;
  summaryOut[summaryOutOffset + 3u] = information;
  summaryOut[summaryOutOffset + 4u] = weightSum;
  summaryOut[summaryOutOffset + 5u] = moranI;
  summaryOut[summaryOutOffset + 6u] = moranExpectation;
  summaryOut[summaryOutOffset + 7u] = moranVariance;
  summaryOut[summaryOutOffset + 8u] = moranZ;
  summaryOut[summaryOutOffset + 9u] = moranP;
  summaryOut[summaryOutOffset + 10u] = residualLag;
  summaryOut[summaryOutOffset + 11u] = responseLag;
  summaryOut[summaryOutOffset + 12u] =
    f32(ROWS) - n;
  statusOut[statusOutOffset] = 0u;
}
`}function Te(e){return Math.floor(16/(e+1))}var Ee=`GPUSpatialTwoStageLeastSquares`,De=64,Oe=4096,ke=64,Y=4,Ae=5,je=class{id;props;rowCount;tileRowCount;tileCount;instrumentOrder;constructor(e){let t=e.id??`spatial-two-stage-least-squares`;this.id=t,this.props=e;let{predictorCount:n,output:r}=e,i=e.instrumentOrder??1;if(i!==1&&i!==2)throw Error(`${t} instrumentOrder must be 1 or 2`);let a=Te(i);if(!Number.isInteger(n)||n<1||n>a)throw Error(`${t} predictorCount must be an integer in [1, ${a}] with instrumentOrder ${i}`);let o=D(t,e.weights);if(x(e.predictors,[`float32`],`${t} predictors`),x(e.response,[`float32`],`${t} response`),e.response.length!==o)throw Error(`${t} response length must equal the weights row count`);if(e.predictors.length!==o*n)throw Error(`${t} predictors length must equal the weights row count * predictorCount`);if(x(r.table,[`float32`],`${t} output.table`),r.table.length<(n+2)*4)throw Error(`${t} output.table must hold (predictorCount + 2) * 4 float32 values`);if(x(r.summary,[`float32`],`${t} output.summary`),r.summary.length<16)throw Error(`${t} output.summary must hold 16 float32 values`);if(x(r.status.code,[`uint32`],`${t} output.status.code`),r.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);if(r.residuals&&(x(r.residuals,[`float32`],`${t} output.residuals`),r.residuals.length<o))throw Error(`${t} output.residuals must hold one row per input row`);this.instrumentOrder=i,e.tileRowCount!==void 0&&h(t,`tileRowCount`,e.tileRowCount),this.rowCount=o,this.tileRowCount=e.tileRowCount??Math.max(ke,Math.ceil(o/Oe)),this.tileCount=Math.ceil(o/this.tileRowCount),g(t,[r.table,r.summary,r.status.code,r.residuals],[e.predictors,e.response,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(e){let{id:t,props:n,rowCount:r,tileRowCount:i,tileCount:a,instrumentOrder:o}=this,{weights:s,output:c,predictorCount:l}=n;m(t,e,[n.predictors,n.response,s.offsets,s.neighbors,s.weights,c.table,c.summary,c.status.code,c.residuals]);let u=(o+1)*l,d=l+1,f=u*u+u*d+u+d,p=(o+1)*l+2,h=p+d+1,g=(n,r)=>b(e,`${t}-${n}`,`float32`,r),_=g(`partial-means`,a*p),v=g(`partial-moments`,a*f),y=g(`moments`,f),x=g(`workspace`,h),S=g(`partial-residuals`,a*Y),w=g(`residual-sums`,Y),T=Ae+d,E=g(`partial-weights-norm`,a),D=g(`partial-anselin-kelejian`,a*T),O=g(`anselin-kelejian-sums`,T),k=c.residuals??g(`residuals`,r),j={offsets:b(e,`${t}-transpose-offsets`,`uint32`,r+1),neighbors:b(e,`${t}-transpose-neighbors`,`uint32`,s.neighbors.length),weights:g(`transpose-weights`,s.neighbors.length)},M=g(`transposed-lag-residual`,r),N=`const ROWS: u32 = ${r}u;
const K: u32 = ${l}u;
const ORDER: u32 = ${o}u;
const M: u32 = ${u}u;
const Q: u32 = ${d}u;
const TILE_ROWS: u32 = ${i}u;
const TILE_COUNT: u32 = ${a}u;
const MEAN_COUNT: u32 = ${p}u;
const MOMENTS: u32 = ${f}u;
const W_DELTA: u32 = ${p}u;
const W_NORM: u32 = ${p+d}u;`,P=e=>`var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + tile * ${e}u + index] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`,F=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),I=[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},F(`weights`,s.weights,`read`),F(`design`,n.predictors,`read`),F(`response`,n.response,`read`)],L=(n,r,i,a,o,s)=>C(e,{id:`${t}-${n}`,operation:Ee,variant:n,bindings:r,invocationCount:i,declarations:a,body:o,...s?{workgroupSize:s}:{}}),R=`${N}
${Me()}`,z=`${R}
${Ne()}`,B=new A({id:`${t}-transpose`,weights:s,output:j}).getCommandNodes(e);return[L(`means-tiles`,[...I,F(`partial`,_,`read_write`)],a,R,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${p}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${l}>;
    var twiceLagged: array<f32, ${l}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, ORDER == 2u);
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
      sums[K + column] = sums[K + column] + lagged[column];
      if (ORDER == 2u) {
        sums[2u * K + 2u + column] = sums[2u * K + 2u + column] + twiceLagged[column];
      }
    }
    sums[2u * K] = sums[2u * K] + lagResponse;
    sums[2u * K + 1u] = sums[2u * K + 1u] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < MEAN_COUNT; slot++) {
    partial[partialOffset + index * MEAN_COUNT + slot] = sums[slot];
  }`,De),L(`means-merge`,[F(`partial`,_,`read`),F(`workspace`,x,`read_write`)],p,N,`${P(p)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);`),L(`weights-norm-tiles`,[I[0],I[1],I[2],F(`partial`,E,`read_write`)],a,N,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        rowSum += abs(weights[weightsOffset + slot]);
      }
    }
    largest = max(largest, rowSum);
  }
  partial[partialOffset + index] = largest;`,De),L(`weights-norm-merge`,[F(`partial`,E,`read`),F(`workspace`,x,`read_write`)],1,N,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + tile]);
  }
  workspace[workspaceOffset + W_NORM] = largest;`),L(`moments-tiles`,[...I,F(`workspace`,x,`read`),F(`partial`,v,`read_write`)],a,z,`let base = partialOffset + index * MOMENTS;
  for (var slot = 0u; slot < MOMENTS; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${u}>;
    var z: array<f32, ${d}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    for (var a = 0u; a < M; a++) {
      for (var b = 0u; b < M; b++) {
        partial[base + a * M + b] += h[a] * h[b];
      }
      for (var b = 0u; b < Q; b++) {
        partial[base + M * M + a * Q + b] += h[a] * z[b];
      }
      partial[base + M * M + M * Q + a] += h[a] * centeredResponse;
    }
    for (var b = 0u; b < Q; b++) {
      partial[base + M * M + M * Q + M + b] += z[b] * z[b];
    }
  }`,De),L(`moments-merge`,[F(`partial`,v,`read`),F(`moments`,y,`read_write`)],f,N,`${P(f)}
  moments[momentsOffset + index] = sum;`),L(`solve`,[F(`moments`,y,`read`),F(`workspace`,x,`read_write`)],1,`${N}
${Pe(u,d)}`,`var delta: array<f32, ${d}>;
  var inverseDiagonal: array<f32, ${d}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${d}>;
  var unusedQuadratic = 0.0;
  var unusedDiagnostics: array<f32, 6>;
  let ok = solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic, &unusedDiagnostics, false);
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_DELTA + column] = select(getNaN(), delta[column], ok);
  }`),L(`residuals-tiles`,[...I,F(`workspace`,x,`read`),F(`residualsOut`,k,`read_write`),F(`partial`,S,`read_write`)],a,z,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${Y}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${u}>;
    var z: array<f32, ${d}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, false);
    var residual = centeredResponse;
    for (var b = 0u; b < Q; b++) {
      residual = residual - z[b] * workspace[workspaceOffset + W_DELTA + b];
    }
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += residual * residual;
    sums[1] += residual;
    sums[2] += centeredResponse * residual;
    sums[3] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${Y}u; slot++) {
    partial[partialOffset + index * ${Y}u + slot] = sums[slot];
  }`,De),L(`residuals-merge`,[F(`partial`,S,`read`),F(`residualSums`,w,`read_write`)],Y,N,`${P(Y)}
  residualSums[residualSumsOffset + index] = sum;`),...B,L(`anselin-kelejian-tiles`,[...I,F(`workspace`,x,`read`),F(`residuals`,k,`read`),F(`partial`,D,`read_write`)],a,R,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${T}>;
  for (var row = firstRow; row < lastRow; row++) {
    let residual = residuals[residualsOffset + row];
    var rowSum = 0.0;
    var lagResidual = 0.0;
    var squareSum = 0.0;
    var crossSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      // Reverse weight w_ji by binary search of this row in the neighbor's sorted row.
      var reverse = 0.0;
      var low = offsets[offsetsOffset + neighbor];
      var high = offsets[offsetsOffset + neighbor + 1u];
      while (low < high) {
        let middle = (low + high) / 2u;
        let candidate = neighbors[neighborsOffset + middle];
        if (candidate == row) {
          reverse = weights[weightsOffset + middle];
          break;
        }
        if (candidate < row) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      let neighborResidual = residuals[residualsOffset + neighbor];
      rowSum += weight;
      lagResidual += weight * neighborResidual;
      squareSum += weight * weight;
      crossSum += weight * reverse;
    }
    sums[0] += residual * lagResidual;
    sums[1] += residual * rowSum;
    sums[2] += rowSum;
    sums[3] += squareSum;
    sums[4] += crossSum;
  }
  // The u'W z products (slots after the fixed ones) come from the next kernel.
  for (var slot = 0u; slot < ${Ae}u; slot++) {
    partial[partialOffset + index * ${T}u + slot] = sums[slot];
  }`,De),L(`transposed-lag-residual`,[{name:`offsets`,view:j.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:j.neighbors,type:`u32`,access:`read`},F(`weights`,j.weights,`read`),F(`residuals`,k,`read`),F(`lagOut`,M,`read_write`)],r,N,`var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot] * residuals[residualsOffset + neighbors[neighborsOffset + slot]];
  }
  lagOut[lagOutOffset + index] = sum;`),L(`anselin-kelejian-products`,[...I,F(`workspace`,x,`read`),F(`transposedLagResidual`,M,`read`),F(`partial`,D,`read_write`)],a,R,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${d}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${l}>;
    var twiceLagged: array<f32, ${l}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, false);
    let product = transposedLagResidual[transposedLagResidualOffset + row];
    for (var column = 0u; column < K; column++) {
      let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
      sums[column] += product * centered;
    }
    sums[K] += product * (lagResponse - workspace[workspaceOffset + 2u * K]);
  }
  for (var column = 0u; column < Q; column++) {
    partial[partialOffset + index * ${T}u + ${Ae}u + column] = sums[column];
  }`,De),L(`anselin-kelejian-merge`,[F(`partial`,D,`read`),F(`akSums`,O,`read_write`)],T,N,`${P(T)}
  akSums[akSumsOffset + index] = sum;`),L(`finish`,[F(`moments`,y,`read`),F(`workspace`,x,`read`),F(`residualSums`,w,`read`),F(`akSums`,O,`read`),F(`tableOut`,c.table,`read_write`),F(`summaryOut`,c.summary,`read_write`),{name:`statusOut`,view:c.status.code,type:`u32`,access:`read_write`}],1,`${N}
${Pe(u,d)}
${Fe(l)}`,`finish();`)]}};function Me(){return`
fn getLags(
  row: u32,
  lagged: ptr<function, array<f32, K>>,
  twiceLagged: ptr<function, array<f32, K>>,
  needSecond: bool
) -> f32 {
  var lagResponse = 0.0;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    lagResponse += weight * response[responseOffset + neighbor];
    for (var column = 0u; column < K; column++) {
      (*lagged)[column] += weight * design[designOffset + neighbor * K + column];
    }
    if (needSecond) {
      var inner: array<f32, K>;
      for (var second = offsets[offsetsOffset + neighbor]; second < offsets[offsetsOffset + neighbor + 1u]; second++) {
        let farther = neighbors[neighborsOffset + second];
        if (farther >= ROWS) {
          continue;
        }
        let farWeight = weights[weightsOffset + second];
        for (var column = 0u; column < K; column++) {
          inner[column] += farWeight * design[designOffset + farther * K + column];
        }
      }
      for (var column = 0u; column < K; column++) {
        (*twiceLagged)[column] += weight * inner[column];
      }
    }
  }
  return lagResponse;
}
`}function Ne(){return`
// h = [x - mean(x), W x - mean(W x)], z = [x - mean(x), W y - mean(W y)].
fn getCenteredRow(
  row: u32,
  h: ptr<function, array<f32, M>>,
  z: ptr<function, array<f32, Q>>,
  centeredResponse: ptr<function, f32>,
  needSecond: bool
) {
  var lagged: array<f32, K>;
  var twiceLagged: array<f32, K>;
  let lagResponse = getLags(row, &lagged, &twiceLagged, needSecond);
  for (var column = 0u; column < K; column++) {
    let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
    (*h)[column] = centered;
    (*z)[column] = centered;
    (*h)[K + column] = lagged[column] - workspace[workspaceOffset + K + column];
    if (ORDER == 2u && needSecond) {
      (*h)[2u * K + column] = twiceLagged[column] - workspace[workspaceOffset + 2u * K + 2u + column];
    }
  }
  (*z)[K] = lagResponse - workspace[workspaceOffset + 2u * K];
  *centeredResponse = response[responseOffset + row] - workspace[workspaceOffset + 2u * K + 1u];
}
`}function Pe(e,t){return`
${E}

${w}

fn regressorMean(column: u32) -> f32 {
  return workspace[workspaceOffset + select(column, 2u * K, column == K)];
}

${s(e)}
${t===e?``:s(t)}
${[...new Set([e,t])].map(e=>o(e)).join(`
`)}
${c(t)}

// With wantDiagnostics, diagnostics = [Belsley condition number of [1, PZ], correlation condition
// number of Z'PZ, largest variance inflation factor of Z'PZ, correlation condition number of the
// instruments, first-stage F of the excluded instruments, their partial R squared].
fn solveSystem(
  delta: ptr<function, array<f32, ${t}>>,
  inverseDiagonal: ptr<function, array<f32, ${t}>>,
  meanQuadratic: ptr<function, f32>,
  second: ptr<function, array<f32, ${t}>>,
  secondQuadratic: ptr<function, f32>,
  diagnostics: ptr<function, array<f32, 6>>,
  wantDiagnostics: bool
) -> bool {
  let instrumentScalars = M * M + M * Q;
  var instrumentScale: array<f32, ${e}>;
  var regressorScale: array<f32, ${t}>;
  for (var a = 0u; a < M; a++) {
    let diagonal = moments[momentsOffset + a * M + a];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    instrumentScale[a] = 1.0 / sqrt(diagonal);
  }
  for (var b = 0u; b < Q; b++) {
    let diagonal = moments[momentsOffset + instrumentScalars + M + b];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    regressorScale[b] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  for (var a = 0u; a < M; a++) {
    for (var c = 0u; c < M; c++) {
      factor[a * M + c] = moments[momentsOffset + a * M + c] * instrumentScale[a] * instrumentScale[c];
    }
  }
  if (wantDiagnostics) {
    var instrumentCorrelation = factor;
    let instrumentRange = getSymmetricEigenvalueRange_${e}(&instrumentCorrelation);
    (*diagnostics)[3] = instrumentRange.y / max(instrumentRange.x, 1.0e-30);
  }
  if (!choleskyFactor_${e}(&factor)) {
    return false;
  }
  if (wantDiagnostics) {
    // First stage of W y on the instruments: explained sums of squares from the forward
    // substitution y = L^-1 (scaled H'z); the first K instruments are the included X.
    var reduced: array<f32, ${e}>;
    for (var a = 0u; a < M; a++) {
      reduced[a] = moments[momentsOffset + M * M + a * Q + K] * instrumentScale[a];
    }
    var explainedFull = 0.0;
    var explainedRestricted = 0.0;
    for (var row = 0u; row < M; row++) {
      var value = reduced[row];
      for (var j = 0u; j < row; j++) {
        value = value - factor[row * M + j] * reduced[j];
      }
      value = value / factor[row * M + row];
      reduced[row] = value;
      explainedFull += value * value;
      if (row < K) {
        explainedRestricted += value * value;
      }
    }
    let total = moments[momentsOffset + instrumentScalars + M + K];
    let excluded = explainedFull - explainedRestricted;
    (*diagnostics)[4] = (excluded / f32(M - K)) / ((total - explainedFull) / (f32(ROWS) - f32(M) - 1.0));
    (*diagnostics)[5] = excluded / (total - explainedRestricted);
  }
  // x1 = (H'H)^-1 H'Z column by column, x2 = (H'H)^-1 H'y (scaled instrument space).
  var projectedInstruments: array<f32, ${e*t}>;
  for (var b = 0u; b < Q; b++) {
    var rhs: array<f32, ${e}>;
    for (var a = 0u; a < M; a++) {
      rhs[a] = moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b];
    }
    choleskySolve_${e}(&factor, &rhs);
    for (var a = 0u; a < M; a++) {
      projectedInstruments[a * Q + b] = rhs[a];
    }
  }
  var projectedResponse: array<f32, ${e}>;
  for (var a = 0u; a < M; a++) {
    projectedResponse[a] = moments[momentsOffset + M * M + M * Q + a] * instrumentScale[a];
  }
  choleskySolve_${e}(&factor, &projectedResponse);
  var normal: array<f32, ${t*t}>;
  var rhs: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    var rhsValue = 0.0;
    for (var a = 0u; a < M; a++) {
      rhsValue += moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b] * projectedResponse[a];
    }
    rhs[b] = rhsValue;
    for (var c = 0u; c < Q; c++) {
      var sum = 0.0;
      for (var a = 0u; a < M; a++) {
        sum += moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b] * projectedInstruments[a * Q + c];
      }
      normal[b * Q + c] = sum;
    }
  }
  var normalDiagonal: array<f32, ${t}>;
  if (wantDiagnostics) {
    for (var b = 0u; b < Q; b++) {
      normalDiagonal[b] = normal[b * Q + b];
    }
    // Correlation condition number of Z'PZ (unit diagonal) and Belsley's of the raw [1, PZ]
    // design with unit columns, from the Cholesky factor of the correlation matrix and the column
    // means (never the raw Gram matrix; see getBelsleyConditionNumberWGSL).
    var secondCorrelation: array<f32, ${t*t}>;
    var alpha: array<f32, ${t}>;
    var beta: array<f32, ${t}>;
    let sampleSize = f32(ROWS);
    for (var b = 0u; b < Q; b++) {
      let meanB = regressorMean(b);
      let centered = normal[b * Q + b] / (regressorScale[b] * regressorScale[b]);
      let rawNorm = sqrt(sampleSize * meanB * meanB + centered);
      alpha[b] = sqrt(sampleSize) * meanB / rawNorm;
      beta[b] = sqrt(centered) / rawNorm;
      for (var c = 0u; c < Q; c++) {
        secondCorrelation[b * Q + c] = select(normal[b * Q + c] / sqrt(normal[b * Q + b] * normal[c * Q + c]), 1.0, b == c);
      }
    }
    var correlationEigen = secondCorrelation;
    let correlationRange = getSymmetricEigenvalueRange_${t}(&correlationEigen);
    (*diagnostics)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
    var belsley = 3.0e38;
    if (choleskyFactor_${t}(&secondCorrelation)) {
      belsley = getBelsleyConditionNumber_${t}(&secondCorrelation, &alpha, &beta);
    }
    (*diagnostics)[0] = belsley;
  }
  if (!choleskyFactor_${t}(&normal)) {
    return false;
  }
  if (wantDiagnostics) {
    var inflation = 0.0;
    for (var b = 0u; b < Q; b++) {
      inflation = max(inflation, choleskyInverseDiagonal_${t}(&normal, b) * normalDiagonal[b]);
    }
    (*diagnostics)[2] = inflation;
  }
  choleskySolve_${t}(&normal, &rhs);
  for (var b = 0u; b < Q; b++) {
    (*delta)[b] = rhs[b] * regressorScale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${t}(&normal, b) * regressorScale[b] * regressorScale[b];
  }
  // Quadratic form of the regressor means, in the scaled space.
  var meansScaled: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    meansScaled[b] = regressorMean(b) * regressorScale[b];
  }
  var solved = meansScaled;
  choleskySolve_${t}(&normal, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    quadratic += meansScaled[b] * solved[b];
  }
  *meanQuadratic = quadratic;
  var secondScaled: array<f32, ${t}>;
  for (var b = 0u; b < Q; b++) {
    secondScaled[b] = (*second)[b] * regressorScale[b];
  }
  var secondSolved = secondScaled;
  choleskySolve_${t}(&normal, &secondSolved);
  var secondSum = 0.0;
  for (var b = 0u; b < Q; b++) {
    secondSum += secondScaled[b] * secondSolved[b];
  }
  *secondQuadratic = secondSum;
  return true;
}
`}function Fe(e){return`
${T}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(e+2)*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 16u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  let degrees = n - f32(Q + 1u);
  if (!(degrees > 0.0)) {
    fail(2u);
    return;
  }
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseResidual) || !isFiniteValue(responseSquares)) {
    fail(select(3u, 1u, !isFiniteValue(workspace[workspaceOffset + W_DELTA])));
    return;
  }
  var delta: array<f32, ${e+1}>;
  var inverseDiagonal: array<f32, ${e+1}>;
  var meanQuadratic = 0.0;
  var productVector: array<f32, ${e+1}>;
  for (var b = 0u; b < Q; b++) {
    productVector[b] = akSums[akSumsOffset + ${Ae}u + b];
  }
  var productQuadratic = 0.0;
  var diagnostics: array<f32, 6>;
  if (!solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &productVector, &productQuadratic, &diagnostics, true)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSquares / n;
  // Intercept: ybar - mean(Z) delta. Its variance is sigma2 / n plus the delta part.
  var intercept = workspace[workspaceOffset + 2u * K + 1u];
  for (var b = 0u; b < Q; b++) {
    intercept -= regressorMean(b) * delta[b];
  }
  let interceptVariance = sigmaSquared * (1.0 / n + meanQuadratic);
  for (var row = 0u; row < Q + 1u; row++) {
    // Rows 0 is the intercept; rows 1..K are the predictors; row K + 1 is rho (delta index K).
    let index = max(row, 1u) - 1u;
    let coefficient = select(delta[index], intercept, row == 0u);
    let variance = select(sigmaSquared * inverseDiagonal[index], interceptVariance, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = residualSquares;
  summaryOut[summaryOutOffset + 3u] = covariance * covariance / (responseSquares * predictedVariance);
  // Anselin-Kelejian (1997) as in spreg akTest: Moran's I of u, corrected by the estimation variance.
  let residualLagProduct = akSums[akSumsOffset];
  let rowSumProduct = akSums[akSumsOffset + 1u];
  let weightsSum = akSums[akSumsOffset + 2u];
  let traceT = akSums[akSumsOffset + 3u] + akSums[akSumsOffset + 4u];
  let moranI = (n * residualLagProduct) / (weightsSum * residualSquares);
  let estimationTerm = rowSumProduct * rowSumProduct / n + productQuadratic;
  let weightScale = weightsSum / n;
  let phiSquared = (traceT + 4.0 * estimationTerm / sigmaSquared) / (weightScale * weightScale * n);
  let anselinKelejian = n * moranI * moranI / phiSquared;
  summaryOut[summaryOutOffset + 4u] = moranI;
  summaryOut[summaryOutOffset + 5u] = anselinKelejian;
  summaryOut[summaryOutOffset + 6u] = erfcValue(sqrt(max(anselinKelejian, 0.0) * 0.5));
  // Conditioning of the second-stage and instrument systems; a warning never changes the status.
  summaryOut[summaryOutOffset + 7u] = diagnostics[0];
  summaryOut[summaryOutOffset + 8u] = diagnostics[1];
  summaryOut[summaryOutOffset + 9u] = diagnostics[2];
  summaryOut[summaryOutOffset + 11u] = diagnostics[3];
  summaryOut[summaryOutOffset + 10u] =
    select(0.0, 1.0, diagnostics[1] > ${S(f)} || diagnostics[3] > ${S(f)});
  summaryOut[summaryOutOffset + 14u] = diagnostics[4];
  summaryOut[summaryOutOffset + 15u] = diagnostics[5];
  // rho is the last delta; admissible when |rho| ||W||_inf < 1 (NaN compares false: flagged).
  let weightsNorm = workspace[workspaceOffset + W_NORM];
  summaryOut[summaryOutOffset + 13u] = weightsNorm;
  summaryOut[summaryOutOffset + 12u] =
    select(1.0, 0.0, abs(delta[K]) * weightsNorm < 1.0);
  statusOut[statusOutOffset] = 0u;
}
`}function Ie(e){if(e.length<16)throw Error(`Spatial two-stage least squares summary must hold 16 values`);return{firstStageF:e[14],firstStagePartialRSquared:e[15],rhoOutsideAdmissibleRange:e[12]===1,weightsInfinityNorm:e[13],instrumentCorrelationConditionNumber:e[11]}}function Le(e,t={}){let{table:n,summary:i}=e;if(i.length<16)throw Error(`Spatial two-stage least squares summary must hold 16 values`);let o=n.length/4-2;if(!Number.isInteger(o)||o<1)throw Error(`Spatial two-stage least squares table must hold (predictorCount + 2) * 4 values`);let s=d(o,t.predictorNames);s.push(`W_y`);let c=s.map((e,t)=>{let r=n[t*4+2];return{name:e,role:t===0?`intercept`:t===s.length-1?`spatial-lag`:`predictor`,estimate:n[t*4],standardError:n[t*4+1],statistic:r,statisticKind:`z`,pValue:Number.isNaN(r)?NaN:a(r*r,1)}}),l=i[7],u=i[8],f=i[11],p=Number.isFinite(l)&&Number.isFinite(u),m=i[5];return r({kind:`spatial-lag-two-stage-least-squares`,coefficients:c,rowCount:i[0],residualSumOfSquares:i[2],standardErrorSigmaSquared:`maximum-likelihood`,statusCode:e.status[0],pseudoRSquared:i[3],logLikelihood:null,conditioning:p?{conditionNumber:l,correlationConditionNumber:Number.isFinite(f)?Math.max(u,f):u,maximumVarianceInflationFactor:i[9]}:null,tests:{anselinKelejian:{statistic:m,degreesOfFreedom:1,pValue:i[6]}}})}var Re=16384,ze=8*1024*1024,Be=F.spatialWeightsTracesProbeCounter.id,Ve=F.spatialWeightsTracesProbeKey.id;function He(e){return e?`${P}
fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  let randomBlock = philox4x32(
    vec4<u32>(probe, row >> 2u, 0u, ${Be}u),
    vec2<u32>(seed, ${Ve}u)
  );
  let word = randomBlock[row & 3u];
  return select(1.0, -1.0, (word & 1u) == 1u);
}`:`fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  _ = seed;
  return select(0.0, 1.0, row == probe);
}`}var Ue=(e,t,n)=>({name:e,view:t,type:`u32`,access:n});function We(e,t,n){return(r,i,a,o,s,c={})=>C(e,{id:`${t}-${r}`,operation:n,variant:r.replace(/-\d+(-\d+)?$/,``),bindings:i,invocationCount:a,declarations:o,body:s,...c.constants?{constants:c.constants}:{},...c.workgroupSize?{workgroupSize:c.workgroupSize}:{}})}function Ge(e){return Math.ceil(e/64)*64}function X(e,n,r,i){return t(r%64==0),e.createDataView(n.buffer,{format:n.format,length:i,byteOffset:n.byteOffset+r*n.rowByteLength})}function Ke(e,t,n,r,i,a){let o=Math.ceil(r/i),s=Ge(i),c=b(e,`${t}-batch-information`,`uint32`,o*64);return{node:We(e,t,n)(`batch-information`,[Ue(`batchInformation`,c,`read_write`),...a?[Ue(`seedParameters`,a,`read`)]:[]],o,`const BATCH_PROBES: u32 = ${i}u;
const PROBE_TOTAL: u32 = ${r}u;
const SLOT_WORDS: u32 = 64u;`,`let start = index * BATCH_PROBES;
  batchInformation[batchInformationOffset + index * SLOT_WORDS] = start;
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 1u] = min(BATCH_PROBES, PROBE_TOTAL - start);
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 2u] = ${a?`seedParameters[seedParametersOffset]`:`0u`};`),batchProbeCount:i,batchCount:o,blockStride:s,getBatchInformation:t=>X(e,c,t*64,3),getBlock:(t,n)=>X(e,t,n*s,i),getSumWGSL:(e,t=`sum`,n=`0u`)=>`var ${t} = 0.0;
  var ${t}Compensation = 0.0;
  for (var sumBatch = 0u; sumBatch < ${o}u; sumBatch++) {
    let sumCount = min(${i}u, ${r}u - sumBatch * ${i}u);
    for (var sumItem = 0u; sumItem < sumCount; sumItem++) {
      let term = ${e}[${e}Offset + ${n} + sumBatch * ${s}u + sumItem] - ${t}Compensation;
      let next = ${t} + term;
      ${t}Compensation = (next - ${t}) - term;
      ${t} = next;
    }
  }`}}var qe=`GPUSpatialWeightsTraces`,Je=64,Ye=32,Xe=1024;function Ze(e,t,n){if(n&&(x(n,[`uint32`],`${e} ${t}`),n.length<1))throw Error(`${e} ${t} must hold 1 uint32`)}var Qe=class{id;props;rowCount;mode;probeTotal;batchProbeCount;tileRowCount;tileCount;includeGramTraces;constructor(e){let t=e.id??`spatial-weights-traces`;this.id=t,this.props=e;let{output:n,maximumPower:r}=e,i=D(t,e.weights);if(h(t,`maximumPower`,r,1,256),e.mode!==void 0&&e.mode!==`exact`&&e.mode!==`stochastic`)throw Error(`${t} mode must be 'exact' or 'stochastic'`);let a=e.mode??(i<=4096?`exact`:`stochastic`);if(a===`exact`&&i>16384)throw Error(`${t} exact mode supports at most ${Re} rows; use mode 'stochastic'`);let o=e.probeCount??64;if(h(t,`probeCount`,o),Ze(t,`parameters`,e.parameters),e.tileRowCount!==void 0&&h(t,`tileRowCount`,e.tileRowCount),x(n.powerTraces,[`float32`],`${t} output.powerTraces`),n.powerTraces.length<r)throw Error(`${t} output.powerTraces must hold maximumPower float32 values`);if(x(n.powerSums,[`float32`],`${t} output.powerSums`),n.powerSums.length<r)throw Error(`${t} output.powerSums must hold maximumPower float32 values`);let s=e.includeGramTraces??n.gramTraces!==void 0;if(s!==(n.gramTraces!==void 0))throw Error(`${t} output.gramTraces is required exactly when includeGramTraces is true`);if(n.gramTraces&&(x(n.gramTraces,[`float32`],`${t} output.gramTraces`),n.gramTraces.length<4))throw Error(`${t} output.gramTraces must hold 4 float32 values`);if(n.infinityNorm&&(x(n.infinityNorm,[`float32`],`${t} output.infinityNorm`),n.infinityNorm.length<1))throw Error(`${t} output.infinityNorm must hold one float32 value`);if(n.status?.approximation&&(x(n.status.approximation,[`uint32`],`${t} output.status.approximation`),n.status.approximation.length<1))throw Error(`${t} output.status.approximation must hold one uint32 row`);this.rowCount=i,this.mode=a,this.includeGramTraces=s,this.probeTotal=a===`exact`?i:o,e.probeBatchSize!==void 0&&h(t,`probeBatchSize`,e.probeBatchSize),this.batchProbeCount=Math.min(this.probeTotal,e.probeBatchSize??1/0,Math.max(1,Math.floor(ze/i))),this.tileRowCount=e.tileRowCount??Math.max(Ye,Math.ceil(i/Xe)),this.tileCount=Math.ceil(i/this.tileRowCount),g(t,[n.powerTraces,n.powerSums,n.gramTraces,n.status?.approximation,n.infinityNorm],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.parameters])}getCommandNodes(e){let{id:t,props:n,rowCount:r,probeTotal:i,batchProbeCount:a,tileRowCount:o,tileCount:s}=this,{weights:c,output:l,maximumPower:u}=n,d=this.mode===`stochastic`,f=this.includeGramTraces;m(t,e,[c.offsets,c.neighbors,c.weights,l.powerTraces,l.powerSums,l.gramTraces,l.status?.approximation,l.infinityNorm,n.parameters]);let p=f?Math.max(u,2):u,h=p,g=h,_=h+1,v=h+(f?2:0),y=Ke(e,t,qe,i,a,n.parameters),x=Ge(s),S=(n,r)=>b(e,`${t}-${n}`,`float32`,r),w=`const ROWS: u32 = ${r}u;
const TILE_ROWS: u32 = ${o}u;
const TILE_COUNT: u32 = ${s}u;
const PROBE_TOTAL: u32 = ${i}u;`,T=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),E=(e,t,n)=>({name:e,view:t,type:`u32`,access:n}),D=(n,r,i,a,o,s,c)=>C(e,{id:`${t}-${n}`,operation:qe,variant:n.replace(/-\d+(-\d+)?$/,``),bindings:r,invocationCount:i,declarations:a,body:o,...s?{constants:s}:{},...c?{workgroupSize:c}:{}}),O=(e,t)=>[E(e?`${e}Offsets`:`offsets`,t.offsets,`read`),E(e?`${e}Neighbors`:`neighbors`,t.neighbors,`read`),T(e?`${e}Weights`:`weights`,t.weights,`read`)],k=(e,t,n,r)=>`var sum = 0.0;
  var compensation = 0.0;
  for (var item = 0u; item < ${r}; item++) {
    let term = ${e}[${e}Offset + ${t} + item * ${n}] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`,j=[],M=O(``,c),N=l.infinityNorm??S(`infinity-norm`,1),P=S(`norm-partial`,s),F=S(`scaled-weights`,c.weights.length);j.push(D(`norm-tiles`,[...M,T(`normPartial`,P,`read_write`)],s,w,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      sum += abs(weights[weightsOffset + slot]);
    }
    largest = max(largest, sum);
  }
  normPartial[normPartialOffset + index] = largest;`,void 0,Je),D(`norm-merge`,[T(`normPartial`,P,`read`),T(`infinityNorm`,N,`read_write`)],1,w,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, normPartial[normPartialOffset + tile]);
  }
  // An empty or non-finite W scales by 1 (non-finite weights then propagate through the chains).
  infinityNorm[infinityNormOffset] = select(1.0, largest, largest > 0.0 && largest < 3.0e38);`),D(`scale-weights`,[T(`weights`,c.weights,`read`),T(`infinityNorm`,N,`read`),T(`scaledWeights`,F,`read_write`)],c.weights.length,``,`scaledWeights[scaledWeightsOffset + index] = weights[weightsOffset + index] / infinityNorm[infinityNormOffset];`));let I=O(``,{offsets:c.offsets,neighbors:c.neighbors,weights:F}),L=S(`sums-partial`,u*x),R=[S(`ones-a`,r),S(`ones-b`,r)];for(let t=1;t<=u;t++){let n=R[(t-2+2)%2],r=R[(t-1)%2],i=t===1;j.push(D(i?`sums-step-first`:`sums-step-${t}`,[...I,...i?[]:[T(`vectorIn`,n,`read`)],T(`vectorOut`,r,`read_write`),T(`sumsPartial`,X(e,L,(t-1)*x,s),`read_write`)],s,w,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var tileSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${i?`1.0`:`vectorIn[vectorInOffset + neighbor]`};
    }
    vectorOut[vectorOutOffset + row] = sum;
    tileSum += sum;
  }
  sumsPartial[sumsPartialOffset + index] = tileSum;`,void 0,Je))}j.push(D(`sums-merge`,[T(`sumsPartial`,L,`read`),T(`powerSums`,l.powerSums,`read_write`)],u,`${w}
const SUMS_STRIDE: u32 = ${x}u;`,`${k(`sumsPartial`,`index * SUMS_STRIDE`,`1u`,`TILE_COUNT`)}
  powerSums[powerSumsOffset + index] = sum;`));let z,B=f?S(`gram-partial`,s*2):void 0;f&&B&&(z={offsets:b(e,`${t}-transpose-offsets`,`uint32`,r+1),neighbors:b(e,`${t}-transpose-neighbors`,`uint32`,c.neighbors.length),weights:S(`transpose-weights`,c.neighbors.length)},j.push(...new A({id:`${t}-transpose`,weights:{offsets:c.offsets,neighbors:c.neighbors,weights:c.weights},output:z}).getCommandNodes(e)),j.push(D(`gram-moments-tiles`,[...M,...O(`transposed`,z),T(`gramPartial`,B,`read_write`)],s,w,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  var columnSquareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      squareSum += weight * weight;
    }
    var columnSquares = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      if (transposedNeighbors[transposedNeighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = transposedWeights[transposedWeightsOffset + slot];
      columnSquares += weight * weight;
    }
    columnSquareSum += columnSquares * columnSquares;
  }
  gramPartial[gramPartialOffset + index * 2u] = squareSum;
  gramPartial[gramPartialOffset + index * 2u + 1u] = columnSquareSum;`,void 0,Je)));let V=y.batchCount*y.blockStride,H=S(`dots`,v*V),U=(t,n)=>y.getBlock(X(e,H,t*V,V),n),ee=d?S(`partial-power`,a*s):void 0,te=f?S(`partial-gram`,a*s):void 0,W=[S(`scratch-a`,a*r),S(`scratch-b`,a*r)];j.push(y.node);let G=He(d),K=(e,t,n,r)=>D(e,[T(`partial`,n,`read`),T(`dots`,U(r,t),`read_write`),E(`batchInformation`,y.getBatchInformation(t),`read`)],a,w,`if (index >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  ${k(`partial`,`index * TILE_COUNT`,`1u`,`TILE_COUNT`)}
  dots[dotsOffset + index] = sum;`,void 0,Je);for(let e=0;e<y.batchCount;e++){let t=E(`batchInformation`,y.getBatchInformation(e),`read`);for(let n=1;n<=p;n++){let r=n===1,i=f&&n===2,o=d&&n<=u,c=o||i,l=W[(n-1)%2],p=W[(n-2+2)%2],m=!d&&n<=u,h=`${r?`first`:`next`}${i?`-gram`:``}${c?`-dot`:``}${m?`-diagonal`:``}`,v=[...I,...r?[]:[T(`vectorIn`,p,`read`)],T(`vectorOut`,l,`read_write`),...o?[T(`partialPower`,ee,`read_write`)]:[],...i?[T(`partialGram`,te,`read_write`)]:[],...m?[T(`dots`,U(n-1,e),`read_write`)]:[],t],y=r?`getProbeValue(probeSeed, globalProbe, neighbor)`:`vectorIn[vectorInOffset + base + neighbor]`;j.push(D(`step-${h}-${e}-${n}`,v,a*s,`${w}
${G}`,`let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var powerDot = 0.0;
  var gramDot = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${y};
    }
    vectorOut[vectorOutOffset + base + row] = sum;
${m?`    if (row == globalProbe) {
      dots[dotsOffset + probe] = sum;
    }
`:``}${o?`    powerDot += getProbeValue(probeSeed, globalProbe, row) * sum;
`:``}${i?`    gramDot += vectorIn[vectorInOffset + base + row] * sum;
`:``}  }
${o?`  partialPower[partialPowerOffset + probe * TILE_COUNT + tile] = powerDot;
`:``}${i?`  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = gramDot;
`:``}  _ = powerDot;
  _ = gramDot;`,void 0,Je)),o&&j.push(K(`probe-merge-${e}-${n}`,e,ee,n-1)),i&&j.push(K(`probe-merge-pair-${e}`,e,te,g)),i&&z&&(j.push(D(`gram-square-${e}`,[...O(`transposed`,z),T(`vectorIn`,p,`read`),T(`partialGram`,te,`read_write`),t],a*s,w,`let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      let neighbor = transposedNeighbors[transposedNeighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += transposedWeights[transposedWeightsOffset + slot] * vectorIn[vectorInOffset + base + neighbor];
    }
    squareSum += sum * sum;
  }
  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = squareSum;`,void 0,Je)),j.push(K(`probe-merge-square-${e}`,e,te,_)))}}let ne=[T(`dots`,H,`read`),T(`infinityNorm`,N,`read`),...B?[T(`gramPartial`,B,`read`)]:[],T(`powerTraces`,l.powerTraces,`read_write`),...l.gramTraces?[T(`gramTraces`,l.gramTraces,`read_write`)]:[]],re=d?`1.0 / f32(PROBE_TOTAL)`:`1.0`;return j.push(D(`final-merge`,ne,v+(f?2:0),`${w}
const POWER_COUNT: u32 = ${u}u;
const DOT_ROWS: u32 = ${v}u;
const POWER_ROWS: u32 = ${h}u;`,`if (index < DOT_ROWS) {
    ${y.getSumWGSL(`dots`,`sum`,`index * ${V}u`)}
    let value = sum * ${re};
    if (index < POWER_COUNT) {
      powerTraces[powerTracesOffset + index] = value;
    }${f?`
    if (index >= POWER_ROWS) {
      // The chains hold W / s: tr(W'W W) = s^3 E[v1' v2] and tr((W'W)^2) = s^2 E[|W' v1|^2].
      let norm = infinityNorm[infinityNormOffset];
      gramTraces[gramTracesOffset + 1u + index - POWER_ROWS] =
        value * select(norm * norm * norm, norm * norm, index > POWER_ROWS);
    }`:``}
  }${f?`
  if (index >= DOT_ROWS) {
    let slot = index - DOT_ROWS;
    ${k(`gramPartial`,`slot`,`2u`,`TILE_COUNT`)}
    gramTraces[gramTracesOffset + select(0u, 3u, slot == 1u)] = sum;
  }`:``}`)),l.status?.approximation&&j.push(D(`approximation`,[E(`approximation`,l.status.approximation,`read_write`)],1,``,`approximation[approximationOffset] = ${d?`1u`:`0u`};`)),j}},$e=.99,et=`GPUSpatialErrorGM`,tt=64,nt=4096,rt=64,Z=7,Q=5,$=3,it=1024,at=40,ot=2,st=class{id;props;rowCount;tileRowCount;tileCount;estimator;constructor(e){let t=e.id??`spatial-error-gm`;this.id=t,this.props=e;let{predictorCount:n,output:r}=e;h(t,`predictorCount`,n,1,15);let i=e.estimator??`homoskedastic`;if(i!==`homoskedastic`&&i!==`kelejian-prucha`)throw Error(`${t} estimator must be 'homoskedastic' or 'kelejian-prucha'`);this.estimator=i,Ze(t,`traceParameters`,e.traceParameters);let a=D(t,e.weights);if(x(e.predictors,[`float32`],`${t} predictors`),x(e.response,[`float32`],`${t} response`),e.response.length!==a)throw Error(`${t} response length must equal the weights row count`);if(e.predictors.length!==a*n)throw Error(`${t} predictors length must equal the weights row count * predictorCount`);if(x(r.table,[`float32`],`${t} output.table`),r.table.length<(n+2)*4)throw Error(`${t} output.table must hold (predictorCount + 2) * 4 float32 values`);if(x(r.summary,[`float32`],`${t} output.summary`),r.summary.length<13)throw Error(`${t} output.summary must hold 13 float32 values`);if(x(r.status.code,[`uint32`],`${t} output.status.code`),r.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);if(r.residuals&&(x(r.residuals,[`float32`],`${t} output.residuals`),r.residuals.length<a))throw Error(`${t} output.residuals must hold one row per input row`);e.tileRowCount!==void 0&&h(t,`tileRowCount`,e.tileRowCount),this.rowCount=a,this.tileRowCount=e.tileRowCount??Math.max(rt,Math.ceil(a/nt)),this.tileCount=Math.ceil(a/this.tileRowCount),g(t,[r.table,r.summary,r.status.code,r.residuals],[e.predictors,e.response,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(e){let{id:t,props:n,rowCount:r,tileRowCount:a,tileCount:l,estimator:u}=this,d=u===`homoskedastic`,{weights:f,output:p,predictorCount:h}=n;m(t,e,[n.predictors,n.response,f.offsets,f.neighbors,f.weights,p.table,p.summary,p.status.code,p.residuals]);let g=h+1,_=g*g+g+1,v=h+1,y=h+4+g,x=y+15,T=(n,r)=>b(e,`${t}-${n}`,`float32`,r),D=T(`ols-residuals`,r),O=b(e,`${t}-ols-status`,`uint32`,1),k=new i({id:`${t}-ols`,predictors:n.predictors,response:n.response,predictorCount:h,tileRowCount:a,output:{coefficients:T(`ols-coefficients`,g),standardErrors:T(`ols-standard-errors`,g),tStatistics:T(`ols-t-statistics`,g),summary:T(`ols-summary`,22),status:{code:O},residuals:D}}).getCommandNodes(e),A=T(`partial-means`,l*v),j=T(`workspace`,x),M=T(`partial-weights-norm`,l),N=T(`lag-residuals`,r),P=T(`partial-moments`,l*Z),F=T(`moments`,Z),I=T(`partial-filtered-moments`,l*_),L=T(`filtered-moments`,_),R=T(`partial-residuals`,l*Q),z=T(`residual-sums`,Q),B=p.residuals??T(`residuals`,r),V=`const ROWS: u32 = ${r}u;
const K: u32 = ${h}u;
const Q: u32 = ${g}u;
const TILE_ROWS: u32 = ${a}u;
const TILE_COUNT: u32 = ${l}u;
const W_YBAR: u32 = ${h}u;
const W_LAMBDA: u32 = ${h+1}u;
const W_OBJECTIVE: u32 = ${h+2}u;
const W_INTERCEPT: u32 = ${h+3}u;
const W_BETA: u32 = ${h+4}u;
const W_BOUND: u32 = ${y}u;
const W_LAMBDA_FINAL: u32 = ${y+1}u;
const W_OBJECTIVE_FINAL: u32 = ${y+2}u;
const W_APPROXIMATION: u32 = ${y+3}u;
const W_MOMENT_MATRIX: u32 = ${y+4}u;
const W_TRACE_TERMS: u32 = ${y+8}u;
const W_POWER_SUMS: u32 = ${y+12}u;`,H=e=>`var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + tile * ${e}u + index] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`,U=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),ee=[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read`},U(`weights`,f.weights,`read`)],te=[...ee,U(`design`,n.predictors,`read`),U(`response`,n.response,`read`),U(`workspace`,j,`read`)],W=(n,r,i,a,o,s)=>C(e,{id:`${t}-${n}`,operation:et,variant:n,bindings:r,invocationCount:i,declarations:a,body:o,...s?{workgroupSize:s}:{}}),G=`${V}
${ct()}`,K=`${E}

${w}`,ne=`${V}
${K}

${s(g)}`,re=(e,t,n,r)=>W(`moments-tiles${e}`,[...ee,U(`residual`,t,`read`),U(`lag`,n,`read`),U(`partial`,r,`read_write`)],l,V,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${Z}>;
  for (var row = firstRow; row < lastRow; row++) {
    let u = residual[residualOffset + row];
    let wu = lag[lagOffset + row];
    var wwu = 0.0;
    var squareSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      wwu += weight * lag[lagOffset + neighbor];
      squareSum += weight * weight;
    }
    sums[0] += u * u;
    sums[1] += u * wu;
    sums[2] += wu * wu;
    sums[3] += u * wwu;
    sums[4] += wu * wwu;
    sums[5] += wwu * wwu;
    sums[6] += squareSum;
  }
  for (var slot = 0u; slot < ${Z}u; slot++) {
    partial[partialOffset + index * ${Z}u + slot] = sums[slot];
  }`,tt),ie=(e,t,n)=>W(`residual-lag${e}`,[...ee,U(`residual`,t,`read`),U(`lag`,n,`read_write`)],r,V,`var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * residual[residualOffset + neighbor];
  }
  lag[lagOffset + index] = sum;`),q=(e,t,n)=>W(`filtered-moments-tiles${e}`,[...te,U(`partial`,n,`read_write`)],l,G,`let lambda = workspace[workspaceOffset + ${t}];
  let base = partialOffset + index * ${_}u;
  for (var slot = 0u; slot < ${_}u; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${g}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    for (var a = 0u; a < Q; a++) {
      for (var b = 0u; b < Q; b++) {
        partial[base + a * Q + b] += f[a] * f[b];
      }
      partial[base + Q * Q + a] += f[a] * ys;
    }
    partial[base + Q * Q + Q] += ys * ys;
  }`,tt),ae=(e,t,n)=>W(`filtered-moments-merge${e}`,[U(`partial`,t,`read`),U(`filteredMoments`,n,`read_write`)],_,V,`${H(_)}
  filteredMoments[filteredMomentsOffset + index] = sum;`),oe=[...k,W(`means-tiles`,[U(`design`,n.predictors,`read`),U(`response`,n.response,`read`),U(`partial`,A,`read_write`)],l,V,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${v}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${v}u; slot++) {
    partial[partialOffset + index * ${v}u + slot] = sums[slot];
  }`,tt),W(`means-merge`,[U(`partial`,A,`read`),U(`workspace`,j,`read_write`)],v,V,`${H(v)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);`),W(`weights-norm-tiles`,[...ee,U(`partial`,M,`read_write`)],l,V,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        rowSum += abs(weights[weightsOffset + slot]);
      }
    }
    largest = max(largest, rowSum);
  }
  partial[partialOffset + index] = largest;`,tt),W(`weights-norm-merge`,[U(`partial`,M,`read`),U(`workspace`,j,`read_write`)],1,`${V}
${K}`,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + tile]);
  }
  let valid = largest > 0.0 && isFiniteValue(largest);
  workspace[workspaceOffset + W_BOUND] = select(getNaN(), ${S($e)} / largest, valid);`)],se=T(`trace-power-traces`,ot),ce=T(`trace-gram-traces`,4),le=T(`trace-infinity-norm`,1),ue=b(e,`${t}-trace-approximation`,`uint32`,1);d&&oe.push(...new Qe({id:`${t}-traces`,weights:f,maximumPower:ot,mode:n.traceMode,probeCount:n.traceProbeCount,parameters:n.traceParameters,includeGramTraces:!0,output:{powerTraces:se,powerSums:T(`trace-power-sums`,ot),gramTraces:ce,infinityNorm:le,status:{approximation:ue}}}).getCommandNodes(e)),oe.push(ie(``,D,N),re(``,D,N,P),W(`moments-merge`,[U(`partial`,P,`read`),U(`moments`,F,`read_write`)],Z,V,`${H(Z)}
  moments[momentsOffset + index] = sum;`),W(`lambda-solve`,[U(`moments`,F,`read`),U(`workspace`,j,`read_write`)],1,d?`${V}
${K}
${dt(!1)}
${lt()}`:`${V}
${K}
${ut()}
${lt()}`,d?`solveFirstStepLambda();`:`solveLambda();`),q(``,`W_LAMBDA`,I),ae(``,I,L),W(`filtered-solve`,[U(`filteredMoments`,L,`read`),U(`workspace`,j,`read_write`)],1,`${ne}
${ft(g,`filteredMoments`,`solveFiltered`,!1)}`,`var beta: array<f32, ${g}>;
  var inverseDiagonal: array<f32, ${g}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
  let ok = solveFiltered(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning);
  var intercept = workspace[workspaceOffset + W_YBAR] + beta[0];
  for (var column = 0u; column < K; column++) {
    intercept -= workspace[workspaceOffset + column] * beta[1u + column];
  }
  workspace[workspaceOffset + W_INTERCEPT] = select(getNaN(), intercept, ok);
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_BETA + column] = select(getNaN(), beta[column], ok);
  }`),W(`residuals-tiles`,[...te,U(`residualsOut`,B,`read_write`),U(`partial`,R,`read_write`)],l,G,`let lambda = workspace[workspaceOffset + W_LAMBDA];
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${Q}>;
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${g}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    var filtered = ys;
    var predicted = workspace[workspaceOffset + W_INTERCEPT];
    for (var column = 0u; column < Q; column++) {
      filtered -= f[column] * workspace[workspaceOffset + W_BETA + column];
    }
    for (var column = 0u; column < K; column++) {
      predicted += design[designOffset + row * K + column] * workspace[workspaceOffset + W_BETA + 1u + column];
    }
    let residual = response[responseOffset + row] - predicted;
    let centeredResponse = response[responseOffset + row] - workspace[workspaceOffset + W_YBAR];
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += filtered * filtered;
    sums[1] += residual;
    sums[2] += residual * residual;
    sums[3] += centeredResponse * residual;
    sums[4] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${Q}u; slot++) {
    partial[partialOffset + index * ${Q}u + slot] = sums[slot];
  }`,tt),W(`residuals-merge`,[U(`partial`,R,`read`),U(`residualSums`,z,`read_write`)],Q,V,`${H(Q)}
  residualSums[residualSumsOffset + index] = sum;`));let de=L;if(d){let e=T(`lag-residuals-fgls`,r),t=T(`partial-moments-fgls`,l*Z),n=T(`moments-fgls`,Z),i=T(`partial-power-sums-first`,l*$),a=T(`power-sums-first`,$),o=T(`partial-power-sums-final`,l*$),s=T(`partial-filtered-moments-final`,l*_),c=T(`filtered-moments-final`,_);de=c;let u=(t,n,r)=>W(`power-sums-tiles${t}`,[U(`residual`,B,`read`),U(`lag`,e,`read`),U(`workspace`,j,`read`),U(`partial`,r,`read_write`)],l,V,`let lambda = workspace[workspaceOffset + ${n}];
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${$}>;
  for (var row = firstRow; row < lastRow; row++) {
    let filtered = residual[residualOffset + row] - lambda * lag[lagOffset + row];
    let square = filtered * filtered;
    sums[0] += square;
    sums[1] += square * filtered;
    sums[2] += square * square;
  }
  for (var slot = 0u; slot < ${$}u; slot++) {
    partial[partialOffset + index * ${$}u + slot] = sums[slot];
  }`,tt);oe.push(ie(`-fgls`,B,e),re(`-fgls`,B,e,t),W(`moments-merge-fgls`,[U(`partial`,t,`read`),U(`moments`,n,`read_write`)],Z,V,`${H(Z)}
  moments[momentsOffset + index] = sum;`),u(`-first`,`W_LAMBDA`,i),W(`power-sums-merge-first`,[U(`partial`,i,`read`),U(`powers`,a,`read_write`)],$,V,`${H($)}
  powers[powersOffset + index] = sum;`),W(`lambda-solve-weighted`,[U(`moments`,n,`read`),U(`powers`,a,`read`),U(`powerTraces`,se,`read`),U(`gramTraces`,ce,`read`),U(`infinityNorm`,le,`read`),{name:`approximation`,view:ue,type:`u32`,access:`read`},U(`workspace`,j,`read_write`)],1,`${V}
${K}
${dt(!0)}
${lt()}`,`solveWeightedLambda();`),u(`-final`,`W_LAMBDA_FINAL`,o),W(`power-sums-merge-final`,[U(`partial`,o,`read`),U(`workspace`,j,`read_write`)],$,V,`${H($)}
  workspace[workspaceOffset + W_POWER_SUMS + index] = sum;`),q(`-final`,`W_LAMBDA_FINAL`,s),ae(`-final`,s,c))}return oe.push(W(`finish`,[U(`filteredMoments`,L,`read`),...d?[U(`filteredMomentsFinal`,de,`read`)]:[],U(`workspace`,j,`read`),U(`residualSums`,z,`read`),{name:`olsStatus`,view:O,type:`u32`,access:`read`},U(`tableOut`,p.table,`read_write`),U(`summaryOut`,p.summary,`read_write`),{name:`statusOut`,view:p.status.code,type:`u32`,access:`read_write`}],1,`${ne}
${o(g)}
${c(g-1)}
${d?`${ft(g,`filteredMoments`,`solveFiltered`,!1)}
${ft(g,`filteredMomentsFinal`,`solveFilteredFinal`,!0)}
${pt()}`:ft(g,`filteredMoments`,`solveFilteredFinal`,!0)}
${mt(h,d)}`,`finish();`)),oe}};function ct(){return`
fn getFilteredRow(row: u32, lambda: f32, f: ptr<function, array<f32, Q>>, ys: ptr<function, f32>) {
  let responseMean = workspace[workspaceOffset + W_YBAR];
  var rowSum = 0.0;
  var lagResponse = 0.0;
  var lagged: array<f32, K>;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    rowSum += weight;
    lagResponse += weight * (response[responseOffset + neighbor] - responseMean);
    for (var column = 0u; column < K; column++) {
      lagged[column] += weight * (design[designOffset + neighbor * K + column] - workspace[workspaceOffset + column]);
    }
  }
  (*f)[0] = 1.0 - lambda * rowSum;
  for (var column = 0u; column < K; column++) {
    let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
    (*f)[1u + column] = centered - lambda * lagged[column];
  }
  *ys = (response[responseOffset + row] - responseMean) - lambda * lagResponse;
}
`}function lt(){return`
fn scanLambda(bound: f32) -> vec2<f32> {
  var bestLambda = -bound;
  var bestObjective = getMomentObjective(-bound);
  let endObjective = getMomentObjective(bound);
  if (endObjective < bestObjective) {
    bestLambda = bound;
    bestObjective = endObjective;
  }
  var previousLambda = -bound;
  var previousSlope = getMomentSlope(-bound);
  for (var scanStep = 1u; scanStep <= ${it}u; scanStep++) {
    let lambda = bound * (-1.0 + 2.0 * f32(scanStep) / ${it}.0);
    let slope = getMomentSlope(lambda);
    if (previousSlope < 0.0 && slope >= 0.0) {
      var low = previousLambda;
      var high = lambda;
      for (var iteration = 0u; iteration < ${at}u; iteration++) {
        let middle = 0.5 * (low + high);
        if (getMomentSlope(middle) < 0.0) {
          low = middle;
        } else {
          high = middle;
        }
      }
      let candidate = 0.5 * (low + high);
      let objective = getMomentObjective(candidate);
      if (objective < bestObjective) {
        bestLambda = candidate;
        bestObjective = objective;
      }
    }
    previousLambda = lambda;
    previousSlope = slope;
  }
  return vec2<f32>(bestLambda, bestObjective);
}
`}function ut(){return`
var<private> momentA: vec3<f32>;
var<private> momentB: vec3<f32>;
var<private> momentG: vec3<f32>;
var<private> momentSigma: vec3<f32>;

// Moment residual with sigma2 at its constrained optimum: least squares in sigma2, clamped at zero.
fn getMomentResidual(lambda: f32) -> vec3<f32> {
  let free = momentG - momentA * lambda - momentB * (lambda * lambda);
  let sigma = max(dot(momentSigma, free) / dot(momentSigma, momentSigma), 0.0);
  return free - momentSigma * sigma;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, residual);
}

// Half the derivative of the objective (envelope theorem: sigma2 is optimal or fixed at zero).
fn getMomentSlope(lambda: f32) -> f32 {
  return -dot(momentA + momentB * (2.0 * lambda), getMomentResidual(lambda));
}

fn solveLambda() {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${Z}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!finite || !isFiniteValue(bound)) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let scale = 1.0 / u2;
  momentSigma = vec3<f32>(1.0, moments[momentsOffset + 6u] / n, 0.0);
  momentA = vec3<f32>(2.0 * uwu, 2.0 * wuwwu, uwwu + wu2) * scale;
  momentB = vec3<f32>(-wu2, -wwu2, -wuwwu) * scale;
  momentG = vec3<f32>(u2, wu2, uwu) * scale;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`}function dt(e){return`
var<private> momentA: vec2<f32>;
var<private> momentB: vec2<f32>;
var<private> momentTarget: vec2<f32>;
var<private> momentWeight: mat2x2<f32>;

fn getMomentResidual(lambda: f32) -> vec2<f32> {
  return momentA * lambda + momentB * (lambda * lambda) - momentTarget;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, momentWeight * residual);
}

// Half the derivative of the objective: (dr/dlambda)' M r.
fn getMomentSlope(lambda: f32) -> f32 {
  return dot(momentA + momentB * (2.0 * lambda), momentWeight * getMomentResidual(lambda));
}

// Fills momentA, momentB, momentTarget from the seven moment sums; false if they are not usable.
fn setMomentSystem() -> bool {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${Z}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  if (!finite) {
    return false;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  momentA = vec2<f32>(2.0 * (wuwwu - ratio * uwu) / scale, uwwu + wu2) / n;
  momentB = vec2<f32>(-(wwu2 - ratio * wu2) / scale, -wuwwu) / n;
  momentTarget = vec2<f32>((wu2 - ratio * u2) / scale, uwu) / n;
  return true;
}
${e?`
var<private> traceTerms: vec4<f32>;

// psi (including the 1 / n) from sigma2 and the fourth moment mu4 of the filtered residuals.
fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

fn getPsiDeterminant(psi: mat2x2<f32>) -> f32 {
  return psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
}

fn invertPsi(psi: mat2x2<f32>) -> mat2x2<f32> {
  return mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / getPsiDeterminant(psi));
}

fn solveWeightedLambda() {
  let n = f32(ROWS);
  let nan = getNaN();
  workspace[workspaceOffset + W_LAMBDA_FINAL] = nan;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = nan;
  workspace[workspaceOffset + W_APPROXIMATION] = f32(approximation[approximationOffset]);
  let bound = workspace[workspaceOffset + W_BOUND];
  var finite = isFiniteValue(bound);
  for (var slot = 0u; slot < ${$}u; slot++) {
    finite = finite && isFiniteValue(powers[powersOffset + slot]);
  }
  for (var slot = 0u; slot < 2u; slot++) {
    finite = finite && isFiniteValue(powerTraces[powerTracesOffset + slot]);
  }
  for (var slot = 0u; slot < 4u; slot++) {
    finite = finite && isFiniteValue(gramTraces[gramTracesOffset + slot]);
  }
  if (!finite || !setMomentSystem()) {
    return;
  }
  // powerTraces hold tr((W/s)^q): tr(W) = s c_1 and tr(W^2) = s^2 c_2.
  let norm = infinityNorm[infinityNormOffset];
  // Trace terms: tr(A1 A1), tr(A1 2 A2), tr(2 A2 A2) and diag(A1)'diag(A1).
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  let scaleSquared = scale * scale;
  traceTerms = vec4<f32>(
    (gramTraces[gramTracesOffset + 2u] - t * ratio) / scaleSquared,
    (2.0 * gramTraces[gramTracesOffset + 1u] - 2.0 * ratio * powerTraces[powerTracesOffset] * norm) / scale,
    powerTraces[powerTracesOffset + 1u] * norm * norm + t,
    (gramTraces[gramTracesOffset + 3u] - t * ratio) / scaleSquared
  );
  // psi at lambda1 from the power sums of u_s = u1 - lambda1 W u1.
  let sigma = powers[powersOffset] / n;
  let psi = getPsi(n, sigma, powers[powersOffset + 2u] / n);
  let determinant = getPsiDeterminant(psi);
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return;
  }
  var weight = invertPsi(psi);
  // Normalize the weight to unit trace: same minimizer, well-scaled objective.
  let weightScale = weight[0][0] + weight[1][1];
  weight = weight * (1.0 / weightScale);
  momentWeight = weight;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA_FINAL] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = best.y * weightScale;
  workspace[workspaceOffset + W_MOMENT_MATRIX] = momentA.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] = momentA.y;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 2u] = momentB.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 3u] = momentB.y;
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_TRACE_TERMS + slot] = traceTerms[slot];
  }
}
`:`
fn solveFirstStepLambda() {
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!isFiniteValue(bound) || !setMomentSystem()) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  momentWeight = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`}
`}function ft(e,t,n,r){let i=e=>`${t}[${t}Offset + ${e}]`;return`
fn ${n}(
  beta: ptr<function, array<f32, ${e}>>,
  inverseDiagonal: ptr<function, array<f32, ${e}>>,
  interceptQuadratic: ptr<function, f32>,
  conditioning: ptr<function, array<f32, 3>>
) -> bool {
  var scale: array<f32, ${e}>;
  for (var a = 0u; a < Q; a++) {
    let diagonal = ${i(`a * Q + a`)};
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    scale[a] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${e*e}>;
  var rhs: array<f32, ${e}>;
  for (var a = 0u; a < Q; a++) {
    for (var b = 0u; b < Q; b++) {
      factor[a * Q + b] = ${i(`a * Q + b`)} * scale[a] * scale[b];
    }
    rhs[a] = ${i(`Q * Q + a`)} * scale[a];
  }
${r?`
  // Correlation condition number of the unit-diagonal matrix that is factored.
  var correlation = factor;
  let correlationRange = getSymmetricEigenvalueRange_${e}(&correlation);
  (*conditioning)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
`:``}
  if (!choleskyFactor_${e}(&factor)) {
    return false;
  }
${r?`
  // Belsley condition number of the raw filtered design [f0, f_j + m_j f0] (f0 is the filtered
  // constant, not orthogonal to the centered columns), without forming its Gram matrix.
  // With the Cholesky factor L of the unit-diagonal Gram of [f0, f_1..f_K] the raw column j is
  // inPlane_j u0 + g_j sum_i L[j][i] e_i (u0 the unit f0, e_i orthonormal), so the helper's
  // alpha_j = inPlane_j / N_j, beta_j = g_j / N_j and factor = L without its first row and column.
  var belsleyFactor: array<f32, ${(e-1)*(e-1)}>;
  var belsleyAlpha: array<f32, ${e-1}>;
  var belsleyBeta: array<f32, ${e-1}>;
  for (var j = 0u; j < K; j++) {
    let column = 1.0 / scale[1u + j];
    let inPlane = workspace[workspaceOffset + j] / scale[0] + column * factor[(1u + j) * Q];
    var tail = 0.0;
    for (var i = 0u; i <= j; i++) {
      belsleyFactor[j * K + i] = factor[(1u + j) * Q + 1u + i];
      tail = tail + factor[(1u + j) * Q + 1u + i] * factor[(1u + j) * Q + 1u + i];
    }
    let rawNorm = sqrt(inPlane * inPlane + column * column * tail);
    belsleyAlpha[j] = inPlane / rawNorm;
    belsleyBeta[j] = column / rawNorm;
  }
  (*conditioning)[0] = getBelsleyConditionNumber_${e-1}(&belsleyFactor, &belsleyAlpha, &belsleyBeta);
  var inflation = 0.0;
  for (var b = 1u; b < Q; b++) {
    inflation = max(inflation, choleskyInverseDiagonal_${e}(&factor, b));
  }
  (*conditioning)[2] = inflation;
`:``}
  choleskySolve_${e}(&factor, &rhs);
  var meansScaled: array<f32, ${e}>;
  meansScaled[0] = scale[0];
  for (var b = 0u; b < K; b++) {
    meansScaled[1u + b] = -workspace[workspaceOffset + b] * scale[1u + b];
  }
  var solved = meansScaled;
  choleskySolve_${e}(&factor, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    (*beta)[b] = rhs[b] * scale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${e}(&factor, b) * scale[b] * scale[b];
    quadratic += meansScaled[b] * solved[b];
  }
  *interceptQuadratic = quadratic;
  return true;
}
`}function pt(){return`
var<private> traceTerms: vec4<f32>;

fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

// Standard error of lambda, NaN if psi is not positive definite.
fn getLambdaStandardError(n: f32, lambda: f32, sigma: f32, mu4: f32) -> f32 {
  for (var slot = 0u; slot < 4u; slot++) {
    traceTerms[slot] = workspace[workspaceOffset + W_TRACE_TERMS + slot];
  }
  let psi = getPsi(n, sigma, mu4);
  let determinant = psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return getNaN();
  }
  let inverse = mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / determinant);
  let j = vec2<f32>(
    workspace[workspaceOffset + W_MOMENT_MATRIX] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 2u],
    workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 3u]
  );
  let quadratic = dot(j, inverse * j);
  return sqrt(1.0 / (quadratic * n));
}
`}function mt(e,t){let n=e=>`summaryOut[summaryOutOffset + ${e}u]`;return`
${T}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(e+2)*4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 13u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  if (!(n > f32(Q + 1u))) {
    fail(2u);
    return;
  }
  if (olsStatus[olsStatusOffset] != 0u) {
    fail(select(1u, 2u, olsStatus[olsStatusOffset] == 2u));
    return;
  }
  let lambda = workspace[workspaceOffset + ${t?`W_LAMBDA_FINAL`:`W_LAMBDA`}];
  if (!isFiniteValue(lambda)) {
    fail(3u);
    return;
  }
  var beta: array<f32, ${e+1}>;
  var inverseDiagonal: array<f32, ${e+1}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
${t?`
  // The reported betas are the FGLS fit at lambda1; their variance uses the system at lambda.
  var unusedInverse: array<f32, ${e+1}>;
  var unusedQuadratic = 0.0;
  if (!solveFiltered(&beta, &unusedInverse, &unusedQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  var finalBeta: array<f32, ${e+1}>;
  if (!solveFilteredFinal(&finalBeta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = workspace[workspaceOffset + W_POWER_SUMS] / n;
  let fourthMoment = workspace[workspaceOffset + W_POWER_SUMS + 2u] / n;
  let lambdaStandardError = getLambdaStandardError(n, lambda, sigmaSquared, fourthMoment);`:`
  if (!solveFilteredFinal(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSums[residualSumsOffset] / n;`}
  let filteredSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let residualSquares = residualSums[residualSumsOffset + 2u];
  let responseResidual = residualSums[residualSumsOffset + 3u];
  let responseSquares = residualSums[residualSumsOffset + 4u];
  if (!isFiniteValue(filteredSquares) || !isFiniteValue(residualSquares) || !isFiniteValue(responseSquares)) {
    fail(3u);
    return;
  }
  for (var row = 0u; row < Q; row++) {
    // Row 0 is the intercept, rows 1..K the predictors.
    let coefficient = select(beta[row], workspace[workspaceOffset + W_INTERCEPT], row == 0u);
    let variance = sigmaSquared * select(inverseDiagonal[row], interceptQuadratic, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  let nan = getNaN();
${t?`
  let lambdaZ = lambda / lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u + 2u] = lambdaZ;
  tableOut[tableOutOffset + Q * 4u + 3u] = erfcValue(abs(lambdaZ) * 0.70710678);`:`
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = nan;
  tableOut[tableOutOffset + Q * 4u + 2u] = nan;
  tableOut[tableOutOffset + Q * 4u + 3u] = nan;`}
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  let bound = workspace[workspaceOffset + W_BOUND];
  summaryOut[summaryOutOffset] = n;
  ${n(1)} = lambda;
  ${n(2)} = sigmaSquared;
  ${n(3)} = covariance * covariance / (responseSquares * predictedVariance);
  ${n(4)} = residualSquares;
  ${n(5)} = workspace[workspaceOffset + ${t?`W_OBJECTIVE_FINAL`:`W_OBJECTIVE`}];
  ${n(6)} = conditioning[0];
  ${n(7)} = conditioning[1];
  ${n(8)} = conditioning[2];
  ${n(9)} =
    select(0.0, 1.0, conditioning[1] > ${S(f)});
  ${n(10)} = bound;
  ${n(11)} = ${t?`workspace[workspaceOffset + W_APPROXIMATION]`:`0.0`};
  ${n(12)} = select(0.0, 1.0, abs(lambda) >= bound * 0.9999);
  statusOut[statusOutOffset] = 0u;
}
`}function ht(e,t={}){let{table:n,summary:i}=e;if(i.length<13)throw Error(`Spatial error GM summary must hold 13 values`);let o=n.length/4-2;if(!Number.isInteger(o)||o<1)throw Error(`Spatial error GM table must hold (predictorCount + 2) * 4 values`);let s=d(o,t.predictorNames);s.push(`lambda`);let c=s.map((e,t)=>{let r=n[t*4+2];return{name:e,role:t===0?`intercept`:t===s.length-1?`spatial-error`:`predictor`,estimate:n[t*4],standardError:n[t*4+1],statistic:r,statisticKind:`z`,pValue:Number.isNaN(r)?NaN:a(r*r,1)}}),l=i[0],u=i[6],f=i[7],p=Number.isFinite(u)&&Number.isFinite(f);return r({kind:`spatial-error-generalized-moments`,coefficients:c,rowCount:l,residualSumOfSquares:i[2]*l,standardErrorSigmaSquared:`maximum-likelihood`,statusCode:e.status[0],pseudoRSquared:i[3],logLikelihood:null,conditioning:p?{conditionNumber:u,correlationConditionNumber:f,maximumVarianceInflationFactor:i[8]}:null})}var gt=j.fields.find(e=>e.name===`significanceLevel`).wordOffset,_t=class{constructor(e){this.props=e}getCommandNodes(e){let{props:t}=this,n=(e,t)=>`let ${e}Statistic = tests[testsOffset + ${t*3}u];
  let ${e}P = tests[testsOffset + ${t*3+2}u];`;return[C(e,{id:`${t.id}-specification`,operation:`GPUSpatialRegressionSpecificationSearch`,bindings:[{name:`tests`,view:t.tests,type:`f32`,access:`read`},{name:`diagnosticsStatus`,view:t.status,type:`u32`,access:`read`},{name:`searchParameters`,view:t.parameters,type:`f32`,access:`read`},{name:`decisionOut`,view:t.decision,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }`,body:`let ALPHA = searchParameters[searchParametersOffset + ${gt}u];
  ${n(`lag`,0)}
  ${n(`error`,1)}
  ${n(`robustLag`,2)}
  ${n(`robustError`,3)}
  var decision = 4u;
  if (diagnosticsStatus[diagnosticsStatusOffset] == 0u && !isNaNValue(lagP) && !isNaNValue(errorP)) {
    let lagSignificant = lagP <= ALPHA;
    let errorSignificant = errorP <= ALPHA;
    if (!lagSignificant && !errorSignificant) {
      decision = 0u;
    } else if (lagSignificant && !errorSignificant) {
      decision = 1u;
    } else if (!lagSignificant && errorSignificant) {
      decision = 2u;
    } else if (!isNaNValue(robustLagP) && !isNaNValue(robustErrorP)) {
      let robustLagSignificant = robustLagP <= ALPHA;
      let robustErrorSignificant = robustErrorP <= ALPHA;
      if (robustLagSignificant && robustErrorSignificant) {
        decision = 3u;
      } else if (robustLagSignificant) {
        decision = 1u;
      } else if (robustErrorSignificant) {
        decision = 2u;
      } else if (lagStatistic >= errorStatistic) {
        decision = 1u;
      } else {
        decision = 2u;
      }
    }
  }
  decisionOut[decisionOutOffset] = decision;`})]}};function vt(e,t){let n=t.id??`spatial-regression`,r=new B(e),a=t.outputs??{},o=t.response.length,{predictorCount:s}=t;R(`addSpatialRegressionRecipe`,t.predictors.length===o*s,`predictors must hold rowCount * predictorCount values`),R(`addSpatialRegressionRecipe`,t.parameters.length>=8,`parameters needs 8 values`);let c=t.parameters.importToGraph(e,`${n}-parameters`),u=a.ols??{},d={coefficients:z(e,`${n}-coefficients`,`float32`,s+1,u.coefficients),standardErrors:z(e,`${n}-standard-errors`,`float32`,s+1,u.standardErrors),tStatistics:z(e,`${n}-t-statistics`,`float32`,s+1,u.tStatistics),summary:z(e,`${n}-ols-summary`,`float32`,22,u.summary),status:{code:z(e,`${n}-ols-status`,`uint32`,1,u.status?.code)},residuals:z(e,`${n}-residuals`,`float32`,o,u.residuals),fitted:z(e,`${n}-fitted`,`float32`,o,u.fitted)};r.add(new i({id:`${n}-ols`,predictors:t.predictors,response:t.response,parameters:t.olsParameters,mask:t.mask,predictorCount:s,tileRowCount:t.tileRowCount,output:d}));let f=a.diagnostics??{},p={tests:z(e,`${n}-tests`,`float32`,18,f.tests),summary:z(e,`${n}-diagnostics-summary`,`float32`,13,f.summary),status:{code:z(e,`${n}-diagnostics-status`,`uint32`,1,f.status?.code)}};r.add(new Se({id:`${n}-diagnostics`,weights:t.weights,predictors:t.predictors,response:t.response,residuals:d.residuals,mask:t.mask,restandardizeRows:t.restandardizeRows,predictorCount:s,tileRowCount:t.tileRowCount,output:p}));let m={decision:z(e,`${n}-specification-decision`,`uint32`,1,a.specification?.decision)};r.add(new _t({id:n,tests:p.tests,status:p.status.code,parameters:c,decision:m.decision}));let h=a.residualMoran??{},g={zScores:z(e,`${n}-moran-z`,`float32`,o,h.zScores),localI:z(e,`${n}-moran-i`,`float32`,o,h.localI),quadrants:z(e,`${n}-moran-quadrants`,`uint32`,o,h.quadrants),pValues:z(e,`${n}-moran-p`,`float32`,o,h.pValues)},_=r.add(new N({id:`${n}-residual-moran`,weights:t.weights,values:d.residuals,mask:t.mask,parameters:c,output:g})),v=e=>{let n=M(e),r=_.getInferenceMetadata(e);return t.parameters.write(n),r},y={contributors:r.contributors,rowCount:o,ols:d,diagnostics:p,residualMoran:g,specification:m,inference:v(t.settings??{}),setSettings:e=>{y.inference=v(e)},outputs:{ols:d,diagnostics:p,specification:m,residualMoran:g},intermediates:{},status:{stages:[{stage:`ordinary-least-squares`,status:{invalidCount:d.status.code}},{stage:`spatial-diagnostics`,status:{invalidCount:p.status.code}}]}},b=t.localFits;if(b){let i=a.localFits??{},c={coefficients:z(e,`${n}-local-coefficients`,`float32`,o*(s+1),i.coefficients),localR2:z(e,`${n}-local-r2`,`float32`,o,i.localR2),residuals:z(e,`${n}-local-residuals`,`float32`,o,i.residuals),selectedBandwidth:z(e,`${n}-selected-bandwidth`,`float32`,2,i.selectedBandwidth),localStatus:z(e,`${n}-local-status`,`uint32`,o,i.localStatus),summary:z(e,`${n}-local-summary`,`float32`,12,i.summary)};r.add(new l({id:`${n}-gwr`,positions:b.positions,spatialContext:b.spatialContext,predictors:t.predictors,predictorCount:s,response:t.response,parameters:b.parameters,maximumBandwidthCount:b.maximumBandwidthCount,maximumNeighborCount:b.maximumNeighborCount,refinement:b.refinement,output:c})),y.localFits=c,y.outputs.localFits=c}return y}var yt=[{id:`income`,label:`ln income per capita`},{id:`poverty`,label:`Poverty rate`},{id:`uninsured`,label:`Uninsured rate`},{id:`unemployed`,label:`Unemployment rate`},{id:`age65`,label:`Age 65+ share`},{id:`disability`,label:`Disability rate`},{id:`noVehicle`,label:`No-vehicle households`},{id:`black`,label:`Black share`},{id:`hispanic`,label:`Hispanic share`}],bt={income:[`income`],economic:[`income`,`poverty`,`uninsured`,`unemployed`],full:[`income`,`poverty`,`uninsured`,`age65`,`disability`,`noVehicle`,`black`,`hispanic`]},xt={diabetes:`Diabetes prevalence`,obesity:`Obesity prevalence`,asthma:`Current asthma prevalence`,depression:`Depression prevalence`,heartDisease:`Coronary heart disease prevalence`,smoking:`Current smoking prevalence`},St=32,Ct=8,wt=[64,64],Tt={knn4:4,knn8:8,knn12:12},Et=[[150,150,158,120],[214,69,65,235],[138,176,232,235],[49,104,190,235],[240,160,60,235]];async function Dt(t){let r=t.datasets.get(`chicago-tracts`),{device:i}=t,a=new v(i,`health-regression`),o=te(r),s=o.featureCount,c=e=>r.column(e),l=c(`population`),d=c(`households`),f=c(`perCapitaIncome`),m=(e,t)=>{let n=c(e);return Float32Array.from(n,(e,n)=>t[n]>0?100*e/t[n]:NaN)},h={income:Float32Array.from(f,e=>e>0?Math.log(e):NaN),poverty:c(`poverty150Pct`),uninsured:c(`uninsuredPct`),unemployed:c(`unemployedPct`),age65:c(`age65Pct`),disability:c(`disabilityPct`),noVehicle:c(`noVehiclePct`),black:m(`nhBlack`,l),hispanic:m(`hispanic`,l)},g=Object.fromEntries(Object.keys(xt).map(e=>[e,c(e)])),_=[];for(let e=0;e<s;e++){let t=l[e]>=300&&d[e]>0;for(let n of Object.values(h))t&&=Number.isFinite(n[e]);for(let n of Object.values(g))t&&=Number.isFinite(n[e]);t&&_.push(e)}let y=K(W(o),_),b=y.length,x=new Uint32Array(s).fill(le);y.forEach((e,t)=>{x[e]=t});let S=de(o,y),C=ie(S),w=r.defaultOrigin,T=new V(w),E=re(C,(e,t)=>T.project(e,t)),D=r.manifest.geoid,A={};for(let{id:e}of yt){let t=Float32Array.from(y,t=>h[e][t]),{mean:n,deviation:r}=ne(t);A[e]=Float32Array.from(t,e=>(e-n)/r)}let j=e=>Float32Array.from(y,t=>g[e][t]),M=fe(S),N=new Uint32Array(s).fill(le);for(let e=0;e<s;e++)x[e]===4294967295&&(N[e]=0);let P=fe(o,N),F=a.createBuffer(`triangles`,M.corners),R=a.createBuffer(`owners`,M.owners),z=a.createBuffer(`excluded-triangles`,P.corners),B=a.createBuffer(`excluded-owners`,P.owners),J=se(o),he=a.createBuffer(`outline`,J),ge=ae(S),_e=a.createBuffer(`positions`,E),ve=a.createBuffer(`vertices`,S.vertices),ye=a.createBuffer(`ring-offsets`,S.ringOffsets),be=a.createBuffer(`feature-offsets`,ue(S)),xe=a.createBuffer(`polygon-offsets`,S.partRingOffsets),Se=a.createBuffer(`response`,j(t.options.outcome)),Ce=b*St,we=a.createBuffer(`offsets`,(b+1)*4),Te=a.createBuffer(`neighbors`,Ce*4),Ee=a.createBuffer(`weight-values`,Ce*4),De=a.createBuffer(`overflow`,4),Oe=a.createBuffer(`total-neighbors`,4),ke=a.createBuffer(`candidate-overflow`,4),Y=a.createParameterBuffer(`search-parameters`,`float32`,12),Ae=a.createParameterBuffer(`ols-parameters`,`float32`,1),Me=a.createParameterBuffer(`moran-parameters`,`float32`,8),Ne=1/0,Pe=1/0,Fe=-1/0,Re=-1/0;for(let e=0;e<b;e++)Ne=Math.min(Ne,E[e*2]),Fe=Math.max(Fe,E[e*2]),Pe=Math.min(Pe,E[e*2+1]),Re=Math.max(Re,E[e*2+1]);let ze=[Ne-500,Pe-500,Fe+500,Re+500],Be=e=>({spatialContext:T.spatialContext,offsets:U(e,`offsets`,we,`uint32`,b+1),neighbors:U(e,`neighbors`,Te,`uint32`,Ce),weights:U(e,`weight-values`,Ee,`float32`,Ce)}),Ve=new Map,He=async e=>{let t=Ve.get(e);if(t)return t;let n=new p(i,{id:`health-weights-${e}`}),r=Be(n),o={overflow:U(n,`overflow`,De,`uint32`,1),requiredCount:U(n,`total-neighbors`,Oe,`uint32`,1)};return e===`queen`||e===`rook`?(n.add(new O({id:`contiguity`,criterion:e,polygons:{kind:`polygons`,positions:U(n,`vertices`,ve,`float32x2`,S.vertices.length/2),featureOffsets:U(n,`feature-offsets`,be,`uint32`,S.featureCount+1),polygonOffsets:U(n,`polygon-offsets`,xe,`uint32`,S.partRingOffsets.length),ringOffsets:U(n,`ring-offsets`,ye,`uint32`,S.ringOffsets.length)},output:{...r,status:{...o,candidateOverflow:U(n,`candidate-overflow`,ke,`uint32`,1)}}})),n.add(new k({spatialContext:T.spatialContext,id:`row-standardize`,operation:`row`,weights:r}))):n.add(new L({spatialContext:T.spatialContext,id:`knn`,mode:`knn`,k:Tt[e],gridSize:wt,positions:U(n,`positions`,_e,`float32x2`,b),parameters:Y.importToGraph(n),output:{...r,status:o}})),t=a.track(await n.compileAsync()),Ve.set(e,t),t};Y.write(I({bounds:ze,radius:1/0,weightKind:`binary`,rowStandardize:!0}));let Ue=new Map,We=async e=>{let n=Ue.get(e);if(n)return n;let r=bt[e],o=r.length;if(o>Ct)throw Error(`Too many predictors`);let s=new Float32Array(b*o);r.forEach((e,t)=>{for(let n=0;n<b;n++)s[n*o+t]=A[e][n]});let c=`${e}-`,l=(e,t)=>a.createBuffer(`${c}${e}`,t),u={predictors:a.createBuffer(`${c}predictors`,s),coefficients:l(`coefficients`,(o+1)*4),standardErrors:l(`standard-errors`,(o+1)*4),tStatistics:l(`t-statistics`,(o+1)*4),olsSummary:l(`ols-summary`,88),olsStatus:l(`ols-status`,4),olsResiduals:l(`ols-residuals`,b*4),fitted:l(`fitted`,b*4),tests:l(`tests`,72),diagnosticsSummary:l(`diagnostics-summary`,52),diagnosticsStatus:l(`diagnostics-status`,4),zScores:l(`moran-z`,b*4),localI:l(`moran-i`,b*4),quadrants:l(`quadrants`,b*4),pValues:l(`moran-p`,b*4),lagTable:l(`lag-table`,(o+2)*4*4),lagSummary:l(`lag-summary`,64),lagStatus:l(`lag-status`,4),lagResiduals:l(`lag-residuals`,b*4),errorTable:l(`error-table`,(o+2)*4*4),errorSummary:l(`error-summary`,52),errorStatus:l(`error-status`,4),errorResiduals:l(`error-residuals`,b*4)},d=new p(i,{id:`health-recipe-${e}`}),f=(e,t,n,r,i)=>U(e,t,n,r,i),m=e=>f(e,`predictors`,u.predictors,`float32`,b*o),h=e=>f(e,`response`,Se,`float32`,b),g=vt(d,{id:`health`,predictors:m(d),response:h(d),predictorCount:o,weights:Be(d),settings:{significanceLevel:t.options.significance},parameters:Me,olsParameters:Ae.importToGraph(d),outputs:{ols:{coefficients:f(d,`coefficients`,u.coefficients,`float32`,o+1),standardErrors:f(d,`standard-errors`,u.standardErrors,`float32`,o+1),tStatistics:f(d,`t-statistics`,u.tStatistics,`float32`,o+1),summary:f(d,`ols-summary`,u.olsSummary,`float32`,22),status:{code:f(d,`ols-status`,u.olsStatus,`uint32`,1)},residuals:f(d,`ols-residuals`,u.olsResiduals,`float32`,b),fitted:f(d,`fitted`,u.fitted,`float32`,b)},diagnostics:{tests:f(d,`tests`,u.tests,`float32`,18),summary:f(d,`diagnostics-summary`,u.diagnosticsSummary,`float32`,13),status:{code:f(d,`diagnostics-status`,u.diagnosticsStatus,`uint32`,1)}},residualMoran:{zScores:f(d,`moran-z`,u.zScores,`float32`,b),localI:f(d,`moran-i`,u.localI,`float32`,b),quadrants:f(d,`quadrants`,u.quadrants,`uint32`,b),pValues:f(d,`moran-p`,u.pValues,`float32`,b)}}}),_=a.track(await d.compileAsync()),v=new p(i,{id:`health-error-${e}`});v.add(new st({id:`error-gm`,weights:Be(v),predictors:m(v),response:h(v),predictorCount:o,output:{table:f(v,`error-table`,u.errorTable,`float32`,(o+2)*4),summary:f(v,`error-summary`,u.errorSummary,`float32`,13),status:{code:f(v,`error-status`,u.errorStatus,`uint32`,1)},residuals:f(v,`error-residuals`,u.errorResiduals,`float32`,b)}}));let y=a.track(await v.compileAsync()),x=new Map,S=async t=>{let n=x.get(t);if(n)return n;let r=new p(i,{id:`health-lag-${e}-${t}`});return r.add(new je({id:`two-stage`,weights:Be(r),predictors:m(r),response:h(r),predictorCount:o,instrumentOrder:t===`2`?2:1,output:{table:f(r,`lag-table`,u.lagTable,`float32`,(o+2)*4),summary:f(r,`lag-summary`,u.lagSummary,`float32`,16),status:{code:f(r,`lag-status`,u.lagStatus,`uint32`,1)},residuals:f(r,`lag-residuals`,u.lagResiduals,`float32`,b)}})),n=a.track(await r.compileAsync()),x.set(t,n),n},C=pe(a,`health-${e}`,[{name:`coefficients`,buffer:u.coefficients,bytes:(o+1)*4},{name:`standardErrors`,buffer:u.standardErrors,bytes:(o+1)*4},{name:`tStatistics`,buffer:u.tStatistics,bytes:(o+1)*4},{name:`olsSummary`,buffer:u.olsSummary,bytes:88},{name:`olsStatus`,buffer:u.olsStatus,bytes:4},{name:`tests`,buffer:u.tests,bytes:72},{name:`diagnosticsSummary`,buffer:u.diagnosticsSummary,bytes:52},{name:`diagnosticsStatus`,buffer:u.diagnosticsStatus,bytes:4},{name:`lagTable`,buffer:u.lagTable,bytes:(o+2)*4*4},{name:`lagSummary`,buffer:u.lagSummary,bytes:64},{name:`lagStatus`,buffer:u.lagStatus,bytes:4},{name:`errorTable`,buffer:u.errorTable,bytes:(o+2)*4*4},{name:`errorSummary`,buffer:u.errorSummary,bytes:52},{name:`errorStatus`,buffer:u.errorStatus,bytes:4},{name:`overflow`,buffer:De,bytes:4},{name:`totalNeighbors`,buffer:Oe,bytes:4},{name:`candidateOverflow`,buffer:ke,bytes:4},{name:`olsResiduals`,buffer:u.olsResiduals,bytes:b*4},{name:`fitted`,buffer:u.fitted,bytes:b*4},{name:`quadrants`,buffer:u.quadrants,bytes:b*4},{name:`lagResiduals`,buffer:u.lagResiduals,bytes:b*4},{name:`errorResiduals`,buffer:u.errorResiduals,bytes:b*4}],t=>Ye(e,o,t)),w={predictorCount:o,names:[`Intercept`,...r.map(e=>yt.find(t=>t.id===e).label)],recipe:_,setSettings:g.setSettings,lag:x,error:y,buffers:{predictors:u.predictors,olsResiduals:u.olsResiduals,fitted:u.fitted,quadrants:u.quadrants,lagResiduals:u.lagResiduals,errorResiduals:u.errorResiduals},addLag:S,reader:C};return Ue.set(e,w),w},Ge=1,X=null,Ke=[0,1],qe=-1,Je=()=>{let e=j(t.options.outcome);Ke=[oe(e,.02),oe(e,.98)],t.setLegendExtent(`value`,Ke)},Ye=(e,r,i)=>{if(e!==t.options.model)return;let a=Ue.get(e),o=i(`olsSummary`).f32,c=i(`coefficients`).f32,l=i(`standardErrors`).f32,u=i(`tStatistics`).f32,d=i(`tests`).f32,f=i(`diagnosticsSummary`).f32,p=i(`lagTable`).f32,m=i(`lagSummary`).f32,h=i(`errorTable`).f32,g=i(`errorSummary`).f32,_=[i(`olsStatus`).u32[0],i(`diagnosticsStatus`).u32[0],i(`lagStatus`).u32[0],i(`errorStatus`).u32[0]];Ge=Math.sqrt(Math.max(0,o[3])),t.setLegendExtent(`residual`,[-2*Ge,2*Ge]),X={olsResiduals:i(`olsResiduals`).f32,fitted:i(`fitted`).f32,lagResiduals:i(`lagResiduals`).f32,errorResiduals:i(`errorResiduals`).f32,quadrants:i(`quadrants`).u32},t.setReadout(`rows`,`${ce(b)} of ${ce(s)} tracts`);let v=i(`overflow`).u32[0],y=i(`candidateOverflow`).u32[0],x=Math.min(i(`totalNeighbors`).u32[0],Ce);t.setReadout(`links`,`${ce(x)} (mean ${(x/b).toFixed(1)} per tract)${v?`, OVERFLOW`:``}${y?` (pair list full: lower bound)`:``}`),t.setReadout(`fit`,`R² ${G(o[1])}, adjusted ${G(o[2])}`),t.setReadout(`criteria`,`AIC ${G(o[5],1)}, BIC ${G(o[6],1)}`),t.setReadout(`normality`,q(o[8])),t.setReadout(`heteroskedasticity`,q(o[10]));let S=e=>({statistic:d[e*3],p:d[e*3+2]}),C=S(5);t.setReadout(`moran`,`I ${G(f[5])} (expected ${G(f[6],4)}), z ${G(C.statistic,1)}, ${q(C.p)}`);let w=S(0),T=S(1),E=S(2),D=S(3),O=S(4),k=e=>`${G(e.statistic,1)}, ${q(e.p)}`;t.setReadout(`lmLag`,k(w)),t.setReadout(`lmError`,k(T)),t.setReadout(`robustLag`,k(E)),t.setReadout(`robustError`,k(D)),t.setReadout(`sarma`,k(O));let A=.05,j;j=!(w.p<=A)&&!(T.p<=A)?`Neither LM test is significant: OLS is adequate.`:w.p<=A&&!(T.p<=A)?`Only LM-lag is significant: fit the spatial lag model.`:!(w.p<=A)&&T.p<=A?`Only LM-error is significant: fit the spatial error model.`:E.p<D.p?`Both significant; the robust lag test is stronger: spatial lag model.`:`Both significant; the robust error test is stronger: spatial error model.`,t.setReadout(`verdict`,j);let M=a.names.slice(1,r+1),N=n({coefficients:c.subarray(0,r+1),standardErrors:l.subarray(0,r+1),tStatistics:u.subarray(0,r+1),summary:o,status:i(`olsStatus`).u32},{predictorNames:M}),P=Le({table:p.subarray(0,(r+2)*4),summary:m,status:i(`lagStatus`).u32},{predictorNames:M}),F=ht({table:h.subarray(0,(r+2)*4),summary:g,status:i(`errorStatus`).u32},{predictorNames:M}),I=e=>`${G(e.estimate,2)} (se ${G(e.standardError,2)}, ${q(e.pValue)})`;for(let e=0;e<=Ct;e++){if(e>r){t.setReadout(`coefficient${e}`,null);continue}t.setReadout(`coefficient${e}`,`${a.names[e]}: OLS ${I(N.coefficients[e])}; lag ${I(P.coefficients[e])}; error ${I(F.coefficients[e])}`)}let L=P.coefficients[r+1];t.setReadout(`rho`,_[2]===0?`ρ ${G(L.estimate)} ± ${G(L.standardError)} (z ${G(L.statistic,1)}, ${q(L.pValue)})`:`fit failed (status ${_[2]})`);let R=Ie(m);t.setReadout(`lagFit`,`pseudo R² ${G(P.pseudoRSquared)}; Anselin-Kelejian ${G(P.tests.anselinKelejian?.statistic??NaN,1)}, ${q(P.tests.anselinKelejian?.pValue??NaN)}; first-stage F ${G(R.firstStageF,1)}${R.rhoOutsideAdmissibleRange?`; ρ outside the admissible range`:``}`);let z=F.coefficients[r+1];t.setReadout(`lambda`,_[3]===0?`λ ${G(z.estimate)} ± ${G(z.standardError)} (z ${G(z.statistic,1)}, ${q(z.pValue)}); pseudo R² ${G(F.pseudoRSquared)}`:`fit failed (status ${_[3]})`);let B=(e,t)=>t.conditioning?`${e} ${G(t.conditioning.conditionNumber,1)}${t.conditioning.illConditioned?` (float32 precision warning)`:``}`:`${e} n/a`;t.setReadout(`conditioning`,`${B(`OLS`,N)}; ${B(`lag`,P)}; ${B(`error`,F)}; largest VIF ${G(N.conditioning?.maximumVarianceInflationFactor??NaN,1)}`),t.setReadout(`status`,_.join(` / `));let V=[0,0,0,0,0];for(let e of X.quadrants)e<5&&V[e]++;t.setReadout(`clusters`,`${V[1]} high-high / ${V[3]} low-low / ${V[2]} low-high / ${V[4]} high-low`),t.requestLayers()},Xe=!0,Ze=!0,Qe=t.options.weights,$e=t.options.model,et=t.options.instruments,tt=()=>{let e=t.options.ridge;Ae.write(u({ridgeLambda:e<=-2?0:10**e}))},nt=()=>{t.setReadout(`localInference`,`α = ${t.options.significance}`),Ue.get($e)?.setSettings({significanceLevel:t.options.significance})};tt(),nt(),Je();let rt=()=>Ue.get($e),Z=async()=>{let{weights:e,model:n,instruments:r}=t.options;return await He(e),await(await We(n)).addLag(r),{weights:e,model:n,order:r}},Q=e=>{e.weights!==Qe&&(Xe=!0),Qe=e.weights,$e=e.model,et=e.order,Ze=!0,t.requestLayers()},$=a.track(new ee);Q(await Z()),t.setStatus(`${ce(b)} Chicago tracts`);let it=()=>t.theme()===`dark`;return{getCompiledGraphs:()=>{let e=rt();return[Ve.get(Qe),e.recipe,e.lag.get(et),e.error]},setOption(e){if(e===`map`){t.requestLayers();return}e===`ridge`&&tt(),e===`significance`&&nt(),e===`outcome`&&(Se.write(j(t.options.outcome)),Je()),(e===`weights`||e===`model`||e===`instruments`)&&$.request({build:Z,commit:Q}),Ze=!0,t.requestLayers()},encode(e,t){t.frameIndex<2&&(Xe=!0,Ze=!0);let n=rt();Xe&&(Ve.get(Qe).encode(e,{parameters:void 0}),Xe=!1,Ze=!0),Ze?(n.recipe.encode(e,{parameters:void 0}),n.lag.get(et).encode(e,{parameters:void 0}),n.error.encode(e,{parameters:void 0}),n.reader.request(e),Ze=!1):n.reader.flush(e)},getLayers(){let n=rt(),r=t.options.map,i=[],a=it();i.push(new me({id:`health-excluded`,triangles:z,owners:B,triangleCount:P.triangleCount,colormap:`uniform`,color:a?[90,94,104,150]:[170,172,178,150]}));let o={triangles:F,owners:R,triangleCount:M.triangleCount,opacity:.86,selectedRow:qe};return r===`clusters`?i.push(new me({...o,id:`health-clusters`,values:n.buffers.quadrants,valueFormat:`uint32`,colormap:`category`,palette:Et,noDataValue:le})):r===`fitted`||r===`observed`?i.push(new me({...o,id:`health-${r}`,values:r===`fitted`?n.buffers.fitted:Se,valueFormat:`float32`,colormap:`ylorrd`,valueRange:Ke,color:[255,255,255,255],noDataColor:[0,0,0,0]})):i.push(new me({...o,id:`health-${r}`,values:r===`residuals`?n.buffers.olsResiduals:r===`lag`?n.buffers.lagResiduals:n.buffers.errorResiduals,valueFormat:`float32`,colormap:`diverging`,valueRange:[-2*Ge,2*Ge],color:[255,255,255,255]})),i.push(new H({id:`health-outline`,coordinateSystem:e.LNGLAT,segments:he,instanceCount:J.length/4,widthPixels:.8,color:a?[225,228,238,70]:[40,44,56,90]})),i},onThemeChange(){t.requestLayers()},getTooltip(e){if(!e.coordinate)return null;let n=ge(e.coordinate[0],e.coordinate[1]);if(n<0||!X)return null;let r=y[n],i=j(t.options.outcome)[n],a=[`Tract ${D[r]}`,`${xt[t.options.outcome]}: ${i.toFixed(1)}%`,`Fitted (OLS): ${X.fitted[n].toFixed(1)}%`,`Residual: ${X.olsResiduals[n]>=0?`+`:``}${X.olsResiduals[n].toFixed(1)} points`,`Income per capita: $${ce(f[r])}`,`Poverty: ${h.poverty[r].toFixed(0)}%`];return t.options.map===`lag`&&a.push(`Lag-model residual: ${X.lagResiduals[n].toFixed(1)}`),t.options.map===`error`&&a.push(`Error-model residual: ${X.errorResiduals[n].toFixed(1)}`),a.join(`
`)},onClick(e){if(!e.coordinate)return!1;let n=ge(e.coordinate[0],e.coordinate[1]);return qe=n===qe?-1:n,t.requestLayers(),!0},destroy(){for(let e of Ue.values())e.reader.stop();a.destroy()}}}export{Dt as createHealthRegression};