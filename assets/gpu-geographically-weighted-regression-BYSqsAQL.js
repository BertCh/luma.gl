import{D as e,E as t,p as n,w as r}from"./analysis-frame-Blu0tYdw.js";import{_ as i,d as a,h as o,w as s,y as c}from"./resources-CwvhEOBe.js";import{t as l}from"./gpu-grid-index-D1JNO067.js";import{l as u,o as d,s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as p,r as m,s as h}from"./wgsl-common-BNRVzgd8.js";import{t as g}from"./geodesic-wgsl-DS0CJZ42.js";var _=1e4;function v(e){let t=e.parameterCount??e.coefficients.length,n=e.rowCount,r=n-t,i=e.residualSumOfSquares,a=e.logLikelihood??null,o=e.informationCriterionParameterCount??t,s=e.conditioning?{...e.conditioning,illConditioned:e.conditioning.correlationConditionNumber>_}:null,c=e.statusCode===0;return{kind:e.kind,coefficients:e.coefficients,rowCount:n,parameterCount:t,degreesOfFreedom:r,sigmaSquared:r>0?i/r:NaN,sigmaSquaredMaximumLikelihood:n>0?i/n:NaN,standardErrorSigmaSquared:e.standardErrorSigmaSquared,residualSumOfSquares:i,rSquared:e.rSquared??NaN,pseudoRSquared:e.pseudoRSquared??NaN,logLikelihood:a,akaikeInformationCriterion:a===null?null:2*o-2*a,schwarzCriterion:a===null?null:o*Math.log(n)-2*a,correctedAkaikeInformationCriterion:e.correctedAkaikeInformationCriterion===void 0?a===null?null:2*o-2*a+2*o*(o+1)/(n-o-1):e.correctedAkaikeInformationCriterion,conditioning:s,tests:e.tests??{},status:{code:e.statusCode,ok:c,converged:e.converged??null,illConditioned:s?.illConditioned??!1},inference:{alternative:`two-sided`,permutationCount:0,multipleTesting:`none`,testedRowsDefinition:`model coefficients`}}}function y(e,t){let n=[`CONSTANT`];for(let r=0;r<e;r++)n.push(t?.[r]??`x${r}`);return n}var b=.001;function x(e={},t=new Float32Array(1)){if(t.length<1)throw Error(`getGPUOrdinaryLeastSquaresParameterValues target must hold 1 element`);return t[0]=e.ridgeLambda??0,t}function S(e){if(!Number.isInteger(e)||e<1)throw Error(`Chi-square degrees of freedom must be a positive integer`);let t=e/2,n=e%2==0?1:.5,r=e%2==0?0:.5*Math.log(Math.PI);for(;n<t;)r+=Math.log(n),n+=1;return r}function ee(e,t){let n=S(t);if(Number.isNaN(e))return NaN;if(!(e>0))return 1;let r=t/2,i=e/2,a=-i+r*Math.log(i)-n;if(i<r+1){let e=1/r,t=e;for(let n=1;n<=200&&(e*=i/(r+n),t+=e,!(e<t*1e-15));n++);return Math.min(1,Math.max(0,1-t*Math.exp(a)))}let o=1e-300,s=i+1-r,c=1/o,l=1/s,u=l;for(let e=1;e<=200;e++){let t=-e*(e-r);s+=2,l=t*l+s,Math.abs(l)<o&&(l=o),c=s+t/c,Math.abs(c)<o&&(c=o),l=1/l;let n=l*c;if(u*=n,Math.abs(n-1)<1e-15)break}return Math.min(1,Math.max(0,Math.exp(a)*u))}var te=400;function C(e,t,n){let r=1e-300,i=t+n,a=t+1,o=t-1,s=1,c=1-i*e/a;Math.abs(c)<r&&(c=r),c=1/c;let l=c;for(let u=1;u<=te;u++){let d=2*u,f=u*(n-u)*e/((o+d)*(t+d));c=1+f*c,Math.abs(c)<r&&(c=r),s=1+f/s,Math.abs(s)<r&&(s=r),c=1/c,l*=c*s,f=-(t+u)*(i+u)*e/((t+d)*(a+d)),c=1+f*c,Math.abs(c)<r&&(c=r),s=1+f/s,Math.abs(s)<r&&(s=r),c=1/c;let p=c*s;if(l*=p,Math.abs(p-1)<1e-16)break}return l}function w(e,t){if(Number.isNaN(e)||!Number.isInteger(t)||t<1)return NaN;if(!Number.isFinite(e))return 0;let n=t/2,r=.5,i=t/(t+e*e);if(i>=1)return 1;let a=S(t+1)-S(t)-S(1)+n*Math.log(i)+r*Math.log1p(-i),o=Math.exp(a);return i<(n+1)/(n+r+2)?Math.min(1,o*C(i,n,r)/n):Math.max(0,1-o*C(1-i,r,n)/r)}function T(e,t={}){let{summary:n}=e;if(n.length<22)throw Error(`Ordinary least squares summary must hold 22 values`);let r=e.coefficients.length,i=r-1,a=n[0],o=a-r,s=y(i,t.predictorNames).map((t,n)=>({name:t,role:n===0?`intercept`:`predictor`,estimate:e.coefficients[n],standardError:e.standardErrors[n],statistic:e.tStatistics[n],statisticKind:`t`,pValue:w(e.tStatistics[n],o)})),c=n[18],l=n[19],u=Number.isFinite(c)&&Number.isFinite(l),d=(e,t,r)=>({statistic:n[e],degreesOfFreedom:r,pValue:n[t]});return v({kind:`ordinary-least-squares`,coefficients:s,rowCount:a,residualSumOfSquares:n[11],standardErrorSigmaSquared:`degrees-of-freedom`,statusCode:e.status[0],rSquared:n[1],logLikelihood:Number.isNaN(n[4])?null:n[4],conditioning:u?{conditionNumber:c,correlationConditionNumber:l,maximumVarianceInflationFactor:n[20]}:null,tests:{jarqueBera:d(7,8,2),breuschPagan:d(16,17,i),koenkerBassett:d(9,10,i)}})}function E(e){k(e);let t=e,n=t*t;return`
fn choleskyFactor_${t}(a: ptr<function, array<f32, ${n}>>) -> bool {
  for (var column = 0u; column < ${t}u; column++) {
    var diagonal = (*a)[column * ${t}u + column];
    for (var k = 0u; k < column; k++) {
      let value = (*a)[column * ${t}u + k];
      diagonal = diagonal - value * value;
    }
    if (!(diagonal > 0.0) || diagonal > 3.0e38) {
      return false;
    }
    let pivot = sqrt(diagonal);
    (*a)[column * ${t}u + column] = pivot;
    for (var row = column + 1u; row < ${t}u; row++) {
      var value = (*a)[row * ${t}u + column];
      for (var k = 0u; k < column; k++) {
        value = value - (*a)[row * ${t}u + k] * (*a)[column * ${t}u + k];
      }
      (*a)[row * ${t}u + column] = value / pivot;
    }
  }
  return true;
}

fn choleskySolve_${t}(a: ptr<function, array<f32, ${n}>>, b: ptr<function, array<f32, ${t}>>) {
  for (var row = 0u; row < ${t}u; row++) {
    var value = (*b)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*a)[row * ${t}u + k] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${t}u + row];
  }
  for (var offset = 0u; offset < ${t}u; offset++) {
    let row = ${t-1}u - offset;
    var value = (*b)[row];
    for (var k = row + 1u; k < ${t}u; k++) {
      value = value - (*a)[k * ${t}u + row] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${t}u + row];
  }
}

fn choleskyInverseDiagonal_${t}(a: ptr<function, array<f32, ${n}>>, column: u32) -> f32 {
  var unit: array<f32, ${t}>;
  for (var row = 0u; row < ${t}u; row++) {
    unit[row] = select(0.0, 1.0, row == column);
  }
  choleskySolve_${t}(a, &unit);
  return unit[column];
}
`}function D(e){k(e);let t=e;return`
fn getSymmetricEigenvalueRange_${t}(a: ptr<function, array<f32, ${t*t}>>) -> vec2<f32> {
  for (var sweep = 0u; sweep < 16u; sweep++) {
    var offDiagonal = 0.0;
    var diagonal = 0.0;
    for (var row = 0u; row < ${t}u; row++) {
      diagonal = diagonal + (*a)[row * ${t}u + row] * (*a)[row * ${t}u + row];
      for (var column = row + 1u; column < ${t}u; column++) {
        offDiagonal = offDiagonal + (*a)[row * ${t}u + column] * (*a)[row * ${t}u + column];
      }
    }
    if (offDiagonal <= 1.0e-14 * diagonal) {
      break;
    }
    for (var first = 0u; first + 1u < ${t}u; first++) {
      for (var second = first + 1u; second < ${t}u; second++) {
        let pivot = (*a)[first * ${t}u + second];
        if (abs(pivot) < 1.0e-30) {
          continue;
        }
        let theta = ((*a)[second * ${t}u + second] - (*a)[first * ${t}u + first]) / (2.0 * pivot);
        let tangent = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < ${t}u; k++) {
          let left = (*a)[k * ${t}u + first];
          let right = (*a)[k * ${t}u + second];
          (*a)[k * ${t}u + first] = cosine * left - sine * right;
          (*a)[k * ${t}u + second] = sine * left + cosine * right;
        }
        for (var k = 0u; k < ${t}u; k++) {
          let top = (*a)[first * ${t}u + k];
          let bottom = (*a)[second * ${t}u + k];
          (*a)[first * ${t}u + k] = cosine * top - sine * bottom;
          (*a)[second * ${t}u + k] = sine * top + cosine * bottom;
        }
      }
    }
  }
  var smallest = (*a)[0];
  var largest = (*a)[0];
  for (var row = 1u; row < ${t}u; row++) {
    smallest = min(smallest, (*a)[row * ${t}u + row]);
    largest = max(largest, (*a)[row * ${t}u + row]);
  }
  return vec2<f32>(smallest, largest);
}
`}function O(e){k(e+1);let t=e,n=t+1;return`
fn getBelsleyConditionNumber_${t}(
  factor: ptr<function, array<f32, ${t*t}>>,
  alpha: ptr<function, array<f32, ${t}>>,
  beta: ptr<function, array<f32, ${t}>>
) -> f32 {
  // Column-major B: b[column * ${n}u + row].
  var b: array<f32, ${n*n}>;
  b[0] = 1.0;
  for (var column = 1u; column < ${n}u; column++) {
    let j = column - 1u;
    b[column * ${n}u] = (*alpha)[j];
    for (var i = 0u; i <= j; i++) {
      b[column * ${n}u + 1u + i] = (*beta)[j] * (*factor)[j * ${t}u + i];
    }
  }
  for (var sweep = 0u; sweep < 16u; sweep++) {
    var rotated = false;
    for (var first = 0u; first + 1u < ${n}u; first++) {
      for (var second = first + 1u; second < ${n}u; second++) {
        var normFirst = 0.0;
        var normSecond = 0.0;
        var product = 0.0;
        for (var row = 0u; row < ${n}u; row++) {
          let x = b[first * ${n}u + row];
          let y = b[second * ${n}u + row];
          normFirst = normFirst + x * x;
          normSecond = normSecond + y * y;
          product = product + x * y;
        }
        if (!(abs(product) > 1.0e-7 * sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        let zeta = (normSecond - normFirst) / (2.0 * product);
        let tangent = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
        let cosine = 1.0 / sqrt(1.0 + tangent * tangent);
        let sine = cosine * tangent;
        for (var row = 0u; row < ${n}u; row++) {
          let x = b[first * ${n}u + row];
          let y = b[second * ${n}u + row];
          b[first * ${n}u + row] = cosine * x - sine * y;
          b[second * ${n}u + row] = sine * x + cosine * y;
        }
      }
    }
    if (!rotated) {
      break;
    }
  }
  var largest = 0.0;
  var smallest = 3.0e38;
  for (var column = 0u; column < ${n}u; column++) {
    var norm = 0.0;
    for (var row = 0u; row < ${n}u; row++) {
      norm = norm + b[column * ${n}u + row] * b[column * ${n}u + row];
    }
    largest = max(largest, sqrt(norm));
    smallest = min(smallest, sqrt(norm));
  }
  return largest / smallest;
}
`}function k(e){if(!Number.isInteger(e)||e<1||e>16)throw Error(`Spatial regression predictor count must be an integer in [1, 16]`)}var A=`GPUOrdinaryLeastSquares`,j=64,M=4096,N=64,ne=class{id;props;rowCount;tileRowCount;tileCount;constructor(e){this.id=e.id??`ordinary-least-squares`,this.props=e;let t=this.id,{predictorCount:n,output:r}=e;a(t,`predictorCount`,n,1,15),c(e.predictors,[`float32`],`${t} predictors`),c(e.response,[`float32`],`${t} response`);let i=e.response.length;if(i<1)throw Error(`${t} needs at least one row`);if(e.predictors.length!==i*n)throw Error(`${t} predictors length must equal response length * predictorCount`);if(e.mask&&(c(e.mask,[`uint32`],`${t} mask`),e.mask.length!==i))throw Error(`${t} mask length must equal response length`);if(e.parameters&&(c(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<1))throw Error(`${t} parameters must hold 1 float32 value`);let s=n+1;for(let e of[`coefficients`,`standardErrors`,`tStatistics`])if(c(r[e],[`float32`],`${t} output.${e}`),r[e].length<s)throw Error(`${t} output.${e} must hold predictorCount + 1 rows`);if(c(r.summary,[`float32`],`${t} output.summary`),r.summary.length<22)throw Error(`${t} output.summary must hold 22 float32 values`);if(c(r.status.code,[`uint32`],`${t} output.status.code`),r.status.code.length<1)throw Error(`${t} output.status.code must hold one uint32 row`);for(let e of[`residuals`,`fitted`]){let n=r[e];if(n&&(c(n,[`float32`],`${t} output.${e}`),n.length<i))throw Error(`${t} output.${e} must hold one row per input row`)}e.tileRowCount!==void 0&&a(t,`tileRowCount`,e.tileRowCount),this.rowCount=i,this.tileRowCount=e.tileRowCount??Math.max(N,Math.ceil(i/M)),this.tileCount=Math.ceil(i/this.tileRowCount),o(t,[r.coefficients,r.standardErrors,r.tStatistics,r.summary,r.status.code,r.residuals,r.fitted],[e.predictors,e.response,e.mask,e.parameters])}getCommandNodes(e){let{props:t,id:n,rowCount:r,tileRowCount:a,tileCount:o}=this,{output:c,predictorCount:l}=t;i(n,e,[t.predictors,t.response,t.mask,t.parameters,c.coefficients,c.standardErrors,c.tStatistics,c.summary,c.status.code,c.residuals,c.fitted]);let u=l,d=F(u),f=(t,r)=>s(e,`${n}-${t}`,`uint32`,r),p=(t,r)=>s(e,`${n}-${t}`,`float32`,r),m=c.residuals??p(`residuals`,r),h=p(`squared-residuals`,r),g=[],_={id:n,operation:A,predictors:t.predictors,mask:t.mask,rowCount:r,k:u,tileRowCount:a,tileCount:o},v=f(`workspace`,d.length);g.push(...re(e,{..._,prefix:n,response:t.response,workspace:v,partialMeans:f(`partial-means`,o*(u+2)),partialMoments:f(`partial-moments`,o*d.momentCount)})),g.push(G(e,{id:`${n}-solve`,variant:`solve`,bindings:[...t.parameters?[V(`params`,t.parameters,`f32`)]:[],W(v)],invocationCount:1,declarations:`${L(d,!0)}
${E(u)}`,body:ie(u,!!t.parameters)})),g.push(G(e,{id:`${n}-residuals`,variant:`residuals`,bindings:[V(`design`,t.predictors,`f32`),V(`response`,t.response,`f32`),...t.mask?[V(`rowMask`,t.mask,`u32`)]:[],U(v),H(`residualOut`,m,`f32`),...c.fitted?[H(`fittedOut`,c.fitted,`f32`)]:[],H(`squaredOut`,h,`f32`)],invocationCount:r,declarations:`${L(d,!1)}
${R(r,u,a,!!t.mask)}`,body:`let nan = getNaN();
  residualOut[residualOutOffset + index] = nan;
  ${c.fitted?`fittedOut[fittedOutOffset + index] = nan;`:``}
  squaredOut[squaredOutOffset + index] = nan;
  if (ws[wsOffset + W_STATUS] != 0u || !isRowValid(index)) {
    return;
  }
  var fit = 0.0;
  for (var j = 0u; j < K; j++) {
    fit = fit + wsGet(W_SLOPES + j) * (design[designOffset + index * K + j] - wsGet(W_MEANS + j));
  }
  let meanResponse = wsGet(W_MEANS + K);
  let residual = (response[responseOffset + index] - meanResponse) - fit;
  residualOut[residualOutOffset + index] = residual;
  ${c.fitted?`fittedOut[fittedOutOffset + index] = meanResponse + fit;`:``}
  squaredOut[squaredOutOffset + index] = residual * residual;`}));let y=f(`partial-residuals`,o*P);g.push(G(e,{id:`${n}-residual-tiles`,variant:`residual-tiles`,workgroupSize:j,bindings:[V(`residualIn`,m,`f32`),H(`partial`,y,`u32`)],invocationCount:o,declarations:`const ROW_COUNT: u32 = ${r}u;
const TILE_ROWS: u32 = ${a}u;
${I}`,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sumSquares = 0.0;
  var sumCubes = 0.0;
  var sumFourth = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e = residualIn[residualInOffset + row];
    if (!isFiniteValue(e)) {
      continue;
    }
    let e2 = e * e;
    sumSquares = sumSquares + e2;
    sumCubes = sumCubes + e2 * e;
    sumFourth = sumFourth + e2 * e2;
  }
  let base = partialOffset + index * ${P}u;
  partial[base] = bitcast<u32>(sumSquares);
  partial[base + 1u] = bitcast<u32>(sumCubes);
  partial[base + 2u] = bitcast<u32>(sumFourth);`})),g.push(G(e,{id:`${n}-residual-merge`,variant:`residual-merge`,bindings:[V(`partial`,y,`u32`),W(v)],invocationCount:P,declarations:`${L(d,!0)}
const TILE_COUNT: u32 = ${o}u;`,body:`var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = bitcast<f32>(partial[partialOffset + tile * ${P}u + index]) - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }
  wsSet(W_RESIDUAL_SUMS + index, sum);`})),g.push(G(e,{id:`${n}-finish`,variant:`finish`,bindings:[U(v),H(`coefficientsOut`,c.coefficients,`f32`),H(`standardErrorsOut`,c.standardErrors,`f32`),H(`tStatisticsOut`,c.tStatistics,`f32`),H(`summaryOut`,c.summary,`f32`),H(`statusOut`,c.status.code,`u32`)],invocationCount:1,declarations:`${L(d,!1)}
${E(u)}
${D(u)}
${O(u)}`,body:z(u)}));let b=f(`bp-workspace`,d.length);return g.push(...re(e,{..._,prefix:`${n}-bp`,response:h,workspace:b,partialMeans:f(`bp-partial-means`,o*(u+2)),partialMoments:f(`bp-partial-moments`,o*d.momentCount)})),g.push(G(e,{id:`${n}-bp-solve`,variant:`solve`,bindings:[W(b)],invocationCount:1,declarations:`${L(d,!0)}
${E(u)}`,body:ie(u,!1)})),g.push(G(e,{id:`${n}-bp-finish`,variant:`breusch-pagan`,bindings:[U(b),V(`mainWs`,v,`u32`),H(`summaryOut`,c.summary,`f32`)],invocationCount:1,declarations:`${L(d,!1)}
${B(u)}`,body:ae(u)})),g}},P=3;function F(e){let t=e+1,n=t*(t+1)/2,r=0+t,i=r+1,a=i+1,o=a+1,s=o+P,c=s+e,l=c+e,u=l+e*e;return{momentCount:n,means:0,count:r,status:i,lambda:a,residualSums:o,scale:s,slopes:c,factor:l,moments:u,length:u+n}}var I=`
${m}

${p}
`;function L(e,t){let n=e.slopes-e.scale;return`
const K: u32 = ${n}u;
const D: u32 = ${n+1}u;
const KK: u32 = ${n*n}u;
const W_MEANS: u32 = ${e.means}u;
const W_COUNT: u32 = ${e.count}u;
const W_STATUS: u32 = ${e.status}u;
const W_LAMBDA: u32 = ${e.lambda}u;
const W_RESIDUAL_SUMS: u32 = ${e.residualSums}u;
const W_SCALE: u32 = ${e.scale}u;
const W_SLOPES: u32 = ${e.slopes}u;
const W_FACTOR: u32 = ${e.factor}u;
const W_MOMENTS: u32 = ${e.moments}u;
${I}
fn wsGet(slot: u32) -> f32 {
  return bitcast<f32>(ws[wsOffset + slot]);
}
${t?`fn wsSet(slot: u32, value: f32) {
  ws[wsOffset + slot] = bitcast<u32>(value);
}`:``}
fn momentAt(first: u32, second: u32) -> f32 {
  let lo = min(first, second);
  let hi = max(first, second);
  return wsGet(W_MOMENTS + lo * (2u * D - lo + 1u) / 2u + (hi - lo));
}
`}function R(e,t,n,r){return`
const ROW_COUNT: u32 = ${e}u;
const TILE_ROWS: u32 = ${n}u;
const ROW_K: u32 = ${t}u;

fn isRowValid(row: u32) -> bool {
  ${r?`if (rowMask[rowMaskOffset + row] == 0u) {
    return false;
  }`:``}
  if (!isFiniteValue(response[responseOffset + row])) {
    return false;
  }
  for (var j = 0u; j < ROW_K; j++) {
    if (!isFiniteValue(design[designOffset + row * ROW_K + j])) {
      return false;
    }
  }
  return true;
}
`}function re(e,t){let{prefix:n,k:r,rowCount:i,tileRowCount:a,tileCount:o,workspace:s}=t,c=r+1,l=F(r),u=l.momentCount,d=!!t.mask,f=[V(`design`,t.predictors,`f32`),V(`response`,t.response,`f32`),...t.mask?[V(`rowMask`,t.mask,`u32`)]:[]],p=`${L(l,!1)}
${R(i,r,a,d)}`,m=`${L(l,!0)}
const TILE_COUNT: u32 = ${o}u;`,h=e=>`var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = bitcast<f32>(partial[partialOffset + tile * ${e} + index]) - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`;return[G(e,{id:`${n}-means-tiles`,variant:`means-tiles`,workgroupSize:j,bindings:[...f,H(`partial`,t.partialMeans,`u32`)],invocationCount:o,declarations:`${I}
${R(i,r,a,d)}`,body:`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sums: array<f32, ${c}>;
  var count = 0u;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    count = count + 1u;
    for (var j = 0u; j < ROW_K; j++) {
      sums[j] = sums[j] + design[designOffset + row * ROW_K + j];
    }
    sums[ROW_K] = sums[ROW_K] + response[responseOffset + row];
  }
  let base = partialOffset + index * ${c+1}u;
  for (var column = 0u; column < ${c}u; column++) {
    partial[base + column] = bitcast<u32>(sums[column]);
  }
  partial[base + ${c}u] = count;`}),G(e,{id:`${n}-means-merge`,variant:`means-merge`,bindings:[V(`partial`,t.partialMeans,`u32`),W(s)],invocationCount:c,declarations:m,body:`var rows = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    rows = rows + partial[partialOffset + tile * ${c+1}u + ${c}u];
  }
  ${h(`${c+1}u`)}
  wsSet(W_MEANS + index, select(0.0, sum / f32(rows), rows > 0u));
  if (index == 0u) {
    ws[wsOffset + W_COUNT] = rows;
  }`}),G(e,{id:`${n}-moments-tiles`,variant:`moments-tiles`,workgroupSize:j,bindings:[...f,U(s),H(`partial`,t.partialMoments,`u32`)],invocationCount:o,declarations:p,body:`var means: array<f32, ${c}>;
  for (var column = 0u; column < D; column++) {
    means[column] = wsGet(W_MEANS + column);
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var accumulator: array<f32, ${u}>;
  var centered: array<f32, ${c}>;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    for (var j = 0u; j < ROW_K; j++) {
      centered[j] = design[designOffset + row * ROW_K + j] - means[j];
    }
    centered[ROW_K] = response[responseOffset + row] - means[ROW_K];
    var slot = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        accumulator[slot] = accumulator[slot] + centered[first] * centered[second];
        slot = slot + 1u;
      }
    }
  }
  for (var slot = 0u; slot < ${u}u; slot++) {
    partial[partialOffset + index * ${u}u + slot] = bitcast<u32>(accumulator[slot]);
  }`}),G(e,{id:`${n}-moments-merge`,variant:`moments-merge`,bindings:[V(`partial`,t.partialMoments,`u32`),W(s)],invocationCount:u,declarations:m,body:`${h(`${u}u`)}
  wsSet(W_MOMENTS + index, sum);`})]}function ie(e,t){return`let rows = ws[wsOffset + W_COUNT];
  ${t?`let requested = params[paramsOffset];
  let lambda = select(0.0, requested, requested > 0.0 && isFiniteValue(requested));`:`let lambda = 0.0;`}
  wsSet(W_LAMBDA, lambda);
  if (rows <= K + 1u) {
    ws[wsOffset + W_STATUS] = 2u;
    return;
  }
  var a: array<f32, ${e*e}>;
  var rhs: array<f32, ${e}>;
  var scale: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    let variance = momentAt(i, i);
    if (!(variance > 0.0) || !isFiniteValue(variance)) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
    scale[i] = sqrt(variance);
  }
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j <= i; j++) {
      var entry = momentAt(i, j) / (scale[i] * scale[j]);
      if (i == j) {
        entry = entry + lambda / (scale[i] * scale[i]);
      }
      a[i * K + j] = entry;
    }
    rhs[i] = momentAt(i, K) / scale[i];
  }
  if (!choleskyFactor_${e}(&a)) {
    ws[wsOffset + W_STATUS] = 1u;
    return;
  }
  for (var i = 0u; i < K; i++) {
    if (!(a[i * K + i] >= ${u(b)})) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
  }
  for (var i = 0u; i < KK; i++) {
    wsSet(W_FACTOR + i, a[i]);
  }
  choleskySolve_${e}(&a, &rhs);
  for (var i = 0u; i < K; i++) {
    wsSet(W_SCALE + i, scale[i]);
    wsSet(W_SLOPES + i, rhs[i] / scale[i]);
  }
  ws[wsOffset + W_STATUS] = 0u;`}function z(e){let t=e+1;return`let rows = ws[wsOffset + W_COUNT];
  let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  statusOut[statusOutOffset] = status;
  if (status != 0u) {
    for (var i = 0u; i < ${t}u; i++) {
      coefficientsOut[coefficientsOutOffset + i] = nan;
      standardErrorsOut[standardErrorsOutOffset + i] = nan;
      tStatisticsOut[tStatisticsOutOffset + i] = nan;
    }
    for (var slot = 0u; slot < 22u; slot++) {
      summaryOut[summaryOutOffset + slot] = nan;
    }
    summaryOut[summaryOutOffset] = f32(rows);
    return;
  }
  let n = f32(rows);
  let freedom = n - ${t}.0;
  let residualSumSquares = wsGet(W_RESIDUAL_SUMS);
  let totalSumSquares = momentAt(K, K);
  var a: array<f32, ${e*e}>;
  for (var i = 0u; i < KK; i++) {
    a[i] = wsGet(W_FACTOR + i);
  }
  let sigmaSquared = residualSumSquares / freedom;
  var intercept = wsGet(W_MEANS + K);
  var u: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    intercept = intercept - wsGet(W_SLOPES + i) * wsGet(W_MEANS + i);
    u[i] = wsGet(W_MEANS + i) / wsGet(W_SCALE + i);
  }
  coefficientsOut[coefficientsOutOffset] = intercept;
  for (var i = 0u; i < K; i++) {
    coefficientsOut[coefficientsOutOffset + 1u + i] = wsGet(W_SLOPES + i);
  }
  // Intercept variance: sigma2 (1 / n + u' R^-1 u) with u = xbar / scale.
  choleskySolve_${e}(&a, &u);
  var quadratic = 0.0;
  for (var i = 0u; i < K; i++) {
    quadratic = quadratic + (wsGet(W_MEANS + i) / wsGet(W_SCALE + i)) * u[i];
  }
  standardErrorsOut[standardErrorsOutOffset] = sqrt(max(sigmaSquared * (1.0 / n + quadratic), 0.0));
  var maximumInflation = 0.0;
  for (var i = 0u; i < K; i++) {
    let scale = wsGet(W_SCALE + i);
    // Diagonal of the inverse correlation matrix is the variance inflation factor of column i.
    let correlationInverse = choleskyInverseDiagonal_${e}(&a, i);
    maximumInflation = max(maximumInflation, correlationInverse);
    let inverse = correlationInverse / (scale * scale);
    standardErrorsOut[standardErrorsOutOffset + 1u + i] = sqrt(max(sigmaSquared * inverse, 0.0));
  }
  for (var i = 0u; i < ${t}u; i++) {
    let standardError = standardErrorsOut[standardErrorsOutOffset + i];
    tStatisticsOut[tStatisticsOutOffset + i] = select(
      nan,
      coefficientsOut[coefficientsOutOffset + i] / standardError,
      standardError > 0.0
    );
  }
  let rSquared = select(nan, 1.0 - residualSumSquares / totalSumSquares, totalSumSquares > 0.0);
  let meanSquare = max(residualSumSquares / n, 1.17549435e-38);
  let logLikelihood = -0.5 * n * (1.8378770664093453 + log(meanSquare) + 1.0);
  var jarqueBera = nan;
  var jarqueBeraP = nan;
  var skewness = nan;
  var kurtosis = nan;
  let m2 = residualSumSquares / n;
  if (m2 > 0.0) {
    let m3 = wsGet(W_RESIDUAL_SUMS + 1u) / n;
    let m4 = wsGet(W_RESIDUAL_SUMS + 2u) / n;
    skewness = m3 / (m2 * sqrt(m2));
    kurtosis = m4 / (m2 * m2);
    let excess = kurtosis - 3.0;
    jarqueBera = n / 6.0 * (skewness * skewness + 0.25 * excess * excess);
    jarqueBeraP = exp(-0.5 * jarqueBera);
  }
  for (var slot = 0u; slot < 22u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset + 0u] = n;
  summaryOut[summaryOutOffset + 1u] = rSquared;
  summaryOut[summaryOutOffset + 2u] = 1.0 - (1.0 - rSquared) * (n - 1.0) / freedom;
  summaryOut[summaryOutOffset + 3u] = sigmaSquared;
  summaryOut[summaryOutOffset + 4u] = logLikelihood;
  summaryOut[summaryOutOffset + 5u] = 2.0 * ${t}.0 - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 6u] = ${t}.0 * log(n) - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 7u] = jarqueBera;
  summaryOut[summaryOutOffset + 8u] = jarqueBeraP;
  summaryOut[summaryOutOffset + 11u] = residualSumSquares;
  summaryOut[summaryOutOffset + 12u] = totalSumSquares;
  summaryOut[summaryOutOffset + 13u] = skewness;
  summaryOut[summaryOutOffset + 14u] = kurtosis;
  summaryOut[summaryOutOffset + 15u] = wsGet(W_LAMBDA);

  // Conditioning. Belsley (spreg condition_index) of the raw [1, X] design, from the Cholesky
  // factor of the unpenalized predictor correlation matrix and the column means, never from the
  // raw Gram matrix (its smallest eigenvalue is below float32 resolution once the condition
  // number passes a few hundred). See getBelsleyConditionNumberWGSL.
  var unpenalized: array<f32, ${e*e}>;
  var alpha: array<f32, ${e}>;
  var beta: array<f32, ${e}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      unpenalized[i * K + j] = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
    }
    let centered = momentAt(i, i);
    let meanI = wsGet(W_MEANS + i);
    let rawNorm = sqrt(n * meanI * meanI + centered);
    alpha[i] = sqrt(n) * meanI / rawNorm;
    beta[i] = sqrt(centered) / rawNorm;
  }
  var belsley = 3.0e38;
  if (choleskyFactor_${e}(&unpenalized)) {
    belsley = getBelsleyConditionNumber_${e}(&unpenalized, &alpha, &beta);
  }
  // The factored (optionally ridge-penalized) predictor correlation matrix.
  var correlation: array<f32, ${e*e}>;
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j < K; j++) {
      var entry = momentAt(i, j) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + j));
      if (i == j) {
        entry = entry + wsGet(W_LAMBDA) / (wsGet(W_SCALE + i) * wsGet(W_SCALE + i));
      }
      correlation[i * K + j] = entry;
    }
  }
  let correlationRange = getSymmetricEigenvalueRange_${e}(&correlation);
  let correlationCondition = correlationRange.y / max(correlationRange.x, 1.0e-30);
  summaryOut[summaryOutOffset + 18u] =
    belsley;
  summaryOut[summaryOutOffset + 19u] =
    correlationCondition;
  summaryOut[summaryOutOffset + 20u] =
    maximumInflation;
  summaryOut[summaryOutOffset + 21u] =
    select(0.0, 1.0, correlationCondition > ${u(_)});`}function B(e){return`
const DEGREES_OF_FREEDOM: u32 = ${e}u;
const LOG_GAMMA: f32 = ${u(S(e))};

fn getChiSquareSurvival(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = f32(DEGREES_OF_FREEDOM) * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - LOG_GAMMA;
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= 200u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= 200u; iteration++) {
    let fi = f32(iteration);
    let an = -fi * (fi - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 1.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`}function ae(e){return`let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  var koenker = nan;
  var koenkerP = nan;
  var breusch = nan;
  var breuschP = nan;
  let totalSumSquares = momentAt(K, K);
  if (status == 0u && totalSumSquares > 0.0) {
    var explained = 0.0;
    for (var i = 0u; i < ${e}u; i++) {
      explained = explained + wsGet(W_SLOPES + i) * momentAt(i, K);
    }
    let rSquared = clamp(explained / totalSumSquares, 0.0, 1.0);
    let n = f32(ws[wsOffset + W_COUNT]);
    koenker = n * rSquared;
    koenkerP = getChiSquareSurvival(koenker);
    // Classic Breusch-Pagan: 0.5 R2 (sum e^4 / s2^2 - n) = 0.5 R2 n (kurtosis - 1), s2 = RSS / n.
    let residualSumSquares = bitcast<f32>(mainWs[mainWsOffset + W_RESIDUAL_SUMS]);
    let sumFourth = bitcast<f32>(mainWs[mainWsOffset + W_RESIDUAL_SUMS + 2u]);
    if (residualSumSquares > 0.0) {
      let kurtosis = sumFourth * n / (residualSumSquares * residualSumSquares);
      breusch = max(0.5 * rSquared * n * (kurtosis - 1.0), 0.0);
      breuschP = getChiSquareSurvival(breusch);
    }
  }
  summaryOut[summaryOutOffset + 9u] = koenker;
  summaryOut[summaryOutOffset + 10u] = koenkerP;
  summaryOut[summaryOutOffset + 16u] = breusch;
  summaryOut[summaryOutOffset + 17u] = breuschP;`}function V(e,t,n){return{name:e,view:t,type:n,access:`read`}}function H(e,t,n){return{name:e,view:t,type:n,access:`read_write`}}function U(e){return V(`ws`,e,`u32`)}function W(e){return H(`ws`,e,`u32`)}function G(e,t){return f(e,{operation:A,...t})}var oe=256;function se(e,t){let{id:n,operation:r,positions:i,rowCount:a,gridSize:o}=t,c=Math.ceil(a/oe),u=o[0]*o[1],d=(t,r,i)=>s(e,`${n}-${t}`,r,Math.max(i,1)),p=d(`tile-bounds`,`float32`,c*4),m=d(`grid-bounds`,`float32`,4),g=d(`cell-offsets`,`uint32`,u+1),_=d(`object-ids`,`uint32`,a),v=d(`sorted-ids`,`uint32`,a+4),y=d(`index-count`,`uint32`,1),b=d(`index-overflow`,`uint32`,1),x=`const ROW_COUNT: u32 = ${a}u;
const TILE_ROWS: u32 = ${oe}u;
const TILE_COUNT: u32 = ${c}u;
const SENTINEL: f32 = 3.0e38;
${h}`;return{nodes:[f(e,{id:`${n}-tile-bounds`,operation:r,variant:`tile-bounds`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`tileBounds`,view:p,type:`f32`,access:`read_write`}],invocationCount:c,declarations:x,body:`let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var bounds = vec4f(SENTINEL, SENTINEL, -SENTINEL, -SENTINEL);
  for (var row = firstRow; row < endRow; row++) {
    let x = positions[positionsOffset + 2u * row];
    let y = positions[positionsOffset + 2u * row + 1u];
    if (isFiniteBits(x) && isFiniteBits(y)) {
      bounds = vec4f(min(bounds.x, x), min(bounds.y, y), max(bounds.z, x), max(bounds.w, y));
    }
  }
  for (var component = 0u; component < 4u; component++) {
    tileBounds[tileBoundsOffset + 4u * index + component] = bounds[component];
  }`}),f(e,{id:`${n}-grid-bounds`,operation:r,variant:`grid-bounds`,bindings:[{name:`tileBounds`,view:p,type:`f32`,access:`read`},{name:`gridBounds`,view:m,type:`f32`,access:`read_write`}],invocationCount:1,declarations:x,body:`var bounds = vec4f(SENTINEL, SENTINEL, -SENTINEL, -SENTINEL);
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let base = tileBoundsOffset + 4u * tile;
    bounds = vec4f(
      min(bounds.x, tileBounds[base]), min(bounds.y, tileBounds[base + 1u]),
      max(bounds.z, tileBounds[base + 2u]), max(bounds.w, tileBounds[base + 3u])
    );
  }
  let isEmpty = bounds.x > bounds.z || bounds.y > bounds.w;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  for (var component = 0u; component < 4u; component++) {
    gridBounds[gridBoundsOffset + component] = select(bounds[component], invalid, isEmpty);
  }`}),...new l({id:`${n}-grid-index`,positions:i,gridSize:[o[0],o[1]],bounds:[0,0,1,1],boundsBuffer:m,cellOffsets:g,objectIds:_,count:y,overflow:b}).getCommandNodes(e),f(e,{id:`${n}-sort-cells`,operation:r,variant:`sort-cells`,bindings:[{name:`cellOffsets`,view:g,type:`u32`,access:`read`},{name:`objectIds`,view:_,type:`u32`,access:`read`},{name:`indexCount`,view:y,type:`u32`,access:`read`},{name:`gridBounds`,view:m,type:`f32`,access:`read`},{name:`sortedIds`,view:v,type:`u32`,access:`read_write`}],invocationCount:a,declarations:`const INDEX_CELL_COUNT: u32 = ${u}u;
const ROW_COUNT: u32 = ${a}u;`,body:`if (index < 4u) {
    sortedIds[sortedIdsOffset + ROW_COUNT + index] = bitcast<u32>(gridBounds[gridBoundsOffset + index]);
  }
  if (index >= min(indexCount[indexCountOffset], ROW_COUNT)) {
    return;
  }
  let row = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = INDEX_CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let start = cellOffsets[cellOffsetsOffset + low];
  let end = cellOffsets[cellOffsetsOffset + low + 1u];
  var rank = 0u;
  for (var slot = start; slot < end; slot++) {
    rank += select(0u, 1u, objectIds[objectIdsOffset + slot] < row);
  }
  sortedIds[sortedIdsOffset + start + rank] = row;`})],cellOffsets:g,sortedIds:v}}function K(e){return`
const INDEX_WIDTH: u32 = ${e[0]}u;
const INDEX_HEIGHT: u32 = ${e[1]}u;

// Same cell mapping as GPUGridIndex.
fn getGridCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
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
}

// Inclusive cell range (columnLow, columnHigh, rowLow, rowHigh) covering distance < bandwidth.
// The reach is padded for f32 rounding of the box edges; rows in the padding get weight 0 or
// a weight below 1e-8 and the exact distance test in the weight rejects the rest.
fn getGridRange(origin: vec2f, bandwidth: f32) -> vec4u {
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let reach = vec2f(bandwidth * 1.0001) + abs(origin) * 2.4e-7;
  let low = clamp(origin - reach, minimum, maximum);
  let high = clamp(origin + reach, minimum, maximum);
  return vec4u(
    getGridCoordinate(low.x, minimum.x, maximum.x, INDEX_WIDTH),
    getGridCoordinate(high.x, minimum.x, maximum.x, INDEX_WIDTH),
    getGridCoordinate(low.y, minimum.y, maximum.y, INDEX_HEIGHT),
    getGridCoordinate(high.y, minimum.y, maximum.y, INDEX_HEIGHT)
  );
}
`}function q(){return`
fn selectNeighborDistances(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  let cellExtent = (maximum - minimum) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
  let cellSlack = min(cellExtent.x, cellExtent.y);
  let lastColumn = i32(INDEX_WIDTH) - 1;
  let lastRow = i32(INDEX_HEIGHT) - 1;
  let centerColumn = i32(getGridCoordinate(clamp(origin.x, minimum.x, maximum.x), minimum.x, maximum.x, INDEX_WIDTH));
  let centerRow = i32(getGridCoordinate(clamp(origin.y, minimum.y, maximum.y), minimum.y, maximum.y, INDEX_HEIGHT));
  let lastRing = max(max(centerColumn, lastColumn - centerColumn), max(centerRow, lastRow - centerRow));
  for (var ring = 0; ring <= lastRing; ring++) {
    let ringColumnLow = max(centerColumn - ring, 0);
    let ringColumnHigh = min(centerColumn + ring, lastColumn);
    for (var cellRow = max(centerRow - ring, 0); cellRow <= min(centerRow + ring, lastRow); cellRow++) {
      let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
      for (var side = 0; side < 2; side++) {
        var firstColumn = ringColumnLow;
        var endColumn = ringColumnHigh;
        if (isEdgeRow) {
          if (side == 1) { continue; }
        } else {
          let column = select(centerColumn + ring, centerColumn - ring, side == 0);
          if (column < 0 || column > lastColumn) { continue; }
          firstColumn = column;
          endColumn = column;
        }
        let rowBase = u32(cellRow) * INDEX_WIDTH;
        let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
        let end = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
        for (var slot = start; slot < end; slot++) {
          let row = sortedIds[sortedIdsOffset + slot];
          let delta = getPosition(row) - origin;
          let squared = dot(delta, delta);
          if (squared < (*list)[maximumK - 1u]) {
            var position = maximumK - 1u;
            loop {
              if (position == 0u || (*list)[position - 1u] <= squared) {
                break;
              }
              (*list)[position] = (*list)[position - 1u];
              position = position - 1u;
            }
            (*list)[position] = squared;
          }
        }
      }
    }
    if ((*list)[maximumK - 1u] < SENTINEL) {
      var gap = 3.0e38;
      if (centerColumn - ring > 0) {
        gap = min(gap, origin.x - (minimum.x + f32(centerColumn - ring) * cellExtent.x));
      }
      if (centerColumn + ring < lastColumn) {
        gap = min(gap, minimum.x + f32(centerColumn + ring + 1) * cellExtent.x - origin.x);
      }
      if (centerRow - ring > 0) {
        gap = min(gap, origin.y - (minimum.y + f32(centerRow - ring) * cellExtent.y));
      }
      if (centerRow + ring < lastRow) {
        gap = min(gap, minimum.y + f32(centerRow + ring + 1) * cellExtent.y - origin.y);
      }
      gap = gap - cellSlack;
      if (gap > 0.0 && (*list)[maximumK - 1u] < gap * gap) {
        break;
      }
    }
  }
}
`}var ce=.05,le=.38197,ue=1e-6,de=4e-6,J=65536,fe=1048576,Y=1024,X=1024,pe=1.00001,me=1e-30,he={gaussian:0,bisquare:1},ge={fixed:0,adaptive:1},Z={OK:0,SINGULAR:1,EXCLUDED:2},_e={RESIDUAL_SUM_OF_SQUARES:0,TRACE_OF_HAT:1,AICC:2,R_SQUARED:3,OBSERVATION_COUNT:4,HAS_VALID_CANDIDATE:5,SINGULAR_ROW_COUNT:6,SIGMA_SQUARED:7,ADJUSTED_ALPHA:8,CRITICAL_T:9,LOG_LIKELIHOOD:10,INCLUDED_ROW_COUNT:11},ve={LOWER:0,LOWER_INTERIOR:1,UPPER:2,UPPER_INTERIOR:3,LOWER_INTERIOR_ROW:4,UPPER_INTERIOR_ROW:5,ITERATION_COUNT:6,ACTIVE:7,OPTIMUM_ROW:8,EVALUATION_COUNT:9,CONVERGED:10,RESERVED:11};function ye(e=32){return 4+e}function be(t,n=32,r=new Float32Array(ye(n))){let i=ye(n);if(r.length<i)throw Error(`GWR parameter target must hold ${i} elements`);let{bandwidths:a}=t;if(a.length<1||a.length>n)throw Error(`GWR bandwidths must hold 1 to ${n} values`);let o=he[t.kernel??`bisquare`],s=ge[t.bandwidthMode??`fixed`];if(o===void 0||s===void 0)throw Error(`GWR kernel or bandwidth mode is not recognized`);r.fill(0,0,i),r[0]=o,r[1]=s,r[2]=a.length;let c=t.significanceLevel;if(c!==void 0){if(!(c>0&&c<1))throw Error(`GWR significanceLevel must be in (0, 1)`);r[3]=c}for(let n=0;n<a.length;n++){if(!Number.isFinite(a[n]))throw Error(`GWR bandwidths must be finite`);s===ge.fixed&&e(`GPUGeographicallyWeightedRegression`,t.spatialContext,a[n]),r[4+n]=a[n]}return r}var xe=ve;function Se(e,t){let{id:n,operation:r,rowCount:i,tileCount:a,candidateWidth:o}=t,s=d(e,{id:`${n}-search-dispatch`,operation:r,shapes:[[i],[a*o],[o]]}),c=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),l=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),u=Object.entries(xe).map(([e,t])=>`const STATE_${e}: u32 = ${t}u;`).join(`
`),p=`${t.common}
const MAXIMUM_ITERATIONS: u32 = ${t.maximumIterations}u;
const GOLDEN_DELTA: f32 = ${le};
const TOLERANCE_ABSOLUTE: f32 = ${ue};
const TOLERANCE_RELATIVE: f32 = ${de};
${u}`,m=`${t.focalRowWGSL}
  let slot = candidateScratchOffset + focal * 2u;
  candidateScratch[slot] = SENTINEL;
  candidateScratch[slot + 1u] = 0.0;
  if (getIsAdaptive() || rowValid[rowValidOffset + focal] == 0u) {
    return;
  }
  let origin = getPosition(focal);
  var nearest = SENTINEL;
  var farthest = 0.0;
  ${t.hasGrid?`// The two smallest squared distances: the row itself, then its nearest neighbour. A
  // coincident row gives 0, which is no positive distance (as for the scan below).
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  selectNeighborDistances(focal, 2u, &neighbors);
  if (neighbors[1] > 0.0 && neighbors[1] < SENTINEL) {
    nearest = sqrt(neighbors[1]);
  }
  // The farthest row is no farther than the farthest corner of the grid extent.
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  farthest = max(
    max(getDistance(origin, minimum), getDistance(origin, maximum)),
    max(getDistance(origin, vec2f(minimum.x, maximum.y)), getDistance(origin, vec2f(maximum.x, minimum.y)))
  );`:`for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let distance = getDistance(origin, getPosition(row));
    if (distance > 0.0) {
      nearest = min(nearest, distance);
    }
    farthest = max(farthest, distance);
  }`}
  candidateScratch[slot] = nearest;
  candidateScratch[slot + 1u] = farthest;`;return{setupNodes:[f(e,{id:`${n}-search-bounds-rows`,operation:r,variant:`search-bounds-rows`,bindings:[...t.candidateBindings,l(`candidateScratch`,t.candidateScratch,`f32`)],invocationCount:i,declarations:t.candidateDeclarations,body:m}),f(e,{id:`${n}-search-bounds-tiles`,operation:r,variant:`search-bounds-tiles`,bindings:[c(`candidateScratch`,t.candidateScratch,`f32`),c(`rowValid`,t.rowValid,`u32`),l(`tileCandidates`,t.tileCandidates,`f32`)],invocationCount:a,declarations:p,body:`let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    nearest = min(nearest, candidateScratch[candidateScratchOffset + row * 2u]);
    farthest = max(farthest, candidateScratch[candidateScratchOffset + row * 2u + 1u]);
  }
  let base = tileCandidatesOffset + index * 3u;
  tileCandidates[base] = nearest;
  tileCandidates[base + 1u] = farthest;
  tileCandidates[base + 2u] = included;`}),f(e,{id:`${n}-search-init`,operation:r,variant:`search-init`,bindings:[c(`params`,t.parameters,`f32`),l(`ladders`,t.ladders,`f32`),c(`tileCandidates`,t.tileCandidates,`f32`),l(`table`,t.candidateTable,`f32`),l(`state`,t.searchState,`f32`),s.binding],invocationCount:1,declarations:p,body:`let nan = getNaN();
  for (var row = 0u; row < TABLE_ROWS; row++) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[tableOffset + row * TABLE_STRIDE + field] = nan;
    }
  }
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  let callerCount = u32(max(params[paramsOffset + 2u], 0.0));
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let base = tileCandidatesOffset + tile * 3u;
    nearest = min(nearest, tileCandidates[base]);
    farthest = max(farthest, tileCandidates[base + 1u]);
    included = included + tileCandidates[base + 2u];
  }
  // mgwr _init_section: adaptive [40 + 2 P, n] (n capped at the neighbour list), fixed
  // [min_dist / 2, 2 max_dist]; a positive caller bound replaces its default.
  var lower = 0.0;
  var upper = 0.0;
  if (isAdaptive) {
    lower = f32(40u + 2u * P);
    upper = min(included, f32(MAXIMUM_NEIGHBORS));
  } else if (included > 0.0 && nearest < SENTINEL) {
    lower = 0.5 * nearest;
    upper = 2.0 * farthest;
  }
  let isSearch = callerCount >= 2u;
  if (isSearch) {
    let givenLower = params[paramsOffset + HEADER];
    let givenUpper = params[paramsOffset + HEADER + 1u];
    if (givenLower > 0.0 && isFiniteBits(givenLower)) {
      lower = givenLower;
    }
    if (givenUpper > 0.0 && isFiniteBits(givenUpper)) {
      upper = givenUpper;
    }
  }
  let low = min(lower, upper);
  let high = max(lower, upper);
  var first = low + GOLDEN_DELTA * (high - low);
  var second = high - GOLDEN_DELTA * (high - low);
  if (isAdaptive) {
    // numpy round: half to even, as WGSL round.
    first = round(first);
    second = round(second);
  }
  if (isSearch) {
    // Block 0 holds the caller header; it now carries the two interior points of iteration 1.
    ladders[laddersOffset + 3u] = 2.0;
    ladders[laddersOffset + 1u + HEADER] = first;
    ladders[laddersOffset + 2u + HEADER] = second;
  }
  state[stateOffset + STATE_LOWER] = low;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = high;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = 0.0;
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = 1.0;
  state[stateOffset + STATE_ITERATION_COUNT] = 0.0;
  state[stateOffset + STATE_ACTIVE] = select(0.0, 1.0, isSearch);
  // A single caller value is a fixed bandwidth: its evaluation is table row 0.
  state[stateOffset + STATE_OPTIMUM_ROW] = select(0.0, -1.0, isSearch);
  state[stateOffset + STATE_EVALUATION_COUNT] = select(1.0, 2.0, isSearch);
  state[stateOffset + STATE_CONVERGED] = 0.0;
  state[stateOffset + STATE_RESERVED] = 0.0;
  if (isSearch) {
    ${s.armWGSL}
  } else {
    ${s.disarmWGSL}
  }`})],gate:s,createIterationNode:i=>f(e,{id:`${n}-search-${i}-iterate`,operation:r,variant:`search-iterate`,bindings:[l(`ladders`,t.ladders,`f32`),c(`table`,t.candidateTable,`f32`),l(`state`,t.searchState,`f32`),s.binding],invocationCount:1,declarations:`${p}
fn getRowScore(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 1u];
}
fn getRowSingular(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 2u];
}`,body:`if (state[stateOffset + STATE_ACTIVE] < 0.5) {
    return;
  }
  let iteration = u32(state[stateOffset + STATE_ITERATION_COUNT]) + 1u;
  let firstRow = u32(state[stateOffset + STATE_LOWER_INTERIOR_ROW]);
  let secondRow = u32(state[stateOffset + STATE_UPPER_INTERIOR_ROW]);
  let firstScore = getRowScore(firstRow);
  let secondScore = getRowScore(secondRow);
  let firstSingular = getRowSingular(firstRow);
  let secondSingular = getRowSingular(secondRow);
  let isFirstValid = isFiniteBits(firstScore);
  let isSecondValid = isFiniteBits(secondScore);
  // mgwr: score_b <= score_d keeps the lower interval. Fewest singular rows first; an invalid
  // score loses to a valid one; with two invalid scores the comparison is false, as for NaN.
  var firstWins = isFirstValid;
  if (isFirstValid && isSecondValid) {
    firstWins = firstSingular < secondSingular || (firstSingular == secondSingular && firstScore <= secondScore);
  }
  let isAdaptive = ladders[laddersOffset + 2u] > 0.5;
  var lower = state[stateOffset + STATE_LOWER];
  var first = state[stateOffset + STATE_LOWER_INTERIOR];
  var upper = state[stateOffset + STATE_UPPER];
  var second = state[stateOffset + STATE_UPPER_INTERIOR];
  let newRow = iteration + 1u;
  var optimumRow = secondRow;
  var nextFirstRow = firstRow;
  var nextSecondRow = secondRow;
  var newValue = 0.0;
  if (firstWins) {
    // c = d; d = b; b = a + delta |c - a|
    optimumRow = firstRow;
    upper = second;
    second = first;
    nextSecondRow = firstRow;
    first = lower + GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      first = round(first);
    }
    newValue = first;
    nextFirstRow = newRow;
  } else {
    // a = b; b = d; d = c - delta |c - a|
    lower = first;
    first = second;
    nextFirstRow = secondRow;
    second = upper - GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      second = round(second);
    }
    newValue = second;
    nextSecondRow = newRow;
  }
  // mgwr loops while |score_b - score_d| > tol; the difference is of the pair just compared.
  var keepSearching = true;
  if (isFirstValid && isSecondValid && firstSingular == secondSingular) {
    let tolerance = max(TOLERANCE_ABSOLUTE, TOLERANCE_RELATIVE * max(abs(firstScore), abs(secondScore)));
    keepSearching = abs(firstScore - secondScore) > tolerance;
  } else if (!isFirstValid && !isSecondValid) {
    keepSearching = false;
  }
  state[stateOffset + STATE_LOWER] = lower;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = upper;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = f32(nextFirstRow);
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = f32(nextSecondRow);
  state[stateOffset + STATE_ITERATION_COUNT] = f32(iteration);
  state[stateOffset + STATE_OPTIMUM_ROW] = f32(optimumRow);
  if (!keepSearching || iteration >= MAXIMUM_ITERATIONS) {
    // Converged or out of iterations: no new candidate, and the later passes launch nothing.
    state[stateOffset + STATE_ACTIVE] = 0.0;
    state[stateOffset + STATE_CONVERGED] = select(0.0, 1.0, !keepSearching);
    ${s.disarmWGSL}
    return;
  }
  // Schedule the one new candidate as block iteration of the ladders and advance the GPU cursor.
  ladders[laddersOffset] = f32(iteration);
  let blockBase = laddersOffset + 1u + iteration * PARAMETER_LENGTH;
  ladders[blockBase] = ladders[laddersOffset + 1u];
  ladders[blockBase + 1u] = ladders[laddersOffset + 2u];
  ladders[blockBase + 2u] = 1.0;
  ladders[blockBase + 3u] = ladders[laddersOffset + 4u];
  ladders[blockBase + HEADER] = newValue;
  state[stateOffset + STATE_EVALUATION_COUNT] = f32(newRow + 1u);`})}}function Ce(e){let{coefficientCount:t,indexGridSize:n,hasConditionNumber:r,hasVariance:i,sphereRadius:a}=e,o=e.gaussianCutoff??null,s=!!n;return`
${a===null?``:g}
${E(t)}
${r?D(t):``}
${n?Ee(n,o):``}
struct LocalFit {
  ok: bool,
  hat: f32,
  weightSum: f32,
  weightedResponse: f32,
  beta: array<f32, ${t}>,
  // diag((X'WX)^-1 X'W^2 X (X'WX)^-1) of the original (uncentered) coefficients; NaN when not
  // finite. Only filled when the squared-weight Gram matrix is accumulated.
  variance: array<f32, ${t}>
}

fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${a===null?`fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return sqrt(dot(delta, delta));
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return dot(delta, delta);
}`:`// Great-circle distance in the units of the sphere radius; positions are (longitude, latitude).
fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  return geodesicHaversineDistance(pointA, pointB, ${u(a)});
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let distance = getDistance(pointA, pointB);
  return distance * distance;
}`}

${o===null?``:`// Gaussian weights vanish beyond this many bandwidths (compile-time gaussianCutoff).
const GAUSSIAN_CUTOFF: f32 = ${u(o)};
`}
fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  let ratio = distance / bandwidth;
  if (getIsBisquare()) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }${o===null?``:`
  if (distance > GAUSSIAN_CUTOFF * bandwidth) {
    return 0.0;
  }`}
  return exp(-0.5 * ratio * ratio);
}

${n&&a===null?q():`// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
fn selectNeighborDistances(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let squared = getSquaredDistance(origin, getPosition(row));
    if (squared < (*list)[maximumK - 1u]) {
      var slot = maximumK - 1u;
      loop {
        if (slot == 0u || (*list)[slot - 1u] <= squared) {
          break;
        }
        (*list)[slot] = (*list)[slot - 1u];
        slot = slot - 1u;
      }
      (*list)[slot] = squared;
    }
  }
}`}

// Positive bandwidth of a ladder value for the focal row of the neighbour list, or -1 when the
// value is invalid there (fixed: not positive; adaptive: k outside [2, MAXIMUM_NEIGHBORS] or more
// than the included rows).
fn getBandwidthForValue(value: f32, isAdaptive: bool, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) -> f32 {
  if (!isAdaptive) {
    return select(-1.0, value, value > 0.0 && isFiniteBits(value));
  }
  if (!(value >= 2.0) || value > f32(MAXIMUM_NEIGHBORS)) {
    return -1.0;
  }
  let squared = (*list)[u32(floor(value + 0.5)) - 1u];
  if (squared >= SENTINEL) {
    return -1.0;
  }
  return sqrt(squared) * ADAPTIVE_FACTOR;
}

fn fitLocation(i: u32, bandwidth: f32) -> LocalFit {
  var a: array<f32, ${t*t}>;
  var b: array<f32, ${t}>;
  var z: array<f32, ${t}>;
  ${i?`var squared: array<f32, ${t*t}>;`:``}
  var fit: LocalFit;
  fit.ok = false;
  fit.hat = 0.0;
  fit.weightSum = 0.0;
  fit.weightedResponse = 0.0;
  for (var column = 0u; column < P; column++) {
    fit.beta[column] = 0.0;
    fit.variance[column] = getNaN();
    b[column] = 0.0;
  }
  for (var entry = 0u; entry < P * P; entry++) {
    a[entry] = 0.0;
    ${i?`squared[entry] = 0.0;`:``}
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${we(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    let y = response[responseOffset + row];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
        ${i?`squared[r * P + c] = squared[r * P + c] + weighted * weight * z[c];`:``}
      }
    }`,s)}
  fit.weightSum = a[0];
  fit.weightedResponse = b[0];
  var scale: array<f32, ${t}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = a[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return fit;
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      a[r * P + c] = a[r * P + c] * scale[r] * scale[c];
    }
    b[r] = b[r] * scale[r];
  }
  if (!choleskyFactor_${t}(&a)) {
    return fit;
  }
  let inverse00 = choleskyInverseDiagonal_${t}(&a, 0u);
  choleskySolve_${t}(&a, &b);
  var finite = isFiniteBits(inverse00);
  for (var r = 0u; r < P; r++) {
    fit.beta[r] = b[r] * scale[r];
    finite = finite && isFiniteBits(fit.beta[r]);
  }
  fit.hat = inverse00 * scale[0] * scale[0];
  fit.ok = finite && fit.hat >= 0.0;
  ${i?`if (fit.ok) {
    // C C' = A^-1 B A^-1 with A = X'WX (centered design) and B = X'W^2 X. Column c of A^-1 comes
    // from the equilibrated factor: A^-1 e_c = s_c * (s o solve(e_c)). The original intercept is
    // gamma_0 - sum x_ic gamma_c, so its direction is t = (1, -x_i) and A^-1 t = s o solve(s o t);
    // variance_c = w' B w.
    for (var column = 0u; column < P; column++) {
      var w: array<f32, ${t}>;
      for (var r = 0u; r < P; r++) {
        w[r] = 0.0;
      }
      var columnScale = scale[column];
      if (column == 0u) {
        columnScale = 1.0;
        w[0] = scale[0];
        for (var r = 1u; r < P; r++) {
          w[r] = -scale[r] * predictors[originBase + r - 1u];
        }
      } else {
        w[column] = 1.0;
      }
      choleskySolve_${t}(&a, &w);
      for (var r = 0u; r < P; r++) {
        w[r] = scale[r] * w[r] * columnScale;
      }
      var quadratic = 0.0;
      for (var r = 0u; r < P; r++) {
        quadratic = quadratic + w[r] * w[r] * squared[r * P + r];
        for (var c = 0u; c < r; c++) {
          quadratic = quadratic + 2.0 * w[r] * w[c] * squared[r * P + c];
        }
      }
      fit.variance[column] = select(getNaN(), quadratic, quadratic >= 0.0 && isFiniteBits(quadratic));
    }
  }`:``}
  return fit;
}

// Weighted R^2 of the local fit over the neighborhood of row i.
fn getLocalR2(i: u32, bandwidth: f32, fit: LocalFit) -> f32 {
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  let weightedMean = fit.weightedResponse / fit.weightSum;
  var residualSum = 0.0;
  var totalSum = 0.0;
  ${we(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    var prediction = fit.beta[0];
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      prediction = prediction + fit.beta[column + 1u] * (predictors[rowBase + column] - predictors[originBase + column]);
    }
    let y = response[responseOffset + row];
    residualSum = residualSum + weight * (y - prediction) * (y - prediction);
    totalSum = totalSum + weight * (y - weightedMean) * (y - weightedMean);`,s)}
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
${r?Te(t,s):``}
`}function we(e,t){return t?`let useGrid = isGridBandwidth(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, getSupportRadius(bandwidth)), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = ROW_COUNT;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      if (rowValid[rowValidOffset + row] == 0u) {
        continue;
      }
      ${e}
    }
  }`:`for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    ${e}
  }`}function Te(e,t){let n=e;return`
fn getLocalConditionNumber(i: u32, bandwidth: f32) -> f32 {
  var moment: array<f32, ${n*n}>;
  var z: array<f32, ${n}>;
  for (var entry = 0u; entry < P * P; entry++) {
    moment[entry] = 0.0;
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${we(`let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let squaredWeight = weight * weight;
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    for (var r = 0u; r < P; r++) {
      for (var c = 0u; c <= r; c++) {
        moment[r * P + c] = moment[r * P + c] + squaredWeight * z[r] * z[c];
      }
    }`,t)}
  // Shift the centered moment to raw coordinates: x = (1, x_i + z), so raw = T moment T'.
  var center: array<f32, ${n}>;
  center[0] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    center[column + 1u] = predictors[originBase + column];
  }
  var raw: array<f32, ${n*n}>;
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      // T = I + e_r' center, with center[0] = 0 and the intercept row of T equal to e_0.
      let m00 = moment[0];
      let mr0 = moment[r * P];
      let mc0 = moment[c * P];
      var value = moment[r * P + c];
      if (r > 0u && c > 0u) {
        value = value + center[r] * mc0 + center[c] * mr0 + center[r] * center[c] * m00;
      } else if (r > 0u) {
        value = value + center[r] * m00;
      }
      raw[r * P + c] = value;
      raw[c * P + r] = value;
    }
  }
  var scale: array<f32, ${n}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = raw[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return getNaN();
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c < P; c++) {
      raw[r * P + c] = raw[r * P + c] * scale[r] * scale[c];
    }
  }
  let eigenvalues = getSymmetricEigenvalueRange_${n}(&raw);
  return sqrt(eigenvalues.y / max(eigenvalues.x, 1.0e-30));
}
`}function Ee(e,t){return`
${K(e)}

// Bisquare has bounded support (fixed or adaptive), so the grid is exact for it; the Gaussian
// kernel is bounded only with a compile-time cutoff.
fn isGridBandwidth(bandwidth: f32) -> bool {
  return ${t===null?`getIsBisquare()`:`true`} && bandwidth > 0.0 && isFiniteBits(bandwidth);
}

// Distance beyond which the weight is exactly 0.
fn getSupportRadius(bandwidth: f32) -> f32 {
  ${t===null?`return bandwidth;`:`return select(bandwidth * GAUSSIAN_CUTOFF, bandwidth, getIsBisquare());`}
}

// Included rows in grid-cell order: invocation \`slot\` fits the slot-th indexed row.
fn getFocalCount() -> u32 {
  return min(cellOffsets[cellOffsetsOffset + INDEX_WIDTH * INDEX_HEIGHT], ROW_COUNT);
}

fn getFocalRow(slot: u32) -> u32 {
  return sortedIds[sortedIdsOffset + slot];
}
`}var De=2e5,Oe=3e3,ke=40;function Ae(){return`
const GWR_T_EPSILON: f32 = 3.0e-7;
const GWR_T_MINIMUM: f32 = 1.0e-30;
const GWR_T_SQRT_PI: f32 = 1.7724538509055159;
const GWR_T_MAXIMUM_DF: f32 = ${De}.0;

// log(1 + z) without the rounding of 1 + z (z small and positive).
fn gwrLog1p(z: f32) -> f32 {
  let u = 1.0 + z;
  if (u == 1.0) {
    return z;
  }
  return log(u) * z / (u - 1.0);
}

// Gamma(a + 1/2) / Gamma(a) for a > 0: recurrence up to a >= 10, then the asymptotic series.
fn gwrGammaHalfRatio(a: f32) -> f32 {
  var s = a;
  var factor = 1.0;
  for (var step = 0u; step < 12u; step++) {
    if (s >= 10.0) {
      break;
    }
    factor = factor * s / (s + 0.5);
    s = s + 1.0;
  }
  let inverse = 1.0 / s;
  let series = 1.0 - 0.125 * inverse +
    inverse * inverse * (1.0 / 128.0 + inverse * (5.0 / 1024.0 -
      inverse * (21.0 / 32768.0 + inverse * (399.0 / 262144.0 - inverse * (869.0 / 4194304.0)))));
  return factor * sqrt(s) * series;
}

// Lentz continued fraction of the incomplete beta function (Numerical Recipes betacf).
fn gwrBetaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < GWR_T_MINIMUM) {
    d = GWR_T_MINIMUM;
  }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= ${Oe}u; m++) {
    let fm = f32(m);
    let m2 = 2.0 * fm;
    var aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    h = h * d * c;
    aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < GWR_T_MINIMUM) {
      d = GWR_T_MINIMUM;
    }
    c = 1.0 + aa / c;
    if (abs(c) < GWR_T_MINIMUM) {
      c = GWR_T_MINIMUM;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < GWR_T_EPSILON) {
      break;
    }
  }
  return h;
}

// P(|T| >= t) for a Student t variate with df degrees of freedom.
fn gwrStudentTwoSidedTail(t: f32, degreesOfFreedom: f32) -> f32 {
  if (!(t > 0.0)) {
    return 1.0;
  }
  let df = min(degreesOfFreedom, GWR_T_MAXIMUM_DF);
  let a = 0.5 * df;
  let ratio = t * t / df;
  let x = 1.0 / (1.0 + ratio);
  let y = ratio / (1.0 + ratio);
  let front = exp(-a * gwrLog1p(ratio)) * sqrt(y) * gwrGammaHalfRatio(a) / GWR_T_SQRT_PI;
  if (x < (a + 1.0) / (a + 2.5)) {
    return front * gwrBetaContinuedFraction(a, 0.5, x) / a;
  }
  return 1.0 - 2.0 * front * gwrBetaContinuedFraction(0.5, a, y);
}

// The t >= 0 with P(|T| >= t) = tail.
fn gwrStudentCriticalValue(tail: f32, degreesOfFreedom: f32) -> f32 {
  if (!(tail > 0.0) || !(degreesOfFreedom >= 1.0) || tail > 3.0e38) {
    return getNaN();
  }
  if (tail >= 1.0) {
    return 0.0;
  }
  var low = 0.0;
  var high = 1.0;
  for (var step = 0u; step < 60u; step++) {
    if (gwrStudentTwoSidedTail(high, degreesOfFreedom) < tail) {
      break;
    }
    low = high;
    high = high * 2.0;
  }
  for (var step = 0u; step < ${ke}u; step++) {
    let middle = 0.5 * (low + high);
    if (gwrStudentTwoSidedTail(middle, degreesOfFreedom) < tail) {
      high = middle;
    } else {
      low = middle;
    }
  }
  return 0.5 * (low + high);
}
`}var je=Object.freeze({...n,measure:`distance`}),Q=`GPUGeographicallyWeightedRegression`,Me=256,Ne=6,Pe=5,Fe=8,Ie=134217728,$={SELECTED_INDEX:0,SELECTED_VALUE:1,KERNEL:2,MODE:3,SIGMA_SQUARED:4,CRITICAL_T:5},Le=class{id;props;rowCount;coefficientCount;maximumBandwidthCount;maximumNeighborCount;indexGridSize;spatialContext;refinementPassCount;bandwidthSearch;maximumSearchIterations;gaussianCutoff;candidateWidth;candidateTableRowCount;constructor(e){this.id=e.id??`geographically-weighted-regression`,this.props=e;let n=this.id,{predictorCount:r,output:i}=e;a(n,`predictorCount`,r,1,11),this.coefficientCount=r+1,this.spatialContext=t(n,e.spatialContext,je),this.maximumBandwidthCount=e.maximumBandwidthCount??32,a(n,`maximumBandwidthCount`,this.maximumBandwidthCount,1,32),this.maximumNeighborCount=e.maximumNeighborCount??128,a(n,`maximumNeighborCount`,this.maximumNeighborCount,1,128);let s=e.refinement??!0;typeof s==`number`&&a(n,`refinement`,s,0,4);let l=e.bandwidthSearch??`ladder`;if(l!==`ladder`&&l!==`golden-section`)throw Error(`${n} bandwidthSearch must be 'ladder' or 'golden-section'`);this.bandwidthSearch=l;let u=l===`golden-section`;if(u){if(this.maximumBandwidthCount<2)throw Error(`${n} golden-section search needs maximumBandwidthCount >= 2`);if(this.maximumNeighborCount<2)throw Error(`${n} golden-section search needs maximumNeighborCount >= 2`)}this.maximumSearchIterations=u?e.maximumSearchIterations??30:0,u&&a(n,`maximumSearchIterations`,this.maximumSearchIterations,1,64);let{gaussianCutoff:d}=e;if(d!==void 0&&!(Number.isFinite(d)&&d>0))throw Error(`${n} gaussianCutoff must be a positive finite number`);if(this.gaussianCutoff=d??null,this.refinementPassCount=this.maximumBandwidthCount<2||u?0:s===!0?2:s===!1?0:s,this.candidateWidth=u?2:this.maximumBandwidthCount,this.candidateTableRowCount=u?this.maximumSearchIterations+1:(1+this.refinementPassCount)*this.maximumBandwidthCount,c(e.positions,[`float32x2`],`${n} positions`),this.rowCount=e.positions.length,this.rowCount<1)throw Error(`${n} needs at least one row`);if(this.spatialContext.metric===`great-circle`){if(e.indexGridSize)throw Error(`${n} indexGridSize is planar-only; omit it for a great-circle context`);this.indexGridSize=null}else if(e.indexGridSize===!1)this.indexGridSize=null;else if(e.indexGridSize){let[t,r]=e.indexGridSize;if(!Number.isInteger(t)||!Number.isInteger(r)||t<1||r<1||t*r>=4294967295)throw Error(`${n} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`);this.indexGridSize=[t,r]}else if(this.rowCount>=1024){let e=Math.min(X,Math.max(1,Math.ceil(Math.sqrt(this.rowCount/8))));this.indexGridSize=[e,e]}else this.indexGridSize=null;if(!this.indexGridSize&&this.rowCount>65536)throw Error(`${n} supports at most ${J} rows without a grid index`);if(this.indexGridSize&&(this.rowCount>1048576||this.rowCount*this.candidateWidth*8>Ie))throw Error(`${n} supports at most ${fe} rows and rows * ${u?`2`:`maximumBandwidthCount`} * 8 <= ${Ie} with a grid index`);if(this.getRowStateWordCount()*4>Ie)throw Error(`${n} row state exceeds one storage binding; reduce rows, predictors or inference outputs`);if(c(e.predictors,[`float32`],`${n} predictors`),e.predictors.length!==this.rowCount*r)throw Error(`${n} predictors length must equal rows * predictorCount`);if(c(e.response,[`float32`],`${n} response`),e.response.length!==this.rowCount)throw Error(`${n} response length must equal the row count`);if(e.mask&&(c(e.mask,[`uint32`],`${n} mask`),e.mask.length!==this.rowCount))throw Error(`${n} mask length must equal the row count`);c(e.parameters,[`float32`],`${n} parameters`);let f=ye(this.maximumBandwidthCount);if(e.parameters.length<f)throw Error(`${n} parameters must hold ${f} float32 values`);let p=this.candidateTableRowCount*7,m=[[`coefficients`,this.rowCount*this.coefficientCount],[`standardErrors`,this.rowCount*this.coefficientCount],[`tValues`,this.rowCount*this.coefficientCount],[`significant`,this.rowCount*this.coefficientCount],[`localR2`,this.rowCount],[`fitted`,this.rowCount],[`residuals`,this.rowCount],[`hatDiagonal`,this.rowCount],[`localStatus`,this.rowCount],[`localConditionNumber`,this.rowCount],[`bandwidthScores`,this.maximumBandwidthCount],[`bandwidthSingularRowCounts`,this.maximumBandwidthCount],[`candidateTable`,p],[`selectedBandwidth`,2],[`searchState`,12],[`summary`,12]];if(i.searchState&&!u)throw Error(`${n} output.searchState needs bandwidthSearch 'golden-section'`);let h=new Set([`localStatus`,`significant`,`bandwidthSingularRowCounts`]);for(let[e,t]of m){let r=i[e];if(!r){if(e===`coefficients`)throw Error(`${n} output.coefficients is required`);continue}if(h.has(e)?c(r,[`uint32`],`${n} output.${e}`):c(r,[`float32`],`${n} output.${e}`),r.length<t)throw Error(`${n} output.${e} must hold at least ${t} values`)}o(n,m.map(([e])=>i[e]),[e.positions,e.predictors,e.response,e.mask,e.parameters])}hasInference(){let{output:e}=this.props;return!!(e.standardErrors||e.tValues||e.significant)}hasTValues(){let{output:e}=this.props;return!!(e.tValues||e.significant)}getRowStateWordCount(){let{rowCount:e,coefficientCount:t}=this;return e*(1+Ne)+(this.hasInference()?e*t:0)+(this.hasTValues()?e*t:0)+Fe}getCommandNodes(e){let{props:t,id:n,rowCount:a,coefficientCount:o,maximumBandwidthCount:c,maximumNeighborCount:l,indexGridSize:u,refinementPassCount:d,maximumSearchIterations:p,candidateWidth:m,candidateTableRowCount:h,gaussianCutoff:g}=this,_=this.bandwidthSearch===`golden-section`,{output:v,predictorCount:y,positions:b,predictors:x,response:S,parameters:ee}=t;i(n,e,[b,x,S,t.mask,ee,...Object.values(v)]);let te=_?p:1+d,C=Math.ceil(a/Me),w=this.hasInference(),T=this.hasTValues(),E=o,D=ye(c),O=(t,r,i)=>s(e,`${n}-${t}`,r,Math.max(i,1)),k=O(`row-state`,`uint32`,this.getRowStateWordCount()),A=O(`candidate-scratch`,`float32`,a*m*2),j=O(`tile-candidates`,`float32`,C*m*Pe),M=v.candidateTable??O(`candidate-table`,`float32`,h*7),N=O(`ladders`,`float32`,1+te*D),ne=O(`tile-statistics`,`float32`,C*3),P=v.selectedBandwidth??O(`selected-bandwidth`,`float32`,2),F=_?v.searchState??O(`search-state`,`float32`,12):null,I=v.summary??O(`summary`,`float32`,12),L=u?O(`grid-positions`,`float32x2`,a):null,R=u&&L?se(e,{id:n,operation:Q,positions:L,rowCount:a,gridSize:u}):null,re=R?.cellOffsets??null,ie=R?.sortedIds??null,z=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),B=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),ae=a*(1+Ne),V=ae+(w?a*E:0),H=V+(T?a*E:0),U=this.spatialContext.metric===`great-circle`,W=U?r(this.spatialContext):null,G=`const ROW_COUNT: u32 = ${a}u;
const PREDICTOR_COUNT: u32 = ${y}u;
const P: u32 = ${E}u;
const LADDER: u32 = ${m}u;
const TABLE_ROWS: u32 = ${h}u;
const MAXIMUM_NEIGHBORS: u32 = ${l}u;
const TILE_ROWS: u32 = ${Me}u;
const TILE_COUNT: u32 = ${C}u;
const LOCAL_STRIDE: u32 = ${Ne}u;
const TILE_CANDIDATE_STRIDE: u32 = ${Pe}u;
const TABLE_STRIDE: u32 = 7u;
// Layout of the rowValid binding: validity flags, local results, standard errors, t values, tail.
const LOCALS_BASE: u32 = ${a}u;
const STANDARD_ERROR_BASE: u32 = ${ae}u;
const T_VALUE_BASE: u32 = ${V}u;
const TAIL_BASE: u32 = ${H}u;
const TAIL_SELECTED_INDEX: u32 = ${$.SELECTED_INDEX}u;
const TAIL_SELECTED_VALUE: u32 = ${$.SELECTED_VALUE}u;
const TAIL_KERNEL: u32 = ${$.KERNEL}u;
const TAIL_MODE: u32 = ${$.MODE}u;
const TAIL_SIGMA_SQUARED: u32 = ${$.SIGMA_SQUARED}u;
const TAIL_CRITICAL_T: u32 = ${$.CRITICAL_T}u;
const HEADER: u32 = 4u;
const PARAMETER_LENGTH: u32 = ${D}u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${pe};
const MINIMUM_VARIANCE: f32 = ${me};
const LN_TWO_PI: f32 = 1.8378770664093453;
${ze}`,oe=u?[z(`cellOffsets`,re,`u32`),z(`sortedIds`,ie,`u32`)]:[],K=t.mask,q=u?`if (index >= getFocalCount()) {
    return;
  }
  let focal = getFocalRow(index);`:`let focal = index;`,le=`${G}
fn getParameter(slot: u32) -> f32 {
  let passCursor = u32(params[paramsOffset]);
  return params[paramsOffset + 1u + passCursor * PARAMETER_LENGTH + slot];
}
fn getIsBisquare() -> bool {
  return getParameter(0u) > 0.5;
}
fn getIsAdaptive() -> bool {
  return getParameter(1u) > 0.5;
}
fn getCandidateCount() -> u32 {
  return min(u32(max(getParameter(2u), 0.0)), LADDER);
}`,ue=`${le}
${Ce({coefficientCount:E,indexGridSize:u,hasConditionNumber:!1,hasVariance:!1,sphereRadius:W,gaussianCutoff:g})}
// Largest requested k among valid adaptive candidates, clamped to [1, MAXIMUM_NEIGHBORS].
fn getMaximumK(candidateCount: u32) -> u32 {
  var maximumK = 1u;
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let k = getParameter(HEADER + candidate);
    if (k >= 1.0 && k <= f32(MAXIMUM_NEIGHBORS)) {
      maximumK = max(maximumK, u32(floor(k + 0.5)));
    }
  }
  return min(maximumK, MAXIMUM_NEIGHBORS);
}`,de=`${G}
fn getIsBisquare() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL]) > 0.5;
}
fn getIsAdaptive() -> bool {
  return bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE]) > 0.5;
}
${Ce({coefficientCount:E,indexGridSize:u,hasConditionNumber:!!v.localConditionNumber,hasVariance:w,sphereRadius:W,gaussianCutoff:g})}`,J=[];J.push(f(e,{id:`${n}-validate`,operation:Q,variant:`validate`,bindings:[z(`positions`,b,`f32`),z(`predictors`,x,`f32`),z(`response`,S,`f32`),...K?[z(`rowMask`,K,`u32`)]:[],B(`rowValid`,k,`u32`)],invocationCount:a,declarations:G,body:`var isValid = true;
  ${K?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  ${U?`if (!(abs(positions[positionsOffset + 2u * index + 1u]) <= 90.0)) {
    isValid = false;
  }`:``}
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`})),R&&L&&J.push(f(e,{id:`${n}-grid-positions`,operation:Q,variant:`grid-positions`,bindings:[z(`positions`,b,`f32`),z(`rowValid`,k,`u32`),B(`gridPositions`,L,`f32`)],invocationCount:a,declarations:G,body:`let included = rowValid[rowValidOffset + index] != 0u;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] =
    select(invalid, positions[positionsOffset + index * 2u], included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] =
    select(invalid, positions[positionsOffset + index * 2u + 1u], included);`}),...R.nodes),J.push(f(e,{id:`${n}-ladder-init`,operation:Q,variant:`ladder-init`,bindings:[z(`params`,ee,`f32`),B(`ladders`,N,`f32`)],invocationCount:1,declarations:G,body:`ladders[laddersOffset] = 0.0;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[laddersOffset + 1u + slot] = params[paramsOffset + slot];
  }`}));let fe=[z(`positions`,b,`f32`),z(`predictors`,x,`f32`),z(`response`,S,`f32`),z(`rowValid`,k,`u32`),z(`params`,N,`f32`),...oe],Y=_&&F?Se(e,{id:n,operation:Q,rowCount:a,tileRows:Me,tileCount:C,candidateWidth:m,maximumIterations:p,tableRowCount:h,common:G,candidateDeclarations:ue,focalRowWGSL:q,hasGrid:!!u,candidateBindings:fe,rowValid:k,ladders:N,parameters:ee,candidateScratch:A,tileCandidates:j,candidateTable:M,searchState:F}):null;Y&&J.push(...Y.setupNodes);for(let t=0;t<te;t++){let r=t===0?``:_?`-search-${t}`:`-refine-${t}`,i=Y&&t>0?Y.gate:null;Y&&t>0?J.push(Y.createIterationNode(t)):t>0&&J.push(f(e,{id:`${n}-refine-${t}-ladder`,operation:Q,variant:`refine-ladder`,bindings:[B(`ladders`,N,`f32`),z(`table`,M,`f32`)],invocationCount:1,declarations:`${G}
${Be}`,body:`refineLadder();`})),J.push(f(e,{id:`${n}${r}-candidates`,operation:Q,variant:`candidates`,...i?.getGate(0),bindings:[...fe,B(`candidateScratch`,A,`f32`)],invocationCount:a,declarations:ue,body:`${q}
  let candidateCount = getCandidateCount();
  if (candidateCount == 0u || rowValid[rowValidOffset + focal] == 0u) {
    // Excluded rows are skipped by the tile sums, so their scratch is never read.
    return;
  }
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  let isAdaptive = getIsAdaptive();
  if (isAdaptive) {
    selectNeighborDistances(focal, getMaximumK(candidateCount), &neighbors);
  }
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    // Marker in the hat slot: -1 singular local system, -2 invalid bandwidth for this row.
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + focal) * 2u;
    candidateScratch[slot] = 0.0;
    candidateScratch[slot + 1u] = -1.0;
    let bandwidth = getBandwidthForValue(getParameter(HEADER + candidate), isAdaptive, &neighbors);
    if (bandwidth <= 0.0) {
      candidateScratch[slot + 1u] = -2.0;
      continue;
    }
    let fit = fitLocation(focal, bandwidth);
    if (!fit.ok) {
      continue;
    }
    let residual = response[responseOffset + focal] - fit.beta[0];
    candidateScratch[slot] = residual * residual;
    candidateScratch[slot + 1u] = fit.hat;
  }`})),J.push(f(e,{id:`${n}${r}-tile-candidates`,operation:Q,variant:`tile-candidates`,...i?.getGate(1),bindings:[z(`candidateScratch`,A,`f32`),z(`rowValid`,k,`u32`),B(`tileCandidates`,j,`f32`)],invocationCount:C*m,declarations:G,body:`let tile = index / LADDER;
  let candidate = index % LADDER;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + row) * 2u;
    let hat = candidateScratch[slot + 1u];
    if (hat < -1.5) {
      invalid = invalid + 1.0;
    } else if (hat < 0.0) {
      singular = singular + 1.0;
    } else {
      residualSum = residualSum + candidateScratch[slot];
      traceSum = traceSum + hat;
    }
  }
  let base = tileCandidatesOffset + index * TILE_CANDIDATE_STRIDE;
  tileCandidates[base] = residualSum;
  tileCandidates[base + 1u] = traceSum;
  tileCandidates[base + 2u] = singular;
  tileCandidates[base + 3u] = invalid;
  tileCandidates[base + 4u] = included;`})),J.push(f(e,{id:`${n}${r}-select`,operation:Q,variant:`select`,...i?.getGate(2),bindings:[z(`tileCandidates`,j,`f32`),z(`params`,N,`f32`),B(`table`,M,`f32`)],invocationCount:m,declarations:le,body:`let candidate = index;
  let passCursor = u32(params[paramsOffset]);
  let nan = getNaN();
  ${_?`// One table row per evaluation, in order: pass 0 fills rows 0 and 1, pass p the row p + 1.
  // Rows of unused candidates keep the NaN fill of the search initialization.
  if (candidate >= getCandidateCount()) {
    return;
  }
  let base = tableOffset + select(passCursor + 1u, candidate, passCursor == 0u) * TABLE_STRIDE;`:`let base = tableOffset + (passCursor * LADDER + candidate) * TABLE_STRIDE;
  if (candidate >= getCandidateCount()) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[base + field] = nan;
    }
    return;
  }`}
  var residualSum = 0.0;
  var traceSum = 0.0;
  var singular = 0.0;
  var invalid = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileBase = tileCandidatesOffset + (tile * LADDER + candidate) * TILE_CANDIDATE_STRIDE;
    residualSum = residualSum + tileCandidates[tileBase];
    traceSum = traceSum + tileCandidates[tileBase + 1u];
    singular = singular + tileCandidates[tileBase + 2u];
    invalid = invalid + tileCandidates[tileBase + 3u];
    included = included + tileCandidates[tileBase + 4u];
  }
  // AICc over the non-singular rows; a row with an invalid bandwidth voids the candidate.
  let count = included - singular;
  let denominator = count - 2.0 - traceSum;
  var score = nan;
  if (invalid == 0.0 && count > 0.0 && denominator > 0.0) {
    let variance = max(residualSum / count, MINIMUM_VARIANCE);
    let candidateScore = count * log(variance) + count * LN_TWO_PI + count * (count + traceSum) / denominator;
    if (isFiniteBits(candidateScore)) {
      score = candidateScore;
    }
  }
  table[base] = getParameter(HEADER + candidate);
  table[base + 1u] = score;
  table[base + 2u] = singular;
  table[base + 3u] = residualSum;
  table[base + 4u] = traceSum;
  table[base + 5u] = count;
  table[base + 6u] = included;`}))}if(Y&&J.push(Y.createIterationNode(te)),J.push(f(e,{id:`${n}-choose`,operation:Q,variant:`choose`,bindings:[z(`params`,ee,`f32`),z(`table`,M,`f32`),B(`rowValid`,k,`u32`),B(`selection`,P,`f32`),B(`summary`,I,`f32`),...F?[z(`state`,F,`f32`)]:[]],invocationCount:1,declarations:`${G}
${Ae()}
${F?Re:``}
const DEFAULT_SIGNIFICANCE_LEVEL: f32 = ${ce};`,body:`let nan = getNaN();
  // Fewest singular rows first, then the smallest AICc; ties go to the lowest index.
  var hasBest = false;
  var bestIndex = 0u;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < TABLE_ROWS; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestIndex = candidate;
      bestSingular = singular;
      bestScore = score;
    }
  }
  ${F?`// Golden section returns the winner of its last comparison (mgwr), when it is valid.
  let optimumRow = state[stateOffset + STATE_OPTIMUM_ROW];
  if (optimumRow >= 0.0) {
    let optimumBase = tableOffset + u32(optimumRow) * TABLE_STRIDE;
    let optimumScore = table[optimumBase + 1u];
    if (isFiniteBits(optimumScore)) {
      hasBest = true;
      bestIndex = u32(optimumRow);
      bestSingular = table[optimumBase + 2u];
      bestScore = optimumScore;
    }
  }`:``}
  let base = tableOffset + bestIndex * TABLE_STRIDE;
  let selectedValue = table[base];
  let residualSum = select(nan, table[base + 3u], hasBest);
  let traceSum = select(nan, table[base + 4u], hasBest);
  let count = table[base + 5u];
  let included = table[tableOffset + 6u];
  let sigmaSquared = select(nan, residualSum / (count - traceSum), hasBest);
  let requestedLevel = params[paramsOffset + 3u];
  let level = select(DEFAULT_SIGNIFICANCE_LEVEL, requestedLevel, requestedLevel > 0.0 && requestedLevel < 1.0);
  let adjustedAlpha = select(nan, level * f32(P) / traceSum, hasBest && traceSum > 0.0);
  let criticalT = select(nan, gwrStudentCriticalValue(adjustedAlpha, count - 1.0), hasBest && traceSum > 0.0);
  let logLikelihood = select(
    nan,
    -0.5 * count * (LN_TWO_PI + log(max(residualSum / count, MINIMUM_VARIANCE)) + 1.0),
    hasBest
  );
  selection[selectionOffset] = f32(bestIndex);
  selection[selectionOffset + 1u] = selectedValue;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_INDEX] = bestIndex;
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE] = bitcast<u32>(selectedValue);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_KERNEL] = bitcast<u32>(params[paramsOffset]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_MODE] = bitcast<u32>(params[paramsOffset + 1u]);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED] = bitcast<u32>(sigmaSquared);
  rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T] = bitcast<u32>(criticalT);
  summary[summaryOffset] = residualSum;
  summary[summaryOffset + 1u] = traceSum;
  summary[summaryOffset + 2u] = select(nan, bestScore, hasBest);
  summary[summaryOffset + 3u] = nan;
  summary[summaryOffset + 4u] = select(included, count, hasBest);
  summary[summaryOffset + 5u] = select(0.0, 1.0, hasBest);
  summary[summaryOffset + 6u] = select(nan, bestSingular, hasBest);
  summary[summaryOffset + 7u] = sigmaSquared;
  summary[summaryOffset + 8u] = adjustedAlpha;
  summary[summaryOffset + 9u] = criticalT;
  summary[summaryOffset + 10u] = logLikelihood;
  summary[summaryOffset + 11u] = included;`})),J.push(f(e,{id:`${n}-fit`,operation:Q,variant:`fit`,bindings:[z(`positions`,b,`f32`),z(`predictors`,x,`f32`),z(`response`,S,`f32`),B(`rowValid`,k,`u32`),...oe,B(`coefficients`,v.coefficients,`f32`)],invocationCount:a,declarations:de,body:`${q}
  let nan = getNaN();
  let coefficientBase = coefficientsOffset + focal * P;
  let localBase = rowValidOffset + LOCALS_BASE + focal * LOCAL_STRIDE;
  var status = ${Z.SINGULAR}u;
  var localR2 = nan;
  var conditionNumber = nan;
  var fittedValue = nan;
  var residual = nan;
  var hat = nan;
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientBase + column] = nan;
    ${w?`rowValid[rowValidOffset + STANDARD_ERROR_BASE + focal * P + column] = bitcast<u32>(nan);`:``}
    ${T?`rowValid[rowValidOffset + T_VALUE_BASE + focal * P + column] = bitcast<u32>(nan);`:``}
  }
  if (rowValid[rowValidOffset + focal] == 0u) {
    status = ${Z.EXCLUDED}u;
  } else {
    let selectedValue = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SELECTED_VALUE]);
    var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
    let isAdaptive = getIsAdaptive();
    if (isAdaptive) {
      let maximumK = clamp(u32(max(floor(selectedValue + 0.5), 1.0)), 1u, MAXIMUM_NEIGHBORS);
      selectNeighborDistances(focal, maximumK, &neighbors);
    }
    let bandwidth = getBandwidthForValue(selectedValue, isAdaptive, &neighbors);
    if (bandwidth > 0.0) {
      let fit = fitLocation(focal, bandwidth);
      if (fit.ok) {
        status = ${Z.OK}u;
        var intercept = fit.beta[0];
        for (var column = 1u; column < P; column++) {
          let slope = fit.beta[column];
          coefficients[coefficientBase + column] = slope;
          intercept = intercept - slope * predictors[predictorsOffset + focal * PREDICTOR_COUNT + column - 1u];
        }
        coefficients[coefficientBase] = intercept;
        fittedValue = fit.beta[0];
        residual = response[responseOffset + focal] - fittedValue;
        hat = fit.hat;
        localR2 = getLocalR2(focal, bandwidth, fit);
        ${v.localConditionNumber?`conditionNumber = getLocalConditionNumber(focal, bandwidth);`:``}
        ${w?`let sigmaSquared = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_SIGMA_SQUARED]);
        for (var column = 0u; column < P; column++) {
          var standardError = nan;
          let variance = fit.variance[column] * sigmaSquared;
          if (isFiniteBits(variance) && variance >= 0.0) {
            standardError = sqrt(variance);
          }
          rowValid[rowValidOffset + STANDARD_ERROR_BASE + focal * P + column] = bitcast<u32>(standardError);
          ${T?`let coefficient = coefficients[coefficientBase + column];
          let tValue = select(nan, coefficient / standardError, isFiniteBits(standardError) && standardError > 0.0 && isFiniteBits(coefficient));
          rowValid[rowValidOffset + T_VALUE_BASE + focal * P + column] = bitcast<u32>(tValue);`:``}
        }`:``}
      }
    }
  }
  rowValid[localBase] = bitcast<u32>(localR2);
  rowValid[localBase + 1u] = bitcast<u32>(fittedValue);
  rowValid[localBase + 2u] = bitcast<u32>(residual);
  rowValid[localBase + 3u] = bitcast<u32>(hat);
  rowValid[localBase + 4u] = status;
  rowValid[localBase + 5u] = bitcast<u32>(conditionNumber);`})),u&&J.push(f(e,{id:`${n}-fit-excluded`,operation:Q,variant:`fit-excluded`,bindings:[B(`rowValid`,k,`u32`),B(`coefficients`,v.coefficients,`f32`)],invocationCount:a,declarations:G,body:`if (rowValid[rowValidOffset + index] != 0u) {
    return;
  }
  let nan = getNaN();
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientsOffset + index * P + column] = nan;
    ${w?`rowValid[rowValidOffset + STANDARD_ERROR_BASE + index * P + column] = bitcast<u32>(nan);`:``}
    ${T?`rowValid[rowValidOffset + T_VALUE_BASE + index * P + column] = bitcast<u32>(nan);`:``}
  }
  let localBase = rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE;
  for (var word = 0u; word < LOCAL_STRIDE; word++) {
    rowValid[localBase + word] = bitcast<u32>(nan);
  }
  rowValid[localBase + 4u] = ${Z.EXCLUDED}u;`})),J.push(f(e,{id:`${n}-tile-statistics`,operation:Q,variant:`tile-statistics`,bindings:[z(`response`,S,`f32`),z(`rowValid`,k,`u32`),B(`tileStatistics`,ne,`f32`)],invocationCount:C,declarations:G,body:`let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var count = 0.0;
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + row * LOCAL_STRIDE + 4u] == ${Z.OK}u) {
      count = count + 1.0;
      sum = sum + response[responseOffset + row];
    }
  }
  let mean = sum / max(count, 1.0);
  var m2 = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + LOCALS_BASE + row * LOCAL_STRIDE + 4u] == ${Z.OK}u) {
      let deviation = response[responseOffset + row] - mean;
      m2 = m2 + deviation * deviation;
    }
  }
  tileStatistics[tileStatisticsOffset + 3u * index] = count;
  tileStatistics[tileStatisticsOffset + 3u * index + 1u] = mean;
  tileStatistics[tileStatisticsOffset + 3u * index + 2u] = m2;`})),J.push(f(e,{id:`${n}-summarize`,operation:Q,variant:`summarize`,bindings:[z(`tileStatistics`,ne,`f32`),B(`summary`,I,`f32`)],invocationCount:1,declarations:G,body:`var count = 0.0;
  var mean = 0.0;
  var m2 = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let tileCountValue = tileStatistics[tileStatisticsOffset + 3u * tile];
    if (tileCountValue == 0.0) {
      continue;
    }
    let tileMean = tileStatistics[tileStatisticsOffset + 3u * tile + 1u];
    let delta = tileMean - mean;
    let total = count + tileCountValue;
    mean = mean + delta * tileCountValue / total;
    m2 = m2 + tileStatistics[tileStatisticsOffset + 3u * tile + 2u] + delta * delta * count * tileCountValue / total;
    count = total;
  }
  let hasValid = summary[summaryOffset + 5u] > 0.5;
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - summary[summaryOffset] / m2, hasValid && m2 > 0.0);`})),v.bandwidthScores||v.bandwidthSingularRowCounts){let t=[z(`table`,M,`f32`)];v.bandwidthScores&&t.push(B(`scores`,v.bandwidthScores,`f32`)),v.bandwidthSingularRowCounts&&t.push(B(`singularRowCounts`,v.bandwidthSingularRowCounts,`u32`)),J.push(f(e,{id:`${n}-publish-candidates`,operation:Q,variant:`publish-candidates`,bindings:t,invocationCount:c,declarations:G,body:`${_?`// The golden-section table can hold fewer rows than the ladder capacity.
  let isRow = index < TABLE_ROWS;
  let base = tableOffset + min(index, TABLE_ROWS - 1u) * TABLE_STRIDE;`:`let base = tableOffset + index * TABLE_STRIDE;`}
  ${v.bandwidthScores?`scores[scoresOffset + index] = ${_?`select(getNaN(), table[base + 1u], isRow)`:`table[base + 1u]`};`:``}
  ${v.bandwidthSingularRowCounts?`let singular = table[base + 2u];\n  singularRowCounts[singularRowCountsOffset + index] = select(0u, u32(singular), ${_?`isRow && `:``}singular >= 0.0 && singular < 4.0e9);`:``}`}))}let X=[[`localR2`,v.localR2,`f32`,0],[`fitted`,v.fitted,`f32`,1],[`residuals`,v.residuals,`f32`,2],[`hatDiagonal`,v.hatDiagonal,`f32`,3],[`localStatus`,v.localStatus,`u32`,4],[`localConditionNumber`,v.localConditionNumber,`f32`,5]].filter(([,e])=>e);if(X.length>0&&J.push(f(e,{id:`${n}-publish`,operation:Q,variant:`publish`,bindings:[z(`rowValid`,k,`u32`),...X.map(([e,t,n])=>B(e,t,n))],invocationCount:a,declarations:G,body:X.map(([e,,t,n])=>t===`f32`?`${e}[${e}Offset + index] = bitcast<f32>(rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${n}u]);`:`${e}[${e}Offset + index] = rowValid[rowValidOffset + LOCALS_BASE + index * LOCAL_STRIDE + ${n}u];`).join(`
  `)})),w){let t=[z(`rowValid`,k,`u32`)],r=[];v.standardErrors&&(t.push(B(`standardErrors`,v.standardErrors,`f32`)),r.push(`standardErrors[standardErrorsOffset + index] = bitcast<f32>(rowValid[rowValidOffset + STANDARD_ERROR_BASE + index]);`)),v.tValues&&(t.push(B(`tValues`,v.tValues,`f32`)),r.push(`tValues[tValuesOffset + index] = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);`)),v.significant&&(t.push(B(`significant`,v.significant,`u32`)),r.push(`let tValue = bitcast<f32>(rowValid[rowValidOffset + T_VALUE_BASE + index]);
  let criticalT = bitcast<f32>(rowValid[rowValidOffset + TAIL_BASE + TAIL_CRITICAL_T]);
  // NaN t values and a NaN critical value are never significant.
  significant[significantOffset + index] =
    select(0u, 1u, isFiniteBits(tValue) && isFiniteBits(criticalT) && abs(tValue) > criticalT);`)),J.push(f(e,{id:`${n}-publish-inference`,operation:Q,variant:`publish-inference`,bindings:t,invocationCount:a*E,declarations:G,body:r.join(`
  `)}))}return J}},Re=Object.entries(ve).map(([e,t])=>`const STATE_${e}: u32 = ${t}u;`).join(`
`),ze=`
${m}

${h}
`,Be=`
fn refineLadder() {
  let nan = getNaN();
  // Advance the GPU pass cursor: this node writes the ladder of pass passCursor.
  let passCursor = u32(ladders[laddersOffset]) + 1u;
  ladders[laddersOffset] = f32(passCursor);
  let callerBase = laddersOffset + 1u;
  let isAdaptive = ladders[callerBase + 1u] > 0.5;
  var hasBest = false;
  var bestValue = 0.0;
  var bestSingular = 0.0;
  var bestScore = 0.0;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    let base = tableOffset + candidate * TABLE_STRIDE;
    let score = table[base + 1u];
    if (!isFiniteBits(score)) {
      continue;
    }
    let singular = table[base + 2u];
    if (!hasBest || singular < bestSingular || (singular == bestSingular && score < bestScore)) {
      hasBest = true;
      bestValue = table[base];
      bestSingular = singular;
      bestScore = score;
    }
  }
  // Adaptive ladders hold integer k, so values compare after rounding.
  if (isAdaptive) {
    bestValue = floor(bestValue + 0.5);
  }
  var lower = -SENTINEL;
  var upper = SENTINEL;
  for (var candidate = 0u; candidate < passCursor * LADDER; candidate++) {
    var value = table[tableOffset + candidate * TABLE_STRIDE];
    if (!isFiniteBits(value)) {
      continue;
    }
    if (isAdaptive) {
      value = floor(value + 0.5);
    }
    if (value < bestValue) {
      lower = max(lower, value);
    } else if (value > bestValue) {
      upper = min(upper, value);
    }
  }
  if (lower <= -SENTINEL) {
    lower = bestValue;
  }
  if (upper >= SENTINEL) {
    upper = bestValue;
  }
  let base = callerBase + passCursor * PARAMETER_LENGTH;
  for (var slot = 0u; slot < PARAMETER_LENGTH; slot++) {
    ladders[base + slot] = 0.0;
  }
  ladders[base] = ladders[callerBase];
  ladders[base + 1u] = ladders[callerBase + 1u];
  ladders[base + 3u] = ladders[callerBase + 3u];
  var count = 0u;
  if (hasBest) {
    if (isAdaptive) {
      let first = max(lower + 1.0, 2.0);
      let last = min(upper - 1.0, f32(MAXIMUM_NEIGHBORS));
      let isInside = first <= bestValue && bestValue <= last;
      let integers = max(last - first + 1.0 - select(0.0, 1.0, isInside), 0.0);
      if (integers > 0.0) {
        count = min(u32(integers), LADDER);
        for (var slot = 0u; slot < count; slot++) {
          var position = f32(slot);
          if (integers > f32(LADDER)) {
            position = floor((f32(slot) + 0.5) * integers / f32(LADDER));
          }
          var value = first + position;
          if (isInside && value >= bestValue) {
            value = value + 1.0;
          }
          ladders[base + HEADER + slot] = value;
        }
      }
    } else {
      let low = max(lower, 0.0);
      let span = upper - low;
      if (span > 1.0e-5 * max(abs(upper), 1.0e-30)) {
        count = LADDER;
        for (var slot = 0u; slot < LADDER; slot++) {
          ladders[base + HEADER + slot] = low + span * f32(slot + 1u) / f32(LADDER + 1u);
        }
      }
    }
  }
  ladders[base + 2u] = f32(count);
}
`;export{T as _,Z as a,v as b,be as c,q as d,ne as f,ee as g,D as h,Y as i,se as l,E as m,pe as n,_e as o,O as p,X as r,ye as s,Le as t,K as u,x as v,y as x,_ as y};