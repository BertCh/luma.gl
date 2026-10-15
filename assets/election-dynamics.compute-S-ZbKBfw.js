import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{t as e}from"./status-BwNARr9H.js";import"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,d as r,h as i,j as a,n as o,t as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as d,s as f,u as p}from"./wgsl-common-BNRVzgd8.js";import{c as m}from"./spatial-weights-D4bd9AQ6.js";import{n as h,t as g}from"./gpu-class-assignment-zYxwZS01.js";import{a as _,i as v,t as y}from"./gpu-class-breaks-DsbEaky8.js";import{t as b}from"./gpu-spatial-lag-Q5OutSKD.js";import{f as x}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";import{t as ee}from"./gpu-local-moran-CvNMe0Gu.js";import{c as te,s as ne}from"./index-Bjk18V2I.js";import{t as re}from"./graph-swap-41Ru7LVu.js";import{t as ie}from"./summary-reader-CsgWxENf.js";import{t as ae}from"./mode-kernels-Dxm2GWvI.js";import{n as S}from"./b4-format-Cgp0VQWI.js";import{a as oe,i as se,r as ce,t as le}from"./b13-polygon-raster-Bd9SkdY6.js";import{t as ue}from"./b13-views-C79lqbkO.js";import{n as de}from"./b4-weights-core-mAiwYc53.js";var C=`
fn getLogGammaValue(z: f32) -> f32 {
  var shifted = z;
  var product = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (shifted >= 8.0) {
      break;
    }
    product = product * shifted;
    shifted = shifted + 1.0;
  }
  let inverse = 1.0 / shifted;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332 + series - log(product);
}

fn getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32 {
  if (isNanBits(value) || !(degreesOfFreedom > 0.0)) {
    return getNaN();
  }
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = degreesOfFreedom * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= 500u; iteration++) {
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
  for (var iteration = 1u; iteration <= 500u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
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
    if (abs(delta - 1.0) < 2.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`,w=`GPUTransitionDiagnostics`,fe=64,T={Q:0,Q_P_VALUE:1,LR:2,LR_P_VALUE:3,DEGREES_OF_FREEDOM:4},E={SPARSE:0,EMPTY:1},D=class{id;props;conditionCount;chainCount;minimumTransitions;constructor(t){let n=t.id??`transition-diagnostics`;this.id=n,this.props=t,this.conditionCount=t.conditionCount??1,this.chainCount=this.conditionCount>1?this.conditionCount+1:1,this.minimumTransitions=t.minimumTransitions??5;let{classCount:o,output:s}=t;if(r(n,`classCount`,o,1,fe),r(n,`conditionCount`,this.conditionCount),r(n,`minimumTransitions`,this.minimumTransitions),a(t.counts,`${n} counts`),t.counts.length<this.conditionCount*o*o)throw Error(`${n} counts is too short`);if(s.homogeneity){if(l(s.homogeneity,[`float32`],`${n} output.homogeneity`),s.homogeneity.length<5)throw Error(`${n} output.homogeneity is too short`);if(this.conditionCount<2)throw Error(`${n} output.homogeneity needs conditionCount of at least 2`)}if(s.steadyStates&&(l(s.steadyStates,[`float32`],`${n} output.steadyStates`),s.steadyStates.length<this.chainCount*o))throw Error(`${n} output.steadyStates is too short`);if(s.stateSparsity&&(a(s.stateSparsity,`${n} output.stateSparsity`),s.stateSparsity.length<2))throw Error(`${n} output.stateSparsity is too short`);if(s.status&&(e(n,s.status),s.status.converged&&!s.steadyStates))throw Error(`${n} output.status.converged needs output.steadyStates`);let c=[s.homogeneity,s.steadyStates,s.stateSparsity,s.status?.converged];i(n,c,[t.counts]);let u=c.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(u).size!==u.length)throw Error(`${n} outputs must use separate buffers`)}getCommandNodes(e){let{id:t,props:r,conditionCount:i,chainCount:a,minimumTransitions:o}=this,{counts:s,classCount:l,output:f}=r,m=f.status?.converged;n(t,e,[s,f.homogeneity,f.steadyStates,f.stateSparsity,m]);let h=`const K: u32 = ${l}u;
const C: u32 = ${i}u;
const MINIMUM_TRANSITIONS: u32 = ${o}u;`,g=[];if((f.homogeneity||f.stateSparsity||m)&&g.push(u(e,{id:`${t}-summary`,operation:w,variant:`summary`,bindings:[{name:`counts`,view:s,type:`u32`,access:`read`},...f.homogeneity?[{name:`homogeneity`,view:f.homogeneity,type:`f32`,access:`read_write`}]:[],...f.stateSparsity?[{name:`stateSparsity`,view:f.stateSparsity,type:`u32`,access:`read_write`}]:[],...m?[{name:`converged`,view:m,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`${h}
${d}
${p}
${C}`,body:`${m?`converged[convergedOffset] = 1u;`:``}
  ${f.stateSparsity?`var sparse = 0u;
  var empty = 0u;
  for (var state = 0u; state < C * K; state++) {
    var stateTotal = 0u;
    for (var to = 0u; to < K; to++) {
      stateTotal += counts[countsOffset + state * K + to];
    }
    if (stateTotal == 0u) {
      empty += 1u;
    } else if (stateTotal < MINIMUM_TRANSITIONS) {
      sparse += 1u;
    }
  }
  stateSparsity[stateSparsityOffset + ${E.SPARSE}u] = sparse;
  stateSparsity[stateSparsityOffset + ${E.EMPTY}u] = empty;`:``}
  ${f.homogeneity?`var q = 0.0;
  var likelihoodRatio = 0.0;
  var degreesOfFreedom = 0.0;
  var pooledRow: array<u32, ${l}>;
  for (var origin = 0u; origin < K; origin++) {
    var pooledTotal = 0u;
    var nonZeroCells = 0u;
    for (var to = 0u; to < K; to++) {
      var pooled = 0u;
      for (var condition = 0u; condition < C; condition++) {
        pooled += counts[countsOffset + (condition * K + origin) * K + to];
      }
      pooledRow[to] = pooled;
      pooledTotal += pooled;
      if (pooled > 0u) {
        nonZeroCells += 1u;
      }
    }
    if (pooledTotal == 0u) {
      continue;
    }
    var regimes = 0u;
    for (var condition = 0u; condition < C; condition++) {
      var regimeTotal = 0u;
      for (var to = 0u; to < K; to++) {
        regimeTotal += counts[countsOffset + (condition * K + origin) * K + to];
      }
      if (regimeTotal == 0u) {
        continue;
      }
      regimes += 1u;
      for (var to = 0u; to < K; to++) {
        let observed = f32(counts[countsOffset + (condition * K + origin) * K + to]);
        let expected = f32(regimeTotal) * f32(pooledRow[to]) / f32(pooledTotal);
        if (pooledRow[to] > 0u) {
          q += (observed - expected) * (observed - expected) / expected;
        }
        if (observed > 0.0) {
          likelihoodRatio += observed * log(observed / expected);
        }
      }
    }
    degreesOfFreedom += f32(regimes - 1u) * f32(nonZeroCells - 1u);
  }
  likelihoodRatio = max(2.0 * likelihoodRatio, 0.0);
  let nan = getNaN();
  homogeneity[homogeneityOffset + ${T.Q}u] = q;
  homogeneity[homogeneityOffset + ${T.Q_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(q, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${T.LR}u] = likelihoodRatio;
  homogeneity[homogeneityOffset + ${T.LR_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(likelihoodRatio, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${T.DEGREES_OF_FREEDOM}u] = degreesOfFreedom;`:``}`})),f.steadyStates){let n=c(e,`${t}-chain-work`,`float32`,a*l*l);g.push(u(e,{id:`${t}-steady-states`,operation:w,variant:`steady-states`,bindings:[{name:`counts`,view:s,type:`u32`,access:`read`},{name:`steadyStates`,view:f.steadyStates,type:`f32`,access:`read_write`},{name:`chainWork`,view:n,type:`f32`,access:`read_write`},...m?[{name:`converged`,view:m,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:a,declarations:`${h}
${d}
// Private memory: 2 * K words (at most 512 bytes), far below the 8,192-byte core limit.
var<private> usedStates: array<u32, ${l}>;
var<private> solution: array<f32, ${l}>;

// Transition count of one chain: chain 0 is the pooled chain when there are conditions.
fn getChainCount(chain: u32, origin: u32, to: u32) -> u32 {
  if (C == 1u) {
    return counts[countsOffset + origin * K + to];
  }
  if (chain == 0u) {
    var pooled = 0u;
    for (var condition = 0u; condition < C; condition++) {
      pooled += counts[countsOffset + (condition * K + origin) * K + to];
    }
    return pooled;
  }
  return counts[countsOffset + ((chain - 1u) * K + origin) * K + to];
}`,body:`let chain = index;
  let workBase = chainWorkOffset + chain * K * K;
  let nan = getNaN();
  var activeCount = 0u;
  var solvable = true;
  for (var state = 0u; state < K; state++) {
    var outgoing = 0u;
    var incoming = 0u;
    for (var other = 0u; other < K; other++) {
      outgoing += getChainCount(chain, state, other);
      incoming += getChainCount(chain, other, state);
    }
    if (outgoing > 0u || incoming > 0u) {
      usedStates[activeCount] = state;
      activeCount += 1u;
      if (outgoing == 0u) {
        solvable = false;
      }
    }
  }
  if (activeCount == 0u) {
    // No transitions at all: nothing to solve, and not a failure of the solver.
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    return;
  }
  let m = activeCount;
  if (solvable) {
    for (var a = 0u; a < m; a++) {
      var rowTotal = 0u;
      for (var b = 0u; b < m; b++) {
        rowTotal += getChainCount(chain, usedStates[a], usedStates[b]);
      }
      for (var b = 0u; b < m; b++) {
        chainWork[workBase + a * m + b] = f32(getChainCount(chain, usedStates[a], usedStates[b])) / f32(rowTotal);
      }
    }
    // Grassmann-Taksar-Heyman: eliminate the last state repeatedly. The sum of the remaining
    // outgoing probabilities of a state is zero only when it cannot reach the lower states.
    for (var k = m - 1u; k >= 1u; k--) {
      var escape = 0.0;
      for (var j = 0u; j < k; j++) {
        escape += chainWork[workBase + k * m + j];
      }
      if (!(escape > 0.0)) {
        solvable = false;
        break;
      }
      for (var i = 0u; i < k; i++) {
        chainWork[workBase + i * m + k] = chainWork[workBase + i * m + k] / escape;
      }
      for (var i = 0u; i < k; i++) {
        for (var j = 0u; j < k; j++) {
          chainWork[workBase + i * m + j] = chainWork[workBase + i * m + j] + chainWork[workBase + i * m + k] * chainWork[workBase + k * m + j];
        }
      }
    }
  }
  if (solvable) {
    solution[0] = 1.0;
    var total = 1.0;
    for (var k = 1u; k < m; k++) {
      var weight = 0.0;
      for (var i = 0u; i < k; i++) {
        weight += solution[i] * chainWork[workBase + i * m + k];
      }
      solution[k] = weight;
      total += weight;
    }
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = 0.0;
    }
    for (var a = 0u; a < m; a++) {
      steadyStates[steadyStatesOffset + chain * K + usedStates[a]] = solution[a] / total;
    }
  } else {
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    ${m?`atomicMin(&converged[convergedOffset], 0u);`:``}
  }`}))}return g}},O=`GPUTransitionMatrix`,k=64,A=2048,j=16,M=256,pe=class{id;props;conditionCount;periodLag;rowCount;constructor(e){let t=e.id??`transition-matrix`;this.id=t,this.props=e,this.periodLag=e.periodLag??1,this.conditionCount=e.conditionCount??1;let{periodCount:n,classCount:o,output:s}=e;if(r(t,`periodLag`,this.periodLag),r(t,`periodCount`,n,this.periodLag+1),r(t,`classCount`,o,1,k),r(t,`conditionCount`,this.conditionCount),!e.conditionClasses&&this.conditionCount!==1)throw Error(`${t} conditionCount needs conditionClasses`);a(e.classes,`${t} classes`);let c=e.classes.length/n;if(!Number.isInteger(c)||c<1)throw Error(`${t} classes length must be a multiple of periodCount`);if(this.rowCount=c,e.conditionClasses&&(a(e.conditionClasses,`${t} conditionClasses`),e.conditionClasses.length!==c*n))throw Error(`${t} conditionClasses length must equal classes length`);if(e.mask&&(a(e.mask,`${t} mask`),e.mask.length!==c))throw Error(`${t} mask length must equal the row count`);let u=this.conditionCount*o*o;if(a(s.counts,`${t} output.counts`),s.counts.length<u)throw Error(`${t} output.counts is too short`);if(s.probabilities&&(l(s.probabilities,[`float32`],`${t} output.probabilities`),s.probabilities.length<u))throw Error(`${t} output.probabilities is too short`);if(s.rowTotals&&(a(s.rowTotals,`${t} output.rowTotals`),s.rowTotals.length<this.conditionCount*o))throw Error(`${t} output.rowTotals is too short`);if(s.ignored&&(a(s.ignored,`${t} output.ignored`),s.ignored.length<1))throw Error(`${t} output.ignored must hold one row`);let d=[s.counts,s.probabilities,s.rowTotals,s.ignored,s.homogeneity,s.steadyStates,s.stateSparsity,s.status?.converged];i(t,d,[e.classes,e.conditionClasses,e.mask]);let f=d.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(f).size!==f.length)throw Error(`${t} outputs must use separate buffers`);this.diagnostics=this.hasDiagnostics()?new D(this.getDiagnosticsProps()):void 0}diagnostics;hasDiagnostics(){let{output:e}=this.props;return!!(e.homogeneity||e.steadyStates||e.stateSparsity||e.status)}getDiagnosticsProps(){let{props:e,conditionCount:t,id:n}=this,{output:r}=e;return{id:`${n}-diagnostics`,counts:r.counts,classCount:e.classCount,conditionCount:t,minimumTransitions:e.minimumTransitions,output:{homogeneity:r.homogeneity,steadyStates:r.steadyStates,stateSparsity:r.stateSparsity,status:r.status}}}getCommandNodes(e){let{id:t,props:r,conditionCount:i,periodLag:a,rowCount:o}=this,{classes:s,conditionClasses:c,mask:l,output:d,periodCount:f,classCount:p}=r;n(t,e,[s,c,l,d.counts,d.probabilities,d.rowTotals,d.ignored,d.homogeneity,d.steadyStates,d.stateSparsity,d.status?.converged]);let m=i*p*p,h=i*p,g=`const ROWS: u32 = ${o}u;
const CLASSES: u32 = ${p}u;
const CONDITIONS: u32 = ${i}u;
const CELLS: u32 = ${m}u;`,_=o*(f-a),v=m+1<=A,y=[{name:`classes`,view:s,type:`u32`,access:`read`},...c?[{name:`conditions`,view:c,type:`u32`,access:`read`}]:[],...l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[],{name:`counts`,view:d.counts,type:`atomic<u32>`,access:`read_write`},...d.ignored?[{name:`ignored`,view:d.ignored,type:`atomic<u32>`,access:`read_write`}]:[]],b=()=>u(e,{id:`${t}-count`,operation:O,variant:`count-private`,bindings:y,invocationCount:Math.ceil(_/j),workgroupSize:M,guardIndex:!1,declarations:`${g}
const TRANSITIONS: u32 = ${_}u;
const ITEMS: u32 = ${j}u;
const TILE_SIZE: u32 = ${M}u;
// Slot CELLS counts the transitions that are not counted.
var<workgroup> tile: array<atomic<u32>, ${m+1}>;`,body:`for (var cell = localInvocationIndex; cell <= CELLS; cell += TILE_SIZE) {
    atomicStore(&tile[cell], 0u);
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * ITEMS;
  for (var item = 0u; item < ITEMS; item++) {
    let transition = tileStart + item * TILE_SIZE + localInvocationIndex;
    if (transition < TRANSITIONS) {
      let period = transition / ROWS;
      let row = transition - period * ROWS;
      let fromClass = classes[classesOffset + transition];
      let toClass = classes[classesOffset + transition + ${a}u * ROWS];
      let condition = ${c?`conditions[conditionsOffset + transition]`:`0u`};
      let selected = ${l?`mask[maskOffset + row] != 0u`:`true`};
      if (selected && fromClass < CLASSES && toClass < CLASSES && condition < CONDITIONS) {
        atomicAdd(&tile[(condition * CLASSES + fromClass) * CLASSES + toClass], 1u);
      } else {
        atomicAdd(&tile[CELLS], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var cell = localInvocationIndex; cell <= CELLS; cell += TILE_SIZE) {
    let value = atomicLoad(&tile[cell]);
    if (value > 0u) {
      if (cell < CELLS) {
        atomicAdd(&counts[countsOffset + cell], value);
      } else {
        ${d.ignored?`atomicAdd(&ignored[ignoredOffset], value);`:``}
      }
    }
  }`}),x=[u(e,{id:`${t}-clear`,operation:O,variant:`clear`,bindings:[{name:`counts`,view:d.counts,type:`u32`,access:`read_write`},...d.ignored?[{name:`ignored`,view:d.ignored,type:`u32`,access:`read_write`}]:[]],invocationCount:m,declarations:g,body:`counts[countsOffset + index] = 0u;
  ${d.ignored?`if (index == 0u) { ignored[ignoredOffset] = 0u; }`:``}`}),v?b():u(e,{id:`${t}-count`,operation:O,variant:`count`,bindings:[{name:`classes`,view:s,type:`u32`,access:`read`},...c?[{name:`conditions`,view:c,type:`u32`,access:`read`}]:[],...l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[],{name:`counts`,view:d.counts,type:`atomic<u32>`,access:`read_write`},...d.ignored?[{name:`ignored`,view:d.ignored,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:o*(f-a),declarations:g,body:`let period = index / ROWS;
  let row = index - period * ROWS;
  let fromClass = classes[classesOffset + index];
  let toClass = classes[classesOffset + index + ${a}u * ROWS];
  let condition = ${c?`conditions[conditionsOffset + index]`:`0u`};
  let selected = ${l?`mask[maskOffset + row] != 0u`:`true`};
  if (selected && fromClass < CLASSES && toClass < CLASSES && condition < CONDITIONS) {
    atomicAdd(&counts[countsOffset + (condition * CLASSES + fromClass) * CLASSES + toClass], 1u);
  } else {
    ${d.ignored?`atomicAdd(&ignored[ignoredOffset], 1u);`:``}
  }`})];return(d.probabilities||d.rowTotals)&&x.push(u(e,{id:`${t}-normalize`,operation:O,variant:`normalize`,bindings:[{name:`counts`,view:d.counts,type:`u32`,access:`read`},...d.probabilities?[{name:`probabilities`,view:d.probabilities,type:`f32`,access:`read_write`}]:[],...d.rowTotals?[{name:`rowTotals`,view:d.rowTotals,type:`u32`,access:`read_write`}]:[]],invocationCount:h,declarations:g,body:`var total = 0u;
  for (var to = 0u; to < CLASSES; to++) {
    total += counts[countsOffset + index * CLASSES + to];
  }
  for (var to = 0u; to < CLASSES; to++) {
    ${d.probabilities?`probabilities[probabilitiesOffset + index * CLASSES + to] = select(0.0, f32(counts[countsOffset + index * CLASSES + to]) / f32(total), total > 0u);`:``}
  }
  ${d.rowTotals?`rowTotals[rowTotalsOffset + index] = total;`:``}`})),this.diagnostics&&x.push(...this.diagnostics.getCommandNodes(e)),x}},N=64;function P(e){return Math.ceil(e/N)*N}function F(e,t,n,r,i=r){return e.createDataView(t.buffer,{format:t.format,length:r,byteOffset:t.byteOffset+n*i*t.rowByteLength})}function I(e,t){let{rows:n,periods:r}=t,i=P(n);return u(e,{id:t.id,operation:t.operation,variant:`pack-periods`,bindings:[{name:`contiguous`,view:t.source,type:`f32`,access:`read`},{name:`padded`,view:t.padded,type:`f32`,access:`read_write`}],invocationCount:n*r,body:`let period = index / ${n}u;
  padded[paddedOffset + period * ${i}u + (index - period * ${n}u)] = contiguous[contiguousOffset + index];`})}function L(e,t){let{rows:n,periods:r}=t,i=P(n);return u(e,{id:t.id,operation:t.operation,variant:`unpack-periods`,bindings:[{name:`padded`,view:t.padded,type:`f32`,access:`read`},{name:`contiguous`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:n*r,body:`let period = index / ${n}u;
  contiguous[contiguousOffset + index] = padded[paddedOffset + period * ${i}u + (index - period * ${n}u)];`})}var me=class{id;props;rowCount;constructor(e){let t=e.id??`spatial-markov`;this.id=t,this.props=e,this.rowCount=m(t,e.weights);let n=this.rowCount*e.periodCount;if(l(e.values,[`float32`],`${t} values`),a(e.classes,`${t} classes`),r(t,`periodCount`,e.periodCount,2),e.values.length!==n||e.classes.length!==n)throw Error(`${t} values and classes length must equal rowCount * periodCount`);if(e.mask&&(a(e.mask,`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the weights row count`);if(e.output.lagValues&&e.output.lagValues.length!==n)throw Error(`${t} output.lagValues length must equal rowCount * periodCount`);if(e.output.lagClasses&&e.output.lagClasses.length!==n)throw Error(`${t} output.lagClasses length must equal rowCount * periodCount`)}getInferenceMetadata(){return{alternative:`greater`,permutationCount:0,multipleTesting:`none`,testedRowsDefinition:`one homogeneity test per statistic (Q, LR) of the conditional transition matrices against the pooled matrix, over all counted transitions of unmasked rows with a valid class and lag class`}}getCommandNodes(e){let{id:t,props:r,rowCount:i}=this,{output:a,weights:o,periodCount:s,mask:l}=r;n(t,e,[r.values,r.classes,r.lagParameters,l,o.offsets,o.neighbors,o.weights,a.counts,a.probabilities,a.rowTotals,a.ignored,a.lagBreaks,a.lagClassCount,a.lagValues,a.lagClasses,a.homogeneity,a.steadyStates,a.stateSparsity,a.status?.converged]);let d=i*s,f=a.lagValues??c(e,`${t}-lag-values`,`float32`,d),p=a.lagClasses??c(e,`${t}-lag-classes`,`uint32`,d),m=P(i),h=c(e,`${t}-padded-values`,`float32`,m*s),_=c(e,`${t}-padded-lag-values`,`float32`,m*s),v=[I(e,{id:`${t}-pack-values`,operation:`GPUSpatialMarkov`,source:r.values,padded:h,rows:i,periods:s})];for(let n=0;n<s;n++)v.push(...new b({id:`${t}-lag-${n}`,values:F(e,h,n,i,m),weights:o,mask:l,normalize:r.normalize??!0,output:F(e,_,n,i,m)}).getCommandNodes(e));v.push(L(e,{id:`${t}-unpack-lag-values`,operation:`GPUSpatialMarkov`,padded:_,output:f,rows:i,periods:s}));let x;return l&&(x=c(e,`${t}-pooled-mask`,`uint32`,d),v.push(u(e,{id:`${t}-pooled-mask`,operation:`GPUSpatialMarkov`,variant:`pooled-mask`,bindings:[{name:`mask`,view:l,type:`u32`,access:`read`},{name:`pooled`,view:x,type:`u32`,access:`read_write`}],invocationCount:d,body:`pooled[pooledOffset + index] = mask[maskOffset + index % ${i}u];`}))),v.push(...new y({id:`${t}-lag-breaks`,values:f,mask:x,parameters:r.lagParameters,maximumClassCount:r.maximumLagClassCount,methods:r.lagMethods??[`quantile`],output:{breaks:a.lagBreaks,classCount:a.lagClassCount}}).getCommandNodes(e),...new g({id:`${t}-lag-classes`,values:f,breaks:a.lagBreaks,classCount:a.lagClassCount,mask:x,output:p}).getCommandNodes(e),...new pe({id:`${t}-transitions`,classes:r.classes,periodCount:s,classCount:r.classCount,periodLag:r.periodLag,conditionClasses:p,conditionCount:r.maximumLagClassCount,mask:l,minimumTransitions:r.minimumTransitions,output:{counts:a.counts,probabilities:a.probabilities,rowTotals:a.rowTotals,ignored:a.ignored,homogeneity:a.homogeneity,steadyStates:a.steadyStates,stateSparsity:a.stateSparsity,status:a.status}}).getCommandNodes(e)),v}},R=`GPULISAMarkov`,he=class{id;props;rowCount;gating;constructor(e){let t=e.id??`lisa-markov`;if(this.id=t,this.props=e,this.rowCount=m(t,e.weights),this.gating=e.gating??`none`,l(e.values,[`float32`],`${t} values`),l(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<8)throw Error(`${t} parameters is too short`);if(r(t,`periodLag`,e.periodLag??1),r(t,`periodCount`,e.periodCount,(e.periodLag??1)+1),e.values.length!==this.rowCount*e.periodCount)throw Error(`${t} values length must equal rowCount * periodCount`);if(e.mask&&(a(e.mask,`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the weights row count`);if(e.output.quadrants&&(a(e.output.quadrants,`${t} output.quadrants`),e.output.quadrants.length!==this.rowCount*e.periodCount))throw Error(`${t} output.quadrants length must equal rowCount * periodCount`)}getInferenceMetadata(e){if(x(e),this.gating!==`none`)return{alternative:`two-sided`,permutationCount:0,multipleTesting:`none`,significanceLevel:e.significanceLevel??.05,testedRowsDefinition:`per period, unmasked rows with a finite value and at least one neighbor among such rows, a nonzero centered value and a defined z-score; each period tested separately`}}getCommandNodes(e){let{id:t,props:r,rowCount:i}=this,{output:a,weights:o,periodCount:s,mask:l}=r;n(t,e,[r.values,r.parameters,l,o.offsets,o.neighbors,o.weights,a.counts,a.probabilities,a.rowTotals,a.ignored,a.quadrants,a.homogeneity,a.steadyStates,a.stateSparsity,a.status?.converged]);let d=i*s,p=P(i),m=p*s,g=a.quadrants??c(e,`${t}-quadrants`,`uint32`,d),_=c(e,`${t}-padded-values`,`float32`,m),v=c(e,`${t}-padded-z-scores`,`float32`,m),y=c(e,`${t}-padded-quadrants`,`uint32`,m),b=c(e,`${t}-padded-neighbor-counts`,`uint32`,m),x=[I(e,{id:`${t}-pack-values`,operation:R,source:r.values,padded:_,rows:i,periods:s})];for(let n=0;n<s;n++)x.push(...new ee({id:`${t}-moran-${n}`,weights:o,values:F(e,_,n,i,p),parameters:r.parameters,mask:l,output:{zScores:F(e,v,n,i,p),quadrants:F(e,y,n,i,p),neighborCounts:F(e,b,n,i,p)},quadrantGating:this.gating===`none`?`none`:`analytic`,minimumNeighbors:1}).getCommandNodes(e));return x.push(u(e,{id:`${t}-states`,operation:R,variant:`states`,bindings:[{name:`values`,view:r.values,type:`f32`,access:`read`},{name:`neighborCounts`,view:b,type:`u32`,access:`read`},{name:`moranQuadrants`,view:y,type:`u32`,access:`read`},{name:`quadrants`,view:g,type:`u32`,access:`read_write`}],invocationCount:d,declarations:f,body:`let period = index / ${i}u;
  let paddedIndex = period * ${p}u + (index - period * ${i}u);
  var state = moranQuadrants[moranQuadrantsOffset + paddedIndex];
  if (!isFiniteBits(values[valuesOffset + index]) || neighborCounts[neighborCountsOffset + paddedIndex] == 0u${this.gating===`none`?` || state == 0u`:``}) {
    state = ${h}u;
  }
  quadrants[quadrantsOffset + index] = state;`})),x.push(...new pe({id:`${t}-transitions`,classes:g,periodCount:s,classCount:5,periodLag:r.periodLag,mask:l,minimumTransitions:r.minimumTransitions,output:{counts:a.counts,probabilities:a.probabilities,rowTotals:a.rowTotals,ignored:a.ignored,homogeneity:a.homogeneity,steadyStates:a.steadyStates,stateSparsity:a.stateSparsity,status:a.status}}).getCommandNodes(e)),x}},z=[2e3,2004,2008,2012,2016,2020,2024],B=5,V=3,H=5,ge=4294967295,U=1.4,_e=[`quantile`,`equal-interval`,`natural-breaks`],ve=[`equal-interval`,`quantile`],W=[[178,24,43,245],[239,138,98,245],[205,205,215,245],[103,169,207,245],[33,102,172,245]],G=[[68,1,84,245],[59,82,139,245],[33,145,140,245],[94,201,98,245],[253,231,37,245]],ye=[[120,130,145,80],[215,48,39,250],[145,191,219,250],[49,54,149,250],[253,174,97,250]],K=[`Not significant`,`HH`,`LH`,`LL`,`HL`],q=[[84,39,136,250],[153,142,195,245],[225,225,225,120],[241,163,64,245],[179,88,6,250]];function J(e){return e===`turnout`?[G[0],G[2],G[4]]:[W[0],W[2],W[4]]}function be(e,t){let{vertices:n,ringOffsets:r,polygonRingOffsets:i,partFeature:a,featureCount:o}=e,s=new Map;for(let e=0;e+1<i.length;e++){let o=a[e];if(t[o])for(let t=i[e];t<i[e+1];t++)for(let e=r[t];e<r[t+1];e++){let t=(Math.round(n[e*2]*1e5)+126e5)*6e6+Math.round(n[e*2+1]*1e5),r=s.get(t);r?r[r.length-1]!==o&&!r.includes(o)&&r.push(o):s.set(t,[o])}}let c=Array.from({length:o},()=>new Set);for(let e of s.values())if(!(e.length<2))for(let t of e)for(let n of e)t!==n&&c[t].add(n);let l=new Uint32Array(o+1);for(let e=0;e<o;e++)l[e+1]=l[e]+c[e].size;let u=new Uint32Array(Math.max(1,l[o])),d=new Float32Array(Math.max(1,l[o]));for(let e=0;e<o;e++){let t=[...c[e]].sort((e,t)=>e-t);t.forEach((n,r)=>{u[l[e]+r]=n,d[l[e]+r]=1/t.length})}return{offsets:l,neighbors:u,weights:d,links:l[o]}}async function Y(e){let{device:n}=e,r=e.datasets.get(`us-counties`),i=e.datasets.get(`us-elections`),a=e.datasets.get(`us-states`),c=r.defaultOrigin,l=r.getDisplayProjection(c),u=ce(r),d=u.featureCount,f=oe(u,l,r.manifest.bbox,4096),p=(r.geojson?.features??[]).map(e=>{let t=e.properties;return`${t?.name??`?`}${t?.state?`, ${t.state}`:``}`}),m=a.geojson?le(a.geojson,l):new Float32Array(4),h=(e,t)=>i.column(`${e===`turnout`?`turnoutProxy`:e}${t}`),b=new s(n,`elections`),ee=b.track(new re),C=b.createBuffer(`cell-feature`,f.cellFeature),w=b.createBuffer(`states`,m),fe=b.createParameterBuffer(`lisa-parameters`,`float32`,8),D=!1,O=!0,k=null,A=null,j=null,M=0,N=0,P=new Map,F=e=>{let t=new Uint8Array(d);for(let n=0;n<d;n++){let r=1;for(let t=0;t<e&&r;t++){let e=z[t],i=h(`demTwoParty`,e)[n],a=h(`turnout`,e)[n];(!Number.isFinite(i)||!Number.isFinite(a))&&(r=0)}t[n]=r}return t};async function I(e,r){let i=e===`2020`?6:7,a=`${e}-${r}`,o=F(i),s=0;for(let e of o)s+=e;let c=be(u,o),l=d*i,f=new Uint32Array(l);for(let e=0;e<i;e++)for(let t=0;t<d;t++)f[e*d+t]=o[t];let p=(e,t)=>b.createBuffer(`${a}-${e}`,t),m=p(`row-mask`,Uint32Array.from(o)),h=p(`pooled-mask`,f),_=p(`values`,l*4),x=p(`offsets`,c.offsets),ee=p(`neighbors`,c.neighbors),te=p(`weights`,c.weights),ne=p(`breaks`,(B+1)*4),re=p(`class-count`,4),S=p(`classes`,l*4),oe=p(`counts`,B**2*4),se=p(`probabilities`,B**2*4),ce=p(`totals`,B*4),le=p(`ignored`,4),C=V*B**2,w=p(`spatial-counts`,C*4),T=p(`spatial-probabilities`,C*4),E=p(`spatial-totals`,V*B*4),O=p(`lag-breaks`,(V+1)*4),M=p(`lag-class-count`,4),N=p(`lag-classes`,l*4),P=p(`homogeneity`,20),I=p(`state-sparsity`,8),L=p(`lisa-counts`,H**2*4),R=p(`lisa-probabilities`,H**2*4),z=p(`lisa-totals`,H*4),ge=p(`quadrants`,l*4),U=p(`display`,(d+1)*4),W=b.createParameterBuffer(`${a}-breaks-parameters`,`float32`,v(B)),G=b.createParameterBuffer(`${a}-lag-parameters`,`float32`,v(V)),ye=b.createParameterBuffer(`${a}-display-parameters`,`uint32`,4),K=new t(n,{id:`elections-analysis-${a}`}),q=ue(K,a),J=q(`values`,_,`float32`,l),Y=q(`row-mask`,m,`uint32`,d),X={offsets:q(`offsets`,x,`uint32`,d+1),neighbors:q(`neighbors`,ee,`uint32`,c.neighbors.length),weights:q(`weights`,te,`float32`,c.weights.length),information:de(d,!0)},xe=q(`breaks`,ne,`float32`,B+1),Z=q(`class-count`,re,`uint32`,1),Q=q(`classes`,S,`uint32`,l),Ce=q(`pooled-mask`,h,`uint32`,l);K.add(new y({id:`${a}-breaks`,values:J,mask:Ce,parameters:W.importToGraph(K),maximumClassCount:B,methods:_e,output:{breaks:xe,classCount:Z}})),K.add(new g({id:`${a}-assign`,values:J,breaks:xe,classCount:Z,mask:Ce,output:Q})),K.add(new pe({id:`${a}-matrix`,classes:Q,periodCount:i,periodLag:r,classCount:B,mask:Y,output:{counts:q(`counts`,oe,`uint32`,B**2),probabilities:q(`probabilities`,se,`float32`,B**2),rowTotals:q(`totals`,ce,`uint32`,B),ignored:q(`ignored`,le,`uint32`,1)}})),K.add(new me({id:`${a}-spatial`,values:J,classes:Q,classCount:B,weights:X,periodCount:i,periodLag:r,maximumLagClassCount:V,lagParameters:G.importToGraph(K),lagMethods:ve,mask:Y,output:{counts:q(`spatial-counts`,w,`uint32`,C),probabilities:q(`spatial-probabilities`,T,`float32`,C),rowTotals:q(`spatial-totals`,E,`uint32`,V*B),lagBreaks:q(`lag-breaks`,O,`float32`,V+1),lagClassCount:q(`lag-class-count`,M,`uint32`,1),lagClasses:q(`lag-classes`,N,`uint32`,l),homogeneity:q(`homogeneity`,P,`float32`,5),stateSparsity:q(`state-sparsity`,I,`uint32`,2)}})),K.add(new he({id:`${a}-lisa`,values:J,weights:X,gating:`significance`,periodCount:i,periodLag:r,parameters:fe.importToGraph(K),mask:Y,output:{counts:q(`lisa-counts`,L,`uint32`,H**2),probabilities:q(`lisa-probabilities`,R,`float32`,H**2),rowTotals:q(`lisa-totals`,z,`uint32`,H),quadrants:q(`quadrants`,ge,`uint32`,l)}}));let we=b.track(await K.compileAsync()),Te=new t(n,{id:`elections-display-${a}`}),$=ue(Te,`${a}-d`);return ae(Te,{id:`${a}-display`,invocationCount:d+1,declarations:`const ROWS: u32 = ${d}u; const PERIODS: u32 = ${i}u; const LAG: u32 = ${r}u;`,bindings:[{name:`parameters`,view:ye.importToGraph(Te),type:`u32`,access:`read`},{name:`rowMask`,view:$(`row-mask`,m,`uint32`,d),type:`u32`,access:`read`},{name:`classes`,view:$(`classes`,S,`uint32`,l),type:`u32`,access:`read`},{name:`lagClasses`,view:$(`lag-classes`,N,`uint32`,l),type:`u32`,access:`read`},{name:`quadrants`,view:$(`quadrants`,ge,`uint32`,l),type:`u32`,access:`read`},{name:`shown`,view:$(`display`,U,`uint32`,d+1),type:`u32`,access:`read_write`}],body:`
  if (index >= ROWS || rowMask[rowMaskOffset + index] == 0u) {
    shown[shownOffset + index] = 0xffffffffu;
    return;
  }
  let period = min(parameters[parametersOffset], PERIODS - 1u);
  let mapView = parameters[parametersOffset + 1u];
  let source = period * ROWS + index;
  var value = classes[classesOffset + source];
  if (mapView == 1u) {
    value = lagClasses[lagClassesOffset + source];
  } else if (mapView == 2u) {
    value = quadrants[quadrantsOffset + source];
  } else if (mapView == 3u) {
    // Class move over the next LAG periods (the last LAG periods have no successor).
    if (period + LAG >= PERIODS) {
      value = 0xffffffffu;
    } else {
      let before = classes[classesOffset + source];
      let after = classes[classesOffset + (period + LAG) * ROWS + index];
      if (before == 0xffffffffu || after == 0xffffffffu) {
        value = 0xffffffffu;
      } else {
        value = u32(clamp(i32(after) - i32(before) + 2, 0, 4));
      }
    }
  }
  shown[shownOffset + index] = value;`}),{key:a,periods:i,lag:r,rows:d,analysis:we,display:b.track(await Te.compileAsync()),values:_,valid:o,breaksParameters:W,lagParameters:G,reader:new ie(b,`${a}-matrices`,[{buffer:oe,size:B**2*4},{buffer:se,size:B**2*4},{buffer:ne,size:(B+1)*4},{buffer:re,size:4},{buffer:O,size:(V+1)*4},{buffer:M,size:4},{buffer:w,size:C*4},{buffer:T,size:C*4},{buffer:L,size:H**2*4},{buffer:R,size:H**2*4},{buffer:le,size:4},{buffer:P,size:20},{buffer:I,size:8}],e=>{if(D||k?.key!==a)return;let t=0,n=n=>{let r=e.slice(t,t+n*4);return t+=n*4,r},r=e=>new Uint32Array(n(e)),i=e=>new Float32Array(n(e));A={counts:r(B**2),probabilities:i(B**2),classEdges:i(B+1),classCount:r(1)[0],lagEdges:i(V+1),lagClassCount:r(1)[0],spatialCounts:r(C),spatialProbabilities:i(C),lisaCounts:r(H**2),lisaProbabilities:i(H**2),ignored:r(1)[0],homogeneity:i(5),sparseStates:r(2)},Se()}),displayReader:new ie(b,`${a}-display`,[{buffer:U,size:(d+1)*4}],e=>{D||k?.key!==a||(j=new Uint32Array(e))}),displayParameters:ye,displayBuffer:U,links:c.links,validCount:s}}async function L(){let{through:t,periodLag:n}=e.options,r=`${t}-${n}`,i=P.get(r);return i||(i=await I(t,Number(n)),P.set(r,i)),i}function R(){if(!k)return;let{variable:t}=e.options,n=new Float32Array(k.rows*k.periods);for(let e=0;e<k.periods;e++){let r=h(t,z[e]);for(let t=0;t<k.rows;t++){let i=r[t];n[e*k.rows+t]=k.valid[t]&&Number.isFinite(i)?i:0}}k.values.write(n),Y(n),O=!0}function Y(t){if(!k)return;let n=e.options,r;if(n.lisaMoments===`pooled`){let e=t??new Float32Array,n=0,i=0;for(let t=0;t<k.periods;t++)for(let r=0;r<k.rows;r++)k.valid[r]&&(n++,i+=e[t*k.rows+r]??0);if(e.length>0&&n>1){let t=i/n,a=0;for(let n=0;n<k.periods;n++)for(let r=0;r<k.rows;r++)k.valid[r]&&(a+=((e[n*k.rows+r]??0)-t)**2);r={count:n,mean:t,variance:Math.max(a/n,1e-12)}}}fe.write(x({significanceLevel:n.lisaSignificance,fixedMoments:r})),O=!0}function X(){if(!k)return;let t=e.options;k.breaksParameters.write(_({method:t.classMethod,classCount:t.classCount},B)),k.lagParameters.write(_({method:t.lagMethod,classCount:t.lagClassCount},V)),O=!0}let xe=t=>(e.options.variable,`${(t*100).toFixed(0)}%`);function Se(){if(!A||!k)return;let t=A,n=e.options,r=Math.min(B,Math.max(1,t.classCount));e.setReadout(`classEdges`,Array.from(t.classEdges.subarray(0,r+1),xe).join(`  |  `)),e.setReadout(`lagEdges`,Array.from(t.lagEdges.subarray(0,Math.max(1,t.lagClassCount)+1),xe).join(`  |  `));let i=e=>e.reduce((e,t)=>e+t,0),a=(e,t,n=0)=>{let r=0,i=0;for(let a=0;a<t;a++)for(let o=0;o<t;o++){let s=e[n+a*t+o];i+=s,a===o&&(r+=s)}return i>0?r/i:NaN},s=e=>Number.isFinite(e)?`${(e*100).toFixed(1)}%`:`n/a`;e.setReadout(`transitions`,`${o(i(t.counts))} county moves (${o(t.ignored)} skipped)`),e.setReadout(`stay`,s(a(t.counts,B))),e.setReadout(`stayLag`,[0,1,2].map(e=>s(a(t.spatialCounts,B,e*B**2))).join(` / `)),e.setReadout(`stayLisa`,s(a(t.lisaCounts,H)));let c=T,l=t.homogeneity[c.DEGREES_OF_FREEDOM];e.setReadout(`homogeneity`,l>0?`Q ${t.homogeneity[c.Q].toFixed(1)} (p ${S(t.homogeneity[c.Q_P_VALUE])}), LR ${t.homogeneity[c.LR].toFixed(1)} (p ${S(t.homogeneity[c.LR_P_VALUE])}), ${l} degrees of freedom`:`not testable (fewer than two lag classes with moves)`);let u=E;e.setReadout(`sparseStates`,`${t.sparseStates[u.SPARSE]} sparse and ${t.sparseStates[u.EMPTY]} empty lag-conditional rows`);let d=t.counts,f=t.probabilities,p=0,m=Array.from({length:B},(e,t)=>`class ${t+1}`),h=r;n.matrix===`lisa`?(d=t.lisaCounts,f=t.lisaProbabilities,m=K,h=H):n.matrix!==`pooled`&&(p=Number(n.matrix.slice(3))*B**2,d=t.spatialCounts,f=t.spatialProbabilities);let g=B;for(let t=0;t<H;t++){if(t>=h){e.setReadout(`row${t}`,`-`);continue}let n=0,r=[];for(let e=0;e<h;e++)n+=d[p+t*g+e],r.push(f[p+t*g+e].toFixed(2));e.setReadout(`row${t}`,`${m[t]}: ${r.join(`  `)}   (n = ${o(n)})`)}}function Z(t){if(!k)return;M=Math.max(0,Math.min(t,k.periods-1));let n=k.lag,r=M+n<k.periods?z[M+n]:null;e.setReadout(`yearShown`,e.options.mapView===`move`&&r?`${z[M]} to ${r}`:String(z[M])),Q=!0}let Q=!0;function Ce(t){k=t,A=null,j=null,e.setReadout(`counties`,`${o(k.validCount)} of ${o(d)}`),e.setReadout(`links`,`${o(k.links)} queen links (${(k.links/Math.max(1,k.validCount)).toFixed(1)} per county)`),R(),X(),Y(we()),Z(e.options.year),e.requestLayers()}function we(){if(!k)return;let t=new Float32Array(k.rows*k.periods);for(let n=0;n<k.periods;n++){let r=h(e.options.variable,z[n]);for(let e=0;e<k.rows;e++)t[n*k.rows+e]=k.valid[e]&&Number.isFinite(r[e])?r[e]:0}return t}return Ce(await L()),{getCompiledGraphs:()=>k?[k.analysis,k.display]:[],setOption(t,n,r){t===`through`||t===`periodLag`?ee.request({build:L,commit:Ce}):t===`variable`?(R(),e.requestLayers()):t===`classMethod`||t===`classCount`||t===`lagMethod`||t===`lagClassCount`?X():t===`lisaSignificance`||t===`lisaMoments`?Y(we()):t===`year`||t===`mapView`?(Z(r.year),e.requestLayers()):t===`matrix`?Se():e.requestLayers()},onThemeChange(){e.requestLayers()},getTooltip({coordinate:t}){if(!k||!t)return null;let n=se(f,t[0],t[1]);if(n<0)return null;let r=[p[n]??`County ${n}`];if(!k.valid[n])return r.push(`Missing returns in this period range`),r.join(`
`);let i=z.slice(0,k.periods).map(t=>{let r=h(e.options.variable,t)[n];return`${t}: ${Number.isFinite(r)?`${(r*100).toFixed(1)}%`:`n/a`}`});r.push(e.options.variable===`turnout`?`Votes per resident`:e.options.variable===`demShare`?`Democratic share of all votes`:`Democratic two-party share`),r.push(i.join(`   `));let a=j?.[n];if(a!==void 0&&a!==ge){let t=e.options;r.push(t.mapView===`class`?`Class ${a+1} in ${z[M]}`:t.mapView===`lag`?`Neighbors in lag class ${a+1}`:t.mapView===`lisa`?`LISA: ${K[a]??a}`:[`Down 2+ classes`,`Down 1 class`,`Same class`,`Up 1 class`,`Up 2+ classes`][a]??``)}return r.join(`
`)},encode(t,n){if(!k)return;let r=e.options;if(r.play?n.timeSeconds-N>U&&(N=n.timeSeconds,Z((M+1)%k.periods)):M!==Math.min(r.year,k.periods-1)&&Z(r.year),(O||n.frameIndex<3)&&(k.analysis.encode(t,{parameters:void 0}),O=!1,k.reader.request(t),Q=!0),Q||n.frameIndex<3){let e=r.mapView===`class`?0:r.mapView===`lag`?1:r.mapView===`lisa`?2:3;k.displayParameters.write(Uint32Array.of(M,e,0,0)),k.display.encode(t,{parameters:void 0}),k.displayReader.request(t),Q=!1}k.reader.flush(t),k.displayReader.flush(t)},getLayers(){if(!k)return[];let t=e.options,n=e.theme()===`dark`,r=[c[0],c[1],0],i=t.mapView===`class`?t.variable===`turnout`?G:W:t.mapView===`lag`?J(t.variable):t.mapView===`lisa`?ye:q,a=[new ne({id:`elections-${t.mapView}`,coordinateOrigin:r,gridSize:[f.width,f.height],bounds:f.bounds,valueIndices:C,values:k.displayBuffer,valueFormat:`uint32`,colormap:`category`,palette:i,noDataColor:[0,0,0,0],opacity:t.opacity,color:[255,255,255,255],tessellation:64})];return t.showStates&&a.push(new te({id:`elections-states`,coordinateOrigin:r,segments:w,instanceCount:m.length/4,widthPixels:.9,colormap:`uniform`,color:n?[235,240,250,150]:[30,40,60,140]})),a},destroy(){D=!0;for(let e of P.values())e.reader.stop(),e.displayReader.stop();b.destroy()}}}export{Y as createElectionDynamics};