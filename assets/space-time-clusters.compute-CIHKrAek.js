import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as e}from"./src-CNCKFW2b.js";import{E as t,p as n,w as r}from"./analysis-frame-Blu0tYdw.js";import{N as i,_ as a,d as o,h as s,n as c,t as l,w as u,x as d,y as f}from"./resources-CwvhEOBe.js";import{c as p,o as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as g}from"./spatial-weights-D4bd9AQ6.js";import{t as _}from"./philox-random-CHwyk0dd.js";import{n as v,t as y}from"./gpu-neighbor-search-Fct8W8hB.js";import{t as b}from"./permutation-random-C7SSHCbB.js";import{t as x}from"./separation-wgsl-OfnvdChu.js";import{_ as S,a as ee,b as C,c as te,d as w,f as T,g as E,h as D,i as O,l as k,m as A,n as j,o as M,p as ne,r as N,s as P,t as F,u as I,v as L,y as R}from"./gpu-knox-test-BgRgyJ5E.js";import{a as re,c as z,s as B}from"./index-Bjk18V2I.js";import{n as V,t as ie}from"./gpu-temporal-reduction-uwWQ39vO.js";import{t as ae}from"./graph-swap-41Ru7LVu.js";import{t as oe}from"./summary-reader-CsgWxENf.js";import{t as se}from"./mode-kernels-Dxm2GWvI.js";import{a as ce,i as le,n as H,r as ue}from"./b13-polygon-raster-Bd9SkdY6.js";import{t as U}from"./b13-views-C79lqbkO.js";var de=2**24,W=99999,fe={high:0,low:1,both:2};function pe(e,t){if(!Number.isInteger(e)||e<1)throw Error(`scan statistic needs at least one zone`);let n=de,r=e*e<=n?e:Math.max(1,Math.floor(n/e)-1),i=Math.min(t??r,e);if(!Number.isInteger(i)||i<1)throw Error(`scan statistic maximumWindowZones must be a positive integer`);let a=i<e?i+1:e;if(a*e>n)throw Error(`scan statistic zones * (maximumWindowZones + 1) must be <= ${n}`);return{usableZones:i,listLength:a}}var me={circle:0,nearest:1};function he(e){let{seed:t,permutationCount:n}=e;if(!Number.isSafeInteger(t)||t<0)throw Error(`scan statistic seed must be a non-negative safe integer`);o(`scan statistic`,`permutationCount`,n,1,2**31-1);let r=e.maximumPopulationFraction??.5;if(!(r>0&&r<=1))throw Error(`scan statistic maximumPopulationFraction must be in (0, 1]`);let i=new Uint32Array(8);return i[0]=t%2**32,i[1]=Math.floor(t/2**32),i[2]=n,i[3]=new Uint32Array(new Float32Array([r]).buffer)[0],i[4]=Math.max(1,Math.floor(e.maximumWindowZones??2**31)),i[5]=e.maximumTimeBuckets===void 0?0:Math.max(1,Math.floor(e.maximumTimeBuckets)),i[6]=me[e.windowShape??`circle`],i[7]=fe[e.rateDirection??`high`],i}var G=`GPUSpatialScanStatistic`,ge=Object.freeze({...n,measure:`distance`}),_e=256,ve=1024,ye=_.spatialScanStatistic.id,be=256,xe=8,Se=256,Ce=2**25,we=4096,Te=1024,Ee=5,De=class{id;props;spatialContext;zones;timeBuckets;model;usableZones;listLength;maximumClusters;batchReplicates;batchCount;constructor(e){let n=e.id??`scan-statistic`;if(this.id=n,this.props=e,this.spatialContext=t(n,e.spatialContext,ge),f(e.positions,[`float32x2`],`${n} positions`),f(e.cases,[`uint32`],`${n} cases`),this.model=e.model??`poisson`,![`poisson`,`bernoulli`,`space-time-permutation`].includes(this.model))throw Error(`${n} model must be 'poisson', 'bernoulli' or 'space-time-permutation'`);if(this.model===`poisson`){if(!e.baseline)throw Error(`${n} the poisson model needs a baseline`);f(e.baseline,[`float32`],`${n} baseline`)}if(this.model===`bernoulli`){if(!e.controls)throw Error(`${n} the bernoulli model needs controls`);f(e.controls,[`uint32`],`${n} controls`)}f(e.parameters,[`uint32`],`${n} parameters`);let{output:r}=e;f(r.clusterIndices,[`uint32`],`${n} output.clusterIndices`),f(r.clusterStatistics,[`float32`],`${n} output.clusterStatistics`),f(r.statistics,[`float32`],`${n} output.statistics`),f(r.summary,[`uint32`],`${n} output.summary`),this.zones=e.positions.length,this.timeBuckets=e.timeBuckets??1,this.maximumClusters=e.maximumClusters??8;let{maximumPermutations:i}=e;if(e.batchReplicates!==void 0&&(!Number.isInteger(e.batchReplicates)||e.batchReplicates<1))throw Error(`${n} batchReplicates must be a positive integer`);if(this.zones<1)throw Error(`${n} needs at least one zone`);o(n,`timeBuckets`,this.timeBuckets,1,32);let a=pe(this.zones,e.maximumWindowZones);this.usableZones=a.usableZones,this.listLength=a.listLength,o(n,`maximumClusters`,this.maximumClusters,1,64),o(n,`maximumPermutations`,i,1,W);let c=this.zones*this.timeBuckets;if(e.cases.length!==c||this.model===`poisson`&&e.baseline.length!==c||this.model===`bernoulli`&&e.controls.length!==c)throw Error(`${n} cases, baseline and controls need zones * timeBuckets rows`);let l=Math.floor(Ce/c)-1;if(l<1)throw Error(`${n} zones * timeBuckets must be <= ${Ce/2}`);if(this.batchReplicates=Math.min(i,we,l,e.batchReplicates??1/0),this.batchCount=Math.ceil(i/this.batchReplicates),this.batchCount>Te)throw Error(`${n} maximumPermutations needs ${this.batchCount} batches of ${this.batchReplicates} replicates, more than ${Te}`);if(e.parameters.length<8)throw Error(`${n} parameters needs 8 words`);if(r.clusterIndices.length<this.maximumClusters*4)throw Error(`${n} output.clusterIndices needs maximumClusters * 4 rows`);if(r.clusterStatistics.length<this.maximumClusters*8)throw Error(`${n} output.clusterStatistics needs maximumClusters * 8 rows`);if(r.statistics.length<i+1)throw Error(`${n} output.statistics needs maximumPermutations + 1 rows`);if(r.summary.length<4)throw Error(`${n} output.summary needs 4 rows`);if(r.zoneStatistics&&(f(r.zoneStatistics,[`float32`],`${n} output.zoneStatistics`),r.zoneStatistics.length<this.zones))throw Error(`${n} output.zoneStatistics needs one row per zone`);let u=[r.clusterIndices,r.clusterStatistics,r.statistics,r.summary,...r.zoneStatistics?[r.zoneStatistics]:[]];if(s(n,u,[e.positions,e.cases,...e.baseline?[e.baseline]:[],...e.controls?[e.controls]:[],e.parameters]),new Set(u.map(e=>e.buffer)).size!==u.length)throw Error(`${n} outputs must use separate buffers`)}getInferenceMetadata(e){he(e);let t=e.rateDirection??`high`;return{alternative:t===`high`?`greater`:t===`low`?`lesser`:`two-sided`,permutationCount:Math.min(e.permutationCount,this.props.maximumPermutations),seed:e.seed,multipleTesting:`none`,testedRowsDefinition:`every window: a center zone, its nearest zones within maximumPopulationFraction of the population, and a run of consecutive time buckets within maximumTimeBuckets; p-values are Monte Carlo maxima over all windows`}}getCommandNodes(t){let{id:n,props:i,zones:o,timeBuckets:s,listLength:c,usableZones:l,maximumClusters:f,model:g}=this,_=x({context:this.spatialContext,geographic:this.spatialContext.coordinateSpace===`longitude-latitude`,sphereRadius:r(this.spatialContext)}),{positions:v,cases:y,parameters:S,maximumPermutations:ee}=i,{clusterIndices:C,clusterStatistics:te,statistics:w,summary:T,zoneStatistics:E}=i.output;a(n,t,[v,y,...i.baseline?[i.baseline]:[],...i.controls?[i.controls]:[],S,C,te,w,T,...E?[E]:[]]);let D=o*s,O=Math.min(_e,D),k=Math.ceil(D/O),A=Math.min(_e,o),j=Math.ceil(o/A),{batchReplicates:M,batchCount:ne}=this,N=M+1,P=(e,r)=>u(t,`${n}-${e}`,`uint32`,r),F=(e,r)=>u(t,`${n}-${e}`,`float32`,r),I=P(`neighbors`,o*c),L=F(`distances`,o*c),R=P(`sort-keys`,o*c),re=P(`sort-zones`,o*c),z=P(`members`,o),B=g===`poisson`?i.baseline:F(`weights`,D),V=g===`bernoulli`?P(`population`,D):null,ie=g===`space-time-permutation`?P(`zone-totals`,o):null,ae=g===`space-time-permutation`?P(`bucket-totals`,s):null,oe=F(`block-baseline`,O),se=P(`block-cases`,O),ce=P(`block-weights`,O),le=P(`block-offsets`,O),H=P(`totals`,Ee),ue=P(`cumulative`,D),U=P(`counts`,N*D),de=Math.ceil(o/Se),W=P(`center-best`,o*xe+o*2),fe=P(`alive`,o),pe=P(`partials`,A*2),me=P(`state`,8),he=[c,o,A],ge=he.map(p),Ce=he.map((e,n)=>d(G,Math.max(e,1),ge[n],t.device.limits.maxComputeWorkgroupsPerDimension)),we=u(t,`${n}-round-dispatch`,`uint32`,3*Ce.length,e.STORAGE|e.INDIRECT),Te=e=>({workgroupSize:ge[e],condition:{id:`${n}-round-gate-${e}`,source:`gpu`,mode:`indirect`,buffer:we.buffer,byteOffset:we.byteOffset+e*12},extraResources:[{buffer:we,usage:`indirect`}]}),De=Ce.map((e,t)=>`roundDispatch[roundDispatchOffset + ${3*t}u] = ${e.x}u;
    roundDispatch[roundDispatchOffset + ${3*t+1}u] = ${e.y}u;
    roundDispatch[roundDispatchOffset + ${3*t+2}u] = ${e.z}u;`).join(`
    `),Oe=Ce.map((e,t)=>`roundDispatch[roundDispatchOffset + ${3*t}u] = 0u;`).join(` `),K=g===`poisson`?M*ve:M,q=m(t,{id:`${n}-batch-dispatch`,operation:G,shapes:[[M*D],[K],[M*de*Se,Se]]}),J=`
const ZONES: u32 = ${o}u;
const LIST: u32 = ${c}u;
const USABLE: u32 = ${l}u;
const BUCKETS: u32 = ${s}u;
const CELLS: u32 = ${D}u;
const BLOCKS: u32 = ${O}u;
const CELLS_PER_BLOCK: u32 = ${k}u;
const ZONE_BLOCKS: u32 = ${A}u;
const ZONES_PER_BLOCK: u32 = ${j}u;
const MAXIMUM_PERMUTATIONS: u32 = ${ee}u;
const BATCH_REPLICATES: u32 = ${M}u;
const CENTER_WORDS: u32 = ${xe}u;
const WINDOW_INFO_START: u32 = ${o*xe}u;
const NO_CENTER: u32 = 0xffffffffu;
// Weight of one cell in the integer cumulative table: its baseline share of 2^31, rounded down.
fn getCellWeight(cellBaseline: f32, totalBaseline: f32) -> u32 {
  if (!(totalBaseline > 0.0) || !(cellBaseline > 0.0)) {
    return 0u;
  }
  return u32(floor(cellBaseline / totalBaseline * 2147483648.0));
}`,Y=`${J}
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutations() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readMaximumFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
fn readMaximumZones() -> u32 {
  return clamp(parameters[parametersOffset + 4u], 1u, USABLE);
}
// 0 selects the default, 50% of the study period.
fn readMaximumBuckets() -> u32 {
  let requested = parameters[parametersOffset + 5u];
  if (requested == 0u) {
    return max(1u, BUCKETS / 2u);
  }
  return clamp(requested, 1u, BUCKETS);
}
fn readWindowShape() -> u32 {
  return parameters[parametersOffset + 6u];
}
fn readRateDirection() -> u32 {
  return parameters[parametersOffset + 7u];
}`,ke=`
// (1 + d) ln(1 + d) - d for d >= -1. Below |d| = 1/4 the Taylor series
// d^2 * sum_k>=2 (-d)^(k - 2) / (k (k - 1)), terms to k = 14 (truncation below 1e-9 relative),
// avoids the cancellation of the direct form.
fn getEntropyExcess(delta: f32) -> f32 {
  if (abs(delta) < 0.25) {
    var series = 1.0 / 182.0;
    series = -1.0 / 156.0 + delta * series;
    series = 1.0 / 132.0 + delta * series;
    series = -1.0 / 110.0 + delta * series;
    series = 1.0 / 90.0 + delta * series;
    series = -1.0 / 72.0 + delta * series;
    series = 1.0 / 56.0 + delta * series;
    series = -1.0 / 42.0 + delta * series;
    series = 1.0 / 30.0 + delta * series;
    series = -1.0 / 20.0 + delta * series;
    series = 1.0 / 12.0 + delta * series;
    series = -1.0 / 6.0 + delta * series;
    series = 1.0 / 2.0 + delta * series;
    return delta * delta * series;
  }
  return (1.0 + delta) * log(1.0 + delta) - delta;
}
// Kulldorff term o ln(o/e) minus its linear part o - e, which sums to 0 over the cells of a full
// partition of the cases, as e * ((1 + d) ln(1 + d) - d) with d = (o - e) / e: near o = e the
// LLR no longer comes from the difference of two large logarithm terms.
fn getObservedTerm(observed: f32, expected: f32) -> f32 {
  if (!(expected > 0.0)) {
    return 0.0;
  }
  if (!(observed > 0.0)) {
    return expected;
  }
  return expected * getEntropyExcess((observed - expected) / expected);
}
fn isDirected(observed: f32, expected: f32, direction: u32) -> bool {
  if (direction == 0u) {
    return observed > expected;
  }
  if (direction == 1u) {
    return observed < expected;
  }
  return observed != expected;
}`,Ae=g===`bernoulli`?`${ke}
// Kulldorff (1997) Bernoulli LLR as the sum of o ln(o/e) over the 2x2 table of
// (cases, controls) x (inside, outside), with e = row total * column total / N.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let outside = totalWeight - windowWeight;
  if (!(windowWeight > 0.0) || !(outside > 0.0)) {
    return 0.0;
  }
  let expectedInside = windowWeight * totalCases / totalWeight;
  if (!isDirected(observed, expectedInside, direction)) {
    return 0.0;
  }
  let totalControls = totalWeight - totalCases;
  let ratio = getObservedTerm(observed, expectedInside) +
    getObservedTerm(max(windowWeight - observed, 0.0), windowWeight * totalControls / totalWeight) +
    getObservedTerm(totalCases - observed, outside * totalCases / totalWeight) +
    getObservedTerm(max(totalControls - windowWeight + observed, 0.0), outside * totalControls / totalWeight);
  return max(ratio, 0.0);
}`:`${ke}
// Poisson LLR c ln(c/e) + (C - c) ln((C - c)/(C - e)); the space-time permutation model passes the
// window's e = (zone totals * bucket totals / C) as the window weight of a total weight C.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let expected = windowWeight * totalCases / totalWeight;
  if (!(expected > 0.0) || !isDirected(observed, expected, direction)) {
    return 0.0;
  }
  // A window holding the whole expectation leaves no outside cell; its LLR is never positive.
  if (!(totalCases - expected > 0.0)) {
    return 0.0;
  }
  let ratio = getObservedTerm(observed, expected) +
    getObservedTerm(max(totalCases - observed, 0.0), totalCases - expected);
  return max(ratio, 0.0);
}`,je=e=>g===`bernoulli`?`population[populationOffset + ${e}]`:`getCellWeight(baseline[baselineOffset + ${e}], totalBaseline)`,Me=V?{name:`population`,view:V,type:`u32`,access:`read`}:{name:`baseline`,view:B,type:`f32`,access:`read`},X=[h(t,{id:`${n}-nearest`,operation:G,variant:`nearest`,bindings:[{name:`positions`,view:v,type:`f32`,access:`read`},{name:`neighbors`,view:I,type:`u32`,access:`read_write`},{name:`distances`,view:L,type:`u32`,access:`read_write`},{name:`sortKeys`,view:R,type:`u32`,access:`read_write`},{name:`sortZones`,view:re,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`${_}
const ZONES: u32 = ${o}u;
const LIST: u32 = ${c}u;
const RADIX_BINS: u32 = ${be}u;
// Separation keys are non-negative f32, so their bits order as u32 keys.
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}`,body:`var histogram: array<u32, RADIX_BINS>;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let row = index * LIST;
  var threshold = 0xffffffffu;
  var equalCount = LIST;
  if (LIST < ZONES) {
    var prefix = 0u;
    var prefixMask = 0u;
    var rank = LIST - 1u;
    for (var level = 0u; level < 4u; level++) {
      let shift = 24u - 8u * level;
      for (var bin = 0u; bin < RADIX_BINS; bin++) {
        histogram[bin] = 0u;
      }
      for (var zone = 0u; zone < ZONES; zone++) {
        let key = getZoneSeparationKey(zone, centerX, centerY);
        if ((key & prefixMask) == prefix) {
          let bin = (key >> shift) & 255u;
          histogram[bin] = histogram[bin] + 1u;
        }
      }
      var digit = 0u;
      loop {
        if (digit == RADIX_BINS - 1u || rank < histogram[digit]) {
          break;
        }
        rank -= histogram[digit];
        digit++;
      }
      prefix = prefix | (digit << shift);
      prefixMask = prefixMask | (255u << shift);
    }
    threshold = prefix;
    equalCount = rank + 1u;
  }
  var filled = 0u;
  var equalTaken = 0u;
  for (var zone = 0u; zone < ZONES; zone++) {
    if (filled >= LIST) {
      break;
    }
    let key = getZoneSeparationKey(zone, centerX, centerY);
    var take = key < threshold;
    if (!take && key == threshold && equalTaken < equalCount) {
      take = true;
      equalTaken++;
    }
    if (take) {
      distances[distancesOffset + row + filled] = key;
      neighbors[neighborsOffset + row + filled] = zone;
      filled++;
    }
  }
  for (var shift = 0u; shift < 32u; shift += 16u) {
    // Digit shift: the table rows to the sort scratch.
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (distances[distancesOffset + row + slot] >> shift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    var running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = distances[distancesOffset + row + slot];
      let bin = (key >> shift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      sortKeys[sortKeysOffset + row + destination] = key;
      sortZones[sortZonesOffset + row + destination] = neighbors[neighborsOffset + row + slot];
    }
    // Digit shift + 8: the sort scratch back to the table rows.
    let nextShift = shift + 8u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (sortKeys[sortKeysOffset + row + slot] >> nextShift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = sortKeys[sortKeysOffset + row + slot];
      let bin = (key >> nextShift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      distances[distancesOffset + row + destination] = key;
      neighbors[neighborsOffset + row + destination] = sortZones[sortZonesOffset + row + slot];
    }
  }`})];g===`bernoulli`&&X.push(h(t,{id:`${n}-population`,operation:G,variant:`population`,bindings:[{name:`cases`,view:y,type:`u32`,access:`read`},{name:`controls`,view:i.controls,type:`u32`,access:`read`},{name:`population`,view:V,type:`u32`,access:`read_write`},{name:`weights`,view:B,type:`f32`,access:`read_write`}],invocationCount:D,declarations:J,body:`let cellPopulation = cases[casesOffset + index] + controls[controlsOffset + index];
  population[populationOffset + index] = cellPopulation;
  weights[weightsOffset + index] = f32(cellPopulation);`})),g===`space-time-permutation`&&X.push(h(t,{id:`${n}-margins`,operation:G,variant:`margins`,bindings:[{name:`cases`,view:y,type:`u32`,access:`read`},{name:`zoneTotals`,view:ie,type:`u32`,access:`read_write`},{name:`bucketTotals`,view:ae,type:`u32`,access:`read_write`}],invocationCount:Math.max(o,s),declarations:J,body:`if (index < ZONES) {
    var zoneSum = 0u;
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      zoneSum += cases[casesOffset + index * BUCKETS + bucket];
    }
    zoneTotals[zoneTotalsOffset + index] = zoneSum;
  }
  if (index < BUCKETS) {
    var bucketSum = 0u;
    for (var zone = 0u; zone < ZONES; zone++) {
      bucketSum += cases[casesOffset + zone * BUCKETS + index];
    }
    bucketTotals[bucketTotalsOffset + index] = bucketSum;
  }`}),h(t,{id:`${n}-weights`,operation:G,variant:`weights`,bindings:[{name:`zoneTotals`,view:ie,type:`u32`,access:`read`},{name:`bucketTotals`,view:ae,type:`u32`,access:`read`},{name:`weights`,view:B,type:`f32`,access:`read_write`}],invocationCount:D,declarations:J,body:`let zone = index / BUCKETS;
  let bucket = index - zone * BUCKETS;
  var total = 0u;
  for (var other = 0u; other < BUCKETS; other++) {
    total += bucketTotals[bucketTotalsOffset + other];
  }
  weights[weightsOffset + index] = select(
    0.0,
    f32(zoneTotals[zoneTotalsOffset + zone]) * f32(bucketTotals[bucketTotalsOffset + bucket]) / f32(total),
    total > 0u
  );`})),X.push(h(t,{id:`${n}-block-totals`,operation:G,variant:`block-totals`,bindings:[{name:`baseline`,view:B,type:`f32`,access:`read`},{name:`cases`,view:y,type:`u32`,access:`read`},{name:`blockBaseline`,view:oe,type:`f32`,access:`read_write`},{name:`blockCases`,view:se,type:`u32`,access:`read_write`}],invocationCount:O,declarations:J,body:`var baselineSum = 0.0;
  var caseSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    baselineSum += baseline[baselineOffset + cell];
    caseSum += cases[casesOffset + cell];
  }
  blockBaseline[blockBaselineOffset + index] = baselineSum;
  blockCases[blockCasesOffset + index] = caseSum;`}),h(t,{id:`${n}-totals`,operation:G,variant:`totals`,bindings:[{name:`blockBaseline`,view:oe,type:`f32`,access:`read`},{name:`blockCases`,view:se,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,declarations:J,body:`var baselineSum = 0.0;
  var caseSum = 0u;
  for (var blockIndex = 0u; blockIndex < BLOCKS; blockIndex++) {
    baselineSum += blockBaseline[blockBaselineOffset + blockIndex];
    caseSum += blockCases[blockCasesOffset + blockIndex];
  }
  totals[totalsOffset] = caseSum;
  totals[totalsOffset + 1u] = bitcast<u32>(baselineSum);
  totals[totalsOffset + 2u] = 0u;
  totals[totalsOffset + 3u] = 0u;
  totals[totalsOffset + 4u] = 0u;`})),g!==`space-time-permutation`&&X.push(h(t,{id:`${n}-weight-sums`,operation:G,variant:`weight-sums`,bindings:[Me,{name:`totals`,view:H,type:`u32`,access:`read`},{name:`blockWeights`,view:ce,type:`u32`,access:`read_write`}],invocationCount:O,declarations:J,body:`let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  var weightSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    weightSum += ${je(`cell`)};
  }
  blockWeights[blockWeightsOffset + index] = weightSum;`}),h(t,{id:`${n}-weight-scan`,operation:G,variant:`weight-scan`,bindings:[{name:`blockWeights`,view:ce,type:`u32`,access:`read`},{name:`blockOffsets`,view:le,type:`u32`,access:`read_write`},{name:`totals`,view:H,type:`u32`,access:`read_write`}],invocationCount:1,declarations:J,body:`var running = 0u;
  for (var blockIndex = 0u; blockIndex < BLOCKS; blockIndex++) {
    blockOffsets[blockOffsetsOffset + blockIndex] = running;
    running += blockWeights[blockWeightsOffset + blockIndex];
  }
  totals[totalsOffset + 2u] = running;`}),h(t,{id:`${n}-cdf`,operation:G,variant:`cdf`,bindings:[Me,{name:`totals`,view:H,type:`u32`,access:`read`},{name:`blockOffsets`,view:le,type:`u32`,access:`read`},{name:`cumulative`,view:ue,type:`u32`,access:`read_write`}],invocationCount:O,declarations:J,body:`let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  var running = blockOffsets[blockOffsetsOffset + index];
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    running += ${je(`cell`)};
    cumulative[cumulativeOffset + cell] = running;
  }`})),X.push(h(t,{id:`${n}-init-counts`,operation:G,variant:`init-counts`,bindings:[{name:`cases`,view:y,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`},{name:`statistics`,view:w,type:`u32`,access:`read_write`}],invocationCount:Math.max(D,w.length),declarations:`${J}
const STATISTIC_WORDS: u32 = ${w.length}u;`,body:`if (index < CELLS) {
    counts[countsOffset + index] = cases[casesOffset + index];
  }
  if (index < STATISTIC_WORDS) {
    statistics[statisticsOffset + index] = 0u;
  }`}));let Ne;Ne=g===`poisson`?e=>h(t,{id:`${n}-scatter-${e}`,...q.getGate(1),operation:G,variant:`scatter`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`cumulative`,view:ue,type:`u32`,access:`read`},{name:`counts`,view:U,type:`atomic<u32>`,access:`read_write`}],invocationCount:K,declarations:`${b}
${Y}
const CASE_BLOCKS: u32 = ${ve}u;
const SCAN_STREAM: u32 = ${ye}u;`,body:`let localReplicate = index / CASE_BLOCKS;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let blockIndex = index - localReplicate * CASE_BLOCKS;
  let totalCases = totals[totalsOffset];
  let totalWeight = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalWeight == 0u) {
    return;
  }
  let chunk = (totalCases + CASE_BLOCKS - 1u) / CASE_BLOCKS;
  let first = blockIndex * chunk;
  let stop = min(first + chunk, totalCases);
  var stream = createPhiloxStream(readSeedKey(), replicate, blockIndex, SCAN_STREAM);
  for (var draw = first; draw < stop; draw++) {
    let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), totalWeight);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > drawTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    atomicAdd(&counts[countsOffset + (localReplicate + 1u) * CELLS + low], 1u);
  }`}):g===`bernoulli`?e=>h(t,{id:`${n}-draw-labels-${e}`,...q.getGate(1),operation:G,variant:`draw-labels`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`cumulative`,view:ue,type:`u32`,access:`read`},{name:`population`,view:V,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:K,declarations:`${b}
${Y}
const SCAN_STREAM: u32 = ${ye}u;`,body:`let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let totalCases = totals[totalsOffset];
  let totalPopulation = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalPopulation == 0u) {
    return;
  }
  let complement = totalCases > totalPopulation - totalCases;
  let drawCount = select(totalCases, totalPopulation - totalCases, complement);
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
  var drawn = 0u;
  while (drawn < drawCount) {
    let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), totalPopulation);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > drawTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let cellPopulation = population[populationOffset + low];
    let unchosen = cellPopulation - counts[row + low];
    if (getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), cellPopulation) < unchosen) {
      counts[row + low] += 1u;
      drawn++;
    }
  }
  if (complement) {
    for (var cell = 0u; cell < CELLS; cell++) {
      counts[row + cell] = population[populationOffset + cell] - counts[row + cell];
    }
  }`}):e=>h(t,{id:`${n}-draw-times-${e}`,...q.getGate(1),operation:G,variant:`draw-times`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`zoneTotals`,view:ie,type:`u32`,access:`read`},{name:`bucketTotals`,view:ae,type:`u32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:K,declarations:`${b}
${Y}
const SCAN_STREAM: u32 = ${ye}u;`,body:`let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  if (replicate >= readPermutations()) {
    return;
  }
  var remaining: array<u32, BUCKETS>;
  for (var bucket = 0u; bucket < BUCKETS; bucket++) {
    remaining[bucket] = bucketTotals[bucketTotalsOffset + bucket];
  }
  var left = totals[totalsOffset];
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
  for (var zone = 0u; zone < ZONES; zone++) {
    let zoneCases = zoneTotals[zoneTotalsOffset + zone];
    for (var draw = 0u; draw < zoneCases; draw++) {
      let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), left);
      var accumulated = 0u;
      var chosen = BUCKETS - 1u;
      for (var bucket = 0u; bucket < BUCKETS; bucket++) {
        accumulated += remaining[bucket];
        if (drawTarget < accumulated) {
          chosen = bucket;
          break;
        }
      }
      counts[row + zone * BUCKETS + chosen] += 1u;
      remaining[chosen] -= 1u;
      left -= 1u;
    }
  }`}),X.push(h(t,{id:`${n}-window-limit`,operation:G,variant:`window-limit`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`neighbors`,view:I,type:`u32`,access:`read`},{name:`baseline`,view:B,type:`f32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read_write`}],invocationCount:o,declarations:Y,body:`let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let limit = readMaximumFraction() * totalBaseline;
  let maximumZones = readMaximumZones();
  var populated = 0.0;
  var slots = 0u;
  var exceeded = false;
  for (var slot = 0u; slot < maximumZones; slot++) {
    let zone = neighbors[neighborsOffset + index * LIST + slot];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    if (populated > limit) {
      exceeded = true;
      break;
    }
    slots = slot + 1u;
  }
  var truncated = 0u;
  if (!exceeded && USABLE < ZONES && slots == USABLE && maximumZones == USABLE) {
    let zone = neighbors[neighborsOffset + index * LIST + USABLE];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    truncated = select(0u, 1u, populated <= limit);
  }
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u] = slots;
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u + 1u] = truncated;`}));let Z=(e,r)=>h(t,{id:e?`${n}-evaluate`:`${n}-batch-evaluate-${r}`,operation:G,variant:e?`evaluate`:`batch-evaluate`,...e?{}:q.getGate(2),bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`neighbors`,view:I,type:`u32`,access:`read`},{name:`distances`,view:L,type:`f32`,access:`read`},{name:`counts`,view:U,type:`u32`,access:`read`},{name:`baseline`,view:B,type:`f32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`statistics`,view:w,type:`atomic<u32>`,access:`read_write`},{name:`centerBest`,view:W,type:`u32`,access:e?`read_write`:`read`}],invocationCount:(e?1:M)*de*Se,workgroupSize:Se,guardIndex:!1,declarations:`${Y}
${Ae}
${e?_:``}
const EVALUATE_GROUPS: u32 = ${de}u;
const EVALUATE_SIZE: u32 = ${Se}u;
var<workgroup> maxima: array<f32, ${Se}>;`,body:`let group = index / EVALUATE_SIZE;
  let localReplicate = group / EVALUATE_GROUPS;
  let center = (group - localReplicate * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  ${e?`let slot = 0u;
  let statisticRow = 0u;
  let isLive = true;`:`let replicate = totals[totalsOffset + 4u] + localReplicate;
  let slot = localReplicate + 1u;
  let statisticRow = replicate + 1u;
  let isLive = replicate < readPermutations();`}
  var best = 0.0;
  if (isLive && center < ZONES) {
  let totalCases = f32(totals[totalsOffset]);
  let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let maximumBuckets = readMaximumBuckets();
  let direction = readRateDirection();
  let nearestShape = readWindowShape() == 1u;
  let windowSlots = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u];
  let slotStart = slot * CELLS;
  var caseSums: array<u32, BUCKETS>;
  var baselineSums: array<f32, BUCKETS>;
  var bestZones = 0u;
  var bestFirst = 0u;
  var bestLast = 0u;
  var bestCases = 0.0;
  var bestExpected = 0.0;
  for (var slotIndex = 0u; slotIndex < windowSlots; slotIndex++) {
    let zone = neighbors[neighborsOffset + center * LIST + slotIndex];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      caseSums[bucket] += counts[countsOffset + slotStart + zone * BUCKETS + bucket];
      baselineSums[bucket] += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    // A circle never ends between two zones at the same distance.
    let separated = nearestShape || slotIndex + 1u >= ZONES ||
      distances[distancesOffset + center * LIST + slotIndex] < distances[distancesOffset + center * LIST + slotIndex + 1u];
    if (!separated) {
      continue;
    }
    for (var first = 0u; first < BUCKETS; first++) {
      var windowCases = 0u;
      var windowBaseline = 0.0;
      for (var last = first; last < min(first + maximumBuckets, BUCKETS); last++) {
        windowCases += caseSums[last];
        windowBaseline += baselineSums[last];
        let ratio = getLogLikelihoodRatio(f32(windowCases), windowBaseline, totalCases, totalBaseline, direction);
        if (ratio > best) {
          best = ratio;
          bestZones = slotIndex + 1u;
          bestFirst = first;
          bestLast = last;
          bestCases = f32(windowCases);
          bestExpected = windowBaseline * totalCases / totalBaseline;
        }
      }
    }
  }
  ${e?`  if (slot == 0u) {
    let record = centerBestOffset + center * CENTER_WORDS;
    centerBest[record] = bitcast<u32>(best);
    centerBest[record + 1u] = bestZones;
    centerBest[record + 2u] = bestFirst;
    centerBest[record + 3u] = bestLast;
    centerBest[record + 4u] = bitcast<u32>(bestCases);
    centerBest[record + 5u] = bitcast<u32>(bestExpected);
    centerBest[record + 6u] = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u + 1u];
    // Squared radius in context units: decode the key of the farthest zone.
    centerBest[record + 7u] = select(
      0u,
      bitcast<u32>(getSquaredMeters(distances[distancesOffset + center * LIST + max(bestZones, 1u) - 1u])),
      bestZones > 0u
    );
  }
`:``}
  }
  // Workgroup max of the window maxima, then one atomicMax per workgroup: LLRs are
  // non-negative, so their f32 bits order as u32, and max is order independent (deterministic).
  maxima[localInvocationIndex] = best;
  workgroupBarrier();
  for (var stride = EVALUATE_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      maxima[localInvocationIndex] = max(maxima[localInvocationIndex], maxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && maxima[0] > 0.0) {
    atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(maxima[0]));
  }`});X.push(Z(!0,0));for(let e=0;e<ne;e++)X.push(h(t,{id:`${n}-batch-control-${e}`,operation:G,variant:`batch-control`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read_write`},q.binding],invocationCount:1,declarations:Y,body:`let base = totals[totalsOffset + 3u];
  totals[totalsOffset + 4u] = base;
  totals[totalsOffset + 3u] = base + BATCH_REPLICATES;
  if (base < readPermutations()) {
    ${q.armWGSL}
  } else {
    ${q.disarmWGSL}
  }`}),h(t,{id:`${n}-batch-clear-${e}`,operation:G,variant:`batch-clear`,...q.getGate(0),bindings:[{name:`counts`,view:U,type:`u32`,access:`read_write`}],invocationCount:M*D,declarations:J,body:`counts[countsOffset + CELLS + index] = 0u;`}),Ne(e),Z(!1,e));X.push(h(t,{id:`${n}-select-init`,operation:G,variant:`select-init`,bindings:[{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`alive`,view:fe,type:`u32`,access:`read_write`},{name:`members`,view:z,type:`u32`,access:`read_write`},{name:`state`,view:me,type:`u32`,access:`read_write`},{name:`clusterIndices`,view:C,type:`u32`,access:`read_write`},{name:`clusterStatistics`,view:te,type:`f32`,access:`read_write`},{name:`roundDispatch`,view:we,type:`u32`,access:`read_write`},...E?[{name:`zoneStatistics`,view:E,type:`f32`,access:`read_write`}]:[]],invocationCount:Math.max(o,f,8),declarations:`${J}
const CLUSTERS: u32 = ${f}u;`,body:`if (index < ZONES) {
    let best = bitcast<f32>(centerBest[centerBestOffset + index * CENTER_WORDS]);
    alive[aliveOffset + index] = select(0u, 1u, best > 0.0);
    members[membersOffset + index] = 0u;
    ${E?`zoneStatistics[zoneStatisticsOffset + index] = best;`:``}
  }
  if (index < 8u) {
    state[stateOffset + index] = 0u;
  }
  if (index == 0u) {
    ${De}
  }
  if (index < CLUSTERS) {
    for (var word = 0u; word < 4u; word++) {
      clusterIndices[clusterIndicesOffset + index * 4u + word] = 0u;
    }
    for (var word = 0u; word < 8u; word++) {
      clusterStatistics[clusterStatisticsOffset + index * 8u + word] = 0.0;
    }
  }`}));for(let e=0;e<f;e++)X.push(h(t,{id:`${n}-round-members-${e}`,operation:G,variant:`round-members`,...Te(0),bindings:[{name:`state`,view:me,type:`u32`,access:`read`},{name:`neighbors`,view:I,type:`u32`,access:`read`},{name:`members`,view:z,type:`u32`,access:`read_write`}],invocationCount:c,declarations:J,body:`if (state[stateOffset + 3u] == 0u || index >= state[stateOffset + 2u]) {
    return;
  }
  let zone = neighbors[neighborsOffset + state[stateOffset + 1u] * LIST + index];
  members[membersOffset + zone] = 1u;`}),h(t,{id:`${n}-round-mask-${e}`,operation:G,variant:`round-mask`,...Te(1),bindings:[{name:`state`,view:me,type:`u32`,access:`read`},{name:`neighbors`,view:I,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`members`,view:z,type:`u32`,access:`read`},{name:`alive`,view:fe,type:`u32`,access:`read_write`}],invocationCount:o,declarations:J,body:`if (state[stateOffset + 3u] == 0u || alive[aliveOffset + index] == 0u) {
    return;
  }
  let ownZones = centerBest[centerBestOffset + index * CENTER_WORDS + 1u];
  for (var own = 0u; own < ownZones; own++) {
    if (members[membersOffset + neighbors[neighborsOffset + index * LIST + own]] == 1u) {
      alive[aliveOffset + index] = 0u;
      return;
    }
  }`}),h(t,{id:`${n}-round-partials-${e}`,operation:G,variant:`round-partials`,...Te(2),bindings:[{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`alive`,view:fe,type:`u32`,access:`read`},{name:`partials`,view:pe,type:`u32`,access:`read_write`}],invocationCount:A,declarations:J,body:`var best = 0.0;
  var bestCenter = NO_CENTER;
  let first = index * ZONES_PER_BLOCK;
  for (var center = first; center < min(first + ZONES_PER_BLOCK, ZONES); center++) {
    let ratio = bitcast<f32>(centerBest[centerBestOffset + center * CENTER_WORDS]);
    if (alive[aliveOffset + center] == 1u && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  partials[partialsOffset + index * 2u] = bitcast<u32>(best);
  partials[partialsOffset + index * 2u + 1u] = bestCenter;`}),h(t,{id:`${n}-round-select-${e}`,operation:G,variant:`round-select`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`partials`,view:pe,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`roundDispatch`,view:we,type:`u32`,access:`read_write`},{name:`statistics`,view:w,type:`f32`,access:`read`},{name:`state`,view:me,type:`u32`,access:`read_write`},{name:`clusterIndices`,view:C,type:`u32`,access:`read_write`},{name:`clusterStatistics`,view:te,type:`f32`,access:`read_write`}],invocationCount:1,declarations:Y,body:`if (state[stateOffset + 4u] == 1u) {
    return;
  }
  var best = 0.0;
  var bestCenter = NO_CENTER;
  for (var blockIndex = 0u; blockIndex < ZONE_BLOCKS; blockIndex++) {
    let ratio = bitcast<f32>(partials[partialsOffset + blockIndex * 2u]);
    let center = partials[partialsOffset + blockIndex * 2u + 1u];
    if (center != NO_CENTER && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  if (bestCenter == NO_CENTER) {
    state[stateOffset + 3u] = 0u;
    state[stateOffset + 4u] = 1u;
    ${Oe}
    return;
  }
  let count = state[stateOffset];
  let record = centerBestOffset + bestCenter * CENTER_WORDS;
  let zoneCount = centerBest[record + 1u];
  let observed = bitcast<f32>(centerBest[record + 4u]);
  let expected = bitcast<f32>(centerBest[record + 5u]);
  var exceeding = 0u;
  let permutations = readPermutations();
  for (var replicate = 1u; replicate <= permutations; replicate++) {
    if (statistics[statisticsOffset + replicate] >= best) {
      exceeding++;
    }
  }
  clusterIndices[clusterIndicesOffset + count * 4u] = bestCenter;
  clusterIndices[clusterIndicesOffset + count * 4u + 1u] = zoneCount;
  clusterIndices[clusterIndicesOffset + count * 4u + 2u] = centerBest[record + 2u];
  clusterIndices[clusterIndicesOffset + count * 4u + 3u] = centerBest[record + 3u];
  let statisticStart = clusterStatisticsOffset + count * 8u;
  clusterStatistics[statisticStart] = best;
  clusterStatistics[statisticStart + 1u] = observed;
  clusterStatistics[statisticStart + 2u] = expected;
  clusterStatistics[statisticStart + 3u] = f32(exceeding + 1u) / f32(permutations + 1u);
  // Word 7 already holds squared distance in the context unit, not a separation key.
  clusterStatistics[statisticStart + 4u] = sqrt(bitcast<f32>(centerBest[record + 7u]));
  clusterStatistics[statisticStart + 5u] = observed / expected;
  clusterStatistics[statisticStart + 6u] = f32(centerBest[record + 6u]);
  state[stateOffset] = count + 1u;
  state[stateOffset + 1u] = bestCenter;
  state[stateOffset + 2u] = zoneCount;
  state[stateOffset + 3u] = 1u;`}));return X.push(h(t,{id:`${n}-summary`,operation:G,variant:`summary`,bindings:[{name:`parameters`,view:S,type:`u32`,access:`read`},{name:`state`,view:me,type:`u32`,access:`read`},{name:`totals`,view:H,type:`u32`,access:`read`},{name:`centerBest`,view:W,type:`u32`,access:`read`},{name:`summary`,view:T,type:`u32`,access:`read_write`}],invocationCount:1,declarations:Y,body:`summary[summaryOffset] = state[stateOffset];
  summary[summaryOffset + 1u] = totals[totalsOffset];
  summary[summaryOffset + 2u] = readPermutations();
  var overflow = 0u;
  for (var center = 0u; center < ZONES; center++) {
    overflow += centerBest[centerBestOffset + center * CENTER_WORDS + 6u];
  }
  summary[summaryOffset + 3u] = overflow;`})),X}},Oe=`GPUMantelTest`,K=3,q=4096,J=class{id;props;rows;constructor(e){let t=e.id??`mantel-test`;if(this.id=t,this.props=e,this.rows=g(t,e.pairs,`pairs`),!e.pairs.distances)throw Error(`${t} pairs.distances is required`);if(f(e.times,[`float32`],`${t} times`),f(e.parameters,[`uint32`],`${t} parameters`),f(e.output.statistics,[`float32`],`${t} output.statistics`),f(e.output.summary,[`float32`],`${t} output.summary`),this.rows<2)throw Error(`${t} needs at least two rows`);if(this.rows>O)throw Error(`${t} supports at most ${O} rows`);if(e.times.length!==this.rows)throw Error(`${t} times needs one row per pairs row`);if(e.parameters.length<4)throw Error(`${t} parameters needs 4 words`);if(o(t,`maximumPermutations`,e.maximumPermutations,1,N),e.output.statistics.length<e.maximumPermutations+1)throw Error(`${t} output.statistics needs maximumPermutations + 1 rows`);if(E(t,e.stopAfterExceedances,e.maximumPermutations,e.output.permutationCount),e.output.summary.length<12)throw Error(`${t} output.summary needs 12 rows`);if(s(t,[e.output.statistics,e.output.summary,e.output.permutationCount],[e.times,e.parameters,e.pairs.offsets,e.pairs.neighbors,e.pairs.weights,e.pairs.distances]),e.output.statistics.buffer===e.output.summary.buffer||e.output.permutationCount&&(e.output.permutationCount.buffer===e.output.statistics.buffer||e.output.permutationCount.buffer===e.output.summary.buffer))throw Error(`${t} outputs must use separate buffers`)}getInferenceMetadata(e){return L(`mantel`,e,this.props.maximumPermutations)}getCommandNodes(e){let{id:t,props:n,rows:r}=this,{pairs:i,times:o,parameters:s,maximumPermutations:c}=n,{statistics:l,summary:d}=n.output,f=i.distances;a(t,e,[i.offsets,i.neighbors,i.weights,f,o,s,l,d,...n.output.permutationCount?[n.output.permutationCount]:[]]);let p=c+1,m=Math.floor(e.device.limits.maxStorageBufferBindingSize/4),[g,v]=P(r,Math.floor(m/(p*K+2))),y=k(g),b=n.stopAfterExceedances,x=b!==void 0,S=D(p),C=x?S:p,E=x?ne(e,t):void 0,O=E?[{name:`state`,view:E,type:`u32`,access:`read`}]:[],N=x?`\nconst BATCH_SIZE: u32 = ${S}u;`:``,F=te(r,c,g,v),L=Math.min(q,r),R=u(e,`${t}-pair-counts`,`uint32`,L),re=u(e,`${t}-pair-sums`,`float32`,L*2),z=u(e,`${t}-pair-count`,`uint32`,1),B=u(e,`${t}-pair-mask`,`uint32`,i.neighbors.length),V=x?void 0:u(e,`${t}-shift`,`float32`,2),ie=u(e,`${t}-partials`,`float32`,C*g*K+g*2),ae=[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`pairMask`,view:B,type:`u32`,access:`read`},{name:`distances`,view:f,type:`f32`,access:`read`},{name:`times`,view:o,type:`f32`,access:`read`}],oe=n=>h(e,{id:n?.id??`${t}-partials`,operation:Oe,variant:`partials`,bindings:[...ae,{name:`parameters`,view:s,type:`u32`,access:`read`},...x?O:[{name:`shift`,view:V,type:`f32`,access:`read`}],{name:`partials`,view:ie,type:`f32`,access:`read_write`}],condition:n?.condition,extraResources:n?.extraResources,workgroupSize:64,invocationCount:C*y*64,guardIndex:!1,declarations:`${F}
${I(g,_.mantelTestShuffle.id)}
const SLOT_COUNT: u32 = ${C}u;${N}
const MOMENTS_BASE: u32 = partialsOffset + ${C*g*K}u;`,body:`${x?j:ee}
  var sumX = 0.0;
  var sumXX = 0.0;
  var sumY = 0.0;
  var sumYY = 0.0;
  var sumXY = 0.0;
  if (runsSlot) {
    ${x?`let shiftDistance = bitcast<f32>(state[stateOffset + ${T.shiftDistance}u]);
    let shiftTime = bitcast<f32>(state[stateOffset + ${T.shiftTime}u]);`:`let shiftDistance = shift[shiftOffset];
    let shiftTime = shift[shiftOffset + 1u];`}
    for (var row = block; row < ROWS; row += BLOCKS) {
      let time = times[timesOffset + getPermutedRow(row, slot)];
      for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
        let other = neighbors[neighborsOffset + entry];
        if (pairMask[pairMaskOffset + entry] != 0u) {
          let x = distances[distancesOffset + entry] - shiftDistance;
          let y = abs(time - times[timesOffset + getPermutedRow(other, slot)]) - shiftTime;
          if (slot == 0u) {
            sumX += x;
            sumXX += x * x;
          }
          sumY += y;
          sumYY += y * y;
          sumXY += x * y;
        }
      }
    }
  }
  if (block < BLOCKS) {
    let base = partialsOffset + (block * SLOT_COUNT + ${x?`localSlot`:`slot`}) * ${K}u;
    partials[base] = sumY;
    partials[base + 1u] = sumYY;
    partials[base + 2u] = sumXY;
  }
  if (slot == 0u && block < BLOCKS) {
    partials[MOMENTS_BASE + 2u * block] = sumX;
    partials[MOMENTS_BASE + 2u * block + 1u] = sumXX;
  }`}),se=n=>h(e,{id:n?.id??`${t}-reduce`,operation:Oe,variant:`reduce`,bindings:[{name:`parameters`,view:s,type:`u32`,access:`read`},...O,{name:`partials`,view:ie,type:`f32`,access:`read`},{name:`pairCount`,view:z,type:`u32`,access:`read`},{name:`statistics`,view:l,type:`f32`,access:`read_write`}],condition:n?.condition,extraResources:n?.extraResources,...x?{workgroupSize:64,invocationCount:S}:{invocationCount:l.length},declarations:`${F}
const SLOT_COUNT: u32 = ${C}u;${N}
const MOMENTS_BASE: u32 = partialsOffset + ${C*g*K}u;`,body:`${x?`let slot = state[stateOffset] * BATCH_SIZE + index;
  let localSlot = index;
  let runs = slot <= readPermutationCount();`:`let slot = index;
  let localSlot = index;
  let runs = index <= readPermutationCount();`}
  var result = 0.0;
  if (runs) {
    var sumX = 0.0;
    var sumXX = 0.0;
    var sumY = 0.0;
    var sumYY = 0.0;
    var sumXY = 0.0;
    for (var block = 0u; block < BLOCKS; block++) {
      let base = partialsOffset + (block * SLOT_COUNT + localSlot) * ${K}u;
      sumX += partials[MOMENTS_BASE + 2u * block];
      sumXX += partials[MOMENTS_BASE + 2u * block + 1u];
      sumY += partials[base];
      sumYY += partials[base + 1u];
      sumXY += partials[base + 2u];
    }
    let count = f32(pairCount[pairCountOffset]);
    let covariance = sumXY - sumX * sumY / count;
    let varianceX = sumXX - sumX * sumX / count;
    let varianceY = sumYY - sumY * sumY / count;
    if (count > 1.0 && varianceX > 0.0 && varianceY > 0.0) {
      result = clamp(covariance / sqrt(varianceX * varianceY), -1.0, 1.0);
    }
  }
  ${x?`if (runs) { statistics[statisticsOffset + slot] = result; }`:`statistics[statisticsOffset + slot] = result;`}`}),ce=x&&E?A(e,{id:t,operation:Oe,stopAfterExceedances:b,slotCount:p,groupsPerSlot:y,workgroupSize:64,statisticType:`f32`,parameters:s,statistics:l,state:E,common:F,createPartialsNode:oe,createReduceNode:se}):[];return[M(e,{id:t,operation:Oe,rows:r,offsets:i.offsets,neighbors:i.neighbors,pairMask:B}),h(e,{id:`${t}-pair-stats`,operation:Oe,variant:`pair-stats`,bindings:[...ae,{name:`pairCounts`,view:R,type:`u32`,access:`read_write`},{name:`pairSums`,view:re,type:`f32`,access:`read_write`}],invocationCount:L,declarations:`const ROWS: u32 = ${r}u;
const PAIR_STAT_BLOCKS: u32 = ${L}u;`,body:`var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var row = index; row < ROWS; row += PAIR_STAT_BLOCKS) {
    for (var entry = offsets[offsetsOffset + row]; entry < offsets[offsetsOffset + row + 1u]; entry++) {
      let other = neighbors[neighborsOffset + entry];
      if (pairMask[pairMaskOffset + entry] != 0u) {
        count++;
        sumDistance += distances[distancesOffset + entry];
        sumTime += abs(times[timesOffset + row] - times[timesOffset + other]);
      }
    }
  }
  pairCounts[pairCountsOffset + index] = count;
  pairSums[pairSumsOffset + 2u * index] = sumDistance;
  pairSums[pairSumsOffset + 2u * index + 1u] = sumTime;`}),h(e,{id:`${t}-shift`,operation:Oe,variant:`shift`,bindings:[{name:`pairCounts`,view:R,type:`u32`,access:`read`},{name:`pairSums`,view:re,type:`f32`,access:`read`},{name:`pairCount`,view:z,type:`u32`,access:`read_write`},x&&E?{name:`state`,view:E,type:`u32`,access:`read_write`}:{name:`shift`,view:V,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const BLOCKS: u32 = ${L}u;`,body:`var count = 0u;
  var sumDistance = 0.0;
  var sumTime = 0.0;
  for (var block = 0u; block < BLOCKS; block++) {
    count += pairCounts[pairCountsOffset + block];
    sumDistance += pairSums[pairSumsOffset + 2u * block];
    sumTime += pairSums[pairSumsOffset + 2u * block + 1u];
  }
  pairCount[pairCountOffset] = count;
  let divisor = max(f32(count), 1.0);
${x?`state[stateOffset + ${T.shiftDistance}u] = bitcast<u32>(sumDistance / divisor);
  state[stateOffset + ${T.shiftTime}u] = bitcast<u32>(sumTime / divisor);`:`shift[shiftOffset] = sumDistance / divisor;
  shift[shiftOffset + 1u] = sumTime / divisor;`}`}),...x?ce:[oe(),se()],h(e,{id:`${t}-summary`,operation:Oe,variant:`summary`,bindings:[{name:`parameters`,view:s,type:`u32`,access:`read`},{name:`statistics`,view:l,type:`f32`,access:`read`},{name:`pairCount`,view:z,type:`u32`,access:`read`},{name:`summary`,view:d,type:`f32`,access:`read_write`},...O,...n.output.permutationCount?[{name:`permutationsUsed`,view:n.output.permutationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:F,body:`let observedPairs = f32(pairCount[pairCountOffset]);
  let timeClosePairs = 0.0;
  let expected = 0.0;
  ${w(`f32`,b)}`})]}};function Y(e,t){let{vertices:n,ringOffsets:r,polygonRingOffsets:i,partFeature:a,featureCount:o}=e,s=new Float64Array(o*3),c=new Float64Array(o*3);for(let e=0;e+1<i.length;e++){let o=a[e];for(let a=i[e];a<i[e+1];a++){let e=r[a],i=r[a+1],l=0,u=0,d=0;for(let r=e;r<i;r++){let a=r+1<i?r+1:e,[s,f]=t.project(n[r*2],n[r*2+1]),[p,m]=t.project(n[a*2],n[a*2+1]),h=s*m-p*f;l+=h,u+=(s+p)*h,d+=(f+m)*h,c[o*3]+=s,c[o*3+1]+=f,c[o*3+2]+=1}Math.abs(l)>1e-6&&(s[o*3]+=u,s[o*3+1]+=d,s[o*3+2]+=l)}}let l=new Float32Array(o*2);for(let e=0;e<o;e++){let t=s[e*3+2];Math.abs(t)>1e-6?(l[e*2]=s[e*3]/(3*t),l[e*2+1]=s[e*3+1]/(3*t)):(l[e*2]=c[e*3]/Math.max(1,c[e*3+2]),l[e*2+1]=c[e*3+1]/Math.max(1,c[e*3+2]))}return l}function ke(e,t){let n=ue(e),r=e.getProjection(t),i=e.geojson?.features??[];return{count:n.featureCount,source:n,centroids:Y(n,r),population:e.column(`population`),geoid:i.map(e=>String(e.properties?.GEOID??``)),communityArea:e.column(`communityArea`)}}var Ae={fungi:{dataset:`chicago-nature`,category:3,label:`fungi observations`},insects:{dataset:`chicago-nature`,category:2,label:`insect observations`},birds:{dataset:`chicago-nature`,category:1,label:`bird observations`},plants:{dataset:`chicago-nature`,category:0,label:`plant observations`},mammals:{dataset:`chicago-nature`,category:4,label:`mammal observations`},rats:{dataset:`chicago-311-rats`,category:null,label:`rodent complaints`}},je=24e3,Me=64,X=999,Ne=14,Z=26,Q=7,Pe=4294967295,$=S,Fe=350,Ie=[`Jan`,`Feb`,`Mar`,`Apr`,`May`,`Jun`,`Jul`,`Aug`,`Sep`,`Oct`,`Nov`,`Dec`],Le=[[140,150,170,55],[215,48,39,235],[244,140,40,235],[140,60,190,235],[30,150,90,235],[30,110,200,235],[200,170,20,235],[90,90,90,235]],Re=e=>{let t=new Date(Date.UTC(2023,0,1+e));return`${Ie[t.getUTCMonth()]} ${t.getUTCDate()}`};function ze(e,t=2){return Number.isFinite(e)?e.toFixed(t):`n/a`}function Be(e){return Number.isFinite(e)?e<.001?`< 0.001`:e.toFixed(3):`n/a`}function Ve(e,t,n){let r=n,i=n;for(let n=0;n<t;n++)r=Math.min(r,e[n]),i=Math.max(i,e[n]);let a=Math.max(i-r,1e-9),o=Array(32).fill(0),s=0;for(let i=0;i<t;i++)o[Math.min(31,Math.floor((e[i]-r)/a*32))]++,e[i]<n&&s++;let c=Math.max(...o,1);return`${o.map(e=>e===0?`·`:`.:-=+*#@`[Math.min(7,Math.round(e/c*7))]).join(``)}  observed beats ${(s/Math.max(t,1)*100).toFixed(1)}% of permutations`}async function He(e){let{device:t}=e,n=e.datasets.get(`chicago-nature`),r=e.datasets.get(`chicago-311-rats`),a=e.datasets.get(`chicago-tracts`),o=n.defaultOrigin,s=n.getProjection(o),u=ke(a,o),d=ce(u.source,a.getProjection(o),a.manifest.bbox,1400),f=u.count,p=new l(t,`clusters`),m=p.track(new ae),h=p.createBuffer(`cell-feature`,d.cellFeature),g=H(u.source,a.getProjection(o)),_=p.createBuffer(`outlines`,g),b=p.createBuffer(`zone-positions`,u.centroids),x=p.createParameterBuffer(`search-parameters`,`float32`,12),S=p.createParameterBuffer(`test-parameters`,`uint32`,4),ee=p.createParameterBuffer(`draw-parameters`,`float32`,2),te=p.createParameterBuffer(`reduction-parameters`,`float32`,2,V(0,Ne*86400)),w=p.createParameterBuffer(`scan-parameters`,`uint32`,8),T=p.createParameterBuffer(`significance`,`float32`,1),E=!1,D=7,O=!0,k=!0,A=performance.now()-1e3,j=null,M=1,ne=1,N=null,P=!1,I=new Map,L=new Map;function ue(e){let t=L.get(e);if(t)return t;let i=Ae[e],a=i.dataset===`chicago-nature`?n:r,s=a.column(`timestamp`),c=i.category===null?null:a.column(`category`),l=a.column(`tract`),u=a.projectColumn(`position`,o),d=[];for(let e=0;e<s.length;e++)(c===null||c[e]===i.category)&&d.push(e);return t={positions:new Float32Array(d.length*2),seconds:new Float32Array(d.length),tract:new Uint32Array(d.length)},d.forEach((e,n)=>{t.positions[n*2]=u[e*2],t.positions[n*2+1]=u[e*2+1],t.seconds[n]=s[e],t.tract[n]=l[e]<f?l[e]:Pe}),L.set(e,t),t}async function de(n,r){let a=r.seconds.length,o=Math.max(1,Math.ceil(a/je)),c=Math.ceil(a/o),l=new Float32Array(c*2),u=new Float32Array(c),d=1/0,f=1/0,m=-1/0,h=-1/0;for(let e=0;e<c;e++){let t=e*o;l[e*2]=r.positions[t*2],l[e*2+1]=r.positions[t*2+1],u[e]=r.seconds[t]/86400,d=Math.min(d,l[e*2]),m=Math.max(m,l[e*2]),f=Math.min(f,l[e*2+1]),h=Math.max(h,l[e*2+1])}let g=[d-100,f-100,m+100,h+100],_=c*Me,v=(e,t)=>p.createBuffer(`${n}-knox-${e}`,t),b=v(`positions`,l),te=v(`times`,u),w=v(`offsets`,(c+1)*4),T=v(`neighbors`,_*4),D=v(`weights`,_*4),O=v(`distances`,_*4),k=v(`overflow`,4),A=v(`knox-statistics`,(X+1)*4),M=v(`knox-summary`,48),ne=v(`mantel-statistics`,(X+1)*4),N=v(`mantel-summary`,48),P=v(`segments`,_*16),I=v(`fade`,_*4),L=new i(t,{id:`knox-${n}`}),R=U(L,`${n}-knox`),re=R(`positions`,b,`float32x2`,c),z=R(`times`,te,`float32`,c),B={spatialContext:s.spatialContext,offsets:R(`offsets`,w,`uint32`,c+1),neighbors:R(`neighbors`,T,`uint32`,_),weights:R(`weights`,D,`float32`,_),distances:R(`distances`,O,`float32`,_),status:{overflow:R(`overflow`,k,`uint32`,1)}},V=S.importToGraph(L);return L.add(new y({spatialContext:s.spatialContext,id:`${n}-search`,mode:`radius`,positions:re,parameters:x.importToGraph(L),gridSize:[64,64],output:B})),L.add(new F({id:`${n}-knox`,pairs:B,times:z,parameters:V,maximumPermutations:X,output:{statistics:R(`knox-statistics`,A,`uint32`,X+1),summary:R(`knox-summary`,M,`float32`,12)}})),L.add(new J({id:`${n}-mantel`,pairs:B,times:z,parameters:V,maximumPermutations:X,output:{statistics:R(`mantel-statistics`,ne,`float32`,X+1),summary:R(`mantel-summary`,N,`float32`,12)}})),se(L,{id:`${n}-segments`,invocationCount:_,declarations:`const EVENT_COUNT: u32 = ${c}u;`,bindings:[{name:`offsets`,view:B.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:B.neighbors,type:`u32`,access:`read`},{name:`positions`,view:re,type:`f32`,access:`read`},{name:`times`,view:z,type:`f32`,access:`read`},{name:`drawParameters`,view:ee.importToGraph(L),type:`f32`,access:`read`},{name:`segments`,view:R(`segments`,P,`float32`,_*4),type:`f32`,access:`read_write`},{name:`fade`,view:R(`fade`,I,`float32`,_),type:`f32`,access:`read_write`}],body:`
  segments[segmentsOffset + index * 4u] = 0.0;
  segments[segmentsOffset + index * 4u + 1u] = 0.0;
  segments[segmentsOffset + index * 4u + 2u] = 0.0;
  segments[segmentsOffset + index * 4u + 3u] = 0.0;
  fade[fadeOffset + index] = 0.0;
  if (index >= offsets[offsetsOffset + EVENT_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  if (neighbor <= low) {
    return;
  }
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  let timeGap = abs(times[timesOffset + low] - times[timesOffset + neighbor]);
  fade[fadeOffset + index] = select(drawParameters[drawParametersOffset + 1u], 1.0, timeGap <= drawParameters[drawParametersOffset]);`}),{part:{compiled:p.track(await L.compileAsync()),eventCount:c,positions:b,times:te,segments:P,fade:I,slotCapacity:_,reader:new oe(p,`${n}-knox`,[{buffer:M,size:48},{buffer:N,size:48},{buffer:k,size:4},{buffer:A,size:(X+1)*4},{buffer:ne,size:(X+1)*4}],t=>{if(E||j?.key!==n)return;let r=new Float32Array(t,0,12),i=new Float32Array(t,48,12),a=new Uint32Array(t,96,1)[0],o=new Uint32Array(t,100,X+1),s=new Float32Array(t,100+(X+1)*4,X+1),c=e.options.permutations;e.setReadout(`pairs`,r[$.pairCount]),e.setReadout(`timeClose`,r[$.timeClosePairs]),e.setReadout(`knoxObserved`,r[$.observed]),e.setReadout(`knoxExpected`,ze(r[$.expected],1)),e.setReadout(`knoxRatio`,`${ze(r[$.observed]/Math.max(r[$.expected],1e-9),2)}x`),e.setReadout(`knoxPermuted`,`${ze(r[$.permutationMean],1)} ± ${ze(Math.sqrt(r[$.permutationVariance]),1)}`),e.setReadout(`knoxP`,Be(r[$.pseudoPGreater])),e.setReadout(`knoxPoisson`,Be(C(r[$.observed],r[$.expected]))),e.setReadout(`knoxDistribution`,Ve(o.subarray(1),c,r[$.observed])),e.setReadout(`mantelR`,ze(i[$.observed],4)),e.setReadout(`mantelPermuted`,`${ze(i[$.permutationMean],4)} ± ${ze(Math.sqrt(i[$.permutationVariance]),4)}`),e.setReadout(`mantelP`,Be(i[$.pseudoPGreater])),e.setReadout(`mantelDistribution`,Ve(s.subarray(1),c,i[$.observed])),e.setReadout(`pairOverflow`,a?`yes: lower the radius`:`no`)})},events:c,searchBounds:g}}async function W(n,r){let a=r.seconds.length,o=new Uint32Array(f*Z);for(let e=0;e<a;e++){let t=r.tract[e];if(t===Pe)continue;let n=Math.floor(r.seconds[e]/(Ne*86400));n<Z&&o[t*Z+n]++}let c=f*Z,l=(e,t)=>p.createBuffer(`${n}-scan-${e}`,t),d=l(`seconds`,r.seconds),m=l(`ones`,new Float32Array(a).fill(1)),h=l(`tract`,r.tract),g=l(`cases`,c*4),_=l(`baseline`,c*4),v=l(`cluster-indices`,Q*4*4),y=l(`cluster-statistics`,Q*8*4),x=l(`statistics`,(X+1)*4),S=l(`summary`,16),ee=l(`zone-statistics`,(f+1)*4),C=l(`membership`,(f+1)*4),D=new Float32Array(f+1).fill(NaN),O=0;for(let e=0;e<f;e++){let t=0;for(let n=0;n<Z;n++)t+=o[e*Z+n];u.population[e]>=300&&(D[e]=t/u.population[e]*1e3)}let k=Float32Array.from(D.filter(Number.isFinite)).sort();O=k[Math.floor(k.length*.97)]??1,ne=Math.max(O,1),e.setLegendExtent(`rate`,[0,ne]);let A=l(`rate`,D),M=e=>l(e,c*4),N=new i(t,{id:`scan-${n}`}),P=U(N,`${n}-scan`);N.add(new ie({id:`${n}-reduction`,cellIds:P(`tract`,h,`uint32`,a),timestamps:P(`seconds`,d,`float32`,a),values:P(`ones`,m,`float32`,a),parameters:te.importToGraph(N),cellCount:f,bucketCount:Z,output:{counts:P(`cases`,g,`uint32`,c),minimums:P(`min`,M(`min`),`float32`,c),maximums:P(`max`,M(`max`),`float32`,c),first:P(`first`,M(`first`),`float32`,c),last:P(`last`,M(`last`),`float32`,c),occupiedSlots:{ids:P(`occupied-ids`,l(`occupied-ids`,c*4),`uint32`,c),status:{count:P(`occupied-count`,l(`occupied-count`,4),`uint32`,1),overflow:P(`occupied-overflow`,l(`occupied-overflow`,4),`uint32`,1)}}}}));let F=P(`zone-positions`,b,`float32x2`,f),I=P(`cluster-indices`,v,`uint32`,Q*4),L=P(`cluster-statistics`,y,`float32`,Q*8);return N.add(new De({spatialContext:s.spatialContext,id:`${n}-scan`,positions:F,cases:P(`cases`,g,`uint32`,c),baseline:P(`baseline`,_,`float32`,c),timeBuckets:Z,maximumPermutations:X,maximumClusters:Q,parameters:w.importToGraph(N),output:{clusterIndices:I,clusterStatistics:L,statistics:P(`statistics`,x,`float32`,X+1),summary:P(`summary`,S,`uint32`,4),zoneStatistics:P(`zone-statistics`,ee,`float32`,f)}})),se(N,{id:`${n}-membership`,invocationCount:f+1,declarations:`const ZONES: u32 = ${f}u; const CLUSTERS: u32 = ${Q}u;`,bindings:[{name:`indices`,view:I,type:`u32`,access:`read`},{name:`stats`,view:L,type:`f32`,access:`read`},{name:`positions`,view:F,type:`f32`,access:`read`},{name:`alpha`,view:T.importToGraph(N),type:`f32`,access:`read`},{name:`membership`,view:P(`membership`,C,`uint32`,f+1),type:`u32`,access:`read_write`}],body:`
  if (index >= ZONES) {
    membership[membershipOffset + index] = 0xffffffffu;
    return;
  }
  var member = 0u;
  let px = positions[positionsOffset + index * 2u];
  let py = positions[positionsOffset + index * 2u + 1u];
  for (var cluster = 0u; cluster < CLUSTERS; cluster++) {
    if (indices[indicesOffset + cluster * 4u + 1u] == 0u) {
      continue;
    }
    if (stats[statsOffset + cluster * 8u + 3u] > alpha[alphaOffset]) {
      continue;
    }
    let center = indices[indicesOffset + cluster * 4u];
    let dx = px - positions[positionsOffset + center * 2u];
    let dy = py - positions[positionsOffset + center * 2u + 1u];
    let radius = stats[statsOffset + cluster * 8u + 4u];
    if (dx * dx + dy * dy <= radius * radius * 1.00002 + 1.0) {
      member = cluster + 1u;
      break;
    }
  }
  membership[membershipOffset + index] = member;`}),{compiled:p.track(await N.compileAsync()),caseCount:a,cases:g,baseline:_,membership:C,zoneStatistics:ee,rate:A,totals:o,reader:new oe(p,`${n}-scan`,[{buffer:S,size:16},{buffer:v,size:Q*16},{buffer:y,size:Q*32},{buffer:C,size:(f+1)*4}],e=>{E||j?.key!==n||_e(e)})}}let fe=[0,0,1,1];async function pe(){let t=e.options.source,n=I.get(t);if(!n){let e=ue(t),{part:r,events:i,searchBounds:a}=await de(t,e);n={key:t,knox:r,scan:await W(t,e),knoxEvents:i,searchBounds:a},I.set(t,n)}return n}function me(){if(!j)return;let{totals:t}=j.scan,n=e.options.baseline,r=new Float32Array(f*Z),i=new Float64Array(Z),a=new Float64Array(f),o=0;for(let e=0;e<f;e++)for(let n=0;n<Z;n++){let r=t[e*Z+n];i[n]+=r,a[e]+=r,o+=r}for(let e=0;e<f;e++)for(let t=0;t<Z;t++){let s=e*Z+t;n===`independence`?r[s]=o>0?a[e]*i[t]/o:0:n===`population`?r[s]=o>0?(u.population[e]||0)*(i[t]/o):0:r[s]=1}j.scan.baseline.write(r),k=!0,A=performance.now()}function G(){let t=e.options;x.write(v({bounds:fe,radius:t.spatialRadius,weightKind:`binary`})),S.write(R({seed:D,permutationCount:t.permutations,timeThreshold:t.timeThreshold})),ee.write(Float32Array.of(t.timeThreshold,t.showSpatialOnly?.14:0)),O=!0}function ge(){let t=e.options;w.write(he({seed:D,permutationCount:t.scanPermutations,maximumPopulationFraction:t.maximumPopulationFraction,maximumWindowZones:t.maximumWindowZones,maximumTimeBuckets:t.maximumTimeBuckets,windowShape:t.windowShape})),T.write(Float32Array.of(t.significance)),k=!0,A=performance.now()}function _e(t){let n=new Uint32Array(t,0,4),r=new Uint32Array(t,16,Q*4),i=new Float32Array(t,16+Q*16,Q*8);N=new Uint32Array(t,16+Q*48,f+1);let a=n[0];e.setReadout(`scanCases`,c(n[1])),e.setReadout(`scanClusters`,`${a} found, ${ve(i,a)} significant`),M=Math.max(1e-6,i[0]),e.setLegendExtent(`llr`,[0,M]);for(let t=0;t<3;t++){if(t>=a){e.setReadout(`cluster${t+1}`,`-`);continue}let n=r[t*4+1],o=r[t*4+2],s=r[t*4+3],l=i[t*8+1],u=i[t*8+2],d=i[t*8+3];e.setReadout(`cluster${t+1}`,`${n} tracts within ${((i[t*8+4]??0)/1e3).toFixed(1)} km, ${Re(o*Ne)} to ${Re((s+1)*Ne-1)}. ${c(l)} cases vs ${u.toFixed(0)} expected (${(i[t*8+5]??0).toFixed(2)}x), LLR ${i[t*8].toFixed(1)}, p ${Be(d)}`)}e.requestLayers()}function ve(t,n){let r=0;for(let i=0;i<n;i++)t[i*8+3]<=e.options.significance&&r++;return r}function ye(t){j=t,fe=t.searchBounds,N=null,e.setReadout(`events`,`${c(j.knoxEvents)} of ${c(j.scan.caseCount)} ${Ae[e.options.source].label}`),me(),G(),ge(),e.requestLayers()}return ye(await pe()),{getCompiledGraphs:()=>j?[j.knox.compiled,j.scan.compiled]:[],setOption(t){t===`source`?m.request({build:pe,commit:ye}):t===`spatialRadius`||t===`timeThreshold`||t===`permutations`||t===`showSpatialOnly`?G():t===`baseline`?me():(t===`maximumPopulationFraction`||t===`maximumWindowZones`||t===`maximumTimeBuckets`||t===`windowShape`||t===`scanPermutations`||t===`significance`)&&ge(),e.requestLayers()},onAction(e){e===`reseed`&&(D++,G(),ge())},onThemeChange(){e.requestLayers()},getTooltip({coordinate:t}){if(!j||!t||e.options.analysis!==`scan`)return null;let n=le(d,t[0],t[1]);if(n<0)return null;let r=[`Tract ${u.geoid[n]??n}`];r.push(`${c(u.population[n]||0)} residents`);let i=0;for(let e=0;e<Z;e++)i+=j.scan.totals[n*Z+e];r.push(`${c(i)} ${Ae[e.options.source].label} in 2023`);let a=N?.[n]??0;return a>0&&a!==Pe&&r.push(`In significant cluster #${a}`),r.join(`
`)},encode(e,t){if(!j)return;(O||t.frameIndex<2)&&(j.knox.compiled.encode(e,{parameters:void 0}),O=!1,j.knox.reader.request(e));let n=performance.now()-A>Fe;(k&&n||!P&&t.frameIndex>3)&&(P=!0,j.scan.compiled.encode(e,{parameters:void 0}),k=!1,j.scan.reader.request(e)),j.knox.reader.flush(e),j.scan.reader.flush(e)},getLayers(){if(!j)return[];let t=e.options,n=e.theme()===`dark`,r=[o[0],o[1],0],i=[];if(t.analysis===`pairs`)return i.push(new z({id:`clusters-pairs`,coordinateOrigin:r,segments:j.knox.segments,weights:j.knox.fade,instanceCount:j.knox.slotCapacity,widthPixels:1.4,colormap:`uniform`,color:n?[255,170,60,220]:[230,110,10,220]}),new re({id:`clusters-events`,coordinateOrigin:r,positions:j.knox.positions,instanceCount:j.knox.eventCount,radiusPixels:2.6,values:j.knox.times,valueFormat:`float32`,colormap:`lajolla`,valueRange:[0,365],color:[255,255,255,235]})),i;let a={coordinateOrigin:r,gridSize:[d.width,d.height],bounds:d.bounds,valueIndices:h,opacity:t.opacity,color:[255,255,255,255]};return t.scanView===`clusters`?i.push(new B({...a,id:`clusters-membership`,values:j.scan.membership,valueFormat:`uint32`,colormap:`category`,palette:Le})):t.scanView===`zone-llr`?i.push(new B({...a,id:`clusters-llr`,values:j.scan.zoneStatistics,valueFormat:`float32`,colormap:`inferno`,valueRange:[0,M],sqrtScale:!0,discardAtOrBelow:0})):i.push(new B({...a,id:`clusters-rate`,values:j.scan.rate,valueFormat:`float32`,colormap:`ylorrd`,valueRange:[0,ne],sqrtScale:!0})),i.push(new z({id:`clusters-outlines`,coordinateOrigin:r,segments:_,instanceCount:g.length/4,widthPixels:.8,colormap:`uniform`,color:n?[230,235,245,70]:[40,50,70,80]})),i},destroy(){E=!0;for(let e of I.values())e.knox.reader.stop(),e.scan.reader.stop();p.destroy()}}}export{He as createSpaceTimeClusters};