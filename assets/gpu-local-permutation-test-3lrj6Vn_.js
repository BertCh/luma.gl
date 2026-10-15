import{J as e}from"./src-CNCKFW2b.js";import{a as t,c as n,d as r,f as i,i as a,l as o,n as s,o as c,p as l,r as u,s as d,t as f,u as p}from"./permutation-inference-kernels-D-xhEewi.js";import{_ as m,d as h,h as g,j as _,w as v,y}from"./resources-CwvhEOBe.js";import{n as b,s as x}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as S,s as C}from"./spatial-weights-D4bd9AQ6.js";import{r as w,t as T}from"./gpu-multiple-testing-adjustment-C7k5NYt0.js";import{t as E}from"./philox-random-CHwyk0dd.js";import{t as D}from"./permutation-random-C7SSHCbB.js";var O=`GPULocalPermutationTest`,k=E.localPermutationTest.id,A=4294967295,j={undefined:0,highHigh:1,lowLow:2,negativeAssociation:3,notSignificant:4};function M(t,n){let r={bivariate:!1,joinCount:!1,centerY:!0,centerX:!1};switch(t){case`localMoran`:return{...r,centerX:!0,drawnA:`dx`,term:`fx * lagA`};case`localGeary`:return{...r,centerX:!0,drawnA:`(fx - dx) * (fx - dx)`,term:`lagA`};case`localG`:return{...r,drawnA:`dx`,term:`lagA`};case`localGStar`:return{...r,drawnA:`dx`,term:`lagA + fx`};case`localBivariateMoran`:return{...r,centerX:!0,bivariate:!0,drawnA:`dy`,term:`fx * lagA`};case`localLeeL`:return{...r,centerX:!0,bivariate:!0,drawnA:`dx`,drawnB:`dy`,term:`lagA * lagB`};case`localJoinCount`:return{...r,joinCount:!0,drawnA:`isOne(dx)`,term:`isOne(fx) * lagA`};case`localBivariateJoinCount`:return n===`BJC`?{...r,joinCount:!0,bivariate:!0,centerY:!1,drawnA:`(1.0 - isOne(dx)) * isOne(dy)`,term:`isOne(fx) * (1.0 - isOne(fy)) * lagA`}:{...r,joinCount:!0,bivariate:!0,centerY:!1,drawnA:`isOne(dx) * isOne(dy)`,term:`isOne(fx) * isOne(fy) * lagA`};default:e(!1)}}function ee(e,t=`test`){let{constantsWGSL:n,accessWGSL:i,maximumPermutations:a,maximumNeighbors:o,stopAfter:s,alternative:c,spec:l,neighborLoopWGSL:u,slotWeightWGSL:f}=e,p=t===`replay`,m=!p&&s!==void 0,h=!p&&c===`folded`;return{declarations:`${n}
${i}
${d(a)}
${D}
${m?`const STOP_AFTER_EXCEEDANCES: u32 = ${s}u;\n`:``}const RANDOM_TAG: u32 = ${k}u;

// Value at a virtual shuffle position: a recorded swap, or the position itself.
fn readSwap(
  swapPositions: ptr<function, array<u32, ${o}>>,
  swapValues: ptr<function, array<u32, ${o}>>,
  swapCount: u32,
  position: u32
) -> u32 {
  for (var entry = 0u; entry < swapCount; entry++) {
    if ((*swapPositions)[entry] == position) {
      return (*swapValues)[entry];
    }
  }
  return position;
}`,body:`let observed = ${p?`bitcast<f32>(replayInfo[replayInfoOffset])`:`observedRaw[observedRawOffset + index]`};
  // The permutation count is never clamped: an invalid one leaves every row untested.
  // Join counts test only rows with at least one join (esda sets the other p-values to NaN).
  if (!isFiniteFloat(observed) || !isPermutationCountValid()${l.joinCount?` || observed == 0.0`:``}) {
    ${p?``:`exceedances[exceedancesOffset + index] = NOT_TESTED;${m?`
    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;`:``}`}
    return;
  }
  let position = rowPositions[rowPositionsOffset + index];
  let others = rowPositions[rowPositionsOffset + ROWS] - 1u;
  let fx = getX(position);
  let fy = getY(position);
  let permutations = readValidPermutationCount();${m?`
  var permutationsRun = permutations;`:``}
  let key = readSeedKey();
  ${p?``:`var greater = 0u;
  var lesser = 0u;
  var folded = 0u;
  var total = 0.0;
  var mean = 0.0;`}
  // The row's active neighbor weights in slot order, read once: a permutation only reassigns
  // values to slots, so the per-draw work is the random draw and one value gather, not four
  // CSR gathers (offsets, neighbor, neighbor position, weight) per slot per permutation.
  var slotWeights: array<f32, ${o}>;
  var neighborTotal = 0u;
  ${u(`if (neighborTotal < MAXIMUM_NEIGHBORS) {
      slotWeights[neighborTotal] = ${f};
      neighborTotal++;
    }`)}
  // Islands (kept) and rows that cannot draw k distinct others are not tested.
  if (neighborTotal == 0u || neighborTotal > others) {
    ${p?``:`exceedances[exceedancesOffset + index] = NOT_TESTED;${m?`
    exceedances[exceedancesOffset + ROWS + index] = NOT_TESTED;`:``}`}
    return;
  }
  ${p?``:`// 'folded' needs the simulated mean first: sweep 0 sums, sweep 1 counts (same deterministic stream).
  for (var sweep = 0u; sweep < ${h?2:1}u; sweep++) {`}
  for (var permutation = 0u; permutation < permutations; permutation++) {
    var stream = createPhiloxStream(key, index, permutation, RANDOM_TAG);
    var swapPositions: array<u32, ${o}>;
    var swapValues: array<u32, ${o}>;
    var swapCount = 0u;
    var lagA = 0.0;
    var lagB = 0.0;
    for (var drawn = 0u; drawn < neighborTotal; drawn++) {
      // Partial Fisher-Yates over the other analysis rows' virtual positions [0, others).
      let pick = drawn + nextPhiloxBelow(&stream, others - drawn);
      let chosen = readSwap(&swapPositions, &swapValues, swapCount, pick);
      if (pick != drawn) {
        let displaced = readSwap(&swapPositions, &swapValues, swapCount, drawn);
        var entry = 0u;
        loop {
          if (entry == swapCount || swapPositions[entry] == pick) {
            break;
          }
          entry++;
        }
        swapPositions[entry] = pick;
        swapValues[entry] = displaced;
        swapCount = max(swapCount, entry + 1u);
      }
      let drawnPosition = chosen + select(0u, 1u, chosen >= position);
      let dx = getX(drawnPosition);
      let dy = getY(drawnPosition);
      lagA += slotWeights[drawn] * getDrawnA(fx, fy, dx, dy);
      ${l.drawnB?`lagB += slotWeights[drawn] * getDrawnB(fx, fy, dx, dy);`:``}
    }
    let simulated = getLocalTerm(fx, fy, lagA, lagB);
    ${p?`nullDistribution[nullDistributionOffset + permutation] = simulated;`:h?`if (sweep == 0u) {
      total += simulated;
    } else {
      folded += select(0u, 1u, abs(simulated - mean) >= abs(observed - mean));
    }`:`greater += select(0u, 1u, simulated >= observed);
    lesser += select(0u, 1u, simulated <= observed);${m?`
    // Besag-Clifford: stop at the first permutation whose exceedance count reaches the target.
    if (${r(c)} >= STOP_AFTER_EXCEEDANCES) {
      permutationsRun = permutation + 1u;
      break;
    }`:``}`}
  }
  ${p?`replayInfo[replayInfoOffset + 1u] = bitcast<u32>(f32(permutations));`:`mean = total / f32(permutations);
  }
  exceedances[exceedancesOffset + index] = ${r(c)};${m?`
  exceedances[exceedancesOffset + ROWS + index] = permutationsRun;`:``}`}`}}var N=class{id;props;replayInputs;constructor(e){let t=e.id??`local-permutation-test`;this.id=t,this.props=e;let r=n({...e,id:t,operation:O,islandCount:e.output.islandCount});if(![`localMoran`,`localGeary`,`localG`,`localGStar`,`localBivariateMoran`,`localLeeL`,`localJoinCount`,`localBivariateJoinCount`].includes(e.statistic))throw Error(`${t} unknown statistic ${e.statistic}`);if(e.statistic===`localBivariateJoinCount`){if(e.joinCase!==`BJC`&&e.joinCase!==`CLC`)throw Error(`${t} localBivariateJoinCount requires joinCase 'BJC' or 'CLC'`)}else if(e.joinCase!==void 0)throw Error(`${t} joinCase applies to localBivariateJoinCount only`);let a=M(e.statistic,e.joinCase).bivariate;if(a&&!e.secondValues)throw Error(`${t} ${e.statistic} requires secondValues`);if(!a&&e.secondValues)throw Error(`${t} ${e.statistic} does not take secondValues`);if(e.output.classes){if(e.statistic!==`localGeary`)throw Error(`${t} classes applies to localGeary only`);_(e.output.classes,`${t} classes`)}if(e.output.quadrants){if(e.statistic!==`localBivariateMoran`)throw Error(`${t} quadrants applies to localBivariateMoran only`);_(e.output.quadrants,`${t} quadrants`)}if(l(t,e.stopAfterExceedances,i(t,e.alternative),e.maximumPermutations),e.output.permutationCounts){if(e.stopAfterExceedances===void 0)throw Error(`${t} permutationCounts requires stopAfterExceedances`);_(e.output.permutationCounts,`${t} permutationCounts`)}C(t,e.islands),h(t,`maximumNeighbors`,e.maximumNeighbors??32,1,64),_(e.output.exceedances,`${t} exceedances`),y(e.output.pseudoPValues,[`float32`],`${t} pseudoPValues`),e.output.observed&&y(e.output.observed,[`float32`],`${t} observed`),e.output.significant&&_(e.output.significant,`${t} significant`),e.output.adjustedPValues&&y(e.output.adjustedPValues,[`float32`],`${t} adjustedPValues`);for(let[n,i]of[[`exceedances`,e.output.exceedances],[`permutationCounts`,e.output.permutationCounts],[`pseudoPValues`,e.output.pseudoPValues],[`observed`,e.output.observed],[`significant`,e.output.significant],[`adjustedPValues`,e.output.adjustedPValues],[`classes`,e.output.classes],[`quadrants`,e.output.quadrants]])if(i&&i.length!==r)throw Error(`${t} ${n} length must equal the weights row count`);let o=w(t,e.multipleTesting),{significant:s,adjustedPValues:c,classes:u}=e.output;if(o!==`none`&&!s&&!c&&!u)throw Error(`${t} multipleTesting requires significant or adjustedPValues`);_(e.output.status.overflow,`${t} output.status.overflow`),h(t,`output.status.overflow length`,e.output.status.overflow.length),g(t,[e.output.exceedances,e.output.permutationCounts,e.output.pseudoPValues,e.output.observed,e.output.significant,e.output.adjustedPValues,e.output.classes,e.output.quadrants,e.output.islandCount,e.output.status.overflow],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.values,e.secondValues,e.mask,e.parameters,e.weights.status?.overflow])}getParameterValues(e){return o(e,{maximumPermutations:this.props.maximumPermutations})}getInferenceMetadata(e){this.getParameterValues(e);let t=C(this.id,this.props.islands)===`keep`,n=this.props.maximumNeighbors??32;return{alternative:i(this.id,this.props.alternative),permutationCount:e.permutationCount,seed:e.seed,multipleTesting:w(this.id,this.props.multipleTesting),significanceLevel:e.significanceLevel??.05,testedRowsDefinition:`rows with a nonzero mask and a finite value that have 1 to ${n} positive-weight neighbors among the analysis rows (islands, rows without such a neighbor, `+(t?`are kept in n but not tested)`:`are excluded from the analysis)`)}}getCommandNodes(e){let{id:t,props:n}=this,{weights:r,parameters:o,statistic:l,maximumPermutations:f}=n;m(t,e,[r.offsets,r.neighbors,r.weights,r.status?.overflow,n.values,n.secondValues,n.mask,o,n.output.exceedances,n.output.permutationCounts,n.output.pseudoPValues,n.output.observed,n.output.significant,n.output.adjustedPValues,n.output.classes,n.output.quadrants,n.output.islandCount,n.output.status.overflow]);let h=n.values.length,g=i(t,n.alternative),_=p(g),y=n.stopAfterExceedances,E=y!==void 0,D=C(t,n.islands)===`keep`,k=w(t,n.multipleTesting),j=n.maximumNeighbors??32,N=M(l,n.joinCase),te=N.bivariate,P=c(e,{...n,islandCount:n.output.islandCount,id:`${t}-inputs`,operation:O,centerX:N.centerX,centerY:N.centerY}),F=P.nodes,{rowPositions:I,compactX:L,compactY:ne,totals:re}=P,R=P.weights,z=v(e,`${t}-observed-raw`,`float32`,h),B=te?v(e,`${t}-pairs`,`float32`,2*h):void 0,V=B?{name:`pairs`,view:B,type:`f32`,access:`read`}:{name:`compactX`,view:L,type:`f32`,access:`read`},H=`const ROWS: u32 = ${h}u;
const CAPACITY: u32 = ${r.neighbors.length}u;
const INVALID_POSITION: u32 = ${u}u;
const MAXIMUM_NEIGHBORS: u32 = ${j}u;
const NOT_TESTED: u32 = ${A}u;
${s}
// A binary join-count variable: exactly 1 is a join, anything else is 0.
fn isOne(value: f32) -> f32 {
  return select(0.0, 1.0, value == 1.0);
}
// The statistic up to a positive per-row constant; identical for observed and simulated terms.
fn getLocalTerm(fx: f32, fy: f32, lagA: f32, lagB: f32) -> f32 {
  return ${N.term};
}
// Per-draw terms of the weighted lags, from the focus (fx, fy) and the drawn row (dx, dy).
fn getDrawnA(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${N.drawnA};
}
fn getDrawnB(fx: f32, fy: f32, dx: f32, dy: f32) -> f32 {
  return ${N.drawnB??`0.0`};
}`,U=B?`fn getX(position: u32) -> f32 { return pairs[pairsOffset + 2u * position]; }
fn getY(position: u32) -> f32 { return pairs[pairsOffset + 2u * position + 1u]; }`:`fn getX(position: u32) -> f32 { return compactX[compactXOffset + position]; }
fn getY(position: u32) -> f32 { return 0.0 * f32(position); }`,W=N.joinCount?`1.0`:`weight`,G=e=>S(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,e),K={constantsWGSL:H,accessWGSL:U,maximumPermutations:f,maximumNeighbors:j,stopAfter:y,alternative:g,spec:N,neighborLoopWGSL:G,slotWeightWGSL:W};e.recording&&(this.replayInputs={kernel:K,bindings:[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`weights`,view:R,type:`f32`,access:`read`},{name:`rowPositions`,view:I,type:`u32`,access:`read`},V,{name:`parameters`,view:o,type:`u32`,access:`read`}],observedRaw:z});let q=r.status?.overflow,J=E?v(e,`${t}-permute-results`,`uint32`,2*h):void 0,Y=E?n.output.permutationCounts??v(e,`${t}-permutation-counts`,`uint32`,h):void 0;F.push(b(e,{id:`${t}-overflow-clear`,operation:O,view:n.output.status.overflow,type:`u32`,value:`0u`,componentCount:1})),q&&F.push(x(e,{id:`${t}-weights-overflow`,operation:O,variant:`weights-overflow`,bindings:[{name:`weightsOverflow`,view:q,type:`u32`,access:`read`},{name:`overflow`,view:n.output.status.overflow,type:`atomic<u32>`,access:`read_write`}],invocationCount:1,body:`if (weightsOverflow[weightsOverflowOffset] != 0u) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`})),B&&F.push(x(e,{id:`${t}-pairs`,operation:O,variant:`pairs`,bindings:[{name:`compactX`,view:L,type:`f32`,access:`read`},{name:`compactY`,view:ne,type:`f32`,access:`read`},{name:`pairs`,view:B,type:`f32`,access:`read_write`}],invocationCount:h,body:`pairs[pairsOffset + 2u * index] = compactX[compactXOffset + index];
  pairs[pairsOffset + 2u * index + 1u] = compactY[compactYOffset + index];`})),F.push(x(e,{id:`${t}-count-validity`,operation:O,variant:`count-validity`,bindings:[{name:`parameters`,view:o,type:`u32`,access:`read`},{name:`overflow`,view:n.output.status.overflow,type:`atomic<u32>`,access:`read_write`}],invocationCount:1,declarations:d(f),body:`if (!isPermutationCountValid()) {
    atomicMax(&overflow[overflowOffset], 1u);
  }`}),x(e,{id:`${t}-observed`,operation:O,variant:`observed`,bindings:[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`weights`,view:R,type:`f32`,access:`read`},{name:`rowPositions`,view:I,type:`u32`,access:`read`},V,{name:`observedRaw`,view:z,type:`f32`,access:`read_write`},{name:`overflow`,view:n.output.status.overflow,type:`atomic<u32>`,access:`read_write`},...n.output.quadrants?[{name:`quadrants`,view:n.output.quadrants,type:`u32`,access:`read_write`}]:[]],invocationCount:h,declarations:`${H}
${U}`,body:`var raw = getQuietNaN(index);
  var quadrant = 0u;
  let position = rowPositions[rowPositionsOffset + index];
  if (position != INVALID_POSITION) {
    let fx = getX(position);
    let fy = getY(position);
    var lagA = 0.0;
    var lagB = 0.0;
    var neighborCount = 0u;
    ${G(`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
    let dx = getX(neighborPosition);
    let dy = getY(neighborPosition);
    lagA += ${W} * getDrawnA(fx, fy, dx, dy);
    ${N.drawnB?`lagB += ${W} * getDrawnB(fx, fy, dx, dy);`:``}
    neighborCount++;`)}
    if (neighborCount > MAXIMUM_NEIGHBORS) {
      atomicMax(&overflow[overflowOffset], 1u);
    } else if (neighborCount > 0u || ${D}) {
      // A kept island has a zero lag: the formula value, never tested.
      raw = getLocalTerm(fx, fy, lagA, lagB);
      // esda Moran_Local_BV.q (PySAL scheme): 1 HH, 2 LH, 3 LL, 4 HL in (x_i, lag of y).
      quadrant = select(select(3u, 4u, fx > 0.0), select(2u, 1u, fx > 0.0), lagA > 0.0);
    }
  }
  ${n.output.quadrants?`quadrants[quadrantsOffset + index] = quadrant;`:``}
  observedRaw[observedRawOffset + index] = raw;`}),x(e,{id:`${t}-permute`,operation:O,variant:`permute`,bindings:[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`weights`,view:R,type:`f32`,access:`read`},{name:`rowPositions`,view:I,type:`u32`,access:`read`},V,{name:`parameters`,view:o,type:`u32`,access:`read`},{name:`observedRaw`,view:z,type:`f32`,access:`read`},{name:`exceedances`,view:J??n.output.exceedances,type:`u32`,access:`read_write`}],invocationCount:h,...ee(K)})),J&&Y&&F.push(x(e,{id:`${t}-split-counts`,operation:O,variant:`split-counts`,bindings:[{name:`permuteResults`,view:J,type:`u32`,access:`read`},{name:`exceedances`,view:n.output.exceedances,type:`u32`,access:`read_write`},{name:`permutationCounts`,view:Y,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`const ROWS: u32 = ${h}u;`,body:`exceedances[exceedancesOffset + index] = permuteResults[permuteResultsOffset + index];
  permutationCounts[permutationCountsOffset + index] = permuteResults[permuteResultsOffset + ROWS + index];`}));let X=n.output.observed??(n.output.classes?v(e,`${t}-observed-geary`,`float32`,h):void 0);if(X){let n=`totals[totalsOffset]`,r=`totals[totalsOffset + ${a.sumSquares}u]`,i=`totals[totalsOffset + ${a.sumSquaresY}u]`,o={localMoran:`(${n} - 1.0) * raw / ${r}`,localGeary:`raw * ${n} / ${r}`,localG:`raw / (totals[totalsOffset + 1u] - compactX[compactXOffset + position])`,localGStar:`raw / totals[totalsOffset + 1u]`,localBivariateMoran:`(${n} - 1.0) * raw / sqrt(${r} * ${i})`,localLeeL:`${n} * raw / sqrt(${r} * ${i})`,localJoinCount:`raw`,localBivariateJoinCount:`raw`}[l];F.push(x(e,{id:`${t}-observed-statistic`,operation:O,variant:`observed-statistic`,bindings:[{name:`observedRaw`,view:z,type:`f32`,access:`read`},{name:`rowPositions`,view:I,type:`u32`,access:`read`},{name:`compactX`,view:L,type:`f32`,access:`read`},{name:`totals`,view:re,type:`f32`,access:`read`},{name:`observed`,view:X,type:`f32`,access:`read_write`}],invocationCount:h,declarations:H,body:`let raw = observedRaw[observedRawOffset + index];
  let position = min(rowPositions[rowPositionsOffset + index], ROWS - 1u);
  let scaled = ${o};
  // Canonical +0: a zero lag times a negative deviation is -0. Tested on the bits because a
  // floating-point compare may be folded by the shader compiler, which treats -0 as 0.
  let scaledBits = bitcast<u32>(scaled);
  let canonical = bitcast<f32>(select(scaledBits, 0u, (scaledBits & 0x7fffffffu) == 0u));
  observed[observedOffset + index] = select(getQuietNaN(index), canonical, isFiniteFloat(raw));`}))}let ie=k!==`none`||n.output.adjustedPValues!==void 0,Z=[{name:`exceedances`,view:n.output.exceedances,type:`u32`,access:`read`},{name:`parameters`,view:o,type:`u32`,access:`read`},{name:`pseudoPValues`,view:n.output.pseudoPValues,type:`f32`,access:`read_write`}];Y&&Z.push({name:`permutationCounts`,view:Y,type:`u32`,access:`read`});let Q=E?`select(min(${_}u * (count + 1u), readValidPermutationCount() + 1u), min(${_}u * STOP_AFTER_EXCEEDANCES, permutationCounts[permutationCountsOffset + index]), count == STOP_AFTER_EXCEEDANCES)`:`min(${_}u * (count + 1u), readValidPermutationCount() + 1u)`,$=E?`select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == STOP_AFTER_EXCEEDANCES)`:`readValidPermutationCount() + 1u`,ae=`significant[significantOffset + index] = select(0u, 1u, tested && f32(${Q}) <= readSignificanceLevel() * f32(${$}));`;n.output.significant&&k===`none`&&Z.push({name:`significant`,view:n.output.significant,type:`u32`,access:`read_write`}),F.push(x(e,{id:`${t}-classify`,operation:O,variant:`classify`,bindings:Z,invocationCount:h,declarations:`${d(f)}
const NOT_TESTED: u32 = ${A}u;${E?`\nconst STOP_AFTER_EXCEEDANCES: u32 = ${y}u;`:``}`,body:`let count = exceedances[exceedancesOffset + index];
  let tested = count != NOT_TESTED;
  let pValue = f32(${Q}) / f32(${$});
  pseudoPValues[pseudoPValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), pValue, tested);
  ${n.output.significant&&k===`none`?ae:``}`}));let oe;if(ie){let r=n.output.adjustedPValues??v(e,`${t}-adjusted-p`,`float32`,h);oe=k===`none`?void 0:r,F.push(...new T({id:`${t}-multiple-testing`,pValues:n.output.pseudoPValues,method:k,output:{adjustedPValues:r}}).getCommandNodes(e)),n.output.significant&&F.push(x(e,{id:`${t}-significant`,operation:O,variant:`significant`,bindings:[{name:`adjustedPValues`,view:r,type:`f32`,access:`read`},{name:`parameters`,view:o,type:`u32`,access:`read`},{name:`significant`,view:n.output.significant,type:`u32`,access:`read_write`}],invocationCount:h,declarations:d(f),body:`let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  significant[significantOffset + index] = select(0u, 1u, adjusted <= readSignificanceLevel());`}))}return n.output.classes&&X&&F.push(...this.getGearyClassificationNodes(e,X,oe,L,I,Y)),F}getGearyClassificationNodes(e,n,r,a,o,c){let{id:l,props:m}=this,h=m.values.length,g=m.output.classes,_=p(i(l,m.alternative)),y=v(e,`${l}-geary-mean-matrix`,`float32`,2*h),b=v(e,`${l}-geary-mean-sums`,`float32`,2),S=m.stopAfterExceedances,C=S===void 0?`min(${_}u * (count + 1u), readValidPermutationCount() + 1u)`:`select(min(${_}u * (count + 1u), readValidPermutationCount() + 1u), min(${_}u * ${S}u, permutationCounts[permutationCountsOffset + index]), count == ${S}u)`,w=S===void 0?`readValidPermutationCount() + 1u`:`select(readValidPermutationCount() + 1u, permutationCounts[permutationCountsOffset + index], count == ${S}u)`,T=r?`adjustedPValues[adjustedPValuesOffset + index] <= readSignificanceLevel()`:`count != NOT_TESTED && f32(${C}) <= readSignificanceLevel() * f32(${w})`;return[x(e,{id:`${l}-geary-mean-inputs`,operation:O,variant:`geary-mean-inputs`,bindings:[{name:`observedView`,view:n,type:`f32`,access:`read`},{name:`matrix`,view:y,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const ROWS: u32 = ${h}u;
${s}`,body:`let value = observedView[observedViewOffset + index];
  let finite = isFiniteFloat(value);
  matrix[matrixOffset + index] = select(0.0, value, finite);
  matrix[matrixOffset + ROWS + index] = select(0.0, 1.0, finite);`}),...t(e,{id:`${l}-geary-mean`,operation:O,matrix:y,rows:h,blockCount:Math.ceil(h/f),first:0,count:2,totals:b}),x(e,{id:`${l}-geary-classes`,operation:O,variant:r?`geary-classes-adjusted`:`geary-classes`,bindings:[{name:`observedView`,view:n,type:`f32`,access:`read`},{name:`compactX`,view:a,type:`f32`,access:`read`},{name:`rowPositions`,view:o,type:`u32`,access:`read`},{name:`sums`,view:b,type:`f32`,access:`read`},{name:`parameters`,view:m.parameters,type:`u32`,access:`read`},r?{name:`adjustedPValues`,view:r,type:`f32`,access:`read`}:{name:`exceedances`,view:m.output.exceedances,type:`u32`,access:`read`},...!r&&c?[{name:`permutationCounts`,view:c,type:`u32`,access:`read`}]:[],{name:`classes`,view:g,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`${d(m.maximumPermutations)}
const INVALID_POSITION: u32 = ${u}u;
const NOT_TESTED: u32 = ${A}u;
${s}`,body:`let value = observedView[observedViewOffset + index];
  let position = rowPositions[rowPositionsOffset + index];
  let mean = sums[sumsOffset] / sums[sumsOffset + 1u];
  ${r?``:`let count = exceedances[exceedancesOffset + index];`}
  var code = 0u;
  if (isFiniteFloat(value) && position != INVALID_POSITION && ${r?`isFiniteFloat(adjustedPValues[adjustedPValuesOffset + index]) && `:`count != NOT_TESTED && `}isFiniteFloat(mean)) {
    if (!(${T})) {
      code = ${j.notSignificant}u;
    } else if (value < mean) {
      let focus = compactX[compactXOffset + position];
      code = select(select(0u, ${j.lowLow}u, focus < 0.0), ${j.highHigh}u, focus > 0.0);
    } else if (value > mean) {
      code = ${j.negativeAssociation}u;
    }
  }
  classes[classesOffset + index] = code;`})]}};export{N as t};