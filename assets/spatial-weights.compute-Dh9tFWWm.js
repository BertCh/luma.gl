import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{_ as e}from"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,d as r,h as i,t as a,w as o,y as s}from"./resources-CwvhEOBe.js";import{t as c}from"./gpu-scan-BuGhTLDQ.js";import{t as l}from"./gpu-sort-Bl3Dztdb.js";import{t as u}from"./gpu-reduction-BD8l40jM.js";import{t as d}from"./gpu-readback-ring-BguS3CAS.js";import{l as f,n as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as h}from"./sorted-segment-sums-DwGkmbx2.js";import{a as g,c as _,i as v,l as y,o as b,r as x,t as S}from"./spatial-weights-D4bd9AQ6.js";import{t as C}from"./scan-total-BdolCtK7.js";import{n as w,t as T}from"./float32-values-C3OXugIo.js";import{t as E}from"./gpu-spatial-lag-Q5OutSKD.js";import{t as D}from"./gpu-lattice-weights-DjJ2AdyH.js";import{t as O}from"./slot-grouping-DuACcF2F.js";import{t as k}from"./gpu-spatial-weights-transpose-DBog2XI3.js";import{t as A}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";import{H as j,a as ee,c as M,o as N,s as P}from"./index-Bjk18V2I.js";import{t as F}from"./graph-buffers-B4_80RlX.js";import{t as I}from"./graph-swap-41Ru7LVu.js";import{t as L}from"./summary-reader-CsgWxENf.js";import{t as R}from"./mode-kernels-Dxm2GWvI.js";import{n as te}from"./polygon-mesh-48KB9j_1.js";import"./anchors-D9E9ucGL.js";import{d as ne,g as re}from"./gazetteer-OhisynFF.js";import{a as ie,l as ae,x as z}from"./class-table-BVEo3-Vs.js";import{a as oe,h as se,m as ce}from"./hue-registry-4FjIJ9xz.js";import{c as le,i as ue,n as B,r as de}from"./live-text-83XBWyr1.js";import{i as fe}from"./picking-DRmnefrA.js";import{r as pe,t as me}from"./segments-BtINkq-i.js";import"./b6-polygons-Bf2djBQx.js";import{f as he,l as ge,o as _e}from"./hot-spots.style-DcEoMUmX.js";import{a as ve}from"./b4-format-Cgp0VQWI.js";import{n as ye,r as be,t as xe}from"./b4-weights-core-mAiwYc53.js";import{n as V,r as Se}from"./b4-geography-Dd13GyrB.js";import{t as Ce}from"./b4-layers-BaIKQJXr.js";import{c as we,d as H,f as Te,g as Ee,h as De,m as Oe,n as ke,o as Ae,p as je,r as Me,s as Ne,t as Pe,u as Fe}from"./spatial-weights.charts-DNiAaRWy.js";var Ie=`0xffffffffu`;function U(e,t){let{id:n,operation:r,rows:i,capacity:a}=t,s=o(e,`${n}-counts`,`uint32`,i),l=o(e,`${n}-starts`,`uint32`,i),u=o(e,`${n}-scan-total`,`uint32`,1),d=`const ROWS: u32 = ${i}u;
const CAPACITY: u32 = ${a}u;
const NO_SLOT: u32 = ${Ie};`,f=`${d}\n${t.declarations??``}`,p=[];p.push(m(e,{id:`${n}-counts`,operation:r,variant:`counts`,bindings:[...t.bindings,{name:`counts`,view:s,type:`u32`,access:`read_write`}],invocationCount:i,declarations:f,body:`var count = 0u;
  ${t.visit(`count++;`)}
  counts[countsOffset + index] = count;`})),p.push(...new c({id:`${n}-scan`,input:s,output:l,mode:`exclusive`}).getCommandNodes(e)),p.push(...C(e,{id:n,operation:r,counts:s,offsets:l,rowCount:i,total:u}),m(e,{id:`${n}-offsets`,operation:r,variant:`offsets`,bindings:[{name:`scanTotal`,view:u,type:`u32`,access:`read`},{name:`starts`,view:l,type:`u32`,access:`read`},{name:`outOffsets`,view:t.targetOffsets,type:`u32`,access:`read_write`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`},...t.totalNeighbors?[{name:`totalNeighbors`,view:t.totalNeighbors,type:`u32`,access:`read_write`}]:[]],invocationCount:i+1,declarations:d,body:`if (index < ROWS) {
    outOffsets[outOffsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    outOffsets[outOffsetsOffset + ROWS] = min(total, CAPACITY);
    ${t.initializesOverflow?`overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);`:`if (total > CAPACITY) {
      overflow[overflowOffset] = 1u;
    }`}
    ${t.totalNeighbors?`totalNeighbors[totalNeighborsOffset] = total;`:``}
  }`}));for(let a of t.attributes)p.push(m(e,{id:`${n}-emit-${a.name}`,operation:r,variant:`emit-${a.name}`,bindings:[...t.bindings,...a.bindings??[],{name:`outOffsets`,view:t.targetOffsets,type:`u32`,access:`read`},{name:`outValues`,view:a.output,type:a.type,access:`read_write`}],invocationCount:i,declarations:`${f}\n${a.declarations??``}`,body:`var next = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  ${t.visit(`if (next < end) {
    outValues[outValuesOffset + next] = ${a.value};
    next++;
  }`)}`}));if(t.sortRows){let a=t.attributes[0];p.push(m(e,{id:`${n}-sort-rows`,operation:r,variant:`sort-rows`,bindings:[{name:`outOffsets`,view:t.targetOffsets,type:`u32`,access:`read`},{name:`outValues`,view:a.output,type:`u32`,access:`read_write`}],invocationCount:i,body:`let begin = outOffsets[outOffsetsOffset + index];
  let end = outOffsets[outOffsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let key = outValues[outValuesOffset + slot];
    var position = slot;
    while (position > begin && outValues[outValuesOffset + position - 1u] > key) {
      outValues[outValuesOffset + position] = outValues[outValuesOffset + position - 1u];
      position--;
    }
    outValues[outValuesOffset + position] = key;
  }`}))}return p}function Le(e,t){return`fn ${e}(begin: u32, end: u32, value: u32) -> bool {
  var low = begin;
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] < value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low < end && ${t}[${t}Offset + low] == value;
}`}function Re(e,t){return`var leftNext = leftOffsets[leftOffsetsOffset + index];
  let leftEnd = leftOffsets[leftOffsetsOffset + index + 1u];
  var rightNext = rightOffsets[rightOffsetsOffset + index];
  let rightEnd = rightOffsets[rightOffsetsOffset + index + 1u];
  loop {
    let hasLeft = leftNext < leftEnd;
    let hasRight = rightNext < rightEnd;
    if (!hasLeft && !hasRight) {
      break;
    }
    let leftId = select(NO_SLOT, leftNeighbors[leftNeighborsOffset + leftNext], hasLeft);
    let rightId = select(NO_SLOT, rightNeighbors[rightNeighborsOffset + rightNext], hasRight);
    let id = min(leftId, rightId);
    var leftSlot = NO_SLOT;
    var rightSlot = NO_SLOT;
    if (leftId == id) {
      leftSlot = leftNext;
      leftNext++;
    }
    if (rightId == id) {
      rightSlot = rightNext;
      rightNext++;
    }
    if (${{union:`true`,intersection:`leftSlot != NO_SLOT && rightSlot != NO_SLOT`,difference:`leftSlot != NO_SLOT && rightSlot == NO_SLOT`,symmetricDifference:`(leftSlot == NO_SLOT) != (rightSlot == NO_SLOT)`}[e]}) {
      ${t}
    }
  }`}var W=`GPUSpatialWeightsAlgebra`,ze={left:`leftValue`,right:`rightValue`,sum:`leftValue + rightValue`,min:`min(leftValue, rightValue)`,max:`max(leftValue, rightValue)`,product:`leftValue * rightValue`,binary:`1.0`};function Be(...e){let t=e[0]??`unknown`;return(t===`raw`||t===`binary`)&&e.every(e=>e===t)?t:`unknown`}function G(...e){return e.every(e=>e.information?.symmetric===!0)?!0:`unknown`}function Ve(e,t){let n={source:W,criterion:e.operation};switch(e.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:{let r=e.operation===`difference`?[e.left]:[e.left,e.right];return{rowCount:t,transform:e.weightRule===`binary`?`binary`:e.weightRule===`sum`?Be(...r.map(e=>e.information?.transform===`binary`?`unknown`:e.information?.transform)):Be(...r.map(e=>e.information?.transform)),symmetric:G(e.left,e.right),hasSelfWeights:!1,provenance:n}}case`higherOrder`:return{rowCount:t,transform:`binary`,symmetric:G(e.weights),hasSelfWeights:!1,provenance:{...n,order:e.order}};case`selfWeight`:return{rowCount:t,transform:e.weights.information?.transform===`raw`?`raw`:`unknown`,symmetric:G(e.weights),hasSelfWeights:!0,provenance:n};case`subgraph`:return{rowCount:t,transform:Be(e.weights.information?.transform),symmetric:G(e.weights),hasSelfWeights:e.weights.information?.hasSelfWeights,provenance:n};default:return{rowCount:t,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:n}}}var K=class{id;props;outputWeights;constructor(t){let n=t.id??`spatial-weights-algebra`;this.id=n,this.props=t;let a=y(n,t.output),o=this.getInputWeights(),c;for(let t of o)c&&t.spatialContext&&e(n,c,t.spatialContext),c??=t.spatialContext;let l=(e,t)=>{if(_(n,e,t)!==a)throw Error(`${n} ${t}.offsets length must equal output.offsets length`)};switch(t.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:if(l(t.left,`left`),l(t.right,`right`),t.weightRule&&!(t.weightRule in ze))throw Error(`${n} unknown weightRule ${t.weightRule}`);if(t.output.distances){let e=t.operation===`union`||t.operation===`symmetricDifference`;if(!t.left.distances||e&&!t.right.distances)throw Error(`${n} output.distances needs operand distances`)}break;case`higherOrder`:l(t.weights,`weights`),r(n,`order`,t.order,1,32),t.workCapacity!==void 0&&r(n,`workCapacity`,t.workCapacity);break;case`selfWeight`:if(l(t.weights,`weights`),typeof t.selfWeight==`number`){if(!(Number.isFinite(t.selfWeight)&&t.selfWeight>=0))throw Error(`${n} selfWeight must be a non-negative finite number`)}else if(t.selfWeight.length!==a)throw Error(`${n} selfWeight view length must equal the row count`);break;case`subgraph`:if(l(t.weights,`weights`),s(t.mask,[`uint32`],`${n} mask`),t.mask.length!==a)throw Error(`${n} mask length must equal the row count`);break;case`block`:if(s(t.groupIds,[`uint32`],`${n} groupIds`),t.groupIds.length!==a)throw Error(`${n} groupIds length must equal the row count`);r(n,`groupCount`,t.groupCount);break;default:throw Error(`${n} unknown operation`)}if(t.output.distances&&(t.operation===`higherOrder`||t.operation===`block`))throw Error(`${n} '${t.operation}' has no distances: omit output.distances`);if(t.output.distances&&(t.operation===`selfWeight`||t.operation===`subgraph`)&&!t.weights.distances)throw Error(`${n} output.distances requires weights.distances`);this.outputWeights=S(t.output,{spatialContext:c,information:Ve(t,a),status:{...t.output.status,requiredCount:t.operation===`higherOrder`&&t.order>1?void 0:t.output.status.requiredCount}}),i(n,[...g(t.output),...v(t.output)],[...o.flatMap(g),t.operation===`subgraph`?t.mask:void 0,t.operation===`block`?t.groupIds:void 0,t.operation===`selfWeight`&&typeof t.selfWeight!=`number`?t.selfWeight:void 0])}getInputWeights(){let{props:e}=this;switch(e.operation){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:return[e.left,e.right];case`higherOrder`:case`selfWeight`:case`subgraph`:return[e.weights];default:return[]}}getCommandNodes(e){let{props:t,id:r}=this;switch(n(r,e,[...this.getInputWeights().flatMap(g),...g(t.output),...v(t.output),t.operation===`subgraph`?t.mask:void 0,t.operation===`block`?t.groupIds:void 0,t.operation===`selfWeight`&&typeof t.selfWeight!=`number`?t.selfWeight:void 0]),t.operation){case`higherOrder`:return this.getHigherOrderNodes(e,t);case`selfWeight`:return this.getSelfWeightNodes(e,t);case`subgraph`:return this.getSubgraphNodes(e,t);case`block`:return this.getBlockNodes(e,t);default:return this.getBinaryNodes(e,t)}}getBinaryNodes(e,t){let{id:n}=this,{left:r,right:i,output:a,operation:o}=t,s=a.offsets.length-1,c=t.weightRule??`left`,l=[{name:`leftOffsets`,view:r.offsets,type:`u32`,access:`read`},{name:`leftNeighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`rightOffsets`,view:i.offsets,type:`u32`,access:`read`},{name:`rightNeighbors`,view:i.neighbors,type:`u32`,access:`read`}],u=[{name:`neighbors`,output:a.neighbors,type:`u32`,value:`id`},{name:`weights`,output:a.weights,type:`f32`,value:`combineWeight(leftSlot, rightSlot)`,bindings:[{name:`leftWeights`,view:r.weights,type:`f32`,access:`read`},{name:`rightWeights`,view:i.weights,type:`f32`,access:`read`}],declarations:`fn combineWeight(leftSlot: u32, rightSlot: u32) -> f32 {
  ${c===`binary`?`return 1.0;`:``}
  if (rightSlot == NO_SLOT) {
    return leftWeights[leftWeightsOffset + leftSlot];
  }
  if (leftSlot == NO_SLOT) {
    return rightWeights[rightWeightsOffset + rightSlot];
  }
  let leftValue = leftWeights[leftWeightsOffset + leftSlot];
  let rightValue = rightWeights[rightWeightsOffset + rightSlot];
  return ${ze[c]};
}`}];if(a.distances){let e=i.distances&&o!==`intersection`&&o!==`difference`;u.push({name:`distances`,output:a.distances,type:`f32`,value:`pickDistance(leftSlot, rightSlot)`,bindings:[{name:`leftDistances`,view:r.distances,type:`f32`,access:`read`},...e?[{name:`rightDistances`,view:i.distances,type:`f32`,access:`read`}]:[]],declarations:`fn pickDistance(leftSlot: u32, rightSlot: u32) -> f32 {
  if (leftSlot != NO_SLOT) {
    return leftDistances[leftDistancesOffset + leftSlot];
  }
  ${e?`return rightDistances[rightDistancesOffset + rightSlot];`:`return 0.0;`}
}`})}return U(e,{id:n,operation:W,rows:s,targetOffsets:a.offsets,capacity:a.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:!0,totalNeighbors:t.output.status.requiredCount,bindings:l,visit:e=>Re(o,e),attributes:u})}getHigherOrderNodes(e,t){let{id:n}=this,{weights:r,output:i,order:a}=t,s=t.cumulative??!1,c=i.offsets.length-1,l=t.workCapacity??i.neighbors.length,u=[],d=t=>({offsets:o(e,`${n}-${t}-offsets`,`uint32`,c+1),neighbors:o(e,`${n}-${t}-neighbors`,`uint32`,l)}),f={offsets:i.offsets,neighbors:i.neighbors},p=!0,m=(e,r,i)=>{let a={id:`${n}-${e}`,operation:W,rows:c,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:p,totalNeighbors:i?t.output.status.requiredCount:void 0,attributes:[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`}]};return p=!1,a},h=(e,t)=>({name:e,view:t,type:`u32`,access:`read`}),g=a===1?f:d(`exact-1`);u.push(...U(e,{...m(`exact-1`,g,a===1),bindings:[h(`baseOffsets`,r.offsets),h(`baseNeighbors`,r.neighbors)],visit:e=>`for (var slot = baseOffsets[baseOffsetsOffset + index]; slot < baseOffsets[baseOffsetsOffset + index + 1u]; slot++) {
    let id = baseNeighbors[baseNeighborsOffset + slot];
    if (id != index && id < ROWS) {
      ${e}
    }
  }`}));let _=g,v=g;for(let t=2;t<=a;t++){let n=t===a,i=n&&!s?f:d(`exact-${t}`);if(u.push(...U(e,{...m(`exact-${t}`,i,n&&!s),bindings:[h(`baseOffsets`,r.offsets),h(`baseNeighbors`,r.neighbors),h(`frontierOffsets`,_.offsets),h(`frontierNeighbors`,_.neighbors),h(`visitedOffsets`,v.offsets),h(`visitedNeighbors`,v.neighbors)],declarations:`${Le(`baseContains`,`baseNeighbors`)}
${Le(`visitedContains`,`visitedNeighbors`)}`,visit:e=>`let frontierBegin = frontierOffsets[frontierOffsetsOffset + index];
  let frontierEnd = frontierOffsets[frontierOffsetsOffset + index + 1u];
  let visitedBegin = visitedOffsets[visitedOffsetsOffset + index];
  let visitedEnd = visitedOffsets[visitedOffsetsOffset + index + 1u];
  for (var frontierSlot = frontierBegin; frontierSlot < frontierEnd; frontierSlot++) {
    let middle = frontierNeighbors[frontierNeighborsOffset + frontierSlot];
    if (middle >= ROWS) {
      continue;
    }
    for (var slot = baseOffsets[baseOffsetsOffset + middle]; slot < baseOffsets[baseOffsetsOffset + middle + 1u]; slot++) {
      let id = baseNeighbors[baseNeighborsOffset + slot];
      if (id == index || id >= ROWS || visitedContains(visitedBegin, visitedEnd, id)) {
        continue;
      }
      var seen = false;
      for (var earlier = frontierBegin; earlier < frontierSlot; earlier++) {
        let earlierMiddle = frontierNeighbors[frontierNeighborsOffset + earlier];
        if (earlierMiddle < ROWS && baseContains(baseOffsets[baseOffsetsOffset + earlierMiddle], baseOffsets[baseOffsetsOffset + earlierMiddle + 1u], id)) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        ${e}
      }
    }
  }`,sortRows:!0})),n&&!s){_=i;break}let o=n?f:d(`visited-${t}`);u.push(...U(e,{...m(`visited-${t}`,o,n),bindings:[h(`leftOffsets`,v.offsets),h(`leftNeighbors`,v.neighbors),h(`rightOffsets`,i.offsets),h(`rightNeighbors`,i.neighbors)],visit:e=>Re(`union`,e)})),_=i,v=o}return u.push(He(e,`${n}-weights`,i)),u}getSelfWeightNodes(e,t){let{weights:n,output:r,selfWeight:i}=t,a=r.offsets.length-1,o=typeof i==`number`?f(i):`selfWeights[selfWeightsOffset + index]`,s=[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`},{name:`weights`,output:r.weights,type:`f32`,value:`select(sourceWeights[sourceWeightsOffset + sourceSlot], ${o}, id == index)`,bindings:[{name:`sourceWeights`,view:n.weights,type:`f32`,access:`read`},...typeof i==`number`?[]:[{name:`selfWeights`,view:i,type:`f32`,access:`read`}]]}];return r.distances&&s.push({name:`distances`,output:r.distances,type:`f32`,value:`select(sourceDistances[sourceDistancesOffset + sourceSlot], 0.0, id == index)`,bindings:[{name:`sourceDistances`,view:n.distances,type:`f32`,access:`read`}]}),U(e,{id:this.id,operation:W,rows:a,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:!0,totalNeighbors:t.output.status.requiredCount,bindings:[{name:`sourceOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`sourceNeighbors`,view:n.neighbors,type:`u32`,access:`read`}],visit:e=>`var inserted = false;
  for (var slot = sourceOffsets[sourceOffsetsOffset + index]; slot < sourceOffsets[sourceOffsetsOffset + index + 1u]; slot++) {
    let neighbor = sourceNeighbors[sourceNeighborsOffset + slot];
    if (!inserted && neighbor >= index) {
      inserted = true;
      {
        let id = index;
        let sourceSlot = slot;
        ${e}
      }
      if (neighbor == index) {
        continue;
      }
    }
    {
      let id = neighbor;
      let sourceSlot = slot;
      ${e}
    }
  }
  if (!inserted) {
    let id = index;
    let sourceSlot = 0u;
    ${e}
  }`,attributes:s})}getSubgraphNodes(e,t){let{weights:n,output:r,mask:i}=t,a=r.offsets.length-1,o=[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`},{name:`weights`,output:r.weights,type:`f32`,value:`sourceWeights[sourceWeightsOffset + slot]`,bindings:[{name:`sourceWeights`,view:n.weights,type:`f32`,access:`read`}]}];return r.distances&&o.push({name:`distances`,output:r.distances,type:`f32`,value:`sourceDistances[sourceDistancesOffset + slot]`,bindings:[{name:`sourceDistances`,view:n.distances,type:`f32`,access:`read`}]}),U(e,{id:this.id,operation:W,rows:a,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:!0,totalNeighbors:t.output.status.requiredCount,bindings:[{name:`sourceOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`sourceNeighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`mask`,view:i,type:`u32`,access:`read`}],visit:e=>`if (mask[maskOffset + index] != 0u) {
    for (var slot = sourceOffsets[sourceOffsetsOffset + index]; slot < sourceOffsets[sourceOffsetsOffset + index + 1u]; slot++) {
      let id = sourceNeighbors[sourceNeighborsOffset + slot];
      if (id < ROWS && mask[maskOffset + id] != 0u) {
        ${e}
      }
    }
  }`,attributes:o})}getBlockNodes(e,t){let{id:n}=this,{output:r,groupIds:i,groupCount:a}=t,s=r.offsets.length-1,c=o(e,`${n}-sort-keys`,`uint32`,s),u=o(e,`${n}-sort-rows`,`uint32`,s),d=o(e,`${n}-sorted-keys`,`uint32`,s),f=o(e,`${n}-sorted-rows`,`uint32`,s);return[m(e,{id:`${n}-sort-prepare`,operation:W,variant:`sort-prepare`,bindings:[{name:`groupIds`,view:i,type:`u32`,access:`read`},{name:`sortKeys`,view:c,type:`u32`,access:`read_write`},{name:`sortRows`,view:u,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`const GROUP_COUNT: u32 = ${a}u;`,body:`sortKeys[sortKeysOffset + index] = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortRows[sortRowsOffset + index] = index;`}),...new l({id:`${n}-sort`,keys:c,values:u,outputKeys:d,outputValues:f,keyBits:h(a)}).getCommandNodes(e),...U(e,{id:n,operation:W,rows:s,targetOffsets:r.offsets,capacity:r.neighbors.length,overflow:t.output.status.overflow,initializesOverflow:!0,totalNeighbors:t.output.status.requiredCount,bindings:[{name:`groupIds`,view:i,type:`u32`,access:`read`},{name:`sortedKeys`,view:d,type:`u32`,access:`read`},{name:`sortedRows`,view:f,type:`u32`,access:`read`}],declarations:`const GROUP_COUNT: u32 = ${a}u;
fn lowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ${s}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,visit:e=>`let group = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  if (group < GROUP_COUNT) {
    let groupEnd = lowerBound(group + 1u);
    for (var position = lowerBound(group); position < groupEnd; position++) {
      let id = sortedRows[sortedRowsOffset + position];
      if (id != index) {
        ${e}
      }
    }
  }`,attributes:[{name:`neighbors`,output:r.neighbors,type:`u32`,value:`id`},{name:`weights`,output:r.weights,type:`f32`,value:`1.0`}]})]}};function He(e,t,n){return m(e,{id:t,operation:W,variant:`weights`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read_write`}],invocationCount:n.offsets.length-1,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    weights[weightsOffset + slot] = 1.0;
  }`})}var q=`GPUSpatialWeightsSummary`,Ue=4096,J=256,Y={statistics:{s0:0,s1:1,s2:2,length:3},counts:{slots:0,asymmetricSlots:1,isolates:2,minimumCardinality:3,maximumCardinality:4,emptyRows:5,length:6}},We=class{id;props;constructor(e){let t=e.id??`spatial-weights-summary`;this.id=t,this.props=e;let n=_(t,e.weights);if(s(e.output.statistics,[`float32`],`${t} statistics`),e.output.statistics.length<Y.statistics.length)throw Error(`${t} statistics must hold at least 3 values`);if(s(e.output.counts,[`uint32`],`${t} counts`),e.output.counts.length<Y.counts.length)throw Error(`${t} counts must hold at least 6 values`);if(e.output.cardinality&&(s(e.output.cardinality,[`uint32`],`${t} cardinality`),e.output.cardinality.length!==n))throw Error(`${t} cardinality length must equal the weights row count`);if(e.symmetryTolerance!==void 0&&!(Number.isFinite(e.symmetryTolerance)&&e.symmetryTolerance>=0))throw Error(`${t} symmetryTolerance must be finite and >= 0`);i(t,[e.output.statistics,e.output.counts,e.output.cardinality],[e.weights.offsets,e.weights.neighbors,e.weights.weights])}getCommandNodes(e){let{id:t,props:r}=this,{weights:i}=r,{statistics:a,counts:s}=r.output;n(t,e,[i.offsets,i.neighbors,i.weights,a,s,r.output.cardinality]);let c=i.offsets.length-1,l=i.neighbors.length,u=f(r.symmetryTolerance??0),d=(n,r,i)=>o(e,`${t}-${n}`,r,i),p=d(`slot-keys`,`uint32`,l),h=d(`slot-indices`,`uint32`,l),g=d(`column-offsets`,`uint32`,c+1),_=d(`column-sums`,`float32`,c),v=d(`row-sums`,`float32`,c),y=d(`s1-partials`,`float32`,c),b=d(`s2-partials`,`float32`,c),S=d(`asymmetric`,`uint32`,c),C=d(`isolates`,`uint32`,c),w=r.output.cardinality??d(`cardinality`,`uint32`,c),T=`const ROWS: u32 = ${c}u;`,E=O(e,{id:`${t}-columns`,operation:q,slotKeys:p,slotIndices:h,columns:c,columnOffsets:g}),D=[m(e,{id:`${t}-slot-keys`,operation:q,variant:`slot-keys`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`slotKeys`,view:p,type:`u32`,access:`read_write`},{name:`slotIndices`,view:h,type:`u32`,access:`read_write`}],invocationCount:l,declarations:T,body:`let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`}),...E.nodes,m(e,{id:`${t}-column-sums`,operation:q,variant:`column-sums`,bindings:[{name:`columnOffsets`,view:g,type:`u32`,access:`read`},{name:`sortedSlots`,view:E.sortedSlots,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`columnSums`,view:_,type:`f32`,access:`read_write`}],invocationCount:c,body:`var sum = 0.0;
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`}),m(e,{id:`${t}-rows`,operation:q,variant:`rows`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`rowSums`,view:v,type:`f32`,access:`read_write`},{name:`s1Partials`,view:y,type:`f32`,access:`read_write`},{name:`asymmetric`,view:S,type:`u32`,access:`read_write`},{name:`cardinality`,view:w,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${T}
const CAPACITY: u32 = ${l}u;
const TOLERANCE: f32 = ${u};
const MISSING: f32 = -1.0;

// Weight w_row,column by binary search of the ascending neighbor IDs of row, or MISSING.
fn findWeight(row: u32, column: u32) -> f32 {
  var low = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
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
    return weights[weightsOffset + low];
  }
  return MISSING;
}`,body:`var sum = 0.0;
  var s1 = 0.0;
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    sum += weight;
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor, index);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > TOLERANCE) {
        unmatched++;
      }
    }
  }
  // Cardinality counts active slots only (the shared neighbor definition).
  var activeCount = 0u;
  ${x(`true`,`activeCount++;`)}
  cardinality[cardinalityOffset + index] = activeCount;
  rowSums[rowSumsOffset + index] = sum;
  s1Partials[s1PartialsOffset + index] = s1;
  asymmetric[asymmetricOffset + index] = unmatched;`}),m(e,{id:`${t}-rows-final`,operation:q,variant:`rows-final`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`rowSums`,view:v,type:`f32`,access:`read`},{name:`columnSums`,view:_,type:`f32`,access:`read`},{name:`s2Partials`,view:b,type:`f32`,access:`read_write`},{name:`cardinality`,view:w,type:`u32`,access:`read`},{name:`isolates`,view:C,type:`u32`,access:`read_write`}],invocationCount:c,body:`let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + index] + columnSums[columnSumsOffset + index];
  s2Partials[s2PartialsOffset + index] = total * total;
  // 0: has an active slot, 1: isolate that lists slots, 2: empty row (no slot at all).
  let isolate = cardinality[cardinalityOffset + index] == 0u;
  isolates[isolatesOffset + index] = select(0u, select(1u, 2u, count == 0u), isolate);`})],k=Math.ceil(c/Ue),A=d(`float-partials`,`float32`,3*k),j=d(`count-partials`,`uint32`,5*k);return D.push(m(e,{id:`${t}-blocks`,operation:q,variant:`blocks`,bindings:[{name:`rowSums`,view:v,type:`f32`,access:`read`},{name:`s1Partials`,view:y,type:`f32`,access:`read`},{name:`s2Partials`,view:b,type:`f32`,access:`read`},{name:`asymmetric`,view:S,type:`u32`,access:`read`},{name:`isolates`,view:C,type:`u32`,access:`read`},{name:`cardinality`,view:w,type:`u32`,access:`read`},{name:`floatPartials`,view:A,type:`f32`,access:`read_write`},{name:`countPartials`,view:j,type:`u32`,access:`read_write`}],workgroupSize:J,invocationCount:k*J,guardIndex:!1,declarations:`${T}
const BLOCK_ROWS: u32 = ${Ue}u;
const BLOCK_COUNT: u32 = ${k}u;
const WORKGROUP_SIZE: u32 = ${J}u;
var<workgroup> partialS0: array<f32, ${J}>;
var<workgroup> partialS1: array<f32, ${J}>;
var<workgroup> partialS2: array<f32, ${J}>;
var<workgroup> partialAsymmetric: array<u32, ${J}>;
var<workgroup> partialIsolates: array<u32, ${J}>;
var<workgroup> partialEmpty: array<u32, ${J}>;
var<workgroup> partialMinimum: array<u32, ${J}>;
var<workgroup> partialMaximum: array<u32, ${J}>;`,body:`let block = index / WORKGROUP_SIZE;
  let local = localInvocationIndex;
  let isInRange = index < INVOCATION_COUNT;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + local; row < end; row += WORKGROUP_SIZE) {
      s0 += rowSums[rowSumsOffset + row];
      s1 += s1Partials[s1PartialsOffset + row];
      s2 += s2Partials[s2PartialsOffset + row];
      asymmetricCount += asymmetric[asymmetricOffset + row];
      let isolateCode = isolates[isolatesOffset + row];
      isolateCount += select(0u, 1u, isolateCode != 0u);
      emptyCount += select(0u, 1u, isolateCode == 2u);
      let degree = cardinality[cardinalityOffset + row];
      minimum = min(minimum, degree);
      maximum = max(maximum, degree);
    }
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (isInRange && local == 0u) {
    floatPartials[floatPartialsOffset + block] = partialS0[0];
    floatPartials[floatPartialsOffset + BLOCK_COUNT + block] = partialS1[0];
    floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block] = partialS2[0];
    countPartials[countPartialsOffset + block] = partialAsymmetric[0];
    countPartials[countPartialsOffset + BLOCK_COUNT + block] = partialIsolates[0];
    countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block] = partialMinimum[0];
    countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block] = partialMaximum[0];
    countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block] = partialEmpty[0];
  }`}),m(e,{id:`${t}-totals`,operation:q,variant:`totals`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`floatPartials`,view:A,type:`f32`,access:`read`},{name:`countPartials`,view:j,type:`u32`,access:`read`},{name:`statistics`,view:a,type:`f32`,access:`read_write`},{name:`counts`,view:s,type:`u32`,access:`read_write`}],workgroupSize:J,invocationCount:J,guardIndex:!1,declarations:`${T}
const BLOCK_COUNT: u32 = ${k}u;
const WORKGROUP_SIZE: u32 = ${J}u;
var<workgroup> partialS0: array<f32, ${J}>;
var<workgroup> partialS1: array<f32, ${J}>;
var<workgroup> partialS2: array<f32, ${J}>;
var<workgroup> partialAsymmetric: array<u32, ${J}>;
var<workgroup> partialIsolates: array<u32, ${J}>;
var<workgroup> partialEmpty: array<u32, ${J}>;
var<workgroup> partialMinimum: array<u32, ${J}>;
var<workgroup> partialMaximum: array<u32, ${J}>;`,body:`let local = localInvocationIndex;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  for (var block = local; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    s0 += floatPartials[floatPartialsOffset + block];
    s1 += floatPartials[floatPartialsOffset + BLOCK_COUNT + block];
    s2 += floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block];
    asymmetricCount += countPartials[countPartialsOffset + block];
    isolateCount += countPartials[countPartialsOffset + BLOCK_COUNT + block];
    minimum = min(minimum, countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block]);
    maximum = max(maximum, countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block]);
    emptyCount += countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block];
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    statistics[statisticsOffset] = partialS0[0];
    statistics[statisticsOffset + 1u] = 0.5 * partialS1[0];
    statistics[statisticsOffset + 2u] = partialS2[0];
    counts[countsOffset] = offsets[offsetsOffset + ROWS];
    counts[countsOffset + 1u] = partialAsymmetric[0];
    counts[countsOffset + 2u] = partialIsolates[0];
    counts[countsOffset + 3u] = partialMinimum[0];
    counts[countsOffset + 4u] = partialMaximum[0];
    counts[countsOffset + 5u] = partialEmpty[0];
  }`})),D}},Ge=`GPUNeighborhoodSummary`,Ke=4294967295,qe=[`count`,`weightSum`,`sum`,`mean`,`min`,`max`,`standardDeviation`,`median`],Je=class{id;props;constructor(e){let t=e.id??`neighborhood-summary`;this.id=t,this.props=e;let n=_(t,e.weights),a=e.statistics??[];for(let e of a)if(!qe.includes(e))throw Error(`${t} unknown statistic ${e}`);if(new Set(a).size!==a.length)throw Error(`${t} statistics must not repeat`);let{output:o}=e,c=!!(o.modes||o.entropy);if(a.length===0&&!c)throw Error(`${t} requests no statistic`);if(a.length>0){if(!e.values||!o.table)throw Error(`${t} statistics require values and output.table`);if(w(e.values,`${t} values`)!==n)throw Error(`${t} values needs one row per weights row`);if(s(o.table,[`float32`],`${t} output.table`),o.table.length!==n*a.length)throw Error(`${t} output.table needs rows * statistics.length values`)}else if(o.table||e.values)throw Error(`${t} values and output.table require statistics`);let l=o.status?.overflow;if(a.includes(`median`)){if(!l)throw Error(`${t} median requires output.status.overflow`);if(s(l,[`uint32`],`${t} output.status.overflow`),l.length<1)throw Error(`${t} output.status.overflow needs one row`)}if(r(t,`maximumNeighbors`,e.maximumNeighbors??32,1,64),c){if(!e.categories)throw Error(`${t} modes and entropy require categories`);if(s(e.categories,[`uint32`],`${t} categories`),e.categories.length!==n)throw Error(`${t} categories needs one row per weights row`)}if(o.modes&&(s(o.modes,[`uint32`],`${t} output.modes`),o.modes.length!==n))throw Error(`${t} output.modes needs one row per weights row`);if(o.entropy&&(s(o.entropy,[`float32`],`${t} output.entropy`),o.entropy.length!==n))throw Error(`${t} output.entropy needs one row per weights row`);if(e.mask&&(s(e.mask,[`uint32`],`${t} mask`),e.mask.length!==n))throw Error(`${t} mask needs one row per weights row`);e.weights.status?.overflow&&s(e.weights.status.overflow,[`uint32`],`${t} weights.status.overflow`);let u=e.focalWeight??1;if(!Number.isFinite(u)||u<0)throw Error(`${t} focalWeight must be finite and non-negative`);i(t,[o.table,l,o.modes,o.entropy],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.status?.overflow,e.values,e.categories,e.mask])}getCommandNodes(e){let{id:t,props:r}=this,{weights:i,mask:a,output:o}=r;n(t,e,[i.offsets,i.neighbors,i.weights,i.status?.overflow,r.values,r.categories,a,o.table,o.status?.overflow,o.modes,o.entropy]);let s=i.offsets.length-1,c=r.statistics??[],l=!!r.includeFocal,u=i.status?.overflow,d=b(i),h=r.maximumNeighbors??32,g=`const ROWS: u32 = ${s}u;
const CAPACITY: u32 = ${i.neighbors.length}u;
const FOCAL_COUNT: u32 = ${l?1:0}u;
const FOCAL_WEIGHT: f32 = ${f(r.focalWeight??1)};
${A}
struct Member {
  row: u32,
  weight: f32,
  valid: bool
}
fn getMemberRange(row: u32) -> vec2<u32> {
  let begin = min(offsets[offsetsOffset + row], CAPACITY);
  let end = min(offsets[offsetsOffset + row + 1u], CAPACITY);
  return vec2<u32>(begin, max(end, begin));
}`,_=e=>a?`mask[maskOffset + ${e}] != 0u`:`true`,v=e=>`
fn getMember(row: u32, begin: u32, visit: u32, scale: f32) -> Member {
  var member = Member(row, FOCAL_WEIGHT, false);
  if (visit >= FOCAL_COUNT) {
    let slot = begin + visit - FOCAL_COUNT;
    member.row = neighbors[neighborsOffset + slot];
    member.weight = weights[weightsOffset + slot];
    // Only active slots are members: another row with a positive weight.
    if (member.row >= ROWS || member.row == row || !(member.weight > 0.0)) {
      return member;
    }
    member.weight = member.weight * scale;
  }
  member.valid = ${_(`member.row`)} && ${e(`member.row`)};
  return member;
}
// Reciprocal sum of the valid slot weights of a row-standardized row (1 for weights as given).
fn getRowScale(row: u32, begin: u32, visits: u32) -> f32 {
  ${d?`var sum = 0.0;
  for (var visit = FOCAL_COUNT; visit < visits; visit++) {
    let member = getMember(row, begin, visit, 1.0);
    if (member.valid) {
      sum += member.weight;
    }
  }
  return select(0.0, 1.0 / sum, sum > 0.0);`:`return 1.0;`}
}`,y=[];if(c.length>0){let n=T(e,`${t}-values`,r.values);y.push(...n.nodes);let l=n.view,d=o.table,f=c.length,b=e=>c.includes(e),x=b(`mean`)||b(`standardDeviation`),S=[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`values`,view:l,type:`f32`,access:`read`},...a?[{name:`mask`,view:a,type:`u32`,access:`read`}]:[],{name:`output`,view:d,type:`f32`,access:`read_write`},...u?[{name:`truncation`,view:u,type:`u32`,access:`read`}]:[]];b(`median`)&&(S.push({name:`overflow`,view:o.status.overflow,type:`atomic<u32>`,access:`read_write`}),y.push(p(e,{id:`${t}-overflow-clear`,operation:Ge,view:o.status.overflow,type:`u32`,value:`0u`,componentCount:1})));let C=e=>{switch(e){case`count`:return`f32(count)`;case`weightSum`:return`weightSum`;case`sum`:return`weightedSum`;case`mean`:return`select(nan, mean, weightSum > 0.0)`;case`min`:return`select(nan, minimum, count > 0u)`;case`max`:return`select(nan, maximum, count > 0u)`;case`standardDeviation`:return`select(nan, sqrt(squares / weightSum), weightSum > 0.0)`;default:return`median`}},w=c.map((e,t)=>`output[outputOffset + index * ${f}u + ${t}u] = ${e===`count`?`0.0`:`nan`};`).join(`
    `),E=c.map((e,t)=>`output[outputOffset + index * ${f}u + ${t}u] = nan;`).join(`
    `),D=c.map((e,t)=>`output[outputOffset + index * ${f}u + ${t}u] = ${C(e)};`).join(`
    `);y.push(m(e,{id:`${t}-numeric`,operation:Ge,variant:c.join(`-`),bindings:S,invocationCount:s,declarations:`${g}
const MAXIMUM_NEIGHBORS: u32 = ${h}u;
${v(e=>`isFiniteFloat(values[valuesOffset + ${e}])`)}`,body:`let nan = getQuietNaN(index);
  if (!(${_(`index`)})) {
    ${w}
    return;
  }
  ${u?`// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${E}
    return;
  }`:``}
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  var count = 0u;
  var weightSum = 0.0;
  var weightedSum = 0.0;
  var minimum = 3.0e38;
  var maximum = -3.0e38;
  ${b(`median`)?`var sorted: array<f32, ${h}>;`:``}
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (!member.valid) {
      continue;
    }
    let value = values[valuesOffset + member.row];
    weightSum += member.weight;
    weightedSum += member.weight * value;
    minimum = min(minimum, value);
    maximum = max(maximum, value);
    ${b(`median`)?`if (count < MAXIMUM_NEIGHBORS) {
      var position = count;
      loop {
        if (position == 0u || sorted[position - 1u] <= value) {
          break;
        }
        sorted[position] = sorted[position - 1u];
        position--;
      }
      sorted[position] = value;
    }`:``}
    count++;
  }
  ${x?`let mean = weightedSum / weightSum;`:``}
  var squares = 0.0;
  ${b(`standardDeviation`)?`for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid) {
      let deviation = values[valuesOffset + member.row] - mean;
      squares += member.weight * deviation * deviation;
    }
  }`:``}
  ${b(`median`)?`var median = nan;
  if (count > MAXIMUM_NEIGHBORS) {
    atomicMax(&overflow[overflowOffset], 1u);
  } else if (count > 0u) {
    let middle = count / 2u;
    median = select(0.5 * (sorted[middle - 1u] + sorted[middle]), sorted[middle], (count & 1u) == 1u);
  }`:``}
  ${D}`}))}if(o.modes||o.entropy){let n=[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`categories`,view:r.categories,type:`u32`,access:`read`},...a?[{name:`mask`,view:a,type:`u32`,access:`read`}]:[],...o.modes?[{name:`modes`,view:o.modes,type:`u32`,access:`read_write`}]:[],...o.entropy?[{name:`entropy`,view:o.entropy,type:`f32`,access:`read_write`}]:[],...u?[{name:`truncation`,view:u,type:`u32`,access:`read`}]:[]];y.push(m(e,{id:`${t}-categorical`,operation:Ge,variant:`categorical${o.modes?`-mode`:``}${o.entropy?`-entropy`:``}`,bindings:n,invocationCount:s,declarations:`${g}
const NO_MODE: u32 = ${Ke}u;
const MAXIMUM_NEIGHBORS: u32 = ${h}u;
${v(()=>`true`)}`,body:`let nan = getQuietNaN(index);
  if (!(${_(`index`)})) {
    ${o.modes?`modes[modesOffset + index] = NO_MODE;`:``}
    ${o.entropy?`entropy[entropyOffset + index] = nan;`:``}
    return;
  }
  ${u?`// Truncated weights: no statistic from them is trustworthy.
  if (truncation[truncationOffset] != 0u) {
    ${o.modes?`modes[modesOffset + index] = NO_MODE;`:``}
    ${o.entropy?`entropy[entropyOffset + index] = nan;`:``}
    return;
  }`:``}
  let range = getMemberRange(index);
  let visits = range.y - range.x + FOCAL_COUNT;
  let scale = getRowScale(index, range.x, visits);
  // Decode the members once. Up to MAXIMUM_NEIGHBORS of them are kept in private arrays so the
  // quadratic frequency scan below reads registers instead of re-decoding every member (four
  // global loads) for every pair; larger rows take the unbounded decoding scan.
  var memberCategories: array<u32, ${h}>;
  var memberWeights: array<f32, ${h}>;
  var memberCount = 0u;
  var weightSum = 0.0;
  for (var visit = 0u; visit < visits; visit++) {
    let member = getMember(index, range.x, visit, scale);
    if (member.valid) {
      weightSum += member.weight;
      if (memberCount < MAXIMUM_NEIGHBORS) {
        memberCategories[memberCount] = categories[categoriesOffset + member.row];
        memberWeights[memberCount] = member.weight;
      }
      memberCount++;
    }
  }
  var bestCategory = NO_MODE;
  var bestFrequency = -1.0;
  var entropyTotal = 0.0;
  if (memberCount <= MAXIMUM_NEIGHBORS) {
    for (var member = 0u; member < memberCount; member++) {
      let category = memberCategories[member];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < memberCount; other++) {
        if (memberCategories[other] == category) {
          if (other < member) {
            firstOccurrence = false;
            break;
          }
          frequency += memberWeights[other];
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  } else {
    for (var visit = 0u; visit < visits; visit++) {
      let member = getMember(index, range.x, visit, scale);
      if (!member.valid) {
        continue;
      }
      let category = categories[categoriesOffset + member.row];
      // Frequency of this category, summed in member order, at its first occurrence only.
      var frequency = 0.0;
      var firstOccurrence = true;
      for (var other = 0u; other < visits; other++) {
        let candidate = getMember(index, range.x, other, scale);
        if (candidate.valid && categories[categoriesOffset + candidate.row] == category) {
          if (other < visit) {
            firstOccurrence = false;
            break;
          }
          frequency += candidate.weight;
        }
      }
      if (!firstOccurrence) {
        continue;
      }
      if (frequency > bestFrequency || (frequency == bestFrequency && category < bestCategory)) {
        bestFrequency = frequency;
        bestCategory = category;
      }
      if (frequency > 0.0) {
        let share = frequency / weightSum;
        entropyTotal -= share * log(share);
      }
    }
  }
  ${o.modes?`modes[modesOffset + index] = bestCategory;`:``}
  ${o.entropy?`entropy[entropyOffset + index] = select(nan, entropyTotal, weightSum > 0.0);`:``}`}))}return y}},X=Y.counts.length,Ye=3+X+1+2+2+3+6,Z=`
  let total = offsets[offsetsOffset + ROW_COUNT];
  if (index >= total) {
    segments[segmentsOffset + index * 4u] = 0.0;
    segments[segmentsOffset + index * 4u + 1u] = 0.0;
    segments[segmentsOffset + index * 4u + 2u] = 0.0;
    segments[segmentsOffset + index * 4u + 3u] = 0.0;
    fade[fadeOffset + index] = 0.0;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  fade[fadeOffset + index] = 1.0;`,Xe=`
  let focus = min(focusRow[focusRowOffset], ROW_COUNT - 1u);
  let start = offsets[offsetsOffset + focus];
  let end = offsets[offsetsOffset + focus + 1u];
  if (index == 0u) {
    var weightSum = 0.0;
    for (var slot = start; slot < end; slot++) {
      weightSum += weights[weightsOffset + slot];
    }
    info[infoOffset] = f32(end - start);
    info[infoOffset + 1u] = weightSum;
  }
  let slot = start + index;
  var other = focus;
  if (slot < end) {
    other = neighbors[neighborsOffset + slot];
  }
  // Unused slots become NaN segments, which the segment layer skips.
  var nanBits = 0x7fc00000u;
  let hidden = bitcast<f32>(nanBits);
  segments[segmentsOffset + index * 4u] = select(hidden, positions[positionsOffset + focus * 2u], slot < end);
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + focus * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + other * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + other * 2u + 1u];
  neighborIds[neighborIdsOffset + index] = other;`;async function Q(e){let{device:n,resources:r,id:i,rowCount:a,slots:o,unionSlots:s,positions:c,values:l,csr:f}=e,p=(e,t)=>r.createBuffer(`${i}-${e}`,t),m={offsets:p(`transpose-offsets`,(a+1)*4),neighbors:p(`transpose-neighbors`,o*4),weights:p(`transpose-weights`,o*4)},h={offsets:p(`union-offsets`,(a+1)*4),neighbors:p(`union-neighbors`,s*4),weights:p(`union-weights`,s*4)},g={weights:{csr:f,slots:o},transpose:{csr:m,slots:o},union:{csr:h,slots:s}},_=p(`transpose-asymmetric`,4),v=p(`union-overflow`,4),y=p(`union-total`,4),b=e.producer.candidateOverflow??p(`no-candidate`,4),x=p(`lag`,a*4),S=p(`lag-extent`,8),C=p(`focus-row`,new Uint32Array([0])),w=e.summary?.statistics??p(`summary-statistics`,12),T=e.summary?.counts??p(`summary-counts`,X*4),D=e.summary?null:p(`cardinality`,a*4),O={weights:p(`focus-info-weights`,8),transpose:p(`focus-info-transpose`,8),union:p(`focus-info-union`,8)},A={weights:p(`focus-segments-weights`,2048),transpose:p(`focus-segments-transpose`,2048),union:p(`focus-segments-union`,2048)},j={weights:p(`focus-neighbors-weights`,512),transpose:p(`focus-neighbors-transpose`,512),union:p(`focus-neighbors-union`,512)},ee=r.track(new d(n,{id:`${i}-kit-summary`,byteLength:Ye*4})),M=(t,n,r,i)=>({spatialContext:e.spatialContext,offsets:F(t,`${n}-offsets`,r.offsets,`uint32`,a+1),neighbors:F(t,`${n}-neighbors`,r.neighbors,`uint32`,i),weights:F(t,`${n}-weights`,r.weights,`float32`,i)}),N=e=>F(e,`positions`,c,`float32x2`,a),P=null;if(!e.summary){let e=new t(n,{id:`${i}-summary`});e.add(new We({id:`weights-summary`,weights:M(e,`a`,f,o),output:{statistics:F(e,`statistics`,w,`float32`,3),counts:F(e,`counts`,T,`uint32`,X),cardinality:F(e,`cardinality`,D,`uint32`,a)}})),P=r.track(await e.compileAsync())}let I=new t(n,{id:`${i}-transpose-union`});{let e=M(I,`a`,f,o),t=M(I,`t`,m,o);I.add(new k({id:`transpose`,weights:e,output:{...t,asymmetricSlots:F(I,`asymmetric`,_,`uint32`,1)}})),I.add(new K({id:`symmetrize-union`,operation:`union`,left:e,right:t,output:{...M(I,`s`,h,s),status:{overflow:F(I,`union-overflow`,v,`uint32`,1),requiredCount:F(I,`union-total`,y,`uint32`,1)}}}))}let L=r.track(await I.compileAsync()),te=new Map;for(let s of[!1,!0]){let c=new t(n,{id:`${i}-lag-${s}`}),d=F(c,`lag`,x,`float32`,a);c.add(new E({id:`spatial-lag`,values:F(c,`values`,l,`float32`,a),mask:F(c,`mask`,e.mask,`uint32`,a),weights:M(c,`a`,f,o),normalize:s,output:d})),c.add(new u({id:`lag-extent`,input:d,output:F(c,`lag-extent`,S,`float32`,2),operation:`extent`})),te.set(s,r.track(await c.compileAsync()))}let ne={};for(let e of[`weights`,`transpose`,`union`]){let o=new t(n,{id:`${i}-focus-${e}`}),s=M(o,`m`,g[e].csr,g[e].slots);R(o,{id:`focus-${e}`,invocationCount:128,bindings:[{name:`focusRow`,view:F(o,`focus`,C,`uint32`,1),type:`u32`,access:`read`},{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`positions`,view:N(o),type:`f32`,access:`read`},{name:`segments`,view:F(o,`segments`,A[e],`float32`,512),type:`f32`,access:`read_write`},{name:`neighborIds`,view:F(o,`ids`,j[e],`uint32`,128),type:`u32`,access:`read_write`},{name:`info`,view:F(o,`info`,O[e],`float32`,2),type:`f32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${a}u;`,body:Xe}),ne[e]=r.track(await o.compileAsync())}let re={weights:null,transpose:null,union:null},ie=[],ae={};if(e.drawAllLinks)for(let e of[`weights`,`transpose`,`union`]){let o=g[e].slots,s=p(`all-segments-${e}`,o*16),c=p(`all-fade-${e}`,o*4),l=new t(n,{id:`${i}-links-${e}`}),u=M(l,`m`,g[e].csr,o);R(l,{id:`links-${e}`,invocationCount:o,bindings:[{name:`offsets`,view:u.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`positions`,view:N(l),type:`f32`,access:`read`},{name:`segments`,view:F(l,`segments`,s,`float32`,o*4),type:`f32`,access:`read_write`},{name:`fade`,view:F(l,`fade`,c,`float32`,o),type:`f32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${a}u;`,body:Z});let d=r.track(await l.compileAsync());ie.push(d),ae[e]=d,re[e]={segments:s,fade:c,slots:o}}let z=()=>[...P?[P]:[],L,...te.values(),...Object.values(ne),...ie],oe=e=>{for(let t of[`weights`,`transpose`,`union`])ne[t].encode(e,{parameters:void 0})};return{compiledGraphs:z,cardinality:D,lag:x,lagExtent:S,focusRow:C,focusSegments:A,focusNeighbors:j,allSegments:re,encodeAnalysis(e,t){P?.encode(e,{parameters:void 0}),L.encode(e,{parameters:void 0}),te.get(t.normalizeLag).encode(e,{parameters:void 0}),oe(e),ae[t.matrix]?.encode(e,{parameters:void 0})},encodeFocus:oe,setFocusRow(e){C.write(new Uint32Array([Math.max(0,Math.min(a-1,e))]))},async readSummary(t){let n=ee.tryAcquire();if(!n)return null;let r=0,i=(e,i)=>{t.copyBufferToBuffer({sourceBuffer:e,destinationBuffer:n.buffer,destinationOffset:r*4,size:i*4}),r+=i};i(w,3),i(T,X),i(_,1),i(S,2),i(v,1),i(y,1),i(e.producer.overflow,1),i(e.producer.total,1),i(b,1),i(O.weights,2),i(O.transpose,2),i(O.union,2),n.markEncoded({byteOffset:0,byteLength:Ye*4}),await new Promise(e=>setTimeout(e,0));let a=await n.read(),o=new Float32Array(a.buffer,a.byteOffset,Ye),s=new Uint32Array(a.buffer,a.byteOffset,Ye),c=Y,l=e=>s[3+e],u=3+X,d=e=>({degree:o[e],weightSum:o[e+1]});return{s0:o[c.statistics.s0],s1:o[c.statistics.s1],s2:o[c.statistics.s2],slots:l(c.counts.slots),asymmetricSlots:l(c.counts.asymmetricSlots),isolates:l(c.counts.isolates),minimumCardinality:l(c.counts.minimumCardinality),maximumCardinality:l(c.counts.maximumCardinality),transposeAsymmetricSlots:s[u],lagMinimum:o[u+1],lagMaximum:o[u+2],unionOverflow:s[u+3]!==0,unionSlots:s[u+4],producerOverflow:s[u+5]!==0,producerTotal:s[u+6],producerCandidateOverflow:s[u+7]!==0,focus:{weights:d(u+8),transpose:d(u+10),union:d(u+12)}}}}}var Ze=[`count`,`weightSum`,`sum`,`mean`,`min`,`max`,`standardDeviation`,`median`],$=Ze.length,Qe=64,$e=`
  let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  // Active neighbours by the library's definition (positive weight, not the row itself), as the
  // weights summary counts them; an island has none.
  let degree = activeCardinality[activeCardinalityOffset + index];
  isolate[isolateOffset + index] = select(0u, 1u, degree == 0u);
  cardinality[cardinalityOffset + index] = f32(degree);
  var oneWayCount = 0u;
  for (var slot = start; slot < end; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other >= ROW_COUNT || other == index || !(slotWeights[slotWeightsOffset + slot] > 0.0)) {
      continue;
    }
    var low = offsets[offsetsOffset + other];
    var high = offsets[offsetsOffset + other + 1u];
    let rowEnd = high;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (neighbors[neighborsOffset + middle] < index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    if (!(low < rowEnd && neighbors[neighborsOffset + low] == index)) {
      oneWayCount++;
    }
  }
  oneWay[oneWayOffset + index] = f32(oneWayCount);
  let focus = focusRow[focusRowOffset];
  var focusValue = 0u;
  if (index == focus) {
    focusValue = 2u;
  } else {
    let focusStart = offsets[offsetsOffset + focus];
    let focusEnd = offsets[offsetsOffset + focus + 1u];
    for (var slot = focusStart; slot < focusEnd; slot++) {
      if (neighbors[neighborsOffset + slot] == index) { focusValue = 1u; }
    }
  }
  focusClass[focusClassOffset + index] = focusValue;`,et=`
  let choice = choiceBuffer[choiceBufferOffset];
  var value = 0.0;
  if (choice == 0u) {
    value = values[valuesOffset + index];
  } else if (choice == 1u) {
    value = cardinality[cardinalityOffset + index];
  } else if (choice == 2u) {
    value = lag[lagOffset + index];
  } else if (choice == 3u) {
    value = oneWay[oneWayOffset + index];
  } else if (choice < ${4+$}u) {
    value = table[tableOffset + index * ${$}u + (choice - 4u)];
  } else {
    value = entropy[entropyOffset + index];
  }
  shown[shownOffset + index] = value;`,tt=`
  let value = shown[shownOffset + index];
  let finite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  let included = mask[maskOffset + index] != 0u && finite;
  valid[validOffset + index] = select(0u, 1u, included);
  var nanBits = 0x7fc00000u;
  shown[shownOffset + index] = select(bitcast<f32>(nanBits), value, included);`;async function nt(e){let{device:n,resources:r,id:i,rows:a,slots:o,positions:s,values:c,mask:l}=e,d=e.displayMask??l,f=(e,t)=>r.createBuffer(`${i}-${e}`,t),p={offsets:f(`analysis-offsets`,(a+1)*4),neighbors:f(`analysis-neighbors`,o*4),weights:f(`analysis-weights`,o*4)},m=await Q({spatialContext:e.spatialContext,device:n,resources:r,id:`${i}-kit`,rowCount:a,slots:o,unionSlots:e.unionSlots,positions:s,values:c,mask:l,csr:p,producer:e.producer,drawAllLinks:e.drawAllLinks}),h=f(`categories`,e.categories),g=f(`neighborhood-table`,a*$*4),_=f(`neighborhood-modes`,a*4),v=f(`neighborhood-entropy`,a*4),y=f(`neighborhood-overflow`,4),b=f(`cardinality-float`,a*4),x=f(`one-way`,a*4),S=f(`isolate`,a*4),C=f(`focus-class`,a*4),w=f(`shown`,a*4),T=f(`shown-valid`,a*4),E=f(`shown-extent`,8),D=f(`display-choice`,new Uint32Array([0])),O=new Map,k=(e,t,n)=>`${e?`focal`:`plain`}-${t}-${n?`row`:`raw`}`,A=async(s,u,d)=>{let f=k(s,u,d);if(O.has(f))return!1;let m=new t(n,{id:`${i}-neighborhood-${f}`});return m.add(new Je({id:`neighborhood`,weights:{spatialContext:e.spatialContext,offsets:F(m,`offsets`,p.offsets,`uint32`,a+1),neighbors:F(m,`neighbors`,p.neighbors,`uint32`,o),weights:F(m,`weights`,p.weights,`float32`,o),information:ye(a,d)},values:F(m,`values`,c,`float32`,a),mask:F(m,`mask`,l,`uint32`,a),categories:F(m,`categories`,h,`uint32`,a),includeFocal:s,focalWeight:u,statistics:Ze,maximumNeighbors:Qe,output:{table:F(m,`table`,g,`float32`,a*$),modes:F(m,`modes`,_,`uint32`,a),entropy:F(m,`entropy`,v,`float32`,a),status:{overflow:F(m,`overflow`,y,`uint32`,1)}}})),O.set(f,r.track(await m.compileAsync())),!0};await A(!1,1,!1);let j=new t(n,{id:`${i}-display`});{let e={offsets:F(j,`offsets`,p.offsets,`uint32`,a+1),neighbors:F(j,`neighbors`,p.neighbors,`uint32`,o),weights:F(j,`weights`,p.weights,`float32`,o)},t=F(j,`cardinality`,b,`float32`,a),n=F(j,`one-way`,x,`float32`,a);R(j,{id:`${i}-classify`,invocationCount:a,bindings:[{name:`offsets`,view:e.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.neighbors,type:`u32`,access:`read`},{name:`slotWeights`,view:e.weights,type:`f32`,access:`read`},{name:`activeCardinality`,view:F(j,`active-cardinality`,m.cardinality,`uint32`,a),type:`u32`,access:`read`},{name:`focusRow`,view:F(j,`focus-row`,m.focusRow,`uint32`,1),type:`u32`,access:`read`},{name:`isolate`,view:F(j,`isolate`,S,`uint32`,a),type:`u32`,access:`read_write`},{name:`focusClass`,view:F(j,`focus-class`,C,`uint32`,a),type:`u32`,access:`read_write`},{name:`cardinality`,view:t,type:`f32`,access:`read_write`},{name:`oneWay`,view:n,type:`f32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${a}u;`,body:$e});let r=F(j,`shown`,w,`float32`,a),s=F(j,`valid`,T,`uint32`,a);R(j,{id:`${i}-display-select`,invocationCount:a,bindings:[{name:`choiceBuffer`,view:F(j,`choice`,D,`uint32`,1),type:`u32`,access:`read`},{name:`values`,view:F(j,`values`,c,`float32`,a),type:`f32`,access:`read`},{name:`cardinality`,view:t,type:`f32`,access:`read`},{name:`lag`,view:F(j,`lag`,m.lag,`float32`,a),type:`f32`,access:`read`},{name:`oneWay`,view:n,type:`f32`,access:`read`},{name:`table`,view:F(j,`table`,g,`float32`,a*$),type:`f32`,access:`read`},{name:`entropy`,view:F(j,`entropy`,v,`float32`,a),type:`f32`,access:`read`},{name:`shown`,view:r,type:`f32`,access:`read_write`}],body:et}),R(j,{id:`${i}-display-valid`,invocationCount:a,bindings:[{name:`shown`,view:r,type:`f32`,access:`read_write`},{name:`mask`,view:F(j,`mask`,d,`uint32`,a),type:`u32`,access:`read`},{name:`valid`,view:s,type:`u32`,access:`read_write`}],body:tt}),j.add(new u({id:`shown-extent`,input:r,mask:s,output:F(j,`shown-extent`,E,`float32`,2),operation:`extent`}))}let ee=r.track(await j.compileAsync()),M=null,N=new L(r,`${i}-snapshot`,[{buffer:b,size:a*4},{buffer:x,size:a*4},{buffer:m.lag,size:a*4},{buffer:_,size:a*4},{buffer:w,size:a*4},{buffer:E,size:8},{buffer:m.focusNeighbors.weights,size:512},{buffer:m.focusNeighbors.transpose,size:512},{buffer:m.focusNeighbors.union,size:512}],e=>ve(`weights snapshot`,()=>{let t=e=>a*4*e;M?.({cardinality:new Float32Array(e,t(0),a),oneWay:new Float32Array(e,t(1),a),lag:new Float32Array(e,t(2),a),modes:new Uint32Array(e,t(3),a),shown:new Float32Array(e,t(4),a),extent:[new Float32Array(e,t(5),2)[0],new Float32Array(e,t(5),2)[1]],focusIds:{weights:new Uint32Array(e,t(5)+8,128),transpose:new Uint32Array(e,t(5)+8+512,128),union:new Uint32Array(e,t(5)+8+1024,128)}})})),P=!1,I=(e,t)=>{switch(e){case`neighbors`:return 1;case`lag`:return 2;case`oneWay`:return 3;case`summary`:return t===`entropy`?4+$:4+Ze.indexOf(t);default:return 0}};return{rows:a,csr:p,slots:o,kit:m,shown:w,shownExtent:E,isolate:S,focusClass:C,prepare:A,setDisplayChoice(e,t){let n=I(e,t);return D.write(new Uint32Array([n])),n},encode(e,t){let n=O.get(k(t.includeFocal,t.focalWeight,t.rowStandardized));n&&(m.encodeAnalysis(e,{normalizeLag:t.normalizeLag,matrix:t.matrix}),n.encode(e,{parameters:void 0}),ee.encode(e,{parameters:void 0}))},requestReadback(e,t,n){M=n,P||(P=!0,m.readSummary(e).then(e=>{e&&ve(`weights summary`,()=>t(e))}).catch(e=>{console.warn(`weights readback failed`,e)}).finally(()=>{P=!1})),N.request(e)},isReading:()=>P||N.isPending,getGraphs:()=>[...m.compiledGraphs(),...O.values(),ee],stop:()=>N.stop()}}var rt=Ke;function it(e){let{contiguityVertices:t,contiguityRingOffsets:n}=e,r=e.featureRingOffsets,i=new Map,a=(e,n)=>{let r=`${t[e*2]},${t[e*2+1]}`,i=`${t[n*2]},${t[n*2+1]}`;return r<i?`${r}|${i}`:`${i}|${r}`},o=t=>{for(let i=0;i<e.count;i++)for(let e=r[i];e<r[i+1];e++){let r=n[e],o=n[e+1];for(let e=r;e<o;e++)t(i,a(e,e+1<o?e+1:r))}};o((e,t)=>i.set(t,(i.get(t)??0)+1));let s=new Uint8Array(e.count);return o((e,t)=>{i.get(t)===1&&(s[e]=1)}),s}var at=4294967295,ot=`
  let focus = min(focusRow[focusRowOffset], ROW_COUNT - 1u);
  let start = offsets[offsetsOffset + focus];
  let end = offsets[offsetsOffset + focus + 1u];
  let slot = start + index;
  var other = 0xffffffffu;
  var value = 0.0;
  if (slot < end) {
    other = neighbors[neighborsOffset + slot];
    value = weights[weightsOffset + slot];
  }
  neighborIds[neighborIdsOffset + index] = other;
  slotWeights[slotWeightsOffset + index] = value;`;async function st(e){let{device:n,resources:r,id:i,rows:a,slots:o,csr:s,focusRow:c}=e,l=r.createBuffer(`${i}-ids`,512),u=r.createBuffer(`${i}-weights`,512),d=new t(n,{id:`${i}-gather`});R(d,{id:`${i}-gather`,invocationCount:128,bindings:[{name:`focusRow`,view:F(d,`focus`,c,`uint32`,1),type:`u32`,access:`read`},{name:`offsets`,view:F(d,`offsets`,s.offsets,`uint32`,a+1),type:`u32`,access:`read`},{name:`neighbors`,view:F(d,`neighbors`,s.neighbors,`uint32`,o),type:`u32`,access:`read`},{name:`weights`,view:F(d,`weights`,s.weights,`float32`,o),type:`f32`,access:`read`},{name:`neighborIds`,view:F(d,`ids`,l,`uint32`,128),type:`u32`,access:`read_write`},{name:`slotWeights`,view:F(d,`slot-weights`,u,`float32`,128),type:`f32`,access:`read_write`}],declarations:`const ROW_COUNT: u32 = ${a}u;`,body:ot});let f=r.track(await d.compileAsync()),p=new L(r,`${i}-focus-row`,[{buffer:l,size:512},{buffer:u,size:512}],t=>{let n=new Uint32Array(t,0,128),r=new Float32Array(t,512,128),i=0;for(;i<128&&n[i]!==at;)i++;e.onRow({ids:n,weights:r,degree:i})});return{slotWeights:u,graph:f,encode:e=>f.encode(e,{parameters:void 0}),request:e=>p.request(e),isReading:()=>p.isPending,stop:()=>p.stop()}}var ct={"us-counties":32e4,"chicago-tracts":48e3},lt={"us-counties":{typical:[-87.65,41.84],corner:[-108.35,36.55],island:[-70.05,41.28]},"chicago-tracts":{typical:[-87.63,41.88],corner:[-87.6107,41.7774],island:[-87.9,41.98]}},ut={"us-counties":110,"chicago-tracts":70},dt=4,ft=4096,pt=32768,mt=3,ht={queen:`queen`,rook:`rook`,knn:`nearest`,band:`distance band`},gt=new Map;function _t(e,t,n,r,i,a){let{geography:o}=e,[s,c]=o.project(...o.getCentroidLngLat(e.focusRow)),l=Math.min(i-1,Math.max(0,Math.floor((s-t)/r)));return Math.min(a-1,Math.max(0,Math.floor((c-n)/r)))*i+l}function vt(e){let t=0,n=0;for(let r=0;r<e.length;r++)Number.isFinite(e[r])&&(t++,n+=e[r]);if(t<2)return 1;let r=n/t,i=0;for(let t=0;t<e.length;t++)Number.isFinite(e[t])&&(i+=(e[t]-r)**2);return Math.sqrt(i/(t-1))}async function yt(e){let{device:n}=e,r=!1,i=new I,o=null,s=0,c=!1,l=0,u=!0,d=!0,f=!0,p=!0,m=null,h={},g=null,_=t=>{let n=gt.get(t);return n||(n=Se(t,e.datasets,e.signal),n.catch(()=>gt.delete(t)),gt.set(t,n)),n},v=t=>{let n=e.options;return{source:n.source,k:n.k,snapTolerance:Number(n.snapTolerance),bandFactor:n.bandFactor,knnCapFactor:n.knnCapFactor,weightKind:n.weightKind,kernel:n.kernel,power:n.power,distanceFloor:n.distanceFloorFactor*t.medianSpacing,rowStandardize:n.rowStandardize,transform:n.transform,transformKernel:n.transformKernel,bandwidthFactor:n.bandwidth===`adaptive`?0:Number(n.bandwidth),doubleSum:n.doubleSum}},y=()=>{let t=e.options;switch(t.combine){case`none`:return null;case`union`:case`intersection`:case`difference`:case`symmetricDifference`:return`${t.combine}-${t.weightRule}`;case`higherOrder`:return`higherOrder-${t.order}-${t.cumulative}`;default:return t.combine}},b=(e,t)=>{let{geography:n,projection:r}=e,i=[];for(let e=n.featureRingOffsets[t];e<n.featureRingOffsets[t+1];e++){let t=[],a=n.contiguityRingOffsets[e],o=n.contiguityRingOffsets[e+1];for(let e=a;e<o;e++)t.push(r.unproject(n.contiguityVertices[e*2],n.contiguityVertices[e*2+1]));t.length&&t.push(t[0]),i.push(t)}return i},x=(e,t)=>e.geography.getCentroidLngLat(t),S=(t,n)=>{let r=e.datasets.get(t.geography.id).geojson?.features[n];return(r?te(r.geometry):null)??x(t,n)},C=(e,t,n,r)=>{let{geography:i}=e,a=new Float32Array(n*4),o=0;for(let e of r)for(let t=i.featureRingOffsets[e];t<i.featureRingOffsets[e+1];t++){let e=i.contiguityRingOffsets[t],r=i.contiguityRingOffsets[t+1];for(let t=e;t<r&&o<n;t++){let n=t+1<r?t+1:e;a[o*4]=i.contiguityVertices[t*2],a[o*4+1]=i.contiguityVertices[t*2+1],a[o*4+2]=i.contiguityVertices[n*2],a[o*4+3]=i.contiguityVertices[n*2+1],o++}}return o>0&&t.write(a.subarray(0,o*4)),o},w=e=>{let t=V(e);return e=>t.digits===0?B(e):e.toFixed(t.digits)},T=t=>{let n=e.ground(),r=V(e.options.variable),i=w(e.options.variable),a=t.geography.getVariable(e.options.variable);return{cardinality:we(n,`neighbours per ${t.geography.unit}`),oneWay:je(n),weights:Ee(n),value:Oe(a,`YlOrBr`,n,{unit:r.unit,format:i,method:`Quantiles of the variable; the lag reuses them`,noDataLabel:`No data`}),difference:Fe(t.spread,n,{unit:r.unit,format:i}),summary:m?.summary??null}},E=()=>{m&&e.setLegendData(`weights`,{ground:e.ground(),tables:m,counts:h})},O=()=>{let e=o;e&&(m=T(e),E())},k=t=>{let n=e.options,r=V(n.variable),i=w(n.variable);switch(n.summary){case`count`:return{scheme:`BuGn`,unit:`neighbours`,format:B,title:`Neighbours counted`};case`weightSum`:return{scheme:`BuGn`,unit:`weight`,format:e=>e.toFixed(2),title:`Sum of weights`};case`entropy`:return{scheme:`Greys`,unit:`nats`,format:e=>e.toFixed(2),title:`Group diversity of the ${t.geography.unitPlural}`};default:return{scheme:`YlOrBr`,unit:r.unit,format:i,title:`Neighbourhood ${n.summary}`}}},A=async t=>{let r=new a(n,`weights-${t.id}`),i=t.count,s=(e,t)=>r.createBuffer(e,t),c=xe({device:n,resources:r,id:`weights-core`,geography:t}),l=ct[t.id],u=t.getVariable(e.options.variable),d=s(`values`,Float32Array.from(u)),f=s(`value-mask`,new Uint32Array(i)),p=s(`display-mask`,new Uint32Array(i).fill(1)),m=await nt({spatialContext:t.analysisFrame.spatialContext,device:n,resources:r,id:`polygons`,rows:i,slots:l,unionSlots:l,positions:c.positions,values:d,mask:f,displayMask:p,categories:t.groups,drawAllLinks:!0,producer:{overflow:c.overflow,total:c.total,candidateOverflow:c.candidateOverflow}}),h=Float32Array.from(t.population).sort(),g=Math.max(0,t.pick(...lt[t.id][e.options.focus])),_=new j(t.origin),v=t.id===`us-counties`?e.datasets.get(`us-states`).geojson:e.datasets.get(`chicago-community-areas`).geojson,y=v?pe(fe(v).flatMap(({polygon:e})=>e),me(t.origin)):new Float32Array,b=await st({device:n,resources:r,id:`weights`,rows:i,slots:l,csr:m.csr,focusRow:m.kit.focusRow,onRow:e=>{let t=o;t&&t.gatherer===b&&qe(t,e)}}),x={geography:t,resources:r,core:c,partner:c.createPartner(`partner`),space:m,buffers:Ce(r,t,`weights`),values:d,valueMask:f,selfWeights:s(`self-weights`,new Float32Array(i)),subgraphMask:s(`subgraph-mask`,new Uint32Array(i).fill(1)),groups:s(`group-ids`,t.groups),algebra:new Map,algebraOverflow:s(`algebra-overflow`,4),algebraTotal:s(`algebra-total`,4),lattice:null,focusRow:g,populationSorted:h,snapshot:null,kitSummary:null,latticeSnapshot:null,latticeKitSummary:null,lagShown:s(`lag-shown`,new Float32Array(i).fill(NaN)),difference:s(`difference`,new Float32Array(i).fill(NaN)),weightRows:s(`weight-rows`,new Float32Array(i).fill(NaN)),gatherer:b,focusData:null,focusOutline:s(`focus-outline`,ft*16),focusOutlineCount:0,neighbourOutline:s(`neighbour-outline`,pt*16),neighbourOutlineCount:0,zoneLines:y.length?{buffer:s(`zone-lines`,y),count:y.length/4}:null,outerEdge:it(t),projection:_,reference:null,cornerRows:[],cornerLinks:null,spread:vt(u)};return R(x),m.kit.setFocusRow(g),x},R=t=>{let n=t.geography.getVariable(e.options.variable);t.valueMask.write(Uint32Array.from(n,e=>Number.isFinite(e)?1:0))},ve=t=>{let n=t.geography.getVariable(e.options.variable);t.values.write(Float32Array.from(n)),t.spread=vt(n),R(t),t.lattice&&Re(t,t.lattice)},ye=t=>{let n=t.geography.pick(...lt[t.geography.id][e.options.focus]);if(!(n<0)&&(t.focusRow=n,t.space.kit.setFocusRow(n),t.lattice)){let[e,n]=t.lattice.bounds;t.lattice.focusRow=_t(t,e,n,t.lattice.cell,t.lattice.columns,t.lattice.rows),t.lattice.space.kit.setFocusRow(t.lattice.focusRow)}},Ie=t=>{t.selfWeights.write(new Float32Array(t.geography.count).fill(e.options.selfWeight))},U=t=>{let n=t.populationSorted,r=n[Math.min(n.length-1,Math.floor(e.options.populationPercentile/100*n.length))];t.subgraphMask.write(Uint32Array.from(t.geography.population,e=>e>=r?1:0))},Le=async e=>{let{geography:t,resources:r}=e,i=ut[t.id],[a,o,s,c]=t.displayBounds,l=(s-a)/i,u=Math.ceil((c-o)/l),d=i*u,f=new Int32Array(d).fill(-1),p=new Float32Array(d*2),m=new Uint32Array(d),h=new Uint32Array(d);for(let e=0;e<u;e++)for(let n=0;n<i;n++){let r=e*i+n,s=a+(n+.5)*l,c=o+(e+.5)*l;p[r*2]=s,p[r*2+1]=c;let u=t.pickMeters(s,c);f[r]=u,u>=0&&(h[r]=1,m[r]=t.groups[u])}let g=(e,t)=>r.createBuffer(`lattice-${e}`,t),_=d*((2*dt+1)**2-1),v=g(`values`,new Float32Array(d)),y=g(`value-mask`,new Uint32Array(d)),b=g(`structure-mask`,Uint32Array.from(h)),x=g(`overflow`,4),S=g(`total`,4),C=await nt({spatialContext:e.projection.spatialContext,device:n,resources:r,id:`lattice`,rows:d,slots:_,unionSlots:_,positions:g(`positions`,p),values:v,mask:y,displayMask:b,categories:m,drawAllLinks:!1,producer:{overflow:x,total:S}}),w={columns:i,rows:u,cell:l,bounds:[a,o,a+i*l,o+u*l],featureOfCell:f,space:C,structureMask:b,values:v,valueMask:y,producers:new Map,overflow:x,total:S,focusRow:_t(e,a,o,l,i,u),insideMask:h};return C.kit.setFocusRow(w.focusRow),Re(e,w),W(w),w},Re=(t,n)=>{let r=t.geography.getVariable(e.options.variable),i=new Float32Array(n.featureOfCell.length),a=new Uint32Array(n.featureOfCell.length);for(let e=0;e<i.length;e++){let t=n.featureOfCell[e],o=t>=0?r[t]:NaN;i[e]=o,a[e]=Number.isFinite(o)?1:0}n.values.write(i),n.valueMask.write(a)},W=t=>{t.structureMask.write(e.options.latticeMask?Uint32Array.from(t.insideMask):new Uint32Array(t.insideMask.length).fill(1))},ze=async(r,i)=>{let{latticeCriterion:a,latticeRadius:o}=e.options,s=`${a}-${o}`;if(r.producers.has(s))return!1;let{space:c}=r,l=new t(n,{id:`lattice-${s}`}),u=r.columns*r.rows;return l.add(new D({id:`lattice`,width:r.columns,height:r.rows,criterion:a,radius:o,cellSize:[r.cell,r.cell],mask:F(l,`mask`,r.structureMask,`uint32`,u),output:{offsets:F(l,`offsets`,c.csr.offsets,`uint32`,u+1),neighbors:F(l,`neighbors`,c.csr.neighbors,`uint32`,c.slots),weights:F(l,`weights`,c.csr.weights,`float32`,c.slots),status:{overflow:F(l,`overflow`,r.overflow,`uint32`,1),requiredCount:F(l,`total`,r.total,`uint32`,1)}}})),r.producers.set(s,i.track(await l.compileAsync())),!0},Be=async r=>{let i=y();if(!i||r.algebra.has(i))return!1;let a=e.options,{geography:o,space:s,core:c}=r,l=o.count,u=new t(n,{id:`weights-algebra-${i}`}),d=(e,t,n)=>({spatialContext:r.geography.analysisFrame.spatialContext,offsets:F(u,`${e}-offsets`,t.offsets,`uint32`,l+1),neighbors:F(u,`${e}-neighbors`,t.neighbors,`uint32`,n),weights:F(u,`${e}-weights`,t.weights,`float32`,n)}),f=d(`a`,c.csr,c.slots),p={id:`algebra`,output:{...d(`out`,s.csr,s.slots),status:{overflow:F(u,`overflow`,r.algebraOverflow,`uint32`,1),requiredCount:F(u,`total`,r.algebraTotal,`uint32`,1)}}};switch(a.combine){case`union`:case`intersection`:case`difference`:case`symmetricDifference`:u.add(new K({...p,operation:a.combine,left:f,right:d(`b`,r.partner.csr,r.partner.slots),weightRule:a.weightRule}));break;case`higherOrder`:u.add(new K({...p,operation:`higherOrder`,weights:f,order:a.order,cumulative:a.cumulative}));break;case`selfWeight`:u.add(new K({...p,operation:`selfWeight`,weights:f,selfWeight:F(u,`self-weights`,r.selfWeights,`float32`,l)}));break;case`subgraph`:u.add(new K({...p,operation:`subgraph`,weights:f,mask:F(u,`mask`,r.subgraphMask,`uint32`,l)}));break;case`block`:u.add(new K({...p,operation:`block`,groupIds:F(u,`groups`,r.groups,`uint32`,l),groupCount:o.groupCount}));break;default:return!1}return r.algebra.set(i,r.resources.track(await u.compileAsync())),!0},G=()=>[`union`,`intersection`,`difference`,`symmetricDifference`].includes(e.options.combine),Ve=(e,t)=>({...v(e.geography),source:t,snapTolerance:0,transform:`none`}),He=e=>{if(e.reference)return e.reference;let{geography:t,resources:r}=e,i=xe({device:n,resources:r,id:`weights-queen`,geography:t}),a=xe({device:n,resources:r,id:`weights-rook`,geography:t}),s=t.count;return e.reference={queen:i,rook:a,reader:new L(r,`weights-reference`,[{buffer:i.csr.offsets,size:(s+1)*4},{buffer:i.csr.neighbors,size:i.slots*4},{buffer:a.csr.offsets,size:(s+1)*4},{buffer:a.csr.neighbors,size:a.slots*4}],t=>{let n=e.reference;if(!n||o!==e)return;let r=(s+1)*4,c=i.slots*4;n.data={queenOffsets:new Uint32Array(t,0,s+1),queenNeighbors:new Uint32Array(t,r,i.slots),rookOffsets:new Uint32Array(t,r+c,s+1),rookNeighbors:new Uint32Array(t,r*2+c,a.slots)},Je(e)}),encoded:!1,data:null},e.reference},q=e=>!y()&&be(v(e.geography)),Ue=async t=>{let n=e.options,r=Number(n.focalWeight);if(n.lattice)t.lattice??=await Le(t),await ze(t.lattice,t.resources),await t.lattice.space.prepare(n.includeFocal,r,!1);else if(await t.core.prepare(v(t.geography)),G()&&await t.partner.prepare(n.partnerK),await Be(t),await t.space.prepare(n.includeFocal,r,q(t)),n.showQueenOnly){let e=He(t);await e.queen.prepare(Ve(t,`queen`)),await e.rook.prepare(Ve(t,`rook`))}},J=t=>{let n=e.options,r=H(n),i=r===`weights`?`focus`:r===`difference`?`lag`:r;t.space.setDisplayChoice(i,n.summary),t.lattice?.space.setDisplayChoice(i,n.summary)},Y=async e=>{await Ue(e),J(e)},We=()=>{let e=o;e&&e.core.writeParameters(v(e.geography))},Ge=(t,n)=>{let{geography:r}=t,i=r.getVariable(e.options.variable),a=new Float32Array(r.count),o=new Float32Array(r.count);for(let e=0;e<r.count;e++){let t=n.cardinality[e]>0&&Number.isFinite(i[e])&&Number.isFinite(n.lag[e]);a[e]=t?n.lag[e]:NaN,o[e]=t?n.lag[e]-i[e]:NaN}return t.lagShown.write(a),t.difference.write(o),{lag:a,difference:o,values:i}},Ke=(t,n,r)=>{if(r){t.latticeSnapshot=n,Z();return}t.snapshot=n;let{lag:i,difference:a,values:o}=Ge(t,n);m||=T(t);let s=e.options;if(H(s)===`summary`){let r=k(t);m={...m,summary:De(n.shown,r.scheme,e.ground(),r)}}h={cardinality:z(n.cardinality,m.cardinality.breaks),oneWay:z(n.oneWay,m.oneWay.breaks),value:z(o,m.value.breaks),lag:z(i,m.value.breaks),difference:z(a,m.difference.breaks),summary:m.summary?z(n.shown,m.summary.breaks):void 0},E(),e.setChart(`cardinality`,Pe(n.cardinality,m.cardinality,e.ground(),t.geography.unitPlural));let c=Me({values:o,lag:i,unit:V(s.variable).unit});e.setChart(`lagScatter`,c),e.setReadout(`lagSlope`,Number.isFinite(c.slope)?c.slope.toFixed(2):null),Z(),Q(),e.requestLayers()};function qe(t,n){t.focusData=n;let{geography:r}=t,i=0;for(let e=0;e<n.degree;e++)n.ids[e]!==t.focusRow&&(i=Math.max(i,n.weights[e]));let a=new Float32Array(r.count).fill(NaN),o=[];for(let e=0;e<n.degree;e++){let s=n.ids[e];s===t.focusRow||s>=r.count||(o.push(s),a[s]=i>0?n.weights[e]/i:0)}t.weightRows.write(a),t.focusOutlineCount=C(t,t.focusOutline,ft,[t.focusRow]),t.neighbourOutlineCount=C(t,t.neighbourOutline,pt,o),X(t),Z(),Xe(),Q(),e.requestLayers()}function Je(t){let n=t.reference?.data;if(!n)return;let r=t.focusRow,i=new Set;for(let e=n.rookOffsets[r];e<n.rookOffsets[r+1];e++)i.add(n.rookNeighbors[e]);let a=[];for(let e=n.queenOffsets[r];e<n.queenOffsets[r+1];e++)i.has(n.queenNeighbors[e])||a.push(n.queenNeighbors[e]);let o=t.geography.count;t.cornerRows=a,t.cornerLinks=(n.queenOffsets[o]-n.rookOffsets[o])/2,Z(),Q(),e.requestLayers()}function X(t){let n=e.options,r=t.focusData,i=v(t.geography),a=[`none`,`row`,`double`,`kernel`].includes(n.transform);if(!r||n.lattice||r.degree===0||!a||!t.core.hasDistances(i)){e.setChart(`kernelCurve`,null);return}let{centroids:o}=t.geography,s=t.focusRow,c=[],l=[];for(let e=0;e<r.degree;e++){let t=r.ids[e];t!==s&&(c.push(Math.hypot(o[t*2]-o[s*2],o[t*2+1]-o[s*2+1])),l.push(r.weights[e]))}let u=c.length?Math.max(...c):0,d=null,f=0;if(n.transform===`kernel`){d=n.transformKernel;let e=t.core.getKernelBandwidth(i);f=e===`adaptive`?u:e}else n.weightKind===`kernel`&&(d=n.kernel,f=n.source===`band`?t.core.getBandMeters(i):u);if(!d||!(f>0)||c.length===0){e.setChart(`kernelCurve`,null);return}e.setChart(`kernelCurve`,ke({kernel:d,bandwidth:f,groundFactor:1,distances:c,weights:l,focusName:t.geography.getName(s)}))}let Ye=t=>{if(e.options.lattice&&t.lattice)return`cell ${t.lattice.focusRow}`;let{geography:n,focusRow:r}=t;return`${n.getName(r)}, ${n.getGroupName(r)}`};function Z(){let t=o;if(!t)return;let{geography:n}=t,r=e.options,i=!!(r.lattice&&t.lattice),a=i?t.latticeSnapshot:t.snapshot,s=i?t.latticeKitSummary:t.kitSummary,c=i?t.lattice.columns*t.lattice.rows:n.count,l=i?t.lattice.insideMask.reduce((e,t)=>e+t,0):c;if(e.setReadout(`rows`,i?`${B(l)} of ${B(c)} cells`:B(c)),e.setReadout(`focusName`,Ye(t)),s){e.setReadout(`links`,B(s.slots)),e.setReadout(`meanNeighbors`,(s.slots/Math.max(1,c-s.isolates)).toFixed(1)),e.setReadout(`asymmetric`,B(s.asymmetricSlots)),e.setReadout(`sums`,`${B(s.s0)} / ${B(s.s1)} / ${B(s.s2)}`),e.setReadout(`union`,s.unionOverflow?`overflow`:`${B(s.unionSlots)} slots`);let t=V(r.variable);e.setReadout(`lagRange`,`${s.lagMinimum.toFixed(t.digits)} to ${s.lagMaximum.toFixed(t.digits)} ${t.unit}`),e.setReadout(`capacity`,s.producerCandidateOverflow?`pair list full: some links dropped, count is a lower bound`:s.producerOverflow?`producer overflow: some links dropped`:`ok`),e.setReadout(`islands`,B(s.isolates));let n=s.focus[r.matrix];e.setReadout(`focusDegree`,B(n.degree)),e.setReadout(`weightSum`,n.weightSum.toFixed(2))}if(a&&!i){let r=[],i=0,o=0,s=0,l=0,u=0;for(let e=0;e<c;e++){let c=a.cardinality[e];c===0&&r.length<mt&&r.push(n.getName(e)),a.oneWay[e]>0&&i++,c>0&&(t.outerEdge[e]?(o+=c,s++):(l+=c,u++))}e.setReadout(`islandNames`,r.length?r.join(`, `):`none`),e.setReadout(`oneWayPlaces`,B(i)),e.setReadout(`edgeMean`,s?(o/s).toFixed(1):null),e.setReadout(`interiorMean`,u?(l/u).toFixed(1):null)}t.cornerLinks!==null&&(e.setReadout(`cornerLinks`,B(t.cornerLinks)),e.setReadout(`cornerFocus`,t.cornerRows.length?t.cornerRows.slice(0,mt).map(e=>`${n.getName(e)}, ${n.getGroupName(e)}`).join(`; `):`none`),e.setReadout(`cornerCount`,B(t.cornerRows.length)));let u=v(n),d=t.core.getBandMeters(u);e.setReadout(`band`,r.source===`band`&&!r.lattice?de(d):null),e.setReadout(`spacing`,de(n.medianSpacing)),e.setReadout(`numerics`,`Distances are metres of ${n.analysisFrame.name}, within ${(Math.ceil(n.analysisFrame.maxScaleError*1e6)/1e4).toFixed(4)}% of ground distance. Band radius ${de(d)}. Weights are float32; lags sum in slot order.`),e.setCost({records:c})}function Xe(){let t=o;if(!t)return;let n=e.options,{geography:r}=t,i=r.id===`us-counties`?`${B(r.count)} counties of the contiguous US (Alaska and Hawaii not in the data)`:`${B(r.count)} census tracts of Chicago, 2020 boundaries`;n.source===`band`&&!n.lattice?e.setFurniture({title:{sample:i},scaleBar:{latitude:x(t,t.focusRow)[1],ticks:[t.core.getBandMeters(v(r))]}}):e.setFurniture({title:{sample:i}})}function Q(){let t=o;if(!t)return;let n=e.options,{geography:r}=t,i=H(n);if(n.lattice){e.setAnnotations(`weights`,null);return}let a=[],s=t.snapshot,c=e=>`${r.getName(e)}, ${r.getGroupName(e)}`;if(i===`focus`||i===`weights`){let e=t.kitSummary?.focus[n.matrix].degree;if(e!==void 0&&a.push({kind:`note`,id:`weights-focus-note`,coordinate:S(t,t.focusRow),title:le(`{n} {noun}, {rule}`,{n:e,noun:e===1?`neighbour`:`neighbours`,rule:ht[n.source]}),text:c(t.focusRow),priority:6}),n.source===`band`){let e=t.core.getBandMeters(v(r));a.push({kind:`ring`,id:`weights-band-ring`,coordinate:x(t,t.focusRow),radiusMeters:e,text:`${de(e)} band`,dashed:!0,tone:`ink`})}if(n.showQueenOnly)for(let e of t.cornerRows.slice(0,mt))a.push({kind:`outline`,id:`weights-corner-outline-${e}`,rings:b(t,e),dashed:!0}),a.push({kind:`point`,id:`weights-corner-label-${e}`,coordinate:S(t,e),text:r.getName(e),marker:`none`,rank:`subject`,priority:5})}if(i===`neighbors`&&s&&r.id===`us-counties`){let e=0;for(let n=0;n<r.count&&e<mt;n++)s.cardinality[n]===0&&(e++,a.push({kind:`point`,id:`weights-island-${n}`,coordinate:S(t,n),text:r.getName(n),marker:`none`,rank:`subject`,priority:5}));let n=ne.places[`rio-grande-valley`].lngLat,[i,o]=r.analysisFrame.project(n[0],n[1]),l=-1,u=1/0;for(let e=0;e<r.count;e++){if(!t.outerEdge[e]||s.cardinality[e]===0)continue;let n=(r.centroids[e*2]-i)**2+(r.centroids[e*2+1]-o)**2;n<u&&(u=n,l=e)}l>=0&&a.push({kind:`note`,id:`weights-edge-note`,coordinate:S(t,l),title:le(`{n} neighbours at the edge`,{n:s.cardinality[l]}),text:c(l),priority:6})}if(i===`oneWay`&&s&&r.id===`us-counties`){let[e,n,i,o]=Ae,l=-1,u=0;for(let a=0;a<r.count;a++){if(s.oneWay[a]<=u)continue;let[r,c]=x(t,a);r<e||r>i||c<n||c>o||(l=a,u=s.oneWay[a])}l>=0&&a.push({kind:`note`,id:`weights-one-way-note`,coordinate:S(t,l),title:le(`{n} one-way links`,{n:u}),text:`${c(l)} lists ${B(s.cardinality[l])} counties, ${B(u)} of which do not list it back`,priority:6})}if((i===`value`||i===`lag`||i===`difference`)&&s&&r.id===`chicago-tracts`){let e=V(n.variable),t=r.getVariable(n.variable);for(let n of[`englewood`,`lincoln-park`]){let i=re.places[n],o=r.pick(i.lngLat[0],i.lngLat[1]);o<0||!Number.isFinite(t[o])||!Number.isFinite(s.lag[o])||a.push({kind:`note`,id:`weights-tract-note-${n}`,coordinate:i.lngLat,title:`${t[o].toFixed(e.digits)} here, ${s.lag[o].toFixed(e.digits)} around`,text:`${i.name}: tract value and neighbourhood lag, ${e.unit}`,priority:6})}}e.setAnnotations(`weights`,a.length?a:null)}let Ze=t=>{let n=o;if(!n)return;let r=!!(e.options.lattice&&n.lattice),i=r?n.lattice.space:n.space;d&&!i.isReading()&&(d=!1,i.requestReadback(t,e=>{r?n.latticeKitSummary=e:n.kitSummary=e,Z(),Q()},e=>Ke(n,e,r))),f&&!r&&!n.gatherer.isReading()&&(f=!1,n.gatherer.request(t));let a=n.reference;p&&a?.encoded&&!a.reader.isPending&&(p=!1,a.reader.request(t))},$=(e,t,n,r,i)=>{e.copyBufferToBuffer({sourceBuffer:t.offsets,destinationBuffer:n.offsets,size:(r+1)*4}),e.copyBufferToBuffer({sourceBuffer:t.neighbors,destinationBuffer:n.neighbors,size:i*4}),e.copyBufferToBuffer({sourceBuffer:t.weights,destinationBuffer:n.weights,size:i*4})},Qe=e=>{e.space.stop(),e.lattice?.space.stop(),e.gatherer.stop(),e.reference?.reader.stop()},$e=()=>{o&&(O(),Xe(),Z(),e.setChart(`cardinality`,null),e.setChart(`kernelCurve`,null),e.setChart(`lagScatter`,null),Q())},et=async t=>{let n=++s;e.setStatus(`Loading ${t===`us-counties`?`counties`:`tracts`}...`);let i=await _(t);if(n!==s||r)return;let a=o;o=null,e.requestLayers(),a&&(Qe(a),setTimeout(()=>a.resources.destroy(),200));let c=await A(i),v;do v=l,await Y(c);while(v!==l&&n===s&&!r);if(n!==s||r){c.resources.destroy();return}o=c,g=null,h={},m=null,We(),Ie(o),U(o),u=!0,d=!0,f=!0,p=!0,$e(),e.setStatus(``),e.requestLayers()},tt=await _(e.options.geography);if(e.signal.aborted)throw Error(`aborted`);o=await A(tt),await Y(o),We(),Ie(o),U(o),$e(),e.setStatus(``);let at=e=>{let t=o;return!t||!e.coordinate?-1:t.geography.pick(e.coordinate[0],e.coordinate[1])},ot=e=>{let t=o,n=t?.lattice;if(!t||!n||!e.coordinate)return-1;let[r,i]=t.geography.project(e.coordinate[0],e.coordinate[1]),a=Math.floor((r-n.bounds[0])/n.cell),s=Math.floor((i-n.bounds[1])/n.cell);return a<0||s<0||a>=n.columns||s>=n.rows?-1:s*n.columns+a},yt=(t,n,r)=>{let i=e.options,a=H(i),o=!!(i.lattice&&t.lattice),s=[0,0,0,0],c=oe[r],l=(e,t,n)=>({values:t,valueFormat:`float32`,colormap:`greys`,...ae(e),highlightClasses:g,noDataColor:n}),u=m??T(t);switch(a){case`weights`:return[{style:l(u.weights,t.weightRows,s)}];case`neighbors`:return[{style:l(u.cardinality,n.shown,c)}];case`oneWay`:return[{style:l(u.oneWay,n.shown,c)}];case`value`:return[{style:l(u.value,o?t.lattice.values:t.values,c)}];case`lag`:return o?[{style:l(u.value,n.shown,c)}]:[{style:l(u.value,t.values,c),side:`a`},{style:l(u.value,t.lagShown,c),side:`b`}];case`difference`:return[{style:l(u.difference,t.difference,c)}];case`summary`:return[{style:l(u.summary??u.value,n.shown,c)}];default:return[{style:{values:n.focusClass,valueFormat:`uint32`,colormap:`category`,palette:Te(r),noDataColor:s}}]}},bt=()=>{let t=o;if(!t)return[];let n=e.options,r=e.ground(),{geography:i,buffers:a}=t,s=[i.origin[0],i.origin[1],0],c=!!(n.lattice&&t.lattice),l=c?t.lattice.space:t.space,u=i.analysisFrame.layerFrame,d=c?void 0:u,f=H(n),p=i.id===`us-counties`?1:.88,m=i.count>2e3?.4:.5,h=se(r),g=ce(r),_=[];(n.showOutlines||c)&&_.push(new M({id:`weights-context-outline`,coordinateOrigin:s,segments:a.outline,instanceCount:a.outlineSegmentCount,widthPixels:m,color:_e(r)}));for(let{style:e,side:n}of yt(t,l,r))if(c){let r=t.lattice;_.push(new P({id:`weights-lattice-fill${n?`-${n}`:``}`,coordinateOrigin:s,gridSize:[r.columns,r.rows],bounds:r.bounds,...e,compareSide:n,opacity:.88}))}else _.push(new N({id:`weights-fill${n?`-${n}`:``}`,coordinateOrigin:s,triangles:a.triangles,features:a.features,vertexCount:a.triangleVertexCount,...e,compareSide:n,opacity:p}));let v=f!==`focus`&&f!==`weights`;if(n.showOutlines&&v&&!c&&_.push(new M({id:`weights-hairline`,coordinateOrigin:s,segments:a.outline,instanceCount:a.outlineSegmentCount,widthPixels:m,color:f===`difference`?_e(r):ge(r)})),n.showOutlines&&t.zoneLines&&!c)if(i.id===`us-counties`){let e=he(r);_.push(new M({id:`weights-state-lines`,coordinateOrigin:s,segments:t.zoneLines.buffer,instanceCount:t.zoneLines.count,widthPixels:e.widthPixels,color:e.color,outlineColor:e.casing,outlineWidthPixels:(e.casingPixels-e.widthPixels)/2}))}else{let e=Ne(r);_.push(new M({id:`weights-area-lines`,coordinateOrigin:s,segments:t.zoneLines.buffer,instanceCount:t.zoneLines.count,widthPixels:e.widthPixels,color:e.color}))}if(!c&&n.showLinks){let e=t.space.kit.allSegments[n.matrix];e&&_.push(new M({id:`weights-links-${n.matrix}`,coordinateOrigin:s,analysisFrame:u,segments:e.segments,weights:e.fade,instanceCount:e.slots,widthPixels:.5,color:_e(r),opacityStops:i.id===`us-counties`?[[5.5,0],[6.5,1]]:[[8.5,0],[9.5,1]]}))}!c&&(f===`focus`||f===`weights`)&&t.neighbourOutlineCount&&_.push(new M({id:`weights-neighbour-outline`,coordinateOrigin:s,segments:t.neighbourOutline,instanceCount:t.neighbourOutlineCount,widthPixels:1.2,color:se(r,210)}));let y=f===`weights`&&!c&&n.matrix===`weights`,b=t.focusData,x=0;if(y&&b)for(let e=0;e<b.degree;e++)x=Math.max(x,b.weights[e]);if(_.push(new M({id:`weights-focus-links-${n.matrix}-${c?`lattice`:`polygons`}${y?`-by-weight`:``}`,coordinateOrigin:s,analysisFrame:d,segments:l.kit.focusSegments[n.matrix],instanceCount:128,widthPixels:1.6,color:h,outlineColor:g,outlineWidthPixels:.8,...y&&x>0?{instanceChannels:t.gatherer.slotWeights,channelStride:1,channels:{width:0},widthDomain:[0,x],widthRange:[.8,3.2],widthScale:`linear`}:{}})),!c&&t.focusOutlineCount&&_.push(new M({id:`weights-focus-outline`,coordinateOrigin:s,segments:t.focusOutline,instanceCount:t.focusOutlineCount,widthPixels:2.5,color:h,outlineColor:g,outlineWidthPixels:1})),!c){let e={coordinateOrigin:s,analysisFrame:u,positions:t.core.positions,instanceCount:i.count,values:t.space.isolate,valueFormat:`uint32`,colormap:`mask`,noDataColor:[0,0,0,0],shape:`ring`};_.push(new ee({...e,id:`weights-island-halo`,color:g,radiusPixels:7.5,outlineWidthPixels:4}),new ee({...e,id:`weights-island-ring`,color:h,radiusPixels:6.5,outlineWidthPixels:1.8}))}return _},xt=(t,n)=>{let r=e.options,{geography:i}=t,a=H(r),o=V(r.variable),s=w(r.variable),c=t.snapshot,l=m??T(t),u=i.getVariable(r.variable)[n],d=Number.isFinite(u)?s(u):`no data`,f=c?.cardinality[n],p=c?.lag[n],h=c!==null&&f!==void 0&&f>0&&Number.isFinite(p)&&Number.isFinite(u),g=(e,t,n,r,i)=>{let a=ie(t,n);return{label:e,value:i,unit:r,swatch:a>=0?t.colors[a]:void 0,emphasis:!0}},_=[],v=t.focusData,y=v?v.ids.indexOf(n):-1,S=v?y>=0&&y<v.degree:!1,C=S?v.weights[y]:null;switch(n===t.focusRow?_.push({label:`Role`,value:`The focus place`,emphasis:!0}):(a===`focus`||a===`weights`)&&_.push({label:`Role`,value:S?`Neighbour of the focus`:`Not a neighbour`,emphasis:!0}),a){case`neighbors`:f!==void 0&&_.push(g(`Neighbours (${ht[r.source]})`,l.cardinality,f,f===1?`neighbour`:`neighbours`,B(f)));break;case`oneWay`:c&&_.push(g(`One-way links`,l.oneWay,c.oneWay[n],`not returned`,B(c.oneWay[n])));break;case`value`:_.push(g(o.label,l.value,u,o.unit,d));break;case`lag`:h&&_.push(g(`Neighbourhood (lag)`,l.value,p,o.unit,s(p)));break;case`difference`:h&&_.push(g(`Lag minus value`,l.difference,p-u,o.unit,`${p-u>=0?`+`:``}${s(p-u)}`));break;default:break}a===`weights`&&C!==null&&_.push({label:`Weight in the focus row`,value:C.toFixed(3),emphasis:!0}),a!==`neighbors`&&f!==void 0&&_.push({label:`Neighbours (${ht[r.source]})`,value:B(f),unit:f===1?`neighbour`:`neighbours`}),a!==`value`&&_.push({label:o.label,value:d,unit:o.unit}),c&&a!==`lag`&&a!==`difference`&&h&&_.push({label:`Neighbourhood (lag)`,value:s(p),unit:o.unit});let E=Ct(t);Number.isFinite(u)&&E.length&&_.push({label:`Rank`,value:`${ue(wt(E,u)*100)} percentile`,unit:`of ${i.unitPlural}`});let D=c?.modes[n];return{title:i.getName(n),subtitle:i.getGroupName(n),rows:_,note:f===0?`Island: no neighbours under this rule`:D!==void 0&&D!==rt&&a===`summary`?`Dominant neighbour group: ${i.groupNames[D]??D}`:void 0,anchor:x(t,n),highlight:{kind:`polygon`,rings:b(t,n)}}},St=new WeakMap,Ct=t=>{let n=e.options.variable,r=St.get(t);if(r?.variable===n)return r.sorted;let i=Float32Array.from(t.geography.getVariable(n).filter(Number.isFinite)).sort();return St.set(t,{variable:n,sorted:i}),i},wt=(e,t)=>{let n=0,r=e.length;for(;n<r;){let i=n+r>>>1;e[i]<=t?n=i+1:r=i}return n/e.length};return{getCompiledGraphs(){let e=o;return e?[...e.core.getGraphs(),...e.algebra.values(),...e.space.getGraphs(),e.gatherer.graph,...e.reference?[...e.reference.queen.getGraphs(),...e.reference.rook.getGraphs()]:[],...e.lattice?[...e.lattice.producers.values(),...e.lattice.space.getGraphs()]:[]]:[]},setOption(t,n){l++;let r=o;if(t===`geography`){et(n);return}if(r){if(t===`showLinks`||t===`showOutlines`){e.requestLayers();return}[`display`,`summary`,`source`,`lattice`,`variable`].includes(t)&&(g=null),t===`variable`&&(ve(r),O()),t===`focus`&&ye(r),t===`populationPercentile`&&U(r),t===`selfWeight`&&Ie(r),t===`latticeMask`&&r.lattice&&W(r.lattice),t===`focus`&&(c=!0),i.request({build:()=>Ue(r),commit:()=>{o===r&&(J(r),We(),c&&Je(r),c=!1,u=!0,d=!0,f=!0,Xe(),Q(),e.requestLayers())}})}},encode(t,n){let r=o;if(!r||i.pending)return;let a=e.options;if(u||n.frameIndex<2){let e=Number(a.focalWeight);if(a.lattice&&r.lattice){let n=r.lattice;n.producers.get(`${a.latticeCriterion}-${a.latticeRadius}`)?.encode(t,{parameters:void 0}),n.space.encode(t,{normalizeLag:a.lagNormalize,matrix:a.matrix,display:H(a),summary:a.summary,includeFocal:a.includeFocal,focalWeight:e,rowStandardized:!1})}else{r.core.encode(t,v(r.geography));let n=y();n?(G()&&r.partner.encode(t,a.partnerK),r.algebra.get(n)?.encode(t,{parameters:void 0})):$(t,r.core.csr,r.space.csr,r.geography.count,r.core.slots);let i=H(a);r.space.encode(t,{normalizeLag:a.lagNormalize,matrix:a.matrix,display:i===`weights`?`focus`:i===`difference`?`lag`:i,summary:a.summary,includeFocal:a.includeFocal,focalWeight:e,rowStandardized:q(r)}),r.gatherer.encode(t);let o=r.reference;a.showQueenOnly&&o&&!o.encoded&&(o.queen.encode(t,Ve(r,`queen`)),o.rook.encode(t,Ve(r,`rook`)),o.encoded=!0,p=!0)}u=!1,d=!0,f=!0}n.frameIndex>=1&&Ze(t)},getLayers:bt,onGroundChange:()=>{O();let t=o;t?.snapshot&&Ke(t,t.snapshot,!1),e.requestLayers()},onLegendFilter(t,n){g=n===null?null:[...n],e.requestLayers()},getTooltip(t){let n=o;if(!n)return null;let r=e.options;if(r.lattice&&n.lattice){let e=ot(t);if(e<0)return null;let i=n.lattice.featureOfCell[e];if(i<0&&r.latticeMask)return null;let a=n.latticeSnapshot;return{title:i>=0?n.geography.getName(i):`Outside the map`,subtitle:i>=0?n.geography.getGroupName(i):`Empty grid cell`,rows:a?[{label:`Neighbours`,value:B(a.cardinality[e]),unit:`cells`,emphasis:!0}]:[]}}let i=at(t);return i<0?null:xt(n,i)},onClick(t){let n=o;if(!n)return!1;if(e.options.lattice&&n.lattice){let e=ot(t);if(e<0)return!1;n.lattice.focusRow=e,n.lattice.space.kit.setFocusRow(e)}else{let e=at(t);if(e<0)return!1;n.focusRow=e,n.space.kit.setFocusRow(e),Je(n)}return u=!0,d=!0,f=!0,Xe(),!0},destroy(){r=!0,s++,i.destroy();let e=o;o=null,e&&(Qe(e),e.resources.destroy())}}}export{yt as createSpatialWeights};