import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,d as r,f as i,h as a,l as o,t as s,v as c,w as l,y as u}from"./resources-CwvhEOBe.js";import{t as d}from"./gpu-sort-Bl3Dztdb.js";import{t as f}from"./gpu-compaction-Cpx1OCRM.js";import{o as p,r as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{r as g}from"./wgsl-common-BNRVzgd8.js";import{c as _}from"./spatial-weights-D4bd9AQ6.js";import{t as v}from"./load-balanced-for-each-OquID6hm.js";import{t as y}from"./gpu-contiguity-weights-BI6hkB9v.js";import{n as b,t as x}from"./gpu-kmeans-Ba8e911_.js";import{c as S}from"./index-Bjk18V2I.js";import{t as C}from"./graph-buffers-B4_80RlX.js";import{t as w}from"./graph-swap-41Ru7LVu.js";import{_ as T,a as E,c as D,d as O,f as k,i as A,o as j,p as M,r as N,s as P,t as F,u as I,v as L,y as R}from"./b6-polygons-Bf2djBQx.js";import{n as z,t as B}from"./b6-fill-layer-CwuOmgDO.js";import{t as ee}from"./b6-colors-D7TnwgyY.js";var V=`
// Slot of row \`row\` whose neighbor is \`needle\`, or NONE. Rows are strictly ascending.
fn findSlot(row: u32, needle: u32) -> u32 {
  let rowEnd = offsets[offsetsOffset + row + 1u];
  var low = offsets[offsetsOffset + row];
  var high = rowEnd;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < needle) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowEnd && neighbors[neighborsOffset + low] == needle) {
    return low;
  }
  return NONE;
}

// First row whose slot range ends after \`slot\` (the row that owns the slot).
fn getRowOfSlot(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}
`,te=`const NONE: u32 = 0xffffffffu;`,H=`GPUSpatialWeightsMinimumSpanningTree`,U=256,ne=class{id;props;constructor(e){let t=e.id??`minimum-spanning-tree`;this.id=t,this.props=e;let n=_(t,e.weights),o=e.columnCount??1,{output:s}=e;if(r(t,`columnCount`,o,1,32),u(e.values,[`float32`],`${t} values`),e.values.length!==n*o)throw Error(`${t} values length must equal rows * columnCount`);if(u(s.treeEdgeFlags,[`uint32`],`${t} output.treeEdgeFlags`),s.treeEdgeFlags.length!==e.weights.neighbors.length)throw Error(`${t} output.treeEdgeFlags length must equal the weights slot capacity`);if(u(s.componentLabels,[`uint32`],`${t} output.componentLabels`),s.componentLabels.length!==n)throw Error(`${t} output.componentLabels length must equal the weights row count`);if(s.standardizedValues){if(u(s.standardizedValues,[`float32`],`${t} output.standardizedValues`),s.standardizedValues.length!==n*o)throw Error(`${t} output.standardizedValues length must equal rows * columnCount`);if(e.standardize===!1)throw Error(`${t} output.standardizedValues requires standardize`)}if(s.edges)i(t,s.edges);else if(s.edgeEndpoints||s.edgeDissimilarities)throw Error(`${t} edge details require output.edges`);let c=s.edges?.ids.length??0;if(s.edgeEndpoints&&(u(s.edgeEndpoints,[`uint32`],`${t} output.edgeEndpoints`),s.edgeEndpoints.length<2*c))throw Error(`${t} output.edgeEndpoints must hold two entries per edge slot`);if(s.edgeDissimilarities&&(u(s.edgeDissimilarities,[`float32`],`${t} output.edgeDissimilarities`),s.edgeDissimilarities.length<c))throw Error(`${t} output.edgeDissimilarities must hold one entry per edge slot`);a(t,[s.treeEdgeFlags,s.componentLabels,s.standardizedValues,s.edges?.ids,s.edges?.status.count,s.edges?.status.overflow,s.edgeEndpoints,s.edgeDissimilarities],[e.values,e.weights.offsets,e.weights.neighbors])}getCommandNodes(e){let{id:t,props:r}=this,{weights:i,output:a}=r,{treeEdgeFlags:o,componentLabels:s}=a;n(t,e,[r.values,o,s,a.standardizedValues,a.edges?.ids,a.edges?.status.count,a.edges?.status.overflow,a.edges?.status.requiredCount,a.edgeEndpoints,a.edgeDissimilarities,i.offsets,i.neighbors]);let c=i.offsets.length-1,u=i.neighbors.length,d=r.columnCount??1,p=r.standardize!==!1,g=[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`}],_=`const ROWS: u32 = ${c}u;
const COLUMNS: u32 = ${d}u;
${te}
${V}`,v=[],y=r.values;p&&(y=a.standardizedValues??l(e,`${t}-table`,`float32`,c*d),v.push(h(e,{id:`${t}-standardize`,operation:H,variant:`standardize`,bindings:[{name:`values`,view:r.values,type:`f32`,access:`read`},{name:`scaled`,view:y,type:`f32`,access:`read_write`}],workgroupSize:U,invocationCount:d*U,guardIndex:!1,declarations:`const ROWS: u32 = ${c}u;
const COLUMNS: u32 = ${d}u;
const LANES: u32 = ${U}u;
var<workgroup> partials: array<f32, ${U}>;

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
    }
    workgroupBarrier();
  }
  let total = partials[0];
  workgroupBarrier();
  return total;
}`,body:`let column = index / LANES;
  let lane = localInvocationIndex;
  // Center before reducing: a rounded mean of a constant column can otherwise
  // create a nonzero deviation and normalize every row to +1 or -1.
  let origin = values[valuesOffset + column];
  var sum = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    sum += values[valuesOffset + row * COLUMNS + column] - origin;
  }
  let mean = reduceLanes(lane, sum) / f32(ROWS);
  var squares = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let delta = (values[valuesOffset + row * COLUMNS + column] - origin) - mean;
    squares += delta * delta;
  }
  let deviation = sqrt(reduceLanes(lane, squares) / f32(ROWS));
  let inverse = select(0.0, 1.0 / deviation, deviation > 1e-20);
  for (var row = lane; row < ROWS; row += LANES) {
    scaled[scaledOffset + row * COLUMNS + column] =
      ((values[valuesOffset + row * COLUMNS + column] - origin) - mean) * inverse;
  }`})));let b=l(e,`${t}-slot-cost`,`float32`,u),x=l(e,`${t}-slot-edge`,`uint32`,u),S=l(e,`${t}-component`,`uint32`,c),C=l(e,`${t}-best-cost`,`uint32`,c),w=l(e,`${t}-best-edge`,`uint32`,c),T=l(e,`${t}-parent-a`,`uint32`,c),E=l(e,`${t}-parent-b`,`uint32`,c),D=l(e,`${t}-minimum-label`,`uint32`,c);v.push(h(e,{id:`${t}-clear-flags`,operation:H,variant:`clear`,bindings:[{name:`flags`,view:o,type:`u32`,access:`read_write`}],invocationCount:u,body:`flags[flagsOffset + index] = 0u;`}),h(e,{id:`${t}-seed-components`,operation:H,variant:`seed`,bindings:[{name:`component`,view:S,type:`u32`,access:`read_write`},{name:`bestCost`,view:C,type:`u32`,access:`read_write`},{name:`bestEdge`,view:w,type:`u32`,access:`read_write`}],invocationCount:c,body:`component[componentOffset + index] = index;
  bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`}),h(e,{id:`${t}-slot-costs`,operation:H,variant:`costs`,bindings:[...g,{name:`table`,view:y,type:`f32`,access:`read`},{name:`slotCost`,view:b,type:`f32`,access:`read_write`},{name:`slotEdge`,view:x,type:`u32`,access:`read_write`}],invocationCount:c,declarations:_,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let low = min(index, neighbor);
    let high = max(index, neighbor);
    var cost = 0.0;
    for (var column = 0u; column < COLUMNS; column++) {
      let delta = table[tableOffset + low * COLUMNS + column] - table[tableOffset + high * COLUMNS + column];
      cost += delta * delta;
    }
    slotCost[slotCostOffset + slot] = cost;
    var edge = slot;
    if (neighbor < index) {
      let twin = findSlot(neighbor, index);
      if (twin != NONE) {
        edge = twin;
      }
    }
    slotEdge[slotEdgeOffset + slot] = edge;
  }`}));let O=Math.max(1,Math.ceil(Math.log2(c)));for(let n=0;n<O;n++){let r=`${t}-round-${n}`;v.push(h(e,{id:`${r}-cheapest-cost`,operation:H,variant:`cheapest-cost`,bindings:[...g,{name:`component`,view:S,type:`u32`,access:`read`},{name:`slotCost`,view:b,type:`f32`,access:`read`},{name:`bestCost`,view:C,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:`const ROWS: u32 = ${c}u;`,body:`let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own) {
      atomicMin(&bestCost[bestCostOffset + own], bitcast<u32>(slotCost[slotCostOffset + slot]));
    }
  }`}),h(e,{id:`${r}-cheapest-edge`,operation:H,variant:`cheapest-edge`,bindings:[...g,{name:`component`,view:S,type:`u32`,access:`read`},{name:`slotCost`,view:b,type:`f32`,access:`read`},{name:`slotEdge`,view:x,type:`u32`,access:`read`},{name:`bestCost`,view:C,type:`u32`,access:`read`},{name:`bestEdge`,view:w,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:`const ROWS: u32 = ${c}u;`,body:`let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own &&
        bitcast<u32>(slotCost[slotCostOffset + slot]) == bestCost[bestCostOffset + own]) {
      atomicMin(&bestEdge[bestEdgeOffset + own], slotEdge[slotEdgeOffset + slot]);
    }
  }`}),h(e,{id:`${r}-hook`,operation:H,variant:`hook`,bindings:[...g,{name:`component`,view:S,type:`u32`,access:`read`},{name:`bestEdge`,view:w,type:`u32`,access:`read`},{name:`parentOut`,view:E,type:`u32`,access:`read_write`},{name:`flags`,view:o,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:_,body:`var parent = index;
  let edge = bestEdge[bestEdgeOffset + index];
  if (component[componentOffset + index] == index && edge != NONE) {
    let first = component[componentOffset + getRowOfSlot(edge)];
    let second = component[componentOffset + neighbors[neighborsOffset + edge]];
    parent = select(first, second, first == index);
    let isMutual = bestEdge[bestEdgeOffset + parent] == edge;
    parent = select(parent, index, isMutual && index < parent);
    atomicStore(&flags[flagsOffset + edge], 1u);
  }
  parentOut[parentOutOffset + index] = parent;`}));let i=Math.max(1,Math.ceil(Math.log2(Math.max(2,Math.ceil(c/2**n)))))+1,a=E,s=T;for(let t=0;t<i-1;t++)v.push(h(e,{id:`${r}-jump-${t}`,operation:H,variant:`jump`,bindings:[{name:`parentIn`,view:a,type:`u32`,access:`read`},{name:`parentOut`,view:s,type:`u32`,access:`read_write`}],invocationCount:c,body:`parentOut[parentOutOffset + index] =
    parentIn[parentInOffset + parentIn[parentInOffset + index]];`})),[a,s]=[s,a];let l=n===O-1;v.push(h(e,{id:`${r}-relabel`,operation:H,variant:`relabel`,bindings:[{name:`root`,view:a,type:`u32`,access:`read`},{name:`component`,view:S,type:`u32`,access:`read_write`},...l?[{name:`minimumLabel`,view:D,type:`u32`,access:`read_write`}]:[{name:`bestCost`,view:C,type:`u32`,access:`read_write`},{name:`bestEdge`,view:w,type:`u32`,access:`read_write`}]],invocationCount:c,body:`let label = component[componentOffset + index];
  component[componentOffset + index] = root[rootOffset + root[rootOffset + label]];
  ${l?`minimumLabel[minimumLabelOffset + index] = 0xffffffffu;`:`bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`}`}))}if(v.push(h(e,{id:`${t}-label-minimum`,operation:H,variant:`label-minimum`,bindings:[{name:`component`,view:S,type:`u32`,access:`read`},{name:`minimumLabel`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,body:`atomicMin(&minimumLabel[minimumLabelOffset + component[componentOffset + index]], index);`}),h(e,{id:`${t}-label-write`,operation:H,variant:`label-write`,bindings:[{name:`component`,view:S,type:`u32`,access:`read`},{name:`minimumLabel`,view:D,type:`u32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read_write`}],invocationCount:c,body:`labels[labelsOffset + index] = minimumLabel[minimumLabelOffset + component[componentOffset + index]];`})),a.edges){let{edges:n}=a,r=l(e,`${t}-slot-ids`,`uint32`,u),i=l(e,`${t}-compact-slots`,`uint32`,u),s=l(e,`${t}-edge-total`,`uint32`,1);if(v.push(h(e,{id:`${t}-slot-ids`,operation:H,variant:`slot-ids`,bindings:[{name:`slotIds`,view:r,type:`u32`,access:`read_write`}],invocationCount:u,body:`slotIds[slotIdsOffset + index] = index;`}),...new f({id:`${t}-compaction`,input:r,flags:o,output:i,count:s}).getCommandNodes(e),m(e,{id:`${t}-publish`,operation:H,requiredCount:s,compactIds:i,output:n})),a.edgeEndpoints||a.edgeDissimilarities){let r=[...g,{name:`edgeIds`,view:n.ids,type:`u32`,access:`read`},{name:`edgeCount`,view:n.status.count,type:`u32`,access:`read`},{name:`slotCost`,view:b,type:`f32`,access:`read`},...a.edgeEndpoints?[{name:`endpoints`,view:a.edgeEndpoints,type:`u32`,access:`read_write`}]:[],...a.edgeDissimilarities?[{name:`dissimilarities`,view:a.edgeDissimilarities,type:`f32`,access:`read_write`}]:[]];v.push(h(e,{id:`${t}-edge-details`,operation:H,variant:`edge-details`,bindings:r,invocationCount:n.ids.length,declarations:_,body:`if (index >= edgeCount[edgeCountOffset]) {
    return;
  }
  let slot = edgeIds[edgeIdsOffset + index];
  ${a.edgeEndpoints?`endpoints[endpointsOffset + 2u * index] = getRowOfSlot(slot);
  endpoints[endpointsOffset + 2u * index + 1u] = neighbors[neighborsOffset + slot];`:``}
  ${a.edgeDissimilarities?`dissimilarities[dissimilaritiesOffset + index] = slotCost[slotCostOffset + slot];`:``}`}))}}return v}},W=`GPUSkaterRegions`;function re(e){return o(e,4)}var G=0,ie=1;function ae(e){let t=e.minimumRegionSize??1;r(`GPUSkaterRegions`,`regionCount`,e.regionCount),r(`GPUSkaterRegions`,`minimumRegionSize`,t);let n=new Uint32Array(2);return n[G]=e.regionCount,n[ie]=t,n}var oe=0,se=1,ce=2,le=3,ue=4,de=5,K=6,q=7,fe=8,pe=9,me=class{id;props;constructor(e){let t=e.id??`skater-regions`;this.id=t,this.props=e;let n=_(t,e.weights),i=e.columnCount??1,{output:o}=e;if(r(t,`columnCount`,i,1,16),r(t,`maximumRegionCount`,e.maximumRegionCount,2),u(e.values,[`float32`],`${t} values`),e.values.length!==n*i)throw Error(`${t} values length must equal rows * columnCount`);if(u(e.treeEdgeFlags,[`uint32`],`${t} treeEdgeFlags`),e.treeEdgeFlags.length!==e.weights.neighbors.length)throw Error(`${t} treeEdgeFlags length must equal the weights slot capacity`);if(u(e.componentLabels,[`uint32`],`${t} componentLabels`),e.componentLabels.length!==n)throw Error(`${t} componentLabels length must equal the weights row count`);if(u(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<2)throw Error(`${t} parameters must hold 2 words`);if(u(o.labels,[`uint32`],`${t} output.labels`),o.labels.length!==n)throw Error(`${t} output.labels length must equal the weights row count`);o.regionCount&&u(o.regionCount,[`uint32`],`${t} output.regionCount`);let s=e.maximumRegionCount-1;if(o.cutEdges&&(u(o.cutEdges,[`uint32`],`${t} output.cutEdges`),o.cutEdges.length<s))throw Error(`${t} output.cutEdges must hold maximumRegionCount - 1 entries`);if(o.cutGains&&(u(o.cutGains,[`float32`],`${t} output.cutGains`),o.cutGains.length<s))throw Error(`${t} output.cutGains must hold maximumRegionCount - 1 entries`);a(t,[o.labels,o.regionCount,o.cutEdges,o.cutGains],[e.values,e.parameters,e.treeEdgeFlags,e.componentLabels,e.weights.offsets,e.weights.neighbors])}getCommandNodes(e){let{id:t,props:r}=this,{weights:i,values:a,output:o}=r,{labels:s}=o;n(t,e,[a,r.treeEdgeFlags,r.componentLabels,r.parameters,s,o.regionCount,o.cutEdges,o.cutGains,i.offsets,i.neighbors]);let u=i.offsets.length-1,d=i.neighbors.length,f=r.columnCount??1,m=re(e.device);c(t,`weights.neighbors rows x (2 + columnCount)`,d*(2+f),m),c(t,`weights rows x (2 + columnCount)`,u*(2+f),m);let g=r.maximumRegionCount-1,_=2+f,v=pe+2*g,y=l(e,`${t}-twin`,`uint32`,d),b=l(e,`${t}-arc-flags`,`uint32`,d),x=l(e,`${t}-arc-info`,`uint32`,d),S=l(e,`${t}-gain-key`,`uint32`,d),C=l(e,`${t}-rank-next-a`,`uint32`,d),w=l(e,`${t}-rank-next-b`,`uint32`,d),T=l(e,`${t}-rank-distance-a`,`uint32`,d),E=l(e,`${t}-rank-distance-b`,`uint32`,d),D=l(e,`${t}-arc-stats`,`float32`,d*_),O=l(e,`${t}-row-down-arc`,`uint32`,u),k=l(e,`${t}-region-stats`,`float32`,u*_),A=l(e,`${t}-state`,`uint32`,v),j=p(e,{id:`${t}-step-dispatch`,operation:W,shapes:[[d],[u]]}),M=[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`}],N=`const ROWS: u32 = ${u}u;
const COLUMNS: u32 = ${f}u;
const STRIDE: u32 = ${_}u;
const STEP_COUNT: u32 = ${g}u;
const MAXIMUM_REGIONS: u32 = ${r.maximumRegionCount}u;
const STATE_BEST_KEY: u32 = ${oe}u;
const STATE_BEST_EDGE: u32 = ${se}u;
const STATE_REGION_COUNT: u32 = ${ce}u;
const STATE_APPLIED: u32 = ${le}u;
const STATE_TARGET: u32 = ${ue}u;
const STATE_MINIMUM_SIZE: u32 = ${de}u;
const STATE_OLD_REGION: u32 = ${K}u;
const STATE_CHILD: u32 = ${q}u;
const STATE_STEP: u32 = ${fe}u;
const STATE_LOG: u32 = ${pe}u;
const SLOT_MASK: u32 = 0x3fffffffu;
${te}`,P=`${V}
fn isLive(flags: u32) -> bool { return (flags & 1u) != 0u; }

// Successor of an arc in the Euler tour of the live arcs: the next live arc after the twin of
// the arc around the arc's head, cyclically.
fn getSuccessor(arc: u32) -> u32 {
  let head = neighbors[neighborsOffset + arc];
  let twinArc = twin[twinOffset + arc];
  let rowStart = offsets[offsetsOffset + head];
  let degree = offsets[offsetsOffset + head + 1u] - rowStart;
  for (var step = 1u; step <= degree; step++) {
    let slot = rowStart + (twinArc - rowStart + step) % degree;
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return twinArc;
}

// First live arc of a row, or NONE.
fn getFirstLiveArc(row: u32) -> u32 {
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    if (isLive(arcFlags[arcFlagsOffset + slot])) {
      return slot;
    }
  }
  return NONE;
}`,F=N,I=`${N}\n${P}`,L=`${N}\nfn getInfoSuccessor(info: u32) -> u32 { return info & SLOT_MASK; }
fn isInfoLive(info: u32) -> bool { return ((info >> 30u) & 1u) != 0u; }
fn isInfoDown(info: u32) -> bool { return ((info >> 31u) & 1u) != 0u; }
fn isInfoLiveFlag(flags: u32) -> bool { return (flags & 1u) != 0u; }
fn getSsd(count: f32, squares: f32, sumSquares: f32) -> f32 {
  return squares - sumSquares / max(count, 1.0);
}`,R=[];R.push(h(e,{id:`${t}-clear-arcs`,operation:W,variant:`clear-arcs`,bindings:[{name:`arcFlags`,view:b,type:`u32`,access:`read_write`}],invocationCount:d,body:`arcFlags[arcFlagsOffset + index] = 0u;`})),R.push(h(e,{id:`${t}-twins`,operation:W,variant:`twins`,bindings:[...M,{name:`treeFlags`,view:r.treeEdgeFlags,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read_write`},{name:`arcFlags`,view:b,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`${N}\n${V}`,body:`for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let twinSlot = findSlot(neighbor, index);
    twin[twinOffset + slot] = twinSlot;
    var edge = slot;
    if (twinSlot != NONE && twinSlot < slot) {
      edge = twinSlot;
    }
    arcFlags[arcFlagsOffset + slot] =
      select(0u, 1u, twinSlot != NONE && treeFlags[treeFlagsOffset + edge] != 0u);
  }`})),R.push(h(e,{id:`${t}-rank-init`,operation:W,variant:`rank-init`,bindings:[...M,{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`treeRoots`,view:r.componentLabels,type:`u32`,access:`read`},{name:`nextOut`,view:C,type:`u32`,access:`read_write`},{name:`distanceOut`,view:T,type:`u32`,access:`read_write`}],invocationCount:d,declarations:I,body:`if (!isLive(arcFlags[arcFlagsOffset + index])) {
    nextOut[nextOutOffset + index] = index;
    distanceOut[distanceOutOffset + index] = 0u;
    return;
  }
  let successor = getSuccessor(index);
  let root = treeRoots[treeRootsOffset + neighbors[neighborsOffset + index]];
  let isLast = successor == getFirstLiveArc(root);
  nextOut[nextOutOffset + index] = select(successor, index, isLast);
  distanceOut[distanceOutOffset + index] = select(1u, 0u, isLast);`}));let z={next:C,distance:T},B={next:w,distance:E},ee=Math.max(1,Math.ceil(Math.log2(Math.max(2,2*u))));for(let n=0;n<ee;n++)R.push(h(e,{id:`${t}-rank-jump-${n}`,operation:W,variant:`rank-jump`,bindings:[{name:`nextIn`,view:z.next,type:`u32`,access:`read`},{name:`distanceIn`,view:z.distance,type:`u32`,access:`read`},{name:`nextOut`,view:B.next,type:`u32`,access:`read_write`},{name:`distanceOut`,view:B.distance,type:`u32`,access:`read_write`}],invocationCount:d,body:`let successor = nextIn[nextInOffset + index];
  distanceOut[distanceOutOffset + index] =
    distanceIn[distanceInOffset + index] + distanceIn[distanceInOffset + successor];
  nextOut[nextOutOffset + index] = nextIn[nextInOffset + successor];`})),[z,B]=[B,z];R.push(h(e,{id:`${t}-orient`,operation:W,variant:`orient`,bindings:[{name:`twin`,view:y,type:`u32`,access:`read`},{name:`distance`,view:z.distance,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read_write`}],invocationCount:d,body:`let flags = arcFlags[arcFlagsOffset + index];
  if ((flags & 1u) != 0u &&
      distance[distanceOffset + index] > distance[distanceOffset + twin[twinOffset + index]]) {
    arcFlags[arcFlagsOffset + index] = flags | 2u;
  }`})),R.push(h(e,{id:`${t}-init-state`,operation:W,variant:`init-state`,bindings:[{name:`parameters`,view:r.parameters,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read_write`},j.binding],invocationCount:v,declarations:F,body:`if (index == 0u) {
    ${j.armWGSL}
  }
  var word = 0u;
  if (index == STATE_TARGET) {
    word = min(parameters[parametersOffset + ${G}u], MAXIMUM_REGIONS);
  } else if (index == STATE_MINIMUM_SIZE) {
    word = max(parameters[parametersOffset + ${ie}u], 1u);
  } else if (index >= STATE_LOG && ((index - STATE_LOG) % 2u) == 0u) {
    word = NONE;
  }
  state[stateOffset + index] = word;`}),h(e,{id:`${t}-init-labels`,operation:W,variant:`init-labels`,bindings:[{name:`treeRoots`,view:r.componentLabels,type:`u32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read_write`},{name:`state`,view:A,type:`atomic<u32>`,access:`read_write`}],invocationCount:u,declarations:F,body:`let root = treeRoots[treeRootsOffset + index];
  labels[labelsOffset + index] = root;
  if (root == index) {
    atomicAdd(&state[stateOffset + STATE_REGION_COUNT], 1u);
  }`}));let H=z.distance;R.push(h(e,{id:`${t}-successors`,operation:W,variant:`successors`,bindings:[...M,{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`arcInfo`,view:x,type:`u32`,access:`read_write`}],invocationCount:d,declarations:I,body:`let flags = arcFlags[arcFlagsOffset + index];
  var info = (flags & 3u) << 30u;
  if (isLive(flags)) {
    info = info | getSuccessor(index);
  }
  arcInfo[arcInfoOffset + index] = info;`}),h(e,{id:`${t}-row-down-arcs`,operation:W,variant:`row-down-arcs`,bindings:[...M,{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`rowDownArc`,view:O,type:`u32`,access:`read_write`}],invocationCount:u,declarations:L,body:`var downArc = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let flags = arcFlags[arcFlagsOffset + slot];
    if (isInfoLiveFlag(flags) && (flags & 2u) == 0u) {
      downArc = twin[twinOffset + slot];
    }
  }
  rowDownArc[rowDownArcOffset + index] = downArc;`}),h(e,{id:`${t}-tree-stats`,operation:W,variant:`tree-stats`,bindings:[...M,{name:`arcInfo`,view:x,type:`u32`,access:`read`},{name:`values`,view:a,type:`f32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read`},{name:`regionStats`,view:k,type:`f32`,access:`read_write`}],invocationCount:u,declarations:L,body:`if (labels[labelsOffset + index] != index) {
    return;
  }
  var count = 1.0;
  var squares = 0.0;
  var sums: array<f32, ${f}>;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + index * COLUMNS + column];
    sums[column] = value;
    squares += value * value;
  }
  var head = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (isInfoLive(arcInfo[arcInfoOffset + slot])) {
      head = slot;
      break;
    }
  }
  if (head != NONE) {
    var arc = head;
    for (var guard = 0u; guard < 2u * ROWS; guard++) {
      let info = arcInfo[arcInfoOffset + arc];
      if (isInfoDown(info)) {
        let row = neighbors[neighborsOffset + arc];
        count += 1.0;
        for (var column = 0u; column < COLUMNS; column++) {
          let value = values[valuesOffset + row * COLUMNS + column];
          sums[column] += value;
          squares += value * value;
        }
      }
      arc = getInfoSuccessor(info);
      if (arc == head) {
        break;
      }
    }
  }
  let base = regionStatsOffset + index * STRIDE;
  regionStats[base] = count;
  regionStats[base + 1u] = squares;
  for (var column = 0u; column < COLUMNS; column++) {
    regionStats[base + 2u + column] = sums[column];
  }`}),h(e,{id:`${t}-arc-stats`,operation:W,variant:`arc-stats`,bindings:[{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcInfo`,view:x,type:`u32`,access:`read`},{name:`values`,view:a,type:`f32`,access:`read`},{name:`arcStats`,view:D,type:`f32`,access:`read_write`}],invocationCount:d,declarations:L,body:`let startInfo = arcInfo[arcInfoOffset + index];
  if (!isInfoLive(startInfo) || !isInfoDown(startInfo)) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let end = twin[twinOffset + index];
  var count = 1.0;
  var squares = 0.0;
  var sums: array<f32, ${f}>;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + child * COLUMNS + column];
    sums[column] = value;
    squares += value * value;
  }
  var arc = getInfoSuccessor(startInfo);
  for (var guard = 0u; guard < 2u * ROWS && arc != end; guard++) {
    let info = arcInfo[arcInfoOffset + arc];
    if (isInfoDown(info)) {
      let row = neighbors[neighborsOffset + arc];
      count += 1.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        sums[column] += value;
        squares += value * value;
      }
    }
    arc = getInfoSuccessor(info);
  }
  let base = arcStatsOffset + index * STRIDE;
  arcStats[base] = count;
  arcStats[base + 1u] = squares;
  for (var column = 0u; column < COLUMNS; column++) {
    arcStats[base + 2u + column] = sums[column];
  }`}));for(let n=0;n<g;n++){let r=`${t}-step-${n}`;R.push(h(e,{id:`${r}-candidates`,operation:W,variant:`candidates`,...j.getGate(0),bindings:[{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`arcStats`,view:D,type:`f32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read`},{name:`regionStats`,view:k,type:`f32`,access:`read`},{name:`gainKey`,view:S,type:`u32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read`}],invocationCount:d,declarations:L,body:`gainKey[gainKeyOffset + index] = 0u;
  let flags = arcFlags[arcFlagsOffset + index];
  if (!isInfoLiveFlag(flags) || (flags & 2u) == 0u ||
      state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    return;
  }
  let child = neighbors[neighborsOffset + index];
  let childBase = arcStatsOffset + index * STRIDE;
  let count = arcStats[childBase];
  let squares = arcStats[childBase + 1u];
  let base = regionStatsOffset + labels[labelsOffset + child] * STRIDE;
  let regionCount = regionStats[base];
  let remainder = regionCount - count;
  let minimum = f32(state[stateOffset + STATE_MINIMUM_SIZE]);
  if (count < minimum || remainder < minimum) {
    return;
  }
  var regionSumSquares = 0.0;
  var childSumSquares = 0.0;
  var remainderSumSquares = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let regionSum = regionStats[base + 2u + column];
    let childSum = arcStats[childBase + 2u + column];
    regionSumSquares += regionSum * regionSum;
    childSumSquares += childSum * childSum;
    let remainderSum = regionSum - childSum;
    remainderSumSquares += remainderSum * remainderSum;
  }
  let gain = getSsd(regionCount, regionStats[base + 1u], regionSumSquares)
    - getSsd(count, squares, childSumSquares)
    - getSsd(remainder, regionStats[base + 1u] - squares, remainderSumSquares);
  gainKey[gainKeyOffset + index] = bitcast<u32>(max(gain, 0.0)) + 1u;`}),h(e,{id:`${r}-reset-best`,operation:W,variant:`reset-best`,bindings:[{name:`state`,view:A,type:`u32`,access:`read_write`}],invocationCount:1,declarations:F,body:`state[stateOffset + STATE_BEST_KEY] = 0u;
  state[stateOffset + STATE_BEST_EDGE] = NONE;
  state[stateOffset + STATE_APPLIED] = 0u;
  state[stateOffset + STATE_STEP] = state[stateOffset + STATE_STEP] + 1u;`}),h(e,{id:`${r}-best-gain`,operation:W,variant:`best-gain`,...j.getGate(0),bindings:[{name:`gainKey`,view:S,type:`u32`,access:`read`},{name:`state`,view:A,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:F,body:`let key = gainKey[gainKeyOffset + index];
  if (key != 0u) {
    atomicMax(&state[stateOffset + STATE_BEST_KEY], key);
  }`}),h(e,{id:`${r}-best-edge`,operation:W,variant:`best-edge`,...j.getGate(0),bindings:[{name:`gainKey`,view:S,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read`},{name:`state`,view:A,type:`atomic<u32>`,access:`read_write`}],invocationCount:d,declarations:F,body:`let key = gainKey[gainKeyOffset + index];
  if (key != 0u && key == atomicLoad(&state[stateOffset + STATE_BEST_KEY])) {
    atomicMin(&state[stateOffset + STATE_BEST_EDGE], min(index, twin[twinOffset + index]));
  }`}),h(e,{id:`${r}-apply`,operation:W,variant:`apply`,bindings:[{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`arcStats`,view:D,type:`f32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read`},{name:`regionStats`,view:k,type:`f32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read_write`},j.binding],invocationCount:1,declarations:L,body:`let edge = state[stateOffset + STATE_BEST_EDGE];
  if (state[stateOffset + STATE_BEST_KEY] == 0u || edge == NONE) {
    ${j.disarmWGSL}
    return;
  }
  let other = twin[twinOffset + edge];
  let down = select(other, edge, (arcFlags[arcFlagsOffset + edge] & 2u) != 0u);
  let child = neighbors[neighborsOffset + down];
  let region = labels[labelsOffset + child];
  let regionBase = regionStatsOffset + region * STRIDE;
  let childBase = regionStatsOffset + child * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  regionStats[childBase] = arcStats[removedBase];
  regionStats[childBase + 1u] = arcStats[removedBase + 1u];
  regionStats[regionBase] = regionStats[regionBase] - arcStats[removedBase];
  regionStats[regionBase + 1u] = regionStats[regionBase + 1u] - arcStats[removedBase + 1u];
  for (var column = 0u; column < COLUMNS; column++) {
    let sum = arcStats[removedBase + 2u + column];
    regionStats[childBase + 2u + column] = sum;
    regionStats[regionBase + 2u + column] = regionStats[regionBase + 2u + column] - sum;
  }
  state[stateOffset + STATE_REGION_COUNT] = state[stateOffset + STATE_REGION_COUNT] + 1u;
  state[stateOffset + STATE_APPLIED] = down + 1u;
  state[stateOffset + STATE_OLD_REGION] = region;
  state[stateOffset + STATE_CHILD] = child;
  // The step index comes from the cursor, so every step shares one pipeline (rule 12).
  let logSlot = STATE_LOG + 2u * (state[stateOffset + STATE_STEP] - 1u);
  state[stateOffset + logSlot] = min(edge, other);
  state[stateOffset + logSlot + 1u] = state[stateOffset + STATE_BEST_KEY];`}),h(e,{id:`${r}-relabel`,operation:W,variant:`relabel`,...j.getGate(1),bindings:[{name:`rowDownArc`,view:O,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read`},{name:`tourDistance`,view:H,type:`u32`,access:`read`},{name:`labels`,view:s,type:`u32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read`}],invocationCount:u,declarations:F,body:`let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u || labels[labelsOffset + index] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  let down = applied - 1u;
  let child = state[stateOffset + STATE_CHILD];
  var isBelow = index == child;
  let parentArc = rowDownArc[rowDownArcOffset + index];
  if (parentArc != NONE) {
    let distance = tourDistance[tourDistanceOffset + parentArc];
    isBelow = isBelow || (distance < tourDistance[tourDistanceOffset + down] &&
      distance > tourDistance[tourDistanceOffset + twin[twinOffset + down]]);
  }
  if (isBelow) {
    labels[labelsOffset + index] = child;
  }`}),h(e,{id:`${r}-update-ancestors`,operation:W,variant:`update-ancestors`,...j.getGate(0),bindings:[{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read`},{name:`tourDistance`,view:H,type:`u32`,access:`read`},{name:`arcStats`,view:D,type:`f32`,access:`read_write`},{name:`labels`,view:s,type:`u32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read`}],invocationCount:d,declarations:L,body:`let applied = state[stateOffset + STATE_APPLIED];
  let flags = arcFlags[arcFlagsOffset + index];
  if (applied == 0u || applied - 1u == index || !isInfoLiveFlag(flags) || (flags & 2u) == 0u) {
    return;
  }
  let down = applied - 1u;
  if (labels[labelsOffset + neighbors[neighborsOffset + index]] != state[stateOffset + STATE_OLD_REGION]) {
    return;
  }
  if (!(tourDistance[tourDistanceOffset + index] > tourDistance[tourDistanceOffset + down] &&
      tourDistance[tourDistanceOffset + twin[twinOffset + index]] < tourDistance[tourDistanceOffset + twin[twinOffset + down]])) {
    return;
  }
  let base = arcStatsOffset + index * STRIDE;
  let removedBase = arcStatsOffset + down * STRIDE;
  for (var word = 0u; word < STRIDE; word++) {
    arcStats[base + word] = arcStats[base + word] - arcStats[removedBase + word];
  }`}),h(e,{id:`${r}-deactivate`,operation:W,variant:`deactivate`,bindings:[{name:`twin`,view:y,type:`u32`,access:`read`},{name:`arcFlags`,view:b,type:`u32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read`},j.binding],invocationCount:1,declarations:F,body:`if (state[stateOffset + STATE_REGION_COUNT] >= state[stateOffset + STATE_TARGET]) {
    ${j.disarmWGSL}
  }
  let applied = state[stateOffset + STATE_APPLIED];
  if (applied == 0u) {
    return;
  }
  let down = applied - 1u;
  arcFlags[arcFlagsOffset + down] = arcFlags[arcFlagsOffset + down] & ~1u;
  let up = twin[twinOffset + down];
  arcFlags[arcFlagsOffset + up] = arcFlags[arcFlagsOffset + up] & ~1u;`}))}let U=[{name:`state`,view:A,type:`u32`,access:`read`},...o.regionCount?[{name:`regionCountOut`,view:o.regionCount,type:`u32`,access:`read_write`}]:[],...o.cutEdges?[{name:`cutEdgesOut`,view:o.cutEdges,type:`u32`,access:`read_write`}]:[],...o.cutGains?[{name:`cutGainsOut`,view:o.cutGains,type:`f32`,access:`read_write`}]:[]];return U.length>1&&R.push(h(e,{id:`${t}-publish`,operation:W,variant:`publish`,bindings:U,invocationCount:Math.max(g,1),declarations:F,body:`${o.regionCount?`if (index == 0u) { regionCountOut[regionCountOutOffset] = state[stateOffset + STATE_REGION_COUNT]; }`:``}
  if (index < STEP_COUNT) {
    let edge = state[stateOffset + STATE_LOG + 2u * index];
    let key = state[stateOffset + STATE_LOG + 2u * index + 1u];
    ${o.cutEdges?`cutEdgesOut[cutEdgesOutOffset + index] = edge;`:``}
    ${o.cutGains?`cutGainsOut[cutGainsOutOffset + index] = select(0.0, bitcast<f32>(key - 1u), key != 0u);`:``}
  }`})),R}},J=`GPURegionPartitionEvaluation`,he=256;function ge(e){return o(e,8)}function _e(e){return o(e,4,1)}function ve(e){return o(e,4)}var Y={regionCount:0,withinSumOfSquares:1,betweenSumOfSquares:2,totalSumOfSquares:3,minimumSize:4,maximumSize:5,crossLinkFraction:6,ignoredRows:7,pseudoF:8,length:9},ye=class{id;props;constructor(e){let t=e.id??`region-partition-evaluation`;this.id=t,this.props=e;let n=e.columnCount??1,{output:i}=e;r(t,`columnCount`,n,1,32),u(e.values,[`float32`],`${t} values`),u(e.labels,[`uint32`],`${t} labels`);let o=e.labels.length;if(o<1||e.values.length!==o*n)throw Error(`${t} values length must equal labels.length * columnCount`);let s=e.labelCapacity??o;if(r(t,`labelCapacity`,s),e.weights&&_(t,e.weights)!==o)throw Error(`${t} weights row count must equal labels.length`);if(u(i.summary,[`float32`],`${t} output.summary`),i.summary.length<Y.length)throw Error(`${t} output.summary must hold ${Y.length} words`);if(i.regionSizes&&(u(i.regionSizes,[`uint32`],`${t} output.regionSizes`),i.regionSizes.length<s))throw Error(`${t} output.regionSizes must hold labelCapacity entries`);if(i.regionWithinSumOfSquares&&(u(i.regionWithinSumOfSquares,[`float32`],`${t} output.regionWithinSumOfSquares`),i.regionWithinSumOfSquares.length<s))throw Error(`${t} output.regionWithinSumOfSquares must hold labelCapacity entries`);a(t,[i.summary,i.regionSizes,i.regionWithinSumOfSquares],[e.values,e.labels,e.weights?.offsets,e.weights?.neighbors])}getCommandNodes(e){let{id:t,props:i}=this,{values:a,labels:o,weights:s,output:u}=i;n(t,e,[a,o,u.summary,u.regionSizes,u.regionWithinSumOfSquares,s?.offsets,s?.neighbors]);let f=o.length,p=i.columnCount??1,m=i.labelCapacity??f;c(t,`labels rows`,f,ge(e.device)),r(t,`labelCapacity`,m,1,_e(e.device)),c(t,`labelCapacity x columnCount`,m*p,ve(e.device));let _=u.regionSizes??l(e,`${t}-sizes`,`uint32`,m),y=u.regionWithinSumOfSquares??l(e,`${t}-within`,`float32`,m),b=s?l(e,`${t}-link-counts`,`uint32`,2*f):void 0,x=l(e,`${t}-label-offsets`,`uint32`,m+1),S=l(e,`${t}-region-column-sums`,`float32`,m*p),C=l(e,`${t}-region-column-deviations`,`float32`,m*p),w=`const ROWS: u32 = ${f}u;
const COLUMNS: u32 = ${p}u;
const LABELS: u32 = ${m}u;`,T=l(e,`${t}-sort-keys`,`uint32`,f),E=l(e,`${t}-row-ids`,`uint32`,f),D=l(e,`${t}-sorted-keys`,`uint32`,f),O=l(e,`${t}-sorted-rows`,`uint32`,f),k=Math.max(1,Math.ceil(Math.log2(m+1))),A=[h(e,{id:`${t}-sort-keys`,operation:J,variant:`sort-keys`,bindings:[{name:`labels`,view:o,type:`u32`,access:`read`},{name:`sortKeys`,view:T,type:`u32`,access:`read_write`},{name:`rowIds`,view:E,type:`u32`,access:`read_write`}],invocationCount:f,declarations:w,body:`let label = labels[labelsOffset + index];
  sortKeys[sortKeysOffset + index] = select(LABELS, label, label < LABELS);
  rowIds[rowIdsOffset + index] = index;`}),...new d({id:`${t}-sort`,keys:T,values:E,outputKeys:D,outputValues:O,keyBits:k}).getCommandNodes(e),h(e,{id:`${t}-label-offsets`,operation:J,variant:`label-offsets`,bindings:[{name:`sortedKeys`,view:D,type:`u32`,access:`read`},{name:`labelOffsets`,view:x,type:`u32`,access:`read_write`},{name:`sizes`,view:_,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${w}
fn getLowerBound(key: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let first = getLowerBound(index);
  let end = getLowerBound(index + 1u);
  labelOffsets[labelOffsetsOffset + index] = first;
  sizes[sizesOffset + index] = end - first;
  if (index + 1u == LABELS) {
    labelOffsets[labelOffsetsOffset + LABELS] = end;
  }`}),...v(e,{id:`${t}-region-sums`,operation:J,offsets:x,itemCapacity:f,laneCount:p,accumulatorType:`f32`,output:S,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},{name:`sortedRows`,view:O,type:`u32`,access:`read`}],declarations:`${w}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  return values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column];
}`}),...v(e,{id:`${t}-region-deviations`,operation:J,offsets:x,itemCapacity:f,laneCount:p,accumulatorType:`f32`,output:C,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},{name:`sortedRows`,view:O,type:`u32`,access:`read`},{name:`regionSums`,view:S,type:`f32`,access:`read`},{name:`sizes`,view:_,type:`u32`,access:`read`}],declarations:`${w}
fn getLoadBalancedContribution(label: u32, item: u32, column: u32) -> f32 {
  // A visited label has at least one row, so the count is positive.
  let mean = regionSums[regionSumsOffset + label * COLUMNS + column] / f32(sizes[sizesOffset + label]);
  let delta = values[valuesOffset + sortedRows[sortedRowsOffset + item] * COLUMNS + column] - mean;
  return delta * delta;
}`}),h(e,{id:`${t}-region-within`,operation:J,variant:`region-within`,bindings:[{name:`regionDeviations`,view:C,type:`f32`,access:`read`},{name:`within`,view:y,type:`f32`,access:`read_write`}],invocationCount:m,declarations:w,body:`var ssd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    ssd += regionDeviations[regionDeviationsOffset + index * COLUMNS + column];
  }
  within[withinOffset + index] = ssd;`})];s&&b&&A.push(...v(e,{id:`${t}-links`,operation:J,offsets:s.offsets,itemCapacity:s.neighbors.length,accumulatorType:`vec2<u32>`,output:b,bindings:[{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`labels`,view:o,type:`u32`,access:`read`}],declarations:`${w}
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> vec2<u32> {
  let own = labels[labelsOffset + row];
  let other = labels[labelsOffset + neighbors[neighborsOffset + item]];
  if (own >= LABELS || other >= LABELS) {
    return vec2<u32>();
  }
  return vec2<u32>(1u, select(0u, 1u, other != own));
}`}));let j=Math.ceil(f/he),M=Math.ceil(m/he),N=l(e,`${t}-row-counts`,`uint32`,j*4),P=l(e,`${t}-row-sums`,`float32`,j*p),F=l(e,`${t}-label-counts`,`uint32`,M*3),I=l(e,`${t}-label-within`,`float32`,M),L=l(e,`${t}-means`,`float32`,p),R=l(e,`${t}-deviations`,`float32`,j*p),z=`${w}
const SUMMARY_TILE: u32 = ${he}u;
const ROW_TILES: u32 = ${j}u;
const LABEL_TILES: u32 = ${M}u;`;return A.push(h(e,{id:`${t}-row-tiles`,operation:J,variant:`row-tiles`,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},{name:`labels`,view:o,type:`u32`,access:`read`},...b?[{name:`links`,view:b,type:`u32`,access:`read`}]:[],{name:`rowCounts`,view:N,type:`u32`,access:`read_write`},{name:`rowSums`,view:P,type:`f32`,access:`read_write`}],invocationCount:j,declarations:z,body:`var labelled = 0u;
  var ignored = 0u;
  var cross = 0u;
  var total = 0u;
  var sums: array<f32, ${p}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      labelled += 1u;
      ${b?`total += links[linksOffset + 2u * row]; cross += links[linksOffset + 2u * row + 1u];`:``}
      for (var column = 0u; column < COLUMNS; column++) {
        sums[column] += values[valuesOffset + row * COLUMNS + column];
      }
    } else {
      ignored += 1u;
    }
  }
  rowCounts[rowCountsOffset + 4u * index] = labelled;
  rowCounts[rowCountsOffset + 4u * index + 1u] = ignored;
  rowCounts[rowCountsOffset + 4u * index + 2u] = cross;
  rowCounts[rowCountsOffset + 4u * index + 3u] = total;
  for (var column = 0u; column < COLUMNS; column++) {
    rowSums[rowSumsOffset + index * COLUMNS + column] = sums[column];
  }`}),h(e,{id:`${t}-label-tiles`,operation:J,variant:`label-tiles`,bindings:[{name:`sizes`,view:_,type:`u32`,access:`read`},{name:`within`,view:y,type:`f32`,access:`read`},{name:`labelCounts`,view:F,type:`u32`,access:`read_write`},{name:`labelWithin`,view:I,type:`f32`,access:`read_write`}],invocationCount:M,declarations:z,body:`var regionCount = 0u;
  var withinSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  let firstLabel = index * SUMMARY_TILE;
  for (var label = firstLabel; label < min(firstLabel + SUMMARY_TILE, LABELS); label++) {
    let size = sizes[sizesOffset + label];
    if (size > 0u) {
      regionCount += 1u;
      withinSsd += within[withinOffset + label];
      minimumSize = min(minimumSize, size);
      maximumSize = max(maximumSize, size);
    }
  }
  labelCounts[labelCountsOffset + 3u * index] = regionCount;
  labelCounts[labelCountsOffset + 3u * index + 1u] = minimumSize;
  labelCounts[labelCountsOffset + 3u * index + 2u] = maximumSize;
  labelWithin[labelWithinOffset + index] = withinSsd;`}),h(e,{id:`${t}-means`,operation:J,variant:`means`,bindings:[{name:`rowCounts`,view:N,type:`u32`,access:`read`},{name:`rowSums`,view:P,type:`f32`,access:`read`},{name:`means`,view:L,type:`f32`,access:`read_write`}],invocationCount:p,declarations:z,body:`var labelled = 0u;
  var sum = 0.0;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    labelled += rowCounts[rowCountsOffset + 4u * tile];
    sum += rowSums[rowSumsOffset + tile * COLUMNS + index];
  }
  means[meansOffset + index] = sum / max(f32(labelled), 1.0);`}),h(e,{id:`${t}-deviation-tiles`,operation:J,variant:`deviation-tiles`,bindings:[{name:`values`,view:a,type:`f32`,access:`read`},{name:`labels`,view:o,type:`u32`,access:`read`},{name:`means`,view:L,type:`f32`,access:`read`},{name:`deviations`,view:R,type:`f32`,access:`read_write`}],invocationCount:j,declarations:z,body:`var squares: array<f32, ${p}>;
  let firstRow = index * SUMMARY_TILE;
  for (var row = firstRow; row < min(firstRow + SUMMARY_TILE, ROWS); row++) {
    if (labels[labelsOffset + row] < LABELS) {
      for (var column = 0u; column < COLUMNS; column++) {
        let delta = values[valuesOffset + row * COLUMNS + column] - means[meansOffset + column];
        squares[column] += delta * delta;
      }
    }
  }
  for (var column = 0u; column < COLUMNS; column++) {
    deviations[deviationsOffset + index * COLUMNS + column] = squares[column];
  }`}),h(e,{id:`${t}-summary`,operation:J,variant:`summary`,bindings:[{name:`rowCounts`,view:N,type:`u32`,access:`read`},{name:`labelCounts`,view:F,type:`u32`,access:`read`},{name:`labelWithin`,view:I,type:`f32`,access:`read`},{name:`deviations`,view:R,type:`f32`,access:`read`},{name:`summary`,view:u.summary,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${z}
${g}`,body:`var regionCount = 0u;
  var withinSsd = 0.0;
  var minimumSize = 0xffffffffu;
  var maximumSize = 0u;
  for (var tile = 0u; tile < LABEL_TILES; tile++) {
    regionCount += labelCounts[labelCountsOffset + 3u * tile];
    withinSsd += labelWithin[labelWithinOffset + tile];
    minimumSize = min(minimumSize, labelCounts[labelCountsOffset + 3u * tile + 1u]);
    maximumSize = max(maximumSize, labelCounts[labelCountsOffset + 3u * tile + 2u]);
  }
  var ignored = 0u;
  var cross = 0u;
  var total = 0u;
  for (var tile = 0u; tile < ROW_TILES; tile++) {
    ignored += rowCounts[rowCountsOffset + 4u * tile + 1u];
    cross += rowCounts[rowCountsOffset + 4u * tile + 2u];
    total += rowCounts[rowCountsOffset + 4u * tile + 3u];
  }
  var totalSsd = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    for (var tile = 0u; tile < ROW_TILES; tile++) {
      totalSsd += deviations[deviationsOffset + tile * COLUMNS + column];
    }
  }
  summary[summaryOffset + 0u] = f32(regionCount);
  summary[summaryOffset + 1u] = withinSsd;
  summary[summaryOffset + 2u] = max(totalSsd - withinSsd, 0.0);
  summary[summaryOffset + 3u] = totalSsd;
  summary[summaryOffset + 4u] = f32(select(minimumSize, 0u, regionCount == 0u));
  summary[summaryOffset + 5u] = f32(maximumSize);
  summary[summaryOffset + 6u] = select(0.0, f32(cross) / f32(total), total > 0u);
  summary[summaryOffset + 7u] = f32(ignored);
  // Calinski-Harabasz: undefined below two regions or without more rows than regions.
  let labelledRows = f32(ROWS - ignored);
  let regions = f32(regionCount);
  let between = max(totalSsd - withinSsd, 0.0);
  var pseudoF = getNaN();
  if (regionCount >= 2u && labelledRows > regions) {
    pseudoF = select(between * (labelledRows - regions) / (withinSsd * (regions - 1.0)), 1.0, withinSsd == 0.0);
  }
  summary[summaryOffset + 8u] = pseudoF;`})),A}},be=[{id:`poverty`,label:`Poverty rate`},{id:`income`,label:`ln income per capita`},{id:`uninsured`,label:`Uninsured rate`},{id:`age65`,label:`Age 65+ share`},{id:`noVehicle`,label:`No-vehicle households`},{id:`diabetes`,label:`Diabetes prevalence`},{id:`black`,label:`Black share`},{id:`hispanic`,label:`Hispanic share`}],xe=20,Se=24,X=16,Ce=24,Z=be.length;function we(e,t){let n=new Map;for(let t of e)n.set(t,(n.get(t)??0)+1);let r=new Map;for(let n=0;n<e.length;n++)for(let i of t[n])if(e[i]!==e[n]){let t=r.get(e[n])??new Set;t.add(e[i]),r.set(e[n],t)}let i=[...n.keys()].sort((e,t)=>n.get(t)-n.get(e)||e-t),a=new Map,o=ee.length;for(let e of i){let t=Array(o).fill(0);for(let n of r.get(e)??[]){let e=a.get(n);e!==void 0&&t[e]++}let n=0;for(let e=1;e<o;e++)t[e]<t[n]&&(n=e);a.set(e,n)}return Uint32Array.from(e,e=>a.get(e))}function Te(e,t){let n=new Uint8Array(e.length),r=0,i=new Set;for(let a=0;a<e.length;a++){if(i.add(e[a]),n[a])continue;r++;let o=[a];for(n[a]=1;o.length;){let r=o.pop();for(let i of t[r])!n[i]&&e[i]===e[r]&&(n[i]=1,o.push(i))}}return{total:r,labelCount:i.size}}function Ee(e,t,n){let r=n.length,i=new Float32Array(t*2);if(r===0)return i;let a=Array.from({length:r},()=>new Float64Array(r));for(let i=0;i<t;i++)for(let o=0;o<r;o++)for(let s=0;s<r;s++)a[o][s]+=e[i*Z+n[o]]*e[i*Z+n[s]]/t;let o=[];for(let e=0;e<Math.min(2,r);e++){let t=Float64Array.from({length:r},(t,n)=>1+n*.37+e);for(let e=0;e<200;e++){let e=new Float64Array(r);for(let n=0;n<r;n++)for(let i=0;i<r;i++)e[n]+=a[n][i]*t[i];for(let t of o){let n=0;for(let i=0;i<r;i++)n+=e[i]*t[i];for(let i=0;i<r;i++)e[i]-=n*t[i]}let n=Math.hypot(...e)||1;t=e.map(e=>e/n)}o.push(t)}for(let a=0;a<t;a++)for(let t=0;t<o.length;t++){let s=0;for(let i=0;i<r;i++)s+=e[a*Z+n[i]]*o[t][i];i[a*2+t]=s}return i}async function De(n){let r=n.datasets.get(`chicago-tracts`),{device:i}=n,a=new s(i,`regionalization`),o=T(r),c=o.featureCount,l=e=>r.column(e),u=l(`population`),d=l(`perCapitaIncome`),f=e=>{let t=l(e);return Float32Array.from(t,(e,t)=>u[t]>0?100*e/u[t]:NaN)},p={poverty:l(`poverty150Pct`),income:Float32Array.from(d,e=>e>0?Math.log(e):NaN),uninsured:l(`uninsuredPct`),age65:l(`age65Pct`),noVehicle:l(`noVehiclePct`),diabetes:l(`diabetes`),black:f(`nhBlack`),hispanic:f(`hispanic`)},m=[];for(let e=0;e<c;e++){let t=u[e]>=300;for(let n of Object.values(p))t&&=Number.isFinite(n[e]);t&&m.push(e)}let h=O(E(o),m),g=h.length,_=new Uint32Array(c).fill(F);h.forEach((e,t)=>{_[e]=t});let v=L(o,h),V=A(v),te=r.manifest.geoid,H=new Float32Array(g*Z),U=new Float32Array(g*Z);be.forEach(({id:e},t)=>{let n=Float32Array.from(h,t=>p[e][t]),{mean:r,deviation:i}=k(n);for(let e=0;e<g;e++)H[e*Z+t]=n[e],U[e*Z+t]=(n[e]-r)/i});let W=R(v),re=new Uint32Array(c).fill(F);for(let e=0;e<c;e++)_[e]===4294967295&&(re[e]=0);let G=R(o,re),ie=a.createBuffer(`triangles`,W.corners),oe=a.createBuffer(`owners`,W.owners),se=a.createBuffer(`excluded-triangles`,G.corners),ce=a.createBuffer(`excluded-owners`,G.owners),le=N(o),ue=a.createBuffer(`outline`,le),de=j(v);a.createBuffer(`centroids`,V);let K=g*Se,q=xe-1,fe=a.createBuffer(`vertices`,v.vertices),pe=a.createBuffer(`ring-offsets`,v.ringOffsets),J=a.createBuffer(`feature-offsets`,I(v)),he=a.createBuffer(`polygon-offsets`,v.partRingOffsets),ge=a.createBuffer(`values`,g*Z*4),_e=a.createBuffer(`standardized`,g*Z*4),ve=a.createBuffer(`offsets`,(g+1)*4),De=a.createBuffer(`neighbors`,K*4),Oe=a.createBuffer(`weight-values`,K*4),ke=a.createBuffer(`overflow`,4),Ae=a.createBuffer(`candidate-overflow`,4),je=a.createBuffer(`tree-flags`,K*4),Me=a.createBuffer(`component-labels`,g*4),Ne=a.createBuffer(`edge-ids`,g*4),Pe=a.createBuffer(`edge-count`,4),Fe=a.createBuffer(`edge-overflow`,4),Ie=a.createBuffer(`edge-costs`,g*4),Le=a.createBuffer(`labels`,g*4),Re=a.createBuffer(`region-count`,4),ze=a.createBuffer(`cut-edges`,q*4),Be=a.createBuffer(`cut-gains`,q*4),Ve=a.createBuffer(`summary`,Y.length*4),He=a.createBuffer(`region-sizes`,g*4),Ue=a.createBuffer(`region-within-ssd`,g*4),We=a.createParameterBuffer(`skater-parameters`,`uint32`,2),Ge=a.createBuffer(`principal-plane`,g*2*4),Ke=a.createBuffer(`kmeans-labels`,g*4),qe=a.createBuffer(`kmeans-centers`,X*2*4),Je=a.createBuffer(`kmeans-sizes`,X*4),Ye=a.createBuffer(`kmeans-iteration-count`,4),Xe=a.createBuffer(`kmeans-converged`,4),Ze=a.createParameterBuffer(`kmeans-parameters`,`uint32`,2,b({seed:7})),Qe=a.createBuffer(`kmeans-summary`,Y.length*4),$e=a.createBuffer(`kmeans-region-sizes`,g*4),et=a.createBuffer(`kmeans-region-within`,g*4),tt=a.createBuffer(`region-colors`,g*4),nt=a.createBuffer(`cluster-colors`,g*4),rt=a.createBuffer(`slot-segments`,K*16),it=a.createBuffer(`slot-cost`,K*4),at=a.createBuffer(`cut-mask`,K*4),ot=e=>({offsets:C(e,`offsets`,ve,`uint32`,g+1),neighbors:C(e,`neighbors`,De,`uint32`,K),weights:C(e,`weight-values`,Oe,`float32`,K)}),st=new Map,ct=async(e,n)=>{let r=`${e}:${n}`,o=st.get(r);if(o)return o;let s=new t(i,{id:`regions-${e}-${n?`z`:`raw`}`}),c=ot(s),l=C(s,`tree-flags`,je,`uint32`,K),u=C(s,`component-labels`,Me,`uint32`,g),d=C(s,`values`,ge,`float32`,g*Z),f=C(s,`standardized`,_e,`float32`,g*Z),p=C(s,`labels`,Le,`uint32`,g);s.add(new y({id:`contiguity`,criterion:e,polygons:{kind:`polygons`,positions:C(s,`vertices`,fe,`float32x2`,v.vertices.length/2),featureOffsets:C(s,`feature-offsets`,J,`uint32`,v.featureCount+1),polygonOffsets:C(s,`polygon-offsets`,he,`uint32`,v.partRingOffsets.length),ringOffsets:C(s,`ring-offsets`,pe,`uint32`,v.ringOffsets.length)},output:{...c,status:{overflow:C(s,`overflow`,ke,`uint32`,1),candidateOverflow:C(s,`candidate-overflow`,Ae,`uint32`,1)}}})),s.add(new ne({id:`tree`,weights:c,values:d,columnCount:Z,standardize:n,output:{treeEdgeFlags:l,componentLabels:u,standardizedValues:n?f:void 0,edges:{ids:C(s,`edge-ids`,Ne,`uint32`,g),status:{count:C(s,`edge-count`,Pe,`uint32`,1),overflow:C(s,`edge-overflow`,Fe,`uint32`,1)}},edgeDissimilarities:C(s,`edge-costs`,Ie,`float32`,g)}}));let m=n?f:d;return s.add(new me({id:`skater`,weights:c,treeEdgeFlags:l,componentLabels:u,values:m,columnCount:Z,maximumRegionCount:xe,parameters:We.importToGraph(s),output:{labels:p,regionCount:C(s,`region-count`,Re,`uint32`,1),cutEdges:C(s,`cut-edges`,ze,`uint32`,q),cutGains:C(s,`cut-gains`,Be,`float32`,q)}})),s.add(new ye({id:`skater-evaluation`,values:m,columnCount:Z,labels:p,weights:c,output:{summary:C(s,`summary`,Ve,`float32`,Y.length),regionSizes:C(s,`region-sizes`,He,`uint32`,g),regionWithinSumOfSquares:C(s,`region-within-ssd`,Ue,`float32`,g)}})),o=a.track(await s.compileAsync()),st.set(r,o),o},lt=new Map,ut=async(e,n)=>{let r=`${e}:${n}`,o=lt.get(r);if(o)return o;let s=new t(i,{id:`kmeans-${e}-${n?`z`:`raw`}`}),c=C(s,`kmeans-labels`,Ke,`uint32`,g);return s.add(new x({id:`kmeans`,positions:C(s,`principal-plane`,Ge,`float32x2`,g),k:e,maximumIterations:Ce,initialization:`kmeans++`,parameters:Ze.importToGraph(s),output:{labels:c,centers:C(s,`kmeans-centers`,qe,`float32x2`,e),sizes:C(s,`kmeans-sizes`,Je,`uint32`,e),status:{iterationCount:C(s,`kmeans-iteration-count`,Ye,`uint32`,1),converged:C(s,`kmeans-converged`,Xe,`uint32`,1)}}})),s.add(new ye({id:`kmeans-evaluation`,values:C(s,`metric`,n?_e:ge,`float32`,g*Z),columnCount:Z,labels:c,labelCapacity:X,weights:ot(s),output:{summary:C(s,`kmeans-summary`,Qe,`float32`,Y.length),regionSizes:C(s,`kmeans-region-sizes`,$e,`uint32`,X),regionWithinSumOfSquares:C(s,`kmeans-region-within`,et,`float32`,X)}})),o=a.track(await s.compileAsync()),lt.set(r,o),o},Q=n.options.criterion,$=n.options.standardize,dt=n.options.kmeansClusters,ft=!0,pt=!0,mt=Array.from({length:g},()=>[]),ht=!1,gt=new Uint32Array(g),_t=new Uint32Array(g),vt=[0,1],yt=0,bt=()=>{let e=be.map(({id:e})=>n.options[e]);e.some(Boolean)||(e[0]=!0);let t=new Float32Array(g*Z);for(let n=0;n<g;n++)for(let r=0;r<Z;r++)t[n*Z+r]=e[r]?H[n*Z+r]:0;ge.write(t),n.options.standardize||_e.write(new Float32Array(g*Z));let r=e.flatMap((e,t)=>e?[t]:[]);Ge.write(Ee(U,g,r)),We.write(ae({regionCount:n.options.regions,minimumRegionSize:n.options.minimumSize}))};n.setReadout(`rows`,`${P(g)} of ${P(c)} tracts`),n.setStatus(`${P(g)} Chicago tracts`),await ct(Q,$),await ut(Number(dt),$);let xt=a.track(new w);bt();let St=z(a,`regions-csr`,[{name:`offsets`,buffer:ve,bytes:(g+1)*4},{name:`neighbors`,buffer:De,bytes:K*4}],e=>{let t=e(`offsets`).u32,r=e(`neighbors`).u32,i=new Float32Array(K*4),a=Array.from({length:g},()=>[]);for(let e=0;e<g;e++)for(let n=t[e];n<t[e+1];n++){let t=r[n];a[e].push(t),i[n*4]=V[e*2],i[n*4+1]=V[e*2+1],i[n*4+2]=V[t*2],i[n*4+3]=V[t*2+1]}rt.write(i),mt=a,ht=!0,Ct.markStale(),n.requestLayers()}),Ct=z(a,`regions-results`,[{name:`summary`,buffer:Ve,bytes:Y.length*4},{name:`labels`,buffer:Le,bytes:g*4},{name:`regionCount`,buffer:Re,bytes:4},{name:`cutEdges`,buffer:ze,bytes:q*4},{name:`cutGains`,buffer:Be,bytes:q*4},{name:`edgeCount`,buffer:Pe,bytes:4},{name:`edgeOverflow`,buffer:Fe,bytes:4},{name:`edgeIds`,buffer:Ne,bytes:g*4},{name:`edgeCosts`,buffer:Ie,bytes:g*4},{name:`componentLabels`,buffer:Me,bytes:g*4},{name:`kSummary`,buffer:Qe,bytes:Y.length*4},{name:`kLabels`,buffer:Ke,bytes:g*4},{name:`kIterationCount`,buffer:Ye,bytes:4},{name:`kConverged`,buffer:Xe,bytes:4},{name:`overflow`,buffer:ke,bytes:4},{name:`candidateOverflow`,buffer:Ae,bytes:4}],e=>{let t=Y,r=e(`summary`).f32,i=e(`kSummary`).f32,a=e(`labels`).u32,o=e(`regionCount`).u32[0],s=e(`cutEdges`).u32,c=e(`cutGains`).f32;yt=e(`edgeCount`).u32[0];let l=e(`edgeIds`).u32,u=e(`edgeCosts`).f32,d=new Set(e(`componentLabels`).u32).size;gt=a,_t=e(`kLabels`).u32;let f=e(`kIterationCount`).u32[0],p=e(`kConverged`).u32[0],m=e=>e[t.totalSumOfSquares]>0?100*e[t.betweenSumOfSquares]/e[t.totalSumOfSquares]:NaN,h=n.options.regions;n.setReadout(`regions`,`${o} (target ${h}${o<h?`, size floor or islands bind`:``})`),n.setReadout(`explained`,`SKATER ${D(m(r),1)}% vs k-means ${D(m(i),1)}% of variance (${D(m(i)-m(r),1)} percentage points, k-means minus SKATER)`),n.setReadout(`pseudoF`,`SKATER ${D(r[t.pseudoF],1)} vs k-means ${D(i[t.pseudoF],1)}`),n.setReadout(`within`,`SKATER ${D(r[t.withinSumOfSquares],0)} vs k-means ${D(i[t.withinSumOfSquares],0)} (of ${D(r[t.totalSumOfSquares],0)})`),n.setReadout(`sizes`,`SKATER ${P(r[t.minimumSize])} to ${P(r[t.maximumSize])} tracts; k-means ${P(i[t.minimumSize])} to ${P(i[t.maximumSize])}`),n.setReadout(`boundary`,`SKATER ${(100*r[t.crossLinkFraction]).toFixed(1)}% vs k-means ${(100*i[t.crossLinkFraction]).toFixed(1)}% of neighbour links`);let _=Te(_t,mt),v=Te(a,mt);n.setReadout(`pieces`,ht?`SKATER ${v.total} piece${v.total===1?``:`s`} for ${v.labelCount} regions; k-means ${_.total} pieces for ${_.labelCount} clusters`:null),n.setReadout(`tree`,`${P(yt)} edges in ${d} tree${d===1?``:`s`}${e(`edgeOverflow`).u32[0]?` (overflow)`:``}`),n.setReadout(`kmeans`,`${f} iterations, ${p?`converged`:`not converged`}`);let y=0,b=new Uint32Array(K);for(let e=0;e<q;e++)s[e]!==4294967295&&(b[s[e]]=1,y=e+1);at.write(b),n.setReadout(`gains`,y>0?Array.from(c.slice(0,Math.min(y,8))).map(e=>D(e,0)).join(` / `)+(y>8?` / …`:``):`no cuts`);let x=new Float32Array(K).fill(NaN),S=[];for(let e=0;e<Math.min(yt,g);e++)x[l[e]]=u[e],S.push(u[e]);it.write(x),vt=[0,Math.max(1e-6,M(S,.95))],n.setLegendExtent(`tree`,vt),ht&&(tt.write(we(a,mt)),nt.write(we(_t,mt))),n.setReadout(`capacity`,e(`overflow`).u32[0]?`overflow: raise slots per tract${e(`candidateOverflow`).u32[0]?` (pair list full)`:``}`:`ok`),n.requestLayers()}),wt=()=>n.theme()===`dark`;return{getCompiledGraphs:()=>[st.get(`${Q}:${$}`),lt.get(`${dt}:${$}`)],setOption(e){if(e===`map`||e===`showTree`||e===`showCuts`||e===`treeByCost`){n.requestLayers();return}if(e===`criterion`||e===`standardize`||e===`kmeansClusters`){let{criterion:e,standardize:t,kmeansClusters:r}=n.options;xt.request({build:async()=>{await ct(e,t),await ut(Number(r),t)},commit:()=>{e!==Q&&(pt=!0),Q=e,$=t,dt=r,bt(),ft=!0,n.requestLayers()}});return}bt(),ft=!0},encode(e,t){t.frameIndex<2&&(ft=!0,pt=!0),ft&&=(st.get(`${Q}:${$}`).encode(e,{parameters:void 0}),lt.get(`${dt}:${$}`).encode(e,{parameters:void 0}),Ct.request(e),pt&&=(St.request(e),!1),!1),St.flush(e),Ct.flush(e)},getLayers(){let t=wt(),{map:r,showTree:i,showCuts:a,treeByCost:o}=n.options,s=[new B({id:`regions-excluded`,triangles:se,owners:ce,triangleCount:G.triangleCount,colormap:`uniform`,color:t?[90,94,104,150]:[170,172,178,150]}),new B({id:`regions-fill-${r}`,triangles:ie,owners:oe,triangleCount:W.triangleCount,values:r===`skater`?tt:nt,valueFormat:`uint32`,colormap:`category`,palette:ee,opacity:t?.78:.82}),new S({id:`regions-outline`,coordinateSystem:e.LNGLAT,segments:ue,instanceCount:le.length/4,widthPixels:.7,color:t?[235,238,248,60]:[30,34,46,70]})];return ht&&i&&s.push(new S({id:`regions-tree`,coordinateSystem:e.LNGLAT,segments:rt,instanceCount:K,values:o?it:je,valueFormat:o?`float32`:`uint32`,colormap:o?`inferno`:`mask`,valueRange:vt,sqrtScale:o,color:[255,255,255,240],noDataColor:[0,0,0,0],widthPixels:2})),ht&&a&&s.push(new S({id:`regions-cuts`,coordinateSystem:e.LNGLAT,segments:rt,instanceCount:K,values:at,valueFormat:`uint32`,colormap:`mask`,color:[230,40,40,255],noDataColor:[0,0,0,0],widthPixels:5})),s},onThemeChange(){n.requestLayers()},getTooltip(e){if(!e.coordinate)return null;let t=de(e.coordinate[0],e.coordinate[1]);if(t<0)return null;let n=h[t],r=be.map(({id:e,label:t})=>`${t}: ${p[e][n].toFixed(e===`income`?2:1)}`);return[`Tract ${te[n]}`,`SKATER region ${gt[t]}, k-means cluster ${_t[t]}`,...r.slice(0,6)].join(`
`)},destroy(){St.stop(),Ct.stop(),a.destroy()}}}export{De as createRegionalization};