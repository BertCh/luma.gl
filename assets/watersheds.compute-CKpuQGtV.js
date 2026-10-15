import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,O as t,T as n,_ as r,h as i,t as a,w as o,x as s,y as c}from"./resources-CwvhEOBe.js";import{t as l}from"./gpu-scan-BuGhTLDQ.js";import{t as u}from"./gpu-histogram-C4GJppav.js";import{n as d,o as f,s as p}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as m,o as h}from"./wgsl-common-BNRVzgd8.js";import{a as g,c as _,i as v,n as y,o as b,r as x,s as S,u as C}from"./raster-band-utils-CA-sh8xt.js";import{c as w}from"./raster-grid-D2bmCwkc.js";import{n as T,r as E,t as D}from"./raster-grid-utils-D1bVIm4m.js";import{c as O,d as ee,f as k,i as te,l as ne,n as A,o as j,p as M,r as N,s as P,u as re}from"./raster-relaxation-BJ40xgro.js";import{a as F,c as I,s as L}from"./index-Bjk18V2I.js";import{t as ie}from"./graph-buffers-B4_80RlX.js";import{t as ae}from"./graph-swap-41Ru7LVu.js";import{t as oe}from"./summary-reader-CsgWxENf.js";import{t as R}from"./mode-kernels-Dxm2GWvI.js";import{n as se,t as ce}from"./gpu-terrain-derivatives-BZk8xB3m.js";import{f as z,g as le,i as ue,m as de,n as B,o as fe,p as pe,r as me,s as he,t as ge}from"./b15-common-Do8LzRH9.js";function V(e){let t=Math.max(2*e,2),n=32-Math.clz32(t-1);return Math.ceil((n+1)/4)*4}function _e(e,t){return`
const TOUR_WIDTH: u32 = ${e}u;
const TOUR_HEIGHT: u32 = ${t}u;
const TOUR_NONE: u32 = 0xffffffffu;

fn getTourNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % TOUR_WIDTH) + columnOffsets[direction];
  let row = i32(cell / TOUR_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(TOUR_WIDTH) || row >= i32(TOUR_HEIGHT)) {
    return TOUR_NONE;
  }
  return u32(row) * TOUR_WIDTH + u32(column);
}

fn getTourFirstChild(cell: u32) -> u32 {
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(cell, direction);
    if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell) { return neighbor; }
  }
  return TOUR_NONE;
}

// The child of 'parent' that follows 'cell' in the direction order of 'parent'.
fn getTourNextSibling(cell: u32, parent: u32) -> u32 {
  var passed = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(parent, direction);
    if (neighbor == TOUR_NONE) { continue; }
    if (neighbor == cell) { passed = true; continue; }
    if (passed && getTourParent(neighbor) == parent) { return neighbor; }
  }
  return TOUR_NONE;
}`}var ve=`
fn isTourEdgeResolved(edge: u32) -> bool {
  let terminal = tourSucc[tourSuccOffset + edge];
  // A terminal is the exit edge (odd) of a cell without a parent. A cycle edge that jumped onto
  // itself has a parent, so it is rejected.
  return (terminal & 1u) == 1u && getTourParent(terminal >> 1u) == TOUR_NONE;
}`;function H(e,t){let n=t.width*t.height,r=2*n,i=n=>D(e,t.operation,t.contributorId,`${t.id}-tour-${n}`,`uint32`,r),a=i(`successor`),o=i(`distance`),s=i(`successor-scratch`),c=i(`distance-scratch`),l=P(e,`${t.id}-tour`,t.operation,r,t.contributorId);return{nodes:[p(e,{id:`${t.id}-tour-link`,operation:t.operation,variant:`tour-link`,bindings:[...t.parentBindings,{name:`tourSucc`,view:a,type:`u32`,access:`read_write`},{name:`tourDist`,view:o,type:`u32`,access:`read_write`}],invocationCount:n,declarations:`${_e(t.width,t.height)}\n${t.declarations}`,body:`let parent = getTourParent(index);
  let firstChild = getTourFirstChild(index);
  let exitEdge = 2u * index + 1u;
  var enterSuccessor = exitEdge;
  if (firstChild != TOUR_NONE) { enterSuccessor = 2u * firstChild; }
  // A root exits to itself: its exit edge is the terminal, at distance 0.
  var exitSuccessor = exitEdge;
  var exitDistance = 0u;
  if (parent != TOUR_NONE) {
    let sibling = getTourNextSibling(index, parent);
    exitSuccessor = select(2u * parent + 1u, 2u * sibling, sibling != TOUR_NONE);
    exitDistance = 1u;
  }
  tourSucc[tourSuccOffset + 2u * index] = enterSuccessor;
  tourSucc[tourSuccOffset + exitEdge] = exitSuccessor;
  tourDist[tourDistOffset + 2u * index] = 1u;
  tourDist[tourDistOffset + exitEdge] = exitDistance;`}),j(e,{id:`${t.id}-tour-reset`,operation:t.operation,state:l}),...A(e,{id:`${t.id}-tour-round`,gateId:`${t.id}-tour-gate`,operation:t.operation,state:l,maxIterations:V(n),groupSize:4,createRound:({nodeId:n,iteration:i,slotBinding:u,markChangedWGSL:d,condition:f,extraResources:m})=>{let[h,g,_,v]=i%2==0?[a,o,s,c]:[s,c,a,o];return p(e,{id:n,operation:t.operation,variant:`tour-round`,bindings:[{name:`succIn`,view:h,type:`u32`,access:`read`},{name:`distIn`,view:g,type:`u32`,access:`read`},{name:`succOut`,view:_,type:`u32`,access:`read_write`},{name:`distOut`,view:v,type:`u32`,access:`read_write`},{name:`status`,view:l.status,type:`atomic<u32>`,access:`read_write`},u],invocationCount:r,body:`let next = succIn[succInOffset + index];
  let jumped = succIn[succInOffset + next];
  var rankedDistance = distIn[distInOffset + index];
  if (next != jumped) {
    rankedDistance = rankedDistance + distIn[distInOffset + next];
    ${d}
  }
  succOut[succOutOffset + index] = jumped;
  distOut[distOutOffset + index] = rankedDistance;`,condition:f,extraResources:m})}})],successor:a,distance:o,state:l,scratch:[s,c]}}var U=`GPUTerrainFlow`,ye=1024;function be(e,t){let n={id:`${t.id}-fill`,operation:U,width:t.width,height:t.height,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations},{relaxation:r,resetNodes:i}=O(e,n,!1),a=p(e,{id:`${t.id}-fill-init`,operation:U,variant:`fill-init`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`tileStamps`,view:r.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.width*t.height,declarations:`${T(t)}\n${k(t.width)}`,body:`_ = settings[settingsOffset];
  let elevationValue = elevation[elevationOffset + index];
  if (!isFiniteValue(elevationValue)) {
    filled[filledOffset + index] = bitcast<u32>(getQuietNaN());
    return;
  }
  var boundary = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`}),o=ne(e,{...n,relaxation:r,sweepAfterIteration:96,values:t.filled,auxiliary:t.elevation,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  var epsilon = settings[settingsOffset + 4u];
  if (!(epsilon >= 0.0) || !isFiniteValue(epsilon)) { epsilon = 0.0; }
  return max(centerAuxiliary, neighborValue + epsilon);
}`}),s=[...i,a,...o];return(t.converged||t.iterationCount)&&s.push(N(e,{id:`${t.id}-fill-finalize`,operation:U,state:r.state,converged:t.converged,iterationCount:t.iterationCount})),{nodes:s,status:r.state.status}}function xe(e,t){let n=[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];for(let[e,r]of[[`directions`,t.directions],[`classes`,t.classes],[`receivers`,t.receivers]])r&&n.push({name:e,view:r,type:`u32`,access:`read_write`});let r=(e,n,r)=>[t.directions?`directions[directionsOffset + index] = ${e};`:``,t.classes?`classes[classesOffset + index] = ${n};`:``,t.receivers?`receivers[receiversOffset + index] = ${r};`:``].join(`
  `),{cellClass:i}=t;return p(e,{id:t.id,operation:U,variant:`flow-direction`,bindings:n,invocationCount:t.width*t.height,declarations:T(t),body:`let row = index / GRID_WIDTH;
  if (!isFiniteValue(elevation[elevationOffset + index])) {
    ${r(`GRID_NONE`,`${i.invalid}u`,`GRID_NONE`)}
    return;
  }
  let centerSurface = surface[surfaceOffset + index];
  var bestDrop = 0.0;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  var boundary = false;
  var hasEqualNeighbor = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      continue;
    }
    let neighborSurface = surface[surfaceOffset + neighbor];
    if (neighborSurface == centerSurface) { hasEqualNeighbor = true; }
    let drop = (centerSurface - neighborSurface) / getD8Distance(direction, row);
    // Strictly greater: ties keep the lowest direction index.
    if (drop > bestDrop) {
      bestDrop = drop;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    ${r(`getD8Code(bestDirection)`,`${i.draining}u`,`bestNeighbor`)}
    return;
  }
  var terminalClass = ${i.pit}u;
  if (boundary) { terminalClass = ${i.outlet}u; }
  else if (hasEqualNeighbor) { terminalClass = ${i.flat}u; }
  ${r(`0u`,`terminalClass`,`GRID_NONE`)}`})}function Se(e,t){return!t.runoff&&!t.area&&8*t.width*t.height<=e.device.limits.maxStorageBufferBindingSize}function Ce(e,t){let n=t.width*t.height,r=H(e,{id:`${t.id}-accumulate`,operation:U,contributorId:t.id,width:t.width,height:t.height,declarations:`fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }`,parentBindings:[{name:`receivers`,view:t.receivers,type:`u32`,access:`read`}]}),i=p(e,{id:`${t.id}-accumulate-count`,operation:U,variant:`tour-accumulation`,bindings:[{name:`acc`,view:t.accumulation,type:`u32`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`tourSucc`,view:r.successor,type:`u32`,access:`read`},{name:`tourDist`,view:r.distance,type:`u32`,access:`read`},{name:`status`,view:r.state.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${_e(t.width,t.height)}
fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }
${ve}`,body:`let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (!isValid) { acc[accOffset + index] = 0xffffffffu; return; }
  if (!isTourEdgeResolved(2u * index)) {
    // The list never reached a root: a receiver cycle. Not converged.
    acc[accOffset + index] = 0xffffffffu;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  // The edges from E(v) to X(v) inclusive are the tour of v's subtree: 2 * size(v) of them.
  let size = (tourDist[tourDistOffset + 2u * index] - tourDist[tourDistOffset + 2u * index + 1u] + 1u) / 2u;
  acc[accOffset + index] = bitcast<u32>(f32(size));`}),a=[...r.nodes,i];return(t.converged||t.iterationCount)&&a.push(N(e,{id:`${t.id}-accumulate-finalize`,operation:U,state:r.state,converged:t.converged,iterationCount:t.iterationCount})),a}function we(e,t){if(Se(e,t))return Ce(e,t);let n=t.width*t.height,r=P(e,`${t.id}-accumulate`,U,n),i=[d(e,{id:`${t.id}-accumulate-init`,operation:U,view:t.accumulation,type:`u32`,value:`0xffffffffu`}),j(e,{id:`${t.id}-accumulate-reset`,operation:U,state:r})],a=[{name:`acc`,view:t.accumulation,type:`atomic<u32>`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];t.runoff&&a.push({name:`runoff`,view:t.runoff,type:`f32`,access:`read`}),a.push({name:`status`,view:r.status,type:`atomic<u32>`,access:`read_write`});let o=`${T(t)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${ye}u;

fn getWeight(cell: u32) -> f32 {
  var weight: f32 = 1.0;
  ${t.runoff?`let runoffValue = runoff[runoffOffset + cell];
  weight = select(0.0, runoffValue, isFiniteValue(runoffValue) && runoffValue >= 0.0);`:``}
  ${t.area?`let ground = getGroundCellSizeAtRowCoordinate(f32(cell / GRID_WIDTH) + 0.5);
  weight = weight * (ground.x * ground.y);`:``}
  return weight;
}

// Returns true when this thread wrote the final value of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&acc[accOffset + cell]) != SENTINEL) { return false; }
  var sum = getWeight(cell);
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(cell, direction);
    if (neighbor != GRID_NONE && receivers[receiversOffset + neighbor] == cell) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`,s=e=>`_ = settings[settingsOffset];
  if (!isFiniteValue(elevation[elevationOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  if (tryFinalize(index)) {
    var walk = receivers[receiversOffset + index];
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == GRID_NONE || !tryFinalize(walk)) { break; }
      walk = receivers[receiversOffset + walk];
    }
  }
  if (atomicLoad(&acc[accOffset + index]) == SENTINEL) {
    ${e}
  }`;return i.push(...A(e,{id:`${t.id}-accumulate-round`,gateId:`${t.id}-accumulate-gate`,operation:U,state:r,maxIterations:t.maxIterations,createRound:({nodeId:t,slotBinding:r,markChangedWGSL:i,condition:c,extraResources:l})=>p(e,{id:t,operation:U,variant:`accumulate-round`,bindings:[...a,r],invocationCount:n,declarations:o,body:s(i),condition:c,extraResources:l})})),(t.converged||t.iterationCount)&&i.push(N(e,{id:`${t.id}-accumulate-finalize`,operation:U,state:r,converged:t.converged,iterationCount:t.iterationCount})),i}function Te(e,t){return p(e,{id:t.id,operation:U,variant:`streams`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`accumulation`,view:t.accumulation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`streams`,view:t.streams,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,body:`let value = accumulation[accumulationOffset + index];
  let isResolved = (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
  let isStream = isResolved && value >= settings[settingsOffset + 5u];
  var stream = select(0u, 1u, isStream);
  // A valid cell without a value was cut off by the accumulation round limit. It is usually the
  // downstream trunk, so 0 ("not a stream") would erase the main river.
  let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (isValid && !isResolved) { stream = ${t.unresolvedValue}u; }
  streams[streamsOffset + index] = stream;`})}function Ee(e){return 32-Math.clz32(e)}function De(e){let t=Ee(Math.max(e,2)-1);return Math.ceil((t+1)/4)*4}function W(e,t,n){let{condition:r,extraResources:i}=e.getGate(t);return{condition:{...r,id:`${n}-gate`},extraResources:i}}function G(e,t){if(t.shapes.length===1&&(t.statusLength??4)===4)return P(e,t.id,t.operation,t.shapes[0][0],t.contributorId);let n=f(e,{id:`${t.id}-dispatch`,operation:t.operation,shapes:t.shapes});return{status:D(e,t.operation,t.contributorId,`${t.id}-status`,`uint32`,t.statusLength??4),dispatch:n.view,gate:n,layout:s(t.operation,Math.max(t.shapes[0][0],1),t.shapes[0][1],e.device.limits.maxComputeWorkgroupsPerDimension),invocationCount:t.shapes[0][0]}}function K(e,t){let n=[];for(let e=0;e<(t.statusLength??4);e++)n.push(e===1?`status[statusOffset + 1u] = select(0u, 1u, enabled);`:`status[statusOffset + ${e}u] = 0u;`);return p(e,{id:t.id,operation:t.operation,variant:`spill-reset`,bindings:[...t.enabledBindings,{name:`status`,view:t.state.status,type:`u32`,access:`read_write`},t.state.gate.binding],invocationCount:1,body:`let enabled = ${t.enabledExpression};
  ${n.join(`
  `)}
  if (enabled) {
    ${t.state.gate.armWGSL}
  } else {
    ${t.state.gate.disarmWGSL}
  }`})}function q(e,t){return`
const SPILL_WIDTH: u32 = ${e}u;
const SPILL_HEIGHT: u32 = ${t}u;
const SPILL_NONE: u32 = 0xffffffffu;
const SPILL_BOUNDARY_FLAG: u32 = 0x80000000u;
const SPILL_ID_MASK: u32 = 0x7fffffffu;
const SPILL_INFINITE_KEY: u32 = 0xffffffffu;

fn getSpillNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % SPILL_WIDTH) + columnOffsets[direction];
  let row = i32(cell / SPILL_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(SPILL_WIDTH) || row >= i32(SPILL_HEIGHT)) {
    return SPILL_NONE;
  }
  return u32(row) * SPILL_WIDTH + u32(column);
}

fn isSpillFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn getSpillKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, bits >= 0x80000000u);
}

fn getSpillValueBits(key: u32) -> u32 {
  return select(~key, key & 0x7fffffffu, key >= 0x80000000u);
}`}function Oe(e,t){let{cellCount:n}=t,r=n+1,i=2*n,a=(n,r)=>D(e,t.operation,t.contributorId,`${t.id}-spill-tree-${n}`,`uint32`,r),o=a(`list-next`,i),s=a(`successor`,i),c=a(`distance`,i),l=a(`successor-scratch`,i),u=a(`distance-scratch`,i),d=a(`parent-scratch`,r),f=a(`parent-key-scratch`,r),m=(n,r,i,a,o)=>p(e,{id:`${t.id}-spill-tree-${n}`,operation:t.operation,variant:`spill-tree-${n}`,bindings:i,invocationCount:r,workgroupSize:256,declarations:`const TREE_NONE: u32 = 0xffffffffu;`,body:a,...o}),h=[{name:`mstFirst`,view:t.mstFirst,type:`u32`,access:`read`},{name:`mstSecond`,view:t.mstSecond,type:`u32`,access:`read`}],g={name:`status`,view:t.mstStatus,type:`u32`,access:`read`},_=m(`lists`,i,[...h,{name:`head`,view:t.head,type:`atomic<u32>`,access:`read_write`},{name:`listNext`,view:o,type:`u32`,access:`read_write`},g],`if (index >= 2u * status[statusOffset + 4u]) { return; }
  let edge = index >> 1u;
  let origin = select(mstFirst[mstFirstOffset + edge], mstSecond[mstSecondOffset + edge], (index & 1u) == 1u);
  // Push this half-edge on the list of its origin; the previous head becomes its successor.
  listNext[listNextOffset + index] = atomicExchange(&head[headOffset + origin], index);`,t.getHalfEdgeGate(`${t.id}-spill-tree-lists`)),v=m(`link`,i,[...h,{name:`head`,view:t.head,type:`atomic<u32>`,access:`read_write`},{name:`listNext`,view:o,type:`u32`,access:`read`},{name:`tourSucc`,view:s,type:`u32`,access:`read_write`},{name:`tourDist`,view:c,type:`u32`,access:`read_write`},g],`if (index >= 2u * status[statusOffset + 4u]) {
    // Unused slots are their own terminals, so the ranking rounds never read out of range.
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
    return;
  }
  let edge = index >> 1u;
  // The half-edge arrives at the other endpoint than its origin.
  let arrival = select(mstSecond[mstSecondOffset + edge], mstFirst[mstFirstOffset + edge], (index & 1u) == 1u);
  var following = listNext[listNextOffset + (index ^ 1u)];
  if (following == TREE_NONE) { following = atomicLoad(&head[headOffset + arrival]); }
  let rootStart = atomicLoad(&head[headOffset + ${n}u]);
  // The tour is cut where it would return to the first half-edge of the ocean.
  if (following == rootStart) {
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
  } else {
    tourSucc[tourSuccOffset + index] = following;
    tourDist[tourDistOffset + index] = 1u;
  }`,t.getHalfEdgeGate(`${t.id}-spill-tree-link`)),y=G(e,{id:`${t.id}-spill-tree-rank`,operation:t.operation,contributorId:t.contributorId,shapes:[[i,256]]}),b=K(e,{id:`${t.id}-spill-tree-rank-reset`,operation:t.operation,state:y,enabledBindings:t.enabledBindings,enabledExpression:t.enabledExpression}),x=A(e,{id:`${t.id}-spill-tree-rank-round`,gateId:`${t.id}-spill-tree-rank-gate`,operation:t.operation,state:y,maxIterations:De(i),createRound:({nodeId:n,iteration:r,slotBinding:a,markChangedWGSL:o,condition:d,extraResources:f})=>{let[m,h,g,_]=r%2==0?[s,c,l,u]:[l,u,s,c];return p(e,{id:n,operation:t.operation,variant:`spill-tree-rank`,bindings:[{name:`succIn`,view:m,type:`u32`,access:`read`},{name:`distIn`,view:h,type:`u32`,access:`read`},{name:`succOut`,view:g,type:`u32`,access:`read_write`},{name:`distOut`,view:_,type:`u32`,access:`read_write`},{name:`status`,view:y.status,type:`atomic<u32>`,access:`read_write`},a],invocationCount:i,workgroupSize:256,body:`let next = succIn[succInOffset + index];
  let jumped = succIn[succInOffset + next];
  var rankedDistance = distIn[distInOffset + index];
  if (next != jumped) {
    rankedDistance = rankedDistance + distIn[distInOffset + next];
    ${o}
  }
  succOut[succOutOffset + index] = jumped;
  distOut[distOutOffset + index] = rankedDistance;`,condition:d,extraResources:f})}}),S=m(`root`,n,[...h,{name:`mstKey`,view:t.mstKey,type:`u32`,access:`read`},{name:`tourDist`,view:c,type:`u32`,access:`read`},{name:`parent`,view:t.parent,type:`u32`,access:`read_write`},{name:`parentKey`,view:t.parentKey,type:`u32`,access:`read_write`},g],`if (index >= status[statusOffset + 4u]) { return; }
  // The half-edge that comes first in the tour leads away from the root (larger distance).
  let forward = tourDist[tourDistOffset + 2u * index] > tourDist[tourDistOffset + 2u * index + 1u];
  let child = select(mstFirst[mstFirstOffset + index], mstSecond[mstSecondOffset + index], forward);
  let owner = select(mstSecond[mstSecondOffset + index], mstFirst[mstFirstOffset + index], forward);
  parent[parentOffset + child] = owner;
  parentKey[parentKeyOffset + child] = mstKey[mstKeyOffset + index];`,t.getEdgeGate(`${t.id}-spill-tree-root`)),C=G(e,{id:`${t.id}-spill-tree-path`,operation:t.operation,contributorId:t.contributorId,shapes:[[r,256]]}),w=K(e,{id:`${t.id}-spill-tree-path-reset`,operation:t.operation,state:C,enabledBindings:t.enabledBindings,enabledExpression:t.enabledExpression}),T=A(e,{id:`${t.id}-spill-tree-path-round`,gateId:`${t.id}-spill-tree-path-gate`,operation:t.operation,state:C,maxIterations:De(r),createRound:({nodeId:n,iteration:i,slotBinding:a,markChangedWGSL:o,condition:s,extraResources:c})=>{let[l,u,m,h]=i%2==0?[t.parent,t.parentKey,d,f]:[d,f,t.parent,t.parentKey];return p(e,{id:n,operation:t.operation,variant:`spill-tree-path`,bindings:[{name:`parentIn`,view:l,type:`u32`,access:`read`},{name:`keyIn`,view:u,type:`u32`,access:`read`},{name:`parentOut`,view:m,type:`u32`,access:`read_write`},{name:`keyOut`,view:h,type:`u32`,access:`read_write`},{name:`status`,view:C.status,type:`atomic<u32>`,access:`read_write`},a],invocationCount:r,workgroupSize:256,body:`let ancestor = parentIn[parentInOffset + index];
  let jumped = parentIn[parentInOffset + ancestor];
  // The path from here to 'jumped' is the path to 'ancestor' followed by the path from 'ancestor'.
  keyOut[keyOutOffset + index] = max(keyIn[keyInOffset + index], keyIn[keyInOffset + ancestor]);
  parentOut[parentOutOffset + index] = jumped;
  if (jumped != ancestor) {
    ${o}
  }`,condition:s,extraResources:c})}});return{nodes:[_,v,b,...x,S,w,...T]}}var J=`GPUTerrainFlow`,ke=32,Ae=12*ke*ke+8,je=0,Me=1,Ne=2,Y=3,Pe=4,Fe=0,Ie=1;function Le(e){return e.device.limits.maxComputeWorkgroupStorageSize>=Ae?ke:16}function Re(e){return Ee(Math.max(e,1))+1}function ze(e,t){return 8*(t.width*t.height+1)<=e.device.limits.maxStorageBufferBindingSize}function Be(e,t){if(!ze(e,t))return be(e,t);let{id:n,width:r,height:i}=t,a=r*i,o=a+1,s=5*a,c=Le(e),l=Math.ceil(r/c),u=l*Math.ceil(i/c),d=u*256,m=(t,r,i)=>D(e,J,n,`${n}-spill-${t}`,r,i),h=m(`mode`,`uint32`,1),g=m(`local-fill`,`float32`,a),_=m(`labels`,`uint32`,a),v=m(`comp`,`uint32`,o),y=m(`best-key`,`uint32`,o),b=m(`best-edge`,`uint32`,o),x=m(`hook-pointer`,`uint32`,o),S=m(`hook-pointer-scratch`,`uint32`,o),C=m(`mst-first`,`uint32`,a),w=m(`mst-second`,`uint32`,a),E=m(`mst-key`,`uint32`,a),ee=m(`head`,`uint32`,o),j=m(`parent`,`uint32`,o),M=m(`parent-key`,`uint32`,o),N=m(`status`,`uint32`,4),P={name:`mode`,view:h,type:`u32`,access:`read`},re=`mode[modeOffset] == 0u`,F=f(e,{id:`${n}-spill-mode-dispatch`,operation:J,shapes:[[d,256],[o,256],[2*a,256],[a,256],[a,256]]}),I=e=>t=>W(F,e,t),L=(t,r,i,a,o,s,c=!0)=>p(e,{id:`${n}-spill-${t}`,operation:J,variant:`spill-${t}`,bindings:i,invocationCount:r,workgroupSize:256,guardIndex:c,declarations:a,body:o,...s}),ie=e=>e.map(e=>`dispatch[dispatchOffset + ${3*e}u] = 0u;`).join(`
    `),ae=p(e,{id:`${n}-spill-select`,operation:J,variant:`spill-select`,bindings:[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`mode`,view:h,type:`u32`,access:`read_write`},F.binding],invocationCount:1,body:`let epsilon = settings[settingsOffset + 4u];
  // Invalid, negative and non-finite epsilon mean 0, as in the relaxation.
  let useRelaxation = (bitcast<u32>(epsilon) & 0x7fffffffu) < 0x7f800000u && epsilon > 0.0;
  mode[modeOffset] = select(0u, 1u, useRelaxation);
  ${F.armWGSL}
  if (useRelaxation) {
    ${ie([je,Me,Ne,Y])}
  } else {
    ${ie([Pe])}
  }`}),oe={id:`${n}-fill`,operation:J,width:r,height:i,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations},{relaxation:R,resetNodes:se}=O(e,oe,!1),ce=se.find(e=>e.id===`${n}-fill-tile-reset`);if(!ce)throw Error(`${n} relaxation tile reset node is missing`);let z=K(e,{id:`${n}-fill-relax-reset`,operation:J,state:R.state,enabledBindings:[P],enabledExpression:`mode[modeOffset] == 1u`}),le=p(e,{id:`${n}-fill-init`,operation:J,variant:`fill-init`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`tileStamps`,view:R.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:a,workgroupSize:256,declarations:`${T(t)}\n${k(r)}`,body:`_ = settings[settingsOffset];
  let elevationValue = elevation[elevationOffset + index];
  if (!isFiniteValue(elevationValue)) {
    filled[filledOffset + index] = bitcast<u32>(getQuietNaN());
    return;
  }
  var boundary = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) {
      boundary = true;
      break;
    }
    if (!isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`,...W(F,Pe,`${n}-fill-init`)}),ue=ne(e,{...oe,relaxation:R,sweepAfterIteration:96,values:t.filled,auxiliary:t.elevation,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  var epsilon = settings[settingsOffset + 4u];
  if (!(epsilon >= 0.0) || !isFiniteValue(epsilon)) { epsilon = 0.0; }
  return max(centerAuxiliary, neighborValue + epsilon);
}`}),de=q(r,i),B=c*c,fe=L(`tile-fill`,d,[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`localFill`,view:g,type:`u32`,access:`read_write`},{name:`labels`,view:_,type:`u32`,access:`read_write`}],`${de}
const TILE: u32 = ${c}u;
const TILE_CELLS: u32 = ${B}u;
const CELLS_PER_THREAD: u32 = ${B/256}u;
const TILES_X: u32 = ${l}u;
const TILE_COUNT: u32 = ${u}u;
var<workgroup> tileKey: array<atomic<u32>, ${B}>;
var<workgroup> tileLabel: array<atomic<u32>, ${B}>;
var<workgroup> tileCategory: array<u32, ${B}>;
var<workgroup> tileChanged: atomic<u32>;
var<workgroup> tileContinue: u32;`,`if (workgroupIndex >= TILE_COUNT) { return; }
  let tileColumn = workgroupIndex % TILES_X;
  let tileRow = workgroupIndex / TILES_X;
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * 256u;
    let localColumn = local % TILE;
    let localRow = local / TILE;
    let column = tileColumn * TILE + localColumn;
    let row = tileRow * TILE + localRow;
    var category = 0u;
    var key = SPILL_INFINITE_KEY;
    var label = SPILL_NONE;
    if (column < SPILL_WIDTH && row < SPILL_HEIGHT) {
      let cell = row * SPILL_WIDTH + column;
      let value = elevation[elevationOffset + cell];
      if (isSpillFinite(value)) {
        var boundary = false;
        for (var direction = 0u; direction < 8u; direction++) {
          let neighbor = getSpillNeighbor(cell, direction);
          if (neighbor == SPILL_NONE) {
            boundary = true;
            break;
          }
          if (!isSpillFinite(elevation[elevationOffset + neighbor])) {
            boundary = true;
            break;
          }
        }
        let perimeter = localColumn == 0u || localRow == 0u || localColumn == TILE - 1u || localRow == TILE - 1u;
        if (boundary) {
          category = 3u;
          key = getSpillKey(value);
          label = cell | SPILL_BOUNDARY_FLAG;
        } else if (perimeter) {
          category = 2u;
          key = getSpillKey(value);
          label = cell;
        } else {
          category = 1u;
        }
      }
    }
    tileCategory[local] = category;
    atomicStore(&tileKey[local], key);
    atomicStore(&tileLabel[local], label);
  }
  workgroupBarrier();

  // Fill: key = max(own, lowest neighbor key) to the fixpoint. Keys only decrease (atomicMin), so
  // the update order does not matter; the cap T * T + 1 passes is the longest simple path plus one
  // pass that confirms the fixpoint, and each changing pass lowers a key, so it always exits.
  for (var fillPass = 0u; fillPass <= TILE_CELLS; fillPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * 256u;
      if (tileCategory[local] != 1u) { continue; }
      let column = tileColumn * TILE + local % TILE;
      let row = tileRow * TILE + local / TILE;
      let ownKey = getSpillKey(elevation[elevationOffset + row * SPILL_WIDTH + column]);
      var lowest = SPILL_INFINITE_KEY;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        lowest = min(lowest, atomicLoad(&tileKey[neighborLocal]));
      }
      let candidate = max(ownKey, lowest);
      if (candidate < atomicLoad(&tileKey[local])) {
        atomicMin(&tileKey[local], candidate);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  // Labels: the smallest outlet index reachable through cells of non-increasing key. A neighbor
  // passes its label on when its key is at most the own key (the chain stays within the own key).
  for (var labelPass = 0u; labelPass <= TILE_CELLS; labelPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * 256u;
      if (tileCategory[local] != 1u) { continue; }
      let ownKey = atomicLoad(&tileKey[local]);
      var best = SPILL_NONE;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        let neighborLabel = atomicLoad(&tileLabel[neighborLocal]);
        if (neighborLabel != SPILL_NONE && atomicLoad(&tileKey[neighborLocal]) <= ownKey) {
          best = min(best, neighborLabel & SPILL_ID_MASK);
        }
      }
      if (best < atomicLoad(&tileLabel[local])) {
        atomicMin(&tileLabel[local], best);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * 256u;
    let column = tileColumn * TILE + local % TILE;
    let row = tileRow * TILE + local / TILE;
    if (column >= SPILL_WIDTH || row >= SPILL_HEIGHT) { continue; }
    let cell = row * SPILL_WIDTH + column;
    if (tileCategory[local] == 0u) {
      localFill[localFillOffset + cell] = 0x7fc00000u;
      labels[labelsOffset + cell] = SPILL_NONE;
    } else {
      localFill[localFillOffset + cell] = getSpillValueBits(atomicLoad(&tileKey[local]));
      labels[labelsOffset + cell] = atomicLoad(&tileLabel[local]);
    }
  }`,W(F,je,`${n}-spill-tile-fill`),!1),pe=`${de}
const SPILL_OCEAN: u32 = ${a}u;
const SPILL_FORWARD_SLOTS: u32 = ${4*a}u;

// The decoded edge lives in module-scope private variables (no struct return value).
var<private> edgeValid: bool;
var<private> edgeFirst: u32;
var<private> edgeSecond: u32;
var<private> edgeKey: u32;

// Edge slots [0, 4 n): cell = slot / 4 and one of the four forward directions (E, SE, S, SW),
// an edge between the labels of two adjacent cells. Slots [4 n, 5 n): the ocean edge of cell
// slot - 4 n, present when that cell is a boundary outlet. The slot is the tie-break of equal keys.
fn loadSpillEdge(slot: u32) {
  edgeValid = false;
  if (slot < SPILL_FORWARD_SLOTS) {
    let cell = slot >> 2u;
    let neighbor = getSpillNeighbor(cell, slot & 3u);
    if (neighbor == SPILL_NONE) { return; }
    let firstWord = labels[labelsOffset + cell];
    let secondWord = labels[labelsOffset + neighbor];
    if (firstWord == SPILL_NONE || secondWord == SPILL_NONE) { return; }
    edgeFirst = firstWord & SPILL_ID_MASK;
    edgeSecond = secondWord & SPILL_ID_MASK;
    if (edgeFirst == edgeSecond) { return; }
    edgeKey = max(
      getSpillKey(bitcast<f32>(localFill[localFillOffset + cell])),
      getSpillKey(bitcast<f32>(localFill[localFillOffset + neighbor]))
    );
    edgeValid = true;
    return;
  }
  let cell = slot - SPILL_FORWARD_SLOTS;
  let word = labels[labelsOffset + cell];
  if (word == SPILL_NONE || (word & SPILL_BOUNDARY_FLAG) == 0u) { return; }
  edgeFirst = cell;
  edgeSecond = SPILL_OCEAN;
  edgeKey = getSpillKey(bitcast<f32>(localFill[localFillOffset + cell]));
  edgeValid = true;
}`,me=[{name:`labels`,view:_,type:`u32`,access:`read`},{name:`localFill`,view:g,type:`u32`,access:`read`}],he=Re(a),ge=De(o),V=G(e,{id:`${n}-spill-boruvka`,operation:J,contributorId:n,shapes:[[o,256],[s,256]],statusLength:8}),_e=K(e,{id:`${n}-spill-boruvka-reset`,operation:J,state:V,statusLength:8,enabledBindings:[P],enabledExpression:re}),ve=L(`init`,o,[{name:`comp`,view:v,type:`u32`,access:`read_write`},{name:`head`,view:ee,type:`u32`,access:`read_write`},{name:`parent`,view:j,type:`u32`,access:`read_write`},{name:`parentKey`,view:M,type:`u32`,access:`read_write`}],``,`comp[compOffset + index] = index;
  head[headOffset + index] = 0xffffffffu;
  parent[parentOffset + index] = index;
  parentKey[parentKeyOffset + index] = 0u;`,W(F,Me,`${n}-spill-init`)),H=G(e,{id:`${n}-spill-boruvka-jump`,operation:J,contributorId:n,shapes:[[o,256]]}),U=[];for(let t=0;t<he;t++){let r=`boruvka-${t}`,i=e=>W(V.gate,Fe,`${n}-spill-${r}-${e}`),c=e=>W(V.gate,Ie,`${n}-spill-${r}-${e}`);U.push(L(`${r}-clear`,o,[{name:`bestKey`,view:y,type:`u32`,access:`read_write`},{name:`bestEdge`,view:b,type:`u32`,access:`read_write`}],``,`bestKey[bestKeyOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`,i(`clear`)),L(`${r}-pick-key`,s,[...me,{name:`comp`,view:v,type:`u32`,access:`read`},{name:`bestKey`,view:y,type:`atomic<u32>`,access:`read_write`}],pe,`loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  atomicMin(&bestKey[bestKeyOffset + firstComponent], edgeKey);
  atomicMin(&bestKey[bestKeyOffset + secondComponent], edgeKey);`,c(`pick-key`)),L(`${r}-pick-edge`,s,[...me,{name:`comp`,view:v,type:`u32`,access:`read`},{name:`bestKey`,view:y,type:`u32`,access:`read`},{name:`bestEdge`,view:b,type:`atomic<u32>`,access:`read_write`}],pe,`loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  if (edgeKey == bestKey[bestKeyOffset + firstComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + firstComponent], index);
  }
  if (edgeKey == bestKey[bestKeyOffset + secondComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + secondComponent], index);
  }`,c(`pick-edge`)),L(`${r}-hook`,o,[...me,{name:`comp`,view:v,type:`u32`,access:`read`},{name:`bestEdge`,view:b,type:`u32`,access:`read`},{name:`hookPointer`,view:x,type:`u32`,access:`read_write`},{name:`status`,view:V.status,type:`atomic<u32>`,access:`read_write`}],pe,`var hookTarget = index;
  if (comp[compOffset + index] == index && index != SPILL_OCEAN) {
    let slot = bestEdge[bestEdgeOffset + index];
    if (slot != SPILL_NONE) {
      loadSpillEdge(slot);
      let firstComponent = comp[compOffset + edgeFirst];
      let secondComponent = comp[compOffset + edgeSecond];
      let other = select(firstComponent, secondComponent, firstComponent == index);
      let mutual = bestEdge[bestEdgeOffset + other] == slot;
      if (!(mutual && index > other)) {
        hookTarget = other;
        atomicStore(&status[statusOffset + 0u], 1u);
      }
    }
  }
  hookPointer[hookPointerOffset + index] = hookTarget;`,i(`hook`)),L(`${r}-record`,o,[...me,{name:`bestEdge`,view:b,type:`u32`,access:`read`},{name:`hookPointer`,view:x,type:`u32`,access:`read`},{name:`mstFirst`,view:C,type:`u32`,access:`read_write`},{name:`mstSecond`,view:w,type:`u32`,access:`read_write`},{name:`mstKey`,view:E,type:`u32`,access:`read_write`},{name:`status`,view:V.status,type:`atomic<u32>`,access:`read_write`}],pe,`if (hookPointer[hookPointerOffset + index] == index) { return; }
  loadSpillEdge(bestEdge[bestEdgeOffset + index]);
  let position = atomicAdd(&status[statusOffset + 4u], 1u);
  // A spanning tree of at most n + 1 nodes has at most n edges.
  if (position < ${a}u) {
    mstFirst[mstFirstOffset + position] = edgeFirst;
    mstSecond[mstSecondOffset + position] = edgeSecond;
    mstKey[mstKeyOffset + position] = edgeKey;
  }`,i(`record`)),K(e,{id:`${n}-spill-${r}-jump-reset`,operation:J,state:H,enabledBindings:[{name:`outer`,view:V.status,type:`u32`,access:`read`}],enabledExpression:`outer[outerOffset + 1u] == 1u && outer[outerOffset + 0u] != 0u`}),...A(e,{id:`${n}-spill-${r}-jump`,gateId:`${n}-spill-${r}-jump-gate`,operation:J,state:H,maxIterations:ge,createRound:({nodeId:t,iteration:n,slotBinding:r,markChangedWGSL:i,condition:a,extraResources:s})=>{let[c,l]=n%2==0?[x,S]:[S,x];return p(e,{id:t,operation:J,variant:`spill-jump`,bindings:[{name:`pointerIn`,view:c,type:`u32`,access:`read`},{name:`pointerOut`,view:l,type:`u32`,access:`read_write`},{name:`status`,view:H.status,type:`atomic<u32>`,access:`read_write`},r],invocationCount:o,workgroupSize:256,body:`let next = pointerIn[pointerInOffset + index];
  let jumped = pointerIn[pointerInOffset + next];
  pointerOut[pointerOutOffset + index] = jumped;
  if (jumped != next) {
    ${i}
  }`,condition:a,extraResources:s})}}),L(`${r}-apply`,o,[{name:`comp`,view:v,type:`u32`,access:`read_write`},{name:`hookPointer`,view:x,type:`u32`,access:`read`}],``,`comp[compOffset + index] = hookPointer[hookPointerOffset + comp[compOffset + index]];`,i(`apply`)),te(e,{id:`${n}-spill-${r}-gate`,operation:J,state:V,maxIterations:he}))}let ye=Oe(e,{id:n,operation:J,contributorId:n,cellCount:a,mstFirst:C,mstSecond:w,mstKey:E,head:ee,mstStatus:V.status,parent:j,parentKey:M,enabledBindings:[P],enabledExpression:re,getHalfEdgeGate:I(Ne),getEdgeGate:I(Y)}),xe=L(`final`,a,[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},...me,{name:`parent`,view:j,type:`u32`,access:`read`},{name:`parentKey`,view:M,type:`u32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`status`,view:V.status,type:`atomic<u32>`,access:`read_write`}],`${de}
const SPILL_OCEAN: u32 = ${a}u;`,`let value = elevation[elevationOffset + index];
  if (!isSpillFinite(value)) {
    filled[filledOffset + index] = 0x7fc00000u;
    return;
  }
  let word = labels[labelsOffset + index];
  if (word == SPILL_NONE || parent[parentOffset + (word & SPILL_ID_MASK)] != SPILL_OCEAN) {
    filled[filledOffset + index] = 0x7fc00000u;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let key = max(
    getSpillKey(bitcast<f32>(localFill[localFillOffset + index])),
    parentKey[parentKeyOffset + (word & SPILL_ID_MASK)]
  );
  filled[filledOffset + index] = getSpillValueBits(key);`,W(F,Y,`${n}-spill-final`)),Se=[P,{name:`relaxationStatus`,view:R.state.status,type:`u32`,access:`read`},{name:`spillStatus`,view:V.status,type:`u32`,access:`read`},{name:`combined`,view:N,type:`u32`,access:`read_write`}];t.converged&&Se.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.iterationCount&&Se.push({name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`});let Ce=p(e,{id:`${n}-spill-publish`,operation:J,variant:`spill-publish`,bindings:Se,invocationCount:1,body:`let useRelaxation = mode[modeOffset] == 1u;
  let roundCount = select(spillStatus[spillStatusOffset + 2u], relaxationStatus[relaxationStatusOffset + 2u], useRelaxation);
  let isConverged = select(spillStatus[spillStatusOffset + 3u], relaxationStatus[relaxationStatusOffset + 3u], useRelaxation);
  combined[combinedOffset + 0u] = 0u;
  combined[combinedOffset + 1u] = 0u;
  combined[combinedOffset + 2u] = roundCount;
  combined[combinedOffset + 3u] = isConverged;
  ${t.converged?`converged[convergedOffset] = isConverged;`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = roundCount;`:``}`});return{nodes:[ae,z,ce,le,...ue,_e,fe,ve,...U,...ye.nodes,xe,Ce],status:N}}var Ve=`GPUTerrainFlow`;function He(e,t){let n=[...t.fillStatus?[t.fillStatus]:[],...t.flatStatuses];if(n.length===0)return[];let r=[];for(let[e,n]of[[`filledElevation`,t.filledElevation],[`directions`,t.directions],[`classes`,t.classes],[`accumulation`,t.accumulation],[`streams`,t.streams]])n&&r.push({name:e,view:n,type:`u32`,access:`read_write`});if(r.length===0)return[];let i=P(e,`${t.id}-unresolved`,Ve,t.cellCount,t.id),{x:a,y:o,z:s}=i.layout,{condition:c,extraResources:l}=ee(i,`${t.id}-unresolved-cells`),u=n.map((e,t)=>({name:`status${t}`,view:e,type:`u32`,access:`read`})),d=n.map((e,t)=>`status${t}[status${t}Offset + 3u] != 0u`),f=(e,t,n)=>t?`${e}[${e}Offset + index] = ${n};`:``,m=p(e,{id:`${t.id}-unresolved-arm`,operation:Ve,variant:`unresolved-arm`,bindings:[...u,{name:`dispatch`,view:i.dispatch,type:`u32`,access:`read_write`}],invocationCount:1,body:`let converged = ${d.join(` && `)};
  dispatch[dispatchOffset] = select(${a}u, 0u, converged);
  dispatch[dispatchOffset + 1u] = ${o}u;
  dispatch[dispatchOffset + 2u] = ${s}u;`}),h=t.fillStatus&&t.filledElevation?[{name:`fillStatus`,view:t.fillStatus,type:`u32`,access:`read`}]:[];return[m,p(e,{id:`${t.id}-unresolved-cells`,operation:Ve,variant:`unresolved-cells`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},...h,...r],invocationCount:i.invocationCount,workgroupSize:256,body:`if ((bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) >= 0x7f800000u) { return; }
  ${f(`directions`,t.directions,`${t.codes.direction}u`)}
  ${f(`classes`,t.classes,`${t.codes.cellClass}u`)}
  ${f(`accumulation`,t.accumulation,`0x7fc00000u`)}
  ${f(`streams`,t.streams,`${t.codes.stream}u`)}
  ${h.length>0?`if (fillStatus[fillStatusOffset + 3u] == 0u) { ${f(`filledElevation`,t.filledElevation,`0x7fc00000u`)} }`:``}`,condition:c,extraResources:l})]}var X=`GPUTerrainFlow`,Ue=`
${m}
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;function We(e,t){let n=t.width*t.height,{cellClass:r}=t,i=r=>D(e,X,t.id,`${t.id}-flats-${r}`,`float32`,n),a=t.continuation?.lower.values??i(`lower-values`),o=i(`lower-aux`),s=t.continuation?.higher.values??i(`higher-values`),c=i(`higher-aux`),l=t.continuation?.maximum.values??i(`maximum-values`),u=i(`mask`),d=t.continuation?i(`lower-seeds`):a,f=t.continuation?i(`higher-seeds`):s,m=p(e,{id:`${t.id}-flats-prepare`,operation:X,variant:`flats-prepare`,bindings:[{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`classes`,view:t.classes,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`lowerValues`,view:d,type:`f32`,access:`read_write`},{name:`lowerAuxiliary`,view:o,type:`f32`,access:`read_write`},{name:`higherValues`,view:f,type:`f32`,access:`read_write`},{name:`higherAuxiliary`,view:c,type:`f32`,access:`read_write`}],invocationCount:n,declarations:T(t),body:`_ = settings[settingsOffset];
  let centerClass = classes[classesOffset + index];
  let centerSurface = surface[surfaceOffset + index];
  let isFlat = centerClass == ${r.flat}u;
  let hasFlow = centerClass == ${r.draining}u || centerClass == ${r.outlet}u;
  var isLowEdge = false;
  var isHighEdge = false;
  if (isFlat || hasFlow) {
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getD8Neighbor(index, direction);
      if (neighbor == GRID_NONE) { continue; }
      let neighborClass = classes[classesOffset + neighbor];
      if (neighborClass == ${r.invalid}u) { continue; }
      let neighborSurface = surface[surfaceOffset + neighbor];
      if (hasFlow && neighborClass == ${r.flat}u && neighborSurface == centerSurface) {
        isLowEdge = true;
      }
      if (isFlat && neighborSurface > centerSurface) { isHighEdge = true; }
    }
  }
  lowerAuxiliary[lowerAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat || isLowEdge);
  lowerValues[lowerValuesOffset + index] = select(getInfinity(), 1.0, isLowEdge);
  higherAuxiliary[higherAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat);
  higherValues[higherValuesOffset + index] = select(getInfinity(), 1.0, isHighEdge);`}),h=(r,i,a,o)=>{let s={id:`${t.id}-flats-${r}`,operation:X,width:t.width,height:t.height,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations,continuation:t.continuation&&{state:t.continuation[r].state,retainConverged:!0,prerequisite:r===`maximum`?t.continuation.higher.state:void 0,restartShapes:[[n]]}},{relaxation:c,resetNodes:l}=O(e,s,!0);return{relaxation:c,resetNodes:l,relaxNodes:ne(e,{...s,relaxation:c,sweepAfterIteration:96,values:i,auxiliary:a,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (neighborAuxiliary == centerAuxiliary) { return ${o}; }
  return getInfinity();
}`})}},g=h(`lower`,a,o,`neighborValue + 1.0`),_=h(`higher`,s,c,`neighborValue + 1.0`),v=h(`maximum`,l,t.surface,`neighborValue`),y=k(t.width),b=p(e,{id:`${t.id}-flats-seed-edges`,operation:X,variant:`flats-seed-edges`,bindings:[{name:`lowerValues`,view:a,type:`f32`,access:`read`},{name:`higherValues`,view:s,type:`f32`,access:`read`},{name:`lowerStamps`,view:g.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`},{name:`higherStamps`,view:_.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${Ue}\n${y}`,body:`if (isFiniteValue(lowerValues[lowerValuesOffset + index])) {
    atomicMax(&lowerStamps[lowerStampsOffset + getRelaxationTile(index)], 1u);
  }
  if (isFiniteValue(higherValues[higherValuesOffset + index])) {
    atomicMax(&higherStamps[higherStampsOffset + getRelaxationTile(index)], 1u);
  }`}),x=p(e,{id:`${t.id}-flats-prepare-maximum`,...v.relaxation.continuation?.getRestartGate(0,`${t.id}-flats-prepare-maximum`),operation:X,variant:`flats-prepare-maximum`,bindings:[{name:`higherValues`,view:s,type:`f32`,access:`read`},{name:`maximumValues`,view:l,type:`f32`,access:`read_write`},{name:`maximumStamps`,view:v.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${Ue}\n${y}`,body:`let higher = higherValues[higherValuesOffset + index];
  let reached = isFiniteValue(higher);
  maximumValues[maximumValuesOffset + index] = select(0.0, -higher, reached);
  if (reached) {
    atomicMax(&maximumStamps[maximumStampsOffset + getRelaxationTile(index)], 1u);
  }`}),S=p(e,{id:`${t.id}-flats-mask`,operation:X,variant:`flats-mask`,bindings:[{name:`lowerValues`,view:a,type:`f32`,access:`read`},{name:`higherValues`,view:s,type:`f32`,access:`read`},{name:`maximumValues`,view:l,type:`f32`,access:`read`},{name:`mask`,view:u,type:`f32`,access:`read_write`}],invocationCount:n,declarations:Ue,body:`let towardLower = lowerValues[lowerValuesOffset + index];
  if (!isFiniteValue(towardLower)) {
    mask[maskOffset + index] = getInfinity();
    return;
  }
  let awayFromHigher = higherValues[higherValuesOffset + index];
  var awayTerm = 0.0;
  if (isFiniteValue(awayFromHigher)) {
    awayTerm = -maximumValues[maximumValuesOffset + index] - awayFromHigher;
  }
  mask[maskOffset + index] = 2.0 * towardLower + awayTerm;`}),C=[{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`mask`,view:u,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`classes`,view:t.classes,type:`u32`,access:`read_write`}];t.directions&&C.push({name:`directions`,view:t.directions,type:`u32`,access:`read_write`}),t.receivers&&C.push({name:`receivers`,view:t.receivers,type:`u32`,access:`read_write`});let w=p(e,{id:`${t.id}-flats-route`,operation:X,variant:`flats-route`,bindings:C,invocationCount:n,declarations:T(t),body:`_ = settings[settingsOffset];
  if (classes[classesOffset + index] != ${r.flat}u) { return; }
  let centerMask = mask[maskOffset + index];
  if (!isFiniteValue(centerMask)) { return; }
  let centerSurface = surface[surfaceOffset + index];
  var bestMask = centerMask;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || surface[surfaceOffset + neighbor] != centerSurface) { continue; }
    let neighborMask = mask[maskOffset + neighbor];
    // Strictly smaller: ties keep the lowest direction index.
    if (isFiniteValue(neighborMask) && neighborMask < bestMask) {
      bestMask = neighborMask;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    classes[classesOffset + index] = ${r.draining}u;
    ${t.directions?`directions[directionsOffset + index] = getD8Code(bestDirection);`:``}
    ${t.receivers?`receivers[receiversOffset + index] = bestNeighbor;`:``}
  }`}),E=[];if(t.continuation){E.push(p(e,{id:`${t.id}-flats-continuation-reset`,operation:X,bindings:[{name:`lower`,view:t.continuation.lower.state,type:`u32`,access:`read_write`},{name:`higher`,view:t.continuation.higher.state,type:`u32`,access:`read_write`},{name:`maximum`,view:t.continuation.maximum.state,type:`u32`,access:`read_write`}],invocationCount:1,body:`let restart = lower[lowerOffset + 5u] != 0u || higher[higherOffset + 5u] != 0u || maximum[maximumOffset + 5u] != 0u ||
        (lower[lowerOffset + 3u] != 0u && higher[higherOffset + 3u] != 0u && maximum[maximumOffset + 3u] != 0u);
        if (restart) {
          lower[lowerOffset + 2u] = 0u; lower[lowerOffset + 3u] = 0u;
          higher[higherOffset + 2u] = 0u; higher[higherOffset + 3u] = 0u;
          maximum[maximumOffset + 2u] = 0u; maximum[maximumOffset + 3u] = 0u;
          lower[lowerOffset + 5u] = 0u; higher[higherOffset + 5u] = 0u; maximum[maximumOffset + 5u] = 0u;
        }`})),E.push(m);for(let[r,i,o,c]of[[g,d,a,`lower`],[_,f,s,`higher`]])E.push(...r.resetNodes,p(e,{id:`${t.id}-flats-${c}-initialize`,operation:X,...r.relaxation.continuation?.getRestartGate(0,`${t.id}-flats-${c}-initialize`),bindings:[{name:`seeds`,view:i,type:`f32`,access:`read`},{name:`values`,view:o,type:`f32`,access:`read_write`}],invocationCount:n,body:`values[valuesOffset + index] = seeds[seedsOffset + index];`}),...r.relaxNodes);E.push(...v.resetNodes,x,...v.relaxNodes,S,w)}else E.push(m,...g.resetNodes,..._.resetNodes,...v.resetNodes,b,...g.relaxNodes,..._.relaxNodes,x,...v.relaxNodes,S,w);return(t.converged||t.iterationCount)&&E.push(p(e,{id:`${t.id}-flats-finalize`,operation:X,variant:`flats-finalize`,bindings:[{name:`lowerStatus`,view:g.relaxation.state.status,type:`u32`,access:`read`},{name:`higherStatus`,view:_.relaxation.state.status,type:`u32`,access:`read`},{name:`maximumStatus`,view:v.relaxation.state.status,type:`u32`,access:`read`},...t.converged?[{name:`converged`,view:t.converged,type:`u32`,access:`read_write`}]:[],...t.iterationCount?[{name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,body:`${t.converged?`converged[convergedOffset] = select(0u, 1u,
    lowerStatus[lowerStatusOffset + 3u] == 1u &&
    higherStatus[higherStatusOffset + 3u] == 1u &&
    maximumStatus[maximumStatusOffset + 3u] == 1u);`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = lowerStatus[lowerStatusOffset + 2u] +
    higherStatus[higherStatusOffset + 2u] + maximumStatus[maximumStatusOffset + 2u];`:``}`})),{nodes:E,statuses:[g,_,v].map(({relaxation:e})=>e.state.status)}}var Ge=`GPUTerrainFlow`;function Ke(e,t){let n=`
// Fraction of 'cell' flow that goes to the neighbor in 'direction' when no lower neighbor exists.
fn getFallbackFraction(cell: u32, direction: u32) -> f32 {
  let receiver = receivers[receiversOffset + cell];
  return select(0.0, 1.0, receiver != GRID_NONE && receiver == getD8Neighbor(cell, direction));
}`;if(e===`d-infinity`)return`${n}
// Tarboton (1997) facets in the documented order: index -> (cardinal, diagonal).
fn getFacetCardinal(facet: u32) -> u32 { return (facet / 2u) * 2u; }
fn getFacetDiagonal(facet: u32) -> u32 {
  let cardinal = getFacetCardinal(facet);
  let lower = select(cardinal - 1u, 1u, cardinal == 0u);
  let upper = select(cardinal + 1u, 7u, cardinal == 0u);
  return select(lower, upper, (facet & 1u) == 1u);
}

${t?`// Writes the cell's D-infinity routing record: [cardinal | diagonal << 4 (8 = no lower facet),
// diagonal fraction bits], or ROUTING_INVALID for an invalid cell.
fn computeRouting(cell: u32) {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) {
    routing[routingOffset + 2u * cell] = ROUTING_INVALID;
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  let row = cell / GRID_WIDTH;
  var bestSlope = 0.0;
  var bestCardinal = 8u;
  var bestDiagonal = 8u;
  var bestDiagonalFraction = 0.0;
  for (var facet = 0u; facet < 8u; facet++) {
    let cardinal = getFacetCardinal(facet);
    let diagonal = getFacetDiagonal(facet);
    let cardinalNeighbor = getD8Neighbor(cell, cardinal);
    let diagonalNeighbor = getD8Neighbor(cell, diagonal);
    var cardinalUsable = false;
    var diagonalUsable = false;
    var e1 = 0.0;
    var e2 = 0.0;
    if (cardinalNeighbor != GRID_NONE) {
      e1 = surface[surfaceOffset + cardinalNeighbor];
      cardinalUsable = isFiniteValue(e1);
    }
    if (diagonalNeighbor != GRID_NONE) {
      e2 = surface[surfaceOffset + diagonalNeighbor];
      diagonalUsable = isFiniteValue(e2);
    }
    if (!cardinalUsable && !diagonalUsable) { continue; }
    let d1 = getD8Distance(cardinal, row);
    let diagonalDistance = getD8Distance(diagonal, row);
    let diagonalRow = f32(row) + f32(getD8RowOffset(diagonal));
    let ground = getGroundCellSizeAtRowCoordinate((f32(row) + diagonalRow + 1.0) * 0.5);
    let d2 = select(ground.x, ground.y, cardinal == 0u || cardinal == 4u);
    var slope = 0.0;
    var fractionDiagonal = 0.0;
    if (cardinalUsable && diagonalUsable) {
      let s1 = (center - e1) / d1;
      let s2 = (e1 - e2) / d2;
      if (s1 > 0.0 && s2 == 0.0) {
        // Exactly cardinal: no sqrt/atan, so equal-height ties compare bit-exactly.
        slope = s1;
      } else if (s1 > 0.0 && s2 > 0.0) {
        let angle = atan2(s2, s1);
        let maximumAngle = atan2(d2, d1);
        if (angle <= maximumAngle) {
          slope = sqrt(s1 * s1 + s2 * s2);
          fractionDiagonal = angle / maximumAngle;
        } else {
          slope = (center - e2) / diagonalDistance;
          fractionDiagonal = 1.0;
        }
      } else if (s1 > 0.0) {
        slope = s1;
      } else if (s2 > 0.0) {
        slope = (center - e2) / diagonalDistance;
        fractionDiagonal = 1.0;
      }
    } else if (cardinalUsable) {
      slope = (center - e1) / d1;
    } else {
      slope = (center - e2) / diagonalDistance;
      fractionDiagonal = 1.0;
    }
    // Strictly greater: ties keep the first facet.
    if (slope > bestSlope) {
      bestSlope = slope;
      bestCardinal = cardinal;
      bestDiagonal = diagonal;
      bestDiagonalFraction = fractionDiagonal;
    }
  }
  routing[routingOffset + 2u * cell] = bestCardinal | (bestDiagonal << 4u);
  routing[routingOffset + 2u * cell + 1u] = bitcast<u32>(bestDiagonalFraction);
}

`:``}
fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let code = routing[routingOffset + 2u * cell];
  if (code == ROUTING_INVALID) { return 0.0; }
  let bestCardinal = code & 15u;
  if (bestCardinal == 8u) { return getFallbackFraction(cell, direction); }
  let bestDiagonal = code >> 4u;
  let bestDiagonalFraction = bitcast<f32>(routing[routingOffset + 2u * cell + 1u]);
  if (direction == bestDiagonal) { return bestDiagonalFraction; }
  if (direction == bestCardinal) { return 1.0 - bestDiagonalFraction; }
  return 0.0;
}`;let r=e===`mfd-quinn`;return`${n}
fn getFlowExponent() -> f32 {
  let exponent = settings[settingsOffset + 6u];
  return select(${r?`1.0`:`1.1`}, exponent, isFiniteValue(exponent) && exponent > 0.0);
}

// Slope (Freeman) or slope times contour length (Quinn) of one lower neighbor.
fn getMultipleFlowScore(drop: f32, direction: u32, row: u32) -> f32 {
  let tangent = drop / getD8Distance(direction, row);
  ${r?`let ground = getGroundCellSizeAtRowCoordinate(f32(row) + 0.5);
  var contourLength = 0.25 * getD8Distance(direction, row);
  if (direction == 0u || direction == 4u) { contourLength = 0.5 * ground.y; }
  if (direction == 2u || direction == 6u) { contourLength = 0.5 * ground.x; }
  return tangent * contourLength;`:`return tangent;`}
}

${t?`// Writes the cell's MFD routing record: [largest score bits (-1 = no lower neighbor), weight total
// bits], or ROUTING_INVALID for an invalid cell.
fn computeRouting(cell: u32) {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) {
    routing[routingOffset + 2u * cell] = ROUTING_INVALID;
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  let row = cell / GRID_WIDTH;
  var hasLower = false;
  var maximumScore = 0.0;
  for (var other = 0u; other < 8u; other++) {
    let neighbor = getD8Neighbor(cell, other);
    if (neighbor == GRID_NONE) { continue; }
    let drop = center - surface[surfaceOffset + neighbor];
    if (isFiniteValue(drop) && drop > 0.0) {
      hasLower = true;
      maximumScore = max(maximumScore, getMultipleFlowScore(drop, other, row));
    }
  }
  if (!hasLower) {
    routing[routingOffset + 2u * cell] = bitcast<u32>(-1.0);
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  var total = 0.0;
  for (var other = 0u; other < 8u; other++) {
    total = total + getMultipleFlowWeight(cell, center, other, maximumScore);
  }
  routing[routingOffset + 2u * cell] = bitcast<u32>(maximumScore);
  routing[routingOffset + 2u * cell + 1u] = bitcast<u32>(total);
}

`:``}
// Unnormalized weight of the move from 'cell' toward 'direction'; 0 unless that neighbor is lower.
fn getMultipleFlowWeight(cell: u32, center: f32, direction: u32, maximumScore: f32) -> f32 {
  let neighbor = getD8Neighbor(cell, direction);
  if (neighbor == GRID_NONE) { return 0.0; }
  let drop = center - surface[surfaceOffset + neighbor];
  if (!(isFiniteValue(drop) && drop > 0.0)) { return 0.0; }
  let score = getMultipleFlowScore(drop, direction, cell / GRID_WIDTH);
  let ratio = select(1.0, score / maximumScore, maximumScore > 0.0);
  return select(0.0, pow(ratio, getFlowExponent()), ratio > 0.0);
}

fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let code = routing[routingOffset + 2u * cell];
  if (code == ROUTING_INVALID) { return 0.0; }
  let maximumScore = bitcast<f32>(code);
  if (maximumScore < 0.0) { return getFallbackFraction(cell, direction); }
  let total = bitcast<f32>(routing[routingOffset + 2u * cell + 1u]);
  let weight = getMultipleFlowWeight(cell, surface[surfaceOffset + cell], direction, maximumScore);
  return select(0.0, weight / total, total > 0.0);
}`}function qe(e,t){let n=t.width*t.height,r=P(e,`${t.id}-accumulate`,Ge,n),i=o(e,`${t.id}-routing-records`,`uint32`,2*n),a=e=>`${T(t)}
const ROUTING_INVALID: u32 = 0xffffffffu;
${Ke(t.routing,e)}`,s=[p(e,{id:`${t.id}-routing-prepare`,operation:Ge,variant:`routing-prepare-${t.routing}`,bindings:[{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`routing`,view:i,type:`u32`,access:`read_write`}],invocationCount:n,declarations:a(!0),body:`_ = receivers[receiversOffset];
  _ = settings[settingsOffset];
  computeRouting(index);`}),d(e,{id:`${t.id}-accumulate-init`,operation:Ge,view:t.accumulation,type:`u32`,value:`0xffffffffu`}),j(e,{id:`${t.id}-accumulate-reset`,operation:Ge,state:r})],c=[{name:`acc`,view:t.accumulation,type:`atomic<u32>`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`routing`,view:i,type:`u32`,access:`read`}];t.runoff&&c.push({name:`runoff`,view:t.runoff,type:`f32`,access:`read`}),c.push({name:`status`,view:r.status,type:`atomic<u32>`,access:`read_write`});let l=`${a(!1)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${ye}u;

fn getWeight(cell: u32) -> f32 {
  var weight: f32 = 1.0;
  ${t.runoff?`let runoffValue = runoff[runoffOffset + cell];
  weight = select(0.0, runoffValue, isFiniteValue(runoffValue) && runoffValue >= 0.0);`:``}
  ${t.area?`let ground = getGroundCellSizeAtRowCoordinate(f32(cell / GRID_WIDTH) + 0.5);
  weight = weight * (ground.x * ground.y);`:``}
  return weight;
}

// Returns true when this thread wrote the final value of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&acc[accOffset + cell]) != SENTINEL) { return false; }
  var sum = getWeight(cell);
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(cell, direction);
    if (neighbor == GRID_NONE) { continue; }
    let fraction = getFlowFraction(neighbor, (direction + 4u) & 7u);
    if (fraction > 0.0) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + fraction * bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`,u=e=>`if (!isFiniteValue(surface[surfaceOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  var walk = index;
  var finalized = tryFinalize(index);
  for (var step = 0u; step < MAX_WALK_LENGTH && finalized; step++) {
    // Try every receiver in direction order; continue from the first one this thread finalized.
    var next = GRID_NONE;
    for (var direction = 0u; direction < 8u; direction++) {
      let receiver = getD8Neighbor(walk, direction);
      if (receiver != GRID_NONE && getFlowFraction(walk, direction) > 0.0) {
        if (tryFinalize(receiver) && next == GRID_NONE) { next = receiver; }
      }
    }
    walk = next;
    finalized = next != GRID_NONE;
  }
  if (atomicLoad(&acc[accOffset + index]) == SENTINEL) {
    ${e}
  }`;return s.push(...A(e,{id:`${t.id}-accumulate-round`,gateId:`${t.id}-accumulate-gate`,operation:Ge,state:r,maxIterations:t.maxIterations,createRound:({nodeId:r,slotBinding:i,markChangedWGSL:a,condition:o,extraResources:s})=>p(e,{id:r,operation:Ge,variant:`accumulate-round-${t.routing}`,bindings:[...c,i],invocationCount:n,declarations:l,body:u(a),condition:o,extraResources:s})})),(t.converged||t.iterationCount)&&s.push(N(e,{id:`${t.id}-accumulate-finalize`,operation:Ge,state:r,converged:t.converged,iterationCount:t.iterationCount})),s}var Je=4294967294,Ye=4294967295,Xe={draining:0,flat:1,pit:2,outlet:3,invalid:4,unresolved:5};function Ze(e,t,n=new Float32Array(8)){return v(`getGPUTerrainFlowParameterValues`,n,8),E(n,t),n[4]=e.fillEpsilon??0,n[5]=e.streamThreshold??1/0,n[6]=e.flowExponent??0,n[7]=0,n}function Qe(e,r){let{output:a}=r,o=[[`filledElevation`,a.filledElevation],[`flowDirections`,a.flowDirections],[`cellClasses`,a.cellClasses],[`accumulation`,a.accumulation],[`streams`,a.streams],[`status.fillConverged`,a.status?.fillConverged],[`status.accumulationConverged`,a.status?.accumulationConverged],[`status.flatsConverged`,a.status?.flatsConverged],[`status.fillIterationCount`,a.status?.fillIterationCount],[`status.flatsIterationCount`,a.status?.flatsIterationCount],[`status.accumulationIterationCount`,a.status?.accumulationIterationCount]];i(e,o.map(([,e])=>e),[...y(r.elevation),r.parameters,r.runoff]);let s=[[`filledElevation`,`flowDirections`,`cellClasses`],[`accumulation`,`streams`]];for(let[r,[i,a]]of o.entries())for(let[c,l]of o.slice(r+1)){if(!a||!l||a.buffer!==l.buffer)continue;if(n(a,l))throw Error(`${e} outputs must not share buffers with overlapping byte ranges (${i} and ${c})`);let r=t(a),o=t(l);if(s.some(e=>e.includes(i)&&e.includes(c))&&r.offset<o.offset+o.size&&o.offset<r.offset+r.size)throw Error(`${e} ${i} and ${c} are written by one node and must be at least 256 bytes apart when they share a buffer`)}}var $e=class{id;props;constructor(e){this.id=e.id??`terrain-flow`,this.props=e;let{id:t}=this,{output:n}=e,r=n.status,a=S(t,e.grid);if(C(t,e.parameters,8),w(t,e.cellSizeMode),_(t,e.grid,e.cellSizeMode),![`cells`,`area`].includes(e.accumulationUnits??`cells`))throw Error(`${t} accumulationUnits must be cells or area`);if(M(t,`maximumFillIterations`,e.maximumFillIterations??512),M(t,`maximumFlatIterations`,e.maximumFlatIterations??512),e.flatContinuation){if(!e.resolveFlats)throw Error(`${t} flatContinuation requires resolveFlats`);let r=Object.values(e.flatContinuation).flatMap(e=>[e.state,e.values]);for(let n of Object.values(e.flatContinuation))if(c(n.state,[`uint32`],`${t} flatContinuation.state`),c(n.values,[`float32`],`${t} flatContinuation.values`),n.state.length!==re(e.grid)||n.values.length!==a)throw Error(`${t} flatContinuation has the wrong length`);i(t,r,[...y(e.elevation),e.parameters,e.runoff,n.filledElevation,n.flowDirections,n.cellClasses,n.accumulation,n.streams,...Object.values(n.status??{})]);for(let e=0;e<r.length;e++)for(let n=e+1;n<r.length;n++)if(r[e].buffer===r[n].buffer)throw Error(`${t} flatContinuation buffers must be distinct`)}if(![`d8`,`d-infinity`,`mfd-freeman`,`mfd-quinn`].includes(e.flowRouting??`d8`))throw Error(`${t} flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn`);if((r?.flatsConverged||r?.flatsIterationCount)&&!e.resolveFlats)throw Error(`${t} output.status.flatsConverged and flatsIterationCount require resolveFlats`);if(r?.fillIterationCount&&!e.fillDepressions)throw Error(`${t} output.status.fillIterationCount requires fillDepressions`);if(r?.accumulationIterationCount&&!n.accumulation&&!n.streams)throw Error(`${t} output.status.accumulationIterationCount requires output.accumulation or output.streams`);if(M(t,`maximumAccumulationIterations`,e.maximumAccumulationIterations??64),!n.filledElevation&&!n.flowDirections&&!n.cellClasses&&!n.accumulation&&!n.streams)throw Error(`${t} requires at least one output`);if((n.filledElevation||r?.fillConverged)&&!e.fillDepressions)throw Error(`${t} output.filledElevation and output.status.fillConverged require fillDepressions`);if(r?.accumulationConverged&&!n.accumulation&&!n.streams)throw Error(`${t} output.status.accumulationConverged requires output.accumulation or output.streams`);for(let[r,i]of[[`output.filledElevation`,n.filledElevation],[`output.accumulation`,n.accumulation],[`runoff`,e.runoff]])if(i&&(c(i,[`float32`],`${t} ${r}`),i.length!==a))throw Error(`${t} ${r} must contain one value per cell`);for(let[e,r]of[[`output.flowDirections`,n.flowDirections],[`output.cellClasses`,n.cellClasses],[`output.streams`,n.streams]])if(r&&(c(r,[`uint32`],`${t} ${e}`),r.length!==a))throw Error(`${t} ${e} must contain one value per cell`);for(let[e,n]of[[`fillConverged`,r?.fillConverged],[`accumulationConverged`,r?.accumulationConverged],[`flatsConverged`,r?.flatsConverged],[`fillIterationCount`,r?.fillIterationCount],[`flatsIterationCount`,r?.flatsIterationCount],[`accumulationIterationCount`,r?.accumulationIterationCount]])if(n&&(c(n,[`uint32`],`${t} output.status.${e}`),n.length!==1))throw Error(`${t} output.status.${e} must contain one row`);Qe(t,e)}getCommandNodes(e){let{id:t,props:n}=this,{output:i}=n,{status:a}=i,{width:s,height:c}=n.grid,l=n.cellSizeMode??`uniform`;g(t,e,n.elevation,[]),r(t,e,[n.parameters,n.runoff,...Object.values(n.flatContinuation??{}).flatMap(e=>[e.state,e.values]),i.filledElevation,i.flowDirections,i.cellClasses,i.accumulation,i.streams,a?.fillConverged,a?.accumulationConverged,a?.flatsConverged,a?.fillIterationCount,a?.flatsIterationCount,a?.accumulationIterationCount]);let u=s*c,d={width:s,height:c,cellSizeMode:l},f=x(e,t,n.elevation,n.grid,!0),p=[...f.nodes],m=f.band.storage.values,h=m,_,v=[],y=()=>He(e,{id:t,cellCount:u,elevation:m,fillStatus:_,flatStatuses:v,filledElevation:i.filledElevation,directions:i.flowDirections,classes:i.cellClasses,accumulation:i.accumulation,streams:i.streams,codes:{direction:Je,cellClass:Xe.unresolved,stream:Ye}});if(n.fillDepressions){let r=i.filledElevation??o(e,`${t}-filled`,`float32`,u),s=Be(e,{...d,id:t,maxIterations:n.maximumFillIterations??512,elevation:m,filled:r,settings:n.parameters,converged:a?.fillConverged,iterationCount:a?.fillIterationCount});p.push(...s.nodes),_=s.status,h=r}let b=!!(i.accumulation||i.streams);if(!i.flowDirections&&!i.cellClasses&&!b)return p.push(...y()),p;let S=b?o(e,`${t}-receivers`,`uint32`,u):void 0,C=i.cellClasses??(n.resolveFlats?o(e,`${t}-classes`,`uint32`,u):void 0);if(p.push(xe(e,{...d,id:`${t}-flow-direction`,elevation:m,surface:h,settings:n.parameters,directions:i.flowDirections,classes:C,receivers:S,cellClass:Xe})),n.resolveFlats&&C){let r=We(e,{...d,id:t,maxIterations:n.maximumFlatIterations??(n.flatContinuation?64:512),continuation:n.flatContinuation,elevation:m,surface:h,settings:n.parameters,classes:C,directions:i.flowDirections,receivers:S,converged:a?.flatsConverged,iterationCount:a?.flatsIterationCount,cellClass:Xe});p.push(...r.nodes),v=r.statuses}if(S){let r=i.accumulation??o(e,`${t}-accumulation`,`float32`,u),s=n.flowRouting??`d8`,c={...d,id:t,maxIterations:n.maximumAccumulationIterations??64,elevation:m,receivers:S,accumulation:r,settings:n.parameters,runoff:n.runoff,area:(n.accumulationUnits??`cells`)===`area`,converged:a?.accumulationConverged,iterationCount:a?.accumulationIterationCount};p.push(...s===`d8`?we(e,c):qe(e,{...c,routing:s,surface:h})),i.streams&&p.push(Te(e,{id:`${t}-streams`,cellCount:u,elevation:m,accumulation:r,streams:i.streams,settings:n.parameters,unresolvedValue:Ye}))}return p.push(...y()),p}};function et(e,t){return`
const GRID_WIDTH: u32 = ${e}u;
const GRID_HEIGHT: u32 = ${t}u;
const NO_CELL: u32 = 0xffffffffu;
// GPU_TERRAIN_FLOW_UNRESOLVED: a valid cell whose direction is unknown. It has no receiver but is
// not a terminal cell, so consumers must not read it as an outlet or a sink.
const UNRESOLVED_DIRECTION: u32 = 0xfffffffeu;

fn getFlowReceiver(cell: u32, code: u32) -> u32 {
  if (code == 0u || code > 128u || (code & (code - 1u)) != 0u) { return NO_CELL; }
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let direction = firstTrailingBit(code);
  let column = i32(cell % GRID_WIDTH) + columnOffsets[direction];
  let row = i32(cell / GRID_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return NO_CELL;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}`}function tt(e){switch(e.kind){case`none`:return{declarations:`fn isStopCell(cell: u32, code: u32) -> bool { return false; }`};case`streams`:return{binding:{name:`stops`,view:e.streams,type:`u32`,access:`read`},declarations:`fn isStopCell(cell: u32, code: u32) -> bool {
  return code != NO_CELL && stops[stopsOffset + cell] != 0u;
}`};case`markers`:return{binding:{name:`stops`,view:e.markers,type:`u32`,access:`read`},declarations:`fn isStopCell(cell: u32, code: u32) -> bool {
  return stops[stopsOffset + cell] != NO_CELL;
}`}}}function nt(e,t){let n=t.width*t.height,r=P(e,`${t.id}-pointer`,t.operation,n,t.contributorId),i=tt(t.stops),a=et(t.width,t.height),o=[{name:`flowDirections`,view:t.flowDirections,type:`u32`,access:`read`},...i.binding?[i.binding]:[],{name:`pointer`,view:t.pointer,type:`u32`,access:`read_write`}],s=[p(e,{id:`${t.id}-pointer-init`,operation:t.operation,variant:`pointer-init`,bindings:o,invocationCount:n,declarations:`${a}\n${i.declarations}`,body:`let code = flowDirections[flowDirectionsOffset + index];
  var jumpTarget = index;
  if (!isStopCell(index, code)) {
    let receiver = getFlowReceiver(index, code);
    if (receiver != NO_CELL) { jumpTarget = receiver; }
  }
  pointer[pointerOffset + index] = jumpTarget;`}),j(e,{id:`${t.id}-pointer-reset`,operation:t.operation,state:r})],c=[{name:`pointer`,view:t.pointer,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:r.status,type:`atomic<u32>`,access:`read_write`}],l=e=>`let current = atomicLoad(&pointer[pointerOffset + index]);
  let next = atomicLoad(&pointer[pointerOffset + current]);
  if (next != current) {
    atomicStore(&pointer[pointerOffset + index], next);
    ${e}
  }`;return s.push(...A(e,{id:`${t.id}-pointer-round`,gateId:`${t.id}-pointer-gate`,operation:t.operation,state:r,maxIterations:t.maxIterations,createRound:({nodeId:r,slotBinding:i,markChangedWGSL:a,condition:o,extraResources:s})=>p(e,{id:r,operation:t.operation,variant:`pointer-round`,bindings:[...c,i],invocationCount:n,body:l(a),condition:o,extraResources:s})})),(t.converged||t.iterationCount)&&s.push(N(e,{id:`${t.id}-pointer-finalize`,operation:t.operation,state:r,converged:t.converged,iterationCount:t.iterationCount})),s}function Z(e,t,n,r,i){if(c(n,[r],`${e} ${t}`),n.length!==i)throw Error(`${e} ${t} must contain one value per cell`)}function rt(e,t){for(let[n,r]of[[`converged`,t?.converged],[`iterationCount`,t?.iterationCount]])if(r&&(c(r,[`uint32`],`${e} output.status.${n}`),r.length!==1))throw Error(`${e} output.status.${n} must contain one row`)}var it=`GPUTerrainHeightAboveDrainage`,at=4294967294,ot=class{id;props;constructor(e){this.id=e.id??`terrain-height-above-drainage`,this.props=e;let{id:t}=this,{output:n}=e,r=S(t,e.grid);if(M(t,`maximumIterations`,e.maximumIterations??32),!n.heightAboveDrainage&&!n.drainageCells)throw Error(`${t} requires at least one output`);if(Z(t,`flowDirections`,e.flowDirections,`uint32`,r),Z(t,`streams`,e.streams,`uint32`,r),n.heightAboveDrainage&&Z(t,`output.heightAboveDrainage`,n.heightAboveDrainage,`float32`,r),n.drainageCells&&Z(t,`output.drainageCells`,n.drainageCells,`uint32`,r),e.elevation.storage.kind===`buffer`&&e.elevation.storage.values.length!==r)throw Error(`${t} elevation must contain one value per cell`);rt(t,n.status);let a=[n.heightAboveDrainage,n.drainageCells,n.status?.converged,n.status?.iterationCount];i(t,a,[...y(e.elevation),e.flowDirections,e.streams]),b(t,a,[])}getCommandNodes(e){let{id:t,props:n}=this,{output:i}=n,{width:a,height:s}=n.grid;g(t,e,n.elevation,[]),r(t,e,[n.flowDirections,n.streams,i.heightAboveDrainage,i.drainageCells,i.status?.converged,i.status?.iterationCount]);let c=a*s,l=[],u;if(i.heightAboveDrainage){let r=x(e,t,n.elevation,n.grid,!0);l.push(...r.nodes),u=r.band.storage.values}let d=o(e,`${t}-pointer`,`uint32`,c);l.push(...nt(e,{id:t,contributorId:t,operation:it,width:a,height:s,maxIterations:n.maximumIterations??32,flowDirections:n.flowDirections,pointer:d,stops:{kind:`streams`,streams:n.streams},converged:i.status?.converged,iterationCount:i.status?.iterationCount}));let f=[{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`},{name:`pointer`,view:d,type:`u32`,access:`read`}];return i.drainageCells&&f.push({name:`drainageOut`,view:i.drainageCells,type:`u32`,access:`read_write`}),u&&i.heightAboveDrainage&&f.push({name:`elevation`,view:u,type:`f32`,access:`read`},{name:`heightOut`,view:i.heightAboveDrainage,type:`f32`,access:`read_write`}),l.push(p(e,{id:`${t}-height`,operation:it,variant:`height-above-drainage`,bindings:f,invocationCount:c,declarations:`${et(a,s)}
${h}
${m}`,body:`let reached = pointer[pointerOffset + index];
  var drainage = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL &&
      flowDirections[flowDirectionsOffset + reached] != NO_CELL) {
    if (pointer[pointerOffset + reached] != reached) {
      // Pointer jumping stopped early: 'reached' is an intermediate cell, not the first stream.
      drainage = ${at}u;
    } else if (streams[streamsOffset + reached] != 0u) {
      // An unresolved stream word stops the path too: the first stream may be that cell or lie
      // further down, so the drainage cell is unknown rather than wrong.
      drainage = select(reached, ${at}u,
        streams[streamsOffset + reached] == ${Ye}u);
    } else if (flowDirections[flowDirectionsOffset + reached] == UNRESOLVED_DIRECTION ||
        flowDirections[flowDirectionsOffset + index] == UNRESOLVED_DIRECTION) {
      // The path ends on a cell whose direction is unknown, not on an outlet or a sink.
      drainage = ${at}u;
    }
  }
  ${u?`var result = getQuietNaN();
  if (drainage < ${at}u) {
    let cellElevation = elevation[elevationOffset + index];
    let drainageElevation = elevation[elevationOffset + drainage];
    if (isFiniteValue(cellElevation) && isFiniteValue(drainageElevation)) {
      result = cellElevation - drainageElevation;
    }
  }
  heightOut[heightOutOffset + index] = result;`:``}
  ${i.drainageCells?`drainageOut[drainageOutOffset + index] = drainage;`:``}`})),l}},st=`GPUTerrainWatersheds`,ct=4294967294,lt=class{id;props;constructor(e){this.id=e.id??`terrain-watersheds`,this.props=e;let{id:t}=this,{output:n}=e,r=S(t,e.grid);M(t,`maximumIterations`,e.maximumIterations??32),Z(t,`flowDirections`,e.flowDirections,`uint32`,r),Z(t,`output.labels`,n.labels,`uint32`,r),e.pourPoints&&c(e.pourPoints,[`uint32`],`${t} pourPoints`),rt(t,n.status);let a=[n.labels,n.status?.converged,n.status?.iterationCount];i(t,a,[e.flowDirections,e.pourPoints]),b(t,a,[])}getCommandNodes(e){let{id:t,props:n}=this,{output:i}=n,{width:a,height:o}=n.grid;r(t,e,[n.flowDirections,n.pourPoints,i.labels,i.status?.converged,i.status?.iterationCount]);let s=a*o,c=D(e,`GPUTerrainWatersheds`,t,`${t}-pointer`,`uint32`,s),l=[],u;n.pourPoints&&(u=D(e,`GPUTerrainWatersheds`,t,`${t}-markers`,`uint32`,s),l.push(d(e,{id:`${t}-markers-init`,operation:st,view:u,type:`u32`,value:`0xffffffffu`}),p(e,{id:`${t}-markers-mark`,operation:st,variant:`mark-pour-points`,bindings:[{name:`pourPoints`,view:n.pourPoints,type:`u32`,access:`read`},{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`markers`,view:u,type:`atomic<u32>`,access:`read_write`}],invocationCount:n.pourPoints.length,declarations:`const CELL_COUNT: u32 = ${s}u;`,body:`let cell = pourPoints[pourPointsOffset + index];
  if (cell < CELL_COUNT && flowDirections[flowDirectionsOffset + cell] != 0xffffffffu) {
    atomicMin(&markers[markersOffset + cell], index);
  }`}))),l.push(...nt(e,{id:t,contributorId:t,operation:st,width:a,height:o,maxIterations:n.maximumIterations??32,flowDirections:n.flowDirections,pointer:c,stops:u?{kind:`markers`,markers:u}:{kind:`none`},converged:i.status?.converged,iterationCount:i.status?.iterationCount}));let f=[{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`pointer`,view:c,type:`u32`,access:`read`},...u?[{name:`markers`,view:u,type:`u32`,access:`read`}]:[],{name:`labels`,view:i.labels,type:`u32`,access:`read_write`}];return l.push(p(e,{id:`${t}-labels`,operation:st,variant:u?`label-pour-points`:`label-basins`,bindings:f,invocationCount:s,declarations:et(a,o),body:`let reached = pointer[pointerOffset + index];
  var label = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL) {
    ${u?`label = markers[markersOffset + reached];
    let isStop = label != NO_CELL;`:`if (flowDirections[flowDirectionsOffset + reached] != NO_CELL) { label = reached; }
    let isStop = false;`}
    // A pointer that is not a fixpoint means the jumping stopped early. A path that ends on an
    // unresolved direction (and no pour point) has an unknown destination, not an outlet.
    if (pointer[pointerOffset + reached] != reached ||
        (!isStop && flowDirections[flowDirectionsOffset + reached] == UNRESOLVED_DIRECTION)) {
      label = ${ct}u;
    }
  }
  labels[labelsOffset + index] = label;`})),l}},Q=`GPUTerrainStreamOrder`,ut=class{id;props;constructor(e){this.id=e.id??`terrain-stream-order`,this.props=e;let{id:t}=this,{output:n}=e,r=S(t,e.grid);M(t,`maximumIterations`,e.maximumIterations??64),Z(t,`flowDirections`,e.flowDirections,`uint32`,r),Z(t,`streams`,e.streams,`uint32`,r),Z(t,`output.streamOrder`,n.streamOrder,`uint32`,r),rt(t,n.status);let a=[n.streamOrder,n.status?.converged,n.status?.iterationCount];i(t,a,[e.flowDirections,e.streams]),b(t,a,[])}getCommandNodes(e){let{width:t,height:n}=this.props.grid;return 8*t*n<=e.device.limits.maxStorageBufferBindingSize?this.getTourCommandNodes(e):this.getPullCommandNodes(e)}getTourCommandNodes(e){let{id:t,props:n}=this,{output:i}=n,{width:a,height:s}=n.grid;r(t,e,[n.flowDirections,n.streams,i.streamOrder,i.status?.converged,i.status?.iterationCount]);let c=a*s,u=2*c,d=(n,r)=>o(e,`${t}-order-${n}`,`uint32`,r),m=d(`member`,c),h=d(`root-length`,c),g=d(`root-offset`,c),_=d(`position`,u),v=d(`level-status`,4),y=[{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`}],b=et(a,s),x=`fn isStreamCell(cell: u32) -> bool {
  return streams[streamsOffset + cell] != 0u && flowDirections[flowDirectionsOffset + cell] != NO_CELL;
}

fn getTourParent(cell: u32) -> u32 {
  if (!isStreamCell(cell)) { return NO_CELL; }
  let receiver = getFlowReceiver(cell, flowDirections[flowDirectionsOffset + cell]);
  if (receiver == NO_CELL || !isStreamCell(receiver)) { return NO_CELL; }
  return receiver;
}`,S=`${b}
${_e(a,s)}
${x}`,C=`${S}
${ve}`,w=H(e,{id:t,operation:Q,contributorId:t,width:a,height:s,declarations:`${b}\n${x}`,parentBindings:y}),T=[{name:`tourSucc`,view:w.successor,type:`u32`,access:`read`},{name:`tourDist`,view:w.distance,type:`u32`,access:`read`}],[E,D]=w.scratch,O=[p(e,{id:`${t}-order-init`,operation:Q,variant:`order-tour-init`,bindings:[...y,{name:`order`,view:i.streamOrder,type:`u32`,access:`read_write`},{name:`member`,view:m,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${b}\n${x}`,body:`let isStream = isStreamCell(index);
  order[orderOffset + index] = select(0u, 1u, isStream);
  member[memberOffset + index] = select(0u, 1u, isStream);`}),...w.nodes,p(e,{id:`${t}-order-root-length`,operation:Q,variant:`order-root-length`,bindings:[...y,...T,{name:`rootLength`,view:h,type:`u32`,access:`read_write`}],invocationCount:c,declarations:C,body:`let isRoot = isTourEdgeResolved(2u * index) && getTourParent(index) == TOUR_NONE;
  rootLength[rootLengthOffset + index] = select(0u, tourDist[tourDistOffset + 2u * index] + 1u, isRoot);`}),...new l({id:`${t}-order-root-scan`,input:h,output:g,mode:`exclusive`}).getCommandNodes(e),p(e,{id:`${t}-order-position`,operation:Q,variant:`order-position`,bindings:[...y,...T,{name:`rootOffset`,view:g,type:`u32`,access:`read`},{name:`position`,view:_,type:`u32`,access:`read_write`}],invocationCount:u,declarations:C,body:`if (!isTourEdgeResolved(index)) {
    position[positionOffset + index] = TOUR_NONE;
    return;
  }
  let root = tourSucc[tourSuccOffset + index] >> 1u;
  position[positionOffset + index] = rootOffset[rootOffsetOffset + root] +
    tourDist[tourDistOffset + 2u * root] - tourDist[tourDistOffset + index];`})],ee=Math.min(32,Math.max(1,31-Math.clz32(Math.max(c,1)))),k=f(e,{id:`${t}-order-level-dispatch`,operation:Q,shapes:[[c,256],[u,256]]});O.push(p(e,{id:`${t}-order-level-reset`,operation:Q,variant:`order-level-reset`,bindings:[{name:`status`,view:v,type:`u32`,access:`read_write`},k.binding],invocationCount:1,body:`status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${k.armWGSL}`}));let te=(e,t)=>{let{condition:n,extraResources:r}=k.getGate(t);return{condition:{...n,id:`${e}-gate`},extraResources:r}};for(let n=0;n<ee;n++){let r=`${t}-order-level-${n}-scatter`,a=`${t}-order-level-${n}-gather`;O.push(p(e,{id:r,operation:Q,variant:`order-level-scatter`,bindings:[...y,{name:`position`,view:_,type:`u32`,access:`read`},{name:`member`,view:m,type:`u32`,access:`read`},{name:`indicator`,view:E,type:`u32`,access:`read_write`}],invocationCount:u,declarations:S,body:`let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var junction = 0u;
  if ((index & 1u) == 0u) {
    let cell = index >> 1u;
    var childCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell && member[memberOffset + neighbor] != 0u) {
        childCount = childCount + 1u;
      }
    }
    junction = select(0u, 1u, childCount >= 2u);
  }
  indicator[indicatorOffset + place] = junction;`,...te(r,1)}),...new l({id:`${t}-order-level-${n}-scan`,input:E,output:D,mode:`exclusive`}).getCommandNodes(e),p(e,{id:a,operation:Q,variant:`order-level-gather`,bindings:[{name:`position`,view:_,type:`u32`,access:`read`},{name:`prefix`,view:D,type:`u32`,access:`read`},{name:`member`,view:m,type:`u32`,access:`read_write`},{name:`order`,view:i.streamOrder,type:`u32`,access:`read_write`},{name:`status`,view:v,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:`const TOUR_NONE: u32 = 0xffffffffu;`,body:`if (member[memberOffset + index] == 0u) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) { return; }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = order[orderOffset + index] + 1u;
    atomicAdd(&status[statusOffset + 0u], 1u);
  } else {
    member[memberOffset + index] = 0u;
  }`,...te(a,0)}),p(e,{id:`${t}-order-level-${n}-gate`,operation:Q,variant:`order-level-gate`,bindings:[{name:`status`,view:v,type:`u32`,access:`read_write`},k.binding],invocationCount:1,body:`if (status[statusOffset + 1u] == 0u) { return; }
  let joined = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  status[statusOffset + 2u] = status[statusOffset + 2u] + 1u;
  if (joined == 0u) {
    status[statusOffset + 1u] = 0u;
    ${k.disarmWGSL}
  }`}))}return O.push(p(e,{id:`${t}-order-unknown-scatter`,operation:Q,variant:`order-unknown-scatter`,bindings:[...y,{name:`position`,view:_,type:`u32`,access:`read`},{name:`indicator`,view:E,type:`u32`,access:`read_write`}],invocationCount:u,declarations:S,body:`let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var seed = 0u;
  let cell = index >> 1u;
  if ((index & 1u) == 0u && isStreamCell(cell)) {
    var isSeed = streams[streamsOffset + cell] == ${Ye}u ||
      flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && isStreamCell(neighbor) &&
          flowDirections[flowDirectionsOffset + neighbor] == UNRESOLVED_DIRECTION) {
        isSeed = true;
      }
    }
    seed = select(0u, 1u, isSeed);
  }
  indicator[indicatorOffset + place] = seed;`}),...new l({id:`${t}-order-unknown-scan`,input:E,output:D,mode:`exclusive`}).getCommandNodes(e),p(e,{id:`${t}-order-unknown-gather`,operation:Q,variant:`order-unknown-gather`,bindings:[...y,{name:`position`,view:_,type:`u32`,access:`read`},{name:`prefix`,view:D,type:`u32`,access:`read`},{name:`order`,view:i.streamOrder,type:`u32`,access:`read_write`},{name:`status`,view:w.state.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:c,declarations:S,body:`if (!isStreamCell(index)) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) {
    order[orderOffset + index] = NO_CELL;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = NO_CELL;
  }`})),(i.status?.converged||i.status?.iterationCount)&&O.push(N(e,{id:`${t}-order-finalize`,operation:Q,state:w.state,converged:i.status.converged,iterationCount:i.status.iterationCount})),O}getPullCommandNodes(e){let{id:t,props:n}=this,{output:i}=n,{width:a,height:o}=n.grid;r(t,e,[n.flowDirections,n.streams,i.streamOrder,i.status?.converged,i.status?.iterationCount]);let s=a*o,c=n.maximumIterations??64,l=P(e,`${t}-order`,Q,s,t),u=et(a,o),d=[p(e,{id:`${t}-order-init`,operation:Q,variant:`order-init`,bindings:[{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`},{name:`order`,view:i.streamOrder,type:`u32`,access:`read_write`}],invocationCount:s,declarations:u,body:`let isStream = streams[streamsOffset + index] != 0u &&
    flowDirections[flowDirectionsOffset + index] != NO_CELL;
  order[orderOffset + index] = select(0u, NO_CELL, isStream);`}),j(e,{id:`${t}-order-reset`,operation:Q,state:l})],f=[{name:`order`,view:i.streamOrder,type:`atomic<u32>`,access:`read_write`},{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`},{name:`status`,view:l.status,type:`atomic<u32>`,access:`read_write`}],m=`${u}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${ye}u;

// Unknown stream status (GPUTerrainFlow ran out of accumulation rounds): never finalized, so the
// cell and every stream cell downstream of it keep the unresolved sentinel.
fn isUnresolvedStream(cell: u32) -> bool {
  return streams[streamsOffset + cell] == ${Ye}u ||
    flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
}

// Returns true when this thread wrote the final order of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&order[orderOffset + cell]) != SENTINEL || isUnresolvedStream(cell)) { return false; }
  var maximumOrder = 0u;
  var maximumCount = 0u;
  for (var direction = 0u; direction < 8u; direction++) {
    // The neighbor in 'direction' donates when its own code points back at this cell.
    let donor = getFlowReceiver(cell, 1u << direction);
    if (donor == NO_CELL) { continue; }
    if (streams[streamsOffset + donor] == 0u) { continue; }
    // A stream neighbor with an unknown direction might donate: this order cannot be known.
    if (flowDirections[flowDirectionsOffset + donor] == UNRESOLVED_DIRECTION) { return false; }
    if (flowDirections[flowDirectionsOffset + donor] != (1u << ((direction + 4u) & 7u))) { continue; }
    let donorOrder = atomicLoad(&order[orderOffset + donor]);
    if (donorOrder == SENTINEL) { return false; }
    if (donorOrder > maximumOrder) {
      maximumOrder = donorOrder;
      maximumCount = 1u;
    } else if (donorOrder == maximumOrder) {
      maximumCount = maximumCount + 1u;
    }
  }
  var result = 1u;
  if (maximumCount > 0u) {
    result = maximumOrder + select(0u, 1u, maximumCount >= 2u);
  }
  return atomicCompareExchangeWeak(&order[orderOffset + cell], SENTINEL, result).exchanged;
}`,h=e=>`if (atomicLoad(&order[orderOffset + index]) != SENTINEL) { return; }
  // Stays unresolved without marking a change: no round can resolve it.
  if (isUnresolvedStream(index)) { return; }
  if (tryFinalize(index)) {
    var walk = getFlowReceiver(index, flowDirections[flowDirectionsOffset + index]);
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == NO_CELL || !tryFinalize(walk)) { break; }
      walk = getFlowReceiver(walk, flowDirections[flowDirectionsOffset + walk]);
    }
  }
  if (atomicLoad(&order[orderOffset + index]) == SENTINEL) {
    ${e}
  }`;return d.push(...A(e,{id:`${t}-order-round`,gateId:`${t}-order-gate`,operation:Q,state:l,maxIterations:c,createRound:({nodeId:t,slotBinding:n,markChangedWGSL:r,condition:i,extraResources:a})=>p(e,{id:t,operation:Q,variant:`order-round`,bindings:[...f,n],invocationCount:s,declarations:m,body:h(r),condition:i,extraResources:a})})),(i.status?.converged||i.status?.iterationCount)&&d.push(N(e,{id:`${t}-order-finalize`,operation:Q,state:l,converged:i.status.converged,iterationCount:i.status.iterationCount})),d}},dt=`GPUTerrainHydrologicIndices`;function ft(e,t,n=new Float32Array(8)){return v(`getGPUTerrainHydrologicIndicesParameterValues`,n,8),E(n,t),n[4]=e.minimumSlope??.001,n[5]=0,n[6]=0,n[7]=0,n}var pt=class{id;props;constructor(e){this.id=e.id??`terrain-hydrologic-indices`,this.props=e;let{id:t}=this,n=S(t,e.grid);C(t,e.parameters,8),w(t,e.cellSizeMode),_(t,e.grid,e.cellSizeMode);let r=this.getOutputs();if(r.length===0)throw Error(`${t} requires at least one output`);for(let[i,a]of[[`accumulation`,e.accumulation],...r.map(([e,t])=>[`output.${e}`,t])])if(c(a,[`float32`],`${t} ${i}`),a.length!==n)throw Error(`${t} ${i} must contain one value per cell`);let a=r.map(([,e])=>e.buffer);if(new Set(a).size!==a.length)throw Error(`${t} outputs must not share buffers with each other`);i(t,r.map(([,e])=>e),[...y(e.elevation),e.accumulation,e.parameters])}getCommandNodes(e){let{id:t,props:n}=this,{width:i,height:a}=n.grid,o=this.getOutputs();g(t,e,n.elevation,[]),r(t,e,[n.accumulation,n.parameters,...o.map(([,e])=>e)]);let s=x(e,t,n.elevation,n.grid,!0),c=[{name:`elevation`,view:s.band.storage.values,type:`f32`,access:`read`},{name:`accumulation`,view:n.accumulation,type:`f32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},...o.map(([e,t])=>({name:e,view:t,type:`f32`,access:`read_write`}))],l=e=>o.map(([t])=>`${t}[${t}Offset + index] = ${e(t)};`).join(`
  `),u={slope:`tangent`,specificCatchmentArea:`catchment`,wetnessIndex:`wetness`,streamPowerIndex:`catchment * tangent`},d=p(e,{id:`${t}-indices`,operation:dt,variant:`indices`,bindings:c,invocationCount:i*a,declarations:T({width:i,height:a,cellSizeMode:n.cellSizeMode??`uniform`}),body:`let row = index / GRID_WIDTH;
  let center = elevation[elevationOffset + index];
  let area = accumulation[accumulationOffset + index];
  if (!isFiniteValue(center) || !isFiniteValue(area)) {
    ${l(()=>`getQuietNaN()`)}
    return;
  }
  var steepest = 0.0;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) { continue; }
    let neighborValue = elevation[elevationOffset + neighbor];
    if (!isFiniteValue(neighborValue)) { continue; }
    steepest = max(steepest, (center - neighborValue) / getD8Distance(direction, row));
  }
  var minimumSlope = settings[settingsOffset + 4u];
  if (!(minimumSlope >= 0.0) || !isFiniteValue(minimumSlope)) { minimumSlope = 0.0; }
  let tangent = max(steepest, minimumSlope);
  let ground = getGroundCellSizeAtRowCoordinate(f32(row) + 0.5);
  let catchment = area / sqrt(ground.x * ground.y);
  // WGSL leaves division by zero and log(0) unspecified, so the limits are written explicitly.
  var wetness = getInfinity();
  if (catchment <= 0.0) {
    wetness = -getInfinity();
  } else if (tangent > 0.0) {
    wetness = log(catchment / tangent);
  }
  ${l(e=>u[e])}`});return[...s.nodes,d]}getOutputs(){let{output:e}=this.props;return[[`slope`,e.slope],[`specificCatchmentArea`,e.specificCatchmentArea],[`wetnessIndex`,e.wetnessIndex],[`streamPowerIndex`,e.streamPowerIndex]].filter(e=>!!e[1])}},mt=256,ht=256,gt=128,_t=[0,1e-4,.001,.01,.1],vt=315,yt=40,$=8,bt=4294967295,xt=5,St=8,Ct=[{name:`Phantom Ranch (Bright Angel Creek)`,lngLat:[-112.0953,36.107],snapRadius:16,maximumAreaKm2:300},{name:`Indian Garden (Garden Creek)`,lngLat:[-112.1262,36.0771],snapRadius:24,maximumAreaKm2:120},{name:`Colorado River at Bright Angel Creek`,lngLat:[-112.093,36.106],snapRadius:16,maximumAreaKm2:1/0}],wt=[[230,90,90,215],[240,170,60,215],[225,215,80,215],[110,205,100,215],[70,200,190,215],[80,140,240,215],[150,110,235,215],[225,100,200,215]],Tt=[255,176,0,170],Et=[[0,0,0,0],[120,190,255,255],[60,225,225,255],[110,235,110,255],[250,230,70,255],[255,150,50,255],[255,80,80,255],[240,90,235,255]],Dt=[1.2,2,3.2,4.6],Ot=[4,8.8],kt=[2,18],At=[.5,3.5],jt=[-1,3];async function Mt(t){let{device:n}=t,r=he(t.datasets.get(`grand-canyon-dem`),1),{width:i,height:o,cellCount:s}=r,c=[r.origin[0],r.origin[1],0],l=r.groundCellSize[0]*r.groundCellSize[1],d=new a(n,`watersheds`),f=!1,p=e=>d.createBuffer(e,s*4),m=d.createBuffer(`elevation`,r.elevation),h=p(`hillshade`),g=p(`filled`),_=p(`directions`),v=p(`accumulation`),y=p(`streams`),b=p(`hand`),x=p(`basin-labels`),S=p(`watershed-labels`),C=p(`stream-order`),w=p(`catchment`),T=p(`wetness`),E=p(`power`),D=p(`accumulation-display`),O=p(`index-display`),ee=p(`basin-class`),k=p(`fill-depth`),te=p(`flood-depth`),ne=p(`flood-mask`),A=d.createBuffer(`stream-segments`,s*16),j=Dt.map((e,t)=>p(`stream-width-${t}`)),M=[`fill`,`flats`,`accumulation`,`hand`,`basins`,`watersheds`,`order`],N=M.map(e=>d.createBuffer(`${e}-converged`,4)),P=[`fill`,`flats`,`accumulation`,`order`],re=P.map(e=>d.createBuffer(`${e}-iterations`,4)),R=d.createBuffer(`stream-counts`,8),B=d.createBuffer(`flood-counts`,8),V=d.createBuffer(`order-histogram`,St*4),_e=d.createBuffer(`watershed-histogram`,$*4),ve=d.createParameterBuffer(`flow-settings`,`float32`,8),H=d.createParameterBuffer(`indices-settings`,`float32`,8),U=d.createParameterBuffer(`derivatives-settings`,`float32`,8),ye=d.createParameterBuffer(`display-parameters`,`float32`,2),be=new Uint32Array($).fill(bt),xe=d.createParameterBuffer(`pour-points`,`uint32`,$,be),Se=d.createBuffer(`pour-point-positions`,$*8),Ce=d.createBuffer(`pour-point-indices`,new Uint32Array(Array.from({length:$},(e,t)=>t))),we=new e(n,{id:`watersheds-setup`});we.add(new ce({id:`derivatives`,grid:r.rasterGrid,elevation:{id:`elevation`,format:`float32`,storage:{kind:`buffer`,values:ie(we,`elevation`,m,`float32`,s)}},parameters:U.importToGraph(we),output:{hillshade:ie(we,`hillshade`,h,`float32`,s)},cellSizeMode:r.cellSizeMode,rowDirection:`south`}));let Te=d.track(await we.compileAsync());async function Ee(t,i){let a=new e(n,{id:`watersheds`}),o=new Map,c=(e,t,n,r=s)=>{let i=o.get(e);return i||(i=ie(a,e,t,n,r),o.set(e,i)),i},l=c(`elevation`,m,`float32`),d=c(`filled`,g,`float32`),f=c(`directions`,_,`uint32`),p=c(`accumulation`,v,`float32`),h=c(`streams`,y,`uint32`),F=c(`stream-order`,C,`uint32`),I=e=>c(`${M[e]}-converged`,N[e],`uint32`,1),L=e=>c(`${P[e]}-iterations`,re[e],`uint32`,1),ae={id:`filled`,format:`float32`,storage:{kind:`buffer`,values:d}};a.add(new $e({id:`flow`,grid:r.rasterGrid,elevation:{id:`elevation`,format:`float32`,storage:{kind:`buffer`,values:l}},parameters:ve.importToGraph(a),cellSizeMode:r.cellSizeMode,fillDepressions:!0,maximumFillIterations:mt,resolveFlats:i,maximumFlatIterations:ht,flowRouting:t,maximumAccumulationIterations:gt,accumulationUnits:`area`,output:{filledElevation:d,flowDirections:f,accumulation:p,streams:h,status:{fillConverged:I(0),fillIterationCount:L(0),...i?{flatsConverged:I(1),flatsIterationCount:L(1)}:{},accumulationConverged:I(2),accumulationIterationCount:L(2)}}})),a.add(new ot({id:`hand`,grid:r.rasterGrid,elevation:ae,flowDirections:f,streams:h,output:{heightAboveDrainage:c(`hand`,b,`float32`),status:{converged:I(3)}}})),a.add(new lt({id:`basins`,grid:r.rasterGrid,flowDirections:f,output:{labels:c(`basin-labels`,x,`uint32`),status:{converged:I(4)}}})),a.add(new lt({id:`watersheds`,grid:r.rasterGrid,flowDirections:f,pourPoints:xe.importToGraph(a),output:{labels:c(`watershed-labels`,S,`uint32`),status:{converged:I(5)}}})),a.add(new ut({id:`order`,grid:r.rasterGrid,flowDirections:f,streams:h,output:{streamOrder:F,status:{converged:I(6),iterationCount:L(3)}}})),a.add(new pt({id:`indices`,grid:r.rasterGrid,elevation:ae,accumulation:p,parameters:H.importToGraph(a),cellSizeMode:r.cellSizeMode,output:{specificCatchmentArea:c(`catchment`,w,`float32`),wetnessIndex:c(`wetness`,T,`float32`),streamPowerIndex:c(`power`,E,`float32`)}}));let oe=ye.importToGraph(a);return ue(a,{id:`accumulation-display`,cellCount:s,input:p,output:c(`accumulation-display`,D,`float32`),useLog:!0}),Nt(a,{cellCount:s,wetness:c(`wetness`,T,`float32`),catchment:c(`catchment`,w,`float32`),power:c(`power`,E,`float32`),kind:oe,output:c(`index-display`,O,`float32`)}),me(a,{id:`fill-depth`,cellCount:s,elevation:l,filled:d,output:c(`fill-depth`,k,`float32`)}),Pt(a,{cellCount:s,hand:c(`hand`,b,`float32`),stage:oe,depth:c(`flood-depth`,te,`float32`),mask:c(`flood-mask`,ne,`uint32`)}),Ft(a,{cellCount:s,labels:c(`basin-labels`,x,`uint32`),output:c(`basin-class`,ee,`uint32`)}),fe(a,{cellCount:s,grid:r,directions:f,streamOrder:F,segments:c(`stream-segments`,A,`float32`,s*4),widths:j.map((e,t)=>c(`stream-width-${t}`,e,`float32`))}),a.add(new u({id:`stream-count`,input:h,output:c(`stream-counts`,R,`uint32`,2),domain:[0,2]})),a.add(new u({id:`flood-count`,input:c(`flood-mask`,ne,`uint32`),output:c(`flood-counts`,B,`uint32`,2),domain:[0,2]})),a.add(new u({id:`order-histogram`,input:F,output:c(`order-histogram`,V,`uint32`,St),domain:[0,St]})),a.add(new u({id:`watershed-histogram`,input:c(`watershed-labels`,S,`uint32`),output:c(`watershed-histogram`,_e,`uint32`,$),domain:[0,$]})),await a.compileAsync()}let De=t.options.routing,W=t.options.resolveFlats,G=d.track(await Ee(De,W)),K=!0,q=[],Oe=!1,J=s,ke=()=>10**t.options.streamArea*1e6,Ae=()=>{ve.write(Ze({fillEpsilon:_t[t.options.fillEpsilon]??0,streamThreshold:ke(),flowExponent:t.options.flowExponent},r.rasterGrid)),K=!0},je=()=>{H.write(ft({minimumSlope:t.options.minimumSlope/100},r.rasterGrid)),K=!0},Me=()=>{let e={wetness:0,catchment:1,power:2}[t.options.indexKind];ye.write(Float32Array.of(e,t.options.floodStage)),K=!0},Ne=()=>{be.fill(bt);let e=new Float32Array($*2);q.forEach((t,n)=>{be[n]=t;let[i,a]=le(r,t);e[2*n]=i,e[2*n+1]=a}),xe.write(be),Se.write(e),K=!0,t.setReadout(`pourPoints`,q.length),t.requestLayers()};U.write(se({azimuthDegrees:vt,altitudeDegrees:yt},r.rasterGrid)),Ae(),je(),Me(),t.setReadout(`grid`,`${i} × ${o} cells`),t.setReadout(`cellSize`,`${r.groundCellSize[0].toFixed(1)} m (${z(l)} m² per cell)`);let Y=1/0,Pe=-1/0;for(let e of r.elevation)e<Y&&(Y=e),e>Pe&&(Pe=e);t.setReadout(`relief`,`${z(Y)} to ${z(Pe)} m (${z(Pe-Y)} m)`),t.setReadout(`pourPoints`,0);let Fe=new oe(d,`watersheds`,[...N.map(e=>({buffer:e,size:4})),{buffer:R,size:8},{buffer:V,size:St*4},...re.map(e=>({buffer:e,size:4})),{buffer:_e,size:$*4},{buffer:B,size:8}],e=>{if(f)return;let n=new Uint32Array(e),r=9+St,i=r+4,a=i+$,o=(e,t,i,a)=>`${t} ${n[e]?`yes`:`NO`} ${n[r+a]}/${i}`;t.setReadout(`converged`,[o(0,`fill`,mt,0),W?o(1,`flats`,ht,1):`flats off`,o(2,`accumulation`,gt,2),n[3]&&n[4]&&n[5]&&n[6]?`tracing yes`:`tracing NO`,`order ${n[r+3]} rounds`].join(`, `));let s=n[8];t.setReadout(`streams`,`${z(s)} cells, ${(s/J*100).toFixed(1)} % of the window`);let c=Array.from(n.slice(10,9+St)),u=c.reduce((e,t,n)=>t>0?n+1:e,0);t.setReadout(`maximumOrder`,u),t.setReadout(`orders`,c.slice(0,Math.max(u,1)).map((e,t)=>`${t+1}: ${z(e)}`).join(`  `));let d=Array.from(n.slice(i,i+q.length));t.setReadout(`watershedAreas`,d.length===0?`no pour points`:d.map((e,t)=>`${t+1}: ${pe(e*l)}`).join(`  `)),t.setReadout(`floodArea`,`${pe(n[a+1]*l)} within ${t.options.floodStage} m of a stream`),Oe||(Oe=!0,Re())});async function Ie(e,t=xt,n=1/0){let r=e%i,a=Math.floor(e/i),s=Math.max(0,a-t),c=Math.min(o-1,a+t),l=c-s+1,u=await v.readAsync(s*i*4,l*i*4),d=new Float32Array(u.buffer,u.byteOffset,l*i),f=e,p=-1/0;for(let e=-t;e<=t;e++){let o=a+e;if(!(o<s||o>c))for(let e=-t;e<=t;e++){let t=r+e;if(t<0||t>=i)continue;let a=d[(o-s)*i+t];Number.isFinite(a)&&a<=n&&a>p&&(f=o*i+t,p=a)}}return f}async function Le(e){let t=await Ie(e);f||(q.length>=$&&q.shift(),q.push(t),Ne())}async function Re(){try{for(let{lngLat:e,snapRadius:t,maximumAreaKm2:n}of Ct){let i=de(r,e[0],e[1]);if(i>=0&&!f){let e=await Ie(i,t,n*1e6);if(f)return;q.push(e)}}Ne()}catch{}}let ze=De,Be=W,Ve=d.track(new ae);function He(){ze=t.options.routing,Be=t.options.resolveFlats,Ve.request({build:async()=>{let e=t.options.routing,n=t.options.resolveFlats;return{graph:d.track(await Ee(e,n)),routing:e,resolveFlats:n}},commit:e=>{let n=G;G=e.graph,De=e.routing,W=e.resolveFlats,d.release(n),K=!0,t.requestLayers()},discard:e=>d.release(e.graph)})}return{getCompiledGraphs:()=>[Te,G],setOption(e,n,r){switch(e){case`routing`:case`resolveFlats`:(r.routing!==ze||r.resolveFlats!==Be)&&He();break;case`flowExponent`:case`streamArea`:case`fillEpsilon`:Ae(),t.requestLayers();break;case`minimumSlope`:je();break;case`indexKind`:case`floodStage`:Me(),t.requestLayers();break;default:t.requestLayers()}},onAction(e){e===`clearPourPoints`?(q.length=0,Ne()):e===`resetPourPoints`&&(q.length=0,Re())},encode(e,t){t.frameIndex<2&&Te.encode(e,{parameters:void 0}),(K||t.frameIndex<2)&&(G.encode(e,{parameters:void 0}),K=!1,Fe.request(e)),Fe.flush(e)},getLayers(){let e=t.options,n={coordinateOrigin:c,gridSize:[i,o],bounds:r.bounds,rowOrigin:`north`},a=t.theme()===`dark`,l=Math.round(e.overlayOpacity*255),u=[new L({...n,id:`watersheds-hillshade`,values:h,valueFormat:`float32`,colormap:`grayscale`,valueRange:[0,1],color:[255,255,255,a?200:170]})],d=(e,t,r,i,a,o)=>new L({...n,id:e,values:t,valueFormat:`float32`,colormap:r,valueRange:i,discardAtOrBelow:a,color:[255,255,255,l],...o?{noDataColor:o}:{}}),f=(e,t,r)=>new L({...n,id:e,values:t,valueFormat:`uint32`,colormap:`category`,palette:r,noDataColor:[0,0,0,0],color:[255,255,255,l]});switch(e.product){case`accumulation`:u.push(d(`watersheds-accumulation`,D,`bamako`,Ot,Ot[0]));break;case`fill`:u.push(d(`watersheds-fill`,k,`inferno`,[0,10],5e-4));break;case`hand`:u.push(d(`watersheds-hand`,b,`inferno`,[0,e.handRange],ge/2,Tt));break;case`flood`:u.push(d(`watersheds-flood`,te,`cividis`,[0,e.floodStage],0,Tt));break;case`watersheds`:u.push(f(`watersheds-labels`,e.basinMode===`pour-points`?S:ee,wt));break;case`indices`:{let t=e.indexKind===`wetness`?kt:e.indexKind===`catchment`?At:jt;u.push(d(`watersheds-index`,O,e.indexKind===`power`?`inferno`:`bamako`,t,ge/2));break}case`stream-order`:Dt.forEach((e,t)=>{u.push(new I({id:`watersheds-order-${t}`,coordinateOrigin:c,segments:A,instanceCount:s,values:C,valueFormat:`uint32`,colormap:`category`,palette:Et,weights:j[t],widthPixels:e}))});break}if(e.showStreams&&e.product!==`stream-order`&&e.product!==`accumulation`&&u.push(new L({...n,id:`watersheds-streams`,values:y,valueFormat:`uint32`,colormap:`mask`,color:a?[90,190,255,255]:[20,110,230,255],noDataColor:Tt})),e.product===`watersheds`&&e.basinMode===`pour-points`){let e=q.length;e>0&&u.push(new F({id:`watersheds-pour-halo`,coordinateOrigin:c,positions:Se,instanceCount:e,radiusPixels:9,color:[255,255,255,255]}),new F({id:`watersheds-pour-points`,coordinateOrigin:c,positions:Se,instanceCount:e,radiusPixels:6,values:Ce,valueFormat:`uint32`,colormap:`category`,palette:wt.map(([e,t,n])=>[e,t,n,255])}))}return u},onClick(e){if(t.options.product!==`watersheds`||t.options.basinMode!==`pour-points`||!e.coordinate)return!1;let n=de(r,e.coordinate[0],e.coordinate[1]);return n<0?!1:(Le(n).catch(()=>{}),!0)},getTooltip(e){if(!e.coordinate)return null;let t=de(r,e.coordinate[0],e.coordinate[1]);return t<0?null:`Elevation ${z(r.elevation[t])} m\n${e.coordinate[1].toFixed(4)}°, ${e.coordinate[0].toFixed(4)}°`},destroy(){f=!0,Fe.stop(),d.destroy()}}}function Nt(e,t){R(e,{id:`index-display`,bindings:[{name:`wetness`,view:t.wetness,type:`f32`,access:`read`},{name:`catchment`,view:t.catchment,type:`f32`,access:`read`},{name:`power`,view:t.power,type:`f32`,access:`read`},{name:`parameters`,view:t.kind,type:`f32`,access:`read`},{name:`display`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:t.cellCount,declarations:B,body:`let kind = u32(parameters[parametersOffset] + 0.5);
  var value = wetness[wetnessOffset + index];
  if (kind == 1u) { value = log2(max(catchment[catchmentOffset + index], 0.000001)) * 0.30102999566; }
  if (kind == 2u) { value = log2(max(power[powerOffset + index], 0.000001)) * 0.30102999566; }
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  display[displayOffset + index] = select(value, ${ge.toFixed(1)}, invalid);`})}function Pt(e,t){R(e,{id:`flood-depth`,bindings:[{name:`hand`,view:t.hand,type:`f32`,access:`read`},{name:`parameters`,view:t.stage,type:`f32`,access:`read`},{name:`depth`,view:t.depth,type:`f32`,access:`read_write`},{name:`mask`,view:t.mask,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,declarations:B,body:`let stage = parameters[parametersOffset + 1u];
  let height = hand[handOffset + index];
  let valid = !isNaNValue(height) && !isInfiniteValue(height) && height >= 0.0;
  let flooded = valid && height <= stage;
  depth[depthOffset + index] = select(0.0, max(stage - height, 0.0001), flooded);
  mask[maskOffset + index] = select(0u, 1u, flooded);`})}function Ft(e,t){R(e,{id:`basin-class`,bindings:[{name:`labels`,view:t.labels,type:`u32`,access:`read`},{name:`classes`,view:t.output,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,body:`let label = labels[labelsOffset + index];
  classes[classesOffset + index] =
    select(((label * 2654435761u) >> 13u) % 8u, 0xffffffffu, label == 0xffffffffu);`})}export{Mt as createWatersheds};