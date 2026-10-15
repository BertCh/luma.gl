import{J as e}from"./src-CNCKFW2b.js";import{x as t,y as n}from"./resources-CwvhEOBe.js";import{o as r,s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as a,t as o}from"./raster-grid-utils-D1bVIm4m.js";var s=1024,c=256;function l(e,t,n){if(!Number.isSafeInteger(n)||n<1||n>1024)throw Error(`${e} ${t} must be an integer in [1, ${s}]`)}function u(e,n,i,a,s=n,l={}){let u=r(e,{id:`${n}-dispatch`,operation:i,shapes:[[a,c]]});return{status:l.status??o(e,i,s,`${n}-status`,`uint32`,4),limitWord:l.limitWord,dispatch:u.view,gate:u,layout:t(i,Math.max(a,1),c,e.device.limits.maxComputeWorkgroupsPerDimension),invocationCount:a}}function d(e,t){let{condition:n,extraResources:r}=e.gate.getGate(0);return{condition:{...n,id:`${t}-gate`},extraResources:r}}function f(e,t){return i(e,{id:t.id,operation:t.operation,variant:`iteration-reset`,bindings:[{name:`status`,view:t.state.status,type:`u32`,access:`read_write`},t.state.gate.binding],invocationCount:1,body:`status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${t.state.gate.armWGSL}`})}function p(e,t){return i(e,{id:t.id,operation:t.operation,variant:`iteration-gate`,bindings:[{name:`status`,view:t.state.status,type:`u32`,access:`read_write`},t.state.gate.binding],invocationCount:1,body:`if (status[statusOffset + 1u] == 0u) { return; }
  let changed = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  let iterationCount = status[statusOffset + 2u] + 1u;
  status[statusOffset + 2u] = iterationCount;
  if (changed == 0u) {
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 3u] = 1u;
    ${t.state.gate.disarmWGSL}
    return;
  }
  if (iterationCount >= ${t.maxIterations}u) {
    status[statusOffset + 1u] = 0u;
    ${t.state.gate.disarmWGSL}
  }`})}var m=64;function h(t,n){let{groupSize:r}=n;if(!Number.isSafeInteger(r)||r<1||r>32)throw Error(`${n.operation}: gate group size must be an integer in [1, 32]`);let a=o(t,n.operation,n.contributorId??n.id,`${n.id}-slots`,`uint32`,(r+1)*m);return{groupSize:r,writeNode:i(t,{id:`${n.id}-slots-write`,operation:n.operation,variant:`gate-slot-table`,bindings:[{name:`slotTable`,view:a,type:`u32`,access:`read_write`}],invocationCount:r+1,body:`slotTable[slotTableOffset + index * ${m}u] = index;`}),getSlot:n=>(e(Number.isSafeInteger(n)&&n>=0&&n<=r),t.createDataView(a.buffer,{format:`uint32`,length:1,byteOffset:a.byteOffset+n*m*4}))}}function g(e,t){return i(e,{id:t.id,operation:t.operation,variant:`iteration-group-gate`,bindings:[{name:`status`,view:t.state.status,type:`u32`,access:`read_write`},t.state.gate.binding,{name:`groupSize`,view:t.groupSize,type:`u32`,access:`read`}],invocationCount:1,body:`if (status[statusOffset + 1u] == 0u) { return; }
  let changedBits = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  let size = groupSize[groupSizeOffset];
  var executed = size;
  var converged = false;
  for (var round = 0u; round < size; round++) {
    if (((changedBits >> round) & 1u) == 0u) {
      executed = round + 1u;
      converged = true;
      break;
    }
  }
  let iterationCount = status[statusOffset + 2u] + executed;
  status[statusOffset + 2u] = iterationCount;
  if (converged) {
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 3u] = 1u;
    ${t.state.gate.disarmWGSL}
    return;
  }
  if (iterationCount >= ${t.state.limitWord===void 0?`${t.maxIterations}u`:`status[statusOffset + ${t.state.limitWord}u]`}) {
    status[statusOffset + 1u] = 0u;
    ${t.state.gate.disarmWGSL}
  }`})}function _(e,t,n,r){return e.recording&&(t.annotation={...t.annotation,...n?{loop:t.annotation?.loop??n}:{},attrs:{...t.annotation?.attrs,loopRole:r}}),t}function v(e,t){let n=Math.min(t.groupSize??4,32,Math.max(t.maxIterations,1)),r=t.gateId??`${t.id}-gate`,i=h(e,{id:r,operation:t.operation,groupSize:n}),a=[_(e,i.writeNode,void 0,`setup`)];for(let o=0;o<t.maxIterations;o+=n){let s=Math.min(n,t.maxIterations-o);for(let n=0;n<s;n++){let r=o+n,s=`${t.id}-${r}`,{condition:c,extraResources:l}=d(t.state,s);a.push(_(e,t.createRound({nodeId:s,iteration:r,slotBinding:{name:`roundSlot`,view:i.getSlot(n),type:`u32`,access:`read`},roundSlotWGSL:`roundSlot[roundSlotOffset]`,markChangedWGSL:`atomicOr(&status[statusOffset], 1u << roundSlot[roundSlotOffset]);`,condition:c,extraResources:l}),{id:t.id,iteration:r,phase:`gated`,maximumIterations:t.maxIterations},`round`))}a.push(_(e,g(e,{id:`${r}-${o}`,operation:t.operation,state:t.state,maxIterations:t.maxIterations,groupSize:i.getSlot(s)}),{id:t.id,iteration:o+s-1,phase:`gated`,maximumIterations:t.maxIterations},`gate`))}return a}function y(e,t){let n=[{name:`status`,view:t.state.status,type:`u32`,access:`read`}];return t.converged&&n.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.iterationCount&&n.push({name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`}),i(e,{id:t.id,operation:t.operation,variant:`iteration-finalize`,bindings:n,invocationCount:1,body:`${t.converged?`converged[convergedOffset] = status[statusOffset + 3u];`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = status[statusOffset + 2u];`:``}`})}var b=64,x=4;function S(e){return b+Math.ceil(e.width/16)*Math.ceil(e.height/16)}function C(e,t,n){let r=Math.ceil(t.width/16),a=r*Math.ceil(t.height/16),{continuation:s}=t;if(s)return T(e,t,s,n);let l=u(e,`${t.id}-relax`,t.operation,a*c,t.id),d=o(e,t.operation,t.id,`${t.id}-tile-stamps`,`uint32`,a),p=[f(e,{id:`${t.id}-relax-reset`,operation:t.operation,state:l}),i(e,{id:`${t.id}-tile-reset`,operation:t.operation,variant:`tile-reset`,bindings:[{name:`tileStamps`,view:d,type:`u32`,access:`read_write`}],invocationCount:a,body:`tileStamps[tileStampsOffset + index] = ${n?1:0}u;`})];return{relaxation:{state:l,tileStamps:d,tilesX:r,tileCount:a},resetNodes:p}}var w=`// Resume only a previous execution that stopped unconverged, unless the caller asked to restart.
  let canResume = state[stateOffset + 5u] == 0u &&
    state[stateOffset + 3u] == 0u && state[stateOffset + 2u] > 0u;`;function T(e,t,n,a){let o=Math.ceil(t.width/16),s=o*Math.ceil(t.height/16),{state:l}=n;if(l.length!==S(t))throw Error(`${t.operation} "${t.id}": continuation state has the wrong length`);let d=u(e,`${t.id}-relax`,t.operation,s*c,t.id,{status:l,limitWord:x}),f=e.createDataView(l.buffer,{format:`uint32`,length:s,byteOffset:l.byteOffset+b*4}),p=r(e,{id:`${t.id}-restart`,operation:t.operation,shapes:[[s],...n.restartShapes??[]]}),m=(e,t)=>{let{condition:n,extraResources:r}=p.getGate(e+1);return{condition:{...n,id:`${t}-gate`},extraResources:r}},h=m(-1,`${t.id}-tile-reset`),g=n.prerequisite?[{name:`prerequisite`,view:n.prerequisite,type:`u32`,access:`read`}]:[],_=`let skipStage = ${n.prerequisite?`prerequisite[prerequisiteOffset + 3u] == 0u || `:``}${n.retainConverged?`(state[stateOffset + 3u] != 0u && state[stateOffset + 5u] == 0u)`:`false`};`,v=[i(e,{id:`${t.id}-restart-gate`,operation:t.operation,variant:`continuation-gate`,bindings:[...g,{name:`state`,view:l,type:`u32`,access:`read`},p.binding],invocationCount:1,body:`${w}
  ${_}
  if (canResume || skipStage) {
    ${p.disarmWGSL}
  } else {
    ${p.armWGSL}
  }`}),i(e,{id:`${t.id}-relax-reset`,operation:t.operation,variant:`continuation-arm`,bindings:[...g,{name:`state`,view:l,type:`u32`,access:`read_write`},d.gate.binding],invocationCount:1,body:`${w}
  ${_}
  if (skipStage) { ${d.gate.disarmWGSL} return; }
  state[stateOffset + 5u] = 0u;
  state[stateOffset + 0u] = 0u;
  state[stateOffset + 1u] = 1u;
  state[stateOffset + 3u] = 0u;
  if (!canResume) { state[stateOffset + 2u] = 0u; }
  state[stateOffset + ${x}u] = state[stateOffset + 2u] + ${t.maxIterations}u;
  ${d.gate.armWGSL}`}),i(e,{id:`${t.id}-tile-reset`,operation:t.operation,variant:`tile-reset`,bindings:[{name:`tileStamps`,view:f,type:`u32`,access:`read_write`}],invocationCount:s,body:`tileStamps[tileStampsOffset + index] = ${a?1:0}u;`,condition:h.condition,extraResources:h.extraResources})];return{relaxation:{state:d,tileStamps:f,tilesX:o,tileCount:s,continuation:{getRestartGate:m}},resetNodes:v}}function E(e){return`
fn getRelaxationTile(cell: u32) -> u32 {
  return (cell / ${e}u / 16u) * ${Math.ceil(e/16)}u +
    (cell % ${e}u) / 16u;
}`}function D(e,t){let{relaxation:r}=t,{state:o}=r;if(t.settledAt){if(t.continuation||r.continuation)throw Error(`${t.id} settledAt is not supported with continuation`);if(n(t.settledAt,[`uint32`],`${t.id} settledAt`),t.settledAt.length!==t.width*t.height)throw Error(`${t.id} settledAt must contain one value per cell`)}let s=r.tilesX,l=t.innerIterations??32,u=t.additiveEdgeCosts??!1,d=!!r.continuation,f=e=>d?`status[statusOffset + ${b}u + ${e}]`:`tileStamps[tileStampsOffset + ${e}]`,p=`${a(t)}
const TILE_SIZE: u32 = 16u;
const HALO_SIZE: u32 = 18u;
const HALO_CELL_COUNT: u32 = 324u;
const TILES_X: u32 = ${s}u;
const TILES_Y: u32 = ${Math.ceil(t.height/16)}u;
const TILE_COUNT: u32 = ${r.tileCount}u;
const INNER_ITERATIONS: u32 = ${l}u;
var<workgroup> haloValues: array<f32, 324>;
var<workgroup> haloAuxiliary: array<f32, 324>;
var<workgroup> tileActive: u32;
var<workgroup> tileChanged: u32;
var<workgroup> tileChangedAtomic: atomic<u32>;
${d?`var<workgroup> roundPhase: u32;`:``}
${t.declarations}`,m=`
fn relaxTile(tileIndex: u32, localInvocationIndex: u32) {
  let tileColumn = tileIndex % TILES_X;
  let tileRow = tileIndex / TILES_X;
  // The gate adds each group's executed iterations to status[2]; roundSlot is this round's place in the group.
  let roundSlotValue = roundSlot[roundSlotOffset];
  let iteration = atomicLoad(&status[statusOffset + 2u]) + roundSlotValue;
  if (localInvocationIndex == 0u) {
    var anyActive = 0u;
    for (var rowDelta = -1; rowDelta <= 1; rowDelta++) {
      for (var columnDelta = -1; columnDelta <= 1; columnDelta++) {
        let neighborColumn = i32(tileColumn) + columnDelta;
        let neighborRow = i32(tileRow) + rowDelta;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(TILES_X) || neighborRow >= i32(TILES_Y)) {
          continue;
        }
        let neighborTile = u32(neighborRow) * TILES_X + u32(neighborColumn);
        if (atomicLoad(&${f(`neighborTile`)}) >= iteration + 1u) {
          anyActive = 1u;
        }
      }
    }
    tileActive = anyActive;
    atomicStore(&tileChangedAtomic, 0u);
  }
  if (workgroupUniformLoad(&tileActive) == 0u) { return; }

  let originColumn = i32(tileColumn * TILE_SIZE) - 1;
  let originRow = i32(tileRow * TILE_SIZE) - 1;
  for (var haloIndex = localInvocationIndex; haloIndex < HALO_CELL_COUNT; haloIndex += ${c}u) {
    let column = originColumn + i32(haloIndex % HALO_SIZE);
    let row = originRow + i32(haloIndex / HALO_SIZE);
    var haloValue = getInfinity();
    var haloAux = getQuietNaN();
    if (column >= 0 && row >= 0 && column < i32(GRID_WIDTH) && row < i32(GRID_HEIGHT)) {
      let cell = u32(row) * GRID_WIDTH + u32(column);
      haloValue = bitcast<f32>(atomicLoad(&values[valuesOffset + cell]));
      haloAux = auxiliary[auxiliaryOffset + cell];
    }
    haloValues[haloIndex] = haloValue;
    haloAuxiliary[haloIndex] = haloAux;
  }
  workgroupBarrier();

  let localColumn = localInvocationIndex % TILE_SIZE;
  let localRow = localInvocationIndex / TILE_SIZE;
  let column = tileColumn * TILE_SIZE + localColumn;
  let row = tileRow * TILE_SIZE + localRow;
  let haloCenter = (localRow + 1u) * HALO_SIZE + localColumn + 1u;
  let centerAuxiliary = haloAuxiliary[haloCenter];
  let originalValue = haloValues[haloCenter];
  let participates = column < GRID_WIDTH && row < GRID_HEIGHT && !isNaNValue(centerAuxiliary);

  ${u?`// Loop-invariant: the edge costs depend only on the auxiliary values, never on the relaxed values.
  var edgeCosts: array<f32, 8>;
  if (participates) {
    for (var direction = 0u; direction < 8u; direction++) {
      let neighborHalo = u32(i32(haloCenter) + getD8RowOffset(direction) * i32(HALO_SIZE) + getD8ColumnOffset(direction));
      let neighborAuxiliary = haloAuxiliary[neighborHalo];
      edgeCosts[direction] = select(getRelaxationEdgeCost(neighborAuxiliary, centerAuxiliary, row, direction), getInfinity(), isNaNValue(neighborAuxiliary));
    }
  }
  let relaxationLimit = getRelaxationLimit();`:``}
  for (var inner = 0u; inner < INNER_ITERATIONS; inner++) {
    var best = haloValues[haloCenter];
    if (participates) {
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborHalo = u32(i32(haloCenter) + getD8RowOffset(direction) * i32(HALO_SIZE) + getD8ColumnOffset(direction));
        ${u?`let candidate = haloValues[neighborHalo] + edgeCosts[direction];
        if (candidate <= relaxationLimit && candidate < best) { best = candidate; }`:`let neighborAuxiliary = haloAuxiliary[neighborHalo];
        if (isNaNValue(neighborAuxiliary)) { continue; }
        let candidate = getRelaxationCandidate(haloValues[neighborHalo], neighborAuxiliary, centerAuxiliary, row, direction);
        if (candidate < best) { best = candidate; }`}
      }
    }
    workgroupBarrier();
    if (best < haloValues[haloCenter]) {
      haloValues[haloCenter] = best;
      atomicStore(&tileChangedAtomic, 1u);
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      tileChanged = atomicExchange(&tileChangedAtomic, 0u);
    }
    if (workgroupUniformLoad(&tileChanged) == 0u) { break; }
  }

  let finalValue = haloValues[haloCenter];
  if (participates && finalValue < originalValue) {
    atomicStore(&values[valuesOffset + row * GRID_WIDTH + column], bitcast<u32>(finalValue));${t.settledAt?`
    settledAt[settledAtOffset + row * GRID_WIDTH + column] = iteration + 1u;`:``}
    atomicStore(&tileChangedAtomic, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&tileChangedAtomic) != 0u) {
    atomicMax(&${f(`tileIndex`)}, iteration + 2u);
    atomicOr(&status[statusOffset], 1u << roundSlotValue);
  }
}`,h=(e,n)=>`
  let lineCount = ${e?r.tilesX:Math.ceil(t.height/16)}u;
  let stepCount = ${e?Math.ceil(t.height/16):r.tilesX}u;
  if (workgroupIndex >= lineCount) { return; }
  for (var step = 0u; step < stepCount; step++) {
    let position = ${n?`stepCount - 1u - step`:`step`};
    let tileIndex = ${e?`position * TILES_X + workgroupIndex`:`workgroupIndex * TILES_X + position`};
    relaxTile(tileIndex, localInvocationIndex);
    // Publish this tile's cells to the next tile's halo load within the workgroup.
    storageBarrier();
    workgroupBarrier();
  }`,g=t.sweepAfterIteration,_=`
  if (localInvocationIndex == 0u) {
    let iteration = atomicLoad(&status[statusOffset + 2u]) + roundSlot[roundSlotOffset];
    roundPhase = ${g===void 0?`0u`:`select(0u, (iteration - ${g}u) % 5u, iteration >= ${g}u)`};
  }
  // 0 tiled; 1 right, 2 left, 3 down, 4 up.
  let phase = workgroupUniformLoad(&roundPhase);
  if (phase == 0u) {
    if (workgroupIndex >= TILE_COUNT) { return; }
    relaxTile(workgroupIndex, localInvocationIndex);
    return;
  }
  let vertical = phase >= 3u;
  let reverse = (phase & 1u) == 0u;
  let lineCount = select(TILES_Y, TILES_X, vertical);
  let stepCount = select(TILES_X, TILES_Y, vertical);
  if (workgroupIndex >= lineCount) { return; }
  for (var step = 0u; step < stepCount; step++) {
    let position = select(step, stepCount - 1u - step, reverse);
    let tileIndex = select(workgroupIndex * TILES_X + position, position * TILES_X + workgroupIndex, vertical);
    relaxTile(tileIndex, localInvocationIndex);
    // Publish this tile's cells to the next tile's halo load within the workgroup.
    storageBarrier();
    workgroupBarrier();
  }`,y=[],x=t.settledAt?[{name:`settledAt`,view:t.settledAt,type:`u32`,access:`read_write`}]:[];t.settledAt&&y.push(i(e,{id:`${t.id}-settled-at-reset`,operation:t.operation,variant:`settled-at-reset`,bindings:x,invocationCount:t.width*t.height,body:`settledAt[settledAtOffset + index] = 0u;`}));let S=[{variant:`sweep-right`,body:h(!1,!1)},{variant:`sweep-left`,body:h(!1,!0)},{variant:`sweep-down`,body:h(!0,!1)},{variant:`sweep-up`,body:h(!0,!0)}];return y.push(...v(e,{id:`${t.id}-relax`,gateId:`${t.id}-relax-gate`,operation:t.operation,state:o,maxIterations:t.maxIterations,groupSize:t.gateInterval,createRound:({nodeId:n,iteration:a,slotBinding:s,condition:l,extraResources:u})=>{let f=t.sweepAfterIteration===void 0||a<t.sweepAfterIteration?0:(a-t.sweepAfterIteration)%(S.length+1),h=d?{variant:`continued-relax`,body:_}:f===0?{variant:`tiled-relax`,body:`
  if (workgroupIndex >= TILE_COUNT) { return; }
  relaxTile(workgroupIndex, localInvocationIndex);`}:S[f-1];return i(e,{id:n,operation:t.operation,variant:h.variant,bindings:[{name:`values`,view:t.values,type:`atomic<u32>`,access:`read_write`},{name:`auxiliary`,view:t.auxiliary,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},...d?[]:[{name:`tileStamps`,view:r.tileStamps,type:`atomic<u32>`,access:`read_write`}],{name:`status`,view:o.status,type:`atomic<u32>`,access:`read_write`},s,...x],invocationCount:o.invocationCount,workgroupSize:c,guardIndex:!1,declarations:`${p}\n${m}`,body:h.body,condition:l,extraResources:u})}})),y}export{g as a,C as c,d,E as f,p as i,D as l,v as n,f as o,l as p,y as r,u as s,h as t,S as u};