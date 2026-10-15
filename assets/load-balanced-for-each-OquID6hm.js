import{J as e}from"./src-CNCKFW2b.js";import{w as t}from"./resources-CwvhEOBe.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var r=256*4;function i(e,t){return Math.max(1,Math.ceil((e+t)/r))}function a(r,i){let a=c(i),{accumulatorType:d}=i,f=o(d),p=d.includes(`f32`)?`f32`:`u32`,m=i.offsets.length-1;e(i.output.length>=m*a.laneCount*f);let h=1+f,g=t(r,`${i.id}-tile-carries`,`uint32`,a.workgroupCount*h),_=(e,t)=>f===1?e:`${e}[${t}]`,v=e=>p===`f32`?`bitcast<u32>(${e})`:e,y=e=>p===`f32`?`bitcast<f32>(${e})`:e,b=(e,t,n)=>Array.from({length:f},(r,i)=>`loadBalancedOutput[loadBalancedOutputOffset + (${e} * LOAD_BALANCED_LANE_COUNT + loadBalancedLane) * ${f}u + ${i}u] ${n} ${_(t,i)};`).join(`
      `),x=e=>f===1?y(`loadBalancedCarries[loadBalancedCarriesOffset + ${e} * ${h}u + 1u]`):`${d}(${Array.from({length:f},(t,n)=>y(`loadBalancedCarries[loadBalancedCarriesOffset + ${e} * ${h}u + ${n+1}u]`)).join(`, `)})`,S={name:`loadBalancedOutput`,view:i.output,type:p,access:`read_write`},C={name:`loadBalancedCarries`,view:g,type:`u32`,access:`read_write`};return[n(r,{id:i.id,operation:i.operation,variant:i.variant??`load-balanced-segment-sum`,bindings:[s(i),S,C,...i.bindings],workgroupSize:256,invocationCount:a.workgroupCount*256,guardIndex:!1,declarations:`${l()}
var<workgroup> loadBalancedCarryRows: array<u32, 256>;
var<workgroup> loadBalancedCarryValues: array<${d}, 256>;
${i.declarations}`,constants:a.constants,body:`var loadBalancedRunning = ${d}();
  var loadBalancedFirstValue = ${d}();
  var loadBalancedHasEmitted = false;
  ${u({consume:`loadBalancedRunning = loadBalancedRunning +
          getLoadBalancedContribution(loadBalancedRow, loadBalancedSlot, loadBalancedLane);`,emit:`if (loadBalancedHasEmitted) {
            ${b(`loadBalancedRow`,`loadBalancedRunning`,`=`)}
          } else {
            loadBalancedFirstValue = loadBalancedRunning;
            loadBalancedHasEmitted = true;
          }
          loadBalancedRunning = ${d}();`})}
  // Inclusive segmented scan of the thread runs, keyed by row (rows never decrease by thread).
  loadBalancedCarryRows[localInvocationIndex] = loadBalancedRow;
  loadBalancedCarryValues[localInvocationIndex] = loadBalancedRunning;
  workgroupBarrier();
  for (var stride = 1u; stride < 256u; stride = stride * 2u) {
    var previousRow = 0xffffffffu;
    var previousValue = ${d}();
    if (localInvocationIndex >= stride) {
      previousRow = loadBalancedCarryRows[localInvocationIndex - stride];
      previousValue = loadBalancedCarryValues[localInvocationIndex - stride];
    }
    workgroupBarrier();
    if (previousRow == loadBalancedCarryRows[localInvocationIndex]) {
      loadBalancedCarryValues[localInvocationIndex] =
        previousValue + loadBalancedCarryValues[localInvocationIndex];
    }
    workgroupBarrier();
  }
  if (loadBalancedIsActive) {
    if (loadBalancedHasEmitted) {
      // Thread t starts in the row where thread t - 1 stopped; its scan holds the earlier runs.
      var carryIn = ${d}();
      if (localInvocationIndex > 0u &&
          loadBalancedCarryRows[localInvocationIndex - 1u] == loadBalancedStartRow) {
        carryIn = loadBalancedCarryValues[localInvocationIndex - 1u];
      }
      let rowValue = carryIn + loadBalancedFirstValue;
      ${b(`loadBalancedStartRow`,`rowValue`,`=`)}
    }
    if (localInvocationIndex == 255u) {
      let carryRow = loadBalancedCarryRows[localInvocationIndex];
      let carryValue = loadBalancedCarryValues[localInvocationIndex];
      let record = loadBalancedCarriesOffset + workgroupIndex * ${h}u;
      loadBalancedCarries[record] = select(LOAD_BALANCED_ROW_COUNT, carryRow, loadBalancedTileLength > 0u);
      ${Array.from({length:f},(e,t)=>`loadBalancedCarries[record + ${t+1}u] = ${v(_(`carryValue`,t))};`).join(`
      `)}
    }
  }`}),n(r,{id:`${i.id}-fixup`,operation:i.operation,variant:`load-balanced-segment-sum-fixup`,bindings:[C,S],invocationCount:a.workgroupCount,workgroupSize:64,declarations:`override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;`,constants:{LOAD_BALANCED_ROW_COUNT:m,LOAD_BALANCED_LANE_COUNT:a.laneCount,LOAD_BALANCED_TILES_PER_LANE:a.tileCount},body:`let loadBalancedLane = index / LOAD_BALANCED_TILES_PER_LANE;
  let tile = index % LOAD_BALANCED_TILES_PER_LANE;
  let row = loadBalancedCarries[loadBalancedCarriesOffset + index * ${h}u];
  if (row >= LOAD_BALANCED_ROW_COUNT) {
    return;
  }
  // Only the first tile of a run of tiles carrying the same row adds the run, in tile order.
  if (tile > 0u && loadBalancedCarries[loadBalancedCarriesOffset + (index - 1u) * ${h}u] == row) {
    return;
  }
  var sum = ${x(`index`)};
  let laneEnd = (loadBalancedLane + 1u) * LOAD_BALANCED_TILES_PER_LANE;
  var next = index + 1u;
  loop {
    if (next >= laneEnd ||
        loadBalancedCarries[loadBalancedCarriesOffset + next * ${h}u] != row) {
      break;
    }
    sum = sum + ${x(`next`)};
    next = next + 1u;
  }
  ${b(`row`,`sum`,`+=`)}`})]}function o(e){return e.startsWith(`vec4`)?4:e.startsWith(`vec2`)?2:1}function s(e){return{name:`loadBalancedOffsets`,view:e.offsets,type:`u32`,access:`read`}}function c(e){let t=e.offsets.length-1,n=e.laneCount??1;if(t<0||!Number.isSafeInteger(e.itemCapacity)||e.itemCapacity<0||!Number.isSafeInteger(n)||n<1||n*(t+e.itemCapacity)>4294967295)throw Error(`${e.operation}: invalid load-balanced topology for ${e.id}`);let r=i(t,e.itemCapacity),a=n*r;if(a*256>4294967295)throw Error(`${e.operation}: load-balanced dispatch of ${e.id} is too large`);return{laneCount:n,tileCount:r,workgroupCount:a,constants:{LOAD_BALANCED_ROW_COUNT:t,LOAD_BALANCED_ITEM_CAPACITY:e.itemCapacity,LOAD_BALANCED_LANE_COUNT:n,LOAD_BALANCED_TILES_PER_LANE:r}}}function l(){return`override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_ITEM_CAPACITY: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;
var<workgroup> loadBalancedTileRowEnds: array<u32, ${r+1}>;
var<workgroup> loadBalancedTileBounds: array<u32, 2>;
/** End of row \`row\` relative to \`itemBase\`, clamped to the items \`[itemBase, itemLimit)\`. */
fn getLoadBalancedRowEnd(row: u32, itemBase: u32, itemLimit: u32) -> u32 {
  return clamp(loadBalancedOffsets[loadBalancedOffsetsOffset + row + 1u], itemBase, itemLimit) - itemBase;
}
/** Rows consumed at merge step \`diagonal\` (Merrill-Garland merge-path search). */
fn searchLoadBalancedPath(diagonal: u32, itemBase: u32, itemLimit: u32) -> u32 {
  let itemCount = itemLimit - itemBase;
  var low = select(0u, diagonal - itemCount, diagonal > itemCount);
  var high = min(diagonal, LOAD_BALANCED_ROW_COUNT);
  loop {
    if (low >= high) { break; }
    let pivot = (low + high) / 2u;
    if (getLoadBalancedRowEnd(pivot, itemBase, itemLimit) <= diagonal - pivot - 1u) {
      low = pivot + 1u;
    } else {
      high = pivot;
    }
  }
  return low;
}`}function u(e){return`let loadBalancedIsActive =
    workgroupIndex < LOAD_BALANCED_LANE_COUNT * LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedLane = workgroupIndex / LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedTile = workgroupIndex % LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedItemBase = min(
    loadBalancedOffsets[loadBalancedOffsetsOffset],
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedItemLimit = clamp(
    loadBalancedOffsets[loadBalancedOffsetsOffset + LOAD_BALANCED_ROW_COUNT],
    loadBalancedItemBase,
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedMergeCount =
    LOAD_BALANCED_ROW_COUNT + loadBalancedItemLimit - loadBalancedItemBase;
  let loadBalancedTileBegin = select(
    loadBalancedMergeCount,
    min(loadBalancedTile * ${r}u, loadBalancedMergeCount),
    loadBalancedIsActive
  );
  let loadBalancedTileEnd = min(loadBalancedTileBegin + ${r}u, loadBalancedMergeCount);
  let loadBalancedTileLength = loadBalancedTileEnd - loadBalancedTileBegin;
  if (localInvocationIndex < 2u) {
    loadBalancedTileBounds[localInvocationIndex] = searchLoadBalancedPath(
      select(loadBalancedTileBegin, loadBalancedTileEnd, localInvocationIndex == 1u),
      loadBalancedItemBase,
      loadBalancedItemLimit
    );
  }
  workgroupBarrier();
  let loadBalancedTileRowBegin = loadBalancedTileBounds[0];
  let loadBalancedTileRowCount = loadBalancedTileBounds[1] - loadBalancedTileRowBegin;
  // Item coordinates are relative to loadBalancedItemBase; rows at or past ROW_COUNT never end.
  let loadBalancedTileItemBegin = loadBalancedTileBegin - loadBalancedTileRowBegin;
  let loadBalancedTileItemCount =
    loadBalancedTileEnd - loadBalancedTileBounds[1] - loadBalancedTileItemBegin;
  for (var entry = localInvocationIndex; entry <= loadBalancedTileRowCount; entry += 256u) {
    let entryRow = loadBalancedTileRowBegin + entry;
    var rowEnd = 0xffffffffu;
    if (entryRow < LOAD_BALANCED_ROW_COUNT) {
      rowEnd = getLoadBalancedRowEnd(entryRow, loadBalancedItemBase, loadBalancedItemLimit);
    }
    loadBalancedTileRowEnds[entry] = rowEnd;
  }
  workgroupBarrier();
  let loadBalancedDiagonal = min(
    localInvocationIndex * 4u,
    loadBalancedTileLength
  );
  var loadBalancedLow = select(
    0u,
    loadBalancedDiagonal - loadBalancedTileItemCount,
    loadBalancedDiagonal > loadBalancedTileItemCount
  );
  var loadBalancedHigh = min(loadBalancedDiagonal, loadBalancedTileRowCount);
  loop {
    if (loadBalancedLow >= loadBalancedHigh) { break; }
    let pivot = (loadBalancedLow + loadBalancedHigh) / 2u;
    if (loadBalancedTileRowEnds[pivot] <= loadBalancedTileItemBegin + loadBalancedDiagonal - pivot - 1u) {
      loadBalancedLow = pivot + 1u;
    } else {
      loadBalancedHigh = pivot;
    }
  }
  var loadBalancedRow = loadBalancedTileRowBegin + loadBalancedLow;
  let loadBalancedStartRow = loadBalancedRow;
  var loadBalancedItem = loadBalancedTileItemBegin + loadBalancedDiagonal - loadBalancedLow;
  let loadBalancedStepCount = min(
    4u,
    loadBalancedTileLength - loadBalancedDiagonal
  );
  for (var walkStep = 0u; walkStep < 4u; walkStep = walkStep + 1u) {
    if (walkStep < loadBalancedStepCount) {
      if (loadBalancedItem < loadBalancedTileRowEnds[loadBalancedRow - loadBalancedTileRowBegin]) {
        let loadBalancedSlot = loadBalancedItemBase + loadBalancedItem;
        ${e.consume}
        loadBalancedItem = loadBalancedItem + 1u;
      } else {
        ${e.emit}
        loadBalancedRow = loadBalancedRow + 1u;
      }
    }
  }`}export{a as t};