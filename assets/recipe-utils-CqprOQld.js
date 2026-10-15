import{w as e}from"./resources-CwvhEOBe.js";import{t}from"./gpu-scan-BuGhTLDQ.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{n as r}from"./spatial-weights-D4bd9AQ6.js";import{n as i,t as a}from"./gpu-cell-topology-6n6YnUDd.js";var o=class{contributors=[];constructor(e){this.graph=e}add(e){return this.graph.add(e),this.contributors.push(e),e}};function s(t,n,r,i,a){return a??e(t,n,r,i)}function c(e,t,n,i,a){let o=r({rowCount:n,capacity:i}).capacity;return{offsets:s(e,`${t}-offsets`,`uint32`,n+1,a?.offsets),neighbors:s(e,`${t}-neighbors`,`uint32`,o,a?.neighbors),weights:s(e,`${t}-weights`,`float32`,o,a?.weights),...a?.distances?{distances:a.distances}:{},status:{overflow:s(e,`${t}-overflow`,`uint32`,1,a?.status?.overflow),requiredCount:s(e,`${t}-required-count`,`uint32`,1,a?.status?.requiredCount)}}}function l(t,r,i){let a=e(t,`${r}-offsets`,`uint32`,i+1);return t.add(n(t,{id:`${r}-offsets`,operation:`GPURecipeIdentityOffsets`,bindings:[{name:`offsets`,view:a,type:`u32`,access:`read_write`}],invocationCount:i+1,body:`offsets[offsetsOffset + index] = index;`})),a}function u(e,t,n){if(!t)throw Error(`${e} ${n}`)}function d(e,t,n){return`fn ${e}(key: vec2u) -> u32 {
  let rowCount = ${n};
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let middleLow = ${t}[${t}Offset + 2u * middle];
    let middleHigh = ${t}[${t}Offset + 2u * middle + 1u];
    if (middleHigh < key.x || (middleHigh == key.x && middleLow < key.y)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowCount && ${t}[${t}Offset + 2u * low] == key.y && ${t}[${t}Offset + 2u * low + 1u] == key.x) {
    return low;
  }
  return 0xffffffffu;
}`}function f(r,o,s,c){let{cells:l,weights:u}=c,f=l.length,p=u.neighbors.length,m=o.add(new a({id:`${s}-disk`,family:c.family,operation:{type:`disk`,k:c.rings},cells:l,mask:c.mask,output:{cells:e(r,`${s}-disk-cells`,`uint32x2`,f*i(c.family,{type:`disk`,k:c.rings}))}})),h=m.stride,g=m.props.output.cells,_=e(r,`${s}-slot-counts`,`uint32`,f),v=e(r,`${s}-slot-ends`,`uint32`,f),y=`const STRIDE: u32 = ${h}u;
const ROWS: u32 = ${f}u;
const CAPACITY: u32 = ${p}u;
const NOT_FOUND: u32 = 0xffffffffu;`,b=`${y}
${d(`findStudyCell`,`cells`,`countScalar[countScalarOffset]`)}`,x=`let entryLow = disk[diskOffset + 2u * (index * STRIDE + entry)];
    let entryHigh = disk[diskOffset + 2u * (index * STRIDE + entry) + 1u];
    if (entryLow == 0u && entryHigh == 0u) {
      continue;
    }
    let found = findStudyCell(vec2u(entryHigh, entryLow));
    if (found == NOT_FOUND || found == index) {
      continue;
    }`;return r.add(n(r,{id:`${s}-slot-counts`,operation:`GPUCellDiskWeights`,variant:`count`,bindings:[{name:`disk`,view:g,type:`u32`,access:`read`},{name:`cells`,view:l,type:`u32`,access:`read`},{name:`countScalar`,view:c.count,type:`u32`,access:`read`},{name:`slotCounts`,view:_,type:`u32`,access:`read_write`}],invocationCount:f,declarations:b,body:`var count = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${x}
    count++;
  }
  slotCounts[slotCountsOffset + index] = count;`})),r.add(new t({id:`${s}-slot-scan`,input:_,output:v,mode:`inclusive`})),r.add(n(r,{id:`${s}-offsets`,operation:`GPUCellDiskWeights`,variant:`offsets`,bindings:[{name:`slotEnds`,view:v,type:`u32`,access:`read`},{name:`rowOffsets`,view:u.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:c.overflow,type:`u32`,access:`read_write`},...c.requiredCount?[{name:`requiredCount`,view:c.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:f+1,declarations:y,body:`var rowOffset = 0u;
  if (index > 0u) {
    rowOffset = min(slotEnds[slotEndsOffset + index - 1u], CAPACITY);
  }
  rowOffsets[rowOffsetsOffset + index] = rowOffset;
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, slotEnds[slotEndsOffset + ROWS - 1u] > CAPACITY);
    ${c.requiredCount?`requiredCount[requiredCountOffset] = slotEnds[slotEndsOffset + ROWS - 1u];`:``}
  }`})),r.add(n(r,{id:`${s}-write`,operation:`GPUCellDiskWeights`,variant:`write`,bindings:[{name:`disk`,view:g,type:`u32`,access:`read`},{name:`cells`,view:l,type:`u32`,access:`read`},{name:`countScalar`,view:c.count,type:`u32`,access:`read`},{name:`slotEnds`,view:v,type:`u32`,access:`read`},{name:`neighbors`,view:u.neighbors,type:`u32`,access:`read_write`},{name:`weightValues`,view:u.weights,type:`f32`,access:`read_write`}],invocationCount:f,declarations:b,body:`var start = 0u;
  if (index > 0u) {
    start = slotEnds[slotEndsOffset + index - 1u];
  }
  var written = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${x}
    let slot = start + written;
    if (slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = found;
      weightValues[weightValuesOffset + slot] = 1.0;
    }
    written++;
  }
  // Ascending neighbor index within the row (insertion sort of the slots that fit).
  let stored = select(0u, min(written, CAPACITY - start), start < CAPACITY);
  for (var i = 1u; i < stored; i++) {
    let value = neighbors[neighborsOffset + start + i];
    var j = i;
    loop {
      if (j == 0u || neighbors[neighborsOffset + start + j - 1u] <= value) {
        break;
      }
      neighbors[neighborsOffset + start + j] = neighbors[neighborsOffset + start + j - 1u];
      j--;
    }
    neighbors[neighborsOffset + start + j] = value;
  }`})),h}export{c as a,l as i,f as n,d as o,u as r,s,o as t};