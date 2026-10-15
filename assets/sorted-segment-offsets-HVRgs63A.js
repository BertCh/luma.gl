import{s as e}from"./wgsl-kernel-nodes-Bm5c4bxp.js";function t(t,n){let{segmentCount:r,sortedKeys:i,segmentOffsets:a}=n;return e(t,{id:n.id,operation:n.operation,variant:n.variant??`segment-offsets`,bindings:[{name:`sortedKeys`,view:i,type:`u32`,access:`read`},{name:`segmentOffsets`,view:a,type:`u32`,access:`read_write`}],invocationCount:r+1,body:`var low = 0u;
  var high = ${i.length}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (sortedKeys[sortedKeysOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  segmentOffsets[segmentOffsetsOffset + index] = low;`})}function n(n,r){let{id:i,operation:a,groupCount:o,sortedGroupKeys:s,segmentOffsets:c,counts:l}=r,u=[t(n,{id:`${i}-segment-offsets`,operation:a,segmentCount:o,sortedKeys:s,segmentOffsets:c})];return l&&u.push(e(n,{id:`${i}-group-counts`,operation:a,variant:`group-counts`,bindings:[{name:`segmentOffsets`,view:c,type:`u32`,access:`read`},{name:`counts`,view:l,type:`u32`,access:`read_write`}],invocationCount:o,body:`counts[countsOffset + index] =
    segmentOffsets[segmentOffsetsOffset + index + 1u] - segmentOffsets[segmentOffsetsOffset + index];`})),u}export{n,t};