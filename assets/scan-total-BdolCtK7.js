import{J as e}from"./src-CNCKFW2b.js";import{s as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var n=4294967295;function r(r,i){return e(i.rowCount<=i.counts.length&&i.rowCount<=i.offsets.length&&(i.totalRow??0)<i.total.length),t(r,{id:`${i.id}-carry`,operation:i.operation,variant:`scan-carry`,bindings:[{name:`counts`,view:i.counts,type:`u32`,access:`read`},{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`total`,view:i.total,type:`atomic<u32>`,access:`read_write`}],invocationCount:i.rowCount,declarations:`override TOTAL_ROW: u32;`,constants:{TOTAL_ROW:i.totalRow??0},body:`let offset = offsets[offsetsOffset + index];
  if (offset + counts[countsOffset + index] < offset) {
    atomicStore(&total[totalOffset + TOTAL_ROW], ${n}u);
  }`})}function i(e,n){return[t(e,{id:`${n.id}-total`,operation:n.operation,variant:`scan-total`,bindings:[{name:`counts`,view:n.counts,type:`u32`,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`total`,view:n.total,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`override ROW_COUNT: u32;
override TOTAL_ROW: u32;`,constants:{ROW_COUNT:n.rowCount,TOTAL_ROW:n.totalRow??0},body:`var sum = 0u;
  if (ROW_COUNT > 0u) {
    sum = offsets[offsetsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  }
  total[totalOffset + TOTAL_ROW] = sum;`}),r(e,n)]}export{r as n,i as t};