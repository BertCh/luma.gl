function e(e){let{counters:t}=e,n=e=>e.operation===`min`?`0xffffffffu`:`0u`,r=t.map((e,t)=>`atomicStore(&workgroupCounters[${t}], ${n(e)});`).join(`
    `),i=t.map((e,t)=>{let r=e.operation===`min`?`atomicMin`:`atomicAdd`;return`let total${t} = atomicLoad(&workgroupCounters[${t}]);
    if (total${t} != ${n(e)}) {
      ${r}(&${e.target}[${e.target}Offset + ${e.word}u], total${t});
    }`}).join(`
    `);return{declarations:`
var<workgroup> workgroupCounters: array<atomic<u32>, ${t.length}>;
fn addCounter(counter: u32, value: u32) {
  if (value > 0u) {
    atomicAdd(&workgroupCounters[counter], value);
  }
}
fn minCounter(counter: u32, value: u32) {
  atomicMin(&workgroupCounters[counter], value);
}
fn processRow(index: u32) {
  ${e.rowSource}
}`,body:`if (localInvocationIndex == 0u) {
    ${r}
  }
  workgroupBarrier();
  if (index < ${e.rowCount}) {
    processRow(index);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    ${i}
  }`}}export{e as t};