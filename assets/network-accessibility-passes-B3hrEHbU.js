import{s as e}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as t}from"./wgsl-common-BNRVzgd8.js";var n=4294967295,r=`
${t}
fn isWithinThreshold(cost: f32, threshold: f32) -> bool {
  // Unreached entries are +Infinity; an infinite threshold must still reject them.
  return (bitcast<u32>(cost) & 0x7f800000u) != 0x7f800000u && cost >= 0.0 && cost <= threshold;
}
fn getDecay(cost: f32, decay: u32, beta: f32, minimumCost: f32) -> f32 {
  if (decay == 1u) {
    return exp(-beta * cost);
  }
  if (decay == 2u) {
    return pow(max(cost, minimumCost), -beta);
  }
  return 1.0;
}`,i=`let threshold = parameters[parametersOffset];
  let decay = u32(max(parameters[parametersOffset + 1u], 0.0));
  let beta = parameters[parametersOffset + 2u];
  let minimumCost = parameters[parametersOffset + 3u];`;function a(t,n){let a=[{name:`costs`,view:n.costs,type:`f32`,access:`read`},{name:`rowWeights`,view:n.rowWeights,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];return n.cumulative&&a.push({name:`cumulative`,view:n.cumulative,type:`f32`,access:`read_write`}),n.gravity&&a.push({name:`gravity`,view:n.gravity,type:`f32`,access:`read_write`}),e(t,{id:n.id,operation:n.operation,variant:`column-gather`,record:{datum:{keySpace:`node`},formula:`gpu-network.accessibility.gravity`},bindings:a,invocationCount:n.columnCount,declarations:`const ROW_COUNT: u32 = ${n.rowCount}u;
const COLUMN_COUNT: u32 = ${n.columnCount}u;
${r}`,body:`${i}
  var cumulativeSum = 0.0;
  var gravitySum = 0.0;
  for (var row = 0u; row < ROW_COUNT; row++) {
    let cost = costs[costsOffset + row * COLUMN_COUNT + index];
    if (!isWithinThreshold(cost, threshold)) {
      continue;
    }
    let weight = rowWeights[rowWeightsOffset + row];
    if (!isFiniteBits(weight)) {
      continue;
    }
    cumulativeSum += weight;
    gravitySum += weight * getDecay(cost, decay, beta, minimumCost);
  }
  ${n.cumulative?`cumulative[cumulativeOffset + index] = cumulativeSum;`:``}
  ${n.gravity?`gravity[gravityOffset + index] = gravitySum;`:``}`})}function o(t,n){let a=[{name:`costs`,view:n.costs,type:`f32`,access:`read`},{name:`columnWeights`,view:n.columnWeights,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];return n.cumulative&&a.push({name:`cumulative`,view:n.cumulative,type:`f32`,access:`read_write`}),n.gravity&&a.push({name:`gravity`,view:n.gravity,type:`f32`,access:`read_write`}),e(t,{id:n.id,operation:n.operation,variant:n.variant,bindings:a,workgroupSize:256,invocationCount:n.rowCount*256,guardIndex:!1,declarations:`const COLUMN_COUNT: u32 = ${n.columnCount}u;
var<workgroup> cumulativePartials: array<f32, 256>;
var<workgroup> gravityPartials: array<f32, 256>;
${r}`,body:`let row = index / 256u;
  let isInRange = index < INVOCATION_COUNT;
  ${i}
  var cumulativeSum = 0.0;
  var gravitySum = 0.0;
  if (isInRange) {
    for (var column = localInvocationIndex; column < COLUMN_COUNT; column += 256u) {
      let cost = costs[costsOffset + row * COLUMN_COUNT + column];
      let weight = columnWeights[columnWeightsOffset + column];
      if (isWithinThreshold(cost, threshold) && isFiniteBits(weight)) {
        cumulativeSum += weight;
        gravitySum += weight * getDecay(cost, decay, beta, minimumCost);
      }
    }
  }
  cumulativePartials[localInvocationIndex] = cumulativeSum;
  gravityPartials[localInvocationIndex] = gravitySum;
  workgroupBarrier();
  for (var stride = ${256/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      cumulativePartials[localInvocationIndex] += cumulativePartials[localInvocationIndex + stride];
      gravityPartials[localInvocationIndex] += gravityPartials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    ${n.cumulative?`cumulative[cumulativeOffset + row] = cumulativePartials[0];`:``}
    ${n.gravity?`gravity[gravityOffset + row] = gravityPartials[0];`:``}
  }`})}export{a as n,o as r,n as t};