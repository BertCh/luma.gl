import{t as e}from"./earth-xB16XgAa.js";import{_ as t,d as n,l as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-group-aggregation-DrNKSCi8.js";import{n as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l}from"./wgsl-common-BNRVzgd8.js";import{o as u}from"./sorted-segment-sums-DwGkmbx2.js";import{r as d}from"./spherical-cell-area-m1npYaRX.js";import{a as f,n as p,o as m,r as h,s as g,u as _}from"./raster-band-utils-CA-sh8xt.js";var v=4;function y(e){return r(e,4)}function b(e){return Math.min(16777215,r(e,4,1))}var x=[[`cellCounts`,`count`],[`valueCounts`,`count`],[`sums`,`sum`],[`means`,`mean`],[`minimums`,`min`],[`maximums`,`max`]],S=[`weightSums`,`variances`,`standardDeviations`];function C(t,n){return`const ZONAL_GRID_WIDTH: u32 = ${t}u;
const ZONAL_GRID_HEIGHT: f32 = ${n}.0;
var<private> extent: array<f32, 4>;
const extentOffset: u32 = 0u;
${d(e)}
fn getCellWeight(cell: u32) -> f32 {
  let northEdge = settings[settingsOffset + 2u];
  // Signed row height: negative when rows advance south, which makes getCellArea negative.
  let rowHeight = (settings[settingsOffset + 3u] - northEdge) / ZONAL_GRID_HEIGHT;
  extent = array<f32, 4>(0.0, northEdge, settings[settingsOffset], rowHeight);
  let area = abs(getCellArea(cell / ZONAL_GRID_WIDTH));
  return select(0.0, area, isFiniteValue(area) && area > 0.0);
}`}function w(e,t){let{output:n}=t,r=t.cellSizeMode===`geographic`||!!(n?.variances||n?.standardDeviations||n?.weightSums);return e===`sorted`&&(r||!!(n?.sums||n?.means))}var T=class{id;props;sumOrder;constructor(e){this.id=e.id??`raster-zonal-statistics`,this.props=e;let{id:t}=this;if(this.sumOrder=e.sumOrder??`sorted`,this.sumOrder!==`atomic`&&this.sumOrder!==`sorted`)throw Error(`${t} sumOrder must be atomic or sorted`);let n=g(t,e.grid);if(a(e.zones,[`uint32`],`${t} zones`),e.zones.length!==n)throw Error(`${t} zones must contain one value per cell`);if(!Number.isSafeInteger(e.zoneCapacity)||e.zoneCapacity<1||e.zoneCapacity>4294967295)throw Error(`${t} zoneCapacity must be a positive integer that fits in uint32`);let r=e.ignoredZone??4294967295;if(!Number.isInteger(r)||r<0||r>4294967295)throw Error(`${t} ignoredZone must be a uint32 integer`);let i=e.cellSizeMode??`uniform`;if(i!==`uniform`&&i!==`geographic`)throw Error(`${t} cellSizeMode must be uniform or geographic`);if(i===`geographic`){if(!e.parameters)throw Error(`${t} geographic cells need parameters`);_(t,e.parameters,v)}else if(e.parameters)throw Error(`${t} parameters need cellSizeMode geographic`);let{output:o}=e;if([...x.filter(([t])=>e.output[t]),...S.filter(t=>e.output[t])].length===0)throw Error(`${t} requires at least one output`);for(let[n,r]of x){let i=e.output[n];if(i&&(r===`count`?a(i,[`uint32`],`${t} output.${n}`):a(i,[`float32`],`${t} output.${n}`),i.length!==e.zoneCapacity))throw Error(`${t} output.${n} must contain zoneCapacity rows`)}for(let n of S){let r=e.output[n];if(r&&(a(r,[`float32`],`${t} output.${n}`),r.length!==e.zoneCapacity))throw Error(`${t} output.${n} must contain zoneCapacity rows`)}let s=o.status?.overflow;if(s&&(a(s,[`uint32`],`${t} output.status.overflow`),s.length!==1))throw Error(`${t} output.status.overflow must contain one uint32 row`);m(t,[...x.map(([t])=>e.output[t]),...S.map(t=>e.output[t]),s],[e.zones,e.parameters,...p(e.values)])}getCommandNodes(e){let{id:r,props:a}=this,{zones:u,zoneCapacity:d,output:p}=a,m=p.status?.overflow,{width:g,height:_}=a.grid,v=a.ignoredZone??4294967295;f(r,e,a.values,[]),t(r,e,[u,m,a.parameters,...x.map(([e])=>p[e]),...S.map(e=>p[e])]),n(r,`zoneCapacity`,d,1,w(this.sumOrder,a)?b(e.device):y(e.device));let T=g*_,E=a.cellSizeMode===`geographic`,O=!!(p.variances||p.standardDeviations),k=E||O||!!p.weightSums,A=!!p.means||O,j=!!p.weightSums||E&&A,M=w(this.sumOrder,a),N=!!p.valueCounts||M&&!!p.means||k&&A,P=k||!!(p.valueCounts||p.sums||p.means||p.minimums||p.maximums),F=[p.cellCounts,N,p.minimums,p.maximums].filter(Boolean).length,I=F>0&&F*4*d<=e.device.limits.maxComputeWorkgroupStorageSize,L=!I&&!!p.cellCounts,R=I?!M&&(k||!!(p.sums||p.means)):k||x.some(([e])=>e!==`cellCounts`&&p[e]),z=[],B,V;if(P){let t=h(e,`${r}-values`,a.values,a.grid,`reuse-canonical`);z.push(...t.nodes),B=t.band.storage.values,V=t.band.validity}m&&z.push(s(e,{id:`${r}-overflow-reset`,operation:`GPURasterZonalStatistics`,view:m,type:`u32`,value:`0u`}));let H=L?i(e,`${r}-zone-mask`,`uint32`,T):void 0,U=R?i(e,`${r}-value-mask`,`uint32`,T):void 0,W=M?i(e,`${r}-sort-rows`,`uint32`,T):void 0,G=M||k?i(e,`${r}-contributions`,`float32`,T):void 0,K=k&&j?i(e,`${r}-weight-contributions`,`float32`,T):void 0,q=I?void 0:m,J=(t,n)=>{let{needsValues:r}=n,i=[{name:`zones`,view:u,type:`u32`,access:`read`}];r&&i.push({name:`values`,view:B,type:`f32`,access:`read`},{name:`validity`,view:V,type:`u32`,access:`read`}),n.valueMask&&i.push({name:`valueMask`,view:n.valueMask,type:`u32`,access:`read_write`}),n.zoneMask&&i.push({name:`zoneMask`,view:n.zoneMask,type:`u32`,access:`read_write`}),n.sortRows&&i.push({name:`sortRows`,view:n.sortRows,type:`u32`,access:`read_write`}),n.contributions&&i.push({name:`contributions`,view:n.contributions,type:`f32`,access:`read_write`}),n.weightContributions&&i.push({name:`weightContributions`,view:n.weightContributions,type:`f32`,access:`read_write`}),E&&n.contributions&&i.push({name:`settings`,view:a.parameters,type:`f32`,access:`read`}),n.maskOverflow&&i.push({name:`overflowFlag`,view:n.maskOverflow,type:`atomic<u32>`,access:`read_write`}),z.push(c(e,{id:t,operation:`GPURasterZonalStatistics`,variant:`mask`,bindings:i,invocationCount:T,declarations:`const CAPACITY: u32 = ${d}u;
const IGNORED: u32 = ${v}u;
${l}
${E&&n.contributions?C(g,_):``}`,body:`let zone = zones[zonesOffset + index];
  let inZone = zone < CAPACITY && zone != IGNORED;
  ${n.zoneMask?`zoneMask[zoneMaskOffset + index] = select(0u, 1u, inZone);`:``}
  ${r?`let value = values[valuesOffset + index];
  let isValid = inZone && validity[validityOffset + index] != 0u && isFiniteValue(value);`:``}
  ${n.valueMask?`valueMask[valueMaskOffset + index] = select(0u, 1u, isValid);`:``}
  ${n.sortRows?`sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);`:``}
  ${n.contributions?E?`let weight = getCellWeight(index);
  contributions[contributionsOffset + index] = select(0.0, weight * value, isValid);`:`contributions[contributionsOffset + index] = select(0.0, value, isValid);`:``}
  ${n.weightContributions?`weightContributions[weightContributionsOffset + index] = select(0.0, ${E?`weight`:`1.0`}, isValid);`:``}
  ${n.maskOverflow?`if (zone >= CAPACITY && zone != IGNORED) {
    atomicStore(&overflowFlag[overflowFlagOffset], 1u);
  }`:``}`}))},Y={zoneMask:H,valueMask:U,sortRows:W,contributions:G,weightContributions:K,maskOverflow:q,needsValues:!!(U||W||G)};1+(Y.needsValues?2:0)+[H,U,W,G,K,q].filter(Boolean).length+(E&&G?1:0)>8?(J(`${r}-mask-zones`,{zoneMask:H,sortRows:W,maskOverflow:q,needsValues:!1}),J(`${r}-mask`,{valueMask:U,contributions:G,weightContributions:K,needsValues:!0})):(H||U||W||G||q)&&J(`${r}-mask`,Y);let X=p.cellCounts,Z=p.valueCounts??(N?i(e,`${r}-value-counts`,`uint32`,d):void 0),Q={cellCounts:X,valueCounts:Z};if(I)z.push(...D(e,{id:r,zones:u,values:B,validity:V,zoneCapacity:d,ignoredZone:v,cellCount:T,cellCounts:X,valueCounts:Z,minimums:p.minimums,maximums:p.maximums,overflow:m}));else if(M||k)for(let t of[`cellCounts`,`valueCounts`])!p[t]&&Q[t]&&z.push(...new o({id:`${r}-${t}`,keys:u,mask:t===`cellCounts`?H:U,output:Q[t],operation:`count`}).getCommandNodes(e));for(let[t,n]of x){let i=p[t];if(!i||(M||k)&&(t===`sums`||t===`means`)||I&&t!==`sums`&&t!==`means`)continue;let a={id:`${r}-${t}`,keys:u,mask:t===`cellCounts`?H:U},s=n===`count`?new o({...a,output:i,operation:n}):new o({...a,values:B,output:i,operation:n});z.push(...s.getCommandNodes(e))}return(M||k)&&z.push(...this.getMomentNodes(e,{weighted:E,needsMeans:A,needsVariance:O,needsWeightSums:j,sortedSums:M,canonicalValues:B,canonicalValidity:V,valueMask:U,sortRows:W,contributions:G,weightContributions:K,valueCounts:Z})),z}getMomentNodes(e,t){let{id:n,props:r}=this,{zones:a,zoneCapacity:o,output:s}=r,{width:u,height:d}=r.grid,f=r.ignoredZone??4294967295,{weighted:p,sortedSums:m}=t,h=[],g=s.sums??(t.needsMeans?i(e,`${n}-sorted-sums`,`float32`,o):void 0),_=s.weightSums??(t.needsWeightSums?i(e,`${n}-weight-sums`,`float32`,o):void 0),v=[...g?[{name:`sums`,contributions:t.contributions,output:g}]:[],..._?[{name:`weights`,contributions:t.weightContributions,output:_}]:[]];h.push(...this.getSumNodes(e,`${n}-sorted`,`${n}-`,m,t,v));let y=s.means??(t.needsVariance?i(e,`${n}-mean-values`,`float32`,o):void 0);if(y){let r=[{name:`sums`,view:g,type:`f32`,access:`read`},{name:`valueCounts`,view:t.valueCounts,type:`u32`,access:`read`}];p&&r.push({name:`weightSums`,view:_,type:`f32`,access:`read`}),r.push({name:`means`,view:y,type:`u32`,access:`read_write`}),h.push(c(e,{id:`${n}-means`,operation:`GPURasterZonalStatistics`,variant:p?`weighted-mean`:`sorted-mean`,bindings:r,invocationCount:o,body:p?`let count = valueCounts[valueCountsOffset + index];
  let weightSum = weightSums[weightSumsOffset + index];
  var mean = 0x7fc00000u;
  // A zone whose valid cells have no area has no weighted mean.
  if (count > 0u && weightSum > 0.0) {
    mean = bitcast<u32>(sums[sumsOffset + index] / weightSum);
  }
  means[meansOffset + index] = mean;`:`let count = valueCounts[valueCountsOffset + index];
  var mean = 0x7fc00000u;
  if (count > 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(count));
  }
  means[meansOffset + index] = mean;`}))}if(!t.needsVariance)return h;let b=u*d,x=i(e,`${n}-deviations`,`float32`,b),S=p?i(e,`${n}-weight-squares`,`float32`,b):void 0,w=[{name:`zones`,view:a,type:`u32`,access:`read`},{name:`values`,view:t.canonicalValues,type:`f32`,access:`read`},{name:`validity`,view:t.canonicalValidity,type:`u32`,access:`read`},{name:`means`,view:y,type:`f32`,access:`read`},{name:`deviations`,view:x,type:`f32`,access:`read_write`}];S&&w.push({name:`weightSquares`,view:S,type:`f32`,access:`read_write`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`}),h.push(c(e,{id:`${n}-deviations`,operation:`GPURasterZonalStatistics`,variant:p?`weighted-deviations`:`deviations`,bindings:w,invocationCount:b,declarations:`const CAPACITY: u32 = ${o}u;
const IGNORED: u32 = ${f}u;
${l}
${p?C(u,d):``}`,body:`let zone = zones[zonesOffset + index];
  let value = values[valuesOffset + index];
  let isValid = zone < CAPACITY && zone != IGNORED && validity[validityOffset + index] != 0u &&
    isFiniteValue(value);
  let weight = ${p?`getCellWeight(index)`:`1.0`};
  var deviation = 0.0;
  var weightSquare = 0.0;
  if (isValid && weight > 0.0) {
    let difference = value - means[meansOffset + zone];
    deviation = weight * difference * difference;
    weightSquare = weight * weight;
  }
  deviations[deviationsOffset + index] = deviation;
  ${S?`weightSquares[weightSquaresOffset + index] = weightSquare;`:``}`}));let T=i(e,`${n}-squared-deviation-sums`,`float32`,o),E=S?i(e,`${n}-weight-square-sums`,`float32`,o):void 0;h.push(...this.getSumNodes(e,`${n}-variance-sorted`,`${n}-variance-`,m,t,[{name:`deviations`,contributions:x,output:T},...S&&E?[{name:`weight-squares`,contributions:S,output:E}]:[]]));let D=[{name:`valueCounts`,view:t.valueCounts,type:`u32`,access:`read`},{name:`squaredDeviationSums`,view:T,type:`f32`,access:`read`}];E&&D.push({name:`weightSums`,view:_,type:`f32`,access:`read`},{name:`weightSquareSums`,view:E,type:`f32`,access:`read`});for(let e of[`variances`,`standardDeviations`])s[e]&&D.push({name:e,view:s[e],type:`u32`,access:`read_write`});return h.push(c(e,{id:`${n}-variances`,operation:`GPURasterZonalStatistics`,variant:p?`weighted-variances`:`variances`,bindings:D,invocationCount:o,body:`let valueCount = valueCounts[valueCountsOffset + index];
  var variance = 0x7fc00000u;
  var deviation = 0x7fc00000u;
  var denominator = 0.0;
  ${E?`let weightSum = weightSums[weightSumsOffset + index];
  if (weightSum != 0.0) {
    denominator = weightSum - weightSquareSums[weightSquareSumsOffset + index] / weightSum;
  }`:`denominator = f32(valueCount) - 1.0;`}
  if (valueCount >= 2u && denominator > 0.0) {
    let result = squaredDeviationSums[squaredDeviationSumsOffset + index] / denominator;
    variance = bitcast<u32>(result);
    deviation = bitcast<u32>(sqrt(result));
  }
  ${s.variances?`variances[variancesOffset + index] = variance;`:``}
  ${s.standardDeviations?`standardDeviations[standardDeviationsOffset + index] = deviation;`:``}`})),h}getSumNodes(e,t,n,r,i,a){return a.length===0?[]:r?u(e,{id:t,operation:`GPURasterZonalStatistics`,segmentCount:this.props.zoneCapacity,segmentKeys:i.sortRows,reductions:a}):a.flatMap(t=>new o({id:`${n}${t.name}`,keys:this.props.zones,mask:i.valueMask,values:t.contributions,output:t.output,operation:`sum`}).getCommandNodes(e))}},E=256;function D(e,t){let{id:n,zoneCapacity:r,cellCount:i}=t,a=[{name:`cellCounts`,view:t.cellCounts,initial:`0u`,merge:`atomicAdd`},{name:`valueCounts`,view:t.valueCounts,initial:`0u`,merge:`atomicAdd`},{name:`minimums`,view:t.minimums,initial:`0xffffffffu`,merge:`atomicMin`},{name:`maximums`,view:t.maximums,initial:`0u`,merge:`atomicMax`}].filter(e=>e.view),o=!!(t.valueCounts||t.minimums||t.maximums),s=Math.min(64,Math.max(8,Math.ceil(r/32))),u=[];u.push(c(e,{id:`${n}-zone-reduce-init`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce-init`,bindings:a.map(e=>({name:e.name,view:e.view,type:`u32`,access:`read_write`})),invocationCount:r,body:a.map(e=>`${e.name}[${e.name}Offset + index] = ${e.initial};`).join(`
  `)}));let d=[{name:`zones`,view:t.zones,type:`u32`,access:`read`}];o&&d.push({name:`values`,view:t.values,type:`f32`,access:`read`},{name:`validity`,view:t.validity,type:`u32`,access:`read`});for(let e of a)d.push({name:e.name,view:e.view,type:`atomic<u32>`,access:`read_write`});t.overflow&&d.push({name:`overflowFlag`,view:t.overflow,type:`atomic<u32>`,access:`read_write`});let f=a.map(e=>`var<workgroup> local_${e.name}: array<atomic<u32>, ${r}>;`).join(`
`),p=!!t.minimums,m=!!t.maximums;u.push(c(e,{id:`${n}-zone-reduce`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce`,bindings:d,invocationCount:Math.ceil(i/s),guardIndex:!1,workgroupSize:E,declarations:`const CAPACITY: u32 = ${r}u;
const IGNORED: u32 = ${t.ignoredZone}u;
const CELL_COUNT: u32 = ${i}u;
const CELLS_PER_THREAD: u32 = ${s}u;
${f}
${l}
fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,body:`for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${E}u) {
    ${a.map(e=>`atomicStore(&local_${e.name}[slot], ${e.initial});`).join(`
    `)}
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${E}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    let zone = zones[zonesOffset + cell];
    if (zone >= CAPACITY || zone == IGNORED) {
      ${t.overflow?`if (zone != IGNORED) {
        atomicStore(&overflowFlag[overflowFlagOffset], 1u);
      }`:``}
      continue;
    }
    ${t.cellCounts?`atomicAdd(&local_cellCounts[zone], 1u);`:``}
    ${o?`let value = values[valuesOffset + cell];
    if (validity[validityOffset + cell] != 0u && isFiniteValue(value)) {
      ${t.valueCounts?`atomicAdd(&local_valueCounts[zone], 1u);`:``}
      ${p||m?`let key = encodeOrderedFloat(value);`:``}
      ${p?`atomicMin(&local_minimums[zone], key);`:``}
      ${m?`atomicMax(&local_maximums[zone], key);`:``}
    }`:``}
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${E}u) {
    ${a.map(e=>`let partial_${e.name} = atomicLoad(&local_${e.name}[slot]);
    if (partial_${e.name} != ${e.initial}) {
      ${e.merge}(&${e.name}[${e.name}Offset + slot], partial_${e.name});
    }`).join(`
    `)}
  }`}));let h=a.filter(e=>e.name===`minimums`||e.name===`maximums`);return h.length>0&&u.push(c(e,{id:`${n}-zone-reduce-decode`,operation:`GPURasterZonalStatistics`,variant:`zone-reduce-decode`,bindings:h.map(e=>({name:e.name,view:e.view,type:`u32`,access:`read_write`})),invocationCount:r,declarations:`fn decodeOrderedKey(key: u32) -> u32 {
  return select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u);
}`,body:h.map(e=>`let key_${e.name} = ${e.name}[${e.name}Offset + index];
  ${e.name}[${e.name}Offset + index] = select(decodeOrderedKey(key_${e.name}), 0x7fc00000u, key_${e.name} == ${e.initial});`).join(`
  `)})),u}export{T as t};