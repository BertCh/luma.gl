import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./earth-xB16XgAa.js";import"./analysis-frame-Blu0tYdw.js";import{i as a}from"./convert-colors-D-TZjU1C.js";import{i as o,o as s,s as c}from"./cell-topology-wgsl-BT8_s49b.js";import{N as l,_ as u,h as d,j as f,n as p,t as m,w as h,y as g}from"./resources-CwvhEOBe.js";import{t as _}from"./gpu-scan-BuGhTLDQ.js";import{t as v}from"./draw-command-buffer-BeqRkhBa.js";import{r as y,s as b}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as x,n as S,r as C,t as w}from"./cell-key-wgsl-BQM8miC7.js";import{a as T,i as E,n as D,o as O,r as k,t as A}from"./gpu-cell-aggregation-BFqjxJvj.js";import{a as j,i as ee,t as M}from"./gpu-class-breaks-DsbEaky8.js";import{n as te,r as N,t as ne}from"./gpu-color-scale-BxTDjZQ3.js";import{r as P,s as F,t as I}from"./recipe-utils-CqprOQld.js";import{b as re,c as ie,m as L,y as ae}from"./index-Bjk18V2I.js";import{t as R}from"./graph-buffers-B4_80RlX.js";import{t as oe}from"./graph-swap-41Ru7LVu.js";import{t as se}from"./summary-reader-CsgWxENf.js";import{n as ce,t as le}from"./b8-areas-CiQcUCr6.js";import{t as ue}from"./mode-kernels-Dxm2GWvI.js";import{c as de,d as fe,i as pe,l as me,o as he,r as ge,s as _e}from"./b8-common-CgT2-JC9.js";var z=`GPUCellRollup`,ve=class{id;props;layout;sumScale;spatialContext;constructor(e){this.id=e.id??`cell-rollup`,this.props=e;let t=this.id;if(this.layout=s(e.family,e.resolution),s(e.family,e.sourceResolution),this.spatialContext=c(t,e.family,e.spatialContext,`cells`,e.resolution),e.resolution>e.sourceResolution)throw Error(`${t} resolution must not exceed sourceResolution`);this.sumScale=e.sumScale??65536,x(t,this.sumScale),O(t,`source`,e.source),O(t,`output`,e.output);let{source:n,output:r}=e;if((r.sums||r.sumValues)&&!n.sums)throw Error(`${t} source.sums is required to roll up sums`);if(r.minimums&&!n.minimums||r.maximums&&!n.maximums)throw Error(`${t} source extremes are required to roll up extremes`);d(t,[...E(r),...e.extraCounts??[]],E(n))}getCommandNodes(e){let{id:t,props:n,layout:r}=this;u(t,e,[...E(n.source),...E(n.output),...n.extraCounts??[]]);let i=n.source.cells.length,a=T(r),s=h(e,`${t}-key-low`,`uint32`,i),c=a?h(e,`${t}-key-high`,`uint32`,i):void 0;return[b(e,{id:`${t}-keys`,operation:z,variant:`parent-keys`,bindings:[{name:`childCells`,view:n.source.cells,type:`u32`,access:`read`},{name:`childCount`,view:n.source.status.count,type:`u32`,access:`read`},{name:`keyLow`,view:s,type:`u32`,access:`read_write`},...c?[{name:`keyHigh`,view:c,type:`u32`,access:`read_write`}]:[]],invocationCount:i,declarations:`const SOURCE_RESOLUTION: u32 = ${n.sourceResolution}u;
const RESOLUTION_MASK: u32 = ${n.family===`quadbin`?`0x1fu`:`0xfu`};
${o}`,body:`var compact = cellShiftLeft(vec2u(0u, 1u), ${r.width}u);
  let key = vec2u(childCells[childCellsOffset + 2u * index + 1u], childCells[childCellsOffset + 2u * index]);
  if (index < childCount[childCountOffset] && ((key.x >> 20u) & RESOLUTION_MASK) == SOURCE_RESOLUTION) {
    compact = cellGetCompactKey(key, ${r.lowBit}u, ${r.width}u);
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${c?`keyHigh[keyHighOffset + index] = compact.x;`:``}`}),...k(e,{id:t,operation:z,layout:r,keyLow:s,keyHigh:c,sorted:!0,source:{kind:`cells`,table:n.source},sumScale:this.sumScale,output:n.output,overflowSources:[n.source.status.overflow],extraCounts:n.extraCounts})]}},B=64,ye=class{id;props;spatialContext;constructor(e){this.id=e.id??`cell-pyramid`,this.props=e;let t=this.id,{resolutions:n,output:r}=e;if(n.length<1)throw Error(`${t} needs at least one level`);if(r.levels.length!==n.length)throw Error(`${t} output.levels must hold one table per resolution`);for(let[i,a]of n.entries())if(s(e.family,a),O(t,`output.levels[${i}]`,r.levels[i]),i>0&&a>=n[i-1])throw Error(`${t} resolutions must strictly decrease`);if(r.levelCounts&&(g(r.levelCounts,[`uint32`],`${t} output.levelCounts`),r.levelCounts.length!==n.length))throw Error(`${t} output.levelCounts must hold one row per level`);this.spatialContext=this.createAggregation(r.levels[0],[]).spatialContext}get levelFirstRows(){return this.props.output.levels.map(D)}getCommandNodes(e){let{id:t,props:n}=this,{levelCounts:r}=n.output;u(t,e,[r]);let i=r?h(e,`${t}-level-count-slots`,`uint32`,n.resolutions.length*B):void 0,a=n.resolutions.map((t,n)=>i?[e.createDataView(i.buffer,{format:`uint32`,length:1,byteOffset:i.byteOffset+n*B*Uint32Array.BYTES_PER_ELEMENT})]:[]),o=this.getLevelTables(e),s=[...this.createAggregation(o[0],a[0]).getCommandNodes(e)];for(let r=1;r<o.length;r++)s.push(...new ve({id:`${t}-level-${r}`,family:n.family,sourceResolution:n.resolutions[r-1],resolution:n.resolutions[r],source:o[r-1],sumScale:n.sumScale,output:o[r],extraCounts:a[r]}).getCommandNodes(e));return r&&i&&s.push(b(e,{id:`${t}-level-counts`,operation:`GPUCellPyramid`,variant:`level-counts`,bindings:[{name:`slots`,view:i,type:`u32`,access:`read`},{name:`levelCounts`,view:r,type:`u32`,access:`read_write`}],invocationCount:n.resolutions.length,body:`levelCounts[levelCountsOffset + index] = slots[slotsOffset + index * ${B}u];`})),s}createAggregation(e,t){let{resolutions:n,output:r,...i}=this.props;return new A({...i,id:`${this.id}-level-0`,resolution:n[0],output:e,extraCounts:t})}getLevelTables(e){let t=this.props.output.levels.map(e=>({...e})),n=!1,r=!1,i=!1;for(let a=t.length-1;a>=0;a--){let o=t[a],s=o.cells.length,c=`${this.id}-level-${a}`;n&&!o.sums&&(o.sums=h(e,`${c}-sums`,`uint32x2`,s)),r&&!o.minimums&&(o.minimums=h(e,`${c}-minimums`,`float32`,s)),i&&!o.maximums&&(o.maximums=h(e,`${c}-maximums`,`float32`,s)),n||=!!(o.sums||o.sumValues),r||=!!o.minimums,i||=!!o.maximums}return t}},be=class{id;props;constructor(e){this.id=e.id??`cell-level-selection`,this.props=e;let t=this.id;if(f(e.levelCounts,`${t} levelCounts`),f(e.activeLevel,`${t} activeLevel`),f(e.output.status.count,`${t} output.status.count`),e.levelCounts.length<1||e.activeLevel.length<1)throw Error(`${t} levelCounts and activeLevel must hold at least one row`);let n=e.levelFirstRows??[];if(e.levelFirstRows&&(n.length!==e.levelCounts.length||n.some(e=>!Number.isInteger(e)||e<0||e>4294967295)))throw Error(`${t} levelFirstRows must hold one uint32 per level`);if(e.output.firstRow&&f(e.output.firstRow,`${t} output.firstRow`),e.output.drawArguments&&(f(e.output.drawArguments,`${t} output.drawArguments`),e.output.drawArguments.length<4))throw Error(`${t} output.drawArguments must hold 4 uint32 values`)}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n;u(t,e,[n.levelCounts,n.activeLevel,r.status.count,r.firstRow,r.drawArguments]);let i=n.levelCounts.length,a=n.levelFirstRows??Array(i).fill(0);return[b(e,{id:`${t}-select`,operation:`GPUCellLevelSelection`,bindings:[{name:`levelCounts`,view:n.levelCounts,type:`u32`,access:`read`},{name:`activeLevel`,view:n.activeLevel,type:`u32`,access:`read`},{name:`countOut`,view:r.status.count,type:`u32`,access:`read_write`},...r.firstRow?[{name:`firstRowOut`,view:r.firstRow,type:`u32`,access:`read_write`}]:[],...r.drawArguments?[{name:`drawArguments`,view:r.drawArguments,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const LEVEL_FIRST_ROWS = array<u32, ${i}>(${a.map(e=>`${e}u`).join(`, `)});`,body:`let level = min(activeLevel[activeLevelOffset], ${i-1}u);
  let count = levelCounts[levelCountsOffset + level];
  var firstRows = LEVEL_FIRST_ROWS;
  let firstRow = firstRows[level];
  countOut[countOutOffset] = count;
  ${r.firstRow?`firstRowOut[firstRowOutOffset] = firstRow;`:``}
  ${r.drawArguments?`drawArguments[drawArgumentsOffset + 1u] = count;
  drawArguments[drawArgumentsOffset + 3u] = firstRow;`:``}`})]}},V=`GPUCellTableCompare`,H=256,xe=256,U=class{id;props;measure;sumScale;zScoreKind;constructor(e){this.id=e.id??`cell-table-compare`,this.props=e;let t=this.id;if(this.measure=e.measure??`count`,this.measure!==`count`&&this.measure!==`sum`)throw Error(`${t} measure must be 'count' or 'sum'`);if(this.sumScale=e.sumScale??65536,x(t,this.sumScale),this.zScoreKind=e.zScore??(this.measure===`sum`?`standardized`:`poisson`),this.zScoreKind!==`poisson`&&this.zScoreKind!==`standardized`)throw Error(`${t} zScore must be 'poisson' or 'standardized'`);if(O(t,`before`,e.before),O(t,`after`,e.after),this.measure===`sum`&&(!e.before.sums||!e.after.sums))throw Error(`${t} measure 'sum' needs sums on both before and after`);let{output:n}=e;g(n.cells,[`uint32x2`],`${t} output.cells`);let r=n.cells.length;if(r<1)throw Error(`${t} output.cells must hold at least one row`);n.presence&&g(n.presence,[`uint32`],`${t} output.presence`),n.mask&&g(n.mask,[`uint32`],`${t} output.mask`);for(let e of[`before`,`after`,`delta`,`ratio`,`percentChange`,`zScore`]){let r=n[e];r&&g(r,[`float32`],`${t} output.${e}`)}for(let e of[`presence`,`mask`,`before`,`after`,`delta`,`ratio`,`percentChange`,`zScore`]){let i=n[e];if(i&&i.length!==r)throw Error(`${t} output.${e} needs output.cells.length rows`)}for(let e of[`count`,`overflow`,`requiredCount`]){let r=n.status[e];if(r&&(g(r,[`uint32`],`${t} output.status.${e}`),r.length<1))throw Error(`${t} output.status.${e} needs one row`)}d(t,Se(n),[...E(e.before),...E(e.after)])}getCommandNodes(e){let{id:t,props:n,measure:r}=this,{before:i,after:a,output:o}=n;u(t,e,[...E(i),...E(a),...Se(o)]);let s=o.cells.length,c=i.cells.length,l=a.cells.length,d=(n,r,i)=>h(e,`${t}-${n}`,r,i),f=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),p=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),m=[],g=d(`total`,`uint32`,1),v=d(`before-rows`,`uint32`,s),x=d(`after-rows`,`uint32`,s),w=o.before??d(`before-values`,`float32`,s),T=o.after??d(`after-values`,`float32`,s),D=o.delta??d(`delta-values`,`float32`,s),O=d(`matched-flags`,`uint32`,c+1),k=d(`matched-prefix`,`uint32`,c+1),A=d(`after-bounds`,`uint32`,c+3),j=`const CAPACITY: u32 = ${s}u;
const CAPACITY_BEFORE: u32 = ${c}u;
const CAPACITY_AFTER: u32 = ${l}u;
const NO_ROW: u32 = 0xffffffffu;
fn getBeforeKey(row: u32) -> vec2u {
  return vec2u(beforeCells[beforeCellsOffset + 2u * row + 1u], beforeCells[beforeCellsOffset + 2u * row]);
}
fn getAfterKey(row: u32) -> vec2u {
  return vec2u(afterCells[afterCellsOffset + 2u * row + 1u], afterCells[afterCellsOffset + 2u * row]);
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`,ee=(e,t)=>`fn ${e}(key: vec2u, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isKeyLess(get${t}Key(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;m.push(b(e,{id:`${t}-flags`,operation:V,variant:`matches`,bindings:[f(`beforeCells`,i.cells),f(`beforeCount`,i.status.count),f(`afterCells`,a.cells),f(`afterCount`,a.status.count),p(`flags`,O),p(`bounds`,A)],invocationCount:c+1,declarations:`${j}
${ee(`lowerBoundAfter`,`After`)}`,body:`let countBefore = min(beforeCount[beforeCountOffset], CAPACITY_BEFORE);
  let countAfter = min(afterCount[afterCountOffset], CAPACITY_AFTER);
  if (index == 0u) {
    bounds[boundsOffset + CAPACITY_BEFORE + 1u] = countBefore;
    bounds[boundsOffset + CAPACITY_BEFORE + 2u] = countAfter;
  }
  var isMatch = false;
  if (index < countBefore) {
    let key = getBeforeKey(index);
    let bound = lowerBoundAfter(key, countAfter);
    bounds[boundsOffset + index] = bound;
    isMatch = bound < countAfter && all(getAfterKey(bound) == key);
  }
  flags[flagsOffset + index] = select(0u, 1u, isMatch);`}),...new _({id:`${t}-scan`,input:O,output:k,mode:`exclusive`}).getCommandNodes(e)),m.push(b(e,{id:`${t}-merge`,operation:V,variant:`merge-path`,bindings:[f(`beforeCells`,i.cells),f(`afterCells`,a.cells),f(`bounds`,A),f(`prefix`,k),p(`cellsOut`,o.cells),p(`beforeRows`,v),p(`afterRows`,x),p(`total`,g)],invocationCount:c+l,declarations:`${j}
${ee(`lowerBoundBefore`,`Before`)}`,body:`let countBefore = bounds[boundsOffset + CAPACITY_BEFORE + 1u];
  let countAfter = bounds[boundsOffset + CAPACITY_BEFORE + 2u];
  if (index == 0u) {
    total[totalOffset] = countBefore + countAfter - prefix[prefixOffset + CAPACITY_BEFORE];
  }
  var key = vec2u(0u);
  var position = 0u;
  var beforeRow = NO_ROW;
  var afterRow = NO_ROW;
  if (index < CAPACITY_BEFORE) {
    if (index >= countBefore) {
      return;
    }
    key = getBeforeKey(index);
    let bound = bounds[boundsOffset + index];
    position = index + bound - prefix[prefixOffset + index];
    beforeRow = index;
    if (bound < countAfter && all(getAfterKey(bound) == key)) {
      afterRow = bound;
    }
  } else {
    let row = index - CAPACITY_BEFORE;
    if (row >= countAfter) {
      return;
    }
    key = getAfterKey(row);
    let bound = lowerBoundBefore(key, countBefore);
    if (bound < countBefore && all(getBeforeKey(bound) == key)) {
      return;
    }
    position = row + bound - prefix[prefixOffset + bound];
    afterRow = row;
  }
  if (position < CAPACITY) {
    cellsOut[cellsOutOffset + 2u * position] = key.y;
    cellsOut[cellsOutOffset + 2u * position + 1u] = key.x;
    beforeRows[beforeRowsOffset + position] = beforeRow;
    afterRows[afterRowsOffset + position] = afterRow;
  }`}));let M=r===`sum`,te=M?i.sums:i.counts,N=M?a.sums:a.counts,ne=(e,t)=>M?`vec2u(${e}Measure[${e}MeasureOffset + 2u * ${t} + 1u], ${e}Measure[${e}MeasureOffset + 2u * ${t}])`:`vec2u(0u, ${e}Measure[${e}MeasureOffset + ${t}])`,P=e=>M?`cellI64ToF32(${e}) / SUM_SCALE`:`cellI64ToF32(${e})`;m.push(b(e,{id:`${t}-measures`,operation:V,variant:M?`sum-measures`:`count-measures`,bindings:[f(`beforeRows`,v),f(`afterRows`,x),f(`beforeMeasure`,te),f(`afterMeasure`,N),f(`total`,g),p(`beforeOut`,w,`f32`),p(`afterOut`,T,`f32`),p(`deltaOut`,D,`f32`)],invocationCount:s,declarations:`const CAPACITY: u32 = ${s}u;
const NO_ROW: u32 = 0xffffffffu;
${S}
${C(this.sumScale)}`,body:`if (index >= min(total[totalOffset], CAPACITY)) {
    beforeOut[beforeOutOffset + index] = 0.0;
    afterOut[afterOutOffset + index] = 0.0;
    deltaOut[deltaOutOffset + index] = getNaN();
    return;
  }
  let beforeRow = beforeRows[beforeRowsOffset + index];
  let afterRow = afterRows[afterRowsOffset + index];
  var beforeValue = vec2u(0u);
  var afterValue = vec2u(0u);
  if (beforeRow != NO_ROW) {
    beforeValue = ${ne(`before`,`beforeRow`)};
  }
  if (afterRow != NO_ROW) {
    afterValue = ${ne(`after`,`afterRow`)};
  }
  let difference = cellAddI64(afterValue, cellNegateI64(beforeValue));
  beforeOut[beforeOutOffset + index] = ${P(`beforeValue`)};
  afterOut[afterOutOffset + index] = ${P(`afterValue`)};
  deltaOut[deltaOutOffset + index] = ${P(`difference`)};`}));let F=o.zScore&&this.zScoreKind===`standardized`?d(`statistics`,`float32`,2):void 0;if(F){let n=Math.max(1,Math.min(xe,Math.ceil(s/H))),r=d(`statistics-partials`,`float32`,2*n),i=e=>`partials[localInvocationIndex] = ${e};
  workgroupBarrier();
  for (var stride = ${H/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,a=e=>`partials[localInvocationIndex] = 0.0;
  if (localInvocationIndex < GROUPS) {
    partials[localInvocationIndex] = groupPartials[groupPartialsOffset + ${e} + localInvocationIndex];
  }
  workgroupBarrier();
  for (var stride = ${H/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`,o=`const CAPACITY: u32 = ${s}u;
const GROUPS: u32 = ${n}u;
const STRIDE: u32 = ${n*H}u;
var<workgroup> partials: array<f32, ${H}>;`;m.push(b(e,{id:`${t}-statistics-sums`,operation:V,variant:`statistics-sums`,bindings:[f(`deltas`,D,`f32`),f(`total`,g),p(`groupPartials`,r,`f32`)],workgroupSize:H,invocationCount:n*H,guardIndex:!1,declarations:o,body:`let rowCount = min(total[totalOffset], CAPACITY);
  var partialSum = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    partialSum += deltas[deltasOffset + row];
  }
  ${i(`partialSum`)}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + index / ${H}u] = partials[0];
  }`}),b(e,{id:`${t}-statistics-squares`,operation:V,variant:`statistics-squares`,bindings:[f(`deltas`,D,`f32`),f(`total`,g),p(`groupPartials`,r,`f32`)],workgroupSize:H,invocationCount:n*H,guardIndex:!1,declarations:o,body:`let rowCount = min(total[totalOffset], CAPACITY);
  ${a(`0u`)}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  var partialSquares = 0.0;
  for (var row = index; row < rowCount; row += STRIDE) {
    let centered = deltas[deltasOffset + row] - mean;
    partialSquares += centered * centered;
  }
  ${i(`partialSquares`)}
  if (localInvocationIndex == 0u) {
    groupPartials[groupPartialsOffset + GROUPS + index / ${H}u] = partials[0];
  }`}),b(e,{id:`${t}-statistics`,operation:V,variant:`statistics`,bindings:[f(`groupPartials`,r,`f32`),f(`total`,g),p(`statistics`,F,`f32`)],workgroupSize:H,invocationCount:H,guardIndex:!1,declarations:o,body:`let rowCount = min(total[totalOffset], CAPACITY);
  ${a(`0u`)}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  ${a(`GROUPS`)}
  if (localInvocationIndex == 0u) {
    statistics[statisticsOffset] = mean;
    statistics[statisticsOffset + 1u] = select(0.0, sqrt(partials[0] / f32(rowCount)), rowCount > 0u);
  }`}))}return(o.ratio||o.percentChange||o.zScore)&&m.push(b(e,{id:`${t}-derived`,operation:V,variant:`derived`,bindings:[f(`beforeValues`,w,`f32`),f(`afterValues`,T,`f32`),f(`deltaValues`,D,`f32`),f(`total`,g),...o.ratio?[p(`ratioOut`,o.ratio,`f32`)]:[],...o.percentChange?[p(`percentOut`,o.percentChange,`f32`)]:[],...o.zScore?[p(`zScoreOut`,o.zScore,`f32`)]:[],...F?[f(`statistics`,F,`f32`)]:[]],invocationCount:s,declarations:`const CAPACITY: u32 = ${s}u;
${S}`,body:`let isOccupied = index < min(total[totalOffset], CAPACITY);
  let nan = getNaN();
  let beforeValue = beforeValues[beforeValuesOffset + index];
  let afterValue = afterValues[afterValuesOffset + index];
  let delta = deltaValues[deltaValuesOffset + index];
  ${o.ratio?`ratioOut[ratioOutOffset + index] = select(nan, afterValue / beforeValue, isOccupied && beforeValue != 0.0);`:``}
  ${o.percentChange?`percentOut[percentOutOffset + index] = select(nan, (100.0 * delta) / beforeValue, isOccupied && beforeValue != 0.0);`:``}
  ${o.zScore?this.zScoreKind===`poisson`?`let denominator = beforeValue + afterValue;
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, delta / sqrt(denominator), denominator != 0.0), isOccupied);`:`let mean = statistics[statisticsOffset];
  let deviation = statistics[statisticsOffset + 1u];
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, (delta - mean) / deviation, deviation != 0.0), isOccupied);`:``}`})),m.push(b(e,{id:`${t}-tail`,operation:V,variant:`tail`,bindings:[f(`beforeRows`,v),f(`afterRows`,x),f(`total`,g),p(`cellsOut`,o.cells),...o.presence?[p(`presenceOut`,o.presence)]:[],...o.mask?[p(`maskOut`,o.mask)]:[]],invocationCount:s,declarations:`const CAPACITY: u32 = ${s}u;
const NO_ROW: u32 = 0xffffffffu;`,body:`if (index >= min(total[totalOffset], CAPACITY)) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    ${o.presence?`presenceOut[presenceOutOffset + index] = 0u;`:``}
    ${o.mask?`maskOut[maskOutOffset + index] = 0u;`:``}
    return;
  }
  ${o.mask?`maskOut[maskOutOffset + index] = 1u;`:``}
  ${o.presence?`presenceOut[presenceOutOffset + index] =
    select(0u, 1u, beforeRows[beforeRowsOffset + index] != NO_ROW) |
    select(0u, 2u, afterRows[afterRowsOffset + index] != NO_ROW);`:``}`})),m.push(y(e,{id:`${t}-publish`,operation:V,requiredCount:g,output:{ids:o.cells,status:o.status},overflowSources:[i.status.overflow,a.status.overflow]})),m}};function Se(e){return[e.cells,e.presence,e.mask,e.before,e.after,e.delta,e.ratio,e.percentChange,e.zScore,e.status.count,e.status.overflow,e.status.requiredCount]}var W=`addPeriodComparisonRecipe`;function Ce(e,t){let n=t.id??`period-comparison`,r=new I(e),i=t.outputs??{},a=t.scratch??{},{tableCapacity:o}=t,s=t.unionCapacity??2*o,c=t.measure??`count`,l=t.classify??`delta`;P(W,o>=1,`tableCapacity must be positive`);let[u,d]=[`before`,`after`].map(i=>{let s=t[i];P(W,!!s.positions!=!!s.cells,`${i} needs exactly one of positions or cells`),P(W,c===`count`||!!s.values,`${i} needs values for the 'sum' measure`);let l=a[i]??{},u={cells:F(e,`${n}-${i}-cells`,`uint32x2`,o,l.cells),counts:F(e,`${n}-${i}-counts`,`uint32`,o,l.counts),sums:s.values?F(e,`${n}-${i}-sums`,`uint32x2`,o,l.sums):l.sums,status:{count:F(e,`${n}-${i}-count`,`uint32`,1,l.status?.count),overflow:F(e,`${n}-${i}-overflow`,`uint32`,1,l.status?.overflow),requiredCount:F(e,`${n}-${i}-required-count`,`uint32`,1,l.status?.requiredCount)}};return r.add(new A({id:`${n}-${i}`,family:t.family,resolution:t.resolution,positions:s.positions,cells:s.cells,values:s.values,mask:s.mask,output:u})),u}),f=i.comparison??{},p={cells:F(e,`${n}-union-cells`,`uint32x2`,s,f.cells),presence:f.presence,before:f.before,after:f.after,delta:F(e,`${n}-delta`,`float32`,s,f.delta),ratio:f.ratio,percentChange:f.percentChange,zScore:f.zScore,status:{count:F(e,`${n}-union-count`,`uint32`,1,f.status?.count),overflow:F(e,`${n}-union-overflow`,`uint32`,1,f.status?.overflow),requiredCount:F(e,`${n}-union-required-count`,`uint32`,1,f.status?.requiredCount)}};l===`ratio`?p.ratio=F(e,`${n}-ratio`,`float32`,s,f.ratio):l===`percentChange`?p.percentChange=F(e,`${n}-percent-change`,`float32`,s,f.percentChange):l===`zScore`&&(p.zScore=F(e,`${n}-z-score`,`float32`,s,f.zScore));let m=F(e,`${n}-union-mask`,`uint32`,s,a.mask);p.mask=m,r.add(new U({id:`${n}-compare`,before:u,after:d,measure:c,zScore:t.zScore,output:p}));let h=p[l],g=F(e,`${n}-breaks`,`float32`,t.maximumClassCount+1,i.breaks),_=F(e,`${n}-class-count`,`uint32`,1,i.classCount),v=F(e,`${n}-colors`,`uint32`,s,i.colors),y=F(e,`${n}-class-indices`,`uint32`,s,i.classIndices);return r.add(new M({id:`${n}-class-breaks`,values:h,mask:m,parameters:t.classBreaksParameters,maximumClassCount:t.maximumClassCount,methods:t.methods,output:{breaks:g,classCount:_}})),r.add(new ne({id:`${n}-color-scale`,values:h,mask:m,domain:g,domainCount:_,palette:t.palette,parameters:t.colorScaleParameters,maximumDomainCount:t.maximumClassCount+1,maximumPaletteCount:t.maximumPaletteCount,output:{colors:v,classIndices:y}})),{contributors:r.contributors,unionCapacity:s,before:u,after:d,comparison:p,mask:m,breaks:g,classCount:_,colors:v,classIndices:y,outputs:{comparison:p,breaks:g,classCount:_,colors:v,classIndices:y},intermediates:{before:u,after:d,mask:m},status:{stages:[{stage:`before-aggregation`,status:u.status},{stage:`after-aggregation`,status:d.status},{stage:`comparison`,status:p.status}]}}}function we(e,t,n,r,i){let a=n.counts.length;ue(e,{id:`${t}-cell-table-columns`,invocationCount:a,bindings:[{name:`counts`,view:n.counts,type:`u32`,access:`read`},...n.sumValues?[{name:`sumValues`,view:n.sumValues,type:`f32`,access:`read`}]:[],...i?[{name:`rowCount`,view:i,type:`u32`,access:`read`}]:[],{name:`columnValues`,view:r.values,type:`f32`,access:`read_write`},{name:`columnMask`,view:r.mask,type:`u32`,access:`read_write`}],body:`let occupied = counts[countsOffset + index];
  let inRange = ${i?`index < rowCount[rowCountOffset]`:`true`};
  columnValues[columnValuesOffset + index] = ${n.sumValues?`sumValues[sumValuesOffset + index]`:`f32(occupied)`};
  columnMask[columnMaskOffset + index] = select(0u, 1u, inRange && occupied > 0u);`})}var Te=64,G={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},Ee={density:0,meanValue:1,packed:2,signed:3},De=`
${ae()}

struct QuadbinStyle {
  // log10 observations per square kilometer at the low and high end, opacity, inset
  range: vec4<f32>,
  // mean/signed value at the low end, at the high end, signed limit, unused
  value: vec4<f32>,
  // mode, colormap index, useRowCount, signed transform (0 linear, 1 log2 ratio)
  config: vec4<u32>,
  // color of cells without a baseline (NaN ratio)
  noBaseline: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> quadbinStyle: QuadbinStyle;
@group(0) @binding(auto) var<storage, read> quadbinKeysEven: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinKeysOdd: array<vec2<u32>>;
@group(0) @binding(auto) var<storage, read> quadbinCountsEven: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinCountsOdd: array<u32>;
// Either an f32 value column (mean, signed) or packed rgba8 colors: one binding keeps the vertex
// stage within the default limit of 8 storage buffers.
@group(0) @binding(auto) var<storage, read> quadbinData: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinActiveLevel: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinFirstRow: array<u32>;
@group(0) @binding(auto) var<storage, read> quadbinRowCount: array<u32>;

struct QuadbinVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn compactEvenBits(value: u32) -> u32 {
  var bits = value & 0x55555555u;
  bits = (bits | (bits >> 1u)) & 0x33333333u;
  bits = (bits | (bits >> 2u)) & 0x0f0f0f0fu;
  bits = (bits | (bits >> 4u)) & 0x00ff00ffu;
  bits = (bits | (bits >> 8u)) & 0x0000ffffu;
  return bits;
}

// Decodes a Quadbin key stored as (low, high) words into (tileX, tileY, zoom).
fn decodeQuadbin(key: vec2<u32>) -> vec3<u32> {
  let zoom = (key.y >> 20u) & 0x1fu;
  let mortonHigh = (key.y << 12u) | (key.x >> 20u);
  let mortonLow = key.x << 12u;
  let x32 = compactEvenBits(mortonLow) | (compactEvenBits(mortonHigh) << 16u);
  let y32 = compactEvenBits(mortonLow >> 1u) | (compactEvenBits(mortonHigh >> 1u) << 16u);
  let shift = 32u - zoom;
  return vec3<u32>(x32 >> shift, y32 >> shift, zoom);
}

fn getMercatorLatitude(tileY: f32, tileCount: f32) -> f32 {
  return degrees(atan(sinh(3.14159265358979 * (1.0 - 2.0 * tileY / tileCount))));
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> QuadbinVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  var output: QuadbinVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let mode = quadbinStyle.config.x;
  if (quadbinStyle.config.z != 0u && instanceIndex >= quadbinRowCount[0]) { return output; }
  // Even and odd pyramid levels live in alternating slabs; the level is the GPU-visible word that
  // GPUCellLevelSelection also reads.
  let isOdd = (quadbinActiveLevel[0] & 1u) == 1u;
  let row = quadbinFirstRow[0] + instanceIndex;
  var count = 1u;
  if (mode <= 1u) {
    count = select(quadbinCountsEven[row], quadbinCountsOdd[row], isOdd);
    if (count == 0u) { return output; }
  }
  let tile = decodeQuadbin(select(quadbinKeysEven[row], quadbinKeysOdd[row], isOdd));
  let tileCount = f32(1u << tile.z);
  let inset = quadbinStyle.range.w;
  let corner = mix(vec2<f32>(inset), vec2<f32>(1.0 - inset), corners[vertexIndex]);
  let longitude = (f32(tile.x) + corner.x) / tileCount * 360.0 - 180.0;
  let latitude = getMercatorLatitude(f32(tile.y) + corner.y, tileCount);

  var color = vec4<f32>(0.0);
  if (mode == 0u) {
    // Observations per square kilometer: a Web Mercator tile is stretched by 1 / cos(latitude).
    let centerLatitude = getMercatorLatitude(f32(tile.y) + 0.5, tileCount);
    let edgeKilometers = ${(2*Math.PI*i/1e3).toFixed(6)} / tileCount * cos(radians(centerLatitude));
    let density = f32(count) / (edgeKilometers * edgeKilometers);
    let t = (log2(max(density, 1.0e-3)) * 0.30103 - quadbinStyle.range.x) /
      max(quadbinStyle.range.y - quadbinStyle.range.x, 1.0e-3);
    color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, clamp(t, 0.0, 1.0)), quadbinStyle.range.z);
  } else if (mode == 1u) {
    let mean = bitcast<f32>(quadbinData[row]) / f32(count);
    let t = (mean - quadbinStyle.value.x) / max(quadbinStyle.value.y - quadbinStyle.value.x, 1.0e-6);
    color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, clamp(t, 0.0, 1.0)), quadbinStyle.range.z);
  } else if (mode == 2u) {
    let packed = quadbinData[row];
    let rgba = vec4<f32>(
      f32(packed & 255u), f32((packed >> 8u) & 255u), f32((packed >> 16u) & 255u), f32(packed >> 24u)
    ) / 255.0;
    if (rgba.a <= 0.0) { return output; }
    color = vec4<f32>(rgba.rgb, rgba.a * quadbinStyle.range.z);
  } else {
    var value = bitcast<f32>(quadbinData[row]);
    if (!isFiniteValue(value) || (quadbinStyle.config.w == 1u && value <= 0.0)) {
      color = vec4<f32>(quadbinStyle.noBaseline.rgb, quadbinStyle.noBaseline.a * quadbinStyle.range.z);
    } else {
      if (quadbinStyle.config.w == 1u) { value = log2(value); }
      let t = 0.5 + 0.5 * clamp(value / max(quadbinStyle.value.z, 1.0e-6), -1.0, 1.0);
      color = vec4<f32>(spatialAnalysisSampleRamp(quadbinStyle.config.y, t), quadbinStyle.range.z);
    }
  }
  var clip = project_position_to_clipspace(
    vec3<f32>(longitude, latitude, 0.0), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: QuadbinVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`,Oe=class extends n{static layerName=`QuadbinTableLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:G};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:Te,usage:r.UNIFORM|r.COPY_DST}),i=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST}),o=new a(t,{...this.getShaders({modules:[e],source:De}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(n,i),parameters:G});this.setState({model:o,styleBuffer:n,placeholder:i})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholder:r}=this.state;t.setBindings(this.getBindings(n,r))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state,r=this.props,i=new ArrayBuffer(Te),a=new Float32Array(i),o=new Uint32Array(i),[s,c]=r.densityLogRange??[0,4],[l,u]=r.meanRange??[0,1];a.set([s,c,r.opacity??.8,r.inset??.03],0),a.set([l,u,r.signedLimit??1,0],4),o.set([Ee[r.mode],L[r.ramp??`viridis`],r.rowCount?1:0,r.signedTransform===`log2`?1:0],8);let d=r.noBaselineColor??[150,150,150,200];a.set([d[0]/255,d[1]/255,d[2]/255,d[3]/255],12),n.write(new Uint8Array(i)),r.drawCommands?(t.setInstanceCount(0),t.draw(e),r.drawCommands.draw(e,0)):(t.setInstanceCount(r.instanceCount??0),t.draw(e))}finalizeState(e){let{model:t,styleBuffer:n,placeholder:r}=this.state;t.destroy(),n.destroy(),r.destroy(),super.finalizeState(e)}getBindings(e,t){let{props:n}=this;return{quadbinStyle:e,quadbinKeysEven:n.cellsEven,quadbinKeysOdd:n.cellsOdd??n.cellsEven,quadbinCountsEven:n.countsEven??t,quadbinCountsOdd:n.countsOdd??n.countsEven??t,quadbinData:n.values??n.colors??t,quadbinActiveLevel:n.activeLevel??t,quadbinFirstRow:n.firstRow??t,quadbinRowCount:n.rowCount??t}}},K=17,ke=4,q=K-ke+1,Ae=1<<18,je=15,Me=16384,Ne=32768,J=8,Pe=250,Fe=14,Ie=[[33,102,172],[103,169,207],[210,210,210],[239,138,98],[178,24,43]],Le=e=>K-e;function Re(e,t){return e===`delta`?Math.max(5,t):e===`percentChange`?100:e===`zScore`?4:2}async function ze(e){let{device:n}=e,r=fe(e.datasets.get(`chicago-nature`)),i=ce(e.datasets.get(`chicago-community-areas`)),a=r.count,o=new m(n,`cell-pyramid`),s=Math.min(Math.ceil(a/1024)*1024,Ae),c=s*Math.ceil(q/2),u=o.createBuffer(`lng-lat`,r.lngLat),d=o.createBuffer(`confirmed`,r.researchGrade),f=new Uint32Array(a).fill(1),h=o.createBuffer(`mask`,f),g=new Uint32Array(a),_=new Uint32Array(a),y=o.createBuffer(`before-mask`,g),b=o.createBuffer(`after-mask`,_),x=o.createBuffer(`area-outline`,i.outlineSegments),S=[0,1].map(e=>o.createBuffer(`cells-slab-${e}`,c*8)),C=[0,1].map(e=>o.createBuffer(`counts-slab-${e}`,c*4)),T=[0,1].map(e=>o.createBuffer(`table-counts-${e}`,q*4)),E=[0,1].map(e=>o.createBuffer(`level-overflow-${e}`,q*4)),D=o.createBuffer(`finest-sums`,s*8),O=o.createBuffer(`rollup-cells`,s*8),k=o.createBuffer(`rollup-counts`,s*4),A=o.createBuffer(`rollup-sums`,s*8),P=o.createBuffer(`rollup-sum-values`,s*4),F=o.createBuffer(`rollup-overflow`,4),I=o.createBuffer(`zero-word`,4),L=o.track(new v(n,{id:`cell-rollup-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),ae=o.createBuffer(`level-counts`,q*4),ue=o.createBuffer(`level-totals`,4),z=o.createParameterBuffer(`active-level`,`uint32`,1),B=o.createBuffer(`active-first-row`,4),V=o.track(new v(n,{id:`cell-pyramid-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]})),H=new l(n,{id:`cell-pyramid`}),xe=new Map,U=(e,t,n,r,i)=>{let a=xe.get(e)??H.importBuffer({id:e.id,byteLength:e.byteLength,usage:e.usage},e);return xe.set(e,a),H.createDataView(a,{format:t,length:i,byteOffset:r*n})},Se=Array.from({length:q},(e,t)=>{let n=t&1,r=(t>>1)*s;return{cells:U(S[n],`uint32x2`,8,r,s),counts:U(C[n],`uint32`,4,r,s),status:{count:U(T[n],`uint32`,4,t,1),overflow:U(E[n],`uint32`,4,t,1),...t===0?{requiredCount:U(ue,`uint32`,4,0,1)}:{}},...t===0?{sums:U(D,`uint32x2`,8,0,s)}:{}}}),W=new ye({id:`nature-pyramid`,family:`quadbin`,positions:R(H,`positions`,u,`float32x2`,a),mask:R(H,`mask`,h,`uint32`,a),values:R(H,`values`,d,`float32`,a),resolutions:Array.from({length:q},(e,t)=>Le(t)),output:{levels:Se,levelCounts:R(H,`level-counts`,ae,`uint32`,q)}});H.add(W);let Te=o.track(await H.compileAsync()),G=new l(n,{id:`cell-level-selection`}),Ee=V.importToGraph(G);G.add(new be({id:`nature-level`,levelCounts:R(G,`level-counts`,ae,`uint32`,q),activeLevel:z.importToGraph(G),levelFirstRows:W.levelFirstRows,output:{status:{count:G.createDataView(Ee.buffer,{format:`uint32`,length:1,byteOffset:Uint32Array.BYTES_PER_ELEMENT})},firstRow:R(G,`active-first-row`,B,`uint32`,1)}}));let De=o.track(await G.compileAsync()),ze=new Map,Be=async e=>{let t=ze.get(e);if(t)return t;let r=new l(n,{id:`cell-rollup-${e}`}),i=new Map,a=(e,t,n,a,o=0)=>{let s=i.get(e);return s||(s=r.importBuffer({id:e.id,byteLength:e.byteLength,usage:e.usage},e),i.set(e,s)),r.createDataView(s,{format:t,length:a,byteOffset:o*n})},c=L.importToGraph(r);r.add(new ve({id:`nature-rollup-${e}`,family:`quadbin`,sourceResolution:K,resolution:e,source:{cells:a(S[0],`uint32x2`,8,s),counts:a(C[0],`uint32`,4,s),sums:a(D,`uint32x2`,8,s),status:{count:a(T[0],`uint32`,4,1),overflow:a(E[0],`uint32`,4,1)}},sumScale:w,output:{cells:a(O,`uint32x2`,8,s),counts:a(k,`uint32`,4,s),sums:a(A,`uint32x2`,8,s),sumValues:a(P,`float32`,4,s),status:{count:r.createDataView(c.buffer,{format:`uint32`,length:1,byteOffset:Uint32Array.BYTES_PER_ELEMENT}),overflow:a(F,`uint32`,4,1)}}}));let u={graph:o.track(await r.compileAsync())};return ze.set(e,u),u},Ve=o.createBuffer(`class-values`,s*4),He=o.createBuffer(`class-mask`,s*4),Ue=o.createBuffer(`class-breaks`,(J+1)*4),We=o.createBuffer(`class-count`,4),Ge=o.createBuffer(`class-colors`,s*4),Ke=[`equal-interval`,`quantile`,`natural-breaks`,`standard-deviation`,`head-tail`,`box-plot`],qe=o.createParameterBuffer(`class-breaks-parameters`,`float32`,ee(J),j({method:`quantile`,classCount:5},J)),Je=o.createParameterBuffer(`class-color-scale-parameters`,`float32`,9,te({scale:`threshold`,domainCount:J+1,paletteCount:5,noDataColor:N(0,0,0,0)})),Ye=o.createBuffer(`class-palette`,J*4),Y=new Map,Xe=async t=>{let r=Y.get(t);if(r)return r;let i=new l(n,{id:`cell-classes-${t}`}),a=R(i,`rollup-counts`,k,`uint32`,s),c=R(i,`rollup-sum-values`,P,`float32`,s),u=L.importToGraph(i),d=i.createDataView(u.buffer,{format:`uint32`,length:1,byteOffset:Uint32Array.BYTES_PER_ELEMENT}),f=R(i,`class-values`,Ve,`float32`,s),m=R(i,`class-mask`,He,`uint32`,s);we(i,`classes-${t}`,t===`confirmed`?{counts:a,sumValues:c}:{counts:a},{values:f,mask:m},d);let h=R(i,`class-breaks`,Ue,`float32`,J+1),g=R(i,`class-count`,We,`uint32`,1);i.add(new M({id:`class-breaks-${t}`,values:f,mask:m,parameters:qe.importToGraph(i),maximumClassCount:J,methods:Ke,output:{breaks:h,classCount:g}})),i.add(new ne({id:`class-colors-${t}`,values:f,mask:m,domain:h,domainCount:g,palette:R(i,`class-palette`,Ye,`uint32`,J),parameters:Je.importToGraph(i),maximumDomainCount:J+1,maximumPaletteCount:J,output:{colors:R(i,`class-colors`,Ge,`uint32`,s)}}));let _=new se(o,`classes-${t}`,[{buffer:Ue,size:(J+1)*4},{buffer:We,size:4}],t=>{let n=new Float32Array(t,0,J+1),r=new Uint32Array(t,(J+1)*4,1)[0];n.slice();let i=Array.from(n.subarray(0,r+1)).map(e=>Number.isFinite(e)?p(e):`n/a`);e.setReadout(`classEdges`,`${r} classes: ${i.join(` | `)}`)}),v={graph:o.track(await i.compileAsync()),reader:_};return Y.set(t,v),v},Ze=()=>{let t=e.options,n=t.classMethod===`box-plot`?6:t.classCount;qe.write(j({method:t.classMethod,classCount:Math.min(n,J)},J)),Je.write(te({scale:`threshold`,domainCount:J+1,paletteCount:n,noDataColor:N(0,0,0,0)}));let r=new Uint32Array(J);for(let e=0;e<n;e++){let[i,a,o]=re(t.ramp,n===1?.5:e/(n-1));r[e]=N(i,a,o,235)}Ye.write(r)},X=new Map,Z=null,Qe=null,$e=20,et=e=>`${e.compareResolution}|${e.compareMeasure}|${e.zScoreKind}`,tt=t=>{let n=e.options,r=n.stableBand;t.classBreaks.write(n.changeClassing===`band`?j({method:`custom`,customEdges:[-1e9,-4*r,-r,r,4*r,1e9]},5):j({method:`standard-deviation`,classCount:5},5))},nt=async e=>{let t=et(e),r=X.get(t);if(r)return r;let i=e.compareResolution,o=e.compareMeasure===`sum`,s=new m(n,`cell-compare-${X.size}`);try{let r=new l(n,{id:`cell-compare-${t}`}),c=(e,t,n,i)=>{let a=s.createBuffer(e,Math.max(1,n)*t);return{buffer:a,view:R(r,e,a,i,n)}},f=Ne,p=c(`union-cells`,8,f,`uint32x2`),m=c(`delta`,4,f,`float32`),h=c(`ratio`,4,f,`float32`),g=c(`percent-change`,4,f,`float32`),_=c(`z-score`,4,f,`float32`),v=c(`before-value`,4,f,`float32`),x=c(`after-value`,4,f,`float32`),S=c(`union-count`,4,1,`uint32`),C=c(`union-overflow`,4,1,`uint32`),w=c(`colors`,4,f,`uint32`),T=c(`before-table-count`,4,1,`uint32`),E=c(`after-table-count`,4,1,`uint32`),D=s.createParameterBuffer(`compare-class-breaks`,`float32`,ee(5)),O=s.createParameterBuffer(`compare-color-scale`,`float32`,9,te({scale:`threshold`,domainCount:6,paletteCount:5,noDataColor:N(0,0,0,0)})),k=s.createBuffer(`compare-palette`,Uint32Array.from(Ie,([e,t,n])=>N(e,t,n,235))),A=(e,t)=>R(r,t,e,`uint32`,a),j=R(r,`positions`,u,`float32x2`,a),M=o?R(r,`values`,d,`float32`,a):void 0;Ce(r,{id:`period`,family:`quadbin`,resolution:i,tableCapacity:Me,unionCapacity:f,before:{positions:j,values:M,mask:A(y,`before-mask`)},after:{positions:j,values:M,mask:A(b,`after-mask`)},scratch:{before:{status:{count:T.view}},after:{status:{count:E.view}}},measure:e.compareMeasure,zScore:e.zScoreKind,outputs:{comparison:{cells:p.view,before:v.view,after:x.view,delta:m.view,ratio:h.view,percentChange:g.view,zScore:_.view,status:{count:S.view,overflow:C.view}},colors:w.view},classify:`delta`,classBreaksParameters:D.importToGraph(r),maximumClassCount:5,methods:[`custom`,`standard-deviation`],palette:R(r,`palette`,k,`uint32`,5),colorScaleParameters:O.importToGraph(r),maximumPaletteCount:5});let ne=s.track(await r.compileAsync());if(_t)return s.destroy(),null;let P=new se(s,`cell-compare-${t}`,[{buffer:S.buffer,size:4},{buffer:C.buffer,size:4},{buffer:T.buffer,size:4},{buffer:E.buffer,size:4},{buffer:p.buffer,size:f*8},{buffer:v.buffer,size:f*4},{buffer:x.buffer,size:f*4},{buffer:m.buffer,size:f*4},{buffer:_.buffer,size:f*4}],e=>{let t=new Uint32Array(e),n=Math.min(t[0],f);Z={rows:n,cells:t.slice(4,4+n*2),before:new Float32Array(e.slice((4+f*2)*4,(4+f*3)*4)),after:new Float32Array(e.slice((4+f*3)*4,(4+f*4)*4)),delta:new Float32Array(e.slice((4+f*4)*4,(4+f*5)*4)),zScore:new Float32Array(e.slice((4+f*5)*4,(4+f*6)*4)),beforeCells:t[2],afterCells:t[3],overflow:t[1]!==0},Qe=null,it()}),F={key:t,resolution:i,resources:s,graph:ne,classBreaks:D,colorScale:O,cells:p.buffer,count:S.buffer,colors:w.buffer,columns:{delta:m.buffer,ratio:h.buffer,percentChange:g.buffer,zScore:_.buffer},reader:P};return tt(F),X.set(t,F),F}catch(e){throw s.destroy(),e}},rt=(e,t)=>{let[n,r]=de(e,t),a=le(i,n,r);return`${a>=0?i.names[a]:`outside the city`} (${r.toFixed(3)}, ${n.toFixed(3)})`},it=()=>{let t=Z;if(!t)return;let n=e.options,r=n.compareMeasure===`sum`?`confirmed`:`observations`,i=n.stableBand,a=0,o=0,s=0,c=0,l=0,u=-1,d=0,f=-1,m=0;for(let e=0;e<t.rows;e++){let n=t.delta[e];s+=t.before[e],c+=t.after[e],n>i?a++:n<-i&&o++,n>l&&(l=n,u=e),n<d&&(d=n,f=e),Math.abs(t.zScore[e])>2&&m++}$e=Math.max(Math.abs(l),Math.abs(d),5),e.setReadout(`compareCells`,`${p(t.beforeCells)} before, ${p(t.afterCells)} after, ${p(t.rows)} in the union${t.overflow?` (OVERFLOW)`:``}`),e.setReadout(`compareTotals`,`${p(s)} to ${p(c)} ${r} (${s>0?((c-s)/s*100).toFixed(1):`n/a`}%)`),e.setReadout(`compareGainLoss`,`${p(a)} cells up and ${p(o)} down by more than ${i} ${r}`),e.setReadout(`compareSignificant`,`${p(m)} cells with |z| > 2 (${n.zScoreKind})`),e.setReadout(`compareLargestGain`,u>=0?`+${p(l)} ${r} in ${rt(t.cells[u*2],t.cells[u*2+1])}`:`none`),e.setReadout(`compareLargestLoss`,f>=0?`${p(d)} ${r} in ${rt(t.cells[f*2],t.cells[f*2+1])}`:`none`),ot(),e.requestLayers()},at=()=>{let t=e.options;return Re(t.changeColumn,$e)*t.colorLimit/100},ot=()=>{e.setLegendExtent(`change`,[-at(),at()])},st=!0,Q=!0,$=!0,ct=!0,lt=new Uint32Array(i.names.length+1),ut=()=>{let t=e.options,n=pe(r,f,{category:t.category,months:[1,12],sampledPercent:t.sampledPercent});h.write(f);let o=_e(r,t.category),[s,c]=t.periodA,[l,u]=t.periodB,d=0,m=0;lt=new Uint32Array(i.names.length+1);for(let e=0;e<a;e++){let n=r.month[e],i=o<0||r.category[e]===o;i&&t.sampledPercent<100&&(i=me(e)*100<t.sampledPercent);let a=i&&n>=s&&n<=c?1:0,p=i&&n>=l&&n<=u?1:0;g[e]=a,_[e]=p,d+=a,m+=p,f[e]&&lt[r.communityArea[e]]++}y.write(g),b.write(_),st=!0,ct=!0,e.setReadout(`points`,`${p(n)} of ${p(a)}`),e.setReadout(`periods`,`${he(t.periodA)}: ${p(d)} observations, ${he(t.periodB)}: ${p(m)} observations`)},dt=0,ft=new se(o,`pyramid`,[{buffer:ae,size:q*4},{buffer:E[0],size:q*4},{buffer:E[1],size:q*4},{buffer:ue,size:4},{buffer:V.buffer,size:8}],t=>{let n=new Uint32Array(t),r=n.subarray(0,q),i=[n.subarray(q,q*2),n.subarray(q*2,q*3)],a=n[q*3],o=(e,t)=>{let n=[];for(let i=e;i<=t;i++)n.push(`${Le(i)}: ${p(r[i])}`);return n.join(`, `)};e.setReadout(`levelsFine`,`res ${o(0,4)}`),e.setReadout(`levelsMiddle`,`res ${o(5,9)}`),e.setReadout(`levelsCoarse`,`res ${o(10,q-1)}`);let s=[];for(let e=0;e<q;e++)i[e&1][e]&&s.push(Le(e));e.setReadout(`overflow`,s.length?`res ${s.join(`, `)} (unclamped ${p(a)})`:`none`),e.options.view===`pyramid`&&e.setReadout(`activeCells`,`${p(n[q*3+2])} cells at res ${Le(dt)}`)}),pt=(e,t)=>{let n=0n;for(let r=0;r<t;r++)n+=BigInt.asIntN(64,BigInt(e[r*2+1])<<32n|BigInt(e[r*2]));return n},mt=(e,t)=>{let n=0;for(let r=0;r<t;r++)n+=e[r];return n},ht=Fe,gt=new se(o,`rollup-totals`,[{buffer:T[0],size:4},{buffer:E[0],size:4},{buffer:L.buffer,size:8},{buffer:F,size:4},{buffer:C[0],size:s*4},{buffer:D,size:s*8},{buffer:k,size:s*4},{buffer:A,size:s*8}],t=>{let n=new Uint32Array(t,0,5),r=n[0],i=n[3],a=(e,n)=>new Uint32Array(t,e,n),o=mt(a(20,s),r),c=pt(a(20+s*4,s*2),r),l=mt(a(20+s*12,s),i),u=pt(a(20+s*16,s*2),i),d=e=>Number(e)/w;e.setReadout(`rollupCells`,`${p(i)} cells at res ${ht} (from ${p(r)} at res ${K})${n[1]||n[4]?` (OVERFLOW)`:``}`),e.setReadout(`rollupCounts`,`${p(o)} to ${p(l)} observations ${o===l?`(conserved exactly)`:`(DIFFERS)`}`),e.setReadout(`rollupSums`,`${p(d(c))} to ${p(d(u))} research-grade observations ${c===u?`(bit-exact)`:`(DIFFERS)`}`)}),_t=!1,vt,yt=o.track(new oe),bt=()=>(vt=void 0,_t?Promise.resolve(!1):yt.request({build:async()=>{let{view:t,rollupResolution:n,classValue:r}=e.options,i=e.options;return(t===`rollup`||t===`classes`)&&await Be(n),t===`classes`&&await Xe(r),t===`compare`&&await nt(i),{view:t,rollupResolution:n}},commit:({view:t,rollupResolution:n})=>{(t===`rollup`||t===`classes`)&&(ht=n,Q=!0),t===`classes`&&(Ze(),$=!0),t===`compare`&&(ct=!0),e.requestLayers()}})),xt=()=>{clearTimeout(vt),vt=setTimeout(()=>void bt(),Pe)};ut(),e.setReadout(`pyramid`,`${q} levels, res ${K} to ${ke}, ${p(s)} rows each`),await bt();let St=()=>X.get(et(e.options))??null,Ct=0,wt=0,Tt=()=>{if(!Z)return null;if(!Qe){Qe=new Map;for(let e=0;e<Z.rows;e++){let{x:t,y:n}=ge(Z.cells[e*2],Z.cells[e*2+1]);Qe.set(n*1048576+t,e)}}return Qe};return{getCompiledGraphs:()=>{let t=[Te,De],n=e.options;if(n.view===`rollup`||n.view===`classes`){let e=ze.get(n.rollupResolution);e&&t.push(e.graph)}if(n.view===`classes`){let e=Y.get(n.classValue);e&&t.push(e.graph)}if(n.view===`compare`){let e=St();e&&t.push(e.graph)}return t},setOption(t){switch(t){case`category`:case`sampledPercent`:case`periodA`:case`periodB`:ut(),Q=!0,$=!0;break;case`view`:case`rollupResolution`:case`classValue`:case`compareResolution`:case`compareMeasure`:case`zScoreKind`:xt(),e.requestLayers();break;case`classMethod`:case`classCount`:Ze(),$=!0;break;case`ramp`:Ze(),$=!0,e.requestLayers();break;case`stableBand`:case`changeClassing`:{let e=St();e&&(tt(e),ct=!0),it();break}case`changeColumn`:case`colorLimit`:ot(),e.requestLayers();break;default:e.requestLayers()}},encode(t,n){let r=e.options,i=r.view;if(i!==`compare`&&st&&(Te.encode(t,{parameters:void 0}),st=!1,Q=!0,Ct++,e.setReadout(`encodes`,`${Ct} pyramid, ${wt} roll-up`),ft.markStale()),i===`pyramid`){let i=n.viewport.zoom,a=r.manualLevel?r.manualResolution:Math.round(i)+r.resolutionOffset,o=Math.min(K,Math.max(ke,a));dt=K-o,z.write(Uint32Array.of(dt)),e.setReadout(`zoom`,`${i.toFixed(2)} (map) to res ${o} (cells)`),De.encode(t,{parameters:void 0}),n.frameIndex%je===1&&ft.markStale()}if((i===`rollup`||i===`classes`)&&Q&&!st){let n=ze.get(r.rollupResolution);n&&(n.graph.encode(t,{parameters:void 0}),Q=!1,$=!0,wt++,e.setReadout(`encodes`,`${Ct} pyramid, ${wt} roll-up`),gt.markStale())}if(i===`classes`&&$&&!Q){let e=Y.get(r.classValue);e&&(e.graph.encode(t,{parameters:void 0}),$=!1,e.reader.markStale())}if(i===`compare`&&ct){let e=St();e&&(e.graph.encode(t,{parameters:void 0}),ct=!1,e.reader.markStale())}ft.flush(t),(i===`rollup`||i===`classes`)&&gt.flush(t),i===`classes`&&Y.get(r.classValue)?.reader.flush(t),i===`compare`&&St()?.reader.flush(t)},getLayers(){let n=e.options,r=[];if(n.view===`pyramid`)r.push(new Oe({id:`cell-pyramid-cells`,cellsEven:S[0],cellsOdd:S[1],countsEven:C[0],countsOdd:C[1],activeLevel:z.buffer,firstRow:B,drawCommands:V,mode:`density`,ramp:n.ramp,densityLogRange:[0,n.densityCeiling]}));else if(n.view===`rollup`)r.push(new Oe({id:`cell-rollup-cells`,cellsEven:O,countsEven:k,values:P,activeLevel:I,firstRow:I,drawCommands:L,mode:n.rollupColor===`researchShare`?`meanValue`:`density`,ramp:n.ramp,densityLogRange:[0,n.densityCeiling],meanRange:[0,n.rateCeiling/100],inset:.04}));else if(n.view===`classes`)r.push(new Oe({id:`cell-class-cells`,cellsEven:O,colors:Ge,activeLevel:I,firstRow:I,drawCommands:L,mode:`packed`,inset:.04,opacity:.9}));else{let e=St();if(e){let t=n.changeColumn===`classes`;r.push(new Oe({id:`cell-compare-cells-${e.key}-${n.changeColumn}`,cellsEven:e.cells,colors:e.colors,values:t?null:e.columns[n.changeColumn],rowCount:e.count,instanceCount:Ne,mode:t?`packed`:`signed`,ramp:`diverging`,signedLimit:at(),signedTransform:n.changeColumn===`ratio`?`log2`:`linear`,inset:.04,opacity:.88}))}}return n.showAreas&&r.push(new ie({id:`cell-pyramid-areas`,coordinateSystem:t.LNGLAT,segments:x,instanceCount:i.outlineSegments.length/4,widthPixels:1.2,color:[255,255,255,140]})),r},getTooltip(t){if(!t.coordinate)return null;let[n,r]=t.coordinate,a=le(i,n,r),o=a>=0?i.names[a]:null,s=e.options;if(s.view===`compare`&&Z){let e=Tt(),t=2**s.compareResolution,i=Math.floor((n+180)/360*t),a=Math.sin(r*Math.PI/180),c=Math.floor((.5-Math.log((1+a)/(1-a))/(4*Math.PI))*t),l=e?.get(c*1048576+i);if(l!==void 0){let e=Z.before[l],t=Z.after[l],n=s.compareMeasure===`sum`?`confirmed`:`observations`,r=Z.delta[l];return`${o??`Outside the city`}: ${p(e)} to ${p(t)} ${n} (${r>=0?`+`:``}${p(r)}${e>0?`, ${((t-e)/e*100).toFixed(0)}%`:``}, z ${Z.zScore[l].toFixed(1)})`}}return o?`${o}: ${p(lt[a+1]??0)} observations in the current filter`:null},destroy(){_t=!0,clearTimeout(vt),ft.stop(),gt.stop();for(let e of Y.values())e.reader.stop();for(let e of X.values())e.reader.stop(),e.resources.destroy();X.clear(),o.destroy()}}}export{ze as createCellPyramid};