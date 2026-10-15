import{a as e,c as t,i as n,o as r,s as i}from"./cell-topology-wgsl-BT8_s49b.js";import{_ as a,h as o,j as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-scan-BuGhTLDQ.js";import{t as d}from"./gpu-uint32-gather-9Dbzzdrh.js";import{t as f}from"./gpu-sort-Bl3Dztdb.js";import{r as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{u as h}from"./wgsl-common-BNRVzgd8.js";import{i as g,n as _,r as v}from"./cell-key-wgsl-BQM8miC7.js";function y(e){return[e.cells,e.counts,e.sums,e.sumValues,e.minimums,e.maximums,e.status.count,e.status.overflow,e.status.requiredCount]}function b(e){return e.cells.byteOffset/e.cells.rowByteLength}function x(e,t,n){l(n.cells,[`uint32x2`],`${e} ${t}.cells`);let r=n.cells.length;if(r<1)throw Error(`${e} ${t}.cells must hold at least one row`);s(n.counts,`${e} ${t}.counts`),n.sums&&l(n.sums,[`uint32x2`],`${e} ${t}.sums`);for(let r of[`sumValues`,`minimums`,`maximums`]){let i=n[r];i&&l(i,[`float32`],`${e} ${t}.${r}`)}for(let i of[`counts`,`sums`,`sumValues`,`minimums`,`maximums`]){let a=n[i];if(a&&a.length!==r)throw Error(`${e} ${t}.${i} must have the same length as ${t}.cells`)}for(let r of[`count`,`overflow`,`requiredCount`]){let i=n.status[r];if(i&&(s(i,`${e} ${t}.status.${r}`),i.length<1))throw Error(`${e} ${t}.status.${r} must contain one uint32 row`)}}var S=256;function C(e){return e.width+1>32}function w(e,t,n){return C(e)?`(${t} >> ${e.width-32}u) != 0u`:`(${n} >> ${e.width}u) != 0u`}function T(e,t){let{id:r,operation:i,layout:a,output:o,source:s}=t,l=t.keyLow.length,h=o.cells.length,g=C(a),y=(t,n)=>c(e,`${r}-${t}`,`uint32`,n),b=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),x=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),T=[],E=t.keyLow,D=t.keyHigh,O;if(!t.sorted){if(!t.rowIds)throw Error(`${r} needs row IDs to sort`);let n=g?32:a.width+1,i=y(`sorted-low-first`,l),o=y(`sorted-rows-first`,l);if(T.push(...new f({id:`${r}-sort-low`,keys:t.keyLow,values:t.rowIds,outputKeys:i,outputValues:o,keyBits:n}).getCommandNodes(e)),g&&t.keyHigh){let n=y(`high-by-low-order`,l);T.push(...new d({id:`${r}-gather-high`,source:t.keyHigh,indices:o,output:n}).getCommandNodes(e)),D=y(`sorted-high`,l),O=y(`permutation`,l),T.push(...new f({id:`${r}-sort-high`,keys:n,values:o,outputKeys:D,outputValues:O,keyBits:a.width+1-32}).getCommandNodes(e)),E=y(`sorted-low`,l),T.push(...new d({id:`${r}-gather-low`,source:t.keyLow,indices:O,output:E}).getCommandNodes(e))}else E=i,O=o}let k=()=>[b(`sortedLow`,E),...g&&D?[b(`sortedHigh`,D)]:[]],A=e=>g?`vec2u(sortedHigh[sortedHighOffset + ${e}], sortedLow[sortedLowOffset + ${e}])`:`vec2u(0u, sortedLow[sortedLowOffset + ${e}])`,j=y(`heads`,l),M=y(`valid-count`,1);T.push(m(e,{id:`${r}-heads`,operation:i,variant:`heads`,bindings:[...k(),x(`heads`,j),x(`validCount`,M)],invocationCount:l,declarations:`const ROW_COUNT: u32 = ${l}u;
fn isInvalidKey(key: vec2u) -> bool {
  return ${w(a,`key.x`,`key.y`)};
}`,body:`let key = ${A(`index`)};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${A(`index - 1u`)});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${A(`index + 1u`)}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`}));let N=y(`cell-indices`,l);T.push(...new u({id:`${r}-scan`,input:j,output:N,mode:`exclusive`}).getCommandNodes(e));let P=y(`offsets`,h+1),F=y(`cell-total`,1),I=y(`row-cells`,l);T.push(m(e,{id:`${r}-offsets`,operation:i,variant:`offsets`,bindings:[b(`heads`,j),b(`cellIndices`,N),b(`validCount`,M),x(`offsets`,P),x(`cellTotal`,F),x(`rowCells`,I)],invocationCount:l,declarations:`const ROW_COUNT: u32 = ${l}u;
const CAPACITY: u32 = ${h}u;`,body:`let isHead = heads[headsOffset + index] != 0u;
  let cell = cellIndices[cellIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  rowCells[rowCellsOffset + index] = select(0xffffffffu, cell, isValid && cell < CAPACITY);
  if (isHead && cell <= CAPACITY) {
    offsets[offsetsOffset + cell] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = cellIndices[cellIndicesOffset + last] + heads[headsOffset + last];
    cellTotal[cellTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`}));let L=`const CAPACITY: u32 = ${h}u;
fn getCellCount() -> u32 {
  return min(cellTotal[cellTotalOffset], CAPACITY);
}`,R=s.kind===`cells`?s.table:void 0;T.push(m(e,{id:`${r}-cells`,operation:i,variant:`cells`,bindings:[b(`offsets`,P),b(`cellTotal`,F),...k(),x(`cellsOut`,o.cells),x(`countsOut`,o.counts)],invocationCount:h,declarations:`${L}
${n}`,body:`if (index >= getCellCount()) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = cellGetKey(${A(`begin`)}, ${a.headerHigh}u, ${a.resolution}u, ${a.lowBit}u);
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  // Roll-ups sum the child counts in the accumulation pass.
  countsOut[countsOutOffset + index] = ${R?`0u`:`offsets[offsetsOffset + index + 1u] - begin`};`}));let z=!!(o.sums||o.sumValues),B=!!(o.minimums||o.maximums),V=s.kind===`rows`?s.values:void 0;if((z||B)&&s.kind===`rows`&&!V)throw Error(`${r} needs values to compute sums and extremes`);if(z&&R&&!R.sums)throw Error(`${r} source table needs sums to roll up sums`);if(R&&(o.minimums&&!R.minimums||o.maximums&&!R.maximums))throw Error(`${r} source table needs the extremes it rolls up`);let H=z?o.sums??c(e,`${r}-sums-scratch`,`uint32x2`,h):void 0,U=o.minimums?y(`minimum-keys`,h):void 0,W=o.maximums?y(`maximum-keys`,h):void 0,G=O?`permutation[permutationOffset + index]`:`index`,K=()=>V?[...O?[b(`permutation`,O)]:[],b(`values`,V,`f32`)]:[];if((H||U||W)&&T.push(m(e,{id:`${r}-clear`,operation:i,variant:`clear`,bindings:[...H?[x(`sums`,H)]:[],...U?[x(`minimumKeys`,U)]:[],...W?[x(`maximumKeys`,W)]:[]],invocationCount:h,body:`${H?`sums[sumsOffset + 2u * index] = 0u;
  sums[sumsOffset + 2u * index + 1u] = 0u;`:``}
  ${U?`minimumKeys[minimumKeysOffset + index] = 0xffffffffu;`:``}
  ${W?`maximumKeys[maximumKeysOffset + index] = 0u;`:``}`})),H||R){let n=V?`cellScaleValue(values[valuesOffset + ${G}])`:`vec2u(childSums[childSumsOffset + 2u * index + 1u], childSums[childSumsOffset + 2u * index])`;T.push(m(e,{id:`${r}-accumulate-sums`,operation:i,variant:`accumulate-sums`,bindings:[b(`rowCells`,I),...K(),...R&&H?[b(`childSums`,R.sums)]:[],...R?[b(`childCounts`,R.counts),{name:`countsOut`,view:o.counts,type:`atomic<u32>`,access:`read_write`}]:[],...H?[{name:`sums`,view:H,type:`atomic<u32>`,access:`read_write`}]:[]],workgroupSize:S,invocationCount:l,guardIndex:!1,declarations:`${v(t.sumScale)}
const TILE_SIZE: u32 = ${S}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${S}>;
${H?`var<workgroup> tileHigh: array<u32, ${S}>;\nvar<workgroup> tileLow: array<u32, ${S}>;`:``}
${R?`var<workgroup> tileCounts: array<u32, ${S}>;`:``}`,body:`let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${H?`var value = vec2u(0u, 0u);`:``}
  ${R?`var childCount = 0u;`:``}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${H?`value = ${n};`:``}
      ${R?`childCount = childCounts[childCountsOffset + index];`:``}
    }
  }
  tileCells[lane] = cell;
  ${H?`tileHigh[lane] = value.x;
  tileLow[lane] = value.y;`:``}
  ${R?`tileCounts[lane] = childCount;`:``}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${H?`var addValue = vec2u(0u, 0u);`:``}
    ${R?`var addCount = 0u;`:``}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${H?`addValue = vec2u(tileHigh[lane - stride], tileLow[lane - stride]);`:``}
      ${R?`addCount = tileCounts[lane - stride];`:``}
    }
    workgroupBarrier();
    ${H?`value = cellAddI64(value, addValue);
    tileHigh[lane] = value.x;
    tileLow[lane] = value.y;`:``}
    ${R?`childCount += addCount;
    tileCounts[lane] = childCount;`:``}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${R?`atomicAdd(&countsOut[countsOutOffset + cell], childCount);`:``}
  ${H?`let previous = atomicAdd(&sums[sumsOffset + 2u * cell], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * cell + 1u], high);
  }`:``}`}))}if(U||W){let t=V?`values[valuesOffset + ${G}]`:`childMinimums[childMinimumsOffset + index]`,n=V?`values[valuesOffset + ${G}]`:`childMaximums[childMaximumsOffset + index]`;T.push(m(e,{id:`${r}-accumulate-extremes`,operation:i,variant:`accumulate-extremes`,bindings:[b(`rowCells`,I),...K(),...R&&U?[b(`childMinimums`,R.minimums,`f32`)]:[],...R&&W?[b(`childMaximums`,R.maximums,`f32`)]:[],...U?[{name:`minimumKeys`,view:U,type:`atomic<u32>`,access:`read_write`}]:[],...W?[{name:`maximumKeys`,view:W,type:`atomic<u32>`,access:`read_write`}]:[]],workgroupSize:S,invocationCount:l,guardIndex:!1,declarations:`${_}
const TILE_SIZE: u32 = ${S}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${S}>;
${U?`var<workgroup> tileMinimums: array<u32, ${S}>;`:``}
${W?`var<workgroup> tileMaximums: array<u32, ${S}>;`:``}`,body:`let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${U?`var minimumKey = 0xffffffffu;`:``}
  ${W?`var maximumKey = 0u;`:``}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${U?`minimumKey = getOrderedKey(${t});`:``}
      ${W?`maximumKey = getOrderedKey(${n});`:``}
    }
  }
  tileCells[lane] = cell;
  ${U?`tileMinimums[lane] = minimumKey;`:``}
  ${W?`tileMaximums[lane] = maximumKey;`:``}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${U?`var otherMinimum = 0xffffffffu;`:``}
    ${W?`var otherMaximum = 0u;`:``}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${U?`otherMinimum = tileMinimums[lane - stride];`:``}
      ${W?`otherMaximum = tileMaximums[lane - stride];`:``}
    }
    workgroupBarrier();
    ${U?`minimumKey = min(minimumKey, otherMinimum);
    tileMinimums[lane] = minimumKey;`:``}
    ${W?`maximumKey = max(maximumKey, otherMaximum);
    tileMaximums[lane] = maximumKey;`:``}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${U?`atomicMin(&minimumKeys[minimumKeysOffset + cell], minimumKey);`:``}
  ${W?`atomicMax(&maximumKeys[maximumKeysOffset + cell], maximumKey);`:``}`}))}return(o.sumValues||U||W)&&T.push(m(e,{id:`${r}-finish`,operation:i,variant:`finish`,bindings:[b(`cellTotal`,F),...o.sumValues&&H?[b(`sums`,H)]:[],...U?[b(`minimumKeys`,U)]:[],...W?[b(`maximumKeys`,W)]:[],...o.sumValues?[x(`sumValuesOut`,o.sumValues,`f32`)]:[],...o.minimums&&U?[x(`minimumsOut`,o.minimums,`f32`)]:[],...o.maximums&&W?[x(`maximumsOut`,o.maximums,`f32`)]:[]],invocationCount:h,declarations:`${L}
${_}
${v(t.sumScale)}`,body:`let isOccupied = index < getCellCount();
  let nan = getNaN();
  ${o.sumValues?`let total = vec2u(sums[sumsOffset + 2u * index + 1u], sums[sumsOffset + 2u * index]);
  sumValuesOut[sumValuesOutOffset + index] = cellI64ToF32(total) / SUM_SCALE;`:``}
  ${o.minimums&&U?`minimumsOut[minimumsOutOffset + index] = select(nan, decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), isOccupied);`:``}
  ${o.maximums&&W?`maximumsOut[maximumsOutOffset + index] = select(nan, decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), isOccupied);`:``}`})),T.push(p(e,{id:`${r}-publish`,operation:i,requiredCount:F,output:{ids:o.counts,status:{count:o.status.count,overflow:o.status.overflow,requiredCount:o.status.requiredCount}},overflowSources:t.overflowSources,extraCounts:t.extraCounts})),T}var E=`GPUCellAggregation`,D=class{id;props;layout;sumScale;spatialContext;constructor(e){this.id=e.id??`cell-aggregation`,this.props=e;let t=this.id;if(e.family!==`quadbin`&&e.family!==`h3`)throw Error(`${t} family must be 'quadbin' or 'h3'`);if(this.layout=r(e.family,e.resolution),this.spatialContext=i(t,e.family,e.spatialContext,e.positions?`points`:`cells`,e.resolution),this.sumScale=e.sumScale??65536,g(t,this.sumScale),!!e.positions==!!e.cells)throw Error(`${t} needs exactly one of positions or cells`);if(e.positions&&e.family!==`quadbin`)throw Error(`${t} positions are supported for the quadbin family only`);let n=(e.positions??e.cells).length;if(n<1)throw Error(`${t} needs at least one row`);e.positions&&l(e.positions,[`float32x2`],`${t} positions`),e.cells&&l(e.cells,[`uint32x2`],`${t} cells`),e.values&&l(e.values,[`float32`],`${t} values`),e.mask&&l(e.mask,[`uint32`],`${t} mask`);for(let[r,i]of[[`values`,e.values],[`mask`,e.mask]])if(i&&i.length!==n)throw Error(`${t} ${r} length must equal the row count`);x(t,`output`,e.output);let{output:a}=e;if(!e.values&&(a.sums||a.sumValues||a.minimums||a.maximums))throw Error(`${t} needs values for sums and extremes`);o(t,[...y(a),...e.extraCounts??[]],[e.positions,e.cells,e.values,e.mask])}getCommandNodes(r){let{id:i,props:o,layout:s}=this;a(i,r,[o.positions,o.cells,o.values,o.mask,...y(o.output),...o.extraCounts??[]]);let l=(o.positions??o.cells).length,u=C(s),d=c(r,`${i}-key-low`,`uint32`,l),f=u?c(r,`${i}-key-high`,`uint32`,l):void 0,p=c(r,`${i}-row-ids`,`uint32`,l),g=[o.positions?{name:`positions`,view:o.positions,type:`f32`,access:`read`}:{name:`cells`,view:o.cells,type:`u32`,access:`read`},...o.mask?[{name:`rowMask`,view:o.mask,type:`u32`,access:`read`}]:[],...o.values?[{name:`values`,view:o.values,type:`f32`,access:`read`}]:[],{name:`keyLow`,view:d,type:`u32`,access:`read_write`},...f?[{name:`keyHigh`,view:f,type:`u32`,access:`read_write`}]:[],{name:`rowIds`,view:p,type:`u32`,access:`read_write`}],_=`cellShiftLeft(vec2u(0u, 1u), ${s.width}u)`,v=o.positions?`let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  if (isValid && !isNanBits(longitude) && !isNanBits(latitude)) {
    let x = quadbinGetTileX(longitude, RESOLUTION);
    let y = quadbinGetTileY(latitude, RESOLUTION);
    compact = quadbinGetCompactKey(x, y);
  }`:`let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${o.wordOrder===`high-low`?`words`:`words.yx`};
  ${o.family===`h3`?`let isValidKey = dggs_h3_is_valid_cell_id(key) && dggs_h3_get_resolution(key) >= RESOLUTION;`:`let isValidKey = cellIsValidQuadbin(key) && ((key.x >> 20u) & 0x1fu) >= RESOLUTION;`}
  if (isValid && isValidKey) {
    compact = cellGetCompactKey(key, ${s.lowBit}u, ${s.width}u);
  }`;return[m(r,{id:`${i}-keys`,operation:E,variant:o.positions?`quadbin-points`:`${o.family}-cells`,bindings:g,invocationCount:l,declarations:`const RESOLUTION: u32 = ${s.resolution}u;
${o.cells&&o.family===`h3`?t.source:``}
${n}
${o.positions?e:``}
${h}`,body:`var isValid = true;
  ${o.mask?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  ${o.values?`let value = values[valuesOffset + index];
  isValid = isValid && (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;`:``}
  var compact = ${_};
  ${v}
  keyLow[keyLowOffset + index] = compact.y;
  ${f?`keyHigh[keyHighOffset + index] = compact.x;`:``}
  rowIds[rowIdsOffset + index] = index;`}),...T(r,{id:i,operation:E,layout:s,keyLow:d,keyHigh:f,rowIds:p,sorted:!1,source:{kind:`rows`,values:o.values},sumScale:this.sumScale,output:o.output,extraCounts:o.extraCounts})]}};export{C as a,y as i,b as n,x as o,T as r,D as t};