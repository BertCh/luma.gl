import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./earth-xB16XgAa.js";import"./analysis-frame-Blu0tYdw.js";import{i as a}from"./convert-colors-D-TZjU1C.js";import{c as o,i as s,o as c}from"./cell-topology-wgsl-BT8_s49b.js";import{N as l,_ as u,h as d,n as f,t as p,w as m,y as h}from"./resources-CwvhEOBe.js";import{t as g}from"./gpu-scan-BuGhTLDQ.js";import{t as _}from"./gpu-sort-Bl3Dztdb.js";import{t as v}from"./draw-command-buffer-BeqRkhBa.js";import{n as y,s as b}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as x,r as S,t as C}from"./gpu-cell-aggregation-BFqjxJvj.js";import{t as w}from"./scan-total-BdolCtK7.js";import{n as T,t as E}from"./gpu-cell-geometry-PcTj7e7c.js";import{t as D}from"./gpu-point-to-cell-DabDU1o1.js";import{t as ee}from"./gpu-cell-set-outline-B1MxUkL7.js";import{n as O,t as k}from"./gpu-cell-topology-6n6YnUDd.js";import{c as A,m as j,y as M}from"./index-Bjk18V2I.js";import{t as N}from"./graph-buffers-B4_80RlX.js";import{t as P}from"./graph-swap-41Ru7LVu.js";import{t as te}from"./summary-reader-CsgWxENf.js";import{n as ne,t as re}from"./b8-areas-CiQcUCr6.js";import{t as F}from"./mode-kernels-Dxm2GWvI.js";import{a as ie,d as ae,i as oe,o as se,u as ce}from"./b8-common-CgT2-JC9.js";var I=`GPUCellCompaction`,L=11;function R(){let e=[1],t=[1];for(let n=1;n<=L;n++)e.push(e[n-1]*7),t.push(t[n-1]*7-1);return{hexagon:e,pentagon:t}}function z(e,t){let n=R(),r=(e,t)=>`const ${e} = array<u32, ${t.length}>(${t.map(e=>`${e}u`).join(`, `)});`;return`
${e===`h3`?o.source:``}
${s}
const BITS_PER_LEVEL: u32 = ${e===`quadbin`?2:3}u;
const HEADER_HIGH: u32 = ${c(e,0).headerHigh}u;
${e===`h3`?r(`H3_HEXAGON_SIZES`,n.hexagon):``}
${e===`h3`?r(`H3_PENTAGON_SIZES`,n.pentagon):``}

fn getLowBit(resolution: u32) -> u32 {
  return ${e===`quadbin`?`52u - 2u * resolution`:`45u - 3u * resolution`};
}

fn getInputKey(words: vec2u) -> vec2u {
  return ${t===`high-low`?`words`:`words.yx`};
}

fn getKeyResolution(key: vec2u) -> u32 {
  return ${e===`quadbin`?`(key.x >> 20u) & 0x1fu`:`dggs_h3_get_resolution(key)`};
}

fn isValidKey(key: vec2u) -> bool {
  return ${e===`quadbin`?`cellIsValidQuadbin(key)`:`dggs_h3_is_valid_cell_id(key)`};
}

/** Number of descendants of a cell with this compact key, \`depth\` levels down; saturates. */
fn getDescendantCount(compact: vec2u, resolution: u32, depth: u32) -> u32 {
  ${e===`quadbin`?`return select(0xffffffffu, 1u << (2u * depth), depth < 16u);`:`if (depth > ${L}u) {
    return 0xffffffffu;
  }
  let pathBits = 3u * resolution;
  let baseCell = cellShiftRight(compact, pathBits).y;
  let isPentagon = dggs_h3_is_base_cell_pentagon(baseCell) &&
    all((compact & cellMaskLow(pathBits)) == vec2u(0u));
  return select(H3_HEXAGON_SIZES[depth], H3_PENTAGON_SIZES[depth], isPentagon);`}
}
`}var le=class{id;props;layout;constructor(e){this.id=e.id??`cell-compaction`,this.props=e;let{id:t}=this,{operation:n,output:r}=e;if(this.layout=c(e.family,n.resolution),n.type===`compact`){let e=n.minimumResolution??0;if(!Number.isInteger(e)||e<0||e>n.resolution)throw Error(`${t} minimumResolution must be an integer in [0, resolution]`)}else if(n.type===`uncompact`){let r=e.family===`quadbin`?15:L,i=n.maximumDepth??8;if(!Number.isInteger(i)||i<0||i>r)throw Error(`${t} maximumDepth must be an integer in [0, ${r}]`)}else throw Error(`${t} operation.type must be 'compact' or 'uncompact'`);if(e.family!==`quadbin`&&e.family!==`h3`)throw Error(`${t} family must be 'quadbin' or 'h3'`);if(e.wordOrder&&e.wordOrder!==`little-endian`&&e.wordOrder!==`high-low`)throw Error(`${t} wordOrder must be 'little-endian' or 'high-low'`);if(h(e.cells,[`uint32x2`],`${t} cells`),e.cells.length<1)throw Error(`${t} cells must hold at least one row`);if(e.mask&&(h(e.mask,[`uint32`],`${t} mask`),e.mask.length!==e.cells.length))throw Error(`${t} mask must have the same length as cells`);for(let[n,i]of[[`count`,e.count],[`output.status.count`,r.status.count],[`output.status.overflow`,r.status.overflow],[`output.status.requiredCount`,r.status.requiredCount],[`output.status.droppedCount`,r.status.droppedCount]])if(i&&(h(i,[`uint32`],`${t} ${n}`),i.length<1))throw Error(`${t} ${n} must contain one uint32 row`);if(h(r.cells,[`uint32x2`],`${t} output.cells`),r.cells.length<1)throw Error(`${t} output.cells must hold at least one row`);d(t,[r.cells,r.status.count,r.status.overflow,r.status.requiredCount,r.status.droppedCount],[e.cells,e.mask,e.count]);let i=[r.cells,r.status.count,r.status.overflow,r.status.requiredCount,r.status.droppedCount].filter(Boolean);if(new Set(i).size!==i.length)throw Error(`${t} output views must be distinct`)}getCommandNodes(e){let{props:t}=this;return u(this.id,e,[t.cells,t.mask,t.count,t.output.cells,t.output.status.count,t.output.status.overflow,t.output.status.requiredCount,t.output.status.droppedCount]),t.operation.type===`compact`?this.getCompactNodes(e,t.operation):this.getUncompactNodes(e,t.operation)}getInputBindings(){let{props:e}=this,t=[{name:`cells`,view:e.cells,type:`u32`,access:`read`}];return e.mask&&t.push({name:`mask`,view:e.mask,type:`u32`,access:`read`}),e.count&&t.push({name:`rowCount`,view:e.count,type:`u32`,access:`read`}),{bindings:t,rowCountWGSL:e.count?`rowCount[rowCountOffset]`:`${e.cells.length}u`,maskWGSL:e.mask?`mask[maskOffset + index] != 0u`:`true`}}getCompactNodes(e,t){let{id:n,props:r,layout:i}=this,{output:a}=r,o=r.cells.length,s=t.minimumResolution??0,c=t.resolution-s,l=t.sorted??!1,u=x(i),d=(t,r=o)=>m(e,`${n}-${t}`,`uint32`,r),f=[],p=this.getInputBindings(),h=z(r.family,r.wordOrder??`little-endian`);a.status.droppedCount&&f.push(y(e,{id:`${n}-clear-dropped`,operation:I,view:a.status.droppedCount,type:`u32`,value:`0u`,componentCount:1}));let v=d(`key-low`),C=u?d(`key-high`):void 0,w=l?void 0:d(`row-ids`);f.push(b(e,{id:`${n}-keys`,operation:I,variant:`compact-keys`,bindings:[...p.bindings,{name:`keyLow`,view:v,type:`u32`,access:`read_write`},...C?[{name:`keyHigh`,view:C,type:`u32`,access:`read_write`}]:[],...w?[{name:`rowIds`,view:w,type:`u32`,access:`read_write`}]:[],...a.status.droppedCount?[{name:`droppedCount`,view:a.status.droppedCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:o,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
${h}`,body:`var compact = cellShiftLeft(vec2u(0u, 1u), ${i.width}u);
  if (index < ${p.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    if (${p.maskWGSL} && isValidKey(key) && getKeyResolution(key) == RESOLUTION) {
      compact = cellGetCompactKey(key, ${i.lowBit}u, ${i.width}u);
    } else {
      ${a.status.droppedCount?`atomicAdd(&droppedCount[droppedCountOffset], 1u);`:``}
    }
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${C?`keyHigh[keyHighOffset + index] = compact.x;`:``}
  ${w?`rowIds[rowIdsOffset + index] = index;`:``}`}));let T=m(e,`${n}-unique-table`,`uint32x2`,o),E=d(`unique-count`,1);f.push(...S(e,{id:`${n}-table`,operation:I,layout:i,keyLow:v,keyHigh:C,rowIds:w,sorted:l,source:{kind:`rows`},sumScale:65536,output:{cells:T,counts:d(`unique-counts`),status:{count:E,overflow:d(`unique-overflow`,1)}}}));let D=d(`flags`),ee=d(`depths`);f.push(b(e,{id:`${n}-analyze`,operation:I,variant:`analyze`,bindings:[{name:`cells`,view:T,type:`u32`,access:`read`},{name:`uniqueCount`,view:E,type:`u32`,access:`read`},{name:`flags`,view:D,type:`u32`,access:`read_write`},{name:`depths`,view:ee,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
const MAXIMUM_DEPTH: u32 = ${c}u;
${h}

fn getCompactAt(row: u32) -> vec2u {
  return cellGetCompactKey(
    vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]), ${i.lowBit}u, ${i.width}u);
}

fn getAncestor(compact: vec2u, depth: u32) -> vec2u {
  return cellShiftRight(compact, BITS_PER_LEVEL * depth);
}

fn isLess(a: vec2u, b: vec2u) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

/** First row whose ancestor is >= (or, for upper bounds, > ) \`ancestor\`. */
fn findBound(ancestor: vec2u, depth: u32, rowCount: u32, isUpper: bool) -> u32 {
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) >> 1u;
    let value = getAncestor(getCompactAt(middle), depth);
    let goRight = select(isLess(value, ancestor), !isLess(ancestor, value), isUpper);
    if (goRight) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let rowCount = uniqueCount[uniqueCountOffset];
  if (index >= rowCount) {
    flags[flagsOffset + index] = 0u;
    depths[depthsOffset + index] = 0u;
    return;
  }
  let compact = getCompactAt(index);
  var bestDepth = 0u;
  var bestStart = index;
  // Completeness is monotone: a complete ancestor implies complete descendants.
  for (var depth = 1u; depth <= MAXIMUM_DEPTH; depth++) {
    let ancestor = getAncestor(compact, depth);
    let lower = findBound(ancestor, depth, rowCount, false);
    let upper = findBound(ancestor, depth, rowCount, true);
    if (upper - lower != getDescendantCount(ancestor, RESOLUTION - depth, depth)) {
      break;
    }
    bestDepth = depth;
    bestStart = lower;
  }
  flags[flagsOffset + index] = select(0u, 1u, bestStart == index);
  depths[depthsOffset + index] = bestDepth;`}));let O=d(`positions`);f.push(...new g({id:`${n}-scan`,input:D,output:O,mode:`exclusive`}).getCommandNodes(e));let k=m(e,`${n}-emitted`,`uint32x2`,o),A=d(`sort-keys`),j=d(`sort-ids`),M=d(`total`,1);f.push(b(e,{id:`${n}-scatter`,operation:I,variant:`scatter`,bindings:[{name:`cells`,view:T,type:`u32`,access:`read`},{name:`flags`,view:D,type:`u32`,access:`read`},{name:`depths`,view:ee,type:`u32`,access:`read`},{name:`positions`,view:O,type:`u32`,access:`read`},{name:`emitted`,view:k,type:`u32`,access:`read_write`},{name:`sortKeys`,view:A,type:`u32`,access:`read_write`},{name:`sortIds`,view:j,type:`u32`,access:`read_write`},{name:`total`,view:M,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
const ROW_COUNT: u32 = ${o}u;
${h}`,body:`let emitCount = positions[positionsOffset + ROW_COUNT - 1u] + flags[flagsOffset + ROW_COUNT - 1u];
  if (index == 0u) {
    total[totalOffset] = emitCount;
  }
  if (index >= emitCount) {
    // Padding rows sort last (resolutions are below 31).
    sortKeys[sortKeysOffset + index] = 31u;
    sortIds[sortIdsOffset + index] = index;
  }
  if (flags[flagsOffset + index] != 0u) {
    let depth = depths[depthsOffset + index];
    let compact = cellGetCompactKey(
      vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]), ${i.lowBit}u, ${i.width}u);
    let resolution = RESOLUTION - depth;
    let key = cellGetKey(cellShiftRight(compact, BITS_PER_LEVEL * depth), HEADER_HIGH, resolution, getLowBit(resolution));
    let slot = positions[positionsOffset + index];
    emitted[emittedOffset + 2u * slot] = key.y;
    emitted[emittedOffset + 2u * slot + 1u] = key.x;
    sortKeys[sortKeysOffset + slot] = resolution;
    sortIds[sortIdsOffset + slot] = slot;
  }`}));let N=d(`ordered-keys`),P=d(`ordered-ids`);return f.push(...new _({id:`${n}-sort-resolution`,keys:A,values:j,outputKeys:N,outputValues:P,keyBits:5}).getCommandNodes(e)),f.push(b(e,{id:`${n}-emit`,operation:I,variant:`emit`,bindings:[{name:`total`,view:M,type:`u32`,access:`read`},{name:`orderedIds`,view:P,type:`u32`,access:`read`},{name:`emitted`,view:k,type:`u32`,access:`read`},{name:`cellsOut`,view:a.cells,type:`u32`,access:`read_write`},{name:`countOut`,view:a.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:a.status.overflow,type:`u32`,access:`read_write`},...a.status.requiredCount?[{name:`totalOut`,view:a.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:a.cells.length,declarations:`const CAPACITY: u32 = ${a.cells.length}u;`,body:`let totalCells = total[totalOffset];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${a.status.requiredCount?`totalOut[totalOutOffset] = totalCells;`:``}
  }
  var low = 0u;
  var high = 0u;
  if (index < count) {
    let slot = orderedIds[orderedIdsOffset + index];
    low = emitted[emittedOffset + 2u * slot];
    high = emitted[emittedOffset + 2u * slot + 1u];
  }
  cellsOut[cellsOutOffset + 2u * index] = low;
  cellsOut[cellsOutOffset + 2u * index + 1u] = high;`})),f}getUncompactNodes(e,t){let{id:n,props:r,layout:i}=this,{output:a}=r,o=r.cells.length,s=t.maximumDepth??8,c=(t,r=o)=>m(e,`${n}-${t}`,`uint32`,r),l=[],u=this.getInputBindings(),d=z(r.family,r.wordOrder??`little-endian`),f=`const RESOLUTION: u32 = ${t.resolution}u;
const MAXIMUM_DEPTH: u32 = ${s}u;
${d}

fn getCompact(key: vec2u, resolution: u32) -> vec2u {
  let lowBit = getLowBit(resolution);
  return cellGetCompactKey(key, lowBit, 52u - lowBit);
}`;a.status.droppedCount&&l.push(y(e,{id:`${n}-clear-dropped`,operation:I,view:a.status.droppedCount,type:`u32`,value:`0u`,componentCount:1}));let p=c(`counts`);l.push(b(e,{id:`${n}-count`,operation:I,variant:`uncompact-count`,bindings:[...u.bindings,{name:`counts`,view:p,type:`u32`,access:`read_write`},...a.status.droppedCount?[{name:`droppedCount`,view:a.status.droppedCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:o,declarations:f,body:`var descendants = 0u;
  if (index < ${u.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    var keep = false;
    if (${u.maskWGSL} && isValidKey(key)) {
      let resolution = getKeyResolution(key);
      if (resolution <= RESOLUTION && RESOLUTION - resolution <= MAXIMUM_DEPTH) {
        keep = true;
        descendants = getDescendantCount(getCompact(key, resolution), resolution, RESOLUTION - resolution);
      }
    }
    if (!keep) {
      ${a.status.droppedCount?`atomicAdd(&droppedCount[droppedCountOffset], 1u);`:``}
    }
  }
  counts[countsOffset + index] = descendants;`}));let h=c(`offsets`);l.push(...new g({id:`${n}-scan`,input:p,output:h,mode:`exclusive`}).getCommandNodes(e));let _=c(`scan-total`,1);return l.push(...w(e,{id:n,operation:I,counts:p,offsets:h,rowCount:o,total:_})),l.push(b(e,{id:`${n}-expand`,operation:I,variant:`uncompact-expand`,bindings:[{name:`cells`,view:r.cells,type:`u32`,access:`read`},{name:`scanTotal`,view:_,type:`u32`,access:`read`},{name:`offsets`,view:h,type:`u32`,access:`read`},{name:`cellsOut`,view:a.cells,type:`u32`,access:`read_write`},{name:`countOut`,view:a.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:a.status.overflow,type:`u32`,access:`read_write`},...a.status.requiredCount?[{name:`totalOut`,view:a.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:a.cells.length,declarations:`const CAPACITY: u32 = ${a.cells.length}u;
const ROW_COUNT: u32 = ${o}u;
${f}

/** The \`rank\`-th (ascending) descendant path \`depth\` levels below \`compact\`. */
fn getDescendant(compact: vec2u, resolution: u32, depth: u32, rankIn: u32) -> vec2u {
  ${r.family===`quadbin`?`return cellShiftLeft(compact, 2u * depth) | vec2u(0u, rankIn);`:`var rank = rankIn;
  var digits = vec2u(0u);
  var isChain = dggs_h3_is_base_cell_pentagon(cellShiftRight(compact, 3u * resolution).y) &&
    all((compact & cellMaskLow(3u * resolution)) == vec2u(0u));
  for (var step = 0u; step < depth; step++) {
    let remaining = depth - 1u - step;
    var digit = 0u;
    if (isChain) {
      // Pentagon: digit 0 keeps the pentagon, digit 1 is deleted, digits 2..6 are hexagons.
      if (rank >= H3_PENTAGON_SIZES[remaining]) {
        rank = rank - H3_PENTAGON_SIZES[remaining];
        digit = 2u + rank / H3_HEXAGON_SIZES[remaining];
        rank = rank % H3_HEXAGON_SIZES[remaining];
        isChain = false;
      }
    } else {
      digit = rank / H3_HEXAGON_SIZES[remaining];
      rank = rank % H3_HEXAGON_SIZES[remaining];
    }
    digits = cellShiftLeft(digits, 3u) | vec2u(0u, digit);
  }
  return cellShiftLeft(compact, 3u * depth) | digits;`}
}`,body:`let totalCells = scanTotal[scanTotalOffset];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${a.status.requiredCount?`totalOut[totalOutOffset] = totalCells;`:``}
  }
  var result = vec2u(0u);
  if (index < count) {
    // Last row whose offset is <= index; zero-count rows share the next row's offset.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) >> 1u;
      if (offsets[offsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let row = low - 1u;
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * row], cells[cellsOffset + 2u * row + 1u]));
    let resolution = getKeyResolution(key);
    let depth = RESOLUTION - resolution;
    let compact = getDescendant(getCompact(key, resolution), resolution, depth, index - offsets[offsetsOffset + row]);
    result = cellGetKey(compact, HEADER_HIGH, RESOLUTION, ${i.lowBit}u);
  }
  cellsOut[cellsOutOffset + 2u * index] = result.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = result.x;`})),l}},B=192,V=8,H={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},U={...H,blendColorDstFactor:`one`},W={uniform:0,ramp:1,category:2,distance:3},G=`
${M()}

struct CellStyle {
  palette: array<vec4<f32>, ${V}>,
  baseColor: vec4<f32>,
  // valueMaximum, filterMaximum, widthPixels, opacity
  numbers: vec4<f32>,
  // stride, colorMode, mode (0 fill, 1 outline), useRowCount
  config: vec4<u32>,
  // colormap index, sqrt scale, unused, unused
  config2: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> cellStyle: CellStyle;
@group(0) @binding(auto) var<storage, read> cellBoundaries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> cellVertexCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> cellValues: array<u32>;
@group(0) @binding(auto) var<storage, read> cellRowCounts: array<u32>;

struct CellVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenCell() -> CellVertexOutput {
  var output: CellVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

fn projectCellPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Alpha 0 means hidden.
fn getCellColor(row: u32) -> vec4<f32> {
  let colorMode = cellStyle.config.y;
  if (colorMode == 0u) {
    return cellStyle.baseColor;
  }
  let raw = cellValues[row];
  if (colorMode == 1u) {
    if (raw == 0u) { return vec4<f32>(0.0); }
    var t = clamp(f32(raw) / max(cellStyle.numbers.x, 1.0), 0.0, 1.0);
    if (cellStyle.config2.y != 0u) { t = sqrt(t); }
    return vec4<f32>(spatialAnalysisSampleRamp(cellStyle.config2.x, t), cellStyle.baseColor.a);
  }
  if (colorMode == 2u) {
    return cellStyle.palette[raw % ${V}u];
  }
  // Distance: hide beyond the filter, fade with distance.
  if (f32(raw) > cellStyle.numbers.y) { return vec4<f32>(0.0); }
  let fade = 1.0 - 0.55 * f32(raw) / max(cellStyle.numbers.y, 1.0);
  return vec4<f32>(cellStyle.baseColor.rgb, cellStyle.baseColor.a * fade);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) row: u32
) -> CellVertexOutput {
  let stride = cellStyle.config.x;
  let vertexTotal = cellVertexCounts[row];
  if (vertexTotal < 3u) { return getHiddenCell(); }
  if (cellStyle.config.w != 0u && row >= cellRowCounts[0]) { return getHiddenCell(); }
  let color = getCellColor(row);
  if (color.a <= 0.0) { return getHiddenCell(); }

  var output: CellVertexOutput;
  output.color = color;
  output.side = 0.0;
  if (cellStyle.config.z == 0u) {
    // Fill: triangle fan from vertex 0.
    let triangle = vertexIndex / 3u;
    let corner = vertexIndex % 3u;
    if (triangle + 2u >= vertexTotal) { return getHiddenCell(); }
    var index = 0u;
    if (corner > 0u) { index = triangle + corner; }
    output.position = projectCellPosition(cellBoundaries[row * stride + index]);
    return output;
  }
  // Outline: one screen-space quad per edge.
  let edge = vertexIndex / 6u;
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  if (edge >= vertexTotal) { return getHiddenCell(); }
  let corner = corners[vertexIndex % 6u];
  let startClip = projectCellPosition(cellBoundaries[row * stride + edge]);
  let endClip = projectCellPosition(cellBoundaries[row * stride + (edge + 1u) % vertexTotal]);
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * cellStyle.numbers.z * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: CellVertexOutput) -> @location(0) vec4<f32> {
  var coverage = 1.0;
  if (cellStyle.config.z == 1u) {
    coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  }
  return vec4<f32>(input.color.rgb, input.color.a * cellStyle.numbers.w * coverage);
}
`,K=class extends n{static layerName=`CellBoundaryLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:H};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:B,usage:r.UNIFORM|r.COPY_DST}),i=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST}),o=this.props.additive?U:H,s=new a(t,{...this.getShaders({modules:[e],source:G}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.getVertexCount(),instanceCount:0,bufferLayout:[],bindings:this.getBindings(n,i),parameters:o});this.setState({model:s,styleBuffer:n,placeholder:i})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholder:r}=this.state;t.setBindings(this.getBindings(n,r)),t.setVertexCount(this.getVertexCount())}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state,r=new ArrayBuffer(B),i=new Float32Array(r),a=new Uint32Array(r),o=this.props.palette??[];for(let e=0;e<V;e++){let t=o[e%Math.max(o.length,1)]??[255,255,255,255];i.set([t[0]/255,t[1]/255,t[2]/255,(t[3]??255)/255],e*4)}let s=this.props.color??[255,255,255,255];i.set([s[0]/255,s[1]/255,s[2]/255,s[3]/255],32),i.set([this.props.valueMaximum??1,this.props.filterMaximum??1e9,this.props.widthPixels??1.5,this.props.opacity??1],36),a.set([this.props.stride,W[this.props.colorMode??`uniform`],this.props.mode===`outline`?1:0,this.props.rowCounts?1:0],40),a.set([j[this.props.ramp??`viridis`],this.props.sqrtScale?1:0],44),n.write(new Uint8Array(r)),t.setInstanceCount(this.props.instanceCount),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n,placeholder:r}=this.state;t.destroy(),n.destroy(),r.destroy(),super.finalizeState(e)}getVertexCount(){let{stride:e,mode:t}=this.props;return t===`outline`?e*6:Math.max(e-2,1)*3}getBindings(e,t){return{cellStyle:e,cellBoundaries:this.props.boundaries,cellVertexCounts:this.props.vertexCounts,cellValues:this.props.values??t,cellRowCounts:this.props.rowCounts??t}}},ue=160,q=`
struct RingStyle {
  palette: array<vec4<f32>, ${V}>,
  // widthPixels, opacity, unused, unused
  numbers: vec4<f32>,
  // ringCount, unused, unused, unused
  config: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> ringStyle: RingStyle;
@group(0) @binding(auto) var<storage, read> ringOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> ringCounts: array<u32>;
@group(0) @binding(auto) var<storage, read> ringPositions: array<vec2<f32>>;

struct RingOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};

fn getHiddenRing() -> RingOutput {
  var output: RingOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

fn projectRingPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Instance = ring vertex slot. Its ring is the last ring whose offset is at or before the slot.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) slot: u32
) -> RingOutput {
  let ringTotal = ringStyle.config.x;
  if (slot >= ringOffsets[ringTotal]) { return getHiddenRing(); }
  var low = 0u;
  var high = ringTotal;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (ringOffsets[middle] <= slot) { low = middle; } else { high = middle; }
  }
  let ring = low;
  let count = ringCounts[ring];
  let first = ringOffsets[ring];
  if (count < 2u || slot >= first + count) { return getHiddenRing(); }
  var next = slot + 1u;
  if (next >= first + count) { next = first; }
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let startClip = projectRingPosition(ringPositions[slot]);
  let endClip = projectRingPosition(ringPositions[next]);
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * ringStyle.numbers.x * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: RingOutput;
  output.position = clipPosition;
  let color = ringStyle.palette[ring % ${V}u];
  output.color = vec4<f32>(color.rgb, color.a * ringStyle.numbers.y);
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: RingOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`,de=class extends n{static layerName=`RingOutlineLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:H};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:ue,usage:r.UNIFORM|r.COPY_DST}),i=new a(t,{...this.getShaders({modules:[e],source:q}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(n),parameters:H});this.setState({model:i,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state,r=new ArrayBuffer(ue),i=new Float32Array(r),a=new Uint32Array(r),{palette:o}=this.props;for(let e=0;e<V;e++){let t=o[e%o.length];i.set([t[0]/255,t[1]/255,t[2]/255,t[3]/255],e*4)}i.set([this.props.widthPixels??2,this.props.opacity??1,0,0],32),a.set([this.props.ringCapacity,0,0,0],36),n.write(new Uint8Array(r)),t.setInstanceCount(this.props.slotCount),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}getBindings(e){return{ringStyle:e,ringOffsets:this.props.ringOffsets,ringCounts:this.props.ringCounts,ringPositions:this.props.ringPositions}}},J={h3:{label:`H3`,defaultResolution:8,minimumResolution:0,maximumResolution:15,tabular:!0,maximumVertexCount:10,exactness:`f32 vs h3-js: exact at res 0-4, ~4e-4 of points off by one cell at res 8, 1e-3 at 9`},quadbin:{label:`Quadbin`,defaultResolution:15,minimumResolution:0,maximumResolution:26,tabular:!0,maximumVertexCount:4,exactness:`integer keys: bit-exact`},quadkey:{label:`Quadkey`,defaultResolution:15,minimumResolution:1,maximumResolution:29,tabular:!1,maximumVertexCount:4,exactness:`integer keys: bit-exact (1.3% off by one tile at zoom 26)`},geohash:{label:`Geohash`,defaultResolution:6,minimumResolution:1,maximumResolution:12,tabular:!1,maximumVertexCount:4,exactness:`integer keys: bit-exact for f32 inputs`},s2:{label:`S2`,defaultResolution:14,minimumResolution:0,maximumResolution:30,tabular:!1,maximumVertexCount:4,exactness:`f32 vs f64: exact to level 10, ~5e-4 at 12, 2.6e-3 at 15 (use <= 20)`}};function fe(e,t){let n=J[e];return Math.min(n.maximumResolution,Math.max(n.minimumResolution,n.defaultResolution+t))}var pe=[1107.7,418.7,158.2,59.81,22.61,8.544,3.229,1.221,.461,.174,.0659,.0249,.0094,.0036,.0013,5e-4],Y=[`5000 x 5000 km`,`1250 x 625 km`,`156 x 156 km`,`39 x 19.5 km`,`4.9 x 4.9 km`,`1.2 x 0.6 km`,`153 x 153 m`,`38 x 19 m`,`4.8 x 4.8 m`,`1.2 x 0.6 m`,`15 x 15 cm`,`4 x 2 cm`];function me(e,t){let n=e=>e>=1?`${e.toFixed(2)} km`:`${Math.round(e*1e3)} m`;return e===`h3`?`edge ~${n(pe[t])}`:e===`geohash`?`${Y[t-1]}`:e===`s2`?`edge ~${n(Math.sqrt(5101e5/(6*4**t)))}`:`tile ~${n(2*Math.PI*i/1e3/2**t*Math.cos(41.88*Math.PI/180))} wide`}var X=65536,Z=65536,he=1<<20,ge=8,_e=250,Q=1<<17,$=4096,ve=[[90,110,190,255],[70,190,210,255],[110,220,130,255],[250,210,80,255],[255,110,90,255]],ye=[[255,150,60,215],[60,215,140,215]],be=[[255,255,255,255],[255,214,64,255],[120,230,255,255],[255,150,200,255]],xe=[[78,201,255,190],[255,148,72,190],[189,122,255,190],[87,235,168,190],[255,105,168,190],[245,220,87,190],[107,158,255,190],[255,120,100,190]],Se=[40,220,255,150];async function Ce(e){let{device:n}=e,r=ae(e.datasets.get(`chicago-nature`)),i=ne(e.datasets.get(`chicago-community-areas`)),a=r.count,o=new p(n,`h3-aggregation`),s=o.createBuffer(`lng-lat`,r.lngLat),c=new Uint32Array(a).fill(1),u=o.createBuffer(`mask`,c),d=o.createBuffer(`area-positions`,i.vertices),m=o.createBuffer(`area-feature-offsets`,i.featureOffsets),h=o.createBuffer(`area-polygon-offsets`,i.polygonOffsets),g=o.createBuffer(`area-ring-offsets`,i.ringOffsets),_=o.createBuffer(`area-outline`,i.outlineSegments),y=o.createBuffer(`selection-position`,Float32Array.of(-87.6325,41.9625)),b=a,x=new Uint32Array(i.names.length+1),S=1,w=0,j=null,M=null,I=new Map,L=null,R,z=!1,B=!1,V=!1,H=!1,U=new Set,W=null,G=null,ue=new Uint32Array(i.names.length),q=null,pe=null,Y=(e,t,n,r,i)=>N(e,`import-${t}`,n,r,i),Ce=()=>{let t=e.options;b=oe(r,c,{category:t.category,months:t.months,sampledPercent:t.sampledPercent}),u.write(c),x=new Uint32Array(i.names.length+1);for(let e=0;e<a;e++)c[e]&&x[r.communityArea[e]]++;j&&(j.dirty=!0),e.setReadout(`points`,`${f(b)} of ${f(a)} (${se(t.months)})`)},we=(e,t,n,r,i)=>(t(r*i*8+r*4),{boundaries:e.createBuffer(`${n}-boundaries`,r*i*8),vertexCounts:e.createBuffer(`${n}-vertex-counts`,r*4),stride:i,rows:r}),Te=(e,t,n,r,i)=>{e.add(new E({id:`${t}-geometry`,family:n,cells:r,maximumVertexCount:i.stride,output:{boundaries:Y(e,`${t}-boundaries`,i.boundaries,`float32x2`,i.rows*i.stride),vertexCounts:Y(e,`${t}-vertex-counts`,i.vertexCounts,`uint32`,i.rows)}}))},Ee=async(e,t)=>{let{tabular:r,maximumVertexCount:i}=J[e],o=++w,c=new p(n,`h3-index-${o}`),d=0,f=e=>{d+=e},m=(e,t)=>(d+=t,c.createBuffer(e,t)),h=r?X:a,g=m(`raw-cells`,a*8),_=r?m(`table-cells`,X*8):null,y=r?m(`table-counts`,X*4):null,b=r?m(`table-count`,4):null,x=r?m(`table-overflow`,4):null,S=r?m(`table-total`,4):null,T=we(c,f,`cells`,h,i),E=new l(n,{id:`h3-index-${o}`}),O=Y(E,`raw-cells`,g,`uint32x2`,a);E.add(new D({id:`point-to-cell`,family:e,resolution:t,positions:Y(E,`positions`,s,`float32x2`,a),mask:Y(E,`mask`,u,`uint32`,a),output:{cells:O}}));let k=O;if(r&&_&&y&&b&&x&&S){let n=Y(E,`table-cells`,_,`uint32x2`,X);E.add(new C({id:`aggregate`,family:e,resolution:t,cells:O,output:{cells:n,counts:Y(E,`table-counts`,y,`uint32`,X),status:{count:Y(E,`table-count`,b,`uint32`,1),overflow:Y(E,`table-overflow`,x,`uint32`,1),requiredCount:Y(E,`table-total`,S,`uint32`,1)}}})),k=n}Te(E,`cell`,e,k,T);let A=c.track(await E.compileAsync()),j=[],M=[],N=null,P=null;if(r&&_&&y&&b){let t=async(t,r,i)=>{try{let a=new l(n,{id:`h3-${t}-${o}`}),s=(e,n,r,i)=>Y(a,`${t}-${e}`,n,r,i),u=s(`table-cells`,_,`uint32x2`,X),d=s(`table-count`,b,`uint32`,1),f;r&&(f=s(`table-groups`,m(`${t}-table-groups`,X*4),`uint32`,X),F(a,{id:`${t}-count-tiers`,invocationCount:X,bindings:[{name:`counts`,view:s(`table-counts`,y,`uint32`,X),type:`u32`,access:`read`},{name:`groups`,view:f,type:`u32`,access:`read_write`}],body:`let value = counts[countsOffset + index];
  var tier = 0u;
  if (value > 0u) { tier = min(firstLeadingBit(value), 4u); }
  groups[groupsOffset + index] = tier;`}));let p=m(`${t}-endpoints`,Q*16),h=r?m(`${t}-groups`,Q*4):null,g=m(`${t}-count`,4),x=m(`${t}-overflow`,4),S=null,C;if(i){let e=m(`${t}-ring-offsets`,($+1)*4),n=m(`${t}-ring-counts`,$*4);S={offsets:e,counts:n,positions:m(`${t}-ring-positions`,Q*8),count:m(`${t}-ring-count`,4),overflow:m(`${t}-ring-overflow`,4),openSegments:m(`${t}-ring-open`,4)};let r=s(`ring-offsets`,e,`uint32`,$+1);C={normalizeWinding:!0,output:{ringOffsets:r,positions:s(`ring-positions`,S.positions,`float32x2`,Q),status:{count:s(`ring-count`,S.count,`uint32`,1),overflow:s(`ring-overflow`,S.overflow,`uint32`,1),openSegmentCount:s(`ring-open`,S.openSegments,`uint32`,1)}}},F(a,{id:`${t}-ring-counts`,invocationCount:$,bindings:[{name:`offsets`,view:r,type:`u32`,access:`read`},{name:`counts`,view:s(`ring-vertex-counts`,n,`uint32`,$),type:`u32`,access:`read_write`}],body:`counts[countsOffset + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];`})}a.add(new ee({id:t,family:e,cells:u,count:d,groups:f,rings:C,output:{rows:s(`rows`,m(`${t}-rows`,Q*4),`uint32`,Q),cells:s(`cells`,m(`${t}-cells`,Q*8),`uint32x2`,Q),edgeIndices:s(`edges`,m(`${t}-edges`,Q*4),`uint32`,Q),endpoints:s(`endpoints`,p,`float32x4`,Q),groups:h?s(`groups`,h,`uint32`,Q):void 0,status:{count:s(`count`,g,`uint32`,1),overflow:s(`overflow`,x,`uint32`,1)}}}));let w=c.track(await a.compileAsync()),T=c.track(new v(n,{id:`h3-${o}-${t}-draw`,type:`draw`,commands:[{vertexCount:6,instanceCount:0}]}));return j.push(w),{rings:S,endpoints:p,groups:h,count:g,overflow:x,drawCommands:T}}catch(e){return M.push(`${t}${i?` with rings`:``}: ${e instanceof Error?e.message:String(e)}`),null}};N=await t(`outline`,!1,!0),N||=await t(`outline`,!1,!1),P=await t(`group-outline`,!0,!1)}let ne=null;if(r&&y&&_&&b&&x&&S){let n=m(`zero-word`,4),r=[b,x,S,N?.count,N?.overflow,P?.count,P?.overflow,N?.rings?.count,N?.rings?.overflow,N?.rings?.openSegments];ne=new te(c,`h3-index-${o}`,[...r.map(e=>({buffer:e??n,size:4})),{buffer:y,size:X*4},{buffer:_,size:X*8}],n=>De(o,t,e,n))}return{serial:o,family:e,resolution:t,resources:c,indexGraph:A,outlineGraphs:j,outlineErrors:M,cells:T,tableCounts:y,tableCount:b,outline:N,groupOutline:P,reader:ne,byteLength:d,dirty:!0}},De=(t,n,r,i)=>{let a=j;if(!a||a.serial!==t||H)return;let o=new Uint32Array(i),s=Math.min(o[0],X),c=o.subarray(10,10+X),l=o.subarray(10+X,10+X*3),u=0,d=0;for(let e=0;e<s;e++)u+=c[e],c[e]>d&&(d=c[e]);G={cells:l.slice(0,s*2),counts:c.slice(0,s),rows:s},W=null,e.setReadout(`occupied`,`${f(s)} of ${f(X)}${o[1]?` (OVERFLOW: ${f(o[2])} cells, smallest keys kept)`:``}`),e.setReadout(`binned`,`${f(u)} observations (${(u/Math.max(s,1)).toFixed(1)} per cell)`),e.setReadout(`busiest`,`${f(d)} observations in one cell`),d!==S&&(S=Math.max(1,d),e.requestLayers()),Oe(),e.setReadout(`outlineSegments`,a.outline?`${f(o[3])} segments${o[4]?` (OVERFLOW, clamped)`:``}`:`unavailable`),e.setReadout(`groupBorders`,a.groupOutline?`${f(o[5])} segments (${f(Math.max(0,o[5]-o[3]))} more than the plain boundary)${o[6]?` (OVERFLOW)`:``}`:`unavailable`),e.setReadout(`rings`,a.outline?.rings?`${f(o[7])} rings, ${f(o[9])} open segments${o[8]?` (OVERFLOW)`:``}`:`unavailable`),Ie()},Oe=()=>{e.setLegendExtent(`cells`,[0,Math.max(1,Math.round(S*e.options.ceiling/100))])},ke=async(e,t,r,a)=>{let{maximumVertexCount:o}=J[e],s=++w,c=new p(n,`h3-zones-${s}`),u=0,f=e=>{u+=e},_=(e,t)=>(u+=t,c.createBuffer(e,t)),v=_(`cover-cells`,Z*8),y=_(`cover-feature-ids`,Z*4),b=_(`cover-core`,Z*4),x=_(`cover-count`,4),S=_(`cover-overflow`,4),C=_(`cover-candidate-overflow`,4),E=_(`cover-invalid-count`,4),D=_(`cover-uncertain-count`,4),ee=_(`cover-total`,4),O=_(`compact-cells`,Z*8),k=_(`compact-count`,4),A=_(`compact-overflow`,4),j=_(`uncompact-cells`,Z*8),M=_(`uncompact-count`,4),N=_(`uncompact-overflow`,4),P=_(`uncompact-dropped`,4),ne=we(c,f,`cover`,Z,o),re=we(c,f,`compact`,Z,o),F=new l(n,{id:`h3-zones-${s}`}),ie=Y(F,`cover-cells`,v,`uint32x2`,Z),ae=Y(F,`cover-count`,x,`uint32`,1),oe=Y(F,`compact-cells`,O,`uint32x2`,Z),se=Y(F,`compact-count`,k,`uint32`,1);F.add(new T({id:`cover`,family:e,resolution:t,containment:r,positions:Y(F,`area-positions`,d,`float32x2`,i.vertices.length/2),featureOffsets:Y(F,`area-feature-offsets`,m,`uint32`,i.featureOffsets.length),polygonOffsets:Y(F,`area-polygon-offsets`,h,`uint32`,i.polygonOffsets.length),ringOffsets:Y(F,`area-ring-offsets`,g,`uint32`,i.ringOffsets.length),candidateCapacity:he,output:{featureIds:Y(F,`cover-feature-ids`,y,`uint32`,Z),core:Y(F,`cover-core`,b,`uint32`,Z),cells:ie,status:{count:ae,requiredCount:Y(F,`cover-total`,ee,`uint32`,1),overflow:Y(F,`cover-overflow`,S,`uint32`,1),invalidCount:Y(F,`cover-invalid-count`,E,`uint32`,1),uncertainCount:Y(F,`cover-uncertain-count`,D,`uint32`,1),candidateOverflow:Y(F,`cover-candidate-overflow`,C,`uint32`,1)}}})),F.add(new le({id:`compact`,family:e,operation:{type:`compact`,resolution:t,minimumResolution:Math.max(0,t-a)},cells:ie,count:ae,output:{cells:oe,status:{count:se,overflow:Y(F,`compact-overflow`,A,`uint32`,1)}}})),F.add(new le({id:`uncompact`,family:e,operation:{type:`uncompact`,resolution:t,maximumDepth:a},cells:oe,count:se,output:{cells:Y(F,`uncompact-cells`,j,`uint32x2`,Z),status:{count:Y(F,`uncompact-count`,M,`uint32`,1),overflow:Y(F,`uncompact-overflow`,N,`uint32`,1),droppedCount:Y(F,`uncompact-dropped`,P,`uint32`,1)}}})),Te(F,`cover`,e,ie,ne),Te(F,`compact`,e,oe,re);let ce=c.track(await F.compileAsync()),I=new te(c,`h3-zones-${s}`,[{buffer:x,size:4},{buffer:S,size:4},{buffer:ee,size:4},{buffer:k,size:4},{buffer:A,size:4},{buffer:M,size:4},{buffer:N,size:4},{buffer:P,size:4},{buffer:E,size:4},{buffer:D,size:4},{buffer:b,size:Z*4},{buffer:y,size:Z*4}],e=>Ae(s,e));return{serial:s,resources:c,graph:ce,cover:{...ne,featureIds:y,core:b,count:x},compacted:{...re,count:k},reader:I,byteLength:u,dirty:!0}},Ae=(t,n)=>{let r=M;if(!r||r.serial!==t||H)return;let a=new Uint32Array(n),o=Math.min(a[0],Z),s=0,c=new Uint32Array(i.names.length);for(let e=0;e<o;e++){a[10+e]===1&&s++;let t=a[10+Z+e];t<c.length&&c[t]++}ue=c;let l=c.reduce((e,t)=>e+(t===0?1:0),0),u=a[8],d=a[9],p=[...u>0?[`${u} features rejected: ambiguous or over-wide rings`]:[],...d>0?[`${d} features beyond the polar lattice limit`]:[]];e.setReadout(`cover`,[`${f(o)} cells (${f(a[2])} unclamped${a[1]?`, OVERFLOW`:``}); ${l} of ${i.names.length} areas get no cell`,...p].join(`; `)),e.setReadout(`core`,`${f(s)} core / ${f(o-s)} border (${o>0?(100*s/o).toFixed(0):0}% core)`);let m=a[0]>0?(1-a[3]/a[0])*100:0;e.setReadout(`compacted`,`${f(a[3])} cells (${m.toFixed(0)}% fewer${a[4]?`, OVERFLOW`:``})`),e.setReadout(`roundTrip`,a[5]===a[0]&&a[7]===0?`uncompact gives back ${f(a[5])} cells: identical count to the cover`:`uncompact gives ${f(a[5])} cells (cover ${f(a[0])}, ${f(a[7])} dropped${a[6]?`, OVERFLOW`:``})`)},je=(e,t,n)=>n.selectionOperation===`disk`?{type:`disk`,k:ge}:n.selectionOperation===`ring`?{type:`ring`,k:n.ringRadius}:n.selectionOperation===`parent`?{type:`parent`,resolution:Math.max(0,t-n.levelsUp)}:{type:`children`,resolution:t+n.levelsDown,inputResolution:t},Me=(e,t)=>{let{selectionOperation:n}=e,r=n===`ring`?e.ringRadius:n===`parent`?e.levelsUp:n===`children`?e.levelsDown:ge;return`${e.family}|${t}|${n}|${r}`},Ne=async(e,t,r)=>{let{maximumVertexCount:i}=J[e],a=je(e,t,r),o=O(e,a),s=++w,c=new p(n,`h3-selection-${s}`),u=()=>{},d=c.createBuffer(`selected-cell`,8),f=c.createBuffer(`result-cells`,o*8),m=a.type===`disk`?c.createBuffer(`result-distances`,o*4):null,h=c.createBuffer(`result-counts`,4),g=we(c,u,`result`,o,i),_=we(c,u,`selected`,1,i),v=new l(n,{id:`h3-selection-${s}`}),b=Y(v,`selected-cell`,d,`uint32x2`,1),x=Y(v,`result-cells`,f,`uint32x2`,o);v.add(new D({id:`selection-key`,family:e,resolution:t,positions:Y(v,`selection-position`,y,`float32x2`,1),output:{cells:b}})),v.add(new k({id:a.type,family:e,operation:a,cells:b,output:{cells:x,distances:m?Y(v,`result-distances`,m,`uint32`,o):void 0,counts:Y(v,`result-counts`,h,`uint32`,1)}})),Te(v,`result`,e,x,g),Te(v,`selected`,e,b,_);let S=c.track(await v.compileAsync()),C=new te(c,`h3-selection-${s}`,[{buffer:d,size:8},{buffer:h,size:4},{buffer:f,size:o*8}],e=>Pe(s,e));return{serial:s,key:Me(r,t),operation:r.selectionOperation,stride:o,resources:c,graph:S,result:{...g,distances:m},resultCells:f,selected:_,reader:C,dirty:!0}},Pe=(e,t)=>{let n=L;!n||n.serial!==e||H||(q=new Uint32Array(t),pe=n,Ie())},Fe=()=>{if(!G)return null;if(!W){W=new Map;for(let e=0;e<G.rows;e++)W.set(ce(G.cells[e*2],G.cells[e*2+1]),G.counts[e])}return W},Ie=()=>{let t=q,n=pe;if(!t||!n||n!==L)return;let r=Fe(),i=ce(t[0],t[1]),a=r?.get(i);e.setReadout(`selected`,`${ie(t[0],t[1])}${a===void 0?r?`: no observations`:``:`: ${f(a)} observations`}`);let o=Math.min(t[2],n.stride),s=e=>ce(t[3+e*2],t[4+e*2]);if(n.operation===`parent`||n.operation===`children`)e.setReadout(`selection`,`${f(o)} ${n.operation===`parent`?`parent cell`:`child cells`} (a different resolution than the table, so no observation count)`);else if(r){let t=0,i=0;for(let e=0;e<o;e++){let n=r.get(s(e));n&&(t+=n,i++)}e.setReadout(`selection`,`${f(o)} ${n.operation===`ring`?`ring`:`disk`} cells, ${f(i)} with observations, ${f(t)} observations`)}},Le=e=>{if(!e)return;e.reader?.stop(),U.add(e);let t=0,n=()=>{++t<4?requestAnimationFrame(n):U.delete(e)&&e.resources.destroy()};requestAnimationFrame(n)},Re=e=>e.family===`h3`?`center`:e.containment,ze=()=>{let t=e.options,n=J[t.family],r=fe(t.family,t.resolutionOffset);if(e.setReadout(`grid`,`${n.label} ${r}`),e.setReadout(`cellSize`,me(t.family,r)),e.setReadout(`exactness`,n.exactness),!n.tabular){for(let t of[`occupied`,`binned`,`busiest`,`selected`,`selection`,`cover`,`core`,`compacted`,`roundTrip`,`outlineSegments`,`rings`,`groupBorders`])e.setReadout(t,t===`occupied`?`one cell per observation (no aggregation contributor for this family)`:`H3 and Quadbin only`);e.setReadout(`binned`,`${f(b)} per-observation cells, summed by overlap`)}},Be=()=>{let t=(j?.byteLength??0)+(M?.byteLength??0)+a*4*3+i.vertices.byteLength;e.setReadout(`memory`,t)},Ve=e=>{e.index?.resources.destroy(),e.zones?.resources.destroy(),e.selection?.resources.destroy()},He=async(t,n,r)=>{let i={...e.options},a=i.family,o=fe(a,i.resolutionOffset),s=J[a].tabular,c={};try{t&&(c.index=await Ee(a,o)),(n||t)&&(c.zones=s?await ke(a,o,Re(i),i.compactDepth):null),(r||t)&&(c.selection=s?await Ne(a,o,i):null)}catch(e){throw Ve(c),e}return c},Ue=t=>{if(t.index){let e=j;j=t.index,Le(e)}if(t.zones!==void 0){let e=M;M=t.zones,Le(e)}if(t.selection!==void 0){for(let e of I.values())Le(e);I.clear(),L=null,q=null,t.selection&&(I.set(t.selection.key,t.selection),L=t.selection)}ze(),Be(),e.requestLayers()},We=o.track(new P),Ge=o.track(new P),Ke=(e,t,n)=>We.request({build:()=>He(e,t,n),commit:Ue,discard:Ve}),qe=()=>{if(R=void 0,H)return;let e=z,t=B,n=V;z=B=V=!1,Ke(e,t,n)},Je=(e,t,n)=>{z||=e,B||=t,V||=n,clearTimeout(R),R=setTimeout(qe,_e)},Ye=t=>{t!==L&&(L=t,t.dirty=!0,q=null),e.requestLayers()},Xe=()=>{let t={...e.options};if(!J[t.family].tabular)return;let n=fe(t.family,t.resolutionOffset),r=Me(t,n),i=I.get(r);if(i){Ye(i);return}Ge.request({build:()=>Ne(t.family,n,t),commit:t=>{let n=e.options,r=I.get(t.key);if(r||t.key!==Me(n,fe(n.family,n.resolutionOffset))){t.resources.destroy(),r&&Ye(r);return}I.set(t.key,t),Ye(t)},discard:e=>e.resources.destroy()})};Ce(),await Ke(!0,!0,!0);let Ze=(t,n)=>{let r=re(i,t,n);if(r<0)return null;let a=i.names[r]??`Area ${r+1}`,o=x[r+1]??0,s=J[e.options.family].tabular,c=ue[r];return`${a}: ${f(o)} observations in the current filter${s&&M?`; ${f(c??0)} cover cells`:``}`};return{getCompiledGraphs:()=>{let e=[];return j&&e.push(j.indexGraph,...j.outlineGraphs),M&&e.push(M.graph),L&&e.push(L.graph),e},setOption(t){switch(t){case`family`:case`resolutionOffset`:Je(!0,!0,!0);break;case`containment`:case`compactDepth`:Je(!1,!0,!1);break;case`selectionOperation`:Xe();break;case`ringRadius`:case`levelsUp`:case`levelsDown`:e.options.selectionOperation==={ringRadius:`ring`,levelsUp:`parent`,levelsDown:`children`}[t]&&(clearTimeout(R),R=setTimeout(()=>{R=void 0,H||Xe()},_e));break;case`category`:case`months`:case`sampledPercent`:Ce();break;case`ceiling`:Oe(),e.requestLayers();break;default:e.requestLayers()}},encode(e){let t=j;if(t){if(t.dirty){t.indexGraph.encode(e,{parameters:void 0});for(let n of t.outlineGraphs)n.encode(e,{parameters:void 0});for(let n of[t.outline,t.groupOutline])n&&e.copyBufferToBuffer({sourceBuffer:n.count,sourceOffset:0,destinationBuffer:n.drawCommands.buffer,destinationOffset:4,size:4});t.dirty=!1,t.reader?.markStale()}M?.dirty&&(M.graph.encode(e,{parameters:void 0}),M.dirty=!1,M.reader.markStale()),L?.dirty&&(L.graph.encode(e,{parameters:void 0}),L.dirty=!1,L.reader.markStale()),t.reader?.flush(e),M?.reader.flush(e),L?.reader.flush(e)}},getLayers(){let n=e.options,r=j;if(!r)return[];let a=J[r.family].tabular,o=[],s=Math.max(1,Math.round(S*n.ceiling/100));n.showCells&&(o.push(new K({id:`cells-fill-${r.serial}`,boundaries:r.cells.boundaries,vertexCounts:r.cells.vertexCounts,stride:r.cells.stride,instanceCount:r.cells.rows,rowCounts:r.tableCount,values:r.tableCounts,mode:`fill`,colorMode:a?`ramp`:`uniform`,ramp:n.ramp,sqrtScale:n.scale===`sqrt`,color:a?[255,255,255,215]:[255,190,90,Math.min(40,Math.round(3*(100/n.ceiling)))],valueMaximum:s,additive:!a})),a&&o.push(new K({id:`cells-outline-${r.serial}`,boundaries:r.cells.boundaries,vertexCounts:r.cells.vertexCounts,stride:r.cells.stride,instanceCount:r.cells.rows,rowCounts:r.tableCount,mode:`outline`,color:[10,10,25,110],widthPixels:.8}))),a&&r.outline&&n.outline!==`off`&&(n.outline===`groups`&&r.groupOutline&&o.push(new A({id:`cells-group-outline-${r.serial}`,coordinateSystem:t.LNGLAT,segments:r.groupOutline.endpoints,drawCommands:r.groupOutline.drawCommands,values:r.groupOutline.groups,valueFormat:`uint32`,colormap:`category`,palette:ve,widthPixels:3.5})),n.outline===`rings`&&r.outline.rings?o.push(new de({id:`cells-set-rings-${r.serial}`,ringOffsets:r.outline.rings.offsets,ringCounts:r.outline.rings.counts,ringPositions:r.outline.rings.positions,slotCount:Q,ringCapacity:$,palette:be,widthPixels:2.8})):n.outline!==`rings`&&o.push(new A({id:`cells-set-outline-${r.serial}`,coordinateSystem:t.LNGLAT,segments:r.outline.endpoints,drawCommands:r.outline.drawCommands,widthPixels:2.6,color:[255,255,255,245]})));let c=M;a&&n.showCover&&c&&(o.push(new K({id:`cells-cover-fill-${c.serial}`,boundaries:c.cover.boundaries,vertexCounts:c.cover.vertexCounts,stride:c.cover.stride,instanceCount:Z,rowCounts:c.cover.count,values:n.coverColor===`core`?c.cover.core:c.cover.featureIds,mode:`fill`,colorMode:`category`,palette:n.coverColor===`core`?ye:xe,opacity:n.showCompacted?.4:.85}),new K({id:`cells-cover-outline-${c.serial}`,boundaries:c.cover.boundaries,vertexCounts:c.cover.vertexCounts,stride:c.cover.stride,instanceCount:Z,rowCounts:c.cover.count,mode:`outline`,color:[255,255,255,n.showCompacted?60:150],widthPixels:.9})),n.showCompacted&&o.push(new K({id:`cells-compact-outline-${c.serial}`,boundaries:c.compacted.boundaries,vertexCounts:c.compacted.vertexCounts,stride:c.compacted.stride,instanceCount:Z,rowCounts:c.compacted.count,mode:`outline`,color:[255,244,120,255],widthPixels:2.6}))),n.showCover&&o.push(new A({id:`cells-area-outline`,coordinateSystem:t.LNGLAT,segments:_,instanceCount:i.outlineSegments.length/4,widthPixels:1.6,color:[255,255,255,235]}));let l=L;if(a&&n.showSelection&&l){let e=l.operation;o.push(new K({id:`cells-selection-fill-${l.serial}`,boundaries:l.result.boundaries,vertexCounts:l.result.vertexCounts,stride:l.result.stride,instanceCount:l.result.rows,values:l.result.distances,mode:`fill`,colorMode:e===`disk`?`distance`:`uniform`,color:e===`parent`?[255,220,80,90]:e===`children`?[110,230,130,150]:e===`ring`?[255,140,60,170]:Se,filterMaximum:n.diskRadius}),new K({id:`cells-selection-outline-${l.serial}`,boundaries:l.result.boundaries,vertexCounts:l.result.vertexCounts,stride:l.result.stride,instanceCount:l.result.rows,values:l.result.distances,mode:`outline`,colorMode:e===`disk`?`distance`:`uniform`,color:[255,255,255,255],filterMaximum:n.diskRadius,widthPixels:e===`parent`?3:1.4}),new K({id:`cells-selected-outline-${l.serial}`,boundaries:l.selected.boundaries,vertexCounts:l.selected.vertexCounts,stride:l.selected.stride,instanceCount:1,mode:`outline`,color:[255,40,120,255],widthPixels:3.2}))}return o},onClick(e){return!e.coordinate||!L?!1:(y.write(Float32Array.of(e.coordinate[0],e.coordinate[1])),L.dirty=!0,!0)},getTooltip(e){return e.coordinate?Ze(e.coordinate[0],e.coordinate[1]):null},destroy(){H=!0,clearTimeout(R),j?.reader?.stop(),M?.reader.stop();for(let e of I.values())e.reader.stop(),e.resources.destroy();I.clear(),j?.resources.destroy(),M?.resources.destroy();for(let e of U)e.resources.destroy();U.clear(),o.destroy()}}}export{Ce as createH3Aggregation};