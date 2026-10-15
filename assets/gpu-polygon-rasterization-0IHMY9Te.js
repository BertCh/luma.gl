import{E as e,u as t}from"./analysis-frame-Blu0tYdw.js";import{_ as n,h as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-scan-BuGhTLDQ.js";import{t as s}from"./gpu-sort-Bl3Dztdb.js";import{s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l}from"./wgsl-common-BNRVzgd8.js";import{a as u}from"./sorted-segment-sums-DwGkmbx2.js";import{i as d,s as f,u as p}from"./raster-band-utils-CA-sh8xt.js";import{t as ee}from"./scan-total-BdolCtK7.js";var m=4294967295;function h(e,t=new Float32Array(4)){d(`getGPUPolygonRasterizationParameterValues`,t,4);let[n,r,i,a,o,s]=e.affine;if(r!==0||a!==0)throw Error(`getGPUPolygonRasterizationParameterValues needs an axis-aligned grid`);let c=e.pixelInterpretation===`point`?-.5:0,l=i+n*c,u=s+o*c;return t[0]=l,t[1]=o<0?u+o*e.height:u,t[2]=n,t[3]=o,t}var g=`
${l}
struct RasterExtent { origin: vec2<f32>, cellSize: vec2<f32>, valid: bool, flipped: bool };
fn getRasterExtent() -> RasterExtent {
  let origin = vec2<f32>(extent[extentOffset], extent[extentOffset + 1u]);
  // A negative cell height means rows advance toward smaller y: the kernels count rows from the
  // smallest y and mirror the row index when they read or write the caller's rasters.
  let signedCellHeight = extent[extentOffset + 3u];
  let cellSize = vec2<f32>(extent[extentOffset + 2u], abs(signedCellHeight));
  let valid = isFiniteValue(origin.x) && isFiniteValue(origin.y) &&
    isFiniteValue(cellSize.x) && isFiniteValue(cellSize.y) && cellSize.x > 0.0 && cellSize.y > 0.0;
  return RasterExtent(origin, cellSize, valid, signedCellHeight < 0.0);
}
// Stored row of a row counted from the smallest y. Needs a HEIGHT constant.
fn getStoredRow(row: u32, rasterExtent: RasterExtent) -> u32 {
  return select(row, HEIGHT - 1u - row, rasterExtent.flipped);
}`;function _(e,t,n){return`
fn ${e}(value: u32) -> u32 {
  // Largest i in [0, rangeCount) with offsets[i] <= value.
  var low = 0u;
  var high = ${n}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (${t}[${t}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) {
    return 0xffffffffu;
  }
  let range = low - 1u;
  if (value >= ${t}[${t}Offset + range + 1u]) {
    return 0xffffffffu;
  }
  return range;
}`}var v=`GPUPolygonRasterization`,y=`0xffffffffu`,b=class{id;spatialContext;props;featureCount;polygonCount;ringCount;singleSort;constructor(n){this.id=n.id??`polygon-rasterization`,this.spatialContext=e(this.id,n.spatialContext,{coordinateSpaces:[`planar`,`longitude-latitude`],metrics:[`none`,`native`],edgeModels:t}),this.props=n;let{id:i}=this,{output:o}=n,s=f(i,n.grid),{width:c,height:l}=n.grid;if(l*(c+1)>4294967295)throw Error(`${i} height * (width + 1) must fit in uint32`);if(p(i,n.parameters,4),a(n.positions,[`float32x2`],`${i} positions`),n.positions.length<1)throw Error(`${i} positions must not be empty`);for(let[e,t]of[[`featureOffsets`,n.featureOffsets],[`polygonOffsets`,n.polygonOffsets],[`ringOffsets`,n.ringOffsets]])if(a(t,[`uint32`],`${i} ${e}`),t.length<2)throw Error(`${i} ${e} requires at least one range and a terminal entry`);if(this.featureCount=n.featureOffsets.length-1,this.polygonCount=n.polygonOffsets.length-1,this.ringCount=n.ringOffsets.length-1,this.featureCount>=4294967295)throw Error(`${i} feature count must be below the no-zone sentinel`);if(!Number.isSafeInteger(n.crossingCapacity)||n.crossingCapacity<2||n.crossingCapacity>=4294967295)throw Error(`${i} crossingCapacity must be an integer of at least 2`);for(let[e,t,n]of[[`zones`,o.zones,s],[`boundary`,o.boundary,s],[`status.overflow`,o.status.overflow,1],[`status.crossingCount`,o.status.crossingCount,1]])if(t&&(a(t,[`uint32`],`${i} output.${e}`),t.length!==n))throw Error(`${i} output.${e} must contain ${n} uint32 rows`);r(i,[o.zones,o.boundary,o.status.overflow,o.status.crossingCount],[n.parameters,n.positions,n.featureOffsets,n.polygonOffsets,n.ringOffsets]),this.singleSort=u(this.polygonCount)+u(s+l-1)<=32}getCommandNodes(e){let{id:t,props:r,polygonCount:a,ringCount:l,featureCount:d,singleSort:f}=this,{parameters:p,positions:h,output:b}=r,{zones:x,boundary:S}=b,{overflow:C,crossingCount:w}=b.status,{width:T,height:E}=r.grid;n(t,e,[p,h,r.featureOffsets,r.polygonOffsets,r.ringOffsets,x,S,C,w]);let D=h.length,O=this.spatialContext.coordinateSpace===`longitude-latitude`,k=O?D*3:D,A=O?`index % ${D}u`:`index`,j=O?`let longitudeShift = f32(i32(index / ${D}u) - 1) * 360.0;`:``,M=e=>O?`loadPosition(${e}) + vec2<f32>(longitudeShift, 0.0)`:`loadPosition(${e})`,N=r.crossingCapacity,P=T*E,F=T+1,I=E*F,L=u(I-1),R=`const WIDTH: u32 = ${T}u;
const HEIGHT: u32 = ${E}u;
const ROW_STRIDE: u32 = ${F}u;
const CAPACITY: u32 = ${N}u;
const POLYGON_COUNT: u32 = ${a}u;
const SINGLE_SORT: bool = ${f};
// Only the single-sort key packs polygons above the row-column bits.
const ROW_COLUMN_BITS: u32 = ${f?L:0}u;
const ROW_COLUMN_LIMIT: u32 = ${I}u;`,z=[],B=i(e,`${t}-primary`,`uint32`,N),V=i(e,`${t}-secondary`,`uint32`,N),te=`${f?a*2**L:I}u`,ne=f?`0u`:`${a}u`,H=[{name:`zonesOut`,view:x,type:`u32`,access:`read_write`},{name:`primary`,view:B,type:`u32`,access:`read_write`},{name:`secondary`,view:V,type:`u32`,access:`read_write`}];S&&H.push({name:`boundaryOut`,view:S,type:`u32`,access:`read_write`}),z.push(c(e,{id:`${t}-reset`,operation:v,variant:`reset`,bindings:H,invocationCount:Math.max(P,N),declarations:`const CELL_COUNT: u32 = ${P}u;
const CAPACITY: u32 = ${N}u;`,body:`if (index < CELL_COUNT) {
    zonesOut[zonesOutOffset + index] = ${m}u;
    ${S?`boundaryOut[boundaryOutOffset + index] = 0u;`:``}
  }
  if (index < CAPACITY) {
    primary[primaryOffset + index] = ${te};
    secondary[secondaryOffset + index] = ${ne};
  }`}));let U=i(e,`${t}-edge-counts`,`uint32`,k),W=i(e,`${t}-edge-next`,`uint32`,k),G=i(e,`${t}-edge-polygons`,`uint32`,k),K=`${g}
fn getRowThreshold(y: f32, rasterExtent: RasterExtent) -> u32 {
  // Smallest row whose center line is at or above y.
  let row = ceil((y - rasterExtent.origin.y) / rasterExtent.cellSize.y - 0.5);
  return u32(clamp(row, 0.0, f32(HEIGHT)));
}
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn isFinitePosition(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y);
}`;z.push(c(e,{id:`${t}-edges`,operation:v,variant:O?`edges-longitude-latitude`:`edges`,bindings:[{name:`positions`,view:h,type:`f32`,access:`read`},{name:`ringOffsets`,view:r.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:r.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`extent`,view:p,type:`f32`,access:`read`},{name:`edgeCounts`,view:U,type:`u32`,access:`read_write`},{name:`edgeNext`,view:W,type:`u32`,access:`read_write`},{name:`edgePolygons`,view:G,type:`u32`,access:`read_write`}],invocationCount:k,declarations:`${R}
${K}
${_(`findRing`,`ringOffsets`,l)}
${_(`findPolygon`,`polygonOffsets`,a)}
${_(`findFeature`,`featureOffsets`,d)}`,body:`var next = ${y};
  var polygon = ${y};
  var count = 0u;
  ${j}
  let ring = findRing(${A});
  if (ring != ${y}) {
    polygon = findPolygon(ring);
  }
  if (polygon != ${y} && findFeature(polygon) != ${y}) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    next = select(${A} + 1u, ringStart, ${A} + 1u >= ringEnd);
    let start = ${M(A)};
    let end = ${M(`next`)};
    let rasterExtent = getRasterExtent();
    if (rasterExtent.valid && isFinitePosition(start) && isFinitePosition(end)) {
      let low = getRowThreshold(min(start.y, end.y), rasterExtent);
      let high = getRowThreshold(max(start.y, end.y), rasterExtent);
      count = high - min(low, high);
      ${O?`// Select longitude strips, retaining off-grid crossings to preserve even-odd parity.
      let gridMaximumX = rasterExtent.origin.x + f32(WIDTH) * rasterExtent.cellSize.x;
      if (gridMaximumX <= longitudeShift - 180.0 || rasterExtent.origin.x >= longitudeShift + 180.0) {
        count = 0u;
        next = ${y};
      }`:``}
    } else {
      next = ${y};
    }
  } else {
    polygon = ${y};
  }
  edgeCounts[edgeCountsOffset + index] = count;
  edgeNext[edgeNextOffset + index] = next;
  edgePolygons[edgePolygonsOffset + index] = polygon;`}));let q=i(e,`${t}-edge-offsets`,`uint32`,k);z.push(...new o({id:`${t}-edge-scan`,input:U,output:q,mode:`exclusive`}).getCommandNodes(e));let J=i(e,`${t}-total`,`uint32`,1),Y=[{name:`total`,view:J,type:`u32`,access:`read`},{name:`overflowOut`,view:C,type:`u32`,access:`read_write`}];w&&Y.push({name:`crossingCountOut`,view:w,type:`u32`,access:`read_write`}),z.push(...ee(e,{id:t,operation:v,counts:U,offsets:q,rowCount:k,total:J}),c(e,{id:`${t}-publish`,operation:v,variant:`publish`,bindings:Y,invocationCount:1,declarations:`const LAST_EDGE: u32 = ${D-1}u;
const CAPACITY: u32 = ${N}u;`,body:`let crossings = total[totalOffset];
  overflowOut[overflowOutOffset] = select(0u, 1u, crossings > CAPACITY);
  ${w?`crossingCountOut[crossingCountOutOffset] = crossings;`:``}`})),z.push(c(e,{id:`${t}-emit`,operation:v,variant:O?`emit-longitude-latitude`:`emit`,bindings:[{name:`positions`,view:h,type:`f32`,access:`read`},{name:`extent`,view:p,type:`f32`,access:`read`},{name:`edgeCounts`,view:U,type:`u32`,access:`read`},{name:`edgeNext`,view:W,type:`u32`,access:`read`},{name:`edgePolygons`,view:G,type:`u32`,access:`read`},{name:`edgeOffsets`,view:q,type:`u32`,access:`read`},{name:`primary`,view:B,type:`u32`,access:`read_write`},{name:`secondary`,view:V,type:`u32`,access:`read_write`}],invocationCount:k,declarations:`${R}
${K}`,body:`let count = edgeCounts[edgeCountsOffset + index];
  if (count == 0u) {
    return;
  }
  let next = edgeNext[edgeNextOffset + index];
  let polygon = edgePolygons[edgePolygonsOffset + index];
  let rasterExtent = getRasterExtent();
  ${j}
  let first = ${M(A)};
  let second = ${M(`next`)};
  // Orient bottom to top so both traversal directions compute bitwise identical crossings.
  let lower = select(second, first, first.y < second.y);
  let upper = select(first, second, first.y < second.y);
  let firstRow = getRowThreshold(lower.y, rasterExtent);
  let base = edgeOffsets[edgeOffsetsOffset + index];
  for (var step = 0u; step < count; step++) {
    let slot = base + step;
    if (slot >= CAPACITY) {
      break;
    }
    let row = firstRow + step;
    let centerY = rasterExtent.origin.y + (f32(row) + 0.5) * rasterExtent.cellSize.y;
    let t = clamp((centerY - lower.y) / (upper.y - lower.y), 0.0, 1.0);
    let crossingX = lower.x + (upper.x - lower.x) * t;
    // First column whose center is at or right of the crossing.
    let column = u32(clamp(ceil((crossingX - rasterExtent.origin.x) / rasterExtent.cellSize.x - 0.5), 0.0, f32(WIDTH)));
    let rowColumn = row * ROW_STRIDE + column;
    if (SINGLE_SORT) {
      primary[primaryOffset + slot] = (polygon << ROW_COLUMN_BITS) | rowColumn;
      secondary[secondaryOffset + slot] = rowColumn;
    } else {
      primary[primaryOffset + slot] = rowColumn;
      secondary[secondaryOffset + slot] = polygon;
    }
  }`}));let X=i(e,`${t}-sorted-keys`,`uint32`,N),Z=i(e,`${t}-sorted-values`,`uint32`,N),Q,$;if(f)z.push(...new s({id:`${t}-sort`,keys:B,values:V,outputKeys:X,outputValues:Z,keyBits:u(a)+L}).getCommandNodes(e)),Q=X,$=Z;else{let n=i(e,`${t}-polygon-keys`,`uint32`,N),r=i(e,`${t}-row-columns`,`uint32`,N);z.push(...new s({id:`${t}-sort-row-column`,keys:B,values:V,outputKeys:X,outputValues:Z,keyBits:u(I)}).getCommandNodes(e),...new s({id:`${t}-sort-polygon`,keys:Z,values:X,outputKeys:n,outputValues:r,keyBits:u(a)}).getCommandNodes(e)),Q=n,$=r}let re=`fn getRecordPolygon(record: u32) -> u32 {
  let key = recordKeys[recordKeysOffset + record];
  return select(key, key >> ROW_COLUMN_BITS, SINGLE_SORT);
}
fn getRecordRowColumn(record: u32) -> u32 {
  return recordRowColumns[recordRowColumnsOffset + record];
}`,ie=i(e,`${t}-segment-flags`,`uint32`,N),ae=i(e,`${t}-ones`,`uint32`,N),oe=i(e,`${t}-ranks`,`uint32`,N);return z.push(c(e,{id:`${t}-segments`,operation:v,variant:`segments`,bindings:[{name:`recordKeys`,view:Q,type:`u32`,access:`read`},{name:`recordRowColumns`,view:$,type:`u32`,access:`read`},{name:`segmentFlags`,view:ie,type:`u32`,access:`read_write`},{name:`ones`,view:ae,type:`u32`,access:`read_write`}],invocationCount:N,declarations:`${R}
${re}`,body:`var isHead = index == 0u;
  if (!isHead) {
    isHead = getRecordPolygon(index) != getRecordPolygon(index - 1u) ||
      getRecordRowColumn(index) / ROW_STRIDE != getRecordRowColumn(index - 1u) / ROW_STRIDE;
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, isHead);
  ones[onesOffset + index] = 1u;`})),z.push(...new o({id:`${t}-rank-scan`,input:ae,output:oe,mode:`exclusive`,segmentFlags:ie}).getCommandNodes(e)),z.push(c(e,{id:`${t}-fill`,operation:v,variant:`fill`,bindings:[{name:`recordKeys`,view:Q,type:`u32`,access:`read`},{name:`recordRowColumns`,view:$,type:`u32`,access:`read`},{name:`ranks`,view:oe,type:`u32`,access:`read`},{name:`featureOffsets`,view:r.featureOffsets,type:`u32`,access:`read`},{name:`total`,view:J,type:`u32`,access:`read`},{name:`extent`,view:p,type:`f32`,access:`read`},{name:`zonesOut`,view:x,type:`atomic<u32>`,access:`read_write`}],invocationCount:N-1,declarations:`${R}
${g}
${re}
${_(`findFeature`,`featureOffsets`,d)}`,body:`if (total[totalOffset] > CAPACITY || (ranks[ranksOffset + index] & 1u) != 0u) {
    return;
  }
  let polygon = getRecordPolygon(index);
  if (polygon >= POLYGON_COUNT || getRecordPolygon(index + 1u) != polygon) {
    return;
  }
  let start = getRecordRowColumn(index);
  let end = getRecordRowColumn(index + 1u);
  let row = start / ROW_STRIDE;
  if (end / ROW_STRIDE != row) {
    return;
  }
  let feature = findFeature(polygon);
  let rowBase = getStoredRow(row, getRasterExtent()) * WIDTH;
  for (var column = start % ROW_STRIDE; column < end % ROW_STRIDE; column++) {
    atomicMin(&zonesOut[zonesOutOffset + rowBase + column], feature);
  }`})),S&&z.push(c(e,{id:`${t}-boundary`,operation:v,variant:O?`boundary-longitude-latitude`:`boundary`,bindings:[{name:`positions`,view:h,type:`f32`,access:`read`},{name:`extent`,view:p,type:`f32`,access:`read`},{name:`edgeNext`,view:W,type:`u32`,access:`read`},{name:`boundaryOut`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:k,declarations:`${R}
${g}
// Cell padding that keeps the marking conservative under float32 rounding: a fixed part plus
// about 8 ulps of the largest magnitude involved, in cell units.
const BASE_PADDING: f32 = 1e-3;
const RELATIVE_PADDING: f32 = 1e-6;
fn loadPosition(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn clampCell(value: f32, limit: u32) -> i32 {
  return i32(clamp(floor(value), -1.0, f32(limit)));
}`,body:`let next = edgeNext[edgeNextOffset + index];
  if (next == ${y}) {
    return;
  }
  let rasterExtent = getRasterExtent();
  ${j}
  let startPosition = ${M(A)};
  let endPosition = ${M(`next`)};
  // Edge in cell units.
  let start = (startPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let end = (endPosition - rasterExtent.origin) / rasterExtent.cellSize;
  let magnitude = (max(abs(startPosition), abs(endPosition)) + abs(rasterExtent.origin)) / rasterExtent.cellSize;
  let padding = BASE_PADDING + RELATIVE_PADDING * max(magnitude.x, magnitude.y);
  let lower = select(end, start, start.y < end.y);
  let upper = select(start, end, start.y < end.y);
  let firstRow = max(clampCell(lower.y - padding, HEIGHT), 0);
  let lastRow = min(clampCell(upper.y + padding, HEIGHT), i32(HEIGHT) - 1);
  for (var row = firstRow; row <= lastRow; row++) {
    // x extent of the segment inside the padded row slab.
    var slabX = vec2<f32>(lower.x, upper.x);
    let deltaY = upper.y - lower.y;
    if (deltaY > 0.0) {
      let t0 = clamp((f32(row) - padding - lower.y) / deltaY, 0.0, 1.0);
      let t1 = clamp((f32(row + 1) + padding - lower.y) / deltaY, 0.0, 1.0);
      slabX = vec2<f32>(lower.x + (upper.x - lower.x) * t0, lower.x + (upper.x - lower.x) * t1);
    }
    let firstColumn = max(clampCell(min(slabX.x, slabX.y) - padding, WIDTH), 0);
    let lastColumn = min(clampCell(max(slabX.x, slabX.y) + padding, WIDTH), i32(WIDTH) - 1);
    for (var column = firstColumn; column <= lastColumn; column++) {
      atomicStore(&boundaryOut[boundaryOutOffset + getStoredRow(u32(row), rasterExtent) * WIDTH + u32(column)], 1u);
    }
  }`})),z}};export{_ as a,h as i,m as n,g as r,b as t};