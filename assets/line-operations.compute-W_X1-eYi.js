import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{D as t,E as n,S as r,p as i,w as a}from"./analysis-frame-Blu0tYdw.js";import{N as o,_ as s,b as c,c as l,d as u,h as d,l as f,n as p,t as m,v as h,w as g,y as _}from"./resources-CwvhEOBe.js";import{t as v}from"./gpu-scan-BuGhTLDQ.js";import{l as y,s as b}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as x}from"./geodesic-wgsl-DS0CJZ42.js";import{t as S}from"./b3-line-views-far-CsyxMovp.js";import{c as C,p as w}from"./spatial-join-passes-C0ZwcuNs.js";import{a as T}from"./nearest-nodes-DBRmVX1E.js";import{n as E,t as D}from"./gpu-nearest-feature-join-DscXfyZB.js";import{a as O,c as ee,i as k,l as A,o as j,s as M,t as N,u as P}from"./line-segmentize-kernels-JWzYlmh-.js";import{r as F}from"./gpu-great-circle-arcs-D0WZYJJk.js";import{n as te,t as I}from"./line-geometry-output-kdXG4gz1.js";import{a as L}from"./index-Bjk18V2I.js";import{t as R}from"./graph-swap-41Ru7LVu.js";import{t as z}from"./summary-reader-CsgWxENf.js";import{i as B,r as ne}from"./b3-layers-B2V1ljKW.js";import{a as V,i as H,l as U,o as re,p as ie,s as ae,t as W}from"./b3-common-BbHtVprn.js";import{i as oe,r as se}from"./b3-city-data-BBHr728V.js";var G=`GPULineSegmentize`,ce=Object.freeze({...i,measure:`distance`}),le=class{id;props;spatialContext;coordinateSystem;radius;maximumPieceCount;constructor(e){this.id=e.id??`line-segmentize`,this.props=e,this.spatialContext=n(this.id,e.spatialContext,ce),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.radius=r(this.spatialContext)??1,this.maximumPieceCount=e.maximumPieceCount??1024;let{id:t}=this;if(e.closedLines!==void 0&&typeof e.closedLines!=`boolean`)throw Error(`${t} closedLines must be a boolean`);_(e.positions,[`float32x2`],`${t} positions`);let i=e.positions.length;if(i<1)throw Error(`${t} needs at least one position`);if(u(t,`maximumPieceCount`,this.maximumPieceCount,1,Math.floor(4294967295/i)),_(e.lineOffsets,[`uint32`],`${t} lineOffsets`),e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs two rows`);if(_(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);te(t,e.output,e.lineOffsets.length-1),d(t,I(e.output),[e.positions,e.lineOffsets,e.parameters])}getCommandNodes(e){let{props:t,id:n,coordinateSystem:r,radius:i}=this,{output:a}=t;s(n,e,[t.positions,t.lineOffsets,t.parameters,...I(a)]);let o=t.positions.length,c=t.lineOffsets.length-1,l=g(e,`${n}-counts`,`uint32`,o),u=g(e,`${n}-starts`,`uint32`,o),d=a.measures?g(e,`${n}-row-measures`,`float32`,o):void 0,f=r===`spherical`?g(e,`${n}-row-shifts`,`float32`,o):void 0,p=!!(t.closedLines&&f&&d&&a.sourceRows),m=[];return(d||f)&&m.push(j(e,{id:`${n}-prefix`,operation:G,positions:t.positions,lineOffsets:t.lineOffsets,coordinateSystem:r,radius:i,closedLines:t.closedLines,rowMeasures:d,rowShifts:f})),m.push(M(e,{id:`${n}-count`,operation:G,positions:t.positions,lineOffsets:t.lineOffsets,parameters:t.parameters,coordinateSystem:r,radius:i,maximumPieceCount:this.maximumPieceCount,counts:l,closedLines:t.closedLines}),...new v({id:`${n}-scan`,input:l,output:u,mode:`exclusive`}).getCommandNodes(e),ee(e,{id:`${n}-emit`,closedLines:t.closedLines,lineOffsets:t.lineOffsets,operation:G,positions:t.positions,coordinateSystem:r,radius:i,counts:l,starts:u,rowMeasures:d,rowShifts:f,outputPositions:a.positions,outputSourceRows:p?void 0:a.sourceRows,outputMeasures:a.measures}),O(e,{id:`${n}-publish`,operation:G,lineOffsets:t.lineOffsets,lineCount:c,rowCount:o,counts:l,starts:u,outputLineOffsets:a.lineOffsets,capacity:a.positions.length,count:a.status.count,overflow:a.status.overflow,requiredCount:a.status.requiredCount,lineCountOutput:a.status.lineCount})),p&&a.sourceRows&&m.push(A(e,{id:`${n}-source-rows`,operation:G,counts:l,starts:u,outputSourceRows:a.sourceRows})),a.sourceLineIndices&&m.push(k(e,{id:`${n}-source-line-indexs`,operation:G,sourceLineIndices:a.sourceLineIndices})),m}},K=1;function ue(e,t){return b(e,{id:t.id,operation:t.operation,variant:`piece-count`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`pieceCounts`,view:t.pieceCounts,type:`u32`,access:`read_write`}],invocationCount:t.lineOffsets.length-1,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const MAXIMUM_PIECES: u32 = ${t.maximumPieces}u;
const IS_CHUNK: bool = ${t.mode===`chunk`};`,body:`let start = min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + index + 1u], start), ROW_COUNT);
  var count = 0u;
  if (end > start) {
    count = 1u;
    let lineLength = rowMeasures[rowMeasuresOffset + end - 1u];
    let chunkLength = parameters[parametersOffset];
    if (IS_CHUNK && lineLength > 0.0 && chunkLength > 0.0) {
      count = u32(min(ceil(lineLength / chunkLength), f32(MAXIMUM_PIECES)));
      count = clamp(count, 1u, MAXIMUM_PIECES);
    }
  }
  pieceCounts[pieceCountsOffset + index] = count;`})}function de(e,t){let n=t.lineOffsets.length-1;return b(e,{id:t.id,operation:t.operation,variant:`piece-range`,bindings:[{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`pieceCounts`,view:t.pieceCounts,type:`u32`,access:`read`},{name:`pieceStarts`,view:t.pieceStarts,type:`u32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read_write`},{name:`vertexCounts`,view:t.vertexCounts,type:`u32`,access:`read_write`},{name:`pieceTotal`,view:t.pieceTotal,type:`u32`,access:`read_write`}],invocationCount:t.vertexCounts.length,declarations:`const LINE_COUNT: u32 = ${n}u;
const ROW_COUNT: u32 = ${t.rowCount}u;
const STRIDE: u32 = 10u;
const IS_CHUNK: bool = ${t.mode===`chunk`};
const PIECE_COPY: u32 = ${K}u;

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// First row in [low, high) whose measure is greater than (strict) or at least (!strict) the value.
fn findRow(low: u32, high: u32, value: f32, strict: bool) -> u32 {
  var first = low;
  var last = high;
  while (first < last) {
    let middle = (first + last) / 2u;
    let measure = getMeasure(middle);
    if ((strict && measure <= value) || (!strict && measure < value)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  return first;
}`,body:`let total = pieceStarts[pieceStartsOffset + LINE_COUNT - 1u] + pieceCounts[pieceCountsOffset + LINE_COUNT - 1u];
  if (index == 0u) {
    pieceTotal[pieceTotalOffset] = total;
  }
  let base = piecesOffset + index * STRIDE;
  if (index >= total) {
    pieces[base] = 0xffffffffu;
    pieces[base + 6u] = 0u;
    vertexCounts[vertexCountsOffset + index] = 0u;
    return;
  }
  // Largest line whose first piece is at most index (lines without pieces are skipped).
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (pieceStarts[pieceStartsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let line = low;
  let local = index - pieceStarts[pieceStartsOffset + line];
  let pieceCount = pieceCounts[pieceCountsOffset + line];
  let start = min(lineOffsets[lineOffsetsOffset + line], ROW_COUNT);
  let end = min(max(lineOffsets[lineOffsetsOffset + line + 1u], start), ROW_COUNT);
  let lineLength = getMeasure(end - 1u);
  var startMeasure = 0.0;
  var endMeasure = lineLength;
  var flags = 0u;
  var firstInterior = start;
  var interiorEnd = end;
  var vertexCount = end - start;
  if (!(lineLength > 0.0)) {
    flags = PIECE_COPY;
  } else {
    if (IS_CHUNK) {
      let chunkLength = parameters[parametersOffset];
      if (chunkLength > 0.0) {
        startMeasure = f32(local) * chunkLength;
        if (local + 1u < pieceCount) {
          endMeasure = f32(local + 1u) * chunkLength;
        }
      }
    } else {
      startMeasure = clamp(parameters[parametersOffset + 1u], 0.0, lineLength);
      endMeasure = clamp(parameters[parametersOffset + 2u], 0.0, lineLength);
    }
    if (startMeasure > endMeasure) {
      vertexCount = 0u;
      firstInterior = start;
      interiorEnd = start;
    } else {
      firstInterior = findRow(start, end, startMeasure, true);
      interiorEnd = max(findRow(start, end, endMeasure, false), firstInterior);
      vertexCount = 2u + interiorEnd - firstInterior;
    }
  }
  pieces[base] = line;
  pieces[base + 1u] = bitcast<u32>(startMeasure);
  pieces[base + 2u] = bitcast<u32>(endMeasure);
  pieces[base + 3u] = flags;
  pieces[base + 4u] = firstInterior;
  pieces[base + 5u] = interiorEnd;
  pieces[base + 6u] = vertexCount;
  pieces[base + 7u] = start;
  pieces[base + 8u] = end;
  vertexCounts[vertexCountsOffset + index] = vertexCount;`})}function fe(e,t){let n=t.coordinateSystem===`spherical`,r=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`rowMeasures`,view:t.rowMeasures,type:`f32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`vertexStarts`,view:t.vertexStarts,type:`u32`,access:`read`},{name:`outputPositions`,view:t.outputPositions,type:`f32`,access:`read_write`}];return n&&t.rowShifts&&r.push({name:`rowShifts`,view:t.rowShifts,type:`f32`,access:`read`}),t.outputMeasures&&r.push({name:`outputMeasures`,view:t.outputMeasures,type:`f32`,access:`read_write`}),t.outputSourceRows&&r.push({name:`outputSourceRows`,view:t.outputSourceRows,type:`u32`,access:`read_write`}),b(e,{id:t.id,operation:t.operation,variant:`piece-emit`,bindings:r,invocationCount:t.outputPositions.length,workgroupSize:128,guardIndex:!1,declarations:`const CAPACITY: u32 = ${t.outputPositions.length}u;
const STRIDE: u32 = 10u;
const PIECE_COPY: u32 = ${K}u;
${n?x:``}
${P(t.vertexStarts.length)}
fn balancedStart(piece: u32) -> u32 { return vertexStarts[vertexStartsOffset + piece]; }
fn balancedCount(piece: u32) -> u32 { return pieces[piecesOffset + piece * STRIDE + 6u]; }

fn getMeasure(row: u32) -> f32 {
  return rowMeasures[rowMeasuresOffset + row];
}

// Position of an input row, with spherical longitudes unwrapped along the line.
fn getVertex(row: u32) -> vec2<f32> {
  var vertex = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  ${n&&t.rowShifts?`vertex.x = vertex.x + rowShifts[rowShiftsOffset + row];`:``}
  return vertex;
}

fn interpolate(segmentRow: u32, measure: f32) -> vec2<f32> {
  let start = getVertex(segmentRow);
  let end = getVertex(segmentRow + 1u);
  let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
  var fraction = 0.0;
  if (segmentMeasure > 0.0) {
    fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
  }
  ${n?`return geodesicInterpolate(start, end, geodesicCentralAngle(start, end), fraction);`:`return mix(start, end, fraction);`}
}

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${t.outputMeasures?`outputMeasures[outputMeasuresOffset + outputRow] = measure;`:``}
  ${t.outputSourceRows?`outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;`:``}
}`,body:`let piece = findBalancedOwner(index, workgroupIndex * BALANCED_LANES, localInvocationIndex);
  if (piece == BALANCED_NONE) {
    return;
  }
  let base = piecesOffset + piece * STRIDE;
  let local = index - vertexStarts[vertexStartsOffset + piece];
  let flags = pieces[base + 3u];
  let firstInterior = pieces[base + 4u];
  let interiorEnd = pieces[base + 5u];
  if ((flags & PIECE_COPY) != 0u) {
    let row = firstInterior + local;
    writeVertex(index, getVertex(row), getMeasure(row), row);
    return;
  }
  let lineStart = pieces[base + 7u];
  let lineEnd = pieces[base + 8u];
  let startMeasure = bitcast<f32>(pieces[base + 1u]);
  let endMeasure = bitcast<f32>(pieces[base + 2u]);
  // A measure lies on the segment ending at the first row past it, clamped to the line's
  // segments (the line has at least two rows because its length is positive).
  let startSegment = clamp(firstInterior, lineStart + 1u, lineEnd - 1u) - 1u;
  let endSegment = clamp(interiorEnd, lineStart + 1u, lineEnd - 1u) - 1u;
  if (local == 0u) {
    writeVertex(index, interpolate(startSegment, startMeasure), startMeasure, startSegment);
  } else if (local <= interiorEnd - firstInterior) {
    let row = firstInterior + local - 1u;
    writeVertex(index, getVertex(row), getMeasure(row), row);
  } else {
    writeVertex(index, interpolate(endSegment, endMeasure), endMeasure, endSegment);
  }`})}function pe(e,t){let n=t.vertexStarts.length,r=[{name:`pieceTotal`,view:t.pieceTotal,type:`u32`,access:`read`},{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`vertexStarts`,view:t.vertexStarts,type:`u32`,access:`read`},{name:`outputLineOffsets`,view:t.outputLineOffsets,type:`u32`,access:`read_write`},{name:`countOut`,view:t.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}];return t.requiredCount&&r.push({name:`totalOut`,view:t.requiredCount,type:`u32`,access:`read_write`}),t.lineCountOutput&&r.push({name:`lineCountOut`,view:t.lineCountOutput,type:`u32`,access:`read_write`}),b(e,{id:t.id,operation:t.operation,variant:`piece-publish`,bindings:r,invocationCount:n+1,declarations:`const LINE_CAPACITY: u32 = ${n}u;
const CAPACITY: u32 = ${t.capacity}u;
const STRIDE: u32 = 10u;`,body:`let lastBase = piecesOffset + (LINE_CAPACITY - 1u) * STRIDE;
  let total = vertexStarts[vertexStartsOffset + LINE_CAPACITY - 1u] + pieces[lastBase + 6u];
  var start = total;
  if (index < LINE_CAPACITY) {
    start = vertexStarts[vertexStartsOffset + index];
  }
  outputLineOffsets[outputLineOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    let pieceTotal = pieceTotal[pieceTotalOffset];
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY || pieceTotal > LINE_CAPACITY);
    ${t.requiredCount?`totalOut[totalOutOffset] = total;`:``}
    ${t.lineCountOutput?`lineCountOut[lineCountOutOffset] = min(pieceTotal, LINE_CAPACITY);`:``}
  }`})}function me(e,t){return b(e,{id:t.id,operation:t.operation,variant:`piece-source-line-indexs`,bindings:[{name:`pieces`,view:t.pieces,type:`u32`,access:`read`},{name:`sourceLineIndices`,view:t.sourceLineIndices,type:`u32`,access:`read_write`}],invocationCount:t.sourceLineIndices.length,declarations:`const STRIDE: u32 = 10u;`,body:`sourceLineIndices[sourceLineIndicesOffset + index] = pieces[piecesOffset + index * STRIDE];`})}var q=`GPULineChunk`,he=Object.freeze({...i,measure:`distance`});function ge(e,n=new Float32Array(4)){if(n.length<4)throw Error(`Line chunk target must hold 4 elements`);let r=e.chunkLength??0,i=e.startMeasure??0,a=Math.min(e.endMeasure??34e37,34e37);if([r,i,a].some(Number.isNaN)||r<0)throw Error(`Line chunk settings must be numbers and chunkLength non-negative`);return n.set([r,Math.max(i,-34e37),a,0]),t(`GPULineChunk`,e.spatialContext,e.chunkLength),n}var _e=class{id;props;spatialContext;coordinateSystem;radius;lineCapacity;constructor(e){this.id=e.id??`line-chunk`,this.props=e,this.spatialContext=n(this.id,e.spatialContext,he),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.radius=r(this.spatialContext)??1;let{id:t}=this;if(e.mode!==`chunk`&&e.mode!==`substring`)throw Error(`${t} mode must be 'chunk' or 'substring'`);_(e.positions,[`float32x2`],`${t} positions`);let i=e.positions.length;if(i<1)throw Error(`${t} needs at least one position`);if(_(e.lineOffsets,[`uint32`],`${t} lineOffsets`),e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs two rows`);let a=e.lineOffsets.length-1;if(_(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);if(this.lineCapacity=e.mode===`chunk`?Math.max(e.output.lineOffsets.length-1,1):a,te(t,e.output,this.lineCapacity),a*(this.lineCapacity+1)>4294967295||2*this.lineCapacity+i>4294967295)throw Error(`${t} line count times line capacity must fit in 32 bits`);d(t,I(e.output),[e.positions,e.lineOffsets,e.parameters])}getCommandNodes(e){let{props:t,id:n,coordinateSystem:r,lineCapacity:i}=this,{output:a}=t;s(n,e,[t.positions,t.lineOffsets,t.parameters,...I(a)]);let o=t.positions.length,c=t.lineOffsets.length-1,l=g(e,`${n}-row-measures`,`float32`,o),u=r===`spherical`?g(e,`${n}-row-shifts`,`float32`,o):void 0,d=g(e,`${n}-piece-counts`,`uint32`,c),f=g(e,`${n}-piece-starts`,`uint32`,c),p=g(e,`${n}-pieces`,`uint32`,i*10),m=g(e,`${n}-vertex-counts`,`uint32`,i),h=g(e,`${n}-vertex-starts`,`uint32`,i),_=g(e,`${n}-piece-total`,`uint32`,1),y=[j(e,{id:`${n}-prefix`,operation:q,positions:t.positions,lineOffsets:t.lineOffsets,coordinateSystem:r,radius:this.radius,rowMeasures:l,rowShifts:u}),ue(e,{id:`${n}-piece-count`,operation:q,mode:t.mode,lineOffsets:t.lineOffsets,rowMeasures:l,parameters:t.parameters,rowCount:o,maximumPieces:i+1,pieceCounts:d}),...new v({id:`${n}-piece-scan`,input:d,output:f}).getCommandNodes(e),de(e,{id:`${n}-piece-range`,operation:q,mode:t.mode,lineOffsets:t.lineOffsets,rowMeasures:l,parameters:t.parameters,pieceCounts:d,pieceStarts:f,rowCount:o,pieces:p,vertexCounts:m,pieceTotal:_}),...new v({id:`${n}-vertex-scan`,input:m,output:h}).getCommandNodes(e),fe(e,{id:`${n}-emit`,operation:q,coordinateSystem:r,positions:t.positions,rowMeasures:l,rowShifts:u,pieces:p,vertexStarts:h,outputPositions:a.positions,outputMeasures:a.measures,outputSourceRows:a.sourceRows}),pe(e,{id:`${n}-publish`,operation:q,pieceTotal:_,pieces:p,vertexStarts:h,capacity:a.positions.length,outputLineOffsets:a.lineOffsets,count:a.status.count,overflow:a.status.overflow,requiredCount:a.status.requiredCount,lineCountOutput:a.status.lineCount})];return a.sourceLineIndices&&y.push(me(e,{id:`${n}-source-line-indexs`,operation:q,pieces:p,sourceLineIndices:a.sourceLineIndices})),y}},J={ok:0,clamped:1,invalid:2},Y=`
const NAN_BITS: u32 = 0x7fc00000u;
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`;function ve(e,t){return b(e,{id:t.id,operation:t.operation,variant:`line-segments`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`segmentStarts`,view:t.segmentStarts,type:`f32`,access:`read_write`},{name:`segmentEnds`,view:t.segmentEnds,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
${N}
${Y}`,body:`let line = findLine(index);
  // A runtime operand keeps the NaN out of const-expression evaluation, where it is an error.
  var start = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
  var end = start;
  if (line != NO_LINE && index + 1u < lineOffsets[lineOffsetsOffset + line + 1u]) {
    start = getPosition(index);
    end = getPosition(index + 1u);
  }
  segmentStarts[segmentStartsOffset + 2u * index] = start.x;
  segmentStarts[segmentStartsOffset + 2u * index + 1u] = start.y;
  segmentEnds[segmentEndsOffset + 2u * index] = end.x;
  segmentEnds[segmentEndsOffset + 2u * index + 1u] = end.y;`})}function ye(e,t){return b(e,{id:t.id,operation:t.operation,variant:`project`,bindings:[{name:`points`,view:t.points,type:`f32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`nearestSegments`,view:t.nearestSegments,type:`u32`,access:`read`},{name:`vertexMeasures`,view:t.vertexMeasures,type:`f32`,access:`read`},{name:`results`,view:t.results,type:`u32`,access:`read_write`}],invocationCount:t.points.length,declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${t.positions.length}u;
const STRIDE: u32 = 9u;
${N}
${Y}`,body:`let base = resultsOffset + index * STRIDE;
  let segment = nearestSegments[nearestSegmentsOffset + index];
  // Unmatched points carry the join's 0xffffffff sentinel; it also fails this bound.
  if (segment >= ROW_COUNT - 1u) {
    results[base] = NO_LINE;
    results[base + 1u] = NO_LINE;
    results[base + 2u] = NAN_BITS;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = bitcast<u32>(-1.0);
    results[base + 6u] = NAN_BITS;
    results[base + 7u] = 0u;
    results[base + 8u] = NAN_BITS;
    return;
  }
  let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
  let start = getPosition(segment);
  let direction = getPosition(segment + 1u) - start;
  let relative = point - start;
  let lengthSquared = dot(direction, direction);
  var fraction = 0.0;
  if (lengthSquared > 0.0) {
    fraction = clamp(dot(relative, direction) / lengthSquared, 0.0, 1.0);
  }
  let footOffset = direction * fraction;
  let distance = length(relative - footOffset);
  let cross = direction.x * relative.y - direction.y * relative.x;
  var side = 0i;
  if (distance > 0.0 && cross > 0.0) {
    side = 1i;
  } else if (distance > 0.0 && cross < 0.0) {
    side = -1i;
  }
  let line = findLine(segment);
  let lineLength = vertexMeasures[vertexMeasuresOffset + lineOffsets[lineOffsetsOffset + line + 1u] - 1u];
  let measure = vertexMeasures[vertexMeasuresOffset + segment] + fraction * sqrt(lengthSquared);
  results[base] = line;
  results[base + 1u] = segment - lineOffsets[lineOffsetsOffset + line];
  results[base + 2u] = bitcast<u32>(fraction);
  results[base + 3u] = bitcast<u32>(start.x + footOffset.x);
  results[base + 4u] = bitcast<u32>(start.y + footOffset.y);
  results[base + 5u] = bitcast<u32>(distance);
  results[base + 6u] = bitcast<u32>(measure);
  results[base + 7u] = bitcast<u32>(side);
  // shapely line_locate_point(normalized=True); zero-length lines report 0.
  results[base + 8u] = bitcast<u32>(select(0.0, measure / lineLength, lineLength > 0.0));`})}function be(e,t){let n=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`table`,view:t.table,type:`f32`,access:`read_write`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`}];return t.candidateCount&&n.push({name:`candidateCount`,view:t.candidateCount,type:`u32`,access:`read_write`}),b(e,{id:t.id,operation:t.operation,variant:`spherical-table`,bindings:n,invocationCount:t.positions.length,declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const TABLE_STRIDE: u32 = 16u;
${x}
${N}
${Y}`,body:`if (index == 0u) {
    overflow[overflowOffset] = 0u;
    ${t.candidateCount?`candidateCount[candidateCountOffset] = 0u;`:``}
  }
  let base = tableOffset + index * TABLE_STRIDE;
  for (var word = 0u; word < TABLE_STRIDE; word++) {
    table[base + word] = 0.0;
  }
  // Bounding cap half-angle of an invalid row: negative, so projection scans skip it.
  table[base + 15u] = -1.0;
  let line = findLine(index);
  if (line == NO_LINE || index + 1u >= lineOffsets[lineOffsetsOffset + line + 1u]) {
    return;
  }
  let start = getPosition(index);
  let end = getPosition(index + 1u);
  let angle = geodesicCentralAngle(start, end);
  table[base + 11u] = 2.0;
  // A degenerate segment is a point: its cap is the start vector with half-angle 0.
  let startVector = geodesicUnitVector(start);
  table[base + 12u] = startVector.x;
  table[base + 13u] = startVector.y;
  table[base + 14u] = startVector.z;
  table[base + 15u] = 0.0;
  if (!(angle > 0.0)) {
    return;
  }
  let bearing = geodesicInitialBearingDegrees(start, end) * GEODESIC_DEGREES_TO_RADIANS;
  let lambda = start.x * GEODESIC_DEGREES_TO_RADIANS;
  let phi = start.y * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi = sin(phi);
  let cosPhi = geodesicCosLatitude(start.y);
  let up = vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sinPhi);
  let east = vec3<f32>(-sin(lambda), cos(lambda), 0.0);
  let north = vec3<f32>(-sinPhi * cos(lambda), -sinPhi * sin(lambda), cosPhi);
  let tangent = cos(bearing) * north + sin(bearing) * east;
  let normal = cross(up, tangent);
  table[base] = up.x;
  table[base + 1u] = up.y;
  table[base + 2u] = up.z;
  table[base + 3u] = tangent.x;
  table[base + 4u] = tangent.y;
  table[base + 5u] = tangent.z;
  table[base + 6u] = normal.x;
  table[base + 7u] = normal.y;
  table[base + 8u] = normal.z;
  table[base + 9u] = angle;
  table[base + 11u] = 1.0;
  // Bounding cap: the arc midpoint and half the arc angle.
  let center = cos(0.5 * angle) * up + sin(0.5 * angle) * tangent;
  table[base + 12u] = center.x;
  table[base + 13u] = center.y;
  table[base + 14u] = center.z;
  table[base + 15u] = 0.5 * angle;`})}function xe(e,t){return b(e,{id:t.id,operation:t.operation,variant:`spherical-arc-bounds`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`table`,view:t.table,type:`f32`,access:`read`},{name:`arcMinima`,view:t.arcMinima,type:`f32`,access:`read_write`},{name:`arcMaxima`,view:t.arcMaxima,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`const TABLE_STRIDE: u32 = 16u;
const BOX_PADDING: f32 = 1.0e-4;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
const PI: f32 = 3.14159265358979;
const DEGREES_PER_RADIAN: f32 = 57.2957795130823;
${Y}`,body:`let base = tableOffset + index * TABLE_STRIDE;
  let kind = u32(table[base + 11u]);
  var minimum = vec2<f32>(FLOAT32_MAXIMUM);
  var maximum = vec2<f32>(-FLOAT32_MAXIMUM);
  if (kind != 0u) {
    let start = getPosition(index);
    var end = start;
    var isFullLongitude = false;
    if (kind == 1u) {
      end = getPosition(index + 1u);
    }
    minimum = min(start, end);
    maximum = max(start, end);
    if (kind == 1u) {
      let upZ = table[base + 2u];
      let tangentZ = table[base + 5u];
      let arcAngle = table[base + 9u];
      let reach = asin(min(sqrt(upZ * upZ + tangentZ * tangentZ), 1.0)) * DEGREES_PER_RADIAN;
      let peak = atan2(tangentZ, upZ);
      let trough = select(peak + PI, peak - PI, peak > 0.0);
      if (peak >= 0.0 && peak <= arcAngle) {
        maximum.y = max(maximum.y, reach);
      }
      if (trough >= 0.0 && trough <= arcAngle) {
        minimum.y = min(minimum.y, -reach);
      }
      isFullLongitude = abs(end.x - start.x) >= 179.0;
    }
    minimum = minimum - vec2<f32>(BOX_PADDING);
    maximum = maximum + vec2<f32>(BOX_PADDING);
    if (isFullLongitude || maximum.y >= 89.9 || minimum.y <= -89.9) {
      minimum.x = -180.0;
      maximum.x = 180.0;
    }
    minimum.y = max(minimum.y, -90.0);
    maximum.y = min(maximum.y, 90.0);
  }
  arcMinima[arcMinimaOffset + index * 2u] = minimum.x;
  arcMinima[arcMinimaOffset + index * 2u + 1u] = minimum.y;
  arcMaxima[arcMaximaOffset + index * 2u] = maximum.x;
  arcMaxima[arcMaximaOffset + index * 2u + 1u] = maximum.y;`})}function Se(e,t){return b(e,{id:t.id,operation:t.operation,variant:`spherical-node-data`,bindings:[{name:`nodeMinima`,view:t.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:t.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:t.leafIds,type:`u32`,access:`read`},{name:`bvhOverflow`,view:t.bvhOverflow,type:`u32`,access:`read`},{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read_write`},{name:`overflow`,view:t.overflow,type:`u32`,access:`read_write`}],invocationCount:t.nodeCount,declarations:`const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const NODE_WORDS: u32 = 5u;`,body:`if (index == 0u) {
    overflow[overflowOffset] = select(0u, 1u, bvhOverflow[bvhOverflowOffset] != 0u);
  }
  let base = nodeDataOffset + index * NODE_WORDS;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(0xffffffffu, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`})}function Ce(e,t){return b(e,{id:t.id,operation:t.operation,variant:`spherical-project`,bindings:[{name:`points`,view:t.points,type:`f32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`table`,view:t.table,type:`f32`,access:`read`},{name:`vertexMeasures`,view:t.vertexMeasures,type:`f32`,access:`read`},{name:`radius`,view:t.radius,type:`f32`,access:`read`},{name:`nodeData`,view:t.nodeData,type:`u32`,access:`read`},{name:`results`,view:t.results,type:`u32`,access:`read_write`}],invocationCount:t.points.length,declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${t.positions.length}u;
const STRIDE: u32 = 9u;
const TABLE_STRIDE: u32 = 16u;
const SPHERE_RADIUS: f32 = ${y(t.sphereRadius)};
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const NODE_WORDS: u32 = 5u;
const STACK_SIZE: u32 = ${2*t.levelCount}u;
// Relative and absolute (radians, about 0.6 m) slack on the pruning bound, so f32 rounding in the
// box bound and in the arc evaluation never drops the best or a tied arc.
const BOUND_SLACK: f32 = 1.000001;
const BOUND_ABSOLUTE_SLACK: f32 = 1.0e-7;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
${x}
${T}
${N}
${Y}

// Lower bound of the central angle from the point to any arc under the node, or FLOAT32_MAXIMUM
// for an empty node.
fn nodeLowerAngle(node: u32, point: vec2<f32>) -> f32 {
  let base = nodeDataOffset + node * NODE_WORDS;
  let nodeMinimum = vec2<f32>(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let nodeMaximum = vec2<f32>(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (nodeMinimum.x > nodeMaximum.x || nodeMinimum.y > nodeMaximum.y) { return FLOAT32_MAXIMUM; }
  return boxLowerAngle(nodeMinimum, nodeMaximum, point, point);
}

fn getTableVector(base: u32, word: u32) -> vec3<f32> {
  return vec3<f32>(table[base + word], table[base + word + 1u], table[base + word + 2u]);
}

struct SegmentHit {
  angle: f32,
  along: f32,
  /** 0 interior of the arc, 1 start endpoint, 2 end endpoint. */
  endpoint: u32,
  normalDot: f32
}

// Angle from the point to segment row of the table (a valid row: kind 1 or 2).
fn evaluateSegment(row: u32, point: vec2<f32>, pointVector: vec3<f32>) -> SegmentHit {
  let tableBase = tableOffset + row * TABLE_STRIDE;
  let kind = u32(table[tableBase + 11u]);
  let start = getPosition(row);
  var angle = 0.0;
  var along = 0.0;
  var endpoint = 1u;
  var normalDot = 0.0;
  if (kind == 1u) {
    let up = getTableVector(tableBase, 0u);
    let tangent = getTableVector(tableBase, 3u);
    let normal = getTableVector(tableBase, 6u);
    let arcAngle = table[tableBase + 9u];
    let startDot = dot(pointVector, up);
    let tangentDot = dot(pointVector, tangent);
    normalDot = dot(pointVector, normal);
    let planeLength = sqrt(startDot * startDot + tangentDot * tangentDot);
    // Angle of the point's projection on the great circle, measured from the segment start.
    var projected = 0.0;
    if (planeLength > 0.0) {
      projected = atan2(tangentDot, startDot);
    }
    if (projected >= 0.0 && projected <= arcAngle) {
      angle = atan2(abs(normalDot), planeLength);
      along = projected;
      endpoint = 0u;
    }
  }
  if (endpoint != 0u) {
    let startAngle = geodesicCentralAngle(point, start);
    angle = startAngle;
    if (kind == 1u) {
      let endAngle = geodesicCentralAngle(point, getPosition(row + 1u));
      if (endAngle < startAngle) {
        angle = endAngle;
        endpoint = 2u;
      }
    }
  }
  return SegmentHit(angle, along, endpoint, normalDot);
}`,body:`let base = resultsOffset + index * STRIDE;
  let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
  let pointVector = geodesicUnitVector(point);
  var bestAngle = FLOAT32_MAXIMUM;
  var bestRow = NO_LINE;
  var bestKind = 0u;
  var bestAlong = 0.0;
  var bestNormalDot = 0.0;
  // The radius only tightens the bound; the result check below still applies it exactly.
  let radiusLimit = radius[radiusOffset];
  var radiusAngle = FLOAT32_MAXIMUM;
  if ((bitcast<u32>(radiusLimit) & 0x7f800000u) != 0x7f800000u && radiusLimit >= 0.0) {
    radiusAngle = radiusLimit / SPHERE_RADIUS;
  }
  let isPointFinite = (bitcast<u32>(point.x) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(point.y) & 0x7f800000u) != 0x7f800000u;
  var stackNode: array<u32, STACK_SIZE>;
  var stackLower: array<f32, STACK_SIZE>;
  var top = 0u;
  let rootLower = nodeLowerAngle(0u, point);
  if (isPointFinite && rootLower < FLOAT32_MAXIMUM) {
    stackNode[0] = 0u;
    stackLower[0] = rootLower;
    top = 1u;
  }
  loop {
    if (top == 0u) { break; }
    top = top - 1u;
    let node = stackNode[top];
    let bound = min(bestAngle, radiusAngle);
    let limit = select(bound, bound * BOUND_SLACK + BOUND_ABSOLUTE_SLACK, bound < 1.0e37);
    if (stackLower[top] > limit) { continue; }
    if (node < INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstLower = nodeLowerAngle(first, point);
      let secondLower = nodeLowerAngle(second, point);
      let firstUsable = firstLower < FLOAT32_MAXIMUM && firstLower <= limit;
      let secondUsable = secondLower < FLOAT32_MAXIMUM && secondLower <= limit;
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
        if (firstLower <= secondLower) {
          stackNode[top] = second; stackLower[top] = secondLower;
          stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
        } else {
          stackNode[top] = first; stackLower[top] = firstLower;
          stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
        }
        top = top + 2u;
      } else if (firstUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = first; stackLower[top] = firstLower;
        top = top + 1u;
      } else if (secondUsable && top + 1u <= STACK_SIZE) {
        stackNode[top] = second; stackLower[top] = secondLower;
        top = top + 1u;
      }
      continue;
    }
    let row = nodeData[nodeDataOffset + node * NODE_WORDS + 4u];
    if (row + 1u >= ROW_COUNT) { continue; }
    let halfAngle = table[tableOffset + row * TABLE_STRIDE + 15u];
    if (halfAngle < 0.0) { continue; }
    let hit = evaluateSegment(row, point, pointVector);
    // Lexicographic (angle, row): the traversal order is spatial, the tie rule is row order.
    if (hit.angle < bestAngle || (hit.angle == bestAngle && row < bestRow)) {
      bestAngle = hit.angle;
      bestRow = row;
      bestKind = hit.endpoint;
      bestAlong = hit.along;
      bestNormalDot = hit.normalDot;
    }
  }
  let radiusMeters = radius[radiusOffset];
  let radiusUsable = (bitcast<u32>(radiusMeters) & 0x7f800000u) != 0x7f800000u && radiusMeters >= 0.0;
  if (bestRow == NO_LINE || !radiusUsable || bestAngle * SPHERE_RADIUS > radiusMeters) {
    results[base] = NO_LINE;
    results[base + 1u] = NO_LINE;
    results[base + 2u] = NAN_BITS;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = bitcast<u32>(-1.0);
    results[base + 6u] = NAN_BITS;
    results[base + 7u] = 0u;
    results[base + 8u] = NAN_BITS;
    return;
  }
  let tableBase = tableOffset + bestRow * TABLE_STRIDE;
  let arcAngle = table[tableBase + 9u];
  let start = getPosition(bestRow);
  let end = getPosition(bestRow + 1u);
  let isArc = table[tableBase + 11u] == 1.0;
  var fraction = 0.0;
  var foot = start;
  var measure = vertexMeasures[vertexMeasuresOffset + bestRow];
  if (bestKind == 0u) {
    fraction = clamp(bestAlong / arcAngle, 0.0, 1.0);
    foot = geodesicInterpolate(start, end, arcAngle, fraction);
    measure = measure + bestAlong * SPHERE_RADIUS;
  } else if (bestKind == 2u) {
    fraction = 1.0;
    foot = end;
    measure = vertexMeasures[vertexMeasuresOffset + bestRow + 1u];
  }
  var side = 0i;
  if (isArc && bestAngle > 0.0 && bestNormalDot > 0.0) {
    side = 1i;
  } else if (isArc && bestAngle > 0.0 && bestNormalDot < 0.0) {
    side = -1i;
  }
  let line = findLine(bestRow);
  let lineLength = vertexMeasures[vertexMeasuresOffset + lineOffsets[lineOffsetsOffset + line + 1u] - 1u];
  results[base] = line;
  results[base + 1u] = bestRow - lineOffsets[lineOffsetsOffset + line];
  results[base + 2u] = bitcast<u32>(fraction);
  results[base + 3u] = bitcast<u32>(foot.x);
  results[base + 4u] = bitcast<u32>(foot.y);
  results[base + 5u] = bitcast<u32>(bestAngle * SPHERE_RADIUS);
  results[base + 6u] = bitcast<u32>(measure);
  results[base + 7u] = bitcast<u32>(side);
  results[base + 8u] = bitcast<u32>(select(0.0, measure / lineLength, lineLength > 0.0));`})}function we(e,t){let n=t.columns.filter(e=>e.view),r=[];for(let i=0;i<n.length;i+=7){let a=n.slice(i,i+7),o=[{name:`results`,view:t.results,type:`u32`,access:`read`},...a.map(e=>({name:e.name,view:e.view,type:e.type,access:`read_write`}))];r.push(b(e,{id:i===0?`${t.id}`:`${t.id}-${i/7}`,operation:t.operation,variant:`scatter`,bindings:o,invocationCount:t.rowCount,declarations:`const STRIDE: u32 = ${t.stride}u;`,body:`let base = resultsOffset + index * STRIDE;
  ${a.map(e=>e.statement).join(`
  `)}`}))}return r}function Te(e){return[{name:`lineIndices`,view:e.lineIndices,type:`u32`,statement:`lineIndices[lineIndicesOffset + index] = results[base];`},{name:`segmentIndices`,view:e.segmentIndices,type:`u32`,statement:`segmentIndices[segmentIndicesOffset + index] = results[base + 1u];`},{name:`fractions`,view:e.fractions,type:`f32`,statement:`fractions[fractionsOffset + index] = bitcast<f32>(results[base + 2u]);`},{name:`footPoints`,view:e.footPoints,type:`f32`,statement:`footPoints[footPointsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  footPoints[footPointsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`},{name:`distances`,view:e.distances,type:`f32`,statement:`distances[distancesOffset + index] = bitcast<f32>(results[base + 5u]);`},{name:`measures`,view:e.measures,type:`f32`,statement:`measures[measuresOffset + index] = bitcast<f32>(results[base + 6u]);`},{name:`normalizedMeasures`,view:e.normalizedMeasures,type:`f32`,statement:`normalizedMeasures[normalizedMeasuresOffset + index] = bitcast<f32>(results[base + 8u]);`},{name:`sides`,view:e.sides,type:`i32`,statement:`sides[sidesOffset + index] = bitcast<i32>(results[base + 7u]);`},{name:`signedOffsets`,view:e.signedOffsets,type:`f32`,statement:`signedOffsets[signedOffsetsOffset + index] = select(
    f32(bitcast<i32>(results[base + 7u])) * bitcast<f32>(results[base + 5u]),
    bitcast<f32>(results[base + 2u]),
    results[base] == 0xffffffffu
  );`}]}function Ee(e,t){let n=t.coordinateSystem===`spherical`,r=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.lineOffsets,type:`u32`,access:`read`},{name:`vertexMeasures`,view:t.vertexMeasures,type:`f32`,access:`read`},{name:`eventLineIndices`,view:t.eventLineIndices,type:`u32`,access:`read`},{name:`eventMeasures`,view:t.eventMeasures,type:`f32`,access:`read`},{name:`results`,view:t.results,type:`u32`,access:`read_write`}];return t.eventOffsets&&r.push({name:`eventOffsets`,view:t.eventOffsets,type:`f32`,access:`read`}),t.parameters&&r.push({name:`parameters`,view:t.parameters,type:`f32`,access:`read`}),b(e,{id:t.id,operation:t.operation,variant:`locate`,bindings:r,invocationCount:t.eventLineIndices.length,declarations:`const LINE_COUNT: u32 = ${t.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${t.positions.length}u;
const STRIDE: u32 = 6u;
const IS_FRACTION: bool = ${t.measureMode===`fraction`};
const STATUS_OK: u32 = ${J.ok}u;
const STATUS_CLAMPED: u32 = ${J.clamped}u;
const STATUS_INVALID: u32 = ${J.invalid}u;
const SPHERE_RADIUS: f32 = ${y(t.sphereRadius??1)};
${n?x:``}
${Y}

fn getMeasure(row: u32) -> f32 {
  return vertexMeasures[vertexMeasuresOffset + row];
}`,body:`let base = resultsOffset + index * STRIDE;
  let line = eventLineIndices[eventLineIndicesOffset + index];
  var lineStart = 0u;
  var lineEnd = 0u;
  if (line < LINE_COUNT) {
    lineStart = min(lineOffsets[lineOffsetsOffset + line], ROW_COUNT);
    lineEnd = min(max(lineOffsets[lineOffsetsOffset + line + 1u], lineStart), ROW_COUNT);
  }
  if (lineEnd <= lineStart) {
    results[base] = NAN_BITS;
    results[base + 1u] = NAN_BITS;
    results[base + 2u] = 0xffffffffu;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = STATUS_INVALID;
    return;
  }
  let total = getMeasure(lineEnd - 1u);
  var measure = eventMeasures[eventMeasuresOffset + index];
  ${t.parameters?`measure = measure * parameters[parametersOffset] + parameters[parametersOffset + 1u];`:``}
  if (IS_FRACTION) {
    measure = measure * total;
  }
  var status = STATUS_OK;
  if (measure < 0.0) {
    measure = 0.0;
    status = STATUS_CLAMPED;
  } else if (measure > total) {
    measure = total;
    status = STATUS_CLAMPED;
  }
  var position = getPosition(lineStart);
  var tangent = vec2<f32>(0.0);
  var segment = 0u;
  if (lineEnd - lineStart >= 2u) {
    // Upper bound: the first row in (lineStart, lineEnd) whose measure exceeds the target.
    var low = lineStart + 1u;
    var high = lineEnd;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getMeasure(middle) <= measure) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let segmentRow = min(low, lineEnd - 1u) - 1u;
    let start = getPosition(segmentRow);
    let end = getPosition(segmentRow + 1u);
    let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
    var fraction = 0.0;
    if (segmentMeasure > 0.0) {
      fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
    }
    ${n?`let angle = geodesicCentralAngle(start, end);
    if (fraction <= 0.0) {
      position = start;
    } else if (fraction >= 1.0) {
      position = vec2<f32>(start.x + geodesicWrapLongitudeDelta(end.x - start.x), end.y);
    } else {
      position = geodesicInterpolate(start, end, angle, fraction);
    }
    if (angle > 0.0) {
      // Compass direction of the arc at the located point. The tangent is rotated along the
      // great circle from the start vector (not derived from the f32-rounded position), then
      // expressed in the east/north basis of the located point.
      let startLambda = start.x * GEODESIC_DEGREES_TO_RADIANS;
      let startPhi = start.y * GEODESIC_DEGREES_TO_RADIANS;
      let startBearing = geodesicInitialBearingDegrees(start, end) * GEODESIC_DEGREES_TO_RADIANS;
      let startUp = geodesicUnitVector(start);
      let startEast = vec3<f32>(-sin(startLambda), cos(startLambda), 0.0);
      let startNorth = cross(startUp, startEast);
      let startTangent = cos(startBearing) * startNorth + sin(startBearing) * startEast;
      let along = fraction * angle;
      let locatedUp = cos(along) * startUp + sin(along) * startTangent;
      let locatedTangent = cos(along) * startTangent - sin(along) * startUp;
      let eastLength = length(locatedUp.xy);
      var locatedEast = vec3<f32>(0.0, 1.0, 0.0);
      if (eastLength > 0.0) {
        locatedEast = vec3<f32>(-locatedUp.y, locatedUp.x, 0.0) / eastLength;
      }
      let locatedNorth = cross(locatedUp, locatedEast);
      tangent = vec2<f32>(dot(locatedTangent, locatedEast), dot(locatedTangent, locatedNorth));
      let bearing = atan2(tangent.x, tangent.y) * GEODESIC_RADIANS_TO_DEGREES;
      ${t.eventOffsets?`if (eventOffsets[eventOffsetsOffset + index] != 0.0) {
        position = geodesicDestination(
          position,
          bearing - 90.0,
          eventOffsets[eventOffsetsOffset + index] / SPHERE_RADIUS
        );
      }`:``}
    }`:`let direction = end - start;
    position = start + direction * fraction;
    let directionLength = length(direction);
    if (directionLength > 0.0) {
      tangent = direction / directionLength;
    }`}
    segment = segmentRow - lineStart;
  }
  ${t.eventOffsets&&!n?`position = position + vec2<f32>(-tangent.y, tangent.x) * eventOffsets[eventOffsetsOffset + index];`:``}
  results[base] = bitcast<u32>(position.x);
  results[base + 1u] = bitcast<u32>(position.y);
  results[base + 2u] = segment;
  results[base + 3u] = bitcast<u32>(tangent.x);
  results[base + 4u] = bitcast<u32>(tangent.y);
  results[base + 5u] = status;`})}var X=`GPULinearReferencing`,De=Object.freeze({...i,measure:`distance`});function Oe(e){return f(e,36)}function ke(e){return E(e)}function Ae(e){return f(e,64)}var je=class{id;props;lineCount;spatialContext;coordinateSystem;sphereRadius;constructor(e){this.id=e.id??`linear-referencing`,this.props=e,this.spatialContext=n(this.id,e.spatialContext,De),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.sphereRadius=this.coordinateSystem===`spherical`?a(this.spatialContext):1;let{id:t}=this,{output:r}=e;if(this.coordinateSystem===`planar`&&!(Number.isInteger(e.candidateCapacity)&&e.candidateCapacity>0))throw Error(`${t} planar needs candidateCapacity`);if(e.leafCapacity!==void 0&&c(t,`leafCapacity`,e.leafCapacity),_(e.queryPositions,[`float32x2`],`${t} queryPositions`),e.queryPositions.length<1)throw Error(`${t} needs at least one query`);if(_(e.positions,[`float32x2`],`${t} positions`),e.positions.length<2)throw Error(`${t} needs at least two positions`);if(_(e.lineOffsets,[`uint32`],`${t} lineOffsets`),e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs two rows`);this.lineCount=e.lineOffsets.length-1;let i=e.queryPositions.length,o={lineIndices:`uint32`,segmentIndices:`uint32`,fractions:`float32`,footPoints:`float32x2`,distances:`float32`,measures:`float32`,normalizedMeasures:`float32`,sides:`sint32`,signedOffsets:`float32`,vertexMeasures:`float32`},s=0,{status:l,...u}=r;for(let[n,r]of Object.entries(u)){if(!r)continue;if(!(n in o))throw Error(`${t} output.${n} is not a linear-referencing column`);_(r,[o[n]],`${t} output.${n}`);let a=n===`vertexMeasures`?e.positions.length:i;if(r.length!==a)throw Error(`${t} output.${n} must hold ${a} rows`);s++}if(s===0)throw Error(`${t} requires at least one output column`);_(l.overflow,[`uint32`],`${t} output.status.overflow`),d(t,[...Object.values(u),l.overflow,l.candidateCount],[e.queryPositions,e.positions,e.lineOffsets,e.radius])}getCommandNodes(e){let{props:t,id:n}=this,{output:r}=t,{status:i,...a}=r;s(n,e,[t.queryPositions,t.positions,t.lineOffsets,t.radius,i.overflow,i.candidateCount,...Object.values(a)]),h(n,`queryPositions rows`,t.queryPositions.length,Oe(e.device)),h(n,`positions rows`,t.positions.length,this.coordinateSystem===`spherical`?Ae(e.device):ke(e.device));let o=t.positions.length,c=t.queryPositions.length,u=this.coordinateSystem===`spherical`,d=r.vertexMeasures??g(e,`${n}-vertex-measures`,`float32`,o),f=g(e,`${n}-results`,`uint32`,c*9),p=j(e,{id:`${n}-vertex-measures`,operation:X,positions:t.positions,lineOffsets:t.lineOffsets,coordinateSystem:this.coordinateSystem,radius:u?this.sphereRadius:1,rowMeasures:d}),m=we(e,{id:`${n}-output`,operation:X,rowCount:c,stride:9,results:f,columns:Te(a)});if(u){let r=g(e,`${n}-segment-table`,`float32`,o*16),a=g(e,`${n}-arc-minima`,`float32x2`,o),s=g(e,`${n}-arc-maxima`,`float32x2`,o),c=t.leafCapacity??l(o),{bvh:u,nodes:h}=w(e,`${n}-arcs`,X,a,s,c,t.spatialSort??C(o)),_=g(e,`${n}-node-data`,`uint32`,u.nodeCount*5);return[be(e,{id:`${n}-segment-table`,operation:X,positions:t.positions,lineOffsets:t.lineOffsets,table:r,overflow:i.overflow,candidateCount:i.candidateCount}),xe(e,{id:`${n}-arc-bounds`,operation:X,positions:t.positions,table:r,arcMinima:a,arcMaxima:s}),...h,Se(e,{id:`${n}-node-data`,operation:X,nodeMinima:u.nodeMinima,nodeMaxima:u.nodeMaxima,leafIds:u.leafIds,bvhOverflow:u.overflow,internalNodeCount:u.internalNodeCount,nodeCount:u.nodeCount,nodeData:_,overflow:i.overflow}),p,Ce(e,{id:`${n}-project`,operation:X,points:t.queryPositions,positions:t.positions,lineOffsets:t.lineOffsets,table:r,vertexMeasures:d,radius:t.radius,sphereRadius:this.sphereRadius,nodeData:_,internalNodeCount:u.internalNodeCount,levelCount:u.levelCount,results:f}),...m]}let _=g(e,`${n}-segment-starts`,`float32x2`,o),v=g(e,`${n}-segment-ends`,`float32x2`,o),y=g(e,`${n}-nearest-segments`,`uint32`,c);return[ve(e,{id:`${n}-segments`,operation:X,positions:t.positions,lineOffsets:t.lineOffsets,segmentStarts:_,segmentEnds:v}),p,...new D({id:`${n}-join`,queries:{kind:`points`,positions:t.queryPositions},features:{kind:`segments`,starts:_,ends:v},radius:t.radius,candidateCapacity:t.candidateCapacity,leafCapacity:t.leafCapacity,spatialSort:t.spatialSort,output:{nearestFeatureIds:y,status:{overflow:i.overflow,candidateCount:i.candidateCount}}}).getCommandNodes(e),ye(e,{id:`${n}-project`,operation:X,points:t.queryPositions,positions:t.positions,lineOffsets:t.lineOffsets,nearestSegments:y,vertexMeasures:d,results:f}),...m]}},Z=`GPULineLocate`,Me=Object.freeze({...i,measure:`distance`});function Ne(e){return f(e,24)}function Pe(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Line locate target must hold 4 elements`);let n=e.measureScale??1,r=e.measureOffset??0;if(!Number.isFinite(n)||!Number.isFinite(r))throw Error(`Line locate measureScale and measureOffset must be finite`);return t.set([n,r,0,0]),t}var Fe=class{id;props;measureMode;spatialContext;coordinateSystem;sphereRadius;constructor(e){this.id=e.id??`line-locate`,this.props=e,this.measureMode=e.measureMode??`distance`,this.spatialContext=n(this.id,e.spatialContext,Me),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`spherical`:`planar`,this.sphereRadius=this.coordinateSystem===`spherical`?a(this.spatialContext):1;let{id:t}=this,{output:r}=e;if(this.measureMode!==`distance`&&this.measureMode!==`fraction`)throw Error(`${t} measureMode must be 'distance' or 'fraction'`);if(_(e.positions,[`float32x2`],`${t} positions`),e.positions.length<1)throw Error(`${t} needs at least one position`);if(_(e.lineOffsets,[`uint32`],`${t} lineOffsets`),e.lineOffsets.length<2)throw Error(`${t} lineOffsets needs two rows`);_(e.eventLineIndices,[`uint32`],`${t} eventLineIndices`);let i=e.eventLineIndices.length;if(i<1)throw Error(`${t} needs at least one event`);for(let[n,r]of[[`eventMeasures`,e.eventMeasures],[`eventOffsets`,e.eventOffsets]])if(r&&(_(r,[`float32`],`${t} ${n}`),r.length!==i))throw Error(`${t} ${n} length must equal eventLineIndices length`);if(e.parameters&&(_(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4))throw Error(`${t} parameters must hold 4 float32 values`);let o={positions:`float32x2`,segmentIndices:`uint32`,tangents:`float32x2`,angles:`float32`,statuses:`uint32`,vertexMeasures:`float32`};for(let[n,a]of Object.entries(r)){if(!a)continue;if(!(n in o))throw Error(`${t} output.${n} is not a line-locate column`);_(a,[o[n]],`${t} output.${n}`);let r=n===`vertexMeasures`?e.positions.length:i;if(a.length!==r)throw Error(`${t} output.${n} must hold ${r} rows`)}d(t,Object.values(r),[e.positions,e.lineOffsets,e.eventLineIndices,e.eventMeasures,e.eventOffsets,e.parameters])}getCommandNodes(e){let{props:t,id:n}=this,{output:r}=t;s(n,e,[t.positions,t.lineOffsets,t.eventLineIndices,t.eventMeasures,t.eventOffsets,t.parameters,...Object.values(r)]);let i=t.eventLineIndices.length;h(n,`eventLineIndices rows`,i,Ne(e.device));let a=r.vertexMeasures??g(e,`${n}-vertex-measures`,`float32`,t.positions.length),o=g(e,`${n}-results`,`uint32`,i*6);return[j(e,{id:`${n}-vertex-measures`,operation:Z,positions:t.positions,lineOffsets:t.lineOffsets,coordinateSystem:this.coordinateSystem,radius:this.coordinateSystem===`spherical`?this.sphereRadius:1,rowMeasures:a}),Ee(e,{id:`${n}-locate`,operation:Z,positions:t.positions,lineOffsets:t.lineOffsets,vertexMeasures:a,eventLineIndices:t.eventLineIndices,eventMeasures:t.eventMeasures,eventOffsets:t.eventOffsets,parameters:t.parameters,measureMode:this.measureMode,coordinateSystem:this.coordinateSystem,sphereRadius:this.sphereRadius,results:o}),...we(e,{id:`${n}-output`,operation:Z,rowCount:i,stride:6,results:o,columns:[{name:`positions`,view:r.positions,type:`f32`,statement:`positions[positionsOffset + 2u * index] = bitcast<f32>(results[base]);
  positions[positionsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 1u]);`},{name:`segmentIndices`,view:r.segmentIndices,type:`u32`,statement:`segmentIndices[segmentIndicesOffset + index] = results[base + 2u];`},{name:`tangents`,view:r.tangents,type:`f32`,statement:`tangents[tangentsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  tangents[tangentsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`},{name:`angles`,view:r.angles,type:`f32`,statement:`{
    let tangent = vec2<f32>(bitcast<f32>(results[base + 3u]), bitcast<f32>(results[base + 4u]));
    // atan2(0, 0) is undefined in WGSL: degenerate tangents report angle 0.
    angles[anglesOffset + index] = select(
      atan2(tangent.y, tangent.x) * 57.2957795130823,
      0.0,
      tangent.x == 0.0 && tangent.y == 0.0
    );
  }`},{name:`statuses`,view:r.statuses,type:`u32`,statement:`statuses[statusesOffset + index] = results[base + 5u];`}]})]}},Q=e.METER_OFFSETS,Ie=e.LNGLAT,Le=10,Re=100,$=40;function ze(e){let t=0;for(let n=0;n<e.pathCount;n++)for(let r=e.offsets[n];r+1<e.offsets[n+1];r++)t+=Math.hypot(e.local[r*2+2]-e.local[r*2],e.local[r*2+3]-e.local[r*2+1]);return t}async function Be(e){let{ctx:t,rail:n}=e,{device:r}=t,i=new m(r,`reshape`),a=n.vertexCount,s=n.pathCount,c=ze(n),l=i.createBuffer(`rail-local`,n.local),u=i.createBuffer(`rail-lnglat`,n.lngLat),d=i.createBuffer(`rail-offsets`,n.offsets),f=ae(i,`rail-static`,n.local,n.offsets),h=0;for(let e=0;e<s;e++){let t=0;for(let r=n.offsets[e];r+1<n.offsets[e+1];r++)t+=Math.hypot(n.local[r*2+2]-n.local[r*2],n.local[r*2+3]-n.local[r*2+1]);h=Math.max(h,t)}t.setLegendExtent(`measure`,[0,h]);let g=new Map,_=null,v=async f=>{let m=f.lineSystem===`spherical`,g=a;for(let e=0;e<s;e++)for(let t=n.offsets[e];t+1<n.offsets[e+1];t++){let e=Math.hypot(n.local[t*2+2]-n.local[t*2],n.local[t*2+3]-n.local[t*2+1]);g+=Math.min(1024,Math.max(1,Math.ceil(e/Le)))-1}g+=16;let _=re(i,`densify-${m}`,g,s),v=i.createParameterBuffer(`densify-parameters`,`float32`,4),b=new o(r,{id:`densify-${m}`}),x=V(b);b.add(new le({id:`densify`,positions:x(`positions`,m?u:l,`float32x2`,a),lineOffsets:x(`offsets`,d,`uint32`,s+1),spatialContext:m?W:e.projection.spatialContext,parameters:v.importToGraph(b),output:ie(b,_,{measures:!0})}));let S=i.track(await b.compileAsync()),C=new z(i,`densify-${m}`,[{buffer:_.count,size:4},{buffer:_.overflow,size:4}],e=>{let n=new Uint32Array(e);t.setReadout(`reshapeVertices`,`${p(n[0])} (${(n[0]/a).toFixed(2)}× the ${p(a)} input vertices)`),t.setReadout(`reshapeSpacing`,U(c/Math.max(1,n[0]-s))),t.setReadout(`reshapeOverflow`,n[1]?`yes`:`no`)}),w={key:`densify-${m}`,compiled:S,output:_,reader:C,dirty:!0,write:e=>{v.write(F({maximumSegmentLength:10**e.densifyLength})),w.dirty=!0},encode:e=>{w.dirty?(S.encode(e,{parameters:void 0}),H(e,_.count,_.drawCommands),C.request(e),w.dirty=!1):C.flush(e)},layers:(t,n)=>{let r=t.lineSystem===`spherical`?Ie:Q,i=[];return t.showOriginal&&i.push(y(n)),i.push(new B({id:`densify-path`,coordinateSystem:r,coordinateOrigin:e.coordinateOrigin,positions:_.positions,pathOffsets:_.offsets,pathOffsetCount:_.offsetCount,vertexCount:_.count,drawCommands:_.drawCommands,values:_.measures,colorSource:`vertex-value`,colormap:`ylgnbu`,valueRange:[0,h],color:[255,255,255,255],widthPixels:2.2}),new L({id:`densify-vertices`,coordinateSystem:r,coordinateOrigin:e.coordinateOrigin,positions:_.positions,drawCommands:_.drawCommands,values:_.measures,valueFormat:`float32`,colormap:`ylgnbu`,valueRange:[0,h],radiusPixels:2.2,color:[255,255,255,235]})),i}};return w},y=t=>new B({id:`rail-original`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,positions:f.positions,pathOffsets:f.offsets,pathOffsetCount:f.offsetCount,vertexCount:f.vertexCount,drawCommands:f.drawCommands,color:t?[150,160,180,90]:[90,100,120,90],widthPixels:4}),b=async(n,f)=>{let m=n.lineSystem===`spherical`,h=f===`chunk`,g=Math.ceil(c/Re)+s+16,_=a+2*g+16,v=re(i,`${f}-${m}`,_,g),b=i.createParameterBuffer(`${f}-parameters`,`float32`,4),x=new o(r,{id:`${f}-${m}`}),S=V(x);x.add(new _e({id:f,positions:S(`positions`,m?u:l,`float32x2`,a),lineOffsets:S(`offsets`,d,`uint32`,s+1),mode:f,spatialContext:m?W:e.projection.spatialContext,parameters:b.importToGraph(x),output:ie(x,v,{lineCount:!0,sourceLineIndices:h})}));let C=i.track(await x.compileAsync()),w=new z(i,`${f}-${m}`,[{buffer:v.count,size:4},{buffer:v.overflow,size:4},{buffer:v.pathCount,size:4}],e=>{let n=new Uint32Array(e);t.setReadout(`reshapeVertices`,p(n[0])),t.setReadout(`reshapeOverflow`,n[1]?`yes (capacity)`:`no`),t.setReadout(`reshapePieces`,`${p(n[2])} of ${p(g)} slots`)}),T={key:`${f}-${m}`,compiled:C,output:v,reader:w,dirty:!0,write:e=>{b.write(ge(h?{chunkLength:10**e.chunkLength}:{startMeasure:e.substringStart*1e3,endMeasure:Math.max(e.substringStart,e.substringEnd)*1e3})),T.dirty=!0},encode:e=>{T.dirty?(C.encode(e,{parameters:void 0}),H(e,v.count,v.drawCommands),w.request(e),T.dirty=!1):w.flush(e)},layers:(t,n)=>{let r=t.lineSystem===`spherical`?Ie:Q,i=[];return(!h||t.showOriginal)&&i.push(y(n)),i.push(new B({id:`${f}-output`,coordinateSystem:r,coordinateOrigin:e.coordinateOrigin,positions:v.positions,pathOffsets:v.offsets,pathOffsetCount:v.offsetCount,vertexCount:v.count,drawCommands:v.drawCommands,colorSource:h?`path-index`:`uniform`,color:h?[255,255,255,235]:[240,150,60,255],widthPixels:h?3:4})),i}};return T},x=async n=>{let c=n.locateMode===`fraction`,u=s*$,f=new Uint32Array(u),m=new Float32Array(u);for(let e=0;e<s;e++)for(let t=0;t<$;t++){let n=e*$+t;f[n]=e,m[n]=c?t/$:t*1e3}let h=i.createBuffer(`event-paths`,f),g=i.createBuffer(`event-measures`,m),_=i.createBuffer(`event-offsets`,u*4),v=i.createParameterBuffer(`locate-parameters`,`float32`,4),b=i.createBuffer(`event-positions-${c}`,u*8),x=i.createBuffer(`event-tangents-${c}`,u*8),S=i.createBuffer(`event-statuses-${c}`,u*4),C=new o(r,{id:`locate-${c}`}),w=V(C);C.add(new Fe({spatialContext:e.projection.spatialContext,id:`locate`,positions:w(`positions`,l,`float32x2`,a),lineOffsets:w(`offsets`,d,`uint32`,s+1),eventLineIndices:w(`event-paths`,h,`uint32`,u),eventMeasures:w(`event-measures`,g,`float32`,u),eventOffsets:w(`event-offsets`,_,`float32`,u),measureMode:c?`fraction`:`distance`,parameters:v.importToGraph(C),output:{positions:w(`event-positions`,b,`float32x2`,u),tangents:w(`event-tangents`,x,`float32x2`,u),statuses:w(`event-statuses`,S,`uint32`,u)}}));let T=i.track(await C.compileAsync()),E=new z(i,`locate-${c}`,[{buffer:S,size:u*4}],e=>{let n=0,r=0;for(let t of new Uint32Array(e))t===1?n++:t===2&&r++;t.setReadout(`reshapePieces`,`${p(u)} events, ${p(n)} clamped to a route end, ${p(r)} invalid`)}),D=0,O={key:`locate-${c}`,compiled:T,reader:E,dirty:!0,write:e=>{let t=new Float32Array(u);for(let n=0;n<u;n++)t[n]=(n%2==0?1:-1)*e.locateLateral;_.write(t),O.dirty=!0},encode:(e,n)=>{let r=t.options,i=r.locateSpacing,a=r.locateAnimate?n/12%1:0,o=c?a/$*i:a*1e3*i;(r.locateAnimate||O.dirty)&&(v.write(Pe({measureScale:i,measureOffset:o})),T.encode(e,{parameters:void 0}),(n-D>.5||O.dirty)&&(E.request(e),D=n),O.dirty=!1),E.flush(e)},layers:(t,n)=>{let r=[];return r.push(y(n)),r.push(new ne({id:`locate-ticks`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,starts:b,ends:x,directionScale:140,instanceCount:u,color:n?[255,255,255,230]:[30,40,60,230],widthPixels:1.6}),new L({id:`locate-events`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,positions:b,instanceCount:u,radiusPixels:4,color:[240,150,60,255]})),r}};return O},S=async e=>{let t=e.lineSystem===`spherical`,n=e.lineTool===`locate`?`locate-${e.locateMode===`fraction`}`:`${e.lineTool}-${t}`,r=g.get(n);return r||(r=e.lineTool===`densify`?await v(e):e.lineTool===`locate`?await x(e):await b(e,e.lineTool),g.set(n,r)),r},C=(e,n)=>{e.write(n),_=e,t.setReadout(`reshapeInputs`,`${s} routes, ${p(a)} vertices, ${U(c)}`),t.setReadout(`reshapeVertices`,null),t.setReadout(`reshapeSpacing`,null),t.setReadout(`reshapePieces`,null),t.setReadout(`reshapeOverflow`,null),e.dirty=!0},w=i.track(new R);return C(await S(t.options),t.options),{getCompiledGraphs:()=>[...g.values()].map(e=>e.compiled),setOption(e,n){e===`lineTool`||e===`lineSystem`||e===`locateMode`||e===`view`?w.request({build:()=>S(t.options),commit:e=>{C(e,t.options),t.requestLayers()}}):_?.write(n)},encode(e,t){_?.encode(e,t.timeSeconds)},getLayers(){return _?_.layers(t.options,t.theme()===`dark`):[]},destroy(){for(let e of g.values())e.reader?.stop();i.destroy()}}}async function Ve(e){let{ctx:t,streets:n,places:r}=e,{device:i}=t,a=new m(i,`snap`),s=r.count,c=n.pathCount,l=n.vertexCount,u=ae(a,`snap-streets`,n.local,n.offsets),d=a.createBuffer(`place-points`,r.local),f=a.createBuffer(`street-positions`,n.local),h=a.createBuffer(`street-offsets`,n.offsets),g=a.createParameterBuffer(`radius`,`float32`,1),_=a.createBuffer(`foot-points`,s*8),v=a.createBuffer(`distances`,s*4),y=a.createBuffer(`measures`,s*4),b=a.createBuffer(`signed-offsets`,s*4),x=a.createBuffer(`path-indices`,s*4),S=a.createBuffer(`overflow`,4),C=a.createBuffer(`candidate-count`,4),w=1<<22,T=new o(i,{id:`snap`}),E=V(T);T.add(new je({spatialContext:e.projection.spatialContext,id:`snap`,queryPositions:E(`points`,d,`float32x2`,s),positions:E(`positions`,f,`float32x2`,l),lineOffsets:E(`offsets`,h,`uint32`,c+1),radius:g.importToGraph(T),candidateCapacity:w,spatialSort:!0,output:{footPoints:E(`foot-points`,_,`float32x2`,s),distances:E(`distances`,v,`float32`,s),measures:E(`measures`,y,`float32`,s),signedOffsets:E(`signed-offsets`,b,`float32`,s),lineIndices:E(`path-indices`,x,`uint32`,s),status:{overflow:E(`overflow`,S,`uint32`,1),candidateCount:E(`candidate-count`,C,`uint32`,1)}}}));let D=a.track(await T.compileAsync()),O={grocery:2,school_education:4,park_recreation:5,arts_culture:10,worship_community:11},ee=new Set(Object.values(O)),k=Object.entries(O).map(([e,t],n)=>{let i=new Float32Array(s*2);for(let e=0;e<s;e++){let n=r.categories[e]===t;i[e*2]=n?r.local[e*2]:-1e8,i[e*2+1]=n?r.local[e*2+1]:-1e8}return{category:e,code:t,buffer:a.createBuffer(`place-category-${e}`,i),shape:[`square`,`triangle`,`diamond`,`star`,`cross`][n]}}),A=0,j=new Uint8Array(s),M=!0,N=new z(a,`snap`,[{buffer:b,size:s*4},{buffer:v,size:s*4},{buffer:x,size:s*4},{buffer:S,size:4},{buffer:C,size:4}],e=>{let n=new Float32Array(e,0,s),r=new Float32Array(e,s*4,s),i=new Uint32Array(e,s*8,s),a=new Uint32Array(e,s*12,2),o=0,c=0,l=0,u=[],d=0;for(let e=0;e<s;e++)j[e]&&Number.isFinite(n[e])&&i[e]!==4294967295&&(o++,u.push(r[e]),r[e]<=10&&d++,n[e]>0?c++:n[e]<0&&l++);u.sort((e,t)=>e-t);let f=e=>u[Math.min(u.length-1,Math.floor((u.length-1)*e))];t.setReadout(`snapMatched`,`${p(o)} of ${p(A)}`),t.setReadout(`snapSides`,`${p(c)} left / ${p(l)} right`),t.setReadout(`snapDistance`,o?U(f(.5)):`n/a`),t.setReadout(`snapP90Distance`,o?U(f(.9)):`n/a`),t.setReadout(`snapWithinTen`,o?`${(100*d/o).toFixed(1)}%`:`n/a`),t.setReadout(`snapCandidates`,`${p(a[1])} of ${p(w)}${a[0]?` (overflow)`:``}`)}),P=e=>{let n=e.placeCategory===`all`?void 0:O[e.placeCategory],i=new Float32Array(s*2);for(let e=0;e<s;e++){let t=ee.has(r.categories[e])&&(n===void 0||r.categories[e]===n);j[e]=Number(t),t?(i[e*2]=r.local[e*2],i[e*2+1]=r.local[e*2+1]):(i[e*2]=-1e8,i[e*2+1]=-1e8)}A=j.reduce((e,t)=>e+t,0),d.write(i);for(let e of k){let t=new Float32Array(s*2);for(let n=0;n<s;n++){let i=j[n]&&r.categories[n]===e.code;t[n*2]=i?r.local[n*2]:-1e8,t[n*2+1]=i?r.local[n*2+1]:-1e8}e.buffer.write(t)}t.setReadout(`snapInputs`,`${p(A)} eligible places, ${p(c)} streets`)},F=e=>{g.write(Float32Array.of(e.snapRadius)),M=!0};return P(t.options),F(t.options),t.setReadout(`snapInputs`,`${p(A)} eligible places, ${p(c)} streets`),{getCompiledGraphs:()=>[D],setOption(e,t){e===`snapRadius`&&F(t),e===`placeCategory`&&(P(t),M=!0)},encode(e){M?(D.encode(e,{parameters:void 0}),N.request(e),M=!1):N.flush(e)},getLayers(){let n=t.options,r=t.theme()===`dark`,i=Math.min(n.snapRadius,40),a=n.snapColor===`side`?[-i,i]:n.snapColor===`measure`?[0,600]:[0,i],o=n.snapColor===`side`?b:n.snapColor===`measure`?y:v;return[new B({id:`snap-streets`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,positions:u.positions,pathOffsets:u.offsets,pathOffsetCount:u.offsetCount,vertexCount:u.vertexCount,drawCommands:u.drawCommands,color:r?[150,160,180,90]:[90,100,120,90],widthPixels:.8}),new ne({id:`snap-lines`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,starts:d,ends:_,instanceCount:s,values:o,colormap:n.snapColor===`side`?`diverging`:n.snapColor===`measure`?`ylgnbu`:`ylorbr`,valueRange:a,color:[255,255,255,255],widthPixels:1.4,opacity:.9}),...k.map(t=>new L({id:`snap-places-${t.category}`,coordinateSystem:Q,coordinateOrigin:e.coordinateOrigin,positions:t.buffer,instanceCount:s,radiusPixels:2.4,shape:t.shape,outlineColor:r?[255,255,255,220]:[20,30,50,220],outlineWidthPixels:.8,color:r?[120,200,235,210]:[0,105,150,210]}))]},destroy(){N.stop(),a.destroy()}}}async function He(e){let t=e.datasets.get(`chicago-roads`),n=e.datasets.get(`cta-transit`),r=e.datasets.get(`chicago-places`),i=e.datasets.get(`ais-vessels`),a=t.defaultOrigin,o=t.getProjection(a),s=se(n,o),c=null,l={projection:o,ctx:e,coordinateOrigin:[a[0],a[1],0],rail:s,get streets(){return c??=oe(t,o),c},places:{local:r.projectColumn(`position`,a),categories:r.column(`category`),count:r.count}},u={reshape:()=>Be(l),snap:()=>Ve(l),tracks:()=>S({ctx:e,vessels:i})},d=new Map,f=!1,p=new R,m=await u[e.options.view]();d.set(e.options.view,m);let h=e.options.view,g=new Map,_=()=>m,v=t=>{p.request({build:async()=>d.get(t)??await u[t](),commit:n=>{d.set(t,n),m=n,h=t;for(let r of g.get(t)??[])n.setOption(r,e.options);g.delete(t),e.requestLayers()},discard:e=>{d.get(t)!==e&&e.destroy()}})},y=(e,t)=>{_().setOption(e,t);for(let t of Object.keys(u)){if(t===h)continue;let n=g.get(t)??new Set;n.add(e),g.set(t,n)}};return{getCompiledGraphs:()=>_().getCompiledGraphs(),setOption(t,n,r){t===`view`?v(r.view):y(t,r),e.requestLayers()},onThemeChange:()=>e.requestLayers(),encode(e,t){f||_().encode(e,t)},getLayers:()=>_().getLayers(),destroy(){f=!0,p.destroy();for(let e of d.values())e.destroy();d.clear()}}}export{He as createLineOperations};