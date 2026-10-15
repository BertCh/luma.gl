import{d as e,l as t,v as n,w as r}from"./resources-CwvhEOBe.js";import{t as i}from"./gpu-scan-BuGhTLDQ.js";import{r as a,s as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as s}from"./gpu-segment-ring-assembly-CMjctMio.js";import{a as c,d as l}from"./raster-utils-DhdnkkOa.js";import{i as u,s as d}from"./raster-band-utils-CA-sh8xt.js";import{a as f,i as p,n as m,o as h,r as g,s as _,t as v}from"./raster-algebra-utils-C7qGzplC.js";function y(e,t,n=new Float32Array(12)){u(`getGPUIsobandsParameterValues`,n,12);let r=e.firstBand??0,i=e.lastBand??e.breakCount;for(let[t,n]of[[`breakCount`,e.breakCount],[`firstBand`,r],[`lastBand`,i]])if(!Number.isSafeInteger(n)||n<0)throw Error(`Isobands ${t} must be a non-negative integer`);let{width:a,height:o}=t;if(!Number.isSafeInteger(a)||!Number.isSafeInteger(o)||a<1||o<1)throw Error(`getGPUIsobandsParameterValues grid width and height must be positive`);l(`Isobands`,e.extent);let[s,d,f,p]=e.extent,[m,h]=c(`Isobands`,e.localOrigin);return n[0]=e.breakCount,n[1]=r,n[2]=i,n[3]=0,n[4]=s-m,n[5]=d-h,n[6]=f-m,n[7]=p-h,n[8]=(f-s)/a,n[9]=(p-d)/o,n[10]=0,n[11]=0,n}function b(e){return`
const CELL_COLUMNS: u32 = WIDTH - 1u;
const NO_EVENT: u32 = 0xffffffffu;

// Event kinds in bits 0..1: 0 corner, 1 entry crossing, 2 exit crossing. Bit 2 is the level
// (0 low, 1 high), bits 3..5 the corner index (corners) or canonical edge index (crossings).
var<private> cornerValues: array<f32, 4>;
var<private> bandStates: array<u32, 4>;
var<private> bandMasks: array<u32, 2>;
var<private> bandJoined: array<bool, 2>;
var<private> cellEvents: array<u32, 12>;
var<private> cellEventCount: u32;
var<private> cellColumn: u32;
var<private> cellRow: u32;
var<private> bandLow: f32;
var<private> bandHigh: f32;
var<private> pieceVertexCount: u32;
var<private> pieceFirst: vec2<f32>;
var<private> piecePrevious: vec2<f32>;
var<private> emittedTriangles: u32;
var<private> cellEventWalks: array<u32, 12>;
var<private> pieceNextIsLevel: bool;
var<private> pieceLastWalk: u32;
var<private> cellEdgeCount: u32;
var<private> bandHasLow: bool;
var<private> bandHasHigh: bool;
var<private> bandIsSliver: bool;
var<private> edgeBase: u32;
var<private> EDGE_START: array<u32, 4> = array<u32, 4>(0u, 1u, 3u, 0u);
var<private> EDGE_END: array<u32, 4> = array<u32, 4>(1u, 2u, 2u, 3u);

fn getParameterCount(value: f32) -> u32 {
  return min(u32(clamp(select(0.0, value, isFiniteValue(value)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
}

fn getWorldPosition(gridX: f32, gridY: f32) -> vec2<f32> {
  let x = params[paramsOffset + 4u] + (gridX + 0.5) * params[paramsOffset + 8u];
  let y = params[paramsOffset + 5u] + (gridY + 0.5) * params[paramsOffset + 9u];
  return vec2<f32>(x, y);
}

fn getCornerPosition(corner: u32) -> vec2<f32> {
  let dx = select(0u, 1u, corner == 1u || corner == 2u);
  let dy = select(0u, 1u, corner >= 2u);
  return getWorldPosition(f32(cellColumn + dx), f32(cellRow + dy));
}

// Crossing of the band's low (level 0) or high (level 1) break on a canonical edge: horizontal
// edges run +x, vertical edges +y, from the lower-index sample to the higher-index sample.
fn getCrossingPosition(level: u32, edge: u32) -> vec2<f32> {
  let boundary = select(bandLow, bandHigh, level == 1u);
  let startValue = cornerValues[EDGE_START[edge]];
  let endValue = cornerValues[EDGE_END[edge]];
  let t = (boundary - startValue) / (endValue - startValue);
  let baseX = f32(cellColumn);
  let baseY = f32(cellRow);
  if (edge == 0u) {
    return getWorldPosition(baseX + t, baseY);
  }
  if (edge == 1u) {
    return getWorldPosition(f32(cellColumn + 1u), baseY + t);
  }
  if (edge == 2u) {
    return getWorldPosition(baseX + t, f32(cellRow + 1u));
  }
  return getWorldPosition(baseX, baseY + t);
}

fn getEventPosition(cellEvent: u32) -> vec2<f32> {
  let index = (cellEvent >> 3u) & 7u;
  if ((cellEvent & 3u) == 0u) {
    return getCornerPosition(index);
  }
  return getCrossingPosition((cellEvent >> 2u) & 1u, index);
}

fn pushEvent(cellEvent: u32, walk: u32) {
  cellEvents[cellEventCount] = cellEvent;
  cellEventWalks[cellEventCount] = walk;
  cellEventCount++;
}

fn buildEvents() {
  cellEventCount = 0u;
  for (var walk = 0u; walk < 4u; walk++) {
    let startState = bandStates[walk];
    let endState = bandStates[(walk + 1u) % 4u];
    if (startState == 1u) {
      pushEvent(walk << 3u, walk);
    }
    if (startState < endState) {
      if (startState == 0u) {
        pushEvent(1u | (walk << 3u), walk);
      }
      if (endState == 2u) {
        pushEvent(2u | 4u | (walk << 3u), walk);
      }
    } else if (startState > endState) {
      if (startState == 2u) {
        pushEvent(1u | 4u | (walk << 3u), walk);
      }
      if (endState == 0u) {
        pushEvent(2u | (walk << 3u), walk);
      }
    }
  }
}

fn isEdgeCrossed(mask: u32, edge: u32) -> bool {
  let startHigh = ((mask >> EDGE_START[edge]) & 1u) != 0u;
  let endHigh = ((mask >> EDGE_END[edge]) & 1u) != 0u;
  return startHigh != endHigh;
}

// Other endpoint of the level segment through the crossing on 'edge'. Saddles (high corners on one
// diagonal) pair edges by the centre rule: joined cuts off the low corners, separated the high.
fn getPartnerEdge(level: u32, edge: u32) -> u32 {
  let mask = bandMasks[level];
  if (mask == 5u || mask == 10u) {
    let pairsAdjacent = (mask == 5u) == bandJoined[level];
    return select(3u - edge, edge ^ 1u, pairsAdjacent);
  }
  for (var other = 0u; other < 4u; other++) {
    if (other != edge && isEdgeCrossed(mask, other)) {
      return other;
    }
  }
  return edge;
}

fn findCrossingEvent(level: u32, edge: u32) -> u32 {
  for (var i = 0u; i < cellEventCount; i++) {
    let cellEvent = cellEvents[i];
    if ((cellEvent & 3u) != 0u && ((cellEvent >> 2u) & 1u) == level && ((cellEvent >> 3u) & 7u) == edge) {
      return i;
    }
  }
  return NO_EVENT;
}

fn getBandState(value: f32) -> u32 {
  if (bandHasLow && value < bandLow) {
    return 0u;
  }
  if (bandHasHigh && value >= bandHigh) {
    return 2u;
  }
  return 1u;
}

// True when the current band's region in a cell with these corner values has zero area: one or two
// corners are in the band (so every region vertex lies on the segment between them), every in-band
// corner equals the lower break exactly (so the crossings sit on those corners), and none is above
// the band.
fn isSliverBand(corners: array<f32, 4>) -> bool {
  var bandCornerCount = 0u;
  for (var corner = 0u; corner < 4u; corner++) {
    let state = getBandState(corners[corner]);
    if (state == 2u) {
      return false;
    }
    if (state == 1u) {
      if (!bandHasLow || corners[corner] != bandLow) {
        return false;
      }
      bandCornerCount++;
    }
  }
  return bandCornerCount == 1u || bandCornerCount == 2u;
}

// True when the neighbouring cell across the cell border 'walk' (0 bottom, 1 right, 2 top, 3 left)
// exists and has no nodata corner, so it carries the same border edge in the same band.
fn isBorderShared(walk: u32) -> bool {
  var neighbourColumn = i32(cellColumn);
  var neighbourRow = i32(cellRow);
  if (walk == 0u) {
    neighbourRow -= 1;
  } else if (walk == 1u) {
    neighbourColumn += 1;
  } else if (walk == 2u) {
    neighbourRow += 1;
  } else {
    neighbourColumn -= 1;
  }
  if (neighbourColumn < 0 || neighbourRow < 0 || neighbourColumn >= i32(CELL_COLUMNS) || neighbourRow >= i32(HEIGHT - 1u)) {
    return false;
  }
  let neighbour = u32(neighbourRow) * CELL_COLUMNS + u32(neighbourColumn);
  // The neighbour's two corners on the shared border are this (valid) cell's border corners;
  // only its two far corners, which sit at the same corner indices as the border, are loaded.
  // isSliverBand does not depend on the corner order.
  var neighbourValues: array<f32, 4>;
  neighbourValues[0] = cornerValues[EDGE_START[walk]];
  neighbourValues[1] = cornerValues[EDGE_END[walk]];
  for (var farCorner = 0u; farCorner < 2u; farCorner++) {
    let sample = getSampleIndex(neighbour, select(EDGE_START[walk], EDGE_END[walk], farCorner == 1u));
    if (!isSampleValid(sample)) {
      return false;
    }
    neighbourValues[2u + farCorner] = values[valuesOffset + sample];
  }
  // A zero-area neighbour region writes no edges, so this cell must keep its own.
  return !isSliverBand(neighbourValues);
}

fn emitPieceEdge(a: vec2<f32>, b: vec2<f32>, isLevel: bool, walk: u32, shouldWrite: bool, band: u32) {
  if (a.x == b.x && a.y == b.y) {
    return;
  }
  if (bandIsSliver || (!isLevel && isBorderShared(walk))) {
    return;
  }
  if (shouldWrite) {
    writeEdge(edgeBase + cellEdgeCount, a, b, band);
  }
  cellEdgeCount++;
}

fn pushPieceVertex(cellEvent: u32, walk: u32, shouldWrite: bool, baseTriangle: u32, band: u32) {
  var position = vec2<f32>(0.0, 0.0);
  if (shouldWrite || EMIT_EDGES) {
    position = getEventPosition(cellEvent);
  }
  if (pieceVertexCount == 0u) {
    pieceFirst = position;
  } else {
    if (EMIT_EDGES) {
      emitPieceEdge(piecePrevious, position, pieceNextIsLevel, pieceLastWalk, shouldWrite, band);
    }
    if (pieceVertexCount >= 2u) {
      if (shouldWrite) {
        writeTriangle(baseTriangle + emittedTriangles, pieceFirst, piecePrevious, position, band);
      }
      emittedTriangles++;
    }
  }
  pieceNextIsLevel = false;
  pieceLastWalk = walk;
  piecePrevious = position;
  pieceVertexCount++;
}

// Closing edge of the piece: a level segment when the piece has crossings, else a border edge.
fn finishPiece(isLevel: bool, shouldWrite: bool, band: u32) {
  if (EMIT_EDGES && pieceVertexCount >= 3u) {
    emitPieceEdge(piecePrevious, pieceFirst, isLevel, pieceLastWalk, shouldWrite, band);
  }
}

fn setupBand(band: u32, breakCount: u32) -> bool {
  let hasLow = band > 0u;
  let hasHigh = band < breakCount;
  bandLow = select(0.0, breaks[breaksOffset + max(band, 1u) - 1u], hasLow);
  bandHigh = select(0.0, breaks[breaksOffset + min(band, MAXIMUM_BREAK_COUNT - 1u)], hasHigh);
  if (hasLow && hasHigh && bandLow >= bandHigh) {
    return false;
  }
  var lowMask = 0u;
  var highMask = 0u;
  for (var corner = 0u; corner < 4u; corner++) {
    let value = cornerValues[corner];
    var state = 1u;
    if (hasLow && value < bandLow) {
      state = 0u;
    } else if (hasHigh && value >= bandHigh) {
      state = 2u;
    }
    bandStates[corner] = state;
    lowMask |= select(0u, 1u << corner, state >= 1u);
    highMask |= select(0u, 1u << corner, state == 2u);
  }
  bandHasLow = hasLow;
  bandHasHigh = hasHigh;
  bandIsSliver = isSliverBand(cornerValues);
  bandMasks[0] = lowMask;
  bandMasks[1] = highMask;
  let centre = ((cornerValues[0] + cornerValues[1]) + (cornerValues[2] + cornerValues[3])) * 0.25;
  bandJoined[0] = centre >= bandLow;
  bandJoined[1] = centre >= bandHigh;
  return true;
}

// Triangles of one band in the loaded cell; writes them from 'baseTriangle' when 'shouldWrite' is set.
fn processBand(shouldWrite: bool, baseTriangle: u32, band: u32, breakCount: u32) -> u32 {
  emittedTriangles = 0u;
  if (!setupBand(band, breakCount)) {
    return 0u;
  }
  buildEvents();
  var hasCrossing = false;
  for (var i = 0u; i < cellEventCount; i++) {
    if ((cellEvents[i] & 3u) != 0u) {
      hasCrossing = true;
    }
  }
  if (!hasCrossing) {
    if (cellEventCount == 4u) {
      pieceVertexCount = 0u;
      for (var i = 0u; i < 4u; i++) {
        pushPieceVertex(cellEvents[i], cellEventWalks[i], shouldWrite, baseTriangle, band);
      }
      finishPiece(false, shouldWrite, band);
    }
    return emittedTriangles;
  }
  var visited = 0u;
  for (var i = 0u; i < cellEventCount; i++) {
    if ((cellEvents[i] & 3u) != 1u || ((visited >> i) & 1u) != 0u) {
      continue;
    }
    pieceVertexCount = 0u;
    var current = i;
    pieceNextIsLevel = false;
    for (var arc = 0u; arc < 8u; arc++) {
      if (arc > 0u) {
        pieceNextIsLevel = true;
      }
      visited |= 1u << current;
      var exitIndex = current;
      for (var stepIndex = 0u; stepIndex < 12u; stepIndex++) {
        pushPieceVertex(cellEvents[exitIndex], cellEventWalks[exitIndex], shouldWrite, baseTriangle, band);
        if ((cellEvents[exitIndex] & 3u) == 2u) {
          break;
        }
        exitIndex = (exitIndex + 1u) % cellEventCount;
      }
      let exitEvent = cellEvents[exitIndex];
      let level = (exitEvent >> 2u) & 1u;
      current = findCrossingEvent(level, getPartnerEdge(level, (exitEvent >> 3u) & 7u));
      if (current == i || current == NO_EVENT) {
        break;
      }
    }
    finishPiece(true, shouldWrite, band);
  }
  return emittedTriangles;
}

fn getSampleIndex(cell: u32, corner: u32) -> u32 {
  let column = cell % CELL_COLUMNS + select(0u, 1u, corner == 1u || corner == 2u);
  let row = cell / CELL_COLUMNS + select(0u, 1u, corner >= 2u);
  return row * WIDTH + column;
}

fn isSampleValid(sample: u32) -> bool {
  // Infinities cannot be interpolated: they are nodata, like NaN.
  let sampleValue = values[valuesOffset + sample];
  var isValid = isFiniteValue(sampleValue) && !isNoDataValue(sampleValue);
  ${e?`isValid = isValid && validity[validityOffset + sample] != 0u;`:``}
  return isValid;
}

// Loads the four corners of a marching-squares cell; false when any corner is nodata.
fn loadCell(cell: u32) -> bool {
  cellColumn = cell % CELL_COLUMNS;
  cellRow = cell / CELL_COLUMNS;
  for (var corner = 0u; corner < 4u; corner++) {
    let sample = getSampleIndex(cell, corner);
    if (!isSampleValid(sample)) {
      return false;
    }
    cornerValues[corner] = values[valuesOffset + sample];
  }
  return true;
}

// Triangles of every emitted band in a cell, in band order.
fn processCell(cell: u32, shouldWrite: bool, baseTriangle: u32) -> u32 {
  cellEdgeCount = 0u;
  if (!loadCell(cell)) {
    return 0u;
  }
  let breakCount = getParameterCount(params[paramsOffset]);
  var minimumClass = 0xffffffffu;
  var maximumClass = 0u;
  for (var corner = 0u; corner < 4u; corner++) {
    let bandClass = countBreaksBelow(0u, breakCount, cornerValues[corner], false);
    minimumClass = min(minimumClass, bandClass);
    maximumClass = max(maximumClass, bandClass);
  }
  let firstBand = max(minimumClass, getParameterCount(params[paramsOffset + 1u]));
  let lastBand = min(maximumClass, min(getParameterCount(params[paramsOffset + 2u]), breakCount));
  var total = 0u;
  for (var band = firstBand; band <= lastBand; band++) {
    total += processBand(shouldWrite, baseTriangle + total, band, breakCount);
  }
  return total;
}`}var x=`GPUIsobands`,S=class{id;props;maximumBreakCount;cellCount;triangleCapacity;edgeCapacity;constructor(e){this.id=e.id??`isobands`,this.props=e;let{id:t}=this,{output:n}=e,r=d(t,e.grid),{width:i,height:a}=e.grid;p(t,`grid width`,i,2,2147483647),p(t,`grid height`,a,2,2147483647),this.cellCount=(i-1)*(a-1),_(t,`values`,e.values,`float32`,r),h(t,e.noDataValue),_(t,`validity`,e.validity,`uint32`,r),_(t,`breaks`,e.breaks,`float32`,1),this.maximumBreakCount=e.breaks.length,_(t,`parameters`,e.parameters,`float32`,12),_(t,`output.bandClasses`,n.bandClasses,`uint32`,r);let o=[n.triangles,n.triangleBands,n.status?.count,n.status?.overflow],s=o.some(e=>e!==void 0);if(s&&o.some(e=>e===void 0))throw Error(`${t} geometry output needs triangles, triangleBands, status.count and status.overflow together`);let c=[n.edges,n.edgeBands,n.status?.edgeCount,n.status?.edgeOverflow],l=c.some(e=>e!==void 0);if(l&&c.some(e=>e===void 0))throw Error(`${t} edge output needs edges, edgeBands, status.edgeCount and status.edgeOverflow together`);if(!l&&n.status?.edgeRequiredCount)throw Error(`${t} status.edgeRequiredCount belongs to the edge output`);if(!n.bandClasses&&!s&&!l)throw Error(`${t} needs output.bandClasses, the geometry output or the edge output`);if(this.edgeCapacity=n.edgeBands?.length??0,l){if(n.edges.format!==`float32x4`)throw Error(`${t} output.edges must be float32x4`);if(_(t,`output.edgeBands`,n.edgeBands,`uint32`,1),n.edges.length<this.edgeCapacity)throw Error(`${t} output.edges must hold edgeBands.length (${this.edgeCapacity}) rows`);for(let[e,r]of[[`status.edgeCount`,n.status?.edgeCount],[`status.edgeOverflow`,n.status?.edgeOverflow],[`status.edgeRequiredCount`,n.status?.edgeRequiredCount]])_(t,`output.${e}`,r,`uint32`,1);if(n.edges.byteStride!==16||n.edges.rowByteLength!==16)throw Error(`${t} output.edges must be packed float32x4`)}if(!s&&(n.status?.requiredCount||n.status?.vertexCount))throw Error(`${t} status.requiredCount and status.vertexCount belong to the geometry output`);if(this.triangleCapacity=n.triangleBands?.length??0,s){if(n.triangles.format!==`float32x2`)throw Error(`${t} output.triangles must be float32x2`);if(_(t,`output.triangleBands`,n.triangleBands,`uint32`,1),n.triangles.length<3*this.triangleCapacity)throw Error(`${t} output.triangles must hold 3 * triangleBands.length (${3*this.triangleCapacity}) rows`);for(let[e,r]of[[`status.count`,n.status?.count],[`status.overflow`,n.status?.overflow],[`status.requiredCount`,n.status?.requiredCount],[`status.vertexCount`,n.status?.vertexCount]])_(t,`output.${e}`,r,`uint32`,1);let e=n.triangles;if(e.byteStride!==8||e.rowByteLength!==8)throw Error(`${t} output.triangles must be packed float32x2`)}g(t,[...[n.bandClasses,n.triangleBands,n.status?.count,n.status?.overflow,n.status?.requiredCount,n.status?.vertexCount,n.edgeBands,n.status?.edgeCount,n.status?.edgeOverflow,n.status?.edgeRequiredCount],n.triangles,n.edges],[e.values,e.validity,e.breaks,e.parameters])}getCommandNodes(e){let{id:t,props:n,maximumBreakCount:s,cellCount:c,triangleCapacity:l,edgeCapacity:u}=this,{output:d}=n,{width:p,height:h}=n.grid;f(t,e,[n.values,n.validity,n.breaks,n.parameters,d.bandClasses,d.triangles,d.triangleBands,d.status?.count,d.status?.overflow,d.status?.requiredCount,d.status?.vertexCount,d.edges,d.edgeBands,d.status?.edgeCount,d.status?.edgeOverflow,d.status?.edgeRequiredCount]);let g=`const EMIT_EDGES: bool = false;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {}`,_=`fn writeTriangle(index: u32, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, band: u32) {}`,y=`const WIDTH: u32 = ${p}u;
const HEIGHT: u32 = ${h}u;
const MAXIMUM_BREAK_COUNT: u32 = ${s}u;
const NO_DATA_CLASS: u32 = 0xffffffffu;
${m(n.noDataValue)}
${v(`countBreaksBelow`,`breaks`)}`,S=[{name:`values`,view:n.values,type:`f32`,access:`read`},{name:`breaks`,view:n.breaks,type:`f32`,access:`read`},{name:`params`,view:n.parameters,type:`f32`,access:`read`}];n.validity&&S.push({name:`validity`,view:n.validity,type:`u32`,access:`read`});let C=[];if(d.bandClasses&&C.push(o(e,{id:`${t}-classes`,operation:x,variant:`classes`,bindings:[...S,{name:`classesOut`,view:d.bandClasses,type:`u32`,access:`read_write`}],invocationCount:p*h,declarations:y,body:`let value = values[valuesOffset + index];
  var isValid = isFiniteValue(value) && !isNoDataValue(value);
  ${n.validity?`isValid = isValid && validity[validityOffset + index] != 0u;`:``}
  let breakCountValue = params[paramsOffset];
  let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
  classesOut[classesOutOffset + index] = select(NO_DATA_CLASS, countBreaksBelow(0u, breakCount, value, false), isValid);`})),d.triangles&&d.triangleBands&&d.status?.count&&d.status?.overflow){let s=b(!!n.validity),u=r(e,`${t}-cell-counts`,`uint32`,c),f=r(e,`${t}-cell-offsets`,`uint32`,c),p=r(e,`${t}-total`,`uint32`,1);C.push(o(e,{id:`${t}-count`,operation:x,variant:`count`,bindings:[...S,{name:`cellCountsOut`,view:u,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${y}
${_}
${g}
${s}`,body:`cellCountsOut[cellCountsOutOffset + index] = processCell(index, false, 0u);`})),C.push(...new i({id:`${t}-scan`,input:u,output:f}).getCommandNodes(e)),C.push(o(e,{id:`${t}-total`,operation:x,variant:`total`,bindings:[{name:`cellCountsIn`,view:u,type:`u32`,access:`read`},{name:`cellOffsets`,view:f,type:`u32`,access:`read`},{name:`totalOut`,view:p,type:`u32`,access:`read_write`}],invocationCount:1,body:`totalOut[totalOutOffset] = cellOffsets[cellOffsetsOffset + ${c-1}u] + cellCountsIn[cellCountsInOffset + ${c-1}u];`})),C.push(o(e,{id:`${t}-scatter`,operation:x,variant:`scatter`,bindings:[...S,{name:`cellOffsets`,view:f,type:`u32`,access:`read`},{name:`trianglesOut`,view:d.triangles,type:`f32`,access:`read_write`},{name:`bandsOut`,view:d.triangleBands,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${y}
const CAPACITY: u32 = ${l}u;
${g}
fn writeTriangle(index: u32, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, band: u32) {
  if (index >= CAPACITY) {
    return;
  }
  let offset = trianglesOutOffset + index * 6u;
  trianglesOut[offset] = a.x;
  trianglesOut[offset + 1u] = a.y;
  trianglesOut[offset + 2u] = b.x;
  trianglesOut[offset + 3u] = b.y;
  trianglesOut[offset + 4u] = c.x;
  trianglesOut[offset + 5u] = c.y;
  bandsOut[bandsOutOffset + index] = band;
}
${s}`,body:`processCell(index, true, cellOffsets[cellOffsetsOffset + index]);`})),C.push(a(e,{id:`${t}-publish`,operation:x,requiredCount:p,output:{ids:d.triangleBands,status:{count:d.status.count,overflow:d.status.overflow,requiredCount:d.status.requiredCount}}})),d.status?.vertexCount&&C.push(o(e,{id:`${t}-vertex-count`,operation:x,variant:`vertex-count`,bindings:[{name:`countIn`,view:d.status?.count,type:`u32`,access:`read`},{name:`vertexCountOut`,view:d.status?.vertexCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`vertexCountOut[vertexCountOutOffset] = 3u * countIn[countInOffset];`}))}if(d.edges&&d.edgeBands&&d.status?.edgeCount&&d.status?.edgeOverflow){let s=b(!!n.validity),l=r(e,`${t}-edge-cell-counts`,`uint32`,c),f=r(e,`${t}-edge-cell-offsets`,`uint32`,c),p=r(e,`${t}-edge-total`,`uint32`,1);C.push(o(e,{id:`${t}-edge-count`,operation:x,variant:`edge-count`,bindings:[...S,{name:`edgeCountsOut`,view:l,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${y}
${_}
const EMIT_EDGES: bool = true;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {}
${s}`,body:`processCell(index, false, 0u);
  edgeCountsOut[edgeCountsOutOffset + index] = cellEdgeCount;`})),C.push(...new i({id:`${t}-edge-scan`,input:l,output:f}).getCommandNodes(e)),C.push(o(e,{id:`${t}-edge-total`,operation:x,variant:`edge-total`,bindings:[{name:`cellCountsIn`,view:l,type:`u32`,access:`read`},{name:`cellOffsets`,view:f,type:`u32`,access:`read`},{name:`totalOut`,view:p,type:`u32`,access:`read_write`}],invocationCount:1,body:`totalOut[totalOutOffset] = cellOffsets[cellOffsetsOffset + ${c-1}u] + cellCountsIn[cellCountsInOffset + ${c-1}u];`})),C.push(o(e,{id:`${t}-edge-scatter`,operation:x,variant:`edge-scatter`,bindings:[...S,{name:`cellOffsets`,view:f,type:`u32`,access:`read`},{name:`edgesOut`,view:d.edges,type:`f32`,access:`read_write`},{name:`edgeBandsOut`,view:d.edgeBands,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${y}
const EDGE_CAPACITY: u32 = ${u}u;
${_}
const EMIT_EDGES: bool = true;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {
  if (index >= EDGE_CAPACITY) {
    return;
  }
  let offset = edgesOutOffset + index * 4u;
  edgesOut[offset] = start.x;
  edgesOut[offset + 1u] = start.y;
  edgesOut[offset + 2u] = end.x;
  edgesOut[offset + 3u] = end.y;
  edgeBandsOut[edgeBandsOutOffset + index] = band;
}
${s}`,body:`edgeBase = cellOffsets[cellOffsetsOffset + index];
  processCell(index, true, 0u);`})),C.push(a(e,{id:`${t}-edge-publish`,operation:x,requiredCount:p,output:{ids:d.edgeBands,status:{count:d.status.edgeCount,overflow:d.status.edgeOverflow,requiredCount:d.status.edgeRequiredCount}}}))}return C}};function C(e){return t(e,16)}function w(e){return t(e,16)}var T=class{id;props;constructor(e){if(this.id=e.id??`isoband-rings`,this.props=e,!Number.isInteger(e.edgeCapacity)||e.edgeCapacity<1)throw Error(`${this.id} edgeCapacity must be a positive integer`)}getCommandNodes(t){let{id:i,props:a}=this,{output:o}=a;e(i,`edgeCapacity`,a.edgeCapacity,1,C(t.device)),o?.ringOffsets&&n(i,`ring capacity (output.ringOffsets.length - 1)`,o.ringOffsets.length-1,w(t.device));let c=r(t,`${i}-edges`,`float32x4`,a.edgeCapacity),l=r(t,`${i}-edge-bands`,`uint32`,a.edgeCapacity),u=o.status.edgeCount??r(t,`${i}-edge-count`,`uint32`,1),d=o.status.edgeOverflow??r(t,`${i}-edge-overflow`,`uint32`,1),f=new S({id:`${i}-isobands`,grid:a.grid,values:a.values,validity:a.validity,noDataValue:a.noDataValue,breaks:a.breaks,parameters:a.parameters,output:{edges:c,edgeBands:l,status:{edgeCount:u,edgeOverflow:d}}}),{edgeCount:p,edgeOverflow:m,...h}=o.status,g={...o,status:h},_=new s({id:`${i}-assembly`,endpoints:c,count:u,groups:l,vertexTolerance:a.vertexTolerance??1e-4,interiorSide:`left`,cancelOpposingSegments:!0,splitTouchingRings:a.splitTouchingRings,output:g});return[...f.getCommandNodes(t),..._.getCommandNodes(t)]}};export{S as n,y as r,T as t};