import{E as e,d as t,f as n,u as r}from"./analysis-frame-Blu0tYdw.js";import{a as i,c as a,i as o,o as s}from"./cell-topology-wgsl-BT8_s49b.js";import{_ as c,d as l,h as u,l as d,v as f,w as p,y as m}from"./resources-CwvhEOBe.js";import{t as h}from"./gpu-scan-BuGhTLDQ.js";import{l as g,n as ee,r as _,s as v}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as y}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{n as b,t as x}from"./exact-geometry-wgsl-DOqxBQzc.js";import{a as S}from"./sorted-segment-sums-DwGkmbx2.js";import{i as C,n as w,r as T,t as E}from"./polygon-edge-slabs-Cmf_6bAe.js";import{t as te}from"./stable-lexicographic-index-sort-CcJLc7b1.js";import{t as D}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{n as O,r as k}from"./gpu-point-to-cell-DabDU1o1.js";var A=[3.7,.56,.5,.32,.12,.046,.017,.0066,.00245,94e-5,35e-5,...[1,2,3,4,5].map(e=>35e-5/Math.sqrt(7)**e)].map(e=>Math.fround(.5*e)),j=2**20,M=4e-5,N=1e-6,ne=2e-4,P=3e-5,F=6e-5;function I(e,t,n){let r=Math.max(e,1);return t?r*8+3*t.slotCapacity*8:r*n}var L=`0xffffffffu`;function R(e,t,n){return`
fn ${e}(value: u32) -> u32 {
  var low = 0u;
  var high = ${n}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (${t}[${t}Offset + middle] <= value) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u) { return ${L}; }
  let range = low - 1u;
  if (value >= ${t}[${t}Offset + range + 1u]) { return ${L}; }
  return range;
}`}function re(e){let t=`
const X_SLACK_BASE: f32 = ${g(M)};
const X_SLACK_RELATIVE: f32 = ${g(N)};
fn coverFloorToInt(value: f32) -> i32 { return i32(clamp(floor(value), -2.0e9, 2.0e9)); }
fn coverCeilToInt(value: f32) -> i32 { return i32(clamp(ceil(value), -2.0e9, 2.0e9)); }
`;if(e.family===`quadbin`){let n=e.resolution;return`${t}${o}${i}
const TILE_WIDTH: f32 = ${g(45*2**(3-n))};
const ROW_SCALE: f32 = ${g(2**(1-n))};
const PI: f32 = 3.14159265358979;
const RESOLUTION: u32 = ${n}u;
const Y_SLACK: f32 = ${g(ne)};
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 3u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb + 2u]; }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverTileLatitude(f32(ranges[rb + 1u] + row) + 0.5);
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let y = ranges[rb + 1u] + row;
  return vec2f(coverTileLatitude(f32(y + 1u)) - Y_SLACK, coverTileLatitude(f32(y)) + Y_SLACK);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = i32(ranges[rb + 1u]);
  let first = i32(quadbinGetTileY(highY + Y_SLACK, RESOLUTION)) - origin - 1;
  let last = i32(quadbinGetTileY(lowY - Y_SLACK, RESOLUTION)) - origin + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 3u]) - 1));
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = i32(ranges[rb]);
  let first = coverFloorToInt((lowX - X_SLACK_BASE + 180.0) / TILE_WIDTH) - 1 - origin;
  let last = coverFloorToInt((highX + X_SLACK_BASE + 180.0) / TILE_WIDTH) + 1 - origin;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 2u]) - 1));
}
`}let n=A[e.resolution];return`${t}
const LATTICE_SPACING: f32 = ${g(n)};
const Y_SLACK: f32 = ${g(P)};
const X_MARGIN_SLACK: f32 = ${g(F)};
fn coverRowCount(rb: u32) -> u32 { return ranges[rb + 1u]; }
fn coverColumnCount(rb: u32) -> u32 { return ranges[rb]; }
fn coverOriginLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 2u]); }
fn coverOriginLat(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 3u]); }
fn coverSpacingLng(rb: u32) -> f32 { return bitcast<f32>(ranges[rb + 4u]); }
fn coverRowLine(rb: u32, row: u32) -> f32 {
  return coverOriginLat(rb) + f32(row) * LATTICE_SPACING;
}
// An accepted cell's centre rounds to its lattice row: its latitude is within half a spacing of the
// row line, plus the f32 error of the rounding quotient (3e-7 of a spacing per row) and of the line.
fn coverRowHalfBand(rb: u32) -> f32 {
  return LATTICE_SPACING * (0.501 + 3.0e-7 * f32(ranges[rb + 1u])) + Y_SLACK;
}
fn coverRowBand(rb: u32, row: u32) -> vec2f {
  let line = coverRowLine(rb, row);
  let half = coverRowHalfBand(rb);
  return vec2f(line - half, line + half);
}
fn coverEdgeRows(rb: u32, lowY: f32, highY: f32) -> vec2i {
  let origin = coverOriginLat(rb);
  let half = coverRowHalfBand(rb);
  let first = coverFloorToInt((lowY - half - origin) / LATTICE_SPACING) - 1;
  let last = coverCeilToInt((highY + half - origin) / LATTICE_SPACING) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb + 1u]) - 1));
}
fn coverColumnMargin(rb: u32) -> f32 {
  return coverSpacingLng(rb) * (0.501 + 3.0e-7 * f32(ranges[rb])) + X_MARGIN_SLACK;
}
fn coverColumnRange(rb: u32, lowX: f32, highX: f32) -> vec2i {
  let origin = coverOriginLng(rb);
  let spacing = coverSpacingLng(rb);
  let margin = coverColumnMargin(rb) + X_SLACK_BASE;
  let first = coverFloorToInt((lowX - margin - origin) / spacing) - 1;
  let last = coverCeilToInt((highX + margin - origin) / spacing) + 1;
  return vec2i(max(first, 0), min(last, i32(ranges[rb]) - 1));
}
`}var ie=`
fn coverSortableKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn coverFromSortableKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`,ae=`${ie}
fn coverGetVertex(vertex: u32) -> vec2f {
  let base = positionsOffset + 2u * vertex;
  return vec2f(positions[base], positions[base + 1u]);
}

/** (valid, xLow, xHigh): x extent of the part of edge a-b inside the latitude band, with slack. */
fn coverClipEdge(a: vec2f, b: vec2f, band: vec2f) -> vec3f {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  if ((upper.y < band.x) || (lower.y > band.y)) {
    return vec3f(0.0, 0.0, 0.0);
  }
  var x0 = lower.x;
  var x1 = upper.x;
  let dy = upper.y - lower.y;
  if (dy > 0.0) {
    let dx = upper.x - lower.x;
    x0 = lower.x + dx * clamp((band.x - lower.y) / dy, 0.0, 1.0);
    x1 = lower.x + dx * clamp((band.y - lower.y) / dy, 0.0, 1.0);
  }
  let slack = X_SLACK_BASE + X_SLACK_RELATIVE * abs(upper.x - lower.x);
  return vec3f(1.0, min(x0, x1) - slack, max(x0, x1) + slack);
}

/** x where edge a-b crosses latitude line; only called when (a.y <= line) != (b.y <= line). */
fn coverGetCrossing(a: vec2f, b: vec2f, line: f32) -> f32 {
  let aIsLower = a.y <= b.y;
  let lower = select(b, a, aIsLower);
  let upper = select(a, b, aIsLower);
  let t = clamp((line - lower.y) / (upper.y - lower.y), 0.0, 1.0);
  return lower.x + (upper.x - lower.x) * t;
}

`;function z(e,t,n){let{id:r,featureCount:i,candidateCapacity:a,positions:o}=t,{slotCapacity:s,force:c}=t.scanline,{counts:l,ranges:u}=n,d=`GPUCellCover`,f=o.length,m=3*s,_=`rangesOffset + ${i}u * 8u`,y=[],b=re(t),x=e=>`${r}-scanline-${e}`,C=(t,n)=>p(e,x(t),`uint32`,Math.max(n,1)),w=`const FEATURE_COUNT: u32 = ${i}u;
const VERTEX_COUNT: u32 = ${f}u;
const STRIDE: u32 = 8u;
const SLOT_CAPACITY: u32 = ${s}u;
const INVALID_ROW: u32 = ${s}u;`,T=[{name:`positions`,view:o,type:`f32`,access:`read`}],E=C(`edge-feature`,f+1),D=C(`edge-next`,f+1);y.push(v(e,{id:x(`edges`),operation:d,variant:`scanline-edges`,bindings:[{name:`ringOffsets`,view:t.ringOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:t.polygonOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:t.featureOffsets,type:`u32`,access:`read`},{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`counts`,view:l,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read_write`},{name:`edgeNext`,view:D,type:`u32`,access:`read_write`}],invocationCount:f+1,declarations:`${w}
${R(`coverFindRing`,`ringOffsets`,t.ringOffsets.length-1)}
${R(`coverFindPolygon`,`polygonOffsets`,t.polygonOffsets.length-1)}
${R(`coverFindFeatureOfPolygon`,`featureOffsets`,i)}`,body:`var feature = ${L};
  var next = ${L};
  if (index < VERTEX_COUNT) {
    let ring = coverFindRing(index);
    if (ring != ${L}) {
      let polygon = coverFindPolygon(ring);
      if (polygon != ${L}) {
        let candidate = coverFindFeatureOfPolygon(polygon);
        if (candidate != ${L}) {
          let rb = rangesOffset + candidate * STRIDE;
          // Non-finite vertices (flag bit 0) or no lattice: the feature keeps the bounding box.
          if (counts[countsOffset + candidate] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
            let ringStart = ringOffsets[ringOffsetsOffset + ring];
            let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
            feature = candidate;
            next = select(index + 1u, ringStart, (index + 1u) >= ringEnd);
          }
        }
      }
    }
  }
  edgeFeature[edgeFeatureOffset + index] = feature;
  edgeNext[edgeNextOffset + index] = next;`}));let O=C(`edge-slots`,f+1),k=C(`demand`,2*i);y.push(ee(e,{id:x(`clear-demand`),operation:d,view:k,type:`u32`,value:`0u`}),v(e,{id:x(`rows`),operation:d,variant:`scanline-rows`,bindings:[...T,{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeNext`,view:D,type:`u32`,access:`read`},{name:`edgeSlots`,view:O,type:`u32`,access:`read_write`},{name:`demand`,view:k,type:`atomic<u32>`,access:`read_write`}],invocationCount:f+1,declarations:`${w}
${b}${ae}`,body:`var count = 0u;
  let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature != ${L}) {
    let a = coverGetVertex(index);
    let b = coverGetVertex(edgeNext[edgeNextOffset + index]);
    let rows = coverEdgeRows(rangesOffset + feature * STRIDE, min(a.y, b.y), max(a.y, b.y));
    if (rows.x <= rows.y) {
      count = u32(rows.y - rows.x + 1);
      // Exact 64-bit total: a wrapped low word carries one into the high word.
      let before = atomicAdd(&demand[demandOffset + 2u * feature], count);
      if (before + count < before) {
        atomicAdd(&demand[demandOffset + 2u * feature + 1u], 1u);
      }
    }
  }
  edgeSlots[edgeSlotsOffset + index] = count;`}));let A=C(`cost`,i+1),j=C(`cost-starts`,i+1);y.push(v(e,{id:x(`choose`),operation:d,variant:`scanline-choose`,bindings:[{name:`counts`,view:l,type:`u32`,access:`read`},{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`demand`,view:k,type:`u32`,access:`read`},{name:`cost`,view:A,type:`u32`,access:`read_write`}],invocationCount:i+1,declarations:`${w}
const FORCE: bool = ${c};
const COST_RATIO: f32 = ${g(4)};
${b}`,body:`var featureCost = 0u;
  if (index < FEATURE_COUNT) {
    let rb = rangesOffset + index * STRIDE;
    if (counts[countsOffset + index] > 0u && (ranges[rb + 5u] & 1u) == 0u) {
      let rows = coverRowCount(rb);
      let slots = select(demand[demandOffset + 2u * index], 0xffffffffu, (demand[demandOffset + 2u * index + 1u] != 0u));
      let cells = f32(rows) * f32(coverColumnCount(rb));
      // Work of the scanline enumeration (events and runs per row, one slot per edge row) against
      // the cells of the bounding box. The row count and the slots must also fit the stage.
      let cheaper = cells > COST_RATIO * (2.0 * f32(rows) + f32(slots));
      if ((FORCE || cheaper) && rows <= SLOT_CAPACITY && slots <= SLOT_CAPACITY) {
        featureCost = min(slots + rows, SLOT_CAPACITY + 1u);
      }
    }
  }
  cost[costOffset + index] = featureCost;`}),...new h({id:x(`cost-scan`),input:A,output:j,mode:`exclusive`}).getCommandNodes(e));let M=C(`count-carry`,i);y.push(v(e,{id:x(`select`),operation:d,variant:`scanline-select`,bindings:[{name:`counts`,view:l,type:`u32`,access:`read_write`},{name:`ranges`,view:u,type:`u32`,access:`read_write`},{name:`cost`,view:A,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`countCarry`,view:M,type:`u32`,access:`read_write`}],invocationCount:i,declarations:w,body:`let rb = rangesOffset + index * STRIDE;
  countCarry[countCarryOffset + index] = 0u;
  let featureCost = cost[costOffset + index];
  // The sum of capped costs stays below 2^32 because (featureCount + 1) * (capacity + 1) does.
  let selected = featureCost > 0u && (costStarts[costStartsOffset + index] + featureCost) <= SLOT_CAPACITY;
  ranges[rb + 5u] = (ranges[rb + 5u] & 1u) | select(0u, 2u, selected);
  ranges[rb + 6u] = 0u;
  ranges[rb + 7u] = 0u;
  if (selected) {
    counts[countsOffset + index] = 0u;
  }`}));let N=C(`edge-offsets`,f+1);y.push(v(e,{id:x(`slots`),operation:d,variant:`scanline-slots`,bindings:[{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeSlots`,view:O,type:`u32`,access:`read_write`}],invocationCount:f+1,declarations:w,body:`let feature = edgeFeature[edgeFeatureOffset + index];
  if (feature == ${L} || (ranges[rangesOffset + feature * STRIDE + 5u] & 2u) == 0u) {
    edgeSlots[edgeSlotsOffset + index] = 0u;
  }`}),...new h({id:x(`edge-scan`),input:O,output:N,mode:`exclusive`}).getCommandNodes(e));let ne=C(`event-row`,m),P=C(`event-position`,m);y.push(v(e,{id:x(`events`),operation:d,variant:`scanline-events`,bindings:[...T,{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`edgeFeature`,view:E,type:`u32`,access:`read`},{name:`edgeNext`,view:D,type:`u32`,access:`read`},{name:`edgeOffsets`,view:N,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`eventRow`,view:ne,type:`u32`,access:`read_write`},{name:`eventPosition`,view:P,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${w}
${b}${ae}`,body:`var rowOpen = INVALID_ROW;
  var rowClose = INVALID_ROW;
  var rowCross = INVALID_ROW;
  var keyOpen = 0u;
  var keyClose = 0u;
  var keyCross = 0u;
  if (index < edgeOffsets[edgeOffsetsOffset + VERTEX_COUNT]) {
    // The edge owning slot index: the last edge whose first slot is at most index.
    var low = 0u;
    var high = VERTEX_COUNT;
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (edgeOffsets[edgeOffsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let edge = low - 1u;
    let feature = edgeFeature[edgeFeatureOffset + edge];
    if (feature != ${L}) {
      let rb = rangesOffset + feature * STRIDE;
      let a = coverGetVertex(edge);
      let b = coverGetVertex(edgeNext[edgeNextOffset + edge]);
      let rows = coverEdgeRows(rb, min(a.y, b.y), max(a.y, b.y));
      let row = rows.x + i32(index - edgeOffsets[edgeOffsetsOffset + edge]);
      if (row <= rows.y) {
        let rowKey = costStarts[costStartsOffset + feature] + u32(row);
        let clip = coverClipEdge(a, b, coverRowBand(rb, u32(row)));
        if (clip.x > 0.0) {
          rowOpen = rowKey;
          rowClose = rowKey;
          keyOpen = coverSortableKey(clip.y);
          keyClose = coverSortableKey(clip.z);
        }
        let line = coverRowLine(rb, u32(row));
        if ((a.y <= line) != (b.y <= line)) {
          rowCross = rowKey;
          keyCross = coverSortableKey(coverGetCrossing(a, b, line));
        }
      }
    }
  }
  let base = 3u * index;
  eventRow[eventRowOffset + base] = rowOpen;
  eventRow[eventRowOffset + base + 1u] = rowClose;
  eventRow[eventRowOffset + base + 2u] = rowCross;
  eventPosition[eventPositionOffset + base] = keyOpen;
  eventPosition[eventPositionOffset + base + 1u] = keyClose;
  eventPosition[eventPositionOffset + base + 2u] = keyCross;`}));let F=C(`event-ids`,m);y.push(v(e,{id:x(`iota`),operation:d,variant:`scanline-iota`,bindings:[{name:`eventIds`,view:F,type:`u32`,access:`read_write`}],invocationCount:m,body:`eventIds[eventIdsOffset + index] = index;`}));let I=te(e,{id:x(`sort`),operation:d,indices:F,keys:[{view:ne,keyBits:S(s)},{view:P}]});y.push(...I.nodes);let z=I.sortedIndices,B=I.sortedPrimaryKeys,V=C(`depth-delta`,m),H=C(`crossing-flag`,m),U=C(`segment-flags`,m),W=C(`depth-sum`,m),G=C(`crossing-sum`,m);y.push(v(e,{id:x(`flags`),operation:d,variant:`scanline-flags`,bindings:[{name:`sortedIndices`,view:z,type:`u32`,access:`read`},{name:`sortedRows`,view:B,type:`u32`,access:`read`},{name:`depthDelta`,view:V,type:`u32`,access:`read_write`},{name:`crossingFlag`,view:H,type:`u32`,access:`read_write`},{name:`segmentFlags`,view:U,type:`u32`,access:`read_write`}],invocationCount:m,declarations:w,body:`let row = sortedRows[sortedRowsOffset + index];
  let valid = row < INVALID_ROW;
  // Event e is slot e / 3: kind 0 opens a span, 1 closes it, 2 is a crossing.
  let kind = sortedIndices[sortedIndicesOffset + index] % 3u;
  var head = index == 0u;
  if (!head) {
    head = row != sortedRows[sortedRowsOffset + index - 1u];
  }
  segmentFlags[segmentFlagsOffset + index] = select(0u, 1u, head);
  depthDelta[depthDeltaOffset + index] = select(0u, select(0u, select(0xffffffffu, 1u, (kind == 0u)), (kind < 2u)), valid);
  crossingFlag[crossingFlagOffset + index] = select(0u, 1u, (valid && kind == 2u));`}),...new h({id:x(`depth-scan`),input:V,output:W,mode:`inclusive`,segmentFlags:U}).getCommandNodes(e),...new h({id:x(`crossing-scan`),input:H,output:G,mode:`inclusive`,segmentFlags:U}).getCommandNodes(e));let K=C(`covered`,m),q=C(`gap-columns`,2*m);y.push(v(e,{id:x(`cover-state`),operation:d,variant:`scanline-cover-state`,bindings:[{name:`sortedRows`,view:B,type:`u32`,access:`read`},{name:`depthSum`,view:W,type:`u32`,access:`read`},{name:`crossingSum`,view:G,type:`u32`,access:`read`},{name:`covered`,view:K,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${w}
const EVENT_COUNT: u32 = ${m}u;`,body:`let row = sortedRows[sortedRowsOffset + index];
  var keep = false;
  // Depth is never negative after all events of one position, so nonzero means a span covers.
  if (row < INVALID_ROW && (index + 1u) < EVENT_COUNT && sortedRows[sortedRowsOffset + index + 1u] == row) {
    keep = depthSum[depthSumOffset + index] != 0u || (crossingSum[crossingSumOffset + index] & 1u) != 0u;
  }
  covered[coveredOffset + index] = select(0u, 1u, keep);`}),v(e,{id:x(`gaps`),operation:d,variant:`scanline-gaps`,bindings:[{name:`sortedIndices`,view:z,type:`u32`,access:`read`},{name:`sortedRows`,view:B,type:`u32`,access:`read`},{name:`eventPosition`,view:P,type:`u32`,access:`read`},{name:`ranges`,view:u,type:`u32`,access:`read`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`covered`,view:K,type:`u32`,access:`read_write`},{name:`gapColumns`,view:q,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${w}
${b}${ie}
${R(`coverFindCostFeature`,`costStarts`,i)}`,body:`if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let lowKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index]];
  let highKey = eventPosition[eventPositionOffset + sortedIndices[sortedIndicesOffset + index + 1u]];
  var keep = 0u;
  // Zero-length gaps add no cell the neighbouring gaps and spans do not already contain.
  if (lowKey < highKey) {
    let feature = coverFindCostFeature(sortedRows[sortedRowsOffset + index]);
    let columns = coverColumnRange(rangesOffset + feature * STRIDE, coverFromSortableKey(lowKey), coverFromSortableKey(highKey));
    if (columns.x <= columns.y) {
      keep = 1u;
      gapColumns[gapColumnsOffset + 2u * index] = u32(columns.x);
      gapColumns[gapColumnsOffset + 2u * index + 1u] = u32(columns.y);
    }
  }
  covered[coveredOffset + index] = keep;`}));let J=C(`gap-slots`,m);y.push(...new h({id:x(`gap-scan`),input:K,output:J,mode:`exclusive`}).getCommandNodes(e),v(e,{id:x(`compact`),operation:d,variant:`scanline-compact`,bindings:[{name:`covered`,view:K,type:`u32`,access:`read`},{name:`gapSlots`,view:J,type:`u32`,access:`read`},{name:`sortedRows`,view:B,type:`u32`,access:`read`},{name:`gapColumns`,view:q,type:`u32`,access:`read`},{name:`ranges`,view:u,type:`u32`,access:`read_write`}],invocationCount:m,declarations:w,body:`if (covered[coveredOffset + index] == 0u) {
    return;
  }
  let run = ${_} + gapSlots[gapSlotsOffset + index] * 8u;
  ranges[run] = sortedRows[sortedRowsOffset + index];
  ranges[run + 1u] = gapColumns[gapColumnsOffset + 2u * index];
  ranges[run + 2u] = gapColumns[gapColumnsOffset + 2u * index + 1u];`}));let Y=C(`run-lengths`,m),X=C(`run-offsets`,m),Z=C(`run-flags`,m),Q=`(gapSlots[gapSlotsOffset + ${m-1}u] + covered[coveredOffset + ${m-1}u])`;return y.push(v(e,{id:x(`runs`),operation:d,variant:`scanline-runs`,bindings:[{name:`ranges`,view:u,type:`u32`,access:`read_write`},{name:`costStarts`,view:j,type:`u32`,access:`read`},{name:`covered`,view:K,type:`u32`,access:`read`},{name:`gapSlots`,view:J,type:`u32`,access:`read`},{name:`runLengths`,view:Y,type:`u32`,access:`read_write`},{name:`runFlags`,view:Z,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`${w}
${R(`coverFindCostFeature`,`costStarts`,i)}`,body:`var runLength = 0u;
  var isHead = false;
  if (index < ${Q}) {
    let run = ${_} + index * 8u;
    let rowKey = ranges[run];
    let columnLow = ranges[run + 1u];
    let columnHigh = ranges[run + 2u];
    let feature = coverFindCostFeature(rowKey);
    var start = columnLow;
    isHead = true;
    if (index > 0u) {
      let previous = run - 8u;
      let previousRow = ranges[previous];
      if (previousRow == rowKey) {
        start = max(start, ranges[previous + 2u] + 1u);
      }
      isHead = coverFindCostFeature(previousRow) != feature;
    }
    runLength = select(0u, columnHigh + 1u - start, (columnHigh >= start));
    ranges[run + 3u] = feature;
    ranges[run + 5u] = start;
    ranges[run + 6u] = rowKey - costStarts[costStartsOffset + feature];
  }
  runLengths[runLengthsOffset + index] = runLength;
  runFlags[runFlagsOffset + index] = select(0u, 1u, isHead);`}),...new h({id:x(`run-scan`),input:Y,output:X,mode:`exclusive`,segmentFlags:Z}).getCommandNodes(e),v(e,{id:x(`finish`),operation:d,variant:`scanline-finish`,bindings:[{name:`ranges`,view:u,type:`atomic<u32>`,access:`read_write`},{name:`runOffsets`,view:X,type:`u32`,access:`read`},{name:`runLengths`,view:Y,type:`u32`,access:`read`},{name:`covered`,view:K,type:`u32`,access:`read`},{name:`gapSlots`,view:J,type:`u32`,access:`read`},{name:`counts`,view:l,type:`atomic<u32>`,access:`read_write`},{name:`countCarry`,view:M,type:`atomic<u32>`,access:`read_write`}],invocationCount:m,declarations:w,body:`if (index >= ${Q}) {
    return;
  }
  let run = ${_} + index * 8u;
  let feature = atomicLoad(&ranges[run + 3u]);
  atomicStore(&ranges[run + 4u], runOffsets[runOffsetsOffset + index]);
  let featureBase = rangesOffset + feature * STRIDE;
  if (index == 0u || atomicLoad(&ranges[run - 8u + 3u]) != feature) {
    atomicStore(&ranges[featureBase + 6u], index);
  }
  atomicAdd(&ranges[featureBase + 7u], 1u);
  // Exact 64-bit candidate count: a wrapped low word carries one into the high word.
  let runLength = runLengths[runLengthsOffset + index];
  let before = atomicAdd(&counts[countsOffset + feature], runLength);
  if (before + runLength < before) {
    atomicAdd(&countCarry[countCarryOffset + feature], 1u);
  }`}),v(e,{id:x(`clamp`),operation:d,variant:`scanline-clamp`,bindings:[{name:`ranges`,view:u,type:`u32`,access:`read_write`},{name:`counts`,view:l,type:`u32`,access:`read_write`},{name:`countCarry`,view:M,type:`u32`,access:`read`}],invocationCount:i,declarations:`${w}
const CAPACITY_PLUS_ONE: u32 = ${a+1}u;`,body:`let flagIndex = rangesOffset + index * STRIDE + 5u;
  if ((ranges[flagIndex] & 2u) != 0u) {
    counts[countsOffset + index] = min(counts[countsOffset + index], CAPACITY_PLUS_ONE);
    // 2^32 or more candidates: far beyond any capacity, and the 32-bit run offsets of the feature
    // wrapped. Its count saturates (candidate overflow) and it falls back to the bounding box.
    if (countCarry[countCarryOffset + index] != 0u) {
      counts[countsOffset + index] = CAPACITY_PLUS_ONE;
      ranges[flagIndex] = ranges[flagIndex] & 1u;
    }
  }`})),y}function B(e,t){let n=`rangesOffset + ${e}u * 8u`;return`var column = candidateLocal % ${t};
  var row = candidateLocal / ${t};
  if ((ranges[rangeBase + 5u] & 2u) != 0u) {
    // Runs of this feature are [first, first + runCount); offsets are local to the feature, and
    // the run holding the candidate is the last one that starts at or before it.
    let firstRun = ranges[rangeBase + 6u];
    var low = firstRun;
    var high = firstRun + ranges[rangeBase + 7u];
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (ranges[${n} + middle * 8u + 4u] <= candidateLocal) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let run = ${n} + (low - 1u) * 8u;
    column = ranges[run + 5u] + (candidateLocal - ranges[run + 4u]);
    row = ranges[run + 6u];
  }`}var V=`GPUCellCover`,H={center:0,full:1,intersects:2},U=2147483648,W=2e-4;function G(e){return e.scanline?8:5}function K(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`featureOffsets`,view:e.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:e.polygonOffsets,type:`u32`,access:`read`},{name:`ringOffsets`,view:e.ringOffsets,type:`u32`,access:`read`}]}var q=`${y}
${x}
fn coverGetVertex(vertex: u32) -> vec2f {
  let rangeBase = positionsOffset + 2u * vertex;
  return vec2f(positions[rangeBase], positions[rangeBase + 1u]);
}

fn coverIsFinite(p: vec2f) -> bool {
  return isFiniteBits(p.x) && isFiniteBits(p.y) && abs(p.x) < 1e30 && abs(p.y) < 1e30;
}

/** Whether the closed segment a-b meets the open rectangle (lo, hi). */
fn coverSegmentHitsRect(a: vec2f, b: vec2f, lo: vec2f, hi: vec2f) -> bool {
  if (!coverIsFinite(a) || !coverIsFinite(b)) { return false; }
  var lower = -1e30;
  var upper = 1e30;
  let d = b - a;
  if (d.x == 0.0) {
    if (!(lo.x < a.x && a.x < hi.x)) { return false; }
  } else {
    let t1 = (lo.x - a.x) / d.x;
    let t2 = (hi.x - a.x) / d.x;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  if (d.y == 0.0) {
    if (!(lo.y < a.y && a.y < hi.y)) { return false; }
  } else {
    let t1 = (lo.y - a.y) / d.y;
    let t2 = (hi.y - a.y) / d.y;
    lower = max(lower, min(t1, t2));
    upper = min(upper, max(t1, t2));
  }
  return lower < upper && lower < 1.0 && upper > 0.0;
}
`,J=`${q}
${D}
struct CoverFeatureInfo {
  bounds: vec4f,
  vertexCount: u32,
  vertexBegin: u32,
  vertexEnd: u32,
  hasNonFinite: bool,
  hasCrossingEdge: bool
}

fn coverGetFeatureInfo(feature: u32) -> CoverFeatureInfo {
  var info = CoverFeatureInfo(vec4f(1e30, 1e30, -1e30, -1e30), 0u, 0u, 0u, false, false);
  var isFirstRing = true;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      if (isFirstRing) {
        info.vertexBegin = firstVertex;
        isFirstRing = false;
      }
      info.vertexEnd = endVertex;
      if (endVertex > firstVertex) {
        info.vertexCount += endVertex - firstVertex;
      }
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let p = coverGetVertex(vertex);
        let next = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        info.hasCrossingEdge = info.hasCrossingEdge || lonLatIsCrossingEdge(p, next);
        if (coverIsFinite(p)) {
          info.bounds = vec4f(min(info.bounds.xy, p), max(info.bounds.zw, p));
        } else {
          info.hasNonFinite = true;
        }
      }
    }
  }
  return info;
}

${b(`coverGetRingWinding`,`coverGetVertex`)}
// Even-odd containment of the SoS-perturbed point p over every ring of the feature.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      // Rings of two vertices or fewer enclose nothing (the helper returns 0 for them).
      winding += coverGetRingWinding(probe, firstVertex, endVertex).x;
    }
  }
  return (winding & 1) != 0;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  let firstPolygon = featureOffsets[featureOffsetsOffset + feature];
  let endPolygon = featureOffsets[featureOffsetsOffset + feature + 1u];
  for (var polygon = firstPolygon; polygon < endPolygon; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let endRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    for (var ring = firstRing; ring < endRing; ring++) {
      let firstVertex = ringOffsets[ringOffsetsOffset + ring];
      let endVertex = ringOffsets[ringOffsetsOffset + ring + 1u];
      for (var vertex = firstVertex; vertex < endVertex; vertex++) {
        let a = coverGetVertex(vertex);
        let b = coverGetVertex(select(vertex + 1u, firstVertex, vertex + 1u == endVertex));
        if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }
      }
    }
  }
  return false;
}
`;function Y(e){return`${q}${C(e)}
// Even-odd containment of the SoS-perturbed point p. Slabbed and walked features share one edge
// loop, so the exact crossing helper has a single call site.
fn coverContains(feature: u32, p: vec2f) -> bool {
  let probe = ExactProbe(p, p, p, EXACT_PROBE_POINT);
  var winding = 0;
  ${w({owner:`feature`,lowY:`p.y`,highY:`p.y`,getVertex:`coverGetVertex`,body:`winding += exactProbeCrossing(probe, a, b).x;`})}
  return (winding & 1) != 0;
}

fn coverEdgesHitRect(feature: u32, lo: vec2f, hi: vec2f) -> bool {
  ${w({owner:`feature`,lowY:`lo.y`,highY:`hi.y`,getVertex:`coverGetVertex`,body:`if (coverSegmentHitsRect(a, b, lo, hi)) { return true; }`})}
  return false;
}
`}function X(e){return`
const FEATURE_COUNT: u32 = ${e}u;

/** Feature whose candidate range [starts[f], starts[f + 1]) contains \`candidate\`. */
fn coverFindFeature(candidate: u32) -> u32 {
  var low = 0u;
  var high = FEATURE_COUNT;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (starts[startsOffset + middle + 1u] > candidate) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return min(low, FEATURE_COUNT - 1u);
}
`}function Z(e){let t=e.resolution;return`
const TILE_WIDTH: f32 = ${g(45*2**(3-t))};
const ROW_SCALE: f32 = ${g(2**(1-t))};
const PI: f32 = 3.14159265358979;
`}function Q(e,t){let n=t.invalidCount?`if (hasCrossingEdge) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`:``,r=e===`h3`?89:85.051129;return`let hasCrossingEdge = info.hasCrossingEdge;
  ${n}
  ${t.uncertainCount?`if (!hasCrossingEdge && bounds.x <= bounds.z && (bounds.w > ${g(r)} || bounds.y < -${g(r)})) { atomicAdd(&uncertainCount[uncertainCountOffset], 1u); }`:``}`}function oe(e){return[`invalidCount`,`uncertainCount`].flatMap(t=>{let n=e[t];return n?[{name:t,view:n,type:`atomic<u32>`,access:`read_write`}]:[]})}function se(e,t,n,r,a={}){let{resolution:s,candidateCapacity:c,family:l}=t,u=Q(l,a),d=c+1,f=t.scanline?`ranges[rangeBase + 5u] = select(0u, 1u, info.hasNonFinite);`:``,p=`${J}
const CAPACITY_PLUS_ONE: u32 = ${d}u;
const RANGE_STRIDE: u32 = ${G(t)}u;
fn coverClampCount(width: u32, height: u32) -> u32 {
  if (width == 0u || height == 0u) { return 0u; }
  if (width > CAPACITY_PLUS_ONE || height > CAPACITY_PLUS_ONE) { return CAPACITY_PLUS_ONE; }
  if (width > CAPACITY_PLUS_ONE / height) { return CAPACITY_PLUS_ONE; }
  return width * height;
}
`,m;if(l===`quadbin`)p+=`${o}${i}const RESOLUTION: u32 = ${s}u;`,m=`if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${f}
  ${u}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w) {
    // Point indexing wraps +180 to column zero; bounds must retain the easternmost column.
    let lastColumn = (1u << RESOLUTION) - 1u;
    let x0 = select(quadbinGetTileX(bounds.x, RESOLUTION), lastColumn, bounds.x >= 180.0);
    let x1 = select(quadbinGetTileX(bounds.z, RESOLUTION), lastColumn, bounds.z >= 180.0);
    let y0 = quadbinGetTileY(bounds.w, RESOLUTION);
    let y1 = quadbinGetTileY(bounds.y, RESOLUTION);
    if (x1 >= x0 && y1 >= y0) {
      ranges[rangeBase] = x0;
      ranges[rangeBase + 1u] = y0;
      ranges[rangeBase + 2u] = x1 - x0 + 1u;
      ranges[rangeBase + 3u] = y1 - y0 + 1u;
      count = coverClampCount(x1 - x0 + 1u, y1 - y0 + 1u);
    }
  }
  counts[countsOffset + index] = count;`;else{let e=A[s];p+=`const LATTICE_SPACING: f32 = ${g(e)};
const MAXIMUM_LATITUDE: f32 = ${g(89)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn coverToCount(value: f32) -> u32 {
  return u32(clamp(value, 1.0, 4.0e9));
}`,m=`if (index == FEATURE_COUNT) { counts[countsOffset + index] = 0u; return; }
  let info = coverGetFeatureInfo(index);
  let bounds = info.bounds;
  let rangeBase = rangesOffset + index * RANGE_STRIDE;
  var count = 0u;
  ${f}
  let originLat = max(bounds.y - LATTICE_SPACING, -MAXIMUM_LATITUDE);
  let endLat = min(bounds.w + LATTICE_SPACING, MAXIMUM_LATITUDE);
  ${u}
  if (!hasCrossingEdge && bounds.x <= bounds.z && bounds.y <= bounds.w && originLat <= endLat) {
    // Longitude spacing from the most equatorward latitude of the band, where a degree of longitude
    // is longest: a spacing taken from the poleward edge is too coarse for the rows nearer the
    // equator and loses cells there.
    let equatorwardLat = select(min(abs(originLat), abs(endLat)), 0.0, originLat <= 0.0 && endLat >= 0.0);
    let spacingLng = LATTICE_SPACING / cos(equatorwardLat * DEGREES_TO_RADIANS);
    let originLng = bounds.x - spacingLng;
    let width = coverToCount(ceil((bounds.z + spacingLng - originLng) / spacingLng) + 1.0);
    let height = coverToCount(ceil((endLat - originLat) / LATTICE_SPACING) + 1.0);
    ranges[rangeBase] = width;
    ranges[rangeBase + 1u] = height;
    ranges[rangeBase + 2u] = bitcast<u32>(originLng);
    ranges[rangeBase + 3u] = bitcast<u32>(originLat);
    ranges[rangeBase + 4u] = bitcast<u32>(spacingLng);
    count = coverClampCount(width, height);
  }
  counts[countsOffset + index] = count;`}return p+=`const FEATURE_COUNT: u32 = ${t.featureCount}u;`,v(e,{id:`${t.id}-count`,operation:V,variant:`count`,bindings:[...K(t),{name:`counts`,view:n,type:`u32`,access:`read_write`},{name:`ranges`,view:r,type:`u32`,access:`read_write`},...oe(a)],invocationCount:t.featureCount+1,declarations:p,body:m})}function ce(e,t,n){let{family:r,resolution:c,candidateCapacity:l,edgeSlabs:u}=t,d=H[t.containment],f=t.scanline?B(t.featureCount,`width`):`let column = candidateLocal % width;
  let row = candidateLocal / width;`,p=u?E(u):[],m=`${u?Y(u):J}${X(t.featureCount)}
const CANDIDATE_CAPACITY: u32 = ${l}u;
const RANGE_STRIDE: u32 = ${G(t)}u;
const MODE: u32 = ${d}u;
const RESOLUTION: u32 = ${c}u;
const CORE_BIT: u32 = ${U}u;
const CORE_MARGIN: f32 = ${g(W)};
`,h;if(r===`quadbin`){let e=s(`quadbin`,c);m+=`${Z(t)}${o}${i}
fn coverTileLatitude(row: f32) -> f32 {
  let t = PI * (1.0 - row * ROW_SCALE);
  return (2.0 * atan(exp(t)) - PI * 0.5) * 57.29577951308232;
}
`,h=`let width = ranges[rangeBase + 2u];
  ${f}
  let x = ranges[rangeBase] + column;
  let y = ranges[rangeBase + 1u] + row;
  let lng0 = f32(x) * TILE_WIDTH - 180.0;
  let lng1 = f32(x + 1u) * TILE_WIDTH - 180.0;
  let latNorth = coverTileLatitude(f32(y));
  let latSouth = coverTileLatitude(f32(y + 1u));
  let centre = vec2f((f32(x) + 0.5) * TILE_WIDTH - 180.0, coverTileLatitude(f32(y) + 0.5));
  // One call site for the exact containment test (see COMMON_WGSL).
  let centreInside = coverContains(feature, centre);
  var accepted = false;
  if (MODE == 0u) {
    accepted = centreInside;
  } else {
    let hit = coverEdgesHitRect(feature, vec2f(lng0, latSouth), vec2f(lng1, latNorth));
    if (MODE == 1u) {
      accepted = !hit && centreInside;
    } else {
      accepted = hit || centreInside;
    }
  }
  var key = vec2u(0u);
  if (accepted) {
    key = cellGetKey(quadbinGetCompactKey(x, y), ${e.headerHigh}u, RESOLUTION, ${e.lowBit}u);
    ${t.computeCore?`// Core: the centre is inside and no polygon edge enters the cell grown by a safety margin.
    let slack = vec2f(CORE_MARGIN + 0.001 * (lng1 - lng0));
    if (centreInside && !coverEdgesHitRect(feature, vec2f(lng0, latSouth) - slack, vec2f(lng1, latNorth) + slack)) {
      key.x = key.x | CORE_BIT;
    }`:``}
  }`}else m+=`${a.source}\n${k}\n${t.computeCore?O:``}`,h=`let width = ranges[rangeBase];
  let originLng = bitcast<f32>(ranges[rangeBase + 2u]);
  let originLat = bitcast<f32>(ranges[rangeBase + 3u]);
  let spacingLng = bitcast<f32>(ranges[rangeBase + 4u]);
  let spacingLat = ${g(A[c])};
  ${f}
  let point = vec2f(originLng + f32(column) * spacingLng, originLat + f32(row) * spacingLat);
  var key = vec2u(0u);
  let cell = cellIndexH3FromLngLat(point, RESOLUTION);
  if (cell.x != 0u || cell.y != 0u) {
    let centre = dggs_h3_get_center_lnglat(cell);
    // Shift the center by whole turns next to the lattice, then round it to the nearest lattice
    // index with one floating point expression of the center alone, so exactly one lattice point
    // claims each cell (no tie ambiguity between neighbouring points).
    let middleLng = originLng + 0.5 * f32(width) * spacingLng;
    let centreLng = centre.x + 360.0 * round((middleLng - centre.x) / 360.0);
    let nearestColumn = floor((centreLng - originLng) / spacingLng + 0.5);
    let nearestRow = floor((centre.y - originLat) / spacingLat + 0.5);
    if (nearestColumn == f32(column) && nearestRow == f32(row) && coverContains(feature, centre)) {
      key = cell;
      ${t.computeCore?`// Core: no polygon edge enters the bounding box of the cell boundary, grown by a margin.
      let boundary = cellIndexH3GetBoundary(cell);
      var lo = boundary.points[0];
      var hi = boundary.points[0];
      for (var vertex = 1u; vertex < boundary.count; vertex++) {
        lo = min(lo, boundary.points[vertex]);
        hi = max(hi, boundary.points[vertex]);
      }
      let slack = vec2f(CORE_MARGIN) + 0.001 * (hi - lo);
      if (boundary.count > 0u && hi.x - lo.x < 90.0 && !coverEdgesHitRect(feature, lo - slack, hi + slack)) {
        key.x = key.x | CORE_BIT;
      }`:``}
    }
  }
  let accepted = key.x != 0u || key.y != 0u;`;return v(e,{id:`${t.id}-test`,operation:V,variant:u?`test-slabs`:`test`,bindings:[...u?[{name:`positions`,view:t.positions,type:`f32`,access:`read`}]:K(t),{name:`ranges`,view:n.ranges,type:`u32`,access:`read`},{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`flags`,view:n.flags,type:`u32`,access:`read_write`},{name:`candidateCells`,view:n.candidateCells,type:`u32`,access:`read_write`},...p],invocationCount:l,declarations:m,body:`let activeCount = min(starts[startsOffset + FEATURE_COUNT], CANDIDATE_CAPACITY);
  if (index >= activeCount) {
    flags[flagsOffset + index] = 0u;
    return;
  }
  let feature = coverFindFeature(index);
  let candidateLocal = index - starts[startsOffset + feature];
  let rangeBase = rangesOffset + feature * RANGE_STRIDE;
  ${h}
  flags[flagsOffset + index] = select(0u, 1u, accepted);
  candidateCells[candidateCellsOffset + 2u * index] = key.y;
  candidateCells[candidateCellsOffset + 2u * index + 1u] = key.x;`})}function le(e,t,n){let r=[{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`accepted`,view:n.accepted,type:`u32`,access:`read`},{name:`candidateCells`,view:n.candidateCells,type:`u32`,access:`read`}];return n.featureIds&&r.push({name:`featureIds`,view:n.featureIds,type:`u32`,access:`read`}),r.push({name:`outputFeatureIds`,view:n.outputFeatureIds,type:`u32`,access:`read_write`},{name:`outputCells`,view:n.outputCells,type:`u32`,access:`read_write`}),n.outputCore&&r.push({name:`outputCore`,view:n.outputCore,type:`u32`,access:`read_write`}),v(e,{id:`${t.id}-write`,operation:V,variant:`write`,bindings:r,invocationCount:t.candidateCapacity,declarations:`${X(t.featureCount)}
const OUTPUT_CAPACITY: u32 = ${n.outputFeatureIds.length}u;
const CORE_BIT: u32 = ${U}u;`,body:`var beforeCount = 0u;
  if (index > 0u) { beforeCount = accepted[acceptedOffset + index - 1u]; }
  let afterCount = accepted[acceptedOffset + index];
  if (afterCount == beforeCount) { return; }
  let slot = afterCount - 1u;
  if (slot >= OUTPUT_CAPACITY) { return; }
  let feature = coverFindFeature(index);
  outputFeatureIds[outputFeatureIdsOffset + slot] = ${n.featureIds?`featureIds[featureIdsOffset + feature]`:`feature`};
  outputCells[outputCellsOffset + 2u * slot] = candidateCells[candidateCellsOffset + 2u * index];
  let highWord = candidateCells[candidateCellsOffset + 2u * index + 1u];
  outputCells[outputCellsOffset + 2u * slot + 1u] = highWord & ~CORE_BIT;
  ${n.outputCore?`outputCore[outputCoreOffset + slot] = select(0u, 1u, (highWord & CORE_BIT) != 0u);`:``}`})}function ue(e,t,n){return v(e,{id:`${t.id}-finalize`,operation:V,variant:`finalize`,bindings:[{name:`starts`,view:n.starts,type:`u32`,access:`read`},{name:`accepted`,view:n.accepted,type:`u32`,access:`read`},{name:`total`,view:n.total,type:`u32`,access:`read_write`},{name:`candidateOverflow`,view:n.candidateOverflow,type:`u32`,access:`read_write`},...n.candidateCount?[{name:`candidateCount`,view:n.candidateCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:`const FEATURE_COUNT: u32 = ${t.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${t.candidateCapacity}u;`,body:`total[totalOffset] = accepted[acceptedOffset + CANDIDATE_CAPACITY - 1u];
  candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, starts[startsOffset + FEATURE_COUNT] > CANDIDATE_CAPACITY);
  ${n.candidateCount?`candidateCount[candidateCountOffset] = starts[startsOffset + FEATURE_COUNT];`:``}`})}function de(e){return d(e,8)}function fe(e){return d(e,8)}var pe=`cell-cover`,me=4e6,he=class{id;props;spatialContext;featureCount;containment;edgeSlabs;candidateEnumeration;scanline;scanlineSlotCapacity;constructor(i){this.id=i.id??pe,this.props=i;let{id:a}=this;if(this.spatialContext=e(a,i.spatialContext??t,{...n,edgeModels:r}),i.family!==`quadbin`&&i.family!==`h3`)throw Error(`${a} family must be 'quadbin' or 'h3'`);if(s(i.family,i.resolution),this.containment=i.containment??`center`,![`center`,`full`,`intersects`].includes(this.containment))throw Error(`${a} containment must be 'center', 'full' or 'intersects'`);if(i.family===`h3`&&this.containment!==`center`)throw Error(`${a} h3 supports only containment 'center'`);m(i.positions,[`float32x2`],`${a} positions`);for(let[e,t]of[[`featureOffsets`,i.featureOffsets],[`polygonOffsets`,i.polygonOffsets],[`ringOffsets`,i.ringOffsets]])if(m(t,[`uint32`],`${a} ${e}`),t.length<1)throw Error(`${a} ${e} requires a terminal entry`);if(this.featureCount=i.featureOffsets.length-1,this.edgeSlabs=(i.edgeSlabs??i.candidateCapacity*i.positions.length>=me)&&i.ringOffsets.length>=2&&i.polygonOffsets.length>=2,i.edgeSlabEntryCapacity!==void 0&&l(a,`edgeSlabEntryCapacity`,i.edgeSlabEntryCapacity),i.featureIds&&(m(i.featureIds,[`uint32`],`${a} featureIds`),i.featureIds.length!==this.featureCount))throw Error(`${a} featureIds length must equal the feature count`);if(l(a,`candidateCapacity`,i.candidateCapacity),(this.featureCount+1)*(i.candidateCapacity+1)>=4294967295)throw Error(`${a} (featureCount + 1) * (candidateCapacity + 1) must be below 2^32`);if(this.candidateEnumeration=i.candidateEnumeration??`auto`,![`bounds`,`scanline`,`auto`].includes(this.candidateEnumeration))throw Error(`${a} candidateEnumeration must be 'bounds', 'scanline' or 'auto'`);this.scanline=(this.candidateEnumeration===`scanline`||this.candidateEnumeration===`auto`&&i.candidateCapacity>=65536)&&this.featureCount>=1&&i.positions.length>=1&&i.ringOffsets.length>=2&&i.polygonOffsets.length>=2,this.scanlineSlotCapacity=i.scanlineSlotCapacity??Math.min(i.candidateCapacity,j,Math.max(16*i.positions.length,16384)),(i.scanlineSlotCapacity!==void 0||this.scanline)&&l(a,`scanlineSlotCapacity`,this.scanlineSlotCapacity,1,Math.min(i.candidateCapacity,j));let{output:o}=i,{status:c}=o;if(m(o.featureIds,[`uint32`],`${a} output.featureIds`),m(o.cells,[`uint32x2`],`${a} output.cells`),o.cells.length!==o.featureIds.length)throw Error(`${a} output.cells and output.featureIds must have equal lengths`);if(o.featureIds.length<1)throw Error(`${a} output capacity must be at least 1`);if(o.core&&(m(o.core,[`uint32`],`${a} output.core`),o.core.length!==o.featureIds.length))throw Error(`${a} output.core must have the same length as output.featureIds`);for(let[e,t]of[[`status.count`,c.count],[`status.requiredCount`,c.requiredCount],[`status.overflow`,c.overflow],[`status.candidateOverflow`,c.candidateOverflow],[`status.candidateCount`,c.candidateCount],[`status.invalidCount`,c.invalidCount],[`status.uncertainCount`,c.uncertainCount]])if(t&&(m(t,[`uint32`],`${a} output.${e}`),t.length<1))throw Error(`${a} output.${e} must contain one uint32 row`);u(a,[o.featureIds,o.cells,o.core,c.count,c.requiredCount,c.overflow,c.candidateOverflow,c.candidateCount,c.invalidCount,c.uncertainCount],[i.positions,i.featureOffsets,i.polygonOffsets,i.ringOffsets,i.featureIds])}getCommandNodes(e){let{id:t,props:n,featureCount:r}=this,{output:i,candidateCapacity:a}=n;c(t,e,[n.positions,n.featureOffsets,n.polygonOffsets,n.ringOffsets,n.featureIds,i.featureIds,i.cells,i.core,i.status.count,i.status.requiredCount,i.status.overflow,i.status.candidateOverflow,i.status.candidateCount,i.status.invalidCount,i.status.uncertainCount]),l(t,`candidateCapacity`,a,1,de(e.device));let o=fe(e.device);this.edgeSlabs&&f(t,`positions rows + edgeSlabEntryCapacity`,n.positions.length+(n.edgeSlabEntryCapacity??1),o);let s=p(e,`${t}-counts`,`uint32`,r+1),u=p(e,`${t}-starts`,`uint32`,r+1),m=this.scanline?{slotCapacity:this.scanlineSlotCapacity,force:this.candidateEnumeration===`scanline`}:void 0,g=I(r,m,5);f(t,`range table words`,g,d(e.device,4));let v=p(e,`${t}-ranges`,`uint32`,g),y=p(e,`${t}-flags`,`uint32`,a),b=p(e,`${t}-accepted`,`uint32`,a),x=p(e,`${t}-candidate-cells`,`uint32x2`,a),S=p(e,`${t}-total`,`uint32`,1),C=this.edgeSlabs?T(e,{id:t,operation:`GPUCellCover`,owner:`feature`,positions:n.positions,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,ringOffsets:n.ringOffsets,entryCapacity:n.edgeSlabEntryCapacity??Math.max(1,Math.min(Math.max(4*n.positions.length,1024),o-n.positions.length)),gate:{counts:s,minimum:16}}):void 0,w={id:t,family:n.family,resolution:n.resolution,containment:this.containment,computeCore:!!i.core,featureCount:r,candidateCapacity:a,edgeSlabs:C?.slabs,scanline:m,positions:n.positions,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,ringOffsets:n.ringOffsets},E=[`invalidCount`,`uncertainCount`].flatMap(n=>{let r=i.status[n];return r?[ee(e,{id:`${t}-clear-${n}`,operation:`GPUCellCover`,view:r,type:`u32`,value:`0u`})]:[]}),te={invalidCount:i.status.invalidCount,uncertainCount:i.status.uncertainCount};return[...E,se(e,w,s,v,te),...m?z(e,{...w,scanline:m},{counts:s,ranges:v}):[],...new h({id:`${t}-candidate-scan`,input:s,output:u,mode:`exclusive`}).getCommandNodes(e),...C?.nodes??[],ce(e,w,{ranges:v,starts:u,flags:y,candidateCells:x}),...new h({id:`${t}-accept-scan`,input:y,output:b,mode:`inclusive`}).getCommandNodes(e),le(e,w,{starts:u,accepted:b,candidateCells:x,featureIds:n.featureIds,outputFeatureIds:i.featureIds,outputCells:i.cells,outputCore:i.core}),ue(e,w,{starts:u,accepted:b,total:S,candidateOverflow:i.status.candidateOverflow,candidateCount:i.status.candidateCount}),_(e,{id:`${t}-publish`,operation:`GPUCellCover`,requiredCount:S,output:{ids:i.featureIds,status:{count:i.status.count,overflow:i.status.overflow,requiredCount:i.status.requiredCount}},overflowSources:[]})]}},ge=`GPUCellGeometry`,$={quadbin:4,h3:6,quadkey:4,geohash:4,s2:4,a5:5},_e=16,ve=class{id;props;maximumVertexCount;edgeModel;outputSpatialContext;constructor(e){this.id=e.id??`cell-geometry`,this.props=e;let n=this.id;if(!(e.family in $))throw Error(`${n} family must be one of ${Object.keys($).join(`, `)}`);if(this.edgeModel=e.family===`h3`||e.family===`s2`||e.family===`a5`?`great-circle`:`linear`,this.outputSpatialContext={...t,edgeModel:this.edgeModel},e.wordOrder&&e.wordOrder!==`little-endian`&&e.wordOrder!==`high-low`)throw Error(`${n} wordOrder must be 'little-endian' or 'high-low'`);let r=$[e.family];if(this.maximumVertexCount=e.maximumVertexCount??r,!Number.isInteger(this.maximumVertexCount)||this.maximumVertexCount<r||this.maximumVertexCount>_e)throw Error(`${n} maximumVertexCount must be an integer in [${r}, ${_e}] for ${e.family}`);let{output:i}=e;m(e.cells,[`uint32x2`],`${n} cells`);let a=e.cells.length;if(a<1)throw Error(`${n} needs at least one row`);if(i.vertexCounts&&!i.boundaries)throw Error(`${n} output.vertexCounts needs output.boundaries`);if(!i.centers&&!i.boundaries)throw Error(`${n} needs output.centers or output.boundaries`);if(i.centers&&(m(i.centers,[`float32x2`],`${n} output.centers`),i.centers.length!==a))throw Error(`${n} output.centers length must equal the row count`);if(i.boundaries&&(m(i.boundaries,[`float32x2`],`${n} output.boundaries`),i.boundaries.length!==a*this.maximumVertexCount))throw Error(`${n} output.boundaries length must equal rows * maximumVertexCount (${a*this.maximumVertexCount})`);if(i.vertexCounts&&(m(i.vertexCounts,[`uint32`],`${n} output.vertexCounts`),i.vertexCounts.length!==a))throw Error(`${n} output.vertexCounts length must equal the row count`);u(n,[i.centers,i.boundaries,i.vertexCounts],[e.cells])}getCommandNodes(e){let{id:t,props:n,maximumVertexCount:r}=this,{family:i,output:s}=n;c(t,e,[n.cells,s.centers,s.boundaries,s.vertexCounts]);let l=[{name:`cells`,view:n.cells,type:`u32`,access:`read`}];s.centers&&l.push({name:`centersOut`,view:s.centers,type:`f32`,access:`read_write`}),s.boundaries&&l.push({name:`boundariesOut`,view:s.boundaries,type:`f32`,access:`read_write`}),s.vertexCounts&&l.push({name:`vertexCountsOut`,view:s.vertexCounts,type:`u32`,access:`read_write`});let u=s.centers?`let center = select(vec2f(getNotANumber(index)), geometryGetCenter(key), isValid);
  centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`:``,d=s.boundaries?`var points: array<vec2f, 16>;
  var vertexCount = 0u;
  if (isValid) {
    vertexCount = geometryFillBoundary(key, &points);
  }
  vertexCount = min(vertexCount, MAXIMUM_VERTEX_COUNT);
  for (var vertex = 0u; vertex < MAXIMUM_VERTEX_COUNT; vertex++) {
    let point = points[vertex];
    let slot = (index * MAXIMUM_VERTEX_COUNT + vertex) * 2u;
    boundariesOut[boundariesOutOffset + slot] = point.x;
    boundariesOut[boundariesOutOffset + slot + 1u] = point.y;
  }
  ${s.vertexCounts?`vertexCountsOut[vertexCountsOutOffset + index] = vertexCount;`:``}`:``;return[v(e,{id:`${t}-geometry`,operation:ge,variant:i,bindings:l,invocationCount:n.cells.length,declarations:`const MAXIMUM_VERTEX_COUNT: u32 = ${r}u;
// WGSL rejects constant-expression NaN, so the payload depends on a runtime value.
fn getNotANumber(runtimeZero: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (runtimeZero & 0u));
}
${a.source}
${o}
${ye}
${Ee[i]}
${i===`h3`?``:Te}`,body:`let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${n.wordOrder===`high-low`?`words`:`words.yx`};
  let isValid = geometryIsValid(key);
  ${u}
  ${d}`})]}},ye=`
fn geometryGetTileBounds(tile: vec3u) -> vec4f {
  let tileScale = f32(1u << tile.z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = dggs_web_mercator_tile_y_to_latitude(f32(tile.y), tileScale);
  let south = dggs_web_mercator_tile_y_to_latitude(f32(tile.y + 1u), tileScale);
  return vec4f(west, south, east, north);
}

fn geometryGetTileCenter(tile: vec3u) -> vec2f {
  let tileScale = f32(1u << tile.z);
  return vec2f(
    (f32(tile.x) + 0.5) / tileScale * 360.0 - 180.0,
    dggs_web_mercator_tile_y_to_latitude(f32(tile.y) + 0.5, tileScale)
  );
}

fn geometryGetRectangleVertex(bounds: vec4f, vertex: u32) -> vec2f {
  return dggs_bounds_get_boundary_point(bounds, vertex);
}

/** Even bits of a 32-bit value packed into 16 bits (inverse of the Morton spread). */
fn geometryCompactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
`,be=`
fn geometryGetQuadbinTile(key: vec2u) -> vec3u {
  let resolution = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * resolution, 2u * resolution);
  let x = geometryCompactEvenBits(morton.y) | (geometryCompactEvenBits(morton.x) << 16u);
  let y = geometryCompactEvenBits(morton.y >> 1u) | (geometryCompactEvenBits(morton.x >> 1u) << 16u);
  return vec3u(x, y, resolution);
}
fn geometryIsValid(key: vec2u) -> bool { return cellIsValidQuadbin(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(geometryGetQuadbinTile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(geometryGetTileBounds(geometryGetQuadbinTile(key)), vertex);
}
`,xe=`
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 26u;
  // Digits fill the low 2 * length bits; every bit between them and the length field is zero.
  let unused = cellMaskLow(58u) & ~cellMaskLow(2u * length);
  return length >= 1u && length <= DGGS_QUADKEY_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(dggs_quadkey_get_tile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(dggs_quadkey_get_bounds(key), vertex);
}
`,Se=`
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 28u;
  let unused = cellMaskLow(60u) & ~cellMaskLow(5u * length);
  return length >= 1u && length <= DGGS_GEOHASH_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f {
  let bounds = dggs_geohash_get_bounds(key);
  return vec2f(0.5 * (bounds.x + bounds.z), 0.5 * (bounds.y + bounds.w));
}
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_geohash_get_boundary_point(key, vertex);
}
`,Ce=`
// dggs_s2_xyz_to_lnglat calls atan2 with a zero argument at the S2 face centers: atan2(0, 0) at the
// poles (faces 2 and 5) is undefined in WGSL, and on Metal atan2(1, 0) returned -pi/2 for the face
// 1 center (x = -u = -0). Exact zeros are therefore handled without atan2. Upstream
// dggs_s2_xyz_to_lnglat (a plain atan2) is not used: it changes the values on the axes.
fn geometryS2XyzToLngLat(xyz: vec3f) -> vec2f {
  let horizontal = length(xyz.xy);
  var longitude = 0.0;
  if (xyz.x == 0.0) {
    longitude = select(0.0, select(-90.0, 90.0, xyz.y > 0.0), xyz.y != 0.0);
  } else if (xyz.y == 0.0) {
    longitude = select(0.0, 180.0, xyz.x < 0.0);
  } else {
    longitude = atan2(xyz.y, xyz.x) * DGGS_RADIANS_TO_DEGREES;
  }
  return vec2f(longitude, atan2(xyz.z, horizontal) * DGGS_RADIANS_TO_DEGREES);
}
fn geometryS2GetPoint(key: vec2u, offset: vec2f) -> vec2f {
  let st = dggs_s2_ij_to_st(dggs_s2_get_ij(key), dggs_s2_get_level(key), offset);
  return geometryS2XyzToLngLat(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(key), dggs_s2_st_to_uv(st)));
}
fn geometryIsValid(key: vec2u) -> bool { return dggs_s2_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryS2GetPoint(key, vec2f(0.5)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryS2GetPoint(key, dggs_s2_get_boundary_offset(vertex));
}
`,we=`
${O}
fn geometryIsValid(key: vec2u) -> bool { return dggs_h3_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_h3_get_center_lnglat(key); }
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let boundary = cellIndexH3GetBoundary(key);
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    (*points)[vertex] = boundary.points[vertex];
  }
  return boundary.count;
}
`,Te=`
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let count = min(geometryGetVertexCount(key), 16u);
  for (var vertex = 0u; vertex < count; vertex++) {
    (*points)[vertex] = geometryGetVertex(key, vertex);
  }
  return count;
}
`,Ee={quadbin:be,quadkey:xe,geohash:Se,s2:Ce,h3:we,a5:`
fn geometryIsValid(key: vec2u) -> bool {
  return !dggs_u64_is_zero(key) && dggs_a5_deserialize(key).valid != 0u;
}
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_a5_get_center_lnglat(key); }
fn geometryGetVertexCount(key: vec2u) -> u32 {
  return dggs_a5_get_shape_vertex_count(dggs_a5_deserialize(key));
}
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_a5_get_boundary_point(key, vertex);
}
`};export{he as n,ve as t};