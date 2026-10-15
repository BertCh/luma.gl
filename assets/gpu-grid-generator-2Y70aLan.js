import{E as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,d as n,h as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as o}from"./exact-orientation-wgsl-D-n1xGZJ.js";var s=`GPUGridGenerator`;function c(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Grid generator target must hold 4 elements`);let{minX:n,minY:r,cellWidth:i,cellHeight:a}=e;for(let e of[n,r,i,a])if(!Number.isFinite(e))throw Error(`Grid generator settings must be finite`);if(i<=0||a<=0)throw Error(`Grid generator cellWidth and cellHeight must be positive`);return t.set([n,r,i,a]),t}function l(e){switch(e){case`square`:return 4;case`hex`:return 6;case`triangle`:return 3;case`point`:return 0;default:throw Error(`Unknown grid type ${String(e)}`)}}function u(e,t,n){return t*n*(e===`triangle`?2:1)}function d(e,t,n){return e===`square`?(t+1)*(n+1):2*(t*n+t+n)}var f=class{id;props;spatialContext;cellCount;verticesPerCell;cornerCount;constructor(t){this.id=t.id??`grid-generator`,this.props=t;let{id:a}=this;if(this.spatialContext=e(a,t.spatialContext),this.verticesPerCell=l(t.gridType),n(a,`columns`,t.columns,1,2**53-1),n(a,`rows`,t.rows,1,2**53-1),this.cellCount=u(t.gridType,t.columns,t.rows),this.cornerCount=t.gridType===`square`||t.gridType===`hex`?d(t.gridType,t.columns,t.rows):0,this.cellCount*Math.max(this.verticesPerCell,1)>2147483647)throw Error(`${a} grid is too large`);if(i(t.parameters,[`float32`],`${a} parameters`),t.parameters.length<4)throw Error(`${a} parameters must hold 4 float32 values`);let{positions:o,centers:s,corners:c,cornerIntersects:f}=t.output;if(c&&(i(c,[`float32x2`],`${a} output.corners`),!this.cornerCount||this.cornerCount>2147483647||c.length!==this.cornerCount))throw Error(`${a} output.corners needs a correctly sized square or hex grid`);if(f&&(i(f,[`uint32`],`${a} output.cornerIntersects`),!c||f.length!==this.cornerCount))throw Error(`${a} output.cornerIntersects needs one row per output corner`);if(t.gridType===`point`){if(!s)throw Error(`${a} point grids need output.centers`);if(o)throw Error(`${a} point grids have no output.positions`)}else if(!o&&!c)throw Error(`${a} ${t.gridType} grids need output.positions`);if(o&&(i(o,[`float32x2`],`${a} output.positions`),o.length!==this.cellCount*this.verticesPerCell))throw Error(`${a} output.positions must hold ${this.cellCount*this.verticesPerCell} rows`);if(s&&(i(s,[`float32x2`],`${a} output.centers`),s.length!==this.cellCount))throw Error(`${a} output.centers must hold ${this.cellCount} rows`);let{intersects:p}=t.output;if(t.hexOrientation!==void 0&&t.hexOrientation!==`pointy`&&t.hexOrientation!==`flat`)throw Error(`${a} hexOrientation must be 'pointy' or 'flat'`);if(t.hexOrientation===`flat`&&t.gridType!==`hex`)throw Error(`${a} hexOrientation 'flat' needs gridType 'hex'`);let{extent:m}=t;if(!!(p||f)!=!!m)throw Error(`${a} extent needs an intersection output and vice versa`);if(m){if(p&&i(p,[`uint32`],`${a} output.intersects`),i(m.positions,[`float32x2`],`${a} extent.positions`),i(m.featureOffsets,[`uint32`],`${a} extent.featureOffsets`),i(m.polygonOffsets,[`uint32`],`${a} extent.polygonOffsets`),i(m.ringOffsets,[`uint32`],`${a} extent.ringOffsets`),p&&p.length!==this.cellCount)throw Error(`${a} output.intersects must hold ${this.cellCount} rows`);if(m.ringOffsets.length<2)throw Error(`${a} extent.ringOffsets needs at least one ring`);if(p&&(t.gridType===`point`?!s:!o))throw Error(`${a} output.intersects needs the cell geometry output`)}r(a,[o,s,c,p,f],[t.parameters,m?.positions,m?.featureOffsets,m?.polygonOffsets,m?.ringOffsets])}getCommandNodes(e){let{props:n,id:r}=this,{positions:i,centers:c,corners:l,cornerIntersects:u}=n.output,{intersects:d}=n.output,{extent:f}=n;t(r,e,[n.parameters,i,c,l,u,d,f?.positions,f?.featureOffsets,f?.polygonOffsets,f?.ringOffsets]);let _=n.gridType===`hex`&&n.hexOrientation===`flat`,v=[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`}];i&&v.push({name:`positionsOut`,view:i,type:`f32`,access:`read_write`}),c&&v.push({name:`centersOut`,view:c,type:`f32`,access:`read_write`});let y=[];(i||c)&&y.push(a(e,{id:`${r}-generate`,operation:s,variant:_?`hex-flat`:n.gridType,bindings:v,invocationCount:this.cellCount,declarations:`const COLUMNS: u32 = ${n.columns}u;
const SQRT3: f32 = 1.7320508;
const CELLS_PER_ROW: u32 = ${n.gridType===`triangle`?2*n.columns:n.columns}u;
${n.gridType===`hex`&&i?m(_):``}

${i?`fn writeVertex(cell: u32, slot: u32, p: vec2<f32>) {
  let base = positionsOutOffset + 2u * (cell * `+this.verticesPerCell+`u + slot);
  positionsOut[base] = p.x;
  positionsOut[base + 1u] = p.y;
}`:``}`,body:`let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let w = parameters[parametersOffset + 2u];
  let h = parameters[parametersOffset + 3u];
  let row = index / CELLS_PER_ROW;
  let column = index % CELLS_PER_ROW;
  let shift = select(0.0, 0.5 * w, (row & 1u) == 1u);
  var center = vec2<f32>(0.0);
${g(n.gridType,!!i,_)}
  ${c?`centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`:``}`})),l&&y.push(a(e,{id:`${r}-corners`,operation:s,variant:_?`corners-hex-flat`:`corners-${n.gridType}`,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`cornersOut`,view:l,type:`f32`,access:`read_write`}],invocationCount:this.cornerCount,declarations:`const COLUMNS: u32 = ${_?n.rows:n.columns}u;
${n.gridType===`square`?``:m(_)}`,body:`let origin = vec2f(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let width = parameters[parametersOffset + 2u];
  let height = parameters[parametersOffset + 3u];
  ${p(n.gridType===`square`,_)}
  cornersOut[cornersOutOffset + 2u * index] = corner.x;
  cornersOut[cornersOutOffset + 2u * index + 1u] = corner.y;`}));for(let[t,p,m,g,_]of[[d,n.gridType===`point`?c:i,this.cellCount,n.gridType===`point`,`intersects`],[u,l,this.cornerCount,!0,`corner-intersects`]])!t||!f||y.push(a(e,{id:`${r}-${_}`,operation:s,variant:`intersects`,bindings:[{name:`cellVertices`,view:p,type:`f32`,access:`read`},{name:`extentPositions`,view:f.positions,type:`f32`,access:`read`},{name:`extentRingOffsets`,view:f.ringOffsets,type:`u32`,access:`read`},{name:`intersectsOut`,view:t,type:`u32`,access:`read_write`}],invocationCount:m,declarations:`const VERTICES_PER_CELL: u32 = ${g?1:this.verticesPerCell}u;
const IS_POINT: bool = ${g};
const RING_COUNT: u32 = ${f.ringOffsets.length-1}u;
const EXTENT_VERTEX_COUNT: u32 = ${f.positions.length}u;
${o}
${h}`,body:`intersectsOut[intersectsOutOffset + index] = select(0u, 1u, cellIntersectsExtent(index));`}));return y}};function p(e,t){return e?`let corner = origin + vec2f(f32(index % (COLUMNS + 1u)) * width,
    f32(index / (COLUMNS + 1u)) * height);`:`var row = 0u;
  var column = 0u;
  var slot = index;
  let firstRowCount = 4u * COLUMNS + 2u;
  if (index < firstRowCount) {
    if (index >= 6u) {
      column = 1u + (index - 6u) / 4u;
      slot = (index - 6u) % 4u;
      slot = select(slot, slot + 2u, slot >= 2u);
    }
  } else {
    let rowCount = 2u * COLUMNS + 2u;
    row = 1u + (index - firstRowCount) / rowCount;
    let local = (index - firstRowCount) % rowCount;
    let firstCellCount = select(4u, 3u, (row & 1u) == 1u);
    slot = local;
    if (local >= firstCellCount) {
      column = 1u + (local - firstCellCount) / 2u;
      slot = (local - firstCellCount) % 2u;
    }
    // The right boundary of an odd row adds the bottom-right vertex of its final cell.
    if ((row & 1u) == 1u && local == rowCount - 1u) {
      column = COLUMNS - 1u;
      slot = 5u;
    }
  }
  let corner = hexLatticeVertex(origin, width, column, row, slot);`}function m(e){return`fn hexLatticeVertex(origin: vec2f, width: f32, latticeColumn: u32, latticeRow: u32, slot: u32) -> vec2f {
  let horizontal = array<i32, 6>(1, 0, -1, -1, 0, 1);
  let vertical = array<i32, 6>(1, 2, 1, -1, -2, -1);
  let lattice = vec2f(f32(2u * latticeColumn + (latticeRow & 1u)) + f32(horizontal[slot]) + 1.0,
    f32(3u * latticeRow) + f32(vertical[slot]) + 2.0);
  let displacement = lattice * vec2f(0.5 * width, 0.5 * width / 1.7320508);
  return origin + displacement${e?`.yx`:``};
}`}var h=`
fn cellVertex(cell: u32, slot: u32) -> vec2f {
  let base = cellVerticesOffset + 2u * (cell * VERTICES_PER_CELL + slot);
  return vec2f(cellVertices[base], cellVertices[base + 1u]);
}
fn extentVertex(vertex: u32) -> vec2f {
  let base = extentPositionsOffset + 2u * vertex;
  return vec2f(extentPositions[base], extentPositions[base + 1u]);
}
fn ringStart(ring: u32) -> u32 { return extentRingOffsets[extentRingOffsetsOffset + ring]; }
fn ringEnd(ring: u32) -> u32 {
  return min(extentRingOffsets[extentRingOffsetsOffset + ring + 1u], EXTENT_VERTEX_COUNT);
}
// Second vertex of the edge leaving 'vertex' in its ring (the ring closes implicitly).
fn nextVertex(vertex: u32, first: u32, last: u32) -> u32 {
  return select(vertex + 1u, first, vertex + 1u >= last);
}
fn insideBox(a: vec2f, b: vec2f, c: vec2f) -> bool {
  return c.x >= min(a.x, b.x) && c.x <= max(a.x, b.x) && c.y >= min(a.y, b.y) && c.y <= max(a.y, b.y);
}
// True when the closed segments p1p2 and q1q2 share a point.
fn segmentsTouch(p1: vec2f, p2: vec2f, q1: vec2f, q2: vec2f) -> bool {
  let o1 = orientSign(p1, p2, q1);
  let o2 = orientSign(p1, p2, q2);
  let o3 = orientSign(q1, q2, p1);
  let o4 = orientSign(q1, q2, p2);
  if (o1 == 2 || o2 == 2 || o3 == 2 || o4 == 2) { return false; }
  if (o1 != o2 && o3 != o4) { return true; }
  if (o1 == 0 && insideBox(p1, p2, q1)) { return true; }
  if (o2 == 0 && insideBox(p1, p2, q2)) { return true; }
  if (o3 == 0 && insideBox(q1, q2, p1)) { return true; }
  if (o4 == 0 && insideBox(q1, q2, p2)) { return true; }
  return false;
}
// Even-odd containment of 'point' in the extent rings; boundary counts as inside.
fn extentContains(point: vec2f) -> bool {
  var isInside = false;
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let a = extentVertex(vertex);
      let b = extentVertex(nextVertex(vertex, first, last));
      let side = orientSign(a, b, point);
      if (side == 2) { continue; }
      if (side == 0 && insideBox(a, b, point)) { return true; }
      if (a.y <= point.y && b.y > point.y && side > 0) { isInside = !isInside; }
      if (b.y <= point.y && a.y > point.y && side < 0) { isInside = !isInside; }
    }
  }
  return isInside;
}
// Convex counter-clockwise cell contains 'point' (boundary included).
fn cellContains(cell: u32, point: vec2f) -> bool {
  for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
    let side = orientSign(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), point);
    if (side < 0 || side == 2) { return false; }
  }
  return true;
}
fn cellIntersectsExtent(cell: u32) -> bool {
  if (IS_POINT) { return extentContains(cellVertex(cell, 0u)); }
  var low = cellVertex(cell, 0u);
  var high = low;
  for (var slot = 1u; slot < VERTICES_PER_CELL; slot++) {
    let corner = cellVertex(cell, slot);
    low = min(low, corner);
    high = max(high, corner);
  }
  // Contact test: an extent edge whose box misses the cell box cannot touch the cell, so only the
  // few edges near the cell pay for exact predicates.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let q1 = extentVertex(vertex);
      let q2 = extentVertex(nextVertex(vertex, first, last));
      if (max(q1.x, q2.x) < low.x || min(q1.x, q2.x) > high.x ||
          max(q1.y, q2.y) < low.y || min(q1.y, q2.y) > high.y) {
        continue;
      }
      for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
        if (segmentsTouch(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), q1, q2)) {
          return true;
        }
      }
    }
  }
  // No contact, so the cell outline lies wholly inside or wholly outside the extent (it never
  // crosses the boundary): one corner decides, instead of one containment scan per corner.
  if (extentContains(cellVertex(cell, 0u))) { return true; }
  // No contact: a ring is either wholly inside the cell or wholly outside it.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    if (ringEnd(ring) > ringStart(ring) && cellContains(cell, extentVertex(ringStart(ring)))) {
      return true;
    }
  }
  return false;
}`;function g(e,t,n){let r=e=>t?e:``;switch(e){case`square`:case`point`:{let t=`let lower = origin + vec2<f32>(f32(column) * w, f32(row) * h);
  center = lower + 0.5 * vec2<f32>(w, h);`;return e===`point`?t:`${t}
  ${r(`// Every vertex from the lattice expression (as the corner output), so neighbours share bits.
  let upper = origin + vec2<f32>(f32(column + 1u) * w, f32(row + 1u) * h);
  writeVertex(index, 0u, lower);
  writeVertex(index, 1u, vec2<f32>(upper.x, lower.y));
  writeVertex(index, 2u, upper);
  writeVertex(index, 3u, vec2<f32>(lower.x, upper.y));`)}`}case`hex`:return n?`let circumradius = w / SQRT3;
  center = origin + vec2<f32>(circumradius + f32(column) * 1.5 * circumradius,
    0.5 * w + f32(row) * w + select(0.0, 0.5 * w, (column & 1u) == 1u));
  ${r(`for (var corner = 0u; corner < 6u; corner++) {
    // Flat top: first corner at angle 0, counter-clockwise. Pointy-frame slot (7 - corner) % 6 is
    // that corner after the axis swap.
    writeVertex(index, corner, hexLatticeVertex(origin, w, row, column, (7u - corner) % 6u));
  }`)}`:`let circumradius = w / SQRT3;
  center = origin + vec2<f32>(0.5 * w + f32(column) * w + shift, circumradius + f32(row) * 1.5 * circumradius);
  ${r(`for (var corner = 0u; corner < 6u; corner++) {
    // Pointy top: first corner at 30 degrees, counter-clockwise.
    writeVertex(index, corner, hexLatticeVertex(origin, w, column, row, corner));
  }`)}`;case`triangle`:return`let pairIndex = column / 2u;
  let half = 0.5 * w;
  let step = 2u * pairIndex + (row & 1u);
  let y0 = origin.y + f32(row) * h;
  let y1 = origin.y + f32(row + 1u) * h;
  var a: vec2<f32>;
  var b: vec2<f32>;
  var c: vec2<f32>;
  if ((column & 1u) == 0u) {
    a = vec2<f32>(origin.x + f32(step) * half, y0);
    b = vec2<f32>(origin.x + f32(step + 2u) * half, y0);
    c = vec2<f32>(origin.x + f32(step + 1u) * half, y1);
  } else {
    a = vec2<f32>(origin.x + f32(step + 2u) * half, y0);
    b = vec2<f32>(origin.x + f32(step + 3u) * half, y1);
    c = vec2<f32>(origin.x + f32(step + 1u) * half, y1);
  }
  center = (a + b + c) / 3.0;
  ${r(`writeVertex(index, 0u, a);
  writeVertex(index, 1u, b);
  writeVertex(index, 2u, c);`)}`;default:throw Error(`Unknown grid type ${String(e)}`)}}export{c as n,l as r,f as t};