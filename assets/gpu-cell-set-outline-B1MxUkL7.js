import{d as e}from"./analysis-frame-Blu0tYdw.js";import{c as t,i as n,n as r,r as i}from"./cell-topology-wgsl-BT8_s49b.js";import{_ as a,h as o,j as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-scan-BuGhTLDQ.js";import{r as d,s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as p}from"./gpu-segment-ring-assembly-CMjctMio.js";import{n as m}from"./gpu-point-to-cell-DabDU1o1.js";var h=`GPUCellSetOutline`,g={h3:10,quadbin:4},_=class{id;props;edgeSlots;edgeModel;outputSpatialContext;interiorSide;constructor(t){this.id=t.id??`cell-set-outline`,this.props=t;let n=this.id;if(t.family!==`h3`&&t.family!==`quadbin`)throw Error(`${n} family must be 'h3' or 'quadbin'`);this.edgeSlots=g[t.family],this.edgeModel=t.family===`h3`?`great-circle`:`linear`,this.outputSpatialContext={...e,edgeModel:this.edgeModel},this.interiorSide=t.family===`h3`?`left`:`right`;let{output:r}=t;l(t.cells,[`uint32x2`],`${n} cells`);let i=t.cells.length;if(i<1)throw Error(`${n} needs at least one cell row`);if(i*this.edgeSlots>=4294967295)throw Error(`${n} cells length times ${this.edgeSlots} must stay below 2^32`);for(let[e,i]of[[`count`,t.count],[`output.status.count`,r.status.count],[`output.status.overflow`,r.status.overflow],[`output.status.requiredCount`,r.status.requiredCount]])if(i&&(s(i,`${n} ${e}`),i.length<1))throw Error(`${n} ${e} must contain one uint32 row`);if(t.groups&&(s(t.groups,`${n} groups`),t.groups.length!==i))throw Error(`${n} groups length must equal the cell row count`);s(r.rows,`${n} output.rows`);let a=r.rows.length;if(a<1)throw Error(`${n} output capacity must be at least 1`);l(r.cells,[`uint32x2`],`${n} output.cells`),s(r.edgeIndices,`${n} output.edgeIndices`),l(r.endpoints,[`float32x4`],`${n} output.endpoints`);for(let[e,t]of[[`cells`,r.cells.length],[`edgeIndices`,r.edgeIndices.length],[`endpoints`,r.endpoints.length]])if(t!==a)throw Error(`${n} output.${e} must have the same length as output.rows`);if(r.groups){if(!t.groups)throw Error(`${n} output.groups needs the groups input`);if(s(r.groups,`${n} output.groups`),r.groups.length!==a)throw Error(`${n} output.groups must have the same length as output.rows`)}o(n,v(r),[t.cells,t.count,t.groups])}getCommandNodes(e){let{id:o,props:s,edgeSlots:l}=this,{output:g,family:_}=s;a(o,e,[s.cells,s.count,s.groups,...v(g)]);let S=s.cells.length,C=S*l,w=g.rows.length,T=c(e,`${o}-masks`,`uint32`,S),E=c(e,`${o}-flags`,`uint32`,S),D=c(e,`${o}-accepted`,`uint32`,S),O=c(e,`${o}-candidate-endpoints`,`float32x4`,C),k=c(e,`${o}-total`,`uint32`,1),A=[{name:`cells`,view:s.cells,type:`u32`,access:`read`}];s.count&&A.push({name:`rowCount`,view:s.count,type:`u32`,access:`read`}),s.groups&&A.push({name:`groups`,view:s.groups,type:`u32`,access:`read`}),A.push({name:`masks`,view:T,type:`u32`,access:`read_write`},{name:`flags`,view:E,type:`u32`,access:`read_write`},{name:`candidateEndpoints`,view:O,type:`f32`,access:`read_write`});let j=_===`h3`,M=f(e,{id:`${o}-classify`,operation:h,variant:_,bindings:A,invocationCount:S,declarations:`const ROWS: u32 = ${S}u;
const EDGE_SLOTS: u32 = ${l}u;
const CELL_TOPOLOGY_K: u32 = 0u;
const CELL_TOPOLOGY_CAPACITY: u32 = 1u;
${t.source}
${n}
${j?i:``}
${j?m:``}
${r}
${y.replace(`GROUP_TEST`,s.groups?`return groups[groupsOffset + neighborRow] == groups[groupsOffset + row];`:`return true;`)}
${j?b:x}`,body:`let validRows = ${s.count?`min(rowCount[rowCountOffset], ROWS)`:`ROWS`};
  masks[masksOffset + index] = 0u;
  if (index < validRows) {
    outlineClassifyRow(index, validRows);
  }
  flags[flagsOffset + index] = countOneBits(masks[masksOffset + index]);`}),N=[{name:`accepted`,view:D,type:`u32`,access:`read`},{name:`masks`,view:T,type:`u32`,access:`read`},{name:`candidateEndpoints`,view:O,type:`f32`,access:`read`},{name:`cells`,view:s.cells,type:`u32`,access:`read`},{name:`outRows`,view:g.rows,type:`u32`,access:`read_write`},{name:`outCells`,view:g.cells,type:`u32`,access:`read_write`},{name:`outEdges`,view:g.edgeIndices,type:`u32`,access:`read_write`},{name:`outEndpoints`,view:g.endpoints,type:`f32`,access:`read_write`}],P=f(e,{id:`${o}-write`,operation:h,variant:`write`,bindings:N,invocationCount:S,declarations:`const EDGE_SLOTS: u32 = ${l}u;
const OUTPUT_CAPACITY: u32 = ${w}u;`,body:`let mask = masks[masksOffset + index];
  if (mask == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  for (var edge = 0u; edge < EDGE_SLOTS; edge++) {
    if ((mask & (1u << edge)) == 0u) { continue; }
    if (outputSlot < OUTPUT_CAPACITY) {
      let candidate = index * EDGE_SLOTS + edge;
      outRows[outRowsOffset + outputSlot] = index;
      outEdges[outEdgesOffset + outputSlot] = edge;
      outCells[outCellsOffset + 2u * outputSlot] = cells[cellsOffset + 2u * index];
      outCells[outCellsOffset + 2u * outputSlot + 1u] = cells[cellsOffset + 2u * index + 1u];
      for (var component = 0u; component < 4u; component++) {
        outEndpoints[outEndpointsOffset + 4u * outputSlot + component] =
          candidateEndpoints[candidateEndpointsOffset + 4u * candidate + component];
      }
    }
    outputSlot++;
  }`}),F=g.groups&&s.groups?f(e,{id:`${o}-write-groups`,operation:h,variant:`write-groups`,bindings:[{name:`accepted`,view:D,type:`u32`,access:`read`},{name:`masks`,view:T,type:`u32`,access:`read`},{name:`groups`,view:s.groups,type:`u32`,access:`read`},{name:`outGroups`,view:g.groups,type:`u32`,access:`read_write`}],invocationCount:S,declarations:`const OUTPUT_CAPACITY: u32 = ${w}u;`,body:`let keptCount = countOneBits(masks[masksOffset + index]);
  if (keptCount == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  let group = groups[groupsOffset + index];
  for (var kept = 0u; kept < keptCount; kept++) {
    if (outputSlot + kept < OUTPUT_CAPACITY) {
      outGroups[outGroupsOffset + outputSlot + kept] = group;
    }
  }`}):null,I=f(e,{id:`${o}-total`,operation:h,variant:`total`,bindings:[{name:`accepted`,view:D,type:`u32`,access:`read`},{name:`total`,view:k,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const ROW_COUNT: u32 = ${S}u;`,body:`total[totalOffset] = accepted[acceptedOffset + ROW_COUNT - 1u];`}),L=s.rings?new p({id:`${o}-rings`,endpoints:g.endpoints,count:g.status.count,groups:g.groups,vertexTolerance:s.rings.vertexTolerance??5e-5,normalizeWinding:s.rings.normalizeWinding,interiorSide:this.interiorSide,spatialContext:this.outputSpatialContext,output:s.rings.output}).getCommandNodes(e):[];return[M,...new u({id:`${o}-scan`,input:E,output:D,mode:`inclusive`}).getCommandNodes(e),P,...F?[F]:[],I,d(e,{id:`${o}-publish`,operation:h,requiredCount:k,output:{ids:g.rows,status:{count:g.status.count,overflow:g.status.overflow,requiredCount:g.status.requiredCount}}}),...L]}};function v(e){return[e.rows,e.cells,e.edgeIndices,e.endpoints,e.groups,e.status.count,e.status.overflow,e.status.requiredCount]}var y=`
const OUTLINE_NO_ROW: u32 = 0xffffffffu;

fn outlineGetKey(row: u32) -> vec2u {
  return vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]);
}

fn outlineIsKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}

/** Row of \`key\` in the sorted set, or OUTLINE_NO_ROW. */
fn outlineFindRow(key: vec2u, validRows: u32) -> u32 {
  if (key.x == 0u && key.y == 0u) {
    return OUTLINE_NO_ROW;
  }
  var low = 0u;
  var high = validRows;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (outlineIsKeyLess(outlineGetKey(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < validRows && all(outlineGetKey(low) == key)) {
    return low;
  }
  return OUTLINE_NO_ROW;
}

/** Whether the neighbor key is in the set with the same group as \`row\`. */
fn outlineHasSameSideNeighbor(row: u32, neighbor: vec2u, validRows: u32) -> bool {
  let neighborRow = outlineFindRow(neighbor, validRows);
  if (neighborRow == OUTLINE_NO_ROW) {
    return false;
  }
  GROUP_TEST
}

fn outlineEmit(row: u32, edge: u32, a: vec2f, b: vec2f) {
  let slot = row * EDGE_SLOTS + edge;
  masks[masksOffset + row] = masks[masksOffset + row] | (1u << edge);
  candidateEndpoints[candidateEndpointsOffset + 4u * slot] = a.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 1u] = a.y;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 2u] = b.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 3u] = b.y;
}
`,b=`
fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!dggs_h3_is_valid_cell_id(key)) {
    return;
  }
  let boundary = cellIndexH3GetBoundary(key);
  var neighbors: array<vec2u, 6>;
  var centers: array<vec2f, 6>;
  for (var direction = 1u; direction <= 6u; direction++) {
    let neighbor = cellTopologyH3Neighbor(key, direction);
    neighbors[direction - 1u] = neighbor;
    if (neighbor.x != 0u || neighbor.y != 0u) {
      centers[direction - 1u] = dggs_h3_get_center_lnglat(neighbor);
    }
  }
  for (var edge = 0u; edge < boundary.count; edge++) {
    let a = boundary.points[edge];
    let b = boundary.points[select(edge + 1u, 0u, edge + 1u == boundary.count)];
    // Wrap the second longitude next to the first so the midpoint is on the short arc.
    let wrappedB = vec2f(a.x + (b.x - a.x) - 360.0 * round((b.x - a.x) / 360.0), b.y);
    let midpoint = 0.5 * (a + wrappedB);
    let cosine = max(cos(radians(midpoint.y)), 0.01);
    var best = 1e30;
    var bestNeighbor = vec2u(0u);
    for (var slot = 0u; slot < 6u; slot++) {
      if (neighbors[slot].x == 0u && neighbors[slot].y == 0u) {
        continue;
      }
      let delta = centers[slot] - midpoint;
      let dx = (delta.x - 360.0 * round(delta.x / 360.0)) * cosine;
      let distance = dx * dx + delta.y * delta.y;
      if (distance < best) {
        best = distance;
        bestNeighbor = neighbors[slot];
      }
    }
    if (!outlineHasSameSideNeighbor(row, bestNeighbor, validRows)) {
      outlineEmit(row, edge, a, b);
    }
  }
}
`,x=`
fn outlineTileLatitude(row: u32, tileScale: f32) -> f32 {
  return dggs_web_mercator_tile_y_to_latitude(f32(row), tileScale);
}

fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!cellIsValidQuadbin(key)) {
    return;
  }
  let z = (key.x >> 20u) & 0x1fu;
  let tile = cellTopologyQuadbinGetTile(key);
  let tileCount = i32(1u << z);
  let tileScale = f32(1u << z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = outlineTileLatitude(tile.y, tileScale);
  let south = outlineTileLatitude(tile.y + 1u, tileScale);
  // Vertices NW, NE, SE, SW; edge e runs vertex e to vertex e + 1. Neighbors: N, E, S, W.
  var corners = array<vec2f, 4>(
    vec2f(west, north), vec2f(east, north), vec2f(east, south), vec2f(west, south)
  );
  var offsets = array<vec2i, 4>(vec2i(0, -1), vec2i(1, 0), vec2i(0, 1), vec2i(-1, 0));
  for (var edge = 0u; edge < 4u; edge++) {
    let neighborTile = vec2i(tile) + offsets[edge];
    var neighbor = vec2u(0u);
    if (neighborTile.y >= 0 && neighborTile.y < tileCount) {
      let x = u32(((neighborTile.x % tileCount) + tileCount) % tileCount);
      neighbor = cellTopologyQuadbinGetKey(x, u32(neighborTile.y), z);
    }
    if (!outlineHasSameSideNeighbor(row, neighbor, validRows)) {
      outlineEmit(row, edge, corners[edge], corners[(edge + 1u) % 4u]);
    }
  }
}
`;export{_ as t};