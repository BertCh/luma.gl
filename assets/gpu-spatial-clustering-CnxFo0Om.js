import{E as e,p as t,w as n}from"./analysis-frame-Blu0tYdw.js";import{D as r,E as i,I as a,R as o,_ as s,f as c,h as l,j as u,k as d,o as f,v as p,w as m,x as h,y as g,z as _}from"./resources-CwvhEOBe.js";import{o as v}from"./gpu-subgroup-utils-BuV3X9Xp.js";import{t as y}from"./gpu-scan-BuGhTLDQ.js";import{t as b}from"./gpu-compaction-Cpx1OCRM.js";import{t as ee}from"./gpu-grid-index-D1JNO067.js";import{t as x}from"./gpu-group-aggregation-DrNKSCi8.js";import{n as S,r as C,s as w}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as T}from"./wgsl-common-BNRVzgd8.js";import{o as E}from"./sorted-segment-sums-DwGkmbx2.js";import{i as D,n as O}from"./key-pair-grouping--fUW7Rco.js";import{t as k}from"./geodesic-wgsl-DS0CJZ42.js";var A=256,j=class{id;props;constructor(e){this.id=e.id??`gpu-flag-offsets`,this.props=Object.freeze({...e,id:this.id}),N(this.props)}getCommandNodes(e){let t=[],{flags:n,count:r}=this.props;for(let t of[...P(n),...P(this.props.offsets),r])if(t.buffer.graph!==e)throw Error(`${this.id} views must belong to the target graph`);let a=i(e,this.props.offsets,n.length);return n.length>0&&t.push(...new y({id:`${this.id}-scan`,input:n,output:a,mode:`exclusive`}).getCommandNodes(e)),t.push(...M(e,{...this.props,offsets:a})),t}};function M(e,t){let n=[],i=t.flags.length>0,o=i?F(t.flags):void 0,s=i?F(t.offsets):void 0,c=h(`GPUFlagOffsetsCount`,1,A,e.device.limits.maxComputeWorkgroupsPerDimension),l=i?`const LAST_FLAG_INDEX: u32 = ${o.length-1}u;
const LAST_OFFSET_INDEX: u32 = ${s.length-1}u;
const FLAG_OFFSET: u32 = ${d(o)}u;
const OFFSETS_OFFSET: u32 = ${d(s)}u;
const COUNT_OFFSET: u32 = ${d(t.count)}u;
@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@compute @workgroup_size(${A})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32) {
  if (localInvocationIndex > 0u) { return; }
  count[COUNT_OFFSET] = offsets[OFFSETS_OFFSET + LAST_OFFSET_INDEX] + flags[FLAG_OFFSET + LAST_FLAG_INDEX];
}`:`const COUNT_OFFSET: u32 = ${d(t.count)}u;
@group(0) @binding(0) var<storage, read_write> count: array<u32>;
@compute @workgroup_size(${A})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32) {
  if (localInvocationIndex > 0u) { return; }
  count[COUNT_OFFSET] = 0u;
}`,u=i?[{name:`flags`,view:o,usage:`storage-read`},{name:`offsets`,view:s,usage:`storage-read`},{name:`count`,view:t.count,usage:`storage-write`}]:[{name:`count`,view:t.count,usage:`storage-write`}];n.push(a({id:`${t.id}-count`,workload:{operation:`GPUFlagOffsetsCount`,commandCount:1,maximumWorkgroupCount:1,maximumInvocationCount:A,readByteLength:i?8:0,writeByteLength:4},resources:u.map(e=>({buffer:e.view,usage:e.usage})),compile:({device:e})=>{let n=new _(e,{id:`${t.id}-count`,source:l,shaderLayout:{bindings:u.map((e,t)=>({name:e.name,type:e.usage===`storage-read`?`read-only-storage`:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:t})=>{let i={};for(let e of u)i[e.name]=r(e.view,t);n.dispatch(e,{bindings:i,x:c.x,y:c.y,z:c.z})},destroy:()=>n.destroy()}}}));let f=v(e,()=>({op:`GPUFlagOffsets`,moduleId:`GPUFlagOffsets/count`,source:l,workgroupSize:[A,1,1],workgroups:[c.x,c.y,c.z],invocationCount:1,formula:`gpu-core.flag-offsets`,bindings:u.map(({name:e,view:t,usage:n})=>({name:e,view:t,access:n===`storage-read`?`read`:`read_write`}))}));return f&&(n[0].annotation=f),n}function N(e){for(let t of P(e.flags))u(t,`${e.id} flags`);for(let t of P(e.offsets))u(t,`${e.id} offsets`);if(u(e.count,`${e.id} count`),e.offsets.length<e.flags.length)throw Error(`${e.id} offsets must contain at least flags.length rows`);if(e.count.length<1)throw Error(`${e.id} result views are too short`)}function P(e){return e instanceof o?e.data:[e]}function F(e){let t=P(e);for(let e=t.length-1;e>=0;e--)if(t[e].length>0)return t[e];throw Error(`Non-empty GPUFlagOffsets input requires a non-empty chunk`)}var I=`GPUSpatialClustering`;function L(e,t,n){let r=n!==void 0,i=r?String(Math.fround(n)):`1`,a=r?`return geodesicHaversineTerm(vec2<f32>(ax, ay), vec2<f32>(bx, by)) <= lattice.epsilonSquared;`:`let deltaX = bx - ax;
  let deltaY = by - ay;
  return deltaX * deltaX + deltaY * deltaY <= lattice.epsilonSquared;`;return`
${r?k:``}
const GEOGRAPHIC: bool = ${r};
const SPHERE_RADIUS: f32 = ${/[.e]/.test(i)?i:`${i}.0`};
const DEGREES_PER_RADIAN: f32 = 57.2957795130823;
const RADIANS_PER_DEGREE: f32 = 0.0174532925199433;
const COLUMNS: u32 = ${e}u;
const ROWS: u32 = ${t}u;
const NOISE: u32 = 0xffffffffu;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  epsilonSquared: f32,
  minimumPoints: u32
}

${T}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.epsilonSquared = 0.0;
  lattice.minimumPoints = 1u;
  let minimumX = parameters[parametersOffset];
  let minimumY = parameters[parametersOffset + 1u];
  let maximumX = parameters[parametersOffset + 2u];
  let maximumY = parameters[parametersOffset + 3u];
  let epsilon = parameters[parametersOffset + 4u];
  let minimumPointsValue = parameters[parametersOffset + 5u];
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  // Planar: neighbors satisfy dx^2 + dy^2 <= epsilon^2, so a cell must be epsilon wide. Geographic:
  // the haversine term of neighbors is at most sin^2(epsilon / (2 R)).
  var epsilonSquared = epsilon * epsilon;
  var cellStepX = epsilon * CELL_MARGIN;
  var cellStepY = cellStepX;
  var inRange = true;
  if (GEOGRAPHIC) {
    let halfAngle = 0.5 * epsilon / SPHERE_RADIUS;
    let sinHalf = sin(halfAngle);
    epsilonSquared = sinHalf * sinHalf;
    cellStepY = 2.0 * halfAngle * DEGREES_PER_RADIAN * CELL_MARGIN;
    let cosLatitude = cos(max(abs(minimumY), abs(maximumY)) * RADIANS_PER_DEGREE);
    let ratio = sinHalf / max(cosLatitude, 1e-6);
    cellStepX = select(2.0 * asin(ratio) * DEGREES_PER_RADIAN * CELL_MARGIN, 360.0, ratio >= 0.99);
    inRange = minimumX >= -180.0 && maximumX <= 180.0 && minimumY >= -90.0 && maximumY <= 90.0 &&
      halfAngle < 1.5707963;
  }
  lattice.valid = inRange &&
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(epsilon) && isFiniteFloat(minimumPointsValue) &&
    isFiniteFloat(width) && isFiniteFloat(height) && isFiniteFloat(epsilonSquared) &&
    isFiniteFloat(cellStepX) && isFiniteFloat(cellStepY) && epsilon > 0.0 && width >= 0.0 &&
    height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.cellWidth = max(width / f32(COLUMNS), cellStepX);
  lattice.cellHeight = max(height / f32(ROWS), cellStepY);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.epsilonSquared = epsilonSquared;
  lattice.minimumPoints = u32(max(minimumPointsValue, 0.0));
  return lattice;
}

fn isWithinEpsilon(lattice: Lattice, ax: f32, ay: f32, bx: f32, by: f32) -> bool {
  ${a}
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}
`}function R(e,t=`true`){return`
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow && ${t}; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn && ${t}; cellColumn++) {
      let cell = cellRow * COLUMNS + cellColumn;
      let start = cellOffsets[cellOffsetsOffset + cell];
      let end = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = start; slot < end && ${t}; slot++) {
        if (isWithinEpsilon(lattice, x, y, cellPositions[cellPositionsOffset + slot * 2u],
            cellPositions[cellPositionsOffset + slot * 2u + 1u])) {
          let neighbor = sortedRows[sortedRowsOffset + slot];
          ${e}
        }
      }
    }
  }`}var z=`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let point = sortedRows[sortedRowsOffset + index];
  let x = cellPositions[cellPositionsOffset + index * 2u];
  let y = cellPositions[cellPositionsOffset + index * 2u + 1u];`;function B(e){return[{name:`cellPositions`,view:e.cellPositions,type:`f32`,access:`read`},{name:`parameters`,view:e.parameters,type:`f32`,access:`read`},{name:`sortedRows`,view:e.sortedRows,type:`u32`,access:`read`},{name:`cellOffsets`,view:e.cellOffsets,type:`u32`,access:`read`},{name:`gridCount`,view:e.gridCount,type:`u32`,access:`read`}]}function V(e,t){return w(e,{id:t.id,operation:I,variant:`grid-bounds`,bindings:[{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`gridBounds`,view:t.gridBounds,type:`f32`,access:`read_write`}],invocationCount:1,declarations:L(t.gridSize[0],t.gridSize[1],t.geographicRadius),body:`let lattice = readLattice();
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridBounds[gridBoundsOffset] = select(invalid, lattice.minimumX, lattice.valid);
  gridBounds[gridBoundsOffset + 1u] = select(invalid, lattice.minimumY, lattice.valid);
  gridBounds[gridBoundsOffset + 2u] = select(invalid,
    max(lattice.maximumX, lattice.minimumX + lattice.cellWidth * f32(COLUMNS)), lattice.valid);
  gridBounds[gridBoundsOffset + 3u] = select(invalid,
    max(lattice.maximumY, lattice.minimumY + lattice.cellHeight * f32(ROWS)), lattice.valid);`})}function H(e,t){return w(e,{id:t.id,operation:I,variant:`grid-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`gridPositions`,view:t.gridPositions,type:`f32`,access:`read_write`},{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`}],invocationCount:t.positions.length,declarations:L(t.gridSize[0],t.gridSize[1],t.geographicRadius),body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] = select(invalid, x, valid);
  gridPositions[gridPositionsOffset + index * 2u + 1u] = select(invalid, y, valid);
  rowIds[rowIdsOffset + index] = index;`})}function U(e,t){let n=t.variant===`cell-positions`,r=n?`f32`:`u32`,i=n?`cellValues[cellValuesOffset + index * 2u] = values[valuesOffset + row * 2u];
  cellValues[cellValuesOffset + index * 2u + 1u] = values[valuesOffset + row * 2u + 1u];`:`cellValues[cellValuesOffset + index] = values[valuesOffset + row];`;return w(e,{id:t.id,operation:I,variant:t.variant,bindings:[{name:`values`,view:t.values,type:r,access:`read`},{name:`sortedRows`,view:t.sortedRows,type:`u32`,access:`read`},{name:`gridCount`,view:t.gridCount,type:`u32`,access:`read`},{name:`cellValues`,view:t.cellValues,type:r,access:`read_write`}],invocationCount:t.sortedRows.length,body:`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  ${i}`})}function W(e,t){let{boxCounts:n}=t;return w(e,{id:t.id,operation:I,variant:n?`core-dense-box`:`core`,bindings:[...B(t),{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read_write`},...n?[{name:`boxCounts`,view:n,type:`u32`,access:`read`}]:[]],invocationCount:t.cellPositions.length,declarations:L(t.gridSize[0],t.gridSize[1],t.geographicRadius),body:`${z}
  let lattice = readLattice();
  var isCore = 0u;
  if (isPointValid(lattice, x, y)) {
    ${n?`if (boxCounts[boxCountsOffset + point] >= lattice.minimumPoints) {
      isCore = 1u;
    } else {`:`{`}
      var neighborCount = 0u;
      ${R(`neighborCount++;`,`neighborCount < lattice.minimumPoints`)}
      isCore = select(0u, 1u, neighborCount >= lattice.minimumPoints);
    }
  }
  coreFlags[coreFlagsOffset + point] = isCore;`})}function G(e,t){return w(e,{id:t.id,operation:I,variant:`box-keys`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`boxHigh`,view:t.boxHigh,type:`u32`,access:`read_write`},{name:`boxLow`,view:t.boxLow,type:`u32`,access:`read_write`}],invocationCount:t.positions.length,declarations:`${L(t.gridSize[0],t.gridSize[1],t.geographicRadius)}
const BOX_SIDE_FACTOR: f32 = 0.7;
const MAXIMUM_BOX_INDEX: f32 = 65536.0;`,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var high = 0xffffffffu;
  var low = 0xffffffffu;
  if (isPointValid(lattice, x, y)) {
    let side = parameters[parametersOffset + 4u] * BOX_SIDE_FACTOR;
    let boxX = floor((x - lattice.minimumX) / side);
    let boxY = floor((y - lattice.minimumY) / side);
    if (boxX >= 0.0 && boxX < MAXIMUM_BOX_INDEX && boxY >= 0.0 && boxY < MAXIMUM_BOX_INDEX) {
      high = u32(boxX);
      low = u32(boxY);
    }
  }
  boxHigh[boxHighOffset + index] = high;
  boxLow[boxLowOffset + index] = low;`})}function K(e,t){return w(e,{id:t.id,operation:I,variant:`box-counts`,bindings:[{name:`sortedItems`,view:t.sortedItems,type:`u32`,access:`read`},{name:`groupIndex`,view:t.groupIndex,type:`u32`,access:`read`},{name:`groupStarts`,view:t.groupStarts,type:`u32`,access:`read`},{name:`boxHigh`,view:t.boxHigh,type:`u32`,access:`read`},{name:`boxCounts`,view:t.boxCounts,type:`u32`,access:`read_write`}],invocationCount:t.sortedItems.length,body:`let item = sortedItems[sortedItemsOffset + index];
  let group = groupIndex[groupIndexOffset + index] - 1u;
  let count = groupStarts[groupStartsOffset + group + 1u] - groupStarts[groupStartsOffset + group];
  boxCounts[boxCountsOffset + item] = select(0u, count, boxHigh[boxHighOffset + item] != 0xffffffffu);`})}function q(e,t){return w(e,{id:t.id,operation:I,variant:`parents-init`,bindings:[{name:`parents`,view:t.parents,type:`u32`,access:`read_write`}],invocationCount:t.parents.length,body:`parents[parentsOffset + index] = index;`})}function J(e,t){return w(e,{id:t.id,operation:I,variant:`union`,bindings:[...B(t),{name:`cellCoreFlags`,view:t.cellCoreFlags,type:`u32`,access:`read`},{name:`parents`,view:t.parents,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.cellPositions.length,declarations:`${L(t.gridSize[0],t.gridSize[1],t.geographicRadius)}
// Path halving: a visited node is a non-root forever, so any ancestor is a valid parent for it.
fn findRoot(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) {
      return current;
    }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    if (grandparent != parent) {
      atomicStore(&parents[parentsOffset + current], grandparent);
    }
    current = parent;
  }
}

// Hooks the larger root under the smaller one; retries with fresh roots when another invocation
// hooked the larger root first (or the weak compare-exchange failed spuriously).
fn unite(first: u32, second: u32) {
  var a = findRoot(first);
  var b = findRoot(second);
  loop {
    if (a == b) {
      return;
    }
    let low = min(a, b);
    let high = max(a, b);
    let result = atomicCompareExchangeWeak(&parents[parentsOffset + high], high, low);
    if (result.exchanged) {
      return;
    }
    a = findRoot(low);
    b = findRoot(high);
  }
}`,body:`${z}
  if (cellCoreFlags[cellCoreFlagsOffset + index] == 0u) {
    return;
  }
  let lattice = readLattice();
  ${R(`if (neighbor < point && cellCoreFlags[cellCoreFlagsOffset + slot] != 0u) {
            unite(point, neighbor);
          }`)}`})}function Y(e,t){return w(e,{id:t.id,operation:I,variant:`resolve`,bindings:[{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read`},{name:`parents`,view:t.parents,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read_write`}],invocationCount:t.rootLabels.length,body:`var root = 0xffffffffu;
  if (coreFlags[coreFlagsOffset + index] != 0u) {
    root = index;
    loop {
      let parent = parents[parentsOffset + root];
      if (parent == root) {
        break;
      }
      root = parent;
    }
  }
  rootLabels[rootLabelsOffset + index] = root;`})}function te(e,t){return w(e,{id:t.id,operation:I,variant:`border`,bindings:[...B(t),{name:`cellCoreFlags`,view:t.cellCoreFlags,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read_write`}],invocationCount:t.cellPositions.length,declarations:L(t.gridSize[0],t.gridSize[1],t.geographicRadius),body:`${z}
  if (cellCoreFlags[cellCoreFlagsOffset + index] != 0u) {
    return;
  }
  let lattice = readLattice();
  if (!isPointValid(lattice, x, y)) {
    return;
  }
  var bestRoot = NOISE;
  ${R(`if (cellCoreFlags[cellCoreFlagsOffset + slot] != 0u) {
            bestRoot = min(bestRoot, rootLabels[rootLabelsOffset + neighbor]);
          }`)}
  rootLabels[rootLabelsOffset + point] = bestRoot;`})}function ne(e,t){return w(e,{id:t.id,operation:I,variant:`root-flags`,bindings:[{name:`coreFlags`,view:t.coreFlags,type:`u32`,access:`read`},{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read`},{name:`rootFlags`,view:t.rootFlags,type:`u32`,access:`read_write`}],invocationCount:t.rootLabels.length,body:`let isRoot = coreFlags[coreFlagsOffset + index] != 0u &&
    rootLabels[rootLabelsOffset + index] == index;
  rootFlags[rootFlagsOffset + index] = select(0u, 1u, isRoot);`})}function re(e,t){let n=[{name:`rootLabels`,view:t.rootLabels,type:`u32`,access:`read`},{name:`clusterOffsets`,view:t.clusterOffsets,type:`u32`,access:`read`},{name:`labels`,view:t.labels,type:`u32`,access:`read_write`}],r=t.memberXs&&t.memberYs;return t.memberXs&&t.memberYs&&n.push({name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`memberXs`,view:t.memberXs,type:`f32`,access:`read_write`},{name:`memberYs`,view:t.memberYs,type:`f32`,access:`read_write`}),w(e,{id:t.id,operation:I,variant:`labels`,bindings:n,invocationCount:t.rootLabels.length,body:`let root = rootLabels[rootLabelsOffset + index];
  labels[labelsOffset + index] = select(clusterOffsets[clusterOffsetsOffset + min(root, INVOCATION_COUNT - 1u)], 0xffffffffu, root == 0xffffffffu);
  ${r?`memberXs[memberXsOffset + index] = positions[positionsOffset + index * 2u];
  memberYs[memberYsOffset + index] = positions[positionsOffset + index * 2u + 1u];`:``}`})}function ie(e,t){return w(e,{id:t.id,operation:I,variant:`centroids`,bindings:[{name:`sizes`,view:t.sizes,type:`u32`,access:`read`},{name:`sumsX`,view:t.sumsX,type:`f32`,access:`read`},{name:`sumsY`,view:t.sumsY,type:`f32`,access:`read`},{name:`centroids`,view:t.centroids,type:`f32`,access:`read_write`}],invocationCount:t.sizes.length,body:`let size = sizes[sizesOffset + index];
  let divisor = f32(max(size, 1u));
  centroids[centroidsOffset + index * 2u] = select(0.0, sumsX[sumsXOffset + index] / divisor, size > 0u);
  centroids[centroidsOffset + index * 2u + 1u] = select(0.0, sumsY[sumsYOffset + index] / divisor, size > 0u);`})}var X=Object.freeze({...t,measure:`distance`}),Z=`GPUSpatialClustering`,Q=2**24-1,$=class{id;props;spatialContext;geographicRadius;constructor(t){this.id=t.id??`spatial-clustering`,this.props=t;let r=this.id;if(this.spatialContext=e(r,t.spatialContext,X),this.spatialContext.coordinateSpace===`longitude-latitude`&&this.spatialContext.metric!==`great-circle`)throw Error(`${r} longitude-latitude positions need the great-circle metric`);if(this.geographicRadius=this.spatialContext.coordinateSpace===`longitude-latitude`?n(this.spatialContext):void 0,this.geographicRadius!==void 0&&t.denseBoxShortcut)throw Error(`${r} denseBoxShortcut is planar only`);let i=t.positions.length;if(t.sumOrder!==void 0&&t.sumOrder!==`atomic`&&t.sumOrder!==`sorted`)throw Error(`${r} sumOrder must be atomic or sorted`);if(g(t.positions,[`float32x2`],`${r} positions`),g(t.parameters,[`float32`],`${r} parameters`),t.parameters.length<8)throw Error(`${r} parameters must hold 8 float32 values`);let[a,o]=t.gridSize;if(!Number.isInteger(a)||!Number.isInteger(o)||a<1||o<1)throw Error(`${r} gridSize must be two positive integers`);for(let[e,n]of[[`sourceIds`,t.sourceIds],[`output.labels`,t.output.labels],[`output.rootRows`,t.output.rootRows],[`output.coreFlags`,t.output.coreFlags]])if(n&&(g(n,[`uint32`],`${r} ${e}`),n.length!==i))throw Error(`${r} ${e} length must equal positions length`);if(t.output.clusterCount&&(g(t.output.clusterCount,[`uint32`],`${r} output.clusterCount`),t.output.clusterCount.length<1))throw Error(`${r} output.clusterCount must contain one uint32 row`);if(t.output.clusters&&(c(r,t.output.clusters),t.output.clusters.ids.length<1))throw Error(`${r} output.clusters.ids must hold at least one row`);if(t.output.drawInstanceCount){if(g(t.output.drawInstanceCount,[`uint32`],`${r} output.drawInstanceCount`),t.output.drawInstanceCount.length<1)throw Error(`${r} output.drawInstanceCount must contain one uint32 row`);if(!t.output.clusters)throw Error(`${r} output.drawInstanceCount requires clusters`)}if((t.output.clusterSizes||t.output.clusterCentroids)&&!t.output.clusters)throw Error(`${r} output.clusterSizes and clusterCentroids require clusters`);let s=t.output.clusters?.ids.length??0;if(t.output.clusterSizes&&(g(t.output.clusterSizes,[`uint32`],`${r} output.clusterSizes`),t.output.clusterSizes.length!==s))throw Error(`${r} output.clusterSizes length must equal clusters.ids length`);if(t.output.clusterCentroids){if(g(t.output.clusterCentroids,[`float32x2`],`${r} output.clusterCentroids`),t.output.clusterCentroids.length!==s)throw Error(`${r} output.clusterCentroids length must equal clusters.ids length`);t.sumOrder!==`atomic`&&p(r,`clusters.ids rows with sorted centroid sums`,s,Q)}l(r,[t.output.labels,t.output.rootRows,t.output.coreFlags,t.output.clusterCount,t.output.clusters?.ids,t.output.clusters?.status.count,t.output.clusters?.status.overflow,t.output.clusters?.status.requiredCount,t.output.clusterSizes,t.output.clusterCentroids,t.output.drawInstanceCount],[t.positions,t.parameters,t.sourceIds])}_getBoxCountNodes(e,t){let{id:n,props:r}=this,i=r.positions.length,a=m(e,`${n}-box-high`,`uint32`,i),o=m(e,`${n}-box-low`,`uint32`,i),s=D(e,`${n}-box`,Z,i,a,o),c=O(e,`${n}-box`,Z,i,a,o,s.sortedItems);return[G(e,{id:`${n}-box-keys`,positions:r.positions,parameters:r.parameters,gridSize:r.gridSize,boxHigh:a,boxLow:o}),...s.nodes,...c.nodes,K(e,{id:`${n}-box-counts`,sortedItems:s.sortedItems,groupIndex:c.groupIndex,groupStarts:c.groupStarts,boxHigh:a,boxCounts:t})]}getCommandNodes(e){let{props:t,id:n}=this,{positions:r,parameters:i,gridSize:a}=t,{clusters:o}=t.output,c=this.geographicRadius;s(n,e,[r,i,t.sourceIds,t.output.labels,t.output.rootRows,t.output.coreFlags,t.output.clusterCount,o?.ids,o?.status.count,o?.status.overflow,o?.status.requiredCount,t.output.clusterSizes,t.output.clusterCentroids,t.output.drawInstanceCount]);let l=a[0]*a[1];p(n,`gridSize cells`,l,f(e.device));let u=r.length,d=o?.ids.length??0,h=!!t.output.clusterCentroids,g=[],_=t.output.clusterCount??m(e,`${n}-cluster-count`,`uint32`,1);if(u===0)return g.push(S(e,{id:`${n}-cluster-count`,operation:Z,view:_,type:`u32`,value:`0u`,componentCount:1})),t.output.clusterSizes&&g.push(S(e,{id:`${n}-cluster-sizes`,operation:Z,view:t.output.clusterSizes,type:`u32`,value:`0u`})),t.output.clusterCentroids&&g.push(S(e,{id:`${n}-cluster-centroids`,operation:Z,view:t.output.clusterCentroids,type:`f32`,value:`0.0`,componentCount:d*2})),o&&g.push(C(e,{id:`${n}-publish`,operation:Z,requiredCount:_,output:o,extraCounts:t.output.drawInstanceCount?[t.output.drawInstanceCount]:[]})),g;let v=m(e,`${n}-grid-bounds`,`float32`,4),y=m(e,`${n}-grid-positions`,`float32x2`,u),w=m(e,`${n}-row-ids`,`uint32`,u),T=m(e,`${n}-cell-offsets`,`uint32`,l+1),D=m(e,`${n}-sorted-rows`,`uint32`,u),O=m(e,`${n}-cell-positions`,`float32x2`,u),k=m(e,`${n}-cell-core-flags`,`uint32`,u),A=m(e,`${n}-grid-count`,`uint32`,1),M=m(e,`${n}-grid-overflow`,`uint32`,1),N=t.output.coreFlags??m(e,`${n}-core-flags`,`uint32`,u),P=m(e,`${n}-parents`,`uint32`,u),F=t.output.rootRows??m(e,`${n}-root-labels`,`uint32`,u),I=m(e,`${n}-root-flags`,`uint32`,u),L=m(e,`${n}-cluster-offsets`,`uint32`,u),R=t.denseBoxShortcut?m(e,`${n}-box-counts`,`uint32`,u):void 0,z={cellPositions:O,parameters:i,sortedRows:D,gridCount:A,cellOffsets:T};g.push(V(e,{id:`${n}-grid-bounds`,parameters:i,gridSize:a,geographicRadius:c,gridBounds:v}),H(e,{id:`${n}-grid-positions`,positions:r,parameters:i,gridSize:a,geographicRadius:c,gridPositions:y,rowIds:w}),...new ee({id:`${n}-grid-index`,positions:y,gridSize:[a[0],a[1]],bounds:[0,0,1,1],boundsBuffer:v,cellOffsets:T,objectIds:D,count:A,overflow:M}).getCommandNodes(e),U(e,{id:`${n}-cell-positions`,variant:`cell-positions`,values:r,sortedRows:D,gridCount:A,cellValues:O}),...R?this._getBoxCountNodes(e,R):[],S(e,{id:`${n}-core-flags-clear`,operation:`GPUSpatialClustering`,view:N,type:`u32`,value:`0u`}),W(e,{id:`${n}-core`,...z,gridSize:a,geographicRadius:c,coreFlags:N,boxCounts:R}),U(e,{id:`${n}-cell-core-flags`,variant:`cell-core-flags`,values:N,sortedRows:D,gridCount:A,cellValues:k}),q(e,{id:`${n}-parents-init`,parents:P}),J(e,{id:`${n}-union`,...z,gridSize:a,geographicRadius:c,cellCoreFlags:k,parents:P}),Y(e,{id:`${n}-resolve`,coreFlags:N,parents:P,rootLabels:F}),te(e,{id:`${n}-border`,...z,gridSize:a,geographicRadius:c,cellCoreFlags:k,rootLabels:F}),ne(e,{id:`${n}-root-flags`,coreFlags:N,rootLabels:F,rootFlags:I}),...new j({id:`${n}-cluster-offsets`,flags:I,offsets:L,count:_}).getCommandNodes(e));let B=h?m(e,`${n}-member-xs`,`float32`,u):void 0,G=h?m(e,`${n}-member-ys`,`float32`,u):void 0;if(g.push(re(e,{id:`${n}-labels`,positions:r,rootLabels:F,clusterOffsets:L,labels:t.output.labels,memberXs:B,memberYs:G})),!o)return g;let K=m(e,`${n}-root-ids`,`uint32`,u),X=m(e,`${n}-compact-count`,`uint32`,1),Q=t.sourceIds??w;g.push(...new b({id:`${n}-cluster-roots`,input:Q,flags:I,output:K,count:X}).getCommandNodes(e));let $=t.output.clusterSizes??(h?m(e,`${n}-cluster-size-values`,`uint32`,d):void 0);if($&&g.push(...new x({id:`${n}-cluster-sizes`,keys:t.output.labels,output:$}).getCommandNodes(e)),t.output.clusterCentroids&&$&&B&&G){let r=m(e,`${n}-cluster-sum-values-x`,`float32`,d),i=m(e,`${n}-cluster-sum-values-y`,`float32`,d);t.sumOrder===`atomic`?g.push(...new x({id:`${n}-cluster-sums-x`,keys:t.output.labels,values:B,output:r,operation:`sum`}).getCommandNodes(e),...new x({id:`${n}-cluster-sums-y`,keys:t.output.labels,values:G,output:i,operation:`sum`}).getCommandNodes(e)):g.push(...E(e,{id:`${n}-cluster-sums`,operation:Z,segmentCount:d,segmentKeys:t.output.labels,reductions:[{name:`x`,contributions:B,output:r},{name:`y`,contributions:G,output:i}]})),g.push(ie(e,{id:`${n}-cluster-centroids`,sizes:$,sumsX:r,sumsY:i,centroids:t.output.clusterCentroids}))}return g.push(C(e,{id:`${n}-publish`,operation:Z,requiredCount:_,compactIds:K,output:o,extraCounts:t.output.drawInstanceCount?[t.output.drawInstanceCount]:[]})),g}};export{$ as t};