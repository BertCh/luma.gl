import{E as e,p as t,w as n,y as r}from"./analysis-frame-Blu0tYdw.js";import{n as i,t as a}from"./parameter-schema-DB2zG6AH.js";import{_ as o,d as s,h as c,v as l,w as u,y as d}from"./resources-CwvhEOBe.js";import{t as f}from"./gpu-scan-BuGhTLDQ.js";import{l as p,n as m,s as h}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{o as g}from"./sorted-segment-sums-DwGkmbx2.js";import{t as _}from"./scan-total-BdolCtK7.js";import{t as v}from"./geodesic-wgsl-DS0CJZ42.js";import{t as y}from"./line-segmentize-kernels-JWzYlmh-.js";function b(e,t){let n=e.coordinateSystem===`spherical`;return`
const LINE_COUNT: u32 = ${e.lineOffsets.length-1}u;
const ROW_COUNT: u32 = ${e.positions.length}u;
const COLUMNS: i32 = ${e.columns};
const ROWS: i32 = ${e.rows};
const CAPACITY: u32 = ${e.capacity}u;
const RADIUS: f32 = ${p(e.radius)};
const INFINITE: f32 = 3.0e38;
const EMITS_PIECES: bool = ${t};
${n?v:``}
${y}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getPieceLength(a: vec2<f32>, b: vec2<f32>, t0: f32, t1: f32) -> f32 {
  ${n?`return geodesicCentralAngle(mix(a, b, t0), mix(a, b, t1)) * RADIUS;`:`return (t1 - t0) * length(b - a);`}
}

fn walkSegment(row: u32, slot: u32) -> u32 {
  let line = findLine(row);
  if (line == NO_LINE) {
    return 0u;
  }
  if (row + 1u >= min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT)) {
    return 0u;
  }
  let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let cellSize = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let a = getPosition(row);
  let b = getPosition(row + 1u);
  let ua = (a - origin) / cellSize;
  let delta = (b - origin) / cellSize - ua;
  // Liang-Barsky against [0, COLUMNS] x [0, ROWS].
  var t0 = 0.0;
  var t1 = 1.0;
  var p = array<f32, 4>(-delta.x, delta.x, -delta.y, delta.y);
  var q = array<f32, 4>(ua.x, f32(COLUMNS) - ua.x, ua.y, f32(ROWS) - ua.y);
  for (var side = 0u; side < 4u; side++) {
    if (p[side] == 0.0) {
      if (q[side] < 0.0) {
        return 0u;
      }
    } else {
      let r = q[side] / p[side];
      if (p[side] < 0.0) {
        if (r > t1) {
          return 0u;
        }
        t0 = max(t0, r);
      } else {
        if (r < t0) {
          return 0u;
        }
        t1 = min(t1, r);
      }
    }
  }
  if (t1 <= t0) {
    return 0u;
  }
  let start = ua + delta * t0;
  let end = ua + delta * t1;
  var cx = clamp(i32(floor(start.x)), 0, COLUMNS - 1);
  var cy = clamp(i32(floor(start.y)), 0, ROWS - 1);
  let endX = clamp(i32(floor(end.x)), 0, COLUMNS - 1);
  let endY = clamp(i32(floor(end.y)), 0, ROWS - 1);
  var stepsX = u32(abs(endX - cx));
  var stepsY = u32(abs(endY - cy));
  let pieceCount = stepsX + stepsY + 1u;
  // The counting pass only needs the piece count; walking the cells would be wasted work.
  if (!EMITS_PIECES) {
    return pieceCount;
  }
  let stepX = select(-1, 1, delta.x > 0.0);
  let stepY = select(-1, 1, delta.y > 0.0);
  var tMaxX = INFINITE;
  var tMaxY = INFINITE;
  if (stepsX > 0u) {
    tMaxX = t0 + (select(f32(cx), f32(cx + 1), delta.x > 0.0) - start.x) / delta.x;
  }
  if (stepsY > 0u) {
    tMaxY = t0 + (select(f32(cy), f32(cy + 1), delta.y > 0.0) - start.y) / delta.y;
  }
  let tDeltaX = select(INFINITE, 1.0 / abs(delta.x), delta.x != 0.0);
  let tDeltaY = select(INFINITE, 1.0 / abs(delta.y), delta.y != 0.0);
  var tCurrent = t0;
  for (var piece = 0u; piece < pieceCount; piece++) {
    let moveX = stepsX > 0u && (stepsY == 0u || tMaxX <= tMaxY);
    var tNext = t1;
    if (piece + 1u < pieceCount) {
      tNext = clamp(select(tMaxY, tMaxX, moveX), tCurrent, t1);
    }
    ${t?`if (slot + piece < CAPACITY) {
      keys[keysOffset + slot + piece] = u32(cy * COLUMNS + cx);
      contributions[contributionsOffset + slot + piece] = getPieceLength(a, b, tCurrent, tNext);
    }`:``}
    if (moveX) {
      cx += stepX;
      stepsX -= 1u;
      tMaxX += tDeltaX;
    } else {
      cy += stepY;
      stepsY -= 1u;
      tMaxY += tDeltaY;
    }
    tCurrent = tNext;
  }
  return pieceCount;
}
`}function x(e,t){return h(e,{id:t.id,operation:t.operation,variant:`count`,bindings:[{name:`positions`,view:t.constants.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.constants.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`counts`,view:t.counts,type:`u32`,access:`read_write`}],invocationCount:t.constants.positions.length,declarations:b(t.constants,!1),body:`counts[countsOffset + index] = walkSegment(index, 0u);`})}function S(e,t){return h(e,{id:t.id,operation:t.operation,variant:`emit`,bindings:[{name:`positions`,view:t.constants.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:t.constants.lineOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`starts`,view:t.starts,type:`u32`,access:`read`},{name:`keys`,view:t.keys,type:`u32`,access:`read_write`},{name:`contributions`,view:t.contributions,type:`f32`,access:`read_write`}],invocationCount:t.constants.positions.length,declarations:b(t.constants,!0),body:`walkSegment(index, starts[startsOffset + index]);`})}function C(e,t){let n=[{name:`scanTotal`,view:t.total,type:`u32`,access:`read`},{name:`overflowOut`,view:t.overflow,type:`u32`,access:`read_write`}];return t.totalRecords&&n.push({name:`totalOut`,view:t.totalRecords,type:`u32`,access:`read_write`}),h(e,{id:t.id,operation:t.operation,variant:`publish`,bindings:n,invocationCount:1,body:`let total = scanTotal[scanTotalOffset];
  overflowOut[overflowOutOffset] = select(0u, 1u, total > ${t.capacity}u);
  ${t.totalRecords?`totalOut[totalOutOffset] = total;`:``}`})}function w(e,t){return h(e,{id:t.id,operation:t.operation,variant:`density`,bindings:[{name:`lengths`,view:t.lengths,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`densities`,view:t.densities,type:`f32`,access:`read_write`}],invocationCount:t.columns*t.rows,declarations:`const COLUMNS: u32 = ${t.columns}u;
const AREA_RADIUS: f32 = ${p(t.areaRadius)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;`,body:`let row = index / COLUMNS;
  let width = parameters[parametersOffset + 2u];
  let height = parameters[parametersOffset + 3u];
  ${t.coordinateSystem===`spherical`?`let south = parameters[parametersOffset + 1u] + f32(row) * height;
  // Exact area of a longitude/latitude cell on the sphere.
  let area = AREA_RADIUS * AREA_RADIUS * width * DEGREES_TO_RADIANS *
    (sin((south + height) * DEGREES_TO_RADIANS) - sin(south * DEGREES_TO_RADIANS));`:`let area = width * height;`}
  densities[densitiesOffset + index] = lengths[lengthsOffset + index] / area;`})}var T=`GPULineDensity`,E=Object.freeze({...t,measure:`area`}),D=16777215,O=a({id:`line-density`,format:`float32`,wordLength:4,fields:[{name:`minimumX`,format:`float32`,wordOffset:0,defaultValue:0,dynamic:!0},{name:`minimumY`,format:`float32`,wordOffset:1,defaultValue:0,dynamic:!0},{name:`cellWidth`,format:`float32`,wordOffset:2,defaultValue:1,minimum:Number.MIN_VALUE,units:`position-units`,dynamic:!0},{name:`cellHeight`,format:`float32`,wordOffset:3,defaultValue:1,minimum:Number.MIN_VALUE,units:`position-units`,dynamic:!0}]});function k(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Line density target must hold 4 elements`);return t.set(i(O,{minimumX:e.minX,minimumY:e.minY,cellWidth:e.cellWidth,cellHeight:e.cellHeight})),t}var A=class{id;props;spatialContext;cellCount;coordinateSystem;radius;areaRadius;recordCapacity;constructor(t){this.id=t.id??`line-density`,this.props=t,this.spatialContext=e(this.id,t.spatialContext,E);let i=this.spatialContext.coordinateSpace===`longitude-latitude`;this.coordinateSystem=i?`spherical`:`planar`,this.radius=i?n(this.spatialContext):1,this.areaRadius=i?r(this.spatialContext):1,this.recordCapacity=t.recordCapacity??Math.max(1024,4*t.positions.length);let{id:a}=this;if(s(a,`columns`,t.columns),s(a,`rows`,t.rows),this.cellCount=t.columns*t.rows,l(a,`columns x rows`,this.cellCount,D),s(a,`recordCapacity`,this.recordCapacity),d(t.positions,[`float32x2`],`${a} positions`),t.positions.length<2)throw Error(`${a} needs two positions`);if(d(t.lineOffsets,[`uint32`],`${a} lineOffsets`),t.lineOffsets.length<2)throw Error(`${a} lineOffsets needs two rows`);if(d(t.parameters,[`float32`],`${a} parameters`),t.parameters.length<4)throw Error(`${a} parameters must hold 4 float32 values`);let{lengths:o,densities:u,status:f}=t.output,{overflow:p,requiredCount:m}=f;if(d(o,[`float32`],`${a} output.lengths`),o.length!==this.cellCount)throw Error(`${a} output.lengths must hold ${this.cellCount} rows`);if(u&&(d(u,[`float32`],`${a} output.densities`),u.length!==this.cellCount))throw Error(`${a} output.densities must hold ${this.cellCount} rows`);for(let[e,t]of[[`overflow`,p],[`requiredCount`,m]])if(t&&(d(t,[`uint32`],`${a} output.status.${e}`),t.length<1))throw Error(`${a} output.status.${e} must contain one uint32 row`);c(a,[o,u,p,m],[t.positions,t.lineOffsets,t.parameters])}getCommandNodes(e){let{props:t,id:n,cellCount:r,recordCapacity:i}=this,{output:a}=t;o(n,e,[t.positions,t.lineOffsets,t.parameters,a.lengths,a.densities,a.status.overflow,a.status.requiredCount]);let s=t.positions.length,c={positions:t.positions,lineOffsets:t.lineOffsets,columns:t.columns,rows:t.rows,coordinateSystem:this.coordinateSystem,radius:this.radius,capacity:i},l=u(e,`${n}-counts`,`uint32`,s),d=u(e,`${n}-starts`,`uint32`,s),p=u(e,`${n}-scan-total`,`uint32`,1),h=u(e,`${n}-keys`,`uint32`,i),v=u(e,`${n}-contributions`,`float32`,i);return[x(e,{id:`${n}-count`,operation:T,constants:c,parameters:t.parameters,counts:l}),...new f({id:`${n}-scan`,input:l,output:d,mode:`exclusive`}).getCommandNodes(e),..._(e,{id:n,operation:T,counts:l,offsets:d,rowCount:s,total:p}),m(e,{id:`${n}-fill-keys`,operation:T,view:h,type:`u32`,value:`${r}u`}),S(e,{id:`${n}-emit`,operation:T,constants:c,parameters:t.parameters,starts:d,keys:h,contributions:v}),C(e,{id:`${n}-publish`,operation:T,capacity:i,total:p,overflow:a.status.overflow,totalRecords:a.status.requiredCount}),...g(e,{id:`${n}-sum`,operation:T,segmentCount:r,segmentKeys:h,sumContributions:v,sums:a.lengths}),...a.densities?[w(e,{id:`${n}-density`,operation:T,columns:t.columns,rows:t.rows,coordinateSystem:this.coordinateSystem,areaRadius:this.areaRadius,parameters:t.parameters,lengths:a.lengths,densities:a.densities})]:[]]}};export{k as n,A as t};