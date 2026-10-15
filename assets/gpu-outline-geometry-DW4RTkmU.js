import{D as e,E as t,w as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,d as i,h as a,y as o}from"./resources-CwvhEOBe.js";import{l as s,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var l=`GPUOutlineGeometry`,u=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({planar:Object.freeze([`native`]),"longitude-latitude":Object.freeze([`great-circle`])}),measure:`distance`});function d(t,n=new Float32Array(4)){if(n.length<4)throw Error(`Outline geometry target must hold 4 elements`);if(!Number.isFinite(t.distance)||t.distance<0)throw Error(`Outline geometry distance must be a non-negative finite number`);return e(`GPUOutlineGeometry`,t.spatialContext,t.distance),n.set([t.distance,0,0,0]),n}function f(e){return 3*(e+2)}var p=class{id;props;spatialContext;radius;joinSegments;verticesPerInput;boundaryOffsets;constructor(e){this.id=e.id??`outline-geometry`,this.props=e,this.spatialContext=t(this.id,e.spatialContext,u),this.radius=n(this.spatialContext),this.joinSegments=e.joinSegments??16,this.verticesPerInput=f(this.joinSegments);let{id:r}=this;if(this.boundaryOffsets=e.geometryType===`lines`?e.lineOffsets:e.geometryType===`polygons`?e.ringOffsets:void 0,![`points`,`lines`,`polygons`].includes(e.geometryType))throw Error(`${r} geometryType must be 'points', 'lines' or 'polygons'`);if(i(r,`joinSegments`,this.joinSegments,3),o(e.positions,[`float32x2`],`${r} positions`),e.positions.length<1)throw Error(`${r} needs at least one position`);if(e.geometryType!==`points`){let t=e.geometryType===`lines`?`lineOffsets`:`ringOffsets`;if(!this.boundaryOffsets)throw Error(`${r} ${t} is required`);if(o(this.boundaryOffsets,[`uint32`],`${r} ${t}`),this.boundaryOffsets.length<2)throw Error(`${r} ${t} needs two rows`);e.geometryType===`polygons`&&(o(e.featureOffsets,[`uint32`],`${r} featureOffsets`),o(e.polygonOffsets,[`uint32`],`${r} polygonOffsets`))}if(o(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);o(e.output.positions,[`float32x2`],`${r} output.positions`);let s=e.positions.length*this.verticesPerInput;if(e.output.positions.length!==s)throw Error(`${r} output.positions must hold ${s} rows`);if(s>2147483647)throw Error(`${r} output is too large`);a(r,[e.output.positions],[e.positions,this.boundaryOffsets,e.parameters])}getCommandNodes(e){let{props:t,id:n,joinSegments:i}=this;r(n,e,[t.positions,this.boundaryOffsets,t.parameters,t.output.positions]);let a=t.geometryType!==`points`,o=[{name:`positions`,view:t.positions,type:`f32`,access:`read`}];a&&this.boundaryOffsets&&o.push({name:`lineOffsets`,view:this.boundaryOffsets,type:`u32`,access:`read`}),o.push({name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`outputPositions`,view:t.output.positions,type:`f32`,access:`read_write`});let u=this.spatialContext.coordinateSpace===`longitude-latitude`;return[c(e,{id:`${n}-generate`,operation:l,variant:t.geometryType,bindings:o,invocationCount:t.positions.length,declarations:`
const LINE_COUNT: u32 = ${this.boundaryOffsets?this.boundaryOffsets.length-1:0}u;
const ROW_COUNT: u32 = ${t.positions.length}u;
const JOIN_SEGMENTS: u32 = ${i}u;
const VERTICES_PER_INPUT: u32 = ${this.verticesPerInput}u;
const RADIUS: f32 = ${s(this.radius)};
const DEGREES_PER_RADIAN: f32 = 57.29577951308232;
const TAU: f32 = 6.283185307179586;

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn writeVertex(slot: u32, base: u32, p: vec2<f32>) {
  let at = outputPositionsOffset + 2u * (base + slot);
  outputPositions[at] = p.x;
  outputPositions[at + 1u] = p.y;
}

// Moves p by distance d along the metric direction (east, north).
fn offsetPoint(p: vec2<f32>, direction: vec2<f32>, d: f32) -> vec2<f32> {
  ${u?`let latitudeScale = d / RADIUS * DEGREES_PER_RADIAN;
  let longitudeScale = latitudeScale / max(cos(radians(p.y)), 1e-6);
  return p + vec2<f32>(direction.x * longitudeScale, direction.y * latitudeScale);`:`return p + direction * d;`}
}

// Unit metric direction of the segment a to b, or zero for a zero-length segment.
fn getDirection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  ${u?`let middleLatitude = radians(0.5 * (a.y + b.y));
  let metric = vec2<f32>((b.x - a.x) * cos(middleLatitude), b.y - a.y);`:`let metric = b - a;`}
  let metricLength = length(metric);
  if (metricLength <= 0.0) {
    return vec2<f32>(0.0);
  }
  return metric / metricLength;
}
${a?m:``}`,body:`let distance = max(parameters[parametersOffset], 0.0);
  let base = index * VERTICES_PER_INPUT;
  let center = getPosition(index);
  // Round join and cap disc: JOIN_SEGMENTS counter-clockwise triangles around the vertex.
  // Each rim point is shared by two triangles, so it is computed once (one sin/cos per segment).
  var rimA = offsetPoint(center, vec2<f32>(1.0, 0.0), distance);
  for (var k = 0u; k < JOIN_SEGMENTS; k++) {
    let angleB = TAU * f32(k + 1u) / f32(JOIN_SEGMENTS);
    let rimB = offsetPoint(center, vec2<f32>(cos(angleB), sin(angleB)), distance);
    writeVertex(3u * k, base, center);
    writeVertex(3u * k + 1u, base, rimA);
    writeVertex(3u * k + 2u, base, rimB);
    rimA = rimB;
  }
  // Rectangle toward the next vertex, or a degenerate pair of triangles.
  var nextRow = ROW_COUNT;
  ${a?h(t.geometryType):``}
  var quad = array<vec2<f32>, 6>(center, center, center, center, center, center);
  if (nextRow < ROW_COUNT) {
    let next = getPosition(nextRow);
    let direction = getDirection(center, next);
    if (dot(direction, direction) > 0.0) {
      let normal = vec2<f32>(-direction.y, direction.x);
      let aLeft = offsetPoint(center, normal, distance);
      let aRight = offsetPoint(center, -normal, distance);
      let bRight = offsetPoint(next, -normal, distance);
      let bLeft = offsetPoint(next, normal, distance);
      quad = array<vec2<f32>, 6>(aLeft, aRight, bRight, aLeft, bRight, bLeft);
    }
  }
  for (var vertex = 0u; vertex < 6u; vertex++) {
    writeVertex(3u * JOIN_SEGMENTS + vertex, base, quad[vertex]);
  }`})]}},m=`
const NO_LINE: u32 = 0xffffffffu;

fn findLine(row: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > LINE_COUNT) {
    return NO_LINE;
  }
  return low - 1u;
}
`;function h(e){return`let line = findLine(index);
  if (line != NO_LINE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], ROW_COUNT);
    if (index + 1u < lineEnd) {
      nextRow = index + 1u;
    }${e===`polygons`?` else if (lineEnd - lineStart >= 2u) {
      nextRow = lineStart;
    }`:``}
  }`}export{d as n,f as r,p as t};