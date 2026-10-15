import{t as e}from"./status-BwNARr9H.js";import{E as t,b as n}from"./analysis-frame-Blu0tYdw.js";import{_ as r,h as i,w as a,y as o}from"./resources-CwvhEOBe.js";import{t as s}from"./gpu-scan-BuGhTLDQ.js";import{s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as l}from"./lon-lat-ring-wgsl-C5XyUe9j.js";import{t as u}from"./geodesic-wgsl-DS0CJZ42.js";import{a as d,c as f,o as p,s as m}from"./line-segmentize-kernels-JWzYlmh-.js";import{a as h,c as g,o as _,s as v}from"./rectangle-clip-kernels-CVaEeFnm.js";function y(e,t={},n=new Float32Array(4)){let r=t.maximumSegmentDegrees??(e===`great-circle`?.1:0);if(!(r>=0))throw Error(`getGPULonLatPolygonPreparationParameterValues maximumSegmentDegrees must be >= 0`);if(n.length<4)throw Error(`getGPULonLatPolygonPreparationParameterValues target must hold 4 elements`);return n.set([r,0,0,0]),n}var b=`GPULonLatPolygonPreparation`;function x(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read`}}function S(e,t,n=`u32`){return{name:e,view:t,type:n,access:`read_write`}}function C(e,t){return c(e,{id:t.id,operation:b,variant:`initialize`,bindings:[S(`diagnostics`,t.diagnostics),S(`rectangle`,t.rectangle,`f32`),S(`stageFlags`,t.stageFlags)],invocationCount:1,body:`for (var stage = 0u; stage < 6u; stage++) { stageFlags[stageFlagsOffset + stage] = 0u; }
  diagnostics[diagnosticsOffset] = 0u;
  diagnostics[diagnosticsOffset + 1u] = 0u;
  rectangle[rectangleOffset] = -180.0;
  rectangle[rectangleOffset + 1u] = -90.0;
  rectangle[rectangleOffset + 2u] = 180.0;
  rectangle[rectangleOffset + 3u] = 90.0;`})}function ee(e,t){return c(e,{id:t.id,operation:b,variant:`classify`,bindings:[x(`positions`,t.positions,`f32`),x(`ringOffsets`,t.ringOffsets),x(`densePositions`,t.densePositions,`f32`),x(`denseOffsets`,t.denseOffsets),x(`parameters`,t.parameters,`f32`),S(`metadata`,t.metadata),S(`counts`,t.counts),S(`diagnostics`,t.diagnostics,`atomic<u32>`)],invocationCount:t.ringOffsets.length-1,declarations:`${l}\n${u}
fn position(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn finitePosition(value: vec2<f32>) -> bool {
  return all((bitcast<vec2<u32>>(value) & vec2<u32>(0x7f800000u)) != vec2<u32>(0x7f800000u));
}`,body:`let begin = ringOffsets[ringOffsetsOffset + index];
  var end = ringOffsets[ringOffsetsOffset + index + 1u];
  var flags = 0u;
  var invalid = false;
  var turn = 0i;
  var winding = 0.0;
  var pole = 0.0;
  var hasSecond = false;
  var hasThird = false;
  let denseBegin = denseOffsets[denseOffsetsOffset + index];
  let denseEnd = denseOffsets[denseOffsetsOffset + index + 1u];
  if (end > begin) {
    let first = position(begin);
    var second = first;
    if (end > begin + 1u && all(first == position(end - 1u))) { end--; }
    var minimum = first.x;
    var maximum = first.x;
    var longitude = first.x;
    for (var row = begin; row < end; row++) {
      let current = position(row);
      let next = position(select(row + 1u, begin, row + 1u >= end));
      if (!finitePosition(current) || abs(current.y) > 90.0) { flags |= 16u; continue; }
      if (any(current != first)) {
        if (!hasSecond) { second = current; hasSecond = true; }
        else if (any(current != second)) { hasThird = true; }
      }
      if (!finitePosition(next)) { continue; }
      if (lonLatIsCrossingEdge(current, next)) { flags |= 1u; }
      if (${t.edgeModel===`linear`?`lonLatIsAmbiguousLinearEdge`:`lonLatIsAmbiguousGreatCircleEdge`}(current, next)) { flags |= 4u; }
      let step = lonLatRingStep(current, next);
      winding += step;
      if (!lonLatIsPoleEdge(current, next)) { turn -= i32(round((next.x - current.x) / 360.0)); }
      longitude = next.x + 360.0 * f32(turn);
      minimum = min(minimum, longitude);
      maximum = max(maximum, longitude);
      let maximumStep = parameters[parametersOffset];
      let extent = max(abs(lonLatWrapLongitudeStep(next.x - current.x)), geodesicCentralAngle(current, next) * GEODESIC_RADIANS_TO_DEGREES);
      if (maximumStep > 0.0 && ceil(extent / maximumStep) > ${t.maximumPieceCount}.0) { flags |= 32u; }
    }
    if (!hasThird) { flags &= 16u; winding = 0.0; }
    if (hasThird && maximum - minimum > 360.0) { flags |= 8u; }
    invalid = (flags & 28u) != 0u ${t.antimeridian===`reject`?`|| (flags & 1u) != 0u`:``};
    if (lonLatIsPoleWinding(winding)) {
      var minimumLatitude = 90.0;
      var maximumLatitude = -90.0;
      for (var row = denseBegin; row < denseEnd; row++) {
        let latitude = densePositions[densePositionsOffset + 2u * row + 1u];
        minimumLatitude = min(minimumLatitude, latitude);
        maximumLatitude = max(maximumLatitude, latitude);
      }
      pole = sign(minimumLatitude + maximumLatitude) * 90.0;
      invalid = invalid || pole == 0.0 ${t.poleRings===`reject`?`|| true`:``};
      if (!invalid) { flags |= 2u; }
    }
  }
  metadata[metadataOffset + 4u * index] = flags;
  metadata[metadataOffset + 4u * index + 1u] = select(0u, 1u, invalid);
  metadata[metadataOffset + 4u * index + 2u] = bitcast<u32>(pole);
  metadata[metadataOffset + 4u * index + 3u] = bitcast<u32>(turn);
  counts[countsOffset + index] = select(0u, denseEnd - denseBegin + select(0u, 3u, (flags & 2u) != 0u), hasThird && !invalid);
  ${t.edgeModel===`great-circle`?`if (parameters[parametersOffset] == 0.0 || (flags & 32u) != 0u) {
    atomicStore(&diagnostics[diagnosticsOffset + 1u], 1u);
  }`:``}`})}function te(e,t){return c(e,{id:t.id,operation:b,variant:`feature-validity`,bindings:[x(`featureOffsets`,t.featureOffsets),x(`polygonOffsets`,t.polygonOffsets),x(`metadata`,t.metadata),S(`counts`,t.counts),S(`diagnostics`,t.diagnostics,`atomic<u32>`)],invocationCount:t.featureOffsets.length-1,body:`let begin = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];
  let end = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index + 1u]];
  var invalid = false;
  for (var ring = begin; ring < end; ring++) { invalid = invalid || metadata[metadataOffset + 4u * ring + 1u] != 0u; }
  if (invalid) {
    atomicAdd(&diagnostics[diagnosticsOffset], 1u);
    for (var ring = begin; ring < end; ring++) { counts[countsOffset + ring] = 0u; }
  } else {
    // An empty shell has no filled area; its holes cannot become standalone polygons.
    for (var polygon = featureOffsets[featureOffsetsOffset + index]; polygon < featureOffsets[featureOffsetsOffset + index + 1u]; polygon++) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (firstRing < lastRing && counts[countsOffset + firstRing] == 0u) {
        for (var ring = firstRing + 1u; ring < lastRing; ring++) { counts[countsOffset + ring] = 0u; }
      }
    }
  }`})}function w(e,t){return c(e,{id:t.id,operation:b,variant:`cap`,bindings:[x(`positions`,t.positions,`f32`),x(`turns`,t.turns,`i32`),x(`offsets`,t.offsets),x(`metadata`,t.metadata),x(`starts`,t.starts),x(`counts`,t.counts),S(`outputPositions`,t.outputPositions,`f32`),S(`outputTurns`,t.outputTurns,`i32`)],invocationCount:t.counts.length,declarations:`fn copyVertex(destination: u32, source: u32, latitude: f32, turn: i32) {
  if (destination >= ${t.outputPositions.length}u) { return; }
  outputPositions[outputPositionsOffset + 2u * destination] = positions[positionsOffset + 2u * source];
  outputPositions[outputPositionsOffset + 2u * destination + 1u] = latitude;
  outputTurns[outputTurnsOffset + destination] = turn;
}`,body:`if (counts[countsOffset + index] == 0u) { return; }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let start = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    copyVertex(start + row - begin, row, positions[positionsOffset + 2u * row + 1u], turns[turnsOffset + row]);
  }
  if ((metadata[metadataOffset + 4u * index] & 2u) != 0u) {
    let firstTurn = turns[turnsOffset + begin];
    let closureTurn = firstTurn + bitcast<i32>(metadata[metadataOffset + 4u * index + 3u]);
    let pole = bitcast<f32>(metadata[metadataOffset + 4u * index + 2u]);
    copyVertex(start + end - begin, begin, positions[positionsOffset + 2u * begin + 1u], closureTurn);
    copyVertex(start + end - begin + 1u, begin, pole, closureTurn);
    copyVertex(start + end - begin + 2u, begin, pole, firstTurn);
  }`})}function T(e,t){return c(e,{id:t.id,operation:b,variant:`ring-offsets`,bindings:[x(`starts`,t.starts),x(`counts`,t.counts),S(`offsets`,t.offsets),...t.stageFlags?[S(`stageFlags`,t.stageFlags)]:[]],invocationCount:t.counts.length+1,body:`var value = 0u;
  if (index < ${t.counts.length}u) { value = starts[startsOffset + index]; }
  else if (index > 0u) { value = starts[startsOffset + index - 1u] + counts[countsOffset + index - 1u]; }
  offsets[offsetsOffset + index] = min(value, ${t.capacity}u);
  ${t.stageFlags?`if (index == ${t.counts.length}u) { stageFlags[stageFlagsOffset + ${t.stage??0}u] = value; }`:``}`})}function ne(e,t){return c(e,{id:t.id,operation:b,variant:`normalize`,bindings:[x(`positions`,t.positions,`f32`),S(`turns`,t.turns,`i32`),x(`ringOffsets`,t.ringOffsets),x(`polygonOffsets`,t.polygonOffsets)],invocationCount:t.polygonOffsets.length-1,declarations:`fn longitude(row: u32) -> f32 { return positions[positionsOffset + 2u * row] + 360.0 * f32(turns[turnsOffset + row]); }`,body:`let shell = polygonOffsets[polygonOffsetsOffset + index];
  let ringEnd = polygonOffsets[polygonOffsetsOffset + index + 1u];
  if (shell == ringEnd) { return; }
  let begin = ringOffsets[ringOffsetsOffset + shell];
  let end = ringOffsets[ringOffsetsOffset + shell + 1u];
  if (begin == end) { return; }
  var minimum = longitude(begin);
  var maximum = minimum;
  for (var row = begin + 1u; row < end; row++) { minimum = min(minimum, longitude(row)); maximum = max(maximum, longitude(row)); }
  let shellTurns = i32(floor((minimum + 180.0) / 360.0));
  let centre = (minimum + maximum) * 0.5;
  for (var ring = shell; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (first == last) { continue; }
    var shift = -shellTurns;
    if (ring != shell) { shift += i32(floor((centre - longitude(first)) / 360.0 + 0.5)); }
    for (var row = first; row < last; row++) { turns[turnsOffset + row] += shift; }
  }`})}function re(e,t){return c(e,{id:t.id,operation:b,variant:`topology`,bindings:[x(`featureOffsets`,t.featureOffsets),x(`polygonOffsets`,t.polygonOffsets),S(`outputFeatureOffsets`,t.outputFeatureOffsets),S(`outputPolygonOffsets`,t.outputPolygonOffsets),S(`ringMap`,t.ringMap)],invocationCount:t.featureOffsets.length,body:`let first = featureOffsets[featureOffsetsOffset + index];
  outputFeatureOffsets[outputFeatureOffsetsOffset + index] = 2u * first;
  if (index == ${t.featureOffsets.length-1}u) {
    outputPolygonOffsets[outputPolygonOffsetsOffset + 2u * first] = 2u * polygonOffsets[polygonOffsetsOffset + first];
    return;
  }
  let last = featureOffsets[featureOffsetsOffset + index + 1u];
  let firstRing = polygonOffsets[polygonOffsetsOffset + first];
  let lastRing = polygonOffsets[polygonOffsetsOffset + last];
  for (var copy = 0u; copy < 2u; copy++) {
    for (var polygon = first; polygon < last; polygon++) {
      let targetPolygon = 2u * first + copy * (last - first) + polygon - first;
      let sourceRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let targetRing = 2u * firstRing + copy * (lastRing - firstRing) + sourceRing - firstRing;
      outputPolygonOffsets[outputPolygonOffsetsOffset + targetPolygon] = targetRing;
      for (var ring = sourceRing; ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u]; ring++) {
        ringMap[ringMapOffset + targetRing + ring - sourceRing] = 2u * ring + copy;
      }
    }
  }`})}var E=`
fn getPosition(row: u32) -> vec2<f32> {
  var longitude = positions[positionsOffset + 2u * row];
  let turn = turns[turnsOffset + row];
  if (turn != 0i) { longitude += 360.0 * f32(turn); }
  return vec2<f32>(longitude, positions[positionsOffset + 2u * row + 1u]);
}
fn ringEnd(begin: u32, last: u32) -> u32 {
  var end = last;
  if (end > begin + 1u && all(getPosition(begin) == getPosition(end - 1u))) { end--; }
  return end;
}
fn keepRow(row: u32, begin: u32) -> bool {
  if (row == begin) { return true; }
  return any(getPosition(row) != getPosition(row - 1u));
}`;function D(e,t){return c(e,{id:t.id,operation:b,variant:`merge-count-${t.copy}`,bindings:[x(`positions`,t.positions,`f32`),x(`turns`,t.turns,`i32`),x(`offsets`,t.offsets),x(`ringMap`,t.ringMap),S(`counts`,t.counts)],invocationCount:t.counts.length,declarations:E,body:`let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${t.copy}u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var count = 0u;
  var second = vec2<f32>(0.0);
  var hasSecond = false;
  var hasThird = false;
  for (var row = begin; row < end; row++) {
    let current = getPosition(row);
    if (any(current != getPosition(begin))) {
      if (!hasSecond) { second = current; hasSecond = true; }
      else if (any(current != second)) { hasThird = true; }
    }
    if (keepRow(row, begin)) { count++; }
  }
  counts[countsOffset + index] = select(0u, count, hasThird);`})}function O(e,t){return c(e,{id:t.id,operation:b,variant:`merge-emit-${t.copy}`,bindings:[x(`positions`,t.positions,`f32`),x(`turns`,t.turns,`i32`),x(`offsets`,t.offsets),x(`ringMap`,t.ringMap),x(`counts`,t.counts),x(`starts`,t.starts),S(`output`,t.output,`f32`)],invocationCount:t.counts.length,declarations:E,body:`let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${t.copy}u || counts[countsOffset + index] == 0u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var destination = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    if (keepRow(row, begin)) {
      if (destination < ${t.output.length}u) {
        let position = getPosition(row);
        output[outputOffset + 2u * destination] = position.x;
        output[outputOffset + 2u * destination + 1u] = position.y;
      }
      destination++;
    }
  }`})}function k(e,t){let n=[x(`counts`,t.counts),x(`starts`,t.starts),S(`stageFlags`,t.stageFlags),x(`metadata`,t.metadata),x(`diagnostics`,t.diagnostics)];for(let e of[`ringFlags`,`invalidCount`,`approximation`])t[e]&&n.push(S(e,t[e]));return c(e,{id:t.id,operation:b,variant:`status`,bindings:n,invocationCount:Math.max(1,t.metadata.length/4),body:`${t.ringFlags?`ringFlags[ringFlagsOffset + index] = metadata[metadataOffset + 4u * index];`:``}
  if (index == 0u) {
    let last = ${t.counts.length-1}u;
    let total = starts[startsOffset + last] + counts[countsOffset + last];
    let denseTotal = stageFlags[stageFlagsOffset];
    var intermediateTotal = 0u;
    for (var stage = 0u; stage < 6u; stage++) {
      intermediateTotal = max(intermediateTotal, stageFlags[stageFlagsOffset + stage]);
    }
    let requiredCount = select(total, 8u * (denseTotal + ${3*t.metadata.length/4}u), intermediateTotal > ${t.capacity}u);
    stageFlags[stageFlagsOffset] = intermediateTotal;
    stageFlags[stageFlagsOffset + 1u] = 0u;
    stageFlags[stageFlagsOffset + 2u] = 0u;
    stageFlags[stageFlagsOffset + 3u] = total;
    stageFlags[stageFlagsOffset + 4u] = requiredCount;
    ${t.invalidCount?`invalidCount[invalidCountOffset] = diagnostics[diagnosticsOffset];`:``}
    ${t.approximation?`approximation[approximationOffset] = diagnostics[diagnosticsOffset + 1u];`:``}
  }`})}var A=`GPULonLatPolygonPreparation`,j=class{id;props;spatialContext;inputContext;constructor(n){this.id=n.id??`lon-lat-polygon-preparation`,this.props=n;let{id:r}=this;this.inputContext=t(r,n.spatialContext,{coordinateSpaces:[`longitude-latitude`],metrics:[`none`,`great-circle`],edgeModels:[`linear`,`great-circle`]}),this.spatialContext={...this.inputContext,edgeModel:`linear`};let a=n.maximumPieceCount??1024;if(!Number.isInteger(a)||a<1||a>65536)throw Error(`${r} maximumPieceCount must be an integer in [1, 65536]`);if(n.antimeridian!==void 0&&![`split`,`reject`].includes(n.antimeridian))throw Error(`${r} antimeridian must be split or reject`);if(n.poleRings!==void 0&&![`cap`,`reject`].includes(n.poleRings))throw Error(`${r} poleRings must be cap or reject`);if(o(n.positions,[`float32x2`],`${r} positions`),n.positions.length<1)throw Error(`${r} needs at least one position`);for(let e of[`featureOffsets`,`polygonOffsets`,`ringOffsets`]){if(o(n[e],[`uint32`],`${r} ${e}`),n[e].length<2)throw Error(`${r} ${e} needs two rows`);o(n.output[e],[`uint32`],`${r} output.${e}`);let t=e===`featureOffsets`?n[e].length:2*n[e].length-1;if(n.output[e].length!==t)throw Error(`${r} output.${e} must hold ${t} rows`)}if(n.output.ringFlags&&(o(n.output.ringFlags,[`uint32`],`${r} output.ringFlags`),n.output.ringFlags.length!==n.ringOffsets.length-1))throw Error(`${r} output.ringFlags must hold ${n.ringOffsets.length-1} rows`);if(o(n.output.positions,[`float32x2`],`${r} output.positions`),o(n.parameters,[`float32`],`${r} parameters`),n.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);e(r,n.output.status),i(r,this.getOutputViews(),this.getInputViews())}getCommandNodes(e){let{id:t,props:i}=this,{output:o}=i;r(t,e,[...this.getInputViews(),...this.getOutputViews()]);let c=i.maximumPieceCount??1024,l=i.ringOffsets.length-1,u=o.positions.length,y=(n,r)=>a(e,`${t}-${n}`,`uint32`,r),b=(n,r)=>a(e,`${t}-${n}`,`float32x2`,r),x=(n,r)=>a(e,`${t}-${n}`,`sint32`,r),S=[],E=(n,r,i)=>{S.push(...new s({id:`${t}-${n}`,input:r,output:i,mode:`exclusive`}).getCommandNodes(e))},j=y(`stage-totals`,6),M=e.createDataView(j.buffer,{format:`uint32`,length:1}),N=y(`diagnostics`,2),ie=e.createDataView(N.buffer,{format:`uint32`,length:1,byteOffset:Uint32Array.BYTES_PER_ELEMENT}),P=a(e,`${t}-rectangle`,`float32`,4),F=y(`metadata`,4*l),I=x(`row-turns`,i.positions.length),L=y(`dense-counts`,i.positions.length),R=y(`dense-starts`,i.positions.length),z=b(`dense-positions`,u),B=x(`dense-turns`,u),V=y(`dense-offsets`,l+1),H=n(this.inputContext),U=H===`linear`?`linear-degrees`:`great-circle`,W={operation:A,positions:i.positions,lineOffsets:i.ringOffsets,coordinateSystem:`spherical`,radius:1,closedLines:!0,polygonEdges:!0};S.push(C(e,{id:`${t}-initialize`,diagnostics:N,rectangle:P,stageFlags:j}),p(e,{...W,id:`${t}-prefix`,rowTurns:I}),m(e,{...W,id:`${t}-dense-count`,parameters:i.parameters,maximumPieceCount:c,pieceCountModel:`degree-step`,approximation:H===`great-circle`?ie:void 0,interpolation:U,counts:L})),E(`dense-scan`,L,R),S.push(f(e,{...W,id:`${t}-dense-emit`,counts:L,starts:R,rowTurns:I,interpolation:U,outputPositions:z,outputTurns:B}),d(e,{id:`${t}-dense-offsets-publish`,operation:A,lineOffsets:i.ringOffsets,lineCount:l,rowCount:i.positions.length,counts:L,starts:R,outputLineOffsets:V,capacity:u,count:y(`dense-count`,1),overflow:y(`dense-overflow`,1),requiredCount:M}));let G=y(`cap-counts`,l),K=y(`cap-starts`,l),q=b(`cap-positions`,u),J=x(`cap-turns`,u),Y=y(`cap-offsets`,l+1);S.push(ee(e,{id:`${t}-classify`,positions:i.positions,ringOffsets:i.ringOffsets,densePositions:z,denseOffsets:V,parameters:i.parameters,metadata:F,counts:G,diagnostics:N,edgeModel:H,maximumPieceCount:c,antimeridian:i.antimeridian??`split`,poleRings:i.poleRings??`cap`}),te(e,{id:`${t}-features`,featureOffsets:i.featureOffsets,polygonOffsets:i.polygonOffsets,metadata:F,counts:G,diagnostics:N})),E(`cap-scan`,G,K),S.push(w(e,{id:`${t}-cap`,positions:z,turns:B,offsets:V,metadata:F,starts:K,counts:G,outputPositions:q,outputTurns:J}),T(e,{id:`${t}-cap-offsets-publish`,starts:K,counts:G,offsets:Y,capacity:u,stageFlags:j,stage:1}),ne(e,{id:`${t}-normalize`,positions:q,turns:J,ringOffsets:Y,polygonOffsets:i.polygonOffsets}));let X=[];for(let n=0;n<2;n++){let r=q,i=J,a=Y,o=[b(`clip-${n}-positions-a`,u),b(`clip-${n}-positions-b`,u)],s=[x(`clip-${n}-turns-a`,u),x(`clip-${n}-turns-b`,u)],c=[y(`clip-${n}-offsets-a`,l+1),y(`clip-${n}-offsets-b`,l+1)];for(let l=0;l<(u===0?0:2);l++){let d=o[l%2],f=s[l%2],p=c[l%2],m=y(`clip-${n}-${l}-counts`,r.length),h=y(`clip-${n}-${l}-starts`,r.length),b={operation:A,boundary:l,positions:r,positionTurns:i,longitudeCopy:l===0?n:0,ringOffsets:a,parameters:P};S.push(_(e,{...b,id:`${t}-clip-${n}-${l}-count`,counts:m})),E(`clip-${n}-${l}-scan`,m,h),S.push(v(e,{...b,id:`${t}-clip-${n}-${l}-emit`,starts:h,outputPositions:d,outputTurns:f}),g(e,{id:`${t}-clip-${n}-${l}-offsets`,operation:A,capacity:u,inputOffsets:a,starts:h,counts:m,outputOffsets:p,stageFlags:j,stage:2+n*2+l})),r=d,i=f,a=p}X.push({positions:r,turns:i,offsets:a})}let Z=y(`ring-map`,2*l),Q=y(`merge-counts`,2*l),$=y(`merge-starts`,2*l);return S.push(re(e,{id:`${t}-topology`,featureOffsets:i.featureOffsets,polygonOffsets:i.polygonOffsets,outputFeatureOffsets:o.featureOffsets,outputPolygonOffsets:o.polygonOffsets,ringMap:Z})),X.forEach((n,r)=>S.push(D(e,{...n,id:`${t}-merge-${r}-count`,ringMap:Z,counts:Q,copy:r}))),E(`merge-scan`,Q,$),u>0&&X.forEach((n,r)=>S.push(O(e,{...n,id:`${t}-merge-${r}-emit`,ringMap:Z,counts:Q,starts:$,output:o.positions,copy:r}))),S.push(T(e,{id:`${t}-output-offsets`,starts:$,counts:Q,offsets:o.ringOffsets,capacity:o.positions.length}),k(e,{id:`${t}-status`,counts:Q,starts:$,stageFlags:j,capacity:u,metadata:F,diagnostics:N,ringFlags:o.ringFlags,invalidCount:o.status.invalidCount,approximation:o.status.approximation}),h(e,{id:`${t}-publish`,operation:A,capacity:o.positions.length,ringCount:2*l,stageFlags:j,count:o.status.count??y(`output-count`,1),overflow:o.status.overflow,requiredCount:o.status.requiredCount,requiredCountIndex:4})),S}getInputViews(){let{props:e}=this;return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.parameters]}getOutputViews(){let{output:e}=this.props;return[e.positions,e.featureOffsets,e.polygonOffsets,e.ringOffsets,e.ringFlags,...Object.values(e.status)].filter(e=>!!e)}};export{y as n,j as t};