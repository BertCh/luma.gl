import{E as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{u as o}from"./wgsl-common-BNRVzgd8.js";import{t as s}from"./exact-orientation-wgsl-D-n1xGZJ.js";import{t as c}from"./gpu-geometry-measures-nkjV4Jk0.js";import{t as l}from"./gpu-group-convex-hull-B1eInSxa.js";function u(e){return[{name:`positions`,view:e.positions,type:`f32`,access:`read`},{name:`ringOffsets`,view:e.ringOffsets,type:`u32`,access:`read`},{name:`featureOffsets`,view:e.featureOffsets,type:`u32`,access:`read`},{name:`polygonOffsets`,view:e.polygonOffsets,type:`u32`,access:`read`}]}function d(e){return`
const ROW_COUNT: u32 = ${e}u;
fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`}function f(e){return`${d(e.positions.length)}
const RING_COUNT: u32 = ${e.ringOffsets.length-1}u;
const POLYGON_COUNT: u32 = ${e.polygonOffsets.length-1}u;

// Feature -> polygons -> rings: a feature owns the rings of its polygons.
fn getRingStart(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}

fn getRingEnd(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}

// Ring role by position: ring 'ring' is a shell when it starts a polygon (binary search of
// polygonOffsets). A ring outside the polygonOffsets range keeps the first-ring rule.
fn isShellRing(ring: u32, isFirstRing: bool) -> bool {
  if (ring < polygonOffsets[polygonOffsetsOffset] || ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    return isFirstRing;
  }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] < ring) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return polygonOffsets[polygonOffsetsOffset + low] == ring;
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}
`}function p(e,t){return a(e,{id:t.id,operation:t.operation,variant:`moments`,bindings:[...u(t),{name:`centroids`,view:t.centroids,type:`f32`,access:`read`},{name:`moments`,view:t.moments,type:`f32`,access:`read_write`}],invocationCount:t.featureCount,declarations:`${f(t)}
${o}

// accumulator = (sum, compensation).
fn addNeumaier(accumulator: ptr<function, vec2<f32>>, value: f32) {
  let sum = (*accumulator).x;
  let total = sum + value;
  var compensation = (*accumulator).y;
  if (abs(sum) >= abs(value)) {
    compensation += (sum - total) + value;
  } else {
    compensation += (value - total) + sum;
  }
  *accumulator = vec2<f32>(total, compensation);
}`,body:`let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let centroid = vec2<f32>(centroids[centroidsOffset + 2u * index], centroids[centroidsOffset + 2u * index + 1u]);
  var totalXX = vec2<f32>(0.0);
  var totalYY = vec2<f32>(0.0);
  var totalXY = vec2<f32>(0.0);
  var totalTwiceArea = 0.0;
  var firstArea = 0.0;
  if (!isNanBits(centroid.x) && !isNanBits(centroid.y)) {
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      var twiceArea = vec2<f32>(0.0);
      var ringXX = vec2<f32>(0.0);
      var ringYY = vec2<f32>(0.0);
      var ringXY = vec2<f32>(0.0);
      if (rowEnd > rowStart + 2u) {
        var previous = getPosition(rowEnd - 1u) - centroid;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - centroid;
          let cross = previous.x * current.y - current.x * previous.y;
          addNeumaier(&twiceArea, cross);
          addNeumaier(&ringXX, cross * (previous.x * previous.x + previous.x * current.x + current.x * current.x));
          addNeumaier(&ringYY, cross * (previous.y * previous.y + previous.y * current.y + current.y * current.y));
          addNeumaier(&ringXY, cross * (previous.x * current.y + 2.0 * previous.x * previous.y + 2.0 * current.x * current.y + current.x * previous.y));
          previous = current;
        }
      }
      let ringTwiceArea = twiceArea.x + twiceArea.y;
      if (ring == ringStart) {
        firstArea = 0.5 * ringTwiceArea;
      }
      let factor = select(-1.0, 1.0, isShellRing(ring, ring == ringStart)) * sign(ringTwiceArea);
      totalTwiceArea += factor * ringTwiceArea;
      addNeumaier(&totalXX, factor * (ringXX.x + ringXX.y) / 12.0);
      addNeumaier(&totalYY, factor * (ringYY.x + ringYY.y) / 12.0);
      addNeumaier(&totalXY, factor * (ringXY.x + ringXY.y) / 24.0);
    }
  }
  // Shell rings count positive and holes negative whatever their winding, so the total is
  // non-negative; the flip only guards a zero-area feature.
  let orientationSign = select(1.0, -1.0, totalTwiceArea < 0.0);
  moments[momentsOffset + 4u * index] = orientationSign * (totalXX.x + totalXX.y);
  moments[momentsOffset + 4u * index + 1u] = orientationSign * (totalYY.x + totalYY.y);
  moments[momentsOffset + 4u * index + 2u] = orientationSign * (totalXY.x + totalXY.y);
  moments[momentsOffset + 4u * index + 3u] = firstArea;`})}function m(e,t){return a(e,{id:t.id,operation:t.operation,variant:`convexity`,bindings:[...u(t),{name:`areas`,view:t.areas,type:`f32`,access:`read`},{name:`convexities`,view:t.convexities,type:`f32`,access:`read_write`}],invocationCount:t.featureCount,declarations:`${f(t)}
${s}`,body:`let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  let area = areas[areasOffset + index];
  var hullArea = 0.0;
  if (rowEnd - rowStart >= 3u && area > 0.0) {
    let origin = getPosition(rowStart);
    // Lowest point, leftmost among ties, lowest row among identical points.
    var startPoint = vec2<f32>(0.0);
    var startRaw = origin;
    for (var row = rowStart; row < rowEnd; row++) {
      let raw = getPosition(row);
      let p = raw - origin;
      if (row == rowStart || p.y < startPoint.y || (p.y == startPoint.y && p.x < startPoint.x)) {
        startPoint = p;
        startRaw = raw;
      }
    }
    var current = startPoint;
    var currentRaw = startRaw;
    var twiceArea = 0.0;
    for (var step = 0u; step < rowEnd - rowStart; step++) {
      var best = current;
      var bestRaw = currentRaw;
      var hasBest = false;
      for (var row = rowStart; row < rowEnd; row++) {
        let qRaw = getPosition(row);
        let q = qRaw - origin;
        if (q.x == current.x && q.y == current.y) {
          continue;
        }
        if (!hasBest) {
          best = q;
          bestRaw = qRaw;
          hasBest = true;
          continue;
        }
        let toBest = best - current;
        let toQ = q - current;
        let turn = orientSign(currentRaw, bestRaw, qRaw);
        // q lies clockwise of best (all others must be left of the hull edge), or farther on a tie.
        if (turn < 0 || (turn == 0 && dot(toQ, toQ) > dot(toBest, toBest))) {
          best = q;
          bestRaw = qRaw;
        }
      }
      if (!hasBest) {
        break;
      }
      twiceArea += current.x * best.y - best.x * current.y;
      current = best;
      currentRaw = bestRaw;
      if (current.x == startPoint.x && current.y == startPoint.y) {
        break;
      }
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`})}function h(e,t){let n=[{name:`areas`,view:t.areas,type:`f32`,access:`read`},{name:`perimeters`,view:t.perimeters,type:`f32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`}];return t.polsbyPopper&&n.push({name:`polsbyPopper`,view:t.polsbyPopper,type:`f32`,access:`read_write`}),t.schwartzberg&&n.push({name:`schwartzberg`,view:t.schwartzberg,type:`f32`,access:`read_write`}),t.sliver&&n.push({name:`sliver`,view:t.sliver,type:`u32`,access:`read_write`}),a(e,{id:t.id,operation:t.operation,variant:`compactness`,bindings:n,invocationCount:t.featureCount,declarations:`fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,body:`let area = areas[areasOffset + index];
  let perimeter = perimeters[perimetersOffset + index];
  let isValid = area > 0.0 && perimeter > 0.0;
  // Divide by the perimeter twice so a 1e6 m perimeter does not square into f32 trouble.
  let ratio = 4.0 * 3.14159265358979 * area / perimeter / perimeter;
  ${t.polsbyPopper?`polsbyPopper[polsbyPopperOffset + index] = select(getNan(index), ratio, isValid);`:``}
  ${t.schwartzberg?`schwartzberg[schwartzbergOffset + index] = select(getNan(index), sqrt(ratio), isValid);`:``}
  ${t.sliver?`sliver[sliverOffset + index] = select(0u, 1u, perimeter > 0.0 && (area <= 0.0 || ratio < parameters[parametersOffset]));`:``}`})}function g(e,t){let n=[{name:`moments`,view:t.moments,type:`f32`,access:`read`}];return t.elongation&&n.push({name:`elongation`,view:t.elongation,type:`f32`,access:`read_write`}),t.orientation&&n.push({name:`orientation`,view:t.orientation,type:`f32`,access:`read_write`}),t.clockwise&&n.push({name:`clockwise`,view:t.clockwise,type:`u32`,access:`read_write`}),a(e,{id:t.id,operation:t.operation,variant:`orientation`,bindings:n,invocationCount:t.featureCount,declarations:`fn getNan(seed: u32) -> f32 {
  // A runtime operand keeps the compiler from folding a constant NaN.
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`,body:`let xx = moments[momentsOffset + 4u * index];
  let yy = moments[momentsOffset + 4u * index + 1u];
  let xy = moments[momentsOffset + 4u * index + 2u];
  let firstArea = moments[momentsOffset + 4u * index + 3u];
  let mean = 0.5 * (xx + yy);
  let radius = sqrt(0.25 * (xx - yy) * (xx - yy) + xy * xy);
  let major = mean + radius;
  let minor = max(mean - radius, 0.0);
  let isValid = major > 0.0;
  ${t.elongation?`elongation[elongationOffset + index] = select(getNan(index), 1.0 - sqrt(minor / major), isValid);`:``}
  ${t.orientation?`orientation[orientationOffset + index] = select(getNan(index), select(0.0, 0.5 * atan2(2.0 * xy, xx - yy), (xy != 0.0) || (xx != yy)), isValid);`:``}
  ${t.clockwise?`clockwise[clockwiseOffset + index] = select(0u, 1u, firstArea < 0.0);`:``}`})}var _=2**24-1;function v(e,t){return e<_&&t<_}function y(e,t){let{id:n,operation:i,featureCount:o}=t,s=t.positions.length,c=r(e,`${n}-labels`,`uint32`,s),p=r(e,`${n}-hull-indices`,`uint32`,s),m=r(e,`${n}-hull-offsets`,`uint32`,o+1),h=r(e,`${n}-hull-counts`,`uint32`,o),g=r(e,`${n}-hull-overflow`,`uint32`,1);return[a(e,{id:`${n}-labels`,operation:i,variant:`convexity-labels`,bindings:[...u(t),{name:`labels`,view:c,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${f(t)}
const FEATURE_COUNT: u32 = ${o}u;
fn getFeatureRowStart(feature: u32) -> u32 { return getRingRowStart(getRingStart(feature)); }
fn getFeatureRowEnd(feature: u32) -> u32 { return max(getRingRowStart(getRingEnd(feature)), getFeatureRowStart(feature)); }`,body:`// Last feature whose first row is at or before this row (skips empty features).
  var low = 0u;
  var high = FEATURE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (getFeatureRowStart(middle) <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let owned = getFeatureRowStart(low) <= index && index < getFeatureRowEnd(low);
  labels[labelsOffset + index] = select(FEATURE_COUNT, low, owned);`}),...new l({id:`${n}-hull`,positions:t.positions,labels:c,groupCount:o,maximumVerticesPerGroup:Math.max(3,s),vertexCapacity:s,output:{vertexIndices:p,offsets:m,counts:h,status:{overflow:g}}}).getCommandNodes(e),a(e,{id:`${n}-hull-area`,operation:i,variant:`convexity-area`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`areas`,view:t.areas,type:`f32`,access:`read`},{name:`hullIndices`,view:p,type:`u32`,access:`read`},{name:`hullOffsets`,view:m,type:`u32`,access:`read`},{name:`hullCounts`,view:h,type:`u32`,access:`read`},{name:`convexities`,view:t.convexities,type:`f32`,access:`read_write`}],invocationCount:o,declarations:d(s),body:`let area = areas[areasOffset + index];
  let hullStart = hullOffsets[hullOffsetsOffset + index];
  let hullCount = hullCounts[hullCountsOffset + index];
  var hullArea = 0.0;
  if (area > 0.0 && hullCount >= 3u) {
    // Shoelace over the counter-clockwise hull, relative to its first vertex for f32 accuracy.
    let origin = getPosition(hullIndices[hullIndicesOffset + hullStart]);
    var previous = getPosition(hullIndices[hullIndicesOffset + hullStart + hullCount - 1u]) - origin;
    var twiceArea = 0.0;
    for (var vertex = 0u; vertex < hullCount; vertex++) {
      let current = getPosition(hullIndices[hullIndicesOffset + hullStart + vertex]) - origin;
      twiceArea += previous.x * current.y - current.x * previous.y;
      previous = current;
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`})]}var b=`GPUShapeDescriptors`;function x(e={},t=new Float32Array(4)){if(t.length<4)throw Error(`Shape descriptors target must hold 4 elements`);let n=e.sliverThreshold??.05;if(!Number.isFinite(n)||n<0)throw Error(`Shape descriptors sliverThreshold must be a non-negative finite number`);return t.set([n,0,0,0]),t}var S=class{id;props;spatialContext;featureCount;useMonotoneChainConvexity;constructor(t){this.id=t.id??`shape-descriptors`,this.props=t,this.spatialContext=e(this.id,t.spatialContext);let{id:r}=this;if(i(t.positions,[`float32x2`],`${r} positions`),t.positions.length<1)throw Error(`${r} needs at least one position`);for(let e of[`ringOffsets`,`polygonOffsets`,`featureOffsets`])if(i(t[e],[`uint32`],`${r} ${e}`),t[e].length<2)throw Error(`${r} ${e} must contain at least two rows`);this.featureCount=t.featureOffsets.length-1;let a=t.convexityMethod??`auto`;if(![`auto`,`gift-wrapping`,`monotone-chain`].includes(a))throw Error(`${r} convexityMethod must be 'auto', 'gift-wrapping' or 'monotone-chain'`);let o=v(t.positions.length,this.featureCount);if(a===`monotone-chain`&&!o)throw Error(`${r} monotone-chain convexity supports fewer than 2^24 rows and features`);if(this.useMonotoneChainConvexity=o&&(a===`monotone-chain`||a===`auto`&&t.positions.length>=256*this.featureCount),i(t.parameters,[`float32`],`${r} parameters`),t.parameters.length<4)throw Error(`${r} parameters must hold 4 float32 values`);let s=Object.entries(t.output).filter(([,e])=>e);if(s.length===0)throw Error(`${r} requires at least one output column`);for(let[e,t]of s)if(i(t,e===`clockwise`||e===`sliver`?[`uint32`]:[`float32`],`${r} output.${e}`),t.length!==this.featureCount)throw Error(`${r} output.${e} must hold ${this.featureCount} rows`);n(r,s.map(([,e])=>e),[t.positions,t.ringOffsets,t.featureOffsets,t.polygonOffsets,t.parameters])}getCommandNodes(e){let{props:n,id:i,featureCount:a}=this,{output:o}=n;t(i,e,[n.positions,n.ringOffsets,n.featureOffsets,n.polygonOffsets,n.parameters,...Object.values(o)]);let s=!!(o.elongation||o.orientation||o.clockwise),l=!!(o.polsbyPopper||o.schwartzberg||o.sliver),u=l||!!o.convexity||!!o.areas,d=l||!!o.perimeters,f=o.areas??(u?r(e,`${i}-areas`,`float32`,a):void 0),_=o.perimeters??(d?r(e,`${i}-perimeters`,`float32`,a):void 0),v=s?r(e,`${i}-centroids`,`float32x2`,a):void 0,x=[];(f||_||v)&&x.push(...new c({spatialContext:this.spatialContext,id:`${i}-measures`,positions:n.positions,geometryType:`polygons`,ringOffsets:n.ringOffsets,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,output:{lengths:_,areas:f,centroids:v}}).getCommandNodes(e));let S={operation:b,positions:n.positions,ringOffsets:n.ringOffsets,featureOffsets:n.featureOffsets,polygonOffsets:n.polygonOffsets,featureCount:a};if(s&&v){let t=r(e,`${i}-moments`,`float32x4`,a);x.push(p(e,{...S,id:`${i}-moments`,centroids:v,moments:t}),g(e,{id:`${i}-orientation`,operation:b,featureCount:a,moments:t,elongation:o.elongation,orientation:o.orientation,clockwise:o.clockwise}))}return l&&f&&_&&x.push(h(e,{id:`${i}-compactness`,operation:b,featureCount:a,areas:f,perimeters:_,parameters:n.parameters,polsbyPopper:o.polsbyPopper,schwartzberg:o.schwartzberg,sliver:o.sliver})),o.convexity&&f&&(this.useMonotoneChainConvexity?x.push(...y(e,{...S,id:`${i}-convexity`,areas:f,convexities:o.convexity})):x.push(m(e,{...S,id:`${i}-convexity`,areas:f,convexities:o.convexity}))),x}};export{x as n,S as t};