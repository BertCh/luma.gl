import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{t as e}from"./status-BwNARr9H.js";import{D as t,E as n,h as r,n as i,o as a,s as o}from"./analysis-frame-Blu0tYdw.js";import{a as s,b as c,d as l,f as u,i as d,n as f,p,r as m,s as h,t as g}from"./pair-statistics-parameters-CO--_nmZ.js";import{N as _,_ as v,d as y,h as b,j as x,n as S,o as C,t as ee,v as w,w as T,y as E}from"./resources-CwvhEOBe.js";import{t as te}from"./gpu-grid-index-D1JNO067.js";import{l as D,n as O,s as k}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as A,r as ne,u as re}from"./wgsl-common-BNRVzgd8.js";import{s as j}from"./raster-band-utils-CA-sh8xt.js";import{n as ie}from"./gpu-raster-grid-BOSvDY5A.js";import{n as ae,t as oe}from"./separation-wgsl-OfnvdChu.js";import{A as se,O as ce,a as le,k as ue,r as de,s as fe,t as pe}from"./index-Bjk18V2I.js";import{t as me}from"./pair-statistics-fixed-point-unit-Byg9Sbw4.js";import{t as M}from"./graph-buffers-B4_80RlX.js";import{t as he}from"./summary-reader-CsgWxENf.js";import{t as ge}from"./mode-kernels-Dxm2GWvI.js";import{n as _e,r as ve,t as N}from"./b7-format-CWQYfYhK.js";import{n as ye,t as be}from"./b7-rainfall-style-D9jUS0Ny.js";function xe(e,t){j(e,t);let[n,r,i,a,o,s]=t.affine;if(r!==0||a!==0)throw Error(`${e} needs an axis-aligned grid`);let c=t.pixelInterpretation===`point`?-.5:0,l=i+n*c,u=s+o*c;return[l,u,l+n*t.width,u+o*t.height]}function Se(e,n,r=new Float32Array(9)){if(r.length<9)throw Error(`Inverse distance weighting parameter target must hold 9 elements`);let i=e.neighborCount??0,a=e.minimumNeighborCount??1;for(let[e,t]of[[`neighborCount`,i],[`minimumNeighborCount`,a]])y(`Inverse distance weighting`,e,t,0);let o=e.taper??`none`;if(o!==`none`&&o!==`modified-shepard`)throw Error(`Inverse distance weighting taper must be 'none' or 'modified-shepard'`);if(o===`modified-shepard`&&!Number.isFinite(e.searchRadius))throw Error(`Inverse distance weighting taper needs a finite searchRadius`);return t(`GPUInverseDistanceWeighting`,e.spatialContext,e.searchRadius),r.set(xe(`getGPUInverseDistanceWeightingParameterValues`,n),0),r[4]=e.searchRadius,r[5]=e.power??2,r[6]=i,r[7]=a,r[8]=o===`modified-shepard`?1:0,r}function Ce(e,t=new Float32Array(4)){if(t.length<4)throw Error(`Focal statistics parameter target must hold 4 elements`);let n=e.minimumCount??1;y(`Focal statistics`,`minimumCount`,n,0);let r=e.shape??`square`;if(r!==`square`&&r!==`circle`)throw Error(`Focal statistics shape must be 'square' or 'circle'`);return t[0]=e.radius,t[1]=r===`circle`?1:0,t[2]=n,t[3]=e.propagateCenterNoData?1:0,t}function P(e){return`
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      // Cells of one index row are contiguous in cellOffsets, so the row span is one slot range.
      let start = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnHigh + 1u];
      for (var slot = start; slot < end; slot++) {
        let sample = sortedIds[sortedIdsOffset + slot];
        ${e}
      }
    }`}function we(e){return`
    let cellExtent = (DOMAIN_MAX - DOMAIN_MIN) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
    let cellSlack = min(cellExtent.x, cellExtent.y);
    let centerPoint = clamp(center, DOMAIN_MIN, DOMAIN_MAX);
    let lowColumn = i32(columnLow);
    let highColumn = i32(columnHigh);
    let lowRow = i32(rowLow);
    let highRow = i32(rowHigh);
    let centerColumn = clamp(i32(getCoordinate(centerPoint.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH)), lowColumn, highColumn);
    let centerRow = clamp(i32(getCoordinate(centerPoint.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT)), lowRow, highRow);
    let lastRing = max(max(centerColumn - lowColumn, highColumn - centerColumn), max(centerRow - lowRow, highRow - centerRow));
    for (var ring = 0; ring <= lastRing; ring++) {
      let ringColumnLow = max(centerColumn - ring, lowColumn);
      let ringColumnHigh = min(centerColumn + ring, highColumn);
      for (var cellRow = max(centerRow - ring, lowRow); cellRow <= min(centerRow + ring, highRow); cellRow++) {
        let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
        for (var side = 0; side < 2; side++) {
          var firstColumn = ringColumnLow;
          var lastColumn = ringColumnHigh;
          if (isEdgeRow) {
            // Top and bottom rows of the ring are one contiguous slot range.
            if (side == 1) { continue; }
          } else {
            // Interior rows of the ring touch only its left and right cells.
            let column = select(centerColumn + ring, centerColumn - ring, side == 0);
            if (column < lowColumn || column > highColumn) { continue; }
            firstColumn = column;
            lastColumn = column;
          }
          let rowBase = u32(cellRow) * INDEX_WIDTH;
          let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
          let end = cellOffsets[cellOffsetsOffset + rowBase + u32(lastColumn) + 1u];
          for (var slot = start; slot < end; slot++) {
            let sample = sortedIds[sortedIdsOffset + slot];
            ${e}
          }
        }
      }
      if (nearCount >= neighborLimit) {
        // Distance from the center to the nearest border of the visited block that still has
        // unvisited cells behind it.
        var gap = 3.0e38;
        if (centerColumn - ring > lowColumn) {
          gap = min(gap, center.x - (DOMAIN_MIN.x + f32(centerColumn - ring) * cellExtent.x));
        }
        if (centerColumn + ring < highColumn) {
          gap = min(gap, DOMAIN_MIN.x + f32(centerColumn + ring + 1) * cellExtent.x - center.x);
        }
        if (centerRow - ring > lowRow) {
          gap = min(gap, center.y - (DOMAIN_MIN.y + f32(centerRow - ring) * cellExtent.y));
        }
        if (centerRow + ring < highRow) {
          gap = min(gap, DOMAIN_MIN.y + f32(centerRow + ring + 1) * cellExtent.y - center.y);
        }
        gap = gap - cellSlack;
        if (gap > 0.0 && nearDistances[neighborLimit - 1u] < gap * gap) {
          break;
        }
      }
    }`}var Te=`GPUInverseDistanceWeighting`,Ee=64,De=class{id;props;cellCount;spatialContext;separation;maximumNeighborCount;constructor(e){this.id=e.id??`inverse-distance-weighting`,this.props=e;let{id:t}=this;if(this.separation=ae(t,e.spatialContext),this.spatialContext=this.separation.context,this.cellCount=j(t,e.grid),e.grid.affine[1]!==0||e.grid.affine[3]!==0)throw Error(`${t} needs an axis-aligned grid`);for(let[n,r]of[[`indexGridSize[0]`,e.indexGridSize[0]],[`indexGridSize[1]`,e.indexGridSize[1]]])y(t,n,r,1,2**53-1);let[n,r,i,a]=e.indexBounds;if(e.indexBounds.length!==4||!e.indexBounds.every(Number.isFinite)||n>i||r>a)throw Error(`${t} indexBounds must be finite and ordered`);this.maximumNeighborCount=e.maximumNeighborCount??0,y(t,`maximumNeighborCount`,this.maximumNeighborCount,0,Ee),E(e.positions,[`float32x2`],`${t} positions`),E(e.values,[`float32`],`${t} values`);let o=e.values.length;if(o<1)throw Error(`${t} needs at least one sample`);if(e.positions.length!==o)throw Error(`${t} positions length must equal values length`);if(e.mask&&(E(e.mask,[`uint32`],`${t} mask`),e.mask.length!==o))throw Error(`${t} mask length must equal values length`);if(E(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<9)throw Error(`${t} parameters must hold 9 float32 values`);if(E(e.output.values,[`float32`],`${t} output.values`),e.output.values.length<this.cellCount)throw Error(`${t} output.values must hold grid.width * grid.height rows`);if(e.output.counts){if(E(e.output.counts,[`uint32`],`${t} output.counts`),e.output.counts.length<this.cellCount)throw Error(`${t} output.counts must hold grid.width * grid.height rows`);if(e.output.counts.buffer===e.output.values.buffer)throw Error(`${t} outputs must not share buffers`)}b(t,[e.output.values,e.output.counts],[e.positions,e.values,e.mask,e.parameters])}getCommandNodes(e){let{id:t,props:n,cellCount:r,maximumNeighborCount:i}=this,{output:a,grid:o,indexGridSize:s,indexBounds:c}=n,{width:l,height:u}=o;v(t,e,[n.positions,n.values,n.mask,n.parameters,a.values,a.counts]),w(t,`indexGridSize cells`,s[0]*s[1],C(e.device));let d=n.values.length,f=s[0]*s[1],p=T(e,`${t}-cell-offsets`,`uint32`,f+1),m=T(e,`${t}-object-ids`,`uint32`,d),h=T(e,`${t}-sorted-ids`,`uint32`,d),g=T(e,`${t}-index-count`,`uint32`,1),_=T(e,`${t}-index-overflow`,`uint32`,1),y=[...new te({id:`${t}-index`,positions:n.positions,gridSize:s,bounds:c,cellOffsets:p,objectIds:m,count:g,overflow:_}).getCommandNodes(e)];y.push(k(e,{id:`${t}-sort-cells`,operation:Te,variant:`sort-cells`,bindings:[{name:`cellOffsets`,view:p,type:`u32`,access:`read`},{name:`objectIds`,view:m,type:`u32`,access:`read`},{name:`indexCount`,view:g,type:`u32`,access:`read`},{name:`sortedIds`,view:h,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const INDEX_CELL_COUNT: u32 = ${f}u;
const SAMPLE_COUNT: u32 = ${d}u;`,body:`if (index >= min(indexCount[indexCountOffset], SAMPLE_COUNT)) {
    return;
  }
  let sample = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = INDEX_CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let start = cellOffsets[cellOffsetsOffset + low];
  let end = cellOffsets[cellOffsetsOffset + low + 1u];
  var rank = 0u;
  for (var slot = start; slot < end; slot++) {
    rank += select(0u, 1u, objectIds[objectIdsOffset + slot] < sample);
  }
  sortedIds[sortedIdsOffset + start + rank] = sample;`}));let b=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`sampleValues`,view:n.values,type:`f32`,access:`read`},...n.mask?[{name:`sampleMask`,view:n.mask,type:`u32`,access:`read`}]:[],{name:`params`,view:n.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:p,type:`u32`,access:`read`},{name:`sortedIds`,view:h,type:`u32`,access:`read`},{name:`valuesOut`,view:a.values,type:`f32`,access:`read_write`},...a.counts?[{name:`countsOut`,view:a.counts,type:`u32`,access:`read_write`}]:[]],x=i>0;return y.push(k(e,{id:`${t}-gather`,operation:Te,variant:x?`gather-nearest`:`gather-radius`,record:{formula:`gpu-spatial-analysis.inverse-distance-weighting`},bindings:b,invocationCount:r,workgroupSize:64,declarations:`const WIDTH: u32 = ${l}u;
const HEIGHT: u32 = ${u}u;
const INDEX_WIDTH: u32 = ${s[0]}u;
const INDEX_HEIGHT: u32 = ${s[1]}u;
const MAX_NEIGHBORS: u32 = ${i}u;
const DOMAIN_MIN: vec2f = vec2f(${D(c[0])}, ${D(c[1])});
const DOMAIN_MAX: vec2f = vec2f(${D(c[2])}, ${D(c[3])});
${Oe}
${oe(this.separation)}
${ke}`,body:Ae(!!n.mask,!!a.counts,x,this.separation.geographic)})),y}},Oe=`
${A}
${re}// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${ne}
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`,ke=`
struct WeightAccumulator {
  maximumLogWeight: f32,
  weightSum: f32,
  weightedSum: f32,
  isEmpty: bool,
}

fn addWeightedSample(accumulator: ptr<function, WeightAccumulator>, power: f32, radius: f32, useTaper: bool, distanceSquared: f32, value: f32) {
  if (distanceSquared == 0.0) {
    return;
  }
  var logWeight = -0.5 * power * log2(distanceSquared);
  if (useTaper) {
    // Franke-Nielson: w = ((R - d) / (R d))^p, zero at and beyond R.
    let distance = sqrt(distanceSquared);
    if (!(distance < radius)) {
      return;
    }
    logWeight = power * (log2(radius - distance) - log2(radius * distance));
  }
  if ((*accumulator).isEmpty) {
    (*accumulator).maximumLogWeight = logWeight;
    (*accumulator).weightSum = 1.0;
    (*accumulator).weightedSum = value;
    (*accumulator).isEmpty = false;
  } else if (logWeight > (*accumulator).maximumLogWeight) {
    let scale = exp2((*accumulator).maximumLogWeight - logWeight);
    (*accumulator).weightSum = (*accumulator).weightSum * scale + 1.0;
    (*accumulator).weightedSum = (*accumulator).weightedSum * scale + value;
    (*accumulator).maximumLogWeight = logWeight;
  } else {
    let weight = exp2(logWeight - (*accumulator).maximumLogWeight);
    (*accumulator).weightSum += weight;
    (*accumulator).weightedSum += weight * value;
  }
}`;function Ae(e,t,n,r){let i=n?`var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;`:``,a=n?`if (neighborLimit == 0u) {
          count += 1u;
          addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(distanceSquared), value);
        } else {
          let last = neighborLimit - 1u;
          let isFull = nearCount >= neighborLimit;
          if (!isFull || distanceSquared < nearDistances[last] ||
              (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
            var position = select(nearCount, last, isFull);
            loop {
              if (position == 0u) { break; }
              let previousDistance = nearDistances[position - 1u];
              let previousRow = nearRows[position - 1u];
              if (previousDistance < distanceSquared ||
                  (previousDistance == distanceSquared && previousRow < sample)) {
                break;
              }
              nearDistances[position] = previousDistance;
              nearRows[position] = previousRow;
              position -= 1u;
            }
            nearDistances[position] = distanceSquared;
            nearRows[position] = sample;
            nearCount = min(nearCount + 1u, neighborLimit);
          }
        }`:`count += 1u;
        addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(distanceSquared), value);`,o=n?`if (neighborLimit > 0u) {
    count = nearCount;
    for (var near = 0u; near < nearCount; near++) {
      let row = nearRows[near];
      addWeightedSample(&accumulator, power, radius, useTaper, getSquaredMeters(nearDistances[near]), sampleValues[sampleValuesOffset + row]);
    }
  }`:``,s=`let value = sampleValues[sampleValuesOffset + sample];
        if (isNanBits(value)) {
          continue;
        }
        ${e?`if (sampleMask[sampleMaskOffset + sample] == 0u) {
          continue;
        }`:``}
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let distanceSquared = getSeparationKey(position - origin, localCenter);
        if (!(distanceSquared <= radiusSquared)) {
          continue;
        }
        if (distanceSquared == 0.0 && sample < exactRow) {
          exactRow = sample;
          exactValue = value;
        }
        ${a}`;return`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let radius = params[paramsOffset + 4u];
  let power = params[paramsOffset + 5u];
  let neighborLimitValue = params[paramsOffset + 6u];
  let minimumNeighborValue = params[paramsOffset + 7u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  // Use the same positive offsets for either row order. Computing southward offsets from the
  // opposite corner can round a coincident sample away from zero, averaging duplicate hits and
  // applying the minimum-neighbor gate instead of the exact-hit rule.
  let gridMinimum = min(gridStart, gridEnd);
  let cellSize = abs(gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let sampleColumn = select(column, WIDTH - 1u - column, gridEnd.x < gridStart.x);
  let sampleRow = select(rasterRow, HEIGHT - 1u - rasterRow, gridEnd.y < gridStart.y);
  let cellOffset = (vec2f(f32(sampleColumn), f32(sampleRow)) + vec2f(0.5)) * cellSize;
  let center = gridMinimum + cellOffset;
  // Great-circle distances take absolute lon/lat, so the origin is zero there.
  let origin = ${r?`vec2f(0.0)`:`gridMinimum`};
  let localCenter = ${r?`center`:`cellOffset`};
  let useTaper = params[paramsOffset + 8u] > 0.5;
  let radiusSquared = getRadiusKey(radius);
  let queryBox = getQueryBox(center, radius);
  let queryMin = queryBox.minimum;
  let queryMax = queryBox.maximum;
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) && radius >= 0.0 &&
    isFiniteValue(power) && power >= 0.0 && (!useTaper || isFiniteValue(radius)) &&
    all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var accumulator = WeightAccumulator(0.0, 0.0, 0.0, true);
  var count = 0u;
  var exactRow = 0xffffffffu;
  var exactValue = 0.0;
  ${i}
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    // With a per-frame k, the k nearest are order independent (total order (d^2, row)), so an
    // expanding-ring walk may stop early. Radius-only sums depend on the visiting order, so they
    // keep the fixed sorted row-major sweep.
    ${n&&!r?`if (neighborLimit > 0u) {
      ${we(s)}
    } else {
      ${P(s)}
    }`:P(s)}
  }
  ${o}
  var result = getNaN();
  if (exactRow != 0xffffffffu) {
    result = exactValue;
  } else if (count >= minimumNeighbors && !accumulator.isEmpty) {
    result = accumulator.weightedSum / accumulator.weightSum;
  }
  valuesOut[valuesOutOffset + index] = result;
  ${t?`countsOut[countsOutOffset + index] = count;`:``}`}var je={spherical:0,exponential:1,gaussian:2};function Me(e,n,r=new Float32Array(12)){if(r.length<12)throw Error(`Kriging parameter target must hold 12 elements`);let{variogram:i}=e,a=je[i.model];if(a===void 0)throw Error(`Unknown variogram model ${String(i.model)}`);if(!(i.nugget>=0)||!(i.sill>=0)||!(i.nugget+i.sill>0))throw Error(`Kriging variogram nugget and sill must be non-negative with a positive total`);if(!(i.range>0)||!Number.isFinite(i.range))throw Error(`Kriging variogram range must be positive and finite`);let o=e.neighborCount??16,s=e.minimumNeighborCount??3;for(let[e,t]of[[`neighborCount`,o],[`minimumNeighborCount`,s]])y(`Kriging`,e,t,0);return t(`GPUKriging`,e.spatialContext,e.searchRadius),t(`GPUKriging`,e.spatialContext,e.variogram.range),r.fill(0),r.set(xe(`getGPUKrigingParameterValues`,n),0),r[4]=e.searchRadius,r[5]=o,r[6]=a,r[7]=i.nugget,r[8]=i.sill,r[9]=i.range,r[10]=s,r}var F=`GPUKriging`,I=16,Ne=1e-6,L=1e-4,Pe=1e-4,R=0,Fe=1,Ie=2,Le=class{id;props;spatialContext;separation;cellCount;maximumNeighborCount;constructor(t){this.id=t.id??`kriging`,this.props=t;let{id:n}=this;if(this.separation=ae(n,t.spatialContext),this.spatialContext=this.separation.context,this.cellCount=j(n,t.grid),t.grid.affine[1]!==0||t.grid.affine[3]!==0)throw Error(`${n} needs an axis-aligned grid`);for(let[e,r]of[[`indexGridSize[0]`,t.indexGridSize[0]],[`indexGridSize[1]`,t.indexGridSize[1]]])y(n,e,r,1,2**53-1);let[r,i,a,o]=t.indexBounds;if(t.indexBounds.length!==4||!t.indexBounds.every(Number.isFinite)||r>a||i>o)throw Error(`${n} indexBounds must be finite and ordered`);this.maximumNeighborCount=t.maximumNeighborCount??I,y(n,`maximumNeighborCount`,this.maximumNeighborCount,1,I);let{output:s}=t;E(t.positions,[`float32x2`],`${n} positions`),E(t.values,[`float32`],`${n} values`);let c=t.values.length;if(c<1)throw Error(`${n} needs at least one sample`);if(t.positions.length!==c)throw Error(`${n} positions length must equal values length`);if(t.mask&&(E(t.mask,[`uint32`],`${n} mask`),t.mask.length!==c))throw Error(`${n} mask length must equal values length`);if(E(t.parameters,[`float32`],`${n} parameters`),t.parameters.length<12)throw Error(`${n} parameters must hold 12 float32 values`);if(E(s.values,[`float32`],`${n} output.values`),s.values.length<this.cellCount)throw Error(`${n} output.values must hold grid.width * grid.height rows`);if(s.variance){if(E(s.variance,[`float32`],`${n} output.variance`),s.variance.length<this.cellCount)throw Error(`${n} output.variance must hold grid.width * grid.height rows`);if(s.variance.buffer===s.values.buffer)throw Error(`${n} outputs must not share buffers`)}if(s.crossValidationVariance&&!s.crossValidationPrediction)throw Error(`${n} output.crossValidationVariance needs output.crossValidationPrediction`);if(s.crossValidationPrediction){for(let[e,t]of[[`crossValidationPrediction`,s.crossValidationPrediction],[`crossValidationVariance`,s.crossValidationVariance]])if(t&&(E(t,[`float32`],`${n} output.${e}`),t.length<c))throw Error(`${n} output.${e} must hold one row per sample`);if(s.crossValidationVariance&&s.crossValidationVariance.buffer===s.crossValidationPrediction.buffer)throw Error(`${n} outputs must not share buffers`)}e(n,{invalidCount:s.status?.invalidCount,uncertainCount:s.status?.uncertainCount}),b(n,[s.values,s.variance,s.crossValidationPrediction,s.crossValidationVariance,s.status?.invalidCount,s.status?.uncertainCount],[t.positions,t.values,t.mask,t.parameters])}getCommandNodes(e){let{id:t,props:n,cellCount:r,maximumNeighborCount:i}=this,{output:a,grid:o,indexGridSize:s,indexBounds:c}=n,{width:l,height:u}=o;v(t,e,[n.positions,n.values,n.mask,n.parameters,a.values,a.variance,a.crossValidationPrediction,a.crossValidationVariance,a.status?.invalidCount,a.status?.uncertainCount]),w(t,`indexGridSize cells`,s[0]*s[1],C(e.device));let d=n.values.length,f=s[0]*s[1],p=T(e,`${t}-cell-offsets`,`uint32`,f+1),m=T(e,`${t}-object-ids`,`uint32`,d),h=T(e,`${t}-index-count`,`uint32`,1),g=T(e,`${t}-index-overflow`,`uint32`,1),_=[...new te({id:`${t}-index`,positions:n.positions,gridSize:s,bounds:c,cellOffsets:p,objectIds:m,count:h,overflow:g}).getCommandNodes(e)],y=n.values;n.mask&&(y=T(e,`${t}-masked-values`,`float32`,d),_.push(k(e,{id:`${t}-apply-mask`,operation:F,variant:`apply-mask`,bindings:[{name:`rawValues`,view:n.values,type:`f32`,access:`read`},{name:`sampleMask`,view:n.mask,type:`u32`,access:`read`},{name:`maskedValues`,view:y,type:`f32`,access:`read_write`}],invocationCount:d,declarations:ne,body:`maskedValues[maskedValuesOffset + index] = select(getNaN(), rawValues[rawValuesOffset + index], sampleMask[sampleMaskOffset + index] != 0u);`})));let b=[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`sampleValues`,view:y,type:`f32`,access:`read`},{name:`params`,view:n.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:p,type:`u32`,access:`read`},{name:`sortedIds`,view:m,type:`u32`,access:`read`}],x=`const INDEX_WIDTH: u32 = ${s[0]}u;
const INDEX_HEIGHT: u32 = ${s[1]}u;
const MAX_NEIGHBORS: u32 = ${i}u;
const SYSTEM_STRIDE: u32 = ${i+2}u;
const DOMAIN_MIN: vec2f = vec2f(${D(c[0])}, ${D(c[1])});
const DOMAIN_MAX: vec2f = vec2f(${D(c[2])}, ${D(c[3])});
const RELATIVE_PIVOT_TOLERANCE: f32 = ${D(Ne)};
const PIVOT_RATIO_TOLERANCE: f32 = ${D(L)};
const NEGATIVE_VARIANCE_TOLERANCE: f32 = ${D(Pe)};
const STATUS_OK: u32 = ${R}u;
const STATUS_INVALID: u32 = ${Fe}u;
const STATUS_UNCERTAIN: u32 = ${Ie}u;
const ORIGIN: vec2f = ${this.separation.geographic?`vec2f(0.0)`:`DOMAIN_MIN`};
${Oe}
${z(this.separation)}`,S=a.status?.invalidCount||a.status?.uncertainCount?T(e,`${t}-cell-status`,`uint32`,r):void 0;if(_.push(k(e,{id:`${t}-solve`,operation:F,variant:`local-ordinary`,record:{formula:`gpu-spatial-analysis.kriging`},bindings:[...b,{name:`valuesOut`,view:a.values,type:`f32`,access:`read_write`},...a.variance?[{name:`varianceOut`,view:a.variance,type:`f32`,access:`read_write`}]:[],...S?[{name:`cellStatus`,view:S,type:`u32`,access:`read_write`}]:[]],invocationCount:r,workgroupSize:64,declarations:`const WIDTH: u32 = ${l}u;
const HEIGHT: u32 = ${u}u;
${x}`,body:`let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  // Planar offsets from the grid start corner stay small whatever the absolute coordinates are;
  // great-circle distances take absolute lon/lat, so the origin is zero.
  let origin = ${this.separation.geographic?`vec2f(0.0)`:`gridStart`};
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let estimate = estimateKriging(center, ${this.separation.geographic?`center`:`cellOffset`}, origin, 0xffffffffu);
  valuesOut[valuesOutOffset + index] = estimate.value;
  ${a.variance?`varianceOut[varianceOutOffset + index] = estimate.variance;`:``}
  ${S?`cellStatus[cellStatusOffset + index] = estimate.status;`:``}`})),a.crossValidationPrediction){let{crossValidationPrediction:n,crossValidationVariance:r}=a;_.push(k(e,{id:`${t}-cross-validation`,operation:F,variant:`leave-one-out`,bindings:[...b,{name:`predictionOut`,view:n,type:`f32`,access:`read_write`},...r?[{name:`predictionVarianceOut`,view:r,type:`f32`,access:`read_write`}]:[]],invocationCount:d,workgroupSize:64,declarations:x,body:`let value = sampleValues[sampleValuesOffset + index];
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var prediction = getNaN();
  var predictionVariance = getNaN();
  if (!isNanBits(value) && isFiniteValue(position.x) && isFiniteValue(position.y)) {
    // Kriged from every OTHER sample: the row itself is excluded from the neighbor search.
    let estimate = estimateKriging(position, position - ORIGIN, ORIGIN, index);
    prediction = estimate.value;
    predictionVariance = estimate.variance;
  }
  predictionOut[predictionOutOffset + index] = prediction;
  ${r?`predictionVarianceOut[predictionVarianceOutOffset + index] = predictionVariance;`:``}`}))}if(S){for(let[n,r]of[[`invalid`,a.status?.invalidCount],[`uncertain`,a.status?.uncertainCount]])r&&_.push(O(e,{id:`${t}-clear-${n}`,operation:F,view:r,type:`u32`,value:`0u`,componentCount:r.length}));_.push(k(e,{id:`${t}-count-status`,operation:F,variant:`count-status`,bindings:[{name:`cellStatus`,view:S,type:`u32`,access:`read`},...a.status?.invalidCount?[{name:`invalidCountOut`,view:a.status?.invalidCount,type:`atomic<u32>`,access:`read_write`}]:[],...a.status?.uncertainCount?[{name:`uncertainCountOut`,view:a.status?.uncertainCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:r,declarations:`const STATUS_INVALID: u32 = ${Fe}u;
const STATUS_UNCERTAIN: u32 = ${Ie}u;`,body:`let status = cellStatus[cellStatusOffset + index];
  ${a.status?.invalidCount?`if (status == STATUS_INVALID) { atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u); }`:``}
  ${a.status?.uncertainCount?`if (status == STATUS_UNCERTAIN) { atomicAdd(&uncertainCountOut[uncertainCountOutOffset], 1u); }`:``}`}))}return _}};function z(e){let t=e.geographic?P:we;return`
${oe(e)}
struct KrigingEstimate {
  value: f32,
  variance: f32,
  status: u32,
}

// Variogram normalized by its total sill; gamma(0) = 0.
fn evaluateVariogram(distance: f32, modelCode: u32, nugget: f32, sill: f32, range: f32) -> f32 {
  if (distance <= 0.0) { return 0.0; }
  let scaled = distance / range;
  var shape = 1.0;
  if (modelCode == 0u) {
    shape = select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0);
  } else if (modelCode == 1u) {
    shape = 1.0 - exp(-3.0 * scaled);
  } else {
    shape = 1.0 - exp(-3.0 * scaled * scaled);
  }
  return (nugget + sill * shape) / (nugget + sill);
}

// Gaussian elimination with partial pivoting on an n x (n + 1) augmented system stored row-major
// with SYSTEM_STRIDE columns; leaves the solution in column n. Returns false when a pivot is
// negligible against the system scale or the pivot ratio (a condition estimate) is too small.
fn solveSystem(system: ptr<function, array<f32, ${(I+1)*(I+2)}>>, n: u32, scale: f32) -> bool {
  var minimumPivot = 3.0e38;
  var maximumPivot = 0.0;
  for (var column = 0u; column < n; column++) {
    var pivotRow = column;
    var pivotMagnitude = abs((*system)[column * SYSTEM_STRIDE + column]);
    for (var row = column + 1u; row < n; row++) {
      let magnitude = abs((*system)[row * SYSTEM_STRIDE + column]);
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (!(pivotMagnitude > RELATIVE_PIVOT_TOLERANCE * scale)) { return false; }
    minimumPivot = min(minimumPivot, pivotMagnitude);
    maximumPivot = max(maximumPivot, pivotMagnitude);
    if (pivotRow != column) {
      for (var k = column; k <= n; k++) {
        let held = (*system)[column * SYSTEM_STRIDE + k];
        (*system)[column * SYSTEM_STRIDE + k] = (*system)[pivotRow * SYSTEM_STRIDE + k];
        (*system)[pivotRow * SYSTEM_STRIDE + k] = held;
      }
    }
    let pivot = (*system)[column * SYSTEM_STRIDE + column];
    for (var row = column + 1u; row < n; row++) {
      let factor = (*system)[row * SYSTEM_STRIDE + column] / pivot;
      for (var k = column; k <= n; k++) {
        (*system)[row * SYSTEM_STRIDE + k] -= factor * (*system)[column * SYSTEM_STRIDE + k];
      }
    }
  }
  if (!(minimumPivot >= PIVOT_RATIO_TOLERANCE * maximumPivot)) { return false; }
  for (var step = 0u; step < n; step++) {
    let row = n - 1u - step;
    var sum = (*system)[row * SYSTEM_STRIDE + n];
    for (var k = row + 1u; k < n; k++) {
      sum -= (*system)[row * SYSTEM_STRIDE + k] * (*system)[k * SYSTEM_STRIDE + n];
    }
    (*system)[row * SYSTEM_STRIDE + n] = sum / (*system)[row * SYSTEM_STRIDE + row];
  }
  return true;
}

// Ordinary kriging estimate at one query point. center is the absolute position (used only to
// find index cells); localCenter and origin give the same point as an offset from origin, and
// every distance is a difference of offsets. excludeRow removes one sample (leave-one-out).
fn estimateKriging(center: vec2f, localCenter: vec2f, origin: vec2f, excludeRow: u32) -> KrigingEstimate {
  let radius = params[paramsOffset + 4u];
  let neighborLimitValue = params[paramsOffset + 5u];
  let modelCode = u32(clamp(params[paramsOffset + 6u], 0.0, 2.0));
  let nugget = params[paramsOffset + 7u];
  let sill = params[paramsOffset + 8u];
  let range = params[paramsOffset + 9u];
  let minimumNeighborValue = params[paramsOffset + 10u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  let radiusKey = getRadiusKey(radius);
  let queryBox = getQueryBox(center, radius);
  let queryMin = queryBox.minimum;
  let queryMax = queryBox.maximum;
  let variogramValid = nugget >= 0.0 && sill >= 0.0 && nugget + sill > 0.0 && range > 0.0 && isFiniteValue(range);
  let valid = variogramValid && neighborLimit > 0u && isFiniteValue(center.x) && isFiniteValue(center.y) &&
    radius >= 0.0 && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var estimate = KrigingEstimate(getNaN(), getNaN(), STATUS_INVALID);
  var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${t(`if (sample == excludeRow) {
          continue;
        }
        let value = sampleValues[sampleValuesOffset + sample];
        if (isNanBits(value)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let distanceSquared = getSeparationKey(position - origin, localCenter);
        if (!(distanceSquared <= radiusKey)) {
          continue;
        }
        let last = neighborLimit - 1u;
        let isFull = nearCount >= neighborLimit;
        if (!isFull || distanceSquared < nearDistances[last] ||
            (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
          var slotPosition = select(nearCount, last, isFull);
          loop {
            if (slotPosition == 0u) { break; }
            let previousDistance = nearDistances[slotPosition - 1u];
            let previousRow = nearRows[slotPosition - 1u];
            if (previousDistance < distanceSquared ||
                (previousDistance == distanceSquared && previousRow < sample)) {
              break;
            }
            nearDistances[slotPosition] = previousDistance;
            nearRows[slotPosition] = previousRow;
            slotPosition -= 1u;
          }
          nearDistances[slotPosition] = distanceSquared;
          nearRows[slotPosition] = sample;
          nearCount = min(nearCount + 1u, neighborLimit);
        }`)}
  }
  if (nearCount == 0u) {
    return estimate;
  }

  // Coalesce samples at identical positions into one kriging point carrying their mean value.
  // Identical positions have identical distances, and the list order is deterministic, so the
  // sums are too.
  var uniqueCount = 0u;
  var uniquePositions: array<vec2f, MAX_NEIGHBORS>;
  var uniqueDistances: array<f32, MAX_NEIGHBORS>;
  var uniqueValueSums: array<f32, MAX_NEIGHBORS>;
  var uniqueMultiplicities: array<f32, MAX_NEIGHBORS>;
  for (var i = 0u; i < nearCount; i++) {
    let row = nearRows[i];
    let localPosition = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]) - origin;
    let value = sampleValues[sampleValuesOffset + row];
    var matched = false;
    for (var j = 0u; j < uniqueCount; j++) {
      if (all(uniquePositions[j] == localPosition)) {
        uniqueValueSums[j] += value;
        uniqueMultiplicities[j] += 1.0;
        matched = true;
        break;
      }
    }
    if (!matched) {
      uniquePositions[uniqueCount] = localPosition;
      uniqueDistances[uniqueCount] = nearDistances[i];
      uniqueValueSums[uniqueCount] = value;
      uniqueMultiplicities[uniqueCount] = 1.0;
      uniqueCount += 1u;
    }
  }

  // Exact hit: the nearest distinct position coincides with the query.
  if (isExactHit(uniqueDistances[0], uniquePositions[0], localCenter)) {
    return KrigingEstimate(uniqueValueSums[0] / uniqueMultiplicities[0], 0.0, STATUS_OK);
  }
  if (uniqueCount < minimumNeighbors) {
    return estimate;
  }

  let n = uniqueCount + 1u;
  var system: array<f32, ${(I+1)*(I+2)}>;
  var targetGamma: array<f32, MAX_NEIGHBORS>;
  var systemScale = 1.0;
  for (var i = 0u; i < uniqueCount; i++) {
    system[i * SYSTEM_STRIDE + i] = 0.0;
    // The variogram matrix is symmetric: one evaluation fills both triangles.
    for (var j = i + 1u; j < uniqueCount; j++) {
      let gamma = evaluateVariogram(getPairMeters(uniquePositions[i], uniquePositions[j]), modelCode, nugget, sill, range);
      system[i * SYSTEM_STRIDE + j] = gamma;
      system[j * SYSTEM_STRIDE + i] = gamma;
      systemScale = max(systemScale, abs(gamma));
    }
    system[i * SYSTEM_STRIDE + uniqueCount] = 1.0;
    system[uniqueCount * SYSTEM_STRIDE + i] = 1.0;
    targetGamma[i] = evaluateVariogram(getSeparationMeters(uniqueDistances[i]), modelCode, nugget, sill, range);
    system[i * SYSTEM_STRIDE + n] = targetGamma[i];
  }
  system[uniqueCount * SYSTEM_STRIDE + uniqueCount] = 0.0;
  system[uniqueCount * SYSTEM_STRIDE + n] = 1.0;
  if (!solveSystem(&system, n, systemScale)) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  let baseValue = uniqueValueSums[0] / uniqueMultiplicities[0];
  var prediction = 0.0;
  var predictionVariance = system[uniqueCount * SYSTEM_STRIDE + n];
  for (var i = 0u; i < uniqueCount; i++) {
    let weight = system[i * SYSTEM_STRIDE + n];
    prediction += weight * (uniqueValueSums[i] / uniqueMultiplicities[i] - baseValue);
    predictionVariance += weight * targetGamma[i];
  }
  // Variance is in units of the total sill and nonnegative for a valid model. Only f32 rounding
  // is clamped; a materially negative value means the weights are unreliable.
  if (!(predictionVariance >= -NEGATIVE_VARIANCE_TOLERANCE) || !isFiniteValue(prediction)) {
    estimate.status = STATUS_UNCERTAIN;
    return estimate;
  }
  return KrigingEstimate(baseValue + prediction, max(predictionVariance, 0.0) * (nugget + sill), STATUS_OK);
}`}var B=`GPUFocalStatistics`,Re=64,ze=class{id;props;cellCount;constructor(e){this.id=e.id??`focal-statistics`,this.props=e;let{id:t}=this;if(this.cellCount=j(t,e.grid),y(t,`maximumRadius`,e.maximumRadius,0,Re),e.noDataValue!==void 0&&!Number.isFinite(e.noDataValue))throw Error(`${t} noDataValue must be finite (NaN cells are always nodata)`);let{output:n}=e,r=V(n);if(r.length===0)throw Error(`${t} needs at least one output`);if(E(e.values,[`float32`],`${t} values`),e.values.length<this.cellCount)throw Error(`${t} values must hold grid.width * grid.height rows`);if(e.validity&&(E(e.validity,[`uint32`],`${t} validity`),e.validity.length<this.cellCount))throw Error(`${t} validity must hold grid.width * grid.height rows`);if(E(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 float32 values`);for(let[e,n]of r)if(e===`counts`?E(n,[`uint32`],`${t} output.counts`):E(n,[`float32`],`${t} output.${e}`),n.length<this.cellCount)throw Error(`${t} output.${e} must hold grid.width * grid.height rows`);let i=r.map(([,e])=>e.buffer);if(new Set(i).size!==i.length)throw Error(`${t} outputs must not share buffers`);b(t,r.map(([,e])=>e),[e.values,e.validity,e.parameters])}getCommandNodes(e){let{id:t,props:n,cellCount:r}=this,{output:i}=n;v(t,e,[n.values,n.validity,n.parameters,...V(i).map(([,e])=>e)]);let a=[{name:`raster`,view:n.values,type:`f32`,access:`read`},...n.validity?[{name:`validity`,view:n.validity,type:`u32`,access:`read`}]:[],{name:`params`,view:n.parameters,type:`f32`,access:`read`}],o=`const WIDTH: u32 = ${n.grid.width}u;
const HEIGHT: u32 = ${n.grid.height}u;
const MAXIMUM_RADIUS: i32 = ${n.maximumRadius};
${Be(!!n.validity,n.noDataValue)}`,s=[],c=(e,t,n=`f32`)=>t?[{name:e,view:t,type:n,access:`read_write`}]:[];return(i.sum||i.mean||i.standardDeviation||i.counts)&&s.push(k(e,{id:`${t}-moments`,operation:B,variant:`moments`,bindings:[...a,...c(`sumOut`,i.sum),...c(`meanOut`,i.mean),...c(`deviationOut`,i.standardDeviation),...c(`countOut`,i.counts,`u32`)],invocationCount:r,workgroupSize:64,declarations:o,body:`${Ve}
  var count = 0u;
  var sum = 0.0;
  ${H(`count += 1u;
      sum += value;`)}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  let mean = sum / f32(max(count, 1u));
  ${i.sum?`sumOut[sumOutOffset + index] = select(nan, sum, isDefined);`:``}
  ${i.mean?`meanOut[meanOutOffset + index] = select(nan, mean, isDefined);`:``}
  ${i.counts?`countOut[countOutOffset + index] = count;`:``}
  ${i.standardDeviation?`var squaredDeviationSum = 0.0;
  if (isDefined) {
    ${H(`let deviation = value - mean;
      squaredDeviationSum += deviation * deviation;`)}
  }
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`:``}`})),(i.min||i.max||i.range)&&s.push(k(e,{id:`${t}-extremes`,operation:B,variant:`extremes`,bindings:[...a,...c(`minOut`,i.min),...c(`maxOut`,i.max),...c(`rangeOut`,i.range)],invocationCount:r,workgroupSize:64,declarations:o,body:`${Ve}
  var count = 0u;
  var minimum = 0.0;
  var maximum = 0.0;
  ${H(`if (count == 0u) {
        minimum = value;
        maximum = value;
      } else {
        minimum = min(minimum, value);
        maximum = max(maximum, value);
      }
      count += 1u;`)}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  ${i.min?`minOut[minOutOffset + index] = select(nan, minimum, isDefined);`:``}
  ${i.max?`maxOut[maxOutOffset + index] = select(nan, maximum, isDefined);`:``}
  ${i.range?`rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);`:``}`})),s}};function V(e){return[`mean`,`sum`,`min`,`max`,`range`,`standardDeviation`,`counts`].flatMap(t=>{let n=e[t];return n?[[t,n]]:[]})}function Be(e,t){return`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${ne}
${A}
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN is nodata, and infinities are kept as values.
  var isValid = (bitcast<u32>(sample) & 0x7fffffffu) <= 0x7f800000u;
  ${t===void 0?``:`isValid = isValid && sample != ${D(t)};`}
  ${e?`isValid = isValid && validity[validityOffset + cell] != 0u;`:``}
  return isValid;
}`}var Ve=`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let radiusValue = params[paramsOffset];
  let isCircle = params[paramsOffset + 1u] == 1.0;
  let minimumCountValue = params[paramsOffset + 2u];
  let propagateCenterNoData = params[paramsOffset + 3u] != 0.0;
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  // NaN or negative radii give an empty window.
  let hasWindow = radiusValue >= 0.0;
  let windowRadius = select(-1, i32(min(floor(radiusValue), f32(MAXIMUM_RADIUS))), hasWindow);
  let radiusSquared = radiusValue * radiusValue;
  var centerValue = 0.0;
  let centerValid = readCell(index, &centerValue);`;function H(e){return`for (var dy = -windowRadius; dy <= windowRadius; dy++) {
    let sampleRow = row + dy;
    if (sampleRow < 0 || sampleRow >= i32(HEIGHT)) {
      continue;
    }
    for (var dx = -windowRadius; dx <= windowRadius; dx++) {
      let sampleColumn = column + dx;
      if (sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
        continue;
      }
      if (isCircle && f32(dx * dx + dy * dy) > radiusSquared) {
        continue;
      }
      var value = 0.0;
      if (!readCell(u32(sampleRow) * WIDTH + u32(sampleColumn), &value)) {
        continue;
      }
      ${e}
    }
  }`}function He(e,t=new Float32Array(8)){if(t.length<8)throw Error(`GPUVariogram target must hold 8 elements`);let n=e.azimuthOffset??0;if(g(`GPUVariogram`,e.bounds,e.maximumDistance),!Number.isFinite(n))throw Error(`GPUVariogram azimuthOffset must be finite`);return t.set([...e.bounds===`study-window`?[NaN,NaN,NaN,NaN]:e.bounds,e.maximumDistance,n,0,0]),t}var U=`GPUVariogram`,Ue=Object.freeze({...r,measure:`distance`}),W=16777216,G=16777216,We=class{id;props;spatialContext;studyWindow;directionCount;constructor(t){this.id=t.id??`variogram`,this.props=t;let r=this.id;this.spatialContext=n(r,t.spatialContext,Ue),this.studyWindow=c(r,t.studyWindow,this.spatialContext),u(r,{...t,parameterLength:8});let{output:i}=t,a=t.directionCount??1;this.directionCount=a,y(r,`lagCount`,t.lagCount),y(r,`directionCount`,a);let o=t.lagCount*a;if(o>512)throw Error(`${r} lagCount * directionCount must be at most 512`);d(r,o,i.robustSemivariances?5:4),i.status?.overflow&&e(r,{overflow:i.status?.overflow});for(let[e,t]of[[`output.semivariances`,i.semivariances],[`output.meanDistances`,i.meanDistances],[`output.robustSemivariances`,i.robustSemivariances]])if(t&&(E(t,[`float32`],`${r} ${e}`),t.length<o))throw Error(`${r} ${e} must hold lagCount * directionCount rows`);if(i.pairCounts&&(x(i.pairCounts,`${r} output.pairCounts`),i.pairCounts.length<o))throw Error(`${r} output.pairCounts must hold lagCount * directionCount rows`);if(i.statistics&&(E(i.statistics,[`float32`],`${r} output.statistics`),i.statistics.length<5))throw Error(`${r} output.statistics must hold 5 rows`);b(r,[i.semivariances,i.pairCounts,i.meanDistances,i.robustSemivariances,i.statistics,i.status?.overflow],[t.positions,t.values,t.parameters,t.mask])}getCommandNodes(e){let{id:t,props:n,directionCount:r}=this,{positions:i,parameters:a,gridSize:o,lagCount:c,output:u}=n;p(t,e.device,i.length),v(t,e,[i,n.values,a,n.mask,u.semivariances,u.pairCounts,u.meanDistances,u.robustSemivariances,u.statistics,u.status?.overflow]);let d=i.length,g=c*r,_=!!u.robustSemivariances,y=_?4:3,b=y+1,x=this.studyWindow===`rectangle`?void 0:this.studyWindow.getViews(e);x&&v(t,e,[x.edges,x.summary,x.slabs.params,x.slabs.slabOffsets,x.slabs.slabEntries,x.sampleCells,x.sampleCellCount,x.packed]);let S=x?l(e,{id:t,operation:U,parameters:a,parameterLength:a.length,window:x}):void 0,C=S?.parameters??a,ee=h(e,{id:t,operation:U,positions:i,parameters:C,gridSize:o,values:n.values,mask:n.mask,studyWindow:x}),w=ee.statistics,T=[...S?.nodes??[],...ee.nodes],E=me(e,{id:t,operation:U,values:n.values,mask:n.mask,statistics:w});T.push(...E.nodes);let te=E.unitVariance,D=`
const MOMENTS: u32 = ${d}u;
const LAG_COUNT: u32 = ${c}u;
const DIRECTION_COUNT: u32 = ${r}u;
const FIXED_POINT_SCALE: f32 = ${W}.0;
const SEMIVARIANCE_UNITS_PER_VARIANCE: f32 = ${G}.0;
const LARGE_TERM_UNIT: f32 = ${2**12}.0;
const PI: f32 = 3.14159265358979;

struct VariogramScales {
  semivariance: f32,
  robust: f32,
  distance: f32
}

fn getVariogramScales() -> VariogramScales {
  let valueRange = statistics[statisticsOffset + MOMENTS + 4u] - statistics[statisticsOffset + MOMENTS + 3u];
  var scales: VariogramScales;
  // The Matheron unit comes from the typical term (the robust variance), not from the value range.
  let unitVariance = unitVarianceView[unitVarianceViewOffset];
  scales.semivariance = select(0.0, SEMIVARIANCE_UNITS_PER_VARIANCE / unitVariance, unitVariance > 0.0);
  scales.robust = select(0.0, FIXED_POINT_SCALE / sqrt(valueRange), valueRange > 0.0);
  scales.distance = FIXED_POINT_SCALE / parameters[parametersOffset + 4u];
  return scales;
}`,O=f(e,{id:`${t}-histogram`,operation:U,positions:i,parameters:C,gridSize:o,sortedRows:ee.sortedRows,cellOffsets:ee.cellOffsets,slotCount:g,channelCount:b,pairOrder:`unordered`,extraBindings:[{name:`statistics`,view:w,type:`f32`,access:`read`},{name:`unitVarianceView`,view:te,type:`f32`,access:`read`}],declarations:D,focusPrologue:`let scales = getVariogramScales();
    let focusValue = statistics[statisticsOffset + focus];
    let azimuthOffset = readParameter(5u);`,pairAction:`let lag = min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u);
            var sector = 0u;
            if (DIRECTION_COUNT > 1u) {
              let angle = atan2(deltaY, deltaX) - azimuthOffset;
              let reduced = angle - floor(angle / PI) * PI;
              sector = min(u32(max(reduced, 0.0) / PI * f32(DIRECTION_COUNT)), DIRECTION_COUNT - 1u);
            }
            let bin = sector * LAG_COUNT + lag;
            let difference = statistics[statisticsOffset + neighbor] - focusValue;
            accumulate(bin, 0u, 1u);
            accumulateScaled(bin, 1u, ${y}u, 0.5 * difference * difference * scales.semivariance);
            accumulate(bin, 2u, quantizePairAmount(pairDistance * scales.distance));
            ${_?`accumulate(bin, 3u, quantizePairAmount(sqrt(abs(difference)) * scales.robust));`:``}`});T.push(...O.nodes);let A=[{name:`accumulators`,view:O.accumulators,type:`u32`,access:`read`},{name:`statistics`,view:w,type:`f32`,access:`read`},{name:`unitVarianceView`,view:te,type:`f32`,access:`read`},{name:`parameters`,view:C,type:`f32`,access:`read`},{name:`semivariances`,view:u.semivariances,type:`f32`,access:`read_write`}];return u.pairCounts&&A.push({name:`pairCounts`,view:u.pairCounts,type:`u32`,access:`read_write`}),u.meanDistances&&A.push({name:`meanDistances`,view:u.meanDistances,type:`f32`,access:`read_write`}),u.robustSemivariances&&A.push({name:`robustSemivariances`,view:u.robustSemivariances,type:`f32`,access:`read_write`}),T.push(k(e,{id:`${t}-finish`,operation:U,variant:`finish`,bindings:A,invocationCount:g,declarations:`${s}
${D}
${m(`accumulators`,b,``,g)}`,body:`let scales = getVariogramScales();
  let pairCount = readPairAccumulator(index, 0u);
  let hasPairs = pairCount > 0.0;
  let nan = getQuietNaN(index);
  // The large channel holds terms in units of LARGE_TERM_UNIT fixed-point units.
  let semivarianceSum = readPairAccumulator(index, 1u) + readPairAccumulator(index, ${y}u) * LARGE_TERM_UNIT;
  // A zero variance means every difference is 0 (scale 0, sum 0).
  let semivariance = select(0.0, semivarianceSum / scales.semivariance / pairCount, scales.semivariance > 0.0);
  semivariances[semivariancesOffset + index] = select(nan, semivariance, hasPairs);
  ${u.pairCounts?`pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);`:``}
  ${u.meanDistances?`meanDistances[meanDistancesOffset + index] =
    select(nan, readPairAccumulator(index, 2u) / scales.distance / pairCount, hasPairs);`:``}
  ${_?`let robustMean = select(0.0, readPairAccumulator(index, 3u) / scales.robust / pairCount, scales.robust > 0.0);
  let robustSquare = robustMean * robustMean;
  let robustValue = robustSquare * robustSquare / (2.0 * (0.457 + 0.494 / pairCount));
  robustSemivariances[robustSemivariancesOffset + index] = select(nan, robustValue, hasPairs);`:``}`})),u.status?.overflow&&T.push(k(e,{id:`${t}-overflow`,operation:U,variant:`overflow`,bindings:[{name:`accumulators`,view:O.accumulators,type:`u32`,access:`read`},{name:`overflow`,view:u.status?.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:m(`accumulators`,b,``,g),body:`var flagged = readPairOverflow();
  for (var bin = 0u; bin < ${g}u; bin++) {
    flagged = flagged || readPairCount(bin, 0u) == 0xffffffffu;
  }
  overflow[overflowOffset] = select(0u, 1u, flagged);`})),u.statistics&&T.push(k(e,{id:`${t}-statistics`,operation:U,variant:`statistics`,bindings:[{name:`statistics`,view:w,type:`f32`,access:`read`},{name:`statisticsOut`,view:u.statistics,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const MOMENTS: u32 = ${d}u;`,body:`let count = statistics[statisticsOffset + MOMENTS];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = statistics[statisticsOffset + MOMENTS + 2u] / count;
  statisticsOut[statisticsOutOffset + 3u] = statistics[statisticsOffset + MOMENTS + 3u];
  statisticsOut[statisticsOutOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`})),T}},Ge=200,Ke=60,K=3;function qe(e,t){switch(e){case`spherical`:return t>=1?1:1.5*t-.5*t**3;case`exponential`:return 1-Math.exp(-3*t);case`gaussian`:return 1-Math.exp(-3*t*t);default:throw Error(`Unknown variogram model ${String(e)}`)}}function Je(e,t){return t<=0?0:e.nugget+e.sill*qe(e.model,t/e.range)}function Ye(e,t){let{model:n}=t,r=t.weighting??`cressie`;if(qe(n,0),r!==`cressie`&&r!==`pairs`&&r!==`none`)throw Error(`Unknown variogram weighting ${String(r)}`);let i=[],a=[],o=[],s=Math.min(e.distances.length,e.semivariances.length,e.pairCounts.length);for(let t=0;t<s;t++){let n=e.distances[t],r=e.semivariances[t],s=e.pairCounts[t];Number.isFinite(n)&&Number.isFinite(r)&&n>0&&s>0&&(i.push(n),a.push(r),o.push(s))}if(i.length<2)throw Error(`Variogram model fitting needs at least two non-empty bins`);let c=o.map(e=>r===`none`?1:e),l=Xe(n,i,a,c);if(r===`cressie`)for(let e=0;e<K;e++){let e=l;c=o.map((t,r)=>{let a=Je({...e,model:n},i[r]);return t/Math.max(a*a,Number.MIN_VALUE)}),l=Xe(n,i,a,c)}return{model:n,...l}}function Xe(e,t,n,r){let i=Math.min(...t),a=Math.max(...t),o=Math.log(i/10),s=Math.log(3*a),c=[];for(let i=0;i<Ge;i++){let a=Math.exp(o+(s-o)*i/(Ge-1));c.push(q(e,a,t,n,r))}let l=0;for(let e=1;e<c.length;e++)c[e].residual<c[l].residual&&(l=e);let u=Math.log(c[Math.max(l-1,0)].range),d=Math.log(c[Math.min(l+1,c.length-1)].range),f=(Math.sqrt(5)-1)/2,p=d-f*(d-u),m=u+f*(d-u),h=q(e,Math.exp(p),t,n,r),g=q(e,Math.exp(m),t,n,r);for(let i=0;i<Ke;i++)h.residual<=g.residual?(d=m,m=p,g=h,p=d-f*(d-u),h=q(e,Math.exp(p),t,n,r)):(u=p,p=m,h=g,m=u+f*(d-u),g=q(e,Math.exp(m),t,n,r));let _=c[l];for(let e of[h,g])e.residual<_.residual&&(_=e);return _}function q(e,t,n,r,i){let a=0,o=0,s=0,c=0,l=0,u=n.map(n=>qe(e,n/t));for(let e=0;e<n.length;e++){let t=i[e],n=u[e];a+=t,o+=t*n,s+=t*n*n,c+=t*r[e],l+=t*n*r[e]}let d=a*s-o*o,f=0,p=0;if(Math.abs(d)>1e-300&&(f=(s*c-o*l)/d,p=(a*l-o*c)/d),!(f>=0&&p>=0)){let e=s>0?Math.max(l/s,0):0,t=a>0?Math.max(c/a,0):0;J(0,e,u,r,i)<=J(t,0,u,r,i)?(f=0,p=e):(f=t,p=0)}return{nugget:f,sill:p,range:t,residual:J(f,p,u,r,i)}}function J(e,t,n,r,i){let a=0;for(let o=0;o<n.length;o++){let s=r[o]-e-t*n[o];a+=i[o]*s*s}return a}var Y=48,Ze=class extends fe{static layerName=`B7TessellatedRasterLayer`;getVertexCount(){return 6*Y*Y}getShaderSource(){let e=super.getShaderSource(),t=e.indexOf(`@vertex fn vertexMain`),n=e.indexOf(`@fragment fn fragmentMain`);if(t<0||n<t)throw Error(`B7TessellatedRasterLayer could not find the raster vertex shader`);let r=`@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> RasterVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let quad = vertexIndex / 6u;
  let cell = corners[vertexIndex % 6u] + vec2<f32>(f32(quad % ${Y}u), f32(quad / ${Y}u));
  let minimum = vec2<f32>(rasterBounds[0], rasterBounds[1]);
  let maximum = vec2<f32>(rasterBounds[2], rasterBounds[3]);
  let worldPosition = minimum + cell / ${Y}.0 * (maximum - minimum);
  var output: RasterVertexOutput;
  output.position = projectSpatialAnalysisPosition(worldPosition);
  output.worldPosition = worldPosition;
  return output;
}

`;return e.slice(0,t)+r+e.slice(n)}},X=384,Z=240,Q=X*Z;function Qe(e){let[t,n,r,i]=e;return ie({width:X,height:Z,cellSize:[(r-t)/X,(i-n)/Z],origin:[t,n],rowDirection:`north`})}var $e=32,et=16,tt=8,nt=[64,64],rt=16,it=4,$=rt*it,at=[32,32],ot=10,st=450,ct=11,lt={kind:`azimuthal-equidistant`,origin:[-83,32]};async function ut(e){let t=e.datasets.get(`ghcn-stations`),{device:n}=e,r=i(lt),s=o(r,t.column(`position`)),c=s.length/2,l={prcp:t.column(`prcp`),prcpPrevDay:t.column(`prcpPrevDay`),tmax:t.column(`tmax`)},u=t.column(`elevation`),d=se(await ce(ue(`ghcn-stations`,`stations.csv`),e.signal));e.signal.throwIfAborted();let f=d.rows.map(e=>e[1]??``),p=d.rows.map(e=>e[2]??``),m=[1/0,1/0,-1/0,-1/0];for(let e=0;e<c;e++)m[0]=Math.min(m[0],s[e*2]),m[1]=Math.min(m[1],s[e*2+1]),m[2]=Math.max(m[2],s[e*2]),m[3]=Math.max(m[3],s[e*2+1]);let h=[m[0]-1e3,m[1]-1e3,m[2]+1e3,m[3]+1e3],g=new ee(n,`rainfall`),v=g.createBuffer(`positions`,s),y=g.createBuffer(`values`,l[e.options.variable]),b=g.createBuffer(`mask`,new Uint32Array(c).fill(1)),x=g.createBuffer(`held-out-ids`,new Uint32Array(c)),C=g.createBuffer(`idw-surface`,Q*4),w=g.createBuffer(`idw-counts`,Q*4),T=g.createBuffer(`kriging-surface`,Q*4),E=g.createBuffer(`kriging-variance`,Q*4),te=g.createBuffer(`selected-surface`,Q*4),D=g.createBuffer(`auxiliary-surface`,Q*4),O={mean:g.createBuffer(`focal-mean`,Q*4),min:g.createBuffer(`focal-min`,Q*4),max:g.createBuffer(`focal-max`,Q*4),range:g.createBuffer(`focal-range`,Q*4),standardDeviation:g.createBuffer(`focal-standard-deviation`,Q*4)},k=g.createBuffer(`semivariances`,$*4),A=g.createBuffer(`robust-semivariances`,$*4),ne=g.createBuffer(`pair-counts`,$*4),re=g.createBuffer(`mean-distances`,$*4),j=g.createBuffer(`variogram-statistics`,20),ae=g.createParameterBuffer(`idw-parameters`,`float32`,9),oe=g.createParameterBuffer(`kriging-parameters`,`float32`,12),fe=g.createParameterBuffer(`focal-parameters`,`float32`,4),me=g.createParameterBuffer(`variogram-parameters`,`float32`,8),xe=g.createParameterBuffer(`mode-parameters`,`float32`,4),P=new _(n,{id:`rainfall`}),we=M(P,`positions`,v,`float32x2`,c),Te=M(P,`values`,y,`float32`,c),Ee=M(P,`mask`,b,`uint32`,c),Oe=M(P,`idw-surface`,C,`float32`,Q),ke=M(P,`kriging-surface`,T,`float32`,Q),Ae=M(P,`kriging-variance`,E,`float32`,Q),je=M(P,`idw-counts`,w,`uint32`,Q),F=M(P,`selected`,te,`float32`,Q);P.add(new De({spatialContext:r.spatialContext,id:`idw`,positions:we,values:Te,mask:Ee,parameters:ae.importToGraph(P),grid:ie({width:X,height:Z}),indexGridSize:nt,indexBounds:h,maximumNeighborCount:$e,output:{values:Oe,counts:je}})),P.add(new Le({spatialContext:r.spatialContext,id:`kriging`,positions:we,values:Te,mask:Ee,parameters:oe.importToGraph(P),grid:ie({width:X,height:Z}),indexGridSize:nt,indexBounds:h,maximumNeighborCount:et,output:{values:ke,variance:Ae}})),ge(P,{id:`select-surface`,invocationCount:Q,bindings:[{name:`idw`,view:Oe,type:`f32`,access:`read`},{name:`kriging`,view:ke,type:`f32`,access:`read`},{name:`variance`,view:Ae,type:`f32`,access:`read`},{name:`counts`,view:je,type:`u32`,access:`read`},{name:`mode`,view:xe.importToGraph(P),type:`f32`,access:`read`},{name:`selected`,view:F,type:`f32`,access:`read_write`},{name:`auxiliary`,view:M(P,`auxiliary`,D,`float32`,Q),type:`f32`,access:`read_write`}],body:`
  var nanBits: u32 = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let a = idw[idwOffset + index];
  let b = kriging[krigingOffset + index];
  let v = variance[varianceOffset + index];
  let aInvalid = (bitcast<u32>(a) & 0x7fffffffu) >= 0x7f800000u;
  let bInvalid = (bitcast<u32>(b) & 0x7fffffffu) >= 0x7f800000u;
  let vInvalid = (bitcast<u32>(v) & 0x7fffffffu) >= 0x7f800000u;
  let surface = u32(mode[modeOffset]);
  var selectedValue = a;
  var invalid = aInvalid;
  if (surface == 1u) {
    selectedValue = b;
    invalid = bInvalid;
  } else if (surface == 2u) {
    selectedValue = b - a;
    invalid = aInvalid || bInvalid;
  }
  selected[selectedOffset + index] = select(selectedValue, nan, invalid);
  if (u32(mode[modeOffset + 1u]) == 1u) {
    auxiliary[auxiliaryOffset + index] = f32(counts[countsOffset + index]);
  } else {
    auxiliary[auxiliaryOffset + index] = select(sqrt(max(v, 0.0)), nan, vInvalid);
  }`}),P.add(new ze({id:`focal`,values:F,grid:ie({width:X,height:Z}),maximumRadius:tt,parameters:fe.importToGraph(P),output:{mean:M(P,`focal-mean`,O.mean,`float32`,Q),min:M(P,`focal-min`,O.min,`float32`,Q),max:M(P,`focal-max`,O.max,`float32`,Q),range:M(P,`focal-range`,O.range,`float32`,Q),standardDeviation:M(P,`focal-standard-deviation`,O.standardDeviation,`float32`,Q)}}));let I=g.track(await P.compileAsync()),Ne=g.createBuffer(`variogram-overflow`,4),L=new _(n,{id:`rainfall-variogram`});L.add(new We({spatialContext:r.spatialContext,id:`variogram`,positions:M(L,`positions`,v,`float32x2`,c),values:M(L,`values`,y,`float32`,c),mask:M(L,`mask`,b,`uint32`,c),parameters:me.importToGraph(L),gridSize:at,lagCount:rt,directionCount:it,output:{semivariances:M(L,`semivariances`,k,`float32`,$),robustSemivariances:M(L,`robust`,A,`float32`,$),pairCounts:M(L,`pair-counts`,ne,`uint32`,$),meanDistances:M(L,`mean-distances`,re,`float32`,$),statistics:M(L,`statistics`,j,`float32`,5),status:{overflow:M(L,`variogram-overflow`,Ne,`uint32`,1)}}}));let Pe=g.track(await L.compileAsync()),R=!1,Fe=!1,Ie=!1,z=!0,B=!0,Re=1,V=0,Be=0,Ve=performance.now(),H=!0,U=[...m],Ue=[...m],W=null,G=null,Ge=new Uint8Array(c),Ke=[],K=null,qe=[...m],Xe=``,q=()=>{Ve=performance.now(),H=!0};function J(){return l[e.options.variable]}function Y(){Ge=new Uint8Array(c),Ke=[];let t=J();Be=0;for(let n=0;n<c;n++){let r=Number.isFinite(t[n]);e.options.holdOut&&r&&ve(n*31+Re)%ot===0?(Ge[n]=1,Ke.push(n)):r&&Be++}V=Ke.length,b.write(Uint32Array.from(Ge,e=>e?0:1)),x.write(Uint32Array.from(Ke)),e.setReadout(`stations`,`${S(c)} gauges`),e.setReadout(`training`,S(Be)),e.setReadout(`heldOut`,e.options.holdOut?S(V):`none`)}function ut(){let e=W?.statistics[2];return e&&e>0?e:1}function dt(){let t=e.options,n=t.searchRadiusKm*1e3;ae.write(Se({searchRadius:n,power:t.power,neighborCount:t.neighborCount,minimumNeighborCount:t.minimumStations},Qe(U)));let r=ut(),i=G??{model:t.variogramModel,nugget:.1*r,sill:.9*r,range:15e4};oe.write(Me({searchRadius:n,neighborCount:t.krigingNeighborCount,minimumNeighborCount:Math.max(t.minimumStations,3),variogram:{model:i.model,nugget:Math.max(i.nugget,1e-4*r),sill:Math.max(i.sill,1e-4*r),range:Math.max(i.range,1e3)}},Qe(U))),fe.write(Ce({radius:t.focalRadius,shape:t.focalShape,minimumCount:t.focalMinimumCount,propagateCenterNoData:t.keepHoles})),xe.write(Float32Array.of(t.surface===`kriging`?1:t.surface===`difference`?2:0,t.surface===`support`?1:0,0,0))}function ft(){me.write(He({bounds:h,maximumDistance:e.options.maximumLagKm*1e3,azimuthOffset:e.options.azimuth*Math.PI/180}))}let pt=new he(g,`rainfall-variogram`,[{buffer:k,size:$*4},{buffer:A,size:$*4},{buffer:ne,size:$*4},{buffer:re,size:$*4},{buffer:j,size:20},{buffer:Ne,size:4}],e=>{let t=new Float32Array(e),n=new Uint32Array(e),r=e=>{let r={distances:[],gammas:[],robust:[],pairs:[]};for(let i=0;i<rt;i++){let a=e*rt+i;r.gammas.push(t[a]),r.robust.push(t[$+a]),r.pairs.push(n[2*$+a]),r.distances.push(t[3*$+a])}return r},i=Array.from({length:it},(e,t)=>r(t)),a={distances:[],gammas:[],robust:[],pairs:[]};for(let e=0;e<rt;e++){let t=0,n=0,r=0,o=0;for(let a of i){let i=a.pairs[e];i>0&&Number.isFinite(a.gammas[e])&&(t+=i,n+=i*a.gammas[e],r+=i*a.robust[e],o+=i*a.distances[e])}a.pairs.push(t),a.gammas.push(t>0?n/t:NaN),a.robust.push(t>0?r/t:NaN),a.distances.push(t>0?o/t:NaN)}Fe=n[4*$+5]!==0,W={sectors:i,omni:a,statistics:t.slice(4*$,4*$+5)},ht()});function mt(t){if(Fe)return null;try{return Ye({distances:t.distances,semivariances:e.options.robust?t.robust:t.gammas,pairCounts:t.pairs},{model:e.options.variogramModel,weighting:e.options.weighting})}catch{return null}}function ht(){if(!W||R)return;let t=e.options,n=be[t.variable].squaredUnit;G=mt(W.omni);let r=t.robust?W.omni.robust:W.omni.gammas,i=W.omni.distances.map(e=>G?Je(G,e):NaN),a=Math.max(...r.filter(Number.isFinite),...i.filter(Number.isFinite),1e-9);e.setReadout(`empirical`,_e(r,a)),e.setReadout(`fitted`,_e(i,a)),e.setReadout(`lagAxis`,`0 to ${t.maximumLagKm} km in ${rt} lags; top of the chart ${N(a)} ${n}`),e.setReadout(`variogramModel`,Fe?`pair counts saturated (overflow): not fitted`:G?`nugget ${N(G.nugget)} · partial sill ${N(G.sill)} ${n} · range ${N(G.range/1e3)} km`:`not fitted`),e.setReadout(`sampleVariance`,`${N(W.statistics[2])} ${n}`),e.setChart(`variogramChart`,{kind:`line`,title:`Empirical and fitted semivariogram`,xLabel:`Gauge separation (km)`,yLabel:`Semivariance (${n})`,series:[{label:t.robust?`Robust empirical`:`Empirical`,x:W.omni.distances.map(e=>e/1e3),y:r,points:!0},{label:`${t.variogramModel} fit`,x:W.omni.distances.map(e=>e/1e3),y:i,dashed:!0}],guides:[{y:W.statistics[2],label:`sample variance`}],markers:G&&Number.isFinite(G.range)?[{x:G.range/1e3,label:`range ${N(G.range/1e3)} km`}]:void 0,description:`GPU-binned semivariance by gauge separation, the fitted variogram model, and the sample-variance reference.`}),gt(),z=!0,q(),e.requestLayers()}function gt(){if(!W)return;let t=e.options;if(t.direction===`all`){e.setReadout(`sectorRange`,`omnidirectional (choose a sector to compare directions)`);return}let n=Number(t.direction),r=(t.azimuth+n*180/it)%180,i=r+180/it,a=mt(W.sectors[n]);e.setReadout(`sectorRange`,a?`${r.toFixed(0)}° to ${i.toFixed(0)}° from east: range ${N(a.range/1e3)} km, sill ${N(a.nugget+a.sill)} (omnidirectional ${N((G?.range??NaN)/1e3)} km)`:`${r.toFixed(0)}° to ${i.toFixed(0)}° from east: too few pairs`)}let _t=new he(g,`rainfall-surfaces`,[{buffer:C,size:Q*4},{buffer:T,size:Q*4},{buffer:E,size:Q*4}],e=>{K={extent:qe,idw:new Float32Array(e,0,Q),kriging:new Float32Array(e,Q*4,Q),variance:new Float32Array(e,Q*8,Q)},yt()});function vt(e,t,n,r){let i=(t[2]-t[0])/X,a=(t[3]-t[1])/Z,o=(n-t[0])/i-.5,s=(r-t[1])/a-.5;if(o<-.5||s<-.5||o>X-.5||s>Z-.5)return NaN;let c=Math.max(0,Math.min(X-2,Math.floor(o))),l=Math.max(0,Math.min(Z-2,Math.floor(s))),u=Math.max(0,Math.min(1,o-c)),d=Math.max(0,Math.min(1,s-l)),f=0,p=0;for(let[t,n,r]of[[0,0,(1-u)*(1-d)],[1,0,u*(1-d)],[0,1,(1-u)*d],[1,1,u*d]]){let i=e[(l+n)*X+c+t];Number.isFinite(i)&&r>0&&(f+=r*i,p+=r)}return p>.5?f/p:NaN}function yt(){if(!K||!e.options.holdOut){for(let t of[`validationIdw`,`validationKriging`,`validationVerdict`])e.setReadout(t,e.options.holdOut?`...`:`turn on "Hold out 10% of gauges"`);e.setChart(`validationChart`,null);return}let t=J(),n=be[e.options.variable].unit,r={idw:{sum:0,abs:0,bias:0,n:0},kriging:{sum:0,abs:0,bias:0,n:0}};for(let e of Ke){let n=s[e*2],i=s[e*2+1];for(let[a,o]of[[`idw`,K.idw],[`kriging`,K.kriging]]){let s=vt(o,K.extent,n,i);if(!Number.isFinite(s))continue;let c=s-t[e];r[a].sum+=c*c,r[a].abs+=Math.abs(c),r[a].bias+=c,r[a].n++}}let i=e=>e.n>0?`RMSE ${N(Math.sqrt(e.sum/e.n))} ${n} · MAE ${N(e.abs/e.n)} · bias ${N(e.bias/e.n)} (n=${e.n})`:`no held-out gauge on screen`;if(e.setReadout(`validationIdw`,i(r.idw)),e.setReadout(`validationKriging`,i(r.kriging)),r.idw.n>0&&r.kriging.n>0){let t={rmse:Math.sqrt(r.idw.sum/r.idw.n),mae:r.idw.abs/r.idw.n,bias:r.idw.bias/r.idw.n},i={rmse:Math.sqrt(r.kriging.sum/r.kriging.n),mae:r.kriging.abs/r.kriging.n,bias:r.kriging.bias/r.kriging.n};e.setChart(`validationChart`,{kind:`dumbbell`,title:`Held-out prediction errors`,xLabel:n,aLabel:`IDW (n=${r.idw.n})`,bLabel:`Kriging (n=${r.kriging.n})`,rows:[{label:`RMSE`,a:t.rmse,b:i.rmse,highlight:!0},{label:`MAE`,a:t.mae,b:i.mae},{label:`Bias`,a:t.bias,b:i.bias}],description:`IDW and kriging error metrics at held-out gauges inside the current raster extent; lower RMSE and MAE are better, while bias is best near zero.`})}else e.setChart(`validationChart`,null);if(r.idw.n>5&&r.kriging.n>5){let t=Math.sqrt(r.idw.sum/r.idw.n),n=(Math.sqrt(r.kriging.sum/r.kriging.n)-t)/t*100;e.setReadout(`validationVerdict`,`kriging RMSE is ${Math.abs(n).toFixed(1)}% ${n<=0?`lower`:`higher`} than IDW at the held-out gauges on screen`)}else e.setReadout(`validationVerdict`,`zoom out so more held-out gauges are visible`)}async function bt(){if(!(Ie||R)){Ie=!0,e.setReadout(`graphTime`,`measuring...`);try{let t={parameters:void 0,completionBuffer:D,signal:e.signal},r=await de(n,I,t),i=await de(n,Pe,{...t,completionBuffer:j});if(R)return;e.setReadout(`graphTime`,`${I.stats.nodeOrder.length} nodes · ${pe(r)}`),e.setReadout(`variogramTime`,`${Pe.stats.nodeOrder.length} nodes · ${pe(i)}`)}catch{R||e.setReadout(`graphTime`,`interrupted`)}finally{Ie=!1}}}Y(),ft(),e.setReadout(`empirical`,`...`),e.setReadout(`fitted`,`...`);function xt(t){let[n,i]=r.project(t[0],t[1]),a=e.getViewport()?.zoom??5,o=ct*(78271.51696*Math.cos(t[1]*Math.PI/180)/2**a),l=J(),d=be[e.options.variable],m=-1,h=o*o;for(let e=0;e<c;e++){if(!Number.isFinite(l[e]))continue;let t=s[e*2]-n,r=s[e*2+1]-i,a=t*t+r*r;a<h&&(h=a,m=e)}if(m>=0){let e=p[m]?`, ${p[m]}`:``;return`${f[m]}${e}\n${N(l[m])} ${d.unit} observed · elevation ${N(u[m],4)} m${Ge[m]?`
held out: not used by the surfaces`:``}`}if(!K)return null;let g=vt(K.idw,K.extent,n,i),_=vt(K.kriging,K.extent,n,i),v=vt(K.variance,K.extent,n,i);if(!Number.isFinite(g)&&!Number.isFinite(_))return null;let y=[`IDW ${N(g)} ${d.unit}`];return Number.isFinite(_)&&y.push(`kriging ${N(_)} ± ${N(Math.sqrt(Math.max(v,0)))} ${d.unit}`),y.join(`
`)}return{getCompiledGraphs:()=>[I,Pe],setOption(t,n,r){switch(t){case`variable`:y.write(J()),Y(),B=!0,z=!0;break;case`holdOut`:Y(),B=!0,z=!0;break;case`variogramModel`:case`weighting`:case`robust`:ht();break;case`maximumLagKm`:case`azimuth`:ft(),B=!0;break;case`direction`:gt();break;case`ramp`:case`opacity`:case`showStations`:case`focalStatistic`:break;default:z=!0}q(),e.requestLayers()},onAction(t){t===`reseed`?(Re++,Y(),B=!0,z=!0,q(),e.requestLayers()):t===`measure`&&bt()},onThemeChange(){e.requestLayers()},getTooltip:e=>e.coordinate?xt(e.coordinate):null,encode(e,t){St(e,t.viewport)},getLayers(){return Ct()},destroy(){R=!0,pt.stop(),_t.stop(),g.destroy()}};function St(t,n){let i=a(n,r);i.some((e,t)=>e!==U[t])&&(U=i,z=!0,q(),e.setReadout(`cellSize`,`${N((U[2]-U[0])/X/1e3)} km × ${N((U[3]-U[1])/Z/1e3)} km`)),B&&(B=!1,Pe.encode(t,{parameters:void 0}),pt.request(t)),pt.flush(t),z&&(z=!1,dt(),I.encode(t,{parameters:void 0}),Ue=[...U]),H&&performance.now()-Ve>st?_t.isPending||(qe=[...Ue],_t.request(t),H=!1):_t.flush(t)}function Ct(){let t=e.options,n=ye(t),r=be[t.variable],i=e.theme()===`dark`,a=t.surface===`error`||t.surface===`support`,o=n.range;if(!o){let e=G?G.nugget+G.sill:ut();o=[0,Math.sqrt(Math.max(e,1e-9))]}let s=`${o[0]}:${o[1]}`;s!==Xe&&(Xe=s,e.setLegendExtent(`surface`,o));let l=t.focalRadius>0?t.focalStatistic:`mean`,u=[new Ze({id:`rainfall-${t.surface}-${l}`,analysisFrame:lt,gridSize:[X,Z],bounds:ae.buffer,rowOrigin:`south`,values:a?D:O[l],valueFormat:`float32`,colormap:n.ramp,valueRange:o,noDataColor:[0,0,0,0],color:[255,255,255,Math.round(255*t.opacity)]})];if(t.showStations){let e=t.ramp,n=r.range;u.push(new le({id:`rainfall-station-halo`,analysisFrame:lt,positions:v,values:y,valueFormat:`float32`,colormap:`grayscale`,valueRange:i?[-1e30,-1e29]:[1e30,1e31],noDataColor:[0,0,0,0],instanceCount:c,radiusPixels:2.6,color:[255,255,255,i?130:190]}),new le({id:`rainfall-stations`,analysisFrame:lt,positions:v,values:y,valueFormat:`float32`,colormap:e,valueRange:n,noDataColor:[0,0,0,0],instanceCount:c,radiusPixels:1.7,color:[255,255,255,255]})),t.holdOut&&V>0&&u.push(new le({id:`rainfall-held-out-ring`,analysisFrame:lt,positions:v,ids:x,instanceCount:V,radiusPixels:5.4,color:i?[255,255,255,255]:[20,20,20,255]}),new le({id:`rainfall-held-out`,analysisFrame:lt,positions:v,ids:x,values:y,valueFormat:`float32`,colormap:e,valueRange:n,noDataColor:[0,0,0,0],instanceCount:V,radiusPixels:3.6,color:[255,255,255,255]}))}return u}}export{ut as createRainfallInterpolation};