import{h as e}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as t,_ as n,l as r,t as i,v as a,w as o,y as s}from"./resources-CwvhEOBe.js";import{a as c,n as l,o as u,r as d,s as f,t as p}from"./gpu-terrain-curvature-C9i5TwO5.js";import{t as m}from"./gpu-reduction-BD8l40jM.js";import{l as h,s as g}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as _,c as v,i as y,n as b,o as x,r as S,s as C,t as w,u as T}from"./raster-band-utils-CA-sh8xt.js";import{a as E,c as ee,l as D,s as O,t as k}from"./raster-grid-D2bmCwkc.js";import{a as A,c as te,r as ne}from"./index-Bjk18V2I.js";import{t as j}from"./graph-buffers-B4_80RlX.js";import{t as re}from"./graph-swap-41Ru7LVu.js";import{t as M}from"./mode-kernels-Dxm2GWvI.js";import"./anchors-D9E9ucGL.js";import{h as ie}from"./gazetteer-OhisynFF.js";import{a as ae}from"./class-table-BVEo3-Vs.js";import{i as oe}from"./hue-registry-4FjIJ9xz.js";import{c as se,n as ce,r as le,s as ue}from"./live-text-83XBWyr1.js";import"./picking-DRmnefrA.js";import{n as N,t as P}from"./terrain-geomorphometry-utils-BlzhWTj6.js";import{a as de,i as fe,t as pe}from"./b14a-session-CMNZQd9U.js";import{n as me,t as he}from"./cpu-dem-DnM3p94o.js";import{a as ge,b as _e,f as ve,p as ye}from"./terrain-furniture-C6hPKNNK.js";import{a as be,i as xe,l as Se,r as Ce,u as F}from"./terrain-places-CXVUagwV.js";import{n as we,t as Te}from"./terrain-ground-BLUqElOt.js";import{C as Ee,S as De,_ as Oe,a as ke,c as I,d as Ae,f as je,h as Me,i as Ne,o as L,s as Pe,t as R,u as Fe,w as Ie,x as Le,y as Re}from"./landforms.style-Cxui8Cu1.js";function ze(e,t,n=new Float32Array(8)){return y(`getGPUGeomorphonsParameterValues`,n,8),N(e,t,n),n[5]=e.flatThresholdDegrees??1,n[6]=e.flatDistance??0,n[7]=0,n}var z=class{id;props;requiredHalo;constructor(e){this.id=e.id??`geomorphons`,this.props=e;let{id:t}=this,{output:n}=e,r=C(t,e.grid),i=e.skipRadius??0;if(!Number.isSafeInteger(e.searchRadius)||e.searchRadius<1)throw Error(`${t} searchRadius must be a positive integer`);if(!Number.isSafeInteger(i)||i<0)throw Error(`${t} skipRadius must be a non-negative integer`);if(e.searchRadius<i+2)throw Error(`${t} searchRadius must be at least skipRadius + 2`);if(![`anglev1`,`anglev2`,`anglev2-distance`].includes(e.comparison??`anglev1`))throw Error(`${t} comparison must be anglev1, anglev2, or anglev2-distance`);if(!n.forms&&!n.ternary&&!n.pattern&&!n.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,i]of[[`forms`,n.forms],[`ternary`,n.ternary],[`pattern`,n.pattern],[`outputValidity`,n.outputValidity]])if(i&&(s(i,[`uint32`],`${t} output.${e}`),i.length!==r))throw Error(`${t} output.${e} must contain one value per pixel`);T(t,e.parameters,8),ee(t,e.cellSizeMode),v(t,e.grid,e.cellSizeMode),D(t,e.rowDirection),x(t,[n.forms,n.ternary,n.pattern,n.outputValidity],[...b(e.elevation),e.parameters]),this.requiredHalo=e.searchRadius}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r,{width:a,height:o}=r.grid;_(t,e,r.elevation,[]),n(t,e,[r.parameters,i.forms,i.ternary,i.pattern,i.outputValidity]);let s=S(e,t,r.elevation,r.grid,!0),c=s.band.validity;if(!c||s.band.storage.kind!==`buffer`)throw Error(`${t} could not canonicalize elevation`);let l=[{name:`elevation`,view:s.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:c,type:`u32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`}];for(let[e,t]of[[`formValues`,i.forms],[`ternaryValues`,i.ternary],[`patternValues`,i.pattern],[`validityValues`,i.outputValidity]])t&&l.push({name:e,view:t,type:`u32`,access:`read_write`});let u=r.comparison??`anglev1`,d=r.skipRadius??0,f=r.cellSizeMode??`uniform`,p=r.rowDirection??`south`;return[...s.nodes,g(e,{id:`${t}-classify`,operation:`GPUGeomorphons`,variant:`${u}-${f}-skip${d}-radius${r.searchRadius}`,bindings:l,invocationCount:a*o,declarations:Be(r,u,d,f,p),body:Ve(r)})]}};function Be(e,t,n,r,i){return`
const WIDTH: u32 = ${e.grid.width}u;
const HEIGHT: u32 = ${e.grid.height}u;
const SEARCH_RADIUS: i32 = ${e.searchRadius};
const SKIP_RADIUS: i32 = ${n};
const COMPARISON: u32 = ${t===`anglev1`?0:t===`anglev2`?1:2}u;
${O(i,`i32`)}
${k}
${w}
${E(r,P)}

// Directions in GRASS order: NE, N, NW, W, SW, S, SE, E.
const DIRECTION_COLUMN = array<i32, 8>(1, 0, -1, -1, -1, 0, 1, 1);
const DIRECTION_NORTH = array<i32, 8>(1, 1, 1, 0, -1, -1, -1, 0);
// Landform codes by minus count (row) and plus count (column); 0 marks impossible pairs.
const FORM_TABLE = array<u32, 81>(
  1u, 1u, 1u, 8u, 8u, 9u, 9u, 9u, 10u,
  1u, 1u, 8u, 8u, 8u, 9u, 9u, 9u, 0u,
  1u, 4u, 6u, 6u, 7u, 7u, 9u, 0u, 0u,
  4u, 4u, 6u, 6u, 6u, 7u, 0u, 0u, 0u,
  4u, 4u, 5u, 6u, 6u, 0u, 0u, 0u, 0u,
  3u, 3u, 5u, 5u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u,
  3u, 3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
  2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

fn isElevationValid(column: i32, row: i32) -> bool {
  // Canonical elevation is NaN exactly where a pixel is invalid, so no validity load is needed.
  return isFiniteValue(elevation[elevationOffset + u32(row) * WIDTH + u32(column)]);
}

fn getElevation(column: i32, row: i32) -> f32 {
  return elevation[elevationOffset + u32(row) * WIDTH + u32(column)];
}

// Re-encodes the digits cyclically shifted by shift, optionally mirrored, in base 3.
fn encodeRotation(digits: array<u32, 8>, shift: u32, mirrored: bool) -> u32 {
  var code = 0u;
  var power = 1u;
  for (var position = 0u; position < 8u; position++) {
    let source = (position + 8u - shift) % 8u;
    let digit = select(digits[source], digits[7u - source], mirrored);
    code += digit * power;
    power *= 3u;
  }
  return code;
}

// Reduces one line of sight, given zenith/nadir heights and step counts, to a digit in {-1, 0, 1}.
fn getPatternDigit(
  zenithHeight: f32, zenithCount: i32, nadirHeight: f32, nadirCount: i32,
  tangent: f32, flatDistance: f32, step: f32
) -> i32 {
  let zenithDistance = f32(zenithCount) * step;
  let nadirDistance = f32(nadirCount) * step;
  let zenithReference = select(zenithDistance, flatDistance, flatDistance > 0.0 && flatDistance < zenithDistance);
  let nadirReference = select(nadirDistance, flatDistance, flatDistance > 0.0 && flatDistance < nadirDistance);
  let zenithAbsolute = abs(zenithHeight);
  let nadirAbsolute = abs(nadirHeight);
  let zenithOver = zenithAbsolute > tangent * zenithReference;
  let nadirOver = nadirAbsolute > tangent * nadirReference;
  // Absolute angles compare as |height| / count, cross-multiplied to stay division free.
  let zenithScaled = zenithAbsolute * f32(nadirCount);
  let nadirScaled = nadirAbsolute * f32(zenithCount);
  if (COMPARISON == 0u) {
    if (!(zenithOver || nadirOver)) { return 0; }
    if (nadirScaled < zenithScaled) { return 1; }
    if (nadirScaled > zenithScaled) { return -1; }
    return 0;
  }
  if (!zenithOver && !nadirOver) { return 0; }
  if (zenithOver && !nadirOver) { return 1; }
  if (nadirOver && !zenithOver) { return -1; }
  if (nadirScaled < zenithScaled) { return 1; }
  if (nadirScaled > zenithScaled) { return -1; }
  if (COMPARISON == 2u) {
    if (nadirCount < zenithCount) { return 1; }
    if (nadirCount > zenithCount) { return -1; }
  }
  return 1;
}`}function Ve(e){return`
  let row = i32(index / WIDTH);
  let column = i32(index % WIDTH);
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  let flatThreshold = settings[settingsOffset + 5u];
  let flatDistance = settings[settingsOffset + 6u];
  let settingsValid = isFiniteValue(groundCell.x) && isFiniteValue(groundCell.y) &&
    groundCell.x > 0.0 && groundCell.y > 0.0 && isFiniteValue(zFactor) &&
    isFiniteValue(flatThreshold) && flatThreshold > 0.0 && flatThreshold < 90.0 &&
    isFiniteValue(flatDistance) && flatDistance >= 0.0;
  let margin = SKIP_RADIUS + 1;
  let insideBand = row >= margin && row < i32(HEIGHT) - margin &&
    column >= margin && column < i32(WIDTH) - margin;
  let isValid = settingsValid && insideBand && isElevationValid(column, row);
  var outputForm = 0u;
  var outputTernary = 0u;
  var outputPattern = 0u;
  if (isValid) {
    let centerElevation = getElevation(column, row);
    let tangent = tan(flatThreshold * DEGREES_TO_RADIANS);
    // The strict search limit is shrunk by 1e-5 so a sample exactly at the limit is excluded even
    // when f32 cell sizes round (for example cos(0) or equal east-west and north-south sizes).
    let searchDistance = f32(SEARCH_RADIUS) * groundCell.y * 0.99999;
    // GRASS r.geomorphon ignores flat_distance at or inside the skip distance and at or beyond
    // the search distance (review C-13).
    let skipDistance = f32(SKIP_RADIUS) * groundCell.y;
    let flatReference = select(
      0.0,
      flatDistance,
      flatDistance > skipDistance && flatDistance < f32(SEARCH_RADIUS) * groundCell.y
    );
    var digits: array<u32, 8>;
    var minusCount = 0u;
    var plusCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      digits[direction] = 1u;
      let columnStep = DIRECTION_COLUMN[direction];
      let northStep = DIRECTION_NORTH[direction];
      let rowStep = northStep * ROW_NORTH_SIGN;
      if (!isElevationValid(column + columnStep, row + rowStep)) { continue; }
      let eastMetres = f32(columnStep) * groundCell.x;
      let northMetres = f32(northStep) * groundCell.y;
      // Axis directions use the exact cell size so the search limit has no square-root rounding.
      var step = sqrt(eastMetres * eastMetres + northMetres * northMetres);
      if (columnStep == 0) { step = groundCell.y; }
      if (northStep == 0) { step = groundCell.x; }
      var zenithHeight = 0.0;
      var zenithCount = 0;
      var nadirHeight = 0.0;
      var nadirCount = 0;
      for (var count = SKIP_RADIUS + 1; count <= i32(max(WIDTH, HEIGHT)); count++) {
        if (!(f32(count) * step < searchDistance)) { break; }
        let sampleColumn = column + count * columnStep;
        let sampleRow = row + count * rowStep;
        if (sampleColumn < 0 || sampleColumn >= i32(WIDTH) || sampleRow < 0 || sampleRow >= i32(HEIGHT)) { break; }
        let sampleElevation = getElevation(sampleColumn, sampleRow);
        if (!isFiniteValue(sampleElevation)) { continue; }
        let height = zFactor * (sampleElevation - centerElevation);
        if (zenithCount == 0 || height * f32(zenithCount) > zenithHeight * f32(count)) {
          zenithHeight = height;
          zenithCount = count;
        }
        if (nadirCount == 0 || height * f32(nadirCount) < nadirHeight * f32(count)) {
          nadirHeight = height;
          nadirCount = count;
        }
      }
      if (zenithCount == 0) { continue; }
      let digit = getPatternDigit(
        zenithHeight, zenithCount, nadirHeight, nadirCount, tangent, flatReference, step
      );
      digits[direction] = u32(digit + 1);
      if (digit > 0) { plusCount++; }
      if (digit < 0) { minusCount++; }
    }
    var code = 0u;
    var power = 1u;
    var rotated = 0xffffffffu;
    for (var position = 0u; position < 8u; position++) {
      code += digits[position] * power;
      power *= 3u;
      rotated = min(rotated, encodeRotation(digits, position, false));
      rotated = min(rotated, encodeRotation(digits, position, true));
    }
    outputPattern = code;
    outputTernary = rotated;
    outputForm = FORM_TABLE[minusCount * 9u + plusCount];
  }
  ${[e.output.forms?`formValues[formValuesOffset + index] = outputForm;`:``,e.output.ternary?`ternaryValues[ternaryValuesOffset + index] = outputTernary;`:``,e.output.pattern?`patternValues[patternValuesOffset + index] = outputPattern;`:``,e.output.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``].join(`
  `)}`}function B(e){return r(e,4)}var V=class{id;props;requiredHalo;constructor(e){this.id=e.id??`terrain-topographic-position`,this.props=e;let{id:t}=this,{output:n}=e,r=C(t,e.grid),{scales:i}=e;if(!Array.isArray(i)||i.length<1||i.length>64)throw Error(`${t} scales must contain 1 to 64 entries`);for(let e of i){let n=e.innerRadius??0;if(!Number.isSafeInteger(e.radius)||e.radius<1||!Number.isSafeInteger(n)||n<0||n>=e.radius)throw Error(`${t} scales need integer radius >= 1 and 0 <= innerRadius < radius`)}let a=e.quantum??.00390625;if(!(a>0)||!Number.isFinite(a)||2**Math.round(Math.log2(a))!==a)throw Error(`${t} quantum must be a positive power of two`);if(this.requiredHalo=Math.max(...i.map(e=>e.radius)),!n.topographicPositionIndex&&!n.deviationFromMean&&!n.maximumDeviation&&!n.maximumDeviationRadius&&!n.outputValidity)throw Error(`${t} requires at least one output`);for(let[e,a,o]of[[`topographicPositionIndex`,n.topographicPositionIndex,r*i.length],[`deviationFromMean`,n.deviationFromMean,r*i.length],[`maximumDeviation`,n.maximumDeviation,r]])if(a&&(s(a,[`float32`],`${t} output.${e}`),a.length!==o))throw Error(`${t} output.${e} must contain ${o} values`);for(let[e,i]of[[`maximumDeviationRadius`,n.maximumDeviationRadius],[`outputValidity`,n.outputValidity]])if(i&&(s(i,[`uint32`],`${t} output.${e}`),i.length!==r))throw Error(`${t} output.${e} must contain one value per pixel`);x(t,[n.topographicPositionIndex,n.deviationFromMean,n.maximumDeviation,n.maximumDeviationRadius,n.outputValidity],b(e.elevation))}getCommandNodes(e){let{id:t,props:r}=this,{output:i,scales:o}=r,{width:s,height:l}=r.grid;_(t,e,r.elevation,[]),n(t,e,[i.topographicPositionIndex,i.deviationFromMean,i.maximumDeviation,i.maximumDeviationRadius,i.outputValidity]);let p=s*l;a(t,`grid cells`,p,f(e.device)),(i.topographicPositionIndex||i.deviationFromMean)&&a(t,`grid cells x scales`,p*r.scales.length,B(e.device));let m=r.quantum??.00390625,v=S(e,t,r.elevation,r.grid,!0),y=u(e,{id:t,width:s,height:l,elevation:v.band,quantum:m}),b=[{name:`lowTable`,view:y.lowTable,type:`u32`,access:`read`},{name:`highTable`,view:y.highTable,type:`u32`,access:`read`}],x=[[`tpiValues`,i.topographicPositionIndex,`f32`],[`deviationValues`,i.deviationFromMean,`f32`],[`maximumDeviationValues`,i.maximumDeviation,`f32`],[`maximumRadiusValues`,i.maximumDeviationRadius,`u32`],[`validityValues`,i.outputValidity,`u32`]];for(let[e,t,n]of x)t&&b.push({name:e,view:t,type:n,access:`read_write`});let C=o.map(e=>`vec2<i32>(${e.radius}i, ${e.innerRadius??0}i)`).join(`, `),w=g(e,{id:`${t}-evaluate`,operation:`GPUTerrainTopographicPosition`,variant:`evaluate`,bindings:b,invocationCount:p,declarations:`const WIDTH: u32 = ${s}u;
const HEIGHT: u32 = ${l}u;
const PIXEL_COUNT: u32 = ${p}u;
const SCALE_COUNT: u32 = ${o.length}u;
const QUANTUM: f32 = ${h(m)};
const INVERSE_QUANTUM: f32 = ${h(1/m)};
const SCALES = array<vec2<i32>, ${o.length}>(${C});
${c}
${d}
fn readClippedBox(column: i32, row: i32, radius: i32) -> TerrainBoxSums {
  return readBoxSums(
    max(column - radius, 0i),
    min(column + radius, i32(WIDTH) - 1i),
    max(row - radius, 0i),
    min(row + radius, i32(HEIGHT) - 1i)
  );
}
// Σ(q - q_c) over a box, exact modulo 2^64.
fn getCentredSum(box: TerrainBoxSums, magnitude: u32, isNegative: bool) -> vec2<u32> {
  var centreTotal = multiply64By32(vec2<u32>(box.count, 0u), magnitude);
  if (isNegative) {
    centreTotal = negate64(centreTotal);
  }
  return subtract64(box.sum, centreTotal);
}`,body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  // A single-cell box returns the centre's quantized elevation and validity exactly.
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let quantized = bitcast<i32>(centre.sum.x);
  let magnitude = u32(abs(quantized));
  let isNegative = quantized < 0i;
  let centreSquare = multiplyWide32(magnitude, magnitude);
  var maximumDeviation = invalidValue;
  var maximumAbsolute = -1.0;
  var maximumRadius = 0u;
  for (var scaleIndex = 0u; scaleIndex < SCALE_COUNT; scaleIndex++) {
    let scale = SCALES[scaleIndex];
    let outer = readClippedBox(column, row, scale.x);
    // Population statistics relative to the centre: d1 = Σ(q - q_c), d2 = Σ(q - q_c)².
    let centredSum = getCentredSum(outer, magnitude, isNegative);
    var twiceCentreSum = multiply64By32(outer.sum, magnitude);
    twiceCentreSum = add64(twiceCentreSum, twiceCentreSum);
    if (isNegative) {
      twiceCentreSum = negate64(twiceCentreSum);
    }
    let centredSquare = add64(
      subtract64(outer.square, twiceCentreSum),
      multiply64By32(centreSquare, outer.count)
    );
    let count = f32(outer.count);
    let meanOffset = signed64ToFloat(centredSum) / count;
    // Review C-16: d2 / n - (d1 / n)^2 in f32 cancels when the centre is far from the window mean.
    // Re-centre exactly on the integer s nearest that mean: sum (d - s)^2 = d2 - 2 s d1 + n s^2 in
    // modular 64-bit integers, so only the residual mean (|r| about 0.5 quanta) is squared in f32.
    let shiftValue = round(meanOffset);
    let shiftMagnitude = u32(abs(shiftValue));
    let shiftNegative = shiftValue < 0.0;
    var shiftedCross = multiply64By32(centredSum, shiftMagnitude);
    shiftedCross = add64(shiftedCross, shiftedCross);
    var shiftTotal = multiply64By32(vec2<u32>(outer.count, 0u), shiftMagnitude);
    if (shiftNegative) {
      shiftedCross = negate64(shiftedCross);
      shiftTotal = negate64(shiftTotal);
    }
    let shiftedSquare = add64(
      subtract64(centredSquare, shiftedCross),
      multiply64By32(multiplyWide32(shiftMagnitude, shiftMagnitude), outer.count)
    );
    let residualMean = signed64ToFloat(subtract64(centredSum, shiftTotal)) / count;
    let variance = unsigned64ToFloat(shiftedSquare) / count - residualMean * residualMean;
    var deviation = 0.0;
    if (variance > 0.0) {
      deviation = -meanOffset / sqrt(variance);
    }
    if (!isValid) {
      deviation = invalidValue;
    }
    ${i.deviationFromMean?`deviationValues[deviationValuesOffset + scaleIndex * PIXEL_COUNT + index] = deviation;`:``}
    if (isValid && abs(deviation) > maximumAbsolute) {
      maximumAbsolute = abs(deviation);
      maximumDeviation = deviation;
      maximumRadius = u32(scale.x);
    }
    ${i.topographicPositionIndex?`let inner = readClippedBox(column, row, scale.y);
    let annulus = TerrainBoxSums(
      subtract64(outer.sum, inner.sum),
      subtract64(outer.square, inner.square),
      outer.count - inner.count
    );
    var tpi = invalidValue;
    if (isValid && annulus.count > 0u) {
      tpi = -signed64ToFloat(getCentredSum(annulus, magnitude, isNegative)) / f32(annulus.count) * QUANTUM;
    }
    tpiValues[tpiValuesOffset + scaleIndex * PIXEL_COUNT + index] = tpi;`:``}
  }
  ${i.maximumDeviation?`maximumDeviationValues[maximumDeviationValuesOffset + index] = maximumDeviation;`:``}
  ${i.maximumDeviationRadius?`maximumRadiusValues[maximumRadiusValuesOffset + index] = maximumRadius;`:``}
  ${i.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`});return[...v.nodes,...y.nodes,w]}};function He(e,t,n=new Float32Array(8)){return y(`getGPUTerrainWeissLandformsParameterValues`,n,8),N(e,t,n),n[5]=e.standardThreshold??1,n[6]=e.slopeThresholdDegrees??5,n[7]=0,n}var Ue=class{id;props;requiredHalo;constructor(e){this.id=e.id??`terrain-weiss-landforms`,this.props=e;let{id:t}=this,{output:n}=e,r=C(t,e.grid),i=this.getScales();for(let e of i){let n=e.innerRadius??0;if(!Number.isSafeInteger(e.radius)||e.radius<1||!Number.isSafeInteger(n)||n<0||n>=e.radius)throw Error(`${t} scales need integer radius >= 1 and 0 <= innerRadius < radius`)}if(this.requiredHalo=Math.max(1,...i.map(e=>e.radius)),![`global`,`local`].includes(e.standardization??`global`))throw Error(`${t} standardization must be global or local`);if(ee(t,e.cellSizeMode),v(t,e.grid,e.cellSizeMode),T(t,e.parameters,8),!n.landforms&&!n.standardizedPosition&&!n.outputValidity)throw Error(`${t} requires at least one output`);if(n.standardizedPosition&&(s(n.standardizedPosition,[`float32`],`${t} output.standardizedPosition`),n.standardizedPosition.length!==2*r))throw Error(`${t} output.standardizedPosition must contain two values per pixel`);for(let[e,i]of[[`landforms`,n.landforms],[`outputValidity`,n.outputValidity]])if(i&&(s(i,[`uint32`],`${t} output.${e}`),i.length!==r))throw Error(`${t} output.${e} must contain one value per pixel`);x(t,[n.landforms,n.standardizedPosition,n.outputValidity],[...b(e.elevation),e.parameters])}getCommandNodes(e){let{id:t,props:r}=this,{output:i}=r,{width:s,height:c}=r.grid;_(t,e,r.elevation,[]),n(t,e,[r.parameters,i.landforms,i.standardizedPosition,i.outputValidity]);let l=s*c;a(t,`grid cells`,l,f(e.device));let u=(r.standardization??`global`)===`global`,d=S(e,t,r.elevation,r.grid,!0),p=d.band.validity;if(!p)throw Error(`${t} canonical elevation is missing validity`);let m=[...d.nodes],h=o(e,`${t}-position`,`float32`,2*l),v=o(e,`${t}-position-validity`,`uint32`,l);m.push(...new V({id:`${t}-position`,grid:r.grid,elevation:d.band,scales:this.getScales(),quantum:r.quantum,output:{...u?{topographicPositionIndex:h}:{deviationFromMean:h},outputValidity:v}}).getCommandNodes(e));let y=o(e,`${t}-slope`,`float32`,l),b=o(e,`${t}-mask`,`uint32`,l);m.push(g(e,{id:`${t}-slope`,operation:`GPUTerrainWeissLandforms`,variant:`slope-${r.cellSizeMode??`uniform`}`,bindings:[{name:`elevationValues`,view:d.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:p,type:`u32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`},{name:`positionValues`,view:h,type:`f32`,access:`read`},{name:`positionValidity`,view:v,type:`u32`,access:`read`},{name:`slopeValues`,view:y,type:`f32`,access:`read_write`},{name:`maskValues`,view:b,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const WIDTH: u32 = ${s}u;
const HEIGHT: u32 = ${c}u;
const PIXEL_COUNT: u32 = ${l}u;
${w}
${E(r.cellSizeMode??`uniform`,P)}
fn readElevation(column: i32, row: i32) -> vec2<f32> {
  let sampleColumn = u32(clamp(column, 0i, i32(WIDTH) - 1i));
  let sampleRow = u32(clamp(row, 0i, i32(HEIGHT) - 1i));
  let sampleIndex = sampleRow * WIDTH + sampleColumn;
  let value = elevationValues[elevationValuesOffset + sampleIndex];
  let isValid = elevationValidity[elevationValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(value, select(0.0, 1.0, isValid));
}`,body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let centre = readElevation(column, row);
  var isValid = centre.y != 0.0;
  // Horn weights over centre-relative differences, which are exact in float32.
  var eastSum = 0.0;
  var southSum = 0.0;
  for (var dy = -1i; dy <= 1i; dy++) {
    for (var dx = -1i; dx <= 1i; dx++) {
      if (dx == 0i && dy == 0i) {
        continue;
      }
      let sample = readElevation(column + dx, row + dy);
      isValid = isValid && sample.y != 0.0;
      let weight = select(1.0, 2.0, dx == 0i || dy == 0i);
      let difference = sample.x - centre.x;
      eastSum += f32(dx) * weight * difference;
      southSum += f32(dy) * weight * difference;
    }
  }
  let groundCell = getGroundCellSize(u32(row));
  let zFactor = settings[settingsOffset + 2u];
  // Only the gradient magnitude matters, so the row direction does not.
  let eastGradient = zFactor * eastSum / (8.0 * groundCell.x);
  let rowGradient = zFactor * southSum / (8.0 * groundCell.y);
  let slope = atan(length(vec2<f32>(eastGradient, rowGradient))) * 57.29577951308232;
  let small = positionValues[positionValuesOffset + index];
  let large = positionValues[positionValuesOffset + PIXEL_COUNT + index];
  isValid = isValid && isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    positionValidity[positionValidityOffset + index] != 0u;
  slopeValues[slopeValuesOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), slope, isValid);
  maskValues[maskValuesOffset + index] = select(0u, 1u, isValid);`}));let x=[{name:`positionValues`,view:h,type:`f32`,access:`read`},{name:`slopeValues`,view:y,type:`f32`,access:`read`},{name:`settings`,view:r.parameters,type:`f32`,access:`read`}];if(u){let t=this.addGlobalStatisticsNodes(e,m,h,b,l);x.push({name:`statisticsValues`,view:t.values,type:`f32`,access:`read`},{name:`statisticsCount`,view:t.count,type:`u32`,access:`read`})}for(let[e,t,n]of[[`landformValues`,i.landforms,`u32`],[`standardizedValues`,i.standardizedPosition,`f32`],[`validityValues`,i.outputValidity,`u32`]])t&&x.push({name:e,view:t,type:n,access:`read_write`});return m.push(g(e,{id:`${t}-classify`,operation:`GPUTerrainWeissLandforms`,variant:u?`classify-global`:`classify-local`,bindings:x,invocationCount:l,declarations:`const PIXEL_COUNT: u32 = ${l}u;
${w}
fn standardize(value: f32, scale: u32) -> f32 {
  ${u?`let count = f32(statisticsCount[statisticsCountOffset]);
  let mean = statisticsValues[statisticsValuesOffset + scale] / count;
  let deviation = sqrt(statisticsValues[statisticsValuesOffset + 2u + scale] / count);
  return select(0.0, (value - mean) / deviation, deviation > 0.0);`:`return value;`}
}
// 0 = low (at or below -threshold), 1 = mid, 2 = high (at or above threshold).
fn getPositionClass(value: f32, threshold: f32) -> u32 {
  if (value <= -threshold) {
    return 0u;
  }
  return select(1u, 2u, value >= threshold);
}`,body:`let slope = slopeValues[slopeValuesOffset + index];
  let threshold = settings[settingsOffset + 5u];
  let slopeThreshold = settings[settingsOffset + 6u];
  let small = standardize(positionValues[positionValuesOffset + index], 0u);
  let large = standardize(positionValues[positionValuesOffset + PIXEL_COUNT + index], 1u);
  let isValid = isFiniteValue(slope) && isFiniteValue(small) && isFiniteValue(large) &&
    isFiniteValue(threshold) && threshold > 0.0 && isFiniteValue(slopeThreshold);
  let smallClass = getPositionClass(small, threshold);
  let largeClass = getPositionClass(large, threshold);
  // Low small position: canyon 1, midslope drainage 2, upland drainage 3.
  var landform = 1u + largeClass;
  if (smallClass == 1u) {
    // Mid small position: valley 4, plain 5, open slope 6, upper slope 7.
    landform = select(select(4u, 7u, largeClass == 2u), select(6u, 5u, slope <= slopeThreshold), largeClass == 1u);
  } else if (smallClass == 2u) {
    // High small position: local ridge 8, midslope ridge 9, mountain top 10.
    landform = 8u + largeClass;
  }
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${i.landforms?`landformValues[landformValuesOffset + index] = select(0u, landform, isValid);`:``}
  ${i.standardizedPosition?`standardizedValues[standardizedValuesOffset + index] = select(invalidValue, small, isValid);
  standardizedValues[standardizedValuesOffset + PIXEL_COUNT + index] = select(invalidValue, large, isValid);`:``}
  ${i.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})),m}getScales(){return[this.props.smallScale??{radius:3},this.props.largeScale??{radius:15}]}addGlobalStatisticsNodes(e,t,n,r,i){let{id:a}=this,s=o(e,`${a}-statistics`,`float32`,4),c=o(e,`${a}-statistics-count`,`uint32`,1),l=t=>e.createDataView(n.buffer,{format:`float32`,length:i,byteOffset:n.byteOffset+t*i*4}),u=t=>e.createDataView(s.buffer,{format:`float32`,length:1,byteOffset:s.byteOffset+t*4});t.push(...new m({id:`${a}-count`,input:r,output:c,operation:`sum`}).getCommandNodes(e));for(let n of[0,1])t.push(...new m({id:`${a}-sum-${n}`,input:l(n),mask:r,output:u(n),operation:`sum`}).getCommandNodes(e));let d=o(e,`${a}-squares`,`float32`,2*i);t.push(g(e,{id:`${a}-squares`,operation:`GPUTerrainWeissLandforms`,variant:`centred-squares`,bindings:[{name:`positionValues`,view:n,type:`f32`,access:`read`},{name:`maskValues`,view:r,type:`u32`,access:`read`},{name:`statisticsValues`,view:s,type:`f32`,access:`read`},{name:`statisticsCount`,view:c,type:`u32`,access:`read`},{name:`squareValues`,view:d,type:`f32`,access:`read_write`}],invocationCount:2*i,declarations:`const PIXEL_COUNT: u32 = ${i}u;`,body:`let scale = index / PIXEL_COUNT;
  let pixel = index - scale * PIXEL_COUNT;
  let mean = statisticsValues[statisticsValuesOffset + scale] / f32(statisticsCount[statisticsCountOffset]);
  let difference = positionValues[positionValuesOffset + index] - mean;
  squareValues[squareValuesOffset + index] = select(0.0, difference * difference, maskValues[maskValuesOffset + pixel] != 0u);`}));for(let n of[0,1])t.push(...new m({id:`${a}-square-sum-${n}`,input:e.createDataView(d.buffer,{format:`float32`,length:i,byteOffset:d.byteOffset+n*i*4}),mask:r,output:u(2+n),operation:`sum`}).getCommandNodes(e));return{values:s,count:c}}},We=8,H=24,U=16,Ge=.12;function Ke(e,t,n){let r=new Float32Array(96),i=Math.min(U,t.colors.length),a=0;for(let e of n.isolate??[])e>=0&&e<U&&(a|=1<<e);r[0]=n.mode===`codes`?0:1,r[1]=i,r[2]=n.dilateRadius??0,r[3]=n.peakCode??0,r[4]=n.pitCode??0,r[5]=a,r[6]=Ge,t.breaks.slice(0,U-1).forEach((e,t)=>{r[We+t]=e});for(let e=0;e<i;e++){let n=t.colors[e];r[H+e*4]=n[0]/255,r[H+e*4+1]=n[1]/255,r[H+e*4+2]=n[2]/255,r[H+e*4+3]=(n[3]??255)/255}e.write(r)}async function qe(e,n,r,i,a,o){let s=i.width*i.height,c=new t(n,{id:`${r}-paint`}),l=i.format===`float32`;return M(c,{id:`${r}-table-paint`,invocationCount:s,bindings:[{name:`source`,view:j(c,`source`,i.buffer,i.format,s),type:l?`f32`:`u32`,access:`read`},{name:`paint`,view:a.importToGraph(c),type:`f32`,access:`read`},{name:`colors`,view:j(c,`colors`,o,`uint32`,s),type:`u32`,access:`read_write`}],declarations:`
const WIDTH: i32 = ${i.width};
const HEIGHT: i32 = ${i.height};
fn isBad(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn tableColor(classIndex: u32) -> vec4<f32> {
  let base = paintOffset + ${H}u + classIndex * 4u;
  return vec4<f32>(paint[base], paint[base + 1u], paint[base + 2u], paint[base + 3u]);
}`,body:`
  let classCount = u32(paint[paintOffset + 1u]);
  var classIndex = 0xffffffffu;
  ${l?`
  let value = source[sourceOffset + index];
  if (!isBad(value)) {
    classIndex = 0u;
    for (var breakIndex = 0u; breakIndex + 1u < classCount; breakIndex = breakIndex + 1u) {
      if (value >= paint[paintOffset + ${We}u + breakIndex]) {
        classIndex = breakIndex + 1u;
      }
    }
  }`:`
  let own = source[sourceOffset + index];
  classIndex = own;
  let reach = i32(paint[paintOffset + 2u]);
  let peakCode = u32(paint[paintOffset + 3u]);
  let pitCode = u32(paint[paintOffset + 4u]);
  if (reach > 0 && own != peakCode && own != pitCode) {
    let column = i32(index) % WIDTH;
    let row = i32(index) / WIDTH;
    var foundPeak = false;
    var foundPit = false;
    for (var rowOffset = -reach; rowOffset <= reach; rowOffset = rowOffset + 1) {
      for (var columnOffset = -reach; columnOffset <= reach; columnOffset = columnOffset + 1) {
        let neighbourColumn = column + columnOffset;
        let neighbourRow = row + rowOffset;
        if (neighbourColumn < 0 || neighbourRow < 0 || neighbourColumn >= WIDTH || neighbourRow >= HEIGHT) {
          continue;
        }
        let neighbour = source[sourceOffset + u32(neighbourRow * WIDTH + neighbourColumn)];
        foundPeak = foundPeak || neighbour == peakCode;
        foundPit = foundPit || neighbour == pitCode;
      }
    }
    if (foundPeak) {
      classIndex = peakCode;
    } else if (foundPit) {
      classIndex = pitCode;
    }
  }`}
  var color = vec4<f32>(0.0);
  if (classIndex < classCount) {
    color = tableColor(classIndex);
    let isolated = u32(paint[paintOffset + 5u]);
    if (isolated != 0u && ((isolated >> classIndex) & 1u) == 0u) {
      color.a = color.a * paint[paintOffset + 6u];
    }
  }
  colors[colorsOffset + index] = pack4x8unorm(color);`}),e.track(await c.compileAsync())}var W={4:1/4,64:1/64,256:1/256},Je=0;function G(e,t,n,r){let i={};for(let[e,n]of Object.entries(r))i[e]={id:`${t}-${e}#${Je++}`,...n,paints:new Map};return{id:t,stage:n,values:i,release(){for(let t of Object.values(i)){for(let n of t.paints.values())e.resources.release(n);t.paints.clear()}e.unregisterStage(n),n.release()}}}function K(e,t){return`${e}|${t.geomorphonSkip}|${t.geomorphonComparison}`}async function Ye(e,t,n){let{grid:r}=e,i=n(),a=`geo-${K(t,i).replace(/\|/g,`-`)}`,o=e.builder(a),s=o.elevation(),c=o.settings(8),l=o.words(`forms`);return o.graph.add(new z({id:`geomorphons`,grid:r.grid,elevation:s,parameters:c.view,searchRadius:t,skipRadius:Math.min(i.geomorphonSkip,t-2),comparison:i.geomorphonComparison,output:{forms:l},cellSizeMode:`web-mercator`,rowDirection:`south`})),G(e,a,await o.finishStage({minIntervalMs:40,write:()=>{let e=n();c.parameters.write(ze({flatThresholdDegrees:e.geomorphonFlatAngle,flatDistance:e.geomorphonFlatDistance},r.grid))}}),{forms:{buffer:o.getBuffer(`forms`),format:`uint32`}})}function Xe(e){let t=Fe(e);return`${t===`profile`||t===`plan`?`standard`:t===`ring-multi-radius`?`ring`:t}|${e.curvatureMethod}|${e.curvatureBorder}|${e.ringRadiusInner}|${e.ringRadiusOuter}|${e.ringSquash}`}async function Ze(e,t){let{grid:n}=e,r=t(),i=Fe(r),a=i===`profile`||i===`plan`,o=`curvature-${a?`standard`:i}`,s=e.builder(o),c=s.elevation(),u=s.settings(12),d={},f={id:`curvature`,grid:n.grid,elevation:c,parameters:u.view,method:r.curvatureMethod,cellSizeMode:`web-mercator`,rowDirection:`south`,borderMode:r.curvatureBorder};if(a){let e=s.floats(`profile`),t=s.floats(`plan`);s.graph.add(new p({...f,output:{curvatures:{profile:e,plan:t}}})),d.profile={buffer:s.getBuffer(`profile`),format:`float32`},d.plan={buffer:s.getBuffer(`plan`),format:`float32`}}else if(i===`ring-multi-radius`){let e=s.floats(`curvature`),t=Math.max(r.ringRadiusOuter,r.ringRadiusInner+1);s.graph.add(new p({...f,output:{ringCurvature:e},ringRadii:[r.ringRadiusInner,t],ringSquash:r.ringSquash?`pade-tanh`:`none`})),d.curvature={buffer:s.getBuffer(`curvature`),format:`float32`}}else{let e=s.floats(`curvature`);s.graph.add(new p({...f,output:{curvatures:{[i]:e}}})),d.curvature={buffer:s.getBuffer(`curvature`),format:`float32`}}return G(e,o,await s.finishStage({write:()=>{let e=t();u.parameters.write(l({zFactor:e.zFactor,flatGradient:Math.max(e.flatGradient,1e-6),ringGains:[e.ringGainInner,e.ringGainOuter]},n.grid))}}),d)}function Qe(e){return e===`profile`?`profile`:e===`plan`?`plan`:`curvature`}function $e(e){return`${e.positionProduct}|${e.scalePreset}|${e.innerFraction}|${e.quantum}`}async function et(e,t,n){let{grid:r}=e,{pixelCount:i}=r,a=n(),o=L[a.scalePreset],s=`position-${t}`,c=e.builder(s),l=c.elevation(),u=o.map(e=>({radius:e,...a.innerFraction>0&&Math.floor(e*a.innerFraction)>=1?{innerRadius:Math.floor(e*a.innerFraction)}:{}})),d={id:`topographic-position`,grid:r.grid,elevation:l,scales:u,quantum:W[a.quantum]};if(t===`tpi`||t===`dev`){let r=c.floats(`planes`,o.length*i);c.graph.add(new V({...d,output:t===`tpi`?{topographicPositionIndex:r}:{deviationFromMean:r}}));let a=c.settings(4),l=c.floats(`value`);return M(c.graph,{id:`${t}-select-scale`,invocationCount:i,bindings:[{name:`planes`,view:r,type:`f32`,access:`read`},{name:`select`,view:a.view,type:`f32`,access:`read`},{name:`value`,view:l,type:`f32`,access:`read_write`}],declarations:`const PIXEL_COUNT: u32 = ${i}u;`,body:`
  let plane = u32(select[selectOffset]);
  value[valueOffset + index] = planes[planesOffset + plane * PIXEL_COUNT + index];`}),G(e,s,await c.finishStage({write:()=>a.parameters.write(Float32Array.of(Math.min(n().scaleIndex,o.length-1),0,0,0))}),{value:{buffer:c.getBuffer(`value`),format:`float32`}})}let f=c.floats(`devmax`),p=c.words(`devmax-radius`);c.graph.add(new V({...d,output:{maximumDeviation:f,maximumDeviationRadius:p}}));let m=c.words(`codes`),h=o.map(e=>`${e}u`).join(`, `);return M(c.graph,{id:`scale-class`,invocationCount:i,bindings:[{name:`devmax`,view:f,type:`f32`,access:`read`},{name:`devmaxRadius`,view:p,type:`u32`,access:`read`},{name:`codes`,view:m,type:`u32`,access:`read_write`}],declarations:`
var<private> SCALE_RADII: array<u32, ${o.length}> = array<u32, ${o.length}>(${h});`,body:`
  let radius = devmaxRadius[devmaxRadiusOffset + index];
  var code = 0u;
  let notFinite = (bitcast<u32>(devmax[devmaxOffset + index]) & 0x7f800000u) == 0x7f800000u;
  if (radius != 0u && !notFinite) {
    for (var scaleIndex = 0u; scaleIndex < ${o.length}u; scaleIndex = scaleIndex + 1u) {
      if (SCALE_RADII[scaleIndex] == radius) {
        code = scaleIndex + 1u;
      }
    }
  }
  codes[codesOffset + index] = code;`}),G(e,s,await c.finishStage({write:()=>{}}),{codes:{buffer:c.getBuffer(`codes`),format:`uint32`}})}function tt(e){return`${e.weissSmall}|${e.weissLarge}|${e.weissStandardization}|${e.quantum}`}async function nt(e,t){let{grid:n}=e,r=t(),i=`weiss`,a=e.builder(i),o=a.elevation(),s=a.settings(8),c=a.words(`codes`);return a.graph.add(new Ue({id:`weiss`,grid:n.grid,elevation:o,parameters:s.view,smallScale:{radius:r.weissSmall},largeScale:{radius:Math.max(r.weissLarge,r.weissSmall+1)},standardization:r.weissStandardization,quantum:W[r.quantum],output:{landforms:c},cellSizeMode:`web-mercator`})),G(e,i,await a.finishStage({write:()=>{let e=t();s.parameters.write(He({standardThreshold:e.weissThreshold,slopeThresholdDegrees:e.weissSlope},n.grid))}}),{codes:{buffer:a.getBuffer(`codes`),format:`uint32`}})}async function rt(e,t,n){let{pixelCount:r}=e.grid,i=`agreement`,a=e.builder(i),o=j(a.graph,`weiss-codes`,t.values.codes.buffer,`uint32`,r),s=j(a.graph,`forms-codes`,n.values.forms.buffer,`uint32`,r),c=a.words(`codes`);return M(a.graph,{id:`agreement-kernel`,invocationCount:r,bindings:[{name:`weissCodes`,view:o,type:`u32`,access:`read`},{name:`formCodes`,view:s,type:`u32`,access:`read`},{name:`agreement`,view:c,type:`u32`,access:`read_write`}],declarations:`
fn weissGroup(code: u32) -> u32 {
  if (code >= 1u && code <= 4u) { return 1u; }
  if (code == 5u || code == 6u) { return 2u; }
  if (code >= 7u && code <= 10u) { return 3u; }
  return 0u;
}
fn formGroup(code: u32) -> u32 {
  if (code == 1u || code == 6u) { return 2u; }
  if (code >= 2u && code <= 5u) { return 3u; }
  if (code >= 7u && code <= 10u) { return 1u; }
  return 0u;
}`,body:`
  let weissKind = weissGroup(weissCodes[weissCodesOffset + index]);
  let formKind = formGroup(formCodes[formCodesOffset + index]);
  var result = 3u;
  if (weissKind != 0u && formKind != 0u) {
    if (weissKind == formKind) {
      result = 0u;
    } else if (weissKind + formKind == 4u && weissKind != 2u) {
      result = 2u;
    } else {
      result = 1u;
    }
  }
  agreement[agreementOffset + index] = result;`}),G(e,i,await a.finishStage({write:()=>{},dependencies:[t.stage,n.stage]}),{codes:{buffer:a.getBuffer(`codes`),format:`uint32`}})}var it={light:{rise:[178,24,43,255],level:[140,140,140,255],fall:[33,102,172,255]},dark:{rise:[255,120,130,255],level:[176,184,196,255],fall:[111,176,219,255]}},q=8;function at(t,n){let r=[n.origin[0],n.origin[1],0],i=t.createBuffer(`rays-segments`,new Float32Array(q*4)),a=t.createBuffer(`rays-signs`,new Uint32Array(q)),o=t.createBuffer(`rays-zeniths`,new Float32Array(q*2)),s=t.createBuffer(`rays-nadirs`,new Float32Array(q*2)),c=t.createBuffer(`rays-pin`,new Float32Array(2)),l=!1,u=(e,t)=>{let[r,i]=n.getLongitudeLatitude(e,t);return n.projection.project(r,i)};return{setCell(e){if(l=e!==null,!e)return;let t=new Float32Array(q*4),n=new Uint32Array(q),r=new Float32Array(q*2),d=new Float32Array(q*2),f=u(e.column,e.row);e.rays.forEach((e,i)=>{let a=u(e.endCell[0],e.endCell[1]);t.set([f[0],f[1],a[0],a[1]],i*4),n[i]=e.sign>0?0:e.sign===0?1:2;let o=e.zenith?u(e.zenith.cell[0],e.zenith.cell[1]):f,s=e.nadir?u(e.nadir.cell[0],e.nadir.cell[1]):f;r.set(o,i*2),d.set(s,i*2)}),i.write(t),a.write(n),o.write(r),s.write(d),c.write(Float32Array.of(f[0],f[1]))},getLayers(t){if(!l)return[];let n=oe[t],u=it[t],d=t===`dark`?[9,12,16,230]:[255,255,255,235],f={coordinateSystem:e.METER_OFFSETS,coordinateOrigin:r},p=J(n.signal),m=J(n.ink),h=e=>e;return[new te({...f,id:`landforms-rays`,segments:h(i),instanceCount:q,values:a,valueFormat:`uint32`,colormap:`category`,palette:[u.rise,u.level,u.fall],widthPixels:2.2,cap:`round`,outlineColor:d,outlineWidthPixels:1.6}),new A({...f,id:`landforms-rays-nadir`,positions:s,instanceCount:q,shape:`ring`,radiusPixels:4.5,outlineWidthPixels:1.6,color:m}),new A({...f,id:`landforms-rays-zenith`,positions:o,instanceCount:q,shape:`circle`,radiusPixels:3.2,color:m,outlineColor:d,outlineWidthPixels:1.2}),new A({...f,id:`landforms-pin-ring`,positions:c,instanceCount:1,shape:`ring`,radiusPixels:9,outlineWidthPixels:2,color:p}),new A({...f,id:`landforms-pin-dot`,positions:c,instanceCount:1,shape:`circle`,radiusPixels:3,color:p})]}}}function J(e){let t=Number.parseInt(e.slice(1),16);return[t>>16&255,t>>8&255,t&255,255]}function ot(e,t,n,r){let i=e.getPixel(t[0],t[1]);if(!i)return null;let{width:a,height:o,cpuElevation:s}=e,c=e.getGroundCellSize(i[1]),l=Math.ceil(n/c),u=null,d=-1/0;for(let e=Math.max(r,i[1]-l);e<=i[1]+l;e++)for(let t=Math.max(r,i[0]-l);t<=i[0]+l;t++){if(t>=a-r||e>=o-r||Math.hypot(t-i[0],e-i[1])*c>n)continue;let l=0;for(let n=-r;n<=r;n++){let i=(e+n)*a+t;for(let e=-r;e<=r;e++)l+=s[i+e]}let f=(2*r+1)**2,p=s[e*a+t]-l/f;p>d&&(d=p,u={column:t,row:e})}return u}var st=80,ct=250,lt=220,ut=10,dt=48,ft=0;function Y(e){let t=new Float64Array(32);for(let n=0;n<e.length;n++){let r=e[n];r<32&&t[r]++}return t}function pt(e,t){let n=Math.max(1,Math.floor(e.length/4e5)),r=new Float32Array(Math.ceil(e.length/n)),i=0;for(let t=0;t<e.length;t+=n){let n=e[t];Number.isFinite(n)&&(r[i++]=Math.abs(n))}return i===0?NaN:r.subarray(0,i).sort()[Math.min(i-1,Math.floor(t*i))]}function mt(e,t,n,r){let i=Array(r).fill(0),a=r/(n-t);for(let n=0;n<e.length;n++){let o=e[n];Number.isFinite(o)&&i[Math.min(r-1,Math.max(0,Math.floor((o-t)*a)))]++}return i}function ht(e,t=new Set){for(let n of e.dependencies)ht(n,t);return t.add(e),t}async function X(e){let{device:t}=e,n=await de(e.datasets.get(`alps-dem`),e.signal),r=await be(e.datasets.get(`alps-context`),e.signal);e.signal.throwIfAborted();let{width:a,height:o,pixelCount:s}=n,c=me(n),l=he(c),u=we(r.glacierGeoJson,c),d=()=>e.options,f=new i(t,`landforms`),p=new pe(e,f,n);p.elevationBuffer.write(n.cpuElevation),p.validityBuffer.write(new Uint32Array(s).fill(1));let m=f.createBuffer(`colors-main`,s*4),h=f.createBuffer(`colors-compare`,s*4),g=f.createParameterBuffer(`paint`,`float32`,96),_=at(f,n),v=e.ground(),y=Te({dem:c,device:t,ground:v,glacierMask:u});await y.prepare(),e.signal.throwIfAborted();let b=!1,x=[],S=n.groundCellSize,C=Ee(v),w=De(v),T=new Map,E=new Map;function ee(){E.clear();for(let[e,t]of T)E.set(e,e.startsWith(`curvature:`)?Le(t,v):Ie(t,v))}function D(e=U){if(!e)return null;switch(e.tableId){case`quiet`:return C.quiet;case`grass`:return C.grass;case`weiss`:return C.weiss;case`agreement`:return C.agreement;case`scale`:return C.scale;case`dev`:return w;default:return E.get(e.signedKey??``)??null}}let O=new Map,k=null,A=null,te=null,j=null,M=null,oe=[],N=new Set,P=new Map,ze=e=>{for(let t of O.values())e(t);for(let t of[k,A,te,j])t&&e(t.build)},z=e=>{for(let t of Object.values(e.values))P.delete(t.id);e.release()};function Be(e){if(oe.includes(e)||M?.builds.includes(e))return!0;for(let t of N)if(t.includes(e))return!0;return!1}function Ve(){let e=d(),t=new Set([K(e.geomorphonRadius,e)]);t.add(K(8,e)),j&&t.add(j.geomorphonKey);for(let n of I)t.add(K(n,e));return t}async function B(e,t){let n=K(e,d()),r=O.get(n);if(r)return O.delete(n),O.set(n,r),t?.push(r),r;let i=await Ye(p,e,d),a=O.get(n);if(a||b){if(z(i),a)return t?.push(a),a;throw Error(`Scene destroyed`)}O.set(n,i),t?.push(i);let o=Ve();for(let[e,t]of O){if(O.size<=ut)break;o.has(e)||Be(t)||(z(t),O.delete(e))}return i}async function V(e,n){e.paints.has(n)||e.paints.set(n,await qe(f,t,`${e.id}-${n}`,{buffer:e.buffer,format:e.format,width:a,height:o},g,n===`main`?m:h))}async function He(e){let t={agreementReplaced:!1,retired:[],geomorphons:[]};N.add(t.geomorphons);try{let n=await Ue(e,t),r={...t,display:n};return await V(n.value,`main`),n.compare&&await V(n.compare.value,`compare`),r}catch(e){throw We(t),e}}async function Ue(e,t){let n={forms:null,agreement:null,weiss:null,scale:null,compare:null,signedKey:null};switch(e.product){case`geomorphons`:{let r=await B(e.geomorphonRadius,t.geomorphons),i=e.compareRadius?await B(8,t.geomorphons):null,a=e.palette===`grass`?`grass`:`quiet`;return{...n,key:`${r.values.forms.id}|${a}|${i?.values.forms.id??``}`,stage:r.stage,value:r.values.forms,mode:`codes`,tableId:a,reads:[r.values.forms],compare:i?{stage:i.stage,value:i.values.forms}:null,dilate:!0,forms:r.values.forms}}case`curvature`:{let r=Xe(e),i=k;i?.key!==r&&(i&&t.retired.push(i.build),i={key:r,build:await Ze(p,d)},t.curvature=i);let a=Fe(e),o=i.build.values[Qe(a)];return{...n,key:`${o.id}|curvature`,stage:i.build.stage,value:o,mode:`breaks`,tableId:`curvature`,reads:[o],dilate:!1,signedKey:Oe(e)}}case`position`:{let r=$e(e),i=A;i?.key!==r&&(i&&t.retired.push(i.build),i={key:r,build:await et(p,e.positionProduct,d)},t.position=i);let a=i.build;if(e.positionProduct===`scale`)return{...n,key:`${a.values.codes.id}|scale`,stage:a.stage,value:a.values.codes,mode:`codes`,tableId:`scale`,reads:[a.values.codes],dilate:!1,scale:a.values.codes};let o=e.positionProduct===`tpi`?`tpi`:`dev`;return{...n,key:`${a.values.value.id}|${o}`,stage:a.stage,value:a.values.value,mode:`breaks`,tableId:o,reads:[a.values.value],dilate:!1,signedKey:Oe(e)}}case`weiss`:{let r=tt(e),i=te,a=j;i?.key!==r&&(a&&t.retired.push(a.build),a=null,t.agreementReplaced=!0,i&&t.retired.push(i.build),i={key:r,build:await nt(p,d)},t.weiss=i);let o=await B(e.geomorphonRadius,t.geomorphons),s=K(e.geomorphonRadius,e);a&&a.geomorphonKey!==s&&(t.retired.push(a.build),a=null,t.agreementReplaced=!0),a||(a={key:r,geomorphonKey:s,build:await rt(p,i.build,o)},t.agreement=a,t.agreementReplaced=!0);let c=i.build.values.codes,l=o.values.forms,u=a.build.values.codes,f=e.view===`agreement`?u:e.view===`geomorphons`?l:c,m=e.view===`agreement`?`agreement`:e.view===`geomorphons`?e.palette===`grass`?`grass`:`quiet`:`weiss`;return{...n,key:`${f.id}|${m}`,stage:a.build.stage,value:f,mode:`codes`,tableId:m,reads:[f,...[c,l,u].filter(e=>e!==f)],dilate:e.view===`geomorphons`,forms:l,agreement:u,weiss:c}}}}function We(e){N.delete(e.geomorphons);for(let t of[e.curvature,e.position,e.weiss,e.agreement])t&&z(t.build)}function H(t){N.delete(t.geomorphons),oe=t.geomorphons,t.curvature&&(k=t.curvature),t.position&&(A=t.position),t.weiss&&(te=t.weiss),t.agreementReplaced&&(j=t.agreement??null);for(let e of t.retired)z(e);e.setStatus(``),St(t.display),It(),Ot(),Nt()}let U=null,Ge=0,W=!0,Je=!1,G=0,it=null,q=[],J=!1,X=!1,gt=0,_t=new Set,vt=!1,yt=``;function Z(){let e=D();!e||!U||(Ke(g,e,{mode:U.mode,dilateRadius:U.dilate?G:0,peakCode:Ne,pitCode:ke,isolate:it}),W=!0)}function bt(){e.setLegendData(`ground`,v),e.setLegendData(`tables`,C)}function xt(){let t=D(),n=U;e.setLegendData(`signedFor`,n?.signedKey??null),e.setLegendData(`signedTable`,n?.signedKey&&t?t:null)}function St(t){let n=!U||U.key!==t.key;U=t,n&&(Ge++,Je=!1,it=null,q=[]),W=!0,X=!0,G=t.dilate?je(e.getViewState().zoom):0,Z(),xt(),kt(),e.requestLayers()}let Ct=f.track(new re);function wt(){return b?Promise.resolve():Ct.request({build:()=>He(d()),commit:H,discard:We}).catch(t=>{b||(e.setStatus(`Cannot build ${d().product}: ${t.message}`),It(),Ot(),Nt())})}let Q=null;function Tt(e=st){Q&&clearTimeout(Q),Q=setTimeout(()=>{Q=null,wt()},e)}let $=e=>e*S;function Et(e){let t=`Geomorphons · L = ${e.geomorphonRadius} px = ${Math.round($(e.geomorphonRadius))} m · flat < ${e.geomorphonFlatAngle.toFixed(1)}°`;switch(e.product){case`geomorphons`:return e.compareRadius?`${t} · against L = 8 px`:t;case`curvature`:return`${Ae(Fe(e))}, 1/m · ${e.curvatureMethod===`florinsky`?`5 x 5`:`3 x 3`} polynomial fit · ${S.toFixed(1)} m cells`;case`position`:{if(e.positionProduct===`scale`)return`Window radius at which a cell stands out most · eight windows`;let t=L[e.scalePreset][e.scaleIndex];return`${e.positionProduct===`tpi`?`Height above the window mean, m`:`Height against the window, SD`} · window ${Math.round($(2*t+1))} m wide`}case`weiss`:return e.view===`geomorphons`?t:e.view===`agreement`?`Weiss against geomorphons (L = ${e.geomorphonRadius} px) · convex, neutral, concave`:`Weiss landforms · windows ${e.weissSmall} and ${e.weissLarge} px · ${e.weissStandardization} standardisation`}}function Dt(e){if(e.product===`geomorphons`||e.product===`weiss`&&e.view===`geomorphons`){let t=[$(e.geomorphonRadius)];return e.product===`geomorphons`&&e.compareRadius&&t.unshift($(8)),t.map(e=>Math.round(e))}if(e.product===`position`&&e.positionProduct!==`scale`){let t=L[e.scalePreset][e.scaleIndex];return[Math.round($(2*t+1))]}}function Ot(){let t=d(),r=Dt(t),i={title:{subtitle:Et(t),sample:ge(n)},scaleBar:{units:`metric`,ticks:r}},a=JSON.stringify(i);a!==yt&&(yt=a,e.setFurniture(i))}function kt(){let t=d(),n=t.geomorphonRadius;if(e.setReadout(`lookout`,`${n} px = ${Math.round($(n))} m`),t.product===`position`&&t.positionProduct!==`scale`){let n=2*L[t.scalePreset][t.scaleIndex]+1;e.setReadout(`naiveReads`,n*n),e.setReadout(`tableReads`,4)}let r=U;if(r){let t=ht(r.stage).size,n=0;for(let e of ht(r.stage))n+=e.graphs.length;e.setCost({records:s,passes:n+1+(r.compare?n+1:0),note:`${t} contributor stage${t===1?``:`s`} · ${ce(O.size)} geomorphon graphs cached`})}}let At=xe(r,{window:c.lngLatBounds,names:[`Gornergletscher`],max:1}),jt=r.glaciers.find(e=>e.name?.toLowerCase().startsWith(`gornergletscher`)),Mt=jt?Ce(jt,c.lngLatBounds):null;function Nt(){let t=d(),n=U,r=n?.forms?P.get(n.forms.id):void 0;if(!n||!r||!Mt||t.product!==`geomorphons`||t.showRays||t.compareRadius){e.setAnnotations(`glacier-note`,null);return}let i=new Float64Array(11),a=0;for(let e=0;e<s;e++){if(u[e]===0)continue;let t=r[e];t>=1&&t<=10&&(i[t]++,a++)}if(a===0){e.setAnnotations(`glacier-note`,null);return}let o=1;for(let e=2;e<=10;e++)i[e]>i[o]&&(o=e);e.setAnnotations(`glacier-note`,[{kind:`note`,id:`glacier-finding`,coordinate:Mt.lngLat,title:se(`{share:percent} of glacier cells read {form}`,{share:i[o]/a,form:R[o]}),text:`Ice is not a class: the classifier sees shape, not ice`,tone:`accent`,priority:5}])}let Pt=ie.places[`hornli-hut`],Ft=ot(n,Pt.lngLat,150,8)??{column:a>>1,row:o>>1};function It(){let t=d();if(t.product!==`geomorphons`||!t.showRays){_.setCell(null),e.setAnnotations(`pin`,null),e.setReadout(`pattern`,null),e.setReadout(`readsPerCell`,null),e.requestLayers();return}let r=l.geomorphonRays(Ft.column,Ft.row,t.geomorphonRadius,t.geomorphonFlatAngle);_.setCell(r);let i=r.rays.reduce((e,t)=>e+t.cellCount,0);e.setReadout(`pattern`,r.pattern),e.setReadout(`readsPerCell`,i);let a=n.getLongitudeLatitude(Ft.column,Ft.row);e.setAnnotations(`pin`,[{kind:`note`,id:`pin-note`,coordinate:[a[0],a[1]],title:se(`{pattern} = {form}`,{pattern:r.pattern,form:R[r.formCode]??`Border`}),text:se(`{up:integer} rise, {down:integer} fall`,{up:r.plusCount,down:r.minusCount}),tone:`signal`,priority:9}]),e.requestLayers()}function Lt(t){let n=0;for(let e=1;e<=10;e++)n+=t[e];if(n===0)return;let r=Array.from({length:11},(e,r)=>r===0?0:t[r]/n);e.setLegendData(`formShares`,r);let i=d().palette===`grass`?C.grass:C.quiet;e.setChart(`classShares`,{kind:`stacked`,description:`Share of cells in each landform class, ridge-like forms first, valley-like forms last.`,segments:ye.map(e=>{let n=ve[e];return{label:R[n],value:t[n],color:Re(i.colors[n],v)}})}),e.setReadout(`cells`,n),e.setReadout(`peakShare`,t[Ne]/n)}function Rt(t){let n=t[1],r=t[2],i=t[0]+n+r;i!==0&&(e.setLegendData(`agreementShares`,[t[0]/i,n/i,r/i]),e.setReadout(`agreeShare`,t[0]/i),e.setReadout(`oppositeShare`,r/i))}function zt(t){let n=0;for(let e=1;e<=8;e++)n+=t[e];n!==0&&e.setLegendData(`scaleShares`,Array.from({length:9},(e,r)=>r===0?0:t[r]/n))}function Bt(t,n){let r=t.signedKey;if(!r)return;if(!T.has(r)&&t.tableId!==`dev`){let e=pt(n,.98);if(!(e>0))return;T.set(r,e),E.set(r,r.startsWith(`curvature:`)?Le(e,v):Ie(e,v)),Z(),xt()}let i=D(t),a=T.get(r);if(r.startsWith(`curvature:`)&&i&&a){let t=a*1.2,r=mt(n,-t,t,dt);e.setChart(`curvatureHistogram`,{kind:`histogram`,values:r,xDomain:[-t,t],breaks:i.breaks,classColors:i.colors.map(e=>Re(e,v)),xLabel:`Curvature (1/m), tails clipped`,formatX:e=>e.toPrecision(2),description:`Histogram of the curvature of every cell, bars coloured by class, with the frozen class breaks.`,yLabel:`Cells`});let o=i.breaks.slice(i.breaks.length/2),s=i.format??(e=>e.toPrecision(2));e.setReadout(`breaks`,`± ${o.map(e=>s(e)).join(`, `)} 1/m`),e.setReadout(`p98`,`${a.toPrecision(2)} 1/m`)}else a&&e.setReadout(`p98`,`${a.toPrecision(2)} m`)}function Vt(t,n){let r=U;if(!r)return;let i=t.format===`uint32`?new Uint32Array(n.buffer,n.byteOffset,n.byteLength>>2):new Float32Array(n.buffer,n.byteOffset,n.byteLength>>2);P.set(t.id,i),i instanceof Uint32Array?t===r.forms?(Lt(Y(i)),Nt()):t===r.agreement?Rt(Y(i)):t===r.scale&&zt(Y(i)):t===r.value&&Bt(r,i),t===r.value&&Ut(r,!1),e.requestLayers()}function Ht(e,t){let n=U;if(!n||M||J)return;if(q.length===0){if(!X||t-gt<lt||n.stage.isPending(ft))return;q=[...n.reads],X=!1}let r=q[0],i=Ge;p.readBulk(e,r.buffer,e=>{J=!1,gt=performance.now(),!(b||i!==Ge)&&Vt(r,e)})&&(J=!0,q.shift())}async function Ut(n,r){if(!(vt||b)&&!(!r&&_t.has(n.key))){_t.add(n.key),vt=!0,e.setReadout(`timing`,`measuring...`);try{let r=0,i=`wall clock`;for(let a of ht(n.stage))for(let o of a.graphs){let a=await ne(t,o,{parameters:void 0,completionBuffer:n.value.buffer,runs:5,warmUpRuns:1,repetitions:2,signal:e.signal});r+=a.milliseconds,i=a.method===`gpu-timestamps`?`GPU`:`wall clock`}b||e.setReadout(`timing`,`${r.toFixed(1)} ms (${i}), ${ce(s)} cells`)}catch{b||e.setReadout(`timing`,null)}finally{vt=!1}}}function Wt(){M||d().product!==`geomorphons`||(M={phase:`compiling`,builds:[],index:0,ran:!1,reading:!1,shares:[]},e.setStatus(`Compiling ${I.length} look-out radii, once each...`),Gt())}function Gt(){let t=M;if(!(!t||b)){if(t.builds.length<I.length){B(I[t.builds.length]).then(e=>{M!==t||b||(t.builds.push(e),x.push(setTimeout(Gt,16)))},t=>{b||(e.setStatus(`Sweep failed: ${t.message}`),M=null)});return}t.phase=`running`,t.index=0,e.setStatus(`Sweeping the look-out radius...`)}}function Kt(e,t){let n=M;if(!n||n.phase!==`running`)return;let r=n.builds[n.index];r&&(n.ran||=(r.stage.markDirty(),r.stage.run(e,t,ft)),!(!n.ran||n.reading)&&p.readBulk(e,r.values.forms.buffer,e=>{if(b||M!==n)return;let t=Y(new Uint32Array(e.buffer,e.byteOffset,e.byteLength>>2)),r=0;for(let e=1;e<=10;e++)r+=t[e];n.shares.push(Pe.map(e=>r>0?e.codes.reduce((e,n)=>e+t[n],0)/r*100:0)),n.index++,n.ran=!1,n.reading=!1,n.index>=n.builds.length&&qt(n)})&&(n.reading=!0))}function qt(t){M=null,e.setStatus(``),e.setChart(`sweepChart`,{kind:`line`,series:Pe.map((e,n)=>({label:e.label,x:I,y:t.shares.map(e=>e[n]),points:!0,color:n})),xScale:`log`,xDomain:[I[0],I[I.length-1]],xLabel:`Look-out radius L (px)`,yLabel:`Share of cells (%)`,formatY:e=>`${e.toFixed(0)}`,link:{option:`geomorphonRadius`,label:e=>`L = ${e} px`},description:`Share of cells in ridge-like, slope, valley-like and flat forms at five look-out radii.`}),e.setCost({records:s,passes:I.length*2,note:`${I.length} radii compiled once each, then cached`}),X=!0,kt()}function Jt(e){let t=d().opacity;switch(e.tableId){case`quiet`:return v===`light`?{alpha:.85*t,blendMode:`multiply`}:{alpha:.85*t,blendMode:`normal`};case`grass`:case`weiss`:return{alpha:.62*t,blendMode:`normal`};case`agreement`:return{alpha:.9*t,blendMode:`normal`};case`scale`:return{alpha:.95*t,blendMode:`normal`};default:return{alpha:t,blendMode:`normal`}}}function Yt(){let t=d(),r=[];t.underlay&&r.push(y.getLayer({opacity:1}));let i=U;if(i&&Je&&D(i)){let t={coordinateOrigin:[n.origin[0],n.origin[1],0],gridSize:[a,o],bounds:n.bounds,...Jt(i)};i.compare&&e.getCompare()?r.push(new fe({...t,id:`landforms-classes-a`,colors:h,compareSide:`a`}),new fe({...t,id:`landforms-classes-b`,colors:m,compareSide:`b`})):r.push(new fe({...t,id:`landforms-classes`,colors:m}))}return t.product===`geomorphons`&&t.showRays&&r.push(..._.getLayers(v)),r}e.setAnnotations(`places`,[...Se([F(`matterhorn`,{priority:6}),F(`gornergrat`,{priority:4})],c),F(`hornli-hut`,{priority:5,minZoom:11}),F(`riffelsee`,{priority:3,minZoom:11}),F(`zermatt`,{priority:2}),...At]);function Xt(e){let t=U;if(!t||!e.coordinate)return null;let r=n.getPixel(e.coordinate[0],e.coordinate[1]);if(!r)return null;let[i,s]=r,c=s*a+i,u=d(),f=n.cpuElevation[c],p=l.hornAt(i,s),m=[{label:`Elevation`,value:Math.round(f).toLocaleString(`en-US`),unit:`m`},{label:`Slope`,value:Math.round(p.slopeDeg),unit:`°`}],h=(e,t)=>e&&t>=0&&t<e.colors.length?Re(e.colors[t],v):void 0,g=D(t),_=()=>{let e=l.geomorphonRays(i,s,u.geomorphonRadius,u.geomorphonFlatAngle),n=t.forms?P.get(t.forms.id):void 0,r=n?n[c]:e.formCode,a=u.palette===`grass`?C.grass:C.quiet;return{title:R[r]??`Border`,subtitle:`${e.plusCount} rays rise, ${e.minusCount} fall`,rows:[{label:`Landform`,value:R[r]??`Border`,swatch:h(a,r),emphasis:!0},{label:`Rays`,value:e.pattern},...m]}};if(t.tableId===`quiet`||t.tableId===`grass`)return _();let y=P.get(t.value.id)?.[c];if(y===void 0)return{title:`Reading the map...`,rows:m};switch(t.tableId){case`weiss`:{let e=_e[y-1];return{title:e??`No class`,rows:[{label:`Landform`,value:e??`No class`,swatch:h(g,y),emphasis:!0},...m]}}case`agreement`:{let e=t.weiss?P.get(t.weiss.id)?.[c]:void 0,n=[`Agree`,`Disagree`,`Opposite`,`Not compared`],r=t.forms?P.get(t.forms.id)?.[c]:void 0;return{title:n[y]??`Not compared`,rows:[{label:`Weiss and geomorphons`,value:n[y]??`Not compared`,swatch:h(g,y),emphasis:!0},...e===void 0?[]:[{label:`Weiss`,value:_e[e-1]??`No class`}],...r===void 0?[]:[{label:`Geomorphon`,value:R[r]??`Border`}],...m]}}case`scale`:{let e=Me(L[u.scalePreset])[y-1];return{title:e?`Strongest at a ${e} window radius`:`No dominant scale`,rows:[{label:`Strongest window radius`,value:e??`none`,swatch:h(g,y),emphasis:!0},...m]}}case`curvature`:{let e=y,t=Fe(u),n=g?ae(g,e):-1,r=n===(g?Math.floor(g.colors.length/2):-1)?`near flat`:e>0?t===`profile`||t===`plan`?`convex`:`positive`:t===`profile`||t===`plan`?`concave`:`negative`;return{title:`${Ae(t)}: ${r}`,rows:[{label:Ae(t),value:Number.isFinite(e)?e.toPrecision(2):`no value`,unit:`1/m`,swatch:h(g,n),emphasis:!0},{label:`Reading`,value:r},...m]}}default:{let e=y,r=L[u.scalePreset][u.scaleIndex],c=2*r+1,l=c*S,d=g?ae(g,e):-1,f=Math.max(0,i-r),p=Math.min(a-1,i+r),_=Math.max(0,s-r),v=Math.min(o-1,s+r),b=n.getLongitudeLatitude(f-.5,_-.5),x=n.getLongitudeLatitude(p+.5,v+.5),C=t.tableId===`tpi`,w=e>=0?`above`:`below`;return{title:C?`${Math.abs(e).toFixed(0)} m ${w} the mean of its ${le(l)} window`:`${Math.abs(e).toFixed(1)} standard deviations ${w} its ${le(l)} window`,rows:[{label:C?`Position`:`Deviation`,value:Number.isFinite(e)?ue(e,C?1:2):`no value`,unit:C?`m`:`standard deviations`,swatch:h(g,d),emphasis:!0},{label:`Window`,value:`${c} x ${c}`,unit:`cells`},...m],highlight:{kind:`box`,bounds:[b[0],x[1],x[0],b[1]]}}}}}let Zt=new Set([`product`,`geomorphonRadius`,`geomorphonSkip`,`geomorphonComparison`,`compareRadius`,`curvatureMethod`,`curvatureMoreKinds`,`curvatureBorder`,`ringRadiusInner`,`ringRadiusOuter`,`ringSquash`,`positionProduct`,`scalePreset`,`innerFraction`,`quantum`,`weissSmall`,`weissLarge`,`weissStandardization`]),Qt=new Set([`geomorphonFlatAngle`,`geomorphonFlatDistance`,`flatGradient`,`zFactor`,`ringGainInner`,`ringGainOuter`,`scaleIndex`,`weissThreshold`,`weissSlope`]),$t=new Set([`geomorphonRadius`,`weissSmall`,`weissLarge`,`ringRadiusInner`,`ringRadiusOuter`,`innerFraction`]),en=()=>ze(e=>{e.stage.markDirty()});return await wt(),bt(),{getCompiledGraphs(){let e=[...p.getCompiledGraphs()];return ze(t=>{for(let n of Object.values(t.values))for(let t of n.paints.values())e.push(t)}),e},setOption(t){let n=t;Zt.has(n)?Tt($t.has(n)?ct:st):Qt.has(n)?(en(),n===`geomorphonFlatAngle`&&It(),kt(),Ot()):n===`curvatureKind`||n===`view`||n===`palette`?Tt(0):n===`showRays`?(It(),Nt()):e.requestLayers(),(n===`scaleIndex`||n===`geomorphonFlatAngle`)&&Ot()},onAction(e){e===`sweep`?Wt():e===`measure`&&U&&Ut(U,!0)},onThemeChange(){e.requestLayers()},onGroundChange(t){v=t,y.setGround(t),C=Ee(t),w=De(t),ee(),bt(),xt(),Z();let n=U?.forms?P.get(U.forms.id):void 0;n instanceof Uint32Array&&Lt(Y(n)),e.requestLayers()},onLegendFilter(t,n){U&&(it=n?t===`landform`?n.filter(e=>e<ye.length).map(e=>ve[ye[e]]):t===`scale`?n.map(e=>e+1):[...n]:null,Z(),e.requestLayers())},onCompareChange(){e.requestLayers()},onClick(e){let t=d();if(t.product!==`geomorphons`||!t.showRays||!e.coordinate)return!1;let r=n.getPixel(e.coordinate[0],e.coordinate[1]);return r?(Ft={column:r[0],row:r[1]},It(),!0):!1},getTooltip:Xt,encode(t){if(b)return;let n=performance.now();Kt(t,n);let r=U;if(!r)return;let i=r.dilate?je(e.getViewState().zoom):0;i!==G&&(G=i,Z());let a=r.stage.run(t,n,ft);r.compare&&(a=r.compare.stage.run(t,n,ft)||a),a&&(W=!0,X=!0),W&&D(r)&&(r.value.paints.get(`main`)?.encode(t,{parameters:void 0}),r.compare&&r.compare.value.paints.get(`compare`)?.encode(t,{parameters:void 0}),W=!1,Je=!0),Ht(t,n)},getLayers:Yt,destroy(){b=!0,Q&&clearTimeout(Q);for(let e of x)clearTimeout(e);x=[];for(let e of[j,te,A,k])e&&e.build.release();for(let e of O.values())e.release();O.clear(),p.destroy(),y.destroy(),f.destroy()}}}export{X as createLandforms};