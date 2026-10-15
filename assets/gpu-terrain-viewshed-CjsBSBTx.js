import{n as e}from"./earth-xB16XgAa.js";import{a as t,i as n,n as r,o as i,r as a}from"./gpu-raster-extrema-pyramid-DhOnk6uD.js";import{_ as o,i as s,w as c,y as l}from"./resources-CwvhEOBe.js";import{s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as d,c as f,i as p,l as m,n as h,o as g,r as _,s as v,t as y,u as b}from"./raster-band-utils-CA-sh8xt.js";import{c as x,o as S,r as C}from"./raster-grid-D2bmCwkc.js";import{t as w}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";function T(e){return`
fn getRayGroundCell(cellSize: vec2<f32>, northEdge: f32, southEdge: f32, rowCoordinate: f32) -> vec2<f32> {
  ${e===`uniform`?`return cellSize;`:`let edge = mix(northEdge, southEdge, rowCoordinate / f32(HEIGHT));
  ${S(e)}
  return cellSize * groundScale;`}
}`}var E=762939453125e-17;function D(e){let{width:t,height:r,traversal:i,layout:a,clearance:o}=e;if(i===`pyramid`&&!a)throw Error(`terrain sight line traversal pyramid requires a pyramid layout`);return`const WIDTH: u32 = ${t}u;
const HEIGHT: u32 = ${r}u;
const HIDDEN: u32 = 0u;
const VISIBLE: u32 = 1u;
const OUT_OF_RANGE: u32 = 2u;
const NO_DATA: u32 = 3u;
const MARGINAL: u32 = 4u;
const MAXIMUM_FLOAT: f32 = 3.4028234663852886e38;
const SKIP_EPSILON: f32 = ${E};
${y}
${i===`pyramid`?n(a):``}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  // Canonical elevation is NaN exactly where a pixel is invalid, so one bit test per corner
  // replaces four validity loads; the interpolation below is unchanged.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00, corner10, fraction.x);
  let bottom = mix(corner01, corner11, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}
fn rayPosition(observer: vec2<f32>, delta: vec2<f32>, sampleIndex: u32, count: u32) -> vec2<f32> {
  let fraction = f32(sampleIndex) / f32(count);
  return observer + delta * fraction;
}
fn rayDistance(targetDistance: f32, sampleIndex: u32, count: u32) -> f32 {
  let fraction = f32(sampleIndex) / f32(count);
  return targetDistance * fraction;
}
struct SightLineResult {
  code: u32,
  clearance: f32
}
${i===`pyramid`?`
// Tries to skip samples \`first\` .. last at one pyramid level; returns last + 1, or 0 when the block
// does not contain the ray long enough or its bound exceeds the threshold.
fn trySkipLevel(
  level: u32,
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  baseFirst: vec2<u32>,
  distanceFirst: f32,
  eye: f32,
  curvature: f32,
  threshold: f32
) -> u32 {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let blockSize = pyramidLevelBlockSize(level);
  let cell = baseFirst / vec2<u32>(blockSize);
  let origin = vec2<f32>(cell * vec2<u32>(blockSize));
  let size = f32(blockSize);
  var exitFraction = 2.0;
  if (delta.x > 0.0) {
    exitFraction = min(exitFraction, (origin.x + size - observer.x) / delta.x);
  } else if (delta.x < 0.0) {
    exitFraction = min(exitFraction, (origin.x - observer.x) / delta.x);
  }
  if (delta.y > 0.0) {
    exitFraction = min(exitFraction, (origin.y + size - observer.y) / delta.y);
  } else if (delta.y < 0.0) {
    exitFraction = min(exitFraction, (origin.y - observer.y) / delta.y);
  }
  var last = u32(clamp(floor(exitFraction * f32(count)), f32(first), f32(count - 1u)));
  var inside = false;
  for (var attempt = 0u; attempt < 4u; attempt++) {
    let baseLast = vec2<u32>(floor(clamp(rayPosition(observer, delta, last, count), vec2<f32>(0.0), gridMaximum)));
    if (all(baseLast / vec2<u32>(blockSize) == cell)) {
      inside = true;
      break;
    }
    if (last <= first) {
      break;
    }
    last = last - 1u;
  }
  if (!inside) {
    return 0u;
  }
  let cellIndex = pyramidLevelIndex(level, baseFirst.x, baseFirst.y);
  let heightMaximum = pyramid[pyramidOffset + cellIndex];
  if (heightMaximum <= PYRAMID_EMPTY_MAXIMUM) {
    return last + 1u;
  }
  let heightMinimum = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
  let distanceLast = rayDistance(targetDistance, last, count);
  let magnitude = max(abs(heightMaximum), abs(heightMinimum));
  let dropFirst = curvature * distanceFirst * distanceFirst;
  let dropLast = curvature * distanceLast * distanceLast;
  let dropLow = min(dropFirst, dropLast);
  let dropAbsolute = max(abs(dropFirst), abs(dropLast));
  let numerator = (heightMaximum - eye) - dropLow +
    SKIP_EPSILON * (magnitude + abs(eye) + dropAbsolute);
  let upper = select(
    (numerator / distanceLast) * (1.0 - SKIP_EPSILON),
    (numerator / distanceFirst) * (1.0 + SKIP_EPSILON),
    numerator >= 0.0
  );
  return select(0u, last + 1u, upper <= threshold);
}
// Returns (index of the first sample that must still be evaluated, coarsest level that skipped).
// Every level is an independent valid skip, so the search starts at the level that worked last,
// descends until one works, then climbs while coarser levels also work.
fn skipSamples(
  observer: vec2<f32>,
  delta: vec2<f32>,
  targetDistance: f32,
  first: u32,
  count: u32,
  eye: f32,
  curvature: f32,
  threshold: f32,
  startLevel: u32
) -> vec2<u32> {
  let gridMaximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let baseFirst = vec2<u32>(floor(clamp(rayPosition(observer, delta, first, count), vec2<f32>(0.0), gridMaximum)));
  let distanceFirst = rayDistance(targetDistance, first, count);
  var level = min(startLevel, PYRAMID_LEVEL_COUNT - 1u);
  var next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  while (next == 0u && level > 0u) {
    level = level - 1u;
    next = trySkipLevel(level, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
  }
  if (next == 0u) {
    return vec2<u32>(first, 0u);
  }
  while (level + 1u < PYRAMID_LEVEL_COUNT) {
    let coarser = trySkipLevel(level + 1u, observer, delta, targetDistance, first, count, baseFirst, distanceFirst, eye, curvature, threshold);
    if (coarser == 0u) {
      break;
    }
    next = max(next, coarser);
    level = level + 1u;
  }
  return vec2<u32>(next, level);
}`:``}
// Observer and target are pixel-center positions; eye is the observer elevation (ground + height);
// targetTop is the target elevation (ground + height) before the curvature drop;
// tolerance is [toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction].
fn traceSightLine(
  observer: vec2<f32>,
  targetPosition: vec2<f32>,
  eye: f32,
  targetTop: f32,
  cellSize: vec2<f32>,
  curvature: f32,
  maxDistance: f32,
  tolerance: vec4<f32>
) -> SightLineResult {
  let delta = targetPosition - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    return SightLineResult(OUT_OF_RANGE, 0.0);
  }
  if (targetDistance == 0.0) {
    return SightLineResult(VISIBLE, MAXIMUM_FLOAT);
  }
  let targetElevation = targetTop - curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - eye) / targetDistance;
  let band = (tolerance.x + tolerance.y * targetDistance / 1000.0) / targetDistance;
  let highLimit = targetSlope + band;
  let lowLimit = targetSlope - band;
  let stopDistance = targetDistance - max(tolerance.z, tolerance.w * targetDistance);
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var maxSlope = -MAXIMUM_FLOAT;
  var tested = false;
  var sampleIndex = 1u;
  var skipLevel = 0u;
  loop {
    if (sampleIndex >= stepCount) { break; }
    if (rayDistance(targetDistance, sampleIndex, stepCount) > stopDistance) { break; }${i===`pyramid`?`
    let threshold = ${o?`maxSlope`:`select(lowLimit, highLimit, maxSlope > lowLimit)`};
    let skipped = skipSamples(observer, delta, targetDistance, sampleIndex, stepCount, eye, curvature, threshold, skipLevel);
    if (skipped.x > sampleIndex) {
      sampleIndex = skipped.x;
      skipLevel = skipped.y;
      continue;
    }`:``}
    let current = sampleIndex;
    sampleIndex = sampleIndex + 1u;
    let elevationSample = sampleElevation(rayPosition(observer, delta, current, stepCount));
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = rayDistance(targetDistance, current, stepCount);
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - eye) / sampleDistance;
    tested = true;
    if (sampleSlope > maxSlope) {
      maxSlope = sampleSlope;${o?``:`
      if (maxSlope > highLimit) {
        return SightLineResult(HIDDEN, 0.0);
      }`}
    }
  }
  var code = MARGINAL;
  if (maxSlope > highLimit) {
    code = HIDDEN;
  } else if (maxSlope <= lowLimit) {
    code = VISIBLE;
  }
  return SightLineResult(code, select(MAXIMUM_FLOAT, (targetSlope - maxSlope) * targetDistance, tested));
}`}var O={hidden:0,visible:1,outOfRange:2,noData:3,marginal:4};function k(e,t,n=new Float32Array(10)){p(`getGPUTerrainViewshedParameterValues`,n,10);let{cellSize:r,northEdge:i,southEdge:a}=C(t);if(!r.every(e=>Number.isFinite(e)&&e>0))throw Error(`getGPUTerrainViewshedParameterValues grid cell size must be finite and positive`);return n.set([e.observer[0],e.observer[1],e.observerHeight??1.7,e.targetHeight??0,e.maximumDistance??0,r[0],r[1],e.curvatureCoefficient??0,i,a]),n}function A(e,t=new Float32Array(4)){p(`getGPUTerrainViewshedToleranceParameterValues`,t,4);let n=[e.toleranceMeters??0,e.tolerancePerKilometer??0,e.targetIgnoreDistance??0,e.targetIgnoreFraction??0];if(!n.every(e=>Number.isFinite(e)&&e>=0))throw Error(`getGPUTerrainViewshedToleranceParameterValues values must be finite and non-negative`);return t.set(n),t}function j(t=.13,n=e){return(1-t)/(2*n)}var M=class{id;props;pyramidLayout;constructor(e){this.id=e.id??`terrain-viewshed`,this.props=e;let{id:n}=this;if(e.heightReference!==void 0&&e.heightReference!==`ground`&&e.heightReference!==`absolute`)throw Error(`${n} heightReference must be 'ground' or 'absolute'`);if(e.observerVerticalDatum!==void 0&&e.heightReference!==`absolute`)throw Error(`${n} observerVerticalDatum needs heightReference absolute`);i(n,e.observerVerticalDatum,e.elevation.verticalDatum);let{output:r}=e,o=v(n,e.grid);if(x(n,e.cellSizeMode),f(n,e.grid,e.cellSizeMode),!r.visibility&&!r.visibilityTexture)throw Error(`${n} requires at least one output`);if(r.visibility&&(l(r.visibility,[`uint32`],`${n} output.visibility`),r.visibility.length!==o))throw Error(`${n} output.visibility must contain one value per pixel`);if(b(n,e.parameters,10),m(n,`output.visibilityTexture`,r.visibilityTexture,[`r32uint`,`rgba32uint`],e.grid),e.traversal!==void 0&&e.traversal!==`march`&&e.traversal!==`pyramid`)throw Error(`${n} traversal must be 'march' or 'pyramid'`);if(e.toleranceParameters&&b(n,e.toleranceParameters,4,`toleranceParameters`),e.pyramid&&e.traversal!==`pyramid`)throw Error(`${n} pyramid requires traversal 'pyramid'`);e.traversal===`pyramid`&&(this.pyramidLayout=e.pyramid?t(n,e.pyramid,e.grid):a(e.grid,{firstBlockSize:4,footprint:`bilinear`})),g(n,[r.visibility],[...h(e.elevation),e.parameters,...e.toleranceParameters?[e.toleranceParameters]:[]])}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:a}=n.grid,l=n.cellSizeMode??`uniform`,f=l===`uniform`?``:T(l);d(t,e,n.elevation,[r.visibilityTexture]),o(t,e,[n.pyramid?.combined,n.parameters,r.visibility,...n.toleranceParameters?[n.toleranceParameters]:[]]);let p=i*a,m=_(e,t,n.elevation,n.grid,!0),h=[...m.nodes],g=r.visibility??c(e,`${t}-visibility`,`uint32`,p),v=m.band.validity;if(n.traversal===`pyramid`||n.toleranceParameters?h.push(...this._getSightLineNodes(e,m.band.storage.values,v,g)):h.push(u(e,{id:`${t}-viewshed`,operation:`GPUTerrainViewshed`,variant:`line-of-sight${l===`uniform`?``:`-${l}`}${n.heightReference===`absolute`?`-absolute`:``}`,bindings:[{name:`elevationValues`,view:m.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:v,type:`u32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`visibilityValues`,view:g,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
const HIDDEN: u32 = ${O.hidden}u;
const VISIBLE: u32 = ${O.visible}u;
const OUT_OF_RANGE: u32 = ${O.outOfRange}u;
const NO_DATA: u32 = ${O.noData}u;
${y}${f}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  // Canonical elevation is NaN exactly where a pixel is invalid, so one bit test per corner
  // replaces four validity loads; the interpolation below is unchanged.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00, corner10, fraction.x);
  let bottom = mix(corner01, corner11, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = ${l===`uniform`?`vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u])`:`getRayGroundCell(vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]), settings[settingsOffset + 8u], settings[settingsOffset + 9u], 0.5 * (observer.y + f32(row)) + 0.5)`};
  let curvature = settings[settingsOffset + 7u];
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
${n.heightReference===`absolute`?``:`  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
`}  if (!observerInside || ${n.heightReference===`absolute`?``:`observerSample.y == 0.0 || `}!isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let observerElevation = ${n.heightReference===`absolute`?`observerHeight`:`observerSample.x + observerHeight`};
  let delta = vec2<f32>(f32(column), f32(row)) - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    visibilityValues[visibilityValuesOffset + index] = OUT_OF_RANGE;
    return;
  }
  if (targetDistance == 0.0) {
    visibilityValues[visibilityValuesOffset + index] = VISIBLE;
    return;
  }
  let targetElevation = getElevation(column, row) + targetHeight -
    curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - observerElevation) / targetDistance;
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var visible = true;
  for (var stepIndex = 1u; stepIndex < stepCount; stepIndex++) {
    let fraction = f32(stepIndex) / f32(stepCount);
    let elevationSample = sampleElevation(observer + delta * fraction);
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = targetDistance * fraction;
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - observerElevation) / sampleDistance;
    // A grazing (equal) sight line stays visible.
    if (sampleSlope > targetSlope) {
      visible = false;
      break;
    }
  }
  visibilityValues[visibilityValuesOffset + index] = select(HIDDEN, VISIBLE, visible);`})),r.visibilityTexture){let n=r.visibilityTexture;h.push(...s(e,()=>new w({id:`${t}-visibility-texture`,input:{id:`${t}-visibility-band`,format:`uint32`,storage:{kind:`buffer`,values:g}},output:n,channel:0}).addToGraph(e)))}return h}_getSightLineNodes(e,t,n,i){let{id:a,props:o,pyramidLayout:s}=this,{width:l,height:d}=o.grid,f=o.cellSizeMode??`uniform`,p=f===`uniform`?``:T(f),m=[],h=[{name:`elevationValues`,view:t,type:`f32`,access:`read`},{name:`elevationValidity`,view:n,type:`u32`,access:`read`}];if(s&&o.pyramid)h.push({name:`pyramid`,view:o.pyramid.combined,type:`f32`,access:`read`});else if(s){let i=c(e,`${a}-pyramid`,`float32`,2*s.length);m.push(...r(e,{id:`${a}-pyramid`,layout:s,values:t,validity:n,combined:i})),h.push({name:`pyramid`,view:i,type:`f32`,access:`read`})}return h.push({name:`settings`,view:o.parameters,type:`f32`,access:`read`}),o.toleranceParameters&&h.push({name:`toleranceBand`,view:o.toleranceParameters,type:`f32`,access:`read`}),h.push({name:`visibilityValues`,view:i,type:`u32`,access:`read_write`}),m.push(u(e,{id:`${a}-viewshed`,operation:`GPUTerrainViewshed`,variant:`${o.traversal===`pyramid`?`pyramid`:`tolerance`}${f===`uniform`?``:`-${f}`}${o.heightReference===`absolute`?`-absolute`:``}`,bindings:h,invocationCount:l*d,declarations:D({width:l,height:d,traversal:o.traversal??`march`,layout:s,clearance:!1})+p,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = ${f===`uniform`?`vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u])`:`getRayGroundCell(vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]), settings[settingsOffset + 8u], settings[settingsOffset + 9u], 0.5 * (observer.y + f32(row)) + 0.5)`};
  let curvature = settings[settingsOffset + 7u];
  ${o.toleranceParameters?`let toleranceValues = vec4<f32>(
    toleranceBand[toleranceBandOffset], toleranceBand[toleranceBandOffset + 1u],
    toleranceBand[toleranceBandOffset + 2u], toleranceBand[toleranceBandOffset + 3u]);`:`let toleranceValues = vec4<f32>(0.0);`}
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
${o.heightReference===`absolute`?``:`  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
`}  if (!observerInside || ${o.heightReference===`absolute`?``:`observerSample.y == 0.0 || `}!isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let result = traceSightLine(
    observer, vec2<f32>(f32(column), f32(row)), ${o.heightReference===`absolute`?`observerHeight`:`observerSample.x + observerHeight`},
    getElevation(column, row) + targetHeight, cellSize, curvature, maxDistance, toleranceValues);
  visibilityValues[visibilityValuesOffset + index] = result.code;`})),m}};export{A as a,k as i,O as n,T as o,j as r,D as s,M as t};