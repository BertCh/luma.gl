import{_ as e,i as t,w as n}from"./resources-CwvhEOBe.js";import{l as r,s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a,c as o,i as s,l as c,n as l,o as u,r as d,s as f,u as p}from"./raster-band-utils-CA-sh8xt.js";import{a as m,c as h,l as g,n as _,u as v}from"./raster-grid-D2bmCwkc.js";import{t as y}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";import{a as b,c as x,l as S,n as C,o as w,r as T,s as E,t as D}from"./terrain-illumination-utils-IbBCdCef.js";var O=65536,k=32767,A={cellSizeIndex:0,northEdgeIndex:4},j=64;function M(e,t){return Math.floor((e*t+32768)/O)}function N(e,t,n){let[r,i]=e;if(!Number.isFinite(r)||!Number.isFinite(i)||r===0&&i===0)throw Error(`Terrain sweep direction must be finite and non-zero`);if(!Number.isSafeInteger(t)||!Number.isSafeInteger(n)||t<1||n<1||t>32767||n>32767)throw Error(`Terrain sweep extents must be integers in [1, ${k}]`);let a=Math.abs(r)>=Math.abs(i),o=a?r:i,s=a?i:r,c=Math.round(s/o*O)+0,l=a?t:n,u=a?n:t,d=M(c,l-1);return{xMajor:a,slopeFixed:c,travelSign:o>0?1:-1,firstLine:-Math.max(0,d)+0,lineCount:u+Math.abs(d),stepPixelLength:Math.fround(Math.hypot(1,c/O))}}var P=64,F=7;function I(e,t){let r=t.directions.length,a=new Uint32Array(r*F),o=new Float32Array(1),s=new Int32Array(1),c=e=>(o[0]=e,new Uint32Array(o.buffer)[0]),l=e=>(s[0]=e,new Uint32Array(s.buffer)[0]);for(let[e,n]of t.directions.entries()){let r=N(n,t.width,t.height);a.set([r.xMajor?1:0,l(r.slopeFixed),l(r.travelSign),c(r.stepPixelLength),c(n[0]),c(n[1]),e],e*F)}let u=n(e,`${t.id}-sweep-sector-records`,`uint32`,Math.max(r,1)*P);return{writeNode:i(e,{id:`${t.id}-sweep-sectors`,operation:`GPUTerrainHorizon`,variant:`sweep-sector-table`,bindings:[{name:`records`,view:u,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const SECTOR_WORDS = array<u32, ${a.length}>(${Array.from(a,e=>`${e}u`).join(`, `)});`,body:`for (var sector = 0u; sector < ${r}u; sector++) {
    for (var word = 0u; word < ${F}u; word++) {
      records[recordsOffset + sector * ${P}u + word] =
        SECTOR_WORDS[sector * ${F}u + word];
    }
  }`}),getRecord:t=>e.createDataView(u.buffer,{format:`uint32`,length:F,byteOffset:u.byteOffset+t*P*4})}}var L=`sectorRecord[sectorRecordOffset + 6u]`;function R(e,t){let{width:n,height:a,geometry:o}=t,s=o.kind===`table`?4:5;if(t.outputBindings.length>s)throw Error(`${t.id} sweep supports at most ${s} output bindings`);if(!Number.isFinite(t.maximumRadius)||t.maximumRadius<=0)throw Error(`${t.id} sweep maximumRadius must be positive`);let c=o.kind===`baked`?N(o.direction,n,a):void 0;if(n>32767||a>32767)throw Error(`${t.id} sweep extents must not exceed ${k}`);let l=t.cellSizeMode===`uniform`?`0u`:`HEIGHT / 2u`,u=o.kind===`settings`?o.index:0,d=o.kind===`settings`?o.groundDirectionIndex:0,f=`${o.kind===`table`?`var xMajor = sectorRecord[sectorRecordOffset] != 0u;
  var slopeFixed = bitcast<i32>(sectorRecord[sectorRecordOffset + 1u]);
  var travelSign = bitcast<i32>(sectorRecord[sectorRecordOffset + 2u]);
  var stepPixelLength = bitcast<f32>(sectorRecord[sectorRecordOffset + 3u]);
  let groundDirection = vec2<f32>(
    bitcast<f32>(sectorRecord[sectorRecordOffset + 4u]),
    bitcast<f32>(sectorRecord[sectorRecordOffset + 5u])
  );`:c?`var xMajor = ${c.xMajor?`true`:`false`};
  var slopeFixed: i32 = ${c.slopeFixed};
  var travelSign: i32 = ${c.travelSign};
  var stepPixelLength: f32 = ${r(c.stepPixelLength)};
  let groundDirection = GROUND_DIRECTION;`:`var xMajor = settings[settingsOffset + ${u}u] != 0.0;
  var slopeFixed = i32(settings[settingsOffset + ${u+1}u]);
  var travelSign = select(-1, 1, settings[settingsOffset + ${u+2}u] > 0.0);
  var stepPixelLength = settings[settingsOffset + ${u+3}u];
  ${o.kind===`settings`&&o.rowDirection===`north`?`var`:`let`} groundDirection = vec2<f32>(
    settings[settingsOffset + ${d}u],
    settings[settingsOffset + ${d+1}u]
  );${o.kind===`settings`&&o.rowDirection===`north`?`
  // The packed line assumes south-advancing rows; mirror it for north-advancing rows. Negating the
  // row component negates the slope, and the travel sign of y-major lines.
  groundDirection.y = -groundDirection.y;
  slopeFixed = -slopeFixed;
  if (!xMajor) { travelSign = -travelSign; }`:``}`}
  // The direction is a ground azimuth; on anisotropic cells the line follows the pixel direction
  // of the reference row instead (CPU twin: getTerrainSweepGroundLineGeometry).
  let referenceCell = getGroundCellSize(${l});
  if (referenceCell.x != referenceCell.y && all(referenceCell > vec2<f32>(0.0)) &&
      isFiniteValue(referenceCell.x + referenceCell.y) && any(groundDirection != vec2<f32>(0.0))) {
    let pixelDirection = groundDirection / referenceCell;
    xMajor = abs(pixelDirection.x) >= abs(pixelDirection.y);
    let majorComponent = select(pixelDirection.y, pixelDirection.x, xMajor);
    let minorComponent = select(pixelDirection.x, pixelDirection.y, xMajor);
    slopeFixed = i32(round((minorComponent / majorComponent) * ${O}.0));
    travelSign = select(-1, 1, majorComponent > 0.0);
    stepPixelLength = length(vec2<f32>(1.0, f32(slopeFixed) / ${O}.0));
  }`,p=[{name:`elevationValues`,view:t.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`hull`,view:t.hull,type:`u32`,access:`read_write`},...o.kind===`table`?[{name:`sectorRecord`,view:o.record,type:`u32`,access:`read`}]:[],...t.outputBindings];return i(e,{id:t.id,operation:`GPUTerrainHorizon`,variant:`sweep-${t.cellSizeMode}`,bindings:p,workgroupSize:j,invocationCount:n+a,declarations:`${z(t)}
${o.kind===`baked`?`const GROUND_DIRECTION: vec2<f32> = vec2<f32>(${r(o.direction[0])}, ${r(o.direction[1])});`:``}`,body:`${f}
  let majorExtent = select(HEIGHT, WIDTH, xMajor);
  let minorExtent = select(WIDTH, HEIGHT, xMajor);
  let lastOffset = getFixedOffset(slopeFixed, i32(majorExtent) - 1);
  let lineCount = i32(minorExtent) + abs(lastOffset);
  if (i32(index) >= lineCount) {
    return;
  }
  sweepXMajor = xMajor;
  sweepSlopeFixed = slopeFixed;
  sweepTravel = travelSign;
  sweepMajorExtent = i32(majorExtent);
  sweepMinorExtent = i32(minorExtent);
  sweepLine = i32(index) - max(0, lastOffset);
  sweepZ = f32(${t.zFactorSign}) * settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  sweepMaximumDistance = settings[settingsOffset + 6u];
  sweepStepPixelLength = stepPixelLength;
  let referenceStep = getStepLength(${l}, xMajor, slopeFixed);
  sweepCurvature = curvature * referenceStep * referenceStep;
  var previous = NONE;
  for (var stepIndex = 0; stepIndex < sweepMajorExtent; stepIndex++) {
    let major = select(stepIndex, sweepMajorExtent - 1 - stepIndex, travelSign > 0);
    let minor = sweepLine + getFixedOffset(slopeFixed, major);
    if (minor < 0 || minor >= sweepMinorExtent) {
      continue;
    }
    let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), xMajor);
    let next = previous;
    previous = pixel;
    var firstAhead = NONE;
    if (next != NONE) {
      firstAhead = select(hull[hullOffset + next], next, isValidPixel(next));
    }
    var horizonAngle = getNaN(pixel);
    if (!isValidPixel(pixel)) {
      hull[hullOffset + pixel] = firstAhead;
    } else {
      sweepElevation = elevationValues[elevationValuesOffset + pixel];
      sweepMajor = major;
      var tangent = firstAhead;
      if (tangent != NONE) {
        var chain = hull[hullOffset + tangent];
        while (chain != NONE && isGrowing(chain, tangent)) {
          tangent = chain;
          chain = hull[hullOffset + chain];
        }
      }
      hull[hullOffset + pixel] = tangent;
      let pixelStep = getStepLength(pixel / WIDTH, xMajor, slopeFixed);
      if (pixelStep > 0.0 && isFiniteValue(pixelStep)) {
        sweepPixelStep = pixelStep;
        var best = NONE;
        if (firstAhead != NONE && isInsideWindow(firstAhead)) {
          if (isInsideWindow(tangent)) {
            best = tangent;
          } else {
            best = queryWindow(firstAhead);
          }
        }
        if (best == NONE) {
          horizonAngle = 0.0;
        } else {
          let groundDistance = getDistanceIndex(best) * pixelStep;
          let rise = sweepZ * (elevationValues[elevationValuesOffset + best] - sweepElevation) -
            curvature * groundDistance * groundDistance;
          horizonAngle = atan(rise / groundDistance) * RADIANS_TO_DEGREES;
        }
      }
    }
    {
      ${t.outputWGSL}
    }
  }`})}function z(e){return`const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const NONE: u32 = 0xffffffffu;
const MAXIMUM_RADIUS: f32 = ${r(e.maximumRadius)};
${C}
${m(e.cellSizeMode,A)}
var<private> sweepXMajor: bool;
var<private> sweepSlopeFixed: i32;
var<private> sweepTravel: i32;
var<private> sweepMajorExtent: i32;
var<private> sweepMinorExtent: i32;
var<private> sweepLine: i32;
var<private> sweepZ: f32;
var<private> sweepCurvature: f32;
var<private> sweepMaximumDistance: f32;
var<private> sweepStepPixelLength: f32;
var<private> sweepPixelStep: f32;
var<private> sweepElevation: f32;
var<private> sweepMajor: i32;
fn getFixedOffset(slopeFixed: i32, major: i32) -> i32 {
  return (slopeFixed * major + 32768) >> 16u;
}
// Ground length of one major step on the pixel row, using the row's cell size.
fn getStepLength(row: u32, xMajor: bool, slopeFixed: i32) -> f32 {
  let slope = f32(slopeFixed) / 65536.0;
  let cell = getGroundCellSize(row);
  return length(select(vec2<f32>(slope, 1.0), vec2<f32>(1.0, slope), xMajor) * cell);
}
// Canonical elevation is NaN exactly where a pixel is invalid.
fn isValidPixel(pixel: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + pixel]);
}
fn getMajorIndex(pixel: u32) -> i32 {
  return i32(select(pixel / WIDTH, pixel % WIDTH, sweepXMajor));
}
fn getDistanceIndex(pixel: u32) -> f32 {
  return f32(abs(getMajorIndex(pixel) - sweepMajor));
}
// Hull-metric rise for the sample at distance index kf (curvature uses the reference step).
fn getRise(pixel: u32, kf: f32) -> f32 {
  return sweepZ * (elevationValues[elevationValuesOffset + pixel] - sweepElevation) -
    sweepCurvature * (kf * kf);
}
// True when sample r has a strictly larger tangent than sample c (cross-multiplied, exact ties).
fn isGrowing(r: u32, c: u32) -> bool {
  let kr = getDistanceIndex(r);
  let kc = getDistanceIndex(c);
  return getRise(r, kr) * kc > getRise(c, kc) * kr;
}
fn isInsideWindow(pixel: u32) -> bool {
  let kf = getDistanceIndex(pixel);
  return kf * sweepStepPixelLength <= MAXIMUM_RADIUS &&
    (sweepMaximumDistance <= 0.0 || kf * sweepPixelStep <= sweepMaximumDistance);
}
// First valid pixel strictly after c on the line, NONE at the end.
fn getValidAfter(c: u32) -> u32 {
  let major = getMajorIndex(c) + sweepTravel;
  if (major < 0 || major >= sweepMajorExtent) { return NONE; }
  let minor = sweepLine + getFixedOffset(sweepSlopeFixed, major);
  if (minor < 0 || minor >= sweepMinorExtent) { return NONE; }
  let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), sweepXMajor);
  return select(hull[hullOffset + pixel], pixel, isValidPixel(pixel));
}
// Exact maximum over the in-window samples at or after start (start is valid and inside).
fn queryWindow(start: u32) -> u32 {
  var best = NONE;
  var t = start;
  loop {
    if (t == NONE || !isInsideWindow(t)) { break; }
    var c = t;
    var r = hull[hullOffset + c];
    while (r != NONE && isInsideWindow(r) && isGrowing(r, c)) {
      c = r;
      r = hull[hullOffset + r];
    }
    if (best == NONE) {
      best = c;
    } else if (isGrowing(c, best)) {
      best = c;
    }
    if (r == NONE || isInsideWindow(r)) { break; }
    t = getValidAfter(c);
  }
  return best;
}
${e.outputDeclarations??``}`}function B(e,t,n){let r=e.anisotropyAzimuthDegrees!==void 0||e.anisotropyLevel!==void 0||e.anisotropyMinimumWeight!==void 0,i=r||n!==void 0&&n.length>=12,a=n??new Float32Array(i?12:8);x(`getGPUTerrainHorizonParameterValues`,t),s(`getGPUTerrainHorizonParameterValues`,a,r?12:8);let o=e.anisotropyAzimuthDegrees??315,c=e.anisotropyLevel??4,l=e.anisotropyMinimumWeight??.4;if(i){if(!Number.isFinite(o))throw Error(`Terrain horizon anisotropy azimuth must be finite`);if(!Number.isSafeInteger(c)||c<0)throw Error(`Terrain horizon anisotropy level must be a non-negative integer`);if(!Number.isFinite(l)||l<0||l>1)throw Error(`Terrain horizon anisotropy minimum weight must be in [0, 1]`)}return a.set([0,0,e.zFactor??1,e.curvatureCoefficient??0,0,0,e.maximumDistance??0,0]),v(a,t,A),i&&a.set([o,c,l,0],8),a}function V(e,t=1){if(!Number.isSafeInteger(e)||e<1)throw Error(`Terrain horizon maximumRadius must be a positive integer`);if(!Number.isFinite(t)||t<1)throw Error(`Terrain horizon stepGrowth must be finite and >= 1`);let n=[];for(let r=1;r<=e;)n.push(r),r=Math.max(r+1,r*t);return n[n.length-1]<e&&n.push(e),Float32Array.from(n)}function H(e,t,n=`south`){let r=2*Math.PI*e/t,i=n===`south`?-1:1,a=e=>Math.abs(e)<1e-9?0:Math.fround(e);return[a(Math.sin(r)),a(i*Math.cos(r))]}function U(e){let{sector:t,directionCount:n,mode:i}=e,a=e.horizonFormat??`float32`;if(i===`nadir`){if(e.horizon||e.sineSum||e.angleSum||e.anisotropicSum)throw Error(`Terrain horizon nadir output only accumulates nadirSum`)}else if(e.nadirSum)throw Error(`Terrain horizon zenith output cannot write nadirSum`);let o=[],s=[],c=``,l=e.sectorWGSL?`(${e.sectorWGSL})`:`${t}u`,u=(n,r)=>e.sectorWGSL?`${n} = select(${n} + ${r}, ${r}, ${l} == 0u);`:`${n} ${t===0?`=`:`+=`} ${r};`;if(e.horizon&&(a===`unorm16`?(o.push({name:`horizon`,view:e.horizon,type:`u32`,access:`read_write`}),c+=`fn encodeHorizonUnorm16(angle: f32) -> u32 {
  if (!isFiniteValue(angle)) { return 0u; }
  return 1u + u32(floor(clamp(angle, -90.0, 90.0) * ${r(D)} + 32767.0 + 0.5));
}
`,s.push(`let element = pixel * ${n}u + ${l};
  let wordIndex = horizonOffset + (element >> 1u);
  let shift = (element & 1u) * 16u;
  horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
    (encodeHorizonUnorm16(horizonAngle) << shift);`)):(o.push({name:`horizon`,view:e.horizon,type:`f32`,access:`read_write`}),s.push(`horizon[horizonOffset + pixel * ${n}u + ${l}] = horizonAngle;`))),e.sineSum&&(o.push({name:`sineSum`,view:e.sineSum,type:`f32`,access:`read_write`}),s.push(u(`sineSum[sineSumOffset + pixel]`,`sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`))),e.angleSum&&(o.push({name:`angleSum`,view:e.angleSum,type:`f32`,access:`read_write`}),s.push(u(`angleSum[angleSumOffset + pixel]`,`horizonAngle`))),e.anisotropicSum&&(o.push({name:`anisotropicSum`,view:e.anisotropicSum,type:`f32`,access:`read_write`}),c+=W(n),s.push(u(`anisotropicSum[anisotropicSumOffset + pixel]`,`getAnisotropicWeight(${l}) *
    sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`))),e.nadirSum&&(o.push({name:`nadirSum`,view:e.nadirSum,type:`f32`,access:`read_write`}),s.push(u(`nadirSum[nadirSumOffset + pixel]`,`horizonAngle`))),o.length===0)throw Error(`Terrain horizon sector output requires at least one output`);let d=o.length+(e.reservedBindingCount??3),f=e.maximumBindingCount??8;if(d>f)throw Error(`Terrain horizon sector output needs ${d} storage bindings, more than the limit of ${f}`);return{bindings:o,declarations:c,wgsl:`{\n  ${s.join(`
  `)}\n  }`}}function W(e){return`fn getAnisotropicWeight(sector: u32) -> f32 {
  let mainDirection = settings[settingsOffset + 8u];
  let level = settings[settingsOffset + 9u];
  let minimumWeight = settings[settingsOffset + 10u];
  // Compass sector d is RVT direction (n - d) mod n; RVT angles run counterclockwise from north.
  let rvtDirection = f32((${e}u - sector) % ${e}u) * ${r(360/e)};
  let cosine = cos((rvtDirection - mainDirection) * 0.5 * DEGREES_TO_RADIANS);
  var power = 1.0;
  if (level > 0.0) {
    // pow is undefined for a zero or negative base; odd integer levels keep the sign like RVT.
    power = select(0.0, pow(abs(cosine), level), abs(cosine) > 0.0);
    if (cosine < 0.0 && fract(level * 0.5) != 0.0) { power = -power; }
  }
  return (1.0 - minimumWeight) * power + minimumWeight;
}
`}var G=class{id;props;directionCount;requiredHalo;stepDistances;horizonFormat;algorithm;constructor(e){this.id=e.id??`terrain-horizon`,this.props=e;let{id:t}=this,{output:n}=e,r=f(t,e.grid),i=e.directionCount??16;if(!Number.isSafeInteger(i)||i<4||i>64)throw Error(`${t} directionCount must be an integer in [4, 64]`);this.directionCount=i;try{this.stepDistances=V(e.maximumRadius,e.stepGrowth??1)}catch(e){throw Error(`${t}: ${e.message}`)}if(this.requiredHalo=e.maximumRadius,!n.horizon&&!n.skyViewFactor&&!n.positiveOpenness&&!n.negativeOpenness&&!n.anisotropicSkyViewFactor&&!n.outputValidity&&!n.skyViewFactorTexture)throw Error(`${t} requires at least one output`);if(b(t,e.horizonFormat??`float32`),this.horizonFormat=e.horizonFormat??`float32`,this.algorithm=e.algorithm??`march`,this.algorithm!==`march`&&this.algorithm!==`sweep`)throw Error(`${t} algorithm must be march or sweep`);if(this.algorithm===`sweep`){if((e.stepGrowth??1)!==1)throw Error(`${t} stepGrowth applies to the march algorithm only`);if(e.grid.width>32767||e.grid.height>32767)throw Error(`${t} sweep supports extents up to ${k}`)}if(e.maximumStepsPerDispatch!==void 0&&!(Number.isFinite(e.maximumStepsPerDispatch)&&e.maximumStepsPerDispatch>0))throw Error(`${t} maximumStepsPerDispatch must be positive`);w(t,`output.horizon`,n.horizon,this.horizonFormat,r,i),S(t,`output.skyViewFactor`,n.skyViewFactor,`float32`,r),S(t,`output.positiveOpenness`,n.positiveOpenness,`float32`,r),S(t,`output.negativeOpenness`,n.negativeOpenness,`float32`,r),S(t,`output.anisotropicSkyViewFactor`,n.anisotropicSkyViewFactor,`float32`,r),S(t,`output.outputValidity`,n.outputValidity,`uint32`,r),p(t,e.parameters,n.anisotropicSkyViewFactor?12:8),c(t,`output.skyViewFactorTexture`,n.skyViewFactorTexture,[`r32float`,`rgba32float`],e.grid),h(t,e.cellSizeMode??`uniform`),o(t,e.grid,e.cellSizeMode),g(t,e.rowDirection??`south`),u(t,[n.horizon,n.skyViewFactor,n.positiveOpenness,n.negativeOpenness,n.anisotropicSkyViewFactor,n.outputValidity],[...l(e.elevation),e.parameters])}getCommandNodes(o){let{id:s,props:c,directionCount:l}=this,{output:u}=c,{width:f,height:p}=c.grid;a(s,o,c.elevation,[u.skyViewFactorTexture]),e(s,o,[c.parameters,u.horizon,u.skyViewFactor,u.positiveOpenness,u.negativeOpenness,u.anisotropicSkyViewFactor,u.outputValidity]);let m=f*p;u.horizon&&E(s,`horizon`,T(this.horizonFormat,m,l)*4,o.device.limits.maxStorageBufferBindingSize);let h=d(o,s,c.elevation,c.grid,!0),g=[...h.nodes],_=h.band.storage.values,v=h.band.validity,b=u.skyViewFactor??(u.skyViewFactorTexture?n(o,`${s}-sky-view-factor`,`float32`,m):void 0),x=!!(b||u.positiveOpenness),S=x?n(o,`${s}-sine-sum`,`float32`,m):void 0,w=x?n(o,`${s}-angle-sum`,`float32`,m):void 0,D=u.anisotropicSkyViewFactor?n(o,`${s}-anisotropic-sum`,`float32`,m):void 0,O=u.negativeOpenness?n(o,`${s}-nadir-sum`,`float32`,m):void 0,k=c.rowDirection??`south`,A=c.cellSizeMode??`uniform`,j=Array.from(this.stepDistances,r).join(`, `),M=this.algorithm===`sweep`?n(o,`${s}-sweep-hull`,`uint32`,m):void 0,N=!M&&!(this.horizonFormat===`unorm16`&&l%2==1);N&&g.push(...Z(o,{id:s,width:f,height:p,directionCount:l,rowDirection:k,distances:j,stepCount:this.stepDistances.length,maximumStepsPerDispatch:c.maximumStepsPerDispatch??X,cellSizeMode:A,elevationValues:_,settings:c.parameters,horizonFormat:this.horizonFormat,horizon:u.horizon,sineSum:S,angleSum:w,anisotropicSum:D,nadirSum:O}));let P=!N&&!M?J(o,{id:s,directionCount:l,sectorsPerRange:1}):void 0;P&&g.push(P.writeNode);let F=M?I(o,{id:s,width:f,height:p,directions:Array.from({length:l},(e,t)=>H(t,l,k))}):void 0;F&&g.push(F.writeNode);for(let e=0;!N&&e<l;e++){if(M){g.push(...K(o,{id:`${s}-horizon-${e}`,width:f,height:p,sectorRecord:F.getRecord(e),directionCount:l,maximumRadius:c.maximumRadius,cellSizeMode:A,elevationValues:_,settings:c.parameters,hull:M,horizonFormat:this.horizonFormat,horizon:u.horizon,sineSum:S,angleSum:w,anisotropicSum:D,nadirSum:O}));continue}g.push(Q(o,{id:`${s}-horizon-${e}`,width:f,height:p,sectorRange:P.getRange(e),directionCount:l,rowDirection:k,distances:j,stepCount:this.stepDistances.length,cellSizeMode:A,elevationValues:_,settings:c.parameters,horizonFormat:this.horizonFormat,horizon:u.horizon,sineSum:S,angleSum:w,anisotropicSum:D,nadirSum:O}))}if(b||u.positiveOpenness||u.outputValidity){let e=[{name:`elevationValidity`,view:v,type:`u32`,access:`read`}];S&&w&&e.push({name:`sineSum`,view:S,type:`f32`,access:`read`},{name:`angleSum`,view:w,type:`f32`,access:`read`}),b&&e.push({name:`skyView`,view:b,type:`f32`,access:`read_write`}),u.positiveOpenness&&e.push({name:`openness`,view:u.positiveOpenness,type:`f32`,access:`read_write`}),u.outputValidity&&e.push({name:`validityValues`,view:u.outputValidity,type:`u32`,access:`read_write`}),g.push(i(o,{id:`${s}-sky-view`,operation:`GPUTerrainHorizon`,variant:`sky-view`,bindings:e,invocationCount:m,declarations:`const DIRECTION_COUNT: f32 = ${l}.0;
${C}`,body:`let isValid = elevationValidity[elevationValidityOffset + index] != 0u${S?` &&
    isFiniteValue(angleSum[angleSumOffset + index])`:``};
  let invalidValue = getNaN(index);
  ${b?`skyView[skyViewOffset + index] = select(invalidValue,
    1.0 - sineSum[sineSumOffset + index] / DIRECTION_COUNT, isValid);`:``}
  ${u.positiveOpenness?`openness[opennessOffset + index] = select(invalidValue,
    90.0 - angleSum[angleSumOffset + index] / DIRECTION_COUNT, isValid);`:``}
  ${u.outputValidity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`}))}if(D&&u.anisotropicSkyViewFactor||O&&u.negativeOpenness){let e=[{name:`elevationValidity`,view:v,type:`u32`,access:`read`}],t=`elevationValidity[elevationValidityOffset + index] != 0u`,n=``;D&&u.anisotropicSkyViewFactor&&(e.push({name:`settings`,view:c.parameters,type:`f32`,access:`read`},{name:`anisotropicSum`,view:D,type:`f32`,access:`read`},{name:`anisotropicSkyView`,view:u.anisotropicSkyViewFactor,type:`f32`,access:`read_write`}),t+=` &&
    isFiniteValue(anisotropicSum[anisotropicSumOffset + index])`,n+=`var weightSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT_U32; sector++) {
    weightSum += getAnisotropicWeight(sector);
  }
  anisotropicSkyView[anisotropicSkyViewOffset + index] = select(invalidValue,
    1.0 - anisotropicSum[anisotropicSumOffset + index] / weightSum, isValid);
  `),O&&u.negativeOpenness&&(e.push({name:`nadirSum`,view:O,type:`f32`,access:`read`},{name:`negativeOpennessValues`,view:u.negativeOpenness,type:`f32`,access:`read_write`}),t+=` &&
    isFiniteValue(nadirSum[nadirSumOffset + index])`,n+=`negativeOpennessValues[negativeOpennessValuesOffset + index] = select(invalidValue,
    90.0 - nadirSum[nadirSumOffset + index] / DIRECTION_COUNT, isValid);
  `),g.push(i(o,{id:`${s}-extended`,operation:`GPUTerrainHorizon`,variant:`extended`,bindings:e,invocationCount:m,declarations:`const DIRECTION_COUNT: f32 = ${l}.0;
const DIRECTION_COUNT_U32: u32 = ${l}u;
${C}
${D?W(l):``}`,body:`let isValid = ${t};
  let invalidValue = getNaN(index);
  ${n}`}))}if(u.skyViewFactorTexture&&b){let e=u.skyViewFactorTexture;g.push(...t(o,()=>new y({id:`${s}-sky-view-texture`,input:{id:`${s}-sky-view-band`,format:`float32`,storage:{kind:`buffer`,values:b}},output:e,channel:0}).addToGraph(o)))}return g}};function K(e,t){let n={width:t.width,height:t.height,geometry:{kind:`table`,record:t.sectorRecord},maximumRadius:t.maximumRadius,cellSizeMode:t.cellSizeMode,elevationValues:t.elevationValues,settings:t.settings,hull:t.hull},r=[];if(t.horizon||t.sineSum||t.angleSum||t.anisotropicSum){let i=U({sector:0,sectorWGSL:L,directionCount:t.directionCount,horizonFormat:t.horizonFormat,horizon:t.horizon,sineSum:t.sineSum,angleSum:t.angleSum,anisotropicSum:t.anisotropicSum,mode:`zenith`,reservedBindingCount:4});r.push(R(e,{...n,id:t.id,zFactorSign:1,outputBindings:i.bindings,outputDeclarations:i.declarations,outputWGSL:i.wgsl}))}if(t.nadirSum){let i=U({sector:0,sectorWGSL:L,directionCount:t.directionCount,nadirSum:t.nadirSum,mode:`nadir`,reservedBindingCount:4});r.push(R(e,{...n,id:`${t.id}-nadir`,zFactorSign:-1,outputBindings:i.bindings,outputDeclarations:i.declarations,outputWGSL:i.wgsl}))}return r}var q=64;function J(e,t){let{directionCount:r,sectorsPerRange:a}=t,o=Math.ceil(r/a),s=n(e,`${t.id}-sector-range-words`,`uint32`,o*q);return{rangeCount:o,writeNode:i(e,{id:`${t.id}-sector-ranges`,operation:`GPUTerrainHorizon`,variant:`sector-ranges`,bindings:[{name:`ranges`,view:s,type:`u32`,access:`read_write`}],invocationCount:1,body:`for (var range = 0u; range < ${o}u; range++) {
    let first = range * ${a}u;
    ranges[rangesOffset + range * ${q}u] = first;
    ranges[rangesOffset + range * ${q}u + 1u] = min(${r}u, first + ${a}u);
  }`}),getRange:t=>e.createDataView(s.buffer,{format:`uint32`,length:2,byteOffset:s.byteOffset+t*q*4})}}var Y=`// Canonical elevation is NaN exactly where a pixel is invalid, so the march kernels need no
// validity binding (its slot holds the sector range word).
fn isValidPixel(column: u32, row: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + row * WIDTH + column]);
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation relative to \`origin\` at an in-grid pixel-center position; .y is 0 when any
// corner is invalid. Corners are made origin-relative before interpolating, so the interpolation
// rounds at the scale of the local relief instead of the absolute elevation (at 4000 m the absolute
// form loses about 2.4e-4 m, 1.4e-3 degrees over one 10 m step).
fn sampleElevation(position: vec2<f32>, origin: f32) -> vec2<f32> {
  let base = vec2<u32>(floor(position));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = position - floor(position);
  // Canonical elevation is NaN exactly where a pixel is invalid: one bit test per corner replaces
  // four validity loads.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00 - origin, corner10 - origin, fraction.x);
  let bottom = mix(corner01 - origin, corner11 - origin, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,X=2**30;function Z(e,t){let{directionCount:n}=t,a=t.horizon||t.sineSum||t.angleSum||t.anisotropicSum?U({sector:0,sectorWGSL:`sector`,directionCount:n,horizonFormat:t.horizonFormat,horizon:t.horizon,sineSum:t.sineSum,angleSum:t.angleSum,anisotropicSum:t.anisotropicSum,mode:`zenith`,reservedBindingCount:3+(t.nadirSum?1:0)}):void 0,o=t.nadirSum?U({sector:0,sectorWGSL:`sector`,directionCount:n,nadirSum:t.nadirSum,mode:`nadir`,reservedBindingCount:3+(a?.bindings.length??0)}):void 0,s=[{name:`elevationValues`,view:t.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},...a?.bindings??[],...o?.bindings??[]],c=!!o,l=t.width*t.height,u=Math.max(1,Math.min(n,Math.floor(t.maximumStepsPerDispatch/(l*t.stepCount)))),d=Array.from({length:n},(e,i)=>{let[a,o]=H(i,n,t.rowDirection);return`vec2<f32>(${r(a)}, ${r(o)})`}).join(`, `),f=t.horizonFormat,p=[];t.horizon&&p.push(f===`unorm16`?`// Both sectors of a word belong to this pixel (even sector count), so no other thread shares it.
      let element = pixel * DIRECTION_COUNT + sector;
      let wordIndex = horizonOffset + (element >> 1u);
      let shift = (element & 1u) * 16u;
      horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
        (encodeHorizonUnorm16(horizonAngle) << shift);`:`horizon[horizonOffset + pixel * DIRECTION_COUNT + sector] = horizonAngle;`),(t.sineSum||t.anisotropicSum)&&p.push(`let sineTerm = sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);`,...t.sineSum?[`sineAccumulator = select(sineAccumulator + sineTerm, sineTerm, sector == 0u);`]:[],...t.anisotropicSum?[`let anisotropicTerm = getAnisotropicWeight(sector) * sineTerm;`,`anisotropicAccumulator = select(anisotropicAccumulator + anisotropicTerm, anisotropicTerm, sector == 0u);`]:[]),t.angleSum&&p.push(`angleAccumulator = select(angleAccumulator + horizonAngle, horizonAngle, sector == 0u);`);let h=[t.sineSum?`var sineAccumulator = select(0.0, sineSum[sineSumOffset + pixel], firstSector > 0u);`:``,t.angleSum?`var angleAccumulator = select(0.0, angleSum[angleSumOffset + pixel], firstSector > 0u);`:``,t.anisotropicSum?`var anisotropicAccumulator = select(0.0, anisotropicSum[anisotropicSumOffset + pixel], firstSector > 0u);`:``,t.nadirSum?`var nadirAccumulator = select(0.0, nadirSum[nadirSumOffset + pixel], firstSector > 0u);`:``].join(`
  `),g=[t.sineSum?`sineSum[sineSumOffset + pixel] = sineAccumulator;`:``,t.angleSum?`angleSum[angleSumOffset + pixel] = angleAccumulator;`:``,t.anisotropicSum?`anisotropicSum[anisotropicSumOffset + pixel] = anisotropicAccumulator;`:``,t.nadirSum?`nadirSum[nadirSumOffset + pixel] = nadirAccumulator;`:``].join(`
  `),v=J(e,{id:t.id,directionCount:n,sectorsPerRange:u}),y=[v.writeNode];for(let r=0;r<v.rangeCount;r++)y.push(i(e,{id:`${t.id}-horizon-march-${r}`,operation:`GPUTerrainHorizon`,variant:`march-fused-${t.cellSizeMode}${c?`-nadir`:``}`,bindings:[...s,{name:`sectorRange`,view:v.getRange(r),type:`u32`,access:`read`}],invocationCount:l,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
const DIRECTION_COUNT: u32 = ${n}u;
const DIRECTIONS: array<vec2<f32>, ${n}> = array<vec2<f32>, ${n}>(${d});
const STEP_COUNT: u32 = ${t.stepCount}u;
const STEP_DISTANCES: array<f32, ${t.stepCount}> = array<f32, ${t.stepCount}>(${t.distances});
${C}
${a?.declarations??``}
${m(t.cellSizeMode,{cellSizeIndex:0,northEdgeIndex:4})}
${Y}
${_}`,body:`let pixel = index;
  let firstSector = sectorRange[sectorRangeOffset];
  let endSector = sectorRange[sectorRangeOffset + 1u];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let centerValid = isValidPixel(column, row);
  let centerElevation = select(0.0, getElevation(column, row), centerValid);
  let limit = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let center = vec2<f32>(f32(column), f32(row));
  ${h}
  for (var sector = firstSector; sector < endSector; sector++) {
    // DIRECTIONS are ground azimuths; the pixel step follows them on anisotropic cells. The ray
    // keeps this row's cell size (see GPUTerrainHorizonAlgorithm for the along-ray error).
    let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
    let groundStep = length(direction * groundCell);
    var horizonAngle = getNaN(index);
    var nadirAngle = getNaN(index);
    if (centerValid && groundStep > 0.0 && isFiniteValue(groundStep)) {
      var hasSample = false;
      var maximumTangent = 0.0;
      var maximumNadirTangent = 0.0;
      for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
        let stepDistance = STEP_DISTANCES[stepIndex];
        let position = center + direction * stepDistance;
        if (any(position < vec2<f32>(0.0)) || any(position > limit)) { break; }
        let groundDistance = stepDistance * groundStep;
        if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
        let elevationSample = sampleElevation(position, centerElevation);
        if (elevationSample.y == 0.0) { continue; }
        let rise = zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let tangent = rise / groundDistance;
        if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
        ${c?`// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
        let nadirRise = -zFactor * elevationSample.x -
          curvature * groundDistance * groundDistance;
        let nadirTangent = nadirRise / groundDistance;
        if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`:``}
        hasSample = true;
      }
      horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
      ${c?`nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);`:``}
    }
    ${p.join(`
    `)}
    ${c?`nadirAccumulator = select(nadirAccumulator + nadirAngle, nadirAngle, sector == 0u);`:``}
  }
  ${g}`}));return y}function Q(e,t){let n=[{name:`elevationValues`,view:t.elevationValues,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`sectorRange`,view:t.sectorRange,type:`u32`,access:`read`}],a=t.horizon||t.sineSum||t.angleSum||t.anisotropicSum?U({sector:0,sectorWGSL:`sector`,directionCount:t.directionCount,horizonFormat:t.horizonFormat,horizon:t.horizon,sineSum:t.sineSum,angleSum:t.angleSum,anisotropicSum:t.anisotropicSum,mode:`zenith`,reservedBindingCount:3+(t.nadirSum?1:0)}):void 0,o=t.nadirSum?U({sector:0,sectorWGSL:`sector`,directionCount:t.directionCount,nadirSum:t.nadirSum,mode:`nadir`,reservedBindingCount:3+(a?.bindings.length??0)}):void 0;n.push(...a?.bindings??[],...o?.bindings??[]);let s=!!o,c=Array.from({length:t.directionCount},(e,n)=>{let[i,a]=H(n,t.directionCount,t.rowDirection);return`vec2<f32>(${r(i)}, ${r(a)})`}).join(`, `);return i(e,{id:t.id,operation:`GPUTerrainHorizon`,variant:`march-${t.cellSizeMode}${s?`-nadir`:``}`,bindings:n,invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
const DIRECTION_COUNT: u32 = ${t.directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${t.directionCount}> = array<vec2<f32>, ${t.directionCount}>(${c});
const STEP_COUNT: u32 = ${t.stepCount}u;
const STEP_DISTANCES: array<f32, ${t.stepCount}> = array<f32, ${t.stepCount}>(${t.distances});
${C}
${a?.declarations??``}
${m(t.cellSizeMode,{cellSizeIndex:0,northEdgeIndex:4})}
${Y}
${_}`,body:`let pixel = index;
  let sector = sectorRange[sectorRangeOffset];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
  let groundStep = length(direction * groundCell);
  var horizonAngle = getNaN(index);
  var nadirAngle = getNaN(index);
  if (isValidPixel(column, row) && groundStep > 0.0 && isFiniteValue(groundStep)) {
    let center = vec2<f32>(f32(column), f32(row));
    let centerElevation = getElevation(column, row);
    let limit = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
    var hasSample = false;
    var maximumTangent = 0.0;
    var maximumNadirTangent = 0.0;
    for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
      let stepDistance = STEP_DISTANCES[stepIndex];
      let position = center + direction * stepDistance;
      if (any(position < vec2<f32>(0.0)) || any(position > limit)) { break; }
      let groundDistance = stepDistance * groundStep;
      if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
      let elevationSample = sampleElevation(position, centerElevation);
      if (elevationSample.y == 0.0) { continue; }
      let rise = zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let tangent = rise / groundDistance;
      if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
      ${s?`// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
      let nadirRise = -zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let nadirTangent = nadirRise / groundDistance;
      if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`:``}
      hasSample = true;
    }
    horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
    ${s?`nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);`:``}
  }
  ${a?.wgsl??``}
  ${o?`{\n  let horizonAngle = nadirAngle;\n  ${o.wgsl}\n  }`:``}`})}export{k as a,A as i,B as n,R as o,V as r,N as s,G as t};