import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{i as e,n as t}from"./earth-xB16XgAa.js";import"./analysis-frame-Blu0tYdw.js";import{a as n,i as r,n as i,o as a,r as o,t as s}from"./gpu-raster-extrema-pyramid-DhOnk6uD.js";import{M as c,N as l,_ as u,d,f,h as p,n as m,t as h,w as g,y as _}from"./resources-CwvhEOBe.js";import{t as v}from"./gpu-compaction-Cpx1OCRM.js";import{t as y}from"./gpu-readback-ring-BguS3CAS.js";import{l as b,n as ee,r as te,s as x}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as ne}from"./wgsl-common-BNRVzgd8.js";import{a as S,i as C,n as w,o as T,r as re,s as ie,t as ae,u as oe}from"./raster-band-utils-CA-sh8xt.js";import{r as se}from"./raster-grid-D2bmCwkc.js";import{n as ce,o as le,s as ue}from"./raster-relaxation-BJ40xgro.js";import{a as E,c as de,r as fe,s as pe,t as me}from"./index-Bjk18V2I.js";import{t as D}from"./graph-buffers-B4_80RlX.js";import{t as he}from"./graph-swap-41Ru7LVu.js";import{t as ge}from"./summary-reader-CsgWxENf.js";import"./anchors-D9E9ucGL.js";import"./gazetteer-OhisynFF.js";import{f as _e,l as ve}from"./class-table-BVEo3-Vs.js";import{i as ye}from"./hue-registry-4FjIJ9xz.js";import"./picking-DRmnefrA.js";import{n as O,r as be}from"./gpu-terrain-viewshed-CjsBSBTx.js";import{n as xe,t as Se}from"./gpu-terrain-horizon-ijRQUZ1j.js";import{a as Ce,n as we,o as Te,r as Ee,s as De,t as Oe}from"./b14b-terrain-C8XHhTMr.js";import{r as ke,t as Ae}from"./cpu-dem-DnM3p94o.js";import{P as je,T as Me,a as Ne,t as Pe}from"./terrain-furniture-C6hPKNNK.js";import{a as Fe,i as Ie,l as Le,n as Re,o as ze,s as Be,u as Ve}from"./horizon-style-BXFvJR-L.js";import{a as He,c as Ue,n as We,u as Ge}from"./terrain-places-CXVUagwV.js";import{n as Ke,t as qe}from"./terrain-ground-BLUqElOt.js";var Je={planar:1,"web-mercator":2},Ye=t;function k(t,n,r,i,a){let o=n.projection??`planar`;if(o!==`planar`&&o!==`web-mercator`)throw Error(`${t} projection must be planar or web-mercator`);i[a]=Je[o];let{cellSize:s,northEdge:c}=se(r);if(!s.every(e=>Number.isFinite(e)&&e>0))throw Error(`${t} grid cell size must be finite and positive`);if(o===`web-mercator`){let[n,a]=s;if(Math.abs(n-a)>1e-6*n||r.affine[4]>=0)throw Error(`${t} web-mercator grid needs square pixels with rows increasing south`);i[1]=2*Math.PI*e/n,i[2]=c/e}else i[1]=s[0],i[2]=s[1];i[0]=n.curvatureCoefficient??0,i[3]=n.maximumDistance??0,i[4]=0}function Xe(e,t,n=new Float32Array(8)){let r=`getGPUPointHorizonProfileParameterValues`;return C(r,n,8),n.fill(0,0,8),k(r,e,t,n,7),n}function Ze(e,t,n=new Float32Array(12)){let r=`getGPUPointHorizonVisibilityParameterValues`;C(r,n,12),n.fill(0,0,12),k(r,e,t,n,11),n[5]=e.toleranceDegrees??.02,n[6]=e.sigmaZ??5,n[7]=e.skylineToleranceDegrees??.05;let i=e.targetIgnoreDistance;if(i!==void 0&&!(Number.isFinite(i)&&i>=0))throw Error(`${r} targetIgnoreDistance must be finite and non-negative`);return n[8]=i??-1,n[9]=e.targetIgnoreFraction??0,n}var Qe=1<<24,A=-8;function j(e){let t=Math.floor(Math.log2(e));for(;2**(t+1)<=e;)t++;for(;2**t>e;)t--;return t}function $e(e){let t=e.minimumDistance??20,{maximumDistance:n,cellSize:r}=e,i=e.stepFactor??35e-5,a=e.nearFactor??.01,o=e.cellSteps??.5,s=e.minimumStep??.25;for(let[e,c]of[[`minimumDistance`,t],[`maximumDistance`,n],[`cellSize`,r],[`stepFactor`,i],[`nearFactor`,a],[`cellSteps`,o],[`minimumStep`,s]])if(!(Number.isFinite(c)&&c>0))throw Error(`Point horizon ${e} must be finite and positive`);if(t<1)throw Error(`Point horizon minimumDistance must be at least 1 meter`);if(n<t)throw Error(`Point horizon maximumDistance must not be below minimumDistance`);let c=j(t),l=j(n),u=[],d=0;for(let e=c;e<=l;e++){let f=2**e,p=Math.max(i*f,Math.min(Math.max(a*f,s),r*o)),m=Math.min(Math.max(j(p),A),e);if(e-m>22)throw Error(`Point horizon distance lattice is too fine; increase stepFactor`);let h=2**m,g=f/h,_=e===c?Math.ceil((t-f)/h):0,v=e===l?Math.floor((n-f)/h)+1:g,y=Math.min(v,g);if(!(y<=_)&&(u.push({exponent:e,base:f,step:h,firstJ:_,endJ:y,firstIndex:d}),d+=y-_,d>Qe))throw Error(`Point horizon distance lattice has too many samples`)}if(d===0)throw Error(`Point horizon distance lattice is empty`);let f=e=>{if(!Number.isInteger(e)||e<0||e>=d)throw RangeError(`Point horizon lattice index ${e} out of range`);let t=u[0];for(let n of u)e>=n.firstIndex&&(t=n);return t.base+(e-t.firstIndex+t.firstJ)*t.step},p=e=>{let t=0,n=d;for(;t<n;){let r=t+n>>>1;f(r)>=e?n=r:t=r+1}return t};return{sampleCount:d,octaves:u,minimumDistance:t,maximumDistance:n,firstDistance:f(0),lastDistance:f(d-1),getDistance:f,getCeilIndex:p,getFloorIndex:e=>{let t=p(e);return t<d&&f(t)===e?t:t-1}}}function et(e,t={}){let n=t.maximumLatitude??80,r=t.segmentTolerance??2e-5;if(!(Number.isFinite(n)&&n>=0&&n<=90))throw Error(`Point horizon maximumLatitude must be in [0, 90]`);if(!(Number.isFinite(r)&&r>0))throw Error(`Point horizon segmentTolerance must be finite and positive`);let i=Ye,a=Math.max(.05,Math.tan(Math.min(80,n+3)*Math.PI/180))/i,o=Math.fround(e.maximumDistance),s=[Math.fround(e.minimumDistance)],c=s[0];for(;c<o;){let e=Math.max(200,Math.sqrt(8*r*c/a));c=Math.min(Math.fround(c+e),o),s.push(c)}return s.length===1&&s.push(s[0]+1),{distances:s,sines:s.map(e=>Math.fround(Math.sin(e/i))),oneMinusCosines:s.map(e=>Math.fround(2*Math.sin(e/(2*i))**2)),endIndices:s.slice(1).map((t,n)=>n===s.length-2?e.sampleCount:e.getCeilIndex(t))}}var tt=4;function nt(e,t,n,r){if(t.observerVerticalDatum!==void 0&&r===`ground`)throw Error(`${e} observerVerticalDatum needs heightReference absolute`);a(e,t.observerVerticalDatum,n.verticalDatum)}function rt(e,t,r){let{width:i,height:a}=t;if(!Number.isSafeInteger(i)||!Number.isSafeInteger(a)||i<2||a<2)throw Error(`${e} dimensions must be integers of at least 2`);if(i*a>4294967295)throw Error(`${e} dimensions must be positive integers`);ie(e,t);let s=r.projection??`planar`;if(s!==`planar`&&s!==`web-mercator`)throw Error(`${e} projection must be planar or web-mercator`);let c=r.rowDirection??`south`;if(c!==`south`&&c!==`north`)throw Error(`${e} rowDirection must be south or north`);let l=r.heightReference??`ground`;if(l!==`ground`&&l!==`absolute`)throw Error(`${e} heightReference must be ground or absolute`);let u=r.traversal??`pyramid`;if(u!==`march`&&u!==`pyramid`)throw Error(`${e} traversal must be march or pyramid`);if(r.pyramid&&u!==`pyramid`)throw Error(`${e} pyramid requires the pyramid traversal`);let d=r.azimuthCount??720;if(!Number.isInteger(d)||d<1||d>Qe)throw Error(`${e} azimuthCount must be an integer in [1, 2^24]`);let f=r.firstAzimuth??0;if(!Number.isInteger(f)||f<0||f>=d)throw Error(`${e} firstAzimuth must be an integer in [0, azimuthCount)`);let p=r.azimuthSpan??d;if(!Number.isInteger(p)||p<1||p>d)throw Error(`${e} azimuthSpan must be an integer in [1, azimuthCount]`);let m;try{m=$e(r)}catch(t){throw Error(`${e} ${t.message}`)}return{grid:t,width:i,height:a,projection:s,rowDirection:c,heightReference:l,traversal:u,azimuthCount:d,firstAzimuth:f,azimuthSpan:p,lattice:m,segments:s===`web-mercator`?et(m,r):null,layout:r.pyramid?n(e,r.pyramid,t):u===`pyramid`?o(t,{firstBlockSize:tt,footprint:`bilinear`}):null,sharedPyramid:r.pyramid??null}}function it(e,t,n,r){c(n,[r],`${e} ${t}`)}var at=b,ot=34028234663852886e22;function M(e){return`array<f32, ${e.length}>(${e.map(at).join(`, `)})`}function N(e){return`array<u32, ${e.length}>(${e.map(e=>`${e}u`).join(`, `)})`}function P(e,t){let n=t[t.length-1];for(let r=t.length-2;r>=0;r--)n=`pinMad(${e}, ${n}, ${t[r]})`;return n}var F=`
const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = 1.57079632679489661923;
const QUARTER_PI: f32 = 0.78539816339744830962;
const DEGREES: f32 = 57.29577951308232;
const INV_TWO_PI: f32 = 0.15915494309189535;
const TWO_PI: f32 = 6.28318530717958647692;
const BIG: f32 = ${at(ot)};
const EARTH_RADIUS: f32 = ${Ye};

// XOR with the always-zero settings word: the compiler cannot fold it, so Metal fast math can
// neither re-associate (v - hi) - lo nor fuse a product into the following add.
fn opaque(x: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(x) ^ bitcast<u32>(settings[settingsOffset + 4u]));
}

// a * b + c with the product and the sum both pinned (no fma contraction, no re-association).
fn pinMad(a: f32, b: f32, c: f32) -> f32 {
  return opaque(opaque(a * b) + c);
}

${ne}
// NaN built from the runtime zero word: WGSL rejects a constant-evaluated NaN.
fn nanValue() -> f32 { return bitcast<f32>(0x7fc00000u | bitcast<u32>(settings[settingsOffset + 4u])); }

// Odd Taylor series of atan for |z| <= tan(pi/12) = 0.2679.
fn atanSeries(z: f32) -> f32 {
  let z2 = opaque(z * z);
  return opaque(z * ${P(`z2`,[`1.0`,`(-1.0 / 3.0)`,`(1.0 / 5.0)`,`(-1.0 / 7.0)`,`(1.0 / 9.0)`,`(-1.0 / 11.0)`,`(1.0 / 13.0)`,`(-1.0 / 15.0)`])});
}

// atan on [0, 1]: pi/6 + atan((z sqrt(3) - 1) / (z + sqrt(3))) above tan(pi/12).
fn atanUnit(a: f32) -> f32 {
  if (a > 0.2679491924311227) {
    let numerator = opaque(opaque(a * 1.7320508075688772) - 1.0);
    return opaque(0.5235987755982988 + atanSeries(numerator / (a + 1.7320508075688772)));
  }
  return atanSeries(a);
}

// atan to about 2 ULP plus one division, without the builtin.
fn atanAccurate(z: f32) -> f32 {
  let a = abs(z);
  var r: f32;
  if (a > 1.0) {
    r = opaque(HALF_PI - atanUnit(1.0 / a));
  } else {
    r = atanUnit(a);
  }
  return select(-r, r, z >= 0.0);
}

// atan2(y, x) for x > 0 with the series branch of mt-image's atanS, builtin for x <= 0.
fn atanS(y: f32, x: f32) -> f32 {
  if (x > 0.0) {
    return atanAccurate(y / x);
  }
  return atan2(y, x);
}

// atanh(z) for |z| < 0.2 (odd series to z^13), log form otherwise.
fn atanhS(z: f32) -> f32 {
  if (abs(z) < 0.2) {
    let z2 = opaque(z * z);
    return opaque(z * ${P(`z2`,[`1.0`,`(1.0 / 3.0)`,`(1.0 / 5.0)`,`(1.0 / 7.0)`,`(1.0 / 9.0)`,`(1.0 / 11.0)`,`(1.0 / 13.0)`])});
  }
  return 0.5 * log((1.0 + z) / (1.0 - z));
}

// sin and cos series for |x| <= pi / 2 (error below 1e-9).
fn sinSeries(x: f32) -> f32 {
  let x2 = opaque(x * x);
  return opaque(x * ${P(`x2`,[`1.0`,`(-1.0 / 6.0)`,`(1.0 / 120.0)`,`(-1.0 / 5040.0)`,`(1.0 / 362880.0)`,`(-1.0 / 39916800.0)`,`(1.0 / 6227020800.0)`,`(-1.0 / 1307674368000.0)`])});
}
fn cosSeries(x: f32) -> f32 {
  let x2 = opaque(x * x);
  return ${P(`x2`,[`1.0`,`(-1.0 / 2.0)`,`(1.0 / 24.0)`,`(-1.0 / 720.0)`,`(1.0 / 40320.0)`,`(-1.0 / 3628800.0)`,`(1.0 / 479001600.0)`,`(-1.0 / 87178291200.0)`])};
}

// Azimuth index i of N (angle i * 360 / N degrees clockwise from north) to (sin, cos) with no
// builtin trigonometry: 8 i = octant * N + r exactly in integers, the octant remainder is folded
// into [0, pi/4], series evaluate it, and a quadrant rotation finishes. (sin, cos) = (east, north).
fn azimuthSinCos(index: u32, count: u32) -> vec2<f32> {
  let eight = 8u * (index % count);
  let octant = eight / count;
  let remainder = eight - octant * count;
  let odd = (octant & 1u) == 1u;
  let folded = select(remainder, count - remainder, odd);
  let angle = opaque(opaque(f32(folded) / f32(count)) * QUARTER_PI);
  let s0 = sinSeries(angle);
  let c0 = cosSeries(angle);
  let s = select(s0, c0, odd);
  let c = select(c0, s0, odd);
  let quadrant = octant >> 1u;
  if (quadrant == 0u) { return vec2<f32>(s, c); }
  if (quadrant == 1u) { return vec2<f32>(c, -s); }
  if (quadrant == 2u) { return vec2<f32>(-s, -c); }
  return vec2<f32>(-c, s);
}

// Normalised Mercator offset (dx, dy) of the great-circle point at angular distance D (sinD, omc =
// 1 - cos D) along azimuth (sinA, cosA) from the eye (sinP1, cosP1, c2 = cosP1^2).
fn bp(sinD: f32, omc: f32, sinA: f32, cosA: f32, sinP1: f32, cosP1: f32, c2: f32) -> vec2<f32> {
  let ds = opaque(opaque(opaque(cosP1 * sinD) * cosA) - opaque(sinP1 * omc)); // sin(phi2) - sin(phi1)
  let den = opaque(c2 - opaque(sinP1 * ds)); // 1 - sin(phi1) sin(phi2)
  let dl = atanS(opaque(opaque(sinA * sinD) * cosP1), opaque(den - omc)); // cos D - sin(phi1) sin(phi2)
  let dy = atanhS(ds / den); // atanh(sin phi2) - atanh(sin phi1)
  return vec2<f32>(opaque(dl * INV_TWO_PI), opaque(-dy * INV_TWO_PI));
}

// sinh with a series below 0.5 (the exp form cancels there).
fn sinhS(x: f32) -> f32 {
  if (abs(x) < 0.5) {
    let x2 = opaque(x * x);
    return opaque(x * ${P(`x2`,[`1.0`,`(1.0 / 6.0)`,`(1.0 / 120.0)`,`(1.0 / 5040.0)`,`(1.0 / 362880.0)`,`(1.0 / 39916800.0)`,`(1.0 / 6227020800.0)`])});
  }
  return 0.5 * (exp(x) - exp(-x));
}
fn coshS(x: f32) -> f32 {
  let e = exp(abs(x));
  return 0.5 * (e + 1.0 / e);
}
// tanh with a series below 0.3, the exp form above.
fn tanhS(x: f32) -> f32 {
  let a = abs(x);
  var r: f32;
  if (a < 0.3) {
    let x2 = opaque(x * x);
    r = opaque(a * ${P(`x2`,[`1.0`,`(-1.0 / 3.0)`,`(2.0 / 15.0)`,`(-17.0 / 315.0)`,`(62.0 / 2835.0)`,`(-1382.0 / 155925.0)`])});
  } else {
    let e = exp(-2.0 * a);
    r = (1.0 - e) / (1.0 + e);
  }
  return select(-r, r, x >= 0.0);
}
// asin for z < 0.2 by series, builtin above.
fn asinS(z: f32) -> f32 {
  if (z < 0.2) {
    let z2 = opaque(z * z);
    return opaque(z * ${P(`z2`,[`1.0`,`(1.0 / 6.0)`,`(3.0 / 40.0)`,`(5.0 / 112.0)`,`(35.0 / 1152.0)`,`(63.0 / 2816.0)`,`(231.0 / 13312.0)`])});
  }
  return asin(min(z, 1.0));
}
`;function st(e,t){let{lattice:n,segments:i,layout:a}=e,o=e.projection===`web-mercator`,s=e.traversal===`pyramid`,c=n.octaves,l=i?i.distances.length-1:1,u=i?i.endIndices:[n.sampleCount],d=e.heightReference===`ground`?`let ground = sampleGround(column, row);
  if (ground.y == 0.0 || !isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = ground.x;
  eye.lo = height;`:`if (!isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = height;
  eye.lo = 0.0;`,f=o?`let m1 = settings[settingsOffset + 2u] - TWO_PI * (row + 0.5) / settings[settingsOffset + 1u];
  eye.sinP = tanhS(m1);
  eye.cosP = 1.0 / coshS(m1);
  eye.m = m1;`:``,p=o?`let worldPixelSize = settings[settingsOffset + 1u];
  let c2 = opaque(eye.cosP * eye.cosP);
  var b0 = bp(SEGMENT_SINES[0], SEGMENT_OMC[0], sinA, cosA, eye.sinP, eye.cosP, c2);`:`let dirU = opaque(sinA / settings[settingsOffset + 1u]);
  let dirV = opaque(${e.rowDirection===`south`?`-`:``}cosA / settings[settingsOffset + 2u]);`,m=o?`let b1 = bp(SEGMENT_SINES[segment + 1u], SEGMENT_OMC[segment + 1u], sinA, cosA, eye.sinP, eye.cosP, c2);
    let dA = SEGMENT_DISTANCES[segment];
    let invLength = opaque(1.0 / (SEGMENT_DISTANCES[segment + 1u] - dA));
    let uA = eye.fx + opaque(b0.x * worldPixelSize);
    let vA = eye.fy + opaque(b0.y * worldPixelSize);
    let du = opaque(opaque((b1.x - b0.x) * worldPixelSize) * invLength);
    let dv = opaque(opaque((b1.y - b0.y) * worldPixelSize) * invLength);`:`let dA = 0.0;
    let uA = eye.fx;
    let vA = eye.fy;
    let du = dirU;
    let dv = dirV;`,h=o?`b0 = b1;`:``,g=s?`if (n >= noTest) {
        var skipTo = n;
        var skipped = false;
        for (var level = 0u; level < PYRAMID_LEVEL_COUNT; level++) {
          let cellIndex = pyramidLevelIndex(level, u32(p.xi), u32(p.yi));
          let hMax = pyramid[pyramidOffset + cellIndex];
          let empty = hMax <= PYRAMID_EMPTY_MAXIMUM;
          if (!empty && !has) { break; }
          let blockSize = i32(pyramidLevelBlockSize(level));
          let blockX = (p.xi / blockSize) * blockSize;
          let blockY = (p.yi / blockSize) * blockSize;
          // Analytic exit of the ray from the block (integer part exact).
          var exitX = BIG;
          if (du > 0.0) { exitX = (f32(blockX + blockSize - p.xi) - p.fx) / du; }
          else if (du < 0.0) { exitX = (f32(blockX - p.xi) - p.fx) / du; }
          var exitY = BIG;
          if (dv > 0.0) { exitY = (f32(blockY + blockSize - p.yi) - p.fy) / dv; }
          else if (dv < 0.0) { exitY = (f32(blockY - p.yi) - p.fy) / dv; }
          let exitDistance = d + min(exitX, exitY);
          var hMin = 0.0;
          if (!empty) {
            hMin = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
            // Cheap rejection with an upper estimate of the last skipped distance: any verified
            // last sample is below the exit distance, the segment end and the ray end, and the
            // bound only loosens as the last distance grows, so failing here loses nothing exact.
            let farthest = min(exitDistance, min(${o?`SEGMENT_DISTANCES[segment + 1u]`:`BIG`}, maximumDistance));
            if (skipBound(hMax, hMin, eye, d, farthest, curvature) > tBest) {
              if (level == 0u) { noTest = max(n + 1u, latCeilIndex(exitDistance)); }
              break;
            }
          }
          // Clamp to the last sample of the segment and of the ray, then verify and back off.
          var candidate = latFloorIndex(exitDistance);
          candidate = min(candidate, min(i32(segmentEnd) - 1, rayLast));
          var verified = false;
          var dEnd = d;
          for (var attempt = 0u; attempt < 4u; attempt++) {
            if (candidate < i32(n)) { break; }
            let candidateOctave = latOctave(u32(candidate));
            let candidateDistance = latDistance(
              candidateOctave, u32(candidate) - LAT_FIRST[candidateOctave] + LAT_J0[candidateOctave]);
            let end = getPosition(candidateDistance, dA, uA, vA, du, dv, eye);
            if (inGrid(end) && end.xi >= blockX && end.xi < blockX + blockSize &&
                end.yi >= blockY && end.yi < blockY + blockSize) {
              verified = true;
              dEnd = candidateDistance;
              break;
            }
            candidate = candidate - 1;
          }
          if (!verified) { break; }
          if (!empty && skipBound(hMax, hMin, eye, d, dEnd, curvature) > tBest) {
            if (level == 0u) { noTest = u32(candidate) + 1u; }
            break;
          }
          skipped = true;
          skipTo = max(skipTo, u32(candidate) + 1u);
        }
        if (skipped) {
          n = skipTo;
          if (n < LAT_COUNT) {
            octave = latOctave(n);
            j = n - LAT_FIRST[octave] + LAT_J0[octave];
          }
          continue;
        }
      }`:``;return`
const WIDTH: u32 = ${e.width}u;
const HEIGHT: u32 = ${e.height}u;
const WIDTH_I: i32 = ${e.width};
const HEIGHT_I: i32 = ${e.height};
const AZIMUTH_COUNT: u32 = ${e.azimuthCount}u;
const FIRST_AZIMUTH: u32 = ${e.firstAzimuth}u;
const AZIMUTH_SPAN: u32 = ${e.azimuthSpan}u;
const SKIP_EPSILON: f32 = 7.62939453125e-6;
${F}
// Exact power-of-two distance lattice: octave k holds 2^k + j * step_k, all exact in f32.
const LAT_OCTAVE_COUNT: u32 = ${c.length}u;
const LAT_K0: i32 = ${c[0].exponent};
const LAT_COUNT: u32 = ${n.sampleCount}u;
const LAT_FIRST_DISTANCE: f32 = ${at(n.firstDistance)};
const LAT_LAST_DISTANCE: f32 = ${at(n.lastDistance)};
const MAXIMUM_DISTANCE: f32 = ${at(n.maximumDistance)};
const LAT_BASE = ${M(c.map(e=>e.base))};
const LAT_STEP = ${M(c.map(e=>e.step))};
const LAT_INV_STEP = ${M(c.map(e=>1/e.step))};
const LAT_J0 = ${N(c.map(e=>e.firstJ))};
const LAT_J1 = ${N(c.map(e=>e.endJ))};
const LAT_FIRST = ${N(c.map(e=>e.firstIndex))};
const SEGMENT_COUNT: u32 = ${l}u;
const SEGMENT_END = ${N(u)};
${i?`const SEGMENT_DISTANCES = ${M(i.distances)};
const SEGMENT_SINES = ${M(i.sines)};
const SEGMENT_OMC = ${M(i.oneMinusCosines)};`:``}
${a?r(a,`pyramid`):``}

fn latDistance(octave: u32, j: u32) -> f32 {
  return LAT_BASE[octave] + f32(j) * LAT_STEP[octave];
}

fn latOctave(index: u32) -> u32 {
  var octave = 0u;
  for (var i = 1u; i < LAT_OCTAVE_COUNT; i++) {
    if (index >= LAT_FIRST[i]) { octave = i; }
  }
  return octave;
}

// Index of the first lattice sample with distance >= x (LAT_COUNT if none). The octave comes from
// the f32 exponent of x; x - 2^k and the division by a power of two are exact.
fn latCeilIndex(x: f32) -> u32 {
  if (!(x > LAT_FIRST_DISTANCE)) { return 0u; }
  if (x > LAT_LAST_DISTANCE) { return LAT_COUNT; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = ceil((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(max(u32(steps), LAT_J0[octave]), LAT_J1[octave]);
  return LAT_FIRST[octave] + j - LAT_J0[octave];
}

// Index of the last lattice sample with distance <= x (-1 if none).
fn latFloorIndex(x: f32) -> i32 {
  if (!(x >= LAT_FIRST_DISTANCE)) { return -1; }
  if (x >= LAT_LAST_DISTANCE) { return i32(LAT_COUNT) - 1; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = floor((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(u32(steps), LAT_J1[octave] - 1u);
  return i32(LAT_FIRST[octave] + j - LAT_J0[octave]);
}

// Bilinear height relative to the eye. Each corner is (v - hi) - lo with the subtraction pinned by
// opaque(): near the camera an f32 height formed after interpolation would carry the rounding of a
// 4-digit height, about 1e-3 degrees at 2 m. Products are pinned too so the result does not depend
// on the compiler's fused multiply-add choices (the pyramid bound relies on a stable value).
// Returns (height, 1), or (0, 0) when a corner is invalid. Canonical elevation is NaN exactly where a
// pixel is invalid, so one bit test per loaded corner replaces four validity loads.
fn bilinearRelative(cell: u32, fx: f32, fy: f32, hi: f32, lo: f32) -> vec2<f32> {
  let v00 = elevationValues[elevationValuesOffset + cell];
  let v10 = elevationValues[elevationValuesOffset + cell + 1u];
  let v01 = elevationValues[elevationValuesOffset + cell + WIDTH];
  let v11 = elevationValues[elevationValuesOffset + cell + WIDTH + 1u];
  if (!isFiniteValue(v00) || !isFiniteValue(v10) || !isFiniteValue(v01) || !isFiniteValue(v11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let a0 = opaque(v00 - hi) - lo;
  let a1 = opaque(v10 - hi) - lo;
  let c0 = opaque(v01 - hi) - lo;
  let c1 = opaque(v11 - hi) - lo;
  let t1 = opaque(a0 - a1);
  let t2 = opaque(t1 - c0);
  let twist = t2 + c1;
  let rowSlope = (c0 - a0) + opaque(twist * fx);
  return vec2<f32>((a0 + opaque((a1 - a0) * fx)) + opaque(rowSlope * fy), 1.0);
}

// Bilinear height relative to (hi, lo) at a pixel-center position inside [0, W-1] x [0, H-1];
// .y is 0 when a corner is invalid.
fn sampleRelative(column: f32, row: f32, hi: f32, lo: f32) -> vec2<f32> {
  let x0 = min(i32(floor(column)), WIDTH_I - 2);
  let y0 = min(i32(floor(row)), HEIGHT_I - 2);
  let cell = u32(y0) * WIDTH + u32(x0);
  return bilinearRelative(cell, column - f32(x0), row - f32(y0), hi, lo);
}

fn sampleGround(column: f32, row: f32) -> vec2<f32> {
  return sampleRelative(column, row, 0.0, 0.0);
}

fn isInsideGrid(column: f32, row: f32) -> bool {
  return column >= 0.0 && row >= 0.0 && column <= f32(WIDTH - 1u) && row <= f32(HEIGHT - 1u);
}

struct EyeState {
  valid: bool,
  column: i32,
  row: i32,
  fx: f32,
  fy: f32,
  hi: f32,
  lo: f32,
  sinP: f32,
  cosP: f32,
  m: f32,
};

// Eye at (column, row) in pixel-center index space: integer pixel plus f32 fraction, and the eye
// height as an f32 (hi, lo) pair so (value - hi) - lo is exact to f32 near the eye.
// Settings packed for the other projection (tag 1 planar, 2 Web Mercator; 0 untagged) are rejected.
fn isProjectionTagValid() -> bool {
  let tag = settings[settingsOffset + ${t}u];
  return tag == 0.0 || tag == ${Je[e.projection]}.0;
}

fn makeEye(column: f32, row: f32, height: f32) -> EyeState {
  var eye: EyeState;
  eye.valid = isInsideGrid(column, row) && isProjectionTagValid();
  if (!eye.valid) { return eye; }
  let floorColumn = floor(column);
  let floorRow = floor(row);
  eye.column = i32(floorColumn);
  eye.row = i32(floorRow);
  eye.fx = column - floorColumn;
  eye.fy = row - floorRow;
  ${d}
  ${f}
  return eye;
}

struct Position { xi: i32, yi: i32, fx: f32, fy: f32 };

// Position of the sample at distance d inside the segment starting at dA, relative to the eye's
// integer pixel; the same code serves the march, the skip verification and both kernels.
fn getPosition(d: f32, dA: f32, uA: f32, vA: f32, du: f32, dv: f32, eye: EyeState) -> Position {
  let f = d - dA;
  let uf = uA + opaque(f * du);
  let vf = vA + opaque(f * dv);
  let floorU = floor(uf);
  let floorV = floor(vf);
  return Position(eye.column + i32(floorU), eye.row + i32(floorV), uf - floorU, vf - floorV);
}

// Bilinear footprint of a sample must be inside the grid, otherwise the ray ends.
fn inGrid(p: Position) -> bool {
  return p.xi >= 0 && p.yi >= 0 && p.xi < WIDTH_I - 1 && p.yi < HEIGHT_I - 1;
}

// Upper bound of the tangent of every sample with distance in [d, dEnd] inside a cell of extrema
// (hMax, hMin), see the proof above. EPS pads every rounding, FMA fusion and re-association.
fn skipBound(hMax: f32, hMin: f32, eye: EyeState, d: f32, dEnd: f32, curvature: f32) -> f32 {
  let magnitude = max(abs(hMax - eye.hi), abs(hMin - eye.hi)) + abs(eye.lo);
  let numerator = (hMax - eye.hi) - eye.lo + SKIP_EPSILON * magnitude;
  let ratio = select(
    (numerator / dEnd) * (1.0 - SKIP_EPSILON),
    (numerator / d) * (1.0 + SKIP_EPSILON),
    numerator >= 0.0
  );
  let drop = min(d * curvature, dEnd * curvature);
  return ratio - drop + SKIP_EPSILON * abs(drop);
}

struct RayResult { has: bool, t: f32, d: f32, tQ: f32, evaluated: u32 };

// Marches one ray. Returns the maximum tangent tBest = hr / d - d c over valid samples and its
// distance. tQ is the running maximum when the first sample with d >= q is reached (q <= 0 or
// !useQ: -BIG). evaluated counts the samples whose bilinear height was read (a debug measure of
// the pyramid's skipping).
fn marchRay(eye: EyeState, sinA: f32, cosA: f32, q: f32, useQ: bool) -> RayResult {
  let curvature = settings[settingsOffset];
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(LAT_LAST_DISTANCE, min(maximumFrame, LAT_LAST_DISTANCE), maximumFrame > 0.0);
  let rayLast = latFloorIndex(maximumDistance);
  ${p}
  var has = false;
  var tBest = 0.0;
  var dBest = 0.0;
  var qDone = !useQ;
  var tQ = -BIG;
  var n = 0u;
  var octave = 0u;
  var j = LAT_J0[0];
  var ended = false;
  var noTest = 0u;
  var evaluated = 0u;
  for (var segment = 0u; segment < SEGMENT_COUNT; segment++) {
    if (ended || i32(n) > rayLast) { break; }
    let segmentEnd = SEGMENT_END[segment];
    ${m}
    loop {
      if (n >= segmentEnd || i32(n) > rayLast) { break; }
      let d = latDistance(octave, j);
      if (!qDone && d >= q) {
        tQ = select(-BIG, tBest, has);
        qDone = true;
      }
      let p = getPosition(d, dA, uA, vA, du, dv, eye);
      if (!inGrid(p)) {
        ended = true;
        break;
      }
      ${g}
      let cell = u32(p.yi) * WIDTH + u32(p.xi);
      evaluated = evaluated + 1u;
      let relativeSample = bilinearRelative(cell, p.fx, p.fy, eye.hi, eye.lo);
      if (relativeSample.y != 0.0) {
        let hr = relativeSample.x;
        let t = hr / d - opaque(d * curvature);
        if (!has || t > tBest) {
          has = true;
          tBest = t;
          dBest = d;
        }
      }
      n = n + 1u;
      j = j + 1u;
      if (j >= LAT_J1[octave]) {
        octave = octave + 1u;
        j = 0u;
      }
    }
    ${h}
  }
  if (!qDone) { tQ = select(-BIG, tBest, has); }
  return RayResult(has, tBest, dBest, tQ, evaluated);
}
`}var ct=64;function lt(e,t){let n=Math.min(t.maximumChunkSize,Math.max(t.itemCount,1)),r=Math.max(Math.ceil(t.itemCount/n),1),i=g(e,`${t.id}-chunk-words`,`uint32`,r*ct);return{chunkCount:r,chunkSize:n,writeNode:x(e,{id:`${t.id}-chunks`,operation:t.operation,variant:`chunk-starts`,bindings:[{name:`chunkWords`,view:i,type:`u32`,access:`read_write`}],invocationCount:r,body:`chunkWords[chunkWordsOffset + index * ${ct}u] = index * ${n}u;`}),getChunkStart:t=>({name:`chunkStart`,view:e.createDataView(i.buffer,{format:`uint32`,length:1,byteOffset:i.byteOffset+t*ct*4}),type:`u32`,access:`read`})}}function I(e,t,n,r){let a=re(e,t,r,n.grid,!0),o=[...a.nodes],s=a.band.storage.values,c=a.band.validity,l=[{name:`elevationValues`,view:s,type:`f32`,access:`read`}];if(n.layout&&n.sharedPyramid)l.push({name:`pyramid`,view:n.sharedPyramid.combined,type:`f32`,access:`read`});else if(n.layout){let r=g(e,`${t}-pyramid`,`float32`,2*n.layout.length);o.push(...i(e,{id:`${t}-pyramid`,layout:n.layout,values:s,validity:c,combined:r})),l.push({name:`pyramid`,view:r,type:`f32`,access:`read`})}return{nodes:o,bindings:l}}var ut=class{id;props;model;constructor(e){this.id=e.id??`point-horizon-profile`,this.props=e;let{id:t}=this;this.model=rt(t,e.grid,e),nt(t,e,e.elevation,this.model.heightReference);let{output:n}=e;if(!n.tangent&&!n.skylineAngle&&!n.distance)throw Error(`${t} requires at least one output`);if(oe(t,e.parameters,8),it(t,`observers`,e.observers,`float32x4`),e.observers.length<1)throw Error(`${t} requires at least one observer`);let r=e.observers.length*this.model.azimuthSpan;for(let[e,i]of[[`tangent`,n.tangent],[`skylineAngle`,n.skylineAngle],[`distance`,n.distance]])if(i&&(it(t,`output.${e}`,i,`float32`),i.length!==r))throw Error(`${t} output.${e} must contain observers * azimuthSpan values`);let i=e.raysPerDispatch??65536;if(!Number.isInteger(i)||i<1)throw Error(`${t} raysPerDispatch must be a positive integer`);T(t,[n.tangent,n.skylineAngle,n.distance,n.samples],[...w(e.elevation),e.parameters,e.observers])}getCommandNodes(e){let{id:t,props:n,model:r}=this,{output:i}=n;S(t,e,n.elevation,[]),u(t,e,[n.pyramid?.combined,n.parameters,n.observers,i.tangent,i.skylineAngle,i.distance,i.samples]);let a=I(e,t,r,n.elevation),o=[...a.nodes],s=[...a.bindings,{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`observers`,view:n.observers,type:`f32`,access:`read`}];i.tangent&&s.push({name:`tangentOutput`,view:i.tangent,type:`f32`,access:`read_write`});let c=!!(i.tangent&&i.skylineAngle);if(i.skylineAngle&&!c&&s.push({name:`skylineAngleOutput`,view:i.skylineAngle,type:`f32`,access:`read_write`}),i.distance&&s.push({name:`distanceOutput`,view:i.distance,type:`f32`,access:`read_write`}),i.samples&&s.push({name:`samplesOutput`,view:i.samples,type:`u32`,access:`read_write`}),s.length>7)throw Error(`${t} needs more than 8 storage bindings`);let l=st(r,7),d=n.observers.length*r.azimuthSpan,f=n.raysPerDispatch??65536,p=e=>[i.tangent?`tangentOutput[tangentOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(-BIG, result.t, result.has)`};`:``,i.skylineAngle&&!c?`skylineAngleOutput[skylineAngleOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(-90.0, atanAccurate(result.t) * DEGREES, result.has)`};`:``,i.distance?`distanceOutput[distanceOutputOffset + ray] = ${e===`nan`?`nanValue()`:`select(0.0, result.d, result.has)`};`:``,i.samples?`samplesOutput[samplesOutputOffset + ray] = ${e===`nan`?`0u`:`result.evaluated`};`:``].join(`
  `),m=lt(e,{id:t,operation:`GPUPointHorizonProfile`,itemCount:d,maximumChunkSize:f});o.push(m.writeNode);for(let n=0;n<m.chunkCount;n++)o.push(x(e,{id:`${t}-march-${n}`,operation:`GPUPointHorizonProfile`,variant:r.traversal,bindings:[...s,m.getChunkStart(n)],invocationCount:m.chunkSize,workgroupSize:64,declarations:`const RAY_COUNT: u32 = ${d}u;
${l}`,body:`let ray = chunkStart[chunkStartOffset] + index;
  if (ray >= RAY_COUNT) { return; }
  let observerIndex = ray / AZIMUTH_SPAN;
  let azimuthOffset = ray - observerIndex * AZIMUTH_SPAN;
  let observerBase = observersOffset + observerIndex * 4u;
  let eye = makeEye(observers[observerBase], observers[observerBase + 1u], observers[observerBase + 2u]);
  if (!eye.valid) {
    ${p(`nan`)}
    return;
  }
  let direction = azimuthSinCos(FIRST_AZIMUTH + azimuthOffset, AZIMUTH_COUNT);
  let result = marchRay(eye, direction.x, direction.y, 0.0, false);
  ${p(`result`)}`}));return i.tangent&&i.skylineAngle&&c&&o.push(x(e,{id:`${t}-skyline-angle`,operation:`GPUPointHorizonProfile`,variant:`skyline-angle`,bindings:[{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`tangent`,view:i.tangent,type:`f32`,access:`read`},{name:`skylineAngleOutput`,view:i.skylineAngle,type:`f32`,access:`read_write`}],invocationCount:d,declarations:`const RAY_COUNT: u32 = ${d}u;
${F}`,body:`if (index >= RAY_COUNT) { return; }
  let t = tangent[tangentOffset + index];
  var angle = t;
  if ((bitcast<u32>(t) & 0x7fffffffu) <= 0x7f800000u) {
    angle = select(-90.0, atanAccurate(t) * DEGREES, t != -BIG);
  }
  skylineAngleOutput[skylineAngleOutputOffset + index] = angle;`})),o}},dt=class{id;props;model;constructor(e){this.id=e.id??`point-horizon-visibility`,this.props=e;let{id:t}=this,{output:n}=e;if(this.model=rt(t,e.grid,{...e,azimuthCount:720,firstAzimuth:0,azimuthSpan:720}),nt(t,e,e.elevation,this.model.heightReference),oe(t,e.parameters,12),it(t,`observers`,e.observers,`float32x4`),it(t,`targets`,e.targets,`float32x4`),e.observers.length<1)throw Error(`${t} requires at least one observer`);if(_(n.visibility,[`uint32`],`${t} output.visibility`),n.visibility.length!==e.targets.length)throw Error(`${t} output.visibility must contain one value per target`);if(n.details&&(it(t,`output.details`,n.details,`float32x4`),n.details.length!==e.targets.length))throw Error(`${t} output.details must contain one row per target`);let r=e.targetsPerDispatch??65536;if(!Number.isInteger(r)||r<1)throw Error(`${t} targetsPerDispatch must be a positive integer`);T(t,[n.visibility,n.details],[...w(e.elevation),e.parameters,e.observers,e.targets])}getCommandNodes(e){let{id:t,props:n,model:r}=this,{output:i}=n;S(t,e,n.elevation,[]),u(t,e,[n.pyramid?.combined,n.parameters,n.observers,n.targets,i.visibility,i.details]);let a=I(e,t,r,n.elevation),o=[...a.nodes],s=[...a.bindings,{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`observers`,view:n.observers,type:`f32`,access:`read`},{name:`targets`,view:n.targets,type:`f32`,access:`read`},{name:`visibilityOutput`,view:i.visibility,type:`u32`,access:`read_write`}];if(i.details&&s.push({name:`detailsOutput`,view:i.details,type:`f32`,access:`read_write`}),s.length>7)throw Error(`${t} needs more than 8 storage bindings`);let c=r.projection===`web-mercator`?`let worldPixelSize = settings[settingsOffset + 1u];
  let halfLambda = PI * deltaColumn / worldPixelSize;
  let halfM = -PI * deltaRow / worldPixelSize;
  let m2 = eye.m + 2.0 * halfM;
  let meanM = eye.m + halfM;
  let cosP2 = 1.0 / coshS(m2);
  let sinHalfLambda = sinAny(halfLambda);
  let cosHalfLambda = cosAny(halfLambda);
  let sinhHalfM = sinhS(halfM);
  let haversine = eye.cosP * cosP2 * (sinhHalfM * sinhHalfM + sinHalfLambda * sinHalfLambda);
  let distance = 2.0 * EARTH_RADIUS * asinS(sqrt(min(haversine, 1.0)));
  let sinDeltaPhi = 2.0 * coshS(meanM) * sinhHalfM * eye.cosP * cosP2;
  let north = sinDeltaPhi + eye.sinP * cosP2 * 2.0 * sinHalfLambda * sinHalfLambda;
  let east = 2.0 * sinHalfLambda * cosHalfLambda * cosP2;
  let rho = sqrt(north * north + east * east);
  let sinA = east / max(rho, 1.0e-30);
  let cosA = north / max(rho, 1.0e-30);
  // One Web Mercator pixel at the target's latitude, in meters (isotropic).
  let targetPixelLength = 2.0 * PI * EARTH_RADIUS * cosP2 / worldPixelSize;`:`let deltaX = deltaColumn * settings[settingsOffset + 1u];
  let deltaY = deltaRow * settings[settingsOffset + 2u];
  let distance = sqrt(deltaX * deltaX + deltaY * deltaY);
  let sinA = deltaX / max(distance, 1.0e-30);
  let cosA = ${r.rowDirection===`south`?`-`:``}deltaY / max(distance, 1.0e-30);
  // Ground length of one pixel step along the ray (the march's pixels-per-meter is (sinA / cellX, cosA / cellY)).
  let targetPixelLength = 1.0 / length(vec2<f32>(sinA / settings[settingsOffset + 1u], cosA / settings[settingsOffset + 2u]));`,l=st(r,11),{targets:d}=n,f=lt(e,{id:t,operation:`GPUPointHorizonVisibility`,itemCount:d.length,maximumChunkSize:n.targetsPerDispatch??65536});o.push(f.writeNode);let p=i.details?`let detailsBase = detailsOutputOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  detailsOutput[detailsBase] = result.details.x;
  detailsOutput[detailsBase + 1u] = result.details.y;
  detailsOutput[detailsBase + 2u] = result.details.z;
  detailsOutput[detailsBase + 3u] = result.details.w;`:``;for(let i=0;i<f.chunkCount;i++)o.push(x(e,{id:`${t}-classify-${i}`,operation:`GPUPointHorizonVisibility`,variant:r.traversal,bindings:[...s,f.getChunkStart(i)],invocationCount:f.chunkSize,workgroupSize:64,declarations:`const TARGET_COUNT: u32 = ${d.length}u;
const OBSERVER_COUNT: u32 = ${n.observers.length}u;
const HIDDEN: u32 = ${O.hidden}u;
const VISIBLE: u32 = ${O.visible}u;
const OUT_OF_RANGE: u32 = ${O.outOfRange}u;
const NO_DATA: u32 = ${O.noData}u;
const MARGINAL: u32 = ${O.marginal}u;
${l}
fn sinAny(x: f32) -> f32 { if (abs(x) > 1.6) { return sin(x); } return sinSeries(x); }
fn cosAny(x: f32) -> f32 { if (abs(x) > 1.6) { return cos(x); } return cosSeries(x); }

struct Classification { code: u32, details: vec4<f32> };

fn classifyTarget(index: u32) -> Classification {
  let notANumber = vec4<f32>(nanValue());
  let base = targetsOffset + (chunkStart[chunkStartOffset] + index) * 4u;
  let targetColumn = targets[base];
  let targetRow = targets[base + 1u];
  let targetHeight = targets[base + 2u];
  let observerValue = targets[base + 3u];
  if (!(observerValue >= 0.0 && observerValue < f32(OBSERVER_COUNT))) {
    return Classification(NO_DATA, notANumber);
  }
  let observerBase = observersOffset + u32(observerValue) * 4u;
  let observerColumn = observers[observerBase];
  let observerRow = observers[observerBase + 1u];
  let eye = makeEye(observerColumn, observerRow, observers[observerBase + 2u]);
  if (!eye.valid || !isInsideGrid(targetColumn, targetRow) || !isFiniteValue(targetHeight)) {
    return Classification(NO_DATA, notANumber);
  }
  let targetSample = sampleRelative(targetColumn, targetRow, eye.hi, eye.lo);
  if (targetSample.y == 0.0) {
    return Classification(NO_DATA, notANumber);
  }
  let relativeHeight = targetSample.x + targetHeight;
  let deltaColumn = targetColumn - observerColumn;
  let deltaRow = targetRow - observerRow;
  ${c}
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(MAXIMUM_DISTANCE, min(maximumFrame, MAXIMUM_DISTANCE), maximumFrame > 0.0);
  if (distance > maximumDistance) {
    return Classification(OUT_OF_RANGE, notANumber);
  }
  if (!(distance > 0.0)) {
    return Classification(VISIBLE, notANumber);
  }
  let curvature = settings[settingsOffset];
  // A negative packed ignore distance is the derived default: one ground pixel at the target.
  let packedIgnoreDistance = settings[settingsOffset + 8u];
  let ignoreDistance = select(packedIgnoreDistance, targetPixelLength, packedIgnoreDistance < 0.0);
  let stop = distance - max(ignoreDistance, settings[settingsOffset + 9u] * distance);
  let ray = marchRay(eye, sinA, cosA, stop, stop > 0.0);
  let targetTangent = relativeHeight / distance - opaque(distance * curvature);
  let targetAngle = atanAccurate(targetTangent) * DEGREES;
  let occluderAngle = select(-90.0, atanAccurate(ray.tQ) * DEGREES, ray.tQ > -3.0e38);
  let skylineAngle = select(-90.0, atanAccurate(ray.t) * DEGREES, ray.has);
  let tolerance = settings[settingsOffset + 5u] + (settings[settingsOffset + 6u] / distance) * DEGREES;
  var code = VISIBLE;
  if (targetAngle < occluderAngle - tolerance) {
    code = HIDDEN;
  } else if (abs(targetAngle - occluderAngle) < tolerance) {
    code = MARGINAL;
  }
  let onSkyline = code != HIDDEN && targetAngle >= skylineAngle - settings[settingsOffset + 7u];
  return Classification(code, vec4<f32>(targetAngle, occluderAngle, skylineAngle, select(0.0, 1.0, onSkyline)));
}`,body:`if (chunkStart[chunkStartOffset] + index >= TARGET_COUNT) { return; }
  let result = classifyTarget(index);
  visibilityOutput[visibilityOutputOffset + chunkStart[chunkStartOffset] + index] = result.code;
  ${p}`}));return o}},L=`GPUProfilePeaks`,ft=4096,pt=1024,mt=4294967295;function ht(e,t=new Float32Array(1)){if(C(`getGPUProfilePeaksParameterValues`,t,1),Number.isNaN(e.minimumProminence))throw Error(`getGPUProfilePeaksParameterValues minimumProminence must not be NaN`);return t[0]=e.minimumProminence,t}var gt=0,R=1,z=2,B=3,_t=class{id;props;minimumSide;suppressionRadius;wrap;maximumSuppressionIterations;constructor(e){this.id=e.id??`profile-peaks`,this.props=e,this.minimumSide=e.minimumSide??4,this.suppressionRadius=e.suppressionRadius??Math.max(2,Math.round(e.window/4)),this.wrap=e.wrap??!1,this.maximumSuppressionIterations=e.maximumSuppressionIterations??16;let{id:t}=this,{output:n}=e,r=n.status?.converged;if(_(e.values,[`float32`],`${t} values`),e.values.length<1)throw Error(`${t} values must contain at least one sample`);let i=e.values.length;if(e.validity&&(_(e.validity,[`uint32`],`${t} validity`),e.validity.length!==i))throw Error(`${t} validity must contain one word per sample`);if(_(e.offsets,[`uint32`],`${t} offsets`),e.offsets.length<1)throw Error(`${t} offsets must contain profileCount + 1 rows`);if(_(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<1)throw Error(`${t} parameters must contain at least 1 value`);if(!Number.isInteger(e.window)||e.window<1||e.window>4096)throw Error(`${t} window must be an integer in [1, ${ft}]`);if(d(t,`minimumSide`,this.minimumSide,0,2**53-1),!Number.isFinite(this.suppressionRadius)||this.suppressionRadius<0||this.suppressionRadius>4096)throw Error(`${t} suppressionRadius must be in [0, ${ft}]`);if(d(t,`maximumSuppressionIterations`,this.maximumSuppressionIterations,0,pt),!n.prominence&&!n.refinedIndex&&!n.refinedValue&&!n.peakMask&&!n.peaks&&!r)throw Error(`${t} requires at least one output`);for(let[e,r,a]of[[`prominence`,n.prominence,`float32`],[`refinedIndex`,n.refinedIndex,`float32`],[`refinedValue`,n.refinedValue,`float32`],[`peakMask`,n.peakMask,`uint32`]])if(r&&(_(r,[a],`${t} output.${e}`),r.length!==i))throw Error(`${t} output.${e} must contain one value per sample`);if(n.peaks&&f(t,n.peaks),r&&(_(r,[`uint32`],`${t} output.status.converged`),r.length<1))throw Error(`${t} output.status.converged must contain one row`);p(t,[n.prominence,n.refinedIndex,n.refinedValue,n.peakMask,r,n.peaks?.ids,n.peaks?.status.count,n.peaks?.status.overflow,n.peaks?.status.requiredCount],[e.values,e.validity,e.offsets,e.parameters])}getCommandNodes(e){let{id:t,props:n,minimumSide:r,suppressionRadius:i,wrap:a,maximumSuppressionIterations:o}=this,{output:s}=n,c=s.status?.converged;u(t,e,[n.values,n.validity,n.offsets,n.parameters,s.prominence,s.refinedIndex,s.refinedValue,s.peakMask,c,s.peaks?.ids,s.peaks?.status.count,s.peaks?.status.overflow,s.peaks?.status.requiredCount]);let l=n.values.length,d=n.offsets.length-1,f=g(e,`${t}-candidate-prominence`,`float32`,l),p=g(e,`${t}-candidate-delta`,`float32`,l),m=g(e,`${t}-candidate-value`,`float32`,l),h=g(e,`${t}-candidate-state`,`uint32`,l),_=U(l,d,a),y=[],ne=[{name:`sampleValues`,view:n.values,type:`f32`,access:`read`}];n.validity&&ne.push({name:`sampleValidity`,view:n.validity,type:`u32`,access:`read`}),ne.push({name:`profileOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`candidateProminence`,view:f,type:`f32`,access:`read_write`},{name:`candidateDelta`,view:p,type:`f32`,access:`read_write`},{name:`candidateValue`,view:m,type:`f32`,access:`read_write`},{name:`candidateState`,view:h,type:`u32`,access:`read_write`}),y.push(x(e,{id:`${t}-candidates`,operation:L,variant:`candidates`,bindings:ne,invocationCount:l,declarations:`${_}
${ae}
const WINDOW: u32 = ${n.window}u;
const MIN_SIDE: u32 = ${r}u;
const STATE_INERT: u32 = ${gt}u;
const STATE_UNDECIDED: u32 = ${R}u;
${H()}
fn isGap(sample: u32) -> bool {
  ${n.validity?`if (sampleValidity[sampleValidityOffset + sample] == 0u) { return true; }`:``}
  return !isFiniteValue(sampleValues[sampleValuesOffset + sample]);
}
fn writeInert(sample: u32) {
  candidateProminence[candidateProminenceOffset + sample] = getNan();
  candidateDelta[candidateDeltaOffset + sample] = 0.0;
  candidateValue[candidateValueOffset + sample] = getNan();
  candidateState[candidateStateOffset + sample] = STATE_INERT;
}
struct Side {
  lowest: f32,
  valid: u32,
}
// Walks away from a candidate; the first higher sample is counted as valid but not as lowest.
fn walkSide(start: u32, count: u32, local: u32, peak: f32, direction: i32) -> Side {
  var side = Side(peak, 0u);
  var limit = WINDOW;
  if (WRAP) { limit = min(WINDOW, count - 1u); }
  for (var step = 1u; step <= limit; step++) {
    let neighbor = neighborLocal(local, direction * i32(step), count);
    if (neighbor < 0) { break; }
    let sample = start + u32(neighbor);
    if (isGap(sample)) { break; }
    side.valid = side.valid + 1u;
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak) { break; }
    if (other < side.lowest) { side.lowest = other; }
  }
  return side;
}`,body:`let range = findProfile(index);
  let count = range.y - range.x;
  if (count == 0u) { writeInert(index); return; }
  let start = range.x;
  let local = index - start;
  if (!WRAP && (local < 1u || local + 1u >= count)) { writeInert(index); return; }
  if (isGap(index)) { writeInert(index); return; }
  let peak = sampleValues[sampleValuesOffset + index];
  for (var offset = -2; offset <= 2; offset++) {
    if (offset == 0) { continue; }
    let neighbor = neighborLocal(local, offset, count);
    if (neighbor < 0) { continue; }
    let sample = start + u32(neighbor);
    if (sample == index || isGap(sample)) { continue; }
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak || (offset < 0 && other == peak)) { writeInert(index); return; }
  }
  let left = walkSide(start, count, local, peak, -1);
  let right = walkSide(start, count, local, peak, 1);
  if (left.valid < MIN_SIDE || right.valid < MIN_SIDE) { writeInert(index); return; }
  let prominence = peak - max(left.lowest, right.lowest);
  if (prominence < settings[settingsOffset]) { writeInert(index); return; }
  var delta = 0.0;
  var refined = peak;
  let previous = start + u32(neighborLocal(local, -1, count));
  let next = start + u32(neighborLocal(local, 1, count));
  if (!isGap(previous) && !isGap(next)) {
    let a = sampleValues[sampleValuesOffset + previous];
    let c = sampleValues[sampleValuesOffset + next];
    let denominator = a - 2.0 * peak + c;
    if (denominator < 0.0) {
      delta = clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
      refined = peak - 0.25 * (a - c) * delta;
    }
  }
  candidateProminence[candidateProminenceOffset + index] = prominence;
  candidateDelta[candidateDeltaOffset + index] = delta;
  candidateValue[candidateValueOffset + index] = refined;
  candidateState[candidateStateOffset + index] = STATE_UNDECIDED;`}));let S=Math.floor(i)+1;if(o>0){let r=ue(e,`${t}-nms`,L,l);y.push(le(e,{id:`${t}-nms-reset`,operation:L,state:r}),...ce(e,{id:`${t}-nms`,gateId:`${t}-nms-gate`,operation:L,state:r,maxIterations:o,createRound:({nodeId:t,slotBinding:a,markChangedWGSL:o,condition:s,extraResources:c})=>x(e,{id:t,operation:L,variant:`suppression`,bindings:[{name:`profileOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`candidateProminence`,view:f,type:`f32`,access:`read`},{name:`candidateDelta`,view:p,type:`f32`,access:`read`},{name:`candidateState`,view:h,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:r.status,type:`atomic<u32>`,access:`read_write`},a],invocationCount:l,condition:s,extraResources:c,declarations:`${_}
const NMS_RADIUS: f32 = ${b(i)};
const SCAN_RADIUS: u32 = ${S}u;
const STATE_UNDECIDED: u32 = ${R}u;
const STATE_KEPT: u32 = ${z}u;
const STATE_SUPPRESSED: u32 = ${B}u;`,body:`if (atomicLoad(&candidateState[candidateStateOffset + index]) != STATE_UNDECIDED) { return; }
  let range = findProfile(index);
  let start = range.x;
  let count = range.y - range.x;
  let local = index - start;
  let prominenceSelf = candidateProminence[candidateProminenceOffset + index];
  let deltaSelf = candidateDelta[candidateDeltaOffset + index];
  var undecided = false;
  for (var step = 1u; step <= SCAN_RADIUS; step++) {
    for (var side = 0; side < 2; side++) {
      let direction = select(-1, 1, side == 1);
      let neighbor = neighborLocal(local, direction * i32(step), count);
      if (neighbor < 0 || u32(neighbor) == local) { continue; }
      let sample = start + u32(neighbor);
      let state = atomicLoad(&candidateState[candidateStateOffset + sample]);
      if (state != STATE_UNDECIDED && state != STATE_KEPT) { continue; }
      let prominenceOther = candidateProminence[candidateProminenceOffset + sample];
      let higher = prominenceOther > prominenceSelf ||
        (prominenceOther == prominenceSelf && u32(neighbor) < local);
      if (!higher) { continue; }
      var distance = abs(f32(neighbor - i32(local)) +
        (candidateDelta[candidateDeltaOffset + sample] - deltaSelf));
      if (WRAP) { distance = min(distance, f32(count) - distance); }
      if (distance > NMS_RADIUS) { continue; }
      if (state == STATE_KEPT) {
        atomicStore(&candidateState[candidateStateOffset + index], STATE_SUPPRESSED);
        ${o}
        return;
      }
      undecided = true;
    }
  }
  if (!undecided) {
    atomicStore(&candidateState[candidateStateOffset + index], STATE_KEPT);
    ${o}
  }`})}))}c&&y.push(ee(e,{id:`${t}-converged-reset`,operation:L,view:c,type:`u32`,value:`1u`,componentCount:1}),x(e,{id:`${t}-converged`,operation:L,variant:`converged`,bindings:[{name:`candidateState`,view:h,type:`u32`,access:`read`},{name:`convergedOut`,view:c,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,body:`if (candidateState[candidateStateOffset + index] == ${R}u) {
    atomicStore(&convergedOut[convergedOutOffset], 0u);
  }`}));let C=s.peakMask??(s.peaks?g(e,`${t}-peak-mask`,`uint32`,l):void 0),w=s.peaks&&s.peakMask?g(e,`${t}-peak-flags`,`uint32`,l):C,T=s.peaks?g(e,`${t}-row-ids`,`uint32`,l):void 0;if(s.prominence||s.refinedValue||C){let n=[{name:`candidateState`,view:h,type:`u32`,access:`read`}];s.prominence&&n.push({name:`candidateProminence`,view:f,type:`f32`,access:`read`},{name:`prominenceOut`,view:s.prominence,type:`f32`,access:`read_write`}),s.refinedValue&&n.push({name:`candidateValue`,view:m,type:`f32`,access:`read`},{name:`refinedValueOut`,view:s.refinedValue,type:`f32`,access:`read_write`}),C&&n.push({name:`peakMaskOut`,view:C,type:`u32`,access:`read_write`}),w&&w!==C&&n.push({name:`flagsOut`,view:w,type:`u32`,access:`read_write`}),T&&n.push({name:`rowIdsOut`,view:T,type:`u32`,access:`read_write`});let r=!!s.peakMask;y.push(x(e,{id:`${t}-outputs`,operation:L,variant:`outputs`,bindings:n,invocationCount:l,declarations:`${H()}
${V()}`,body:`let state = candidateState[candidateStateOffset + index];
  let kept = state == ${z}u;
  let unresolved = state == ${R}u;
  ${s.prominence?`prominenceOut[prominenceOutOffset + index] = select(select(getNan(), candidateProminence[candidateProminenceOffset + index], kept), getInfinity(), unresolved);`:``}
  ${s.refinedValue?`refinedValueOut[refinedValueOutOffset + index] = select(select(getNan(), candidateValue[candidateValueOffset + index], kept), getInfinity(), unresolved);`:``}
  ${C?r?`peakMaskOut[peakMaskOutOffset + index] = select(select(0u, 1u, kept), ${mt}u, unresolved);`:`peakMaskOut[peakMaskOutOffset + index] = select(0u, 1u, kept);`:``}
  ${w&&w!==C?`flagsOut[flagsOutOffset + index] = select(0u, 1u, kept);`:``}
  ${T?`rowIdsOut[rowIdsOutOffset + index] = index;`:``}`}))}if(s.refinedIndex&&y.push(x(e,{id:`${t}-refined-index`,operation:L,variant:`refined-index`,bindings:[{name:`profileOffsets`,view:n.offsets,type:`u32`,access:`read`},{name:`candidateDelta`,view:p,type:`f32`,access:`read`},{name:`candidateState`,view:h,type:`u32`,access:`read`},{name:`refinedIndexOut`,view:s.refinedIndex,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`${_}
${H()}
${V()}`,body:`let state = candidateState[candidateStateOffset + index];
  if (state != ${z}u) {
    refinedIndexOut[refinedIndexOutOffset + index] =
      select(getNan(), getInfinity(), state == ${R}u);
    return;
  }
  let range = findProfile(index);
  let count = f32(range.y - range.x);
  var refined = f32(index - range.x) + candidateDelta[candidateDeltaOffset + index];
  if (WRAP) {
    if (refined < 0.0) { refined = refined + count; }
    if (refined >= count) { refined = refined - count; }
  }
  refinedIndexOut[refinedIndexOutOffset + index] = refined;`})),s.peaks&&w&&T){let n=g(e,`${t}-compact-rows`,`uint32`,l),r=g(e,`${t}-peak-total`,`uint32`,1),i=g(e,`${t}-incomplete`,`uint32`,1);y.push(...new v({id:`${t}-compaction`,input:T,flags:w,output:n,count:r}).getCommandNodes(e),te(e,{id:`${t}-publish`,operation:L,requiredCount:r,compactIds:n,output:s.peaks}),ee(e,{id:`${t}-incomplete-clear`,operation:L,view:i,type:`u32`,value:`0u`,componentCount:1}),x(e,{id:`${t}-incomplete`,operation:L,variant:`incomplete`,bindings:[{name:`candidateState`,view:h,type:`u32`,access:`read`},{name:`incomplete`,view:i,type:`atomic<u32>`,access:`read_write`}],invocationCount:l,body:`if (candidateState[candidateStateOffset + index] == ${R}u) {
    atomicStore(&incomplete[incompleteOffset], 1u);
  }`}),x(e,{id:`${t}-publish-unconverged`,operation:L,variant:`publish-unconverged`,bindings:[{name:`incomplete`,view:i,type:`u32`,access:`read`},{name:`overflowOut`,view:s.peaks.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,body:`overflowOut[overflowOutOffset] = overflowOut[overflowOutOffset] |
    select(0u, 2u, incomplete[incompleteOffset] != 0u);`}))}return y}};function V(){return`fn getInfinity() -> f32 {
  var bits = 0x7f800000u;
  return bitcast<f32>(bits);
}`}function H(){return`fn getNan() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}`}function U(e,t,n){return`const SAMPLE_COUNT: u32 = ${e}u;
const PROFILE_COUNT: u32 = ${t}u;
const WRAP: bool = ${n};
// Returns [start, end) of the profile owning \`sample\`, or [0, 0) when it belongs to none.
fn findProfile(sample: u32) -> vec2<u32> {
  var low = 0u;
  var high = PROFILE_COUNT;
  for (var step = 0u; step < 32u && low < high; step++) {
    let middle = low + (high - low) / 2u;
    if (profileOffsets[profileOffsetsOffset + middle + 1u] > sample) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  if (low >= PROFILE_COUNT) { return vec2<u32>(0u, 0u); }
  let start = profileOffsets[profileOffsetsOffset + low];
  let end = min(profileOffsets[profileOffsetsOffset + low + 1u], SAMPLE_COUNT);
  if (start > sample || sample >= end) { return vec2<u32>(0u, 0u); }
  return vec2<u32>(start, end);
}
// Local index of the neighbour \`delta\` samples away, or -1 when it is outside a non-wrapped profile.
fn neighborLocal(local: u32, delta: i32, count: u32) -> i32 {
  let position = i32(local) + delta;
  let signedCount = i32(count);
  if (WRAP) { return ((position % signedCount) + signedCount) % signedCount; }
  if (position < 0 || position >= signedCount) { return -1; }
  return position;
}`}function W(e){let t=(e%360+360)%360;return t<180?t+360:t}var G=e=>Number.isFinite(e)&&e>-89.5;function K(e,t){let n=[],r=[];return e.forEach((e,i)=>{n.length&&(n.push(NaN),r.push(NaN)),n.push(e,e),r.push(t[i],t[i])}),{x:n,y:r}}function vt(e){let{azimuthCount:t}=e,n=Math.round(180/360*t),r=[],i=[],a=1/0,o=-1/0;for(let s=0;s<=t;s++){let c=e.skylineAngles[(n+s)%t];r.push(180+s*360/t),G(c)?(i.push(c),a=Math.min(a,c),o=Math.max(o,c)):i.push(NaN)}let s=Number.isFinite(a),c=[{label:`Skyline`,x:r,y:i,area:!0,color:2,width:1.75}];if(e.view===`visibility`){let t=e.peaks.filter(e=>e.code===`visible`),n=e.peaks.filter(e=>e.code===`marginal`),r=e.peaks.filter(e=>e.code===`hidden`);if(r.length){let e=K(r.map(e=>W(e.azimuth)),r.map(e=>e.angle));c.push({label:`Hidden peaks`,x:e.x,y:e.y,ghost:!0,width:5})}n.length&&c.push({label:`Marginal peaks`,x:n.map(e=>W(e.azimuth)),y:n.map(e=>e.angle),color:4,points:!0,width:0}),t.length&&c.push({label:`Visible peaks`,x:t.map(e=>W(e.azimuth)),y:t.map(e=>e.angle),color:0,points:!0,width:0})}return e.view===`skyline-peaks`&&e.skylinePeaks.length&&c.push({label:`Skyline peaks`,x:e.skylinePeaks.map(e=>W(e.azimuth)),y:e.skylinePeaks.map(e=>e.angle),color:1,points:!0,width:0}),{kind:`line`,height:176,series:c,xDomain:[180,540],yDomain:[s?Math.floor(Math.min(a,0)):-6,s?Math.ceil(o)+1:12],xLabel:`Bearing from the eye, south to south by way of west and north`,yLabel:`Elevation angle (°)`,formatX:e=>ze(e),formatY:e=>`${e}°`,markers:e.markers.map(e=>({x:W(e.azimuth),label:e.label})),...e.linkLabel?{link:{option:`bearing`,label:e.linkLabel}}:{},table:!1,description:`Panorama from the eye: the highest elevation angle at every bearing as a filled skyline, with the named peaks marked. In the visibility view visible peaks touch the skyline and hidden peaks sit below it.`}}function yt(e){let{ray:t}=e,n=Array.from(t.distanceMeters,e=>e/1e3),r=[];return t.ridge&&r.push({x:t.ridge.distanceMeters/1e3,label:e.targetName?`Ridge`:`Skyline`}),e.targetName&&r.push({x:t.lengthMeters/1e3,label:e.targetName}),{kind:`line`,height:150,series:[{label:`Ground`,x:n,y:Array.from(t.terrainAngle),ghost:!0,width:1},{label:`Highest so far`,x:n,y:Array.from(t.runningMaxAngle),color:2,width:2}],xLabel:`Distance from the eye (km)`,yLabel:`Angle (°)`,formatY:e=>`${e}°`,...e.targetAngle!==null&&Number.isFinite(e.targetAngle)?{guides:[{y:e.targetAngle,label:`Peak angle`}]}:{},markers:r,table:!1,description:e.targetName?`Elevation angle against distance along the ray to ${e.targetName}: the ground, the highest angle so far, the ridge that sets it and the angle of the peak.`:`Elevation angle against distance along the ray: the ground, the highest angle so far and the skyline where it is set.`}}function bt(e,t){return{kind:`bars`,horizontal:!0,values:e.map(e=>e.prominence),labels:e.map(e=>e.label),highlight:[0],formatX:e=>`${e.toFixed(1)}°`,...t?{onBarClick:t}:{},table:!1,description:`Skyline peaks ranked by prominence in degrees: how far each stands above the lower of its two surrounding valleys on the skyline.`}}function xt(e){let t=e.split(/ \/ | - /).map(e=>e.trim());if(t.length===1)return e;let n=t.find(e=>/(horn|spitze|kuppe|gipfel|stock|hubel)$/i.test(e));return n?/^(west|ost|nord|süd|sued)gipfel$/i.test(n)?`${t[0].split(` `)[0]} ${n}`:n:t[0]}function q(e){return e.normalize(`NFD`).replace(/[̀-ͯ]/g,``).toLowerCase().replace(/[^a-z0-9]+/g,``)}var J=[`Matterhorn`,`Dufourspitze`,`Weisshorn`,`Dom`,`Dent Blanche`,`Liskamm`,`Breithorn`,`Castor`,`Ober Gabelhorn`,`Dent d'Hérens`,`Klein Matterhorn`,`Zinalrothorn`];function St(e){let t=[];for(let n of J){let r=q(n),i=e.findIndex((e,i)=>!t.includes(i)&&(q(e.displayName)===r||e.displayName.toLowerCase().startsWith(`${n.toLowerCase()} `)));i>=0&&t.push(i)}let n=e.map((e,t)=>t).filter(e=>!t.includes(e)).sort((t,n)=>e[n].publishedElevationMeters-e[t].publishedElevationMeters);return[...t,...n]}function Ct(e,t,n,r){let i=e.peaks.filter(e=>e.name&&e.elevationMeters!==null&&e.elevationMeters>=3500).sort((e,t)=>(t.elevationMeters??0)-(e.elevationMeters??0)),a=[];for(let e of i){let i=Ue(t,e.lngLat);if(!i)continue;let{column:o,row:s}=i;if(o<3||s<3||o>=t.width-3||s>=t.height-3)continue;let c=n(o,s);if(!(Math.hypot(c[0]-r[0],c[1]-r[1])<600)&&!a.some(e=>Math.hypot(e.meters[0]-c[0],e.meters[1]-c[1])<1500)&&(a.push({id:e.osmId,name:e.name,displayName:xt(e.name),publishedElevationMeters:e.elevationMeters,demElevationMeters:i.elevation,column:o,row:s,lngLat:i.lngLat,meters:c}),a.length>=64))break}return a}var wt=180/Math.PI;function Tt(e){return e===`none`?1:e===`gdal`?1/7:.13}function Et(e,t){let n=e.sampleRay(t.from,t.to,t.stepMeters,{refraction:t.refraction,eyeHeight:t.eyeHeightMeters,targetHeight:0}),r=Math.max(0,n.count-1),i=new Float64Array(r),a=new Float64Array(r),o=new Float64Array(r),s=new Float64Array(r),c=t.cutoffMeters??1/0,l=-1,u=-90;for(let e=0;e<r;e++){let t=e+1,r=n.distance[t];i[e]=r,s[e]=n.terrain[t],a[e]=Math.atan2(n.loweredTerrain[t]-n.eyeElevation,r)*wt,o[e]=n.runningMaxAngle[t],r<=c&&a[e]>u&&(u=a[e],l=e)}let d=l>=0?{distanceMeters:i[l],angleDegrees:u,heightMeters:s[l],lngLat:[n.lngLat[(l+1)*2],n.lngLat[(l+1)*2+1]]}:null;return{distanceMeters:i,terrainAngle:a,runningMaxAngle:o,terrainHeight:s,eyeElevationMeters:n.eyeElevation,lengthMeters:n.totalDistance,ridge:d,endAngleDegrees:r>0?a[r-1]:NaN}}function Dt(e,t,n,r,i){let a=n*Math.PI/180,o=r;for(let n=0;n<200;n++){let n=t[0]+o*Math.sin(a),r=t[1]+o*Math.cos(a),s=i(n,r),[c,l]=e.getPixelCoordinates(s[0],s[1]);if(c>=1&&l>=1&&c<=e.width-1&&l<=e.height-1||o<100)return{lngLat:s,meters:[n,r],lengthMeters:o};o*=.97}return{lngLat:i(t[0],t[1]),meters:[t[0],t[1]],lengthMeters:0}}var Ot=3,kt=2e4,At=36,jt=64,Mt=64,Nt=22,Pt=14,Ft=20,It=.75,Lt=60,Rt=5,zt=7,Bt=8,Vt=6,Ht=300,Y=O,Ut={[Y.hidden]:`Hidden`,[Y.visible]:`Visible`,[Y.marginal]:`Marginal`,[Y.outOfRange]:`Out of range`,[Y.noData]:`No data`},Wt=(e,t)=>Math.abs(e-t)>.15*Math.max(e,t)+120,Gt=e=>Number.isFinite(e)&&e>-89.5,Kt=e=>(e%360+360)%360,qt=(e,t)=>{let n=Math.abs(Kt(e)-Kt(t));return Math.min(n,360-n)},Jt=e=>{let t=Kt(e);return t<180?t+360:t};async function X(e){let{device:t}=e;e.setStatus(`Loading the wide terrain and the OpenStreetMap peaks`),e.setReadout(`viewSummary`,`Loading terrain and the summit catalogue…`),e.setReadout(`peaksInRange`,`Loading summit catalogue…`),e.setReadout(`highestPeak`,`Loading summit catalogue…`);let n=await He(e.datasets.get(`alps-context`),e.signal);e.signal.throwIfAborted();let r=e.datasets.get(`alps-dem-wide`),i=Ce(r,1),a=Ce(r,Ot),c=ke(i),u=Ae(c),{width:d,height:f,pixelCount:p}=i,g=[i.origin[0],i.origin[1],0],_=qe({dem:c,device:t,ground:e.ground(),glacierMask:Ke(n.glacierGeoJson,c)});e.setStatus(`Building the shaded relief`),await _.prepare(),e.signal.throwIfAborted();let v=new h(t,`horizon`),b=!1,ee=Ve(e.ground());e.setLegendData(`tables`,ee);let te=v.createBuffer(`elevation`,i.elevation),x=v.createBuffer(`validity`,i.validity),ne=v.createBuffer(`coarse-elevation`,a.elevation),S=v.createBuffer(`coarse-validity`,a.validity),C=v.createBuffer(`sky-view`,a.pixelCount*4),w=v.createParameterBuffer(`horizon-settings`,`float32`,8),T=v.createParameterBuffer(`visibility-settings`,`float32`,12),re=v.createParameterBuffer(`peak-settings`,`float32`,1),ie=v.createParameterBuffer(`grid-settings`,`float32`,8),ae=v.createBuffer(`observer`,new Float32Array(4)),oe=v.createBuffer(`observer-marker`,new Float32Array(2)),se=v.createBuffer(`ray-before`,new Float32Array(4)),ce=v.createBuffer(`ray-after`,new Float32Array(4)),le=new l(t,{id:`horizon-setup`}),ue=Oe(le,`setup`,te,x,p),O=o(i.grid,{firstBlockSize:4,footprint:`bilinear`}),Ue=v.createBuffer(`pyramid`,O.length*8);le.add(new s({id:`extrema-pyramid`,grid:i.grid,input:ue,firstBlockSize:4,footprint:`bilinear`,output:{combined:D(le,`pyramid`,Ue,`float32`,2*O.length)}}));let Je=v.track(await le.compileAsync()),Ye=i.observers.find(e=>e.name===`Gornergrat`),k=Ye?i.getPixel(Ye.longitude,Ye.latitude):[d/2,f/2],Qe=i.getMeters(k[0],k[1]),A=Ct(n,c,i.getMeters,Qe),j=A.length,$e=St(A),et=new Float32Array(j*4);A.forEach((e,t)=>{et.set([e.column,e.row,0,0],t*4)});let tt=v.createBuffer(`targets`,et),nt=v.createBuffer(`target-markers`,Float32Array.from(A.flatMap(e=>e.meters))),rt=v.createBuffer(`target-visibility`,j*4),it=v.createBuffer(`target-details`,j*16),at=v.createBuffer(`target-on-skyline`,j*4),ot=!0,M=!0,N=null,P=null,F=null,st=!0,ct=!1,lt=!1,I=!0,L=new Float32Array,ft=16,pt=v.track(new y(t,{id:`horizon-grid-read`,byteLength:192})),mt=!1,gt=!1,R=new Float32Array,z=new Float32Array,B=new Float32Array,V=!1,H=new Uint32Array(j),U=new Float32Array(j*4),W=new Float64Array(j),G=new Float64Array(j),K=[],xt=0,q=null,J=-1,wt=-1,X=null,Yt=null,Xt=``,Zt=!1,Qt=()=>be(Tt(e.options.refraction)),Z=()=>i.getMeters(k[0],k[1]),$t=()=>i.getLongitudeLatitude(k[0],k[1]),en=()=>e.options.maxDistance*1e3,tn=(e,t)=>i.projection.unproject(e,t);function nn(){let[e,t]=Z();A.forEach((n,r)=>{let i=n.meters[0]-e,a=n.meters[1]-t;W[r]=Math.hypot(i,a),G[r]=Ee(i,a)})}let Q=e=>W[e]<=en();function rn(e){return{layout:O,combined:D(e,`pyramid`,Ue,`float32`,2*O.length)}}function an(){let t=e.options;return[t.projection,t.traversal,t.azimuthCount,t.peakWindow].join(`|`)}async function on(){let n=e.options,r=an(),a=Number(n.azimuthCount),o=Number(n.peakWindow),s=v.createBuffer(`skyline-angle`,a*4),c=v.createBuffer(`skyline-distance`,a*4),u=v.createBuffer(`skyline-samples`,a*4),d=v.createBuffer(`peak-prominence`,a*4),f=v.createBuffer(`peak-index`,a*4),m=v.createBuffer(`peak-value`,a*4),h=v.createBuffer(`peak-mask`,a*4),g=v.createBuffer(`peak-converged`,4),_=v.createBuffer(`skyline-ring`,a*16),y=v.createBuffer(`skyline-weights`,a*4),b=v.createBuffer(`skyline-fan`,jt*16),ee=v.createBuffer(`peak-markers`,Mt*8),ne=v.createBuffer(`skyline-offsets`,Uint32Array.of(0,a)),S=new l(t,{id:`horizon-skyline`}),C=D(S,`angle`,s,`float32`,a);return S.add(new ut({id:`skyline`,grid:yn(n.projection),elevation:Oe(S,`skyline`,te,x,p),projection:n.projection,traversal:n.traversal,pyramid:n.traversal===`pyramid`?rn(S):void 0,azimuthCount:a,maximumDistance:kt,cellSize:i.groundCellSize,observers:D(S,`observer`,ae,`float32x4`,1),parameters:w.importToGraph(S),output:{skylineAngle:C,distance:D(S,`distance`,c,`float32`,a),samples:D(S,`samples`,u,`uint32`,a)}})),S.add(new _t({id:`skyline-peaks`,values:C,offsets:D(S,`offsets`,ne,`uint32`,2),parameters:re.importToGraph(S),wrap:!0,window:o,output:{prominence:D(S,`prominence`,d,`float32`,a),refinedIndex:D(S,`refined-index`,f,`float32`,a),refinedValue:D(S,`refined-value`,m,`float32`,a),peakMask:D(S,`peak-mask`,h,`uint32`,a),status:{converged:D(S,`converged`,g,`uint32`,1)}}})),{key:r,azimuthCount:a,compiled:v.track(await S.compileAsync()),angle:s,distance:c,samples:u,prominence:d,refinedIndex:f,refinedValue:m,peakMask:h,converged:g,ring:_,weights:y,fan:b,peakMarkers:ee,reader:new ge(v,`skyline`,[{buffer:s,size:a*4},{buffer:c,size:a*4},{buffer:u,size:a*4},{buffer:d,size:a*4},{buffer:f,size:a*4},{buffer:g,size:4}],e=>Tn(e))}}function sn(){let t=e.options;return[t.projection,t.traversal].join(`|`)}async function cn(){let n=e.options,r=sn(),a=new l(t,{id:`horizon-visibility`});return a.add(new dt({id:`visibility`,grid:yn(n.projection),elevation:Oe(a,`visibility`,te,x,p),projection:n.projection,traversal:n.traversal,pyramid:n.traversal===`pyramid`?rn(a):void 0,maximumDistance:kt,cellSize:i.groundCellSize,observers:D(a,`observer`,ae,`float32x4`,1),targets:D(a,`targets`,tt,`float32x4`,j),parameters:T.importToGraph(a),output:{visibility:D(a,`visibility`,rt,`uint32`,j),details:D(a,`details`,it,`float32x4`,j)}})),{key:r,compiled:v.track(await a.compileAsync()),reader:new ge(v,`visibility`,[{buffer:rt,size:j*4},{buffer:it,size:j*16}],e=>En(e))}}function ln(){let t=e.options;return[t.gridDirections,t.gridRadius,t.gridAlgorithm].join(`|`)}async function un(){let n=e.options,r=ln(),i=Number(n.gridDirections),o=new l(t,{id:`horizon-grid`}),s=v.createBuffer(`grid-horizon-${n.gridDirections}`,a.pixelCount*i*4);return o.add(new Se({id:`terrain-horizon`,grid:a.grid,elevation:Oe(o,`grid`,ne,S,a.pixelCount),parameters:ie.importToGraph(o),directionCount:i,maximumRadius:Number(n.gridRadius),algorithm:n.gridAlgorithm,cellSizeMode:`web-mercator`,rowDirection:`south`,output:{horizon:D(o,`horizon`,s,`float32`,a.pixelCount*i),skyViewFactor:D(o,`sky-view`,C,`float32`,a.pixelCount)}})),{key:r,directions:i,compiled:v.track(await o.compileAsync()),horizon:s}}let dn=v.track(new he);function fn(e){e.reader.stop(),v.release(e.compiled);for(let t of[e.angle,e.distance,e.samples,e.prominence,e.refinedIndex,e.refinedValue,e.peakMask,e.converged,e.ring,e.weights,e.fan,e.peakMarkers])v.release(t)}function pn(e){e.reader.stop(),v.release(e.compiled)}function mn(e){v.release(e.compiled),v.release(e.horizon)}function hn(e){e.skyline&&fn(e.skyline),e.visibility&&pn(e.visibility),e.grid&&mn(e.grid)}async function gn(){let t={skyline:null,visibility:null,grid:null};try{(!N||N.key!==an())&&(t.skyline=await on()),(!P||P.key!==sn())&&(t.visibility=await cn()),e.options.view===`sky-view`&&(!F||F.key!==ln())&&(t.grid=await un())}catch(e){throw hn(t),e}return t}function _n(t){if(t.skyline){let e=N;N=t.skyline,e&&fn(e),ot=!0,xt=0,R=new Float32Array}if(t.visibility){let e=P;P=t.visibility,e&&pn(e),M=!0}if(t.grid){let e=F;F=t.grid,e&&mn(e),st=!0,ct=!1,L=new Float32Array,I=!0}$(),e.requestLayers()}function vn(){return dn.request({build:gn,commit:_n,discard:hn})}function yn(e){return e===`web-mercator`?i.grid:i.groundGrid}function bn(){let t=e.options;return{projection:t.projection,curvatureCoefficient:Qt(),maximumDistance:t.maxDistance*1e3}}function xn(){ae.write(Float32Array.of(k[0],k[1],e.options.observerHeight,0)),oe.write(Float32Array.from(Z())),nn(),ot=!0,M=!0,I=!0}function Sn(){let t=e.options;w.write(Xe(bn(),yn(t.projection))),T.write(Ze({...bn(),toleranceDegrees:t.toleranceDegrees,sigmaZ:t.sigmaZ,skylineToleranceDegrees:t.skylineToleranceDegrees,targetIgnoreDistance:t.targetIgnoreDistance,targetIgnoreFraction:t.targetIgnoreFraction},yn(t.projection))),re.write(ht({minimumProminence:t.minProminence})),ie.write(xe({curvatureCoefficient:Qt()},a.grid)),e.setReadout(`drop`,`${(Qt()*(t.maxDistance*1e3)**2).toFixed(1)} m at ${t.maxDistance} km`),xn()}function Cn(){let t=Kt(e.options.bearing);if(J>=0&&Q(J)&&qt(G[J],t)<=It)return;let n=-1,r=It;A.forEach((e,i)=>{if(!Q(i))return;let a=qt(G[i],t);a<r&&(n=i,r=a)}),J=n}function wn(t){e.setOptions({bearing:Jt(t)})}function Tn(t){let n=N;if(!n)return;let r=n.azimuthCount;R=Te(t,r),z=Te(t.slice(r*4),r);let i=De(t.slice(r*8),r),a=Te(t.slice(r*12),r),o=Te(t.slice(r*16),r),s=De(t.slice(r*20),1)[0],[c,l]=Z();B=new Float32Array(r*2);let u=new Uint8Array(r),d=-1/0,f=-1,p=0;for(let e=0;e<r;e++){let t=e*2*Math.PI/r,n=Gt(R[e])&&z[e]>0;u[e]=n?1:0;let a=n?z[e]:0;B[e*2]=c+a*Math.sin(t),B[e*2+1]=l+a*Math.cos(t),p+=i[e],n&&R[e]>d&&(d=R[e],f=e)}let h=new Float32Array(r*4),g=new Float32Array(r);for(let e=0;e<r;e++){let t=(e+1)%r;h.set([B[e*2],B[e*2+1],B[t*2],B[t*2+1]],e*4),g[e]=u[e]&&u[t]&&!Wt(z[e],z[t])?1:0}n.ring.write(h),n.weights.write(g);let _=new Float32Array(jt*4);for(let e=0;e<At;e++){let t=Math.round(e*r/At)%r;u[t]&&_.set([c,l,B[t*2],B[t*2+1]],e*4)}n.fan.write(_),q=f>=0?{angle:d,azimuth:f*360/r,meters:[B[f*2],B[f*2+1]]}:null,K=[];for(let e=0;e<r;e++){if(!Number.isFinite(a[e]))continue;let t=o[e],n=Kt(t*360/r),i=(Math.round(t)%r+r)%r,s=z[i],u=n*Math.PI/180;K.push({azimuth:n,angle:R[i],prominence:a[e],distance:s,meters:[c+s*Math.sin(u),l+s*Math.cos(u)]})}K.sort((e,t)=>t.prominence-e.prominence),xt=Math.min(K.length,Mt),n.peakMarkers.write(Float32Array.from(K.slice(0,xt).flatMap(e=>e.meters))),e.setReadout(`skylinePeakCount`,K.length),e.setReadout(`peakSuppression`,s?`converged`:`not converged (the suppression rounds ran out)`),e.setReadout(`highestAngle`,q?`${q.angle.toFixed(1)}° ${Be(q.azimuth)}`:null),e.setReadout(`samples`,`${m(p)} over ${r} rays (${(p/r).toFixed(0)} per ray)`),$(),e.requestLayers(),Dn()}function En(t){H=De(t,j),U=Te(t.slice(j*4),j*4);let n=new Uint32Array(j);for(let e=0;e<j;e++)n[e]=U[e*4+3]>.5?1:0;at.write(n),V=!0,$(),e.requestLayers()}function Dn(){if(!N||L.length===0||R.length!==N.azimuthCount)return;let t=N.azimuthCount,n=0,r=0,i=0;for(let e=0;e<ft;e++){let a=Math.round(e/ft*t)%t,o=R[a],s=L[e];if(!Gt(o)||!Number.isFinite(s))continue;let c=Math.abs(o-s);n+=c,r=Math.max(r,c),i++}e.setReadout(`gridCompare`,i?`mean ${(n/i).toFixed(2)}°, largest ${r.toFixed(2)}° over ${i} sectors`:null)}function On(t){if(!F||lt)return;let n=pt.tryAcquire();if(!n)return;let[r,i]=a.getPixel(...$t()),o=Math.min(Math.max(Math.round(i),0),a.height-1)*a.width+Math.min(Math.max(Math.round(r),0),a.width-1),s=F.directions;t.copyBufferToBuffer({sourceBuffer:F.horizon,sourceOffset:o*s*4,destinationBuffer:n.buffer,destinationOffset:0,size:s*4}),t.copyBufferToBuffer({sourceBuffer:C,sourceOffset:o*4,destinationBuffer:n.buffer,destinationOffset:128,size:4}),n.markEncoded({byteOffset:0,byteLength:132}),lt=!0,I=!1,new Promise(e=>setTimeout(e,0)).then(()=>n.read()).then(t=>{if(b)return;ft=s,L=new Float32Array(t.slice(0,s*4).buffer);let n=new Float32Array(t.slice(128,132).buffer)[0];e.setReadout(`skyViewHere`,Number.isFinite(n)?n.toFixed(2):null),Dn(),e.requestLayers()}).catch(()=>{}).finally(()=>{lt=!1})}async function kn(){if(N)try{e.setReadout(`timing`,`measuring...`);let n=await fe(t,N.compiled,{parameters:void 0,completionBuffer:N.converged,signal:e.signal,runs:5});e.setReadout(`timing`,`${N.compiled.stats.nodeOrder.length} nodes · ${me(n)}`)}catch(t){b||e.setReadout(`timing`,`failed: ${t.message}`)}}function An(){X=null;let t=e.options;if(t.view!==`visibility`||!N||R.length!==N.azimuthCount)return;let n=Z(),r=$t(),a=Tt(t.refraction);if(J>=0&&Q(J)){let o=A[J],s=W[J],c=Et(u,{from:r,to:o.lngLat,stepMeters:Ft,eyeHeightMeters:t.observerHeight,refraction:a,cutoffMeters:s-Math.max(t.targetIgnoreDistance,t.targetIgnoreFraction*s)}),l=V?H[J]:Y.visible,d=V?U[J*4]:c.endAngleDegrees,f=V?U[J*4+1]:c.ridge?.angleDegrees??NaN,p=c.ridge,m=p?`${Math.round(p.heightMeters).toLocaleString(`en-US`)} m ridge ${(p.distanceMeters/1e3).toFixed(1)} km from the eye`:`no ridge in front`,h;h=l===Y.hidden?`${o.displayName} is hidden by a ${m}: ${f.toFixed(1)}° against ${d.toFixed(1)}°`:l===Y.marginal?`${o.displayName} is marginal: ${d.toFixed(1)}° is within the tolerance of the ${m} (${f.toFixed(1)}°)`:`${o.displayName} is visible: ${d.toFixed(1)}° clears the highest ground in front (${f.toFixed(1)}°)`,X={ray:c,targetIndex:J,verdict:h,ridgeAngle:f};let g=o.meters;if(l===Y.hidden&&p){let[e,t]=i.projection.project(p.lngLat[0],p.lngLat[1]);se.write(Float32Array.of(n[0],n[1],e,t)),ce.write(Float32Array.of(e,t,g[0],g[1]))}else se.write(Float32Array.of(n[0],n[1],g[0],g[1])),ce.write(new Float32Array(4));e.setChart(`rayProfile`,yt({ray:c,targetName:o.displayName,targetAngle:d})),e.setReadout(`rayVerdict`,h);return}let o=Kt(t.bearing),s=Dt(c,n,o,Math.min(en(),kt),tn),l=Et(u,{from:r,to:s.lngLat,stepMeters:Ft,eyeHeightMeters:t.observerHeight,refraction:a}),d=Math.round(o/360*N.azimuthCount)%N.azimuthCount,f=R[d],p=l.ridge,m=p?`Towards ${ze(o)} the skyline is ${f.toFixed(1)}° high, set by ground ${(p.distanceMeters/1e3).toFixed(1)} km away`:`Towards ${ze(o)} no ground rises above the eye`;if(X={ray:l,targetIndex:-1,verdict:m,ridgeAngle:p?.angleDegrees??null},p){let[e,t]=i.projection.project(p.lngLat[0],p.lngLat[1]);se.write(Float32Array.of(n[0],n[1],e,t)),ce.write(Float32Array.of(e,t,s.meters[0],s.meters[1]))}else se.write(Float32Array.of(n[0],n[1],s.meters[0],s.meters[1])),ce.write(new Float32Array(4));e.setChart(`rayProfile`,yt({ray:l,targetName:null,targetAngle:null})),e.setReadout(`rayVerdict`,m)}function jn(e){if(e===`sky-view`)return[];let t=$e.filter(Q);if(e===`plain`)return t.slice(0,zt);if(e!==`visibility`)return t.slice(0,Rt);let n=[];if(J>=0&&Q(J)&&n.push(J),V)for(let e of[Y.visible,Y.hidden]){let r=0;for(let i of t){if(r>=2)break;H[i]===e&&!n.includes(i)&&(n.push(i),r++)}}for(let e of t){if(n.length>=Rt)break;n.includes(e)||n.push(e)}return n.slice(0,Rt)}function Mn(){let e=Z(),t=Math.hypot(e[0]-Qe[0],e[1]-Qe[1])<Ht,n=$t();return t?Ge(`gornergrat-station`,{id:`horizon-eye`,coordinate:n,marker:`none`,rank:`subject`,minZoom:0,priority:100}):{kind:`point`,id:`horizon-eye`,coordinate:n,text:`The eye`,detail:We(i.sampleElevation(k[0],k[1])),marker:`none`,rank:`subject`,minZoom:0,priority:100}}function Nn(){let t=e.options,n=[Mn()];if(jn(t.view).forEach((e,r)=>{let i=A[e];n.push({kind:`landform`,id:`horizon-peak-${i.id}`,coordinate:i.lngLat,text:i.displayName,elevationMeters:i.publishedElevationMeters,marker:t.view===`visibility`?`none`:`peak`,tone:W[e]>1e4?`muted`:`ink`,priority:50-r})}),(t.view===`rays`||t.view===`skyline`)&&q&&n.push({kind:`note`,id:`horizon-steepest`,coordinate:tn(...q.meters),title:`${q.angle.toFixed(1)}° at ${Be(q.azimuth)}`,text:`steepest skyline angle`,priority:80}),t.view===`visibility`&&X?.ray.ridge){let{ridge:e}=X.ray;n.push({kind:`note`,id:`horizon-ridge`,coordinate:e.lngLat,title:`${X.ridgeAngle?.toFixed(1)??e.angleDegrees.toFixed(1)}° ridge`,text:`${(e.distanceMeters/1e3).toFixed(1)} km from the eye`,priority:80})}(t.view===`rays`||t.view===`sky-view`)&&n.push({kind:`frame`,id:`horizon-edge`,bounds:i.lngLatBounds,text:`Data ends here`}),e.setAnnotations(`horizon`,n)}function Pn(){let t=e.options;if(t.view===`visibility`)J>=0&&Q(J)?e.setHighlight({kind:`point`,coordinate:A[J].lngLat,radiusPixels:13}):e.setHighlight(null),Zt=!0;else if(t.view===`skyline-peaks`&&wt>=0){let t=K[wt];t&&(e.setHighlight({kind:`point`,coordinate:tn(...t.meters),radiusPixels:11,tone:`signal`,pulse:!0}),Zt=!0)}else Zt&&=(e.setHighlight(null),!1)}function Fn(e){let t=-1,n=1.2;return A.forEach((r,i)=>{if(!Q(i))return;let a=qt(G[i],e.azimuth),o=V?Math.abs(U[i*4]-e.angle):0;a<n&&o<.6&&(t=i,n=a)}),t>=0?A[t].displayName:ze(e.azimuth)}function In(){let t=e.options;if(!N||R.length!==N.azimuthCount)return;let n=jn(t.view),r=[];t.view===`visibility`&&V&&A.forEach((e,t)=>{let n=H[t],i=U[t*4];!Q(t)||!Number.isFinite(i)||(n===Y.visible?r.push({azimuth:G[t],angle:i,code:`visible`}):n===Y.marginal?r.push({azimuth:G[t],angle:i,code:`marginal`}):n===Y.hidden&&r.push({azimuth:G[t],angle:i,code:`hidden`}))});let i=K.slice(0,Vt),a=t.view===`skyline-peaks`?i.map(e=>({azimuth:e.azimuth,label:Fn(e)})):n.filter(e=>e!==J||t.view!==`visibility`).map(e=>({azimuth:G[e],label:A[e].displayName}));e.setChart(`panorama`,vt({view:t.view,azimuthCount:N.azimuthCount,skylineAngles:R,peaks:r,markers:a,skylinePeaks:i,...t.view===`visibility`?{linkLabel:e=>J>=0?A[J].displayName:ze(e)}:{}})),e.setChart(`skylinePeaks`,K.length?bt(K.slice(0,Bt).map(e=>({label:Fn(e),prominence:e.prominence})),e=>{wt=e,Pn()}):null)}function Ln(){let t=e.options,n=A.map((e,t)=>t).filter(Q);e.setReadout(`peaksInRange`,n.length);let r=-1;for(let e of n)(r<0||A[e].publishedElevationMeters>A[r].publishedElevationMeters)&&(r=e);if(e.setReadout(`highestPeak`,r>=0?`${A[r].displayName} ${We(A[r].publishedElevationMeters)}`:null),e.setReadout(`viewSummary`,r>=0?`${n.length} named peaks above ${m(Re)} m lie in range; the highest is ${A[r].displayName} at ${We(A[r].publishedElevationMeters)}.`:`No catalogue peaks lie within the current range.`),V){let t=0,r=0,i=0,a=0;for(let e of n){let n=H[e];n===Y.visible?t++:n===Y.marginal?r++:n===Y.hidden&&i++,U[e*4+3]>.5&&a++}e.setReadout(`visibleCount`,t),e.setReadout(`marginalCount`,r),e.setReadout(`hiddenCount`,i),e.setReadout(`onSkylineCount`,a)}e.setReadout(`rayCount`,`${t.azimuthCount} rays`),e.setReadout(`rayStep`,`${(360/Number(t.azimuthCount)).toFixed(2)}°`),e.setReadout(`eyeHeight`,`${t.observerHeight} m`),e.setReadout(`gridDirectionsReadout`,`${t.gridDirections} sectors`),e.setReadout(`gridRadiusReadout`,`${(Rn()/1e3).toFixed(1)} km`)}let Rn=()=>Number(e.options.gridRadius)*a.groundCellSize;function zn(){let t=e.options;return t.view===`sky-view`?[Rn()]:t.view===`plain`?[]:t.view===`visibility`?J>=0?[W[J]]:[]:[en()]}function Bn(){let t=e.options,n=zn(),r=Le(t,{rayCount:Number(t.azimuthCount),gridRadiusKilometers:Rn()/1e3}),a=`${r}|${n.map(Math.round).join(`,`)}`;a!==Xt&&(Xt=a,e.setFurniture({title:{subtitle:r,sample:Ne(i),chips:[Pe]},scaleBar:{units:`metric`,ticks:n}}))}function Vn(){let t=e.options;t.view===`sky-view`?e.setCost({records:a.pixelCount,note:`${t.gridDirections} sectors, ${t.gridRadius} steps each`}):e.setCost({records:Number(t.azimuthCount),note:`rays, and ${j} peaks tested`})}function Hn(){Yt=null,!b&&(An(),Ln(),Bn(),In(),Nn(),Pn(),Vn(),e.requestLayers())}function $(){Yt!==null||b||(Yt=setTimeout(Hn,Lt))}function Un(t,n){let r=e.getViewport();if(!r)return 1/0;let[i,a]=tn(t[0],t[1]),[o,s]=r.project([i,a]);return Math.hypot(o-n.pixel[0],s-n.pixel[1])}function Wn(t,n){let r=n?jn(e.options.view):A.map((e,t)=>t),i=-1,a=Pt;for(let e of r){if(!Q(e))continue;let n=Un(A[e].meters,t);n<a&&(i=e,a=n)}return i}function Gn(e){if(!e.coordinate)return;let[t,n]=i.projection.project(e.coordinate[0],e.coordinate[1]);k=i.getPixelFromMeters(t,n),xn(),J>=0&&wn(G[J]),$()}function Kn(t){let n=W[t],r=i.sampleElevation(k[0],k[1])+e.options.observerHeight;return Math.atan2(A[t].demElevationMeters-Qt()*n**2-r,n)*180/Math.PI}function qn(t){let n=e.options,r=A[t],i=je(e.ground()),a=n.view===`visibility`&&V,o=H[t],s=V?U[t*4]:Kn(t),c=[],l;if(a&&c.push({label:`Visibility`,value:Ut[o]??`code ${o}`,swatch:o===Y.visible?i.visible:o===Y.marginal?i.marginal:i.hidden,emphasis:!0}),c.push({label:`Angle above the horizontal`,value:s.toFixed(1),unit:`°`,emphasis:!a}),a&&Number.isFinite(U[t*4+1])&&c.push({label:`Highest ground in front`,value:U[t*4+1].toFixed(1),unit:`°`}),c.push({label:`Distance`,value:`${we(W[t])} ${Be(G[t])}`},{label:`Elevation`,value:`${Math.round(r.publishedElevationMeters).toLocaleString(`en-US`)} m`,unit:`(DEM cell ${Math.round(r.demElevationMeters).toLocaleString(`en-US`)} m)`}),a&&o===Y.hidden){let e=W[t],i=Et(u,{from:$t(),to:r.lngLat,stepMeters:Ft,eyeHeightMeters:n.observerHeight,refraction:Tt(n.refraction),cutoffMeters:e-Math.max(n.targetIgnoreDistance,n.targetIgnoreFraction*e)});i.ridge&&(l=`Hidden by the ridge at ${we(i.ridge.distanceMeters)} (${U[t*4+1].toFixed(1)} against ${s.toFixed(1)}°)`)}else a&&U[t*4+3]>.5&&(l=`On the skyline`);return{title:r.displayName,subtitle:`Summit`,rows:c,...l?{note:l}:{},highlight:{kind:`point`,coordinate:r.lngLat}}}function Jn(t){let[n,r]=i.projection.project(t[0],t[1]),[a,o]=i.getPixelFromMeters(n,r),s=Z(),c=n-s[0],l=r-s[1],u=Math.hypot(c,l),d=i.sampleElevation(a,o),f=i.sampleElevation(k[0],k[1])+e.options.observerHeight,p=[{label:`Elevation`,value:Math.round(d).toLocaleString(`en-US`),unit:`m`,emphasis:!0}];if(u>100){let e=Math.atan2(d-Qt()*u**2-f,u)*180/Math.PI;p.push({label:`Angle from the eye`,value:e.toFixed(1),unit:`°`},{label:`Distance`,value:`${we(u)} ${Be(Ee(c,l))}`})}return{title:`Ground`,rows:p}}e.setReadout(`catalogue`,`${j} named summits above ${m(Re)} m, one per massif`),e.setReadout(`peakFloor`,`${m(Re)} m`),e.setReadout(`grid`,`${d} × ${f} cells at ${i.groundCellSize.toFixed(1)} m (Web Mercator ${i.mercatorCellSize.toFixed(1)} m); grid horizon ${a.width} × ${a.height} at ${a.groundCellSize.toFixed(0)} m`),Sn(),Cn();{let e=t.createCommandEncoder({id:`horizon-setup-encoder`});Je.encode(e,{parameters:void 0}),t.submit(e.finish()),mt=!0}await vn(),Ln(),$();let Yn=new Set([`projection`,`traversal`,`azimuthCount`,`peakWindow`,`gridDirections`,`gridRadius`,`gridAlgorithm`]);return{getCompiledGraphs:()=>{let e=[Je];return N&&e.push(N.compiled),P&&e.push(P.compiled),F&&e.push(F.compiled),e},setOption(t){if(t===`view`){vn(),I=!0,$(),e.requestLayers();return}if(t===`bearing`){Cn(),$(),e.requestLayers();return}if(Yn.has(t)){Sn(),vn(),$(),e.requestLayers();return}t===`refraction`&&(st=!0),Sn(),Cn(),$()},onAction(e){e===`measure`&&kn()},onGroundChange(t){_.setGround(t),ee=Ve(t),e.setLegendData(`tables`,ee),Xt=``,e.requestLayers()},onThemeChange(){e.requestLayers()},encode(t){!mt||!N||!P||(e.options.view===`sky-view`&&F&&(st&&(F.compiled.encode(t,{parameters:void 0}),st=!1,ct=!0,I=!0),I&&On(t)),ot&&=(N.compiled.encode(t,{parameters:void 0}),N.reader.request(t),!1),M&&=(P.compiled.encode(t,{parameters:void 0}),P.reader.request(t),!1),N.reader.flush(t),P.reader.flush(t))},getLayers(){let t=e.options,n=e.ground(),r=ye[n],i=_e(r.ink),o=_e(r.halo,Math.round(r.haloAlpha*255)),s=Me(n),c=je(n),l=[0,0,0,0],u=[_.getLayer()],d=N,f=d!==null&&R.length===d.azimuthCount;if(t.view===`sky-view`&&ct&&u.push(new pe({id:`horizon-sky-view`,coordinateOrigin:g,gridSize:[a.width,a.height],bounds:a.bounds,rowOrigin:`north`,values:C,valueFormat:`float32`,colormap:`grayscale`,valueRange:Fe,rampRange:Ie,blending:`multiply`})),f&&d){let e=`horizon-skyline-${d.key}`;t.view===`visibility`?u.push(new de({id:`${e}-context`,coordinateOrigin:g,segments:d.ring,instanceCount:d.azimuthCount,weights:d.weights,color:[i[0],i[1],i[2],120],widthPixels:1.4,outlineColor:o,outlineWidthPixels:.8})):t.view!==`plain`&&t.view!==`sky-view`&&u.push(new de({id:e,coordinateOrigin:g,segments:d.ring,instanceCount:d.azimuthCount,weights:d.weights,values:d.angle,valueFormat:`float32`,colormap:`grayscale`,...ve(ee.skyline),widthPixels:2.5,outlineColor:o,outlineWidthPixels:1})),t.view===`rays`&&u.push(new de({id:`horizon-fan-${d.key}`,coordinateOrigin:g,segments:d.fan,instanceCount:At,color:[i[0],i[1],i[2],77],widthPixels:.8})),t.view===`skyline-peaks`&&xt>0&&u.push(new E({id:`horizon-skyline-ticks-${d.key}`,coordinateOrigin:g,positions:d.peakMarkers,instanceCount:xt,shape:`diamond`,radiusPixels:5,color:i,outlineColor:o,outlineWidthPixels:1.5}))}if(t.view===`visibility`&&(X&&u.push(new de({id:`horizon-ray-after`,coordinateOrigin:g,segments:ce,instanceCount:1,color:[i[0],i[1],i[2],170],widthPixels:1.6,dashArray:[5,4],cap:`butt`}),new de({id:`horizon-ray-before`,coordinateOrigin:g,segments:se,instanceCount:1,color:i,widthPixels:2.2,outlineColor:o,outlineWidthPixels:1.2,cap:`round`})),V&&j>0)){let e={coordinateOrigin:g,positions:nt,instanceCount:j,values:rt,valueFormat:`uint32`,colormap:`category`},t=(e,t)=>[0,1,2,3,4].map(n=>n===e?t:l);u.push(new E({...e,id:`horizon-hidden-fill`,palette:t(Y.hidden,[o[0],o[1],o[2],150]),radiusPixels:6.5}),new E({...e,id:`horizon-hidden-ring`,palette:t(Y.hidden,c.hidden),shape:`ring`,radiusPixels:7,outlineWidthPixels:2}),new E({id:`horizon-on-skyline`,coordinateOrigin:g,positions:nt,instanceCount:j,values:at,valueFormat:`uint32`,colormap:`category`,palette:[l,i],shape:`ring`,radiusPixels:10.5,outlineWidthPixels:1.5}),new E({...e,id:`horizon-marginal-ring`,palette:t(Y.marginal,c.marginal),shape:`ring`,radiusPixels:7,outlineWidthPixels:2}),new E({...e,id:`horizon-marginal-dot`,palette:t(Y.marginal,c.marginal),radiusPixels:3,outlineColor:o,outlineWidthPixels:1}),new E({...e,id:`horizon-visible`,palette:t(Y.visible,c.visible),radiusPixels:6,outlineColor:o,outlineWidthPixels:1.5}))}return u.push(new E({id:`horizon-observer-halo`,coordinateOrigin:g,positions:oe,instanceCount:1,radiusPixels:12.5,color:s.halo,outlineColor:s.halo,outlineWidthPixels:1}),new E({id:`horizon-observer`,coordinateOrigin:g,positions:oe,instanceCount:1,radiusPixels:8,color:s.eye,outlineColor:s.outline,outlineWidthPixels:2})),u},getTooltip(t){if(!t.coordinate)return null;if(e.getViewport()&&Un(Z(),t)<=Pt&&e.options.view!==`sky-view`){let t=i.sampleElevation(k[0],k[1]);return{title:`The eye`,subtitle:`Drag it to move the viewpoint`,rows:[{label:`Eye elevation`,value:Math.round(t+e.options.observerHeight).toLocaleString(`en-US`),unit:`m`,emphasis:!0},{label:`Ground`,value:Math.round(t).toLocaleString(`en-US`),unit:`m`},{label:`Eye height`,value:e.options.observerHeight,unit:`m`}]}}if(e.options.view!==`sky-view`&&e.options.view!==`plain`){let n=Wn(t,e.options.view!==`visibility`);if(n>=0)return qn(n)}else if(e.options.view===`plain`){let e=Wn(t,!0);if(e>=0)return qn(e)}return Jn(t.coordinate)},onClick(t){if(e.options.view!==`visibility`||!t.coordinate)return!1;let n=Wn(t,!1);if(n>=0)J=n,wn(G[n]);else{let[e,n]=i.projection.project(t.coordinate[0],t.coordinate[1]),[r,a]=Z();J=-1,wn(Ee(e-r,n-a)),Cn()}return $(),e.requestLayers(),!0},onDragStart(e){return Un(Z(),e)>Nt?!1:(gt=!0,!0)},onDrag(e){gt&&Gn(e)},onDragEnd(e){gt&&=(Gn(e),!1)},destroy(){b=!0,Yt!==null&&clearTimeout(Yt),N?.reader.stop(),P?.reader.stop(),_.destroy(),v.destroy()}}}export{X as createHorizon};