import{y as e}from"./resources-CwvhEOBe.js";import{l as t}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as n}from"./wgsl-common-BNRVzgd8.js";import{r,t as i}from"./raster-grid-D2bmCwkc.js";var a=`
const PI: f32 = 3.141592653589793;
${i}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${n}
// Takes a runtime operand (unlike the shared getNaN) so the NaN is not constant-folded away.
fn getNaN(index: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (index & 0u)); }`;function o(e,t){if(!r(t).cellSize.every(e=>Number.isFinite(e)&&e>0))throw Error(`${e} grid cell size must be finite and positive`)}function s(t,n,r,i,a){if(r&&(e(r,[i],`${t} ${n}`),r.length!==a))throw Error(`${t} ${n} must contain ${a} ${i} values`)}function c(e,t,n,r){if(n>r)throw Error(`${e} ${t} needs ${n} bytes in one binding, above the device limit ${r}; use fewer directions or a smaller tile`)}var l=180/65534,u=Math.fround(65534/180);function d(e,t){if(![`float32`,`unorm16`].includes(t))throw Error(`${e} horizonFormat must be float32 or unorm16`)}function f(e,t,n){let r=t*n;return e===`unorm16`?Math.ceil(r/2):r}function p(e,t,n,r,i,a){s(e,t,n,r===`unorm16`?`uint32`:`float32`,f(r,i,a))}function m(e,n){return e===`float32`?`fn readHorizonAngle(element: u32) -> f32 {
  return ${n}[${n}Offset + element];
}`:`fn readHorizonAngle(element: u32) -> f32 {
  let word = ${n}[${n}Offset + (element >> 1u)];
  let code = (word >> ((element & 1u) * 16u)) & 0xffffu;
  if (code == 0u) { return getNaN(element); }
  return f32(code - 1u) * ${t(l)} - 90.0;
}`}function h(e,n,r){return`const DIRECTION_COUNT: u32 = ${r}u;
const SECTOR_DEGREES: f32 = ${t(360/r)};
${m(e,n)}
fn getHorizonAtAzimuth(pixel: u32, azimuthDegrees: f32) -> f32 {
  let wrappedAzimuth = azimuthDegrees - 360.0 * floor(azimuthDegrees / 360.0);
  let sectorPosition = wrappedAzimuth / SECTOR_DEGREES;
  let lowerSector = min(u32(floor(sectorPosition)), DIRECTION_COUNT - 1u);
  let upperSector = (lowerSector + 1u) % DIRECTION_COUNT;
  let blend = clamp(sectorPosition - f32(lowerSector), 0.0, 1.0);
  return mix(
    readHorizonAngle(pixel * DIRECTION_COUNT + lowerSector),
    readHorizonAngle(pixel * DIRECTION_COUNT + upperSector),
    blend
  );
}
fn getSolarDiskVisibility(altitudeDegrees: f32, radiusDegrees: f32, horizonAngle: f32) -> f32 {
  var visibility = 0.0;
  if (altitudeDegrees + radiusDegrees > 0.0) {
    if (radiusDegrees > 0.0) {
      let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
      visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
    } else {
      visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
    }
  }
  return visibility;
}`}export{d as a,o as c,h as i,s as l,a as n,p as o,f as r,c as s,u as t};