import{i as e,r as t}from"./earth-xB16XgAa.js";import{l as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var r=t*(2-t),i=e*Math.PI/180,a=.017453292519943295,o=`const DEGREES_TO_RADIANS: f32 = ${a};`;function s(e){let[t,n,,r,i,a]=e.affine,o=e.pixelInterpretation===`point`?-.5:0;return{cellSize:[Math.hypot(t,r),Math.hypot(n,i)],northEdge:i*o+a,southEdge:i*(e.height+o)+a}}function c(e,t,n){let{cellSize:r,northEdge:i,southEdge:a}=s(t);e[n.cellSizeIndex]=r[0],e[n.cellSizeIndex+1]=r[1],e[n.northEdgeIndex]=i,e[n.northEdgeIndex+1]=a}function l(t,n){if(t===`uniform`)return[1,1];if(t===`web-mercator`){let t=n/e,i=Math.tanh(t),a=1-r*i*i,o=1/Math.sqrt(a),s=1/Math.cosh(t);return[o*s,o*(1-r)/a*s]}let a=n*Math.PI/180,o=Math.sin(a),s=1-r*o*o,c=i/Math.sqrt(s);return[c*Math.cos(a),c*(1-r)/s]}function u(t){let o=n(r),s=n(1-r);return t===`web-mercator`?`// WGS84 ground meters per EPSG:3857 meter at the row latitude atan(sinh(y / a)):
  // x = N cos(lat) / a, y = M cos(lat) / a, with cos(lat) = 1 / cosh(y / a).
  let mercatorY = edge * ${n(1/e)};
  let sinLatitude = tanh(mercatorY);
  let curvature = 1.0 - ${o} * sinLatitude * sinLatitude;
  let primeVertical = 1.0 / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical, primeVertical * ${s} / curvature) / cosh(mercatorY);`:`// WGS84 meters per degree at the row latitude: x = N cos(lat) PI / 180, y = M PI / 180.
  let latitude = edge * ${n(a)};
  let sinLatitude = sin(latitude);
  let curvature = 1.0 - ${o} * sinLatitude * sinLatitude;
  let primeVertical = ${n(i)} / sqrt(curvature);
  let groundScale = vec2<f32>(primeVertical * cos(latitude), primeVertical * ${s} / curvature);`}function d(e,t=`f32`){let n=e===`south`?-1:1;return`const ROW_NORTH_SIGN: ${t} = ${t===`f32`?`${n}.0`:n};`}function f(e,t){let n=t.settingsName??`settings`,r=`${n}Offset`,i=`vec2<f32>(${n}[${r} + ${t.cellSizeIndex}u], ${n}[${r} + ${t.cellSizeIndex+1}u])`;return e===`uniform`?`fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${i};
  return cellSize;
}`:`fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${i};
  let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = ${`mix(${n}[${r} + ${t.northEdgeIndex}u], ${n}[${r} + ${t.northEdgeIndex+1}u], rowFraction)`};
  ${u(e)}
  return cellSize * groundScale;
}`}function p(e,t,n,r){let[i,a]=t.cellSize;if(e===`uniform`)return[i,a];let o=t.northEdge??0,[s,c]=l(e,o+((t.southEdge??0)-o)*(n/r));return[i*s,a*c]}var m=`
// Unit pixel step along a ground-space unit direction on cells of groundCell meters.
fn getPixelDirection(groundDirection: vec2<f32>, groundCell: vec2<f32>) -> vec2<f32> {
  if (groundCell.x == groundCell.y) { return groundDirection; }
  return normalize(groundDirection / groundCell);
}`;function h(e,t){if(![`uniform`,`web-mercator`,`geographic`].includes(t??`uniform`))throw Error(`${e} cellSizeMode must be uniform, web-mercator, or geographic`)}function g(e,t){if(![`south`,`north`].includes(t??`south`))throw Error(`${e} rowDirection must be south or north`)}export{f as a,h as c,p as i,g as l,m as n,u as o,s as r,d as s,o as t,c as u};