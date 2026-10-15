import{M as e}from"./resources-CwvhEOBe.js";import{l as t,s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var r=4;Math.PI/180;function i(t,n){if(!n)throw Error(`${t} longitude-latitude rasters need parameters`);if(e(n,[`float32`],`${t} parameters`),n.length<r)throw Error(`${t} parameters must contain at least ${r} values`)}function a(e){return`
const SPHERE_RADIUS: f32 = ${t(e)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn getCellAreaOfHeight(row: u32, cellHeight: f32) -> f32 {
  let cellWidth = extent[extentOffset + 2u];
  let middle = (extent[extentOffset + 1u] + (f32(row) + 0.5) * cellHeight) * DEGREES_TO_RADIANS;
  // R^2 dLon (sin(top) - sin(bottom)) = 2 R^2 dLon cos(middle) sin(dLat / 2).
  return 2.0 * SPHERE_RADIUS * SPHERE_RADIUS * (cellWidth * DEGREES_TO_RADIANS) * cos(middle) *
    sin(0.5 * cellHeight * DEGREES_TO_RADIANS);
}
fn getCellArea(row: u32) -> f32 {
  return getCellAreaOfHeight(row, extent[extentOffset + 3u]);
}`}function o(e){return`${a(e)}
fn getPlacedCellArea(row: u32) -> f32 {
  let signedCellHeight = extent[extentOffset + 3u];
  let countedRow = select(row, HEIGHT - 1u - row, signedCellHeight < 0.0);
  return getCellAreaOfHeight(countedRow, abs(signedCellHeight));
}`}function s(e,t){return n(e,{id:t.id,operation:t.operation,variant:`spherical-cell-weights${t.cellWeights?`-masked`:``}`,bindings:[{name:`extent`,view:t.extent,type:`f32`,access:`read`},...t.cellWeights?[{name:`cellWeights`,view:t.cellWeights,type:`f32`,access:`read`}]:[],{name:`output`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:t.cellCount,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
${o(t.radius)}`,body:`var weight = ${t.cellWeights?`cellWeights[cellWeightsOffset + index]`:`1.0`};
  weight = weight * getPlacedCellArea(index / WIDTH);
  // Non-finite (NaN or infinity) and non-positive weights carry no area.
  if ((bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u || !(weight > 0.0)) {
    weight = 0.0;
  }
  output[outputOffset + index] = weight;`})}export{i,o as n,a as r,s as t};