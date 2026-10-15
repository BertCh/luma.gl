import{_ as e,d as t,y as n}from"./resources-CwvhEOBe.js";import{l as r}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as i,r as a}from"./wgsl-common-BNRVzgd8.js";function o(e){return`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${a}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
${i}
fn isNoDataValue(value: f32) -> bool {
  return isNaNValue(value)${e===void 0?``:` || value == ${r(e)}`};
}`}function s(e,t,r,i,a){if(r&&(n(r,[i],`${e} ${t}`),r.length<a))throw Error(`${e} ${t} must hold at least ${a} rows`)}function c(e,n,r,i,a){t(e,n,r,i,a)}function l(e,t){if(t!==void 0&&!Number.isFinite(t))throw Error(`${e} noDataValue must be finite (NaN cells are always nodata)`)}function u(e,t,n){let r=t.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(r).size!==r.length)throw Error(`${e} outputs must not share buffers`);let i=new Set(n.filter(e=>e!==void 0).map(e=>e.buffer));if(r.some(e=>i.has(e)))throw Error(`${e} outputs must not share buffers with inputs`)}function d(t,n,r){e(t,n,r)}function f(e,t){return`
fn ${e}(first: u32, count: u32, value: f32, closedRight: bool) -> u32 {
  var low = 0u;
  var high = count;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let boundary = ${t}[${t}Offset + first + middle];
    let isBelow = select(boundary <= value, boundary < value, closedRight);
    if (isBelow) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`}export{d as a,c as i,o as n,l as o,u as r,s,f as t};