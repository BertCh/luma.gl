import{D as e,I as t,O as n,T as r,z as i}from"./resources-CwvhEOBe.js";var a=`
fn isFiniteGridValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

// The scaled cross-zero branch avoids overflowing maximum - minimum for valid float32 bounds.
fn getGridCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
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
}
`;function o(e){return Number.isFinite(Math.fround(e))}function s(e){let t=Math.fround(e);if(!Number.isFinite(t))throw Error(`GPU spatial numeric properties must be representable as finite float32 values`);if(Object.is(t,-0))return`-0.0`;let n=String(t);return n.includes(`.`)||/e/i.test(n)?n:`${n}.0`}function c(e,t,n){for(let[i,a]of n.entries())if(t.some(e=>r(e,a))||n.slice(0,i).some(e=>r(e,a)))throw Error(`${e} writable views must not overlap inputs or other outputs`)}function l(r,a){for(let e of Object.values(a.bindings))if(n(e).size>r.device.limits.maxStorageBufferBindingSize)throw Error(`${a.id} active chunks must fit a storage binding`);return[t({id:a.id,resources:a.resources,compile:({device:t})=>{let n=new i(t,{id:a.id,source:a.source,shaderLayout:{bindings:Object.keys(a.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}}),r=Object.entries(a.bindings),o=[],s={},c={};return{encode:({computePass:t,getBuffer:i})=>{let l=r.map(([,e])=>i(e));if(l.some((e,t)=>e!==o[t])||l.length!==o.length){o=l,s={};for(let[t,[n,i]]of r.entries()){let r=l[t];s[n]=e(i,()=>r)}c={}}n.dispatch(t,{bindings:s,_bindGroupCacheKeys:{0:c},x:a.dispatch.x,y:a.dispatch.y,z:a.dispatch.z})},destroy:()=>n.destroy()}}})]}export{c as a,o as i,l as n,s as r,a as t};