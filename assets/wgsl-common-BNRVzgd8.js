var e=`
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
`,t=`
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
`,n=`
fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }
`,r=`
fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
`,i=`
fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
`,a=`
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`,o=`
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`,s=`
fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`,c=`
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`,l=`
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
`,u=`
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}
`,d=`
fn erfcValue(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}
`;export{n as a,o as c,u as d,l as f,s as i,a as l,d as n,t as o,e as r,i as s,c as t,r as u};