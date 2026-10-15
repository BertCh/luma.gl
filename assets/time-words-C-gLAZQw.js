var e=-(2n**63n),t=2n**63n-1n,n=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;function r(n){let r,i=0;if(typeof n==`bigint`)r=n;else{if(!Number.isFinite(n))throw Error(`Time words require a finite instant`);let e=Math.floor(n);if(i=Math.fround(n-e),i>=1&&(e+=1,i=0),!Number.isSafeInteger(e))throw Error(`Time words require an instant whose integer part is a safe integer`);r=BigInt(e)}if(r<e||r>t)throw Error(`Time words require an instant inside the signed 64-bit range`);let a=BigInt.asUintN(64,r);return{low:Number(a&4294967295n),high:Number(a>>32n),fraction:i}}function i(e){if(!n)throw Error(`Int64 time words require a little-endian platform`);return new Uint32Array(e.buffer,e.byteOffset,e.length*2)}function a(e,t=new Uint32Array(8)){if(t.length<8)throw Error(`Time word window target must hold 8 elements`);let n=e.startFadeDuration??0,i=e.endFadeDuration??0;for(let e of[n,i])if(!Number.isFinite(e)||e<0)throw Error(`Time window fade durations must be finite and non-negative`);let a=r(e.start),o=r(e.end),s=new Float32Array([a.fraction,o.fraction,n,i]);return t.set([a.low,a.high,o.low,o.high]),t.set(new Uint32Array(s.buffer),4),t}var o=`
fn timeWordsSubtract(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let borrow = select(0u, 1u, a.x < b.x);
  return vec2<u32>(a.x - b.x, a.y - b.y - borrow);
}

fn timeWordsIsNegative(difference: vec2<u32>) -> bool {
  return (difference.y & 0x80000000u) != 0u;
}

fn timeWordsToF32(difference: vec2<u32>) -> f32 {
  let negative = timeWordsIsNegative(difference);
  var low = difference.x;
  var high = difference.y;
  if (negative) {
    low = ~difference.x + 1u;
    high = ~difference.y + select(0u, 1u, low == 0u);
  }
  let magnitude = f32(high) * 4294967296.0 + f32(low);
  return select(magnitude, -magnitude, negative);
}

// Signed order: high words with the sign bit flipped compare as unsigned, then the low words.
fn timeWordsCompare(a: vec2<u32>, b: vec2<u32>) -> i32 {
  let aHigh = a.y ^ 0x80000000u;
  let bHigh = b.y ^ 0x80000000u;
  if (aHigh != bHigh) {
    return select(-1i, 1i, aHigh > bHigh);
  }
  if (a.x != b.x) {
    return select(-1i, 1i, a.x > b.x);
  }
  return 0i;
}

// The magnitude |a - b| is below 2^64, so subtracting the smaller from the larger is exact as an
// unsigned 64-bit value even when the signed difference overflows Int64.
fn timeWordsDifferenceToF32(a: vec2<u32>, b: vec2<u32>) -> f32 {
  let isForward = timeWordsCompare(a, b) >= 0i;
  let magnitudeWords = select(timeWordsSubtract(b, a), timeWordsSubtract(a, b), isForward);
  let magnitude = f32(magnitudeWords.y) * 4294967296.0 + f32(magnitudeWords.x);
  return select(-magnitude, magnitude, isForward);
}

fn isTimeWordsAtLeast(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> bool {
  let order = timeWordsCompare(a, b);
  if (order == 0i) {
    return aFraction >= bFraction;
  }
  return order > 0i;
}

fn timeWordsDifference(a: vec2<u32>, aFraction: f32, b: vec2<u32>, bFraction: f32) -> f32 {
  return timeWordsDifferenceToF32(a, b) + (aFraction - bFraction);
}
`;export{a as n,i as r,o as t};