var e=`
fn isOrderStatisticProbability(p: f32) -> bool {
  // Ordered comparisons with NaN are false, but a compiler may fold them; test the bits.
  let bits = bitcast<u32>(p);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return false;
  }
  return p >= 0.0 && p <= 1.0;
}

fn getOrderStatisticKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

// Identity the compiler cannot fold: the bit operation on a runtime zero stops multiply-add
// fusion, so every f32 step rounds exactly like a Math.fround mirror.
fn roundOrderStatisticStep(value: f32, zero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | zero);
}

fn interpolateOrderStatistic(low: f32, high: f32, fraction: f32, zero: u32) -> f32 {
  if (fraction == 0.0 || low == high) {
    return low;
  }
  let difference = roundOrderStatisticStep(high - low, zero);
  if ((bitcast<u32>(difference) & 0x7f800000u) != 0x7f800000u) {
    return low + roundOrderStatisticStep(difference * fraction, zero);
  }
  // high - low overflowed; a convex combination of two finite values cannot.
  return roundOrderStatisticStep(low * (1.0 - fraction), zero) +
    roundOrderStatisticStep(high * fraction, zero);
}

fn getOrderStatisticMidpoint(low: f32, high: f32, zero: u32) -> f32 {
  if (low == high) {
    return low;
  }
  let sum = roundOrderStatisticStep(low + high, zero);
  if ((bitcast<u32>(sum) & 0x7f800000u) != 0x7f800000u) {
    return sum * 0.5;
  }
  return roundOrderStatisticStep(low * 0.5, zero) + roundOrderStatisticStep(high * 0.5, zero);
}
`;export{e as t};