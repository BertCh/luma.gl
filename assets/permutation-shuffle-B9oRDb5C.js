function e(e){return`
const PERMUTATION_SHUFFLE_STREAM: u32 = ${e>>>0}u;
const PERMUTATION_SHUFFLE_MAXIMUM_ROUNDS: u32 = 150u;
var<workgroup> swapOrNotKeys: array<u32, 150>;
var<workgroup> swapOrNotSalts: array<u32, 150>;

fn getSwapOrNotRoundCount(count: u32) -> u32 {
  return select(6u * (32u - countLeadingZeros(count - 1u)) + 6u, 0u, count <= 1u);
}

fn hashSwapOrNotUint32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn loadSwapOrNotRounds(key: vec2<u32>, permutation: u32, count: u32, local: u32, size: u32) {
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = local; round < roundCount; round += size) {
    var stream = createPhiloxStream(key, permutation, round, PERMUTATION_SHUFFLE_STREAM);
    swapOrNotKeys[round] = nextPhiloxBelow(&stream, count);
    swapOrNotSalts[round] = nextPhiloxUint32(&stream);
  }
}

fn getSwapOrNotPermutationIndex(index: u32, count: u32) -> u32 {
  var position = index;
  let roundCount = getSwapOrNotRoundCount(count);
  for (var round = 0u; round < roundCount; round++) {
    var partner = swapOrNotKeys[round] + count - position;
    if (partner >= count) {
      partner = partner - count;
    }
    let canonical = max(position, partner);
    if ((hashSwapOrNotUint32(canonical ^ swapOrNotSalts[round]) >> 31u) != 0u) {
      position = partner;
    }
  }
  return position;
}
`}export{e as t};