var e={lineIntegralConvolutionNoise:{id:1,word:`key`},streamlinesJitter:{id:2,word:`key`},randomPointsOnLine:{id:3,word:`key`},particleAdvectionSpawn:{id:4,word:`key`},particleAdvectionDrop:{id:5,word:`key`},dotDensityRemainder:{id:6,word:`key`},dotDensityCandidate:{id:7,word:`key`},maxPRegionsSeedOrder:{id:8,word:`key`},ripleyEnvelopeSimulation:{id:1380536390,word:`key`},spatialWeightsTracesProbeKey:{id:1414676803,word:`key`},spatialWeightsTracesProbeCounter:{id:1347571522,word:`counter`},permutationFeistel:{id:4262821392,word:`counter`},permutationShuffle:{id:1510693387,word:`counter`},knoxTestShuffle:{id:1263423320,word:`counter`},mantelTestShuffle:{id:1296125524,word:`counter`},geographicallyWeightedRegressionNonstationarityShuffle:{id:1196905038,word:`counter`},localPermutationTest:{id:68769,word:`counter`},spatialScanStatistic:{id:1554470727,word:`counter`},networkKFunction:{id:1262900558,word:`counter`}},t=`
fn philoxMulHiLo(a: u32, b: u32) -> vec2<u32> {
  let aLow = a & 0xffffu;
  let aHigh = a >> 16u;
  let bLow = b & 0xffffu;
  let bHigh = b >> 16u;
  let lowLow = aLow * bLow;
  let highLow = aHigh * bLow;
  let lowHigh = aLow * bHigh;
  let highHigh = aHigh * bHigh;
  let middle = (lowLow >> 16u) + (highLow & 0xffffu) + (lowHigh & 0xffffu);
  let high = highHigh + (highLow >> 16u) + (lowHigh >> 16u) + (middle >> 16u);
  return vec2<u32>(high, a * b);
}

fn philoxRound(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
  let product0 = philoxMulHiLo(3528531795u, counter.x);
  let product1 = philoxMulHiLo(3449720151u, counter.z);
  return vec4<u32>(
    product1.x ^ counter.y ^ key.x,
    product1.y,
    product0.x ^ counter.w ^ key.y,
    product0.y
  );
}

fn philox4x32(counterIn: vec4<u32>, keyIn: vec2<u32>) -> vec4<u32> {
  var counter = counterIn;
  var key = keyIn;
  for (var roundIndex = 0u; roundIndex < 9u; roundIndex = roundIndex + 1u) {
    counter = philoxRound(counter, key);
    key = key + vec2<u32>(2654435769u, 3144134277u);
  }
  return philoxRound(counter, key);
}

fn philoxUnitFloat(word: u32) -> f32 {
  return f32(word >> 8u) * (1.0 / 16777216.0);
}
`;export{t as n,e as t};