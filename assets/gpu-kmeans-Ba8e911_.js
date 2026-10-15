import{E as e,h as t}from"./analysis-frame-Blu0tYdw.js";import{_ as n,d as r,h as i,w as a,y as o}from"./resources-CwvhEOBe.js";import{t as s}from"./gpu-compaction-Cpx1OCRM.js";import{t as ee}from"./gpu-group-aggregation-DrNKSCi8.js";import{o as te,s as c}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as ne,c as re,u as ie}from"./wgsl-common-BNRVzgd8.js";import{t as ae}from"./spatial-clustering-parameters-OGQKgHDK.js";var l=Object.freeze({...t,measure:`distance`}),u=`GPUKMeans`,d=`0xffffffffu`,f=256,oe=16,se=256;function p(e={}){let t=e.seed??0,n=e.tolerance??0;if(!Number.isInteger(t)||t<0||t>4294967295)throw Error(`GPUKMeans seed must be a uint32`);if(!Number.isFinite(n)||n<0)throw Error(`GPUKMeans tolerance must be a finite number >= 0`);let r=new Uint32Array(2);return r[0]=t,new Float32Array(r.buffer)[1]=n,r}var m=class{id;props;initialization;restartCount;spatialContext;constructor(t){let n=t.id??`kmeans`;this.id=n,this.props=t,this.initialization=t.initialization??`kmeans++`,this.restartCount=t.restartCount??1,this.spatialContext=e(n,t.spatialContext,l),o(t.positions,[`float32x2`],`${n} positions`);let a=t.positions.length;if(a<1)throw Error(`${n} positions must hold at least one row`);if(!Number.isInteger(t.k)||t.k<1||t.k>256)throw Error(`${n} k must be an integer in [1, 256]`);if(r(n,`maximumIterations`,t.maximumIterations,1,64),this.initialization!==`first-valid`&&this.initialization!==`kmeans++`)throw Error(`${n} initialization must be 'first-valid' or 'kmeans++'`);if(!Number.isInteger(this.restartCount)||this.restartCount<1||this.restartCount>32)throw Error(`${n} restartCount must be an integer in [1, 32]`);if(this.restartCount>1&&this.initialization!==`kmeans++`)throw Error(`${n} restartCount above 1 needs the 'kmeans++' initialization`);if(o(t.parameters,[`uint32`],`${n} parameters`),t.parameters.length<2)throw Error(`${n} parameters must hold 2 uint32`);let{output:s}=t;for(let[e,t]of[[`inertia`,s.inertia],[`status.iterationCount`,s.status?.iterationCount],[`status.converged`,s.status?.converged]])if(t&&(o(t,e===`inertia`?[`float32`]:[`uint32`],`${n} output.${e}`),t.length!==1))throw Error(`${n} output.${e} must hold one row`);if(o(s.labels,[`uint32`],`${n} output.labels`),s.labels.length!==a)throw Error(`${n} output.labels length must equal positions length`);if(o(s.centers,[`float32x2`],`${n} output.centers`),s.centers.length!==t.k)throw Error(`${n} output.centers length must equal k`);if(s.sizes&&(o(s.sizes,[`uint32`],`${n} output.sizes`),s.sizes.length!==t.k))throw Error(`${n} output.sizes length must equal k`);if(s.squaredDistances&&(o(s.squaredDistances,[`float32`],`${n} output.squaredDistances`),s.squaredDistances.length!==a))throw Error(`${n} output.squaredDistances length must equal positions length`);i(n,h(s),[t.positions,t.parameters])}getCommandNodes(e){let{id:t,props:r}=this,{positions:i,k:o,maximumIterations:l,parameters:p,output:m}=r,{labels:g,centers:_}=m;n(t,e,[i,p,...h(m)]);let v=i.length,y=this.restartCount,b=y>1,x=(n,r,i)=>a(e,`${t}-${n}`,r,i),S=x(`valid-flags`,`uint32`,v),C=x(`row-ids`,`uint32`,v),w=x(`xs`,`float32`,v),T=x(`ys`,`float32`,v),E=m.sizes??x(`sizes`,`uint32`,o),D=b?a(e,`${t}-work-centers`,`float32x2`,o):_,O=b?x(`work-sizes`,`uint32`,o):E,k=m.squaredDistances??(b||m.inertia?x(`squared-distances`,`float32`,v):void 0),A=x(`state`,`uint32`,3),j=x(`seed-cursor`,`uint32`,2),M=Math.min(se,Math.ceil(v/f)),N=Math.ceil(v/M/f)*f,P=Math.min(oe,Math.floor(f/o)),F=x(`partial-counts`,`uint32`,o*M),I=x(`partial-sums-x`,`float32`,o*M),L=x(`partial-sums-y`,`float32`,o*M),R=x(`iteration-distances`,`float32`,v),z=a(e,`${t}-cluster-sums`,`float32x2`,o),B=te(e,{id:`${t}-iteration-dispatch`,operation:u,shapes:[[v],[M*f,f],[o*f,f],[f,f]]}),V=`const K: u32 = ${o}u;
const NOISE: u32 = ${ae}u;

// Mirrors getKMeansRestartSeed: restart 0 uses the seed itself, later restarts a murmur3 mix.
// The seed is the per-frame parameters word 0, so the source is the same for every seed.
fn getRestartSeed(seed: u32, restart: u32) -> u32 {
  if (restart == 0u) {
    return seed;
  }
  var hash = seed + restart * 0x9e3779b9u;
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x85ebca6bu;
  hash = hash ^ (hash >> 13u);
  hash = hash * 0xc2b2ae35u;
  return hash ^ (hash >> 16u);
}

${re}
${ie}

${ne}`,H=[c(e,{id:`${t}-prepare`,operation:u,variant:`prepare`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`validFlags`,view:S,type:`u32`,access:`read_write`},{name:`rowIds`,view:C,type:`u32`,access:`read_write`},{name:`xs`,view:w,type:`f32`,access:`read_write`},{name:`ys`,view:T,type:`f32`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read_write`}],invocationCount:v,declarations:V,body:`if (index == 0u) {
    seedCursor[seedCursorOffset] = 0xffffffffu;
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isFiniteFloat(x) && isFiniteFloat(y);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  rowIds[rowIdsOffset + index] = index;
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);`})],U=this.initialization===`kmeans++`,W=U?x(`min-distance`,`float32`,v):void 0,G=U?x(`keys`,`uint32`,v):void 0,K=U?x(`best-key`,`uint32`,1):void 0,q=U?x(`best-row`,`uint32`,1):void 0,ce=m.inertia??x(`best-inertia`,`float32`,1),J=b?a(e,`${t}-best-centers`,`float32x2`,o):void 0,Y=b?x(`best-state`,`uint32`,3):void 0,X=(t,n,r,a)=>c(e,{id:t,operation:u,variant:n?`assign-final`:`assign`,...n?{}:B.getGate(0),bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`centers`,view:r,type:`f32`,access:`read`},{name:`labels`,view:g,type:`u32`,access:`read_write`},...n?[]:[{name:`state`,view:A,type:`u32`,access:`read`}],...a?[{name:`squaredDistances`,view:a,type:`f32`,access:`read_write`}]:[]],invocationCount:v,declarations:V,body:`${n?``:`// Converged: centers no longer move, so labels are already final.
  if (state[stateOffset] != 0u) {
    return;
  }
  `}let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var best = NOISE;
  var bestDistance = getQuietNaN(index);
  if (isFiniteFloat(x) && isFiniteFloat(y)) {
    // Strict comparison: the lowest center ID wins ties; NaN centers never win.
    for (var center = 0u; center < K; center++) {
      let deltaX = centers[centersOffset + center * 2u] - x;
      let deltaY = centers[centersOffset + center * 2u + 1u] - y;
      let distance = deltaX * deltaX + deltaY * deltaY;
      if (!isNanBits(distance) && (best == NOISE || distance < bestDistance)) {
        best = center;
        bestDistance = distance;
      }
    }
  }
  labels[labelsOffset + index] = best;
  ${a?`squaredDistances[squaredDistancesOffset + index] = select(getQuietNaN(index), bestDistance, best != NOISE);`:``}`}),le=t=>{if(this.initialization===`first-valid`){let n=x(`compact-ids`,`uint32`,v),r=x(`valid-count`,`uint32`,1);H.push(...new s({id:`${t}-valid-rows`,input:C,flags:S,output:n,count:r}).getCommandNodes(e),c(e,{id:`${t}-initial-centers`,operation:u,variant:`initial-centers`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`compactIds`,view:n,type:`u32`,access:`read`},{name:`validCount`,view:r,type:`u32`,access:`read`},{name:`centers`,view:D,type:`f32`,access:`read_write`}],invocationCount:o,declarations:V,body:`let nan = getQuietNaN(index);
    var x = nan;
    var y = nan;
    if (index < validCount[validCountOffset]) {
      let row = compactIds[compactIdsOffset + index];
      x = positions[positionsOffset + row * 2u];
      y = positions[positionsOffset + row * 2u + 1u];
    }
    centers[centersOffset + index * 2u] = x;
    centers[centersOffset + index * 2u + 1u] = y;`}))}else for(let n=0;n<o;n++){let r=`${t}-seed-${n}`;H.push(c(e,{id:`${r}-begin`,operation:u,variant:`seed-begin`,bindings:[{name:`bestKey`,view:K,type:`u32`,access:`read_write`},{name:`bestRow`,view:q,type:`u32`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read_write`}],invocationCount:1,body:`bestKey[bestKeyOffset] = ${d};
    bestRow[bestRowOffset] = ${d};
    seedCursor[seedCursorOffset + 1u] = seedCursor[seedCursorOffset + 1u] + 1u;`}),c(e,{id:`${r}-key`,operation:u,variant:`seed-key`,bindings:[{name:`validFlags`,view:S,type:`u32`,access:`read`},{name:`minDistance`,view:W,type:`f32`,access:`read`},{name:`keys`,view:G,type:`u32`,access:`read_write`},{name:`bestKey`,view:K,type:`atomic<u32>`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read`},{name:`parameters`,view:p,type:`u32`,access:`read`}],invocationCount:v,workgroupSize:f,guardIndex:!1,declarations:`${V}
  var<workgroup> groupBestKey: atomic<u32>;

  fn hashRow(row: u32, roundSeed: u32) -> u32 {
    var h = row ^ roundSeed;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
  }`,body:`if (localInvocationIndex == 0u) {
      atomicStore(&groupBestKey, 0xffffffffu);
    }
    workgroupBarrier();
    let round = seedCursor[seedCursorOffset + 1u];
    // Same value as the former per-round constant: imul(restartSeed, 0x9e3779b1) + imul(round, 0x85ebca6b).
    let roundSeed = getRestartSeed(parameters[parametersOffset], seedCursor[seedCursorOffset]) * 0x9e3779b1u + round * 0x85ebca6bu;
    var key = 0xffffffffu;
    if (index < INVOCATION_COUNT && validFlags[validFlagsOffset + index] != 0u) {
      // Exponential race: the minimum of -ln(u) / w samples a row with probability proportional to w.
      let weight = select(minDistance[minDistanceOffset + index], 1.0, round == 0u);
      let uniformValue = (f32(hashRow(index, roundSeed) >> 8u) + 0.5) / 16777216.0;
      key = select(0x7f800000u, bitcast<u32>(-log(uniformValue) / weight), weight > 0.0);
      atomicMin(&groupBestKey, key);
      keys[keysOffset + index] = key;
    } else if (index < INVOCATION_COUNT) {
      keys[keysOffset + index] = key;
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      atomicMin(&bestKey[bestKeyOffset], atomicLoad(&groupBestKey));
    }`}),c(e,{id:`${r}-row`,operation:u,variant:`seed-row`,bindings:[{name:`keys`,view:G,type:`u32`,access:`read`},{name:`bestKey`,view:K,type:`u32`,access:`read`},{name:`bestRow`,view:q,type:`atomic<u32>`,access:`read_write`}],invocationCount:v,body:`let best = bestKey[bestKeyOffset];
    if (best != 0xffffffffu && keys[keysOffset + index] == best) {
      atomicMin(&bestRow[bestRowOffset], index);
    }`}),c(e,{id:`${r}-update`,operation:u,variant:`seed-update`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`validFlags`,view:S,type:`u32`,access:`read`},{name:`bestRow`,view:q,type:`u32`,access:`read`},{name:`minDistance`,view:W,type:`f32`,access:`read_write`},{name:`centers`,view:D,type:`f32`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read`}],invocationCount:v,declarations:V,body:`let round = seedCursor[seedCursorOffset + 1u];
    let row = bestRow[bestRowOffset];
    let hasCenter = row != 0xffffffffu;
    var centerX = getQuietNaN(index);
    var centerY = centerX;
    if (hasCenter) {
      centerX = positions[positionsOffset + row * 2u];
      centerY = positions[positionsOffset + row * 2u + 1u];
    }
    if (index == 0u) {
      centers[centersOffset + round * 2u] = centerX;
      centers[centersOffset + round * 2u + 1u] = centerY;
    }
    if (hasCenter && validFlags[validFlagsOffset + index] != 0u) {
      let deltaX = positions[positionsOffset + index * 2u] - centerX;
      let deltaY = positions[positionsOffset + index * 2u + 1u] - centerY;
      let distance = deltaX * deltaX + deltaY * deltaY;
      minDistance[minDistanceOffset + index] =
        select(min(minDistance[minDistanceOffset + index], distance), distance, round == 0u);
    }`}))}},ue=t=>{for(let n=0;n<l;n++){let r=`${t}-iteration-${n}`;H.push(X(`${r}-assign`,!1,D,R),c(e,{id:`${r}-partial`,operation:u,variant:`partial`,bindings:[{name:`labels`,view:g,type:`u32`,access:`read`},{name:`xs`,view:w,type:`f32`,access:`read`},{name:`ys`,view:T,type:`f32`,access:`read`},{name:`state`,view:A,type:`u32`,access:`read`},{name:`partialCounts`,view:F,type:`u32`,access:`read_write`},{name:`partialSumsX`,view:I,type:`f32`,access:`read_write`},{name:`partialSumsY`,view:L,type:`f32`,access:`read_write`}],workgroupSize:f,invocationCount:M*f,guardIndex:!1,...B.getGate(1),declarations:`${V}
  const ROWS: u32 = ${v}u;
  const CHUNK_SIZE: u32 = ${N}u;
  const CHUNK_COUNT: u32 = ${M}u;
  const TILE_COUNT: u32 = ${N/f}u;
  const SLICES: u32 = ${P}u;
  const LANES: u32 = ${P*o}u;
  var<workgroup> tileLabels: array<u32, ${f}>;
  var<workgroup> tileXs: array<f32, ${f}>;
  var<workgroup> tileYs: array<f32, ${f}>;
  var<workgroup> laneCounts: array<u32, ${f}>;
  var<workgroup> laneSumsX: array<f32, ${f}>;
  var<workgroup> laneSumsY: array<f32, ${f}>;`,body:`let chunk = index / ${f}u;
    let converged = state[stateOffset] != 0u;
    let begin = chunk * CHUNK_SIZE;
    let end = min(begin + CHUNK_SIZE, ROWS);
    let lane = localInvocationIndex;
    let cluster = lane % K;
    let slice = lane / K;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let row = begin + tile * ${f}u + lane;
      var label = NOISE;
      var x = 0.0;
      var y = 0.0;
      if (!converged && row < end) {
        label = labels[labelsOffset + row];
        x = xs[xsOffset + row];
        y = ys[ysOffset + row];
      }
      tileLabels[lane] = label;
      tileXs[lane] = x;
      tileYs[lane] = y;
      workgroupBarrier();
      if (lane < LANES) {
        // Lane (slice, cluster) owns tile entries slice, slice + SLICES, ...: a fixed order.
        for (var entry = slice; entry < ${f}u; entry += SLICES) {
          if (tileLabels[entry] == cluster) {
            count++;
            sumX += tileXs[entry];
            sumY += tileYs[entry];
          }
        }
      }
      workgroupBarrier();
    }
    laneCounts[lane] = count;
    laneSumsX[lane] = sumX;
    laneSumsY[lane] = sumY;
    workgroupBarrier();
    if (lane < K) {
      var chunkCountTotal = 0u;
      var chunkSumX = 0.0;
      var chunkSumY = 0.0;
      for (var sliceIndex = 0u; sliceIndex < SLICES; sliceIndex++) {
        chunkCountTotal += laneCounts[sliceIndex * K + lane];
        chunkSumX += laneSumsX[sliceIndex * K + lane];
        chunkSumY += laneSumsY[sliceIndex * K + lane];
      }
      let slot = lane * CHUNK_COUNT + chunk;
      partialCounts[partialCountsOffset + slot] = chunkCountTotal;
      partialSumsX[partialSumsXOffset + slot] = chunkSumX;
      partialSumsY[partialSumsYOffset + slot] = chunkSumY;
    }`}),c(e,{id:`${r}-update`,operation:u,variant:`update`,bindings:[{name:`partialCounts`,view:F,type:`u32`,access:`read`},{name:`partialSumsX`,view:I,type:`f32`,access:`read`},{name:`partialSumsY`,view:L,type:`f32`,access:`read`},{name:`sizes`,view:O,type:`u32`,access:`read_write`},{name:`clusterSums`,view:z,type:`f32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read`}],workgroupSize:f,invocationCount:o*f,guardIndex:!1,...B.getGate(2),declarations:`const CHUNK_COUNT: u32 = ${M}u;
  var<workgroup> reducedCounts: array<u32, ${f}>;
  var<workgroup> reducedSumsX: array<f32, ${f}>;
  var<workgroup> reducedSumsY: array<f32, ${f}>;`,body:`let cluster = index / ${f}u;
    let lane = localInvocationIndex;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${f}u) {
      let slot = cluster * CHUNK_COUNT + chunk;
      count += partialCounts[partialCountsOffset + slot];
      sumX += partialSumsX[partialSumsXOffset + slot];
      sumY += partialSumsY[partialSumsYOffset + slot];
    }
    reducedCounts[lane] = count;
    reducedSumsX[lane] = sumX;
    reducedSumsY[lane] = sumY;
    workgroupBarrier();
    for (var stride = ${f/2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        reducedCounts[lane] += reducedCounts[lane + stride];
        reducedSumsX[lane] += reducedSumsX[lane + stride];
        reducedSumsY[lane] += reducedSumsY[lane + stride];
      }
      workgroupBarrier();
    }
    // Converged: centers no longer move; the partials are stale and the final sizes come later.
    if (lane == 0u && state[stateOffset] == 0u) {
      sizes[sizesOffset + cluster] = reducedCounts[0];
      clusterSums[clusterSumsOffset + cluster * 2u] = reducedSumsX[0];
      clusterSums[clusterSumsOffset + cluster * 2u + 1u] = reducedSumsY[0];
    }`}),c(e,{id:`${r}-relocate`,operation:u,variant:`relocate`,bindings:[{name:`positions`,view:i,type:`f32`,access:`read`},{name:`labels`,view:g,type:`u32`,access:`read`},{name:`distances`,view:R,type:`f32`,access:`read`},{name:`clusterSums`,view:z,type:`f32`,access:`read`},{name:`sizes`,view:O,type:`u32`,access:`read_write`},{name:`centers`,view:D,type:`f32`,access:`read_write`},{name:`state`,view:A,type:`atomic<u32>`,access:`read_write`}],workgroupSize:f,invocationCount:f,guardIndex:!1,...B.getGate(3),declarations:`${V}
  const ROWS: u32 = ${v}u;
  var<workgroup> activeFlag: u32;
  var<workgroup> stopFlag: u32;
  var<workgroup> emptyCount: u32;
  var<workgroup> pickDistance: f32;
  var<workgroup> pickRow: u32;
  var<workgroup> clusterCounts: array<u32, ${f}>;
  var<workgroup> clusterSumsX: array<f32, ${f}>;
  var<workgroup> clusterSumsY: array<f32, ${f}>;
  var<workgroup> emptyClusters: array<u32, ${f}>;
  var<workgroup> laneDistances: array<f32, ${f}>;
  var<workgroup> laneRows: array<u32, ${f}>;`,body:`let lane = localInvocationIndex;
  if (lane == 0u) {
    activeFlag = select(0u, 1u, atomicLoad(&state[stateOffset]) == 0u);
  }
  // Converged: the partials are stale and the centers no longer move.
  if (workgroupUniformLoad(&activeFlag) == 0u) {
    return;
  }
  if (lane < K) {
    clusterCounts[lane] = sizes[sizesOffset + lane];
    clusterSumsX[lane] = clusterSums[clusterSumsOffset + lane * 2u];
    clusterSumsY[lane] = clusterSums[clusterSumsOffset + lane * 2u + 1u];
  }
  workgroupBarrier();
  if (lane == 0u) {
    // Clusters whose center could not be initialized (NaN) are not empty clusters to relocate.
    var count = 0u;
    for (var cluster = 0u; cluster < K; cluster++) {
      let centerX = centers[centersOffset + cluster * 2u];
      if (clusterCounts[cluster] == 0u && isFiniteFloat(centerX)) {
        emptyClusters[count] = cluster;
        count++;
      }
    }
    emptyCount = count;
    stopFlag = 0u;
    pickDistance = 0.0;
    pickRow = 0u;
  }
  let pickLimit = workgroupUniformLoad(&emptyCount);
  for (var pickIndex = 0u; pickIndex < pickLimit; pickIndex++) {
    // scikit-learn returns without relocating when every point sits on its center.
    if (workgroupUniformLoad(&stopFlag) != 0u) {
      break;
    }
    let previousDistance = pickDistance;
    let previousRow = pickRow;
    var bestDistance = -1.0;
    var bestRow = NOISE;
    for (var row = lane; row < ROWS; row += ${f}u) {
      let distance = distances[distancesOffset + row];
      if (isFiniteFloat(distance) && labels[labelsOffset + row] != NOISE) {
        // Rank (distance descending, row ascending) strictly after the previous winner.
        let ranked = pickIndex == 0u || distance < previousDistance ||
          (distance == previousDistance && row > previousRow);
        if (ranked && distance > bestDistance) {
          bestDistance = distance;
          bestRow = row;
        }
      }
    }
    laneDistances[lane] = bestDistance;
    laneRows[lane] = bestRow;
    workgroupBarrier();
    for (var stride = ${f/2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherDistance = laneDistances[lane + stride];
        let otherRow = laneRows[lane + stride];
        if (otherDistance > laneDistances[lane] ||
          (otherDistance == laneDistances[lane] && otherRow < laneRows[lane])) {
          laneDistances[lane] = otherDistance;
          laneRows[lane] = otherRow;
        }
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let row = laneRows[0];
      let distance = laneDistances[0];
      if (row == NOISE || (pickIndex == 0u && distance == 0.0)) {
        stopFlag = 1u;
      } else {
        pickDistance = distance;
        pickRow = row;
        let emptyCluster = emptyClusters[pickIndex];
        let donor = labels[labelsOffset + row];
        let x = positions[positionsOffset + row * 2u];
        let y = positions[positionsOffset + row * 2u + 1u];
        clusterSumsX[donor] -= x;
        clusterSumsY[donor] -= y;
        clusterCounts[donor] -= 1u;
        clusterSumsX[emptyCluster] = x;
        clusterSumsY[emptyCluster] = y;
        clusterCounts[emptyCluster] = 1u;
      }
    }
    workgroupBarrier();
  }
  // A cluster still empty (every point on its center, or its only member was taken) moves to the
  // mean of the largest cluster, lowest ID on ties, like scikit-learn's \`_average_centers\`.
  var largest = 0u;
  for (var cluster = 1u; cluster < K; cluster++) {
    if (clusterCounts[cluster] > clusterCounts[largest]) {
      largest = cluster;
    }
  }
  if (lane < K) {
    let size = clusterCounts[lane];
    let previousX = centers[centersOffset + lane * 2u];
    let previousY = centers[centersOffset + lane * 2u + 1u];
    var nextX = previousX;
    var nextY = previousY;
    if (size > 0u) {
      nextX = clusterSumsX[lane] / f32(size);
      nextY = clusterSumsY[lane] / f32(size);
    } else if (isFiniteFloat(previousX) && clusterCounts[largest] > 0u) {
      nextX = clusterSumsX[largest] / f32(clusterCounts[largest]);
      nextY = clusterSumsY[largest] / f32(clusterCounts[largest]);
    }
    sizes[sizesOffset + lane] = size;
    let deltaX = nextX - previousX;
    let deltaY = nextY - previousY;
    let shift = sqrt(deltaX * deltaX + deltaY * deltaY);
    // A never-initialized (NaN) center does not move. Any other non-finite shift (overflowed
    // coordinate sums or squares, or an infinite center) counts as +infinity, so the iteration
    // cannot report convergence.
    if (isFiniteFloat(shift)) {
      atomicMax(&state[stateOffset + 2u], bitcast<u32>(shift));
    } else if (!isNanBits(previousX)) {
      atomicMax(&state[stateOffset + 2u], 0x7f800000u);
    }
    centers[centersOffset + lane * 2u] = nextX;
    centers[centersOffset + lane * 2u + 1u] = nextY;
  }`}),c(e,{id:`${r}-converge`,operation:u,variant:`converge`,bindings:[{name:`state`,view:A,type:`u32`,access:`read_write`},{name:`parameters`,view:p,type:`u32`,access:`read`},B.binding],invocationCount:1,body:`if (state[stateOffset] != 0u) {
      return;
    }
    state[stateOffset + 1u] = state[stateOffset + 1u] + 1u;
    // The tolerance is the per-frame parameters word 1 (float32 bits).
    if (bitcast<f32>(state[stateOffset + 2u]) <= bitcast<f32>(parameters[parametersOffset + 1u])) {
      state[stateOffset] = 1u;
      ${B.disarmWGSL}
    }
    state[stateOffset + 2u] = 0u;`}))}},Z=x(`score-partial-sums`,`float32`,M),Q=x(`score-partial-counts`,`uint32`,M),$=t=>{H.push(c(e,{id:`${t}-partial`,operation:u,variant:`score-partial`,bindings:[{name:`squaredDistances`,view:k,type:`f32`,access:`read`},{name:`partialSums`,view:Z,type:`f32`,access:`read_write`},{name:`partialCounts`,view:Q,type:`u32`,access:`read_write`}],workgroupSize:f,invocationCount:M*f,guardIndex:!1,declarations:`${V}
const ROWS: u32 = ${v}u;
const CHUNK_SIZE: u32 = ${N}u;
var<workgroup> laneSums: array<f32, ${f}>;
var<workgroup> laneCounts: array<u32, ${f}>;`,body:`let lane = localInvocationIndex;
  let chunk = index / ${f}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  var sum = 0.0;
  var compensation = 0.0;
  var count = 0u;
  for (var row = begin + lane; row < end; row += ${f}u) {
    let value = squaredDistances[squaredDistancesOffset + row];
    if (isFiniteFloat(value)) {
      let corrected = value - compensation;
      let next = sum + corrected;
      compensation = (next - sum) - corrected;
      sum = next;
      count++;
    }
  }
  laneSums[lane] = sum;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${f/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    partialSums[partialSumsOffset + chunk] = laneSums[0];
    partialCounts[partialCountsOffset + chunk] = laneCounts[0];
  }`}),c(e,{id:t,operation:u,variant:b?`restart-score`:`inertia`,bindings:[{name:`partialSums`,view:Z,type:`f32`,access:`read`},{name:`partialCounts`,view:Q,type:`u32`,access:`read`},{name:`inertia`,view:ce,type:`f32`,access:`read_write`},...b?[{name:`centers`,view:D,type:`f32`,access:`read`},{name:`bestCenters`,view:J,type:`f32`,access:`read_write`},{name:`state`,view:A,type:`u32`,access:`read`},{name:`bestState`,view:Y,type:`u32`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read`}]:[]],workgroupSize:f,invocationCount:f,guardIndex:!1,declarations:`${V}
const CHUNK_COUNT: u32 = ${M}u;
var<workgroup> laneSums: array<f32, ${f}>;
var<workgroup> laneCounts: array<u32, ${f}>;
var<workgroup> improves: u32;`,body:`let lane = localInvocationIndex;
  var sum = 0.0;
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${f}u) {
    sum += partialSums[partialSumsOffset + chunk];
    count += partialCounts[partialCountsOffset + chunk];
  }
  laneSums[lane] = sum;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${f/2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let total = select(getQuietNaN(0u), laneSums[0], laneCounts[0] > 0u);${b?`
    let previous = inertia[inertiaOffset];
    let firstRestart = seedCursor[seedCursorOffset] == 0u;
    // Strictly lower wins; a finite score also replaces a NaN one.
    let better = firstRestart || total < previous || (!isFiniteFloat(previous) && isFiniteFloat(total));
    improves = select(0u, 1u, better);
    if (better) {
      inertia[inertiaOffset] = total;
      bestState[bestStateOffset] = state[stateOffset];
      bestState[bestStateOffset + 1u] = state[stateOffset + 1u];
    }`:`
    inertia[inertiaOffset] = total;`}
  }${b?`
  if (workgroupUniformLoad(&improves) != 0u) {
    for (var slot = lane; slot < ${2*o}u; slot += ${f}u) {
      bestCenters[bestCentersOffset + slot] = centers[centersOffset + slot];
    }
  }`:``}`}))};for(let n=0;n<y;n++){let r=b?`${t}-restart-${n}`:t;H.push(c(e,{id:`${r}-begin`,operation:u,variant:`restart-begin`,bindings:[{name:`state`,view:A,type:`u32`,access:`read_write`},{name:`seedCursor`,view:j,type:`u32`,access:`read_write`},B.binding],invocationCount:1,body:`state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
  seedCursor[seedCursorOffset] = seedCursor[seedCursorOffset] + 1u;
  seedCursor[seedCursorOffset + 1u] = 0xffffffffu;
  ${B.armWGSL}`})),le(r),ue(r),b&&(H.push(X(`${r}-score-assign`,!0,D,k)),$(`${r}-score`))}b&&H.push(c(e,{id:`${t}-publish-centers`,operation:u,variant:`publish-centers`,bindings:[{name:`bestCenters`,view:J,type:`f32`,access:`read`},{name:`centers`,view:_,type:`f32`,access:`read_write`}],invocationCount:2*o,body:`centers[centersOffset + index] = bestCenters[bestCentersOffset + index];`})),H.push(X(`${t}-final-assign`,!0,_,k),...new ee({id:`${t}-final-sizes`,keys:g,output:E}).getCommandNodes(e)),m.inertia&&!b&&$(`${t}-inertia`);for(let[n,r,i]of[[`iteration-count`,m.status?.iterationCount,1],[`converged`,m.status?.converged,0]])r&&H.push(c(e,{id:`${t}-publish-${n}`,operation:u,variant:`publish-state-word`,bindings:[{name:`state`,view:b?Y:A,type:`u32`,access:`read`},{name:`stateWord`,view:r,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const WORD: u32 = ${i}u;`,body:`stateWord[stateWordOffset] = state[stateOffset + WORD];`}));return H}};function h(e){return[e.labels,e.centers,e.sizes,e.squaredDistances,e.inertia,e.status?.iterationCount,e.status?.converged]}export{p as n,m as t};