import{E as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,d as n,h as r,w as i,y as a}from"./resources-CwvhEOBe.js";import{t as o}from"./gpu-scan-BuGhTLDQ.js";import{t as s}from"./gpu-sort-Bl3Dztdb.js";import{t as c}from"./gpu-reduction-BD8l40jM.js";import{n as l,s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as d}from"./wgsl-common-BNRVzgd8.js";import{n as f}from"./sorted-segment-offsets-HVRgs63A.js";import{a as p}from"./sorted-segment-sums-DwGkmbx2.js";var m=`GPUGroupConvexHull`,h=2**24-1,g=29,_=30,v=8192,y=256,b=8,x=65536,S=256,C=16,w=256,T=2**g,E=2,D=class{id;props;spatialContext;groupCount;prefilterLevels;extremePointFilter;constructor(t){this.id=t.id??`group-convex-hull`,this.props=t,this.spatialContext=e(this.id,t.spatialContext);let{id:i}=this,{output:o}=t;a(t.positions,[`float32x2`],`${i} positions`);let s=t.positions.length;if(s<1)throw Error(`${i} needs at least one row`);if(s>h)throw Error(`${i} supports fewer than 2^24 rows`);if(a(t.labels,[`uint32`],`${i} labels`),t.labels.length!==s)throw Error(`${i} labels length must equal positions length`);if(this.groupCount=t.groupCount,n(i,`groupCount`,this.groupCount,1,h),t.noiseLabel!==void 0&&n(i,`noiseLabel`,t.noiseLabel,0),n(i,`maximumVerticesPerGroup`,t.maximumVerticesPerGroup,3),n(i,`vertexCapacity`,t.vertexCapacity),this.prefilterLevels=t.prefilterLevels??(s>=v?E:0),n(i,`prefilterLevels`,this.prefilterLevels,0,4),this.extremePointFilter=t.extremePointFilter??this.groupCount<=x,this.extremePointFilter&&this.groupCount>x)throw Error(`${i} extremePointFilter supports at most ${x} groups`);let c=this.groupCount,l=[[`vertexIndices`,o.vertexIndices,[`uint32`],t.vertexCapacity],[`vertexPositions`,o.vertexPositions,[`float32x2`],t.vertexCapacity],[`offsets`,o.offsets,[`uint32`],c+1],[`counts`,o.counts,[`uint32`],c],[`sizes`,o.sizes,[`uint32`],c],[`overflow`,o.status.overflow,[`uint32`],1]];for(let[e,t,n,r]of l)if(t&&(a(t,n,`${i} output.${e}`),t.length<r))throw Error(`${i} output.${e} must hold at least ${r} rows`);r(i,l.map(([,e])=>e),[t.positions,t.labels])}getCommandNodes(e){let{props:n,id:r,groupCount:a}=this,{output:h,positions:v,labels:b,maximumVerticesPerGroup:x,vertexCapacity:S}=n,{status:C,...w}=h;t(r,e,[v,b,...Object.values(w),C.overflow]);let T=v.length,E=(t,n)=>i(e,`${r}-${t}`,`float32`,n),D=(t,n)=>i(e,`${r}-${t}`,`uint32`,n),O=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),A=(e,t,n=`f32`)=>({name:e,view:t,type:n,access:`read_write`}),j=(t,n,i,o,s=``)=>u(e,{id:`${r}-${t}`,operation:m,variant:t,bindings:i,invocationCount:n,declarations:`const GROUP_COUNT: u32 = ${a}u;\n${s}`,body:o}),M=[],N=D(`group-keys`,T),P=D(`valid-flags`,T),F=E(`xs`,T),I=E(`ys`,T),L=D(`row-indices`,T),ee=n.noiseLabel!==void 0&&n.noiseLabel<a?`valid = valid && label != ${n.noiseLabel}u;`:``;M.push(j(`keys`,T,[O(`positions`,v,`f32`),O(`labels`,b,`u32`),A(`groupKeys`,N,`u32`),A(`validFlags`,P,`u32`),A(`xs`,F),A(`ys`,I),A(`rowIndices`,L,`u32`)],`let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${ee}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);
  rowIndices[rowIndicesOffset + index] = index;`));let te=E(`extent-x`,2),ne=E(`extent-y`,2),re=E(`lattice`,3);M.push(...new c({id:`${r}-extent-x`,input:F,mask:P,output:te,operation:`extent`}).getCommandNodes(e),...new c({id:`${r}-extent-y`,input:I,mask:P,output:ne,operation:`extent`}).getCommandNodes(e),j(`lattice`,1,[O(`extentX`,te,`f32`),O(`extentY`,ne,`f32`),A(`lattice`,re)],`let width = extentX[extentXOffset + 1u] - extentX[extentXOffset];
  let height = extentY[extentYOffset + 1u] - extentY[extentYOffset];
  let extent = max(width, height);
  var exponent = i32((bitcast<u32>(extent) >> 23u) & 255u) - 127;
  exponent = max(exponent, -99);
  let scale = bitcast<f32>(u32(127 + ${g-1} - exponent) << 23u);
  lattice[latticeOffset] = extentX[extentXOffset];
  lattice[latticeOffset + 1u] = extentY[extentYOffset];
  lattice[latticeOffset + 2u] = select(scale, 1.0, !(extent > 0.0));`));let R=D(`lattice-x`,T),z=D(`lattice-y`,T);M.push(j(`quantize`,T,[O(`xs`,F,`f32`),O(`ys`,I,`f32`),O(`validFlags`,P,`u32`),O(`lattice`,re,`f32`),A(`latticeX`,R,`u32`),A(`latticeY`,z,`u32`)],`let isValid = validFlags[validFlagsOffset + index] != 0u;
  let scale = lattice[latticeOffset + 2u];
  let dx = xs[xsOffset + index] - lattice[latticeOffset];
  let dy = ys[ysOffset + index] - lattice[latticeOffset + 1u];
  latticeX[latticeXOffset + index] = select(0u, u32(floor(dx * scale + 0.5)), isValid);
  latticeY[latticeYOffset + index] = select(0u, u32(floor(dy * scale + 0.5)), isValid);`));let B=N;this.extremePointFilter&&(B=D(`filtered-group-keys`,T),M.push(...this.getExtremePointFilterNodes(e,{rows:T,groupKeys:N,latticeX:R,latticeY:z,filteredGroupKeys:B})));let ie=D(`sorted-y-keys`,T),V=D(`permutation-y`,T),ae=D(`keys-x`,T),oe=D(`sorted-x-keys`,T),H=D(`permutation-x`,T),U=D(`keys-group`,T),W=D(`sorted-group-keys`,T),G=D(`permutation`,T);M.push(...new s({id:`${r}-sort-y`,keys:z,values:L,outputKeys:ie,outputValues:V,keyBits:_}).getCommandNodes(e),j(`gather-x`,T,[O(`permutation`,V,`u32`),O(`latticeX`,R,`u32`),A(`keys`,ae,`u32`)],`keys[keysOffset + index] = latticeX[latticeXOffset + permutation[permutationOffset + index]];`),...new s({id:`${r}-sort-x`,keys:ae,values:V,outputKeys:oe,outputValues:H,keyBits:_}).getCommandNodes(e),j(`gather-group`,T,[O(`permutation`,H,`u32`),O(`groupKeys`,B,`u32`),A(`keys`,U,`u32`)],`keys[keysOffset + index] = groupKeys[groupKeysOffset + permutation[permutationOffset + index]];`),...new s({id:`${r}-sort-group`,keys:U,values:H,outputKeys:W,outputValues:G,keyBits:p(a)}).getCommandNodes(e));let K=D(`segment-offsets`,a+1);M.push(...f(e,{id:r,operation:m,groupCount:a,sortedGroupKeys:W,segmentOffsets:K}));let q=D(`sorted-lattice-x`,T),J=D(`sorted-lattice-y`,T);M.push(j(`gather-lattice`,T,[O(`permutation`,G,`u32`),O(`latticeX`,R,`u32`),O(`latticeY`,z,`u32`),A(`sortedLatticeX`,q,`u32`),A(`sortedLatticeY`,J,`u32`)],`let row = permutation[permutationOffset + index];
  sortedLatticeX[sortedLatticeXOffset + index] = latticeX[latticeXOffset + row];
  sortedLatticeY[sortedLatticeYOffset + index] = latticeY[latticeYOffset + row];`));let Y=K,X=q,Z=J,Q=G;for(let t=0;t<this.prefilterLevels;t++){let n=`prefilter${t}`,i=Math.ceil(T/y),s=D(`${n}-survivor-flags`,T),c=D(`${n}-survivor-offsets`,T),l=D(`${n}-scratch`,2*T+2),u=D(`${n}-compact-offsets`,a+1),d=t===0?[O(`groupKeys`,W,`u32`)]:[O(`groupOffsets`,Y,`u32`)],f=t===0?`${T}u`:`groupOffsets[groupOffsetsOffset + GROUP_COUNT]`,p=t===0?`let key = groupKeys[groupKeysOffset + runBegin];
    // The excluded-row run is last in sorted order: skip it whole instead of walking it.
    var runEnd = chunkEnd;
    if (key < GROUP_COUNT) {
      runEnd = runBegin + 1u;
      while (runEnd < chunkEnd && groupKeys[groupKeysOffset + runEnd] == key) {
        runEnd++;
      }
    }`:`let key = findGroup(runBegin);
    let runEnd = min(chunkEnd, groupOffsets[groupOffsetsOffset + key + 1u]);`;M.push(j(n,i,[...d,O(`latticeX`,X,`u32`),O(`latticeY`,Z,`u32`),A(`flags`,s,`u32`),A(`scratch`,l,`u32`)],`let chunkStart = index * ${y}u;
  let slotEnd = min(chunkStart + ${y}u, ${T}u);
  for (var row = chunkStart; row < slotEnd; row++) {
    flags[flagsOffset + row] = 0u;
  }
  let chunkEnd = min(slotEnd, ${f});
  var runBegin = chunkStart;
  while (runBegin < chunkEnd) {
    ${p}
    if (key < GROUP_COUNT) {
      let begin = runBegin;
      let base = 2u * begin;
      var lastRep = runEnd - 1u;
      while (!isRepresentative(lastRep, begin)) {
        lastRep = lastRep - 1u;
      }
      if (lastRep == begin) {
        flags[flagsOffset + begin] = 1u;
      } else {
        var count = 0u;
        for (var row = begin; row <= lastRep; row++) {
          if (!isRepresentative(row, begin)) {
            continue;
          }
          while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = row;
          count = count + 1u;
        }
        let lowerCount = count + 1u;
        var back = lastRep;
        while (back > begin) {
          back = back - 1u;
          if (!isRepresentative(back, begin)) {
            continue;
          }
          while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], back) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = back;
          count = count + 1u;
        }
        for (var vertex = 0u; vertex + 1u < count; vertex++) {
          flags[flagsOffset + scratch[scratchOffset + base + vertex]] = 1u;
        }
      }
    }
    runBegin = runEnd;
  }`,t===0?k:`${k}
// Largest group whose offset is at or before the row (skips empty groups).
fn findGroup(row: u32) -> u32 {
  var low = 0u;
  var high = GROUP_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (groupOffsets[groupOffsetsOffset + middle] <= row) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}`),...new o({id:`${r}-${n}-scan`,input:s,output:c,mode:`exclusive`}).getCommandNodes(e),j(`${n}-offsets`,a+1,[O(`segmentOffsets`,Y,`u32`),O(`flags`,s,`u32`),O(`survivorOffsets`,c,`u32`),A(`compactOffsets`,u,`u32`)],`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let total = survivorOffsets[survivorOffsetsOffset + ${T-1}u] + flags[flagsOffset + ${T-1}u];
  compactOffsets[compactOffsetsOffset + index] = select(survivorOffsets[survivorOffsetsOffset + min(begin, ${T-1}u)], total, begin >= ${T}u);`));let m=D(`${n}-lattice-x`,T),h=D(`${n}-lattice-y`,T),g=D(`${n}-permutation`,T);M.push(j(`${n}-compact`,T,[O(`flags`,s,`u32`),O(`survivorOffsets`,c,`u32`),O(`permutation`,Q,`u32`),O(`latticeX`,X,`u32`),O(`latticeY`,Z,`u32`),A(`compactLatticeX`,m,`u32`),A(`compactLatticeY`,h,`u32`),A(`compactPermutation`,g,`u32`)],`if (flags[flagsOffset + index] != 0u) {
    let slot = survivorOffsets[survivorOffsetsOffset + index];
    compactLatticeX[compactLatticeXOffset + slot] = latticeX[latticeXOffset + index];
    compactLatticeY[compactLatticeYOffset + slot] = latticeY[latticeYOffset + index];
    compactPermutation[compactPermutationOffset + slot] = permutation[permutationOffset + index];
  }`)),X=m,Z=h,Q=g,Y=u}let se=D(`chain-scratch`,T+a),ce=h.sizes??D(`true-sizes`,a);M.push(j(`chain`,a,[O(`segmentOffsets`,Y,`u32`),O(`latticeX`,X,`u32`),O(`latticeY`,Z,`u32`),A(`scratch`,se,`u32`),A(`sizes`,ce,`u32`)],`let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  let base = begin + index;
  if (end == begin) {
    sizes[sizesOffset + index] = 0u;
    return;
  }
  var lastRep = end - 1u;
  while (!isRepresentative(lastRep, begin)) {
    lastRep = lastRep - 1u;
  }
  if (lastRep == begin) {
    scratch[scratchOffset + base] = begin;
    sizes[sizesOffset + index] = 1u;
    return;
  }
  var count = 0u;
  for (var row = begin; row <= lastRep; row++) {
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  let lowerCount = count + 1u;
  var row = lastRep;
  while (row > begin) {
    row = row - 1u;
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  sizes[sizesOffset + index] = count - 1u;`,k));let $=D(`eligible-sizes`,a);return M.push(l(e,{id:`${r}-zero-overflow`,operation:m,view:h.status.overflow,type:`u32`,value:`0u`,componentCount:1}),j(`eligible`,a,[O(`sizes`,ce,`u32`),A(`eligible`,$,`u32`),A(`overflow`,h.status.overflow,`atomic<u32>`)],`let size = sizes[sizesOffset + index];
  let isEligible = size <= ${x}u;
  eligible[eligibleOffset + index] = select(0u, size, isEligible);
  if (!isEligible) {
    atomicOr(&overflow[overflowOffset], 1u);
  }`),...new o({id:`${r}-offset-scan`,input:$,output:h.offsets,mode:`exclusive`}).getCommandNodes(e),j(`offset-total`,1,[O(`eligible`,$,`u32`),A(`offsets`,h.offsets,`u32`)],`offsets[offsetsOffset + ${a}u] =
    offsets[offsetsOffset + ${a-1}u] + eligible[eligibleOffset + ${a-1}u];`),l(e,{id:`${r}-fill-indices`,operation:m,view:h.vertexIndices,type:`u32`,value:`0xffffffffu`,componentCount:S}),j(`emit`,a,[O(`segmentOffsets`,Y,`u32`),O(`offsets`,h.offsets,`u32`),O(`eligible`,$,`u32`),O(`scratch`,se,`u32`),O(`permutation`,Q,`u32`),A(`vertexIndices`,h.vertexIndices,`u32`),A(`counts`,h.counts,`u32`),A(`overflow`,h.status.overflow,`atomic<u32>`)],`let size = eligible[eligibleOffset + index];
  let offset = offsets[offsetsOffset + index];
  let base = segmentOffsets[segmentOffsetsOffset + index] + index;
  var emitted = 0u;
  if (size > 0u) {
    if (offset + size > ${S}u) {
      atomicOr(&overflow[overflowOffset], 2u);
    } else {
      emitted = size;
      for (var vertex = 0u; vertex < size; vertex++) {
        vertexIndices[vertexIndicesOffset + offset + vertex] = permutation[permutationOffset + scratch[scratchOffset + base + vertex]];
      }
    }
  }
  counts[countsOffset + index] = emitted;`)),h.vertexPositions&&M.push(j(`vertex-positions`,S,[O(`vertexIndices`,h.vertexIndices,`u32`),O(`positions`,v,`f32`),A(`vertexPositions`,h.vertexPositions)],`let row = vertexIndices[vertexIndicesOffset + index];
  let isEmpty = row == 0xffffffffu;
  let nan = getQuietNaN(index);
  vertexPositions[vertexPositionsOffset + index * 2u] = select(positions[positionsOffset + row * 2u], nan, isEmpty);
  vertexPositions[vertexPositionsOffset + index * 2u + 1u] = select(positions[positionsOffset + row * 2u + 1u], nan, isEmpty);`,d)),M}getExtremePointFilterNodes(e,t){let{id:n,groupCount:r}=this,{rows:a,groupKeys:o,latticeX:s,latticeY:c,filteredGroupKeys:d}=t,f=r*b,p=i(e,`${n}-extreme-values`,`uint32`,f),h=i(e,`${n}-extreme-ties`,`uint32`,f),g=i(e,`${n}-octagon`,`uint32`,2*f),_=(e,t,n=`u32`,r=`read`)=>({name:e,view:t,type:n,access:r}),v=r<=S,y=`const GROUP_COUNT: u32 = ${r}u;
const LATTICE_BIAS: u32 = ${T}u;`,x=(t,r,i,l)=>{let d=`let group = groupKeys[groupKeysOffset + row];
    if (group < GROUP_COUNT) {
      let x = latticeX[latticeXOffset + row];
      let y = latticeY[latticeYOffset + row];
      let sum = x + y;
      let difference = x + LATTICE_BIAS - y;
      let base = group * ${b}u;
      ${l((e,t)=>v?`atomicMin(&localTable[${e}], ${t});`:`atomicMin(&table[tableOffset + ${e}], ${t});`)}
    }`,p=[_(`groupKeys`,o),_(`latticeX`,s),_(`latticeY`,c),...i,_(`table`,r,`atomic<u32>`,`read_write`)];if(!v)return u(e,{id:`${n}-${t}`,operation:m,variant:t,bindings:p,invocationCount:a,declarations:y,body:`let row = index;
  ${d}`});let h=Math.ceil(a/C),g=Math.ceil(h/w)*w;return u(e,{id:`${n}-${t}`,operation:m,variant:t,bindings:p,invocationCount:h,workgroupSize:w,guardIndex:!1,declarations:`${y}
const TABLE_LENGTH: u32 = ${f}u;
var<workgroup> localTable: array<atomic<u32>, ${f}>;`,body:`for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${w}u) {
    atomicStore(&localTable[slot], 0xffffffffu);
  }
  workgroupBarrier();
  for (var rowStep = 0u; rowStep < ${C}u; rowStep++) {
    let row = index + rowStep * ${g}u;
    if (row < ${a}u) {
    ${d}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${w}u) {
    let value = atomicLoad(&localTable[slot]);
    if (value != 0xffffffffu) {
      atomicMin(&table[tableOffset + slot], value);
    }
  }`})},E=(t,r)=>l(e,{id:`${n}-fill-${t}`,operation:m,view:r,type:`u32`,value:`0xffffffffu`,componentCount:f}),D=`let minX = values[valuesOffset + base];
      let maxX = ~values[valuesOffset + base + 1u];
      let minY = values[valuesOffset + base + 2u];
      let maxY = ~values[valuesOffset + base + 3u];
      let minSum = values[valuesOffset + base + 4u];
      let maxSum = ~values[valuesOffset + base + 5u];
      let minDifference = values[valuesOffset + base + 6u];
      let maxDifference = ~values[valuesOffset + base + 7u];`;return[E(`extreme-values`,p),x(`extreme-values`,p,[],e=>`${e(`base`,`x`)}
      ${e(`base + 1u`,`~x`)}
      ${e(`base + 2u`,`y`)}
      ${e(`base + 3u`,`~y`)}
      ${e(`base + 4u`,`sum`)}
      ${e(`base + 5u`,`~sum`)}
      ${e(`base + 6u`,`difference`)}
      ${e(`base + 7u`,`~difference`)}`),E(`extreme-ties`,h),x(`extreme-ties`,h,[_(`values`,p)],e=>`${D}
      if (x == minX) { ${e(`base`,`y`)} }
      if (x == maxX) { ${e(`base + 1u`,`y`)} }
      if (y == minY) { ${e(`base + 2u`,`x`)} }
      if (y == maxY) { ${e(`base + 3u`,`x`)} }
      if (sum == minSum) { ${e(`base + 4u`,`x`)} }
      if (sum == maxSum) { ${e(`base + 5u`,`x`)} }
      if (difference == minDifference) { ${e(`base + 6u`,`x`)} }
      if (difference == maxDifference) { ${e(`base + 7u`,`x`)} }`),u(e,{id:`${n}-extreme-points`,operation:m,variant:`extreme-points`,bindings:[_(`values`,p),_(`ties`,h),_(`octagon`,g,`u32`,`read_write`)],invocationCount:r,declarations:`const LATTICE_BIAS: u32 = ${T}u;`,body:`let base = index * ${b}u;
  let outputBase = index * 16u;
  ${D.replace(/^ {6}/gm,`  `)}
  // Empty groups decode to arbitrary words; no row ever reads them, and u32 arithmetic wraps.
  var points = array<vec2<u32>, 8>(
    vec2<u32>(ties[tiesOffset + base + 2u], minY),
    vec2<u32>(ties[tiesOffset + base + 7u], ties[tiesOffset + base + 7u] + LATTICE_BIAS - maxDifference),
    vec2<u32>(maxX, ties[tiesOffset + base + 1u]),
    vec2<u32>(ties[tiesOffset + base + 5u], maxSum - ties[tiesOffset + base + 5u]),
    vec2<u32>(ties[tiesOffset + base + 3u], maxY),
    vec2<u32>(ties[tiesOffset + base + 6u], ties[tiesOffset + base + 6u] + LATTICE_BIAS - minDifference),
    vec2<u32>(minX, ties[tiesOffset + base]),
    vec2<u32>(ties[tiesOffset + base + 4u], minSum - ties[tiesOffset + base + 4u])
  );
  for (var vertex = 0u; vertex < 8u; vertex++) {
    octagon[octagonOffset + outputBase + 2u * vertex] = points[vertex].x;
    octagon[octagonOffset + outputBase + 2u * vertex + 1u] = points[vertex].y;
  }`}),u(e,{id:`${n}-filter-rows`,operation:m,variant:`filter-rows`,bindings:[_(`groupKeys`,o),_(`latticeX`,s),_(`latticeY`,c),_(`octagon`,g),_(`filteredKeys`,d,`u32`,`read_write`)],invocationCount:a,declarations:`${y}
${O}`,body:`let group = groupKeys[groupKeysOffset + index];
  var key = group;
  if (group < GROUP_COUNT) {
    let px = i32(latticeX[latticeXOffset + index]);
    let py = i32(latticeY[latticeYOffset + index]);
    var edgeCount = 0u;
    var isInterior = true;
    for (var edge = 0u; edge < 8u; edge++) {
      let startIndex = octagonOffset + group * 16u + edge * 2u;
      let endIndex = octagonOffset + group * 16u + ((edge + 1u) & 7u) * 2u;
      let ax = i32(octagon[startIndex]);
      let ay = i32(octagon[startIndex + 1u]);
      let bx = i32(octagon[endIndex]);
      let by = i32(octagon[endIndex + 1u]);
      if (ax == bx && ay == by) {
        continue;
      }
      edgeCount++;
      if (orientationOfPoints(ax, ay, bx, by, px, py) <= 0) {
        isInterior = false;
        break;
      }
    }
    if (isInterior && edgeCount > 0u) {
      key = GROUP_COUNT;
    }
  }
  filteredKeys[filteredKeysOffset + index] = key;`})]}},O=`
// Exact 32 x 32 -> 64 bit unsigned product as (high, low); operands are below 2^30.
fn multiplyUnsigned(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let middle = a0 * b1 + a1 * b0;
  let low = a0 * b0 + (middle << 16u);
  let carry = select(0u, 1u, low < a0 * b0);
  return vec2<u32>(a1 * b1 + (middle >> 16u) + carry, low);
}

fn getSign(value: i32) -> i32 { return select(select(0, 1, value > 0), -1, value < 0); }

// Sign of the exact value left - right where each side is (sign, 64-bit magnitude).
fn compareProducts(leftSign: i32, left: vec2<u32>, rightSign: i32, right: vec2<u32>) -> i32 {
  if (leftSign != rightSign) {
    return select(-1, 1, leftSign > rightSign);
  }
  if (leftSign == 0) {
    return 0;
  }
  var magnitude = 0;
  if (left.x != right.x) {
    magnitude = select(-1, 1, left.x > right.x);
  } else if (left.y != right.y) {
    magnitude = select(-1, 1, left.y > right.y);
  }
  return magnitude * leftSign;
}

// Exact orientation of points a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientationOfPoints(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
  let abx = bx - ax;
  let aby = by - ay;
  let acx = cx - ax;
  let acy = cy - ay;
  let leftSign = getSign(abx) * getSign(acy);
  let rightSign = getSign(aby) * getSign(acx);
  let left = multiplyUnsigned(u32(abs(abx)), u32(abs(acy)));
  let right = multiplyUnsigned(u32(abs(aby)), u32(abs(acx)));
  return compareProducts(leftSign, left, rightSign, right);
}
`,k=`${O}
fn isRepresentative(row: u32, begin: u32) -> bool {
  return row == begin || latticeX[latticeXOffset + row] != latticeX[latticeXOffset + row - 1u] ||
    latticeY[latticeYOffset + row] != latticeY[latticeYOffset + row - 1u];
}

// Exact orientation of sorted rows a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientation(a: u32, b: u32, c: u32) -> i32 {
  return orientationOfPoints(
    i32(latticeX[latticeXOffset + a]), i32(latticeY[latticeYOffset + a]),
    i32(latticeX[latticeXOffset + b]), i32(latticeY[latticeYOffset + b]),
    i32(latticeX[latticeXOffset + c]), i32(latticeY[latticeYOffset + c])
  );
}
`;export{D as t};