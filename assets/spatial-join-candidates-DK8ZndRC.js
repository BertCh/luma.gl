import{w as e}from"./resources-CwvhEOBe.js";import{t}from"./gpu-scan-BuGhTLDQ.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{d as r}from"./wgsl-common-BNRVzgd8.js";import{n as i}from"./scan-total-BdolCtK7.js";import{t as a}from"./geodesic-wgsl-DS0CJZ42.js";import{t as o}from"./spatial-join-passes-C0ZwcuNs.js";var s=`GPUSpatialJoinCandidates`;function c(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}function l(l,u){let{id:d,leftCount:f,rightCount:p,candidateCapacity:m,tree:h,state:g,candidatePairs:_}=u,v=[],y=e(l,`${d}-left-counts`,`uint32`,f),b=e(l,`${d}-left-offsets`,`uint32`,f),x=typeof u.margin==`number`?void 0:u.margin,{sphereRadius:S}=u,C=e=>[{name:`leftMinima`,view:u.leftMinima,type:`f32`,access:`read`},{name:`leftMaxima`,view:u.leftMaxima,type:`f32`,access:`read`},{name:`nodeMinima`,view:h.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:h.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:h.leafIds,type:`u32`,access:`read`},...x?[{name:`marginRow`,view:x,type:`f32`,access:`read`}]:[],...e],w=x?`fn getMargin() -> f32 {
  let margin = marginRow[marginRowOffset];
  return select(0.0, margin, margin >= 0.0 && margin <= 3.4028234e38);
}`:`fn getMargin() -> f32 { return ${c(Math.fround(u.margin))}; }`,T=`${o}
const INTERNAL_NODE_COUNT: u32 = ${h.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${p}u;
const CANDIDATE_CAPACITY: u32 = ${m}u;
${w}
${r}
${S===void 0?``:`${a}
const SPHERE_RADIUS: f32 = ${c(S)};
const WRAP_SHIFT: vec2f = vec2f(360.0, 0.0);`}`,E=S===void 0?`let queryMinimum = boxMinimum - vec2f(margin);
    let queryMaximum = boxMaximum + vec2f(margin);`:`var queryMinimum = boxMinimum;
    var queryMaximum = boxMaximum;
    var wrapDown = false;
    var wrapUp = false;
    if (margin > 0.0) {
      let angle = margin / SPHERE_RADIUS * 1.001;
      let deltaLatitude = angle * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
      queryMinimum.y = boxMinimum.y - deltaLatitude;
      queryMaximum.y = boxMaximum.y + deltaLatitude;
      let farLatitude = max(abs(queryMinimum.y), abs(queryMaximum.y));
      var ratio = 2.0;
      if (farLatitude < 90.0) { ratio = sin(0.5 * angle) / geodesicCosLatitude(farLatitude); }
      if (ratio < 1.0) {
        let deltaLongitude = 2.0 * asin(ratio) * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
        queryMinimum.x = boxMinimum.x - deltaLongitude;
        queryMaximum.x = boxMaximum.x + deltaLongitude;
        wrapDown = queryMaximum.x > 180.0;
        wrapUp = queryMinimum.x < -180.0;
      } else {
        // The cap reaches a pole (or is wider than a hemisphere in longitude): every longitude.
        queryMinimum.x = -FLOAT32_MAXIMUM;
        queryMaximum.x = FLOAT32_MAXIMUM;
      }
    }`,D=S===void 0?`nodeOverlaps(node, queryMinimum, queryMaximum)`:`(nodeOverlaps(node, queryMinimum, queryMaximum) ||
        (wrapDown && nodeOverlaps(node, queryMinimum - WRAP_SHIFT, queryMaximum - WRAP_SHIFT)) ||
        (wrapUp && nodeOverlaps(node, queryMinimum + WRAP_SHIFT, queryMaximum + WRAP_SHIFT)))`,O=(e,t,n)=>`let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${t}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let margin = getMargin();
    ${E}
    var node = 0u;
    loop {
      if (${D}) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${e}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${n}`;v.push(n(l,{id:`${d}-count-candidates`,operation:s,variant:`count-candidates`,bindings:C([{name:`leftCounts`,view:y,type:`u32`,access:`read_write`}]),invocationCount:f,declarations:T,body:O(`found = found + 1u;`,`var found = 0u;`,`leftCounts[leftCountsOffset + index] = found;`)})),v.push(...new t({id:`${d}-scan-candidates`,input:y,output:b,mode:`exclusive`}).getCommandNodes(l)),v.push(n(l,{id:`${d}-total-candidates`,operation:s,variant:`total-candidates`,bindings:[{name:`leftCounts`,view:y,type:`u32`,access:`read`},{name:`leftOffsets`,view:b,type:`u32`,access:`read`},{name:`state`,view:g,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset] = leftOffsets[leftOffsetsOffset + ${f-1}u] + leftCounts[leftCountsOffset + ${f-1}u];`})),v.push(i(l,{id:`${d}-candidates`,operation:s,counts:y,offsets:b,rowCount:f,total:g}));let k=h.spatialSort?`
fn readCandidateRight(start: u32, slot: u32) -> u32 {
  return candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u];
}
fn writeCandidateRight(start: u32, slot: u32, value: u32) {
  candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u] = value;
}
fn siftCandidateDown(start: u32, size: u32, value: u32) {
  var parent = 0u;
  loop {
    var child = parent * 2u + 1u;
    if (child >= size) { break; }
    if (child + 1u < size && readCandidateRight(start, child + 1u) > readCandidateRight(start, child)) {
      child = child + 1u;
    }
    let childValue = readCandidateRight(start, child);
    if (childValue <= value) { break; }
    writeCandidateRight(start, parent, childValue);
    parent = child;
  }
  writeCandidateRight(start, parent, value);
}`:``;return v.push(n(l,{id:`${d}-write-candidates`,operation:s,variant:`write-candidates`,bindings:C([{name:`leftOffsets`,view:b,type:`u32`,access:`read`},{name:`candidatePairs`,view:_,type:`u32`,access:`read_write`}]),invocationCount:f,declarations:T+k,body:O(h.spatialSort?`if (found < slotEnd - slotStart) {
            var child = found;
            found = found + 1u;
            loop {
              if (child == 0u) { break; }
              let parent = (child - 1u) / 2u;
              let parentValue = readCandidateRight(slotStart, parent);
              if (parentValue >= rightRow) { break; }
              writeCandidateRight(slotStart, child, parentValue);
              child = parent;
            }
            writeCandidateRight(slotStart, child, rightRow);
          } else if (rightRow < readCandidateRight(slotStart, 0u)) {
            siftCandidateDown(slotStart, found, rightRow);
          }`:`let slot = slotStart + found;
          candidatePairs[candidatePairsOffset + slot * 2u] = index;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          found = found + 1u;
          if (slot + 1u >= slotEnd) { break; }`,`var found = 0u;
  let slotStart = leftOffsets[leftOffsetsOffset + index];
  var slotEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${f}u) {
    slotEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY);
  }
  if (slotStart >= slotEnd) { return; }`,h.spatialSort?`var remaining = found;
  loop {
    if (remaining <= 1u) { break; }
    remaining = remaining - 1u;
    let largest = readCandidateRight(slotStart, 0u);
    let replacement = readCandidateRight(slotStart, remaining);
    writeCandidateRight(slotStart, remaining, largest);
    siftCandidateDown(slotStart, remaining, replacement);
  }
  for (var slot = 0u; slot < found; slot = slot + 1u) {
    candidatePairs[candidatePairsOffset + (slotStart + slot) * 2u] = index;
  }`:``)})),v}export{l as t};