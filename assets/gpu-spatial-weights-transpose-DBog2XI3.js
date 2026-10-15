import{_ as e,h as t,w as n,y as r}from"./resources-CwvhEOBe.js";import{t as i}from"./gpu-reduction-BD8l40jM.js";import{l as a,s as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as s,t as c}from"./spatial-weights-D4bd9AQ6.js";import{t as l}from"./slot-grouping-DuACcF2F.js";var u=`GPUSpatialWeightsTranspose`,d=class{id;props;rowCount;columnCount;outputWeights;constructor(e){let n=e.id??`spatial-weights-transpose`;this.id=n,this.props=e;let{weights:i,output:a}=e,o=s(n,i),l=e.columnCount??o;if(!Number.isInteger(l)||l<1)throw Error(`${n} columnCount must be a positive integer`);if(s(n,a,`output`)!==l)throw Error(`${n} output.offsets must hold columnCount + 1 entries`);if(a.neighbors.length<i.neighbors.length)throw Error(`${n} output capacity must be >= the input capacity`);if(i.distances&&!a.distances)throw Error(`${n} output.distances is required with weights.distances`);if(e.output.asymmetricSlots){if(r(e.output.asymmetricSlots,[`uint32`],`${n} asymmetricSlots`),e.output.asymmetricSlots.length!==1)throw Error(`${n} asymmetricSlots must hold one uint32 row`);if(o!==l)throw Error(`${n} asymmetricSlots requires square weights`)}if(e.symmetryTolerance!==void 0&&!(Number.isFinite(e.symmetryTolerance)&&e.symmetryTolerance>=0))throw Error(`${n} symmetryTolerance must be finite and >= 0`);a.distances&&r(a.distances,[`float32`],`${n} output.distances`),this.rowCount=o,this.columnCount=l;let d=i.information?.transform??`unknown`;this.outputWeights=c(a,{spatialContext:i.spatialContext,information:{rowCount:l,transform:d===`row`||d===`variance`?`unknown`:d,symmetric:o===l?i.information?.symmetric??`unknown`:`unknown`,hasSelfWeights:i.information?.hasSelfWeights,provenance:{...i.information?.provenance,source:u,criterion:`transpose`}},status:i.status}),t(n,[a.offsets,a.neighbors,a.weights,a.distances,e.output.asymmetricSlots],[i.offsets,i.neighbors,i.weights,i.distances])}getCommandNodes(t){let{id:r,props:s,rowCount:c,columnCount:d}=this,{weights:f,output:p}=s;e(r,t,[f.offsets,f.neighbors,f.weights,f.distances,p.offsets,p.neighbors,p.weights,p.distances,s.output.asymmetricSlots]);let m=f.neighbors.length,h=(e,i,a)=>n(t,`${r}-${e}`,i,a),g=h(`slot-keys`,`uint32`,m),_=h(`slot-indices`,`uint32`,m),v=`const ROWS: u32 = ${c}u;
const COLUMNS: u32 = ${d}u;`,y=l(t,{id:r,operation:u,slotKeys:g,slotIndices:_,columns:d,columnOffsets:p.offsets}),{sortedKeys:b,sortedSlots:x}=y,S=[o(t,{id:`${r}-slot-keys`,operation:u,variant:`slot-keys`,bindings:[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read`},{name:`slotKeys`,view:g,type:`u32`,access:`read_write`},{name:`slotIndices`,view:_,type:`u32`,access:`read_write`}],invocationCount:m,declarations:v,body:`let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < COLUMNS;
  // Unused and out-of-range slots share the key COLUMNS, which sorts after every column.
  slotKeys[slotKeysOffset + index] = select(COLUMNS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`}),...y.nodes,o(t,{id:`${r}-gather`,operation:u,variant:f.distances?`gather-distances`:`gather`,bindings:[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`weights`,view:f.weights,type:`f32`,access:`read`},{name:`sortedKeys`,view:b,type:`u32`,access:`read`},{name:`sortedSlots`,view:x,type:`u32`,access:`read`},{name:`outputNeighbors`,view:p.neighbors,type:`u32`,access:`read_write`},{name:`outputWeights`,view:p.weights,type:`f32`,access:`read_write`},...f.distances?[{name:`distances`,view:f.distances,type:`f32`,access:`read`},{name:`outputDistances`,view:p.distances,type:`f32`,access:`read_write`}]:[]],invocationCount:p.neighbors.length,declarations:`${v}
const INPUT_CAPACITY: u32 = ${m}u;

// Row of a used slot: the last row whose offset is at most the slot.
fn findRow(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] <= slot) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`var neighbor = 0u;
  var weight = 0.0;
  var distance = 0.0;
  if (index < INPUT_CAPACITY && sortedKeys[sortedKeysOffset + index] < COLUMNS) {
    let slot = sortedSlots[sortedSlotsOffset + index];
    neighbor = findRow(slot);
    weight = weights[weightsOffset + slot];
    ${f.distances?`distance = distances[distancesOffset + slot];`:``}
  }
  outputNeighbors[outputNeighborsOffset + index] = neighbor;
  outputWeights[outputWeightsOffset + index] = weight;
  ${f.distances?`outputDistances[outputDistancesOffset + index] = distance;`:``}`})];if(s.output.asymmetricSlots){let e=h(`row-mismatches`,`uint32`,c);S.push(o(t,{id:`${r}-symmetry-rows`,operation:u,variant:`symmetry-rows`,bindings:[{name:`offsets`,view:f.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:f.neighbors,type:`u32`,access:`read`},{name:`weights`,view:f.weights,type:`f32`,access:`read`},{name:`transposeOffsets`,view:p.offsets,type:`u32`,access:`read`},{name:`transposeNeighbors`,view:p.neighbors,type:`u32`,access:`read`},{name:`transposeWeights`,view:p.weights,type:`f32`,access:`read`},{name:`rowMismatches`,view:e,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`${v}
const TOLERANCE: f32 = ${a(s.symmetryTolerance??0)};
const NONE: u32 = 0xffffffffu;`,body:`var first = offsets[offsetsOffset + index];
  let firstEnd = offsets[offsetsOffset + index + 1u];
  var second = transposeOffsets[transposeOffsetsOffset + index];
  let secondEnd = transposeOffsets[transposeOffsetsOffset + index + 1u];
  var mismatches = 0u;
  loop {
    // Skip out-of-range neighbors of W; the transpose never holds them.
    while (first < firstEnd && neighbors[neighborsOffset + first] >= COLUMNS) {
      first++;
    }
    if (first >= firstEnd && second >= secondEnd) {
      break;
    }
    let firstColumn = select(NONE, neighbors[neighborsOffset + first], first < firstEnd);
    let secondColumn = select(
      NONE,
      transposeNeighbors[transposeNeighborsOffset + second],
      second < secondEnd
    );
    if (firstColumn == secondColumn) {
      if (abs(weights[weightsOffset + first] - transposeWeights[transposeWeightsOffset + second]) > TOLERANCE) {
        mismatches++;
      }
      first++;
      second++;
    } else if (firstColumn < secondColumn) {
      mismatches++;
      first++;
    } else {
      mismatches++;
      second++;
    }
  }
  rowMismatches[rowMismatchesOffset + index] = mismatches;`}),...new i({id:`${r}-symmetry-total`,input:e,output:s.output.asymmetricSlots,operation:`sum`}).getCommandNodes(t))}return S}};export{d as t};