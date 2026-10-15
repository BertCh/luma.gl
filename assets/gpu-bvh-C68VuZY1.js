import{n as e}from"./gpu-vector-format-zKtof1xn.js";import{D as t,I as n,M as r,S as i,T as a,j as o,k as s,x as c,z as l}from"./resources-CwvhEOBe.js";var u=256,d=128,f=64,p=4294967295,m=class{id;strategy;resolvedStrategy;minima;maxima;sourceIds;leafCapacity;nodeMinima;nodeMaxima;nodeChildren;leafIds;count;overflow;dimension;nodeCount;internalNodeCount;levelCount;rootNode=0;topology=`complete-binary`;updatePolicy=`refit`;stats;constructor(t){if(this.id=t.id??`gpu-bvh`,this.strategy=t.strategy??`auto`,this.minima=t.minima,this.maxima=t.maxima,this.sourceIds=t.sourceIds,this.leafCapacity=t.leafCapacity,this.nodeMinima=t.nodeMinima,this.nodeMaxima=t.nodeMaxima,this.nodeChildren=t.nodeChildren,this.leafIds=t.leafIds,this.count=t.count,this.overflow=t.overflow,this.dimension=this.minima.format===`float32x2`?2:3,this.minima.length>p)throw Error(`${this.id} source row count exceeds uint32 range`);if(!Number.isSafeInteger(this.leafCapacity)||!x(this.leafCapacity))throw Error(`${this.id} leafCapacity must be a positive power of two`);if(![`auto`,`fused`,`level`].includes(this.strategy))throw Error(`${this.id} strategy must be auto, fused, or level`);let n=this.minima.buffer.graph.device.limits,i=this.leafCapacity<=d&&this.leafCapacity<=n.maxComputeInvocationsPerWorkgroup&&this.leafCapacity<=n.maxComputeWorkgroupSizeX&&this.leafCapacity*f<=n.maxComputeWorkgroupStorageSize;if(this.strategy===`fused`&&!i)throw Error(`${this.id} fused strategy exceeds portable single-workgroup limits`);if(this.resolvedStrategy=this.strategy===`level`||!i?`level`:`fused`,this.nodeCount=this.leafCapacity*2-1,this.internalNodeCount=this.leafCapacity-1,this.levelCount=Math.log2(this.leafCapacity)+1,!Number.isSafeInteger(this.nodeCount)||this.nodeCount>p)throw Error(`${this.id} node count exceeds uint32 range`);if(r(this.minima,[`float32x2`,`float32x3`],`${this.id} minima`),r(this.maxima,[`float32x2`,`float32x3`],`${this.id} maxima`),r(this.nodeMinima,[`float32x2`,`float32x3`],`${this.id} nodeMinima`),r(this.nodeMaxima,[`float32x2`,`float32x3`],`${this.id} nodeMaxima`),r(this.nodeChildren,[`uint32x2`],`${this.id} nodeChildren`),o(this.leafIds,`${this.id} leafIds`),o(this.count,`${this.id} count`),o(this.overflow,`${this.id} overflow`),this.sourceIds&&o(this.sourceIds,`${this.id} sourceIds`),this.minima.format!==this.maxima.format||this.minima.length!==this.maxima.length)throw Error(`${this.id} minima and maxima must have matching formats and lengths`);if(this.sourceIds&&this.sourceIds.length!==this.minima.length)throw Error(`${this.id} sourceIds.length must equal bounds length`);if(this.nodeMinima.format!==this.minima.format||this.nodeMaxima.format!==this.minima.format||this.nodeMinima.length!==this.nodeCount||this.nodeMaxima.length!==this.nodeCount)throw Error(`${this.id} node bounds must match source format and node count`);if(this.nodeChildren.length!==this.nodeCount)throw Error(`${this.id} nodeChildren.length must equal node count`);if(this.leafIds.length!==this.leafCapacity)throw Error(`${this.id} leafIds.length must equal leafCapacity`);if(this.count.length<1||this.overflow.length<1)throw Error(`${this.id} count and overflow must each contain one uint32 row`);v(this);let a=e(this.minima.format).byteLength;this.stats={dimension:this.dimension,leafCapacity:this.leafCapacity,internalNodeCount:this.internalNodeCount,nodeCount:this.nodeCount,levelCount:this.levelCount,outputByteLength:this.nodeCount*(a*2+Uint32Array.BYTES_PER_ELEMENT*2)+this.leafCapacity*Uint32Array.BYTES_PER_ELEMENT+Uint32Array.BYTES_PER_ELEMENT*2}}getCommandNodes(e){let t=[];if([this.minima,this.maxima,...this.sourceIds?[this.sourceIds]:[],this.nodeMinima,this.nodeMaxima,this.nodeChildren,this.leafIds,this.count,this.overflow].some(t=>t.buffer.graph!==e))throw Error(`${this.id} views must belong to the target graph`);if(this.resolvedStrategy===`fused`)t.push(...h(e,this));else{t.push(...g(e,this,c(`${this.id}-load-leaves`,this.nodeCount,u,e.device.limits.maxComputeWorkgroupsPerDimension)));for(let n=this.levelCount-2;n>=0;n--)t.push(...y(e,this,n))}return this.sourceIds&&t.push(..._(e,this,this.sourceIds)),t}};function h(e,t){let n=[],r=`
const SOURCE_COUNT: u32 = ${t.minima.length}u;
const STORED_COUNT: u32 = ${Math.min(t.minima.length,t.leafCapacity)}u;
const LEAF_CAPACITY: u32 = ${t.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const DIMENSION: u32 = ${t.dimension}u;
const MINIMA_OFFSET: u32 = ${s(t.minima)}u;
const MAXIMA_OFFSET: u32 = ${s(t.maxima)}u;
const NODE_MINIMA_OFFSET: u32 = ${s(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${s(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${s(t.nodeChildren)}u;
const LEAF_IDS_OFFSET: u32 = ${s(t.leafIds)}u;
const COUNT_OFFSET: u32 = ${s(t.count)}u;
const OVERFLOW_OFFSET: u32 = ${s(t.overflow)}u;
@group(0) @binding(0) var<storage, read> sourceMinima: array<f32>;
@group(0) @binding(1) var<storage, read> sourceMaxima: array<f32>;
@group(0) @binding(2) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeChildren: array<u32>;
@group(0) @binding(5) var<storage, read_write> leafIds: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(7) var<storage, read_write> outputOverflow: array<u32>;

var<workgroup> sharedMinima: array<vec4<f32>, ${t.leafCapacity*2}>;
var<workgroup> sharedMaxima: array<vec4<f32>, ${t.leafCapacity*2}>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${t.leafCapacity}) fn main(
  @builtin(local_invocation_index) localIndex: u32
) {
  var minimum = vec4<f32>(3.402823466e+38);
  var maximum = vec4<f32>(-3.402823466e+38);
  let leafNode = INTERNAL_NODE_COUNT + localIndex;
  let leafComponent = leafNode * DIMENSION;
  let leafChildComponent = leafNode * 2u;
  nodeChildren[CHILDREN_OFFSET + leafChildComponent] = ${p}u;
  nodeChildren[CHILDREN_OFFSET + leafChildComponent + 1u] = ${p}u;
  leafIds[LEAF_IDS_OFFSET + localIndex] = ${p}u;

  if (localIndex < STORED_COUNT) {
    let sourceComponent = localIndex * DIMENSION;
    var valid = true;
    for (var axis = 0u; axis < DIMENSION; axis++) {
      let sourceMinimum = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
      let sourceMaximum = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
      valid = valid && finite(sourceMinimum) && finite(sourceMaximum) &&
        sourceMinimum <= sourceMaximum;
    }
    leafIds[LEAF_IDS_OFFSET + localIndex] = localIndex;
    if (valid) {
      for (var axis = 0u; axis < DIMENSION; axis++) {
        minimum[axis] = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
        maximum[axis] = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
      }
    }
  }

  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + leafComponent + axis] = minimum[axis];
    nodeMaxima[NODE_MAXIMA_OFFSET + leafComponent + axis] = maximum[axis];
  }
  sharedMinima[localIndex] = minimum;
  sharedMaxima[localIndex] = maximum;

  if (localIndex < INTERNAL_NODE_COUNT) {
    let childComponent = localIndex * 2u;
    nodeChildren[CHILDREN_OFFSET + childComponent] = localIndex * 2u + 1u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = localIndex * 2u + 2u;
  }
  if (localIndex == 0u) {
    outputCount[COUNT_OFFSET] = SOURCE_COUNT;
    outputOverflow[OVERFLOW_OFFSET] = select(0u, 1u, SOURCE_COUNT > LEAF_CAPACITY);
  }
  workgroupBarrier();

  var sourceOffset = 0u;
  var destinationOffset = LEAF_CAPACITY;
  for (var levelNodeCount = LEAF_CAPACITY / 2u;
       levelNodeCount > 0u;
       levelNodeCount = levelNodeCount / 2u) {
    if (localIndex < levelNodeCount) {
      let firstChild = sourceOffset + localIndex * 2u;
      let reducedMinimum = min(sharedMinima[firstChild], sharedMinima[firstChild + 1u]);
      let reducedMaximum = max(sharedMaxima[firstChild], sharedMaxima[firstChild + 1u]);
      sharedMinima[destinationOffset + localIndex] = reducedMinimum;
      sharedMaxima[destinationOffset + localIndex] = reducedMaximum;

      let nodeIndex = levelNodeCount - 1u + localIndex;
      let nodeComponent = nodeIndex * DIMENSION;
      for (var axis = 0u; axis < DIMENSION; axis++) {
        nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] = reducedMinimum[axis];
        nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] = reducedMaximum[axis];
      }
    }
    workgroupBarrier();
    let previousSourceOffset = sourceOffset;
    sourceOffset = destinationOffset;
    destinationOffset = previousSourceOffset;
  }
}`,i=[{buffer:t.minima,usage:`storage-read`},{buffer:t.maxima,usage:`storage-read`},{buffer:t.nodeMinima,usage:`storage-write`},{buffer:t.nodeMaxima,usage:`storage-write`},{buffer:t.nodeChildren,usage:`storage-write`},{buffer:t.leafIds,usage:`storage-write`},{buffer:t.count,usage:`storage-write`},{buffer:t.overflow,usage:`storage-write`}];return n.push(...b(e,{id:`${t.id}-fused-refit`,source:r,resources:i,bindings:{sourceMinima:t.minima,sourceMaxima:t.maxima,nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren,leafIds:t.leafIds,outputCount:t.count,outputOverflow:t.overflow},dispatch:{x:1,y:1,z:1}})),n}function g(e,t,n){let r=[],a=`
const SOURCE_COUNT: u32 = ${t.minima.length}u;
const STORED_COUNT: u32 = ${Math.min(t.minima.length,t.leafCapacity)}u;
const LEAF_CAPACITY: u32 = ${t.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${t.internalNodeCount}u;
const NODE_COUNT: u32 = ${t.nodeCount}u;
const DIMENSION: u32 = ${t.dimension}u;
const MINIMA_OFFSET: u32 = ${s(t.minima)}u;
const MAXIMA_OFFSET: u32 = ${s(t.maxima)}u;
const NODE_MINIMA_OFFSET: u32 = ${s(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${s(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${s(t.nodeChildren)}u;
const LEAF_IDS_OFFSET: u32 = ${s(t.leafIds)}u;
const COUNT_OFFSET: u32 = ${s(t.count)}u;
const OVERFLOW_OFFSET: u32 = ${s(t.overflow)}u;
@group(0) @binding(0) var<storage, read> sourceMinima: array<f32>;
@group(0) @binding(1) var<storage, read> sourceMaxima: array<f32>;
@group(0) @binding(2) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeChildren: array<u32>;
@group(0) @binding(5) var<storage, read_write> leafIds: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(7) var<storage, read_write> outputOverflow: array<u32>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${i(n,u)}
  let nodeIndex = index;
  if (nodeIndex >= NODE_COUNT) { return; }
  let nodeComponent = nodeIndex * DIMENSION;
  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] = 3.402823466e+38;
    nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] = -3.402823466e+38;
  }
  let childComponent = nodeIndex * 2u;
  if (nodeIndex < INTERNAL_NODE_COUNT) {
    nodeChildren[CHILDREN_OFFSET + childComponent] = nodeIndex * 2u + 1u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = nodeIndex * 2u + 2u;
  } else {
    nodeChildren[CHILDREN_OFFSET + childComponent] = ${p}u;
    nodeChildren[CHILDREN_OFFSET + childComponent + 1u] = ${p}u;
    let leafIndex = nodeIndex - INTERNAL_NODE_COUNT;
    leafIds[LEAF_IDS_OFFSET + leafIndex] = ${p}u;
    if (leafIndex < STORED_COUNT) {
      var valid = true;
      let sourceComponent = leafIndex * DIMENSION;
      for (var axis = 0u; axis < DIMENSION; axis++) {
        let minimum = sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
        let maximum = sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
        valid = valid && finite(minimum) && finite(maximum) && minimum <= maximum;
      }
      leafIds[LEAF_IDS_OFFSET + leafIndex] = leafIndex;
      if (valid) {
        for (var axis = 0u; axis < DIMENSION; axis++) {
          nodeMinima[NODE_MINIMA_OFFSET + nodeComponent + axis] =
            sourceMinima[MINIMA_OFFSET + sourceComponent + axis];
          nodeMaxima[NODE_MAXIMA_OFFSET + nodeComponent + axis] =
            sourceMaxima[MAXIMA_OFFSET + sourceComponent + axis];
        }
      }
    }
  }
  if (nodeIndex == 0u) {
    outputCount[COUNT_OFFSET] = SOURCE_COUNT;
    outputOverflow[OVERFLOW_OFFSET] = select(0u, 1u, SOURCE_COUNT > LEAF_CAPACITY);
  }
}`,o=[{buffer:t.minima,usage:`storage-read`},{buffer:t.maxima,usage:`storage-read`},{buffer:t.nodeMinima,usage:`storage-write`},{buffer:t.nodeMaxima,usage:`storage-write`},{buffer:t.nodeChildren,usage:`storage-write`},{buffer:t.leafIds,usage:`storage-write`},{buffer:t.count,usage:`storage-write`},{buffer:t.overflow,usage:`storage-write`}];return r.push(...b(e,{id:`${t.id}-load-leaves`,source:a,resources:o,bindings:{sourceMinima:t.minima,sourceMaxima:t.maxima,nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren,leafIds:t.leafIds,outputCount:t.count,outputOverflow:t.overflow},dispatch:n})),r}function _(e,t,n){let r=[],a=Math.min(n.length,t.leafCapacity),o=c(`${t.id}-remap-source-ids`,a,u,e.device.limits.maxComputeWorkgroupsPerDimension),l=`
const STORED_COUNT: u32 = ${a}u;
const SOURCE_IDS_OFFSET: u32 = ${s(n)}u;
const LEAF_IDS_OFFSET: u32 = ${s(t.leafIds)}u;
@group(0) @binding(0) var<storage, read> sourceIds: array<u32>;
@group(0) @binding(1) var<storage, read_write> leafIds: array<u32>;

@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${i(o,u)}
  let leafIndex = index;
  if (leafIndex >= STORED_COUNT) { return; }
  let sourceIndex = leafIds[LEAF_IDS_OFFSET + leafIndex];
  if (sourceIndex == ${p}u) { return; }
  leafIds[LEAF_IDS_OFFSET + leafIndex] = sourceIds[SOURCE_IDS_OFFSET + sourceIndex];
}`;return r.push(...b(e,{id:`${t.id}-remap-source-ids`,source:l,resources:[{buffer:n,usage:`storage-read`},{buffer:t.leafIds,usage:`storage-read-write`}],bindings:{sourceIds:n,leafIds:t.leafIds},dispatch:o})),r}function v(e){let t=[e.minima,e.maxima,...e.sourceIds?[e.sourceIds]:[]],n=[e.nodeMinima,e.nodeMaxima,e.nodeChildren,e.leafIds,e.count,e.overflow];for(let r=0;r<n.length;r++){let i=n[r];if(t.some(e=>a(e,i)))throw Error(`${e.id} output views must not overlap source inputs`);if(n.slice(r+1).some(e=>a(i,e)))throw Error(`${e.id} output views must not overlap one another`)}}function y(e,t,n){let r=[],a=2**n-1,o=2**n,l=c(`${t.id}-refit-depth-${n}`,o,u,e.device.limits.maxComputeWorkgroupsPerDimension),d=`
const FIRST_NODE: u32 = ${a}u;
const LEVEL_NODE_COUNT: u32 = ${o}u;
const DIMENSION: u32 = ${t.dimension}u;
const NODE_MINIMA_OFFSET: u32 = ${s(t.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${s(t.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${s(t.nodeChildren)}u;
@group(0) @binding(0) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(1) var<storage, read_write> nodeMaxima: array<f32>;
@group(0) @binding(2) var<storage, read> nodeChildren: array<u32>;

@compute @workgroup_size(${u}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${i(l,u)}
  if (index >= LEVEL_NODE_COUNT) { return; }
  let nodeIndex = FIRST_NODE + index;
  let childComponent = nodeIndex * 2u;
  let left = nodeChildren[CHILDREN_OFFSET + childComponent];
  let right = nodeChildren[CHILDREN_OFFSET + childComponent + 1u];
  for (var axis = 0u; axis < DIMENSION; axis++) {
    nodeMinima[NODE_MINIMA_OFFSET + nodeIndex * DIMENSION + axis] = min(
      nodeMinima[NODE_MINIMA_OFFSET + left * DIMENSION + axis],
      nodeMinima[NODE_MINIMA_OFFSET + right * DIMENSION + axis]
    );
    nodeMaxima[NODE_MAXIMA_OFFSET + nodeIndex * DIMENSION + axis] = max(
      nodeMaxima[NODE_MAXIMA_OFFSET + left * DIMENSION + axis],
      nodeMaxima[NODE_MAXIMA_OFFSET + right * DIMENSION + axis]
    );
  }
}`;return r.push(...b(e,{id:`${t.id}-refit-depth-${n}`,source:d,resources:[{buffer:t.nodeMinima,usage:`storage-read-write`},{buffer:t.nodeMaxima,usage:`storage-read-write`},{buffer:t.nodeChildren,usage:`storage-read`}],bindings:{nodeMinima:t.nodeMinima,nodeMaxima:t.nodeMaxima,nodeChildren:t.nodeChildren},dispatch:l})),r}function b(e,r){let i=[];return i.push(n({id:r.id,resources:r.resources,compile:({device:e})=>{let n=new l(e,{id:r.id,source:r.source,shaderLayout:{bindings:Object.keys(r.bindings).map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:i})=>{let a={};for(let[e,n]of Object.entries(r.bindings))a[e]=t(n,i);n.dispatch(e,{bindings:a,x:r.dispatch.x,y:r.dispatch.y,z:r.dispatch.z})},destroy:()=>n.destroy()}}})),i}function x(e){return e>0&&Number.isInteger(Math.log2(e))}export{m as t};