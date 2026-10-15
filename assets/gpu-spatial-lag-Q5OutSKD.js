import{_ as e,d as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as o}from"./spatial-weights-D4bd9AQ6.js";import{t as s}from"./load-balanced-for-each-OquID6hm.js";import{n as c,t as l}from"./float32-values-C3OXugIo.js";var u=`GPUSpatialLag`,d=class{id;props;constructor(e){let r=e.id??`spatial-lag`;this.id=r,this.props=e;let a=o(r,e.weights),s=c(e.values,`${r} values`);i(e.output,[`float32`],`${r} output`);let l=e.columnCount??1;if(t(r,`columnCount`,l),s%l!==0||s===0)throw Error(`${r} values length must be a multiple of columnCount`);if(e.output.length!==a*l)throw Error(`${r} output length must equal rows * columnCount`);if(e.mask){if(s/l!==a)throw Error(`${r} mask requires square weights`);if(i(e.mask,[`uint32`],`${r} mask`),e.mask.length!==a)throw Error(`${r} mask length must equal the weights row count`)}n(r,[e.output],[e.values,e.mask,e.weights.offsets,e.weights.neighbors,e.weights.weights])}getValueRowCount(){return this.props.values.length/(this.props.columnCount??1)}getCommandNodes(t){let{id:n,props:i}=this,{weights:o,mask:c,output:d}=i;e(n,t,[i.values,c,d,o.offsets,o.neighbors,o.weights]);let f=o.offsets.length-1,p=this.getValueRowCount(),m=i.columnCount??1,h=l(t,`${n}-values`,i.values),g=c?`mask[maskOffset + row] != 0u`:`true`,_=c?`mask[maskOffset + neighbor] != 0u`:`true`,v=[...h.nodes,...s(t,{id:`${n}-lag`,operation:u,variant:i.normalize?`normalized`:`sum`,offsets:o.offsets,itemCapacity:o.neighbors.length,laneCount:m,accumulatorType:`f32`,output:d,bindings:[{name:`neighbors`,view:o.neighbors,type:`u32`,access:`read`},{name:`weights`,view:o.weights,type:`f32`,access:`read`},{name:`values`,view:h.view,type:`f32`,access:`read`},...c?[{name:`mask`,view:c,type:`u32`,access:`read`}]:[]],declarations:`const ROWS: u32 = ${p}u;
const COLUMNS: u32 = ${m}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${g}) || neighbor >= ROWS || !(${_})) {
    return f32();
  }
  let weight = weights[weightsOffset + item];
  let lagTerm = weight * values[valuesOffset + neighbor * COLUMNS + lane];
  return lagTerm;
}`})];if(i.normalize){let e=r(t,`${n}-weight-sums`,`float32`,f);v.push(...s(t,{id:`${n}-weights`,operation:u,offsets:o.offsets,itemCapacity:o.neighbors.length,accumulatorType:`f32`,output:e,bindings:[{name:`neighbors`,view:o.neighbors,type:`u32`,access:`read`},{name:`weights`,view:o.weights,type:`f32`,access:`read`},...c?[{name:`mask`,view:c,type:`u32`,access:`read`}]:[]],declarations:`fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${g}) || neighbor >= ${p}u || !(${_})) {
    return 0.0;
  }
  return weights[weightsOffset + item];
}`}),a(t,{id:`${n}-normalize`,operation:u,variant:`normalize`,bindings:[{name:`weightSums`,view:e,type:`f32`,access:`read`},{name:`output`,view:d,type:`f32`,access:`read_write`}],invocationCount:f*m,body:`let lag = output[outputOffset + index];
  let weightSum = weightSums[weightSumsOffset + index / ${m}u];
  output[outputOffset + index] = select(0.0, lag / weightSum, weightSum > 0.0);`}))}return v}};export{d as t};