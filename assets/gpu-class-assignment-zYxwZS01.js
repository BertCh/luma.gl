import{_ as e,h as t,j as n,y as r}from"./resources-CwvhEOBe.js";import{s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var a=`GPUClassAssignment`,o=4294967295,s=class{id;props;classifiedValues;constructor(e){let i=e.id??`class-assignment`;if(this.id=i,this.props=e,this.classifiedValues={classes:e.output,mask:e.mask,classCount:e.classCount},r(e.values,[`float32`],`${i} values`),r(e.breaks,[`float32`],`${i} breaks`),n(e.classCount,`${i} classCount`),n(e.output,`${i} output`),e.classCount.length<1)throw Error(`${i} classCount must hold one row`);if(e.breaks.length<2)throw Error(`${i} breaks must hold at least two rows`);if(e.output.length!==e.values.length)throw Error(`${i} output length must equal values length`);if(e.mask&&(n(e.mask,`${i} mask`),e.mask.length!==e.values.length))throw Error(`${i} mask length must equal values length`);t(i,[e.output],[e.values,e.breaks,e.classCount,e.mask])}getCommandNodes(t){let{id:n,props:r}=this,{values:s,breaks:c,classCount:l,mask:u,output:d}=r;return e(n,t,[s,c,l,u,d]),[i(t,{id:`${n}-assign`,operation:a,bindings:[{name:`values`,view:s,type:`f32`,access:`read`},{name:`breaks`,view:c,type:`f32`,access:`read`},{name:`classCount`,view:l,type:`u32`,access:`read`},...u?[{name:`mask`,view:u,type:`u32`,access:`read`}]:[],{name:`output`,view:d,type:`u32`,access:`read_write`}],invocationCount:s.length,declarations:`const BREAK_ROWS: u32 = ${c.length}u;`,body:`let value = values[valuesOffset + index];
  let count = min(classCount[classCountOffset], BREAK_ROWS - 1u);
  var assigned = ${o}u;
  // WGSL gives no NaN guarantees to comparisons, so test the exponent and mantissa bits.
  let isNaN = (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
  if (count > 0u && !isNaN && ${u?`mask[maskOffset + index] != 0u`:`true`}) {
    var below = 0u;
    for (var edge = 1u; edge < count; edge++) {
      if (breaks[breaksOffset + edge] <= value) {
        below++;
      }
    }
    assigned = below;
  }
  output[outputOffset + index] = assigned;`})]}};export{o as n,s as t};