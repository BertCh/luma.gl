import{a as e}from"./pair-statistics-parameters-CO--_nmZ.js";import{w as t}from"./resources-CwvhEOBe.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as r}from"./column-quantiles-nodes-DnezST_r.js";var i=2**-32,a=2**-12,o=1.3489795,s=1e-30;function c(c,l){let{id:u,operation:d,values:f,mask:p,statistics:m}=l,h=f.length,g=t(c,`${u}-unit-parameters`,`float32`,6),_=t(c,`${u}-unit-quantiles`,`float32`,2),v=t(c,`${u}-unit-valid-count`,`uint32`,1),y=t(c,`${u}-unit-variance`,`float32`,1);return{nodes:[n(c,{id:`${u}-unit-parameters`,operation:d,variant:`unit-parameters`,bindings:[{name:`quantileParameters`,view:g,type:`f32`,access:`read_write`}],invocationCount:1,body:`quantileParameters[quantileParametersOffset] = 3.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  quantileParameters[quantileParametersOffset + 4u] = 0.25;
  quantileParameters[quantileParametersOffset + 5u] = 0.75;`}),...r(c,{id:`${u}-unit-quantile`,operation:d,values:f,mask:p,parameters:g,quantileCount:2,output:{quantiles:_,validCount:v}}),n(c,{id:`${u}-unit-variance`,operation:d,variant:`unit-variance`,bindings:[{name:`statistics`,view:m,type:`f32`,access:`read`},{name:`quantiles`,view:_,type:`f32`,access:`read`},{name:`unitVariance`,view:y,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${e}
const MOMENTS: u32 = ${h}u;`,body:`let variance = statistics[statisticsOffset + MOMENTS + 2u] / statistics[statisticsOffset + MOMENTS];
  let spread = (quantiles[quantilesOffset + 1u] - quantiles[quantilesOffset]) / ${o};
  let usable = isFiniteFloat(spread) && spread > 0.0;
  let robustVariance = select(variance * ${a}, spread * spread, usable);
  let unit = clamp(robustVariance, variance * ${i}, variance);
  unitVariance[unitVarianceOffset] =
    select(0.0, unit, isFiniteFloat(variance) && variance > ${s});`})],unitVariance:y}}export{c as t};