import{_ as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,w as n,y as r}from"./resources-CwvhEOBe.js";import{t as i}from"./gpu-reduction-BD8l40jM.js";import{l as a,s as o}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as s}from"./wgsl-common-BNRVzgd8.js";import{c,t as l}from"./spatial-weights-D4bd9AQ6.js";var u=`GPUSpatialWeightsTransform`,d=1.0000001,f={row:`row`,binary:`binary`,kernel:`kernel`,symmetrize:`symmetric`,double:`double`,variance:`variance`};function p(e){let t=e.weights.information,n=t?.symmetric===!0,r=e.operation===`binary`||e.operation===`double`||e.operation===`symmetrize`||e.operation===`kernel`&&typeof e.bandwidth==`number`;return{rowCount:e.weights.offsets.length-1,transform:f[e.operation],symmetric:n&&r?!0:`unknown`,hasSelfWeights:t?.hasSelfWeights,provenance:{...t?.provenance,source:u,...e.operation===`kernel`&&typeof e.bandwidth==`number`?{radius:e.bandwidth}:{}}}}var m=class{id;props;outputWeights;constructor(t){let n=t.id??`spatial-weights-transform`;if(this.id=n,this.props=t,![`row`,`binary`,`kernel`,`symmetrize`,`double`,`variance`].includes(t.operation))throw Error(`${n} unknown operation ${t.operation}`);if(c(n,t.weights),t.weights.spatialContext&&t.spatialContext&&e(n,t.weights.spatialContext,t.spatialContext),t.output&&(r(t.output,[`float32`],`${n} output`),t.output.length!==t.weights.neighbors.length))throw Error(`${n} output length must equal the slot capacity`);if(t.operation===`kernel`){if(!t.weights.distances)throw Error(`${n} 'kernel' needs weights.distances`);let{bandwidth:e}=t;if(e===void 0)throw Error(`${n} 'kernel' needs a bandwidth`);if(e!==`adaptive`&&!(Number.isFinite(e)&&e>0))throw Error(`${n} bandwidth must be 'adaptive' or > 0`);if(e===`adaptive`&&t.weights.information?.provenance?.criterion!==`knn`)throw Error(`${n} bandwidth 'adaptive' needs kNN weights`);let r=t.kernel??`triangular`;if(![`gaussian`,`triangular`,`epanechnikov`,`bisquare`,`uniform`].includes(r))throw Error(`${n} unknown kernel ${r}`)}if(t.doubleSum!==void 0&&![`one`,`rows`].includes(t.doubleSum))throw Error(`${n} doubleSum must be 'one' or 'rows'`);if(t.operation===`symmetrize`&&this.isInPlace())throw Error(`${n} 'symmetrize' needs a separate output`);this.outputWeights=l({...t.weights,weights:t.output??t.weights.weights},{information:p(t),status:t.weights.status})}isInPlace(){let{output:e,weights:t}=this.props;return!e||e.buffer===t.weights.buffer}getCommandNodes(e){let{id:r,props:c}=this,{weights:l,operation:f}=c;t(r,e,[l.offsets,l.neighbors,l.weights,l.distances,c.output]);let p=l.offsets.length-1,m=l.neighbors.length,h=this.isInPlace(),g=h?`values`:`source`,_=[{name:`offsets`,view:l.offsets,type:`u32`,access:`read`},...h?[{name:`values`,view:l.weights,type:`f32`,access:`read_write`}]:[{name:`source`,view:l.weights,type:`f32`,access:`read`},{name:`values`,view:c.output,type:`f32`,access:`read_write`}]],v=e=>`${g}[${g}Offset + ${e}]`,y=(e,t)=>`values[valuesOffset + ${e}] = ${t};`,b=`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];`,x=s;if(f===`row`)return[o(e,{id:`${r}-row`,operation:u,variant:`row`,bindings:_,invocationCount:p,declarations:x,body:`${b}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${v(`slot`)};
  }
  let scale = select(0.0, 1.0 / sum, sum > 0.0 && isFiniteFloat(sum));
  for (var slot = begin; slot < end; slot++) {
    ${y(`slot`,`${v(`slot`)} * scale`)}
  }`})];if(f===`double`||f===`variance`){let t=n(e,`${r}-partials`,`float32`,p),a=n(e,`${r}-total`,`float32`,1),s=f===`variance`,l=`var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${v(`slot`)};
    sum += value;
    squares += value * value;
  }
  let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);`,d=s?`let totalValue = total[totalOffset];
  let scale = select(0.0, f32(ROWS) / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`:`let totalValue = total[totalOffset];
  let scale = select(0.0, ${c.doubleSum===`rows`?`f32(ROWS)`:`1.0`} / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`;return[o(e,{id:`${r}-partials`,operation:u,variant:`${f}-partials`,bindings:[..._,{name:`partials`,view:t,type:`f32`,access:`read_write`}],invocationCount:p,declarations:x,body:`${b}
  ${s?`${l}
  partials[partialsOffset + index] = select(0.0, sum / norm, valid);`:`var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    sum += ${v(`slot`)};
  }
  partials[partialsOffset + index] = select(0.0, sum, isFiniteFloat(sum));`}`}),...new i({id:`${r}-reduce`,input:t,output:a,operation:`sum`}).getCommandNodes(e),o(e,{id:`${r}-apply`,operation:u,variant:`${f}-apply`,bindings:[..._,{name:`total`,view:a,type:`f32`,access:`read`}],invocationCount:p,declarations:`${x}\nconst ROWS: u32 = ${p}u;`,body:`${b}
  ${d}
  ${s?`${l}
  let rowScale = select(0.0, scale / norm, valid);`:`let rowScale = scale;`}
  for (var slot = begin; slot < end; slot++) {
    let result = ${v(`slot`)} * rowScale;
    ${y(`slot`,`select(0.0, result, isFiniteFloat(result) && result >= 0.0)`)}
  }`})]}if(f===`binary`)return[o(e,{id:`${r}-binary`,operation:u,variant:`binary`,bindings:_,invocationCount:p,body:`${b}
  for (var slot = begin; slot < end; slot++) {
    ${y(`slot`,`select(0.0, 1.0, ${v(`slot`)} > 0.0)`)}
  }`})];if(f===`kernel`){let t=c.kernel??`triangular`,n=c.bandwidth,i=[_[0],{name:`distances`,view:l.distances,type:`f32`,access:`read`},{name:`values`,view:c.output??l.weights,type:`f32`,access:`read_write`}],s={gaussian:`exp(-0.5 * z * z) * 0.3989422804014327`,triangular:`max(1.0 - z, 0.0)`,epanechnikov:`0.75 * max(1.0 - z * z, 0.0)`,bisquare:`0.9375 * max(1.0 - z * z, 0.0) * max(1.0 - z * z, 0.0)`,uniform:`select(0.0, 0.5, z <= 1.0)`}[t];return[o(e,{id:`${r}-kernel`,operation:u,variant:`kernel`,bindings:i,invocationCount:p,declarations:x,body:`${b}
  ${n===`adaptive`?`var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  // libpysal Kernel: h_i = (distance of the k-th neighbor) * eps, eps = 1.0000001.
  bandwidth = bandwidth * ${a(d)};`:`let bandwidth = ${a(n)};`}
  for (var slot = begin; slot < end; slot++) {
    let z = select(0.0, distances[distancesOffset + slot] / bandwidth, bandwidth > 0.0);
    let weight = ${s};
    ${y(`slot`,`select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0)`)}
  }`})]}return[o(e,{id:`${r}-symmetrize`,operation:u,variant:`symmetrize`,bindings:[_[0],{name:`neighbors`,view:l.neighbors,type:`u32`,access:`read`},_[1],_[2]],invocationCount:m,declarations:`const ROWS: u32 = ${p}u;`,body:`if (index >= offsets[offsetsOffset + ROWS]) {
    return;
  }
  var row = 0u;
  var high = ROWS;
  while (row + 1u < high) {
    let middle = (row + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      row = middle;
    } else {
      high = middle;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  var reverse = 0.0;
  if (neighbor < ROWS) {
    var low = offsets[offsetsOffset + neighbor];
    var upper = offsets[offsetsOffset + neighbor + 1u];
    while (low < upper) {
      let middle = (low + upper) / 2u;
      let candidate = neighbors[neighborsOffset + middle];
      if (candidate == row) {
        reverse = source[sourceOffset + middle];
        break;
      }
      if (candidate < row) {
        low = middle + 1u;
      } else {
        upper = middle;
      }
    }
  }
  values[valuesOffset + index] = 0.5 * (source[sourceOffset + index] + reverse);`})]}};export{d as n,m as t};