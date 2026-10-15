import{t as e}from"./status-BwNARr9H.js";import{d as t,y as n}from"./resources-CwvhEOBe.js";function r(e,t){let n=t??`exclude`;if(n!==`exclude`&&n!==`keep`)throw Error(`${e} islands must be 'exclude' or 'keep'`);return n}function i(e){return e.information?.transform===`row`}function a(e,t,n=`index`){return`{
    let activeRow = ${n};
    let activeBegin = min(offsets[offsetsOffset + activeRow], CAPACITY);
    let activeEnd = min(offsets[offsetsOffset + activeRow + 1u], CAPACITY);
    for (var slot = activeBegin; slot < activeEnd; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || neighbor == activeRow) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      if (!(weight > 0.0) || !(${e})) {
        continue;
      }
      ${t}
    }
  }`}function o(r,i,a=`weights`){if(n(i.offsets,[`uint32`],`${r} ${a}.offsets`),n(i.neighbors,[`uint32`],`${r} ${a}.neighbors`),n(i.weights,[`float32`],`${r} ${a}.weights`),t(r,`${a}.offsets length`,i.offsets.length,2),t(r,`${a}.neighbors length`,i.neighbors.length),i.weights.length!==i.neighbors.length)throw Error(`${r} ${a}.weights length must equal neighbors length`);if(i.distances&&(n(i.distances,[`float32`],`${r} ${a}.distances`),i.distances.length!==i.neighbors.length))throw Error(`${r} ${a}.distances length must equal neighbors length`);let o=i.offsets.length-1;i.status&&e(`${r} ${a}.status`,i.status);let{information:c}=i;if(c){if(c.rowCount!==o)throw Error(`${r} ${a}.information.rowCount must equal the row count`);if(!s.includes(c.transform))throw Error(`${r} ${a}.information.transform is unknown`);if(![!0,!1,`unknown`].includes(c.symmetric))throw Error(`${r} ${a}.information.symmetric must be a boolean or 'unknown'`)}return o}var s=[`unknown`,`raw`,`binary`,`row`,`double`,`variance`,`kernel`,`symmetric`];function c(e,t){let n=o(e,t,`output`);if(!t.status?.overflow)throw Error(`${e} output.status.overflow is required`);return n}function l(e){return[e.offsets,e.neighbors,e.weights,e.distances]}function u(e){return[e.status.overflow,e.status.requiredCount,e.status.candidateOverflow]}function d(e,t){let n={...e.status,...t.status};return{...e,information:t.information,spatialContext:t.spatialContext??e.spatialContext,status:Object.keys(n).length>0?n:void 0}}function f(e){let{rowCount:n,capacity:r}=e;t(`getGPUSpatialWeightsCapacityPlan`,`rowCount`,n),t(`getGPUSpatialWeightsCapacityPlan`,`capacity`,r);let i=Math.max(e.headroom??.25,0),a=e.requiredCount??r;if(!(Number.isSafeInteger(a)&&a>=0))throw Error(`getGPUSpatialWeightsCapacityPlan requiredCount must be >= 0`);let o=a<=r?r:Math.ceil(a*(1+i)),s=8+(e.distances?4:0),c=(n+1)*Uint32Array.BYTES_PER_ELEMENT;return{capacity:o,estimatedRequiredCount:a,limited:o<a,offsetsByteLength:c,estimatedOutputBytes:o*s+c,estimatedPeakTransientBytes:0}}export{l as a,o as c,u as i,c as l,f as n,i as o,a as r,r as s,d as t};