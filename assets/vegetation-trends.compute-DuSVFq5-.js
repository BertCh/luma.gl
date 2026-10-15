import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,_ as t,d as n,h as r,n as i,t as a,y as o}from"./resources-CwvhEOBe.js";import{s}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{i as c,u as l}from"./raster-band-utils-CA-sh8xt.js";import{D as u,T as d,U as f,a as p,k as m}from"./index-Bjk18V2I.js";import{t as h}from"./summary-reader-CsgWxENf.js";import{i as g,n as _,o as v,r as y,t as b}from"./b16-common-3eXVh1Vw.js";function ee(e,t=new Float32Array(5)){return c(`getGPUChangeDetectionParameterValues`,t,5),t[0]=e.beforeSlice,t[1]=e.afterSlice,t[2]=e.epsilon??1e-6,t[3]=e.alpha??.05,t[4]=e.splitSlice??1,t}var x=3e-7,S=`
fn quietNan() -> f32 {
  // A runtime value: WGSL rejects constant-expression NaN.
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isNanValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}
`,C=`
fn logGammaPositive(argument: f32) -> f32 {
  var x = argument;
  var shiftProduct = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (x < 8.0) {
      shiftProduct *= x;
      x += 1.0;
    }
  }
  let inverse = 1.0 / x;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (x - 0.5) * log(x) - x + 0.9189385332 + series - log(shiftProduct);
}

fn betaContinuedFraction(a: f32, b: f32, x: f32) -> f32 {
  let tiny = 1e-30;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  var c = 1.0;
  var d = 1.0 - qab * x / qap;
  if (abs(d) < tiny) { d = tiny; }
  d = 1.0 / d;
  var h = d;
  for (var m = 1u; m <= 64u; m++) {
    let mf = f32(m);
    let m2 = 2.0 * mf;
    var aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (abs(d) < tiny) { d = tiny; }
    c = 1.0 + aa / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h *= delta;
    if (abs(delta - 1.0) < ${x}) {
      break;
    }
  }
  return h;
}

fn studentTTwoSidedP(t: f32, degreesOfFreedom: f32) -> f32 {
  let t2 = t * t;
  if (t2 == 0.0) {
    return 1.0;
  }
  if (t2 > 1e30) {
    return 0.0;
  }
  let a = 0.5 * degreesOfFreedom;
  let b = 0.5;
  let y = t2 / (degreesOfFreedom + t2);
  let x = degreesOfFreedom / (degreesOfFreedom + t2);
  let logX = -log(1.0 + t2 / degreesOfFreedom);
  let logFront = a * logX + b * log(y)
    - (logGammaPositive(a) + logGammaPositive(b) - logGammaPositive(a + b));
  var incomplete: f32;
  if (x < (a + 1.0) / (a + b + 2.0)) {
    incomplete = exp(logFront) * betaContinuedFraction(a, b, x) / a;
  } else {
    incomplete = 1.0 - exp(logFront) * betaContinuedFraction(b, a, y) / b;
  }
  return clamp(incomplete, 0.0, 1.0);
}
`,w=`
fn complementaryError(value: f32) -> f32 {
  let z = abs(value);
  let t = 1.0 / (1.0 + 0.5 * z);
  let polynomial = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
    + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
    + t * (-0.82215223 + t * 0.17087277))))))));
  let result = t * exp(polynomial);
  return select(2.0 - result, result, value >= 0.0);
}
`,T=`GPUChangeDetection`,E=256,D=64,O=16,k=[`tStatistic`,`tDegreesOfFreedom`,`tPValue`,`senSlope`,`mannKendallS`,`mannKendallZ`,`mannKendallP`,`significance`],A=[`difference`,`logRatio`,`percentChange`,`tStatistic`,`tDegreesOfFreedom`,`tPValue`,`senSlope`,`mannKendallZ`,`mannKendallP`,`changeMagnitude`],te=class{id;props;bandCount;significanceSource;constructor(e){this.id=e.id??`change-detection`,this.props=e;let t=this.id,{cellCount:i,sliceCount:a,output:s}=e,c=e.bandCount??1;if(this.bandCount=c,this.significanceSource=e.significanceSource??`t-test`,n(t,`cellCount`,i,1,2147483647),n(t,`bandCount`,c,1,2147483647),n(t,`sliceCount`,a,2,E),i*a*c>2147483647)throw Error(`${t} cellCount * sliceCount * bandCount must not exceed 2^31 - 1`);if(this.significanceSource!==`t-test`&&this.significanceSource!==`mann-kendall`)throw Error(`${t} significanceSource must be 't-test' or 'mann-kendall'`);if(o(e.slices,[`float32`],`${t} slices`),e.slices.length<i*a*c)throw Error(`${t} slices must hold cellCount * sliceCount * bandCount rows`);if(e.mask&&(o(e.mask,[`uint32`],`${t} mask`),e.mask.length<i))throw Error(`${t} mask must hold cellCount rows`);l(t,e.parameters,5);let u=Object.entries(s).filter(([,e])=>e);if(u.length===0)throw Error(`${t} needs at least one output`);for(let e of A)s[e]&&o(s[e],[`float32`],`${t} output.${e}`);s.mannKendallS&&o(s.mannKendallS,[`sint32`],`${t} output.mannKendallS`),s.significance&&o(s.significance,[`uint32`],`${t} output.significance`);for(let e of k)if(s[e]&&c!==1)throw Error(`${t} output.${e} requires bandCount 1`);if(s.senSlope&&a>D)throw Error(`${t} output.senSlope supports at most ${D} slices`);if((s.changeMagnitude||s.changeDirection)&&(c<2||c>O))throw Error(`${t} changeMagnitude and changeDirection need bandCount in [2, ${O}]`);s.changeDirection&&o(s.changeDirection,[c===2?`float32`:`uint32`],`${t} output.changeDirection`);for(let[e,n]of u){let r=e===`difference`||e===`logRatio`||e===`percentChange`?i*c:i;if(n.length<r)throw Error(`${t} output.${e} must hold ${r} rows`)}r(t,u.map(([,e])=>e),[e.slices,e.mask,e.parameters])}getCommandNodes(e){let{props:n,id:r,bandCount:i}=this,{output:a,cellCount:o,sliceCount:c}=n;t(r,e,[n.slices,n.mask,n.parameters,...Object.values(a)]);let l=(e,t)=>({name:e,view:t,type:`f32`,access:`read`}),u=()=>[l(`slices`,n.slices),...n.mask?[{name:`cellMask`,view:n.mask,type:`u32`,access:`read`}]:[]],d=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),f=`const SLICE_COUNT: u32 = ${c}u;
const BAND_COUNT: u32 = ${i}u;
${S}`,p=n.mask?`if (cellMask[cellMaskOffset + cell] == 0u) {
    return;
  }`:``,m=(e,t=`0u`)=>`(cell * SLICE_COUNT + ${e}) * BAND_COUNT + ${t}`,h=[];if(a.difference||a.logRatio||a.percentChange){let t=[...u(),l(`params`,n.parameters)],c=[],g=[];a.difference&&(t.push(d(`differenceOut`,a.difference,`f32`)),c.push(`differenceOut[differenceOutOffset + index] = quietNan();`),g.push(`differenceOut[differenceOutOffset + index] = after - before;`)),a.logRatio&&(t.push(d(`logRatioOut`,a.logRatio,`f32`)),c.push(`logRatioOut[logRatioOutOffset + index] = quietNan();`),g.push(`if (after > -epsilon && before > -epsilon) {
    logRatioOut[logRatioOutOffset + index] = log((after + epsilon) / (before + epsilon));
  }`)),a.percentChange&&(t.push(d(`percentChangeOut`,a.percentChange,`f32`)),c.push(`percentChangeOut[percentChangeOutOffset + index] = quietNan();`),g.push(`if (before != 0.0) {
    percentChangeOut[percentChangeOutOffset + index] = 100.0 * (after - before) / abs(before);
  }`)),h.push(s(e,{id:`${r}-two-slice`,operation:T,variant:`two-slice`,bindings:t,invocationCount:o*i,declarations:f,body:`let cell = index / BAND_COUNT;
  let band = index % BAND_COUNT;
  ${c.join(`
  `)}
  ${p}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  let epsilon = params[paramsOffset + 2u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  let before = slices[slicesOffset + ${m(`u32(beforeSlice)`,`band`)}];
  let after = slices[slicesOffset + ${m(`u32(afterSlice)`,`band`)}];
  if (isNanValue(before) || isNanValue(after)) {
    return;
  }
  ${g.join(`
  `)}`}))}if(a.changeMagnitude||a.changeDirection){let t=[...u(),l(`params`,n.parameters)],c=[],g=[];if(a.changeMagnitude&&(t.push(d(`magnitudeOut`,a.changeMagnitude,`f32`)),c.push(`magnitudeOut[magnitudeOutOffset + cell] = quietNan();`),g.push(`magnitudeOut[magnitudeOutOffset + cell] = sqrt(squaredNorm);`)),a.changeDirection){let e=i===2;t.push(d(`directionOut`,a.changeDirection,e?`f32`:`u32`)),c.push(`directionOut[directionOutOffset + cell] = ${e?`quietNan()`:`0xffffffffu`};`),g.push(e?`if (differences[0] != 0.0 || differences[1] != 0.0) {
    directionOut[directionOutOffset + cell] = atan2(differences[1], differences[0]);
  } else {
    directionOut[directionOutOffset + cell] = 0.0;
  }`:`directionOut[directionOutOffset + cell] = code;`)}h.push(s(e,{id:`${r}-multiband`,operation:T,variant:`multiband`,bindings:t,invocationCount:o,declarations:f,body:`let cell = index;
  ${c.join(`
  `)}
  ${p}
  let beforeSlice = params[paramsOffset];
  let afterSlice = params[paramsOffset + 1u];
  if (!(beforeSlice >= 0.0 && beforeSlice < f32(SLICE_COUNT) && afterSlice >= 0.0 && afterSlice < f32(SLICE_COUNT))) {
    return;
  }
  var differences: array<f32, ${i}>;
  var squaredNorm = 0.0;
  var code = 0u;
  for (var band = 0u; band < BAND_COUNT; band++) {
    let before = slices[slicesOffset + ${m(`u32(beforeSlice)`,`band`)}];
    let after = slices[slicesOffset + ${m(`u32(afterSlice)`,`band`)}];
    if (isNanValue(before) || isNanValue(after)) {
      return;
    }
    let difference = after - before;
    differences[band] = difference;
    squaredNorm += difference * difference;
    if (difference > 0.0) {
      code |= 1u << band;
    } else if (difference < 0.0) {
      code |= 1u << (16u + band);
    }
  }
  ${g.join(`
  `)}`}))}let g=a.tStatistic||a.tDegreesOfFreedom||a.tPValue,_=a.significance&&this.significanceSource===`t-test`;if(g||_){let t=[...u(),l(`params`,n.parameters)],i=[],c=[];for(let[e,n,r]of[[`tStatisticOut`,a.tStatistic,`t`],[`tDegreesOut`,a.tDegreesOfFreedom,`degreesOfFreedom`],[`tPValueOut`,a.tPValue,`pValue`]])n&&(t.push(d(e,n,`f32`)),i.push(`${e}[${e}Offset + cell] = quietNan();`),c.push(`${e}[${e}Offset + cell] = ${r};`));_&&(t.push(d(`significanceOut`,a.significance,`u32`)),i.push(`significanceOut[significanceOutOffset + cell] = 0u;`),c.push(`if (pValue < alpha) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, t > 0.0);
  }`)),h.push(s(e,{id:`${r}-t-test`,operation:T,variant:`t-test`,bindings:t,invocationCount:o,declarations:`${f}\n${C}`,body:`let cell = index;
  ${i.join(`
  `)}
  ${p}
  let alpha = params[paramsOffset + 3u];
  let splitSlice = params[paramsOffset + 4u];
  if (!(splitSlice >= 1.0 && splitSlice <= f32(SLICE_COUNT - 1u))) {
    return;
  }
  let split = u32(splitSlice);
  var count1 = 0u;
  var count2 = 0u;
  var origin1 = 0.0;
  var origin2 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${m(`slice`)}];
    if (isNanValue(value)) {
      continue;
    }
    if (slice < split) {
      if (count1 == 0u) {
        origin1 = value;
      }
      count1++;
      sum1 += value - origin1;
    } else {
      if (count2 == 0u) {
        origin2 = value;
      }
      count2++;
      sum2 += value - origin2;
    }
  }
  if (count1 < 2u || count2 < 2u) {
    return;
  }
  // Center before accumulating: constant groups have exactly zero residuals even
  // when division by their non-power-of-two sample counts is rounded by the GPU.
  let mean1 = sum1 / f32(count1);
  let mean2 = sum2 / f32(count2);
  var squares1 = 0.0;
  var squares2 = 0.0;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = slices[slicesOffset + ${m(`slice`)}];
    if (isNanValue(value)) {
      continue;
    }
    if (slice < split) {
      let residual = (value - origin1) - mean1;
      squares1 += residual * residual;
    } else {
      let residual = (value - origin2) - mean2;
      squares2 += residual * residual;
    }
  }
  let errorSquared1 = squares1 / f32(count1 - 1u) / f32(count1);
  let errorSquared2 = squares2 / f32(count2 - 1u) / f32(count2);
  let errorSquared = errorSquared1 + errorSquared2;
  let denominator = errorSquared1 * errorSquared1 / f32(count1 - 1u)
    + errorSquared2 * errorSquared2 / f32(count2 - 1u);
  if (!(errorSquared > 0.0 && denominator > 0.0)) {
    return;
  }
  let t = ((origin2 - origin1) + (mean2 - mean1)) / sqrt(errorSquared);
  let degreesOfFreedom = errorSquared * errorSquared / denominator;
  let pValue = studentTTwoSidedP(t, degreesOfFreedom);
  ${c.join(`
  `)}`}))}if(a.senSlope){let t=c*(c-1)/2,n=Math.floor(t/2)+1;h.push(s(e,{id:`${r}-sen-slope`,operation:T,variant:`sen-slope`,workgroupSize:64,bindings:[...u(),d(`senOut`,a.senSlope,`f32`)],invocationCount:o,declarations:f,body:`let cell = index;
  senOut[senOutOffset + cell] = quietNan();
  ${p}
  var validCount = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    if (!isNanValue(slices[slicesOffset + ${m(`slice`)}])) {
      validCount++;
    }
  }
  if (validCount < 2u) {
    return;
  }
  let pairCount = validCount * (validCount - 1u) / 2u;
  let heapSize = pairCount / 2u + 1u;
  var heap: array<f32, ${n}>;
  var size = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${m(`i`)}];
    if (isNanValue(valueI)) {
      continue;
    }
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${m(`j`)}];
      if (isNanValue(valueJ)) {
        continue;
      }
      let slope = (valueJ - valueI) / f32(j - i);
      if (size < heapSize) {
        var child = size;
        size++;
        heap[child] = slope;
        loop {
          if (child == 0u) { break; }
          let parent = (child - 1u) / 2u;
          if (heap[parent] >= heap[child]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[child];
          heap[child] = swapped;
          child = parent;
        }
      } else if (slope < heap[0]) {
        heap[0] = slope;
        var parent = 0u;
        loop {
          let left = 2u * parent + 1u;
          if (left >= size) { break; }
          var larger = left;
          if (left + 1u < size && heap[left + 1u] > heap[left]) { larger = left + 1u; }
          if (heap[parent] >= heap[larger]) { break; }
          let swapped = heap[parent];
          heap[parent] = heap[larger];
          heap[larger] = swapped;
          parent = larger;
        }
      }
    }
  }
  var median = heap[0];
  if (pairCount % 2u == 0u) {
    var secondLargest = heap[1];
    if (size > 2u) { secondLargest = max(secondLargest, heap[2]); }
    median = 0.5 * (heap[0] + secondLargest);
  }
  senOut[senOutOffset + cell] = median;`}))}let v=a.mannKendallS||a.mannKendallZ||a.mannKendallP,y=a.significance&&this.significanceSource===`mann-kendall`;if(v||y){let t=[...u()],i=[],c=[];y&&t.push(l(`params`,n.parameters)),a.mannKendallS&&(t.push(d(`sOut`,a.mannKendallS,`i32`)),i.push(`sOut[sOutOffset + cell] = 0;`),c.push(`sOut[sOutOffset + cell] = score;`)),a.mannKendallZ&&(t.push(d(`zOut`,a.mannKendallZ,`f32`)),i.push(`zOut[zOutOffset + cell] = quietNan();`),c.push(`zOut[zOutOffset + cell] = z;`)),a.mannKendallP&&(t.push(d(`pOut`,a.mannKendallP,`f32`)),i.push(`pOut[pOutOffset + cell] = quietNan();`),c.push(`pOut[pOutOffset + cell] = pValue;`)),y&&(t.push(d(`significanceOut`,a.significance,`u32`)),i.push(`significanceOut[significanceOutOffset + cell] = 0u;`),c.push(`if (pValue < params[paramsOffset + 3u] && score != 0) {
    significanceOut[significanceOutOffset + cell] = select(2u, 1u, score > 0);
  }`)),h.push(s(e,{id:`${r}-mann-kendall`,operation:T,variant:`mann-kendall`,workgroupSize:64,bindings:t,invocationCount:o,declarations:`${f}\n${w}`,body:`let cell = index;
  ${i.join(`
  `)}
  ${p}
  var validCount = 0u;
  var score = 0;
  var tieTerm = 0u;
  for (var i = 0u; i < SLICE_COUNT; i++) {
    let valueI = slices[slicesOffset + ${m(`i`)}];
    if (isNanValue(valueI)) {
      continue;
    }
    validCount++;
    var seenBefore = false;
    for (var k = 0u; k < i; k++) {
      if (slices[slicesOffset + ${m(`k`)}] == valueI) {
        seenBefore = true;
      }
    }
    var tied = 1u;
    for (var j = i + 1u; j < SLICE_COUNT; j++) {
      let valueJ = slices[slicesOffset + ${m(`j`)}];
      if (isNanValue(valueJ)) {
        continue;
      }
      if (valueJ > valueI) {
        score += 1;
      } else if (valueJ < valueI) {
        score -= 1;
      } else {
        tied++;
      }
    }
    if (!seenBefore && tied > 1u) {
      tieTerm += tied * (tied - 1u) * (2u * tied + 5u);
    }
  }
  if (validCount < 2u) {
    return;
  }
  let varianceNumerator = validCount * (validCount - 1u) * (2u * validCount + 5u) - tieTerm;
  var z = 0.0;
  var pValue = 1.0;
  if (varianceNumerator > 0u) {
    let standardDeviation = sqrt(f32(varianceNumerator) / 18.0);
    let correctedScore = select(select(0, score + 1, score < 0), score - 1, score > 0);
    z = f32(correctedScore) / standardDeviation;
    pValue = complementaryError(abs(z) * 0.70710678118);
  }
  ${c.join(`
  `)}`}))}return h}};function j(e){let t=Math.abs(e),n=1/(1+.5*t),r=n*Math.exp(-t*t-1.26551223+n*(1.00002368+n*(.37409196+n*(.09678418+n*(-.18628806+n*(.27886807+n*(-1.13520398+n*(1.48851587+n*(-.82215223+n*.17087277)))))))));return e>=0?r:2-r}function M(e){let t=[76.18009172947146,-86.50532032941678,24.01409824083091,-1.231739572450155,.001208650973866179,-5395239384953e-18],n=e,r=e+5.5;r-=(e+.5)*Math.log(r);let i=1.000000000190015;for(let e of t)i+=e/++n;return-r+Math.log(2.5066282746310007*i/e)}function N(e,t,n){let r=1e-30,i=1,a=1-(e+t)*n/(e+1);Math.abs(a)<r&&(a=r),a=1/a;let o=a;for(let s=1;s<=200;s++){let c=2*s,l=s*(t-s)*n/((e+c-1)*(e+c));a=1+l*a,Math.abs(a)<r&&(a=r),i=1+l/i,Math.abs(i)<r&&(i=r),a=1/a,o*=a*i,l=-(e+s)*(e+t+s)*n/((e+c)*(e+c+1)),a=1+l*a,Math.abs(a)<r&&(a=r),i=1+l/i,Math.abs(i)<r&&(i=r),a=1/a;let u=a*i;if(o*=u,Math.abs(u-1)<1e-12)break}return o}function P(e,t,n){if(n<=0)return 0;if(n>=1)return 1;let r=Math.exp(M(e+t)-M(e)-M(t)+e*Math.log(n)+t*Math.log(1-n));return n<(e+1)/(e+t+2)?r*N(e,t,n)/e:1-r*N(t,e,1-n)/t}function F(e,t){let n=e.length,r=[],i=[];for(let t=0;t<n;t++)Number.isFinite(e[t])&&(r.push(t),i.push(e[t]));let a=i.length,o=0,s=[];for(let e=0;e<a;e++)for(let t=e+1;t<a;t++)o+=Math.sign(i[t]-i[e]),s.push((i[t]-i[e])/(r[t]-r[e]));let c=new Map;for(let e of i)c.set(e,(c.get(e)??0)+1);let l=0;for(let e of c.values())l+=e*(e-1)*(2*e+5);let u=(a*(a-1)*(2*a+5)-l)/18,d=NaN,f=NaN;a>=2&&(u>0?(d=(o-Math.sign(o))/Math.sqrt(u),f=j(Math.abs(d)/Math.SQRT2)):(d=0,f=1));let p=NaN;if(s.length>0){s.sort((e,t)=>e-t);let e=s.length>>1;p=s.length%2?s[e]:(s[e-1]+s[e])/2}let m=i.filter((e,n)=>r[n]<t),h=i.filter((e,n)=>r[n]>=t),g=NaN,_=NaN;if(m.length>=2&&h.length>=2){let e=e=>e.reduce((e,t)=>e+t,0)/e.length,t=(e,t)=>e.reduce((e,n)=>e+(n-t)*(n-t),0)/(e.length-1),n=e(m),r=e(h),i=t(m,n)/m.length,a=t(h,r)/h.length,o=Math.sqrt(i+a);if(o>0){g=(r-n)/o;let e=(i+a)**2/(i**2/(m.length-1)+a**2/(h.length-1));_=P(e/2,.5,e/(e+g*g))}}return{validCount:a,mannKendallS:o,mannKendallZ:d,mannKendallP:f,senSlope:p,tStatistic:g,tPValue:_}}var ne=350,I=[`.`,`:`,`-`,`=`,`+`,`*`,`#`,`@`],re=400,ie=[`difference`,`logRatio`,`percentChange`,`tStatistic`,`tDegreesOfFreedom`,`tPValue`,`senSlope`,`mannKendallZ`,`mannKendallP`],L=`ndvi-timeseries`;async function ae(e){let t=await u(m(L,`manifest.json`),e),n=new Uint8Array(await d(m(L,`ndvi.bin`),e)),{width:r,height:i,depth:a}=t.raster,o=r*i,s=new Float32Array(o*a);for(let e=0;e<a;e++){let t=e*o;for(let r=0;r<o;r++){let i=n[t+r];s[r*a+e]=i===0?NaN:i/127.5-1}}return{manifest:t,width:r,height:i,cellCount:o,sliceCount:a,dates:t.properties.dates,meanNdvi:t.properties.scenes.map(e=>e.meanNDVI),cellMajor:s}}async function R(t){let{device:n}=t;t.setStatus(`Loading the NDVI stack`);let r=await ae(t.signal);t.signal.throwIfAborted();let{width:o,height:s,cellCount:c,sliceCount:l,manifest:u}=r,d=y([(u.bbox[0]+u.bbox[2])/2,(u.bbox[1]+u.bbox[3])/2],{boundsProjected:u.raster.boundsProjected,cellSizeM:u.raster.cellSizeM,width:o,height:s}),m=new a(n,`trend`),x=new Float32Array(r.cellMajor),S=m.createBuffer(`slices`,x),C=m.createBuffer(`mask`,new Uint32Array(c).fill(1)),w=m.createParameterBuffer(`parameters`,`float32`,5),T={};for(let e of ie)T[e]=m.createBuffer(e,c*4);let E=m.createBuffer(`mann-kendall-s`,c*4),D=m.createBuffer(`significance-t`,c*4),O=m.createBuffer(`significance-mk`,c*4),k={before:m.createBuffer(`ndvi-before`,c*4),after:m.createBuffer(`ndvi-after`,c*4)},A=m.createBuffer(`marker`,new Float32Array([0,0])),j=async t=>{let r=new e(n,{id:`trend-${t}`}),i=_(r),a=(e,t,n=c)=>i(e,t,`float32`,n);return r.add(new te({id:`change-${t}`,slices:a(`slices`,S,c*l),mask:i(`mask`,C,`uint32`,c),parameters:w.importToGraph(r),cellCount:c,sliceCount:l,significanceSource:t,output:t===`t-test`?{difference:a(`difference`,T.difference),logRatio:a(`log-ratio`,T.logRatio),percentChange:a(`percent-change`,T.percentChange),tStatistic:a(`t-statistic`,T.tStatistic),tDegreesOfFreedom:a(`t-df`,T.tDegreesOfFreedom),tPValue:a(`t-p`,T.tPValue),senSlope:a(`sen-slope`,T.senSlope),mannKendallS:i(`mk-s`,E,`sint32`,c),mannKendallZ:a(`mk-z`,T.mannKendallZ),mannKendallP:a(`mk-p`,T.mannKendallP),significance:i(`significance`,D,`uint32`,c)}:{significance:i(`significance`,O,`uint32`,c)}})),m.track(await r.compileAsync())},M={"t-test":await j(`t-test`),"mann-kendall":await j(`mann-kendall`)},N=!1,P=!0,L=performance.now(),R=!0,z=``,B=-1,V=``,H=-1,U={},W=null,se=()=>{P=!0,L=performance.now()},G=new Uint8Array(c),ce=e=>{let[n,a]=[Math.min(...e.dateWindow),Math.max(...e.dateWindow)],o=r.meanNdvi.slice(n,a+1);t.setReadout(`windowMeanSeries`,`${o[0].toFixed(2)} on ${r.dates[n]}, ${Math.min(...o).toFixed(2)} lowest, ${o[o.length-1].toFixed(2)} on ${r.dates[a]}`);let s=`${n}-${a}`;if(s!==z){z=s;for(let e=0;e<c;e++){let t=e*l,i=0;for(let e=0;e<l;e++){let o=e>=n&&e<=a?r.cellMajor[t+e]:NaN;x[t+e]=o,o===o&&i++}G[e]=i}S.write(x),B=-1}if(e.minimumValid!==B){B=e.minimumValid;let n=new Uint32Array(c),r=0;for(let t=0;t<c;t++)G[t]>=e.minimumValid&&(n[t]=1,r++);C.write(n),t.setReadout(`analysed`,`${i(r)} of ${i(c)} cells (at least ${e.minimumValid} valid dates)`)}let u=`${e.beforeSlice}-${e.afterSlice}-${s}`;if(u!==V){V=u;for(let[t,n]of[[k.before,e.beforeSlice],[k.after,e.afterSlice]]){let e=new Float32Array(c);for(let t=0;t<c;t++)e[t]=x[t*l+n];t.write(e)}}},le=e=>{w.write(ee({beforeSlice:e.beforeSlice,afterSlice:e.afterSlice,epsilon:e.epsilon,alpha:e.alpha,splitSlice:Math.min(Math.max(e.splitSlice,1),l-1)}))},K=[`senSlope`,`mannKendallZ`,`mannKendallP`,`tStatistic`,`tPValue`,`difference`],q=new h(m,`trend-read`,[...K.map(e=>({buffer:T[e],size:c*4})),{buffer:E,size:c*4},{buffer:D,size:c*4},{buffer:O,size:c*4}],e=>{if(N)return;let n=0;for(let t of K)U[t]=new Float32Array(e,n,c),n+=c*4;U.mannKendallS=new Int32Array(e,n,c),n+=c*4,U.significance=new Uint32Array(e,n,c),n+=c*4,U.significanceMannKendall=new Uint32Array(e,n,c),W={split:t.options.splitSlice,window:z},ue()}),J=e=>x.subarray(e*l,(e+1)*l),Y=e=>Array.from(e,e=>Number.isFinite(e)?I[Math.max(0,Math.min(I.length-1,Math.floor((e+.1)/1*I.length)))]:`·`).join(``);function ue(){let e=t.options,n=e.significanceSource===`mann-kendall`?U.significanceMannKendall:U.significance;if(n){let r=0,a=0,o=0;for(let e=0;e<c;e++)X(e)&&o++,n[e]===1?r++:n[e]===2&&a++;t.setReadout(`shares`,`${v(o?a/o:0,1)} declining · ${v(o?r/o:0,1)} increasing · ${v(o?1-(r+a)/o:0,1)} no significant trend (alpha ${e.alpha})`),t.setReadout(`falsePositives`,`about ${v(e.alpha,1)} of cells (${i(Math.round(o*e.alpha))}) would pass even for pure noise: alpha is not corrected for the number of cells`)}de(),Z()}let X=e=>G[e]>=t.options.minimumValid;function de(){let e=U.senSlope,n=U.mannKendallZ,r=U.mannKendallS,i=U.tStatistic;if(!e||!n||!r||!i)return;let a=f(2021),o=0,s=0,l=0,u=0,d=0,p=W?.split??t.options.splitSlice;for(let t=0;t<re;t++){let t=Math.floor(a()*c);if(!X(t))continue;let f=F(J(t),p);o++,f.mannKendallS!==r[t]&&s++,Number.isFinite(f.mannKendallZ)&&Number.isFinite(n[t])&&(l=Math.max(l,Math.abs(f.mannKendallZ-n[t]))),Number.isFinite(f.senSlope)&&Number.isFinite(e[t])&&(u=Math.max(u,Math.abs(f.senSlope-e[t]))),Number.isFinite(f.tStatistic)&&Number.isFinite(i[t])&&(d=Math.max(d,Math.abs(f.tStatistic-i[t])))}t.setReadout(`parity`,`${o} random pixels: Mann-Kendall S differs in ${s}; largest |dZ| ${l.toExponential(1)}, |dSen| ${u.toExponential(1)}, |dt| ${d.toExponential(1)}`)}function Z(){if(H<0){t.setReadout(`pixel`,`click the map`),t.setReadout(`pixelSeries`,`click the map`);return}let e=J(H),n=F(e,W?.split??t.options.splitSlice),r=H%o,i=Math.floor(H/o),[a,s]=d.cellToLngLat(r,i);t.setReadout(`pixel`,`column ${r}, row ${i} (${s.toFixed(4)}, ${a.toFixed(4)})`);let c=e=>U[e]?.[H]??NaN;t.setReadout(`pixelSeries`,`${Y(e)}  ·  Sen ${g(c(`senSlope`),4)} (CPU ${g(n.senSlope,4)}) · MK Z ${g(c(`mannKendallZ`),2)} (CPU ${g(n.mannKendallZ,2)}) · t ${g(c(`tStatistic`),2)} (CPU ${g(n.tStatistic,2)}) · p(t) ${g(c(`tPValue`),4)} (CPU ${g(n.tPValue,4)})`)}t.setReadout(`meanSeries`,Y(r.meanNdvi)+`  (${r.meanNdvi[0].toFixed(2)} in ${r.dates[0].slice(0,4)}, ${Math.min(...r.meanNdvi).toFixed(2)} lowest, ${r.meanNdvi[r.meanNdvi.length-1].toFixed(2)} in ${r.dates[r.dates.length-1].slice(0,4)})`);let Q=u.properties.scenes.map(e=>e.validFraction);t.setReadout(`acquisitionContract`,`${l} irregular summer acquisitions · ${r.dates[0]} to ${r.dates[l-1]} · ${(Math.min(...Q)*100).toFixed(1)}–${(Math.max(...Q)*100).toFixed(1)}% valid pixels per scene`),t.setReadout(`pixel`,`click the map`),t.setReadout(`pixelSeries`,`click the map`),t.setStatus(``);let $=e=>{if(!e)return-1;let[t,n]=d.lngLatToCell(e[0],e[1]),r=Math.floor(t),i=Math.floor(n);return r<0||i<0||r>=o||i>=s?-1:i*o+r};return{getCompiledGraphs:()=>[M[`t-test`],M[`mann-kendall`]],setOption(){se(),R=!0,t.requestLayers()},encode(e,n){let r=t.options;(P||n.frameIndex<2)&&(ce(r),le(r),M[`t-test`].encode(e,{parameters:void 0}),r.significanceSource===`mann-kendall`&&M[`mann-kendall`].encode(e,{parameters:void 0}),P=!1,R=!0),R&&performance.now()-L>ne&&!q.isPending&&(q.request(e),R=!1),q.flush(e)},getLayers(){let e=t.options,n={id:`trend-${e.layer}`,frame:d,opacity:e.opacity,noDataColor:[0,0,0,0]},r;if(e.layer===`significance`)r=new b({...n,values:e.significanceSource===`mann-kendall`?O:D,valueFormat:`uint32`,colormap:`category`,palette:[[140,140,140,70],[38,166,91,235],[214,69,65,235]]});else if(e.layer===`ndviBefore`||e.layer===`ndviAfter`)r=new b({...n,values:e.layer===`ndviBefore`?k.before:k.after,valueFormat:`float32`,colormap:e.ramp,valueRange:[-.1,.9]});else if(e.layer===`tPValue`||e.layer===`mannKendallP`)r=new b({...n,values:T[e.layer],valueFormat:`float32`,colormap:e.ramp,valueRange:[0,.2*e.colorScale]});else{let t=oe(e.layer)*e.colorScale;r=new b({...n,values:T[e.layer],valueFormat:`float32`,colormap:`diverging`,valueRange:[-t,t]})}let i=[r];return H>=0&&i.push(new p({id:`trend-marker`,coordinateOrigin:[d.origin[0],d.origin[1],0],positions:A,instanceCount:1,radiusPixels:7,color:t.theme()===`dark`?[255,255,255,255]:[20,20,20,255]})),i},getTooltip(e){let n=$(e.coordinate);if(n<0)return null;let i=J(n),a=[Y(i)],o=U.senSlope,s=U.mannKendallZ;o&&s&&a.push(`Sen ${g(o[n],4)} NDVI / date · MK Z ${g(s[n],2)}`);let c=i[t.options.beforeSlice],l=i[t.options.afterSlice];return a.push(`NDVI ${g(c,2)} (${r.dates[t.options.beforeSlice]}) → ${g(l,2)} (${r.dates[t.options.afterSlice]})`),a.join(`
`)},onClick(e){let n=$(e.coordinate);if(n<0)return!1;H=n;let[r,i]=d.cellToMeters(n%o,Math.floor(n/o));return A.write(Float32Array.of(r,i)),t.requestLayers(),Z(),!0},onThemeChange(){t.requestLayers()},destroy(){N=!0,q.stop(),m.destroy()}}}function oe(e){switch(e){case`senSlope`:return .04;case`mannKendallZ`:return 4;case`tStatistic`:return 8;case`difference`:return .6;case`percentChange`:return 100;case`logRatio`:return 1;default:return 1}}export{R as createVegetationTrends};