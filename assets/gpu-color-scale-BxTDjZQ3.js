import{_ as e,d as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{t as o}from"./column-classification-shared-Cj6aSTEo.js";var s=`
const NO_CLASS: u32 = 0xffffffffu;

// -0 and +0 compare equal, like d3 bisectRight. The zero test is on bits because a compiler may
// fold a float select on \`x == 0.0\` back to \`x\`.
fn getComparisonKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  if ((bits & 0x7fffffffu) == 0u) {
    return 0x80000000u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn signedPower(x: f32, exponent: f32) -> f32 {
  let magnitude = abs(x);
  if (magnitude == 0.0) {
    return 0.0;
  }
  let result = pow(magnitude, exponent);
  return select(result, -result, x < 0.0);
}

// Scale transform before normalisation. Codes: 0 linear, 1 sqrt, 2 pow, 3 log, 4 symlog.
fn transformValue(x: f32, scale: u32, exponent: f32, logFloor: f32) -> f32 {
  if (scale == 1u) {
    let magnitude = sqrt(abs(x));
    return select(magnitude, -magnitude, x < 0.0);
  }
  if (scale == 2u) {
    return signedPower(x, exponent);
  }
  if (scale == 3u) {
    var positive = x;
    if (!(positive > 0.0)) {
      positive = logFloor;
    }
    if (isNanBits(positive) || !(positive > 0.0)) {
      return getNaN();
    }
    return log(positive);
  }
  if (scale == 4u) {
    let magnitude = log(1.0 + abs(x));
    return select(magnitude, -magnitude, x < 0.0);
  }
  return x;
}

fn unpackChannel(color: u32, channel: u32) -> f32 {
  return f32((color >> (channel * 8u)) & 0xffu);
}

fn packChannel(value: f32) -> u32 {
  return u32(clamp(floor(value + 0.5), 0.0, 255.0));
}

fn blendColors(first: u32, second: u32, fraction: f32) -> u32 {
  var result = 0u;
  for (var channel = 0u; channel < 4u; channel++) {
    let a = unpackChannel(first, channel);
    let b = unpackChannel(second, channel);
    result = result | (packChannel(a + (b - a) * fraction) << (channel * 8u));
  }
  return result;
}
`,c={linear:0,sqrt:1,pow:2,log:3,symlog:4,quantize:5,threshold:6,quantile:7,ordinal:8};function l(e,t=new Float32Array(9)){if(t.length<9)throw Error(`GPUColorScale parameter target must hold 9 elements`);if(!(e.scale in c))throw Error(`GPUColorScale unknown scale '${String(e.scale)}'`);for(let[t,n]of[[`domainCount`,e.domainCount],[`paletteCount`,e.paletteCount]])if(!Number.isInteger(n)||n<0)throw Error(`GPUColorScale ${t} must be a non-negative integer`);let n=(e.noDataColor??0)>>>0;return t[0]=c[e.scale],t[1]=e.domainCount,t[2]=e.paletteCount,t[3]=e.interpolation===`linear`?1:0,t[4]=e.clamp?1:0,t[5]=n&65535,t[6]=n>>>16,t[7]=e.logFloor??1e-5,t[8]=e.exponent??1,t}function u(e,t,n,r=255){return(e&255|(t&255)<<8|(n&255)<<16|(r&255)<<24)>>>0}var d=`GPUColorScale`,f=class{id;props;constructor(e){this.id=e.id??`color-scale`,this.props=e;let r=this.id,{output:a,maximumDomainCount:o,maximumPaletteCount:s}=e;if(t(r,`maximumDomainCount`,o),t(r,`maximumPaletteCount`,s),!a.colors&&!a.classIndices&&!a.classCounts)throw Error(`${r} needs at least one output (colors, classIndices, or classCounts)`);let c=e.values.length;if(c<1)throw Error(`${r} needs at least one row`);if(i(e.values,[`float32`,`uint32`,`sint32`],`${r} values`),e.integerValues!==void 0&&![`ordinal`,`numeric`].includes(e.integerValues))throw Error(`${r} integerValues must be 'ordinal' or 'numeric'`);if(i(e.domain,[`float32`],`${r} domain`),i(e.palette,[`uint32`],`${r} palette`),i(e.parameters,[`float32`],`${r} parameters`),e.mask&&(i(e.mask,[`uint32`],`${r} mask`),e.mask.length!==c))throw Error(`${r} mask length must equal values length`);if(e.domain.length<o)throw Error(`${r} domain must hold maximumDomainCount rows`);if(e.palette.length<s)throw Error(`${r} palette must hold maximumPaletteCount rows`);if(e.parameters.length<9)throw Error(`${r} parameters must hold 9 float32 values`);if(e.domainCount&&(i(e.domainCount,[`uint32`],`${r} domainCount`),e.domainCount.length<1))throw Error(`${r} domainCount must contain one uint32 row`);for(let e of[`colors`,`classIndices`]){let t=a[e];if(t&&(i(t,[`uint32`],`${r} output.${e}`),t.length<c))throw Error(`${r} output.${e} must hold one row per value`)}if(a.classCounts&&(i(a.classCounts,[`uint32`],`${r} output.classCounts`),a.classCounts.length<s))throw Error(`${r} output.classCounts must hold maximumPaletteCount rows`);let l=[a.colors,a.classIndices,a.classCounts].filter(e=>!!e);if(new Set(l.map(e=>e.buffer)).size!==l.length)throw Error(`${r} outputs must not share buffers with each other`);n(r,l,[e.values,e.mask,e.domain,e.domainCount,e.palette,e.parameters])}getCommandNodes(t){let{props:n,id:i}=this,{output:c,maximumDomainCount:l,maximumPaletteCount:u}=n;e(i,t,[n.values,n.mask,n.domain,n.domainCount,n.palette,n.parameters,c.colors,c.classIndices,c.classCounts]);let f=n.values.length,h=n.values.format===`uint32`&&n.integerValues!==`numeric`,g=n.values.format===`float32`?void 0:n.values.format===`uint32`?`u32`:`i32`,_=c.classIndices??r(t,`${i}-classes`,`uint32`,f),v=r(t,`${i}-positions`,`float32`,f),y=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),b=`const MAXIMUM_DOMAIN_COUNT: u32 = ${l}u;
const MAXIMUM_PALETTE_COUNT: u32 = ${u}u;
${o}
${s}`,x=[];return c.classCounts&&x.push(a(t,{id:`${i}-init`,operation:d,variant:`init`,bindings:[{name:`counts`,view:c.classCounts,type:`u32`,access:`read_write`}],invocationCount:u,body:`counts[countsOffset + index] = 0u;`})),x.push(a(t,{id:`${i}-classify`,operation:d,variant:h?`classify-ordinal`:g?`classify-integer`:`classify`,bindings:[y(`values`,n.values,g??`f32`),...n.mask?[y(`rowMask`,n.mask,`u32`)]:[],y(`domain`,n.domain,`f32`),...n.domainCount?[y(`domainCountIn`,n.domainCount,`u32`)]:[],y(`params`,n.parameters,`f32`),{name:`classes`,view:_,type:`u32`,access:`read_write`},{name:`positions`,view:v,type:`f32`,access:`read_write`}],invocationCount:f,declarations:`const DOMAIN_LENGTH: u32 = ${n.domain.length}u;
${b}`,body:`classes[classesOffset + index] = NO_CLASS;
  positions[positionsOffset + index] = 0.0;
  ${n.mask?`if (rowMask[rowMaskOffset + index] == 0u) {
    return;
  }`:``}
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  if (paletteCount == 0u) {
    return;
  }
    ${h?p:m(n,g!==void 0)}`})),(c.colors||c.classCounts)&&x.push(a(t,{id:`${i}-colorize`,operation:d,variant:`colorize`,bindings:[y(`classes`,_,`u32`),y(`positions`,v,`f32`),y(`palette`,n.palette,`u32`),y(`params`,n.parameters,`f32`),...c.colors?[{name:`colors`,view:c.colors,type:`u32`,access:`read_write`}]:[],...c.classCounts?[{name:`counts`,view:c.classCounts,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:f,declarations:`const IS_ORDINAL: bool = ${h};
${b}`,body:`let classIndex = classes[classesOffset + index];
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  let isValid = classIndex != NO_CLASS && classIndex < paletteCount;
  ${c.classCounts?`if (isValid) {
    atomicAdd(&counts[countsOffset + classIndex], 1u);
  }`:``}
  ${c.colors?`let noDataColor = u32(params[paramsOffset + 5u]) | (u32(params[paramsOffset + 6u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classIndex];
    let scale = u32(max(params[paramsOffset], 0.0));
    if (!IS_ORDINAL && scale <= 4u && params[paramsOffset + 3u] == 1.0) {
      let position = positions[positionsOffset + index];
      let lower = min(u32(floor(position)), paletteCount - 1u);
      let upper = min(lower + 1u, paletteCount - 1u);
      color = blendColors(
        palette[paletteOffset + lower],
        palette[paletteOffset + upper],
        position - f32(lower)
      );
    }
  }
  colors[colorsOffset + index] = color;`:``}`})),x}},p=`let code = values[valuesOffset + index];
  if (code < paletteCount) {
    classes[classesOffset + index] = code;
  }`;function m(e,t){return`var domainCount = u32(max(params[paramsOffset + 1u], 0.0));
  ${e.domainCount?`let classTotal = domainCountIn[domainCountInOffset];
  domainCount = select(0u, min(classTotal, MAXIMUM_DOMAIN_COUNT) + 1u, classTotal > 0u);`:``}
  domainCount = min(domainCount, min(MAXIMUM_DOMAIN_COUNT, DOMAIN_LENGTH));
  if (domainCount == 0u) {
    return;
  }
  let value = ${t?`f32(values[valuesOffset + index])`:`values[valuesOffset + index]`};
  if (isNanBits(value)) {
    return;
  }
  let scale = u32(max(params[paramsOffset], 0.0));
  let isClamped = params[paramsOffset + 4u] != 0.0;
  let logFloor = params[paramsOffset + 7u];
  let exponent = params[paramsOffset + 8u];
  var classIndex = NO_CLASS;
  var rampPosition = 0.0;
  if (scale == 6u || scale == 7u) {
    // Threshold and quantile: count the inner edges domain[1 .. domainCount - 2] that are <= value.
    if (domainCount < 2u) {
      return;
    }
    let valueKey = getComparisonKey(value);
    var low = 1u;
    var high = domainCount - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getComparisonKey(domain[domainOffset + middle]) <= valueKey) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    classIndex = min(low - 1u, paletteCount - 1u);
  } else if (scale == 5u) {
    if (domainCount < 2u) {
      return;
    }
    let lowEdge = domain[domainOffset];
    let span = domain[domainOffset + domainCount - 1u] - lowEdge;
    var t = select(0.5, (value - lowEdge) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    t = clamp(t, 0.0, 1.0);
    classIndex = min(u32(floor(t * f32(paletteCount))), paletteCount - 1u);
  } else if (scale <= 4u) {
    if (domainCount < 2u) {
      return;
    }
    let firstStop = transformValue(domain[domainOffset], scale, exponent, logFloor);
    let lastStop = transformValue(domain[domainOffset + domainCount - 1u], scale, exponent, logFloor);
    let transformed = transformValue(value, scale, exponent, logFloor);
    let span = lastStop - firstStop;
    let t = select(0.5, (transformed - firstStop) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    if (domainCount > 2u && paletteCount == domainCount) {
      // Multi-stop scale: stop i maps to palette entry i, piecewise linear in transformed space.
      if (t <= 0.0) {
        rampPosition = 0.0;
      } else if (t >= 1.0) {
        rampPosition = f32(paletteCount - 1u);
      } else {
        var segment = 0u;
        for (var stopIndex = 1u; stopIndex + 1u < domainCount; stopIndex++) {
          if (transformValue(domain[domainOffset + stopIndex], scale, exponent, logFloor) <= transformed) {
            segment = stopIndex;
          }
        }
        let segmentStart = transformValue(domain[domainOffset + segment], scale, exponent, logFloor);
        let segmentEnd = transformValue(domain[domainOffset + segment + 1u], scale, exponent, logFloor);
        let width = segmentEnd - segmentStart;
        let fraction = select(0.0, clamp((transformed - segmentStart) / width, 0.0, 1.0), width > 0.0);
        rampPosition = f32(segment) + fraction;
      }
      classIndex = min(u32(floor(rampPosition)), paletteCount - 1u);
    } else {
      let clampedT = clamp(t, 0.0, 1.0);
      rampPosition = clampedT * f32(paletteCount - 1u);
      classIndex = min(u32(floor(clampedT * f32(paletteCount))), paletteCount - 1u);
    }
  } else {
    return;
  }
  classes[classesOffset + index] = classIndex;
  positions[positionsOffset + index] = rampPosition;`}export{s as i,l as n,u as r,f as t};