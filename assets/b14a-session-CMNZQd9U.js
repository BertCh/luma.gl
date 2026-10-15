import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./earth-xB16XgAa.js";import{i as a}from"./convert-colors-D-TZjU1C.js";import{N as o,_ as s,i as c,w as l}from"./resources-CwvhEOBe.js";import{t as u}from"./gpu-readback-ring-BguS3CAS.js";import{l as d,s as f}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as p,c as m,i as h,l as g,n as _,o as v,r as y,s as b,u as x}from"./raster-band-utils-CA-sh8xt.js";import{a as S,c as C,l as w,s as T,u as E}from"./raster-grid-D2bmCwkc.js";import{t as ee}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";import{n as te}from"./gpu-raster-grid-BOSvDY5A.js";import{t as ne}from"./gpu-raster-edges-DbRocSpD.js";import{H as re,T as ie,k as ae,l as oe,m as se,r as ce,y as D}from"./index-Bjk18V2I.js";import{t as O}from"./graph-buffers-B4_80RlX.js";import{t as le}from"./mode-kernels-Dxm2GWvI.js";import{l as k,n as A}from"./terrain-illumination-utils-IbBCdCef.js";var j=.72,ue=65,M={cellSizeIndex:0,northEdgeIndex:3},N=20,P=44,F=76,de=[{azimuthDegrees:225,altitudeDegrees:30},{azimuthDegrees:270,altitudeDegrees:30},{azimuthDegrees:315,altitudeDegrees:30},{azimuthDegrees:360,altitudeDegrees:30}];function I(e,t,n=new Float32Array(80)){h(`getGPUReliefShadingParameterValues`,n,80);let r=e.lights??`mdow`,i=r===`mdow`?de:r;if(i.length<1||i.length>8)throw Error(`Relief shading needs 1 to 8 lights`);let a=e.elevationStops??[];if(a.length>8)throw Error(`Relief shading supports at most 8 elevation stops`);for(let e=1;e<a.length;e++)if(!(a[e].elevation>a[e-1].elevation))throw Error(`Relief shading elevation stops must be strictly ascending`);let o=e.lightWeighting??(r===`mdow`?`aspect`:`fixed`),s=e.warmColor??[1,.95,.85],c=e.coolColor??[.8,.85,1];n.fill(0,0,80),n.set([0,0,e.zFactor??1,0,0,i.length,o===`imhof-swing`?2:o===`aspect`?1:0,e.hillshadeStrength??1,e.skyViewStrength??1,e.textureShadeStrength??.5,e.exposure??1,e.tintStrength??.3,s[0],s[1],s[2],a.length,c[0],c[1],c[2],e.imhofSwingDegrees??(o===`imhof-swing`?ue:0)]);for(let[e,t]of i.entries())n.set([t.azimuthDegrees,t.altitudeDegrees??45,t.weight??1],N+e*3);for(let[e,t]of a.entries())n.set([t.elevation,...t.color],P+e*4);return n.set([e.curvatureStrength??1,e.contrastLowElevation??0,e.contrastHighElevation??0,e.contrastStrength??0],F),E(n,t,M),n}var L=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`relief-shading`,this.props=e;let{id:t}=this,{output:n}=e,r=b(t,e.grid);if(!n.hillshade&&!n.relief&&!n.color&&!n.outputValidity&&!n.hillshadeTexture&&!n.reliefTexture)throw Error(`${t} requires at least one output`);for(let[i,a]of[[`skyViewFactor`,e.skyViewFactor],[`textureShade`,e.textureShade],[`curvature`,e.curvature],[`output.hillshade`,n.hillshade],[`output.relief`,n.relief]])k(t,i,a,`float32`,r);k(t,`output.color`,n.color,`uint32`,r),k(t,`output.outputValidity`,n.outputValidity,`uint32`,r),x(t,e.parameters,80);for(let[r,i]of[[`output.hillshadeTexture`,n.hillshadeTexture],[`output.reliefTexture`,n.reliefTexture]])g(t,r,i,[`r32float`,`rgba32float`],e.grid);C(t,e.cellSizeMode??`uniform`),m(t,e.grid,e.cellSizeMode),w(t,e.rowDirection??`south`),v(t,[n.hillshade,n.relief,n.color,n.outputValidity],[..._(e.elevation),e.parameters,e.skyViewFactor,e.textureShade,e.curvature])}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:a}=n.grid;p(t,e,n.elevation,[r.hillshadeTexture,r.reliefTexture]),s(t,e,[n.parameters,n.skyViewFactor,n.textureShade,n.curvature,r.hillshade,r.relief,r.color,r.outputValidity]);let o=i*a,u=n.borderMode??`clamp`,d=!!(r.relief||r.color||r.outputValidity||r.reliefTexture),m=y(e,t,n.elevation,n.grid,d),h=[...m.nodes],g=[`x`,`y`].map(n=>{let r=l(e,`${t}-gradient-${n}-values`,`float32`,o),s=l(e,`${t}-gradient-${n}-validity`,`uint32`,o);return h.push(...c(e,()=>new ne({id:`${t}-gradient-${n}`,width:i,height:a,input:m.band,output:r,outputValidity:s,operator:`sobel`,direction:n,borderMode:u,scale:1}).addToGraph(e))),{values:r,validity:s}}),_=r.hillshade??l(e,`${t}-hillshade`,`float32`,o),v=n.cellSizeMode??`uniform`;h.push(f(e,{id:`${t}-hillshade`,operation:`GPUReliefShading`,variant:`hillshade-${v}`,bindings:[{name:`gradientX`,view:g[0].values,type:`f32`,access:`read`},{name:`gradientY`,view:g[1].values,type:`f32`,access:`read`},{name:`gradientValidity`,view:g[0].validity,type:`u32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`hillshadeValues`,view:_,type:`f32`,access:`read_write`}],invocationCount:o,declarations:`const WIDTH: u32 = ${i}u;
const HEIGHT: u32 = ${a}u;
${T(n.rowDirection??`south`)}
const LIGHT_OFFSET: u32 = ${N}u;
const MAX_LIGHT_COUNT: u32 = 8u;
${A}
${S(v,M)}`,body:`let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let isFlat = eastGradient == 0.0 && northGradient == 0.0;
  // Downslope aspect clockwise from north.
  let aspect = atan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let aspectWeighting = settings[settingsOffset + 6u] != 0.0;
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let azimuth = settings[slot] * DEGREES_TO_RADIANS;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    var weight = settings[slot + 2u];
    if (aspectWeighting) {
      let angle = sin(aspect - azimuth);
      weight = select(angle * angle, 1.0, isFlat);
    }
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getNaN(index), shade, isValid);`})),n.imhofSwing&&h.push(fe(e,{id:`${t}-hillshade-swing`,width:i,height:a,cellSizeMode:v,rowDirection:n.rowDirection??`south`,gradientX:g[0].values,gradientY:g[1].values,gradientValidity:g[0].validity,settings:n.parameters,hillshade:_}));let b=r.relief??(r.reliefTexture?l(e,`${t}-relief`,`float32`,o):void 0),x=n.textureShade,C=3+[n.skyViewFactor,n.textureShade,n.curvature,r.relief??r.reliefTexture,r.color,r.outputValidity].filter(Boolean).length,w=d&&!!n.curvature&&C>8;w&&(x=l(e,`${t}-detail`,`float32`,o),h.push(pe(e,{id:`${t}-detail`,pixelCount:o,settings:n.parameters,textureShade:n.textureShade,curvature:n.curvature,detail:x}))),d&&h.push(R(e,{id:`${t}-compose`,pixelCount:o,elevation:m.band.storage.values,hillshade:_,settings:n.parameters,skyViewFactor:n.skyViewFactor,textureShade:x,textureShadeFolded:w,curvature:w?void 0:n.curvature,relief:b,color:r.color,validity:r.outputValidity}));for(let[n,i,a]of[[`hillshade`,_,r.hillshadeTexture],[`relief`,b,r.reliefTexture]])a&&i&&h.push(...c(e,()=>new ee({id:`${t}-${n}-texture`,input:{id:`${t}-${n}-band`,format:`float32`,storage:{kind:`buffer`,values:i}},output:a,channel:0}).addToGraph(e)));return h}};function fe(e,t){return f(e,{id:t.id,operation:`GPUReliefShading`,variant:`hillshade-swing-${t.cellSizeMode}`,bindings:[{name:`gradientX`,view:t.gradientX,type:`f32`,access:`read`},{name:`gradientY`,view:t.gradientY,type:`f32`,access:`read`},{name:`gradientValidity`,view:t.gradientValidity,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read_write`}],invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
${T(t.rowDirection)}
const LIGHT_OFFSET: u32 = ${N}u;
const MAX_LIGHT_COUNT: u32 = 8u;
${A}
${S(t.cellSizeMode,M)}`,body:`if (settings[settingsOffset + 6u] != 2.0) {
    return;
  }
  let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let aspect = atan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let swingRadians = settings[settingsOffset + 19u] * DEGREES_TO_RADIANS;
  let swingFactor = smoothstep(0.05, 0.4, length(normal.xy));
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let baseAzimuth = settings[slot] * DEGREES_TO_RADIANS;
    let azimuth = baseAzimuth + swingRadians * sin(aspect - baseAzimuth) * swingFactor;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    let weight = settings[slot + 2u];
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getNaN(index), shade, isValid);`})}function pe(e,t){let n=[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`curvatureValues`,view:t.curvature,type:`f32`,access:`read`}];return t.textureShade&&n.push({name:`textureShade`,view:t.textureShade,type:`f32`,access:`read`}),n.push({name:`detailValues`,view:t.detail,type:`f32`,access:`read_write`}),f(e,{id:t.id,operation:`GPUReliefShading`,variant:`detail`,bindings:n,invocationCount:t.pixelCount,declarations:A,body:`var detail = settings[settingsOffset + ${F}u] * curvatureValues[curvatureValuesOffset + index];
  ${t.textureShade?`detail += settings[settingsOffset + 9u] * textureShade[textureShadeOffset + index];`:``}
  detailValues[detailValuesOffset + index] = select(getNaN(index), detail, isFiniteValue(detail));`})}function R(e,t){let n=[{name:`elevationValues`,view:t.elevation,type:`f32`,access:`read`},{name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];return t.skyViewFactor&&n.push({name:`skyView`,view:t.skyViewFactor,type:`f32`,access:`read`}),t.textureShade&&n.push({name:`textureShade`,view:t.textureShade,type:`f32`,access:`read`}),t.curvature&&n.push({name:`curvatureValues`,view:t.curvature,type:`f32`,access:`read`}),t.relief&&n.push({name:`reliefValues`,view:t.relief,type:`f32`,access:`read_write`}),t.color&&n.push({name:`colorValues`,view:t.color,type:`u32`,access:`read_write`}),t.validity&&n.push({name:`validityValues`,view:t.validity,type:`u32`,access:`read_write`}),f(e,{id:t.id,operation:`GPUReliefShading`,variant:`compose`,bindings:n,invocationCount:t.pixelCount,declarations:`const STOP_OFFSET: u32 = ${P}u;
const CONTRAST_PIVOT: f32 = ${d(j)};
const MAX_STOP_COUNT: u32 = 8u;
${A}
fn getSettingsColor(slot: u32) -> vec3<f32> {
  return vec3<f32>(settings[settingsOffset + slot], settings[settingsOffset + slot + 1u],
    settings[settingsOffset + slot + 2u]);
}
fn getElevationColor(elevation: f32) -> vec3<f32> {
  let stopCount = min(u32(max(settings[settingsOffset + 15u], 0.0)), MAX_STOP_COUNT);
  if (stopCount == 0u) { return vec3<f32>(0.85); }
  var color = getSettingsColor(STOP_OFFSET + 1u);
  if (elevation <= settings[settingsOffset + STOP_OFFSET]) { return color; }
  for (var stop = 1u; stop < stopCount; stop++) {
    let slot = STOP_OFFSET + stop * 4u;
    let lowerElevation = settings[settingsOffset + slot - 4u];
    let upperElevation = settings[settingsOffset + slot];
    color = getSettingsColor(slot + 1u);
    if (elevation < upperElevation) {
      let blend = (elevation - lowerElevation) / (upperElevation - lowerElevation);
      return mix(getSettingsColor(slot - 3u), color, blend);
    }
  }
  return color;
}`,body:`let hillshade = hillshadeValues[hillshadeValuesOffset + index];
  var shade = mix(1.0, hillshade, settings[settingsOffset + 7u]);
  var isValid = isFiniteValue(hillshade);
  ${t.skyViewFactor?`let skyViewFactor = skyView[skyViewOffset + index];
  isValid = isValid && isFiniteValue(skyViewFactor);
  shade *= mix(1.0, skyViewFactor, settings[settingsOffset + 8u]);`:``}
  ${t.textureShade?`let textureValue = textureShade[textureShadeOffset + index];
  isValid = isValid && isFiniteValue(textureValue);
  shade += ${t.textureShadeFolded?``:`settings[settingsOffset + 9u] * `}textureValue;`:``}
  ${t.curvature?`let curvatureValue = curvatureValues[curvatureValuesOffset + index];
  isValid = isValid && isFiniteValue(curvatureValue);
  shade += settings[settingsOffset + ${F}u] * curvatureValue;`:``}
  let contrastStrength = settings[settingsOffset + ${F+3}u];
  if (contrastStrength != 0.0) {
    let low = settings[settingsOffset + ${F+1}u];
    let high = settings[settingsOffset + ${F+2}u];
    let elevation = elevationValues[elevationValuesOffset + index];
    let ramp = select(select(0.0, 1.0, elevation >= low), smoothstep(low, high, elevation), high > low);
    shade = mix(CONTRAST_PIVOT, shade, mix(1.0 - contrastStrength, 1.0, ramp));
    isValid = isValid && isFiniteValue(shade);
  }
  let relief = clamp(shade * settings[settingsOffset + 10u], 0.0, 1.0);
  ${t.relief?`reliefValues[reliefValuesOffset + index] = select(getNaN(index), relief, isValid);`:``}
  ${t.color?`let tintStrength = settings[settingsOffset + 11u];
  let lightness = clamp(2.0 * hillshade - 1.0, -1.0, 1.0);
  var tint = mix(vec3<f32>(1.0), getSettingsColor(12u), lightness * tintStrength);
  if (lightness < 0.0) {
    tint = mix(vec3<f32>(1.0), getSettingsColor(16u), -lightness * tintStrength);
  }
  let base = getElevationColor(elevationValues[elevationValuesOffset + index]);
  let color = clamp(base * relief * tint, vec3<f32>(0.0), vec3<f32>(1.0));
  colorValues[colorValuesOffset + index] = select(0u, pack4x8unorm(vec4<f32>(color, 1.0)), isValid);`:``}
  ${t.validity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})}var z=2*Math.PI*i;function B(e){let t=e*Math.PI/180;return z/(2*Math.PI)*Math.log(Math.tan(Math.PI/4+t/2))}function V(e){return(2*Math.atan(Math.exp(e/(z/(2*Math.PI))))-Math.PI/2)*180/Math.PI}async function H(e,t){let n=e.raster;if(!n)throw Error(`alps-dem has no raster`);let r=n.spec,i=r.boundsMercator,a=r.cellSizeMercatorM,o=await ie(ae(e.info.id,String(r.file)),t),s=await createImageBitmap(new Blob([o],{type:`image/png`}),{premultiplyAlpha:`none`,colorSpaceConversion:`none`}),{width:c,height:l}=s,u=new OffscreenCanvas(c,l).getContext(`2d`,{willReadFrequently:!0});if(!u)throw Error(`2D canvas unavailable for raster decoding`);u.drawImage(s,0,0),s.close();let d=u.getImageData(0,0,c,l).data,f=new Uint32Array(d.buffer.slice(0)),[p,m,h,g]=n.bounds,_=[(p+h)/2,(m+g)/2],v=new re(_),[y,b]=v.project(p,m),[x,S]=v.project(h,g),C=te({width:c,height:l,cellSize:a,origin:[i[0],i[3]]}),w=V((i[1]+i[3])/2),T=e.properties,E=Object.entries(T.observers??{}).map(([e,t])=>({name:e,longitude:t.lon,latitude:t.lat,elevationMeters:t.elevationM}));return{width:c,height:l,pixelCount:c*l,packedWords:f,cpuElevation:n.values,bounds:[y,b,x,S],origin:_,projection:v,lngLatBounds:[p,m,h,g],mercatorCellSize:a,grid:C,groundCellSize:a*Math.cos(w*Math.PI/180),boundsMercator:i,observers:E,getPixelCoordinates(e,t){let n=(e+180)/360*z-z/2,r=B(t);return[(n-i[0])/a,(i[3]-r)/a]},getPixel(e,t){let n=(e+180)/360*z-z/2,r=B(t),o=Math.floor((n-i[0])/a),s=Math.floor((i[3]-r)/a);return o<0||s<0||o>=c||s>=l?null:[o,s]},getLongitudeLatitude(e,t){let n=i[0]+(e+.5)*a,r=i[3]-(t+.5)*a;return[(n+z/2)/z*360-180,V(r)]},getGroundCellSize:e=>{let t=i[3]-(e+.5)*a;return a*Math.cos(V(t)*Math.PI/180)}}}var U={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},W={...U,blendColorSrcFactor:`dst`,blendColorDstFactor:`zero`,blendAlphaSrcFactor:`zero`,blendAlphaDstFactor:`one`},me=4,he=1,G=80,ge=`
struct LayerStyle {
  minimum: vec2<f32>,
  maximum: vec2<f32>,
  gridSize: vec2<u32>,
  lightStrength: f32,
  opacity: f32,
  lightGain: f32,
  lightFloor: f32,
  // Swipe compare: 0 not compared, 1 side a, 2 side b, 3 hidden entirely.
  compareSide: u32,
  compareDivider: f32,
  hatchColor: vec4<f32>,
  hatchEnabled: u32,
  devicePixelRatio: f32,
  multiply: u32,
  hatchPitch: f32,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> cellColors: array<u32>;
@group(0) @binding(auto) var<storage, read> cellLight: array<f32>;

fn projectCell(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteCell(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}

// True when a fragment lies on the hidden side of the swipe divider (fragment x is in device pixels).
fn isCompareHidden(fragmentPosition: vec4<f32>) -> bool {
  let side = layerStyle.compareSide;
  if (side == 0u) {
    return false;
  }
  if (side == 3u) {
    return true;
  }
  let isLeft = fragmentPosition.x < layerStyle.compareDivider;
  return select(isLeft, !isLeft, side == 1u);
}

// Coverage of the 45 degree hatch stripes at a fragment: pitch and width are in CSS pixels.
fn getHatchCoverage(fragmentPosition: vec4<f32>) -> f32 {
  let ratio = max(layerStyle.devicePixelRatio, 1e-3);
  let spacing = max(layerStyle.hatchPitch, 1.0);
  let phase = (fragmentPosition.x + fragmentPosition.y) / ratio * 0.7071067811865476 / spacing;
  let fraction = fract(phase);
  let stripeDistance = min(fraction, 1.0 - fraction) * spacing;
  let halfWidth = ${he.toFixed(1)} * 0.5;
  let softness = 0.5 / ratio;
  return clamp((halfWidth + softness - stripeDistance) / (2.0 * softness), 0.0, 1.0);
}

// Multiply blending needs mix(white, rgb, a): the blend state multiplies it into the target.
fn finishColor(color: vec4<f32>) -> vec4<f32> {
  if (layerStyle.multiply != 0u) {
    return vec4<f32>(mix(vec3<f32>(1.0), color.rgb, color.a), color.a);
  }
  return color;
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  var output: VertexOutput;
  output.position = projectCell(mix(layerStyle.minimum, layerStyle.maximum, corner));
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  if (isCompareHidden(input.position)) {
    discard;
  }
  let columns = i32(layerStyle.gridSize.x);
  let rows = i32(layerStyle.gridSize.y);
  let column = clamp(i32(floor(input.local.x * f32(columns))), 0, columns - 1);
  // Row 0 is the north edge (the raster the contributors read), so flip the row.
  let row = clamp(rows - 1 - i32(floor(input.local.y * f32(rows))), 0, rows - 1);
  let index = u32(row * columns + column);
  var color = unpack4x8unorm(cellColors[index]);
  if (color.a <= 0.0) {
    if (layerStyle.hatchEnabled == 0u) {
      discard;
    }
    let coverage = getHatchCoverage(input.position) * layerStyle.hatchColor.a;
    if (coverage <= 0.0) {
      discard;
    }
    return finishColor(vec4<f32>(layerStyle.hatchColor.rgb, coverage * layerStyle.opacity));
  }
  if (layerStyle.lightStrength > 0.0) {
    var light = cellLight[index];
    if (isNonFiniteCell(light)) {
      light = 0.0;
    }
    let lit = clamp(light * layerStyle.lightGain, 0.0, 1.0);
    color = vec4<f32>(color.rgb * mix(1.0, mix(layerStyle.lightFloor, 1.0, lit), layerStyle.lightStrength), color.a);
  }
  return finishColor(vec4<f32>(color.rgb, color.a * layerStyle.opacity));
}
`,K=class extends n{static layerName=`ColorRasterLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:U};getAttributeManager(){return null}getBindings(e){return{cellColors:this.props.colors,cellLight:this.props.light??e}}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:G,usage:r.UNIFORM|r.COPY_DST}),i=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST});this.setState({model:null,styleBuffer:n,placeholderBuffer:i});let o=new a(t,{...this.getShaders({modules:[e],source:ge}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:{...this.getBindings(i),layerStyle:n},parameters:this.props.blendMode===`multiply`?W:U});this.setState({model:o})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t&&n&&r&&t.setBindings({...this.getBindings(r),layerStyle:n})}getModels(){let e=this.state.model;return e?[e]:[]}_drawLayer(e){super._drawLayer({...e,parameters:{...e.parameters,...this.props.blendMode===`multiply`?W:U}})}getCompareUniforms(){let{compareSide:e}=this.props,t=oe();if(!e||!t)return{side:0,divider:0};if(t.showing!==`both`)return{side:t.showing===e?0:3,divider:0};let n=this.context?.viewport,r=this.getDevicePixelRatio(),i=Math.min(Math.max(t.position,0),1),a=((n?.x??0)+i*(n?.width??1))*r;return{side:e===`a`?1:2,divider:a}}getDevicePixelRatio(){try{return this.context.device.getDefaultCanvasContext().cssToDeviceRatio()}catch{return 1}}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;let{bounds:r,gridSize:i,lightStrength:a=0,alpha:o=1,lightGain:s=1,lightFloor:c=0,noDataHatch:l}=this.props,u=new ArrayBuffer(G),d=new Float32Array(u),f=new Uint32Array(u);d.set([r[0],r[1],r[2],r[3]]),f.set(i,4),d[6]=this.props.light?a:0,d[7]=o,d[8]=s,d[9]=c;let p=this.getCompareUniforms();f[10]=p.side,d[11]=p.divider,l&&(d.set([l[0]/255,l[1]/255,l[2]/255,(l[3]??255)/255],12),f[16]=1),d[17]=this.getDevicePixelRatio(),f[18]=this.props.blendMode===`multiply`?1:0,d[19]=me,n.write(new Uint8Array(u)),t.setInstanceCount(1),t.draw(e)}finalizeState(e){let t=this.state;t.model?.destroy(),t.styleBuffer?.destroy(),t.placeholderBuffer?.destroy(),this.setState({model:null,styleBuffer:null,placeholderBuffer:null}),super.finalizeState(e)}};function q(e,t){e.write(Float32Array.of(t.mode===`ramp`?0:t.mode===`aspect`?1:2,se[t.ramp],t.low,t.high,t.alpha,t.fadeMiddle?1:0,t.gamma??1,0))}var _e=.62,ve=.95;async function J(e,t,n,r,i,a,s=[]){let c=new o(t,{id:`${n}-paint`}),l=r.format===`float32`,u=Math.max(1,s.length),d=(s.length>0?s:[[0,0,0]]).map(([e,t,n])=>`vec3<f32>(${(e/255).toFixed(4)}, ${(t/255).toFixed(4)}, ${(n/255).toFixed(4)})`).join(`, `);return le(c,{id:`${n}-colorize`,invocationCount:r.length,bindings:[{name:`source`,view:O(c,`source`,r.buffer,r.format,r.length),type:l?`f32`:`u32`,access:`read`},{name:`paint`,view:i.importToGraph(c),type:`f32`,access:`read`},{name:`colors`,view:O(c,`colors`,a,`uint32`,r.length),type:`u32`,access:`read_write`}],declarations:`
${D()}
var<private> CLASS_COLORS: array<vec3<f32>, ${u}> = array<vec3<f32>, ${u}>(${d});
const CLASS_COUNT: u32 = ${s.length}u;
const ASPECT_SATURATION: f32 = ${_e};
const ASPECT_VALUE: f32 = ${ve};
fn isBad(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}
fn aspectChannel(n: f32, hue: f32) -> f32 {
  let shifted = n + hue * 6.0;
  let k = shifted - 6.0 * floor(shifted / 6.0);
  return ASPECT_VALUE - ASPECT_VALUE * ASPECT_SATURATION * clamp(min(k, 4.0 - k), 0.0, 1.0);
}`,body:`
  let mode = u32(paint[paintOffset]);
  let rampIndex = u32(paint[paintOffset + 1u]);
  let low = paint[paintOffset + 2u];
  let high = paint[paintOffset + 3u];
  let alpha = paint[paintOffset + 4u];
  var color = vec4<f32>(0.0);
  ${l?`
  let value = source[sourceOffset + index];
  if (!isBad(value)) {
    if (mode == 1u) {
      if (value >= 0.0) {
        let hue = fract(value / 360.0);
        color = vec4<f32>(aspectChannel(5.0, hue), aspectChannel(3.0, hue), aspectChannel(1.0, hue), alpha);
      } else {
        color = vec4<f32>(0.78, 0.78, 0.78, alpha);
      }
    } else {
      let t = clamp((value - low) / max(high - low, 1e-30), 0.0, 1.0);
      let shaped = pow(t, max(paint[paintOffset + 6u], 0.05));
      var cellAlpha = alpha;
      if (paint[paintOffset + 5u] > 0.5) {
        cellAlpha = alpha * mix(0.18, 1.0, abs(2.0 * t - 1.0));
      }
      color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, shaped), cellAlpha);
    }
  }`:`
  let code = source[sourceOffset + index];
  if (mode == 2u) {
    if (code >= 1u && code <= CLASS_COUNT) {
      color = vec4<f32>(CLASS_COLORS[code - 1u], alpha);
    }
  } else if (code > 0u) {
    // Code rasters shown on a ramp (for example geomorphon ternary patterns); 0 is nodata.
    let t = clamp((f32(code) - low) / max(high - low, 1e-30), 0.0, 1.0);
    color = vec4<f32>(spatialAnalysisSampleRamp(rampIndex, t), alpha);
  }`}
  colors[colorsOffset + index] = pack4x8unorm(color);`}),e.track(await c.compileAsync())}var Y=1024,X=32,ye=350,Z=class{graphs;dirty=!0;runCount=0;lastRun=-1/0;seenElevationVersion=-1;seenDependencyRuns;constructor(e,t,n,r=[],i=0,a=()=>{}){this.id=e,this.writeParameters=n,this.dependencies=r,this.minIntervalMs=i,this.releaseResources=a,this.graphs=t,this.seenDependencyRuns=r.map(()=>-1)}markDirty(){this.dirty=!0}isPending(e){return this.dirty||this.seenElevationVersion!==e||this.dependencies.some((t,n)=>t.runCount!==this.seenDependencyRuns[n]||t.isPending(e))}run(e,t,n){for(let r of this.dependencies)r.run(e,t,n);if(!(this.dirty||this.seenElevationVersion!==n||this.dependencies.some((e,t)=>e.runCount!==this.seenDependencyRuns[t]))||t-this.lastRun<this.minIntervalMs)return!1;this.writeParameters();for(let t of this.graphs)t.encode(e,{parameters:void 0});return this.dirty=!1,this.seenElevationVersion=n,this.seenDependencyRuns=this.dependencies.map(e=>e.runCount),this.runCount++,this.lastRun=t,!0}release(){this.releaseResources()}},be=class{graph;owned=[];buffers=new Map;constructor(e,t){this.session=e,this.id=t,this.graph=new o(e.device,{id:`terrain-${t}`})}newGraph(e){return new o(this.session.device,{id:`terrain-${this.id}-${e}`})}elevation(e=this.graph){return this.session.importElevation(e)}buffer(e,t){let n=this.session.resources.createBuffer(`${this.id}-${e}`,t);return this.owned.push({destroy:()=>this.session.resources.release(n)}),this.buffers.set(e,n),n}floats(e,t=this.session.grid.pixelCount,n=this.graph){return O(n,e,this.buffer(e,t*4),`float32`,t)}words(e,t=this.session.grid.pixelCount,n=this.graph){return O(n,e,this.buffer(e,t*4),`uint32`,t)}reimport(e,t,n){let r=this.buffers.get(e);if(!r)throw Error(`Unknown product buffer ${e}`);return O(n,e,r,t,r.byteLength/4)}settings(e,t=this.graph){let n=this.session.resources.createParameterBuffer(`${this.id}-settings-${this.owned.length}`,`float32`,e);return this.owned.push({destroy:()=>this.session.resources.release(n)}),{parameters:n,view:n.importToGraph(t)}}async finishStage(e){let{resources:t}=this.session,n=await Promise.all((e.graphs??[this.graph]).map(async e=>t.track(await e.compileAsync()))),r=new Z(this.id,n,e.write,e.dependencies??[],e.minIntervalMs??0,()=>{for(let e of n)t.release(e);for(let e=this.owned.length-1;e>=0;e--)this.owned[e].destroy()});return this.session.registerStage(r),r}getBuffer(e){let t=this.buffers.get(e);if(!t)throw Error(`Unknown product buffer ${e}`);return t}async finish(e){let t=await this.finishStage(e),n=this.getBuffer(e.value);return{id:this.id,configKey:``,stage:t,value:{buffer:n,format:e.format},palette:e.palette,release:()=>{this.session.unregisterStage(t),t.release()},paintGraph:null}}},Q=class{ring;pendingFlag=!1;stopped=!1;constructor(e,t,n){this.ring=e.track(new u(e.device,{id:t,byteLength:n}))}get pending(){return this.pendingFlag}stop(){this.stopped=!0}request(e,t,n){if(this.pendingFlag||this.stopped)return!1;let r=this.ring.tryAcquire();if(!r)return!1;let i=0;for(let n of t)e.copyBufferToBuffer({sourceBuffer:n.buffer,sourceOffset:n.offset,destinationBuffer:r.buffer,destinationOffset:i,size:n.size}),i+=n.size;return r.markEncoded({byteOffset:0,byteLength:i}),this.pendingFlag=!0,new Promise(e=>setTimeout(e,0)).then(()=>r.read()).then(e=>{this.stopped||n(e.slice())}).catch(()=>{}).finally(()=>{this.pendingFlag=!1}),!0}},xe=class{device;resources;grid;elevationBuffer;validityBuffer;colorsBuffer;onStats=null;describeHover=({value:e})=>String(e);ctx;paintParameters;statsReader;hoverReader;bulkReader=null;builds=new Map;stages=new Set;current=null;paint={mode:`ramp`,ramp:`viridis`,low:0,high:1,alpha:1};paintDirty=!0;statsDirty=!1;lastComputeTime=0;elevationVersion=0;hoverIndex=-1;hoverCache=null;underlay=null;ground=null;destroyed=!1;constructor(e,t,n){this.ctx=e,this.device=e.device,this.resources=t,this.grid=n,this.elevationBuffer=t.createBuffer(`elevation`,n.pixelCount*4),this.validityBuffer=t.createBuffer(`validity`,n.pixelCount*4),this.colorsBuffer=t.createBuffer(`colors`,n.pixelCount*4),this.paintParameters=t.createParameterBuffer(`paint`,`float32`,8),this.statsReader=new Q(t,`terrain-stats`,n.pixelCount*4),this.hoverReader=new Q(t,`terrain-hover`,8)}readBulk(e,t,n){return this.bulkReader??=new Q(this.resources,`terrain-bulk`,this.grid.pixelCount*4),this.bulkReader.request(e,[{buffer:t,offset:0,size:this.grid.pixelCount*4}],e=>{this.destroyed||n(e)})}registerStage(e){this.stages.add(e)}unregisterStage(e){this.stages.delete(e)}createBuild(e,t,n,r,i){return{id:e,configKey:``,stage:t,value:{buffer:n,format:r},release:()=>{},palette:i,paintGraph:null}}createStaticBuild(e,t,n,r){return this.createBuild(e,new Z(`${e}-static`,[],()=>{}),t,n,r)}importElevation(e){return{id:`elevation`,format:`float32`,storage:{kind:`buffer`,values:O(e,`elevation`,this.elevationBuffer,`float32`,this.grid.pixelCount)},validity:O(e,`validity`,this.validityBuffer,`uint32`,this.grid.pixelCount)}}builder(e){return new be(this,e)}elevationChanged(){this.elevationVersion++,this.statsDirty=!0,this.hoverCache=null}getBuild(e,t){let n=this.builds.get(e);return n&&n.configKey===t?n:void 0}addBuild(e,t){e.configKey=t;let n=this.builds.get(e.id);return n&&n!==e&&(this.current===n&&(this.current=null),this.releaseBuild(n)),this.builds.set(e.id,e),e}releaseBuild(e){e.paintGraph&&this.resources.release(e.paintGraph),e.paintGraph=null,e.release()}discardBuild(e){this.releaseBuild(e)}async prepareBuild(e){if(!e.paintGraph&&e.value.format!==`rgba8`){let t=e.value.format;e.paintGraph=await J(this.resources,this.device,e.id,{buffer:e.value.buffer,format:t,length:this.grid.pixelCount},this.paintParameters,this.colorsBuffer,e.palette)}}activate(e,t){this.current=e,this.hoverCache=null,this.setPaint(t),this.statsDirty=e.value.format!==`rgba8`,this.lastComputeTime=performance.now()}setPaint(e){this.paint={...this.paint,...e},q(this.paintParameters,this.paint),this.paintDirty=!0}get active(){return this.current}markDirty(e=this.current){e&&(e.stage.markDirty(),this.statsDirty=!0,this.lastComputeTime=performance.now())}markAllDirty(){for(let e of this.stages)e.markDirty();this.statsDirty=!0,this.lastComputeTime=performance.now()}setGround(e){this.ground=e}async enableUnderlay(){if(this.underlay)return;let{grid:e,resources:t,device:n}=this,r=t.createParameterBuffer(`underlay-settings`,`float32`,80);r.write(I({lights:`mdow`,lightWeighting:`aspect`,hillshadeStrength:1,skyViewStrength:0,textureShadeStrength:0,tintStrength:0,exposure:1,elevationStops:[]},e.grid));let i=t.createBuffer(`underlay-hillshade`,e.pixelCount*4),a=t.createBuffer(`underlay-colors`,e.pixelCount*4),s=t.createParameterBuffer(`underlay-paint`,`float32`,8);q(s,{mode:`ramp`,ramp:`grayscale`,low:.05,high:.95,alpha:1});let c=new o(n,{id:`terrain-underlay`});c.add(new L({id:`underlay-shading`,grid:e.grid,elevation:this.importElevation(c),parameters:r.importToGraph(c),cellSizeMode:`web-mercator`,rowDirection:`south`,output:{hillshade:O(c,`hillshade`,i,`float32`,e.pixelCount)}}));let l=t.track(await c.compileAsync()),u=await J(t,n,`underlay`,{buffer:i,format:`float32`,length:e.pixelCount},s,a);this.underlay={graphs:[l],paintGraph:u,colors:a,version:-1}}getCompiledGraphs(){let e=[];this.underlay&&e.push(...this.underlay.graphs,this.underlay.paintGraph);for(let t of this.stages)e.push(...t.graphs);for(let t of this.builds.values())t.paintGraph&&e.push(t.paintGraph);return e}async measure(){let e=this.current;if(!e)return[];let t=[],n=[...$(e.stage)].flatMap(e=>e.graphs);for(let[r,i]of n.entries()){let n=await ce(this.device,i,{parameters:void 0,completionBuffer:e.value.buffer,runs:5,warmUpRuns:1,repetitions:2,signal:this.ctx.signal});t.push({id:`${e.id}-${r}`,milliseconds:n.milliseconds,method:n.method})}return t}encode(e){if(this.destroyed)return;let t=performance.now();if(this.underlay&&this.underlay.version!==this.elevationVersion){for(let t of this.underlay.graphs)t.encode(e,{parameters:void 0});this.underlay.paintGraph.encode(e,{parameters:void 0}),this.underlay.version=this.elevationVersion}let n=this.current;if(!n)return;n.stage.run(e,t,this.elevationVersion)&&(this.paintDirty=!0,this.statsDirty=!0,this.lastComputeTime=t),this.paintDirty&&n.paintGraph&&(n.paintGraph.encode(e,{parameters:void 0}),this.paintDirty=!1);let r=!n.stage.isPending(this.elevationVersion)&&t-this.lastComputeTime>ye;if(this.statsDirty&&r&&!this.paintDirty&&n.value.format!==`rgba8`&&this.statsReader.request(e,[{buffer:n.value.buffer,offset:0,size:this.grid.pixelCount*4}],e=>{this.destroyed||this.current!==n||n.value.format!==`rgba8`&&this.onStats?.(n.id,Se(e,n.value.format))})&&(this.statsDirty=!1),r&&this.hoverIndex>=0&&!this.hoverReader.pending&&n.value.format!==`rgba8`){let t=this.hoverCache;if(!t||t.index!==this.hoverIndex||t.version!==this.elevationVersion){let t=this.hoverIndex,r=this.elevationVersion;this.hoverReader.request(e,[{buffer:n.value.buffer,offset:t*4,size:4},{buffer:this.elevationBuffer,offset:t*4,size:4}],e=>{if(this.current!==n)return;let i=new Uint32Array(e.buffer,e.byteOffset,2),a=n.value.format===`uint32`?i[0]:new Float32Array(e.buffer,e.byteOffset,1)[0],o=new Float32Array(e.buffer,e.byteOffset+4,1)[0],s=t%this.grid.width,c=Math.floor(t/this.grid.width);this.hoverCache={index:t,version:r,buildId:n.id,text:this.describeHover({value:a,elevation:o,column:s,row:c})}})}}}getTooltip(e){if(!e.coordinate)return this.hoverIndex=-1,null;let t=this.grid.getPixel(e.coordinate[0],e.coordinate[1]);if(!t)return this.hoverIndex=-1,null;this.hoverIndex=t[1]*this.grid.width+t[0];let n=this.hoverCache;if(n&&n.version===this.elevationVersion&&n.buildId===this.current?.id){let e=Math.abs(n.index%this.grid.width-t[0]),r=Math.abs(Math.floor(n.index/this.grid.width)-t[1]);return e<=3&&r<=3?n.text:null}return null}getLayers(e){let{grid:t}=this,n=[],r={coordinateOrigin:[t.origin[0],t.origin[1],0],gridSize:[t.width,t.height],bounds:t.bounds};return e.underlay&&this.ground?n.push(this.ground.getLayer({opacity:e.underlayAlpha})):e.underlay&&this.underlay&&n.push(new K({...r,id:`terrain-underlay`,colors:this.underlay.colors,alpha:e.underlayAlpha})),this.current&&e.showProduct!==!1&&n.push(new K({...r,id:`terrain-product`,colors:this.current.value.format===`rgba8`?this.current.value.buffer:this.colorsBuffer,alpha:e.alpha,light:e.light?.buffer??null,lightStrength:e.light?.strength??0,lightGain:e.light?.gain??1,lightFloor:e.light?.floor??0})),n}destroy(){this.destroyed=!0,this.statsReader.stop(),this.hoverReader.stop(),this.bulkReader?.stop()}};function Se(e,t){if(t===`uint32`){let t=new Uint32Array(e.buffer,e.byteOffset,e.byteLength>>2),n=new Float64Array(X),r=0;for(let e of t)e>0&&e<X&&(n[e]++,r++);return{kind:`classes`,total:r,counts:n}}let n=new Float32Array(e.buffer,e.byteOffset,e.byteLength>>2),r=1/0,i=-1/0,a=0,o=0;for(let e of n)Number.isFinite(e)&&(e<r&&(r=e),e>i&&(i=e),a+=e,o++);let s=new Uint32Array(Y);if(o>0&&i>r){let e=(Y-1)/(i-r);for(let t of n)Number.isFinite(t)&&s[Math.floor((t-r)*e)]++}else s[0]=o;return{kind:`float`,count:o,min:r,max:i,mean:o>0?a/o:NaN,histogram:s,quantile(e){if(o===0)return NaN;let t=Math.min(Math.max(e,0),1)*o,n=0;for(let e=0;e<Y;e++)if(n+=s[e],n>=t)return r+(e+.5)/Y*(i-r);return i}}}function $(e,t=new Set){for(let n of e.dependencies)$(n,t);return t.add(e),t}export{H as a,K as i,J as n,L as o,q as r,I as s,xe as t};