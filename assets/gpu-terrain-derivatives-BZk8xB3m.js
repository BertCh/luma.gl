import{_ as e,i as t,w as n,y as r}from"./resources-CwvhEOBe.js";import{s as i}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a,c as o,i as s,l as c,n as l,o as u,r as d,s as f,t as p,u as m}from"./raster-band-utils-CA-sh8xt.js";import{a as h,c as g,l as _,s as v,t as y,u as b}from"./raster-grid-D2bmCwkc.js";import{t as x}from"./gpu-raster-buffer-to-texture-BroAAmI3.js";import{t as S}from"./gpu-raster-edges-DbRocSpD.js";var C={cellSizeIndex:0,northEdgeIndex:5};function w(e,t,n=new Float32Array(8)){return s(`getGPUTerrainDerivativesParameterValues`,n,8),n[2]=e.zFactor??1,n[3]=e.azimuthDegrees??315,n[4]=e.altitudeDegrees??45,n[7]=0,b(n,t,C),n}var T=class{id;props;requiredHalo=1;constructor(e){this.id=e.id??`terrain-derivatives`,this.props=e;let{id:t}=this,{output:n}=e,i=f(t,e.grid);if(!n.slope&&!n.aspect&&!n.hillshade&&!n.outputValidity&&!n.hillshadeTexture)throw Error(`${t} requires at least one output`);for(let[e,a,o]of[[`slope`,n.slope,`float32`],[`aspect`,n.aspect,`float32`],[`hillshade`,n.hillshade,`float32`],[`outputValidity`,n.outputValidity,`uint32`]])if(a&&(r(a,[o],`${t} output.${e}`),a.length!==i))throw Error(`${t} output.${e} must contain one value per pixel`);if(m(t,e.parameters,8),c(t,`output.hillshadeTexture`,n.hillshadeTexture,[`r32float`,`rgba32float`],e.grid),g(t,e.cellSizeMode),o(t,e.grid,e.cellSizeMode),![`degrees`,`percent`].includes(e.slopeUnits??`degrees`))throw Error(`${t} slopeUnits must be degrees or percent`);_(t,e.rowDirection),u(t,[n.slope,n.aspect,n.hillshade,n.outputValidity],[...l(e.elevation),e.parameters])}getCommandNodes(r){let{id:i,props:o}=this,{output:s}=o,{width:c,height:l}=o.grid;a(i,r,o.elevation,[s.hillshadeTexture]),e(i,r,[o.parameters,s.slope,s.aspect,s.hillshade,s.outputValidity]);let u=c*l,f=o.borderMode??`clamp`,p=d(r,i,o.elevation,o.grid,!1),m=[...p.nodes],h=[`x`,`y`].map(e=>{let a=n(r,`${i}-gradient-${e}-values`,`float32`,u),o=n(r,`${i}-gradient-${e}-validity`,`uint32`,u);return m.push(...t(r,()=>new S({id:`${i}-gradient-${e}`,width:c,height:l,input:p.band,output:a,outputValidity:o,operator:`sobel`,direction:e,borderMode:f,scale:1}).addToGraph(r))),{values:a,validity:o}}),g=s.hillshade??(s.hillshadeTexture?n(r,`${i}-hillshade`,`float32`,u):void 0);if(m.push(E(r,{id:`${i}-shade`,width:c,height:l,gradientX:h[0].values,gradientY:h[1].values,gradientValidity:h[0].validity,settings:o.parameters,slope:s.slope,aspect:s.aspect,hillshade:g,validity:s.outputValidity,cellSizeMode:o.cellSizeMode??`uniform`,slopeUnits:o.slopeUnits??`degrees`,rowDirection:o.rowDirection??`south`})),s.hillshadeTexture&&g){let e=s.hillshadeTexture;m.push(...t(r,()=>new x({id:`${i}-hillshade-texture`,input:{id:`${i}-hillshade-band`,format:`float32`,storage:{kind:`buffer`,values:g}},output:e,channel:0}).addToGraph(r)))}return m}};function E(e,t){let n=[{name:`gradientX`,view:t.gradientX,type:`f32`,access:`read`},{name:`gradientY`,view:t.gradientY,type:`f32`,access:`read`},{name:`gradientValidity`,view:t.gradientValidity,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];return t.slope&&n.push({name:`slopeValues`,view:t.slope,type:`f32`,access:`read_write`}),t.aspect&&n.push({name:`aspectValues`,view:t.aspect,type:`f32`,access:`read_write`}),t.hillshade&&n.push({name:`hillshadeValues`,view:t.hillshade,type:`f32`,access:`read_write`}),t.validity&&n.push({name:`validityValues`,view:t.validity,type:`u32`,access:`read_write`}),i(e,{id:t.id,operation:`GPUTerrainDerivatives`,variant:`shade-${t.cellSizeMode}`,bindings:n,invocationCount:t.width*t.height,declarations:`const WIDTH: u32 = ${t.width}u;
const HEIGHT: u32 = ${t.height}u;
const PI: f32 = 3.141592653589793;
${y}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${v(t.rowDirection)}
${p}
${h(t.cellSizeMode,C)}`,body:`let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let larger = max(abs(eastGradient), abs(northGradient));
  let smaller = min(abs(eastGradient), abs(northGradient));
  var gradientMagnitude = 0.0;
  if (larger > 0.0) {
    let ratio = smaller / larger;
    gradientMagnitude = larger * sqrt(1.0 + ratio * ratio);
  }
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient) && isFiniteValue(gradientMagnitude);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${t.slope?`let slope = ${t.slopeUnits===`degrees`?`atan(gradientMagnitude) * RADIANS_TO_DEGREES`:`gradientMagnitude * 100.0`};
  slopeValues[slopeValuesOffset + index] = select(invalidValue, slope, isValid);`:``}
  ${t.aspect?`var aspect = -1.0;
  if (gradientMagnitude > 0.0) {
    aspect = atan2(-eastGradient, -northGradient) * RADIANS_TO_DEGREES;
    if (aspect < 0.0) { aspect += 360.0; }
    if (aspect >= 360.0) { aspect -= 360.0; }
  }
  aspectValues[aspectValuesOffset + index] = select(invalidValue, aspect, isValid);`:``}
  ${t.hillshade?`let azimuth = settings[settingsOffset + 3u] * DEGREES_TO_RADIANS;
  let altitude = settings[settingsOffset + 4u] * DEGREES_TO_RADIANS;
  let light = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let shade = max(dot(normal, light), 0.0);
  hillshadeValues[hillshadeValuesOffset + index] = select(invalidValue, shade, isValid);`:``}
  ${t.validity?`validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`:``}`})}export{w as n,T as t};