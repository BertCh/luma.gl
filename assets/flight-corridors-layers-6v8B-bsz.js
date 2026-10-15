import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{n as i}from"./earth-xB16XgAa.js";import{n as a}from"./analysis-frame-Blu0tYdw.js";import{i as o}from"./convert-colors-D-TZjU1C.js";import{m as s,y as c}from"./index-Bjk18V2I.js";import{o as l}from"./b12-tracks-BINmCvAR.js";var u=`poopdeck-adsb-paths`,d=3600/1852,f=[[64,160,255,255],[255,140,50,255],[140,210,150,255],[200,150,230,255]],p=Math.PI/180;function m(e,t,n,r){let i=t*p,a=r*p,o=(n-e)*p,s=Math.sin(o)*Math.cos(a),c=Math.cos(i)*Math.sin(a)-Math.sin(i)*Math.cos(a)*Math.cos(o);return(Math.atan2(s,c)/p%360+360)%360}function h(e,t,n,r){let a=t*p,o=r*p,s=Math.sin((o-a)/2)**2+Math.cos(a)*Math.cos(o)*Math.sin((n-e)*p/2)**2;return 2*i*Math.asin(Math.min(1,Math.sqrt(s)))}function g(e){return e>=45&&e<135?0:e>=225&&e<315?1:e>=315||e<45?2:3}function _(e){let t=e.column(`vertices`),n=e.column(`pathOffsets`),r=t.length/2,i=n.length-1,o=[-96,38.5],s=a({kind:`azimuthal-equidistant`,origin:o}),c=new Float32Array(t.length),u=180,d=90,f=-180,p=-90;for(let e=0;e<r;e++){let n=t[e*2],r=t[e*2+1],[i,a]=s.project(n,r);c[e*2]=i,c[e*2+1]=a,u=Math.min(u,n),f=Math.max(f,n),d=Math.min(d,r),p=Math.max(p,r)}let _=l({offsets:n,positions:c,lngLat:t,timestamps:Float32Array.from(e.column(`timestamp`)),origin:o,analysisFrame:s.frame,spatialContext:s.spatialContext,project:s.project,unproject:s.unproject,drawInDegrees:!0}),v=Float32Array.from(e.column(`altitude`)),y=Float32Array.from(e.column(`groundSpeed`),e=>e*3),b=new Uint32Array(r),x=new Uint32Array(i),S=new Float32Array(i),C=new Float32Array(i);for(let e=0;e<i;e++){let r=n[e],i=n[e+1]-1,a=0;for(let e=r;e<=i;e++)a=Math.max(a,v[e]),e>r&&(b[e]=g(m(t[(e-1)*2],t[(e-1)*2+1],t[e*2],t[e*2+1])));b[r]=b[Math.min(i,r+1)],C[e]=a,x[e]=g(m(t[r*2],t[r*2+1],t[i*2],t[i*2+1])),S[e]=h(t[r*2],t[r*2+1],t[i*2],t[i*2+1])}return{..._,altitude:v,reportedKnots:y,vertexDirection:b,trackDirection:x,trackChord:S,trackPeak:C,bbox:[u,d,f,p],timeOriginMs:e.properties.timeOriginMs??15782688e5}}function v(e,t=e.length){if(t===0)return 0;let n=Float64Array.from({length:t},(t,n)=>e[n]).sort(),r=t>>1;return t%2?n[r]:.5*(n[r-1]+n[r])}function y(e){let t=e=>{let t=Math.floor((e%86400+86400)%86400);return`${String(Math.floor(t/3600)).padStart(2,`0`)}:${String(Math.floor(t%3600/60)).padStart(2,`0`)}`};return`${t(e)} UTC (${t(e-5*3600)} EST, ${t(e-8*3600)} PST)`}var b={uniform:0,altitude:1,value:2,category:3},x=[[86,180,233,255],[240,150,60,255],[150,150,150,255],[120,200,120,255]],S=224,C=`
struct FlightStyle {
  baseColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
  sizePixels: f32,
  opacity: f32,
  elevationScale: f32,
  colormap: u32,
  valueRange: vec2<f32>,
  valueScale: f32,
  colorMode: u32,
  useIds: u32,
  useWeights: u32,
  useClip: u32,
  paletteSize: u32,
  minAltitude: f32,
  directionFilter: u32,
  coneCosine: f32,
  useHeadings: u32,
  valueIsFloat: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(auto) var<uniform> flightStyle: FlightStyle;
@group(0) @binding(auto) var<storage, read> flightLngLat: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flightElevations: array<f32>;
@group(0) @binding(auto) var<storage, read> flightValues: array<u32>;
@group(0) @binding(auto) var<storage, read> flightIds: array<u32>;

${c()}

// Color of one row from the elevation (metres) and its value word.
fn getFlightColor(row: u32, altitude: f32) -> vec4<f32> {
  let mode = flightStyle.colorMode;
  if (mode == 0u) { return flightStyle.baseColor; }
  var t = 0.0;
  if (mode == 3u) {
    return flightStyle.palette[flightValues[row] % max(flightStyle.paletteSize, 1u)];
  }
  var value = altitude;
  if (mode == 2u) {
    let raw = flightValues[row];
    value = select(f32(raw), bitcast<f32>(raw), flightStyle.valueIsFloat != 0u) * flightStyle.valueScale;
    if ((raw & 0x7fffffffu) >= 0x7f800000u) { return vec4<f32>(0.0); }
  }
  t = clamp((value - flightStyle.valueRange.x) / max(flightStyle.valueRange.y - flightStyle.valueRange.x, 1e-20), 0.0, 1.0);
  let rgb = spatialAnalysisSampleRamp(flightStyle.colormap, t);
  return vec4<f32>(rgb, flightStyle.baseColor.a);
}

fn projectFlight(lngLat: vec2<f32>, altitude: f32) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(lngLat, altitude * flightStyle.elevationScale), vec3<f32>(0.0), vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`,w=`
${C}
@group(0) @binding(auto) var<storage, read> flightEndVertices: array<u32>;
@group(0) @binding(auto) var<storage, read> flightWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> flightClip: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flightHeadings: array<f32>;

struct SegmentOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn hiddenSegment() -> SegmentOutput {
  var output: SegmentOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.side = 0.0;
  return output;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> SegmentOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var row = instanceIndex;
  if (flightStyle.useIds != 0u) { row = flightIds[instanceIndex]; }
  let endVertex = flightEndVertices[row];
  let startVertex = endVertex - 1u;
  let altitudeStart = flightElevations[startVertex];
  let altitudeEnd = flightElevations[endVertex];
  if (flightStyle.minAltitude > 0.0 && min(altitudeStart, altitudeEnd) < flightStyle.minAltitude) {
    return hiddenSegment();
  }
  if (flightStyle.directionFilter != 0u) {
    let heading = cos(flightHeadings[endVertex]);
    let east = heading > flightStyle.coneCosine;
    let west = heading < -flightStyle.coneCosine;
    if (!select(west, east, flightStyle.directionFilter == 1u)) { return hiddenSegment(); }
  }
  var color = getFlightColor(endVertex, 0.5 * (altitudeStart + altitudeEnd));
  if (flightStyle.useWeights != 0u) { color.a = color.a * flightWeights[row]; }
  var clip = vec2<f32>(0.0, 1.0);
  if (flightStyle.useClip != 0u) { clip = flightClip[row]; }
  if (color.a <= 0.0 || clip.y <= clip.x) { return hiddenSegment(); }
  let start = flightLngLat[startVertex];
  let end = flightLngLat[endVertex];
  let clippedStart = mix(start, end, clip.x);
  let clippedEnd = mix(start, end, clip.y);
  let startClip = projectFlight(clippedStart, mix(altitudeStart, altitudeEnd, clip.x));
  let endClip = projectFlight(clippedEnd, mix(altitudeStart, altitudeEnd, clip.y));
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) { direction = screenDirection / directionLength; }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * flightStyle.sizePixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  var output: SegmentOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: SegmentOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * flightStyle.opacity * coverage);
}
`,T=`
${C}

struct PointOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> PointOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: PointOutput;
  var row = instanceIndex;
  if (flightStyle.useIds != 0u) { row = flightIds[instanceIndex]; }
  let source = flightLngLat[row];
  let altitude = flightElevations[row];
  let color = getFlightColor(row, altitude);
  if (source.x != source.x || source.y != source.y || color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    output.corner = vec2<f32>(0.0);
    return output;
  }
  let corner = corners[vertexIndex];
  var clipPosition = projectFlight(source, altitude);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * flightStyle.sizePixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: PointOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) { discard; }
  let rim = smoothstep(0.55, 0.8, radiusSquared);
  let rgb = mix(input.color.rgb, input.color.rgb * 0.35, rim);
  let coverage = 1.0 - smoothstep(0.85, 1.0, radiusSquared);
  return vec4<f32>(rgb, input.color.a * flightStyle.opacity * coverage);
}
`,E={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},D=class extends n{static layerName=`FlightBaseLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:E};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:S,usage:r.UNIFORM|r.COPY_DST}),i=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST});this.setState({model:null,styleBuffer:n,placeholderBuffer:i});let a=new o(t,{...this.getShaders({modules:[e],source:this.getShaderSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:{...this.getBindings(i),flightStyle:n},parameters:E});this.setState({model:a})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t&&n&&r&&t.setBindings({...this.getBindings(r),flightStyle:n})}getModels(){let e=this.state.model;return e?[e]:[]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;this.writeStyle(n);let{drawCommands:r,instanceCount:i=0}=this.props;r?(t.setInstanceCount(0),t.draw(e),r.draw(e,0)):(t.setInstanceCount(i),t.draw(e))}finalizeState(e){let t=this.state;t.model?.destroy(),t.styleBuffer?.destroy(),t.placeholderBuffer?.destroy(),this.setState({model:null,styleBuffer:null,placeholderBuffer:null}),super.finalizeState(e)}getCommonBindings(e){return{flightLngLat:this.props.lngLat,flightElevations:this.props.elevations,flightValues:this.props.values??e,flightIds:this.props.ids??e}}packStyle(e,t){let n=this.props,r=new ArrayBuffer(S),i=new Float32Array(r),a=new Uint32Array(r),o=(e,t)=>{i.set([t[0]/255,t[1]/255,t[2]/255,(t[3]??255)/255],e/4)};o(0,n.color??[255,255,255,255]);let c=n.palette??x;for(let e=0;e<8;e++)o(16+e*16,c[e%c.length]);let l=n.colorMode??`uniform`;i[36]=t.sizePixels,i[37]=n.opacity??1,i[38]=n.elevationScale??0,a[39]=s[n.ramp??`viridis`],i.set(n.valueRange??[0,1],40),i[42]=n.valueScale??1,a[43]=b[l],a[44]=n.ids?1:0,a[45]=t.useWeights?1:0,a[46]=t.useClip?1:0,a[47]=Math.min(8,c.length),i[48]=t.minAltitude??0,a[49]=t.directionFilter??0,i[50]=t.coneCosine??.7071,a[51]=0,a[52]=t.valueIsFloat?1:0,e.write(new Uint8Array(r))}},O=class extends D{static layerName=`FlightSegmentLayer`;getShaderSource(){return w}getBindings(e){return{...this.getCommonBindings(e),flightEndVertices:this.props.endVertices,flightWeights:this.props.weights??e,flightClip:this.props.clipFractions??e,flightHeadings:this.props.headings??e}}writeStyle(e){let{directionFilter:t=`all`,directionHalfAngle:n=Math.PI/4}=this.props;this.packStyle(e,{sizePixels:this.props.widthPixels??1.5,useWeights:!!this.props.weights,useClip:!!this.props.clipFractions,minAltitude:this.props.minAltitude,directionFilter:this.props.headings?t===`east`?1:t===`west`?2:0:0,coneCosine:Math.cos(n),valueIsFloat:this.props.valuesAreFloat??!0})}},k=class extends D{static layerName=`FlightPointLayer`;getShaderSource(){return T}getBindings(e){return this.getCommonBindings(e)}writeStyle(e){this.packStyle(e,{sizePixels:this.props.radiusPixels??3,valueIsFloat:this.props.valuesAreFloat??!1})}};export{d as a,_ as c,f as i,O as n,y as o,u as r,v as s,k as t};