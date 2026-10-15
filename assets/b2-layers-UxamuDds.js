import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{N as a}from"./resources-CwvhEOBe.js";import{i as o,t as s}from"./gpu-polygon-rasterization-0IHMY9Te.js";import{n as c}from"./gpu-raster-grid-BOSvDY5A.js";import{s as l}from"./index-Bjk18V2I.js";import{n as u,t as d}from"./graph-buffers-B4_80RlX.js";function f(e,t,n){return{set:n,positions:e.createBuffer(`${t}-positions`,n.positions),featureOffsets:e.createBuffer(`${t}-feature-offsets`,n.featureOffsets),polygonOffsets:e.createBuffer(`${t}-polygon-offsets`,n.polygonOffsets),ringOffsets:e.createBuffer(`${t}-ring-offsets`,n.ringOffsets),outline:e.createBuffer(`${t}-outline`,n.outline),outlineRows:e.createBuffer(`${t}-outline-rows`,n.outlineRows)}}function p(e,t,n){return{set:n,positions:e.createBuffer(`${t}-positions`,n.positions),lineOffsets:e.createBuffer(`${t}-line-offsets`,n.lineOffsets),segments:e.createBuffer(`${t}-segments`,n.segments),segmentRows:e.createBuffer(`${t}-segment-rows`,n.segmentRows)}}function m(e,t,n){let{set:r}=n;return{kind:`polygons`,positions:d(e,`${t}-positions`,n.positions,`float32x2`,r.positions.length/2),featureOffsets:d(e,`${t}-feature-offsets`,n.featureOffsets,`uint32`,r.featureOffsets.length),polygonOffsets:d(e,`${t}-polygon-offsets`,n.polygonOffsets,`uint32`,r.polygonOffsets.length),ringOffsets:d(e,`${t}-ring-offsets`,n.ringOffsets,`uint32`,r.ringOffsets.length)}}function h(e,t,n){let{set:r}=n;return{kind:`lines`,positions:d(e,`${t}-positions`,n.positions,`float32x2`,r.positions.length/2),lineOffsets:d(e,`${t}-line-offsets`,n.lineOffsets,`uint32`,r.lineOffsets.length)}}function g(e,t,n,r){let i=0;for(let a=0;a<e.outline.length;a+=4){let o=(e.outline[a+1]-t)/n-.5,s=(e.outline[a+3]-t)/n-.5,c=Math.max(0,Math.ceil(Math.min(o,s))),l=Math.min(r-1,Math.ceil(Math.max(o,s))-1);l>=c&&(i+=l-c+1)}return i}async function _(e,t,n,r,i){let[l,f,p,h]=r,_=p-l,v=h-f,y=Math.max(_,v)/i,b=Math.ceil(_/y),x=Math.ceil(v/y),S=b*x,C=g(n.set,f,y,x),w=Math.max(1024,Math.ceil(C*1.25)),T=e.createBuffer(`${t}-zones`,S*4),E=e.createBuffer(`${t}-boundary`,S*4),D=e.createBuffer(`${t}-overflow`,4),O=e.createBuffer(`${t}-crossings`,4),k=c({width:b,height:x,cellSize:y,origin:[l,f],rowDirection:`north`}),A=e.createParameterBuffer(`${t}-extent`,`float32`,4,o(k)),j=new a(e.device,{id:`${t}-rasterize`}),M=m(j,`${t}-polygons`,n);j.add(new s({id:`${t}-rasterization`,grid:k,parameters:A.importToGraph(j),positions:M.positions,featureOffsets:M.featureOffsets,polygonOffsets:M.polygonOffsets,ringOffsets:M.ringOffsets,crossingCapacity:w,output:{zones:d(j,`${t}-zones`,T,`uint32`,S),boundary:d(j,`${t}-boundary`,E,`uint32`,S),status:{overflow:d(j,`${t}-overflow`,D,`uint32`,1),crossingCount:d(j,`${t}-crossings`,O,`uint32`,1)}}}));let N=await j.compileAsync();try{u(e.device,N);let t=new Uint32Array((await D.readAsync()).slice().buffer),n=new Uint32Array((await O.readAsync()).slice().buffer);return{width:b,height:x,bounds:[l,f,l+b*y,f+x*y],zones:T,boundary:E,crossingCount:n[0],overflowed:t[0]!==0}}finally{N.destroy()}}function v(e,t,n,r=255){return(e&255|(t&255)<<8|(n&255)<<16|(r&255)<<24)>>>0}function y(e,t){let n=Math.min(Math.max(t,0),1)*(e.length-1),r=Math.min(Math.floor(n),e.length-2),i=n-r,a=e[r],o=e[r+1];return[a[0]+(o[0]-a[0])*i,a[1]+(o[1]-a[1])*i,a[2]+(o[2]-a[2])*i]}function b(e,t){let n=new Uint32Array(t);for(let r=0;r<t;r++){let[i,a,o]=y(e,t===1?.5:r/(t-1));n[r]=v(Math.round(i),Math.round(a),Math.round(o))}return n}var x=`fn getSpatialAnalysisColor(valueRow: u32) -> vec4<f32> {`,S=4294967295,C=class extends l{static layerName=`ZoneFillLayer`;getShaderSource(){let e=super.getShaderSource();if(!e.includes(x))throw Error(`ZoneFillLayer: the shared raster shader changed`);return`${e.replace(x,`fn getSpatialAnalysisStyleColor(valueRow: u32) -> vec4<f32> {`)}
fn getSpatialAnalysisColor(zone: u32) -> vec4<f32> {
  if (zone == ${S}u) { return vec4<f32>(0.0); }
  return getSpatialAnalysisStyleColor(zone);
}
`}},w=class extends l{static layerName=`PackedZoneFillLayer`;getShaderSource(){let e=super.getShaderSource();if(!e.includes(x))throw Error(`PackedZoneFillLayer: the shared raster shader changed`);return`${e.replace(x,`fn getSpatialAnalysisStyleColor(valueRow: u32) -> vec4<f32> {`)}
fn getSpatialAnalysisColor(zone: u32) -> vec4<f32> {
  if (zone == ${S}u) { return vec4<f32>(0.0); }
  return unpack4x8unorm(styleValues[zone]);
}
`}},T=48,E={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},D=`
struct LinkStyle {
  color: vec4<f32>,
  widthPixels: f32,
  slotCapacity: u32,
  slotCount: u32,
  firstSlot: u32,
  queryStride: u32,
  useTargetIds: u32,
  fade: f32,
  _padding: u32,
};

@group(0) @binding(auto) var<uniform> linkStyle: LinkStyle;
@group(0) @binding(auto) var<storage, read> linkQueries: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> linkTargets: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> linkTargetIds: array<u32>;

struct LinkVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) alpha: f32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> LinkVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: LinkVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.alpha = 0.0;
  let slotCount = max(linkStyle.slotCount, 1u);
  let query = (instanceIndex / slotCount) * max(linkStyle.queryStride, 1u);
  let slotOffset = instanceIndex % slotCount;
  let slot = linkStyle.firstSlot + slotOffset;
  let index = query * linkStyle.slotCapacity + slot;
  var endPoint: vec2<f32>;
  if (linkStyle.useTargetIds != 0u) {
    let id = linkTargetIds[index];
    if (id == ${S}u) { return output; }
    endPoint = linkTargets[id];
  } else {
    endPoint = linkTargets[index];
  }
  let point = linkQueries[query];
  if (endPoint.x != endPoint.x || endPoint.y != endPoint.y || point.x != point.x) { return output; }
  var startClip = project_position_to_clipspace(vec3<f32>(point, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var endClip = project_position_to_clipspace(vec3<f32>(endPoint, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  startClip.z = (startClip.z + startClip.w) * 0.5;
  endClip.z = (endClip.z + endClip.w) * 0.5;
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * linkStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.alpha = linkStyle.color.a * (1.0 - linkStyle.fade * f32(slotOffset) / f32(slotCount));
  return output;
}

@fragment fn fragmentMain(input: LinkVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(linkStyle.color.rgb, input.alpha * coverage);
}
`,O=class extends n{static layerName=`LinkLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:E};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:T,usage:r.UNIFORM|r.COPY_DST}),a=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST}),o=new i(t,{...this.getShaders({modules:[e],source:D}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(n,a),parameters:E});this.setState({model:o,styleBuffer:n,placeholderBuffer:a})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t.setBindings(this.getBindings(n,r))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state,{color:r=[255,255,255,200],widthPixels:i=1.2,slotCapacity:a=1,slotCount:o=1,firstSlot:s=0,queryStride:c=1,fade:l=.5,queryCount:u,targetIds:d}=this.props,f=new ArrayBuffer(T),p=new Float32Array(f),m=new Uint32Array(f);p.set([r[0]/255,r[1]/255,r[2]/255,r[3]/255],0),p[4]=i,m[5]=a,m[6]=o,m[7]=s,m[8]=c,m[9]=d?1:0,p[10]=l,n.write(new Uint8Array(f)),t.setInstanceCount(Math.ceil(u/Math.max(c,1))*o),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t.destroy(),n.destroy(),r.destroy(),super.finalizeState(e)}getBindings(e,t){return{linkStyle:e,linkQueries:this.props.queries,linkTargets:this.props.targets,linkTargetIds:this.props.targetIds??t}}};export{g as a,m as c,_ as i,p as l,w as n,b as o,C as r,h as s,O as t,f as u};