import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{n as i}from"./analysis-frame-Blu0tYdw.js";import{i as a}from"./convert-colors-D-TZjU1C.js";import{H as o,d as s,m as c,y as l}from"./index-Bjk18V2I.js";import{i as u,n as d,s as f}from"./quake-regions-BlG33Nev.js";var p={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},m=128,h=l(),g=`fn getQuakeClipPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}`;function _(e){return`${s(e)}
fn getQuakeClipPosition(position: vec2<f32>) -> vec4<f32> {
  return projectSpatialAnalysisPosition(position);
}`}function v(e){return`
struct QuakeStyle {
  palette: array<vec4<f32>, 3>,
  outlineColor: vec4<f32>,
  playhead: f32,
  fadeDays: f32,
  sizePixels: f32,
  sizeGrowth: f32,
  opacity: f32,
  ghostAlpha: f32,
  depthMax: f32,
  popStrength: f32,
  colorMode: u32,
  colormap: u32,
  staticMode: u32,
  shallowLimit: f32,
  deepLimit: f32,
  minimumMagnitude: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> quakeStyle: QuakeStyle;
@group(0) @binding(auto) var<storage, read> quakePositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> quakeTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> quakeMagnitudes: array<f32>;
@group(0) @binding(auto) var<storage, read> quakeDepths: array<f32>;

struct QuakeVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
};

${h}

${e?_(e):g}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> QuakeVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: QuakeVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.corner = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  let magnitude = quakeMagnitudes[instanceIndex];
  if (magnitude < quakeStyle.minimumMagnitude) {
    return output;
  }
  let age = quakeStyle.playhead - quakeTimes[instanceIndex];
  var alpha = 1.0;
  var scale = 1.0;
  var ageFraction = 0.0;
  if (quakeStyle.staticMode == 0u) {
    if (age < 0.0) {
      return output;
    }
    if (age > quakeStyle.fadeDays) {
      if (quakeStyle.ghostAlpha <= 0.0) {
        return output;
      }
      alpha = quakeStyle.ghostAlpha;
      scale = 0.55;
      ageFraction = 1.0;
    } else {
      ageFraction = age / max(quakeStyle.fadeDays, 1e-3);
      alpha = max(pow(1.0 - ageFraction, 1.4), quakeStyle.ghostAlpha);
      // A new event swells and settles within the first few percent of its life.
      scale = 1.0 + quakeStyle.popStrength * exp(-age / max(quakeStyle.fadeDays * 0.06, 0.02));
    }
  }
  let depth = quakeDepths[instanceIndex];
  var rgb = vec3<f32>(1.0);
  if (quakeStyle.colorMode == 0u) {
    rgb = spatialAnalysisSampleRamp(quakeStyle.colormap, 1.0 - clamp(depth / max(quakeStyle.depthMax, 1.0), 0.0, 1.0));
  } else if (quakeStyle.colorMode == 1u) {
    rgb = spatialAnalysisSampleRamp(quakeStyle.colormap, 1.0 - ageFraction);
  } else {
    var slot = 0u;
    if (depth >= quakeStyle.deepLimit) {
      slot = 2u;
    } else if (depth >= quakeStyle.shallowLimit) {
      slot = 1u;
    }
    rgb = quakeStyle.palette[slot].rgb;
  }
  let radius = quakeStyle.sizePixels * pow(quakeStyle.sizeGrowth, magnitude - 4.0) * scale;
  let corner = corners[vertexIndex];
  var clipPosition = getQuakeClipPosition(quakePositions[instanceIndex]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radius),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = vec4<f32>(rgb, alpha);
  return output;
}

@fragment fn fragmentMain(input: QuakeVertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.corner, input.corner);
  if (radiusSquared > 1.0) {
    discard;
  }
  let ring = smoothstep(0.55, 0.85, radiusSquared);
  let rgb = mix(input.color.rgb, quakeStyle.outlineColor.rgb, ring * quakeStyle.outlineColor.a);
  let cover = 1.0 - smoothstep(0.88, 1.0, radiusSquared);
  return vec4<f32>(rgb, input.color.a * quakeStyle.opacity * cover * mix(0.62, 1.0, ring));
}
`}var y=class extends n{static layerName=`QuakeEventLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:p};getAttributeManager(){return null}initializeState({device:e}){if(e.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let t=e.createBuffer({id:`${this.id}-style`,byteLength:m,usage:r.UNIFORM|r.COPY_DST}),n=v(this.props.analysisFrame);this.setState({model:this.createModel(e,n,t),styleBuffer:t,source:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,source:r}=this.state,i=v(this.props.analysisFrame);if(i!==r){t.destroy(),this.setState({model:this.createModel(this.context.device,i,n),source:i});return}t.setBindings(this.getBindings(n))}_drawLayer(e){let{shaderModuleProps:n}=e;if(!this.props.analysisFrame||!n?.project){super._drawLayer(e);return}super._drawLayer({...e,shaderModuleProps:{...n,project:{...n.project,coordinateSystem:t.LNGLAT,coordinateOrigin:[0,0,0]}}})}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n),t.setInstanceCount(this.props.instanceCount),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}createModel(t,n,r){return new a(t,{...this.getShaders({modules:[e],source:n}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(r),parameters:p})}getBindings(e){return{quakeStyle:e,quakePositions:this.props.positions,quakeTimes:this.props.times,quakeMagnitudes:this.props.magnitudes,quakeDepths:this.props.depths}}writeStyle(e){let t=this.props,n=new ArrayBuffer(m),r=new Float32Array(n),i=new Uint32Array(n);(t.palette??[[255,154,98],[78,209,181],[185,149,255]]).forEach((e,t)=>{r.set([e[0]/255,e[1]/255,e[2]/255,(e[3]??255)/255],t*4)});let a=t.outlineColor??[10,14,22,200];r.set([a[0]/255,a[1]/255,a[2]/255,(a[3]??255)/255],12);let o=t.classLimits??[70,300];r[16]=t.playhead??0,r[17]=t.fadeDays??30,r[18]=t.sizePixels??2.5,r[19]=t.sizeGrowth??1.9,r[20]=t.opacity??1,r[21]=t.ghostAlpha??0,r[22]=t.depthMax??300,r[23]=t.popStrength??1.2,i[24]=t.colorMode===`age`?1:t.colorMode===`class`?2:0,i[25]=c[t.ramp??`magma`],i[26]=t.staticMode?1:0,r[27]=o[0],r[28]=o[1],r[29]=t.minimumMagnitude??0,e.write(new Uint8Array(n))}};function b(e,t){let[n,r,i,a]=t,o=1/0,s=1/0,c=-1/0,l=-1/0;for(let t=0;t<=8;t++){let u=n+(i-n)*t/8,d=r+(a-r)*t/8;for(let[t,f]of[[u,r],[u,a],[n,d],[i,d]]){let[n,r]=e.project(t,f);o=Math.min(o,n),s=Math.min(s,r),c=Math.max(c,n),l=Math.max(l,r)}}return[o,s,c,l]}function x(e,t,n){let r=f[t],[a,s,c,l]=r.bbox,d=[(a+c)/2,(s+l)/2],p=new o(d),m=i({kind:`azimuthal-equidistant`,origin:d}),h=e.column(`position`),g=e.column(`timestamp`),_=e.column(`magnitude`),v=e.column(`depth`),y=e.column(`type`),x=(Number(e.properties.timeOriginMs??u)-u)/864e5,S=[];for(let e=0;e<g.length;e++){let t=h[e*2],r=h[e*2+1];y[e]===0&&_[e]>=n-1e-4&&t>=a&&t<=c&&r>=s&&r<=l&&S.push(e)}let C=S.length,w={count:C,origin:d,projection:p,analysisFrame:m.frame,spatialContext:m.spatialContext,analysisProjection:m,positions:new Float32Array(C*2),mercatorPositions:new Float32Array(C*2),days:new Float32Array(C),magnitude:new Float32Array(C),depth:new Float32Array(C),windowMeters:b(p,r.bbox),analysisWindowMeters:b(m,r.bbox)};return S.forEach((e,t)=>{let[n,r]=m.project(h[e*2],h[e*2+1]);w.positions[t*2]=n,w.positions[t*2+1]=r;let[i,a]=p.project(h[e*2],h[e*2+1]);w.mercatorPositions[t*2]=i,w.mercatorPositions[t*2+1]=a,w.days[t]=g[e]/86400+x,w.magnitude[t]=_[e],w.depth[t]=Math.max(0,v[e])}),w}function S(e,t,n,r){let[i,a,o,s]=f[n].bbox,c=e.column(`vertices`),l=e.column(`stepClass`),u=e.column(`convergence`),d=(e,t)=>e>=i-r&&e<=o+r&&t>=a-r&&t<=s+r,p=[];for(let e=0;e<l.length;e++){let t=e*4;(d(c[t],c[t+1])||d(c[t+2],c[t+3]))&&p.push(e)}let m={count:p.length,segments:new Float32Array(p.length*4),stepClass:new Uint8Array(p.length),convergence:new Float32Array(p.length)};return p.forEach((e,n)=>{let r=e*4,[i,a]=t.analysisProjection.project(c[r],c[r+1]),[o,s]=t.analysisProjection.project(c[r+2],c[r+3]);m.segments.set([i,a,o,s],n*4),m.stepClass[n]=l[e],m.convergence[n]=u[e]}),m}function C(e,t,n,r){let i=new Float32Array(r*2),a=new Uint32Array(r),o=0,s=!1;for(let c=0;c<e.count&&!s;c++){let l=e.stepClass[c];if(!t(l))continue;let u=c*4,[f,p,m,h]=[e.segments[u],e.segments[u+1],e.segments[u+2],e.segments[u+3]],g=Math.max(1,Math.ceil(Math.hypot(m-f,h-p)/n));for(let e=0;e<=g;e++){if(o>=r){s=!0;break}let t=e/g;i[o*2]=f+(m-f)*t,i[o*2+1]=p+(h-p)*t,a[o]=d[l],o++}}return{count:o,positions:i,families:a,overflow:s}}function w(e,t,n,r){let i=0,a=0;for(let o=0;o<e.length;o++){let s=t+o*n;s<r-1e-6||(i+=e[o],a+=e[o]*s)}if(i<2)return{count:i,b:NaN,standardError:NaN,a:NaN};let o=a/i,s=Math.LOG10E/(o-(r-n/2));return{count:i,b:s,standardError:s/Math.sqrt(i),a:Math.log10(i)+s*r}}function T(e,t){let n=0;for(let t=0;t<e.length;t++)n+=e[t];if(n===0)return NaN;let r=0;for(let i=0;i<e.length;i++)if(r+=e[i],r>=n/2)return(i+.5)*t;return NaN}export{x as a,S as i,w as n,y as o,T as r,C as t};