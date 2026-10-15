import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{m as a,y as o}from"./index-Bjk18V2I.js";import{t as s}from"./earcut-BGio7-ZG.js";var c=320,l=16,u={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},d=`
struct ChoroplethStyle {
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, ${l}>,
  valueRange: vec2<f32>,
  opacity: f32,
  mode: u32,
  colormap: u32,
  selectedRow: i32,
  paletteSize: u32,
  sqrtScale: u32,
  selectedBoost: f32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
};

@group(0) @binding(auto) var<uniform> choroplethStyle: ChoroplethStyle;
@group(0) @binding(auto) var<storage, read> choroplethPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> choroplethFeatures: array<u32>;
@group(0) @binding(auto) var<storage, read> choroplethValues: array<u32>;

${o()}

struct ChoroplethVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn getChoroplethColor(row: u32) -> vec4<f32> {
  let raw = choroplethValues[row];
  var color = choroplethStyle.noDataColor;
  if (choroplethStyle.mode == 0u) {
    // Packed rgba8, r in the low byte (the GPUColorScale layout).
    color = unpack4x8unorm(raw);
  } else if (choroplethStyle.mode == 1u) {
    let isNonFinite = (raw & 0x7fffffffu) >= 0x7f800000u;
    if (!isNonFinite) {
      let value = bitcast<f32>(raw);
      let range = choroplethStyle.valueRange;
      var t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
      if (choroplethStyle.sqrtScale != 0u) {
        t = sqrt(t);
      }
      color = vec4<f32>(clamp(spatialAnalysisSampleRamp(choroplethStyle.colormap, t), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
  } else {
    // Category: palette index, the largest value is no data.
    if (raw < choroplethStyle.paletteSize) {
      color = choroplethStyle.palette[raw];
    }
  }
  if (choroplethStyle.selectedRow >= 0 && i32(row) == choroplethStyle.selectedRow) {
    color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0), choroplethStyle.selectedBoost), max(color.a, 0.9));
  }
  return color;
}

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> ChoroplethVertexOutput {
  var output: ChoroplethVertexOutput;
  let row = choroplethFeatures[vertexIndex];
  let color = getChoroplethColor(row);
  if (color.a <= 0.0) {
    output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    output.color = vec4<f32>(0.0);
    return output;
  }
  let position = choroplethPositions[vertexIndex];
  var clip = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clip.z = (clip.z + clip.w) * 0.5;
  output.position = clip;
  output.color = vec4<f32>(color.rgb, color.a * choroplethStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: ChoroplethVertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`,f=class extends n{static layerName=`ChoroplethFillLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:u};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:c,usage:r.UNIFORM|r.COPY_DST}),a=new i(t,{...this.getShaders({modules:[e],source:d}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.props.vertexCount,instanceCount:1,bufferLayout:[],bindings:this.getBindings(n),parameters:u});this.setState({model:a,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n)),t.setVertexCount(this.props.vertexCount)}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n),t.setInstanceCount(1),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}getBindings(e){return{choroplethStyle:e,choroplethPositions:this.props.positions,choroplethFeatures:this.props.featureRows,choroplethValues:this.props.values}}writeStyle(e){let{mode:t=`packed`,ramp:n=`viridis`,valueRange:r=[0,1],sqrtScale:i=!1,palette:o=[],noDataColor:s=[0,0,0,0],selectedRow:u=-1,fillOpacity:d=.85}=this.props,f=new ArrayBuffer(c),p=new Float32Array(f),m=new Uint32Array(f),h=new Int32Array(f),g=(e,t)=>{p.set([t[0]/255,t[1]/255,t[2]/255,(t[3]??255)/255],e/4)};g(0,s),o.slice(0,l).forEach((e,t)=>{g(16+t*16,e)});let _=16+l*16;p.set(r,_/4),p[(_+8)/4]=d,m[(_+12)/4]=t===`packed`?0:t===`ramp`?1:2,m[(_+16)/4]=a[n],h[(_+20)/4]=u,m[(_+24)/4]=Math.min(o.length,l),m[(_+28)/4]=i?1:0,p[(_+32)/4]=.45,e.write(new Uint8Array(f))}};function p(e){let t=e.column(`vertices`),n=e.column(`ringOffsets`),r=e.column(`polygonRingOffsets`),i=r.length-1,a,o;if(e.hasColumn(`partFeature`))a=e.column(`partFeature`),o=e.count;else{let t=e.column(`countyPolygonOffsets`);o=t.length-1,a=new Uint32Array(i);for(let e=0;e<o;e++)for(let n=t[e];n<t[e+1];n++)a[n]=e}let s=new Uint32Array(o+1),c=0;for(let e=0;e<o;e++)for(s[e]=r[c];c<i&&a[c]===e;)c++;s[o]=r[i];let l=new Uint32Array(o+1);for(let e=0;e<i;e++)l[a[e]+1]++;for(let e=0;e<o;e++)l[e+1]+=l[e];return{featureCount:o,vertices:t,ringOffsets:n,polygonRingOffsets:r,partFeature:a,featureRingOffsets:s,featurePolygonOffsets:l}}function m(e){let{vertices:t,ringOffsets:n,polygonRingOffsets:r,partFeature:i}=e,a=[],o=[],c=[],l=r.length-1;for(let e=0;e<l;e++){let l=i[e],u=[],d=[];for(let i=r[e];i<r[e+1];i++){let a=n[i],o=n[i+1];if(o-a>1&&t[a*2]===t[(o-1)*2]&&t[a*2+1]===t[(o-1)*2+1]&&o--,!(o-a<3)){i>r[e]&&d.push(u.length/2);for(let e=a;e<o;e++){u.push(t[e*2],t[e*2+1]);let n=e+1<o?e+1:a;c.push(t[e*2],t[e*2+1],t[n*2],t[n*2+1])}}}if(u.length<6)continue;let f=s(u,d.length?d:null,2);for(let e of f)a.push(u[e*2],u[e*2+1]),o.push(l)}return{positions:Float32Array.from(a),featureRows:Uint32Array.from(o),triangleCount:a.length/6,outline:Float32Array.from(c)}}function h(e){let{featureCount:t,vertices:n,ringOffsets:r,featureRingOffsets:i}=e,a=new Float32Array(t*4),o=new Float32Array(t*2);for(let e=0;e<t;e++){let t=1/0,s=1/0,c=-1/0,l=-1/0,u=r[i[e]],d=r[i[e+1]];for(let e=u;e<d;e++){let r=n[e*2],i=n[e*2+1];r<t&&(t=r),r>c&&(c=r),i<s&&(s=i),i>l&&(l=i)}a.set([t,s,c,l],e*4),o.set([(t+c)/2,(s+l)/2],e*2)}return{locate:(e,o)=>{for(let s=0;s<t;s++){if(e<a[s*4]||e>a[s*4+2]||o<a[s*4+1]||o>a[s*4+3])continue;let t=!1;for(let a=i[s];a<i[s+1];a++){let i=r[a],s=r[a+1];for(let r=i,a=s-1;r<s;a=r++){let i=n[r*2],s=n[r*2+1],c=n[a*2],l=n[a*2+1];s>o!=l>o&&e<(c-i)*(o-s)/(l-s)+i&&(t=!t)}}if(t)return s}return-1},centers:o}}export{f as i,h as n,p as r,m as t};