import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{m as a,y as o}from"./index-Bjk18V2I.js";import{t as s}from"./summary-reader-CsgWxENf.js";function c(e,t,n,r){return new s(e,t,n.map(({buffer:e,bytes:t})=>({buffer:e,size:t})),e=>{let i=new Map,a=new Map,o=0;for(let e of n)i.set(e.name,o),a.set(e.name,e.bytes),o+=e.bytes;try{r(t=>{let n=i.get(t);if(n===void 0)throw Error(`Unknown reader slice "${t}"`);let r=e.slice(n,n+a.get(t));return{f32:new Float32Array(r),u32:new Uint32Array(r)}})}catch(e){console.error(`reader ${t}:`,e)}})}var l=`
struct FillStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  palette: array<vec4<f32>, 8>,
  valueRange: vec2<f32>,
  opacity: f32,
  colormap: u32,
  valueFormat: u32,
  valueStride: u32,
  valueOffset: u32,
  paletteSize: u32,
  noDataValue: u32,
  selectedRow: u32,
};

@group(0) @binding(auto) var<uniform> fillStyle: FillStyle;
@group(0) @binding(auto) var<storage, read> fillTriangles: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> fillOwners: array<u32>;
@group(0) @binding(auto) var<storage, read> fillValues: array<u32>;

${o()}

fn getFillColor(row: u32) -> vec4<f32> {
  let colormap = fillStyle.colormap;
  if (colormap == 0u || fillStyle.valueFormat == 0u) {
    return fillStyle.baseColor;
  }
  let raw = fillValues[row * fillStyle.valueStride + fillStyle.valueOffset];
  if (fillStyle.valueFormat == 1u && raw == fillStyle.noDataValue) {
    return fillStyle.noDataColor;
  }
  if (colormap == 4u) {
    var index = raw;
    if (fillStyle.valueFormat == 2u) {
      index = u32(max(bitcast<f32>(raw), 0.0));
    }
    return fillStyle.palette[index % max(fillStyle.paletteSize, 1u)];
  }
  if (colormap == 5u) {
    let on = select(raw != 0u, bitcast<f32>(raw) != 0.0, fillStyle.valueFormat == 2u);
    return select(fillStyle.noDataColor, fillStyle.baseColor, on);
  }
  let value = select(f32(raw), bitcast<f32>(raw), fillStyle.valueFormat == 2u);
  let isNonFinite = fillStyle.valueFormat == 2u && (raw & 0x7fffffffu) >= 0x7f800000u;
  if (isNonFinite) {
    return fillStyle.noDataColor;
  }
  let range = fillStyle.valueRange;
  let t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  let rgb = spatialAnalysisSampleRamp(colormap, t);
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), fillStyle.baseColor.a);
}

struct FillOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) triangle: u32
) -> FillOutput {
  var output: FillOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let row = fillOwners[triangle];
  if (row == 0xffffffffu) {
    return output;
  }
  var color = getFillColor(row);
  if (color.a <= 0.0) {
    return output;
  }
  if (row == fillStyle.selectedRow) {
    color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0), 0.45), 1.0);
  }
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(fillTriangles[triangle * 3u + vertexIndex], 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  output.position = clipPosition;
  output.color = vec4<f32>(color.rgb, color.a * fillStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: FillOutput) -> @location(0) vec4<f32> {
  return input.color;
}
`,u=208,d={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},f=class extends n{static layerName=`PolygonFillLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:d};getAttributeManager(){return null}getBindings(e){return{fillStyle:e.styleBuffer,fillTriangles:this.props.triangles,fillOwners:this.props.owners,fillValues:this.props.values??e.placeholder}}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:u,usage:r.UNIFORM|r.COPY_DST}),a=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST}),o=new i(t,{...this.getShaders({modules:[e],source:l}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:3,instanceCount:0,bufferLayout:[],bindings:this.getBindings({styleBuffer:n,placeholder:a}),parameters:d});this.setState({model:o,styleBuffer:n,placeholder:a})}updateState(e){super.updateState(e);let t=this.state;t.model.setBindings(this.getBindings(t))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state,r=this.props,i=new ArrayBuffer(u),o=new Float32Array(i),s=new Uint32Array(i),c=(e,t)=>o.set([t[0]/255,t[1]/255,t[2]/255,(t[3]??255)/255],e/4);c(0,r.color??[255,255,255,255]),c(16,r.noDataColor??[0,0,0,0]);let l=r.palette??[[255,255,255,255]];for(let e=0;e<8;e++)c(32+e*16,l[e%l.length]);o.set(r.valueRange??[0,1],160/4),o[168/4]=r.opacity??1,s[172/4]=a[r.colormap??`uniform`],s[176/4]=r.values?r.valueFormat===`uint32`?1:2:0,s[180/4]=r.valueStride??1,s[184/4]=r.valueOffset??0,s[188/4]=Math.min(8,l.length),s[192/4]=r.noDataValue??4294967295,s[196/4]=r.selectedRow!==void 0&&r.selectedRow>=0?r.selectedRow:4294967295,n.write(new Uint8Array(i)),t.setInstanceCount(r.triangleCount),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n,placeholder:r}=this.state;t.destroy(),n.destroy(),r.destroy(),super.finalizeState(e)}};export{c as n,f as t};