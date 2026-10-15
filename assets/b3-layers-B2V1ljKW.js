import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{d as a,m as o,y as s}from"./index-Bjk18V2I.js";import{t as c}from"./b3-palette-APlD7x7U.js";var l={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},u=c.map(([e,t,n])=>`vec3<f32>(${(e/255).toFixed(4)}, ${(t/255).toFixed(4)}, ${(n/255).toFixed(4)})`).join(`, `),d=448,f=`fn projectGeometryPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position + geometryStyle.positionOffset, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`;function p(e){return`${a(e)}
fn projectGeometryPosition(position: vec2<f32>) -> vec4<f32> {
  return projectSpatialAnalysisPosition(position + geometryStyle.positionOffset);
}
`}function m(e,t){if(!t)return e;if(!e.includes(f))throw Error(`Geometry shader has no projection to replace`);return e.replace(f,()=>p(t))}var h=`
struct GeometryStyle {
  color: vec4<f32>,
  valueRange: vec2<f32>,
  positionOffset: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  colormap: u32,
  valueSource: u32,
  pathOffsetCount: u32,
  directionScale: f32,
  _padding0: u32,
  _padding1: u32,
  extraFloats: vec4<f32>,
  extraWords: vec4<u32>,
  secondaryColor: vec4<f32>,
  classPalette: array<vec4<f32>, 16>,
  classBreaks: array<vec4<f32>, 4>,
  classCount: u32,
  _classPadding0: u32,
  _classPadding1: u32,
  _classPadding2: u32,
};

@group(0) @binding(auto) var<uniform> geometryStyle: GeometryStyle;
@group(0) @binding(auto) var<storage, read> geometryValues: array<f32>;

${s()}

// Maps a scalar to a color. Alpha 0 means "no data". extraWords.x: 0 ramp, 1 category, 2 flag.
fn getGeometryColor(value: f32) -> vec4<f32> {
  let bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) >= 0x7f800000u) {
    return vec4<f32>(0.0);
  }
  let mapping = geometryStyle.extraWords.x;
  if (mapping == 2u) {
    return select(geometryStyle.secondaryColor, geometryStyle.color, value != 0.0);
  }
  if (mapping == 1u) {
    let palette = array<vec3<f32>, 8>(${u});
    return vec4<f32>(palette[u32(max(value, 0.0)) % 8u], geometryStyle.color.a);
  }
  if (mapping == 3u) {
    var classIndex = 0u;
    for (var index = 0u; index + 1u < geometryStyle.classCount; index = index + 1u) {
      if (value >= geometryStyle.classBreaks[index / 4u][index % 4u]) {
        classIndex = index + 1u;
      }
    }
    return geometryStyle.classPalette[classIndex];
  }
  let range = geometryStyle.valueRange;
  var t = clamp((value - range.x) / max(range.y - range.x, 1e-20), 0.0, 1.0);
  if (geometryStyle.extraWords.y != 0u) {
    t = sqrt(t);
  }
  let rgb = spatialAnalysisSampleRamp(geometryStyle.colormap, t);
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), geometryStyle.color.a);
}

${f}
struct GeometryVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

// Expands a segment between two projected points into a screen-space quad corner.
fn expandGeometrySegment(
  startClip: vec4<f32>,
  endClip: vec4<f32>,
  vertexIndex: u32,
  color: vec4<f32>
) -> GeometryVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(
      (normal * corner.y + along) * geometryStyle.widthPixels * 0.5
    ),
    clipPosition.z,
    clipPosition.w
  );
  var output: GeometryVertexOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

fn getHiddenGeometryVertex() -> GeometryVertexOutput {
  var output: GeometryVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}
`,g=`
@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity * coverage);
}
`,_=`
@fragment fn fragmentMain(input: GeometryVertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * geometryStyle.opacity);
}
`,v=`
${h}
${g}
@group(0) @binding(auto) var<storage, read> pathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexCount: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let vertex = instanceIndex;
  let count = pathVertexCount[0];
  // extraWords.w: closed rings. Their last vertex connects back to the path start.
  let closed = geometryStyle.extraWords.w != 0u;
  if (vertex >= count || (!closed && vertex + 1u >= count)) {
    return getHiddenGeometryVertex();
  }
  // First offset row that is greater than the vertex: the next path start (or the end row).
  var low = 0u;
  var high = geometryStyle.pathOffsetCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let path = max(low, 1u) - 1u;
  var endIndex = vertex + 1u;
  if (vertex + 1u >= count || (low < geometryStyle.pathOffsetCount && pathOffsets[low] == vertex + 1u)) {
    if (!closed) {
      return getHiddenGeometryVertex();
    }
    endIndex = pathOffsets[path];
  }
  let start = pathPositions[vertex];
  let end = pathPositions[endIndex];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 1u) {
    color = getGeometryColor(geometryValues[path]);
  } else if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[vertex]);
  } else if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${u});
    color = vec4<f32>(palette[path % 8u], geometryStyle.color.a);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`,y=`
${h}
${g}
@group(0) @binding(auto) var<storage, read> keptVertexPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keptIds: array<u32>;
@group(0) @binding(auto) var<storage, read> keptVertexLines: array<u32>;
@group(0) @binding(auto) var<storage, read> keptCount: array<u32>;

// Instance i joins kept vertices i and i + 1 when they belong to the same line.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  if (instanceIndex + 1u >= keptCount[0]) {
    return getHiddenGeometryVertex();
  }
  let startRow = keptIds[instanceIndex];
  let endRow = keptIds[instanceIndex + 1u];
  if (keptVertexLines[startRow] != keptVertexLines[endRow]) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${u});
    color = vec4<f32>(palette[keptVertexLines[startRow] % 8u], geometryStyle.color.a);
  }
  return expandGeometrySegment(
    projectGeometryPosition(keptVertexPositions[startRow]),
    projectGeometryPosition(keptVertexPositions[endRow]),
    vertexIndex,
    color
  );
}
`,b=`
${h}
${g}
@group(0) @binding(auto) var<storage, read> pairStarts: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pairEnds: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let start = pairStarts[instanceIndex];
  var end = pairEnds[instanceIndex];
  if (geometryStyle.directionScale != 0.0) {
    // The second column is a unit direction: draw a tick of directionScale meters.
    end = start + end * geometryStyle.directionScale;
  }
  if (start.x != start.x || end.x != end.x || start.y != start.y || end.y != end.y) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[instanceIndex]);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`,x=`
${h}
${_}
@group(0) @binding(auto) var<storage, read> triangleCorners: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> triangleFeatures: array<u32>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let feature = triangleFeatures[instanceIndex];
  var color = geometryStyle.color;
  if (geometryStyle.valueSource != 0u) {
    color = getGeometryColor(geometryValues[feature]);
  }
  let corner = triangleCorners[instanceIndex * 3u + vertexIndex];
  if (color.a <= 0.0 || corner.x != corner.x || corner.y != corner.y) {
    return getHiddenGeometryVertex();
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(corner);
  output.side = 0.0;
  output.color = color;
  return output;
}
`,S=`
${h}
${_}
@group(0) @binding(auto) var<storage, read> triangleVertices: array<vec2<f32>>;

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let position = triangleVertices[instanceIndex * 3u + vertexIndex];
  if (position.x != position.x || position.y != position.y) {
    return getHiddenGeometryVertex();
  }
  var output: GeometryVertexOutput;
  output.position = projectGeometryPosition(position);
  output.side = 0.0;
  output.color = geometryStyle.color;
  return output;
}
`,C=`
${h}
${g}
@group(0) @binding(auto) var<storage, read> ringPositions: array<vec2<f32>>;

// Rings are fixed-stride slots: slot s owns rows [s * stride, (s + 1) * stride); the ring closes.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GeometryVertexOutput {
  let stride = geometryStyle.extraWords.z;
  let slot = instanceIndex / stride;
  let local = instanceIndex % stride;
  var next = local + 1u;
  if (next >= stride) {
    next = 0u;
  }
  let start = ringPositions[slot * stride + local];
  let end = ringPositions[slot * stride + next];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenGeometryVertex();
  }
  var color = geometryStyle.color;
  if (geometryStyle.valueSource == 3u) {
    let palette = array<vec3<f32>, 8>(${u});
    color = vec4<f32>(palette[slot % 8u], geometryStyle.color.a);
  } else if (geometryStyle.valueSource == 2u) {
    color = getGeometryColor(geometryValues[slot]);
  }
  if (color.a <= 0.0) {
    return getHiddenGeometryVertex();
  }
  return expandGeometrySegment(
    projectGeometryPosition(start), projectGeometryPosition(end), vertexIndex, color
  );
}
`,w=class extends n{static layerName=`B3BaseLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:l};getAttributeManager(){return null}getVertexCount(){return 6}getPathOffsetCount(){return 0}getDirectionScale(){return 0}getStride(){return 1}getClosed(){return 0}initializeState({device:e}){if(e.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let t=e.createBuffer({id:`${this.id}-style`,byteLength:d,usage:r.UNIFORM|r.COPY_DST}),n=e.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST}),i=this.getFrameShaderSource(),a=this.createModel(e,i,t,n);this.setState({model:a,styleBuffer:t,placeholderBuffer:n,source:i})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r,source:i}=this.state;if(t&&n&&r){let e=this.getFrameShaderSource();if(e!==i){t.destroy(),this.setState({model:this.createModel(this.context.device,e,n,r),source:e});return}t.setBindings({...this.getBindings(r),geometryStyle:n})}}_drawLayer(e){let{shaderModuleProps:n}=e;if(!this.props.analysisFrame||!n?.project){super._drawLayer(e);return}super._drawLayer({...e,shaderModuleProps:{...n,project:{...n.project,coordinateSystem:t.LNGLAT,coordinateOrigin:[0,0,0]}}})}getFrameShaderSource(){return m(this.getShaderSource(),this.props.analysisFrame)}createModel(t,n,r,a){return new i(t,{...this.getShaders({modules:[e],source:n}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.getVertexCount(),instanceCount:0,bufferLayout:[],bindings:{...this.getBindings(a),geometryStyle:r},parameters:l})}getModels(){let e=this.state.model;return e?[e]:[]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;let r=new ArrayBuffer(d),i=new Float32Array(r),a=new Uint32Array(r),[s,c,l,u=255]=this.props.color??[255,255,255,255];i.set([s/255,c/255,l/255,u/255],0),i.set(this.props.valueRange??[0,1],4),i.set(this.props.positionOffset??[0,0],6),i[8]=this.props.widthPixels??2,i[9]=this.props.opacity??1,a[10]=o[this.props.colormap??`viridis`],a[11]=this.getValueSource(),a[12]=this.getPathOffsetCount(),i[13]=this.getDirectionScale(),a[20]={ramp:0,category:1,flag:2,class:3}[this.props.valueMapping??`ramp`],a[21]=this.props.sqrtScale?1:0,a[22]=this.getStride(),a[23]=this.getClosed();let[f,p,m,h=0]=this.props.secondaryColor??[0,0,0,0];if(i.set([f/255,p/255,m/255,h/255],24),(this.props.valueMapping??`ramp`)===`class`){let e=(this.props.classBreaks??this.props.classTable?.breaks??[]).slice(0,15),t=this.props.classColors??this.props.classTable?.colors??[];a[108]=e.length+1,i.set(e,92);for(let e=0;e<16;e++){let[n,r,a,o=255]=t[e]??t[t.length-1]??[255,255,255,255];i.set([n/255,r/255,a/255,o/255],28+e*4)}}n.write(new Uint8Array(r)),this.drawInstances(t,e)}finalizeState(e){let t=this.state;t.model?.destroy(),t.styleBuffer?.destroy(),t.placeholderBuffer?.destroy(),this.setState({model:null,styleBuffer:null,placeholderBuffer:null,source:``}),super.finalizeState(e)}},T=class extends w{static layerName=`B3PathOutputLayer`;getShaderSource(){return v}getBindings(e){return{geometryValues:this.props.values??e,pathPositions:this.props.positions,pathOffsets:this.props.pathOffsets,pathVertexCount:this.props.vertexCount}}getValueSource(){let e=this.props.colorSource??`uniform`;return{uniform:0,"path-value":1,"vertex-value":2,"path-index":3}[e]}getPathOffsetCount(){return this.props.pathOffsetCount}getClosed(){return this.props.closed?1:0}drawInstances(e,t){e.setInstanceCount(0),e.draw(t),this.props.drawCommands.draw(t,0)}},E=class extends w{static layerName=`B3PairSegmentLayer`;getShaderSource(){return b}getBindings(e){return{geometryValues:this.props.values??e,pairStarts:this.props.starts,pairEnds:this.props.ends}}getValueSource(){return this.props.values?2:0}getDirectionScale(){return this.props.directionScale??0}drawInstances(e,t){e.setInstanceCount(this.props.instanceCount),e.draw(t)}},D=class extends w{static layerName=`B3FeatureTriangleLayer`;getShaderSource(){return x}getVertexCount(){return 3}getBindings(e){return{geometryValues:this.props.values??e,triangleCorners:this.props.corners,triangleFeatures:this.props.featureRows}}getValueSource(){return this.props.values?2:0}drawInstances(e,t){e.setInstanceCount(this.props.instanceCount),e.draw(t)}},O=class extends w{static layerName=`B3TriangleListLayer`;getShaderSource(){return S}getVertexCount(){return 3}getBindings(e){return{geometryValues:e,triangleVertices:this.props.positions}}getValueSource(){return 0}drawInstances(e,t){e.setInstanceCount(this.props.triangleCount),e.draw(t)}},k=class extends w{static layerName=`B3RingEdgeLayer`;getShaderSource(){return C}getBindings(e){return{geometryValues:this.props.values??e,ringPositions:this.props.positions}}getValueSource(){let e=this.props.colorSource??`uniform`;return{uniform:0,"slot-value":2,"slot-index":3}[e]}getStride(){return this.props.stride}drawInstances(e,t){e.setInstanceCount(this.props.ringCount*this.props.stride),e.draw(t)}},A=class extends w{static layerName=`B3KeptSegmentLayer`;getShaderSource(){return y}getBindings(e){return{geometryValues:e,keptVertexPositions:this.props.positions,keptIds:this.props.keptIds,keptVertexLines:this.props.vertexLines,keptCount:this.props.keptCount}}getValueSource(){return this.props.colorSource===`line-index`?3:0}drawInstances(e,t){e.setInstanceCount(0),e.draw(t),this.props.drawCommands.draw(t,0)}};export{k as a,T as i,A as n,O as o,E as r,D as t};