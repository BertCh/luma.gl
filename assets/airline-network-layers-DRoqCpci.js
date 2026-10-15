import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{l as a}from"./index-Bjk18V2I.js";import{i as o}from"./airline-network-palette-Bkn9dW7J.js";var s={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},c={...s,blendColorDstFactor:`one`,blendAlphaDstFactor:`one`},l=s,u=12,d=256,f={all:0,within:1,between:2},p=`
struct RouteStyle {
  palette: array<vec4<f32>, ${u}>,
  neutral: vec4<f32>,
  positionOffset: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  pathOffsetCount: u32,
  drawPass: u32,
  colorMode: u32,
  compareSide: u32,
  compareDivider: f32,
  dimAlpha: f32,
  egoActive: u32,
  betweenIndex: u32,
};

@group(0) @binding(auto) var<uniform> routeStyle: RouteStyle;

struct RouteVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

// The colour of a route from its category float; alpha 0 means "not drawn in this layer".
fn getRouteColor(raw: f32) -> vec4<f32> {
  if (raw != raw) {
    return vec4<f32>(0.0);
  }
  let emphasised = raw >= 16.0;
  var slot = u32(max(raw, 0.0));
  if (emphasised) {
    slot = u32(max(raw - 16.0, 0.0));
  }
  let isBetween = slot == routeStyle.betweenIndex;
  if (routeStyle.drawPass == 1u && isBetween) {
    return vec4<f32>(0.0);
  }
  if (routeStyle.drawPass == 2u && !isBetween) {
    return vec4<f32>(0.0);
  }
  var color = routeStyle.palette[min(slot, ${u-1}u)];
  if (routeStyle.colorMode == 1u) {
    color = routeStyle.neutral;
  }
  if (routeStyle.egoActive != 0u) {
    if (emphasised) {
      color.a = max(color.a, ${o.toFixed(2)});
    } else {
      color.a = color.a * routeStyle.dimAlpha;
    }
  }
  return color;
}

fn getHiddenRouteVertex() -> RouteVertexOutput {
  var output: RouteVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

fn projectRoutePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(
    vec3<f32>(position + routeStyle.positionOffset, 0.0),
    vec3<f32>(0.0),
    vec3<f32>(0.0)
  );
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Expands a segment between two projected points into a screen-space quad corner.
fn expandRouteSegment(
  startClip: vec4<f32>,
  endClip: vec4<f32>,
  vertexIndex: u32,
  color: vec4<f32>
) -> RouteVertexOutput {
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
      (normal * corner.y + along) * routeStyle.widthPixels * 0.5
    ),
    clipPosition.z,
    clipPosition.w
  );
  var output: RouteVertexOutput;
  output.position = clipPosition;
  output.side = corner.y;
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: RouteVertexOutput) -> @location(0) vec4<f32> {
  // Swipe compare: side a shows left of the divider, side b right of it (3 = hidden entirely).
  if (routeStyle.compareSide == 3u) {
    discard;
  }
  if (routeStyle.compareSide != 0u) {
    let isLeft = input.position.x < routeStyle.compareDivider;
    if (select(isLeft, !isLeft, routeStyle.compareSide == 1u)) {
      discard;
    }
  }
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * routeStyle.opacity * coverage);
}
`,m=`
${p}
@group(0) @binding(auto) var<storage, read> pathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> pathOffsets: array<u32>;
@group(0) @binding(auto) var<storage, read> pathVertexCount: array<u32>;
@group(0) @binding(auto) var<storage, read> pathCategories: array<f32>;

// Instance i joins output vertices i and i + 1 of the same path.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> RouteVertexOutput {
  let vertex = instanceIndex;
  let count = pathVertexCount[0];
  if (vertex + 1u >= count) {
    return getHiddenRouteVertex();
  }
  // First offset row that is greater than the vertex: the next path start (or the end row).
  var low = 0u;
  var high = routeStyle.pathOffsetCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pathOffsets[middle] <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let path = max(low, 1u) - 1u;
  if (low < routeStyle.pathOffsetCount && pathOffsets[low] == vertex + 1u) {
    return getHiddenRouteVertex();
  }
  let start = pathPositions[vertex];
  let end = pathPositions[vertex + 1u];
  if (start.x != start.x || end.x != end.x) {
    return getHiddenRouteVertex();
  }
  let color = getRouteColor(pathCategories[path]);
  if (color.a <= 0.0) {
    return getHiddenRouteVertex();
  }
  return expandRouteSegment(projectRoutePosition(start), projectRoutePosition(end), vertexIndex, color);
}
`,h=`
${p}
@group(0) @binding(auto) var<storage, read> bundlePaths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bundleCategories: array<f32>;

// One line strip per path; a hidden path collapses every vertex off screen.
@vertex fn vertexMain(
  @builtin(vertex_index) pointIndex: u32,
  @builtin(instance_index) pathIndex: u32
) -> RouteVertexOutput {
  let color = getRouteColor(bundleCategories[pathIndex]);
  if (color.a <= 0.0) {
    return getHiddenRouteVertex();
  }
  let position = bundlePaths[pathIndex * routeStyle.pathOffsetCount + pointIndex];
  var output: RouteVertexOutput;
  output.position = projectRoutePosition(position);
  output.side = 0.0;
  output.color = color;
  return output;
}
`;function g(e,t,n,r){let i=new ArrayBuffer(d),o=new Float32Array(i),s=new Uint32Array(i);t.palette.slice(0,u).forEach((e,t)=>{o.set([e[0]/255,e[1]/255,e[2]/255,(e[3]??255)/255],t*4)});let c=t.neutral;o.set([c[0]/255,c[1]/255,c[2]/255,c[3]/255],48),o.set(t.positionOffset??[0,0],52),o[54]=t.widthPixels??.75,o[55]=t.opacity??1,s[56]=n,s[57]=f[t.drawPass??`all`],s[58]=t.colorMode===`neutral`?1:0;let l=a(),p=0,m=0;if(t.compareSide&&l)if(l.showing===`both`){let e=r.viewport,n=1;try{n=r.device.getDefaultCanvasContext().cssToDeviceRatio()}catch{}p=t.compareSide===`a`?1:2,m=((e?.x??0)+Math.min(Math.max(l.position,0),1)*(e?.width??1))*n}else l.showing!==t.compareSide&&(p=3);s[59]=p,o[60]=m,o[61]=t.dimAlpha??1,s[62]=t.egoActive?1:0,s[63]=8,e.write(new Uint8Array(i))}var _=class extends n{static layerName=`AirlineNetworkCategoryArcLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:s};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:d,usage:r.UNIFORM|r.COPY_DST}),a=new i(t,{...this.getShaders({modules:[e],source:m}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(n),parameters:this.props.parameters??s});this.setState({model:a,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;g(n,this.props,this.props.pathOffsetCount,this.context),t.setInstanceCount(0),t.draw(e),this.props.drawCommands.draw(e,0)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}getBindings(e){return{routeStyle:e,pathPositions:this.props.positions,pathOffsets:this.props.pathOffsets,pathVertexCount:this.props.vertexCount,pathCategories:this.props.categories}}},v=class extends n{static layerName=`AirlineNetworkCategoryPathLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:s};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:d,usage:r.UNIFORM|r.COPY_DST}),a=new i(t,{...this.getShaders({modules:[e],source:h}),id:`${this.id}-model`,topology:`line-strip`,isInstanced:!0,vertexCount:this.props.pointsPerPath,instanceCount:this.props.pathCount,bufferLayout:[],bindings:this.getBindings(n),parameters:this.props.parameters??s});this.setState({model:a,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n)),t.setInstanceCount(this.props.pathCount)}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;g(n,this.props,this.props.pointsPerPath,this.context),t.setInstanceCount(this.props.pathCount),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}getBindings(e){return{routeStyle:e,bundlePaths:this.props.paths,bundleCategories:this.props.categories}}};export{l as i,_ as n,v as r,c as t};