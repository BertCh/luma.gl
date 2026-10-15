import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{c as a,l as o}from"./index-Bjk18V2I.js";var s={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},c=8,l=208,u=.12,d=`
struct BundleStyle {
  palette: array<vec4<f32>, ${c}>,
  flatColor: vec4<f32>,
  maximumValue: f32,
  widthMin: f32,
  widthMax: f32,
  opacity: f32,
  pointsPerPath: u32,
  pathCount: u32,
  useMask: u32,
  useFlatColor: u32,
  straight: u32,
  widthByValue: u32,
  compareSide: u32,
  heaviestLast: u32,
  compareDivider: f32,
  highlightMask: u32,
  dimAlpha: f32,
  padding: f32,
};

@group(0) @binding(auto) var<uniform> bundleStyle: BundleStyle;
@group(0) @binding(auto) var<storage, read> bundlePaths: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> bundleValues: array<f32>;
@group(0) @binding(auto) var<storage, read> bundleClasses: array<f32>;
@group(0) @binding(auto) var<storage, read> bundleMask: array<u32>;

struct BundleVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
  @location(2) widths: vec2<f32>,
};

fn getHiddenBundleVertex() -> BundleVertexOutput {
  var output: BundleVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  output.widths = vec2<f32>(0.0);
  return output;
}

// Control point of a path; with straight on, the straight line between the pinned ends.
fn getBundlePoint(path: u32, point: u32) -> vec2<f32> {
  let pointsPerPath = bundleStyle.pointsPerPath;
  if (bundleStyle.straight != 0u) {
    let first = bundlePaths[path * pointsPerPath];
    let last = bundlePaths[path * pointsPerPath + pointsPerPath - 1u];
    return mix(first, last, f32(point) / f32(max(pointsPerPath, 2u) - 1u));
  }
  return bundlePaths[path * pointsPerPath + point];
}

fn projectBundlePoint(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

fn getBundleScreen(clipPosition: vec4<f32>) -> vec2<f32> {
  return clipPosition.xy / clipPosition.w * project.viewportSize;
}

fn getBundleDirection(fromPoint: vec2<f32>, toPoint: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
  let delta = toPoint - fromPoint;
  let deltaLength = length(delta);
  if (deltaLength < 1e-4) {
    return fallback;
  }
  return delta / deltaLength;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> BundleVertexOutput {
  let segmentsPerPath = max(bundleStyle.pointsPerPath, 2u) - 1u;
  let slot = instanceIndex / segmentsPerPath;
  let segment = instanceIndex % segmentsPerPath;
  // Paths are ranked heaviest first: drawing the last rank first puts the heaviest on top.
  var path = slot;
  if (bundleStyle.heaviestLast != 0u) {
    path = bundleStyle.pathCount - 1u - slot;
  }
  if (bundleStyle.useMask != 0u && bundleMask[path] == 0u) {
    return getHiddenBundleVertex();
  }
  let classIndex = u32(max(bundleClasses[path], 0.0));
  var color = bundleStyle.flatColor;
  if (bundleStyle.useFlatColor == 0u) {
    color = bundleStyle.palette[min(classIndex, ${c-1}u)];
  }
  // A transparent slot draws nothing: skip the projection (per-class passes rely on this).
  if (color.a <= 0.0) {
    return getHiddenBundleVertex();
  }
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let lastPoint = max(bundleStyle.pointsPerPath, 2u) - 1u;
  let point = segment + u32(corner.x);
  let hasPrevious = point > 0u;
  let hasNext = point < lastPoint;

  let startPosition = getBundlePoint(path, segment);
  let endPosition = getBundlePoint(path, segment + 1u);
  if (startPosition.x != startPosition.x || endPosition.x != endPosition.x) {
    return getHiddenBundleVertex();
  }
  let startClip = projectBundlePoint(startPosition);
  let endClip = projectBundlePoint(endPosition);
  let segmentDirection = getBundleDirection(
    getBundleScreen(startClip), getBundleScreen(endClip), vec2<f32>(1.0, 0.0)
  );

  // Both quads that meet at a control point share its offset (the mean of the two directions), so
  // the ribbon has no gaps or overlaps at the joints.
  var centerClip = startClip;
  if (point != segment) {
    centerClip = endClip;
  }
  let center = getBundleScreen(centerClip);
  var incoming = segmentDirection;
  if (hasPrevious) {
    let previousClip = projectBundlePoint(getBundlePoint(path, select(point, point - 1u, hasPrevious)));
    incoming = getBundleDirection(getBundleScreen(previousClip), center, segmentDirection);
  }
  var outgoing = segmentDirection;
  if (hasNext) {
    let nextClip = projectBundlePoint(getBundlePoint(path, select(point, point + 1u, hasNext)));
    outgoing = getBundleDirection(center, getBundleScreen(nextClip), segmentDirection);
  }
  let tangent = getBundleDirection(vec2<f32>(0.0), incoming + outgoing, segmentDirection);
  let normal = vec2<f32>(-tangent.y, tangent.x);
  let miter = 1.0 / max(dot(tangent, segmentDirection), 0.5);

  let value = bundleValues[path];
  let strength = sqrt(clamp(value / max(bundleStyle.maximumValue, 1e-20), 0.0, 1.0));
  var widthPixels = bundleStyle.widthMin;
  if (bundleStyle.widthByValue != 0u) {
    widthPixels = mix(bundleStyle.widthMin, bundleStyle.widthMax, strength);
  }
  // The quad is a pixel wider than the line so the edge can fade; coverage is set in the fragment.
  let halfWidth = widthPixels * 0.5;
  let halfGeometry = halfWidth + 0.5;

  if (bundleStyle.highlightMask != 0u && ((bundleStyle.highlightMask >> min(classIndex, 31u)) & 1u) == 0u) {
    color.a = color.a * bundleStyle.dimAlpha;
  }
  color.a = color.a * bundleStyle.opacity;

  var output: BundleVertexOutput;
  output.position = vec4<f32>(
    centerClip.xy + project_pixel_size_to_clipspace(normal * corner.y * halfGeometry * miter),
    centerClip.z,
    centerClip.w
  );
  output.side = corner.y;
  output.color = color;
  output.widths = vec2<f32>(halfGeometry, halfWidth);
  return output;
}

@fragment fn fragmentMain(input: BundleVertexOutput) -> @location(0) vec4<f32> {
  // Swipe compare: side a shows left of the divider, side b right of it (3 = hidden entirely).
  if (bundleStyle.compareSide == 3u) {
    discard;
  }
  if (bundleStyle.compareSide != 0u) {
    let isLeft = input.position.x < bundleStyle.compareDivider;
    if (select(isLeft, !isLeft, bundleStyle.compareSide == 1u)) {
      discard;
    }
  }
  let edgeDistance = abs(input.side) * input.widths.x;
  let coverage = clamp(input.widths.y + 0.5 - edgeDistance, 0.0, 1.0);
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`,f=class extends n{static layerName=`BixiBundleRibbonLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:s};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:l,usage:r.UNIFORM|r.COPY_DST}),a=new i(t,{...this.getShaders({modules:[e],source:d}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:this.getInstanceCount(),bufferLayout:[],bindings:this.getBindings(n),parameters:this.props.parameters??s});this.setState({model:a,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n)),t.setInstanceCount(this.getInstanceCount())}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n),t.setInstanceCount(this.getInstanceCount()),t.draw(e)}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}getInstanceCount(){return this.props.pathCount*Math.max(this.props.pointsPerPath-1,1)}getBindings(e){return{bundleStyle:e,bundlePaths:this.props.paths,bundleValues:this.props.values,bundleClasses:this.props.classes,bundleMask:this.props.edgeMask??this.props.values}}writeStyle(e){let t=this.props,n=new ArrayBuffer(l),r=new Float32Array(n),i=new Uint32Array(n);t.palette.slice(0,c).forEach((e,t)=>{r.set([e[0]/255,e[1]/255,e[2]/255,e[3]/255],t*4)});let a=t.flatColor??[255,255,255,255];r.set([a[0]/255,a[1]/255,a[2]/255,a[3]/255],32),r[36]=t.maximumValue,r[37]=t.widthMinPixels??1,r[38]=t.widthMaxPixels??t.widthMinPixels??1,r[39]=t.opacity??1,i[40]=t.pointsPerPath,i[41]=t.pathCount,i[42]=t.edgeMask?1:0,i[43]=t.flatColor?1:0,i[44]=t.straight?1:0,i[45]=t.widthByValue?1:0;let o=this.getCompareUniforms();i[46]=o.side,i[47]=t.heaviestLast===!1?0:1,r[48]=o.divider,i[49]=(t.highlightClasses??[]).reduce((e,t)=>t>=0&&t<32?e|1<<t:e,0),r[50]=u,e.write(new Uint8Array(n))}getCompareUniforms(){let{compareSide:e}=this.props,t=o();if(!e||!t)return{side:0,divider:0};if(t.showing!==`both`)return{side:t.showing===e?0:3,divider:0};let n=this.context.viewport,r=1;try{r=this.context.device.getDefaultCanvasContext().cssToDeviceRatio()}catch{}let i=((n?.x??0)+Math.min(Math.max(t.position,0),1)*(n?.width??1))*r;return{side:e===`a`?1:2,divider:i}}},p=`color.a = color.a * segmentWeights[row];`,m=class extends a{static layerName=`BixiRideTrailLayer`;constructor(e){super(e)}getShaderSource(){let e=super.getShaderSource();if(!e.includes(p))throw Error(`RideTrailLayer could not find the segment weight line`);return e.replace(p,`color.a = color.a * segmentWeights[row] * segmentWeights[row];`)}};export{m as n,f as t};