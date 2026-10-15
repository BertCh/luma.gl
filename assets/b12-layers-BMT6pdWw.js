import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{d as a,m as o,y as s}from"./index-Bjk18V2I.js";var c={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},l=`
fn getMovementClipPosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}
`,u=`
var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`,d=class extends n{static layerName=`MovementLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:c};getAttributeManager(){return null}getDrawCommandIndex(){return 0}getInstanceCount(){return 0}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:this.getStyleByteLength(),usage:r.UNIFORM|r.COPY_DST}),o=new i(t,{...this.getShaders({modules:[e],source:this.props.analysisFrame?this.getSource().replace(l,`${a(this.props.analysisFrame)}
fn getMovementClipPosition(position: vec2<f32>) -> vec4<f32> {
  return projectSpatialAnalysisPosition(position);
}`):this.getSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:this.getBindings(n),parameters:c});this.setState({model:o,styleBuffer:n})}updateState(e){if(super.updateState(e),e.props.analysisFrame!==e.oldProps.analysisFrame){let{model:e,styleBuffer:t}=this.state;e.destroy(),t.destroy(),this.initializeState(this.context)}let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n))}getModels(){return[this.state.model]}_drawLayer(e){let{shaderModuleProps:n}=e;this.props.analysisFrame&&n?.project&&(n={...n,project:{...n.project,coordinateSystem:t.LNGLAT,coordinateOrigin:[0,0,0]}}),super._drawLayer({...e,shaderModuleProps:n})}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n);let r=this.getDrawCommands();r?(t.setInstanceCount(0),t.draw(e),r.draw(e,this.getDrawCommandIndex())):(t.setInstanceCount(this.getInstanceCount()),t.draw(e))}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}};function f(e,t,n){e[t]=n[0]/255,e[t+1]=n[1]/255,e[t+2]=n[2]/255,e[t+3]=(n[3]??255)/255}var p=208,m=`
struct VesselStyle {
  palette: array<vec4<f32>, 8>,
  outlineColor: vec4<f32>,
  sizePixels: f32,
  speedForFullColor: f32,
  opacity: f32,
  colorMode: u32,
  categoryFilter: u32,
  colormap: u32,
  selectedTrack: u32,
  paletteSize: u32,
  stoppedSpeed: f32,
  speedClassCount: u32,
  _padding2: u32,
  _padding3: u32,
  speedClassBreaks: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> vesselStyle: VesselStyle;
@group(0) @binding(auto) var<storage, read> vesselIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vesselPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> vesselHeadings: array<f32>;
@group(0) @binding(auto) var<storage, read> vesselSpeeds: array<f32>;
@group(0) @binding(auto) var<storage, read> vesselCategories: array<u32>;

struct VesselVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) highlight: f32,
  @location(3) stopped: f32,
};
${l}
${u}
${s()}

fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
  return length(p - a - ab * t);
}

fn triangleSign(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
}

fn insideTriangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
  let d1 = triangleSign(p, a, b);
  let d2 = triangleSign(p, b, c);
  let d3 = triangleSign(p, c, a);
  let negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
  let positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
  return !(negative && positive);
}

// Signed distance to an arrowhead: tip a, wings b and c, notch n. Negative inside.
fn arrowDistance(p: vec2<f32>) -> f32 {
  let a = vec2<f32>(1.0, 0.0);
  let b = vec2<f32>(-0.8, 0.7);
  let n = vec2<f32>(-0.35, 0.0);
  let c = vec2<f32>(-0.8, -0.7);
  var d = segmentDistance(p, a, b);
  d = min(d, segmentDistance(p, b, n));
  d = min(d, segmentDistance(p, n, c));
  d = min(d, segmentDistance(p, c, a));
  let inside = insideTriangle(p, a, b, n) || insideTriangle(p, a, n, c);
  return select(d, -d, inside);
}

// Signed distance to the axis-aligned square drawn for a stopped vessel. Negative inside.
fn stoppedSquareDistance(p: vec2<f32>) -> f32 {
  let q = abs(p) - vec2<f32>(0.42);
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// Class of a speed against up to four ascending breaks (m/s): 0 below the first break.
fn getSpeedClass(speed: f32) -> u32 {
  var speedClass = 0u;
  let breakCount = min(vesselStyle.speedClassCount, 5u) - 1u;
  for (var index = 0u; index < 4u; index = index + 1u) {
    if (index < breakCount && speed >= vesselStyle.speedClassBreaks[index]) {
      speedClass = index + 1u;
    }
  }
  return speedClass;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VesselVertexOutput {
  var output: VesselVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.local = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  output.highlight = 0.0;
  output.stopped = 0.0;
  let track = vesselIds[instanceIndex];
  let category = vesselCategories[track];
  if (vesselStyle.categoryFilter != 0xffffffffu && vesselStyle.categoryFilter != category) {
    return output;
  }
  let heading = vesselHeadings[track];
  let speed = vesselSpeeds[track];
  let isStopped = vesselStyle.stoppedSpeed > 0.0 && speed < vesselStyle.stoppedSpeed;
  // A stopped vessel is an unrotated square: its heading carries no meaning.
  let rotation = select(vec2<f32>(cos(heading), sin(heading)), vec2<f32>(1.0, 0.0), isStopped);
  output.stopped = select(0.0, 1.0, isStopped);
  let corner = QUAD_CORNERS[vertexIndex] * 1.2;
  let rotated = vec2<f32>(
    corner.x * rotation.x - corner.y * rotation.y,
    corner.x * rotation.y + corner.y * rotation.x
  );
  var scale = vesselStyle.sizePixels;
  if (track == vesselStyle.selectedTrack) {
    scale = scale * 1.6;
    output.highlight = 1.0;
  }
  var clipPosition = getMovementClipPosition(vesselPositions[track]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(rotated * scale),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.local = corner;
  if (vesselStyle.colorMode == 0u) {
    output.color = vesselStyle.palette[category % max(vesselStyle.paletteSize, 1u)];
  } else if (vesselStyle.colorMode == 2u) {
    output.color = vesselStyle.palette[min(getSpeedClass(speed), max(vesselStyle.paletteSize, 1u) - 1u)];
  } else {
    let t = clamp(speed / max(vesselStyle.speedForFullColor, 1e-6), 0.0, 1.0);
    output.color = vec4<f32>(spatialAnalysisSampleRamp(vesselStyle.colormap, t), 1.0);
  }
  return output;
}

@fragment fn fragmentMain(input: VesselVertexOutput) -> @location(0) vec4<f32> {
  let d = select(arrowDistance(input.local), stoppedSquareDistance(input.local), input.stopped > 0.5);
  let size = max(vesselStyle.sizePixels * select(1.0, 1.6, input.highlight > 0.5), 1.0);
  let aa = 1.0 / size;
  let cover = 1.0 - smoothstep(-aa * 0.5, aa * 0.5, d);
  if (cover <= 0.0) { discard; }
  let outlineWidth = 1.2 * aa;
  let edge = smoothstep(-outlineWidth - aa, -outlineWidth, d);
  var rgb = mix(input.color.rgb, vesselStyle.outlineColor.rgb, edge * vesselStyle.outlineColor.a);
  if (input.highlight > 0.5) {
    rgb = mix(rgb, vec3<f32>(1.0), edge);
  }
  return vec4<f32>(rgb, cover * vesselStyle.opacity * input.color.a);
}
`,h=class extends d{static layerName=`VesselMarkerLayer`;getSource(){return m}getStyleByteLength(){return p}getDrawCommands(){return this.props.drawCommands}getBindings(e){return{vesselStyle:e,vesselIds:this.props.ids,vesselPositions:this.props.positions,vesselHeadings:this.props.headings,vesselSpeeds:this.props.speeds,vesselCategories:this.props.categories}}writeStyle(e){let t=this.props,n=new ArrayBuffer(p),r=new Float32Array(n),i=new Uint32Array(n);for(let e=0;e<8;e++)f(r,e*4,t.palette[e%t.palette.length]);f(r,32,t.outlineColor??[18,22,30,230]),r[36]=t.sizePixels??8,r[37]=t.speedForFullColor??12,r[38]=t.opacity??1,i[39]=t.colorMode===`speed`?1:t.colorMode===`speedClasses`?2:0,i[40]=t.categoryFilter===null||t.categoryFilter===void 0?4294967295:t.categoryFilter,i[41]=o[t.ramp??`viridis`],i[42]=t.selectedTrack===null||t.selectedTrack===void 0?4294967295:t.selectedTrack,i[43]=Math.min(8,t.palette.length),r[44]=t.stoppedSpeed??0;let a=(t.speedClassBreaks??[]).slice(0,4);i[45]=a.length+1;for(let e=0;e<4;e++)r[48+e]=a[e]??3e38;e.write(new Uint8Array(n))}},g=144,_=`
struct StopStyle {
  baseRadiusPixels: f32,
  radiusPerSqrtSecond: f32,
  maximumRadiusPixels: f32,
  durationForFullColor: f32,
  opacity: f32,
  ringWidthPixels: f32,
  classCount: f32,
  useRingColor: f32,
  classBreaks: vec4<f32>,
  classColors: array<vec4<f32>, 5>,
  ringColor: vec4<f32>,
};

@group(0) @binding(auto) var<uniform> stopStyle: StopStyle;
@group(0) @binding(auto) var<storage, read> stopCentroids: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> stopDurations: array<f32>;

struct StopVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) ringStart: f32,
};
${l}
${u}

// Light pink for short dwells through hot pink to deep red for the longest.
fn getStopColor(t: f32) -> vec3<f32> {
  let low = vec3<f32>(1.0, 0.78, 0.88);
  let middle = vec3<f32>(1.0, 0.16, 0.5);
  let high = vec3<f32>(0.75, 0.0, 0.2);
  if (t < 0.5) {
    return mix(low, middle, t * 2.0);
  }
  return mix(middle, high, (t - 0.5) * 2.0);
}

// Exact class colour of a duration (seconds) against up to four ascending breaks.
fn getStopClassColor(duration: f32) -> vec4<f32> {
  let classCount = u32(stopStyle.classCount);
  var stopClass = 0u;
  for (var index = 0u; index < 4u; index = index + 1u) {
    if (index + 1u < classCount && duration >= stopStyle.classBreaks[index]) {
      stopClass = index + 1u;
    }
  }
  return stopStyle.classColors[min(stopClass, 4u)];
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> StopVertexOutput {
  var output: StopVertexOutput;
  let duration = stopDurations[instanceIndex];
  let centroid = stopCentroids[instanceIndex];
  let radiusPixels = min(
    stopStyle.baseRadiusPixels + stopStyle.radiusPerSqrtSecond * sqrt(max(duration, 0.0)),
    stopStyle.maximumRadiusPixels
  );
  let corner = QUAD_CORNERS[vertexIndex];
  var clipPosition = getMovementClipPosition(centroid);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * radiusPixels),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  if (stopStyle.classCount > 0.5) {
    output.color = getStopClassColor(duration);
  } else {
    output.color = vec4<f32>(
      getStopColor(clamp(sqrt(duration / stopStyle.durationForFullColor), 0.0, 1.0)),
      0.72
    );
  }
  output.ringStart = 1.0 - stopStyle.ringWidthPixels / radiusPixels;
  return output;
}

@fragment fn fragmentMain(input: StopVertexOutput) -> @location(0) vec4<f32> {
  let radius = length(input.corner);
  if (radius > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.92, 1.0, radius);
  let ring = smoothstep(input.ringStart - 0.04, input.ringStart, radius);
  let ringRgb = select(vec3<f32>(1.0), stopStyle.ringColor.rgb, stopStyle.useRingColor > 0.5);
  let ringAlpha = select(1.0, stopStyle.ringColor.a, stopStyle.useRingColor > 0.5);
  let rgb = mix(input.color.rgb, ringRgb, ring);
  let alpha = mix(input.color.a, ringAlpha, ring) * stopStyle.opacity * edge;
  return vec4<f32>(rgb, alpha);
}
`,v=class extends d{static layerName=`StopMarkerLayer`;getSource(){return _}getStyleByteLength(){return g}getDrawCommands(){return this.props.drawCommands}getBindings(e){return{stopStyle:e,stopCentroids:this.props.centroids,stopDurations:this.props.durations}}writeStyle(e){let t=this.props,n=new Float32Array(g/4),r=(t.classColors??[]).slice(0,5),i=(t.classBreaks??[]).slice(0,4);n.set([t.baseRadiusPixels??3,t.radiusPerSqrtSecond??.45,t.maximumRadiusPixels??14,t.durationForFullColor??600,t.opacity??1,t.ringWidthPixels??1.5,r.length>0?Math.min(r.length,i.length+1):0,t.ringColor?1:0]);for(let e=0;e<4;e++)n[8+e]=i[e]??3e38;for(let e=0;e<5;e++){let t=r[Math.min(e,r.length-1)];f(n,12+e*4,t??[0,0,0,0])}f(n,32,t.ringColor??[255,255,255,255]),e.write(n)}},y=`
struct KeptSegmentStyle {
  color: vec4<f32>,
  widthPixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> keptSegmentStyle: KeptSegmentStyle;
@group(0) @binding(auto) var<storage, read> vertexPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> keptIds: array<u32>;
@group(0) @binding(auto) var<storage, read> vertexLines: array<u32>;
@group(0) @binding(auto) var<storage, read> keptCount: array<u32>;

struct KeptSegmentOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
};
${l}

// Instance i connects kept vertices i and i + 1 when both belong to the same line.
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> KeptSegmentOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  var output: KeptSegmentOutput;
  output.side = 0.0;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  if (instanceIndex + 1u >= keptCount[0]) {
    return output;
  }
  let startRow = keptIds[instanceIndex];
  let endRow = keptIds[instanceIndex + 1u];
  if (vertexLines[startRow] != vertexLines[endRow]) {
    return output;
  }
  let startClip = getMovementClipPosition(vertexPositions[startRow]);
  let endClip = getMovementClipPosition(vertexPositions[endRow]);
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * keptSegmentStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  return output;
}

@fragment fn fragmentMain(input: KeptSegmentOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(keptSegmentStyle.color.rgb, keptSegmentStyle.color.a * keptSegmentStyle.opacity * coverage);
}
`,b=class extends d{static layerName=`KeptSegmentLayer`;getSource(){return y}getStyleByteLength(){return 32}getDrawCommands(){return this.props.drawCommands}getBindings(e){return{keptSegmentStyle:e,vertexPositions:this.props.positions,keptIds:this.props.keptIds,vertexLines:this.props.vertexLines,keptCount:this.props.keptCount}}writeStyle(e){let[t,n,r,i=255]=this.props.color??[255,150,40,255];e.write(Float32Array.of(t/255,n/255,r/255,i/255,this.props.widthPixels??2,this.props.opacity??1,0,0))}},x=`
struct EventStyle {
  enterColor: vec4<f32>,
  exitColor: vec4<f32>,
  sizePixels: f32,
  opacity: f32,
  _padding0: f32,
  _padding1: f32,
};

@group(0) @binding(auto) var<uniform> eventStyle: EventStyle;
@group(0) @binding(auto) var<storage, read> eventPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> eventTracks: array<u32>;
@group(0) @binding(auto) var<storage, read> eventTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> eventTypes: array<u32>;
@group(0) @binding(auto) var<storage, read> trackStartTimes: array<f32>;
// [playhead seconds, pulse seconds, show-all flag, unused]
@group(0) @binding(auto) var<storage, read> eventClock: array<f32>;

struct EventVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) pulse: f32,
};
${l}
${u}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> EventVertexOutput {
  var output: EventVertexOutput;
  let track = eventTracks[instanceIndex];
  let time = trackStartTimes[track] + eventTimes[instanceIndex];
  let age = eventClock[0] - time;
  let pulseSeconds = max(eventClock[1], 1.0);
  var color = select(eventStyle.enterColor, eventStyle.exitColor, eventTypes[instanceIndex] == 1u);
  var scale = 0.55;
  var ring = 0.0;
  if (age >= 0.0 && age <= pulseSeconds) {
    let phase = age / pulseSeconds;
    scale = 1.0 + 2.2 * phase;
    color.a = 1.0 - phase;
    ring = 1.0;
  } else if (eventClock[2] > 0.5) {
    color.a = 0.28;
  } else {
    color.a = 0.0;
  }
  let corner = QUAD_CORNERS[vertexIndex];
  var clipPosition = getMovementClipPosition(eventPositions[instanceIndex]);
  clipPosition = vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace(corner * eventStyle.sizePixels * scale),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.corner = corner;
  output.color = color;
  output.pulse = ring;
  return output;
}

@fragment fn fragmentMain(input: EventVertexOutput) -> @location(0) vec4<f32> {
  let radius = length(input.corner);
  if (radius > 1.0 || input.color.a <= 0.0) { discard; }
  let edge = 1.0 - smoothstep(0.85, 1.0, radius);
  // Pulsing events are rings; resting events are small solid dots.
  let ringMask = select(1.0, smoothstep(0.45, 0.7, radius), input.pulse > 0.5);
  return vec4<f32>(input.color.rgb, input.color.a * eventStyle.opacity * edge * ringMask);
}
`,S=class extends d{static layerName=`ZoneEventMarkerLayer`;getSource(){return x}getStyleByteLength(){return 48}getDrawCommands(){return this.props.drawCommands}getDrawCommandIndex(){return this.props.drawCommandIndex??0}getBindings(e){return{eventStyle:e,eventPositions:this.props.positions,eventTracks:this.props.eventTracks,eventTimes:this.props.eventTimes,eventTypes:this.props.eventTypes,trackStartTimes:this.props.trackStartTimes,eventClock:this.props.clock}}writeStyle(e){let t=new Float32Array(12);f(t,0,this.props.enterColor??[77,230,140,255]),f(t,4,this.props.exitColor??[255,115,64,255]),t[8]=this.props.sizePixels??5,t[9]=this.props.opacity??1,e.write(t)}};export{S as i,v as n,h as r,b as t};