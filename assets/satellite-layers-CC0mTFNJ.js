import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import{mt as r}from"./src-CNCKFW2b.js";import{i}from"./convert-colors-D-TZjU1C.js";import{m as a,y as o}from"./index-Bjk18V2I.js";var s={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},c=`
fn getSatelliteDisplayAltitude(altitude: f32, scale: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return scale * 400000.0 * log2(1.0 + max(altitude, 0.0) / 400000.0);
  }
  return scale * altitude;
}

fn getSatelliteClipPosition(position: vec2<f32>, height: f32) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, height), vec3<f32>(0.0), vec3<f32>(0.0));
  // No depth test is used, so keep the point inside the depth range instead of clipping tall orbits.
  clipPosition.z = clipPosition.w * 0.5;
  return clipPosition;
}

// 0 at 200 km, 1 at 25,600 km (seven doublings), matching the legend extent.
fn getSatelliteAltitudeT(altitude: f32) -> f32 {
  return clamp(log2(max(altitude, 1.0) / 200000.0) / 7.0, 0.0, 1.0);
}

var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
var<private> QUAD_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`,l=class extends n{static layerName=`SatelliteBaseLayer`;static defaultProps={coordinateSystem:t.LNGLAT,parameters:s};getAttributeManager(){return null}getVertexCount(){return 6}getDrawCommands(){return null}getInstanceCount(){return 0}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:this.getStyleByteLength(),usage:r.UNIFORM|r.COPY_DST}),a=new i(t,{...this.getShaders({modules:[e],source:this.getSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:this.getVertexCount(),instanceCount:0,bufferLayout:[],bindings:this.getBindings(n),parameters:s});this.setState({model:a,styleBuffer:n})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n}=this.state;t.setBindings(this.getBindings(n))}getModels(){return[this.state.model]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;this.writeStyle(n);let r=this.getDrawCommands();r?(t.setInstanceCount(0),t.draw(e),r.draw(e,0)):(t.setInstanceCount(this.getInstanceCount()),t.draw(e))}finalizeState(e){let{model:t,styleBuffer:n}=this.state;t.destroy(),n.destroy(),super.finalizeState(e)}};function u(e,t,n){e[t]=n[0]/255,e[t+1]=n[1]/255,e[t+2]=n[2]/255,e[t+3]=(n[3]??255)/255}function d(e,t){for(let n=0;n<8;n++)u(e,n*4,t[n%t.length])}var f=192,p=`
struct MarkerStyle {
  palette: array<vec4<f32>, 8>,
  outlineColor: vec4<f32>,
  sizePixels: f32,
  opacity: f32,
  altitudeScale: f32,
  altitudeMode: u32,
  colorMode: u32,
  groupFilter: u32,
  colormap: u32,
  paletteSize: u32,
  drawStems: u32,
  selectedTrack: u32,
  stemWidth: f32,
  _padding: u32,
};

@group(0) @binding(auto) var<uniform> markerStyle: MarkerStyle;
@group(0) @binding(auto) var<storage, read> markerIds: array<u32>;
@group(0) @binding(auto) var<storage, read> markerPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> markerElevations: array<f32>;
@group(0) @binding(auto) var<storage, read> markerColors: array<u32>;
@group(0) @binding(auto) var<storage, read> markerGroups: array<u32>;
@group(0) @binding(auto) var<storage, read> markerSatellites: array<u32>;

struct MarkerOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) highlight: f32,
};
${c}
${o()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> MarkerOutput {
  var output: MarkerOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.local = vec2<f32>(0.0);
  output.color = vec4<f32>(0.0);
  output.highlight = 0.0;
  let track = markerIds[instanceIndex];
  if (markerColors[track] == 0xffffffffu) {
    return output;
  }
  if (markerStyle.groupFilter != 0xffffffffu && markerStyle.groupFilter != markerGroups[track]) {
    return output;
  }
  let altitude = markerElevations[track];
  let height = getSatelliteDisplayAltitude(altitude, markerStyle.altitudeScale, markerStyle.altitudeMode);
  var color: vec4<f32>;
  if (markerStyle.colorMode == 0u) {
    color = markerStyle.palette[markerColors[track] % max(markerStyle.paletteSize, 1u)];
  } else {
    color = vec4<f32>(spatialAnalysisSampleRamp(markerStyle.colormap, getSatelliteAltitudeT(altitude)), 1.0);
  }
  output.color = color;
  let groundPosition = markerPositions[track];
  let top = getSatelliteClipPosition(groundPosition, height);
  if (markerStyle.drawStems != 0u) {
    // A vertical line from the ground point to the satellite.
    let bottom = getSatelliteClipPosition(groundPosition, 0.0);
    let corner = SEGMENT_CORNERS[vertexIndex];
    let direction = (top.xy / top.w - bottom.xy / bottom.w) * project.viewportSize;
    let directionLength = length(direction);
    var unit = vec2<f32>(0.0, 1.0);
    if (directionLength > 1e-6) {
      unit = direction / directionLength;
    }
    let normal = vec2<f32>(-unit.y, unit.x);
    var clipPosition = mix(bottom, top, corner.x);
    clipPosition = vec4<f32>(
      clipPosition.xy + project_pixel_size_to_clipspace(normal * corner.y * markerStyle.stemWidth * 0.5),
      clipPosition.z,
      clipPosition.w
    );
    output.position = clipPosition;
    output.local = vec2<f32>(0.0, corner.y);
    output.color = vec4<f32>(color.rgb, color.a * 0.55);
    return output;
  }
  var scale = markerStyle.sizePixels;
  if (markerSatellites[track] == markerStyle.selectedTrack) {
    scale = scale * 1.8;
    output.highlight = 1.0;
  }
  let corner = QUAD_CORNERS[vertexIndex];
  output.position = vec4<f32>(
    top.xy + project_pixel_size_to_clipspace(corner * scale),
    top.z,
    top.w
  );
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: MarkerOutput) -> @location(0) vec4<f32> {
  if (markerStyle.drawStems != 0u) {
    let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.local.y));
    return vec4<f32>(input.color.rgb, input.color.a * markerStyle.opacity * coverage);
  }
  let radiusSquared = dot(input.local, input.local);
  if (radiusSquared > 1.0) { discard; }
  let coverage = 1.0 - smoothstep(0.7, 1.0, radiusSquared);
  let ring = smoothstep(0.45, 0.62, radiusSquared);
  var rgb = mix(input.color.rgb, markerStyle.outlineColor.rgb, ring * markerStyle.outlineColor.a);
  if (input.highlight > 0.5) {
    rgb = mix(rgb, vec3<f32>(1.0), ring);
  }
  return vec4<f32>(rgb, coverage * markerStyle.opacity * input.color.a);
}
`,m=class extends l{static layerName=`SatelliteMarkerLayer`;getSource(){return p}getStyleByteLength(){return f}getDrawCommands(){return this.props.drawCommands}getBindings(e){return{markerStyle:e,markerIds:this.props.ids,markerPositions:this.props.positions,markerElevations:this.props.elevations,markerColors:this.props.colors,markerGroups:this.props.groups,markerSatellites:this.props.satellites}}writeStyle(e){let t=this.props,n=new ArrayBuffer(f),r=new Float32Array(n),i=new Uint32Array(n);d(r,t.palette),u(r,32,t.outlineColor??[14,18,26,230]),r[36]=t.sizePixels??6,r[37]=t.opacity??1,r[38]=t.altitudeScale??1,i[39]=t.altitudeDisplay===`compressed`?1:0,i[40]=t.colorMode===`altitude`?1:0,i[41]=t.groupFilter===null||t.groupFilter===void 0?4294967295:t.groupFilter,i[42]=a[t.ramp??`viridis`],i[43]=Math.min(8,t.palette.length),i[44]=t.stems?1:0,i[45]=t.selectedSatellite===null||t.selectedSatellite===void 0?4294967295:t.selectedSatellite,r[46]=t.stemWidthPixels??1,e.write(new Uint8Array(n))}},h=176,g=`
struct TrailStyle {
  palette: array<vec4<f32>, 8>,
  widthPixels: f32,
  opacity: f32,
  altitudeScale: f32,
  altitudeMode: u32,
  colorMode: u32,
  groupFilter: u32,
  colormap: u32,
  paletteSize: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
  _padding3: u32,
};

@group(0) @binding(auto) var<uniform> trailStyle: TrailStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailAltitudes: array<f32>;
@group(0) @binding(auto) var<storage, read> trailSegmentEnds: array<u32>;
@group(0) @binding(auto) var<storage, read> trailSegmentTracks: array<u32>;
@group(0) @binding(auto) var<storage, read> trailIds: array<u32>;
@group(0) @binding(auto) var<storage, read> trailWeights: array<f32>;
@group(0) @binding(auto) var<storage, read> trailClips: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailColors: array<u32>;
@group(0) @binding(auto) var<storage, read> trailGroups: array<u32>;

struct TrailOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};
${c}
${o()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> TrailOutput {
  var output: TrailOutput;
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  let segment = trailIds[instanceIndex];
  let track = trailSegmentTracks[segment];
  if (trailColors[track] == 0xffffffffu) {
    return output;
  }
  if (trailStyle.groupFilter != 0xffffffffu && trailStyle.groupFilter != trailGroups[track]) {
    return output;
  }
  let clip = trailClips[segment];
  let weight = trailWeights[segment];
  if (clip.y <= clip.x || weight <= 0.0) {
    return output;
  }
  let endVertex = trailSegmentEnds[segment];
  let startVertex = endVertex - 1u;
  let a = trailPositions[startVertex];
  let b = trailPositions[endVertex];
  let altitudeA = trailAltitudes[startVertex];
  let altitudeB = trailAltitudes[endVertex];
  let start = mix(a, b, clip.x);
  let end = mix(a, b, clip.y);
  let startAltitude = mix(altitudeA, altitudeB, clip.x);
  let endAltitude = mix(altitudeA, altitudeB, clip.y);
  let startClip = getSatelliteClipPosition(start, getSatelliteDisplayAltitude(startAltitude, trailStyle.altitudeScale, trailStyle.altitudeMode));
  let endClip = getSatelliteClipPosition(end, getSatelliteDisplayAltitude(endAltitude, trailStyle.altitudeScale, trailStyle.altitudeMode));
  var color: vec4<f32>;
  if (trailStyle.colorMode == 0u) {
    color = trailStyle.palette[trailColors[track] % max(trailStyle.paletteSize, 1u)];
  } else {
    color = vec4<f32>(spatialAnalysisSampleRamp(trailStyle.colormap, getSatelliteAltitudeT(0.5 * (startAltitude + endAltitude))), 1.0);
  }
  let corner = SEGMENT_CORNERS[vertexIndex];
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
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * trailStyle.widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
  output.position = clipPosition;
  output.side = corner.y;
  output.color = vec4<f32>(color.rgb, color.a * weight);
  return output;
}

@fragment fn fragmentMain(input: TrailOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * trailStyle.opacity * coverage);
}
`,_=class extends l{static layerName=`SatelliteTrailLayer`;getSource(){return g}getStyleByteLength(){return h}getDrawCommands(){return this.props.drawCommands}getBindings(e){return{trailStyle:e,trailPositions:this.props.positions,trailAltitudes:this.props.altitudes,trailSegmentEnds:this.props.segmentEnds,trailSegmentTracks:this.props.segmentTracks,trailIds:this.props.ids,trailWeights:this.props.weights,trailClips:this.props.clipFractions,trailColors:this.props.colors,trailGroups:this.props.groups}}writeStyle(e){let t=this.props,n=new ArrayBuffer(h),r=new Float32Array(n),i=new Uint32Array(n);d(r,t.palette),r[32]=t.widthPixels??1.5,r[33]=t.opacity??1,r[34]=t.altitudeScale??1,i[35]=t.altitudeDisplay===`compressed`?1:0,i[36]=t.colorMode===`altitude`?1:0,i[37]=t.groupFilter===null||t.groupFilter===void 0?4294967295:t.groupFilter,i[38]=a[t.ramp??`viridis`],i[39]=Math.min(8,t.palette.length),e.write(new Uint8Array(n))}},v=112,y=`
struct CellStyle {
  baseColor: vec4<f32>,
  noDataColor: vec4<f32>,
  grid: vec4<f32>,
  valueRange: vec2<f32>,
  discardAtOrBelow: f32,
  opacity: f32,
  size: vec2<u32>,
  colormap: u32,
  sqrtScale: u32,
  valueScale: f32,
  useDiscard: u32,
  uniformColor: u32,
  hatchNoData: u32,
  discardAbove: f32,
  useDiscardAbove: u32,
  elapsedClasses: u32,
  densityClasses: u32,
};

@group(0) @binding(auto) var<uniform> cellStyle: CellStyle;
@group(0) @binding(auto) var<storage, read> cellValues: array<f32>;

struct CellOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) noData: f32,
};
${c}
${o()}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> CellOutput {
  var output: CellOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.local = vec2<f32>(0.0);
  output.noData = 0.0;
  let column = instanceIndex % cellStyle.size.x;
  let row = instanceIndex / cellStyle.size.x;
  let raw = cellValues[instanceIndex];
  let bits = bitcast<u32>(raw);
  var value = raw * cellStyle.valueScale;
  var color: vec4<f32>;
  if ((bits & 0x7fffffffu) >= 0x7f800000u) {
    color = cellStyle.noDataColor;
  } else if (cellStyle.useDiscard != 0u && value <= cellStyle.discardAtOrBelow) {
    color = cellStyle.noDataColor;
    output.noData = 1.0;
  } else if (cellStyle.useDiscardAbove != 0u && value > cellStyle.discardAbove) {
    color = cellStyle.noDataColor;
    output.noData = 1.0;
  } else if (cellStyle.uniformColor != 0u) {
    color = cellStyle.baseColor;
  } else {
    var t = clamp((value - cellStyle.valueRange.x) / max(cellStyle.valueRange.y - cellStyle.valueRange.x, 1e-20), 0.0, 1.0);
    if (cellStyle.sqrtScale != 0u) {
      t = sqrt(t);
    }
    if (cellStyle.elapsedClasses != 0u) {
      // Ordered first-reach classes: 0-30, 30-60, 60-120 and 120-180 minutes.
      if (value <= 1800.0) {
        color = vec4<f32>(0.18, 0.66, 0.72, cellStyle.baseColor.a);
      } else if (value <= 3600.0) {
        color = vec4<f32>(0.34, 0.76, 0.56, cellStyle.baseColor.a);
      } else if (value <= 7200.0) {
        color = vec4<f32>(0.95, 0.70, 0.22, cellStyle.baseColor.a);
      } else {
        color = vec4<f32>(0.72, 0.34, 0.70, cellStyle.baseColor.a);
      }
    } else if (cellStyle.densityClasses != 0u) {
      // Six equal-interval, ordered ice-to-violet density classes; breaks are supplied by the
      // scene's observed maximum and shared with its class-table legend.
      let level = min(5u, u32(floor(t * 6.0)));
      let colors = array<vec3<f32>, 6>(
        vec3<f32>(0.80, 0.91, 0.96), vec3<f32>(0.61, 0.80, 0.91),
        vec3<f32>(0.45, 0.66, 0.84), vec3<f32>(0.43, 0.52, 0.76),
        vec3<f32>(0.48, 0.37, 0.68), vec3<f32>(0.34, 0.20, 0.52)
      );
      color = vec4<f32>(colors[level], cellStyle.baseColor.a);
    } else {
      color = vec4<f32>(spatialAnalysisSampleRamp(cellStyle.colormap, t), cellStyle.baseColor.a);
    }
  }
  if (color.a <= 0.0) {
    return output;
  }
  let corner = QUAD_CORNERS[vertexIndex] * 0.5 + vec2<f32>(0.5);
  let lngLat = vec2<f32>(
    cellStyle.grid.x + (f32(column) + corner.x) * cellStyle.grid.z,
    cellStyle.grid.y + (f32(row) + corner.y) * cellStyle.grid.w
  );
  output.position = getSatelliteClipPosition(lngLat, 0.0);
  output.color = color;
  output.local = corner;
  return output;
}

@fragment fn fragmentMain(input: CellOutput) -> @location(0) vec4<f32> {
  if (cellStyle.hatchNoData != 0u && input.noData > 0.5) {
    let stripe = fract((input.local.x + input.local.y) * 12.0);
    if (stripe > 0.22) { discard; }
  }
  return vec4<f32>(input.color.rgb, input.color.a * cellStyle.opacity);
}
`,b=class extends l{static layerName=`SatelliteCellLayer`;getSource(){return y}getStyleByteLength(){return v}getInstanceCount(){return this.props.gridSize[0]*this.props.gridSize[1]}getBindings(e){return{cellStyle:e,cellValues:this.props.values}}writeStyle(e){let t=this.props,n=new ArrayBuffer(v),r=new Float32Array(n),i=new Uint32Array(n);u(r,0,t.color??[255,255,255,255]),u(r,4,t.noDataColor??[0,0,0,0]),r.set(t.grid,8),r[12]=t.valueRange?.[0]??0,r[13]=t.valueRange?.[1]??1,r[14]=t.discardAtOrBelow??0,r[15]=t.opacity??1,i[16]=t.gridSize[0],i[17]=t.gridSize[1],i[18]=a[t.colormap??`viridis`],i[19]=t.sqrtScale?1:0,r[20]=t.valueScale??1,i[21]=t.discardAtOrBelow===void 0?0:1,i[22]=t.color?1:0,i[23]=t.hatchNoData?1:0,r[24]=t.discardAbove??0,i[25]=t.discardAbove===void 0?0:1,i[26]=t.elapsedClasses?1:0,i[27]=t.densityClasses?1:0,e.write(new Uint8Array(n))}},x=48,S=`
struct SwathStyle {
  color: vec4<f32>,
  timeLimit: f32,
  opacity: f32,
  rowsPerInput: u32,
  _padding0: u32,
  _padding1: vec4<u32>,
};

@group(0) @binding(auto) var<uniform> swathStyle: SwathStyle;
@group(0) @binding(auto) var<storage, read> swathPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> swathTimes: array<f32>;
@group(0) @binding(auto) var<storage, read> swathSatellites: array<u32>;
@group(0) @binding(auto) var<storage, read> swathVisible: array<u32>;

struct SwathOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};
${c}

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> SwathOutput {
  var output: SwathOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  let input = vertexIndex / swathStyle.rowsPerInput;
  if (swathTimes[input] > swathStyle.timeLimit || swathVisible[swathSatellites[input]] == 0u) {
    return output;
  }
  output.position = getSatelliteClipPosition(swathPositions[vertexIndex], 0.0);
  output.color = swathStyle.color;
  return output;
}

@fragment fn fragmentMain(input: SwathOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color.rgb, input.color.a * swathStyle.opacity);
}
`,C=class extends l{static layerName=`SatelliteSwathLayer`;getSource(){return S}getStyleByteLength(){return x}getVertexCount(){return this.props.vertexCount}getInstanceCount(){return 1}getBindings(e){return{swathStyle:e,swathPositions:this.props.positions,swathTimes:this.props.times,swathSatellites:this.props.satellites,swathVisible:this.props.visibleSatellites}}writeStyle(e){let t=this.props,n=new ArrayBuffer(x),r=new Float32Array(n),i=new Uint32Array(n);u(r,0,t.color),r[4]=t.timeLimit,r[5]=t.opacity??1,i[6]=t.rowsPerInput,e.write(new Uint8Array(n))}};export{_ as i,m as n,C as r,b as t};