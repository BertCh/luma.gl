import{mt as e}from"./src-CNCKFW2b.js";import{S as t,f as n,i as r,u as i}from"./index-Bjk18V2I.js";var a=24,o=a*6,s=o+3,c=[243,239,230,235],l=2.2,u=7,d=.45,f=12;function p(e,t){let r=!!e.channels;return`
${n(e)}
${t?`@group(0) @binding(auto) var<storage, read> flowOrigins: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> flowDestinations: array<vec2<f32>>;`:`@group(0) @binding(auto) var<storage, read> flowEndpoints: array<vec4<f32>>;`}
@group(0) @binding(auto) var<storage, read> flowIds: array<u32>;
// [0] maxValue, maxWidth, minWidth, curvature; [1] startOffset, endOffset, arrowMinLength,
// arrowLengthFactor; [2] arrowWidthRatio (0 = no arrowhead), baseWidth, halo, useWeight; [3] haloColor.
@group(0) @binding(auto) var<uniform> flowParams: array<vec4<f32>, 4>;

struct FlowVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
  @location(2) @interpolate(flat) shape: vec4<f32>,
};

const FLOW_SEGMENTS = ${a}u;
const FLOW_BODY_VERTICES = ${o}u;

// Width in CSS pixels: 0.5 + k * sqrt(weight / maxValue) clamped, or the base width without weight.
fn getFlowWidthPixels(valueRow: u32) -> f32 {
  let limits = flowParams[0];
  let options = flowParams[2];
  var weight = 0.0;
  var hasWeight = false;
${r?`  if (spatialAnalysisStyle.channelWidth != SPATIAL_ANALYSIS_NO_GROUP) {
    weight = getSpatialAnalysisChannel(valueRow, spatialAnalysisStyle.channelWidth);
    hasWeight = true;
  }
`:``}  if (!hasWeight && options.w > 0.5 && spatialAnalysisStyle.valueFormat != 0u && spatialAnalysisStyle.colormap != SPATIAL_ANALYSIS_RGBA_COLORMAP) {
    let raw = styleValues[valueRow];
    weight = select(f32(raw), bitcast<f32>(raw), spatialAnalysisStyle.valueFormat == 2u) * spatialAnalysisStyle.valueScale;
    hasWeight = true;
  }
  var width = options.y;
  if (hasWeight) {
    var t = clamp(weight / max(limits.x, 1e-20), 0.0, 1.0);
    if (isSpatialAnalysisNanWord(bitcast<u32>(weight))) {
      t = 0.0;
    }
    if (spatialAnalysisStyle.widthScale == 0u) {
      t = sqrt(t);
    }
    width = 0.5 + (limits.y - 0.5) * t;
  }
  return clamp(width, limits.z, ${f.toFixed(1)});
}

fn getFlowPoint(start: vec2<f32>, control: vec2<f32>, end: vec2<f32>, t: f32) -> vec2<f32> {
  let u = 1.0 - t;
  return u * u * start + 2.0 * u * t * control + t * t * end;
}

// Clip-space position of a point of the curve.
fn getFlowClip(start: vec2<f32>, control: vec2<f32>, end: vec2<f32>, t: f32) -> vec4<f32> {
  return projectSpatialAnalysisPosition(getFlowPoint(start, control, end, t));
}

// CSS-pixel screen position of a clip-space position (y up).
fn getFlowScreen(clip: vec4<f32>) -> vec2<f32> {
  return clip.xy / clip.w * project.viewportSize * 0.5 / max(project.devicePixelRatio, 1e-3);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> FlowVertexOutput {
  var output: FlowVertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.color = vec4<f32>(0.0);
  output.local = vec2<f32>(0.0);
  output.shape = vec4<f32>(0.0);
  var row = instanceIndex;
  if (spatialAnalysisStyle.useIds != 0u) {
    row = flowIds[instanceIndex];
  }
${t?`  let ends = vec4<f32>(flowOrigins[row], flowDestinations[row]);`:`  let ends = flowEndpoints[row];`}
  let valueRow = getSpatialAnalysisValueRow(row);
  let color = getSpatialAnalysisColor(valueRow);
  if (ends.x != ends.x || ends.y != ends.y || ends.z != ends.z || ends.w != ends.w || color.a <= 0.0) {
    return output;
  }
  let limits = flowParams[0];
  let offsets = flowParams[1];
  let options = flowParams[2];
  let start = getSpatialAnalysisPosition(ends.xy);
  let end = getSpatialAnalysisPosition(ends.zw);
  let chord = end - start;
  let groundLength = length(chord);
  if (groundLength <= 0.0) {
    return output;
  }
  // Bend to the left of the travel direction (positive curvature), so A to B and B to A separate.
  let perpendicular = vec2<f32>(-chord.y, chord.x) / groundLength;
  let control = (start + end) * 0.5 + perpendicular * limits.w * groundLength;
  let startScreen = getFlowScreen(projectSpatialAnalysisPosition(start));
  let endScreen = getFlowScreen(projectSpatialAnalysisPosition(end));
  let lengthPixels = length(endScreen - startScreen);
  let width = getFlowWidthPixels(valueRow);
  let halo = options.z;
  let usable = lengthPixels - offsets.x - offsets.y;
  if (usable < 2.0) {
    return output;
  }
  var arrowLength = 0.0;
  if (options.x > 0.0) {
    arrowLength = min(max(offsets.w * width, offsets.z), usable * 0.7);
  }
  let tStart = offsets.x / lengthPixels;
  let tTip = 1.0 - offsets.y / lengthPixels;
  let tBase = tTip - arrowLength / lengthPixels;

  if (vertexIndex < FLOW_BODY_VERTICES) {
    let corners = array<vec2<f32>, 6>(
      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    let corner = corners[vertexIndex % 6u];
    let s = (f32(vertexIndex / 6u) + corner.x) / f32(FLOW_SEGMENTS);
    let t = mix(tStart, tBase, s);
    var clip = getFlowClip(start, control, end, t);
    let before = getFlowScreen(getFlowClip(start, control, end, max(t - 0.01, 0.0)));
    let after = getFlowScreen(getFlowClip(start, control, end, min(t + 0.01, 1.0)));
    var direction = vec2<f32>(1.0, 0.0);
    let directionLength = length(after - before);
    if (directionLength > 1e-6) {
      direction = (after - before) / directionLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    let across = corner.y * (width * 0.5 + halo);
    clip = vec4<f32>(clip.xy + project_pixel_size_to_clipspace(normal * across), clip.z, clip.w);
    output.position = clip;
    output.local = vec2<f32>(0.0, across);
    output.shape = vec4<f32>(0.0, width * 0.5, halo, 0.0);
  } else {
    if (arrowLength <= 0.0) {
      return output;
    }
    let baseClip = getFlowClip(start, control, end, tBase);
    let tipScreen = getFlowScreen(getFlowClip(start, control, end, tTip));
    let baseScreen = getFlowScreen(baseClip);
    var direction = vec2<f32>(1.0, 0.0);
    let directionLength = length(tipScreen - baseScreen);
    if (directionLength > 1e-6) {
      direction = (tipScreen - baseScreen) / directionLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    // Arrowhead frame: u along the direction from the base, v across. The drawn triangle is
    // grown by the halo along its two sides (not behind the base, which joins the body).
    let halfWidth = max(arrowLength * options.x, width * 0.5 + 1.0);
    let sideLength = length(vec2<f32>(halfWidth, arrowLength));
    let cornerV = (halfWidth * arrowLength + halo * sideLength) / arrowLength;
    let tipU = arrowLength + halo * sideLength / halfWidth;
    var local = vec2<f32>(tipU, 0.0);
    let index = vertexIndex - FLOW_BODY_VERTICES;
    if (index == 0u) {
      local = vec2<f32>(0.0, -cornerV);
    } else if (index == 1u) {
      local = vec2<f32>(0.0, cornerV);
    }
    let offset = direction * local.x + normal * local.y;
    output.position = vec4<f32>(baseClip.xy + project_pixel_size_to_clipspace(offset), baseClip.z, baseClip.w);
    output.local = local;
    output.shape = vec4<f32>(1.0, arrowLength, halfWidth, halo);
  }
  output.color = color;
  return output;
}

@fragment fn fragmentMain(input: FlowVertexOutput) -> @location(0) vec4<f32> {
  if (isSpatialAnalysisCompareHidden(input.position)) { discard; }
  // Signed distance in pixels inside the fill (positive) and the halo width around it.
  var distance = 0.0;
  var haloWidth = 0.0;
  if (input.shape.x < 0.5) {
    distance = input.shape.y - abs(input.local.y);
    haloWidth = input.shape.z;
  } else {
    let arrowLength = input.shape.y;
    let halfWidth = input.shape.z;
    let sideLength = length(vec2<f32>(halfWidth, arrowLength));
    distance = (halfWidth * arrowLength - halfWidth * input.local.x - arrowLength * abs(input.local.y)) / sideLength;
    haloWidth = input.shape.w;
  }
  let fillCoverage = clamp(distance + 0.5, 0.0, 1.0);
  let haloCoverage = clamp(distance + haloWidth + 0.5, 0.0, 1.0);
  let haloColor = flowParams[3];
  let alpha = mix(haloColor.a * haloCoverage, input.color.a, fillCoverage) * spatialAnalysisStyle.opacity;
  if (alpha <= 0.0) { discard; }
  return finishSpatialAnalysisColor(vec4<f32>(mix(haloColor.rgb, input.color.rgb, fillCoverage), alpha));
}
`}var m=class extends r{static layerName=`SpatialAnalysisFlowLayer`;paramsBuffer=null;getShaderFeatures(){return{channels:!!this.props.instanceChannels}}hasSplitEndpoints(){return!this.props.flows&&!!(this.props.origins&&this.props.destinations)}getShaderSource(){return p(this.getShaderFeatures(),this.hasSplitEndpoints())}getVertexCount(){return s}getParamsBuffer(){return this.paramsBuffer??=this.context.device.createBuffer({id:`${this.id}-flow-params`,byteLength:64,usage:e.UNIFORM|e.COPY_DST}),this.paramsBuffer}getBindings(e){let{flows:t,origins:n,destinations:r,ids:i}=this.props;return{...this.getStyleBindings(e),flowEndpoints:t??e,flowOrigins:n??e,flowDestinations:r??e,flowIds:i??e,flowParams:this.getParamsBuffer()}}writeLayerStyle(e){let{props:n}=this,r=this.getStyleFrame(),a=t(n.widthPixels??2,r.zoom);i(e,n,{sizePixels:a,useIds:!!n.ids,frame:r});let o=n.blending===`additive`,s=n.outlineWidthPixels===void 0?o?0:.8:Math.max(0,t(n.outlineWidthPixels,r.zoom)),p=n.outlineColor??c,m=n.colormap??`uniform`,h=n.widthFromValues??!(m===`category`||m===`mask`||m===`rgba`),g=Math.min(f,Math.max(.5,n.maxWidthPixels??8)),_=new Float32Array(16);_.set([Math.max(n.maxValue??1,1e-20),g,Math.max(.5,n.minWidthPixels??.5),n.curvature??.15,Math.max(0,n.startOffsetPixels??0),Math.max(0,n.endOffsetPixels??0),u,l,n.arrowheads===!1?0:d,a,s,h?1:0,p[0]/255,p[1]/255,p[2]/255,s>0?(p[3]??255)/255:0]),this.getParamsBuffer().write(_)}finalizeState(e){this.paramsBuffer?.destroy(),this.paramsBuffer=null,super.finalizeState(e)}};export{m as t};