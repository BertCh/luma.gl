import{d as e,h as t,t as n}from"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as r}from"./src-CNCKFW2b.js";import"./analysis-frame-Blu0tYdw.js";import{i}from"./convert-colors-D-TZjU1C.js";import{N as a,_ as o,h as s,r as c,t as l,y as u}from"./resources-CwvhEOBe.js";import{t as ee}from"./gpu-histogram-C4GJppav.js";import{s as d}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as f,c as p,i as m,n as h,r as g,s as _,u as v}from"./raster-band-utils-CA-sh8xt.js";import{c as y}from"./raster-grid-D2bmCwkc.js";import{n as b,r as x}from"./raster-grid-utils-D1bVIm4m.js";import{n as te,r as ne,t as re}from"./gpu-particle-advection-B0dTzuCj.js";import{h as S,s as C}from"./index-Bjk18V2I.js";import{t as w}from"./graph-buffers-B4_80RlX.js";import{t as ie}from"./summary-reader-CsgWxENf.js";import{t as ae}from"./mode-kernels-Dxm2GWvI.js";import{n as oe,t as se}from"./gpu-terrain-derivatives-BZk8xB3m.js";import{f as T,l as ce,m as E,n as D,s as O,u as k}from"./b15-common-Do8LzRH9.js";var A=`GPUTerrainFlowField`;function j(e,t,n=new Float32Array(8)){return m(`getGPUTerrainFlowFieldParameterValues`,n,8),x(n,t),n[4]=e.wind[0],n[5]=e.wind[1],n[6]=e.verticalExaggeration??1,n[7]=0,n}var le=class{id;props;constructor(e){this.id=e.id??`terrain-flow-field`,this.props=e;let{id:t}=this,{output:n}=e,r=_(t,e.grid);if(v(t,e.parameters,8),y(t,e.cellSizeMode),p(t,e.grid,e.cellSizeMode),u(n.velocities,[`float32x2`],`${t} output.velocities`),n.velocities.length!==r)throw Error(`${t} output.velocities must contain one row per cell`);s(t,[n.velocities],[...h(e.elevation),e.parameters])}getCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:a}=n.grid;f(t,e,n.elevation,[]),o(t,e,[n.parameters,r.velocities]);let s=g(e,t,n.elevation,n.grid,!0),c=s.band.storage.values,l={width:i,height:a,cellSizeMode:n.cellSizeMode??`uniform`},u=d(e,{id:`${t}-deflect`,operation:A,variant:`deflect`,bindings:[{name:`elevation`,view:c,type:`f32`,access:`read`},{name:`settings`,view:n.parameters,type:`f32`,access:`read`},{name:`velocities`,view:r.velocities,type:`f32`,access:`read_write`}],invocationCount:i*a,declarations:`${b(l)}
// Direction indices of the grid helpers: 2 = next row, 6 = previous row.
const DIRECTION_NEXT_ROW: u32 = 2u;
const DIRECTION_PREVIOUS_ROW: u32 = 6u;`,body:`let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let column0 = select(column - 1u, 0u, column == 0u);
  let column1 = min(column + 1u, GRID_WIDTH - 1u);
  let row0 = select(row - 1u, 0u, row == 0u);
  let row1 = min(row + 1u, GRID_HEIGHT - 1u);
  let center = elevation[elevationOffset + index];
  let previousColumn = elevation[elevationOffset + row * GRID_WIDTH + column0];
  let nextColumn = elevation[elevationOffset + row * GRID_WIDTH + column1];
  let previousRow = elevation[elevationOffset + row0 * GRID_WIDTH + column];
  let nextRow = elevation[elevationOffset + row1 * GRID_WIDTH + column];
  let output = 2u * index + velocitiesOffset;
  if (!isFiniteValue(center) || !isFiniteValue(previousColumn) || !isFiniteValue(nextColumn) ||
      !isFiniteValue(previousRow) || !isFiniteValue(nextRow)) {
    velocities[output] = getQuietNaN();
    velocities[output + 1u] = getQuietNaN();
    return;
  }
  let exaggeration = settings[settingsOffset + 6u];
  var gradientX = 0.0;
  if (column1 > column0) {
    let spanX = f32(column1 - column0) * getGroundCellSizeAtRowCoordinate(f32(row) + 0.5).x;
    gradientX = exaggeration * ((nextColumn - previousColumn) / spanX);
  }
  var gradientY = 0.0;
  if (row1 > row0) {
    var spanY = 0.0;
    if (row1 > row) { spanY = spanY + getD8Distance(DIRECTION_NEXT_ROW, row); }
    if (row0 < row) { spanY = spanY + getD8Distance(DIRECTION_PREVIOUS_ROW, row); }
    gradientY = exaggeration * ((nextRow - previousRow) / spanY);
  }
  let wind = vec2<f32>(settings[settingsOffset + 4u], settings[settingsOffset + 5u]);
  let gradient = vec2<f32>(gradientX, gradientY);
  let deflection = dot(wind, gradient) / (1.0 + dot(gradient, gradient));
  let velocity = wind - deflection * gradient;
  velocities[output] = velocity.x;
  velocities[output + 1u] = velocity.y;`});return[...s.nodes,u]}},M={depthWriteEnabled:!1,depthCompare:`always`,blend:!0,blendColorOperation:`add`,blendAlphaOperation:`add`,blendColorSrcFactor:`src-alpha`,blendColorDstFactor:`one-minus-src-alpha`,blendAlphaSrcFactor:`one`,blendAlphaDstFactor:`one-minus-src-alpha`},N=`
fn projectStoragePosition(position: vec2<f32>) -> vec4<f32> {
  var clipPosition = project_position_to_clipspace(vec3<f32>(position, 0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // Deck's project32 matrices use OpenGL depth; WebGPU clip space requires [0, w].
  clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
  return clipPosition;
}

// Non-finite floats are tested through the exponent bits: compilers may fold x != x.
fn isNonFiniteStorageFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u;
}
`;function P(e){let t=S[e],n=t.map(([e,t,n])=>`vec3<f32>(${(e/255).toFixed(4)}, ${(t/255).toFixed(4)}, ${(n/255).toFixed(4)})`).join(`, `);return`
fn getFlowSpeedColor(t: f32) -> vec3<f32> {
  var stops = array<vec3<f32>, ${t.length}>(${n});
  let scaled = clamp(t, 0.0, 1.0) * ${(t.length-1).toFixed(1)};
  let index = min(u32(floor(scaled)), ${t.length-2}u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index));
}
`}var F=class extends n{static layerName=`StorageModelLayer`;static defaultProps={coordinateSystem:t.METER_OFFSETS,parameters:M};getAttributeManager(){return null}initializeState({device:t}){if(t.type!==`webgpu`)throw Error(`${this.id} requires WebGPU`);let n=t.createBuffer({id:`${this.id}-style`,byteLength:this.getStyleByteLength(),usage:r.UNIFORM|r.COPY_DST}),a=t.createBuffer({id:`${this.id}-placeholder`,byteLength:16,usage:r.STORAGE|r.COPY_DST});this.setState({model:null,styleBuffer:n,placeholderBuffer:a});let o=new i(t,{...this.getShaders({modules:[e],source:this.getShaderSource()}),id:`${this.id}-model`,topology:`triangle-list`,isInstanced:!0,vertexCount:6,instanceCount:0,bufferLayout:[],bindings:{...this.getStorageBindings(a),layerStyle:n},parameters:M});this.setState({model:o})}updateState(e){super.updateState(e);let{model:t,styleBuffer:n,placeholderBuffer:r}=this.state;t&&n&&r&&t.setBindings({...this.getStorageBindings(r),layerStyle:n})}getModels(){let e=this.state.model;return e?[e]:[]}draw({renderPass:e}){let{model:t,styleBuffer:n}=this.state;if(!t||!n)return;this.writeStyle(n);let{drawCommands:r,drawCommandIndex:i=0}=this.props;r?(t.setInstanceCount(0),t.draw(e),r.draw(e,i)):(t.setInstanceCount(this.getInstanceCount()),t.draw(e))}finalizeState(e){let t=this.state;t.model?.destroy(),t.styleBuffer?.destroy(),t.placeholderBuffer?.destroy(),this.setState({model:null,styleBuffer:null,placeholderBuffer:null}),super.finalizeState(e)}},I=`
// Expands the segment (startClip, endClip) into a screen-space quad corner; x is t along the
// segment, y the side across it.
fn expandSegment(startClip: vec4<f32>, endClip: vec4<f32>, corner: vec2<f32>, widthPixels: f32) -> vec4<f32> {
  let screenDirection = (endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize;
  let directionLength = length(screenDirection);
  var direction = vec2<f32>(1.0, 0.0);
  if (directionLength > 1e-6) {
    direction = screenDirection / directionLength;
  }
  let normal = vec2<f32>(-direction.y, direction.x);
  var clipPosition = mix(startClip, endClip, corner.x);
  let along = direction * (corner.x * 2.0 - 1.0) * 0.5;
  return vec4<f32>(
    clipPosition.xy + project_pixel_size_to_clipspace((normal * corner.y + along) * widthPixels * 0.5),
    clipPosition.z,
    clipPosition.w
  );
}
`,L=`
var<private> SEGMENT_CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
);
`,R=e=>`
struct LayerStyle {
  speedRange: vec2<f32>,
  widthPixels: f32,
  opacity: f32,
  ringLength: u32,
  particleCount: u32,
  padding: vec2<u32>,
};
@group(0) @binding(auto) var<uniform> layerStyle: LayerStyle;
@group(0) @binding(auto) var<storage, read> trailPositions: array<vec2<f32>>;
@group(0) @binding(auto) var<storage, read> trailSpeeds: array<f32>;
@group(0) @binding(auto) var<storage, read> trailWords: array<u32>;
${N}
${P(e)}
${L}
${I}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) side: f32,
  @location(1) color: vec4<f32>,
};

fn hiddenOutput() -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  output.side = 0.0;
  output.color = vec4<f32>(0.0);
  return output;
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  let segmentsPerParticle = layerStyle.ringLength - 1u;
  let particle = instanceIndex / segmentsPerParticle;
  let segment = instanceIndex % segmentsPerParticle;
  if (particle >= layerStyle.particleCount) {
    return hiddenOutput();
  }
  // Frame f writes slot f % L, so the oldest slot is (f + 1) % L; segment k joins the k-th and
  // (k + 1)-th slots in age order.
  let frame = trailWords[1];
  let startSlot = (frame + 1u + segment) % layerStyle.ringLength;
  let endSlot = (frame + 2u + segment) % layerStyle.ringLength;
  let start = trailPositions[particle * layerStyle.ringLength + startSlot];
  let end = trailPositions[particle * layerStyle.ringLength + endSlot];
  let speed = trailSpeeds[particle];
  if (isNonFiniteStorageFloat(start.x) || isNonFiniteStorageFloat(end.x) ||
      isNonFiniteStorageFloat(start.y) || isNonFiniteStorageFloat(end.y)) {
    return hiddenOutput();
  }
  let startClip = projectStoragePosition(start);
  let endClip = projectStoragePosition(end);
  // A respawn fills the ring with one position: skip zero-length segments (also hides dead cells).
  let pixelLength = length((endClip.xy / endClip.w - startClip.xy / startClip.w) * project.viewportSize);
  if (pixelLength < 0.02) {
    return hiddenOutput();
  }
  let age = f32(segment + 1u) / f32(segmentsPerParticle);
  let speedRatio = (speed - layerStyle.speedRange.x) /
    max(layerStyle.speedRange.y - layerStyle.speedRange.x, 1e-20);
  var output: VertexOutput;
  output.position = expandSegment(startClip, endClip, SEGMENT_CORNERS[vertexIndex], layerStyle.widthPixels);
  output.side = SEGMENT_CORNERS[vertexIndex].y;
  output.color = vec4<f32>(getFlowSpeedColor(speedRatio), age * age * layerStyle.opacity);
  return output;
}

@fragment fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let coverage = 1.0 - smoothstep(0.55, 1.0, abs(input.side));
  return vec4<f32>(input.color.rgb, input.color.a * coverage);
}
`,ue=class extends F{static layerName=`FlowTrailLayer`;getShaderSource(){return R(this.props.ramp??`viridis`)}getStyleByteLength(){return 32}getInstanceCount(){return this.props.particleCount*(this.props.ringLength-1)}getStorageBindings(){return{trailPositions:this.props.trailPositions,trailSpeeds:this.props.speeds,trailWords:this.props.wordParameters}}writeStyle(e){let{speedRange:t,widthPixels:n=1.4,opacity:r=.9,ringLength:i,particleCount:a}=this.props,o=new ArrayBuffer(32);new Float32Array(o).set([t[0],t[1],n,r]),new Uint32Array(o).set([i,a],4),e.write(new Uint8Array(o))}},z=4e4,B=20,de=315,fe=40;async function V(e){let{device:t}=e,n=O(e.datasets.get(`grand-canyon-dem`),1),{width:r,height:i,cellCount:o}=n,s=[n.origin[0],n.origin[1],0],u=new l(t,`terrain-flow`),d=!1,f=u.createBuffer(`elevation-south-first`,k(n.elevation,r,i)),p=u.createBuffer(`hillshade`,o*4),m=u.createBuffer(`velocity`,o*8),h=u.createBuffer(`speed-ratio`,o*4),g=u.createBuffer(`ratio-histogram`,B*4),_=u.createBuffer(`particle-positions`,z*8),v=u.createBuffer(`particle-ages`,z*4),y=u.createBuffer(`particle-generations`,z*4),b=u.createBuffer(`particle-speeds`,z*4),x=u.createBuffer(`particle-trails`,z*24*8),S=u.createParameterBuffer(`derivatives-settings`,`float32`,8),A=u.createParameterBuffer(`wind-settings`,`float32`,8),M=u.createParameterBuffer(`particle-parameters`,`float32`,12),N=u.createParameterBuffer(`particle-words`,`uint32`,4),P=ce(n),F=e=>({id:`elevation`,format:`float32`,storage:{kind:`buffer`,values:w(e,`elevation`,f,`float32`,o)}}),I=new a(t,{id:`terrain-flow-setup`});I.add(new se({id:`derivatives`,grid:P,elevation:F(I),parameters:S.importToGraph(I),output:{hillshade:w(I,`hillshade`,p,`float32`,o)},cellSizeMode:n.cellSizeMode,rowDirection:`north`}));let L=u.track(await I.compileAsync()),R=new a(t,{id:`terrain-flow-field`}),V=w(R,`velocity`,m,`float32x2`,o),H=A.importToGraph(R);R.add(new le({id:`wind-field`,grid:P,elevation:F(R),parameters:H,cellSizeMode:n.cellSizeMode,output:{velocities:V}}));let U=w(R,`speed-ratio`,h,`float32`,o);ae(R,{id:`speed-ratio`,bindings:[{name:`velocity`,view:V,type:`f32`,access:`read`},{name:`settings`,view:H,type:`f32`,access:`read`},{name:`ratio`,view:U,type:`f32`,access:`read_write`}],invocationCount:o,declarations:D,body:`let u = velocity[velocityOffset + 2u * index];
  let v = velocity[velocityOffset + 2u * index + 1u];
  let wind = max(length(vec2<f32>(settings[settingsOffset + 4u], settings[settingsOffset + 5u])), 0.000001);
  let share = length(vec2<f32>(u, v)) / wind;
  ratio[ratioOffset + index] = select(share, -1.0, isNaNValue(u) || isNaNValue(v));`}),R.add(new ee({id:`speed-histogram`,input:U,output:w(R,`ratio-histogram`,g,`uint32`,B),domain:[0,1.0001]}));let W=u.track(await R.compileAsync()),G=new a(t,{id:`terrain-flow-particles`});G.add(new re({id:`particles`,velocities:w(G,`velocity`,m,`float32x2`,o),fieldWidth:r,fieldHeight:i,parameters:M.importToGraph(G),uint32Parameters:N.importToGraph(G),state:{positions:w(G,`positions`,_,`float32x2`,z),ages:w(G,`ages`,v,`uint32`,z),generations:w(G,`generations`,y,`uint32`,z)},output:{speeds:w(G,`speeds`,b,`float32`,z),trails:{positions:w(G,`trails`,x,`float32x2`,z*24),length:24}}}));let K=u.track(await G.compileAsync()),q=!0,J=!0,Y=0,X=e.options.seed,Z=()=>{let t=e.options.windBearing*Math.PI/180;A.write(j({wind:[e.options.windSpeed*Math.sin(t),e.options.windSpeed*Math.cos(t)],verticalExaggeration:e.options.exaggeration},P)),q=!0,Q()},Q=()=>{let t=e.options.windSpeed*e.options.timeStep*e.options.speedScale;e.setReadout(`travel`,`${t.toFixed(0)} m per frame at free-stream speed`),e.setReadout(`trail`,`${T(t*23)} m (24 frames)`)};S.write(oe({azimuthDegrees:de,altitudeDegrees:fe},P)),Z(),e.setReadout(`particles`,z);let $=new ie(u,`terrain-flow`,[{buffer:g,size:B*4}],t=>{if(d)return;let n=new Uint32Array(t),r=0,i=0,a=0;for(let e=0;e<B;e++)r+=n[e],e<B/2&&(i+=n[e]),e<B/10&&(a+=n[e]);r!==0&&(e.setReadout(`sheltered`,`${(i/r*100).toFixed(1)} % of cells below half the free-stream wind`),e.setReadout(`blocked`,`${(a/r*100).toFixed(1)} % of cells below a tenth`))});return{getCompiledGraphs:()=>[L,W,K],setOption(t,n,r){switch(t){case`windSpeed`:case`windBearing`:case`exaggeration`:Z(),e.requestLayers();break;case`seed`:r.seed!==X&&(X=r.seed,J=!0);break;case`timeStep`:case`speedScale`:Q();break;default:e.requestLayers()}},onAction(e){e===`reset`&&(J=!0)},encode(t,a){let o=e.options;if(a.frameIndex<2&&L.encode(t,{parameters:void 0}),(q||a.frameIndex<2)&&(W.encode(t,{parameters:void 0}),q=!1,$.request(t)),$.flush(t),o.paused&&!J)return;let s;if(o.spawnRegion===`view`){let e=c(a.viewport,n.projection);s=[Math.max(e[0],n.bounds[0]),Math.max(e[1],n.bounds[1]),Math.min(e[2],n.bounds[2]),Math.min(e[3],n.bounds[3])],(s[0]>=s[2]||s[1]>=s[3])&&(s=void 0)}M.write(te({fieldExtent:[n.bounds[0],n.bounds[1],n.displayCellSize[0],n.displayCellSize[1]],timeStep:o.timeStep,speedScale:o.speedScale,dropRate:o.dropRate,minimumSpeed:o.minimumSpeed,spawnBounds:s},[r,i])),N.write(ne({seed:o.seed,frame:Y,maximumAge:o.maximumAge,reset:J})),K.encode(t,{parameters:void 0}),Y++,J=!1},getLayers(){let t=e.options,a=e.theme()===`dark`,o={coordinateOrigin:s,gridSize:[r,i],bounds:n.bounds,rowOrigin:`south`},c=[new C({...o,id:`terrain-flow-hillshade`,values:p,valueFormat:`float32`,colormap:`grayscale`,valueRange:[0,1],color:[255,255,255,t.background===`speed`?120:a?190:235]})];return t.background===`speed`&&c.push(new C({...o,id:`terrain-flow-speed`,values:h,valueFormat:`float32`,colormap:`magma`,valueRange:[0,1],discardAtOrBelow:-.5,color:[255,255,255,215]})),c.push(new ue({id:`terrain-flow-trails-${t.ramp}`,coordinateOrigin:s,trailPositions:x,speeds:b,wordParameters:N.buffer,ringLength:24,particleCount:z,speedRange:[0,t.windSpeed*1.1],ramp:t.ramp,widthPixels:t.trailWidth,opacity:t.trailOpacity})),c},getTooltip(e){if(!e.coordinate)return null;let t=E(n,e.coordinate[0],e.coordinate[1]);return t<0?null:`Elevation ${T(n.elevation[t])} m`},destroy(){d=!0,$.stop(),u.destroy()}}}export{V as createTerrainFlow};