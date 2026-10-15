import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import"./analysis-frame-Blu0tYdw.js";import{N as e,t}from"./resources-CwvhEOBe.js";import{s as n}from"./index-Bjk18V2I.js";import{n as r,t as i}from"./gpu-column-profile-LA2oflyh.js";import{t as a}from"./graph-buffers-B4_80RlX.js";import{t as o}from"./summary-reader-CsgWxENf.js";import{s,u as c}from"./b15-common-Do8LzRH9.js";import{t as l}from"./frontier-kernel-F8YIHMy_.js";function u(e,t=new Float32Array(14)){let n=e.source??[-1e6,-1e6];return t.set([e.cellSize[0],e.cellSize[1],e.timeStep,e.gravity??9.81,e.manningCoefficient??.04,e.rainfallRate??0,e.infiltrationRate??0,e.minimumDepth??.002,e.maximumCourantNumber??.2,e.maximumDepth??30,n[0],n[1],e.sourceRadius??0,e.sourceRate??0]),t}var d=class{id;props;stepsPerEncoding;constructor(e){if(this.id=e.id??`frontier-flood`,this.props=e,this.stepsPerEncoding=e.stepsPerEncoding??2,e.width<2||e.height<2||!Number.isInteger(this.stepsPerEncoding))throw Error(`FrontierFloodSimulation requires a valid grid and integer step count`)}getCommandNodes(e){let t=[];for(let n=0;n<this.stepsPerEncoding;n++)t.push(this.createFaceNode(e,`x`,n),this.createFaceNode(e,`y`,n),this.createCellNode(e,n));return t}createFaceNode(e,t,n){let{width:r,height:i,terrain:a,parameters:o,state:s}=this.props,c=t===`x`,u=c?r+1:r,d=c?i:i+1,f=c?s.xFlux:s.yFlux;return l(e,{id:`${this.id}-${n}-${t}-flux`,operation:`FrontierFloodSimulation`,variant:`${t}-flux`,bindings:[{name:`terrain`,view:a,type:`f32`,access:`read`},{name:`depth`,view:s.depth,type:`f32`,access:`read`},{name:`parameters`,view:o,type:`f32`,access:`read`},{name:`flux`,view:f,type:`f32`,access:`read_write`}],invocationCount:u*d,declarations:`
const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${i}u;
const FACE_WIDTH: u32 = ${u}u;

fn updateFlux(previous: f32, firstCell: u32, secondCell: u32, spacing: f32) -> f32 {
  let firstTerrain = terrain[terrainOffset + firstCell];
  let secondTerrain = terrain[terrainOffset + secondCell];
  let firstDepth = depth[depthOffset + firstCell];
  let secondDepth = depth[depthOffset + secondCell];
  let firstSurface = firstTerrain + firstDepth;
  let secondSurface = secondTerrain + secondDepth;
  let flowDepth = max(max(firstSurface, secondSurface) - max(firstTerrain, secondTerrain), 0.0);
  let minimumDepth = parameters[parametersOffset + 7u];
  if (flowDepth < minimumDepth) { return 0.0; }
  let timeStep = parameters[parametersOffset + 2u];
  let gravity = parameters[parametersOffset + 3u];
  let manning = parameters[parametersOffset + 4u];
  let slope = (secondSurface - firstSurface) / spacing;
  let denominator = 1.0 + gravity * timeStep * manning * manning * abs(previous) /
    max(pow(flowDepth, 7.0 / 3.0), 1.0e-12);
  let unconstrained = (previous - gravity * flowDepth * timeStep * slope) / denominator;
  let maximumFlux = parameters[parametersOffset + 8u] * spacing * flowDepth / timeStep;
  return clamp(unconstrained, -maximumFlux, maximumFlux);
}`,body:c?`let faceColumn = index % FACE_WIDTH;
  let row = index / FACE_WIDTH;
  if (faceColumn == 0u || faceColumn == WIDTH) { flux[fluxOffset + index] = 0.0; return; }
  let left = row * WIDTH + faceColumn - 1u;
  flux[fluxOffset + index] = updateFlux(flux[fluxOffset + index], left, left + 1u, parameters[parametersOffset]);`:`let column = index % WIDTH;
  let faceRow = index / WIDTH;
  if (faceRow == 0u || faceRow == HEIGHT) { flux[fluxOffset + index] = 0.0; return; }
  let south = (faceRow - 1u) * WIDTH + column;
  flux[fluxOffset + index] = updateFlux(flux[fluxOffset + index], south, south + WIDTH, parameters[parametersOffset + 1u]);`})}createCellNode(e,t){let{width:n,height:r,parameters:i,state:a,display:o}=this.props,s=[{name:`parameters`,view:i,type:`f32`,access:`read`},{name:`xFlux`,view:a.xFlux,type:`f32`,access:`read`},{name:`yFlux`,view:a.yFlux,type:`f32`,access:`read`},{name:`depth`,view:a.depth,type:`f32`,access:`read_write`}];return o&&s.push({name:`display`,view:o,type:`f32`,access:`read_write`}),l(e,{id:`${this.id}-${t}-depth`,operation:`FrontierFloodSimulation`,variant:`depth`,bindings:s,invocationCount:n*r,declarations:`const WIDTH: u32 = ${n}u;`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let xFaceWidth = WIDTH + 1u;
  let west = xFlux[xFluxOffset + row * xFaceWidth + column];
  let east = xFlux[xFluxOffset + row * xFaceWidth + column + 1u];
  let south = yFlux[yFluxOffset + row * WIDTH + column];
  let north = yFlux[yFluxOffset + (row + 1u) * WIDTH + column];
  let cellSizeX = parameters[parametersOffset];
  let cellSizeY = parameters[parametersOffset + 1u];
  let timeStep = parameters[parametersOffset + 2u];
  let delta = vec2<f32>(f32(column), f32(row)) - vec2<f32>(
    parameters[parametersOffset + 10u], parameters[parametersOffset + 11u]
  );
  let radius = max(parameters[parametersOffset + 12u], 0.0001);
  let source = max(1.0 - length(delta) / radius, 0.0) * parameters[parametersOffset + 13u];
  let forcing = parameters[parametersOffset + 5u] + source - parameters[parametersOffset + 6u];
  let nextDepth = clamp(
    depth[depthOffset + index] + timeStep *
      (forcing + (west - east) / cellSizeX + (south - north) / cellSizeY),
    0.0,
    parameters[parametersOffset + 9u]
  );
  depth[depthOffset + index] = nextDepth;
  ${o?`display[displayOffset + index] = nextDepth;`:``}`})}},f=8,p=3;async function m(l){let m=s(l.datasets.get(`grand-canyon-dem`),f),{device:h}=l,{width:g,height:_,cellCount:v}=m,y=new t(h,`flood-lab`),b=y.createBuffer(`terrain`,c(m.elevation,g,_)),x=y.createBuffer(`depth`,new Float32Array(v)),S=y.createBuffer(`x-flux`,(g+1)*_*4),C=y.createBuffer(`y-flux`,g*(_+1)*4),w=y.createBuffer(`display`,v*4),T=y.createBuffer(`depth-statistics`,44),E=y.createParameterBuffer(`parameters`,`float32`,14),D=[Math.floor(g*.47),Math.floor(_*.46)],O=new e(h,{id:`flood-lab`});O.add(new d({width:g,height:_,terrain:a(O,`terrain`,b,`float32`,v),parameters:E.importToGraph(O),state:{depth:a(O,`depth`,x,`float32`,v),xFlux:a(O,`x-flux`,S,`float32`,(g+1)*_),yFlux:a(O,`y-flux`,C,`float32`,g*(_+1))},display:a(O,`display`,w,`float32`,v),stepsPerEncoding:p}));let k=y.track(await O.compileAsync()),A=new e(h,{id:`flood-lab-profile`});A.add(new i({id:`depth-profile`,columns:[{values:a(A,`depth`,x,`float32`,v)}],histogramBinCount:1,hyperLogLogPrecision:4,topCategoryCount:1,output:{statistics:a(A,`statistics`,T,`float32`,11)}}));let j=y.track(await A.compileAsync()),M=0,N=0,P=!0,F=2.5,I=m.groundCellSize[0]*m.groundCellSize[1],L=I*v,R=e=>{let t=0,n=Math.ceil(e);for(let r=-n;r<=n;r++)for(let i=-n;i<=n;i++)t+=Math.max(1-Math.hypot(i,r)/Math.max(e,1e-4),0);return t},z=()=>{let e=l.options,t=e.rainfall/1e3/3600*L,n=e.sourceRate/1e3*R(e.sourceRadius)*I,r=e.infiltration/1e3/3600*L;l.setReadout(`forcing`,`${(t+n).toFixed(1)} m³/s input · ≤${r.toFixed(1)} m³/s infiltration capacity`)},B=()=>{let e=l.options;E.write(u({cellSize:m.groundCellSize,timeStep:e.timeStep,rainfallRate:e.rainfall/1e3/3600,infiltrationRate:e.infiltration/1e3/3600,manningCoefficient:e.roughness,maximumDepth:20,source:D,sourceRadius:e.sourceRadius,sourceRate:e.sourceRate/1e3}))},V=()=>{x.write(new Float32Array(v)),S.write(new Float32Array((g+1)*_)),C.write(new Float32Array(g*(_+1))),w.write(new Float32Array(v)),M=0,N=0,P=!0,l.setReadout(`simulatedTime`,`0 min`)};V(),l.setReadout(`grid`,`${g.toLocaleString()} × ${_.toLocaleString()} cells · ${Math.round(m.groundCellSize[0])} m`),z();let H=!1,U=new o(y,`flood-lab-depth`,[{buffer:T,size:44}],e=>{if(H)return;let t=new Float32Array(e),n=t[r.maximum],i=t[r.sum];if(!Number.isFinite(n)||!Number.isFinite(i))return;l.setReadout(`maximumDepth`,n),l.setReadout(`storedWater`,`${Math.round(i*I).toLocaleString(`en-US`)} m³`);let a=Math.max(n,.01);l.setLegendExtent(`depth`,[0,a]),Math.abs(a-F)>F*.01&&(F=a,l.requestLayers())});return{getCompiledGraphs:()=>[k,j],encode(e,t){B(),l.options.play&&(k.encode(e,{parameters:void 0}),M+=l.options.timeStep*p,N++,N%20==0&&(P=!0),t.frameIndex%20==0&&l.setReadout(`simulatedTime`,`${Math.round(M/60)} min`)),P&&=(j.encode(e,{parameters:void 0}),U.request(e),!1),U.flush(e)},getLayers(){return[new n({id:`flood-lab-depth`,coordinateOrigin:[m.origin[0],m.origin[1],0],gridSize:[g,_],bounds:m.bounds,rowOrigin:`south`,values:w,valueFormat:`float32`,colormap:`cividis`,valueRange:[0,F],discardAtOrBelow:.002,color:[255,255,255,Math.round(l.options.opacity*255)]})]},setOption(e){e===`opacity`&&l.requestLayers(),(e===`rainfall`||e===`infiltration`||e===`sourceRate`||e===`sourceRadius`)&&z()},onAction(e){e===`reset`&&V()},onThemeChange(){l.requestLayers()},destroy(){H=!0,U.stop(),y.destroy()}}}export{m as createFloodLab};