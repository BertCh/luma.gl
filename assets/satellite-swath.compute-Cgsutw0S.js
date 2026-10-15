import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{n as e}from"./earth-xB16XgAa.js";import"./analysis-frame-Blu0tYdw.js";import{N as t,n,t as r}from"./resources-CwvhEOBe.js";import{n as i,r as a,t as o}from"./gpu-outline-geometry-DW4RTkmU.js";import{t as s}from"./graph-buffers-B4_80RlX.js";import{t as c}from"./summary-reader-CsgWxENf.js";import{t as l}from"./mode-kernels-Dxm2GWvI.js";import{t as u}from"./playback-BqFO3Eom.js";import"./b12-tracks-BINmCvAR.js";import{r as d,t as ee}from"./satellite-layers-CC0mTFNJ.js";import{c as te,i as f,l as ne,o as p,r as re,t as ie}from"./satellite-tracks-Bes9xzLr.js";var ae=[`OLI`,`C-SAR IW`,`MSI`,`OLCI`,`MODIS`,`TROPOMI`,`AVHRR`,`VIIRS`],m={minLongitude:-180,minLatitude:-85,cell:1,columns:360,rows:170},h=16,oe=e*Math.PI/180/1e3,se=`
const ARC_DEGREES_TO_RADIANS: f32 = 0.017453292;
fn arcUnitVector(lngLat: vec2<f32>) -> vec3<f32> {
  let lambda = lngLat.x * ARC_DEGREES_TO_RADIANS;
  let phi = lngLat.y * ARC_DEGREES_TO_RADIANS;
  let cosPhi = sin((90.0 - abs(lngLat.y)) * ARC_DEGREES_TO_RADIANS);
  return vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sin(phi));
}
fn arcAngle(u: vec3<f32>, v: vec3<f32>) -> f32 {
  return atan2(length(cross(u, v)), dot(u, v));
}
fn arcDistanceAndFraction(cellVector: vec3<f32>, startVector: vec3<f32>, endVector: vec3<f32>) -> vec2<f32> {
  let normalRaw = cross(startVector, endVector);
  let normalLength = length(normalRaw);
  let arcAngleTotal = atan2(normalLength, dot(startVector, endVector));
  if (normalLength < 1.0e-6) {
    return vec2<f32>(arcAngle(cellVector, startVector), 0.0);
  }
  let normal = normalRaw / normalLength;
  let offset = dot(cellVector, normal);
  let foot = cellVector - normal * offset;
  let footLength = length(foot);
  if (footLength > 1.0e-6) {
    let footVector = foot / footLength;
    if (dot(cross(startVector, footVector), normal) >= 0.0 && dot(cross(footVector, endVector), normal) >= 0.0) {
      let along = arcAngle(startVector, footVector);
      return vec2<f32>(asin(clamp(abs(offset), 0.0, 1.0)), clamp(along / arcAngleTotal, 0.0, 1.0));
    }
  }
  let toStart = arcAngle(cellVector, startVector);
  let toEnd = arcAngle(cellVector, endVector);
  if (toStart <= toEnd) {
    return vec2<f32>(toStart, 0.0);
  }
  return vec2<f32>(toEnd, 1.0);
}`,g=150;async function _(_){let v=te(_.datasets.get(ie)),y=new r(_.device,`satellite-swath`),b=a(h),x=m.columns*m.rows,ce=!1,S=ne(v,[4]),C=f(v,S),w=y.createBuffer(`satellite-visible`,new Uint32Array(v.satelliteCount).fill(1)),T=e=>v.satellites[e];function E(e,t){return{positions:y.createBuffer(`${e}-positions`,t.positions),offsets:y.createBuffer(`${e}-offsets`,t.offsets),outline:y.createBuffer(`${e}-outline`,t.vertexCount*b*8),times:y.createBuffer(`${e}-times`,t.timestamps),satellites:y.createBuffer(`${e}-satellites`,t.vertexSatellite)}}let D=new t(_.device,{id:`satellite-swath-nominal`}),O=[];for(let e of ae){let t=S.filter(t=>T(v.satelliteIndex[t]).sensor===e);if(!t.length)continue;let n=f(v,t),r=n.swathKm[0],a=e.replace(/[^A-Za-z0-9]/g,`-`),c=E(`nominal-${a}`,n),l=y.createParameterBuffer(`nominal-${a}-distance`,`float32`,4,i({distance:r*1e3/2}));D.add(new o({id:`outline-${a}`,positions:s(D,`${a}-positions`,c.positions,`float32x2`,n.vertexCount),geometryType:`lines`,lineOffsets:s(D,`${a}-offsets`,c.offsets,`uint32`,n.trackCount+1),spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`great-circle`,units:`meters`},joinSegments:h,parameters:l.importToGraph(D),output:{positions:s(D,`${a}-outline`,c.outline,`float32x2`,n.vertexCount*b)}})),O.push({sensor:e,subset:n,swathKm:r,...c})}let k=y.track(await D.compileAsync()),A=E(`custom`,C),le=y.createParameterBuffer(`custom-distance`,`float32`,4,i({distance:_.options.customWidth*1e3/2})),j=new t(_.device,{id:`satellite-swath-custom`});j.add(new o({id:`outline-custom`,positions:s(j,`positions`,A.positions,`float32x2`,C.vertexCount),geometryType:`lines`,lineOffsets:s(j,`offsets`,A.offsets,`uint32`,C.trackCount+1),spatialContext:{coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`great-circle`,units:`meters`},joinSegments:h,parameters:le.importToGraph(j),output:{positions:s(j,`outline`,A.outline,`float32x2`,C.vertexCount*b)}}));let M=y.track(await j.compileAsync()),N=0;for(let e=0;e<C.trackCount;e++)N+=Math.max(0,C.offsets[e+1]-C.offsets[e]-1);let P=new Float32Array(N*4),F=new Float32Array(N*2),I=new Uint32Array(N),L=0;for(let e=0;e<C.trackCount;e++)for(let t=C.offsets[e];t<C.offsets[e+1]-1;t++)P.set(C.positions.subarray(t*2,t*2+4),L*4),F[L*2]=C.timestamps[t],F[L*2+1]=C.timestamps[t+1],I[L]=C.satelliteIndex[e],L++;let ue=y.createBuffer(`coverage-segments`,P),de=y.createBuffer(`coverage-segment-times`,F),R=y.createBuffer(`coverage-half-widths`,N*4),z=y.createBuffer(`first-cover`,x*4),B=new t(_.device,{id:`satellite-coverage`});l(B,{id:`first-cover`,invocationCount:x,declarations:`const COLUMNS: u32 = ${m.columns}u;
const SEGMENTS: u32 = ${N}u;
const CELL: f32 = ${m.cell.toFixed(4)};
const MIN_LONGITUDE: f32 = ${m.minLongitude.toFixed(4)};
const MIN_LATITUDE: f32 = ${m.minLatitude.toFixed(4)};
const EARTH_RADIUS_METERS: f32 = ${e.toFixed(1)};
const RADIANS_TO_DEGREES: f32 = 57.29578;
${se}`,bindings:[{name:`segments`,view:s(B,`segments`,ue,`float32`,N*4),type:`f32`,access:`read`},{name:`segmentTimes`,view:s(B,`segment-times`,de,`float32`,N*2),type:`f32`,access:`read`},{name:`halfWidths`,view:s(B,`half-widths`,R,`float32`,N),type:`f32`,access:`read`},{name:`firstCover`,view:s(B,`first-cover`,z,`float32`,x),type:`f32`,access:`read_write`}],body:`let column = index % COLUMNS;
  let row = index / COLUMNS;
  let cellLongitude = MIN_LONGITUDE + (f32(column) + 0.5) * CELL;
  let cellLatitude = MIN_LATITUDE + (f32(row) + 0.5) * CELL;
  let cellVector = arcUnitVector(vec2<f32>(cellLongitude, cellLatitude));
  var bestTime = 1.0e30;
  for (var segment = 0u; segment < SEGMENTS; segment = segment + 1u) {
    let halfWidth = halfWidths[halfWidthsOffset + segment];
    if (halfWidth <= 0.0) {
      continue;
    }
    let base = segmentsOffset + segment * 4u;
    let startLngLat = vec2<f32>(segments[base], segments[base + 1u]);
    let endLngLat = vec2<f32>(segments[base + 2u], segments[base + 3u]);
    let halfAngle = halfWidth * 1000.0 / EARTH_RADIUS_METERS;
    // Latitude prefilter; the 0.1 degree margin covers the arc bulging past its endpoints.
    let reach = halfAngle * RADIANS_TO_DEGREES + 0.1;
    let southLatitude = min(startLngLat.y, endLngLat.y);
    let northLatitude = max(startLngLat.y, endLngLat.y);
    if (cellLatitude < southLatitude - reach || cellLatitude > northLatitude + reach) {
      continue;
    }
    let timeBase = segmentTimesOffset + segment * 2u;
    let startTime = segmentTimes[timeBase];
    let endTime = segmentTimes[timeBase + 1u];
    if (startTime >= bestTime) {
      continue;
    }
    let hit = arcDistanceAndFraction(cellVector, arcUnitVector(startLngLat), arcUnitVector(endLngLat));
    if (hit.x <= halfAngle) {
      bestTime = min(bestTime, startTime + hit.y * (endTime - startTime));
    }
  }
  firstCover[firstCoverOffset + index] = select(-1.0, bestTime, bestTime < 1.0e29);`});let V=y.track(await B.compileAsync()),fe=e=>T(e).swathKm;function pe(e,t){let n=T(e);return n.group===4?t===`all`?!0:t===`narrow`?n.swathKm<300:t===`wide`?n.swathKm>1e3:n.sensor===t:!1}function H(e){return _.options.swathMode===`custom`?_.options.customWidth:fe(e)}let U=0;function W(){let e=_.options.satellites,t=new Uint32Array(v.satelliteCount);U=0;for(let n=0;n<v.satelliteCount;n++)t[n]=pe(n,e)?1:0,U+=t[n];w.write(t);let n=new Float32Array(N);for(let e=0;e<N;e++){let r=I[e];n[e]=t[r]?H(r)/2:0}R.write(n),G=!0;let r=v.satellites.findIndex((e,n)=>t[n]===1);if(r<0){_.setReadout(`selectedWidth`,`unknown`),_.setAnnotations(`swath-width`,null);return}let i=T(r),a=H(r),o=v.satelliteIndex.findIndex(e=>e===r),s=o<0?-1:v.offsets[o];if(!(a>0)||s<0){_.setReadout(`selectedWidth`,`${i.name}: unknown width`),_.setAnnotations(`swath-width`,null);return}let c=v.positions[s*2],l=v.positions[s*2+1],u=a/(2*oe*Math.max(Math.cos(l*Math.PI/180),.1));_.setReadout(`selectedWidth`,`${i.name} · ${a.toLocaleString()} km${_.options.swathMode===`custom`?` hypothetical`:` nominal`}`),_.setAnnotations(`swath-width`,[{kind:`bracket`,id:`swath-width-bracket`,from:[c-u,l],to:[c+u,l],text:`${a.toLocaleString()} km${_.options.swathMode===`custom`?` hypothetical`:` nominal swath`}`,side:`right`,tone:`accent`,priority:900}])}let G=!0,K=!0,me=!1,q=null,J=new Float64Array(m.rows),Y=0;for(let e=0;e<m.rows;e++){let t=m.minLatitude+(e+.5)*m.cell;J[e]=Math.cos(t*Math.PI/180),Y+=J[e]*m.columns}let X=Math.ceil(v.durationSeconds/g),Z=new c(y,`first-cover`,[{buffer:z,size:x*4}],e=>{if(ce)return;let t=new Float32Array(e),n=new Float64Array(X+1),r=0;for(let e=0;e<x;e++){let i=t[e];if(i<0)continue;let a=J[Math.floor(e/m.columns)];n[Math.min(X,Math.floor(i/g))]+=a,r+=a}let i=new Float64Array(X+1),a=0;for(let e=0;e<=X;e++)a+=n[e],i[e]=a/Y;q={firstCover:t,cumulative:i,covered:r/Y},he()});function Q(e){if(!q)return`...`;for(let t=0;t<=X;t++)if(q.cumulative[t]>=e){let e=(t+1)*g;return`${Math.floor(e/3600)} h ${String(Math.round(e%3600/60)).padStart(2,`0`)} min`}return null}function he(){if(!q)return;let e=Array.from({length:X+1},(e,t)=>t*g/3600),t=Float64Array.from(q.cumulative,e=>e*100);_.setChart(`coverageChart`,{kind:`line`,xLabel:`hours since 00:00 UTC`,yLabel:`% of analysed area reached`,xDomain:[0,v.durationSeconds/3600],yDomain:[0,100],height:130,series:[{label:`covered`,x:e,y:t,area:!0}],markers:[{x:$.time/3600,label:`now`}],formatX:e=>e.toFixed(1),formatY:e=>`${Math.round(e)}`,guides:[{y:50,label:`50%`},{y:90,label:`90%`}],description:`Share of the Earth between 85 S and 85 N inside at least one swath, against time. Cells are 1 degree and weighted by area.`}),_.setReadout(`covered3h`,`${(q.covered*100).toFixed(1)}% of analysed area (85° S–85° N)`),_.setReadout(`t50`,Q(.5)),_.setReadout(`t90`,Q(.9)),_.setReadout(`t99`,Q(.99)),ge()}function ge(){if(!q)return;let e=Math.min(X,Math.floor($.time/g));_.setReadout(`coveredNow`,`${(q.cumulative[e]*100).toFixed(1)}%`)}let $=u(_,{time:`time`,play:`play`,speed:`speed`,loop:`loop`},{range:[0,v.durationSeconds],rate:1,step:30});W(),_.setReadout(`satelliteCount`,`${U} satellites, ${n(N)} track segments`);let _e=-1;return{getCompiledGraphs:()=>[k,M,V],setOption(e){switch(e){case`satellites`:case`swathMode`:W(),K=!0,_.setReadout(`satelliteCount`,`${U} satellites, ${n(N)} track segments`),_.requestLayers();break;case`customWidth`:le.write(i({distance:_.options.customWidth*1e3/2})),K=!0,W(),_.requestLayers();break;case`time`:case`play`:case`speed`:case`loop`:break;default:_.requestLayers()}},onThemeChange(){_.requestLayers()},onGroundChange(){_.requestLayers()},encode(e,t){let n=$.advance(t);_.setReadout(`clock`,p(v,n)),me||=(k.encode(e,{parameters:void 0}),!0),K&&_.options.swathMode===`custom`&&(M.encode(e,{parameters:void 0}),K=!1),G?(V.encode(e,{parameters:void 0}),G=!1,Z.request(e)):Z.flush(e);let r=Math.floor(n/g);r!==_e&&(_e=r,he())},getLayers(){let e=_.options,t=_.ground()===`dark`,n=[];if(e.showCoverage&&n.push(new ee({id:`satellite-coverage-cells`,values:z,gridSize:[m.columns,m.rows],grid:[m.minLongitude,m.minLatitude,m.cell,m.cell],valueRange:[0,v.durationSeconds],discardAtOrBelow:-.5,discardAbove:$.time,noDataColor:t?[190,205,225,160]:[55,70,100,145],hatchNoData:!0,elapsedClasses:!0,opacity:.8})),e.showSwaths){let r=re[4],i=t?[r[0],r[1],r[2],255]:[120,60,200,255],a=e.swathMode===`custom`?[{id:`custom`,subset:C,outline:A.outline,times:A.times,satellites:A.satellites}]:O.map(e=>({id:e.sensor,subset:e.subset,outline:e.outline,times:e.times,satellites:e.satellites}));if(e.compareGeometry&&e.swathMode===`custom`)for(let r of O)n.push(new d({id:`satellite-swath-nominal-compare-${r.sensor}`,positions:r.outline,times:r.times,satellites:r.satellites,visibleSatellites:w,rowsPerInput:b,vertexCount:r.subset.vertexCount*b,timeLimit:$.time,color:t?[135,190,240,255]:[36,105,168,255],opacity:Math.min(.42,e.swathOpacity+.08)}));for(let t of a)n.push(new d({id:`satellite-swath-${t.id}`,positions:t.outline,times:t.times,satellites:t.satellites,visibleSatellites:w,rowsPerInput:b,vertexCount:t.subset.vertexCount*b,timeLimit:$.time,color:i,opacity:e.swathOpacity}))}return n},getTooltip(e){if(!q||!e.coordinate)return null;let[t,n]=e.coordinate,r=Math.floor((t-m.minLongitude)/m.cell),i=Math.floor((n-m.minLatitude)/m.cell);if(r<0||r>=m.columns||i<0||i>=m.rows)return null;let a=q.firstCover[i*m.columns+r];return a<0?`not reached by any selected swath in 3 h`:`first covered at ${p(v,a)}`},destroy(){ce=!0,Z.stop(),y.destroy()}}}export{_ as createSatelliteSwath};