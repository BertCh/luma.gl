import{i as e,r as t}from"./earth-xB16XgAa.js";import{n}from"./gpu-raster-grid-BOSvDY5A.js";import{H as r,T as i,k as a,w as o}from"./index-Bjk18V2I.js";import{t as s}from"./mode-kernels-Dxm2GWvI.js";function c(e,t,n,r){if(r<=1)return{values:e,width:t,height:n};let i=Math.floor(t/r),a=Math.floor(n/r),o=new Float32Array(i*a),s=r*r;for(let n=0;n<a;n++)for(let a=0;a<i;a++){let c=0;for(let i=0;i<r;i++){let o=(n*r+i)*t+a*r;for(let t=0;t<r;t++)c+=e[o+t]}o[n*i+a]=c/s}return{values:o,width:i,height:a}}function l(e,t){let i=e.raster;if(!i)throw Error(`grand-canyon-dem has no raster`);let a=i.spec,o=c(i.values,i.width,i.height,t),s=e.defaultOrigin,l=new r(s),[u,d,f,p]=i.bounds,[m,h]=l.project(u,d),[g,_]=l.project(f,p),v=a.cellSizeMercatorM*t;return{id:`grand-canyon-dem`,width:o.width,height:o.height,cellCount:o.width*o.height,elevation:o.values,landCover:null,cellSizeMode:`web-mercator`,rasterGrid:n({width:o.width,height:o.height,cellSize:v,origin:[a.boundsMercator[0],a.boundsMercator[3]]}),groundCellSize:[a.cellSizeGroundM*t,a.cellSizeGroundM*t],origin:s,projection:l,bounds:[m,h,g,_],displayCellSize:[(g-m)/o.width,(_-h)/o.height],bbox:i.bounds}}function u(n,r,i){let a=e,o=t,s=.9996,c=o/(2-o),l=a/(1+c)*(1+c*c/4+c**4/64),u=[c/2-2*c*c/3+37*c**3/96,c*c/48+c**3/15,17*c**3/480],d=[2*c-2*c*c/3-2*c**3,7*c*c/3-8*c**3/5,56*c**3/15],f=r/(s*l),p=(n-5e5)/(s*l),m=f,h=p;for(let e=1;e<=3;e++)m-=u[e-1]*Math.sin(2*e*f)*Math.cosh(2*e*p),h-=u[e-1]*Math.cos(2*e*f)*Math.sinh(2*e*p);let g=Math.asin(Math.sin(m)/Math.cosh(h)),_=g;for(let e=1;e<=3;e++)_+=d[e-1]*Math.sin(2*e*g);return[(i-1)*6-180+3+Math.atan2(Math.sinh(h),Math.cos(m))*180/Math.PI,_*180/Math.PI]}async function d(e,t){let s=e.raster;if(!s)throw Error(`dixie-fire has no raster`);let c=e.manifest,l=await o(await i(a(`dixie-fire`,c.rasters.worldcover.file),t),`uint8-classes`),[d,f,p,m]=c.raster.boundsProjected,h=u((d+p)/2,(f+m)/2,10),g=new r(h),_=(p-d)/2,v=(m-f)/2,y=c.raster.cellSizeM;return{id:`dixie-fire`,width:s.width,height:s.height,cellCount:s.width*s.height,elevation:s.values,landCover:l.values,cellSizeMode:`uniform`,rasterGrid:n({width:s.width,height:s.height,cellSize:y,origin:[d,m]}),groundCellSize:[y,y],origin:h,projection:g,bounds:[-_,-v,_,v],displayCellSize:[2*_/s.width,2*v/s.height],bbox:s.bounds}}function f(e,t,n){let[r,i]=e.projection.project(t,n),a=Math.floor((r-e.bounds[0])/e.displayCellSize[0]),o=Math.floor((e.bounds[3]-i)/e.displayCellSize[1]);return a<0||a>=e.width||o<0||o>=e.height?-1:o*e.width+a}function p(e,t,n){let[r,i]=e.projection.project(t,n),a=Math.min(e.width-1,Math.max(0,Math.floor((r-e.bounds[0])/e.displayCellSize[0])));return Math.min(e.height-1,Math.max(0,Math.floor((e.bounds[3]-i)/e.displayCellSize[1])))*e.width+a}function m(e,t){return[e.bounds[0]+(t%e.width+.5)*e.displayCellSize[0],e.bounds[3]-(Math.floor(t/e.width)+.5)*e.displayCellSize[1]]}function h(e){let{width:t,height:r,affine:i}=e.rasterGrid;return n({width:t,height:r,cellSize:[i[0],-i[4]],origin:[i[2],i[5]+i[4]*r],rowDirection:`north`})}function g(e,t,n){let r=new e.constructor(e.length);for(let i=0;i<n;i++)r.set(e.subarray(i*t,(i+1)*t),(n-1-i)*t);return r}function _(e){let t=e/1e6;return t>=100?`${t.toFixed(0)} km²`:t>=1?`${t.toFixed(1)} km²`:`${t.toFixed(2)} km²`}function v(e){return Math.round(e).toLocaleString(`en-US`)}function y(e){if(!Number.isFinite(e))return`unreached`;let t=Math.round(e),n=Math.floor(t/60),r=t%60;return n>0?`${n} h ${String(r).padStart(2,`0`)} min`:`${r} min`}var b=-9999,x=`
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn isInfiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u; }
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`;function S(e,t){s(e,{id:t.id,bindings:[{name:`source`,view:t.input,type:`f32`,access:`read`},{name:`display`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:t.cellCount,declarations:x,body:`let value = source[sourceOffset + index];
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  ${t.useLog?`let shown = log2(max(value, 0.000001)) * 0.30102999566;`:`let shown = value;`}
  display[displayOffset + index] = select(shown, ${b.toFixed(1)}, invalid);`})}function C(e,t){s(e,{id:t.id,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`filled`,view:t.filled,type:`f32`,access:`read`},{name:`depth`,view:t.output,type:`f32`,access:`read_write`}],invocationCount:t.cellCount,declarations:x,body:`let value = filled[filledOffset + index] - elevation[elevationOffset + index];
  depth[depthOffset + index] = select(value, 0.0, isNaNValue(value));`})}function w(e){let t=String(Math.fround(e));return/[.eE]/.test(t)?t:`${t}.0`}function T(e,t){let{grid:n}=t;s(e,{id:`stream-segments`,bindings:[{name:`directions`,view:t.directions,type:`u32`,access:`read`},{name:`orders`,view:t.streamOrder,type:`u32`,access:`read`},{name:`segments`,view:t.segments,type:`f32`,access:`read_write`},...t.widths.map((e,t)=>({name:`width${t}`,view:e,type:`f32`,access:`read_write`}))],invocationCount:t.cellCount,declarations:`
const GRID_WIDTH: u32 = ${n.width}u;
const MINIMUM_X: f32 = ${w(n.bounds[0])};
const MAXIMUM_Y: f32 = ${w(n.bounds[3])};
const CELL_X: f32 = ${w(n.displayCellSize[0])};
const CELL_Y: f32 = ${w(n.displayCellSize[1])};`,body:`let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let code = directions[directionsOffset + index];
  var columnStep = 0.0;
  var rowStep = 0.0;
  if (code == 1u) { columnStep = 1.0; }
  else if (code == 2u) { columnStep = 1.0; rowStep = 1.0; }
  else if (code == 4u) { rowStep = 1.0; }
  else if (code == 8u) { columnStep = -1.0; rowStep = 1.0; }
  else if (code == 16u) { columnStep = -1.0; }
  else if (code == 32u) { columnStep = -1.0; rowStep = -1.0; }
  else if (code == 64u) { rowStep = -1.0; }
  else if (code == 128u) { columnStep = 1.0; rowStep = -1.0; }
  let startX = MINIMUM_X + (f32(column) + 0.5) * CELL_X;
  let startY = MAXIMUM_Y - (f32(row) + 0.5) * CELL_Y;
  let base = segmentsOffset + 4u * index;
  segments[base] = startX;
  segments[base + 1u] = startY;
  segments[base + 2u] = startX + columnStep * CELL_X;
  segments[base + 3u] = startY - rowStep * CELL_Y;
  let order = orders[ordersOffset + index];
  let isStream = order > 0u && order != 0xffffffffu;
  width0[width0Offset + index] = select(0.0, 1.0, isStream && order == 1u);
  width1[width1Offset + index] = select(0.0, 1.0, isStream && order == 2u);
  width2[width2Offset + index] = select(0.0, 1.0, isStream && order == 3u);
  width3[width3Offset + index] = select(0.0, 1.0, isStream && order >= 4u);`})}function E(e,t){let{grid:n}=t;s(e,{id:t.id,bindings:[{name:`ids`,view:t.ids,type:`u32`,access:`read`},{name:`count`,view:t.count,type:`u32`,access:`read`},{name:`segments`,view:t.segments,type:`f32`,access:`read_write`}],invocationCount:t.capacity,declarations:`
const GRID_WIDTH: u32 = ${n.width}u;
const ORIGIN_X: f32 = ${w(n.bounds[0])};
const ORIGIN_Y: f32 = ${w(n.bounds[3])};
const CELL_WIDTH: f32 = ${w(n.displayCellSize[0])};
const CELL_HEIGHT: f32 = ${w(n.displayCellSize[1])};
${x}
fn getCellCenter(cell: u32) -> vec2<f32> {
  return vec2<f32>(
    ORIGIN_X + (f32(cell % GRID_WIDTH) + 0.5) * CELL_WIDTH,
    ORIGIN_Y - (f32(cell / GRID_WIDTH) + 0.5) * CELL_HEIGHT
  );
}`,body:`let pathCount = count[countOffset];
  let base = segmentsOffset + index * 4u;
  if (index < pathCount) {
    let start = getCellCenter(ids[idsOffset + index]);
    let end = getCellCenter(ids[idsOffset + min(index + 1u, pathCount - 1u)]);
    segments[base] = start.x;
    segments[base + 1u] = start.y;
    segments[base + 2u] = end.x;
    segments[base + 3u] = end.y;
  } else {
    let nan = getQuietNaN();
    segments[base] = nan;
    segments[base + 1u] = nan;
    segments[base + 2u] = nan;
    segments[base + 3u] = nan;
  }`})}export{E as a,d as c,y as d,v as f,m as g,p as h,S as i,h as l,f as m,x as n,T as o,_ as p,C as r,l as s,b as t,g as u};