import"./layer-DFQ5HqjV.js";import"./dist-CCh72y72.js";import{mt as e}from"./src-CNCKFW2b.js";import{i as t,n,t as r}from"./earth-xB16XgAa.js";import{E as i,T as a,c as o,u as s,v as c,y as l}from"./analysis-frame-Blu0tYdw.js";import{l as u}from"./permutation-inference-kernels-D-xhEewi.js";import{c as d,i as f}from"./cell-topology-wgsl-BT8_s49b.js";import{M as p,N as m,_ as h,h as g,i as _,j as v,k as y,t as b,w as x,y as S}from"./resources-CwvhEOBe.js";import{t as C}from"./gpu-reduction-BD8l40jM.js";import{t as w}from"./gpu-histogram-C4GJppav.js";import{l as T,n as E,s as D}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as ee,l as O}from"./wgsl-common-BNRVzgd8.js";import{o as k,s as A}from"./spatial-weights-D4bd9AQ6.js";import{t as te}from"./gpu-cell-aggregation-BFqjxJvj.js";import{n as ne,t as re}from"./gpu-cell-geometry-PcTj7e7c.js";import{n as j,t as ie}from"./gpu-point-to-cell-DabDU1o1.js";import{t as M}from"./h3-measures-wgsl-C0gyVg8N.js";import{n as N}from"./gpu-cell-topology-6n6YnUDd.js";import{t as P}from"./gpu-class-breaks-DsbEaky8.js";import{r as F,t as I}from"./gpu-multiple-testing-adjustment-C7k5NYt0.js";import{t as L}from"./gpu-lattice-weights-DjJ2AdyH.js";import{a as R,c as z,f as B,l as V,n as ae,o as H,r as U,s as W,t as G}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";import{t as K}from"./gpu-local-moran-CvNMe0Gu.js";import{a as oe,c as se,d as ce,f as le,i as ue,l as de,m as fe,o as q,p as pe,r as me,s as he,u as J}from"./gpu-pairwise-point-in-polygon-DeHbidtn.js";import{n as ge}from"./gpu-point-in-polygon-join-B9XcIwiW.js";import{n as _e,t as ve}from"./gpu-neighbor-search-Fct8W8hB.js";import{t as ye}from"./gpu-local-permutation-test-3lrj6Vn_.js";import{t as be}from"./gpu-color-scale-BxTDjZQ3.js";import{t as xe}from"./gpu-lon-lat-polygon-preparation-CInGqYKK.js";import{a as Se,n as Ce,o as we,r as Te,s as Y,t as Ee}from"./recipe-utils-CqprOQld.js";import{a as De,b as Oe,c as ke,o as Ae}from"./index-Bjk18V2I.js";import{t as X}from"./graph-buffers-B4_80RlX.js";import{t as je}from"./graph-swap-41Ru7LVu.js";import{t as Me}from"./summary-reader-CsgWxENf.js";import{t as Ne}from"./mode-kernels-Dxm2GWvI.js";import"./anchors-D9E9ucGL.js";import{c as Pe,g as Fe}from"./gazetteer-OhisynFF.js";import{C as Ie,a as Le,b as Re,l as ze,m as Be,v as Ve,w as He,x as Ue}from"./class-table-BVEo3-Vs.js";import{a as We,g as Ge}from"./hue-registry-4FjIJ9xz.js";import{a as Ke,i as qe,n as Z,r as Je,s as Ye}from"./live-text-83XBWyr1.js";import{i as Xe}from"./picking-DRmnefrA.js";import{r as Ze,t as Qe}from"./segments-BtINkq-i.js";import"./b6-polygons-Bf2djBQx.js";import{a as $e,c as et,d as tt,f as nt,i as rt,l as it,n as at,o as ot,r as st,s as ct,t as lt,u as ut}from"./hot-spots.style-DcEoMUmX.js";import{n as dt,t as ft}from"./b4-format-Cgp0VQWI.js";import{n as pt,t as mt}from"./b4-weights-core-mAiwYc53.js";import{n as ht,r as gt}from"./b4-geography-Dd13GyrB.js";import{t as _t}from"./b4-layers-BaIKQJXr.js";var vt=`GPUCellMeasures`,yt=class{id;props;maximumEdgeCount;units;measures;constructor(e){this.id=e.id??`cell-measures`,this.props=e;let t=this.id;if(e.family!==`h3`)throw Error(`${t} family must be 'h3'`);if(e.wordOrder&&e.wordOrder!==`little-endian`&&e.wordOrder!==`high-low`)throw Error(`${t} wordOrder must be 'little-endian' or 'high-low'`);if(this.units=e.units??`meters`,this.units!==`meters`&&this.units!==`kilometers`)throw Error(`${t} units must be 'meters' or 'kilometers'`);if(this.measures={lengthModel:`great-circle`,areaModel:`sphere-polygon`,lengthUnits:this.units,areaUnits:c(this.units)},this.maximumEdgeCount=e.maximumEdgeCount??6,!Number.isInteger(this.maximumEdgeCount)||this.maximumEdgeCount<6||this.maximumEdgeCount>10)throw Error(`${t} maximumEdgeCount must be an integer in [6, 10]`);let{output:n}=e,r=[[`output.areas`,n.areas],[`output.perimeters`,n.perimeters],[`output.edgeLengths`,n.edgeLengths],[`output.edgeCounts`,n.edgeCounts],[`output.pentagons`,n.pentagons]];S(e.cells,[`uint32x2`],`${t} cells`);let i=e.cells.length;if(i<1)throw Error(`${t} needs at least one row`);if(!r.some(([,e])=>e))throw Error(`${t} needs at least one output`);for(let[e,r]of[[`output.areas`,n.areas],[`output.perimeters`,n.perimeters]])if(r&&(S(r,[`float32`],`${t} ${e}`),r.length!==i))throw Error(`${t} ${e} length must equal the row count`);if(n.edgeLengths&&(S(n.edgeLengths,[`float32`],`${t} output.edgeLengths`),n.edgeLengths.length!==i*this.maximumEdgeCount))throw Error(`${t} output.edgeLengths length must equal rows * maximumEdgeCount (${i*this.maximumEdgeCount})`);for(let[e,r]of[[`output.edgeCounts`,n.edgeCounts],[`output.pentagons`,n.pentagons]])if(r&&(S(r,[`uint32`],`${t} ${e}`),r.length!==i))throw Error(`${t} ${e} length must equal the row count`);g(t,r.map(([,e])=>e),[e.cells])}getCommandNodes(e){let{id:t,props:n,maximumEdgeCount:r}=this,{output:i}=n;h(t,e,[n.cells,...Object.values(i)]);let a=[{name:`cells`,view:n.cells,type:`u32`,access:`read`}],o=(e,t,n)=>{t&&a.push({name:e,view:t,type:n,access:`read_write`})};o(`areasOut`,i.areas,`f32`),o(`perimetersOut`,i.perimeters,`f32`),o(`edgeLengthsOut`,i.edgeLengths,`f32`),o(`edgeCountsOut`,i.edgeCounts,`u32`),o(`pentagonsOut`,i.pentagons,`u32`);let s=!!(i.areas||i.perimeters||i.edgeLengths||i.edgeCounts),c=`let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${n.wordOrder===`high-low`?`words`:`words.yx`};
  ${i.pentagons?`pentagonsOut[pentagonsOutOffset + index] = select(0u, 1u, cellMeasuresIsPentagon(key));`:``}
  ${s?`let measures = cellMeasuresGetH3(key);
  ${i.areas?`areasOut[areasOutOffset + index] = measures.area * EARTH_RADIUS * EARTH_RADIUS;`:``}
  ${i.perimeters?`perimetersOut[perimetersOutOffset + index] = measures.perimeter * EARTH_RADIUS;`:``}
  ${i.edgeCounts?`edgeCountsOut[edgeCountsOutOffset + index] = measures.count;`:``}
  ${i.edgeLengths?`for (var edge = 0u; edge < MAXIMUM_EDGE_COUNT; edge++) {
    edgeLengthsOut[edgeLengthsOutOffset + index * MAXIMUM_EDGE_COUNT + edge] =
      select(0.0, measures.edges[edge] * EARTH_RADIUS, edge < measures.count);
  }`:``}`:``}`,u=l({coordinateSpace:`longitude-latitude`,metric:`great-circle`,units:this.units});return[D(e,{id:`${t}-measures`,operation:vt,variant:`h3`,bindings:a,invocationCount:n.cells.length,declarations:`const MAXIMUM_EDGE_COUNT: u32 = ${r}u;
const EARTH_RADIUS: f32 = ${T(u)};
${d.source}
${f}
${j}
${M}`,body:c})]}};function bt(e){if(e)return Object.freeze({coordinateSpace:`planar`,metric:`native`,frame:Object.freeze({kind:`crs`,id:e,crs:e})})}function xt(e){let t=new Uint32Array(e.length*8),n=new Float32Array(t.buffer);return e.forEach((e,r)=>{let i=r*8;n.set(e.bounds,i),t[i+4]=e.escape,t[i+5]=e.firstPatch,t[i+6]=e.patchEnd}),t}var St=10,Ct=.01,wt=8,Tt=7,Et=4096,Dt=32,Ot=16,kt=11,At=1e-13;function jt(e){let{projection:t,bounds:n,tolerance:r=Ct,precision:i=`local-f32`,maxDepth:a=wt,degree:o=3,sampleCount:s=Tt,maxPatches:c=Et}=e;Lt({projection:t,bounds:n,tolerance:r,precision:i,maxDepth:a,degree:o,sampleCount:s,maxPatches:c});let l=Rt(t),u=l(cn(n)),d=[];return zt({project:l,destinationOrigin:u,precision:i,degree:o,tolerance:r,maxDepth:a,maxPatches:c,sampleCount:s,patches:d},n,0),{precision:i,bounds:[...n],destinationOrigin:u,patches:d,degree:o,tolerance:r,maxError:d.reduce((e,t)=>Math.max(e,t.maxError),0),float32MaxError:d.reduce((e,t)=>Math.max(e,t.float32MaxError),0),doubleSingleMaxError:d.reduce((e,t)=>Math.max(e,t.doubleSingleMaxError),0)}}function Mt(e){let t=e.patches.length*64,n=xt(e.routingIndex??[]),r=e.strictDomains?e.patches.length*20:0,i=new Uint32Array(Nt(e));i.set(n,t+12+r);let a=new DataView(i.buffer);for(let n of e.patches){let r=n.id*64;if(!Number.isSafeInteger(n.id)||n.id<0||n.id>=e.patches.length)throw Error(`projection patch IDs must be contiguous zero-based indices`);a.setFloat64((r+0)*4,n.sourceOrigin[0],!0),a.setFloat64((r+2)*4,n.sourceOrigin[1],!0),a.setFloat32((r+4)*4,n.sourceScale[0],!0),a.setFloat32((r+5)*4,n.sourceScale[1],!0),a.setFloat32((r+6)*4,n.destinationOrigin[0]-e.destinationOrigin[0],!0),a.setFloat32((r+7)*4,n.destinationOrigin[1]-e.destinationOrigin[1],!0),a.setFloat32((r+8)*4,(n.bounds[0]-n.sourceOrigin[0])/n.sourceScale[0],!0),a.setFloat32((r+9)*4,(n.bounds[1]-n.sourceOrigin[1])/n.sourceScale[1],!0),a.setFloat32((r+10)*4,n.sourceOrigin[0]-Math.fround(n.sourceOrigin[0]),!0),a.setFloat32((r+11)*4,n.sourceOrigin[1]-Math.fround(n.sourceOrigin[1]),!0),i[r+12]=n.degree,a.setFloat32((r+13)*4,n.sourceOrigin[0],!0),a.setFloat32((r+14)*4,n.sourceOrigin[1],!0),i[r+15]=0,a.setFloat64((r+16)*4,n.destinationOrigin[0],!0),a.setFloat64((r+18)*4,n.destinationOrigin[1],!0);for(let e=0;e<St;e++){let t=n.doubleSingleCoefficientsX[e]??0,i=n.doubleSingleCoefficientsY[e]??0,o=Math.fround(t),s=Math.fround(i);a.setFloat32((r+20+e)*4,o,!0),a.setFloat32((r+30+e)*4,s,!0),a.setFloat32((r+40+e)*4,t-o,!0),a.setFloat32((r+50+e)*4,i-s,!0)}if(a.setFloat32((r+60)*4,n.sourceScale[0]-Math.fround(n.sourceScale[0]),!0),a.setFloat32((r+61)*4,n.sourceScale[1]-Math.fround(n.sourceScale[1]),!0),e.strictDomains){let e=t+12+n.id*20;n.bounds.forEach((t,n)=>{let r=Math.fround(t),i=Math.fround(t-r);(n<2&&r+i<t||n>=2&&r+i>t)&&(i=It(i,n<2?1:-1)),a.setFloat64((e+n*2)*4,t,!0),a.setFloat32((e+8+n*2)*4,r,!0),a.setFloat32((e+9+n*2)*4,i,!0),a.setFloat32((e+16+n)*4,n<2?Pt(t):Ft(t),!0)})}}for(let n=0;n<e.bounds.length;n++)a.setFloat64((t+n*2)*Uint32Array.BYTES_PER_ELEMENT,e.bounds[n],!0);let o=[Pt(e.bounds[0]),Pt(e.bounds[1]),Ft(e.bounds[2]),Ft(e.bounds[3])];for(let e=0;e<o.length;e++)a.setFloat32((t+8+e)*Uint32Array.BYTES_PER_ELEMENT,o[e],!0);return i}function Nt(e){return e.patches.length*(64+(e.strictDomains?20:0))+12+(e.routingIndex?.length??0)*8}function Pt(e){let t=Math.fround(e);return t>=e?t:It(t,1)}function Ft(e){let t=Math.fround(e);return t<=e?t:It(t,-1)}function It(e,t){let n=new Float32Array(1),r=new Uint32Array(n.buffer);return n[0]=e,e===0?r[0]=t===1?1:2147483649:e>0==(t===1)?r[0]++:r[0]--,n[0]}function Lt(e){let{projection:t,bounds:n,tolerance:r,precision:i,maxDepth:a,degree:o,sampleCount:s,maxPatches:c}=e;if(!(typeof t==`function`||typeof t==`object`&&t&&typeof t.project==`function`))throw Error(`projection must be a function or expose a project method`);if(!Array.isArray(n)||n.length!==4||!n.every(Number.isFinite)||n[0]>=n[2]||n[1]>=n[3])throw Error(`projection bounds must contain finite, increasing source coordinates`);if(!Number.isFinite(r)||r<=0)throw Error(`projection tolerance must be a positive finite number`);if(i!==`local-f32`&&i!==`double-single`)throw Error(`projection precision must be local-f32 or double-single`);if(!Number.isSafeInteger(a)||a<0||a>Ot)throw Error(`projection maxDepth must be an integer between 0 and 16`);if(o!==1&&o!==2&&o!==3)throw Error(`projection polynomial degree must be 1, 2, or 3`);if(!Number.isSafeInteger(s)||s<o+1||s>Dt)throw Error(`projection sampleCount must be between degree + 1 and 32`);if(!Number.isSafeInteger(c)||c<1)throw Error(`projection maxPatches must be a positive safe integer`)}function Rt(e){return t=>{let n=[t[0],t[1]],r=typeof e==`function`?e(n):e.project(n);if(!Array.isArray(r)||r.length<2||!Number.isFinite(r[0])||!Number.isFinite(r[1]))throw Error(`projection provider must return two finite destination coordinates`);return[r[0],r[1]]}}function zt(e,t,n){let r=cn(t),i=[(t[2]-t[0])/2,(t[3]-t[1])/2];if(i.some(e=>e<=0||!Number.isFinite(Math.fround(e))||Math.fround(e)===0))throw Error(`projection source bounds must have finite, nonzero float32 local scales`);let a=e.project(r),o=Bt(e,t,r,i,a);if(o.maxError<=e.tolerance){if(e.patches.length>=e.maxPatches)throw Error(`projection approximation exceeds the configured maximum patch count`);e.patches.push({id:e.patches.length,bounds:[...t],sourceOrigin:r,sourceScale:i,destinationOrigin:a,coefficientsX:o.coefficientsX,coefficientsY:o.coefficientsY,doubleSingleCoefficientsX:o.doubleSingleCoefficientsX,doubleSingleCoefficientsY:o.doubleSingleCoefficientsY,degree:e.degree,maxError:o.maxError,float32MaxError:o.float32MaxError,doubleSingleMaxError:o.doubleSingleMaxError});return}if(n>=e.maxDepth)throw Error(`projection approximation exceeds its tolerance at the maximum subdivision depth`);let[s,c]=r;if(s<=t[0]||s>=t[2]||c<=t[1]||c>=t[3])throw Error(`projection source bounds cannot be subdivided at binary64 precision`);zt(e,[t[0],t[1],s,c],n+1),zt(e,[s,t[1],t[2],c],n+1),zt(e,[t[0],c,s,t[3]],n+1),zt(e,[s,c,t[2],t[3]],n+1)}function Bt(e,t,n,r,i){let a=ln(e.degree),o=new Float64Array(a*a),s=new Float64Array(a),c=new Float64Array(a),l=new Float64Array(a);for(let t=0;t<e.sampleCount;t++){let u=un(t,e.sampleCount);for(let t=0;t<e.sampleCount;t++){let d=un(t,e.sampleCount),f=[n[0]+d*r[0],n[1]+u*r[1]];Yt(l,e.precision===`double-single`?qt(f,n,r):Kt(f,n,r),e.degree);let p=e.project(f),m=p[0]-i[0],h=p[1]-i[1];for(let e=0;e<a;e++){let t=l[e];s[e]+=t*m,c[e]+=t*h;for(let n=0;n<a;n++)o[e*a+n]+=t*l[n]}}}let u=Ut(o,s,a),d=Ut(o,c,a),f=Float32Array.from(u),p=Float32Array.from(d),m=0,h=0;for(let a=0;a<kt;a++){let o=-1+2*a/(kt-1);for(let s=0;s<kt;s++){let c=-1+2*s/(kt-1),l=[s===0?t[0]:s===kt-1?t[2]:n[0]+c*r[0],a===0?t[1]:a===kt-1?t[3]:n[1]+o*r[1]],g=e.project(l),_=Kt(l,n,r);m=Math.max(m,Vt(e,i,f,p,_,g)),h=Math.max(h,Ht(i,u,d,l,n,r,e.degree,g,`binary64`));let v=[Math.fround(l[0]),Math.fround(l[1])];if(v.every(Number.isFinite)&&sn(t,v)){let t=v[0]===l[0]&&v[1]===l[1]?g:e.project(v),a=Jt(v,n,r);m=Math.max(m,Vt(e,i,f,p,a,t)),h=Math.max(h,Ht(i,u,d,v,n,r,e.degree,t,`float32`))}}}return{coefficientsX:f,coefficientsY:p,doubleSingleCoefficientsX:u,doubleSingleCoefficientsY:d,maxError:e.precision===`double-single`?h:m,float32MaxError:m,doubleSingleMaxError:h}}function Vt(e,t,n,r,i,a){let o=Xt(n,i,e.degree),s=Xt(r,i,e.degree),c=Math.fround(Math.fround(t[0]-e.destinationOrigin[0])+o),l=Math.fround(Math.fround(t[1]-e.destinationOrigin[1])+s);return Math.hypot(c-(a[0]-e.destinationOrigin[0]),l-(a[1]-e.destinationOrigin[1]))}function Ht(e,t,n,r,i,a,o,s,c){let l=c===`float32`?Gt(r,i,a):Wt(r,i,a),u=Zt(t,l,o),d=Zt(n,l,o),f=tn(Q(e[0]),u),p=tn(Q(e[1]),d);return Math.hypot(f[0]+f[1]-s[0],p[0]+p[1]-s[1])}function Ut(e,t,n){let r=e.slice(),i=t.slice();for(let e=0;e<n;e++){let t=e,a=Math.abs(r[e*n+e]);for(let i=e+1;i<n;i++){let o=Math.abs(r[i*n+e]);o>a&&(a=o,t=i)}if(!Number.isFinite(a)||a<=At)throw Error(`projection samples cannot support a stable polynomial approximation`);if(t!==e){for(let i=e;i<n;i++){let a=e*n+i,o=t*n+i;[r[a],r[o]]=[r[o],r[a]]}[i[e],i[t]]=[i[t],i[e]]}let o=r[e*n+e];for(let t=e+1;t<n;t++){let a=r[t*n+e]/o;r[t*n+e]=0;for(let i=e+1;i<n;i++)r[t*n+i]-=a*r[e*n+i];i[t]-=a*i[e]}}for(let e=n-1;e>=0;e--){let t=i[e];for(let a=e+1;a<n;a++)t-=r[e*n+a]*i[a];i[e]=t/r[e*n+e]}return i}function Wt(e,t,n){let r=Q(e[0]-t[0]),i=Q(e[1]-t[1]);return[an(r,Q(n[0])),an(i,Q(n[1]))]}function Gt(e,t,n){let r=nn([Math.fround(e[0]),0],Q(t[0])),i=nn([Math.fround(e[1]),0],Q(t[1]));return[an(r,Q(n[0])),an(i,Q(n[1]))]}function Kt(e,t,n){return[Math.fround(Math.fround(e[0]-t[0])/Math.fround(n[0])),Math.fround(Math.fround(e[1]-t[1])/Math.fround(n[1]))]}function qt(e,t,n){return[(e[0]-t[0])/n[0],(e[1]-t[1])/n[1]]}function Jt(e,t,n){let r=Math.fround(t[0]),i=Math.fround(t[1]),a=Math.fround(t[0]-r),o=Math.fround(t[1]-i),s=Math.fround(Math.fround(e[0]-r)-a),c=Math.fround(Math.fround(e[1]-i)-o);return[Math.fround(s/Math.fround(n[0])),Math.fround(c/Math.fround(n[1]))]}function Yt(e,t,n){let[r,i]=t,a=0;for(let t=0;t<=n;t++)for(let n=0;n<=t;n++){let o=t-n;e[a++]=r**o*i**n}}function Xt(e,t,n){let[r,i]=t,a=ln(n),o=t=>t<a?e[t]:0,s=on(r,on(r,o(6),o(3)),o(1)),c=on(r,o(7),o(4)),l=on(r,o(8),o(5));return on(i,on(i,on(i,o(9),l),on(r,c,o(2))),on(r,s,o(0)))}function Zt(e,t,n){let[r,i]=t,a=ln(n),o=t=>Q(t<a?e[t]:0),s=(e,t,n)=>tn(n,rn(e,t)),c=s(r,s(r,o(6),o(3)),o(1)),l=s(r,o(7),o(4)),u=s(r,o(8),o(5));return s(i,s(i,s(i,o(9),u),s(r,l,o(2))),s(r,c,o(0)))}function Q(e){let t=Math.fround(e);return[t,Math.fround(e-t)]}function Qt(e,t){let n=Math.fround(e+t);return[n,Math.fround(t-Math.fround(n-e))]}function $t(e,t){let n=Math.fround(e+t),r=Math.fround(n-e),i=Math.fround(e-Math.fround(n-r)),a=Math.fround(t-r);return[n,Math.fround(i+a)]}function en(e,t){let n=Math.fround(e*t);return[n,Math.fround(e*t-n)]}function tn(e,t){let n=$t(e[0],t[0]),r=$t(e[1],t[1]);return n=[n[0],Math.fround(n[1]+r[0])],n=Qt(n[0],n[1]),n=[n[0],Math.fround(n[1]+r[1])],Qt(n[0],n[1])}function nn(e,t){return tn(e,[-t[0],-t[1]])}function rn(e,t){let n=en(e[0],t[0]);return n=[n[0],Math.fround(n[1]+Math.fround(e[0]*t[1]))],n=Qt(n[0],n[1]),n=[n[0],Math.fround(n[1]+Math.fround(e[1]*t[0]))],Qt(n[0],n[1])}function an(e,t){let n=Math.fround(1/t[0]),r=rn(e,[n,0]),i=nn(e,rn(t,r))[0];return tn(r,en(n,i))}function on(e,t,n){return Math.fround(Math.fround(e*t)+n)}function sn(e,t){return t[0]>=e[0]&&t[0]<=e[2]&&t[1]>=e[1]&&t[1]<=e[3]}function cn(e){return[e[0]+(e[2]-e[0])/2,e[1]+(e[3]-e[1])/2]}function ln(e){return(e+1)*(e+2)/2}function un(e,t){let n=Math.cos((2*e+1)*Math.PI/(2*t));return Math.abs(n)<2**-52?0:n}function dn(e){let{doubleSingle:t,patchIdOffset:n,validityOffset:r}=e;return`
${e.inputDeclaration}
${fn(e)}
@group(0) @binding(auto) var<storage, read_write> outputPositions: array<${t?`vec4f`:`vec2f`}>;
${n===void 0?``:`@group(0) @binding(auto) var<storage, read> projectionPatchIds: array<u32>;`}
${r===void 0?``:`@group(0) @binding(auto) var<storage, read_write> projectionValidity: array<u32>;`}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_id) localId: vec3u) {
  ${e.invocationIndexSource}
  if (index >= ${e.elementCount}u) { return; }
  ${n===void 0?``:`if (projectionPatchIds[${n}u + index] >= PATCH_COUNT) {
    outputPositions[${e.outputOffset}u + index] = ${t?`vec4f(0.0)`:`vec2f(0.0)`};
    ${r===void 0?``:`projectionValidity[${r}u + index] = 0u;`}
    return;
  }`}
  let result = project(${e.readPosition}, ${n===void 0?`INVALID_PATCH`:`projectionPatchIds[${n}u + index]`});
  outputPositions[${e.outputOffset}u + index] = result.position;
  ${r===void 0?``:`projectionValidity[${r}u + index] = result.valid;`}
}`}function fn(e){let{precise:t,doubleSingle:n,patchCount:r,planOffset:i,doubleSingleInput:a,inputBoundsOffset:o}=e,s=t?`RawPoint`:a?`vec4f`:`vec2f`,c=e.routingNodeCount??0,l=i+r*64+12,u=l+(e.strictDomains?r*20:0),d=t?`vec2f(sub_fp64u32_to_f32(position.x, vec2u(0u)), sub_fp64u32_to_f32(position.y, vec2u(0u)))`:a?`vec2f(position.x + position.y, position.z + position.w)`:`position`,f=e.strictDomains?`
  let domainOffset = ${l}u + patchIndex * 20u;
  for (var boundIndex = 0u; boundIndex < 4u; boundIndex += 1u) {
    ${t?`let offset = domainOffset + boundIndex * 2u;
    let bound = vec2u(projectionPlans[offset + 1u], projectionPlans[offset]);
    let coordinate = select(position.x, position.y, (boundIndex & 1u) != 0u);
    let comparison = compareFiniteBinary64(coordinate, bound);`:a?`let offset = domainOffset + 8u + boundIndex * 2u;
    let bound = vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
    let coordinate = select(position.xy, position.zw, (boundIndex & 1u) != 0u);
    let comparison = compare_fp64(coordinate, bound);`:`let bound = bitcast<f32>(projectionPlans[domainOffset + 16u + boundIndex]);
    let coordinate = position[boundIndex & 1u];
    let comparison = select(select(0, 1, coordinate > bound), -1, coordinate < bound);`}
    if ((boundIndex < 2u && comparison < 0) || (boundIndex >= 2u && comparison > 0)) { return false; }
  }`:``,p=a?`let offset = projectionSourceOffsetFP64(position, patchIndex);
  return vec2f(offset.x.x + offset.x.y, offset.y.x + offset.y.y);`:t?`let originX = vec2u(projectionPlanWord(patchIndex, 1u), projectionPlanWord(patchIndex, 0u));
  let originY = vec2u(projectionPlanWord(patchIndex, 3u), projectionPlanWord(patchIndex, 2u));
  return vec2f(
    sub_fp64u32_to_f32(position.x, originX),
    sub_fp64u32_to_f32(position.y, originY)
  );`:`let sourceOriginHigh = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 14u))
  );
  let sourceOriginLow = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 10u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 11u))
  );
  // Route the rounded high-limb subtraction through an opaque runtime-zero integer operation.
  // Metal otherwise reassociates these subtractions and silently drops the low origin limb.
  let highOffsetBits = bitcast<vec2u>(position - sourceOriginHigh) ^
    vec2u(projectionPlanWord(patchIndex, 15u));
  let highOffset = bitcast<vec2f>(highOffsetBits);
  return highOffset - sourceOriginLow;`,m=a?`is_finite_fp64(position.xy) && is_finite_fp64(position.zw)`:t?`rawPointIsFinite(position)`:`all((bitcast<vec2u>(position) & vec2u(0x7f800000u)) != vec2u(0x7f800000u))`,h=a?`return compare_fp64(position.xy, projectionInputBound(0u)) >= 0 &&
    compare_fp64(position.zw, projectionInputBound(1u)) >= 0 &&
    compare_fp64(position.xy, projectionInputBound(2u)) <= 0 &&
    compare_fp64(position.zw, projectionInputBound(3u)) <= 0;`:t?`let minimumX = projectionPlanRawBound(0u);
  let minimumY = projectionPlanRawBound(1u);
  let maximumX = projectionPlanRawBound(2u);
  let maximumY = projectionPlanRawBound(3u);
  return compareFiniteBinary64(position.x, minimumX) >= 0 &&
    compareFiniteBinary64(position.y, minimumY) >= 0 &&
    compareFiniteBinary64(position.x, maximumX) <= 0 &&
    compareFiniteBinary64(position.y, maximumY) <= 0;`:`let minimum = vec2f(
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 8u]),
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 9u])
  );
  let maximum = vec2f(
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 10u]),
    bitcast<f32>(projectionPlans[PLAN_BOUNDS_OFFSET + 11u])
  );
  return all(position >= minimum) && all(position <= maximum);`,g=t?`fn projectionPlanRawBound(boundIndex: u32) -> vec2u {
  let wordOffset = PLAN_BOUNDS_OFFSET + boundIndex * 2u;
  return vec2u(
    projectionPlans[wordOffset + 1u],
    projectionPlans[wordOffset]
  );
}

fn compareFiniteBinary64(firstBits: vec2u, secondBits: vec2u) -> i32 {
  let first = fp64_decode_bits(firstBits);
  let second = fp64_decode_bits(secondBits);
  if (first.isZero && second.isZero) {
    return 0;
  }
  if (first.sign != second.sign) {
    return select(1, -1, first.sign == 1u);
  }
  let magnitudeComparison = fp64_finite_magnitude_compare(first, second);
  return select(magnitudeComparison, -magnitudeComparison, first.sign == 1u);
}`:``,_=n?`struct ProjectionPointFP64 { x: vec2f, y: vec2f }

fn projectionSourceOffsetFP64(
  position: ${s},
  patchIndex: u32
) -> ProjectionPointFP64 {
  ${a?`return ProjectionPointFP64(
    sub_fp64(position.xy, vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 10u))
    )),
    sub_fp64(position.zw, vec2f(
      bitcast<f32>(projectionPlanWord(patchIndex, 14u)),
      bitcast<f32>(projectionPlanWord(patchIndex, 11u))
    ))
  );`:t?`let originX = vec2u(projectionPlanWord(patchIndex, 1u), projectionPlanWord(patchIndex, 0u));
  let originY = vec2u(projectionPlanWord(patchIndex, 3u), projectionPlanWord(patchIndex, 2u));
  return ProjectionPointFP64(
    sub_fp64u32_to_fp64(position.x, originX),
    sub_fp64u32_to_fp64(position.y, originY)
  );`:`let sourceOriginX = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 13u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 10u))
  );
  let sourceOriginY = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 14u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 11u))
  );
  return ProjectionPointFP64(
    sub_fp64(vec2f(position.x, 0.0), sourceOriginX),
    sub_fp64(vec2f(position.y, 0.0), sourceOriginY)
  );`}
}

fn normalizeProjectionPositionFP64(
  position: ${s},
  patchIndex: u32
) -> ProjectionPointFP64 {
  let sourceOffset = projectionSourceOffsetFP64(position, patchIndex);
  let sourceScaleX = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 4u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 60u))
  );
  let sourceScaleY = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 5u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 61u))
  );
  return ProjectionPointFP64(
    div_fp64(sourceOffset.x, sourceScaleX),
    div_fp64(sourceOffset.y, sourceScaleY)
  );
}

fn projectionCoefficientFP64(
  patchIndex: u32,
  highOffset: u32,
  lowOffset: u32,
  coefficientIndex: u32
) -> vec2f {
  return vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, highOffset + coefficientIndex)),
    bitcast<f32>(projectionPlanWord(patchIndex, lowOffset + coefficientIndex))
  );
}

fn projectionMultiplyAddFP64(multiplier: vec2f, multiplicand: vec2f, addend: vec2f) -> vec2f {
  return sum_fp64(addend, mul_fp64(multiplier, multiplicand));
}

fn evaluateProjectionPolynomialFP64(
  patchIndex: u32,
  highOffset: u32,
  lowOffset: u32,
  normalized: ProjectionPointFP64
) -> vec2f {
  let coefficient0 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 0u);
  let coefficient1 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 1u);
  let coefficient2 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 2u);
  let coefficient3 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 3u);
  let coefficient4 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 4u);
  let coefficient5 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 5u);
  let coefficient6 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 6u);
  let coefficient7 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 7u);
  let coefficient8 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 8u);
  let coefficient9 = projectionCoefficientFP64(patchIndex, highOffset, lowOffset, 9u);

  let xQuadratic = projectionMultiplyAddFP64(normalized.x, coefficient6, coefficient3);
  let xLinear = projectionMultiplyAddFP64(normalized.x, xQuadratic, coefficient1);
  let mixedLinear = projectionMultiplyAddFP64(normalized.x, coefficient7, coefficient4);
  let yQuadraticX = projectionMultiplyAddFP64(normalized.x, coefficient8, coefficient5);
  let yQuadratic = projectionMultiplyAddFP64(normalized.y, coefficient9, yQuadraticX);
  let yLinearX = projectionMultiplyAddFP64(normalized.x, mixedLinear, coefficient2);
  let yLinear = projectionMultiplyAddFP64(normalized.y, yQuadratic, yLinearX);
  let xContribution = projectionMultiplyAddFP64(normalized.x, xLinear, coefficient0);
  return projectionMultiplyAddFP64(normalized.y, yLinear, xContribution);
}

fn projectionDestinationOriginFP64(patchIndex: u32, wordOffset: u32) -> vec2f {
  let value = vec2u(
    projectionPlanWord(patchIndex, wordOffset + 1u),
    projectionPlanWord(patchIndex, wordOffset)
  );
  return sub_fp64u32_to_fp64(value, vec2u(0u));
}`:``,v=n?`let normalizedFP64 = normalizeProjectionPositionFP64(position, patchIndex);
  let projectedX = sum_fp64(
    projectionDestinationOriginFP64(patchIndex, 16u),
    evaluateProjectionPolynomialFP64(patchIndex, 20u, 40u, normalizedFP64)
  );
  let projectedY = sum_fp64(
    projectionDestinationOriginFP64(patchIndex, 18u),
    evaluateProjectionPolynomialFP64(patchIndex, 30u, 50u, normalizedFP64)
  );
  let projected = vec4f(
    projectedX.x,
    projectedX.y,
    projectedY.x,
    projectedY.y
  );`:`let normalized = normalizeProjectionPosition(position, patchIndex);
  let destinationOffset = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 6u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 7u))
  );
  let projected = destinationOffset + vec2f(
    evaluateProjectionPolynomial(patchIndex, 20u, normalized),
    evaluateProjectionPolynomial(patchIndex, 30u, normalized)
  );`;return`
${t?ue:``}
const PATCH_COUNT: u32 = ${r}u;
const PATCH_WORD_LENGTH: u32 = 64u;
const PLAN_OFFSET: u32 = ${i}u;
const PLAN_BOUNDS_OFFSET: u32 = PLAN_OFFSET + PATCH_COUNT * PATCH_WORD_LENGTH;
const INVALID_PATCH: u32 = 0xffffffffu;

@group(0) @binding(auto) var<storage, read> projectionPlans: array<u32>;
struct ProjectionResult { position: ${n?`vec4f`:`vec2f`}, valid: u32 }
${a?`fn projectionInputBound(boundIndex: u32) -> vec2f {
  let offset = ${o}u + boundIndex * 2u;
  return vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
}`:``}

fn projectionPlanWord(patchIndex: u32, wordIndex: u32) -> u32 {
  return projectionPlans[PLAN_OFFSET + patchIndex * PATCH_WORD_LENGTH + wordIndex];
}

${g}

fn projectionPlanContains(position: ${s}) -> bool {
  ${h}
}

fn projectionSourceOffset(position: ${s}, patchIndex: u32) -> vec2f {
  ${p}
}

fn normalizeProjectionPosition(position: ${s}, patchIndex: u32) -> vec2f {
  let sourceScale = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 4u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 5u))
  );
  return projectionSourceOffset(position, patchIndex) / sourceScale;
}

fn projectionPatchContains(position: ${s}, patchIndex: u32) -> bool {
  ${f}
  let normalized = normalizeProjectionPosition(position, patchIndex);
  let minimum = vec2f(
    bitcast<f32>(projectionPlanWord(patchIndex, 8u)),
    bitcast<f32>(projectionPlanWord(patchIndex, 9u))
  );
  // Source scales are half-extents, so the opposite boundary is exactly two normalized units
  // away. Two float32 ULPs keep legitimate binary64 endpoints from falling through a seam.
  let maximum = minimum + vec2f(2.0);
  let boundaryTolerance = vec2f(2.384185791015625e-7);
  return all(normalized >= minimum - boundaryTolerance) &&
    all(normalized <= maximum + boundaryTolerance);
}

fn findProjectionPatch(position: ${s}) -> u32 {
  ${c?`let routingPosition = ${d};
  var nodeIndex = 0u;
  // Bound traversal even for malformed caller-owned storage. A valid traversal visits each
  // node at most once; corrupt escape links must not cause an unbounded GPU loop.
  for (var visitIndex = 0u; visitIndex < ${c}u; visitIndex += 1u) {
    if (nodeIndex >= ${c}u) { break; }
    let offset = ${u}u + nodeIndex * 8u;
    let minimum = vec2f(bitcast<f32>(projectionPlans[offset]), bitcast<f32>(projectionPlans[offset + 1u]));
    let maximum = vec2f(bitcast<f32>(projectionPlans[offset + 2u]), bitcast<f32>(projectionPlans[offset + 3u]));
    if (any(routingPosition < minimum) || any(routingPosition > maximum)) {
      nodeIndex = projectionPlans[offset + 4u];
    } else {
      let firstPatch = projectionPlans[offset + 5u];
      let storedPatchEnd = projectionPlans[offset + 6u];
      // Reject malformed borrowed storage before scanning. Internal nodes have empty ranges;
      // valid leaves never exceed the builder's fixed capacity.
      if (firstPatch >= PATCH_COUNT || storedPatchEnd < firstPatch ||
          storedPatchEnd > PATCH_COUNT || storedPatchEnd - firstPatch > 4u) {
        return INVALID_PATCH;
      }
      let patchEnd = min(storedPatchEnd, PATCH_COUNT);
      for (var patchIndex = firstPatch; patchIndex < patchEnd; patchIndex += 1u) {
        if (projectionPatchContains(position, patchIndex)) { return patchIndex; }
      }
      nodeIndex += 1u;
    }
  }`:`
  for (var patchIndex: u32 = 0u; patchIndex < PATCH_COUNT; patchIndex += 1u) {
    if (projectionPatchContains(position, patchIndex)) {
      return patchIndex;
    }
  }
  `}
  return INVALID_PATCH;
}

fn evaluateProjectionPolynomial(
  patchIndex: u32,
  coefficientOffset: u32,
  normalized: vec2f
) -> f32 {
  // Lower-degree records are zero-padded, so one branch-free Horner expression handles every
  // supported degree without dynamically indexed temporary arrays or nested shader loops.
  let coefficient0 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 0u));
  let coefficient1 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 1u));
  let coefficient2 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 2u));
  let coefficient3 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 3u));
  let coefficient4 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 4u));
  let coefficient5 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 5u));
  let coefficient6 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 6u));
  let coefficient7 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 7u));
  let coefficient8 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 8u));
  let coefficient9 = bitcast<f32>(projectionPlanWord(patchIndex, coefficientOffset + 9u));

  let xQuadratic = coefficient3 + normalized.x * coefficient6;
  let xLinear = coefficient1 + normalized.x * xQuadratic;
  let mixedLinear = coefficient4 + normalized.x * coefficient7;
  let yQuadraticX = coefficient5 + normalized.x * coefficient8;
  let yQuadratic = yQuadraticX + normalized.y * coefficient9;
  let yLinearX = coefficient2 + normalized.x * mixedLinear;
  let yLinear = yLinearX + normalized.y * yQuadratic;
  let xContribution = coefficient0 + normalized.x * xLinear;
  return xContribution + normalized.y * yLinear;
}

${_}

fn project(position: ${s}, assignedPatch: u32) -> ProjectionResult {
  let invalid = ProjectionResult(${n?`vec4f(0.0)`:`vec2f(0.0)`}, 0u);
  if (!(${m}) || !projectionPlanContains(position)) { return invalid; }
  var patchIndex = assignedPatch;
  if (patchIndex == INVALID_PATCH) { patchIndex = findProjectionPatch(position); }
  if (patchIndex >= PATCH_COUNT || !projectionPatchContains(position, patchIndex)) {
    return invalid;
  }
  ${v}
  if (!all((bitcast<${n?`vec4u`:`vec2u`}>(projected) &
    ${n?`vec4u`:`vec2u`}(0x7f800000u)) != ${n?`vec4u`:`vec2u`}(0x7f800000u))) {
    return invalid;
  }
  return ProjectionResult(projected, 1u);
}`}var pn=class{id;positions;precision;spatialContext;output;patchIds;validity;planBuffer;projectionPlan;ownedPlanBuffer;hasRegisteredGraph=!1;destroyed=!1;constructor(e){if(this.id=e.id??`gpu-projection`,this.positions=e.positions,this.precision=e.precision??`local-f32`,this.spatialContext=bt(e.targetCrs),this.output=e.output,this.projectionPlan=e.plan,this.patchIds=e.patchIds,this.validity=e.validity,this.planBuffer=e.planBuffer,mn(e.plan,this.id),e.plan.precision!==this.precision)throw Error(`${this.id} precision must match its compiled projection plan`);fe(this.positions,me,`${this.id} positions`),fe(this.output,[this.precision===`double-single`?`float32x4`:`float32x2`],`${this.id} output`),pe(this.positions,this.output,`${this.id} positions and output`);let t=[[`positions`,this.positions]];if(this.patchIds&&(fe(this.patchIds,[`uint32`],`${this.id} patch IDs`),pe(this.positions,this.patchIds,`${this.id} positions and patch IDs`),t.push([`patch IDs`,this.patchIds])),this.validity&&(fe(this.validity,[`uint32`],`${this.id} validity`),pe(this.positions,this.validity,`${this.id} positions and validity`)),this.planBuffer){if(p(this.planBuffer,[`uint32`],`${this.id} plan buffer`),this.planBuffer.length<Nt(e.plan))throw Error(`${this.id} plan buffer is smaller than its packed projection plan`);t.push([`plan buffer`,this.planBuffer])}le(this.id,t,[[`output`,this.output],...this.validity?[[`validity`,this.validity]]:[]])}get plan(){return this.projectionPlan}updatePlan(t){if(this.assertAvailable(),mn(t,this.id),t.precision!==this.precision)throw Error(`${this.id} updated projection plan must retain its precision`);if(t.patches.length!==this.projectionPlan.patches.length)throw Error(`${this.id} updated projection plan must retain the same patch count`);if(!!t.strictDomains!=!!this.projectionPlan.strictDomains||(t.routingIndex?.length??0)!==(this.projectionPlan.routingIndex?.length??0))throw Error(`updated projection plan must retain its routing and domain layout`);let n=Mt(t);if(this.ownedPlanBuffer)this.ownedPlanBuffer.write(n);else if(this.planBuffer){let t=this.planBuffer.buffer.defaultBuffer;if(!t)throw Error(`${this.id} caller-owned projection plan must be updated by its owner`);if((this.planBuffer.buffer.usage&e.COPY_DST)===0)throw Error(`${this.id} caller-owned projection plan updates require COPY_DST usage`);t.write(n,this.planBuffer.byteOffset)}this.projectionPlan=t}addToGraph(e){if(this.assertAvailable(),this.hasRegisteredGraph)throw Error(`${this.id} projection contributor has already been added to a graph`);let t=[this.positions,this.output];if(this.patchIds&&t.push(this.patchIds),this.validity&&t.push(this.validity),q(e,t,this.id),this.planBuffer&&this.planBuffer.buffer.graph!==e)throw Error(`${this.id} projection plan buffer must belong to the target graph`);let n=this.planBuffer??this.createOwnedPlanView(e),r=J(this.positions),i=J(this.output),a=this.patchIds?J(this.patchIds):void 0,o=this.validity?J(this.validity):void 0;for(let t=0;t<r.length;t++){let s=r[t];s.length!==0&&this.addProjectionPass(e,{chunkIndex:t,input:s,output:i[t],patchIds:a?.[t],validity:o?.[t],plan:n})}this.hasRegisteredGraph=!0}destroy(){this.destroyed||=(this.ownedPlanBuffer?.destroy(),this.ownedPlanBuffer=void 0,!0)}createOwnedPlanView(t){let n=Mt(this.projectionPlan),r=`${this.id}-projection-plan`,i=t.device.createBuffer({id:r,data:n,usage:e.STORAGE|e.COPY_DST});this.ownedPlanBuffer=i;let a=t.importBuffer({id:r,byteLength:n.byteLength,usage:i.usage},i);return t.createDataView(a,{format:`uint32`,length:n.length})}addProjectionPass(e,t){let{chunkIndex:n,input:r,output:i,patchIds:a,validity:o,plan:s}=t,c=de(`positions`,r),l=he(r.length,e.device.limits.maxComputeWorkgroupsPerDimension),u=[{buffer:r,usage:`storage-read`},{buffer:s,usage:`storage-read`},{buffer:i,usage:`storage-write`}],d={positions:r,projectionPlans:s,outputPositions:i};a&&(u.push({buffer:a,usage:`storage-read`}),d.projectionPatchIds=a),o&&(u.push({buffer:o,usage:`storage-write`}),d.projectionValidity=o);let f=dn({precise:c.precise,doubleSingle:this.precision===`double-single`,inputDeclaration:c.declaration,readPosition:c.read(`index`),elementCount:r.length,patchCount:this.projectionPlan.patches.length,routingNodeCount:this.projectionPlan.routingIndex?.length,strictDomains:this.projectionPlan.strictDomains,outputOffset:y(i)/(this.precision===`double-single`?4:2),planOffset:y(s),patchIdOffset:a?y(a):void 0,validityOffset:o?y(o):void 0,invocationIndexSource:se(l)});oe(e,{id:ce(this.positions)?`${this.id}-chunk-${n}`:this.id,source:f,resources:u,bindings:d,dispatchLayout:l,precise:c.precise||this.precision===`double-single`})}assertAvailable(){if(this.destroyed)throw Error(`${this.id} projection contributor has been destroyed`)}};function mn(e,t){if(!e||!Array.isArray(e.patches)||e.patches.length===0)throw Error(`${t} projection plan must contain at least one patch`);if(!Array.isArray(e.bounds)||e.bounds.length!==4||!e.bounds.every(Number.isFinite)||e.bounds[0]>=e.bounds[2]||e.bounds[1]>=e.bounds[3])throw Error(`${t} projection plan must contain finite, increasing source bounds`);if(e.precision!==`local-f32`&&e.precision!==`double-single`||e.patches.some((e,t)=>e.id!==t||!e.sourceOrigin.every(Number.isFinite)||!e.destinationOrigin.every(Number.isFinite)||!e.sourceScale.every(e=>e>0&&Number.isFinite(Math.fround(e))&&Math.fround(e)>0)||!(e.doubleSingleCoefficientsX instanceof Float64Array)||!(e.doubleSingleCoefficientsY instanceof Float64Array)||!e.doubleSingleCoefficientsX.every(Number.isFinite)||!e.doubleSingleCoefficientsY.every(Number.isFinite)))throw Error(`${t} projection plan must contain finite, consecutively numbered patches`);if(!e.destinationOrigin.every(Number.isFinite))throw Error(`${t} projection destination origin must contain finite coordinates`)}var hn=class{id;frame;spatialContext;plan;projection;constructor(e){this.id=e.id??`gpu-local-frame-projection`,this.frame=e.frame,this.spatialContext=e.frame.spatialContext,this.plan={...jt({projection:t=>e.frame.project(t[0],t[1]),bounds:e.bounds,tolerance:e.tolerance??.01}),destinationOrigin:[0,0]},this.projection=new pn({id:this.id,positions:e.positions,output:e.output,validity:e.validity,plan:this.plan})}getCommandNodes(e){return _(e,()=>this.projection.addToGraph(e))}destroy(){this.projection.destroy()}},gn=8,_n=`GPUHotSpotAnalysis`,vn=class{id;props;multipleTesting;islands;minimumNeighbors;constructor(e){this.id=e.id??`hot-spot-analysis`,this.props=e;let t=this.id;this.islands=A(t,e.islands),this.minimumNeighbors=H(t,e.minimumNeighbors),this.multipleTesting=F(t,e.multipleTesting);let n=W({...e,id:t,globalStatistics:e.output.globalStatistics}),r=e.selfWeight??1;if(!Number.isFinite(r)||r<0)throw Error(`${t} selfWeight must be a finite number >= 0`);p(e.output.zScores,[`float32`],`${t} zScores`),e.output.bins&&p(e.output.bins,[`sint32`],`${t} bins`),e.output.pValues&&p(e.output.pValues,[`float32`],`${t} pValues`),e.output.adjustedPValues&&p(e.output.adjustedPValues,[`float32`],`${t} adjustedPValues`),e.output.neighborCounts&&v(e.output.neighborCounts,`${t} neighborCounts`);for(let[r,i]of[[`zScores`,e.output.zScores],[`bins`,e.output.bins],[`pValues`,e.output.pValues],[`adjustedPValues`,e.output.adjustedPValues],[`neighborCounts`,e.output.neighborCounts]])if(i&&i.length!==n)throw Error(`${t} ${r} length must equal the weights row count`);g(t,[e.output.zScores,e.output.bins,e.output.pValues,e.output.adjustedPValues,e.output.neighborCounts,e.output.globalStatistics],[e.weights.offsets,e.weights.neighbors,e.weights.weights,e.weights.status?.overflow,e.values,e.parameters,e.mask])}getInferenceMetadata(e={}){return B(e),{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,testedRowsDefinition:`included rows with at least ${this.minimumNeighbors} active neighbor${this.minimumNeighbors===1?``:`s`}${this.islands===`exclude`?` (islands excluded)`:``} and a defined z-score`}}getCommandNodes(e){let{id:t,props:n}=this,{weights:r,parameters:i}=n,{zScores:a}=n.output,o=n.selfWeight??1;h(t,e,[r.offsets,r.neighbors,r.weights,r.status?.overflow,n.values,i,n.mask,a,n.output.bins,n.output.pValues,n.output.adjustedPValues,n.output.neighborCounts,n.output.globalStatistics]);let s=n.values.length,c=U(e,{...n,globalStatistics:n.output.globalStatistics,id:t,islands:this.islands,operation:_n}),l=c.nodes,u=G,d=[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`weights`,view:r.weights,type:`f32`,access:`read`},{name:`statistics`,view:c.statistics,type:`f32`,access:`read`},{name:`zScores`,view:a,type:`f32`,access:`read_write`}];n.output.neighborCounts&&d.push({name:`neighborCounts`,view:n.output.neighborCounts,type:`u32`,access:`read_write`});let f=this.multipleTesting!==`none`,p=!!(n.output.bins||n.output.pValues||n.output.adjustedPValues),m=f&&!!(n.output.bins||n.output.adjustedPValues),g=Number(!!n.output.bins)+Number(!!n.output.pValues)+Number(!!n.output.adjustedPValues),_=p&&!m&&d.length+g<=gn,v=m?n.output.pValues??x(e,`${t}-p-values`,`float32`,s):void 0,y=m?n.output.adjustedPValues??x(e,`${t}-adjusted-p-values`,`float32`,s):void 0,[b,S,C]=z.map(T),[w,E,ee]=V.map(T),O=[];n.output.bins&&O.push({name:`bins`,view:n.output.bins,type:`i32`,access:`read_write`}),n.output.pValues&&O.push({name:`pValues`,view:n.output.pValues,type:`f32`,access:`read_write`}),n.output.adjustedPValues&&O.push({name:`adjustedPValues`,view:n.output.adjustedPValues,type:`f32`,access:`read_write`});let A=`let finite = isFiniteFloat(zScore);
  var level = 0;
  if (finite) {
    let absoluteZ = abs(zScore);
    level = select(select(select(0, 1, absoluteZ >= ${b}), 2, absoluteZ >= ${S}), 3, absoluteZ >= ${C});
  }
  ${n.output.bins?`bins[binsOffset + index] = select(level, -level, zScore < 0.0);`:``}
  ${n.output.pValues||n.output.adjustedPValues?`let pValue = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);
  ${n.output.pValues?`pValues[pValuesOffset + index] = pValue;`:``}
  ${n.output.adjustedPValues?`adjustedPValues[adjustedPValuesOffset + index] = pValue;`:``}`:``}`;return _&&d.push(...O),v&&d.push({name:`pValues`,view:v,type:`f32`,access:`read_write`}),l.push(D(e,{id:`${t}-neighbors`,operation:_n,variant:`${o===0?`gi`:`gi-star`}${_?`-classify`:``}${m?`-p-value`:``}`,bindings:d,invocationCount:s,declarations:`${u}
${ae(s,r.neighbors.length)}
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${k(r)};
const SELF_WEIGHT: f32 = ${T(o)};`,body:`let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${R(`weightedSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`)}
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      weightedSum = weightedSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    let count = statistics[statisticsOffset + MOMENTS];
    let variance = statistics[statisticsOffset + MOMENTS + 2u];
    if (neighborCount >= MINIMUM_NEIGHBORS) {
      ${o===0?`// Ord-Getis Gi: standardize with the n - 1 other observations. The centered values of the
      // population sum to zero, so the others sum to -centered.
      let others = count - 1.0;
      let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
      let otherMean = -centered / others;
      let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
      let spread = (others * squareSum - weightSum * weightSum) / (others - 1.0);
      if (others >= 2.0 && otherVariance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = (weightedSum - weightSum * otherMean) / (sqrt(otherVariance) * sqrt(spread));
      }`:`weightedSum += SELF_WEIGHT * centered;
      weightSum += SELF_WEIGHT;
      squareSum += SELF_WEIGHT * SELF_WEIGHT;
      let spread = (count * squareSum - weightSum * weightSum) / (count - 1.0);
      if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = weightedSum / (sqrt(variance) * sqrt(spread));
      }`}
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${n.output.neighborCounts?`neighborCounts[neighborCountsOffset + index] = neighborCount;`:``}
  ${v?`pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));`:``}
  ${_?A:``}`})),p&&!m&&!_&&l.push(D(e,{id:`${t}-classify`,operation:_n,variant:`classify`,bindings:[{name:`zScores`,view:a,type:`f32`,access:`read`},...O],invocationCount:s,declarations:u,body:`let zScore = zScores[zScoresOffset + index];
  ${A}`})),m&&l.push(...new I({id:`${t}-multiple-testing`,pValues:v,method:this.multipleTesting,output:{adjustedPValues:y}}).getCommandNodes(e)),!m||!n.output.bins||l.push(D(e,{id:`${t}-classify`,operation:_n,variant:`classify-adjusted`,bindings:[{name:`zScores`,view:a,type:`f32`,access:`read`},{name:`adjustedPValues`,view:y,type:`f32`,access:`read`},{name:`bins`,view:n.output.bins,type:`i32`,access:`read_write`}],invocationCount:s,declarations:G,body:`let zScore = zScores[zScoresOffset + index];
  let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  var level = 0;
  if (isFiniteFloat(zScore) && isFiniteFloat(adjusted)) {
    level = select(select(select(0, 1, adjusted <= ${w}), 2, adjusted <= ${E}), 3, adjusted <= ${ee});
  }
  bins[binsOffset + index] = select(level, -level, zScore < 0.0);`})),l}},$=`GPUHotSpotAnalysisRecipe`,yn=Object.freeze({coordinateSpaces:Object.freeze([`longitude-latitude`]),metrics:Object.freeze([`great-circle`]),measure:`distance`}),bn=r/1e3;function xn(){return`const PI: f32 = 3.14159265358979;
const EARTH_RADIUS_SQUARED_KILOMETERS: f32 = ${T(bn*bn)};
fn compactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
fn getQuadbinAreaKilometers(key: vec2u) -> f32 {
  if (!cellIsValidQuadbin(key)) {
    return 0.0;
  }
  let z = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * z, 2u * z);
  let y = compactEvenBits(morton.y >> 1u) | (compactEvenBits(morton.x >> 1u) << 16u);
  let tiles = f32(1u << z);
  let step = 2.0 * PI / tiles;
  let north = PI * f32(i32(1u << z) - 2 * i32(y)) / tiles;
  let south = north - step;
  return EARTH_RADIUS_SQUARED_KILOMETERS * step * sinh(step) / (cosh(north) * cosh(south));
}`}function Sn(e,t){return N(e,{type:`disk`,k:t})-1}function Cn(e,t){let n=t.id??`hot-spot-recipe`,r=t.spatialContext??{coordinateSpace:`longitude-latitude`,metric:`none`};if(t.prepare){if(r.coordinateSpace!==`longitude-latitude`)throw Error(`${n} prepare needs a longitude-latitude spatialContext`);if(t.preparedPositionCapacity===void 0)throw Error(`${n} prepare needs preparedPositionCapacity`);if(!Number.isInteger(t.preparedPositionCapacity)||t.preparedPositionCapacity<1)throw Error(`${n} preparedPositionCapacity must be a positive integer`);if(!t.preparationParameters)throw Error(`${n} prepare needs preparationParameters`)}if(t.prepare&&t.source.kind!==`points`)throw Error(`${n} prepare needs a points source`);let o=new Ee(e),c=t.outputs??{},l=t.scratch??{},{source:u}=t,d=t.minimumNeighbors??8;Te($,Number.isInteger(d)&&d>=0,`minimumNeighbors must be a non-negative integer`);let p=t.multipleTesting??`benjamini-hochberg`;Te($,t.parameters.length>=8,`parameters needs 8 values`),Te($,!t.permutation||t.permutation.parameters.length>=4,`permutation.parameters needs 4 values`);let m=t.parameters.importToGraph(e,`${n}-parameters`),h,g,_,v,y,b,S,C,w=[],T=Y(e,`${n}-weights-overflow`,`uint32`,1,c.status?.weightsOverflow),k=Y(e,`${n}-weights-required-count`,`uint32`,1,c.status?.weightsRequiredCount),A={overflow:T,requiredCount:k},j;if(u.kind===`points`){j=`study-area cells`,h=u.cellCapacity,Te($,Number.isInteger(h)&&h>=1,`cellCapacity must be positive`);let p=t.analyze??`density`;Te($,p===`density`||p===`count`,`analyze must be 'density' or 'count'`),Te($,!(p===`count`&&u.family===`quadbin`),`analyze 'count' is not allowed for quadbin: Web Mercator cells vary in area with latitude, use 'density'`);let{distanceBand:m}=u;m&&(Te($,u.neighborRings===void 0&&u.neighborCapacity===void 0,`distanceBand is exclusive with neighborRings and neighborCapacity`),i(`${$} distanceBand`,m.spatialContext,yn));let O=u.neighborRings??(u.family===`h3`?2:1),M=Sn(u.family,O);Te($,Number.isInteger(O)&&O>=1,`neighborRings must be an integer >= 1`),Te($,!!m||d<=M,`minimumNeighbors ${d} exceeds the ${M} neighbors of a ${u.family} disk with neighborRings ${O}; raise neighborRings or lower minimumNeighbors`);let N=m?.neighborCapacity??u.neighborCapacity??h*M,P=u.pointCellCapacity??h,F=u.studyArea.coverCapacity??h,I=l.table??{},L=l.pointTable??{},R={featureIds:x(e,`${n}-cover-features`,`uint32`,F),cells:x(e,`${n}-cover-cells`,`uint32x2`,F),status:{count:x(e,`${n}-cover-count`,`uint32`,1),requiredCount:Y(e,`${n}-cover-required-count`,`uint32`,1,c.studyArea?.status?.requiredCount),overflow:x(e,`${n}-cover-overflow`,`uint32`,1),candidateOverflow:x(e,`${n}-cover-candidate-overflow`,`uint32`,1),invalidCount:Y(e,`${n}-cover-invalid-count`,`uint32`,1,c.studyArea?.status?.invalidCount),uncertainCount:Y(e,`${n}-cover-uncertain-count`,`uint32`,1,c.studyArea?.status?.uncertainCount)}},z=u.studyArea,B=i(n,r,{coordinateSpaces:[`longitude-latitude`],metrics:[`none`,`great-circle`]});t.prepare||a(n,`studyArea`,u.studyAreaSpatialContext,B,s);let V=ge(B);if(t.prepare){let r={positions:Y(e,`${n}-prepare-positions`,`float32x2`,t.preparedPositionCapacity),featureOffsets:Y(e,`${n}-prepare-features`,`uint32`,z.featureOffsets.length),polygonOffsets:Y(e,`${n}-prepare-polygons`,`uint32`,2*z.polygonOffsets.length-1),ringOffsets:Y(e,`${n}-prepare-rings`,`uint32`,2*z.ringOffsets.length-1),status:{overflow:Y(e,`${n}-prepare-overflow`,`uint32`,1),requiredCount:Y(e,`${n}-prepare-required-count`,`uint32`,1),invalidCount:Y(e,`${n}-prepare-invalid-count`,`uint32`,1),approximation:Y(e,`${n}-prepare-approximation`,`uint32`,1)}},i=o.add(new xe({...z,id:`${n}-prepare`,spatialContext:u.studyAreaSpatialContext??V,parameters:t.preparationParameters,output:r}));u.studyAreaSpatialContext!==void 0&&a(n,`studyArea`,i.spatialContext,{...B,edgeModel:`linear`},s),z={...z,...r},V=ge(i.spatialContext),w.push({stage:`study-area-preparation`,status:r.status})}o.add(new ne({id:`${n}-cover`,family:u.family,resolution:u.resolution,containment:`center`,spatialContext:V,positions:z.positions,featureOffsets:z.featureOffsets,polygonOffsets:z.polygonOffsets,ringOffsets:z.ringOffsets,candidateCapacity:u.studyArea.candidateCapacity,output:R}));let ae=x(e,`${n}-cover-mask`,`uint32`,F);e.add(D(e,{id:`${n}-cover-mask`,operation:$,variant:`cover-mask`,bindings:[{name:`coverCount`,view:R.status.count,type:`u32`,access:`read`},{name:`coverMask`,view:ae,type:`u32`,access:`read_write`}],invocationCount:F,body:`coverMask[coverMaskOffset + index] = select(0u, 1u, index < coverCount[coverCountOffset]);`}));let H={cells:Y(e,`${n}-cells`,`uint32x2`,h,I.cells),counts:Y(e,`${n}-study-counts`,`uint32`,h,I.counts),status:{count:Y(e,`${n}-count`,`uint32`,1,I.status?.count),overflow:Y(e,`${n}-table-overflow`,`uint32`,1,I.status?.overflow),requiredCount:Y(e,`${n}-study-required-count`,`uint32`,1,I.status?.requiredCount)}};o.add(new te({id:`${n}-study-cells`,family:u.family,resolution:u.resolution,cells:R.cells,mask:ae,output:H}));let U={cells:Y(e,`${n}-point-cells`,`uint32x2`,P,L.cells),counts:Y(e,`${n}-point-counts`,`uint32`,P,L.counts),sumValues:u.values?Y(e,`${n}-point-sum-values`,`float32`,P,L.sumValues):void 0,status:{count:Y(e,`${n}-point-count`,`uint32`,1,L.status?.count),overflow:Y(e,`${n}-point-overflow`,`uint32`,1,L.status?.overflow),requiredCount:Y(e,`${n}-point-required-count`,`uint32`,1,L.status?.requiredCount)}};if(u.family===`quadbin`)o.add(new te({id:`${n}-aggregate`,family:`quadbin`,resolution:u.resolution,positions:u.positions,values:u.values,mask:u.mask,output:U}));else{let t=Y(e,`${n}-point-keys`,`uint32x2`,u.positions.length,l.pointKeys);o.add(new ie({id:`${n}-point-to-cell`,family:u.family,resolution:u.resolution,positions:u.positions,mask:u.mask,output:{cells:t}})),o.add(new te({id:`${n}-aggregate`,family:u.family,resolution:u.resolution,cells:t,values:u.values,mask:u.mask,output:U}))}let W=x(e,`${n}-joined-counts`,`uint32`,h),G=u.values?x(e,`${n}-joined-sums`,`float32`,h):void 0;e.add(D(e,{id:`${n}-join`,operation:$,variant:`join`,bindings:[{name:`studyCells`,view:H.cells,type:`u32`,access:`read`},{name:`pointCells`,view:U.cells,type:`u32`,access:`read`},{name:`pointCounts`,view:U.counts,type:`u32`,access:`read`},...U.sumValues?[{name:`pointSums`,view:U.sumValues,type:`f32`,access:`read`}]:[],{name:`pointCount`,view:U.status.count,type:`u32`,access:`read`},{name:`joinedCounts`,view:W,type:`u32`,access:`read_write`},...G?[{name:`joinedSums`,view:G,type:`f32`,access:`read_write`}]:[]],invocationCount:h,declarations:we(`findPointCell`,`pointCells`,`pointCount[pointCountOffset]`),body:`let key = vec2u(studyCells[studyCellsOffset + 2u * index + 1u], studyCells[studyCellsOffset + 2u * index]);
  let row = findPointCell(key);
  let found = row != 0xffffffffu;
  let safeRow = select(0u, row, found);
  joinedCounts[joinedCountsOffset + index] = select(0u, pointCounts[pointCountsOffset + safeRow], found);
  ${G?`joinedSums[joinedSumsOffset + index] = select(0.0, pointSums[pointSumsOffset + safeRow], found);`:``}`})),S=Y(e,`${n}-points-outside`,`uint32`,1,c.status?.pointsOutsideStudyArea),e.add(E(e,{id:`${n}-points-outside-clear`,operation:$,view:S,type:`u32`,value:`0u`})),e.add(D(e,{id:`${n}-points-outside-count`,operation:$,variant:`outside`,bindings:[{name:`pointCells`,view:U.cells,type:`u32`,access:`read`},{name:`pointCounts`,view:U.counts,type:`u32`,access:`read`},{name:`pointCount`,view:U.status.count,type:`u32`,access:`read`},{name:`studyCells`,view:H.cells,type:`u32`,access:`read`},{name:`studyCount`,view:H.status.count,type:`u32`,access:`read`},{name:`outside`,view:S,type:`atomic<u32>`,access:`read_write`}],invocationCount:P,declarations:we(`findStudyCell`,`studyCells`,`studyCount[studyCountOffset]`),body:`if (index >= pointCount[pointCountOffset]) {
    return;
  }
  let key = vec2u(pointCells[pointCellsOffset + 2u * index + 1u], pointCells[pointCellsOffset + 2u * index]);
  if (findStudyCell(key) == 0xffffffffu) {
    atomicAdd(&outside[outsideOffset], pointCounts[pointCountsOffset + index]);
  }`})),y=Y(e,`${n}-centers`,`float32x2`,h,l.centers),o.add(new re({id:`${n}-geometry`,family:u.family,cells:H.cells,output:{centers:y}})),g=Y(e,`${n}-values`,`float32`,h,l.values),_=Y(e,`${n}-mask`,`uint32`,h,l.mask);let K;p===`density`&&u.family===`h3`&&(K=x(e,`${n}-areas`,`float32`,h),o.add(new yt({id:`${n}-measures`,family:`h3`,cells:H.cells,units:`kilometers`,output:{areas:K}})));let oe=p===`density`&&u.family===`quadbin`,se=oe?`getQuadbinAreaKilometers(key)`:K?`areas[areasOffset + index]`:`1.0`;e.add(D(e,{id:`${n}-analysis-values`,operation:$,variant:`analysis-values`,bindings:[{name:`joinedCounts`,view:W,type:`u32`,access:`read`},...G?[{name:`joinedSums`,view:G,type:`f32`,access:`read`}]:[],...oe?[{name:`studyCells`,view:H.cells,type:`u32`,access:`read`}]:[],...K?[{name:`areas`,view:K,type:`f32`,access:`read`}]:[],{name:`studyCount`,view:H.status.count,type:`u32`,access:`read`},{name:`analysisValues`,view:g,type:`f32`,access:`read_write`},{name:`analysisMask`,view:_,type:`u32`,access:`read_write`}],invocationCount:h,declarations:`${f}
${ee}
${oe?xn():``}`,body:`${oe?`let key = vec2u(studyCells[studyCellsOffset + 2u * index + 1u], studyCells[studyCellsOffset + 2u * index]);`:``}
  let area = ${se};
  let rowIsStudy = index < studyCount[studyCountOffset];
  let valid = rowIsStudy && area > 0.0;
  let raw = ${G?`joinedSums[joinedSumsOffset + index]`:`f32(joinedCounts[joinedCountsOffset + index])`};
  analysisMask[analysisMaskOffset + index] = select(0u, 1u, valid);
  analysisValues[analysisValuesOffset + index] = select(getQuietNaN(index), ${p===`density`?`raw / area`:`raw`}, valid);`}));let ce=Se(e,`${n}-weights`,h,N,{...l.weights,status:A});if(b=ce,m){C=Y(e,`${n}-weights-invalid-count`,`uint32`,1,c.status?.weightsInvalidCount);let t=new ve({id:`${n}-neighbors`,mode:`radius`,positions:y,mask:_,spatialContext:m.spatialContext,parameters:m.neighborSearchParameters,gridSize:m.gridSize,output:{...ce,status:{...A,invalidCount:C}},rowStandardize:m.rowStandardize??!1,weightTransform:m.rowStandardize?`row`:`binary`});o.add(t),b=t.outputWeights}else Ce(e,o,`${n}-neighbors`,{family:u.family,cells:H.cells,count:H.status.count,mask:_,rings:O,weights:ce,overflow:T,requiredCount:k}),b={...ce,information:{rowCount:h,transform:`binary`,symmetric:!0,provenance:{source:`GPUCellTopology`,criterion:`disk`,order:O}}};v={cells:H.cells,counts:W,sumValues:G,status:{count:H.status.count,overflow:H.status.overflow}},w.push({stage:`study-area`,status:R.status},{stage:`study-table`,status:H.status},{stage:`point-aggregation`,status:U.status})}else{j=`lattice cells`,h=u.width*u.height,g=u.values,_=u.mask;let t=Se(e,`${n}-weights`,h,u.neighborCapacity,{...l.weights,status:A}),r=new L({id:`${n}-lattice`,width:u.width,height:u.height,criterion:u.criterion??`queen`,radius:u.radius,mask:_,output:t});o.add(r),b=r.outputWeights}w.push({stage:`weights`,status:{overflow:T,requiredCount:k,...C?{invalidCount:C}:{}}});let M=Y(e,`${n}-z-scores`,`float32`,h,c.zScores),N=Y(e,`${n}-bins`,`sint32`,h,c.bins),F=Y(e,`${n}-p-values`,`float32`,h,c.pValues),I=Y(e,`${n}-neighbor-counts`,`uint32`,h,c.neighborCounts),R=Y(e,`${n}-global-statistics`,`float32`,5,c.globalStatistics),z=o.add(new vn({id:`${n}-gi-star`,weights:b,selfWeight:t.selfWeight,values:g,mask:_,parameters:m,multipleTesting:p,minimumNeighbors:Math.max(1,d),output:{zScores:M,bins:N,pValues:F,neighborCounts:I,globalStatistics:R}})),V=Y(e,`${n}-invalid-count`,`uint32`,1,c.status?.invalidCount);e.add(E(e,{id:`${n}-invalid-clear`,operation:$,view:V,type:`u32`,value:`0u`}));let ae=[{name:`analysisValues`,view:g,type:`f32`,access:`read`},..._?[{name:`analysisMask`,view:_,type:`u32`,access:`read`}]:[],{name:`zScores`,view:M,type:`f32`,access:`read`},{name:`invalid`,view:V,type:`atomic<u32>`,access:`read_write`}];e.add(D(e,{id:`${n}-invalid-tally`,operation:$,variant:`invalid-tally`,bindings:ae,invocationCount:h,declarations:O,body:`let included = ${_?`analysisMask[analysisMaskOffset + index] != 0u && `:``}isFiniteValue(analysisValues[analysisValuesOffset + index]);
  if (included && !isFiniteValue(zScores[zScoresOffset + index])) {
    atomicAdd(&invalid[invalidOffset], 1u);
  }`})),w.push({stage:`minimum-neighbors`,status:{invalidCount:V}});let H={contributors:o.contributors,rowCount:h,table:v,centers:y,values:g,mask:_,weights:b,zScores:M,bins:N,pValues:F,neighborCounts:I,globalStatistics:R,inference:void 0,setSettings:e=>G(e),outputs:{zScores:M,bins:N,pValues:F,neighborCounts:I,globalStatistics:R},intermediates:{table:v,centers:y,values:g,mask:_,weights:b},status:{weightsOverflow:T,weightsRequiredCount:k,...C?{weightsInvalidCount:C}:{},invalidCount:V,...S?{pointsOutsideStudyArea:S}:{},stages:w}},{permutation:U}=t,W;if(U){let t=c.permutation??{},r={exceedances:Y(e,`${n}-exceedances`,`uint32`,h,t.exceedances),pseudoPValues:Y(e,`${n}-pseudo-p-values`,`float32`,h,t.pseudoPValues),significant:Y(e,`${n}-significant`,`uint32`,h,t.significant),status:{overflow:Y(e,`${n}-permutation-overflow`,`uint32`,1,t.status?.overflow)}},i=U.alternative??`two-sided`;W=o.add(new ye({id:`${n}-permutation`,weights:b,values:g,mask:_,statistic:U.statistic??`localGStar`,alternative:i,parameters:U.parameters.importToGraph(e,`${n}-permutation-parameters`),maximumPermutations:U.maximumPermutations,maximumNeighbors:U.maximumNeighbors,multipleTesting:U.multipleTesting,output:{exceedances:r.exceedances,pseudoPValues:r.pseudoPValues,significant:r.significant,status:r.status}})),H.permutation=r,H.outputs.permutation=r,H.status.stages=[...H.status.stages,{stage:`permutation`,status:r.status}]}function G(e){let{seed:n,permutationCount:r,...i}=e,a=B(i),o={...z.getInferenceMetadata(i),testedRowsDefinition:d>0?`${j} with at least ${d} neighbors`:j},s,c;if(W&&U){let e={seed:n??0,permutationCount:r??U.maximumPermutations,significanceLevel:i.significanceLevel};s=W.getParameterValues(e),c={...W.getInferenceMetadata(e),testedRowsDefinition:o.testedRowsDefinition}}t.parameters.write(a),s&&U.parameters.write(s),H.inference=o,c&&(H.permutationInference=c)}G(t.settings??{});let{color:K}=t;if(K){let t=c.color??{},r={breaks:Y(e,`${n}-breaks`,`float32`,K.maximumClassCount+1,t.breaks),classCount:Y(e,`${n}-class-count`,`uint32`,1,t.classCount),colors:Y(e,`${n}-colors`,`uint32`,h,t.colors),classIndices:Y(e,`${n}-class-indices`,`uint32`,h,t.classIndices)};o.add(new P({id:`${n}-class-breaks`,values:M,mask:_,parameters:K.classBreaksParameters,maximumClassCount:K.maximumClassCount,methods:K.methods,output:{breaks:r.breaks,classCount:r.classCount}}));let i={id:`${n}-color-scale`,values:M,mask:_,domain:r.breaks,domainCount:r.classCount,palette:K.palette,parameters:K.colorScaleParameters,maximumDomainCount:K.maximumClassCount+1,maximumPaletteCount:K.maximumPaletteCount,output:{colors:r.colors,classIndices:r.classIndices}};o.add(new be(i)),H.color=r,H.outputs.color=r}return H}function wn(e,t,n){let r=(t,r)=>e.createBuffer(`${n}-${t}`,r);return{zScores:r(`z-scores`,t*4),pValues:r(`p-values`,t*4),base:r(`base`,t*4),classes:r(`classes`,t*4),counts:r(`counts`,28),statistics:r(`statistics`,20),neighborCounts:r(`neighbor-counts`,t*4),exceedances:r(`exceedances`,t*4),pseudoPValues:r(`pseudo-p-values`,t*4),significant:r(`significant`,t*4),permutationOverflow:r(`permutation-overflow`,4),significantCounts:r(`significant-counts`,8),valuesExtent:r(`values-extent`,8)}}async function Tn(e){let{device:t,resources:n,rows:r,slots:i,buffers:a,variant:o}=e,s=new m(t,{id:e.id}),c={spatialContext:e.spatialContext,offsets:X(s,`offsets`,e.weights.offsets,`uint32`,r+1),neighbors:X(s,`neighbors`,e.weights.neighbors,`uint32`,i),weights:X(s,`weights`,e.weights.weights,`float32`,i),information:pt(r,o.rowStandardized)},l=X(s,`values`,e.values,`float32`,r),u=X(s,`mask`,e.mask,`uint32`,r),d=e.parameters.importToGraph(s),f=e.permutationParameters.importToGraph(s),p=X(s,`z-scores`,a.zScores,`float32`,r),h=X(s,`p-values`,a.pValues,`float32`,r),g=X(s,`statistics`,a.statistics,`float32`,5),_=X(s,`neighbor-counts`,a.neighborCounts,`uint32`,r),v=o.falseDiscoveryRate?`benjamini-hochberg`:`none`,y=o.statistic===`gi-star`,b=o.inference===`permutation`,x=y?`sint32`:`uint32`,S=X(s,`base`,a.base,x,r),T=X(s,`significant`,a.significant,`uint32`,r),E=X(s,`classes`,a.classes,x,r);return o.precomputed||(y?s.add(new vn({id:`gi-star`,weights:c,values:l,mask:u,parameters:d,output:{zScores:p,bins:S,pValues:h,neighborCounts:_,globalStatistics:g},selfWeight:o.selfWeight,multipleTesting:v})):s.add(new K({id:`local-moran`,weights:c,values:l,mask:u,parameters:d,output:{zScores:p,pValues:h,quadrants:S,neighborCounts:_,globalStatistics:g},quadrantGating:o.inference===`analytic`?`analytic`:`none`,multipleTesting:o.inference===`analytic`?v:`none`})),b&&s.add(new ye({id:`local-permutation`,weights:c,values:l,mask:u,statistic:y?o.selfWeight===1?`localGStar`:`localG`:`localMoran`,alternative:o.alternative,parameters:f,maximumPermutations:999,maximumNeighbors:o.maximumNeighbors,multipleTesting:v,output:{exceedances:X(s,`exceedances`,a.exceedances,`uint32`,r),pseudoPValues:X(s,`pseudo-p-values`,a.pseudoPValues,`float32`,r),significant:T,status:{overflow:X(s,`permutation-overflow`,a.permutationOverflow,`uint32`,1)}}}))),b&&s.add(new w({id:`significant-counts`,input:T,output:X(s,`significant-counts`,a.significantCounts,`uint32`,2),edges:[0,1,2]})),Ne(s,{id:`${e.id}-gate`,invocationCount:r,bindings:[{name:`base`,view:S,type:`u32`,access:`read`},{name:`significant`,view:T,type:`u32`,access:`read`},{name:`mask`,view:u,type:`u32`,access:`read`},{name:`classes`,view:E,type:`u32`,access:`read_write`}],body:`
  if (mask[maskOffset + index] == 0u) {
    classes[classesOffset + index] = 100u;
    return;
  }
  var value = base[baseOffset + index];
  let confirmed = significant[significantOffset + index];
  if (${b?`true`:`false`} && confirmed == 0u) { value = 0u; }
  classes[classesOffset + index] = value;`}),s.add(new w({id:`class-counts`,input:E,output:X(s,`counts`,a.counts,`uint32`,y?7:5),edges:y?[-3,-2,-1,0,1,2,3,4]:[0,1,2,3,4,5]})),s.add(new C({id:`values-extent`,input:l,mask:u,output:X(s,`values-extent`,a.valuesExtent,`float32`,2),operation:`extent`})),n.track(await s.compileAsync())}var En=16384,Dn=40,On=[0,0,0,0],kn=3,An=32,jn={coordinateSpace:`longitude-latitude`,datum:`EPSG:4326`,metric:`great-circle`,units:`meters`},Mn=4,Nn={4:22606,5:8544,6:3229,7:1220,8:461,9:174},Pn=2*Math.PI*t,Fn=Math.PI/180,In=e=>e.toFixed(e<10?1:0),Ln=new Map;async function Rn(e){let{device:t}=e,r=!1,i=null,a=0,s=!1,c=new je,l=0,d=!0,f=!0,p=null,h=null,g=-1,_=[`places`,`cellSize`,`hot`,`cold`,`notSignificant`,`outliers`,`hotTotal`,`coldTotal`,`notSignificantShare`,`confirmed`,`permutation`,`moments`,`capacity`,`scale`,`gvfNatural`,`gvfEqual`],v=t=>{e.setStatus(t),e.setReadout(`analysisStatus`,t);for(let t of _)e.setReadout(t,`pending`);e.setChart(`resultBalance`,null),e.setChart(`zHistogram`,null)},y=t=>{if(r||e.signal.aborted)return;let n=`Analysis failed: ${t instanceof Error?t.message:String(t)}`;e.setStatus(n),e.setReadout(`analysisStatus`,n);for(let t of _)e.setReadout(t,`unavailable`)};v(`Loading analysis inputs; results are not ready.`);let x=(t,n)=>{if(t===`nature-cells`){let t=e.datasets.get(`chicago-nature`),n=new Date(Date.UTC(2023,0,1)+t.column(`timestamp`)[0]*1e3).getUTCFullYear();e.setFurniture({title:{sample:`${Z(t.count)} iNaturalist records, Chicago, ${n}`}})}else n&&e.setFurniture({title:{sample:`${Z(n.count)} ${n.unitPlural}, CDC PLACES model estimates`}})},S=t=>{let n=Ln.get(t);return n||(n=gt(t,e.datasets,e.signal),n.catch(()=>Ln.delete(t)),Ln.set(t,n)),n},C=(t=!1)=>{let n=e.options,r=n.statistic===`gi-star`;return{statistic:n.statistic,inference:r&&n.inference===`none`?`analytic`:n.inference,selfWeight:r&&n.selfWeight===`exclude`?0:1,falseDiscoveryRate:n.falseDiscoveryRate&&(r||n.inference!==`none`),alternative:n.alternative,rowStandardized:n.weightTransform===`row`,maximumNeighbors:Number(n.maximumNeighbors),precomputed:t}},w=e=>[e.statistic,e.inference,e.selfWeight,e.falseDiscoveryRate,e.rowStandardized,e.inference===`permutation`?`${e.alternative}-${e.maximumNeighbors}`:``,e.precomputed?`pre`:`full`].join(`:`),T=()=>e.options.source===`nature-cells`?`No observations`:`No data`,E=()=>et(e.ground(),T()),D=()=>{e.setLegendData(`giTable`,E())},ee=(t,n)=>{let r=e.options;if(r.display===`zscore`)return{values:t.zScores,valueFormat:`float32`,colormap:`rdbu`,valueRange:$e,noDataColor:n};let i=E();return{values:t.classes,valueFormat:`uint32`,colormap:`category`,palette:r.statistic===`gi-star`?ct(i,r.showNotSignificant):ut(i,r.showNotSignificant),noDataValue:100,noDataColor:n,highlightClasses:h}},O=(t,n)=>{let r=e.options,i=We[e.ground()];if(!p)return[{style:{values:t,valueFormat:`float32`,colormap:r.ramp,extent:n,sqrtScale:r.source===`nature-cells`,noDataColor:i}}];let a=(e,n)=>({style:{values:t,valueFormat:`float32`,colormap:r.ramp,...ze(e),highlightClasses:h,noDataColor:i},side:n});return r.compareBreaks&&p.alternate?[a(p.alternate.table,`a`),a(p.table,`b`)]:[a(p.table)]},k=()=>{let t=e.options,n=i?.getValues();if(!n||t.classification===`continuous`||n.length<2){p=null,e.setLegendData(`valueClasses`,null);return}let r=t.source===`nature-cells`,a=ht(t.variable),o=Ie(n),s=i=>{let s=Re(n,t.classCount,i),c=He(n,s);return{method:i,table:Be({breaks:s,colors:Ge(r?`natureWeights`:`people`,e.ground(),s.length+1),unit:r?`records per km²`:a.unit,extent:o,format:r?Z:e=>e.toFixed(a.digits),method:`${Ve[i].label}, GVF ${c.toFixed(2)}`,noData:{label:T()}}),gvf:c}},c=s(t.compareBreaks?`natural-breaks`:t.classification);p={...c,counts:Ue(n,c.table.breaks),alternate:t.compareBreaks?s(`equal-interval`):null},e.setLegendData(`valueClasses`,p),e.setReadout(`gvfNatural`,c.gvf.toFixed(2)),e.setReadout(`gvfEqual`,p.alternate?.gvf.toFixed(2)??null)},A=()=>(g!==-1&&(g=-1,N(null)),null),te=(e,t)=>{let n=0,r=e.length;for(;n<r;){let i=n+r>>1;e[i]<=t?n=i+1:r=i}return e.length?n/e.length:NaN},ne=(e,t)=>(t.sortedValues??=Float32Array.from(e.getValues()??[]).sort(),t.sortedValues),re=e=>{let t=L(e);return`${t[0].toUpperCase()}${t.slice(1)}`},j=t=>{if(t===100)return;if(e.options.statistic===`gi-star`){let e=t>2147483647?t-4294967296:t;return E().colors[e+3]}let n=tt(E());return[n.notSignificant,n.highHigh,n.lowHigh,n.lowLow,n.highLow][t]},ie=(t,n,r,i,a)=>{let o=e.options,s=t.zScores[n];if(o.display===`zscore`){let[e,t,n]=Oe(`rdbu`,(s-$e[0])/($e[1]-$e[0]));return{label:o.statistic===`gi-star`?`Gi* z-score`:`Local Moran z-score`,value:Ye(s,2),swatch:[e,t,n,255],emphasis:!0}}if(o.display===`values`){let e=p?Le(p.table,r):-1;return{label:i,value:a,swatch:p&&e>=0?p.table.colors[e]:void 0,emphasis:!0}}return{label:o.statistic===`gi-star`?`Gi* result`:`Local Moran result`,value:re(t.classes[n]),swatch:j(t.classes[n]),emphasis:!0}},M=(t,n)=>{let r=e.options,i=[];return r.display!==`zscore`&&i.push({label:r.statistic===`gi-star`?`Gi* z-score`:`Local Moran z-score`,value:Ye(t.zScores[n],2)}),i.push({label:`Analytic p-value`,value:dt(t.pValues[n])}),r.inference===`permutation`&&i.push({label:`Permutation p-value`,value:dt(t.pseudoPValues[n])}),i.push({label:`Neighbours`,value:Z(t.neighborCounts[n])}),i},N=t=>{let n=i,r=n?.snapshot;if(!n||!r){e.setChart(`zHistogram`,null);return}let a=e.options,o=Array(An).fill(0),[s,c]=$e,l=Math.min(r.tableCount,r.classes.length);for(let e=0;e<l;e++){let t=r.zScores[e];r.classes[e]===100||!Number.isFinite(t)||o[Math.min(An-1,Math.max(0,Math.floor((t-s)/(c-s)*An)))]++}let u=a.statistic===`gi-star`;e.setChart(`zHistogram`,{kind:`histogram`,values:o,xDomain:$e,title:u?`Gi* z-scores`:`Local Moran z-scores`,xLabel:`z-score (standard deviations)`,yLabel:`Places`,...u?{breaks:lt,classColors:E().colors}:{},markers:at.flatMap(e=>[{x:e,label:Ye(e,2)},{x:-e}]),now:t??void 0,nowLabel:`This place`,table:!1,description:`Histogram of z-scores: most places sit near zero; the guides mark the 90, 95 and 99 percent critical values.`})},P=()=>{let t=e.options,n=i,r=n?.snapshot;if(!n||!r||!t.labelHotSpots||t.display===`values`){e.setAnnotations(`hot-spots`,null),e.setAnnotations(`band`,null);return}let a=t.statistic===`gi-star`,o=[],s=Math.min(r.tableCount,r.classes.length);for(let e=0;e<s;e++){let t=r.classes[e];(a?t>=1&&t<=3:t===1)&&Number.isFinite(r.zScores[e])&&o.push(e)}o.sort((e,t)=>r.zScores[t]-r.zScores[e]);let c=n.labelSeparationMeters(),l=[];for(let e of o){let[t,r]=n.rowMeters(e);if(l.every(e=>{let[i,a]=n.rowMeters(e);return(i-t)**2+(a-r)**2>=c**2})&&l.push(e),l.length===kn)break}let u=t.source===`nature-cells`?`records per km²`:ht(t.variable).unit,d=l.map((e,i)=>{let{coordinate:a,name:o,value:s}=n.describeRow(e),c=Ye(r.zScores[e],1),l=o??Pe(Fe,a,{maxDistanceMeters:8e3});return{kind:`note`,id:`hot-${i}`,coordinate:a,title:t.source===`nature-cells`?`${Z(s)} ${u}, z ${c}`:`${ft(s,1)} ${u}`,text:t.source===`nature-cells`?l??void 0:`${l}, z ${c}`,tone:i===0?`accent`:`ink`,priority:6-i}});e.setAnnotations(`hot-spots`,d.length?d:null);let f=n.bandMeters();t.showBand&&f&&l.length?e.setAnnotations(`band`,[{kind:`ring`,id:`neighbourhood`,coordinate:n.describeRow(l[0]).coordinate,radiusMeters:f,text:`neighbourhood ${Je(f)}`,dashed:!0}]):e.setAnnotations(`band`,null)},F=(e,t,n)=>{let r=0,i=t=>{let n=new Float32Array(e,r,t);return r+=t*4,n},a=t=>{let n=new Uint32Array(e,r,t);return r+=t*4,n},o=a(7),s=i(4),c=a(1)[0],l=a(2),u=i(2),d=a(1)[0],f=i(t),p=i(t),m=i(t),h=a(t),g=a(t),_=t,v=0,y=null,b=null;return n&&(_=a(1)[0],v=a(1)[0],y=i(t*2),b=i(t)),{counts:o,statistics:s,permutationOverflow:c,significantCounts:l,extent:[u[0],u[1]],weightsOverflow:d,tableCount:_,tableOverflow:v,zScores:f,pValues:p,pseudoPValues:m,classes:h,neighborCounts:g,centers:y,values:b}},I=(e,t,n,r)=>[{buffer:e.counts,size:28},{buffer:e.statistics,size:16},{buffer:e.permutationOverflow,size:4},{buffer:e.significantCounts,size:8},{buffer:e.valuesExtent,size:8},{buffer:n,size:4},{buffer:e.zScores,size:t*4},{buffer:e.pValues,size:t*4},{buffer:e.pseudoPValues,size:t*4},{buffer:e.classes,size:t*4},{buffer:e.neighborCounts,size:t*4},...r?[{buffer:r.tableCount,size:4},{buffer:r.tableOverflow,size:4},{buffer:r.centers,size:t*8},{buffer:r.values,size:t*4}]:[]],L=t=>{let n=e.options;if(t===100)return`no data`;if(n.statistic===`gi-star`){let e=t>2147483647?t-4294967296:t;if(e===0)return`not significant`;let n=[90,95,99][Math.abs(e)-1]??0;return`${e>0?`hot spot`:`cold spot`}, ${n}%`}return[`not significant`,`High-High cluster`,`Low-High outlier`,`Low-Low cluster`,`High-Low outlier`][t]??`unknown`},R=()=>{let t=i,n=t?.snapshot;if(!t||!n)return;let r=e.options,a=e=>Z(n.counts[e]),o=n.tableCount,s=Math.max(0,n.statistics[0])||o,c=r.statistic===`gi-star`,l=c?7:5,u=0;for(let e=0;e<l;e++)u+=n.counts[e];if(e.setReadout(`places`,Z(s)),c){e.setReadout(`hot`,`${a(6)} / ${a(5)} / ${a(4)}`),e.setReadout(`cold`,`${a(0)} / ${a(1)} / ${a(2)}`),e.setReadout(`notSignificant`,a(3)),e.setReadout(`outliers`,`n/a for Gi*`),e.setReadout(`hotTotal`,Z(n.counts[4]+n.counts[5]+n.counts[6])),e.setReadout(`coldTotal`,Z(n.counts[0]+n.counts[1]+n.counts[2])),e.setReadout(`notSignificantShare`,Ke(n.counts[3]/Math.max(u,1)));let t=et(e.ground(),`No data`);e.setChart(`resultBalance`,{kind:`bars`,title:`Tested places by result`,values:[n.counts[4]+n.counts[5]+n.counts[6],n.counts[0]+n.counts[1]+n.counts[2],n.counts[3]],labels:[`Hot spot`,`Cold spot`,`Not significant`],colors:[t.colors[6],t.colors[0],t.colors[3]],horizontal:!0,table:!1,description:`Counts of tested places classified as hot, cold or not significant at the selected inference settings.`})}else{e.setReadout(`hot`,a(1)),e.setReadout(`cold`,a(3)),e.setReadout(`outliers`,`${a(2)} / ${a(4)}`),e.setReadout(`notSignificant`,a(0)),e.setReadout(`hotTotal`,a(1)),e.setReadout(`coldTotal`,a(3)),e.setReadout(`notSignificantShare`,Ke(n.counts[0]/Math.max(u,1)));let t=tt(et(e.ground(),`No data`));e.setChart(`resultBalance`,{kind:`bars`,title:`Tested places by local Moran result`,values:[n.counts[1],n.counts[3],n.counts[2]+n.counts[4],n.counts[0]],labels:[`High-high`,`Low-low`,`Outlier`,`Not significant`],colors:[t.highHigh,t.lowLow,t.highLow,t.notSignificant],horizontal:!0,table:!1,description:`Counts of tested places classified as high-high, low-low, spatial outliers or not significant.`})}if(e.setReadout(`moments`,`${ft(n.statistics[1],2)} / ${ft(n.statistics[3],2)}`),r.inference===`permutation`){let i=Z(n.significantCounts[1]);e.setReadout(`confirmed`,i),e.setReadout(`permutation`,`${i} ${t.rows===n.tableCount&&!n.centers?`places`:`cells`} confirmed (${r.permutations} permutations)`)}else e.setReadout(`confirmed`,null),e.setReadout(`permutation`,r.inference===`none`?`no test: every quadrant shown`:`analytic p-values`);let d=n.weightsOverflow!==0||n.tableOverflow!==0?`overflow: reduce the radius or the resolution`:n.permutationOverflow===0?`ok`:`ok; rows with more than ${r.maximumNeighbors} neighbours are skipped by the permutation test`;e.setReadout(`capacity`,d),e.setReadout(`scale`,t.describe()),e.setReadout(`cellSize`,t.getCellText()),Number.isFinite(n.extent[0])&&Number.isFinite(n.extent[1])&&e.setLegendExtent(`display`,n.extent),e.setLegendData(`classCounts`,c?Array.from(n.counts.slice(0,7)):Array.from(n.counts.slice(0,5))),e.setLegendData(`basis`,t.getCellText()?`per ${t.getCellText()} cell`:void 0),D(),k(),P(),N(null),e.setCost({records:s,passes:t.getPassCount()}),e.setStatus(``),e.setReadout(`analysisStatus`,`Ready: analysis and readback complete.`)},z=(e,t,n,r,i,a)=>new Me(e,t,n,e=>{try{a(F(e,r,i))}catch(e){y(e)}}),V=r=>{let a=e.datasets.get(`chicago-nature`),l=new b(t,`hotspots-cells`),u=(e,t)=>l.createBuffer(e,t),d=a.count,f=a.column(`position`),p=a.column(`timestamp`),h=a.column(`category`),_=u(`nature-positions`,Float32Array.from(f)),v=u(`nature-mask`,new Uint32Array(d).fill(1)),y=new Uint8Array(d);for(let e=0;e<d;e++)y[e]=Math.floor(p[e]%86400/3600);let x=En,S=x*Dn,T=u(`table-cells`,x*8),E=u(`table-counts`,x*4),D=u(`table-count`,4),k=u(`table-overflow`,4),re=u(`centers-degrees`,x*8),j=e.datasets.get(`chicago-tracts`),P=j.column(`vertices`),F=j.column(`polygonRingOffsets`),L=r.featurePolygonOffsets,B={vertices:u(`study-vertices`,Float32Array.from(P)),vertexCount:P.length/2,featureOffsets:u(`study-feature-offsets`,L),polygonOffsets:u(`study-polygon-offsets`,Uint32Array.from(F)),ringOffsets:u(`study-ring-offsets`,Uint32Array.from(j.column(`ringOffsets`)))},V=u(`centers-meters`,x*8),ae={offsets:u(`offsets`,(x+1)*4),neighbors:u(`neighbors`,S*4),weights:u(`weights`,S*4)},H=u(`weights-overflow`,4),U=u(`values`,x*4),W=u(`mask`,x*4),G=wn(l,x,`cells`),K=l.createParameterBuffer(`autocorrelation-parameters`,`float32`,8),oe=l.createParameterBuffer(`permutation-parameters`,`uint32`,4),se=l.createParameterBuffer(`search-parameters`,`float32`,12),[ce,le]=r.origin,ue=r.analysisFrame,de=_t(l,r,`cells-tracts`),fe=()=>e.options.family===`quadbin`?e.options.resolution:e.options.h3Resolution,q=()=>{let t=e.options;return t.family===`quadbin`?Pn/2**t.resolution*Math.cos(le*Fn):Math.sqrt(3)*(Nn[t.h3Resolution]??461)},pe=()=>{let t=e.options;return t.family===`quadbin`?q()/2:Nn[t.h3Resolution]??461},me=new Map,he=new Map,J=new Map,ge=()=>{let t=e.options,n=C();return[t.family,fe(),n.selfWeight,n.falseDiscoveryRate,n.rowStandardized,t.statistic===`gi-star`&&t.inference===`permutation`?`perm-${t.alternative}-${t.maximumNeighbors}`:`plain`].join(`:`)},ve=()=>({significanceLevel:e.options.significance,seed:e.options.seed,permutationCount:e.options.permutations}),ye=async()=>{let n=e.options,i=C(),a=new m(t,{id:`hotspots-recipe-${ge()}`}),s=(e,t,n,r)=>X(a,e,t,n,r),c={cells:s(`table-cells`,T,`uint32x2`,x),counts:s(`table-counts`,E,`uint32`,x),status:{count:s(`table-count`,D,`uint32`,1),overflow:s(`table-overflow`,k,`uint32`,1)}},u=s(`centers-degrees`,re,`float32x2`,x),f=n.statistic===`gi-star`&&n.inference===`permutation`,p=Cn(a,{id:`hot-spot`,source:{kind:`points`,positions:s(`nature-positions`,_,`float32x2`,d),mask:s(`nature-mask`,v,`uint32`,d),family:n.family,resolution:fe(),studyArea:{kind:`polygons`,positions:s(`study-vertices`,B.vertices,`float32x2`,B.vertexCount),featureOffsets:s(`study-feature-offsets`,B.featureOffsets,`uint32`,L.length),polygonOffsets:s(`study-polygon-offsets`,B.polygonOffsets,`uint32`,F.length),ringOffsets:s(`study-ring-offsets`,B.ringOffsets,`uint32`,j.column(`ringOffsets`).length),candidateCapacity:x*4},cellCapacity:x,distanceBand:{spatialContext:jn,neighborSearchParameters:se.importToGraph(a),gridSize:[128,128],neighborCapacity:S,rowStandardize:i.rowStandardized}},analyze:`density`,minimumNeighbors:Mn,settings:ve(),parameters:K,selfWeight:i.selfWeight,multipleTesting:i.falseDiscoveryRate?`benjamini-hochberg`:`none`,outputs:{zScores:s(`z-scores`,G.zScores,`float32`,x),bins:s(`bins`,G.base,`sint32`,x),pValues:s(`p-values`,G.pValues,`float32`,x),neighborCounts:s(`neighbor-counts`,G.neighborCounts,`uint32`,x),globalStatistics:s(`statistics`,G.statistics,`float32`,5),status:{weightsOverflow:s(`weights-overflow`,H,`uint32`,1)},permutation:f?{exceedances:s(`exceedances`,G.exceedances,`uint32`,x),pseudoPValues:s(`pseudo-p-values`,G.pseudoPValues,`float32`,x),significant:s(`significant`,G.significant,`uint32`,x)}:void 0},scratch:{table:c,centers:u,values:s(`values`,U,`float32`,x),mask:s(`mask`,W,`uint32`,x),weights:{offsets:s(`offsets`,ae.offsets,`uint32`,x+1),neighbors:s(`neighbors`,ae.neighbors,`uint32`,S),weights:s(`weights`,ae.weights,`float32`,S)}},permutation:f?{parameters:oe,maximumPermutations:999,statistic:i.selfWeight===1?`localGStar`:`localG`,alternative:n.alternative,maximumNeighbors:Number(n.maximumNeighbors),multipleTesting:i.falseDiscoveryRate?`benjamini-hochberg`:`none`}:void 0}),[h,g,y,b]=r.bbox;return a.add(l.track(new hn({id:`cells-centers`,frame:o({kind:`azimuthal-equidistant`,datum:`EPSG:4326`,origin:r.origin}),positions:u,output:s(`centers-meters`,V,`float32x2`,x),bounds:[h-2,g-2,y+2,b+2]}))),he.set(ge(),p),l.track(await a.compileAsync())},be={rows:x,getValues(){let e=be.snapshot;if(!e?.values)return null;let t=Math.min(e.tableCount,e.values.length);return e.values.slice(0,t).filter(e=>e>0)},describeRow(e){let t=be.snapshot,n=t?.centers?.[e*2]??0,r=t?.centers?.[e*2+1]??0;return{coordinate:ue.unproject(n,r),name:null,value:t?.values?.[e]??NaN}},getCellText:()=>Je(q()),labelSeparationMeters:()=>q()*6,rowMeters(e){let t=be.snapshot?.centers;return[t?.[e*2]??0,t?.[e*2+1]??0]},bandMeters:()=>e.options.radiusCells*q(),resources:l,reader:z(l,`hotspots-cells`,I(G,x,H,{tableCount:D,tableOverflow:k,centers:V,values:U}),x,!0,t=>{i!==be||s||c.pending||(be.snapshot=t,R(),e.requestLayers())}),snapshot:null,prepare:async()=>{let n=ge();me.has(n)||me.set(n,await ye());let r=C(e.options.statistic===`gi-star`),i=w(r);J.has(i)||J.set(i,await Tn({spatialContext:jn,device:t,resources:l,id:`hotspots-cells-${i}`,rows:x,slots:S,weights:ae,values:U,mask:W,buffers:G,parameters:K,permutationParameters:oe,variant:r}))},writeParameters:()=>{let t=e.options,[i,a,o,s]=r.bbox,c=q()*4/n/Fn,l=c/Math.cos(Math.max(Math.abs(a),Math.abs(s))*Fn);se.write(_e({spatialContext:jn,bounds:[i-l,a-c,o+l,s+c],radius:t.radiusCells*q(),weightKind:`binary`,rowStandardize:t.weightTransform===`row`})),he.get(ge())?.setSettings(ve())},writeInputs:()=>{let t=e.options,n=t.category===`all`?-1:Number(t.category),[r,i]=t.hours;v.write(Uint32Array.from(y,(e,t)=>(n<0||h[t]===n)&&e>=r&&e<i?1:0))},describe(){let t=q();return`${(t/1e3).toFixed(2)} km cells, band ${(e.options.radiusCells*t/1e3).toFixed(2)} km`},encode(t){me.get(ge()).encode(t,{parameters:void 0});let n=C(e.options.statistic===`gi-star`);J.get(w(n)).encode(t,{parameters:void 0})},getGraphs:()=>[...me.values(),...J.values()],getPassCount:()=>(me.get(ge())?.stats.nodeOrder.length??0)+(J.get(w(C(e.options.statistic===`gi-star`)))?.stats.nodeOrder.length??0),getLayers(){let t=e.options,n=e.ground(),i=[ce,le,0],a=t.display===`values`?O(U,G.valuesExtent).map(({style:e,side:t})=>({style:{...e,discardAtOrBelow:0},side:t})):[{style:ee(G,On),side:void 0}],o=[];t.showOutlines&&o.push(new ke({id:`hotspot-tract-outline`,coordinateOrigin:i,segments:de.outline,instanceCount:de.outlineSegmentCount,widthPixels:.5,color:ot(n)}));for(let{style:e,side:n}of a)o.push(new De({id:`hotspot-cells-${t.family}-${fe()}${n?`-${n}`:``}`,coordinateOrigin:i,positions:V,analysisFrame:r.analysisFrame.layerFrame,instanceCount:x,shape:t.family===`quadbin`?`square`:`hexagon`,radiusMeters:pe()*.98,...e,compareSide:n,opacity:.88}));return o},getTooltip(t){let n=be.snapshot;if(!n?.centers||!n.values||!t.coordinate)return A();let[r,i]=ue.project(t.coordinate[0],t.coordinate[1]),a=e.options,o=pe()*1.05,s=-1,c=o*o;for(let e=0;e<Math.min(n.tableCount,x);e++){let t=n.centers[e*2]-r,a=n.centers[e*2+1]-i,o=t*t+a*a;o<c&&(c=o,s=e)}if(s<0)return A();let l=n.values[s];s!==g&&(g=s,N(n.zScores[s]));let u=pe(),d=n.centers[s*2],f=n.centers[s*2+1],[p,m]=ue.unproject(d-u,f-u),[h,_]=ue.unproject(d+u,f+u),v=ue.unproject(d,f),y=a.family===`quadbin`?{kind:`box`,bounds:[p,m,h,_]}:{kind:`circle`,coordinate:v,radiusMeters:u},b=ne(be,n),S=[ie(n,s,l,`Records per km²`,In(l)),{label:`Rank`,value:`${qe(te(b,l)*100)} percentile`,unit:`of records per km² over the cells`},...a.display===`values`?[]:[{label:`Records per km²`,value:In(l)}],...M(n,s)];return{title:`Cell of ${Je(q())}`,subtitle:Pe(Fe,v,{maxDistanceMeters:8e3})??void 0,rows:S,anchor:v,highlight:y}}};return be},ae=n=>{let r=new b(t,`hotspots-${n.id}`),a=(e,t)=>r.createBuffer(e,t),o=n.count,l=mt({device:t,resources:r,id:`hotspots-core`,geography:n}),d=a(`values`,Float32Array.from(n.getVariable(e.options.variable))),f=a(`mask`,new Uint32Array(o)),p=wn(r,o,`polygons`),m=r.createParameterBuffer(`autocorrelation-parameters`,`float32`,8),h=r.createParameterBuffer(`permutation-parameters`,`uint32`,4),_=_t(r,n,`hotspots`),v=n.id===`us-counties`?e.datasets.get(`us-states`).geojson:null,y=v?Ze(Xe(v).flatMap(({polygon:e})=>e),Qe(n.origin)):new Float32Array,x=y.length?a(`state-lines`,y):null,S=new Map,T=!0,E=()=>{let t=e.options;return{source:t.weights,k:t.k,snapTolerance:0,bandFactor:t.bandFactor,knnCapFactor:0,weightKind:`binary`,kernel:`triangular`,power:1,distanceFloor:0,rowStandardize:!1,transform:t.weightTransform===`row`?`row`:`none`,transformKernel:`bisquare`,bandwidthFactor:0,doubleSum:`one`}},D=async()=>{await l.prepare(E());let e=C(),i=w(e);S.has(i)||S.set(i,await Tn({spatialContext:n.analysisFrame.spatialContext,device:t,resources:r,id:`hotspots-${n.id}-${i}`,rows:o,slots:l.slots,weights:l.csr,values:d,mask:f,buffers:p,parameters:m,permutationParameters:h,variant:e}))},k=()=>{let t=n.getVariable(e.options.variable);d.write(Float32Array.from(t)),f.write(Uint32Array.from(t,e=>Number.isFinite(e)?1:0))},re=()=>{let t=e.options;l.writeParameters(E()),m.write(B({significanceLevel:t.significance})),h.write(u({seed:t.seed,permutationCount:t.permutations,significanceLevel:t.significance})),T=!0},j=z(r,`hotspots-${n.id}`,I(p,o,l.overflow,null),o,!1,t=>{i!==F||s||c.pending||(F.snapshot=t,R(),e.requestLayers())});k();let P=n.analysisFrame,F={rows:o,getValues(){return n.getVariable(e.options.variable).filter(e=>Number.isFinite(e))},describeRow(t){return{coordinate:n.getCentroidLngLat(t),name:`${n.getName(t)}, ${n.getGroupName(t)}`,value:n.getVariable(e.options.variable)[t]}},labelSeparationMeters:()=>n.medianSpacing*(n.count>2e3?8:4),rowMeters:e=>[n.centroids[e*2],n.centroids[e*2+1]],bandMeters:()=>null,getCellText:()=>null,resources:r,reader:j,snapshot:null,prepare:D,writeParameters:re,writeInputs:k,describe(){let t=e.options;return t.weights===`band`?`${(l.getBandMeters(E())/1e3).toFixed(1)} km band`:t.weights===`knn`?`${t.k} nearest centroids`:`${t.weights} contiguity`},encode(e){T&&=(l.encode(e,E()),!1),S.get(w(C())).encode(e,{parameters:void 0})},getGraphs:()=>[...l.getGraphs(),...S.values()],getPassCount:()=>S.get(w(C()))?.stats.nodeOrder.length??0,getLayers(){let t=e.options,r=e.ground(),i=[n.origin[0],n.origin[1],0],a=(t.display===`values`?O(d,p.valuesExtent):[{style:ee(p,We[r]),side:void 0}]).map(({style:e,side:t})=>new Ae({id:`hotspot-fill${t?`-${t}`:``}`,coordinateOrigin:i,triangles:_.triangles,features:_.features,vertexCount:_.triangleVertexCount,...e,compareSide:t,opacity:n.id===`us-counties`?1:.88}));if(t.showOutlines&&a.push(new ke({id:`hotspot-outline`,coordinateOrigin:i,segments:_.outline,instanceCount:_.outlineSegmentCount,widthPixels:n.count>2e3?.4:.5,color:it(r)})),t.showOutlines&&x){let e=nt(r);a.push(new ke({id:`hotspot-state-lines`,coordinateOrigin:i,segments:x,instanceCount:y.length/4,widthPixels:e.widthPixels,color:e.color,outlineColor:e.casing,outlineWidthPixels:(e.casingPixels-e.widthPixels)/2}))}return a},getTooltip(t){if(!t.coordinate)return A();let r=n.pick(t.coordinate[0],t.coordinate[1]);if(r<0)return A();let i=e.options,a=ht(i.variable),o=n.getVariable(i.variable)[r],s=F.snapshot,c=Number.isFinite(o)?o.toFixed(a.digits):`no data`,l=[];s&&(r!==g&&(g=r,N(s.zScores[r])),l.push(ie(s,r,o,a.label,c)));let u=s?ne(F,s):null;l.push({label:`Rank`,value:u?`${qe(te(u,o)*100)} percentile`:`–`,unit:`of ${n.unitPlural}`},...i.display===`values`&&s?[]:[{label:a.label,value:c,unit:a.unit}]),s&&l.push(...M(s,r));let d=[];for(let e=n.featureRingOffsets[r];e<n.featureRingOffsets[r+1];e++){let t=[];for(let r=n.contiguityRingOffsets[e];r<n.contiguityRingOffsets[e+1];r++)t.push(P.unproject(n.analysisContiguityVertices[r*2],n.analysisContiguityVertices[r*2+1]));d.push(t)}return{title:n.getName(r),subtitle:n.getGroupName(r),rows:l,anchor:n.getCentroidLngLat(r),highlight:{kind:`polygon`,rings:d}}}};return F},H=async e=>e===`nature-cells`?V(await S(`chicago-tracts`)):ae(await S(e)),U=async t=>{let n=++a,o=null;try{if(v(`Loading and compiling the selected source; results are not ready.`),s=!0,p=null,e.setFurniture({title:{sample:`Loading selected source…`}}),e.setAnnotations(`hot-spots`,null),e.setAnnotations(`band`,null),e.requestLayers(),o=await H(t),n!==a||r){o.reader.stop(),o.resources.destroy();return}let c;do c=l,await o.prepare();while(c!==l&&n===a&&!r);if(n!==a||r){o.reader.stop(),o.resources.destroy();return}let u=i;i=o,u&&(u.reader.stop(),setTimeout(()=>u.resources.destroy(),200)),x(t,t===`nature-cells`?void 0:await S(t)),o.writeInputs(),o.writeParameters(),d=!0,f=!0,s=!1,v(`Computing selected source; waiting for GPU readback.`),e.requestLayers()}catch(e){o&&i!==o&&(o.reader.stop(),o.resources.destroy()),n===a&&y(e)}};try{if(i=await H(e.options.source),e.signal.aborted)throw Error(`aborted`);x(e.options.source,e.options.source===`nature-cells`?void 0:await S(e.options.source)),i.writeInputs(),v(`Compiling analysis; results are not ready.`),await i.prepare(),i.writeParameters(),v(`Computing analysis; waiting for GPU readback.`)}catch(e){throw y(e),i?.reader.stop(),i?.resources.destroy(),e}return{getCompiledGraphs:()=>i?.getGraphs()??[],setOption(t,n){if(l++,[`source`,`statistic`,`display`,`classification`,`classCount`,`compareBreaks`].includes(t)&&(h=null),t===`source`){U(n);return}let r=i;if(!(!r||s)){if(t===`classification`||t===`classCount`||t===`ramp`||t===`compareBreaks`){k(),e.requestLayers();return}if([`display`,`labelHotSpots`,`showBand`,`showNotSignificant`,`showOutlines`].includes(t)){P(),e.requestLayers();return}[`category`,`hours`,`variable`].includes(t)&&r.writeInputs(),v(`Updating analysis; waiting for GPU readback.`),c.request({build:()=>r.prepare(),commit:()=>{i!==r||s||(r.writeParameters(),d=!0,f=!0,e.requestLayers())}}).catch(e=>{i===r&&!s&&y(e)})}},encode(e,t){let n=i;!n||s||c.pending||((d||t.frameIndex<2)&&(n.encode(e),d=!1,f=!0),f&&t.frameIndex>=1&&!n.reader.isPending&&(f=!1,n.reader.request(e)))},getLayers:()=>s?[]:i?.getLayers()??[],onGroundChange:()=>{D(),k(),N(null),e.requestLayers()},onLegendFilter(t,n){h=n===null?null:t===`gi-classes`?n.map(e=>st[e]):t===`moran-classes`?n.map(e=>rt[e]):[...n],e.requestLayers()},getTooltip:e=>s?null:i?.getTooltip(e)??null,destroy(){r=!0,a++,c.destroy();let e=i;i=null,e&&(e.reader.stop(),e.resources.destroy())}}}export{Rn as createHotSpots};