import{_ as e,d as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{t as a}from"./gpu-scan-BuGhTLDQ.js";import{l as o,s}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{a as c,i as l,l as u,n as d,t as f}from"./spatial-weights-D4bd9AQ6.js";import{t as p}from"./scan-total-BdolCtK7.js";var m=2,h=`GPULatticeWeights`,g=class{id;props;outputWeights;constructor(e){let r=e.id??`lattice-weights`;if(this.id=r,this.props=e,e.criterion!==`rook`&&e.criterion!==`queen`)throw Error(`${r} criterion must be 'rook' or 'queen'`);t(r,`width`,e.width),t(r,`height`,e.height),t(r,`width * height`,e.width*e.height,1,2**31-1);let a=e.radius??1;if(t(r,`radius`,a,1,32),e.cellSize&&!(e.cellSize.length===2&&e.cellSize.every(e=>Number.isFinite(e)&&e>0)))throw Error(`${r} cellSize must be two positive finite numbers`);let o=e.width*e.height;if(e.mask&&(i(e.mask,[`uint32`],`${r} mask`),e.mask.length!==o))throw Error(`${r} mask length must equal width * height`);if(u(r,e.output)!==o)throw Error(`${r} output rows must equal width * height`);this.outputWeights=f(e.output,{information:{rowCount:o,transform:`binary`,symmetric:!0,hasSelfWeights:!1,provenance:{source:h,criterion:e.criterion,order:a}},status:e.output.status}),n(r,[...c(e.output),...l(e.output)],[e.mask])}getCapacityPlan(e={}){let t=this.props.width*this.props.height,n=d({rowCount:t,capacity:this.props.output.neighbors.length,requiredCount:e.requiredCount,distances:!!this.props.output.distances});return{slots:n,estimatedWorkItems:t+n.capacity,estimatedOutputBytes:n.estimatedOutputBytes,estimatedPeakTransientBytes:m*t*Uint32Array.BYTES_PER_ELEMENT}}getCommandNodes(t){let{id:n,props:i}=this,{width:u,height:d,mask:f}=i,m=i.output,{overflow:g,requiredCount:_}=m.status;e(n,t,[f,...c(m),...l(m)]);let v=u*d,y=m.neighbors.length,b=i.radius??1,[x,S]=i.cellSize??[1,1],C=r(t,`${n}-counts`,`uint32`,v),w=r(t,`${n}-starts`,`uint32`,v),T=f?[{name:`mask`,view:f,type:`u32`,access:`read`}]:[],E=`const WIDTH: i32 = ${u};
const HEIGHT: i32 = ${d};
const RADIUS: i32 = ${b};
const QUEEN: bool = ${i.criterion===`queen`};
const CELL_WIDTH: f32 = ${o(x)};
const CELL_HEIGHT: f32 = ${o(S)};`,D=e=>`let cellX = i32(index) % WIDTH;
  let cellY = i32(index) / WIDTH;
  if (${f?`mask[maskOffset + index] != 0u`:`true`}) {
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
      let y = cellY + dy;
      if (y < 0 || y >= HEIGHT) {
        continue;
      }
      for (var dx = -RADIUS; dx <= RADIUS; dx++) {
        let x = cellX + dx;
        if (x < 0 || x >= WIDTH || (dx == 0 && dy == 0)) {
          continue;
        }
        if (!QUEEN && abs(dx) + abs(dy) > RADIUS) {
          continue;
        }
        let neighbor = u32(y * WIDTH + x);
        if (${f?`mask[maskOffset + neighbor] != 0u`:`true`}) {
          ${e}
        }
      }
    }
  }`,O=r(t,`${n}-scan-total`,`uint32`,1);return[s(t,{id:`${n}-counts`,operation:h,variant:`counts`,bindings:[...T,{name:`counts`,view:C,type:`u32`,access:`read_write`}],invocationCount:v,declarations:E,body:`var count = 0u;
  ${D(`count++;`)}
  counts[countsOffset + index] = count;`}),...new a({id:`${n}-scan`,input:C,output:w,mode:`exclusive`}).getCommandNodes(t),...p(t,{id:n,operation:h,counts:C,offsets:w,rowCount:v,total:O}),s(t,{id:`${n}-offsets`,operation:h,variant:`offsets`,bindings:[{name:`scanTotal`,view:O,type:`u32`,access:`read`},{name:`starts`,view:w,type:`u32`,access:`read`},{name:`offsets`,view:m.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:g,type:`u32`,access:`read_write`},..._?[{name:`requiredCount`,view:_,type:`u32`,access:`read_write`}]:[]],invocationCount:v+1,declarations:`const ROWS: u32 = ${v}u;
const CAPACITY: u32 = ${y}u;`,body:`if (index < ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    offsets[offsetsOffset + ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${_?`requiredCount[requiredCountOffset] = total;`:``}
  }`}),s(t,{id:`${n}-emit`,operation:h,variant:`emit`,bindings:[...T,{name:`offsets`,view:m.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:m.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:m.weights,type:`f32`,access:`read_write`},...m.distances?[{name:`distances`,view:m.distances,type:`f32`,access:`read_write`}]:[]],invocationCount:v,declarations:E,body:`var next = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${D(`if (next < end) {
            neighbors[neighborsOffset + next] = neighbor;
            weights[weightsOffset + next] = 1.0;
            ${m.distances?`distances[distancesOffset + next] = sqrt(f32(dx * dx) * CELL_WIDTH * CELL_WIDTH + f32(dy * dy) * CELL_HEIGHT * CELL_HEIGHT);`:``}
            next++;
          }`)}`})]}};export{g as t};