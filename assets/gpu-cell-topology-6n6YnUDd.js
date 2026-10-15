import{c as e,i as t,n,o as r,r as i,t as a}from"./cell-topology-wgsl-BT8_s49b.js";import{_ as o,h as s,m as c,y as l}from"./resources-CwvhEOBe.js";import{s as u}from"./wgsl-kernel-nodes-Bm5c4bxp.js";var d=`GPUCellTopology`,f=4096;function p(e){return 3*e*(e+1)+1}function m(e,t){if(t.type===`disk`||t.type===`ring`){let{k:n}=t;if(!Number.isInteger(n)||n<0||n>8)throw Error(`${t.type} k must be an integer in [0, 8]`);return t.type===`disk`?e===`h3`?p(n):(2*n+1)**2:n===0?1:(e===`h3`?6:8)*n}if(r(e,t.resolution),t.type===`parent`)return 1;r(e,t.inputResolution);let n=t.resolution-t.inputResolution;if(n<0)throw Error(`children resolution must not be below inputResolution`);let i=(e===`h3`?7:4)**n;if(i>4096)throw Error(`children stride ${i} exceeds ${f}; reduce the resolution depth`);return i}var h=class{id;props;stride;constructor(e){this.id=e.id??`cell-topology`,this.props=e;let t=this.id;if(e.family!==`quadbin`&&e.family!==`h3`)throw Error(`${t} family must be 'quadbin' or 'h3'`);try{this.stride=m(e.family,e.operation)}catch(e){throw Error(`${t} ${e.message}`)}let n=e.cells.length;if(n<1)throw Error(`${t} needs at least one row`);if(l(e.cells,[`uint32x2`],`${t} cells`),e.mask&&(l(e.mask,[`uint32`],`${t} mask`),e.mask.length!==n))throw Error(`${t} mask length must equal the row count`);let{output:r}=e;if(l(r.cells,[`uint32x2`],`${t} output.cells`),r.cells.length!==n*this.stride)throw Error(`${t} output.cells length must be rows * stride = ${n*this.stride}`);if(r.distances){if(e.operation.type!==`disk`)throw Error(`${t} output.distances is only available for the disk operation`);if(l(r.distances,[`uint32`],`${t} output.distances`),r.distances.length!==n*this.stride)throw Error(`${t} output.distances length must equal output.cells length`)}if(r.counts&&(l(r.counts,[`uint32`],`${t} output.counts`),r.counts.length!==n))throw Error(`${t} output.counts length must equal the row count`);s(t,[r.cells,r.distances,r.counts],[e.cells,e.mask])}getCommandNodes(r){let{id:s,props:l,stride:f}=this,{family:h,operation:g,output:_}=l;o(s,r,[l.cells,l.mask,...Object.values(_)]);let v=l.cells.length;c(r.device,s,`output.cells`,v*f*8,`reduce the rows or the disk k`);let y=h===`h3`,b=[{name:`cells`,view:l.cells,type:`u32`,access:`read`},...l.mask?[{name:`rowMask`,view:l.mask,type:`u32`,access:`read`}]:[],{name:`outCells`,view:_.cells,type:`u32`,access:`read_write`},..._.distances?[{name:`outDistances`,view:_.distances,type:`u32`,access:`read_write`}]:[],..._.counts?[{name:`outCounts`,view:_.counts,type:`u32`,access:`read_write`}]:[]],x=g.type===`disk`||g.type===`ring`,S=x?g.k:0,C=x?y?p(S):m(h,{type:`disk`,k:S}):1,w=g.type===`parent`||g.type===`children`?g.resolution:0,T=g.type===`children`?g.inputResolution:0,E=g.type===`children`?g.resolution-T:0,D=y?`dggs_h3_get_resolution(key)`:`((key.x >> 20u) & 0x1fu)`,O=y?`dggs_h3_is_valid_cell_id(key)`:`cellIsValidQuadbin(key)`,k=`true`;g.type===`parent`?k=`${D} >= TARGET_RESOLUTION`:g.type===`children`&&(k=`${D} == INPUT_RESOLUTION`);let A=e=>`
  let words = vec2u(cells[cellsOffset + 2u * ${e}], cells[cellsOffset + 2u * ${e} + 1u]);
  let key = ${l.wordOrder===`high-low`?`words`:`words.yx`};
  var isValid = ${O} && ${k};
  ${l.mask?`isValid = isValid && rowMask[rowMaskOffset + ${e}] != 0u;`:``}`,j=(e,t,n)=>`
  outCells[outCellsOffset + 2u * ${e}] = ${t}.y;
  outCells[outCellsOffset + 2u * ${e} + 1u] = ${t}.x;
  ${_.distances?`outDistances[outDistancesOffset + ${e}] = ${n??`0u`};`:``}`,M;if(x)M=`${A(`index`)}
  var written = 0u;
  if (isValid) {
    written = ${y?`cellTopologyH3Neighborhood`:`cellTopologyQuadbinNeighborhood`}(key, ${g.type===`ring`});
  }
  for (var j = 0u; j < STRIDE; j++) {
    let slot = index * STRIDE + j;
    if (j < written) {${j(`slot`,`cellTopologyCells[j]`,`cellTopologyDistances[j]`)}
    } else {${j(`slot`,`vec2u(0u)`)}
    }
  }
  ${_.counts?`outCounts[outCountsOffset + index] = written;`:``}`;else if(g.type===`parent`){let e=y?`dggs_h3_get_parent(key, TARGET_RESOLUTION)`:`cellTopologyQuadbinGetParent(key, TARGET_RESOLUTION)`;M=`${A(`index`)}
  var parent = vec2u(0u);
  if (isValid) {
    parent = ${e};
  }${j(`index`,`parent`)}
  ${_.counts?`outCounts[outCountsOffset + index] = select(0u, 1u, isValid);`:``}`}else{let e=y?`cellTopologyH3GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)`:`cellTopologyQuadbinGetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)`,t=y?`cellTopologyH3GetChildCount(key, DEPTH)`:`STRIDE`;M=`let row = index / STRIDE;
  let slotInRow = index % STRIDE;${A(`row`)}
  var child = vec2u(0u);
  if (isValid) {
    child = ${e};
  }${j(`index`,`child`)}
  ${_.counts?`if (slotInRow == 0u) {\n    outCounts[outCountsOffset + row] = select(0u, ${t}, isValid);\n  }`:``}`}return[u(r,{id:`${s}-topology`,operation:d,variant:`${h}-${g.type}`,bindings:b,invocationCount:g.type===`children`?v*f:v,declarations:`const STRIDE: u32 = ${f}u;
const CELL_TOPOLOGY_K: u32 = ${S}u;
const CELL_TOPOLOGY_CAPACITY: u32 = ${C}u;
const TARGET_RESOLUTION: u32 = ${w}u;
const INPUT_RESOLUTION: u32 = ${T}u;
const DEPTH: u32 = ${E}u;
${y?e.source:``}
${t}
${y?i:``}
${n}
${y?a:``}`,body:M})]}};export{m as n,h as t};