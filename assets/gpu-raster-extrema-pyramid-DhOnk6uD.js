import{M as e,_ as t}from"./resources-CwvhEOBe.js";import{s as n}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{l as r}from"./wgsl-common-BNRVzgd8.js";import{a as i,n as a,o,r as s,s as c}from"./raster-band-utils-CA-sh8xt.js";function l(e,t,n,r=`observer`){if(t!==void 0&&n!==void 0&&t.trim().toUpperCase()!==n.trim().toUpperCase())throw Error(`${e} ${r} heights are ${t} but elevation is ${n}; convert to one vertical datum first`)}var u=-34028234663852886e22,d=34028234663852886e22,f=4,p=16,m=256;function h(e,t={}){c(`GPURasterExtremaPyramid`,e);let{width:n,height:r}=e,i=t.firstBlockSize??4;if(!Number.isInteger(i)||i<1||i>m||i&i-1)throw Error(`GPURasterExtremaPyramid firstBlockSize must be a power of two in [1, 256]`);let a=t.footprint??`bilinear`;if(a!==`cell`&&a!==`bilinear`)throw Error(`GPURasterExtremaPyramid footprint must be 'cell' or 'bilinear'`);let o=t.levelCount??1/0;if(o!==1/0&&(!Number.isInteger(o)||o<1))throw Error(`GPURasterExtremaPyramid levelCount must be an integer >= 1`);let s=[],l=0;for(let e=0;e<o;e++){let t=i*2**e,a=Math.ceil(n/t),o=Math.ceil(r/t);if(s.push({level:e,blockSize:t,width:a,height:o,offset:l}),l+=a*o,a===1&&o===1)break}return{width:n,height:r,firstBlockSize:i,footprint:a,levels:s,length:l}}function g(e,t,n){let{layout:r,combined:i}=t,{width:a,height:o}=n;if(!r||!i)throw Error(`${e} pyramid must provide a layout and a combined view`);if(r.width!==a||r.height!==o)throw Error(`${e} pyramid covers ${r.width}x${r.height} pixels but the grid is ${a}x${o}`);if(r.footprint!==`bilinear`)throw Error(`${e} pyramid must use the 'bilinear' footprint`);if(r.levels.length<1)throw Error(`${e} pyramid has no levels`);return y(e,`pyramid combined`,i,2*r.length),r}function _(e){return`${Math.fround(e)}`.replace(/^(-?\d+)$/,`$1.0`)}function v(e){return e.replace(/([a-z0-9])([A-Z])/g,`$1_$2`).toUpperCase()}function y(t,n,r,i){if(e(r,[`float32`],`${t} ${n}`),r.length!==i)throw Error(`${t} ${n} must contain ${i} values`)}var b=class{id;props;layout;output;constructor(e){this.id=e.id??`raster-extrema-pyramid`,this.props=e;let{id:t}=this;c(t,e.grid),this.layout=h(e.grid,e);let{maximum:n,minimum:r,combined:i}=e.output;if(i&&(n||r))throw Error(`${t} output.combined cannot be used together with maximum or minimum`);if(!n&&!r&&!i)throw Error(`${t} requires at least one output`);for(let[e,i]of[[`maximum`,n],[`minimum`,r]])i&&y(t,`output.${e}`,i,this.layout.length);i&&y(t,`output.combined`,i,2*this.layout.length),o(t,[n,r,i],a(e.input)),i&&(this.output={layout:this.layout,combined:i})}getCommandNodes(e){let{id:n,props:r,layout:a}=this;i(n,e,r.input,[]);let{maximum:o,minimum:c,combined:l}=r.output;t(n,e,[o,c,l]);let u=s(e,n,r.input,r.grid,`reuse-canonical`);return[...u.nodes,...x(e,{id:n,layout:a,values:u.band.storage.values,validity:u.band.validity,maximum:o,minimum:c,combined:l})]}};function x(e,t){let{id:i,layout:a,values:o,validity:s,maximum:c,minimum:l,combined:m}=t;if(m){if(c||l)throw Error(`${i} combined cannot be used together with maximum or minimum`);y(i,`combined`,m,2*a.length)}else if(!c&&!l)throw Error(`${i} requires at least one output`);for(let[e,t]of[[`maximum`,c],[`minimum`,l]])t&&y(i,e,t,a.length);let h=a.width*a.height;if(o.length!==h||s.length!==h)throw Error(`${i} values and validity must contain one value per pixel`);let g=_(u),v=_(d),b=a.footprint===`bilinear`?1:0,x=m?[{name:`combinedLevels`,view:m,type:`f32`,access:`read_write`}]:[...c?[{name:`maximumLevels`,view:c,type:`f32`,access:`read_write`}]:[],...l?[{name:`minimumLevels`,view:l,type:`f32`,access:`read_write`}]:[]],S=!!(m||c),C=!!(m||l),w=m?`combinedLevels`:`maximumLevels`,T=m?`combinedLevels`:`minimumLevels`,E=m?`combinedLevelsOffset`:`maximumLevelsOffset`,D=m?`(combinedLevelsOffset + ${a.length}u)`:`minimumLevelsOffset`,O=[];for(let t of a.levels){let c=t.width*t.height,l=`const LEVEL_WIDTH: u32 = ${t.width}u;
const LEVEL_OFFSET: u32 = ${t.offset}u;
const EMPTY_MAXIMUM: f32 = ${g};
const EMPTY_MINIMUM: f32 = ${v};`,u=`${S?`${w}[${E} + LEVEL_OFFSET + index] = maximumValue;\n  `:``}${C?`${T}[${D} + LEVEL_OFFSET + index] = minimumValue;`:``}`;t.level===0&&O.push(n(e,{id:`${i}-level-0`,operation:`GPURasterExtremaPyramid`,variant:`level-0`,bindings:[{name:`values`,view:o,type:`f32`,access:`read`},{name:`validity`,view:s,type:`u32`,access:`read`},...x],invocationCount:c,declarations:`const WIDTH: u32 = ${a.width}u;
const HEIGHT: u32 = ${a.height}u;
const BLOCK_SIZE: u32 = ${t.blockSize}u;
const FOOTPRINT_EXTRA: u32 = ${b}u;
${l}
${r}`,body:`let cellColumn = index % LEVEL_WIDTH;
  let cellRow = index / LEVEL_WIDTH;
  let x0 = cellColumn * BLOCK_SIZE;
  let y0 = cellRow * BLOCK_SIZE;
  let x1 = min(x0 + BLOCK_SIZE + FOOTPRINT_EXTRA, WIDTH);
  let y1 = min(y0 + BLOCK_SIZE + FOOTPRINT_EXTRA, HEIGHT);
  var maximumValue = EMPTY_MAXIMUM;
  var minimumValue = EMPTY_MINIMUM;
  for (var y = y0; y < y1; y++) {
    for (var x = x0; x < x1; x++) {
      let pixel = y * WIDTH + x;
      if (validity[validityOffset + pixel] != 0u) {
        let value = values[valuesOffset + pixel];
        if (isFiniteValue(value)) {
          if (value > maximumValue) { maximumValue = value; }
          if (value < minimumValue) { minimumValue = value; }
        }
      }
    }
  }
  ${u}`}))}for(let t=0;t<a.levels.length-1;t+=f){let r=a.levels[t],o=a.levels.slice(t+1,t+1+f),s=Math.ceil(r.width/p),c=Math.ceil(r.height/p),l=o.map((e,t)=>{let n=p>>t,r=n>>1,i=(e,t,r)=>`for (var child = 0u; child < 4u; child++) {
      let value = ${e}[(2u * stageY + (child >> 1u)) * ${n}u + 2u * stageX + (child & 1u)];
      if (value ${t} ${r}) { ${r} = value; }
    }`,a=t===o.length-1,s=`LEVEL_OFFSET_${t}`;return`
  // Level ${e.level}: ${r} x ${r} cells per tile.
  {
    let isStageThread = localInvocationIndex < ${r*r}u;
    let stageX = localInvocationIndex % ${r}u;
    let stageY = localInvocationIndex / ${r}u;
    var maximumValue = EMPTY_MAXIMUM;
    var minimumValue = EMPTY_MINIMUM;
    if (isStageThread) {
      ${S?i(`tileMaximum`,`>`,`maximumValue`):``}
      ${C?i(`tileMinimum`,`<`,`minimumValue`):``}
    }
    workgroupBarrier();
    if (isStageThread) {
      ${S?`tileMaximum[localInvocationIndex] = maximumValue;`:``}
      ${C?`tileMinimum[localInvocationIndex] = minimumValue;`:``}
      let cellColumn = tileX * ${r}u + stageX;
      let cellRow = tileY * ${r}u + stageY;
      if (cellColumn < ${e.width}u && cellRow < ${e.height}u) {
        let cellIndex = ${s} + cellRow * ${e.width}u + cellColumn;
        ${S?`${w}[${E} + cellIndex] = maximumValue;`:``}
        ${C?`${T}[${D} + cellIndex] = minimumValue;`:``}
      }
    }
    ${a?``:`workgroupBarrier();`}
  }`});O.push(n(e,{id:`${i}-levels-${o[0].level}-${o[o.length-1].level}`,operation:`GPURasterExtremaPyramid`,variant:`reduce-fused`,workgroupSize:p*p,bindings:x,invocationCount:s*c*p*p,guardIndex:!1,declarations:`const EMPTY_MAXIMUM: f32 = ${g};
const EMPTY_MINIMUM: f32 = ${v};
const BASE_WIDTH: u32 = ${r.width}u;
const BASE_HEIGHT: u32 = ${r.height}u;
const BASE_OFFSET: u32 = ${r.offset}u;
const TILES_X: u32 = ${s}u;
${o.map((e,t)=>`const LEVEL_OFFSET_${t}: u32 = ${e.offset}u;`).join(`
`)}
${S?`var<workgroup> tileMaximum: array<f32, ${p*p}>;`:``}
${C?`var<workgroup> tileMinimum: array<f32, ${p*p}>;`:``}`,body:`let tileIndex = (index - localInvocationIndex) / ${p*p}u;
  let tileX = tileIndex % TILES_X;
  let tileY = tileIndex / TILES_X;
  let baseColumn = tileX * ${p}u + localInvocationIndex % ${p}u;
  let baseRow = tileY * ${p}u + localInvocationIndex / ${p}u;
  var baseMaximum = EMPTY_MAXIMUM;
  var baseMinimum = EMPTY_MINIMUM;
  if (baseColumn < BASE_WIDTH && baseRow < BASE_HEIGHT) {
    let baseCell = BASE_OFFSET + baseRow * BASE_WIDTH + baseColumn;
    ${S?`baseMaximum = ${w}[${E} + baseCell];`:``}
    ${C?`baseMinimum = ${T}[${D} + baseCell];`:``}
  }
  ${S?`tileMaximum[localInvocationIndex] = baseMaximum;`:`_ = baseMaximum;`}
  ${C?`tileMinimum[localInvocationIndex] = baseMinimum;`:`_ = baseMinimum;`}
  workgroupBarrier();
  ${l.join(`
`)}`}))}return O}function S(e,t=`pyramid`){let n=v(t),r=e.levels.length,i=t=>`array<u32, ${r}>(${e.levels.map(e=>`${t(e)}u`).join(`, `)})`;return`const ${n}_LEVEL_COUNT: u32 = ${r}u;
const ${n}_EMPTY_MAXIMUM: f32 = ${_(u)};
const ${n}_EMPTY_MINIMUM: f32 = ${_(d)};
const ${n}_MINIMUM_OFFSET: u32 = ${e.length}u;
const ${n}_BLOCK_SIZES = ${i(e=>e.blockSize)};
const ${n}_LEVEL_WIDTHS = ${i(e=>e.width)};
const ${n}_LEVEL_OFFSETS = ${i(e=>e.offset)};
fn ${t}LevelBlockSize(level: u32) -> u32 {
  return ${n}_BLOCK_SIZES[level];
}
fn ${t}LevelIndex(level: u32, column: u32, row: u32) -> u32 {
  let blockSize = ${n}_BLOCK_SIZES[level];
  return ${n}_LEVEL_OFFSETS[level] + (row / blockSize) * ${n}_LEVEL_WIDTHS[level] + column / blockSize;
}`}export{g as a,S as i,x as n,l as o,h as r,b as t};