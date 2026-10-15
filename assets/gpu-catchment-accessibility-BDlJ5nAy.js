import{_ as e}from"./analysis-frame-Blu0tYdw.js";import{_ as t,h as n,w as r,y as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{s as o}from"./wgsl-common-BNRVzgd8.js";import{c as s}from"./spatial-weights-D4bd9AQ6.js";var c=`GPUCatchmentAccessibility`,l=class{id;props;facilityCount;demandCount;constructor(t){let r=t.id??`catchment-accessibility`;if(this.id=r,this.props=t,t.method!==`2sfca`&&t.method!==`3sfca`)throw Error(`${r} method must be '2sfca' or '3sfca'`);if(this.facilityCount=s(r,t.facilityWeights,`facilityWeights`),this.demandCount=t.demandWeights?s(r,t.demandWeights,`demandWeights`):this.facilityCount,!t.demandWeights&&t.supply.length!==t.demand.length)throw Error(`${r} needs demandWeights unless supply and demand have equal length`);t.demandWeights?.spatialContext&&t.facilityWeights?.spatialContext&&e(r,t.demandWeights.spatialContext,t.facilityWeights.spatialContext);let{output:a}=t;for(let[e,n]of[[`supply`,t.supply],[`demand`,t.demand],[`output.accessibility`,a.accessibility],[`output.ratios`,a.ratios]])n&&i(n,[`float32`],`${r} ${e}`);if(a.reachableFacilities&&i(a.reachableFacilities,[`uint32`],`${r} output.reachableFacilities`),t.supply.length!==this.facilityCount)throw Error(`${r} supply length must equal the facilityWeights row count`);if(t.demand.length!==this.demandCount)throw Error(`${r} demand length must equal the demandWeights row count`);if(a.accessibility.length!==this.demandCount)throw Error(`${r} output.accessibility length must equal the demand count`);if(a.ratios&&a.ratios.length!==this.facilityCount)throw Error(`${r} output.ratios length must equal the facility count`);if(a.reachableFacilities&&a.reachableFacilities.length!==this.demandCount)throw Error(`${r} output.reachableFacilities length must equal the demand count`);let o=u(t),c=[a.accessibility,a.ratios,a.reachableFacilities];n(r,c,[t.supply,t.demand,...o]);let l=c.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(l).size!==l.length)throw Error(`${r} outputs must use separate buffers`)}getCommandNodes(e){let{id:n,props:i,facilityCount:s,demandCount:l}=this,{supply:d,demand:f,facilityWeights:p,output:m}=i,{accessibility:h}=m,g=i.demandWeights??p;t(n,e,[d,f,h,m.ratios,m.reachableFacilities,...u(i)]);let _=i.method===`3sfca`,v=m.ratios??r(e,`${n}-ratios`,`float32`,s),y=_?r(e,`${n}-selection`,`float32`,l):void 0,b=[];y&&b.push(a(e,{id:`${n}-selection`,operation:c,variant:`selection`,bindings:[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read`},{name:`weights`,view:g.weights,type:`f32`,access:`read`},{name:`selection`,view:y,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`const PARTNERS: u32 = ${s}u;`,body:`var total = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      total += weights[weightsOffset + slot];
    }
  }
  selection[selectionOffset + index] = total;`}));let x=y?[{name:`selection`,view:y,type:`f32`,access:`read`}]:[];return b.push(a(e,{id:`${n}-ratios`,operation:c,variant:`ratios`,bindings:[{name:`offsets`,view:p.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:p.neighbors,type:`u32`,access:`read`},{name:`weights`,view:p.weights,type:`f32`,access:`read`},{name:`demand`,view:f,type:`f32`,access:`read`},{name:`supply`,view:d,type:`f32`,access:`read`},...x,{name:`ratios`,view:v,type:`f32`,access:`read_write`}],invocationCount:s,declarations:`const PARTNERS: u32 = ${l}u;
${o}`,body:`var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and captures nothing.
      if (isFiniteBits(partnerDemand)) {
        ${_?`let total = selection[selectionOffset + partner];
        if (total > 0.0) {
          // Wan, Zou and Sternberg (2012): selection weight G = w / T times the decay w.
          let weight = weights[weightsOffset + slot];
          captured += partnerDemand * weight * weight / total;
        }`:`captured += partnerDemand * weights[weightsOffset + slot];`}
      }
    }
  }
  ratios[ratiosOffset + index] = select(0.0, supply[supplyOffset + index] / captured, captured > 0.0);`}),a(e,{id:`${n}-scores`,operation:c,variant:`scores`,record:_?void 0:{formula:`gpu-spatial-analysis.catchment-accessibility`},bindings:[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read`},{name:`weights`,view:g.weights,type:`f32`,access:`read`},{name:`ratios`,view:v,type:`f32`,access:`read`},...x,{name:`accessibility`,view:h,type:`f32`,access:`read_write`}],invocationCount:l,declarations:`const PARTNERS: u32 = ${s}u;
${o}`,body:`var score = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let ratio = ratios[ratiosOffset + partner];
      // A non-finite ratio (a non-finite supply) is missing and adds nothing.
      if (isFiniteBits(ratio)) {
        let weight = weights[weightsOffset + slot];
        score += ${_?`weight * weight`:`weight`} * ratio;
      }
    }
  }
  ${_?`let total = selection[selectionOffset + index];
  score = select(0.0, score / total, total > 0.0);`:``}
  accessibility[accessibilityOffset + index] = score;`})),m.reachableFacilities&&b.push(a(e,{id:`${n}-reachable`,operation:c,variant:`reachable`,bindings:[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read`},{name:`reachable`,view:m.reachableFacilities,type:`u32`,access:`read_write`}],invocationCount:l,declarations:`const PARTNERS: u32 = ${s}u;`,body:`var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      count++;
    }
  }
  reachable[reachableOffset + index] = count;`})),b}};function u(e){let t=[];for(let n of[e.facilityWeights,e.demandWeights])n&&t.push(n.offsets,n.neighbors,n.weights);return t}export{l as t};