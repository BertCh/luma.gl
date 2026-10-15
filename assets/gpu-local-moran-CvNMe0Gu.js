import{M as e,_ as t,h as n,j as r,w as i}from"./resources-CwvhEOBe.js";import{s as a}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{o,s}from"./spatial-weights-D4bd9AQ6.js";import{r as c,t as l}from"./gpu-multiple-testing-adjustment-C7k5NYt0.js";import{a as u,f as d,i as f,n as p,o as m,r as h,s as g,t as _,u as v}from"./spatial-autocorrelation-kernels-BgrN4KSs.js";var y=`GPULocalMoran`,b=class{id;props;multipleTesting;islands;minimumNeighbors;constructor(t){this.id=t.id??`local-moran`,this.props=t;let i=this.id;this.islands=s(i,t.islands),this.minimumNeighbors=m(i,t.minimumNeighbors),this.multipleTesting=c(i,t.multipleTesting);let a=g({...t,id:i,globalStatistics:t.output.globalStatistics});for(let[n,r]of[[`zScores`,t.output.zScores],[`localI`,t.output.localI],[`spatialLag`,t.output.spatialLag],[`pValues`,t.output.pValues],[`adjustedPValues`,t.output.adjustedPValues]])r&&e(r,[`float32`],`${i} ${n}`);for(let[e,n]of[[`quadrants`,t.output.quadrants],[`neighborCounts`,t.output.neighborCounts]])n&&r(n,`${i} ${e}`);for(let[e,n]of[[`zScores`,t.output.zScores],[`localI`,t.output.localI],[`spatialLag`,t.output.spatialLag],[`quadrants`,t.output.quadrants],[`pValues`,t.output.pValues],[`adjustedPValues`,t.output.adjustedPValues],[`neighborCounts`,t.output.neighborCounts]])if(n&&n.length!==a)throw Error(`${i} ${e} length must equal the weights row count`);if(t.quadrantGating===`none`&&this.multipleTesting!==`none`)throw Error(`${i} quadrantGating 'none' cannot be combined with multipleTesting`);n(i,[t.output.zScores,t.output.localI,t.output.spatialLag,t.output.quadrants,t.output.pValues,t.output.adjustedPValues,t.output.neighborCounts,t.output.globalStatistics],[t.weights.offsets,t.weights.neighbors,t.weights.weights,t.weights.status?.overflow,t.values,t.parameters,t.mask])}getInferenceMetadata(e={}){d(e);let t=e.significanceLevel??.05;return{alternative:`two-sided`,permutationCount:0,multipleTesting:this.multipleTesting,significanceLevel:t,testedRowsDefinition:`included rows with at least ${this.minimumNeighbors} active neighbor${this.minimumNeighbors===1?``:`s`}${this.islands===`exclude`?` (islands excluded)`:``}, a nonzero centered value and a defined z-score`}}getCommandNodes(e){let{id:n,props:r}=this,{weights:s,parameters:c}=r,{zScores:d}=r.output;t(n,e,[s.offsets,s.neighbors,s.weights,s.status?.overflow,r.values,c,r.mask,d,r.output.localI,r.output.spatialLag,r.output.quadrants,r.output.pValues,r.output.adjustedPValues,r.output.neighborCounts,r.output.globalStatistics]);let m=r.values.length,g=h(e,{...r,globalStatistics:r.output.globalStatistics,id:n,islands:this.islands,operation:y}),b=g.nodes,x=`${p(m,s.neighbors.length)}
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${o(s)};`,S=`${_}
${x}`,C=`${f()}
${x}`,{HIGH_HIGH:w,LOW_HIGH:T,LOW_LOW:E,HIGH_LOW:D,NOT_SIGNIFICANT:O}=v,k=this.multipleTesting!==`none`,A=!!(r.output.quadrants||r.output.pValues||r.output.adjustedPValues),j=k&&!!(r.output.adjustedPValues||r.output.quadrants)&&A,M=j?r.output.pValues??i(e,`${n}-p-values`,`float32`,m):void 0,N=j?r.output.adjustedPValues??i(e,`${n}-adjusted-p-values`,`float32`,m):void 0,P=e=>A&&!e&&(!j||!!r.output.quadrants),F=e.device.limits.maxStorageBuffersPerShaderStage,I=(e,t)=>5+(r.output.spatialLag||r.output.localI&&!e||P(t)?1:0)+(r.output.neighborCounts?1:0)+(r.output.localI&&e?1:0)+(j?1:0)+(t?1+(r.output.quadrants?1:0)+(r.output.pValues?1:0)+(r.output.adjustedPValues?1:0):0),L=A&&!j,R=!!r.output.localI,z=L;I(R,z)>F&&(R=!1),I(R,z)>F&&(z=!1,R=!!r.output.localI&&I(!0,!1)<=F);let B=r.output.spatialLag||r.output.localI&&!R||P(z)?r.output.spatialLag??i(e,`${n}-spatial-lag`,`float32`,m):void 0,V=(e,t,n)=>`
  var quadrant = ${O}u;
  if (${n}) {
    if (${e} > 0.0 && ${t} > 0.0) {
      quadrant = ${w}u;
    } else if (${e} < 0.0 && ${t} > 0.0) {
      quadrant = ${T}u;
    } else if (${e} < 0.0 && ${t} < 0.0) {
      quadrant = ${E}u;
    } else if (${e} > 0.0 && ${t} < 0.0) {
      quadrant = ${D}u;
    }
  }
  quadrants[quadrantsOffset + index] = quadrant;`,H=(e,t,n)=>`let pValue = select(getQuietNaN(index), getTwoSidedPValue(${e}), isFiniteFloat(${e}));
  ${r.output.quadrants?V(t,n,r.quadrantGating===`none`?`isFiniteFloat(${t})`:`isFiniteFloat(${e}) && pValue <= readParameter(0u)`):``}
  ${r.output.pValues?`pValues[pValuesOffset + index] = pValue;`:``}
  ${r.output.adjustedPValues?`adjustedPValues[adjustedPValuesOffset + index] = pValue;`:``}`,U=[{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read`},{name:`weights`,view:s.weights,type:`f32`,access:`read`},{name:`statistics`,view:g.statistics,type:`f32`,access:`read`},{name:`zScores`,view:d,type:`f32`,access:`read_write`}];if(B&&U.push({name:`spatialLag`,view:B,type:`f32`,access:`read_write`}),r.output.neighborCounts&&U.push({name:`neighborCounts`,view:r.output.neighborCounts,type:`u32`,access:`read_write`}),r.output.localI&&R&&U.push({name:`localI`,view:r.output.localI,type:`f32`,access:`read_write`}),M&&U.push({name:`pValues`,view:M,type:`f32`,access:`read_write`}),z&&(U.push({name:`parameters`,view:c,type:`f32`,access:`read`}),r.output.quadrants&&U.push({name:`quadrants`,view:r.output.quadrants,type:`u32`,access:`read_write`}),r.output.pValues&&U.push({name:`pValues`,view:r.output.pValues,type:`f32`,access:`read_write`}),r.output.adjustedPValues&&U.push({name:`adjustedPValues`,view:r.output.adjustedPValues,type:`f32`,access:`read_write`})),b.push(a(e,{id:`${n}-neighbors`,operation:y,variant:`local-moran${r.output.localI&&R?`-local-i`:``}${z?`-classify`:``}${j?`-p-value`:``}`,bindings:U,...e.recording&&{record:{formula:`local-moran`}},invocationCount:m,declarations:z?C:S,body:`let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var lag = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var neighborSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${u(`neighborSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`)}
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      neighborSum = neighborSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    lag = neighborSum;
    let others = statistics[statisticsOffset + MOMENTS] - 1.0;
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    let otherMean = -centered / others;
    let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
    let lagVariance = otherVariance * (others * squareSum - weightSum * weightSum) / (others - 1.0);
    if (others >= 2.0 && centered != 0.0 && neighborCount >= MINIMUM_NEIGHBORS &&
        lagVariance > 0.0 && isFiniteFloat(lagVariance)) {
      let lagZ = (neighborSum - weightSum * otherMean) / sqrt(lagVariance);
      zScore = select(lagZ, -lagZ, centered < 0.0);
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${B?`spatialLag[spatialLagOffset + index] = lag;`:``}
  ${r.output.neighborCounts?`neighborCounts[neighborCountsOffset + index] = neighborCount;`:``}
  ${M?`pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));`:``}
  ${r.output.localI&&R?`{
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
    let scale = select(getQuietNaN(index), (statistics[statisticsOffset + MOMENTS] - 1.0) / sumOfSquares, sumOfSquares > 0.0);
    // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
    localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));
  }`:``}
  ${z?H(`zScore`,`centered`,`lag`):``}`})),r.output.localI&&!R&&b.push(a(e,{id:`${n}-local-i`,operation:y,variant:`local-i`,bindings:[{name:`statistics`,view:g.statistics,type:`f32`,access:`read`},{name:`spatialLag`,view:B,type:`f32`,access:`read`},{name:`localI`,view:r.output.localI,type:`f32`,access:`read_write`}],invocationCount:m,declarations:S,body:`let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
  // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
  let scale = select(getQuietNaN(index), (statistics[statisticsOffset + MOMENTS] - 1.0) / sumOfSquares, sumOfSquares > 0.0);
  // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
  localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));`})),j&&b.push(...new l({id:`${n}-multiple-testing`,pValues:M,method:this.multipleTesting,output:{adjustedPValues:N}}).getCommandNodes(e)),!P(z))return b;let W=[{name:`parameters`,view:c,type:`f32`,access:`read`},{name:`statistics`,view:g.statistics,type:`f32`,access:`read`},{name:`zScores`,view:d,type:`f32`,access:`read`},{name:`spatialLag`,view:B,type:`f32`,access:`read`}];return j&&W.push({name:`adjustedPValues`,view:N,type:`f32`,access:`read`}),r.output.quadrants&&W.push({name:`quadrants`,view:r.output.quadrants,type:`u32`,access:`read_write`}),!j&&r.output.pValues&&W.push({name:`pValues`,view:r.output.pValues,type:`f32`,access:`read_write`}),!j&&r.output.adjustedPValues&&W.push({name:`adjustedPValues`,view:r.output.adjustedPValues,type:`f32`,access:`read_write`}),b.push(a(e,{id:`${n}-classify`,operation:y,variant:j?`classify-adjusted`:`classify`,bindings:W,invocationCount:m,declarations:f(),body:`let zScore = zScores[zScoresOffset + index];
  let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  ${j?`let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  ${V(`centered`,`lag`,`isFiniteFloat(zScore) && isFiniteFloat(adjusted) && adjusted <= readParameter(0u)`)}`:H(`zScore`,`centered`,`lag`)}`})),b}};export{b as t};