// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationFormulaRef,
  GPUComputationFormulaSymbol,
  GPUComputationGraphInput
} from '@luma.gl/gpgpu';
import type {ViewPackNoFormula} from '../../registry/view-registration';

function symbol(
  id: string,
  role: GPUComputationFormulaSymbol['role'],
  binding: string
): GPUComputationFormulaSymbol {
  return {id, role, binding: `wgsl:${binding}`};
}

function formula(
  id: string,
  tex: string,
  ascii: string,
  symbols: GPUComputationFormulaSymbol[]
): GPUComputationFormulaRef {
  return {id: `gpu-core.${id}`, tex, ascii, symbols, derivation: 'curated'};
}

const input = () => symbol('x', 'input', 'inputValues');
const output = () => symbol('y', 'output', 'outputValues');

/** Mirrors gpu-scan.ts:517–594: unsigned addition, head resets, and exclusive subtraction. */
const scan = formula(
  'scan',
  String.raw`\term{y}{y_i}=\left(\sum_{j=s(i)}^{i-1+\delta}\term{x}{x_j}\right)\bmod 2^{32},\quad s(i)=\max(\{0\}\cup\{j\le i:\term{heads}{h_j}\ne0\})`,
  'y[i] = sum(x[s(i)..i-1+delta]) mod 2^32; delta=0 exclusive (default), 1 inclusive; s(i) is the last nonzero segment flag or 0.',
  [input(), output(), symbol('heads', 'input', 'segmentFlags')]
);

/** Mirrors gpu-flag-offsets.ts count write and GPUScan: raw flags are scanned, not booleanized. */
const flagOffsets = formula(
  'flag-offsets',
  String.raw`\term{offsets}{o_i}=\sum_{j<i}\term{flags}{f_j}\bmod2^{32},\quad\term{count}{c}=(o_{n-1}+f_{n-1})\bmod2^{32}`,
  'offsets[i] = sum(flags[0..i-1]) mod 2^32; count = offsets[n-1]+flags[n-1] mod 2^32, or 0 when empty. Flags are raw uint32 values.',
  [
    symbol('flags', 'input', 'flags'),
    symbol('offsets', 'output', 'offsets'),
    symbol('count', 'output', 'count')
  ]
);

/** Mirrors gpu-compaction.ts:322–366: raw scan offsets, min(flag,1), bounded scatter, last count. */
const compaction = formula(
  'compaction',
  String.raw`\term{offsets}{o_i}=\sum_{j<i}\term{flags}{f_j}\bmod2^{32},\quad\term{y}{y_{o_i}}=\term{x}{x_i}\ (f_i\ne0),\quad\term{count}{c}=o_{n-1}+\operatorname{min}(f_{n-1},1)`,
  'o[i]=exclusive uint32 scan(flags); if flags[i]!=0 and o[i] is in output capacity, y[o[i]]=x[i]. count=(o[n-1]+min(flags[n-1],1)) mod 2^32, empty=0. Dense stable compaction requires binary flags; raw nonbinary flags leave gaps.',
  [
    input(),
    output(),
    symbol('flags', 'input', 'flags'),
    symbol('offsets', 'intermediate', 'offsets'),
    symbol('count', 'output', 'outputCount')
  ]
);

/** Mirrors gpu-mask.ts:198,252–264: booleanize all operands, then write exactly zero or one. */
const mask = formula(
  'mask',
  String.raw`\term{y}{y_i}=\mathbf1[\operatorname{op}(b_{0i},\ldots,b_{mi})],\quad b_{ji}=[x_{ji}\ne0]`,
  'y[i]=0 or 1 after booleanizing inputs by !=0; op is and, or, xor (odd parity), difference (first AND NOT every remaining input), or not (one input).',
  [symbol('y', 'output', 'outputMask')]
);

/** Mirrors gpu-reduction.ts:415–450,629: masked sum filters nonfinite f32; unmasked sum propagates them. */
const reduction = formula(
  'reduction',
  String.raw`\term{y}{y}=\operatorname{op}_{i\in S}\term{x}{x_i},\quad S=\{i:\term{mask}{m_i}\ne0\},\quad\operatorname{op}\in\{\operatorname{sum},\operatorname{min},\operatorname{max},\operatorname{extent}\}`,
  'Reduce selected x by operation=sum|min|max|extent (min,max). Missing mask selects all. Min/max/extent ignore nonfinite f32; masked f32 sum also ignores them, unmasked sum does not. Empty/all-invalid min/max/extent and empty sum return 0. Integer sums wrap at 32 bits.',
  [input(), output(), symbol('mask', 'input', 'inputValidity')]
);

/** Mirrors gpu-segmented-reduction.ts:164–186: CSR half-open ranges and empty-segment zero. */
const segmentedReduction = formula(
  'segmented-reduction',
  String.raw`\term{y}{y_s}=\operatorname{op}_{\term{offsets}{o_s}\le j<o_{s+1}}\term{x}{x_j},\quad\operatorname{op}\in\{\operatorname{sum},\operatorname{min},\operatorname{max}\}`,
  'y[s]=sum|min|max of x[offsets[s]..offsets[s+1]-1], empty=0; no mask or nonfinite filter. Min/max fold from type extrema (finite f32 extrema); integer sums wrap at 32 bits.',
  [input(), output(), symbol('offsets', 'input', 'segmentOffsets')]
);

/** Mirrors gpu-gather.ts:282–301: copies row words; GPUGather passes invalidValue=0 to its shared helper. */
const gather = formula(
  'gather',
  String.raw`\term{y}{y_{i,w}}=\begin{cases}\term{x}{x_{\term{indices}{p_i},w}}&p_i<n\\0&p_i\ge n\end{cases}`,
  'Copy each word of source row indices[i] to output row i; out-of-range indices fill every word with 0 (an all-zero row).',
  [
    symbol('x', 'input', 'sourceWords'),
    symbol('y', 'output', 'outputWords'),
    symbol('indices', 'input', 'indices')
  ]
);

/** Mirrors gpu-histogram.ts:484–504,600–607,657–674: inclusive maximum; reject outside, never clamp inputs. */
const histogram = formula(
  'histogram',
  String.raw`\term{counts}{H_b}=\sum_{i\in S}\mathbf1[\operatorname{bin}(\term{x}{x_i})=b],\quad\operatorname{bin}(v)=\operatorname{min}(B-1,\lfloor B(v-a)/(z-a)\rfloor)`,
  'H[b]=count of finite, mask-selected values in bin b. Reject outside [a,z]. Uniform bins use floor(B*(v-a)/(z-a)), with z in B-1; a=z sends equal values to bin 0. Explicit strictly increasing finite edges use [e[b],e[b+1]), final bin includes last edge; invalid GPU edges yield zeros. Integer uniform binning uses exact multiply/divide floor; float uses f32 arithmetic.',
  [
    input(),
    symbol('counts', 'output', 'outputCounts'),
    symbol('mask', 'input', 'selectionMask'),
    symbol('edges', 'parameter', 'edgeValues')
  ]
);

/** Mirrors gpu-elementwise.ts:286–307: the complete expression switch, including non-fused multiply-add. */
const elementwise = formula(
  'elementwise',
  String.raw`\term{y}{y_i}=\operatorname{op}(\term{x}{a_i},\term{b}{b_i},\term{c}{c_i}),\quad\operatorname{op}\in\{\operatorname{copy},\operatorname{add},\operatorname{subtract},\operatorname{multiply},\operatorname{multiply\text{-}add},\operatorname{min},\operatorname{max}\}`,
  'copy:a; add:a+b; subtract:a-b; multiply:a*b; multiply-add:a*b+c; min:min(a,b); max:max(a,b), per element in the declared scalar type (32-bit integer wrap).',
  [input(), output(), symbol('b', 'input', 'inputBValues'), symbol('c', 'input', 'inputCValues')]
);

const gridRule =
  'cell=row*width+column; coordinate(v,a,z,size)=min(size-1,trunc((v-a)/(z-a)*size)), a=z or v=a gives 0, v=z gives size-1. Cross-zero bounds scale v,a,z by max(abs(a),abs(z)) before division. Ignore nonfinite/out-of-bounds points; bounds are inclusive and must be finite and ordered.';
/** Mirrors gpu-grid-binning.ts:319–340 and gpu-spatial-utils.ts:23–37 (overflow-safe coordinates). */
const gridBinning = formula(
  'grid-binning',
  String.raw`\term{counts}{H_c}=\sum_i\mathbf1[\operatorname{cell}(\term{positions}{p_i})=c],\quad c=rW+q`,
  `H[c]=count of accepted points in cell c; counts clear each run. ${gridRule}`,
  [symbol('positions', 'input', 'positions'), symbol('counts', 'output', 'outputCounts')]
);

/** Mirrors gpu-grid-aggregation.ts:374–390,452–468: finite weights and NaN empty means/extrema. */
const gridAggregation = formula(
  'grid-aggregation',
  String.raw`S_c=\{i:\operatorname{cell}(\term{positions}{p_i})=c\},\quad\term{y}{y_c}=\operatorname{op}_{i\in S_c}\term{weights}{w_i},\quad\term{counts}{n_c}=|S_c|`,
  `Per-cell sum|min|max|mean of finite weights; optional counts count accepted weights. Empty sum=0, empty min/max/mean=NaN. mean=sum/count. ${gridRule}`,
  [
    symbol('positions', 'input', 'positions'),
    symbol('weights', 'input', 'weights'),
    output(),
    symbol('counts', 'output', 'outputCounts')
  ]
);

/** Mirrors gpu-group-aggregation.ts:400–401,540–544,594–605: dense keys, mask, finite-value filtering. */
const groupAggregation = formula(
  'group-aggregation',
  String.raw`S_g=\{i:\term{keys}{k_i}=g,\term{mask}{m_i}\ne0\},\quad\term{y}{y_g}=\operatorname{op}_{i\in S_g}\term{x}{x_i},\quad\term{counts}{n_g}=|S_g|`,
  'Dense groups g<groupCount; ignore out-of-range keys and zero mask (missing mask selects all). count counts rows without values; sum|min|max|mean accept only finite f32 values and optional counts count those values. sum[g]=sum(x), mean=sum/count; empty count/sum=0, empty min/max/mean=NaN.',
  [
    input(),
    output(),
    symbol('keys', 'input', 'groupKeys'),
    symbol('mask', 'input', 'selectionMask'),
    symbol('counts', 'output', 'outputCounts')
  ]
);

/** Mirrors gpu-matvec.ts:113–131: row-major dense indexing and chunk accumulation. */
const matvec = formula(
  'mat-vec',
  String.raw`\term{y}{y_r}=\sum_{j=0}^{C-1}\term{matrix}{A_{rC+j}}\term{vector}{x_j}`,
  'y[r]=sum(j=0..columns-1, A[r*columns+j]*x[j]); row-major float32, partials accumulate across chunks.',
  [output(), symbol('matrix', 'input', 'matrixValues'), symbol('vector', 'input', 'vectorValues')]
);

/** Mirrors gpu-matmul.ts:147–175: row-major left, right, and result tiles. */
const matmul = formula(
  'mat-mul',
  String.raw`\term{y}{C_{rN+c}}=\sum_{j=0}^{K-1}\term{left}{A_{rK+j}}\term{right}{B_{jN+c}}`,
  'C[r*N+c]=sum(j=0..K-1, A[r*K+j]*B[j*N+c]); row-major float32, tiled partial sums across chunks.',
  [output(), symbol('left', 'input', 'leftValues'), symbol('right', 'input', 'rightValues')]
);

/** Mirrors gpu-convolution.ts:887–894,924–933: subtract kernel displacement (convolution, not correlation). */
const convolution = formula(
  'convolution',
  String.raw`\term{y}{y_{r,c}}=\sum_{u,v}\term{kernel}{K_{u,v}}\term{x}{\widetilde{x}_{r-u+\lfloor H_K/2\rfloor,c-v+\lfloor W_K/2\rfloor}}`,
  'y[r,c]=sum(K[u,v]*extendedX[r-u+floor(kernelHeight/2),c-v+floor(kernelWidth/2)]); true convolution. Boundary zero skips outside samples; wrap uses positive modulo height/width. Row-major float32; FFT strategy computes the same orientation.',
  [input(), output(), symbol('kernel', 'input', 'kernelValues')]
);

/** Mirrors gpu-fft1d.ts:554–556,627–640: forward negative sign, inverse positive and final 1/N. */
const fft1d = formula(
  'fft1d',
  String.raw`\term{y}{Y_k}=a\sum_{j=0}^{N-1}\term{x}{x_j}e^{s2\pi\mathrm{i}jk/N},\quad(s,a)=(-1,1)\ \text{forward};\ (+1,1/N)\ \text{inverse}`,
  'Y[k]=a*sum(x[j]*exp(s*2*pi*i*j*k/N)); forward s=-1,a=1; inverse s=+1,a=1/N applied once on the final pass, independently per batch. Complex vec2f input/output.',
  [input(), output()]
);

/** Mirrors gpu-fft2d-shaders.ts:35–37,67–85 and gpu-fft2d.ts:213–216 final normalization. */
const fft2d = formula(
  'fft2d',
  String.raw`\term{y}{Y_{v,u}}=a\sum_{r=0}^{H-1}\sum_{c=0}^{W-1}\term{x}{x_{rW+c}}e^{s2\pi\mathrm{i}(uc/W+vr/H)}`,
  'Y[v,u]=a*sum(x[r*W+c]*exp(s*2*pi*i*(u*c/W+v*r/H))); forward s=-1,a=1; inverse s=+1,a=1/(W*H), once on final pass per batch. Row-major complex vec2f.',
  [
    input(),
    output(),
    symbol('sign', 'parameter', 'directionSign'),
    symbol('normalization', 'parameter', 'normalizationScale')
  ]
);

/**
 * Mirrors gpu-conjugate-gradient-program.ts:77–197; lowered vector shader gpu-elementwise-scalar.ts:99–101
 * and scalar arena gpu-scalar.ts:getGPUScalarWGSLStore. Scalars share gpuValues, not invented alpha WGSL locals.
 */
const conjugateGradient = formula(
  'conjugate-gradient',
  String.raw`r_0=b-Ax_0,\ p_0=r_0,\ q=Ap,\quad\term{alpha}{\alpha}=\frac{r^Tr}{p^Tq},\quad\term{solution}{x'}=x+\alpha p,\quad r'=r-\alpha q,\quad\term{beta}{\beta}=\frac{r'^Tr'}{r^Tr},\quad p'=r'+\beta p`,
  'r0=b-A*x0; p0=r0; rr=r dot r. While rr>toleranceSquared, at most maxIterations: q=A*p; pq=p dot q; breakdown=(pq<=0), active=(pq>0); alpha=rr/pq; x+=alpha*p; r-=alpha*q; newRR=r dot r; beta=newRR/rr; p=r+beta*p; rr=newRR; active=(rr>toleranceSquared). Active gates subsequent lowered operations; no preconditioner and no relative tolerance.',
  [
    symbol('alpha', 'intermediate', 'gpuValues'),
    symbol('beta', 'intermediate', 'gpuValues'),
    symbol('solution', 'output', 'outputValues')
  ]
);

/** Curated kernel formulas, or a procedural explanation directing the reader to the graph. */
export const GPU_CORE_FORMULAS: Record<string, GPUComputationFormulaRef | ViewPackNoFormula> = {
  GPUScan: scan,
  GPUFlagOffsets: flagOffsets,
  GPUCompaction: compaction,
  GPUMask: mask,
  GPUReduction: reduction,
  GPUSegmentedReduction: segmentedReduction,
  GPUSort: {
    formula: 'none',
    reason:
      'Procedural radix, bitonic, and chunk merge passes reorder keys; SkeletonView/graph shows the steps.'
  },
  GPUSegmentedSort: {
    formula: 'none',
    reason:
      'Procedural per-segment sorting passes reorder keys; SkeletonView/graph shows the steps.'
  },
  GPUGather: gather,
  GPUHistogram: histogram,
  GPUElementwise: elementwise,
  GPUGridIndex: {
    formula: 'none',
    reason:
      'Procedural count, scan, and scatter build cell ranges; SkeletonView/graph shows the steps.'
  },
  GPUGridBinning: gridBinning,
  GPUGridAggregation: gridAggregation,
  GPUGroupAggregation: groupAggregation,
  GPUHashIndex: {
    formula: 'none',
    reason:
      'Procedural bounded probing resolves collisions and publishes slots; SkeletonView/graph shows the steps.'
  },
  GPUHashJoin: {
    formula: 'none',
    reason:
      'Procedural probing, flag scanning, and pair scattering join rows; SkeletonView/graph shows the steps.'
  },
  GPUMatVec: matvec,
  GPUMatMul: matmul,
  GPUConvolution: convolution,
  GPUFFT1D: fft1d,
  GPUFFT2D: fft2d,
  createGPUConjugateGradientProgram: conjugateGradient
};

/**
 * Curated GPU Core formula refs keyed by formula id (`gpu-core.histogram`), the id kernels put in
 * their node annotations. Records of other entries that compose core primitives merge these so
 * that every node formula id resolves (see {@link mergeGPUCoreFormulaRefs}).
 */
export const GPU_CORE_FORMULA_REFS: Readonly<Record<string, GPUComputationFormulaRef>> =
  Object.fromEntries(
    Object.values(GPU_CORE_FORMULAS)
      .filter((entry): entry is GPUComputationFormulaRef => 'tex' in entry)
      .map(entry => [entry.id, entry])
  );

/**
 * Returns `formulas` plus the GPU Core refs for every id in `formulaIds` that `formulas` lacks.
 * Ids that are neither local nor core stay unresolved; callers decide how to handle them.
 */
export function mergeGPUCoreFormulaRefs(
  formulaIds: Iterable<string>,
  formulas: readonly GPUComputationFormulaRef[] = []
): GPUComputationFormulaRef[] {
  const merged = [...formulas];
  const known = new Set(formulas.map(formula => formula.id));
  for (const id of formulaIds) {
    const core = GPU_CORE_FORMULA_REFS[id];
    if (core && !known.has(id)) {
      merged.push(core);
      known.add(id);
    }
  }
  return merged;
}

/**
 * Returns `formulas` plus the GPU Core refs for every node `formula` id in `graphs` that `formulas`
 * lacks. Lens harnesses call this when assembling a record, because composed core primitives
 * (scan, compaction, reduction, …) annotate their nodes with `gpu-core.*` formula ids.
 */
export function mergeGPUCoreFormulaRefsForGraphs(
  graphs: readonly GPUComputationGraphInput[],
  formulas: readonly GPUComputationFormulaRef[] = []
): GPUComputationFormulaRef[] {
  const formulaIds = new Set<string>();
  for (const input of graphs) {
    const graph = 'graph' in input ? input.graph : input;
    for (const node of graph.nodes) if (node.formula !== undefined) formulaIds.add(node.formula);
  }
  return mergeGPUCoreFormulaRefs(formulaIds, formulas);
}
