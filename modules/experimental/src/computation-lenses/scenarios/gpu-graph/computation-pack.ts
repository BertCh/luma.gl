// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {
  ViewPackData,
  ViewPackNoFormula
} from '../../../computation-views/registry/view-registration';
import {registerDefaultViewPack} from '../../../computation-views/registry/default-registry';
import {GPU_GRAPH_STRUCTURE_PACK} from './pack-structure';
import {COMMUNITIES_LAYOUTS_PACK} from './pack-communities-layouts';

/** Domain shared by the rail graph scenarios and their views. */
export const GPU_GRAPH_COMPUTATION_PACK_ID = 'gpu-graph';

/** Kernel-bound formulas keyed by exported contributor name; extend one element at a time. */
export const GPU_GRAPH_FORMULAS: Readonly<
  Record<string, GPUComputationFormulaRef | ViewPackNoFormula>
> = {
  ...GPU_GRAPH_STRUCTURE_PACK.formulas,
  ...COMMUNITIES_LAYOUTS_PACK.formulas,
  GPUGraphPageRank: {
    id: 'gpu-graph.page-rank-update',
    derivation: 'curated',
    tex: String.raw`\term{r}{r}^{(k+1)}_v=\frac{\term{provisional}{\widetilde r}_v}{\term{sum}{S}},\quad
\term{provisional}{\widetilde r}_v=\frac{1-\term{damping}{d}}{\term{count}{N}}+\term{damping}{d}\Big(\sum_{u\in\term{incoming}{N^-(v)}}\frac{\term{r}{r}^{(k)}_u}{\term{degree}{k^+_u}}+\frac{\term{dangling}{D}^{(k)}}{\term{count}{N}}\Big),\quad
\term{dangling}{D}^{(k)}=\sum_{\term{degree}{k^+_u}=0}\term{r}{r}^{(k)}_u,\quad
\term{sum}{S}=\sum_u\term{provisional}{\widetilde r}_u`,
    ascii:
      'Initialize r=1/max(N,1). Each round: D=sum(r[u] for outDegree[u]=0); provisional[v]=(1-d)/N+d*(sum(r[u]/outDegree[u] over incoming edges)+D/N); S=sum(provisional); r[v]=provisional[v]/S. Overflow publishes zero; normalization also publishes zero unless S>0, S=S and abs(S)<=3.402823466e38. Optional final residual=sum(abs(new-old)), or zero on overflow/invalid S. Float32 reductions; edge multiplicity is preserved, weights are ignored.',
    symbols: [
      {id: 'r', role: 'output', label: 'normalized rank', binding: 'wgsl:output'},
      {id: 'count', role: 'constant', label: 'vertex count', binding: 'wgsl:VERTEX_COUNT'},
      {id: 'damping', role: 'parameter', label: 'damping', binding: 'wgsl:DAMPING'},
      {id: 'degree', role: 'input', label: 'outgoing degree', binding: 'wgsl:degree'},
      {
        id: 'incoming',
        role: 'input',
        label: 'incoming edge neighbors',
        binding: 'wgsl:incomingNeighbors'
      },
      {id: 'dangling', role: 'intermediate', label: 'dangling mass', binding: 'wgsl:danglingMass'},
      {
        id: 'provisional',
        role: 'intermediate',
        label: 'unnormalized rank',
        binding: 'wgsl:workspace'
      },
      {id: 'sum', role: 'intermediate', label: 'rank sum', binding: 'wgsl:rankSum'}
    ]
  }
};

/** Plain view metadata; importing this pack does not load a GPU scenario. */
export const GPU_GRAPH_COMPUTATION_PACK: ViewPackData = {
  id: GPU_GRAPH_COMPUTATION_PACK_ID,
  operators: {
    ...GPU_GRAPH_STRUCTURE_PACK.operators,
    ...COMMUNITIES_LAYOUTS_PACK.operators,
    GPUGraphPageRank: {scope: 'focal', scopeSecondary: 'global', reduction: 'sum'}
  },
  formulas: GPU_GRAPH_FORMULAS,
  analysisStages: ['topology', 'parameters', 'iteration', 'publication'],
  captureStages: {
    ...GPU_GRAPH_STRUCTURE_PACK.captureStages,
    ...COMMUNITIES_LAYOUTS_PACK.captureStages,
    'csr-offsets': 'topology',
    'csr-neighbors': 'topology',
    'station-positions': 'topology',
    'page-rank': 'iteration',
    'page-rank-stations': 'publication',
    'page-rank-residual': 'publication'
  },
  variableKinds: {
    ...GPU_GRAPH_STRUCTURE_PACK.variableKinds,
    ...COMMUNITIES_LAYOUTS_PACK.variableKinds,
    'page-rank': {level: 'ratio', extent: 'intensive'}
  },
  neighbourhoods: {vertex: {kind: 'csr-star', relation: 'adjacency'}},
  metadata: {coordinateModel: 'geographic', weighted: false, dataset: 'g-nl-rail-graph'}
};

registerDefaultViewPack({...GPU_GRAPH_COMPUTATION_PACK, views: []});
