// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {
  ViewPackData,
  ViewPackNoFormula
} from '../../../computation-views/registry/view-registration';

/** Curated formulas for the community-detection and layout contributors, keyed by class name. */
const formulas: Record<string, GPUComputationFormulaRef | ViewPackNoFormula> = {
  GPUGraphConnectedComponents: {
    id: 'gpu-graph.connected-components-hook-jump',
    derivation: 'curated',
    tex: String.raw`\term{label}{c}\big[\max(\term{label}{c}_v,\term{label}{c}_u)\big]\leftarrow\min\big(\term{label}{c}_v,\term{label}{c}_u\big)\ \ \forall u\in\term{neighbors}{N(v)},\qquad
\term{label}{c}_v\leftarrow\min\big(\term{label}{c}_v,\ \term{label}{c}\big[\term{label}{c}_v\big]\big)`,
    ascii:
      'Initialize c[v]=v. Each iteration, hooking: for every vertex v and forward neighbor u, lower=min(c[v],c[u]), higher=max(c[v],c[u]); if they differ, c[higher]=atomicMin(c[higher], lower). Then pointer jumping: parentComponent=c[c[v]]; if smaller, c[v]=atomicMin(c[v], parentComponent). Directed edges join both endpoints. Labels converge to the lowest vertex id of each component; forward-adjacency overflow publishes 0xffffffff.',
    symbols: [
      {id: 'label', role: 'output', label: 'component label', binding: 'wgsl:output'},
      {id: 'neighbors', role: 'input', label: 'CSR neighbors', binding: 'wgsl:neighbors'}
    ]
  },
  GPUGraphLabelPropagation: {
    id: 'gpu-graph.label-propagation-vote',
    derivation: 'curated',
    tex: String.raw`\term{label}{\ell}^{(k+1)}_v=\operatorname*{arg\,max}_{c\in\{\term{label}{\ell}^{(k)}_v\}\cup\{\term{label}{\ell}^{(k)}_u:\,u\in\term{neighbors}{N(v)}\}}\term{votes}{\mathrm{votes}}_v(c),\quad
\term{votes}{\mathrm{votes}}_v(c)=[\term{label}{\ell}^{(k)}_v=c]+\sum_{u\in\term{neighbors}{N(v)},\,u\ne v}[\term{label}{\ell}^{(k)}_u=c]`,
    ascii:
      'Labels start at the vertex id. Each synchronous round, every vertex counts votes(c) for each candidate label c: one self vote plus one vote per weak neighbor (self-loops skipped; duplicate edges vote separately). It takes the label with most votes, the lowest label on ties, written to a scratch column and published after the round. Directed graphs also vote through reverse CSR. Overflow publishes 0xffffffff.',
    symbols: [
      {id: 'label', role: 'output', label: 'community label', binding: 'wgsl:output'},
      {id: 'neighbors', role: 'input', label: 'CSR neighbors', binding: 'wgsl:forwardNeighbors'},
      {id: 'votes', role: 'intermediate', label: 'votes for a label', binding: 'wgsl:votes'}
    ]
  },
  GPUGraphModularityOptimization: {
    id: 'gpu-graph.modularity-optimization-move-gain',
    derivation: 'curated',
    tex: String.raw`\term{gain}{\Delta Q}_{v\to c'}=\frac{(\term{outgoingWeight}{w_{v,c'}}-\term{currentOutgoingWeight}{w_{v,c}})+(\term{incomingWeight}{w_{c',v}}-\term{currentIncomingWeight}{w_{c,v}})}{\term{total}{T}}
-\term{resolution}{\gamma}\Big[\frac{\term{vertexStatistics}{k^{\mathrm{in}}_v}}{T}\,\frac{\term{candidateStatistics}{\Sigma^{\mathrm{out}}_{c'}}-(\term{currentStatistics}{\Sigma^{\mathrm{out}}_{c}}-k^{\mathrm{out}}_v)}{T}
+\frac{k^{\mathrm{out}}_v}{T}\,\frac{\Sigma^{\mathrm{in}}_{c'}-(\Sigma^{\mathrm{in}}_{c}-k^{\mathrm{in}}_v)}{T}\Big]`,
    ascii:
      'Each round every vertex v scores moving from its community c to each neighboring community c2 (or one unused singleton): gain = ((w(v,c2)-w(v,c)) out + in)/T - resolution*(kin_v/T*(Sout_c2-(Sout_c-kout_v))/T + kout_v/T*(Sin_c2-(Sin_c-kin_v))/T), T = total edge weight (doubled for undirected), in = out for undirected graphs. Only moves with gain > minimumGain count; the single largest gain in the graph (lowest vertex id on ties) is applied, one move per round. Overflow, invalid labels or weights publish 0xffffffff communities and zero modularity. The final score is the GPUGraphModularity pass.',
    symbols: [
      {id: 'gain', role: 'intermediate', label: 'modularity gain of a move', binding: 'wgsl:gain'},
      {
        id: 'outgoingWeight',
        role: 'input',
        label: 'weight from v into candidate community',
        binding: 'wgsl:outgoingWeight'
      },
      {
        id: 'currentOutgoingWeight',
        role: 'input',
        label: 'weight from v into its own community',
        binding: 'wgsl:currentOutgoingWeight'
      },
      {
        id: 'incomingWeight',
        role: 'input',
        label: 'weight from candidate community into v',
        binding: 'wgsl:incomingWeight'
      },
      {
        id: 'currentIncomingWeight',
        role: 'input',
        label: 'weight from own community into v',
        binding: 'wgsl:currentIncomingWeight'
      },
      {id: 'total', role: 'constant', label: 'total edge weight', binding: 'wgsl:total'},
      {id: 'resolution', role: 'parameter', label: 'resolution', binding: 'wgsl:RESOLUTION'},
      {
        id: 'vertexStatistics',
        role: 'input',
        label: 'vertex strengths (out, in)',
        binding: 'wgsl:vertexStatistics'
      },
      {
        id: 'candidateStatistics',
        role: 'input',
        label: 'candidate community volumes',
        binding: 'wgsl:candidateStatistics'
      },
      {
        id: 'currentStatistics',
        role: 'input',
        label: 'own community volumes',
        binding: 'wgsl:currentStatistics'
      }
    ]
  },
  GPUGraphModularity: {
    id: 'gpu-graph.modularity-contribution',
    derivation: 'curated',
    tex: String.raw`\term{contribution}{Q_c}=2\,\frac{\term{internal}{L_c}}{\term{volume}{V}}-\term{resolution}{\gamma}\,\frac{\term{outgoing}{d_c}}{V}\,\frac{\term{incoming}{d_c}}{V},\qquad
\term{score}{Q}=\sum_c\term{contribution}{Q_c}`,
    ascii:
      'Edges accumulate per community label c: internal L_c (weight of edges with both ends in c), outgoing and incoming volumes d_c (an undirected edge adds its weight to both end communities, so V = sum of volumes = 2W). contribution[c] = factor*L_c/V - resolution*outgoing_c/V*incoming_c/V with factor 2 for undirected and 1 for directed graphs (incoming = outgoing when undirected); Q = sum of contributions via a GPU sum reduction. Invalid labels or weights, or zero total weight, publish zero score and contributions.',
    symbols: [
      {
        id: 'contribution',
        role: 'output',
        label: 'community contribution',
        binding: 'wgsl:contribution'
      },
      {id: 'internal', role: 'input', label: 'internal edge weight', binding: 'wgsl:internal'},
      {id: 'volume', role: 'constant', label: 'total volume', binding: 'wgsl:volume'},
      {id: 'resolution', role: 'parameter', label: 'resolution', binding: 'wgsl:RESOLUTION'},
      {id: 'outgoing', role: 'input', label: 'outgoing volume', binding: 'wgsl:outgoing'},
      {id: 'incoming', role: 'input', label: 'incoming volume', binding: 'wgsl:incoming'},
      {id: 'score', role: 'output', label: 'modularity score', binding: 'wgsl:output'}
    ]
  },
  GPUGraphForceLayout: {
    id: 'gpu-graph.force-layout-step',
    derivation: 'curated',
    tex:
      String.raw`\term{force}{F}_v=-\term{gravity}{g}\,\term{position}{p}_v+\term{repulsion}{R}\sum_{u\ne v}\frac{\term{position}{p}_v-\term{position}{p}_u}{\max(\lVert \term{position}{p}_v-\term{position}{p}_u\rVert^2,\term{epsilon}{\varepsilon})}+\term{attraction}{A}\sum_{u\in\term{neighbors}{N(v)}}(\term{position}{p}_u-\term{position}{p}_v)` +
      String.raw`;\qquad
\term{velocity}{w}_v\leftarrow\operatorname{clamp}_{\term{maxVelocity}{w_{\max}}}\big((\term{velocity}{w}_v+\term{force}{F}_v\,\term{timeStep}{\Delta t})\,\term{damping}{\delta}\big),\qquad \term{position}{p}_v\leftarrow\term{position}{p}_v+\term{velocity}{w}_v\,\term{timeStep}{\Delta t}`,
    ascii:
      'Per iteration, two globally synchronized passes. Force pass: F = -GRAVITY*p + REPULSION*sum over all other vertices (p-q)/max(|p-q|^2, MINIMUM_DISTANCE_SQUARED) + ATTRACTION*sum over CSR neighbors (q-p) (plus reverse CSR for directed graphs); velocity = (velocity + F*TIME_STEP)*DAMPING, capped to MAX_VELOCITY. Integration pass: position += velocity*TIME_STEP. Pinned vertices keep their positions; adjacency overflow clears velocities. Edge weights are ignored.',
    symbols: [
      {id: 'force', role: 'intermediate', label: 'net force', binding: 'wgsl:force'},
      {id: 'gravity', role: 'parameter', label: 'gravity', binding: 'wgsl:GRAVITY'},
      {id: 'position', role: 'output', label: 'position', binding: 'wgsl:positions'},
      {id: 'repulsion', role: 'parameter', label: 'repulsion', binding: 'wgsl:REPULSION'},
      {
        id: 'epsilon',
        role: 'constant',
        label: 'minimum squared distance',
        binding: 'wgsl:MINIMUM_DISTANCE_SQUARED'
      },
      {id: 'attraction', role: 'parameter', label: 'attraction', binding: 'wgsl:ATTRACTION'},
      {id: 'neighbors', role: 'input', label: 'CSR neighbors', binding: 'wgsl:forwardNeighbors'},
      {id: 'velocity', role: 'intermediate', label: 'velocity', binding: 'wgsl:velocity'},
      {id: 'maxVelocity', role: 'parameter', label: 'velocity cap', binding: 'wgsl:MAX_VELOCITY'},
      {id: 'timeStep', role: 'parameter', label: 'time step', binding: 'wgsl:TIME_STEP'},
      {id: 'damping', role: 'parameter', label: 'damping', binding: 'wgsl:DAMPING'}
    ]
  },
  GPUGraphSpatialForceLayout: {
    id: 'gpu-graph.spatial-force-layout-step',
    derivation: 'curated',
    tex:
      String.raw`\term{force}{F}_v=-\term{gravity}{g}\,\term{position}{p}_v+\term{repulsion}{R}\Big[\sum_{u\in\mathrm{near}(v)}\frac{\term{position}{p}_v-\term{position}{p}_u}{\max(\lVert \term{position}{p}_v-\term{position}{p}_u\rVert^2,\term{epsilon}{\varepsilon})}+\sum_{c\in\mathrm{far}(v)}\term{mass}{m_c}\frac{\term{position}{p}_v-\term{center}{\bar p_c}}{\max(\lVert \term{position}{p}_v-\bar p_c\rVert^2,\varepsilon)}\Big]+\term{attraction}{A}\sum_{u\in\term{neighbors}{N(v)}}(p_u-p_v)` +
      String.raw`;\qquad c\in\mathrm{far}(v)\iff \operatorname{cheb}(c,\mathrm{cell}(v))>\term{nearRadius}{r}\ \wedge\ d_{\mathrm{cell}}^2<\term{theta}{\theta}^2\,\lVert p_v-\bar p_c\rVert^2`,
    ascii:
      "Each iteration rebuilds a uniform grid index from current positions (cell offsets, vertex ids, per-cell mean position). For vertex v: F = -GRAVITY*p + REPULSION*(exact all-pairs terms over vertices in cells within NEAR_CELL_RADIUS (Chebyshev) of v's cell, plus for every other occupied cell whose squared diagonal < THETA^2 * |p-center|^2, one monopole: mass*(p-center)/max(|p-center|^2, MINIMUM_DISTANCE_SQUARED)) + ATTRACTION*sum over forward CSR neighbors (q-p), plus reverse CSR neighbors when a reverse topology is supplied (incident edges). Cells that are neither near nor accepted are visited vertex by vertex, so THETA=0 is exact. Velocity = (velocity + F*TIME_STEP)*DAMPING capped to MAX_VELOCITY, then position += velocity*TIME_STEP. Out-of-bounds positions or index overflow freeze positions and clear velocities.",
    symbols: [
      {id: 'force', role: 'intermediate', label: 'net force', binding: 'wgsl:force'},
      {id: 'gravity', role: 'parameter', label: 'gravity', binding: 'wgsl:GRAVITY'},
      {id: 'position', role: 'output', label: 'position', binding: 'wgsl:positions'},
      {id: 'repulsion', role: 'parameter', label: 'repulsion', binding: 'wgsl:REPULSION'},
      {
        id: 'epsilon',
        role: 'constant',
        label: 'minimum squared distance',
        binding: 'wgsl:MINIMUM_DISTANCE_SQUARED'
      },
      {id: 'mass', role: 'input', label: 'vertices in the far cell', binding: 'wgsl:mass'},
      {id: 'center', role: 'input', label: 'far cell center of mass', binding: 'wgsl:center'},
      {id: 'attraction', role: 'parameter', label: 'attraction', binding: 'wgsl:ATTRACTION'},
      {id: 'neighbors', role: 'input', label: 'CSR neighbors', binding: 'wgsl:forwardNeighbors'},
      {
        id: 'nearRadius',
        role: 'parameter',
        label: 'near-cell radius',
        binding: 'wgsl:NEAR_CELL_RADIUS'
      },
      {id: 'theta', role: 'parameter', label: 'opening criterion', binding: 'wgsl:THETA'}
    ]
  }
};

/** Pack tables for the communities and layouts family; `computation-pack.ts` spreads them in. */
export const COMMUNITIES_LAYOUTS_PACK: {
  operators: NonNullable<ViewPackData['operators']>;
  formulas: typeof formulas;
  variableKinds: NonNullable<ViewPackData['variableKinds']>;
  captureStages: NonNullable<ViewPackData['captureStages']>;
} = {
  operators: {
    // atomicMin hooking: each vertex lowers its higher neighbor's label to the minimum seen.
    GPUGraphConnectedComponents: {scope: 'focal', scopeSecondary: 'global', reduction: 'argmin'},
    // Majority vote over the closed neighborhood.
    GPUGraphLabelPropagation: {scope: 'focal', reduction: 'mode'},
    // One best move per round: a global extremum over per-vertex candidate gains.
    GPUGraphModularityOptimization: {scope: 'solver', scopeSecondary: 'focal', reduction: 'argmin'},
    // Per-community sums of edge weights, then one global sum.
    GPUGraphModularity: {scope: 'zonal', scopeSecondary: 'global', reduction: 'sum'},
    // All-pairs repulsion sum plus incident-edge attraction.
    GPUGraphForceLayout: {scope: 'pairwise', scopeSecondary: 'focal', reduction: 'sum'},
    // Grid-bounded repulsion: exact near cells plus one monopole per accepted far cell.
    GPUGraphSpatialForceLayout: {scope: 'pairwise', scopeSecondary: 'zonal', reduction: 'sum'}
  },
  formulas,
  variableKinds: {
    'connected-components': {level: 'nominal'},
    'connected-components-converged': {level: 'nominal'},
    'label-propagation': {level: 'nominal'},
    'label-propagation-converged': {level: 'nominal'},
    'modularity-seed': {level: 'nominal'},
    'modularity-optimization': {level: 'nominal'},
    'modularity-optimization-score': {level: 'interval'},
    'modularity-optimization-converged': {level: 'nominal'},
    'modularity-optimization-valid': {level: 'nominal'},
    'modularity-partition': {level: 'nominal'},
    'modularity-score': {level: 'interval'},
    'modularity-contributions': {level: 'interval'},
    'modularity-valid': {level: 'nominal'},
    'force-layout-start': {level: 'interval'},
    'force-layout': {level: 'interval'},
    'spatial-force-layout-start': {level: 'interval'},
    'spatial-force-layout': {level: 'interval'}
  },
  captureStages: {
    'connected-components': 'iteration',
    'connected-components-stations': 'publication',
    'connected-components-converged': 'publication',
    'label-propagation': 'iteration',
    'label-propagation-stations': 'publication',
    'label-propagation-converged': 'publication',
    'modularity-seed': 'topology',
    'modularity-optimization': 'iteration',
    'modularity-optimization-stations': 'publication',
    'modularity-optimization-score': 'publication',
    'modularity-optimization-converged': 'publication',
    'modularity-optimization-valid': 'publication',
    'modularity-partition': 'topology',
    'modularity-partition-stations': 'topology',
    'modularity-score': 'publication',
    'modularity-contributions': 'publication',
    'modularity-valid': 'publication',
    'force-layout-start': 'parameters',
    'force-layout': 'iteration',
    'force-layout-geometry': 'publication',
    'spatial-force-layout-start': 'parameters',
    'spatial-force-layout': 'iteration',
    'spatial-force-layout-geometry': 'publication'
  }
};
