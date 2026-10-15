// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {OperatorDescriptor, VariableKind} from '../../../computation-views/types';

/** Operator, formula, variable-kind and capture-stage tables for the structure and path family. */
export type GPUGraphPackTables = {
  operators: Readonly<Record<string, OperatorDescriptor>>;
  formulas: Readonly<Record<string, GPUComputationFormulaRef>>;
  variableKinds: Readonly<Record<string, VariableKind>>;
  captureStages: Readonly<Record<string, string>>;
};

/** Kernel-bound pack data for topology, degree, breadth-first search, shortest paths and cores. */
export const GPU_GRAPH_STRUCTURE_PACK: GPUGraphPackTables = {
  operators: {
    GPUGraphTopology: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'scan-sort'},
    GPUGraphDegree: {scope: 'focal', reduction: 'count'},
    GPUGraphBreadthFirstSearch: {scope: 'frontier', reduction: 'argmin'},
    GPUGraphSingleSourceShortestPath: {
      scope: 'focal',
      scopeSecondary: 'solver',
      reduction: 'argmin'
    },
    GPUGraphCoreNumber: {scope: 'focal', scopeSecondary: 'solver', reduction: 'count'},
    GPUGraphLocalClusteringCoefficient: {
      scope: 'focal',
      scopeSecondary: 'pairwise',
      reduction: 'count'
    }
  },
  formulas: {
    GPUGraphTopology: {
      id: 'gpu-graph.topology-exclusive-scan',
      derivation: 'curated',
      tex: String.raw`\term{offset}{o}_v=\term{scanned}{s}_v=\sum_{u<v}\term{degree}{k}_u,\quad
\term{offset}{o}_{\term{count}{N}}=\term{scanned}{s}_{N-1}+\term{degree}{k}_{N-1},\quad
\mathrm{slot}(e)=\term{scanned}{s}_v+\term{cursor}{c}_v`,
      ascii:
        'degrees[v] = accepted edges whose first endpoint is v (source for forward, target for reverse; undirected non-loop edges also count their second endpoint). scannedOffsets = exclusive scan of degrees; offsets[v] = scannedOffsets[v]; offsets[N] = scannedOffsets[N-1] + degrees[N-1]. Each edge takes slot scannedOffsets[v] + atomicAdd(cursors[v], 1), so neighbor order within a vertex is unspecified. Slots at or beyond the neighbor capacity are dropped and flagged by overflow; offsets stay untruncated.',
      symbols: [
        {id: 'offset', role: 'output', label: 'CSR offset', binding: 'wgsl:offsets'},
        {
          id: 'scanned',
          role: 'intermediate',
          label: 'scanned offset',
          binding: 'wgsl:scannedOffsets'
        },
        {id: 'degree', role: 'intermediate', label: 'accepted degree', binding: 'wgsl:degrees'},
        {id: 'cursor', role: 'intermediate', label: 'placement cursor', binding: 'wgsl:cursors'},
        {id: 'count', role: 'constant', label: 'vertex count', binding: 'wgsl:VERTEX_COUNT'}
      ]
    },
    GPUGraphDegree: {
      id: 'gpu-graph.degree-offset-difference',
      derivation: 'curated',
      tex: String.raw`\term{degree}{k}_v=\term{offsets}{o}_{v+1}-\term{offsets}{o}_v`,
      ascii:
        'output[v] = offsets[v+1] - offsets[v], read from the untruncated CSR offsets of the chosen direction; no neighbor list is touched.',
      symbols: [
        {id: 'degree', role: 'output', label: 'vertex degree', binding: 'wgsl:output'},
        {id: 'offsets', role: 'input', label: 'CSR offsets', binding: 'wgsl:offsets'}
      ]
    },
    GPUGraphBreadthFirstSearch: {
      id: 'gpu-graph.breadth-first-frontier-expansion',
      derivation: 'curated',
      tex: String.raw`\term{distance}{d}_v\leftarrow\min\big(\term{distance}{d}_v,\ k+1\big)\ \text{for }v\in\term{neighbors}{N^{+}}(u),\ u\in\term{frontier}{F}_k,\quad
\term{parent}{\pi}_v=\min\{u\in\term{frontier}{F}_k:\ v\in N^{+}(u),\ d_v\ge k+1\}`,
      ascii:
        'distances start at 0xffffffff; seeds get 0. In round k each frontier vertex u with distances[u] = k lowers distances[v] to k+1 (atomicMin) for every neighbor v, and predecessors[v] = min(predecessors[v], u) among the parents that offered distance k+1. A vertex whose previous distance was 0xffffffff joins the next compact frontier. Overflow in the chosen adjacency leaves everything unreachable.',
      symbols: [
        {id: 'distance', role: 'output', label: 'hop distance', binding: 'wgsl:distances'},
        {id: 'neighbors', role: 'input', label: 'CSR neighbors', binding: 'wgsl:neighbors'},
        {
          id: 'frontier',
          role: 'intermediate',
          label: 'current frontier',
          binding: 'wgsl:currentFrontier'
        },
        {id: 'parent', role: 'output', label: 'lowest parent', binding: 'wgsl:predecessors'}
      ]
    },
    GPUGraphSingleSourceShortestPath: {
      id: 'gpu-graph.shortest-path-synchronous-relaxation',
      derivation: 'curated',
      tex: String.raw`\term{distance}{d}^{(k+1)}_v=\min\Big(\term{distance}{d}^{(k)}_v,\ \min_{u\to v}\big(\term{previous}{p}_u+\term{weight}{w}_{uv}\big)\Big),\quad
\term{previous}{p}_u=d^{(k)}_u,\quad
\term{parent}{\pi}_v=\min\{u:\ p_u+w_{uv}=d^{(k+1)}_v<p_v\}`,
      ascii:
        'distances[source] = 0 and +Infinity elsewhere. Each round first copies distances to previousDistances, then every vertex u with finite previousDistances[u] offers previousDistances[u] + edgeWeights[slot] to each neighbor through an atomic float-ordered min on distances. Predecessors are reset where a round improved a vertex and set to the lowest u whose offer equals the new distance. Missing edge weights cost 1.0. A converged flag stops later rounds; there is no CPU priority queue.',
      symbols: [
        {id: 'distance', role: 'output', label: 'travel time', binding: 'wgsl:distances'},
        {
          id: 'previous',
          role: 'intermediate',
          label: 'previous round',
          binding: 'wgsl:previousDistances'
        },
        {id: 'weight', role: 'input', label: 'edge weight', binding: 'wgsl:edgeWeights'},
        {
          id: 'parent',
          role: 'output',
          label: 'lowest improving parent',
          binding: 'wgsl:predecessors'
        }
      ]
    },
    GPUGraphCoreNumber: {
      id: 'gpu-graph.core-number-h-index',
      derivation: 'curated',
      tex: String.raw`\term{core}{c}^{(k+1)}_v=\max\{\term{candidate}{h}\le c^{(k)}_v:\ \term{supporters}{S}_v(h)\ge h\},\quad
\term{supporters}{S}_v(h)=\#\{u\in N(v):\ \term{neighborCore}{c}^{(k)}_u\ge h\},\quad
c^{(0)}_v=|N(v)|`,
      ascii:
        'Start from the distinct weak-neighbor degree. Each synchronized round replaces output[v] with the largest candidate h <= previous output[v] for which at least h distinct neighbors have output >= h (binary search over h, countSupportingNeighbors). Results go to scratch and are published together, so every vertex reads the preceding round. Converged when no vertex changed; otherwise values are upper bounds.',
      symbols: [
        {id: 'core', role: 'output', label: 'core estimate', binding: 'wgsl:output'},
        {id: 'candidate', role: 'index', label: 'candidate core', binding: 'wgsl:candidate'},
        {
          id: 'supporters',
          role: 'intermediate',
          label: 'supporting neighbors',
          binding: 'wgsl:supporters'
        },
        {
          id: 'neighborCore',
          role: 'input',
          label: 'neighbor estimate',
          binding: 'wgsl:neighborCore'
        }
      ]
    },
    GPUGraphLocalClusteringCoefficient: {
      id: 'gpu-graph.local-clustering-directed-closures',
      derivation: 'curated',
      tex: String.raw`\term{coefficient}{C}_v=\frac{\term{closures}{T}_v}{\term{degree}{d}_v\,(\term{degree}{d}_v-1)}\ (d_v\ge2,\ \text{else }0),\quad
T_v=\sum_{a<b\in N(v)}\big(\term{forwardClosure}{[a\to b]}+\term{reverseClosure}{[b\to a]}\big)`,
      ascii:
        'Direct kernel (chosen for this sparse graph): degree counts distinct weak neighbors; closureCount sums, over every unordered neighbor pair a < b, forwardClosure (a->b exists) + reverseClosure (b->a exists); output = closureCount / (degree * (degree - 1)) when degree >= 2 else 0. On undirected CSR each triangle contributes two closures, so this equals 2T / (d(d-1)). Overflow publishes zero. The canonical (dense) kernel computes the same quantity by merging sorted rows.',
      symbols: [
        {
          id: 'coefficient',
          role: 'output',
          label: 'clustering coefficient',
          binding: 'wgsl:output'
        },
        {
          id: 'closures',
          role: 'intermediate',
          label: 'closure count',
          binding: 'wgsl:closureCount'
        },
        {id: 'degree', role: 'intermediate', label: 'distinct neighbors', binding: 'wgsl:degree'},
        {
          id: 'forwardClosure',
          role: 'intermediate',
          label: 'edge a to b',
          binding: 'wgsl:forwardClosure'
        },
        {
          id: 'reverseClosure',
          role: 'intermediate',
          label: 'edge b to a',
          binding: 'wgsl:reverseClosure'
        }
      ]
    }
  },
  variableKinds: {
    'csr-row-offsets': {level: 'ratio', extent: 'extensive'},
    'out-degree': {level: 'ratio', extent: 'extensive'},
    'hop-distance': {level: 'ratio', extent: 'intensive'},
    'travel-time': {level: 'ratio', extent: 'intensive'},
    'core-number': {level: 'ordinal'},
    'clustering-coefficient': {level: 'ratio', extent: 'intensive'}
  },
  captureStages: {
    'csr-row-offsets': 'topology',
    'csr-row-offsets-stations': 'publication',
    'out-degree': 'publication',
    'out-degree-stations': 'publication',
    'hop-distance': 'iteration',
    'hop-distance-stations': 'publication',
    'travel-time': 'iteration',
    'travel-time-stations': 'publication',
    'core-number': 'iteration',
    'core-number-stations': 'publication',
    'clustering-coefficient': 'publication',
    'clustering-coefficient-stations': 'publication'
  }
};
