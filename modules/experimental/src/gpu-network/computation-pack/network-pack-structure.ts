// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {NetworkPackPart} from './network-pack-part';

const STRUCTURE_FORMULA_REFS: Record<string, GPUComputationFormulaRef> = {
  'gpu-network.analytics.normalize': {
    id: 'gpu-network.analytics.normalize',
    derivation: 'curated',
    tex: String.raw`\term{normalized}{s}=\begin{cases}(\term{degree}{x}-\term{lowest}{a})/(\term{highest}{b}-\term{lowest}{a})&\term{highest}{b}>\term{lowest}{a}\\0&\text{otherwise}\end{cases}`,
    ascii:
      'normalized = highest > lowest ? (raw - lowest) / (highest - lowest) : 0, in f32. lowest and highest are the two words of the extent reduction (minimum, maximum of the raw column over all nodes); a column with a single value (no range) normalizes to 0 for every node. The raw degree is the outgoing CSR row length from the degree algorithm.',
    symbols: [
      {id: 'normalized', role: 'output', label: 'scaled degree', binding: 'wgsl:normalized'},
      {id: 'degree', role: 'input', label: 'raw column value', binding: 'wgsl:raw'},
      {id: 'lowest', role: 'intermediate', label: 'column minimum', binding: 'wgsl:lowest'},
      {id: 'highest', role: 'intermediate', label: 'column maximum', binding: 'wgsl:highest'}
    ]
  },
  'gpu-network.centrality.closeness': {
    id: 'gpu-network.centrality.closeness',
    derivation: 'curated',
    tex: String.raw`\term{closeness}{C}=\begin{cases}(\term{reachedPartial}{r}-1)/\term{totalCost}{S}&\term{totalCost}{S}>0\\0&\text{otherwise}\end{cases}`,
    ascii:
      'closeness = totalCost > 0 ? (reachedPartial[0] - 1) / totalCost : 0, per pivot. Outward closeness: totalCost is the f32 sum of the search costs of every node the pivot reaches (infinite costs are skipped), added by a fixed 256-lane tree reduction so the value does not depend on thread order. reachedPartial[0] counts the reached nodes including the pivot itself. The Wasserman-Faust factor (r - 1) / (N - 1) is not applied; the unit is 1 / cost (1 / second on travel-time weights).',
    symbols: [
      {id: 'closeness', role: 'output', label: 'outward closeness', binding: 'wgsl:closeness'},
      {
        id: 'reachedPartial',
        role: 'intermediate',
        label: 'reached nodes, pivot included',
        binding: 'wgsl:reachedPartial'
      },
      {
        id: 'totalCost',
        role: 'intermediate',
        label: 'sum of reached costs',
        binding: 'wgsl:totalCost'
      }
    ]
  },
  'gpu-network.statistics.degree-histogram': {
    id: 'gpu-network.statistics.degree-histogram',
    derivation: 'curated',
    tex: String.raw`\term{localHistogram}{H}[\term{getBin}{b}(\term{outValue}{d})]\leftarrow\term{localHistogram}{H}[\term{getBin}{b}(\term{outValue}{d})]+1`,
    ascii:
      'localHistogram[getBin(outValue)] += 1 (atomic) for every live node, where outValue is its live out-degree and getBin(d) = min(floor(d / binWidth), BIN_COUNT - 1) for the linear rule (the last bin saturates; log2 binning uses bin 0 for d = 0, else min(1 + floor(log2 d), BIN_COUNT - 1)). Each workgroup then adds its non-empty bins to the summary words, so integer addition makes the histogram independent of thread order. The in-degree and total-degree histograms use the same rule.',
    symbols: [
      {
        id: 'localHistogram',
        role: 'intermediate',
        label: 'workgroup histogram',
        binding: 'wgsl:localHistogram'
      },
      {id: 'getBin', role: 'intermediate', label: 'degree bin', binding: 'wgsl:getBin'},
      {id: 'outValue', role: 'input', label: 'live out-degree', binding: 'wgsl:outValue'}
    ]
  },
  'gpu-network.subgraph-filter.range': {
    id: 'gpu-network.subgraph-filter.range',
    derivation: 'curated',
    tex: String.raw`\term{live}{m}\leftarrow\term{live}{m}\land\bigl(\term{parameters}{p}[\term{base}{b}]\le\term{value}{x}<\term{parameters}{p}[\term{base}{b}+1]\bigr)`,
    ascii:
      'For every enabled range, live = live && value >= parameters[base] && value < parameters[base + 1]: half-open, so the maximum itself is excluded, and a NaN value never passes. A disabled range (parameters[base + 2] == 0) is skipped. The mask starts at 1 (or at the caller mask), so the result is the AND of all ranges; an edge slot is finally kept only when both of its endpoints are live, and dropIsolated also removes nodes left without a live slot. Integer and comparison arithmetic only, so the result is exact.',
    symbols: [
      {id: 'live', role: 'output', label: 'slot is kept', binding: 'wgsl:live'},
      {id: 'parameters', role: 'parameter', label: 'range bounds', binding: 'wgsl:parameters'},
      {id: 'base', role: 'intermediate', label: 'range record offset', binding: 'wgsl:base'},
      {id: 'value', role: 'input', label: 'column value', binding: 'wgsl:value'}
    ]
  },
  'gpu-network.adjacency-matrix.bin': {
    id: 'gpu-network.adjacency-matrix.bin',
    derivation: 'curated',
    tex: String.raw`\term{getBin}{b}(\term{position}{p},\term{start}{s},\term{end}{e})=\left\lfloor\frac{(\term{position}{p}-\term{start}{s})\,\term{RESOLUTION}{R}}{\term{end}{e}-\term{start}{s}}\right\rfloor`,
    ascii:
      'getBin(position, start, end) = (position - start) * RESOLUTION / (end - start) with u32 division (floor). A live slot u -> v adds 1 (atomically) to cell getBin(order[u]) * RESOLUTION + getBin(order[v]): rows are source positions, columns are target positions, and slots whose positions fall outside the window [start, end) are dropped. Counts are exact integers, so the matrix does not depend on thread order; with fewer bins than nodes, each bin covers several consecutive matrix positions.',
    symbols: [
      {id: 'getBin', role: 'output', label: 'matrix bin', binding: 'wgsl:getBin'},
      {id: 'position', role: 'input', label: 'matrix position', binding: 'wgsl:position'},
      {id: 'start', role: 'parameter', label: 'window start', binding: 'wgsl:start'},
      {id: 'end', role: 'parameter', label: 'window end', binding: 'wgsl:end'},
      {id: 'RESOLUTION', role: 'parameter', label: 'bins per axis', binding: 'wgsl:RESOLUTION'}
    ]
  },
  'gpu-network.line-graph.arc-weight': {
    id: 'gpu-network.line-graph.arc-weight',
    derivation: 'curated',
    tex: String.raw`\term{lineWeights}{w_{ef}}=\begin{cases}\term{baseCost}{w_f}+\max(\term{getTurnCost}{p_{ef}},0)&\term{baseCost}{w_f}\ge0\\-1&\text{otherwise}\end{cases}`,
    ascii:
      'lineWeights[arc e -> f] = baseCost >= 0 ? baseCost + max(turnCost, 0) : -1, where baseCost = weights[f] is the cost of the NEXT edge f (not of the entering edge e) and turnCost = getTurnCost(e, f) = angleCost * |angle| plus the left or right extra beyond straightAngle; a turn with |angle| >= uTurnAngle pays uTurnCost instead, and a negative uTurnCost bans it (no arc is emitted). Banned pairs are removed too, so every emitted arc has turnCost >= 0. The angle is between the great-circle arrival bearing of e and the departure bearing of f at the turn node, positive to the left. Arcs of one edge are listed in ascending next-edge order.',
    symbols: [
      {id: 'lineWeights', role: 'output', label: 'arc cost', binding: 'wgsl:lineWeights'},
      {id: 'baseCost', role: 'input', label: 'cost of the next edge', binding: 'wgsl:baseCost'},
      {id: 'getTurnCost', role: 'intermediate', label: 'turn cost', binding: 'wgsl:getTurnCost'}
    ]
  }
};

/** Network pack part: structure and representation (batch B). */
export const NETWORK_STRUCTURE_PACK: NetworkPackPart = {
  formulaRefs: STRUCTURE_FORMULA_REFS,
  formulas: {
    GPUNetworkAnalyticsColumns: STRUCTURE_FORMULA_REFS['gpu-network.analytics.normalize'],
    GPUNetworkCentrality: STRUCTURE_FORMULA_REFS['gpu-network.centrality.closeness'],
    GPUNetworkStatistics: STRUCTURE_FORMULA_REFS['gpu-network.statistics.degree-histogram'],
    GPUNetworkSubgraphFilter: STRUCTURE_FORMULA_REFS['gpu-network.subgraph-filter.range'],
    GPUNetworkCoarsening: {
      formula: 'none',
      reason:
        'fixed-point group accumulation and the sort-and-segment superedge build are procedural; the graph rung shows the fills, sort, scans and publish'
    },
    GPUAdjacencyMatrix: STRUCTURE_FORMULA_REFS['gpu-network.adjacency-matrix.bin'],
    GPUAdjacencyMatrixOrder: {
      formula: 'none',
      reason:
        'two stable radix sorts and an inverse-permutation scatter have no scalar equation; the graph rung shows identity, gather, sort and scatter'
    },
    GPUNetworkLineGraph: STRUCTURE_FORMULA_REFS['gpu-network.line-graph.arc-weight']
  },
  captureStages: {
    degree: 'summary',
    normalized: 'summary',
    extent: 'summary',
    closeness: 'summary',
    reachedCounts: 'summary',
    networkStatistics: 'summary',
    outDegreeHistogram: 'summary',
    nodeMask: 'representation',
    edgeMask: 'representation',
    liveCounts: 'summary',
    zones: 'assignment',
    groupNodeCount: 'summary',
    groupCentroid: 'summary',
    edgeCounts: 'summary',
    groups: 'assignment',
    order: 'representation',
    matrixCounts: 'summary',
    lineNeighbors: 'representation',
    lineWeights: 'representation',
    legalTurns: 'representation'
  },
  variableKinds: {
    degree: {level: 'ratio', extent: 'extensive'},
    normalized: {level: 'ratio', extent: 'intensive'},
    closeness: {level: 'ratio', extent: 'intensive'},
    reachedCounts: {level: 'ratio', extent: 'extensive'},
    networkStatistics: {level: 'ratio', extent: 'extensive'},
    outDegreeHistogram: {level: 'ratio', extent: 'extensive'},
    nodeMask: {level: 'nominal'},
    edgeMask: {level: 'nominal'},
    liveCounts: {level: 'ratio', extent: 'extensive'},
    zones: {level: 'nominal'},
    groupNodeCount: {level: 'ratio', extent: 'extensive'},
    edgeCounts: {level: 'ratio', extent: 'extensive'},
    groups: {level: 'nominal'},
    order: {level: 'ordinal'},
    sampleNodes: {level: 'ordinal'},
    groupsSample: {level: 'nominal'},
    orderSample: {level: 'ordinal'},
    matrixCounts: {level: 'ratio', extent: 'extensive'},
    lineWeights: {level: 'ratio', extent: 'intensive'},
    legalTurns: {level: 'ratio', extent: 'extensive'}
  }
};
