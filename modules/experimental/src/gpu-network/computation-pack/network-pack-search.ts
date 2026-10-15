// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {NetworkPackPart} from './network-pack-part';

const SEARCH_FORMULA_REFS: Record<string, GPUComputationFormulaRef> = {
  'gpu-network.service-areas.label': {
    id: 'gpu-network.service-areas.label',
    derivation: 'curated',
    tex: String.raw`\term{labels}{L}[\term{targetNode}{v}]\leftarrow\min\bigl(\term{labels}{L}[\term{targetNode}{v}],\ \term{sourceLabel}{\ell_u}\bigr)\quad\text{over tight edges }u\to v:\ \term{tight}{c_u+w_{uv}=c_v}`,
    ascii:
      'Label phase, after the cost search: on every tight edge u->v (f32 equality costs[u] + w == costs[v], no tolerance, zero-weight edges included) labels[v] = min(labels[v], labels[u]) with an atomic minimum, seeded at the facility nodes. The fixpoint is the smallest facility row among all facilities that reach v at its minimum cost, independent of thread order. Unreached nodes keep NONE (0xffffffff).',
    symbols: [
      {id: 'labels', role: 'output', label: 'facility row per node', binding: 'wgsl:labels'},
      {id: 'targetNode', role: 'intermediate', label: 'edge target v', binding: 'wgsl:targetNode'},
      {
        id: 'sourceLabel',
        role: 'intermediate',
        label: 'label of the edge source u',
        binding: 'wgsl:sourceLabel'
      },
      {id: 'tight', role: 'input', label: 'tight-edge bit', binding: 'wgsl:tightBits'}
    ]
  },
  'gpu-network.neighborhood.hops': {
    id: 'gpu-network.neighborhood.hops',
    derivation: 'curated',
    tex: String.raw`\term{hopDistances}{h_v}=\begin{cases}\operatorname{u32}(\term{cost}{c_v})&\term{cost}{c_v}\le\term{limit}{k}\\\term{none}{\mathrm{NONE}}&\text{otherwise}\end{cases}`,
    ascii:
      'hopDistances[v] = cost[v] <= limit ? uint(cost[v]) : NONE, where cost is the unit-weight search cost, limit = min(hops, maximumHops) and NONE = 0xffffffff. NaN and +Infinity costs compare false and stay outside the ego network.',
    symbols: [
      {id: 'hopDistances', role: 'output', label: 'hop distance', binding: 'wgsl:hopDistances'},
      {id: 'cost', role: 'input', label: 'unit-weight cost', binding: 'wgsl:cost'},
      {id: 'limit', role: 'parameter', label: 'hop radius', binding: 'wgsl:limit'},
      {id: 'none', role: 'constant', label: 'outside the ego network', binding: 'wgsl:NONE'}
    ]
  },
  'gpu-network.isochrones.cost': {
    id: 'gpu-network.isochrones.cost',
    derivation: 'curated',
    tex: String.raw`\term{cost}{c}=\left|\term{sourceCost}{d_u}+\term{t}{t}\,\term{weight}{w_{uv}}+\term{walk}{a}\,\term{pixelDistance}{r}\right|,\qquad \text{pixel}=\min c`,
    ascii:
      'Per edge sample at fraction t along u->v and per pixel within the buffer radius: cost = abs(sourceCost + t*weight + walk*pixelDistance); each pixel keeps the minimum cost over all samples (min mode; atomic minimum on order-preserving f32 bits, so the result does not depend on thread order). Samples are linear along straight segments between node positions, so this is a sampled raster, not an exact polygonal travel-time boundary; pixels no sample reaches stay unreached.',
    symbols: [
      {id: 'cost', role: 'output', label: 'pixel cost candidate', binding: 'wgsl:cost'},
      {id: 'sourceCost', role: 'input', label: 'cost at edge source', binding: 'wgsl:sourceCost'},
      {id: 't', role: 'intermediate', label: 'sample fraction', binding: 'wgsl:t'},
      {id: 'weight', role: 'input', label: 'edge travel time', binding: 'wgsl:weight'},
      {id: 'walk', role: 'parameter', label: 'walk cost per metre', binding: 'wgsl:walk'},
      {
        id: 'pixelDistance',
        role: 'intermediate',
        label: 'sample-to-pixel distance',
        binding: 'wgsl:pixelDistance'
      }
    ]
  },
  'gpu-network.catchment.band': {
    id: 'gpu-network.catchment.band',
    derivation: 'curated',
    tex: String.raw`\term{band}{b}=\min\{k:\ \term{time}{t}<\term{bandBreaks}{B}[\term{k}{k}]\}`,
    ascii:
      'band = first k with time < bandBreaks[k], else 0xffffffff (no band). Upper bounds are strict; unreached, unsnapped or non-finite demand times (>= 3.0e38) get no band. time = min(cost[source] + sourceCost, cost[target] + targetCost) on the snapped edge, from the demand-time kernel.',
    symbols: [
      {id: 'band', role: 'output', label: 'drive-time band', binding: 'wgsl:bands'},
      {id: 'time', role: 'input', label: 'demand drive time', binding: 'wgsl:time'},
      {id: 'bandBreaks', role: 'input', label: 'ascending band bounds', binding: 'wgsl:bandBreaks'},
      {id: 'k', role: 'intermediate', label: 'band index', binding: 'wgsl:k'}
    ]
  },
  'gpu-network.snapping.nearest': {
    id: 'gpu-network.snapping.nearest',
    derivation: 'curated',
    tex: String.raw`\term{bestEdge}{e^*}=\operatorname*{argmin}_{e}\ \term{distance}{d(p,e)}`,
    ascii:
      'For each point p the scan visits every edge row e in ascending order and keeps the edge with the smallest point-to-segment distance d(p,e): bestEdge = e when bestEdge is NONE or d < bestDistance (strict, so ties go to the smallest edge row; NaN never wins). Distances are in the spatial context length unit (meters for a great-circle longitude/latitude context); with a maximum snap distance, edges beyond it are skipped and an unsnapped point gets NONE.',
    symbols: [
      {id: 'bestEdge', role: 'output', label: 'snapped edge row', binding: 'wgsl:bestEdge'},
      {id: 'distance', role: 'intermediate', label: 'distance to edge', binding: 'wgsl:distance'},
      {
        id: 'bestDistance',
        role: 'intermediate',
        label: 'smallest distance so far',
        binding: 'wgsl:bestDistance'
      }
    ]
  },
  'gpu-network.accessibility.gravity': {
    id: 'gpu-network.accessibility.gravity',
    derivation: 'curated',
    tex: String.raw`\term{gravitySum}{A_j}=\sum_{i}\term{weight}{O_i}\,e^{-\term{beta}{\beta}\,\term{cost}{c_{ij}}}\quad\text{over }c_{ij}\le\term{threshold}{T}`,
    ascii:
      'Exponential-decay variant, opportunity-rows orientation: for every column node j the kernel sums rows i in ascending order and adds weight[i] * exp(-beta * cost[i,j]) for finite costs 0 <= cost <= threshold and finite weights (unreached +Infinity entries and non-finite weights are skipped; the sum is deterministic, no float atomics, and weights are not normalised). The cumulative variant omits the decay and the power variant uses max(cost, minimumCost)^-beta; they are alternative branches of the same kernel.',
    symbols: [
      {id: 'gravitySum', role: 'output', label: 'gravity score', binding: 'wgsl:gravitySum'},
      {id: 'weight', role: 'input', label: 'opportunity weight', binding: 'wgsl:weight'},
      {id: 'beta', role: 'parameter', label: 'decay rate', binding: 'wgsl:beta'},
      {id: 'cost', role: 'input', label: 'matrix cost', binding: 'wgsl:cost'},
      {id: 'threshold', role: 'parameter', label: 'cost cutoff', binding: 'wgsl:threshold'}
    ]
  }
};

/** Network pack part: search and accessibility (batch A). */
export const NETWORK_SEARCH_PACK: NetworkPackPart = {
  formulaRefs: SEARCH_FORMULA_REFS,
  formulas: {
    GPUNetworkServiceAreas: SEARCH_FORMULA_REFS['gpu-network.service-areas.label'],
    GPUNetworkPathExtraction: {
      formula: 'none',
      reason:
        'bounded predecessor-chain walking, then scans that order and compact the paths, are procedural; the graph rung shows measure, scan, write and publish'
    },
    GPUNetworkNeighborhood: SEARCH_FORMULA_REFS['gpu-network.neighborhood.hops'],
    GPUNetworkIsochrones: SEARCH_FORMULA_REFS['gpu-network.isochrones.cost'],
    addDriveTimeCatchmentRecipe: SEARCH_FORMULA_REFS['gpu-network.catchment.band'],
    GPUNetworkSnapping: SEARCH_FORMULA_REFS['gpu-network.snapping.nearest'],
    GPUNetworkCostMatrix: {
      formula: 'none',
      reason:
        'the matrix batches an existing reachability search over a lane-expanded network and gathers the selected columns; it defines no new numeric kernel'
    },
    GPUNetworkAccessibility: SEARCH_FORMULA_REFS['gpu-network.accessibility.gravity']
  },
  captureStages: {
    assignments: 'assignment',
    hopDistances: 'search',
    triangleBands: 'assignment',
    demandTimes: 'search',
    demandBands: 'assignment',
    snapDistances: 'assignment',
    matrixCosts: 'search',
    gravity: 'summary',
    pathFound: 'summary'
  },
  variableKinds: {
    assignments: {level: 'nominal'},
    hopDistances: {level: 'ordinal'},
    triangleBands: {level: 'ordinal'},
    demandTimes: {level: 'ratio', extent: 'intensive'},
    demandBands: {level: 'ordinal'},
    snapDistances: {level: 'ratio', extent: 'intensive'},
    matrixCosts: {level: 'ratio', extent: 'intensive'},
    gravity: {level: 'ratio', extent: 'extensive'},
    pathFound: {level: 'nominal'}
  }
};
