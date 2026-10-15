// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const CANDIDATE = {kind: 'candidate', label: 'candidate point', indexSpace: '1d'} as const;
const SAMPLE = {kind: 'sample', label: 'profile sample', indexSpace: '1d'} as const;

/** Terrain pack entries of the `features` family, keyed by exported class name. */
export const TERRAIN_FEATURES_OPERATORS: RasterOperatorTable = {
  GPUTerrainPeakProminence: {
    family: 'features',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {
      dem: 'elevation',
      peakProminence: 'measure',
      prominenceKeySaddle: 'elevation',
      prominenceParentPeak: 'category'
    },
    stages: ['neighbourhood', 'operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Prominence is read off a join tree built on the GPU from radix-sorted spill edges by parent splicing and ancestor doubling; it has no closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence.ts',
      'modules/experimental/src/gpu-terrain/terrain-features/terrain-peak-prominence-tree.ts'
    ]
  },
  GPUTerrainSummits: {
    family: 'features',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {dem: 'elevation', summitMask: 'category', summitsDrop: 'measure'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'radius', unit: 'm', sweepable: true},
      {stage: 'operator', name: 'minimumDrop', label: 'minimum drop', unit: 'm', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'incompleteNeighborhood',
        alternatives: ['reject', 'ignore'],
        sweepable: false
      }
    ],
    // As implemented: a summit is the strict (height, -index) maximum of a metric disc; its drop
    // is measured against the highest pixel of the disc's boundary ring.
    formula: {
      id: 'gpu-terrain.terrain-summits.drop',
      tex: String.raw`\term{drop}{\delta}(p) = \term{ownHeight}{h_p} - \max_{q \in \mathrm{Ring}(p)\setminus\{p\}} \term{ringHeight}{h_q},\quad p \text{ is a summit} \iff h_p \text{ is the disc maximum and } \delta(p) \ge \term{minimumDrop}{\delta_{\min}}`,
      ascii:
        'p is a summit iff it is the strict maximum of its metric disc (ties: lowest row-major index wins) and drop(p) = h(p) - max(h(q)) over the disc ring >= minimumDrop; the disc is ((dx*cx)^2 + (dy*cy)^2) <= radius^2 at the centre row; reject policy: a disc leaving the grid or touching nodata is no summit; drop is +Infinity with an empty ring; NaN where there is no summit',
      derivation: 'curated',
      symbols: [
        {id: 'drop', role: 'output', label: 'δ', binding: 'wgsl:dropValue'},
        {id: 'ownHeight', role: 'input', label: 'h_p', binding: 'wgsl:ownHeight'},
        {id: 'ringHeight', role: 'intermediate', label: 'h_q', binding: 'wgsl:ringHeight'},
        {id: 'minimumDrop', role: 'parameter', label: 'δ_min', binding: 'wgsl:minimumDrop'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-features/gpu-terrain-summits.ts']
  },
  GPUTerrainPeakSnap: {
    family: 'features',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {
      peakCandidates: 'measure',
      peakSnapPositions: 'measure',
      peakSnapHeights: 'elevation',
      peakSnapStatus: 'category',
      peakSnapDistance: 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CANDIDATE],
    neighbourhood: 'window',
    guards: ['metric-cell-size'],
    choices: [
      {stage: 'neighbourhood', name: 'radius', unit: 'm', sweepable: true},
      {stage: 'operator', name: 'maximumMove', label: 'maximum move', unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'maximumHeightChange',
        label: 'maximum height change',
        unit: 'm',
        sweepable: true
      },
      {stage: 'operator', name: 'interior', alternatives: [true, false], sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Each candidate takes the strict (height, -index) maximum of a metric disc, then passes an ordered chain of accept rules (ring, move length, height change); the outcome is a status code, not a formula.'
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-features/gpu-terrain-peak-snap.ts']
  },
  GPUTerrainCriticalPoints: {
    family: 'features',
    descriptor: {scope: 'focal', reduction: 'count'},
    variables: {
      dem: 'elevation',
      criticalClasses: 'category',
      criticalSignChanges: 'measure',
      criticalCounts: 'measure'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'connectivity', alternatives: [8, 6], sweepable: false}
    ],
    // Peucker and Douglas as implemented: sign changes of "neighbour is higher" around the ring,
    // with ties broken by row-major index; edge pixels and nodata rings are boundary.
    formula: {
      id: 'gpu-terrain.terrain-critical-points.rule',
      tex: String.raw`\term{changes}{c} = \#\{k : s_k \ne s_{k+1}\},\quad s_k = [\,h_k > h_p \ \lor\ (h_k = h_p \land k > p)\,],\quad \text{class} = \begin{cases}\text{peak} & c = 0,\ \term{higher}{H} = 0\\ \text{pit} & c = 0,\ H = \text{ring}\\ \text{regular} & c = 2\\ \text{saddle} & c \ge 4\end{cases}`,
      ascii:
        'walk the 8-neighbour ring (or the 6-ring) of p; s_k = 1 when neighbour k is higher (equal heights: higher iff its row-major index is larger); c = number of sign changes around the ring; c = 0 and no neighbour higher: peak; c = 0 and all higher: pit; c = 2: regular; c >= 4: saddle (multiplicity c/2 - 1); grid-edge pixels and rings touching nodata are boundary, invalid pixels are noData',
      derivation: 'curated',
      symbols: [
        {id: 'changes', role: 'intermediate', label: 'c', binding: 'wgsl:changeCount'},
        {id: 'higher', role: 'intermediate', label: 'H', binding: 'wgsl:higherCount'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-features/gpu-terrain-critical-points.ts'
    ]
  },
  GPUProfilePeaks: {
    family: 'features',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      profileValues: 'elevation',
      profilePeakProminence: 'measure',
      profileRefinedIndex: 'measure',
      profileRefinedValue: 'elevation',
      profilePeakMask: 'category',
      profilePeaksConverged: 'category'
    },
    stages: ['neighbourhood', 'operator', 'iteration'],
    datumKinds: [SAMPLE],
    neighbourhood: 'window',
    guards: ['convergence'],
    choices: [
      {stage: 'operator', name: 'minimumProminence', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'window', unit: 'samples', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumSide', unit: 'samples', sweepable: true},
      {stage: 'operator', name: 'suppressionRadius', unit: 'samples', sweepable: true},
      {stage: 'iteration', name: 'maximumSuppressionIterations', sweepable: true}
    ],
    // As implemented: windowed prominence of a local maximum over +-2 samples; non-maximum
    // suppression is gated fixed-point rounds over candidates ranked by prominence.
    formula: {
      id: 'gpu-terrain.profile-peaks.prominence',
      tex: String.raw`\term{prominence}{P} = \term{peak}{v} - \max\bigl(\term{lowestLeft}{\ell_L},\ \term{lowestRight}{\ell_R}\bigr),\quad \ell_{\pm} = \min\{w_j\} \text{ walking up to the window until a sample } w_j > v,\quad \text{kept} \iff P \ge \term{minimumProminence}{P_{\min}}`,
      ascii:
        'candidate: sample i with value v that is no lower than its +-2 neighbours (equal on the left rejects); per side walk up to window samples, lowest = min of the values seen, stop at the first sample higher than v; reject if either side has fewer than minimumSide valid samples; prominence = v - max(lowestLeft, lowestRight), rejected below minimumProminence; kept peaks survive non-maximum suppression within suppressionRadius samples, ranked by prominence (gated rounds)',
      derivation: 'curated',
      symbols: [
        {id: 'prominence', role: 'output', label: 'P', binding: 'wgsl:prominence'},
        {id: 'peak', role: 'input', label: 'v', binding: 'wgsl:peak'},
        {id: 'lowestLeft', role: 'intermediate', label: 'ℓ_L', binding: 'wgsl:lowest'},
        {id: 'lowestRight', role: 'intermediate', label: 'ℓ_R', binding: 'wgsl:lowest'},
        {
          id: 'minimumProminence',
          role: 'parameter',
          label: 'P_min',
          binding: 'wgsl:minimumProminence'
        }
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-features/gpu-profile-peaks.ts']
  }
};
