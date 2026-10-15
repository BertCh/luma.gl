// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialChoiceTemplate} from '../choices';
import type {SpatialFamilyOperators} from '../types';

/** Contiguity rule that built the adjacency the regionalization reads (a weights choice). */
const CRITERION_CHOICE: SpatialChoiceTemplate = {
  name: 'criterion',
  label: 'Adjacency rule',
  stage: 'neighbourhood',
  alternatives: ['queen', 'rook'],
  sweepable: true
};

/** Whether attribute columns are scaled to unit variance before distances are taken. */
const STANDARDIZE_CHOICE: SpatialChoiceTemplate = {
  name: 'standardize',
  label: 'Standardize columns',
  stage: 'operator',
  alternatives: [true, false],
  sweepable: true
};

const SEED_CHOICE: SpatialChoiceTemplate = {name: 'seed', stage: 'operator', sweepable: true};

/**
 * Spatial computation-pack entries for the `statistics` family (regionalization and location allocation), keyed by exported name.
 * Merged into `STATISTICS_OPERATORS`. Every entry must use `family: 'statistics'`.
 */
export const STATISTICS_REGION_OPERATORS: SpatialFamilyOperators = {
  GPUAZPRegions: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'solve'},
    variables: {
      values: 'rate',
      initialLabels: 'category',
      labels: 'category',
      objective: 'measure',
      status: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      CRITERION_CHOICE,
      STANDARDIZE_CHOICE,
      {name: 'maximumIterations', label: 'Rounds of moves', stage: 'operator', sweepable: true},
      SEED_CHOICE
    ],
    formula: {
      id: 'azp-objective',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{objective}{\mathrm{SSD}}&=\sum_{A}\sum_{i\in A}\lVert x_i-\bar{x}_A\rVert^2\\
\term{gain}{g_{i\to B}}&=\term{sourceAfter}{\tfrac{(S_A-x_i)^2}{n_A-1}}+\term{destinationAfter}{\tfrac{(S_B+x_i)^2}{n_B+1}}-\term{sourceBefore}{\tfrac{S_A^2}{n_A}}-\term{destinationBefore}{\tfrac{S_B^2}{n_B}}
\end{aligned}`,
      ascii:
        'Objective, not the procedure: SSD = sum over regions A of sum over tracts i in A of |x_i - mean_A|^2, in standardized units unless standardize is false. A move of tract i from region A to adjacent region B gains g = (S_A - x_i)^2/(n_A - 1) + (S_B + x_i)^2/(n_B + 1) - S_A^2/n_A - S_B^2/n_B, summed over columns, where S is a region column sum and n its size. A move needs g above a small fraction of the four terms and must keep A connected.',
      symbols: [
        {id: 'objective', role: 'output', binding: 'wgsl:objective'},
        {id: 'gain', role: 'intermediate', binding: 'wgsl:gainBits'},
        {id: 'sourceAfter', role: 'intermediate', binding: 'wgsl:sourceAfter'},
        {id: 'destinationAfter', role: 'intermediate', binding: 'wgsl:destinationAfter'},
        {id: 'sourceBefore', role: 'intermediate', binding: 'wgsl:sourceBefore'},
        {id: 'destinationBefore', role: 'intermediate', binding: 'wgsl:destinationBefore'}
      ]
    }
  },
  GPUMaxPRegions: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'solve'},
    variables: {
      values: 'rate',
      extensiveValues: 'count',
      threshold: 'count',
      labels: 'category',
      objective: 'measure',
      status: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      CRITERION_CHOICE,
      STANDARDIZE_CHOICE,
      {
        name: 'threshold',
        label: 'Minimum extensive total per region',
        stage: 'operator',
        sweepable: true
      },
      {name: 'startCount', label: 'Independent starts', stage: 'operator', sweepable: true},
      {name: 'maximumGrowthSteps', stage: 'operator', sweepable: true},
      SEED_CHOICE
    ],
    formula: {
      id: 'max-p-objective',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\text{maximize } p\quad&\text{subject to}\quad \term{regionTotal}{\sum_{i\in r}e_i}\ge\term{threshold}{T}\text{ for every region }r\\
\term{objective}{W}&=\sum_{r}\sum_{i\in r}\lVert x_i-\bar{x}_r\rVert^2
\end{aligned}`,
      ascii:
        'Objective, not the procedure: maximize the region count p such that every contiguous region r keeps sum e_i >= T (e extensive, such as population). Among starts the fewest under-threshold regions win, then the largest p, then the smallest within-region sum of squares W = sum over r of sum over i in r of |x_i - mean_r|^2 (standardized units unless standardize is false), then the lowest start index. Regions grow one tract per round, so p is a heuristic maximum, not a proof.',
      symbols: [
        {id: 'regionTotal', role: 'intermediate', binding: 'wgsl:extensive'},
        {id: 'threshold', role: 'parameter', binding: 'wgsl:threshold'},
        {id: 'objective', role: 'output', binding: 'wgsl:withinSquares'}
      ]
    }
  },
  GPUWardRegions: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'solve'},
    variables: {
      values: 'rate',
      labels: 'category',
      objective: 'measure',
      status: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      CRITERION_CHOICE,
      STANDARDIZE_CHOICE,
      {name: 'targetRegionCount', label: 'Target region count', stage: 'operator', sweepable: true},
      {
        name: 'mergeSchedule',
        label: 'Merge order',
        stage: 'operator',
        alternatives: ['sequential', 'mutual-nearest'],
        sweepable: true
      },
      SEED_CHOICE
    ],
    formula: {
      id: 'ward-merge-cost',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{increase}{\Delta W_{ab}}&=\frac{\term{firstCount}{n_a}\,\term{secondCount}{n_b}}{n_a+n_b}\sum_{c}\term{difference}{(\bar{x}_{ac}-\bar{x}_{bc})^2}\\
\term{objective}{W}&=\sum_{\text{merges}}\Delta W_{ab}
\end{aligned}`,
      ascii:
        'Merge cost, not the procedure: joining adjacent regions a and b raises the within-region sum of squares by dW = n_a n_b / (n_a + n_b) * sum over columns c of (mean_ac - mean_bc)^2. Every tract starts alone with W = 0 and each merge adds its dW, so the final W is the exact within-region sum of squares (standardized units unless standardize is false). sequential merges the cheapest adjacent pair per iteration; mutual-nearest merges every mutually cheapest pair per round and can stop at a different partition.',
      symbols: [
        {id: 'increase', role: 'intermediate', binding: 'wgsl:increase'},
        {id: 'firstCount', role: 'intermediate', binding: 'wgsl:firstCount'},
        {id: 'secondCount', role: 'intermediate', binding: 'wgsl:secondCount'},
        {id: 'difference', role: 'intermediate', binding: 'wgsl:difference'},
        {id: 'objective', role: 'output', binding: 'wgsl:objective'}
      ]
    }
  },
  GPUSpatialWeightsMinimumSpanningTree: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'select'},
    variables: {
      values: 'rate',
      standardizedValues: 'score',
      treeEdgeFlags: 'category',
      componentLabels: 'category',
      edgeDissimilarities: 'measure',
      treeSegments: 'coordinate'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [CRITERION_CHOICE, STANDARDIZE_CHOICE],
    formula: {
      id: 'minimum-spanning-tree-cost',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{standardized}{z_{ic}}&=\frac{x_{ic}-\bar{x}_c}{\term{deviation}{s_c}}\\
\term{cost}{d_{ij}}&=\sum_{c}\term{delta}{(z_{ic}-z_{jc})^2},\qquad T=\arg\min_{\text{spanning forest}}\sum_{(i,j)\in T}d_{ij}
\end{aligned}`,
      ascii:
        'Edge cost, not the procedure: d_ij = sum over columns c of (z_ic - z_jc)^2, the squared Euclidean distance between standardized attribute rows, on each contiguity link. z = (x - mean) / s with s the population standard deviation (a constant column becomes 0). The tree T minimizes the total cost of a spanning forest; ties break on the lower-row CSR slot, so the forest is unique.',
      symbols: [
        {id: 'standardized', role: 'intermediate', binding: 'wgsl:scaled'},
        {id: 'deviation', role: 'intermediate', binding: 'wgsl:deviation'},
        {id: 'cost', role: 'output', binding: 'wgsl:cost'},
        {id: 'delta', role: 'intermediate', binding: 'wgsl:delta'}
      ]
    }
  },
  GPUSkaterRegions: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'solve'},
    variables: {
      standardizedValues: 'score',
      labels: 'category',
      regionCount: 'count',
      cutEdges: 'category',
      cutGains: 'measure'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      CRITERION_CHOICE,
      {name: 'regionCount', label: 'Target region count', stage: 'operator', sweepable: true},
      {name: 'minimumRegionSize', label: 'Smallest region', stage: 'operator', sweepable: true},
      {name: 'maximumRegionCount', stage: 'operator', sweepable: false}
    ],
    formula: {
      id: 'skater-cut-gain',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{gain}{g_e}&=\term{ssd}{\mathrm{SSD}}(R)-\mathrm{SSD}(C_e)-\mathrm{SSD}(R\setminus C_e)\\
\mathrm{SSD}(S)&=\term{squares}{\sum_{i\in S}\sum_c x_{ic}^2}-\frac{\term{sumSquares}{\sum_c\left(\sum_{i\in S}x_{ic}\right)^2}}{\term{size}{n_S}}
\end{aligned}`,
      ascii:
        'Cut gain, not the procedure: cutting tree edge e splits region R into the child side C_e and the remainder, lowering the within-region sum of squares by g_e = SSD(R) - SSD(C_e) - SSD(R \\ C_e), with SSD(S) = sum_i sum_c x_ic^2 - sum_c (sum_i x_ic)^2 / n_S. Each step applies the largest gain (ties to the lowest edge slot) whose two sides both keep minimumRegionSize rows, until the target region count is reached.',
      symbols: [
        {id: 'gain', role: 'output', binding: 'wgsl:gain'},
        {id: 'ssd', role: 'intermediate', binding: 'wgsl:getSsd'},
        {id: 'squares', role: 'intermediate', binding: 'wgsl:squares'},
        {id: 'sumSquares', role: 'intermediate', binding: 'wgsl:sumSquares'},
        {id: 'size', role: 'intermediate', binding: 'wgsl:count'}
      ]
    }
  },
  GPURegionPartitionEvaluation: {
    family: 'statistics',
    descriptor: {scope: 'zonal', scopeSecondary: 'global', reduction: 'sum'},
    variables: {
      values: 'rate',
      labels: 'category',
      summary: 'score',
      regionSizes: 'count',
      regionWithinSumOfSquares: 'measure'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [{name: 'labelCapacity', label: 'Label values', stage: 'operator', sweepable: false}],
    formula: {
      id: 'partition-pseudo-f',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{within}{W}&=\sum_{r}\sum_{i\in r}\lVert x_i-\bar{x}_r\rVert^2,&\term{total}{T}&=\sum_{i}\lVert x_i-\bar{x}\rVert^2\\
\term{between}{B}&=T-W,&\term{pseudoF}{F}&=\frac{B/(\term{regions}{k}-1)}{W/(\term{labelled}{n}-k)}
\end{aligned}`,
      ascii:
        'W = sum over regions r of sum over rows i in r of |x_i - mean_r|^2; T = sum over labelled rows of |x_i - mean|^2; B = T - W; pseudo-F = (B / (k - 1)) / (W / (n - k)), the Calinski-Harabasz score, with k non-empty regions and n labelled rows. Labels at or above labelCapacity and 0xffffffff are ignored. F is 1 when W = 0 and NaN when k < 2 or n <= k. Larger F means better separated regions of the same table.',
      symbols: [
        {id: 'within', role: 'output', binding: 'wgsl:withinSsd'},
        {id: 'total', role: 'output', binding: 'wgsl:totalSsd'},
        {id: 'between', role: 'output', binding: 'wgsl:between'},
        {id: 'pseudoF', role: 'output', binding: 'wgsl:pseudoF'},
        {id: 'regions', role: 'intermediate', binding: 'wgsl:regions'},
        {id: 'labelled', role: 'intermediate', binding: 'wgsl:labelledRows'}
      ]
    }
  },
  GPULocationAllocation: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'pairwise', reduction: 'argmin'},
    variables: {
      demandWeights: 'count',
      costs: 'measure',
      selectedFacilities: 'category',
      assignments: 'category',
      assignedCosts: 'measure',
      objective: 'measure',
      status: 'count'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        name: 'costMetric',
        label: 'Cost between demand and facility',
        stage: 'support',
        alternatives: ['straight-line-km', 'network-time'],
        sweepable: false
      },
      {
        name: 'operation',
        label: 'Objective',
        stage: 'operator',
        alternatives: ['p-median', 'maximum-coverage', 'set-covering'],
        sweepable: true
      },
      {name: 'maximumFacilityCount', label: 'Facilities p', stage: 'operator', sweepable: true},
      {name: 'maximumSwapIterations', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'p-median-objective',
      derivation: 'curated',
      tex: String.raw`\term{objective}{Z}=\sum_{j\ \text{reachable}}\term{weight}{w_j}\,\term{assignedCost}{\min_{i\in S}c_{ji}},\qquad |S|\le p`,
      ascii:
        'Objective of the p-median operation, not the greedy and swap procedure: Z = sum over reachable demand rows j of w_j * min over selected facilities i in S of c_ji, with |S| <= p. Unreachable rows (infinite cost to every selected facility) are left out of Z and counted in the uncovered status word. Maximum coverage instead sums the weight of rows within coverageDistance, and set covering counts selected facilities.',
      symbols: [
        {id: 'objective', role: 'output', binding: 'wgsl:objective'},
        {id: 'weight', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'assignedCost', role: 'intermediate', binding: 'wgsl:assignedCost'}
      ]
    }
  }
};
