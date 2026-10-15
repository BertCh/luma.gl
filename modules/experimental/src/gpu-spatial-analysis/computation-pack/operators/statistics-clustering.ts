// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {SpatialChoiceTemplate} from '../choices';
import type {SpatialFamilyOperators} from '../types';

/** Guard metadata of the planar clustering contributors: positions are metres in a local frame. */
const PLANAR_METRES = {coordinateModel: 'projected', metricUnit: 'm'} as const;

const SEED_CHOICE: SpatialChoiceTemplate = {name: 'seed', stage: 'operator', sweepable: true};

/** Multiple-testing correction of the space-time hot spot decision (read by the uncorrected-p guard). */
const P_CORRECTION_CHOICE: SpatialChoiceTemplate = {
  name: 'pCorrection',
  label: 'Multiple-testing correction',
  stage: 'inference',
  alternatives: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'],
  sweepable: true
};

/** Space-time neighbourhood choices shared by the emerging hot spot element and its recipe. */
const SPACE_TIME_CHOICES: readonly SpatialChoiceTemplate[] = [
  {name: 'radius', label: 'Lattice radius (cells)', stage: 'neighbourhood', sweepable: true},
  {
    name: 'temporalWindow',
    label: 'Previous slices in the window',
    stage: 'neighbourhood',
    sweepable: true
  },
  {name: 'confidenceLevel', stage: 'inference', alternatives: [0.9, 0.95, 0.99], sweepable: true},
  P_CORRECTION_CHOICE,
  {
    name: 'finalSlice',
    label: 'Final slice rule',
    stage: 'operator',
    alternatives: ['require', 'last-valid'],
    sweepable: false
  }
];

/** The Gi* and Mann-Kendall formula of the emerging hot spot analysis, bound to its WGSL. */
const EMERGING_HOT_SPOT_FORMULA: GPUComputationFormulaRef = {
  id: 'emerging-hot-spot-gi-star',
  derivation: 'curated',
  tex: String.raw`\begin{aligned}
\term{giZScores}{z_b}&=\frac{\term{weightedSum}{\sum_j w_j\,(x_j-\bar{X})}}{\term{deviation}{S}\,\sqrt{\term{spread}{\Delta_b}}},\qquad \Delta_b=\frac{n\,S_1-W^2}{n-1}\\
\term{trendS}{S_{\mathrm{MK}}}&=\sum_{i<j}\operatorname{sign}(z_j-z_i),\qquad \term{trendZScore}{Z_{\mathrm{MK}}}=\frac{S_{\mathrm{MK}}-\operatorname{sign}(S_{\mathrm{MK}})}{\sqrt{\mathrm{Var}(S_{\mathrm{MK}})}}\\
&\text{hot: } z_b>0 \text{ and } \term{adjustedP}{\tilde{p}_b}\le\term{level}{1-\mathrm{confidence}}
\end{aligned}`,
  ascii:
    'z_b = sum_j w_j (x_j - mean) / (S sqrt(Delta_b)) over the valid bins j of the cell neighbourhood in the current and temporalWindow previous slices, Delta_b = (n S1 - W^2)/(n - 1), with S the population standard deviation of all n valid bins, W = sum w_j and S1 = sum w_j^2 (binary lattice weights: S1 = W = k, the bins counted). NaN when n < 2, S = 0 or Delta_b <= 0. Per cell, Mann-Kendall S = sum over slice pairs i < j of sign(z_j - z_i) with tie-corrected variance, Z = (S - sign(S)) / sqrt(Var). A bin is hot (cold) when its two-sided normal p-value, adjusted over all valid bins by pCorrection, is at most 1 - confidenceLevel and z > 0 (z < 0). The 17 categories then follow from the hot and cold slice counts and the trend.',
  symbols: [
    {id: 'giZScores', role: 'output', binding: 'wgsl:zScore'},
    {id: 'weightedSum', role: 'intermediate', binding: 'wgsl:weightedSum'},
    {id: 'deviation', role: 'intermediate', binding: 'wgsl:deviation'},
    {id: 'spread', role: 'intermediate', binding: 'wgsl:spread'},
    {id: 'trendS', role: 'output', binding: 'wgsl:statistic'},
    {id: 'trendZScore', role: 'output', binding: 'wgsl:trendZScore'},
    {id: 'adjustedP', role: 'intermediate', binding: 'wgsl:adjustedPValues'},
    {id: 'level', role: 'parameter', binding: 'wgsl:significanceLevel'}
  ]
};

/**
 * Spatial computation-pack entries for the `statistics` family (directions, clustering, groups, similarity, outliers and space-time hot spots), keyed by exported name.
 * Merged into `STATISTICS_OPERATORS`. Every entry must use `family: 'statistics'`.
 */
export const STATISTICS_CLUSTERING_OPERATORS: SpatialFamilyOperators = {
  GPURoseStatistic: {
    family: 'statistics',
    descriptor: {scope: 'global', scopeSecondary: 'zonal', reduction: 'histogram'},
    variables: {
      starts: 'coordinate',
      ends: 'coordinate',
      weights: 'measure',
      summary: 'score',
      bins: 'measure'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'line', label: 'segment', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'mode',
        label: 'Circular mode',
        stage: 'operator',
        alternatives: ['directional', 'axial'],
        sweepable: true
      },
      {name: 'binCount', label: 'Rose bins', stage: 'presentation', sweepable: true}
    ],
    formula: {
      id: 'rose-circular-moments',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{circularAngle}{\phi_i}&=m\,\theta_i,\qquad m=1\ \text{(directional) or } 2\ \text{(axial)}\\
\term{cosineSum}{C}&=\sum_i w_i\cos\phi_i,\qquad \term{sineSum}{S}=\sum_i w_i\sin\phi_i,\qquad \term{totalWeight}{W}=\sum_i w_i\\
\term{resultant}{R}&=\frac{\sqrt{C^2+S^2}}{W},\qquad \term{meanAngle}{\bar{\theta}}=\tfrac{1}{m}\operatorname{atan2}(S,C),\qquad V=1-R\\
\term{rayleighZ}{z}&=\frac{(\sum_i\cos\phi_i)^2+(\sum_i\sin\phi_i)^2}{\term{validCount}{n}}=n\,\bar{R}_0^2
\end{aligned}`,
      ascii:
        'theta_i = atan2(dy, dx) of segment i, in [0, 2 pi) (axial: folded into [0, pi)); phi_i = m theta_i with m = 1 (directional) or 2 (axial). C = sum w_i cos(phi_i), S = sum w_i sin(phi_i), W = sum w_i over segments with positive finite length and weight. Mean angle = atan2(S, C) / m, resultant length R = sqrt(C^2 + S^2) / W, circular variance V = 1 - R, Rayleigh z = ((sum cos(phi_i))^2 + (sum sin(phi_i))^2) / n = n Rbar_0^2, using each of the n valid directions once, independently of descriptive weights. Rbar_0 is the unweighted resultant, distinct from the weighted R. This is not giddy Rose permutation inference; no p-value is returned. Bins are left-closed equal-width sectors of the period (2 pi, or pi when axial) holding the summed weight. When no valid segment remains, count and bins are zero and the other summary fields are NaN.',
      symbols: [
        {id: 'circularAngle', role: 'intermediate', binding: 'wgsl:circularAngle'},
        {id: 'cosineSum', role: 'intermediate', binding: 'wgsl:sharedCosine'},
        {id: 'sineSum', role: 'intermediate', binding: 'wgsl:sharedSine'},
        {id: 'totalWeight', role: 'intermediate', binding: 'wgsl:totalWeight'},
        {id: 'resultant', role: 'output', binding: 'wgsl:resultant'},
        {id: 'meanAngle', role: 'output', binding: 'wgsl:meanAngle'},
        {id: 'rayleighZ', role: 'output', binding: 'wgsl:rayleighZ'},
        {id: 'validCount', role: 'intermediate', binding: 'wgsl:validCount'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUSpatialClustering: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'solver', reduction: 'count'},
    variables: {
      positions: 'coordinate',
      labels: 'category',
      coreFlags: 'category',
      clusterCount: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'epsilon',
        label: 'Neighbour radius',
        stage: 'neighbourhood',
        unit: 'm',
        sweepable: true
      },
      {name: 'minimumPoints', label: 'Minimum points', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'dbscan-core-test',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{neighborCount}{\lvert N_\varepsilon(i)\rvert}&=\sum_j \bigl[\,\lVert x_j-x_i\rVert^2\le\term{epsilonSquared}{\varepsilon^2}\,\bigr]\\
\term{coreFlags}{\mathrm{core}(i)}&\iff \lvert N_\varepsilon(i)\rvert\ \ge\ \term{minimumPoints}{m}
\end{aligned}`,
      ascii:
        'Core test, not the labelling: |N(i)| = number of valid points j (point i itself included) with dx^2 + dy^2 <= epsilon^2 in f32; point i is core when |N(i)| >= minimumPoints. Clusters are the connected components of core points under that relation, found by lock-free union-find; a border point (valid, not core, with a core neighbour) joins the adjacent cluster with the smallest root, every other point is noise (0xffffffff). Compact cluster ids follow the smallest core row, so labels are canonical.',
      symbols: [
        {id: 'neighborCount', role: 'intermediate', binding: 'wgsl:neighborCount'},
        {id: 'epsilonSquared', role: 'parameter', binding: 'wgsl:epsilonSquared'},
        {id: 'coreFlags', role: 'output', binding: 'wgsl:coreFlags'},
        {id: 'minimumPoints', role: 'parameter', binding: 'wgsl:minimumPoints'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUKMeans: {
    family: 'statistics',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'solve'},
    variables: {
      positions: 'coordinate',
      labels: 'category',
      centers: 'coordinate',
      sizes: 'count',
      squaredDistances: 'measure',
      inertia: 'measure',
      iterationCount: 'count',
      converged: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'k', label: 'Clusters k', stage: 'operator', sweepable: true},
      {name: 'maximumIterations', label: 'Lloyd iterations', stage: 'operator', sweepable: true},
      {
        name: 'initialization',
        stage: 'operator',
        alternatives: ['kmeans++', 'first-valid'],
        sweepable: true
      },
      {name: 'restartCount', label: 'Restarts', stage: 'operator', sweepable: true},
      SEED_CHOICE
    ],
    formula: {
      id: 'kmeans-inertia',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{inertia}{J}&=\sum_i \term{distance}{\lVert x_i-c_{\ell(i)}\rVert^2},\qquad \ell(i)=\arg\min_k\lVert x_i-c_k\rVert^2\\
\term{center}{c_k}&\leftarrow\frac{\sum_{i:\,\ell(i)=k}x_i}{\term{size}{n_k}}
\end{aligned}`,
      ascii:
        'Objective, not the procedure: inertia J = sum over valid points i of |x_i - c_l(i)|^2, with l(i) the nearest center (lowest center id wins ties) in f32. Each Lloyd iteration reassigns every point, then moves each center to the mean of its members, c_k = sum x_i / n_k, summed in fixed order. A center that gets no point is relocated to the points farthest from their centers. Initial centers are k-means++ (seeded) or the first k valid points; with restarts the lowest J wins.',
      symbols: [
        {id: 'inertia', role: 'output', binding: 'wgsl:total'},
        {id: 'distance', role: 'intermediate', binding: 'wgsl:bestDistance'},
        {id: 'center', role: 'intermediate', binding: 'wgsl:nextX'},
        {id: 'size', role: 'intermediate', binding: 'wgsl:size'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUHierarchicalDensityClustering: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', scopeSecondary: 'solver', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      labels: 'category',
      coreDistances: 'measure',
      clusterCount: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'minimumClusterSize',
        label: 'Minimum cluster size',
        stage: 'operator',
        sweepable: true
      },
      {
        name: 'minimumSamples',
        label: 'Core distance neighbours',
        stage: 'neighbourhood',
        sweepable: true
      }
    ],
    formula: {
      id: 'hdbscan-mutual-reachability',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{coreDistance}{d_{\mathrm{core}}(a)}&=\text{distance to the } \term{minimumSamples}{s}\text{-th nearest point, itself counted}\\
\term{mutual}{d_{\mathrm{mr}}(a,b)}&=\max\bigl(d_{\mathrm{core}}(a),\ d_{\mathrm{core}}(b),\ \lVert x_a-x_b\rVert\bigr)\\
\term{stability}{S(C)}&=\sum_{p\in C}\bigl(\lambda_p-\term{birth}{\lambda_{\mathrm{birth}}(C)}\bigr),\qquad \lambda=1/d
\end{aligned}`,
      ascii:
        'Pieces of the pipeline, not the whole procedure: core distance = distance to the minimumSamples-th nearest point (the point counts as its own first neighbour); mutual reachability d_mr(a, b) = max(core(a), core(b), |x_a - x_b|); the minimum spanning tree over d_mr gives the single-linkage dendrogram, which is condensed to clusters of at least minimumClusterSize points. Stability S(C) = sum over points p of C of (lambda_p - lambda_birth(C)) with lambda = 1 / distance. Excess-of-mass selection keeps a cluster when its stability exceeds the summed stability of its selected descendants; every other point is noise (0xffffffff).',
      symbols: [
        {id: 'coreDistance', role: 'intermediate', binding: 'wgsl:coreDistances'},
        {id: 'minimumSamples', role: 'parameter', binding: 'wgsl:MINIMUM_SAMPLES'},
        {id: 'mutual', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'stability', role: 'intermediate', binding: 'wgsl:contributions'},
        {id: 'birth', role: 'intermediate', binding: 'wgsl:birth'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUGroupGeometry: {
    family: 'statistics',
    descriptor: {scope: 'zonal', scopeSecondary: 'pairwise', reduction: 'weighted-mean'},
    variables: {
      positions: 'coordinate',
      labels: 'category',
      counts: 'count',
      bounds: 'coordinate',
      meanCenters: 'coordinate',
      medoidIndices: 'category',
      standardDistances: 'measure',
      ellipses: 'measure'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'groupCount', label: 'Group slots', stage: 'operator', sweepable: false},
      {
        name: 'medoidMaximumGroupSize',
        label: 'Largest group with a medoid',
        stage: 'operator',
        sweepable: false
      }
    ],
    formula: {
      id: 'group-geometry-centre-medoid',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{meanCenters}{\bar{p}_g}&=\frac{1}{\term{counts}{n_g}}\sum_{i\in g}p_i\\
\term{medoidCost}{D_i}&=\sum_{j\in g}\lVert p_i-p_j\rVert,\qquad \term{medoid}{m_g}=\arg\min_{i\in g}D_i
\end{aligned}`,
      ascii:
        'Per label group g over the rows with a finite position and a label below groupCount: n_g = rows in g, mean center = sum p_i / n_g, bounds = exact min and max of x and y, medoid = the row i minimizing D_i = sum over rows j of g of |p_i - p_j| (lowest row on ties; groups over medoidMaximumGroupSize get no medoid). The standard distance and the standard deviational ellipse come from GPUGeographicDistribution on the same groups. Sums are fixed-order, so results are reproducible.',
      symbols: [
        {id: 'meanCenters', role: 'output', binding: 'wgsl:meanCenters'},
        {id: 'counts', role: 'output', binding: 'wgsl:counts'},
        {id: 'medoidCost', role: 'intermediate', binding: 'wgsl:cost'},
        {id: 'medoid', role: 'output', binding: 'wgsl:medoids'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUGroupConvexHull: {
    family: 'statistics',
    descriptor: {scope: 'zonal', scopeSecondary: 'frontier', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      labels: 'category',
      groupIds: 'category',
      vertexIndices: 'category',
      vertexPositions: 'coordinate',
      offsets: 'count',
      counts: 'count',
      sizes: 'count',
      overflow: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'groupCount', label: 'Group slots', stage: 'operator', sweepable: false},
      {
        name: 'maximumVerticesPerGroup',
        label: 'Largest hull',
        stage: 'operator',
        sweepable: false
      },
      {name: 'extremePointFilter', stage: 'operator', alternatives: [true, false], sweepable: false}
    ],
    formula: {
      id: 'convex-hull-orientation',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{lattice}{q_i}&=\lfloor \term{scale}{s}\,(p_i-p_{\min})+\tfrac{1}{2}\rfloor\\
\term{orientation}{\operatorname{orient}(a,b,c)}&=\operatorname{sign}\bigl((b_x-a_x)(c_y-a_y)-(b_y-a_y)(c_x-a_x)\bigr)
\end{aligned}`,
      ascii:
        'Predicate of the monotone chain, not the chain: coordinates are snapped to an integer lattice q = floor(s (p - p_min) + 1/2) with one power-of-two scale s so the data extent spans 2^29 steps, and orient(a, b, c) = sign((b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)) is evaluated exactly with emulated 64-bit integers. Rows are sorted by (label, x, y); the chain pops the last vertex while orient <= 0, so collinear points are never hull vertices. The vertices returned are the original float32 positions, counter-clockwise from the smallest (x, y).',
      symbols: [
        {id: 'lattice', role: 'intermediate', binding: 'wgsl:latticeX'},
        {id: 'scale', role: 'parameter', binding: 'wgsl:scale'},
        {id: 'orientation', role: 'output', binding: 'wgsl:orientationOfPoints'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUSimilarLocations: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'scan-sort'},
    variables: {
      values: 'rate',
      referenceMask: 'category',
      distances: 'score',
      ranks: 'rank',
      topIds: 'category',
      topCount: 'count'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        name: 'standardization',
        label: 'Column standardization',
        stage: 'support',
        alternatives: ['zscore', 'rank'],
        sweepable: true
      },
      {name: 'resultCount', label: 'Rows reported', stage: 'presentation', sweepable: false},
      {
        name: 'direction',
        label: 'Most or least similar',
        stage: 'operator',
        alternatives: ['most', 'least'],
        sweepable: true
      },
      {name: 'excludeReference', stage: 'operator', alternatives: [true, false], sweepable: false}
    ],
    formula: {
      id: 'similar-locations-distance',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{standardized}{z_{ic}}&=\frac{x_{ic}-\bar{x}_c}{\term{deviation}{s_c}},\qquad \term{reference}{r_c}=\frac{1}{\lvert R\rvert}\sum_{i\in R}z_{ic}\\
\term{distance}{d_i}&=\sqrt{\sum_c \term{weight}{w_c}\,\bigl(z_{ic}-r_c\bigr)^2}
\end{aligned}`,
      ascii:
        'Per attribute column c over the valid rows: z_ic = (x_ic - mean_c) / s_c with s_c the population standard deviation (0 for a constant column); with standardization rank, z is the average-tie percentile rank (rank - 1) / (n - 1) instead. The reference vector r_c is the mean of the standardized values of the selected reference rows R. Distance d_i = sqrt(sum_c w_c (z_ic - r_c)^2) with non-negative column weights w_c. Rows are ranked by d (lowest row on ties; rank 0 is the most similar, or the least with direction least); invalid, masked and reference rows get no rank and a NaN distance.',
      symbols: [
        {id: 'standardized', role: 'intermediate', binding: 'wgsl:standardized'},
        {id: 'deviation', role: 'intermediate', binding: 'wgsl:deviation'},
        {id: 'reference', role: 'intermediate', binding: 'wgsl:reference'},
        {id: 'distance', role: 'output', binding: 'wgsl:distance'},
        {id: 'weight', role: 'parameter', binding: 'wgsl:weight'}
      ]
    }
  },
  GPULocalOutlierFactor: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'pairwise', reduction: 'weighted-mean'},
    variables: {
      positions: 'coordinate',
      kDistance: 'measure',
      localReachabilityDensity: 'rate',
      lof: 'score',
      outlier: 'category',
      outlierCount: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'point', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'k', label: 'Neighbours k', stage: 'neighbourhood', sweepable: true},
      {name: 'threshold', label: 'Outlier cut', stage: 'presentation', sweepable: true},
      {
        name: 'densityFloor',
        label: 'Density floor',
        stage: 'operator',
        unit: 'm',
        sweepable: true
      }
    ],
    formula: {
      id: 'local-outlier-factor',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{kDistance}{d_k(j)}&=\max_{o\in N(j)}d(j,o),\qquad \term{reach}{\mathrm{reach}(i,j)}=\max\bigl(d_k(j),\ d(i,j)\bigr)\\
\term{density}{\mathrm{lrd}(i)}&=\frac{1}{\overline{\mathrm{reach}}(i)+\term{densityFloor}{\epsilon}}\\
\term{lof}{\mathrm{LOF}(i)}&=\frac{\overline{\mathrm{lrd}}\bigl(N(i)\bigr)}{\mathrm{lrd}(i)}
\end{aligned}`,
      ascii:
        'Over the kNN table: k-distance d_k(j) = largest neighbour distance of row j; reach(i, j) = max(d_k(j), d(i, j)); local reachability density lrd(i) = 1 / (mean over neighbours j of reach(i, j) + densityFloor); LOF(i) = mean over neighbours j of lrd(j) / lrd(i). A row with no neighbours gets density 0 and LOF 1. Scores near 1 are as dense as their neighbours; well above 1 is locally sparse. There is no null distribution, so the outlier mask uses a threshold the caller chooses.',
      symbols: [
        {id: 'kDistance', role: 'intermediate', binding: 'wgsl:kDistance'},
        {id: 'reach', role: 'intermediate', binding: 'wgsl:distances'},
        {id: 'density', role: 'intermediate', binding: 'wgsl:density'},
        {id: 'densityFloor', role: 'parameter', binding: 'wgsl:parameters'},
        {id: 'lof', role: 'output', binding: 'wgsl:lof'}
      ]
    },
    metadata: PLANAR_METRES
  },
  GPUEmergingHotSpots: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      cube: 'count',
      giZScores: 'score',
      adjustedPValues: 'probability',
      trendZ: 'score',
      trendP: 'probability',
      trendS: 'score',
      category: 'category',
      hotSliceCount: 'count',
      coldSliceCount: 'count'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'cell', label: 'lattice cell', indexSpace: '2d'}],
    guards: ['uncorrected-p'],
    choices: SPACE_TIME_CHOICES,
    formula: EMERGING_HOT_SPOT_FORMULA
  },
  addSpaceTimeHotSpotsRecipe: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      cube: 'count',
      giZScores: 'score',
      trendZ: 'score',
      category: 'category',
      hotSliceCount: 'count',
      coldSliceCount: 'count'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'cell', label: 'lattice cell', indexSpace: '2d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['uncorrected-p'],
    choices: [
      {
        name: 'step',
        label: 'Time step',
        stage: 'support',
        alternatives: ['year', 'quarter', 'month', 'week', 'day', 'hour'],
        sweepable: true
      },
      {name: 'cellSize', label: 'Cell size', stage: 'support', unit: 'm', sweepable: true},
      ...SPACE_TIME_CHOICES
    ],
    formula: {
      id: 'space-time-cube-binning',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{slice}{t_e}&=\left\lfloor\frac{\term{stepOffset}{\sigma(\mathrm{time}_e)-\sigma(\mathrm{origin})}}{I}\right\rfloor,\qquad \term{cell}{c_e}=\lfloor (y_e-y_{\min})/h\rfloor\,W+\lfloor (x_e-x_{\min})/h\rfloor\\
\term{cube}{N_{c,t}}&=\sum_e\,[\,c_e=c\,]\,[\,t_e=t\,]
\end{aligned}`,
      ascii:
        'Binning step of the recipe: sigma counts whole calendar steps (month: year * 12 + month - 1) of the local event time, the slice is floor((sigma(time) - sigma(origin)) / I) for I steps per slice, the cell is the lattice cell of the planar position (cell width h = bounds width / W columns, row 0 southmost), and N[c, t] counts the events of cell c and slice t (integer atomics, so the cube is exact). Events outside the lattice or the slice range are skipped and counted. The cube then goes through GPUEmergingHotSpots: Gi* z per bin, a per-cell Mann-Kendall trend and the 17 emerging categories.',
      symbols: [
        {id: 'slice', role: 'intermediate', binding: 'wgsl:slice'},
        {id: 'stepOffset', role: 'intermediate', binding: 'wgsl:stepOffset'},
        {id: 'cell', role: 'intermediate', binding: 'wgsl:cell'},
        {id: 'cube', role: 'output', binding: 'wgsl:cubeOutput'}
      ]
    },
    metadata: PLANAR_METRES
  }
};
