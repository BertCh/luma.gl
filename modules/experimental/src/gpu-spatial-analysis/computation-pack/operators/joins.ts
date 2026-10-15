// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Recorded operator contracts for the joins family. */
export const JOINS_OPERATORS: SpatialFamilyOperators = {
  GPUSpatialPredicateJoin: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {leftIds: 'category', rightIds: 'category', tractMatches: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'tract', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'predicate',
        sweepable: false,
        alternatives: [
          'intersects',
          'contains',
          'within',
          'covers',
          'coveredBy',
          'touches',
          'crosses',
          'overlaps',
          'equals',
          'containsProperly',
          'dwithin',
          'relate'
        ]
      },
      {
        stage: 'operator',
        name: 'engine',
        sweepable: false,
        alternatives: ['auto', 'fast', 'relate']
      }
    ],
    // spatial-relate-types.ts getPredicateRelatePatterns / getRelateMatchesWGSL; spatial-relate-wgsl.ts relateWorkgroupPair.
    formula: {
      id: 'gpu-spatial-analysis.spatial-predicate-join',
      tex: String.raw`\term{hit}{\operatorname{intersects}} = [\term{matrix}{M_{II}}\ne F\;\lor\;\term{matrix}{M_{IB}}\ne F\;\lor\;\term{matrix}{M_{BI}}\ne F\;\lor\;\term{matrix}{M_{BB}}\ne F]`,
      ascii:
        'intersects iff at least one interior/interior, interior/boundary, boundary/interior, boundary/boundary DE-9IM cell is nonempty (T******** OR *T******* OR ***T***** OR ****T****).',
      symbols: [
        {id: 'hit', role: 'output', label: 'intersects', binding: 'wgsl:relateMatches'},
        {id: 'matrix', role: 'intermediate', label: 'DE-9IM cell', binding: 'wgsl:m'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUSpatialJoinPrepared: {
    family: 'joins',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {tractMatches: 'count', candidateRight: 'category', rightIds: 'category'},
    stages: ['representation', 'neighbourhood'],
    datumKinds: [{kind: 'tract', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'representation', name: 'spatialSort', sweepable: false},
      {stage: 'representation', name: 'edgeSlabs', sweepable: false}
    ],

    formula: {
      formula: 'none',
      reason:
        'Build and reuse persistent bounds and BVH storage; each consuming join recomputes its candidates and predicate results.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUSpatialJoinCandidates: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {leftIds: 'category', rightIds: 'category', tractMatches: 'count'},
    stages: ['neighbourhood'],
    datumKinds: [{kind: 'tract', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'neighbourhood', name: 'distance', sweepable: false, unit: 'm'}],

    formula: {
      formula: 'none',
      reason:
        'Traverse the right-side BVH with left bounding boxes, then publish sorted candidate pairs without exact refinement.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUPointInPolygonJoin: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {tractIds: 'category', featureCounts: 'count', referenceTract: 'category'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'boundaryRule',
        sweepable: false,
        alternatives: ['half-open', 'closed', 'open']
      },
      {stage: 'representation', name: 'edgeSlabs', sweepable: false}
    ],
    // gpu-point-in-polygon-join.ts classifyExactly and assign pass; half-open crossing parity, not a winding-number sum.
    formula: {
      id: 'gpu-spatial-analysis.point-in-polygon-join',
      tex: String.raw`\term{inside}{P_j(\mathbf{x}_{\epsilon})}=\bigoplus_{\term{edge}{e}\in\partial\term{polygon}{j}}\term{crossing}{C_e(\mathbf{x}_{\epsilon})},\quad \term{owner}{a(\mathbf{x})}=\min\{\term{polygon}{j}:\term{inside}{P_j(\mathbf{x}_{\epsilon})}=1\}`,
      ascii:
        'Half-open: evaluate crossing parity at x_epsilon = x + (epsilon^2, epsilon), epsilon infinitesimal. Toggle when endpoint y sides differ and the exact orientation (ties: sign(dx), then sign(-dy)) agrees with edge direction. XOR rings within a part, OR parts; assign the smallest qualifying feature row. Uncertain candidates are not assigned; no owner is 0xffffffff.',
      symbols: [
        {
          id: 'inside',
          role: 'intermediate',
          label: 'crossing parity',
          binding: 'wgsl:polygonOwned'
        },
        {id: 'edge', role: 'input', label: 'edge endpoints', binding: 'wgsl:current'},
        {id: 'polygon', role: 'input', label: 'polygon feature', binding: 'wgsl:featureRow'},
        {
          id: 'crossing',
          role: 'intermediate',
          label: 'exact half-open crossing',
          binding: 'wgsl:side'
        },
        {id: 'owner', role: 'output', label: 'assigned feature', binding: 'wgsl:pointFeatureIds'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUNearestFeatureJoin: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {neighborIds: 'category', neighborDistances: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'k', sweepable: false},
      {stage: 'operator', name: 'ties', sweepable: false, alternatives: ['lowest-id', 'all']}
    ],
    // nearest-wgsl.ts getNearestPairWGSL / getNearestTraversalWGSL; nearest-nodes.ts columns takes square roots.
    formula: {
      id: 'gpu-spatial-analysis.nearest-feature-join',
      tex: String.raw`\term{nearest}{j^*(i)}=\underset{\term{feature}{j}}{\operatorname{argmin}}\;(\term{distance}{\lVert\mathbf{x}_i-\mathbf{f}_j\rVert_2^2},\term{feature}{j})`,
      ascii:
        'For this planar point-to-point k=1 run, choose the lexicographically smallest (squared Euclidean distance, feature row); publish sqrt(distanceSquared) in metres. General geometry uses pairResult minimum edge distance and containment distance zero.',
      symbols: [
        {id: 'nearest', role: 'output', label: 'nearest library', binding: 'wgsl:neighborIds'},
        {id: 'feature', role: 'input', label: 'feature row', binding: 'wgsl:featureRow'},
        {
          id: 'distance',
          role: 'intermediate',
          label: 'squared planar distance',
          binding: 'wgsl:pairResult'
        }
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUNearestFeatureWeights: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {weights: 'score', distances: 'measure', neighbors: 'category'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'k', sweepable: false},
      {
        stage: 'operator',
        name: 'weightType',
        sweepable: false,
        alternatives: ['binary', 'inverse-distance']
      },
      {stage: 'operator', name: 'power', sweepable: false},
      {stage: 'operator', name: 'distanceFloor', sweepable: false, unit: 'm'},
      {stage: 'operator', name: 'excludeSelf', sweepable: false}
    ],
    // gpu-nearest-feature-weights.ts emit kernel: safeDistance, pow, and finite-weight guard.
    formula: {
      id: 'gpu-spatial-analysis.nearest-feature-weights',
      tex: String.raw`\term{weight}{w_{ij}}=\begin{cases}\term{raw}{q_{ij}}&\term{raw}{q_{ij}}<3\cdot10^{38}\\0&\text{otherwise}\end{cases},\quad\term{raw}{q_{ij}}=\max(\term{distance}{d_{ij}},\term{floor}{\delta})^{-\term{power}{p}}`,
      ascii:
        'Retain first k valid neighbors, optionally exclude self, then sort IDs ascending in CSR. In inverse-distance mode q = 1 / max(distance, distanceFloor)^power; store q when q < 3e38, otherwise zero. Binary mode stores 1. No row standardisation.',
      symbols: [
        {id: 'weight', role: 'output', label: 'stored weight', binding: 'wgsl:values'},
        {id: 'raw', role: 'intermediate', label: 'raw weight', binding: 'wgsl:weight'},
        {id: 'distance', role: 'input', label: 'join distance', binding: 'wgsl:keptDistances'},
        {id: 'floor', role: 'parameter', label: 'distance floor', binding: 'wgsl:FLOOR'},
        {id: 'power', role: 'parameter', label: 'power', binding: 'wgsl:POWER'}
      ],
      derivation: 'curated'
    },
    metadata: {
      coordinateModel: 'projected',
      metricUnit: 'm',
      projection: 'azimuthal-equidistant',
      weightsStandardisation: 'other'
    }
  },
  GPUPairGather: {
    family: 'joins',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {
      parkAreas: 'measure',
      joinedAreas: 'measure',
      parkRows: 'category',
      tractRows: 'category'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'tract', label: 'tract', indexSpace: '1d'}],
    guards: [],
    choices: [{stage: 'operator', name: 'how', sweepable: false, alternatives: ['inner', 'left']}],

    formula: {
      formula: 'none',
      reason:
        'Gather attribute columns by matched left and right row indices, bounded by the output row capacity.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUOffsetExpansion: {
    family: 'joins',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {owners: 'category', localIndex: 'rank', offsets: 'count'},
    stages: ['representation'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: [],
    choices: [],

    formula: {
      formula: 'none',
      reason:
        'Binary-search monotone offsets for each child row to emit its owner and its index within that owner.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUBoundsFilter: {
    family: 'joins',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {bounds: 'measure', selected: 'category', count: 'count'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'mode',
        sweepable: false,
        alternatives: ['intersects', 'within', 'contains']
      }
    ],
    // gpu-bounds-filter.ts mask kernel, comparison.within (closed inequalities).
    formula: {
      id: 'gpu-spatial-analysis.bounds-filter',
      tex: String.raw`\term{selected}{s}=\mathbf{1}[\term{finite}{\mathrm{finite}(F)}\land\term{minimum}{F_{\min,x}}\ge\term{boxMin}{B_{\min,x}}\land\term{maximum}{F_{\max,x}}\le\term{boxMax}{B_{\max,x}}\land\term{minimum}{F_{\min,y}}\ge\term{boxMin}{B_{\min,y}}\land\term{maximum}{F_{\max,y}}\le\term{boxMax}{B_{\max,y}}]`,
      ascii:
        'Within mode: finite bounds AND fMin.x >= boxMin.x AND fMax.x <= boxMax.x AND fMin.y >= boxMin.y AND fMax.y <= boxMax.y. All comparisons are inclusive, directly in f32.',
      symbols: [
        {id: 'selected', role: 'output', label: 'mask', binding: 'wgsl:maskOut'},
        {id: 'finite', role: 'intermediate', label: 'finite bounds', binding: 'wgsl:finite'},
        {id: 'minimum', role: 'input', label: 'feature minimum', binding: 'wgsl:fMin'},
        {id: 'maximum', role: 'input', label: 'feature maximum', binding: 'wgsl:fMax'},
        {id: 'boxMin', role: 'parameter', label: 'box minimum', binding: 'wgsl:boxMin'},
        {id: 'boxMax', role: 'parameter', label: 'box maximum', binding: 'wgsl:boxMax'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  },
  GPUBufferSelection: {
    family: 'joins',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {selected: 'category', distances: 'measure'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'observation', label: 'observation', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'neighbourhood', name: 'distance', sweepable: false, unit: 'm'}],
    // gpu-nearest-feature-join.ts radius acceptance; gpu-buffer-selection.ts mask kernel converts valid feature IDs to 1.
    formula: {
      id: 'gpu-spatial-analysis.buffer-selection',
      tex: String.raw`\term{selected}{s_i}=\mathbf{1}[0\le\min_{\term{feature}{j}}\term{distance}{\lVert\mathbf{x}_i-\mathbf{f}_j\rVert_2}\le\term{radius}{r}]`,
      ascii:
        'For finite nonnegative radius r, select a point iff its nearest-library planar distance is >= 0 and <= r; r=300 metres here. Invalid radii select nothing; unmatched output distance is -1.',
      symbols: [
        {id: 'selected', role: 'output', label: 'selection mask', binding: 'wgsl:maskOut'},
        {id: 'feature', role: 'input', label: 'feature', binding: 'wgsl:featureIds'},
        {id: 'distance', role: 'intermediate', label: 'planar distance', binding: 'wgsl:distance'},
        {id: 'radius', role: 'parameter', label: 'buffer radius', binding: 'wgsl:radius'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', projection: 'azimuthal-equidistant'}
  }
};
