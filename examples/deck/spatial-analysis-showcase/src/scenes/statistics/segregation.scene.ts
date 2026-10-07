// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {SegregationOptions} from './segregation.compute';

const GROUP_LEGEND = [
  {color: [213, 94, 0, 235], label: 'Black'},
  {color: [86, 180, 233, 235], label: 'White'},
  {color: [0, 158, 115, 235], label: 'Hispanic'},
  {color: [240, 228, 66, 235], label: 'Asian'},
  {color: [150, 150, 150, 235], label: 'Other or multiracial'}
] as const;

const GROUP_NAMES: Record<string, string> = {
  nhBlack: 'Black',
  nhWhite: 'White',
  hispanic: 'Hispanic',
  nhAsian: 'Asian',
  nhOther: 'Other'
};

export default defineScene<SegregationOptions>({
  id: 'segregation',
  title: 'How segregated is Chicago, and at what scale?',
  chapter: 'statistics',
  order: 3,
  summary:
    'Multigroup segregation indices for Chicago tracts: dissimilarity, entropy, isolation, interaction and Atkinson, global and local, aspatial and across a ladder of neighbourhood scales.',
  contributors: ['GPUSegregation', 'GPUNeighborSearch', 'GPUSpatialWeightsTransform'],
  datasets: [
    {id: 'chicago-tracts', role: '791 tracts with race and ethnicity counts (SVI 2022)'},
    {id: 'chicago-community-areas', role: 'community area names for tooltips'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},

  options: [
    {
      kind: 'select',
      id: 'localIndex',
      label: 'Map shows',
      group: 'Map',
      apply: 'param',
      default: 'dominant',
      help: 'A display kernel picks one local quantity per tract from the outputs of GPUSegregation; switching is a parameter write.',
      options: [
        {value: 'dominant', label: 'Largest group (categorical)'},
        {
          value: 'composition',
          label: 'Focal group share of the local environment',
          help: 'The environment is the distance-weighted population around the tract.'
        },
        {
          value: 'dissimilarity',
          label: 'Contribution to the focal group dissimilarity D',
          help: 'Tract terms add up to the global D of the group.'
        },
        {
          value: 'entropy',
          label: 'Local diversity (entropy)',
          help: 'High where all groups share the environment.'
        },
        {
          value: 'theil',
          label: 'Contribution to the entropy index H',
          help: 'Positive where the tract is less diverse than the city, negative where it is more diverse.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'focalGroup',
      label: 'Focal group',
      group: 'Map',
      apply: 'param',
      default: 'nhBlack',
      help: 'The group whose composition, dissimilarity and isolation are mapped and read out.',
      disabledWhen: state =>
        state.localIndex === 'dominant' ||
        state.localIndex === 'entropy' ||
        state.localIndex === 'theil',
      options: Object.entries(GROUP_NAMES).map(([value, label]) => ({value, label}))
    },
    {
      kind: 'slider',
      id: 'scale',
      label: 'Scale of the environment',
      group: 'Scale',
      apply: 'param',
      min: 0,
      max: 5,
      step: 1,
      default: 2,
      format: value => (value === 0 ? 'aspatial' : `scale ${value} of 5`),
      help: '0 is the aspatial index (every tract on its own). 1 to 5 widen the neighbourhood: each scale is a distance band 1, 2, 4, 8 and 16 times the bandwidth below. The profile readout compares all scales.'
    },
    {
      kind: 'slider',
      id: 'bandwidth',
      label: 'Base bandwidth',
      group: 'Scale',
      apply: 'param',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      default: 1,
      unit: 'km',
      help: 'Radius of scale 1. The neighbour search re-runs with the new radius; nothing is recompiled.'
    },
    {
      kind: 'select',
      id: 'weights',
      label: 'Distance weighting',
      group: 'Environment weights',
      apply: 'compile',
      default: 'band',
      help: 'How neighbours count in the environment. Binary band: every neighbour inside the radius counts once. The kernels (GPUSpatialWeightsTransform, adaptive bandwidth) give nearer tracts more weight. Compile-time: compiles another graph on demand.',
      options: [
        {value: 'band', label: 'Binary distance band'},
        {value: 'epanechnikov', label: 'Epanechnikov kernel'},
        {value: 'gaussian', label: 'Gaussian kernel'},
        {value: 'bisquare', label: 'Bisquare kernel'},
        {value: 'triangular', label: 'Triangular kernel'},
        {value: 'uniform', label: 'Uniform kernel'}
      ]
    },
    {
      kind: 'toggle',
      id: 'rowStandardize',
      label: 'Row-standardise weights',
      group: 'Environment weights',
      apply: 'compile',
      default: false,
      help: "Rescale each tract's neighbour weights to sum to 1, so the environment is a weighted average rather than a sum. Compile-time."
    },
    {
      kind: 'select',
      id: 'selfWeight',
      label: 'Weight of a tract in its own environment',
      group: 'Environment weights',
      apply: 'compile',
      default: '1',
      help: 'GPUSegregation adds this weight for the tract itself; neighbour lists never include it. 0 excludes the tract. Compile-time.',
      options: [
        {value: '0', label: '0 (excluded)'},
        {value: '0.5', label: '0.5'},
        {value: '1', label: '1 (default)'},
        {value: '2', label: '2'}
      ]
    },
    {
      kind: 'select',
      id: 'spatialForm',
      label: 'Spatial form',
      group: 'Index definition',
      apply: 'compile',
      default: 'environment',
      help: "Environment (Reardon and O'Sullivan): sums keep actual tract populations, only the compositions come from the environment. Smoothed population: the smoothed table replaces the population (what PySAL spatially implicit indices do). Compile-time.",
      options: [
        {value: 'environment', label: "Environment (Reardon-O'Sullivan)"},
        {value: 'smoothed-population', label: 'Smoothed population (PySAL style)'}
      ]
    },
    {
      kind: 'select',
      id: 'atkinsonB',
      label: 'Atkinson parameter b',
      group: 'Index definition',
      apply: 'compile',
      default: '0.5',
      help: 'Inequality aversion of the Atkinson index. Small b weights the middle of the distribution, b near 1 the tails. Compile-time.',
      options: [
        {value: '0.1', label: '0.1'},
        {value: '0.25', label: '0.25'},
        {value: '0.5', label: '0.5 (default)'},
        {value: '0.75', label: '0.75'},
        {value: '0.9', label: '0.9'}
      ]
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'Stronger tract outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Outline every tract more boldly.'
    }
  ],

  readouts: [
    {id: 'tracts', label: 'Tracts', format: 'integer'},
    {id: 'scaleLabel', label: 'Environment'},
    {
      id: 'entropy',
      label: 'Multigroup indices',
      help: 'H: information-theory (entropy) index. D: multigroup dissimilarity. Diversity: the entropy of the citywide composition in nats.'
    },
    {
      id: 'group',
      label: 'Focal group indices',
      help: 'Dissimilarity D of the group against everyone else, isolation (probability the average group member meets another member), and Atkinson.'
    },
    {
      id: 'interaction',
      label: 'Exposure of the focal group to',
      help: "Interaction index: share of the average focal-group member's environment that belongs to each other group. Isolation plus these sum to 1."
    },
    {
      id: 'profile',
      label: 'Multiscale profile of multigroup D',
      help: 'Segregation falls as the environment widens because every neighbourhood becomes more mixed.'
    },
    {id: 'profileGroup', label: 'Multiscale profile of the focal group D'},
    {
      id: 'localSum',
      label: 'Check: local terms add up',
      help: 'The per-tract maps are the terms of the sums: they add up to the global index of the same scale.'
    },
    {id: 'overflow', label: 'Neighbour capacity overflow'},
    {id: 'selected', label: 'Selected tract', help: 'Click a tract to pin its numbers.'}
  ],

  legends: state => {
    if (state.localIndex === 'dominant') {
      return [
        {
          kind: 'categories',
          title: 'Largest group in the tract',
          entries: GROUP_LEGEND,
          note: 'Aspatial view of the data: scale does not change this map.'
        }
      ];
    }
    const group = GROUP_NAMES[state.focalGroup] ?? 'the focal group';
    if (state.localIndex === 'composition') {
      return [
        {
          kind: 'ramp',
          title: `${group} share of the local environment`,
          ramp: 'magma',
          extent: [0, 1],
          format: value => `${Math.round(value * 100)}%`
        }
      ];
    }
    if (state.localIndex === 'dissimilarity') {
      return [
        {
          kind: 'ramp',
          id: 'local',
          title: `Tract contribution to D (${group})`,
          ramp: 'magma',
          extent: 'gpu',
          format: value => value.toFixed(4)
        }
      ];
    }
    if (state.localIndex === 'entropy') {
      return [
        {
          kind: 'ramp',
          id: 'local',
          title: 'Local diversity (entropy, nats)',
          ramp: 'viridis',
          extent: 'gpu',
          format: value => value.toFixed(2)
        }
      ];
    }
    return [
      {
        kind: 'ramp',
        id: 'local',
        title: 'Contribution to entropy index H',
        ramp: 'diverging',
        extent: 'gpu',
        format: value => value.toFixed(4)
      }
    ];
  },

  snippet: state => `import {
  GPUNeighborSearch, GPUSegregation, GPUSpatialWeightsTransform,
  getGPUSegregationLayout, getGPUNeighborSearchParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// One distance band per scale; radius is a parameter-buffer write.
scales.forEach(({weights, searchParameters}, s) => {
  graph.add(new GPUNeighborSearch({
    mode: 'radius', gridSize: [32, 32], positions: tractCentroids,
    parameters: searchParameters, weights, overflow
  }));${
    state.weights === 'band'
      ? ''
      : `
  graph.add(new GPUSpatialWeightsTransform({
    operation: 'kernel', kernel: '${state.weights}', bandwidth: 'adaptive', weights
  }));`
  }${
    state.rowStandardize
      ? `
  graph.add(new GPUSpatialWeightsTransform({operation: 'row', weights}));`
      : ''
  }
});

graph.add(new GPUSegregation({
  unitCount: 791, groupCount: 5, groupCounts,       // tracts x [Black, White, Hispanic, Asian, other]
  scales: [null, ...scales.map(s => s.weights)],       // null = aspatial
  selfWeight: ${state.selfWeight}, spatialForm: '${state.spatialForm}', atkinsonB: ${state.atkinsonB},
  indices,                                            // layout from getGPUSegregationLayout(5)
  local: {environment, entropy, dissimilarity, theil}  // per-tract terms
}));`,

  about: {
    what: '`GPUSegregation` computes PySAL-style multigroup segregation indices (entropy H, multigroup and per-group dissimilarity D, isolation, interaction, Atkinson) for every tract, aspatially and in the local environment defined by a distance band, and returns the per-tract terms whose sums are the global indices. `GPUNeighborSearch` builds the band at each scale and `GPUSpatialWeightsTransform` turns it into a kernel.',
    why: 'A single citywide number hides the geography of segregation. Looking at how the indices change with the size of the neighbourhood shows whether segregation is a block-level pattern or a regional one, and the local terms show which tracts produce it.',
    howToRead:
      "Indices run from 0 (every tract mirrors the city) to 1 (complete separation). The dominant-group map is the raw geography. The composition and dissimilarity maps show the environment around each tract at the chosen scale. Matches PySAL segregation (`Dissim`, `MultiDissim`, `MultiInfoTheory`, `Isolation`, `Interaction`, `Atkinson`) at the aspatial scale; the spatial forms follow Reardon and O'Sullivan (2004)."
  },

  create: async ctx => (await import('./segregation.compute')).createSegregation(ctx),

  story: [
    {
      id: 'question',
      title: "Where do Chicago's groups live?",
      body: 'Chicago is often called one of the most segregated big cities in the United States. Before any index, look at the raw geography: this map colors every one of the 791 census tracts by its **largest group** (the first choice of **Map shows** below; 2018-2022 ACS counts from the CDC social vulnerability index). Black-majority tracts fill the South and West sides, Hispanic-majority tracts form a belt through Pilsen, Little Village and the Northwest side, and White-majority tracts hold the North Side lakefront.\n\nSegregation measures turn this picture into numbers and ask: at which scale does it show up?',
      options: {localIndex: 'dominant'},
      controls: ['localIndex'],
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.9}
    },
    {
      id: 'global-indices',
      title: 'Dissimilarity: how unevenly is a group spread?',
      body: "`GPUSegregation` computes the **dissimilarity index** *D* of a group against everyone else: the share of that group that would have to move for every tract to match the citywide mix, `D = ½ Σ |x_i / X − (t_i − x_i) / (T − X)|`. At the aspatial scale (scale 0) the Black-versus-rest *D* is about 0.77, White about 0.57 and Hispanic about 0.58: exactly the values PySAL gives for these tracts.\n\nSlide **Scale of the environment** to 0, pick the **Focal group**, and read **Focal group indices**: *isolation* is the chance the average Black resident's tract population is Black, and *exposure* shows what else they meet.",
      options: {localIndex: 'composition', focalGroup: 'nhBlack', scale: 0},
      controls: ['scale', 'focalGroup'],
      readouts: ['group', 'interaction'],
      highlight: {readout: 'group'}
    },
    {
      id: 'local-environment',
      title: 'The neighbourhood around each tract',
      body: 'A tract is not an island: a resident also lives among the neighbouring tracts. Move **Scale of the environment** to **2** and each tract is described by the population within a distance band of 2 km (`GPUNeighborSearch` finds the neighbours, `GPUSegregation` aggregates their counts). The map now shows the **Black share of the environment**: edges soften and the large contiguous South and West side region emerges.\n\nChange **Focal group** to see the others: Hispanic and Asian environments are far more fragmented.',
      options: {localIndex: 'composition', focalGroup: 'nhBlack', scale: 2},
      controls: ['scale', 'focalGroup'],
      readouts: ['scaleLabel'],
      highlight: {readout: 'scaleLabel'}
    },
    {
      id: 'contributions',
      title: 'Which tracts produce the segregation?',
      body: "Switch **Map shows** to *Contribution to the focal group dissimilarity D*. Every tract contributes a term `t_i |π_i − P| / (2 T P (1 − P))`: large where a populous tract's environment differs most from the city. The terms add up to *D*: see the readout **Check: local terms add up**.\n\nThe brightest tracts are not the most Black or most White; they are the big, homogeneous places that pull the city average away from an even mix.",
      options: {localIndex: 'dissimilarity'},
      controls: ['localIndex', 'focalGroup'],
      readouts: ['localSum'],
      highlight: {readout: 'localSum'}
    },
    {
      id: 'multiscale',
      title: 'Multiscale: how big is the pattern?',
      body: 'Drag **Scale of the environment** from 0 to 5 and read **Multiscale profile of multigroup D**. *D* declines as the environment widens from a tract to 16 km because every large neighbourhood contains more of the city. A pattern that holds on until 8 km is regional (the South Side as a whole); one that vanishes by 2 km is local (block-level sorting). **Base bandwidth** stretches the whole ladder.\n\nSwitch **Map shows** to *Local diversity (entropy)* to see where groups genuinely share a neighbourhood: Uptown, Rogers Park and the area around Hyde Park stand out as pockets of mixture.',
      options: {localIndex: 'entropy', scale: 3},
      controls: ['scale', 'bandwidth', 'localIndex'],
      readouts: ['profile'],
      highlight: {readout: 'profile'}
    },
    {
      id: 'definitions',
      title: 'Definitions are choices',
      body: "How a neighbour counts changes the answer. **Distance weighting** compares a binary band with kernels that weight nearby tracts more; **Row-standardise** makes the environment an average; **Weight of a tract in its own environment** controls how much a tract is its own neighbour. **Spatial form** switches between the Reardon-O'Sullivan environment form and the PySAL-style smoothed-population form.\n\nThese are compile-time options of `GPUSegregation` and `GPUSpatialWeightsTransform`, so changing one compiles another graph the first time (the *rebuild* badge) and then switches instantly.",
      options: {
        localIndex: 'composition',
        focalGroup: 'hispanic',
        scale: 2,
        weights: 'epanechnikov',
        selfWeight: '1'
      },
      controls: ['weights', 'rowStandardize', 'selfWeight', 'spatialForm']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: 'Tracts are an arbitrary partition (the modifiable areal unit problem), counts are ACS estimates with margins of error, and five categories flatten identities. Indices summarise unevenness, not inequality of opportunity. Tract populations in the data are those used for the SVI, and a few unpopulated tracts are drawn gray.\n\nTry: set **Map shows** to *Local diversity (entropy)* at scale 5; set **Atkinson parameter b** to 0.9 and watch the Atkinson value in **Focal group indices**; switch **Focal group** to *Asian*, a small group with high isolation in few tracts; compare the *Smoothed population* **Spatial form**.',
      options: {localIndex: 'composition', focalGroup: 'nhAsian', scale: 1},
      controls: ['localIndex', 'scale', 'atkinsonB', 'focalGroup', 'spatialForm'],
      readouts: ['group']
    }
  ]
});
