// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {NO_DATA_COLOR, RACE_GROUP_LABELS} from '../../cartography/hue-registry';
import {formatPercent} from '../../cartography/live-text';
import {defineScene, type LegendSpec} from '../scene';
import type {SegregationLegendData, SegregationOptions} from './segregation.compute';
import {
  DOMINANCE_TIER_LABELS,
  getDiversityTable,
  getDominanceColors,
  getRelativeTable,
  SEGREGATION_GROUPS
} from './segregation.style';

/** The South Side, where the radius ring of the environment step stays readable. */
const SOUTH_SIDE_BOUNDS = [-87.8, 41.66, -87.52, 41.9] as const;
/** Chicago at the sibling city frame. */
const CITY_CAMERA = {...CITY_FRAMES.chicago, transitionMs: 1400} as const;

/** Radii of the five environment steps, in km at the default base radius. */
const RADIUS_LADDER_KILOMETERS = [1, 2, 4, 8, 16] as const;

/** The cartouche of one step (the standing sample line is identical across the story). */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  sample: '791 census tracts, ACS 2018-2022 via CDC SVI 2022',
  chips: ['Estimates'] as const
});

const CREDIT = joinCredits(
  CREDITS.usCensus,
  'CDC/ATSDR Social Vulnerability Index 2022 (ACS 2018-2022)',
  CREDITS.cityOfChicago,
  CREDITS.colorBrewer
);

const GROUP_OPTIONS = SEGREGATION_GROUPS.map(group => ({value: group.id, label: group.label}));

function getLegends(
  state: SegregationOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const shared = data['segregation'] as SegregationLegendData | undefined;
  if (!shared) return [];
  const ground = shared.groundIsDark ? 'dark' : 'light';
  if (state.localIndex === 'dominant') {
    const colors = getDominanceColors(ground);
    const rows: string[] = [];
    const swatches: (typeof colors)[number][] = [];
    RACE_GROUP_LABELS.forEach((label, registryGroup) => {
      const cells = shared.dominanceCells.slice(registryGroup * 3, registryGroup * 3 + 3);
      const total = cells.reduce((sum, count) => sum + count, 0);
      if (total === 0) return;
      rows.push(`${label} (${total})`);
      swatches.push(...colors.slice(registryGroup * 3, registryGroup * 3 + 3));
    });
    return [
      {
        kind: 'matrix',
        title: 'Largest group in the tract',
        rows,
        columns: DOMINANCE_TIER_LABELS,
        colors: swatches,
        rowTitle: 'Group (tracts)',
        columnTitle: 'Share of the tract in that group',
        note: 'Hue says which group; tint says how large its share is.'
      },
      {
        kind: 'categories',
        title: 'No data',
        entries: [
          {
            color: NO_DATA_COLOR[ground],
            label: 'No residents',
            count: shared.emptyTracts,
            shape: 'hatch'
          }
        ]
      }
    ];
  }
  if (state.localIndex === 'relative') {
    return [
      getClassTableLegend(getRelativeTable(ground), {
        title: `${shared.focalLabel.split(' ')[0]} share of ${shared.environmentLabel}, against the city`,
        id: 'relative-classes',
        counts: shared.relativeCounts ?? undefined,
        layout: 'list',
        interactive: true,
        note: `Orange: more ${shared.focalLabel.split(' ')[0]} residents than the city (${formatPercent(shared.focalCityShare)} citywide). Purple: fewer. The pale middle class is about the city share.`
      })
    ];
  }
  return [
    getClassTableLegend(getDiversityTable(ground), {
      title: `Diversity of ${shared.environmentLabel}`,
      id: 'diversity-classes',
      counts: shared.diversityCounts ?? undefined,
      layout: 'list',
      interactive: true,
      note: `Citywide: ${shared.cityDiversity.toFixed(2)}. Grey means between groups: the darker, the more even the mix of all five.`
    })
  ];
}

function getSnippet(state: SegregationOptions): string {
  const kernel =
    state.weights === 'band'
      ? ''
      : `
  graph.add(new GPUSpatialWeightsTransform({
    operation: 'kernel', kernel: '${state.weights}', bandwidth: 'adaptive', weights
  }));`;
  const row = state.rowStandardize
    ? `
  graph.add(new GPUSpatialWeightsTransform({operation: 'row', weights}));`
    : '';
  return `import {
  GPUNeighborSearch, GPUSegregation, GPUSpatialWeightsTransform,
  getGPUSegregationLayout, getGPUNeighborSearchParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// One distance band per radius; the radius is a parameter-buffer write.
scales.forEach(({weights, searchParameters}) => {
  graph.add(new GPUNeighborSearch({
    mode: 'radius', gridSize: [32, 32], positions: tractPoints,
    parameters: searchParameters, weights, overflow
  }));${kernel}${row}
});

const common = {
  unitCount: 791, groupCount: 5,                       // tracts x [Black, White, Hispanic, Asian, other]
  scales: [null, ...scales.map(s => s.weights)],       // null = aspatial
  selfWeight: ${state.selfWeight}, spatialForm: '${state.spatialForm}', atkinsonB: ${state.atkinsonB}
};
graph.add(new GPUSegregation({
  ...common, groupCounts, indices,                     // layout from getGPUSegregationLayout(5)
  local: {environment, entropy, dissimilarity, theil}  // per-tract terms that sum to the indices
}));
// The checkerboard null: the same weights, whole tract rows shuffled with a seed.
graph.add(new GPUSegregation({...common, groupCounts: shuffledCounts, indices: shuffledIndices}));`;
}

export default defineScene<SegregationOptions>({
  id: 'segregation',
  title: 'The same index, two different cities',
  chapter: 'statistics',
  order: 5,
  summary:
    'Dissimilarity D cannot see where tracts are: shuffle them and it does not move. Environments of growing radius can, and the profile of D by radius tells whether a residential pattern is local or regional.',
  contributors: ['GPUSegregation', 'GPUNeighborSearch', 'GPUSpatialWeightsTransform'],
  datasets: [
    {id: 'chicago-tracts', role: '791 tracts with race and ethnicity counts (SVI 2022)'},
    {id: 'chicago-community-areas', role: 'community-area units and names'}
  ],
  initialView: CITY_FRAMES.chicago,
  basemap: ground('paperCity'),
  furniture: {
    title: cartouche("Where do Chicago's groups live?", 'Largest group of each tract'),
    credit: CREDIT
  },

  options: [
    {
      kind: 'select',
      id: 'localIndex',
      label: 'Map shows',
      group: 'Map',
      apply: 'param',
      default: 'dominant',
      help: 'A display kernel picks one local quantity per tract from the outputs of GPUSegregation; switching is a parameter write. The largest-group map is drawn from the counts.',
      options: [
        {value: 'dominant', label: 'Largest group and how dominant it is'},
        {
          value: 'relative',
          label: 'Focal group, against the city share',
          help: 'The share of the focal group in the environment of each tract, divided by its citywide share. Orange: over-represented; purple: under-represented.'
        },
        {
          value: 'diversity',
          label: 'Local diversity (entropy)',
          help: 'Entropy of the environment over its maximum, ln 5: darkest where all groups share the environment.'
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
      help: 'The group whose over- and under-representation, dissimilarity, isolation and Atkinson index are mapped and read out.',
      disabledWhen: state => state.localIndex !== 'relative',
      options: GROUP_OPTIONS
    },
    {
      kind: 'select',
      id: 'units',
      label: 'Units',
      group: 'Map',
      apply: 'param',
      display: 'segmented',
      default: 'tracts',
      help: 'The same residents counted in 791 tracts or in 77 community areas. The aspatial D of the areas is recomputed on the CPU from the counts summed by area.',
      disabledWhen: state => state.localIndex !== 'relative',
      options: [
        {value: 'tracts', label: 'Tracts'},
        {value: 'areas', label: 'Community areas'}
      ]
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
      default: 0,
      format: value => (value === 0 ? 'aspatial' : `${RADIUS_LADDER_KILOMETERS[value - 1]} km`),
      describe: (value, state) =>
        value === 0
          ? 'Each tract on its own'
          : `Everyone within ${Number((RADIUS_LADDER_KILOMETERS[value - 1] * state.bandwidth).toFixed(2))} km of the tract point`,
      help: 'Aspatial is each tract on its own. Radii 1, 2, 4, 8 and 16 times the base radius define the environment of a tract: every tract point within that distance. The radius is a parameter write; nothing is recompiled.',
      autoSweep: {durationMs: 8000}
    },
    {
      kind: 'slider',
      id: 'bandwidth',
      label: 'Base radius',
      group: 'Scale',
      apply: 'param',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      default: 1,
      unit: 'km',
      expert: true,
      help: 'The radius of the first rung; the ladder doubles it four times. At 0.5 km most tract points have no neighbour at all, so the first rung is then almost the aspatial index. The neighbour search re-runs with the new radii; nothing is recompiled.'
    },
    {
      kind: 'toggle',
      id: 'shuffle',
      label: 'Shuffle the tracts',
      group: 'Checkerboard test',
      apply: 'param',
      default: false,
      help: 'Moves whole tract rows to other tracts with a seeded permutation: every tract keeps its own counts, only its place changes. GPUSegregation runs a second time on the shuffled counts; aspatial D cannot change.'
    },
    {
      kind: 'slider',
      id: 'shuffleSeed',
      label: 'Shuffle seed',
      group: 'Checkerboard test',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 1,
      disabledWhen: state => !state.shuffle,
      help: 'The same seed gives the same shuffled city. Rewriting the shuffled count buffer is a buffer write, not a recompile.'
    },
    {
      kind: 'toggle',
      id: 'showRing',
      label: "Show one tract's environment",
      group: 'Environment',
      apply: 'param',
      default: false,
      help: 'Outlines the chosen tract, draws the radius as a dashed ring in true metres and dims every tract outside it. Click any tract to choose another.'
    },
    {
      kind: 'select',
      id: 'focusArea',
      label: 'Place to inspect',
      group: 'Environment',
      apply: 'param',
      default: 'Englewood',
      help: 'Picks the most populous tract of a community area as the tract whose environment is drawn.',
      options: [
        {value: 'Englewood', label: 'Englewood'},
        {value: 'Lower West Side', label: 'Lower West Side (Pilsen)'},
        {value: 'Hyde Park', label: 'Hyde Park'},
        {value: 'Rogers Park', label: 'Rogers Park'},
        {value: 'Austin', label: 'Austin'},
        {value: 'Lincoln Park', label: 'Lincoln Park'}
      ]
    },
    {
      kind: 'select',
      id: 'weights',
      label: 'Distance weighting',
      group: 'Environment weights',
      apply: 'compile',
      default: 'band',
      help: "How neighbours count in the environment. Binary band: every neighbour inside the radius counts once. The kernels (GPUSpatialWeightsTransform, adaptive bandwidth) give nearer tracts more weight; the bandwidth is each row's farthest neighbour, not the radius. Compile-time: compiles another graph on demand.",
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
      display: 'segmented',
      default: '1',
      help: 'GPUSegregation adds this weight for the tract itself; neighbour lists never include it. 0 excludes the tract. Compile-time.',
      options: [
        {value: '0', label: '0'},
        {value: '0.5', label: '0.5'},
        {value: '1', label: '1'},
        {value: '2', label: '2'}
      ]
    },
    {
      kind: 'select',
      id: 'spatialForm',
      label: 'Spatial form',
      group: 'Index definition',
      apply: 'compile',
      display: 'segmented',
      default: 'environment',
      help: "Environment (Reardon and O'Sullivan): sums keep actual tract populations, only the compositions come from the environment. Smoothed population: the smoothed table replaces the population (what PySAL spatially implicit indices do). Compile-time.",
      options: [
        {value: 'environment', label: 'Environment'},
        {value: 'smoothed-population', label: 'Smoothed population'}
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
    }
  ],

  readouts: [
    {id: 'residents', label: 'Residents', format: 'integer'},
    {
      id: 'supermajority',
      label: 'Tracts with 80% or more in one group',
      format: 'integer',
      emphasis: 'tile'
    },
    {
      id: 'noMajority',
      label: 'Tracts with no group above half',
      format: 'integer',
      emphasis: 'tile',
      help: 'Populated tracts whose largest group is under 50% of residents: the plurality tier of the map.'
    },
    {id: 'focalLabel', label: 'Focal group'},
    {id: 'cityShare', label: 'Focal group, citywide share'},
    {
      id: 'dAspatial',
      label: 'Dissimilarity D, five groups, aspatial',
      emphasis: 'tile',
      help: 'Multigroup dissimilarity of the 791 tracts (Reardon and Firebaugh): 0 when every tract mirrors the city, 1 at complete separation.'
    },
    {
      id: 'dFocal',
      label: 'D of the focal group, aspatial',
      help: 'The focal group against everyone else.'
    },
    {
      id: 'dAreas',
      label: 'Dissimilarity D, five groups, community areas',
      emphasis: 'tile',
      help: 'The same formula on the counts summed into 77 community areas, computed on the CPU.'
    },
    {id: 'dFocalAreas', label: 'D of the focal group, community areas'},
    {
      id: 'dAspatialShuffled',
      label: 'Dissimilarity D, tracts shuffled, aspatial',
      help: 'GPUSegregation on the shuffled counts. Aspatial D does not depend on where tracts are, so it equals the real value up to float rounding.'
    },
    {id: 'dMidReal', label: 'D at the middle radius, real tracts', emphasis: 'tile'},
    {id: 'dMidShuffled', label: 'D at the middle radius, tracts shuffled', emphasis: 'tile'},
    {
      id: 'profile',
      label: 'Multigroup D by environment radius',
      kind: 'chart',
      help: 'Real tracts solid; the shuffled city dashed once Shuffle the tracts is on. Click the chart to set the scale.'
    },
    {id: 'radius', label: 'Environment radius'},
    {id: 'radiusMid', label: 'Middle radius'},
    {id: 'radiusWide', label: 'Wide radius'},
    {
      id: 'members',
      label: 'Other tracts inside the ring',
      format: 'integer',
      help: 'Tract points within the radius of the chosen tract point, the chosen tract excluded.'
    },
    {id: 'ownShare', label: 'Focal group in the tract itself'},
    {id: 'envShare', label: 'Focal group within the radius', emphasis: 'tile'},
    {
      id: 'composition',
      label: 'Composition: tract, environment, city',
      kind: 'chart',
      help: 'Group shares of the chosen tract, of its environment as GPUSegregation computed it (kernel and self weight included) and of the whole city.'
    },
    {
      id: 'profileAll',
      label: 'D of each group by environment radius',
      kind: 'chart',
      help: 'Chart colours follow the chart palette: blue White, orange Black, green Hispanic, purple Asian; Other is the grey line.'
    },
    {
      id: 'cityDiversity',
      label: 'Citywide diversity',
      help: 'Entropy of the citywide composition over its maximum, ln 5.'
    },
    {id: 'blackWide', label: 'D of Black residents, wide radius'},
    {id: 'hispanicWide', label: 'D of Hispanic residents, wide radius'},
    {id: 'dScale', label: 'Multigroup D at this scale', emphasis: 'tile'},
    {
      id: 'entropyIndex',
      label: 'Entropy index H at this scale',
      help: 'Theil information-theory index: how much less diverse the environments are than the city.'
    },
    {
      id: 'isolation',
      label: 'Isolation of the focal group',
      help: 'The chance that a typical member of the focal group meets another member in their environment.'
    },
    {
      id: 'atkinson',
      label: 'Atkinson index of the focal group',
      help: 'Inequality-averse unevenness; the parameter b is a compile-time choice.'
    },
    {id: 'selected', label: 'Chosen tract', help: 'Click a tract to choose it.'},
    {
      id: 'localSum',
      label: 'Check: local terms add up',
      hood: true,
      help: 'The per-tract terms are the terms of the sums: they add up to the global index of the same scale.'
    },
    {id: 'overflow', label: 'Neighbour capacity overflow', hood: true},
    {
      id: 'capacity',
      label: 'Neighbour slots',
      hood: true,
      help: 'Capacity is tracts squared per radius: safe, but wasteful. A metro of 2,000 tracts would need about 4 million slots per array and radius.'
    }
  ],

  pipeline: [
    {
      id: 'points',
      label: 'Tract points',
      detail: 'One point per tract: the mean of its outline vertices',
      show: {option: 'localIndex', value: 'dominant'}
    },
    {
      id: 'search',
      label: 'Radius search',
      detail: 'GPUNeighborSearch lists the points within each radius',
      show: {option: 'showRing', value: true}
    },
    {
      id: 'environments',
      label: 'Environments',
      detail: 'Counts of the neighbours, weighted and standardised, become shares',
      show: {option: 'localIndex', value: 'relative'}
    },
    {
      id: 'indices',
      label: 'Indices',
      detail: 'GPUSegregation sums per-tract terms into D, H, isolation and Atkinson',
      show: {option: 'shuffle', value: true}
    },
    {
      id: 'profile',
      label: 'Profile',
      detail: 'The same indices at every radius, as a curve',
      show: {option: 'localIndex', value: 'diversity'}
    }
  ],

  legends: getLegends,
  snippet: getSnippet,

  about: {
    what: '`GPUSegregation` computes PySAL-style multigroup segregation indices (entropy H, multigroup and per-group dissimilarity D, isolation, interaction, Atkinson) for tracts, aspatially and in the local environment defined by a distance band, and returns the per-tract terms whose sums are the global indices. `GPUNeighborSearch` builds the band at each radius and `GPUSpatialWeightsTransform` turns it into a kernel. A second `GPUSegregation` node reads the same weights on tract rows shuffled with a seed: the checkerboard null.',
    why: 'An aspatial index cannot see arrangement: shuffle the tracts and it does not move. Indices over environments of growing radius can, and the profile of the index by radius shows whether a residential pattern is local (it fades within a few kilometres) or regional (it persists).',
    howToRead:
      "Indices run from 0 (every tract mirrors the city) to 1 (complete separation). On the largest-group map, hue is the group and tint is the share of the largest group. On the environment maps, orange is more of the focal group than the city and purple less; grey is the diversity of the mix. Matches PySAL segregation (`Dissim`, `MultiDissim`, `MultiInfoTheory`, `Isolation`, `Interaction`, `Atkinson`) at the aspatial scale; the spatial forms follow Reardon and O'Sullivan (2004)."
  },

  create: async ctx => (await import('./segregation.compute')).createSegregation(ctx),

  story: [
    {
      id: 'where',
      title: "Where do Chicago's groups live?",
      headline: 'Most tracts are dominated by one group',
      textAlternative:
        'Map of Chicago census tracts coloured by their largest group: blue White, orange Black and green Hispanic tracts form large contiguous regions, with darker tints where one group holds 80 percent or more and pale tints where no group holds half.',
      body: 'Each tract takes the hue of its largest group, and the tint says how large that share is. **{{supermajority}}** tracts have 80% or more of their residents in one group; only **{{noMajority}}** have no group above half. Switch **Map shows** to see the same tracts as over- and under-representation.\n\n*Hue says which group; tint says how sure.*',
      options: {localIndex: 'dominant'},
      optionsMode: 'fresh',
      controls: ['localIndex'],
      readouts: ['supermajority', 'noMajority', 'residents'],
      camera: CITY_CAMERA,
      basemap: ground('paperCity'),
      furniture: {
        title: cartouche(
          "Where do Chicago's groups live?",
          'Largest group of each tract, by share'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, [
        'englewood',
        'austin',
        'bronzeville',
        'pilsen',
        'little-village',
        'rogers-park',
        'lincoln-park',
        'albany-park',
        'hyde-park',
        'lake-michigan'
      ]),
      stage: 'points'
    },
    {
      id: 'one-number',
      title: 'One number for the whole city',
      headline: 'Larger reporting units hide unevenness',
      textAlternative:
        'Map of Chicago tracts in seven orange to purple classes by how over- or under-represented the focal group is against its citywide share; the middle classes are pale and the South and West sides are strongly orange.',
      body: 'D is the share of a group that would have to move for every tract to match the city. For all five groups together it is **{{dAspatial}}**; for **{{focalLabel}}** alone, **{{dFocal}}**. Count the same people in community areas (**Units**) and D falls to **{{dAreas}}** (**{{dFocalAreas}}**). Pick the **Focal group** to remap.\n\n*Bigger units hide unevenness; the modifiable areal unit problem is taught in Wildlife by community area, joined to who lives there.*',
      options: {localIndex: 'relative', focalGroup: 'nhBlack', scale: 0, units: 'tracts'},
      optionsMode: 'fresh',
      controls: ['focalGroup', 'units'],
      readouts: ['dAspatial', 'dFocal', 'dAreas', 'dFocalAreas'],
      camera: CITY_CAMERA,
      furniture: {
        title: cartouche(
          'One number for the whole city',
          'Focal group share of each tract against the city share; dissimilarity D, aspatial'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, [
        'englewood',
        'austin',
        'lincoln-park',
        'pilsen',
        'lake-michigan'
      ]),
      stage: 'indices'
    },
    {
      id: 'checkerboard',
      title: 'The same number, two different cities',
      headline: 'Shuffled tracts give the same D',
      textAlternative:
        'Chicago tracts coloured by largest group on either side of a draggable divider: on the left the real city in large blocks, on the right the same tracts shuffled into a salt-and-pepper mosaic; a line chart below shows D falling much faster with radius for the shuffled city.',
      body: 'Shuffling moves whole tracts to other places and keeps every count. Aspatial D stays **{{dAspatial}}** (shuffled: **{{dAspatialShuffled}}**) because D never sees where tracts are. At **{{radiusMid}}** the real city holds **{{dMidReal}}**, the shuffled one **{{dMidShuffled}}**. Drag the divider; try another **Shuffle seed**.\n\n*The checkerboard problem: an aspatial index cannot see arrangement.*',
      options: {localIndex: 'dominant', shuffle: true, scale: 0},
      optionsMode: 'fresh',
      controls: ['shuffle', 'shuffleSeed'],
      readouts: ['profile', 'dAspatial', 'dMidReal', 'dMidShuffled'],
      camera: CITY_CAMERA,
      compare: {labels: ['Real tracts', 'Shuffled tracts'], position: 0.5},
      furniture: {
        title: cartouche(
          'Shuffle the tracts: D does not notice',
          'Largest group of each tract, real | shuffled'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, [
        'englewood',
        'austin',
        'pilsen',
        'rogers-park',
        'lake-michigan'
      ]),
      stage: 'indices'
    },
    {
      id: 'environment',
      title: 'The neighbourhood around each tract',
      headline: 'Wider surroundings look more like the city',
      textAlternative:
        'South Side of Chicago with one Englewood tract outlined, a dashed ring of the chosen radius around it, the tracts inside the ring at full colour and the rest dimmed; a stacked bar compares the tract, its environment and the city.',
      body: 'The dashed ring marks everyone within **{{radius}}** of the outlined tract: **{{members}}** other tracts. Their residents are **{{envShare}}** {{focalLabel}}, against **{{ownShare}}** in the tract itself and **{{cityShare}}** in the city. Move **Scale** or pick another **Place**; click any tract.\n\n*An environment is a choice of radius.*',
      options: {
        localIndex: 'relative',
        focalGroup: 'nhBlack',
        scale: 3,
        showRing: true,
        focusArea: 'Englewood',
        units: 'tracts'
      },
      optionsMode: 'fresh',
      controls: ['scale', 'focusArea'],
      readouts: ['composition', 'members', 'ownShare', 'envShare'],
      camera: {bounds: SOUTH_SIDE_BOUNDS, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Each tract in its surroundings',
          'Focal group share of everyone within the radius, against the city share'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, [
        'englewood',
        'hyde-park',
        'bronzeville',
        'south-shore',
        'lake-michigan'
      ]),
      stage: 'search'
    },
    {
      id: 'profile',
      title: 'How big is the pattern?',
      headline: 'Some patterns are regional, some local',
      textAlternative:
        'Chicago tracts in five grey classes by local diversity at a 2 km radius, darkest where the mix is most even, as in Rogers Park, Uptown and Albany Park, and a line chart of the dissimilarity of each group by radius in which the Black curve stays high and the Hispanic curve falls to near zero.',
      body: 'At **{{radiusWide}}** the Black D is still **{{blackWide}}** while the Hispanic D has fallen to **{{hispanicWide}}**: one pattern is regional, the other local. The grey map shows diversity at **{{radius}}** against **{{cityDiversity}}** for the whole city. Slide **Scale**. Data end at the city limit, so edge tracts see only part of a disc.\n\n*The scale of a pattern is a finding, not a setting.*',
      options: {localIndex: 'diversity', scale: 2},
      optionsMode: 'fresh',
      controls: ['scale'],
      readouts: ['profileAll', 'cityDiversity', 'blackWide', 'hispanicWide'],
      camera: CITY_CAMERA,
      furniture: {
        title: cartouche(
          'Some patterns are regional, some local',
          'Diversity of the environment; D of each group by radius'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, ['lake-michigan', 'evanston', 'oak-park'], {
        evanston: {tone: 'muted', detail: 'outside the data'},
        'oak-park': {tone: 'muted', detail: 'outside the data'}
      }),
      stage: 'profile'
    },
    {
      id: 'definitions',
      title: 'Definitions are choices',
      headline: 'Definitions nudge the curve; scale moves it',
      textAlternative:
        'Chicago tracts in orange to purple classes by Hispanic share of the environment at 2 km, with the profile of D by radius beside it; the curve moves only slightly when the weighting, self weight or spatial form change.',
      body: 'How a neighbour counts changes the answer a little. At **{{radius}}** D is **{{dScale}}**; change **Distance weighting**, **Weight of a tract in its own environment** or **Spatial form** and watch the curve. Each is compile-time, so a first use compiles a graph. Tracts are ACS estimates, small groups rest on small counts and each tract is one point. Everything else is in All controls.\n\n*Report the definition with the number.*',
      options: {localIndex: 'relative', focalGroup: 'hispanic', scale: 2},
      optionsMode: 'fresh',
      controls: ['weights', 'selfWeight', 'spatialForm'],
      readouts: ['profile', 'dScale', 'isolation', 'atkinson'],
      camera: CITY_CAMERA,
      furniture: {
        title: cartouche(
          'Definitions are choices',
          'Focal group share of the environment, against the city share'
        ),
        credit: CREDIT
      },
      annotations: labelsFor(CHICAGO, ['pilsen', 'little-village', 'lake-michigan']),
      stage: 'environments'
    }
  ]
});
