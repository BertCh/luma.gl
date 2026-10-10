// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {ElectionDynamicsOptions} from './election-dynamics.compute';

const YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024];

const SHARE_COLORS = [
  [178, 24, 43, 245],
  [239, 138, 98, 245],
  [205, 205, 215, 245],
  [103, 169, 207, 245],
  [33, 102, 172, 245]
] as const;
const TURNOUT_COLORS = [
  [68, 1, 84, 245],
  [59, 82, 139, 245],
  [33, 145, 140, 245],
  [94, 201, 98, 245],
  [253, 231, 37, 245]
] as const;
const LISA = [
  {color: [120, 130, 145, 80], label: 'Not significant'},
  {color: [215, 48, 39, 250], label: 'HH: high, high neighbors'},
  {color: [145, 191, 219, 250], label: 'LH: low, high neighbors'},
  {color: [49, 54, 149, 250], label: 'LL: low, low neighbors'},
  {color: [253, 174, 97, 250], label: 'HL: high, low neighbors'}
] as const;

const variableName = (variable: ElectionDynamicsOptions['variable']) =>
  variable === 'demTwoParty'
    ? 'Democratic two-party share'
    : variable === 'demShare'
      ? 'Democratic share of all votes'
      : 'Votes per resident';

export default defineScene<ElectionDynamicsOptions>({
  id: 'election-dynamics',
  title: 'US county vote-class transitions, 2000–2024',
  chapter: 'time',
  order: 3,
  summary:
    'Transition, spatial Markov and LISA Markov models use county presidential returns from 2000–2024 to map class persistence and neighbor-conditioned change; county units, missing returns and class breaks constrain comparisons.',
  contributors: ['GPUTransitionMatrix', 'GPUSpatialMarkov', 'GPULISAMarkov'],
  datasets: [
    {id: 'us-counties', role: 'county boundaries'},
    {id: 'us-elections', role: 'presidential returns 2000 to 2024'},
    {id: 'us-states', role: 'state outlines'}
  ],
  initialView: {longitude: -96.2, latitude: 38.2, zoom: 3.6},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'US county vote transitions',
      subtitle: 'Presidential returns, 2000–2024'
    },
    scaleBar: {units: 'metric'},
    credit: 'MIT Election Data and Science Lab; US Census Bureau',
    caveat: 'Results count counties, not voters; some 2024 county returns are unavailable.'
  },

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable',
      group: 'Data',
      apply: 'param',
      default: 'demTwoParty',
      options: [
        {value: 'demTwoParty', label: 'Democratic two-party share'},
        {value: 'demShare', label: 'Democratic share of all votes'},
        {value: 'turnout', label: 'Turnout proxy (votes per resident)'}
      ],
      help: 'Which column is classified. Writing a different column into the values buffer never recompiles. Turnout is votes over 2020 residents, so it is comparable across counties within a year but inflated by later population growth.'
    },
    {
      kind: 'select',
      id: 'through',
      label: 'Elections included',
      group: 'Data',
      apply: 'compile',
      default: '2024',
      options: [
        {value: '2020', label: '2000 to 2020 (six elections)'},
        {value: '2024', label: '2000 to 2024 (seven, fewer counties)'}
      ],
      help: 'The number of periods is fixed when the chain compiles. The 2024 returns are rebuilt from precinct data and five states (Indiana, Louisiana, New Jersey, New York, Alaska) are missing, so including 2024 drops those counties from every period.'
    },
    {
      kind: 'select',
      id: 'periodLag',
      label: 'Transition step',
      group: 'Data',
      apply: 'compile',
      default: '1',
      options: [
        {value: '1', label: 'One election (4 years)'},
        {value: '2', label: 'Two elections (8 years)'}
      ],
      help: 'The number of periods between the two ends of a transition. A longer step lets slow movers show up.'
    },
    {
      kind: 'select',
      id: 'classMethod',
      label: 'Class method',
      group: 'Classes',
      apply: 'param',
      default: 'equal-interval',
      options: [
        {
          value: 'equal-interval',
          label: 'Equal interval',
          help: 'Bands of equal width over the pooled range; classes read as vote-share bands.'
        },
        {
          value: 'quantile',
          label: 'Quantile',
          help: 'Equal numbers of county-years per class; classes read as relative standing.'
        },
        {
          value: 'natural-breaks',
          label: 'Natural breaks (Jenks)',
          help: 'Breaks that minimise within-class variance.'
        }
      ],
      help: 'How the pooled values of every county and year are cut into classes. One set of breaks is shared by all years, so one legend serves the time slider.'
    },
    {
      kind: 'slider',
      id: 'classCount',
      label: 'Number of classes',
      group: 'Classes',
      apply: 'param',
      min: 2,
      max: 5,
      step: 1,
      default: 5,
      help: 'More classes give a finer matrix but fewer transitions per cell.'
    },
    {
      kind: 'select',
      id: 'lagMethod',
      label: 'Neighbor class method',
      group: 'Spatial Markov',
      apply: 'param',
      default: 'equal-interval',
      options: [
        {value: 'equal-interval', label: 'Equal interval'},
        {value: 'quantile', label: 'Quantile'}
      ],
      help: "How the average of a county's neighbors (its spatial lag) is classified. The spatial Markov splits every transition by this class."
    },
    {
      kind: 'slider',
      id: 'lagClassCount',
      label: 'Neighbor classes',
      group: 'Spatial Markov',
      apply: 'param',
      min: 2,
      max: 3,
      step: 1,
      default: 3,
      help: 'Number of classes of the neighbor average. Three gives low, middle and high surroundings.'
    },
    {
      kind: 'slider',
      id: 'lisaSignificance',
      label: 'LISA significance level',
      group: 'LISA Markov',
      apply: 'param',
      min: 0.01,
      max: 0.2,
      step: 0.01,
      default: 0.05,
      help: 'Local Moran z-test level (analytic, not permutation). Counties below it are state "not significant" in that year.'
    },
    {
      kind: 'select',
      id: 'lisaMoments',
      label: 'Local Moran reference',
      group: 'LISA Markov',
      apply: 'param',
      default: 'per-year',
      options: [
        {value: 'per-year', label: 'Each election on its own mean and variance'},
        {value: 'pooled', label: 'One pooled mean and variance for all elections'}
      ],
      help: 'Per year, "high" means above that year\'s national mean. Pooled, "high" means above the 24-year mean, so a shift of the whole country moves counties between states.'
    },
    {
      kind: 'select',
      id: 'mapView',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'class',
      options: [
        {value: 'class', label: 'Class in the election year'},
        {value: 'lag', label: "Class of the neighbors' average"},
        {value: 'lisa', label: 'LISA state (local Moran)'},
        {value: 'move', label: 'Move to the next election (classes up or down)'}
      ],
      help: 'Each view gathers one period from a buffer the chain already filled.'
    },
    {
      kind: 'slider',
      id: 'year',
      label: 'Election',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 6,
      step: 1,
      default: 6,
      format: value => String(YEARS[value] ?? ''),
      help: 'Which presidential election the map shows (clamped when 2024 is not included).'
    },
    {
      kind: 'toggle',
      id: 'play',
      label: 'Play through the elections',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Steps the election every second or so by rewriting a two-word parameter.'
    },
    {
      kind: 'select',
      id: 'matrix',
      label: 'Transition matrix shown',
      group: 'Display',
      apply: 'param',
      default: 'pooled',
      options: [
        {value: 'pooled', label: 'All counties (Markov)'},
        {value: 'lag0', label: 'Spatial Markov: low-lag neighbors'},
        {value: 'lag1', label: 'Spatial Markov: middle-lag neighbors'},
        {value: 'lag2', label: 'Spatial Markov: high-lag neighbors'},
        {value: 'lisa', label: 'LISA Markov (local Moran states)'}
      ],
      help: 'Rows are the class now, columns the class one step later; each row sums to 1.'
    },
    {
      kind: 'toggle',
      id: 'showStates',
      label: 'State outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws state borders over the counties.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 0.92,
      help: 'Lower it to see the basemap under the counties.'
    }
  ],

  readouts: [
    {
      id: 'counties',
      label: 'Counties analysed',
      help: 'Counties with returns in every included election.'
    },
    {
      id: 'links',
      label: 'Queen neighbors',
      help: 'Counties that share at least one boundary vertex.'
    },
    {id: 'yearShown', label: 'Election shown'},
    {
      id: 'classEdges',
      label: 'Class edges (pooled)',
      help: 'Break values shared by every election.'
    },
    {id: 'lagEdges', label: 'Neighbor class edges'},
    {id: 'transitions', label: 'Transitions counted'},
    {
      id: 'stay',
      label: 'Stay probability (all)',
      help: 'Share of transitions that remain in the same class: the diagonal of the matrix.'
    },
    {id: 'stayLag', label: 'Stay by neighbors low / mid / high'},
    {id: 'stayLisa', label: 'Stay (LISA states)'},
    {id: 'row0', label: 'Matrix row 1'},
    {id: 'row1', label: 'Matrix row 2'},
    {id: 'row2', label: 'Matrix row 3'},
    {id: 'row3', label: 'Matrix row 4'},
    {id: 'row4', label: 'Matrix row 5'}
  ],

  legends: state => {
    const classEntries = (state.variable === 'turnout' ? TURNOUT_COLORS : SHARE_COLORS)
      .slice(0, state.classCount === 5 ? 5 : 5)
      .map((color, index) => ({
        color: [...color] as [number, number, number, number],
        label:
          index === 0
            ? 'Class 1 (lowest)'
            : index === 4
              ? 'Class 5 (highest)'
              : `Class ${index + 1}`
      }));
    switch (state.mapView) {
      case 'class':
        return [
          {
            kind: 'categories',
            title: `${variableName(state.variable)}, class`,
            entries: classEntries,
            note: 'Classes use one set of breaks for all elections; the exact edges are in the readouts. Only the first classes are used when you pick fewer.'
          }
        ];
      case 'lag':
        return [
          {
            kind: 'categories',
            title: "Class of the neighbors' average",
            entries: [
              {
                color: (state.variable === 'turnout'
                  ? TURNOUT_COLORS
                  : SHARE_COLORS)[0] as unknown as [number, number, number, number],
                label: 'Low neighbors'
              },
              {
                color: (state.variable === 'turnout'
                  ? TURNOUT_COLORS
                  : SHARE_COLORS)[2] as unknown as [number, number, number, number],
                label: 'Middle neighbors'
              },
              {
                color: (state.variable === 'turnout'
                  ? TURNOUT_COLORS
                  : SHARE_COLORS)[4] as unknown as [number, number, number, number],
                label: 'High neighbors'
              }
            ]
          }
        ];
      case 'lisa':
        return [
          {
            kind: 'categories',
            title: 'Local Moran state',
            entries: LISA.map(entry => ({
              color: [...entry.color] as [number, number, number, number],
              label: entry.label
            })),
            note: `Significance level ${state.lisaSignificance}.`
          }
        ];
      default:
        return [
          {
            kind: 'categories',
            title: 'Move to the next election',
            entries: [
              {color: [84, 39, 136, 250], label: 'Down two or more classes'},
              {color: [153, 142, 195, 245], label: 'Down one class'},
              {color: [225, 225, 225, 160], label: 'Same class'},
              {color: [241, 163, 64, 245], label: 'Up one class'},
              {color: [179, 88, 6, 250], label: 'Up two or more classes'}
            ],
            note: 'Up means a higher class of the chosen variable (more Democratic or higher turnout).'
          }
        ];
    }
  },

  snippet: state => `import {
  getGPUClassBreaksParameterValues, GPUClassBreaks
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUClassAssignment, GPULISAMarkov, GPUSpatialMarkov, GPUTransitionMatrix
} from '@luma.gl/experimental/gpu-spatial-analysis';

// values: rows * periods float32, period-major; weights: queen CSR {offsets, neighbors, weights}
graph.add(new GPUClassBreaks({values, mask: pooledMask, parameters: breaksParameters,
  maximumClassCount: 5, methods: ['quantile', 'equal-interval', 'natural-breaks'],
  output: {breaks, classCount}}));
graph.add(new GPUClassAssignment({values, breaks, classCount, mask: pooledMask, output: classes}));
graph.add(new GPUTransitionMatrix({classes, rows, periods: ${state.through === '2024' ? 7 : 6},
  periodLag: ${state.periodLag}, classCount: 5, mask: rowMask,
  output: {counts, probabilities, rowTotals, ignored}}));
graph.add(new GPUSpatialMarkov({values, classes, classCount: 5, weights, periods: ${state.through === '2024' ? 7 : 6},
  periodLag: ${state.periodLag}, lagMaximumClassCount: 3, lagParameters, lagMethods: ['equal-interval', 'quantile'],
  mask: rowMask, output: {counts: spatialCounts, lagBreaks, lagClassCount, lagClasses}}));
graph.add(new GPULISAMarkov({values, weights, periods: ${state.through === '2024' ? 7 : 6}, periodLag: ${state.periodLag},
  parameters: lisaParameters, mask: rowMask, output: {counts: lisaCounts, quadrants}}));

// parameters, per frame, no rebuild
breaksParameters.write(getGPUClassBreaksParameterValues(
  {method: '${state.classMethod}', classCount: ${state.classCount}}, 5));
lisaParameters.write(getGPUSpatialAutocorrelationParameterValues(
  {significanceLevel: ${state.lisaSignificance}${state.lisaMoments === 'pooled' ? ', fixedMoments: {count, mean, variance}' : ''}}));`,

  about: {
    what: "`GPUTransitionMatrix` counts how many counties moved from class a to class b between elections (giddy `Markov`). `GPUSpatialMarkov` splits those counts by the class of the neighbors' average at the start (Rey 2001). `GPULISAMarkov` counts moves between local Moran states (HH, LH, LL, HL, not significant).",
    why: 'A single map says who votes how; a transition matrix says how sticky that is. If a county in a Democratic neighborhood is much less likely to switch than one in a Republican neighborhood, regional context shapes change, which a non-spatial matrix cannot show.',
    howToRead:
      'Each matrix row is the class a county is in now; the numbers are the chances of each class at the next election. A strong diagonal means persistence. Compare the three neighbor-conditioned matrices: differences between them are the spatial effect.'
  },

  create: async ctx => (await import('./election-dynamics.compute')).createElectionDynamics(ctx),

  story: [
    {
      id: 'question',
      title: "How sticky is a county's vote?",
      headline: 'Most counties remain in the same vote class',
      textAlternative:
        'US counties are assigned consistent red-to-blue vote-share classes for the selected presidential election.',
      body: 'Presidential returns for about 3,000 US counties, 2000 to 2024. Each county-year is put in one of five classes of Democratic two-party share, using **one set of breaks for all years** so a color means the same share in 2000 and in 2024. Red classes are Republican-leaning, blue Democratic-leaning.\n\nDrag the **Election** slider or tick **Play through the elections** below to watch the map change. **Variable** picks what is classified. Most counties barely move; the interesting question is how many move, which way, and whether it depends on the neighbors.',
      options: {mapView: 'class', year: 6},
      controls: ['variable', 'year', 'play'],
      readouts: ['yearShown', 'counties']
    },
    {
      id: 'classes',
      title: 'Classes: equal bands or equal counts',
      headline: 'Class definitions change county assignments',
      textAlternative:
        'The county map and class-edge readout update as equal-interval, quantile or natural-break definitions are selected.',
      body: '**`GPUClassBreaks`** computes breaks over the pooled county-years and **`GPUClassAssignment`** puts every county-year in a class. With equal-interval breaks the readout **Class edges** are vote-share bands; with *quantile* each class holds the same number of county-years and classes mean relative standing, which hides a national swing.\n\nTry **Class method** *Natural breaks (Jenks)* and a lower **Number of classes**. The class method and count are per-frame parameters; the whole chain re-runs without recompiling. The 2024 returns are rebuilt from precinct data and are missing for Indiana, Louisiana, New Jersey, New York and Alaska, so those states are blank when 2024 is included.',
      options: {year: 0},
      camera: {zoom: 3.6, transitionMs: 1000},
      controls: ['classMethod', 'classCount'],
      readouts: ['classEdges']
    },
    {
      id: 'markov',
      title: 'The transition matrix: how often does a county stay put?',
      headline: 'Middle vote classes change most often',
      textAlternative:
        'County move colors accompany a transition matrix whose diagonal reports persistence between presidential elections.',
      body: '**`GPUTransitionMatrix`** counts, for every pair of consecutive elections, how many counties moved from class a to class b. Each row of the matrix is the class now, each column the class one election (four years) later, and the numbers are probabilities (`p_ab = n_ab / n_a`). A strong diagonal means persistence.\n\nThe readouts below show the rows of the matrix chosen in **Transition matrix shown**, and the share of transitions that stayed in the same class. Counties in the middle classes are the likeliest to move, since they have a neighbor class on both sides; the extreme classes can only move one way. Switch **Transition step** to two elections (eight years) and the diagonal typically weakens: longer steps give more time to move.',
      options: {matrix: 'pooled', mapView: 'move', year: 4},
      highlight: {readout: 'stay'},
      controls: ['matrix', 'periodLag'],
      readouts: ['stay', 'transitions', 'row0']
    },
    {
      id: 'spatial-markov',
      title: 'Does the neighborhood matter? The spatial Markov',
      headline: 'Persistence differs by neighboring vote class',
      textAlternative:
        'Counties are shaded by neighbor-average class while conditioned transition rows compare low, middle and high surroundings.',
      body: "**`GPUSpatialMarkov`** repeats the matrix but splits every transition by the class of the **average of the county's neighbors** at the start (queen contiguity, row-standardised). Three matrices result: for counties in low, middle and high surroundings.\n\nIf the three matrices were the same, location would not matter. Switch **Transition matrix shown** between *low-lag*, *middle-lag* and *high-lag* and compare the rows and **Stay by neighbors**. Setting **Map shows** to *Class of the neighbors' average* shows the context itself. The lag classes come from their own pooled breaks, set by **Neighbor class method** and **Neighbor classes**.",
      options: {matrix: 'lag0', mapView: 'lag', year: 6},
      highlight: {readout: 'stayLag'},
      controls: ['matrix', 'mapView', 'lagMethod', 'lagClassCount'],
      readouts: ['stayLag', 'row0', 'row1']
    },
    {
      id: 'lisa-markov',
      title: 'Clusters through time: the LISA Markov',
      headline: 'Local Moran states persist across elections',
      textAlternative:
        'Counties are classified as high-high, low-low, spatial outliers or nonsignificant for the selected election.',
      body: '**`GPULISAMarkov`** runs a local Moran test in each election and counts how counties move between the states HH (high among high), LL (low among low), LH, HL and not significant. HH and LL are the regional blocs, for example the Democratic urban and coastal counties and the Republican Great Plains and South.\n\nThe **Local Moran reference** choice matters: on each year\'s own mean, "high" is relative to that election; pooled, "high" means above the 24-year mean, so a national shift moves counties between states. The test is analytic (a z-test), not a permutation test, so p-values differ from esda\'s permutation inference. **LISA significance level** sets the threshold.',
      options: {matrix: 'lisa', mapView: 'lisa', year: 6, lisaSignificance: 0.05},
      highlight: {readout: 'stayLisa'},
      controls: ['lisaMoments', 'lisaSignificance'],
      readouts: ['stayLisa']
    },
    {
      id: 'movers',
      title: 'Who moved? Class changes between elections',
      headline: 'County class changes form regional patterns',
      textAlternative:
        'Purple and orange counties mark downward and upward class changes to the next election, with gray counties unchanged.',
      body: 'The *Move to the next election* view of **Map shows** colors each county by how many classes it changed by to the next election: purple down, orange up, gray the same. The dataset notes that 239 counties changed winner from 2012 to 2016, 78 from 2016 to 2020 and 74 from 2020 to 2024, and that the median Republican two-party swing from 2020 to 2024 was +1.5 points, largest in Texas border counties.\n\nStep the **Election** slider through 2000 to 2020 and compare the pattern of movers: is it scattered or regional? That regionality is what the spatial Markov and LISA Markov quantify. Set **Elections included** to *2000 to 2020* to keep Indiana, Louisiana, New Jersey and New York in the analysis (a rebuild).',
      options: {mapView: 'move', year: 5, matrix: 'pooled'},
      controls: ['mapView', 'year', 'through'],
      readouts: ['yearShown']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      headline: 'Turnout transitions depend on population denominator',
      textAlternative:
        'Counties are classified by votes per 2020 resident, exposing denominator effects across election years.',
      body: 'Counties are not equal voters and their boundaries are not equal areas; results are about *counties*. Pooled equal-interval breaks make the classes sensitive to the extremes; quantile breaks hide national swings. Connecticut is reported by planning region in the boundaries but by legacy county in the returns, so it is blank. Turnout is votes per 2020 resident, so it is inflated for fast-growing counties in early years.\n\nTry: **Variable** *Turnout proxy* (a sequential palette); **Transition step** two elections; **Class method** *Quantile*; and the pooled **Local Moran reference**. Compare the readouts across variants: the stay probability is a single number worth comparing.',
      options: {variable: 'turnout', mapView: 'class', year: 6},
      controls: ['variable', 'periodLag', 'classMethod', 'lisaMoments'],
      readouts: ['stay']
    }
  ]
});
