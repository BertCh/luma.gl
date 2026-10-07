// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type OptionSpec} from '../scene';
import {REFERENCE_PRESETS, SIMILARITY_ATTRIBUTES} from './b5-variables';
import type {SimilarPlacesOptions} from './similar-places.compute';

const weightOptions: OptionSpec<SimilarPlacesOptions>[] = SIMILARITY_ATTRIBUTES.map(
  (attribute, k) => ({
    kind: 'slider' as const,
    id: `weight${k}` as const,
    label: attribute.label,
    group: 'Attribute weights',
    apply: 'param' as const,
    min: 0,
    max: 3,
    step: 0.25,
    default: 1,
    help: `Weight of ${attribute.label.toLowerCase()} (${attribute.unit}) in the distance. 0 ignores it; 3 counts it three times. Standardised before weighting.`
  })
);

export default defineScene<SimilarPlacesOptions>({
  id: 'similar-places',
  title: 'Which counties are most like yours?',
  chapter: 'statistics',
  order: 5,
  summary:
    'Pick a county and rank every other US county by how closely twelve census and health attributes match, with your own weights. Similar places need not be neighbours.',
  contributors: ['GPUSimilarLocations'],
  datasets: [{id: 'us-counties', role: 'county polygons and twelve attributes per county'}],
  initialView: {longitude: -96.5, latitude: 38.2, zoom: 3.55},

  options: [
    {
      kind: 'select',
      id: 'preset',
      label: 'Reference county',
      group: 'Reference',
      apply: 'param',
      default: 'loudoun',
      help: 'Choosing here replaces the selection. Or click any county on the map to make it the reference.',
      options: REFERENCE_PRESETS.map(preset => ({value: preset.value, label: preset.label}))
    },
    {
      kind: 'toggle',
      id: 'multiSelect',
      label: 'Click adds to the reference set',
      group: 'Reference',
      apply: 'param',
      default: false,
      help: 'With several references the target is the mean of their standardised attributes: "places like both of these". Click a reference again to remove it (up to six).'
    },
    {
      kind: 'select',
      id: 'direction',
      label: 'Rank',
      group: 'Ranking',
      apply: 'param',
      default: 'most',
      help: 'Most similar ranks the nearest counties first; least similar ranks the farthest first.',
      options: [
        {value: 'most', label: 'Most similar first'},
        {value: 'least', label: 'Least similar first'}
      ]
    },
    {
      kind: 'slider',
      id: 'resultCount',
      label: 'Matches listed',
      group: 'Ranking',
      apply: 'param',
      min: 1,
      max: 32,
      step: 1,
      default: 12,
      help: 'How many top-ranked counties are written to the compact topIds output and marked on the map (up to the compiled capacity of 32).'
    },
    {
      kind: 'toggle',
      id: 'excludeReference',
      label: 'Exclude the reference from the ranking',
      group: 'Ranking',
      apply: 'param',
      default: true,
      help: 'Keeps the selected counties out of their own ranking. Off: they rank first with distance 0.'
    },
    {
      kind: 'select',
      id: 'standardization',
      label: 'Standardisation',
      group: 'Ranking',
      apply: 'compile',
      default: 'zscore',
      help: 'How columns are put on a common scale. Z-score: (x - mean) / sd. Percentile rank: robust to outliers but loses magnitude. Compile-time: the rank graph compiles the first time it is chosen.',
      options: [
        {value: 'zscore', label: 'Z-score'},
        {value: 'rank', label: 'Percentile rank'}
      ]
    },
    ...weightOptions,
    {
      kind: 'slider',
      id: 'falloff',
      label: 'Color falloff (counties considered similar)',
      group: 'Display',
      apply: 'param',
      min: 10,
      max: 1000,
      step: 10,
      default: 150,
      help: 'The map colors similarity = exp(-distance / d), with d the distance of the county at this rank. A larger value brightens more counties.'
    },
    {
      kind: 'toggle',
      id: 'showMatches',
      label: 'Mark the matches',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Pink dots at the centroids of the top-ranked counties, drawn straight from the topIds buffer.'
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'Stronger county outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Outline counties more boldly.'
    }
  ],

  readouts: [
    {
      id: 'reference',
      label: 'Reference',
      help: 'The counties whose mean standardised profile is the target.'
    },
    {
      id: 'matches',
      label: 'Closest matches (distance)',
      help: 'Weighted Euclidean distance in standardised attribute space.'
    },
    {
      id: 'distances',
      label: 'Distance spread',
      help: 'Nearest, 10th, 100th, median and farthest ranked county.'
    },
    {id: 'ranked', label: 'Ranked'}
  ],

  legends: state => [
    {
      kind: 'ramp',
      title:
        state.direction === 'most'
          ? 'Similarity to the reference'
          : 'Similarity to the reference (least similar are darkest)',
      ramp: 'viridis',
      extent: [0, 1],
      labels: ['unlike', 'alike'],
      unit: 'exp(-distance / falloff)'
    },
    {
      kind: 'categories',
      title: 'Markers',
      entries: [
        {color: [20, 24, 40, 255], label: 'Reference county (click to change)'},
        {
          color: [220, 40, 100, 255],
          label: `Top ${state.resultCount} ${state.direction === 'most' ? 'most' : 'least'} similar`
        }
      ]
    }
  ],

  snippet: state => `import {
  GPUSimilarLocations, getGPUSimilarLocationsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// attributes: 3,109 counties x 12 float32 (row-major); selection: uint32 per county (1 = reference)
graph.add(new GPUSimilarLocations({
  attributes, attributeCount: 12, selection,
  parameters: similarityParameters,
  standardization: '${state.standardization}',
  maximumResultCount: 32,
  output: {ranks, distances, topIds, count}      // rank 0 = most similar
}));

// Per frame: parameter and selection writes only.
similarityParameters.write(getGPUSimilarLocationsParameterValues({
  resultCount: ${state.resultCount}, direction: '${state.direction}', excludeReference: ${state.excludeReference},
  weights: [${SIMILARITY_ATTRIBUTES.map((_, k) => Number(state[`weight${k}` as `weight${number}`] ?? 1)).join(', ')}]
}, 12));
selection.write(referenceMask);   // click = write one 1 and re-encode`,

  about: {
    what: '`GPUSimilarLocations` ranks every row by its weighted Euclidean distance to a reference row, or to the mean of several reference rows, in standardised attribute space. It validates rows, standardises the columns by z-score or percentile rank, reduces the reference vector, measures distances and orders them with a stable GPU sort.',
    why: 'Analysts looking for peer places (comparable markets, benchmark districts, sites like a successful one) cannot use geography: the best match for a Virginia suburb may be in Colorado. Weights make the question explicit: similar in what respect?',
    howToRead:
      'Brighter counties are closer to the reference in the weighted attribute space. Pink dots mark the listed matches. Hover a county to compare its values with the reference. Matches the CARTO FIND_SIMILAR_LOCATIONS model and the ArcGIS Similarity Search.'
  },

  create: async ctx => (await import('./similar-places.compute')).createSimilarPlaces(ctx),

  story: [
    {
      id: 'question',
      title: 'Which counties are most like Loudoun County, Virginia?',
      body: "Loudoun County, an affluent suburb of Washington, has the highest incomes of the 3,109 counties here. Where else is like it? Its nearest matches are Howard County, Maryland, Williamson County, Tennessee (Nashville), Fairfax, Virginia, Forsyth County, Georgia (Atlanta) and Douglas County, Colorado (Denver): prosperous, educated, family-heavy suburbs of other metros, hundreds of miles away.\n\n`GPUSimilarLocations` standardises the twelve columns, measures the weighted Euclidean distance of every county from Loudoun and ranks them. Bright counties are alike; **pink dots** mark the twelve closest. Hover a county to see its values next to Loudoun's.",
      options: {preset: 'loudoun', resultCount: 12},
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['preset', 'resultCount'],
      readouts: ['matches'],
      highlight: {readout: 'matches'}
    },
    {
      id: 'how-it-works',
      title: 'Distance in standardised space',
      body: 'Each attribute is converted to a z-score, `(x − mean) / sd`, so a thousand dollars of income and a percentage point of poverty compare fairly. The distance to the reference is `sqrt(Σ w_k (z_k − r_k)²)` with a weight *w* per attribute. The ranking is a stable GPU sort of the distances, so ties keep the lowest county order.\n\nTry the weights below, such as **Median household income**, **Poverty (below 150% of the line)** and **Population density (log)**. Setting a weight to 0 ignores that attribute, 3 counts it three times: the **Closest matches (distance)** readout and the map update immediately because weights are parameter writes.',
      options: {resultCount: 12},
      controls: ['weight0', 'weight1', 'weight2'],
      readouts: ['matches', 'distances']
    },
    {
      id: 'weights',
      title: 'Similar in what respect?',
      body: 'Now ask a narrower question: which counties have Loudoun\'s *economy and education*? Set every weight to 0 except **Median household income** (3), **Poverty (below 150% of the line)** (2) and **No high school diploma** (2). Health and housing no longer matter, and the list of matches changes.\n\nThe weights are the model: "similar" means nothing until you say what you care about.',
      options: {
        weight0: 3,
        weight1: 2,
        weight2: 0,
        weight3: 0,
        weight4: 0,
        weight5: 0,
        weight6: 2,
        weight7: 0,
        weight8: 0,
        weight9: 0,
        weight10: 0,
        weight11: 0
      },
      controls: ['weight0', 'weight1', 'weight6'],
      readouts: ['matches']
    },
    {
      id: 'least-similar',
      title: 'The opposite of Loudoun',
      body: 'Flip **Rank** to *Least similar first*. The same distances, ordered the other way: the counties least like an affluent Virginia suburb include Kenedy and Hudspeth counties in Texas, Issaquena County in the Mississippi Delta and Oglala Lakota County on Pine Ridge: poor, sparsely populated, with many mobile homes and little internet. The listed matches now mark the far end of the scale.\n\nChanging the direction is a one-word parameter write; the graph is not rebuilt.',
      options: {
        direction: 'least',
        weight0: 1,
        weight1: 1,
        weight2: 1,
        weight3: 1,
        weight4: 1,
        weight5: 1,
        weight6: 1,
        weight7: 1,
        weight8: 1,
        weight9: 1,
        weight10: 1,
        weight11: 1
      },
      controls: ['direction', 'resultCount'],
      readouts: ['matches']
    },
    {
      id: 'another-reference',
      title: 'Choose another reference, or several',
      body: 'Switch **Reference county** to McDowell County, West Virginia, then **click any county** to set your own. Turn on **Click adds to the reference set** and click two or three counties: the target becomes the *mean* of their standardised profiles, which finds places like all of them at once, a way to define a "type" of place from examples.\n\nEvery click writes the selection buffer and re-encodes the same compiled graph.',
      options: {preset: 'mcdowell', direction: 'most'},
      controls: ['preset', 'multiSelect'],
      readouts: ['reference']
    },
    {
      id: 'standardisation',
      title: 'Z-score or percentile rank?',
      body: 'Z-scores keep the magnitude of differences, so one extreme attribute can dominate. Set **Standardisation** to *Percentile rank*, which replaces each value by its position in the national ordering, which is robust to outliers (population density is log-transformed here for the same reason) but treats a small difference near the median like a large one in the tails.\n\nThis is a compile-time option of `GPUSimilarLocations`: the first time you choose it the panel marks a rebuild, afterwards it switches instantly.',
      options: {standardization: 'rank', preset: 'loudoun'},
      controls: ['standardization'],
      readouts: ['matches']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: 'Similarity is only as good as the attributes: twelve census and health columns say nothing about climate, industry or politics. County averages hide huge internal differences (Cook County is Chicago and its suburbs). Equal weights are an assumption, and Euclidean distance treats correlated attributes (poverty and no insurance) as independent evidence.\n\nTry: **Reference county** Story County, Iowa (a college town) and look for other college towns; Maricopa County, AZ with **Population density (log)** weighted 3; turn on **Click adds to the reference set** and add Boulder and Story together as references.',
      options: {preset: 'story', standardization: 'zscore'},
      controls: ['preset', 'multiSelect', 'weight2'],
      readouts: ['reference']
    }
  ]
});
