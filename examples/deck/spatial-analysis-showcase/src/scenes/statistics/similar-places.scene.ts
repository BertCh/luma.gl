// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import {REFERENCE_PRESETS, SIMILARITY_ATTRIBUTES} from './b5-variables';
import type {SimilarPlacesLegendData, SimilarPlacesOptions} from './similar-places.compute';
import {getWeights, WEIGHT_SETS} from './similar-places.style';

/** The contiguous US, the frame of every national county step. */
const CONUS_BOUNDS = [-124.8, 24.4, -66.9, 49.4] as const;
/** The Washington region around the reference, for the neighbours step. */
const WASHINGTON_BOUNDS = [-79.3, 37.9, -75.8, 40.3] as const;

const COUNTY_SAMPLE = '3,109 counties of the contiguous US (Alaska and Hawaii not in the data)';

/** The cartouche of one step (every step replaces the whole `title`). */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  sample: COUNTY_SAMPLE,
  chips: ['Estimates', 'Modelled'] as const
});

const SIMILARITY_CREDIT = joinCredits(
  CREDITS.usCensus,
  'CDC Social Vulnerability Index 2022 and PLACES 2024 (public domain)',
  'USDA Economic Research Service (public domain)',
  CREDITS.colorBrewer
);

const furnitureOf = (title: string, subtitle: string) => ({
  ...NATIONAL_FURNITURE,
  title: cartouche(title, subtitle),
  credit: SIMILARITY_CREDIT
});

/** Weight options with `1` on the listed attributes and `0` elsewhere (all 1 without a list). */
function weightValues(indices?: readonly number[]): Record<`weight${number}`, number> {
  const values: Record<string, number> = {};
  getWeights(SIMILARITY_ATTRIBUTES.length, indices).forEach((weight, k) => {
    values[`weight${k}`] = weight;
  });
  return values as Record<`weight${number}`, number>;
}

/** Labels of the reference presets, checked against the data (Loudoun has the highest income). */
const PRESET_LABELS: Record<string, string> = {
  loudoun: 'Loudoun, VA (highest median income)',
  cook: 'Cook, IL (Chicago)',
  mcdowell: 'McDowell, WV (lowest median income)',
  boulder: 'Boulder, CO',
  maricopa: 'Maricopa, AZ (Phoenix)',
  robeson: 'Robeson, NC',
  story: 'Story, IA (Ames)'
};

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

function getLegends(
  state: SimilarPlacesOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const shared = data['similarPlaces'] as SimilarPlacesLegendData | undefined;
  if (!shared) return [];
  const {table, counts, missingCount, referenceName, referenceCount, groundIsDark} = shared;
  const tone = groundIsDark ? 'dark' : 'light';
  const classes = getClassTableLegend(table, {
    title: `Similarity rank to ${referenceName}`,
    id: 'rank-classes',
    counts,
    interactive: true,
    layout: 'list',
    note: `Weighted distance on ${state.standardization === 'zscore' ? 'z-scores' : 'percentile ranks'}. The reference is excluded from its own ranking.`
  });
  const ink = hexToRgba(MAP_INK[tone].ink);
  const entries: Extract<LegendSpec, {kind: 'categories'}>['entries'][number][] = [
    {
      color: hexToRgba(MAP_INK[tone].signal),
      label: referenceCount > 1 ? 'Reference counties (their mean profile)' : 'Reference county',
      shape: 'ring'
    },
    {
      color: ink,
      label: `Numbered: ${state.direction === 'most' ? 'closest' : 'least similar'} ${state.resultCount}`,
      shape: 'dot'
    }
  ];
  if (state.showArcs) {
    entries.push({color: ink, label: 'Great-circle line to a match', shape: 'line'});
  }
  return [
    {...classes, noData: {...classes.noData, count: missingCount}},
    {kind: 'categories', title: 'Marks', entries, layout: 'list'}
  ];
}

function getSnippet(state: SimilarPlacesOptions): string {
  const weights = SIMILARITY_ATTRIBUTES.map((_, k) =>
    Number(state[`weight${k}` as `weight${number}`] ?? 1)
  ).join(', ');
  return `import {
  GPUSimilarLocations, getGPUSimilarLocationsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// attributes: 3,109 counties x 12 float32 (row-major); selection: uint32 per county (1 = reference)
graph.add(new GPUSimilarLocations({
  attributes, attributeCount: 12, selection,
  parameters: similarityParameters,
  standardization: '${state.standardization}',   // compile-time
  maximumResultCount: 32,
  output: {ranks, distances, topIds, count}      // rank 0 = most similar
}));

// A small kernel turns ranks into rank classes (0..${state.resultCount - 1} | <50 | <200 | <600 | rest)
// and the polygon layer draws them as a category palette.

// Per frame: parameter and selection writes only.
similarityParameters.write(getGPUSimilarLocationsParameterValues({
  resultCount: ${state.resultCount}, direction: '${state.direction}', excludeReference: ${state.excludeReference},
  weights: [${weights}]
}, 12));
selection.write(referenceMask);   // a click writes one 1 and re-encodes`;
}

export default defineScene<SimilarPlacesOptions>({
  id: 'similar-places',
  title: 'Which counties are most like yours?',
  chapter: 'statistics',
  order: 6,
  summary:
    'Pick a county and rank every other US county by how closely twelve census and health attributes match, with your own weights. Similar places need not be neighbours.',
  contributors: ['GPUSimilarLocations'],
  datasets: [
    {id: 'us-counties', role: 'county polygons and twelve attributes per county'},
    {id: 'us-states', role: 'state lines over the counties'}
  ],
  initialView: {longitude: -96, latitude: 38.3, zoom: 3.9},

  options: [
    {
      kind: 'select',
      id: 'preset',
      label: 'Reference county',
      group: 'Reference',
      apply: 'param',
      default: 'loudoun',
      help: 'Choosing here replaces the reference. Or set On map click to Set or Add and click any county.',
      options: [
        ...REFERENCE_PRESETS.map(preset => ({
          value: preset.value as string,
          label: PRESET_LABELS[preset.value] ?? preset.label
        })),
        {value: 'custom', label: 'Chosen on the map'}
      ]
    },
    {
      kind: 'select',
      id: 'click',
      label: 'On map click',
      group: 'Reference',
      apply: 'param',
      display: 'segmented',
      default: 'inspect',
      help: 'Inspect: compare a county with the reference (profile chart). Set: make it the reference. Add: add it to the reference set (up to six); the target is then the mean of their standardised attributes, and clicking a reference again removes it.',
      options: [
        {value: 'inspect', label: 'Inspect'},
        {value: 'set', label: 'Set'},
        {value: 'add', label: 'Add'}
      ]
    },
    {
      kind: 'select',
      id: 'direction',
      label: 'Rank order',
      group: 'Ranking',
      apply: 'param',
      display: 'segmented',
      default: 'most',
      help: 'Most similar ranks the nearest counties first; least similar ranks the farthest first. The map switches from green to purple classes so far is never read as near.',
      options: [
        {value: 'most', label: 'Most similar'},
        {value: 'least', label: 'Least similar'}
      ]
    },
    {
      kind: 'slider',
      id: 'resultCount',
      label: 'Matches listed',
      group: 'Ranking',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 12,
      help: 'How many top-ranked counties are written to the compact topIds output, numbered on the map and listed in the card. The first rank class is exactly this set.'
    },
    {
      kind: 'select',
      id: 'standardization',
      label: 'Standardisation',
      group: 'Ranking',
      apply: 'compile',
      display: 'segmented',
      default: 'zscore',
      help: 'How columns are put on a common scale. Z-score: (x - mean) / sd, which keeps magnitudes. Percentile rank: robust to outliers but loses magnitude. Compile-time: the rank graph compiles the first time it is chosen.',
      options: [
        {value: 'zscore', label: 'Z-score'},
        {value: 'rank', label: 'Percentile rank'}
      ]
    },
    {
      kind: 'toggle',
      id: 'excludeReference',
      label: 'Exclude the reference from the ranking',
      group: 'Ranking',
      apply: 'param',
      default: true,
      expert: true,
      help: 'Keeps the chosen counties out of their own ranking. Off: they rank first with distance 0.'
    },
    {
      kind: 'preset',
      id: 'weightSet',
      label: 'Weight set',
      group: 'Attribute weights',
      help: 'Each chip writes every weight explicitly. Economy: income, poverty, schooling and internet access. Health: diabetes and obesity.',
      presets: [
        {
          label: 'Economy only',
          values: weightValues([...WEIGHT_SETS.economy]) as Partial<SimilarPlacesOptions>
        },
        {
          label: 'Health only',
          values: weightValues([...WEIGHT_SETS.health]) as Partial<SimilarPlacesOptions>
        },
        {label: 'All equal', values: weightValues() as Partial<SimilarPlacesOptions>}
      ]
    },
    ...weightOptions,
    {
      kind: 'toggle',
      id: 'showMatches',
      label: 'Number the matches',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Numbered markers and ink outlines on the counties of the first rank class, in rank order. The card lists the same names.'
    },
    {
      kind: 'toggle',
      id: 'showArcs',
      label: 'Show distances',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws a dashed great-circle line from the reference to each match, and a ring at the median distance of the matches. Needs a single reference.'
    },
    {
      kind: 'toggle',
      id: 'showNeighbours',
      label: 'Label the nearest neighbours',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Names the counties geographically nearest the reference with their similarity rank. Needs a single reference.'
    },
    {
      kind: 'toggle',
      id: 'outlineSelected',
      label: 'Outline the inspected match',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Outlines the inspected county (rank 1 until you click another) with a note on the attribute that contributes most to its distance.'
    }
  ],

  readouts: [
    {
      id: 'reference',
      label: 'Reference',
      help: 'The counties whose mean standardised profile is the target.'
    },
    {
      id: 'ranked',
      label: 'Counties ranked',
      format: 'integer',
      emphasis: 'tile',
      help: 'Counties with every attribute present that are not excluded as references.'
    },
    {
      id: 'matchList',
      label: 'The matches, with distance from the reference',
      layout: 'block',
      help: 'Numbered as on the map: great-circle distance between county label points.'
    },
    {
      id: 'medianKm',
      label: 'Median distance of the matches',
      emphasis: 'tile',
      help: 'Great-circle distance from the reference (the nearest reference when there are several) to each listed match, then the median.'
    },
    {
      id: 'neighbourKm',
      label: 'Median distance of the nearest counties',
      help: 'The same count of geographically nearest counties, for comparison with the matches.'
    },
    {
      id: 'neighbourRank',
      label: 'Median similarity rank of the nearest counties',
      emphasis: 'tile',
      help: 'Similarity rank of each geographically nearest county, then the median. A random county ranks near the middle of the list.'
    },
    {
      id: 'neighbourHits',
      label: 'Nearest counties that are also matches',
      emphasis: 'tile'
    },
    {
      id: 'neighbourList',
      label: 'The six nearest counties and their similarity rank',
      layout: 'block'
    },
    {
      id: 'selectedMatch',
      label: 'Inspected county',
      help: 'Click a county with On map click set to Inspect. Rank 1 until then.'
    },
    {id: 'topAttribute', label: 'Largest term of its distance'},
    {
      id: 'profile',
      label: 'Standardised attributes, reference against county',
      kind: 'chart',
      help: 'Each weighted attribute, sorted by its share of the distance.'
    },
    {id: 'contribution', label: 'Share of the squared distance', kind: 'chart'},
    {
      id: 'persist',
      label: 'Matches shared with the equal-weight z-score ranking',
      emphasis: 'tile',
      help: 'How many of the listed matches are also in the match set of the default ranking: same reference, equal weights, z-scores. Computed on the CPU from topIds.'
    },
    {
      id: 'topDistance',
      label: 'Distance of the first county in the list',
      help: 'Weighted Euclidean distance in standardised units (percentile units for percentile ranks).'
    },
    {
      id: 'farSouth',
      label: 'Listed counties in the South',
      emphasis: 'tile',
      help: 'Counties of the Census Bureau South region (including DC) among the listed.'
    },
    {
      id: 'strongestPair',
      label: 'Most correlated pair of weighted attributes',
      help: 'Pearson correlation of the z-scores over all counties. Correlated attributes count the same evidence twice in a Euclidean distance.'
    },
    {id: 'cpuCheck', label: 'CPU cross-check', hood: true},
    {
      id: 'spread',
      label: 'Distance spread',
      hood: true,
      help: 'Nearest, median and farthest ranked county, in standardised units.'
    }
  ],

  pipeline: [
    {
      id: 'standardise',
      label: 'Standardise',
      detail: 'Each attribute to a z-score or a percentile rank over the valid counties'
    },
    {
      id: 'reference',
      label: 'Reference',
      detail: 'The mean standardised profile of the selected counties'
    },
    {
      id: 'distance',
      label: 'Distance',
      detail: 'Weighted Euclidean distance of every county to the reference'
    },
    {
      id: 'sort',
      label: 'Sort',
      detail: 'Stable GPU sort of the distance bits; ties keep the lowest id'
    },
    {
      id: 'ranks',
      label: 'Ranks',
      detail: 'Ranks, distances, topIds and the rank classes the map draws'
    }
  ],

  legends: getLegends,

  basemap: ground('paperSheet'),
  furniture: furnitureOf(
    'Which counties are most like Loudoun?',
    'Rank by weighted distance across 12 attributes'
  ),

  snippet: getSnippet,

  about: {
    what: '`GPUSimilarLocations` is the ArcGIS Similarity Search method on the GPU. It validates rows, standardises each attribute (z-score or percentile rank), averages the standardised rows of the reference into one target, measures the weighted Euclidean distance of every row to it and orders the rows with a stable sort. A small kernel then turns ranks into rank classes for the map.',
    why: 'Analysts looking for peer places (comparable markets, benchmark districts, sites like a successful one) cannot use geography alone: the best match for a Virginia suburb may be in Colorado. Weights make the question explicit: similar in what respect?',
    howToRead:
      'Darker counties rank closer to the reference; the first class is the numbered match set. Colour encodes rank, not distance: classes are Top N, then 50, 200 and 600, then the rest, in green (most similar) or purple (least similar first). Dashed lines join the reference to each match. Limits: twelve census and health attributes say nothing about climate or industry; county averages hide differences inside counties; correlated attributes (poverty and no internet) count more than once; the ArcGIS Similarity Search tool is the reference for the method.'
  },

  create: async ctx => (await import('./similar-places.compute')).createSimilarPlaces(ctx),

  story: [
    {
      id: 'twins',
      title: 'Which counties are most like Loudoun?',
      headline: "Loudoun's twins are scattered across the country",
      textAlternative:
        'Map of the contiguous US in green rank classes: the numbered, darkest counties lie around Washington, New York, Nashville, Atlanta, Dallas, Denver and the San Francisco Bay, joined to Loudoun County by dashed lines.',
      body: '`GPUSimilarLocations` standardises every attribute, then ranks the other **{{ranked}}** counties by weighted distance from Loudoun, the highest-income county in the data. The numbered counties are the closest. Their median distance from Loudoun is **{{medianKm}}**; its nearest counties are a median **{{neighbourKm}}** away. Change **Matches listed** or **Show distances**.\n\n*Feature space is not geographic space.*',
      options: {preset: 'loudoun', resultCount: 12, showArcs: true},
      optionsMode: 'fresh',
      controls: ['resultCount', 'showArcs'],
      readouts: ['medianKm', 'neighbourKm', 'matchList', 'ranked'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      basemap: ground('paperSheet'),
      furniture: furnitureOf(
        'Which counties are most like Loudoun?',
        'Rank by weighted distance across 12 attributes'
      ),
      annotations: labelsFor(US, ['state-tn', 'state-ga', 'state-co', 'state-tx', 'state-ca']),
      stage: 'ranks'
    },
    {
      id: 'neighbours',
      title: "Are Loudoun's neighbours its twins?",
      headline: 'Near neighbours resemble Loudoun, yet few are twins',
      textAlternative:
        'The Washington region in green rank classes: labels give the similarity rank of the counties nearest Loudoun, some in the match set and others far down the ranking.',
      body: "Of the counties nearest Loudoun, **{{neighbourHits}}** are among its matches and their median rank is **{{neighbourRank}}**: neighbours do resemble it, but rarely best. The labels give each one's rank. Toggle **Label the nearest neighbours** and change **Matches listed**.\n\n*Tobler's first law, tested in attribute space.*",
      options: {preset: 'loudoun', resultCount: 12, showNeighbours: true},
      optionsMode: 'fresh',
      controls: ['showNeighbours', 'resultCount'],
      readouts: ['neighbourRank', 'neighbourHits', 'neighbourList'],
      camera: {bounds: WASHINGTON_BOUNDS, transitionMs: 1600},
      furniture: furnitureOf(
        "Are Loudoun's neighbours its twins?",
        'Similarity rank of the nearest counties'
      ),
      stage: 'ranks'
    },
    {
      id: 'why',
      title: 'Why does a county match?',
      headline: 'Every match is a sum of differences',
      textAlternative:
        'The national map with the first match outlined, and a dumbbell chart of its standardised attributes against Loudoun, sorted by share of the distance.',
      body: "A distance is a sum over attributes of weight times the squared difference in standard deviations. The chart sets **{{selectedMatch}}** against Loudoun: **{{topAttribute}}**. Loudoun's income lies far above the national mean, so income can dominate. With **On map click** on Inspect, click any county; toggle **Show distances**.\n\n*Standardising puts attributes on one scale; it does not make them equally informative.*",
      options: {
        preset: 'loudoun',
        resultCount: 12,
        click: 'inspect',
        outlineSelected: true,
        showArcs: true
      },
      optionsMode: 'fresh',
      controls: ['click', 'showArcs'],
      readouts: ['profile', 'contribution', 'selectedMatch', 'topAttribute'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      furniture: furnitureOf(
        'Why does each county match?',
        'Standardised attributes, reference against county'
      ),
      stage: 'distance'
    },
    {
      id: 'weights',
      title: 'Similar in what respect?',
      headline: 'Change the weights, change the twins',
      textAlternative:
        'The national map with matches for economic attributes only; the numbered counties differ from the equal-weight set.',
      body: 'Weights say what counts. Under this setting **{{persist}}** of the equal-weight twins remain, and the matches lie a median **{{medianKm}}** from Loudoun. Pick another **Weight set**, or move **Median household income**, and watch the list.\n\n*A similarity search answers only the question its weights ask.*',
      options: {
        preset: 'loudoun',
        resultCount: 12,
        showArcs: true,
        ...weightValues([...WEIGHT_SETS.economy])
      },
      optionsMode: 'fresh',
      controls: ['weightSet', 'weight0'],
      readouts: ['persist', 'medianKm', 'matchList'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: furnitureOf(
        'Which attributes should count?',
        'Top matches under the chosen weights'
      ),
      stage: 'distance'
    },
    {
      id: 'far-end',
      title: 'The least alike counties',
      headline: 'The least alike counties lie mostly in the South',
      textAlternative:
        'The national map in purple rank classes, least similar first: the numbered counties are concentrated across the South and Southwest, with three labelled by the attribute that sets them apart.',
      body: 'Unlike is a kind of place too. With **Rank order** on Least similar, the same distances rank from the other end: **{{farSouth}}** of the listed counties lie in the South, and the first sits **{{topDistance}}** from Loudoun. Notes name what separates each. Raise **Matches listed** to see more.\n\n*The far end of a ranking is as informative as the near end.*',
      options: {preset: 'loudoun', direction: 'least', resultCount: 12},
      optionsMode: 'fresh',
      controls: ['direction', 'resultCount'],
      readouts: ['farSouth', 'topDistance', 'matchList'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: furnitureOf(
        'Which counties are least like Loudoun?',
        'Rank from the least similar end'
      ),
      stage: 'sort'
    },
    {
      id: 'your-own',
      title: 'Choose your own reference',
      headline: 'Z-scores and percentile ranks pick different twins',
      textAlternative:
        'The national map for a reference county of your choice, with its numbered matches in green rank classes.',
      body: 'Pick a **Reference county**, or set **On map click** to Set or Add to blend several into one target. Switch **Standardisation** to percentile ranks: **{{persist}}** of the z-score twins persist and the closest match sits **{{topDistance}}** away. Correlated attributes count twice; the strongest pair is **{{strongestPair}}**. County averages hide what happens inside (see *A vulnerability index you can reweight*).\n\n*Every scaler is a modelling choice.*',
      options: {preset: 'story', resultCount: 12, click: 'set', showArcs: true},
      optionsMode: 'fresh',
      controls: ['preset', 'click', 'standardization'],
      readouts: ['persist', 'topDistance', 'strongestPair', 'matchList'],
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: furnitureOf('Which counties are like yours?', 'Your reference, your scaler'),
      stage: 'standardise'
    }
  ]
});
