// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SpaceTimeClustersOptions} from './space-time-clusters.compute';

const CLUSTER_COLORS = [
  [215, 48, 39],
  [244, 140, 40],
  [140, 60, 190],
  [30, 150, 90],
  [30, 110, 200],
  [200, 170, 20],
  [90, 90, 90]
] as const;

export default defineScene<SpaceTimeClustersOptions>({
  id: 'space-time-clusters',
  title: 'Chicago event pair excess and scan clusters',
  chapter: 'time',
  order: 2,
  summary:
    'Knox, Mantel and Kulldorff scan tests use Chicago nature and complaint records to map close-pair excess and tract-fortnight clusters; volunteer effort, thinning and multiple candidate windows limit interpretation.',
  contributors: ['GPUKnoxTest', 'GPUMantelTest', 'GPUSpatialScanStatistic'],
  datasets: [
    {id: 'chicago-nature', role: 'nature observations'},
    {id: 'chicago-311-rats', role: 'rodent complaints'},
    {id: 'chicago-tracts', role: 'scan zones and residents'}
  ],
  initialView: {longitude: -87.68, latitude: 41.835, zoom: 9.9},
  basemap: ground('night'),
  furniture: {
    title: {
      title: 'Chicago space-time clusters',
      subtitle: 'Pair tests and tract-fortnight scan statistics'
    },
    scaleBar: {units: 'metric'},
    credit: 'iNaturalist contributors; City of Chicago; US Census Bureau',
    caveat: 'Observation effort and candidate-window selection affect significance.'
  },

  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Events',
      group: 'Events',
      apply: 'compile',
      default: 'fungi',
      options: [
        {value: 'fungi', label: 'Fungi (3,762)'},
        {value: 'insects', label: 'Insects (10,353)'},
        {value: 'birds', label: 'Birds (11,198)'},
        {value: 'plants', label: 'Plants (15,258)'},
        {value: 'mammals', label: 'Mammals (1,089)'},
        {value: 'rats', label: '311 rodent complaints (48,636)'}
      ],
      help: 'Which observations are the events. The row count is fixed when the graphs are compiled, so each source compiles once and is cached. The Knox and Mantel tests use at most 24,000 events (every n-th observation in time order); the scan uses all of them.'
    },
    {
      kind: 'select',
      id: 'analysis',
      label: 'Map shows',
      group: 'Events',
      apply: 'param',
      default: 'pairs',
      options: [
        {value: 'pairs', label: 'Pairs: events and near-in-space links'},
        {value: 'scan', label: 'Scan: tracts and clusters'}
      ],
      help: 'Both analyses always run; this only switches which one is drawn.'
    },
    {
      kind: 'slider',
      id: 'spatialRadius',
      label: 'Spatial threshold',
      group: 'Knox and Mantel',
      apply: 'param',
      min: 50,
      max: 1000,
      step: 25,
      default: 150,
      unit: 'm',
      help: 'Two events are "close in space" when they are within this distance. It is the radius of the neighbor search that lists the pairs, so changing it rewrites a buffer and re-lists the pairs without recompiling.'
    },
    {
      kind: 'slider',
      id: 'timeThreshold',
      label: 'Time threshold',
      group: 'Knox and Mantel',
      apply: 'param',
      min: 1,
      max: 90,
      step: 1,
      default: 14,
      unit: 'days',
      help: 'Two events are "close in time" when they are within this many days. Knox counts pairs that are close in both. Mantel ignores the threshold and correlates the continuous gaps.'
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Permutations (Knox, Mantel)',
      group: 'Knox and Mantel',
      apply: 'param',
      min: 19,
      max: 999,
      step: 10,
      default: 499,
      help: 'How many times the event times are shuffled over the fixed locations to build the null distribution. More permutations give a finer pseudo p-value (the smallest is 1 / (permutations + 1)).'
    },
    {
      kind: 'toggle',
      id: 'showSpatialOnly',
      label: 'Show pairs close in space only',
      group: 'Knox and Mantel',
      apply: 'param',
      default: false,
      help: 'Also draws the pairs that are near in space but far apart in time, faintly. Orange links are the pairs Knox counts.'
    },
    {
      kind: 'select',
      id: 'baseline',
      label: 'Expected cases (baseline)',
      group: 'Scan statistic',
      apply: 'param',
      default: 'independence',
      options: [
        {
          value: 'independence',
          label: 'Space x time independence',
          help: 'Each tract keeps its own yearly total and each fortnight keeps its city-wide share. A cluster is a tract group and fortnight that exceed that product.'
        },
        {
          value: 'population',
          label: 'Residents x city-wide timing',
          help: 'Expected cases follow resident population. Tracts with parks and preserves but few residents (the lakefront, forest preserves) look like hot spots.'
        },
        {
          value: 'uniform',
          label: 'Uniform (every tract and fortnight equal)',
          help: 'No model at all: clusters simply follow where the cases are.'
        }
      ],
      help: 'The Poisson model the observed counts are compared with. The baseline is a buffer, so switching it never recompiles.'
    },
    {
      kind: 'slider',
      id: 'maximumPopulationFraction',
      label: 'Largest window (share of baseline)',
      group: 'Scan statistic',
      apply: 'param',
      min: 0.02,
      max: 0.5,
      step: 0.01,
      default: 0.1,
      format: value => `${Math.round(value * 100)}%`,
      help: "A window stops growing when it would hold more than this share of the total baseline. SaTScan's default is 50%; smaller windows find tighter clusters."
    },
    {
      kind: 'slider',
      id: 'maximumWindowZones',
      label: 'Largest window (tracts)',
      group: 'Scan statistic',
      apply: 'param',
      min: 1,
      max: 32,
      step: 1,
      default: 20,
      unit: 'tracts',
      help: 'Windows are the k nearest tracts of each center for every k up to this limit (at most 32, the compile-time bound).'
    },
    {
      kind: 'slider',
      id: 'maximumTimeBuckets',
      label: 'Longest cluster duration',
      group: 'Scan statistic',
      apply: 'param',
      min: 1,
      max: 26,
      step: 1,
      default: 6,
      format: value => `${value} fortnights (${value * 14} days)`,
      help: 'A space-time cylinder spans at most this many consecutive fortnights. 1 finds short bursts; 26 allows year-long clusters.'
    },
    {
      kind: 'select',
      id: 'windowShape',
      label: 'Window shape',
      group: 'Scan statistic',
      apply: 'param',
      default: 'circle',
      options: [
        {value: 'circle', label: 'Circle (never splits equidistant tracts)'},
        {value: 'nearest', label: 'k nearest tracts (ties by index)'}
      ],
      help: 'Both use the same nearest-first window lists. A circle keeps tracts at exactly the same distance together; the nearest shape cuts between them.'
    },
    {
      kind: 'slider',
      id: 'scanPermutations',
      label: 'Monte Carlo replicates',
      group: 'Scan statistic',
      apply: 'param',
      min: 19,
      max: 999,
      step: 10,
      default: 199,
      help: 'Each replicate re-draws the cases under the baseline and records the largest likelihood ratio. The p-value is (1 + replicates at least as extreme) / (replicates + 1). The scan re-runs a moment after the slider stops.'
    },
    {
      kind: 'slider',
      id: 'significance',
      label: 'Significance level',
      group: 'Scan statistic',
      apply: 'param',
      min: 0.005,
      max: 0.2,
      step: 0.005,
      default: 0.05,
      help: 'Only clusters with a p-value at or below this are coloured on the map. The smallest possible p-value is limited by the replicate count.'
    },
    {
      kind: 'select',
      id: 'scanView',
      label: 'Scan map shows',
      group: 'Display',
      apply: 'param',
      default: 'clusters',
      options: [
        {value: 'clusters', label: 'Significant clusters (colour = rank)'},
        {value: 'zone-llr', label: 'Best likelihood ratio of windows centred on each tract'},
        {value: 'rate', label: 'Observations per 1,000 residents, whole year'}
      ],
      help: "The cluster view is computed on the GPU from the cluster list; the likelihood ratio view is the contributor's per-zone output."
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Lower it to read the basemap under the tracts.'
    },
    {
      kind: 'button',
      id: 'reseed',
      label: 'New random seed',
      group: 'Display',
      help: 'Draws new permutations and Monte Carlo replicates. Stable results barely move; fragile ones do.'
    }
  ],

  readouts: [
    {id: 'events', label: 'Events (Knox sample of all)'},
    {id: 'pairs', label: 'Pairs close in space', format: 'integer'},
    {id: 'timeClose', label: 'Pairs close in time (all events)', format: 'integer'},
    {id: 'knoxObserved', label: 'Knox: observed pairs close in both', format: 'integer'},
    {id: 'knoxExpected', label: 'Knox: expected under independence'},
    {id: 'knoxRatio', label: 'Knox: observed / expected'},
    {id: 'knoxPermuted', label: 'Knox: permutation mean ± sd'},
    {id: 'knoxP', label: 'Knox: pseudo p (X >= observed)'},
    {id: 'knoxPoisson', label: 'Knox: classic Poisson p'},
    {id: 'knoxDistribution', label: 'Knox null distribution'},
    {id: 'mantelR', label: 'Mantel r (distance vs time gap)'},
    {id: 'mantelPermuted', label: 'Mantel: permutation mean ± sd'},
    {id: 'mantelP', label: 'Mantel: pseudo p'},
    {id: 'mantelDistribution', label: 'Mantel null distribution'},
    {id: 'pairOverflow', label: 'Pair list overflow'},
    {id: 'scanCases', label: 'Scan: observations in tracts'},
    {id: 'scanClusters', label: 'Scan: clusters'},
    {id: 'cluster1', label: 'Cluster 1 (most likely)'},
    {id: 'cluster2', label: 'Cluster 2'},
    {id: 'cluster3', label: 'Cluster 3'}
  ],

  legends: state =>
    state.analysis === 'pairs'
      ? [
          {
            kind: 'ramp',
            title: 'Event date',
            ramp: 'lajolla',
            extent: [0, 365],
            labels: ['1 Jan', '31 Dec'],
            unit: 'day of 2023'
          },
          {
            kind: 'categories',
            title: 'Links',
            entries: [
              {
                color: [230, 110, 10, 220],
                label: `Close in space (${state.spatialRadius} m) and time (${state.timeThreshold} d)`
              }
            ],
            note: state.showSpatialOnly ? 'Faint links are close in space only.' : undefined
          }
        ]
      : state.scanView === 'clusters'
        ? [
            {
              kind: 'categories',
              title: 'Significant clusters',
              entries: [
                {color: [140, 150, 170, 90], label: 'Not in a significant cluster'},
                ...CLUSTER_COLORS.map((rgb, index) => ({
                  color: [...rgb, 235] as [number, number, number, number],
                  label: index === 0 ? 'Cluster 1 (most likely)' : `Cluster ${index + 1}`
                }))
              ],
              note: 'Tracts in each window; the time span is in the readouts.'
            }
          ]
        : state.scanView === 'zone-llr'
          ? [
              {
                kind: 'ramp',
                id: 'llr',
                title: 'Best log-likelihood ratio of windows centred here',
                ramp: 'inferno',
                extent: 'gpu',
                sqrtScale: true,
                unit: 'LLR',
                format: value => value.toFixed(0)
              }
            ]
          : [
              {
                kind: 'ramp',
                id: 'rate',
                title: 'Observations per 1,000 residents in 2023',
                ramp: 'ylorrd',
                extent: 'gpu',
                sqrtScale: true,
                unit: 'per 1,000',
                labels: ['0', 'top 3% and above']
              }
            ],

  snippet: state => `import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  getGPUSpatialScanParameterValues,
  GPUKnoxTest, GPUMantelTest, GPUNeighborSearch, GPUSpatialScanStatistic
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTemporalReduction} from '@luma.gl/experimental/gpu-dataframe';

// Pairs: events within ${state.spatialRadius} m are the spatial pairs of both tests
graph.add(new GPUNeighborSearch({mode: 'radius', positions, parameters: searchParameters,
  gridSize: [64, 64], weights: pairs, overflow}));
graph.add(new GPUKnoxTest({pairs, times: days, parameters: testParameters,
  maximumPermutations: 999, statistics: knoxStatistics, summary: knoxSummary}));
graph.add(new GPUMantelTest({pairs, times: days, parameters: testParameters,
  maximumPermutations: 999, statistics: mantelStatistics, summary: mantelSummary}));
testParameters.write(getGPUSpaceTimeParameterValues({
  seed: 1, permutations: ${state.permutations}, timeThreshold: ${state.timeThreshold}   // days
}));

// Scan: cases per (tract, fortnight) from a temporal reduction, then Kulldorff's statistic
graph.add(new GPUTemporalReduction({cellIds: tractIds, timestamps: seconds, values: ones,
  parameters: bucketParameters, cellCount: 791, bucketCount: 26, output: reduction}));
graph.add(new GPUSpatialScanStatistic({
  positions: tractCentroids, cases: reduction.counts, baseline,
  timeBuckets: 26, maximumWindowZones: 32, maximumPermutations: 999, maximumClusters: 7,
  parameters: scanParameters, clusterIndices, clusterStatistics, statistics, summary, zoneStatistics
}));
scanParameters.write(getGPUSpatialScanParameterValues({
  seed: 1, permutations: ${state.scanPermutations},
  maximumPopulationFraction: ${state.maximumPopulationFraction}, maximumWindowZones: ${state.maximumWindowZones},
  maximumTimeBuckets: ${state.maximumTimeBuckets}, windowShape: '${state.windowShape}'
}));`,

  about: {
    what: '`GPUKnoxTest` counts event pairs that are close in space and in time and compares the count with the same events given shuffled dates; `GPUMantelTest` correlates spatial distance with time gap over the same pairs; `GPUSpatialScanStatistic` slides space-time cylinders over tracts and fortnights and reports the ones with the largest Poisson likelihood ratio, with Monte Carlo p-values.',
    why: 'Knox and Mantel answer "is there contagion at all" (a fruiting flush, observers returning to a patch) without picking a place. The scan statistic answers "where and when", and its baseline decides what counts as unusual: population, history or nothing.',
    howToRead:
      'On the pairs map, orange links join events that are close in space and time; if there are far more than the permutations produce, the readouts show a ratio well above 1 and a p-value near 1 / (permutations + 1). On the scan map, coloured tracts form significant clusters ranked by likelihood ratio.'
  },

  create: async ctx => (await import('./space-time-clusters.compute')).createSpaceTimeClusters(ctx),

  story: [
    {
      id: 'question',
      title: 'Test short-range space-time association',
      headline: 'Fungi pairs exceed space-time independence',
      textAlternative:
        'Orange links connect Chicago fungi observations within 150 meters and 14 days, over points colored by date.',
      body: '**`GPUKnoxTest`** evaluates short-range space-time association in 3,762 Chicago fungi observations from 2023. It counts pairs within 150 m (**Spatial threshold**) and 14 days (**Time threshold**), then permutes dates over the observed locations to estimate the count under space-time independence. This conditional null preserves the sampled locations and their observer-access pattern.\n\nOrange links are qualifying pairs; points encode observation date from January to December. The readouts report the observed count, permutation expectation and pseudo p-value.',
      controls: ['source', 'spatialRadius', 'timeThreshold'],
      readouts: ['knoxObserved', 'knoxExpected', 'knoxP']
    },
    {
      id: 'knox',
      title: 'Reading the Knox test',
      headline: 'Observed close pairs exceed every permutation',
      textAlternative:
        'Temporal-close and spatial-only fungi links are mapped while the permutation readout places observed pair counts beyond the null.',
      body: 'Knox counts `X` = pairs close in space **and** time. Under space-time independence, `E[X] = S * T / (n (n - 1) / 2)` where `S` is the number of spatial pairs and `T` the number of time-close pairs. For fungi at 150 m and 14 days the observed count is about 16,100 against about 7,400 expected, a ratio near 2.2. No permutation reaches the observed count, so the pseudo p-value is the minimum possible, `1 / (permutations + 1)`, set by **Permutations (Knox, Mantel)**. **Show pairs close in space only** also draws pairs that are near in space but not in time.\n\nThe **Knox null distribution** readout charts the permuted counts. The result combines biological seasonality with repeated observer sampling; this design cannot estimate their separate contributions.',
      options: {showSpatialOnly: true},
      highlight: {readout: 'knoxP'},
      controls: ['showSpatialOnly', 'permutations'],
      readouts: ['knoxRatio', 'knoxP', 'knoxDistribution']
    },
    {
      id: 'thresholds',
      title: 'The answer depends on the thresholds',
      headline: 'Tighter thresholds increase measured pair excess',
      textAlternative:
        'Fungi pair links contract to smaller space-time thresholds while Knox and Mantel statistics update.',
      body: 'Move **Spatial threshold** and **Time threshold**. The spatial radius re-lists candidate pairs, while the time threshold filters the existing pair list, so neither option recompiles the graph. At 150 m and 7 days the ratio is about 3.2; at 600 m and 60 days it approaches 1.2. The statistic therefore describes a selected support rather than a scale-invariant process.\n\n**`GPUMantelTest`** avoids a binary time threshold: it correlates spatial distance with temporal separation over the same pairs. A positive coefficient means spatially closer pairs tend to have smaller time gaps. Its p-value uses the same permutation design. Press **New random seed** to inspect Monte Carlo stability.',
      options: {spatialRadius: 150, timeThreshold: 7},
      highlight: {readout: 'mantelR'},
      controls: ['spatialRadius', 'timeThreshold', 'reseed'],
      readouts: ['knoxRatio', 'mantelR', 'mantelP']
    },
    {
      id: 'scan',
      title: 'Where and when? The scan statistic',
      headline: 'Scan statistics localize tract-fortnight excess',
      textAlternative:
        'Colored Chicago tracts form ranked space-time scan clusters with readouts for dates, observed cases and expected cases.',
      body: 'Knox says "yes, there is interaction"; it does not say where. **`GPUSpatialScanStatistic`** slides cylinders over the 791 census tracts and 26 fortnights: each window is a tract, its nearest neighbors (a circle) and a run of consecutive fortnights. For each it computes a Poisson log-likelihood ratio comparing observed sightings with the number expected from the **baseline**; the largest ratio is the most likely cluster.\n\nThe tract counts come from `GPUTemporalReduction` (observations per tract and fortnight, on the GPU). Colored tracts form clusters whose p-value is at or below the **Significance level** (**Scan map shows** switches the view); the readouts give each cluster\'s tracts, radius, dates and observed versus expected counts.',
      options: {analysis: 'scan'},
      camera: {zoom: 10.1, transitionMs: 1200},
      controls: ['analysis', 'significance', 'scanView'],
      readouts: ['scanClusters', 'cluster1', 'cluster2']
    },
    {
      id: 'baseline',
      title: 'The baseline decides what "unusual" means',
      headline: 'Population baselines shift detected clusters',
      textAlternative:
        'Significant tract clusters move toward high observation rates relative to residents under the population baseline.',
      body: 'The default **independence** baseline gives each tract its own yearly total and each fortnight its citywide share, so a cluster is a burst in a place and time that exceeds both. Set **Expected cases (baseline)** to *Residents x city-wide timing* and clusters can shift toward tracts where sightings are high relative to residents, which favors parks and preserves with few residents; set it to *Uniform* and clusters simply follow where the observations are.\n\nShrink **Largest window (share of baseline)** or **Largest window (tracts)** for tighter clusters, and change **Longest cluster duration** from 6 fortnights to 1 to find single-fortnight bursts. The scan re-runs a moment after you stop dragging; the p-values are limited by the number of **Monte Carlo replicates**.',
      options: {analysis: 'scan', baseline: 'population', maximumWindowZones: 12},
      controls: [
        'baseline',
        'maximumPopulationFraction',
        'maximumWindowZones',
        'maximumTimeBuckets',
        'scanPermutations'
      ],
      readouts: ['scanClusters', 'cluster1']
    },
    {
      id: 'other-events',
      title: 'Try other groups, and know the limits',
      headline: 'Event sources produce different clustering estimates',
      textAlternative:
        'Insect observations produce a distinct network of close pairs and different Knox statistics under the same thresholds.',
      body: 'Switch **Events** to *Insects*, *Birds* or *311 rodent complaints*. Insects swarm in July and birds pass through in May, so each shows its own season in the pairs and in the scan. Birds and plants pile up so densely at a few sites that the pair list may overflow; lower **Spatial threshold** if the readout says so. Rodent complaints differ again: a complaint pair close in space and time may be the same neighborhood reporting the same alley.\n\nCaveats: the sightings are volunteer records, so a cluster can be a bioblitz or a popular trail as much as a biological flush (the City Nature Challenge weekend, 28 April to 1 May 2023, is a built-in time cluster); Knox and Mantel use at most 24,000 events (a regular thinning that keeps the null valid); the scan finds the *most likely* clusters, and secondary ones are tested against the same maxima as in SaTScan; the 365th day is dropped from the fortnight buckets.',
      options: {analysis: 'pairs', source: 'insects', spatialRadius: 100, timeThreshold: 7},
      controls: ['source', 'spatialRadius', 'timeThreshold'],
      readouts: ['knoxRatio', 'knoxP']
    }
  ]
});
