// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import type {JobAccessibilityOptions} from './job-accessibility.compute';

const options: readonly OptionSpec<JobAccessibilityOptions>[] = [
  {
    kind: 'toggle',
    id: 'transit',
    label: 'Ride the CTA',
    group: 'Travel',
    apply: 'param',
    default: true,
    help: 'On: walk plus the CTA bus and rail hop graph. Off: walking only. Rewrites the edge weights (boarding edges become impassable) and re-runs the cost matrix.'
  },
  {
    kind: 'slider',
    id: 'walkSpeed',
    label: 'Walking speed',
    group: 'Travel',
    apply: 'param',
    min: 0.8,
    max: 1.8,
    step: 0.05,
    default: 1.34,
    unit: 'm/s',
    format: value => `${value.toFixed(2)} m/s (${(value * 3.6).toFixed(1)} km/h)`,
    help: 'Pace on sidewalks and to and from stops. 1.34 m/s is the common planning value of 4.8 km/h. Rewrites weights and re-runs the matrix.'
  },
  {
    kind: 'slider',
    id: 'waitFactor',
    label: 'Waiting at stops',
    group: 'Travel',
    apply: 'param',
    min: 0,
    max: 2,
    step: 0.1,
    default: 1,
    unit: 'x',
    format: value => (value === 0 ? 'no wait' : `${value.toFixed(1)}x half the headway`),
    help: 'Boarding costs the expected wait: half the average headway of all weekday departures from the stop, between 45 s and 15 min, times this factor. 0 assumes perfect timing.'
  },
  {
    kind: 'slider',
    id: 'matrixLimitMinutes',
    label: 'Search horizon (matrix cost limit)',
    group: 'Travel',
    apply: 'param',
    min: 20,
    max: 90,
    step: 5,
    default: 60,
    unit: 'min',
    help: 'Cost limit of every search in the matrix: entries above it stay +Infinity. The scoring threshold cannot exceed it. Re-runs the matrix.'
  },
  {
    kind: 'slider',
    id: 'maxSnapDistance',
    label: 'Maximum snap distance',
    group: 'Snapping',
    apply: 'param',
    min: 50,
    max: 2000,
    step: 50,
    default: 500,
    unit: 'm',
    help: 'Opportunities farther than this from every walkable street snap to nothing and drop out of the matrix. A per-frame value of GPUNetworkSnapping.'
  },
  {
    kind: 'select',
    id: 'snapSearch',
    label: 'Snapping search',
    group: 'Snapping',
    apply: 'compile',
    default: 'exact',
    options: [
      {value: 'exact', label: 'Exact scan of every edge'},
      {value: 'bvh', label: 'Bounding-volume hierarchy (candidateCapacity)'}
    ],
    help: 'Exact scans every edge for every point and needs no radius. The BVH path probes a spatial index within the maximum snap distance, which wins with many points. Same answers, compile-time choice.'
  },
  {
    kind: 'select',
    id: 'seedDirection',
    label: 'Seed direction',
    group: 'Snapping',
    apply: 'compile',
    default: 'both',
    options: [
      {value: 'both', label: 'Both ends of the snapped edge'},
      {value: 'forward', label: 'Forward end only'},
      {value: 'reverse', label: 'Reverse end only'}
    ],
    help: 'Which endpoints of the snapped edge start the search. Both is right for walking. Forward or reverse mimics a one-way street: the opportunity can only be entered from one end.'
  },
  {
    kind: 'select',
    id: 'opportunityRows',
    label: 'Opportunity tracts (matrix rows)',
    group: 'Matrix',
    apply: 'compile',
    default: '256',
    options: [
      {value: '128', label: '128 retained tracts'},
      {value: '256', label: '256 retained tracts'},
      {value: '384', label: '384 retained tracts'}
    ],
    help: 'The job-richest census tracts, one shortest-path search each. The row count is the shape of the matrix, so changing it rebuilds both graphs. Scratch and matrix memory grow with it.'
  },
  {
    kind: 'select',
    id: 'laneCount',
    label: 'Searches per batch (lanes)',
    group: 'Matrix',
    apply: 'compile',
    default: '32',
    options: [
      {value: '8', label: '8 lanes'},
      {value: '16', label: '16 lanes'},
      {value: '32', label: '32 lanes (default)'},
      {value: '64', label: '64 lanes'}
    ],
    help: 'How many rows are searched together in one lane-expanded batch. The matrix is bit-identical for every value; more lanes mean fewer graph nodes and more scratch memory. Compile-time.'
  },
  {
    kind: 'select',
    id: 'measure',
    label: 'Accessibility measure',
    group: 'Scoring',
    apply: 'param',
    default: 'cumulative',
    options: [
      {value: 'cumulative', label: 'Cumulative opportunities within the threshold'},
      {value: 'gravity-exponential', label: 'Gravity, exponential decay'},
      {value: 'gravity-power', label: 'Gravity, power decay'},
      {value: 'two-step', label: '2SFCA (jobs per competing worker)'}
    ],
    help: 'Which of the three outputs of GPUNetworkAccessibility is drawn. All of them are scored from the same retained matrix, and the decay code is part of a four-word parameter buffer.'
  },
  {
    kind: 'slider',
    id: 'thresholdMinutes',
    label: 'Travel-time threshold',
    group: 'Scoring',
    apply: 'param',
    min: 5,
    max: 90,
    step: 5,
    default: 45,
    unit: 'min',
    help: 'Opportunities farther than this contribute nothing. Must not exceed the search horizon. Re-scores the matrix, no new search.'
  },
  {
    kind: 'slider',
    id: 'beta',
    label: 'Exponential decay (beta)',
    group: 'Scoring',
    apply: 'param',
    min: 0.1,
    max: 4,
    step: 0.05,
    default: 1,
    unit: 'per 10 min',
    disabledWhen: state => state.measure !== 'gravity-exponential',
    help: 'f(c) = exp(-beta * c), with c in units of 10 minutes. At 1 an opportunity 10 minutes away counts 37%, 20 minutes away 14%.'
  },
  {
    kind: 'slider',
    id: 'powerExponent',
    label: 'Power decay exponent',
    group: 'Scoring',
    apply: 'param',
    min: 0.5,
    max: 3,
    step: 0.1,
    default: 1.5,
    disabledWhen: state => state.measure !== 'gravity-power',
    help: 'f(c) = (c / c0)^-beta with c floored at the minimum cost below. Heavier tails than the exponential: far opportunities still count a little.'
  },
  {
    kind: 'slider',
    id: 'minimumCostSeconds',
    label: 'Power decay floor',
    group: 'Scoring',
    apply: 'param',
    min: 10,
    max: 600,
    step: 10,
    default: 60,
    unit: 's',
    disabledWhen: state => state.measure !== 'gravity-power',
    help: 'minimumCost: travel times below it are treated as this value, so an opportunity right next door does not get an infinite weight.'
  },
  {
    kind: 'toggle',
    id: 'showTransit',
    label: 'CTA rail and bus lines',
    group: 'Layers',
    apply: 'param',
    default: true,
    help: 'The L lines in their official colours and the bus routes faintly.'
  },
  {
    kind: 'toggle',
    id: 'showStops',
    label: 'Stops',
    group: 'Layers',
    apply: 'param',
    default: false,
    help: 'Bus stops (small) and rail stations (bright).'
  },
  {
    kind: 'toggle',
    id: 'showOpportunities',
    label: 'Opportunities (job tracts)',
    group: 'Layers',
    apply: 'param',
    default: true,
    help: 'The tract centroids that carry the jobs, with their snapped position on the street.'
  },
  {
    kind: 'toggle',
    id: 'showSnaps',
    label: 'Snap lines',
    group: 'Layers',
    apply: 'param',
    default: false,
    help: 'Orange segments from each opportunity to its snapped point on the nearest walkable edge.'
  }
];

// Published comparison tables. The renderer, legend and tooltip deliberately share these breaks;
// they are not a quantile ramp that shifts as a control changes.
const CUMULATIVE_CLASSES = makeClassTable({
  breaks: [1, 5_000, 25_000, 100_000, 250_000],
  scheme: 'YlGnBu',
  reverse: true,
  labels: ['0 jobs', '1–5k jobs', '5–25k jobs', '25–100k jobs', '100–250k jobs', '>250k jobs'],
  unit: 'jobs',
  noData: {color: [130, 130, 130, 255], label: 'zero / no reachable jobs'},
  method: 'Authored cumulative opportunity classes; zero is neutral grey.'
});
const GRAVITY_CLASSES = makeClassTable({
  breaks: [5_000, 20_000, 60_000, 150_000],
  scheme: 'YlGnBu',
  reverse: true,
  unit: 'decay-weighted jobs',
  noData: {color: [130, 130, 130, 255], label: 'zero / no reachable jobs'},
  method: 'Frozen comparison classes derived from the retained opportunity rows.'
});
const TWO_STEP_CLASSES = makeClassTable({
  breaks: [2, 8, 20, 50],
  scheme: 'YlGnBu',
  reverse: true,
  unit: 'jobs per 1,000 competing workers',
  noData: {color: [130, 130, 130, 255], label: 'zero / no reachable jobs'},
  method: 'Frozen 2SFCA comparison classes; zero is neutral grey.'
});

export default defineScene<JobAccessibilityOptions>({
  id: 'job-accessibility',
  title: 'How many jobs can you reach without a car?',
  chapter: 'networks',
  order: 3,
  summary:
    'Cumulative, gravity and two-step job accessibility for every Chicago intersection on foot and by CTA, scored from one cost matrix of the 256 job-richest census tracts.',
  contributors: ['GPUNetworkSnapping', 'GPUNetworkCostMatrix', 'GPUNetworkAccessibility'],
  datasets: [
    {id: 'chicago-roads', role: 'walkable streets'},
    {id: 'cta-transit', role: 'stops and hops'},
    {id: 'chicago-tracts', role: 'jobs and workers'}
  ],
  initialView: {longitude: -87.69, latitude: 41.83, zoom: 9.9},
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Jobs reachable without a car'},
    credit: 'LEHD · CTA · Census · OpenStreetMap contributors (ODbL)'
  },
  options,

  readouts: [
    {
      id: 'network',
      label: 'Network',
      help: 'Walkable intersections plus two nodes per CTA stop (a walk node and a vehicle node), and the edges that connect them.'
    },
    {
      id: 'matrix',
      label: 'Cost matrix',
      help: 'Rows x nodes and its size in memory, plus the lane batching.'
    },
    {
      id: 'opportunityShare',
      label: 'Jobs in the rows',
      kind: 'chart',
      help: 'Share of all Chicago jobs (LEHD 2021) that sit in the opportunity tracts the matrix searches from.'
    },
    {id: 'topOpportunity', label: 'Top loaded opportunity'},
    {id: 'selectedRow', label: 'Selected matrix row'},
    {id: 'decayChart', label: 'Decay response', kind: 'chart'},
    {id: 'transitComparison', label: 'Walk / scheduled CTA comparison', kind: 'chart'},
    {id: 'competitionComparison', label: 'Cumulative / 2SFCA comparison', kind: 'chart'},
    {
      id: 'snapped',
      label: 'Snapped opportunities',
      help: 'Opportunities that found a walkable street within the maximum snap distance.'
    },
    {
      id: 'snapDistance',
      label: 'Snap distance (mean / max)',
      help: 'Planar distance from each opportunity to its snapped position.'
    },
    {
      id: 'converged',
      label: 'Matrix converged',
      help: 'GPUNetworkCostMatrix reports 1 when every batch reached a fixpoint within the round limit.'
    },
    {
      id: 'matrixTime',
      label: 'Matrix time (snap + searches)',
      help: 'Median GPU time to re-run snapping and every search. Measured outside the frame after changes settle.'
    },
    {
      id: 'scoreTime',
      label: 'Re-score time',
      help: 'Median GPU time of GPUNetworkAccessibility over the retained matrix: a few linear passes, no search.'
    },
    {
      id: 'encodes',
      label: 'Encodes (matrix / score)',
      help: 'How often each graph has been encoded since the last rebuild: sliders move the score count, not the matrix count.'
    },
    {
      id: 'reached',
      label: 'Intersections with access',
      help: 'Share of walkable intersections that reach at least one opportunity within the threshold.'
    },
    {
      id: 'median',
      label: 'Median score',
      help: 'Median of the displayed measure over road intersections.'
    },
    {
      id: 'mean',
      label: 'Mean score',
      help: 'Mean of the displayed measure over road intersections.'
    },
    {
      id: 'top',
      label: 'Top of the scale',
      help: 'Maximum observed score, reported against the fixed class table.'
    }
  ],

  legends: state => {
    const table =
      state.measure === 'cumulative'
        ? CUMULATIVE_CLASSES
        : state.measure === 'two-step'
          ? TWO_STEP_CLASSES
          : GRAVITY_CLASSES;
    const legends: LegendSpec[] = [
      getClassTableLegend(table, {
        id: 'score',
        title:
          state.measure === 'two-step'
            ? '2SFCA jobs per 1,000 competing workers'
            : state.measure === 'cumulative'
              ? `Jobs within ${state.thresholdMinutes} minutes`
              : 'Decay-weighted jobs',
        note: table.method
      })
    ];
    legends.push({
      kind: 'categories',
      title: 'CTA rail lines',
      entries: [
        {color: [198, 12, 48, 255], label: 'Red'},
        {color: [0, 161, 222, 255], label: 'Blue'},
        {color: [98, 54, 27, 255], label: 'Brown'},
        {color: [0, 155, 58, 255], label: 'Green'},
        {color: [249, 70, 28, 255], label: 'Orange'},
        {color: [82, 35, 152, 255], label: 'Purple'},
        {color: [226, 126, 166, 255], label: 'Pink'},
        {color: [249, 227, 0, 255], label: 'Yellow'}
      ]
    });
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkSnapping,
  GPUNetworkCostMatrix,
  GPUNetworkAccessibility,
  encodeGPUNetworkAccessibilityParameters
} from '@luma.gl/experimental/gpu-network';

// Graph 1: encoded only when the network or the opportunities change.
const matrixGraph = new GPUCommandGraph(device, {id: 'access-matrix'});
matrixGraph.add(new GPUNetworkSnapping({
  points: opportunities,                       // float32x2, one job-rich tract each
  nodePositions: roadNodes, edgeSources, edgeTargets, edgeCosts: walkSeconds,
  maxSnapDistance: maxSnap.importToGraph(matrixGraph),   // ${state.maxSnapDistance} m, per frame${
    state.snapSearch === 'bvh'
      ? '\n  candidateCapacity: 1 << 20, spatialSort: true,           // BVH instead of an exact scan'
      : ''
  }
  seedDirection: '${state.seedDirection}',
  snappedEdges, snappedPositions, seedNodes, seedCosts     // two seeds per point
}));
matrixGraph.add(new GPUNetworkCostMatrix({
  offsets, neighbors, weights,                 // REVERSED walk + transit CSR: cost from each node TO the opportunity
  seedNodes, seedCosts, seedsPerRow: 2,
  costLimit: limit.importToGraph(matrixGraph), // ${state.matrixLimitMinutes * 60} s
  laneCount: ${state.laneCount}, maxIterations: 48, localIterations: 16,
  costs: matrix, converged                      // [${state.opportunityRows} rows x nodes], row-major
}));

// Graph 2: encoded whenever a scoring parameter changes. No search is re-run.
const scoreGraph = new GPUCommandGraph(device, {id: 'access-score'});
scoreGraph.add(new GPUNetworkAccessibility({
  costs: matrix, opportunityWeights: jobs, parameters: scoring.importToGraph(scoreGraph),
  cumulative, gravity,
  catchment: {demand: workers, output: twoStep, ratios}   // 2SFCA
}));
scoring.write(encodeGPUNetworkAccessibilityParameters(${
    state.measure === 'gravity-exponential'
      ? `{threshold: ${state.thresholdMinutes * 60}, decay: 'exponential', beta: ${(state.beta / 600).toFixed(5)}}`
      : state.measure === 'gravity-power'
        ? `{threshold: ${state.thresholdMinutes * 60}, decay: 'power', beta: ${state.powerExponent}, minimumCost: ${state.minimumCostSeconds}}`
        : `{threshold: ${state.thresholdMinutes * 60}, decay: 'none'}`
  }));`,

  about: {
    what: '`GPUNetworkSnapping` puts each opportunity (here a census tract with its jobs) onto the nearest walkable street edge and returns two seeds with their costs. `GPUNetworkCostMatrix` searches from every opportunity over the reversed walk-plus-transit network and keeps a matrix of travel times to every node. `GPUNetworkAccessibility` then scores each node from that matrix.',
    why: 'Accessibility is the planning question behind transit equity: not "how far is the nearest bus stop" but "how many jobs can a person from this block reach in 45 minutes". Splitting the expensive search from the cheap scoring lets you explore thresholds and decay functions interactively.',
    howToRead:
      'Brighter streets occupy higher fixed authored classes. Zero is neutral grey; cumulative and 2SFCA use different labelled units. The scheduled CTA comparison uses median hops and expected half-headway waiting, not observed reliability.'
  },

  create: async ctx => (await import('./job-accessibility.compute')).createJobAccessibility(ctx),

  story: [
    {
      id: 'opportunities',
      title: 'Jobs are not evenly distributed',
      headline: 'Opportunity comes before travel.',
      textAlternative:
        'Tract outlines and hollow circles sized by loaded workplace jobs; the adjacent chart reports their live shares.',
      optionsMode: 'fresh',
      controls: ['showOpportunities'],
      readouts: ['opportunityShare', 'topOpportunity'],
      options: {showOpportunities: true, showTransit: false},
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.2},
      body: 'Hollow circles are area-scaled workplace jobs in the retained, loaded tracts. The share chart and top-tract annotation are calculated from those rows, rather than asserting a city total or a downtown share.'
    },
    {
      id: 'walk',
      title: 'Walking draws a small labour market',
      headline: 'The threshold has fixed job classes.',
      textAlternative:
        'Classed cumulative walk-only accessibility with visible zero streets and a metric scale bar.',
      optionsMode: 'fresh',
      controls: ['thresholdMinutes', 'walkSpeed'],
      readouts: ['reached', 'median'],
      options: {transit: false, measure: 'cumulative', thresholdMinutes: 30},
      body: 'Walking-only scores use the published zero, 1–5k, 5–25k, 25–100k, 100–250k and >250k job classes. Zero stays a neutral street, not missing data.'
    },
    {
      id: 'transit',
      title: 'Transit opens narrow corridors',
      headline: 'Scheduled CTA changes the same classes.',
      textAlternative:
        'The same cumulative classes compare walking with the scheduled CTA network while rail remains neutral and cased.',
      optionsMode: 'fresh',
      controls: ['transit', 'waitFactor'],
      readouts: ['transitComparison', 'median', 'encodes'],
      options: {transit: true, measure: 'cumulative', thresholdMinutes: 45, showTransit: true},
      body: 'This uses scheduled median hops and expected half-headway waiting: a scheduled approximation, not a reliability percentile. The fixed classes make the walk and CTA comparison legible.'
    },
    {
      id: 'matrix-row',
      title: 'One row is one destination',
      headline: 'A retained opportunity has street bands.',
      textAlternative:
        'Snap leaders link retained opportunity centroids to streets and a selected matrix row is shown in ten-minute travel bands.',
      optionsMode: 'fresh',
      controls: ['showSnaps', 'maxSnapDistance', 'opportunityRows'],
      readouts: ['matrix', 'selectedRow', 'snapDistance'],
      options: {showSnaps: true, showOpportunities: true, transit: true},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 12.2},
      body: 'Each retained opportunity is snapped before the reverse cost-matrix search. The selected row is copied from the retained matrix for a local ten-minute band display; it does not trigger a CPU matrix readback.'
    },
    {
      id: 'decay',
      title: 'A threshold is a cliff',
      headline: 'Decay re-scores without another search.',
      textAlternative:
        'A chart compares the cumulative threshold cliff with exponential and power decay responses linked to the threshold and beta controls.',
      optionsMode: 'fresh',
      controls: ['thresholdMinutes', 'beta', 'measure'],
      readouts: ['scoreTime', 'matrixTime', 'encodes'],
      options: {transit: true, measure: 'gravity-exponential', thresholdMinutes: 45, beta: 1},
      body: 'The cumulative, exponential and power responses are charted from the current scoring controls. Only the score graph changes when threshold or decay changes; the matrix encode count remains fixed.'
    },
    {
      id: 'competition',
      title: 'Jobs per competing worker',
      headline: 'Competition changes accessibility counts.',
      textAlternative:
        'Fixed cumulative and 2SFCA comparisons label the latter as jobs per 1,000 competing workers.',
      optionsMode: 'fresh',
      controls: ['measure', 'thresholdMinutes'],
      readouts: ['competitionComparison', 'median', 'mean', 'top'],
      options: {transit: true, measure: 'two-step', thresholdMinutes: 45},
      body: '2SFCA divides jobs by workers able to reach each opportunity and is shown as jobs per 1,000 competing workers. It shares the scheduled, centroid and retained-row limitations of this demonstration.'
    }
  ]
});
