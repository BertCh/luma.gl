// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
      {value: '128', label: '128 tracts (81% of jobs)'},
      {value: '256', label: '256 tracts (90% of jobs)'},
      {value: '384', label: '384 tracts (95% of jobs)'}
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
  },
  {
    kind: 'select',
    id: 'ramp',
    label: 'Colour ramp',
    group: 'Layers',
    apply: 'param',
    default: 'inferno',
    options: [
      {value: 'inferno', label: 'Inferno'},
      {value: 'magma', label: 'Magma'},
      {value: 'viridis', label: 'Viridis'},
      {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
    ],
    help: 'The accessibility score is drawn on a square-root scale clipped at the 98th percentile.'
  }
];

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
      help: 'Share of all Chicago jobs (LEHD 2021) that sit in the opportunity tracts the matrix searches from.'
    },
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
      help: '98th percentile (the end of the colour ramp) and the maximum.'
    }
  ],

  legends: state => {
    const unit =
      state.measure === 'cumulative'
        ? `jobs within ${state.thresholdMinutes} min`
        : state.measure === 'two-step'
          ? 'jobs per competing worker'
          : 'decay-weighted jobs';
    const legends: LegendSpec[] = [
      {
        kind: 'ramp',
        id: 'score',
        title: 'Accessibility score',
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: true,
        unit,
        format: value =>
          value === 0
            ? '0'
            : value >= 1000
              ? `${(value / 1000).toFixed(0)}k`
              : value < 1
                ? value.toExponential(1)
                : value.toFixed(0)
      }
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
      'Brighter streets reach more jobs. The scale is a square root, clipped at the 98th percentile, so the Loop does not wash out the rest of the map. Compare the map with and without transit, and look along the L lines: access follows the rail corridors.'
  },

  create: async ctx => (await import('./job-accessibility.compute')).createJobAccessibility(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['transit', 'thresholdMinutes'],
      readouts: ['median', 'reached'],
      title: 'How many jobs can you reach in 45 minutes without a car?',
      body: 'Chicago has about 1.3 million jobs, and a quarter of them sit in a single downtown tract. For a resident without a car the reachable share depends on where they live and on the CTA. This map colours every walkable intersection by the **number of jobs reachable within 45 minutes** of walking and riding buses and trains, using 2021 LEHD workplace counts for the 256 job-richest tracts (90% of all jobs).\n\nThe bright spines follow the L lines out of the Loop. The dark patches in between are neighbourhoods where the nearest train is a long walk away. Switch **Ride the CTA** and move the **Travel-time threshold** below to see who gains.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 9.9, transitionMs: 1200},
      options: {transit: true, measure: 'cumulative', thresholdMinutes: 45, showTransit: true},
      highlight: {readout: 'median'}
    },
    {
      id: 'snapping',
      controls: ['showSnaps', 'maxSnapDistance', 'seedDirection'],
      readouts: ['snapDistance', 'snapped'],
      title: 'Opportunities must be on the street first',
      body: 'A tract centroid is somewhere inside a block, not on a street. `GPUNetworkSnapping` finds the nearest walkable edge for every opportunity, the fraction along it, and the two **seed costs** (walking time to each end of the edge). Those become a two-seed start for the search, so an opportunity mid-block is reached from both ends.\n\nThe **Snap lines** below are on in the Loop: each orange segment is the walk from the tract centroid to the street. Try **Maximum snap distance** and **Seed direction** below: a one-ended seed behaves like a one-way street.',
      camera: {longitude: -87.64, latitude: 41.88, zoom: 12.4, transitionMs: 1400},
      options: {showSnaps: true, showOpportunities: true, showStops: true},
      highlight: {readout: 'snapDistance'}
    },
    {
      id: 'cost-matrix',
      controls: ['transit', 'opportunityRows', 'laneCount'],
      readouts: ['matrix', 'matrixTime'],
      title: 'One search per opportunity: the cost matrix',
      body: '`GPUNetworkCostMatrix` runs a bounded shortest-path search from each opportunity over the **reversed** network (so the cost it stores is from every node *to* the opportunity) and keeps every result: 256 rows by about 51,000 nodes, 52 MiB. The searches are batched into lanes of 32 so the GPU stays busy, and the numbers are bit-identical for every lane count.\n\nThe network is walking on every street but the expressways, plus the CTA: board a stop (paying half the headway as wait), ride the GTFS hop graph, alight. This step opens with **Ride the CTA** off, so the matrix is for walking alone; switch it on and the matrix is re-computed with the transit hops. **Matrix time (snap + searches)** shows what that costs, and **Opportunity tracts (matrix rows)** and **Searches per batch (lanes)** change the matrix itself.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 9.9, transitionMs: 1400},
      options: {showSnaps: false, showStops: false, transit: false},
      highlight: {readout: 'matrixTime'}
    },
    {
      id: 'rescoring',
      controls: ['transit', 'thresholdMinutes'],
      readouts: ['scoreTime', 'matrixTime', 'encodes'],
      title: 'Re-scoring is nearly free',
      body: 'Walking alone reaches almost nothing outside downtown. Turn **Ride the CTA** back on, then drag the **Travel-time threshold** between 15 and 60 minutes: `GPUNetworkAccessibility` re-reads the retained matrix and sums the jobs within the threshold for every node, deterministically and without atomics. Compare **Re-score time** with **Matrix time (snap + searches)** and watch **Encodes (matrix / score)**: the score count moves, the matrix count does not.\n\nThe cumulative measure is a cliff: a job 44 minutes away counts fully, one 46 minutes away not at all.',
      options: {transit: true, thresholdMinutes: 30},
      highlight: {readout: 'scoreTime'}
    },
    {
      id: 'gravity',
      controls: ['measure', 'beta', 'thresholdMinutes'],
      readouts: ['top'],
      title: 'Soft thresholds: gravity decay',
      body: 'A gravity measure replaces the cliff with a **decay function**: opportunities count for less the farther they are. `exp(-beta * minutes / 10)` is the exponential form; the power form `(c / c0)^-beta` keeps a heavier tail. The decay kind, `beta` and the power floor `minimumCost` are four floats in a parameter buffer.\n\nThe step opens on the exponential form: change **Exponential decay (beta)** below, or switch **Accessibility measure** to the power form and tune **Power decay exponent** and **Power decay floor** in the All controls tab. The map smooths and the ranking of neighbourhoods shifts, yet no search ran. A steep beta approximates "only walking distance matters"; a shallow one rewards the reach of the L.',
      options: {measure: 'gravity-exponential', thresholdMinutes: 60, beta: 1.2},
      highlight: {readout: 'top'}
    },
    {
      id: 'two-step',
      controls: ['measure', 'thresholdMinutes'],
      readouts: ['top', 'median'],
      title: 'Jobs per competing worker: 2SFCA',
      body: 'More jobs within reach does not help if hundreds of thousands of other workers can reach the same ones. The **two-step floating catchment area** (2SFCA) divides each opportunity’s jobs by the workers who can reach it, then sums those ratios for every origin. Here demand is the workers who live in each tract (LEHD resident workers) placed on the nearest street.\n\nRead the map as *jobs per competing worker*. The downtown spine falls from first to merely good, and outlying areas with few competitors rise.',
      options: {measure: 'two-step', thresholdMinutes: 45},
      highlight: {readout: 'top'}
    },
    {
      id: 'limits',
      controls: [
        'waitFactor',
        'matrixLimitMinutes',
        'walkSpeed',
        'opportunityRows',
        'seedDirection'
      ],
      title: 'Limits, and things to try',
      body: 'This is an average-weekday, schedule-free model: the wait is half the headway of all departures from a stop, transfers between routes at a stop are free, and in-vehicle times are the median scheduled hop. Jobs are workplace counts per tract, noise-infused by the Census for privacy, and each tract is a single point. Only the 256 largest job tracts are opportunities (90% of jobs), and the walk network excludes expressways but not unsafe crossings.\n\nTry, with the controls below: set **Waiting at stops** to 0 and then 2; lengthen the **Search horizon (matrix cost limit)** to 90 minutes; raise **Walking speed** to 1.6 m/s; compare 128 and 384 tracts in **Opportunity tracts (matrix rows)**; switch **Seed direction** to forward; and use the BVH **Snapping search** (in the All controls tab).',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {measure: 'cumulative', thresholdMinutes: 45}
    }
  ]
});
