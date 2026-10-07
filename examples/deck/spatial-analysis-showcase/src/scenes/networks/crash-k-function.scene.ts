// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {CrashKOptions} from './crash-k-function.compute';

const RAMP_CHOICES = [
  {value: 'inferno', label: 'Inferno'},
  {value: 'magma', label: 'Magma'},
  {value: 'viridis', label: 'Viridis'},
  {value: 'cividis', label: 'Cividis'}
] as const;

const EVENT_SET_LABELS: Record<CrashKOptions['events'], string> = {
  all: 'all crashes',
  injury: 'crashes with an injury',
  severe: 'incapacitating and fatal crashes',
  fatal: 'fatal crashes',
  rush: 'weekday rush-hour crashes (15:00 to 19:00)',
  night: 'night crashes (00:00 to 05:00)'
};

/** Network K function of 2023 Chicago crashes. GPU work is in `crash-k-function.compute.ts`. */
export default defineScene<CrashKOptions>({
  id: 'crash-k-function',
  title: 'Do Chicago crashes cluster along the streets?',
  chapter: 'networks',
  order: 4,
  summary:
    'Network-constrained Ripley K of 2023 traffic crashes, measured along the street graph and tested against random points on the same streets. The whole analysis, including 19 simulated patterns, runs on the GPU.',
  contributors: ['GPUNetworkKFunction'],
  datasets: [
    {id: 'chicago-roads', role: 'street graph (CSR, lengths in meters)'},
    {id: 'chicago-crashes', role: 'crash events (position, time, severity)'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10.0},

  options: [
    {
      kind: 'select',
      id: 'events',
      label: 'Which crashes',
      group: 'Events',
      apply: 'param',
      default: 'all',
      help: 'Chooses the crashes that become events. The sample is drawn from this set; the road colors count the same set.',
      options: [
        {value: 'all', label: 'All crashes (109,711)'},
        {value: 'injury', label: 'Crashes with an injury'},
        {value: 'severe', label: 'Incapacitating and fatal (2,018)'},
        {value: 'fatal', label: 'Fatal crashes (142)'},
        {value: 'rush', label: 'Weekday rush hour, 15:00 to 19:00'},
        {value: 'night', label: 'Night, 00:00 to 05:00'}
      ]
    },
    {
      kind: 'slider',
      id: 'eventCount',
      label: 'Events in the sample',
      group: 'Events',
      apply: 'param',
      min: 20,
      max: 256,
      step: 4,
      default: 192,
      help: 'A random sample of this many crashes is analysed. Cost grows with events times patterns, so a sample keeps the analysis interactive; K is an average and is stable for a few hundred events.'
    },
    {
      kind: 'slider',
      id: 'sampleSeed',
      label: 'Sample seed',
      group: 'Events',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 1,
      help: 'Draws a different random sample of the same crashes. A real clustering result survives a change of seed.'
    },
    {
      kind: 'slider',
      id: 'maxDistance',
      label: 'Maximum distance',
      group: 'K function',
      apply: 'param',
      min: 200,
      max: 2000,
      step: 100,
      default: 1200,
      unit: 'm',
      help: 'Largest network distance d of the K function. The bands run from 0 to this distance. Per-frame parameter: shortest-path searches are limited by it.'
    },
    {
      kind: 'slider',
      id: 'maxSnapDistance',
      label: 'Maximum snap distance',
      group: 'K function',
      apply: 'param',
      min: 20,
      max: 200,
      step: 10,
      default: 60,
      unit: 'm',
      help: 'Crashes farther than this from any street are dropped instead of being forced onto the network. The data builder snapped within 60 m.'
    },
    {
      kind: 'slider',
      id: 'simulations',
      label: 'Simulated patterns',
      group: 'Envelope',
      apply: 'param',
      min: 0,
      max: 19,
      step: 1,
      default: 19,
      help: 'Random patterns of the same size drawn on the same streets, in proportion to street length. The envelope is their minimum and maximum; 19 patterns give a pointwise 5 percent test. Compile-time maximum is 19.'
    },
    {
      kind: 'slider',
      id: 'envelopeSeed',
      label: 'Envelope seed',
      group: 'Envelope',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      help: 'Seed of the Philox stream that draws the random patterns. Each seed is a new, reproducible set of 19 patterns.'
    },
    {
      kind: 'select',
      id: 'bandCount',
      label: 'Distance bands',
      group: 'Compile-time',
      apply: 'compile',
      default: '24',
      help: 'Number of distance thresholds d_k. More bands give a smoother K curve at the same search cost. Rebuilds the graph.',
      options: [
        {value: '12', label: '12 bands'},
        {value: '24', label: '24 bands'},
        {value: '48', label: '48 bands'},
        {value: '96', label: '96 bands'}
      ]
    },
    {
      kind: 'select',
      id: 'rowsPerBlock',
      label: 'Searches per block',
      group: 'Compile-time',
      apply: 'compile',
      default: '128',
      help: 'Shortest-path searches that run together. Scratch memory grows with this number; fewer rows per block use less memory and need more blocks (more passes). Rebuilds the graph.',
      options: [
        {value: '32', label: '32 rows (small scratch, 160 blocks)'},
        {value: '64', label: '64 rows'},
        {value: '128', label: '128 rows (about 128 MB scratch)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'spatialSort',
      label: 'Morton-sort edges for snapping',
      group: 'Compile-time',
      apply: 'compile',
      default: false,
      help: 'Sorts the street edges along a Morton curve before the BVH build of the snapping search. Same result; can change snapping time. Rebuilds the graph.'
    },
    {
      kind: 'select',
      id: 'roadStyle',
      label: 'Street color',
      group: 'Display',
      apply: 'param',
      default: 'crashes',
      help: 'Shade each block by the number of crashes of the selected set, or draw the plain street graph.',
      options: [
        {value: 'crashes', label: 'Crashes per block'},
        {value: 'plain', label: 'Plain streets'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      disabledWhen: state => state.roadStyle !== 'crashes',
      help: 'Ramp for crashes per block (square-root scaled).',
      options: RAMP_CHOICES
    },
    {
      kind: 'toggle',
      id: 'showEvents',
      label: 'Show sampled events',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the crashes in the sample as dots.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the K function',
      group: 'Timing',
      help: 'Runs the compiled graph (observed pattern plus every simulation) outside the frame and reports the median time.'
    }
  ],

  legends: state => [
    ...(state.roadStyle === 'crashes'
      ? [
          {
            kind: 'ramp' as const,
            id: 'crashes',
            title: `Crashes per block (${EVENT_SET_LABELS[state.events]})`,
            ramp: state.ramp,
            extent: 'gpu' as const,
            sqrtScale: true,
            format: (value: number) => Math.round(value).toString()
          }
        ]
      : []),
    {
      kind: 'categories' as const,
      title: 'Events',
      entries: [
        {color: [0, 160, 220, 235] as const, label: 'Sampled crash (an event of the K function)'}
      ],
      note: 'Plain streets: both directions of every street, drawn once.'
    }
  ],

  readouts: [
    {id: 'network', label: 'Street graph'},
    {
      id: 'length',
      label: 'Network length L',
      help: 'Every street counted once: half the sum of the directed edge lengths.'
    },
    {id: 'eligible', label: 'Crashes in the selected set', format: 'integer'},
    {id: 'eventsUsed', label: 'Events requested', format: 'integer'},
    {id: 'events', label: 'Events snapped to the network', help: 'The n of the K function.'},
    {
      id: 'pairs',
      label: 'Event pairs within the maximum distance',
      format: 'integer',
      help: 'Unordered pairs with network distance strictly below d_max.'
    },
    {
      id: 'kAtMaximum',
      label: 'Observed K at the maximum distance',
      format: 'decimal',
      help: 'K(d) = 2 * pairs(d) * L / n^2, in meters.'
    },
    {id: 'envelopeRange', label: 'Simulated range at the maximum distance'},
    {
      id: 'ratio',
      label: 'Observed K over mean simulated K',
      help: 'Above 1 means more close pairs than random points on the same streets would produce.'
    },
    {
      id: 'profile',
      label: 'Observed K by distance',
      help: 'One bar per band from short to long distance, scaled to the largest value.'
    },
    {
      id: 'strip',
      label: 'Against the envelope',
      help: 'Per band: up triangle above the simulated maximum (clustered), down triangle below the minimum (dispersed), dot inside.'
    },
    {id: 'bands', label: 'Bands outside the envelope'},
    {id: 'verdict', label: 'Reading'},
    {id: 'converged', label: 'Searches converged'},
    {id: 'rows', label: 'Searches (events x patterns)', format: 'integer'},
    {id: 'blocks', label: 'Search blocks', format: 'integer'},
    {id: 'time', label: 'Graph time'}
  ],

  snippet: state => `import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues
} from '@luma.gl/experimental/gpu-network';

// Street graph: symmetric CSR (both directions of every street), weights = meters.
const k = new GPUNetworkKFunction({
  points: eventPositions,            // float32x2 planar meters, FAR_AWAY rows are ignored
  nodePositions, offsets, neighbors, weights,
  maxSnapDistance: snapDistance,     // one float: ${state.maxSnapDistance} m
  candidateCapacity: events * 512,   // snapping candidates
  maxDistance,                       // one float: ${state.maxDistance} m
  networkLength,                     // one float: half the sum of weights
  parameters,                        // [seedLow, seedHigh, activeSimulations, 0]
  bandCount: ${state.bandCount},                 // compile-time
  simulationCount: 19,               // compile-time envelope size
  rowsPerBlock: ${state.rowsPerBlock},              // compile-time scratch/pass trade
  spatialSort: ${state.spatialSort},
  kValues, envelope, pairCounts, snappedEventCount, overflow, converged
});
graph.add(k);
const compiled = graph.compile();

// Per frame (or when a control moves): parameter writes only, then encode.
parameters.write(getGPUNetworkKFunctionParameterValues({
  seed: ${state.envelopeSeed}, activeSimulations: ${state.simulations}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUNetworkKFunction` computes the network-constrained Ripley K function (Okabe and Yamada; `spaghetti` `GlobalAutoK`). Each event is snapped onto its street, a bounded shortest-path search measures the distance along the streets to every other event, and unordered pairs are counted per distance band: `K(d) = 2 * pairs(d) * L / n^2`, with `L` the network length and `n` the events. The same count is repeated for simulated patterns of `n` random points on the same streets to build an envelope.',
    why: 'Crashes only happen on streets, so a planar K function against a uniform background says "clustered" for any street-bound data. Measuring distance along the network and simulating on the network asks the right question: are crashes bunched up on particular blocks beyond what the street layout alone produces?',
    howToRead:
      'The readouts replace a K chart. **Against the envelope** shows one symbol per distance band: an up triangle where observed K exceeds every simulated pattern (clustering), a down triangle below all of them (dispersion), a dot inside the envelope (consistent with random). **Observed K over mean simulated K** is the size of the effect. Streets are shaded by crashes per block.'
  },

  create: async ctx => (await import('./crash-k-function.compute')).createCrashKFunction(ctx),

  story: [
    {
      id: 'question',
      controls: ['events', 'roadStyle', 'showEvents'],
      title: 'Do crashes bunch up beyond what the streets explain?',
      body: 'Chicago recorded 109,711 traffic crashes with a location in 2023, and the map shades every block by how many fell on it. The busiest blocks are plainly on the arterials: Western, Pulaski, Cicero, Ashland, Halsted, and Lake Shore Drive. But crashes can only happen on streets, and busy streets carry more of them for the boring reason that they are long and straight.\n\nThe analyst question is sharper: **if the same number of crashes were scattered at random over the same streets, would they sit this close together?** `GPUNetworkKFunction` answers it on the GPU. The cyan dots are the sample of events it analyses; **Which crashes**, **Street color** and **Show sampled events** below change what you see.',
      options: {events: 'all', roadStyle: 'crashes'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200}
    },
    {
      id: 'what-k-counts',
      controls: ['maxDistance'],
      readouts: ['pairs', 'kAtMaximum', 'profile'],
      title: 'What the network K function counts',
      body: 'Each event is snapped to its nearest street. For every event the GPU runs a shortest-path search along the streets, so the distance to another crash is the distance you would drive, not the distance a crow flies. For each distance `d` it counts **pairs of crashes closer than d**, and scales the count into `K(d) = 2 * pairs(d) * L / n^2`. `L` is the total street length, so K is comparable between cities of different size.\n\nThe readouts **Event pairs within the maximum distance** and **Observed K at the maximum distance** are those numbers at the maximum distance. **Observed K by distance** is the curve: it rises because more pairs fit inside a longer distance. On its own that curve says nothing; it needs a yardstick.',
      options: {maxDistance: 1200},
      highlight: {readout: 'kAtMaximum'}
    },
    {
      id: 'envelope',
      controls: ['simulations', 'envelopeSeed'],
      readouts: ['strip', 'ratio'],
      title: 'The yardstick: 19 random patterns on the same streets',
      body: 'The graph also draws **19 random patterns** of the same number of events, placed on streets with probability proportional to street length (a Philox stream keyed by the seed), and computes K for each. The **envelope** is their minimum and maximum per band. With 19 simulations an observed value above every simulated one is significant at about 5 percent for that distance.\n\nRead **Against the envelope**: up triangles are bands where the crashes are more clustered than every random pattern. Change **Envelope seed** and the triangles hold: the verdict does not depend on one lucky draw.',
      options: {simulations: 19, envelopeSeed: 1},
      highlight: {readout: 'strip'}
    },
    {
      id: 'severity',
      controls: ['events', 'eventCount'],
      readouts: ['ratio'],
      title: 'Serious crashes: a smaller, sharper pattern',
      body: 'Set **Which crashes** below to incapacitating and fatal crashes (2,018 of them). With fewer events the sample is almost the entire set, the dots thin out on the local streets and concentrate on the arterials and expressways, and the K curve compared with its envelope shows how strongly severe crashes concentrate at short distances.\n\nTry **Fatal crashes** (142): with so few events the envelope is wide, and the honest answer can be "cannot tell from random".',
      options: {events: 'severe', eventCount: 256}
    },
    {
      id: 'distance',
      controls: ['maxDistance'],
      readouts: ['profile', 'converged'],
      title: 'The scale of the clustering',
      body: 'Slide **Maximum distance** below. The bands always span 0 to d, so a short maximum zooms the K curve onto the single block and the next one, and a long one shows the neighborhood scale. Clustering that is clear at 300 m and absent at 2 km means crashes bunch on blocks and intersections; clustering that grows with distance means whole corridors.\n\nThe distance is a parameter write: the shortest-path searches are limited by it and the graph is not rebuilt, as **Under the hood** confirms. Check the readout **Searches converged**: if it says no, the search ran out of rounds before reaching d.',
      options: {events: 'all', maxDistance: 500, eventCount: 192},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 12.4, transitionMs: 1600}
    },
    {
      id: 'night',
      controls: ['events'],
      readouts: ['strip'],
      title: 'Same streets, different hours',
      body: 'Switch **Which crashes** below between weekday rush hour and night. The street graph and the null model are unchanged; only the events differ. Rush-hour crashes follow the commuting corridors; night crashes (00:00 to 05:00) are fewer and lean toward the arterials and expressways where traffic still moves fast.\n\nThe road shading counts the same set, so you can compare where the blocks are with what the K curve says about how tightly they bunch.',
      options: {events: 'night', maxDistance: 1200},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1600}
    },
    {
      id: 'limits',
      controls: ['sampleSeed', 'bandCount', 'rowsPerBlock', 'measure'],
      readouts: ['time'],
      title: 'Limits, parity and what to try',
      body: 'The implementation is pinned to `spaghetti` 1.7.6 (`GlobalAutoK` and `NetworkK`): pairs are counted with strict `<`, parallel roads count as one road, and unlike spaghetti the envelope is the plain minimum, mean and maximum of the simulations. The test is **pointwise**, not a global envelope test, so with 23 bands expect the occasional triangle by chance. Cross-K is not provided, and distances are along the streets of the dataset: crashes with no street within 60 m are not events.\n\nTry it: draw a new sample with **Sample seed**; raise **Distance bands** to 96 (a graph rebuild, flagged in the panel); lower **Searches per block** to 32 to trade scratch memory for passes; press **Time the K function** to see the cost of the whole analysis.',
      options: {events: 'all', bandCount: '24'}
    }
  ]
});
