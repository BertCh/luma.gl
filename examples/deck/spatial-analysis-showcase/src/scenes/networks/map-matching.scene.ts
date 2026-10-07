// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MapMatchingOptions} from './map-matching.compute';

const RAW_COLOR = [135, 83, 180, 255] as const;

/** HMM map matching of simulated GPS traces on Chicago. GPU work is in `map-matching.compute.ts`. */
export default defineScene<MapMatchingOptions>({
  id: 'map-matching',
  title: 'Which streets did these GPS traces follow?',
  chapter: 'networks',
  order: 5,
  summary:
    'An honest nearest-edge baseline and a route-aware HMM explain the same noisy synthetic Chicago fixes.',
  contributors: ['GPUMapMatching'],
  datasets: [
    {id: 'chicago-roads', role: 'street graph with polyline geometry'},
    {id: 'chicago-gps-traces', role: 'simulated noisy GPS traces with ground truth'}
  ],
  initialView: {longitude: -87.634, latitude: 41.882, zoom: 16.25},
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Simulated GPS · live noise and sample interval'},
    credit: 'Synthetic traces derived from OSM, ODbL; not observed vehicles.',
    scaleBar: {units: 'metric'}
  },

  options: [
    {
      kind: 'slider',
      id: 'sigma',
      label: 'GPS error (emission sigma)',
      group: 'Model',
      apply: 'param',
      min: 2,
      max: 80,
      step: 1,
      default: 20,
      unit: 'm',
      help: 'How far a fix is expected to be from the true road. Small values trust each fix and snap to the nearest street; large values let the route, not the fix, decide.'
    },
    {
      kind: 'slider',
      id: 'beta',
      label: 'Route tolerance (transition beta)',
      group: 'Model',
      apply: 'param',
      min: 5,
      max: 300,
      step: 5,
      default: 60,
      unit: 'm',
      help: 'Scale of exp(-|straight distance - route distance| / beta). Small beta insists that the drive between two fixes is as long as the straight line; large beta forgives detours.'
    },
    {
      kind: 'slider',
      id: 'searchRadius',
      label: 'Candidate search radius',
      group: 'Model',
      apply: 'param',
      min: 10,
      max: 200,
      step: 5,
      default: 60,
      unit: 'm',
      help: 'Streets farther than this from a fix are not candidates. It is capped at twice the edge-grid cell size (changing the cell size changes the cap).'
    },
    {
      kind: 'slider',
      id: 'routeFactor',
      label: 'Route length factor',
      group: 'Model',
      apply: 'param',
      min: 1,
      max: 6,
      step: 0.5,
      default: 3,
      help: 'A route between two candidates may be at most factor x straight distance + slack long. Longer routes are treated as unreachable (a break).'
    },
    {
      kind: 'slider',
      id: 'routeSlack',
      label: 'Route slack',
      group: 'Model',
      apply: 'param',
      min: 0,
      max: 400,
      step: 10,
      default: 100,
      unit: 'm',
      help: 'The additive part of the route allowance. Needed because two fixes 5 m apart can sit on different sides of a block.'
    },
    {
      kind: 'slider',
      id: 'extraNoise',
      label: 'Extra GPS noise',
      group: 'Traces',
      apply: 'param',
      min: 0,
      max: 60,
      step: 1,
      default: 0,
      unit: 'm',
      help: 'Adds Gaussian noise (sigma per axis) on top of the simulated 10 to 25 m noise already in the traces. Watch accuracy fall as downtown blocks (about 100 m) become ambiguous.'
    },
    {
      kind: 'slider',
      id: 'noiseSeed',
      label: 'Noise seed',
      group: 'Traces',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      help: 'Re-rolls the extra noise.'
    },
    {
      kind: 'slider',
      id: 'trackFocus',
      label: 'Selected trace',
      group: 'Traces',
      apply: 'param',
      min: 1,
      max: 200,
      step: 1,
      default: 12,
      help: 'Trace used for street-scale raw-fix, candidate, and snap evidence.'
    },
    {
      kind: 'slider',
      id: 'fixFocus',
      label: 'Selected fix',
      group: 'Traces',
      apply: 'param',
      min: 1,
      max: 240,
      step: 1,
      default: 3,
      help: 'Focused fix for the candidate ring, snap leaders, and transition context.'
    },
    {
      kind: 'select',
      id: 'evidenceMode',
      label: 'Evidence',
      group: 'Display',
      apply: 'param',
      default: 'raw',
      help: 'Story state controlling whether raw, nearest, candidate, or limit evidence is foregrounded.',
      options: [
        {value: 'raw', label: 'Raw fixes'},
        {value: 'nearest', label: 'Nearest baseline'},
        {value: 'candidates', label: 'Candidates'},
        {value: 'tradeoff', label: 'Trade-off'},
        {value: 'stress', label: 'Stress'}
      ]
    },
    {
      kind: 'select',
      id: 'candidateCount',
      label: 'Candidates per fix',
      group: 'Compile-time',
      apply: 'compile',
      default: '8',
      help: 'Distinct street pieces kept per fix, 1 to 8. A two-way street takes two slots (one per direction), so 2 candidates is one street; near a junction 8 is needed to keep every approach. Rebuilds the graph.',
      options: [
        {value: '2', label: '2 (one street)'},
        {value: '4', label: '4'},
        {value: '6', label: '6'},
        {value: '8', label: '8'}
      ]
    },
    {
      kind: 'select',
      id: 'routeNodeBudget',
      label: 'Route search node budget',
      group: 'Compile-time',
      apply: 'compile',
      default: '64',
      help: 'Nodes each bounded route search may settle. A small budget is faster but can overestimate or miss a route in dense downtown streets. Rebuilds the graph.',
      options: [
        {value: '16', label: '16 nodes'},
        {value: '32', label: '32 nodes'},
        {value: '64', label: '64 nodes'},
        {value: '128', label: '128 nodes'}
      ]
    },
    {
      kind: 'select',
      id: 'cellSize',
      label: 'Edge grid cell size',
      group: 'Compile-time',
      apply: 'compile',
      default: '60',
      help: 'Cell size of the uniform edge grid used to find candidates; the search radius is capped at twice this. Rebuilds the graph.',
      options: [
        {value: '40', label: '40 m (radius up to 80 m)'},
        {value: '60', label: '60 m (radius up to 120 m)'},
        {value: '80', label: '80 m (radius up to 160 m)'},
        {value: '100', label: '100 m (radius up to 200 m)'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the matcher',
      group: 'Timing',
      help: 'Runs the whole compiled graph (noise, matching, drawing buffers) outside the frame and reports the median time.'
    }
  ],

  legends: () => [
    {
      kind: 'line',
      title: 'Inference outcome',
      entries: [
        {color: [40, 125, 210, 255], widthPixels: 3.5, label: 'HMM truth edge'},
        {
          color: [122, 185, 232, 255],
          widthPixels: 3.5,
          dashed: true,
          label: 'HMM reverse direction'
        },
        {color: [205, 76, 47, 255], widthPixels: 4.5, label: 'HMM wrong street'},
        {color: [105, 92, 75, 255], widthPixels: 2, dashed: true, label: 'Nearest-edge baseline'}
      ]
    },
    {
      kind: 'categories',
      title: 'Raw and candidate evidence',
      entries: [
        {color: RAW_COLOR, label: 'Raw fix and halo'},
        {color: [212, 163, 45, 255], label: 'Candidate segments and search ring'}
      ]
    }
  ],

  readouts: [
    {id: 'noise', label: 'Simulated GPS RMS'},
    {id: 'interval', label: 'Sample interval'},
    {id: 'baseline', label: 'Nearest baseline accuracy'},
    {id: 'hmm', label: 'HMM accuracy'},
    {id: 'mismatch', label: 'Nearest/HMM disagreements'},
    {id: 'breaks', label: 'HMM breaks'},
    {id: 'snapHistogram', label: 'Snap distances', kind: 'chart'},
    {id: 'noiseComparison', label: 'Accuracy under added noise', kind: 'chart'},
    {id: 'overflow', label: 'Grid overflow', hood: true},
    {id: 'time', label: 'Graph time', hood: true}
  ],

  snippet: state => `import {
  GPUMapMatching,
  encodeGPUMapMatchingParameters
} from '@luma.gl/experimental/gpu-network';

// The CSR row is the edge id: list both directions of every two-way street.
graph.add(new GPUMapMatching({
  points,                       // float32x2 planar meters, tracks back to back
  trackOffsets,                 // trackCount + 1
  nodePositions, offsets, edgeTargets,
  parameters,                   // per-frame, see below
  candidateCount: ${state.candidateCount},        // compile-time
  routeNodeBudget: ${state.routeNodeBudget},      // compile-time
  cellSize: ${state.cellSize},                // compile-time edge grid
  bounds: {minimum: [minX, minY], maximum: [maxX, maxY]},
  output: {matchedEdges, snappedPositions, snapDistances, breaks}
}));

// Per frame: parameter writes only.
parameters.write(encodeGPUMapMatchingParameters({
  sigma: ${state.sigma}, beta: ${state.beta}, searchRadius: ${state.searchRadius},
  routeFactor: ${state.routeFactor}, routeSlack: ${state.routeSlack}
}));

`,

  about: {
    what: '`GPUMapMatching` is a hidden Markov model map matcher (Newson and Krumm 2009). Each GPS fix has candidate edges within the search radius; Gaussian distance emissions and bounded route-continuity transitions select a Viterbi sequence. The nearest-edge baseline is a separate scene-local geometric calculation.',
    why: 'Synthetic OSM-derived GPS fixes have known truth edges, so the display can measure whether continuity changes a nearest snap. These traces are not observed vehicles.',
    howToRead:
      'Purple dots are raw fixes and the achromatic casing is truth. Blue is a correct HMM edge, pale dashed blue is reverse direction, and vermillion is a different street. Gold candidates and rings are scene-local display evidence, not GPU output.'
  },

  create: async ctx => (await import('./map-matching.compute')).createMapMatching(ctx),

  story: [
    {
      id: 'raw-fixes',
      headline: 'The fixes miss the street.',
      textAlternative:
        'Purple raw-fix halos, a dashed connector, live sigma ring, warm-grey roads and thick truth casing appear at street scale.',
      controls: ['trackFocus', 'sigma'],
      readouts: ['noise', 'interval'],
      optionsMode: 'fresh',
      title: 'Raw evidence',
      body: 'Purple raw fixes are linked in time. Their live sigma ring and the thick achromatic truth casing make the uncertainty visible at street scale. These are synthetic OSM-derived traces, not observed vehicles.',
      options: {trackFocus: 12, fixFocus: 3, sigma: 20, extraNoise: 0, evidenceMode: 'raw'},
      camera: {longitude: -87.634, latitude: 41.882, zoom: 16.25, transitionMs: 900}
    },
    {
      id: 'scale',
      headline: 'Twenty metres vanish when zoomed out.',
      textAlternative:
        'The same selected trace and live sigma evidence are shown at city scale without changing the evidence.',
      controls: ['trackFocus'],
      readouts: ['noise'],
      optionsMode: 'fresh',
      title: 'Scale changes meaning',
      body: 'This is the same trace and sigma ring, now near zoom 12. The uncertainty remains in metres but becomes small relative to the city street graph.',
      options: {trackFocus: 12, fixFocus: 3, sigma: 20, extraNoise: 0, evidenceMode: 'raw'},
      camera: {longitude: -87.634, latitude: 41.882, zoom: 12, transitionMs: 900}
    },
    {
      id: 'nearest',
      headline: 'Nearest is not a route.',
      textAlternative:
        'Dashed nearest-edge snaps, HMM outcomes and truth casing compare an honest baseline with the inferred route.',
      controls: ['trackFocus', 'extraNoise'],
      readouts: ['baseline', 'hmm', 'mismatch'],
      optionsMode: 'fresh',
      title: 'Nearest baseline',
      body: 'Dashed warm-grey leaders are direct nearest-edge snaps, measured independently from the HMM. The live disagreement and accuracy readouts identify actual differences without naming an unsupported failure location.',
      options: {trackFocus: 12, fixFocus: 3, sigma: 20, extraNoise: 0, evidenceMode: 'nearest'},
      camera: {longitude: -87.634, latitude: 41.882, zoom: 16.1, transitionMs: 900}
    },
    {
      id: 'hmm',
      headline: 'Distance and continuity vote together.',
      textAlternative:
        'A focused raw fix has a gold search ring, candidate street segments, snap leaders and adjacent route context.',
      controls: ['fixFocus', 'searchRadius', 'sigma'],
      readouts: ['snapHistogram'],
      optionsMode: 'fresh',
      title: 'Candidate evidence',
      body: 'Gold geometry is a bounded scene-local display query over the road index, not GPU output. Candidate segments, search radius, snap leaders and neighboring fixes show why continuity can beat a nearest edge.',
      options: {
        trackFocus: 12,
        fixFocus: 3,
        searchRadius: 60,
        sigma: 20,
        evidenceMode: 'candidates'
      },
      camera: {longitude: -87.634, latitude: 41.882, zoom: 16.35, transitionMs: 900}
    },
    {
      id: 'tradeoff',
      headline: 'Trust the fix or the path.',
      textAlternative:
        'HMM outcomes and visible break crosses respond to sigma, continuity tolerance and the bounded route search.',
      controls: ['sigma', 'beta', 'routeNodeBudget'],
      readouts: ['hmm', 'breaks'],
      optionsMode: 'fresh',
      title: 'Model trade-off',
      body: 'Sigma controls how much distance matters; beta controls route continuity. Blue is the truth edge, pale dashed blue the reverse direction, vermillion the wrong street, and crosses show bounded-search restarts.',
      options: {
        trackFocus: 12,
        fixFocus: 3,
        sigma: 20,
        beta: 60,
        routeNodeBudget: '64',
        evidenceMode: 'tradeoff'
      },
      camera: {longitude: -87.634, latitude: 41.882, zoom: 16.1, transitionMs: 900}
    },
    {
      id: 'stress',
      headline: 'Noise finds the model’s limits.',
      textAlternative:
        'A deterministic chart compares nearest-edge and HMM accuracy across added synthetic noise levels.',
      controls: ['extraNoise', 'noiseSeed', 'candidateCount'],
      readouts: ['noiseComparison', 'baseline', 'hmm', 'breaks'],
      optionsMode: 'fresh',
      title: 'Limits',
      body: 'The deterministic sweep uses loaded synthetic traces. Limits: planar distances; no speed, heading or turn penalty; isotropic synthetic noise; and bounded candidate and route search.',
      options: {
        trackFocus: 12,
        fixFocus: 3,
        extraNoise: 40,
        noiseSeed: 1,
        candidateCount: '8',
        evidenceMode: 'stress'
      },
      camera: {longitude: -87.634, latitude: 41.882, zoom: 13, transitionMs: 900}
    }
  ]
});
