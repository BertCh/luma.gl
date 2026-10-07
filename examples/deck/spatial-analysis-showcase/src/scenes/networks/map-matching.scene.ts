// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {ACCURACY_COLORS} from './b10-scene-constants';
import type {MapMatchingOptions} from './map-matching.compute';

const RAW_COLOR = [255, 140, 60, 255] as const;
const TRUTH_COLOR = [90, 96, 115, 255] as const;
const PLAIN_MATCH_COLOR = [40, 205, 255, 255] as const;

/** HMM map matching of simulated GPS traces on Chicago. GPU work is in `map-matching.compute.ts`. */
export default defineScene<MapMatchingOptions>({
  id: 'map-matching',
  title: 'Which streets did these GPS traces follow?',
  chapter: 'networks',
  order: 5,
  summary:
    'A hidden Markov model snaps 200 noisy simulated GPS traces onto the Chicago street graph, one Viterbi thread per trace, and scores every fix against the ground-truth street. GPULineMerge joins street segments into chains.',
  contributors: ['GPUMapMatching', 'GPULineMerge'],
  datasets: [
    {id: 'chicago-roads', role: 'street graph with polyline geometry'},
    {id: 'chicago-gps-traces', role: 'simulated noisy GPS traces with ground truth'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10.0},

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
      label: 'Show one trace (0 = all)',
      group: 'Traces',
      apply: 'param',
      min: 0,
      max: 200,
      step: 1,
      default: 0,
      help: 'Hides every trace except this one. Matching still runs on all 200 tracks; only the drawing is filtered.'
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
      kind: 'toggle',
      id: 'showRaw',
      label: 'Raw GPS fixes',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The noisy fixes (including any extra noise), joined in time order.'
    },
    {
      kind: 'toggle',
      id: 'showMatched',
      label: 'Matched route',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Snapped positions of consecutive matched fixes, hidden across breaks.'
    },
    {
      kind: 'select',
      id: 'matchedColor',
      label: 'Matched route color',
      group: 'Display',
      apply: 'param',
      default: 'accuracy',
      help: 'Color by whether the matched edge is the true one, or a single color.',
      options: [
        {value: 'accuracy', label: 'Against ground truth'},
        {value: 'plain', label: 'One color'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTruth',
      label: 'Ground-truth route',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'The noise-free positions on the true route, as a thick gray line under the others.'
    },
    {
      kind: 'select',
      id: 'roadStyle',
      label: 'Street layer',
      group: 'Display',
      apply: 'param',
      default: 'plain',
      help: 'Plain street segments, or maximal chains between junctions joined by GPULineMerge (one color per chain). Both are prepared up front; this only switches the drawn layer.',
      options: [
        {value: 'plain', label: 'Street segments'},
        {value: 'chains', label: 'Merged chains (GPULineMerge)'},
        {value: 'off', label: 'Hidden'}
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

  legends: state => [
    ...(state.showMatched
      ? [
          state.matchedColor === 'accuracy'
            ? {
                kind: 'categories' as const,
                title: 'Matched route against ground truth',
                entries: [
                  {color: ACCURACY_COLORS[1], label: 'Right street segment'},
                  {color: ACCURACY_COLORS[2], label: 'Right street, other direction'},
                  {color: ACCURACY_COLORS[0], label: 'Wrong street'}
                ],
                note: 'A fix on a two-way street can only be told from its mirror edge by the route the model chooses.'
              }
            : {
                kind: 'categories' as const,
                title: 'Matched route',
                entries: [{color: PLAIN_MATCH_COLOR, label: 'Snapped positions of matched fixes'}]
              }
        ]
      : []),
    {
      kind: 'categories' as const,
      title: 'Traces and streets',
      entries: [
        ...(state.showRaw ? [{color: RAW_COLOR, label: 'Raw GPS fixes'}] : []),
        ...(state.showTruth ? [{color: TRUTH_COLOR, label: 'Ground-truth route'}] : []),
        ...(state.roadStyle === 'chains'
          ? [
              {
                color: [31, 119, 180, 255] as const,
                label: 'Street chain (one color each, eight cycled)'
              }
            ]
          : state.roadStyle === 'plain'
            ? [{color: [96, 108, 135, 255] as const, label: 'Street segment'}]
            : [])
      ]
    }
  ],

  readouts: [
    {id: 'fixes', label: 'Traces'},
    {
      id: 'network',
      label: 'Matching graph',
      help: 'Every polyline vertex of the street graph is a node, so the matcher sees real street geometry.'
    },
    {
      id: 'noise',
      label: 'GPS error in the data',
      help: 'Root-mean-square distance between each fix and its true position.'
    },
    {id: 'matched', label: 'Fixes matched'},
    {
      id: 'breaks',
      label: 'Breaks (model restarts)',
      format: 'integer',
      help: 'Points where no transition between candidates was feasible.'
    },
    {
      id: 'exact',
      label: 'Right street segment',
      help: 'Share of all fixes whose matched edge equals the ground-truth edge of that fix.'
    },
    {
      id: 'sameStreet',
      label: 'Right street, either direction',
      help: 'Also counts the opposite-direction edge of the true street.'
    },
    {id: 'snap', label: 'Mean distance to matched street'},
    {id: 'overflow', label: 'Edge grid overflow'},
    {id: 'chains', label: 'Merged chains'},
    {id: 'time', label: 'Graph time'}
  ],

  snippet: state => `import {
  GPUMapMatching,
  encodeGPUMapMatchingParameters
} from '@luma.gl/experimental/gpu-network';
import {GPULineMerge} from '@luma.gl/experimental/gpu-spatial-analysis';

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
  output: {matchedEdges, snappedPositions, breaks, matchedCount, breakCount, overflow}
}));

// Per frame: parameter writes only.
parameters.write(encodeGPUMapMatchingParameters({
  sigma: ${state.sigma}, beta: ${state.beta}, searchRadius: ${state.searchRadius},
  routeFactor: ${state.routeFactor}, routeSlack: ${state.routeSlack}
}));

// Join street segments into chains between junctions.
mergeGraph.add(new GPULineMerge({
  positions, lineOffsets,
  output: {chainOffsets, positions: chainPositions, count: chainCount}
}));`,

  about: {
    what: '`GPUMapMatching` is a hidden Markov model map matcher (Newson and Krumm 2009). Each GPS fix has candidate edges within the search radius; the emission probability is Gaussian in the perpendicular distance, the transition probability exponential in the difference between the straight distance of two fixes and the route distance between their candidates. A Viterbi pass, one GPU thread per trace, picks the most likely edge sequence and restarts after a break. `GPULineMerge` joins line segments that meet at a shared endpoint into maximal chains.',
    why: 'GPS fixes are 10 to 25 m wrong, and downtown blocks are about 100 m. Snapping each fix to the nearest street gets many of them wrong; using the route between fixes fixes most of that. Matched edges give you speeds per street, trip routes and map-based analytics from raw traces.',
    howToRead:
      'Orange lines are the raw fixes. The matched route is green where the matched edge is the true one, amber where the model picked the opposite direction of the true street, and red where it chose another street. The readouts count the same classes over all 27,605 fixes.'
  },

  create: async ctx => (await import('./map-matching.compute')).createMapMatching(ctx),

  story: [
    {
      id: 'question',
      controls: ['showRaw', 'showMatched', 'matchedColor'],
      readouts: ['exact'],
      title: 'Which roads did these vehicles drive?',
      body: 'A fleet reports a position every 5 to 15 seconds, and each position is off by 10 to 25 meters. Plot them on a map and the cars appear to drive through buildings and along the wrong block. Operators need the **street** each vehicle was on, to compute speeds, travel times and routes.\n\nThese are 200 **simulated** traces (27,605 fixes) made by routing random trips on the Chicago street graph and adding noise, so every fix has a known true street and the matcher can be scored. Orange is the raw GPS; the colored line is the matched route. Use **Raw GPS fixes**, **Matched route** and **Matched route color** below to compare them.',
      options: {showRaw: true, showMatched: true, matchedColor: 'accuracy'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200}
    },
    {
      id: 'one-trace',
      controls: ['trackFocus', 'showTruth'],
      title: 'Follow one trace',
      body: 'Set **Show one trace (0 = all)** below to a single trace (it is on 12) and zoom in. The orange line wanders; the green line sits on streets. Turn on **Ground-truth route** to see the thick gray path the vehicle really took.\n\nThe matcher found the streets with **`GPUMapMatching`**: for every fix it collects candidate edges from a GPU edge grid, then Viterbi chooses the sequence of edges that is both close to the fixes and *drivable* in between.',
      options: {trackFocus: 12, showTruth: true},
      camera: {longitude: -87.66, latitude: 41.89, zoom: 12.4, transitionMs: 1600}
    },
    {
      id: 'sigma',
      controls: ['sigma'],
      readouts: ['exact', 'sameStreet'],
      title: 'How much to trust each fix: emission sigma',
      body: "The **emission** term says how likely a fix is given a candidate street: Gaussian in the perpendicular distance with standard deviation sigma. Slide **GPS error (emission sigma)** to 5 m and the model believes every fix, so it hops to whichever street is closest; at 20 m (the data's noise) it tolerates the error; at 80 m it barely cares which street is nearest and lets the route decide.\n\nThe accuracy readouts update as you move the slider. Matching is a per-frame parameter write, so nothing is recompiled.",
      options: {trackFocus: 0, showTruth: false, sigma: 5},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 12.6, transitionMs: 1400},
      highlight: {readout: 'exact'}
    },
    {
      id: 'beta',
      controls: ['beta', 'routeFactor', 'routeSlack'],
      readouts: ['breaks'],
      title: 'How much to trust the route: transition beta',
      body: 'The **transition** term compares how far apart two fixes are with how far you would drive between the candidate streets: `exp(-|straight - route| / beta)`. A small **Route tolerance** forbids detours, so a trace that jogs around a block becomes a break; a large one accepts any plausible route.\n\n**Breaks (model restarts)** are counted in the readout below. The model restarts after a break instead of failing the whole trace, and the matched line is hidden across it. Raise **Route length factor** and **Route slack** to allow longer routes between fixes after a dropout.',
      options: {sigma: 20, beta: 15},
      highlight: {readout: 'breaks'}
    },
    {
      id: 'noise',
      controls: ['extraNoise', 'noiseSeed', 'candidateCount', 'routeNodeBudget', 'measure'],
      readouts: ['exact'],
      title: 'Push it until it breaks',
      body: 'Add **Extra GPS noise** below. The simulated traces already have 10 to 25 m of error; at 50 m of extra noise a fix in the Loop can be nearer to a parallel street than to the true one, and more matches turn red (wrong street): the **Right street segment** readout, the share of fixes matched to their true street, falls. Use **Noise seed** to re-roll the noise: the accuracy is stable, which is what you want from a benchmark.\n\nLowering **Candidates per fix** or the **Route search node budget** (compile-time, flagged with a rebuild badge) makes the matcher cheaper and loses accuracy and continuity in dense blocks. Press **Time the matcher** to measure it.',
      options: {beta: 60, extraNoise: 40},
      camera: {longitude: -87.63, latitude: 41.88, zoom: 13.4, transitionMs: 1600}
    },
    {
      id: 'chains',
      controls: ['roadStyle'],
      readouts: ['chains'],
      title: 'GPULineMerge: streets as chains',
      body: 'The street graph is stored as 100,000 short segments. **`GPULineMerge`** joins segments that share an endpoint with exactly one other segment into maximal chains (the Shapely `linemerge` rule) and stops at junctions of three or more. Switch **Street layer** to *Merged chains*: each color is one chain, and the readout counts how many chains replace the segments.\n\nChains are what you want for labels, per-street statistics and anything that should treat a block as one object.',
      options: {extraNoise: 0, roadStyle: 'chains', showRaw: false, showMatched: false},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 13.2, transitionMs: 1400},
      highlight: {readout: 'chains'}
    },
    {
      id: 'limits',
      controls: ['sigma', 'beta', 'matchedColor', 'trackFocus'],
      title: 'Limits and what to try',
      body: 'The data is **simulated**, with Gaussian noise and no urban-canyon multipath, so real accuracy will be lower. Distances are planar (a local meter projection); the route search is bounded by the node budget and can overestimate or miss a route in very dense streets; both directions of a two-way street each take a candidate slot; and the model does not use time or speed. The outputs follow Valhalla/OSRM-style HMM matching, and `fmm` and `leuvenmapmatching` are the reference implementations to compare with.\n\nTry it: raise **GPS error (emission sigma)** and **Route tolerance (transition beta)** together, switch **Matched route color** to one color to see only the route, or set **Show one trace (0 = all)** to 40, 77 or 150 and read how its fixes line up with the truth.',
      options: {roadStyle: 'plain', showRaw: true, showMatched: true, sigma: 20, beta: 60}
    }
  ]
});
