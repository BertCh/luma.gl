// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, MONTREAL} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {directionFor} from '../../engine/ramps';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-bundles.md?raw';
import type {BixiBundlesOptions} from './bixi-bundles.compute';
import {
  BIKE_INK,
  BIKE_SPEED_EXTENT,
  BIKE_SPEED_RANGE,
  formatRideClock,
  getCrossingLegend,
  getDemotedBundleInk,
  getRideClassTable,
  getTrunkWidthLegend,
  RIDE_TRAIL_INK,
  RIDE_WINDOW_SECONDS,
  RIDES_ORIGIN_MS
} from './bixi-bundles-style';
import {FLOW_CREDITS} from './flows-style';

/** The night ground in both page themes: luminous trunks need a dark ground. */
const GROUND = 'dark' as const;

const MONTREAL_VIEW = {
  ...CITY_FRAMES.montreal,
  longitude: MONTREAL.places.plateau.lngLat[0],
  latitude: MONTREAL.places.plateau.lngLat[1],
  zoom: 11.5
} as const;

/** The camera of the playing step: the Plateau and downtown at the scale where bikes follow streets. */
const RIDES_VIEW = {...MONTREAL_VIEW, zoom: 12.3} as const;

const CREDIT = joinCredits(
  FLOW_CREDITS.bixi,
  FLOW_CREDITS.osrmRoutes,
  FLOW_CREDITS.montrealBoroughs
);

/** Orientation labels of the night ground (the basemap draws none): downtown, the Plateau and the mountain. */
const ORIENTATION = labelsFor(MONTREAL, ['downtown', 'plateau', 'mount-royal'], {
  downtown: {minZoom: 10.5},
  plateau: {minZoom: 10.5},
  'mount-royal': {minZoom: 10.5, tone: 'muted'}
});

/** The cartouche of a step: line 1 here, the sample line comes from the data. */
const cartouche = (title: string, subtitle: string, chips: readonly string[] = []) => ({
  title: {title, subtitle, chips}
});

export default defineScene<BixiBundlesOptions>({
  id: 'bixi-bundles',
  title: 'What does a month of Montreal bike trips look like?',
  chapter: 'flows',
  order: 11,
  summary:
    'The 30,000 busiest BIXI station pairs of August 2024, bundled on the GPU into trunks whose width follows the rides, with the routed rides of one morning running over them to show that bundles are not streets.',
  contributors: ['GPUEdgeBundling', 'GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'bixi-flows', role: 'station pairs, August 2024'},
    {id: 'montreal-boroughs', role: 'borough outlines (context)'},
    {id: 'poopdeck-bixi-rides', role: 'routed rides, 15 Aug 2024 07:30-10:00'}
  ],
  initialView: {...MONTREAL_VIEW},

  options: [
    {
      kind: 'slider',
      id: 'iterations',
      label: 'Iterations',
      group: 'Bundling',
      apply: 'param',
      min: 0,
      max: 32,
      step: 1,
      default: 16,
      format: value => (value === 0 ? 'straight' : String(value)),
      // Play sweeps the rounds from none to the default: the trunks form in front of the reader.
      autoSweep: {from: 0, to: 16, durationMs: 7000, ease: 'in-out'},
      help: 'How many advect, resample and smooth rounds run. The GPU gates every round past this count, so cost follows the slider while the compiled maximum stays fixed.'
    },
    {
      kind: 'slider',
      id: 'kernelRadius',
      label: 'Kernel radius',
      group: 'Bundling',
      apply: 'param',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.03,
      format: value => value.toFixed(3),
      help: 'Initial attraction radius as a share of the work box, which is the extent of the live pairs. Larger radii merge lines that are farther apart into fewer, thicker trunks. The readouts say it in metres.'
    },
    {
      kind: 'preset',
      id: 'radiusPreset',
      label: 'Radius presets',
      group: 'Bundling',
      presets: [
        {label: 'Fine', values: {kernelRadius: 0.01}},
        {label: 'Medium', values: {kernelRadius: 0.03}},
        {label: 'Coarse', values: {kernelRadius: 0.06}}
      ],
      help: 'Three kernel radii: fine keeps neighbourhoods apart, medium finds the corridors, coarse is too wide and merges neighbouring corridors into one blob.'
    },
    {
      kind: 'slider',
      id: 'decay',
      label: 'Radius decay',
      group: 'Bundling',
      apply: 'param',
      min: 0.5,
      max: 0.9,
      step: 0.01,
      default: 0.85,
      format: value => value.toFixed(2),
      help: 'The radius is multiplied by this every round. Lower values anneal quickly (coarse trunks, then tight); higher values keep pulling at long range.'
    },
    {
      kind: 'slider',
      id: 'stiffness',
      label: 'Stiffness (smoothing)',
      group: 'Bundling',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.5,
      format: value => value.toFixed(2),
      help: 'Strength of the Laplacian smoothing after each step. High values give smooth arcs; zero leaves kinks.'
    },
    {
      kind: 'slider',
      id: 'stepScale',
      label: 'Step scale',
      group: 'Bundling',
      apply: 'param',
      expert: true,
      min: 0.25,
      max: 2,
      step: 0.05,
      default: 1,
      format: value => value.toFixed(2),
      help: 'Multiplier on the advection step, which is one kernel radius at 1. Larger steps converge faster but can overshoot.'
    },
    {
      kind: 'slider',
      id: 'edges',
      label: 'Pairs drawn',
      group: 'Pairs',
      apply: 'param',
      min: 500,
      max: 30000,
      step: 500,
      default: 30000,
      unit: 'pairs',
      help: 'How many of the busiest station pairs enter the bundling, busiest first. The rest of the compiled capacity is masked, so this is a buffer write. The Pareto chart shows what the cut keeps.'
    },
    {
      kind: 'slider',
      id: 'minRides',
      label: 'Minimum rides on a pair',
      group: 'Pairs',
      apply: 'param',
      min: 1,
      max: 400,
      step: 1,
      default: 1,
      unit: 'rides',
      help: 'Also masks pairs with fewer rides in August. Masked pairs leave the density and the picture.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour by',
      group: 'Pairs',
      apply: 'param',
      display: 'segmented',
      default: 'rides',
      help: 'Rides on the pair in five classes (width always follows rides), or a qualitative pair: pairs that stay inside one borough against pairs that cross a borough line.',
      options: [
        {value: 'rides', label: 'Rides'},
        {value: 'crossing', label: 'Crossing'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showStraight',
      label: 'Straight edges (ghost)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the original straight edges as a faint 0.4 px ghost under the bundles, to see what bundling changed.'
    },
    {
      kind: 'toggle',
      id: 'compareStraight',
      label: 'Straight against bundled',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Splits the map at a divider: straight edges on the left, bundles on the right, with the same widths and classes.'
    },
    {
      kind: 'toggle',
      id: 'showBoroughs',
      label: 'Borough outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the borough and linked-city outlines as faint 0.5 px lines, the boundaries the crossing colours refer to.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Stations and hubs',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A dot for every station; the six stations with the most rides are ringed. Hover a station for its rides and its busiest pair.'
    },
    {
      kind: 'select',
      id: 'radiusRing',
      label: 'Ring shows',
      group: 'Display',
      apply: 'param',
      display: 'segmented',
      default: 'round',
      help: 'The dashed ring at downtown is the kernel radius in metres: the radius of the round the slider has reached, or the radius the first round starts with.',
      options: [
        {value: 'round', label: 'This round'},
        {value: 'start', label: 'First round'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRides',
      label: 'Show the morning rides',
      group: 'Rides',
      apply: 'param',
      default: false,
      help: 'Draws the routed rides of 15 August 2024, 07:30 to 10:00: a white bike per ride in progress and a fading trail. The bundles drop back to a faint 0.5 px.'
    },
    ...playbackOptions<BixiBundlesOptions>({
      ids: {play: 'play', time: 'time', speed: 'speed', loop: 'loop'},
      group: 'Rides',
      playing: false,
      time: {
        min: 0,
        max: RIDE_WINDOW_SECONDS,
        step: 30,
        default: 1800,
        label: 'Time of day',
        format: formatRideClock,
        help: 'Montreal local time (America/Toronto) between 07:30 and 10:00. Play advances it and the slider follows.'
      },
      speed: {
        min: 1,
        max: 20,
        step: 1,
        default: 4,
        unit: 'min/s',
        label: 'Speed',
        help: 'Minutes of the morning per real second. 4 min/s plays the whole window in about 40 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Rides',
      apply: 'param',
      min: 0.5,
      max: 15,
      step: 0.5,
      default: 4,
      unit: 'min',
      help: 'How far behind each bike its trail runs. The window of trail segments is a four-number parameter buffer.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Trail fade',
      group: 'Rides',
      apply: 'param',
      expert: true,
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Fraction of the trail over which it fades out, from the tail forward.'
    },
    {
      kind: 'toggle',
      id: 'bikeSpeed',
      label: 'Colour bikes by modelled speed',
      group: 'Rides',
      apply: 'param',
      default: false,
      help: 'Modelled, not measured: the routing engine spreads each ride over its route, so speed shows the route, not the rider.'
    },
    {
      kind: 'select',
      id: 'pointsPerEdge',
      label: 'Control points per edge',
      group: 'Compile-time',
      apply: 'compile',
      expert: true,
      default: '16',
      help: 'Polyline resolution of each edge (2 to 64 allowed). More points follow tighter bends but cost memory and splat work. Rebuilds the graph.',
      options: [
        {value: '8', label: '8 (coarse)'},
        {value: '16', label: '16'},
        {value: '24', label: '24'},
        {value: '32', label: '32 (fine)'}
      ]
    },
    {
      kind: 'select',
      id: 'densityResolution',
      label: 'Density grid',
      group: 'Compile-time',
      apply: 'compile',
      expert: true,
      default: '256',
      help: 'Cells per axis of the density grid the control points splat onto. Finer grids resolve smaller gaps between bundles. Rebuilds the graph.',
      options: [
        {value: '128', label: '128 x 128'},
        {value: '256', label: '256 x 256'},
        {value: '512', label: '512 x 512'}
      ]
    }
  ],

  readouts: [
    {
      id: 'drawn',
      label: 'Pairs drawn',
      emphasis: 'tile',
      help: 'Station pairs in the bundling out of every pair with at least 3 rides in August; rides in both directions are summed.'
    },
    {
      id: 'ridesShare',
      label: 'Rides on the pairs drawn',
      format: 'percent',
      help: 'Share of all rides between two different stations that run on the pairs drawn.'
    },
    {
      id: 'lengthBias',
      label: 'Median pair length, drawn vs left out',
      help: 'Straight-line length of the pairs: busy pairs are short and central, so a top-N cut biases the map toward them.'
    },
    {
      id: 'paretoChart',
      label: 'Pareto: pairs against share of rides',
      kind: 'chart',
      help: 'Pairs ranked by rides (log axis) against the cumulative share of rides. Click the curve to move the cutoff.'
    },
    {
      id: 'iteration',
      label: 'Iteration',
      format: 'integer',
      help: 'Rounds of advect, resample and smooth that ran.'
    },
    {
      id: 'radiusNow',
      label: 'Kernel radius, this round',
      format: 'meters',
      help: 'Initial radius times the decay for each round already run, in metres on the ground: the share of the work box times the box side.'
    },
    {
      id: 'radiusStart',
      label: 'Kernel radius, first round',
      format: 'meters',
      help: 'The initial radius in metres on the ground: its share of the work box times the box side.'
    },
    {
      id: 'stretch',
      label: 'Mean path stretch',
      format: 'decimal',
      help: 'Total bundled length over total straight length of the pairs drawn.'
    },
    {
      id: 'stretchChart',
      label: 'Stretch of bundled pairs',
      kind: 'chart',
      help: 'Histogram of bundled length over straight length for the pairs longer than 300 m.'
    },
    {
      id: 'crossingShare',
      label: 'Rides that cross a borough line',
      format: 'percent',
      help: 'Share of the rides on the pairs drawn whose two stations are in different boroughs or municipalities, as BIXI publishes them.'
    },
    {
      id: 'bikes',
      label: 'Bikes on the road',
      format: 'integer',
      help: 'Rides active at the playhead, from the GPU count.'
    },
    {id: 'clock', label: 'Montreal time', help: 'Local time (America/Toronto) of the playhead.'},
    {
      id: 'ridesChart',
      label: 'Rides in progress',
      kind: 'chart',
      help: 'Routed rides in progress each minute between 07:30 and 10:00 on 15 August 2024; the marker is the playhead. Click to move the clock.'
    },
    {
      id: 'offStreetBundled',
      label: 'Bundle length off the streets',
      format: 'percent',
      help: 'Share of the bundled lines that lies farther from every routed ride than the distance below. The rides are one morning, so quiet streets count as off.'
    },
    {
      id: 'offStreetStraight',
      label: 'Straight length off the streets',
      format: 'percent',
      help: 'The same share for the straight lines between the stations.'
    },
    {
      id: 'streetDistance',
      label: 'Street distance',
      format: 'meters',
      hood: true,
      help: 'A line counts as on a street when a routed ride passes within this distance of it.'
    },
    {
      id: 'areaStraight',
      label: 'Map touched, straight',
      format: 'decimal',
      unit: 'km²',
      help: 'Area of the 0.002 degree map cells (about 220 by 160 m) that a straight line passes through.'
    },
    {
      id: 'areaBundled',
      label: 'Map touched, bundled',
      format: 'decimal',
      unit: 'km²',
      help: 'The same area for the bundled lines: the ink the bundling saved is the difference.'
    },
    {id: 'rides', label: 'Rides in the morning window', format: 'integer', hood: true},
    {id: 'trailSegments', label: 'Trail segments', format: 'integer', hood: true},
    {id: 'controlPoints', label: 'Control points', format: 'integer', hood: true},
    {
      id: 'boxSide',
      label: 'Work box side',
      format: 'meters',
      hood: true,
      help: 'The square the kernel radius is a share of: the larger side of the live pairs, with longitude scaled by the cosine of the latitude, plus 5 % on each side.'
    }
  ],

  pipeline: [
    {
      id: 'select',
      label: 'Busiest pairs',
      detail: 'Pairs ranked by rides; the cut and the ride filter are an edge mask, a buffer write'
    },
    {
      id: 'bundle',
      label: 'Bundle',
      detail:
        'GPUEdgeBundling: splat density, advect uphill, resample, smooth; the radius anneals every round'
    },
    {
      id: 'draw',
      label: 'Ribbons',
      detail: 'The layer reads the paths buffer: width by rides, heaviest drawn last'
    },
    {
      id: 'playhead',
      label: 'Playhead',
      detail: 'One thread per ride: binary-search its times, interpolate its position'
    },
    {
      id: 'window',
      label: 'Time window',
      detail: 'One thread per segment: overlap with the trail, fade weight, clip fractions'
    }
  ],

  legends: (state, data) => {
    const legends: LegendSpec[] = [];
    const counts = data['rideClassCounts'] as number[] | undefined;
    const maximumRides = (data['maximumRides'] as number | undefined) ?? 2000;
    if (state.showRides && !state.compareStraight) {
      legends.push({
        kind: 'categories',
        title: 'Routed rides, 15 August',
        entries: [
          {color: RIDE_TRAIL_INK[GROUND], label: 'Route behind a bike', shape: 'line'},
          {color: BIKE_INK, label: 'Bike now', shape: 'dot'},
          {
            color: getDemotedBundleInk(GROUND),
            label: 'Bundled pair (faint)',
            shape: 'line'
          }
        ],
        note: 'Routes are modelled on the street network, not GPS traces.'
      });
      if (state.bikeSpeed) {
        legends.push({
          kind: 'ramp',
          title: 'Bike speed, modelled',
          ramp: 'magma',
          reverse: directionFor(GROUND, 'magma').reverse,
          range: BIKE_SPEED_RANGE,
          extent: BIKE_SPEED_EXTENT,
          unit: 'm/s',
          note: 'Modelled: the routing engine spreads each ride over its route.'
        });
      }
      return legends;
    }
    if (state.colorBy === 'rides') {
      legends.push(
        getClassTableLegend(getRideClassTable(GROUND), {
          title: 'Rides on the pair, August 2024',
          id: 'ride-classes',
          counts,
          layout: 'list',
          interactive: true
        })
      );
    } else {
      legends.push(getCrossingLegend(GROUND, state.showBoroughs));
    }
    legends.push(getTrunkWidthLegend(maximumRides, GROUND));
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUEdgeBundling, createGPUEdgeBundlingParameterValues} from '@luma.gl/experimental/gpu-network';
import {GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'bundling'});
graph.add(new GPUEdgeBundling({
  positions, sourceVertices, targetVertices,   // 905 stations, the busiest pairs by rides
  edgeMask,                                    // busiest ${state.edges}, at least ${state.minRides} rides
  geographic: true,                            // longitude scaled by cos(latitude): a round kernel
  pointsPerEdge: ${state.pointsPerEdge},
  iterations: 32,                              // compiled maximum
  densityResolution: ${state.densityResolution},
  parameters: params.importToGraph(graph),
  paths                                        // float32x2 polylines, drawn as is
}));
const compiled = graph.compile();

// kernelRadius is a share of the square work box around the live pairs
params.write(createGPUEdgeBundlingParameterValues(
  {activeIterations: ${state.iterations}, kernelRadius: ${state.kernelRadius}, lambda: ${state.decay}, smoothing: ${state.stiffness}, stepScale: ${state.stepScale}}, 'uint32'));
compiled.encode(commandEncoder, {parameters: undefined});

// width by rides, heaviest last: instance i draws pair (count - 1 - i)
width = 0.5 + 5.5 * sqrt(rides / maxRides);

// rides: one more graph, parameters per frame
playheadParams.write(getGPUTrajectoryPlayheadParameterValues({playhead: seconds, maxGap: 0}));`,

  basemap: ground('night'),
  furniture: {
    title: {
      title: 'A month of Montreal bike rides',
      subtitle: 'Rides per station pair, August 2024, kernel-density bundling'
    },
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat: 'Bundled lines are an abstraction of pairs, not routes.'
  },
  annotations: ORIENTATION,

  about: {
    what: 'Previously: communities against boroughs in the BIXI network. This closes the flows chapter.\n\n`GPUEdgeBundling` turns straight edges into bundled polylines with kernel-density edge bundling on the GPU, as in the airline story, here with `geographic: true` so the kernel is round on the ground at Montreal. A `GPUTrajectoryPlayhead` graph interpolates every routed ride at the clock and a `GPUTimeWindowFilter` graph selects the trail segments that are still visible.',
    why: 'A flow map of every station pair is a hairball. Bundling shows where traffic concentrates, and width by rides keeps the volume readable. Running a morning of routed rides over it shows what bundling is: a summary of pairs. The corridors are an optimisation outcome, not streets.',
    howToRead:
      'Each trunk is many station pairs merged: its width follows the rides on the pair (square-root scale, one scale for every step) and its colour is the ride class. The pairs drawn are a top-N cut, so the map over-represents short, central, repeated trips; the Pareto chart shows what is left out. The rides are OSRM bicycle-profile routes between the start and end station with times spread along the route, so they are modelled, not GPS traces, and speeds are modelled too. Rides under way at 07:30 begin at the first frame; the bundles use the whole month, the rides one morning. The bundling schedule is chaotic in 32-bit floats: two runs agree only statistically.'
  },

  create: async ctx => (await import('./bixi-bundles.compute')).createBixiBundles(ctx),

  story: storyFromMarkdown<BixiBundlesOptions>(narrative, {
    straight: {
      headline: 'The busiest pairs still make a hairball',
      textAlternative:
        'Dark map of Montreal with thousands of straight gold lines between bike stations, thicker for busier pairs, piled into a bright tangle downtown, with a Pareto curve of pairs against share of rides.',
      optionsMode: 'fresh',
      options: {iterations: 0},
      controls: ['edges'],
      readouts: ['drawn', 'ridesShare', 'lengthBias', 'paretoChart'],
      camera: {...MONTREAL_VIEW, transitionMs: 1400},
      furniture: cartouche(
        'A month of rides, drawn straight',
        'Rides per station pair, August 2024'
      ),
      annotations: labelsFor(MONTREAL, ['old-montreal', 'mile-end', 'parc-la-fontaine'], {
        'old-montreal': {minZoom: 11, tone: 'muted'},
        'mile-end': {minZoom: 11, tone: 'muted'},
        'parc-la-fontaine': {minZoom: 11, tone: 'muted'}
      }),
      stage: 'select'
    },
    bundle: {
      headline: 'Busy pairs gather into luminous trunks',
      textAlternative:
        'The same map after bundling: straight lines pulled into a few thick gold trunks that run into downtown, with a dashed ring showing the kernel radius and a note on the busiest trunk.',
      optionsMode: 'fresh',
      options: {iterations: 16, showStraight: true},
      controls: ['iterations'],
      readouts: ['iteration', 'radiusNow', 'stretch'],
      camera: {...MONTREAL_VIEW, transitionMs: 1400},
      furniture: cartouche(
        'Busy pairs gather into trunks',
        'Kernel-density bundling, ring = kernel radius'
      ),
      annotations: labelsFor(MONTREAL, ['old-montreal', 'mile-end'], {
        'old-montreal': {minZoom: 11, tone: 'muted'},
        'mile-end': {minZoom: 11, tone: 'muted'}
      }),
      stage: 'bundle'
    },
    radius: {
      headline: 'The radius sets the scale of a trunk',
      textAlternative:
        'Bundled trunks at three kernel radii: fine radii keep separate neighbourhood corridors, the coarsest merges them into one blob; crossing mode paints borough-crossing pairs orange and the rest slate.',
      optionsMode: 'fresh',
      options: {iterations: 16, radiusRing: 'start', showBoroughs: true},
      controls: ['radiusPreset', 'colorBy'],
      readouts: ['radiusStart', 'crossingShare', 'stretch', 'stretchChart'],
      camera: {...MONTREAL_VIEW, zoom: 11.9, transitionMs: 1400},
      furniture: cartouche(
        'The radius sets the scale',
        'Trunks at three kernel radii, ring = first round'
      ),
      annotations: labelsFor(MONTREAL, ['lachine-canal', 'old-montreal'], {
        'lachine-canal': {minZoom: 11, tone: 'muted'},
        'old-montreal': {minZoom: 11, tone: 'muted'}
      }),
      stage: 'bundle'
    },
    'not-streets': {
      headline: 'Bundles are not streets',
      textAlternative:
        'Faint gold bundles under white bikes with cool blue trails that follow the street grid of the Plateau and downtown, with Mount Royal labelled to the west.',
      optionsMode: 'fresh',
      options: {showRides: true, play: true, time: 1800, speed: 4, trailMinutes: 4},
      controls: ['play', 'speed'],
      readouts: ['bikes', 'clock', 'offStreetBundled', 'ridesChart'],
      camera: {...RIDES_VIEW, transitionMs: 1400},
      furniture: {
        ...cartouche('Bundles are not streets', 'Routed rides, 15 August 2024, 07:30 to 10:00', [
          'Modelled routes'
        ]),
        clock: {
          option: 'time',
          // The archive counts seconds from 07:30 Montreal time, which is 11:30 UTC.
          time: {origin: new Date(RIDES_ORIGIN_MS).toISOString(), unit: 'seconds'},
          zones: ['America/Toronto', 'UTC'],
          show: 'time',
          progress: [0, RIDE_WINDOW_SECONDS]
        }
      },
      annotations: labelsFor(MONTREAL, ['mile-end', 'old-montreal', 'parc-la-fontaine'], {
        'mile-end': {minZoom: 11, tone: 'muted'},
        'old-montreal': {minZoom: 11, tone: 'muted'},
        'parc-la-fontaine': {minZoom: 11, tone: 'muted'}
      }),
      stage: 'playhead'
    },
    trust: {
      headline: 'Bundling saves ink and costs truth',
      textAlternative:
        'A map split by a divider: straight gold lines on the left, bundled trunks on the right, with the same widths and classes, and the area each touches in the readouts.',
      optionsMode: 'fresh',
      options: {iterations: 16, compareStraight: true},
      controls: ['iterations', 'kernelRadius', 'edges'],
      readouts: ['areaStraight', 'areaBundled', 'stretch'],
      compare: {labels: ['Straight', 'Bundled'], position: 0.5},
      camera: {...MONTREAL_VIEW, transitionMs: 1400},
      furniture: {
        ...cartouche('Bundling saves ink and costs truth', 'Straight against bundled, same widths'),
        clock: false
      },
      annotations: labelsFor(MONTREAL, ['old-montreal', 'lachine-canal'], {
        'old-montreal': {minZoom: 11, tone: 'muted'},
        'lachine-canal': {minZoom: 11, tone: 'muted'}
      }),
      stage: 'draw'
    }
  })
});
