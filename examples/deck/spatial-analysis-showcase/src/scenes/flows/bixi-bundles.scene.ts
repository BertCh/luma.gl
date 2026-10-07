// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-bundles.md?raw';
import type {BixiBundlesOptions} from './bixi-bundles.compute';

const VALUE_TITLES: Record<string, {title: string; unit: string}> = {
  rides: {title: 'Rides on the pair, August', unit: 'rides'},
  length: {title: 'Pair length', unit: 'km'},
  crossing: {title: 'Crosses a borough boundary', unit: ''}
};

const formatRideTime = (seconds: number): string => {
  const total = 7.5 * 3600 + seconds;
  return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}`;
};

export default defineScene<BixiBundlesOptions>({
  id: 'bixi-bundles',
  title: 'Montreal bike corridors, bundled',
  chapter: 'flows',
  order: 12,
  summary:
    'GPUEdgeBundling pulls the 30,000 busiest BIXI station pairs of August 2024 into corridors, with 8,365 routed rides of one morning running over them.',
  contributors: ['GPUEdgeBundling', 'GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'bixi-flows', role: 'station pairs, August 2024'},
    {id: 'poopdeck-bixi-rides', role: 'routed rides, 15 Aug 2024 07:30-10:00'}
  ],
  initialView: {longitude: -73.59, latitude: 45.52, zoom: 11.3},

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
      format: value => (value === 0 ? '0 (straight lines)' : String(value)),
      help: 'How many advect, resample and smooth rounds run. The GPU gates every round past this count, so cost follows the slider while the compiled maximum stays 32.'
    },
    {
      kind: 'slider',
      id: 'edges',
      label: 'Edges used',
      group: 'Bundling',
      apply: 'param',
      min: 500,
      max: 30000,
      step: 500,
      default: 8000,
      help: 'How many of the busiest station pairs enter the bundling, busiest first. The rest of the compiled capacity (30,000 edges) is masked, so this is a buffer write.'
    },
    {
      kind: 'slider',
      id: 'minRides',
      label: 'Minimum rides on a pair',
      group: 'Bundling',
      apply: 'param',
      min: 1,
      max: 400,
      step: 1,
      default: 1,
      unit: 'rides',
      help: 'Also masks pairs with fewer rides in August. Masked edges leave the density and the picture.'
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
      default: 0.02,
      format: value => value.toFixed(3),
      help: 'Initial attraction radius as a fraction of the work box side. Larger radii merge edges that are farther apart into fewer, thicker bundles.'
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
      help: 'The radius is multiplied by this every iteration. Lower values anneal quickly (coarse bundles, then tight); higher values keep pulling at long range.'
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
      min: 0.25,
      max: 2,
      step: 0.05,
      default: 1,
      format: value => value.toFixed(2),
      help: 'Multiplier on the advection step, which is one kernel radius at 1. Larger steps converge faster but can overshoot.'
    },
    {
      kind: 'select',
      id: 'pointsPerEdge',
      label: 'Control points per edge',
      group: 'Compile-time',
      apply: 'compile',
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
      default: '256',
      help: 'Cells per axis of the density grid the control points splat onto. Finer grids resolve smaller gaps between bundles. Rebuilds the graph.',
      options: [
        {value: '128', label: '128 x 128'},
        {value: '256', label: '256 x 256'},
        {value: '512', label: '512 x 512'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour edges by',
      group: 'Display',
      apply: 'param',
      default: 'rides',
      help: 'Edge attribute mapped through the ramp: rides on the pair, its length, or whether its two stations are in different boroughs.',
      options: [
        {value: 'rides', label: 'Rides on the pair'},
        {value: 'length', label: 'Pair length'},
        {value: 'crossing', label: 'Crosses a borough boundary'},
        {value: 'plain', label: 'Single colour'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy === 'plain',
      help: 'Viridis and cividis stay legible on both basemaps; magma and inferno fade to black at the low end.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Edge opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.45,
      help: 'Lines are 1 px and translucent, so they add up and crowded bundles read brighter. Lower it for the straight hairball.'
    },
    {
      kind: 'toggle',
      id: 'showStraight',
      label: 'Straight edges (ghost)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the original straight edges faintly under the bundles, to see what bundling changed.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Stations and hubs',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots for every station; orange dots mark the 24 stations with the most rides. Hover a station for its name.'
    },
    {
      kind: 'toggle',
      id: 'showRides',
      label: 'Show the morning rides',
      group: 'Rides',
      apply: 'param',
      default: false,
      help: 'Draws the 8,365 routed rides of 15 August 2024, 07:30 to 10:00: a bike per ride in progress and a fading trail.'
    },
    ...playbackOptions<BixiBundlesOptions>({
      ids: {play: 'play', time: 'time', speed: 'speed', loop: 'loop'},
      group: 'Rides',
      playing: false,
      time: {
        min: 0,
        max: 9000,
        step: 30,
        default: 1800,
        label: 'Time of day',
        format: formatRideTime,
        help: 'Time of the morning between 07:30 and 10:00, Montreal local time. Play advances it and the slider follows.'
      },
      speed: {
        min: 1,
        max: 20,
        step: 1,
        default: 4,
        unit: 'min/s',
        label: 'Speed',
        help: 'Minutes of the morning per real second. 4 min/s plays the 2.5 hours in about 40 seconds.'
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
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Fraction of the trail over which it fades out, from the tail forward.'
    },
    {
      kind: 'slider',
      id: 'bikeSize',
      label: 'Bike size',
      group: 'Rides',
      apply: 'param',
      min: 2,
      max: 10,
      step: 0.5,
      default: 4,
      unit: 'px',
      help: 'Radius of the bike dots, coloured by speed.'
    }
  ],

  readouts: [
    {
      id: 'ridesChart',
      label: 'Rides in progress',
      kind: 'chart',
      help: 'Routed rides in progress each minute between 07:30 and 10:00 on 15 August 2024; the rule is the playhead.'
    },
    {id: 'clock', label: 'Time of day'},
    {
      id: 'bikes',
      label: 'Bikes in motion',
      format: 'integer',
      help: 'Rides active at the playhead, from the GPU count.'
    },
    {id: 'trailSegments', label: 'Trail segments', format: 'integer'},
    {id: 'rides', label: 'Rides in the morning window', format: 'integer'},
    {
      id: 'stretchChart',
      label: 'Stretch of bundled pairs',
      kind: 'chart',
      help: 'Histogram of path length over straight length for the pairs that pass the filters.'
    },
    {
      id: 'inkSaved',
      label: 'Ink saved by bundling',
      format: 'percent',
      help: 'One minus the map cells (about 150 m) touched by the bundled lines over the cells touched by the straight lines.'
    },
    {
      id: 'stretch',
      label: 'Mean path stretch',
      format: 'decimal',
      help: 'Total bundled length over total straight length.'
    },
    {id: 'edges', label: 'Edges bundled'},
    {id: 'controlPoints', label: 'Control points', format: 'integer'},
    {
      id: 'ridesCovered',
      label: 'Station-pair rides covered',
      format: 'percent',
      help: 'Share of the rides between stations (pairs with at least 3 rides) on the edges that pass the filters.'
    },
    {id: 'coverageStraight', label: 'Map cells, straight', format: 'integer'},
    {id: 'coverageBundled', label: 'Map cells, bundled', format: 'integer'},
    {id: 'ridesShare', label: 'Rides on the 30,000 busiest pairs', format: 'percent'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.colorBy !== 'plain') {
      legends.push({
        kind: 'ramp',
        id: 'edge-value',
        title: VALUE_TITLES[state.colorBy].title,
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: state.colorBy === 'rides',
        unit: VALUE_TITLES[state.colorBy].unit || undefined
      });
    }
    if (state.showRides) {
      legends.push({
        kind: 'ramp',
        title: 'Bike speed',
        ramp: 'inferno',
        extent: [0, 7],
        unit: 'm/s',
        labels: ['slow', 'fast']
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUEdgeBundling, createGPUEdgeBundlingParameterValues} from '@luma.gl/experimental/gpu-network';
import {GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'bundling'});
graph.add(new GPUEdgeBundling({
  positions, sourceVertices, targetVertices,   // 905 stations, 30,000 edges
  edgeMask,                                    // rides >= ${state.minRides}, busiest ${state.edges}
  geographic: true,
  pointsPerEdge: ${state.pointsPerEdge},
  iterations: 32,                              // compiled maximum
  densityResolution: ${state.densityResolution},
  parameters: params.importToGraph(graph),
  paths                                        // float32x2 polylines, drawn as is
}));
const compiled = graph.compile();

params.write(createGPUEdgeBundlingParameterValues(
  {activeIterations: ${state.iterations}, kernelRadius: ${state.kernelRadius}, lambda: ${state.decay}, smoothing: ${state.stiffness}, stepScale: ${state.stepScale}}, 'uint32'));
compiled.encode(commandEncoder, {parameters: undefined});

// rides: one more graph, parameters per frame
playheadParams.write(getGPUTrajectoryPlayheadParameterValues({playhead: seconds, maxGap: 0}));`,

  about: {
    what: '`GPUEdgeBundling` turns straight edges into bundled polylines with kernel-density edge bundling on the GPU. A `GPUTrajectoryPlayhead` graph interpolates every ride at the clock and a `GPUTimeWindowFilter` graph selects the trail segments that are still visible.',
    why: 'A flow map of every station pair is a hairball. Bundling shows where traffic concentrates, and running the day over it shows when: the same corridors that carry the month carry the morning peak.',
    howToRead:
      'Lines are station pairs, coloured by the attribute you pick; where many pairs share a route they merge into a bright corridor. Dots are bikes, coloured by modelled speed, with a fading trail behind them. The chart counts rides in progress.'
  },

  create: async ctx => (await import('./bixi-bundles.compute')).createBixiBundles(ctx),

  story: storyFromMarkdown<BixiBundlesOptions>(narrative, {
    'the-question': {
      controls: ['iterations', 'edges'],
      readouts: ['edges', 'ridesCovered'],
      camera: {longitude: -73.59, latitude: 45.52, zoom: 11.3, transitionMs: 1400},
      options: {iterations: 0, edges: 8000, showStraight: false, showRides: false, play: false},
      highlight: {readout: 'ridesCovered'}
    },
    bundle: {
      controls: ['iterations', 'kernelRadius', 'edges'],
      readouts: ['inkSaved', 'stretch', 'stretchChart'],
      options: {iterations: 16, kernelRadius: 0.02},
      highlight: {readout: 'inkSaved'}
    },
    tune: {
      controls: ['decay', 'stiffness', 'colorBy', 'minRides'],
      readouts: ['coverageBundled', 'controlPoints'],
      camera: {longitude: -73.58, latitude: 45.52, zoom: 11.8, transitionMs: 1200},
      options: {colorBy: 'crossing', ramp: 'viridis'},
      highlight: {readout: 'coverageBundled'}
    },
    rides: {
      controls: ['showRides', 'play', 'speed', 'trailMinutes'],
      readouts: ['clock', 'bikes', 'ridesChart'],
      camera: {longitude: -73.58, latitude: 45.52, zoom: 12.3, transitionMs: 1400},
      options: {
        showRides: true,
        play: true,
        speed: 4,
        time: 1800,
        colorBy: 'plain',
        opacity: 0.3,
        trailMinutes: 4
      },
      highlight: {readout: 'bikes'}
    },
    limits: {
      controls: ['colorBy', 'showStraight', 'kernelRadius', 'showStations'],
      readouts: ['inkSaved', 'rides'],
      camera: {longitude: -73.59, latitude: 45.52, zoom: 11.3, transitionMs: 1200},
      options: {showRides: false, play: false, colorBy: 'rides', opacity: 0.45}
    }
  })
});
