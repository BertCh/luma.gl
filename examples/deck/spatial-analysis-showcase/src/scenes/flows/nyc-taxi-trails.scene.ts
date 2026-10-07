// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-trails.md?raw';
import type {NycTaxiTrailsOptions} from './nyc-taxi-trails.compute';

/** 07:59:00 local, the time origin of the dataset, in seconds since midnight. */
const ORIGIN_SECONDS_OF_DAY = 7 * 3600 + 59 * 60;
/** Value ranges of the colour attributes; mirrors the compute module. */
const RANGES = {fare: [4, 40], distance: [0.5, 10]} as const;

const formatClock = (seconds: number): string => {
  const total = ORIGIN_SECONDS_OF_DAY + Math.round(seconds);
  return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
};

export default defineScene<NycTaxiTrailsOptions>({
  id: 'nyc-taxi-trails',
  title: 'The Friday morning rush, route by route',
  chapter: 'flows',
  order: 5,
  summary:
    'Nine thousand OSRM-routed taxi trips between 08:00 and 08:30 on Friday 2 January 2015: a GPU playhead interpolates every taxi, a time-window graph draws fading trails, and a clock you can scrub follows the slider.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'poopdeck-nyc-taxi-paths', role: '9,000 routed taxi trips with a time on every vertex'}
  ],
  initialView: {longitude: -73.965, latitude: 40.745, zoom: 11.4},

  options: [
    ...playbackOptions<NycTaxiTrailsOptions>({
      ids: {play: 'play', time: 'time', speed: 'playSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: 1920,
        step: 5,
        default: 480,
        label: 'Clock',
        format: formatClock,
        help: 'Time of the playhead, local New York time on Friday 2 January 2015. The dataset starts at 07:59:00; the playback clock moves this slider.'
      },
      speed: {
        min: 10,
        max: 240,
        step: 10,
        default: 60,
        unit: 'x',
        label: 'Playback speed',
        help: 'Simulated seconds per real second. 60x plays one minute per second; 240x plays the whole window in eight seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails',
      apply: 'param',
      min: 0.5,
      max: 10,
      step: 0.5,
      default: 3,
      unit: 'min',
      help: 'Width of the time window behind the playhead, in minutes of taxi time. It is a parameter-buffer write.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Fade',
      group: 'Trails',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Fraction of the trail whose tail fades out (the window start fade duration). 0 draws the whole trail at full strength.'
    },
    {
      kind: 'slider',
      id: 'trailWidth',
      label: 'Trail width',
      group: 'Trails',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.25,
      default: 1.75,
      unit: 'px',
      help: 'Width of the trail lines in pixels.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails',
      apply: 'param',
      default: true,
      help: 'Turns the time-window graph off (it is not encoded) and leaves only the moving heads.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every route',
      group: 'Trails',
      apply: 'param',
      default: false,
      help: 'Draws all 402,000 route segments of every trip as a faint network, so you can see the roads the 9,000 trips use.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour by',
      group: 'Colour',
      apply: 'param',
      default: 'fare',
      help: 'A per-trip attribute read through the segment track index, so trails and heads share it. None uses one warm colour.',
      options: [
        {value: 'fare', label: 'Fare (USD, no tips)'},
        {value: 'distance', label: 'Trip distance (miles)'},
        {value: 'none', label: 'None (one colour)'}
      ]
    },
    {
      kind: 'slider',
      id: 'hideBelow',
      label: 'Hide trips below',
      group: 'Colour',
      apply: 'param',
      min: 0,
      max: 40,
      step: 1,
      default: 0,
      disabledWhen: state => state.colorBy === 'none',
      help: 'Hides trips whose colour attribute is at or below this value (USD or miles): a shader threshold, so geometry and graphs are untouched.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Colour',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy === 'none',
      help: 'Perceptually uniform ramps; cividis is optimised for colour-vision deficiency.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Head radius',
      group: 'Colour',
      apply: 'param',
      min: 1,
      max: 8,
      step: 0.5,
      default: 3,
      unit: 'px',
      help: 'Radius of each taxi head in pixels.'
    }
  ],

  readouts: [
    {id: 'clock', label: 'Clock'},
    {id: 'trips', label: 'Trips in the dataset', format: 'integer'},
    {id: 'vertices', label: 'Route vertices', format: 'integer'},
    {id: 'segmentsTotal', label: 'Route segments', format: 'integer'},
    {
      id: 'active',
      label: 'Taxis on the road',
      format: 'integer',
      help: 'Trips whose start is before the clock and whose end is after it, from the playhead graph.'
    },
    {
      id: 'segments',
      label: 'Trail segments drawn',
      format: 'integer',
      help: 'Segments inside the trail window, from the time-window graph.'
    },
    {
      id: 'overflow',
      label: 'List overflow',
      help: 'Yes means a compact id list was shorter than the number of live taxis or segments.'
    },
    {id: 'roadChart', label: 'Taxis on the road', kind: 'chart'}
  ],

  legends: state => {
    if (state.colorBy === 'none') {
      return [
        {
          kind: 'categories',
          title: 'Taxi trails',
          entries: [{color: [255, 214, 120, 255], label: 'Last few minutes of a route'}],
          note: 'The bright dot is the taxi now.'
        }
      ];
    }
    const range = RANGES[state.colorBy];
    return [
      {
        kind: 'ramp',
        title: state.colorBy === 'fare' ? 'Fare' : 'Trip distance',
        ramp: state.ramp,
        extent: range,
        unit: state.colorBy === 'fare' ? 'USD' : 'miles',
        format: value => (state.colorBy === 'fare' ? `$${value.toFixed(0)}` : value.toFixed(0))
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTimeWindowFilter} from '@luma.gl/experimental/gpu-dataframe';
import {GPUTrajectoryPlayhead} from '@luma.gl/experimental/gpu-spatial-analysis';

// every taxi at the clock: interpolated position, heading, a compact list of live taxis
const playheadGraph = new GPUCommandGraph(device);
playheadGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets, parameters: playhead.importToGraph(playheadGraph),
  currentPositions, headings, speeds, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                                  // indirect draw count for the heads
}));

// route segments of the last ${state.trailMinutes} min, with fade weights and clip fractions
const trailGraph = new GPUCommandGraph(device);
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStartTimes, endTimestamps: segmentEndTimes,
  window: window.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, trackIds: segmentTracks, trackVisibleCounts, drawInstanceCount
}));

// every frame: write the clock and the window (no recompile), then encode both
playhead.write(getGPUTrajectoryPlayheadParameterValues({playhead: seconds}));
window.write(getGPUTimeWindowParameterValues({
  start: seconds - ${state.trailMinutes * 60}, end: seconds, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));
playheadCompiled.encode(commandEncoder, {parameters: undefined});
trailCompiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryPlayhead` interpolates every trajectory to one instant in a compute pass and compacts the ids of the live ones; `GPUTimeWindowFilter` selects the route segments inside a time window, weights them by how recent they are and clips the segment at the window edge. The layers read both outputs directly from GPU storage, with the instance counts written by the GPU.',
    why: 'Replaying movement is the way to see rhythm: where trips concentrate, which corridors carry them, how the picture changes in minutes. Because the clock and the window are small buffers, one compiled graph plays any speed, any trail length and any scrub position.',
    howToRead:
      'Each line is a taxi route over the last few minutes, brightest at the head; colour is the trip fare or distance. The chart under the map counts trips on the road. The routes come from a routing engine between the recorded pickup and dropoff, not from GPS, so a taxi never queues and every taxi on an avenue is routed along the same line.'
  },

  create: async ctx => (await import('./nyc-taxi-trails.compute')).createNycTaxiTrails(ctx),

  story: storyFromMarkdown<NycTaxiTrailsOptions>(narrative, {
    'the-question': {
      controls: ['play', 'playSpeed'],
      readouts: ['clock', 'active'],
      options: {
        play: true,
        playSpeed: 60,
        trailMinutes: 3,
        tailFade: 0.8,
        colorBy: 'fare',
        hideBelow: 0,
        showBackdrop: false
      },
      camera: {
        longitude: -73.965,
        latitude: 40.745,
        zoom: 11.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1400
      }
    },
    trails: {
      controls: ['trailMinutes', 'tailFade', 'trailWidth'],
      readouts: ['segments', 'active'],
      options: {trailMinutes: 6, play: true},
      camera: {longitude: -73.98, latitude: 40.75, zoom: 12.4, transitionMs: 1800}
    },
    clock: {
      controls: ['play', 'time', 'playSpeed'],
      readouts: ['clock', 'active', 'roadChart'],
      options: {trailMinutes: 3, play: false, time: 900},
      camera: {longitude: -73.965, latitude: 40.745, zoom: 11.4, transitionMs: 1600}
    },
    fare: {
      controls: ['colorBy', 'hideBelow', 'ramp'],
      readouts: ['active', 'segments'],
      options: {colorBy: 'fare', hideBelow: 20, play: true},
      camera: {longitude: -73.95, latitude: 40.72, zoom: 10.9, transitionMs: 1800}
    },
    routes: {
      controls: ['showBackdrop', 'showTrails', 'trailMinutes'],
      readouts: ['segmentsTotal', 'trips'],
      options: {hideBelow: 0, showBackdrop: true, trailMinutes: 2, play: true},
      camera: {longitude: -73.985, latitude: 40.74, zoom: 11.8, transitionMs: 1800}
    },
    limits: {
      controls: ['loop', 'playSpeed'],
      readouts: ['overflow', 'vertices'],
      options: {showBackdrop: false, play: true}
    }
  })
});
