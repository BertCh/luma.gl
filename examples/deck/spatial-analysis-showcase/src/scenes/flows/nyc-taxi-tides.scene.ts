// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-tides.md?raw';
import type {NycTaxiTidesOptions} from './nyc-taxi-tides.compute';
import {formatTaxiTime} from './nyc-taxi-data';

/** Mirrors the compute module without importing it (scene files stay light). */
const FIRST_HOUR = 1;
const LAST_HOUR = 37.5;

const compact = (value: number): string =>
  value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k` : value.toFixed(0);

export default defineScene<NycTaxiTidesOptions>({
  id: 'nyc-taxi-tides',
  title: 'Where does Manhattan fill up and empty?',
  chapter: 'flows',
  order: 3,
  summary:
    'Net taxi arrivals minus departures on a hexagon lattice, by time window: 440,000 real trips, two GPUFlowAggregation graphs, and a playback clock that sweeps the window through New Year’s Day and Friday morning.',
  contributors: ['GPUFlowAggregation'],
  datasets: [
    {id: 'poopdeck-nyc-taxi', role: '440,000 yellow-taxi origin-destination pairs with times'}
  ],
  initialView: {longitude: -73.97, latitude: 40.74, zoom: 10.9},

  options: [
    ...playbackOptions<NycTaxiTidesOptions>({
      ids: {play: 'play', time: 'time', speed: 'playSpeed', loop: 'loop'},
      time: {
        min: FIRST_HOUR,
        max: LAST_HOUR,
        step: 0.25,
        default: 32,
        label: 'Window start',
        format: formatTaxiTime,
        help: 'Start of the time window, in hours since midnight on Thursday 1 January 2015 (local time). Hour 32 is Friday 2 January, 08:00. The playback clock moves this slider.'
      },
      speed: {
        min: 0.25,
        max: 4,
        step: 0.25,
        default: 1,
        unit: 'h/s',
        label: 'Playback speed',
        help: 'Hours of taxi time that pass per real second. At 1 h/s the 37 hours take about 37 seconds.'
      },
      playing: false,
      loop: true
    }),
    {
      kind: 'slider',
      id: 'windowHours',
      label: 'Window length',
      group: 'Playback',
      apply: 'param',
      min: 0.25,
      max: 6,
      step: 0.25,
      default: 1,
      unit: 'h',
      help: 'Length of the time window. Departures are trips picked up in the window, arrivals trips dropped off in it. Both are the same four numbers of a parameter buffer.'
    },
    {
      kind: 'select',
      id: 'totals',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'net',
      help: 'Net balance (arrivals minus departures, diverging ramp), or the volume of one side (sequential ramp).',
      options: [
        {value: 'net', label: 'Net balance (arrivals - departures)'},
        {value: 'departures', label: 'Departures'},
        {value: 'arrivals', label: 'Arrivals'}
      ]
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Weight',
      group: 'Display',
      apply: 'param',
      default: 'trips',
      help: 'What each trip contributes: one trip, its fare in USD, or its passengers. Rewriting the weight buffer re-runs the same graphs.',
      options: [
        {value: 'trips', label: 'Trips'},
        {value: 'fare', label: 'Fare revenue (USD, no tips)'},
        {value: 'passengers', label: 'Passengers'}
      ]
    },
    {
      kind: 'select',
      id: 'scaleMode',
      label: 'Colour scale',
      group: 'Display',
      apply: 'param',
      default: 'fixed',
      help: 'Fixed keeps one scale while the clock plays, so hours can be compared; auto stretches the ramp to the busiest hexagon of each window.',
      options: [
        {value: 'fixed', label: 'Fixed range'},
        {value: 'auto', label: 'Auto (busiest hexagon)'}
      ]
    },
    {
      kind: 'slider',
      id: 'colorRange',
      label: 'Fixed range',
      group: 'Display',
      apply: 'param',
      min: 20,
      max: 800,
      step: 20,
      default: 200,
      unit: 'trips',
      disabledWhen: state => state.scaleMode === 'auto',
      help: 'Trips per hexagon at the ends of the ramp. For fare or passenger weights it scales with the mean weight of a trip.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.totals === 'net',
      help: 'Ramp of the departure and arrival volumes. Net balance always uses the diverging ramp.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneOpacity',
      label: 'Lattice opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it to read street names under the lattice.'
    },
    {
      kind: 'select',
      id: 'zones',
      label: 'Zones',
      group: 'Lattice',
      apply: 'compile',
      default: 'hexagon',
      help: 'A hexagon lattice or a square grid assigns each trip end to a zone. The zone kind is compile-time; the zone size is not.',
      options: [
        {value: 'hexagon', label: 'Hexagon lattice'},
        {value: 'grid', label: 'Square grid'}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneSize',
      label: 'Zone size',
      group: 'Lattice',
      apply: 'param',
      min: 400,
      max: 3000,
      step: 100,
      default: 800,
      unit: 'm',
      help: 'Hexagon radius or square cell size. The lattice capacity is fixed for 400 m; the radius and active lattice are per-frame values, so this never recompiles.'
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-zone trips',
      group: 'Lattice',
      apply: 'compile',
      default: true,
      help: 'Rejects trips whose pickup and dropoff fall in one zone. The net balance is unchanged (such trips cancel); the arcs and the pair count lose the local trips.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Summation order',
      group: 'Lattice',
      apply: 'compile',
      default: 'sorted',
      help: '`sorted` accumulates weights in a fixed tree, bitwise identical on every run. `atomic` adds with compare-exchange and is faster only when almost every row has its own pair.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    },
    {
      kind: 'slider',
      id: 'arcs',
      label: 'Arcs drawn',
      group: 'Arcs',
      apply: 'param',
      min: 0,
      max: 256,
      step: 4,
      default: 60,
      help: 'Heaviest departure flows of the window, drawn from the top-256 list the contributor writes. Zero hides them.'
    },
    {
      kind: 'slider',
      id: 'arcWidth',
      label: 'Widest arc',
      group: 'Arcs',
      apply: 'param',
      min: 1,
      max: 12,
      step: 0.5,
      default: 5,
      unit: 'px',
      help: 'Width of the heaviest arc; others scale with the square root of their weight.'
    },
    {
      kind: 'slider',
      id: 'arcOpacity',
      label: 'Arc opacity',
      group: 'Arcs',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.55,
      help: 'Lower it to read the lattice through the arcs.'
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Trips on the GPU',
      format: 'integer',
      help: 'Origin-destination pairs uploaded once.'
    },
    {id: 'clock', label: 'Clock'},
    {id: 'window', label: 'Window'},
    {id: 'zones', label: 'Lattice'},
    {
      id: 'departures',
      label: 'Departures in window',
      format: 'integer',
      help: 'Accepted trips picked up in the window (same-zone trips excluded when that option is on).'
    },
    {
      id: 'arrivals',
      label: 'Arrivals in window',
      format: 'integer',
      help: 'Accepted trips dropped off in the window. Differs from departures by the trips in flight over the window edges.'
    },
    {
      id: 'gain',
      label: 'Biggest gain',
      help: 'The zone with the largest positive net balance, with the position of its center.'
    },
    {id: 'loss', label: 'Biggest loss', help: 'The zone with the largest negative net balance.'},
    {id: 'activeZones', label: 'Active zones'},
    {id: 'flowRows', label: 'Flows listed', format: 'integer'},
    {
      id: 'pairs',
      label: 'Distinct pairs',
      format: 'integer',
      help: 'Distinct (origin zone, destination zone) pairs in the window.'
    },
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      help: 'Yes means more distinct pairs than the hash table holds (524,288): totals would be incomplete.'
    },
    {id: 'midtownChart', label: 'Net arrivals in Midtown by hour', kind: 'chart'}
  ],

  legends: state => {
    const unit =
      state.weight === 'trips' ? 'trips' : state.weight === 'fare' ? 'USD' : 'passengers';
    return [
      {
        kind: 'ramp',
        id: 'zones',
        title:
          state.totals === 'net'
            ? `Net arrivals per zone (${unit})`
            : `${state.totals === 'arrivals' ? 'Arrivals' : 'Departures'} per zone (${unit})`,
        ramp: state.totals === 'net' ? 'diverging' : state.ramp,
        extent: 'gpu',
        sqrtScale: state.totals !== 'net',
        ...(state.totals === 'net' ? {labels: ['more leave', 'more arrive'] as const} : {}),
        format: value =>
          state.totals === 'net' && value > 0 ? `+${compact(value)}` : compact(value)
      },
      {
        kind: 'categories',
        title: 'Arcs',
        entries: [
          {color: [255, 176, 64, 255], label: 'Pickup end'},
          {color: [64, 224, 255, 255], label: 'Dropoff end'}
        ]
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';

// departures gate on the pickup time, arrivals on the dropoff time: same trips, two graphs
const make = (gateTimes) => {
  const graph = new GPUCommandGraph(device);
  graph.add(new GPUFlowAggregation({
    zones: {kind: '${state.zones}', bounds, gridSize: capacity,   // capacity fixed for 400 m
            activeGridSize, ${state.zones === 'hexagon' ? 'radius, ' : ''}},
    origins, destinations,                         // float32x2 meters, uploaded once
    weights,                                       // trips, fare or passengers
    timeWindow: {timestamps: gateTimes, window},   // four numbers, rewritten every frame
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: 524288,
    output: {ids, count, overflow, totalCount},
    zoneOutWeights, zoneInWeights, zoneOutCounts, zoneInCounts,
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights
  }));
  return graph.compile();
};
const departures = make(pickupHours);
const arrivals = make(dropoffHours);

// every frame: move the window, run both, read the zone totals back once
window.write(getGPUTimeWindowParameterValues({start: hour, end: hour + ${state.windowHours}}));
departures.encode(commandEncoder, {parameters: undefined});
arrivals.encode(commandEncoder, {parameters: undefined});
// net[zone] = zoneInWeights[zone] - zoneOutWeights[zone]`,

  about: {
    what: '`GPUFlowAggregation` assigns every origin-destination row to a zone (here a hexagon lattice), keeps the rows whose timestamp is inside a time window, and writes per-zone departure and arrival totals plus a weight-ranked list of the heaviest zone-to-zone flows. Two copies of the graph, gated on the pickup time and on the dropoff time, give departures and arrivals for the same window.',
    why: 'Net flow shows where a city gains and loses people through the day, which is the input to staffing, curb space and transit decisions. Because the window is a four-number buffer, the clock can sweep the whole day and the answer keeps up.',
    howToRead:
      'Red hexagons took in more riders than they sent out in the window; blue ones sent out more; white is balance. The scale is fixed by default, so the same colour means the same net number across hours. Arcs run from orange (pickup) to cyan (dropoff) and are the heaviest departures. The line chart is the hourly net of one Midtown box.'
  },

  create: async ctx => (await import('./nyc-taxi-tides.compute')).createNycTaxiTides(ctx),

  story: storyFromMarkdown<NycTaxiTidesOptions>(narrative, {
    'the-question': {
      controls: ['time', 'totals'],
      readouts: ['window', 'departures', 'arrivals'],
      options: {time: 32, windowHours: 1, play: false, totals: 'net', weight: 'trips', arcs: 60},
      camera: {
        longitude: -73.97,
        latitude: 40.745,
        zoom: 11.2,
        pitch: 0,
        bearing: 0,
        transitionMs: 1400
      }
    },
    'morning-rush': {
      controls: ['windowHours', 'weight'],
      readouts: ['gain', 'loss'],
      options: {windowHours: 3, time: 31},
      camera: {longitude: -73.97, latitude: 40.76, zoom: 12.2, transitionMs: 1600},
      callout: {coordinate: [-73.979, 40.764], text: 'Midtown West: net +611 in the 08:00 hour'}
    },
    'play-the-day': {
      controls: ['play', 'playSpeed', 'loop'],
      readouts: ['clock', 'midtownChart'],
      options: {windowHours: 1, time: 1, play: true, playSpeed: 1},
      camera: {longitude: -73.97, latitude: 40.745, zoom: 11.2, transitionMs: 1600}
    },
    'two-clocks': {
      controls: ['totals', 'time'],
      readouts: ['departures', 'arrivals'],
      options: {play: false, time: 32, totals: 'departures'}
    },
    'hexagon-size': {
      controls: ['zoneSize', 'zones'],
      readouts: ['zones', 'activeZones', 'gain'],
      options: {totals: 'net', zoneSize: 1600},
      camera: {longitude: -73.97, latitude: 40.745, zoom: 11, transitionMs: 1400}
    },
    airports: {
      controls: ['arcs', 'totals'],
      readouts: ['gain', 'loss'],
      options: {time: 18, windowHours: 3, zoneSize: 800, totals: 'net', arcs: 120},
      camera: {longitude: -73.84, latitude: 40.7, zoom: 10.2, transitionMs: 1800}
    },
    limits: {
      controls: ['excludeSelf', 'sumOrder'],
      readouts: ['pairs', 'pairOverflow'],
      options: {time: 32, windowHours: 1, arcs: 60}
    }
  })
});
