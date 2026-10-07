// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-tides.md?raw';
import type {BixiTidesOptions} from './bixi-tides.compute';

const compact = (value: number): string => {
  const magnitude = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  return magnitude >= 1e3
    ? `${sign}${(magnitude / 1e3).toFixed(magnitude >= 1e4 ? 0 : 1)}k`
    : `${sign}${magnitude.toFixed(0)}`;
};

export default defineScene<BixiTidesOptions>({
  id: 'bixi-tides',
  title: 'The morning tide of BIXI bikes',
  chapter: 'flows',
  order: 11,
  summary:
    'GPUFlowAggregation on 1.93 million Montreal bike-share rides: net flow per station for any hour of the day, played through the day, with a sparkline for the station you click.',
  contributors: ['GPUFlowAggregation'],
  datasets: [{id: 'bixi-flows', role: 'station flows by weekday, weekend and hour'}],
  initialView: {longitude: -73.58, latitude: 45.525, zoom: 11.4},

  options: [
    ...playbackOptions<BixiTidesOptions>({
      ids: {play: 'play', time: 'hour', speed: 'speed', loop: 'loop'},
      group: 'Time window',
      playing: false,
      time: {
        min: 0,
        max: 23.75,
        step: 0.25,
        default: 8,
        label: 'Window start',
        format: formatPlaybackTime.hour,
        help: 'First start hour of the window, Montreal local time. The window is a four-number parameter buffer: moving it never recompiles. Play slides the start across the day.'
      },
      speed: {
        min: 0.5,
        max: 6,
        step: 0.5,
        default: 2,
        unit: 'h/s',
        label: 'Play speed',
        help: 'Hours of the day per real second. 2 h/s sweeps the whole day in 12 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'hourLength',
      label: 'Window length',
      group: 'Time window',
      apply: 'param',
      min: 0.25,
      max: 24,
      step: 0.25,
      default: 1,
      unit: 'h',
      help: 'Hours covered by the window; 24 keeps the whole day. The window stops at midnight.'
    },
    {
      kind: 'select',
      id: 'dayType',
      label: 'Day type',
      group: 'Time window',
      apply: 'param',
      default: 'weekday',
      help: 'Weekdays (Mon to Fri), weekends (Sat and Sun) or both. A per-record mask buffer, rewritten when you change it.',
      options: [
        {value: 'weekday', label: 'Weekdays'},
        {value: 'weekend', label: 'Weekends'},
        {value: 'all', label: 'All days'}
      ]
    },
    {
      kind: 'select',
      id: 'totals',
      label: 'Station shading',
      group: 'Display',
      apply: 'param',
      default: 'net',
      help: 'Per-station totals written by the contributor: net balance (arrivals minus departures, a diverging scale), rides leaving, or rides arriving.',
      options: [
        {value: 'net', label: 'Net balance (arrivals - departures)'},
        {value: 'departures', label: 'Departures'},
        {value: 'arrivals', label: 'Arrivals'}
      ]
    },
    {
      kind: 'slider',
      id: 'minBalance',
      label: 'Hide balances below',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 600,
      step: 10,
      default: 0,
      unit: 'rides',
      disabledWhen: state => state.totals !== 'net',
      help: 'Hides stations whose absolute net balance in the window (rides over August) is smaller than this, leaving only the big drains and sinks.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Stations',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the stations coloured by the shading.'
    },
    {
      kind: 'slider',
      id: 'stationSize',
      label: 'Station size',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 12,
      step: 0.5,
      default: 5,
      unit: 'px',
      help: 'Radius of the station dots.'
    },
    {
      kind: 'slider',
      id: 'arcs',
      label: 'Flows drawn',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 512,
      step: 5,
      default: 40,
      help: 'How many of the top-512 ranked flows are drawn, heaviest first. The list is always complete up to 512; this only limits drawing.'
    },
    {
      kind: 'slider',
      id: 'arcWidth',
      label: 'Widest arc',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 24,
      step: 1,
      default: 8,
      unit: 'px',
      help: 'Pixel width of the heaviest flow. Others scale with the square root of their weight.'
    },
    {
      kind: 'slider',
      id: 'arcOpacity',
      label: 'Arc opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.7,
      help: 'Lower it when many arcs overlap downtown.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.totals === 'net',
      help: 'Ramp for departures and arrivals. Net balance always uses the diverging ramp.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-station rides',
      group: 'Aggregation',
      apply: 'compile',
      default: true,
      help: 'Rejects records whose origin and destination are the same station. The trip table here already has none, so this changes nothing but the compiled graph.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Summation order',
      group: 'Aggregation',
      apply: 'compile',
      default: 'sorted',
      help: '`sorted` accumulates weights in a fixed tree: bitwise identical on every run and device. `atomic` is faster only when nearly every record has its own pair, and its rounding depends on GPU scheduling.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    }
  ],

  readouts: [
    {
      id: 'cityChart',
      label: 'Rides per hour of the day',
      kind: 'chart',
      help: 'Average rides starting in each hour, weekdays against weekends, over all of Montreal. The rules mark the time window.'
    },
    {
      id: 'stationChart',
      label: 'Net balance of the clicked station',
      kind: 'chart',
      help: 'Arrivals minus departures per hour of an average day, for the day type selected. The dot is the start of the window.'
    },
    {id: 'station', label: 'Clicked station', layout: 'block'},
    {
      id: 'extremesChart',
      label: 'Biggest drains and sinks',
      kind: 'chart',
      help: 'The five stations with the most negative and most positive net balance in the window, in rides over August.'
    },
    {id: 'window', label: 'Time window'},
    {
      id: 'volume',
      label: 'Rides in window',
      help: 'Sum of the weight of every accepted record: rides leaving all stations.'
    },
    {id: 'topDrain', label: 'Biggest drain'},
    {id: 'topFill', label: 'Biggest sink'},
    {id: 'busiest', label: 'Busiest origin'},
    {
      id: 'records',
      label: 'Distinct station pairs',
      format: 'integer',
      help: 'Pairs retained in the GPU hash table in this window, including the folded "other stations" zone.'
    },
    {
      id: 'share',
      label: 'Share carried by drawn arcs',
      format: 'percent',
      help: 'Weight of the drawn flows divided by the total weight in the window.'
    },
    {id: 'flow1', label: 'Largest flow'},
    {id: 'flow2', label: '2nd largest'},
    {id: 'flow3', label: '3rd largest'},
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      help: 'Set when distinct pairs exceeded the hash table capacity.'
    }
  ],

  legends: state => [
    {
      kind: 'ramp',
      id: 'stations',
      title:
        state.totals === 'net'
          ? 'Net balance (rides)'
          : state.totals === 'arrivals'
            ? 'Arrivals (rides)'
            : 'Departures (rides)',
      ramp: state.totals === 'net' ? 'diverging' : state.ramp,
      extent: 'gpu',
      sqrtScale: state.totals !== 'net',
      labels: state.totals === 'net' ? ['loses bikes', 'gains bikes'] : undefined,
      format: compact
    },
    {
      kind: 'categories',
      title: 'Flow arcs',
      entries: [
        {color: [255, 176, 64, 255], label: 'Origin end'},
        {color: [64, 224, 255, 255], label: 'Destination end'}
      ],
      note: 'Width follows the square root of the flow weight; the arc bends right of travel.'
    }
  ],

  snippet: state => `import {GPUCommandGraph, DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'tides'});
graph.add(
  new GPUFlowAggregation({
    zones: {kind: 'ids', zoneCount: 906},      // 905 stations + one zone for rare pairs
    originZoneIds, destinationZoneIds,         // uint32 per record
    weights,                                   // rides per record
    mask,                                      // uint32 per record: weekday / weekend
    timeWindow: {timestamps: hours, window: windowParams.importToGraph(graph)},
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: 131072,
    output: {ids, count, overflow, totalCount},   // top-512 pairs, heaviest first
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights,
    zoneOutWeights, zoneInWeights,                // per-station totals
    drawInstanceCount: graph.importGPUData('arcs', drawCommands.getInstanceCountData(0))
  })
);
const compiled = graph.compile();                // once
// every frame: write the window, then net = zoneInWeights - zoneOutWeights per station
windowParams.write(getGPUTimeWindowParameterValues({start: ${state.hour}, end: ${Math.min(24, state.hour + state.hourLength)}}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUFlowAggregation` assigns every origin-destination record to an origin zone and a destination zone, sums weights per zone pair in a GPU hash table, ranks the pairs, and writes a top-K list plus per-zone departure and arrival totals. Here the zones are BIXI stations.',
    why: 'A bike-share system is only useful if there is a bike where you start and a free dock where you end. The net balance per station and hour shows where the riders themselves drain and fill the network, which is where an operator has to move bikes.',
    howToRead:
      'Blue stations send out more rides than they receive in the window, red ones receive more. Arcs run from orange (origin) to cyan (destination); a wider arc is a heavier flow. The charts show the day in rides and, for a station you click, its own balance hour by hour.'
  },

  create: async ctx => (await import('./bixi-tides.compute')).createBixiTides(ctx),

  story: storyFromMarkdown<BixiTidesOptions>(narrative, {
    'the-question': {
      controls: ['hour', 'dayType', 'totals'],
      readouts: ['window', 'topDrain', 'topFill'],
      camera: {longitude: -73.58, latitude: 45.525, zoom: 11.4, transitionMs: 1400},
      options: {
        hour: 8,
        hourLength: 1,
        dayType: 'weekday',
        totals: 'net',
        play: false,
        arcs: 40,
        minBalance: 0
      },
      highlight: {readout: 'topDrain'}
    },
    window: {
      controls: ['hour', 'hourLength', 'dayType', 'arcs'],
      readouts: ['volume', 'share', 'flow1', 'records'],
      camera: {longitude: -73.58, latitude: 45.52, zoom: 11.8},
      options: {hour: 8, hourLength: 1, arcs: 60, play: false},
      highlight: {readout: 'share'}
    },
    play: {
      controls: ['play', 'speed', 'loop', 'dayType'],
      readouts: ['window', 'cityChart'],
      options: {hourLength: 1, arcs: 40, play: true, speed: 2, hour: 5},
      highlight: {readout: 'window'}
    },
    station: {
      controls: ['hour', 'dayType', 'minBalance'],
      readouts: ['station', 'stationChart', 'extremesChart'],
      camera: {longitude: -73.58, latitude: 45.525, zoom: 11.6, transitionMs: 1200},
      options: {hour: 8, hourLength: 1, play: false, minBalance: 120, arcs: 0, dayType: 'weekday'},
      highlight: {readout: 'extremesChart'}
    },
    limits: {
      controls: ['dayType', 'totals', 'sumOrder'],
      readouts: ['cityChart', 'topDrain'],
      options: {hour: 14, hourLength: 1, dayType: 'weekend', play: false, minBalance: 0, arcs: 40},
      camera: {longitude: -73.58, latitude: 45.525, zoom: 11.2, transitionMs: 1200}
    }
  })
});
