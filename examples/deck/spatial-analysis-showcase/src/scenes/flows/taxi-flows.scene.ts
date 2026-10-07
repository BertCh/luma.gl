// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './taxi-flows.md?raw';
import type {TaxiFlowOptions} from './taxi-flows.compute';

const compact = (value: number): string =>
  value >= 1e6
    ? `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)}M`
    : value >= 1e3
      ? `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}k`
      : value.toFixed(0);

const WEIGHT_TITLES: Record<string, string> = {
  trips: 'trips',
  fare: 'USD of fares',
  duration: 'ride hours',
  all: 'jobs',
  low: 'low-earning jobs',
  mid: 'mid-earning jobs',
  high: 'high-earning jobs'
};

export default defineScene<TaxiFlowOptions>({
  id: 'taxi-flows',
  title: 'Where do the taxis go?',
  chapter: 'flows',
  order: 1,
  summary:
    'Origin-destination aggregation on the GPU: 2023 Chicago taxi trips between community areas and census-tract commutes, ranked into top-K flows with a time window you can slide or play.',
  contributors: ['GPUFlowAggregation'],
  datasets: [
    {id: 'chicago-taxi-od', role: 'taxi flows by area, weekday and hour'},
    {id: 'chicago-community-areas', role: 'zone polygons'},
    {id: 'chicago-lodes-od', role: 'home-to-work flows between tracts'},
    {id: 'chicago-tracts', role: 'tract polygons'}
  ],
  initialView: {longitude: -87.72, latitude: 41.85, zoom: 10.1},

  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Flow dataset',
      group: 'Data',
      apply: 'compile',
      default: 'taxi',
      help: 'Taxi trips (2023, 77 community areas, with pickup hour) or home-to-work commutes (LODES 2021, 791 tracts, no timestamps). Switching rebuilds the graph for the new rows.',
      options: [
        {value: 'taxi', label: 'Taxi trips between community areas'},
        {value: 'commute', label: 'Commutes between census tracts'}
      ]
    },
    {
      kind: 'select',
      id: 'zones',
      label: 'Zones',
      group: 'Data',
      apply: 'compile',
      default: 'native',
      help: 'The dataset’s own areas (`ids`), or a hexagon or square lattice built from area centroids. The zone kind is compile-time; the lattice size is not.',
      options: [
        {value: 'native', label: 'Dataset areas (caller zone IDs)'},
        {value: 'hexagon', label: 'Hexagon lattice'},
        {value: 'grid', label: 'Square grid'}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneSize',
      label: 'Lattice size',
      group: 'Data',
      apply: 'param',
      min: 400,
      max: 9000,
      step: 100,
      default: 2500,
      unit: 'm',
      disabledWhen: state => state.zones === 'native',
      help: 'Hexagon radius or square cell size. Both the lattice dimensions and the radius are per-frame values under a compile-time capacity, so this never recompiles.'
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-zone flows',
      group: 'Data',
      apply: 'compile',
      default: true,
      help: 'Rejects records whose origin and destination zones are equal, so arcs show movement between places and zone totals leave out local trips.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Summation order',
      group: 'Data',
      apply: 'compile',
      default: 'sorted',
      help: '`sorted` accumulates weights in a fixed tree: bitwise identical on every run and device. `atomic` is faster only when nearly every record has its own pair, and its rounding depends on GPU scheduling.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    },
    {
      kind: 'button',
      id: 'compareSum',
      label: 'Compare summation orders',
      group: 'Data',
      help: 'Builds the other summation order, times both graphs outside the frame and reports the largest difference between their zone totals.'
    },
    {
      kind: 'select',
      id: 'taxiWeight',
      label: 'Weight (taxi)',
      group: 'Weight',
      apply: 'param',
      default: 'trips',
      disabledWhen: state => state.source !== 'taxi',
      help: 'What each record contributes to its flow. Rewriting the weight buffer re-runs the same graph.',
      options: [
        {value: 'trips', label: 'Trips'},
        {value: 'fare', label: 'Fare revenue (USD, no tips)'},
        {value: 'duration', label: 'Ride time (hours)'}
      ]
    },
    {
      kind: 'select',
      id: 'commuteWeight',
      label: 'Weight (commute)',
      group: 'Weight',
      apply: 'param',
      default: 'all',
      disabledWhen: state => state.source !== 'commute',
      help: 'Jobs by monthly earnings: low up to $1,250, mid $1,251 to $3,333, high above $3,333.',
      options: [
        {value: 'all', label: 'All jobs'},
        {value: 'low', label: 'Low earnings'},
        {value: 'mid', label: 'Middle earnings'},
        {value: 'high', label: 'High earnings'}
      ]
    },
    ...playbackOptions<TaxiFlowOptions>({
      ids: {play: 'play', time: 'hourStart', speed: 'playSpeed', loop: 'loop'},
      group: 'Time window',
      playing: false,
      disabledWhen: state => state.source !== 'taxi',
      time: {
        min: 0,
        max: 23.75,
        step: 0.25,
        default: 0,
        label: 'Window start',
        format: formatPlaybackTime.hour,
        help: 'First pickup hour of the window (local time). The window is a 4-number parameter buffer: moving it never recompiles. Play slides the start across the day.'
      },
      speed: {
        min: 0.5,
        max: 6,
        step: 0.5,
        default: 2,
        unit: 'h/s',
        label: 'Play speed',
        help: 'Simulated hours per second. 2 h/s sweeps the whole day in 12 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'hourLength',
      label: 'Window length',
      group: 'Time window',
      apply: 'param',
      min: 1,
      max: 24,
      step: 1,
      default: 24,
      unit: 'h',
      disabledWhen: state => state.source !== 'taxi',
      help: 'Hours covered by the window; 24 keeps the whole day. The window stops at midnight.'
    },
    {
      kind: 'select',
      id: 'dayType',
      label: 'Day type',
      group: 'Time window',
      apply: 'param',
      default: 'all',
      disabledWhen: state => state.source !== 'taxi',
      help: 'Weekdays (Mon to Fri), weekends (Sat and Sun) or both. A per-record mask buffer, rewritten when you change it.',
      options: [
        {value: 'all', label: 'All days'},
        {value: 'weekday', label: 'Weekdays'},
        {value: 'weekend', label: 'Weekends'}
      ]
    },
    {
      kind: 'select',
      id: 'totals',
      label: 'Zone shading',
      group: 'Display',
      apply: 'param',
      default: 'departures',
      help: 'Per-zone totals written by the contributor: weight leaving the zone, arriving, or the net balance (arrivals minus departures, a diverging scale).',
      options: [
        {value: 'departures', label: 'Departures (origin totals)'},
        {value: 'arrivals', label: 'Arrivals (destination totals)'},
        {value: 'net', label: 'Net balance (arrivals − departures)'}
      ]
    },
    {
      kind: 'slider',
      id: 'arcs',
      label: 'Flows drawn',
      group: 'Display',
      apply: 'param',
      min: 5,
      max: 512,
      step: 5,
      default: 120,
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
      default: 10,
      unit: 'px',
      help: 'Pixel width of the heaviest flow. Others scale with the square root of their weight.'
    },
    {
      kind: 'slider',
      id: 'arcOpacity',
      label: 'Arc opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Lower it when many arcs overlap.'
    },
    {
      kind: 'toggle',
      id: 'showZones',
      label: 'Shade zones',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Colour each zone by its total.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Area outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the boundaries of the dataset areas, also under a lattice.'
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
        {value: 'inferno', label: 'Inferno'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneOpacity',
      label: 'Zone opacity',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.6,
      help: 'Opacity of the zone shading.'
    }
  ],

  readouts: [
    {
      id: 'hourlyChart',
      label: 'The day in pickups',
      kind: 'chart',
      help: 'Share of the day total in each pickup hour, weekdays against weekends, for the weight on the map. The rules mark the time window. Taxi data only.'
    },
    {
      id: 'topFlowsChart',
      label: 'The ten largest flows',
      kind: 'chart',
      help: 'Weight of the first ten flows of the ranked list inside the window.'
    },
    {
      id: 'concentrationChart',
      label: 'How concentrated are the flows?',
      kind: 'chart',
      help: 'Cumulative share of the total weight carried by the largest flows. The rule is the number of arcs drawn. Steep means a few pairs dominate.'
    },
    {
      id: 'volume',
      label: 'Flow in window',
      help: 'Sum of the weight of every accepted record (the departure totals of all zones).'
    },
    {
      id: 'records',
      label: 'Records accepted',
      format: 'integer',
      help: 'Source rows (origin, destination, day type, hour) that passed the window, mask and zone checks.'
    },
    {
      id: 'pairs',
      label: 'Distinct zone pairs',
      format: 'integer',
      help: 'Pairs retained in the GPU hash table.'
    },
    {id: 'drawn', label: 'Arcs drawn', format: 'integer'},
    {
      id: 'share',
      label: 'Share carried by drawn arcs',
      format: 'percent',
      help: 'Weight of the drawn flows divided by the total weight in the window.'
    },
    {
      id: 'truncated',
      label: 'Top-512 list truncated',
      help: 'The ranked list holds 512 flows. Zone totals stay exact even when pairs are not listed.'
    },
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      help: 'Set when distinct pairs exceeded the hash table capacity.'
    },
    {id: 'zones', label: 'Zones'},
    {id: 'activeZones', label: 'Zones with traffic'},
    {id: 'window', label: 'Time window'},
    {id: 'busiestOrigin', label: 'Busiest origin'},
    {id: 'busiestDestination', label: 'Busiest destination'},
    {id: 'flow1', label: 'Largest flow'},
    {id: 'flow2', label: '2nd largest'},
    {id: 'flow3', label: '3rd largest'},
    {
      id: 'sumTiming',
      label: 'Sum order timing',
      help: 'GPU time per run of each summation order, from the compare button.'
    },
    {
      id: 'sumDifference',
      label: 'Sum order difference',
      help: 'Largest difference between the zone totals of the two summation orders.'
    }
  ],

  legends: state => {
    const weight = state.source === 'taxi' ? state.taxiWeight : state.commuteWeight;
    const net = state.totals === 'net';
    const title = net
      ? `Net balance, ${WEIGHT_TITLES[weight]}`
      : `${state.totals === 'arrivals' ? 'Arrivals' : 'Departures'}, ${WEIGHT_TITLES[weight]}`;
    return [
      {
        kind: 'ramp',
        id: 'zones',
        title,
        ramp: net ? 'diverging' : state.ramp,
        extent: 'gpu',
        sqrtScale: !net,
        labels: net ? ['sends more out', 'receives more'] : undefined,
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
    ];
  },

  snippet: state => `import {GPUCommandGraph, DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'flows'});
graph.add(
  new GPUFlowAggregation({
    zones: ${
      state.zones === 'native'
        ? "{kind: 'ids', zoneCount: 77},\n    originZoneIds, destinationZoneIds,   // uint32 per record"
        : `{
      kind: '${state.zones === 'grid' ? 'grid' : 'hexagon'}',
      bounds: bounds.importToGraph(graph),             // per-frame [minX, minY, maxX, maxY]
      gridSize: capacity,                              // compile-time upper bound
      activeGridSize: activeGrid.importToGraph(graph), // per-frame [columns, rows]${
        state.zones === 'hexagon' ? '\n      radius: radius.importToGraph(graph),' : ''
      }
    },
    origins, destinations,               // float32x2 per record`
    }
    weights,                             // float32 per record, rewritten for a new weight
    mask,                                // uint32 per record (weekday / weekend)${
      state.source === 'taxi'
        ? `
    timeWindow: {timestamps: hours, window: windowParams.importToGraph(graph)},`
        : ''
    }
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: 16384,
    output: {ids, count, overflow, totalCount},         // top-K pairs, heaviest first
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights,
    zoneOutWeights, zoneInWeights,                      // per-zone totals
    drawInstanceCount: graph.importGPUData('arcs', drawCommands.getInstanceCountData(0))
  })
);
const compiled = graph.compile();   // once
// every frame: write parameter buffers, then
windowParams.write(getGPUTimeWindowParameterValues({start: 17, end: 20}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUFlowAggregation` assigns every origin-destination record to an origin zone and a destination zone, sums weights per zone pair in a GPU hash table, ranks the pairs, and writes a top-K list plus per-zone departure and arrival totals.',
    why: 'Raw trip tables are too large to read and too dense to draw. A ranked list of the heaviest flows, with zone totals alongside, answers “where does movement concentrate, and when?” and feeds an arc layer without any CPU work.',
    howToRead:
      'Arcs run from orange (origin) to cyan (destination); a wider arc is a heavier flow. The zone shading comes from the same pass: departures, arrivals, or net balance. The time window, the day type and the weight all re-run the same compiled graph.'
  },

  create: async ctx => (await import('./taxi-flows.compute')).createTaxiFlows(ctx),

  story: storyFromMarkdown<TaxiFlowOptions>(narrative, {
    'the-question': {
      controls: ['source', 'totals', 'arcs'],
      readouts: ['volume'],
      camera: {longitude: -87.72, latitude: 41.85, zoom: 10.1, transitionMs: 1200},
      options: {source: 'taxi', arcs: 120, totals: 'departures'},
      callout: {coordinate: [-87.632, 41.884], text: 'The Loop'},
      highlight: {readout: 'volume'}
    },
    'top-flows': {
      controls: ['arcs', 'excludeSelf'],
      readouts: ['share', 'concentrationChart', 'topFlowsChart', 'truncated'],
      camera: {longitude: -87.72, latitude: 41.9, zoom: 10.5},
      options: {arcs: 25, arcWidth: 14},
      callout: {coordinate: [-87.905, 41.978], text: "O'Hare airport"},
      highlight: {readout: 'share'}
    },
    'time-window': {
      controls: ['hourStart', 'hourLength', 'dayType', 'play', 'totals'],
      readouts: ['window', 'hourlyChart'],
      options: {
        arcs: 60,
        hourStart: 17,
        hourLength: 3,
        dayType: 'weekday',
        totals: 'net',
        play: false
      },
      highlight: {readout: 'window'}
    },
    'zone-size': {
      controls: ['zones', 'zoneSize'],
      readouts: ['zones', 'activeZones'],
      camera: {longitude: -87.7, latitude: 41.85, zoom: 10.3},
      options: {
        zones: 'hexagon',
        zoneSize: 2500,
        totals: 'departures',
        arcs: 80,
        hourStart: 0,
        hourLength: 24,
        dayType: 'all'
      },
      highlight: {readout: 'zones'}
    },
    commute: {
      controls: ['source', 'arcs', 'totals'],
      readouts: ['share', 'concentrationChart'],
      camera: {longitude: -87.68, latitude: 41.87, zoom: 10.6},
      options: {
        source: 'commute',
        zones: 'native',
        totals: 'arrivals',
        arcs: 150,
        arcWidth: 10,
        ramp: 'viridis'
      },
      callout: {coordinate: [-87.633, 41.879], text: 'Loop: 315k jobs'}
    },
    earnings: {
      controls: ['commuteWeight'],
      readouts: ['share', 'concentrationChart'],
      options: {commuteWeight: 'low'},
      highlight: {readout: 'share'}
    },
    limits: {
      controls: ['taxiWeight', 'excludeSelf', 'sumOrder', 'compareSum'],
      readouts: ['sumTiming', 'sumDifference'],
      camera: {longitude: -87.72, latitude: 41.85, zoom: 10.1, transitionMs: 1400},
      options: {source: 'taxi', arcs: 120, totals: 'departures', play: false}
    }
  })
});
