// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {TRANSIT_HUB_NAMES} from './transit-data';
import type {TransitReachabilityOptions} from './transit-reachability.compute';
import {ground} from '../../cartography/grounds';
import {RAIL_ORIENTATION, RANDSTAD_RAIL_CREDIT} from './randstad-network-cartography';

const NETHERLANDS_VIEW = {longitude: 5.4, latitude: 52.15, zoom: 6.9};
const REACH_BAND_COLORS = [
  [8, 48, 107, 170],
  [43, 140, 190, 170],
  [173, 216, 191, 170]
] as const;

export default defineScene<TransitReachabilityOptions>({
  id: 'transit-reachability',
  title: 'How far can you get by train?',
  chapter: 'networks',
  order: 9,
  summary:
    'Click a station of the Dutch rail network and see everything reachable in 30, 60 and 90 minutes on the scheduled timetable, plus the fastest-route tree and a cost matrix between fourteen hub stations.',
  contributors: ['GPUNetworkReachability', 'GPUNetworkIsochrones', 'GPUNetworkCostMatrix'],
  datasets: [
    {id: 'gtfs-nl-rail-graph', role: 'rail graph with scheduled travel times'},
    {id: 'natural-earth', role: 'land, water and border context'}
  ],
  initialView: NETHERLANDS_VIEW,
  basemap: ground('paperSheet', {world: 'land', labels: 'above', labelPreset: 'orientation'}),
  furniture: {
    title: {
      subtitle: 'Rail reach · scheduled median travel time',
      chips: ['Best-case connections']
    },
    scaleBar: {units: 'metric', latitude: 52},
    credit: `${RANDSTAD_RAIL_CREDIT} · Made with Natural Earth`,
    caveat:
      'No wait or transfer time; perfect connections and planar last-mile buffers are model assumptions. Natural Earth masks water.'
  },
  annotations: RAIL_ORIENTATION,

  options: [
    {
      kind: 'select',
      id: 'origin',
      label: 'Start station',
      group: 'Search',
      apply: 'param',
      default: 'hub:Utrecht',
      help: 'The station the search starts from: one of fourteen hubs, or any station you click on the map (then this reads "Clicked station"). The origin is a one-word parameter write, so the search re-runs without a rebuild.',
      options: [
        ...TRANSIT_HUB_NAMES.map(name => {
          const label = name.replace(/ Centraal$/, '');
          return {value: `hub:${label}`, label};
        }),
        {value: 'clicked', label: 'Clicked station'}
      ]
    },
    {
      kind: 'select',
      id: 'trainTypes',
      label: 'Train types',
      group: 'Search',
      apply: 'param',
      default: 'all',
      help: 'Which trains you may ride. Edges of the other types get a negative weight, which GPUNetworkReachability treats as impassable; the weight buffer is rewritten, nothing recompiles.',
      options: [
        {value: 'all', label: 'All trains'},
        {value: 'fast', label: 'Intercity and express only'},
        {value: 'stopping', label: 'Stopping trains only (Sprinter, Stoptrein)'}
      ]
    },
    {
      kind: 'slider',
      id: 'dwellSeconds',
      label: 'Dwell per station',
      group: 'Search',
      apply: 'param',
      min: 0,
      max: 300,
      step: 15,
      default: 30,
      unit: 's',
      help: 'Seconds added at every station a journey passes. This is a dwell sensitivity only: it is not a transfer penalty, and the graph still models perfect transfers with no waiting.'
    },
    {
      kind: 'slider',
      id: 'costLimitMinutes',
      label: 'Search cut-off',
      group: 'Search',
      apply: 'param',
      min: 30,
      max: 240,
      step: 15,
      default: 150,
      unit: 'min',
      help: 'The search stops expanding stations whose cost is above this (the per-frame costLimit of GPUNetworkReachability). Stations beyond it stay unreached.'
    },
    {
      kind: 'select',
      id: 'localIterations',
      label: 'Hops per round',
      group: 'Search',
      apply: 'compile',
      default: '32',
      help: 'Compile-time localIterations of GPUNetworkReachability: how many hops one workgroup chains in a round. The costs are identical for every value; only the rounds needed change. Check the Solver readout.',
      options: [
        {value: '16', label: '16'},
        {value: '32', label: '32'},
        {value: '64', label: '64'}
      ]
    },
    {
      kind: 'select',
      id: 'lastMile',
      label: 'Last mile',
      group: 'Isochrones',
      apply: 'param',
      default: 'bike',
      help: 'How a traveller gets from the station to the door. Every station marks the pixels within the last-mile radius with its arrival cost plus the time to travel from the station; "stations only" marks just the station pixel. Walking is 1.34 m/s, cycling 4.2 m/s (15 km/h).',
      options: [
        {value: 'none', label: 'Stations only'},
        {value: 'walk', label: 'Walk (4.8 km/h)'},
        {value: 'bike', label: 'Cycle (15 km/h)'}
      ]
    },
    {
      kind: 'slider',
      id: 'lastMileMinutes',
      label: 'Last-mile time',
      group: 'Isochrones',
      apply: 'param',
      min: 0,
      max: 30,
      step: 5,
      default: 15,
      unit: 'min',
      disabledWhen: state => state.lastMile === 'none',
      help: 'Longest walk or ride from the station. The buffer radius is speed times this time, capped at 16 raster pixels (about 6 km at this resolution), as the Last-mile readout says.'
    },
    {
      kind: 'select',
      id: 'serviceHour',
      label: 'Service hour',
      group: 'Evidence',
      apply: 'param',
      default: '7',
      help: 'Station tooltips sum each outgoing edge’s actual hourly trips-per-hour component at this local hour; no morning average is substituted.',
      options: [
        {value: '5', label: '05:00'},
        {value: '7', label: '07:00'},
        {value: '9', label: '09:00'},
        {value: '12', label: '12:00'},
        {value: '17', label: '17:00'},
        {value: '21', label: '21:00'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBands',
      label: 'Show isochrone bands',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The reachable area in bands, from the triangles GPUNetworkIsochrones writes. The draw call reads its vertex count from the GPU.'
    },
    {
      kind: 'toggle',
      id: 'showTree',
      label: 'Show fastest routes',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The neutral shortest-path predecessor tree: one line from each station to the station you reach it from. It comes from the predecessors GPUNetworkReachability writes; the bands, not the tree, encode arrival time.'
    },
    {
      kind: 'toggle',
      id: 'showEdges',
      label: 'Show the rail network',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Every station-to-station edge of the graph. Edges your train-type choice forbids are faded.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Show stations',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Every station as a dot; hover one for its travel time from the start. The start station is the orange dot.'
    },
    {
      kind: 'toggle',
      id: 'showDistanceReference',
      label: 'Show 60 km geodesic reference',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'A dashed true geodesic ring around the selected origin is a comparison reference, not a rail-time contour.'
    }
  ],

  readouts: [
    {
      id: 'reachCurve',
      label: 'Stations reached by time',
      kind: 'chart',
      help: 'How many stations are within each travel time of the start. Vertical lines are the isochrone breaks. Steep stretches are the intercity hops that add many stations at once.'
    },
    {
      id: 'hubTimes',
      label: 'Time to the hubs (cost matrix)',
      kind: 'chart',
      help: 'Scheduled rail travel time between the start station and fourteen hub stations, read from the GPUNetworkCostMatrix row of each hub. The start station itself is highlighted (zero).'
    },
    {
      id: 'hubMatrix',
      label: 'Hub × hub cost matrix',
      kind: 'chart',
      help: 'Scheduled minutes for every hub pair. Blank matrix cells mean no modelled connection, never zero.'
    },
    {id: 'origin', label: 'Start station'},
    {id: 'reached', label: 'Stations reached'},
    {
      id: 'bandTable',
      label: 'Stations within each break',
      layout: 'block',
      help: 'Cumulative number of stations within each break of the isochrones, by rail travel time alone.'
    },
    {id: 'farthest', label: 'Farthest station', help: 'The reached station with the highest cost.'},
    {
      id: 'solver',
      label: 'Solver',
      help: 'Relaxation rounds GPUNetworkReachability needed, and whether it reached a fixpoint.'
    },
    {
      id: 'matrix',
      label: 'Cost matrix',
      help: 'Rows (hubs) and columns (stations) of the GPUNetworkCostMatrix result, and whether every batch converged.'
    },
    {
      id: 'hubMean',
      label: 'Between the hubs',
      help: 'Mean scheduled time between two different hub stations.'
    },
    {id: 'lastMile', label: 'Last-mile radius'},
    {id: 'stations', label: 'Network'},
    {id: 'raster', label: 'Isochrone raster'}
  ],

  legends: state => {
    return [
      ...(state.showBands
        ? [
            {
              kind: 'categories' as const,
              title: 'Time from the start station',
              entries: ['Up to 30 min', 'Up to 60 min', 'Up to 90 min'].map((label, index) => ({
                color: REACH_BAND_COLORS[index],
                label
              })),
              note:
                state.lastMile === 'none'
                  ? 'Up to the station: nearest band is strongest.'
                  : `Up to ${state.lastMileMinutes} min ${state.lastMile === 'bike' ? 'cycling' : 'walking'} from a station; water is masked.`
            }
          ]
        : []),
      ...(state.showTree
        ? [
            {
              kind: 'line' as const,
              title: 'Fastest-route tree',
              entries: [
                {
                  color: [30, 37, 52, 185] as const,
                  widthPixels: 1.5,
                  label: 'neutral predecessor tree'
                }
              ]
            }
          ]
        : [])
    ];
  },

  snippet: state => `import {
  GPUNetworkReachability, GPUNetworkIsochrones, GPUNetworkCostMatrix,
  getGPUNetworkIsochroneParameterValues
} from '@luma.gl/experimental/gpu-network';

// offsets / neighbors / weights: CSR of the rail graph, weights = scheduled seconds (+ dwell)
const graph = new GPUCommandGraph(device, {id: 'rail-reach'});
graph.add(new GPUNetworkReachability({
  offsets, neighbors, weights,
  sources: originParameter.importToGraph(graph),       // one word: the clicked station
  costLimit: costLimitParameter.importToGraph(graph),  // ${state.costLimitMinutes * 60} s
  maxIterations: 16, localIterations: ${state.localIterations},      // compile time
  costs, predecessors, converged
}));
graph.add(new GPUNetworkIsochrones({
  offsets: stationOnlyOffsets, neighbors: stationOnlyNeighbors, weights: zeros,   // sample stations, not the line between them
  nodePositions, costs, breaks: breaksParameter.importToGraph(graph),
  parameters: isochroneParameters.importToGraph(graph),
  raster: {width: 800, height, mode: 'min', maximumBufferPixels: 16, maximumSamplesPerEdge: 2,
           output: {triangles, triangleBands, count, overflow, vertexCount}}
}));

// per change: parameter writes
originParameter.write(Uint32Array.of(stationIndex));
breaksParameter.write(Float32Array.of(1800, 3600, 5400, 5400)); // fixed up-to 30/60/90 min bands
isochroneParameters.write(getGPUNetworkIsochroneParameterValues({
  breakCount: 3, extent, bufferRadius: ${state.lastMile === 'none' ? 0 : Math.round(({walk: 1.34, bike: 4.2} as const)[state.lastMile] * state.lastMileMinutes * 60)}, walkCostPerUnit: ${state.lastMile === 'none' ? 0 : (1 / ({walk: 1.34, bike: 4.2} as const)[state.lastMile]).toFixed(3)}
}));

// many-to-all travel times between hubs (one lane-expanded search per hub)
matrixGraph.add(new GPUNetworkCostMatrix({
  offsets, neighbors, weights, seedNodes: hubNodes, laneCount: hubNodes.length,
  costs: matrix                                   // hubs x stations, +Infinity when unreached
}));`,

  about: {
    what: '`GPUNetworkReachability` is a frontier-based single-source shortest-path search over a CSR graph: it keeps the cheapest cost and the predecessor of every station. `GPUNetworkIsochrones` turns node costs into banded polygons by splatting each station to a raster (cost plus last-mile travel) and contouring it. `GPUNetworkCostMatrix` runs one search per hub in lane-expanded batches and keeps the full matrix.',
    why: 'Accessibility by public transport is the question behind station location, housing and job-market studies: not "how far is it" but "how much of the country can I reach, and in how long". A scheduled graph answers it for any station and any train type in a fraction of a second, so you can compare stations, test a service change or read the whole matrix between cities.',
    howToRead:
      'Orange is the start. Coloured regions are places within up to 30, 60 and 90 minutes: rail time to a station plus a planar last mile. Neutral ink lines are predecessor links in the fastest-route tree. **Perfect connections are assumed**: the graph holds in-vehicle times only, so it models no wait or transfer time. Real journeys are slower.'
  },

  create: async ctx =>
    (await import('./transit-reachability.compute')).createTransitReachability(ctx),

  story: [
    {
      id: 'utrecht',
      headline: 'One hour from the middle',
      textAlternative: 'Nested rail-time bands extend from Utrecht across the Netherlands.',
      optionsMode: 'fresh',
      title: 'From Utrecht, what is within an hour of the train?',
      body: 'Utrecht is the default because the rail graph connects it broadly to the country. `GPUNetworkReachability` writes the cheapest scheduled in-vehicle cost to every station; `GPUNetworkIsochrones` turns those node costs into nested 30-, 60- and 90-minute bands. The table and curve supply the live counts—do not read the bands as a guarantee of a real journey.',
      camera: {...NETHERLANDS_VIEW, transitionMs: 1200},
      options: {origin: 'hub:Utrecht'},
      callout: {coordinate: [5.1101, 52.0894], text: 'Utrecht Centraal'},
      controls: ['origin'],
      readouts: ['bandTable', 'reachCurve']
    },
    {
      id: 'last-mile',
      headline: 'From station to door',
      textAlternative:
        'Rail-time bands widen around stations with a chosen walking or cycling last mile.',
      optionsMode: 'fresh',
      title: 'From station to door',
      body: "A station is a point; you live somewhere near one. **Last mile** gives each station a catchment: a pixel within reach of a station is marked with the station's arrival time plus the walk or ride from it, so the colour is the **door-to-door** time. With cycling for up to 15 minutes each station grows into a blob several kilometres across and strings of stations merge into corridors; with *Stations only* it collapses to dots. The Natural Earth mask prevents a planar buffer from becoming a claim over water.",
      options: {lastMile: 'bike', lastMileMinutes: 15},
      camera: {longitude: 5.1, latitude: 52.1, zoom: 8.2, transitionMs: 1400},
      controls: ['lastMile', 'lastMileMinutes'],
      readouts: ['lastMile', 'raster']
    },
    {
      id: 'network-edge',
      headline: 'Rail distance is not a circle',
      textAlternative: 'Groningen’s fastest-route tree differs from a simple 60 kilometre circle.',
      optionsMode: 'fresh',
      title: 'Rail distance is not a circle',
      body: '**Click any station** and the origin parameter changes while the compiled graph stays intact. An edge-city origin makes the distinction between straight distance and rail cost visible: the neutral tree follows tracks, and the live reach curve records when stations enter. Regional access is ordered by connections, not by radius.',
      options: {origin: 'hub:Groningen', lastMile: 'bike', showDistanceReference: true},
      camera: {longitude: 5.9, latitude: 52.7, zoom: 7.2, transitionMs: 1600},
      callout: {coordinate: [6.5646, 53.2113], text: 'Groningen'},
      controls: ['origin', 'showDistanceReference'],
      readouts: ['bandTable', 'farthest', 'reachCurve']
    },
    {
      id: 'connections',
      headline: 'Perfect connections are a best case',
      textAlternative:
        'Changing train types and dwell time changes the best-case rail network bands.',
      optionsMode: 'fresh',
      title: 'Perfect connections are a best case',
      body: 'Train-type filtering rewrites weights: forbidden edges become impassable without recompiling. Dwell sensitivity adds a cost at each stop, but it is not a transfer model. The graph has no waits, misses or platform changes, so the result is a deterministic best case; departure-window routing would answer a different question.',
      options: {origin: 'hub:Utrecht', trainTypes: 'fast', dwellSeconds: 30},
      camera: {...NETHERLANDS_VIEW, transitionMs: 1400},
      controls: ['trainTypes', 'dwellSeconds'],
      readouts: ['reached', 'bandTable']
    },
    {
      id: 'matrix',
      headline: 'Every hub in one matrix',
      textAlternative:
        'A live hub-by-hub rail cost matrix and ordered hub bars show scheduled travel time.',
      optionsMode: 'fresh',
      title: 'Every hub to every station',
      body: '`GPUNetworkCostMatrix` runs one lane-expanded search for each hub and retains a hub-by-station matrix. The matrix and bars read that result directly: an em dash means no modelled connection, never the origin’s zero. The model uses scheduled in-vehicle medians with perfect transfers, no waits and a planar last mile; it does not invent a transfer penalty.',
      options: {origin: 'hub:Amsterdam', trainTypes: 'all'},
      camera: {longitude: 5.4, latitude: 52.2, zoom: 7.1, transitionMs: 1400},
      controls: ['origin', 'localIterations', 'serviceHour'],
      readouts: ['hubMatrix', 'hubTimes', 'matrix', 'hubMean']
    }
  ]
});
