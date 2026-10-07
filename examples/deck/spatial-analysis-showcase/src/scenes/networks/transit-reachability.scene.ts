// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {getBandColor} from './b9-shared';
import {TRANSIT_HUB_NAMES} from './transit-data';
import type {TransitReachabilityOptions} from './transit-reachability.compute';

const NETHERLANDS_VIEW = {longitude: 5.4, latitude: 52.15, zoom: 6.9};
const BAND_ALPHA = 170;

export default defineScene<TransitReachabilityOptions>({
  id: 'transit-reachability',
  title: 'How far can you get by train?',
  chapter: 'networks',
  order: 22,
  summary:
    'Click a station of the Dutch rail network and see everything reachable in 30, 60 and 90 minutes on the scheduled timetable, plus the fastest-route tree and a cost matrix between fourteen hub stations.',
  contributors: ['GPUNetworkReachability', 'GPUNetworkIsochrones', 'GPUNetworkCostMatrix'],
  datasets: [{id: 'gtfs-nl-rail-graph', role: 'rail graph with scheduled travel times'}],
  initialView: NETHERLANDS_VIEW,

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
      help: 'Seconds added at every station a journey passes. A scheduled edge counts arrival at the next stop minus departure from this one, so the 30 to 60 seconds a train waits at each intermediate stop are missing; this restores them. Raise it to approximate the friction of transfers, which the graph does not model.'
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
      id: 'bandMinutes',
      label: 'Band width',
      group: 'Isochrones',
      apply: 'param',
      default: '30',
      help: 'Width of one isochrone band; the breaks are multiples of it (30 gives 30, 60, 90 minutes). An isochrone break is a per-frame parameter.',
      options: [
        {value: '15', label: '15 minutes'},
        {value: '20', label: '20 minutes'},
        {value: '30', label: '30 minutes'}
      ]
    },
    {
      kind: 'slider',
      id: 'bandCount',
      label: 'Number of bands',
      group: 'Isochrones',
      apply: 'param',
      min: 2,
      max: 4,
      step: 1,
      default: 3,
      help: 'How many bands are drawn (at most four breaks). Space beyond the last break is left empty.'
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
      help: 'The shortest-path tree: one line from each station to the station you reach it from, coloured by arrival time. It comes from the predecessors GPUNetworkReachability writes.'
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
      kind: 'select',
      id: 'ramp',
      label: 'Band color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp the bands are drawn with, near to far. All four are perceptually uniform.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
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
    const minutes = Number(state.bandMinutes);
    return [
      ...(state.showBands
        ? [
            {
              kind: 'categories' as const,
              title: 'Time from the start station',
              entries: Array.from({length: state.bandCount}, (_, band) => ({
                color: getBandColor(state.ramp, band, state.bandCount, BAND_ALPHA),
                label: `${band * minutes} to ${(band + 1) * minutes} min`
              })),
              note:
                state.lastMile === 'none'
                  ? 'Rail time to the station only.'
                  : `Rail time plus up to ${state.lastMileMinutes} min ${state.lastMile === 'bike' ? 'cycling' : 'walking'} from the station.`
            }
          ]
        : []),
      ...(state.showTree
        ? [
            {
              kind: 'ramp' as const,
              title: 'Fastest route: arrival time',
              ramp: 'inferno' as const,
              extent: [0, minutes * state.bandCount] as const,
              unit: 'min',
              format: (value: number) => value.toFixed(0)
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
breaksParameter.write(Float32Array.of(${[1, 2, 3, 4].map(band => Math.min(band, state.bandCount) * Number(state.bandMinutes) * 60).join(', ')}));
isochroneParameters.write(getGPUNetworkIsochroneParameterValues({
  breakCount: ${state.bandCount}, extent, bufferRadius: ${state.lastMile === 'none' ? 0 : Math.round(({walk: 1.34, bike: 4.2} as const)[state.lastMile] * state.lastMileMinutes * 60)}, walkCostPerUnit: ${state.lastMile === 'none' ? 0 : (1 / ({walk: 1.34, bike: 4.2} as const)[state.lastMile]).toFixed(3)}
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
      'Orange is the start. Coloured regions are the places within 30, 60 and 90 minutes (see the legend): rail time to a station plus the last mile on foot or bicycle, so each station seeds a blob and strings of stations make corridors. The fine lines are the fastest route to each station, coloured by arrival time. **Perfect connections are assumed**: the graph holds in-vehicle times only, so a transfer costs nothing and waiting costs nothing apart from the dwell per station you choose. Real journeys are slower.'
  },

  create: async ctx =>
    (await import('./transit-reachability.compute')).createTransitReachability(ctx),

  story: [
    {
      id: 'the-question',
      title: 'From Utrecht, what is within an hour of the train?',
      body: 'Utrecht Centraal is the middle of the Dutch rail network, which is why it is the default start. The map shows the **scheduled** rail graph of a Friday in October 2026: 485 stations and 1,411 station-to-station edges, each with the median in-vehicle time of the timetable.\n\n**`GPUNetworkReachability`** runs one shortest-path search from the orange start station and writes the cheapest travel time to every station. **`GPUNetworkIsochrones`** bands those times: from Utrecht you reach **70 stations in 30 minutes, 228 in 60 and 313 in 90** (the table below), counting rail time alone. The thin coloured lines are the fastest route to each station; the chart counts stations by travel time.',
      camera: {...NETHERLANDS_VIEW, transitionMs: 1200},
      options: {origin: 'hub:Utrecht'},
      callout: {coordinate: [5.1101, 52.0894], text: 'Utrecht Centraal'},
      controls: ['origin'],
      readouts: ['bandTable', 'reachCurve']
    },
    {
      id: 'last-mile',
      title: 'From station to door',
      body: "A station is a point; you live somewhere near one. **Last mile** gives each station a catchment: a pixel within reach of a station is marked with the station's arrival time plus the walk or ride from it, so the colour is the **door-to-door** time and the bands spread out from every station. With cycling for up to 15 minutes each station grows into a blob several kilometres across and strings of stations merge into corridors; with *Stations only* it collapses to dots.\n\nTry **Last mile** *Walk*, then slide **Last-mile time** below, and change **Band width** to 15 minutes for a finer picture. The radius is capped at 16 raster pixels (the Last-mile readout says when the cap bites).",
      options: {lastMile: 'bike', lastMileMinutes: 15, bandMinutes: '30'},
      camera: {longitude: 5.1, latitude: 52.1, zoom: 8.2, transitionMs: 1400},
      controls: ['lastMile', 'lastMileMinutes', 'bandMinutes'],
      readouts: ['lastMile', 'raster']
    },
    {
      id: 'click',
      title: 'Start somewhere else',
      body: '**Click any station** on the map (or choose one under **Start station** below) and the search re-runs from there: a single word of the origin parameter changes, the compiled graph stays the same. From **Groningen** in the north-east the same hour reaches only **55 stations** and 91 in 90 minutes, a quarter of what Utrecht reaches; the periphery pays for being at the end of the network.\n\nCompare the **Farthest station** and the chart with Utrecht: the curve rises much more slowly, because intercity hops add few stations until the line meets the Randstad.',
      options: {origin: 'hub:Groningen', lastMile: 'bike'},
      camera: {longitude: 5.9, latitude: 52.7, zoom: 7.2, transitionMs: 1600},
      callout: {coordinate: [6.5646, 53.2113], text: 'Groningen'},
      controls: ['origin'],
      readouts: ['bandTable', 'farthest', 'reachCurve']
    },
    {
      id: 'train-types',
      title: 'What the slow trains are for',
      body: 'Switch **Train types** to *Intercity and express only* with **Start station** Utrecht: the search can now use only the fast network. Edges of the other types get a negative weight, which `GPUNetworkReachability` treats as impassable, so nothing recompiles. Only **185 of the 485 stations** can be reached at all, and the 60-minute region shrinks to **63 stations**. With *Stopping trains only* you reach 425 stations but only 132 within an hour.\n\nThe fast trains give the far reach, the slow ones the access: a station of the network is mostly a sprinter stop. Raise **Dwell per station** to add friction at every stop and watch the bands shrink.',
      options: {origin: 'hub:Utrecht', trainTypes: 'fast', dwellSeconds: 30},
      camera: {...NETHERLANDS_VIEW, transitionMs: 1400},
      controls: ['trainTypes', 'dwellSeconds'],
      readouts: ['reached', 'bandTable']
    },
    {
      id: 'cost-matrix',
      title: 'Every hub to every station',
      body: '**`GPUNetworkCostMatrix`** runs one search for each of fourteen hub stations in lane-expanded batches that share the same graph, and keeps the whole matrix: 14 rows by 485 columns. The bar chart reads one column of it: the scheduled travel time between the start station and each hub. From Amsterdam, Utrecht is 27 minutes away, Rotterdam about 40 and Groningen well under two hours.\n\nChange **Start station** below and the bars update from the matrix on the CPU, with no new search; change **Train types** and the matrix is recomputed. **Between the hubs** is the mean of all the hub-to-hub times.',
      options: {origin: 'hub:Amsterdam', trainTypes: 'all'},
      camera: {longitude: 5.4, latitude: 52.2, zoom: 7.1, transitionMs: 1400},
      controls: ['origin'],
      readouts: ['hubTimes', 'hubMean', 'matrix']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'The graph is **scheduled in-vehicle time**: it holds no transfer times and no waiting, so connections are perfect and every journey is a best case; real trips are slower, and a rider who misses a connection on a 30-minute interval loses half an hour. Edge times are medians over Friday 9 October 2026 (the feed has that day), while the trips of the other two scenes are from 3 July; the rail timetable is the same pattern. Stations are parent stations of the feed, and the last mile is a straight-line buffer, not a street route.\n\n**Try it:** set **Last mile** to *Walk* and compare Maastricht with Eindhoven; raise **Dwell per station** to 120 seconds to see how much of the 60-minute region is dwell; click a terminal station at the end of a branch line and watch how slowly the curve rises.',
      options: {origin: 'hub:Maastricht', lastMile: 'walk', lastMileMinutes: 15, trainTypes: 'all'},
      camera: {longitude: 5.7, latitude: 51.3, zoom: 7.4, transitionMs: 1600},
      controls: ['origin', 'lastMile', 'dwellSeconds'],
      readouts: ['bandTable', 'solver']
    }
  ]
});
