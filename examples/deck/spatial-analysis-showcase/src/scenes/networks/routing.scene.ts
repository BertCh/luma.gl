// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import type {RoutingOptions} from './routing.compute';

const PLACE_OPTIONS = [
  {value: 'willis', label: 'Willis Tower (Loop)'},
  {value: 'ohare', label: "O'Hare Airport"},
  {value: 'midway', label: 'Midway Airport'},
  {value: 'wrigley', label: 'Wrigley Field'},
  {value: 'soldier', label: 'Soldier Field'},
  {value: 'unitedCenter', label: 'United Center'},
  {value: 'msi', label: 'Museum of Science and Industry'},
  {value: 'southShore', label: 'South Shore'},
  {value: 'austin', label: 'Austin (West Side)'}
] as const;

const options: readonly OptionSpec<RoutingOptions>[] = [
  {
    kind: 'select',
    id: 'originPlace',
    label: 'Origin',
    group: 'Trip',
    apply: 'param',
    default: 'willis',
    options: PLACE_OPTIONS,
    help: 'The root of the shortest-path tree. The nearest street intersection is used. You can also click the map to move the origin or the first destination.'
  },
  {
    kind: 'select',
    id: 'destinationPlace',
    label: 'First destination',
    group: 'Trip',
    apply: 'param',
    default: 'ohare',
    options: PLACE_OPTIONS,
    help: 'Destination 1 (magenta), also the one the turn-aware route and the turn counts are measured for.'
  },
  {
    kind: 'slider',
    id: 'destinationCount',
    label: 'Destinations extracted',
    group: 'Trip',
    apply: 'param',
    min: 1,
    max: 6,
    step: 1,
    default: 1,
    help: 'How many of the six preset destinations to extract routes to. All of them come from the same tree: this only changes the target count of GPUNetworkPathExtraction.'
  },
  {
    kind: 'slider',
    id: 'costLimitMinutes',
    label: 'Search budget',
    group: 'Trip',
    apply: 'param',
    min: 10,
    max: 180,
    step: 5,
    default: 90,
    unit: 'min',
    help: 'Cost limit of the search: intersections more than this far away are never reached (and stay uncoloured). Also the end of the colour ramp.'
  },
  {
    kind: 'toggle',
    id: 'closeExpressways',
    label: 'Close the expressways',
    group: 'Network state',
    apply: 'param',
    default: false,
    help: 'Rewrites the weight of every motorway and trunk edge to -1 (impassable), as if I-90/94, I-290, I-55 and Lake Shore Drive were closed. The CSR is rewritten, nothing recompiles.'
  },
  {
    kind: 'slider',
    id: 'expresswaySlowdown',
    label: 'Expressway congestion',
    group: 'Network state',
    apply: 'param',
    min: 1,
    max: 4,
    step: 0.25,
    default: 1,
    unit: 'x',
    format: value => `${value.toFixed(2)}x travel time`,
    help: 'Multiplies the free-flow travel time of motorway and trunk edges. 1 is free flow; 3 is a bad rush hour.'
  },
  {
    kind: 'slider',
    id: 'intersectionDelay',
    label: 'Delay per intersection',
    group: 'Network state',
    apply: 'param',
    min: 0,
    max: 30,
    step: 1,
    default: 8,
    unit: 's',
    help: 'Seconds added to every edge for the intersection it ends at (signal or stop). The dataset only has free-flow travel time; this is what makes dense street grids slower than expressways.'
  },
  {
    kind: 'toggle',
    id: 'showTurnRoute',
    label: 'Turn-aware route (line graph)',
    group: 'Turns',
    apply: 'param',
    default: false,
    help: 'Builds the edge-based line graph with GPUNetworkLineGraph and routes over it. The blue route pays the costs below; the orange route ignores turns. Switching it on only starts encoding the third graph.'
  },
  {
    kind: 'slider',
    id: 'angleCost',
    label: 'Cost per radian turned',
    group: 'Turns',
    apply: 'param',
    min: 0,
    max: 60,
    step: 1,
    default: 6,
    unit: 's',
    disabledWhen: state => !state.showTurnRoute,
    help: 'Every turn costs this much per radian of heading change: about 9 s for a right angle at 6 s/rad. Straight-on moves cost nothing.'
  },
  {
    kind: 'slider',
    id: 'leftTurnCost',
    label: 'Extra cost of a left turn',
    group: 'Turns',
    apply: 'param',
    min: 0,
    max: 120,
    step: 5,
    default: 25,
    unit: 's',
    disabledWhen: state => !state.showTurnRoute,
    help: 'Added to turns beyond the straight-on threshold. Left turns across oncoming traffic are slower and less safe; delivery fleets often avoid them.'
  },
  {
    kind: 'slider',
    id: 'rightTurnCost',
    label: 'Extra cost of a right turn',
    group: 'Turns',
    apply: 'param',
    min: 0,
    max: 60,
    step: 5,
    default: 0,
    unit: 's',
    disabledWhen: state => !state.showTurnRoute,
    help: 'Added to right turns beyond the straight-on threshold.'
  },
  {
    kind: 'select',
    id: 'uTurns',
    label: 'U-turns',
    group: 'Turns',
    apply: 'param',
    default: 'banned',
    options: [
      {value: 'banned', label: 'Banned (negative cost)'},
      {value: 'allowed', label: 'Allowed at 90 s'}
    ],
    disabledWhen: state => !state.showTurnRoute,
    help: 'A negative U-turn cost removes the arc, so a route can never double back along the same street.'
  },
  {
    kind: 'slider',
    id: 'straightAngle',
    label: 'Straight-on threshold',
    group: 'Turns',
    apply: 'param',
    min: 0.1,
    max: 1.2,
    step: 0.05,
    default: 0.5,
    unit: 'rad',
    disabledWhen: state => !state.showTurnRoute,
    help: 'Heading changes smaller than this count as going straight. 0.5 rad is about 29 degrees. The left/right counts in the readouts use the same threshold.'
  },
  {
    kind: 'select',
    id: 'banLeftTurns',
    label: 'Banned turns',
    group: 'Turns',
    apply: 'param',
    default: 'none',
    options: [
      {value: 'none', label: 'None'},
      {value: 'arterials', label: 'No left turns between arterials'}
    ],
    disabledWhen: state => !state.showTurnRoute,
    help: 'Passes a list of (from edge, to edge) pairs as bannedTurns. The line graph checks every pair for every arc, so keep lists to a few thousand.'
  },
  {
    kind: 'slider',
    id: 'hops',
    label: 'Ego network radius',
    group: 'Neighbourhood',
    apply: 'param',
    min: 1,
    max: 24,
    step: 1,
    default: 8,
    unit: 'hops',
    help: 'k for GPUNetworkNeighborhood: every intersection within k street segments of the origin. A hop count ignores length and speed.'
  },
  {
    kind: 'select',
    id: 'localIterations',
    label: 'Hops chained per round',
    group: 'Solver',
    apply: 'compile',
    default: '32',
    options: [
      {value: '8', label: '8 (more rounds)'},
      {value: '16', label: '16'},
      {value: '32', label: '32 (default)'},
      {value: '64', label: '64 (fewest rounds)'}
    ],
    help: 'Compile-time localIterations of GPUNetworkReachability. The costs are identical for every value; only the rounds needed to converge change. Check the Solver readout: 8 needs many more rounds than 64.'
  },
  {
    kind: 'toggle',
    id: 'showRoute',
    label: 'Show the plain route',
    group: 'Display',
    apply: 'param',
    default: true,
    help: 'Draws the route(s) extracted from the predecessor tree in orange.'
  },
  {
    kind: 'select',
    id: 'base',
    label: 'Colour the streets by',
    group: 'Display',
    apply: 'param',
    default: 'time',
    options: [
      {value: 'time', label: 'Drive time from the origin'},
      {value: 'hops', label: 'Intersections from the origin (ego network)'},
      {value: 'none', label: 'Nothing (plain street grid)'}
    ],
    help: 'Drive time is the reachability cost tree. Intersections is the k-hop ego network of GPUNetworkNeighborhood; only streets inside the hop radius are coloured.'
  },
  {
    kind: 'select',
    id: 'ramp',
    label: 'Colour ramp',
    group: 'Display',
    apply: 'param',
    default: 'viridis',
    options: [
      {value: 'viridis', label: 'Viridis'},
      {value: 'magma', label: 'Magma'},
      {value: 'inferno', label: 'Inferno'},
      {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
    ],
    help: 'Perceptually uniform ramps. The route colours stay fixed so they remain visible on every ramp.'
  }
];

export default defineScene<RoutingOptions>({
  id: 'routing',
  title: 'From the Loop to O’Hare',
  chapter: 'networks',
  order: 1,
  summary:
    'Shortest-path trees, routes to several destinations, k-hop ego networks and turn-aware routing on the full Chicago street graph, all recomputed on the GPU when you close a road or move an endpoint.',
  contributors: [
    'GPUNetworkReachability',
    'GPUNetworkPathExtraction',
    'GPUNetworkNeighborhood',
    'GPUNetworkLineGraph'
  ],
  datasets: [{id: 'chicago-roads', role: 'directed street graph'}],
  initialView: {longitude: -87.74, latitude: 41.86, zoom: 10.1},
  options,

  readouts: [
    {
      id: 'nodes',
      label: 'Intersections',
      help: 'Nodes of the CSR (largest strongly connected component of the OSM drive network).'
    },
    {
      id: 'edges',
      label: 'Directed edges',
      help: 'Edges of the CSR; a two-way street is two edges.'
    },
    {
      id: 'routeTime',
      label: 'Route 1 drive time',
      help: 'pathCosts[0] of GPUNetworkPathExtraction: the cost of the shortest path to the first destination.'
    },
    {
      id: 'routeTimes',
      label: 'All extracted routes',
      help: 'Drive time to each extracted destination, in order.'
    },
    {
      id: 'routeLength',
      label: 'Route 1 length',
      help: 'Sum of the lengths of the extracted edges of route 1.'
    },
    {
      id: 'plainTurns',
      label: 'Route 1 turns (plain)',
      help: 'Left, right and U-turns along the plain route, measured on the CPU from the extracted edges.'
    },
    {
      id: 'turnTime',
      label: 'Turn-aware route 1 (cost)',
      help: 'Total cost of the cheapest route when turns are charged: drive time plus the turn penalties it pays.'
    },
    {
      id: 'turnTurns',
      label: 'Route 1 turns (turn-aware)',
      help: 'Left, right and U-turns along the turn-aware route.'
    },
    {
      id: 'turnCost',
      label: 'Detour for fewer turns',
      help: 'Travel time (without turn penalties) and length of the turn-aware route compared with the plain route.'
    },
    {
      id: 'reached',
      label: 'Intersections reached',
      help: 'Nodes with a finite cost within the search budget.'
    },
    {
      id: 'routeSize',
      label: 'Extracted paths',
      help: 'Total node and edge rows written by GPUNetworkPathExtraction.'
    },
    {
      id: 'hood',
      label: 'Ego network size',
      help: 'Nodes and induced street segments within the hop radius.'
    },
    {
      id: 'arcs',
      label: 'Line graph arcs',
      help: 'Allowed turns written by GPUNetworkLineGraph, against its compile-time capacity.'
    },
    {
      id: 'bans',
      label: 'Candidate banned turns',
      help: 'Left turns between two primary or secondary roads that the "banned turns" option can forbid.'
    },
    {id: 'expressways', label: 'Expressways'},
    {
      id: 'solver',
      label: 'Solver',
      help: 'Rounds the frontier search used before its queue emptied.'
    }
  ],

  legends: state => {
    const legends: LegendSpec[] = [
      ...(state.base === 'time'
        ? [
            {
              kind: 'ramp' as const,
              title: 'Drive time from the origin',
              ramp: state.ramp,
              extent: [0, state.costLimitMinutes] as const,
              unit: 'min',
              format: (value: number) => value.toFixed(0)
            }
          ]
        : state.base === 'hops'
          ? [
              {
                kind: 'ramp' as const,
                title: 'Intersections from the origin',
                ramp: state.ramp,
                extent: [0, state.hops] as const,
                unit: 'hops',
                format: (value: number) => value.toFixed(0)
              }
            ]
          : []),
      {
        kind: 'categories' as const,
        title: 'Routes and places',
        entries: [
          ...(state.showRoute
            ? [{color: [255, 96, 64, 255] as const, label: 'Fastest route (no turn costs)'}]
            : []),
          ...(state.showTurnRoute
            ? [{color: [40, 190, 255, 255] as const, label: 'Turn-aware route'}]
            : []),
          {color: [255, 255, 255, 255] as const, label: 'Origin'},
          {
            color: [255, 70, 200, 255] as const,
            label: 'Destination 1 (other destinations: other hues)'
          }
        ],
        note: 'Hover the map for the drive time to the nearest intersection. Click to move the destination; click near the origin to move the origin.'
      }
    ];
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkReachability,
  GPUNetworkPathExtraction,
  GPUNetworkNeighborhood${state.showTurnRoute ? ',\n  GPUNetworkLineGraph,\n  getGPUNetworkLineGraphParameterValues' : ''}
} from '@luma.gl/experimental/gpu-network';

// offsets / neighbors: directed CSR of the street graph, weights: seconds per edge (-1 = closed)
const graph = new GPUCommandGraph(device, {id: 'routing'});
graph.add(new GPUNetworkReachability({
  offsets, neighbors, weights,
  sources: origin.importToGraph(graph),          // one row: the origin node
  costLimit: budget.importToGraph(graph),        // seconds
  maxIterations: 80,
  localIterations: ${state.localIterations},                     // compile-time
  costs, predecessors, converged, iterationCount
}));
graph.add(new GPUNetworkPathExtraction({
  predecessors, costs,
  targets: destinations.importToGraph(graph),     // up to 6 rows, count: ${state.destinationCount}
  targetCount: destinationCount.importToGraph(graph),
  maxPathLength: 2048,
  output: {ids: pathNodes, count: pathNodeCount, overflow: pathNodeOverflow},
  edges: {offsets, neighbors, weights, output: {ids: pathEdges, count: pathEdgeCount, overflow}}
}));
graph.add(new GPUNetworkNeighborhood({
  offsets, neighbors,
  seeds: origin.importToGraph(graph),
  hops: hops.importToGraph(graph),               // k = ${state.hops}, per frame
  maxHops: 24,
  hopDistances, edgeMask
}));${
    state.showTurnRoute
      ? `
// Turn-aware: one line-graph node per directed edge, one arc per allowed turn.
graph.add(new GPUNetworkLineGraph({
  offsets, neighbors, weights, nodePositions,
  parameters: turnParameters.importToGraph(graph),
  bannedTurns,                                   // optional (fromEdge, toEdge) pairs
  lineOffsets, lineNeighbors, lineWeights, arcCount, overflow
}));
turnParameters.write(getGPUNetworkLineGraphParameterValues({
  angleCost: ${state.angleCost}, leftTurnCost: ${state.leftTurnCost}, rightTurnCost: ${state.rightTurnCost},
  uTurnCost: ${state.uTurns === 'banned' ? -1 : 90}, straightAngle: ${state.straightAngle}
}));
// then a second GPUNetworkReachability over lineOffsets/lineNeighbors/lineWeights,
// seeded with the out-edges of the origin at their own weights.`
      : ''
  }
const compiled = graph.compile();                // once
// closing a road, moving the origin, changing a turn cost = a buffer write, then
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUNetworkReachability` runs a frontier-based Bellman-Ford from the origin and writes the cheapest cost and the predecessor of every intersection. `GPUNetworkPathExtraction` walks predecessors back from up to six destinations in parallel, `GPUNetworkNeighborhood` computes hop counts for the k-hop ego network, and `GPUNetworkLineGraph` rebuilds the graph so that a node is a directed street segment and an arc is a turn with its own cost.',
    why: 'This is the engine behind "how long to get there", "what is within reach", "where would a closure push traffic" and "can this fleet avoid left turns". Because the CSR weights are an ordinary buffer, a closure or a congestion level is a write, not a rebuild.',
    howToRead:
      'Street colour is drive time from the origin (or hops from it in ego-network mode); uncoloured streets are beyond the budget. The orange line is the cheapest path to the first destination, the blue one the cheapest path when turns cost time. Compare their lengths and the left/right counts in the readouts.'
  },

  create: async ctx => (await import('./routing.compute')).createRouting(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['originPlace', 'destinationPlace'],
      readouts: ['routeTime'],
      title: 'How long from the Loop to O’Hare?',
      body: 'Chicago’s drive network has about 30,000 intersections and 77,000 directed street segments. The first question any routing engine answers is *how long does it take to get from here to there?* Here the origin is Willis Tower and the destination is O’Hare Airport, 25 km northwest.\n\nEverything you see is computed on the GPU in one compiled graph. Hover the map for the drive time to any intersection, or click to pick another destination; **Origin** and **First destination** below pick from presets.',
      camera: {longitude: -87.77, latitude: 41.89, zoom: 10.0, transitionMs: 1200},
      options: {destinationCount: 1, base: 'time'},
      highlight: {readout: 'routeTime'}
    },
    {
      id: 'the-tree',
      controls: ['costLimitMinutes', 'base'],
      readouts: ['solver', 'reached'],
      title: 'One search, every destination: the shortest-path tree',
      body: '`GPUNetworkReachability` is a single-source shortest-path search. It keeps a *frontier* of intersections whose cost just improved and relaxes their outgoing edges in rounds, so the whole city converges in a handful of dispatches. Each intersection ends up with a **cost** (seconds from the origin) and a **predecessor** (the intersection you came from).\n\nThe street colours are that cost: dark is near, bright is far, so the expressways glow far out because they are fast. Uncoloured streets lie beyond the **search budget**: drag **Search budget** below down to 25 minutes and watch the tree shrink. The **Solver** readout says how many rounds the search needed before its queue emptied.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 9.7, transitionMs: 1200},
      options: {costLimitMinutes: 60, base: 'time'},
      highlight: {readout: 'solver'}
    },
    {
      id: 'many-routes',
      controls: ['destinationCount', 'showRoute'],
      readouts: ['routeTimes'],
      title: 'Walking the tree back: routes to many places',
      body: '`GPUNetworkPathExtraction` takes the predecessor array and a list of targets and walks every target back to the origin in parallel, then packs the node and edge lists with a prefix sum. Raise **Destinations extracted** below to six: O’Hare, Midway, Wrigley Field, Soldier Field, the Museum of Science and Industry and the United Center all come out of **one** tree.\n\nThe routes are drawn by scattering the extracted edge ids into a flag buffer that the road layer reads, so nothing is copied to the CPU to draw them. The readouts do read the small result lists, throttled, after a change.',
      camera: {longitude: -87.72, latitude: 41.87, zoom: 10.1, transitionMs: 1200},
      options: {destinationCount: 6},
      highlight: {readout: 'routeTimes'}
    },
    {
      id: 'rush-hour',
      controls: ['expresswaySlowdown', 'closeExpressways', 'intersectionDelay'],
      readouts: ['routeTime', 'expressways'],
      title: 'Rush hour on the expressways',
      body: 'The graph weights are just a buffer, so traffic is a rewrite. Here **Expressway congestion** is set to 4x, so every motorway and trunk edge takes four times its free-flow time and the whole search runs again from scratch. Compare the tree with the previous step: the bright corridors have faded, and the route to O’Hare may leave the Kennedy for arterial streets.\n\nTry the two other controls below: **Close the expressways** sets their weights to -1 and the airport simply becomes unreachable (the route reads "not found"), and **Delay per intersection** shows how much of a trip is the street grid itself.',
      options: {destinationCount: 1, expresswaySlowdown: 4},
      highlight: {readout: 'routeTime'}
    },
    {
      id: 'turn-costs',
      controls: ['showTurnRoute', 'leftTurnCost', 'angleCost', 'uTurns'],
      readouts: ['turnTurns', 'plainTurns', 'turnCost'],
      title: 'Turns cost time too',
      body: 'A node-based graph cannot say "no left turn here": the cost of leaving an intersection cannot depend on how you arrived. `GPUNetworkLineGraph` fixes that by making every **directed street segment** a node and every allowed **turn** an arc whose cost is the next segment plus a penalty for the heading change.\n\nThe blue route (**Turn-aware route (line graph)**) pays 120 s per left turn (**Extra cost of a left turn**) plus 25 s per radian of any turn (**Cost per radian turned**) and may never U-turn (**U-turns**); the orange one ignores turns. From the United Center to Wrigley Field it abandons the Kennedy detour for a straight run up Western Avenue. The turn counts in the readouts show why. Change the costs below: the line graph is rebuilt every encoding, so the sliders need no recompile.',
      camera: {longitude: -87.665, latitude: 41.915, zoom: 11.5, transitionMs: 1400},
      options: {
        expresswaySlowdown: 1,
        originPlace: 'unitedCenter',
        destinationPlace: 'wrigley',
        showTurnRoute: true,
        leftTurnCost: 120,
        angleCost: 25
      },
      highlight: {readout: 'turnTurns'}
    },
    {
      id: 'ego-network',
      controls: ['base', 'hops'],
      readouts: ['hood'],
      title: 'Counting intersections instead of minutes',
      body: '`GPUNetworkNeighborhood` is the *topological* neighbourhood: every intersection within **k hops**, whatever the length or speed of the street. The colouring is set to *Intersections from the origin* (**Colour the streets by**); slide **Ego network radius**: on Chicago’s grid the ego network is a diamond, because each block is one hop in one of four directions.\n\nHop distance and drive time disagree wherever blocks are unequal or streets are one-way, which is exactly why routing uses weighted costs. The ego network also returns the induced street segments, so it doubles as a subgraph extractor.',
      camera: {longitude: -87.63, latitude: 41.88, zoom: 12.4, transitionMs: 1200},
      options: {
        base: 'hops',
        hops: 12,
        showRoute: false,
        showTurnRoute: false,
        originPlace: 'willis',
        destinationPlace: 'ohare',
        destinationCount: 1
      },
      highlight: {readout: 'hood'}
    },
    {
      id: 'limits',
      controls: ['banLeftTurns', 'localIterations', 'originPlace', 'ramp'],
      readouts: ['solver'],
      title: 'Limits, and things to try',
      body: 'Costs are free-flow times from OpenStreetMap speed limits plus a fixed intersection delay: no live traffic, signal timing or time-of-day variation. OSM turn restrictions are not applied (the **Banned turns** list is illustrative); a real application would convert each restriction relation to `(from edge, to edge)` pairs once on the CPU. Weights must be non-negative, and are floored at 0.1 s so equal-cost plateaus stay rare. Walking and transit are in the *job accessibility* scene.\n\nTry, with the controls below (the turn-aware route is on in this step): set **Banned turns** to no left turns between arterials and drag the destination around the Loop; set **Hops chained per round** to 8 and read the **Solver** rounds; set **Origin** to Austin or South Shore and compare drive times to the Loop; switch **Colour ramp** to cividis.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 10.0, transitionMs: 1200},
      options: {base: 'time', showRoute: true, showTurnRoute: true, hops: 8}
    }
  ]
});
