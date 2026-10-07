// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import type {RoutingOptions} from './routing.compute';

const ROUTING_TIME_TABLE = makeClassTable({
  breaks: [600, 1200, 1800, 2700, 3600],
  scheme: 'YlGnBu',
  reverse: true,
  labels: ['0–10', '10–20', '20–30', '30–45', '45–60', '60–90 min'],
  unit: 'min',
  extent: [0, 5400],
  noData: {color: [117, 122, 132, 105], label: 'Unreached'}
});
const SCENARIO_DELTA_TABLE = makeClassTable({
  breaks: [0, 300, 600, 1200],
  scheme: 'PuRd',
  labels: ['No added time', '0–5', '5–10', '10–20', '>20 min'],
  unit: 'min',
  extent: [0, 1200],
  transparent: [0],
  noData: {color: [117, 122, 132, 105], label: 'Unreachable in scenario'}
});
const DESTINATION_NUMBER_ANNOTATIONS = [
  {coordinate: [-87.9048, 41.9786] as const, text: '1', detail: 'O’Hare', offset: [8, -8] as const},
  {coordinate: [-87.7524, 41.7868] as const, text: '2', detail: 'Midway', offset: [8, 10] as const},
  {
    coordinate: [-87.6553, 41.9484] as const,
    text: '3',
    detail: 'Wrigley',
    offset: [8, -8] as const
  },
  {
    coordinate: [-87.6167, 41.8623] as const,
    text: '4',
    detail: 'Soldier Field',
    offset: [8, 10] as const
  },
  {
    coordinate: [-87.5831, 41.7906] as const,
    text: '5',
    detail: 'Museum of Science and Industry',
    offset: [8, 10] as const
  },
  {
    coordinate: [-87.6742, 41.8807] as const,
    text: '6',
    detail: 'United Center',
    offset: [8, -8] as const
  }
].map((annotation, index) => ({
  ...annotation,
  id: `routing-destination-${index + 1}`,
  kind: 'point' as const,
  marker: 'ring' as const,
  tone: 'signal' as const,
  priority: 4
}));

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
    max: 90,
    step: 5,
    default: 90,
    unit: 'min',
    help: 'Cost limit of the search: intersections more than this far away are unreached. The fixed 0–90 minute classes do not rescale.'
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
    help: 'Multiplies motorway and trunk free-flow travel time for an illustrative slowdown counterfactual; it is not a traffic prediction.'
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
    help: 'Builds the edge-based line graph with GPUNetworkLineGraph. The purple dashed route pays the costs below; the orange route ignores turns.'
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
      {value: 'delta', label: 'Minutes added versus free flow'},
      {value: 'hops', label: 'Intersections from the origin (ego network)'},
      {value: 'none', label: 'Nothing (plain street grid)'}
    ],
    help: 'Drive time is the current reachability cost tree. Minutes added compares it with a separately retained free-flow tree using fixed 30 s, 2 min and 5 min classes. Intersections is the k-hop ego network of GPUNetworkNeighborhood.'
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
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Free-flow route through Chicago'},
    credit: 'OpenStreetMap contributors (ODbL)',
    caveat: 'Class-default speeds + constant intersection delay; not observed traffic.',
    scaleBar: {units: 'metric'}
  },
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
      label: 'Selected destination time',
      help: 'pathCosts[0] of GPUNetworkPathExtraction: the cost of the shortest path to the first destination.'
    },
    {
      id: 'freeFlowRoute',
      label: 'Selected free-flow time',
      help: 'The same origin, destination and search budget solved against retained OSM free-flow edge costs, before congestion, closures or intersection delay.'
    },
    {
      id: 'scenarioDelta',
      label: 'Scenario delay',
      help: 'Current selected-route cost minus its retained free-flow cost. A closure can leave the scenario route unreached.'
    },
    {id: 'scenarioComparison', label: 'Free flow versus scenario', kind: 'chart'},
    {
      id: 'routeTimes',
      label: 'All extracted routes',
      help: 'Drive time to each extracted destination, in order.'
    },
    {id: 'routeComparison', label: 'Extracted destination times', kind: 'chart'},
    {id: 'reachCurve', label: 'Cumulative intersections reached', kind: 'chart'},
    {
      id: 'routeLength',
      label: 'Selected route length',
      help: 'Sum of the lengths of the extracted edges of route 1.'
    },
    {
      id: 'plainTurns',
      label: 'Plain-route turns',
      help: 'Left, right and U-turns along the plain route, measured on the CPU from the extracted edges.'
    },
    {
      id: 'turnTime',
      label: 'Turn-aware route 1 (cost)',
      help: 'Total cost of the cheapest route when turns are charged: drive time plus the turn penalties it pays.'
    },
    {
      id: 'turnTurns',
      label: 'Turn-aware-route turns',
      help: 'Left, right and U-turns along the turn-aware route.'
    },
    {
      id: 'turnCost',
      label: 'Detour for fewer turns',
      help: 'Travel time (without turn penalties) and length of the turn-aware route compared with the plain route.'
    },
    {id: 'turnComparison', label: 'Plain versus turn-aware turns', kind: 'chart'},
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
        ? [getClassTableLegend(ROUTING_TIME_TABLE, {title: 'Free-flow travel time'})]
        : state.base === 'hops'
          ? [
              {
                kind: 'ramp' as const,
                title: 'Intersections from the origin',
                ramp: 'ylgnbu' as const,
                extent: [0, state.hops] as const,
                unit: 'hops',
                format: (value: number) => value.toFixed(0)
              }
            ]
          : state.base === 'delta'
            ? [
                getClassTableLegend(SCENARIO_DELTA_TABLE, {
                  title: 'Minutes added versus free flow',
                  note: 'Zero is transparent; dashed plum marks are free-flow-reachable but scenario-unreachable.'
                })
              ]
            : []),
      {
        kind: 'categories' as const,
        title: 'Routes and places',
        entries: [
          ...(state.showRoute
            ? [{color: [230, 120, 48, 255] as const, label: 'Fastest route (no turn costs)'}]
            : []),
          ...(state.showTurnRoute
            ? [{color: [113, 72, 150, 255] as const, label: 'Turn-aware route (dashed)'}]
            : []),
          {color: [255, 255, 255, 255] as const, label: 'Origin'},
          {
            color: [255, 70, 200, 255] as const,
            label: 'Numbered destination ring-dots match the destination-time chart'
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
      'Street colour is fixed free-flow travel-time classes (or hops in ego-network mode); neutral streets are unreached. The orange line is the selected path and purple dashes are the turn-aware path. Compare their lengths and turn counts in the readouts.'
  },

  create: async ctx => (await import('./routing.compute')).createRouting(ctx),

  story: [
    {
      id: 'one-route',
      headline: 'One path starts at the Loop',
      textAlternative:
        'A cased orange route runs from Willis Tower in the Loop to O’Hare, over classed free-flow street times.',
      optionsMode: 'fresh',
      controls: ['originPlace', 'destinationPlace'],
      readouts: ['routeTime', 'routeLength'],
      title: 'One path from the Loop',
      body: 'The orange line walks predecessors backward from O’Hare to Willis Tower. It is a best-case free-flow model, not observed traffic. Change the two places to recompute the selected path from the same street graph.',
      camera: {longitude: -87.78, latitude: 41.93, zoom: 10.2, transitionMs: 1200},
      options: {destinationCount: 1, base: 'time'},
      highlight: {readout: 'routeTime'}
    },
    {
      id: 'whole-tree',
      headline: 'One search reaches every street',
      textAlternative:
        'Chicago streets are grouped in six fixed travel-time classes, with a cumulative chart of intersections reached by minute.',
      optionsMode: 'fresh',
      controls: ['costLimitMinutes', 'base'],
      readouts: ['reached', 'reachCurve', 'solver'],
      title: 'One search, every street',
      body: 'The shortest-path tree retains one cost and predecessor for every reached intersection. Fixed 0–10 through 60–90 minute classes keep later comparisons honest; the cumulative chart is counted directly from those node costs.',
      camera: {longitude: -87.7, latitude: 41.84, zoom: 9.7, transitionMs: 1200},
      options: {costLimitMinutes: 60, base: 'time'},
      highlight: {readout: 'reachCurve'}
    },
    {
      id: 'many-routes',
      headline: 'Six routes share one tree',
      textAlternative:
        'Six numbered destination ring-dots radiate from the Loop; the orange selected route is cased and the comparison routes remain thin, with a named time chart.',
      optionsMode: 'fresh',
      controls: ['destinationCount', 'showRoute'],
      readouts: ['routeTimes', 'routeComparison'],
      title: 'Walk the tree back',
      body: 'One predecessor field yields routes to O’Hare, Midway, Wrigley Field, Soldier Field, the Museum of Science and Industry and United Center. Ring-dot numbers and chart labels use the same places; the selected route stays above the thinner comparisons.',
      camera: {longitude: -87.72, latitude: 41.87, zoom: 10.1, transitionMs: 1200},
      annotations: DESTINATION_NUMBER_ANNOTATIONS,
      options: {destinationCount: 6},
      highlight: {readout: 'routeComparison'}
    },
    {
      id: 'scenario-delta',
      headline: 'A slowdown changes the model',
      textAlternative:
        'A fixed PuRd map shows minutes added to free flow; zero is absent and free-flow-reachable streets made unreachable are dashed plum.',
      optionsMode: 'fresh',
      controls: ['expresswaySlowdown', 'closeExpressways', 'intersectionDelay'],
      readouts: ['freeFlowRoute', 'scenarioDelta', 'scenarioComparison', 'expressways'],
      title: 'Scenario delta',
      body: 'This illustrative slowdown counterfactual rewrites motorway and trunk weights; it is not observed traffic. The retained free-flow tree and fixed PuRd 0, 0–5, 5–10, 10–20 and >20 minute table make actual minutes-added visible.',
      options: {destinationCount: 1, expresswaySlowdown: 4, base: 'delta'},
      highlight: {readout: 'scenarioDelta'}
    },
    {
      id: 'turns',
      headline: 'Turns change the route',
      textAlternative:
        'An orange plain route and a purple dashed turn-aware route compare between United Center and Wrigley Field, with paired turn-count bars.',
      optionsMode: 'fresh',
      controls: ['leftTurnCost', 'angleCost', 'uTurns'],
      readouts: ['plainTurns', 'turnTurns', 'turnCost', 'turnComparison'],
      title: 'Turns cost time too',
      body: 'A line graph makes each directed edge a node, so the next edge can charge an angle or left-turn cost. Orange is the plain route; purple dashes pay turn costs. The paired readouts count turns from the two extracted paths.',
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
      id: 'hops',
      headline: 'Blocks are not minutes',
      textAlternative:
        'A hop-distance street diamond is compared with a ten-minute free-flow outline around the Loop.',
      optionsMode: 'fresh',
      readouts: ['hood', 'solver'],
      title: 'Blocks are not minutes',
      body: 'A k-hop neighbourhood counts street segments, not their length or speed. It therefore forms a different shape from the travel-time tree. Solver rounds and capacities are implementation details; free-flow weights, nearest-node snapping, graph clipping and simplified turn restrictions limit this model.',
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
      highlight: {readout: 'hood'},
      controls: ['base', 'hops', 'localIterations']
    }
  ]
});
