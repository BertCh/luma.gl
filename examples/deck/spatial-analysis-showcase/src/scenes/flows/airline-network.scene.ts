// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './airline-network.md?raw';
import type {AirlineNetworkOptions} from './airline-network.compute';
import {NETWORK_PALETTE} from './airline-network-palette';

const WORLD_VIEW = {longitude: 10, latitude: 26, zoom: 1.75};

const SIZE_TITLES: Record<AirlineNetworkOptions['sizeBy'], string> = {
  pagerank: 'Disc radius ~ sqrt(PageRank)',
  degree: 'Disc radius ~ sqrt(connections)',
  core: 'Disc radius ~ sqrt(core number)',
  uniform: ''
};

export default defineScene<AirlineNetworkOptions>({
  id: 'airline-network',
  title: 'The airline network as a graph',
  chapter: 'flows',
  order: 22,
  summary:
    'PageRank, core numbers, communities and modularity on the OpenFlights route graph, then a morph from geography to a force layout, drawn as great-circle arcs or bundles coloured by community.',
  contributors: [
    'GPUGraph',
    'GPUGraphTopology',
    'GPUGraphDegree',
    'GPUGraphPageRank',
    'GPUGraphCoreNumber',
    'GPUGraphLabelPropagation',
    'GPUGraphModularity',
    'GPUGraphModularityOptimization',
    'GPUGraphForceLayout',
    'GPUGraphSpatialForceLayout',
    'GPUGreatCircleArcs',
    'GPUEdgeBundling'
  ],
  datasets: [{id: 'openflights', role: 'airports and route pairs'}],
  initialView: WORLD_VIEW,

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'View',
      group: 'Display',
      apply: 'param',
      default: 'arcs',
      help: 'Great-circle arcs on the map, edge bundles on the map, or the morph between the map and the force layout. Bundling and the layout only run while their view is open.',
      options: [
        {value: 'arcs', label: 'Great-circle arcs (GPUGreatCircleArcs)'},
        {value: 'bundles', label: 'Edge bundles (GPUEdgeBundling)'},
        {value: 'morph', label: 'Geography to force layout'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour groups by',
      group: 'Display',
      apply: 'param',
      default: 'community',
      help: 'Colour airports and routes by detected community, or by continent for comparison. The six largest groups get their own colour; routes between two groups are red-orange.',
      options: [
        {value: 'community', label: 'Community'},
        {value: 'continent', label: 'Continent'}
      ]
    },
    {
      kind: 'select',
      id: 'sizeBy',
      label: 'Disc size by',
      group: 'Display',
      apply: 'param',
      default: 'pagerank',
      help: 'The measure that sizes each airport disc, as the square root of the value. Core number is the deepest k-core the airport belongs to.',
      options: [
        {value: 'pagerank', label: 'PageRank'},
        {value: 'degree', label: 'Connections (degree)'},
        {value: 'core', label: 'Core number'},
        {value: 'uniform', label: 'Same size'}
      ]
    },
    {
      kind: 'slider',
      id: 'maxRadius',
      label: 'Largest disc',
      group: 'Display',
      apply: 'param',
      min: 6,
      max: 40,
      step: 1,
      default: 22,
      unit: 'px',
      help: 'Radius of the most important airport; everything else scales from it.'
    },
    {
      kind: 'slider',
      id: 'edgeOpacity',
      label: 'Route opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.5,
      help: 'Routes are 1 px lines that add up where they overlap. Lower it for dense regions.'
    },
    {
      kind: 'select',
      id: 'edgeFilter',
      label: 'Routes shown',
      group: 'Display',
      apply: 'param',
      default: 'all',
      help: 'Keep every route, only routes inside a group, or only the routes that run between two groups. A display filter: the graph algorithms always see every route.',
      options: [
        {value: 'all', label: 'All routes'},
        {value: 'within', label: 'Within groups only'},
        {value: 'between', label: 'Between groups only'}
      ]
    },
    {
      kind: 'slider',
      id: 'minRecords',
      label: 'Minimum route records',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 12,
      step: 1,
      default: 1,
      help: 'Keep pairs with at least this many airline-route records (both directions, all carriers). OpenFlights has no frequencies, so this is the closest it has to traffic. Display filter only.'
    },
    {
      kind: 'slider',
      id: 'arcSegmentKm',
      label: 'Arc segment length',
      group: 'Display',
      apply: 'param',
      min: 50,
      max: 1000,
      step: 50,
      default: 250,
      unit: 'km',
      disabledWhen: state => state.view !== 'arcs',
      help: 'Longest straight piece of a great-circle arc. Shorter pieces follow the curve more closely and cost more vertices (the arc readout shows the total).'
    },
    {
      kind: 'slider',
      id: 'pageRankDamping',
      label: 'PageRank damping',
      group: 'Ranking',
      apply: 'compile',
      min: 0.5,
      max: 0.99,
      step: 0.01,
      default: 0.85,
      format: value => value.toFixed(2),
      help: 'Probability of following a route instead of jumping to a random airport. Near 1 favours the dense core; near 0 it approaches a uniform ranking. A shader constant, so it rebuilds the analysis graph.'
    },
    {
      kind: 'slider',
      id: 'pageRankIterations',
      label: 'PageRank iterations',
      group: 'Ranking',
      apply: 'compile',
      min: 5,
      max: 200,
      step: 5,
      default: 40,
      help: 'Compiled power-iteration rounds. The PageRank residual readout is the size of the last change; if it is not tiny, add rounds. Rebuilds the analysis graph.'
    },
    {
      kind: 'slider',
      id: 'propagationRounds',
      label: 'Propagation rounds',
      group: 'Communities',
      apply: 'compile',
      min: 4,
      max: 64,
      step: 4,
      default: 32,
      help: 'Synchronous majority-vote rounds of label propagation. The status readout says whether the last round still changed labels. Rebuilds the analysis graph.'
    },
    {
      kind: 'select',
      id: 'communityMethod',
      label: 'Community method',
      group: 'Communities',
      apply: 'param',
      default: 'modularity',
      help: 'Label propagation (unweighted majority vote, no objective) or modularity optimization started from the propagation result. Falls back to propagation until the optimizer has reported.',
      options: [
        {value: 'propagation', label: 'Label propagation'},
        {value: 'modularity', label: 'Modularity optimization'}
      ]
    },
    {
      kind: 'slider',
      id: 'resolution',
      label: 'Resolution',
      group: 'Communities',
      apply: 'compile',
      min: 0.2,
      max: 3,
      step: 0.1,
      default: 1,
      format: value => value.toFixed(1),
      help: 'Modularity resolution gamma: below 1 favours fewer, larger communities, above 1 many small ones. A shader constant, so each value is its own compiled graph.'
    },
    {
      kind: 'select',
      id: 'rounds',
      label: 'Optimizer rounds',
      group: 'Communities',
      apply: 'compile',
      default: '128',
      help: 'Budget of single-vertex moves (one per round). More rounds can improve the score further; the optimizer status says whether it stopped at a local optimum. Rebuilds the optimizer and the sweep.',
      options: [
        {value: '64', label: '64'},
        {value: '128', label: '128'},
        {value: '256', label: '256'},
        {value: '512', label: '512'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumGain',
      label: 'Minimum gain',
      group: 'Communities',
      apply: 'compile',
      min: 0,
      max: 0.001,
      step: 0.00005,
      default: 0,
      format: value => value.toExponential(1),
      help: 'A move is accepted only if it raises modularity by more than this. Larger values stop the optimizer earlier, at a coarser partition. Rebuilds the optimizer and the sweep.'
    },
    {
      kind: 'button',
      id: 'rerunSweep',
      label: 'Rerun resolution sweep',
      group: 'Communities',
      help: 'Recomputes the modularity-versus-resolution chart, one compiled graph per resolution.'
    },
    {
      kind: 'slider',
      id: 'morph',
      label: 'Morph',
      group: 'Layout',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0,
      format: value =>
        value === 0 ? 'geography' : value === 1 ? 'force layout' : value.toFixed(2),
      help: 'Blend between real airport positions (0) and the force-layout positions (1), fitted to the map. A parameter-buffer write.'
    },
    {
      kind: 'toggle',
      id: 'animateMorph',
      label: 'Animate morph',
      group: 'Layout',
      apply: 'param',
      default: false,
      help: 'Ping-pongs the morph slider between geography and the layout.'
    },
    {
      kind: 'slider',
      id: 'morphSeconds',
      label: 'Morph leg',
      group: 'Layout',
      apply: 'param',
      min: 2,
      max: 20,
      step: 1,
      default: 8,
      unit: 's',
      help: 'Seconds for one leg of the animated morph.'
    },
    {
      kind: 'toggle',
      id: 'layoutRunning',
      label: 'Run layout',
      group: 'Layout',
      apply: 'param',
      default: true,
      help: 'Advance the force simulation each frame. Turn it off to freeze the layout and inspect it.'
    },
    {
      kind: 'select',
      id: 'layoutMode',
      label: 'Repulsion',
      group: 'Layout',
      apply: 'compile',
      default: 'exact',
      help: 'Exact all-pairs repulsion, or the spatial grid that treats distant cells as one mass (faster at large sizes, approximate). If an airport leaves the grid bounds the spatial layout freezes; the layout status says so. Rebuilds the layout graph.',
      options: [
        {value: 'exact', label: 'Exact all-pairs (GPUGraphForceLayout)'},
        {value: 'spatial', label: 'Spatial approximation (GPUGraphSpatialForceLayout)'}
      ]
    },
    {
      kind: 'slider',
      id: 'theta',
      label: 'Opening angle theta',
      group: 'Layout',
      apply: 'compile',
      min: 0,
      max: 1.5,
      step: 0.1,
      default: 0.6,
      format: value => value.toFixed(1),
      disabledWhen: state => state.layoutMode !== 'spatial',
      help: 'How far a grid cell must be before it counts as one mass. Zero is exact; larger is faster and rougher. Rebuilds the layout graph.'
    },
    {
      kind: 'slider',
      id: 'repulsion',
      label: 'Repulsion strength',
      group: 'Layout',
      apply: 'compile',
      min: 0.00005,
      max: 0.002,
      step: 0.00005,
      default: 0.0002,
      format: value => value.toExponential(1),
      help: 'How hard airports push each other apart. More spreads the layout out. Rebuilds the layout graph and keeps its positions.'
    },
    {
      kind: 'slider',
      id: 'attraction',
      label: 'Route spring strength',
      group: 'Layout',
      apply: 'compile',
      min: 0.005,
      max: 0.15,
      step: 0.005,
      default: 0.045,
      format: value => value.toFixed(3),
      help: 'How hard each route pulls its two airports together. More tightens communities. Rebuilds the layout graph.'
    },
    {
      kind: 'slider',
      id: 'gravity',
      label: 'Gravity',
      group: 'Layout',
      apply: 'compile',
      min: 0,
      max: 0.1,
      step: 0.005,
      default: 0.025,
      format: value => value.toFixed(3),
      help: 'Pull towards the centre that keeps disconnected pieces from drifting away. Rebuilds the layout graph.'
    },
    {
      kind: 'slider',
      id: 'layoutDamping',
      label: 'Velocity damping',
      group: 'Layout',
      apply: 'compile',
      min: 0.5,
      max: 0.98,
      step: 0.01,
      default: 0.85,
      format: value => value.toFixed(2),
      help: 'Share of velocity kept each step. Higher glides longer and can oscillate. Rebuilds the layout graph.'
    },
    {
      kind: 'slider',
      id: 'maxVelocity',
      label: 'Speed limit',
      group: 'Layout',
      apply: 'compile',
      min: 0.01,
      max: 0.2,
      step: 0.005,
      default: 0.045,
      format: value => value.toFixed(3),
      help: 'Largest movement per step, in layout units. Lower is calmer but settles slower. Rebuilds the layout graph.'
    },
    {
      kind: 'select',
      id: 'iterationsPerFrame',
      label: 'Steps per frame',
      group: 'Layout',
      apply: 'compile',
      default: '4',
      help: 'Force steps encoded per frame. More settles faster and costs GPU time (exact repulsion is quadratic in airports). Rebuilds the layout graph.',
      options: [
        {value: '1', label: '1'},
        {value: '2', label: '2'},
        {value: '4', label: '4'},
        {value: '8', label: '8'}
      ]
    },
    {
      kind: 'toggle',
      id: 'autoFit',
      label: 'Fit layout to the map',
      group: 'Layout',
      apply: 'param',
      default: true,
      help: 'Scales and centres the layout from its measured extent so it fills the map. Turn it off to use the manual scale.'
    },
    {
      kind: 'slider',
      id: 'layoutScale',
      label: 'Manual layout scale',
      group: 'Layout',
      apply: 'param',
      min: 5,
      max: 150,
      step: 5,
      default: 40,
      unit: 'deg/unit',
      disabledWhen: state => state.autoFit,
      help: 'Degrees of map per layout unit when the fit is off.'
    },
    {
      kind: 'button',
      id: 'resetLayout',
      label: 'Reset layout',
      group: 'Layout',
      help: 'Re-seeds every airport at a deterministic random position and starts the simulation again.'
    },
    {
      kind: 'slider',
      id: 'bundleIterations',
      label: 'Bundle iterations',
      group: 'Bundling',
      apply: 'param',
      min: 0,
      max: 32,
      step: 1,
      default: 15,
      format: value => (value === 0 ? '0 (straight lines)' : String(value)),
      disabledWhen: state => state.view !== 'bundles',
      help: 'How many advect, resample and smooth rounds run. A parameter write; the compiled maximum is 32.'
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
      disabledWhen: state => state.view !== 'bundles',
      help: 'Initial attraction radius as a fraction of the map. Larger merges farther-apart routes into thicker bundles.'
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
      disabledWhen: state => state.view !== 'bundles',
      help: 'The radius is multiplied by this every iteration. Lower anneals quickly.'
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
      disabledWhen: state => state.view !== 'bundles',
      help: 'Strength of the smoothing after each step. High gives smooth arcs; zero leaves kinks.'
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
      disabledWhen: state => state.view !== 'bundles',
      help: 'Multiplier on the advection step. Larger converges faster and can overshoot.'
    },
    {
      kind: 'select',
      id: 'pointsPerEdge',
      label: 'Control points per route',
      group: 'Bundling',
      apply: 'compile',
      default: '16',
      disabledWhen: state => state.view !== 'bundles',
      help: 'Polyline resolution of each bundled route. More follows tighter bends and costs memory. Rebuilds the bundling graph.',
      options: [
        {value: '8', label: '8 (coarse)'},
        {value: '16', label: '16'},
        {value: '24', label: '24 (fine)'}
      ]
    },
    {
      kind: 'select',
      id: 'densityResolution',
      label: 'Density grid',
      group: 'Bundling',
      apply: 'compile',
      default: '256',
      disabledWhen: state => state.view !== 'bundles',
      help: 'Cells per axis of the density field the control points splat onto. Rebuilds the bundling graph.',
      options: [
        {value: '128', label: '128 x 128'},
        {value: '256', label: '256 x 256'},
        {value: '512', label: '512 x 512'}
      ]
    }
  ],

  readouts: [
    {id: 'airports', label: 'Airports', format: 'integer'},
    {
      id: 'edges',
      label: 'Routes shown',
      help: 'Route pairs that pass the display filters, of all 18,930.'
    },
    {
      id: 'engineStatus',
      label: 'Graph status',
      help: 'Whether the compressed adjacency was complete and every partition passed validation.'
    },
    {id: 'topHubs', label: 'PageRank top 8', help: 'Highest PageRank first.'},
    {id: 'topByDegree', label: 'Connections top 8', help: 'Most connections first.'},
    {
      id: 'rankAgreement',
      label: 'PageRank vs connections',
      format: 'decimal',
      help: 'Spearman rank correlation of PageRank and degree over all airports. Near 1 means they order airports alike.'
    },
    {id: 'medianDegree', label: 'Median connections', format: 'integer'},
    {id: 'maxDegree', label: 'Most connections', format: 'integer'},
    {
      id: 'leafShare',
      label: 'Dead-end airports',
      format: 'percent',
      help: 'Share of airports with exactly one connection.'
    },
    {
      id: 'residual',
      label: 'PageRank residual',
      help: 'L1 change of the last PageRank iteration. Small means the ranking has settled.'
    },
    {
      id: 'degeneracy',
      label: 'Deepest core',
      format: 'integer',
      help: 'The largest k for which a k-core exists: the airports in it all have at least k partners inside it.'
    },
    {id: 'coreStatus', label: 'Core numbers'},
    {id: 'propagationStatus', label: 'Label propagation'},
    {
      id: 'communities',
      label: 'Communities',
      format: 'integer',
      help: 'Distinct labels in the chosen partition, including single-airport groups.'
    },
    {
      id: 'purity',
      label: 'Continent purity',
      format: 'percent',
      help: 'Share of airports that sit on the dominant continent of their community. 100% means communities never cross continents.'
    },
    {id: 'withinShare', label: 'Routes within communities', format: 'percent'},
    {
      id: 'qCommunities',
      label: 'Modularity: communities',
      format: 'decimal',
      help: 'Newman modularity of the chosen partition (propagation at gamma 1, optimized at the Resolution slider). Higher keeps more routes inside groups than a degree-matched random network would.'
    },
    {
      id: 'qContinents',
      label: 'Modularity: continents',
      format: 'decimal',
      help: 'The same score for the six continents at the same resolution, the baseline communities have to beat.'
    },
    {id: 'optimizerStatus', label: 'Optimizer'},
    {id: 'communityList', label: 'Largest communities', layout: 'block'},
    {id: 'sweepStatus', label: 'Resolution sweep'},
    {
      id: 'layoutCorrelation',
      label: 'Edge length correlation',
      format: 'decimal',
      help: "Pearson correlation between each route's length on the map and in the force layout. Near 1: geography explains the layout. Near 0: the layout ignores distance."
    },
    {id: 'layoutSteps', label: 'Layout steps', format: 'integer'},
    {id: 'layoutStatus', label: 'Layout status'},
    {id: 'arcVertices', label: 'Arc vertices'},
    {id: 'pageRankChart', label: 'PageRank top 20', kind: 'chart'},
    {id: 'degreeChart', label: 'Connections per airport', kind: 'chart'},
    {id: 'modularityChart', label: 'Modularity versus resolution', kind: 'chart'}
  ],

  legends: (state, data) => {
    const groups = data.groups as
      | {entries: readonly {color: readonly [number, number, number, number]; label: string}[]}
      | undefined;
    const legends: LegendSpec[] = [
      {
        kind: 'categories',
        title: state.colorBy === 'continent' ? 'Continent' : 'Community (named by its top airport)',
        entries: [
          ...(groups?.entries ?? []),
          {color: NETWORK_PALETTE[7], label: 'Route between two groups'}
        ],
        note: 'A route takes its group colour when both airports share one.'
      }
    ];
    if (state.sizeBy !== 'uniform') {
      legends.push({
        kind: 'size',
        title: SIZE_TITLES[state.sizeBy],
        entries: [
          {radiusPixels: 3, label: 'low'},
          {radiusPixels: Math.max(5, Math.round(state.maxRadius / 2)), label: 'middle'},
          {radiusPixels: state.maxRadius, label: 'highest'}
        ]
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph, GPUGraphTopology, GPUGraphDegree, GPUGraphPageRank, GPUGraphLabelPropagation,
  GPUGraphModularity, GPUGraphModularityOptimization, GPUGraphForceLayout
} from '@luma.gl/gpgpu/gpu-graph';

// 3,257 airports, 18,930 pairs: an undirected graph over borrowed GPU columns.
const graph = new GPUGraph({vertexCount, sourceVertices, targetVertices, directed: false});
const topology = new GPUGraphTopology({graph, forward /* symmetric CSR, 2 x pairs */, invalidEdgeCount});

const analysis = new GPUCommandGraph(device, {id: 'airline-analysis'});
topology.addToGraph(analysis);
new GPUGraphDegree({topology, output: degrees}).addToGraph(analysis);
new GPUGraphPageRank({
  topology, output: pageRank, damping: ${state.pageRankDamping}, iterations: ${state.pageRankIterations}
}).addToGraph(analysis);
new GPUGraphLabelPropagation({
  topology, output: communities, iterations: ${state.propagationRounds}
}).addToGraph(analysis);
new GPUGraphModularity({graph, communities, output: modularity}).addToGraph(analysis);
analysis.compile().encode(commandEncoder, {parameters: undefined});

// Improve the propagation result; resolution and rounds are shader constants.
new GPUGraphModularityOptimization({
  topology, output: improved, modularity: score, initialCommunities: communities,
  resolution: ${state.resolution}, iterations: ${state.rounds}, minimumGain: ${state.minimumGain}
}).addToGraph(optimization);

// The positions buffer is also a vertex buffer (usage STORAGE | VERTEX).
new GPUGraphForceLayout({
  topology, positions, velocities, reset,
  iterationsPerFrame: ${state.iterationsPerFrame}, repulsion: ${state.repulsion},
  attraction: ${state.attraction}, gravity: ${state.gravity}, damping: ${state.layoutDamping},
  maxVelocity: ${state.maxVelocity}
}).addToGraph(frameGraph);   // encoded every frame, warm-started`,

  about: {
    what: 'This scene uses the `@luma.gl/gpgpu/gpu-graph` classes directly. `GPUGraph` and `GPUGraphTopology` describe the airline routes as an undirected graph with GPU-built adjacency; `GPUGraphDegree`, `GPUGraphPageRank` and `GPUGraphCoreNumber` score airports; `GPUGraphLabelPropagation`, `GPUGraphModularityOptimization` and `GPUGraphModularity` find and score communities; `GPUGraphForceLayout` (or the approximate `GPUGraphSpatialForceLayout`) positions airports by connection. Routes are drawn with `GPUGreatCircleArcs` or `GPUEdgeBundling`.',
    why: 'A flight map shows geography; a graph shows structure. Ranking, communities and layout answer which airports hold the network together, whether regions are really separate systems, and where geography misleads.',
    howToRead:
      'Disc size is the chosen centrality measure. Colour is the community (or continent); routes between two groups are red-orange. The charts give the PageRank top 20, the degree distribution on log axes, and modularity against resolution. OpenFlights is a 2014 snapshot with one edge per airport pair, not weighted by frequency.'
  },

  create: async ctx => (await import('./airline-network.compute')).createAirlineNetwork(ctx),

  story: storyFromMarkdown<AirlineNetworkOptions>(narrative, {
    backbone: {
      controls: ['sizeBy', 'pageRankDamping'],
      readouts: ['topHubs', 'topByDegree', 'pageRankChart', 'degreeChart'],
      camera: {...WORLD_VIEW, transitionMs: 1200},
      options: {view: 'arcs', colorBy: 'continent', sizeBy: 'pagerank', edgeOpacity: 0.3},
      callout: {coordinate: [-84.43, 33.64], text: 'Atlanta (ATL)'}
    },
    communities: {
      controls: ['colorBy', 'communityMethod', 'propagationRounds'],
      readouts: ['communities', 'purity', 'qCommunities', 'qContinents', 'communityList'],
      options: {colorBy: 'community', communityMethod: 'propagation', edgeOpacity: 0.4},
      callout: {coordinate: [28.82, 41.26], text: 'Istanbul (IST)'},
      highlight: {readout: 'purity'}
    },
    resolution: {
      controls: ['resolution', 'rounds', 'minimumGain'],
      readouts: ['qCommunities', 'qContinents', 'optimizerStatus', 'modularityChart'],
      options: {communityMethod: 'modularity'},
      highlight: {readout: 'qCommunities'}
    },
    bundles: {
      controls: ['edgeFilter', 'bundleIterations', 'kernelRadius'],
      readouts: ['edges', 'withinShare'],
      camera: {...WORLD_VIEW, transitionMs: 1200},
      options: {view: 'bundles', edgeOpacity: 0.5, edgeFilter: 'all'}
    },
    morph: {
      controls: ['animateMorph', 'morph', 'layoutRunning', 'repulsion'],
      readouts: ['layoutCorrelation', 'layoutSteps', 'layoutStatus'],
      options: {view: 'morph', edgeFilter: 'all', animateMorph: true, edgeOpacity: 0.3},
      highlight: {readout: 'layoutCorrelation'}
    },
    limits: {
      controls: ['layoutMode', 'minRecords', 'pageRankIterations'],
      readouts: ['residual', 'coreStatus', 'propagationStatus'],
      camera: {...WORLD_VIEW, transitionMs: 1000},
      options: {view: 'arcs', animateMorph: false, morph: 0, edgeOpacity: 0.5, colorBy: 'community'}
    }
  })
});
