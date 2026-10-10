// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {WORLD, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatCount} from '../../cartography/live-text';
import {getSizeLegendEntries} from '../../cartography/proportional';
import {GLOBAL_FURNITURE, mercatorCaveat} from '../../cartography/projection-notes';
import {defineScene} from '../scene';
import type {LegendSpec} from '../scene';
import type {AirlineNetworkOptions, NetworkLegendData} from './airline-network.compute';
import {NEUTRAL_NODE_INK} from './airline-network-palette';
import {FLOW_CREDITS} from './flows-style';

const WORLD_VIEW = {longitude: 10, latitude: 26, zoom: 1.75};

/** Orientation names of the map steps: the dark ground carries no basemap labels. */
const OCEANS = labelsFor(WORLD, ['atlantic-ocean', 'pacific-ocean', 'indian-ocean'], {
  'atlantic-ocean': {tone: 'muted'},
  'pacific-ocean': {tone: 'muted'},
  'indian-ocean': {tone: 'muted'}
});

/** The cartouche of one step: the claim, the variable and method, and the vintage chip. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Frozen 2014'] as const
});

const SIZE_LEGENDS: Record<
  Exclude<AirlineNetworkOptions['sizeBy'], 'uniform'>,
  {title: string; unit: string; scale: number}
> = {
  pagerank: {title: 'Disc area ~ PageRank', unit: 'PageRank x 1000', scale: 1000},
  degree: {title: 'Disc area ~ connections', unit: 'connections', scale: 1},
  core: {title: 'Disc area ~ core number', unit: 'core number', scale: 1},
  bridges: {title: 'Disc area ~ bridge routes', unit: 'routes between groups', scale: 1}
};

export default defineScene<AirlineNetworkOptions>({
  id: 'airline-network',
  title: 'The airline network as a graph',
  chapter: 'flows',
  order: 6,
  summary:
    'PageRank, core numbers, communities and modularity on the OpenFlights route graph: who holds the network together, whether its communities follow the continents, which routes bridge them, and what a force layout makes of it once geography is taken away.',
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
      display: 'segmented',
      help: 'Great-circle arcs on the map, edge bundles on the map, or the morph between the map and the force layout. Bundling and the layout only run while their view is open.',
      options: [
        {value: 'arcs', label: 'Arcs'},
        {value: 'bundles', label: 'Bundles'},
        {value: 'morph', label: 'Morph'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour groups by',
      group: 'Display',
      apply: 'param',
      default: 'community',
      display: 'segmented',
      help: 'No colour (one neutral ink), the six continents, or the communities the graph algorithm found. Communities wear the hue of the continent they overlap most; groups beyond the seven largest are grey, and routes between two groups are off-white.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'continent', label: 'Continent'},
        {value: 'community', label: 'Community'}
      ]
    },
    {
      kind: 'select',
      id: 'sizeBy',
      label: 'Disc size by',
      group: 'Display',
      apply: 'param',
      default: 'pagerank',
      display: 'chips',
      help: 'The measure that sizes each airport disc: disc area is proportional to the value. Connections count distinct neighbours, PageRank also weighs how well connected those are, core number is the deepest k-core the airport belongs to, and bridge routes counts the routes that leave its group.',
      options: [
        {value: 'degree', label: 'Connections'},
        {value: 'pagerank', label: 'PageRank'},
        {value: 'core', label: 'Core number'},
        {value: 'bridges', label: 'Bridge routes'},
        {value: 'uniform', label: 'Same size'}
      ]
    },
    {
      kind: 'select',
      id: 'labels',
      label: 'Airport labels',
      group: 'Display',
      apply: 'param',
      default: 'hubs',
      help: 'Which airports are named on the map, picked from the data: the largest by the size metric, the top airport of each large group, those groups plus the one that crosses continents most, or the airports that carry the most bridge routes.',
      options: [
        {value: 'hubs', label: 'Largest airports'},
        {value: 'groups', label: 'Hub of each group'},
        {value: 'disagreement', label: 'Groups and where they cross'},
        {value: 'bridges', label: 'Bridge airports'},
        {value: 'none', label: 'None'}
      ]
    },
    {
      kind: 'toggle',
      id: 'ego',
      label: 'Click an airport to isolate its routes',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'A click picks the nearest airport on the CPU: its routes light up at full strength, its neighbours are ringed and everything else dims. Click it again, or empty sea, to clear.'
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
      help: 'Radius of the most important airport; everything else scales from it by the square root, so area stays proportional.'
    },
    {
      kind: 'slider',
      id: 'edgeOpacity',
      label: 'Route brightness',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 2,
      step: 0.1,
      default: 1,
      marks: [{value: 1, label: 'default'}],
      help: 'Routes are thin lines of light that add up where they overlap. Lower it to cut the glow in dense regions; raise it to see the sparse ones.'
    },
    {
      kind: 'select',
      id: 'edgeFilter',
      label: 'Routes shown',
      group: 'Display',
      apply: 'param',
      default: 'all',
      display: 'segmented',
      help: 'Keep every route, only routes inside a group, or only the routes that run between two groups. A display filter: the graph algorithms always see every route.',
      options: [
        {value: 'all', label: 'All'},
        {value: 'within', label: 'Within groups'},
        {value: 'between', label: 'Between groups'}
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
      expert: true,
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
      default: 500,
      unit: 'km',
      expert: true,
      disabledWhen: state => state.view !== 'arcs',
      help: 'Longest straight piece of a great-circle arc. Shorter pieces follow the curve more closely and cost more vertices (the arc readout shows the total).'
    },
    {
      kind: 'slider',
      id: 'pageRankDamping',
      label: 'PageRank damping',
      group: 'Ranking',
      apply: 'compile',
      min: 0.05,
      max: 0.99,
      step: 0.01,
      default: 0.85,
      marks: [{value: 0.85, label: 'usual'}],
      format: value => value.toFixed(2),
      describe: value =>
        value > 0.95
          ? 'close to the connection order; needs more rounds to settle'
          : value < 0.3
            ? 'close to a uniform ranking'
            : 'a mix of connections and who they connect to',
      help: 'Probability of following a route instead of jumping to a random airport. On an undirected network, damping near 1 gives the connection order and near 0 gives every airport the same score. A shader constant: the graph rebuilds once you stop dragging.'
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
      expert: true,
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
      expert: true,
      help: 'Synchronous majority-vote rounds of label propagation. The status readout says whether the last round still changed labels. Rebuilds the analysis graph.'
    },
    {
      kind: 'select',
      id: 'communityMethod',
      label: 'Community method',
      group: 'Communities',
      apply: 'param',
      default: 'modularity',
      display: 'segmented',
      help: 'Label propagation (unweighted majority vote, no objective) or modularity optimization started from the propagation result. Falls back to propagation until the optimizer has reported.',
      options: [
        {value: 'propagation', label: 'Label propagation'},
        {value: 'modularity', label: 'Modularity'}
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
      marks: [{value: 1, label: 'standard'}],
      danger: [2.3, 3],
      format: value => value.toFixed(1),
      describe: value =>
        value < 1
          ? 'merges communities: fewer, larger'
          : value > 1
            ? 'splits communities: more, smaller'
            : 'standard modularity',
      help: 'Modularity resolution gamma: below 1 favours fewer, larger communities, above 1 many small ones. A shader constant, so each value is its own compiled graph; it builds once you stop dragging.'
    },
    {
      kind: 'select',
      id: 'rounds',
      label: 'Optimizer rounds',
      group: 'Communities',
      apply: 'compile',
      default: '128',
      expert: true,
      help: 'Budget of single-vertex moves (one per round). More rounds can improve the score further; the optimizer status says whether it stopped at a local optimum. Rebuilds the optimizer and clears the optional sweep.',
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
      expert: true,
      format: value => value.toExponential(1),
      help: 'A move is accepted only if it raises modularity by more than this. Larger values stop the optimizer earlier, at a coarser partition. Rebuilds the optimizer and clears the optional sweep.'
    },
    {
      kind: 'button',
      id: 'rerunSweep',
      label: 'Rerun resolution sweep',
      group: 'Communities',
      expert: true,
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
      expert: true,
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
      default: 'spatial',
      expert: true,
      help: 'The spatial grid treats distant cells as one mass and keeps this 3,257-airport layout interactive. Exact all-pairs repulsion is available for comparison but is substantially more expensive. If an airport leaves the grid bounds the spatial layout freezes; the layout status says so. Rebuilds the layout graph.',
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
      expert: true,
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
      expert: true,
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
      expert: true,
      format: value => value.toFixed(3),
      help: 'Largest movement per step, in layout units. Lower is calmer but settles slower. Rebuilds the layout graph.'
    },
    {
      kind: 'select',
      id: 'iterationsPerFrame',
      label: 'Steps per frame',
      group: 'Layout',
      apply: 'compile',
      default: '2',
      expert: true,
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
      expert: true,
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
      expert: true,
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
      help: 'How many advect, resample and smooth rounds run. A parameter write; the compiled maximum is 32. The bridges step fixes it at the default.'
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
      expert: true,
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
      expert: true,
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
    {id: 'airports', label: 'Airports', format: 'integer', emphasis: 'tile'},
    {
      id: 'routes',
      label: 'Route pairs',
      format: 'integer',
      emphasis: 'tile',
      help: 'Distinct airport pairs in the dataset; both directions of a route count once.'
    },
    {
      id: 'edges',
      label: 'Routes shown',
      help: 'Route pairs that pass the display filters, of every pair in the dataset.'
    },
    {id: 'topHub', label: 'Top airport by PageRank'},
    {
      id: 'hubShare',
      label: 'Routes touching the 10 best-connected airports',
      format: 'percent',
      help: 'Share of all route pairs with at least one end at one of the ten airports with the most connections.'
    },
    {
      id: 'rankAgreement',
      label: 'PageRank vs connections (Spearman)',
      format: 'decimal',
      help: 'Spearman rank correlation of PageRank and degree over all airports. Near 1 means they order airports alike.'
    },
    {
      id: 'communities',
      label: 'Communities',
      format: 'integer',
      help: 'Distinct labels in the chosen partition, including single-airport groups.'
    },
    {
      id: 'purity',
      label: 'Airports on their community’s main continent',
      format: 'percent',
      help: 'Share of airports that sit on the dominant continent of their community. 100% means communities never cross continents.'
    },
    {
      id: 'qContinents',
      label: 'Modularity Q: continents',
      format: 'decimal',
      help: 'Newman modularity of the six continents at the same resolution: how many more routes stay inside the continents than a degree-matched random network would give. Newman and Girvan (2004) report that real networks typically score between about 0.3 and 0.7.'
    },
    {
      id: 'qCommunities',
      label: 'Modularity Q: communities',
      format: 'decimal',
      help: 'Newman modularity of the chosen partition (propagation at gamma 1, optimized at the Resolution slider). Higher keeps more routes inside groups than a degree-matched random network would.'
    },
    {id: 'resolutionNow', label: 'Resolution gamma'},
    {
      id: 'betweenShare',
      label: 'Routes between two groups',
      format: 'percent',
      help: 'Share of route pairs whose two airports are in different groups of the partition the map shows.'
    },
    {
      id: 'bridgeAirports',
      label: 'Top bridge airports',
      help: 'The airports with the most routes that leave their group, most first.'
    },
    {
      id: 'layoutCorrelation',
      label: 'Route length: map vs layout (Pearson)',
      format: 'decimal',
      help: "Pearson correlation between each route's length on the map and in the force layout. One means the layout reproduces geography; zero means the layout ignores distance."
    },
    {id: 'layoutSteps', label: 'Layout steps', format: 'integer', hood: true},
    {id: 'layoutStatus', label: 'Layout status', hood: true},
    {id: 'engineStatus', label: 'Graph status', hood: true},
    {id: 'topHubs', label: 'PageRank top 8', hood: true, help: 'Highest PageRank first.'},
    {id: 'topByDegree', label: 'Connections top 8', hood: true, help: 'Most connections first.'},
    {id: 'medianDegree', label: 'Median connections', format: 'integer', hood: true},
    {id: 'maxDegree', label: 'Most connections', format: 'integer', hood: true},
    {
      id: 'leafShare',
      label: 'Dead-end airports',
      format: 'percent',
      hood: true,
      help: 'Share of airports with exactly one connection.'
    },
    {
      id: 'residual',
      label: 'PageRank residual',
      hood: true,
      help: 'L1 change of the last PageRank iteration. Small means the ranking has settled.'
    },
    {
      id: 'degeneracy',
      label: 'Deepest core',
      format: 'integer',
      hood: true,
      help: 'The largest k for which a k-core exists: the airports in it all have at least k partners inside it.'
    },
    {id: 'coreStatus', label: 'Core numbers', hood: true},
    {id: 'propagationStatus', label: 'Label propagation', hood: true},
    {id: 'optimizerStatus', label: 'Optimizer', hood: true},
    {id: 'withinShare', label: 'Routes within communities', format: 'percent', hood: true},
    {id: 'communityList', label: 'Largest communities', layout: 'block', hood: true},
    {id: 'sweepStatus', label: 'Resolution sweep', hood: true},
    {id: 'arcVertices', label: 'Arc vertices', hood: true},
    {id: 'pageRankChart', label: 'PageRank top 20', kind: 'chart'},
    {id: 'degreeChart', label: 'Connections per airport', kind: 'chart'},
    {id: 'modularityChart', label: 'Modularity versus resolution', kind: 'chart'}
  ],

  pipeline: [
    {
      id: 'arcs',
      label: 'Great circles',
      detail: 'Every pair is sampled along its great circle on the GPU and drawn from that buffer',
      show: {option: 'view', value: 'arcs'}
    },
    {
      id: 'pagerank',
      label: 'PageRank',
      detail:
        'Power iteration over the CSR adjacency: pull from neighbours, ping-pong buffers, residual reduction'
    },
    {
      id: 'propagation',
      label: 'Label propagation',
      detail: 'Every airport adopts its neighbours’ most common label; ties go to the smaller label'
    },
    {
      id: 'optimise',
      label: 'Modularity moves',
      detail: 'One accepted vertex move per round: a polish of the propagation result, not Louvain'
    },
    {
      id: 'bundle',
      label: 'Bundling',
      detail: 'Kernel-density edge bundling of the routes, with fixed settings',
      show: {option: 'view', value: 'bundles'}
    },
    {
      id: 'layout',
      label: 'Force layout',
      detail:
        'Repulsion, route springs and gravity; the positions buffer is also the vertex buffer',
      show: {option: 'view', value: 'morph'}
    }
  ],

  legends: (state, data) => {
    const info = data.network as NetworkLegendData | undefined;
    const legends: LegendSpec[] = [];
    const between = {
      color: info?.between ?? [244, 241, 232, 255],
      label: 'Route between two groups',
      count: info?.betweenCount ?? 0,
      shape: 'line' as const
    };
    const asEntries = (entries: NetworkLegendData['continent']) =>
      entries.map(entry => ({
        color: entry.color,
        label: entry.label,
        count: entry.count,
        shape: 'dot' as const
      }));
    if (state.colorBy === 'none' && !info?.comparing) {
      legends.push({
        kind: 'categories',
        title: 'The airline network',
        entries: [
          {color: NEUTRAL_NODE_INK.dark, label: 'Airport', shape: 'dot'},
          {color: [188, 202, 224, 200], label: 'Route pair', shape: 'line'}
        ],
        layout: 'list'
      });
    } else if (info) {
      const showContinents = info.comparing || state.colorBy === 'continent';
      const showCommunities = info.comparing || state.colorBy === 'community';
      if (showContinents) {
        legends.push({
          kind: 'categories',
          title: 'Continent',
          entries: [...asEntries(info.continent), ...(showCommunities ? [] : [between])],
          layout: 'list',
          note: 'Airports counted by the continent of their country.'
        });
      }
      if (showCommunities) {
        legends.push({
          kind: 'categories',
          title: 'Community (named by its top airport)',
          entries: [...asEntries(info.community), between],
          layout: 'list',
          note: 'A community wears the hue of the continent it overlaps most; grey is every smaller community.'
        });
      }
    }
    if (state.sizeBy !== 'uniform' && info) {
      const metric = SIZE_LEGENDS[state.sizeBy];
      const maximum = info.sizeMaxima[state.sizeBy];
      // Nice values are picked in the scaled unit (PageRank x 1000); area stays proportional.
      const entries = getSizeLegendEntries(maximum * metric.scale, state.maxRadius, {
        count: 3,
        minRadiusPixels: 1.5,
        format: value => formatCount(value)
      });
      legends.push({
        kind: 'size',
        title: metric.title,
        entries,
        layout: 'nested',
        unit: metric.unit,
        color: [0, 0, 0, 0],
        outline: [200, 208, 220, 200]
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph, GPUGraphTopology, GPUGraphDegree, GPUGraphPageRank, GPUGraphLabelPropagation,
  GPUGraphModularity, GPUGraphModularityOptimization, GPUGraphForceLayout
} from '@luma.gl/gpgpu/gpu-graph';

// OpenFlights pairs: an undirected graph over borrowed GPU columns.
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

// Hues follow the continent a community overlaps most (cartography/stable-hues).
const slots = matchByOverlap(continentOfAirport, communityRank, 7);

// The positions buffer is also a vertex buffer (usage STORAGE | VERTEX).
new GPUGraphForceLayout({
  topology, positions, velocities, reset,
  iterationsPerFrame: ${state.iterationsPerFrame}, repulsion: ${state.repulsion},
  attraction: ${state.attraction}, gravity: ${state.gravity}, damping: ${state.layoutDamping},
  maxVelocity: ${state.maxVelocity}
}).addToGraph(frameGraph);   // encoded every frame, warm-started`,

  about: {
    what: 'Previously: [taxi trails](#/story/nyc-taxi-trails). Next: [flight bundling](#/story/flight-bundling).\n\nThis scene uses the `@luma.gl/gpgpu/gpu-graph` classes directly. `GPUGraph` and `GPUGraphTopology` describe the airline routes as an undirected graph with GPU-built adjacency; `GPUGraphDegree`, `GPUGraphPageRank` and `GPUGraphCoreNumber` score airports; `GPUGraphLabelPropagation`, `GPUGraphModularityOptimization` and `GPUGraphModularity` find and score communities; `GPUGraphForceLayout` (or the approximate `GPUGraphSpatialForceLayout`) positions airports by connection. Routes are drawn with `GPUGreatCircleArcs` or `GPUEdgeBundling`.',
    why: 'A flight map shows geography; a graph shows structure. Ranking, communities and layout answer which airports hold the network together, whether regions are really separate systems, and where geography misleads.',
    howToRead:
      'Disc area is the chosen centrality measure. Colour is the continent or the community, with identity-stable hues; routes between two groups are off-white and drawn last. OpenFlights is a community-maintained snapshot from 2014 with one unweighted edge per airport pair: no seats, no schedules. Lines are great circles on a Web Mercator map, so they bow towards the poles and the north looks larger than it is.'
  },

  // Night ground: the routes are light. The morph step flattens it (no geography left to show).
  basemap: ground('night'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: cartouche(
      'Who holds the airline network together?',
      'Route pairs as great circles, unweighted'
    ),
    credit: joinCredits(FLOW_CREDITS.openFlights, CREDITS.okabeIto),
    caveat: mercatorCaveat({latitudes: [0, 60], kind: 'area'})
  },

  create: async ctx => (await import('./airline-network.compute')).createAirlineNetwork(ctx),

  story: [
    {
      id: 'hairball',
      title: 'Every route at once',
      headline: 'Every route at once makes a tangle',
      textAlternative:
        'Dark world map covered in thin pale great-circle routes that glow where they overlap, with four large airports and the oceans named.',
      body: 'OpenFlights lists **{{routes}}** airport pairs between **{{airports}}** airports. Drawn as great circles in thin pale light, they add up to a glow where they overlap, and nothing in the tangle says which airports matter. Raise or lower **Route brightness** below to see how much the overplotting hides.\n\nWhich airports hold this together?',
      optionsMode: 'fresh',
      options: {view: 'arcs', colorBy: 'none', sizeBy: 'uniform', labels: 'hubs'},
      controls: ['edgeOpacity'],
      readouts: ['airports', 'routes'],
      stage: 'arcs',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('Who holds the airline network together?', 'Route pairs as great circles')
      },
      annotations: OCEANS
    },
    {
      id: 'hubs',
      title: 'A few hubs hold the network together',
      headline: 'A few hubs hold the network together',
      textAlternative:
        'World map of airports as discs sized by PageRank and coloured by continent over thin routes, with the largest airports named.',
      body: 'The ten best-connected airports touch **{{hubShare}}** of all routes, and **{{topHub}}** tops the PageRank. Switch **Disc size by** below: connections, PageRank and core number rank airports differently, and PageRank agrees with connections at **{{rankAgreement}}**. On an undirected network, damping near one gives the connection order; drag **PageRank damping** to see it. Click an airport to isolate its routes.\n\n*Disc area, not radius, carries the value.*',
      optionsMode: 'fresh',
      options: {
        view: 'arcs',
        colorBy: 'continent',
        sizeBy: 'pagerank',
        labels: 'hubs',
        ego: true
      },
      controls: ['sizeBy', 'pageRankDamping'],
      readouts: ['topHub', 'hubShare', 'rankAgreement', 'pageRankChart'],
      stage: 'pagerank',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'A few hubs hold the network together',
          'Disc area: PageRank, connections or core'
        )
      },
      annotations: OCEANS,
      highlight: {readout: 'hubShare'}
    },
    {
      id: 'communities',
      title: 'Communities mostly follow the continents',
      headline: 'Communities mostly follow the continents',
      textAlternative:
        'World map split by a vertical divider: continents coloured on the left, communities found by label propagation on the right, nearly the same hues.',
      body: 'Label propagation knows nothing about maps: each airport adopts its neighbours’ commonest label. Drag the divider: continents left, communities right. Modularity scores both, **{{qContinents}}** for continents and **{{qCommunities}}** for **{{communities}}** communities; **{{purity}}** of airports sit on their community’s main continent, and the note marks where one does not. The [flight matrix](#/story/flight-matrix) tests this against chance. Switch **Community method** below.',
      optionsMode: 'fresh',
      options: {
        view: 'arcs',
        colorBy: 'community',
        communityMethod: 'propagation',
        sizeBy: 'degree',
        labels: 'disagreement'
      },
      controls: ['communityMethod'],
      readouts: ['qContinents', 'qCommunities', 'communities', 'purity'],
      stage: 'propagation',
      compare: {mode: 'swipe', labels: ['Continents', 'Communities'], position: 0.5},
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('Do communities follow the continents?', 'Modularity of two partitions')
      },
      annotations: OCEANS,
      highlight: {readout: 'qCommunities'}
    },
    {
      id: 'resolution',
      title: 'Resolution decides how many communities exist',
      headline: 'Resolution decides how many communities exist',
      textAlternative:
        'World map of airports coloured by community with a line chart of modularity against the resolution parameter, which the slider and the chart both set.',
      body: 'Modularity has a knob. **Resolution** is gamma, now **{{resolutionNow}}**: lower merges communities, higher splits them. The optimiser finds **{{communities}}** communities scoring **{{qCommunities}}**. Push gamma up until the map fragments; hues follow the continent each group overlaps, so a split keeps its colour. The optimiser makes one move per round, a polish on label propagation rather than Louvain.\n\n*A parameter you choose is a claim you make.*',
      optionsMode: 'fresh',
      options: {
        view: 'arcs',
        colorBy: 'community',
        communityMethod: 'modularity',
        sizeBy: 'degree',
        labels: 'groups',
        resolution: 1
      },
      controls: ['resolution'],
      readouts: ['resolutionNow', 'communities', 'qCommunities', 'modularityChart'],
      stage: 'optimise',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'How many communities, at what resolution?',
          'Modularity optimisation, one gamma'
        )
      },
      annotations: OCEANS,
      highlight: {readout: 'communities'}
    },
    {
      id: 'bridges',
      title: 'Few routes bridge the communities',
      headline: 'Few routes bridge the communities',
      textAlternative:
        'World map with only the routes between communities, bundled into corridors in off-white, over airports coloured by community and sized by bridge routes.',
      body: 'A minority, **{{betweenShare}}**, of routes join two groups. They are drawn alone, bundled with fixed settings as in [flight bundling](#/story/flight-bundling), off-white over the groups they join. Discs size by bridge routes, led by **{{bridgeAirports}}**. Bundles show corridors, not flight paths. Change **Routes shown** below, or compare continents with **Colour groups by**.',
      optionsMode: 'fresh',
      options: {
        view: 'bundles',
        colorBy: 'community',
        communityMethod: 'modularity',
        edgeFilter: 'between',
        sizeBy: 'bridges',
        labels: 'bridges'
      },
      controls: ['edgeFilter', 'colorBy'],
      readouts: ['betweenShare', 'bridgeAirports', 'edges'],
      stage: 'bundle',
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('Which routes bridge the communities?', 'Routes between groups, bundled')
      },
      annotations: OCEANS,
      highlight: {readout: 'betweenShare'}
    },
    {
      id: 'topology',
      title: 'Without geography, the communities pull apart',
      headline: 'Without geography, the communities pull apart',
      textAlternative:
        'Airports and routes on a flat dark ground animate between their map positions and a force layout, where airports of one community clump together.',
      body: 'The force layout ignores longitude and latitude: airports repel, routes pull. Press **Animate morph** or drag **Morph** and watch whether the communities separate. Route length on the map and in the layout correlate at **{{layoutCorrelation}}**; zero would be unrelated, one identical. The ground is flat because no geography is left to show. In All controls, change the disc size or the view.',
      optionsMode: 'fresh',
      options: {
        view: 'morph',
        colorBy: 'community',
        communityMethod: 'modularity',
        sizeBy: 'degree',
        labels: 'none',
        animateMorph: true
      },
      controls: ['morph', 'animateMorph', 'colorBy'],
      readouts: ['layoutCorrelation', 'layoutSteps'],
      stage: 'layout',
      basemap: ground('night', {style: 'none'}),
      camera: {...WORLD_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche('What if geography is taken away?', 'Force layout against real positions')
      },
      highlight: {readout: 'layoutCorrelation'}
    }
  ]
});
