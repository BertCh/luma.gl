// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-communities.md?raw';
import type {BixiCommunitiesOptions} from './bixi-communities.compute';

const COMMUNITY_COLORS = [
  [86, 180, 233, 255],
  [240, 160, 20, 255],
  [214, 110, 170, 255],
  [30, 175, 125, 255],
  [225, 205, 50, 255],
  [225, 95, 40, 255],
  [150, 150, 190, 255]
] as const;

export default defineScene<BixiCommunitiesOptions>({
  id: 'bixi-communities',
  title: 'Communities of riding in Montreal',
  chapter: 'flows',
  order: 10,
  summary:
    'Label propagation and modularity optimization on the BIXI station graph (1.93 million rides, August 2024), compared with the boroughs the stations sit in.',
  contributors: [
    'GPUGraphLabelPropagation',
    'GPUGraphModularityOptimization',
    'GPUGraphModularity'
  ],
  datasets: [{id: 'bixi-flows', role: 'station pairs, August 2024'}],
  initialView: {longitude: -73.61, latitude: 45.53, zoom: 10.7},

  options: [
    {
      kind: 'select',
      id: 'partition',
      label: 'Partition shown',
      group: 'Display',
      apply: 'param',
      default: 'optimized',
      help: 'Which grouping of stations colours the map: the refined communities, the label-propagation proposal, or the boroughs published with the data. All three are always computed and scored.',
      options: [
        {value: 'optimized', label: 'Refined communities (modularity optimization)'},
        {value: 'propagation', label: 'Label propagation'},
        {value: 'boroughs', label: 'Boroughs (as published)'}
      ]
    },
    {
      kind: 'slider',
      id: 'edges',
      label: 'Strongest links drawn',
      group: 'Display',
      apply: 'param',
      min: 200,
      max: 20000,
      step: 100,
      default: 3000,
      help: 'How many of the busiest station pairs are drawn, busiest first. The graph itself always uses every pair that passes the filters.'
    },
    {
      kind: 'toggle',
      id: 'showBetween',
      label: 'Between communities',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the faint grey links that cross a community boundary. Turn off to see only the inside of each group.'
    },
    {
      kind: 'slider',
      id: 'edgeWidth',
      label: 'Link width',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1.4,
      unit: 'px',
      help: 'Line width in pixels. Busier pairs are more opaque, not wider.'
    },
    {
      kind: 'slider',
      id: 'edgeOpacity',
      label: 'Link opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.75,
      help: 'Lower it when thousands of links overlap downtown.'
    },
    {
      kind: 'slider',
      id: 'stationSize',
      label: 'Station size',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 9,
      step: 0.5,
      default: 4,
      unit: 'px',
      help: 'Radius of the station dots.'
    },
    {
      kind: 'slider',
      id: 'neighbors',
      label: 'Strongest links per station',
      group: 'Graph',
      apply: 'param',
      min: 2,
      max: 60,
      step: 1,
      default: 8,
      help: 'An edge stays in the graph when it is among this many strongest links of either of its stations. Dense graphs make label propagation collapse into one blob; a sparse backbone keeps the groups apart. A buffer write: no rebuild.'
    },
    {
      kind: 'slider',
      id: 'minRides',
      label: 'Minimum rides on a pair',
      group: 'Graph',
      apply: 'param',
      min: 1,
      max: 200,
      step: 1,
      default: 1,
      unit: 'rides',
      help: 'Drops pairs with fewer rides in August. Filtered edges get an out-of-domain endpoint, which the graph contributors ignore.'
    },
    {
      kind: 'select',
      id: 'weighting',
      label: 'Edge weighting',
      group: 'Graph',
      apply: 'param',
      default: 'rides',
      help: 'How a pair counts in the modularity objective and its score: by rides, by the square root of rides (evens out the downtown giants) or equally. Label propagation ignores weights.',
      options: [
        {value: 'rides', label: 'Rides'},
        {value: 'sqrt', label: 'Square root of rides'},
        {value: 'equal', label: 'Equal'}
      ]
    },
    {
      kind: 'slider',
      id: 'propagationRounds',
      label: 'Label propagation rounds',
      group: 'Algorithm',
      apply: 'compile',
      min: 1,
      max: 100,
      step: 1,
      default: 32,
      help: 'Synchronous voting rounds, all encoded without early exit. A compile-time constant, so changing it rebuilds the graph.'
    },
    {
      kind: 'slider',
      id: 'optimizeRounds',
      label: 'Refinement rounds',
      group: 'Algorithm',
      apply: 'compile',
      min: 0,
      max: 1024,
      step: 32,
      default: 512,
      help: 'Rounds of modularity optimization; each accepts only the single best station move. Zero keeps the propagation partition. Compile-time.'
    },
    {
      kind: 'slider',
      id: 'resolution',
      label: 'Resolution',
      group: 'Algorithm',
      apply: 'compile',
      min: 0.25,
      max: 3,
      step: 0.25,
      default: 1,
      help: 'Modularity resolution gamma. Below 1 favours few large communities, above 1 many small ones. It scales the null-model term, so it is a shader constant and rebuilds the graph.'
    },
    {
      kind: 'select',
      id: 'minimumGain',
      label: 'Minimum gain',
      group: 'Algorithm',
      apply: 'compile',
      default: '0',
      help: 'A move must raise modularity by more than this to be accepted. Larger values stop earlier and keep the partition closer to the propagation.',
      options: [
        {value: '0', label: '0 (any improvement)'},
        {value: '0.0001', label: '0.0001'},
        {value: '0.001', label: '0.001'}
      ]
    }
  ],

  readouts: [
    {
      id: 'sizesChart',
      label: 'Community sizes',
      kind: 'chart',
      help: 'Stations in each of the twelve largest communities of the partition shown.'
    },
    {
      id: 'qualityChart',
      label: 'Modularity of three partitions',
      kind: 'chart',
      help: 'Weighted modularity on the current graph: the borough partition, label propagation, and the refined partition.'
    },
    {
      id: 'communityCount',
      label: 'Communities',
      format: 'integer',
      help: 'Distinct groups in the partition shown.'
    },
    {id: 'largest', label: 'Largest community'},
    {id: 'modularityBoroughs', label: 'Modularity, boroughs', format: 'decimal'},
    {id: 'modularityPropagation', label: 'Modularity, propagation', format: 'decimal'},
    {id: 'modularityOptimized', label: 'Modularity, refined', format: 'decimal'},
    {
      id: 'withinShare',
      label: 'Rides inside a community',
      format: 'percent',
      help: 'Share of the rides on the kept edges whose two stations share a community in the partition shown.'
    },
    {
      id: 'withinBoroughShare',
      label: 'Rides inside a borough',
      format: 'percent',
      help: 'The same share for the borough partition.'
    },
    {
      id: 'agreement',
      label: 'Agreement with boroughs',
      format: 'decimal',
      help: 'Normalised mutual information between the partition shown and the boroughs: 1 identical, 0 unrelated.'
    },
    {id: 'stations', label: 'Stations', format: 'integer'},
    {id: 'edgesKept', label: 'Edges in the graph'},
    {
      id: 'ridesKept',
      label: 'Rides kept',
      format: 'percent',
      help: 'Share of all inter-station pair rides on the kept edges.'
    },
    {id: 'drawn', label: 'Links drawn', format: 'integer'},
    {id: 'convergence', label: 'Convergence', layout: 'block'},
    {id: 'validity', label: 'Graph status'},
    {id: 'selected', label: 'Selected station', layout: 'block'}
  ],

  legends: (_state, data) => {
    const entries = (data?.communities as
      | {color: readonly number[]; label: string}[]
      | undefined) ?? [{color: COMMUNITY_COLORS[0], label: 'computing'}];
    return [
      {
        kind: 'categories',
        title: 'Community (largest first)',
        entries: entries.map(entry => ({
          color: entry.color as [number, number, number],
          label: entry.label
        })),
        note: 'Lines inside a community share its colour; grey lines cross a boundary.'
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph, GPUGraphTopology, GPUGraphLabelPropagation,
  GPUGraphModularityOptimization, GPUGraphModularity
} from '@luma.gl/gpgpu/gpu-graph';

// Caller-owned vectors: edge columns (rides as weights), CSR, and every output.
const graph = new GPUGraph({vertexCount: 905, sourceVertices, targetVertices,
  edgeWeights: rides, directed: false});
const topology = new GPUGraphTopology({graph, forward, invalidEdgeCount});

const commandGraph = new GPUCommandGraph(device, {id: 'communities'});
topology.addToGraph(commandGraph);
new GPUGraphLabelPropagation({
  topology, output: proposal, iterations: ${state.propagationRounds}, converged
}).addToGraph(commandGraph);
new GPUGraphModularityOptimization({
  topology, output: refined, modularity: refinedScore,
  initialCommunities: proposal,                    // warm start
  resolution: ${state.resolution}, iterations: ${state.optimizeRounds}, minimumGain: ${state.minimumGain},
  converged, valid
}).addToGraph(commandGraph);
new GPUGraphModularity({graph, communities: boroughs, output: boroughScore,
  resolution: ${state.resolution}}).addToGraph(commandGraph);

const compiled = commandGraph.compile();            // once
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUGraphLabelPropagation` groups stations by the label most common among their neighbours, `GPUGraphModularityOptimization` improves that partition by moving one station at a time to the community that raises weighted modularity most, and `GPUGraphModularity` scores any partition, here also the published boroughs.',
    why: 'Service areas, rebalancing routes and station placement should follow how people ride, not how a map is drawn. A community partition measured against the borough partition tells you where the two disagree, with a number instead of an impression.',
    howToRead:
      'Dots and lines take the colour of their community, largest first; the legend names each after its main boroughs. Grey lines cross boundaries. Higher modularity means more rides inside groups than a random network with the same degrees would keep there.'
  },

  create: async ctx => (await import('./bixi-communities.compute')).createBixiCommunities(ctx),

  story: storyFromMarkdown<BixiCommunitiesOptions>(narrative, {
    'the-question': {
      controls: ['partition', 'edges'],
      readouts: ['communityCount', 'sizesChart'],
      camera: {longitude: -73.61, latitude: 45.53, zoom: 10.7, transitionMs: 1400},
      options: {partition: 'optimized', neighbors: 8, edges: 3000},
      highlight: {readout: 'communityCount'}
    },
    propagation: {
      controls: ['partition', 'neighbors', 'propagationRounds', 'minRides'],
      readouts: ['communityCount', 'modularityPropagation', 'convergence'],
      camera: {longitude: -73.58, latitude: 45.52, zoom: 11.4, transitionMs: 1400},
      options: {partition: 'propagation', neighbors: 8},
      highlight: {readout: 'modularityPropagation'}
    },
    refine: {
      controls: ['partition', 'optimizeRounds', 'resolution', 'weighting'],
      readouts: ['modularityOptimized', 'qualityChart', 'convergence'],
      options: {partition: 'optimized', optimizeRounds: 512, resolution: 1},
      highlight: {readout: 'modularityOptimized'}
    },
    'vs-boroughs': {
      controls: ['partition', 'showBetween'],
      readouts: ['withinShare', 'withinBoroughShare', 'agreement', 'qualityChart'],
      camera: {longitude: -73.6, latitude: 45.52, zoom: 11.1, transitionMs: 1400},
      options: {partition: 'boroughs'},
      highlight: {readout: 'withinBoroughShare'}
    },
    limits: {
      controls: ['resolution', 'weighting', 'optimizeRounds', 'showBetween'],
      readouts: ['selected', 'sizesChart'],
      options: {partition: 'optimized', resolution: 1, optimizeRounds: 512},
      camera: {longitude: -73.61, latitude: 45.53, zoom: 10.7, transitionMs: 1200}
    }
  })
});
