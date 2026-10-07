// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {ROAD_CLASS_NAMES} from './b10-road-graph';
import {
  COMMUNITY_COLORS,
  COMPONENT_COLORS,
  LENGTH_MAXIMUM,
  SPEED_MAXIMUM
} from './b10-scene-constants';
import type {CentralityMetric, CentralityOptions} from './street-centrality.compute';

const METRIC_TITLES: Record<CentralityMetric, string> = {
  pageRank: 'PageRank of the street graph',
  degree: 'Intersection degree',
  inDegree: 'Incoming streets per intersection',
  core: 'k-core number',
  community: 'Label-propagation community',
  component: 'Connected component'
};

const FALLBACK_SCALAR_CLASSES = makeClassTable({
  breaks: [0, 1, 2, 3, 4, 5],
  scheme: 'YlOrRd',
  labels: ['lowest', 'very low', 'low', 'middle', 'high', 'very high', 'highest'],
  unit: 'metric value',
  noData: {color: [115, 120, 135, 255], label: 'no metric value'},
  method: 'Waiting for the first topology readback.'
});
const CORE_CLASSES = makeClassTable({
  breaks: [1, 2, 3, 4],
  colors: [
    [110, 110, 110],
    [255, 237, 160],
    [254, 178, 76],
    [240, 59, 32],
    [189, 0, 38]
  ],
  labels: ['core 0', 'core 1', 'core 2', 'core 3', 'core 4+'],
  unit: 'core number',
  noData: {color: [115, 120, 135, 255], label: 'no core value'},
  method: 'Ordinal k-core classes.'
});

/** Street-network analytics of Chicago. GPU work is in `street-centrality.compute.ts`. */
export default defineScene<CentralityOptions>({
  id: 'street-centrality',
  title: 'Which streets does Chicago lean on?',
  chapter: 'networks',
  order: 4,
  summary:
    'PageRank, degree, k-core, communities and components of the Chicago street graph as node columns on the GPU, filtered by speed, length, class and importance, with live network statistics of whatever survives.',
  contributors: ['GPUNetworkAnalyticsColumns', 'GPUNetworkStatistics', 'GPUNetworkSubgraphFilter'],
  datasets: [
    {id: 'chicago-roads', role: '29,557 intersections, 77,140 directed edges, class and speed'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10.0},
  basemap: ground('night'),
  furniture: {
    title: {title: 'Structural centrality of Chicago streets'},
    credit: 'OpenStreetMap contributors (ODbL)'
  },

  options: [
    {
      kind: 'select',
      id: 'metric',
      label: 'Color streets by',
      group: 'Display',
      apply: 'param',
      default: 'pageRank',
      help: 'Which node column of GPUNetworkAnalyticsColumns styles the streets. A street takes the average of its two intersections (labels: its start intersection). Switching copies one column and re-runs a styling kernel; nothing is recompiled.',
      options: [
        {value: 'pageRank', label: 'PageRank (importance)'},
        {value: 'degree', label: 'Degree (streets meeting)'},
        {value: 'inDegree', label: 'In-degree (one-way streets only)'},
        {value: 'core', label: 'k-core number'},
        {value: 'community', label: 'Community (label propagation)'},
        {value: 'component', label: 'Connected components'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRemoved',
      label: 'Show filtered-out streets faintly',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every street faintly under the live ones so you can see what the filter removed.'
    },
    {
      kind: 'toggle',
      id: 'oneWay',
      label: 'Respect one-way streets (directed)',
      group: 'Centrality',
      apply: 'compile',
      default: false,
      help: 'Off: every street is two-way (symmetric graph), the convention for undirected metrics. On: the graph is directed with a reverse CSR, PageRank follows traffic direction, in-degree becomes available and statistics count directed edges. Rebuilds every graph.'
    },
    {
      kind: 'select',
      id: 'damping',
      label: 'PageRank damping',
      group: 'Centrality',
      apply: 'compile',
      default: '0.85',
      help: 'Probability of following a street instead of teleporting to a random intersection. Higher values make rank flow farther along the network. Compile-time: rebuilds the analytics graph.',
      options: [
        {value: '0.5', label: '0.50'},
        {value: '0.7', label: '0.70'},
        {value: '0.85', label: '0.85 (classic)'},
        {value: '0.95', label: '0.95'},
        {value: '0.99', label: '0.99'}
      ]
    },
    {
      kind: 'select',
      id: 'pageRankIterations',
      label: 'PageRank iterations',
      group: 'Centrality',
      apply: 'compile',
      default: '40',
      help: 'Power-iteration steps. The residual readout shows how much the ranks still moved in the last step. Rebuilds the analytics graph.',
      options: [
        {value: '5', label: '5'},
        {value: '10', label: '10'},
        {value: '20', label: '20'},
        {value: '40', label: '40 (default)'},
        {value: '100', label: '100'}
      ]
    },
    {
      kind: 'select',
      id: 'communityIterations',
      label: 'Community iterations',
      group: 'Centrality',
      apply: 'compile',
      default: '32',
      help: 'Rounds of label propagation. Few rounds leave many tiny communities; more rounds merge them. The result is a bounded heuristic, not modularity optimization. Rebuilds the analytics graph.',
      options: [
        {value: '2', label: '2'},
        {value: '5', label: '5'},
        {value: '10', label: '10'},
        {value: '32', label: '32 (default)'},
        {value: '100', label: '100'}
      ]
    },
    {
      kind: 'select',
      id: 'componentIterations',
      label: 'Component iterations',
      group: 'Centrality',
      apply: 'compile',
      default: '128',
      help: 'Relaxation rounds of the connected-component labelling, used by the analytics columns and the statistics. Too few leave components unconverged and the counts too high. Rebuilds the analytics and statistics graphs.',
      options: [
        {value: '4', label: '4'},
        {value: '16', label: '16'},
        {value: '32', label: '32'},
        {value: '128', label: '128 (default)'}
      ]
    },
    {
      kind: 'range',
      id: 'speedRange',
      label: 'Speed limit',
      group: 'Subgraph filter',
      apply: 'param',
      min: 0,
      max: SPEED_MAXIMUM,
      step: 5,
      default: [0, SPEED_MAXIMUM],
      unit: 'km/h',
      format: value => (value >= SPEED_MAXIMUM ? 'no limit' : `${value}`),
      help: 'Keeps streets whose speed limit is in this range (half-open: min <= speed < max; the top end means no limit). Edge predicate column 1.'
    },
    {
      kind: 'range',
      id: 'lengthRange',
      label: 'Block length',
      group: 'Subgraph filter',
      apply: 'param',
      min: 0,
      max: LENGTH_MAXIMUM,
      step: 25,
      default: [0, LENGTH_MAXIMUM],
      unit: 'm',
      format: value => (value >= LENGTH_MAXIMUM ? `${value}+` : `${value}`),
      help: 'Keeps edges whose length is in this range; the top end means no limit. Edge predicate column 2.'
    },
    {
      kind: 'range',
      id: 'classRange',
      label: 'Road class',
      group: 'Subgraph filter',
      apply: 'param',
      min: 0,
      max: ROAD_CLASS_NAMES.length - 1,
      step: 1,
      default: [0, ROAD_CLASS_NAMES.length - 1],
      format: value => ROAD_CLASS_NAMES[value] ?? String(value),
      help: 'Keeps road classes from the first to the second handle, inclusive: motorway, trunk, primary, secondary, tertiary, residential, service/other. Edge predicate column 3.'
    },
    {
      kind: 'slider',
      id: 'topPercent',
      label: 'Most important intersections',
      group: 'Subgraph filter',
      apply: 'param',
      min: 5,
      max: 100,
      step: 5,
      default: 100,
      unit: '%',
      help: 'Keeps only intersections in the top share by PageRank, a vertex predicate on the analytics output. A street survives only if both ends do.'
    },
    {
      kind: 'toggle',
      id: 'dropIsolated',
      label: 'Drop isolated intersections',
      group: 'Subgraph filter',
      apply: 'compile',
      default: false,
      help: 'Also removes intersections left with no live street after filtering. Changes the live intersection count and the isolated count. Rebuilds the filter and statistics graph.'
    },
    {
      kind: 'toggle',
      id: 'pairSlots',
      label: 'Require both directions to pass',
      group: 'Subgraph filter',
      apply: 'compile',
      default: true,
      disabledWhen: state => state.oneWay,
      help: 'Undirected graphs list each street twice. With pairing on, a street is live only if both slots pass their predicates, so the mask is always symmetric. Ignored for directed graphs. Rebuilds the filter and statistics graph.'
    },
    {
      kind: 'slider',
      id: 'resolution',
      label: 'Modularity resolution',
      group: 'Statistics',
      apply: 'param',
      min: 0.2,
      max: 3,
      step: 0.1,
      default: 1,
      help: 'The gamma of Q = intra/2m - gamma * sum (d_c/2m)^2 for the community labels. Higher values penalize large communities.'
    },
    {
      kind: 'select',
      id: 'degreeBinning',
      label: 'Degree histogram bins',
      group: 'Statistics',
      apply: 'compile',
      default: 'linear',
      help: 'Linear: bins of fixed width. Log2: bin 0 for degree 0, then 1, 2-3, 4-7, ... Rebuilds the filter and statistics graph.',
      options: [
        {value: 'linear', label: 'Linear'},
        {value: 'log2', label: 'Log2'}
      ]
    },
    {
      kind: 'slider',
      id: 'degreeBinWidth',
      label: 'Linear bin width',
      group: 'Statistics',
      apply: 'param',
      min: 1,
      max: 4,
      step: 1,
      default: 1,
      disabledWhen: state => state.degreeBinning === 'log2',
      help: 'Degrees per histogram bin (12 bins). A per-frame parameter.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the analytics and the filter',
      group: 'Timing',
      help: 'Runs the analytics graph and the filter-plus-statistics graph outside the frame and reports median times.'
    }
  ],

  legends: (state, data) => [
    state.metric === 'community' || state.metric === 'component'
      ? {
          kind: 'categories' as const,
          title: METRIC_TITLES[state.metric],
          entries:
            state.metric === 'component'
              ? [
                  {color: COMPONENT_COLORS[0], label: 'Largest component'},
                  {color: COMPONENT_COLORS[1], label: 'Disconnected islands'}
                ]
              : COMMUNITY_COLORS.slice(0, 4).map((color, index) => ({
                  color,
                  label: `Stable community slot ${index + 1}`
                })),
          note:
            state.metric === 'community'
              ? 'Display slots are retained by overlap across iteration presets; small overflow groups are grey.'
              : 'Islands are street fragments with no path to the main network.'
        }
      : getClassTableLegend(
          state.metric === 'core'
            ? CORE_CLASSES
            : ((
                data['centralityClasses'] as {
                  tables?: Partial<Record<CentralityMetric, typeof FALLBACK_SCALAR_CLASSES>>;
                }
              )?.tables?.[state.metric] ?? FALLBACK_SCALAR_CLASSES),
          {
            title: `${METRIC_TITLES[state.metric]} (frozen topology classes)`,
            note:
              state.metric === 'core'
                ? 'Ordinal classes, not a continuous ramp.'
                : 'Readback-derived and frozen while damping / iterations are compared; tooltip gives endpoints.'
          }
        )
  ],

  readouts: [
    {id: 'network', label: 'Street graph'},
    {id: 'length', label: 'Network length', help: 'Every street counted once.'},
    {id: 'slots', label: 'CSR slots'},
    {
      id: 'liveVertices',
      label: 'Live intersections',
      help: 'From GPUNetworkStatistics over the filtered subgraph.'
    },
    {
      id: 'liveEdges',
      label: 'Live streets',
      format: 'integer',
      help: 'Undirected: each street once. Directed: each directed edge.'
    },
    {
      id: 'filterCounts',
      label: 'Filter output counts',
      help: 'liveVertexCount and liveSlotCount decoded from GPUNetworkSubgraphFilter counts.'
    },
    {id: 'statComponents', label: 'Components of the live network'},
    {id: 'largestComponent', label: 'Largest live component'},
    {id: 'isolated', label: 'Isolated intersections', format: 'integer'},
    {id: 'maxDegree', label: 'Largest degree', format: 'integer'},
    {
      id: 'modularity',
      label: 'Modularity of the communities',
      format: 'decimal',
      help: 'Q of the label-propagation communities on the live network (deterministic fixed-point sums).'
    },
    {
      id: 'histogram',
      label: 'Selected-metric class histogram',
      kind: 'chart',
      help: 'Readback values counted in the same frozen classes used by the selected street metric.'
    },
    {id: 'histogramBins', label: 'Histogram bins'},
    {id: 'communities', label: 'Communities'},
    {id: 'components', label: 'Components (full network)'},
    {id: 'pageRankPeak', label: 'Peak PageRank'},
    {id: 'pageRankResidual', label: 'PageRank residual', format: 'decimal'},
    {id: 'pageRankResiduals', label: 'PageRank residual by iteration', kind: 'chart'},
    {id: 'degreeCorrelation', label: 'Degree / PageRank correlation', kind: 'chart'},
    {id: 'communitySizes', label: 'Community sizes', kind: 'chart'},
    {id: 'degeneracy', label: 'Degeneracy (largest core number)'},
    {id: 'analyticsTime', label: 'Analytics graph time'},
    {id: 'filterTime', label: 'Filter and statistics time'}
  ],

  snippet: state => `import {
  GPUNetworkAnalyticsColumns,
  GPUNetworkSubgraphFilter,
  GPUNetworkStatistics,
  getGPUNetworkSubgraphFilterParameterValues,
  encodeGPUNetworkStatisticsParameters
} from '@luma.gl/experimental/gpu-network';

// 1. Node columns for styling. No reverse CSR = undirected (symmetric CSR required).
analyticsGraph.add(new GPUNetworkAnalyticsColumns({
  offsets, neighbors,${state.oneWay ? '\n  reverseOffsets, reverseNeighbors,        // presence makes it directed\n  inDegree: {output: inDegree, normalized: inDegreeNormalized},' : ''}
  degree: {output: degree, normalized: degreeNormalized},
  pageRank: {output: pageRank, normalized: pageRankNormalized,
             damping: ${state.damping}, iterations: ${state.pageRankIterations}, residual},       // compile-time
  coreNumber: {output: core, normalized: coreNormalized, degeneracy, converged},
  components: {output: components, iterations: ${state.componentIterations}},
  communities: {output: communities, iterations: ${state.communityIterations}}
}));

// 2. Predicates on the analytics output and on edge attributes -> masks.
graph.add(new GPUNetworkSubgraphFilter({
  offsets, neighbors, directed: ${state.oneWay}, dropIsolated: ${state.dropIsolated}, pairUndirectedSlots: ${state.pairSlots},
  vertexColumns: [pageRankNormalized],
  edgeColumns: [speed, length, roadClass],
  parameters: filterParameters,
  output: {vertexMask, edgeMask, counts}
}));

// 3. Statistics of the live network (masks in, one packed summary out).
graph.add(new GPUNetworkStatistics({
  offsets, neighbors, directed: ${state.oneWay}, vertexMask, edgeMask, communities,
  degreeBinning: '${state.degreeBinning}', degreeBinCount: 12, componentIterations: ${state.componentIterations},
  parameters: statisticsParameters, output: statistics
}));

// Ranges are half-open [min, max) float parameters: no recompile.
filterParameters.write(getGPUNetworkSubgraphFilterParameterValues(layout, {
  vertexRanges: [[pageRankThreshold, Infinity]],
  edgeRanges: [[${state.speedRange[0]}, ${state.speedRange[1] >= SPEED_MAXIMUM ? 'Infinity' : state.speedRange[1]}], null, [${state.classRange[0]}, ${state.classRange[1] + 1}]]
}));
statisticsParameters.write(encodeGPUNetworkStatisticsParameters({resolution: ${state.resolution}}));`,

  about: {
    what: '`GPUNetworkAnalyticsColumns` runs gpu-graph algorithms on a CSR and publishes node columns: degree, in-degree, PageRank, k-core number, weak components and label-propagation communities, each optionally normalized to [0, 1]. `GPUNetworkSubgraphFilter` turns attribute ranges on vertex and edge columns into live masks. `GPUNetworkStatistics` reduces the masked network to one summary: counts, components, isolated vertices, degree histograms and modularity.',
    why: 'Which streets does a network depend on, and what is left of it when you keep only fast roads or long blocks? The same three tools answer that for roads, transit, airlines or social graphs, and the columns feed deck.gl styling directly with no readback.',
    howToRead:
      'Streets are colored by the chosen column. The filter draws live streets in color over a faint copy of everything else. The statistics readouts always describe the live network; the community, component and PageRank readouts describe the full network.'
  },

  create: async ctx => (await import('./street-centrality.compute')).createStreetCentrality(ctx),

  story: [
    {
      id: 'degree',
      headline: 'Centrality is structural importance, not traffic',
      textAlternative:
        'Fixed degree classes and a live histogram show a mostly regular, clipped street graph.',
      optionsMode: 'fresh',
      controls: ['metric'],
      readouts: ['network', 'histogram', 'histogramBins'],
      title: 'Degree exposes a regular grid',
      body: 'Degree counts streets meeting at an intersection. Raw degree values are classed for colour, and the live histogram makes the regular grid and clipped city boundary visible without implying traffic or a continuous ranking. The scale bar and OSM/ODbL credit describe the graph, not the city beyond its edge.',
      options: {metric: 'degree'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.0, transitionMs: 1200}
    },
    {
      id: 'pagerank',
      headline: 'A regular grid gives PageRank little hierarchy',
      textAlternative:
        'Fixed PageRank classes with live degree correlation and bounded-iteration residual diagnostic.',
      optionsMode: 'fresh',
      controls: ['damping', 'pageRankIterations'],
      readouts: ['pageRankResiduals', 'degreeCorrelation', 'pageRankResidual'],
      title: 'PageRank on a street grid',
      body: 'PageRank is a bounded structural iteration, not traffic. Fixed classes remain comparable while damping and the chosen iteration limit change; the live scatter reports its relationship with degree and the residual says how much the last bounded step moved. A residual is not a convergence claim.',
      options: {metric: 'pageRank'},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 11.6, transitionMs: 1600},
      highlight: {readout: 'pageRankPeak'}
    },
    {
      id: 'core',
      headline: 'Core numbers are ordinal layers of connectivity',
      textAlternative:
        'Ordinal k-core street classes and a degree histogram expose peeling layers.',
      optionsMode: 'fresh',
      controls: ['metric'],
      readouts: ['degeneracy', 'histogram'],
      title: 'Degree and k-core: how tangled is a place?',
      body: 'Streets are colored by **k-core number** (change it with **Color streets by**). The k-core is what remains after repeatedly deleting intersections with fewer than k streets. Dead ends and cul-de-sacs peel away first (core 1); the grid itself sits in the 2-core and 3-core, and the **Degeneracy (largest core number)** readout shows it. **Degree** counts the streets meeting at an intersection: its raw values are classed for colour in the selected-metric histogram.\n\nBoth are exact integer columns. Pick **Degree** to see the T-junctions that dominate the outskirts.',
      options: {metric: 'core'},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 11.8, transitionMs: 1200},
      highlight: {readout: 'degeneracy'}
    },
    {
      id: 'communities',
      headline: 'Label propagation depends on a stopping rule',
      textAlternative:
        'Stable community colours and size bars show a bounded label-propagation heuristic.',
      optionsMode: 'fresh',
      controls: ['communityIterations', 'resolution'],
      readouts: ['communitySizes', 'communities', 'modularity'],
      title: 'Communities and modularity',
      body: 'Label propagation is a bounded heuristic, not modularity optimization. Display hues retain identity by overlap across iteration presets; smaller overflow groups are grey rather than hue-cycled. Community size bars, modularity and the stopping-rule caveat keep the result diagnostic.',
      options: {metric: 'community'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.6, transitionMs: 1600},
      highlight: {readout: 'modularity'}
    },
    {
      id: 'filter',
      headline: 'A half-open filter turns a graph into a testable skeleton',
      textAlternative:
        'Faint removed streets remain beneath a live filtered graph with components, largest component and isolated-node evidence.',
      optionsMode: 'fresh',
      controls: ['classRange', 'speedRange', 'lengthRange'],
      readouts: ['statComponents', 'liveVertices', 'largestComponent'],
      title: 'Keep only the arterials: the subgraph filter',
      body: '`GPUNetworkSubgraphFilter` takes ranges over edge attributes (speed, length, road class) and vertex attributes (here the PageRank column) and writes a **vertex mask** and an **edge mask**: a street is live only if it passes every enabled range and both ends are live. Ranges are half-open and live in a parameter buffer, so dragging them recompiles nothing.\n\n**Road class** below is set to motorway through secondary. The faint streets are gone from the live set; `GPUNetworkStatistics` then reports the live network: how many intersections remain, how many **Components of the live network** the arterial network falls into, and how large the biggest one is.',
      options: {metric: 'pageRank', classRange: [0, 3] as const},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.6, transitionMs: 1400},
      highlight: {readout: 'statComponents'}
    }
  ]
});
