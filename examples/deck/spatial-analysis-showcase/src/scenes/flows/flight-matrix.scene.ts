// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './flight-matrix.md?raw';
import type {FlightMatrixOptions} from './flight-matrix.compute';
import {CONTINENT_COLORS, CONTINENT_NAMES} from './b11-geography';

const MATRIX_VIEW = {longitude: 0, latitude: 0, zoom: 2.5};

export default defineScene<FlightMatrixOptions>({
  id: 'flight-matrix',
  title: 'The airline network as a matrix',
  chapter: 'flows',
  order: 21,
  summary:
    'An adjacency matrix of 3,257 airports ordered by continent, country and degree on the GPU, with a zoom window and a coarsened view that collapses the network into supernodes and superedges.',
  contributors: ['GPUAdjacencyMatrix', 'GPUAdjacencyMatrixOrder', 'GPUNetworkCoarsening'],
  datasets: [{id: 'openflights', role: 'airports and route pairs'}],
  initialView: {longitude: 10, latitude: 26, zoom: 1.75},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Show',
      group: 'View',
      apply: 'param',
      default: 'map',
      help: 'The network map, the matrix card, or both. The matrix card is drawn around longitude 0, latitude 0: zoom to it, or use the story steps.',
      options: [
        {value: 'map', label: 'Network map'},
        {value: 'matrix', label: 'Adjacency matrix'},
        {value: 'both', label: 'Map and matrix'}
      ]
    },
    {
      kind: 'select',
      id: 'order',
      label: 'Order airports by',
      group: 'Matrix order',
      apply: 'param',
      default: 'input',
      help: '`GPUAdjacencyMatrixOrder` sorts by a group label, then by a tie key. Group and tie keys are buffer writes; nothing recompiles.',
      options: [
        {value: 'input', label: 'Input order (alphabetical IATA)'},
        {value: 'degree', label: 'Degree only (hubs first)'},
        {value: 'continent', label: 'Continent'},
        {value: 'continent-degree', label: 'Continent, hubs first'},
        {value: 'country', label: 'Country within continent, hubs first'}
      ]
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Cell value',
      group: 'Matrix cells',
      apply: 'param',
      default: 'count',
      help: 'Edge count per cell, or the fixed-point weight sum per cell (bit-identical regardless of GPU thread order).',
      options: [
        {value: 'count', label: 'Airport pairs (count)'},
        {value: 'weight', label: 'Weight sum'}
      ]
    },
    {
      kind: 'select',
      id: 'weightBy',
      label: 'Edge weight',
      group: 'Matrix cells',
      apply: 'param',
      default: 'routes',
      disabledWhen: state => state.statistic !== 'weight',
      help: 'What each edge adds to a weight sum: airline-route records, airlines serving the pair, or distance in units of 100 km. Also used for coarsened superedge weights.',
      options: [
        {value: 'routes', label: 'Route records'},
        {value: 'airlines', label: 'Airlines serving the pair'},
        {value: 'distance', label: 'Distance (100 km)'}
      ]
    },
    {
      kind: 'select',
      id: 'resolution',
      label: 'Resolution',
      group: 'Matrix cells',
      apply: 'compile',
      default: '512',
      help: 'Bins per axis of the matrix image (1 to 4096 allowed). Higher resolution separates more airports per cell and costs R × R memory. Rebuilds the graph.',
      options: [
        {value: '256', label: '256 × 256'},
        {value: '512', label: '512 × 512'},
        {value: '1024', label: '1024 × 1024'},
        {value: '2048', label: '2048 × 2048'}
      ]
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Focus block',
      group: 'Zoom window',
      apply: 'param',
      default: 'all',
      help: 'Zooms the window to one continent block. Needs a continent or country order (continents are contiguous in both); with any other order the manual zoom is used.',
      options: [
        {value: 'all', label: 'Manual zoom and pan'},
        ...CONTINENT_NAMES.map(name => ({value: name, label: name}))
      ]
    },
    {
      kind: 'slider',
      id: 'zoom',
      label: 'Zoom',
      group: 'Zoom window',
      apply: 'param',
      min: 1,
      max: 24,
      step: 0.5,
      default: 1,
      unit: '×',
      disabledWhen: state => state.focus !== 'all',
      help: 'Shrinks the window to 1/zoom of the airports on both axes. The window is four uint32 words.'
    },
    {
      kind: 'slider',
      id: 'panX',
      label: 'Pan columns',
      group: 'Zoom window',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.5,
      disabledWhen: state => state.focus !== 'all',
      format: value => `${Math.round(value * 100)}%`,
      help: 'Centre of the window along the columns, as a share of the matrix.'
    },
    {
      kind: 'slider',
      id: 'panY',
      label: 'Pan rows',
      group: 'Zoom window',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.5,
      disabledWhen: state => state.focus !== 'all',
      format: value => `${Math.round(value * 100)}%`,
      help: 'Centre of the window along the rows, as a share of the matrix.'
    },
    {
      kind: 'select',
      id: 'coarsenBy',
      label: 'Coarsen by',
      group: 'Coarsening',
      apply: 'param',
      default: 'continent',
      help: 'Group label for `GPUNetworkCoarsening`. Labels are a buffer, so relabelling re-runs the same graph. Capacity is 256 groups.',
      options: [
        {value: 'continent', label: 'Continent (6 groups)'},
        {value: 'country', label: 'Country (about 225 groups)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showGroups',
      label: 'Supernodes and superedges',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the coarsened network on the map: discs sized by airports, arcs sized by routes between groups.'
    },
    {
      kind: 'toggle',
      id: 'showEdges',
      label: 'Route pairs (faint)',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every airport pair as a faint straight line on the map.'
    },
    {
      kind: 'toggle',
      id: 'showAirports',
      label: 'Airports',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots coloured by continent.'
    },
    {
      kind: 'toggle',
      id: 'showBlocks',
      label: 'Block boundaries and strips',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws continent (thick) and country (thin) boundaries on the matrix and coloured strips beside it.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Matrix colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp for matrix cells. Empty cells are transparent.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'slider',
      id: 'arcWidth',
      label: 'Widest superedge',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 30,
      step: 1,
      default: 14,
      unit: 'px',
      help: 'Pixel width of the superedge with the most routes.'
    }
  ],

  readouts: [
    {
      id: 'degreeChart',
      label: 'Routes per airport',
      kind: 'chart',
      help: 'Airports by number of routes, in power-of-two bins. A long tail: most airports have a few routes, a few hubs have hundreds. Hubs are the bright rows and columns of the matrix.'
    },
    {
      id: 'cellChart',
      label: 'How full is a cell?',
      kind: 'chart',
      help: 'Occupied matrix cells by the number of airport pairs inside, from GPUAdjacencyMatrix. It shows why the matrix is dark almost everywhere.'
    },
    {
      id: 'groupChart',
      label: 'Largest groups',
      kind: 'chart',
      help: 'Airports in the eight largest groups of GPUNetworkCoarsening, by continent or country.'
    },
    {
      id: 'orderCheck',
      label: 'Order check',
      help: 'The GPU permutation compared with the CPU twin `computeAdjacencyMatrixOrder`.'
    },
    {id: 'resolution', label: 'Matrix size'},
    {id: 'window', label: 'Window'},
    {
      id: 'maxCell',
      label: 'Brightest cell',
      format: 'integer',
      help: 'Largest cell value in the window (`maxCount` or `maxWeightSum`).'
    },
    {
      id: 'occupied',
      label: 'Cells occupied',
      format: 'integer',
      help: 'Cells with at least one pair. Read back for resolutions up to 1024.'
    },
    {id: 'filled', label: 'Matrix fill', format: 'percent'},
    {id: 'groups', label: 'Groups', format: 'integer', help: 'Groups with at least one airport.'},
    {id: 'superedges', label: 'Superedges', help: 'Distinct group pairs.'},
    {
      id: 'intraShare',
      label: 'Routes inside their group',
      format: 'percent',
      help: 'Intra-group edges divided by all counted edges: the weight of the diagonal blocks.'
    },
    {id: 'interEdges', label: 'Routes between groups', format: 'integer'},
    {
      id: 'dropped',
      label: 'Dropped (label overflow)',
      format: 'integer',
      help: 'Edges and airports with a label at or above the group capacity.'
    },
    {id: 'largestGroup', label: 'Largest group'}
  ],

  legends: state => [
    {
      kind: 'ramp',
      id: 'matrix',
      title: state.statistic === 'count' ? 'Airport pairs per cell' : 'Weight per cell',
      ramp: state.ramp,
      extent: 'gpu',
      sqrtScale: true,
      format: (value: number) =>
        value >= 100 ? Math.round(value).toLocaleString('en-US') : value.toFixed(0)
    },
    {
      kind: 'categories',
      title: 'Continents',
      entries: CONTINENT_NAMES.map((name, index) => ({
        color: CONTINENT_COLORS[index],
        label: name
      })),
      note: 'Same colours for airports, strips and supernodes.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  GPUNetworkCoarsening,
  encodeGPUAdjacencyMatrixWindow
} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'flight-matrix'});
// 1. vertex -> matrix position: sort by group, then by tie key (two stable GPU sorts)
graph.add(new GPUAdjacencyMatrixOrder({groups, tieKeys, order}));
// 2. R x R image of the CSR graph through that order, zoomable without recompiling
graph.add(
  new GPUAdjacencyMatrix({
    offsets, neighbors, weights, order,
    window: window.importToGraph(graph),   // [rowStart, rowEnd, colStart, colEnd)
    resolution: ${state.resolution},
    output: {counts, weightSums, maxCount, maxWeightSum}
  })
);
// 3. supernodes and superedges by label (continent or country)
graph.add(
  new GPUNetworkCoarsening({
    offsets, neighbors, weights, labels, positions,
    groupCapacity: 256,
    groupVertexCount, groupIntraEdgeCount, groupCentroid,
    edges: {ids, count, overflow}, edgeTargets, edgeCounts, edgeWeights, summary
  })
);
const compiled = graph.compile();            // once
// when order, window or labels change: write the buffers, then
window.write(encodeGPUAdjacencyMatrixWindow({rowStart: 0, rowEnd: 3257, colStart: 0, colEnd: 3257}, ${state.resolution}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUAdjacencyMatrix` bins a graph into a zoomable image with exact integer counts and fixed-point weight sums. `GPUAdjacencyMatrixOrder` computes the row and column order from group labels and tie keys. `GPUNetworkCoarsening` collapses the graph by label into supernodes and superedges.',
    why: 'Node-link maps hide structure when graphs are large. An ordered matrix shows communities as blocks, and a coarsened graph gives the zoomed-out summary: how big each group is and how strongly groups connect.',
    howToRead:
      'Rows and columns are airports in the chosen order. Bright blocks on the diagonal are connections inside a group; off-diagonal blocks are connections between groups. In the coarsened map, disc size is airport count and arc width is route count.'
  },

  create: async ctx => (await import('./flight-matrix.compute')).createFlightMatrix(ctx),

  story: storyFromMarkdown<FlightMatrixOptions>(narrative, {
    question: {
      controls: ['view'],
      readouts: ['degreeChart', 'groups'],
      camera: {longitude: 10, latitude: 26, zoom: 1.75, transitionMs: 1200},
      options: {view: 'map', showGroups: false},
      callout: {coordinate: [-0.46, 51.47], text: 'London Heathrow'}
    },
    'matrix-input': {
      controls: ['order'],
      readouts: ['occupied', 'maxCell', 'cellChart'],
      camera: {...MATRIX_VIEW, transitionMs: 1600},
      options: {view: 'matrix', order: 'input'},
      highlight: {readout: 'occupied'}
    },
    'order-continent': {
      controls: ['order', 'showBlocks'],
      readouts: ['orderCheck'],
      options: {order: 'continent'},
      highlight: {readout: 'orderCheck'}
    },
    'order-country': {
      controls: ['order'],
      readouts: ['maxCell'],
      options: {order: 'country'}
    },
    'zoom-europe': {
      controls: ['focus', 'resolution'],
      readouts: ['window'],
      options: {focus: 'Europe'},
      highlight: {readout: 'window'}
    },
    coarsen: {
      controls: ['coarsenBy', 'showGroups', 'showEdges'],
      readouts: ['intraShare', 'interEdges', 'groupChart', 'dropped'],
      camera: {longitude: 10, latitude: 26, zoom: 1.75, transitionMs: 1600},
      options: {view: 'map', showGroups: true, coarsenBy: 'continent', showEdges: false},
      highlight: {readout: 'intraShare'}
    },
    limits: {
      controls: ['order', 'statistic', 'weightBy', 'view'],
      readouts: ['occupied']
    }
  })
});
