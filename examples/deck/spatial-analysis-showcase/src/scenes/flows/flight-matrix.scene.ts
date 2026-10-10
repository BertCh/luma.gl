// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import type {MapGround} from '../../cartography/hue-registry';
import {formatCount} from '../../cartography/live-text';
import {getSizeLegendEntries} from '../../cartography/proportional';
import {GLOBAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './flight-matrix.md?raw';
import type {FlightMatrixOptions} from './flight-matrix.compute';
import {CONTINENT_NAMES} from './b11-geography';
import {
  getChanceTable,
  getContinentPalette,
  getCountTable,
  MATRIX_CAMERA_BOUNDS
} from './flight-matrix-style';
import {
  BETWEEN_GROUPS_INK,
  FLOW_CREDITS,
  FLOW_INK,
  getFlowWidthLegend,
  inkFor,
  withInkAlpha
} from './flows-style';

/** The world frame of the map steps (the same as the other airline stories). */
const WORLD_VIEW = {longitude: 10, latitude: 26, zoom: 1.75};

/** The cartouche of one step: the claim, the variable and method, and the frozen-data chip. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Frozen 2014'] as const
});

/** Caveat line of the matrix steps (the map steps keep the Mercator caveat). */
const MATRIX_CAVEAT = 'A cell counts airport pairs, not passengers; routes as of June 2014.';

/** A matrix is not geography: keep the paper ground, but remove its orientation labels. */
const MATRIX_BASEMAP = ground('paperSheet', {labels: 'none', labelPreset: 'none'});

export default defineScene<FlightMatrixOptions>({
  id: 'flight-matrix',
  title: 'The airline network as a matrix',
  chapter: 'flows',
  order: 8,
  summary:
    'The world airline network as an adjacency matrix: sort the airports by continent on the GPU and blocks appear, compare the blocks with what chance would give, and collapse the network into supernodes and superedges.',
  contributors: ['GPUAdjacencyMatrix', 'GPUAdjacencyMatrixOrder', 'GPUNetworkCoarsening'],
  datasets: [{id: 'openflights', role: 'airports and route pairs'}],
  initialView: {...WORLD_VIEW},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Show',
      group: 'View',
      apply: 'param',
      default: 'map',
      display: 'segmented',
      help: 'The network map, or the adjacency matrix on its own sheet. A matrix is not geography, so the story switches the ground with it.',
      options: [
        {value: 'map', label: 'Network map'},
        {value: 'matrix', label: 'Adjacency matrix'}
      ]
    },
    {
      kind: 'select',
      id: 'order',
      label: 'Order airports by',
      group: 'Matrix order',
      apply: 'param',
      default: 'input',
      display: 'chips',
      help: '`GPUAdjacencyMatrixOrder` sorts by a group label, then by a tie key; both are buffer writes, nothing recompiles. Changing the order slides every airport to its new position.',
      options: [
        {value: 'input', label: 'Input (A to Z)', help: 'Alphabetical IATA code: no structure.'},
        {
          value: 'shuffle',
          label: 'Shuffle (null)',
          help: 'Continent labels reassigned at random with the same group sizes, busiest first inside each: the control for blocks.'
        },
        {value: 'degree', label: 'Busiest first', help: 'By number of routes only: no groups.'},
        {
          value: 'continent',
          label: 'Continent',
          help: 'By continent, the busiest airport first in each block.'
        },
        {
          value: 'country',
          label: 'Country + hubs',
          help: 'By country inside continent, the busiest airport first in each block.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'resolutionPower',
      label: 'Resolution',
      group: 'Matrix cells',
      apply: 'compile',
      min: 7,
      max: 11,
      step: 1,
      default: 9,
      display: 'stepper',
      format: value => `${2 ** value} × ${2 ** value} bins`,
      describe: value =>
        `${2 ** value} bins requested. Past the number of airports in the window the cells cannot shrink (a cell never holds less than one airport per side), so the resolution is clamped to one cell per airport instead of skipping bins.`,
      help: 'Bins per axis of the matrix image, a compile-time option of `GPUAdjacencyMatrix`: each size compiles once, the first time you ask for it. Every drawn cell holds a whole number of airports per side.'
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Focus block',
      group: 'Zoom window',
      apply: 'param',
      default: 'all',
      disabledWhen: state => state.order === 'input' || state.order === 'degree',
      help: 'Zooms the window to one continent block. Needs a continent, shuffled or country order (blocks are contiguous in all three); with any other order the manual zoom is used.',
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
      display: 'segmented',
      help: 'Group label for `GPUNetworkCoarsening`. Labels are a buffer, so relabelling re-runs the same graph. Capacity is 256 groups.',
      options: [
        {value: 'continent', label: 'Continent'},
        {value: 'country', label: 'Country'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showGroups',
      label: 'Supernodes and superedges',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the coarsened network on the map: discs sized by airports, arcs sized by the routes between two groups.'
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'Routes',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Every route pair as a faint line: inside a continent in its hue, between continents in off-white.'
    },
    {
      kind: 'toggle',
      id: 'showAirports',
      label: 'Airports',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots sized by the number of routes, coloured by continent.'
    },
    {
      kind: 'toggle',
      id: 'showBlocks',
      label: 'Blocks and labels',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Continent (thick) and country (thin) block lines, the continent strips and the names beside the matrix.'
    },
    {
      kind: 'toggle',
      id: 'showChance',
      label: 'Tint blocks',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Tints every continent-by-continent block by observed over expected routes (orange: more than chance, purple: fewer).'
    },
    {
      kind: 'slider',
      id: 'arcWidth',
      label: 'Widest superedge',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 12,
      step: 1,
      default: 10,
      unit: 'px',
      help: 'Pixel width of the superedge with the most routes; the others follow the square root of their routes on the same scale.'
    }
  ],

  readouts: [
    {
      id: 'airports',
      label: 'Airports',
      format: 'integer',
      help: 'Airports with at least one route.'
    },
    {
      id: 'routes',
      label: 'Route pairs',
      format: 'integer',
      help: 'Undirected pairs of airports joined by at least one airline route.'
    },
    {
      id: 'degreeChart',
      label: 'Routes per airport',
      kind: 'chart',
      help: 'Airports by number of routes, in power-of-two bins: most have a few, a few hubs have hundreds. Hubs are the bright rows and columns of the matrix.'
    },
    {
      id: 'occupied',
      label: 'Cells occupied',
      format: 'integer',
      help: 'Cells of the shown matrix with at least one airport pair, counted from the GPU image (up to 1024 bins).'
    },
    {
      id: 'fillBound',
      label: 'Most cells that could fill',
      format: 'percent',
      help: 'The route slots divided by the cells: no matrix at this size can be fuller than this.'
    },
    {
      id: 'cellChart',
      label: 'How full is a cell?',
      kind: 'chart',
      help: 'Occupied cells by the class of their count, in the legend colours: nearly all hold one or two pairs.'
    },
    {
      id: 'cellScale',
      label: 'Cell size',
      help: 'Airports per cell side and cells per side. Every cell holds a whole number of airports; a window smaller than the resolution is clamped to one airport per cell.'
    },
    {id: 'blockAirports', label: 'Airports in the window', format: 'integer'},
    {
      id: 'orderCheck',
      label: 'Order check',
      help: 'The permutation `GPUAdjacencyMatrixOrder` computed, compared with its CPU twin `computeAdjacencyMatrixOrder`.'
    },
    {
      id: 'blockShare',
      label: 'Routes inside a block',
      format: 'percent',
      help: 'Route pairs whose two airports are in the same block of the current order; a dash when the order has no blocks.'
    },
    {
      id: 'expectedShare',
      label: 'Inside a block by chance',
      format: 'percent',
      help: 'The same share if every airport kept its number of routes and the route ends were paired at random: the sum over blocks of (route ends of the block / all route ends) squared.'
    },
    {
      id: 'modularity',
      label: 'Modularity Q',
      format: 'decimal',
      help: 'Observed minus expected share inside blocks (Newman). Near zero means the grouping explains nothing.'
    },
    {
      id: 'chanceMatrix',
      label: 'Routes against chance',
      kind: 'chart',
      help: 'Observed route ends between each pair of groups divided by the degree-preserving expectation D(a) D(b) / 2m, on a log scale: orange more than chance, purple fewer, grey as expected.'
    },
    {
      id: 'intraShare',
      label: 'Routes inside their group',
      format: 'percent',
      help: 'From the GPU: intra-group edges divided by all counted edges of the current Coarsen by grouping.'
    },
    {
      id: 'shareContinent',
      label: 'Inside a continent',
      format: 'percent',
      help: 'Route pairs whose airports are on the same continent (the CPU twin of the coarsening summary).'
    },
    {
      id: 'shareCountry',
      label: 'Inside a country',
      format: 'percent',
      help: 'Route pairs whose airports are in the same country.'
    },
    {id: 'superedges', label: 'Superedges', help: 'Distinct pairs of groups joined by a route.'},
    {
      id: 'largestPair',
      label: 'Busiest pair of groups',
      help: 'The superedge with the most routes.'
    },
    {id: 'window', label: 'Window', hood: true},
    {id: 'resolution', label: 'Matrix size', hood: true},
    {
      id: 'groups',
      label: 'Groups',
      format: 'integer',
      hood: true,
      help: 'Groups with at least one airport.'
    },
    {
      id: 'dropped',
      label: 'Dropped (label overflow)',
      format: 'integer',
      hood: true,
      help: 'Edges and airports with a label at or above the group capacity.'
    }
  ],

  pipeline: [
    {
      id: 'order',
      label: 'Two sorts',
      detail:
        'A group label and a tie key become a position per airport: two stable GPU sorts, a buffer write, never a recompile'
    },
    {
      id: 'bin',
      label: 'Bin pairs',
      detail:
        'One thread per airport walks its routes and adds one to the cell of its row and column position: exact integer atomics'
    },
    {
      id: 'draw',
      label: 'Class and draw',
      detail:
        'The raster layer reads the count buffer straight from the GPU and colours the integers by class'
    },
    {
      id: 'coarsen',
      label: 'Coarsen',
      detail:
        'Airports and routes counted per group; the group pairs are sorted and segmented into superedges'
    }
  ],

  legends: (state, data) => {
    const ground = (data.ground as MapGround | undefined) ?? 'light';
    const palette = getContinentPalette(ground);
    const sizes = data.continentSizes as number[] | undefined;
    const continents: LegendSpec = {
      kind: 'categories',
      title: 'Continents',
      entries: CONTINENT_NAMES.map((name, index) => ({
        color: palette[index],
        label: name,
        count: sizes?.[index],
        shape: 'dot' as const
      })),
      layout: 'list',
      note: 'One hue per continent on every dot, supernode and strip, whatever the order.'
    };
    const legends: LegendSpec[] = [];
    if (state.view === 'matrix') {
      const baseTable =
        (data.countTable as ReturnType<typeof getCountTable> | undefined) ?? getCountTable(ground);
      const cellCounts = data.cellCounts as number[] | undefined;
      const emptyCells = data.emptyCells as number | undefined;
      const table = {
        ...baseTable,
        noData: {...baseTable.noData, ...(emptyCells === undefined ? {} : {count: emptyCells})}
      };
      legends.push(
        getClassTableLegend(table, {
          title: 'Airport pairs per cell',
          id: 'matrix',
          counts: cellCounts,
          layout: 'list',
          note: 'Integer counts in fixed classes; an empty cell is left blank.'
        })
      );
      if (state.showChance) {
        const chance =
          (data.chanceTable as ReturnType<typeof getChanceTable> | undefined) ??
          getChanceTable(ground);
        legends.push(
          getClassTableLegend(chance, {
            title: 'Routes against chance',
            id: 'chance',
            layout: 'list',
            note: 'Observed over the degree-preserving expectation; orange is more than chance.'
          })
        );
      }
      if (state.showBlocks) legends.push(continents);
      return legends;
    }
    legends.push(continents);
    if (state.showAirports) {
      const maximumDegree = data.maximumDegree as number | undefined;
      if (maximumDegree) {
        legends.push({
          kind: 'size',
          title: 'Airport dot area ~ routes',
          entries: getSizeLegendEntries(maximumDegree, 6.5, {
            minRadiusPixels: 1.1,
            format: value => `${formatCount(value)} routes`
          }),
          layout: 'nested',
          note: 'Largest drawn first, so the small airports stay visible.'
        });
      }
    }
    if (state.showRoutes && !state.showGroups) {
      legends.push({
        kind: 'categories',
        title: 'Routes',
        entries: [
          {color: palette[0], label: 'Inside a continent (its hue)', shape: 'line'},
          {
            color: withInkAlpha(inkFor(BETWEEN_GROUPS_INK, ground), 255),
            label: 'Between continents',
            shape: 'line'
          }
        ],
        note: 'Between-continent routes are drawn on top.'
      });
    }
    if (state.showGroups) {
      const super_ = data.superNodes as
        | {maximum: number; maximumRadius: number; flowMaximum: number}
        | undefined;
      if (super_) {
        legends.push(
          {
            kind: 'size',
            title: 'Supernode area ~ airports',
            entries: getSizeLegendEntries(super_.maximum, super_.maximumRadius, {
              minRadiusPixels: 2.5,
              format: value => `${formatCount(value)} airports`
            }),
            layout: 'nested',
            note: 'Each supernode sits on its group’s busiest airport.'
          },
          getFlowWidthLegend({
            title: 'Superedge width ~ routes',
            maxValue: super_.flowMaximum,
            maxWidthPixels: state.arcWidth,
            color: inkFor(FLOW_INK, 'dark'),
            format: value => `${formatCount(value)} routes`
          })
        );
      }
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  GPUNetworkCoarsening,
  encodeGPUAdjacencyMatrixWindow
} from '@luma.gl/experimental/gpu-network';

// 1. airport -> matrix position: sort by group, then by tie key (two stable GPU sorts)
orderGraph.add(new GPUAdjacencyMatrixOrder({groups, tieKeys, order}));

// 2. R x R image of the CSR graph through that order (one graph per resolution)
matrixGraph.add(
  new GPUAdjacencyMatrix({
    offsets, neighbors, order: displayOrder,
    window: window.importToGraph(matrixGraph),   // [rowStart, rowEnd, colStart, colEnd)
    resolution: ${2 ** state.resolutionPower},
    output: {counts, maxCount}
  })
);
// A window holds fewer airports than bins? Make end - start an exact multiple of R, so every
// cell covers k whole airports and no bin is skipped:
const k = Math.ceil(extent / ${2 ** state.resolutionPower});
window.write(encodeGPUAdjacencyMatrixWindow(
  {rowStart, rowEnd: rowStart + k * ${2 ** state.resolutionPower}, colStart, colEnd: colStart + k * ${2 ** state.resolutionPower}},
  ${2 ** state.resolutionPower}
));

// 3. supernodes and superedges by label (${state.coarsenBy})
coarsenGraph.add(
  new GPUNetworkCoarsening({
    offsets, neighbors, labels,
    groupCapacity: 256,
    groupVertexCount, groupIntraEdgeCount,
    edges: {ids, count, overflow}, edgeTargets, edgeCounts, summary
  })
);
// Reorder: the matrix reads displayOrder; slide it from old to new positions, then copy the
// GPU permutation over it.
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: 'Previously: the airline network as a map, with its metrics and its bundles. Next: the same ideas on bike stations, [BIXI tides](#/story/bixi-tides).\n\n`GPUAdjacencyMatrixOrder` computes the row and column order from group labels and tie keys with two stable GPU sorts. `GPUAdjacencyMatrix` bins the airport pairs into a zoomable image of exact integer counts. `GPUNetworkCoarsening` collapses the graph by label into supernodes and superedges.',
    why: 'A node-link map keeps geography but hides structure when a graph is large; a matrix removes the overplotting but throws geography away, which sorting by continent puts back. Comparing the blocks with what chance would give keeps a big group from looking like a strong one.',
    howToRead:
      'Rows and columns are airports in the chosen order, and a cell counts the airport pairs it holds (classed integers; blank means no route). Blocks on the diagonal are routes inside a group, the others routes between groups. Data: OpenFlights, frozen in June 2014; continents come from a country list (Turkey and Russia are placed in Europe, Egypt in Africa); pairs are undirected and unweighted.'
  },

  // Map steps: the night ground; the matrix steps switch to the paper sheet.
  basemap: ground('night'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: cartouche('Where is the airline network organised?', 'Route pairs and airports, 2014'),
    credit: joinCredits(FLOW_CREDITS.openFlights, CREDITS.colorBrewer, CREDITS.okabeIto)
  },

  create: async ctx => (await import('./flight-matrix.compute')).createFlightMatrix(ctx),

  story: storyFromMarkdown<FlightMatrixOptions>(narrative, {
    question: {
      headline: 'A hairball of routes hides any structure',
      textAlternative:
        'Dark world map of faint route lines and airports as dots coloured by continent, sized by number of routes, with the busiest airports of each continent named.',
      optionsMode: 'fresh',
      options: {view: 'map', showRoutes: true, showAirports: true, showGroups: false},
      controls: ['showRoutes'],
      readouts: ['airports', 'routes', 'degreeChart'],
      camera: {...WORLD_VIEW, transitionMs: 1400},
      basemap: ground('night'),
      furniture: {
        ...GLOBAL_FURNITURE,
        title: cartouche('Where is the structure?', 'Route pairs and airports, 2014')
      }
    },
    matrix: {
      headline: 'Alphabetical order turns the matrix into noise',
      textAlternative:
        'A square matrix on a blank paper sheet with scattered blue-green cells and no pattern, airports in alphabetical order.',
      optionsMode: 'fresh',
      options: {view: 'matrix', order: 'input', showBlocks: true},
      controls: ['order'],
      readouts: ['occupied', 'fillBound', 'cellChart'],
      stage: 'bin',
      camera: {bounds: MATRIX_CAMERA_BOUNDS, transitionMs: 1600},
      basemap: MATRIX_BASEMAP,
      furniture: {
        scaleBar: false,
        caveat: MATRIX_CAVEAT,
        title: cartouche('Unordered, the matrix is noise', 'Airport × airport, in input order')
      }
    },
    order: {
      headline: 'Sorted by continent, the matrix shows blocks',
      textAlternative:
        'The same matrix sorted by continent: bright blocks on the diagonal in six continent colours, with the busiest airports of each block at its top left; the shuffled order is an even speckle.',
      optionsMode: 'fresh',
      options: {view: 'matrix', order: 'continent', showBlocks: true, coarsenBy: 'continent'},
      controls: ['order'],
      readouts: ['blockShare', 'orderCheck'],
      stage: 'order',
      camera: {bounds: MATRIX_CAMERA_BOUNDS, transitionMs: 1000},
      basemap: MATRIX_BASEMAP,
      furniture: {
        scaleBar: false,
        caveat: MATRIX_CAVEAT,
        title: cartouche('Sort by continent and blocks appear', 'Airport × airport, by continent')
      }
    },
    chance: {
      headline: 'Far more routes stay home than chance predicts',
      textAlternative:
        'The continent-sorted matrix with each block tinted orange or purple by observed over expected routes, and a six by six chart of the same ratios.',
      optionsMode: 'fresh',
      options: {
        view: 'matrix',
        order: 'continent',
        showBlocks: true,
        showChance: true,
        coarsenBy: 'continent'
      },
      controls: ['showChance', 'order'],
      readouts: ['chanceMatrix', 'blockShare', 'expectedShare', 'modularity'],
      stage: 'draw',
      camera: {bounds: MATRIX_CAMERA_BOUNDS, transitionMs: 1000},
      basemap: MATRIX_BASEMAP,
      furniture: {
        scaleBar: false,
        caveat: MATRIX_CAVEAT,
        title: cartouche('Far more routes stay home than chance', 'Observed over expected routes')
      }
    },
    'one-block': {
      headline: 'Inside a continent, countries form their own blocks',
      textAlternative:
        'One continent block of the matrix filling the card, divided into country blocks with the largest countries named at the left.',
      optionsMode: 'fresh',
      options: {
        view: 'matrix',
        order: 'country',
        focus: 'Europe',
        resolutionPower: 9,
        showBlocks: true
      },
      controls: ['resolutionPower', 'focus'],
      readouts: ['cellScale', 'blockAirports', 'occupied'],
      stage: 'bin',
      camera: {bounds: MATRIX_CAMERA_BOUNDS, transitionMs: 1000},
      basemap: MATRIX_BASEMAP,
      furniture: {
        scaleBar: false,
        caveat: MATRIX_CAVEAT,
        title: cartouche('Inside Europe, countries form blocks', 'Airport × airport, by country')
      }
    },
    coarsen: {
      headline: 'The grouping you choose sets how modular it looks',
      textAlternative:
        'Dark world map with one disc per continent on its busiest airport, sized by airports, joined by gold arcs sized by routes between continents.',
      optionsMode: 'fresh',
      options: {
        view: 'map',
        showRoutes: false,
        showAirports: true,
        showGroups: true,
        coarsenBy: 'continent'
      },
      controls: ['coarsenBy', 'order', 'view'],
      readouts: ['shareContinent', 'shareCountry', 'superedges', 'largestPair'],
      stage: 'coarsen',
      camera: {...WORLD_VIEW, transitionMs: 1600},
      basemap: ground('night'),
      furniture: {
        ...GLOBAL_FURNITURE,
        title: cartouche('Collapse it: continents and countries', 'Supernodes and superedges, 2014')
      }
    }
  })
});
