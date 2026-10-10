// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import type {LegendSpec} from '../scene';
import {defineScene} from '../scene';
import {CATEGORY_SELECT_OPTIONS, CHICAGO_VIEW, MONTH_NAMES} from './b8-common';
import type {H3AggregationOptions} from './h3-aggregation.compute';

const MONTROSE = [-87.6325, 41.9625] as const;

const tierColors = [
  [90, 110, 190, 255],
  [70, 190, 210, 255],
  [110, 220, 130, 255],
  [250, 210, 80, 255],
  [255, 110, 90, 255]
] as const;

const FAMILY_NAMES = {
  h3: 'H3',
  quadbin: 'Quadbin',
  quadkey: 'Quadkey',
  geohash: 'Geohash',
  s2: 'S2'
} as const;

const isTabularOnly = (state: H3AggregationOptions) =>
  state.family !== 'h3' && state.family !== 'quadbin';

/** Chicago observations keyed to H3, Quadbin and other grid cells on the GPU. GPU work is in `h3-aggregation.compute.ts`. */
export default defineScene<H3AggregationOptions>({
  id: 'h3-aggregation',
  title: 'Chicago observations by global grid cell',
  chapter: 'cells',
  order: 1,
  summary:
    'GPU cell indexing aggregates the 2023 Chicago iNaturalist observations across H3, Quadbin and three comparison grids. The outputs include counts, topology and polygon covers; finite capacity and observer effort limit interpretation.',
  contributors: [
    'GPUPointToCell',
    'GPUCellAggregation',
    'GPUCellGeometry',
    'GPUCellTopology',
    'GPUCellCompaction',
    'GPUCellSetOutline',
    'GPUCellCover'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-community-areas', role: 'polygons to polyfill'}
  ],
  initialView: {...CHICAGO_VIEW},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Chicago observations by grid cell',
      subtitle: 'Counts, occupied-set topology and community-area covers'
    },
    scaleBar: {units: 'metric'},
    credit: 'iNaturalist contributors · City of Chicago Data Portal',
    caveat: 'Cell counts reflect observer effort and the selected grid resolution.'
  },

  options: [
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Observations',
      apply: 'param',
      default: 'all',
      help: 'Keeps one iNaturalist group (plants, birds, insects, ...). The observation mask is a buffer write: only the index graph re-encodes, nothing recompiles.',
      options: CATEGORY_SELECT_OPTIONS
    },
    {
      kind: 'range',
      id: 'months',
      label: 'Months of 2023',
      group: 'Observations',
      apply: 'param',
      min: 1,
      max: 12,
      step: 1,
      default: [1, 12],
      format: value => MONTH_NAMES[Math.round(value) - 1],
      help: 'Keeps observations from the first to the last chosen month (inclusive).'
    },
    {
      kind: 'slider',
      id: 'sampledPercent',
      label: 'Sampled share of observations',
      group: 'Observations',
      apply: 'param',
      min: 5,
      max: 100,
      step: 5,
      default: 100,
      unit: '%',
      help: 'Thins the observations with a deterministic hash. Counts per cell drop proportionally; occupied cells at fine resolutions drop faster.'
    },
    {
      kind: 'select',
      id: 'family',
      label: 'Cell family',
      group: 'Grid',
      apply: 'compile',
      default: 'h3',
      help: 'The family fixes the key layout and the kernels, so changing it recompiles the graphs once after a short pause. H3 and Quadbin have aggregation, topology, cover and outline contributors; quadkey, geohash and S2 only key and decode (one translucent cell per observation, overlap is the density).',
      options: [
        {value: 'h3', label: 'H3 (hexagons, table + topology + cover)'},
        {value: 'quadbin', label: 'Quadbin (Web Mercator tiles, table + topology + cover)'},
        {value: 'quadkey', label: 'Quadkey (per-observation cells)'},
        {value: 'geohash', label: 'Geohash (per-observation cells)'},
        {value: 's2', label: 'S2 (per-observation cells)'}
      ]
    },
    {
      kind: 'slider',
      id: 'resolutionOffset',
      label: 'Resolution (relative to default)',
      group: 'Grid',
      apply: 'compile',
      min: -2,
      max: 2,
      step: 1,
      default: 0,
      format: value => `${value >= 0 ? '+' : ''}${value}`,
      help: 'Steps finer (+) or coarser (-) than the family default: H3 8, Quadbin 15, quadkey 15, geohash 6, S2 14. Each step is about 2x smaller cells in each direction (H3: 2.65x). The Grid readout shows the actual level and cell size. Compile-time.'
    },
    {
      kind: 'select',
      id: 'outline',
      label: 'Cell-set outline',
      group: 'Outline (GPUCellSetOutline)',
      apply: 'param',
      default: 'off',
      disabledWhen: isTabularOnly,
      help: 'All three outline graphs are compiled with the index graph; this selects which one is drawn. Segments come straight from the contributor with a GPU-written count.',
      options: [
        {value: 'off', label: 'Off'},
        {value: 'boundary', label: 'Boundary of the occupied set (segments)'},
        {value: 'rings', label: 'Closed rings with holes (ring assembly)'},
        {value: 'groups', label: 'Borders between count tiers (group labels)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSelection',
      label: 'Show clicked cell and neighbourhood',
      group: 'Neighbourhood (click the map)',
      apply: 'param',
      default: false,
      disabledWhen: isTabularOnly,
      help: 'Click anywhere: GPUPointToCell keys the click and GPUCellTopology expands it. Magenta outline = the clicked cell.'
    },
    {
      kind: 'select',
      id: 'selectionOperation',
      label: 'Topology operation',
      group: 'Neighbourhood (click the map)',
      apply: 'compile',
      default: 'disk',
      disabledWhen: isTabularOnly,
      help: 'Disk: every cell within grid distance k. Ring: only cells at distance exactly k. Parent: the ancestor some levels coarser. Children: all descendants some levels finer. The operation fixes the output stride, so the graph for each is compiled the first time you pick it.',
      options: [
        {value: 'disk', label: 'Grid disk (gridDisk, with distances)'},
        {value: 'ring', label: 'Grid ring (gridRing)'},
        {value: 'parent', label: 'Parent cell (cellToParent)'},
        {value: 'children', label: 'Children (cellToChildren)'}
      ]
    },
    {
      kind: 'slider',
      id: 'diskRadius',
      label: 'Disk radius k',
      group: 'Neighbourhood (click the map)',
      apply: 'param',
      min: 0,
      max: 8,
      step: 1,
      default: 3,
      disabledWhen: state => isTabularOnly(state) || state.selectionOperation !== 'disk',
      help: 'Filters the always-computed k = 8 disk by its distance column on the GPU, so the slider never recompiles.'
    },
    {
      kind: 'slider',
      id: 'ringRadius',
      label: 'Ring radius k',
      group: 'Neighbourhood (click the map)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      disabledWhen: state => isTabularOnly(state) || state.selectionOperation !== 'ring',
      help: 'A ring has 6k cells for H3 and 8k for Quadbin. k sets the output stride, so each radius is its own compiled graph.'
    },
    {
      kind: 'slider',
      id: 'levelsUp',
      label: 'Parent levels up',
      group: 'Neighbourhood (click the map)',
      apply: 'compile',
      min: 1,
      max: 4,
      step: 1,
      default: 1,
      disabledWhen: state => isTabularOnly(state) || state.selectionOperation !== 'parent',
      help: 'How many resolutions coarser the parent is. The cell resolution of the parent is a kernel constant.'
    },
    {
      kind: 'slider',
      id: 'levelsDown',
      label: 'Children levels down',
      group: 'Neighbourhood (click the map)',
      apply: 'compile',
      min: 1,
      max: 3,
      step: 1,
      default: 1,
      disabledWhen: state => isTabularOnly(state) || state.selectionOperation !== 'children',
      help: 'How many resolutions finer the children are: 7 per level for H3 (7, 49, 343), 4 per level for Quadbin (4, 16, 64).'
    },
    {
      kind: 'toggle',
      id: 'showCover',
      label: 'Polyfill the 77 community areas',
      group: 'Community-area cover (GPUCellCover)',
      apply: 'param',
      default: false,
      disabledWhen: isTabularOnly,
      help: 'Draws the cells GPUCellCover finds for every community area, with the area borders on top. Hover an area for its observation and cover-cell counts.'
    },
    {
      kind: 'select',
      id: 'containment',
      label: 'Containment rule',
      group: 'Community-area cover (GPUCellCover)',
      apply: 'compile',
      default: 'center',
      disabledWhen: state => state.family !== 'quadbin',
      help: 'center: the cell centre is inside the polygon (the only rule H3 supports). full: the whole cell is inside. intersects: any overlap. Quadbin only; compile-time.',
      options: [
        {value: 'center', label: 'center: centre inside'},
        {value: 'full', label: 'full: whole cell inside'},
        {value: 'intersects', label: 'intersects: any overlap'}
      ]
    },
    {
      kind: 'select',
      id: 'coverColor',
      label: 'Cover color',
      group: 'Community-area cover (GPUCellCover)',
      apply: 'param',
      default: 'area',
      disabledWhen: isTabularOnly,
      help: 'By community area (8 colors cycle), or by the core flag: core cells are provably inside their polygon, border cells straddle an edge.',
      options: [
        {value: 'area', label: 'By community area'},
        {value: 'core', label: 'Core cells vs border cells'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showCompacted',
      label: 'Outline the compacted cover',
      group: 'Community-area cover (GPUCellCompaction)',
      apply: 'param',
      default: false,
      disabledWhen: isTabularOnly,
      help: 'Yellow outlines show the compacted set: groups of sibling cells merged into their parent, recursively. A matching uncompact node expands it back and the readout compares the counts.'
    },
    {
      kind: 'slider',
      id: 'compactDepth',
      label: 'Merge up to N levels',
      group: 'Community-area cover (GPUCellCompaction)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 2,
      disabledWhen: isTabularOnly,
      help: 'The coarsest cell a merge may reach is N resolutions above the cover (minimumResolution), and the uncompact reverses at most N levels (maximumDepth). Compile-time.'
    },
    {
      kind: 'toggle',
      id: 'showCells',
      label: 'Show cells',
      group: 'Cell color',
      apply: 'param',
      default: true,
      help: 'Draws the aggregated cells (or the per-observation cells), colored by observations per cell.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Cell color',
      apply: 'param',
      default: 'ylgnbu',
      disabledWhen: isTabularOnly,
      help: 'Sequential ramp for observations per cell; yellow-green-blue is the paper-ground default.',
      options: [
        {value: 'ylgnbu', label: 'Yellow-green-blue'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Value scale',
      group: 'Cell color',
      apply: 'param',
      default: 'sqrt',
      disabledWhen: isTabularOnly,
      help: 'Observation counts are heavy tailed: the square root lifts ordinary cells so they are not all dark next to Montrose Point.',
      options: [
        {value: 'sqrt', label: 'Square root'},
        {value: 'linear', label: 'Linear'}
      ]
    },
    {
      kind: 'slider',
      id: 'ceiling',
      label: 'Color ceiling',
      group: 'Cell color',
      apply: 'param',
      min: 5,
      max: 100,
      step: 5,
      default: 100,
      unit: '% of the busiest cell',
      help: 'The count that maps to the top of the ramp, as a share of the busiest cell. Lower it to separate ordinary cells; busier cells saturate. For per-observation families it raises the opacity of each cell.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations in the filter'},
    {id: 'grid', label: 'Family and resolution'},
    {
      id: 'cellSize',
      label: 'Cell size at Chicago',
      help: 'Approximate edge length (H3, S2) or width (Quadbin tile, geohash).'
    },
    {id: 'exactness', label: 'f32 accuracy of the keys'},
    {id: 'occupied', label: 'Occupied cells'},
    {id: 'binned', label: 'Observations binned'},
    {id: 'busiest', label: 'Busiest cell'},
    {id: 'outlineSegments', label: 'Occupied-set boundary'},
    {id: 'rings', label: 'Assembled rings'},
    {id: 'groupBorders', label: 'Tier borders'},
    {id: 'selected', label: 'Clicked cell'},
    {id: 'selection', label: 'Topology result'},
    {id: 'cover', label: 'Cover cells'},
    {id: 'core', label: 'Core and border cells'},
    {id: 'compacted', label: 'Compacted cells'},
    {id: 'roundTrip', label: 'Uncompact round trip'},
    {id: 'memory', label: 'GPU buffers', format: 'bytes'}
  ],

  legends: state => {
    const tabular = !isTabularOnly(state);
    const legends: LegendSpec[] = [];
    if (state.showCells) {
      if (tabular) {
        legends.push({
          kind: 'ramp',
          id: 'cells',
          title: `Observations per ${state.family === 'h3' ? 'hexagon' : 'tile'}`,
          ramp: state.ramp,
          extent: 'gpu',
          sqrtScale: state.scale === 'sqrt',
          unit: 'observations',
          format: value => Math.round(value).toLocaleString('en-US')
        });
      } else {
        legends.push({
          kind: 'categories',
          title: 'One translucent cell per observation',
          entries: [
            {color: [255, 190, 90, 255], label: 'Overlap adds up: brighter = more observations'}
          ],
          note: `${FAMILY_NAMES[state.family]} has no aggregation contributor, so GPUPointToCell and GPUCellGeometry draw every observation's own cell.`
        });
      }
    }
    if (tabular && state.outline === 'groups') {
      legends.push({
        kind: 'categories',
        title: 'Border between count tiers',
        entries: ['1', '2-3', '4-7', '8-15', '16+'].map((label, index) => ({
          color: tierColors[index],
          label: `${label} observations`
        })),
        note: 'Tier = floor(log2(count)), capped at 4. A border is drawn on the side of the cell that belongs to the tier, wherever the neighbour differs.'
      });
    } else if (tabular && state.outline === 'boundary') {
      legends.push({
        kind: 'categories',
        title: 'Outline',
        entries: [
          {color: [255, 255, 255, 245], label: 'Edge between an occupied and an empty cell'}
        ]
      });
    } else if (tabular && state.outline === 'rings') {
      legends.push({
        kind: 'categories',
        title: 'Closed rings',
        entries: [
          {color: [255, 255, 255, 255], label: 'Ring 0'},
          {color: [255, 214, 64, 255], label: 'Ring 1'},
          {color: [120, 230, 255, 255], label: 'Ring 2'},
          {color: [255, 150, 200, 255], label: 'Ring 3, then repeating'}
        ],
        note: 'Shells run counter-clockwise, holes clockwise.'
      });
    }
    if (tabular && state.showCover) {
      legends.push(
        state.coverColor === 'core'
          ? {
              kind: 'categories',
              title: 'Cover cells',
              entries: [
                {color: [60, 215, 140, 215], label: 'Core: provably inside'},
                {color: [255, 150, 60, 215], label: 'Border: touches a polygon edge'}
              ]
            }
          : {
              kind: 'categories',
              title: 'Cover cells by community area',
              entries: [{color: [78, 201, 255, 190], label: 'Area index mod 8 (colors cycle)'}]
            }
      );
      if (state.showCompacted) {
        legends.push({
          kind: 'categories',
          title: 'Compaction',
          entries: [{color: [255, 244, 120, 255], label: 'Outline of each compacted cell'}]
        });
      }
    }
    if (tabular && state.showSelection) {
      legends.push({
        kind: 'categories',
        title: 'Clicked neighbourhood',
        entries: [
          {color: [255, 40, 120, 255], label: 'Clicked cell'},
          {
            color:
              state.selectionOperation === 'children'
                ? [110, 230, 130, 255]
                : state.selectionOperation === 'parent'
                  ? [255, 220, 80, 255]
                  : state.selectionOperation === 'ring'
                    ? [255, 140, 60, 255]
                    : [40, 220, 255, 255],
            label: {
              disk: 'Grid disk, fading with distance',
              ring: 'Grid ring at distance k',
              parent: 'Parent cell',
              children: 'Child cells'
            }[state.selectionOperation]
          }
        ]
      });
    }
    return legends;
  },

  snippet: state => {
    const tabular = !isTabularOnly(state);
    const family = `'${state.family}'`;
    const lines = [
      `import {`,
      `  GPUPointToCell, ${tabular ? 'GPUCellAggregation, ' : ''}GPUCellGeometry${tabular ? ',' : ''}`,
      ...(tabular ? [`  GPUCellSetOutline, GPUCellTopology, GPUCellCover, GPUCellCompaction`] : []),
      `} from '@luma.gl/experimental/gpu-spatial-analysis';`,
      ``,
      `// 1. Longitude/latitude to a 64-bit cell key per observation (observation mask = per-frame buffer).`,
      `graph.add(new GPUPointToCell({family: ${family}, resolution, positions, mask, output: {cells: keys}}));`
    ];
    if (tabular) {
      lines.push(
        ``,
        `// 2. Sort + count into a table of (cell, count), keys ascending, capacity-bounded.`,
        `graph.add(new GPUCellAggregation({family: ${family}, resolution, cells: keys,`,
        `  output: {cells: tableCells, counts, count, overflow}}));`
      );
    }
    lines.push(
      ``,
      `// ${tabular ? '3' : '2'}. Decode every key into boundary vertices that the layer reads from storage.`,
      `graph.add(new GPUCellGeometry({family: ${family}, cells: ${tabular ? 'tableCells' : 'keys'},`,
      `  maximumVertexCount: ${state.family === 'h3' ? 10 : 4}, output: {boundaries, vertexCounts}}));`
    );
    if (tabular) {
      if (state.outline !== 'off') {
        lines.push(
          ``,
          `// Boundary of the occupied set${state.outline === 'groups' ? ' (an edge is kept when the neighbour has another group label)' : ''}.`,
          `graph.add(new GPUCellSetOutline({family: ${family}, cells: tableCells, count${state.outline === 'groups' ? ', groups: tiers' : ''},`,
          `${state.outline === 'rings' ? '  rings: {normalizeWinding: true, output: ringOutput},\n' : ''}  output: {rows, cells, edgeIndices, endpoints${state.outline === 'groups' ? ', groups' : ''}, count, overflow}}));`
        );
      }
      if (state.showSelection) {
        const operation = {
          disk: `{type: 'disk', k: 8}`,
          ring: `{type: 'ring', k: ${state.ringRadius}}`,
          parent: `{type: 'parent', resolution: resolution - ${state.levelsUp}}`,
          children: `{type: 'children', resolution: resolution + ${state.levelsDown}, inputResolution: resolution}`
        }[state.selectionOperation];
        lines.push(
          ``,
          `// Click: key the position, then expand it.`,
          `graph.add(new GPUCellTopology({family: ${family}, operation: ${operation},`,
          `  cells: clickedCell, output: {cells: result${state.selectionOperation === 'disk' ? ', distances' : ''}, counts}}));`
        );
      }
      if (state.showCover) {
        lines.push(
          ``,
          `// Polyfill the community areas.`,
          `graph.add(new GPUCellCover({family: ${family}, resolution, containment: '${state.family === 'h3' ? 'center' : state.containment}',`,
          `  polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity: 1 << 20,`,
          `  output: {featureIds, cells: coverCells, core, count, overflow}}));`
        );
        if (state.showCompacted) {
          lines.push(
            `graph.add(new GPUCellCompaction({family: ${family},`,
            `  operation: {type: 'compact', resolution, minimumResolution: resolution - ${state.compactDepth}},`,
            `  cells: coverCells, count: coverCount, output: {cells: compacted, count, overflow}}));`
          );
        }
      }
    }
    return lines.join('\n');
  },

  about: {
    what: '`GPUPointToCell` turns every longitude/latitude into a 64-bit cell key (Quadbin, H3, quadkey, geohash or S2). `GPUCellAggregation` sorts the keys and counts them into a compact table, `GPUCellGeometry` decodes the table into boundary polygons, `GPUCellSetOutline` traces the borders of the occupied set, `GPUCellTopology` expands one cell into a disk, ring, parent or children, and `GPUCellCover` plus `GPUCellCompaction` polyfill and compact polygons.',
    why: 'A discrete global grid is a common key. Once observations, 311 calls, places and census counts sit in the same cells they can be joined, compared across years and rolled up to any scale without geometry tests. The GPU does all of it for 261 thousand points without a round trip to the CPU.',
    howToRead:
      'Each cell is colored by the observations inside it (legend, square-root scale). White or colored lines trace where occupied cells meet empty ones. The magenta cell is the one you clicked; cyan to orange cells are its topological neighbours. Cover cells are the cells inside a community area.'
  },

  create: async ctx => (await import('./h3-aggregation.compute')).createH3Aggregation(ctx),

  story: [
    {
      id: 'question',
      title: 'Where do sightings pile up, at any scale?',
      headline: 'Lakefront cells contain the highest counts',
      textAlternative:
        'Blue-green H3 hexagons cover Chicago, with the highest observation counts concentrated near Montrose Point and other lakefront sites.',
      body: 'People logged **43,557 wild plants, animals and fungi in Chicago on iNaturalist in 2023**. A dot map hides the answer behind overplotting, and a choropleth by community area forces one scale on every question.\n\nA **discrete global grid** gives a better common key: every place on Earth has a cell ID at every resolution. This map is already that answer: each observation was keyed to an **H3 hexagon** and counted, all on the GPU. Bright hexagons hold the most observations (legend, square-root scale). Filter by **Group** or **Months of 2023** below to see how the pattern moves; the next steps open the pipeline one contributor at a time.',
      options: {family: 'h3', resolutionOffset: 0},
      camera: {...CHICAGO_VIEW, transitionMs: 1200},
      highlight: {readout: 'busiest'},
      controls: ['category', 'months'],
      readouts: ['points', 'busiest']
    },
    {
      id: 'resolution',
      title: 'Keys first, then counts',
      headline: 'Finer cells separate local observation concentrations',
      textAlternative:
        'A finer H3 grid divides broad concentrations into smaller occupied hexagons while readouts report grid resolution, occupancy and busiest-cell count.',
      body: '**`GPUPointToCell`** turns each longitude/latitude into a 64-bit H3 key. **`GPUCellAggregation`** then radix-sorts the keys and counts runs, writing a compact table of `(cell, count)` with integer atomics, so the result is exact and does not depend on thread order. Readouts show how many cells are occupied and the busiest one.\n\nSlide **Resolution (relative to default)** below by one step. H3 resolution 9 cells are about 2.7 times smaller in each direction: the Montrose Point hot spot splits into the harbor, the dunes and the meadow and the hot spots sharpen, while the map gets noisier. Changing it recompiles the graphs once (the badge says so); the observation mask, in contrast, is only a buffer write.',
      options: {resolutionOffset: 1},
      camera: {longitude: MONTROSE[0], latitude: MONTROSE[1], zoom: 12.6, transitionMs: 1600},
      callout: {coordinate: MONTROSE, text: 'Montrose Point: the busiest cells'},
      highlight: {readout: 'occupied'},
      controls: ['resolutionOffset', 'category'],
      readouts: ['grid', 'occupied', 'busiest']
    },
    {
      id: 'families',
      title: 'Five grids, one pipeline',
      headline: 'Grid families partition the same observations differently',
      textAlternative:
        'H3, Quadbin, quadkey, geohash and S2 cells can replace one another over the same Chicago observations, revealing their different shapes and sizes.',
      body: 'The same keying runs for **Quadbin** (Web Mercator tiles, as in CARTO), quadkey, geohash and S2. Quadbin keys are bit-exact integers. H3 and S2 use f32 sphere math, so the readout shows how accuracy degrades at finer resolutions: use H3 up to about res 10 here.\n\nOnly H3 and Quadbin have an aggregation table. Set **Cell family** to *Quadkey*, *Geohash* or *S2*: `GPUCellGeometry` decodes one cell per observation and translucent overlap becomes the density.',
      options: {family: 'quadbin', resolutionOffset: 0},
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.9, transitionMs: 1400},
      highlight: {readout: 'cellSize'},
      controls: ['family', 'resolutionOffset'],
      readouts: ['cellSize', 'exactness']
    },
    {
      id: 'outlines',
      title: 'Outline the occupied set',
      headline: 'Boundary segments isolate the occupied cell set',
      textAlternative:
        'Cell fills are supplemented by an outline around occupied cells or borders between count tiers, with segment and ring totals reported.',
      body: '**`GPUCellSetOutline`** finds, for every cell, which edges face a neighbour that is not in the table, and emits only those segments. Set **Cell-set outline** below to *Closed rings*: the contributor can also assemble the segments into closed rings with shells and holes. Choose *Borders between count tiers* and it labels each cell with `floor(log2(count))`, so edges appear wherever the neighbour sits in another tier: instant isolines of observation density.\n\nSegment counts come from the GPU, so the layer draws exactly that many instances through an indirect draw record.',
      options: {family: 'h3', resolutionOffset: 0, outline: 'groups'},
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.9, transitionMs: 1400},
      highlight: {readout: 'groupBorders'},
      controls: ['outline'],
      readouts: ['outlineSegments', 'rings', 'groupBorders']
    },
    {
      id: 'neighbourhood',
      title: 'Click a cell: its neighbourhood',
      headline: 'Selected cells expand into ordered neighbourhoods',
      textAlternative:
        'A selected Chicago cell has a magenta outline and surrounding cyan cells fade with grid distance for disk, ring, parent or child operations.',
      body: '**`GPUCellTopology`** expands one cell into its grid **disk** (all cells within distance k), a **ring** (distance exactly k), its **parent** at a coarser resolution, or its **children** at a finer one. Click any hexagon. The magenta outline is the clicked cell; cyan cells fade with distance. **Topology operation** below switches between the four.\n\nThe disk is always computed at k = 8 and **Disk radius k** only filters its distance column on the GPU. The readout adds up the observations of the neighbourhood: a quick "how much is seen around here" without any spatial join.',
      options: {outline: 'off', showSelection: true, selectionOperation: 'disk', diskRadius: 3},
      camera: {longitude: MONTROSE[0], latitude: MONTROSE[1], zoom: 13, transitionMs: 1600},
      callout: {coordinate: MONTROSE, text: 'Default selection'},
      highlight: {readout: 'selection'},
      controls: ['showSelection', 'selectionOperation', 'diskRadius'],
      readouts: ['selected', 'selection']
    },
    {
      id: 'cover',
      title: 'Polyfill the community areas, then compact',
      headline: 'Compaction preserves the community-area cell cover',
      textAlternative:
        'Cells fill Chicago community-area polygons; compacted parent outlines reduce representation while the round-trip readout confirms the original cover count.',
      body: '**`GPUCellCover`** does the reverse of keying: it fills polygons with cells. Here are the 77 Chicago community areas. With H3 only the **center** rule exists; with Quadbin (set **Cell family** below) you can compare **full** (whole cell inside) and **intersects** (any overlap) with **Containment rule**, which bracket the true area. Set **Cover color** to *Core cells vs border cells* to see which cells are provably inside, so a point join could skip the exact polygon test.\n\n**`GPUCellCompaction`** merges complete groups of sibling cells into their parent (**Outline the compacted cover**, yellow outlines; **Merge up to N levels** sets the reach) and an uncompact node expands it back: the readout confirms the round trip returns the same number of cells.',
      options: {
        showSelection: false,
        showCover: true,
        coverColor: 'core',
        showCompacted: true,
        showCells: false,
        compactDepth: 2
      },
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.9, transitionMs: 1600},
      highlight: {readout: 'compacted'},
      controls: ['family', 'containment', 'coverColor', 'showCompacted', 'compactDepth'],
      readouts: ['core', 'compacted', 'roundTrip']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      headline: 'Fine grids expose capacity and precision limits',
      textAlternative:
        'The cell map remains visible with controls for family, resolution and observation group, and readouts indicate overflow or reduced key precision.',
      body: 'Caveats: the table holds 65,536 cells and says OVERFLOW when a resolution is too fine; and observations follow the observers, so cells measure **effort** (where people look and upload) as much as wildlife.\n\nTry it: set **Group** to *Fungi* or *Birds* and **Months of 2023** to a single season; set **Topology operation** to *Children* and click a cell; or set **Cell family** to *Geohash* with **Resolution (relative to default)** at +2 to see where the f32 keys and a 5-character cell stop making sense.',
      options: {
        showCells: true,
        showCover: false,
        showCompacted: false,
        showSelection: true,
        coverColor: 'area'
      },
      camera: {...CHICAGO_VIEW, transitionMs: 1400},
      controls: ['category', 'months', 'selectionOperation', 'family', 'resolutionOffset']
    }
  ]
});
