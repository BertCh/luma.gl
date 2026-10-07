// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import type {OsmDrawsItselfOptions} from './osm-history-draws-itself.compute';

/** Time origin of the dataset (first node creation, 27 Jun 2007 20:05 UTC), in epoch ms. */
const ORIGIN_MS = 1_182_974_728_000;
const MAX_DAY = 6_900;
const dayOf = (year: number, month: number, day = 1) =>
  Math.round((Date.UTC(year, month - 1, day) - ORIGIN_MS) / 86_400_000 / 7) * 7;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const formatDay = (day: number) => {
  const date = new Date(ORIGIN_MS + day * 86_400_000);
  return `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
};

export default defineScene<OsmDrawsItselfOptions>({
  id: 'osm-history-draws-itself',
  title: 'How does New York draw itself on OpenStreetMap?',
  chapter: 'dataframe',
  order: 1,
  summary:
    'Replay 19 years of OpenStreetMap node creations in New York City: 400,000 sampled nodes appear as ink on the day a mapper placed them, while a GPU space-time reduction counts nodes per cell and per time bucket.',
  contributors: ['GPUTemporalReduction', 'GPUTimeWindowFilter'],
  datasets: [
    {
      id: 'poopdeck-osm-nyc',
      role: 'node creations 2007-2026 (400k sample), © OpenStreetMap contributors'
    }
  ],
  initialView: {longitude: -73.97, latitude: 40.71, zoom: 10.4},

  options: [
    ...playbackOptions<OsmDrawsItselfOptions>({
      playing: false,
      time: {
        min: 0,
        max: MAX_DAY,
        step: 7,
        default: MAX_DAY,
        label: 'Date',
        format: formatDay,
        help: 'Nodes created up to this date are drawn. 0 is 27 Jun 2007, the first node; the last is in May 2026.'
      },
      speed: {
        min: 0.25,
        max: 4,
        step: 0.25,
        default: 1,
        unit: 'x',
        help: 'At 1x the 19 years play in 60 seconds, about 4 days per frame.'
      },
      loop: true
    }),
    {
      kind: 'toggle',
      id: 'showPoints',
      label: 'Show nodes',
      group: 'Ink (time window)',
      apply: 'param',
      default: true,
      help: 'Draws every sampled node created before the date. GPUTimeWindowFilter selects them on the GPU and writes the draw count.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color nodes by',
      group: 'Ink (time window)',
      apply: 'param',
      default: 'year',
      disabledWhen: state => !state.showPoints,
      help: 'Creation date on a ramp, or the kind of node (land, transport, places...), a grouping derived by poopdeck.gl from the tags.',
      options: [
        {value: 'year', label: 'Creation date'},
        {value: 'kind', label: 'Kind of node'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Ink (time window)',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp for the creation date and for the cell densities. Darker ramp ends vanish on a dark map.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Node size',
      group: 'Ink (time window)',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1.1,
      unit: 'px',
      disabledWhen: state => !state.showPoints,
      help: 'Disc radius in pixels. A node is a point, so at city scale small discs read as fine line-work.'
    },
    {
      kind: 'toggle',
      id: 'showGlow',
      label: 'Highlight recent nodes',
      group: 'Ink (time window)',
      apply: 'param',
      default: false,
      help: 'Draws a second window, the last few days before the date, as larger bright discs so you can see where mapping is happening right now.'
    },
    {
      kind: 'slider',
      id: 'glowDays',
      label: 'Recent window',
      group: 'Ink (time window)',
      apply: 'param',
      min: 7,
      max: 365,
      step: 7,
      default: 60,
      unit: 'days',
      disabledWhen: state => !state.showGlow,
      help: 'Width of the recent window, [date - window, date]. It is a window-parameter write, no recompile.'
    },
    {
      kind: 'toggle',
      id: 'showCells',
      label: 'Show cell densities',
      group: 'Cells (temporal reduction)',
      apply: 'param',
      default: false,
      help: 'Colors each grid cell by the nodes inside it, summed over the time buckets up to the date. The counts come from GPUTemporalReduction.'
    },
    {
      kind: 'select',
      id: 'cellSize',
      label: 'Cell size',
      group: 'Cells (temporal reduction)',
      apply: 'compile',
      default: '500',
      help: 'Compile-time: the reduction table has cells x buckets rows, so a new size rebuilds the reduction and the cell kernels. 250 m allocates about 70 MB of slot tables.',
      options: [
        {value: '1000', label: '1 km'},
        {value: '500', label: '500 m'},
        {value: '250', label: '250 m'}
      ]
    },
    {
      kind: 'select',
      id: 'bucket',
      label: 'Time bucket',
      group: 'Cells (temporal reduction)',
      apply: 'param',
      default: 'year',
      help: 'Bucket width written into the reduction parameters (origin and width, no recompile). Cells change once per bucket as the date advances.',
      options: [
        {value: 'quarter', label: 'Quarter year'},
        {value: 'half', label: 'Half year'},
        {value: 'year', label: 'Year'}
      ]
    },
    {
      kind: 'select',
      id: 'cellMode',
      label: 'Cell value',
      group: 'Cells (temporal reduction)',
      apply: 'param',
      default: 'cumulative',
      help: 'Nodes created up to and including the current bucket, or only the nodes of the current bucket (where mapping happened during that period).',
      options: [
        {value: 'cumulative', label: 'Cumulative to the date'},
        {value: 'bucket', label: 'Current bucket only'}
      ]
    },
    {
      kind: 'slider',
      id: 'ceiling',
      label: 'Density ceiling',
      group: 'Cells (temporal reduction)',
      apply: 'param',
      min: 20,
      max: 1000,
      step: 10,
      default: 300,
      unit: 'nodes / km2',
      help: 'Sampled nodes per square kilometer that map to the top of the ramp. The color scale is square-root, so low densities stay visible. Multiply by 2.25 for the full history.'
    },
    {
      kind: 'slider',
      id: 'cellOpacity',
      label: 'Cell opacity',
      group: 'Cells (temporal reduction)',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.7,
      disabledWhen: state => !state.showCells,
      help: 'Opacity of the cell layer over the basemap.'
    }
  ],

  story: [
    {
      id: 'finished-city',
      title: 'How did a city get drawn, one node at a time?',
      body: 'Every dot is a tagged OpenStreetMap node in New York City (a bench, a hydrant, a shop, an address), placed where a mapper created it. The map shows a **400,000-node sample** of the **901,827** nodes ever created, colored by creation date: purple is 2007, yellow is 2026. Pick a **Color ramp** below, then compare the **Node size**.',
      camera: {
        longitude: -73.97,
        latitude: 40.71,
        zoom: 10.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1200
      },
      options: {play: false, time: MAX_DAY},
      controls: ['ramp', 'pointSize'],
      readouts: ['rows', 'shown']
    },
    {
      id: 'ink',
      title: 'Press play: the map fills with ink',
      body: 'Each node appears on the day it was created and stays. `GPUTimeWindowFilter` selects the nodes created before the playhead on the GPU every frame and writes the draw count, with no recompile. Press **Play** and watch Manhattan arrive in 2007 and 2008, then the city fill in later. Drag **Date** to scrub.',
      camera: {zoom: 10.8, transitionMs: 1200},
      options: {play: true, time: 0, speed: 1},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'shown']
    },
    {
      id: 'cells',
      title: 'Count the ink: nodes per cell per year',
      body: '`GPUTemporalReduction` folds every node into one record per **(cell, time bucket)**: the slot is `cell x buckets + bucket`. A small kernel sums the buckets up to the date into one density per cell. Switch to a **Time bucket** of a year and a **Cell size** of 500 m, and read the legend in sampled nodes per km2.',
      options: {showPoints: false, showCells: true, play: false, time: MAX_DAY},
      controls: ['cellSize', 'bucket', 'ceiling'],
      readouts: ['grid', 'records', 'peak', 'cells']
    },
    {
      id: 'imports-by-year',
      title: 'See where mapping happened in one period',
      body: 'Set **Cell value** to the current bucket only, and drag the date to December 2013. The cells that light up are where mappers worked in that year, not where the city has most nodes overall. The chart shows nodes created per bucket across the city: the tall bars are bulk imports, which the next two scenes dissect.',
      options: {
        showPoints: false,
        showCells: true,
        cellMode: 'bucket',
        bucket: 'year',
        play: false,
        time: dayOf(2013, 12)
      },
      camera: {longitude: -73.95, latitude: 40.7, zoom: 10.3, transitionMs: 1500},
      controls: ['cellMode', 'time'],
      readouts: ['perBucket', 'clock']
    },
    {
      id: 'recent-work',
      title: 'Where is mapping happening right now?',
      body: 'Switch the nodes back on, turn on **Highlight recent nodes** and widen or narrow the **Recent window**. The bright discs are nodes created in the last few weeks before the date: a second `GPUTimeWindowFilter`, a different window, the same buffers. Try scrubbing through 2021 to see the bursts.',
      options: {
        showPoints: true,
        showCells: false,
        showGlow: true,
        glowDays: 60,
        play: false,
        time: dayOf(2021, 6)
      },
      camera: {longitude: -73.95, latitude: 40.72, zoom: 10.6, transitionMs: 1500},
      controls: ['showGlow', 'glowDays', 'time'],
      readouts: ['clock', 'glowing']
    }
  ],

  legends: state =>
    (
      [
        state.showPoints
          ? state.colorBy === 'year'
            ? {
                kind: 'ramp',
                title: 'Node creation date',
                ramp: state.ramp,
                extent: [2007.5, 2026.4],
                format: (value: number) => value.toFixed(0)
              }
            : {
                kind: 'categories',
                title: 'Kind of node',
                entries: [
                  {color: [86, 190, 120, 235], label: 'Land and water'},
                  {color: [160, 160, 185, 235], label: 'Other tagged nodes'},
                  {color: [240, 165, 60, 235], label: 'Transport'},
                  {color: [205, 105, 220, 235], label: 'Infrastructure'},
                  {color: [80, 170, 245, 235], label: 'Places and amenities'},
                  {color: [240, 95, 95, 235], label: 'Buildings'}
                ],
                note: 'poopdeck.gl groups OSM tags into these kinds.'
              }
          : undefined,
        state.showCells
          ? {
              kind: 'ramp',
              title: 'Sampled nodes per km2 (sqrt scale)',
              ramp: state.ramp,
              extent: [0, state.ceiling],
              sqrtScale: true,
              unit: 'nodes / km2'
            }
          : undefined,
        state.showGlow
          ? {
              kind: 'categories',
              title: 'Recent window',
              entries: [
                {color: [255, 250, 220, 255], label: `Created in the last ${state.glowDays} days`}
              ]
            }
          : undefined
      ] as (LegendSpec | undefined)[]
    ).filter((entry): entry is LegendSpec => entry !== undefined),

  readouts: [
    {id: 'clock', label: 'Date', help: 'The playhead: nodes created up to this day are drawn.'},
    {
      id: 'rows',
      label: 'Nodes in the sample',
      help: 'The sample is a seeded uniform draw of the full history, so patterns and shares are unbiased.'
    },
    {
      id: 'shown',
      label: 'Drawn (full-history estimate)',
      help: 'Sampled nodes inside the window, divided by the sample fraction (0.44).'
    },
    {
      id: 'glowing',
      label: 'Recent (full-history estimate)',
      help: 'Nodes in the recent window, scaled from the sample.'
    },
    {
      id: 'grid',
      label: 'Grid',
      help: 'Cells of the reduction grid over the bounding box of the data.'
    },
    {
      id: 'records',
      label: 'Reduction records',
      help: 'Occupied (cell, time bucket) slots reported by GPUTemporalReduction.'
    },
    {
      id: 'cells',
      label: 'Cells with nodes',
      help: 'Cells whose value is above zero at the current date and mode.'
    },
    {
      id: 'peak',
      label: 'Densest cell',
      help: 'Highest cell value at the current date, scaled to a full-history estimate.'
    },
    {id: 'perBucket', label: 'Nodes per bucket', kind: 'chart'}
  ],

  snippet: state => `const reduction = new GPUTemporalReduction({
  cellIds,            // uint32 per node, from a kernel over positions (${state.cellSize} m cells)
  timestamps: days,   // float32 days since 2007-06-27 (exact below 2^24)
  values: ones,
  parameters: bucketParameters.importToGraph(graph),
  cellCount, bucketCount: 80,
  output: {counts, min, max, first, last, occupiedSlots}
});
// per frame, no recompile
bucketParameters.write(getGPUTemporalReductionParameterValues(0, ${state.bucket === 'year' ? '365.2425' : state.bucket === 'half' ? '182.62' : '91.31'}));
windowParameters.write(getGPUTimeWindowParameterValues({start: -1, end: playhead}));`,

  about: {
    what: 'A playback of every sampled node creation in OpenStreetMap New York City from 2007 to 2026. GPUTimeWindowFilter selects the nodes created before the playhead; GPUTemporalReduction counts nodes per (cell, time bucket) in one dense table.',
    why: 'Edit history tells you how a map was made: when areas were first drawn, where activity moved, and how much of the city was added in short bursts. A GPU reduction keeps that count live while you scrub.',
    howToRead:
      'Purple dots are early, yellow dots are recent. In the cell view, brighter means more nodes per km2 (square-root scale). Counts shown as "full-history estimate" divide the sample by its sampling fraction (400,000 of 901,827). The `kind` grouping is poopdeck.gl\'s. Data: © OpenStreetMap contributors, ODbL, via the poopdeck.gl archive osm-nyc-nodes. Only node creations are counted, not later edits or ways. No user names or ids are used.'
  },

  create: async ctx =>
    (await import('./osm-history-draws-itself.compute')).createOsmDrawsItself(ctx)
});
