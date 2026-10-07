// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import type {OsmImportWavesOptions} from './osm-history-import-waves.compute';

const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];
/** Month index 0 is June 2007. */
const formatMonth = (index: number) => {
  const year = 2007 + Math.floor((index + 5) / 12);
  return `${MONTH_NAMES[(index + 5) % 12]} ${year}`;
};
const monthOf = (year: number, month: number) => (year - 2007) * 12 + (month - 6);
const MONTH_COUNT = 228;

export default defineScene<OsmImportWavesOptions>({
  id: 'osm-history-import-waves',
  title: 'When did OpenStreetMap New York grow in waves?',
  chapter: 'dataframe',
  order: 4,
  summary:
    'A GPU dataframe turns 400,000 sampled OpenStreetMap node creations into a monthly histogram and a per-kind count for a sliding window. Spikes mark bulk imports; the share held by one account tells an import from a crowd.',
  contributors: [
    'GPUDataFrameHistogramQuery',
    'GPUDataFrameGroupedAggregationQuery',
    'GPUTimeWindowFilter'
  ],
  datasets: [
    {
      id: 'poopdeck-osm-nyc',
      role: 'node creations 2007-2026 (400k sample), © OpenStreetMap contributors'
    }
  ],
  initialView: {longitude: -73.97, latitude: 40.71, zoom: 10.4},

  options: [
    ...playbackOptions<OsmImportWavesOptions>({
      playing: false,
      time: {
        min: 0,
        max: MONTH_COUNT - 1,
        step: 1,
        default: monthOf(2014, 5),
        label: 'Window start',
        format: formatMonth,
        help: 'First month of the window. 0 is June 2007; the last is May 2026.'
      },
      speed: {
        min: 0.5,
        max: 4,
        step: 0.5,
        default: 1,
        unit: 'x',
        help: 'At 1x the 19 years sweep by in 70 seconds, about three months per second.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'width',
      label: 'Window width',
      group: 'Window',
      apply: 'param',
      min: 1,
      max: 24,
      step: 1,
      default: 1,
      unit: 'months',
      help: 'Number of calendar months in the window. The window is two query parameters (from, to) of the compiled dataframe query and the window of GPUTimeWindowFilter: no recompile.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color window nodes by',
      group: 'Display',
      apply: 'param',
      default: 'kind',
      help: 'Kind of node, or whether the node was made by one of the most active contributors (anonymous ranks).',
      options: [
        {value: 'kind', label: 'Kind of node'},
        {value: 'prolific', label: 'Prolific contributors'}
      ]
    },
    {
      kind: 'slider',
      id: 'prolificCount',
      label: 'Prolific contributors',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 200,
      step: 1,
      default: 10,
      format: value => `top ${value}`,
      disabledWhen: state => state.colorBy !== 'prolific',
      help: 'How many of the top-ranked contributors (full-history rank, anonymous) count as prolific.'
    },
    {
      kind: 'slider',
      id: 'spikeRatio',
      label: 'Spike sensitivity',
      group: 'Spikes',
      apply: 'param',
      min: 2,
      max: 8,
      step: 0.5,
      default: 4,
      unit: 'x',
      help: 'A month is a spike when it exceeds this multiple of the median of the 12 months either side of it. Consecutive spike months are marked once, at the peak.'
    },
    {
      kind: 'slider',
      id: 'spikeMinimum',
      label: 'Minimum spike size',
      group: 'Spikes',
      apply: 'param',
      min: 1000,
      max: 30000,
      step: 500,
      default: 5000,
      unit: 'nodes',
      format: value => formatCountShort(value),
      help: 'Ignore months with fewer nodes than this (full-history estimate), so quiet months cannot be called spikes.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Node size',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1.5,
      unit: 'px',
      help: 'Disc radius in pixels.'
    },
    {
      kind: 'toggle',
      id: 'showContext',
      label: 'Show all other nodes',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every sampled node faintly under the window.'
    },
    {
      kind: 'slider',
      id: 'contextOpacity',
      label: 'Context opacity',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.5,
      step: 0.02,
      default: 0.1,
      disabledWhen: state => !state.showContext,
      help: 'Opacity of the faint all-nodes layer.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Does a map grow steadily, or in waves?',
      body: 'The chart is the number of nodes created in New York per calendar month since June 2007. A `GPUDataFrameHistogramQuery` with irregular edges at the true month boundaries counts the **400,000-node sample** on the GPU, scaled up to the full history (solid), and the dashed line is the exact count. The tall spikes are waves. Choose how sensitive the detector is with **Spike sensitivity** below.',
      camera: {
        longitude: -73.97,
        latitude: 40.71,
        zoom: 10.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1200
      },
      options: {time: monthOf(2014, 5), width: 1},
      controls: ['spikeRatio', 'spikeMinimum'],
      readouts: ['monthly', 'spikes']
    },
    {
      id: 'first-wave',
      title: 'September 2007: one account, 73,000 nodes',
      body: 'Set the window to **Sep 2007** with the **Window start** slider below. The month holds about 73,000 nodes and one account made 99% of them (the share chart): this is a bulk upload, consistent with the TIGER-era imports of the first years, not people mapping. The map shows where it landed.',
      options: {time: monthOf(2007, 9), width: 1},
      controls: ['time', 'width'],
      readouts: ['clock', 'window', 'topAccount', 'check', 'share']
    },
    {
      id: 'building-import',
      title: 'October 2013 to May 2014: the NYC building import',
      body: 'Widen the window to **8 months** from Oct 2013. Four months top 30,000 nodes each, and the kind bars show almost all of them in one kind (the poopdeck.gl grouping puts them in Other tagged nodes, not Buildings: the tags decide the group, so the data cannot say more). The top account made 39% to 78% of each month: several accounts working in parallel, an import team rather than one upload. The map fills block by block across the boroughs.',
      options: {time: monthOf(2013, 10), width: 8},
      controls: ['width', 'time'],
      readouts: ['clock', 'window', 'topKind', 'kinds', 'monthly']
    },
    {
      id: 'prolific-hands',
      title: 'Who made them? Color by prolific contributor',
      body: 'Switch **Color window nodes by** to prolific contributors and choose how many count as prolific with **Prolific contributors**. In **May and June 2021** (a window of two months) nearly every node is orange: 92% came from a single account. Compare with the 2013 to 2014 window, where the top 10 ranks made about 76% of the nodes.',
      options: {time: monthOf(2021, 5), width: 2, colorBy: 'prolific', prolificCount: 10},
      controls: ['colorBy', 'prolificCount'],
      readouts: ['clock', 'topAccount', 'window', 'share']
    },
    {
      id: 'crowd',
      title: 'A spike is not always an import: March 2024',
      body: 'Set the window to **Mar 2024**, two months wide. Volume is high, but the top account made under a fifth of it, none of the top 10 contributors took part in the sample, and the kinds are land and transport, not the import kind: many hands were mapping. The **share by the top account** is what separates a crowd from a bulk upload. Try **Play** and watch the share chart.',
      options: {time: monthOf(2024, 3), width: 2, colorBy: 'kind'},
      controls: ['play', 'time', 'width'],
      readouts: ['clock', 'topAccount', 'topKind', 'kinds', 'share']
    }
  ],

  legends: state =>
    (
      [
        state.colorBy === 'kind'
          ? {
              kind: 'categories',
              title: 'Kind of node in the window',
              entries: [
                {color: [86, 190, 120, 235], label: 'Land and water'},
                {color: [160, 160, 185, 235], label: 'Other tagged nodes'},
                {color: [240, 165, 60, 235], label: 'Transport'},
                {color: [205, 105, 220, 235], label: 'Infrastructure'},
                {color: [80, 170, 245, 235], label: 'Places and amenities'},
                {color: [240, 95, 95, 235], label: 'Buildings'}
              ]
            }
          : {
              kind: 'categories',
              title: 'Nodes in the window',
              entries: [
                {color: [255, 150, 40, 255], label: `Top ${state.prolificCount} contributors`},
                {color: [100, 150, 210, 235], label: 'Everyone else'}
              ],
              note: 'Anonymous rank over the full history.'
            }
      ] as LegendSpec[]
    ).filter(Boolean),

  readouts: [
    {id: 'clock', label: 'Window', help: 'Calendar months covered by the window.'},
    {
      id: 'rows',
      label: 'Nodes in the sample',
      help: 'A seeded uniform sample of the full history.'
    },
    {
      id: 'window',
      label: 'Nodes in the window (estimate)',
      help: 'From GPUTimeWindowFilter, divided by the sample fraction (0.44).'
    },
    {
      id: 'topKind',
      label: 'Largest kind in the window',
      help: 'Largest bar of the grouped dataframe query.'
    },
    {
      id: 'topAccount',
      label: "Top account's share",
      help: "Share of the window start month's nodes made by that month's single most active contributor, exact over the full history. Anonymous."
    },
    {
      id: 'check',
      label: 'Sample against exact (start month)',
      help: 'The GPU histogram of the sample scaled up, against the exact count for the same month.'
    },
    {
      id: 'spikes',
      label: 'Detected spikes',
      layout: 'block',
      help: "Peak month of each run of spike months, with its exact count and the top account's share."
    },
    {id: 'monthly', label: 'Nodes per month', kind: 'chart'},
    {id: 'share', label: "Share by the month's top account", kind: 'chart'},
    {id: 'kinds', label: 'Nodes by kind in the window', kind: 'chart'}
  ],

  snippet:
    () => `const dataframe = new GPUDataFrame({table, ownership: 'borrowed'}); // day, kind, contributor
const monthly = dataframe
  .histogram('day', {edges: calendarMonthStartDays}) // 229 edges = 228 months
  .compile(new GPUCommandGraph(device));
const byKind = dataframe
  .filter(and(column('day').greaterThanOrEqual(parameter('from', 0)),
              column('day').lessThan(parameter('to', 6900))))
  .groupBy('kind', {groupCount: 6})
  .aggregate({nodes: 'count'})
  .compile(new GPUCommandGraph(device));
// per frame: move the window with two scalars, no recompile
byKind.encode(commandEncoder, {from: windowStartDay, to: windowEndDay});`,

  about: {
    what: 'A GPU dataframe over 400,000 sampled OpenStreetMap node creations in New York City. A histogram query with irregular edges counts nodes per calendar month; a filtered, grouped query counts nodes per kind inside a sliding window; GPUTimeWindowFilter draws that window on the map.',
    why: 'Import waves change what a map is: they add detail fast, with the conventions of whoever ran the import, and they set the baseline for later human edits. Seeing when they happened, and whether one account or many made them, is the first step in judging data quality.',
    howToRead:
      'Solid line: the sample scaled up to the full history; dashed line: exact monthly counts. Rules mark detected spikes (a month far above the median of the surrounding 24 months). The lower chart is the share of each month made by its single most active contributor: near 100% is a bulk upload by one account, near 0 is a crowd. Kinds are poopdeck.gl groupings of tags; which real-world import a spike is cannot be read from this data, only that it looks like one. Only node creations are counted. No names or ids are used. Data: © OpenStreetMap contributors, ODbL.'
  },

  create: async ctx =>
    (await import('./osm-history-import-waves.compute')).createOsmImportWaves(ctx)
});

function formatCountShort(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}k` : String(value);
}
