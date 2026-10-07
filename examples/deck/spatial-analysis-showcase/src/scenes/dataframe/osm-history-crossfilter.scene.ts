// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {OsmCrossfilterOptions} from './osm-history-crossfilter.compute';

const FIRST_YEAR = 2007.5;
const LAST_YEAR = 2026.4;

export default defineScene<OsmCrossfilterOptions>({
  id: 'osm-history-crossfilter',
  title: 'Which nodes, when? Brush OpenStreetMap New York by year and kind',
  chapter: 'dataframe',
  order: 2,
  summary:
    'Linked brushing over 400,000 OpenStreetMap node creations in New York City: brush the years, the kind, the top contributors or the current map view, and the map, the time histogram and the kind bars all update from one GPU graph.',
  contributors: ['GPUCrossfilter'],
  datasets: [
    {
      id: 'poopdeck-osm-nyc',
      role: 'node creations 2007-2026 (400k sample), © OpenStreetMap contributors'
    }
  ],
  initialView: {longitude: -73.97, latitude: 40.71, zoom: 10.4},

  options: [
    {
      kind: 'range',
      id: 'years',
      label: 'Creation years',
      group: 'Brushes',
      apply: 'param',
      min: FIRST_YEAR,
      max: LAST_YEAR,
      step: 0.25,
      default: [FIRST_YEAR, LAST_YEAR],
      format: value => value.toFixed(2),
      help: 'Keeps only nodes created inside this interval (decimal years). It is an inclusive range selection on the creation-day column; a write into a small control buffer, then one re-encode of the graph.'
    },
    {
      kind: 'select',
      id: 'kind',
      label: 'Kind of node',
      group: 'Brushes',
      apply: 'param',
      default: 'all',
      help: 'Keeps one kind (an exact categorical selection, a range with equal ends). Kinds are poopdeck.gl groupings of OSM tags, not OSM terms.',
      options: [
        {value: 'all', label: 'All kinds'},
        {value: '0', label: 'Land and water'},
        {value: '1', label: 'Other tagged nodes'},
        {value: '2', label: 'Transport'},
        {value: '3', label: 'Infrastructure'},
        {value: '4', label: 'Places and amenities'},
        {value: '5', label: 'Buildings'}
      ]
    },
    {
      kind: 'select',
      id: 'topContributors',
      label: 'Contributors',
      group: 'Brushes',
      apply: 'param',
      default: 'all',
      help: 'Keeps nodes made by the N most active contributors over the full history. Contributors are anonymous ranks (0 is the most active); no names or ids are in the data.',
      options: [
        {value: 'all', label: 'Everyone'},
        {value: '1', label: 'The most active contributor'},
        {value: '10', label: 'Top 10'},
        {value: '100', label: 'Top 100'},
        {value: '1000', label: 'Top 1,000'}
      ]
    },
    {
      kind: 'toggle',
      id: 'viewBrush',
      label: 'Brush to the map view',
      group: 'Brushes',
      apply: 'param',
      default: false,
      help: 'Keeps only the nodes inside the current map rectangle, a bounds selection that follows pan and zoom. The charts then describe just what you can see.'
    },
    {
      kind: 'button',
      id: 'reset',
      label: 'Clear all brushes',
      group: 'Brushes',
      help: 'Removes every selection at once (clearAll).'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color selected nodes by',
      group: 'Display',
      apply: 'param',
      default: 'kind',
      help: 'The visible rows are drawn from the GPU-compacted id list; this chooses their color.',
      options: [
        {value: 'kind', label: 'Kind of node'},
        {value: 'year', label: 'Creation date'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy !== 'year',
      help: 'Ramp for the creation date.',
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
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1.4,
      unit: 'px',
      help: 'Disc radius of the selected nodes in pixels.'
    },
    {
      kind: 'toggle',
      id: 'showContext',
      label: 'Show filtered-out nodes',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every node faintly underneath, so you can see what the brushes removed.'
    },
    {
      kind: 'slider',
      id: 'contextOpacity',
      label: 'Context opacity',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.6,
      step: 0.02,
      default: 0.12,
      disabledWhen: state => !state.showContext,
      help: 'Opacity of the faint all-nodes layer.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'What was added to the map, and when?',
      body: 'The map shows a **400,000-node sample** of OpenStreetMap New York, colored by kind. The chart is the number of nodes created per quarter since 2007. `GPUCrossfilter` links this map and these charts: brush one and the others answer. Start by choosing how to color the nodes with **Color selected nodes by** below.',
      camera: {
        longitude: -73.97,
        latitude: 40.71,
        zoom: 10.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1200
      },
      controls: ['colorBy'],
      readouts: ['selected', 'timeChart']
    },
    {
      id: 'brush-years',
      title: 'Brush two years: the 2013 to 2014 surge',
      body: 'Drag **Creation years** to 2013.5 to 2014.5. The map keeps only nodes created then, the histogram keeps its full shape (a view never filters itself, so you still see the context) and shades the part inside your brush, and the kind bars recount. Almost everything in that window sits in one kind.',
      options: {years: [2013.5, 2014.5]},
      controls: ['years'],
      readouts: ['selected', 'share', 'topKind', 'timeChart', 'kindChart']
    },
    {
      id: 'kind',
      title: 'Add a second brush: one kind',
      body: 'Pick **Kind of node** to cut the same window by kind. The kind bars never filter themselves, so they keep showing what the other brushes leave, with your pick highlighted. Compare the **Other tagged nodes** with **Transport**: one is a bulk import, the other steady human mapping.',
      options: {years: [2013.5, 2014.5], kind: '1'},
      controls: ['kind', 'years'],
      readouts: ['selected', 'topKind', 'kindChart']
    },
    {
      id: 'view',
      title: 'Brush the map by looking at it',
      body: 'Turn on **Brush to the map view** and pan or zoom. The rectangle you see is a bounds selection, re-written when the view moves and re-run on the GPU; the charts then describe only the neighbourhood on screen. Try Brooklyn, then Manhattan, with the 2013 to 2014 brush still on.',
      camera: {longitude: -73.93, latitude: 40.68, zoom: 11.6, transitionMs: 1600},
      options: {years: [2013.5, 2014.5], kind: '1', viewBrush: true},
      controls: ['viewBrush', 'years'],
      readouts: ['selected', 'sampleRows', 'timeChart']
    },
    {
      id: 'prolific',
      title: 'Who made the nodes? Brush contributors',
      body: 'Clear the other brushes and choose **Contributors**: the top 10 accounts, out of 11,361, made over a third of all nodes ever created. Compare **Top 100** with **Everyone** in the histogram: whole bursts of the history belong to a few accounts. The next scenes explain how concentrated that is.',
      camera: {longitude: -73.97, latitude: 40.71, zoom: 10.4, transitionMs: 1400},
      options: {topContributors: '10'},
      controls: ['topContributors', 'kind'],
      readouts: ['selected', 'share', 'timeChart']
    }
  ],

  legends: state =>
    (
      [
        state.colorBy === 'kind'
          ? {
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
              note: 'Selected nodes only; filtered-out nodes are the faint grey layer.'
            }
          : {
              kind: 'ramp',
              title: 'Node creation date',
              ramp: state.ramp,
              extent: [2007.5, 2026.4],
              format: (value: number) => value.toFixed(0)
            }
      ] as LegendSpec[]
    ).filter(Boolean),

  readouts: [
    {
      id: 'selected',
      label: 'Nodes selected (full-history estimate)',
      help: 'Rows passing every brush, from the GPU count view, divided by the sample fraction (0.44).'
    },
    {id: 'sampleRows', label: 'Rows selected (sample)', help: 'The same count in sampled rows.'},
    {
      id: 'share',
      label: 'Share of all nodes',
      help: 'Selected rows as a share of the 400,000 sampled rows.'
    },
    {
      id: 'topKind',
      label: 'Most common kind',
      help: 'Largest bar of the per-kind group view (it excludes the kind brush).'
    },
    {id: 'timeChart', label: 'Nodes per quarter', kind: 'chart'},
    {id: 'kindChart', label: 'Nodes by kind', kind: 'chart'}
  ],

  snippet: () => `const filter = new GPUCrossfilter(graph, {
  dimensions: [
    {id: 'map', kind: 'bounds', x, y},
    {id: 'time', kind: 'range', input: days},
    {id: 'kind', kind: 'range', input: kind},
    {id: 'contributor', kind: 'range', input: contributorRank}
  ],
  views: [
    {id: 'time-others', kind: 'histogram', dimension: 'time', input: days, domain: [0, 6940], output: bins},
    {id: 'kind-others', kind: 'group', dimension: 'kind', keys: kind, output: kindCounts},
    {id: 'visible', kind: 'visibility', output: visibleIds, count: drawInstanceCount}
  ]
});
filter.addToGraph(graph);
const compiled = graph.compile();
// a brush changes: write the control buffer, encode the same graph again
filter.setRange('time', [yearToDay(2013.5), yearToDay(2014.5)]);
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: 'Linked brushing over 400,000 sampled OpenStreetMap node creations in New York City. GPUCrossfilter keeps four selections (map rectangle, creation time, kind, contributor rank), and publishes self-excluding histograms and group counts plus a compacted list of the visible rows for the map.',
    why: 'Exploring edit history is a chain of small questions: when, what kind, where, who. Each answer should change every other view immediately; with the rows resident on the GPU a brush is a tiny buffer write and one encode.',
    howToRead:
      'Bright points are the rows that pass every brush; the faint grey layer is everything else. In the time chart the red line shows what passes the map, kind and contributor filters, and the blue area the part inside the time brush. Counts marked "estimate" divide the sample by its fraction (400,000 of 901,827). Kinds and contributor ranks come from the poopdeck.gl export; contributors are anonymous ranks, no names or ids are used. Data: © OpenStreetMap contributors, ODbL.'
  },

  create: async ctx => (await import('./osm-history-crossfilter.compute')).createOsmCrossfilter(ctx)
});
