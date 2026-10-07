// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {OsmContributorsOptions} from './osm-history-contributors.compute';

const FIRST_YEAR = 2007.5;
const LAST_YEAR = 2026.4;

export default defineScene<OsmContributorsOptions>({
  id: 'osm-history-contributors',
  title: 'Who drew New York? How concentrated is the work?',
  chapter: 'dataframe',
  order: 3,
  summary:
    'Group 400,000 OpenStreetMap node creations by anonymous contributor rank on the GPU, then read the concentration: a Lorenz curve, the share of the top N contributors, and the first, median and last month of each leading contributor.',
  contributors: ['GPUGroupStatistics'],
  datasets: [
    {
      id: 'poopdeck-osm-nyc',
      role: 'node creations 2007-2026 (400k sample), © OpenStreetMap contributors'
    }
  ],
  initialView: {longitude: -73.97, latitude: 40.71, zoom: 10.4},

  options: [
    {
      kind: 'slider',
      id: 'topN',
      label: 'Top N contributors',
      group: 'Highlight',
      apply: 'param',
      min: 1,
      max: 500,
      step: 1,
      default: 10,
      format: value => `top ${value}`,
      help: 'Contributors are ranked by nodes over the full history (0 is the most active, no names or ids are used). Nodes from the top N are drawn in orange, the rest grey. Only the threshold changes: it is a parameter write.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color nodes by',
      group: 'Highlight',
      apply: 'param',
      default: 'highlight',
      help: 'Top-N highlight against everyone else, or the contributor rank on a ramp (square-root scale, 0 to 1,000).',
      options: [
        {value: 'highlight', label: 'Top N against the rest'},
        {value: 'rank', label: 'Contributor rank'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Rank ramp',
      group: 'Highlight',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.colorBy !== 'rank',
      help: 'Ramp used for the contributor rank. Bright is a prolific contributor.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'range',
      id: 'years',
      label: 'Creation years',
      group: 'Window',
      apply: 'param',
      min: FIRST_YEAR,
      max: LAST_YEAR,
      step: 0.25,
      default: [FIRST_YEAR, LAST_YEAR],
      format: value => value.toFixed(2),
      help: 'Group only the nodes created in this interval. A mask kernel marks them; GPUGroupStatistics skips the rest. No recompile.'
    },
    {
      kind: 'select',
      id: 'kind',
      label: 'Kind of node',
      group: 'Window',
      apply: 'param',
      default: 'all',
      help: 'Group only one kind of node (poopdeck.gl grouping of tags).',
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
      kind: 'slider',
      id: 'inspect',
      label: 'Inspect contributor',
      group: 'Statistics',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      format: value => `#${value}`,
      help: 'Rank of the contributor whose count and first, median and last creation month are read out. The statistics come from the GPU group table.'
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
      default: 1.2,
      unit: 'px',
      help: 'Disc radius in pixels.'
    },
    {
      kind: 'slider',
      id: 'contextOpacity',
      label: 'Opacity of the rest',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.8,
      step: 0.02,
      default: 0.2,
      disabledWhen: state => state.colorBy !== 'highlight',
      help: 'Opacity of the nodes outside the top N.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Did a crowd draw New York, or a handful?',
      body: 'The orange nodes are everything created by the **top 10 contributors** out of 11,361 (anonymous ranks over the full history). The map is a **400,000-node sample** of all 901,827 nodes. Raise **Top N contributors** below and watch the orange spread.',
      camera: {
        longitude: -73.97,
        latitude: 40.71,
        zoom: 10.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1200
      },
      controls: ['topN'],
      readouts: ['window', 'topShare', 'fullShare']
    },
    {
      id: 'lorenz',
      title: 'Read the concentration curve',
      body: '`GPUGroupStatistics` groups the nodes by contributor on the GPU, one dense row per contributor. The Lorenz curve sorts contributors from least to most active: if everyone made equal shares it would follow the dashed diagonal. It hugs the floor and shoots up at the right: the vertical rule marks where your **Top N contributors** start, and the Gini coefficient measures the bend (0 equal, 1 one person).',
      controls: ['topN'],
      readouts: ['lorenz', 'gini', 'active']
    },
    {
      id: 'leaders',
      title: 'Meet the leaders by rank',
      body: 'Use **Inspect contributor** to step down the ranking. Each readout comes from the same group table: how many nodes that contributor made, and the minimum, median and maximum creation day of those nodes. Some leaders worked for years, others in one short burst.',
      options: {inspect: 1},
      controls: ['inspect'],
      readouts: ['inspected', 'topBars']
    },
    {
      id: 'surge',
      title: 'Zoom into the 2013 to 2014 surge',
      body: 'Set **Creation years** to 2013.5 to 2014.5 and **Kind of node** to Other tagged nodes. In this window five accounts make over half of the sampled nodes (about 54%, against 27% for the whole history): the top share jumps. Those are the bulk imports of the next scene, seen as people.',
      options: {years: [2013.5, 2014.5], kind: '1', topN: 5},
      controls: ['years', 'kind'],
      readouts: ['topShare', 'gini', 'lorenz', 'inspected']
    },
    {
      id: 'honest',
      title: 'How far to trust a sample',
      body: 'This is a uniform sample, so each contributor appears in proportion to their work, but the **long tail of occasional contributors is thinned**: with 44% of the nodes, someone who made 3 nodes is missing about one time in six. Set **Creation years** back to the whole range and compare the sampled top share with the exact full-history share in the readouts. Try **Rank ramp**: color by rank.',
      options: {topN: 100, colorBy: 'rank'},
      controls: ['colorBy', 'topN'],
      readouts: ['topShare', 'fullShare', 'gini']
    }
  ],

  legends: state =>
    (
      [
        state.colorBy === 'highlight'
          ? {
              kind: 'categories',
              title: 'Nodes in the window',
              entries: [
                {color: [255, 70, 130, 255], label: 'Most active contributor (#1)'},
                {color: [255, 170, 50, 255], label: `Top ${state.topN} contributors`},
                {color: [150, 155, 170, 160], label: 'Everyone else'}
              ],
              note: 'Rank is by nodes over the full history; anonymous.'
            }
          : {
              kind: 'ramp',
              title: 'Contributor rank (0 = most active)',
              ramp: state.ramp,
              extent: [0, 1000],
              sqrtScale: true,
              unit: 'rank'
            }
      ] as LegendSpec[]
    ).filter(Boolean),

  readouts: [
    {
      id: 'rows',
      label: 'Nodes in the sample',
      help: 'A seeded uniform sample of the full history.'
    },
    {
      id: 'window',
      label: 'Nodes in the window (estimate)',
      help: 'Grouped rows divided by the sample fraction (0.44).'
    },
    {
      id: 'active',
      label: 'Contributors in the window',
      help: 'Contributors with at least one sampled node in the window. The sample misses some occasional contributors, so this is a lower bound.'
    },
    {
      id: 'topShare',
      label: 'Share of the top N (sample)',
      help: 'Nodes made by the N top-ranked contributors, as a share of the window.'
    },
    {
      id: 'fullShare',
      label: 'Share of the top N (exact)',
      help: 'Computed over all 901,827 nodes at build time; only defined for the whole history.'
    },
    {
      id: 'gini',
      label: 'Gini coefficient',
      help: 'Inequality of nodes per contributor: 0 is equal, 1 is one contributor.'
    },
    {
      id: 'inspected',
      label: 'Inspected contributor',
      layout: 'block',
      help: 'Count, first, median and last creation month from GPUGroupStatistics.'
    },
    {id: 'lorenz', label: 'Lorenz curve', kind: 'chart'},
    {id: 'topBars', label: 'Ten most active contributors', kind: 'chart'}
  ],

  snippet: () => `graph.add(
  new GPUGroupStatistics({
    keys: contributorRank,            // uint32 per node, dense in [0, 11361)
    mask: windowMask,                 // years and kind, written by a kernel
    keyCount: 11361,                  // dense: row k is contributor k
    columns: [{
      values: days,
      statistics: ['minimum', 'maximum', 'mean', 'median'],
      output: {minimums, maximums, means, medians}
    }],
    output: {keys, counts, count, overflow}
  })
);
// after readback: sort counts ascending on the CPU -> Lorenz curve, Gini`,

  about: {
    what: "A GPU group-by of 400,000 sampled OpenStreetMap node creations in New York City by anonymous contributor rank. GPUGroupStatistics returns each contributor's count and minimum, maximum, mean and median creation day; the CPU only sorts 11,361 counts for the Lorenz curve.",
    why: 'A map built by many people and a map built by a few accounts need different care: who corrects errors, whose conventions win, and what disappears if one contributor leaves. Concentration is the first number to know.',
    howToRead:
      'Orange nodes belong to the top N contributors (by full-history rank), grey nodes to everyone else. The Lorenz curve plots the cumulative share of nodes against contributors sorted from least to most active; a curve far below the diagonal means concentration. Shares and Gini from the sample are close to, but not equal to, the exact full-history values shown beside them, because the sample thins the long tail. No user names or ids are shipped or shown; ranks are anonymous. Data: © OpenStreetMap contributors, ODbL.'
  },

  create: async ctx =>
    (await import('./osm-history-contributors.compute')).createOsmContributors(ctx)
});
