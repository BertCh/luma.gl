// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import type {QuakeAftershocksOptions} from './quake-aftershocks.compute';
import {
  formatQuakeDay,
  getQuakeDay,
  QUAKE_DAY_COUNT,
  QUAKE_REGION_OPTIONS,
  QUAKE_REGIONS
} from './quake-regions';

/** Days of the doublet's first shock (2023-02-06) and of the Noto Peninsula shock (2024-01-01). */
const DOUBLET_DAY = getQuakeDay(2023, 2, 6);
const NOTO_DAY = getQuakeDay(2024, 1, 1);

export default defineScene<QuakeAftershocksOptions>({
  id: 'quake-aftershocks',
  title: 'Do earthquakes cluster in time as well as space?',
  chapter: 'time',
  order: 5,
  summary:
    'Replay five years of magnitude 4+ earthquakes, then test on the GPU whether events near each other in space are also near in time more often than chance allows (a Knox test with a permutation reference), and bin them into a grid cell by month table to find where activity comes in bursts.',
  contributors: ['GPUNeighborSearch', 'GPUKnoxTest', 'GPUTemporalReduction'],
  datasets: [{id: 'poopdeck-earthquakes', role: 'USGS M4+ catalog, 2020 to 2024'}],
  initialView: QUAKE_REGIONS.turkiye.view,

  options: [
    ...playbackOptions<QuakeAftershocksOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: QUAKE_DAY_COUNT,
        step: 0.25,
        default: 0,
        label: 'Date (UTC)',
        format: formatQuakeDay,
        help: 'The slider covers 1 Jan 2020 to 31 Dec 2024 in quarter-day steps.'
      },
      speed: {
        kind: 'select',
        default: '30',
        options: [
          {value: '0.25', label: '6 hours per second'},
          {value: '1', label: '1 day per second'},
          {value: '7', label: '1 week per second'},
          {value: '30', label: '1 month per second'},
          {value: '91', label: '3 months per second'}
        ],
        help: 'How much catalog time passes per real second. A month per second plays the five years in about a minute; six hours per second lets you watch a sequence unfold.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'region',
      label: 'Region',
      group: 'Events',
      apply: 'compile',
      default: 'turkiye',
      options: QUAKE_REGION_OPTIONS,
      help: 'The analysis window. Each window is a separate planar metric frame (meters around its center) and a separate compiled set of graphs, built the first time you pick it.'
    },
    {
      kind: 'select',
      id: 'minimumMagnitude',
      label: 'Minimum magnitude',
      group: 'Events',
      apply: 'compile',
      default: '4.5',
      options: [
        {value: '4', label: 'M4.0 and above'},
        {value: '4.5', label: 'M4.5 and above'},
        {value: '5', label: 'M5.0 and above'}
      ],
      help: 'Events below this are left out of everything: the map, the pairs, the Knox test and the grid. The row count is fixed when a graph is compiled. Below about M4.5 the global catalog is increasingly incomplete, mostly right after large shocks.'
    },
    {
      kind: 'slider',
      id: 'fadeDays',
      label: 'Event lifetime',
      group: 'Appearance',
      apply: 'param',
      min: 1,
      max: 365,
      step: 1,
      default: 60,
      unit: 'days',
      help: 'How long an event stays on the map. It swells when it happens and fades to nothing over this many days. Short lifetimes show a sequence as it happens; long ones show where activity piles up.'
    },
    {
      kind: 'slider',
      id: 'sizeScale',
      label: 'Marker size',
      group: 'Appearance',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.1,
      default: 1,
      format: value => `${value.toFixed(1)}x`,
      help: 'Scales every disc. Radius grows by a factor of 1.9 per magnitude unit, so area grows roughly 3.6 times per unit while energy grows 32 times.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color events by',
      group: 'Appearance',
      apply: 'param',
      default: 'depth',
      options: [
        {value: 'depth', label: 'Depth', help: 'Bright is shallow, dark is deep (0 to 300 km).'},
        {value: 'age', label: 'Age', help: 'Bright is new, dark is about to fade.'}
      ],
      help: 'Depth separates crustal events from the deep slab events of subduction zones; age makes the order of a sequence visible.'
    },
    {
      kind: 'toggle',
      id: 'showPast',
      label: 'Keep past events faintly',
      group: 'Appearance',
      apply: 'param',
      default: false,
      help: 'Events older than their lifetime stay as faint dots, so the map accumulates the catalog instead of clearing it.'
    },
    {
      kind: 'slider',
      id: 'spatialRadius',
      label: 'Spatial threshold',
      group: 'Knox test',
      apply: 'param',
      min: 10,
      max: 200,
      step: 5,
      default: 50,
      unit: 'km',
      help: 'Two events are close in space when they are within this distance. It is the radius of the neighbor search that lists the pairs, so changing it rewrites a buffer and re-lists the pairs without recompiling. Large radii in dense sequences overflow the pair list (see the readout).'
    },
    {
      kind: 'slider',
      id: 'timeThreshold',
      label: 'Time threshold',
      group: 'Knox test',
      apply: 'param',
      min: 0.25,
      max: 60,
      step: 0.25,
      default: 7,
      unit: 'days',
      help: 'Two events are close in time when they are within this many days. Knox counts the spatial pairs that are also close in time.'
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Permutations',
      group: 'Knox test',
      apply: 'param',
      min: 19,
      max: 999,
      step: 10,
      default: 499,
      help: 'How many times the event times are shuffled over the fixed locations to build the reference distribution. The smallest possible pseudo p-value is 1 / (permutations + 1).'
    },
    {
      kind: 'button',
      id: 'reseed',
      label: 'Draw new permutations'
    },
    {
      kind: 'toggle',
      id: 'showPairs',
      label: 'Show close pairs',
      group: 'Knox test',
      apply: 'param',
      default: false,
      help: 'Draws a line between every pair that is close in space and in time, the pairs the Knox count is made of. A sequence shows as a dense knot.'
    },
    {
      kind: 'select',
      id: 'gridView',
      label: 'Cell grid shows',
      group: 'Month by cell grid',
      apply: 'param',
      default: 'off',
      options: [
        {value: 'off', label: 'Nothing'},
        {
          value: 'month-count',
          label: 'Events in the playhead month',
          help: 'Count per cell of the 30-day bucket holding the playhead.'
        },
        {
          value: 'month-max',
          label: 'Largest magnitude in the month',
          help: 'The reduction keeps the maximum magnitude of each (cell, month) slot.'
        },
        {
          value: 'burst',
          label: 'Burstiness',
          help: 'Busiest month over the average month of each cell: 1 is steady, 60 is everything in one month.'
        }
      ],
      help: 'GPUTemporalReduction reduces every event into a (grid cell, month) slot: count, minimum, maximum, first and last magnitude. The map reads that table.'
    },
    {
      kind: 'slider',
      id: 'gridCells',
      label: 'Cells per side',
      group: 'Month by cell grid',
      apply: 'param',
      min: 8,
      max: 48,
      step: 4,
      default: 24,
      help: 'The window is cut into this many cells in each direction. The reduction has room for 48 by 48 cells and 60 months; the cell count is a parameter, so changing it re-runs the reduction without recompiling.'
    },
    {
      kind: 'slider',
      id: 'minimumCellEvents',
      label: 'Minimum events per cell',
      group: 'Month by cell grid',
      apply: 'param',
      min: 1,
      max: 100,
      step: 1,
      default: 20,
      help: 'Burstiness needs enough events to mean something: cells with fewer events in the five years are left blank.'
    },
    {
      kind: 'slider',
      id: 'gridOpacity',
      label: 'Grid opacity',
      group: 'Month by cell grid',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.7,
      help: 'Opacity of the cell grid under the events.'
    }
  ],

  story: [
    {
      id: 'bursts',
      title: 'Earthquakes arrive in bursts',
      body: 'Every disc is a magnitude 4.5+ earthquake from the USGS catalog, from 2020 to 2024, in the belt from the Aegean to Iran. Radius is **magnitude**, color is **depth**, and each event swells when it happens and fades over **Event lifetime**. Watch the date run: the background is a steady trickle, but now and then one place lights up for weeks.\n\nThe question for this scene is the one a hazard analyst asks of any catalog: are events that happen near each other *also* near in time more often than chance, and where does that clustering live? **Playback speed** below controls how fast the years pass; drag **Date** to jump.',
      camera: {...QUAKE_REGIONS.turkiye.view, pitch: 0, bearing: 0, transitionMs: 1500},
      options: {
        region: 'turkiye',
        playing: true,
        time: 0,
        playbackSpeed: '30',
        fadeDays: 60,
        minimumMagnitude: '4.5'
      },
      controls: ['time', 'playbackSpeed', 'fadeDays'],
      readouts: ['clock', 'windowCount', 'windowLargest']
    },
    {
      id: 'doublet',
      title: 'One day in February 2023',
      body: 'At 01:17 UTC on 6 February 2023 a **magnitude 7.8** earthquake ruptured the East Anatolian Fault in southern Türkiye, a left-lateral strike-slip rupture some 300 km long at shallow depth. Nine hours later, at 10:24 UTC, a **magnitude 7.5** shock struck about 90 km to the north on a different, east-west striking fault. Two events of that size so close in time are a *doublet*, and the second is not simply an aftershock of the first. Playback is slowed to six hours per second: watch the swarm of M4.5+ aftershocks that follows.\n\nThe counts here are catalog events, not damage. **Event lifetime** is short (3 days) so the picture is the sequence alone; lengthen it below to see it accumulate.',
      camera: {longitude: 37.4, latitude: 37.6, zoom: 6.4, transitionMs: 2200},
      options: {
        playing: true,
        time: DOUBLET_DAY - 2,
        playbackSpeed: '0.25',
        fadeDays: 3,
        showPast: false
      },
      controls: ['playing', 'time', 'fadeDays'],
      readouts: ['clock', 'windowCount', 'windowLargest'],
      callout: {coordinate: [37.01, 37.23], text: 'M7.8, 6 Feb 2023, 01:17 UTC'}
    },
    {
      id: 'knox',
      title: 'Is that more clustering than chance?',
      body: '`GPUNeighborSearch` lists every pair of events closer than **Spatial threshold**. `GPUKnoxTest` counts how many of those pairs are also within **Time threshold** of each other: the *Knox count*. To know what chance would give, it reassigns the five years of event times over the fixed locations **Permutations** times (a Monte Carlo test that makes no Poisson assumption) and compares. Close pairs are drawn as lines: look at the knot at the doublet.\n\nThe observed count towers over the permutation histogram: with the defaults it is roughly nine times what independence predicts, and the p-value sits at its floor of 1 / (permutations + 1). The useful number is the **ratio** and how it changes: it rises when you shrink the **Time threshold** to a day, and falls toward 3 or 4 when you widen both thresholds (200 km, 60 days), because the background of unrelated events starts to count.',
      camera: {...QUAKE_REGIONS.turkiye.view, transitionMs: 1800},
      options: {
        playing: false,
        time: DOUBLET_DAY + 3,
        fadeDays: 14,
        showPast: true,
        showPairs: true,
        spatialRadius: 50,
        timeThreshold: 7
      },
      controls: ['spatialRadius', 'timeThreshold', 'permutations', 'showPairs'],
      readouts: [
        'pairs',
        'timeClose',
        'knoxObserved',
        'knoxExpected',
        'knoxRatio',
        'knoxP',
        'knoxHistogram'
      ],
      highlight: {readout: 'knoxRatio'}
    },
    {
      id: 'noto',
      title: 'Another region, another sequence',
      body: 'Switch the window to **Japan, Izu-Bonin and the Kurils**. On 1 January 2024 a magnitude 7.5 earthquake struck the Noto Peninsula on the Sea of Japan coast; its aftershocks dominate the map for weeks. Play is set to six hours per second again.\n\nThe Knox ratio here is higher than in Türkiye (roughly 19 times with the defaults, against 9): this window holds many separate sequences, and each adds close pairs. Try **Minimum magnitude** below: the ratio usually grows as you keep only larger events, because small events are mostly background and the large ones carry the aftershocks.',
      camera: {longitude: 138.5, latitude: 37.2, zoom: 6.2, transitionMs: 2200},
      options: {
        region: 'japan',
        playing: true,
        time: NOTO_DAY - 0.5,
        playbackSpeed: '0.25',
        fadeDays: 4,
        showPast: false,
        showPairs: true
      },
      controls: ['region', 'minimumMagnitude', 'spatialRadius'],
      readouts: ['events', 'knoxRatio', 'knoxP', 'knoxHistogram'],
      callout: {coordinate: [137.27, 37.49], text: 'M7.5, Noto, 1 Jan 2024'}
    },
    {
      id: 'grid',
      title: 'Where do the bursts hide?',
      body: '`GPUTemporalReduction` cuts the window into a grid and the five years into 60 thirty-day months, then reduces every event into its (cell, month) slot: count, smallest, largest, first and last magnitude. The map now shows **burstiness**: the busiest month of a cell divided by its average month. A cell with a steady trickle scores near 1; a cell whose five years of events fell in a single month scores up to 60.\n\nBright cells are sequences, not background. Hover a cell for its busiest month, and switch **Cell grid shows** to the largest magnitude per month to see the same table as a map of the playhead month. The chart below the map is the same reduction summed over cells.',
      camera: {...QUAKE_REGIONS.japan.view, transitionMs: 1600},
      options: {
        region: 'japan',
        playing: false,
        time: NOTO_DAY + 5,
        fadeDays: 30,
        showPast: true,
        showPairs: false,
        gridView: 'burst',
        gridCells: 24,
        minimumCellEvents: 20,
        gridOpacity: 0.75
      },
      controls: ['gridView', 'gridCells', 'minimumCellEvents'],
      readouts: ['busiestMonth', 'monthly']
    },
    {
      id: 'limits',
      title: 'What this does not tell you',
      body: 'Read the numbers as a **screen for clustering**, not a forecast. A Knox test does not know about aftershock physics: the Omori-Utsu law says the rate of aftershocks decays roughly as 1 / (c + t) after a mainshock, so most close pairs fall in the first days. The test also counts every pair, so a single large sequence with hundreds of events dominates the count (n events make n(n-1)/2 pairs).\n\nThe frame is planar: each window is meters around its center, so distances are exact on the center latitude and off by up to about 12% at the top and bottom. The catalog is incomplete for small events right after a large shock. Try the other regions with a short **Time threshold** (a day or less) to isolate the aftershock burst from the background.',
      camera: {...QUAKE_REGIONS.sunda.view, transitionMs: 1800},
      options: {
        region: 'sunda',
        playing: true,
        playbackSpeed: '30',
        time: 0,
        fadeDays: 60,
        showPast: false,
        showPairs: false,
        gridView: 'off'
      },
      controls: ['region', 'timeThreshold', 'spatialRadius'],
      readouts: ['events', 'knoxRatio', 'knoxP', 'pairOverflow']
    }
  ],

  about: {
    what: 'Three contributors on one catalog. `GPUNeighborSearch` lists the pairs of events within a radius; `GPUKnoxTest` counts how many are also close in time and compares with a Monte Carlo permutation of the times (Baker 1996); `GPUTemporalReduction` reduces events into (grid cell, 30-day month) slots. A bespoke layer draws the catalog directly from the buffers.',
    why: 'Earthquakes are the textbook space-time interaction: aftershocks and doublets make events near in space also near in time. A test of that interaction tells an analyst whether a catalog needs declustering before its background rate is used for hazard, and the month by cell table shows where the bursts are.',
    howToRead:
      "Radius is magnitude, color is depth (or age). In the Knox histogram the bars are the counts under random time assignments and the vertical line is the observed count: the further right of the bars, the stronger the interaction. A ratio near 1 means no clustering beyond chance. On the burstiness map bright means a few months hold most of a cell's events."
  },

  legends: (state): readonly LegendSpec[] => {
    const entries: LegendSpec[] = [
      state.colorBy === 'depth'
        ? {
            kind: 'ramp',
            title: 'Depth',
            ramp: 'magma',
            extent: [0, 300],
            unit: 'km',
            labels: ['300 km or more', 'surface']
          }
        : {
            kind: 'ramp',
            title: 'Age of the event',
            ramp: 'magma',
            extent: [0, state.fadeDays],
            unit: 'days',
            labels: [`${state.fadeDays} days`, 'just happened']
          },
      {
        kind: 'size',
        title: 'Radius is magnitude',
        entries: [4, 5, 6, 7, 8].map(magnitude => ({
          radiusPixels: 2.5 * state.sizeScale * 1.9 ** (magnitude - 4),
          label: `M${magnitude}`
        })),
        color: [230, 140, 90, 255]
      }
    ];
    if (state.gridView === 'month-count') {
      entries.push({
        kind: 'ramp',
        title: 'Events in the playhead month',
        ramp: 'viridis',
        extent: [0, 20],
        sqrtScale: true,
        unit: 'events per cell'
      });
    } else if (state.gridView === 'month-max') {
      entries.push({
        kind: 'ramp',
        title: 'Largest magnitude of the month',
        ramp: 'viridis',
        extent: [4, 8],
        unit: 'M'
      });
    } else if (state.gridView === 'burst') {
      entries.push({
        kind: 'ramp',
        title: 'Burstiness: busiest month / average month',
        ramp: 'viridis',
        extent: [1, 40],
        sqrtScale: true,
        unit: 'times the average'
      });
    }
    return entries;
  },

  readouts: [
    {
      id: 'clock',
      label: 'Playhead',
      help: 'Date and time (UTC) of the playhead.'
    },
    {
      id: 'windowCount',
      label: 'Events on the map',
      format: 'integer',
      help: 'Events whose age is between 0 and the event lifetime at the playhead.'
    },
    {
      id: 'windowLargest',
      label: 'Largest on the map',
      help: 'Largest magnitude among those events.'
    },
    {id: 'events', label: 'Events in the window'},
    {
      id: 'pairs',
      label: 'Pairs close in space',
      format: 'integer',
      help: 'Unordered pairs of events within the spatial threshold, from GPUNeighborSearch.'
    },
    {
      id: 'timeClose',
      label: 'Pairs close in time (all)',
      format: 'integer',
      help: 'Unordered pairs of events within the time threshold anywhere in the window, counted exactly from the sorted times.'
    },
    {
      id: 'knoxObserved',
      label: 'Knox count (observed)',
      format: 'integer',
      help: 'Pairs that are close in space and close in time.'
    },
    {
      id: 'knoxExpected',
      label: 'Expected under independence',
      help: 'Spatial pairs times time-close pairs divided by all pairs.'
    },
    {
      id: 'knoxRatio',
      label: 'Observed / expected',
      help: 'How many times more space-time pairs than independence predicts.'
    },
    {
      id: 'knoxPermuted',
      label: 'Permutations (mean ± sd)',
      help: 'Knox count when the event times are shuffled over the locations.'
    },
    {
      id: 'knoxZ',
      label: 'z-score',
      help: '(observed - permutation mean) / permutation standard deviation.'
    },
    {
      id: 'knoxP',
      label: 'Pseudo p-value',
      help: '(1 + permutations with a count at least as large) / (permutations + 1).'
    },
    {
      id: 'knoxPoisson',
      label: 'Poisson p-value (classic)',
      help: 'The classic Knox p-value with a Poisson reference; shown for comparison.'
    },
    {
      id: 'pairOverflow',
      label: 'Pair list overflow',
      help: 'The pair list has room for 256 pairs per event. When a dense sequence overflows it, the Knox count is partial: lower the spatial threshold.'
    },
    {
      id: 'knoxHistogram',
      label: 'Knox count under permutation',
      kind: 'chart'
    },
    {
      id: 'busiestMonth',
      label: 'Busiest month',
      help: 'The 30-day bucket with the most events in the window, from the (cell, month) reduction.'
    },
    {
      id: 'monthly',
      label: 'Events per month',
      kind: 'chart'
    }
  ],

  snippet: state => `import {
  GPUNeighborSearch, GPUKnoxTest, getGPUNeighborSearchParameterValues, getGPUSpaceTimeParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// Pairs within ${state.spatialRadius} km (planar meters around the window center), then the Knox test.
graph.add(new GPUNeighborSearch({
  mode: 'radius', positions, parameters: searchParameters, gridSize: [64, 64], weights: pairs, overflow
}));
graph.add(new GPUKnoxTest({
  pairs, times: days, parameters: testParameters,
  maximumPermutations: 999, statistics, summary
}));

// Events by (grid cell, 30.44-day month): count, min, max, first and last magnitude.
monthlyGraph.add(new GPUTemporalReduction({
  cellIds, timestamps: days, values: magnitude, parameters: reductionParameters,
  cellCount: 48 * 48, bucketCount: 60, output: {counts, min, max, first, last, occupiedSlots}
}));

// Per change (parameter writes, no recompile):
searchParameters.write(getGPUNeighborSearchParameterValues({bounds, radius: ${state.spatialRadius * 1000}, weightKind: 'binary'}));
testParameters.write(getGPUSpaceTimeParameterValues({seed, permutations: ${state.permutations}, timeThreshold: ${state.timeThreshold}}));
reductionParameters.write(getGPUTemporalReductionParameterValues(0, 365.25 / 12));`,

  create: async ctx => (await import('./quake-aftershocks.compute')).createQuakeAftershocks(ctx)
});
