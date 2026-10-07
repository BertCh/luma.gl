// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {MigrationTimingOptions} from './migration-timing.compute';
import {formatYearDay} from './migration-shared';

const TIMING_VIEW = {longitude: -16, latitude: 36, zoom: 3.5};

export default defineScene<MigrationTimingOptions>({
  id: 'migration-timing',
  title: 'When does the population pass each latitude?',
  chapter: 'movement',
  order: 13,
  summary:
    'A space-time cube of 226,000 tag fixes: GPUTemporalReduction counts them per latitude band and week, and the band-by-week matrix is drawn on the map at its true latitudes, with a playback cursor sweeping the year.',
  contributors: ['GPUTemporalReduction'],
  datasets: [
    {id: 'poopdeck-animals', role: 'GPS tracks of 42 birds, years folded onto one calendar'}
  ],
  initialView: TIMING_VIEW,

  options: [
    {
      kind: 'select',
      id: 'species',
      label: 'Species',
      group: 'Space-time cube',
      apply: 'param',
      default: 'all',
      help: 'Which species are folded into the matrix: a mask passed to the display kernel. The reduction itself keeps one cell per species and band, so switching never recompiles.',
      options: [
        {value: 'all', label: 'All three species'},
        {value: 'marsh', label: 'Western marsh harrier'},
        {value: 'montagu', label: "Montagu's harrier"},
        {value: 'spoonbill', label: 'Eurasian spoonbill'}
      ]
    },
    {
      kind: 'slider',
      id: 'bucketDays',
      label: 'Bucket width',
      group: 'Space-time cube',
      apply: 'param',
      min: 3,
      max: 14,
      step: 1,
      default: 7,
      unit: 'days',
      help: 'Width of one time bucket: a parameter-buffer write. 226,000 fixes are reduced to at most 60 cells x 128 buckets. The cube holds at most 128 buckets, so the minimum is 3 days.'
    },
    {
      kind: 'select',
      id: 'cellValue',
      label: 'Cell value',
      group: 'Space-time cube',
      apply: 'param',
      default: 'share',
      help: "Share is the part of the species' fixes of that week that fall in the band (a column sums to 100%): where the population is. Count is the number of fixes: it follows how many tags were running. Speed is the fastest two-hour step in the cell, from the reduction's maximum.",
      options: [
        {value: 'share', label: "Share of the week's fixes"},
        {value: 'count', label: 'Number of fixes'},
        {value: 'speed', label: 'Fastest ground speed (km/h)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'liftLow',
      label: 'Square-root color scale',
      group: 'Space-time cube',
      apply: 'param',
      default: true,
      help: 'Applies a square root after normalizing so thin passages show beside the long residences. The range ends at the 99th percentile of the occupied cells.'
    },
    {
      kind: 'slider',
      id: 'probeLatitude',
      label: 'Probe latitude',
      group: 'Probe',
      apply: 'param',
      min: 6,
      max: 64,
      step: 0.5,
      default: 36,
      unit: 'N',
      help: 'The 3-degree band the Probe readout summarizes: its busiest buckets in the first and second half of the year. Drawn as a line across the map when Show the probe is on.'
    },
    {
      kind: 'toggle',
      id: 'showProbe',
      label: 'Show the probe',
      group: 'Probe',
      apply: 'param',
      default: true,
      help: 'Draws the probe latitude as a line across the map and the matrix, to compare the matrix row with the geography.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show the tracks',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every animal-year as a thin line on the map, so the rows of the matrix can be compared with the real routes.'
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.2,
      disabledWhen: state => !state.showTracks,
      help: 'Lower it to read the matrix; raise it to follow the routes.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp of the matrix.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    ...playbackOptions<MigrationTimingOptions>({
      ids: {play: 'play', time: 'day', speed: 'playSpeed', loop: 'loop'},
      group: 'Playback',
      playing: false,
      time: {
        min: 0,
        max: 365,
        step: 1,
        default: 120,
        label: 'Day of the year',
        format: formatYearDay,
        help: 'Moves the cursor across the matrix; the profile and the cursor readout follow it.'
      },
      speed: {
        min: 1,
        max: 30,
        step: 1,
        default: 6,
        unit: 'days/s',
        label: 'Play speed',
        help: 'Days of the folded year per real second. At 6 days per second a year takes a minute.'
      },
      loop: true
    })
  ],

  readouts: [
    {
      id: 'bandChart',
      label: 'The population moves through five latitude groups',
      kind: 'chart',
      help: "Share of the selected species' fixes in each 12-degree latitude group, week by week (the five lines add up to 100%). The hand-over from the northern line to the southern one in late summer is the autumn migration; the way back in spring is slower. The rule is the cursor."
    },
    {
      id: 'profileChart',
      label: 'Where is everyone at the cursor?',
      kind: 'chart',
      help: "The share of fixes in each 3-degree latitude band in the cursor's time bucket (bars are labelled by the southern edge). Press Play and watch the mass move south and north."
    },
    {id: 'tracks', label: 'Tracks'},
    {
      id: 'reduction',
      label: 'Temporal reduction',
      help: 'Fixes reduced to occupied (species, band, bucket) slots by GPUTemporalReduction.'
    },
    {id: 'cursor', label: 'Cursor'},
    {
      id: 'probe',
      label: 'Probe band',
      help: "The probe band's fix count, and the busiest buckets of its first and second half of the year. For a band the birds merely cross, those are the northbound and southbound passages; for a band they live in, they are just the busiest weeks."
    }
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      id: 'matrix',
      title:
        state.cellValue === 'share'
          ? "Share of the week's fixes in the band"
          : state.cellValue === 'count'
            ? 'Fixes in the band and week'
            : 'Fastest two-hour step in the band and week',
      ramp: state.ramp,
      extent: 'gpu' as const,
      sqrtScale: state.liftLow,
      unit: state.cellValue === 'share' ? '%' : state.cellValue === 'count' ? 'fixes' : 'km/h',
      format: (value: number) => (value >= 10 ? value.toFixed(0) : value.toFixed(1))
    },
    {
      kind: 'categories' as const,
      title: 'Where to find the chart on the map',
      entries: [
        {
          color: [150, 156, 168, 255] as const,
          label:
            'West of Africa: x is the day of the year (1 Jan to 31 Dec), y is the true latitude'
        }
      ]
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// One cell per (species, 3-degree latitude band) = 3 x 20 cells, 128 time buckets
// cellIds: uint32 per fix, timestamps: float32 seconds of the year, values: step speed in km/h
graph.add(new GPUTemporalReduction({
  cellIds, timestamps, values: speeds,
  parameters: bucketParameters.importToGraph(graph),
  cellCount: 60, bucketCount: 128,
  output: {counts, min, max, first, last, occupiedSlots: {ids, count, overflow}}
}));
// Per frame, a buffer write: the bucket origin and width in seconds
bucketParameters.write(getGPUTemporalReductionParameterValues(0, ${state.bucketDays * 86400}));

// Display kernel: sum the chosen species' counts of each band and bucket into a 20 x 128 matrix
//   (${state.cellValue}); a raster layer draws it at its true latitudes, a cursor at day ${state.day}.`,

  about: {
    what: '`GPUTemporalReduction` bins every fix into one slot per cell (here a species and a 3-degree latitude band) and coarse time bucket, and keeps the count, minimum, maximum, first and last value of each slot. A kernel then folds the chosen species into a latitude-by-week matrix.',
    why: 'Timing is conservation data: when a wetland or a flyway must be quiet, and how fast the population moves through a band, decide where protection is needed and for how long. A space-time cube answers it for every latitude at once, and the GPU keeps the bucket width a live slider.',
    howToRead:
      'The colored block west of Africa is a chart drawn on the map: columns are weeks of the folded year (1 January at the left edge, 31 December at the right), rows are the 3-degree latitude bands at their true latitudes, so you can read across to the coast the birds fly along. A diagonal stripe going down to the right is a southbound migration, going up a northbound one; horizontal blocks are stays. Lines mark the 12-degree latitude groups, the months and the playback cursor.'
  },

  create: async ctx => (await import('./migration-timing.compute')).createMigrationTiming(ctx),

  story: [
    {
      id: 'the-question',
      title: 'When is the population at each latitude?',
      body: "Tracks tell you where a bird went; ecologists also need **when the population was there**. The block of color west of Africa is a chart drawn on the map: each column is a week of the year, each row a 3-degree latitude band at its true latitude, and the color is the share of that week's tag fixes that fell in the band.\n\nRead it from the left (1 January). The harriers sit in the bottom rows, the Sahel, through the winter, climb a diagonal in spring, sit in the top rows for the breeding season and fall back down in late summer.",
      camera: {...TIMING_VIEW, transitionMs: 1200},
      controls: [],
      readouts: ['bandChart', 'tracks']
    },
    {
      id: 'temporal-reduction',
      title: 'A space-time cube in one pass',
      body: '**`GPUTemporalReduction`** reads all 226,000 fixes once and drops each into a slot: a species and latitude band (the cell) times a time bucket. For every slot it keeps the number of fixes and the minimum, maximum, first and last speed. 60 cells by up to 122 buckets is only 7,000 slots, so the whole cube is read back in one small copy.\n\nChange **Bucket width** below: the bucket is a parameter, so the cube is recomputed without a recompile. Three days shows the stopovers as separate blocks, two weeks smooths them into a ramp. Look at the **Temporal reduction** readout for the slots actually occupied.',
      options: {showTracks: true, trackOpacity: 0.15},
      highlight: {readout: 'reduction'},
      controls: ['bucketDays', 'liftLow'],
      readouts: ['reduction']
    },
    {
      id: 'the-diagonal',
      title: 'Autumn is a sprint, spring a slog',
      body: "Choose **Marsh harrier** in **Species**. The marsh harriers in this sample leave the 50 N band in the first week of September (the median bird crosses 50 N on about 6 September) and reach 20 N about fifteen days later: the autumn diagonal is steep. In spring the diagonal is shallower: the median bird leaves 20 N in early March and is back at 50 N by about 4 April, and the **average latitude of the harriers climbs for about ten weeks** against six weeks of falling in autumn. Most of the delay is between 20 and 33 N, the Sahara crossing and the Moroccan plains.\n\nThe chart above the matrix shows the same thing as five lines handing the population over to each other. Select **Montagu's harrier** and the whole picture moves about a month later in spring, with a short summer in the north.",
      options: {species: 'marsh', cellValue: 'share'},
      highlight: {readout: 'probe'},
      controls: ['species', 'bucketDays'],
      readouts: ['bandChart', 'probe']
    },
    {
      id: 'speed',
      title: 'How fast do they cross a band?',
      body: 'Change **Cell value** to *Fastest ground speed*. Each cell now shows the **maximum** the reduction kept for the slot: the fastest two-hour step of any bird in that band that week. The breeding and wintering blocks are dull (the fastest step in a typical breeding-band week is about 20 km/h), while the migration diagonal lights up at around 50 km/h.\n\nThe maximum is a statistic of the most active bird of the week, and one fast step in a thin cell is enough to light it up, so read the diagonal, not a single cell. Switch to *Number of fixes* to see that the fixes per week are not flat (roughly 3,500 in winter to 5,500 in summer), which is why the default is a share.',
      options: {cellValue: 'speed', species: 'all'},
      highlight: {readout: 'reduction'},
      controls: ['cellValue', 'species'],
      readouts: ['reduction']
    },
    {
      id: 'play-the-year',
      title: 'Play the year, and know the limits',
      body: 'Press **Play** and the cursor sweeps from January to December; the bars show the share of fixes in each latitude band at the cursor, and the cursor readout gives the mean latitude. Drag **Day of the year** to jump, and **Probe latitude** to read the busiest weeks of any band: set it to 36 N, the Strait of Gibraltar, to see the two passages.\n\nThe limits are real. These are 42 birds, and some contribute several years, which the archive folds onto one calendar: "week 38" blends every year. Shares weigh the fixes, and a tag that fails stops contributing, so the late-year columns rest on fewer birds. A fix is not a bird: a bird sitting in a band for a week counts 84 times. **Try:** set **Species** to *Eurasian spoonbill* and watch a population that never leaves 33 to 54 N.',
      options: {play: true, cellValue: 'share', species: 'all', playSpeed: 6},
      camera: {...TIMING_VIEW, transitionMs: 1200},
      controls: ['play', 'day', 'probeLatitude'],
      readouts: ['cursor', 'profileChart', 'probe']
    }
  ]
});
