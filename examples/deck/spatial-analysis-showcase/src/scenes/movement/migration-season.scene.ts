// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {MigrationSeasonOptions} from './migration-season.compute';
import {formatYearDay} from './migration-shared';

const EUROPE_AFRICA_VIEW = {longitude: 2, latitude: 33, zoom: 3.2};

export default defineScene<MigrationSeasonOptions>({
  id: 'migration-season',
  title: 'Play a year of migration',
  chapter: 'movement',
  order: 14,
  summary:
    "Every marsh harrier, Montagu's harrier and spoonbill of the data set as a moving dot, interpolated on the GPU at a clock that runs through one folded year, with fading trails, a follow camera and the share of birds in Africa day by day.",
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'poopdeck-animals', role: 'GPS tracks of 42 birds, years folded onto one calendar'}
  ],
  initialView: EUROPE_AFRICA_VIEW,

  options: [
    ...playbackOptions<MigrationSeasonOptions>({
      ids: {play: 'play', time: 'day', speed: 'playSpeed', loop: 'loop'},
      group: 'Playback',
      playing: true,
      time: {
        min: 0,
        max: 365,
        step: 1,
        default: 60,
        label: 'Day of the year',
        format: formatYearDay,
        help: 'The playhead, in days since 1 January of the folded year. The clock writes it back while it plays.'
      },
      speed: {
        min: 1,
        max: 20,
        step: 1,
        default: 5,
        unit: 'days/s',
        label: 'Play speed',
        help: 'Days of the folded year per real second. At 5 days per second the year takes about 75 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'maxGapHours',
      label: 'Hold a bird after a gap of',
      group: 'Markers (playhead)',
      apply: 'param',
      min: 3,
      max: 168,
      step: 1,
      default: 48,
      unit: 'h',
      help: 'When the playhead falls inside a gap between two fixes longer than this (a tag that went quiet), the bird is held at its last fix instead of sliding across the gap. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Marker size',
      group: 'Markers (playhead)',
      apply: 'param',
      min: 2,
      max: 12,
      step: 0.5,
      default: 5,
      unit: 'px',
      help: 'Radius of the bird markers in screen pixels.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the segments that fall in the trail window behind the playhead, from the compacted list GPUTimeWindowFilter writes each frame.'
    },
    {
      kind: 'slider',
      id: 'trailDays',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 14,
      unit: 'days',
      disabledWhen: state => !state.showTrails,
      help: 'How far behind the playhead a trail reaches. A parameter write: the window is two numbers.'
    },
    {
      kind: 'slider',
      id: 'tailFade',
      label: 'Tail fade',
      group: 'Trails (time window)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.8,
      disabledWhen: state => !state.showTrails,
      help: 'Fraction of the trail over which the opacity ramps from zero at the tail to full at the bird. The filter writes the fade weights.'
    },
    {
      kind: 'slider',
      id: 'trailOpacity',
      label: 'Trail opacity',
      group: 'Trails (time window)',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.7,
      disabledWhen: state => !state.showTrails,
      help: 'Overall opacity of the trails.'
    },
    {
      kind: 'toggle',
      id: 'showFlyways',
      label: 'Show every track faintly',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws all tracks as a faint backdrop so the dots can be read against the routes.'
    },
    {
      kind: 'toggle',
      id: 'followSelected',
      label: 'Follow the selected bird',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Keeps the camera on the selected bird while the clock runs (click a dot or a track to select; the first is the longest full year). The bird has a thin white track behind it.'
    }
  ],

  readouts: [
    {
      id: 'africaChart',
      label: 'How long is each species in Africa?',
      kind: 'chart',
      help: "Share of each species' tagged birds that are south of 35 N on each day of the folded year, interpolated from the fixes. The rule is the playhead."
    },
    {id: 'clock', label: 'Date'},
    {
      id: 'birds',
      label: 'Birds with data',
      help: "Animal-years whose track covers the playhead (status active), read back from GPUTrajectoryPlayhead, by species: marsh harrier + Montagu's harrier + spoonbill."
    },
    {id: 'africa', label: 'In Africa', help: 'Active birds south of 35 N at the playhead.'},
    {id: 'latitude', label: 'Latitude of the birds'},
    {id: 'trailSegments', label: 'Trail segments in the window'},
    {id: 'tracks', label: 'Tracks'},
    {id: 'selected', label: 'Selected bird'}
  ],

  legends: state => [
    {
      kind: 'categories' as const,
      title: 'Species (markers and trails)',
      entries: [
        {color: [86, 180, 233, 255] as const, label: 'Western marsh harrier'},
        {color: [240, 150, 30, 255] as const, label: "Montagu's harrier"},
        {color: [204, 121, 167, 255] as const, label: 'Eurasian spoonbill'}
      ],
      note: state.showTrails
        ? `Trails reach ${state.trailDays} days behind each bird.`
        : 'Trails are off.'
    }
  ],

  snippet: state => `import {GPUCommandGraph, DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';
import {GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';

// 1. Every track interpolated at the playhead, on the GPU (positions: lng/lat degrees, timestamps: float32 s of the year)
playheadGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets, parameters: playheadParameters.importToGraph(playheadGraph),
  currentPositions, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow},
  drawInstanceCount: playheadGraph.importGPUData('marker-count', markerDraw.getInstanceCountData(0))
}));

// 2. The segments of the last ${state.trailDays} days, with a fade
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStartTimes, endTimestamps: segmentEndTimes, window: windowParameters.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow}, fadeWeights, clipFractions,
  drawInstanceCount: trailGraph.importGPUData('trail-count', trailDraw.getInstanceCountData(0))
}));

// Per frame: two parameter writes and two encodes, nothing else
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead: day * 86400, maxGap: ${state.maxGapHours * 3600}}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: day * 86400 - ${state.trailDays * 86400}, end: day * 86400, startFadeDuration: ${Math.round(state.trailDays * 86400 * state.tailFade)}
}));`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every track, the two fixes either side of the playhead time with a binary search and interpolates between them, writing the current position of every track, its status and a compact list of the tracks that have data now. `GPUTimeWindowFilter` selects the segments whose time span overlaps the trail window and writes a fade weight and a clip fraction for each.',
    why: 'An animation is the quickest way to see a population as a whole: when the departures start, how scattered they are, how long they spend in Africa. Doing the interpolation and the windowing on the GPU means the clock, the trail length and the gap rule are all live, and the animation costs two small compute passes a frame.',
    howToRead:
      'Each dot is one animal-year at the playhead date, colored by species; the line behind it is the last few days of its route, fading toward the tail. Years are folded onto one calendar, so the same bird in different years appears as several dots on top of each other. The chart under the map shows the share of each species south of 35 N through the year; the vertical rule is the playhead.'
  },

  create: async ctx => (await import('./migration-season.compute')).createMigrationSeason(ctx),

  story: [
    {
      id: 'a-year',
      title: 'How long do the birds spend in Africa?',
      body: "Every dot is a GPS-tagged bird, placed where it was on this day of the year: **55 marsh harrier**, **23 Montagu's harrier** and **23 spoonbill** animal-years, folded onto one calendar. The clock starts on 1 March and runs through the year. Watch the harriers (blue and orange) leave the Low Countries in late summer, pour south and wait out the winter in the Sahel; the spoonbills (pink) hardly move.\n\nThe controls are the clock: **Play**, **Day of the year** and **Play speed**.",
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      options: {play: true, day: 60},
      controls: ['play', 'day', 'playSpeed'],
      readouts: ['clock', 'birds']
    },
    {
      id: 'playhead',
      title: 'Every bird, interpolated on the GPU',
      body: 'The tags log a fix about every two hours, but the clock has no reason to land on one. **`GPUTrajectoryPlayhead`** searches each track for the two fixes around the playhead and interpolates, for all 101 tracks at once, then compacts the list of tracks that have data now and sets the marker count with an indirect draw: no CPU loop touches a bird.\n\nA tag that goes quiet leaves a gap. **Hold a bird after a gap of** below decides when a bird sitting inside a long gap is held at its last fix instead of sliding across it; drag it to 3 hours and many birds freeze between fixes. **Birds with data** counts the tracks the playhead covers.',
      options: {play: false, day: 260, maxGapHours: 48},
      camera: {longitude: 0, latitude: 40, zoom: 4.2, transitionMs: 1400},
      highlight: {readout: 'birds'},
      controls: ['maxGapHours', 'markerSize', 'day'],
      readouts: ['clock', 'birds']
    },
    {
      id: 'trails',
      title: 'Trails from a time window',
      body: "The comet tails come from **`GPUTimeWindowFilter`**: every frame it picks the segments of every track whose time span overlaps the window behind the playhead, writes them as a compact list and gives each a fade weight and a clip fraction, so a tail dissolves smoothly instead of popping.\n\nSlide **Trail length** from 1 to 60 days: at 3 days you see this week's movement, at 60 days the whole autumn passage. **Tail fade** sets how much of the trail is faded. The window is two numbers, so nothing recompiles.",
      options: {play: true, day: 235, showTrails: true, trailDays: 14, tailFade: 0.8, playSpeed: 3},
      camera: {longitude: -2, latitude: 36, zoom: 3.6, transitionMs: 1400},
      highlight: {readout: 'trailSegments'},
      controls: ['showTrails', 'trailDays', 'tailFade'],
      readouts: ['trailSegments', 'birds']
    },
    {
      id: 'africa',
      title: "Six months for the marsh harrier, seven for Montagu's",
      body: "The chart is computed from the tracks once: for each day, the share of each species' tagged birds south of 35 N, roughly the line of the Mediterranean. The marsh harriers are mostly south of it from about **15 September to 26 March**, six months; Montagu's harriers from about **9 September to 20 April**, nearly seven. The spoonbills never cross.\n\nThe playhead is the rule on the chart. Scrub **Day of the year** to a date in October and read the **In Africa** readout: nearly every harrier is south, and the median latitude is about 14 N.",
      options: {play: false, day: 300},
      camera: {longitude: -8, latitude: 24, zoom: 3.3, transitionMs: 1400},
      highlight: {readout: 'africa'},
      controls: ['day', 'play'],
      readouts: ['africa', 'latitude', 'africaChart']
    },
    {
      id: 'follow-one',
      title: 'Follow one bird, and know the limits',
      body: "Click a dot (or a track) to select a bird, then turn on **Follow the selected bird**: the camera now rides with it while the clock runs. The white line is its whole track, and the readout names it. The first selection is the longest track that covers most of the year.\n\nThe limits: 42 birds, which is not the population; some contribute several years, which are folded onto one calendar, so a dot in March may be the same bird as another dot in March of a different year; and the interpolation between two-hourly fixes is a straight line, not the bird's true path. **Try:** play at 20 days per second with **Trail length** at 60 and watch the whole population draw the flyway.",
      options: {play: true, day: 190, followSelected: true, playSpeed: 3, trailDays: 14},
      camera: {zoom: 5, transitionMs: 1000},
      controls: ['followSelected', 'play', 'playSpeed'],
      readouts: ['selected', 'clock']
    }
  ]
});
