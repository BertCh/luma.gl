// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {
  formatDayOfYear,
  HURRICANE_CATEGORY_COLORS,
  HURRICANE_CATEGORY_LABELS
} from './hurricane-data';
import type {HurricaneSeasonOptions} from './hurricane-season.compute';

const BASIN_VIEW = {longitude: -60, latitude: 26, zoom: 3.2};

const dayLabel = (day: number) => `${formatDayOfYear(day)} (day ${Math.floor(day) + 1})`;

export default defineScene<HurricaneSeasonOptions>({
  id: 'hurricane-season',
  title: 'What does a hurricane season look like when you stack 46 of them?',
  chapter: 'earth',
  order: 12,
  summary:
    'Every Atlantic storm since 1980 replayed on one calendar: all 46 seasons overlaid by day of the year, with a fading trail, the storm heads colored by their wind and a live count of storms alive.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [{id: 'ibtracs-north-atlantic', role: 'storm tracks (IBTrACS, 1980-2025)'}],
  initialView: BASIN_VIEW,

  options: [
    ...playbackOptions<HurricaneSeasonOptions>({
      ids: {play: 'playing', time: 'time', speed: 'speed', loop: 'loop'},
      time: {
        min: 0,
        max: 365,
        step: 0.25,
        default: 240,
        label: 'Day of the year',
        format: dayLabel,
        help: 'The calendar day in every season at once. Day 1 is 1 January 00:00 UTC; 8 September is day 251.'
      },
      speed: {
        min: 1,
        max: 30,
        step: 1,
        default: 6,
        unit: 'd/s',
        help: 'Calendar days per real second. 6 days per second plays June to November in about half a minute.'
      },
      loop: true
    }),
    {
      kind: 'range',
      id: 'dayRange',
      label: 'Playback window',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 365,
      step: 5,
      default: [140, 345],
      format: dayLabel,
      help: 'The part of the year the clock loops over. The default is 20 May to 11 December: 97% of all fixes fall between June and November.'
    },
    {
      kind: 'range',
      id: 'seasons',
      label: 'Seasons shown',
      group: 'Playback',
      apply: 'param',
      min: 1980,
      max: 2025,
      step: 1,
      default: [1980, 2025],
      help: 'Only storms of these seasons are drawn and counted. The storms are filtered on the GPU; the playhead still runs over all of them.'
    },
    {
      kind: 'slider',
      id: 'maxGapHours',
      label: 'Maximum fix gap',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 48,
      step: 6,
      default: 0,
      format: value => (value === 0 ? 'off' : `${value} h`),
      help: 'A storm whose neighbouring fixes are further apart than this is flagged as in a gap and hidden instead of drawn at a guessed position. The fixes are six-hourly, so 0 (off) never hides one; 6 hides none except gaps in the record.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'The part of every track inside the window behind the playhead, chosen on the GPU by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailDays',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 6,
      unit: 'd',
      help: 'How far back a trail reaches. Longer trails show whole tracks, shorter ones read as comets.'
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
      help: 'Share of the trail that fades out toward its tail; 0 gives a solid trail.'
    },
    {
      kind: 'select',
      id: 'headColor',
      label: 'Color storm heads by',
      group: 'Display',
      apply: 'param',
      default: 'wind',
      help: 'The wind at the head is interpolated between the two six-hourly fixes around the playhead.',
      options: [
        {value: 'wind', label: 'Wind now (Saffir-Simpson class)'},
        {value: 'peak', label: 'Peak wind of the storm'},
        {value: 'season', label: 'Season (year)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Season ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.headColor !== 'season',
      help: 'Ramp of the season color, from 1980 to 2025.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'slider',
      id: 'headSize',
      label: 'Head size',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 10,
      step: 0.5,
      default: 5,
      unit: 'px',
      help: 'Radius of the storm heads.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show all tracks faintly',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every track of the chosen seasons as a faint line, so the heads and trails show against the whole record.'
    },
    {
      kind: 'slider',
      id: 'backdropOpacity',
      label: 'Faint track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.5,
      step: 0.02,
      default: 0.12,
      disabledWhen: state => !state.showBackdrop,
      help: 'Opacity of the faint tracks.'
    }
  ],

  legends: state => [
    state.headColor === 'season'
      ? {
          kind: 'ramp' as const,
          title: 'Season of the storm',
          ramp: state.ramp,
          extent: [1980, 2025] as const,
          format: (value: number) => value.toFixed(0)
        }
      : {
          kind: 'categories' as const,
          title: state.headColor === 'wind' ? 'Wind now' : 'Peak wind of the storm',
          entries: HURRICANE_CATEGORY_LABELS.map((label, index) => ({
            color: HURRICANE_CATEGORY_COLORS[index],
            label
          })),
          note: 'Saffir-Simpson classes come from the sustained wind. Trails are colored by the wind at each fix.'
        }
  ],

  readouts: [
    {
      id: 'aliveSpark',
      label: 'Storms alive through the year',
      kind: 'chart',
      help: 'Average number of storms alive on each day of the year in the chosen seasons. The dot is the playhead.'
    },
    {
      id: 'classChart',
      label: 'Storms alive now by class',
      kind: 'chart',
      help: 'Storms alive on the playhead day, by the class of the wind at their head, summed over all chosen seasons.'
    },
    {id: 'clock', label: 'Playhead', help: 'Calendar day and time in every season at once.'},
    {
      id: 'alive',
      label: 'Storms alive now',
      format: 'integer',
      help: 'Storms of the chosen seasons that have a position at the playhead day, summed over all seasons.'
    },
    {
      id: 'strongest',
      label: 'Strongest now',
      help: 'The strongest of the storms alive on the playhead day.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Track segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {id: 'storms', label: 'Storms'},
    {id: 'seasonsShown', label: 'Seasons shown'}
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// timestamps: float32 DAYS since 1 January 00:00 UTC of each storm's own year, so every season
// lies on one calendar. positions: float32x2 longitude/latitude, trackOffsets: uint32.
playheadGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets,
  parameters: playheadParameters.importToGraph(playheadGraph),
  currentPositions, segmentRows, segmentFractions, status,   // rows + fractions give the wind at the head
  activeTracks: {ids: activeIds, count: activeCount, overflow},
  drawInstanceCount
}));
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStartDays, endTimestamps: segmentEndDays,   // interval mode
  window: windowParameters.importToGraph(trailGraph),
  additionalPredicates: [{kind: 'selection', mask: segmentMask}],   // the chosen seasons
  output: {ids: trailIds, count: trailCount, overflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));

// every frame: plain parameter writes, no recompile
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapHours / 24}}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailDays}, end: playhead, startFadeDuration: ${(state.trailDays * state.tailFade).toFixed(1)}
}));`,

  about: {
    what: 'Each storm is re-based to the 1 January of its own year, so the 46 seasons of 1980 to 2025 share one calendar. `GPUTrajectoryPlayhead` finds, for every storm at once, the two fixes either side of the playhead day and interpolates its position. `GPUTimeWindowFilter` keeps the track segments inside a window behind the playhead and fades them toward the tail.',
    why: 'A single season is anecdote; the stack shows the seasonal clock. Forecasters and planners ask when the first storm usually appears, when the peak is, and where storms are on a given date, and the overlay answers them from 46 years instead of one.',
    howToRead:
      'Every dot is a different storm from a different year, alive on the same calendar day. A dot is colored by its wind now; its trail shows where it has been. The sparkline is the average number of storms alive on each day of the year, with the playhead as a dot. Counts are summed over all chosen seasons, so "12 alive" on 10 September is 12 storms from 46 different years.'
  },

  create: async ctx => (await import('./hurricane-season.compute')).createHurricaneSeason(ctx),

  story: [
    {
      id: 'the-question',
      title: 'When is the Atlantic hurricane season?',
      body: 'Press **Play** and watch June, July and August of **46 seasons at once**. Each dot is a storm from the **NOAA IBTrACS** best tracks (1980-2025), placed at the same calendar day in its own year and colored by the wind there (see the legend). The trails are the last six days of each track.\n\nThe quiet start is real: of all fixes in the record, **97% fall between June and November**. The clock is a number on the calendar slider that the GPU compares against every storm at once, and the **Day of the year** slider follows it. Drag the slider to jump anywhere and keep playing.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {playing: true, speed: 6, time: 150, trailDays: 6, showBackdrop: true},
      highlight: {readout: 'alive'},
      controls: ['playing', 'time', 'speed'],
      readouts: ['clock', 'alive', 'aliveSpark']
    },
    {
      id: 'calendar',
      title: 'One calendar, 46 seasons',
      body: 'The trick is the time column: every fix is re-based to **days since 1 January of its own year** (a float32 day count is exact to a few seconds over a year). **`GPUTrajectoryPlayhead`** then binary-searches each track for the two fixes around the playhead day and interpolates the head, one thread per storm.\n\nThe clock is paused at **8 September (day 251)**, the climatological peak. On that day, summed over 46 years, the **Storms alive now** readout counts the storms from every season that were alive; the bars below split them by the class of their wind. The sparkline shows the average number alive on every day of the year, and its dot is the playhead.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {playing: false, time: 250, trailDays: 6},
      highlight: {readout: 'strongest'},
      controls: ['time', 'headColor'],
      readouts: ['clock', 'alive', 'strongest', 'classChart', 'aliveSpark']
    },
    {
      id: 'trails',
      title: 'Where have they been? A window in time',
      body: '**`GPUTimeWindowFilter`** treats each segment between two fixes as a time interval and keeps those that overlap `[playhead - length, playhead]`. It writes a fade weight and a clip fraction (so the oldest segment is cut part-way), compacts the live segment ids and writes the count straight into the draw call. The seasons you choose are a selection mask in the same pass.\n\nSet **Trail length** to 20 days and the whole life of a long-track storm is drawn; shorten it to 2 days and they read as comets. **Tail fade** at 0 gives solid trails. Turn off **Show all tracks faintly** to see only what is alive within the window.',
      options: {playing: false, time: 250, trailDays: 20, tailFade: 0.8},
      controls: ['trailDays', 'tailFade', 'showBackdrop'],
      readouts: ['trailSegments']
    },
    {
      id: 'peak',
      title: 'The peak, in the numbers',
      body: 'Scrub **Day of the year** along the sparkline: the average number of storms alive is zero in April, only about 0.4 on 1 August, then **peaks around 8 September at about 1.9 storms alive at once**, and falls to about 1.1 on 1 October and 0.4 on 1 November. Storms are not spread evenly in space either: early-season storms typically form in the Gulf and off the Southeast coast, and the long tracks from the Cape Verde islands belong to the peak.\n\nDrag the time slider to **250** and back to **180** and watch the **Storms alive now** readout; set **Color storm heads by** to *Peak wind* to see which of the living storms went on to be major hurricanes.',
      options: {playing: false, time: 250, headColor: 'wind'},
      highlight: {readout: 'alive'},
      controls: ['time', 'headColor'],
      readouts: ['alive', 'aliveSpark', 'classChart']
    },
    {
      id: 'eras',
      title: 'Is the early record the same as the recent one?',
      body: 'Use **Seasons shown** to compare eras. In 1980-1999 the record has **13.8 storms per season** and its peak (average storms alive) falls around 8 September; in 2006-2025 it has **17.3 per season** and the peak is a week later, with about **2.1** alive at once. The readout shows the count for the seasons you pick.\n\nBe careful what that means: more storms per season is partly **better detection**. Short-lived and weak systems are far more likely to be found and named now than in the early satellite years, so the hurricane and major-hurricane counts (5.8 and 2.1 per season in 1980-1999, 7.5 and 3.3 in 2006-2025) are the steadier comparison, and even they are noisy over 20 seasons.',
      options: {playing: true, speed: 6, time: 200, seasons: [1980, 1999], headColor: 'peak'},
      controls: ['seasons', 'playing', 'headColor'],
      readouts: ['seasonsShown', 'aliveSpark']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'This is **46 seasons overlaid, not one season**: a cluster of storms on one date was never at sea together. The six-hourly fixes are straight-lined, so a head between fixes is on the chord; tracks west of 105 W are cut; the **Maximum fix gap** option can hide a storm over a gap in the record. Each storm is placed on the calendar of the year it started, so a storm that lives past 1 January has days beyond 365 that the slider never reaches.\n\n**Try it:** set **Seasons shown** to a single year (for example 2005 to 2005) and replay it; set the **Playback window** to 1 June to 1 December (or 1 August to 1 October for the peak); compare *Wind now* with *Peak wind of the storm* on 15 September; slow **Playback speed** to 2 days per second and follow one storm from birth to death.',
      options: {
        playing: true,
        speed: 4,
        seasons: [2005, 2005],
        headColor: 'wind',
        trailDays: 12,
        showBackdrop: true,
        backdropOpacity: 0.2
      },
      controls: ['seasons', 'dayRange', 'maxGapHours'],
      readouts: ['clock', 'seasonsShown', 'strongest']
    }
  ]
});
