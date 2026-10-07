// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {WORLD, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import {formatDayOfYear, HURRICANE_CATEGORY_LABELS} from './hurricane-data';
import {HURRICANE_CLASS} from '../../cartography/hue-registry';
import type {HurricaneSeasonOptions} from './hurricane-season.compute';

const BASIN_VIEW = {longitude: -60, latitude: 26, zoom: 3.2};

const dayLabel = (day: number) => `${formatDayOfYear(day)} (day ${Math.floor(day) + 1})`;
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['folded seasons'] as const
});
const SEASON_LABELS = labelsFor(WORLD, ['atlantic-ocean', 'caribbean-sea', 'gulf-of-mexico']);

export default defineScene<HurricaneSeasonOptions>({
  id: 'hurricane-season',
  title: 'When does the Atlantic season peak?',
  chapter: 'earth',
  order: 3,
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
      kind: 'select',
      id: 'denominator',
      label: 'Campfire denominator',
      group: 'Seasonal chart',
      apply: 'param',
      default: 'active',
      help: 'Show the selected record as active storms, or divide the same stacked totals by the number of selected seasons. Both modes retain the same storms and calendar.',
      options: [
        {value: 'active', label: 'Storms active'},
        {value: 'share', label: 'Share of selected seasons active'}
      ]
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
        {value: 'peak', label: 'Peak wind of the storm'}
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
    {
      kind: 'categories' as const,
      title: state.headColor === 'wind' ? 'Wind now' : 'Peak wind of the storm',
      entries: HURRICANE_CATEGORY_LABELS.map((label, index) => ({
        color: HURRICANE_CLASS.dark[index],
        label
      })),
      note: 'Saffir-Simpson classes come from sustained wind. Trails are colored by wind at each fix.'
    }
  ],

  readouts: [
    {
      id: 'seasonCampfire',
      label: 'Active storm campfire',
      kind: 'chart',
      help: 'Five-day-smoothed active storm bands from 1 May through 31 December. The rule marks the external NHC climatological reference on 10 September; it is not a result from this subset.'
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
    {id: 'seasonsShown', label: 'Selected seasons'},
    {id: 'peakActive', label: 'Peak active total'},
    {id: 'peakShare', label: 'Peak season share'},
    {id: 'categoryComposition', label: 'Peak category composition'}
  ],

  pipeline: [
    {
      id: 'fold',
      label: 'Fold calendar',
      detail: 'Place each season on its own January-to-December clock'
    },
    {id: 'playhead', label: 'Playhead', detail: 'Interpolate every storm at one folded day'},
    {id: 'window', label: 'Time window', detail: 'Keep the trailing six-hour segments'},
    {id: 'draw', label: 'Draw', detail: 'Classed heads, trails and seasonal summaries'}
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
      'Every dot is a different storm from a different year, alive on the same calendar day. A dot is colored by its wind now; its trail shows where it has been. The campfire stacks active strength bands with the playhead rule. Counts are summed over all chosen seasons, while share divides that same total by the selected season count.'
  },

  basemap: ground('abyss', {labels: 'none'}),
  furniture: {
    title: cartouche(
      'When does the Atlantic season peak?',
      'Active storms · folded calendar · NOAA IBTrACS · 1980–2025'
    ),
    credit: joinCredits(CREDITS.noaaNhc, 'NOAA IBTrACS', CREDITS.naturalEarth),
    caveat: 'Years are stacked: the map is a seasonal calendar, not storms simultaneously at sea.'
  },
  annotations: SEASON_LABELS,

  create: async ctx => (await import('./hurricane-season.compute')).createHurricaneSeason(ctx),

  story: [
    {
      id: 'fold',
      title: 'When is the Atlantic hurricane season?',
      headline: 'Fold every season onto one calendar',
      textAlternative: 'Classed hurricane heads from many years share one Atlantic calendar day.',
      optionsMode: 'fresh',
      body: 'Press **Play** to fold every available season onto one calendar. Each dot is a NOAA IBTrACS best-track storm placed at the same day of its own year and colored by wind; trails show its recent route. The **Storms alive now** card and seasonal chart report the selected record directly.\n\nThe clock is a calendar-day value that the GPU compares against every storm at once. Drag **Day of the year** to jump anywhere, or let playback reveal the quiet start, broad peak, and late-season decline.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {playing: true, speed: 6, time: 150, trailDays: 6, showBackdrop: true},
      highlight: {readout: 'alive'},
      controls: ['playing', 'time', 'speed'],
      readouts: ['clock', 'alive', 'seasonsShown']
    },
    {
      id: 'move',
      title: 'One calendar, 46 seasons',
      headline: 'Storms gather, travel, and fade',
      textAlternative: 'Atlantic hurricane trails fade behind category-coloured heads.',
      optionsMode: 'fresh',
      body: 'The time column is re-based to **days since 1 January of each storm’s year**. **`GPUTrajectoryPlayhead`** binary-searches every track for the fixes around the playhead and interpolates one head per storm.\n\nAt the selected calendar day, **Storms alive now** counts storms from all selected seasons. Trails make elapsed time visible; the campfire belongs to the next step, where the stacked calendar is read as a summary.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {playing: false, time: 250, trailDays: 6},
      highlight: {readout: 'strongest'},
      controls: ['time', 'headColor'],
      readouts: ['alive', 'strongest', 'classChart']
    },
    {
      id: 'peak',
      title: 'The seasonal campfire',
      headline: 'The season has a broad peak',
      textAlternative:
        'Stacked hurricane classes form a seasonal campfire with a labelled NHC reference rule.',
      optionsMode: 'fresh',
      body: 'The campfire is a five-day-smoothed stack of TD, TS, hurricane, and major-hurricane activity from 1 May to 31 December. The vertical rule is **NHC climatological reference: 10 September**—an external reference, not a peak calculated from this subset.\n\nScrub **Day of the year** to move the rule through the same selected storms and compare live class composition without claiming one literal peak date.',
      options: {playing: false, time: 250, headColor: 'wind'},
      highlight: {readout: 'alive'},
      controls: ['time', 'headColor'],
      readouts: ['alive', 'seasonCampfire', 'classChart']
    },
    {
      id: 'denominator',
      title: 'Counts and season shares differ',
      headline: 'The denominator changes the unit',
      textAlternative:
        'The same stacked campfire switches between storms active and the share of selected seasons active.',
      optionsMode: 'fresh',
      body: 'Choose **Campfire denominator**. *Storms active* stacks every selected season; *share of selected seasons active* divides the identical bands by selected season N. Neither mode changes the map or the selected record.\n\nThis makes the denominator explicit: a stacked calendar can contain many storms from different years, while a season share is bounded by the selected years.',
      options: {playing: false, time: 250, denominator: 'share'},
      controls: ['denominator', 'seasons'],
      readouts: ['seasonsShown', 'peakActive', 'peakShare', 'categoryComposition']
    },
    {
      id: 'eras',
      title: 'Is the early record the same as the recent one?',
      headline: 'The observing record changes',
      textAlternative: 'An early-era folded storm map makes its smaller seasonal sample explicit.',
      optionsMode: 'fresh',
      body: 'Use **Seasons shown** to compare periods. The live season card and sparkline recalculate for the years you choose rather than preserving a historical summary.\n\nInterpret differences cautiously: changing observation and naming practices affect short-lived and weak systems, while any short period is noisy. This folded map compares the record, not a climate attribution.',
      options: {playing: true, speed: 6, time: 200, seasons: [1980, 1999], headColor: 'peak'},
      controls: ['seasons', 'playing', 'headColor'],
      readouts: ['seasonsShown', 'seasonCampfire', 'peakShare']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      headline: 'A folded calendar is not one season',
      textAlternative: 'Faint Atlantic context tracks make clear that years have been stacked.',
      optionsMode: 'fresh',
      body: 'This is an overlay of seasons, not storms that were simultaneously at sea. Six-hourly fixes are joined by straight chords; a head between fixes is interpolated, and **Maximum fix gap** can hide uncertain intervals. A storm that crosses a year boundary may also extend beyond this folded calendar.\n\n**Try it:** show a single season and replay it; narrow the **Playback window** around the peak; compare *Wind now* with *Peak wind of the storm*; then slow playback and follow one route from birth to death.',
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
