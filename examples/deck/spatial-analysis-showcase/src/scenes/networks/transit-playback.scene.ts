// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {TransitPlaybackOptions} from './transit-playback.compute';
import {
  formatTransitClock,
  getTransitLegendEntries,
  TRANSIT_MODE_LABELS,
  TRANSIT_MODES
} from './transit-data';
import {
  getTransitSpeedClassColors,
  RANDSTAD_DATA_FRAME,
  RANDSTAD_ORIENTATION,
  RANDSTAD_SCHEDULE_CREDIT,
  TRANSIT_SPEED_LABELS
} from './randstad-network-cartography';

const RANDSTAD_VIEW = {longitude: 4.75, latitude: 52.12, zoom: 8.6};

export default defineScene<TransitPlaybackOptions>({
  id: 'transit-playback',
  title: 'A timetable in motion',
  chapter: 'networks',
  order: 8,
  summary:
    'Every scheduled tram, bus, metro, train and ferry of the Randstad between 07:00 and 09:00, interpolated on the GPU at the playhead with fading trails, coloured by mode, with a vehicles-in-service curve.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [{id: 'poopdeck-gtfs-nl', role: 'scheduled trips (OVapi GTFS, 3 July 2026)'}],
  initialView: RANDSTAD_VIEW,
  basemap: ground('night', {labels: 'above', labelPreset: 'places-only'}),
  furniture: {
    title: {
      subtitle: 'Randstad timetable · scheduled positions',
      chips: ['Scheduled, not observed']
    },
    scaleBar: {units: 'metric'},
    credit: RANDSTAD_SCHEDULE_CREDIT,
    caveat: 'Interpolated timetable positions, not real-time vehicles.',
    clock: {
      option: 'time',
      time: {origin: '2026-07-03T07:00:00+02:00', unit: 'seconds'},
      zones: ['Europe/Amsterdam'],
      show: 'time',
      progress: [0, 7200]
    }
  },
  annotations: RANDSTAD_ORIENTATION,
  timeline: {
    time: 'time',
    play: 'play',
    speed: 'speed',
    format: value => `${formatTransitClock(value)} CEST`,
    ticks: [
      {at: 0, label: '07:00'},
      {at: 3600, label: '08:00'},
      {at: 7200, label: '09:00'}
    ]
  },

  options: [
    ...playbackOptions<TransitPlaybackOptions>({
      time: {
        min: 0,
        max: 7200,
        step: 10,
        default: 3600,
        label: 'Time of day',
        format: value => `${formatTransitClock(value)} CEST`,
        help: 'Local time on Friday 3 July 2026 (CEST, UTC+2). The window is 07:00 to 09:00.'
      },
      speed: {
        kind: 'select',
        label: 'Playback speed',
        default: '120',
        options: [
          {value: '30', label: '30x (two minutes of timetable per four seconds)'},
          {value: '60', label: '60x'},
          {value: '120', label: '120x (the whole window in one minute)'},
          {value: '300', label: '300x'},
          {value: '600', label: '600x'}
        ],
        help: 'Simulated seconds per real second. At 120x the two hours play in one minute.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'modeFilter',
      label: 'Show mode',
      group: 'Vehicles',
      apply: 'param',
      default: 'all',
      help: 'Filters markers (culled in the vertex shader) and trails (an extra predicate mask in the time-window graph, rewritten once when you change the mode). The faint backdrop keeps every route.',
      options: [
        {value: 'all', label: 'All modes'},
        ...TRANSIT_MODES.map(mode => ({value: mode, label: TRANSIT_MODE_LABELS[mode]}))
      ]
    },
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Color arrows by',
      group: 'Vehicles',
      apply: 'param',
      default: 'mode',
      help: 'The mode of the trip, or the scheduled speed of the segment the vehicle is on, as reported by the playhead.',
      options: [
        {value: 'mode', label: 'Mode'},
        {value: 'speed', label: 'Speed'}
      ]
    },
    {
      kind: 'select',
      id: 'hierarchy',
      label: 'Symbol hierarchy',
      group: 'Vehicles',
      apply: 'param',
      default: 'weighted',
      display: 'segmented',
      help: 'Weighted stacks fixed symbols: bus 4 px at the bottom, tram 6, ferry and metro 8, rail 9 at the top. Flat makes every mode 6 px.',
      options: [
        {value: 'flat', label: 'Flat'},
        {value: 'weighted', label: 'Weighted'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the part of every trip inside the sliding window behind the playhead. Which segments are live is decided on the GPU by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 4,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width, written into the window parameter buffer as [playhead - length, playhead]. Keep it longer than the gap between two vertices of a trip or the trail blanks.'
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
      default: 1,
      disabledWhen: state => !state.showTrails,
      format: value => (value === 0 ? 'none' : `${Math.round(value * 100)}% of the trail`),
      help: 'How much of the trail fades toward its oldest end (the start-fade duration of the window). 100% fades the whole trail, 0 keeps it solid.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every route faintly',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the line of every trip of the window as a thin gray line so you can see the network the vehicles follow.'
    }
  ],

  readouts: [
    {
      id: 'inService',
      label: 'Vehicles in service',
      kind: 'chart',
      help: 'Scheduled trips running in each minute of the window, all modes and one mode (trains unless a mode is selected). The line is the playhead. Counted from trip start and end times.'
    },
    {
      id: 'modeChart',
      label: 'In service now, by mode',
      kind: 'chart',
      help: 'Vehicles in service at the playhead per mode, counted from the status the GPU writes for every trip.'
    },
    {
      id: 'speedChart',
      label: 'How fast are they going?',
      kind: 'chart',
      help: 'Scheduled speed of every vehicle in service at the playhead, 10 km/h bins. The bulk is trams and buses below 40 km/h; trains form the long tail.'
    },
    {
      id: 'binarySearch',
      label: 'Selected interpolation bracket',
      kind: 'chart',
      help: 'A deterministic active trip supplies its actual timetable vertices, bracketing pair and interpolation fraction at the playhead.'
    },
    {id: 'clock', label: 'Playhead', help: 'Local time on 3 July 2026.'},
    {
      id: 'active',
      label: 'Vehicles in service now',
      format: 'integer',
      help: 'Trips whose first and last time bracket the playhead: the arrows drawn.'
    },
    ...TRANSIT_MODES.map(mode => ({
      id: `active-${mode}`,
      label: `${TRANSIT_MODE_LABELS[mode]}s in service`,
      format: 'integer' as const,
      help: `Vehicles of mode ${mode} in service now.`
    })),
    {
      id: 'waiting',
      label: 'Not yet started',
      format: 'integer',
      help: 'Trips whose first time in the window is after the playhead.'
    },
    {
      id: 'finished',
      label: 'Already finished',
      format: 'integer',
      help: 'Trips whose last time in the window is before the playhead.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {id: 'trips', label: 'Trips in the dataset'},
    {
      id: 'peak',
      label: 'Busiest minute',
      help: 'The minute of the window with most vehicles in service.'
    },
    {
      id: 'selected',
      label: 'Selected vehicle',
      help: 'Click a vehicle to read its line, its scheduled window and its speed now.'
    }
  ],

  legends: (state, data) => [
    ...(state.markerColor === 'mode' || state.showTrails
      ? [
          {
            kind: 'categories' as const,
            title: 'Mode',
            entries: getTransitLegendEntries(),
            note: 'GTFS route type: tram, bus, metro, rail (train) and ferry.'
          }
        ]
      : []),
    ...(state.markerColor === 'speed'
      ? [
          {
            kind: 'categories' as const,
            title: 'Scheduled speed',
            entries: TRANSIT_SPEED_LABELS.map((label, index) => ({
              color: getTransitSpeedClassColors(data.ground !== 'light')[index],
              label: `${label} km/h`
            })),
            note: 'Fixed classes; speed is timetable interpolation, not a GPS measurement.'
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTrajectoryPlayhead, getGPUTrajectoryPlayheadParameterValues} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// positions: float32x2 planar meters, timestamps: float32 seconds since 07:00, trackOffsets: uint32 (trips + 1)
const playGraph = new GPUCommandGraph(device, {id: 'timetable-playhead'});
playGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets,
  parameters: playheadParameters.importToGraph(playGraph),
  currentPositions, headings, speeds, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                         // indirect draw record: no readback to draw
}));
const play = playGraph.compile();           // once

const trailGraph = new GPUCommandGraph(device, {id: 'timetable-trails'});
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,   // interval mode: one row per segment
  window: windowParameters.importToGraph(trailGraph),
  additionalPredicates: [{kind: 'selection', mask: modeMask}],   // ${state.modeFilter === 'all' ? 'all modes' : `only ${state.modeFilter}`}
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, trackIds: segmentTrips, trackVisibleCounts, drawInstanceCount
}));

// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailMinutes * 60}, end: playhead, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));
play.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every trip at once, the two timetable vertices either side of the clock (a binary search per trip) and interpolates position, heading and speed. `GPUTimeWindowFilter` keeps the trip segments that overlap a sliding window and fades them toward the tail. Both write their results straight into an indirect draw record, so nothing returns to the CPU to draw.',
    why: 'A timetable is a promise about movement. Seeing every scheduled vehicle of a region at once shows where service is dense, where modes hand over to each other and how the morning peak builds, which is the starting point for questions about capacity, frequency and access.',
    howToRead:
      'Arrows are vehicles at the playhead, pointing along their route and coloured by mode or speed. Trails show the last few minutes of each trip. These are **scheduled** positions interpolated between stops along the route shape, not real-time positions: a vehicle that was late on the day is drawn where the timetable says it should be.'
  },

  create: async ctx => (await import('./transit-playback.compute')).createTransitPlayback(ctx),

  story: [
    {
      id: 'morning',
      headline: 'A morning in motion',
      textAlternative:
        'Scheduled vehicles move through the Randstad at the current local timetable time.',
      optionsMode: 'fresh',
      title: 'What does the Randstad run at 08:00?',
      body: 'Each arrow is a scheduled vehicle at the local playhead; its tail makes the preceding four minutes legible without overwhelming the map. Press **Play** or drag time: the GPU compares that one clock with every trip, and the timeline marks the active instant. This is a timetable animation, not a claim about where vehicles actually were.',
      camera: {...RANDSTAD_VIEW, transitionMs: 1200},
      options: {time: 3600, play: false, hierarchy: 'weighted', trailMinutes: 4},
      highlight: {readout: 'active'},
      controls: ['play', 'time'],
      readouts: ['clock', 'active', 'inService']
    },
    {
      id: 'hierarchy',
      headline: 'A swarm still needs hierarchy',
      textAlternative: 'Larger rail and metro markers remain legible above bus and tram service.',
      optionsMode: 'fresh',
      title: 'Trains, trams, buses and ferries',
      body: 'Colour says what kind of vehicle it is, from the GTFS route type: **trams** green, **buses** blue, **metros** pink, **trains** orange and **ferries** purple. Buses are by far the most numerous, but the orange arrows are the ones that cross the whole map.\n\nUse **Show mode** below to isolate one mode: the markers are culled in the vertex shader and the trail mask is rewritten once, with no graph rebuilt. Pick *Train* to see only the rail network the Randstad is built around, or *Ferry* for the handful of boats. The bars show how many of each mode are in service now.',
      options: {time: 3600, play: false, modeFilter: 'all'},
      camera: {longitude: 4.7, latitude: 52.12, zoom: 9.2, transitionMs: 1400},
      controls: ['modeFilter', 'hierarchy'],
      readouts: ['active', 'modeChart']
    },
    {
      id: 'speed',
      headline: 'Scheduled speed comes in classes',
      textAlternative: 'Active trips are coloured by their scheduled segment speed class.',
      optionsMode: 'fresh',
      title: 'How fast is a timetable?',
      body: 'Switch **Colour arrows by** to speed. The playhead reports the scheduled speed of its current segment, not a GPS observation, and the histogram uses the same active vehicles. Fast rail chords are a simplification of a timetable shape, so use them to compare scheduled movement—not to infer track speed.',
      options: {markerColor: 'speed', time: 3900, play: false, showTrails: false},
      camera: {longitude: 4.95, latitude: 52.2, zoom: 9.6, transitionMs: 1400},
      callout: {coordinate: [5.1101, 52.0894], text: 'Utrecht Centraal'},
      controls: ['markerColor'],
      readouts: ['speedChart', 'active']
    },
    {
      id: 'pulse',
      headline: 'The timetable has a pulse',
      textAlternative: 'A long rail trail and the time bar show service through the morning.',
      optionsMode: 'fresh',
      title: 'Where have they just been? A window in time',
      body: '**`GPUTimeWindowFilter`** treats every segment between two timetable vertices as a time interval and keeps those that overlap `[playhead - length, playhead]`. It writes a fade weight (the old end is transparent) and a clip fraction (the oldest segment is cut part-way), compacts the live ids and writes the count straight into the draw call.\n\nSlide **Trail length** below to 15 minutes: trams draw short stubs around their lines, trains long streaks along the corridors. Set **Tail fade** to 0 for solid trails, or turn **Show every route faintly** off to see only what moved in the last minutes.',
      options: {
        markerColor: 'mode',
        showTrails: true,
        trailMinutes: 15,
        play: false,
        time: 3300,
        modeFilter: 'rail'
      },
      camera: {longitude: 4.75, latitude: 52.1, zoom: 8.9, transitionMs: 1400},
      controls: ['trailMinutes', 'showBackdrop'],
      readouts: ['trailSegments']
    },
    {
      id: 'binary-search',
      headline: 'Find every vehicle without a readback',
      textAlternative: 'The selected trip is bracketed by timetable vertices around the playhead.',
      title: 'Find every vehicle without a readback',
      body: 'For each trip, `GPUTrajectoryPlayhead` binary-searches the timetable vertices around the clock, interpolates position and heading, then writes compact active IDs directly to an indirect draw command. `GPUTimeWindowFilter` performs the matching overlap test for trails. The CPU reads only occasional summaries for the charts, never a frame of positions.',
      options: {
        play: true,
        speed: '300',
        time: 0,
        showTrails: true,
        trailMinutes: 8,
        modeFilter: 'rail'
      },
      camera: {longitude: 4.9, latitude: 52.1, zoom: 8.4, transitionMs: 1400},
      annotations: RANDSTAD_DATA_FRAME,
      controls: ['play', 'time'],
      readouts: ['binarySearch', 'selected', 'active']
    }
  ]
});
