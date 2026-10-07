// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {TransitPlaybackOptions} from './transit-playback.compute';
import {
  formatTransitClock,
  getTransitLegendEntries,
  TRANSIT_MODE_LABELS,
  TRANSIT_MODES
} from './transit-data';

const RANDSTAD_VIEW = {longitude: 4.75, latitude: 52.12, zoom: 8.6};

export default defineScene<TransitPlaybackOptions>({
  id: 'transit-playback',
  title: 'A timetable in motion',
  chapter: 'networks',
  order: 20,
  summary:
    'Every scheduled tram, bus, metro, train and ferry of the Randstad between 07:00 and 09:00, interpolated on the GPU at the playhead with fading trails, coloured by mode, with a vehicles-in-service curve.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [{id: 'poopdeck-gtfs-nl', role: 'scheduled trips (OVapi GTFS, 3 July 2026)'}],
  initialView: RANDSTAD_VIEW,

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
      kind: 'slider',
      id: 'markerSize',
      label: 'Arrow size',
      group: 'Vehicles',
      apply: 'param',
      min: 4,
      max: 14,
      step: 1,
      default: 6,
      unit: 'px',
      help: 'Half-length of each arrow in screen pixels. Arrows point along the heading of the segment the vehicle is on.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Speed color ramp',
      group: 'Vehicles',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.markerColor !== 'speed',
      help: 'Ramp for the speed colouring. All four are perceptually uniform.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
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
      default: 5,
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

  legends: state => [
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
            kind: 'ramp' as const,
            title: 'Scheduled speed',
            ramp: state.ramp,
            extent: [0, 108] as const,
            unit: 'km/h',
            format: (value: number) => value.toFixed(0)
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
      id: 'the-question',
      title: 'What does the Randstad run at 08:00?',
      body: 'On a Friday morning in July 2026 the timetable of the western Netherlands holds **8,448 scheduled trips** between 07:00 and 09:00: trams in Amsterdam, The Hague and Rotterdam, hundreds of bus lines, the metro, the ferries on the Nieuwe Waterweg and the intercity trains that tie the cities together. Each arrow is one of them at the playhead.\n\nPress **Play** below, or drag **Time of day**: the clock is a number the GPU compares with every trip at once, and the slider follows it. The chart under the map counts vehicles in service in every minute of the window; the line is the playhead. Change **Playback speed** to slow the morning down or run it faster.',
      camera: {...RANDSTAD_VIEW, transitionMs: 1200},
      options: {time: 3600, play: true},
      highlight: {readout: 'active'},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'active', 'inService']
    },
    {
      id: 'modes',
      title: 'Trains, trams, buses and ferries',
      body: 'Colour says what kind of vehicle it is, from the GTFS route type: **trams** green, **buses** blue, **metros** pink, **trains** orange and **ferries** purple. Buses are by far the most numerous, but the orange arrows are the ones that cross the whole map.\n\nUse **Show mode** below to isolate one mode: the markers are culled in the vertex shader and the trail mask is rewritten once, with no graph rebuilt. Pick *Train* to see only the rail network the Randstad is built around, or *Ferry* for the handful of boats. The bars show how many of each mode are in service now.',
      options: {time: 3600, play: false, modeFilter: 'all'},
      camera: {longitude: 4.7, latitude: 52.12, zoom: 9.2, transitionMs: 1400},
      controls: ['modeFilter', 'markerSize'],
      readouts: ['active', 'modeChart']
    },
    {
      id: 'speed',
      title: 'How fast is a timetable?',
      body: "Switch **Color arrows by** to *Speed*. The playhead reports the speed of the segment each vehicle is on, so the colour is the schedule's own speed: trams and buses crawl through the cities at 15 to 30 km/h, while the intercity trains between Amsterdam, Utrecht and Rotterdam glow at 120 km/h and more.\n\nThe histogram is the speed of every vehicle in service at the playhead. The tall block on the left is the urban fleet; the thin tail is the trains. Try **Speed color ramp** *Magma* below if viridis blends with the basemap.",
      options: {markerColor: 'speed', time: 3900, play: false, showTrails: false},
      camera: {longitude: 4.95, latitude: 52.2, zoom: 9.6, transitionMs: 1400},
      callout: {coordinate: [5.1101, 52.0894], text: 'Utrecht Centraal'},
      controls: ['markerColor', 'ramp'],
      readouts: ['speedChart', 'active']
    },
    {
      id: 'trails',
      title: 'Where have they just been? A window in time',
      body: '**`GPUTimeWindowFilter`** treats every segment between two timetable vertices as a time interval and keeps those that overlap `[playhead - length, playhead]`. It writes a fade weight (the old end is transparent) and a clip fraction (the oldest segment is cut part-way), compacts the live ids and writes the count straight into the draw call.\n\nSlide **Trail length** below to 15 minutes: trams draw short stubs around their lines, trains long streaks along the corridors. Set **Tail fade** to 0 for solid trails, or turn **Show every route faintly** off to see only what moved in the last minutes.',
      options: {
        markerColor: 'mode',
        showTrails: true,
        trailMinutes: 15,
        play: false,
        time: 3300,
        modeFilter: 'all'
      },
      camera: {longitude: 4.75, latitude: 52.1, zoom: 8.9, transitionMs: 1400},
      controls: ['trailMinutes', 'tailFade', 'showBackdrop'],
      readouts: ['trailSegments']
    },
    {
      id: 'peak',
      title: 'The morning peak, minute by minute',
      body: 'The line chart counts the trips running in every minute of the window. All modes together climb from about **1,550 vehicles at 07:00** to a plateau of about **2,100 from 08:00**, with the busiest minute around 08:34 (see the readout). The second line is the trains: it stays almost flat near 135 because intercity and sprinter trains run all day at fixed intervals, while buses and trams add the 500 extra vehicles of the rush hour.\n\nPress **Play** at the fastest **Playback speed** and watch the line follow the clock. Select a mode with **Show mode** and the second line follows it.',
      options: {
        play: true,
        speed: '300',
        time: 0,
        showTrails: true,
        trailMinutes: 8,
        modeFilter: 'rail'
      },
      camera: {longitude: 4.9, latitude: 52.1, zoom: 8.4, transitionMs: 1400},
      controls: ['play', 'speed', 'modeFilter'],
      readouts: ['inService', 'peak', 'active']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'These positions are **schedule, not reality**: the GTFS feed has no vehicle positions, so a delayed train is still drawn on time. 97.5 percent of trips have exact shape-distance timing; the rest are straight lines between stops. Each trip is simplified to within 20 metres in space and time, and trips are clipped to the Randstad box, so vehicles leave the map at the edges.\n\nA lot of "buses" are on-demand services (named *Flex* in the feed) that the timetable lists as scheduled trips. **Try it:** select *Metro* and follow a line under Amsterdam, click a vehicle to read its line and window, or run **Playback speed** 600x with trails of 20 minutes to see the whole morning draw itself.',
      options: {
        modeFilter: 'all',
        play: true,
        speed: '120',
        showTrails: true,
        trailMinutes: 10
      },
      camera: {...RANDSTAD_VIEW, transitionMs: 1400},
      controls: ['modeFilter', 'speed', 'trailMinutes'],
      readouts: ['selected', 'active']
    }
  ]
});
