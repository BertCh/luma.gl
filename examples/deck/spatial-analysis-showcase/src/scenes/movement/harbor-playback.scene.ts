// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {HarborPlaybackOptions} from './harbor-playback.compute';
import {getVesselLegendEntries, VESSEL_CATEGORIES, VESSEL_CATEGORY_LABELS} from './b12-tracks';

const HARBOR_VIEW = {longitude: -74.05, latitude: 40.665, zoom: 10.9};

export default defineScene<HarborPlaybackOptions>({
  id: 'harbor-playback',
  title: 'Who is moving in New York Harbor?',
  chapter: 'movement',
  order: 1,
  summary:
    'Replay a full day of AIS traffic in New York / New Jersey Harbor on the GPU: every vessel interpolated at the playhead, fading trails from a time window, evenly resampled routes and stop detection for moored craft.',
  contributors: [
    'GPUTrajectoryPlayhead',
    'GPUTimeWindowFilter',
    'GPUTrajectoryResample',
    'GPUTrajectoryMetrics'
  ],
  datasets: [{id: 'ais-vessels', role: 'vessel tracks (AIS, 12 June 2024)'}],
  initialView: HARBOR_VIEW,

  options: [
    ...playbackOptions<HarborPlaybackOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: 86400,
        step: 60,
        default: 68400,
        label: 'Time of day (UTC)',
        format: formatPlaybackTime.clockUtc,
        help: 'New York was on daylight time, so local time is four hours earlier (68,400 s is 19:00 UTC, 3 pm).'
      },
      speed: {
        min: 30,
        max: 1800,
        step: 30,
        default: 240,
        unit: 'x',
        help: 'Simulated seconds per real second. 240x plays one hour in 15 seconds; 1,800x plays the whole day in under a minute.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'maxGapMinutes',
      label: 'Maximum fix gap',
      group: 'Playback',
      apply: 'param',
      min: 0,
      max: 20,
      step: 1,
      default: 0,
      unit: 'min',
      format: value => (value === 0 ? 'off' : `${value} min`),
      help: 'When the time between the two fixes either side of the playhead is longer than this, the vessel is flagged as in a gap and hidden instead of drawn at a guessed position. 0 turns the test off. Moored vessels report every 5 minutes, so a limit under 5 hides them.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: true,
      help: 'Draws the part of every track inside the sliding window behind the playhead. Which segments are live is decided on the GPU by GPUTimeWindowFilter.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails (time window)',
      apply: 'param',
      min: 2,
      max: 120,
      step: 1,
      default: 20,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width. It is written into the window parameter buffer: [playhead - length, playhead].'
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
      help: 'How much of the trail fades out toward its oldest end (the start-fade duration of the window). 100% fades the whole trail; 0 keeps it solid.'
    },
    {
      kind: 'select',
      id: 'trailColor',
      label: 'Color trails by',
      group: 'Trails (time window)',
      apply: 'param',
      default: 'category',
      disabledWhen: state => !state.showTrails,
      help: 'Vessel type, or the speed of each step from GPUTrajectoryMetrics (the step that ends at the segment end).',
      options: [
        {value: 'category', label: 'Vessel type'},
        {value: 'speed', label: 'Step speed'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'Show resampled routes',
      group: 'Resampled routes',
      apply: 'param',
      default: false,
      help: 'Draws every track rebuilt as a fixed number of evenly spaced samples by GPUTrajectoryResample, with a dot at each sample.'
    },
    {
      kind: 'select',
      id: 'routeSamples',
      label: 'Samples per route',
      group: 'Resampled routes',
      apply: 'compile',
      default: '32',
      disabledWhen: state => !state.showRoutes,
      help: 'Compile-time: the dense output table has tracks x samples rows, so changing it rebuilds the resample graph.',
      options: [
        {value: '16', label: '16 samples'},
        {value: '32', label: '32 samples'},
        {value: '64', label: '64 samples'}
      ]
    },
    {
      kind: 'select',
      id: 'routeSpacing',
      label: 'Sample spacing',
      group: 'Resampled routes',
      apply: 'compile',
      default: 'arc-length',
      disabledWhen: state => !state.showRoutes,
      help: 'Compile-time. Arc length places samples at equal distances along the path; time places them at equal time steps, so slow stretches (docks, anchorages) collect many samples.',
      options: [
        {value: 'arc-length', label: 'Equal distance along the path'},
        {value: 'time', label: 'Equal time steps'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showStops',
      label: 'Show stops',
      group: 'Stops',
      apply: 'param',
      default: false,
      help: 'Marks every place a vessel dwelled: a run of slow steps lasting at least the minimum duration. Radius and color grow with the duration.'
    },
    {
      kind: 'slider',
      id: 'stopSpeedKnots',
      label: 'Stop speed threshold',
      group: 'Stops',
      apply: 'param',
      min: 0.1,
      max: 3,
      step: 0.1,
      default: 0.5,
      unit: 'kn',
      help: 'A step is slow when the vessel covers less than this speed times the step time. Parameter buffer: changing it re-runs the graph, never recompiles.'
    },
    {
      kind: 'slider',
      id: 'stopMinutes',
      label: 'Minimum stop duration',
      group: 'Stops',
      apply: 'param',
      min: 2,
      max: 120,
      step: 1,
      default: 15,
      unit: 'min',
      help: 'A run of slow steps becomes a stop only when it lasts at least this long. Shorter runs (a ferry waiting for a berth) are ignored.'
    },
    {
      kind: 'select',
      id: 'vesselFilter',
      label: 'Show vessel type',
      group: 'Vessels',
      apply: 'param',
      default: 'all',
      help: 'Filters markers and trails to one AIS category. Markers are culled in the vertex shader; trails use an extra predicate mask in the time-window graph, written once when you change the type.',
      options: [
        {value: 'all', label: 'All vessels'},
        ...VESSEL_CATEGORIES.map(category => ({
          value: category,
          label: VESSEL_CATEGORY_LABELS[category]
        }))
      ]
    },
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Color arrows by',
      group: 'Vessels',
      apply: 'param',
      default: 'category',
      help: 'Vessel type from the AIS ship-type code, or speed over ground (the playhead reports the speed of the segment each vessel is on).',
      options: [
        {value: 'category', label: 'Vessel type'},
        {value: 'speed', label: 'Speed'}
      ]
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Arrow size',
      group: 'Vessels',
      apply: 'param',
      min: 4,
      max: 16,
      step: 1,
      default: 8,
      unit: 'px',
      help: 'Half-length of each arrow in screen pixels. Arrows point along the heading of the segment the vessel is on.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Speed color ramp',
      group: 'Vessels',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp used wherever speed is the color (arrows or trails). All four are perceptually uniform.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every track faintly',
      group: 'Vessels',
      apply: 'param',
      default: true,
      help: 'Draws all 897 tracks as thin gray lines so you can see the shipping lanes the vessels follow.'
    }
  ],

  readouts: [
    {
      id: 'hourlyChart',
      label: 'Traffic through the day',
      kind: 'chart',
      help: 'Tracks with a position in each UTC hour. The line is the playhead.'
    },
    {
      id: 'speedChart',
      label: 'How fast do tracks go?',
      kind: 'chart',
      help: 'Mean speed of every track from GPUTrajectoryMetrics. The tall bar at 0 to 1 kn is moored vessels; ferries and tows fill 5 to 20 kn.'
    },
    {
      id: 'dwellChart',
      label: 'How long are the stops?',
      kind: 'chart',
      help: 'Duration of every stop found at the current speed threshold and minimum duration. Empty until stops are detected.'
    },
    {id: 'clock', label: 'Playhead', help: 'Simulated time on 12 June 2024.'},
    {
      id: 'active',
      label: 'Vessels with a position now',
      format: 'integer',
      help: 'Tracks whose first and last fix bracket the playhead (and pass the gap test): the arrows drawn.'
    },
    {
      id: 'beforeStart',
      label: 'Not yet reporting',
      format: 'integer',
      help: 'Tracks whose first fix is after the playhead.'
    },
    {
      id: 'afterEnd',
      label: 'Already gone',
      format: 'integer',
      help: 'Tracks whose last fix is before the playhead.'
    },
    {
      id: 'inGap',
      label: 'In a data gap',
      format: 'integer',
      help: 'Tracks hidden because the fixes around the playhead are further apart than the maximum gap.'
    },
    {
      id: 'trailSegments',
      label: 'Trail segments live',
      format: 'integer',
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'tracks',
      label: 'Tracks',
      help: 'AIS tracks; a vessel has more than one when it stopped reporting for over 20 minutes.'
    },
    {id: 'vertices', label: 'Fixes'},
    {
      id: 'distance',
      label: 'Distance sailed',
      help: 'Sum of every track path length, in nautical miles.'
    },
    {
      id: 'fastest',
      label: 'Fastest step',
      help: 'Highest speed over any single step. Fast ferries top 40 knots.'
    },
    {id: 'stops', label: 'Stops detected', help: 'Number of stops at the current thresholds.'},
    {id: 'stopTracks', label: 'Tracks with a stop'},
    {id: 'longestStop', label: 'Longest stop'},
    {id: 'resampled', label: 'Resampled table'},
    {
      id: 'selected',
      label: 'Selected vessel',
      help: 'Click a vessel to read its type, MMSI, length, distance and speeds.'
    }
  ],

  legends: state => {
    const usesCategory =
      state.markerColor === 'category' ||
      (state.showTrails && state.trailColor === 'category') ||
      state.showRoutes;
    const usesSpeed =
      state.markerColor === 'speed' || (state.showTrails && state.trailColor === 'speed');
    return [
      ...(usesCategory
        ? [
            {
              kind: 'categories' as const,
              title: 'Vessel type',
              entries: getVesselLegendEntries(),
              note: 'AIS ship-type code grouped into seven classes.'
            }
          ]
        : []),
      ...(usesSpeed
        ? [
            {
              kind: 'ramp' as const,
              title: 'Speed over ground',
              ramp: state.ramp,
              extent: [0, 25] as const,
              unit: 'kn',
              format: (value: number) => value.toFixed(0)
            }
          ]
        : []),
      ...(state.showStops
        ? [
            {
              kind: 'categories' as const,
              title: 'Stops (radius and color grow with the dwell)',
              entries: [
                {color: [255, 199, 224, 255] as const, label: 'A few minutes'},
                {color: [255, 41, 128, 255] as const, label: 'About 1.5 hours'},
                {color: [191, 0, 51, 255] as const, label: '6 hours or more'}
              ]
            }
          ]
        : [])
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryPlayhead, GPUTrajectoryMetrics, GPUTrajectoryResample,
  getGPUTrajectoryPlayheadParameterValues, getGPUTrajectoryMetricsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// positions: float32x2 planar meters, timestamps: float32 seconds, trackOffsets: uint32 (tracks + 1)
const playGraph = new GPUCommandGraph(device, {id: 'playback'});
playGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets,
  parameters: playheadParameters.importToGraph(playGraph),
  currentPositions, headings, speeds, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                       // indirect draw record: no readback to draw
}));
const play = playGraph.compile();         // once

const trailGraph = new GPUCommandGraph(device, {id: 'trails'});
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,   // interval mode: one row per segment
  window: windowParameters.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));

const stopGraph = new GPUCommandGraph(device, {id: 'stops'});
stopGraph.add(new GPUTrajectoryMetrics({
  positions, timestamps, trackOffsets,
  parameters: stopParameters.importToGraph(stopGraph),
  stepSpeeds, averageSpeeds, maximumSpeeds, trackStopCounts,
  stops: {output: {ids, count, overflow}, centroids, durations, drawInstanceCount: stopDrawCount}
}));
${
  state.showRoutes
    ? `
const routeGraph = new GPUCommandGraph(device, {id: 'routes'});
routeGraph.add(new GPUTrajectoryResample({
  positions, timestamps, trackOffsets,
  sampleCount: ${state.routeSamples}, spacing: '${state.routeSpacing}',   // compile-time
  samples                                   // tracks x ${state.routeSamples} float32x2 rows
}));
`
    : ''
}
// every frame: parameters are plain buffer writes
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: ${state.maxGapMinutes * 60}}));
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailMinutes * 60}, end: playhead, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));
stopParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${((state.stopSpeedKnots * 1852) / 3600).toFixed(2)},   // m/s (${state.stopSpeedKnots} kn)
  stopMinimumDuration: ${state.stopMinutes * 60}                              // s
}));
play.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUTrajectoryPlayhead` finds, for every vessel at once, the two AIS fixes either side of the clock (a binary search per track) and linearly interpolates the position, heading and speed. `GPUTimeWindowFilter` keeps the track segments that overlap a sliding window and fades them toward the tail. `GPUTrajectoryMetrics` measures each track and detects stops, and `GPUTrajectoryResample` rebuilds each track with a fixed number of samples.',
    why: 'A harbor is a moving system: pilots, port authorities and researchers ask who is where at a given time, what a vessel did in the last half hour, and where things dwell. Doing the interpolation, windowing and stop detection on the GPU means the whole day of tracks stays live as you scrub, with parameter writes instead of rebuilds.',
    howToRead:
      'Arrows are vessels now, pointing the way they are heading, colored by type or speed. Trails show where each vessel has been; a trail that is brighter and longer means a faster vessel. Pink discs are stops, larger and darker for longer dwells. AIS reports positions about once a minute, so everything between fixes is a straight-line estimate.'
  },

  create: async ctx => (await import('./harbor-playback.compute')).createHarborPlayback(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Who is moving in New York Harbor at 3 pm?',
      body: 'On Wednesday 12 June 2024 the harbor was a crowd of Staten Island ferries, tugs pushing barges, container ships bound for Port Newark and pleasure boats. The Coast Guard AIS record holds **897 tracks from 471 vessels** that day, one position about every minute.\n\nEach arrow is a vessel at the playhead (19:00 UTC, 3 pm local), pointing along its heading and colored by type (see the legend). Press **Play** below (or drag **Time of day (UTC)**) and watch the day unfold: the clock is a number the GPU compares against every track at once, and the **Time of day** slider follows it. Drag the slider to jump anywhere and keep playing. The chart below shows how many tracks report in each hour; its line is the playhead, and the afternoon peak is the busiest part of the day.',
      camera: {...HARBOR_VIEW, transitionMs: 1200},
      options: {time: 68400, playing: true},
      highlight: {readout: 'active'},
      controls: ['playing', 'time', 'loop'],
      readouts: ['clock', 'active', 'hourlyChart']
    },
    {
      id: 'playhead',
      title: 'Between the fixes: the playhead',
      body: '**`GPUTrajectoryPlayhead`** runs one thread per track. It binary-searches the track for the first fix after the playhead, then interpolates position, heading and speed along that segment. A track whose first fix is still ahead is *before start*, one whose last fix has passed is *after end*; the **Vessels with a position now** readout counts the rest.\n\nSwitch **Color arrows by** to *Speed*: slow tugs and moored boats turn dark, the fast ferries glow. The speed is the speed of the segment, so it changes in steps at each fix, exactly as the data does.\n\nThe histogram below is the mean speed of every track from `GPUTrajectoryMetrics`: a tall bar at 0 to 1 kn (vessels tied up all day) and a long tail to 20 kn, the ferries.',
      camera: {longitude: -74.035, latitude: 40.672, zoom: 12.1, transitionMs: 1600},
      options: {markerColor: 'speed', time: 68400, playing: false},
      callout: {coordinate: [-74.042, 40.672], text: 'Staten Island Ferry lane'},
      controls: ['markerColor', 'time'],
      readouts: ['clock', 'active', 'speedChart']
    },
    {
      id: 'gaps',
      title: 'Do not guess across a gap',
      body: 'AIS silence is data too. Moving vessels report once a minute, but moored ones were thinned to once every **5 minutes**, and receivers drop out. Between two distant fixes a straight line is a poor guess.\n\nSlide **Maximum fix gap** to 4 minutes: every vessel whose neighbouring fixes are more than 4 minutes apart is flagged *in a data gap* and disappears, instead of being drawn somewhere it may not be. Watch the moored craft at the piers vanish while the moving traffic stays. Set it back to *off* to bring them back.',
      options: {maxGapMinutes: 4, markerColor: 'speed', playing: false},
      highlight: {readout: 'inGap'},
      controls: ['maxGapMinutes'],
      readouts: ['inGap', 'active']
    },
    {
      id: 'trails',
      title: 'Where have they been? A window in time',
      body: '**`GPUTimeWindowFilter`** treats every segment between two fixes as a time interval and keeps the ones that overlap the window `[playhead - length, playhead]`. It also writes a fade weight (old end transparent) and a clip fraction (the oldest segment is cut part-way), then compacts the live ids and writes the count straight into the draw call. Nothing returns to the CPU.\n\nTrails are now 45 minutes long (**Trail length**) and colored by the **step speed** (**Color trails by**) that `GPUTrajectoryMetrics` measured: ferries leave long bright streaks across the bay, tugs crawl. Try **Tail fade** at 0 for solid trails, or set **Show vessel type** to *Passenger* to see the ferry shuttles.',
      options: {
        trailMinutes: 45,
        trailColor: 'speed',
        markerColor: 'category',
        maxGapMinutes: 0,
        playing: false
      },
      camera: {longitude: -74.05, latitude: 40.665, zoom: 11.2, transitionMs: 1400},
      controls: ['trailMinutes', 'trailColor', 'tailFade', 'vesselFilter'],
      readouts: ['trailSegments']
    },
    {
      id: 'resample',
      title: 'Every track as 32 evenly spaced points',
      body: 'Many analyses need tracks of the same length: similarity, clustering, a fixed-size trail. **`GPUTrajectoryResample`** rebuilds every track as a fixed number of samples. Here each route is 32 points placed at **equal distances along the path**, drawn as dots on a polyline (trails are hidden to keep the picture clear).\n\nThe **Samples per route** and **Sample spacing** are *compile-time* options, so their controls show a rebuild badge. Switch **Sample spacing** to *Equal time steps* below: now docks and anchorages, where vessels spend hours, soak up most of the dots, and the long fast legs get only a few.',
      options: {
        showTrails: false,
        showRoutes: true,
        routeSamples: '32',
        routeSpacing: 'arc-length'
      },
      camera: {longitude: -74.05, latitude: 40.655, zoom: 10.6, transitionMs: 1400},
      controls: ['showRoutes', 'routeSamples', 'routeSpacing'],
      readouts: ['resampled']
    },
    {
      id: 'stops',
      title: 'Where do vessels stop?',
      body: 'A **stop** is a run of slow steps (below the speed threshold) that lasts at least the minimum duration. `GPUTrajectoryMetrics` finds them for every track in one pass: a speed-plus-duration rule (the MovingPandas stop detector uses a diameter plus a duration instead). Pink discs are stops, sized and colored by dwell; at 0.5 knots and 15 minutes you see the anchorages in the Upper Bay, the Bayonne and Port Newark piers and the ferry slips light up.\n\nDrag **Stop speed threshold** up to 2 knots: drifting craft and slow tows join in. Raise **Minimum stop duration** to 60 minutes and only the long stays remain; the histogram shows the duration of every stop, with most under an hour and a tail of all-day berths. Both are parameter-buffer writes, so the stop list updates without any recompile.',
      options: {
        showTrails: false,
        showRoutes: false,
        showStops: true,
        stopSpeedKnots: 0.5,
        stopMinutes: 15,
        markerColor: 'category'
      },
      camera: {longitude: -74.07, latitude: 40.665, zoom: 11.3, transitionMs: 1400},
      callout: {coordinate: [-74.039, 40.66], text: 'Upper Bay anchorage 21B'},
      highlight: {readout: 'stops'},
      controls: ['showStops', 'stopSpeedKnots', 'stopMinutes'],
      readouts: ['stops', 'stopTracks', 'longestStop', 'dwellChart']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: "Interpolation is a straight line between fixes, so a vessel cutting a corner between two minute-spaced fixes is drawn on the chord, and a 5-minute gap on a moored boat hides real movement. Speeds are planar speeds between fixes, not the vessel's reported speed through water. Positions are an AIS day in UTC; New York local time is four hours earlier. The zones of the next chapters (anchorages, channels) are official only where noted.\n\n**Try it:** set **Show vessel type** to *Tug or tow* and watch pushes cross the Kill Van Kull; set **Playback speed** to 1,800x to see the day in under a minute; compare the busiest hour (19:00 UTC) with the quietest (06:00); change **Stop speed threshold** and see how many tracks have a stop.",
      options: {
        showStops: true,
        showTrails: true,
        playing: true,
        playbackSpeed: 600,
        trailMinutes: 30
      },
      camera: {...HARBOR_VIEW, transitionMs: 1400},
      controls: ['vesselFilter', 'playbackSpeed', 'stopSpeedKnots'],
      readouts: ['stops', 'stopTracks']
    }
  ]
});
