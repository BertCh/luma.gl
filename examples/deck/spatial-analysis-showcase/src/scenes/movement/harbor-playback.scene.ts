// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NYC, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {joinCredits} from '../../cartography/credits';
import type {MapGround} from '../../cartography/hue-registry';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {VESSEL_CATEGORIES, VESSEL_CATEGORY_LABELS, formatUtcClockSeconds} from './b12-tracks';
import type {HarborPlaybackOptions} from './harbor-playback.compute';
import {getFocusLegend, getStopLegends, getUniformVesselLegend} from './harbor-playback-style';
import {getShipSpeedLegend, getVesselGroupLegend, MOVEMENT_CREDITS} from './movement-style';

const HARBOR_VIEW = {longitude: -74.05, latitude: 40.655, zoom: 10.9, pitch: 0, bearing: 0};

/** The cartouche of one step: the claim, the variable and method (the sample line is set from the data). */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Interpolated positions'] as const
});

/** `[west, south, east, north]` around two gazetteer places, padded by `padding` degrees. */
function getBoundsAround(
  ids: readonly string[],
  padding: number
): [number, number, number, number] {
  const points = ids.map(id => NYC.places[id].lngLat);
  return [
    Math.min(...points.map(point => point[0])) - padding,
    Math.min(...points.map(point => point[1])) - padding,
    Math.max(...points.map(point => point[0])) + padding,
    Math.max(...points.map(point => point[1])) + padding
  ];
}

/** Both slips of the Staten Island Ferry: the lane step 3 frames. */
const FERRY_LANE_BOUNDS = getBoundsAround(['st-george-terminal', 'whitehall-terminal'], 0.012);

/**
 * Place labels from the NYC gazetteer, lowered to the harbour zooms (the gazetteer's own zooms
 * suit city maps, where the terminals and the bridge would be hidden at the harbour frames).
 */
function getHarborLabels(ids: readonly string[]) {
  return labelsFor(NYC, ids, Object.fromEntries(ids.map(id => [id, {minZoom: 10}])));
}

export default defineScene<HarborPlaybackOptions>({
  id: 'harbor-playback',
  title: 'Who is moving in New York Harbor?',
  chapter: 'movement',
  order: 1,
  summary:
    'Replay a day of AIS traffic in New York / New Jersey Harbor on the GPU: every vessel interpolated at the playhead, trails from a sliding time window, tracks resampled to a fixed length and stops found by a rule.',
  contributors: [
    'GPUTrajectoryPlayhead',
    'GPUTimeWindowFilter',
    'GPUTrajectoryResample',
    'GPUTrajectoryMetrics'
  ],
  datasets: [
    {id: 'ais-vessels', role: 'vessel tracks (AIS, one day)'},
    {id: 'ais-zones', role: 'NOAA anchorage outlines (the stops step)'}
  ],
  initialView: HARBOR_VIEW,

  options: [
    ...playbackOptions<HarborPlaybackOptions>({
      ids: {play: 'playing', time: 'time', speed: 'playbackSpeed', loop: 'loop'},
      time: {
        min: 0,
        max: 86400,
        step: 10,
        default: 68400,
        label: 'Time of day (UTC)',
        format: value => `${formatUtcClockSeconds(value)} UTC`,
        help: 'New York was on daylight time, so local time is four hours earlier: 19:00 UTC is 3 pm.'
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
    }).map(option => (option.id === 'loop' ? {...option, expert: true} : option)),
    {
      kind: 'select',
      id: 'markerColor',
      label: 'Colour arrows by',
      group: 'Vessels',
      apply: 'param',
      default: 'uniform',
      display: 'segmented',
      help: 'One amber for every vessel, the four vessel groups (hue says what it is), or the five speed classes (value says how fast). The shape of the mark is a third variable: see the squares switch.',
      options: [
        {value: 'uniform', label: 'One colour'},
        {value: 'group', label: 'Group'},
        {value: 'speed', label: 'Speed'}
      ]
    },
    {
      kind: 'toggle',
      id: 'stoppedSquares',
      label: 'Squares for stopped vessels',
      group: 'Vessels',
      apply: 'param',
      default: false,
      help: 'Vessels slower than the stop speed are drawn as small squares instead of heading arrows, the chart-plotter convention: squares are stopped, arrows are moving. The stop speed is the threshold in the Stops group.'
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every track faintly',
      group: 'Vessels',
      apply: 'param',
      default: true,
      help: 'Draws the whole day of every track as thin, additive lines, so the shipping lanes glow where many vessels went.'
    },
    {
      kind: 'select',
      id: 'vesselFilter',
      label: 'Show vessel type',
      group: 'Vessels',
      apply: 'param',
      default: 'all',
      expert: true,
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
      expert: true,
      help: 'Base half-length of an arrow. The map draws it one pixel smaller below zoom 11.5 and one larger above, so marks keep their weight as the scale changes.'
    },
    {
      kind: 'toggle',
      id: 'labelFerry',
      label: 'Label a ferry',
      group: 'Vessels',
      apply: 'param',
      default: false,
      expert: true,
      help: 'Follows the fastest Staten Island Ferry with a live note: its name on the map and its speed from the same interpolation that places the arrow.'
    },
    {
      kind: 'slider',
      id: 'maxGapMinutes',
      label: 'Maximum fix gap',
      group: 'Between the fixes',
      apply: 'param',
      min: 0,
      max: 20,
      step: 1,
      default: 0,
      unit: 'min',
      format: value => (value === 0 ? 'off' : `${value} min`),
      help: 'When the two fixes either side of the playhead are further apart than this, the vessel is flagged as in a gap and hidden instead of drawn at a guessed position. 0 turns the test off. Moored vessels were thinned to one fix every few minutes by our preprocessing, so a limit under that hides them for a reason that is ours, not AIS silence.'
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Focus on one vessel',
      group: 'Between the fixes',
      apply: 'param',
      default: 'none',
      help: 'Draws the raw fixes of one vessel above the rest, which fade back: a Staten Island ferry mid-crossing (fixes, chord and fraction at the playhead), or one long passage that moves, waits and moves on (every fix and its resampled points).',
      options: [
        {value: 'none', label: 'No one'},
        {value: 'ferry', label: 'A ferry mid-crossing'},
        {value: 'transit', label: 'A passage with a wait'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Trails (time window)',
      apply: 'param',
      default: false,
      help: 'Draws the part of every track inside the sliding window behind the playhead, coloured by the speed of each step. Which segments are live is decided on the GPU by GPUTimeWindowFilter.'
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
      default: 30,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Window width. It is written into the window parameter buffer: [playhead - length, playhead]. A fast vessel covers more distance in the same time, so its trail is longer.'
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
      id: 'routeSpacing',
      label: 'Sample spacing',
      group: 'Resampled routes',
      apply: 'compile',
      default: 'arc-length',
      display: 'segmented',
      help: 'Compile-time. Arc length places samples at equal distances along the path; time places them at equal time steps, so the stretches where a vessel waits collect many samples.',
      options: [
        {value: 'arc-length', label: 'Equal distance'},
        {value: 'time', label: 'Equal time'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'All routes',
      group: 'Resampled routes',
      apply: 'param',
      default: false,
      help: 'Draws every track rebuilt as the same number of evenly spaced samples, in one neutral ink: the fingerprint of the harbour.'
    },
    {
      kind: 'select',
      id: 'routeSamples',
      label: 'Samples per route',
      group: 'Resampled routes',
      apply: 'compile',
      default: '32',
      expert: true,
      help: 'Compile-time: the dense output table has tracks x samples rows, so changing it rebuilds the resample graph.',
      options: [
        {value: '16', label: '16 samples'},
        {value: '32', label: '32 samples'},
        {value: '64', label: '64 samples'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showStops',
      label: 'Show stops',
      group: 'Stops',
      apply: 'param',
      default: false,
      help: 'Marks every place a vessel dwelled: a run of slow steps lasting at least the minimum duration. Radius grows with the dwell and colour says its class.'
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
      help: 'A step is slow when the vessel covers less than this speed times the step time. Parameter buffer: changing it re-runs the graph, never recompiles. It also decides which vessels are drawn as squares.'
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
    }
  ],

  readouts: [
    {
      id: 'active',
      label: 'Vessels with a position now',
      format: 'integer',
      emphasis: 'tile',
      help: 'Tracks whose first and last fix bracket the playhead (and pass the gap test): the arrows drawn.'
    },
    {
      id: 'busiestHour',
      label: 'Busiest hour (local time)',
      help: 'The hour of the day in which the most tracks report at least one fix.'
    },
    {
      id: 'activityChart',
      label: 'Tracks reporting through the day',
      kind: 'chart',
      help: 'Tracks with at least one AIS fix in each 15-minute bin. The vertical rule follows the playhead, and the reporting silence remains visible as a break in coverage.'
    },
    {
      id: 'silence',
      label: 'Longest silence of the whole feed',
      help: 'The longest stretch in which no vessel at all reported a fix.'
    },
    {
      id: 'stoppedNow',
      label: 'Vessels under the stop speed',
      format: 'integer',
      emphasis: 'tile',
      help: 'Active vessels slower than the stop speed threshold: the squares.'
    },
    {id: 'stoppedShare', label: 'Share not underway', help: 'Stopped vessels over active ones.'},
    {
      id: 'speedChart',
      label: 'How fast do tracks go?',
      kind: 'chart',
      help: 'Mean speed of every track from GPUTrajectoryMetrics, in the same five classes as the map. The tall first bar is vessels tied up all day.'
    },
    {
      id: 'fixDiagram',
      label: 'Two fixes and a fraction',
      kind: 'chart',
      help: 'The two fixes either side of the playhead for the vessel in focus, and how far along the chord the arrow sits.'
    },
    {
      id: 'fraction',
      label: 'Fraction along the chord',
      hood: true,
      help: 'Time since fix A over the time between fix A and fix B: what the binary search and the linear interpolation return for this vessel.'
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
      hood: true,
      help: 'Segments inside the time window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'keptShare',
      label: 'Segments kept this frame',
      help: 'Live trail segments over all segments of the day.'
    },
    {
      id: 'sampleSpacing',
      label: 'What the spacing did',
      help: 'For equal distance, the gap between samples along the route; for equal time, the most samples within 150 m of one spot.'
    },
    {id: 'resampled', label: 'Resampled table', hood: true},
    {id: 'stops', label: 'Stops detected', help: 'Number of stops at the current thresholds.'},
    {
      id: 'longestStop',
      label: 'Longest stop',
      help: 'Longest dwell in the stop list; vessels moored all day reach the end of the record.'
    },
    {
      id: 'dwellChart',
      label: 'How long are the stops?',
      kind: 'chart',
      help: 'Duration of every stop at the current thresholds, coloured by the four dwell classes of the map.'
    },
    {
      id: 'stopSweep',
      label: 'Stops found at each speed threshold',
      kind: 'chart',
      help: 'The same compiled graph run once per threshold with a parameter write each. Click the chart to set the threshold.'
    },
    {
      id: 'anchoredShare',
      label: 'Stops inside an anchorage',
      help: 'Share of stops whose centre lies inside a NOAA anchorage outline.'
    },
    {
      id: 'stopShare',
      label: 'Stops by tugs and ferries',
      help: 'Share of stops made by tug and passenger vessels (the group of each stopped track).'
    },
    {
      id: 'sweepSpread',
      label: 'Spread of the stop count over the sweep',
      hood: true,
      help: 'Highest minus lowest stop count over the swept thresholds, over the highest.'
    },
    {
      id: 'numerics',
      label: 'Numerical notes',
      hood: true,
      help: 'What the interpolation assumes and how exact it is.'
    },
    {id: 'fixCadence', label: 'Typical fix interval', hood: true},
    {id: 'mooredCadence', label: 'Fix interval while moored', hood: true},
    {id: 'beforeStart', label: 'Not yet reporting', format: 'integer', hood: true},
    {id: 'afterEnd', label: 'Already gone', format: 'integer', hood: true},
    {id: 'tracks', label: 'Tracks', hood: true},
    {id: 'vertices', label: 'Fixes', hood: true},
    {id: 'distance', label: 'Distance sailed', hood: true},
    {id: 'fastest', label: 'Fastest step', hood: true},
    {
      id: 'selected',
      label: 'Selected vessel',
      help: 'Click a vessel to read its type, MMSI, length, distance and speeds.'
    }
  ],

  pipeline: [
    {
      id: 'playhead',
      label: 'Playhead',
      detail: 'One thread per track: a binary search, then a lerp between two fixes'
    },
    {
      id: 'window',
      label: 'Time window',
      detail: 'Segments overlapping the window are fading, clipped and compacted into the draw call'
    },
    {id: 'metrics', label: 'Metrics', detail: 'Step speeds per track, and the stop rule'},
    {
      id: 'resample',
      label: 'Resample',
      detail: 'Every track rebuilt as the same number of points, by distance or by time'
    }
  ],

  timeline: {time: 'time', play: 'playing', speed: 'playbackSpeed'},

  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: cartouche(
      'Who is moving in New York Harbor?',
      'Interpolated AIS positions · 12 June 2024'
    ),
    scaleBar: {units: 'nautical'},
    credit: joinCredits(MOVEMENT_CREDITS.harborAis),
    clock: {
      option: 'time',
      time: {origin: '2024-06-12T00:00:00Z', unit: 'seconds'},
      zones: ['America/New_York', 'UTC'],
      progress: [0, 86400]
    }
  },

  legends: (state, data) => {
    const ground = ((data['ground'] as MapGround | undefined) ?? 'dark') as MapGround;
    if (state.showStops) {
      return getStopLegends(ground, data['stopClassCounts'] as readonly number[] | undefined);
    }
    const legends: LegendSpec[] = [];
    if (state.focus !== 'none') {
      legends.push(getFocusLegend(state.focus, ground));
      return legends;
    }
    if (state.markerColor === 'group') {
      legends.push(
        getVesselGroupLegend(ground, data['groupCounts'] as readonly number[] | undefined)
      );
    } else if (state.markerColor === 'speed') {
      legends.push(getShipSpeedLegend(ground));
    } else {
      legends.push(getUniformVesselLegend(ground));
    }
    if (state.showTrails && state.markerColor !== 'speed') {
      legends.push(getShipSpeedLegend(ground, 'Trails take the colour of each step.'));
    }
    return legends;
  },

  snippet: state => {
    const stopSpeed = ((state.stopSpeedKnots * 1852) / 3600).toFixed(2);
    const resampled = state.focus === 'transit' || state.showRoutes;
    return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
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
${
  state.showTrails
    ? `
const trailGraph = new GPUCommandGraph(device, {id: 'trails'});
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,   // interval mode: one row per segment
  window: windowParameters.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, drawInstanceCount: trailDrawCount
}));
`
    : ''
}${
  state.showStops
    ? `
const stopGraph = new GPUCommandGraph(device, {id: 'stops'});
stopGraph.add(new GPUTrajectoryMetrics({spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
  positions, timestamps, trackOffsets,
  parameters: stopParameters.importToGraph(stopGraph),
  stepSpeeds, averageSpeeds, maximumSpeeds, trackStopCounts,
  stops: {output: {ids, count, overflow}, centroids, durations, drawInstanceCount: stopDrawCount}
}));
`
    : ''
}${
  resampled
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
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({
  playhead, maxGap: ${state.maxGapMinutes * 60}${state.maxGapMinutes === 0 ? '   // 0 turns the gap test off' : ''}
}));${
      state.showTrails
        ? `
windowParameters.write(getGPUTimeWindowParameterValues({
  start: playhead - ${state.trailMinutes * 60}, end: playhead, startFadeDuration: ${Math.round(state.trailMinutes * 60 * state.tailFade)}
}));`
        : ''
    }${
      state.showStops
        ? `
stopParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${stopSpeed},   // m/s (${state.stopSpeedKnots} kn)
  stopMinimumDuration: ${state.stopMinutes * 60}                              // s
}));`
        : ''
    }
play.encode(commandEncoder, {parameters: undefined});

// the arrows read the GPU's own outputs
new VesselMarkerLayer({
  ids: activeIds, positions: currentPositions, headings, speeds, drawCommands: markerDraw,
  colorMode: '${state.markerColor === 'speed' ? 'speedClasses' : 'category'}',${
    state.markerColor === 'speed'
      ? '\n  speedClassBreaks: [0.5, 3, 8, 15].map(knots => knots * 1852 / 3600),'
      : ''
  }${state.stoppedSquares ? `\n  stoppedSpeed: ${stopSpeed},   // slower vessels draw as squares` : ''}
});`;
  },

  about: {
    what: 'Previously: this is where the movement chapter starts, with ships and one day. Next: [Which vessels meet, and do they share a route?](#/story/vessel-encounters).\n\n`GPUTrajectoryPlayhead` finds, for every vessel at once, the two AIS fixes either side of the clock (one thread per track, a binary search) and linearly interpolates the position, heading and speed. `GPUTimeWindowFilter` keeps the track segments that overlap a sliding window and fades them toward the tail, then compacts the live ids straight into an indirect draw. `GPUTrajectoryMetrics` measures each track and applies the stop rule, and `GPUTrajectoryResample` rebuilds each track with a fixed number of samples.',
    why: 'A harbour is a moving system: pilots, port authorities and researchers ask who is where at a given time, what a vessel did in the last half hour, and where things wait. Doing the interpolation, windowing and stop detection on the GPU keeps the whole day of tracks live as you scrub, with parameter writes instead of rebuilds. The map is also a lesson in visual variables (Bertin): hue for kinds, value for amounts, shape for state.',
    howToRead:
      'Arrows are vessels now, pointing the way they steer; squares are vessels below the stop speed. A trail is a span of time, not a distance, so a longer trail means a faster vessel. Discs are stops: area grows with the dwell and colour says its class. AIS reports positions about once a minute and thins moored vessels, so everything between fixes is a straight-line estimate, and the whole feed has one silent stretch in the afternoon. A stop is a rule, not an observation: the rule is taught in [Where do migrating raptors stop, and for how long?](#/story/migration-stopovers).'
  },

  create: async ctx => (await import('./harbor-playback.compute')).createHarborPlayback(ctx),

  story: [
    {
      id: 'the-question',
      title: 'A Wednesday afternoon in the harbour',
      headline: 'Ferries keep shuttling across the Upper Bay',
      textAlternative:
        'Dark map of New York Harbor with faint grey shipping lanes and many small amber arrows, the Staten Island ferries moving between St. George and Whitehall.',
      body: 'Each arrow is a vessel with a position now, **{{active}}** of them, pointing the way it moves. AIS reports about every {{fixCadence}}, so between fixes an arrow is an estimate. Press **Play** and one clock drives every track at once; drag **Time of day (UTC)** to jump. The busiest hour starts at {{busiestHour}}. Later no vessel reports for {{silence}}.',
      evidence:
        '**{{active}}** vessels have an interpolated position now; the full-day activity chart peaks at **{{busiestHour}}**.',
      caveat:
        'The feed contains a shared reporting gap of **{{silence}}**; a quiet map then is missing coverage, not an empty harbour.',
      optionsMode: 'fresh',
      options: {
        time: 68400,
        playing: true,
        playbackSpeed: 240,
        markerColor: 'uniform',
        labelFerry: true
      },
      controls: ['playing', 'time'],
      readouts: ['active', 'activityChart', 'busiestHour', 'silence'],
      stage: 'playhead',
      camera: {...HARBOR_VIEW, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Who is moving in New York Harbor?',
          'Interpolated AIS positions · 12 June 2024'
        )
      },
      annotations: getHarborLabels([
        'upper-bay',
        'lower-bay',
        'verrazzano',
        'kill-van-kull',
        'staten-island',
        'manhattan',
        'brooklyn',
        'statue-of-liberty',
        'st-george-terminal',
        'whitehall-terminal',
        'port-newark'
      ])
    },
    {
      id: 'what-and-how-fast',
      title: 'Hue says what, value says how fast',
      headline: 'Most vessels here are tied up, not underway',
      textAlternative:
        'The Upper Bay with vessels as coloured arrows and squares; most marks are small squares at the piers, with a few arrows crossing the water.',
      body: 'Set **Colour arrows by** to *Speed*: value says how fast, in the speed classes of the histogram. *Group* gives hue instead: what a vessel is. Shape adds a third variable: squares are slower than the stop speed, arrows are underway. **{{stoppedNow}}** of {{active}} vessels here are squares, {{stoppedShare}}. Drag **Time of day (UTC)** to see it change. *Hue for kinds, value for amounts, shape for state.*',
      evidence:
        '**{{stoppedNow}}** of **{{active}}** active vessels are below the current stop-speed threshold; the histogram gives the distribution behind the classes.',
      caveat:
        '“Stopped” is a thresholded state, so the same **{{active}}** vessels can divide differently when the stop-speed rule changes.',
      optionsMode: 'fresh',
      options: {
        time: 68400,
        playing: false,
        markerColor: 'group',
        stoppedSquares: true
      },
      controls: ['markerColor', 'time'],
      readouts: ['stoppedNow', 'active', 'speedChart'],
      stage: 'playhead',
      camera: {
        longitude: NYC.places['upper-bay'].lngLat[0],
        latitude: NYC.places['upper-bay'].lngLat[1],
        zoom: 12,
        transitionMs: 1600
      },
      furniture: {
        title: cartouche(
          'Hue says what, value says how fast',
          'Vessel group · speed over ground (kn) · interpolated at the playhead'
        )
      },
      annotations: getHarborLabels([
        'upper-bay',
        'statue-of-liberty',
        'governors-island',
        'st-george-terminal',
        'whitehall-terminal',
        'red-hook-terminal'
      ])
    },
    {
      id: 'between-the-fixes',
      title: 'Between two fixes',
      headline: 'The arrow is a guess between two fixes',
      textAlternative:
        'A Staten Island ferry mid-crossing: hollow rings for its AIS fixes, two filled rings either side of the arrow, and a straight chord between them.',
      body: 'Rings are real AIS fixes; the filled pair brackets the playhead. The GPU finds it by binary search and puts the arrow **{{fraction}}** of the way along the chord. Drag **Time of day (UTC)** and watch it run. Moored vessels were thinned to one fix per {{mooredCadence}} by our preprocessing: raise **Maximum fix gap** and they vanish for a reason that is ours, not AIS silence.',
      evidence:
        'For the highlighted ferry, the live playhead lies **{{fraction}}** of the way between the two filled fixes; **{{inGap}}** tracks are currently rejected as gaps.',
      caveat:
        'A gap count of **{{inGap}}** depends on the chosen maximum fix gap, and moored reports were already thinned before this analysis.',
      optionsMode: 'fresh',
      options: {
        time: 68870,
        playing: false,
        markerColor: 'uniform',
        stoppedSquares: true,
        focus: 'ferry'
      },
      controls: ['time', 'maxGapMinutes'],
      readouts: ['fixDiagram', 'fraction', 'inGap', 'active'],
      stage: 'playhead',
      camera: {bounds: FERRY_LANE_BOUNDS, transitionMs: 1600},
      furniture: {
        title: cartouche(
          'An arrow is a guess between two fixes',
          'Linear interpolation of AIS fixes · one vessel'
        )
      },
      annotations: getHarborLabels(['st-george-terminal', 'whitehall-terminal'])
    },
    {
      id: 'trails-are-time',
      title: 'Trails are time',
      headline: 'Every trail spans the same time; length is speed',
      textAlternative:
        'The harbour with fading trails behind each vessel, long bright streaks behind ferries and short stubs behind tugs, coloured by speed class.',
      body: 'A trail keeps the segments whose time interval overlaps a window behind the playhead, fading toward the tail. The GPU kept **{{keptShare}}** of all segments this frame and wrote the count into the draw call, so drawing needs no readback. Change **Trail length** and every streak stretches at once; set **Tail fade** to none and the fade goes. Colours are the speed classes of the arrows.',
      evidence:
        'The current window contains **{{trailSegments}}** route segments, **{{keptShare}}** of the day’s segment table.',
      caveat:
        '**{{trailSegments}}** is a segment count, not a vessel count; fast vessels contribute longer-looking trails over the same time window.',
      optionsMode: 'fresh',
      options: {
        time: 68400,
        playing: true,
        playbackSpeed: 120,
        markerColor: 'speed',
        stoppedSquares: true,
        showTrails: true,
        trailMinutes: 30,
        tailFade: 1
      },
      controls: ['trailMinutes', 'tailFade'],
      readouts: ['keptShare', 'trailSegments'],
      stage: 'window',
      camera: {longitude: -74.05, latitude: 40.665, zoom: 11.3, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'A trail is time, not distance',
          'Speed over ground (kn) · segments in a sliding time window'
        )
      },
      annotations: getHarborLabels([
        'upper-bay',
        'kill-van-kull',
        'port-newark',
        'verrazzano',
        'staten-island'
      ])
    },
    {
      id: 'one-route-many-samples',
      title: 'Equal distance or equal time',
      headline: 'Time spacing piles samples where vessels wait',
      textAlternative:
        'One vessel passage with hollow rings for its fixes and orange dots for the resampled points, spread evenly along the route or piled where it waited.',
      body: 'Many analyses need routes of equal length, so resampling rebuilds a track as a fixed number of points. Rings are real fixes, dots are the new samples. Switch **Sample spacing** and read what the GPU did: {{sampleSpacing}}. Spacing is a compile-time option. Then turn on **All routes**: the fingerprint of the harbour, every track with the same number of points.',
      evidence:
        'The selected route is rebuilt as **{{resampled}}**; under the current rule that means **{{sampleSpacing}}**.',
      caveat:
        '**{{resampled}}** standardises row length, not information content: time spacing can pile samples into a wait while distance spacing smooths it away.',
      optionsMode: 'fresh',
      options: {
        time: 68400,
        playing: false,
        markerColor: 'uniform',
        stoppedSquares: true,
        focus: 'transit',
        routeSpacing: 'arc-length',
        showRoutes: false
      },
      controls: ['routeSpacing', 'showRoutes'],
      readouts: ['sampleSpacing', 'resampled'],
      stage: 'resample',
      furniture: {
        title: cartouche(
          'Equal distance or equal time?',
          'Tracks resampled to fixed-length routes · spacing by arc length or by time'
        )
      }
    },
    {
      id: 'where-vessels-stop',
      title: 'Where vessels stop',
      headline: 'Most stops happen outside the official anchorages',
      textAlternative:
        'Harbour map with thin outlines of the official anchorages and discs for stops, sized and coloured by how long each vessel stayed; most discs sit at slips and yards, few inside the outlines.',
      body: 'A stop is a rule: slower than **Stop speed threshold** for at least **Minimum stop duration**. Only **{{anchoredShare}}** of the {{stops}} stops fall inside an official anchorage (the hairlines); tugs and passenger boats make {{stopShare}} of them. Across the sweep of thresholds the count moves by {{sweepSpread}}: try the duration. *A stop is a rule, not an observation.* Next: [which vessels meet?](#/story/vessel-encounters)',
      evidence:
        'Of **{{stops}}** detected stops, **{{anchoredShare}}** fall in official anchorages; the threshold-sweep chart shows a **{{sweepSpread}}** spread in the count.',
      caveat:
        'The longest detected stop is **{{longestStop}}**, but every stop—including that one—is conditional on the speed and duration rules.',
      optionsMode: 'fresh',
      options: {
        time: 68400,
        playing: false,
        markerColor: 'uniform',
        stoppedSquares: true,
        showStops: true,
        stopSpeedKnots: 0.5,
        stopMinutes: 15
      },
      controls: ['stopSpeedKnots', 'stopMinutes'],
      readouts: ['stops', 'dwellChart', 'stopSweep', 'longestStop'],
      stage: 'metrics',
      camera: {longitude: -74.06, latitude: 40.665, zoom: 11.2, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Most stops are not at anchorages',
          'Stops: slower than a threshold for a minimum time · disc radius = dwell'
        ),
        credit: joinCredits(MOVEMENT_CREDITS.harborAis, MOVEMENT_CREDITS.harborZones)
      },
      annotations: getHarborLabels([
        'upper-bay',
        'kill-van-kull',
        'st-george-terminal',
        'whitehall-terminal',
        'port-newark'
      ])
    }
  ]
});
