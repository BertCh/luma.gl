// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {joinCredits} from '../../cartography/credits';
import {labelsFor, NYC} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-trails.md?raw';
import type {NycTaxiTrailsOptions} from './nyc-taxi-trails.compute';
import {FLOW_CREDITS} from './flows-style';
import {
  BACKDROP_INK,
  CHORD_INK,
  getFadeDiagram,
  getFareTable,
  HEAD_INK,
  TRAIL_INK
} from './nyc-taxi-trails-style';

/** 07:59:00 local, the time origin of the dataset, in seconds since midnight. */
const ORIGIN_SECONDS_OF_DAY = 7 * 3600 + 59 * 60;
/** The window of the records on the dataset clock: 08:00 to 08:30 is 60 to 1,860 s. */
const FIRST_SECOND = 60;
const LAST_SECOND = 1860;

const formatClock = (seconds: number): string => {
  const total = ORIGIN_SECONDS_OF_DAY + Math.round(seconds);
  return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
};

const MIDTOWN = NYC.places.midtown.lngLat;

/** The camera of the playing steps: Midtown at the scale where trails read as streets. */
const MIDTOWN_VIEW = {
  longitude: MIDTOWN[0],
  latitude: MIDTOWN[1],
  zoom: 12.2,
  pitch: 0,
  bearing: 0
} as const;

const CREDIT = joinCredits(FLOW_CREDITS.nycTaxi, FLOW_CREDITS.osrmRoutes);

/** Place labels of the night ground: the basemap draws none, so every name is ours. */
const PLACES = labelsFor(NYC, ['midtown', 'penn-station', 'grand-central', 'central-park'], {
  midtown: {minZoom: 10},
  'penn-station': {minZoom: 11.5},
  'grand-central': {minZoom: 11.5},
  'central-park': {minZoom: 10}
});

/** The cartouche of a step: line 1 here, the sample line and chips come from the scene. */
const cartouche = (title: string) => ({
  title: {title, subtitle: 'Yellow-cab trips, Fri 2 Jan 2015, 08:00-08:30'}
});

type FareData = {breaks: number[]; extent: [number, number]; counts: number[]};

function getLegends(
  state: NycTaxiTrailsOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const legends: LegendSpec[] = [];
  const fare = data['fare'] as FareData | undefined;
  if (state.colorBy === 'fare' && fare) {
    const match = state.compareRamps ? 'matched' : state.rampMatch;
    legends.push(
      getClassTableLegend(getFareTable({breaks: fare.breaks, extent: fare.extent, match}), {
        title: 'Fare of the trip',
        id: 'fare-classes',
        basis: 'five classes, a fifth of the trips each',
        counts: fare.counts,
        layout: 'list',
        note: state.compareRamps
          ? 'Right of the divider, matched to the dark ground. Left, the same classes run against it.'
          : match === 'matched'
            ? 'Brightest is dearest: the ramp runs toward contrast with the ground.'
            : 'Mismatched: the dearest trips are the darkest and sink into the ground.'
      })
    );
  } else if (state.colorBy === 'heading') {
    legends.push({
      kind: 'cyclic',
      title: 'Heading of travel',
      ramp: 'romao',
      labels: ['N', 'E', 'S', 'W'],
      note: 'The direction each route segment points, clockwise from north. The ramp ends where it starts.'
    });
  } else {
    legends.push({
      kind: 'categories',
      title: 'Taxi trails',
      entries: [
        {color: TRAIL_INK, label: 'Route behind a cab', shape: 'line'},
        {color: HEAD_INK, label: 'Cab now', shape: 'dot'}
      ],
      note: 'A trail fades with age: opacity is (1 - age / trail length) squared.'
    });
  }
  const context = [
    ...(state.showBackdrop
      ? [{color: BACKDROP_INK, label: 'Every route in the sample', shape: 'line' as const}]
      : []),
    ...(state.showChords
      ? [{color: CHORD_INK, label: 'Straight chord, pickup to drop-off', shape: 'line' as const}]
      : [])
  ];
  if (context.length) {
    legends.push({kind: 'categories', title: 'Context', entries: context});
  }
  return legends;
}

function getSnippet(state: NycTaxiTrailsOptions): string {
  const trailSeconds = state.trailMinutes * 60;
  return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTimeWindowFilter} from '@luma.gl/experimental/gpu-dataframe';
import {GPUTrajectoryPlayhead} from '@luma.gl/experimental/gpu-spatial-analysis';

// every taxi at the clock: interpolated position, heading, a compact list of live taxis
const playheadGraph = new GPUCommandGraph(device);
playheadGraph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets, parameters: playhead.importToGraph(playheadGraph),
  currentPositions, headings, speeds, status,
  activeTracks: {ids: activeIds, count: activeCount, overflow: activeOverflow},
  drawInstanceCount                                  // indirect draw count for the heads
}));

// route segments of the last ${state.trailMinutes} min, with fade weights and clip fractions
const trailGraph = new GPUCommandGraph(device);
trailGraph.add(new GPUTimeWindowFilter({
  timestamps: segmentStartTimes, endTimestamps: segmentEndTimes,
  window: window.importToGraph(trailGraph),
  output: {ids: trailIds, count: trailCount, overflow: trailOverflow},
  fadeWeights, clipFractions, trackIds: segmentTracks, trackVisibleCounts, drawInstanceCount
}));

// every frame: write the clock and the window (no recompile), then encode both
playhead.write(getGPUTrajectoryPlayheadParameterValues({playhead: seconds}));
window.write(getGPUTimeWindowParameterValues({
  start: seconds - ${trailSeconds}, end: seconds, startFadeDuration: ${trailSeconds}   // weight 1 at the head, 0 at the tail
}));
playheadCompiled.encode(commandEncoder, {parameters: undefined});
trailCompiled.encode(commandEncoder, {parameters: undefined});

// the layer squares the fade weight: alpha = (1 - age / tail)^2
color.a = color.a * segmentWeights[row] * segmentWeights[row];`;
}

export default defineScene<NycTaxiTrailsOptions>({
  id: 'nyc-taxi-trails',
  title: 'What does a Friday rush hour look like?',
  chapter: 'flows',
  order: 5,
  summary:
    'Nine thousand OSRM-routed taxi trips on a Friday morning: a GPU playhead interpolates every cab, a time-window graph draws fading trails, and the story teaches trail length as memory, routes versus traces, heading as a cycle and ramps matched to the ground.',
  contributors: ['GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'poopdeck-nyc-taxi-paths', role: '9,000 routed taxi trips with a time on every vertex'}
  ],
  initialView: {...MIDTOWN_VIEW},

  options: [
    ...playbackOptions<NycTaxiTrailsOptions>({
      ids: {play: 'play', time: 'time', speed: 'playSpeed', loop: 'loop'},
      time: {
        min: FIRST_SECOND,
        max: LAST_SECOND,
        step: 5,
        default: 480,
        label: 'Clock',
        format: formatClock,
        help: 'Time of the playhead, local New York time on Friday 2 January 2015 (08:00 to 08:30). The playback clock moves this slider.'
      },
      speed: {
        min: 10,
        max: 240,
        step: 10,
        default: 60,
        unit: 'x',
        label: 'Playback speed',
        help: 'Simulated seconds per real second. 60x plays one minute per second; 240x plays the whole window in eight seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Trails',
      apply: 'param',
      min: 0.5,
      max: 10,
      step: 0.5,
      default: 2,
      unit: 'min',
      marks: [{value: 2, label: 'default'}],
      autoSweep: {from: 0.5, to: 10, durationMs: 9000},
      describe: value => `Each cab shows the last ${value} min of its route`,
      help: 'Width of the time window behind the playhead, in minutes of taxi time: a four-number parameter-buffer write. Short trails say where the cabs are; long ones say where they go, until the map becomes a line-density map.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour by',
      group: 'Colour',
      apply: 'param',
      display: 'segmented',
      default: 'none',
      help: 'One taxi yellow; the heading of each route segment on a cyclic ramp; or the fare of the trip in five quantile classes. A per-segment or per-trip buffer read through the segment row, so trails and legend share it.',
      options: [
        {value: 'none', label: 'One colour'},
        {value: 'heading', label: 'Heading'},
        {value: 'fare', label: 'Fare'}
      ]
    },
    {
      kind: 'toggle',
      id: 'compareRamps',
      label: 'Compare ramps',
      group: 'Colour',
      apply: 'param',
      default: false,
      disabledWhen: state => state.colorBy !== 'fare',
      help: 'Draws the fare classes twice with a divider: the ramp run against the dark ground on the left, matched to it on the right.'
    },
    {
      kind: 'select',
      id: 'rampMatch',
      label: 'Ramp',
      group: 'Colour',
      apply: 'param',
      display: 'segmented',
      default: 'matched',
      disabledWhen: state => state.colorBy !== 'fare' || state.compareRamps,
      help: 'Matched: brightest is dearest, toward contrast with the dark ground. Mismatched: the same ramp reversed, so the dearest trips are the darkest.',
      options: [
        {value: 'matched', label: 'Matched'},
        {value: 'mismatched', label: 'Mismatched'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBackdrop',
      label: 'Show every route',
      group: 'Context',
      apply: 'param',
      default: false,
      help: 'Draws all route segments of every trip as a faint network below the trails, so the roads the sample uses are visible.'
    },
    {
      kind: 'toggle',
      id: 'showChords',
      label: 'Chords',
      group: 'Context',
      apply: 'param',
      default: false,
      help: 'Draws, for up to 40 cabs on the road, the straight line from pickup to drop-off (dashed) and the full route that was modelled between them. Pause the clock to freeze the set.'
    },
    {
      kind: 'toggle',
      id: 'inspect',
      label: 'Inspect one taxi',
      group: 'Inspect',
      apply: 'param',
      default: false,
      help: 'Picks a long route among the cabs on the road and draws its whole route in grey with a tick at each vertex of its timestamp table: the table that the GPU binary-searches. The camera flies to it only while the clock is paused.'
    },
    {
      kind: 'button',
      id: 'pickTaxi',
      label: 'Pick another long taxi',
      group: 'Inspect',
      help: 'Inspects the longest route among the other cabs on the road now.'
    }
  ],

  readouts: [
    {id: 'clock', label: 'Local time', help: 'New York local time of the playhead.'},
    {
      id: 'active',
      label: 'Cabs on the road',
      format: 'integer',
      emphasis: 'tile',
      help: 'Trips whose start is before the clock and whose end is after it, from the playhead graph. It counts the sample, about half of the trips of the window.'
    },
    {id: 'trips', label: 'Trips in the sample', format: 'integer'},
    {
      id: 'segments',
      label: 'Trail segments drawn',
      format: 'integer',
      help: 'Route segments inside the trail window, from the time-window graph.'
    },
    {
      id: 'circuity',
      label: 'Median route over chord',
      help: 'Circuity: the length of the modelled route divided by the straight line between its ends, for trips whose ends are more than 200 m apart.'
    },
    {
      id: 'detourShare',
      label: 'Trips with a clear detour',
      help: 'Share of those trips whose route is more than 1.3 times its chord.'
    },
    {
      id: 'dominantHeading',
      label: 'Busiest heading',
      help: 'The share of cabs on the road heading within the busiest 30 degrees of the compass.'
    },
    {
      id: 'searchSteps',
      label: 'Binary search of this cab',
      help: "Steps of the playhead search in this cab's timestamp table: about log2 of its vertex count."
    },
    {id: 'inspectRoute', label: 'Route of this cab'},
    {
      id: 'roadChart',
      label: 'Cabs on the road',
      kind: 'chart',
      help: 'Trips on the road per half minute in the sample. Click to move the clock. The window opens with few cabs already under way, so the first minutes are shaded.'
    },
    {
      id: 'headingRose',
      label: 'Heading of the cabs now',
      kind: 'chart',
      help: 'The headings of the cabs on the road, in ten-degree sectors clockwise from north, read back every few frames; the faint ring is an even spread.'
    },
    {
      id: 'fareHistogram',
      label: 'Fares of the sample',
      kind: 'chart',
      help: 'The fares with the four quintile breaks marked, in the colours of the map.'
    },
    {id: 'vertices', label: 'Route vertices', format: 'integer', hood: true},
    {id: 'segmentsTotal', label: 'Route segments', format: 'integer', hood: true},
    {
      id: 'overflow',
      label: 'List overflow',
      hood: true,
      help: 'Yes means a compact id list was shorter than the number of live taxis or segments.'
    }
  ],

  pipeline: [
    {
      id: 'playhead',
      label: 'Playhead',
      detail: 'One thread per cab: binary-search its timestamps, interpolate position and heading'
    },
    {
      id: 'window',
      label: 'Time window',
      detail: 'One thread per segment: overlap with the trail, fade weight, clip fractions'
    },
    {
      id: 'compact',
      label: 'Compact',
      detail: 'Live cabs and segments gathered into id lists; counts written for the draw'
    },
    {id: 'draw', label: 'Draw', detail: 'The layers read the buffers directly: no CPU geometry'}
  ],

  legends: getLegends,

  basemap: ground('night'),
  furniture: {
    title: {
      title: 'What does a Friday rush hour look like?',
      subtitle: 'Yellow-cab trips, Fri 2 Jan 2015, 08:00-08:30',
      chips: ['Sample', 'Modelled routes']
    },
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat: 'Routes are modelled between pickup and drop-off, not GPS traces.',
    clock: {
      option: 'time',
      // The archive stores New York local time as if UTC: 07:59:00 local is 12:59:00 UTC.
      time: {origin: '2015-01-02T12:59:00Z', unit: 'seconds'},
      zones: ['America/New_York', 'UTC'],
      show: 'time',
      progress: [FIRST_SECOND, LAST_SECOND]
    }
  },
  annotations: PLACES,

  snippet: getSnippet,

  about: {
    what: 'Previously: net flow on a hexagon lattice. Next: the airline network as a graph.\n\n`GPUTrajectoryPlayhead` interpolates every trajectory to one instant in a compute pass and compacts the ids of the live ones; `GPUTimeWindowFilter` selects the route segments inside a time window, weights them by how recent they are and clips the segment at the window edge. The layers read both outputs directly from GPU storage, with the instance counts written by the GPU. Under the hood: the playhead (the mechanics are taught in "Who is moving in New York Harbor?" in the movement chapter).',
    why: 'Replaying movement is the way to see rhythm: where trips concentrate, which corridors carry them, how the picture changes in minutes. Because the clock and the window are small buffers, one compiled graph plays any speed, any trail length and any scrub position. The story reads the same data four ways: as memory (trail length), as provenance (routes, not traces), as a cycle (heading) and as a ramp on a ground.',
    howToRead:
      'Each white dot is a taxi now; the line behind it is the last stretch of its route, fading with age. The routes come from a routing engine between the recorded pickup and drop-off, not from GPS: a cab never queues, and every cab on an avenue is routed along the same line. The data are a seeded half of the yellow-cab trips of a holiday-week Friday in 2015, so counts are lower bounds, and the first minutes undercount because the window opens with few cabs under way. Times are New York local time; the archive stores them as if UTC.'
  },

  create: async ctx => (await import('./nyc-taxi-trails.compute')).createNycTaxiTrails(ctx),

  story: storyFromMarkdown<NycTaxiTrailsOptions>(narrative, {
    'eight-oclock': {
      headline: 'A rush hour, one cab at a time',
      textAlternative:
        'Dark map of Midtown Manhattan with hundreds of white dots, each a taxi, trailing faint yellow lines along the avenues.',
      optionsMode: 'fresh',
      options: {
        play: true,
        time: 480,
        playSpeed: 60,
        trailMinutes: 2,
        colorBy: 'none',
        showBackdrop: false,
        showChords: false,
        inspect: false
      },
      controls: ['play', 'playSpeed'],
      readouts: ['clock', 'active', 'trips', 'roadChart'],
      camera: {...MIDTOWN_VIEW, transitionMs: 1400},
      furniture: cartouche('A rush hour, one cab at a time'),
      stage: 'playhead'
    },
    'trail-is-memory': {
      headline: 'Longer trails turn dots into streets',
      textAlternative:
        'The same map with trails that grow longer: short trails are dots with tails, long trails draw the avenue grid.',
      optionsMode: 'fresh',
      options: {play: true, time: 480, playSpeed: 60, trailMinutes: 2, colorBy: 'none'},
      controls: ['trailMinutes'],
      readouts: ['segments', 'active'],
      camera: {...MIDTOWN_VIEW, transitionMs: 0},
      furniture: cartouche('A trail is the map’s memory'),
      diagram: getFadeDiagram(),
      stage: 'window'
    },
    'routes-not-traces': {
      headline: 'These lines are routes, not GPS',
      textAlternative:
        'Faint grey network of every route, with dashed straight chords and bright full routes for a few dozen taxis frozen at one moment.',
      optionsMode: 'fresh',
      options: {
        play: false,
        time: 900,
        trailMinutes: 1,
        colorBy: 'none',
        showBackdrop: true,
        showChords: true
      },
      controls: ['time', 'showChords'],
      readouts: ['circuity', 'detourShare', 'active'],
      camera: {...MIDTOWN_VIEW, transitionMs: 1400},
      furniture: cartouche('Routes are modelled, not recorded'),
      stage: 'draw'
    },
    heading: {
      headline: 'Avenues move one way, in pairs',
      textAlternative:
        'Midtown trails coloured by compass heading on a cyclic ramp: avenues form opposite pairs of hues, with a rose chart of cab headings showing four spikes.',
      optionsMode: 'fresh',
      options: {
        play: true,
        time: 600,
        playSpeed: 60,
        trailMinutes: 1.5,
        colorBy: 'heading',
        showBackdrop: false
      },
      controls: ['colorBy'],
      readouts: ['dominantHeading', 'headingRose', 'active'],
      camera: {...MIDTOWN_VIEW, zoom: 13.2, transitionMs: 0},
      furniture: cartouche('Direction is a cycle'),
      stage: 'playhead'
    },
    'fare-colour': {
      headline: 'Match the ramp to the ground',
      textAlternative:
        'Trails coloured by fare classes, split by a divider: on the left the ramp runs against the dark ground and the dearest trips fade away, on the right the dearest are brightest.',
      optionsMode: 'fresh',
      options: {
        play: true,
        time: 600,
        playSpeed: 60,
        trailMinutes: 1.5,
        colorBy: 'fare',
        compareRamps: true
      },
      controls: ['compareRamps', 'rampMatch'],
      readouts: ['fareHistogram', 'active'],
      compare: {labels: ['Mismatched ramp', 'Matched ramp'], position: 0.5},
      camera: {...MIDTOWN_VIEW, zoom: 13.2, transitionMs: 0},
      furniture: cartouche('A ramp must face its ground'),
      stage: 'draw'
    },
    'how-the-gpu-does-it': {
      headline: 'Every frame is a binary search per cab',
      textAlternative:
        'One long taxi route drawn in grey over a faint network, with a tick at each vertex of its timestamp table and a ring around its white head.',
      optionsMode: 'fresh',
      options: {
        play: false,
        time: 900,
        playSpeed: 20,
        trailMinutes: 2,
        colorBy: 'none',
        showBackdrop: true,
        inspect: true
      },
      controls: ['pickTaxi', 'trailMinutes', 'colorBy'],
      readouts: ['searchSteps', 'segments', 'overflow'],
      camera: {...MIDTOWN_VIEW, transitionMs: 1400},
      furniture: cartouche('One cab, one binary search'),
      stage: 'playhead'
    }
  })
});
