// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, NYC} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatCount} from '../../cartography/live-text';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-tides.md?raw';
import type {NycTaxiTidesOptions} from './nyc-taxi-tides.compute';
import {formatTaxiTime} from './nyc-taxi-data';
import {FLOW_CREDITS, FLOW_INK, getFlowWidthLegend, NET_FLOW_WORDS} from './flows-style';
import {getMagnitudeTable, getNetTable, getShareTable, NET_STEPS} from './nyc-taxi-tides-style';

/** Mirrors the compute module without importing it (scene files stay light). */
const FIRST_HOUR = 1;
const LAST_HOUR = 37.5;
/** Width of the heaviest and lightest flow arrow, as the compute module draws them. */
const FLOW_MAX_PIXELS = 3.5;
const FLOW_MIN_PIXELS = 1.2;

/** Manhattan and its edges: the camera of the signed-flow steps. */
const MANHATTAN_BOUNDS = [-74.03, 40.69, -73.9, 40.83] as const;
/** Manhattan, LaGuardia and JFK in one frame. */
const AIRPORT_BOUNDS = [-74.03, 40.61, -73.74, 40.82] as const;

const CREDIT = joinCredits(FLOW_CREDITS.nycTaxi, FLOW_CREDITS.osrmRoutes, CREDITS.colorBrewer);

/** Place labels shared by the Manhattan steps; names show from a lower zoom than the default. */
const MANHATTAN_LABELS = labelsFor(NYC, ['midtown', 'central-park', 'lower-manhattan'], {
  midtown: {minZoom: 10},
  'central-park': {minZoom: 10},
  'lower-manhattan': {minZoom: 10}
});

/** The cartouche of a step: line 1 here, the live subtitle and sample line come from the scene. */
const cartouche = (title: string) => ({title: {title, chips: ['Sample'] as const}});

type TideData = Readonly<Record<string, unknown>>;

function getLegends(state: NycTaxiTidesOptions, data: TideData): LegendSpec[] {
  const tide = (data['ground'] as 'light' | 'dark' | undefined) ?? 'light';
  const counts = data['counts'] as {net: number[]; share: number[]} | undefined;
  const noun = state.zones === 'hexagon' ? 'hexagon' : 'cell';
  const legends: LegendSpec[] = [];
  if (state.normalise === 'share') {
    legends.push(
      getClassTableLegend(getShareTable(tide, state.minVolume), {
        title: `Imbalance share per ${noun}`,
        id: 'tide-classes',
        basis: 'of arrivals plus departures',
        counts: counts?.share,
        interactive: true,
        layout: 'list',
        note: `${NET_FLOW_WORDS.low} (orange) to ${NET_FLOW_WORDS.high} (purple). Hatched: too few trips to call.`
      })
    );
  } else {
    legends.push(
      getClassTableLegend(getNetTable(tide), {
        title: `Net taxi arrivals per ${noun}`,
        id: 'tide-classes',
        basis: 'arrivals minus departures',
        counts: counts?.net,
        interactive: true,
        layout: 'list',
        note: `${NET_FLOW_WORDS.low} (orange) to ${NET_FLOW_WORDS.high} (purple); ends clipped at ${NET_STEPS[3]} trips. The classes are the same for every hour.`
      })
    );
    if (state.compareMidpoint) {
      legends.push(
        getClassTableLegend(getMagnitudeTable(tide), {
          title: 'Size of the net, direction lost',
          id: 'magnitude-classes',
          basis: 'absolute arrivals minus departures',
          layout: 'list',
          note: 'The wrong map: one ramp cannot say arrive or leave.'
        })
      );
    }
  }
  if (state.flowCount > 0) {
    const widths = getFlowWidthLegend({
      title: 'Flow between hexagons',
      maxValue: Math.max(1, (data['flowMaximum'] as number | undefined) ?? 1),
      maxWidthPixels: FLOW_MAX_PIXELS,
      color: FLOW_INK[tide],
      format: value => `${formatCount(value)} trips`
    });
    if (widths.kind === 'line') {
      legends.push({
        ...widths,
        entries: widths.entries.map(entry => ({
          ...entry,
          widthPixels: Math.max(FLOW_MIN_PIXELS, entry.widthPixels)
        }))
      });
    }
  }
  return legends;
}

function getSnippet(state: NycTaxiTidesOptions): string {
  return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';

// departures gate on the pickup time, arrivals on the dropoff time: same trips, two graphs
const make = (gateTimes) => {
  const graph = new GPUCommandGraph(device);
  graph.add(new GPUFlowAggregation({
    zones: {kind: '${state.zones}', bounds, gridSize: capacity,   // capacity fixed for 400 m
            activeGridSize, ${state.zones === 'hexagon' ? 'radius, ' : ''}},   // ${state.zoneSize} m: a parameter write
    origins, destinations,                         // float32x2 meters, uploaded once
    weights,                                       // one per trip
    timeWindow: {timestamps: gateTimes, window},   // four numbers, rewritten every frame
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: 524288,
    output: {ids, count, overflow, requiredCount},
    zoneOutCounts, zoneInCounts,
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights   // top-K flows
  }));
  return graph.compile();
};
const departures = make(pickupHours);
const arrivals = make(dropoffHours);

// every frame: move the window (no recompile), run both graphs
window.write(getGPUTimeWindowParameterValues({start: hour, end: hour + ${state.windowHours}}));
departures.encode(commandEncoder, {parameters: undefined});
arrivals.encode(commandEncoder, {parameters: undefined});

// after the readback (a few KB): net per zone on the CPU, then one buffer write
net[zone] = inCounts[zone] - outCounts[zone];
share[zone] = volume < ${state.minVolume} ? MASKED : net[zone] / volume;   // volume = in + out`;
}

export default defineScene<NycTaxiTidesOptions>({
  id: 'nyc-taxi-tides',
  title: 'Where does Manhattan fill up and empty?',
  chapter: 'flows',
  order: 4,
  summary:
    'Net taxi arrivals minus departures on a hexagon lattice by time window: two GPUFlowAggregation graphs, a diverging scale with a real middle, shares instead of counts, and a hexagon size that changes the answer.',
  contributors: ['GPUFlowAggregation'],
  datasets: [
    {id: 'poopdeck-nyc-taxi', role: '440,000 yellow-taxi origin-destination pairs with times'}
  ],
  initialView: {...CITY_FRAMES.nyc},

  options: [
    ...playbackOptions<NycTaxiTidesOptions>({
      ids: {play: 'play', time: 'time', speed: 'playSpeed', loop: 'loop'},
      time: {
        min: FIRST_HOUR,
        max: LAST_HOUR,
        step: 0.25,
        default: 32,
        label: 'Window start',
        format: formatTaxiTime,
        help: 'Start of the time window, in local New York time since midnight on Thursday 1 January 2015. The default is Friday 2 January, 08:00. The playback clock moves this slider.'
      },
      speed: {
        min: 0.25,
        max: 4,
        step: 0.25,
        default: 1,
        unit: 'h/s',
        label: 'Playback speed',
        help: 'Hours of taxi time that pass per real second. At 1 h/s the 37 hours take about 37 seconds.'
      },
      playing: false,
      loop: true
    }),
    {
      kind: 'slider',
      id: 'windowHours',
      label: 'Window length',
      group: 'Window',
      apply: 'param',
      min: 0.25,
      max: 6,
      step: 0.25,
      default: 1,
      unit: 'h',
      help: 'Length of the time window. Departures are trips picked up in the window, arrivals trips dropped off in it. A longer window averages more trips: the time window is a modifiable unit too.'
    },
    {
      kind: 'select',
      id: 'normalise',
      label: 'Show',
      group: 'Map',
      apply: 'param',
      display: 'segmented',
      default: 'net',
      help: 'Net trips (arrivals minus departures) follow the volume of traffic. The imbalance share, net over arrivals plus departures, compares places of any size.',
      options: [
        {value: 'net', label: 'Net trips'},
        {value: 'share', label: 'Imbalance share'}
      ]
    },
    {
      kind: 'slider',
      id: 'minVolume',
      label: 'Minimum trips',
      group: 'Map',
      apply: 'param',
      min: 0,
      max: 60,
      step: 1,
      default: 20,
      unit: 'trips',
      marks: [{value: 20, label: 'default'}],
      danger: [0, 9],
      disabledWhen: state => state.normalise !== 'share',
      describe: value =>
        value < 10
          ? 'Too low: a few trips make a share swing to the extremes'
          : `Hexagons with fewer than ${value} trips are hatched`,
      help: 'A share of a handful of trips is noise: three arrivals and no departure is 100 percent. Hexagons under this volume (arrivals plus departures) are hatched instead of coloured.'
    },
    {
      kind: 'toggle',
      id: 'compareMidpoint',
      label: 'Compare with a one-ramp map',
      group: 'Map',
      apply: 'param',
      default: false,
      disabledWhen: state => state.normalise !== 'net',
      help: 'Draws the size of the net on a sequential ramp left of a divider and the diverging net right of it, so the direction that the sequential ramp loses is visible.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Hexagon hairlines',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Thin lines between hexagons that carry trips, so single cells stay countable. Turn off to see the fill alone.'
    },
    {
      kind: 'preset',
      id: 'hexSizes',
      label: 'Hexagon size',
      group: 'Lattice',
      presets: [
        {label: '400 m', values: {zoneSize: 400}},
        {label: '800 m', values: {zoneSize: 800}},
        {label: '1,600 m', values: {zoneSize: 1600}},
        {label: '3 km', values: {zoneSize: 3000}}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneSize',
      label: 'Zone size',
      group: 'Lattice',
      apply: 'param',
      min: 400,
      max: 3000,
      step: 100,
      default: 800,
      unit: 'm',
      marks: [{value: 800, label: 'default'}],
      help: 'Hexagon radius or square cell size. The lattice capacity is fixed for 400 m; the radius and the active lattice are per-frame values, so this never recompiles.'
    },
    {
      kind: 'toggle',
      id: 'showRadius',
      label: 'Show one hexagon radius',
      group: 'Lattice',
      apply: 'param',
      default: false,
      help: 'A dashed ring of one zone radius around the biggest gain, with the scale-bar tick at the same length.'
    },
    {
      kind: 'slider',
      id: 'flowCount',
      label: 'Flows drawn',
      group: 'Flows',
      apply: 'param',
      min: 0,
      max: 60,
      step: 2,
      default: 0,
      help: 'Heaviest departure flows of the window, drawn from the top-256 list the contributor writes and reads back. Width is the square root of the trips, one scale for every window of a zone size.'
    },
    {
      kind: 'select',
      id: 'zones',
      label: 'Zones',
      group: 'Under the hood',
      apply: 'compile',
      display: 'segmented',
      default: 'hexagon',
      expert: true,
      help: 'A hexagon lattice or a square grid assigns each trip end to a zone. Hexagons have six equidistant neighbours, so adjacent cells compare fairly; squares do not. The zone kind is compile-time; the zone size is not.',
      options: [
        {value: 'hexagon', label: 'Hexagons'},
        {value: 'grid', label: 'Squares'}
      ]
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-zone trips',
      group: 'Under the hood',
      apply: 'compile',
      default: true,
      expert: true,
      help: 'Rejects trips whose pickup and dropoff fall in one zone. The net balance is unchanged (such trips cancel); the flows and the pair count lose the local trips.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Summation order',
      group: 'Under the hood',
      apply: 'compile',
      display: 'segmented',
      default: 'sorted',
      expert: true,
      help: '`sorted` accumulates weights in a fixed tree, bitwise identical on every run. `atomic` adds with compare-exchange and is faster only when almost every row has its own pair.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    }
  ],

  readouts: [
    {id: 'window', label: 'Window', help: 'The time window, local New York time.'},
    {id: 'clock', label: 'Window start'},
    {
      id: 'cellSize',
      label: 'Hexagon radius',
      help: 'Centre to corner of one hexagon (side of one square for the grid).'
    },
    {
      id: 'gain',
      label: 'Biggest gain',
      help: 'The zone with the largest positive net: the most trips arriving over leaving.'
    },
    {id: 'loss', label: 'Biggest loss', help: 'The zone with the largest negative net.'},
    {
      id: 'citySum',
      label: 'City net, summed',
      help: 'Arrivals minus departures over every zone. It would be zero in a closed system; the residual is trips in flight across the window edges.'
    },
    {
      id: 'balanced',
      label: 'About balanced',
      help: 'Zones with trips whose net is within 5 trips of zero (the middle class).'
    },
    {
      id: 'nonBalanced',
      label: 'Out of balance',
      unit: 'hexagons',
      help: 'Zones with trips whose net is beyond 5 trips either way.'
    },
    {id: 'maxNet', label: 'Largest net', help: 'The biggest absolute net of any zone.'},
    {
      id: 'hidden',
      label: 'Hidden by the rule',
      unit: 'hexagons',
      help: 'Zones with trips but fewer than the minimum volume: drawn hatched in the share map.'
    },
    {
      id: 'sharpest',
      label: 'Most one-sided busy hexagon',
      help: 'The largest share of arrivals minus departures among zones that pass the minimum volume.'
    },
    {id: 'lgaNet', label: 'LaGuardia net', help: 'Net of the zones in the LaGuardia box.'},
    {id: 'jfkNet', label: 'JFK net', help: 'Net of the zones in the JFK box.'},
    {
      id: 'medianTrip',
      label: 'Median trip time',
      help: 'The routed duration (a derived value, not metered) that separates the departure clock from the arrival clock.'
    },
    {
      id: 'dayChart',
      label: 'Net arrivals per hour',
      kind: 'chart',
      help: 'Hourly net of a Midtown box and an Upper East Side box. Click to move the window.'
    },
    {
      id: 'panelChart',
      label: 'Four Friday hours',
      kind: 'chart',
      help: 'The net of four boxes in four Friday hours: Midtown fills at breakfast and the Upper East Side empties.'
    },
    {id: 'rows', label: 'Trips on the GPU', format: 'integer', hood: true},
    {id: 'zones', label: 'Lattice', hood: true},
    {id: 'departures', label: 'Departures in window', format: 'integer', hood: true},
    {id: 'arrivals', label: 'Arrivals in window', format: 'integer', hood: true},
    {id: 'activeZones', label: 'Zones with trips', hood: true},
    {id: 'flowRows', label: 'Flows listed', format: 'integer', hood: true},
    {id: 'pairs', label: 'Distinct pairs', format: 'integer', hood: true},
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      hood: true,
      help: 'Yes means more distinct pairs than the hash table holds (524,288): zone totals would be incomplete.'
    }
  ],

  pipeline: [
    {
      id: 'gate',
      label: 'Gate',
      detail: 'Keep trips whose pickup (or drop-off) time is in the window'
    },
    {
      id: 'locate',
      label: 'Locate',
      detail: 'Origin and destination points to zone ids at the radius'
    },
    {id: 'pair', label: 'Pair', detail: 'Hash each (origin, destination) pair and sum the trips'},
    {
      id: 'net',
      label: 'Net',
      detail: 'Per-zone totals read back; arrivals minus departures on the CPU'
    },
    {
      id: 'top-k',
      label: 'Top K',
      detail: 'The heaviest flows, read back (a few KB) and drawn as arrows'
    }
  ],

  timeline: {
    time: 'time',
    play: 'play',
    speed: 'playSpeed',
    format: formatTaxiTime,
    bands: [
      {from: FIRST_HOUR, to: 6, label: 'night'},
      {from: 18, to: 30, label: 'night'}
    ],
    ticks: [
      {at: 12, label: 'noon'},
      {at: 24, label: 'Fri'},
      {at: 36, label: 'noon'}
    ]
  },

  legends: getLegends,

  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Where does Manhattan fill and empty?', chips: ['Sample']},
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat: 'Yellow cabs only; drop-off time is pickup plus the routed duration.'
  },
  annotations: MANHATTAN_LABELS,

  snippet: getSnippet,

  about: {
    what: 'Previously: from points to a density field. Next: movement in time, cab by cab.\n\n`GPUFlowAggregation` assigns every origin-destination row to a zone (here a hexagon lattice), keeps the rows whose timestamp is inside a time window, and writes per-zone departure and arrival totals plus a weight-ranked list of the heaviest zone-to-zone flows. Two copies of the graph, gated on the pickup time and on the drop-off time, give departures and arrivals for the same window. The drop-off time is the pickup plus the routed duration, a derived value; the net is assembled on the CPU after a small readback and written back as one buffer (a GPU subtract node would remove that round trip).',
    why: "Net flow shows where a city gains and loses people through the day, the input to staffing, curb space and transit decisions. It is also the chapter's lesson in signed data: a diverging scheme needs a meaningful midpoint, a net count follows volume so a share compares places, and the size of the zone is part of the answer. The window is a four-number buffer, so the clock can sweep the whole day and the answer keeps up.",
    howToRead:
      "Orange hexagons sent out more riders than they took in during the window; purple took in more; pale is about balanced. Hexagons without any trip are transparent: nothing is not balanced. The classes are fixed, so the same colour means the same net at every hour and hexagon size. The data are yellow cabs only, a sample, and Thursday was New Year's Day: holiday traffic is not a normal day."
  },

  create: async ctx => (await import('./nyc-taxi-tides.compute')).createNycTaxiTides(ctx),

  story: storyFromMarkdown<NycTaxiTidesOptions>(narrative, {
    'the-tide': {
      headline: 'On a Friday morning the core fills',
      textAlternative:
        'Map of Manhattan in hexagons coloured from orange to purple by net taxi arrivals: Midtown near Grand Central is deep purple, the east side of Central Park orange.',
      optionsMode: 'fresh',
      options: {time: 32, windowHours: 1, normalise: 'net'},
      controls: ['time', 'windowHours'],
      readouts: ['window', 'gain', 'loss', 'citySum'],
      camera: {bounds: MANHATTAN_BOUNDS, transitionMs: 1400},
      furniture: cartouche('Where does Manhattan fill and empty?'),
      annotations: labelsFor(NYC, ['grand-central', 'penn-station'], {
        'grand-central': {minZoom: 10},
        'penn-station': {minZoom: 10}
      }),
      stage: 'net'
    },
    midpoint: {
      headline: 'Without a midpoint the map loses direction',
      textAlternative:
        'The same hexagons split by a divider: grey classes of the size of the net on the left, orange and purple net classes on the right.',
      optionsMode: 'fresh',
      options: {time: 32, windowHours: 1, normalise: 'net', compareMidpoint: true},
      controls: ['time'],
      readouts: ['balanced', 'nonBalanced', 'maxNet'],
      compare: {labels: ['Size of the net', 'Net, diverging'], position: 0.5},
      camera: {bounds: MANHATTAN_BOUNDS, transitionMs: 1200},
      furniture: cartouche('A middle is what diverging needs'),
      stage: 'net'
    },
    share: {
      headline: 'Busy places look balanced; quiet ones do not',
      textAlternative:
        'Hexagons coloured by imbalance share: many are hatched because they carry too few trips, and a few quiet places are strongly orange or purple.',
      optionsMode: 'fresh',
      options: {time: 32, windowHours: 1, normalise: 'share', minVolume: 20},
      controls: ['normalise', 'minVolume'],
      readouts: ['hidden', 'sharpest'],
      camera: {bounds: MANHATTAN_BOUNDS, transitionMs: 1200},
      furniture: cartouche('Shares compare places, counts do not'),
      annotations: labelsFor(NYC, ['penn-station'], {'penn-station': {minZoom: 10}}),
      stage: 'net'
    },
    'play-the-day': {
      headline: 'The tide turns through the day',
      textAlternative:
        'The net map animated over a day and a half, with a line chart of Midtown and the Upper East Side and four small bar charts of Friday morning.',
      optionsMode: 'fresh',
      options: {play: true, time: 1, windowHours: 1, playSpeed: 1, normalise: 'net'},
      controls: ['play', 'playSpeed'],
      readouts: ['clock', 'dayChart', 'panelChart'],
      camera: {bounds: MANHATTAN_BOUNDS, transitionMs: 0},
      furniture: {
        title: {title: 'The tide turns through the day', chips: ['Sample']},
        clock: {
          option: 'time',
          // The archive stores New York local time as if UTC: local midnight is 05:00 UTC.
          time: {origin: '2015-01-01T05:00:00Z', unit: 'hours'},
          zones: ['America/New_York', 'UTC'],
          show: 'datetime'
        }
      },
      stage: 'gate'
    },
    'hex-size': {
      headline: 'Hexagon size changes the answer',
      textAlternative:
        'The same hour in hexagons of one size with a dashed ring showing one hexagon radius around the biggest gain.',
      optionsMode: 'fresh',
      options: {time: 32, windowHours: 1, normalise: 'net', zoneSize: 400, showRadius: true},
      controls: ['hexSizes'],
      readouts: ['cellSize', 'maxNet', 'nonBalanced'],
      camera: {bounds: MANHATTAN_BOUNDS, transitionMs: 1200},
      furniture: {
        title: {title: 'Hexagon size changes the answer', chips: ['Sample']},
        clock: false
      },
      stage: 'locate'
    },
    airports: {
      headline: 'Evening flows leave from the airports',
      textAlternative:
        'Map from Manhattan to the two airports in orange and purple hexagons with the heaviest flows as arrows; the hexagons at LaGuardia and JFK are strongly orange.',
      optionsMode: 'fresh',
      options: {time: 18, windowHours: 3, normalise: 'net', zoneSize: 800, flowCount: 20},
      controls: ['time', 'windowHours', 'normalise'],
      readouts: ['lgaNet', 'jfkNet', 'medianTrip'],
      camera: {bounds: AIRPORT_BOUNDS, transitionMs: 1800},
      furniture: {
        title: {title: 'Evening: the airports send cabs out', chips: ['Sample']},
        clock: false
      },
      annotations: labelsFor(NYC, ['jfk', 'lga']),
      stage: 'top-k'
    }
  })
});
