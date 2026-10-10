// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, MONTREAL} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import type {ClassTable} from '../../cartography/types';
import {formatCount} from '../../cartography/live-text';
import {getSizeLegendEntries} from '../../cartography/proportional';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-tides.md?raw';
import type {BixiTidesOptions} from './bixi-tides.compute';
import {
  getNetTable,
  getTidesScale,
  TIDES_DISC_MAX_PIXELS,
  TIDES_DISC_MIN_PIXELS,
  type TidesScale
} from './bixi-tides-style';
import {FLOW_CREDITS, FLOW_INK, getFlowWidthLegend, NET_FLOW_WORDS} from './flows-style';

/** Mirrors the compute module without importing it (scene files stay light). */
const FIRST_HOUR = 0;
const LAST_HOUR = 23;
const FLOW_MAX_PIXELS = 6;
/** Weekdays of August 2024, only for the legend before any data has been read. */
const FALLBACK_WEEKDAYS = 22;

const CREDIT = joinCredits(FLOW_CREDITS.bixi, CREDITS.openStreetMap, CREDITS.colorBrewer);

/** The cartouche of a step: line 1 here, the subtitle and sample line come from the compute module. */
const cartouche = (title: string) => ({title: {title}});

type TidesLegendData = Readonly<{
  table?: ClassTable;
  scale?: TidesScale;
  counts?: number[];
  flowMaximum?: number;
  ground?: 'light' | 'dark';
}>;

function getLegends(
  state: BixiTidesOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const tides = (data['tides'] ?? {}) as TidesLegendData;
  const mapGround = tides.ground ?? 'light';
  const scale =
    tides.scale ??
    getTidesScale({
      units: state.units,
      scaleMode: state.scaleMode,
      dayType: state.dayType,
      hourMaximum: 1,
      weekdayCount: FALLBACK_WEEKDAYS
    });
  const table = tides.table ?? getNetTable(scale, mapGround);
  const words = NET_FLOW_WORDS;
  const legends: LegendSpec[] = [
    getClassTableLegend(table, {
      title: 'Net bikes per station',
      id: 'net-classes',
      basis: 'arrivals minus departures',
      counts: tides.counts,
      interactive: true,
      layout: 'list',
      note: `${words.low} (orange) to ${words.high} (purple); pale is ${words.midpoint.toLowerCase()}. ${scale.note}`
    }),
    {
      kind: 'size',
      layout: 'nested',
      title: 'Disc size: distance from balance',
      unit: scale.unit,
      entries: getSizeLegendEntries(scale.sizeMaximum, TIDES_DISC_MAX_PIXELS, {
        minRadiusPixels: TIDES_DISC_MIN_PIXELS,
        format: value => formatCount(value)
      }),
      note: 'Disc area grows with the size of the net; the largest discs are clipped.'
    }
  ];
  if (state.flowCount > 0) {
    legends.push(
      getFlowWidthLegend({
        title: 'Flow between stations',
        maxValue: Math.max(1, tides.flowMaximum ?? 1),
        maxWidthPixels: FLOW_MAX_PIXELS,
        color: FLOW_INK[mapGround],
        scale: state.flowWidth,
        format: value =>
          `${formatCount(value)} ${state.units === 'total' ? 'rides in August' : 'rides per hour'}`
      })
    );
  }
  return legends;
}

function getSnippet(state: BixiTidesOptions): string {
  return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'tides'});
graph.add(
  new GPUFlowAggregation({
    zones: {kind: 'ids', zoneCount: 906},      // 905 stations + one zone for rare pairs
    originZoneIds, destinationZoneIds,         // uint32 per row
    weights,                                   // rides per (pair, day type, hour) row
    mask,                                      // uint32 per row: ${state.dayType === 'weekday' ? 'weekday' : 'weekend'}
    timeWindow: {timestamps: hours, window: windowParams.importToGraph(graph)},
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: 131072,
    output: {ids, count, overflow, requiredCount},   // top-512 pairs, heaviest first
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights,
    zoneOutWeights, zoneInWeights                 // per-station totals
  })
);
const compiled = graph.compile();                // once
// the window is closed at both ends: one hour slot is [hour, hour + 0.5]
windowParams.write(getGPUTimeWindowParameterValues({start: ${state.hour}, end: ${state.hour + 0.5}}));
compiled.encode(commandEncoder, {parameters: undefined});

// after the readback (a few KB), on the CPU:
net[station] = (zoneInWeights[station] - zoneOutWeights[station]) / ${state.dayType === 'weekday' ? 'weekdays' : 'weekendDays'};   // ${state.units === 'rate' ? 'rides per average hour' : 'skip the division for August totals'}
// one value, one size and one draw-order buffer feed SpatialAnalysisPointLayer; the top-K flows feed SpatialAnalysisFlowLayer`;
}

export default defineScene<BixiTidesOptions>({
  id: 'bixi-tides',
  title: 'Which BIXI docks drain and fill?',
  chapter: 'flows',
  order: 9,
  summary:
    'GPUFlowAggregation on BIXI rides: net bikes per station for any hour of an average weekday or weekend day, classed on one fixed scale, with the failing per-hour scale and August totals shown next to the rates.',
  contributors: ['GPUFlowAggregation'],
  datasets: [{id: 'bixi-flows', role: 'station flows by weekday, weekend and hour'}],
  initialView: {...CITY_FRAMES.montreal},

  options: [
    ...playbackOptions<BixiTidesOptions>({
      ids: {play: 'play', time: 'hour', speed: 'speed', loop: 'loop'},
      group: 'Time window',
      playing: false,
      time: {
        min: FIRST_HOUR,
        max: LAST_HOUR,
        step: 1,
        default: 8,
        label: 'Window start',
        format: formatPlaybackTime.hour,
        help: 'Start hour of the one-hour window, Montreal local time. A ride is counted in the hour it starts. The window is a four-number parameter buffer: moving it never recompiles. Play sweeps all 24 start hours on one fixed scale.'
      },
      speed: {
        min: 0.5,
        max: 4,
        step: 0.5,
        default: 2,
        unit: 'h/s',
        label: 'Play speed',
        help: 'Hours of the day per real second.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'dayType',
      label: 'Day type',
      group: 'Time window',
      apply: 'param',
      display: 'segmented',
      default: 'weekday',
      help: 'Average weekday (Monday to Friday) or average weekend day (Saturday and Sunday) of August 2024. A per-record mask buffer, rewritten when you change it.',
      options: [
        {value: 'weekday', label: 'Weekdays'},
        {value: 'weekend', label: 'Weekends'}
      ]
    },
    {
      kind: 'select',
      id: 'units',
      label: 'Unit',
      group: 'Scale',
      apply: 'param',
      display: 'segmented',
      default: 'rate',
      help: 'Rides per average hour of the chosen day type, or the raw August sum. A total adds up 22 weekdays but only nine weekend days, so totals of the two day types cannot be compared; rates can.',
      options: [
        {value: 'rate', label: 'Per average hour'},
        {value: 'total', label: 'August total'}
      ]
    },
    {
      kind: 'select',
      id: 'scaleMode',
      label: 'Scale',
      group: 'Scale',
      apply: 'param',
      display: 'segmented',
      default: 'fixed',
      help: 'Fixed: one set of classes and disc sizes for every hour. Per hour: every hour rescales to its own largest station, so every hour looks equally dramatic. The second is the failing state.',
      options: [
        {value: 'fixed', label: 'Fixed for the day'},
        {value: 'perHour', label: 'Per hour'}
      ]
    },
    {
      kind: 'slider',
      id: 'flowCount',
      label: 'Flows drawn',
      group: 'Flows',
      apply: 'param',
      min: 0,
      max: 40,
      step: 5,
      default: 0,
      help: 'How many of the heaviest station-to-station flows of the window are drawn, from the top-512 list the contributor writes and reads back (a few KB).'
    },
    {
      kind: 'select',
      id: 'flowWidth',
      label: 'Arrow width',
      group: 'Flows',
      apply: 'param',
      display: 'segmented',
      default: 'sqrt',
      disabledWhen: state => state.flowCount === 0,
      help: 'Square root of the flow keeps the drawn area proportional to the flow; linear makes the heaviest flow dominate. One maximum for every window, so widths compare across hours.',
      options: [
        {value: 'sqrt', label: 'Square root'},
        {value: 'linear', label: 'Linear'}
      ]
    },
    {
      kind: 'select',
      id: 'selection',
      label: 'Station',
      group: 'Station',
      apply: 'param',
      display: 'chips',
      default: 'none',
      help: 'Outline a station and chart its day: the biggest weekday-morning drain or fill, or the station you click on the map.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'top-drain', label: 'Top morning drain'},
        {value: 'top-fill', label: 'Top morning fill'},
        {value: 'clicked', label: 'Clicked'}
      ]
    },
    {
      kind: 'slider',
      id: 'compareHour',
      label: 'Compare with hour',
      group: 'Time window',
      apply: 'param',
      min: 0,
      max: 23,
      step: 1,
      default: 8,
      unit: 'h',
      expert: true,
      help: 'Start hour of the second window shown by the compare button of the evening step. A second compiled copy of the graph reads it.'
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-station rides',
      group: 'Under the hood',
      apply: 'compile',
      default: true,
      expert: true,
      help: 'Rejects rows whose origin and destination are the same station. The trip table here has none, so this changes only the compiled graph.'
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
      help: '`sorted` accumulates weights in a fixed tree: bitwise identical on every run and device. `atomic` is faster only when nearly every row has its own pair, and its rounding depends on GPU scheduling.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    }
  ],

  readouts: [
    {id: 'window', label: 'Window', help: 'The one-hour window and day type, Montreal local time.'},
    {
      id: 'windowRides',
      label: 'Rides leaving in the window',
      help: 'Rides that start in the window at any station, per average hour of the day type (or the August total).'
    },
    {
      id: 'topDrain',
      label: 'Biggest drain',
      help: 'The station with the most negative net in the window: more bikes leave than arrive.'
    },
    {
      id: 'topFill',
      label: 'Biggest fill',
      help: 'The station with the most positive net in the window: more bikes arrive than leave.'
    },
    {
      id: 'reversal',
      label: 'Stations that flip sign',
      format: 'percent',
      help: 'Of the stations at least one ride per hour out of balance in both windows, the share whose net changes sign between them.'
    },
    {
      id: 'busiestFlow',
      label: 'Busiest flow',
      help: 'The heaviest station-to-station flow of the window.'
    },
    {
      id: 'dayCounts',
      label: 'Days in August 2024',
      help: 'Counted from the dates of the month the data cover.'
    },
    {
      id: 'weekdayTotal',
      label: 'Weekday rides, whole month',
      help: 'Rides between stations on weekdays, and the average per weekday.'
    },
    {
      id: 'weekendTotal',
      label: 'Weekend rides, whole month',
      help: 'Rides between stations on weekend days, and the average per weekend day.'
    },
    {id: 'station', label: 'Outlined station', layout: 'block'},
    {
      id: 'stationEvidence',
      label: 'Morning ↔ evening evidence',
      layout: 'block',
      help: 'The selected station at 08:00 and 17:00 on an average weekday: net, arrivals and departures. This is evidence for the tide story, not dock occupancy, because capacity and truck rebalancing are absent.'
    },
    {
      id: 'sourceBasis',
      label: 'Source and filtering basis',
      layout: 'block',
      hood: true,
      help: 'Retained source rides, preprocessing exclusions, and the distinction between exact station totals and the frequent-pair arrows.'
    },
    {
      id: 'cityChart',
      label: 'Rides per hour of the day',
      kind: 'chart',
      help: 'Rides leaving all stations in each hour of an average weekday and an average weekend day. Click to move the window.'
    },
    {
      id: 'stationChart',
      label: 'Net bikes of the outlined station',
      kind: 'chart',
      help: 'Arrivals minus departures per hour of an average weekday and weekend day. Above zero the dock fills. Click to move the window.'
    },
    {id: 'rows', label: 'Rows on the GPU', format: 'integer', hood: true},
    {id: 'pairs', label: 'Distinct station pairs', format: 'integer', hood: true},
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      hood: true,
      help: 'Yes means more distinct pairs than the hash table holds: station totals would be incomplete.'
    },
    {
      id: 'cpuMatch',
      label: 'GPU totals match the CPU profile',
      hood: true,
      help: 'Every station total of the window, read back from the GPU, is compared with the same sum over the rows on the CPU.'
    }
  ],

  pipeline: [
    {
      id: 'gate',
      label: 'Gate',
      detail: 'Keep rows of the chosen day type whose start hour is in the window'
    },
    {id: 'pair', label: 'Pair', detail: 'Hash each (origin, destination) station pair'},
    {id: 'sum', label: 'Sum', detail: 'Add the rides of every pair, in a fixed order'},
    {
      id: 'rank',
      label: 'Rank',
      detail: 'The heaviest flows, read back (a few KB) and drawn as arrows'
    },
    {id: 'net', label: 'Net', detail: 'Per-station out and in totals; net per day on the CPU'}
  ],

  timeline: {
    time: 'hour',
    play: 'play',
    speed: 'speed',
    format: formatPlaybackTime.hour,
    ticks: [
      {at: 8, label: '08'},
      {at: 12, label: 'noon'},
      {at: 17, label: '17'},
      {at: 22, label: '22'}
    ]
  },

  legends: getLegends,

  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Which docks drain in the morning?'},
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat:
      'No dock capacity or truck rebalancing in the data: net is rider pressure, not occupancy.'
  },

  snippet: getSnippet,

  about: {
    what: 'Previously: net flow on a hexagon lattice (nyc-taxi-tides). Next: do riders follow the borough map?\n\n`GPUFlowAggregation` assigns every origin-destination row to a station, keeps the rows whose start hour is inside a time window and day type, sums the pairs in a GPU hash table and writes a ranked top-K flow list plus per-station departure and arrival totals. The net, arrivals minus departures per average day-type hour, is one subtraction and one division on the CPU after a small readback. A second copy of the graph reads the compare window.',
    why: 'A bike-share system works only if there is a bike where you start and a free dock where you end. Net flow shows where the riders themselves drain and fill the network, which is where an operator would move bikes. It is also a small lesson in animation: one fixed scale keeps hours comparable, and rates, not month totals, keep day types comparable.',
    howToRead:
      'Orange stations send out more bikes than they take in during the hour, purple ones gain, pale ones are about balanced. The disc grows with the size of the imbalance, in rides per average weekday hour, on one scale for every hour. A ride is counted in the hour it starts, so a ride across 09:00 counts at 08:00. Rebalancing trucks are not in the data, rare pairs are folded into a residual zone (station totals stay exact) and capacity is missing, so a drain here is a pressure on a dock of unknown size. One month, August 2024.'
  },

  create: async ctx => (await import('./bixi-tides.compute')).createBixiTides(ctx),

  story: storyFromMarkdown<BixiTidesOptions>(narrative, {
    morning: {
      headline: 'Morning rides empty some docks and fill others',
      textAlternative:
        'Map of Montreal with BIXI stations as discs: orange discs where more bikes leave than arrive at 08:00 on weekdays, purple discs downtown where bikes pile up.',
      optionsMode: 'fresh',
      options: {hour: 8, dayType: 'weekday', units: 'rate', scaleMode: 'fixed', flowCount: 0},
      controls: ['hour'],
      readouts: ['window', 'topDrain', 'topFill', 'windowRides'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1400},
      furniture: cartouche('Which docks drain in the morning?'),
      annotations: labelsFor(MONTREAL, [
        'downtown',
        'plateau',
        'mile-end',
        'rosemont',
        'villeray',
        'mount-royal',
        'saint-lawrence'
      ]),
      stage: 'net'
    },
    evening: {
      headline: 'At five the tide reverses',
      textAlternative:
        'The same stations at 17:00: downtown discs are now orange and the neighbourhoods purple; holding the compare button shows the morning.',
      optionsMode: 'fresh',
      options: {
        hour: 17,
        compareHour: 8,
        dayType: 'weekday',
        units: 'rate',
        scaleMode: 'fixed',
        flowCount: 20,
        flowWidth: 'sqrt'
      },
      controls: ['flowWidth'],
      readouts: ['reversal', 'busiestFlow', 'topDrain'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      // Side a is the live evening window, side b the morning: the step opens on the evening.
      compare: {mode: 'toggle', labels: ['17:00', '08:00']},
      furniture: cartouche('Where do the bikes go at five?'),
      annotations: labelsFor(MONTREAL, [
        'downtown',
        'plateau',
        'hochelaga',
        'verdun',
        'mount-royal',
        'saint-lawrence'
      ]),
      stage: 'rank'
    },
    'play-the-day': {
      headline: 'Fix the scale before you animate',
      textAlternative:
        'The net map animated through the day on one fixed scale, with a line chart of rides per hour for weekdays and weekends and a moving clock.',
      optionsMode: 'fresh',
      options: {
        play: true,
        hour: FIRST_HOUR,
        speed: 2,
        dayType: 'weekday',
        units: 'rate',
        scaleMode: 'fixed',
        flowCount: 0
      },
      controls: ['play', 'scaleMode'],
      readouts: ['window', 'windowRides', 'cityChart'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 0},
      furniture: {
        title: {title: 'One scale for the whole day'},
        clock: {
          option: 'hour',
          // Local midnight in August is 04:00 UTC (EDT); the average day has no date, so show the time only.
          time: {origin: '2024-08-07T04:00:00Z', unit: 'hours'},
          zones: ['America/Toronto', 'UTC'],
          show: 'time'
        }
      },
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'villeray', 'saint-lawrence']),
      stage: 'gate'
    },
    rates: {
      headline: 'A month total is not a rate',
      textAlternative:
        'The morning net map in August totals for weekdays; switching to weekends makes every disc look weaker only because the month has fewer weekend days.',
      optionsMode: 'fresh',
      options: {
        hour: 8,
        dayType: 'weekday',
        units: 'total',
        scaleMode: 'fixed',
        flowCount: 0
      },
      controls: ['units', 'dayType'],
      readouts: ['dayCounts', 'weekdayTotal', 'weekendTotal', 'sourceBasis'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      furniture: {title: {title: 'Totals or rates?'}, clock: false},
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'mile-end', 'saint-lawrence']),
      stage: 'net'
    },
    'one-station': {
      headline: 'One dock shows the whole day’s tide',
      textAlternative:
        'Zoomed map around one outlined orange station with a line chart of its net bikes for every hour of an average weekday and weekend day.',
      optionsMode: 'fresh',
      options: {
        hour: 8,
        dayType: 'weekday',
        units: 'rate',
        scaleMode: 'fixed',
        selection: 'top-drain',
        flowCount: 0
      },
      controls: ['hour', 'dayType', 'units'],
      readouts: ['station', 'stationEvidence', 'stationChart'],
      furniture: {title: {title: 'One station through the day'}, clock: false},
      annotations: labelsFor(MONTREAL, [
        'parc-jeanne-mance',
        'mount-royal',
        'mile-end',
        'parc-la-fontaine'
      ]),
      stage: 'net'
    }
  })
});
