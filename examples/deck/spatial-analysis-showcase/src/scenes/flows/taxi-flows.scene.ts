// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatCount} from '../../cartography/live-text';
import {getClassTableLegend} from '../../cartography/class-table';
import {getSizeLegendEntries} from '../../cartography/proportional';
import type {ClassTable} from '../../cartography/types';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import {FLOW_CREDITS, FLOW_INK, getFlowWidthLegend, inkFor, withInkAlpha} from './flows-style';
import {
  INTERIOR_CIRCLE_COUNT,
  MAXIMUM_FLOW_WIDTH,
  MAXIMUM_INTERIOR_RADIUS,
  TOP_FLOW_COUNT,
  formatTaxiDayHour
} from './taxi-flows-constants';
import type {TaxiFlowOptions} from './taxi-flows.compute';

const WEIGHT_NAMES: Record<string, string> = {
  trips: 'trips',
  fare: 'dollars of fares',
  duration: 'ride hours',
  all: 'jobs',
  low: 'low-earning jobs',
  mid: 'middle-earning jobs',
  high: 'high-earning jobs'
};

/** The cartouche of one step: the claim, then the variable, unit and method (the sample line is set from the data). */
const cartouche = (title: string, subtitle: string, chips?: readonly string[]) => ({
  title,
  subtitle,
  ...(chips ? {chips} : {})
});

/** Place names that orient the reader in every step (drawn above the data from the gazetteer). */
const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan', 'loop', 'ohare', 'midway'], {
  loop: {priority: 4},
  ohare: {priority: 4},
  midway: {priority: 3}
});

/** The extra neighbourhood names of the establishing shot (twelve places in all). */
const ESTABLISHING_PLACES = labelsFor(
  CHICAGO,
  [
    'near-north-side',
    'evanston',
    'oak-park',
    'lakeview',
    'austin',
    'pilsen',
    'hyde-park',
    'englewood'
  ],
  {
    'near-north-side': {priority: 3},
    evanston: {tone: 'muted'},
    'oak-park': {tone: 'muted'},
    austin: {tone: 'muted'},
    englewood: {tone: 'muted'}
  }
);

const LOOP_CENTER = CHICAGO.places.loop.lngLat;

export default defineScene<TaxiFlowOptions>({
  id: 'taxi-flows',
  title: 'Which pairs of areas carry Chicago’s taxis?',
  chapter: 'flows',
  order: 1,
  summary:
    'A year of Chicago taxi trips ranked on the GPU into the pairs of community areas that carry them, drawn as a designed flow map, with what a flow map leaves out: trips inside an area, the choice of zones, the hour. Then the same machinery on census-tract commutes.',
  contributors: ['GPUFlowAggregation'],
  datasets: [
    {id: 'chicago-taxi-od', role: 'taxi flows by area, weekday and hour'},
    {id: 'chicago-community-areas', role: 'zone polygons'},
    {id: 'chicago-lodes-od', role: 'home-to-work flows between tracts'},
    {id: 'chicago-tracts', role: 'tract polygons'}
  ],
  initialView: {...CITY_FRAMES.chicago},

  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Dataset',
      group: 'Data',
      apply: 'compile',
      default: 'taxi',
      display: 'segmented',
      help: 'Taxi trips between community areas (a year, with pickup hour) or home-to-work commutes between census tracts (LODES, no timestamps). Switching rebuilds the graph for the new rows.',
      options: [
        {value: 'taxi', label: 'Taxi trips'},
        {value: 'commute', label: 'Commutes'}
      ]
    },
    {
      kind: 'select',
      id: 'zones',
      label: 'Zones',
      group: 'Data',
      apply: 'compile',
      default: 'native',
      help: 'The dataset’s own areas (`ids`), or a hexagon or square lattice laid over the area centres. The zone kind is compile-time; the lattice size is not.',
      options: [
        {value: 'native', label: 'Dataset areas (caller zone IDs)'},
        {value: 'hexagon', label: 'Hexagon lattice'},
        {value: 'grid', label: 'Square grid'}
      ]
    },
    {
      kind: 'slider',
      id: 'zoneSize',
      label: 'Lattice size',
      group: 'Data',
      apply: 'param',
      min: 400,
      max: 9000,
      step: 100,
      default: 3000,
      unit: 'm',
      disabledWhen: state => state.zones === 'native',
      help: 'Hexagon radius or square cell size. Both the lattice dimensions and the radius are per-frame values under a compile-time capacity, so this never recompiles.'
    },
    {
      kind: 'preset',
      id: 'latticePreset',
      label: 'Zone size',
      group: 'Data',
      help: 'The dataset’s own areas, or a hexagon lattice of three radii. The same trips, assigned to different zones.',
      presets: [
        {label: 'Areas', values: {zones: 'native'}},
        {label: '1.5 km', values: {zones: 'hexagon', zoneSize: 1500}},
        {label: '3 km', values: {zones: 'hexagon', zoneSize: 3000}},
        {label: '6 km', values: {zones: 'hexagon', zoneSize: 6000}}
      ]
    },
    {
      kind: 'toggle',
      id: 'excludeSelf',
      label: 'Exclude same-zone flows',
      group: 'Data',
      apply: 'compile',
      default: true,
      help: 'Rejects records whose origin and destination zones are equal. A flow map cannot draw them; keep them to see the interior circles and the interior share.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Summation order',
      group: 'Data',
      apply: 'compile',
      default: 'sorted',
      expert: true,
      help: '`sorted` accumulates weights in a fixed tree: bitwise identical on every run and device. `atomic` is faster only when nearly every record has its own pair, and its rounding depends on GPU scheduling.',
      options: [
        {value: 'sorted', label: 'Sorted (deterministic)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    },
    {
      kind: 'button',
      id: 'compareSum',
      label: 'Compare summation orders',
      group: 'Data',
      expert: true,
      help: 'Builds the other summation order, times both graphs outside the frame and reports the largest difference between their zone totals.'
    },
    {
      kind: 'select',
      id: 'taxiWeight',
      label: 'Weight (taxi)',
      group: 'Weight',
      apply: 'param',
      default: 'trips',
      disabledWhen: state => state.source !== 'taxi',
      help: 'What each record contributes to its flow. Rewriting the weight buffer re-runs the same graph.',
      options: [
        {value: 'trips', label: 'Trips'},
        {value: 'fare', label: 'Fare revenue (USD, no tips)'},
        {value: 'duration', label: 'Ride time (hours)'}
      ]
    },
    {
      kind: 'select',
      id: 'commuteWeight',
      label: 'Earnings',
      group: 'Weight',
      apply: 'param',
      default: 'all',
      display: 'chips',
      disabledWhen: state => state.source !== 'commute',
      help: 'Jobs by monthly earnings: low up to $1,250, middle $1,251 to $3,333, high above $3,333. Widths and zone colours use one scale for every choice, so a thin flow is a small flow.',
      options: [
        {value: 'all', label: 'All jobs'},
        {value: 'low', label: 'Low'},
        {value: 'mid', label: 'Middle'},
        {value: 'high', label: 'High'}
      ]
    },
    ...playbackOptions<TaxiFlowOptions>({
      ids: {play: 'play', time: 'hourStart', speed: 'playSpeed', loop: 'loop'},
      group: 'Time of day',
      playing: false,
      disabledWhen: state => state.source !== 'taxi',
      time: {
        min: 0,
        max: 23,
        step: 1,
        default: 0,
        label: 'Window start',
        format: formatTaxiDayHour,
        help: 'First hour of the window. The taxi day starts at 04:00 (local time), so a window can run through midnight. Whole hours only: the data are hourly.'
      },
      speed: {
        min: 0.5,
        max: 4,
        step: 0.5,
        default: 1,
        unit: 'h/s',
        label: 'Play speed',
        help: 'Hours of the day per second. At 1 h/s the whole day takes 24 seconds.'
      },
      loop: true
    }),
    {
      kind: 'range',
      id: 'hours',
      label: 'Time window',
      group: 'Time of day',
      apply: 'param',
      min: 0,
      max: 24,
      step: 1,
      default: [0, 24],
      format: formatTaxiDayHour,
      disabledWhen: state => state.source !== 'taxi',
      help: 'Pickup hours kept, from the first to the end of the last, on the taxi day that starts at 04:00. The window is a 4-number parameter buffer: moving it never recompiles.'
    },
    {
      kind: 'select',
      id: 'dayType',
      label: 'Days',
      group: 'Time of day',
      apply: 'param',
      default: 'all',
      display: 'segmented',
      disabledWhen: state => state.source !== 'taxi',
      help: 'Weekdays (Monday to Friday), weekends (Saturday and Sunday) or every day. A per-record mask buffer, rewritten when you change it.',
      options: [
        {value: 'all', label: 'All'},
        {value: 'weekday', label: 'Weekdays'},
        {value: 'weekend', label: 'Weekends'}
      ]
    },
    {
      kind: 'select',
      id: 'totals',
      label: 'Zone shading',
      group: 'Display',
      apply: 'param',
      default: 'departures',
      display: 'segmented',
      disabledWhen: state => state.source !== 'taxi',
      help: 'Per-zone totals written by the contributor, as a rate per km²: weight leaving the zone or arriving in it.',
      options: [
        {value: 'departures', label: 'Departures'},
        {value: 'arrivals', label: 'Arrivals'}
      ]
    },
    {
      kind: 'select',
      id: 'flowStyle',
      label: 'Flow style',
      group: 'Display',
      apply: 'param',
      default: 'designed',
      display: 'segmented',
      help: 'Raw: the largest pairs as straight, equal, unordered lines. Designed: width by the square root of the trips on one scale, curved so a flow and its return part, arrowheads for direction, heaviest drawn last.',
      options: [
        {value: 'raw', label: 'Raw'},
        {value: 'designed', label: 'Designed'}
      ]
    },
    {
      kind: 'slider',
      id: 'arcs',
      label: 'Flows drawn',
      group: 'Display',
      apply: 'param',
      min: 5,
      max: TOP_FLOW_COUNT,
      step: 5,
      default: 60,
      help: 'How many of the ranked flows are drawn, heaviest first: the threshold of a flow map. The list read back from the GPU holds the largest pairs; this only limits drawing.'
    },
    {
      kind: 'select',
      id: 'annotate',
      label: 'Finding notes',
      group: 'Display',
      apply: 'param',
      default: 'none',
      display: 'chips',
      help: 'Notes on the map read back from the GPU: the three largest flows, the largest interior area, or the busiest flow of the window.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'top-flows', label: 'Three largest flows'},
        {value: 'interior', label: 'Largest interior area'},
        {value: 'top-flow', label: 'Busiest flow'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showZones',
      label: 'Shade zones',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Colour each zone by its total per km² in five fixed classes: the quiet backdrop of the flows.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Area outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the boundaries of the dataset areas, also under a lattice.'
    }
  ],

  readouts: [
    {
      id: 'volume',
      label: 'Trips mapped',
      emphasis: 'tile',
      help: 'The weight of every record that passed the window, the day mask and the zone checks: trips with both an origin and a destination area, without same-zone trips while they are excluded.'
    },
    {
      id: 'pairs',
      label: 'Pairs found',
      format: 'integer',
      help: 'Distinct (origin, destination) pairs held in the GPU hash table.'
    },
    {
      id: 'drawn',
      label: 'Flows drawn',
      format: 'integer',
      help: 'Flows between zones drawn on the map.'
    },
    {
      id: 'share',
      label: 'Carried by the drawn flows',
      format: 'percent',
      emphasis: 'tile',
      help: 'Weight of the drawn flows divided by the weight of all flows between zones in the window.'
    },
    {
      id: 'concentrationChart',
      label: 'How concentrated are the flows?',
      kind: 'chart',
      help: 'Cumulative share of the trips between zones carried by the largest flows. Steep means a few pairs dominate. The marker is the number of flows drawn; click the chart to set it.'
    },
    {
      id: 'interiorShare',
      label: 'Interior share of all trips',
      format: 'percent',
      emphasis: 'tile',
      help: 'Trips that start and end in the same zone, from the ranked list (which holds almost every same-zone pair), as a share of all trips. Needs same-zone flows to be kept.'
    },
    {
      id: 'interiorTop',
      label: 'Largest interior area',
      help: 'The zone with the most trips that start and end inside it.'
    },
    {
      id: 'odMatrix',
      label: 'Origin-destination matrix',
      kind: 'chart',
      help: 'Trips between the community areas of the ranked list, origin down and destination across, areas ordered by throughput. The outlined diagonal is the interior trips a flow map cannot draw. Click a cell to outline its two areas on the map.'
    },
    {
      id: 'window',
      label: 'Time window',
      help: 'Local clock hours of the window on the taxi day.'
    },
    {
      id: 'hourlyChart',
      label: 'The day in pickups',
      kind: 'chart',
      help: 'Share of each day type’s trips in every pickup hour, starting at 04:00. The shaded band is the time window.'
    },
    {
      id: 'topFlow',
      label: 'Busiest pair',
      help: 'The largest flow between two zones in the window.'
    },
    {
      id: 'top25Share',
      label: 'Carried by the top 25 flows',
      format: 'percent',
      help: 'Share of the trips between zones carried by the 25 largest flows.'
    },
    {
      id: 'jobsTop',
      label: 'Jobs in the busiest tract',
      help: 'The workplace tract that receives the most commuters in these flows.'
    },
    {id: 'zones', label: 'Zones', hood: true},
    {id: 'activeZones', label: 'Zones with traffic', hood: true},
    {
      id: 'truncated',
      label: 'Ranked list truncated',
      hood: true,
      help: 'The ranked list holds the largest pairs. Zone totals stay exact even when pairs are not listed.'
    },
    {
      id: 'pairTable',
      label: 'Pair table load',
      hood: true,
      help: 'Distinct pairs against the compile-time capacity of the hash table. It overflows only when the pairs outnumber the slots.'
    },
    {
      id: 'pairOverflow',
      label: 'Pair table overflow',
      hood: true,
      help: 'Set when distinct pairs exceeded the hash table capacity.'
    },
    {
      id: 'scaleMax',
      label: 'Width scale maximum',
      hood: true,
      help: 'The one value drawn at the widest line: the largest flow of the whole dataset under any zoning, any window and any day type (any earnings weight for commutes).'
    },
    {
      id: 'sumTiming',
      label: 'Sum order timing',
      hood: true,
      help: 'GPU time per run of each summation order, from the compare button.'
    },
    {
      id: 'sumDifference',
      label: 'Sum order difference',
      hood: true,
      help: 'Largest difference between the zone totals of the two summation orders.'
    }
  ],

  pipeline: [
    {
      id: 'gate',
      label: 'Gate',
      detail: 'Rows are kept by time window and day mask: four numbers and a buffer, no rebuild'
    },
    {
      id: 'zones',
      label: 'Zone ids',
      detail:
        'Origin and destination become zone ids: caller areas or a lattice; same-zone rows can drop'
    },
    {
      id: 'pairs',
      label: 'Pair table',
      detail:
        'Each pair is hashed into a fixed-capacity table and its weights summed in a fixed order'
    },
    {id: 'rank', label: 'Rank', detail: 'Two stable sorts: the top-K pairs and the zone totals'},
    {
      id: 'draw',
      label: 'Read back',
      detail: 'A few kilobytes of the top-K list become arrows; the totals become the zone colours'
    }
  ],

  timeline: {
    time: 'hourStart',
    play: 'play',
    speed: 'playSpeed',
    window: 'hours',
    format: formatTaxiDayHour
  },

  legends: (state, data) => {
    const zones = data['zones'] as
      | {table: ClassTable; source: 'taxi' | 'commute'; ground: 'light' | 'dark'}
      | undefined;
    const scale = data['flowScale'] as {maxFlow: number; maxInterior: number} | undefined;
    const taxi = state.source === 'taxi';
    const surface = zones?.ground ?? 'dark';
    const ink = inkFor(FLOW_INK, surface);
    const weight = WEIGHT_NAMES[taxi ? state.taxiWeight : state.commuteWeight];
    const legends: LegendSpec[] = [];
    if (state.showZones && zones?.table) {
      legends.push(
        getClassTableLegend(zones.table, {
          id: 'zones',
          title: taxi ? (state.totals === 'arrivals' ? 'Arrivals' : 'Departures') : 'Jobs arriving',
          basis: taxi ? 'per km², per hour of an average day' : 'per km²'
        })
      );
    }
    if (state.flowStyle === 'raw') {
      legends.push({
        kind: 'categories',
        title: 'Flows',
        entries: [{color: withInkAlpha(ink, 110), label: 'One pair of areas', shape: 'line'}],
        note: `The ${TOP_FLOW_COUNT} largest pairs, straight, one width, no order.`
      });
    } else if (scale) {
      legends.push(
        getFlowWidthLegend({
          title: `Flow width (${weight})`,
          maxValue: scale.maxFlow,
          maxWidthPixels: MAXIMUM_FLOW_WIDTH,
          color: ink,
          format: value => formatCount(value),
          note: taxi
            ? 'Width grows with the square root of the flow. One scale for every window, day type and zoning.'
            : 'Width grows with the square root of the flow. One scale for every earnings group.'
        })
      );
    }
    if (taxi && !state.excludeSelf && scale && scale.maxInterior > 0) {
      legends.push({
        kind: 'size',
        title: `Trips inside one zone (${weight})`,
        layout: 'nested',
        entries: getSizeLegendEntries(scale.maxInterior, MAXIMUM_INTERIOR_RADIUS, {
          format: value => formatCount(value)
        }),
        color: withInkAlpha(ink, 64),
        outline: ink,
        unit: weight,
        note: `Area follows the trips. The ${INTERIOR_CIRCLE_COUNT} largest are drawn, under the flows.`
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUFlowAggregation} from '@luma.gl/experimental/gpu-network';
import {getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'flows'});
graph.add(
  new GPUFlowAggregation({
    zones: ${
      state.zones === 'native'
        ? "{kind: 'ids', zoneCount: 77},\n    originZoneIds, destinationZoneIds,   // uint32 per record"
        : `{
      kind: '${state.zones === 'grid' ? 'grid' : 'hexagon'}',
      bounds: bounds.importToGraph(graph),             // per-frame [minX, minY, maxX, maxY]
      gridSize: capacity,                              // compile-time upper bound
      activeGridSize: activeGrid.importToGraph(graph), // per-frame [columns, rows]${
        state.zones === 'hexagon' ? '\n      radius: radius.importToGraph(graph),' : ''
      }
    },
    origins, destinations,               // float32x2 per record`
    }
    weights,                             // float32 per record, rewritten for a new weight
    mask,                                // uint32 per record (weekday / weekend)${
      state.source === 'taxi'
        ? `
    timeWindow: {timestamps: taxiDayHours, window: windowParams.importToGraph(graph)},`
        : ''
    }
    excludeSelfFlows: ${state.excludeSelf},
    sumOrder: '${state.sumOrder}',
    pairCapacity: ${state.source === 'taxi' ? '16384' : '131072'},
    output: {ids, count, overflow, requiredCount},         // top-${TOP_FLOW_COUNT} pairs, heaviest first
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights,
    zoneOutWeights, zoneInWeights                       // per-zone totals
  })
);
const compiled = graph.compile();   // once
// every frame: write parameter buffers, then
windowParams.write(getGPUTimeWindowParameterValues({start: 13, end: 16}));
compiled.encode(commandEncoder, {parameters: undefined});
// when the list settles: a few kilobytes back, then arrows
const arrows = buildFlowArrows({originZones, destinationZones, weights, count, getZoneCenter, limit: ${state.arcs}});
new SpatialAnalysisFlowLayer({
  flows: arrows.flows, values: arrows.weights, ids: arrows.order,   // heaviest last
  maxValue: datasetMaximum, maxWidthPixels: ${MAXIMUM_FLOW_WIDTH}, instanceCount: arrows.count
});`,

  about: {
    what: 'Previously: the networks chapter asked how far you can get; this chapter asks who actually goes where. Next: [the New York taxi story](#/story/nyc-taxi-crossfilter).\n\n`GPUFlowAggregation` gates records by a time window and a day mask, assigns each origin and destination to a zone, hashes every pair into a fixed-capacity table, sums the weights in a fixed order, ranks the pairs and writes a top-K list plus per-zone departure and arrival totals. The scene reads that list back, a few kilobytes, and draws it as arrows; the totals become the zone colours.',
    why: 'Raw trip tables are too large to read and too dense to draw. A ranked list of the heaviest flows answers where movement concentrates and when, and, drawn with care, it shows what a flow map leaves out: trips inside an area, the effect of the zones chosen, and the hour of the day.',
    howToRead:
      'Flows join the centres of areas, not streets: an arrow says which areas are linked, never which route a taxi took. Width follows the square root of the trips on one scale for the whole dataset; the arrow shows direction and the curve bends left of travel, so a flow and its return part. Zone colour is a rate per km², in five classes whose breaks never move. Trips whose pickup or drop-off area the city suppressed for privacy, or that fall outside Chicago, are not in the data. Commutes are LODES counts, noise-infused for privacy, and only the largest tract pairs are kept. `sorted` summation gives bitwise-identical totals on every run; the compare button in the options times it against `atomic`.'
  },

  // Night ground: luminous flows add their light and the labels are ours, above the data.
  basemap: ground('night'),
  furniture: {
    title: cartouche('Which pairs carry Chicago’s taxis?', 'Taxi trips between community areas'),
    scaleBar: {units: 'metric'},
    credit: joinCredits(FLOW_CREDITS.chicagoTaxi, CREDITS.cityOfChicago, CREDITS.colorBrewer)
  },
  annotations: ORIENTATION,

  create: async ctx => (await import('./taxi-flows.compute')).createTaxiFlows(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Every pair at once',
      headline: 'Drawn all at once, flows are a tangle',
      textAlternative:
        'Dark map of Chicago with hundreds of thin straight gold lines joining the centres of community areas: a bright knot downtown and a faint web beyond.',
      body: 'A year of taxi trips, **{{volume}}**, falls into **{{pairs}}** pairs of community areas. Drawing the largest pairs as straight lines of one width gives a tangle: a bright knot downtown, and nothing the eye can rank.\n\nSwitch **Flow style** to *Designed* and watch width, curve and order do the work.',
      optionsMode: 'fresh',
      options: {flowStyle: 'raw', showZones: false, showOutlines: false, excludeSelf: true},
      controls: ['flowStyle'],
      readouts: ['volume', 'pairs'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'Which pairs carry Chicago’s taxis?',
          'The largest pairs of areas, all drawn alike'
        )
      },
      annotations: ESTABLISHING_PLACES,
      stage: 'draw'
    },
    {
      id: 'flow-map',
      title: 'Width, curve and order',
      headline: 'Width, curve and order make it readable',
      textAlternative:
        'The same map as a designed flow map: a few wide curved gold arrows between the Loop, the Near North Side and O’Hare over a quiet blue backdrop, with three labelled flows.',
      body: 'Width now follows the **square root** of the trips, on one scale. Flows curve so A to B and B to A part, arrowheads give direction, and the heaviest are drawn last. **{{drawn}}** flows carry **{{share}}** of all trips between areas.\n\nHold the compare button, or change **Flows drawn**: the threshold belongs on the map.\n\n*Width by the square root, one scale, heaviest on top.*',
      optionsMode: 'fresh',
      options: {flowStyle: 'designed', arcs: 60, annotate: 'top-flows'},
      controls: ['arcs'],
      readouts: ['share', 'concentrationChart', 'drawn'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      compare: {mode: 'toggle', labels: ['Designed', 'Raw']},
      furniture: {
        title: cartouche(
          'Which pairs carry Chicago’s taxis?',
          'Taxi trips between community areas, the top pairs drawn'
        )
      },
      stage: 'rank'
    },
    {
      id: 'interior',
      title: 'The flow a map cannot draw',
      headline: 'The biggest flow never leaves its area',
      textAlternative:
        'Close map of the Loop and the Near North Side with gold circles over areas, the largest on the Near North Side, drawn under the flow arrows.',
      body: 'A flow map draws between places, so trips that begin and end in one area vanish. With **Exclude same-zone flows** off, **{{interiorShare}}** of all trips are interior; the largest, **{{interiorTop}}**, outweighs every arrow.\n\nIn the matrix the outlined diagonal is the same fact; click a cell to outline its pair.\n\n*A flow map draws between places; the diagonal of the matrix is invisible to it.*',
      optionsMode: 'fresh',
      options: {excludeSelf: false, arcs: 25, annotate: 'interior'},
      controls: ['excludeSelf'],
      readouts: ['interiorShare', 'interiorTop', 'odMatrix'],
      camera: {longitude: LOOP_CENTER[0], latitude: LOOP_CENTER[1], zoom: 11, transitionMs: 1600},
      furniture: {
        title: cartouche('Which trips never leave home?', 'Trips that start and end in one area')
      },
      stage: 'zones'
    },
    {
      id: 'taxi-day',
      title: 'A taxi day',
      headline: 'The busiest pairs change with the hour',
      textAlternative:
        'The flow map for an evening window with a clock and a time bar below: the busiest pair is labelled, and the blue zone colours show departures per square kilometre for the window.',
      body: 'The taxi day starts at 04:00, just after the quietest hour, so a window can cross midnight unbroken. **{{window}}** ranks **{{topFlow}}** first.\n\nDrag the handles of **Time window** on the bar, or press Play to sweep it. Zone colour is a rate per km² on breaks fixed for the whole year, so a colour change is a data change. Choose **Days** and read the two curves.',
      optionsMode: 'fresh',
      options: {
        hours: [13, 16],
        hourStart: 13,
        dayType: 'weekday',
        arcs: 60,
        annotate: 'top-flow',
        play: false
      },
      controls: ['hours', 'dayType'],
      readouts: ['window', 'hourlyChart', 'topFlow'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: cartouche('When do the pairs change?', 'Taxi trips in a window of the taxi day'),
        clock: {
          option: 'hourStart',
          // Hour 0 of the taxi day is 04:00 in Chicago (09:00 UTC in summer).
          time: {origin: '2023-06-01T09:00:00Z', unit: 'hours'},
          zones: ['America/Chicago'],
          show: 'time'
        }
      },
      stage: 'gate'
    },
    {
      id: 'zones-are-a-choice',
      title: 'Zones are a choice',
      headline: 'Same trips, different zones, different flows',
      textAlternative:
        'Hexagons coloured by taxi departures with gold flows between hexagon centres, a dashed ring at the Loop showing the hexagon radius, and a scale bar ticked at that radius.',
      body: 'The same trips, assigned to hexagons instead of community areas. Pick a **Zone size** and watch the interior share, **{{interiorShare}}**; the share carried by the largest flows, **{{top25Share}}**; and the busiest pair, **{{topFlow}}**.\n\nAt the largest size neighbouring areas merge, trips vanish into interiors and the rest concentrates: the *modifiable areal unit problem*, met in [the wildlife map](#/story/nature-density).\n\n*Change the boundaries, change the flows.*',
      optionsMode: 'fresh',
      options: {
        zones: 'hexagon',
        zoneSize: 1500,
        excludeSelf: false,
        arcs: 25,
        annotate: 'top-flow'
      },
      controls: ['latticePreset'],
      readouts: ['interiorShare', 'top25Share', 'topFlow'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: cartouche('Does the zoning change the flows?', 'The same trips between hexagons')
      },
      stage: 'zones'
    },
    {
      id: 'commute',
      title: 'People, not taxis',
      headline: 'Commuting is far more dispersed than taxi trips',
      textAlternative:
        'Paper map of Chicago census tracts shaded in five orange classes by jobs arriving per square kilometre, darkest in the Loop, with thin dark flows from all over the city converging on it.',
      body: 'Paper now, because the jobs choropleth becomes a co-subject. These are home-to-work flows between census tracts, and the busiest tract takes **{{jobsTop}}**. The same number of flows carries only **{{share}}** of commuters, where taxi flows carried most trips.\n\n**Dataset** returns to the taxis; **Earnings** compares groups on one width scale; **Flows drawn** sets the threshold.',
      optionsMode: 'fresh',
      options: {
        source: 'commute',
        zones: 'native',
        totals: 'arrivals',
        arcs: 60,
        annotate: 'top-flow'
      },
      controls: ['source', 'commuteWeight', 'arcs'],
      readouts: ['share', 'jobsTop', 'concentrationChart'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      basemap: ground('paperCity'),
      furniture: {
        title: cartouche('Where do commuters go?', 'Home-to-work jobs between census tracts', [
          'Noise-infused counts'
        ]),
        credit: joinCredits(FLOW_CREDITS.lodes, CREDITS.cityOfChicago, CREDITS.colorBrewer)
      },
      stage: 'pairs'
    }
  ]
});
