// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, NYC} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {DROPOFF_INK, FLOW_CREDITS, PICKUP_INK} from './flows-style';
import type {NycTaxiCrossfilterOptions} from './nyc-taxi-crossfilter.compute';
import {
  GHOST_INK,
  getPlaceCenter,
  getPlacesBounds,
  type TaxiClassTables
} from './nyc-taxi-crossfilter-style';
import {
  DISTANCE_DOMAIN,
  FARE_DOMAIN,
  formatTaxiTime,
  HOUR_DOMAIN,
  PASSENGER_DOMAIN
} from './nyc-taxi-data';

const hourLabel = (value: number) => formatTaxiTime(value).replace(/ \d+ Jan/, '');

/** The cartouche of one step: the claim, the variable and method, and the sample chip. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Sample'] as const
});

/** Place names from the NYC gazetteer, drawn above the data. Zooms are lowered for the city frame. */
const places = (ids: readonly string[]) =>
  labelsFor(NYC, ids, {
    midtown: {minZoom: 9},
    'lower-manhattan': {minZoom: 9},
    'central-park': {minZoom: 9, tone: 'muted'},
    'penn-station': {minZoom: 11.2},
    'grand-central': {minZoom: 11.2}
  });

/** The ghost swatch of the legends: the trips a brush removed. */
const GHOST_ENTRY = {
  color: [GHOST_INK[0], GHOST_INK[1], GHOST_INK[2], 255] as const,
  label: 'Filtered out by a brush',
  shape: 'dot' as const
};

/** Camera frames derived from gazetteer places (no typed coordinates). */
const MIDTOWN_AND_DOWNTOWN = getPlacesBounds(['midtown', 'lower-manhattan'], 0.02);
const CITY_AND_AIRPORTS = getPlacesBounds(['midtown', 'jfk', 'lga'], 0.05);
const JFK_CENTER = getPlaceCenter('jfk');

export default defineScene<NycTaxiCrossfilterOptions>({
  id: 'nyc-taxi-crossfilter',
  title: 'Which trips make a taxi city?',
  chapter: 'flows',
  order: 2,
  summary:
    'GPUCrossfilter over real New York taxi trips: brush the hour, distance, fare, party size or a map rectangle and watch the map, the linked charts and the statistics update on the GPU. Overplotting, brushing and linking, and why a skewed fare needs quantile classes.',
  contributors: ['GPUCrossfilter'],
  datasets: [
    {id: 'poopdeck-nyc-taxi', role: 'yellow-taxi trips (origin, destination, time, fare)'}
  ],
  initialView: {...CITY_FRAMES.nyc},

  options: [
    {
      kind: 'preset',
      id: 'hourPresets',
      label: 'Hour windows',
      group: 'Brushes',
      help: 'Writes the pickup-hour brush: the first hours of the new year, a Friday morning rush, or every hour.',
      presets: [
        {label: 'New Year 00-02', values: {hours: [0, 2]}},
        {label: 'Friday 07-10', values: {hours: [31, 34]}},
        {label: 'All hours', values: {hours: [HOUR_DOMAIN[0], HOUR_DOMAIN[1]]}}
      ]
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Pickup hours',
      group: 'Brushes',
      apply: 'param',
      min: HOUR_DOMAIN[0],
      max: HOUR_DOMAIN[1],
      step: 0.5,
      default: [HOUR_DOMAIN[0], HOUR_DOMAIN[1]],
      format: hourLabel,
      help: 'A half-open range [from, to) on the pickup time, in hours since midnight on Thursday 1 January. Setting it writes five words; the covered ends of the slider mean no bound.'
    },
    {
      kind: 'range',
      id: 'distance',
      label: 'Trip distance',
      group: 'Brushes',
      apply: 'param',
      min: DISTANCE_DOMAIN[0],
      max: DISTANCE_DOMAIN[1],
      step: 0.5,
      default: [DISTANCE_DOMAIN[0], DISTANCE_DOMAIN[1]],
      unit: 'mi',
      help: 'Half-open range on the metered distance. At the right end of the slider the brush is unbounded, so trips over 15 miles stay selected.'
    },
    {
      kind: 'range',
      id: 'fare',
      label: 'Fare',
      group: 'Brushes',
      apply: 'param',
      min: FARE_DOMAIN[0],
      max: FARE_DOMAIN[1],
      step: 1,
      default: [FARE_DOMAIN[0], FARE_DOMAIN[1]],
      unit: 'USD',
      help: 'Half-open range on the metered fare, without tips. Unbounded at the right end of the slider.'
    },
    {
      kind: 'range',
      id: 'passengers',
      label: 'Passengers',
      group: 'Brushes',
      apply: 'param',
      min: PASSENGER_DOMAIN[0],
      max: PASSENGER_DOMAIN[1],
      step: 1,
      default: [PASSENGER_DOMAIN[0], PASSENGER_DOMAIN[1]],
      help: 'Inclusive range of party sizes. The dimension is a range over small integers, and the bar chart is a group view of the same column.'
    },
    {
      kind: 'select',
      id: 'area',
      label: 'Map area',
      group: 'Brushes',
      apply: 'param',
      default: 'none',
      help: 'A rectangular `bounds` brush on the pickup position. Choose a preset, or draw your own with Shift-drag.',
      options: [
        {value: 'none', label: 'Whole city'},
        {value: 'midtown', label: 'Midtown'},
        {value: 'downtown', label: 'Lower Manhattan'},
        {value: 'jfk', label: 'JFK airport'},
        {value: 'laguardia', label: 'LaGuardia airport'},
        {value: 'drawn', label: 'Drawn on the map'}
      ]
    },
    {
      kind: 'toggle',
      id: 'brushMap',
      label: 'Drag to brush the map',
      group: 'Brushes',
      apply: 'param',
      default: false,
      help: 'While on, dragging draws a rectangle instead of panning. Shift-drag always draws one, so you can leave this off.'
    },
    {
      kind: 'button',
      id: 'clearAll',
      label: 'Clear every brush',
      group: 'Brushes',
      help: 'Calls `clearAll()` and resets the sliders and the day switch: every trip is selected again.'
    },
    {
      kind: 'select',
      id: 'day',
      label: 'Pickup day',
      group: 'Live rows',
      apply: 'param',
      default: 'all',
      help: 'Rows of a day that is switched off become dead in the `liveMask`: they leave every histogram, count and visible list, even the self-excluding ones. Rewriting the mask never recompiles.',
      options: [
        {value: 'all', label: 'Both days'},
        {value: 'jan1', label: 'Thursday 1 January only'},
        {value: 'jan2', label: 'Friday 2 January only (to 15:00)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'selfExclude',
      label: 'Charts ignore their own brush',
      group: 'Live rows',
      apply: 'compile',
      default: true,
      help: 'On, each chart shows the rows that pass every other brush, so you keep seeing the distribution you are cutting. Off (`includeOwnSelection: true`) a chart shows only what its own brush keeps. Compile-time: the effective masks change.'
    },
    {
      kind: 'select',
      id: 'show',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'pickups',
      display: 'segmented',
      help: 'Which end of the selected trips to draw. Both ends use the same visible-id list; amber is the pickup, sky blue the drop-off.',
      options: [
        {value: 'pickups', label: 'Pickups'},
        {value: 'dropoffs', label: 'Drop-offs'},
        {value: 'both', label: 'Both ends'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour points by',
      group: 'Display',
      apply: 'param',
      default: 'none',
      disabledWhen: state => state.show === 'both',
      help: 'The attribute a point is coloured by, read from a per-trip buffer through the visible-id list. Fare, distance and party size are classed with breaks fixed at load; the hour of day is a cycle.',
      options: [
        {value: 'none', label: 'Nothing (one colour)'},
        {value: 'fare', label: 'Fare (classes)'},
        {value: 'distance', label: 'Trip distance (classes)'},
        {value: 'hour', label: 'Hour of day (cycle)'},
        {value: 'passengers', label: 'Passengers (classes)'}
      ]
    },
    {
      kind: 'select',
      id: 'fareClasses',
      label: 'Fare classes',
      group: 'Display',
      apply: 'param',
      default: 'quantile',
      display: 'segmented',
      disabledWhen: state => state.colorBy !== 'fare' || state.show === 'both',
      help: 'Quantile classes hold the same number of trips each; equal intervals split the fare range into equal steps. The swipe draws both over the same trips.',
      options: [
        {value: 'quantile', label: 'Quantiles'},
        {value: 'equal', label: 'Equal'},
        {value: 'swipe', label: 'Swipe both'}
      ]
    },
    {
      kind: 'select',
      id: 'blending',
      label: 'Dot blending',
      group: 'Display',
      apply: 'param',
      default: 'additive',
      display: 'segmented',
      help: 'Additive: overlapping dots add their light, so stacks glow brighter. Normal: each dot paints over the last, so a stack of a hundred looks like a stack of five (overplotting).',
      options: [
        {value: 'normal', label: 'Normal'},
        {value: 'additive', label: 'Additive'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showLinks',
      label: 'Trip links',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.show !== 'both',
      help: 'Draws a faint straight line from every selected pickup to its drop-off, while the selection is small enough to read (about a tenth of the trips).'
    },
    {
      kind: 'toggle',
      id: 'showFiltered',
      label: 'Show filtered-out trips',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every trip a brush removed as a faint grey ghost under the selection, so you can see what the brushes took away.'
    },
    {
      kind: 'slider',
      id: 'pointScale',
      label: 'Point size',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 2,
      step: 0.25,
      default: 1,
      help: 'Scales the radius of every dot. Dense areas saturate; shrink it or zoom in to separate trips.'
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Trips on the GPU',
      format: 'integer',
      emphasis: 'tile',
      help: 'Trips uploaded once; each brush re-tests every row.'
    },
    {
      id: 'inView',
      label: 'Trips in this view',
      format: 'percent',
      help: 'Share of all pickups that fall inside the map as framed now, counted once the camera settles.'
    },
    {
      id: 'selected',
      label: 'Selected trips',
      format: 'integer',
      emphasis: 'tile',
      help: 'Rows passing every brush, from the `count` view.'
    },
    {id: 'share', label: 'Share of all trips', format: 'percent'},
    {
      id: 'fareShare',
      label: 'Share of all fares',
      format: 'percent',
      help: 'Sum of the fares of the selected trips over the sum of every fare. No tips.'
    },
    {id: 'window', label: 'Pickup window'},
    {
      id: 'fareMean',
      label: 'Mean fare',
      help: 'Sum of fares over the selected trips (a group view) divided by their count. No tips.'
    },
    {id: 'distanceMean', label: 'Mean distance'},
    {
      id: 'perMileDelta',
      label: 'Fare per mile, selection vs all',
      help: 'Total fares over total miles of the selection, then of every trip. Higher when the streets are slow.'
    },
    {id: 'passengersMean', label: 'Mean party size'},
    {
      id: 'modalFare',
      label: 'Most common fare',
      help: 'The fullest half-dollar bin of the fare histogram of the selected trips.'
    },
    {
      id: 'equalMajor',
      label: 'Trips in the busiest equal class',
      format: 'percent',
      help: 'Six equal-width fare classes over the whole data range, counted over every trip.'
    },
    {
      id: 'quantileMajor',
      label: 'Trips in the busiest quantile class',
      format: 'percent',
      help: 'Six quantile fare classes, counted over every trip. Ties in half-dollar fares make the classes slightly uneven.'
    },
    {id: 'hourChart', label: 'Trips by pickup hour', kind: 'chart'},
    {id: 'distanceChart', label: 'Trips by distance', kind: 'chart'},
    {id: 'fareChart', label: 'Trips by fare, with class edges', kind: 'chart'},
    {id: 'passengerChart', label: 'Trips by party size', kind: 'chart'},
    {id: 'funnelChart', label: 'Trips passing each chart’s view', kind: 'chart'},
    {
      id: 'views',
      label: 'Dimensions and views',
      hood: true,
      help: 'The map rectangle plus four ranges; three histograms, four groups, a count and the visible-id list.'
    },
    {
      id: 'readbackBytes',
      label: 'Bytes read back per brush',
      format: 'bytes',
      hood: true,
      help: 'Histogram bins, group sums and the count: the only data that crosses to the CPU.'
    },
    {
      id: 'gpuBytes',
      label: 'Bytes held on the GPU',
      format: 'bytes',
      hood: true,
      help: 'Every column, position and compacted list the brushes act on.'
    }
  ],

  pipeline: [
    {
      id: 'brush',
      label: 'Brush write',
      detail: 'A brush is five words in a small parameter buffer'
    },
    {
      id: 'mask',
      label: 'Row masks',
      detail: 'One pass tests every row against every dimension and writes a bitmask'
    },
    {
      id: 'views',
      label: 'Linked views',
      detail: 'Histograms and groups add atomically, each on its all-except-own mask'
    },
    {
      id: 'compact',
      label: 'Compaction',
      detail: 'The visible ids and the indirect draw count: the map draws exactly the selection'
    },
    {id: 'draw', label: 'Draw', detail: 'Only a few hundred numbers are read back'}
  ],

  legends: (state, data) => {
    const tables = data['tables'] as TaxiClassTables | undefined;
    const ghost = {...GHOST_ENTRY};
    if (state.show === 'both') {
      return [
        {
          kind: 'categories',
          title: 'End of the trip',
          entries: [
            {color: PICKUP_INK.dark, label: 'Pickup', shape: 'dot'},
            {color: DROPOFF_INK.dark, label: 'Drop-off', shape: 'dot'},
            ghost
          ],
          note: 'Faint lines join the two ends of every selected trip.'
        }
      ];
    }
    if (state.colorBy === 'none' || !tables) {
      return [
        {
          kind: 'categories',
          title: 'Trips',
          entries: [{color: PICKUP_INK.dark, label: 'Selected pickup', shape: 'dot'}, ghost],
          note:
            state.blending === 'additive'
              ? 'Additive: where dots stack, their light adds up.'
              : 'Normal blending: stacked dots hide each other.'
        }
      ];
    }
    const legends: LegendSpec[] = [];
    switch (state.colorBy) {
      case 'fare':
        if (state.fareClasses === 'swipe') {
          legends.push(
            getClassTableLegend(tables.fareEqual, {
              title: 'Fare, equal intervals (left)',
              counts: tables.fareEqualCounts,
              layout: 'list'
            }),
            getClassTableLegend(tables.fareQuantile, {
              title: 'Fare, quantiles (right)',
              counts: tables.fareQuantileCounts,
              layout: 'list'
            })
          );
        } else {
          legends.push(
            getClassTableLegend(
              state.fareClasses === 'equal' ? tables.fareEqual : tables.fareQuantile,
              {
                title: 'Metered fare'
              }
            )
          );
        }
        break;
      case 'distance':
        legends.push(getClassTableLegend(tables.distance, {title: 'Trip distance'}));
        break;
      case 'passengers':
        legends.push(getClassTableLegend(tables.passengers, {title: 'Party size'}));
        break;
      default:
        legends.push({
          kind: 'cyclic',
          title: 'Pickup, hour of day',
          ramp: 'romao',
          labels: ['00', '06', '12', '18'],
          note: 'A 24 h cycle: Thursday and Friday overlay, so 23:00 and 01:00 are neighbours.'
        });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';

const graph = new GPUCommandGraph(device, {id: 'taxi'});
const filter = new GPUCrossfilter(graph, {
  dimensions: [
    {id: 'map', kind: 'bounds', x: pickupX, y: pickupY},
    {id: 'hour', kind: 'range', input: hour, exclusiveMaximum: true},
    {id: 'distance', kind: 'range', input: distance, exclusiveMaximum: true},
    {id: 'fare', kind: 'range', input: fare, exclusiveMaximum: true},
    {id: 'passengers', kind: 'range', input: passengers}
  ],
  liveMask,                      // nonzero = live; dead rows never reach a view
  views: [
    {id: 'hours', kind: 'histogram', dimension: 'hour', input: hour, domain: [0, 39],
     output: hourBins${state.selfExclude ? '' : ',\n     includeOwnSelection: true'}},
    {id: 'fares', kind: 'histogram', dimension: 'fare', input: fare, domain: [0, 60],
     output: fareBins},            // 120 half-dollar bins: the modal fare is exact
    {id: 'party', kind: 'group', keys: passengerKeys, output: partyCounts},
    {id: 'selected', kind: 'count', output: selectedCount},
    {id: 'visible', kind: 'visibility', output: visibleIds, count: drawInstanceCount}
  ]
});
filter.addToGraph(graph);
const compiled = graph.compile();   // once

// a brush is a five-word write; then encode again and read the small outputs back
filter.setRange('hour', [31, 34]);
filter.setBounds('map', [minX, minY, maxX, maxY]);
filter.clear('fare');
compiled.encode(commandEncoder, {parameters: undefined});

// the map draws the compacted ids: no CPU round trip
new SpatialAnalysisPointLayer({positions, ids: visibleIds, drawCommands: indirectDraw,
  blending: '${state.blending}', classBreaks: fareBreaks, classColors});   // breaks fixed at load`,

  about: {
    what: 'Previously: the flow map drew straight trips between zones. Next: where trips begin, as a density. `GPUCrossfilter` links brushes and views over the same GPU-resident rows. Each dimension (a range on one column, or a rectangle on two) turns into a per-row mask; the controller intersects them, builds the "all brushes except this one" mask each chart needs, and runs histograms, group statistics, a count and a compaction of the visible row ids, all as compute passes.',
    why: 'Exploring a table with several attributes at once ("short trips, at night, with groups") is a question about intersections. When the intersection is recomputed in a few milliseconds, the analyst can follow hunches instead of writing queries. On the map the same idea is overplotting, focus plus context, and the choice of classes for a skewed variable.',
    howToRead:
      'Bright dots are the selected trips; the faint grey ghost is what the brushes removed. Each chart shows the rows that pass the other brushes. **Sample bias:** yellow cabs only, a sample of the TLC records, 38 hours from New Year 2015 (the second day ends mid-afternoon); drop-off times are the routed OSRM duration, not the metered one. Chart axes clip at their ends, while the brushes do not.'
  },

  // Night ground in both page themes: additive light needs a dark ground.
  basemap: ground('night'),
  furniture: {
    title: cartouche('Which trips make a taxi city?', 'Yellow-taxi trips, 1-2 Jan 2015'),
    scaleBar: {units: 'metric'},
    credit: joinCredits(FLOW_CREDITS.nycTaxi, FLOW_CREDITS.osrmRoutes, CREDITS.carto),
    caveat: 'Yellow cabs only: a map of where taxis were hailed.'
  },

  create: async ctx =>
    (await import('./nyc-taxi-crossfilter.compute')).createNycTaxiCrossfilter(ctx),

  story: [
    {
      id: 'glow',
      title: 'A city of dots',
      headline: 'Rides saturate the core of the city',
      textAlternative:
        'Dark map of New York covered in amber dots, a solid mass in Midtown and Lower Manhattan, thinning toward the outer boroughs and the two airports.',
      body: 'Each dot is one yellow-taxi pickup, **{{rows}}** of them. With normal blending the last dot drawn hides the rest, so Midtown is a solid mass, and **{{inView}}** of all trips are in this view. Flip **Dot blending** to additive: overlapping dots add their light and the density appears. Which streets glow next?\n\n*Overplotting hides density.*',
      optionsMode: 'fresh',
      options: {blending: 'normal', colorBy: 'none', show: 'pickups'},
      controls: ['blending'],
      readouts: ['rows', 'inView'],
      stage: 'draw',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1400},
      furniture: {title: cartouche('Which trips make a taxi city?', 'One dot per pickup')},
      annotations: places([
        'midtown',
        'lower-manhattan',
        'brooklyn',
        'queens',
        'jfk',
        'lga',
        'central-park'
      ])
    },
    {
      id: 'brush-the-clock',
      title: 'Brush the clock',
      headline: 'New Year night looks nothing like Friday morning',
      textAlternative:
        'Manhattan in glowing dots coloured by hour of day, with the trips outside the chosen hours left as a faint grey ghost.',
      body: 'Pick a window in **Hour windows** or drag **Pickup hours**: the trips inside stay bright, the rest fade to a ghost, and every chart redraws. This window holds **{{selected}}** trips, {{share}} of all. Fare per mile, window against all trips: **{{perMileDelta}}**. A chart ignores its own brush, so the hour chart still shows every hour.\n\n*Brushing and linking: focus on a few trips, keep the rest as context.*',
      optionsMode: 'fresh',
      options: {blending: 'additive', colorBy: 'hour', hours: [0, 2]},
      controls: ['hourPresets', 'hours'],
      readouts: ['selected', 'share', 'perMileDelta', 'hourChart'],
      stage: 'brush',
      camera: {bounds: MIDTOWN_AND_DOWNTOWN, transitionMs: 1600},
      furniture: {title: cartouche('Which hours light up Manhattan?', 'Pickups by hour of day')},
      annotations: places(['midtown', 'lower-manhattan', 'penn-station', 'grand-central'])
    },
    {
      id: 'long-trips',
      title: 'A few long trips',
      headline: 'A few long trips carry much of the fare',
      textAlternative:
        'Amber pickups and sky-blue drop-offs of long trips joined by faint straight lines, many ending at the two airports.',
      body: 'Brush **Trip distance** to long trips and keep **Trip links** on: each faint line joins a pickup to its drop-off, sky blue at the far end. These trips are **{{share}}** of all rides but **{{fareShare}}** of the fares; per mile, selection against all trips, they cost **{{perMileDelta}}**. Watch where the lines end.\n\n*A selection can matter more than its count.*',
      optionsMode: 'fresh',
      options: {blending: 'additive', distance: [8, 15], show: 'both', colorBy: 'none'},
      controls: ['distance', 'showLinks'],
      readouts: ['share', 'fareShare', 'perMileDelta'],
      stage: 'compact',
      camera: {bounds: CITY_AND_AIRPORTS, transitionMs: 1800},
      furniture: {title: cartouche('Which trips carry the money?', 'Long trips, both ends')},
      annotations: places(['midtown', 'jfk', 'lga'])
    },
    {
      id: 'airports',
      title: 'Draw a rectangle',
      headline: 'A rectangle on the map is a filter too',
      textAlternative:
        'A dashed rectangle around JFK airport with pickups inside it coloured by fare class, almost all in the highest class.',
      body: 'The map is a dimension too. **Map area** keeps pickups inside a rectangle, tested for every trip on the GPU; the dashed frame shows its size and the scale bar is ticked at its half-width. Here **{{selected}}** pickups, {{share}} of all, are coloured by fare quantiles, and the most common fare is **{{modalFare}}**. Switch on **Drag to brush the map** to draw your own.',
      optionsMode: 'fresh',
      options: {
        blending: 'additive',
        area: 'jfk',
        colorBy: 'fare',
        fareClasses: 'quantile',
        show: 'pickups'
      },
      controls: ['area', 'brushMap'],
      readouts: ['selected', 'share', 'fareMean', 'modalFare'],
      stage: 'mask',
      camera: {longitude: JFK_CENTER[0], latitude: JFK_CENTER[1], zoom: 11.6, transitionMs: 1800},
      furniture: {
        title: cartouche(
          'What does an airport pickup cost?',
          'Pickups in a rectangle, fare classes'
        )
      },
      annotations: places(['jfk', 'lga', 'queens'])
    },
    {
      id: 'classes',
      title: 'Classes follow the data',
      headline: 'Equal steps hide the typical fare',
      textAlternative:
        'Two maps of the same pickups split by a swipe divider: on the left one flat dark colour, on the right six distinct fare classes.',
      body: 'Fares are skewed, so the classes matter. Left of the divider, six equal intervals put **{{equalMajor}}** of all trips in one class and the map says nothing. Right, quantiles put at most **{{quantileMajor}}** in a class. Both sets of breaks were computed once from every trip and never move with a brush. The choropleth story compares more methods.\n\n*Classes should follow the data, not the axis.*',
      optionsMode: 'fresh',
      options: {blending: 'additive', colorBy: 'fare', fareClasses: 'swipe'},
      controls: ['fareClasses'],
      readouts: ['equalMajor', 'quantileMajor', 'fareChart'],
      stage: 'draw',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1600},
      compare: {mode: 'swipe', labels: ['Equal intervals', 'Quantiles']},
      furniture: {title: cartouche('Which classes show the fares?', 'Fare classes, fixed at load')},
      annotations: places(['midtown', 'jfk', 'lga', 'brooklyn'])
    },
    {
      id: 'linked-views',
      title: 'Linked views',
      headline: 'Every chart ignores its own brush',
      textAlternative:
        'Manhattan in amber dots for a Friday morning and larger parties, beside a funnel of bars counting the trips passing all brushes but one.',
      body: 'Brush **Pickup hours** and **Passengers** together. The funnel counts trips passing every brush except the one named: each chart sees all the others. Switch **Charts ignore their own brush** off and the bars collapse onto the selection. Only **{{readbackBytes}}** come back from **{{gpuBytes}}** on the GPU.\n\n*Brushing and linking: select in one view, read the effect in all.* Now combine your own.',
      optionsMode: 'fresh',
      options: {
        blending: 'additive',
        colorBy: 'none',
        hours: [31, 34],
        passengers: [4, 6],
        selfExclude: true
      },
      controls: ['selfExclude', 'hours', 'passengers'],
      readouts: ['funnelChart', 'views', 'readbackBytes', 'gpuBytes'],
      stage: 'views',
      camera: {...CITY_FRAMES.nyc, transitionMs: 1600},
      furniture: {
        title: cartouche('What does each brush remove?', 'Trips passing all brushes but one')
      },
      annotations: places(['midtown', 'jfk', 'lga'])
    }
  ]
});
