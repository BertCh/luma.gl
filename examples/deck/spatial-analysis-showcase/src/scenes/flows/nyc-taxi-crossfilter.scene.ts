// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-crossfilter.md?raw';
import type {NycTaxiCrossfilterOptions} from './nyc-taxi-crossfilter.compute';
import {
  DISTANCE_DOMAIN,
  FARE_DOMAIN,
  formatTaxiTime,
  HOUR_DOMAIN,
  PASSENGER_DOMAIN,
  TAXI_COLOR_RANGES
} from './nyc-taxi-data';

const COLOR_LEGENDS = {
  fare: {title: 'Metered fare', unit: 'USD', format: (value: number) => `$${value.toFixed(0)}`},
  distance: {title: 'Trip distance', unit: 'miles', format: (value: number) => value.toFixed(0)},
  time: {title: 'Pickup time', unit: undefined, format: (value: number) => formatTaxiTime(value)},
  passengers: {title: 'Passengers', unit: undefined, format: (value: number) => value.toFixed(0)}
} as const;

const hourLabel = (value: number) => formatTaxiTime(value).replace(' Jan', '');

export default defineScene<NycTaxiCrossfilterOptions>({
  id: 'nyc-taxi-crossfilter',
  title: 'Brush 440,000 taxi trips',
  chapter: 'flows',
  order: 2,
  summary:
    'The first GPUCrossfilter scene: brush the hour, distance, fare, passengers or a map rectangle over 440,000 real New York taxi trips and watch the map, four charts and the fare statistics update on the GPU.',
  contributors: ['GPUCrossfilter'],
  datasets: [
    {id: 'poopdeck-nyc-taxi', role: '440,000 yellow-taxi trips (origin, destination, time, fare)'}
  ],
  initialView: {longitude: -73.96, latitude: 40.735, zoom: 10.7},

  options: [
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
      help: 'Calls `clearAll()` and resets the sliders and the day switch: the full 440,000 rows are selected again.'
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
      label: 'Histograms ignore their own brush',
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
      help: 'Which end of the selected trips to draw. Both ends use the same visible-id list; orange is the pickup, cyan the dropoff.',
      options: [
        {value: 'pickups', label: 'Pickups'},
        {value: 'dropoffs', label: 'Dropoffs'},
        {value: 'both', label: 'Both ends'}
      ]
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour points by',
      group: 'Display',
      apply: 'param',
      default: 'fare',
      disabledWhen: state => state.show === 'both',
      help: 'The attribute a point is coloured by, read from a per-trip buffer through the visible-id list.',
      options: [
        {value: 'fare', label: 'Fare'},
        {value: 'distance', label: 'Trip distance'},
        {value: 'time', label: 'Pickup time'},
        {value: 'passengers', label: 'Passengers'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.show === 'both',
      help: 'Perceptually uniform ramps; cividis is optimised for colour-vision deficiency.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Point radius',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 1.25,
      unit: 'px',
      help: 'Radius of each selected trip in pixels. Dense areas saturate; shrink it or zoom in to separate trips.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Point opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.55,
      help: 'Lower opacity makes overlapping points add up, which reads as density.'
    },
    {
      kind: 'toggle',
      id: 'showFiltered',
      label: 'Show filtered-out trips',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every trip as a faint grey dot under the selection, so you can see what the brushes removed.'
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Rows on the GPU',
      format: 'integer',
      help: 'Trips uploaded once; each brush re-tests every row.'
    },
    {id: 'dimensions', label: 'Brush dimensions'},
    {id: 'views', label: 'Linked views'},
    {
      id: 'selected',
      label: 'Selected trips',
      format: 'integer',
      help: 'Rows passing every brush, from the `count` view.'
    },
    {id: 'share', label: 'Share of all trips', format: 'percent'},
    {id: 'window', label: 'Pickup window'},
    {
      id: 'fareMean',
      label: 'Mean fare',
      help: 'Sum of fares over the selected trips (a group view) divided by their count. No tips.'
    },
    {
      id: 'fareTotal',
      label: 'Total fares',
      help: 'Sum of the fares of the selected trips, from the `sum` group view.'
    },
    {id: 'distanceMean', label: 'Mean distance'},
    {
      id: 'perMile',
      label: 'Fare per mile',
      help: 'Total fares over total miles of the selection. Higher when the streets are slow.'
    },
    {id: 'passengersMean', label: 'Mean party size'},
    {id: 'hourChart', label: 'Trips by pickup hour', kind: 'chart'},
    {id: 'distanceChart', label: 'Trips by distance', kind: 'chart'},
    {id: 'fareChart', label: 'Trips by fare', kind: 'chart'},
    {id: 'passengerChart', label: 'Trips by party size', kind: 'chart'},
    {id: 'partyFareChart', label: 'Mean fare by party size', kind: 'chart'}
  ],

  legends: state => {
    if (state.show === 'both') {
      return [
        {
          kind: 'categories',
          title: 'End of the trip',
          entries: [
            {color: [255, 170, 60, 255], label: 'Pickup'},
            {color: [70, 215, 255, 255], label: 'Dropoff'}
          ],
          note: 'Grey dots are trips the brushes removed.'
        }
      ];
    }
    const spec = COLOR_LEGENDS[state.colorBy];
    return [
      {
        kind: 'ramp',
        title: spec.title,
        ramp: state.ramp,
        extent: TAXI_COLOR_RANGES[state.colorBy],
        unit: spec.unit,
        format: spec.format
      }
    ];
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
    {id: 'party', kind: 'group', keys: passengerKeys, output: partyCounts},
    {id: 'fares', kind: 'group', keys: passengerKeys, operation: 'sum', values: fare, output: fareSums},
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
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUCrossfilter` links brushes and views over the same GPU-resident rows. Each dimension (a range on one column, or a rectangle on two) turns into a per-row mask; the controller intersects them, builds the "all brushes except this one" mask each chart needs, and runs histograms, group statistics, a count and a compaction of the visible row ids, all as compute passes.',
    why: 'Exploring a table with several attributes at once ("short trips, at night, with groups") is a question about intersections. When the intersection is recomputed in a few milliseconds, the analyst can follow hunches instead of writing queries.',
    howToRead:
      'Bright dots are the selected trips, coloured by the attribute in the legend; grey dots are rows the brushes removed. Each chart shows the rows that pass the other brushes, with the bars inside its own brush highlighted. The numbers on the left describe exactly the selected trips.'
  },

  create: async ctx =>
    (await import('./nyc-taxi-crossfilter.compute')).createNycTaxiCrossfilter(ctx),

  story: storyFromMarkdown<NycTaxiCrossfilterOptions>(narrative, {
    'the-question': {
      controls: ['colorBy', 'show'],
      readouts: ['selected', 'fareMean'],
      options: {colorBy: 'fare', show: 'pickups'},
      camera: {
        longitude: -73.96,
        latitude: 40.735,
        zoom: 10.7,
        pitch: 0,
        bearing: 0,
        transitionMs: 1400
      }
    },
    'brush-time': {
      controls: ['hours', 'day'],
      readouts: ['selected', 'window', 'fareMean', 'perMile'],
      options: {hours: [31, 34]},
      camera: {longitude: -73.975, latitude: 40.745, zoom: 11.4, transitionMs: 1600}
    },
    'long-trips': {
      controls: ['distance', 'show'],
      readouts: ['selected', 'share', 'fareMean'],
      options: {hours: [0, 39], distance: [8, 15], show: 'both'},
      camera: {longitude: -73.88, latitude: 40.71, zoom: 9.7, transitionMs: 1800},
      callout: {coordinate: [-73.7822, 40.6446], text: 'JFK'}
    },
    airports: {
      controls: ['area', 'brushMap', 'show'],
      readouts: ['selected', 'fareMean', 'distanceMean'],
      options: {distance: [0, 15], area: 'jfk', show: 'pickups', colorBy: 'fare'},
      camera: {longitude: -73.79, latitude: 40.65, zoom: 11.3, transitionMs: 1800}
    },
    'party-size': {
      controls: ['passengers', 'hours'],
      readouts: ['selected', 'share', 'passengersMean', 'fareMean'],
      options: {area: 'none', passengers: [4, 6], colorBy: 'passengers', show: 'pickups'},
      camera: {longitude: -73.96, latitude: 40.735, zoom: 10.7, transitionMs: 1800}
    },
    limits: {
      controls: ['selfExclude', 'day', 'showFiltered'],
      readouts: ['rows', 'dimensions', 'views'],
      options: {passengers: [1, 6], hours: [31, 34]}
    }
  })
});
