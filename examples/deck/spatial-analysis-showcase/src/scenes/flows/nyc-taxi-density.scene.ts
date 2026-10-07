// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './nyc-taxi-density.md?raw';
import type {NycTaxiDensityOptions} from './nyc-taxi-density.compute';
import {formatTaxiTime} from './nyc-taxi-data';

const hourLabel = (value: number) => formatTaxiTime(value).replace(' Jan', '');

const MEASURE_TITLES = {
  trips: 'trips',
  fare: 'USD of fares',
  passengers: 'passengers',
  distance: 'miles'
} as const;

export default defineScene<NycTaxiDensityOptions>({
  id: 'nyc-taxi-density',
  title: 'Pickups against dropoffs',
  chapter: 'flows',
  order: 4,
  summary:
    'GPUPointDensity over 880,000 taxi pickup and dropoff points: counts, fare sums and means, a signed balance map of where rides end minus where they start, and an hour mask, on a grid that follows the camera.',
  contributors: ['GPUPointDensity'],
  datasets: [{id: 'poopdeck-nyc-taxi', role: '440,000 trips as 880,000 pickup and dropoff points'}],
  initialView: {longitude: -73.97, latitude: 40.745, zoom: 10.9},

  options: [
    {
      kind: 'select',
      id: 'show',
      label: 'Show',
      group: 'Field',
      apply: 'param',
      default: 'pickups',
      help: 'Pickups, dropoffs, or the balance (dropoffs minus pickups). All three read one 880,000-point buffer: a mask picks the half and signed weights make the balance. Switching rewrites buffers only.',
      options: [
        {value: 'pickups', label: 'Pickups'},
        {value: 'dropoffs', label: 'Dropoffs'},
        {value: 'balance', label: 'Balance (dropoffs - pickups)'}
      ]
    },
    {
      kind: 'select',
      id: 'measure',
      label: 'Measure',
      group: 'Field',
      apply: 'param',
      default: 'trips',
      help: 'What each point contributes: one trip (a count), its fare, its passengers or its distance. A weights buffer is rewritten; the graph changes only between count and sum.',
      options: [
        {value: 'trips', label: 'Trips'},
        {value: 'fare', label: 'Fare (USD, no tips)'},
        {value: 'passengers', label: 'Passengers'},
        {value: 'distance', label: 'Trip distance (miles)'}
      ]
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic',
      group: 'Field',
      apply: 'compile',
      default: 'total',
      disabledWhen: state => state.show === 'balance' || state.measure === 'trips',
      help: 'Total sums the measure per cell; average divides by the number of points (a rate, such as mean fare). Compile-time: sum and mean are different graphs, both kept after first use.',
      options: [
        {value: 'total', label: 'Total per cell'},
        {value: 'average', label: 'Average per trip'}
      ]
    },
    {
      kind: 'select',
      id: 'binning',
      label: 'Cell shape',
      group: 'Grid',
      apply: 'compile',
      default: 'grid',
      help: 'Squares can be smoothed; hexagons have six equidistant neighbours. Compile-time binning.',
      options: [
        {value: 'grid', label: 'Square grid'},
        {value: 'hexagon', label: 'Hexagons'}
      ]
    },
    {
      kind: 'select',
      id: 'resolution',
      label: 'Resolution',
      group: 'Grid',
      apply: 'compile',
      default: 'medium',
      help: 'Number of cells across the screen (`gridSize`, compile-time). Finer grids show blocks; coarser grids show neighbourhoods.',
      options: [
        {value: 'coarse', label: 'Coarse (about 70 cells across)'},
        {value: 'medium', label: 'Medium (110 across)'},
        {value: 'fine', label: 'Fine (170 across)'}
      ]
    },
    {
      kind: 'select',
      id: 'smoothing',
      label: 'Smoothing',
      group: 'Grid',
      apply: 'param',
      default: 'gaussian',
      disabledWhen: state => state.binning === 'hexagon',
      help: 'A Gaussian blur of the cell field (square grid only). The kernel is a parameter buffer, so switching it or changing sigma never recompiles.',
      options: [
        {value: 'off', label: 'Off (raw cell values)'},
        {value: 'gaussian', label: 'Gaussian kernel'}
      ]
    },
    {
      kind: 'slider',
      id: 'sigma',
      label: 'Smoothing radius (sigma)',
      group: 'Grid',
      apply: 'param',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      default: 1,
      unit: 'cells',
      disabledWhen: state => state.binning === 'hexagon' || state.smoothing === 'off',
      help: 'Standard deviation of the Gaussian in cells. Larger values give a regional pattern; smaller ones keep single blocks.'
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hour of day',
      group: 'Time',
      apply: 'param',
      min: 0,
      max: 39,
      step: 0.5,
      default: [0, 39],
      format: hourLabel,
      help: 'Keeps points whose time is in [from, to), measured from midnight on Thursday 1 January. Pickups use the pickup time, dropoffs the dropoff time. The mask buffer is rewritten; nothing is recompiled.'
    },
    {
      kind: 'toggle',
      id: 'invertHours',
      label: 'Outside that window',
      group: 'Time',
      apply: 'param',
      default: false,
      help: 'Keeps the hours outside the range instead, which is how you ask for an evening or a night.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'Ramp of the field, and of the dropoff surplus in balance mode. All four are perceptually uniform; cividis is optimised for colour-vision deficiency.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'select',
      id: 'lossRamp',
      label: 'Pickup-surplus ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      disabledWhen: state => state.show !== 'balance',
      help: 'Ramp of the cells where more rides start than end, drawn by a second layer over the same field with the sign flipped.',
      options: [
        {value: 'cividis', label: 'Cividis'},
        {value: 'viridis', label: 'Viridis'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it to read street names under the field.'
    }
  ],

  readouts: [
    {
      id: 'points',
      label: 'Points on the GPU',
      format: 'integer',
      help: '440,000 pickups plus 440,000 dropoffs, uploaded once.'
    },
    {id: 'included', label: 'Passing the mask', help: 'Points kept by the Show and hour settings.'},
    {id: 'window', label: 'Time window'},
    {
      id: 'statisticUsed',
      label: 'Contributor statistic',
      help: 'The statistic the compiled graph computes for these settings.'
    },
    {id: 'grid', label: 'Grid'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Follows the zoom: the bounds buffer is rewritten every frame.'
    },
    {
      id: 'peak',
      label: 'Peak cell value',
      help: 'Maximum of the displayed field (and the minimum in balance mode), read back once the camera settles.'
    },
    {
      id: 'kernelRadius',
      label: 'Kernel radius',
      help: 'Three sigma, capped at the compile-time kernel of 8 cells.'
    },
    {id: 'valueChart', label: 'Cells by value', kind: 'chart'},
    {id: 'pulseChart', label: 'Pickups and dropoffs per hour', kind: 'chart'}
  ],

  legends: state => {
    const measure = MEASURE_TITLES[state.measure];
    if (state.show === 'balance') {
      return [
        {
          kind: 'ramp',
          id: 'density',
          title: `Dropoff surplus (${measure})`,
          ramp: state.ramp,
          extent: 'gpu',
          sqrtScale: true,
          format: value => value.toFixed(0)
        },
        {
          kind: 'ramp',
          id: 'loss',
          title: `Pickup surplus (${measure})`,
          ramp: state.lossRamp,
          extent: 'gpu',
          sqrtScale: true,
          format: value => value.toFixed(0)
        }
      ];
    }
    const average = state.measure !== 'trips' && state.statistic === 'average';
    const what = state.show === 'pickups' ? 'Pickups' : 'Dropoffs';
    return [
      {
        kind: 'ramp',
        id: 'density',
        title: average
          ? `Mean ${state.measure} per trip`
          : state.measure === 'trips'
            ? `${what} per cell`
            : `${what}: ${measure} per cell`,
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: !average,
        unit: state.binning === 'grid' && state.smoothing !== 'off' ? 'smoothed' : undefined,
        format: value => (value < 10 ? value.toFixed(1) : value.toFixed(0))
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUPointDensity} from '@luma.gl/experimental/gpu-spatial-analysis';

// pickups are rows 0..N-1 and dropoffs rows N..2N-1 of one float32x2 buffer
const graph = new GPUCommandGraph(device, {id: 'taxi-density'});
graph.add(
  new GPUPointDensity({
    positions,                         // 880,000 points, uploaded once
    mask,                              // uint32: which half, which hours (rewritten on change)
    weights,                           // ${state.show === 'balance' ? '-measure for pickups, +measure for dropoffs' : 'the measure per point'}
    bounds: bounds.importToGraph(graph), // rewritten from the camera every frame
    gridSize: ${state.binning === 'grid' ? '[110, 70]' : '[52, 38]'},
    binning: '${state.binning}',${
      state.binning === 'grid'
        ? `
    smoothing: {kernel, kernelWidth: 17, kernelHeight: 17, strategy: 'direct'},`
        : `
    hexagonRadius: hexagonRadius.importToGraph(graph),`
    }
    statistic: '${state.show === 'balance' ? 'sum' : state.measure === 'trips' ? 'count' : state.statistic === 'average' ? 'mean' : 'sum'}',
    output: {values, extent, histogram}
  })
);
const compiled = graph.compile();      // once per statistic, lattice and resolution
// every frame: write the small parameter buffers, then
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUPointDensity` bins points into a square grid or a hexagon lattice that covers the visible map, optionally convolves the field with a Gaussian kernel, and reduces it to an extent and a histogram, all in compute shaders. The statistic per cell is a count, a sum of a weight, or its mean.',
    why: 'Where rides start and where they end are different questions: the difference between them is the demand that moves people from one part of the city to another, and the mean fare shows how far demand travels. With signed weights one graph answers all of them.',
    howToRead:
      'Brighter cells hold more of the measure; the scale uses a square root so quiet neighbourhoods stay visible. In balance mode, warm cells are where more rides end than start and cool cells where more start than end. Empty cells are transparent. Yellow taxis only, so quiet cells in the outer boroughs mean few yellow taxis, not few people.'
  },

  create: async ctx => (await import('./nyc-taxi-density.compute')).createNycTaxiDensity(ctx),

  story: storyFromMarkdown<NycTaxiDensityOptions>(narrative, {
    'the-question': {
      controls: ['show', 'smoothing'],
      readouts: ['included', 'peak'],
      options: {
        show: 'pickups',
        measure: 'trips',
        smoothing: 'gaussian',
        sigma: 1,
        binning: 'grid',
        hours: [0, 39]
      },
      camera: {
        longitude: -73.97,
        latitude: 40.745,
        zoom: 10.9,
        pitch: 0,
        bearing: 0,
        transitionMs: 1400
      }
    },
    'follow-the-camera': {
      controls: ['resolution', 'binning', 'sigma'],
      readouts: ['cellSize', 'grid'],
      options: {resolution: 'fine'},
      camera: {longitude: -73.985, latitude: 40.752, zoom: 13.2, transitionMs: 1800}
    },
    balance: {
      controls: ['show', 'ramp', 'lossRamp'],
      readouts: ['peak'],
      options: {show: 'balance', resolution: 'medium', sigma: 0.75},
      camera: {longitude: -73.96, latitude: 40.745, zoom: 10.9, transitionMs: 1800}
    },
    fare: {
      controls: ['measure', 'statistic', 'show'],
      readouts: ['statisticUsed', 'peak'],
      options: {show: 'pickups', measure: 'fare', statistic: 'average', sigma: 0.75},
      camera: {longitude: -73.88, latitude: 40.7, zoom: 10.1, transitionMs: 1800}
    },
    clock: {
      controls: ['hours', 'invertHours', 'show'],
      readouts: ['included', 'window', 'pulseChart'],
      options: {measure: 'trips', statistic: 'total', hours: [0, 2], sigma: 1},
      camera: {longitude: -73.985, latitude: 40.735, zoom: 11.6, transitionMs: 1800}
    },
    limits: {
      controls: ['binning', 'resolution'],
      readouts: ['kernelRadius', 'valueChart'],
      options: {hours: [0, 39], binning: 'hexagon', resolution: 'medium'}
    }
  })
});
