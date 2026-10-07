// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {CHICAGO_VIEW, formatCategory} from './b1-nature-data';
import type {NatureDensityOptions} from './nature-density.compute';

const CATEGORY_NAMES = [
  'Plants',
  'Birds',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles',
  'Snails and mussels',
  'Fish',
  'Other life'
];

const WEIGHT_NAMES = {
  researchGrade: 'community-confirmed identifications',
  introduced: 'introduced-species observations',
  animal: 'animal observations'
} as const;

const hourLabel = (hour: number) => `${String(hour).padStart(2, '0')}:00`;

export default defineScene<NatureDensityOptions>({
  id: 'nature-density',
  title: 'Where does Chicago wildlife get noticed?',
  chapter: 'points',
  order: 1,
  summary:
    'A camera-following heat map of 43,557 iNaturalist observations of wild plants, animals and fungi in Chicago in 2023, binned on the GPU. Switch squares for hexagons, smooth with a separable Gaussian, and slice by hour, weekday and group.',
  contributors: ['GPUPointDensity'],
  datasets: [{id: 'chicago-nature', role: 'iNaturalist observations (2023)'}],
  initialView: {...CHICAGO_VIEW},

  options: [
    {
      kind: 'select',
      id: 'binning',
      label: 'Cell shape',
      group: 'Grid',
      apply: 'compile',
      default: 'grid',
      help: 'Squares can be smoothed; hexagons have six equidistant neighbours, so they avoid the grid-aligned look squares can imprint. Compile-time binning.',
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
      help: 'Number of cells across the screen (gridSize, compile-time). Finer grids show individual parks and paths; coarser grids show neighbourhoods and are more stable.',
      options: [
        {value: 'coarse', label: 'Coarse (about 70 cells across)'},
        {value: 'medium', label: 'Medium (110 across)'},
        {value: 'fine', label: 'Fine (170 across)'}
      ]
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic per cell',
      group: 'Grid',
      apply: 'compile',
      default: 'count',
      help: 'Count of observations, sum of a 0/1 attribute (how many are introduced species), or the mean of it (the share that are introduced). sum and mean read a weights buffer.',
      options: [
        {value: 'count', label: 'Count of observations'},
        {value: 'sum', label: 'Sum of the attribute'},
        {value: 'mean', label: 'Mean of the attribute (a rate)'}
      ]
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Attribute',
      group: 'Grid',
      apply: 'param',
      default: 'researchGrade',
      disabledWhen: state => state.statistic === 'count',
      help: 'Which 0/1 flag feeds sum and mean. It is a per-point float32 weights buffer, so switching is a buffer write.',
      options: [
        {value: 'researchGrade', label: 'Research grade (identification confirmed)'},
        {value: 'introduced', label: 'Introduced (non-native) taxon'},
        {value: 'animal', label: 'Animal (not a plant, fungus or other life)'}
      ]
    },
    {
      kind: 'select',
      id: 'sumAccumulation',
      label: 'Sum accumulation',
      group: 'Grid',
      apply: 'compile',
      default: 'workgroup',
      disabledWhen: state => state.statistic === 'count',
      help: 'How weights are added per cell: sorted per-workgroup partial sums (default) or one atomic add per point. Same answer; the atomic path is the better choice when a single cell holds a huge share of the points.',
      options: [
        {value: 'workgroup', label: 'Workgroup partial sums'},
        {value: 'atomic', label: 'Atomic adds'}
      ]
    },
    {
      kind: 'select',
      id: 'smoothing',
      label: 'Smoothing',
      group: 'Smoothing',
      apply: 'param',
      default: 'gaussian-separable',
      disabledWhen: state => state.binning === 'hexagon',
      help: 'A Gaussian blur of the cell field (square grid only). The kernel weights are a parameter buffer: switching or changing sigma never recompiles.',
      options: [
        {value: 'off', label: 'Off (raw counts)'},
        {
          value: 'gaussian-2d',
          label: '2D Gaussian (dense kernel)',
          help: 'One pass over a 17 × 17 kernel for every cell.'
        },
        {
          value: 'gaussian-separable',
          label: 'Separable Gaussian (two 1D passes)',
          help: 'A horizontal then a vertical pass: 34 taps instead of 289, same field.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'sigma',
      label: 'Smoothing radius (sigma)',
      group: 'Smoothing',
      apply: 'param',
      min: 0.5,
      max: 2.5,
      step: 0.25,
      default: 1.5,
      unit: 'cells',
      disabledWhen: state => state.binning === 'hexagon' || state.smoothing === 'off',
      help: 'Standard deviation of the Gaussian in cells. Larger values trade park-level detail for a regional pattern.'
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hour of day',
      group: 'Time and group',
      apply: 'param',
      min: 0,
      max: 24,
      step: 1,
      default: [0, 24],
      format: hourLabel,
      help: 'Keeps observations whose hour falls in [from, to). The mask buffer is rewritten on the CPU and the GPU skips masked points; the graph is not rebuilt.'
    },
    {
      kind: 'toggle',
      id: 'invertHours',
      label: 'Outside that window',
      group: 'Time and group',
      apply: 'param',
      default: false,
      help: 'Keeps the hours outside the range instead, which is how you ask for an evening window such as 19:00 to 05:00.'
    },
    {
      kind: 'select',
      id: 'dayType',
      label: 'Days',
      group: 'Time and group',
      apply: 'param',
      default: 'all',
      help: 'Weekdays (Mon to Fri), weekends (Sat and Sun) or every day of 2023.',
      options: [
        {value: 'all', label: 'Every day'},
        {value: 'weekdays', label: 'Weekdays'},
        {value: 'weekends', label: 'Weekends'}
      ]
    },
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Time and group',
      apply: 'param',
      default: 'all',
      help: 'Restrict the map to one iNaturalist group. Also a mask write.',
      options: [
        {value: 'all', label: 'All groups'},
        ...CATEGORY_NAMES.map(name => ({value: name, label: formatCategory(name)}))
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'All four are perceptually uniform; cividis is optimised for color-vision deficiency.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
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
      default: 0.85,
      help: 'Lower it to read street names under the field.'
    },
    {
      kind: 'toggle',
      id: 'showPoints',
      label: 'Show source points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws every observation as a faint dot under the heat map. Positions are precise, so single trees and ponds show as separate dots.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time 2D vs separable smoothing',
      group: 'Compare',
      help: 'Runs both smoothing graphs outside the frame (square grid) and reports the GPU time of each.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations on the GPU', format: 'integer', help: 'Uploaded once.'},
    {
      id: 'included',
      label: 'Passing the filters',
      help: 'Observations kept by the hour, day and group mask.'
    },
    {id: 'window', label: 'Hour window'},
    {id: 'grid', label: 'Grid'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Follows the zoom: the bounds buffer is rewritten every frame.'
    },
    {
      id: 'peak',
      label: 'Peak cell value',
      format: 'decimal',
      help: 'The maximum of the displayed field, read back once the camera settles.'
    },
    {
      id: 'kernelRadius',
      label: 'Kernel radius',
      help: 'Three sigma, capped at the compile-time kernel of 8 cells.'
    },
    {id: 'denseTime', label: '2D Gaussian graph'},
    {id: 'separableTime', label: 'Separable graph'},
    {id: 'speedup', label: 'Separable vs 2D'},
    {
      id: 'difference',
      label: 'Max difference',
      help: 'Largest absolute difference between the two smoothed fields, read back once the camera settles.'
    }
  ],

  legends: state => {
    const weightName = WEIGHT_NAMES[state.weight];
    const title =
      state.statistic === 'count'
        ? 'Observations per cell'
        : state.statistic === 'sum'
          ? `${weightName[0].toUpperCase()}${weightName.slice(1)} per cell`
          : `Share that are ${weightName}`;
    const smoothed = state.binning === 'grid' && state.smoothing !== 'off';
    return [
      {
        kind: 'ramp',
        id: 'density',
        title,
        ramp: state.ramp,
        extent: state.statistic === 'mean' ? [0, 1] : 'gpu',
        sqrtScale: state.statistic !== 'mean',
        unit:
          state.statistic === 'mean'
            ? smoothed
              ? 'smoothed share'
              : 'share'
            : smoothed
              ? 'smoothed'
              : undefined,
        format: value =>
          state.statistic === 'mean'
            ? `${(value * 100).toFixed(0)}%`
            : value.toFixed(value < 10 ? 1 : 0)
      }
    ];
  },

  snippet: state => {
    const weighted = state.statistic !== 'count';
    const sizes = {
      grid: {coarse: '[100, 62]', medium: '[160, 100]', fine: '[240, 150]'},
      hexagon: {coarse: '[48, 36]', medium: '[72, 52]', fine: '[110, 80]'}
    };
    return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUPointDensity} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'nature-density'});
graph.add(
  new GPUPointDensity({
    positions,                         // float32x2 meters, uploaded once
    mask,                              // uint32: hour/day/group filter, rewritten on change${
      weighted
        ? `
    weights: ${state.weight},            // float32 0/1 flag per observation`
        : ''
    }
    bounds: bounds.importToGraph(graph), // [minX, minY, maxX, maxY], rewritten every frame
    gridSize: ${sizes[state.binning][state.resolution]},
    binning: '${state.binning}',${
      state.binning === 'grid'
        ? `
    smoothing: {
      ${state.smoothing === 'gaussian-2d' ? 'kernel: kernel.importToGraph(graph), kernelWidth: 17, kernelHeight: 17' : state.smoothing === 'off' ? '// no smoothing: raw cell values' : 'separableKernel: {horizontal: line, vertical: line}, kernelWidth: 17, kernelHeight: 17'}
    },`
        : `
    hexagonRadius: hexagonRadius.importToGraph(graph),`
    }
    statistic: '${state.statistic}',${weighted ? `\n    sumAccumulation: '${state.sumAccumulation}',` : ''}
    output: {values, extent, histogram}
  })
);
const compiled = graph.compile();      // once
// every frame: write the small parameter buffers, then
compiled.encode(commandEncoder, {parameters: undefined});`;
  },

  about: {
    what: '`GPUPointDensity` bins points into a square grid or a hexagon lattice that covers the visible map, optionally convolves the field with a Gaussian kernel, and reduces it to an extent and a histogram, all in compute shaders. The statistic per cell is a count, a sum of a weight, or its mean.',
    why: 'Density answers "where is activity concentrated?" without choosing a boundary first. Because the grid follows the camera, the answer stays sharp at every zoom, and the same graph answers different questions (when, which group, what share) through small buffers.',
    howToRead:
      'Brighter cells hold more observations. The scale uses a square root so quiet neighbourhoods stay visible next to hot spots; the legend range is the real min and max read back from the GPU. Cells with a value of 0 are transparent. Observations follow observers, so a bright cell means many people looked and logged what they saw there, not only that much lives there.'
  },

  create: async ctx => (await import('./nature-density.compute')).createNatureDensity(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['statistic', 'smoothing'],
      readouts: ['included', 'peak'],
      title: 'Where does Chicago wildlife get noticed?',
      body: 'Naturalists logged 43,557 wild plants, animals and fungi inside the city on iNaturalist in 2023, 4,823 different taxa in all. **`GPUPointDensity`** counts the observations that fall in each cell of a grid laid over the screen, then blurs the counts slightly (**Smoothing**) so the picture reads as a surface instead of speckle.\n\nThe brightest ridge hugs the lakefront and the North Side parks; most of the West and South Sides are far quieter. Observations follow observers as much as wildlife, so read this as "where people look and log", not "where nature lives".',
      options: {binning: 'grid', smoothing: 'gaussian-separable', sigma: 2, statistic: 'count'}
    },
    {
      id: 'follow-the-camera',
      controls: ['resolution'],
      readouts: ['cellSize', 'peak'],
      title: 'The grid follows the screen',
      body: 'Zoom into the lakefront at Uptown. The grid is not a fixed raster: every frame the four-number **bounds** parameter buffer is rewritten from the camera, so the 110 × 70 cells always span what you see and the **Cell size** readout shrinks as you zoom. The **Resolution** select below sets how many cells there are.\n\nThe densest 500 m cell in the data sits around Montrose Point Bird Sanctuary and holds 3,455 observations, nearly 8% of the whole year. Nothing was recompiled to get this sharper picture.',
      camera: {longitude: -87.6325, latitude: 41.9625, zoom: 13.2, transitionMs: 1800},
      callout: {coordinate: [-87.6325, 41.9625], text: 'Densest cell: Montrose Point'}
    },
    {
      id: 'hexagons',
      controls: ['binning', 'resolution'],
      readouts: ['cellSize'],
      title: 'Hexagons instead of squares',
      body: 'Squares have two kinds of neighbour (edge and corner) at different distances, which can imprint a grid-aligned look on a pattern. **Hexagons** (switch **Cell shape** below) have six neighbours at the same distance, so adjacent cells compare fairly.\n\nThe lattice is a *compile-time* option, so the panel marks it with a rebuild badge and **Under the hood** counts a new graph. Hexagon mode shows raw counts: smoothing applies to the square grid. Try **Resolution** too: it changes `gridSize`, also compile-time.',
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.8, transitionMs: 1800},
      options: {binning: 'hexagon'}
    },
    {
      id: 'smoothing',
      controls: ['smoothing', 'sigma', 'measure'],
      readouts: ['kernelRadius', 'difference'],
      title: 'Smoothing is a weight buffer',
      body: 'Back on squares, the **Gaussian** spreads each cell over its neighbours so one busy cell stops dominating: the field becomes a kernel density estimate. The kernel weights live in a parameter buffer, so dragging **Smoothing radius (sigma)** changes the map instantly without recompiling.\n\nA large sigma gives the regional picture (the lakefront ribbon against the inland city); a small one separates individual parks. A Gaussian is separable, so a horizontal pass plus a vertical pass equals the full 2D blur with 34 taps per cell instead of 289. Press **Time 2D vs separable smoothing** to measure it; the two fields agree to floating-point rounding.',
      options: {binning: 'grid', smoothing: 'gaussian-2d', sigma: 2.5},
      highlight: {readout: 'difference'}
    },
    {
      id: 'dawn-chorus',
      controls: ['hours', 'invertHours', 'dayType', 'category'],
      readouts: ['included'],
      title: 'Ask "when?" with a mask',
      body: 'Pick an **Hour of day** window and the GPU skips every observation outside it; no graph is rebuilt, only a 43,557-entry mask buffer is rewritten. Here the window keeps **19:00 to 05:00**, the hours when most people are asleep: only 4,281 observations remain, and 65% of them are insects (24% overall), because moths and other bugs drawn to porch lights get logged after dark.\n\nBirds run the other way, with their busiest hours at 07:00 to 10:00 (about 32% of all bird records). Combine the window with **Days** and **Group** below (try Birds on weekends) to build your own question.',
      options: {hours: [5, 19], invertHours: true, smoothing: 'gaussian-separable', sigma: 2}
    },
    {
      id: 'research-grade',
      controls: ['statistic', 'weight', 'sumAccumulation'],
      readouts: ['peak'],
      title: 'From counts to shares with sum and mean',
      body: 'Switching the **Statistic per cell** to *mean* with the **Attribute** *Research grade* divides confirmed identifications by observations in each cell: where the community most often agreed on the species. Overall 63% of observations reach research grade. *Sum* counts the confirmed ones themselves. Both read a per-point **weights** buffer; **Sum accumulation** chooses how the adds are done (workgroup partial sums or atomic), with identical results.\n\nCells with no observations show 0 and are transparent. Smoothing is off here so each cell is a plain share, and the scale runs 0 to 100%. Try the *Introduced* attribute too: 15.7% of observations are non-native taxa.',
      options: {
        statistic: 'mean',
        weight: 'researchGrade',
        smoothing: 'off',
        hours: [0, 24],
        invertHours: false,
        resolution: 'coarse'
      }
    },
    {
      id: 'limits',
      controls: ['category', 'resolution', 'showPoints'],
      title: 'Limits and things to try',
      body: 'Density is not abundance: it measures observer effort as much as wildlife, ignores how many people visit a cell, and says nothing about places nobody photographs. Shares over small cell counts are noisy, which is why the share step uses coarse cells.\n\nTry, with **Group** and **Days**: Fungi, which are logged in far fewer places than birds; Insects at fine resolution; *Introduced* mean over hexagons with *Plants*; then turn on **Show source points** below to see individual observations along paths and ponds.',
      options: {
        statistic: 'count',
        resolution: 'medium',
        smoothing: 'gaussian-separable',
        showPoints: true
      }
    }
  ]
});
