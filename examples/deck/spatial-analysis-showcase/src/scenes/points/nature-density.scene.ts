// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {EFFORT_CAVEAT} from '../../cartography/hue-registry';
import {formatCount} from '../../cartography/live-text';
import {directionFor} from '../../engine/ramps';
import {defineScene, type LegendSpec} from '../scene';
import {formatCategory} from './b1-nature-data';
import {nextStoryLine} from './b1-points-look';
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

/** The cartouche of one step: the claim, the variable and method and the chip (the sample line is set from the data). */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Observer effort'] as const
});

/** Names that orient the reader in every step, from the Chicago gazetteer (drawn above the data). */
const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan', 'loop', 'jackson-park', 'humboldt-park'], {
  loop: {minZoom: 10.2},
  'jackson-park': {tone: 'muted', minZoom: 10.5},
  'humboldt-park': {tone: 'muted', minZoom: 10.5}
});

/** Warm dot colour of the point layer and its cool partner over the surface (legend swatches). */
const DOT_COLOR = [254, 196, 79, 255] as const;
const DOT_COLOR_OVER_SURFACE = [140, 215, 255, 255] as const;
/** The ramp trim of the layer and the legend (rule 3). */
const RAMP_RANGE = [0.15, 1] as const;

export default defineScene<NatureDensityOptions>({
  id: 'nature-density',
  title: 'Where does Chicago wildlife get noticed?',
  chapter: 'points',
  order: 1,
  summary:
    'A camera-following map of 43,557 iNaturalist records—not 43,557 organisms—reveals where Chicago nature gets noticed. Move from points to a GPU density estimate, then test bandwidth, grid and observer-time assumptions.',
  contributors: ['GPUPointDensity'],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-parks', role: 'park and preserve outlines (context)'},
    {id: 'chicago-boundary', role: 'city limit and lake shore (context)'}
  ],
  initialView: {...CITY_FRAMES.chicago},

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
      marks: [{value: 1.5, label: 'default'}],
      // Too small to smooth: every footpath is a peak.
      danger: [0.5, 0.75],
      describe: value =>
        `${value} cells; a record reaches ${Math.min(8, Math.ceil(3 * value))} cells`,
      // Play sweeps the bandwidth from too narrow to too wide.
      autoSweep: {from: 0.5, to: 2.5, durationMs: 9000, ease: 'in-out'},
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
      help: 'Keeps observations whose Chicago local wall-clock hour falls in [from, to). Times are stored as if UTC and are never timezone-converted. The mask buffer is rewritten on the CPU; the graph is not rebuilt.'
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
      id: 'view',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'density',
      help: 'The raw point process (one dot per observation), the density surface computed from it, or both. Switching is a layer change; nothing is recompiled.',
      options: [
        {value: 'points', label: 'Observations (dots)'},
        {value: 'density', label: 'Density surface'},
        {value: 'both', label: 'Both'}
      ]
    },
    {
      kind: 'select',
      id: 'blending',
      label: 'Dot blending',
      group: 'Display',
      apply: 'param',
      default: 'additive',
      disabledWhen: state => state.view === 'density',
      help: 'Additive: overlapping dots add their light, so stacks glow brighter. Normal: each dot paints over the last, so a stack of a hundred looks like a stack of five (overplotting).',
      options: [
        {value: 'additive', label: 'Additive (overlaps add up)'},
        {value: 'normal', label: 'Normal (overlaps hide)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'Ramps made for a dark ground: brightest means most. They are trimmed so the lowest colour is not black. Inferno is the default; mako is a cool alternative.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'fire', label: 'Fire (Crameri)'},
        {value: 'mako', label: 'Mako (cool)'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Surface opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      disabledWhen: state => state.view === 'points',
      help: 'Lower it to see the streets and the dots under the surface.'
    },
    {
      kind: 'toggle',
      id: 'showKernel',
      label: 'Show the kernel',
      group: 'Smoothing',
      apply: 'param',
      default: false,
      disabledWhen: state => state.binning === 'hexagon' || state.smoothing === 'off',
      help: 'Draws the Gaussian at the map centre: the inner ring is one sigma, the dashed ring the 3-sigma reach beyond which a point no longer contributes. Its size in metres follows the zoom, because sigma is measured in cells.'
    },
    {
      kind: 'toggle',
      id: 'showContext',
      label: 'Parks and city limit',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws park and preserve outlines, the lake shore and a dashed frame on the city limit, where the data ends.'
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
    {id: 'records', label: 'Records on the map', help: 'Uploaded once.'},
    {id: 'taxa', label: 'Different taxa', help: 'Distinct species or taxa among the records.'},
    {
      id: 'parkShare',
      label: 'Records in mapped green space',
      help: 'Share of all records inside an OpenStreetMap park, preserve or woodland polygon.'
    },
    {id: 'kept', label: 'Records kept', help: 'Records passing the hour, day and group filters.'},
    {
      id: 'challengePulse',
      label: 'City Nature Challenge pulse',
      help: 'Selected records made from 28 April through 1 May 2023, divided by all selected records.'
    },
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Follows the zoom: the bounds buffer is rewritten every frame.'
    },
    {
      id: 'peak',
      label: 'Busiest cell',
      help: 'The maximum of the displayed field, read back once the camera settles.'
    },
    {id: 'peakShare', label: 'Share in the busiest cell'},
    {
      id: 'peakPlace',
      label: 'Busiest cell is',
      help: 'Nearest named place in the Chicago gazetteer.'
    },
    {
      id: 'peakCount',
      label: 'Separate peaks',
      help: 'Local maxima above 2 % of the maximum: the number of peaks the bandwidth leaves.'
    },
    {id: 'bandwidth', label: 'Bandwidth (one sigma)', help: 'Sigma in metres at the current zoom.'},
    {
      id: 'distribution',
      label: 'Distribution of cell values',
      kind: 'chart',
      help: 'Non-empty cells by record count; the marker follows the hovered cell.'
    },
    {
      id: 'kernelProfile',
      label: 'The kernel',
      kind: 'chart',
      help: 'The weight a record gives to cells at each distance; click to set sigma.'
    },
    {id: 'window', label: 'Hour window', hood: true},
    {id: 'grid', label: 'Grid', hood: true},
    {
      id: 'kernelRadius',
      label: 'Kernel radius',
      hood: true,
      help: 'Three sigma, capped at the compile-time kernel of 8 cells.'
    },
    {id: 'denseTime', label: '2D Gaussian graph', hood: true},
    {id: 'separableTime', label: 'Separable graph', hood: true},
    {id: 'speedup', label: 'Separable vs 2D', hood: true},
    {
      id: 'difference',
      label: 'Max difference',
      hood: true,
      help: 'Largest absolute difference between the two smoothed fields, read back once the camera settles.'
    }
  ],

  pipeline: [
    {
      id: 'grid',
      label: 'Grid build',
      detail: 'A grid laid over the screen, rewritten from the camera'
    },
    {
      id: 'sum',
      label: 'Sum',
      detail: 'Every record adds itself to its cell',
      show: {option: 'smoothing', value: 'off'}
    },
    {
      id: 'blur-h',
      label: 'Gaussian H',
      detail: 'A horizontal pass of the bell-shaped kernel',
      show: {option: 'smoothing', value: 'gaussian-separable'}
    },
    {
      id: 'blur-v',
      label: 'Gaussian V',
      detail: 'A vertical pass: two 1D passes equal the full 2D blur',
      show: {option: 'smoothing', value: 'gaussian-separable'}
    },
    {id: 'extent', label: 'Extent', detail: 'Reductions find the range and the histogram'},
    {
      id: 'draw',
      label: 'Draw',
      detail: 'The raster layer reads the same GPU buffer',
      show: {option: 'view', value: 'density'}
    }
  ],

  legends: (state, data) => {
    const weightName = WEIGHT_NAMES[state.weight];
    const title =
      state.statistic === 'count'
        ? 'Observations'
        : state.statistic === 'sum'
          ? `${weightName[0].toUpperCase()}${weightName.slice(1)}`
          : `Share that are ${weightName}`;
    const smoothed = state.binning === 'grid' && state.smoothing !== 'off';
    const legends: LegendSpec[] = [];
    if (state.view !== 'density') {
      legends.push({
        kind: 'categories',
        title: 'Observations',
        entries: [
          {
            color: state.view === 'points' ? DOT_COLOR : DOT_COLOR_OVER_SURFACE,
            label: 'One iNaturalist record',
            shape: 'dot'
          }
        ],
        note:
          state.blending === 'additive'
            ? 'Additive: where dots stack, their light adds up toward white.'
            : 'Normal blending: stacked dots hide each other.'
      });
    }
    if (state.view !== 'points') {
      const histogram = data['histogram'] as number[] | undefined;
      const basis = data['basis'] as string | undefined;
      const marker = data['marker'] as number | null | undefined;
      legends.push({
        kind: 'ramp',
        id: 'density',
        title,
        ramp: state.ramp,
        // The layer runs this ramp from its bright end on the dark ground; the legend must agree.
        reverse: directionFor('dark', state.ramp).reverse,
        range: RAMP_RANGE,
        extent: state.statistic === 'mean' ? [0, 1] : 'gpu',
        sqrtScale: state.statistic !== 'mean',
        unit:
          state.statistic === 'count'
            ? 'records'
            : state.statistic === 'sum'
              ? 'flagged records'
              : 'share',
        basis,
        marker: marker ?? undefined,
        // Bar heights are the square root of the cell counts so the long tail stays visible.
        histogram:
          state.statistic !== 'mean' && histogram
            ? histogram.map(count => Math.sqrt(count))
            : undefined,
        note:
          state.statistic === 'mean'
            ? 'Cells with no records are not drawn.'
            : `${smoothed ? 'Smoothed by a Gaussian. ' : ''}Top 2 % of cells clipped to the brightest colour; empty cells are not drawn.`,
        format: value =>
          state.statistic === 'mean' ? `${(value * 100).toFixed(0)}%` : formatCount(value)
      });
    }
    return legends;
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
compiled.encode(commandEncoder, {parameters: undefined});
${
  state.view === 'points'
    ? ''
    : `
// The field is drawn straight from the GPU buffer: no readback.
new SpatialAnalysisRasterLayer({gridSize, bounds: bounds.buffer, values, valueRange: [0, p98],
  colormap: '${state.ramp}', rampRange: [0.15, 1], sqrtScale: ${state.statistic !== 'mean'},
  binning: '${state.binning}'});`
}${
  state.view === 'density'
    ? ''
    : `
new SpatialAnalysisPointLayer({positions, instanceCount,
  radiusPixels: DENSE_POINT_RADIUS_STOPS,   // 1.2 px at z10 to 4 px at z14
  blending: '${state.blending}', color: [254, 196, 79, ${state.blending === 'additive' ? 20 : 64}]});`
}`;
  },

  about: {
    what: '`GPUPointDensity` bins points into a square grid or a hexagon lattice that covers the visible map, optionally convolves the field with a Gaussian kernel, and reduces it to an extent and a histogram, all in compute shaders. The statistic per cell is a count, a sum of a weight, or its mean.',
    why: 'Density answers "where is activity concentrated?" without choosing a boundary first. Because the grid follows the camera, the answer stays sharp at every zoom, and the same graph answers different questions (when, which group, what share) through small buffers.',
    howToRead:
      'Brighter cells hold more observations. The scale uses a square root so quiet neighbourhoods stay visible next to hot spots; the legend range is the real min and max read back from the GPU. Cells with a value of 0 are transparent. Observations follow observers, so a bright cell means many people looked and logged what they saw there, not only that much lives there.'
  },

  // Night ground in both page themes: additive light needs a dark ground.
  basemap: ground('night'),
  furniture: {
    title: cartouche('Where does Chicago notice nature?', 'Wildlife records, 2023'),
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.iNaturalist, CREDITS.openStreetMap, CREDITS.cityOfChicago),
    // Rule 15: observer effort is taught here and stated once per chapter.
    caveat: EFFORT_CAVEAT
  },
  annotations: ORIENTATION,

  create: async ctx => (await import('./nature-density.compute')).createNatureDensity(ctx),

  story: [
    {
      id: 'the-point-cloud',
      title: 'Every dot is a sighting',
      headline: 'Dots pile up faster than the eye can count',
      textAlternative:
        'Dark map of Chicago covered in glowing amber dots, densest along the lakefront and in the large parks.',
      body: 'Each dot is one iNaturalist **record** made inside Chicago in 2023: **{{records}}** records spanning {{taxa}} taxa. It is not one organism, and repeated visits, prolific observers and many photographs at one site all make more dots. Light on a dark ground is the figure, and **additive blending** lets overlapping dots add their light, so stacks glow.\n\nSwitch **Dot blending** to *Normal*: a pile of a hundred now looks like a pile of five. That is overplotting.',
      evidence:
        'The map starts with the complete filtered sample: **{{records}} records**, not an estimated population of organisms.',
      caveat:
        'Absence of a record can mean nobody looked, nobody uploaded, or nothing was observed; these data cannot separate those explanations.',
      optionsMode: 'fresh',
      options: {view: 'points', blending: 'additive'},
      controls: ['blending'],
      readouts: ['records', 'taxa'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {title: cartouche('Where does Chicago notice nature?', 'One dot per record')},
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park'])
    },
    {
      id: 'dots-meet-the-parks',
      title: 'Dots meet the parks',
      headline: 'Observers cluster in parks and along paths',
      textAlternative:
        'Amber dots over thin grey outlines of parks and the lake shore, with a dashed frame on the city limit.',
      body: 'Switch on **Parks and city limit**: **{{parkShare}}** of the records fall inside mapped green space. Observers go where there is something to see and a path to walk.\n\nThe dashed frame is the city limit. Data ends there, so any cell beside it has neighbours on one side only.\n\n*A map of records is a map of observers.*',
      optionsMode: 'fresh',
      options: {view: 'points', blending: 'additive', showContext: true},
      controls: ['showContext'],
      readouts: ['parkShare', 'records'],
      camera: {longitude: -87.65, latitude: 41.93, zoom: 11.2, transitionMs: 1600},
      furniture: {title: cartouche('Where do the dots sit?', 'Records and green space')},
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park'])
    },
    {
      id: 'count-the-cells',
      title: 'Count the dots in each cell',
      headline: 'A handful of cells hold the glow',
      textAlternative:
        'Density map of Chicago in inferno on a dark ground: a few bright cells on the lakefront and many faint ones inland.',
      body: 'Without changing the camera, the first GPU stage lays a grid over the same point cloud and adds each dot to its cell. The busiest cell, **{{cellSize}}** across, holds **{{peak}}**, {{peakShare}} of the kept records, {{peakPlace}}.\n\nThe colour scale is a square root, clipped at the 98th percentile, so the long tail does not paint the city black. Hover a cell for its rank. This is still observer activity, now aggregated—not a census of abundance.',
      optionsMode: 'fresh',
      options: {view: 'density', smoothing: 'off', binning: 'grid', resolution: 'medium'},
      controls: ['resolution', 'view'],
      readouts: ['peak', 'peakShare', 'cellSize', 'distribution'],
      stage: 'sum',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      furniture: {title: cartouche('Where are the busiest cells?', 'Records per cell')}
    },
    {
      id: 'bandwidth',
      title: 'Bandwidth is a choice',
      headline: 'Change the bandwidth and the peaks change',
      textAlternative:
        'Smoothed density map with two dashed rings at the map centre showing the kernel width and its reach.',
      body: "A Gaussian spreads each cell over its neighbours: a **kernel density estimate**. Its bandwidth, sigma, here **{{bandwidth}}**, matters most. Press Play on **Smoothing radius**: too small and every footpath is a peak ({{peakCount}}); too large and the lakefront and the parks merge into one blur.\n\nThe dashed rings show sigma and its 3-sigma reach.\n\n*The bandwidth is the analyst's choice, not the data's.*",
      optionsMode: 'fresh',
      options: {
        view: 'density',
        binning: 'grid',
        smoothing: 'gaussian-separable',
        sigma: 1.5,
        showKernel: true
      },
      controls: ['sigma', 'smoothing', 'measure'],
      readouts: ['peakCount', 'bandwidth', 'kernelProfile'],
      stage: 'blur-h',
      camera: {longitude: -87.655, latitude: 41.93, zoom: 11.5, transitionMs: 1600},
      furniture: {title: cartouche('How wide should the kernel be?', 'Smoothed records per cell')},
      highlight: {readout: 'peakCount'}
    },
    {
      id: 'grid-and-hexagons',
      title: 'The grid is part of the answer',
      headline: 'Same dots, different cells, different peaks',
      textAlternative:
        'Hexagon density map of the lakefront at Uptown; each hexagon is coloured by its record count.',
      body: 'The grid follows the screen, so every pan re-bins the dots and the cell size, **{{cellSize}}**, changes with the scale bar. Switch **Cell shape** to hexagons: the same dots now give **{{peak}}** in the busiest cell.\n\nMoving or reshaping the boundaries changes the picture, a small dose of the **modifiable areal unit problem**.\n\n*Boundaries are part of the data.*',
      optionsMode: 'fresh',
      options: {view: 'density', binning: 'hexagon', resolution: 'medium'},
      controls: ['binning', 'resolution'],
      readouts: ['cellSize', 'peak'],
      stage: 'grid',
      camera: {longitude: -87.64, latitude: 41.945, zoom: 11.6, transitionMs: 1800},
      furniture: {title: cartouche('Does the cell shape matter?', 'Records per hexagon')}
    },
    {
      id: 'ask-your-own',
      title: 'Now ask your own question',
      headline: 'Counts follow observers, so ask who looked',
      textAlternative:
        'Density map of Chicago over amber dots, ready to filter by hour, group and statistic.',
      body: `A density surface answers *where records concentrate*, not *how much lives there*. **{{kept}}** records are kept now and the busiest cell holds **{{peak}}**. The City Nature Challenge window, 28 April through 1 May, contributes **{{challengePulse}}** of the selected sample—a coordinated observer pulse that can look biological.\n\nAsk your own question: Birds at dawn, Insects after dark, weekends only. Hour labels are **Chicago local wall-clock time stored as if UTC**; there is no timezone conversion. Or switch **Statistic** to a share: a rate needs a denominator, and its N must stay visible.\n\n${nextStoryLine('nature-density')}`,
      optionsMode: 'fresh',
      options: {view: 'both', smoothing: 'gaussian-separable', sigma: 1.5},
      controls: ['category', 'hours', 'statistic'],
      readouts: ['kept', 'challengePulse', 'peak'],
      stage: 'draw',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      furniture: {title: cartouche('What would you ask?', 'Filtered records per cell')}
    }
  ]
});
