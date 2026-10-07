// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {SiteSuitabilityOptions} from './site-suitability.compute';
import {COVER_COLORS, COVER_NAMES} from './b16-colors';

const RAMPS = [
  {value: 'viridis', label: 'Viridis'},
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'},
  {value: 'grayscale', label: 'Grayscale'}
] as const;

const GREENVILLE: readonly [number, number] = [-120.9511, 40.1396];

const coverSlider = (
  id: keyof SiteSuitabilityOptions & string,
  label: string,
  defaultValue: number,
  help: string
) => ({
  kind: 'slider' as const,
  id,
  label,
  group: 'Land cover table (GPUWeightedOverlay table mode)',
  apply: 'param' as const,
  min: 0,
  max: 1,
  step: 0.05,
  default: defaultValue,
  help
});

export default defineScene<SiteSuitabilityOptions>({
  id: 'site-suitability',
  title: 'Where should post-fire erosion crews work first?',
  chapter: 'raster',
  order: 2,
  summary:
    'After the Dixie Fire, a weighted overlay of burn severity, slope, land cover, distance to Greenville and remaining greenness ranks every 20 m cell. Cell statistics show where the criteria disagree; sampling and zonal statistics turn the raster into candidate sites and per-cover numbers.',
  contributors: [
    'GPUWeightedOverlay',
    'GPURasterCellStatistics',
    'GPURasterSampling',
    'GPURasterZonalStatistics',
    'GPUTerrainDerivatives',
    'GPURasterReclassify',
    'GPURasterArithmetic',
    'GPURasterConditional'
  ],
  datasets: [{id: 'dixie-fire', role: 'Sentinel-2 bands, WorldCover, DEM'}],
  initialView: {longitude: -121.0, latitude: 40.17, zoom: 11.5},

  options: [
    {
      kind: 'range',
      id: 'severityRange',
      label: 'Severity ramp (0 to 1 over)',
      group: 'Criteria ranges (linear rescale)',
      apply: 'param',
      min: -0.5,
      max: 1.5,
      step: 0.05,
      default: [0.1, 1],
      unit: 'dNBR',
      help: 'dNBR at or below the first value scores 0; at or above the second scores 1.'
    },
    {
      kind: 'range',
      id: 'slopeRange',
      label: 'Slope ramp',
      group: 'Criteria ranges (linear rescale)',
      apply: 'param',
      min: 0,
      max: 60,
      step: 1,
      default: [5, 35],
      unit: 'degrees',
      help: 'Slope in degrees from the Horn method. Debris-flow hazard studies often use 30 degrees and up as a steep-slope class.'
    },
    {
      kind: 'range',
      id: 'proximityRange',
      label: 'Closeness ramp (inverted)',
      group: 'Criteria ranges (linear rescale)',
      apply: 'param',
      min: 0,
      max: 15000,
      step: 250,
      default: [0, 8000],
      unit: 'm',
      help: 'Distance from Greenville: at or below the first value scores 1, at or beyond the second scores 0.'
    },
    {
      kind: 'range',
      id: 'greennessRange',
      label: 'Greenness ramp (inverted)',
      group: 'Criteria ranges (linear rescale)',
      apply: 'param',
      min: -0.2,
      max: 0.9,
      step: 0.05,
      default: [0.1, 0.6],
      unit: 'NDVI',
      help: 'Post-fire NDVI: at or below the first value scores 1 (bare), at or above the second scores 0 (green).'
    },
    {
      kind: 'slider',
      id: 'weightSeverity',
      label: 'Burn severity weight',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.3,
      help: 'Higher dNBR means more vegetation was lost and the soil is exposed.'
    },
    {
      kind: 'slider',
      id: 'weightSlope',
      label: 'Slope weight',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.3,
      help: 'Steeper ground sheds water and debris faster.'
    },
    {
      kind: 'slider',
      id: 'weightCover',
      label: 'Land cover weight',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.1,
      help: 'Weight of the land cover table below.'
    },
    {
      kind: 'slider',
      id: 'weightProximity',
      label: 'Closeness to Greenville weight',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.2,
      help: 'Closer to the town means more people and roads below the burn. The distance is inverted: near scores high.'
    },
    {
      kind: 'slider',
      id: 'weightGreenness',
      label: 'Loss of greenness weight',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.1,
      help: 'Post-fire NDVI, inverted: bare ground (low NDVI) scores high.'
    },
    {
      kind: 'toggle',
      id: 'normalizeWeights',
      label: 'Normalize weights',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      default: true,
      help: 'Divides the weighted sum by the sum of absolute weights, so the score stays between 0 and 1 whatever you set.'
    },
    {
      kind: 'select',
      id: 'noDataPolicy',
      label: 'Missing criterion',
      group: 'Weights (GPUWeightedOverlay)',
      apply: 'param',
      default: 'propagate',
      help: 'Propagate makes a cell nodata when any criterion is nodata. Ignore skips the missing layer and renormalizes by the weights that remain.',
      options: [
        {value: 'propagate', label: 'Propagate (cell becomes no data)'},
        {value: 'ignore', label: 'Ignore (score from the layers that exist)'}
      ]
    },
    coverSlider('coverTree', 'Tree cover', 0.7, 'Burned forest loses its canopy and litter layer.'),
    coverSlider('coverShrub', 'Shrubland', 0.8, 'Brush resprouts quickly but burns hot.'),
    coverSlider('coverGrass', 'Grassland', 0.5, 'Grass roots hold soil; low score.'),
    coverSlider('coverCrop', 'Cropland', 0.3, 'Flat, managed ground.'),
    coverSlider('coverBare', 'Bare / sparse vegetation', 1, 'Already exposed soil.'),
    {
      kind: 'toggle',
      id: 'restrictBuiltWater',
      label: 'Restrict built-up land and open water',
      group: 'Land cover table (GPUWeightedOverlay table mode)',
      apply: 'param',
      default: true,
      help: 'A NaN value in the remap table marks a restricted class: the cell is excluded from the score, like ArcGIS "Restricted". Turn it off to score these classes instead.'
    },
    {
      kind: 'select',
      id: 'statisticsPolicy',
      label: 'Missing layer policy',
      group: 'Cell statistics (GPURasterCellStatistics)',
      apply: 'param',
      default: 'ignore',
      help: 'Ignore computes statistics over the valid layers of each cell (ArcGIS "DATA"); Propagate makes any cell with a nodata layer nodata.',
      options: [
        {value: 'ignore', label: 'Ignore nodata layers'},
        {value: 'propagate', label: 'Propagate nodata'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumValid',
      label: 'Minimum valid layers',
      group: 'Cell statistics (GPURasterCellStatistics)',
      apply: 'param',
      min: 1,
      max: 5,
      step: 1,
      default: 1,
      help: 'Cells with fewer valid layers become nodata. Set 5 and restricted cells vanish.'
    },
    {
      kind: 'toggle',
      id: 'showSites',
      label: 'Show candidate sites',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      default: false,
      help: 'A jittered 200 m lattice of points, each sampling the suitability raster on the GPU.'
    },
    {
      kind: 'slider',
      id: 'siteCount',
      label: 'Active sites',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      min: 100,
      max: 5400,
      step: 100,
      default: 3000,
      help: 'The active point count is a one-row parameter: rows beyond it are written NaN. No recompile.'
    },
    {
      kind: 'slider',
      id: 'siteThreshold',
      label: 'Show sites scoring above',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.55,
      help: 'Sampled values at or below this are discarded in the point shader.'
    },
    {
      kind: 'slider',
      id: 'siteSize',
      label: 'Site size',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      min: 1.5,
      max: 8,
      step: 0.5,
      default: 3.5,
      unit: 'px',
      help: 'Point radius on screen.'
    },
    {
      kind: 'select',
      id: 'sampleMethod',
      label: 'Interpolation',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      default: 'bilinear',
      help: 'How a value is read between cell centers. Nearest is exact, bilinear blends four cells, bicubic (Catmull-Rom) blends sixteen.',
      options: [
        {value: 'nearest', label: 'Nearest cell'},
        {value: 'bilinear', label: 'Bilinear'},
        {value: 'bicubic', label: 'Bicubic (Catmull-Rom)'}
      ]
    },
    {
      kind: 'select',
      id: 'sampleNoData',
      label: 'Nodata in the support',
      group: 'Candidate sites (GPURasterSampling)',
      apply: 'param',
      default: 'strict',
      help: 'Strict: any nodata neighbour makes the sample NaN, so sites near water and built-up land vanish. Renormalize: weights over the valid neighbours are rescaled.',
      options: [
        {value: 'strict', label: 'Strict (NaN near nodata)'},
        {value: 'renormalize', label: 'Renormalize over valid cells'}
      ]
    },
    {
      kind: 'toggle',
      id: 'ignoreTrees',
      label: 'Leave tree cover out of the zones',
      group: 'Zonal statistics (GPURasterZonalStatistics)',
      apply: 'param',
      default: false,
      help: 'Rewrites the zone raster so tree cover carries the ignored zone id (0xffffffff). Its cells are skipped silently.'
    },
    {
      kind: 'select',
      id: 'zonalOrder',
      label: 'Sum order',
      group: 'Zonal statistics (GPURasterZonalStatistics)',
      apply: 'compile',
      default: 'sorted',
      help: 'Compile-time: both variants are compiled and timed by the button below. Sorted sums are bitwise reproducible; atomic sums depend on accumulation order and are slow when many neighbouring cells share a zone, as with land cover.',
      options: [
        {value: 'sorted', label: 'Sorted (reproducible)'},
        {value: 'atomic', label: 'Atomic (order-dependent)'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time sorted vs atomic zonal sums',
      group: 'Zonal statistics (GPURasterZonalStatistics)',
      help: 'Runs both zonal graphs outside the frame and reports GPU time per run.'
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'score',
      help: 'All of these are already in GPU memory; this chooses which buffer the layer draws.',
      options: [
        {value: 'score', label: 'Suitability score (weighted overlay)'},
        {
          value: 'meanCriteria',
          label: 'Mean of the rescaled criteria (cell statistics)',
          help: 'Equal-weight average of the five 0 to 1 criteria.'
        },
        {value: 'spread', label: 'Spread: where criteria disagree (std dev)'},
        {value: 'weakest', label: 'Weakest criterion (minimum)'},
        {value: 'strongest', label: 'Strongest criterion (maximum)'},
        {value: 'range', label: 'Range (maximum - minimum)'},
        {value: 'validLayers', label: 'Valid criteria per cell (count)'},
        {value: 'c-severity', label: 'Criterion: burn severity (dNBR)'},
        {value: 'c-slope', label: 'Criterion: slope'},
        {value: 'c-cover', label: 'Criterion: land cover table'},
        {value: 'c-proximity', label: 'Criterion: closeness to Greenville'},
        {value: 'c-greenness', label: 'Criterion: loss of greenness'},
        {value: 'cover', label: 'ESA WorldCover classes (the zones)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'For the continuous layers.',
      options: RAMPS,
      disabledWhen: state => state.layer === 'cover'
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
      help: 'Lower it to see roads under the raster.'
    },
    {
      kind: 'slider',
      id: 'minimumScore',
      label: 'Hide scores at or below',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0,
      disabledWhen: state => state.layer !== 'score',
      help: 'A discard threshold in the layer shader: only the best cells stay visible. 0 shows everything.'
    }
  ],

  readouts: [
    {
      id: 'scoreRange',
      label: 'Score range',
      help: 'Exact minimum and maximum of the defined scores, from the overlay scoreRange output.'
    },
    {
      id: 'sites',
      label: 'Candidate sites',
      help: 'Active sites above the threshold, from the sampling output read back after the controls settle.'
    },
    {
      id: 'parity',
      label: 'GPU vs CPU check',
      help: 'With the nearest method the sampled score is recomputed on the CPU from the criteria and compared.'
    },
    {
      id: 'zoneTree',
      label: 'Tree cover',
      help: 'Share of the window, zonal mean score with its range, and the burned share (dNBR > 0.27) with the CPU reference stored with the dataset.'
    },
    {id: 'zoneGrass', label: 'Grassland'},
    {id: 'zoneCrop', label: 'Cropland'},
    {id: 'zoneBuilt', label: 'Built-up'},
    {id: 'zoneBare', label: 'Bare / sparse'},
    {id: 'zoneWater', label: 'Permanent water'},
    {
      id: 'zonalOverflow',
      label: 'Zone ids beyond capacity',
      help: 'The overflow flag: set when any cell holds a zone id at or above zoneCapacity that is not the ignored zone.'
    },
    {
      id: 'zonalSums',
      label: 'Zonal sum order',
      help: 'Whether the displayed means come from the reproducible sorted reduction or from float atomics, and how far atomics drift.'
    },
    {id: 'zonalTiming', label: 'Zonal sum timing'}
  ],

  legends: state => {
    const legends: Array<
      | {
          kind: 'ramp';
          id?: string;
          title: string;
          ramp: SiteSuitabilityOptions['ramp'];
          extent: 'gpu' | readonly [number, number];
          labels?: readonly [string, string];
          format?: (value: number) => string;
          unit?: string;
        }
      | {
          kind: 'categories';
          title: string;
          entries: {color: readonly [number, number, number, number]; label: string}[];
          note?: string;
        }
    > = [];
    if (state.layer === 'score') {
      legends.push({
        kind: 'ramp',
        id: 'score',
        title: 'Suitability score',
        ramp: state.ramp,
        extent: 'gpu',
        format: value => value.toFixed(2)
      });
    } else if (state.layer === 'cover') {
      legends.push({
        kind: 'categories',
        title: 'ESA WorldCover 2021',
        entries: COVER_NAMES.slice(0, 8).map((name, index) => ({
          color: COVER_COLORS[index],
          label: name
        })),
        note: 'These class indexes are the zone ids of the zonal statistics.'
      });
    } else if (state.layer === 'validLayers') {
      legends.push({
        kind: 'ramp',
        title: 'Valid criteria',
        ramp: state.ramp,
        extent: [0, 5],
        format: value => value.toFixed(0)
      });
    } else {
      const titles: Record<string, string> = {
        meanCriteria: 'Mean criterion score',
        spread: 'Standard deviation of the criteria',
        weakest: 'Weakest criterion',
        strongest: 'Strongest criterion',
        range: 'Range of the criteria',
        'c-severity': 'Burn severity, rescaled',
        'c-slope': 'Slope, rescaled',
        'c-cover': 'Land cover score',
        'c-proximity': 'Closeness to Greenville, rescaled',
        'c-greenness': 'Loss of greenness, rescaled'
      };
      legends.push({
        kind: 'ramp',
        title: titles[state.layer] ?? 'Value',
        ramp: state.ramp,
        extent: [0, state.layer === 'spread' ? 0.5 : 1],
        format: value => value.toFixed(2)
      });
    }
    if (state.showSites) {
      legends.push({
        kind: 'ramp',
        title: 'Sampled score at candidate sites',
        ramp: 'viridis',
        extent: [0, 1],
        format: value => value.toFixed(1)
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUWeightedOverlay, GPURasterCellStatistics, GPURasterSampling, GPURasterZonalStatistics,
  getGPUWeightedOverlayParameterValues, getGPURasterCellStatisticsParameterValues,
  getGPURasterSamplingParameterValues
} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'suitability'});
graph.add(new GPUWeightedOverlay({
  stack: rawStack,            // band-sequential float32: dNBR, slope, cover code, distance, NDVI
  layerCount: 5, cellCount,
  parameters, remapBreaks, remapValues, maximumBreakCount: 8,
  output: {score, scoreRange}
}));
graph.add(new GPURasterCellStatistics({stack: normalizedStack, layerCount: 5, cellCount,
  parameters: statisticsParameters, output: {mean, standardDeviation, minimum, maximum, range, count}}));
graph.add(new GPURasterSampling({width, height, values: score, positions: sites, pointCount,
  parameters: samplingParameters, output: {values: siteScores}}));
graph.add(new GPURasterZonalStatistics({width, height, zones, values: scoreBand, zoneCapacity: 8,
  sumOrder: '${state.zonalOrder}', output: {cellCounts, means, minimums, maximums}, overflow}));
const compiled = graph.compile();          // once

// per frame, no recompile
parameters.write(getGPUWeightedOverlayParameterValues({
  normalizeWeights: ${state.normalizeWeights}, noDataPolicy: '${state.noDataPolicy}',
  layers: [
    {weight: ${state.weightSeverity}, inputMin: ${state.severityRange[0]}, inputMax: ${state.severityRange[1]}},
    {weight: ${state.weightSlope}, inputMin: ${state.slopeRange[0]}, inputMax: ${state.slopeRange[1]}},
    {weight: ${state.weightCover}, mode: 'table', breakCount: 8},          // NaN value = restricted
    {weight: ${state.weightProximity}, inputMin: ${state.proximityRange[0]}, inputMax: ${state.proximityRange[1]}, invert: true},
    {weight: ${state.weightGreenness}, inputMin: ${state.greennessRange[0]}, inputMax: ${state.greennessRange[1]}, invert: true}
  ]}));
statisticsParameters.write(getGPURasterCellStatisticsParameterValues({
  noDataPolicy: '${state.statisticsPolicy}', minimumValidCount: ${state.minimumValid}}));
samplingParameters.write(getGPURasterSamplingParameterValues({
  width, height, extent, method: '${state.sampleMethod}', noDataPolicy: '${state.sampleNoData}'}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUWeightedOverlay` scores every cell as a weighted sum of up to 16 criteria, each rescaled linearly or through a class table; `GPURasterCellStatistics` summarises the same criteria per cell (mean, spread, extremes); `GPURasterSampling` reads the result at points; and `GPURasterZonalStatistics` summarises it per land cover class. Slope comes from `GPUTerrainDerivatives`, dNBR from `GPURasterArithmetic`.',
    why: 'A weighted overlay is the standard way to turn "steep, badly burned, close to town" into a map a crew can act on. Because the weights, ranges and class table are parameters, the analyst can argue with the model live instead of rerunning a script.',
    howToRead:
      'Bright cells score high: they are steep, severely burned, bare and close to Greenville in the proportions you set. The spread map is the audit: where it is bright, one criterion says "treat" while another says "leave", and the score hides a disagreement worth inspecting.'
  },

  create: async ctx => (await import('./site-suitability.compute')).createSiteSuitability(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where should erosion crews work first after the Dixie Fire?',
      body: 'A burned slope above a town sheds ash and soil into the first heavy rain. Emergency teams (in the US, BAER teams) have to pick where mulch, straw wattles and debris basins go. Here the question is asked of the 15 km square around **Greenville**, which the fire destroyed on 4 August 2021.\n\nFive criteria become one map: **burn severity** (dNBR), **slope**, **land cover**, **closeness to Greenville** and **loss of greenness**. The map shows the weighted result: bright cells are the highest priority. The weights are yours to change, with the sliders below.',
      camera: {longitude: -121.0, latitude: 40.17, zoom: 11.5, transitionMs: 1200},
      controls: [
        'weightSeverity',
        'weightSlope',
        'weightCover',
        'weightProximity',
        'weightGreenness'
      ],
      readouts: ['scoreRange'],
      callout: {coordinate: GREENVILLE, text: 'Greenville'},
      highlight: {readout: 'scoreRange'}
    },
    {
      id: 'rescaling',
      title: 'Every criterion is rescaled to 0 to 1',
      body: "Slope is measured in degrees and dNBR is unitless, so each layer is first mapped to 0 to 1. This map is the **slope criterion**: `GPUTerrainDerivatives` (Horn's 3 x 3 method) derives it from the DEM, and a linear ramp from 5 to 35 degrees turns it into a score. Cells flatter than 5 degrees score 0; cells steeper than 35 score 1.\n\nDrag **Slope ramp** below, or switch **Map shows** to another criterion and drag its ramp. The overlay does the same rescaling internally from `inputMin` and `inputMax`, and `invert` flips a layer so near scores high. Both are per-frame parameters.",
      options: {layer: 'c-slope', ramp: 'viridis'},
      controls: ['slopeRange', 'layer']
    },
    {
      id: 'weights',
      title: 'The weighted overlay combines them',
      body: '`GPUWeightedOverlay` computes `score = sum(weight_i x score_i)`, divided by the sum of the weights when **Normalize weights** is on. The loop runs in a fixed layer order in float32, so the result is repeatable. The readout shows the exact range of the defined scores.\n\nMove **Slope weight** to 1 and the map becomes a slope map; move **Burn severity weight** to 1 and it becomes a severity map. Negative weights are allowed by the contributor (they subtract), but the sliders here stay positive. The method matches the ArcGIS Weighted Overlay tool.',
      options: {layer: 'score'},
      controls: ['normalizeWeights', 'weightSlope', 'weightSeverity'],
      readouts: ['scoreRange']
    },
    {
      id: 'land-cover',
      title: 'Land cover enters through a table, with restricted classes',
      body: "Land cover is categorical, so the overlay classifies it through a **break table** (`mode: 'table'`): WorldCover codes 10, 20, 30 ... fall into classes and each class gets the score you set (**Tree cover**, **Bare / sparse vegetation** and the other class sliders below). Water and built-up land get **NaN**, which marks a **restricted** class: those cells are excluded from the score entirely.\n\nTurn off **Restrict built-up land and open water** below and they are scored instead. The same table also drives a `GPURasterReclassify` node that feeds the cell statistics below.",
      options: {layer: 'cover'},
      controls: ['coverTree', 'coverBare', 'restrictBuiltWater', 'weightCover']
    },
    {
      id: 'agreement',
      title: 'Cell statistics audit the score',
      body: '`GPURasterCellStatistics` reads the five rescaled layers of every cell and reports the **standard deviation** (population, centred second pass). A bright cell is one where the criteria disagree: for example steep but unburned.\n\nChange **Missing layer policy** below to *Propagate* and every restricted cell becomes nodata; set **Minimum valid layers** to 5 for the same effect. Switch **Map shows** to *Weakest criterion* to see the layer holding each cell back.',
      options: {layer: 'spread', ramp: 'magma'},
      controls: ['statisticsPolicy', 'minimumValid', 'layer']
    },
    {
      id: 'sites',
      title: 'Sampling turns the raster into candidate sites',
      body: '`GPURasterSampling` reads the score raster at jittered points, up to 5,400 of them and 3,000 active here ("extract values to points"). Only sites above the threshold are drawn. The active point count is a parameter, so **Active sites** needs no recompile; **Show sites scoring above** hides the weak ones.\n\nTry the three **Interpolation** methods below: *Nearest cell* is exact (the readout recomputes the score on the CPU and reports the largest difference), *Bilinear* blends four cells and *Bicubic* sixteen. With **Nodata in the support** set to *Strict*, a site next to water or town returns NaN; *Renormalize* keeps it.',
      options: {
        layer: 'score',
        showSites: true,
        siteCount: 3000,
        sampleMethod: 'nearest',
        siteThreshold: 0.55,
        opacity: 0.55
      },
      controls: ['siteCount', 'siteThreshold', 'sampleMethod', 'sampleNoData'],
      readouts: ['parity', 'sites'],
      highlight: {readout: 'parity'}
    },
    {
      id: 'zonal',
      title: 'Zonal statistics summarise by land cover, then the caveats',
      body: "`GPURasterZonalStatistics` takes the WorldCover class raster as zones and reports each class's cell count, mean, minimum and maximum score. A second instance averages a 0/1 burned raster, so its mean is the **burned share per class**: tree cover 66%, grassland 31%, cropland 27%, built-up 23% in the CPU reference, and the GPU matches. Press **Time sorted vs atomic zonal sums** to see why sorted sums are the default for contiguous zones.\n\n**Limits:** the criteria are proxies, not a hydrologic model; the weights are a judgement; 20 m pixels average mixed ground; and the overflow readout shows the zone capacity (8) being exceeded by a single wetland pixel. **Try:** **Leave tree cover out of the zones**, bicubic sampling with renormalize, severity weight 1 and slope weight 1.",
      options: {layer: 'score', ignoreTrees: false},
      controls: ['ignoreTrees', 'zonalOrder', 'measure'],
      readouts: ['zoneTree', 'zoneGrass', 'zonalTiming'],
      highlight: {readout: 'zoneTree'}
    }
  ]
});
