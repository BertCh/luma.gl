// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {BurnSeverityOptions} from './burn-severity.compute';
import {SEVERITY_COLORS, SEVERITY_NAMES} from './b16-colors';

const RAMPS = [
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'},
  {value: 'viridis', label: 'Viridis'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'},
  {value: 'grayscale', label: 'Grayscale'},
  {value: 'diverging', label: 'Diverging (blue to red)'}
] as const;

/** Greenville, California: destroyed on 2021-08-04. */
const GREENVILLE: readonly [number, number] = [-120.9511, 40.1396];

const severityLegend = (state: BurnSeverityOptions) => {
  if (state.classScheme === 'binary') {
    return [
      {
        color: SEVERITY_COLORS[2],
        label: `Below ${state.moderateBreak.toFixed(2)}: unburned or low`
      },
      {color: SEVERITY_COLORS[6], label: `At or above ${state.moderateBreak.toFixed(2)}: burned`}
    ];
  }
  if (state.classScheme === 'custom') {
    return [
      {color: SEVERITY_COLORS[2], label: `Unburned (dNBR < ${state.lowBreak.toFixed(2)})`},
      {
        color: SEVERITY_COLORS[3],
        label: `Low (${state.lowBreak.toFixed(2)} to ${state.moderateBreak.toFixed(2)})`
      },
      {
        color: SEVERITY_COLORS[5],
        label: `Moderate (${state.moderateBreak.toFixed(2)} to ${state.highBreak.toFixed(2)})`
      },
      {color: SEVERITY_COLORS[6], label: `High (dNBR >= ${state.highBreak.toFixed(2)})`}
    ];
  }
  const ranges = [
    '< -0.25',
    '-0.25 to -0.10',
    '-0.10 to 0.10',
    '0.10 to 0.27',
    '0.27 to 0.44',
    '0.44 to 0.66',
    '> 0.66'
  ];
  return SEVERITY_NAMES.map((name, index) => ({
    color: SEVERITY_COLORS[index],
    label: `${name} (${ranges[index]})`
  }));
};

export default defineScene<BurnSeverityOptions>({
  id: 'burn-severity',
  title: 'Dixie Fire: dNBR severity near Greenville',
  chapter: 'raster',
  order: 1,
  summary:
    'GPU band arithmetic converts pre- and post-fire Sentinel-2 imagery into dNBR, severity classes and patch metrics near Greenville. Canopy response, acquisition dates and 20-m pixels limit interpretation.',
  contributors: [
    'GPURasterArithmetic',
    'GPURasterConditional',
    'GPURasterStretch',
    'GPURasterReclassify',
    'GPURasterPatchMetrics',
    'GPURasterSieve',
    'GPURasterConnectedComponents',
    'GPURasterDenseComponents',
    'GPURasterStatistics'
  ],
  datasets: [{id: 'dixie-fire', role: 'Sentinel-2 bands before and after, WorldCover, DEM'}],
  initialView: {longitude: -121.0, latitude: 40.17, zoom: 11.5},
  basemap: ground('relief'),
  furniture: {
    title: {title: 'Dixie Fire severity', subtitle: 'Sentinel-2 dNBR near Greenville'},
    scaleBar: {units: 'metric'},
    credit: 'Copernicus Sentinel-2; USGS burn-severity thresholds',
    caveat: 'dNBR measures spectral canopy change, not direct ground severity.'
  },

  options: [
    {
      kind: 'select',
      id: 'indexFormula',
      label: 'Index formula',
      group: 'Band math (GPURasterArithmetic)',
      apply: 'param',
      default: 'normalizedDifference',
      help: 'The operation of the arithmetic node. It is a parameter, so changing it re-evaluates NBR and NDVI without recompiling. Thresholds below assume the normalized difference.',
      options: [
        {
          value: 'normalizedDifference',
          label: '(A - B) / (A + B)',
          help: 'Normalized difference: NBR = (NIR - SWIR) / (NIR + SWIR), NDVI = (NIR - Red) / (NIR + Red).'
        },
        {
          value: 'divide',
          label: 'A / B (simple ratio)',
          help: 'Plain band ratio; unbounded, not comparable with USGS classes.'
        },
        {
          value: 'subtract',
          label: 'A - B (raw difference)',
          help: 'Reflectance difference; keeps the brightness signal that normalizing removes.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'changeOperation',
      label: 'Change operation',
      group: 'Band math (GPURasterArithmetic)',
      apply: 'param',
      default: 'subtract',
      help: 'How the before and after indices are combined. dNBR is before minus after, so a burn is positive.',
      options: [
        {
          value: 'subtract',
          label: 'before - after (dNBR)',
          help: 'Signed: positive where the index dropped.'
        },
        {
          value: 'absoluteDifference',
          label: '|before - after|',
          help: 'Unsigned magnitude of change; loses the direction (regrowth looks like burn).'
        }
      ]
    },
    {
      kind: 'select',
      id: 'stretchMode',
      label: 'Stretch',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      default: 'percentile',
      help: 'How the raster values are scaled to 0 to 1 before the ramp. All three modes use the same compiled graph.',
      options: [
        {
          value: 'linear',
          label: 'Linear (min to max)',
          help: 'Exact minimum and maximum of the included cells.'
        },
        {
          value: 'percentile',
          label: 'Percentile clip',
          help: 'Ignores the tails so a few extreme pixels do not flatten the map.'
        },
        {
          value: 'equalize',
          label: 'Histogram equalization',
          help: 'Maps values through the cumulative histogram: equal area per color.'
        }
      ]
    },
    {
      kind: 'range',
      id: 'percentiles',
      label: 'Percentile clip',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      min: 0,
      max: 100,
      step: 1,
      default: [2, 98],
      unit: '%',
      disabledWhen: state => state.stretchMode !== 'percentile',
      help: 'Low and high percentile of the 1024-bin histogram; accurate to one bin width.'
    },
    {
      kind: 'select',
      id: 'domain',
      label: 'Value domain',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      default: 'auto',
      help: 'Auto reads the exact range of the included cells. Fixed uses [-1, 1] for NBR and NDVI and [-0.5, 1.5] for the change layers, so two maps compare directly.',
      options: [
        {value: 'auto', label: 'Automatic (data range)'},
        {value: 'fixed', label: 'Fixed physical range'}
      ]
    },
    {
      kind: 'slider',
      id: 'gamma',
      label: 'Gamma',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      min: 0.3,
      max: 3,
      step: 0.05,
      default: 1,
      help: 'Power applied to the stretched value: below 1 brightens the low end, above 1 darkens it.'
    },
    {
      kind: 'slider',
      id: 'sigmoidContrast',
      label: 'Sigmoidal contrast',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      min: 0,
      max: 20,
      step: 0.5,
      default: 0,
      help: 'rio-color style S-curve. 0 turns it off; higher values separate shadows from highlights around the midpoint.'
    },
    {
      kind: 'slider',
      id: 'sigmoidMidpoint',
      label: 'Sigmoid midpoint',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      min: 0.1,
      max: 0.9,
      step: 0.05,
      default: 0.5,
      disabledWhen: state => state.sigmoidContrast === 0,
      help: 'Stretched value that stays fixed while the contrast steepens around it.'
    },
    {
      kind: 'toggle',
      id: 'stretchToView',
      label: 'Stretch to the visible extent',
      group: 'Contrast stretch (GPURasterStretch)',
      apply: 'param',
      default: false,
      help: 'Rewrites the statistics window from the camera every frame, like QGIS "stretch to visible extent". Zoom into one drainage and the contrast follows it.'
    },
    {
      kind: 'toggle',
      id: 'maskClouds',
      label: 'Screen clouds with the SCL layer',
      group: 'Cloud screen (GPURasterConditional)',
      apply: 'param',
      default: true,
      help: 'A conditional on the Sentinel-2 scene classification of the after image turns cloud pixels into no data.'
    },
    {
      kind: 'slider',
      id: 'cloudClass',
      label: 'Cloud if SCL class is at least',
      group: 'Cloud screen (GPURasterConditional)',
      apply: 'param',
      min: 3,
      max: 11,
      step: 1,
      default: 8,
      disabledWhen: state => !state.maskClouds,
      help: 'SCL codes: 3 cloud shadow, 8 medium cloud, 9 high cloud, 10 thin cirrus, 11 snow. 8 masks all three cloud classes.'
    },
    {
      kind: 'select',
      id: 'comparison',
      label: 'Burned if dNBR is',
      group: 'Burned area (GPURasterConditional)',
      apply: 'param',
      default: '>',
      help: 'The comparison of the conditional that makes the burned mask. `between` is inclusive on both ends.',
      options: [
        {value: '>', label: '> threshold'},
        {value: '>=', label: '>= threshold'},
        {value: 'between', label: 'between threshold and upper'},
        {value: '<', label: '< threshold (regrowth)'}
      ]
    },
    {
      kind: 'slider',
      id: 'threshold',
      label: 'Threshold',
      group: 'Burned area (GPURasterConditional)',
      apply: 'param',
      min: -0.5,
      max: 1.2,
      step: 0.01,
      default: 0.27,
      autoSweep: {from: 0.1, to: 0.66, durationMs: 6000, ease: 'in-out'},
      unit: 'dNBR',
      help: '0.27 is the USGS moderate-low severity boundary. Per-frame parameter: the map follows the slider.'
    },
    {
      kind: 'slider',
      id: 'upperThreshold',
      label: 'Upper threshold',
      group: 'Burned area (GPURasterConditional)',
      apply: 'param',
      min: -0.5,
      max: 1.5,
      step: 0.01,
      default: 0.66,
      unit: 'dNBR',
      disabledWhen: state => state.comparison !== 'between',
      help: 'Upper bound of the `between` comparison, for example moderate severity only: 0.27 to 0.66.'
    },
    {
      kind: 'select',
      id: 'classScheme',
      label: 'Class scheme',
      group: 'Severity classes (GPURasterReclassify)',
      apply: 'param',
      default: 'usgs',
      help: 'Which break table is written into the parameter buffer. The break count changes per frame; the graph does not.',
      options: [
        {
          value: 'usgs',
          label: 'USGS / FIREMON (7 classes)',
          help: 'Breaks at -0.25, -0.10, 0.10, 0.27, 0.44, 0.66.'
        },
        {
          value: 'custom',
          label: 'Custom (4 classes)',
          help: 'Three breaks you set with the sliders below.'
        },
        {value: 'binary', label: 'Binary (1 break)', help: 'One break: the moderate break slider.'}
      ]
    },
    {
      kind: 'select',
      id: 'closed',
      label: 'Closed interval side',
      group: 'Severity classes (GPURasterReclassify)',
      apply: 'param',
      default: 'left',
      help: 'Which side of each class interval owns a value that sits exactly on a break. Left gives [b0, b1); right gives (b0, b1].',
      options: [
        {value: 'left', label: 'Left [a, b)'},
        {value: 'right', label: 'Right (a, b]'}
      ]
    },
    {
      kind: 'slider',
      id: 'lowBreak',
      label: 'Low break',
      group: 'Severity classes (GPURasterReclassify)',
      apply: 'param',
      min: -0.2,
      max: 0.4,
      step: 0.01,
      default: 0.1,
      unit: 'dNBR',
      disabledWhen: state => state.classScheme !== 'custom',
      help: 'Lower bound of low severity (custom scheme).'
    },
    {
      kind: 'slider',
      id: 'moderateBreak',
      label: 'Moderate break',
      group: 'Severity classes (GPURasterReclassify)',
      apply: 'param',
      min: 0.1,
      max: 0.8,
      step: 0.01,
      default: 0.27,
      unit: 'dNBR',
      disabledWhen: state => state.classScheme === 'usgs',
      help: 'Lower bound of moderate severity (custom) or the single break (binary).'
    },
    {
      kind: 'slider',
      id: 'highBreak',
      label: 'High break',
      group: 'Severity classes (GPURasterReclassify)',
      apply: 'param',
      min: 0.4,
      max: 1.4,
      step: 0.01,
      default: 0.66,
      unit: 'dNBR',
      disabledWhen: state => state.classScheme !== 'custom',
      help: 'Lower bound of high severity (custom scheme).'
    },
    {
      kind: 'select',
      id: 'patchTarget',
      label: 'Patches of',
      group: 'Patches (metrics and sieve)',
      apply: 'param',
      default: 'burned',
      help: 'Burned patches measure the scar. Unburned patches are the islands inside it, the ones a sieve fills in.',
      options: [
        {value: 'burned', label: 'Burned pixels'},
        {value: 'unburned', label: 'Unburned pixels (islands)'}
      ]
    },
    {
      kind: 'select',
      id: 'connectivity',
      label: 'Pixel connectivity',
      group: 'Patches (metrics and sieve)',
      apply: 'compile',
      default: '4',
      help: 'Compile-time: the labelling and sieve graphs are compiled for 4 (edges) and 8 (edges and corners) up front and this selects one. With 8, diagonal neighbours join into one patch.',
      options: [
        {value: '4', label: '4 neighbours'},
        {value: '8', label: '8 neighbours'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumPatchHa',
      label: 'Minimum patch size',
      group: 'Patches (metrics and sieve)',
      apply: 'param',
      min: 0.04,
      max: 40,
      step: 0.04,
      default: 0.4,
      unit: 'ha',
      format: value => `${value.toFixed(2)} ha`,
      help: 'The sieve threshold, in hectares (one 20 m pixel is 0.04 ha). Smaller patches are removed; it is a per-frame parameter.'
    },
    {
      kind: 'select',
      id: 'sieveMode',
      label: 'Sieve mode',
      group: 'Patches (metrics and sieve)',
      apply: 'compile',
      default: 'remove',
      help: 'Compile-time: both modes are compiled and this selects the one drawn. Remove clears small patches. Merge hands them to the largest adjacent large patch; clumps of one mask never touch another patch, so here it matches remove.',
      options: [
        {value: 'remove', label: 'Remove small patches'},
        {value: 'merge', label: 'Merge into the largest neighbour'}
      ]
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'dnbr',
      help: 'Every product of the graph stays in GPU memory; this only chooses which buffer the layer draws.',
      options: [
        {
          value: 'dnbr',
          label: 'dNBR (burn severity index)',
          help: 'NBR before minus NBR after, stretched.'
        },
        {
          value: 'severity',
          label: 'Severity classes (USGS)',
          help: 'dNBR reclassified through the break table.'
        },
        {value: 'burned', label: 'Burned mask', help: 'Pixels where the condition on dNBR holds.'},
        {
          value: 'patches',
          label: 'Burn patches',
          help: 'Connected components of the mask, after the sieve.'
        },
        {value: 'nbr-before', label: 'NBR before (2021-07-13)'},
        {value: 'nbr-after', label: 'NBR after (2021-09-21)'},
        {value: 'ndvi-before', label: 'NDVI before (2021-07-13)'},
        {value: 'ndvi-after', label: 'NDVI after (2021-09-21)'},
        {value: 'dndvi', label: 'dNDVI (greenness lost)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'Applies to the continuous layers (indices and dNBR).',
      options: RAMPS,
      disabledWhen: state => ['severity', 'burned', 'patches'].includes(state.layer)
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
      help: 'Lower it to see roads and Lake Almanor under the raster.'
    }
  ],

  readouts: [
    {
      id: 'grid',
      label: 'Grid',
      help: '750 x 750 pixels of 20 m on one UTM zone 10N grid, row 0 north.'
    },
    {
      id: 'validCells',
      label: 'Valid dNBR pixels',
      help: 'Pixels with data in both scenes and not screened as cloud. Counted by GPURasterStatistics.'
    },
    {
      id: 'meanDnbr',
      label: 'Mean dNBR (GPU vs CPU)',
      help: 'GPU mean of the cloud-screened dNBR next to the CPU reference stored with the dataset.'
    },
    {
      id: 'meanNbr',
      label: 'Mean NBR before / after',
      help: 'GPU means with the CPU reference in brackets.'
    },
    {
      id: 'meanNdvi',
      label: 'Mean NDVI before / after',
      help: 'GPU means with the CPU reference in brackets.'
    },
    {id: 'meanDndvi', label: 'Mean dNDVI (GPU vs CPU)'},
    {
      id: 'burnedShare',
      label: 'Burned share (GPU vs CPU)',
      help: 'Fraction of valid pixels passing the burned condition; the reference is dNBR > 0.27.'
    },
    {
      id: 'highShare',
      label: 'High severity share (GPU vs CPU)',
      help: 'Fraction of valid pixels in the top USGS class, dNBR > 0.66.'
    },
    {
      id: 'stretchRange',
      label: 'Stretch bounds',
      help: 'The low and high values the stretch maps to 0 and 1, read from the statistics output.'
    },
    {
      id: 'histogram',
      label: 'Histogram (log counts)',
      help: 'The 1024-bin histogram of the stretch, summed to 40 bars; the stretch maps its cumulative distribution.'
    },
    {
      id: 'classShares',
      label: 'Class shares',
      help: 'Reclassify class counts as a share of valid pixels.'
    },
    {id: 'patchCount', label: 'Patches kept'},
    {id: 'patchArea', label: 'Patch area'},
    {id: 'largestPatch', label: 'Largest patch'},
    {
      id: 'edgeDensity',
      label: 'Edge density',
      help: 'Total perimeter of the kept patches per hectare of patch area (FRAGSTATS ED).'
    },
    {id: 'sieved', label: 'Sieved away'},
    {id: 'selectedPatch', label: 'Selected patch (click)'}
  ],

  legends: state => {
    const continuous = !['severity', 'burned', 'patches'].includes(state.layer);
    const legends = [];
    if (continuous) {
      const titles: Record<string, string> = {
        dnbr: 'dNBR (before - after)',
        'nbr-before': 'NBR before',
        'nbr-after': 'NBR after',
        'ndvi-before': 'NDVI before',
        'ndvi-after': 'NDVI after',
        dndvi: 'dNDVI (before - after)'
      };
      legends.push({
        kind: 'ramp' as const,
        id: 'index',
        title: titles[state.layer] ?? 'Index',
        ramp: state.ramp,
        extent: 'gpu' as const,
        unit: 'index',
        labels:
          state.stretchMode === 'equalize'
            ? (['lowest values', 'highest values'] as const)
            : undefined,
        format: (value: number) => value.toFixed(2)
      });
    } else if (state.layer === 'severity') {
      legends.push({
        kind: 'categories' as const,
        title: 'Burn severity (dNBR)',
        entries: severityLegend(state),
        note: 'Class = number of breaks at or below the value.'
      });
    } else if (state.layer === 'burned') {
      legends.push({
        kind: 'categories' as const,
        title: 'Burned mask',
        entries: [{color: [255, 90, 40, 210] as const, label: 'Condition on dNBR holds'}],
        note: 'Everything else, including screened clouds, is transparent.'
      });
    } else {
      legends.push({
        kind: 'categories' as const,
        title: state.patchTarget === 'burned' ? 'Burn patches' : 'Unburned islands',
        entries: [{color: [78, 201, 255, 235] as const, label: 'Colours repeat every 8 labels'}],
        note: 'Each connected patch gets its own label; sieved patches disappear.'
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterArithmetic, GPURasterConditional, GPURasterReclassify, GPURasterStretch,
  GPURasterConnectedComponents, GPURasterDenseComponents, GPURasterPatchMetrics, GPURasterSieve,
  getGPURasterArithmeticParameterValues, getGPURasterConditionalParameterValues,
  getGPURasterReclassifyParameterValues, getGPURasterStretchParameterValues,
  getGPURasterSieveParameterValues
} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'burn'});
// Band math: bands are float32 digital numbers (0 = no data); scale puts them in reflectance.
graph.add(new GPURasterArithmetic({id: 'nbr-before', cellCount, a: nirBefore, b: swirBefore,
  noDataValue: 0, parameters: indexParameters, output: {values: nbrBefore}}));
graph.add(new GPURasterArithmetic({id: 'nbr-after', cellCount, a: nirAfter, b: swirAfter,
  noDataValue: 0, parameters: indexParameters, output: {values: nbrAfter}}));
graph.add(new GPURasterArithmetic({id: 'dnbr', cellCount, a: nbrBefore, b: nbrAfter,
  parameters: changeParameters, output: {values: dnbr}}));
// Cloud screen and the burned condition
graph.add(new GPURasterConditional({id: 'cloud', cellCount, conditionValues: scl,
  parameters: cloudParameters, output: {values: cloudFlag, mask: cloudMask}}));
graph.add(new GPURasterConditional({id: 'clean', cellCount, mask: cloudMask, b: dnbr,
  parameters: cleanParameters, output: {values: cleanDnbr}}));
graph.add(new GPURasterConditional({id: 'burned', cellCount, conditionValues: cleanDnbr,
  parameters: burnedParameters, output: {values: burnedFlag, mask: burnedMask}}));
graph.add(new GPURasterReclassify({id: 'severity', values: cleanDnbr, breaks, parameters: reclassifyParameters,
  output: {classes, classCounts}}));
const compiled = graph.compile();                         // once

// per frame, no recompile
indexParameters.write(getGPURasterArithmeticParameterValues({
  operation: '${state.indexFormula}', scaleA: 1e-4, scaleB: 1e-4}));
changeParameters.write(getGPURasterArithmeticParameterValues({operation: '${state.changeOperation}'}));
cloudParameters.write(getGPURasterConditionalParameterValues({
  comparison: '>=', threshold: ${state.maskClouds ? state.cloudClass : 99}}));
cleanParameters.write(getGPURasterConditionalParameterValues({constantA: NaN}));
burnedParameters.write(getGPURasterConditionalParameterValues({
  comparison: '${state.comparison}', threshold: ${state.threshold}${state.comparison === 'between' ? `, upperThreshold: ${state.upperThreshold}` : ''}}));
reclassifyParameters.write(getGPURasterReclassifyParameterValues({
  breakCount: ${state.classScheme === 'usgs' ? 6 : state.classScheme === 'custom' ? 3 : 1}, closed: '${state.closed}'}));
stretchParameters.write(getGPURasterStretchParameterValues({
  mode: '${state.stretchMode}', percentiles: [${state.percentiles[0]}, ${state.percentiles[1]}],
  gamma: ${state.gamma}${state.sigmoidContrast ? `, sigmoidContrast: ${state.sigmoidContrast}, sigmoidMidpoint: ${state.sigmoidMidpoint}` : ''}}));
sieveParameters.write(getGPURasterSieveParameterValues({minimumPixels: ${Math.max(1, Math.round(state.minimumPatchHa / 0.04))}}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: 'A chain of small GPU kernels computes burn severity from two Sentinel-2 scenes: `GPURasterArithmetic` evaluates the normalized burn ratio (NBR) of each date and their difference (dNBR), `GPURasterConditional` screens clouds and makes a burned mask, `GPURasterReclassify` sorts dNBR into USGS severity classes, `GPURasterStretch` scales any layer for display, and `GPURasterConnectedComponents` with `GPURasterPatchMetrics` and `GPURasterSieve` measure and clean the burned patches.',
    why: 'dNBR is how fire agencies map severity from space (Key and Benson, FIREMON, 2006). Because every step is a per-frame parameter write, an analyst can move a threshold, change a break or re-stretch the map and see all downstream products update at once.',
    howToRead:
      'Bright (high) dNBR means the near-infrared reflectance of living leaves vanished and shortwave-infrared rose, which is what charred ground and scorched canopy do. Compare the GPU numbers in the readouts with the CPU reference values stored with the dataset: they agree to float32 rounding.'
  },

  create: async ctx => (await import('./burn-severity.compute')).createBurnSeverity(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Dixie Fire dNBR is elevated around Greenville',
      textAlternative:
        'A dNBR raster shows Sentinel-2 spectral change across the Greenville study area.',
      title: 'Where did the Dixie Fire burn hardest around Greenville?',
      body: 'The Dixie Fire started on **13 July 2021** near Cresta Dam, destroyed **Greenville on 4 August** and finally burned about 963,000 acres, the second largest fire in California history. This map covers a 15 km square around Greenville at 20 m resolution.\n\nTwo Sentinel-2 scenes frame the fire: **2021-07-13** (the ignition day) and **2021-09-21**. The colors show **dNBR**, how much the burn ratio fell between them. Brighter means a harder burn; **Map shows** below switches layers. Everything you see is computed live on the GPU from the raw red, near-infrared and shortwave-infrared bands.',
      evidence:
        'The analysis covers **{{grid}}**; **{{validCells}}** survive the paired-scene and cloud mask, with mean **{{meanDnbr}}**.',
      caveat:
        'dNBR is spectral canopy change between two dates, not observed ground severity. The 20 m cells can mix burned and surviving cover.',
      camera: {longitude: -121.0, latitude: 40.17, zoom: 11.5, transitionMs: 1200},
      controls: ['layer', 'opacity'],
      readouts: ['grid', 'validCells', 'meanDnbr'],
      callout: {coordinate: GREENVILLE, text: 'Greenville'},
      highlight: {readout: 'meanDnbr'}
    },
    {
      id: 'band-math',
      headline: 'NBR declines across the mapped burn scar',
      textAlternative:
        'Before and after normalized burn-ratio rasters derive from near- and shortwave-infrared bands.',
      title: 'Step one is band math: the burn ratio',
      body: '`GPURasterArithmetic` evaluates `(A - B) / (A + B)` for every pixel. With A = near-infrared (band 8) and B = shortwave-infrared (band 12) that is the **normalized burn ratio, NBR**. Healthy leaves reflect a lot of near-infrared and little shortwave, so NBR is high (the legend is bright); char and bare soil flip it.\n\nThis is the **before** scene. The same kernel runs again on the after bands, and a third time on red and near-infrared for **NDVI**. Switch **Map shows** below to *NBR after* to see the scar appear, and try **Index formula** for a simple ratio: the operation is a parameter, not a recompile. The readouts quote the CPU reference next to each GPU mean.',
      evidence:
        '**{{meanNbr}}** across the cloud-screened cells; switching before and after preserves one authored display domain.',
      caveat:
        'NBR responds to water, soil, shadow and canopy moisture as well as fire. Band math alone does not identify a causal mechanism.',
      options: {layer: 'nbr-before', stretchMode: 'linear', domain: 'fixed', ramp: 'cividis'},
      controls: ['layer', 'indexFormula'],
      readouts: ['meanNbr'],
      highlight: {readout: 'meanNbr'}
    },
    {
      id: 'differencing',
      headline: 'Positive dNBR marks reduced post-fire vegetation response',
      textAlternative:
        'The raster subtracts post-fire NBR from pre-fire NBR for every valid pixel.',
      title: 'Differencing the dates gives dNBR',
      body: 'A second `GPURasterArithmetic` node subtracts the two NBR rasters: **dNBR = NBR before - NBR after**. Positive values mean the burn ratio fell. The mean over the window is **0.489** on the CPU and the GPU matches it; 63% of pixels exceed 0.27, the USGS moderate-low severity boundary.\n\nThe operation is again a parameter. Switch **Change operation** below to the absolute difference and regrowth becomes indistinguishable from burn, a good reason for keeping the sign. The reference method is `gdal_calc.py` or rasterio band math.',
      evidence:
        'The live paired-scene result is **{{meanDnbr}}**; the signed operation retains the direction that absolute difference discards.',
      caveat:
        'The after image is 70 days after ignition while the fire was still evolving, so this is not a settled final-severity census.',
      options: {layer: 'dnbr'},
      controls: ['changeOperation'],
      readouts: ['meanDnbr'],
      highlight: {readout: 'meanDnbr'}
    },
    {
      id: 'stretch',
      headline: 'Percentile clipping increases visible dNBR contrast',
      textAlternative:
        'A stretched continuous raster redistributes colors across the selected dNBR value range.',
      title: 'A stretch makes the signal readable',
      body: '`GPURasterStretch` computes the exact range, a 1024-bin histogram and its cumulative distribution on the GPU, then maps every pixel to 0 to 1. The default clips the 2nd and 98th percentile so a few extreme pixels do not flatten the map; *Linear* uses the full range and *Histogram equalization* gives each color the same area; pick one with **Stretch** below.\n\nTry **Percentile clip**, **Gamma**, the **Sigmoidal contrast** S-curve, and **Stretch to the visible extent**: zoom into a single canyon and the statistics window is rewritten from the camera every frame, as in QGIS. Percentile bounds are accurate to one histogram bin.',
      evidence:
        'The active stretch maps **{{stretchRange}}** across the displayed histogram; changing the view can recompute that display window.',
      caveat:
        'Stretching changes visual contrast, not dNBR values. A view-dependent stretch cannot support comparisons between different map extents.',
      options: {stretchMode: 'percentile', percentiles: [2, 98], gamma: 1.2, stretchToView: false},
      controls: ['stretchMode', 'percentiles', 'gamma', 'sigmoidContrast', 'stretchToView'],
      readouts: ['stretchRange', 'histogram'],
      highlight: {readout: 'stretchRange'}
    },
    {
      id: 'burned-mask',
      headline: 'The threshold classifies most valid cells as burned',
      textAlternative:
        'A binary raster separates cells above and below the selected dNBR threshold.',
      title: 'A conditional turns the index into a decision',
      body: '`GPURasterConditional` is "where X take A, else B". First it uses the Sentinel-2 scene classification to drop cloud pixels from dNBR, then it compares what is left with a threshold to build the **burned mask**. At the USGS value of 0.27 the burned share is **63.4%**, matching the CPU reference.\n\nToggle **Screen clouds with the SCL layer**, then drag the **Threshold** and the map follows; set **Burned if dNBR is** to *between* to isolate a band of severity, or to *<* to find vegetation that grew. The comparison and thresholds are parameters of one compiled node.',
      evidence:
        '**{{burnedShare}}** among **{{validCells}}**. Play the threshold sweep to see classification sensitivity without changing the underlying raster.',
      caveat:
        'The threshold is a rule, not a discovered boundary. Cloud screening and mixed 20 m cells change the population being classified.',
      options: {layer: 'burned', comparison: '>', threshold: 0.27},
      controls: ['maskClouds', 'threshold', 'comparison'],
      readouts: ['burnedShare', 'validCells'],
      highlight: {readout: 'burnedShare'}
    },
    {
      id: 'severity-classes',
      headline: 'USGS breaks partition dNBR into seven classes',
      textAlternative:
        'Categorical colors encode enhanced regrowth through high-severity dNBR classes.',
      title: 'Reclassify into severity classes',
      body: '`GPURasterReclassify` finds, by binary search in a break table, how many breaks are at or below each value: that count is the class. The USGS table has six breaks and seven classes, from enhanced regrowth through **high severity above 0.66**. The class counts come from integer atomics, so they are exact and repeatable.\n\nSwitch **Class scheme** below to *Custom* and move the breaks (**Low break**, **Moderate break** and **High break** stay greyed out under the USGS scheme and wake up with *Custom*), or flip **Closed interval side** to see which class owns a value that sits exactly on a break. The same node serves every scheme because only the break count and values change.',
      evidence:
        '**{{highShare}}**; the complete partition is **{{classShares}}** and updates from the same break table used by the map.',
      caveat:
        'The published breaks are regional rules of thumb. Moving a break changes labels, not the measured reflectance or ecological condition.',
      options: {layer: 'severity', classScheme: 'usgs'},
      controls: ['classScheme', 'lowBreak', 'moderateBreak', 'highBreak', 'closed'],
      readouts: ['highShare', 'classShares'],
      highlight: {readout: 'highShare'}
    },
    {
      id: 'patches',
      headline: 'Connectivity groups burned cells into measurable patches',
      textAlternative:
        'Connected burned pixels form labeled patches whose areas and perimeters are summarized.',
      title: 'Measure the scar, sieve the specks, mind the limits',
      body: 'The burned mask is labelled into connected **patches** (`GPURasterConnectedComponents`, `GPURasterDenseComponents`), each measured for area, perimeter and bounding box (`GPURasterPatchMetrics`, the FRAGSTATS way), and `GPURasterSieve` drops patches under a minimum size like `gdal_sieve`. Raise **Minimum patch size** below and the specks vanish; switch **Patches of** to *Unburned pixels (islands)* to see the islands inside the scar. Click a patch for its numbers.\n\n**Limits:** Sentinel-2 sees canopy, not ground; a 20 m pixel can hide unburned pockets; dNBR depends on the dates (the after image is 70 days post-ignition, the burn was still growing); and the 0.27 and 0.66 breaks are regional rules of thumb. **Try:** **Pixel connectivity** 8, a 10 ha sieve, the threshold at 0.44, or percentile 0 to 100.',
      evidence:
        'After connectivity and sieving, **{{patchCount}}** remain; **{{largestPatch}}**, while **{{sieved}}** were removed by the current size rule.',
      caveat:
        'Patch identity depends on the threshold, four- versus eight-neighbor connectivity and sieve size. It is a raster topology, not a field-mapped fire perimeter.',
      options: {layer: 'patches', minimumPatchHa: 1.2},
      controls: ['patchTarget', 'minimumPatchHa', 'connectivity', 'sieveMode'],
      readouts: ['patchCount', 'largestPatch', 'sieved'],
      highlight: {readout: 'patchCount'}
    }
  ]
});
