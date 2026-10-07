// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {ChoroplethClassesOptions, ClassesLegendData} from './choropleth-classes.compute';
import {formatCompact, getLegendData, unpackColor} from './b5-legend-bus';
import {CLASS_PALETTE_OPTIONS} from './b5-palettes';
import {COUNTY_VARIABLES, getCountyVariable} from './b5-variables';

const METHOD_OPTIONS = [
  {
    value: 'quantile',
    label: 'Quantile',
    help: 'Same number of counties in every class. Great for ranking, but breaks can split nearly identical values.'
  },
  {
    value: 'equal-interval',
    label: 'Equal interval',
    help: 'Classes of equal width between the minimum and maximum. Honest about the scale, terrible for skewed data.'
  },
  {
    value: 'natural-breaks',
    label: 'Natural breaks (Jenks)',
    help: 'Exact Fisher-Jenks over a histogram: minimises variance inside classes. The bin count is a compile-time option.'
  },
  {
    value: 'standard-deviation',
    label: 'Standard deviation',
    help: 'Classes centred on the mean, one standard deviation wide (adjustable). Best for roughly normal data.'
  },
  {
    value: 'head-tail',
    label: 'Head/tail breaks',
    help: 'For heavy-tailed data: split at the mean, then split the head again while it stays a minority. Produces fewer classes than requested.'
  },
  {
    value: 'box-plot',
    label: 'Box plot (6 classes)',
    help: 'Whisker outliers, the quartile boxes and the median, always six classes.'
  },
  {
    value: 'maximum-breaks',
    label: 'Maximum breaks',
    help: 'Cuts at the largest gaps between sorted values, so genuine clusters stay together.'
  },
  {
    value: 'custom',
    label: 'Custom round-number edges',
    help: 'Fixed, human-friendly edges (such as 10, 20, 30) spanning the 2nd to 98th percentile.'
  }
] as const;

const SCALE_OPTIONS = [
  {
    value: 'threshold',
    label: 'Classed (threshold)',
    help: 'Each class gets one palette color: the classic choropleth.'
  },
  {
    value: 'quantize',
    label: 'Quantize',
    help: 'Equal-width intervals between the first and last break, one per palette entry.'
  },
  {
    value: 'linear',
    label: 'Linear',
    help: 'Continuous scale over the break domain. Turn on blending for smooth gradients.'
  },
  {
    value: 'sqrt',
    label: 'Square root',
    help: 'Lifts small values; a common remedy for right-skewed counts.'
  },
  {value: 'pow', label: 'Power (exponent)', help: 'Generalises the square root: value ^ exponent.'},
  {
    value: 'log',
    label: 'Logarithmic',
    help: 'Equal ratios get equal color steps. Needs positive values; the floor replaces zeros.'
  },
  {
    value: 'symlog',
    label: 'Symmetric log',
    help: 'Log-like for large values and linear near zero, so zeros and negatives are fine.'
  }
] as const;

const isMethod = (state: ChoroplethClassesOptions, ...methods: string[]) =>
  methods.includes(state.method);

export default defineScene<ChoroplethClassesOptions>({
  id: 'choropleth-classes',
  title: 'Choosing classes for a county map',
  chapter: 'statistics',
  order: 2,
  summary:
    'The same county data tells different stories under different class breaks. Compare eight classification methods, color scales, percentile filters and a bivariate value-by-alpha map, all computed on the GPU.',
  contributors: [
    'GPUClassBreaks',
    'GPUClassAssignment',
    'GPUClassificationFit',
    'GPUColorScale',
    'GPUColumnQuantiles',
    'GPUColumnProfile',
    'GPUBivariateClassification'
  ],
  datasets: [{id: 'us-counties', role: 'county polygons and 40 census and health attributes'}],
  initialView: {longitude: -96.5, latitude: 38.2, zoom: 3.55},

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable to map',
      group: 'Data',
      apply: 'param',
      default: 'popDensity',
      help: 'Writes the chosen county column into the values buffer. Nothing is recompiled.',
      options: COUNTY_VARIABLES.map(variable => ({
        value: variable.id,
        label: `${variable.label} (${variable.unit})`
      }))
    },
    {
      kind: 'select',
      id: 'palette',
      label: 'Palette',
      group: 'Data',
      apply: 'param',
      default: 'ylorrd',
      help: 'Packed rgba8 palette buffer sampled to the class count.',
      options: CLASS_PALETTE_OPTIONS
    },
    {
      kind: 'toggle',
      id: 'reversePalette',
      label: 'Reverse palette',
      group: 'Data',
      apply: 'param',
      default: false,
      help: 'Flip the palette when high values should read as light.'
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Classification method',
      group: 'Classes',
      apply: 'param',
      default: 'quantile',
      help: 'The method code is a per-frame parameter; every method is compiled into the graph.',
      options: METHOD_OPTIONS
    },
    {
      kind: 'slider',
      id: 'classCount',
      label: 'Number of classes',
      group: 'Classes',
      apply: 'param',
      min: 2,
      max: 9,
      step: 1,
      default: 5,
      help: 'Requested class count (up to the compiled maximum of 9). Box plot always gives 6; head/tail and maximum breaks may give fewer.'
    },
    {
      kind: 'slider',
      id: 'standardDeviationInterval',
      label: 'Class width',
      group: 'Classes',
      apply: 'param',
      min: 0.25,
      max: 2,
      step: 0.25,
      default: 1,
      unit: 'sd',
      disabledWhen: state => !isMethod(state, 'standard-deviation'),
      help: 'Standard deviation method: width of one class in standard deviations.'
    },
    {
      kind: 'slider',
      id: 'headTailRatio',
      label: 'Head share that keeps splitting',
      group: 'Classes',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.4,
      disabledWhen: state => !isMethod(state, 'head-tail'),
      help: "Head/tail breaks: keep splitting while the head is at most this share of the values. 0.4 is Jiang's rule; 1 is the mapclassify rule."
    },
    {
      kind: 'slider',
      id: 'boxPlotHinge',
      label: 'Whisker length',
      group: 'Classes',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.25,
      default: 1.5,
      unit: '× IQR',
      disabledWhen: state => !isMethod(state, 'box-plot'),
      help: 'Box plot: whiskers extend this many interquartile ranges; counties beyond them are outliers.'
    },
    {
      kind: 'select',
      id: 'naturalBreaksBinCount',
      label: 'Natural breaks histogram bins',
      group: 'Classes',
      apply: 'compile',
      default: '1024',
      disabledWhen: state => !isMethod(state, 'natural-breaks'),
      help: 'Jenks runs exactly on a histogram, so breaks snap to bin edges. More bins are closer to exact Jenks. Compile-time: rebuilds the graph.',
      options: [
        {value: '256', label: '256 bins'},
        {value: '512', label: '512 bins'},
        {value: '1024', label: '1,024 bins'},
        {value: '2048', label: '2,048 bins'}
      ]
    },
    {
      kind: 'button',
      id: 'compareMethods',
      label: 'Compare all methods',
      group: 'Classes',
      help: 'Runs every method in turn at the current class count and fills the goodness-of-fit table in the legend.'
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Color scale',
      group: 'Color scale',
      apply: 'param',
      default: 'threshold',
      help: 'The scale type is a per-frame code of GPUColorScale, so switching never recompiles.',
      options: SCALE_OPTIONS
    },
    {
      kind: 'toggle',
      id: 'smoothBlend',
      label: 'Blend between palette colors',
      group: 'Color scale',
      apply: 'param',
      default: false,
      disabledWhen: state => state.scale === 'threshold',
      help: 'Interpolate adjacent palette entries per channel for the continuous scales instead of stepping.'
    },
    {
      kind: 'toggle',
      id: 'clamp',
      label: 'Clamp outside the domain',
      group: 'Color scale',
      apply: 'param',
      default: true,
      help: 'Out-of-domain values take the first or last color. Off: they become no-data (transparent).'
    },
    {
      kind: 'slider',
      id: 'exponent',
      label: 'Power exponent',
      group: 'Color scale',
      apply: 'param',
      min: 0.2,
      max: 3,
      step: 0.1,
      default: 0.5,
      disabledWhen: state => state.scale !== 'pow',
      help: 'Exponent of the power scale. Below 1 lifts small values, above 1 emphasises the large ones.'
    },
    {
      kind: 'slider',
      id: 'logFloorExponent',
      label: 'Log floor (10 ^ x)',
      group: 'Color scale',
      apply: 'param',
      min: -5,
      max: 1,
      step: 1,
      default: -2,
      disabledWhen: state => state.scale !== 'log',
      help: 'Zero and negative values (and domain ends) are replaced by 10 ^ x before the logarithm.'
    },
    {
      kind: 'slider',
      id: 'lowerPercentile',
      label: 'Lower percentile cut',
      group: 'Percentile filter',
      apply: 'param',
      min: 0,
      max: 49,
      step: 1,
      default: 0,
      format: value => `${value}%`,
      help: 'GPUColumnQuantiles drops counties below this percentile before breaks are computed (deck.gl lowerPercentile semantics).'
    },
    {
      kind: 'slider',
      id: 'upperPercentile',
      label: 'Upper percentile cut',
      group: 'Percentile filter',
      apply: 'param',
      min: 51,
      max: 100,
      step: 1,
      default: 100,
      format: value => `${value}%`,
      help: 'Drops counties above this percentile. Removing the top 1% is the classic cure for one outlier stretching the scale.'
    },
    {
      kind: 'select',
      id: 'quantileInterpolation',
      label: 'Quartile interpolation',
      group: 'Percentile filter',
      apply: 'param',
      default: 'linear',
      help: 'How the quartile readout interpolates between order statistics (numpy and d3 conventions).',
      options: [
        {value: 'linear', label: 'Linear (numpy, d3, R-7)'},
        {value: 'lower', label: 'Lower'},
        {value: 'higher', label: 'Higher'},
        {value: 'nearest', label: 'Nearest'},
        {value: 'midpoint', label: 'Midpoint'}
      ]
    },
    {
      kind: 'select',
      id: 'noDataColor',
      label: 'Filtered-out counties',
      group: 'Percentile filter',
      apply: 'param',
      default: 'transparent',
      help: 'The no-data color of GPUColorScale and GPUBivariateClassification for counties removed by the percentile cut or missing a value.',
      options: [
        {value: 'transparent', label: 'Transparent'},
        {value: 'gray', label: 'Light gray'}
      ]
    },
    {
      kind: 'toggle',
      id: 'bivariate',
      label: 'Bivariate map',
      group: 'Bivariate',
      apply: 'param',
      default: false,
      help: 'GPUBivariateClassification colors each county by its class in the mapped variable (across) and a second variable (up).'
    },
    {
      kind: 'select',
      id: 'bivariateVariable',
      label: 'Second variable',
      group: 'Bivariate',
      apply: 'param',
      default: 'poverty150',
      disabledWhen: state => !state.bivariate,
      help: 'The Y axis of the bivariate palette.',
      options: COUNTY_VARIABLES.map(variable => ({
        value: variable.id,
        label: `${variable.label} (${variable.unit})`
      }))
    },
    {
      kind: 'slider',
      id: 'bivariateClasses',
      label: 'Classes per axis',
      group: 'Bivariate',
      apply: 'param',
      min: 2,
      max: 4,
      step: 1,
      default: 3,
      disabledWhen: state => !state.bivariate,
      help: 'The palette is an n × n grid; 3 × 3 is the cartographic standard.'
    },
    {
      kind: 'select',
      id: 'bivariateMethod',
      label: 'Axis breaks',
      group: 'Bivariate',
      apply: 'param',
      default: 'quantile',
      disabledWhen: state => !state.bivariate,
      help: 'Each axis is classified by its own GPUClassBreaks. Quantile balances the nine cells; equal interval keeps the scales proportional.',
      options: [
        {value: 'quantile', label: 'Quantile'},
        {value: 'equal-interval', label: 'Equal interval'}
      ]
    },
    {
      kind: 'toggle',
      id: 'valueByAlpha',
      label: 'Value-by-alpha (fade small counties)',
      group: 'Bivariate',
      apply: 'param',
      default: false,
      disabledWhen: state => !state.bivariate,
      help: 'Fades each county by its population: fully opaque at 500,000 residents, faintest below 5,000. Uncertain places stop shouting.'
    },
    {
      kind: 'slider',
      id: 'minimumAlpha',
      label: 'Faintest opacity',
      group: 'Bivariate',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.2,
      disabledWhen: state => !state.bivariate || !state.valueByAlpha,
      help: 'Opacity factor for counties at or below 5,000 residents.'
    },
    {
      kind: 'select',
      id: 'histogramBins',
      label: 'Profile histogram bins',
      group: 'Field profile',
      apply: 'compile',
      default: '24',
      help: 'GPUColumnProfile histogram resolution for the distribution readout. Compile-time: rebuilds the graphs.',
      options: [
        {value: '12', label: '12 bins'},
        {value: '24', label: '24 bins'},
        {value: '48', label: '48 bins'},
        {value: '96', label: '96 bins'}
      ]
    },
    {
      kind: 'select',
      id: 'hllPrecision',
      label: 'Distinct-count precision',
      group: 'Field profile',
      apply: 'compile',
      default: '10',
      help: 'HyperLogLog precision p: 2^p registers, relative error about 1.04 / sqrt(2^p). The readout compares the estimate with the exact distinct count. Compile-time.',
      options: [
        {value: '6', label: 'p = 6 (about 13% error)'},
        {value: '8', label: 'p = 8 (about 6.5%)'},
        {value: '10', label: 'p = 10 (about 3.3%)'},
        {value: '12', label: 'p = 12 (about 1.6%)'},
        {value: '14', label: 'p = 14 (about 0.8%)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'County outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draw boundaries over the colors.'
    }
  ],

  readouts: [
    {
      id: 'valid',
      label: 'Counties classified',
      help: 'Counties with a value that pass the percentile filter.'
    },
    {
      id: 'profile',
      label: 'Field profile (GPUColumnProfile)',
      help: 'Mean, standard deviation and an estimated number of distinct values (HyperLogLog).'
    },
    {
      id: 'histogram',
      label: 'Distribution',
      help: 'A 24-bin histogram of the filtered values, from minimum to maximum.'
    },
    {id: 'histogramY', label: 'Second variable distribution'},
    {
      id: 'rucc',
      label: 'Rural-urban codes among the classified counties',
      help: 'A category column of GPUColumnProfile: the top USDA rural-urban continuum codes and their counts follow the percentile filter.'
    },
    {id: 'quartiles', label: 'Quartiles (Q1 / median / Q3)'},
    {id: 'filter', label: 'Percentile filter bounds'},
    {id: 'classes', label: 'Classes produced'},
    {
      id: 'fit',
      label: 'Fit GADF / GVF',
      help: 'Goodness of absolute deviation fit and goodness of variance fit: 1 means classes explain all the variation.'
    },
    {
      id: 'deviations',
      label: 'Deviations ADCM / ADAM',
      help: 'Absolute deviation around class medians and around the array median.'
    },
    {id: 'methodFits', label: 'GADF by method', help: 'Filled by the Compare button.'},
    {id: 'selected', label: 'Selected county', help: 'Click a county to pin its values here.'}
  ],

  legends: state => {
    const data = getLegendData<ClassesLegendData>('choropleth-classes');
    const variable = getCountyVariable(state.variable);
    const legends: LegendSpec[] = [];
    if (!data) {
      return [
        {kind: 'categories', title: variable.label, entries: [], note: 'Classifying on the GPU...'}
      ];
    }
    if (state.bivariate) {
      const other = getCountyVariable(state.bivariateVariable);
      const size = data.bivariate?.n ?? state.bivariateClasses;
      const entries = [];
      for (let y = size - 1; y >= 0; y--) {
        for (let x = 0; x < size; x++) {
          const index = y * size + x;
          const level = (value: number, last: number) =>
            value === 0 ? 'low' : value === last ? 'high' : 'mid';
          entries.push({
            color: unpackColor(data.bivariate!.colors[index]),
            label: `${variable.label.split(' ')[0]} ${level(x, size - 1)}, ${other.label.split(' ')[0]} ${level(y, size - 1)}: ${data.bivariate!.counts[index] ?? 0} counties`
          });
        }
      }
      legends.push({
        kind: 'categories',
        title: `${variable.label} (across) by ${other.label} (up)`,
        entries,
        note: `${variable.label} edges ${data.bivariate!.edgesX.map(formatCompact).join(' · ')} ${variable.unit}; ${other.label} edges ${data.bivariate!.edgesY.map(formatCompact).join(' · ')} ${other.unit}.${state.valueByAlpha ? ' Opacity follows population.' : ''}`
      });
    } else {
      const entries = [];
      for (let index = 0; index < data.classCount; index++) {
        entries.push({
          color: unpackColor(data.colors[index] ?? 0),
          label: `${formatCompact(data.breaks[index])} to ${formatCompact(data.breaks[index + 1])}: ${data.counts[index] ?? 0} counties`
        });
      }
      legends.push({
        kind: 'categories',
        title: `${variable.label} (${variable.unit})`,
        entries,
        note: `${data.methodLabel}, ${data.classCount} classes${state.scale === 'threshold' ? '' : `, ${SCALE_OPTIONS.find(option => option.value === state.scale)?.label.toLowerCase()} scale (class ranges follow the scale transform)`}.`
      });
    }
    if (data.fits.length > 1) {
      legends.push({
        kind: 'categories',
        title: `Goodness of absolute deviation fit at ${data.fitClassCount} classes (higher is better)`,
        entries: data.fits.map(fit => ({
          color: [90 + Math.round(fit.gadf * 150), 140, 255 - Math.round(fit.gadf * 120), 255],
          label: `${fit.label}: GADF ${fit.gadf.toFixed(3)}, GVF ${fit.gvf.toFixed(3)}`
        }))
      });
    }
    return legends;
  },

  snippet: state => `import {
  GPUClassBreaks, GPUColorScale, GPUColumnQuantiles,
  getGPUClassBreaksParameterValues, getGPUColorScaleParameterValues,
  getGPUColumnQuantilesParameterValues
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUClassAssignment, GPUClassificationFit} from '@luma.gl/experimental/gpu-spatial-analysis';

// One graph, compiled once: percentile filter -> breaks -> colors.
graph.add(new GPUColumnQuantiles({
  values, parameters: quantileParameters, quantileCount: 3,
  output: {quantiles, validCount, filterMask, filterBounds}
}));
graph.add(new GPUClassBreaks({
  values, mask: filterMask, parameters: breaksParameters,
  maximumClassCount: 9, naturalBreaksBinCount: ${state.naturalBreaksBinCount},
  output: {breaks, classCount}
}));
graph.add(new GPUColorScale({
  values, mask: filterMask, domain: breaks, domainCount: classCount,
  palette, parameters: scaleParameters,
  maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors, classIndices, classCounts}   // colors: packed rgba8 for the layer
}));
graph.add(new GPUClassAssignment({values, breaks, classCount, mask: filterMask, output: classes}));
graph.add(new GPUClassificationFit({values, classes, classCount: 9, output: fit}));

// Per frame: parameter writes only.
quantileParameters.write(getGPUColumnQuantilesParameterValues({
  quantiles: [0.25, 0.5, 0.75], interpolation: '${state.quantileInterpolation}',
  filterRange: [${state.lowerPercentile / 100}, ${state.upperPercentile / 100}]
}));
breaksParameters.write(getGPUClassBreaksParameterValues(
  {method: '${state.method}', classCount: ${state.classCount}}, 9));
scaleParameters.write(getGPUColorScaleParameterValues({
  scale: '${state.scale}', domainCount: ${state.classCount + 1}, paletteCount: ${state.classCount},
  interpolation: '${state.smoothBlend && state.scale !== 'threshold' ? 'linear' : 'step'}', clamp: ${state.clamp}
}));${
    state.bivariate
      ? `

// Bivariate: two GPUClassBreaks axes, then the n x n palette.
graph.add(new GPUBivariateClassification({
  valuesX, valuesY, breaksX, breaksY, palette: bivariatePalette,${state.valueByAlpha ? '\n  alphaValues: population,' : ''}
  parameters, maximumClassCount: 4, output: {colors: bivariateColors, classCounts}
}));
bivariateParameters.write(getGPUBivariateClassificationParameterValues({
  classCountX: ${state.bivariateClasses}, classCountY: ${state.bivariateClasses}${state.valueByAlpha ? `,\n  valueByAlpha: {domain: [5000, 500000], minimumAlpha: ${state.minimumAlpha}}` : ''}
}));`
      : ''
  }`,

  about: {
    what: '`GPUColumnQuantiles` finds exact quantiles and a percentile range, `GPUClassBreaks` computes class edges by one of eight methods, `GPUColorScale` turns values into packed rgba8 colors, `GPUClassAssignment` and `GPUClassificationFit` score the classes, `GPUBivariateClassification` colors two variables at once and `GPUColumnProfile` summarises the field. Everything stays in GPU buffers; the map layer reads the packed colors directly.',
    why: 'Choropleth classing is a modelling decision. The same data yields a "one county is special" map under equal intervals and a "half the country is dark" map under quantiles. Seeing the methods side by side, with a fit score, turns the choice into evidence.',
    howToRead:
      'The legend lists every class range with its county count. GADF and GVF run from 0 to 1: higher means counties inside a class are more alike. Matches mapclassify (Fisher-Jenks, quantiles, box plot, head/tail, maximum breaks) and d3 color scales.'
  },

  create: async ctx => (await import('./choropleth-classes.compute')).createChoroplethClasses(ctx),

  story: [
    {
      id: 'question',
      title: 'How do you color a map of population density?',
      body: 'Population density in US counties runs from 0.03 to 28,000 people per square kilometre. We want one color per class: five classes, equal intervals, the "obvious" choice.\n\n`GPUClassBreaks` computes the class edges on the GPU and `GPUColorScale` paints the counties. Look at the map and the legend: nearly every county falls in the lowest class. One enormous value, New York County, stretches the scale.',
      options: {variable: 'popDensity', method: 'equal-interval', classCount: 5, palette: 'ylorrd'},
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['variable', 'method', 'classCount'],
      readouts: ['classes'],
      highlight: {readout: 'classes'}
    },
    {
      id: 'quantile',
      title: 'Quantiles: equal counts, arbitrary edges',
      body: 'Switch **Classification method** to *Quantile*. Each of the five classes now holds about 620 counties, so the map shows rank, not magnitude: the Plains and Mountain West are visibly sparse and the East dense.\n\nThe price: the top class spans from 160 to 28,000 people per km², and two counties with nearly equal density can land in different classes. The class edges come from `GPUColumnQuantiles`-style order statistics computed exactly on the GPU.',
      options: {method: 'quantile'},
      controls: ['method', 'classCount'],
      readouts: ['classes']
    },
    {
      id: 'head-tail',
      title: 'Heavy tails: head/tail breaks',
      body: '**Head/tail breaks** were designed for exactly this shape. Split the values at the mean; the *head* (values above the mean) is a small minority, so split it again, and keep going while the head stays under 40% of what is left (**Head share that keeps splitting**). The edges 0 to 329, 329 to 1,200, 1,200 to 3,500 and so on reveal the structure of the tail: suburbs, cities, then Manhattan.\n\nThe readouts **Fit GADF / GVF** come from `GPUClassificationFit`. GADF is the share of absolute deviation around the median that the classes explain: it jumped from 0.24 (equal interval) to about 0.6.',
      options: {method: 'head-tail', classCount: 6},
      controls: ['method', 'headTailRatio', 'classCount'],
      readouts: ['classes', 'fit'],
      highlight: {readout: 'fit'}
    },
    {
      id: 'compare',
      title: 'Compare every method',
      body: 'Press **Compare all methods**. The tool runs each method at six classes and the legend fills a goodness-of-fit table. It is the same compiled graph each time with a different method code in the parameter buffer.\n\nOn this skewed variable *head/tail breaks* and *natural breaks (Jenks)* lead on GADF, *quantile* and *box plot* score low because they ignore magnitude. GVF flatters *equal interval* (0.86) because variance is dominated by one outlier: judge a scheme by GADF and by looking at the map. Natural breaks is exact Fisher-Jenks over a histogram; its **Natural breaks histogram bins** option is a compile-time setting that rebuilds the graph.',
      options: {method: 'natural-breaks', classCount: 6},
      controls: ['compareMethods', 'method', 'naturalBreaksBinCount'],
      readouts: ['methodFits'],
      highlight: {readout: 'methodFits'}
    },
    {
      id: 'scale-and-filter',
      title: 'Color scales and the percentile filter',
      body: 'Now map **Variable to map** = *Median household income* (a much tamer variable). **Color scale** controls how classes become colors: *Classed* gives one flat color per class; *Linear* with *Blend between palette colors* gives a continuous gradient from the same breaks; *Square root* and *Log* stretch the low end.\n\nThe **Upper percentile cut** drops the richest counties before the breaks are computed, so a few extremes cannot set the scale. Counties filtered out turn transparent, or gray if you change **Filtered-out counties**, and the **Rural-urban codes** readout shows how the cut changes who is left.',
      options: {
        variable: 'medianHouseholdIncome',
        method: 'quantile',
        palette: 'viridis',
        scale: 'linear',
        smoothBlend: true,
        upperPercentile: 98
      },
      camera: {longitude: -96.5, latitude: 38.2, zoom: 3.55},
      controls: ['variable', 'scale', 'smoothBlend', 'upperPercentile', 'noDataColor'],
      readouts: ['rucc']
    },
    {
      id: 'bivariate',
      title: 'Two variables at once, with uncertainty',
      body: '`GPUBivariateClassification` classifies two variables on their own axes and colors each county from a 3 × 3 palette: income across, poverty up. Teal counties (low income, high poverty) gather in the Deep South, Appalachia and along the Rio Grande; rose counties (high income, low poverty) ring the big metros. The dark corner (high on both) is nearly empty because the two variables move in opposite directions. The legend counts each cell.\n\nChange **Second variable** to compare other pairs, and turn on **Value-by-alpha (fade small counties)**: counties fade with their population, so a 500-person county no longer shouts as loudly as Los Angeles.',
      options: {
        variable: 'medianHouseholdIncome',
        method: 'quantile',
        palette: 'viridis',
        scale: 'threshold',
        smoothBlend: false,
        upperPercentile: 100,
        bivariate: true,
        bivariateVariable: 'poverty150',
        valueByAlpha: true
      },
      camera: {longitude: -88, latitude: 35, zoom: 4.4},
      controls: ['bivariate', 'bivariateVariable', 'valueByAlpha']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: 'Class breaks change what a map says, not what the data says: always show the legend. Natural breaks on a different class count gives different edges; quantiles and Jenks are not comparable across maps. A percentile cut hides real counties.\n\nTry, on this diabetes-prevalence map: change the **Class width** of the *Standard deviation* classes; set **Classification method** to *Box plot* and lengthen **Whisker length**; turn **Bivariate map** back on and set **Second variable** to *Aged 65 and over*; click a county to read its class.',
      options: {
        bivariate: false,
        valueByAlpha: false,
        variable: 'places_diabetes_ageAdj',
        method: 'standard-deviation',
        palette: 'spectral',
        scale: 'threshold'
      },
      controls: [
        'standardDeviationInterval',
        'method',
        'boxPlotHinge',
        'bivariate',
        'bivariateVariable'
      ]
    }
  ]
});
