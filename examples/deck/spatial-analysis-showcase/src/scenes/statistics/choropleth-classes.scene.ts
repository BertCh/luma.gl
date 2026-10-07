// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, US} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatCount} from '../../cartography/live-text';
import {NATIONAL_FURNITURE} from '../../cartography/projection-notes';
import {BIVARIATE_PALETTES} from '../../engine/ramps';
import {defineScene, type LegendSpec} from '../scene';
import {COUNTY_VARIABLES, getCountyVariable} from './b5-variables';
import type {ChoroplethClassesOptions, ClassesLegendData} from './choropleth-classes.compute';
import {formatClassValue} from './choropleth-classes.style';

/** The contiguous US, fitted to the free map area on desktop and phone. */
const CONUS_BOUNDS = [-124.8, 24.4, -66.9, 49.4] as const;
/** The Deep South, Appalachia and the Rio Grande Valley of the bivariate step. */
const DEEP_SOUTH_BOUNDS = [-100, 25.2, -77, 39.5] as const;

const CREDIT = joinCredits(
  CREDITS.usCensus,
  'ACS 2018-2022 via CDC/ATSDR SVI 2022; SAIPE 2022 via USDA ERS; CDC PLACES 2024 (public domain)',
  CREDITS.colorBrewer
);

/** The cartouche of one step; the live subtitle, sample line and chips refine it from the data. */
const cartouche = (title: string, subtitle: string, chips: readonly string[] = ['Estimates']) => ({
  title,
  subtitle,
  chips
});

const METHOD_OPTIONS = [
  {
    value: 'equal-interval',
    label: 'Equal interval',
    help: 'Classes of equal width between the minimum and maximum. Honest about the scale, terrible for skewed data.'
  },
  {
    value: 'quantile',
    label: 'Quantile',
    help: 'Same number of counties in every class. Great for ranking, but breaks can split nearly identical values.'
  },
  {
    value: 'natural-breaks',
    label: 'Natural breaks',
    help: 'Exact Fisher-Jenks over a histogram of equal-width bins: minimises variance inside classes. The bin count is a compile-time option.'
  },
  {
    value: 'head-tail',
    label: 'Head/tail',
    help: 'For heavy-tailed data: split at the mean, then split the head again while it stays a minority. May produce fewer classes than requested.'
  },
  {
    value: 'standard-deviation',
    label: 'Std. deviation',
    help: 'Classes centred on the county mean, one standard deviation wide (adjustable), in a diverging scheme with a neutral middle class. Best for roughly normal data.'
  },
  {
    value: 'box-plot',
    label: 'Box plot',
    help: 'Whisker outliers, the quartile boxes and the median: six classes in a diverging scheme around the median.'
  },
  {
    value: 'maximum-breaks',
    label: 'Max breaks',
    help: 'Cuts at the largest gaps between sorted values, so genuine clusters stay together.'
  },
  {
    value: 'custom',
    label: 'Round numbers',
    help: 'Fixed, human-friendly edges spanning the 2nd to 98th percentile.'
  }
] as const;

const SCALE_OPTIONS = [
  {
    value: 'threshold',
    label: 'Classed (threshold)',
    help: 'Each class gets one palette colour: the classic choropleth.'
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
    help: 'Equal ratios get equal colour steps. Needs positive values; the floor replaces zeros.'
  },
  {
    value: 'symlog',
    label: 'Symmetric log',
    help: 'Log-like for large values and linear near zero, so zeros and negatives are fine.'
  }
] as const;

const isMethod = (state: ChoroplethClassesOptions, ...methods: string[]) =>
  methods.includes(state.method);

/** Orientation labels of the national steps: one state per quarter of the map. */
const STATE_LABELS = ['state-ca', 'state-mt', 'state-tx', 'state-mn', 'state-ga', 'state-ny'];

function getLegends(
  state: ChoroplethClassesOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const variable = getCountyVariable(state.variable);
  const legend = data['classes'] as ClassesLegendData | undefined;
  if (!legend) {
    return [
      {kind: 'categories', title: variable.label, entries: [], note: 'Classifying on the GPU...'}
    ];
  }
  if (state.bivariate) {
    const other = getCountyVariable(state.bivariateVariable);
    const edges = (values: number[]) => values.slice(1, -1).map(formatClassValue).join(' | ');
    const specs: LegendSpec[] = [
      {
        kind: 'bivariate',
        id: 'bivariate',
        title: `${variable.label} by ${other.label}`,
        size: 3,
        colors: BIVARIATE_PALETTES.tealPink,
        xLabel: variable.label,
        yLabel: other.label,
        xEnds: ['lower', 'higher'],
        yEnds: ['lower', 'higher'],
        note: `${state.bivariateMethod === 'quantile' ? 'Tertile' : 'Equal-interval'} edges: ${variable.label} ${edges(legend.bivariate?.edgesX ?? [])} ${variable.unit}; ${other.label} ${edges(legend.bivariate?.edgesY ?? [])} ${other.unit}.`
      }
    ];
    if (state.valueByAlpha) {
      specs.push({
        kind: 'alpha',
        id: 'alpha',
        title: 'Opacity follows population',
        colors: [
          BIVARIATE_PALETTES.tealPink[6],
          BIVARIATE_PALETTES.tealPink[2],
          BIVARIATE_PALETTES.tealPink[8]
        ],
        ends: ['fewer residents', 'more residents'],
        alphaRange: [state.minimumAlpha, 1],
        steps: 4,
        note: 'Log scale of residents per county.'
      });
    }
    return specs;
  }
  const title =
    state.variable === 'population' ? 'Residents (a count, not a rate)' : variable.label;
  if (legend.side === 'a' && legend.baseline) {
    // The toggle compare holds the frozen side: the legend names what the map shows.
    return [
      getClassTableLegend(legend.baseline, {
        title,
        id: 'classes',
        layout: 'bar',
        histogram: legend.histogram,
        note: `${legend.baseline.method}, frozen. Counties per class: ${legend.baselineCounts.map(count => formatCount(count)).join(' | ')}.`
      })
    ];
  }
  const diverging = state.method === 'standard-deviation' || state.method === 'box-plot';
  const notes = [
    legend.table.method,
    diverging
      ? state.method === 'box-plot'
        ? 'Orange above the median, purple below.'
        : 'Red above the county mean, blue below; the middle class is within half a class width of it.'
      : undefined,
    `Counties per class: ${legend.counts.map(count => formatCount(count)).join(' | ')}.`,
    legend.baseline ? `Compared with: ${legend.baseline.method}, frozen.` : undefined
  ];
  return [
    getClassTableLegend(legend.table, {
      title,
      id: 'classes',
      layout: 'bar',
      histogram: legend.histogram,
      interactive: state.scale === 'threshold',
      note: notes.filter(Boolean).join(' ')
    })
  ];
}

function getSnippet(state: ChoroplethClassesOptions): string {
  return `import {
  GPUClassBreaks, GPUColorScale, GPUColumnQuantiles, GPUColumnProfile,
  getGPUClassBreaksParameterValues, getGPUColorScaleParameterValues,
  getGPUColumnQuantilesParameterValues
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUClassAssignment, GPUClassificationFit} from '@luma.gl/experimental/gpu-spatial-analysis';

// One graph, compiled once: percentile filter -> breaks -> colours.
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
  palette,                       // the exact ColorBrewer table of the scheme, packed rgba8
  parameters: scaleParameters, maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors, classIndices, classCounts}   // colors: packed rgba8 for the layer
}));
graph.add(new GPUClassAssignment({values, breaks, classCount, mask: filterMask, output: classes}));
graph.add(new GPUClassificationFit({values, classes, classCount: 9, output: fit}));
graph.add(new GPUColumnProfile({columns: [{values}], mask: filterMask, histogramBinCount: ${state.histogramBins}, output: profile}));

// Per frame: parameter writes only (switching the method never recompiles).
breaksParameters.write(getGPUClassBreaksParameterValues(
  {method: '${state.method}', classCount: ${state.classCount}${state.method === 'standard-deviation' ? `, standardDeviationInterval: ${state.standardDeviationInterval}` : ''}}, 9));
scaleParameters.write(getGPUColorScaleParameterValues({
  scale: '${state.scale}', domainCount: ${state.classCount + 1}, paletteCount: ${state.classCount},
  interpolation: '${state.smoothBlend && state.scale !== 'threshold' ? 'linear' : 'step'}', clamp: ${state.clamp}, noDataColor
}));
quantileParameters.write(getGPUColumnQuantilesParameterValues({
  quantiles: [0.25, 0.5, 0.75], filterRange: [${state.lowerPercentile / 100}, ${state.upperPercentile / 100}]
}));

// The layer reads the packed colours as they are: no readback to render.
new SpatialAnalysisPolygonLayer({triangles, features, vertexCount, values: colors,
  valueFormat: 'uint32', colormap: 'rgba', opacity: 1});${
    state.bivariate
      ? `

// Bivariate: two GPUClassBreaks axes (tertiles), then a 3 x 3 palette.
graph.add(new GPUBivariateClassification({
  valuesX, valuesY, breaksX, breaksY, palette: tealPink,${state.valueByAlpha ? '\n  alphaValues: logPopulation,' : ''}
  parameters, maximumClassCount: 4, output: {colors: bivariateColors, classCounts}
}));${
          state.valueByAlpha
            ? `
bivariateParameters.write(getGPUBivariateClassificationParameterValues({
  classCountX: 3, classCountY: 3,
  valueByAlpha: {domain: [Math.log10(5000), Math.log10(500000)], minimumAlpha: ${state.minimumAlpha}}
}));`
            : ''
        }`
      : ''
  }`;
}

/**
 * Choropleth classes: the same county data under eight classification methods on the GPU, with
 * exact ColorBrewer tables, a swipe compare, a histogram legend and a bivariate map. GPU work is in
 * `choropleth-classes.compute.ts`.
 */
export default defineScene<ChoroplethClassesOptions>({
  id: 'choropleth-classes',
  title: 'Classes are a choice',
  chapter: 'statistics',
  order: 1,
  summary:
    'The same county data tells different stories under different class edges. Compare classification methods on the GPU with a swipe, see the edges move over the histogram, diverge around a real midpoint, and read two variables in one bivariate legend.',
  contributors: [
    'GPUColumnQuantiles',
    'GPUClassBreaks',
    'GPUColorScale',
    'GPUClassAssignment',
    'GPUClassificationFit',
    'GPUColumnProfile',
    'GPUBivariateClassification'
  ],
  datasets: [
    {id: 'us-counties', role: 'county polygons and the census and health attributes'},
    {id: 'us-states', role: 'state lines over the counties'}
  ],
  initialView: {longitude: -96, latitude: 38.3, zoom: 3.9},

  // An atlas page: no tiles, a flat sheet, opaque fills, our state lines and labels.
  basemap: ground('paperSheet'),
  furniture: {
    ...NATIONAL_FURNITURE,
    title: cartouche('One county stretches the scale', 'Population density, US counties'),
    credit: CREDIT
  },

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable to map',
      group: 'Data',
      apply: 'param',
      default: 'popDensity',
      help: 'Writes the chosen county column into the values buffer. Each question has its own hue: people in yellow-brown, income in blue, health burden in pink-purple. Nothing is recompiled.',
      options: COUNTY_VARIABLES.map(variable => ({
        value: variable.id,
        label: `${variable.label} (${variable.unit})`
      }))
    },
    {
      kind: 'preset',
      id: 'shape',
      label: 'Variable',
      group: 'Data',
      help: 'Three variables with three distribution shapes, each in its own hue.',
      presets: [
        {label: 'Population density', values: {variable: 'popDensity'}},
        {label: 'Median income', values: {variable: 'medianHouseholdIncome'}},
        {label: 'Diabetes', values: {variable: 'places_diabetes_ageAdj'}}
      ]
    },
    {
      kind: 'preset',
      id: 'normalise',
      label: 'Variable',
      group: 'Data',
      help: 'Residents (a count) or residents per km² (a density) in the same hue and the same classes.',
      presets: [
        {label: 'Residents (counts)', values: {variable: 'population'}},
        {label: 'People per km² (density)', values: {variable: 'popDensity'}}
      ]
    },
    {
      kind: 'preset',
      id: 'methodChoice',
      label: 'Method',
      group: 'Classes',
      help: 'Four classification methods. The legend ticks move over the histogram when you switch.',
      presets: [
        {label: 'Equal interval', values: {method: 'equal-interval'}},
        {label: 'Quantile', values: {method: 'quantile'}},
        {label: 'Natural breaks', values: {method: 'natural-breaks'}},
        {label: 'Head/tail', values: {method: 'head-tail'}}
      ]
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Classification method',
      group: 'Classes',
      apply: 'param',
      default: 'equal-interval',
      display: 'chips',
      help: 'The method code is a per-frame parameter; every method is compiled into the graph. Standard deviation and box plot switch to a diverging scheme with a real midpoint.',
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
      help: 'Requested class count (up to the compiled maximum of 9). Box plot always gives 6; head/tail and maximum breaks may give fewer. ColorBrewer publishes tables for 3 to 9 sequential classes.'
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
      help: 'Standard deviation method: width of one class in standard deviations. With an odd class count the middle class spans half a width either side of the mean.'
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
      expert: true,
      disabledWhen: state => !isMethod(state, 'natural-breaks'),
      help: 'Jenks runs exactly on a histogram of equal-width bins, so breaks snap to bin edges. More bins are closer to exact Jenks. Compile-time: rebuilds the graph.',
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
      help: 'Runs every method in turn at the current class count and draws the fit (GADF) of each as bars.'
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Colour scale',
      group: 'Colour scale',
      apply: 'param',
      default: 'threshold',
      help: 'The scale type is a per-frame code of GPUColorScale, so switching never recompiles.',
      options: SCALE_OPTIONS
    },
    {
      kind: 'toggle',
      id: 'smoothBlend',
      label: 'Blend between palette colours',
      group: 'Colour scale',
      apply: 'param',
      default: false,
      disabledWhen: state => state.scale === 'threshold',
      help: 'Interpolate adjacent palette entries per channel for the continuous scales instead of stepping.'
    },
    {
      kind: 'toggle',
      id: 'clamp',
      label: 'Clamp outside the domain',
      group: 'Colour scale',
      apply: 'param',
      default: true,
      help: 'Out-of-domain values take the first or last colour. Off: they become no-data (grey).'
    },
    {
      kind: 'slider',
      id: 'exponent',
      label: 'Power exponent',
      group: 'Colour scale',
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
      group: 'Colour scale',
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
      group: 'Percentile cut',
      apply: 'param',
      min: 0,
      max: 49,
      step: 1,
      default: 0,
      format: value => `${value}%`,
      help: 'GPUColumnQuantiles drops counties below this percentile before breaks are computed (deck.gl lowerPercentile semantics). They are drawn grey and named in the legend.'
    },
    {
      kind: 'slider',
      id: 'upperPercentile',
      label: 'Upper percentile cut',
      group: 'Percentile cut',
      apply: 'param',
      min: 51,
      max: 100,
      step: 1,
      default: 100,
      format: value => `${value}%`,
      help: 'Drops counties above this percentile. Removing the top 1% is the classic cure for one outlier stretching the scale, and it hides real counties.'
    },
    {
      kind: 'select',
      id: 'quantileInterpolation',
      label: 'Quartile interpolation',
      group: 'Percentile cut',
      apply: 'param',
      default: 'linear',
      expert: true,
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
      kind: 'toggle',
      id: 'bivariate',
      label: 'Bivariate map',
      group: 'Bivariate',
      apply: 'param',
      default: false,
      help: 'GPUBivariateClassification colours each county by its tertile in the mapped variable (across) and a second variable (up). Off: the single-variable map.'
    },
    {
      kind: 'select',
      id: 'bivariateVariable',
      label: 'Second variable',
      group: 'Bivariate',
      apply: 'param',
      default: 'poverty150',
      disabledWhen: state => !state.bivariate,
      help: 'The y axis of the bivariate palette.',
      options: COUNTY_VARIABLES.map(variable => ({
        value: variable.id,
        label: `${variable.label} (${variable.unit})`
      }))
    },
    {
      kind: 'select',
      id: 'bivariateMethod',
      label: 'Axis breaks',
      group: 'Bivariate',
      apply: 'param',
      default: 'quantile',
      display: 'segmented',
      disabledWhen: state => !state.bivariate,
      help: 'Each axis is classified by its own GPUClassBreaks. Quantile gives tertiles and balances the nine cells; equal interval keeps the scales proportional.',
      options: [
        {value: 'quantile', label: 'Tertiles'},
        {value: 'equal-interval', label: 'Equal interval'}
      ]
    },
    {
      kind: 'toggle',
      id: 'valueByAlpha',
      label: 'Fade small counties',
      group: 'Bivariate',
      apply: 'param',
      default: false,
      disabledWhen: state => !state.bivariate,
      help: 'Value-by-alpha: opacity follows the logarithm of the population, fully opaque at 500,000 residents and faintest at 5,000 or fewer. Uncertain places stop shouting.'
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
      default: 0.3,
      disabledWhen: state => !state.bivariate || !state.valueByAlpha,
      help: 'Opacity factor for counties at or below 5,000 residents.'
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'County boundaries',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A thin hairline between counties. State lines are always drawn.'
    },
    {
      kind: 'select',
      id: 'compareBaseline',
      label: 'Swipe baseline',
      group: 'Display',
      apply: 'param',
      default: 'none',
      expert: true,
      help: 'The frozen classification drawn left of the swipe divider. The story sets it; its breaks are computed once and do not follow the method.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'equal-interval', label: 'Equal interval'},
        {value: 'quantile', label: 'Quantile'}
      ]
    },
    {
      kind: 'select',
      id: 'annotationSet',
      label: 'Map notes',
      group: 'Display',
      apply: 'param',
      default: 'none',
      expert: true,
      help: 'Which counties the story labels from the data: the extremes, three named counties and their class, or the counts-versus-density pair.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'outlier', label: 'The largest value'},
        {value: 'methods', label: 'Cook, Maricopa, Loudoun'},
        {value: 'extremes', label: 'Highest and lowest'},
        {value: 'counts', label: 'Los Angeles and an empty county'}
      ]
    },
    {
      kind: 'select',
      id: 'histogramBins',
      label: 'Histogram bins',
      group: 'Field profile',
      apply: 'compile',
      default: '48',
      expert: true,
      help: 'GPUColumnProfile histogram resolution for the legend strip and the distribution chart. Compile-time: rebuilds the graphs.',
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
      expert: true,
      help: 'HyperLogLog precision p: 2^p registers, relative error about 1.04 / sqrt(2^p). The profile readout compares the estimate with the exact distinct count. Compile-time.',
      options: [
        {value: '6', label: 'p = 6 (about 13% error)'},
        {value: '8', label: 'p = 8 (about 6.5%)'},
        {value: '10', label: 'p = 10 (about 3.3%)'},
        {value: '12', label: 'p = 12 (about 1.6%)'},
        {value: '14', label: 'p = 14 (about 0.8%)'}
      ]
    }
  ],

  readouts: [
    {
      id: 'lowestShare',
      label: 'Counties in the lowest class',
      emphasis: 'tile',
      help: 'Share of the classified counties that fall in the first class.'
    },
    {
      id: 'outsideLowest',
      label: 'Counties above the lowest class',
      help: 'Counties that carry all the other colours of the map.'
    },
    {id: 'largest', label: 'Largest value', help: 'The maximum of the mapped variable.'},
    {
      id: 'classEdges',
      label: 'Class edges',
      help: 'Lower edge, the interior breaks and the upper edge, in the units of the variable.'
    },
    {
      id: 'gadf',
      label: 'Fit of this method (GADF)',
      emphasis: 'tile',
      help: 'Goodness of absolute deviation fit from GPUClassificationFit: the share of the absolute deviation around the median that the classes explain. 1 means counties inside a class are identical.'
    },
    {
      id: 'gadfBaseline',
      label: 'Fit of the frozen baseline (GADF)',
      help: 'The same score for the classification drawn left of the swipe divider, computed on the same counties.'
    },
    {
      id: 'gvf',
      label: 'Fit of this method (GVF)',
      help: 'Goodness of variance fit. Variance is dominated by outliers, so it flatters equal intervals on a heavy tail.',
      hood: true
    },
    {
      id: 'distribution',
      label: 'Distribution of the variable',
      kind: 'chart',
      help: 'Counties by value in equal-width bins from GPUColumnProfile; the ticks are the class edges and the bars take the colour of their class. Heavy-tailed variables use a log axis.'
    },
    {
      id: 'binsUsed',
      label: 'Natural-breaks bins in use',
      help: 'Bins of the natural-breaks histogram that hold at least one county. The bins are equal in value, so a heavy tail leaves most of them empty.'
    },
    {
      id: 'firstBinShare',
      label: 'Counties in the first bin',
      help: 'Share of counties in the lowest of the natural-breaks bins.'
    },
    {
      id: 'meanSd',
      label: 'County mean and standard deviation',
      help: 'Unweighted over counties: every county counts once, whatever its population.'
    },
    {
      id: 'neutralShare',
      label: 'Counties in the middle class',
      help: 'Share of counties in the neutral middle class of the diverging scheme.'
    },
    {
      id: 'topAreaShare',
      label: 'Land area of the darkest class',
      help: 'Share of the land area of the classified counties that carries the darkest colour.'
    },
    {
      id: 'topPopulationShare',
      label: 'Residents in the darkest class',
      help: 'Share of the residents of the classified counties that live in the darkest class.'
    },
    {
      id: 'deprived',
      label: 'Pink corner: low across, high up',
      help: 'Counties in the lowest class of the first variable and the highest class of the second: for income by poverty, low income and high poverty.'
    },
    {
      id: 'bothHigh',
      label: 'Top class on both',
      help: 'Counties in the highest class of both variables.'
    },
    {
      id: 'smallCounties',
      label: 'Counties below the opacity floor',
      help: 'Counties with fewer residents than the faint end of the value-by-alpha scale.'
    },
    {
      id: 'methodFits',
      label: 'Fit of every method (GADF)',
      kind: 'chart',
      help: 'Filled by the Compare all methods button.'
    },
    {id: 'valid', label: 'Counties classified', hood: true},
    {
      id: 'profile',
      label: 'Field profile (GPUColumnProfile)',
      hood: true,
      help: 'Mean, standard deviation and an estimated number of distinct values (HyperLogLog).'
    },
    {
      id: 'rucc',
      label: 'Rural-urban codes among the classified counties',
      hood: true,
      help: 'A category column of GPUColumnProfile: the top USDA rural-urban continuum codes and their counts follow the percentile cut.'
    },
    {id: 'quartiles', label: 'Quartiles (Q1 / median / Q3)', hood: true},
    {id: 'filter', label: 'Percentile cut bounds', hood: true},
    {id: 'classes', label: 'Classes produced', hood: true},
    {
      id: 'deviations',
      label: 'Deviations ADCM / ADAM',
      hood: true,
      help: 'Absolute deviation around class medians and around the array median.'
    },
    {
      id: 'selected',
      label: 'Selected county',
      hood: true,
      help: 'Click a county to pin its values here.'
    }
  ],

  pipeline: [
    {
      id: 'quantiles',
      label: 'Quantiles',
      detail: 'Exact order statistics by four radix-select passes over the values: no sort'
    },
    {
      id: 'breaks',
      label: 'Breaks',
      detail: 'One kernel set for every method; the method code is a per-frame parameter'
    },
    {
      id: 'scale',
      label: 'Colour scale',
      detail: 'Binary search over the edges; packed rgba8 read by the layer, no readback to render'
    },
    {
      id: 'fit',
      label: 'Fit',
      detail: 'Class assignment, then GADF and GVF from class medians and means'
    }
  ],

  legends: getLegends,

  snippet: getSnippet,

  about: {
    what: '`GPUColumnQuantiles` finds exact quantiles and a percentile cut, `GPUClassBreaks` computes class edges by one of eight methods, `GPUColorScale` turns values into packed rgba8 colours from an exact ColorBrewer table, `GPUClassAssignment` and `GPUClassificationFit` score the classes, `GPUBivariateClassification` colours two variables at once and `GPUColumnProfile` supplies the histogram under the legend. Everything stays in GPU buffers; the map layer reads the packed colours directly. The swipe compares the live classification with a frozen equal-interval or quantile one over the same counties.',
    why: 'Choropleth classing is a modelling decision. The same data yields a "one county is special" map under equal intervals and a "half the country is dark" map under quantiles. Seeing the methods side by side, with the class edges over the histogram and a fit score, turns the choice into evidence.',
    howToRead:
      'The legend swatches are the classes, the histogram above them is the distribution, and its ticks are the class edges. GADF and GVF run from 0 to 1: higher means counties inside a class are more alike. Try the Class width of the standard-deviation classes, the Box plot whisker length, the Percentile cut (cut counties turn grey), a continuous Colour scale, or other pairs in the bivariate map. Click a county to pin its values. Methods match mapclassify (Fisher-Jenks, quantiles, box plot, head/tail, maximum breaks) and d3 colour scales.'
  },

  create: async ctx => (await import('./choropleth-classes.compute')).createChoroplethClasses(ctx),

  story: [
    {
      id: 'one-outlier',
      title: 'Equal intervals on a heavy tail',
      headline: 'One county stretches the whole scale',
      textAlternative:
        'Map of US counties in equal-width density classes: almost every county is the palest colour and a handful of dense counties are darker.',
      body: 'Equal-width classes of population density put **{{lowestShare}}** of counties in the palest one. Everything above it holds only **{{outsideLowest}}**, because the densest county, **{{largest}}**, sets the top of the scale. More classes with **Number of classes** do not fix it.\n\n*Equal intervals describe the range, not the counties.*',
      optionsMode: 'fresh',
      options: {
        variable: 'popDensity',
        method: 'equal-interval',
        classCount: 5,
        annotationSet: 'outlier'
      },
      controls: ['classCount'],
      readouts: ['lowestShare', 'outsideLowest', 'largest', 'classEdges'],
      stage: 'breaks',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1600},
      furniture: {
        title: cartouche('One county stretches the scale', 'Population density, equal intervals')
      },
      annotations: labelsFor(US, STATE_LABELS)
    },
    {
      id: 'same-data',
      title: 'Same counties, different edges',
      headline: 'The same counties tell four different stories',
      textAlternative:
        'The same county density map split by a divider: equal intervals on the left leave nearly every county pale, the method you choose on the right puts more counties in the darker classes.',
      body: 'Left of the divider, equal intervals; right, the method you pick with **Method**. Same counties, same hue, different edges: the fit (GADF, higher is better) is **{{gadf}}** for this method against **{{gadfBaseline}}** for equal intervals. Watch Cook, Maricopa and Loudoun change class, and set the count with **Number of classes**.\n\n*Classification is an argument; the data do not settle it.*',
      optionsMode: 'fresh',
      options: {
        variable: 'popDensity',
        method: 'head-tail',
        classCount: 5,
        compareBaseline: 'equal-interval',
        annotationSet: 'methods'
      },
      controls: ['methodChoice', 'classCount'],
      readouts: ['gadf', 'gadfBaseline', 'classEdges'],
      stage: 'fit',
      compare: {labels: ['Equal interval', 'The method you choose'], position: 0.5},
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: {
        title: cartouche('Same counties, four classifications', 'Population density, two methods')
      },
      annotations: labelsFor(US, ['state-ca', 'state-mt', 'state-ga'])
    },
    {
      id: 'shape-of-data',
      title: 'Choose the method by the shape',
      headline: 'Heavy tails break natural breaks; bell curves do not',
      textAlternative:
        'County map in natural-breaks classes with a histogram of the variable: on population density almost all counties sit in the first bin, on income and diabetes they spread across the histogram.',
      body: 'Natural breaks cut where the histogram thins, but their bins are equal in value. On this variable **{{firstBinShare}}** of counties sit in the first bin and only **{{binsUsed}}** bins are used. Pick another **Variable** and the counties spread out; or change **Method**: heavy tails suit head/tail breaks, bell curves suit quantiles or standard deviations.\n\n*Match the method to the shape of the data.*',
      optionsMode: 'fresh',
      options: {
        variable: 'popDensity',
        method: 'natural-breaks',
        classCount: 5,
        annotationSet: 'extremes'
      },
      controls: ['shape', 'methodChoice'],
      readouts: ['distribution', 'firstBinShare', 'binsUsed', 'gadf'],
      stage: 'breaks',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: {title: cartouche('Which method fits the shape?', 'Natural breaks')},
      annotations: labelsFor(US, ['state-ca', 'state-tx', 'state-mn', 'state-ny'])
    },
    {
      id: 'midpoint',
      title: 'A midpoint needs a meaning',
      headline: 'Diverge only around a meaningful middle',
      textAlternative:
        'County map of diabetes prevalence: one-hue quantile classes, and with the compare button held, red and blue standard-deviation classes around the county mean with a pale middle class.',
      body: 'Press and hold the compare button to swap quantile classes (one hue, darker is more) for standard-deviation classes around the county mean of **{{meanSd}}**. The pale middle class holds **{{neutralShare}}** of counties; red is above the mean, blue below. Change **Class width** to widen the middle.\n\n*Diverge only around a real midpoint.*',
      optionsMode: 'fresh',
      options: {
        variable: 'places_diabetes_ageAdj',
        method: 'standard-deviation',
        classCount: 7,
        standardDeviationInterval: 1,
        compareBaseline: 'quantile',
        annotationSet: 'extremes'
      },
      controls: ['standardDeviationInterval'],
      readouts: ['distribution', 'meanSd', 'neutralShare'],
      stage: 'scale',
      compare: {
        mode: 'toggle',
        labels: ['Sequential quantiles', 'SD classes around the mean']
      },
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: {
        title: cartouche('Above or below the county mean?', 'Diabetes prevalence', ['Modelled'])
      },
      annotations: labelsFor(US, ['black-belt', 'front-range', 'state-ca', 'state-ny'])
    },
    {
      id: 'counts-vs-density',
      title: 'Count or rate?',
      headline: 'Counts on unequal areas map size, not density',
      textAlternative:
        'County map of residents in quantile classes: the darkest class covers more land than the darkest class of the density map, which is limited to the crowded places.',
      body: 'The darkest class of the variable covers **{{topAreaShare}}** of the land and holds **{{topPopulationShare}}** of residents. Large counties turn dark because they hold more people, not because they are crowded. Switch **Variable** to density and the first number falls: darkness now means crowding.\n\n*Map rates, not counts.*',
      optionsMode: 'fresh',
      options: {
        variable: 'population',
        method: 'quantile',
        classCount: 5,
        annotationSet: 'counts'
      },
      controls: ['normalise'],
      readouts: ['topAreaShare', 'topPopulationShare', 'classEdges'],
      stage: 'scale',
      camera: {bounds: CONUS_BOUNDS, transitionMs: 1400},
      furniture: {title: cartouche('Count or density?', 'Residents per county')},
      annotations: labelsFor(US, ['state-ca', 'state-tx', 'state-mt'])
    },
    {
      id: 'two-variables',
      title: 'Two variables, one legend',
      headline: 'Low income and high poverty go together',
      textAlternative:
        'Bivariate county map of the South in nine colours: pink counties with low income and high poverty fill the Black Belt, eastern Kentucky and the Rio Grande Valley.',
      body: 'Income runs across and poverty up, each in tertiles. **{{deprived}}** sit in the pink corner (low income, high poverty); only **{{bothHigh}}** are high on both. Switch **Bivariate map** off for the two single maps, try another **Second variable**, or **Fade small counties** so opacity follows population and small counties stop shouting (more in *Small numbers, loud maps*).\n\n*Two variables, one legend.*',
      optionsMode: 'fresh',
      options: {
        variable: 'medianHouseholdIncome',
        method: 'quantile',
        classCount: 3,
        bivariate: true,
        bivariateVariable: 'poverty150',
        bivariateMethod: 'quantile',
        valueByAlpha: false
      },
      controls: ['bivariate', 'bivariateVariable', 'valueByAlpha'],
      readouts: ['deprived', 'bothHigh', 'smallCounties'],
      stage: 'scale',
      camera: {bounds: DEEP_SOUTH_BOUNDS, transitionMs: 1800},
      furniture: {
        title: cartouche('Where do low income and poverty meet?', 'Income by poverty, tertiles')
      },
      annotations: labelsFor(US, [
        'black-belt',
        'appalachia',
        'rio-grande-valley',
        'state-ms',
        'state-ga',
        'state-tn'
      ])
    }
  ]
});
