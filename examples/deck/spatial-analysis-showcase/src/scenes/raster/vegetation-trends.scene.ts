// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {VegetationTrendsOptions} from './vegetation-trends.compute';

/** Acquisition dates of the 16 slices (`properties.dates` of the dataset). */
const DATES = [
  '2017-09-02',
  '2018-07-29',
  '2018-08-08',
  '2019-08-23',
  '2020-06-03',
  '2020-08-02',
  '2021-06-03',
  '2021-07-13',
  '2021-09-11',
  '2021-09-21',
  '2022-06-23',
  '2022-08-07',
  '2023-07-13',
  '2023-08-02',
  '2024-06-22',
  '2024-08-01'
] as const;

const formatDate = (value: number) => DATES[Math.round(value)] ?? String(value);

const SIGNED_RANGE: Record<string, number> = {
  senSlope: 0.04,
  mannKendallZ: 4,
  tStatistic: 8,
  difference: 0.6,
  percentChange: 100,
  logRatio: 1
};

const SIGNED_TITLES: Record<string, string> = {
  senSlope: 'Theil-Sen slope (NDVI per date step)',
  mannKendallZ: 'Mann-Kendall Z',
  tStatistic: 'Welch t statistic (after - before)',
  difference: 'NDVI difference (after - before)',
  percentChange: 'Percent change (after vs before)',
  logRatio: 'Log ratio ln((after + e) / (before + e))'
};

export default defineScene<VegetationTrendsOptions>({
  id: 'vegetation-trends',
  title: 'Greenville vegetation: NDVI trend across 16 dates',
  chapter: 'raster',
  order: 3,
  summary:
    'GPU change detection applies Welch, Mann-Kendall and Theil-Sen statistics to 16 Sentinel-2 NDVI dates around Greenville. The outputs describe sampled-date trends, with cloud, season and serial dependence limitations.',
  contributors: ['GPUChangeDetection'],
  // `ndvi-timeseries` is fetched by the scene itself (see the handoff: the shared catalog cannot
  // decode its raw uint8 stack); restore `datasets: [{id: 'ndvi-timeseries', ...}]` once it can.
  datasets: [],
  initialView: {longitude: -121.0, latitude: 40.17, zoom: 10.6},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Greenville vegetation trends',
      subtitle: 'NDVI change across 16 Sentinel-2 dates'
    },
    scaleBar: {units: 'metric'},
    credit: 'Copernicus Sentinel-2',
    caveat: 'Sparse seasonal acquisitions do not isolate fire recovery from phenology or weather.'
  },

  options: [
    {
      kind: 'slider',
      id: 'beforeSlice',
      label: 'Before date',
      group: 'Two-date comparison',
      apply: 'param',
      min: 0,
      max: 15,
      step: 1,
      default: 7,
      format: formatDate,
      help: 'For the difference, percent change and log ratio. 2021-07-13 is the day the fire started.'
    },
    {
      kind: 'slider',
      id: 'afterSlice',
      label: 'After date',
      group: 'Two-date comparison',
      apply: 'param',
      min: 0,
      max: 15,
      step: 1,
      default: 9,
      autoSweep: {from: 7, to: 15, durationMs: 6500, ease: 'in-out'},
      format: formatDate,
      help: '2021-09-21 is the post-fire image used in the burn severity story.'
    },
    {
      kind: 'slider',
      id: 'epsilon',
      label: 'Log ratio guard (epsilon)',
      group: 'Two-date comparison',
      apply: 'param',
      min: 0.001,
      max: 0.3,
      step: 0.001,
      default: 0.05,
      format: value => value.toFixed(3),
      help: 'Added to both dates in ln((after + e) / (before + e)); NDVI can be near zero, so a guard keeps the ratio finite.'
    },
    {
      kind: 'select',
      id: 'significanceSource',
      label: 'Significance from',
      group: 'Test (GPUChangeDetection)',
      apply: 'compile',
      default: 't-test',
      help: 'Compile-time: the contributor decides the class from either statistic. Both graphs are compiled up front and this selects the one that runs.',
      options: [
        {
          value: 't-test',
          label: 'Welch t-test (before vs after groups)',
          help: 'Compares the mean of the dates before the split with the mean after it.'
        },
        {
          value: 'mann-kendall',
          label: 'Mann-Kendall (monotonic trend)',
          help: 'Rank test for a consistent upward or downward drift over all dates.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'alpha',
      label: 'Significance level (alpha)',
      group: 'Test (GPUChangeDetection)',
      apply: 'param',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      default: 0.05,
      format: value => value.toFixed(3),
      help: 'Two-sided p-value threshold of the significance class. Lower alpha demands stronger evidence. Not corrected for testing 65,000 cells.'
    },
    {
      kind: 'slider',
      id: 'splitSlice',
      label: 'First date of the "after" group',
      group: 'Test (GPUChangeDetection)',
      apply: 'param',
      min: 1,
      max: 15,
      step: 1,
      default: 8,
      format: formatDate,
      help: 'The Welch t-test compares dates before this one with this one and later. The default 2021-09-11 puts the fire between the groups.'
    },
    {
      kind: 'range',
      id: 'dateWindow',
      label: 'Dates used',
      group: 'Series',
      apply: 'param',
      min: 0,
      max: 15,
      step: 1,
      default: [0, 15],
      format: formatDate,
      help: 'Dates outside the window become NaN in the stack (a buffer write), and every test drops them. Use 2021-09-21 to 2024-08-01 for a pure recovery trend.'
    },
    {
      kind: 'slider',
      id: 'minimumValid',
      label: 'Minimum valid dates per pixel',
      group: 'Series',
      apply: 'param',
      min: 2,
      max: 16,
      step: 1,
      default: 8,
      help: 'The cell mask: pixels with fewer cloud-free dates inside the window are skipped (the contributor writes NaN and class 0).'
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'significance',
      help: 'Every statistic stays in GPU memory; this picks the buffer the layer draws.',
      options: [
        {
          value: 'significance',
          label: 'Significant trend (class)',
          help: 'Green: significant increase. Red: significant decrease. Grey: no trend at this alpha.'
        },
        {
          value: 'senSlope',
          label: 'Theil-Sen slope',
          help: 'Median of the pairwise slopes: a robust size of the trend.'
        },
        {value: 'mannKendallZ', label: 'Mann-Kendall Z'},
        {value: 'mannKendallP', label: 'Mann-Kendall p-value'},
        {value: 'tStatistic', label: 'Welch t statistic'},
        {value: 'tPValue', label: 'Welch t p-value'},
        {value: 'difference', label: 'Difference: after - before date'},
        {value: 'percentChange', label: 'Percent change between the two dates'},
        {value: 'logRatio', label: 'Log ratio between the two dates'},
        {value: 'ndviBefore', label: 'NDVI on the before date'},
        {value: 'ndviAfter', label: 'NDVI on the after date'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'For NDVI and the p-values. The signed statistics always use the diverging ramp, centred on zero.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    },
    {
      kind: 'slider',
      id: 'colorScale',
      label: 'Color range scale',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1,
      unit: 'x',
      help: 'Multiplies the half-width of the color range. Below 1 stretches contrast around zero; above 1 shows the extremes.'
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
      id: 'meanSeries',
      label: 'Window mean NDVI, 16 dates',
      help: 'Mean NDVI of the whole 25.6 km window on each date (from the dataset): the fire is the step in 2021.'
    },
    {
      id: 'acquisitionContract',
      label: 'Evidence · acquisition coverage',
      help: 'The number and date span of the irregular summer acquisitions, plus the range of valid-pixel coverage after cloud screening.'
    },
    {id: 'analysed', label: 'Cells analysed'},
    {
      id: 'shares',
      label: 'Significant trends',
      help: 'Share of the analysed cells in each class for the current test and alpha, counted from the GPU class raster.'
    },
    {
      id: 'falsePositives',
      label: 'Multiple testing',
      help: 'At alpha 0.05, about one cell in twenty passes by chance.'
    },
    {
      id: 'parity',
      label: 'GPU vs CPU (400 pixels)',
      help: 'A seeded sample of pixels is recomputed in float64 on the CPU: Mann-Kendall S, Z, Theil-Sen slope and Welch t.'
    },
    {id: 'pixel', label: 'Selected pixel (click)'},
    {
      id: 'pixelSeries',
      label: 'Pixel NDVI series and statistics',
      help: 'Sparkline of the 16 dates (a dot is a masked or cloudy date) with the GPU statistics and the CPU values in brackets.'
    }
  ],

  legends: state => {
    const legends: Array<
      | {
          kind: 'ramp';
          title: string;
          ramp: VegetationTrendsOptions['ramp'] | 'diverging';
          extent: readonly [number, number];
          labels?: readonly [string, string];
          format?: (value: number) => string;
        }
      | {
          kind: 'categories';
          title: string;
          entries: {color: readonly [number, number, number, number]; label: string}[];
          note?: string;
        }
    > = [];
    if (state.layer === 'significance') {
      legends.push({
        kind: 'categories',
        title: `Trend at alpha ${state.alpha.toFixed(3)}`,
        entries: [
          {color: [38, 166, 91, 235], label: 'Significant increase (greening)'},
          {color: [214, 69, 65, 235], label: 'Significant decrease (browning)'},
          {color: [140, 140, 140, 90], label: 'No significant trend or masked'}
        ],
        note:
          state.significanceSource === 't-test'
            ? 'Welch t-test between the dates before and after the split.'
            : 'Mann-Kendall test over all dates in the window.'
      });
    } else if (state.layer === 'ndviBefore' || state.layer === 'ndviAfter') {
      legends.push({
        kind: 'ramp',
        title: `NDVI on ${formatDate(state.layer === 'ndviBefore' ? state.beforeSlice : state.afterSlice)}`,
        ramp: state.ramp,
        extent: [-0.1, 0.9],
        format: v => v.toFixed(1)
      });
    } else if (state.layer === 'tPValue' || state.layer === 'mannKendallP') {
      legends.push({
        kind: 'ramp',
        title: 'p-value (dark = small = significant)',
        ramp: state.ramp,
        extent: [0, 0.2 * state.colorScale],
        format: v => v.toFixed(2)
      });
    } else {
      const range = (SIGNED_RANGE[state.layer] ?? 1) * state.colorScale;
      legends.push({
        kind: 'ramp',
        title: SIGNED_TITLES[state.layer] ?? 'Value',
        ramp: 'diverging',
        extent: [-range, range],
        labels: ['browning', 'greening'],
        format: v => (Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1))
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUChangeDetection, getGPUChangeDetectionParameterValues}
  from '@luma.gl/experimental/gpu-raster';

// slices: float32, cell-major (cell * 16 + date), NaN = cloud or no data
const graph = new GPUCommandGraph(device, {id: 'trends'});
graph.add(new GPUChangeDetection({
  slices, mask,                          // mask: uint32 per cell, 0 skips the cell
  parameters, cellCount, sliceCount: 16,
  significanceSource: '${state.significanceSource}',   // compile-time
  output: {
    difference, logRatio, percentChange,                // two dates
    tStatistic, tDegreesOfFreedom, tPValue,              // Welch groups
    senSlope,                                            // Theil-Sen, exact median
    mannKendallS, mannKendallZ, mannKendallP,            // tie-corrected
    significance                                         // 0 none, 1 increase, 2 decrease
  }
}));
const compiled = graph.compile();                         // once

// per frame, no recompile
parameters.write(getGPUChangeDetectionParameterValues({
  beforeSlice: ${state.beforeSlice}, afterSlice: ${state.afterSlice},
  splitSlice: ${state.splitSlice}, alpha: ${state.alpha}, epsilon: ${state.epsilon}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUChangeDetection` runs one thread per pixel over its 16 NDVI values: two-date difference, log ratio and percent change; a Welch two-sample t-test between the dates before and after a split; the Theil-Sen slope (exact median of all pairwise slopes); and the Mann-Kendall S, Z and p. A significance class (increase, decrease, none) comes from the t-test or from Mann-Kendall.',
    why: 'A single before and after picture says how big a change is; a trend test says whether it is real given the noise of 16 imperfect images. The Mann-Kendall test and the Theil-Sen slope are the standard non-parametric tools for greening and browning maps because they ignore outliers such as an unmasked cloud.',
    howToRead:
      'Red cells are significantly browning, green cells greening, grey cells indistinguishable from noise at the chosen alpha. The fire is a step, not a ramp: the t-test (split at the fire) sees it sharply, Mann-Kendall sees a monotone decline over all dates. The GPU values agree with a float64 CPU recomputation on a random sample of pixels.\n\n*Data: the `ndvi-timeseries` dataset, contains modified Copernicus Sentinel data 2017 to 2024 (ESA): 16 cloud-screened Sentinel-2 L2A summer scenes at 100 m.*'
  },

  create: async ctx => (await import('./vegetation-trends.compute')).createVegetationTrends(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'NDVI trajectories vary across the burn scar',
      textAlternative:
        'A per-pixel vegetation trend raster summarizes 16 Sentinel-2 acquisition dates around Greenville.',
      title: 'Has the forest around Greenville come back since the Dixie Fire?',
      body: 'Sixteen cloud-screened Sentinel-2 summer scenes, from **2017 to 2024**, give every 100 m pixel around Greenville a time series of NDVI, a measure of green leaf cover. The **Dixie Fire** burned through in July to October 2021.\n\n`GPUChangeDetection` tests all 65,000 series at once. The map shows where the NDVI trend is **significantly down** (red), **significantly up** (green) or indistinguishable from noise (grey). The sparkline in the panel is the window mean: a step down in 2021 and a partial recovery. **Map shows** below picks the statistic to draw.',
      evidence:
        '**{{acquisitionContract}}**. The window mean series shows the 2021 step and the sampled trajectory that follows.',
      caveat:
        'Sixteen summer acquisitions are a sparse, irregular sample. They cannot separate fire recovery from rainfall, phenology, sun angle or residual cloud effects.',
      camera: {longitude: -121.0, latitude: 40.17, zoom: 10.6, transitionMs: 1200},
      controls: ['layer'],
      readouts: ['meanSeries', 'acquisitionContract'],
      highlight: {readout: 'meanSeries'}
    },
    {
      id: 'two-dates',
      headline: 'Post-fire NDVI falls across much of Greenville',
      textAlternative:
        'A signed raster subtracts pre-fire NDVI from a selected post-fire observation.',
      title: 'Start with two dates: the fire as a difference',
      body: 'The simplest change detector subtracts one date from another. `GPUChangeDetection` writes the **difference**, the **log ratio** and the **percent change** between the *before* date (**2021-07-13**, the day the fire started) and the *after* date (**2021-09-21**).\n\nBlue is NDVI lost. A two-date map has no idea of noise: a hazy morning or a different sun angle also shows up. Drag **Before date** and **After date** below to see the difference move, and try the log ratio in **Map shows**: its guard, **Log ratio guard (epsilon)**, keeps near-zero NDVI finite.',
      evidence:
        '**{{acquisitionContract}}**. Play **After date** to watch the signed change depend on which post-fire observation closes the pair.',
      caveat:
        'A two-date difference has no sampling distribution. Atmospheric residue, phenology and illumination can move it along with vegetation.',
      options: {layer: 'difference', beforeSlice: 7, afterSlice: 9},
      controls: ['beforeSlice', 'afterSlice', 'layer', 'epsilon'],
      readouts: ['acquisitionContract']
    },
    {
      id: 't-test',
      headline: 'Welch statistics separate periods with unequal variance',
      textAlternative: 'Per-pixel t statistics compare selected before and after NDVI date groups.',
      title: 'A Welch t-test asks whether before and after really differ',
      body: 'The t-test splits each series at a date: slices before it are one group, the rest the other. It compares the means, allowing different variances (**Welch**), and converts the **t statistic** into a p-value with the regularized incomplete beta function (the contributor is accurate to about 1e-4 for the degrees of freedom here).\n\nWith **First date of the "after" group** at **2021-09-11** the fire sits between the groups, and the scar lights up red. **Significance level (alpha)** is the threshold: lower it and only the strongest changes keep their color. Cells need at least two valid dates in each group.',
      evidence:
        '**{{shares}}** among **{{analysed}}**. Both the class and its denominator update when the split, alpha or valid-date rule changes.',
      caveat:
        'The per-pixel p-values are uncorrected across roughly 65,000 spatially dependent tests, and the before/after grouping still treats acquisition dates as exchangeable samples.',
      options: {layer: 'significance', significanceSource: 't-test', splitSlice: 8, alpha: 0.05},
      controls: ['splitSlice', 'alpha'],
      readouts: ['shares', 'analysed'],
      highlight: {readout: 'shares'}
    },
    {
      id: 'mann-kendall',
      headline: 'Mann-Kendall identifies monotonic sampled-date trends',
      textAlternative:
        'A diverging raster maps positive and negative rank-based NDVI trend statistics.',
      title: 'Mann-Kendall tests for a drift without assuming a shape',
      body: 'Mann-Kendall counts, over every pair of dates, whether NDVI rose or fell: **S = sum of sign(x_j - x_i)**. A strongly positive or negative S means a consistent drift. The variance is corrected for ties, and **Z** (with continuity correction) becomes a normal p-value.\n\nSwitching **Significance from** below to Mann-Kendall is a *compile-time* option of the contributor, so the control is marked *rebuild*; the two graphs were compiled up front and this just selects one. The pattern differs: Mann-Kendall favours steady trends, so slowly recovering or dying pixels appear that the step test misses.',
      evidence:
        '**{{shares}}**. The seeded CPU parity sample reports **{{parity}}**, keeping the GPU statistic auditable.',
      caveat:
        'Mann-Kendall tests monotonic ordering on the sampled dates; it does not model the 2021 intervention, unequal time gaps or spatial dependence between neighboring pixels.',
      options: {layer: 'significance', significanceSource: 'mann-kendall'},
      controls: ['significanceSource', 'alpha'],
      readouts: ['shares', 'parity'],
      highlight: {readout: 'parity'}
    },
    {
      id: 'sen-slope',
      headline: 'Theil-Sen estimates median NDVI change per date',
      textAlternative:
        'A signed slope raster reports the median pairwise NDVI change at each pixel.',
      title: 'Theil-Sen gives the size of the trend',
      body: 'A p-value says *whether*; the **Theil-Sen slope** says *how much*: the median of the slopes `(x_j - x_i) / (j - i)` over all pairs of dates. Because it is a median, one cloudy outlier barely moves it, unlike a least-squares line. The contributor selects the exact median inside each thread with a bounded heap (limited to 64 dates).\n\nThe unit is **NDVI per date step**, and the dates are not evenly spaced (two per summer), so read it as a rate on the ordinal axis. Use **Color range scale** below to stretch the contrast. Click any pixel: the readout shows its series with the GPU and the CPU values side by side.',
      evidence:
        'For the clicked pixel, **{{pixelSeries}}** shows all sampled values beside the GPU and CPU slope and test statistics.',
      caveat:
        'The slope unit is NDVI per acquisition step, not per day or year. Unequal gaps mean equal horizontal steps do not represent equal elapsed time.',
      options: {layer: 'senSlope', significanceSource: 'mann-kendall'},
      controls: ['layer', 'colorScale'],
      readouts: ['pixelSeries', 'pixel'],
      highlight: {readout: 'pixelSeries'}
    },
    {
      id: 'recovery',
      headline: 'Later date windows alter estimated vegetation recovery',
      textAlternative:
        'Restricting the acquisition window recalculates post-fire NDVI trend estimates.',
      title: 'Restrict the dates to see the recovery',
      body: 'The 2021 step dominates a trend over all dates. To ask about regrowth alone, restrict **Dates used** below to **2021-09-21 or later**. The contributor drops the excluded dates (they become NaN in the stack, one buffer write) and the tests run on what is left.\n\nNow green cells are real recovery: significant greening since the fire. They are fewer and weaker, which matches the window mean rising only from 0.35 to about 0.47 by 2024.',
      evidence:
        'Within the selected recovery window, **{{analysed}}** and **{{shares}}**; the map and counts use the same valid-date mask.',
      caveat:
        'Post-fire greening is an observed spectral trajectory, not proof of ecological recovery or attribution to the fire alone.',
      options: {
        layer: 'significance',
        significanceSource: 'mann-kendall',
        dateWindow: [9, 15],
        minimumValid: 5,
        splitSlice: 12
      },
      controls: ['dateWindow', 'minimumValid'],
      readouts: ['shares', 'analysed'],
      highlight: {readout: 'shares'}
    },
    {
      id: 'caveats',
      headline: 'Sparse dates limit trend attribution',
      textAlternative:
        'Trend colors reflect available clear observations and cannot isolate fire recovery from seasonal effects.',
      title: 'What these tests can and cannot say',
      body: '**Limits:** 16 dates are few for a trend test. Dates are treated as ordinal, not as time. Summer NDVI also varies with rainfall, phenology and sun angle. Nearby pixels are not independent, so the p-values are optimistic, and **alpha is not corrected for the number of cells**: at 0.05 about one pixel in twenty passes by chance (see the **Multiple testing** readout). Clouds are masked, so a masked date is simply missing.\n\n**Try:** **Significance level (alpha)**, lowered from 0.05 to see the chance passes disappear, the split at 2022-06-23 (with **Significance from** on the Welch t-test), **Dates used** 2017 to 2021-07-13 (pre-fire noise only: the trend map should be mostly grey), or **Minimum valid dates per pixel** at 14.',
      evidence:
        '**{{falsePositives}}**. The current mask retains **{{analysed}}** under the chosen date window and minimum-valid rule.',
      caveat:
        'Lowering alpha changes a decision threshold; it does not correct multiplicity, restore missing acquisitions or make nearby pixels independent.',
      options: {
        layer: 'significance',
        significanceSource: 'mann-kendall',
        dateWindow: [0, 15],
        alpha: 0.05,
        minimumValid: 8
      },
      controls: ['alpha', 'significanceSource', 'splitSlice', 'dateWindow', 'minimumValid'],
      readouts: ['falsePositives', 'analysed', 'acquisitionContract'],
      highlight: {readout: 'falsePositives'}
    }
  ]
});
