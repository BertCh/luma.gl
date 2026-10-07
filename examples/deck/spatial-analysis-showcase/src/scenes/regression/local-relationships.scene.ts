// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {LocalRelationshipsOptions} from './local-relationships.compute';

const MAP_LABELS: Record<LocalRelationshipsOptions['map'], string> = {
  income: 'Local effect of ln median household income',
  density: 'Local effect of ln population density',
  age65: 'Local effect of age 65+ share',
  education: 'Local effect of no-diploma share',
  intercept: 'Local intercept',
  localR2: 'Local R²',
  condition: 'Local condition number',
  influence: 'Hat-matrix diagonal',
  residuals: 'GWR residual',
  olsResiduals: 'OLS (national) residual'
};

/** Geographically weighted regression of county diabetes prevalence. */
export default defineScene<LocalRelationshipsOptions>({
  id: 'local-relationships',
  title: 'Does the income-health link change across America?',
  chapter: 'regression',
  order: 2,
  summary:
    'One national regression says income and diabetes are linked. Geographically weighted regression fits a separate model around every one of 3,100 counties and shows where, and how strongly, that link changes. A Monte Carlo test says which variation is more than chance.',
  contributors: [
    'GPUGeographicallyWeightedRegression',
    'GPUGeographicallyWeightedRegressionNonstationarityTest',
    'GPUOrdinaryLeastSquares'
  ],
  datasets: [
    {id: 'us-counties', role: 'county polygons with CDC PLACES diabetes, income and SVI'},
    {id: 'us-states', role: 'state borders'}
  ],
  initialView: {longitude: -96.3, latitude: 38.4, zoom: 3.55},

  options: [
    {
      kind: 'slider',
      id: 'bandwidth',
      label: 'Bandwidth (ladder step)',
      group: 'Bandwidth',
      apply: 'param',
      min: 0,
      max: 8,
      step: 1,
      default: 0,
      format: value => (value <= 0 ? 'auto (lowest AICc)' : `ladder step ${value} of 8`),
      help: 'Auto searches the 8-step ladder and keeps the bandwidth with the lowest AICc. A step fixes it. Adaptive ladder: 24, 36, 48, 64, 80, 96, 112, 128 nearest counties. Fixed ladder: 150, 250, 350, 500, 700, 1000, 1400, 2000 km. A small bandwidth is local but noisy; a large one approaches the national model.'
    },
    {
      kind: 'select',
      id: 'mode',
      label: 'Bandwidth type',
      group: 'Bandwidth',
      apply: 'param',
      default: 'adaptive',
      help: 'Adaptive uses the k nearest counties, so the area stretches over sparse Plains counties and shrinks around dense Appalachia. Fixed uses one distance everywhere.',
      options: [
        {value: 'adaptive', label: 'Adaptive: k nearest counties'},
        {value: 'fixed', label: 'Fixed: distance in km'}
      ]
    },
    {
      kind: 'select',
      id: 'kernel',
      label: 'Kernel',
      group: 'Bandwidth',
      apply: 'param',
      default: 'bisquare',
      help: 'How weights fall with distance. Bisquare reaches exactly zero at the bandwidth (compact support); Gaussian never does.',
      options: [
        {value: 'bisquare', label: 'Bisquare'},
        {value: 'gaussian', label: 'Gaussian'}
      ]
    },
    {
      kind: 'select',
      id: 'subset',
      label: 'Counties included (mask)',
      group: 'Sample',
      apply: 'param',
      default: 'all',
      help: 'A row mask: excluded counties neither get a local fit nor serve as neighbours. Metro is rural-urban continuum codes 1 to 3; South is the Census South region.',
      options: [
        {value: 'all', label: 'All counties'},
        {value: 'metro', label: 'Metropolitan counties only'},
        {value: 'nonmetro', label: 'Non-metropolitan counties only'},
        {value: 'south', label: 'Census South region only'}
      ]
    },
    {
      kind: 'toggle',
      id: 'spatialIndex',
      label: 'Grid index for fixed bisquare',
      group: 'Sample',
      apply: 'compile',
      default: false,
      help: 'A compile-time option (indexGridSize), needs a device with at least 10 storage buffers per shader stage; the readout says whether it is active. Bisquare with a fixed bandwidth has bounded support, so each county visits only grid cells within reach instead of scanning every row. Gaussian and adaptive bandwidths ignore it.'
    },
    {
      kind: 'slider',
      id: 'permutations',
      label: 'Monte Carlo permutations',
      group: 'Non-stationarity test',
      apply: 'param',
      min: 9,
      max: 99,
      step: 10,
      default: 19,
      help: 'How many random relabellings of the counties the test refits. More gives finer p-values at proportional cost (each permutation refits every county).'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Permutation seed',
      group: 'Non-stationarity test',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 1,
      help: 'The same seed always gives the same permutations and p-values.'
    },
    {
      kind: 'button',
      id: 'runTest',
      label: 'Run the Monte Carlo test',
      group: 'Non-stationarity test',
      help: 'Shuffles which county holds which observation, refits at the selected bandwidth and counts how often the shuffled coefficient surface varies as much as the real one.'
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Map',
      group: 'Display',
      apply: 'param',
      default: 'income',
      help: 'Which local result the fill shows. Coefficients are per standard deviation of the predictor, in percentage points of diabetes prevalence.',
      options: [
        {value: 'income', label: 'Local effect of income'},
        {value: 'density', label: 'Local effect of population density'},
        {value: 'age65', label: 'Local effect of age 65+ share'},
        {value: 'education', label: 'Local effect of no-diploma share'},
        {value: 'intercept', label: 'Local intercept'},
        {value: 'localR2', label: 'Local R²'},
        {value: 'condition', label: 'Local condition number (collinearity)'},
        {value: 'influence', label: 'Hat-matrix diagonal (influence)'},
        {value: 'residuals', label: 'GWR residual'},
        {value: 'olsResiduals', label: 'National OLS residual'}
      ]
    },
    {
      kind: 'select',
      id: 'center',
      label: 'Center the colors on',
      group: 'Display',
      apply: 'param',
      default: 'zero',
      help: 'Zero shows where the effect is positive or negative. The national coefficient shows where the local effect is stronger or weaker than the national average. Applies to the coefficient maps.',
      options: [
        {value: 'zero', label: 'Zero effect'},
        {value: 'global', label: 'National OLS coefficient'}
      ]
    },
    {
      kind: 'toggle',
      id: 'borders',
      label: 'State borders',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws state outlines over the county fill.'
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Counties in the model',
      help: 'Counties with every variable present (contiguous US plus DC).'
    },
    {
      id: 'bandwidth',
      label: 'Selected bandwidth',
      help: 'The ladder value with the lowest AICc, or the one you fixed.'
    },
    {
      id: 'scores',
      label: 'AICc per ladder step',
      help: 'Corrected Akaike criterion of each candidate bandwidth, smallest first step first. Lower is better.'
    },
    {
      id: 'aicc',
      label: 'AICc: GWR vs national OLS',
      help: 'Same formula for both, with the GWR hat trace as its effective parameter count.'
    },
    {id: 'rSquared', label: 'R²'},
    {
      id: 'trace',
      label: 'Effective parameters (hat trace)',
      help: 'Trace of the hat matrix: how many parameters the local fits use, in total. The national model uses five.'
    },
    {
      id: 'spread',
      label: 'Local coefficient 5th to 95th percentile',
      help: 'For the coefficient on the map, with the national OLS value.'
    },
    {id: 'globalCoefficients', label: 'National OLS coefficients'},
    {
      id: 'condition',
      label: 'Local condition number',
      help: 'mgwr local_collinearity: values above 30 flag nearly collinear local designs, whose coefficients cannot be trusted.'
    },
    {id: 'singular', label: 'Local fits'},
    {
      id: 'index',
      label: 'Neighbour search',
      help: 'Whether the fit kernel scans every county or uses the grid index.'
    },
    {
      id: 'test',
      label: 'Monte Carlo test',
      help: 'Is the spatial variation of each coefficient real? Pseudo p-value (g + 1) / (P + 1).'
    },
    {id: 'test0', label: 'Intercept'},
    {id: 'test1', label: 'ln income'},
    {id: 'test2', label: 'ln density'},
    {id: 'test3', label: 'Age 65+'},
    {id: 'test4', label: 'No diploma'}
  ],

  legends: state => {
    const entries: LegendSpec[] = [];
    if (state.map === 'localR2') {
      entries.push({
        kind: 'ramp',
        title: MAP_LABELS.localR2,
        ramp: 'viridis',
        extent: [0, 1],
        labels: ['0: no fit', '1: perfect fit']
      });
    } else if (state.map === 'condition') {
      entries.push({
        kind: 'ramp',
        title: MAP_LABELS.condition,
        ramp: 'magma',
        extent: [1, 30],
        labels: ['1', '30 or more: unreliable']
      });
    } else if (state.map === 'influence') {
      entries.push({
        kind: 'ramp',
        title: MAP_LABELS.influence,
        ramp: 'inferno',
        extent: [0, 1],
        labels: ['low', 'high (98th percentile)']
      });
    } else if (state.map === 'residuals' || state.map === 'olsResiduals') {
      entries.push({
        kind: 'ramp',
        id: 'residual',
        title: `${MAP_LABELS[state.map]} (observed minus fitted)`,
        ramp: 'diverging',
        extent: 'gpu',
        unit: 'percentage points',
        labels: ['over-predicted', 'under-predicted'],
        format: value => value.toFixed(1)
      });
    } else {
      entries.push({
        kind: 'ramp',
        id: 'coefficient',
        title: MAP_LABELS[state.map],
        ramp: 'diverging',
        extent: 'gpu',
        unit: 'points of prevalence per standard deviation',
        labels: [
          state.center === 'zero' ? 'more negative' : 'weaker than national',
          state.center === 'zero' ? 'more positive' : 'stronger than national'
        ],
        format: value => value.toFixed(2)
      });
    }
    entries.push({
      kind: 'categories',
      title: 'Counties',
      entries: [{color: [182, 184, 190, 170], label: 'No local fit (missing data or masked out)'}]
    });
    return entries;
  },

  snippet: state => `import {
  GPUGeographicallyWeightedRegression, GPUGeographicallyWeightedRegressionNonstationarityTest,
  getGPUGeographicallyWeightedRegressionParameterValues, getGPUPermutationParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions are planar meters (Albers equal-area), predictors are z-scores, row-major
graph.add(new GPUGeographicallyWeightedRegression({
  positions, predictors, predictorCount: 4, response, mask,
  parameters,                          // kernel, mode, bandwidth ladder: a per-frame buffer
  maximumBandwidthCount: 8, maximumNeighborCount: 128,${state.spatialIndex ? '' : '\n  indexGridSize: false,              // scan every row'}
  output: {coefficients, localR2, residuals, hatDiagonal, localStatus,
           localConditionNumber, bandwidthScores, selectedBandwidth, summary}
}));

// Per-frame changes are writes: no rebuild
parameters.write(getGPUGeographicallyWeightedRegressionParameterValues({
  kernel: '${state.kernel}', bandwidthMode: '${state.mode}',
  bandwidths: ${state.bandwidth <= 0 ? (state.mode === 'adaptive' ? '[24, 36, 48, 64, 80, 96, 112, 128]' : '[150e3, 250e3, 350e3, 500e3, 700e3, 1e6, 1.4e6, 2e6]') : '[/* one ladder value */]'}
}, 8));

// Is the spatial variation real? Permute, refit at the selected bandwidth, count exceedances.
testGraph.add(new GPUGeographicallyWeightedRegressionNonstationarityTest({
  positions, predictors, predictorCount: 4, response, mask,
  bandwidthParameters: parameters, selectedBandwidth, coefficients,
  parameters: permutationParameters, maximumPermutations: 99,
  output: {table, summary}
}));
permutationParameters.write(getGPUPermutationParameterValues({seed: ${state.seed}, permutations: ${state.permutations}}));`,

  about: {
    what: '`GPUGeographicallyWeightedRegression` fits a weighted least-squares model around every county, giving each its own coefficients. Weights fall with distance (a bisquare or Gaussian kernel) and the bandwidth is chosen automatically by AICc, as in mgwr and ArcGIS GWR. `GPUOrdinaryLeastSquares` provides the national baseline and `GPUGeographicallyWeightedRegressionNonstationarityTest` checks whether the variation could be chance.',
    why: 'A single coefficient assumes one relationship everywhere. If the link between income and diabetes is steeper in the Black Belt than in New England, a national average misleads policy in both places.',
    howToRead:
      'Colors show a local coefficient: for the income map, blue means that higher income goes with lower diabetes. Compare with the national OLS value (center the colors on it). The local R² map shows where the four variables explain diabetes well and where they do not. High condition numbers mean the local fit is unreliable.'
  },

  create: async ctx =>
    (await import('./local-relationships.compute')).createLocalRelationships(ctx),

  story: [
    {
      id: 'question',
      title: 'Is “richer means healthier” true everywhere?',
      body: 'Across 3,100 US counties, age-adjusted **diabetes prevalence** falls as **median household income** rises (CDC PLACES and Census SAIPE data). That is one average slope for a whole country. But the Mississippi Delta, Appalachia, the Southwest and the Northeast have very different histories, so **where does the link bend**?\n\nThe map shows the finished answer for income: one local coefficient per county from **`GPUGeographicallyWeightedRegression`**. Blue counties are where more income goes with less diabetes (**Map** below picks what is shown); the story below explains how it is made.',
      options: {map: 'income'},
      camera: {longitude: -96.3, latitude: 38.4, zoom: 3.55},
      controls: ['map']
    },
    {
      id: 'national',
      title: 'First, one line for the whole country',
      body: '**`GPUOrdinaryLeastSquares`** fits `diabetes = β₀ + β₁·ln income + β₂·ln density + β₃·age 65+ + β₄·no diploma` once, with all predictors standardized, so each coefficient is points of prevalence per standard deviation. The readout **National OLS coefficients** lists them.\n\nThe map shows the national model’s **residuals**: observed minus fitted. They are clearly not random: whole regions are red (more diabetes than the model expects), notably the Deep South, and others blue. Residuals with a spatial pattern are the sign that coefficients may not be constant.',
      options: {map: 'olsResiduals'},
      controls: ['map'],
      readouts: ['globalCoefficients']
    },
    {
      id: 'gwr',
      title: 'A separate regression around every county',
      body: 'Geographically weighted regression fits a weighted least-squares model **at each county**, giving nearby counties large weights and distant ones little (`w = K(d / h)`). The kernel width **h** is the bandwidth. Here **Bandwidth type** is *adaptive*: the 24 to 128 nearest counties, chosen by lowest AICc. This is the model behind mgwr and ArcGIS “Geographically Weighted Regression”.\n\nThe map is the local coefficient of **ln income**: the center of the color scale is zero. Read **Selected bandwidth** and **AICc: GWR vs national OLS** in the readouts: a lower AICc means local fits improve on the national model even after paying for their extra effective parameters (the **hat trace**).',
      options: {map: 'income', center: 'zero'},
      controls: ['map', 'mode'],
      readouts: ['bandwidth', 'aicc', 'trace']
    },
    {
      id: 'center',
      title: 'Stronger or weaker than the national average?',
      body: '**Center the colors on** is now *National OLS coefficient* (set it back to zero to compare). White now means “the same as the national slope”. Blue counties have a stronger link between income and lower diabetes than the nation, orange a weaker or even reversed one. Check the **Local coefficient 5th to 95th percentile** readout for the range.',
      options: {center: 'global'},
      controls: ['center', 'map'],
      readouts: ['spread']
    },
    {
      id: 'bandwidth',
      title: 'The bandwidth is the whole trade',
      body: 'Slide **Bandwidth (ladder step)** from auto to *ladder step 1* (the 24 nearest counties). Local fits become more local and noisier: patches break up. Step 8 (128 counties) approaches a regional average. Auto picks the step with the lowest **AICc**; the **AICc per ladder step** readout lists all eight.\n\nChanging the bandwidth only writes the 8-value ladder into a parameter buffer, so the compiled graph is simply encoded again. **Kernel** (bisquare or Gaussian) and **Bandwidth type** (adaptive or fixed distance) are parameters too.',
      options: {center: 'global', bandwidth: 1},
      controls: ['bandwidth', 'mode', 'kernel'],
      readouts: ['bandwidth', 'scores']
    },
    {
      id: 'reliability',
      title: 'Where is the local model reliable?',
      body: 'Not every local fit deserves trust. Set **Map** to *Local R²*, *Local condition number (collinearity)* or *Hat-matrix diagonal (influence)*. The **local R²** map shows where the four variables explain diabetes well. The **local condition number** (mgwr `local_collinearity`) flags neighbourhoods where income, density, age and education are nearly collinear: above 30, local coefficients become unstable. The **hat-matrix diagonal** shows how much each county’s own value drives its fit.\n\nSwitch between these maps and compare them with the coefficient pattern before reading a coefficient as a finding.',
      options: {bandwidth: 0, map: 'localR2'},
      controls: ['map'],
      readouts: ['rSquared', 'condition']
    },
    {
      id: 'test',
      title: 'Is the variation real, or just noise?',
      body: 'Local coefficients always vary, even for data with no spatial structure. **`GPUGeographicallyWeightedRegressionNonstationarityTest`** measures how much each coefficient surface varies (its standard deviation across counties), then shuffles which county holds which observation, refits all counties at the selected bandwidth and counts the shuffles that vary as much. Press **Run the Monte Carlo test** and read the pseudo p-value of each coefficient: small means more variation than chance.\n\nTry the **Counties included (mask)** (metro only, South only), *Fixed* **Bandwidth type** with a *Gaussian* **Kernel**, more **Monte Carlo permutations** and other **Permutation seed** values. **Limits:** counties are large and heterogeneous; PLACES values are modelled estimates; and a GWR with several correlated predictors can show spurious local patterns, so check the condition numbers. mgwr’s multiscale (MGWR) variant is not included.',
      options: {map: 'income', center: 'global'},
      controls: ['runTest', 'permutations', 'subset', 'mode', 'kernel'],
      readouts: ['test', 'test1', 'test2']
    }
  ]
});
