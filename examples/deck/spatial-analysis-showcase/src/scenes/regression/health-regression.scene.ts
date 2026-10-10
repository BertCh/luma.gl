// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {HealthRegressionOptions} from './health-regression.compute';

const CLUSTER_RGBA = {
  hh: [214, 69, 65, 235],
  lh: [138, 176, 232, 235],
  ll: [49, 104, 190, 235],
  hl: [240, 160, 60, 235],
  none: [150, 150, 158, 120]
} as const;

const OUTCOME_UNITS: Record<HealthRegressionOptions['outcome'], string> = {
  diabetes: '% of adults with diabetes',
  obesity: '% of adults with obesity',
  asthma: '% of adults with asthma',
  depression: '% of adults with depression',
  heartDisease: '% of adults with heart disease',
  smoking: '% of adults who smoke'
};

/** Health regression on Chicago tracts: OLS, diagnostics, spatial lag and spatial error models. */
export default defineScene<HealthRegressionOptions>({
  id: 'health-regression',
  title: 'Chicago diabetes residuals across tract regression models',
  chapter: 'regression',
  order: 1,
  summary:
    'OLS, spatial diagnostics, lag and error models use CDC PLACES and ACS data for Chicago tracts to map prevalence residuals and local clusters; the ecological, model-based inputs do not support causal or individual inference.',
  contributors: [
    'GPUOrdinaryLeastSquares',
    'GPUSpatialRegressionDiagnostics',
    'GPUSpatialTwoStageLeastSquares',
    'GPUSpatialErrorGM',
    'addSpatialRegressionRecipe'
  ],
  datasets: [
    {id: 'chicago-tracts', role: 'tract polygons with CDC PLACES outcomes, ACS and SVI covariates'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Chicago diabetes residuals',
      subtitle: 'Tract models using CDC PLACES and ACS covariates'
    },
    scaleBar: {units: 'metric'},
    credit: 'CDC PLACES; US Census Bureau ACS',
    caveat: 'Ecological associations from model-based tract estimates; not causal effects.'
  },

  options: [
    {
      kind: 'select',
      id: 'outcome',
      label: 'Health outcome (response)',
      group: 'Model',
      apply: 'param',
      default: 'diabetes',
      help: 'CDC PLACES model-based prevalence per tract (percent of adults). Changing it rewrites the response buffer; nothing is recompiled.',
      options: [
        {value: 'diabetes', label: 'Diabetes'},
        {value: 'obesity', label: 'Obesity'},
        {value: 'asthma', label: 'Current asthma'},
        {value: 'depression', label: 'Depression'},
        {value: 'heartDisease', label: 'Coronary heart disease'},
        {value: 'smoking', label: 'Current smoking'}
      ]
    },
    {
      kind: 'select',
      id: 'model',
      label: 'Predictors',
      group: 'Model',
      apply: 'compile',
      default: 'economic',
      help: 'Which covariates enter the model (all standardized to z-scores, so a coefficient is points of prevalence per one standard deviation). The predictor count is a compile-time property of every contributor, so each set compiles its graphs the first time you choose it.',
      options: [
        {value: 'income', label: 'Income only (1)', help: 'ln income per capita.'},
        {
          value: 'economic',
          label: 'Economic (4)',
          help: 'Income, poverty, uninsured and unemployment rates.'
        },
        {
          value: 'full',
          label: 'Full (8)',
          help: 'Adds age 65+, disability, no-vehicle households, Black and Hispanic shares. Predictors overlap heavily, so coefficients become unstable: try the ridge penalty.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'ridge',
      label: 'Ridge penalty λ (log₁₀)',
      group: 'Model',
      apply: 'param',
      min: -2,
      max: 3,
      step: 0.1,
      default: -2,
      format: value => (value <= -2 ? 'off' : (10 ** value).toPrecision(2)),
      help: 'Adds λ to the diagonal of XᵀX (intercept unpenalized). Coefficients shrink toward zero and standard errors tighten; watch the full model stabilize. A per-frame parameter write.'
    },
    {
      kind: 'select',
      id: 'weights',
      label: 'Spatial weights',
      group: 'Neighbours',
      apply: 'compile',
      default: 'queen',
      help: 'Who counts as a neighbour in the diagnostics, the residual map and the lag and error models. Weights are always row-standardized. Contiguity is symmetric; k nearest neighbours is directed.',
      options: [
        {value: 'queen', label: 'Queen contiguity', help: 'Tracts sharing a boundary point.'},
        {value: 'rook', label: 'Rook contiguity', help: 'Tracts sharing a boundary edge.'},
        {value: 'knn4', label: '4 nearest centroids'},
        {value: 'knn8', label: '8 nearest centroids'},
        {value: 'knn12', label: '12 nearest centroids'}
      ]
    },
    {
      kind: 'select',
      id: 'instruments',
      label: 'Lag-model instruments',
      group: 'Neighbours',
      apply: 'compile',
      default: '1',
      help: 'Instruments of the spatially lagged response W·y in two-stage least squares (spreg w_lags): [1, X, WX], or also W²X.',
      options: [
        {value: '1', label: 'X and WX (w_lags = 1)'},
        {value: '2', label: 'X, WX and W²X (w_lags = 2)'}
      ]
    },
    {
      kind: 'slider',
      id: 'significance',
      label: 'Residual cluster significance',
      group: 'Diagnostics',
      apply: 'param',
      min: 0.001,
      max: 0.2,
      step: 0.001,
      default: 0.05,
      format: value => `p ≤ ${value.toFixed(3)}`,
      help: 'Level of the analytic local Moran test on the OLS residuals (used by the residual clusters map).'
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Map',
      group: 'Display',
      apply: 'param',
      default: 'residuals',
      help: 'What the tract fill shows. Switching only changes which result buffer the fill layer binds.',
      options: [
        {value: 'residuals', label: 'OLS residuals'},
        {value: 'fitted', label: 'OLS fitted value'},
        {value: 'observed', label: 'Observed value'},
        {value: 'clusters', label: 'OLS residual clusters (local Moran)'},
        {value: 'lag', label: 'Spatial lag model residuals'},
        {value: 'error', label: 'Spatial error model residuals'}
      ]
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Tracts in the model',
      help: 'Tracts with every variable present, population of at least 300, in the largest contiguous group (the O’Hare tract is an island).'
    },
    {id: 'links', label: 'Weight links'},
    {
      id: 'fit',
      label: 'OLS R² and adjusted R²',
      help: 'R² and adjusted R² of the ordinary least squares fit.'
    },
    {
      id: 'criteria',
      label: 'Information criteria',
      help: 'AIC and BIC, GeoDa convention. Lower is better; compare between models of the same outcome.'
    },
    {
      id: 'normality',
      label: 'Jarque-Bera (normal residuals)',
      help: 'Small p: residuals are not normally distributed.'
    },
    {
      id: 'heteroskedasticity',
      label: 'Breusch-Pagan (constant variance)',
      help: 'Small p: residual variance changes with the predictors.'
    },
    {
      id: 'coefficient0',
      label: 'β₀',
      help: 'Coefficient (t statistic, standard error) from OLS, then the same coefficient in the lag and error models.'
    },
    {id: 'coefficient1', label: 'β₁'},
    {id: 'coefficient2', label: 'β₂'},
    {id: 'coefficient3', label: 'β₃'},
    {id: 'coefficient4', label: 'β₄'},
    {id: 'coefficient5', label: 'β₅'},
    {id: 'coefficient6', label: 'β₆'},
    {id: 'coefficient7', label: 'β₇'},
    {id: 'coefficient8', label: 'β₈'},
    {
      id: 'moran',
      label: 'Moran’s I of the residuals',
      help: 'Spatial autocorrelation of OLS residuals with the chosen weights; positive means similar residuals sit next to each other.'
    },
    {
      id: 'lmLag',
      label: 'LM-lag',
      help: 'Lagrange multiplier test for a spatially lagged response (statistic, p).'
    },
    {
      id: 'lmError',
      label: 'LM-error',
      help: 'Lagrange multiplier test for spatially correlated errors.'
    },
    {
      id: 'robustLag',
      label: 'Robust LM-lag',
      help: 'LM-lag test robust to a possible spatial error process.'
    },
    {
      id: 'robustError',
      label: 'Robust LM-error',
      help: 'LM-error test robust to a possible spatial lag.'
    },
    {id: 'sarma', label: 'LM-SARMA', help: 'Joint test for lag and error.'},
    {
      id: 'verdict',
      label: 'Which spatial model?',
      help: 'The classical decision rule on the LM tests at 5 percent (Anselin 2005).'
    },
    {
      id: 'rho',
      label: 'Lag model ρ',
      help: 'Spatial lag coefficient from two-stage least squares: how much neighbours’ prevalence spills into a tract after controlling for X.'
    },
    {id: 'lagFit', label: 'Lag model fit'},
    {
      id: 'lambda',
      label: 'Error model λ',
      help: 'Spatial error coefficient from generalized moments (Kelejian-Prucha).'
    },
    {
      id: 'clusters',
      label: 'Residual clusters',
      help: 'Significant local Moran quadrants of the OLS residuals.'
    },
    {
      id: 'status',
      label: 'Status (OLS / diagnostics / lag / error)',
      help: '0 means every fit succeeded.'
    }
  ],

  legends: state => {
    const unit = OUTCOME_UNITS[state.outcome];
    const entries: LegendSpec[] = [];
    if (state.map === 'fitted' || state.map === 'observed') {
      entries.push({
        kind: 'ramp',
        id: 'value',
        title: state.map === 'fitted' ? 'Fitted by OLS' : 'Observed',
        ramp: 'ylorrd',
        extent: 'gpu',
        unit,
        format: value => value.toFixed(1)
      });
    } else if (state.map === 'clusters') {
      entries.push({
        kind: 'categories',
        title: 'Cluster of OLS residuals',
        entries: [
          {color: CLUSTER_RGBA.hh, label: 'High-High: model under-predicts'},
          {color: CLUSTER_RGBA.ll, label: 'Low-Low: model over-predicts'},
          {color: CLUSTER_RGBA.lh, label: 'Low-High outlier'},
          {color: CLUSTER_RGBA.hl, label: 'High-Low outlier'},
          {color: CLUSTER_RGBA.none, label: 'Not significant'}
        ],
        note: 'Analytic local Moran of the residuals at the chosen level. Clusters are what a lag or error term should absorb.'
      });
    } else {
      const names = {
        residuals: 'OLS residual',
        lag: 'Lag model residual',
        error: 'Error model residual'
      };
      entries.push({
        kind: 'ramp',
        id: 'residual',
        title: `${names[state.map]} (observed minus fitted)`,
        ramp: 'diverging',
        midpoint: 0,
        extent: 'gpu',
        unit: 'percentage points',
        labels: ['over-predicted', 'under-predicted'],
        format: value => value.toFixed(1)
      });
    }
    entries.push({
      kind: 'categories',
      title: 'Tracts',
      entries: [
        {color: [170, 172, 178, 150], label: 'Left out of the model (missing data or island)'}
      ]
    });
    return entries;
  },

  snippet: state => {
    const knn = state.weights.startsWith('knn');
    return `import {
  addSpatialRegressionRecipe, GPUContiguityWeights, GPUSpatialWeightsTransform,
  GPUSpatialTwoStageLeastSquares, GPUSpatialErrorGM
} from '@luma.gl/experimental/gpu-spatial-analysis';

// 1. Weights ${knn ? `(directed k nearest neighbours, row-standardized)` : `(${state.weights} contiguity from shared vertices, then row-standardized)`}
${
  knn
    ? `graph.add(new GPUNeighborSearch({mode: 'knn', k: ${state.weights.slice(3)}, gridSize: [64, 64], positions, parameters, weights, overflow}));`
    : `graph.add(new GPUContiguityWeights({criterion: '${state.weights}', positions: vertices, ringOffsets, polygonOffsets, weights, overflow}));
graph.add(new GPUSpatialWeightsTransform({operation: 'row', weights}));`
}

// 2. OLS + LM diagnostics + residual local Moran, one recipe
const regression = addSpatialRegressionRecipe(graph, {
  predictors, response, predictorCount: ${state.model === 'income' ? 1 : state.model === 'economic' ? 4 : 8},
  weights, parameters: moranParameters,
  olsParameters: ridgeParameters,      // [lambda]: a per-frame buffer write
  ols: {coefficients, standardErrors, tStatistics, summary, status, residuals, fitted},
  diagnostics: {tests, summary, status},
  residualMoran: {zScores, localI, quadrants, pValues}
});

// 3. The two alternatives the diagnostics choose between
lagGraph.add(new GPUSpatialTwoStageLeastSquares({
  weights, predictors, response, predictorCount, instrumentOrder: ${state.instruments}, output: lag
}));
errorGraph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount, output: error}));

// 4. Per frame only on change: ridgeParameters.write(getGPUOrdinaryLeastSquaresParameterValues(${state.ridge <= -2 ? 0 : (10 ** state.ridge).toPrecision(2)}))`;
  },

  about: {
    what: '`GPUOrdinaryLeastSquares` fits the regression and reports R², AIC/BIC, Jarque-Bera and Breusch-Pagan. `GPUSpatialRegressionDiagnostics` tests its residuals for spatial dependence (LM-lag, LM-error, robust variants, SARMA, Moran’s I). `GPUSpatialTwoStageLeastSquares` fits the spatial lag model and `GPUSpatialErrorGM` the spatial error model. `addSpatialRegressionRecipe` wires OLS, diagnostics and a residual local Moran in one graph.',
    why: 'Neighbouring tracts share air, food environments, clinics and history. If the residuals of an ordinary regression are spatially clustered, the standard errors are too small and coefficients can be biased. The diagnostics tell you which spatial model to fit.',
    howToRead:
      'On the residual maps blue tracts have less diabetes than the model predicts and red tracts more. A model that has captured the spatial structure leaves residuals with no clusters (Moran’s I near its expected value). Compare the coefficients on the three fits: a predictor whose effect collapses once the spatial term is added was partly standing in for place.'
  },

  create: async ctx => (await import('./health-regression.compute')).createHealthRegression(ctx),

  story: [
    {
      id: 'question',
      title: 'What do incomes explain about diabetes in Chicago?',
      headline: 'Observed diabetes prevalence varies across Chicago tracts',
      textAlternative:
        'Chicago tracts are shaded by estimated diabetes prevalence, with higher values concentrated on the South and West sides.',
      body: 'Diabetes is not spread evenly across Chicago: **CDC PLACES** estimates its prevalence for every census tract, and the highest rates sit on the South and West sides. Part of that is income. The question for a planner is sharper: **how much does a tract’s income and poverty explain, and where does the model get it wrong?**\n\nThe map shows the observed prevalence (**Map** below) for the tracts in the model; pick another **Health outcome (response)** to see its geography. Tracts left out (missing data, or O’Hare, which touches no other tract) are gray.',
      options: {model: 'income', map: 'observed'},
      controls: ['outcome', 'map'],
      readouts: ['rows']
    },
    {
      id: 'ols',
      title: 'One predictor, one line: ordinary least squares',
      headline: 'Income-only residuals remain spatially patterned',
      textAlternative:
        'A diverging tract map shows adjacent areas with similarly positive or negative income-only OLS residuals.',
      body: '**`GPUOrdinaryLeastSquares`** fits `prevalence = β₀ + β₁·ln(income per capita) + ε` by least squares on the GPU, and reports the fit in the readouts: R², AIC and BIC, and the **Jarque-Bera** and **Breusch-Pagan** tests of the residuals. Predictors are standardized, so β₁ is points of prevalence per standard deviation of ln income.\n\nThe map now shows the **residual**: observed minus fitted. Blue tracts have less diabetes than their income predicts, red tracts more. Hover a tract for its numbers.',
      options: {map: 'residuals'},
      controls: ['map'],
      readouts: ['fit', 'normality', 'heteroskedasticity']
    },
    {
      id: 'predictors',
      title: 'Adding poverty, insurance and employment',
      headline: 'Economic predictors improve tract-level fit',
      textAlternative:
        'Chicago tracts remain shaded by model residual while fit statistics compare the expanded economic model with the income-only model.',
      body: '**Predictors** is now *Economic (4)*. Poverty, uninsured and unemployment rates add explanatory power and the coefficients are now *partial* effects: the change in prevalence per standard deviation with the others held fixed. Read the t statistics in the **β** readouts, and compare adjusted R² and AIC with the single-predictor fit.\n\nThe predictor count is compile-time in every contributor, so this switch compiles its graphs once (the rebuild badge in **Under the hood**).',
      options: {model: 'economic'},
      controls: ['model'],
      readouts: ['fit', 'criteria', 'coefficient1']
    },
    {
      id: 'ridge',
      title: 'When predictors overlap: ridge',
      headline: 'Ridge shrinkage reduces coefficient instability',
      textAlternative:
        'The residual map accompanies coefficient readouts that contract as the ridge penalty increases for correlated tract predictors.',
      body: '**Predictors** is now *Full (8)*, which adds age, disability, vehicle access and race/ethnicity shares. Tract covariates are highly correlated, so individual coefficients become unstable and their standard errors inflate. Slide the **Ridge penalty λ (log₁₀)**: `GPUOrdinaryLeastSquares` adds λ to the diagonal of XᵀX, which shrinks the coefficients and tightens the standard errors at the price of a little bias. The penalty is a one-float parameter buffer, so the slider is a buffer write and no graph is rebuilt.',
      options: {model: 'full', ridge: 0.5},
      controls: ['model', 'ridge'],
      readouts: ['coefficient1', 'coefficient2', 'coefficient3']
    },
    {
      id: 'diagnostics',
      title: 'Is the leftover spatial?',
      headline: 'OLS residual clusters remain statistically significant',
      textAlternative:
        'Red and blue tract groups mark significant high-high and low-low clusters of economic-model residuals.',
      body: 'Ordinary least squares assumes independent errors. **`GPUSpatialRegressionDiagnostics`** tests that against the spatial weights: **LM-lag** (does a neighbour’s outcome matter?), **LM-error** (are the errors correlated?), their **robust** versions and **Moran’s I** of the residuals. The matching spreg tests are `LMtests` and `MoranRes`.\n\n**Map** now shows *OLS residual clusters (local Moran)*: significant **clusters** of OLS residuals from a local Moran run inside the same graph by **`addSpatialRegressionRecipe`**: red clusters are places the model under-predicts, blue clusters where it over-predicts. If the model had absorbed the geography there would be few of them.',
      options: {model: 'economic', ridge: -2, map: 'clusters'},
      controls: ['weights', 'significance', 'map'],
      readouts: ['moran', 'lmLag', 'lmError']
    },
    {
      id: 'lag',
      title: 'Spillover: the spatial lag model',
      headline: 'Spatial lag absorbs part of neighboring dependence',
      textAlternative:
        'A diverging map shows spatial-lag residuals while readouts report the neighbor coefficient and model fit.',
      body: 'If neighbours’ outcomes influence a tract’s own (shared food environments, clinics, behaviour), the right model is `y = ρ·Wy + Xβ + ε`. **`GPUSpatialTwoStageLeastSquares`** estimates ρ with `WX` (and optionally `W²X`) as instruments, the same estimator as spreg `GM_Lag`. The **Anselin-Kelejian** test on its residuals checks that dependence is gone.\n\nThe map shows the lag model’s residuals. Compare each **β** line: `OLS to lag, error`. Try **Lag-model instruments** with W²X; it recompiles the lag graph only.',
      options: {map: 'lag'},
      controls: ['instruments', 'map'],
      readouts: ['rho', 'lagFit', 'coefficient1']
    },
    {
      id: 'error',
      title: 'Or unmeasured shared causes: the spatial error model',
      headline: 'Spatial error captures correlated omitted structure',
      textAlternative:
        'Spatial-error residuals are mapped by tract beside lag and error coefficients used to compare model specifications.',
      body: 'If neighbours are alike because of *omitted* factors (shared history, services, housing quality) rather than direct spillover, the error model `u = λ·Wu + ε` fits better. **`GPUSpatialErrorGM`** finds λ by a global scan of the generalized-moments objective (spreg `GM_Error`), then re-estimates β on spatially filtered data.\n\nThe LM tests and robust variants decide between the two: the **Which spatial model?** readout applies the classical rule. Try other **Spatial weights** (rook, k nearest) and another **Health outcome (response)**: diagnostics, ρ and λ all depend on how neighbours are defined, and on whether the model is mis-specified rather than truly spatial.\n\n**Limits:** PLACES values are model-based estimates, not tract surveys; these are ecological associations, not individual risk; and predictors are standardized to compare effects, not to be causal.',
      options: {map: 'error'},
      controls: ['weights', 'outcome', 'map'],
      readouts: ['verdict', 'lambda', 'rho']
    }
  ]
});
