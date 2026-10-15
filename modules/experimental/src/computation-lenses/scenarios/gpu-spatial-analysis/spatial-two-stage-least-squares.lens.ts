// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'spatial-two-stage-least-squares',
  element: 'GPUSpatialTwoStageLeastSquares',
  title:
    'Diabetes prevalence with a spatially lagged response, instrumented by neighbouring predictors',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {weightsCriterion: 'queen', weightsTransform: 'row', instrumentOrder: 1},
  metadata: {weightsStandardisation: 'row'},
  loadExample: () => import('./spatial-two-stage-least-squares.example'),
  prepare(execution, input, model) {
    const table = model.extractTable(input.graph, {
      id: 'coefficientColumns',
      source: execution.output.table,
      stride: 4,
      rowCount: 5,
      columns: model.ESTIMATOR_TABLE_COLUMNS,
      keySpace: {
        id: 'coefficient',
        label: 'Intercept, predictors and rho',
        size: 5,
        indexSpace: '1d'
      },
      meaning: 'intercept, poverty150Pct, noVehiclePct, uninsuredPct, rho'
    });
    return {
      views: {
        ...table.views,
        summary: execution.output.summary,
        status: execution.output.status.code,
        residuals: execution.output.residuals,
        weightsOverflow: execution.weightsOverflow
      },
      tags: table.tags,
      keySpaces: table.keySpaces
    };
  }
});
