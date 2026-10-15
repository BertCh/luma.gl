// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'spatial-regression-diagnostics',
  element: 'GPUSpatialRegressionDiagnostics',
  title: 'Which spatial model do the diabetes OLS residuals call for?',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {weightsCriterion: 'queen', weightsTransform: 'row', restandardizeRows: false},
  metadata: {weightsStandardisation: 'row', pCorrection: 'none'},
  loadExample: () => import('./spatial-regression-diagnostics.example'),
  prepare(execution, input, model) {
    const table = model.extractTable(input.graph, {
      id: 'diagnosticsColumns',
      source: execution.output.tests,
      stride: 3,
      rowCount: 6,
      columns: {statistics: 0, degreesOfFreedom: 1, pValues: 2},
      keySpace: {id: 'test', label: 'Diagnostic test', size: 6, indexSpace: '1d'},
      meaning:
        'LM-lag, LM-error, robust LM-lag, robust LM-error, LM-SARMA, Moran I of residuals (z, two-sided p)'
    });
    return {
      views: {
        ...table.views,
        summary: execution.output.summary,
        status: execution.output.status.code,
        weightsOverflow: execution.weightsOverflow
      },
      tags: table.tags,
      keySpaces: table.keySpaces
    };
  }
});
