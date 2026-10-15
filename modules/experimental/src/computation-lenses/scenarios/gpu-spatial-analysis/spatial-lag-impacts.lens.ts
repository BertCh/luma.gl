// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'spatial-lag-impacts',
  element: 'GPUSpatialLagImpacts',
  title: 'Direct, indirect and total effects on diabetes prevalence in the spatial lag model',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {maximumPower: 128, rhoBound: 1},
  metadata: {
    weightsStandardisation: 'row',
    weightsCriterion: 'queen',
    weightsTransform: 'row',
    fitEstimator: 'GPUSpatialLagMaximumLikelihood'
  },
  loadExample: () => import('./spatial-lag-impacts.example'),
  prepare(execution, input, model) {
    const table = model.extractTable(input.graph, {
      id: 'impactColumns',
      source: execution.output.impacts,
      stride: 3,
      rowCount: 3,
      columns: {direct: 0, indirect: 1, total: 2},
      keySpace: {id: 'predictor', label: 'Predictor', size: 3, indexSpace: '1d'},
      meaning: 'poverty150Pct, noVehiclePct, uninsuredPct; diabetes percentage points per point'
    });
    return {
      views: {
        ...table.views,
        table: execution.fitOutput.table,
        status: execution.fitOutput.status.code,
        infinityNorm: execution.fitOutput.infinityNorm,
        weightsOverflow: execution.weightsOverflow
      },
      tags: table.tags,
      keySpaces: table.keySpaces
    };
  }
});
