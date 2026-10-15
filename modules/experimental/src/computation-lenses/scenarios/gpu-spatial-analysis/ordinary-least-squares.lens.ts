// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'ordinary-least-squares',
  element: 'GPUOrdinaryLeastSquares',
  title: 'Diabetes prevalence on poverty, car-free and uninsured shares across Chicago tracts',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {ridgeLambda: 0},
  loadExample: () => import('./ordinary-least-squares.example'),
  prepare(execution) {
    const rowNames = 'intercept, poverty150Pct, noVehiclePct, uninsuredPct';
    const meaning = `${rowNames}; diabetes percentage points per percentage point`;
    const tag = (name: string) => ({
      role: 'output' as const,
      keySpace: 'coefficient',
      meaning: `${name}; rows: ${meaning}`
    });
    return {
      views: execution.views,
      tags: {
        coefficients: tag('coefficients'),
        standardErrors: tag('standardErrors'),
        tStatistics: tag('tStatistics')
      },
      keySpaces: [{id: 'coefficient', label: 'Intercept and predictors', size: 4, indexSpace: '1d'}]
    };
  }
});
