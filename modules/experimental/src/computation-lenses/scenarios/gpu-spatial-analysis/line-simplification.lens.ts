// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-simplification';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineSimplification',
  title: 'Simplify Wicker Park streets at five metres',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./line-simplification.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPULineSimplificationParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineSimplification',
          datasets: [data.id],
          retain: [
            'importance',
            'ids',
            'lineCounts',
            'count',
            'overflow',
            'converged',
            'iterationCount',
            'simplifiedRoads',
            'simplifiedOffsets'
          ],
          keySpaces: [
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 16N metres)',
            neighborhoodRadius: 3,
            method: 'douglas-peucker',
            metric: 'segment',
            tolerance: 5,
            maximumIterations: 64,
            computeImportance: true
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULineSimplificationParameterValues({tolerance: 5}),
              'float32'
            );
            const output = await runExample(graph, {roads, parameters});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              importance: {
                view: output.importance,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning: 'Douglas-Peucker vertex importance in m; endpoints have positive Infinity'
              },
              ids: {view: output.ids, role: 'intermediate'},
              lineCounts: {
                view: output.lineCounts,
                role: 'intermediate',
                keySpace: 'road',
                meaning: 'retained vertex count per road (vertices)'
              },
              count: {view: output.status.count, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              converged: {view: output.status.converged, role: 'intermediate'},
              iterationCount: {view: output.status.iterationCount, role: 'intermediate'},
              simplifiedOffsets: {view: output.simplified.lineOffsets, role: 'intermediate'},
              simplifiedRoads: {
                view: output.simplified.positions,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'simplifiedOffsets'}),
                meaning:
                  'GPU-gathered retained vertices; scanned line counts exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
