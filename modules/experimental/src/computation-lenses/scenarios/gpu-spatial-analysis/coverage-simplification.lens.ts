// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.coverage-simplification';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCoverageSimplification',
  title: 'Gap-free simplification of Chicago community areas',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-community-areas',
    source: 'showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./coverage-simplification.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    const {getGPULineSimplificationParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    // The community-area dataset has exactly one ring per area, so the drawn paths are the areas.
    assert(data.ringOffsets.length - 1 === data.featureCount);
    const vertexCount = data.positionsProjected.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCoverageSimplification',
          datasets: [data.id],
          retain: [
            'simplifiedAreas',
            'simplifiedOffsets',
            'outputMask',
            'topologyStatistics',
            'overflow',
            'converged'
          ],
          keySpaces: [
            {id: 'vertex', label: 'area vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'area', label: 'community area', size: data.featureCount, indexSpace: '1d'},
            {id: 'statistic', label: 'topology statistic', size: 4, indexSpace: '1d'}
          ],
          choices: {
            snapTolerance: 0,
            tolerance: 50,
            simplifyBoundary: true,
            maximumIterations: 32,
            topologyIterationCount: 4,
            topologyPairCapacity: Math.max(256, 4 * vertexCount)
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const areas = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULineSimplificationParameterValues({tolerance: 50}),
              'float32'
            );
            const output = await runExample(graph, {areas, parameters});
            return {
              areas: {
                view: areas.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: areas.ringOffsets, role: 'intermediate'},
              simplifiedOffsets: {view: output.ringOffsets, role: 'intermediate'},
              outputMask: {
                view: output.outputMask,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning: '1 when the input vertex is kept, else 0; ring starts are always kept'
              },
              topologyStatistics: {
                view: output.topologyStatistics,
                role: 'intermediate',
                keySpace: 'statistic',
                meaning:
                  'crossings before repair, crossings remaining, vertices restored, pair-capacity overflow (0 or 1)'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              converged: {view: output.status.converged, role: 'intermediate'},
              simplifiedAreas: {
                view: output.simplifiedAreas,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'simplifiedOffsets'}),
                meaning: 'GPU-kept vertices of each ring; GPU offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
