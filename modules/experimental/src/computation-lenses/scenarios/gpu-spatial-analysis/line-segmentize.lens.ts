// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-segmentize';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineSegmentize',
  title: 'Densified Wicker Park streets',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
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
        import('./line-segmentize.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPULineSegmentizeParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    const lineCapacity = data.lineCount;
    const vertexCapacity = (data.positionsProjected.length / 2) * 64;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineSegmentize',
          datasets: [data.id],
          retain: [
            'densified',
            'outputOffsets',
            'sourceLineIndices',
            'count',
            'lineCount',
            'overflow'
          ],
          keySpaces: [
            {id: 'road', label: 'road', size: lineCapacity, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            spatialContext: 'native planar (UTM 16N metres)',
            maximumSegmentLength: 25,
            maximumPieceCount: 64,
            closedLines: false
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULineSegmentizeParameterValues({maximumSegmentLength: 25}),
              'float32'
            );
            const output = await runExample(graph, {
              roads,
              parameters,
              vertexCapacity,
              lineCapacity
            });
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              outputOffsets: {view: output.lineOffsets, role: 'intermediate'},
              sourceLineIndices: {
                view: output.sourceLineIndices,
                role: 'intermediate',
                keySpace: 'road',
                meaning: 'source road index (zero-based); unused slots are 0xffffffff'
              },
              count: {view: output.status.count, role: 'intermediate'},
              lineCount: {view: output.status.lineCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              densified: {
                view: output.densified,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'outputOffsets'}),
                meaning: 'GPU-emitted vertices; GPU offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
