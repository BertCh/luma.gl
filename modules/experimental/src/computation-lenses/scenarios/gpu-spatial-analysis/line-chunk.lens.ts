// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-chunk';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineChunk',
  title: 'Chunks Wicker Park streets',
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
        import('./line-chunk.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPULineChunkParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    // Sum per-segment upper bounds; add one piece per road for degenerate lines and rounding.
    let lineCapacity = data.lineCount;
    for (let road = 0; road < data.lineCount; road++) {
      for (let vertex = data.lineOffsets[road]; vertex + 1 < data.lineOffsets[road + 1]; vertex++) {
        lineCapacity += Math.ceil(
          Math.hypot(
            data.positionsProjected[vertex * 2 + 2] - data.positionsProjected[vertex * 2],
            data.positionsProjected[vertex * 2 + 3] - data.positionsProjected[vertex * 2 + 1]
          ) / 100
        );
      }
    }
    const vertexCapacity = data.positionsProjected.length / 2 + 2 * lineCapacity;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineChunk',
          datasets: [data.id],
          retain: [
            'chunks',
            'outputOffsets',
            'sourceLineIndices',
            'count',
            'lineCount',
            'overflow'
          ],
          keySpaces: [
            {id: 'piece', label: 'piece', size: lineCapacity, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            spatialContext: 'native planar (UTM 16N metres)',
            mode: 'chunk',
            chunkLength: 100,
            startMeasure: 0,
            endMeasure: Math.fround(3.4e38)
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULineChunkParameterValues({
                chunkLength: 100,
                startMeasure: 0,
                endMeasure: 3.4e38
              }),
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
                keySpace: 'piece',
                meaning: 'source road index (zero-based); unused slots are 0xffffffff'
              },
              count: {view: output.status.count, role: 'intermediate'},
              lineCount: {view: output.status.lineCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              chunks: {
                view: output.chunks,
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
