// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.rectangle-clip';
/** Clip rectangle in UTM 16N metres, inside the roads' extent (x 3606-5085, y 9246-10444). */
const RECTANGLE = {minX: 3900, minY: 9500, maxX: 4800, maxY: 10200};

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPURectangleClip',
  title: 'Wicker Park streets clipped to a rectangle',
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
        import('./rectangle-clip.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPURectangleClipParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    const vertexCount = data.positionsProjected.length / 2;
    // Worst case: every segment becomes its own two-vertex line.
    const segmentCount = vertexCount - data.lineCount;
    const lineCapacity = segmentCount;
    const vertexCapacity = 2 * segmentCount;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPURectangleClip',
          datasets: [data.id],
          retain: [
            'clipped',
            'clippedOffsets',
            'sourceLineIndices',
            'count',
            'lineCount',
            'overflow'
          ],
          keySpaces: [
            {id: 'clippedLine', label: 'clipped line', size: lineCapacity, indexSpace: '1d'},
            {id: 'vertex', label: 'road vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'road', label: 'road offset', size: data.lineOffsets.length, indexSpace: '1d'}
          ],
          choices: RECTANGLE,
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPURectangleClipParameterValues(RECTANGLE),
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
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate', keySpace: 'road'},
              clippedOffsets: {view: output.lineOffsets, role: 'intermediate'},
              sourceLineIndices: {
                view: output.sourceLineIndices,
                role: 'intermediate',
                keySpace: 'clippedLine',
                meaning: 'source road index (zero-based); unused slots are 0xffffffff'
              },
              count: {view: output.status.count, role: 'intermediate'},
              lineCount: {view: output.status.lineCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              clipped: {
                view: output.positions,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'clippedOffsets'}),
                meaning:
                  'road pieces inside the rectangle (x 3900-4800 m, y 9500-10200 m); GPU offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
