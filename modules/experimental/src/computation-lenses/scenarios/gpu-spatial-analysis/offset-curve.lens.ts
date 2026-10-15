// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.offset-curve';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUOffsetCurve',
  title: '20 m left offset of Wicker Park streets',
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
        import('./offset-curve.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPUOffsetCurveParameterValues, getGPUOffsetCurveRowsPerVertex} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    const vertexCount = data.positionsProjected.length / 2;
    const rowsPerVertex = getGPUOffsetCurveRowsPerVertex('round', 8);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUOffsetCurve',
          datasets: [data.id],
          retain: ['offsetCurve', 'offsetCounts'],
          keySpaces: [
            {id: 'vertex', label: 'street vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'},
            {
              id: 'offset-row',
              label: 'offset row',
              size: vertexCount * rowsPerVertex,
              indexSpace: '1d'
            }
          ],
          choices: {
            distance: 20,
            mitreLimit: 5,
            joinStyle: 'round',
            insideTurns: 'intersection',
            quadSegments: 8
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPUOffsetCurveParameterValues({distance: 20}),
              'float32'
            );
            // Display index: road r owns rows [lineOffsets[r] * k, lineOffsets[r + 1] * k); not data.
            const offsetLineOffsets = uploads.view(
              'offsetLineOffsets',
              Uint32Array.from(data.lineOffsets, offset => offset * rowsPerVertex),
              'uint32'
            );
            const output = await runExample(graph, {roads, parameters});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate', keySpace: 'road'},
              offsetLineOffsets: {view: offsetLineOffsets, role: 'intermediate'},
              offsetCounts: {
                view: output.counts,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning:
                  'distinct offset points written for the street vertex (count); the other rows repeat the last point'
              },
              offsetCurve: {
                view: output.offsetCurve,
                role: 'output',
                keySpace: 'offset-row',
                geometry: projectedGeometry('lines', {offsetsCapture: 'offsetLineOffsets'}),
                meaning: `left offset of each street by 20 m, ${rowsPerVertex} rows per vertex; padding rows repeat the last point and draw nothing`
              }
            };
          }
        })
    };
  }
});
