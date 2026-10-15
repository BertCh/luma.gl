// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-density';
/** Grid origin (south-west corner) and square cell size in UTM 16N metres, covering every road. */
const GRID = {minX: 3600, minY: 9200, cellWidth: 100, cellHeight: 100, columns: 15, rows: 13};

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineDensity',
  title: 'Street length per 100 m cell in Wicker Park',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata, sumCapture}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./line-density.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const {getGPULineDensityParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    const {columns, rows, cellWidth, cellHeight, minX, minY} = GRID;
    // Total road length from the uploaded vertices (f64); the grid covers every vertex.
    const positions = data.positionsProjected;
    let totalLength = 0;
    for (let line = 0; line < data.lineCount; line++) {
      for (let row = data.lineOffsets[line]; row + 1 < data.lineOffsets[line + 1]; row++) {
        totalLength += Math.hypot(
          positions[2 * row + 2] - positions[2 * row],
          positions[2 * row + 3] - positions[2 * row + 1]
        );
      }
    }
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineDensity',
          datasets: [data.id],
          retain: ['lengths', 'densities', 'overflow', 'pieceCount'],
          keySpaces: [
            {
              id: 'cell',
              label: '100 m cell, rows from south',
              size: columns * rows,
              indexSpace: '2d',
              shape: [columns, rows]
            },
            {id: 'vertex', label: 'road vertex', size: positions.length / 2, indexSpace: '1d'},
            {id: 'road', label: 'road offset', size: data.lineOffsets.length, indexSpace: '1d'}
          ],
          choices: {minX, minY, cellWidth, cellHeight, columns, rows},
          metadata: snapshots => ({
            ...projectedMetadata(data.projected),
            mappedVariableExtent: 'extensive',
            denominator: 'cell area (constant 100 m x 100 m)',
            // Road length is conserved by clipping to a grid that covers every vertex (f32 pieces).
            sumBefore: totalLength,
            sumAfter: sumCapture(snapshots, 'lengths') ?? Number.NaN,
            conservationTolerance: 1e-5
          }),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULineDensityParameterValues({minX, minY, cellWidth, cellHeight}),
              'float32'
            );
            const output = await runExample(graph, {roads, parameters, columns, rows});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate', keySpace: 'road'},
              densities: {
                view: output.densities,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'metres of road per square metre of cell (cell area 10,000 m²)'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              pieceCount: {view: output.status.requiredCount, role: 'intermediate'},
              lengths: {
                view: output.lengths,
                role: 'output',
                keySpace: 'cell',
                geometry: {
                  kind: 'raster',
                  shape: [columns, rows],
                  coordinateModel: 'projected',
                  cellSize: cellWidth,
                  unit: 'm',
                  origin: [minX, minY],
                  rowDirection: 'north'
                },
                meaning: 'metres of road inside each 100 m cell'
              }
            };
          }
        })
    };
  }
});
