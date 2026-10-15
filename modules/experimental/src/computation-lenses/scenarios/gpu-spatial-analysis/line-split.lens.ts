// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-split';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineSplit',
  title: 'Split Wicker Park streets at intersections',
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
        import('./line-split.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    const intersectionCapacity = 4096;
    const pieceCapacity = data.lineCount + 4 * intersectionCapacity;
    const vertexCapacity = data.positionsProjected.length / 2 + 8 * intersectionCapacity;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineSplit',
          datasets: [data.id],
          retain: [
            'piecePositions',
            'pieceOffsets',
            'sourceIds',
            'count',
            'vertexCount',
            'overflow',
            'candidateOverflow',
            'uncertainCount'
          ],
          keySpaces: [
            {id: 'piece', label: 'piece', size: pieceCapacity, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {intersectionCapacity},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const output = await runExample(graph, {
              roads,
              intersectionCapacity,
              pieceCapacity,
              vertexCapacity
            });
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              pieceOffsets: {view: output.pieceOffsets, role: 'intermediate'},
              sourceIds: {
                view: output.pieces.sourceIds,
                role: 'intermediate',
                keySpace: 'piece',
                meaning: 'source road index (zero-based); unused slots are 0xffffffff'
              },
              count: {view: output.pieces.status.count, role: 'intermediate'},
              vertexCount: {view: output.pieces.status.vertexCount, role: 'intermediate'},
              overflow: {view: output.pieces.status.overflow, role: 'intermediate'},
              candidateOverflow: {
                view: output.pieces.status.candidateOverflow,
                role: 'intermediate'
              },
              uncertainCount: {view: output.uncertainCount, role: 'intermediate'},
              piecePositions: {
                view: output.piecePositions,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'pieceOffsets'}),
                meaning: 'GPU piece offsets delimit only complete emitted paths'
              }
            };
          }
        })
    };
  }
});
