// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-noding';
const SEGMENT_CAPACITY = 2048;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineNoding',
  title: 'Wicker Park street centrelines noded at their crossings',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata, createFixedOffsets},
      {runExample},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./line-noding.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
    ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineNoding',
          datasets: [data.id],
          retain: [
            'sourceFeatureIds',
            'sourceEdgeIds',
            'count',
            'requiredCount',
            'overflow',
            'candidateOverflow'
          ],
          keySpaces: [
            {id: 'segment', label: 'noded segment', size: SEGMENT_CAPACITY, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'road vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 16N metres)',
            intersectionCapacity: 1024,
            spatialSort: 'element default',
            leafCapacity: 'element default'
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const segmentSlots = uploads.view(
              'segments',
              new Float32Array(4 * SEGMENT_CAPACITY).fill(Number.NaN),
              'float32x4'
            );
            // Display only: the same segment bytes read as two float32x2 endpoints per segment.
            const endpoints = graph.createDataView(segmentSlots.buffer, {
              format: 'float32x2',
              length: 2 * SEGMENT_CAPACITY
            });
            // Display index for the segment endpoints (2 per segment); not data.
            const segmentOffsets = uploads.view(
              'segmentOffsets',
              createFixedOffsets(SEGMENT_CAPACITY, 2),
              'uint32'
            );
            const output = await runExample(graph, {roads, segmentSlots});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'}),
                meaning: '223 Wicker Park street centrelines, UTM 16N metres'
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate', keySpace: 'road'},
              segmentOffsets: {view: segmentOffsets, role: 'intermediate'},
              sourceFeatureIds: {
                view: output.sourceFeatureIds,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'source road of each noded segment (zero-based line row)'
              },
              sourceEdgeIds: {
                view: output.sourceEdgeIds,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'edge of the source road that each noded segment lies on'
              },
              count: {
                view: output.status.count,
                role: 'intermediate',
                meaning: 'noded segments written'
              },
              requiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              candidateOverflow: {view: output.status.candidateOverflow, role: 'intermediate'},
              nodedSegments: {
                view: endpoints,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'segmentOffsets'}),
                meaning:
                  'noded segments, two endpoints each, UTM 16N metres; unwritten slots are NaN (uploaded as NaN) so only GPU-written segments draw'
              }
            };
          }
        })
    };
  }
});
