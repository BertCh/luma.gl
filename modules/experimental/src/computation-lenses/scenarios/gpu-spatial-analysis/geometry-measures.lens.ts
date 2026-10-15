// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.geometry-measures';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeometryMeasures',
  title: 'WGS84 geodesic lengths of Wicker Park streets',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, geographicGeometry, geographicMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./geometry-measures.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);

    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGeometryMeasures',
          datasets: [data.id],
          retain: ['lengths'],
          keySpaces: [
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'},
            {id: 'vertex', label: 'vertex', size: data.positionsLonLat.length / 2, indexSpace: '1d'}
          ],
          choices: {
            geometryType: 'lines',
            spatialContext: 'longitude-latitude, ellipsoidal WGS84, meters',
            ellipsoidalEdgeModel: 'geodesic'
          },
          metadata: geographicMetadata('geodesic'),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsLonLat, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const output = await runExample(graph, {roads});
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: geographicGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              lengths: {
                view: output.lengths,
                role: 'intermediate',
                keySpace: 'road',
                meaning:
                  'WGS84 geodesic road length in m; capped Vincenty inverse with Lambert fallback'
              },
              measuredRoads: {
                view: output.measuredRoads,
                role: 'output',
                keySpace: 'vertex',
                geometry: geographicGeometry('lines', {
                  offsetsCapture: 'lineOffsets',
                  valuesCapture: 'lengths'
                })
              }
            };
          }
        })
    };
  }
});
