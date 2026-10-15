// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.local-frame-projection';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULocalFrameProjection',
  title: 'Wicker Park streets from lon/lat to local metres',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, projectedGeometry, geographicMetadata},
      {runExample},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./local-frame-projection.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
    ]);
    const {createGPULocalFrame} = await import('@luma.gl/experimental/gpu-spatial-analysis');
    const longitudes = data.positionsLonLat.filter((_, index) => index % 2 === 0);
    const latitudes = data.positionsLonLat.filter((_, index) => index % 2 === 1);
    const bounds: [number, number, number, number] = [
      Math.min(...longitudes) - 0.001,
      Math.min(...latitudes) - 0.001,
      Math.max(...longitudes) + 0.001,
      Math.max(...latitudes) + 0.001
    ];
    const origin: [number, number] = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2];
    const frame = createGPULocalFrame({origin, kind: 'azimuthal-equidistant', datum: 'EPSG:4326'});
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULocalFrameProjection',
          datasets: [data.id],
          retain: ['projectedRoads', 'validity'],
          keySpaces: [
            {id: 'vertex', label: 'vertex', size: data.positionsLonLat.length / 2, indexSpace: '1d'}
          ],
          choices: {
            frame: `azimuthal-equidistant, origin ${origin.join(', ')}, EPSG:4326`,
            bounds: bounds.join(', '),
            tolerance: 0.01
          },
          metadata: {
            ...geographicMetadata(),
            outputCoordinateModel: 'projected',
            outputCrs: `${frame.kind} local frame at ${frame.origin.join(', ')}`,
            outputUnit: 'm'
          },
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsLonLat, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const output = await runExample(graph, {roads, frame, bounds});
            uploads.own(output.projection);
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: geographicGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              validity: {
                view: output.validity,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning: 'projection validity: 1 accepted, 0 outside bounds (unitless)'
              },
              projectedRoads: {
                view: output.projectedRoads,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'}),
                meaning:
                  'local azimuthal-equidistant metres; input uses geographic degrees, output uses a different coordinate model'
              }
            };
          }
        })
    };
  }
});
