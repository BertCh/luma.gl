// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.vertex-snap';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUVertexSnap',
  title: 'Chicago park vertices snapped to community-area vertices',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks + s3-chicago-community-areas',
    source:
      'showcase public/data/chicago-parks (OpenStreetMap via Overpass) + showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors + City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data, areas] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./vertex-snap.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks()),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    const {getGPUVertexSnapParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUVertexSnap',
          datasets: [data.id, areas.id],
          retain: ['snappedParks', 'referenceRows'],
          keySpaces: [
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'ring', size: data.ringOffsets.length - 1, indexSpace: '1d'},
            {
              id: 'area-vertex',
              label: 'community-area vertex',
              size: areas.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {tolerance: 50},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            // Parks and community areas share the UTM 16N frame, so their vertices are comparable.
            const areaVertices = uploads.view(
              'areaVertices',
              areas.positionsProjected,
              'float32x2'
            );
            const areaRingOffsets = uploads.view('areaRingOffsets', areas.ringOffsets, 'uint32');
            const parameters = uploads.view(
              'parameters',
              getGPUVertexSnapParameterValues({tolerance: 50}),
              'float32'
            );
            const output = await runExample(graph, {parks, areaVertices, parameters});
            return {
              areas: {
                view: areaVertices,
                role: 'context',
                keySpace: 'area-vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'areaRingOffsets'}),
                meaning: 'community-area boundaries, whose vertices are the snap references'
              },
              areaRingOffsets: {view: areaRingOffsets, role: 'intermediate'},
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate', keySpace: 'ring'},
              referenceRows: {
                view: output.referenceRows,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning:
                  'row of the community-area vertex the park vertex moved to; 0xffffffff when it stayed'
              },
              snappedParks: {
                view: output.snappedParks,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning: 'park vertices after snapping to area vertices closer than 50 m'
              }
            };
          }
        })
    };
  }
});
