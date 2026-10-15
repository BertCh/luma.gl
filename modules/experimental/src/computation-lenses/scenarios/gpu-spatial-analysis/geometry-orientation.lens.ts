// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {assert} from '@luma.gl/core';

const SCENARIO_ID = 'gpu-spatial-analysis.geometry-orientation';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeometryOrientation',
  title: 'Winding of Chicago parks reoriented to GeoJSON',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks',
    source: 'OpenStreetMap via Overpass (showcase chicago-parks, 2026-10-07)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./geometry-orientation.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    const {getGPUGeometryOrientationParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    // The park dataset has exactly one ring per park, so per-ring flags are per-park values.
    assert(data.ringOffsets.length - 1 === data.featureCount);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGeometryOrientation',
          datasets: [data.id],
          retain: ['orientedParks', 'reversedRings'],
          keySpaces: [
            {id: 'park', label: 'park (one ring each)', size: data.featureCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {mode: 'orient-polygons', exteriorClockwise: false},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPUGeometryOrientationParameterValues({exteriorClockwise: false}),
              'float32'
            );
            const output = await runExample(graph, {parks, parameters});
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              reversedRings: {
                view: output.reversedRings,
                role: 'intermediate',
                keySpace: 'park',
                meaning: '1 when the park ring was reversed to counter-clockwise, else 0'
              },
              orientedParks: {
                view: output.orientedParks,
                role: 'intermediate',
                keySpace: 'vertex',
                meaning:
                  'park vertices after orientation; same outlines, reversed order where flagged'
              },
              reversedParks: {
                view: parks.positions,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'ringOffsets',
                  valuesCapture: 'reversedRings'
                }),
                meaning: 'input outlines coloured by whether their ring was reversed'
              }
            };
          }
        })
    };
  }
});
