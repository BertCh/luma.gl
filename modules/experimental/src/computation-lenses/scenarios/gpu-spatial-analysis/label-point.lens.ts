// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.label-point';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULabelPoint',
  title: 'Label points of Chicago parks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
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
        import('./label-point.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULabelPoint',
          datasets: [data.id],
          retain: ['labelPoints', 'inscribedRadii', 'degenerate'],
          keySpaces: [
            {id: 'park', label: 'park', size: data.featureCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            initialGridSize: 16,
            refinementGridSize: 8,
            refinementRounds: 6,
            refinementCandidates: 4
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const output = await runExample(graph, {parks});
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              inscribedRadii: {
                view: output.distances,
                role: 'intermediate',
                keySpace: 'park',
                meaning: 'radius in m of the largest circle found around the label point'
              },
              degenerate: {
                view: output.degenerate,
                role: 'intermediate',
                keySpace: 'park',
                meaning: '1 when no strictly interior point exists, else 0'
              },
              labelPoints: {
                view: output.labelPoints,
                role: 'output',
                keySpace: 'park',
                geometry: projectedGeometry('points', {valuesCapture: 'inscribedRadii'}),
                meaning: 'label point of each park, coloured by its inscribed radius in m'
              }
            };
          }
        })
    };
  }
});
