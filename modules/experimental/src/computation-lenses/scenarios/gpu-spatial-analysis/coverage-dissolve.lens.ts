// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.coverage-dissolve';
const LABEL_COUNT = 3;
const RING_CAPACITY = 64;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCoverageDissolve',
  title: 'Chicago community areas dissolved into west-to-east bands',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-community-areas',
    source: 'showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./coverage-dissolve.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    // The community-area dataset has exactly one ring per area, so the drawn paths are the areas.
    assert(data.ringOffsets.length - 1 === data.featureCount);
    const vertexCount = data.positionsProjected.length / 2;
    // The dataset carries no grouping attribute, so the labels come from the input geometry:
    // areas ranked by mean vertex easting, cut into three equal-count west-to-east bands.
    const meanEasting = Array.from({length: data.featureCount}, (_, area) => {
      let sum = 0;
      for (let vertex = data.ringOffsets[area]; vertex < data.ringOffsets[area + 1]; vertex++) {
        sum += data.positionsProjected[vertex * 2];
      }
      return sum / (data.ringOffsets[area + 1] - data.ringOffsets[area]);
    });
    const ranked = meanEasting
      .map((_, area) => area)
      .sort((a, b) => meanEasting[a] - meanEasting[b]);
    const labelValues = new Uint32Array(data.featureCount);
    ranked.forEach((area, rank) => {
      labelValues[area] = Math.floor((rank * LABEL_COUNT) / data.featureCount);
    });
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCoverageDissolve',
          datasets: [data.id],
          retain: [
            'dissolved',
            'dissolvedOffsets',
            'ringLabels',
            'ringCount',
            'overflow',
            'boundarySegmentCount'
          ],
          keySpaces: [
            {id: 'area', label: 'community area', size: data.featureCount, indexSpace: '1d'},
            {id: 'vertex', label: 'area vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'ring', label: 'dissolved ring', size: RING_CAPACITY, indexSpace: '1d'}
          ],
          choices: {snapTolerance: 0, vertexTolerance: 1e-6, splitTouchingRings: true},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const areas = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const labels = uploads.view('labels', labelValues, 'uint32');
            const output = await runExample(graph, {
              areas,
              labels,
              ringCapacity: RING_CAPACITY,
              vertexCapacity: vertexCount + RING_CAPACITY
            });
            return {
              areas: {
                view: areas.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: areas.ringOffsets, role: 'intermediate'},
              labels: {
                view: labels,
                role: 'intermediate',
                keySpace: 'area',
                meaning:
                  'band label 0, 1 or 2: west, middle or east third of the 17 areas by mean vertex easting (derived from the input geometry)'
              },
              dissolvedOffsets: {view: output.ringOffsets, role: 'intermediate'},
              ringLabels: {
                view: output.ringGroups,
                role: 'intermediate',
                keySpace: 'ring',
                meaning: 'band label of each dissolved ring; rows past the ring count are unused'
              },
              ringCount: {view: output.status.count, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              boundarySegmentCount: {
                view: output.status.boundarySegmentCount,
                role: 'intermediate'
              },
              dissolved: {
                view: output.dissolved,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'dissolvedOffsets'}),
                meaning:
                  'GPU-assembled boundary rings per band label; GPU offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
