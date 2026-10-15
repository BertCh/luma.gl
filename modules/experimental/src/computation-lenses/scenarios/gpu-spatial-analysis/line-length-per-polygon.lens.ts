// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-length-per-polygon';

/** Loads streets crossing three named community areas in a shared UTM 16N metre frame. */
export async function loadLineLengthPerPolygonInputs() {
  const [roadData, areas] = await Promise.all([
    import('../../data/s3-chicago-area-roads.json').then(module => module.default),
    import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
  ]);
  // Both extracts must use the same projected coordinate frame.
  assert(roadData.projected.crs === areas.projected.crs);
  assert(roadData.projected.unit === areas.projected.unit);
  assert(
    roadData.projected.falseOrigin.every(
      (coordinate, index) => coordinate === areas.projected.falseOrigin[index]
    )
  );
  // These source features each have one polygon with one ring.
  assert(areas.ringOffsets.length === areas.featureCount + 1);
  assert(areas.polygonOffsets.length === areas.featureCount + 1);
  const selectedNames = ['area 7', 'area 22', 'area 24'];
  const selectedPositions: number[] = [];
  const selectedOffsets = [0];
  for (const name of selectedNames) {
    const feature = areas.names.indexOf(name);
    // The curated selection is Lincoln Park, Logan Square and West Town, in that order.
    assert(feature >= 0);
    const start = areas.ringOffsets[feature];
    const end = areas.ringOffsets[feature + 1];
    selectedPositions.push(...areas.positionsProjected.subarray(2 * start, 2 * end));
    selectedOffsets.push(selectedPositions.length / 2);
  }
  return {
    roads: {
      id: roadData.id,
      projected: areas.projected,
      positionsProjected: Float32Array.from(roadData.positionsProjected),
      lineOffsets: Uint32Array.from(roadData.lineOffsets)
    },
    areaDatasetId: areas.id,
    selectedNames,
    selectedPositions: Float32Array.from(selectedPositions),
    selectedOffsets: Uint32Array.from(selectedOffsets)
  };
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineLengthPerPolygon',
  title: 'Street length inside community areas',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-area-roads + s3-chicago-community-areas',
    source:
      'showcase public/data/chicago-roads (OpenStreetMap) + showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors + City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata},
      {runExample},
      {roads, areaDatasetId, selectedNames, selectedPositions, selectedOffsets}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./line-length-per-polygon.example'),
      loadLineLengthPerPolygonInputs()
    ]);
    const areaCount = selectedNames.length;
    assert(areaCount > 0);
    const identityOffsets = Uint32Array.from({length: areaCount + 1}, (_, index) => index);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineLengthPerPolygon',
          datasets: [roads.id, areaDatasetId],
          retain: ['lengths', 'segmentCounts', 'overflow'],
          keySpaces: [
            {id: 'area', label: 'community area', size: areaCount, indexSpace: '1d'},
            {
              id: 'areaVertex',
              label: 'area vertex',
              size: selectedPositions.length / 2,
              indexSpace: '1d'
            },
            {
              id: 'roadVertex',
              label: 'road vertex',
              size: roads.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'area ring', size: areaCount + 1, indexSpace: '1d'}
          ],
          choices: {maximumCrossingCount: 64, pairCapacity: 4096},
          metadata: projectedMetadata(roads.projected),
          async build(graph, uploads) {
            const roadFields = {
              positions: uploads.view('roads', roads.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', roads.lineOffsets, 'uint32')
            };
            const areaFields = {
              positions: uploads.view('areas', selectedPositions, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', selectedOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', identityOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', identityOffsets, 'uint32')
            };
            const output = await runExample(graph, {roads: roadFields, areas: areaFields});
            return {
              roads: {
                view: roadFields.positions,
                role: 'input',
                keySpace: 'roadVertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roadFields.lineOffsets, role: 'intermediate'},
              areas: {
                view: areaFields.positions,
                role: 'input',
                keySpace: 'areaVertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning:
                  'Lincoln Park (7), Logan Square (22), West Town (24); sampled streets cross all three'
              },
              ringOffsets: {view: areaFields.ringOffsets, role: 'intermediate', keySpace: 'ring'},
              segmentCounts: {
                view: output.segmentCounts,
                role: 'intermediate',
                keySpace: 'area',
                meaning: 'road segments with positive length inside the area (count)'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              lengths: {
                view: output.lengths,
                role: 'intermediate',
                keySpace: 'area',
                meaning: 'metres of road inside the area'
              },
              areasByRoadLength: {
                view: areaFields.positions,
                role: 'output',
                keySpace: 'areaVertex',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'ringOffsets',
                  valuesCapture: 'lengths'
                }),
                meaning: 'community-area outlines coloured by metres of road inside'
              }
            };
          }
        })
    };
  }
});
