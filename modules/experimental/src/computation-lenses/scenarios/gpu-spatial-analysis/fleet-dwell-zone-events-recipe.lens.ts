// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.fleet-dwell-zone-events-recipe';
const TRACK_EVENT_CAPACITY = 32;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'addFleetDwellZoneEventsRecipe',
  title: 'Vessel dwell per anchorage from zone events',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata},
      {runExample},
      {getFeatureShells},
      {loadS3NewYorkHarborTracks, loadS3NewYorkHarborZones}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./fleet-dwell-zone-events-recipe.example'),
      import('./trajectory-lens-support'),
      import('./s3-lens-data')
    ]);
    const [data, zones] = await Promise.all([
      loadS3NewYorkHarborTracks(),
      loadS3NewYorkHarborZones()
    ]);
    const shells = getFeatureShells(zones);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'addFleetDwellZoneEventsRecipe',
          datasets: [data.id, zones.id],
          retain: ['zoneVisits', 'zoneDwell', 'fleet-dwell-zone-events-event-overflow'],
          keySpaces: [
            {id: 'zone', label: 'anchorage', size: zones.featureCount, indexSpace: '1d'},
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'},
            {id: 'fix', label: 'AIS fix', size: data.timestamps.length, indexSpace: '1d'}
          ],
          choices: {
            candidateCapacity: 1 << 17,
            trackEventCapacity: TRACK_EVENT_CAPACITY,
            eventCapacity: data.trackCount * TRACK_EVENT_CAPACITY
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              timestamps: uploads.view('timestamps', data.timestamps, 'float32'),
              trackOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            const zoneViews = {
              positions: uploads.view('zonePositions', zones.positionsProjected, 'float32x2'),
              featureOffsets: uploads.view('zoneFeatureOffsets', zones.featureOffsets, 'uint32'),
              polygonOffsets: uploads.view('zonePolygonOffsets', zones.polygonOffsets, 'uint32'),
              ringOffsets: uploads.view('zoneRingOffsets', zones.ringOffsets, 'uint32')
            };
            // Drawing only: the closed shell of each anchorage (holes and extra parts stay in the run).
            const anchorages = uploads.view('anchorages', shells.positions, 'float32x2');
            const anchorageOffsets = uploads.view('anchorageOffsets', shells.offsets, 'uint32');
            const result = await runExample(graph, {
              tracks,
              zones: zoneViews,
              trackEventCapacity: TRACK_EVENT_CAPACITY
            });
            return {
              tracks: {
                view: tracks.positions,
                role: 'context',
                geometry: projectedGeometry('lines', {offsetsCapture: 'trackOffsets'}),
                meaning: 'context'
              },
              trackOffsets: {view: tracks.trackOffsets, role: 'intermediate'},
              anchorageOffsets: {view: anchorageOffsets, role: 'intermediate'},
              anchorages: {
                view: anchorages,
                role: 'input',
                keySpace: 'zone',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'anchorageOffsets'})
              },
              zoneVisits: {
                view: result.zoneVisits,
                role: 'intermediate',
                keySpace: 'zone',
                meaning: 'tracks that visited the anchorage'
              },
              eventOverflow: {
                view: result.dwell.events.status.overflow,
                role: 'intermediate',
                meaning:
                  'zone-events stage overflow word; 0 means every event fit and dwell totals are complete'
              },
              zoneDwell: {
                view: result.zoneDwell,
                role: 'intermediate',
                keySpace: 'zone',
                meaning: 'total dwell in s, summed over tracks'
              },
              dwellAnchorages: {
                view: anchorages,
                role: 'output',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'anchorageOffsets',
                  valuesCapture: 'zoneDwell'
                })
              }
            };
          }
        })
    };
  }
});
