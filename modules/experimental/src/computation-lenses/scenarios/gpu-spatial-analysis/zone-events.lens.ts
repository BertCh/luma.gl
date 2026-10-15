// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.zone-events';
const TRACK_EVENT_CAPACITY = 32;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUZoneEvents',
  title: 'Anchorage entries and exits of harbour vessels',
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
      import('./zone-events.example'),
      import('./trajectory-lens-support'),
      import('./s3-lens-data')
    ]);
    const [data, zones] = await Promise.all([
      loadS3NewYorkHarborTracks(),
      loadS3NewYorkHarborZones()
    ]);
    const shells = getFeatureShells(zones);
    const eventCapacity = data.trackCount * TRACK_EVENT_CAPACITY;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUZoneEvents',
          datasets: [data.id, zones.id],
          retain: [
            'eventTracks',
            'eventCount',
            'eventOverflow',
            'eventZones',
            'eventTypes',
            'eventTimes',
            'dwellTimes',
            'visitCounts',
            'trackEventCounts'
          ],
          keySpaces: [
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'},
            {id: 'fix', label: 'AIS fix', size: data.timestamps.length, indexSpace: '1d'},
            {id: 'event', label: 'event slot', size: eventCapacity, indexSpace: '1d'},
            {
              id: 'trackZone',
              label: 'track and zone',
              size: data.trackCount * zones.featureCount,
              indexSpace: '1d'
            }
          ],
          choices: {
            candidateCapacity: 1 << 17,
            trackEventCapacity: TRACK_EVENT_CAPACITY
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
            const zoneShells = uploads.view('zoneShells', shells.positions, 'float32x2');
            const zoneShellOffsets = uploads.view('zoneShellOffsets', shells.offsets, 'uint32');
            const output = await runExample(graph, {
              tracks,
              zones: zoneViews,
              trackEventCapacity: TRACK_EVENT_CAPACITY
            });
            const unused = 'rows past the event count are unused';
            return {
              zones: {
                view: zoneShells,
                role: 'context',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'zoneShellOffsets'}),
                meaning: 'context'
              },
              zoneShellOffsets: {view: zoneShellOffsets, role: 'intermediate'},
              tracks: {
                view: tracks.positions,
                role: 'input',
                keySpace: 'fix',
                geometry: projectedGeometry('lines', {offsetsCapture: 'trackOffsets'})
              },
              trackOffsets: {view: tracks.trackOffsets, role: 'intermediate'},
              eventTracks: {
                view: output.events.ids,
                role: 'intermediate',
                keySpace: 'event',
                meaning: `track index of each event; ${unused}`
              },
              eventZones: {
                view: output.events.eventZones,
                role: 'intermediate',
                keySpace: 'event',
                meaning: `zone index of each event; ${unused}`
              },
              eventTypes: {
                view: output.events.eventTypes,
                role: 'intermediate',
                keySpace: 'event',
                meaning: `0 enter, 1 exit; ${unused}`
              },
              eventTimes: {
                view: output.events.eventTimes,
                role: 'intermediate',
                keySpace: 'event',
                meaning: `crossing time in s since the track start; ${unused}`
              },
              eventCount: {view: output.events.status.count, role: 'intermediate'},
              eventOverflow: {view: output.events.status.overflow, role: 'intermediate'},
              dwellTimes: {
                view: output.dwellTimes,
                role: 'intermediate',
                keySpace: 'trackZone',
                meaning: `dwell in s; row = track * ${zones.featureCount} + zone`
              },
              visitCounts: {
                view: output.visitCounts,
                role: 'intermediate',
                keySpace: 'trackZone',
                meaning: `visits; row = track * ${zones.featureCount} + zone`
              },
              trackEventCounts: {
                view: output.trackEventCounts,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'zone entries and exits per track (unclamped)'
              },
              countedTracks: {
                view: tracks.positions,
                role: 'output',
                geometry: projectedGeometry('lines', {
                  offsetsCapture: 'trackOffsets',
                  valuesCapture: 'trackEventCounts'
                })
              }
            };
          }
        })
    };
  }
});
