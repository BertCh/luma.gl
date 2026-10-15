// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.trajectory-playhead';
const PLAYHEAD_SECONDS = 3600;
const MAXIMUM_GAP_SECONDS = 300;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrajectoryPlayhead',
  title: 'Harbour vessels at 15:00Z',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./trajectory-playhead.example'),
        import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
      ]);
    const {getGPUTrajectoryPlayheadParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrajectoryPlayhead',
          datasets: [data.id],
          retain: [
            'currentPositions',
            'headings',
            'speeds',
            'trackStatus',
            'segmentRows',
            'segmentFractions',
            'activeTrackIds',
            'activeTrackCount',
            'activeTrackOverflow'
          ],
          keySpaces: [
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'},
            {id: 'fix', label: 'AIS fix', size: data.timestamps.length, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 18N metres)',
            playhead: PLAYHEAD_SECONDS,
            maximumGap: MAXIMUM_GAP_SECONDS
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              timestamps: uploads.view('timestamps', data.timestamps, 'float32'),
              trackOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPUTrajectoryPlayheadParameterValues({
                playhead: PLAYHEAD_SECONDS,
                maximumGap: MAXIMUM_GAP_SECONDS
              }),
              'float32'
            );
            const output = await runExample(graph, {tracks, parameters});
            return {
              tracks: {
                view: tracks.positions,
                role: 'input',
                keySpace: 'fix',
                geometry: projectedGeometry('lines', {offsetsCapture: 'trackOffsets'})
              },
              trackOffsets: {view: tracks.trackOffsets, role: 'intermediate'},
              trackStatus: {
                view: output.trackStatus,
                role: 'intermediate',
                keySpace: 'track',
                meaning:
                  'status at the playhead: 0 empty, 1 active, 2 before start, 3 after end, 4 gap'
              },
              speeds: {
                view: output.speeds,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'speed at the playhead in m/s; zero unless the track is active'
              },
              headings: {
                view: output.headings,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'heading in radians counterclockwise from east'
              },
              segmentRows: {
                view: output.segmentRows,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'first row of the bracketing segment'
              },
              segmentFractions: {
                view: output.segmentFractions,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'interpolation fraction inside the bracketing segment'
              },
              activeTrackIds: {
                view: output.activeTracks.ids,
                role: 'intermediate',
                meaning: 'active track indices in ascending order; rows past the count are unused'
              },
              activeTrackCount: {view: output.activeTracks.status.count, role: 'intermediate'},
              activeTrackOverflow: {
                view: output.activeTracks.status.overflow,
                role: 'intermediate'
              },
              currentPositions: {
                view: output.currentPositions,
                role: 'output',
                keySpace: 'track',
                geometry: projectedGeometry('points', {valuesCapture: 'trackStatus'}),
                meaning: 'position of each track at 3600 s after 14:00Z (clamped outside its range)'
              }
            };
          }
        })
    };
  }
});
