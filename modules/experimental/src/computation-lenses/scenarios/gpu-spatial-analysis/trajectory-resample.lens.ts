// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.trajectory-resample';
const SAMPLE_COUNT = 16;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrajectoryResample',
  title: 'Harbour tracks resampled to 16 points',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata, createFixedOffsets},
      {runExample},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./trajectory-resample.example'),
      import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
    ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrajectoryResample',
          datasets: [data.id],
          retain: ['samples', 'sampleTimes'],
          keySpaces: [
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'},
            {id: 'fix', label: 'AIS fix', size: data.timestamps.length, indexSpace: '1d'},
            {
              id: 'sample',
              label: 'track and sample',
              size: data.trackCount * SAMPLE_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {
            spatialContext: 'native planar (UTM 18N metres)',
            spacing: 'arc-length',
            sampleCount: SAMPLE_COUNT
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              timestamps: uploads.view('timestamps', data.timestamps, 'float32'),
              trackOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            // Arc-length spacing writes exactly sampleCount samples per track, so the display
            // offsets are the fixed index 0, 16, 32, ... (an index, not data).
            const sampleOffsets = uploads.view(
              'sampleOffsets',
              createFixedOffsets(data.trackCount, SAMPLE_COUNT),
              'uint32'
            );
            const output = await runExample(graph, {tracks, sampleCount: SAMPLE_COUNT});
            return {
              tracks: {
                view: tracks.positions,
                role: 'input',
                keySpace: 'fix',
                geometry: projectedGeometry('lines', {offsetsCapture: 'trackOffsets'})
              },
              trackOffsets: {view: tracks.trackOffsets, role: 'intermediate'},
              sampleOffsets: {view: sampleOffsets, role: 'intermediate'},
              sampleTimes: {
                view: output.sampleTimes,
                role: 'intermediate',
                keySpace: 'sample',
                meaning: 'sample time in s since the track start; row = track * 16 + sample'
              },
              samples: {
                view: output.samples,
                role: 'output',
                keySpace: 'sample',
                geometry: projectedGeometry('lines', {offsetsCapture: 'sampleOffsets'}),
                meaning: 'GPU-written samples, 16 per track in track order'
              }
            };
          }
        })
    };
  }
});
