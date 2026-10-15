// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.trajectory-metrics';
const STOP_CAPACITY = 64;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrajectoryMetrics',
  title: 'Average speed of twelve harbour vessels',
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
        import('./trajectory-metrics.example'),
        import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
      ]);
    const {getGPUTrajectoryMetricsParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrajectoryMetrics',
          datasets: [data.id],
          retain: [
            'trackLengths',
            'trackDurations',
            'averageSpeeds',
            'maximumSpeeds',
            'trackStopCounts',
            'stopTracks',
            'stopCount',
            'stopOverflow',
            'stopCentroids',
            'stopDurations'
          ],
          keySpaces: [
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'},
            {id: 'fix', label: 'AIS fix', size: data.timestamps.length, indexSpace: '1d'},
            {id: 'stop', label: 'stop slot', size: STOP_CAPACITY, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 18N metres)',
            stopSpeedThreshold: 0.5,
            stopMinimumDuration: 300
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
              getGPUTrajectoryMetricsParameterValues({
                stopSpeedThreshold: 0.5,
                stopMinimumDuration: 300
              }),
              'float32'
            );
            const output = await runExample(graph, {
              tracks,
              parameters,
              stopCapacity: STOP_CAPACITY
            });
            const unused = 'rows past the stop count are unused';
            return {
              tracks: {
                view: tracks.positions,
                role: 'input',
                keySpace: 'fix',
                geometry: projectedGeometry('lines', {offsetsCapture: 'trackOffsets'})
              },
              trackOffsets: {view: tracks.trackOffsets, role: 'intermediate'},
              timestamps: {
                view: tracks.timestamps,
                role: 'intermediate',
                keySpace: 'fix',
                meaning: `seconds since 2024-06-12T14:00:00Z`
              },
              trackLengths: {
                view: output.trackLengths,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'track length in m'
              },
              trackDurations: {
                view: output.trackDurations,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'track duration in s'
              },
              averageSpeeds: {
                view: output.averageSpeeds,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'average speed in m/s (length over duration)'
              },
              maximumSpeeds: {
                view: output.maximumSpeeds,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'maximum step speed in m/s'
              },
              trackStopCounts: {
                view: output.trackStopCounts,
                role: 'intermediate',
                keySpace: 'track',
                meaning: 'stops per track (unclamped)'
              },
              stopTracks: {
                view: output.stops.ids,
                role: 'intermediate',
                keySpace: 'stop',
                meaning: `track index of each stop; ${unused}`
              },
              stopDurations: {
                view: output.stops.durations,
                role: 'intermediate',
                keySpace: 'stop',
                meaning: `stop duration in s; ${unused}`
              },
              stopCentroids: {
                view: output.stops.centroids,
                role: 'intermediate',
                keySpace: 'stop',
                meaning: `mean stop position in UTM 18N m; ${unused}`
              },
              stopCount: {view: output.stops.status.count, role: 'intermediate'},
              stopOverflow: {view: output.stops.status.overflow, role: 'intermediate'},
              measuredTracks: {
                view: output.measuredTracks,
                role: 'output',
                geometry: projectedGeometry('lines', {
                  offsetsCapture: 'trackOffsets',
                  valuesCapture: 'averageSpeeds'
                })
              }
            };
          }
        })
    };
  }
});
