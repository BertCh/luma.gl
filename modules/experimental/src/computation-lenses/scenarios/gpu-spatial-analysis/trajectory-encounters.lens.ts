// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.trajectory-encounters';
const DISTANCE_METRES = 200;
const CELL_SIZE_METRES = 250;
const CLOCK_START_SECONDS = 0;
const CLOCK_STEP_SECONDS = 60;
const BUCKET_COUNT = 120;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrajectoryEncounters',
  title: 'Vessel pairs within 200 m on a 60 s clock',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [{runS3Lens, projectedMetadata}, {runExample}, {getSnappedExtent}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./trajectory-encounters.example'),
        import('./trajectory-lens-support'),
        import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
      ]);
    const {getGPUTrajectoryClockParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    // At most one pair per two distinct tracks (track < partner), so this capacity cannot overflow.
    const pairCapacity = (data.trackCount * (data.trackCount - 1)) / 2;
    const bounds = getSnappedExtent(data.positionsProjected, CELL_SIZE_METRES);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrajectoryEncounters',
          datasets: [data.id],
          retain: [
            'pairTracks',
            'pairCount',
            'pairOverflow',
            'pairPartners',
            'minimumDistances',
            'firstBuckets',
            'bucketCounts',
            'candidateOverflow'
          ],
          keySpaces: [
            {id: 'pair', label: 'track pair', size: pairCapacity, indexSpace: '1d'},
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 18N metres)',
            bucketCount: BUCKET_COUNT,
            cellSize: CELL_SIZE_METRES,
            bounds: bounds.join(', '),
            distance: DISTANCE_METRES,
            hitCapacity: 1 << 16
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              timestamps: uploads.view('timestamps', data.timestamps, 'float32'),
              trackOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            const clock = uploads.view(
              'clock',
              getGPUTrajectoryClockParameterValues({
                start: CLOCK_START_SECONDS,
                step: CLOCK_STEP_SECONDS
              }),
              'float32'
            );
            const distance = uploads.view('distance', Float32Array.of(DISTANCE_METRES), 'float32');
            const output = await runExample(graph, {
              tracks,
              clock,
              distance,
              bucketCount: BUCKET_COUNT,
              bounds,
              pairCapacity
            });
            const unused = 'rows past the pair count are unused';
            // Pairs are a table (no per-pair geometry exists on the GPU): the first rows are the
            // GPU-written pairs in (track, partner) order.
            return {
              pairTracks: {
                view: output.pairs.ids,
                role: 'output',
                keySpace: 'pair',
                meaning: `lower track index of the pair; ${unused}`
              },
              pairPartners: {
                view: output.pairs.partners,
                role: 'output',
                keySpace: 'pair',
                meaning: `higher track index of the pair; ${unused}`
              },
              minimumDistances: {
                view: output.pairs.minimumDistances,
                role: 'output',
                keySpace: 'pair',
                meaning: `smallest clock-sample distance in m; ${unused}`
              },
              firstBuckets: {
                view: output.pairs.firstBuckets,
                role: 'output',
                keySpace: 'pair',
                meaning: `first 60 s bucket within 200 m; ${unused}`
              },
              bucketCounts: {
                view: output.pairs.bucketCounts,
                role: 'output',
                keySpace: 'pair',
                meaning: `buckets within 200 m; ${unused}`
              },
              pairCount: {view: output.pairs.status.count, role: 'intermediate'},
              pairOverflow: {view: output.pairs.status.overflow, role: 'intermediate'},
              candidateOverflow: {view: output.status.candidateOverflow, role: 'intermediate'}
            };
          }
        })
    };
  }
});
