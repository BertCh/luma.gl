// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.trajectory-closest-approach';
const DISTANCE_METRES = 200;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrajectoryClosestApproach',
  title: 'Exact vessel encounters within 200 m',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [{runS3Lens, projectedMetadata}, {runExample}, data] = await Promise.all([
      import('./s3-lens-run'),
      import('./trajectory-closest-approach.example'),
      import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
    ]);
    // At most one pair per two distinct tracks (track < partner), so this capacity cannot overflow.
    const pairCapacity = (data.trackCount * (data.trackCount - 1)) / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrajectoryClosestApproach',
          datasets: [data.id],
          retain: [
            'pairTracks',
            'pairCount',
            'pairOverflow',
            'pairPartners',
            'minimumDistances',
            'closestTimes',
            'firstTimes',
            'durations',
            'entryOverflow',
            'hitOverflow'
          ],
          keySpaces: [
            {id: 'pair', label: 'track pair', size: pairCapacity, indexSpace: '1d'},
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'}
          ],
          choices: {
            cellSize: 250,
            timeCellSize: 120,
            distance: DISTANCE_METRES,
            entryCapacity: 1 << 17,
            hitCapacity: 1 << 16
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              timestamps: uploads.view('timestamps', data.timestamps, 'float32'),
              trackOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            const distance = uploads.view('distance', Float32Array.of(DISTANCE_METRES), 'float32');
            const output = await runExample(graph, {tracks, distance, pairCapacity});
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
                meaning: `closest approach in m; ${unused}`
              },
              closestTimes: {
                view: output.pairs.closestTimes,
                role: 'output',
                keySpace: 'pair',
                meaning: `closest-approach time in s since the lower track's first fix; ${unused}`
              },
              firstTimes: {
                view: output.pairs.firstTimes,
                role: 'output',
                keySpace: 'pair',
                meaning: `first time within 200 m in s since the lower track's first fix; ${unused}`
              },
              durations: {
                view: output.pairs.durations,
                role: 'output',
                keySpace: 'pair',
                meaning: `time within 200 m in s; ${unused}`
              },
              pairCount: {view: output.pairs.status.count, role: 'intermediate'},
              pairOverflow: {view: output.pairs.status.overflow, role: 'intermediate'},
              entryOverflow: {view: output.status.entryOverflow, role: 'intermediate'},
              hitOverflow: {view: output.status.hitOverflow, role: 'intermediate'}
            };
          }
        })
    };
  }
});
