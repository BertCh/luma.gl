// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.track-similarity';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUTrackSimilarity',
  title: 'Hausdorff and Frechet distances between harbour tracks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-ais',
    source: 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
    licence: 'CC0 1.0'
  },
  load: async () => {
    const [{runS3Lens, projectedMetadata}, {runExample}, data] = await Promise.all([
      import('./s3-lens-run'),
      import('./track-similarity.example'),
      import('./s3-lens-data').then(module => module.loadS3NewYorkHarborTracks())
    ]);
    // Every pair of two distinct tracks (a < b), in row-major order: derived from the track count only.
    const pairA: number[] = [];
    const pairB: number[] = [];
    for (let first = 0; first < data.trackCount; first++) {
      for (let second = first + 1; second < data.trackCount; second++) {
        pairA.push(first);
        pairB.push(second);
      }
    }
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUTrackSimilarity',
          datasets: [data.id],
          retain: ['hausdorff', 'frechet', 'maximumDistance', 'pairStatus'],
          keySpaces: [
            {id: 'pair', label: 'track pair', size: pairA.length, indexSpace: '1d'},
            {id: 'track', label: 'vessel track', size: data.trackCount, indexSpace: '1d'}
          ],
          choices: {
            spatialContext: 'native planar (UTM 18N metres)',
            densify: 0.25,
            maximumFrechetVertexCount: 512
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const tracks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('trackOffsets', data.trackOffsets, 'uint32')
            };
            const pairViewA = uploads.view('pairA', Uint32Array.from(pairA), 'uint32');
            const pairViewB = uploads.view('pairB', Uint32Array.from(pairB), 'uint32');
            const output = await runExample(graph, {
              tracks,
              pairA: pairViewA,
              pairB: pairViewB
            });
            // All 66 pairs are written, so the table rows are exactly the GPU output.
            return {
              pairA: {
                view: pairViewA,
                role: 'input',
                keySpace: 'pair',
                meaning: 'first track index of the pair'
              },
              pairB: {
                view: pairViewB,
                role: 'input',
                keySpace: 'pair',
                meaning: 'second track index of the pair'
              },
              hausdorff: {
                view: output.hausdorff,
                role: 'output',
                keySpace: 'pair',
                meaning: 'symmetric Hausdorff distance in m (vertex to segment)'
              },
              frechet: {
                view: output.frechet,
                role: 'output',
                keySpace: 'pair',
                meaning: 'discrete Frechet distance in m'
              },
              maximumDistance: {
                view: output.maximumDistance,
                role: 'output',
                keySpace: 'pair',
                meaning: 'largest vertex-to-vertex distance in m'
              },
              pairStatus: {
                view: output.pairStatus,
                role: 'intermediate',
                keySpace: 'pair',
                meaning: 'status bits: 1 empty track, 2 Frechet cap exceeded, 4 invalid pair'
              }
            };
          }
        })
    };
  }
});
