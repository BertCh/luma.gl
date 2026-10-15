// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.segment-ring-assembly';
const RING_CAPACITY = 64;
/** Stride that reorders the segments; coprime to the segment count (3184 = 2^4 x 199). */
const SHUFFLE_STRIDE = 7;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUSegmentRingAssembly',
  title: 'Community-area edges chained into the outline of the coverage',
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
        import('./segment-ring-assembly.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    // Every ring edge of every area as a directed segment, then reordered by a fixed stride so the
    // assembly has to find the neighbours itself. The data rings are closed (last vertex repeats).
    const ordered: number[] = [];
    for (let ring = 0; ring < data.ringOffsets.length - 1; ring++) {
      const start = data.ringOffsets[ring] as number;
      const end = data.ringOffsets[ring + 1] as number;
      for (let vertex = start; vertex < end - 1; vertex++) {
        ordered.push(
          data.positionsProjected[2 * vertex] as number,
          data.positionsProjected[2 * vertex + 1] as number,
          data.positionsProjected[2 * vertex + 2] as number,
          data.positionsProjected[2 * vertex + 3] as number
        );
      }
    }
    const segmentCount = ordered.length / 4;
    const segments = new Float32Array(ordered.length);
    for (let row = 0; row < segmentCount; row++) {
      const source = (row * SHUFFLE_STRIDE) % segmentCount;
      segments.set(ordered.slice(4 * source, 4 * source + 4), 4 * row);
    }
    const vertexCount = data.positionsProjected.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUSegmentRingAssembly',
          datasets: [data.id],
          retain: [
            'ringOffsets',
            'ringPositions',
            'ringAreas',
            'ringIsHole',
            'ringCount',
            'overflow',
            'openSegmentCount'
          ],
          keySpaces: [
            {id: 'ring', label: 'assembled ring', size: RING_CAPACITY, indexSpace: '1d'},
            {id: 'areaVertex', label: 'area ring vertex', size: vertexCount, indexSpace: '1d'},
            {
              id: 'areaRing',
              label: 'area ring',
              size: data.ringOffsets.length - 1,
              indexSpace: '1d'
            },
            {
              id: 'ringVertex',
              label: 'assembled ring vertex',
              size: segmentCount + RING_CAPACITY,
              indexSpace: '1d'
            }
          ],
          choices: {
            vertexTolerance: 0.01,
            interiorSide: 'right',
            normalizeWinding: true,
            pairedOpposites: false,
            cancelOpposingSegments: true,
            splitTouchingRings: true
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const areas = uploads.view('positions', data.positionsProjected, 'float32x2');
            const areaRingOffsets = uploads.view('areaRingOffsets', data.ringOffsets, 'uint32');
            const segmentRows = uploads.view('segments', segments, 'float32x4');
            const output = await runExample(graph, {
              segments: segmentRows,
              ringCapacity: RING_CAPACITY
            });
            return {
              positions: {
                view: areas,
                role: 'input',
                keySpace: 'areaVertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'areaRingOffsets'}),
                meaning:
                  'the 17 community areas whose edges (3184 directed segments, shuffled by a fixed stride) are the input'
              },
              areaRingOffsets: {view: areaRingOffsets, role: 'intermediate', keySpace: 'areaRing'},
              ringAreas: {
                view: output.ringAreas,
                role: 'intermediate',
                keySpace: 'ring',
                meaning: 'signed shoelace area of each ring in m² (counter-clockwise positive)'
              },
              ringIsHole: {
                view: output.ringIsHole,
                role: 'intermediate',
                keySpace: 'ring',
                meaning: '1 when the ring is a hole, 0 for a shell'
              },
              ringCount: {
                view: output.status.count,
                role: 'intermediate',
                meaning: 'rings written'
              },
              openSegmentCount: {
                view: output.status.openSegmentCount,
                role: 'intermediate',
                meaning: 'segments on no closed ring'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              ringOffsets: {view: output.ringOffsets, role: 'intermediate'},
              ringPositions: {
                view: output.positions,
                role: 'output',
                keySpace: 'ringVertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning:
                  'closed rings after shared borders cancel, UTM 16N metres; offsets past the ring count repeat the last one, so unused slots are empty'
              }
            };
          }
        })
    };
  }
});
