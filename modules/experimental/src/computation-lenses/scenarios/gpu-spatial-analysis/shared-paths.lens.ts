// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {assert} from '@luma.gl/core';

const SCENARIO_ID = 'gpu-spatial-analysis.shared-paths';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUSharedPaths',
  title: 'Shared boundaries of West Town and neighboring Chicago areas',
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
        import('./shared-paths.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    // Each feature is exactly one polygon and one explicitly closed boundary ring.
    assert(data.featureOffsets.every((offset, index) => offset === index));
    assert(data.polygonOffsets.every((offset, index) => offset === index));
    const areaIndex = data.names.indexOf('area 24');
    assert(areaIndex >= 0);
    const leftPositions = data.positionsProjected.slice(
      data.ringOffsets[areaIndex] * 2,
      data.ringOffsets[areaIndex + 1] * 2
    );
    const rightPositions: number[] = [];
    const rightOffsets = [0];
    for (let area = 0; area < data.featureCount; area++) {
      const start = data.ringOffsets[area];
      const end = data.ringOffsets[area + 1];
      // A line boundary must include its closing edge.
      assert(data.positionsProjected[start * 2] === data.positionsProjected[end * 2 - 2]);
      assert(data.positionsProjected[start * 2 + 1] === data.positionsProjected[end * 2 - 1]);
      if (area === areaIndex) continue;
      rightPositions.push(...data.positionsProjected.slice(start * 2, end * 2));
      rightOffsets.push(rightPositions.length / 2);
    }
    const intersectionCapacity = 4096;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUSharedPaths',
          datasets: [data.id],
          retain: [
            'sharedPaths',
            'runOffsets',
            'leftLineIds',
            'rightLineIds',
            'forward',
            'count',
            'overflow',
            'uncertainCount'
          ],
          keySpaces: [
            {id: 'run', label: 'run', size: intersectionCapacity, indexSpace: '1d'},
            {
              id: 'leftVertex',
              label: 'leftVertex',
              size: leftPositions.length / 2,
              indexSpace: '1d'
            },
            {
              id: 'rightVertex',
              label: 'rightVertex',
              size: rightPositions.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {intersectionCapacity},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const left = {
              positions: uploads.view('leftPositions', leftPositions, 'float32x2'),
              lineOffsets: uploads.view(
                'leftOffsets',
                new Uint32Array([0, leftPositions.length / 2]),
                'uint32'
              )
            };
            const right = {
              positions: uploads.view(
                'rightPositions',
                Float32Array.from(rightPositions),
                'float32x2'
              ),
              lineOffsets: uploads.view('rightOffsets', Uint32Array.from(rightOffsets), 'uint32')
            };
            const output = await runExample(graph, {left, right, intersectionCapacity});
            return {
              leftBoundaries: {
                view: left.positions,
                role: 'input',
                keySpace: 'leftVertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'leftOffsets'}),
                meaning: 'West Town, community area 24 boundary'
              },
              rightBoundaries: {
                view: right.positions,
                role: 'input',
                keySpace: 'rightVertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'rightOffsets'}),
                meaning: 'other 16 community-area boundaries, in dataset order'
              },
              leftOffsets: {view: left.lineOffsets, role: 'intermediate'},
              rightOffsets: {view: right.lineOffsets, role: 'intermediate'},
              runOffsets: {view: output.runOffsets, role: 'intermediate'},
              leftLineIds: {
                view: output.runs.leftLineIds,
                role: 'intermediate',
                keySpace: 'run',
                meaning: 'left boundary index (0 is West Town); unused slots are 0xffffffff'
              },
              rightLineIds: {
                view: output.runs.rightLineIds,
                role: 'intermediate',
                keySpace: 'run',
                meaning: 'right boundary index in the other 16 areas; unused slots are 0xffffffff'
              },
              forward: {
                view: output.runs.forward,
                role: 'intermediate',
                keySpace: 'run',
                meaning:
                  'relative direction code (unitless): 1 same, 0 opposite; only count runs are valid'
              },
              count: {view: output.status.count, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              uncertainCount: {view: output.status.uncertainCount, role: 'intermediate'},
              sharedPaths: {
                view: output.sharedPaths,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'runOffsets'})
              }
            };
          }
        })
    };
  }
});
