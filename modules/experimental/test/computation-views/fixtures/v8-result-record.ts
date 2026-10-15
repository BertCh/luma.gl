// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';

/** Capture builder for the V-8 geometry fixtures. */
export function makeV8Capture(
  name: string,
  values: number[],
  metadata: Partial<GPUComputationCapture> = {}
): GPUComputationCapture {
  return {
    id: `${name}-id`,
    name,
    values,
    elementType: 'f32',
    provenance: 'gpu-readback',
    ...metadata
  };
}

/** Final polygons, raster, and recorded/unrecorded graph layout cases. */
export function makeV8ResultRecord(
  kind: 'polygons' | 'raster' | 'graph' | 'layout' = 'polygons'
): LensRecord {
  let snapshots: GPUComputationCapture[];
  if (kind === 'polygons') {
    snapshots = [
      makeV8Capture('input', [-88, 41, -87, 41, -87, 42, -88, 42, -86, 41, -86, 42], {
        components: 2,
        role: 'input',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      }),
      makeV8Capture('polygons', [-88, 41, -87, 41, -87, 42, -86, 41, -85, 41, -85, 42], {
        components: 2,
        iteration: 3,
        role: 'output',
        geometry: {
          kind: 'polygons',
          coordinateModel: 'geographic',
          offsetsCapture: 'offsets-id',
          valuesCapture: 'values'
        }
      }),
      makeV8Capture('offsets', [0, 3, 6]),
      makeV8Capture('values', [1, 5])
    ];
  } else if (kind === 'raster') {
    snapshots = [
      makeV8Capture(
        'raster',
        Array.from({length: 20000}, (_, index) => index),
        {
          role: 'output',
          geometry: {
            kind: 'raster',
            shape: [200, 100],
            cellSize: 10,
            unit: 'm',
            coordinateModel: 'projected'
          }
        }
      )
    ];
  } else {
    const nodeCount = 40;
    snapshots = [
      makeV8Capture('graph', [], {
        role: 'output',
        geometry: {
          kind: 'graph',
          offsetsCapture: 'offsets',
          targetsCapture: 'targets-id',
          layoutCapture: kind === 'layout' ? 'layout-id' : undefined,
          coordinateModel: 'projected'
        }
      }),
      makeV8Capture(
        'offsets',
        Array.from({length: nodeCount + 1}, (_, index) => index)
      ),
      makeV8Capture(
        'targets',
        Array.from({length: nodeCount}, (_, index) => (index + 1) % nodeCount)
      ),
      ...(kind === 'layout'
        ? [
            makeV8Capture(
              'layout',
              Array.from({length: nodeCount}, (_, index) => [
                index % 8,
                Math.floor(index / 8)
              ]).flat(),
              {components: 2}
            )
          ]
        : [])
    ];
  }
  return {
    schema: 'luma.computation/1',
    id: 'gpu-core.geometry',
    runId: 'v8-result-run',
    createdAt: '2026-10-14T00:00:00.000Z',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['gpu-core'],
    graphs: [],
    axes: {axes: [{id: 'entity', label: 'entity', kind: 'data'}], keySpaces: [], relations: []},
    refs: {formulas: [], sources: []},
    decisions: [],
    choices: [],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'v8-result-run',
      snapshots,
      stamps: [],
      replays: []
    }
  };
}
