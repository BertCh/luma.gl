// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {log} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-encounters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUTrajectoryEncounters warns on the derived maximum distance in context units', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
  let serial = 0;
  const view = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `encounters-warning-${serial++}`, format, length);
  const createEncounters = (cellSize: number, spatialContext?: GPUSpatialContext) =>
    new GPUTrajectoryEncounters({
      spatialContext,
      trackCount: 2,
      bucketCount: 1,
      hitCapacity: 2,
      samples: view('float32x2', 2),
      distance: view('float32', 1),
      cellSize,
      bounds: [0, 0, 0.001, 0.001],
      output: {
        pairs: {
          ids: view('uint32', 1),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        status: {candidateOverflow: view('uint32', 1)}
      }
    });
  try {
    createEncounters(0.00001);
    createEncounters(0.00001, {coordinateSpace: 'planar', metric: 'native', units: 'meters'});
    const geographic = {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    } as const;
    createEncounters(0.001, geographic);
    expect(warning).not.toHaveBeenCalled();
    const encounters = createEncounters(0.00001, geographic);
    expect(encounters.maximumDistance).toBeGreaterThan(1);
    expect(encounters.maximumDistance).toBeLessThan(2);
    expect(warning).toHaveBeenCalledExactlyOnceWith(
      'GPUTrajectoryEncounters distance is too short for float32 lon/lat; use createGPULocalFrame'
    );
  } finally {
    warning.mockRestore();
    device.destroy();
  }
});
