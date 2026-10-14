// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUVertexSnap} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;

it('GPUVertexSnap validates its tolerance distance frame', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {positions: createTransientView(graph, 'snapped', 'float32x2', 4)}
  };
  expect(() => new GPUVertexSnap({...props, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(() => new GPUVertexSnap({...props, spatialContext: DISTANCE_FRAME})).not.toThrow();
  device.destroy();
});
