// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology';
import {createTerrainFlowFillNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-passes';
import {createTerrainFlowSpillFillNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-fill';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Which depression fill a GPU spec runs on the device. */
export type FillImplementation = 'spill' | 'relaxation';

/** A compiled single-fill graph with its buffers, reusable across frames. */
export type FillRunner = {
  /** Runs the compiled graph once and reads the filled surface and the status words. */
  run: (epsilon?: number) => Promise<{filled: Float32Array; converged: number; iterations: number}>;
  /** Submits one frame without reading anything back. */
  submit: (epsilon?: number) => void;
  destroy: () => void;
};

/**
 * Builds a graph holding only the depression fill (the spill graph fill or the previous tiled
 * relaxation) over `elevation`, for GPU specs that compare the two without the rest of
 * GPUTerrainFlow. `fillEpsilon` is a per-frame parameter, as in GPUTerrainFlow.
 */
export function createFillRunner(
  device: Device,
  implementation: FillImplementation,
  elevation: Float32Array,
  width: number,
  height: number
): FillRunner {
  const cellCount = width * height;
  const grid = makeGPURasterMetadata({width, height});
  const elevationBuffer = createInputBuffer(device, elevation);
  const filledBuffer = createOutputBuffer(device, cellCount);
  const convergedBuffer = createOutputBuffer(device, 1);
  const iterationBuffer = createOutputBuffer(device, 1);
  const settings = new GPUParameterBuffer(device, {
    id: 'fill-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues({}, grid)
  });
  const graph = new GPUCommandGraph(device, {id: `fill-${implementation}`});
  const producer: GPUCommandNodeProducer = {
    getCommandNodes: nodeGraph => {
      const props = {
        id: 'fill',
        width,
        height,
        cellSizeMode: 'uniform' as const,
        maxIterations: 512,
        elevation: importGraphBuffer(nodeGraph, 'elevation', elevationBuffer, 'float32', cellCount),
        filled: importGraphBuffer(nodeGraph, 'filled', filledBuffer, 'float32', cellCount),
        settings: settings.importToGraph(nodeGraph),
        converged: importGraphBuffer(nodeGraph, 'converged', convergedBuffer, 'uint32', 1),
        iterationCount: importGraphBuffer(nodeGraph, 'iterations', iterationBuffer, 'uint32', 1)
      };
      return (
        implementation === 'spill'
          ? createTerrainFlowSpillFillNodes(nodeGraph, props)
          : createTerrainFlowFillNodes(nodeGraph, props)
      ).nodes;
    }
  };
  graph.add(producer);
  const compiled = graph.compile();
  const submit = (epsilon = 0) => {
    settings.write(Float32Array.from([epsilon]), 4);
    submitGraph(device, compiled, undefined);
  };
  return {
    submit,
    run: async epsilon => {
      submit(epsilon);
      return {
        filled: Float32Array.from(await readFloat32(filledBuffer, cellCount)),
        converged: (await readUint32(convergedBuffer, 1))[0],
        iterations: (await readUint32(iterationBuffer, 1))[0]
      };
    },
    destroy: () => {
      compiled.destroy();
      settings.destroy();
      for (const buffer of [elevationBuffer, filledBuffer, convergedBuffer, iterationBuffer]) {
        buffer.destroy();
      }
    }
  };
}
