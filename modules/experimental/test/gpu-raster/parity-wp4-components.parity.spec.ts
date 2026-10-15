// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterConnectedComponents,
  GPURasterDenseComponents
} from '@luma.gl/experimental/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  componentCases,
  decodeComponentRuns,
  type ComponentCase
} from '../oracles/parity-wp4-round2-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

type Storage = {buffer: Buffer; view: GraphDataView<'uint32'>; length: number};

function createStorage(device: Device, graph: GPUCommandGraph, buffers: Buffer[]) {
  return (name: string, content: Uint32Array | number): Storage => {
    const length = typeof content === 'number' ? content : content.length;
    const buffer =
      typeof content === 'number'
        ? createOutputBuffer(device, length)
        : createInputBuffer(device, content);
    buffers.push(buffer);
    return {buffer, length, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
}

async function compareOutput(
  className: string,
  scene: ComponentCase,
  metric: string,
  storage: Storage,
  expected: ArrayLike<number>,
  suffix = ''
) {
  compareRasterParity(
    className,
    `${scene.name}${suffix}`,
    scene.reference,
    metric,
    await readUint32(storage.buffer, storage.length),
    expected,
    'E'
  );
}

for (const scene of componentCases) {
  it(`GPURasterConnectedComponents and dense pipeline: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device, {id: `parity-wp4-components-${scene.name}`});
    const buffers: Buffer[] = [];
    const create = createStorage(device, graph, buffers);
    const length = scene.width * scene.height;
    const source = create('source', decodeComponentRuns(scene.values, length));
    const validity = create('validity', decodeComponentRuns(scene.validity, length));
    const sparse = create('sparse', length);
    const sparseValidity = create('sparse-validity', length);
    const converged = create('converged', 1);
    const iterationCount = create('iteration-count', 1);
    const dense = create('dense', length);
    const denseValidity = create('dense-validity', length);
    const componentCount = create('component-count', 1);
    const requiredCount = create('required-count', 1);
    const overflow = create('overflow', 1);
    new GPURasterConnectedComponents({
      width: scene.width,
      height: scene.height,
      connectivity: scene.connectivity === 4 ? 4 : 8,
      maximumIterations: 32,
      input: {
        id: 'foreground',
        format: 'uint32',
        storage: {kind: 'buffer', values: source.view},
        validity: validity.view,
        noDataValue: scene.noDataValue
      },
      output: sparse.view,
      outputValidity: sparseValidity.view,
      converged: converged.view,
      iterationCount: iterationCount.view
    }).addToGraph(graph);
    new GPURasterDenseComponents({
      width: scene.width,
      height: scene.height,
      input: sparse.view,
      inputValidity: sparseValidity.view,
      converged: converged.view,
      output: dense.view,
      outputValidity: denseValidity.view,
      componentCount: componentCount.view,
      requiredComponentCount: requiredCount.view,
      overflow: overflow.view
    }).addToGraph(graph);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const expectedValidity = decodeComponentRuns(scene.expectedValidity, length);
      await compareOutput('GPURasterConnectedComponents', scene, 'converged', converged, [1]);
      await compareOutput(
        'GPURasterConnectedComponents',
        scene,
        'all-canonical-minimum-roots',
        sparse,
        decodeComponentRuns(scene.sparse, length)
      );
      await compareOutput(
        'GPURasterConnectedComponents',
        scene,
        'all-observation-validity',
        sparseValidity,
        expectedValidity
      );
      const rounds = await readUint32(iterationCount.buffer, 1);
      compareRasterParity(
        'GPURasterConnectedComponents',
        scene.name,
        'documented bounded stabilization',
        'active-rounds-in-budget',
        [Number(rounds[0] >= 1 && rounds[0] <= 32)],
        [1],
        'E'
      );
      await compareOutput(
        'GPURasterDenseComponents',
        scene,
        'all-dense-labels',
        dense,
        decodeComponentRuns(scene.dense, length)
      );
      await compareOutput(
        'GPURasterDenseComponents',
        scene,
        'all-observation-validity',
        denseValidity,
        expectedValidity
      );
      await compareOutput('GPURasterDenseComponents', scene, 'component-count', componentCount, [
        scene.componentCount
      ]);
      await compareOutput('GPURasterDenseComponents', scene, 'required-count', requiredCount, [
        scene.componentCount
      ]);
      await compareOutput('GPURasterDenseComponents', scene, 'overflow', overflow, [0]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }, 120_000);

  // Direct oracle roots isolate dense relabeling from upstream component failures.
  for (const capacity of [
    ...new Set([0, Math.floor(scene.componentCount / 2), scene.componentCount])
  ]) {
    it(`GPURasterDenseComponents independent roots: ${scene.name}, capacity ${capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device, {id: `parity-wp4-dense-${scene.name}-${capacity}`});
      const buffers: Buffer[] = [];
      const create = createStorage(device, graph, buffers);
      const length = scene.width * scene.height;
      const expectedDense = decodeComponentRuns(scene.dense, length);
      const expectedValidity = decodeComponentRuns(scene.expectedValidity, length);
      for (let index = 0; index < length; index++) {
        if (expectedDense[index] > capacity) {
          expectedDense[index] = 0;
          expectedValidity[index] = 0;
        }
      }
      const sparse = create('oracle-roots', decodeComponentRuns(scene.sparse, length));
      const validity = create(
        'oracle-validity',
        decodeComponentRuns(scene.expectedValidity, length)
      );
      const converged = create('oracle-converged', new Uint32Array([1]));
      const dense = create('dense', length);
      const outputValidity = create('output-validity', length);
      const componentCount = create('component-count', 1);
      const requiredCount = create('required-count', 1);
      const overflow = create('overflow', 1);
      new GPURasterDenseComponents({
        width: scene.width,
        height: scene.height,
        capacity,
        input: sparse.view,
        inputValidity: validity.view,
        converged: converged.view,
        output: dense.view,
        outputValidity: outputValidity.view,
        componentCount: componentCount.view,
        requiredComponentCount: requiredCount.view,
        overflow: overflow.view
      }).addToGraph(graph);
      const compiled = graph.compile();
      const suffix = `/oracle-roots/capacity-${capacity}`;
      try {
        submitGraph(device, compiled, undefined);
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'all-bounded-dense-labels',
          dense,
          expectedDense,
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'all-bounded-validity',
          outputValidity,
          expectedValidity,
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'bounded-count',
          componentCount,
          [capacity],
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'required-count',
          requiredCount,
          [scene.componentCount],
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'overflow',
          overflow,
          [Number(scene.componentCount > capacity)],
          suffix
        );
        // Re-encoding after upstream failure must clear previous successful results.
        converged.buffer.write(new Uint32Array([0]));
        submitGraph(device, compiled, undefined);
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'nonconverged-labels',
          dense,
          new Uint32Array(length),
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'nonconverged-validity',
          outputValidity,
          new Uint32Array(length),
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'nonconverged-count',
          componentCount,
          [0],
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'nonconverged-required-count',
          requiredCount,
          [0],
          suffix
        );
        await compareOutput(
          'GPURasterDenseComponents',
          scene,
          'nonconverged-overflow',
          overflow,
          [0],
          suffix
        );
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }, 120_000);
  }
}
