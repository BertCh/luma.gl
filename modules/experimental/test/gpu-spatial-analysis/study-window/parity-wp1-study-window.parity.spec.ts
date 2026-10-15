// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {it} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import fixture from '../../oracles/fixtures/parity-wp1-study-window.json';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of fixture.cases) {
  for (const fallback of [false, true]) {
    it(`GPUStudyWindowPreparation GEOS ${scene.name} fallback=${fallback}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      const summary = rig.output('float32', 8);
      const preparation = new GPUStudyWindowPreparation({
        positions: rig.input(
          new Float32Array(scene.positions),
          'float32x2',
          scene.positions.length / 2
        ),
        featureOffsets: rig.input(new Uint32Array(scene.featureOffsets), 'uint32'),
        polygonOffsets: rig.input(new Uint32Array(scene.polygonOffsets), 'uint32'),
        ringOffsets: rig.input(new Uint32Array(scene.ringOffsets), 'uint32'),
        samplingGrid: [scene.samplingGrid[0], scene.samplingGrid[1]],
        ...(fallback ? {crossingCapacity: 2, slabEntryCapacity: 2} : {}),
        output: {summary}
      });
      try {
        rig.graph.add(preparation);
        const views = preparation.getViews(rig.graph);
        const copy = <Format extends 'float32' | 'uint32'>(
          source: GraphDataView<Format>,
          format: Format
        ) => {
          const destination = rig.output(format, source.length);
          rig.graph.add(
            createWGSLKernelNode(rig.graph, {
              id: `copy-${source.buffer.id}`,
              operation: 'ParityStudyWindow',
              invocationCount: source.length,
              bindings: [
                {name: 'source', view: source, type: 'u32', access: 'read'},
                {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
              ],
              body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
            })
          );
          return destination;
        };
        const edges = copy(views.edges, 'float32');
        const cells = copy(views.sampleCells, 'uint32');
        const count = copy(views.sampleCellCount, 'uint32');
        rig.run();
        const compare = (metric: string, actual: number[], expected: number[], exact = false) =>
          compareReference(
            'GPUStudyWindowPreparation',
            {...scene, name: `${scene.name}/fallback=${fallback}`},
            metric,
            actual,
            expected,
            exact
          );
        const summaryResult = await rig.readFloat(summary);
        compare('bounds', summaryResult.slice(0, 4), scene.summary.slice(0, 4), true);
        compare('area', [summaryResult[4]], [scene.summary[4]]);
        compare('perimeter', [summaryResult[5]], [scene.summary[5]]);
        compare('edge-count-flags', summaryResult.slice(6), scene.summary.slice(6), true);
        compare('every-directed-edge', await rig.readFloat(edges), scene.edges, true);
        compare('sample-count', await rig.readUint(count), [scene.sampleCells.length], true);
        const expectedCells = [
          ...scene.sampleCells,
          ...new Array(cells.length - scene.sampleCells.length).fill(0)
        ];
        compare(
          'every-sample-id-and-boundary-mask',
          await rig.readUint(cells),
          expectedCells,
          true
        );
      } finally {
        rig.destroy();
        preparation.destroy();
      }
    });
  }
}
