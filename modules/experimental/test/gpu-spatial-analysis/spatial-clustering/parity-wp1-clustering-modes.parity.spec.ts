// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUSpatialClustering,
  getGPUSpatialClusteringParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import fixture from '../../oracles/fixtures/parity-wp1-clustering-modes.json';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of fixture.cases) {
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    for (const denseBoxShortcut of scene.geographic ? [false] : [false, true]) {
      for (const capacity of [1, 4]) {
        it(`GPUSpatialClustering summaries ${scene.name}/${sumOrder}/${denseBoxShortcut}/${capacity}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = new GraphRig(device, 'parity-cluster-modes');
          const count = scene.labels.length;
          const labels = rig.output('uint32', count);
          const coreFlags = rig.output('uint32', count);
          const rootRows = rig.output('uint32', count);
          const clusterCount = rig.output('uint32', 1);
          const ids = rig.output('uint32', capacity);
          const stored = rig.output('uint32', 1);
          const overflow = rig.output('uint32', 1);
          const sizes = rig.output('uint32', capacity);
          const centroids = rig.output('float32', 2 * capacity);
          const draw = rig.output('uint32', 1);
          try {
            rig.run(
              new GPUSpatialClustering({
                positions: rig.input(new Float32Array(scene.positions), 'float32x2', count),
                ...(scene.geographic
                  ? {
                      spatialContext: {
                        coordinateSpace: 'longitude-latitude',
                        metric: 'great-circle',
                        sphereRadius: 6371008.8
                      }
                    }
                  : {}),
                parameters: rig.input(
                  getGPUSpatialClusteringParameterValues({
                    bounds: [scene.bounds[0], scene.bounds[1], scene.bounds[2], scene.bounds[3]],
                    epsilon: scene.epsilon,
                    minimumPoints: scene.minimumPoints
                  }),
                  'float32'
                ),
                sourceIds: rig.input(
                  Uint32Array.from({length: count}, (_, index) => 100 + index),
                  'uint32'
                ),
                gridSize: [32, 32],
                sumOrder,
                denseBoxShortcut,
                output: {
                  labels: labels.view,
                  coreFlags: coreFlags.view,
                  rootRows: rootRows.view,
                  clusterCount: clusterCount.view,
                  clusters: {ids: ids.view, status: {count: stored.view, overflow: overflow.view}},
                  clusterSizes: sizes.view,
                  clusterCentroids: importGraphBuffer(
                    rig.graph,
                    'centroids-pairs',
                    centroids.buffer,
                    'float32x2',
                    capacity
                  ),
                  drawInstanceCount: draw.view
                }
              })
            );
            const compare = (metric: string, actual: number[], expected: number[], exact = true) =>
              compareReference(
                'GPUSpatialClustering',
                {...scene, name: `${scene.name}/${sumOrder}/${denseBoxShortcut}/${capacity}`},
                metric,
                actual,
                expected,
                exact
              );
            compare('labels', await labels.readUint32(), scene.labels);
            compare('core-flags', await coreFlags.readUint32(), scene.coreMask);
            compare('root-rows', await rootRows.readUint32(), scene.rootRows);
            compare('cluster-count', await clusterCount.readUint32(), [scene.roots.length]);
            compare('stored-count', await stored.readUint32(), [
              Math.min(capacity, scene.roots.length)
            ]);
            compare('overflow', await overflow.readUint32(), [
              Number(capacity < scene.roots.length)
            ]);
            compare('draw-count', await draw.readUint32(), [
              Math.min(capacity, scene.roots.length)
            ]);
            compare(
              'source-ids',
              (await ids.readUint32()).slice(0, Math.min(capacity, scene.roots.length)),
              scene.roots.slice(0, capacity).map(root => root + 100)
            );
            compare(
              'sizes',
              await sizes.readUint32(),
              Array.from({length: capacity}, (_, index) => scene.sizes[index] ?? 0)
            );
            compare(
              'centroids',
              await centroids.readFloat32(),
              Array.from({length: capacity}, (_, index) => scene.centroids[index] ?? [0, 0]).flat(),
              false
            );
          } finally {
            rig.destroy();
          }
        });
      }
    }
  }
}
