// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUCostDistance,
  GPUCostDistancePath,
  getGPUCostDistanceParameterValues
} from '../../../src/gpu-raster/cost-distance';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-cost-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type CostCase = {
  name: string;
  mode: 'uniform' | 'web-mercator' | 'geographic';
  rowDirection: 'north' | 'south';
  width: number;
  height: number;
  cellSize: [number, number];
  origin: [number, number];
  raw: number[];
  validity: number[];
  calibration: number[];
  sources: number[];
  sourceCosts: number[];
  sourceCount: number;
  sourceKind: string;
  sourceMask: number[];
  costLimit: number;
  costs: number[];
  edges: number[];
  breaks: number[];
  bands: number[];
  bandCounts: number[];
};
const fixture = decodeOracleFixture<{reference: string; cases: CostCase[]}>(fixtureJSON);
const OFFSETS = [
  [0, 1],
  [1, 1],
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1]
];
for (const scene of fixture.cases) {
  it(`GPUCostDistance ground metric/source costs/bands: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, scene.name);
    const grid = makeGPURasterMetadata(scene);
    const length = scene.width * scene.height;
    const costs = owner.output('costs', 'float32', length);
    const links = owner.output('links', 'uint32', length);
    const bands = owner.output('bands', 'uint32', length);
    const counts = owner.output('band-counts', 'uint32', scene.breaks.length);
    const converged = owner.output('converged', 'uint32', 1);
    owner.graph.add(
      new GPUCostDistance({
        grid,
        cellSizeMode: scene.mode,
        maximumIterations: 64,
        maximumTieIterations: 64,
        friction: {
          id: 'friction',
          format: 'float32',
          noDataValue: -9999,
          storage: {
            kind: 'buffer',
            values: owner.input('raw', 'float32', new Float32Array(scene.raw))
          },
          validity: owner.input('validity', 'uint32', new Uint32Array(scene.validity))
        },
        frictionParameters: owner.input(
          'calibration',
          'float32',
          new Float32Array(scene.calibration)
        ),
        parameters: owner.input(
          'parameters',
          'float32',
          getGPUCostDistanceParameterValues({costLimit: scene.costLimit}, grid)
        ),
        sources:
          scene.sourceKind === 'mask'
            ? undefined
            : owner.input('sources', 'uint32', new Uint32Array(scene.sources)),
        sourceCosts:
          scene.sourceKind === 'mask'
            ? undefined
            : owner.input('source-costs', 'float32', new Float32Array(scene.sourceCosts)),
        sourceCount:
          scene.sourceKind === 'mask'
            ? undefined
            : owner.input('source-count', 'uint32', Uint32Array.of(scene.sourceCount)),
        sourceMask:
          scene.sourceKind === 'list'
            ? undefined
            : owner.input('source-mask', 'uint32', new Uint32Array(scene.sourceMask)),
        costBreaks: owner.input('breaks', 'float32', new Float32Array(scene.breaks)),
        output: {
          costs: costs.view,
          backLinks: links.view,
          bands: bands.view,
          bandCounts: counts.view,
          status: {converged: converged.view}
        }
      })
    );
    const paths = [1, 3, length].map((capacity, index) => {
      const ids = owner.output(`ids-${index}`, 'uint32', capacity);
      const count = owner.output(`count-${index}`, 'uint32', 1);
      const required = owner.output(`required-${index}`, 'uint32', 1);
      const overflow = owner.output(`overflow-${index}`, 'uint32', 1);
      const target = Math.floor(length / 2);
      owner.graph.add(
        new GPUCostDistancePath({
          id: `path-${index}`,
          grid,
          backLinks: links.view,
          target: owner.input(`target-${index}`, 'uint32', Uint32Array.of(target)),
          output: {
            path: {
              ids: ids.view,
              status: {count: count.view, requiredCount: required.view, overflow: overflow.view}
            }
          }
        })
      );
      return {ids, count, required, overflow, target, capacity};
    });
    const compiled = owner.graph.compile();
    const compare = (
      className: string,
      metric: string,
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
      profile: 'E' | 'L' = 'E'
    ) =>
      compareRasterParity(
        className,
        scene.name,
        fixture.reference,
        metric,
        actual,
        expected,
        profile
      );
    try {
      submitGraph(device, compiled, undefined);
      compare('GPUCostDistance', 'converged', await readUint32(converged.buffer, 1), [1]);
      compare(
        'GPUCostDistance',
        'all-costs',
        await readFloat32(costs.buffer, length),
        scene.costs,
        'L'
      );
      compare('GPUCostDistance', 'all-bands', await readUint32(bands.buffer, length), scene.bands);
      compare(
        'GPUCostDistance',
        'band-counts',
        await readUint32(counts.buffer, scene.breaks.length),
        scene.bandCounts
      );
      const backlinks = await readUint32(links.buffer, length);
      const walks: number[][] = [];
      const objectives = Float64Array.from(scene.costs, () => Infinity);
      for (let start = 0; start < length; start++) {
        let cell = start;
        let cost = 0;
        const seen = new Set<number>();
        const walk: number[] = [];
        while (cell >= 0 && cell < length && !seen.has(cell)) {
          seen.add(cell);
          walk.push(cell);
          const code = backlinks[cell];
          if (code === 0) {
            const source = scene.sources.slice(0, scene.sourceCount).indexOf(cell);
            const initial = scene.sourceMask[cell]
              ? 0
              : source >= 0
                ? scene.sourceCosts[source]
                : Infinity;
            objectives[start] = cost + initial;
            break;
          }
          const direction = Math.log2(code);
          if (!Number.isInteger(direction) || direction < 0 || direction >= 8) break;
          const [rowOffset, columnOffset] = OFFSETS[direction];
          const row = Math.floor(cell / scene.width) + rowOffset;
          const column = (cell % scene.width) + columnOffset;
          if (row < 0 || row >= scene.height || column < 0 || column >= scene.width) break;
          cost += scene.edges[cell * 8 + direction];
          cell = row * scene.width + column;
        }
        walks.push(Number.isFinite(objectives[start]) ? walk : []);
      }
      compare(
        'GPUCostDistance',
        'every-backlink-walk-independent-objective',
        objectives,
        scene.costs,
        'L'
      );
      for (const path of paths) {
        const expected = walks[path.target];
        compare(
          'GPUCostDistancePath',
          `capacity-${path.capacity}-status`,
          [
            ...(await readUint32(path.count.buffer, 1)),
            ...(await readUint32(path.required.buffer, 1)),
            ...(await readUint32(path.overflow.buffer, 1))
          ],
          [
            Math.min(expected.length, path.capacity),
            expected.length,
            Number(expected.length > path.capacity)
          ]
        );
        compare(
          'GPUCostDistancePath',
          `capacity-${path.capacity}-complete-certified-walk`,
          await readUint32(path.ids.buffer, Math.min(expected.length, path.capacity)),
          expected.slice(0, path.capacity)
        );
      }
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
