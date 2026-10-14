// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {countDistinctPipelines} from '../../utils/gpu-contributor-test-utils';

it('GPUCoverageSimplification validates props and declares nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'coverage-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const make = () => ({
    positions: view('float32x2', 12),
    ringOffsets: view('uint32', 4),
    polygonOffsets: view('uint32', 4),
    featureOffsets: view('uint32', 4),
    parameters: view('float32', 4),
    output: {
      positions: view('float32x2', 12),
      ringOffsets: view('uint32', 4),
      status: {overflow: view('uint32', 1)}
    }
  });
  const base = make();
  expect(() => new GPUCoverageSimplification({...base, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(
    () => new GPUCoverageSimplification({...base, spatialContext: DISTANCE_FRAME})
  ).not.toThrow();
  expect(() => new GPUCoverageSimplification({...base, snapTolerance: -1})).toThrow(
    /snapTolerance/
  );
  expect(() => new GPUCoverageSimplification({...base, parameters: view('float32', 2)})).toThrow(
    /parameters/
  );
  expect(
    () =>
      new GPUCoverageSimplification({
        ...base,
        output: {...base.output, ringOffsets: view('uint32', 5)}
      })
  ).toThrow(/ringOffsets length/);
  expect(
    () =>
      new GPUCoverageSimplification({
        ...base,
        output: {...base.output, outputMask: view('uint32', 3)}
      })
  ).toThrow(/outputMask length/);
  expect(() => new GPUCoverageSimplification({...base, topologyIterationCount: -1})).toThrow(
    /topologyIterationCount/
  );
  expect(() => new GPUCoverageSimplification({...base, topologyPairCapacity: 0})).toThrow(
    /topologyPairCapacity/
  );
  expect(
    () =>
      new GPUCoverageSimplification({
        ...base,
        output: {...base.output, topologyStatistics: view('uint32', 3)}
      })
  ).toThrow(/topologyStatistics/);
  const ids = new GPUCoverageSimplification(base).getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('coverage-simplification-feature-ring-starts');
  expect(ids[1]).toBe('coverage-simplification-vertex-topology');
  expect(ids.at(-1)).toBe('coverage-simplification-emit-offsets');
  expect(ids).toContain('coverage-simplification-arcs-init');
  expect(ids).toContain('coverage-simplification-ring-minimum');
  expect(ids).toContain('coverage-simplification-round-0-classify');
  // The measuring-only last round is built only when topologyStatistics consumes it.
  expect(ids).not.toContain('coverage-simplification-round-4-classify');
  expect(ids).toContain('coverage-simplification-round-3-classify');
  const statsIds = new GPUCoverageSimplification({
    ...base,
    id: 'stats',
    output: {...base.output, topologyStatistics: view('uint32', 4)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(statsIds).toContain('stats-round-4-classify');
  expect(statsIds).not.toContain('stats-round-4-repair');
  expect(ids).toContain('coverage-simplification-round-3-repair');
  const plainIds = new GPUCoverageSimplification({...base, id: 'plain', topologyIterationCount: 0})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plainIds.some(id => id.includes('plain-round-'))).toBe(false);
});

it('GPUCoverageSimplification builds the same pipelines for any number of topology iterations', () => {
  const countPipelines = (topologyIterationCount: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: `rounds-${topologyIterationCount}`
    });
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `view-${serial++}`, format, length);
    const nodes = new GPUCoverageSimplification({
      positions: view('float32x2', 12),
      ringOffsets: view('uint32', 4),
      polygonOffsets: view('uint32', 4),
      featureOffsets: view('uint32', 4),
      parameters: view('float32', 4),
      topologyIterationCount,
      output: {
        positions: view('float32x2', 12),
        ringOffsets: view('uint32', 4),
        status: {overflow: view('uint32', 1)},
        topologyStatistics: view('uint32', 4)
      }
    }).getCommandNodes(graph);
    return countDistinctPipelines(nodes);
  };
  // The round index must not reach the classify source or constants (rule 12).
  expect(countPipelines(3)).toBeGreaterThan(10);
  expect(countPipelines(3)).toBe(countPipelines(4));
  expect(countPipelines(4)).toBe(countPipelines(5));
});

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
