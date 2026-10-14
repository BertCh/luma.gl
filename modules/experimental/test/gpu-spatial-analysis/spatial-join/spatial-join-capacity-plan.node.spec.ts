// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {getGPUCapacityRecovery} from '../../../src/analysis-core/contracts/capacity-planning';
import {
  getGPUSpatialJoinCapacityPlan,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicateJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type Kind = GPUSpatialJoinGeometry['kind'];

function createGeometry(
  graph: GPUCommandGraph,
  name: string,
  kind: Kind,
  featureCount: number
): GPUSpatialJoinGeometry {
  const view = <F extends 'uint32' | 'float32x2'>(suffix: string, format: F, length: number) =>
    createTransientView(graph, `input-${name}-${suffix}`, format, length);
  if (kind === 'points') {
    return {kind, positions: view('positions', 'float32x2', featureCount)};
  }
  if (kind === 'lines') {
    return {
      kind,
      positions: view('positions', 'float32x2', 3 * featureCount),
      lineOffsets: view('line-offsets', 'uint32', featureCount + 1)
    };
  }
  return {
    kind,
    positions: view('positions', 'float32x2', 4 * featureCount),
    featureOffsets: view('feature-offsets', 'uint32', featureCount + 1),
    polygonOffsets: view('polygon-offsets', 'uint32', featureCount + 1),
    ringOffsets: view('ring-offsets', 'uint32', featureCount + 1)
  };
}

/** Sums the bytes of the transients the join declared itself (its ID prefix, not the inputs). */
function getJoinTransientBytes(graph: GPUCommandGraph, id: string): number {
  const buffers = (
    graph as unknown as {buffers: Map<string, {id: string; byteLength: number; transient: boolean}>}
  ).buffers;
  let bytes = 0;
  for (const buffer of buffers.values()) {
    if (buffer.transient && buffer.id.startsWith(`${id}-`)) {
      bytes += buffer.byteLength;
    }
  }
  return bytes;
}

describe('getGPUSpatialJoinCapacityPlan', () => {
  // E-16: the plan's transient bytes are the join's own allocations, computed from the same sizes.
  for (const [leftKind, rightKind, predicate] of [
    ['lines', 'lines', 'intersects'],
    ['points', 'polygons', 'intersects'],
    ['polygons', 'polygons', 'touches'],
    ['lines', 'polygons', 'crosses']
  ] as const) {
    it(`equals the declared transient bytes of a ${leftKind}/${rightKind} ${predicate} join`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const leftCount = 10;
      const rightCount = 20;
      const candidateCapacity = 37;
      const pairCapacity = 9;
      const pairView = (name: string, length: number) =>
        createTransientView(graph, `input-pairs-${name}`, 'uint32', length);
      const props: GPUSpatialPredicateJoinProps = {
        id: 'join',
        left: createGeometry(graph, 'left', leftKind, leftCount),
        right: createGeometry(graph, 'right', rightKind, rightCount),
        predicate,
        candidateCapacity,
        output: {
          pairs: {
            leftIds: pairView('left', pairCapacity),
            rightIds: pairView('right', pairCapacity),
            status: {
              count: pairView('count', 1),
              overflow: pairView('overflow', 1)
            }
          }
        }
      };
      const join = new GPUSpatialPredicateJoin(props);
      join.getCommandNodes(graph);
      const plan = getGPUSpatialJoinCapacityPlan({
        leftCount,
        rightCount,
        leftKind,
        rightKind,
        relate: join.usesRelateEngine,
        candidateCapacity,
        pairCapacity
      });
      expect(plan.estimatedPeakTransientBytes).toBe(getJoinTransientBytes(graph, 'join'));
      // The join's own plan for the next frame, from the counts it reported, is the same plan.
      const next = join.getCapacityPlan({
        candidateCount: candidateCapacity,
        requiredCount: pairCapacity
      });
      expect(next.estimatedPeakTransientBytes).toBe(plan.estimatedPeakTransientBytes);
      expect(plan.estimatedOutputBytes).toBe(pairCapacity * 8);
    });
  }

  it('defaults the join candidate capacity from the plan', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const join = new GPUSpatialPredicateJoin({
      id: 'join',
      left: createGeometry(graph, 'left', 'lines', 10),
      right: createGeometry(graph, 'right', 'lines', 3),
      predicate: 'intersects',
      output: {
        pairs: {
          leftIds: createTransientView(graph, 'input-left-ids', 'uint32', 4),
          rightIds: createTransientView(graph, 'input-right-ids', 'uint32', 4),
          status: {
            count: createTransientView(graph, 'input-count', 'uint32', 1),
            overflow: createTransientView(graph, 'input-overflow', 'uint32', 1)
          }
        }
      }
    });
    expect(join.candidateCapacity).toBe(
      getGPUSpatialJoinCapacityPlan({leftCount: 10, rightCount: 3}).candidates.capacity
    );
    expect(join.candidateCapacity).toBe(30);
  });

  it('drives a candidate-then-output grow loop from observed counts', () => {
    // Frame 1 ran with the default estimate and overflowed the candidate stage.
    const first = getGPUSpatialJoinCapacityPlan({leftCount: 100, rightCount: 50});
    const recovery = getGPUCapacityRecovery(
      {kind: 'grow-on-next-frame'},
      {
        candidate: {
          capacity: first.candidates.capacity,
          requiredCount: 1200,
          overflow: true
        },
        output: {capacity: first.pairs.capacity, count: 100, requiredCount: 100, overflow: false}
      }
    );
    expect(recovery.action).toBe('grow');
    // Frame 2 plans from the observed candidate requirement.
    const second = getGPUSpatialJoinCapacityPlan({
      leftCount: 100,
      rightCount: 50,
      observedCandidateCount: 1200,
      observedRequiredCount: 180
    });
    expect(second.candidates.capacity).toBe(1200);
    expect(second.pairs.capacity).toBe(180);
    expect(second.estimatedPeakTransientBytes).toBeGreaterThan(first.estimatedPeakTransientBytes);
  });

  it('plans measured candidate and pair stages independently', () => {
    const join = getGPUSpatialJoinCapacityPlan({
      leftCount: 100,
      rightCount: 50,
      candidateCapacity: 300,
      pairCapacity: 120,
      observedCandidateCount: 275,
      observedRequiredCount: 80
    });
    expect(join.candidates).toMatchObject({capacity: 300, estimatedRequiredCount: 275});
    expect(join.pairs).toMatchObject({capacity: 120, estimatedRequiredCount: 80});
    expect(join.estimatedWorkItems).toBe(425);
    expect(join.estimatedPeakTransientBytes).toBeGreaterThan(join.estimatedOutputBytes);
  });

  it('uses the default per-left estimates and lets observations override them', () => {
    // Defaults: min(rightCount, 8) candidates and min(rightCount, 1) matches per left row.
    const estimated = getGPUSpatialJoinCapacityPlan({leftCount: 10, rightCount: 3});
    expect(estimated.candidates.estimatedRequiredCount).toBe(30);
    expect(estimated.pairs.estimatedRequiredCount).toBe(10);
    expect(estimated.candidates.limited).toBe(false);
    const limited = getGPUSpatialJoinCapacityPlan({
      leftCount: 10,
      rightCount: 100,
      candidateCapacity: 20
    });
    expect(limited.candidates).toMatchObject({
      estimatedRequiredCount: 80,
      capacity: 20,
      limited: true
    });
    expect(
      getGPUSpatialJoinCapacityPlan({leftCount: 10, rightCount: 3, observedCandidateCount: 0})
        .candidates.estimatedRequiredCount
    ).toBe(0);
    expect(() => getGPUSpatialJoinCapacityPlan({leftCount: -1, rightCount: 3})).toThrow(
      /leftCount/
    );
    expect(() =>
      getGPUSpatialJoinCapacityPlan({leftCount: 1, rightCount: 3, pairCapacity: -1})
    ).toThrow(/capacity/);
  });
});
