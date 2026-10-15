// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {analysisFixtures} from './parity-wp3-network-analysis-fixture';
import {evaluateRankResidual} from '../../../../gpgpu/test/gpu-graph/parity-wp3-graph-metrics-fixture';

for (const fixture of analysisFixtures) {
  it(`network reference integrity only: ${fixture.input.case}`, () => {
    const {input, reverse} = fixture;
    const {nodeCount} = input;
    expect(input.versions['networkx']).toBeTruthy();
    expect(new Set(input.nodeIds).size).toBe(nodeCount);
    expect(input.offsets).toHaveLength(nodeCount + 1);
    expect(input.offsets[nodeCount]).toBe(input.neighbors.length);
    const edges: [number, number, number][] = [];
    const originalEdges = new Map<number, [number, number]>();
    for (let source = 0; source < nodeCount; source++) {
      for (let edge = input.offsets[source]; edge < input.offsets[source + 1]; edge++) {
        edges.push([source, input.neighbors[edge], 1]);
        originalEdges.set(input.edgeIds[edge], [source, input.neighbors[edge]]);
      }
    }
    expect(originalEdges.size).toBe(input.neighbors.length);
    for (let source = 0; source < nodeCount; source++) {
      for (let edge = reverse.offsets[source]; edge < reverse.offsets[source + 1]; edge++) {
        expect(originalEdges.get(reverse.edgeIds[edge])).toEqual([reverse.neighbors[edge], source]);
      }
    }
    for (const metric of fixture.metrics) {
      expect(metric.values).toHaveLength(nodeCount);
      expect(metric.extent).toEqual([Math.min(...metric.values), Math.max(...metric.values)]);
      expect(metric.normalized).toEqual(
        metric.values.map(value =>
          metric.extent[1] > metric.extent[0]
            ? (value - metric.extent[0]) / (metric.extent[1] - metric.extent[0])
            : 0
        )
      );
      if (metric.metric === 'degree' || metric.metric === 'inDegree') {
        const offsets = metric.metric === 'degree' ? input.offsets : reverse.offsets;
        expect(metric.values).toEqual(offsets.slice(1).map((value, node) => value - offsets[node]));
      }
      if (metric.metric === 'pageRank') {
        expect(metric.values.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
        const residual = evaluateRankResidual(
          {nodeCount, edges, directed: true},
          metric.values,
          Math.fround(0.85)
        );
        expect(Math.max(...residual)).toBeLessThan(1e-13);
      }
    }
  });
  it(`neighborhood certificates and provenance only: ${fixture.input.case}`, () => {
    const {input} = fixture;
    for (const scenario of fixture.neighborhoods) {
      const topology = scenario.incoming ? fixture.reverse : input;
      const limit = Math.min(scenario.hops, scenario.maximumHops);
      const seeds = new Set(
        scenario.starts.slice(0, scenario.startCount).filter(node => node < input.nodeCount)
      );
      expect(scenario.hopDistances).toHaveLength(input.nodeCount);
      expect(scenario.nodeMask).toEqual(scenario.hopDistances.map(value => Number(value <= limit)));
      const predecessors = new Set<number>();
      const mask: number[] = [];
      for (let source = 0; source < input.nodeCount; source++) {
        for (let edge = topology.offsets[source]; edge < topology.offsets[source + 1]; edge++) {
          const target = topology.neighbors[edge];
          const distance = scenario.hopDistances[source];
          const targetDistance = scenario.hopDistances[target];
          mask.push(Number(scenario.nodeMask[source] && scenario.nodeMask[target]));
          if (distance < limit) expect(targetDistance).toBeLessThanOrEqual(distance + 1);
          if (targetDistance === distance + 1) predecessors.add(target);
        }
      }
      // Triangle inequalities plus a decreasing predecessor certificate prove shortest
      // hop values; no production reachability oracle or GPU relaxation is reproduced.
      for (let node = 0; node < input.nodeCount; node++) {
        const distance = scenario.hopDistances[node];
        if (distance === 0) expect(seeds.has(node)).toBe(true);
        if (seeds.has(node)) expect(distance).toBe(0);
        if (distance > 0 && distance <= limit) expect(predecessors.has(node)).toBe(true);
      }
      expect(scenario.edgeMask).toEqual(mask);
      const nodes = input.nodeIds.filter((_, node) => scenario.nodeMask[node]);
      const edges = topology.edgeIds.filter((_, edge) => mask[edge]);
      expect(scenario.nodeIds).toEqual(nodes.slice(0, scenario.nodeCapacity));
      expect(scenario.edgeIds).toEqual(edges.slice(0, scenario.edgeCapacity));
      expect([scenario.nodeCount, scenario.nodeOverflow]).toEqual([
        Math.min(nodes.length, scenario.nodeCapacity),
        Number(nodes.length > scenario.nodeCapacity)
      ]);
      expect([scenario.edgeCount, scenario.edgeOverflow]).toEqual([
        Math.min(edges.length, scenario.edgeCapacity),
        Number(edges.length > scenario.edgeCapacity)
      ]);
    }
  });
}
