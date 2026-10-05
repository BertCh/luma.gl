// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUGeographicDistributionParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPUGeographicDistribution,
  type GPUGeographicDistributionOutput,
  type GPUGeographicDistributionProps
} from '../../../src/gpu-spatial-analysis/geographic-distribution';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeGeographicDistribution} from './geographic-distribution-oracle';

const ROWS = 8;
let viewCounter = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  name: string,
  format: Format,
  length: number
) {
  return createTransientView(graph, `${name}-${viewCounter++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGeographicDistributionProps> = {},
  output: GPUGeographicDistributionOutput = {
    meanCenters: view(graph, 'out-mean', 'float32x2', 1)
  }
): GPUGeographicDistributionProps {
  return {
    id: 'g',
    positions: view(graph, 'positions', 'float32x2', ROWS),
    parameters: view(graph, 'parameters', 'float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
    output,
    ...overrides
  };
}

const closeTo = (actual: number, expected: number, tolerance = 1e-9) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);

it('GPUGeographicDistribution validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'validation'
  });
  expect(() => new GPUGeographicDistribution(createProps(graph))).not.toThrow();
  expect(() => new GPUGeographicDistribution(createProps(graph, {}, {}))).toThrow(
    'at least one output'
  );
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {
          weights: view(graph, 'weights', 'float32', ROWS - 1)
        })
      )
  ).toThrow('weights length must equal positions length');
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {
          mask: view(graph, 'mask', 'float32' as 'uint32', ROWS)
        })
      )
  ).toThrow('uint32');
  expect(() => new GPUGeographicDistribution(createProps(graph, {groupCount: 3}))).toThrow(
    'groupCount must be 1 without groupIds'
  );
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {
          groupIds: view(graph, 'groups', 'uint32', ROWS),
          groupCount: 0
        })
      )
  ).toThrow('groupCount');
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {}, {directionalMeans: view(graph, 'out-directional', 'float32', 3)})
      )
  ).toThrow('requires lineEnds');
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {}, {ellipses: view(graph, 'out-ellipses', 'float32', 2)})
      )
  ).toThrow('output.ellipses must hold at least 3 rows');
  expect(() => new GPUGeographicDistribution(createProps(graph, {medianIterations: 0}))).toThrow(
    'medianIterations'
  );
  expect(() => new GPUGeographicDistribution(createProps(graph, {polygonVertexCount: 2}))).toThrow(
    'polygonVertexCount'
  );
  expect(
    () =>
      new GPUGeographicDistribution(
        createProps(graph, {
          parameters: view(graph, 'short-parameters', 'float32', 4)
        })
      )
  ).toThrow('parameters must hold 8');
  const positions = view(graph, 'aliased', 'float32x2', ROWS);
  expect(
    () => new GPUGeographicDistribution(createProps(graph, {positions}, {meanCenters: positions}))
  ).toThrow('outputs must not share buffers with inputs');
});

it('GPUGeographicDistribution returns deterministic node IDs within the binding limit', () => {
  const create = () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: 'nodes'
    });
    const output: GPUGeographicDistributionOutput = {
      counts: view(graph, 'o-counts', 'uint32', 3),
      meanCenters: view(graph, 'o-mean', 'float32x2', 3),
      medianCenters: view(graph, 'o-median', 'float32x2', 3),
      standardDistances: view(graph, 'o-std', 'float32', 3),
      ellipses: view(graph, 'o-ellipses', 'float32', 9),
      directionalMeans: view(graph, 'o-direction', 'float32', 9),
      ellipseVertices: view(graph, 'o-ellipse-vertices', 'float32x2', 12),
      circleVertices: view(graph, 'o-circle-vertices', 'float32x2', 12)
    };
    const contributor = new GPUGeographicDistribution(
      createProps(
        graph,
        {
          groupIds: view(graph, 'groups', 'uint32', ROWS),
          groupCount: 3,
          lineEnds: view(graph, 'ends', 'float32x2', ROWS),
          polygonVertexCount: 4,
          medianIterations: 2
        },
        output
      )
    );
    return contributor.getCommandNodes(graph).map(node => node.id);
  };
  const first = create();
  expect(create()).toEqual(first);
  expect(first.every(nodeId => nodeId.startsWith('g-'))).toBe(true);
  expect(new Set(first).size).toBe(first.length);
  expect(first).toContain('g-median-1-update');
  expect(first).toContain('g-direction-publish');
});

it('getGPUGeographicDistributionParameterValues packs and validates', () => {
  expect(Array.from(getGPUGeographicDistributionParameterValues())).toEqual([
    0,
    0,
    1,
    Math.fround(Math.SQRT2),
    0,
    Math.fround(1e-3),
    0,
    0
  ]);
  const values = getGPUGeographicDistributionParameterValues({
    origin: [10, -20],
    standardDeviations: 2,
    ellipseConvention: 'standard',
    orientationOnly: true,
    medianTolerance: 0.5
  });
  expect(Array.from(values)).toEqual([10, -20, 2, 1, 1, 0.5, 0, 0]);
  const target = new Float32Array(10).fill(9);
  expect(getGPUGeographicDistributionParameterValues({}, target)).toBe(target);
  expect(target[8]).toBe(9);
  expect(() => getGPUGeographicDistributionParameterValues({standardDeviations: 0})).toThrow(
    'positive'
  );
  expect(() => getGPUGeographicDistributionParameterValues({origin: [NaN, 0]})).toThrow('finite');
  expect(() => getGPUGeographicDistributionParameterValues({medianTolerance: -1})).toThrow(
    'negative'
  );
  expect(() =>
    getGPUGeographicDistributionParameterValues({
      ellipseConvention: 'other' as 'arcgis'
    })
  ).toThrow('ellipseConvention');
  expect(() => getGPUGeographicDistributionParameterValues({}, new Float32Array(3))).toThrow(
    'must hold 8'
  );
});

it('oracle: axis-aligned data gives the known ellipse', () => {
  // Variances: Sxx = 2, Syy = 0.5, so the long axis is x and ArcGIS calls it the "y" axis.
  const positions = [-2, 0, 2, 0, 0, -1, 0, 1];
  const arcgis = computeGeographicDistribution({positions, groupCount: 1});
  const [angle, sigmaX, sigmaY] = arcgis.ellipses;
  closeTo(angle, Math.PI / 2);
  closeTo(sigmaX, 1);
  closeTo(sigmaY, 2);
  closeTo(arcgis.standardDistances[0], Math.sqrt(2.5));
  expect(arcgis.counts).toEqual([4]);
  expect(arcgis.meanCenters).toEqual([0, 0]);
  const standard = computeGeographicDistribution({
    positions,
    groupCount: 1,
    ellipseConvention: 'standard',
    standardDeviations: 2
  });
  closeTo(standard.ellipses[1], 2 * Math.sqrt(0.5));
  closeTo(standard.ellipses[2], 2 * Math.SQRT2);
  closeTo(standard.standardDistances[0], 2 * Math.sqrt(2.5));
  // For the ArcGIS convention the ring touches the data extreme (2, 0).
  const arcgisRing = arcgis.ellipseVertices;
  const peak = Math.max(...arcgisRing.filter((_, index) => index % 2 === 0));
  closeTo(peak, 2, 1e-9);
});

it('oracle: collinear, single and rotated data', () => {
  const diagonal = computeGeographicDistribution({
    positions: [0, 0, 1, 1, 2, 2, 3, 3],
    groupCount: 1,
    ellipseConvention: 'standard'
  });
  // Long axis at +45 degrees, so the short ("x") axis is at 135 degrees = -45 degrees.
  closeTo(diagonal.ellipses[0], -Math.PI / 4);
  closeTo(diagonal.ellipses[1], 0, 1e-12);
  closeTo(diagonal.ellipses[2], Math.sqrt(2.5));
  const single = computeGeographicDistribution({
    positions: [3, 4],
    groupCount: 1
  });
  expect(single.meanCenters).toEqual([3, 4]);
  expect(single.ellipses.slice(1)).toEqual([0, 0]);
  expect(single.standardDistances).toEqual([0]);
  expect(single.medianConverged).toEqual([1]);

  const rotation = Math.PI / 6;
  const points: number[] = [];
  for (const [x, y] of [
    [-3, 0.5],
    [3, 0.5],
    [-3, -0.5],
    [3, -0.5],
    [0, 1],
    [0, -1]
  ]) {
    points.push(
      x * Math.cos(rotation) - y * Math.sin(rotation),
      x * Math.sin(rotation) + y * Math.cos(rotation)
    );
  }
  const rotated = computeGeographicDistribution({
    positions: points,
    groupCount: 1,
    ellipseConvention: 'standard'
  });
  // Long axis at 30 degrees, so the short x axis is at 120 degrees = -60 degrees.
  closeTo(rotated.ellipses[0], rotation + Math.PI / 2 - Math.PI);
  expect(rotated.ellipses[2]).toBeGreaterThan(rotated.ellipses[1]);
});

it('oracle: weights, masks, groups and empty groups', () => {
  const duplicated = computeGeographicDistribution({
    positions: [0, 0, 4, 0, 4, 0, 1, 5],
    groupCount: 1
  });
  const weighted = computeGeographicDistribution({
    positions: [0, 0, 4, 0, 1, 5],
    weights: [1, 2, 1],
    groupCount: 1
  });
  for (const key of ['meanCenters', 'standardDistances', 'ellipses'] as const) {
    duplicated[key].forEach((value, index) => closeTo(value, weighted[key][index], 1e-12));
  }
  const grouped = computeGeographicDistribution({
    positions: [0, 0, 2, 0, 9, 9, 7, 7, NaN, 1],
    weights: [1, 1, 1, -1, 1],
    groupIds: [0, 0, 1, 1, 0],
    mask: [1, 1, 1, 1, 1],
    groupCount: 3
  });
  expect(grouped.counts).toEqual([2, 1, 0]);
  expect(grouped.weightSums).toEqual([2, 1, 0]);
  expect(grouped.meanCenters.slice(0, 4)).toEqual([1, 0, 9, 9]);
  expect(Number.isNaN(grouped.meanCenters[4])).toBe(true);
  expect(Number.isNaN(grouped.ellipses[6])).toBe(true);
  expect(grouped.medianConverged[2]).toBe(0);
});

it('oracle: median centre and directional mean', () => {
  // Triangle with a weighted vertex: the median of 3 collinear points is the middle one.
  const median = computeGeographicDistribution({
    positions: [0, 0, 1, 0, 10, 0, 1, 0.001],
    groupCount: 1,
    medianIterations: 200
  });
  closeTo(median.medianCenters[0], 1, 0.02);
  const lines = computeGeographicDistribution({
    positions: [0, 0, 0, 0],
    lineEnds: [2, 0, 0, 2],
    groupCount: 1
  });
  closeTo(lines.directionalMeans[0], Math.PI / 4, 1e-12);
  closeTo(lines.directionalMeans[1], 1 - Math.SQRT1_2, 1e-12);
  closeTo(lines.directionalMeans[2], 2, 1e-12);
  const reversed = {
    positions: [0, 0, 0, 0],
    lineEnds: [2, 0, -2, 0],
    groupCount: 1
  };
  const directed = computeGeographicDistribution(reversed);
  expect(Number.isNaN(directed.directionalMeans[0])).toBe(true);
  closeTo(directed.directionalMeans[1], 1);
  const undirected = computeGeographicDistribution({
    ...reversed,
    orientationOnly: true
  });
  closeTo(undirected.directionalMeans[0], 0, 1e-12);
  closeTo(undirected.directionalMeans[1], 0, 1e-12);
  // Zero-length lines are skipped by the directional mean.
  const degenerate = computeGeographicDistribution({
    positions: [0, 0, 5, 5],
    lineEnds: [0, 0, 5, 7],
    groupCount: 1
  });
  closeTo(degenerate.directionalMeans[0], Math.PI / 2, 1e-12);
  closeTo(degenerate.directionalMeans[2], 2, 1e-12);
});
