// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUGeographicDistributionParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPUGeographicDistribution,
  type GPUGeographicDistributionParameters
} from '../../../src/geospatial/geographic-distribution';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGeographicDistribution,
  type GeographicDistributionOracleResult
} from './geographic-distribution-oracle';

const VERTEX_COUNT = 16;
const MEDIAN_ITERATIONS = 24;

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

type Scene = {
  positions: Float32Array;
  lineEnds?: Float32Array;
  weights?: Float32Array;
  groupIds?: Uint32Array;
  mask?: Uint32Array;
  groupCount: number;
};

/** Grouped anisotropic clusters at a large offset, with excluded rows and an empty last group. */
function createScene(seed: number, rows: number, groupCount: number, lines = false): Scene {
  const random = createRandom(seed);
  const positions = new Float32Array(rows * 2);
  const lineEnds = new Float32Array(rows * 2);
  const weights = new Float32Array(rows);
  const groupIds = new Uint32Array(rows);
  const mask = new Uint32Array(rows);
  // The last group (when there is more than one) never receives rows.
  const usedGroups = Math.max(groupCount - (groupCount > 1 ? 1 : 0), 1);
  for (let row = 0; row < rows; row++) {
    const group = Math.floor(random() * usedGroups);
    const angle = group * 0.7;
    const a = (random() - 0.5) * 40 * (1 + group * 0.1);
    const b = (random() - 0.5) * 8;
    positions[row * 2] = 100000 + group * 30 + a * Math.cos(angle) - b * Math.sin(angle);
    positions[row * 2 + 1] = 200000 - group * 20 + a * Math.sin(angle) + b * Math.cos(angle);
    lineEnds[row * 2] = positions[row * 2] + (random() - 0.3) * 10;
    lineEnds[row * 2 + 1] = positions[row * 2 + 1] + (random() - 0.4) * 10;
    const roll = random();
    weights[row] = roll < 0.03 ? NaN : roll < 0.06 ? -1 : roll < 0.08 ? 0 : 0.25 + random() * 3;
    groupIds[row] = random() < 0.03 ? groupCount + 4 : group;
    mask[row] = random() < 0.1 ? 0 : 1;
  }
  // Rows with a non-finite coordinate are excluded too.
  positions[6] = NaN;
  if (lines) {
    // A zero-length line is skipped by the directional mean only.
    lineEnds[10] = positions[10];
    lineEnds[11] = positions[11];
  }
  return {positions, lineEnds, weights, groupIds, mask, groupCount};
}

type Outputs = Record<
  | 'counts'
  | 'weightSums'
  | 'meanCenters'
  | 'medianCenters'
  | 'medianConverged'
  | 'standardDistances'
  | 'ellipses'
  | 'directionalMeans'
  | 'ellipseVertices'
  | 'circleVertices',
  Buffer
>;

type Fixture = {
  run(
    parameters?: GPUGeographicDistributionParameters
  ): Promise<GeographicDistributionOracleResult>;
  /** Number of `graph.compile()` calls beyond the first. */
  rebuildCount: number;
  destroy(): void;
};

function createFixture(
  device: Device,
  scene: Scene,
  options: {
    useLines?: boolean;
    unweighted?: boolean;
    groupless?: boolean;
  } = {}
): Fixture {
  const rows = scene.positions.length / 2;
  const groups = scene.groupCount;
  const graph = new GPUCommandGraph(device, {
    id: 'geographic-distribution-graph'
  });
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const sizes = {
    counts: groups,
    weightSums: groups,
    meanCenters: groups * 2,
    medianCenters: groups * 2,
    medianConverged: groups,
    standardDistances: groups,
    ellipses: groups * 3,
    directionalMeans: groups * 3,
    ellipseVertices: groups * VERTEX_COUNT * 2,
    circleVertices: groups * VERTEX_COUNT * 2
  };
  const outputs = {} as Outputs;
  for (const [name, length] of Object.entries(sizes)) {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[name as keyof Outputs] = buffer;
  }
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'distribution-parameters',
    format: 'float32',
    length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
  });
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    key: keyof Outputs,
    format: Format
  ) =>
    importGraphBuffer(
      graph,
      `out-${key}`,
      outputs[key],
      format,
      sizes[key] / (format === 'float32x2' ? 2 : 1)
    );
  graph.add(
    new GPUGeographicDistribution({
      id: 'distribution',
      positions: importGraphBuffer(graph, 'positions', input(scene.positions), 'float32x2', rows),
      weights:
        scene.weights && !options.unweighted
          ? importGraphBuffer(graph, 'weights', input(scene.weights), 'float32', rows)
          : undefined,
      groupIds:
        scene.groupIds && !options.groupless
          ? importGraphBuffer(graph, 'group-ids', input(scene.groupIds), 'uint32', rows)
          : undefined,
      groupCount: groups,
      mask: scene.mask
        ? importGraphBuffer(graph, 'mask', input(scene.mask), 'uint32', rows)
        : undefined,
      lineEnds:
        scene.lineEnds && options.useLines
          ? importGraphBuffer(graph, 'line-ends', input(scene.lineEnds), 'float32x2', rows)
          : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      medianIterations: MEDIAN_ITERATIONS,
      polygonVertexCount: VERTEX_COUNT,
      output: {
        counts: view('counts', 'uint32'),
        weightSums: view('weightSums', 'float32'),
        meanCenters: view('meanCenters', 'float32x2'),
        medianCenters: view('medianCenters', 'float32x2'),
        medianConverged: view('medianConverged', 'uint32'),
        standardDistances: view('standardDistances', 'float32'),
        ellipses: view('ellipses', 'float32'),
        directionalMeans: options.useLines ? view('directionalMeans', 'float32') : undefined,
        ellipseVertices: view('ellipseVertices', 'float32x2'),
        circleVertices: view('circleVertices', 'float32x2')
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(parameters = {}) {
      parameterBuffer.write(getGPUGeographicDistributionParameterValues(parameters));
      submitGraph(device, compiled, undefined);
      return {
        counts: await readUint32(outputs.counts, sizes.counts),
        weightSums: await readFloat32(outputs.weightSums, sizes.weightSums),
        meanCenters: await readFloat32(outputs.meanCenters, sizes.meanCenters),
        medianCenters: await readFloat32(outputs.medianCenters, sizes.medianCenters),
        medianConverged: await readUint32(outputs.medianConverged, sizes.medianConverged),
        standardDistances: await readFloat32(outputs.standardDistances, sizes.standardDistances),
        ellipses: await readFloat32(outputs.ellipses, sizes.ellipses),
        directionalMeans: options.useLines
          ? await readFloat32(outputs.directionalMeans, sizes.directionalMeans)
          : [],
        ellipseVertices: await readFloat32(outputs.ellipseVertices, sizes.ellipseVertices),
        circleVertices: await readFloat32(outputs.circleVertices, sizes.circleVertices)
      };
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectClose(name: string, actual: number[], expected: number[], tolerance: number) {
  expect(actual.length, name).toBeGreaterThanOrEqual(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const a = actual[index];
    const e = expected[index];
    if (Number.isNaN(e)) {
      expect(a, `${name}[${index}]`).toBeNaN();
    } else {
      expect(Math.abs(a - e), `${name}[${index}] ${a} vs ${e}`).toBeLessThanOrEqual(tolerance);
    }
  }
}

function expectParity(
  actual: GeographicDistributionOracleResult,
  scene: Scene,
  parameters: GPUGeographicDistributionParameters,
  options: {
    useLines?: boolean;
    unweighted?: boolean;
    groupless?: boolean;
  } = {}
) {
  const expected = computeGeographicDistribution({
    positions: scene.positions,
    weights: options.unweighted ? undefined : scene.weights,
    groupIds: options.groupless ? undefined : scene.groupIds,
    mask: scene.mask,
    lineEnds: options.useLines ? scene.lineEnds : undefined,
    groupCount: scene.groupCount,
    vertexCount: VERTEX_COUNT,
    medianIterations: MEDIAN_ITERATIONS,
    ...parameters
  });
  expect(actual.counts).toEqual(expected.counts);
  // Positions near 1e5 have an f32 ulp of 0.0078, so centres and rings compare to a few ulps.
  expectClose('weightSums', actual.weightSums, expected.weightSums, 1e-3 * 100);
  expectClose('meanCenters', actual.meanCenters, expected.meanCenters, 0.05);
  expectClose('medianCenters', actual.medianCenters, expected.medianCenters, 0.1);
  expectClose('standardDistances', actual.standardDistances, expected.standardDistances, 0.05);
  // Ellipse: axes relative to scale; angle only where the ellipse is not nearly circular.
  for (let group = 0; group < scene.groupCount; group++) {
    const [angle, sigmaX, sigmaY] = expected.ellipses.slice(group * 3, group * 3 + 3);
    const actualEllipse = actual.ellipses.slice(group * 3, group * 3 + 3);
    if (Number.isNaN(sigmaX)) {
      expect(actualEllipse.every(Number.isNaN)).toBe(true);
      continue;
    }
    expect(Math.abs(actualEllipse[1] - sigmaX)).toBeLessThanOrEqual(0.02 * sigmaY + 0.02);
    expect(Math.abs(actualEllipse[2] - sigmaY)).toBeLessThanOrEqual(0.02 * sigmaY + 0.02);
    if (sigmaY > 1.5 * sigmaX) {
      expect(Math.abs(actualEllipse[0] - angle)).toBeLessThanOrEqual(0.02);
    }
  }
  expect(actual.ellipseVertices.length).toBe(expected.ellipseVertices.length);
  if (!options.groupless) {
    expectClose('circleVertices', actual.circleVertices, expected.circleVertices, 0.1);
  }
  if (options.useLines) {
    expectClose('directionalMeans', actual.directionalMeans, expected.directionalMeans, 2e-3);
  }
  // Convergence flags are only compared away from the tolerance boundary.
  return expected;
}

it('GPUGeographicDistribution matches the CPU oracle on random grouped weighted data', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [seed, groupCount] of [
    [1, 1],
    [2, 5],
    [3, 37]
  ] as const) {
    const scene = createScene(seed, 3000, groupCount);
    const fixture = createFixture(device, scene);
    const parameters = {origin: [100000, 200000] as [number, number]};
    const actual = await fixture.run(parameters);
    const expected = expectParity(actual, scene, parameters);
    expect(expected.counts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(2000);
    if (groupCount > 1) {
      // Last group is empty: zero counts and NaN statistics.
      const last = groupCount - 1;
      expect(actual.counts[last]).toBe(0);
      expect(actual.weightSums[last]).toBe(0);
      expect(actual.meanCenters[last * 2]).toBeNaN();
      expect(actual.medianCenters[last * 2]).toBeNaN();
      expect(actual.standardDistances[last]).toBeNaN();
      expect(actual.ellipses[last * 3]).toBeNaN();
      expect(actual.ellipseVertices[last * VERTEX_COUNT * 2]).toBeNaN();
      expect(actual.medianConverged[last]).toBe(0);
    }
    fixture.destroy();
  }
});

it('GPUGeographicDistribution works unweighted without groups and without a local origin', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(4, 1500, 1);
  const fixture = createFixture(device, scene, {
    unweighted: true,
    groupless: true
  });
  const actual = await fixture.run({
    standardDeviations: 2,
    ellipseConvention: 'standard'
  });
  const expected = expectParity(
    actual,
    scene,
    {standardDeviations: 2, ellipseConvention: 'standard'},
    {unweighted: true, groupless: true}
  );
  expect(expected.counts[0]).toBeGreaterThan(1000);
  fixture.destroy();
});

it('GPUGeographicDistribution rings match the ellipse and circle definitions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(5, 2000, 3);
  const fixture = createFixture(device, scene);
  const parameters = {
    origin: [100000, 200000] as [number, number],
    standardDeviations: 2
  };
  const actual = await fixture.run(parameters);
  for (const group of [0, 1]) {
    const [mx, my] = actual.meanCenters.slice(group * 2, group * 2 + 2);
    const [angle, sigmaX, sigmaY] = actual.ellipses.slice(group * 3, group * 3 + 3);
    const radius = actual.standardDistances[group];
    expect(sigmaY).toBeGreaterThanOrEqual(sigmaX);
    for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
      const offset = (group * VERTEX_COUNT + vertex) * 2;
      // Un-rotate into the ellipse frame: x'^2 / sigmaX^2 + y'^2 / sigmaY^2 = 1.
      const dx = actual.ellipseVertices[offset] - mx;
      const dy = actual.ellipseVertices[offset + 1] - my;
      const u = dx * Math.cos(angle) + dy * Math.sin(angle);
      const v = -dx * Math.sin(angle) + dy * Math.cos(angle);
      expect(Math.abs((u / sigmaX) ** 2 + (v / sigmaY) ** 2 - 1)).toBeLessThan(0.05);
      const circleDistance = Math.hypot(
        actual.circleVertices[offset] - mx,
        actual.circleVertices[offset + 1] - my
      );
      expect(Math.abs(circleDistance - radius)).toBeLessThan(0.05);
    }
    // Counter-clockwise ring starting on the ellipse x axis.
    const first = actual.circleVertices.slice(
      group * VERTEX_COUNT * 2,
      group * VERTEX_COUNT * 2 + 2
    );
    expect(Math.abs(first[0] - mx - radius)).toBeLessThan(0.05);
  }
  fixture.destroy();
});

it('GPUGeographicDistribution computes linear directional means, including orientationOnly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(6, 2500, 4, true);
  const fixture = createFixture(device, scene, {useLines: true});
  for (const orientationOnly of [false, true]) {
    const parameters = {
      origin: [100000, 200000] as [number, number],
      orientationOnly
    };
    const actual = await fixture.run(parameters);
    const expected = expectParity(actual, scene, parameters, {
      useLines: true
    });
    expect(expected.directionalMeans.slice(0, 3).every(Number.isFinite)).toBe(true);
  }
  // Opposite lines cancel when directed and agree when undirected.
  const opposite: Scene = {
    positions: Float32Array.from([0, 0, 0, 0]),
    lineEnds: Float32Array.from([2, 0, -2, 0]),
    groupCount: 1
  };
  const oppositeFixture = createFixture(device, opposite, {useLines: true});
  const directed = await oppositeFixture.run({});
  expect(directed.directionalMeans[0]).toBeNaN();
  expect(directed.directionalMeans[1]).toBeCloseTo(1, 5);
  expect(directed.directionalMeans[2]).toBeCloseTo(2, 5);
  const undirected = await oppositeFixture.run({orientationOnly: true});
  expect(undirected.directionalMeans[0]).toBeCloseTo(0, 5);
  expect(undirected.directionalMeans[1]).toBeCloseTo(0, 5);
  oppositeFixture.destroy();
  fixture.destroy();
});

it('GPUGeographicDistribution recovers a known ellipse, a single point and the median', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const known: Scene = {
    positions: Float32Array.from([-2, 0, 2, 0, 0, -1, 0, 1]),
    groupCount: 1
  };
  const fixture = createFixture(device, known);
  const actual = await fixture.run({});
  expect(actual.ellipses[0]).toBeCloseTo(Math.PI / 2, 5);
  expect(actual.ellipses[1]).toBeCloseTo(1, 5);
  expect(actual.ellipses[2]).toBeCloseTo(2, 5);
  expect(actual.standardDistances[0]).toBeCloseTo(Math.sqrt(2.5), 5);
  expect(actual.counts).toEqual([4]);
  fixture.destroy();

  const single: Scene = {positions: Float32Array.from([3, 4]), groupCount: 1};
  const singleFixture = createFixture(device, single);
  const point = await singleFixture.run({});
  expect(point.meanCenters).toEqual([3, 4]);
  expect(point.medianCenters).toEqual([3, 4]);
  expect(point.standardDistances).toEqual([0]);
  expect(point.ellipses.slice(1)).toEqual([0, 0]);
  expect(point.medianConverged).toEqual([1]);
  singleFixture.destroy();
});

it('GPUGeographicDistribution is bitwise reproducible across encodings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(7, 6000, 9, true);
  const parameters = {origin: [100000, 200000] as [number, number]};
  const bits = (result: GeographicDistributionOracleResult) =>
    JSON.stringify(
      Object.values(result).map(values =>
        Array.from(new Uint32Array(Float32Array.from(values as number[]).buffer))
      )
    );
  const first = createFixture(device, scene, {useLines: true});
  const baseline = bits(await first.run(parameters));
  for (let repeat = 0; repeat < 3; repeat++) {
    expect(bits(await first.run(parameters))).toBe(baseline);
  }
  const second = createFixture(device, scene, {useLines: true});
  expect(bits(await second.run(parameters))).toBe(baseline);
  second.destroy();
  first.destroy();
});

it('GPUGeographicDistribution changes parameters between encodings without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(8, 2000, 4, true);
  const fixture = createFixture(device, scene, {useLines: true});
  const settings: GPUGeographicDistributionParameters[] = [
    {origin: [100000, 200000]},
    {origin: [99990, 200010], standardDeviations: 3},
    {
      origin: [100000, 200000],
      ellipseConvention: 'standard',
      orientationOnly: true
    },
    {origin: [0, 0], standardDeviations: 0.5, medianTolerance: 100}
  ];
  for (const parameters of settings) {
    const actual = await fixture.run(parameters);
    expectParity(actual, scene, parameters, {useLines: true});
  }
  // A loose tolerance reports every non-empty group as converged; the default does not have to.
  const loose = await fixture.run({
    origin: [100000, 200000],
    medianTolerance: 1000
  });
  expect(loose.medianConverged.slice(0, 3)).toEqual([1, 1, 1]);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});
