// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialLagImpacts} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts';
import {
  GPUSpatialWeightsTraces,
  type GPUSpatialWeightsTracesProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  getSpatialLagImpactsFromTraces,
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantTruncationBound,
  getSpatialLogDeterminantWGSL,
  getSpatialMultiplierTraceFromTraces,
  getSpatialMultiplierTraceWGSL,
  getSpatialMultiplierTotalFromPowerSums,
  getSpatialMultiplierTotalWGSL,
  getSpatialMultiplierTruncationBound,
  getSpatialWeightsTracesOnCPU,
  getSpatialWeightsTracesProbeValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createNearestNeighborScene,
  multiply,
  toDenseWeights
} from './spatial-regression-diagnostics-oracle';
import {createLagScene} from './spatial-two-stage-least-squares-oracle';
import {
  SPATIAL_WEIGHTS_TRACES_REFERENCE,
  SPATIAL_WEIGHTS_TRACES_REFERENCE_MAXIMUM_POWER
} from './spatial-weights-traces-reference';

const Q = SPATIAL_WEIGHTS_TRACES_REFERENCE_MAXIMUM_POWER;

const scenes = [
  ['lattice', createLagScene(12, 21, 0.5, true).weights, SPATIAL_WEIGHTS_TRACES_REFERENCE.lattice],
  [
    'rawLattice',
    createLagScene(10, 8, 0.12, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.rawLattice
  ],
  [
    'knn60',
    createNearestNeighborScene(60, 3, 13, 0.6, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn60
  ],
  [
    'knn90',
    createNearestNeighborScene(90, 3, 29, 0.6, false).weights,
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn90
  ]
] as const;

function expectRelative(label: string, actual: number, expected: number, relative: number) {
  const floor = 1e-6;
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

let serial = 0;

it('GPUSpatialWeightsTraces validates its inputs and declares its kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const createProps = (
    overrides: Partial<GPUSpatialWeightsTracesProps> = {}
  ): GPUSpatialWeightsTracesProps => ({
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    maximumPower: 5,
    output: {powerTraces: view('float32', 5), powerSums: view('float32', 5)},
    ...overrides
  });
  const expectThrows = (overrides: Partial<GPUSpatialWeightsTracesProps>, message: RegExp) =>
    expect(() => new GPUSpatialWeightsTraces(createProps(overrides))).toThrow(message);
  expectThrows({maximumPower: 0}, /maximumPower/);
  expectThrows({maximumPower: 257}, /maximumPower/);
  expectThrows({maximumPower: 2.5}, /maximumPower/);
  expectThrows({mode: 'dense' as 'exact'}, /mode/);
  expectThrows({probeCount: 0}, /probeCount/);
  expectThrows({seed: -1}, /seed/);
  expectThrows({seed: 2 ** 32}, /seed/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows(
    {output: {powerTraces: view('float32', 4), powerSums: view('float32', 5)}},
    /output.powerTraces/
  );
  expectThrows(
    {output: {powerTraces: view('float32', 5), powerSums: view('float32', 4)}},
    /output.powerSums/
  );
  expectThrows({includeGramTraces: true}, /gramTraces/);
  expectThrows(
    {
      output: {
        powerTraces: view('float32', 5),
        powerSums: view('float32', 5),
        gramTraces: view('float32', 3)
      }
    },
    /output.gramTraces/
  );
  expectThrows(
    {
      includeGramTraces: false,
      output: {
        powerTraces: view('float32', 5),
        powerSums: view('float32', 5),
        gramTraces: view('float32', 4)
      }
    },
    /gramTraces/
  );
  expectThrows(
    {
      output: {
        powerTraces: view('float32', 5),
        powerSums: view('float32', 5),
        approximation: view('uint32', 0)
      }
    },
    /approximation/
  );
  // Exact mode beyond the explicit row limit.
  expectThrows(
    {
      mode: 'exact',
      weights: {
        offsets: view('uint32', 16386),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      }
    },
    /exact mode/
  );
  const exact = new GPUSpatialWeightsTraces(createProps());
  expect(exact.mode).toBe('exact');
  expect(exact.probeTotal).toBe(10);
  expect(new GPUSpatialWeightsTraces(createProps({mode: 'stochastic'})).probeTotal).toBe(64);
  // Defaults to stochastic above 4096 rows.
  const large = new GPUSpatialWeightsTraces(
    createProps({
      weights: {
        offsets: view('uint32', 4098),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      }
    })
  );
  expect(large.mode).toBe('stochastic');
  // 4096 rows x 8M-element scratch budget: 2048 probes per batch.
  expect(
    new GPUSpatialWeightsTraces(
      createProps({
        mode: 'exact',
        weights: {
          offsets: view('uint32', 4097),
          neighbors: view('uint32', 4),
          weights: view('float32', 4)
        }
      })
    ).batchProbeCount
  ).toBe(2048);

  const nodes = exact.getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(0);
  const ids = nodes.map(node => node.id.replace('spatial-weights-traces-', ''));
  expect(ids).toContain('sums-merge');
  expect(ids).toContain('final-merge');
  expect(ids.some(id => id.startsWith('transpose-'))).toBe(false);
  const withGram = new GPUSpatialWeightsTraces(
    createProps({
      id: 'gram-traces',
      mode: 'stochastic',
      maximumPower: 1,
      output: {
        powerTraces: view('float32', 1),
        powerSums: view('float32', 1),
        gramTraces: view('float32', 4),
        approximation: view('uint32', 1)
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id.replace('gram-traces-', ''));
  expect(withGram).toContain('transpose-slot-keys');
  expect(withGram).toContain('gram-moments-tiles');
  expect(withGram.some(id => id.startsWith('gram-square-'))).toBe(true);
  expect(withGram).toContain('approximation');
  device.destroy();
});

it('GPUSpatialLagImpacts validates its inputs and declares one kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `impacts-view-${serial++}`, format, length);
  const createProps = (overrides: Record<string, unknown> = {}) =>
    ({
      coefficients: view('float32', 16),
      coefficientStride: 4,
      rhoIndex: 3,
      predictorCount: 2,
      rowCount: 144,
      maximumPower: 8,
      traces: {
        powerTraces: view('float32', 8),
        powerSums: view('float32', 8),
        infinityNorm: view('float32', 1)
      },
      output: {impacts: view('float32', 6)},
      ...overrides
    }) as ConstructorParameters<typeof GPUSpatialLagImpacts>[0];
  const expectThrows = (overrides: Record<string, unknown>, message: RegExp) =>
    expect(() => new GPUSpatialLagImpacts(createProps(overrides))).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({rowCount: 0}, /rowCount/);
  expectThrows({maximumPower: 300}, /maximumPower/);
  expectThrows({coefficientStride: 0}, /coefficientStride/);
  expectThrows({rhoIndex: 2}, /rhoIndex/);
  expectThrows({coefficients: view('float32', 12)}, /coefficients/);
  expectThrows({output: {impacts: view('float32', 5)}}, /output.impacts/);
  expectThrows({rhoBound: 0}, /rhoBound/);
  expectThrows(
    {
      traces: {
        powerTraces: view('float32', 7),
        powerSums: view('float32', 8),
        infinityNorm: view('float32', 1)
      }
    },
    /traces/
  );
  expectThrows(
    {
      traces: {
        powerTraces: view('float32', 8),
        powerSums: view('float32', 8),
        infinityNorm: view('float32', 0)
      }
    },
    /traces/
  );
  expectThrows({status: view('uint32', 0)}, /status/);
  const nodes = new GPUSpatialLagImpacts(createProps({status: view('uint32', 1)})).getCommandNodes(
    graph
  );
  expect(nodes).toHaveLength(1);
  device.destroy();
});

it('exact CPU traces and sums match numpy matrix_power on every pinned scene', () => {
  for (const [name, weights, reference] of scenes) {
    const result = getSpatialWeightsTracesOnCPU(weights, Q, {includeGramTraces: true});
    for (let power = 0; power < Q; power++) {
      expectRelative(
        `${name} tr(W^${power + 1})`,
        result.powerTraces[power] * result.infinityNorm ** (power + 1),
        reference.powerTraces[power],
        1e-9
      );
      expectRelative(
        `${name} 1'W^${power + 1}1`,
        result.powerSums[power] * result.infinityNorm ** (power + 1),
        reference.powerSums[power],
        1e-9
      );
    }
    reference.gramTraces.forEach((expected, index) => {
      expectRelative(`${name} gram ${index}`, result.gramTraces![index], expected, 1e-9);
    });
  }
  // Gram traces also hold when only one power is requested.
  const single = getSpatialWeightsTracesOnCPU(scenes[2][1], 1, {includeGramTraces: true});
  scenes[2][2].gramTraces.forEach((expected, index) => {
    expectRelative(`knn60 gram ${index} Q=1`, single.gramTraces![index], expected, 1e-9);
  });
});

it('self-loops are allowed in the CPU traces', () => {
  // W = [[0.5, 0.5], [1, 0]]: tr(W) = 0.5, W^2 = [[0.75, 0.25], [0.5, 0.5]], tr(W^2) = 1.25.
  const weights = {
    offsets: [0, 2, 3],
    neighbors: [0, 1, 0],
    weights: [0.5, 0.5, 1]
  };
  const result = getSpatialWeightsTracesOnCPU(weights, 2, {includeGramTraces: true});
  expect(result.powerTraces[0]).toBeCloseTo(0.5, 12);
  expect(result.powerTraces[1]).toBeCloseTo(1.25, 12);
  // tr(W'W) = 0.25 + 0.25 + 1.
  expect(result.gramTraces![0]).toBeCloseTo(1.5, 12);
});

it('stochastic CPU traces are seeded, reproducible and unbiased within the Hutchinson bound', () => {
  const [, weights, reference] = scenes[2];
  const options = {
    mode: 'stochastic',
    probeCount: 4000,
    seed: 11,
    includeGramTraces: true
  } as const;
  const first = getSpatialWeightsTracesOnCPU(weights, 4, options);
  const second = getSpatialWeightsTracesOnCPU(weights, 4, options);
  expect(Array.from(second.powerTraces)).toEqual(Array.from(first.powerTraces));
  const other = getSpatialWeightsTracesOnCPU(weights, 4, {...options, seed: 12});
  expect(Array.from(other.powerTraces)).not.toEqual(Array.from(first.powerTraces));
  // Var(z'Az) <= 2 ||A||_F^2 for Rademacher z, so six standard errors bound the error.
  const dense = toDenseWeights(weights);
  let power = dense;
  for (let q = 1; q <= 4; q++) {
    if (q > 1) {
      power = multiply(power, dense);
    }
    const frobenius = Math.sqrt(
      power.reduce((sum, row) => sum + row.reduce((a, b) => a + b * b, 0), 0)
    );
    const bound = 6 * Math.sqrt(2 / options.probeCount) * frobenius;
    expect(
      Math.abs(first.powerTraces[q - 1] * first.infinityNorm ** q - reference.powerTraces[q - 1]),
      `tr(W^${q})`
    ).toBeLessThan(bound);
    // The ones-vector chain is exact regardless of mode.
    expectRelative(
      `sums ${q}`,
      first.powerSums[q - 1] * first.infinityNorm ** q,
      reference.powerSums[q - 1],
      1e-9
    );
  }
  // The exact column-sum moments do not depend on probes.
  expectRelative("tr(W'W)", first.gramTraces![0], reference.gramTraces[0], 1e-9);
  expectRelative('sum c^2', first.gramTraces![3], reference.gramTraces[3], 1e-9);
  // The probe stream is plus or minus one and balanced.
  let sum = 0;
  for (let row = 0; row < 4000; row++) {
    const value = getSpatialWeightsTracesProbeValue(3, 5, row);
    expect(Math.abs(value)).toBe(1);
    sum += value;
  }
  expect(Math.abs(sum)).toBeLessThan(6 * Math.sqrt(4000));
});

it('log-determinant and multiplier series match numpy within the stated truncation bounds', () => {
  for (const [name, weights, reference] of scenes) {
    const n = weights.offsets.length - 1;
    const radius = reference.spectralRadius;
    const result = getSpatialWeightsTracesOnCPU(weights, Q);
    for (const key of Object.keys(reference.logDeterminant)) {
      const rho = Number(key);
      if (Math.abs(rho) * radius >= 1) {
        // Outside the convergence region the series is not a valid log-determinant; the bound says so.
        expect(getSpatialLogDeterminantTruncationBound(rho, Q, n, radius)).toBe(Infinity);
        continue;
      }
      const logDeterminant = getSpatialLogDeterminantFromTraces(
        rho,
        result.powerTraces,
        result.infinityNorm
      );
      const logBound = getSpatialLogDeterminantTruncationBound(rho, Q, n, radius);
      const logError = Math.abs(
        logDeterminant - reference.logDeterminant[key as keyof typeof reference.logDeterminant]
      );
      expect(logError, `${name} logdet rho ${rho}`).toBeLessThanOrEqual(
        logBound * (1 + 1e-9) + 1e-9
      );
      // The 1e-6 slack covers float32 row sums of 1 + 1e-7 (max row sum, not the spectral radius, bounds 1'W^q1).
      const multiplierBound = getSpatialMultiplierTruncationBound(rho, Q, n, radius);
      const traceError = Math.abs(
        getSpatialMultiplierTraceFromTraces(rho, result.powerTraces, n, result.infinityNorm) -
          reference.multiplierTrace[key as keyof typeof reference.multiplierTrace]
      );
      expect(traceError, `${name} multiplier trace rho ${rho}`).toBeLessThanOrEqual(
        multiplierBound * (1 + 1e-6) + 1e-9
      );
      const totalError = Math.abs(
        getSpatialMultiplierTotalFromPowerSums(rho, result.powerSums, n, result.infinityNorm) -
          reference.multiplierTotal[key as keyof typeof reference.multiplierTotal]
      );
      expect(totalError, `${name} multiplier total rho ${rho}`).toBeLessThanOrEqual(
        multiplierBound * (1 + 1e-6) + 1e-9
      );
    }
  }
});

it('log-determinant truncation shrinks geometrically with the series length', () => {
  const [, weights, reference] = scenes[0];
  const n = weights.offsets.length - 1;
  const errors = [12, 30, 80].map(power => {
    const traced = getSpatialWeightsTracesOnCPU(weights, power);
    return Math.abs(
      getSpatialLogDeterminantFromTraces(0.5, traced.powerTraces, traced.infinityNorm) -
        reference.logDeterminant['0.5']
    );
  });
  // rho = 0.5 on a bipartite lattice: Q = 12 is visibly truncated, Q = 80 is at float64 precision.
  expect(errors[0]).toBeGreaterThan(1e-5);
  expect(errors[0]).toBeLessThanOrEqual(getSpatialLogDeterminantTruncationBound(0.5, 12, n));
  expect(errors[1]).toBeLessThan(errors[0] / 1e3);
  expect(errors[2]).toBeLessThan(1e-9);
  // rho = 0.9 needs far more terms: Q = 12 is poor, which the bound reflects.
  const slow = getSpatialWeightsTracesOnCPU(weights, 12);
  expect(
    Math.abs(
      getSpatialLogDeterminantFromTraces(0.9, slow.powerTraces, slow.infinityNorm) -
        reference.logDeterminant['0.9']
    )
  ).toBeGreaterThan(0.1);
});

it('impacts mirror reproduces spreg full multipliers on the row-standardized scenes', () => {
  for (const [name, weights, reference] of [scenes[0], scenes[2], scenes[3]]) {
    const spreg = reference.spreg;
    const n = weights.offsets.length - 1;
    const result = getSpatialWeightsTracesOnCPU(weights, 40);
    const [direct, indirect, total] = spreg.full.multipliers;
    for (let predictor = 0; predictor < 2; predictor++) {
      const beta = spreg.betas[1 + predictor];
      const impacts = getSpatialLagImpactsFromTraces(
        spreg.rho,
        beta,
        result.powerTraces,
        result.powerSums,
        n,
        result.infinityNorm
      );
      expectRelative(`${name} direct ${predictor}`, impacts.direct, direct * beta, 1e-5);
      expectRelative(`${name} indirect ${predictor}`, impacts.indirect, indirect * beta, 1e-4);
      expectRelative(`${name} total ${predictor}`, impacts.total, total * beta, 1e-5);
    }
    // The exact general-W definitions agree with the series.
    const [exactDirect, , exactTotal] = spreg.exact.multipliers;
    const impacts = getSpatialLagImpactsFromTraces(
      spreg.rho,
      1,
      result.powerTraces,
      result.powerSums,
      n,
      result.infinityNorm
    );
    expectRelative(`${name} exact direct`, impacts.direct, exactDirect, 1e-8);
    expectRelative(`${name} exact total`, impacts.total, exactTotal, 1e-8);
  }
});

it("impacts on a non-row-standardized W use the exact 1'W^q1 sums, not 1 / (1 - rho)", () => {
  const [, weights, reference] = scenes[1];
  const n = weights.offsets.length - 1;
  // rho * spectralRadius = 0.2 * 3.84 < 1: the series converges, slowly (0.77^121 is below 1e-13).
  const rho = 0.2;
  const result = getSpatialWeightsTracesOnCPU(weights, 120);
  const impacts = getSpatialLagImpactsFromTraces(
    rho,
    1,
    result.powerTraces,
    result.powerSums,
    n,
    result.infinityNorm
  );
  expectRelative('total', impacts.total, reference.multiplierTotal['0.2'] / n, 1e-9);
  expectRelative('direct', impacts.direct, reference.multiplierTrace['0.2'] / n, 1e-9);
  // Row sums reach 4, so the total multiplier is well above spreg's 1 / (1 - rho) = 1.25.
  expect(impacts.total).toBeGreaterThan(1 / (1 - rho) + 0.5);
});

it('WGSL series generators emit Horner evaluations and reject bad arguments', () => {
  const logDeterminant = getSpatialLogDeterminantWGSL(12, 'powerTraces');
  expect(logDeterminant).toContain('fn getSpatialLogDeterminant(rho: f32) -> f32');
  expect(logDeterminant).toContain('powerTraces[powerTracesOffset + q - 1u] / f32(q)');
  expect(logDeterminant).toContain('step < 12u');
  expect(logDeterminant).toContain('return -t * accumulator');
  expect(logDeterminant).toContain('let t = rho * (1.0);');
  expect(getSpatialLogDeterminantWGSL(12, 'powerTraces', 'norm[0]')).toContain(
    'let t = rho * (norm[0]);'
  );
  const trace = getSpatialMultiplierTraceWGSL(7, 'traces', 144);
  expect(trace).toContain('fn getSpatialMultiplierTrace(rho: f32) -> f32');
  expect(trace).toContain('+ f32(144u)');
  expect(trace).not.toContain('f32(q)');
  expect(getSpatialMultiplierTotalWGSL(7, 'sums', 144)).toContain('fn getSpatialMultiplierTotal');
  expect(() => getSpatialLogDeterminantWGSL(0, 'x')).toThrow(/maximumPower/);
  expect(() => getSpatialLogDeterminantWGSL(257, 'x')).toThrow(/maximumPower/);
  expect(() => getSpatialMultiplierTraceWGSL(5, 'bad name', 4)).toThrow(/bindingName/);
  // The Horner evaluation (mirrored on the CPU) equals the direct sum.
  const traces = [0.3, 1.1, -0.4, 2.2];
  const rho = 0.35;
  const direct = -traces.reduce(
    (sum, value, index) => sum + (rho ** (index + 1) * value) / (index + 1),
    0
  );
  expect(getSpatialLogDeterminantFromTraces(rho, traces)).toBeCloseTo(direct, 14);
});
