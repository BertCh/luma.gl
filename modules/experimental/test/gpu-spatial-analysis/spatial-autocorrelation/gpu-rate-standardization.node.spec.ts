// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUAssuncaoReisStandardization,
  GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH,
  type GPUAssuncaoReisStandardizationProps
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-rate-standardization';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeAssuncaoReisOracle,
  getRateInputs,
  getRateReferenceZ,
  RATE_ROW_COUNT,
  toFloat64,
  type RateDataset
} from './gpu-rate-standardization-fixture';
import {RATE_REFERENCE} from './gpu-rate-standardization-reference';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUAssuncaoReisStandardizationProps> = {}
): GPUAssuncaoReisStandardizationProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    events: view('float32', 10),
    populations: view('float32', 10),
    standardizedRates: view('float32', 10),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUAssuncaoReisStandardizationProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUAssuncaoReisStandardization(createProps(graph, overrides(graph)))).toThrow(
    message
  );
  device.destroy();
}

// Oracle: fix/work/A/oracle-u8-rate-moran.py (esda 2.10.0 / libpysal 4.15.0)
it('the CPU oracle reproduces esda assuncao_rate, including the negative-variance fallback', () => {
  for (const dataset of ['full', 'missing', 'under'] as const satisfies readonly RateDataset[]) {
    const {events, populations} = getRateInputs(dataset);
    const oracle = computeAssuncaoReisOracle(events, populations);
    const expected = getRateReferenceZ(dataset);
    expect(oracle.z.length).toBe(RATE_ROW_COUNT);
    for (const [row, value] of expected.entries()) {
      if (Number.isNaN(value)) {
        expect(oracle.z[row], `${dataset} row ${row}`).toBeNaN();
      } else {
        expect(Math.abs(oracle.z[row] - value), `${dataset} row ${row}`).toBeLessThan(1e-12);
      }
    }
  }
  const under = computeAssuncaoReisOracle(
    getRateInputs('under').events,
    getRateInputs('under').populations
  );
  // a < 0 (not clamped) and some rows fall back to the Poisson variance b / n.
  expect(under.statistics[3]).toBeLessThan(0);
  expect(under.negativeVarianceRows).toBe(RATE_REFERENCE.underNegativeVarianceRows);
  expect(under.negativeVarianceRows).toBeGreaterThan(0);
  const full = computeAssuncaoReisOracle(
    getRateInputs('full').events,
    getRateInputs('full').populations
  );
  for (const [slot, value] of RATE_REFERENCE.fullStatistics.entries()) {
    expect(Math.abs(full.statistics[slot] - value) / Math.abs(value)).toBeLessThan(1e-12);
  }
});

it('the missing dataset drops exactly the three invalid rows', () => {
  const rows = Object.keys(RATE_REFERENCE.invalidRows).map(Number);
  const z = toFloat64(RATE_REFERENCE.missingZ);
  expect(rows.sort((a, b) => a - b)).toEqual([5, 20, 33]);
  expect(z.filter(Number.isNaN).length).toBe(3);
  for (const row of rows) {
    expect(z[row]).toBeNaN();
  }
});

it('GPUAssuncaoReisStandardization validates its inputs', () => {
  expectThrows(
    graph => ({populations: createTransientView(graph, 'short-populations', 'float32', 4)}),
    /populations length/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, 'short-mask', 'uint32', 4)}),
    /mask length/
  );
  expectThrows(
    graph => ({
      statistics: createTransientView(
        graph,
        'short-statistics',
        'float32',
        GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH - 1
      )
    }),
    /statistics must hold 5/
  );
  expectThrows(
    graph => ({invalidCount: createTransientView(graph, 'invalid-float', 'float32', 1) as never}),
    /invalidCount/
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  expect(
    () => new GPUAssuncaoReisStandardization({...base, standardizedRates: base.events})
  ).toThrow(/outputs must not share/);
  device.destroy();
});

it('GPUAssuncaoReisStandardization creates deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const node = new GPUAssuncaoReisStandardization(
    createProps(graph, {
      mask: createTransientView(graph, 'mask', 'uint32', 10),
      statistics: createTransientView(graph, 'statistics', 'float32', 5),
      invalidCount: createTransientView(graph, 'invalid-count', 'uint32', 1)
    })
  );
  const ids = node.getCommandNodes(graph).map(commandNode => commandNode.id);
  expect(ids).toEqual([
    'assuncao-reis-standardization-totals-blocks',
    'assuncao-reis-standardization-totals-reduce',
    'assuncao-reis-standardization-mean-correction-blocks',
    'assuncao-reis-standardization-mean-correction-reduce',
    'assuncao-reis-standardization-variance-blocks',
    'assuncao-reis-standardization-variance-reduce',
    'assuncao-reis-standardization-standardize'
  ]);
  device.destroy();
});
