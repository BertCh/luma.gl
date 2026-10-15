// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  createGPUComputationCaptures,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationCapture
} from '@luma.gl/gpgpu';
import {
  captureNetworkKFunction,
  describeStructure
} from '../../../src/computation-lenses/scenarios/gpu-network/network-k-function.lens';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {getViewNodeText} from '../../../src/computation-views/view-node';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

// Synthetic readbacks exercise presentation and serialization only; no CPU analysis or GPU run.
function makeReadbacks() {
  const captures = new Map<string, GPUComputationCapture & {values: Float32Array | Uint32Array}>();
  for (const [name, length] of [
    ['crashPositions', 116],
    ['kValues', 160],
    ['pairCounts', 160],
    ['envelope', 24],
    ['converged', 1],
    ['snappedEventCount', 1]
  ] as const) {
    const integer = ['pairCounts', 'converged', 'snappedEventCount'].includes(name);
    captures.set(name, {
      id: name,
      name,
      elementType: integer ? 'u32' : 'f32',
      provenance: 'gpu-readback',
      ...(name === 'crashPositions' && {components: 2}),
      values: integer
        ? Uint32Array.from({length}, (_, index) => index + 1)
        : Float32Array.from({length}, (_, index) => index + 0.5)
    });
  }
  captures.get('snappedEventCount')!.values[0] = 58;
  return {captures, final: (name: string) => captures.get(name)!};
}

test('K captures keep observed bands separate from simulations and align the envelope rows', () => {
  const {final} = makeReadbacks();
  const snapshots = captureNetworkKFunction(final);
  for (const name of ['pairCounts', 'kValues']) {
    const capture = snapshots.find(snapshot => snapshot.name === name)!;
    expect(capture.values).toEqual(final(name).values.slice(0, 8));
    expect(capture.keySpace).toBe('band');
    expect(final(name).values).toHaveLength(160);
  }
  for (const [row, statistic] of ['minimum', 'mean', 'maximum'].entries()) {
    const capture = snapshots.find(snapshot => snapshot.name === `envelope-${statistic}`)!;
    expect(capture.values).toEqual(final('envelope').values.slice(row * 8, (row + 1) * 8));
    expect(capture).toMatchObject({keySpace: 'band', role: 'output', provenance: 'gpu-readback'});
    expect(capture.meaning).toContain('19 Philox patterns (seed 17)');
    expect(capture.meaning).toContain('not a confidence interval');
  }
});

test('K capture rejects unconverged searches and insufficient snapped events', () => {
  const {final} = makeReadbacks();
  final('converged').values[0] = 0;
  expect(() => captureNetworkKFunction(final)).toThrow(/converge/);
  final('converged').values[0] = 1;
  final('snappedEventCount').values[0] = 1;
  expect(() => captureNetworkKFunction(final)).toThrow(/fewer than two/);
});

test('seeded K graph includes GPU simulations and envelope with room inside the record budget', async () => {
  const device = createNullWebGPUDevice();
  try {
    const record = await describeStructure(device);
    expect(validateGPUComputationRecord(record).issues).toEqual([]);
    expect(record.choices.find(choice => choice.name === 'simulations')?.value).toBe(19);
    const nodes = record.graphs[0].nodes;
    expect(nodes.some(node => node.id === 'network-k-function-simulate')).toBe(true);
    expect(nodes.some(node => node.id === 'network-k-function-envelope')).toBe(true);
    const {final} = makeReadbacks();
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      snapshots: captureNetworkKFunction(final)
    });
    const result = {...record, captures};
    expect(validateGPUComputationSidecars(record, result).issues).toEqual([]);
    const serialized = JSON.stringify(
      toRecordedLensFile(result, {
        scenarioId: 'gpu-network.network-k-function',
        recordedAt: '2026-10-15T00:00:00.000Z',
        adapterLabel: 'null device: synthetic captures for budget test only'
      })
    );
    // GPU metrics and adapter metadata need space too; the actual capture must pass record-budget.
    expect(new TextEncoder().encode(serialized).byteLength).toBeLessThan(512 * 1024 - 4096);
    const context = createViewContext(result);
    const model = buildResultViewModel(context);
    expect(model.grammar).toBe('table');
    const text = getViewNodeText(renderResultView(model, context));
    for (const name of ['kValues', 'envelope-minimum', 'envelope-mean', 'envelope-maximum']) {
      expect(text).toContain(name);
    }
  } finally {
    device.destroy();
  }
});
