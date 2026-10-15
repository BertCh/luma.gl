// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/test-utils';
import {describe, expect, test, vi} from 'vitest';
import {selectGPUStrategy, type GPUStrategyCandidate} from '../../src/gpu-core/gpu-strategy';
import {getAdaptiveGPUReductionStrategy} from '../../src/gpu-core/gpu-reduction-strategy';

describe('GPU strategy decision recording', () => {
  test.each([
    undefined,
    'small'
  ] as const)('preserves selection with preferred id %s', preferredId => {
    const device = new NullDevice({id: 'strategy-device'});
    const candidates: GPUStrategyCandidate<string, number, {size: number}>[] = [
      {
        id: 'small',
        score: () => 10,
        reason: () => 'small workload',
        createDetails: () => ({size: 64})
      },
      {
        id: 'large',
        score: () => 20,
        reason: () => 'large workload',
        createDetails: () => ({size: 256})
      },
      {id: 'tied', score: () => 20, reason: () => 'same score', createDetails: () => ({size: 128})},
      {
        id: 'unsupported',
        isSupported: () => false,
        score: vi.fn(() => 100),
        reason: vi.fn(() => 'unused'),
        unsupportedReason: () => 'missing subgroup support',
        failingFeature: () => 'subgroups',
        createDetails: vi.fn(() => ({size: 32}))
      },
      {
        id: 'fallback',
        isSupported: () => false,
        score: () => 0,
        reason: () => 'unused',
        createDetails: () => ({size: 1})
      }
    ];
    try {
      const props = {device, workload: 1, preferredId, candidates};
      const plain = selectGPUStrategy(props);
      const onDecision = vi.fn();
      expect(selectGPUStrategy({...props, onDecision, subject: 'reduction'})).toEqual(plain);
      expect(onDecision).toHaveBeenCalledExactlyOnceWith({
        id: 'strategy:reduction',
        subject: 'reduction',
        kind: 'strategy',
        chosen: preferredId ?? 'large',
        reason: preferredId ? 'small workload' : 'large workload',
        method: preferredId ? 'preferred' : 'score',
        candidates: [
          {id: 'small', supported: true, score: 10, reason: 'small workload'},
          {id: 'large', supported: true, score: 20, reason: 'large workload'},
          {id: 'tied', supported: true, score: 20, reason: 'same score'},
          {
            id: 'unsupported',
            supported: false,
            reason: 'missing subgroup support',
            failingFeature: 'subgroups'
          },
          {id: 'fallback', supported: false, reason: 'unsupported'}
        ]
      });
      expect(candidates[3].score).not.toHaveBeenCalled();
      expect(candidates[3].reason).not.toHaveBeenCalled();
      expect(candidates[3].createDetails).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  });

  test('does no extra candidate evaluation without recording', () => {
    const device = new NullDevice({id: 'strategy-off-device'});
    const rejectedReason = vi.fn(() => 'not selected');
    const unsupportedReason = vi.fn(() => 'not supported');
    const onDecision = vi.fn();
    try {
      const candidates = [
        {id: 'chosen', score: () => 2, reason: () => 'selected', createDetails: () => ({})},
        {id: 'rejected', score: () => 1, reason: rejectedReason, createDetails: () => ({})},
        {
          id: 'unsupported',
          isSupported: () => false,
          score: () => 3,
          reason: rejectedReason,
          unsupportedReason,
          createDetails: () => ({})
        }
      ];
      selectGPUStrategy({device, workload: 0, candidates});
      expect(rejectedReason).not.toHaveBeenCalled();
      expect(unsupportedReason).not.toHaveBeenCalled();
      selectGPUStrategy({device, workload: 0, candidates, onDecision});
      expect(rejectedReason).toHaveBeenCalledTimes(1);
      expect(unsupportedReason).toHaveBeenCalledTimes(1);
      expect(onDecision.mock.calls[0][0]).toMatchObject({
        id: 'strategy:gpu-strategy',
        subject: 'gpu-strategy'
      });
      for (const recording of [false, true]) {
        expect(() =>
          selectGPUStrategy({
            device,
            workload: 0,
            candidates,
            preferredId: 'unsupported',
            onDecision: recording ? onDecision : undefined
          })
        ).toThrow(/not supported/);
      }
    } finally {
      device.destroy();
    }
  });

  test.each([
    [64, 'wg64'],
    [128, 'wg128'],
    [256, 'wg256']
  ] as const)('keeps reduction identity and reason at limit %s', (limit, strategyId) => {
    const device = new NullDevice({id: 'reduction-strategy-device'});
    Object.defineProperty(device.limits, 'maxComputeInvocationsPerWorkgroup', {value: limit});
    try {
      const onDecision = vi.fn();
      const strategy = getAdaptiveGPUReductionStrategy(device, 65536, {onDecision, subject: 'sum'});
      expect(strategy).toEqual(getAdaptiveGPUReductionStrategy(device, 65536));
      expect(strategy.strategyId).toBe(strategyId);
      expect(strategy.reason).toBe(onDecision.mock.calls[0][0].reason);
      expect(onDecision.mock.calls[0][0]).toMatchObject({subject: 'sum', chosen: strategyId});
      if (limit === 64)
        expect(onDecision.mock.calls[0][0].candidates.slice(1)).toEqual([
          {
            id: 'wg128',
            supported: false,
            reason: 'maxComputeInvocationsPerWorkgroup < 128',
            failingFeature: 'maxComputeInvocationsPerWorkgroup'
          },
          {
            id: 'wg256',
            supported: false,
            reason: 'maxComputeInvocationsPerWorkgroup < 256',
            failingFeature: 'maxComputeInvocationsPerWorkgroup'
          }
        ]);
    } finally {
      device.destroy();
    }
  });
});
