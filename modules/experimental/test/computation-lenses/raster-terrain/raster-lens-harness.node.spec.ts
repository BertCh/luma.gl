// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {describe, expect, it} from 'vitest';
import {selectKeyFrames} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-harness';

function frame(afterNode: string, value: number): GPUComputationCapture {
  return {
    id: `costs-${afterNode}`,
    name: 'costs',
    afterNode,
    values: [value],
    elementType: 'f32',
    provenance: 'gpu-readback'
  };
}

describe('raster lens key frames', () => {
  const rounds = new Map([
    ['reset', 0],
    ['relax-0', 1],
    ['relax-1', 2],
    ['relax-gate-0', 2],
    ['relax-2', 3],
    ['relax-3', 4],
    ['finalize', 4]
  ]);

  it('keeps the last frame per completed round and labels the final observation', () => {
    const frames = [
      'reset',
      'relax-0',
      'relax-1',
      'relax-gate-0',
      'relax-2',
      'relax-3',
      'finalize'
    ].map((node, index) => frame(node, index));
    const selected = selectKeyFrames(frames, rounds, 8);
    expect(selected.map(({iteration}) => iteration)).toEqual([0, 1, 2, 3, 4]);
    // The gate frame replaces the round-2 frame; the post-loop frame is the final state.
    expect(selected.map(({frame: kept}) => kept.afterNode)).toEqual([
      'reset',
      'relax-0',
      'relax-gate-0',
      'relax-2',
      'finalize'
    ]);
  });

  it('thins log-spaced to the maximum and always keeps the first and final rounds', () => {
    const many = new Map(Array.from({length: 20}, (_, index) => [`relax-${index}`, index + 1]));
    const frames = [...many.keys()].map((node, index) => frame(node, index));
    const selected = selectKeyFrames(frames, many, 6);
    expect(selected).toHaveLength(6);
    expect(selected[0].iteration).toBe(1);
    expect(selected.at(-1)?.iteration).toBe(20);
    expect(selected.at(-1)?.frame).toBe(frames.at(-1));
    // Log spacing keeps more early rounds than late ones.
    const iterations = selected.map(({iteration}) => iteration);
    expect(iterations.filter(iteration => iteration <= 10).length).toBeGreaterThan(3);
  });

  it('drops rounds past the executed iteration count', () => {
    const many = new Map(Array.from({length: 20}, (_, index) => [`relax-${index}`, index + 1]));
    const frames = [...many.keys()].map((node, index) => frame(node, index));
    const selected = selectKeyFrames(frames, many, 8, 5);
    expect(selected.map(({iteration}) => iteration)).toEqual([1, 2, 3, 4, 5]);
    // The last kept round shows the final observation (gated rounds after it never ran).
    expect(selected.at(-1)?.frame).toBe(frames.at(-1));
  });
});
