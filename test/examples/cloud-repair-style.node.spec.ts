// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {prepareCloudRepairStyle} from '../../examples/deck/spatial-analysis-showcase/src/scenes/raster/cloud-repair-style';

function makePair() {
  const width = 160;
  const height = 128;
  const prior = new Uint8Array(width * height * 4);
  const target = new Uint8Array(prior.length);
  const mask = new Uint8Array(width * height);
  const gains = [1.2, 0.8, 1.1];
  const offsets = [0.03, 0.08, -0.02];
  for (let cell = 0; cell < mask.length; cell++) {
    for (let channel = 0; channel < 3; channel++) {
      const pixel = cell * 4 + channel;
      prior[pixel] = 30 + ((cell * 73 + channel * 31) % 150);
      target[pixel] = Math.round(prior[pixel] * gains[channel] + offsets[channel] * 255);
    }
    prior[cell * 4 + 3] = target[cell * 4 + 3] = 255;
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (column >= 52 && column < 108 && row >= 40 && row < 88) mask[cell] = 1;
  }
  return {width, height, prior, target, mask, gains, offsets};
}

describe('cloud repair current-day style', () => {
  test('recovers current-day colour and contrast in large holes before GPU refinement', () => {
    const {width, height, prior, target, mask, gains, offsets} = makePair();
    const result = prepareCloudRepairStyle(target, prior, mask, width, height);
    for (let channel = 0; channel < 3; channel++) {
      expect(result.style.gain[channel]).toBeCloseTo(gains[channel], 2);
      expect(result.style.offset[channel]).toBeCloseTo(offsets[channel], 2);
    }
    let matchedError = 0;
    let pasteError = 0;
    for (let cell = 0; cell < mask.length; cell++) {
      for (let channel = 0; channel < 3; channel++) {
        const reconstructed =
          result.priorValues[cell * 3 + channel] + result.state[cell * 4 + channel];
        if (mask[cell]) {
          matchedError += Math.abs(reconstructed * 255 - target[cell * 4 + channel]);
          pasteError += Math.abs(prior[cell * 4 + channel] - target[cell * 4 + channel]);
        } else if (!result.repairMask[cell]) {
          expect(Math.round(reconstructed * 255)).toBe(target[cell * 4 + channel]);
        }
      }
    }
    expect(matchedError).toBeLessThan(pasteError * 0.05);
  });

  test('masked cloud brightness cannot influence the fitted style or initialized fill', () => {
    const {width, height, prior, target, mask} = makePair();
    const original = prepareCloudRepairStyle(target, prior, mask, width, height);
    for (let cell = 0; cell < mask.length; cell++) {
      if (mask[cell]) target.fill(cell % 2 ? 255 : 0, cell * 4, cell * 4 + 3);
    }
    const contaminated = prepareCloudRepairStyle(target, prior, mask, width, height);
    expect(contaminated).toEqual(original);
  });

  test('excludes missed bright cloud fringes and dark shadows from the fill', () => {
    const {width, height, prior, target, mask} = makePair();
    for (let row = 40; row < 88; row++) {
      for (let column = 46; column < 114; column++) {
        if (column >= 52 && column < 108) continue;
        const cell = row * width + column;
        target.fill(column < 52 ? 230 : 8, cell * 4, cell * 4 + 3);
      }
    }
    const result = prepareCloudRepairStyle(target, prior, mask, width, height);
    for (const column of [48, 110]) {
      const cell = 64 * width + column;
      expect(result.repairMask[cell]).toBe(3);
      expect(result.state[cell * 4 + 3]).toBe(0);
      expect(result.priorValues[cell * 3] + result.state[cell * 4]).toBeGreaterThan(0.1);
      expect(result.priorValues[cell * 3] + result.state[cell * 4]).toBeLessThan(0.9);
    }
  });

  test('falls back to the prior when no clear observation is available', () => {
    const {width, height, prior, target, mask} = makePair();
    mask.fill(1);
    const result = prepareCloudRepairStyle(target, prior, mask, width, height);
    expect(result.style.sampleCount).toBe(0);
    expect(result.state.every(value => value === 0)).toBe(true);
    expect(result.priorValues[0]).toBeCloseTo(prior[0] / 255);
  });

  test('keeps a completely clear image observed, including narrow grids', () => {
    const prior = new Uint8Array([50, 60, 70, 255, 90, 100, 110, 255]);
    const target = new Uint8Array([60, 70, 80, 255, 100, 110, 120, 255]);
    const result = prepareCloudRepairStyle(target, prior, new Uint8Array(2), 1, 2);
    expect(Array.from(result.repairMask)).toEqual([0, 0]);
    expect(result.state[3]).toBe(1);
    expect(result.state[7]).toBe(1);
  });
});
