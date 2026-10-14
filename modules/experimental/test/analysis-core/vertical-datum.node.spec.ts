// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GPU_ELLIPSOIDAL_VERTICAL_DATUM,
  assertCompatibleGPUVerticalDatums
} from '../../src/analysis-core/contracts/vertical-datum';

describe('vertical datum contract', () => {
  it('reserves the ellipsoidal datum key', () => {
    expect(GPU_ELLIPSOIDAL_VERTICAL_DATUM).toBe('ellipsoidal');
  });

  it('reports mismatched observer and elevation datums', () => {
    expect(() => assertCompatibleGPUVerticalDatums('X', 'EPSG:5703', 'EPSG:5773')).toThrowError(
      new Error(
        'X observer heights are EPSG:5703 but elevation is EPSG:5773; convert to one vertical datum first'
      )
    );
  });

  it('names custom height inputs in the diagnostic', () => {
    expect(() =>
      assertCompatibleGPUVerticalDatums('X', 'EPSG:5703', 'EPSG:5773', 'candidate')
    ).toThrowError(
      new Error(
        'X candidate heights are EPSG:5703 but elevation is EPSG:5773; convert to one vertical datum first'
      )
    );
  });

  it('accepts either undeclared datum', () => {
    expect(() => assertCompatibleGPUVerticalDatums('X', undefined, 'EPSG:5773')).not.toThrow();
    expect(() => assertCompatibleGPUVerticalDatums('X', 'EPSG:5703', undefined)).not.toThrow();
    expect(() => assertCompatibleGPUVerticalDatums('X', undefined, undefined)).not.toThrow();
  });

  it('ignores case and surrounding whitespace', () => {
    expect(() => assertCompatibleGPUVerticalDatums('X', ' epsg:5773', 'EPSG:5773')).not.toThrow();
    expect(() => assertCompatibleGPUVerticalDatums('X', 'EPSG:5773', ' epsg:5773 ')).not.toThrow();
  });

  it('does not alias geoid names to EPSG codes', () => {
    expect(() => assertCompatibleGPUVerticalDatums('X', 'EGM96', 'EPSG:5773')).toThrowError(
      new Error(
        'X observer heights are EGM96 but elevation is EPSG:5773; convert to one vertical datum first'
      )
    );
  });
});
