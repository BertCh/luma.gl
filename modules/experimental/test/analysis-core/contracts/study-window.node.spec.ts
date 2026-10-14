// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPU_STUDY_WINDOW_SUMMARY_LENGTH,
  GPU_STUDY_WINDOW_SUMMARY,
  resolveGPUStudyWindow,
  type GPUPolygonStudyWindow,
  type GPUStudyWindow
} from '../../../src/analysis-core/contracts/study-window';

const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};

describe('study window contract', () => {
  it('defines the eight-slot summary layout', () => {
    expect(GPU_STUDY_WINDOW_SUMMARY_LENGTH).toBe(8);
    expect(GPU_STUDY_WINDOW_SUMMARY).toEqual({
      minX: 0,
      minY: 1,
      maxX: 2,
      maxY: 3,
      area: 4,
      perimeter: 5,
      edgeCount: 6,
      flags: 7
    });
  });

  it('resolves omitted and explicit rectangle windows', () => {
    expect(resolveGPUStudyWindow('X', undefined, PLANAR)).toBe('rectangle');
    expect(resolveGPUStudyWindow('X', 'rectangle', PLANAR)).toBe('rectangle');
  });

  it.each([
    'polygon',
    {},
    {kind: 'polygon'},
    {
      kind: 'polygon',
      id: 'w',
      spatialContext: PLANAR
    },
    42
  ])('rejects invalid window %j', window => {
    expect(() => resolveGPUStudyWindow('X', window as GPUStudyWindow, PLANAR)).toThrowError(
      new Error("X studyWindow must be 'rectangle' or a GPUStudyWindowPreparation")
    );
  });

  it('preserves polygon handles in the same frame without requesting GPU views', () => {
    const frame = createGPULocalFrame({origin: [-87.6, 41.8]});
    const otherFrame = createGPULocalFrame({origin: [-87.7, 41.8]});
    const window: GPUPolygonStudyWindow = {
      kind: 'polygon',
      id: 'w',
      spatialContext: frame.spatialContext,
      getViews: () => {
        throw new Error('GPU views must not be requested during resolution');
      }
    };
    expect(resolveGPUStudyWindow('X', window, frame.spatialContext)).toBe(window);
    expect(() => resolveGPUStudyWindow('X', window, otherFrame.spatialContext)).toThrow(
      /^X studyWindow incompatible spatial contexts: planar frames differ/
    );
    expect(() =>
      resolveGPUStudyWindow('X', window, {
        coordinateSpace: 'longitude-latitude',
        metric: 'none'
      })
    ).toThrowError(
      new Error(
        'X studyWindow incompatible spatial contexts: coordinate spaces differ (planar and longitude-latitude)'
      )
    );
  });
});
