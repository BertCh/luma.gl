// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getSegmentPolygonizationDiagnostics,
  type PolygonizationSegment
} from '../../../src/gpu-spatial-analysis/ring-assembly';

const SQUARE: PolygonizationSegment[] = [
  [0, 0, 1, 0],
  [1, 0, 1, 1],
  [1, 1, 0, 1],
  [0, 1, 0, 0]
];

describe('getSegmentPolygonizationDiagnostics', () => {
  it('reports a closed ring and consumes all of its segments', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({segments: SQUARE});

    expect(diagnostics.closedRings).toHaveLength(1);
    expect(diagnostics.closedRings[0].segmentIndices).toEqual([0, 1, 2, 3]);
    expect(diagnostics.closedRings[0].positions[0]).toEqual(
      diagnostics.closedRings[0].positions.at(-1)
    );
    expect(diagnostics.closedRings[0].signedArea).toBe(1);
    expect(diagnostics.closedRingSegmentIndices).toEqual([0, 1, 2, 3]);
    expect(diagnostics.openChains).toEqual([]);
    expect(diagnostics.unusedSegmentIndices).toEqual([]);
  });

  it('preserves a maximal open chain', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({
      segments: [
        [0, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 3, 0]
      ]
    });

    expect(diagnostics.closedRings).toEqual([]);
    expect(diagnostics.openChains).toEqual([[0, 1, 2]]);
    expect(diagnostics.dangleSegmentIndices).toEqual([0, 1, 2]);
    expect(diagnostics.unusedSegmentIndices).toEqual([0, 1, 2]);
  });

  it('reports a bridge between two rings as a cut edge', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({
      segments: [...SQUARE, [2, 1, 3, 1], [3, 1, 3, 2], [3, 2, 2, 2], [2, 2, 2, 1], [1, 1, 2, 1]]
    });

    expect(diagnostics.closedRings).toHaveLength(2);
    expect(diagnostics.cutEdgeSegmentIndices).toEqual([8]);
    expect(diagnostics.dangleSegmentIndices).toEqual([]);
    expect(diagnostics.openChains).toEqual([[8]]);
    expect(diagnostics.unusedSegmentIndices).toEqual([8]);
    expect(diagnostics.segmentClassifications[8]).toBe('cut-edge');
  });

  it('recursively peels every segment in a dangle', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({
      segments: [...SQUARE, [1, 1, 2, 1], [2, 1, 3, 1]]
    });

    expect(diagnostics.closedRings).toHaveLength(1);
    expect(diagnostics.dangleSegmentIndices).toEqual([4, 5]);
    expect(diagnostics.cutEdgeSegmentIndices).toEqual([]);
    expect(diagnostics.openChains).toEqual([[4, 5]]);
    expect(diagnostics.unusedSegmentIndices).toEqual([4, 5]);
  });

  it('classifies all unused segments, including degenerate input', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({
      segments: [...SQUARE, [3, 3, 3, 3]]
    });

    expect(diagnostics.closedRingSegmentIndices).toEqual([0, 1, 2, 3]);
    expect(diagnostics.unusedSegmentIndices).toEqual([4]);
    expect(diagnostics.segmentClassifications).toEqual([
      'closed-ring',
      'closed-ring',
      'closed-ring',
      'closed-ring',
      'unused'
    ]);
  });

  it('respects group boundaries and vertex tolerance', () => {
    const diagnostics = getSegmentPolygonizationDiagnostics({
      segments: [
        [0, 0, 1, 0],
        [1.00001, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 0, 0]
      ],
      groups: [7, 7, 7, 7],
      vertexTolerance: 0.001
    });

    expect(diagnostics.closedRings).toHaveLength(1);
    expect(diagnostics.unusedSegmentIndices).toEqual([]);
  });
});
