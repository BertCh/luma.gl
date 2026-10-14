// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-line-noding';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

// Review A §7.2: these contributors build their delegates (line noding, segment intersection,
// polygon overlay, nearest-feature join) in getCommandNodes, so a bad forwarded capacity used to
// surface late, under the delegate's id. The constructor must reject it under its own id.

type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'sint32';
const INVALID_VALUES = [0, 1.5, Number.NaN];

function createViewFactory(graph: GPUCommandGraph) {
  let serial = 0;
  return <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `capacity-${serial++}`, format, length);
}

type ViewFactory = ReturnType<typeof createViewFactory>;

function createTopologyOutput(view: ViewFactory) {
  const segments = 64;
  const rings = 8;
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  return {
    nodedSegments: {
      endpoints: view('float32x4', segments),
      sourceFeatureIds: view('uint32', segments),
      sourceRingIds: view('uint32', segments),
      sourceEdgeIds: view('uint32', segments),
      sourceStartParameters: view('float32', segments),
      sourceEndParameters: view('float32', segments),
      status: status()
    },
    directedEdges: {
      endpoints: view('float32x4', 2 * segments),
      sourceSegmentIds: view('uint32', 2 * segments),
      sourceFeatureIds: view('uint32', 2 * segments),
      twinIds: view('uint32', 2 * segments),
      ringIds: view('uint32', 2 * segments),
      flags: view('uint32', 2 * segments),
      status: status()
    },
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 2 * segments),
      featureOffsets: view('uint32', rings + 1),
      polygonOffsets: view('uint32', rings + 1),
      ringOffsets: view('uint32', rings + 1),
      sourceIds: view('uint32', rings)
    },
    diagnostics: {
      edgeClasses: view('uint32', segments),
      closedRingCount: view('uint32', 1),
      openChainCount: view('uint32', 1),
      cutEdgeCount: view('uint32', 1),
      dangleCount: view('uint32', 1),
      unusedEdgeCount: view('uint32', 1)
    },
    status: status()
  };
}

const PRECISION = {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'};

type Builder = (view: ViewFactory, overrides: Record<string, number>) => {id: string};

const BUILDERS: Record<string, {names: string[]; build: Builder}> = {
  'line-noding': {
    names: ['intersectionCapacity', 'leafCapacity'],
    build: (view, overrides) =>
      new GPULineNoding({
        id: 'line-noding',
        lines: {kind: 'lines', positions: view('float32x2', 8), lineOffsets: view('uint32', 3)},
        intersectionCapacity: 64,
        output: createTopologyOutput(view).nodedSegments,
        ...overrides
      } as never)
  },
  polygonize: {
    names: ['intersectionCapacity', 'leafCapacity'],
    build: (view, overrides) =>
      new GPUPolygonize({
        id: 'polygonize',
        lines: {kind: 'lines', positions: view('float32x2', 8), lineOffsets: view('uint32', 3)},
        intersectionCapacity: 64,
        precision: PRECISION,
        output: createTopologyOutput(view),
        ...overrides
      } as never)
  },
  'make-valid': {
    names: ['intersectionCapacity', 'leafCapacity'],
    build: (view, overrides) =>
      new GPUMakeValid({
        id: 'make-valid',
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 8),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 3)
        },
        intersectionCapacity: 64,
        precision: PRECISION,
        output: createTopologyOutput(view),
        ...overrides
      } as never)
  },
  'linear-referencing': {
    names: ['leafCapacity'],
    build: (view, overrides) =>
      new GPULinearReferencing({
        id: 'linear-referencing',
        queryPositions: view('float32x2', 8),
        positions: view('float32x2', 12),
        lineOffsets: view('uint32', 4),
        radius: view('float32', 1),
        candidateCapacity: 64,
        output: {measures: view('float32', 8), status: {overflow: view('uint32', 1)}},
        ...overrides
      } as never)
  },
  'buffer-surface': {
    names: ['leafCapacity'],
    build: (view, overrides) =>
      new GPUBufferSurface({
        id: 'buffer-surface',
        geometry: {kind: 'lines', positions: view('float32x2', 5), lineOffsets: view('uint32', 2)},
        parameters: view('float32', 4),
        vertexTolerance: 1e-5,
        capacity: {intersections: 64, nodedSegments: 128},
        output: {
          geometry: {
            kind: 'polygons',
            positions: view('float32x2', 64),
            featureOffsets: view('uint32', 9),
            polygonOffsets: view('uint32', 9),
            ringOffsets: view('uint32', 9)
          },
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        },
        ...overrides
      } as never)
  },
  'buffer-selection': {
    names: ['leafCapacity'],
    build: (view, overrides) =>
      new GPUBufferSelection({
        id: 'buffer-selection',
        points: {kind: 'points', positions: view('float32x2', 10)},
        features: {kind: 'segments', starts: view('float32x2', 3), ends: view('float32x2', 3)},
        distance: view('float32', 1),
        candidateCapacity: 16,
        output: {outputMask: view('uint32', 10)},
        ...overrides
      } as never)
  }
};

describe('forwarded capacities are validated by the composing constructor', () => {
  for (const [id, {names, build}] of Object.entries(BUILDERS)) {
    it(`${id} accepts its defaults and rejects 0, 1.5 and NaN under its own id`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `capacity-${id}`});
      const view = createViewFactory(graph);
      expect(build(view, {}).id).toBe(id);
      for (const name of names) {
        for (const value of INVALID_VALUES) {
          expect(() => build(view, {[name]: value}), `${name} = ${value}`).toThrow(
            new RegExp(`^${id} ${name} must be`)
          );
        }
      }
    });
  }
});
